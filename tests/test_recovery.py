# no-opencc
"""Seat recovery: Codex quota RPC, half-open probes, ledger lock (issue #72-D)."""
import asyncio
import contextlib
import fcntl
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import dispatch
from providers import health
from providers.codex_quota import parse_rate_limits, read_rate_limits


CANARY = "CANARY-PII-9f3"
# A reset one year past the moment the module loads: a literal epoch went stale
# on 2026-09-15 and turned five cooldown assertions red on an untouched tree.
FUTURE_RESET = int(time.time()) + 365 * 24 * 3600

FAKE_CODEX = r'''#!/usr/bin/env python3
import fcntl
import json
import os
import sys
import time

def _bump():
    path = os.environ.get("FAKE_CODEX_COUNT")
    if not path:
        return
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().decode("utf-8").strip() or "0"
        n = int(raw) + 1
        handle.seek(0)
        handle.truncate()
        handle.write(str(n).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())

def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()

pid_path = os.environ.get("FAKE_CODEX_PID")
if pid_path:
    with open(pid_path, "w") as handle:
        handle.write(str(os.getpid()))
        handle.flush()

if len(sys.argv) < 2 or sys.argv[1] != "app-server":
    sys.exit(0)

_bump()
delay = float(os.environ.get("FAKE_CODEX_DELAY", "0") or "0")
if delay:
    time.sleep(delay)
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
if mode == "hang":
    time.sleep(3600)
    sys.exit(0)

reply_path = os.environ.get("FAKE_CODEX_REPLY")
custom = None
if reply_path and os.path.isfile(reply_path):
    with open(reply_path, encoding="utf-8") as handle:
        custom = json.load(handle)

used = float(os.environ.get("FAKE_CODEX_USED", "100"))
resets_raw = os.environ.get("FAKE_CODEX_RESETS", "1789447060")
try:
    resets_at = int(resets_raw)
except ValueError:
    resets_at = resets_raw
reached = os.environ.get("FAKE_CODEX_REACHED", "")
if reached == "":
    reached_type = "rate_limit_reached" if used >= 95 else None
elif reached in ("0", "false", "no"):
    reached_type = None
else:
    reached_type = reached

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        if mode == "init_error":
            _send({"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32000,
                             "message": os.environ.get("FAKE_CODEX_ERRMSG", "boom")}})
            continue
        _send({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {}}})
        continue
    if method == "initialized":
        continue
    if method != "account/rateLimits/read":
        continue
    if mode == "huge":
        sys.stdout.write(("CANARY-PII-9f3" * 80) * 400)
        sys.stdout.flush()
        time.sleep(3600)
        break
    if mode == "rpc_error":
        _send({"jsonrpc": "2.0", "id": mid,
               "error": {"code": -32001,
                         "message": os.environ.get("FAKE_CODEX_ERRMSG", "boom")}})
        continue
    if custom is not None:
        result = custom
    else:
        result = {
            "rateLimits": {
                "primary": {
                    "usedPercent": used,
                    "windowDurationMins": 10080,
                    "resetsAt": resets_at,
                },
                "rateLimitReachedType": reached_type,
                "planType": "plus",
            }
        }
    if mode == "canary":
        result["accountId"] = "CANARY-PII-9f3"
        result["rateLimitUpsell"] = {"url": "CANARY-PII-9f3"}
        result["rateLimitResetCredits"] = {"n": "CANARY-PII-9f3"}
    _send({"jsonrpc": "2.0", "id": mid, "result": result})
'''


def _run(coro):
    return asyncio.run(coro)


def _envelope(ok, reason=None, model="m"):
    res = {"ok": ok, "text": "hi" if ok else "", "model": model}
    if reason:
        res["reason"] = reason
    return res


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spawn_count(path):
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip() or "0"
        return int(raw)
    except (OSError, ValueError):
        return 0


class RecoveryCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp(prefix="omni-recovery-")
        self.addCleanup(shutil.rmtree, self.state, ignore_errors=True)
        self._saved_env = {}
        for key in ("OMNI_LLM_STATE_DIR", "OMNI_LLM_CODEX_SESSIONS_DIR",
                    "OMNI_LLM_CHILD_ENV_PASS",
                    "PATH", "FAKE_CODEX_COUNT", "FAKE_CODEX_MODE",
                    "FAKE_CODEX_USED", "FAKE_CODEX_RESETS", "FAKE_CODEX_DELAY",
                    "FAKE_CODEX_REPLY", "FAKE_CODEX_PID", "FAKE_CODEX_REACHED",
                    "FAKE_CODEX_ERRMSG"):
            if key in os.environ:
                self._saved_env[key] = os.environ[key]
            os.environ.pop(key, None)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state
        self.bindir = tempfile.mkdtemp(prefix="omni-fake-codex-")
        self.addCleanup(shutil.rmtree, self.bindir, ignore_errors=True)
        self.fake_bin = os.path.join(self.bindir, "codex")
        with open(self.fake_bin, "w", encoding="utf-8") as handle:
            handle.write(FAKE_CODEX)
        os.chmod(self.fake_bin, 0o755)
        self.count_path = os.path.join(self.state, "spawn.count")
        os.environ["PATH"] = self.bindir + os.pathsep + self._saved_env.get(
            "PATH", os.defpath)
        os.environ["FAKE_CODEX_COUNT"] = self.count_path
        # The quota child gets an allowlisted environment (#138): the fake
        # vendor's control variables are passed by name.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = ",".join((
            "FAKE_CODEX_COUNT", "FAKE_CODEX_MODE", "FAKE_CODEX_USED",
            "FAKE_CODEX_RESETS", "FAKE_CODEX_DELAY", "FAKE_CODEX_REPLY",
            "FAKE_CODEX_PID", "FAKE_CODEX_REACHED", "FAKE_CODEX_ERRMSG"))
        os.environ["FAKE_CODEX_MODE"] = "ok"
        os.environ["FAKE_CODEX_USED"] = "100"
        os.environ["FAKE_CODEX_RESETS"] = str(FUTURE_RESET)
        health.clear_all()

    def tearDown(self):
        try:
            health.clear_all()
        except Exception:
            pass
        for key in ("OMNI_LLM_STATE_DIR", "OMNI_LLM_CODEX_SESSIONS_DIR",
                    "OMNI_LLM_CHILD_ENV_PASS",
                    "PATH", "FAKE_CODEX_COUNT", "FAKE_CODEX_MODE",
                    "FAKE_CODEX_USED", "FAKE_CODEX_RESETS", "FAKE_CODEX_DELAY",
                    "FAKE_CODEX_REPLY", "FAKE_CODEX_PID", "FAKE_CODEX_REACHED",
                    "FAKE_CODEX_ERRMSG"):
            os.environ.pop(key, None)
        os.environ.update(self._saved_env)

    def _events(self):
        captured = []

        def grab(event, **fields):
            captured.append((event, fields))

        return captured, grab

    def _dispatch(self, chain, router, **kwargs):
        with mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
                mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router) as routed:
            res = _run(dispatch.dispatch_llm_request(
                "p", chain, 30, None, "default", **kwargs))
        models = [call.args[1] for call in routed.call_args_list]
        return res, models

    def _write_probe_after(self, model, profile, probe_after, source=None):
        path = os.path.join(self.state, "health.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        key = health.pool_key(model, profile)
        data[key]["probe_after"] = probe_after
        if source is not None:
            data[key]["source"] = source
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.write("\n")

    def _canary_in(self, *blobs):
        for blob in blobs:
            if blob and CANARY in str(blob):
                return True
        return False

    def _child_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["OMNI_LLM_STATE_DIR"] = self.state
        return env

    def _spawn_py(self, code):
        return subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=_REPO_ROOT,
            env=self._child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class LedgerIsolationTest(RecoveryCase):
    def test_default_ledger_is_never_selected(self):
        seen = []
        real = health._ledger_path

        def tracing():
            path = real()
            seen.append(os.path.abspath(path))
            return path

        with mock.patch.object(health, "_ledger_path", side_effect=tracing):
            health.flag("codex", "default", source="quota")
            health.clear_all()
        self.assertTrue(seen)
        state = os.path.abspath(self.state)
        for path in seen:
            self.assertEqual(os.path.dirname(path), state)


class ParseRateLimitsTest(RecoveryCase):
    def test_acceptance_shape_drops_account_id(self):
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 100,
                    "windowDurationMins": 10080,
                    "resetsAt": FUTURE_RESET,
                },
                "rateLimitReachedType": "rate_limit_reached",
                "planType": "plus",
            },
            "accountId": "X",
        })
        self.assertEqual(info, {
            "used_percent": 100.0,
            "resets_at": FUTURE_RESET,
            "window_mins": 10080,
            "reached": True,
            "primary_reached": True,
            "plan": "plus",
        })
        self.assertNotIn("accountId", info)

    def test_w_iso_resets_at(self):
        iso = "2027-06-01T00:00:00Z"
        expected = int(datetime(2027, 6, 1, tzinfo=timezone.utc).timestamp())
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 60,
                    "resetsAt": iso,
                },
                "planType": "plus",
            }
        })
        self.assertEqual(info["resets_at"], expected)
        self.assertEqual(info["used_percent"], 10.0)
        self.assertFalse(info["reached"])

    def test_primary_null_and_type_errors(self):
        info = parse_rate_limits({
            "rateLimits": {
                "primary": None,
                "rateLimitReachedType": "rate_limit_reached",
                "planType": 1,
            }
        })
        self.assertIsNone(info["used_percent"])
        self.assertTrue(info["reached"])
        self.assertIsNone(info["plan"])
        info2 = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": "nope",
                    "windowDurationMins": "x",
                    "resetsAt": {"n": 1},
                }
            }
        })
        self.assertIsNone(info2["used_percent"])
        self.assertIsNone(info2["window_mins"])
        self.assertIsNone(info2["resets_at"])
        self.assertFalse(info2["reached"])


class QuotaProbeTest(RecoveryCase):
    def test_a_reached_flags_until_and_quota_source(self):
        info = {
            "used_percent": 100.0,
            "resets_at": FUTURE_RESET,
            "window_mins": 10080,
            "reached": True,
            "plan": "plus",
        }
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)):
            flagged = health.codex_passive_probe("codex/gpt-5.6-sol", "default")
        self.assertTrue(flagged)
        key = health.pool_key("codex", "default")
        entry = health.snapshot()[key]
        self.assertEqual(entry["until"], FUTURE_RESET)
        self.assertEqual(entry["source"], "quota")
        self.assertEqual(entry["model"], "codex")

    def test_b_healthy_probe_clears_quota_only_and_logs(self):
        health.flag("codex", "default", source="quota")
        health.flag("codex/gpt-5.6-sol", "default", source="rate_limit")
        info = {
            "used_percent": 10.0,
            "resets_at": FUTURE_RESET,
            "window_mins": 60,
            "reached": False,
            "plan": "plus",
        }
        events, grab = self._events()
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)), \
                mock.patch.object(health, "log_event", side_effect=grab):
            health.codex_passive_probe("codex/gpt-5.6-sol", "default")
        quota_key = health.pool_key("codex", "default")
        model_key = health.pool_key("codex/gpt-5.6-sol", "default")
        self.assertNotIn(quota_key, health.snapshot())
        self.assertIn(model_key, health.snapshot())
        self.assertEqual(health.snapshot()[model_key]["source"], "rate_limit")
        recovered = [e for e in events if e[0] == "seat_recovered"]
        self.assertTrue(recovered)
        self.assertEqual(recovered[0][1].get("source"), "quota")

    def test_c_manual_flag_kept(self):
        health.flag("codex/gpt-5.6-sol", "default", source="manual")
        info = {
            "used_percent": 10.0,
            "resets_at": FUTURE_RESET,
            "window_mins": 60,
            "reached": False,
            "plan": "plus",
        }
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)):
            health.codex_passive_probe("codex/gpt-5.6-sol", "default")
        self.assertTrue(health.active("codex/gpt-5.6-sol", "default"))
        key = health.pool_key("codex/gpt-5.6-sol", "default")
        self.assertEqual(health.snapshot()[key]["source"], "manual")

    def test_d_probe_error_leaves_ledger(self):
        health.flag("codex/gpt-5.6-sol", "default", source="rate_limit")
        before = health.snapshot()
        err = {"ok": False, "reason": "timeout", "detail": "quota_rpc_deadline"}
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(None, err)):
            result = health.codex_passive_probe("codex/gpt-5.6-sol", "default")
        self.assertFalse(result)
        self.assertEqual(health.snapshot(), before)

    def test_e_two_probes_within_ttl_spawn_once(self):
        first, err1 = read_rate_limits("default")
        self.assertIsNone(err1)
        self.assertEqual(first["used_percent"], 100.0)
        second, err2 = read_rate_limits("default")
        self.assertIsNone(err2)
        self.assertEqual(_spawn_count(self.count_path), 1)

    def test_errors_are_not_cached(self):
        os.environ["FAKE_CODEX_MODE"] = "rpc_error"
        info, err = read_rate_limits("default", timeout_s=5)
        self.assertIsNone(info)
        self.assertEqual(err["reason"], "runtime_error")
        os.environ["FAKE_CODEX_MODE"] = "ok"
        info2, err2 = read_rate_limits("default", timeout_s=5)
        self.assertIsNone(err2)
        self.assertIsNotNone(info2)
        self.assertGreaterEqual(_spawn_count(self.count_path), 2)

    def test_hang_times_out_and_kills_child(self):
        os.environ["FAKE_CODEX_MODE"] = "hang"
        pid_path = os.path.join(self.state, "child.pid")
        os.environ["FAKE_CODEX_PID"] = pid_path
        start = time.time()
        info, err = read_rate_limits("default", timeout_s=1)
        elapsed = time.time() - start
        self.assertIsNone(info)
        self.assertEqual(err["reason"], "timeout")
        self.assertLess(elapsed, 4)
        self.assertTrue(os.path.isfile(pid_path))
        pid = int(open(pid_path, encoding="utf-8").read().strip())
        dead = False
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except OSError:
                dead = True
                break
            time.sleep(0.05)
        self.assertTrue(dead, "quota child still running after timeout")


class ActivePathTest(RecoveryCase):
    def test_f_codex_rate_limit_flags_until_resets_at(self):
        info = {
            "used_percent": 100.0,
            "resets_at": FUTURE_RESET,
            "window_mins": 10080,
            "reached": True,
            "plan": "plus",
            "checked_at": time.time(),
        }

        async def router(prompt, model, *a, **k):
            return _envelope(False, "rate_limit", model=model)

        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)) as probed, \
                mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router):
            res = _run(dispatch.dispatch_single(
                "p", "codex/gpt-5.6-sol", 30, None, "default"))
        self.assertEqual(res["reason"], "rate_limit")
        self.assertTrue(probed.called)
        self.assertTrue(probed.call_args.kwargs.get("force"))
        key = health.pool_key("codex/gpt-5.6-sol", "default")
        entry = health.snapshot()[key]
        self.assertEqual(entry["until"], FUTURE_RESET)
        self.assertEqual(entry["source"], "quota")

    def test_n_stale_healthy_cache_then_rate_limit_forces_refresh(self):
        os.environ["FAKE_CODEX_USED"] = "10"
        os.environ["FAKE_CODEX_REACHED"] = "0"
        info, err = read_rate_limits("default")
        self.assertIsNone(err)
        self.assertEqual(info["used_percent"], 10.0)
        self.assertEqual(_spawn_count(self.count_path), 1)
        os.environ["FAKE_CODEX_USED"] = "100"
        os.environ["FAKE_CODEX_REACHED"] = "rate_limit_reached"
        os.environ["FAKE_CODEX_RESETS"] = str(FUTURE_RESET)

        async def router(prompt, model, *a, **k):
            return _envelope(False, "rate_limit", model=model)

        with mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router):
            _run(dispatch.dispatch_single(
                "p", "codex/gpt-5.6-sol", 30, None, "default"))
        self.assertEqual(_spawn_count(self.count_path), 2)
        key = health.pool_key("codex/gpt-5.6-sol", "default")
        self.assertEqual(health.snapshot()[key]["until"], FUTURE_RESET)
        self.assertEqual(health.snapshot()[key]["source"], "quota")


class HalfOpenTest(RecoveryCase):
    def test_g_probing_grok_dispatched_first_and_cleared(self):
        until = time.time() + 3600
        health.flag("grok-4.6", "default", until=until, source="rate_limit")
        self._write_probe_after("grok-4.6", "default", time.time() - 1)
        events, grab = self._events()

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        with mock.patch.object(dispatch, "log_event", side_effect=grab):
            res, models = self._dispatch("grok-4.6,claude/x", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["grok-4.6"])
        self.assertFalse(health.active("grok-4.6", "default"))
        self.assertTrue(any(e[0] == "seat_probe" for e in events))
        self.assertTrue(any(
            e[0] == "seat_recovered" and e[1].get("source") == "probe"
            for e in events))

    def test_h_probing_rate_limit_reprobe_continues(self):
        until = time.time() + 3600
        health.flag("grok-4.6", "default", until=until, source="rate_limit")
        self._write_probe_after("grok-4.6", "default", time.time() - 1)
        before = health.snapshot()[health.pool_key("grok-4.6", "default")]

        async def router(prompt, model, *a, **k):
            if model == "grok-4.6":
                return _envelope(False, "rate_limit", model=model)
            return _envelope(True, model=model)

        res, models = self._dispatch("grok-4.6,claude/x", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["grok-4.6", "claude/x"])
        self.assertEqual(res["tried"][0]["reason"], "rate_limit")
        after = health.snapshot()[health.pool_key("grok-4.6", "default")]
        self.assertEqual(after["until"], before["until"])
        self.assertGreater(after["probe_after"], before["probe_after"])

    def test_i_only_one_probing_seat_per_request(self):
        until = time.time() + 3600
        health.flag("grok-4.6", "default", until=until, source="rate_limit")
        health.flag("claude/x", "default", until=until, source="rate_limit")
        now = time.time()
        self._write_probe_after("grok-4.6", "default", now - 1)
        self._write_probe_after("claude/x", "default", now - 1)

        async def router(prompt, model, *a, **k):
            if model == "grok-4.6":
                return _envelope(False, "backend_error", model=model)
            return _envelope(True, model=model)

        res, models = self._dispatch("grok-4.6,claude/x,backup", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["grok-4.6", "backup"])
        skipped = [t for t in res.get("tried", [])
                   if t.get("model") == "claude/x"]
        self.assertTrue(skipped)
        self.assertEqual(skipped[0]["reason"], "skipped_cooldown")
        self.assertTrue(health.active("claude/x", "default"))

    def test_m_manual_flag_never_probed(self):
        until = time.time() + 3600
        health.flag("grok-4.6", "default", until=until, source="manual")
        self._write_probe_after("grok-4.6", "default", time.time() - 1)
        self.assertEqual(health.seat_kind("grok-4.6", "default"), "cooling")

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        res, models = self._dispatch("grok-4.6,claude/x", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["claude/x"])
        self.assertEqual(res["tried"][0]["reason"], "skipped_cooldown")
        self.assertTrue(health.active("grok-4.6", "default"))
        self.assertEqual(
            health.snapshot()[health.pool_key("grok-4.6", "default")]["source"],
            "manual")

    def test_s_legacy_entry_never_probing(self):
        until = time.time() + 3600
        key = health.pool_key("grok-4.6", "default")
        payload = {
            key: {
                "model": "grok-4.6",
                "profile": "default",
                "until": until,
                "source": "rate_limit",
                "flagged_at": time.time(),
            }
        }
        with open(os.path.join(self.state, "health.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        self.assertEqual(health.seat_kind("grok-4.6", "default"), "cooling")
        self.assertNotEqual(health.seat_kind("grok-4.6", "default"), "probing")

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        res, models = self._dispatch("grok-4.6,claude/x", router)
        self.assertEqual(models, ["claude/x"])
        self.assertEqual(res["tried"][0]["reason"], "skipped_cooldown")

    def test_v_quota_source_never_probing(self):
        health.flag("codex/gpt-5.6-sol", "default", until=FUTURE_RESET,
                    source="quota")
        self._write_probe_after("codex/gpt-5.6-sol", "default",
                                time.time() - 1, source="quota")
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-sol", "default"), "cooling")

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        res, models = self._dispatch("codex/gpt-5.6-sol,backup", router)
        self.assertEqual(models, ["backup"])
        self.assertEqual(res["tried"][0]["reason"], "skipped_cooldown")

    def test_crash_source_is_half_open(self):
        until = time.time() + 3600
        health.flag("grok-4.6", "default", until=until, source="crash")
        self._write_probe_after("grok-4.6", "default", time.time() - 1,
                                source="crash")
        self.assertEqual(health.seat_kind("grok-4.6", "default"), "probing")

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        res, models = self._dispatch("grok-4.6,claude/x", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["grok-4.6"])
        self.assertFalse(health.active("grok-4.6", "default"))


class ExpiryLockCliTest(RecoveryCase):
    def test_j_and_o_expired_logged_once_on_maintain(self):
        health.flag("m-exp", "default", until=time.time() - 5, source="rate_limit")
        events, grab = self._events()
        with mock.patch.object(health, "log_event", side_effect=grab):
            for _ in range(10):
                self.assertFalse(health.active("m-exp", "default"))
            recovered = [e for e in events if e[0] == "seat_recovered"]
            self.assertEqual(recovered, [])
            health.maintain()
            recovered = [e for e in events if e[0] == "seat_recovered"]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0][1].get("source"), "expired")
        health.maintain()
        with mock.patch.object(health, "log_event", side_effect=grab):
            health.maintain()
        expired = [e for e in events if e[0] == "seat_recovered"
                   and e[1].get("source") == "expired"]
        self.assertEqual(len(expired), 1)

    def test_z_active_snapshot_side_effect_free(self):
        health.flag("m-exp", "default", until=time.time() - 5, source="rate_limit")
        path = os.path.join(self.state, "health.json")
        before = open(path, "rb").read()
        mtime = os.path.getmtime(path)
        self.assertFalse(health.active("m-exp", "default"))
        snap = health.snapshot()
        self.assertNotIn(health.pool_key("m-exp", "default"), snap)
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.path.getmtime(path), mtime)

    def test_p_malformed_ledger_and_cache_recover(self):
        with open(os.path.join(self.state, "health.json"), "w") as handle:
            handle.write("{not json")
        with open(os.path.join(self.state, "quota.json"), "w") as handle:
            handle.write("[")

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        res, models = self._dispatch("A,B", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["A"])
        health.flag("m3", "default", source="rate_limit")
        path = os.path.join(self.state, "health.json")
        data = json.load(open(path, encoding="utf-8"))
        self.assertIn(health.pool_key("m3", "default"), data)
        health.write_quota_cache({"default": {"used_percent": 1,
                                              "checked_at": time.time()}})
        cached = health.read_quota_cache()
        self.assertIn("default", cached)

    def test_q_lock_held_skips_mutation(self):
        health.flag("keep", "default", source="rate_limit")
        lock_path = os.path.join(self.state, "health.lock")
        events, grab = self._events()
        with open(lock_path, "a+b") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
            with mock.patch.object(health, "HEALTH_LOCK_WAIT_S", 0.15), \
                    mock.patch.object(health, "log_event", side_effect=grab):
                result = health.flag("other", "default", source="rate_limit")
        self.assertIsNone(result)
        self.assertTrue(any(e[0] == "health_skipped" for e in events))
        snap = health.snapshot()
        self.assertIn(health.pool_key("keep", "default"), snap)
        self.assertNotIn(health.pool_key("other", "default"), snap)

    def test_k_twenty_distinct_writers_keep_union(self):
        models = ["w%02d" % i for i in range(20)]
        procs = []
        for model in models:
            code = (
                "from providers import health; "
                "health.flag(%r, 'default', source='rate_limit')" % model
            )
            procs.append(self._spawn_py(code))
        for proc in procs:
            self.assertEqual(proc.wait(), 0, proc.stderr.read())
        snap = health.snapshot()
        for model in models:
            self.assertIn(health.pool_key(model, "default"), snap)

    def test_u_same_seat_flag_clear_valid_file(self):
        flag_code = (
            "from providers import health; "
            "health.flag('same', 'default', source='rate_limit')"
        )
        clear_code = (
            "from providers import health; "
            "health.clear('same', 'default')"
        )
        procs = [self._spawn_py(flag_code), self._spawn_py(clear_code)]
        for proc in procs:
            proc.wait()
        path = os.path.join(self.state, "health.json")
        data = json.load(open(path, encoding="utf-8"))
        self.assertIsInstance(data, dict)
        key = health.pool_key("same", "default")
        if key in data:
            self.assertIsInstance(data[key], dict)
            self.assertIn("until", data[key])

    def test_t_two_processes_spawn_once(self):
        os.environ["FAKE_CODEX_DELAY"] = "0.4"
        code = (
            "from providers.codex_quota import read_rate_limits as r; "
            "info, err = r('default', timeout_s=10); "
            "assert err is None, err"
        )
        procs = [self._spawn_py(code), self._spawn_py(code)]
        for proc in procs:
            rc = proc.wait()
            self.assertEqual(rc, 0, proc.stderr.read())
        self.assertEqual(_spawn_count(self.count_path), 1)

    def test_l_health_probe_cli_has_used_percent_no_account_id(self):
        import omni_cli
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            rc = omni_cli.main(["health", "--probe"])
        self.assertEqual(rc, 0)
        text = stdout.getvalue()
        self.assertIn("used_percent", text)
        self.assertNotIn("accountId", text)
        payload = json.loads(text)
        self.assertIn("quota", payload)
        self.assertIn("used_percent", payload["quota"]["codex"]["default"])

    def test_y_probe_cli_writes_flag_and_clear(self):
        import omni_cli
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = omni_cli.main(["health", "--probe"])
        self.assertEqual(rc, 0)
        key = health.pool_key("codex", "default")
        self.assertIn(key, health.snapshot())
        self.assertEqual(health.snapshot()[key]["source"], "quota")

        os.environ["FAKE_CODEX_USED"] = "10"
        os.environ["FAKE_CODEX_REACHED"] = "0"
        # Force a new spawn so the healthy reply is not hidden by TTL.
        with mock.patch("providers.codex_quota.QUOTA_PROBE_TTL_S", 0):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = omni_cli.main(["health", "--probe"])
        self.assertEqual(rc, 0)
        self.assertNotIn(key, health.snapshot())

    def test_r_canary_pii_never_surfaces(self):
        import omni_cli
        os.environ["FAKE_CODEX_MODE"] = "canary"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            rc = omni_cli.main(["health", "--probe"])
        self.assertEqual(rc, 0)
        blobs = [stdout.getvalue(), stderr.getvalue()]
        for name in ("health.json", "quota.json"):
            path = os.path.join(self.state, name)
            if os.path.isfile(path):
                blobs.append(open(path, encoding="utf-8").read())
        self.assertFalse(self._canary_in(*blobs), blobs[:2])

        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        stdout2 = io.StringIO()
        stderr2 = io.StringIO()
        with contextlib.redirect_stdout(stdout2), \
                contextlib.redirect_stderr(stderr2), \
                mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router):
            _run(dispatch.dispatch_llm_request(
                "p", "codex/gpt-5.6-sol,backup", 30, None, "default"))
        blobs2 = [stdout2.getvalue(), stderr2.getvalue()]
        for name in ("health.json", "quota.json"):
            path = os.path.join(self.state, name)
            if os.path.isfile(path):
                blobs2.append(open(path, encoding="utf-8").read())
        self.assertFalse(self._canary_in(*blobs2))

    def test_x_error_message_canary_never_surfaces(self):
        os.environ["FAKE_CODEX_MODE"] = "rpc_error"
        os.environ["FAKE_CODEX_ERRMSG"] = CANARY
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            info, err = read_rate_limits("default", timeout_s=5)
        self.assertIsNone(info)
        self.assertEqual(err["reason"], "runtime_error")
        self.assertNotIn(CANARY, json.dumps(err))
        self.assertNotIn(CANARY, err.get("detail") or "")
        self.assertNotIn(CANARY, stderr.getvalue())
        self.assertIn("code=", err.get("detail") or "")

        os.environ["FAKE_CODEX_MODE"] = "huge"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            info, err = read_rate_limits("default", timeout_s=1)
        self.assertEqual(err["reason"], "timeout")
        self.assertNotIn(CANARY, stderr.getvalue())
        self.assertNotIn(CANARY, json.dumps(err))


if __name__ == "__main__":
    unittest.main()
