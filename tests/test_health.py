# no-opencc
"""Acceptance tests — seat-health circuit breaker (issue #18 proposal A, plan R1).

Pinned contract:
  * ledger lives under $OMNI_LLM_STATE_DIR/health.json, keyed "<pool>@<profile>";
    profile is the given profile (None → "default") on every road, never collapsed;
    cursor/<slug> keys by Cursor billing pool (cursor-models / other-models / auto);
  * a final rate_limit envelope flags the seat; an ok envelope clears it;
  * a multi-seat chain skips cooling seats (recorded in the envelope's "tried");
  * an ALL-cooling chain still dispatches — the breaker degrades to advisory and
    must never turn a partial outage into a total one;
  * single-model calls are flagged but never blocked;
  * a corrupt ledger file reads as empty, never raises.
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-health-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import dispatch  # noqa: E402
from providers import health  # noqa: E402

_real_account_identity = health._account_identity
_account_identity_patcher = mock.patch.object(
    health, "_account_identity", return_value=None)


def setUpModule():
    _account_identity_patcher.start()


def tearDownModule():
    _account_identity_patcher.stop()


def _run(coro):
    return asyncio.run(coro)


def _envelope(ok, reason=None):
    res = {"ok": ok, "text": "hi" if ok else "", "model": "m"}
    if reason:
        res["reason"] = reason
    return res


class PoolKeyTest(unittest.TestCase):
    def test_cli_roads_keep_profile(self):
        # Issue #72: a non-default profile is a separate login; never collapse.
        self.assertNotEqual(health.pool_key("codex/gpt-5.6-sol", "alt"),
                            health.pool_key("codex/gpt-5.6-sol", "default"))
        self.assertEqual(health.pool_key("codex/gpt-5.6-sol", "alt"),
                         "codex/gpt-5.6-sol@alt")
        self.assertNotEqual(health.pool_key("claude/opus", "alt"),
                            health.pool_key("claude/opus", "default"))
        self.assertNotEqual(health.pool_key("pi/xai/grok-4.6", "alt"),
                            health.pool_key("pi/xai/grok-4.6", "default"))
        self.assertNotEqual(health.pool_key("cursor/sol", "alt"),
                            health.pool_key("cursor/sol", "default"))
        self.assertNotEqual(health.pool_key("cursor/fable", "work"),
                            health.pool_key("cursor/fable", "default"))
        self.assertEqual(health.pool_key("codex/gpt-5.6-sol", None),
                         health.pool_key("codex/gpt-5.6-sol", "default"))
        self.assertEqual(health.pool_key("cursor/sol", None),
                         health.pool_key("cursor/sol", "default"))
        self.assertEqual(health.pool_key("grok-4.5", None),
                         health.pool_key("grok-4.5", "default"))

    def test_profiled_providers_keep_profile(self):
        self.assertNotEqual(health.pool_key("grok-4.5", "a"),
                            health.pool_key("grok-4.5", "b"))
        self.assertEqual(health.pool_key("grok-4.5", "a"), "grok-4.5@a")

    def test_cursor_other_models_share_one_key(self):
        keys = [
            health.pool_key("cursor/gpt-5.6-sol-high", "default"),
            health.pool_key("cursor/claude-fable-5-1", "default"),
            health.pool_key("cursor/fable", "default"),
            health.pool_key("cursor/sol", "default"),
        ]
        self.assertEqual(set(keys), {"cursor:other-models@default"})

    def test_cursor_models_pool_shares_one_key(self):
        a = health.pool_key("cursor/composer-2.5", "default")
        b = health.pool_key("cursor/cursor-grok-4.6-high", "default")
        other = health.pool_key("cursor/sol", "default")
        self.assertEqual(a, b)
        self.assertEqual(a, "cursor:cursor-models@default")
        self.assertNotEqual(a, other)

    def test_cursor_auto_is_its_own_pool(self):
        auto = health.pool_key("cursor/auto", "default")
        self.assertEqual(auto, "cursor:auto@default")
        self.assertNotEqual(auto, health.pool_key("cursor/sol", "default"))
        self.assertNotEqual(
            auto, health.pool_key("cursor/composer-2.5", "default"))


class SplitPoolKeyTest(unittest.TestCase):
    def test_t9_split_pool_key_shapes(self):
        self.assertEqual(
            health.split_pool_key("cursor:cursor-models@x"),
            ("cursor", "bucket", "cursor-models", None, "x"))
        self.assertEqual(
            health.split_pool_key("codex/sol@x"),
            ("codex", "model", "sol", None, "x"))
        self.assertEqual(
            health.split_pool_key("codex@x"),
            ("codex", "account", None, None, "x"))
        self.assertEqual(
            health.split_pool_key("codex#week@x"),
            ("codex", "account", None, "week", "x"))

    def test_invalid_quota_window_names_are_unknown(self):
        self.assertEqual(
            health.split_pool_key("codex#m300@p"),
            ("codex", "account", None, "m300", "p"))
        self.assertEqual(
            health.split_pool_key("codex#week/model@p"),
            ("codex", "unknown", None, None, "p"))
        self.assertEqual(
            health.split_pool_key("codex#@p"),
            ("codex", "unknown", None, None, "p"))
        self.assertEqual(
            health.split_pool_key("codex#weekly@p"),
            ("codex", "unknown", None, None, "p"))


class LedgerTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_flag_active_clear_roundtrip(self):
        self.assertFalse(health.active("m1", "default"))
        health.flag("m1", "default", source="test")
        self.assertTrue(health.active("m1", "default"))
        health.clear("m1", "default")
        self.assertFalse(health.active("m1", "default"))

    def test_shorter_flag_never_shortens_an_active_cooldown(self):
        long_until = int(time.time()) + 21600
        health.flag("m5", "default", until=long_until, source="credit_exhausted")
        health.flag("m5", "default", until=int(time.time()) + 2700, source="rate_limit")
        entry = [e for e in health._read_ledger().values() if e.get("model") == "m5"][0]
        self.assertEqual(entry["until"], long_until)
        self.assertEqual(entry["source"], "credit_exhausted")
        self.assertTrue(health.active("m5", "default"))

    def test_longer_flag_extends_an_active_cooldown(self):
        short_until = int(time.time()) + 2700
        long_until = int(time.time()) + 21600
        health.flag("m6", "default", until=short_until, source="rate_limit")
        health.flag("m6", "default", until=long_until, source="credit_exhausted")
        entry = [e for e in health._read_ledger().values() if e.get("model") == "m6"][0]
        self.assertEqual(entry["until"], long_until)
        self.assertEqual(entry["source"], "credit_exhausted")

    def test_expired_cooldown_is_replaced_by_a_shorter_new_flag(self):
        health.flag("m7", "default", until=time.time() - 5, source="credit_exhausted")
        new_until = int(time.time()) + 2700
        health.flag("m7", "default", until=new_until, source="rate_limit")
        entry = [e for e in health._read_ledger().values() if e.get("model") == "m7"][0]
        self.assertEqual((entry["until"], entry["source"]), (new_until, "rate_limit"))

    def test_profile_isolation(self):
        health.flag("grok-4.5", "a", source="test")
        self.assertTrue(health.active("grok-4.5", "a"))
        self.assertFalse(health.active("grok-4.5", "b"))

    def test_expired_cooldown_is_inactive(self):
        health.flag("m2", "default", until=time.time() - 5, source="test")
        self.assertFalse(health.active("m2", "default"))

    def test_corrupt_file_reads_as_empty(self):
        with open(os.path.join(_TMP, "health.json"), "w") as f:
            f.write("{not json")
        self.assertFalse(health.active("m3", "default"))
        health.flag("m3", "default", source="test")  # must also survive a write
        self.assertTrue(health.active("m3", "default"))

    def test_snapshot_lists_flags(self):
        health.flag("m4", "default", source="test")
        snap = health.snapshot()
        self.assertIn(health.pool_key("m4", "default"), snap)


def _wipe_leases():
    def mutator(data):
        data.pop(health._LEASES_KEY, None)
        return None
    health._locked_mutate("lease_test_wipe", mutator)


class LeaseTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()
        _wipe_leases()

    def test_l1_acquire_shows_in_leases(self):
        token = str(uuid.uuid4())
        now = time.time()
        rec = health.lease_acquire(token, "codex-default", "wo-l1", 60)
        self.assertEqual(rec["token"], token)
        self.assertEqual(rec["pool"], "codex-default")
        self.assertEqual(rec["order_id"], "wo-l1")
        self.assertAlmostEqual(rec["expires_at"], now + 60, delta=2)
        rows = health.leases()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["token"], token)
        self.assertEqual(rows[0]["pool"], "codex-default")
        self.assertEqual(rows[0]["order_id"], "wo-l1")
        self.assertAlmostEqual(rows[0]["expires_at"], now + 60, delta=2)

    def test_l2_acquire_same_active_token_refuses(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-l2", 60)
        with self.assertRaises(ValueError) as cm:
            health.lease_acquire(token, "codex-default", "wo-l2b", 60)
        self.assertEqual(str(cm.exception), "lease_exists")

    def test_l3_renew_extends_unknown_raises(self):
        token = str(uuid.uuid4())
        first = health.lease_acquire(token, "codex-default", "wo-l3", 30)
        time.sleep(0.05)
        renewed = health.lease_renew(token)
        self.assertGreater(renewed["expires_at"], first["expires_at"])
        missing = str(uuid.uuid4())
        with self.assertRaises(KeyError) as cm:
            health.lease_renew(missing)
        self.assertEqual(cm.exception.args[0], "unknown_lease")

    def test_l4_release_removes_then_unknown(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-l4", 60)
        health.lease_release(token)
        self.assertEqual(health.leases(), [])
        with self.assertRaises(KeyError) as cm:
            health.lease_release(token)
        self.assertEqual(cm.exception.args[0], "unknown_lease")

    def test_l5_expired_absent_and_does_not_block_reacquire(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-l5", 60)
        path = health._ledger_path()
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored[health._LEASES_KEY][token]["expires_at"] = time.time() - 30
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        self.assertEqual(health.leases(), [])
        rec = health.lease_acquire(token, "codex-default", "wo-l5", 60)
        self.assertEqual(rec["token"], token)
        self.assertEqual(len(health.leases()), 1)

    def test_l6_snapshot_skips_leases_and_flag_clear_roundtrip(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-l6", 60)
        snap = health.snapshot()
        self.assertNotIn(health._LEASES_KEY, snap)
        self.assertFalse(health.active("m-lease", "default"))
        health.flag("m-lease", "default", source="test")
        self.assertTrue(health.active("m-lease", "default"))
        self.assertEqual(len(health.leases()), 1)
        self.assertNotIn(health._LEASES_KEY, health.snapshot())
        health.clear("m-lease", "default")
        self.assertFalse(health.active("m-lease", "default"))
        self.assertEqual(len(health.leases()), 1)
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertIn(health._LEASES_KEY, stored)
        self.assertIn(token, stored[health._LEASES_KEY])
        self.assertNotIn(health._LEASES_KEY, health.snapshot())

    def test_leases_read_does_not_notify_expired_cooldown(self):
        health.flag("m-expired", "default", until=time.time() - 5, source="test")
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-read", 60)
        with mock.patch.object(health, "notify") as note:
            rows = health.leases()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["token"], token)
        self.assertEqual(note.call_args_list, [])

    def test_leases_sorts_numeric_and_iso_expires_at(self):
        now = time.time()
        token_num = str(uuid.uuid4())
        token_iso = str(uuid.uuid4())
        earlier = now + 60
        later = now + 120
        iso = datetime.fromtimestamp(later, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z")
        path = health._ledger_path()
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored[health._LEASES_KEY] = {
            token_num: {
                "token": token_num,
                "pool": "codex-default",
                "order_id": "num",
                "ttl_s": 60,
                "acquired_at": now,
                "renewed_at": now,
                "expires_at": earlier,
            },
            token_iso: {
                "token": token_iso,
                "pool": "codex-default",
                "order_id": "iso",
                "ttl_s": 120,
                "acquired_at": now,
                "renewed_at": now,
                "expires_at": iso,
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        rows = health.leases()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["token"] for row in rows], [token_num, token_iso])

    def test_acquire_rejects_bad_ttl(self):
        token = str(uuid.uuid4())
        for ttl in (0, -1, float("nan"), float("inf"), float("-inf"), "nope", None):
            with self.assertRaises(ValueError) as cm:
                health.lease_acquire(token, "codex-default", "wo-ttl", ttl)
            self.assertEqual(str(cm.exception), "bad_ttl")
        self.assertEqual(health.leases(), [])

    def test_renew_rejects_bad_ttl(self):
        token = str(uuid.uuid4())
        first = health.lease_acquire(token, "codex-default", "wo-ttl", 60)
        for ttl in (0, -1, float("nan"), float("inf"), float("-inf"), "nope"):
            with self.assertRaises(ValueError) as cm:
                health.lease_renew(token, ttl_s=ttl)
            self.assertEqual(str(cm.exception), "bad_ttl")
        rows = health.leases()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expires_at"], first["expires_at"])

    def test_flag_prunes_expired_lease_without_notify(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-prune", 60)
        path = health._ledger_path()
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored[health._LEASES_KEY][token]["expires_at"] = time.time() - 30
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with mock.patch.object(health, "notify") as note:
            health.flag("m-flag", "default", source="test")
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        leases = stored.get(health._LEASES_KEY)
        if isinstance(leases, dict):
            self.assertNotIn(token, leases)
        else:
            self.assertTrue(leases is None or not leases)
        self.assertEqual(_notify_calls(note, "seat_cleared"), [])

    def test_clear_all_keeps_lease_bucket(self):
        token = str(uuid.uuid4())
        health.lease_acquire(token, "codex-default", "wo-keep", 60)
        health.flag("m-wipe", "default", source="test")
        health.clear_all()
        self.assertFalse(health.active("m-wipe", "default"))
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertIn(health._LEASES_KEY, stored)
        self.assertIn(token, stored[health._LEASES_KEY])
        self.assertEqual(len(health.leases()), 1)
        self.assertEqual(health.leases()[0]["token"], token)


class CrashStreakTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_note_failure_flags_at_limit(self):
        for _ in range(2):
            health.note_failure("m", "default", "runtime_error")
        self.assertFalse(health.active("m", "default"))
        health.note_failure("m", "default", "runtime_error")
        key = health.pool_key("m", "default")
        entry = health.snapshot()[key]
        self.assertEqual(entry["source"], "crash")
        self.assertEqual(entry["crash_streak"], 3)
        now = time.time()
        self.assertGreater(entry["until"], now + 250)
        self.assertLess(entry["until"], now + 350)

    def test_signal_exit_counts_as_crash(self):
        for _ in range(3):
            health.note_failure("m", "default", "signal_exit")
        self.assertEqual(
            health.snapshot()[health.pool_key("m", "default")]["source"],
            "crash")

    def test_non_crash_reason_resets_streak(self):
        for _ in range(2):
            health.note_failure("m", "default", "runtime_error")
        health.note_failure("m", "default", "auth_error")
        for _ in range(2):
            health.note_failure("m", "default", "runtime_error")
        self.assertFalse(health.active("m", "default"))

    def test_clear_resets_streak(self):
        for _ in range(2):
            health.note_failure("m", "default", "runtime_error")
        health.clear("m", "default")
        health.note_failure("m", "default", "runtime_error")
        self.assertFalse(health.active("m", "default"))

    def test_crash_source_uses_half_open_probe(self):
        until = time.time() + 3600
        health.flag("m", "default", until=until, source="crash")
        now = time.time()
        self.assertEqual(health.seat_kind("m", "default", now=now), "cooling")
        self.assertEqual(
            health.seat_kind(
                "m", "default", now=now + health.HALF_OPEN_MIN_S),
            "probing")

    def test_rate_limit_flag_preserves_crash_streak(self):
        health.note_failure("m", "default", "runtime_error")
        health.note_failure("m", "default", "runtime_error")
        flagged = health.flag("m", "default", source="rate_limit")
        self.assertEqual(flagged["crash_streak"], 2)
        self.assertIsNotNone(flagged.get("last_crash_at"))
        health.note_failure("m", "default", "runtime_error")
        key = health.pool_key("m", "default")
        entry = health.snapshot()[key]
        self.assertEqual(entry["source"], "crash")
        self.assertEqual(entry["crash_streak"], 3)

    def test_streak_window_expiry_starts_at_one(self):
        now = 1_700_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            health.note_failure("m", "default", "runtime_error")
            health.note_failure("m", "default", "runtime_error")
        later = now + health.CRASH_STREAK_WINDOW_S + 1
        with mock.patch.object(health.time, "time", return_value=later):
            entry = health.note_failure("m", "default", "runtime_error")
        self.assertEqual(entry["crash_streak"], 1)
        self.assertFalse(health.active("m", "default"))

    def test_post_crash_cooldown_first_crash_does_not_reflag(self):
        now = 1_700_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            for _ in range(3):
                health.note_failure("m", "default", "runtime_error")
            self.assertTrue(health.active("m", "default"))
        later = now + health.CRASH_COOLDOWN_S + 1
        with mock.patch.object(health.time, "time", return_value=later):
            entry = health.note_failure("m", "default", "runtime_error")
        self.assertEqual(entry["crash_streak"], 1)
        self.assertNotEqual(entry.get("source"), "crash")
        self.assertFalse(health.active("m", "default"))


class DispatchIntegrationTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def _dispatch(self, chain, router):
        with mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
             mock.patch.object(dispatch, "route_to_provider",
                               side_effect=router) as routed:
            res = _run(dispatch.dispatch_llm_request(
                "p", chain, 30, None, "default"))
        return res, [c.args[1] for c in routed.call_args_list]

    def test_rate_limit_flags_and_next_chain_call_skips(self):
        async def router(prompt, model, *a, **k):
            return _envelope(False, "rate_limit") if model == "A" \
                else _envelope(True)
        res, models = self._dispatch("A,B", router)
        self.assertTrue(res["ok"])
        self.assertTrue(health.active("A", "default"))

        res2, models2 = self._dispatch("A,B", router)
        self.assertTrue(res2["ok"])
        self.assertNotIn("A", models2, "cooling seat A was dispatched again")
        skipped = [t for t in res2.get("tried", [])
                   if t.get("model") == "A" and "cooldown" in str(t.get("reason", ""))]
        self.assertTrue(skipped, "skip not recorded in tried: %r" % res2.get("tried"))

    def test_success_clears_flag(self):
        health.flag("B", "default", source="test")

        async def router(prompt, model, *a, **k):
            return _envelope(True)
        res, _ = self._dispatch("B", router)
        self.assertTrue(res["ok"])
        self.assertFalse(health.active("B", "default"))

    def test_all_cooling_chain_still_dispatches(self):
        health.flag("A", "default", source="test")
        health.flag("B", "default", source="test")

        async def router(prompt, model, *a, **k):
            return _envelope(True)
        res, models = self._dispatch("A,B", router)
        self.assertTrue(res["ok"])
        self.assertTrue(models, "all-cooling chain dispatched nothing")

    def test_single_model_call_never_blocked(self):
        health.flag("A", "default", source="test")

        async def router(prompt, model, *a, **k):
            return _envelope(True)
        res, models = self._dispatch("A", router)
        self.assertTrue(res["ok"])
        self.assertEqual(models, ["A"])

    def test_skipped_cooling_seat_retried_after_live_failure(self):
        health.flag("A", "default", source="test")

        async def router(prompt, model, *a, **k):
            return _envelope(False, "backend_error")

        res, models = self._dispatch("A,B", router)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "backend_error")
        self.assertEqual(models, ["B", "A"])
        self.assertEqual([item["model"] for item in res["tried"]],
                         ["A", "B", "A"])

    def test_health_hooks_cannot_change_dispatch_envelopes(self):
        rate_limit = _envelope(False, "rate_limit")
        success = _envelope(True)

        async def rate_router(prompt, model, *a, **k):
            return rate_limit

        async def success_router(prompt, model, *a, **k):
            return success

        with mock.patch.object(health, "flag", side_effect=OSError("flag")), \
                mock.patch.object(health, "clear", side_effect=OSError("clear")), \
                mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=rate_router):
            rate_result = _run(dispatch.dispatch_single(
                "p", "A", 30, None, "default"))

        with mock.patch.object(health, "clear", side_effect=OSError("clear")), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=success_router):
            ok_result = _run(dispatch.dispatch_single(
                "p", "A", 30, None, "default"))

        self.assertEqual(rate_result, rate_limit)
        self.assertEqual(ok_result, success)

    def test_three_runtime_errors_flag_crash_and_fourth_skips(self):
        async def router(prompt, model, *a, **k):
            return _envelope(False, "runtime_error") if model == "A" \
                else _envelope(True)
        for _ in range(3):
            res, _ = self._dispatch("A,B", router)
            self.assertTrue(res["ok"], res)
        key = health.pool_key("A", "default")
        snap = health.snapshot()
        self.assertIn(key, snap)
        self.assertEqual(snap[key]["source"], "crash")
        self.assertEqual(snap[key]["crash_streak"], 3)
        until = snap[key]["until"]
        now = time.time()
        self.assertGreater(until, now + 250)
        self.assertLess(until, now + 350)

        res2, models2 = self._dispatch("A,B", router)
        self.assertTrue(res2["ok"], res2)
        self.assertNotIn("A", models2)
        skipped = [t for t in res2.get("tried", []) if t.get("model") == "A"]
        self.assertTrue(skipped, res2.get("tried"))
        self.assertEqual(skipped[0]["reason"], "skipped_cooldown")

    def test_success_resets_crash_streak(self):
        async def fail(prompt, model, *a, **k):
            return _envelope(False, "runtime_error")

        async def ok(prompt, model, *a, **k):
            return _envelope(True)

        for _ in range(2):
            self._dispatch("A", fail)
        self._dispatch("A", ok)
        for _ in range(2):
            self._dispatch("A", fail)
        self.assertFalse(health.active("A", "default"))
        self._dispatch("A", fail)
        key = health.pool_key("A", "default")
        self.assertEqual(health.snapshot()[key]["source"], "crash")

    def test_probing_runtime_error_still_notes_failure(self):
        async def router(prompt, model, *a, **k):
            return _envelope(False, "runtime_error")
        with mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
             mock.patch.object(dispatch, "route_to_provider",
                               side_effect=router):
            for _ in range(3):
                _run(dispatch.dispatch_single(
                    "p", "A", 30, None, "default", probing=True))
        key = health.pool_key("A", "default")
        snap = health.snapshot()
        self.assertEqual(snap[key]["source"], "crash")
        self.assertEqual(snap[key]["crash_streak"], 3)

    def test_probing_success_still_clears(self):
        health.flag("A", "default", source="test")

        async def router(prompt, model, *a, **k):
            return _envelope(True)
        with mock.patch.object(dispatch, "route_to_provider",
                               side_effect=router):
            res = _run(dispatch.dispatch_single(
                "p", "A", 30, None, "default", probing=True))
        self.assertTrue(res["ok"])
        self.assertFalse(health.active("A", "default"))


class CodexPassiveProbeTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_latest_codex_rate_limits_probe_and_null_primary(self):
        with tempfile.TemporaryDirectory() as sessions:
            nested = os.path.join(sessions, "2026", "08")
            os.makedirs(nested)
            path = os.path.join(nested, "session.jsonl")

            with open(path, "w") as handle:
                json.dump({"rate_limits": {
                    "primary": {"used_percent": 97,
                                "resets_at": time.time() + 600}
                }}, handle)
                handle.write("\n")

            async def router(prompt, model, *args, **kwargs):
                return _envelope(True)

            with mock.patch.dict(os.environ, {
                "OMNI_LLM_CODEX_SESSIONS_DIR": sessions,
            }), mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
                    mock.patch.object(dispatch, "route_to_provider",
                                      side_effect=router) as routed:
                result = _run(dispatch.dispatch_llm_request(
                    "p", "codex/gpt-5.6-sol,backup", 30, None, "default"))
            models = [call.args[1] for call in routed.call_args_list]
            self.assertTrue(result["ok"])
            self.assertEqual(models, ["backup"])
            key = health.pool_key("codex/gpt-5.6-sol", "default")
            self.assertEqual(health.snapshot()[key]["source"], "codex_passive")

            health.clear_all()
            with open(path, "w") as handle:
                json.dump({"rate_limits": {"primary": None}}, handle)
                handle.write("\n")

            with mock.patch.dict(os.environ, {
                "OMNI_LLM_CODEX_SESSIONS_DIR": sessions,
            }), mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
                    mock.patch.object(dispatch, "route_to_provider",
                                      side_effect=router) as routed:
                result = _run(dispatch.dispatch_llm_request(
                    "p", "codex/gpt-5.6-sol,backup", 30, None, "default"))
            self.assertTrue(result["ok"])
            self.assertEqual([call.args[1] for call in routed.call_args_list],
                             ["codex/gpt-5.6-sol"])
            self.assertNotIn(key, health.snapshot())

    def test_threshold_95_flags_and_skips_but_94_does_not(self):
        with tempfile.TemporaryDirectory() as sessions:
            path = os.path.join(sessions, "session.jsonl")

            def write_probe(used_percent):
                with open(path, "w") as handle:
                    json.dump({"rate_limits": {
                        "primary": {"used_percent": used_percent,
                                    "resets_at": time.time() + 600}
                    }}, handle)
                    handle.write("\n")

            async def router(prompt, model, *args, **kwargs):
                return _envelope(True)

            write_probe(94)
            with mock.patch.dict(os.environ, {
                "OMNI_LLM_CODEX_SESSIONS_DIR": sessions,
            }), mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router) as routed:
                result = _run(dispatch.dispatch_llm_request(
                    "p", "codex/gpt-5.6-sol,backup", 30, None, "default"))
            key = health.pool_key("codex/gpt-5.6-sol", "default")
            self.assertTrue(result["ok"])
            self.assertEqual([call.args[1] for call in routed.call_args_list],
                             ["codex/gpt-5.6-sol"])
            self.assertNotIn(key, health.snapshot())
            self.assertNotIn("tried", result)

            health.clear_all()
            write_probe(95)
            with mock.patch.dict(os.environ, {
                "OMNI_LLM_CODEX_SESSIONS_DIR": sessions,
            }), mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router) as routed:
                result = _run(dispatch.dispatch_llm_request(
                    "p", "codex/gpt-5.6-sol,backup", 30, None, "default"))
            self.assertTrue(result["ok"])
            self.assertEqual([call.args[1] for call in routed.call_args_list],
                             ["backup"])
            self.assertIn(key, health.snapshot())
            self.assertEqual(result["tried"][0], {
                "model": "codex/gpt-5.6-sol", "reason": "skipped_cooldown",
            })

    def test_stale_reset_does_not_write_a_flag(self):
        with tempfile.TemporaryDirectory() as sessions:
            path = os.path.join(sessions, "session.jsonl")
            with open(path, "w") as handle:
                json.dump({"rate_limits": {
                    "primary": {"used_percent": 95,
                                "resets_at": time.time() - 1}
                }}, handle)
                handle.write("\n")

            with mock.patch.dict(os.environ, {
                "OMNI_LLM_CODEX_SESSIONS_DIR": sessions,
            }), mock.patch.object(health, "flag") as flagged:
                self.assertFalse(health.codex_passive_probe(
                    "codex/gpt-5.6-sol", "default"))
            flagged.assert_not_called()
            self.assertFalse(health.active("codex/gpt-5.6-sol", "default"))

    def test_probe_errors_are_logged_and_return_false(self):
        with mock.patch.object(health, "_latest_session_file",
                               side_effect=OSError("probe broke")), \
                mock.patch.object(health, "log_event") as logged:
            self.assertFalse(health.codex_passive_probe(
                "codex/gpt-5.6-sol", "default"))
        logged.assert_called_once()
        args, fields = logged.call_args
        self.assertEqual(args, ("health_error",))
        self.assertEqual(fields["operation"], "passive_probe")
        self.assertIn("probe broke", fields["detail"])


class CliHealthTest(unittest.TestCase):
    def test_health_flag_command_writes_manual_snapshot(self):
        import contextlib
        import io
        import omni_cli

        health.clear_all()
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                code = omni_cli.main([
                    "health", "flag", "grok-4.5", "--profile", "premium",
                ])
            self.assertEqual(code, 0)
            snapshot = json.loads(stdout.getvalue())
            key = health.pool_key("grok-4.5", "premium")
            self.assertEqual(snapshot[key]["source"], "manual")
        finally:
            health.clear_all()

    def test_health_quiet_payload_shape(self):
        import contextlib
        import io
        import omni_cli

        health.clear_all()
        health.flag("grok-4.5", "default", source="test")
        canned = {
            "codex": {"default": {}, "b": {}},
            "claude": {"default": {}},
        }
        stdout = io.StringIO()
        try:
            with mock.patch("providers.profile_check.probe_all",
                            return_value=canned), \
                    contextlib.redirect_stdout(stdout):
                code = omni_cli.main([
                    "health", "--probe", "--all-profiles", "--quiet",
                ])
            self.assertEqual(code, 0)
            raw = stdout.getvalue()
            self.assertEqual(raw.count("\n"), 1)
            payload = json.loads(raw)
            self.assertEqual(set(payload.keys()),
                             {"ok", "probed", "cooling", "repaired"})
            self.assertIs(payload["ok"], True)
            self.assertEqual(payload["probed"], 3)
            self.assertIsInstance(payload["cooling"], list)
            self.assertEqual(payload["repaired"], 0)
            key = health.pool_key("grok-4.5", "default")
            self.assertIn(key, payload["cooling"])
        finally:
            health.clear_all()


class TestProbeAll(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_probe_all_codex_quota_claude_signal_none(self):
        from providers.profile_check import probe_all

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "fixtures", "probe_all_registry.json")
        with open(fixture, encoding="utf-8") as handle:
            rows = json.load(handle)["profiles"]
        info = {
            "used_percent": 12, "resets_at": None, "window_mins": 300,
            "reached": False, "plan": "plus", "checked_at": 1,
        }
        seen = []

        def fake_read(profile):
            seen.append(profile)
            return (info, None)

        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           side_effect=fake_read) as rr, \
                mock.patch("providers.health.apply_codex_quota",
                           wraps=health.apply_codex_quota) as applied, \
                mock.patch("providers.health.maintain",
                           wraps=health.maintain) as maintained:
            quota = probe_all()
        self.assertEqual(seen, ["default", "b"])
        self.assertEqual(rr.call_count, 2)
        self.assertEqual(applied.call_count, 2)
        applied.assert_any_call("codex", "default", info, None)
        applied.assert_any_call("codex", "b", info, None)
        maintained.assert_called_once()
        self.assertEqual(quota["claude"]["default"]["signal"], "none")
        self.assertIn("cooldown", quota["claude"]["default"])
        self.assertEqual(quota["codex"]["default"]["used_percent"], 12)
        self.assertEqual(quota["codex"]["b"]["used_percent"], 12)
        blob = json.dumps(quota)
        self.assertNotIn("xyz789", blob)
        self.assertNotIn("abc123", blob)
        self.assertNotIn("synth@example.test", blob)
        self.assertNotIn("identity", blob)
        self.assertNotIn("emailAddress", blob)


class FlagRecheckVisibilityTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_manual_probe_after_matches_expiry_not_half_open(self):
        now = 1_800_000_000.0
        until = now + 3600
        with mock.patch.object(health.time, "time", return_value=now):
            health.flag("grok-4.5", "a", until=until, source="manual")
            health.flag("grok-4.6", "b", until=until, source="quota")
        manual = health.snapshot()[health.pool_key("grok-4.5", "a")]
        quota = health.snapshot()[health.pool_key("grok-4.6", "b")]
        # Mutation (Z3): drop manual from the no-early-probe branch — then
        # probe_after becomes now+HALF_OPEN_MIN_S and the two diverge.
        self.assertEqual(manual["probe_after"], quota["probe_after"])
        self.assertEqual(manual["probe_after"], int(until))
        self.assertNotEqual(
            manual["probe_after"], now + health.HALF_OPEN_MIN_S)
        self.assertEqual(
            health.seat_kind("grok-4.5", "a",
                             now=now + health.HALF_OPEN_MIN_S),
            "cooling")

    def test_will_recheck_quota_follows_probe_all_not_source_literal(self):
        from providers.profile_check import probe_all

        rows = [
            {"road": "codex", "name": "codex-a"},
            {"road": "claude", "name": "claude-a"},
            {"road": "cursor", "name": "cursor-a"},
        ]
        applied = []

        def spy_apply(model, profile, info, err):
            applied.append(profile)
            return False

        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           return_value=({}, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy_apply), \
                mock.patch("providers.health.maintain"):
            probe_all()
        covered = {row["road"] for row in rows if row["name"] in applied}
        # Mutation (Z1): a per-source literal makes every quota flag the
        # same; True and False must both appear, matching probe_all.
        seen = set()
        samples = (
            ("codex", "codex-a"),
            ("claude/opus", "claude-a"),
            ("cursor/sol", "cursor-a"),
        )
        for model, profile in samples:
            health.clear_all()
            health.flag(model, profile, source="quota")
            entry = health.snapshot()[health.pool_key(model, profile)]
            road, _kind, _name, _window, _profile = health.split_pool_key(
                health.pool_key(model, profile))
            self.assertEqual(entry["will_recheck"], road in covered)
            seen.add(entry["will_recheck"])
        self.assertEqual(seen, {True, False})

    def test_will_recheck_false_when_nothing_will_recheck(self):
        health.flag("claude/opus", "default", source="vendor_reset")
        health.flag("grok-4.5", "default", source="manual")
        health.flag("claude/sonnet", "default", source="rate_limit")
        health.flag("pi/xai/grok-4.6", "default", source="crash")
        # Mutation (Z2): every flag claims a re-check — vendor_reset and
        # manual must stay false; rate_limit / crash stay true (half-open).
        self.assertFalse(
            health.snapshot()[
                health.pool_key("claude/opus", "default")]["will_recheck"])
        self.assertFalse(
            health.snapshot()[
                health.pool_key("grok-4.5", "default")]["will_recheck"])
        self.assertTrue(
            health.snapshot()[
                health.pool_key("claude/sonnet", "default")]["will_recheck"])
        self.assertTrue(
            health.snapshot()[
                health.pool_key("pi/xai/grok-4.6", "default")]["will_recheck"])

    def test_will_recheck_tracks_a_road_dropped_from_the_probe_table(self):
        from providers import profile_check
        model = "codex"
        health.flag(model, "default", source="quota")
        key = health.pool_key(model, "default")
        self.assertTrue(health.snapshot()[key]["will_recheck"])
        # Mutation (U2): restating a road literal in health ignores this
        # patch; the declared table is the only place coverage is stated.
        with mock.patch.object(profile_check, "PROBE_POOLS", {}):
            self.assertFalse(health.snapshot()[key]["will_recheck"])

    def test_will_recheck_tracks_a_second_road_added_to_the_probe_table(self):
        from providers import profile_check
        from providers.profile_check import probe_all

        model = "claude/opus"
        profile = "claude-a"
        health.flag(model, profile, source="quota")
        key = health.pool_key(model, profile)
        self.assertFalse(health.snapshot()[key]["will_recheck"])

        visited = []

        def spy(applied_model, name, info, err):
            visited.append(health.pool_key(applied_model, name))
            return False

        rows = [{"road": "claude", "name": profile}]
        with mock.patch.dict(profile_check.PROBE_POOLS,
                             {"claude": lambda: model}), \
                mock.patch("providers.profile_check.list_profiles",
                           return_value=rows), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           return_value=({}, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy), \
                mock.patch("providers.health.maintain"):
            probe_all()
            # Mutation (U5): an answer derived for codex specifically — a
            # "codex/%s" template, or the road hardcoded — leaves this False
            # while probe_all visits the key. One edit to the table has to
            # move both sides, on a road that is not codex.
            self.assertTrue(health.snapshot()[key]["will_recheck"])
            # Mutation (U8): answering only the first covered road passes or
            # fails on whichever road the iteration happens to yield first,
            # which is seed-dependent. Assert the whole set instead: every
            # covered road, or the claim is not the one issue #92 asked for.
            codex_key = health.pool_key("codex", profile)
            week_key = health.pool_key("codex#week", profile)
            health.flag("codex", profile, source="quota")
            self.assertEqual(health._scheduled_probe_keys(profile),
                             {key, codex_key, week_key})
            self.assertTrue(health.snapshot()[codex_key]["will_recheck"])
        self.assertEqual(visited, [key])
        self.assertFalse(health.snapshot()[key]["will_recheck"])

    def test_will_recheck_survives_a_pool_that_cannot_answer(self):
        from providers import profile_check

        def raises():
            raise RuntimeError("vendor table is broken")

        model = "codex"
        health.flag(model, "default", source="quota")
        key = health.pool_key(model, "default")
        # Mutation (U7): letting the exception out breaks every ledger read
        # (omni health, list_models) over a field that is advisory. Fail
        # closed instead — pessimistic is the safe direction here.
        with mock.patch.dict(profile_check.PROBE_POOLS, {"codex": raises}):
            snap = health.snapshot()
        self.assertFalse(snap[key]["will_recheck"])
        self.assertEqual(snap[key]["source"], "quota")
        self.assertTrue(health.snapshot()[key]["will_recheck"])

    def test_probe_coverage_is_derived_from_the_pool_table(self):
        from providers import profile_check
        # Mutation (U6): a second declaration of coverage (a set literal
        # beside the table) lets a road be covered with no pool to apply
        # the result to, which is the shape this issue closed.
        self.assertEqual(profile_check.probe_covered_roads(),
                         frozenset(profile_check.PROBE_POOLS))
        with mock.patch.dict(profile_check.PROBE_POOLS,
                             {"claude": lambda: "claude/opus"}):
            self.assertIn("claude", profile_check.probe_covered_roads())
            self.assertEqual(profile_check.probe_pool_model("claude"),
                             "claude/opus")
        self.assertNotIn("claude", profile_check.probe_covered_roads())
        self.assertIsNone(profile_check.probe_pool_model("claude"))

    def test_a_road_off_the_ring_is_not_coverage(self):
        from providers import profile_check
        from providers.profiles import RING_ROADS

        road = "pi"
        self.assertNotIn(road, RING_ROADS)
        model = "pi/synthetic-model"
        health.flag(model, "default", source="quota")
        key = health.pool_key(model, "default")
        with mock.patch.dict(profile_check.PROBE_POOLS,
                             {road: lambda: model}):
            # probe_all iterates list_profiles(), which drops every road off
            # the ring — so the table entry names a pool nothing visits.
            self.assertEqual(
                [r for r in profile_check.list_profiles()
                 if r.get("road") == road], [])
            # Mutation (U9): coverage taken straight off the table claims a
            # re-check here that no scheduled run will ever perform, which is
            # the failure issue #92 was filed about, one level up.
            self.assertNotIn(road, profile_check.probe_covered_roads())
            self.assertNotIn(key, health._scheduled_probe_keys("default"))
            self.assertFalse(health.snapshot()[key]["will_recheck"])

    def test_one_unanswerable_road_does_not_erase_the_others(self):
        from providers import profile_check

        def boom():
            raise RuntimeError("this road cannot name its pool")

        model = "codex"
        health.flag(model, "default", source="quota")
        key = health.pool_key(model, "default")
        week = health.pool_key("codex#week", "default")
        # Mutation (U10): failing closed around the whole loop throws away
        # keys the probe really does visit every run — the pessimistic answer
        # for a road that works is what gets a live flag cleared by hand.
        with mock.patch.dict(profile_check.PROBE_POOLS, {"cursor": boom}):
            self.assertEqual(health._scheduled_probe_keys("default"),
                             {key, week})
            self.assertTrue(health.snapshot()[key]["will_recheck"])

    def test_probe_pool_model_resolves_a_road_it_does_not_cover(self):
        from providers import profile_check
        from providers.profiles import RING_ROADS

        road = "pi"
        self.assertNotIn(road, RING_ROADS)
        with mock.patch.dict(profile_check.PROBE_POOLS,
                             {road: lambda: "pi/synthetic-model"}):
            # The resolver answers for a table entry; coverage is a separate
            # question with a different answer here. Pinned because the
            # docstring used to claim this returns None for a road the
            # scheduled probe does not cover, and a caller that asks only
            # this one would have believed it.
            self.assertEqual(profile_check.probe_pool_model(road), "pi/synthetic-model")
            self.assertNotIn(road, profile_check.probe_covered_roads())

    def test_will_recheck_answers_the_key_not_the_profile_roster(self):
        model = "codex"
        key = health.pool_key(model, "never-enumerated")
        health.flag(model, "never-enumerated", source="quota")
        # Decided, not overlooked (issue #92): the field answers what the
        # probe does when it runs for this profile, not whether
        # list_profiles() still enumerates it. Coupling the ledger read to
        # account enumeration would cost that walk on every snapshot to
        # catch a flag that expires at its own `until` regardless — so a
        # ledger read must not enumerate accounts.
        enumerated = []

        def roster(*args, **kwargs):
            enumerated.append(True)
            return []

        with mock.patch("providers.profile_check.list_profiles",
                        side_effect=roster):
            self.assertTrue(health.snapshot()[key]["will_recheck"])
        self.assertEqual(enumerated, [])

    def test_will_recheck_unreadable_coverage_covers_no_road(self):
        model = "codex"
        health.flag(model, "default", source="quota")
        key = health.pool_key(model, "default")
        self.assertTrue(health.snapshot()[key]["will_recheck"])
        # Mutation (U3): fail-open (every road covered) would keep this True.
        with mock.patch.dict(sys.modules, {"providers.profile_check": None}):
            self.assertFalse(health.snapshot()[key]["will_recheck"])

    def test_will_recheck_quota_only_the_key_the_probe_visits(self):
        from providers.profile_check import probe_all

        account_model = "codex"
        other_model = "codex/gpt-6-astra"
        self.assertNotEqual(account_model, other_model)
        profile = "default"
        visited = []

        def spy(model, name, info, err):
            visited.append(health.pool_key(model, name))
            return False

        rows = [{"road": "codex", "name": profile}]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           return_value=({}, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy), \
                mock.patch("providers.health.maintain"):
            probe_all()
        account_key = health.pool_key(account_model, profile)
        other_key = health.pool_key(other_model, profile)
        self.assertEqual(visited, [account_key])
        self.assertNotIn(other_key, visited)

        health.flag(account_model, profile, source="quota")
        health.flag(other_model, profile, source="quota")
        snap = health.snapshot()
        # Mutation (W1): deciding on the road alone makes both True.
        # Mutation (W3): a non-default key claiming a re-check fails here.
        self.assertTrue(snap[account_key]["will_recheck"])
        self.assertFalse(snap[other_key]["will_recheck"])
        self.assertEqual(snap[other_key]["will_recheck"],
                         other_key in visited)
        self.assertEqual(health.seat_kind(account_model, profile), "cooling")
        self.assertEqual(health.seat_kind(other_model, profile), "cooling")

    def test_will_recheck_lands_on_the_account_key_not_a_model(self):
        from providers import config
        other = "gpt-6-astra"
        self.assertNotEqual(config.DEFAULT_CODEX_MODEL, other)
        account_key = health.pool_key("codex", "default")
        week_key = health.pool_key("codex#week", "default")
        old_model = "codex/%s" % config.DEFAULT_CODEX_MODEL
        new_model = "codex/%s" % other
        health.flag("codex", "default", source="quota")
        health.flag(old_model, "default", source="quota")
        health.flag(new_model, "default", source="quota")
        old_key = health.pool_key(old_model, "default")
        new_key = health.pool_key(new_model, "default")
        self.assertTrue(health.snapshot()[account_key]["will_recheck"])
        self.assertFalse(health.snapshot()[old_key]["will_recheck"])
        self.assertFalse(health.snapshot()[new_key]["will_recheck"])
        # Mutation (W2): the landing is the road literal; patching the
        # default model must not move the write set.
        with mock.patch.object(config, "DEFAULT_CODEX_MODEL", other):
            self.assertTrue(health.snapshot()[account_key]["will_recheck"])
            self.assertFalse(health.snapshot()[old_key]["will_recheck"])
            self.assertFalse(health.snapshot()[new_key]["will_recheck"])
            self.assertEqual(
                health._scheduled_probe_keys("default"),
                {account_key, week_key})

    def test_will_recheck_is_derived_not_stored(self):
        health.flag("claude/opus", "default", source="quota")
        key = health.pool_key("claude/opus", "default")
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertNotIn("will_recheck", stored[key])
        self.assertIn("will_recheck", health.snapshot()[key])


class CodexAccountQuotaTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def _profile(self):
        return "p%s" % uuid.uuid4().hex[:12]

    def _set_probe_after(self, model, profile, ts):
        data = health._read_ledger_unlocked()
        key = health.pool_key(model, profile)
        entry = dict(data[key])
        entry["probe_after"] = ts
        data[key] = entry
        health._write_ledger(data)

    def test_t1_quota_writers_pass_the_codex_road(self):
        import contextlib
        import io
        import omni_cli
        from providers.profile_check import _quota_one, probe_all

        profile = self._profile()
        info = {"used_percent": 1, "reached": False}
        seen = []
        kinds = []

        def spy_apply(model, prof, inf, err):
            seen.append(model)
            return False

        def spy_kind(model, prof, now=None):
            kinds.append(model)
            return "healthy"

        rows = [{"road": "codex", "name": profile}]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           return_value=(info, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy_apply), \
                mock.patch("providers.health.maintain"):
            probe_all()
        row = {"road": "codex", "name": profile, "cooldown": None}
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy_apply), \
                mock.patch("providers.health.seat_kind",
                           side_effect=spy_kind):
            _quota_one(row)
        stdout = io.StringIO()
        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy_apply), \
                mock.patch.object(omni_cli.jobs_mod, "sweep_orphans",
                                  return_value=[]), \
                contextlib.redirect_stdout(stdout):
            omni_cli.main(["health", "--probe", "--profile", profile])
        self.assertEqual(seen, ["codex", "codex", "codex"])
        self.assertTrue(all("/" not in item for item in seen))
        self.assertTrue(kinds)
        self.assertTrue(all(item == "codex" for item in kinds))

    def test_passive_probe_applies_quota_on_the_codex_road(self):
        seen = []

        def spy(model, profile, info, err):
            seen.append(model)
            return False

        with mock.patch.object(health, "_latest_session_file",
                               return_value=None), \
                mock.patch("providers.codex_quota.read_rate_limits",
                           return_value=({}, None)), \
                mock.patch("providers.health.apply_codex_quota",
                           side_effect=spy):
            health.codex_passive_probe("codex/gpt-5.6-sol", "p")
        self.assertEqual(seen, ["codex"])

    def test_t2_account_record_cools_models_on_that_profile_only(self):
        profile = self._profile()
        other = self._profile()
        health.flag("codex", profile, source="quota")
        self.assertTrue(health.active("codex/gpt-5.6-sol", profile))
        self.assertTrue(health.active("codex/gpt-5.6-luna", profile))
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-sol", profile), "cooling")
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-luna", profile), "cooling")
        self.assertFalse(health.active("codex/gpt-5.6-sol", other))
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-sol", other), "healthy")

    def test_t3_model_429_on_sol_does_not_cool_luna(self):
        profile = self._profile()
        health.flag("codex/gpt-5.6-sol", profile, source="rate_limit")
        self.assertTrue(health.active("codex/gpt-5.6-sol", profile))
        self.assertFalse(health.active("codex/gpt-5.6-luna", profile))
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-luna", profile), "healthy")

    def test_t5_merge_cooling_beats_probing(self):
        profile = self._profile()
        now = 1_800_000_000.0
        until = now + 3600
        with mock.patch.object(health.time, "time", return_value=now):
            health.flag("codex/gpt-5.6-sol", profile, until=until,
                        source="rate_limit")
            health.flag("codex", profile, until=until, source="rate_limit")
        self._set_probe_after("codex/gpt-5.6-sol", profile, now - 1)
        self._set_probe_after("codex", profile, now + 60)
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-sol", profile, now=now),
            "cooling")
        self._set_probe_after("codex", profile, now - 1)
        self.assertEqual(
            health.seat_kind("codex/gpt-5.6-sol", profile, now=now),
            "probing")

    def test_t6_windows_flag_and_clear_independently(self):
        profile = self._profile()
        future = time.time() + 3600
        exhausted_both = {
            "used_percent": 100, "reached": True, "resets_at": future,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }
        health.apply_codex_quota("codex", profile, exhausted_both, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["model"], "codex")
        self.assertEqual(ledger[week]["source"], "quota")

        healthy_primary = {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }
        health.apply_codex_quota("codex", profile, healthy_primary, None)
        ledger = health._read_ledger_unlocked()
        self.assertFalse(health._is_active(ledger.get(acc)))
        self.assertTrue(health._is_active(ledger.get(week)))

        health.apply_codex_quota("codex", profile, exhausted_both, None)
        healthy_week = {
            "used_percent": 100, "reached": True, "resets_at": future,
            "window_mins": 300,
            "secondary": {
                "used_percent": 10, "resets_at": None,
                "window_mins": 10080,
            },
        }
        health.apply_codex_quota("codex", profile, healthy_week, None)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertFalse(health._is_active(ledger.get(week)))

    def test_t7_healthy_quota_does_not_clear_model_crash(self):
        profile = self._profile()
        health.flag("codex/sol", profile, source="crash")
        health.flag("codex", profile, source="quota")
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "window_mins": 300,
        }, None)
        crash_key = health.pool_key("codex/sol", profile)
        acc = health.pool_key("codex", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(crash_key)))
        self.assertEqual(ledger[crash_key]["source"], "crash")
        self.assertFalse(health._is_active(ledger.get(acc)))

    def test_t8_will_recheck_account_quota_not_model_429(self):
        profile = self._profile()
        health.flag("codex", profile, source="quota")
        acc = health.pool_key("codex", profile)
        self.assertTrue(health.snapshot()[acc]["will_recheck"])
        health.clear_all()
        health.flag("codex/gpt-5.6-sol", profile, source="quota")
        sol = health.pool_key("codex/gpt-5.6-sol", profile)
        self.assertFalse(health.snapshot()[sol]["will_recheck"])
        health.clear_all()
        health.flag("codex/gpt-5.6-sol", profile, source="rate_limit")
        sol = health.pool_key("codex/gpt-5.6-sol", profile)
        self.assertNotIn(sol, health._scheduled_probe_keys(profile))

    def test_present_nonweek_window_will_recheck(self):
        profile = self._profile()
        health.flag("codex#m300", profile, source="quota", record_model="codex")
        key = health.pool_key("codex#m300", profile)
        self.assertTrue(health.snapshot()[key]["will_recheck"])
        self.assertIn(key, health._scheduled_probe_keys(profile))

    def test_t10_read_rate_limits_parses_secondary(self):
        from providers.codex_quota import parse_rate_limits

        future = int(time.time()) + 10_000
        present = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": future,
                },
                "secondary": {
                    "usedPercent": 80,
                    "windowDurationMins": 10080,
                    "resetsAt": future,
                },
            }
        })
        self.assertEqual(present["secondary"]["used_percent"], 80.0)
        self.assertEqual(present["secondary"]["window_mins"], 10080)
        self.assertEqual(present["secondary"]["resets_at"], future)
        absent = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": future,
                },
            }
        })
        self.assertNotIn("secondary", absent)
        none_primary = parse_rate_limits({
            "rateLimits": {
                "primary": None,
                "secondary": {
                    "usedPercent": 90,
                    "windowDurationMins": 10080,
                    "resetsAt": future,
                },
            }
        })
        self.assertIsNone(none_primary["used_percent"])
        self.assertEqual(none_primary["secondary"]["used_percent"], 90.0)
        self.assertFalse(present["secondary"]["reached"])
        self.assertFalse(none_primary["secondary"]["reached"])
        self.assertFalse(present["primary_reached"])
        self.assertFalse(present["reached"])

    def test_stale_week_key_cleared_when_secondary_renamed_to_m1440(self):
        profile = self._profile()
        future = int(time.time()) + 3600
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }, None)
        week = health.pool_key("codex#week", profile)
        self.assertTrue(health._is_active(health._read_ledger_unlocked().get(week)))
        later = int(time.time()) + 7200
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": later,
                "window_mins": 1440,
            },
        }, None)
        ledger = health._read_ledger_unlocked()
        m1440 = health.pool_key("codex#m1440", profile)
        self.assertFalse(health._is_active(ledger.get(week)))
        self.assertNotIn(week, ledger)
        self.assertTrue(health._is_active(ledger.get(m1440)))
        self.assertEqual(ledger[m1440]["source"], "quota")
        self.assertEqual(ledger[m1440]["until"], later)

    def test_empty_rate_limits_leaves_account_and_week_untouched(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        future = int(time.time()) + 3600
        health.flag("codex", profile, until=future, source="quota")
        health.flag(
            "codex#week", profile, until=future,
            source="quota", record_model="codex")
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        health.apply_codex_quota(
            "codex", profile, parse_rate_limits({"rateLimits": {}}), None)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertEqual(ledger[week]["source"], "quota")

    def test_primary_only_five_hour_leaves_week_untouched(self):
        profile = self._profile()
        future = int(time.time()) + 3600
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }, None)
        week = health.pool_key("codex#week", profile)
        self.assertTrue(health._is_active(health._read_ledger_unlocked().get(week)))
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
        }, None)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], future)
        self.assertEqual(ledger[week]["source"], "quota")

    def test_window_authority_leaves_manual_and_model_records(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        future = int(time.time()) + 3600
        later = future + 60
        model_pool = "codex/gpt-5.6-sol"
        observations = (
            parse_rate_limits({"rateLimits": {}}),
            {
                "used_percent": 10, "reached": False, "resets_at": None,
                "window_mins": 300,
            },
            {
                "used_percent": 10, "reached": False, "resets_at": None,
                "window_mins": 300,
                "secondary": {
                    "used_percent": 100, "resets_at": later,
                    "window_mins": 1440,
                },
            },
        )
        for info in observations:
            health.clear_all()
            health.flag(
                "codex#week", profile, source="manual", record_model="codex")
            health.flag(model_pool, profile, source="quota")
            health.apply_codex_quota("codex", profile, info, None)
            ledger = health._read_ledger_unlocked()
            week = health.pool_key("codex#week", profile)
            model_key = health.pool_key(model_pool, profile)
            self.assertTrue(health._is_active(ledger.get(week)))
            self.assertEqual(ledger[week]["source"], "manual")
            self.assertTrue(health._is_active(ledger.get(model_key)))
            self.assertEqual(ledger[model_key]["source"], "quota")
        m1440 = health.pool_key("codex#m1440", profile)
        self.assertTrue(health._is_active(
            health._read_ledger_unlocked().get(m1440)))

    def test_weekly_primary_reached_rewrites_kept_week_until(self):
        profile = self._profile()
        now = int(time.time())
        old_until = now + 30 * 24 * 3600
        new_reset = now + 3600
        health.flag(
            "codex#week", profile, until=old_until,
            source="quota", record_model="codex")
        health.apply_codex_quota("codex", profile, {
            "used_percent": 100,
            "reached": True,
            "primary_reached": True,
            "resets_at": new_reset,
            "window_mins": 10080,
        }, None)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], new_reset)
        self.assertEqual(ledger[week]["source"], "quota")

    def test_invalid_secondary_window_mins_are_ignored(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        future = int(time.time()) + 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 100,
                    "windowDurationMins": 300,
                    "resetsAt": future,
                },
                "secondary": {
                    "usedPercent": 100,
                    "windowDurationMins": -1,
                    "resetsAt": future,
                },
            }
        })
        with mock.patch.object(health, "log_event") as logged:
            health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        bad = health.pool_key("codex#m-1", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertNotIn(bad, ledger)
        window_errors = [
            call for call in logged.call_args_list
            if call.args == ("health_error",)
            and call.kwargs.get("operation") == "quota_window"
        ]
        self.assertEqual(len(window_errors), 1)

    def test_unattributed_reached_flags_primary_and_logs_once(self):
        profile = self._profile()
        future = int(time.time()) + 3600
        info = {
            "reached": True,
            "primary_reached": False,
            "used_percent": 10,
            "resets_at": future,
            "window_mins": 300,
        }
        with mock.patch.object(health, "log_event") as logged:
            health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["source"], "quota")
        unattributed = [
            call for call in logged.call_args_list
            if call.args == ("health_error",)
            and call.kwargs.get("operation") == "quota_reached_unattributed"
        ]
        self.assertEqual(len(unattributed), 1)

    def test_manual_week_record_survives_quota_reconcile(self):
        profile = self._profile()
        health.flag("codex#week", profile, source="manual", record_model="codex")
        other = self._profile()
        health.flag("codex#week", other, source="quota", record_model="codex")
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "window_mins": 300,
        }, None)
        week = health.pool_key("codex#week", profile)
        other_week = health.pool_key("codex#week", other)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["source"], "manual")
        self.assertTrue(health._is_active(ledger.get(other_week)))
        self.assertEqual(ledger[other_week]["source"], "quota")

    def test_weekly_limit_reached_flags_week_key_not_five_hour_reset(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        primary_reset = int(time.time()) + 5 * 3600
        week_reset = int(time.time()) + 7 * 24 * 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": primary_reset,
                },
                "secondary": {
                    "windowDurationMins": 10080,
                    "resetsAt": week_reset,
                },
                "rateLimitReachedType": "weekly_limit_reached",
            }
        })
        self.assertTrue(info["reached"])
        self.assertFalse(info["primary_reached"])
        self.assertTrue(info["secondary"]["reached"])
        health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertFalse(health._is_active(ledger.get(acc)))
        self.assertNotIn(acc, ledger)
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], week_reset)
        self.assertEqual(ledger[week]["source"], "quota")

    def test_fake_vendor_weekly_primary_flags_account_keeps_week(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        week_reset = int(time.time()) + 7 * 24 * 3600
        health.flag(
            "codex#week", profile, until=week_reset,
            source="quota", record_model="codex")
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 10080,
                    "resetsAt": week_reset,
                },
                "rateLimitReachedType": "weekly_limit_reached",
            }
        })
        self.assertTrue(info["reached"])
        self.assertTrue(info["primary_reached"])
        self.assertNotIn("secondary", info)
        health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["until"], week_reset)
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["source"], "quota")

    def test_secondary_limit_reached_flags_weekly_secondary(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        primary_reset = int(time.time()) + 5 * 3600
        week_reset = int(time.time()) + 7 * 24 * 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": primary_reset,
                },
                "secondary": {
                    "usedPercent": 10,
                    "windowDurationMins": 10080,
                    "resetsAt": week_reset,
                },
                "rateLimitReachedType": "secondary_limit_reached",
            }
        })
        self.assertTrue(info["reached"])
        self.assertFalse(info["primary_reached"])
        self.assertTrue(info["secondary"]["reached"])
        health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertFalse(health._is_active(ledger.get(acc)))
        self.assertNotIn(acc, ledger)
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], week_reset)
        self.assertEqual(ledger[week]["source"], "quota")

    def test_unknown_reached_type_flags_primary(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        primary_reset = int(time.time()) + 5 * 3600
        week_reset = int(time.time()) + 7 * 24 * 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": primary_reset,
                },
                "secondary": {
                    "usedPercent": 10,
                    "windowDurationMins": 10080,
                    "resetsAt": week_reset,
                },
                "rateLimitReachedType": "unknown_limit",
            }
        })
        self.assertTrue(info["reached"])
        self.assertTrue(info["primary_reached"])
        self.assertFalse(info["secondary"]["reached"])
        health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["until"], primary_reset)
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertFalse(health._is_active(ledger.get(week)))
        self.assertNotIn(week, ledger)

    def test_apply_quota_model_claude_writes_claude_week_not_codex(self):
        profile = self._profile()
        future = int(time.time()) + 3600
        health.apply_codex_quota("claude", profile, {
            "used_percent": 10, "reached": False, "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080, "reached": True,
            },
        }, None)
        claude_week = health.pool_key("claude#week", profile)
        codex_week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(claude_week)))
        self.assertEqual(ledger[claude_week]["model"], "claude")
        self.assertEqual(ledger[claude_week]["until"], future)
        self.assertEqual(ledger[claude_week]["source"], "quota")
        self.assertNotIn(codex_week, ledger)

    def test_one_locked_mutate_per_quota_observation(self):
        profile = self._profile()
        future = time.time() + 3600
        exhausted_both = {
            "used_percent": 100, "reached": True, "resets_at": future,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }
        healthy_no_secondary = {
            "used_percent": 10, "reached": False, "window_mins": 300,
        }
        with mock.patch.object(
                health, "_locked_mutate", wraps=health._locked_mutate) as spy:
            health.apply_codex_quota("codex", profile, exhausted_both, None)
            self.assertEqual(spy.call_count, 1)
            spy.reset_mock()
            health.apply_codex_quota(
                "codex", profile, healthy_no_secondary, None)
            self.assertEqual(spy.call_count, 1)

    def test_healthy_observation_leaves_manual_account_record(self):
        profile = self._profile()
        health.flag("codex", profile, source="manual")
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "window_mins": 300,
        }, None)
        acc = health.pool_key("codex", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["source"], "manual")

    def test_expired_stale_week_quota_removed_when_secondary_absent(self):
        profile = self._profile()
        health.flag(
            "codex#week", profile, until=time.time() - 5,
            source="quota", record_model="codex")
        week = health.pool_key("codex#week", profile)
        self.assertIn(week, health._read_ledger_unlocked())
        self.assertFalse(
            health._is_active(health._read_ledger_unlocked().get(week)))
        health.apply_codex_quota("codex", profile, {
            "used_percent": 10, "reached": False, "window_mins": 300,
        }, None)
        self.assertNotIn(week, health._read_ledger_unlocked())

    def test_back_to_back_observations_keep_only_last_window_keys(self):
        profile = self._profile()
        future = time.time() + 3600
        health.apply_codex_quota("codex", profile, {
            "used_percent": 100, "reached": True, "resets_at": future,
            "secondary": {
                "used_percent": 100, "resets_at": future,
                "window_mins": 10080,
            },
        }, None)
        later = future + 60
        health.apply_codex_quota("codex", profile, {
            "used_percent": 100, "reached": True, "resets_at": later,
            "secondary": {
                "used_percent": 100, "resets_at": later,
                "window_mins": 1440,
            },
        }, None)
        ledger = health._read_ledger_unlocked()
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        m1440 = health.pool_key("codex#m1440", profile)
        self.assertEqual(set(ledger), {acc, m1440})
        self.assertNotIn(week, ledger)
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertTrue(health._is_active(ledger.get(m1440)))
        self.assertEqual(ledger[m1440]["until"], later)

    def test_lookalike_window_keys_ignored_by_covering_and_probe(self):
        profile = self._profile()
        future = time.time() + 3600
        lookalikes = ("codex#week/model", "codex#", "codex#weekly")
        for pool in lookalikes:
            health.flag(
                pool, profile, until=future, source="quota",
                record_model="codex")
        covering = health._covering_keys("codex/gpt-5.6-sol", profile)
        probe = health._scheduled_probe_keys(profile)
        for pool in lookalikes:
            key = health.pool_key(pool, profile)
            self.assertNotIn(key, covering)
            self.assertNotIn(key, probe)
        self.assertFalse(health.active("codex", profile))
        self.assertFalse(health.active("codex/gpt-5.6-sol", profile))
        health.flag(
            "codex#m300", profile, until=future, source="quota",
            record_model="codex")
        m300 = health.pool_key("codex#m300", profile)
        self.assertIn(m300, health._covering_keys("codex/gpt-5.6-sol", profile))
        self.assertIn(m300, health._scheduled_probe_keys(profile))
        self.assertTrue(health.active("codex/gpt-5.6-sol", profile))

    def test_older_quota_observation_is_noop(self):
        profile = self._profile()
        week_reset = int(time.time()) + 7 * 24 * 3600
        newer = {
            "checked_at": 200,
            "used_percent": 10,
            "reached": True,
            "primary_reached": False,
            "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100,
                "resets_at": week_reset,
                "window_mins": 10080,
                "reached": True,
            },
        }
        older = {
            "checked_at": 100,
            "used_percent": 10,
            "reached": False,
            "window_mins": 300,
        }
        health.apply_codex_quota("codex", profile, newer, None)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], week_reset)
        self.assertEqual(ledger[week]["observed_at"], 200)
        with mock.patch.object(health, "log_event") as logged:
            health.apply_codex_quota("codex", profile, older, None)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], week_reset)
        self.assertEqual(ledger[week]["source"], "quota")
        self.assertEqual(ledger[week]["observed_at"], 200)
        stale = [
            call for call in logged.call_args_list
            if call.kwargs.get("operation") == "quota_stale_observation"
        ]
        self.assertEqual(len(stale), 1)

    def test_newer_quota_observation_applies_after_older(self):
        profile = self._profile()
        week_reset = int(time.time()) + 7 * 24 * 3600
        older = {
            "checked_at": 100,
            "used_percent": 10,
            "reached": False,
            "window_mins": 300,
        }
        newer = {
            "checked_at": 200,
            "used_percent": 10,
            "reached": True,
            "primary_reached": False,
            "resets_at": None,
            "window_mins": 300,
            "secondary": {
                "used_percent": 100,
                "resets_at": week_reset,
                "window_mins": 10080,
                "reached": True,
            },
        }
        health.apply_codex_quota("codex", profile, older, None)
        acc = health.pool_key("codex", profile)
        self.assertFalse(
            health._is_active(health._read_ledger_unlocked().get(acc)))
        health.apply_codex_quota("codex", profile, newer, None)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(week)))
        self.assertEqual(ledger[week]["until"], week_reset)
        self.assertEqual(ledger[week]["observed_at"], 200)
        self.assertFalse(health._is_active(ledger.get(acc)))

    def test_secondary_reached_invalid_window_flags_primary(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        future = int(time.time()) + 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": future,
                },
                "secondary": {
                    "usedPercent": 10,
                    "windowDurationMins": -1,
                    "resetsAt": future,
                },
                "rateLimitReachedType": "secondary_limit_reached",
            }
        })
        self.assertTrue(info["reached"])
        self.assertFalse(info["primary_reached"])
        self.assertTrue(info["secondary"]["reached"])
        with mock.patch.object(health, "log_event") as logged:
            health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        bad = health.pool_key("codex#m-1", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertNotIn(bad, ledger)
        self.assertNotIn(week, ledger)
        unattributed = [
            call for call in logged.call_args_list
            if call.args == ("health_error",)
            and call.kwargs.get("operation") == "quota_reached_unattributed"
        ]
        self.assertEqual(len(unattributed), 1)

    def test_missing_secondary_duration_is_unnamed(self):
        from providers.codex_quota import parse_rate_limits

        profile = self._profile()
        future = int(time.time()) + 3600
        info = parse_rate_limits({
            "rateLimits": {
                "primary": {
                    "usedPercent": 10,
                    "windowDurationMins": 300,
                    "resetsAt": future,
                },
                "secondary": {
                    "usedPercent": 80,
                    "resetsAt": future,
                },
                "rateLimitReachedType": "primary_limit_reached",
            }
        })
        self.assertTrue(info["primary_reached"])
        self.assertIsNone(info["secondary"]["window_mins"])
        with mock.patch.object(health, "log_event") as logged:
            health.apply_codex_quota("codex", profile, info, None)
        acc = health.pool_key("codex", profile)
        week = health.pool_key("codex#week", profile)
        ledger = health._read_ledger_unlocked()
        self.assertTrue(health._is_active(ledger.get(acc)))
        self.assertEqual(ledger[acc]["source"], "quota")
        self.assertNotIn(week, ledger)
        window_errors = [
            call for call in logged.call_args_list
            if call.args == ("health_error",)
            and call.kwargs.get("operation") == "quota_window"
        ]
        self.assertEqual(len(window_errors), 1)

    def test_reached_type_naming_both_windows_attributes_both(self):
        from providers.codex_quota import parse_rate_limits, _attribute_reached

        future = int(time.time()) + 10_000
        types = (
            "primary_and_weekly_limit_reached",
            "secondary_and_5h",
            "hour_and_secondary",
        )
        for reached_type in types:
            info = parse_rate_limits({
                "rateLimits": {
                    "primary": {
                        "usedPercent": 10,
                        "windowDurationMins": 300,
                        "resetsAt": future,
                    },
                    "secondary": {
                        "usedPercent": 10,
                        "windowDurationMins": 10080,
                        "resetsAt": future,
                    },
                    "rateLimitReachedType": reached_type,
                }
            })
            self.assertTrue(info["primary_reached"], reached_type)
            self.assertTrue(info["secondary"]["reached"], reached_type)
        self.assertEqual(
            _attribute_reached("primary_and_weekly", 300, 10080),
            (True, True))


class VendorResetParserTest(unittest.TestCase):
    def _iso_z(self, ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")

    def test_json_resets_at_iso(self):
        now = 1_800_000_000.0
        expected = now + 3600
        iso = self._iso_z(expected)
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", json.dumps({"resets_at": iso}), "")
        self.assertEqual(ts, expected)
        with mock.patch.object(health.time, "time", return_value=now):
            ts_mut = health.vendor_reset_from_output(
                "claude", json.dumps({"resets_at": self._iso_z(expected + 1)}), "")
        self.assertEqual(ts_mut, expected + 1)

    def test_json_resetsAt_nested(self):
        now = 1_800_000_000.0
        expected = now + 6 * 3600
        iso = self._iso_z(expected)
        payload = json.dumps({"error": {"resetsAt": iso}})
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output("cursor", payload, "")
        self.assertEqual(ts, expected)

    def test_json_retry_after_seconds(self):
        now = 1_800_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", json.dumps({"retry_after": 90}), "")
        self.assertEqual(ts, now + 90)

    def test_json_retryAfter(self):
        now = 1_800_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", json.dumps({"retryAfter": 15}), "")
        self.assertEqual(ts, now + 15)

    def test_http_retry_after_seconds(self):
        now = 1_800_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "cursor", "", "Retry-After: 120")
        self.assertEqual(ts, now + 120)

    def test_text_resets_at_iso(self):
        now = 1_800_000_000.0
        expected = now + 90 * 60
        iso = self._iso_z(expected)
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", "rate limited; resets at %s" % iso, "")
        self.assertEqual(ts, expected)

    def test_text_resets_at_iso_offset_unchanged(self):
        now = 1_800_000_000.0
        iso = "2027-01-15T16:30:00+08:00"
        expected = now + 1800
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", "resets at %s" % iso, "")
        self.assertEqual(ts, expected)

    def test_text_resets_at_hhmm_uses_local_clock(self):
        now = 1_800_000_000.0
        local = datetime.fromtimestamp(now).astimezone()
        expected_dt = local.replace(hour=15, minute=4, second=0, microsecond=0)
        if expected_dt.timestamp() < now:
            expected_dt = expected_dt + timedelta(days=1)
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", "resets at 15:04", "")
        self.assertEqual(ts, expected_dt.timestamp())

    def test_text_try_again_minutes(self):
        now = 1_800_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "cursor", "try again in 5 minutes", "")
        self.assertEqual(ts, now + 5 * 60)

    def test_text_try_again_seconds(self):
        now = 1_800_000_000.0
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", "try again in 30 seconds", "")
        self.assertEqual(ts, now + 30)

    def test_window_rejects_zero_elapsed_far_future(self):
        now = 1_800_000_000.0
        elapsed = self._iso_z(now - 60)
        with mock.patch.object(health.time, "time", return_value=now):
            self.assertIsNone(health.vendor_reset_from_output(
                "cursor", "", "Retry-After: 0"))
            self.assertIsNone(health.vendor_reset_from_output(
                "claude", json.dumps({"resets_at": elapsed}), ""))
            self.assertIsNone(health.vendor_reset_from_output(
                "claude", "try again in 0 seconds", ""))
            self.assertIsNone(health.vendor_reset_from_output(
                "claude", json.dumps({"retry_after": 25 * 3600}), ""))

    def test_window_accepts_thirty_minutes_and_epoch_ms(self):
        now = 1_800_000_000.0
        ahead = now + 30 * 60
        with mock.patch.object(health.time, "time", return_value=now):
            ts = health.vendor_reset_from_output(
                "claude", json.dumps({"resets_at": ahead}), "")
            ts_ms = health.vendor_reset_from_output(
                "cursor", json.dumps({"resetsAt": int(ahead * 1000)}), "")
        self.assertEqual(ts, ahead)
        self.assertEqual(ts_ms, ahead)

    def test_garbage_is_none(self):
        self.assertIsNone(health.vendor_reset_from_output(
            "claude", "totally unrelated failure", "nope"))
        self.assertIsNone(health.vendor_reset_from_output("claude", "", ""))
        self.assertIsNone(health.vendor_reset_from_output(
            "claude", "{not json", "Retry-Later: 9"))


class VendorResetDispatchTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_claude_rate_limit_detail_sets_until(self):
        now = 1_800_000_000.0
        expected = now + 5 * 3600
        iso = datetime.fromtimestamp(expected, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")

        async def router(prompt, model, *a, **k):
            return {
                "ok": False,
                "reason": "rate_limit",
                "detail": "resets at %s" % iso,
                "model": model,
            }

        with mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router), \
                mock.patch.object(health.time, "time", return_value=now), \
                mock.patch.object(dispatch, "log_event") as logged:
            res = _run(dispatch.dispatch_single(
                "p", "claude/sonnet", 30, None, "a"))
        self.assertEqual(res["reason"], "rate_limit")
        key = health.pool_key("claude/sonnet", "a")
        entry = health.snapshot()[key]
        self.assertEqual(entry["until"], int(expected))
        self.assertEqual(entry["source"], "vendor_reset")
        self.assertEqual(entry["probe_after"], int(expected))
        self.assertEqual(
            health.seat_kind("claude/sonnet", "a", now=now + 10 * 60),
            "cooling")
        events = [c.args[0] for c in logged.call_args_list]
        self.assertIn("vendor_reset_applied", events)

    def test_unparsed_rate_limit_uses_default_cooldown(self):
        now = 1_800_000_000.0

        async def router(prompt, model, *a, **k):
            return {
                "ok": False,
                "reason": "rate_limit",
                "detail": "quota exceeded",
                "model": model,
            }

        with mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 0), \
                mock.patch.object(dispatch, "route_to_provider",
                                  side_effect=router), \
                mock.patch.object(health.time, "time", return_value=now), \
                mock.patch.object(dispatch, "log_event"):
            res = _run(dispatch.dispatch_single(
                "p", "claude/sonnet", 30, None, "default"))
        self.assertEqual(res["reason"], "rate_limit")
        key = health.pool_key("claude/sonnet", "default")
        entry = health.snapshot()[key]
        self.assertEqual(entry["until"], int(now + health.DEFAULT_COOLDOWN_S))
        self.assertEqual(entry["source"], "rate_limit")
        self.assertEqual(entry["probe_after"], now + health.HALF_OPEN_MIN_S)
        self.assertEqual(
            health.seat_kind("claude/sonnet", "default",
                             now=now + health.HALF_OPEN_MIN_S),
            "probing")


def _notify_calls(note, event):
    return [c for c in note.call_args_list if c.args and c.args[0] == event]


class NotifyEmitTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()
        os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

    def test_h1_emit_seat_nan_repeat_is_zero(self):
        with mock.patch.object(health, "notify") as note:
            health._emit_seat(
                "seat_flagged", "claude/opus", "default",
                "rate_limit", 1_800_000_600, float("nan"))
        self.assertEqual(len(note.call_args_list), 1)
        self.assertEqual(note.call_args.kwargs["repeat"], 0)

    def test_coerce_repeat_overflow_is_zero(self):
        self.assertEqual(health._coerce_repeat(10 ** 10000), 0)

    def test_flag_emits_seat_flagged_fields(self):
        until = 1_800_000_600
        with mock.patch.object(health, "notify") as note:
            health.flag("claude/opus", "premium", until=until, source="quota")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        fields = calls[0].kwargs
        self.assertEqual(fields["road"], "claude")
        self.assertEqual(fields["profile"], "premium")
        self.assertEqual(fields["model"], "claude/opus")
        self.assertEqual(fields["source"], "quota")
        self.assertEqual(fields["until"], until)
        self.assertEqual(fields["repeat"], 1)

    def test_flag_lock_skip_does_not_emit(self):
        with mock.patch.object(health, "_locked_mutate",
                               return_value=health._LOCK_SKIPPED), \
                mock.patch.object(health, "notify") as note:
            health.flag("claude/opus", "default", source="rate_limit")
        self.assertEqual(_notify_calls(note, "seat_flagged"), [])

    def test_clear_emits_seat_cleared_fields(self):
        health.flag("codex/gpt-5.6-sol", "alt", source="rate_limit")
        with mock.patch.object(health, "notify") as note:
            health.clear("codex/gpt-5.6-sol", "alt")
        calls = _notify_calls(note, "seat_cleared")
        self.assertEqual(len(calls), 1)
        fields = calls[0].kwargs
        self.assertEqual(fields["road"], "codex")
        self.assertEqual(fields["profile"], "alt")
        self.assertEqual(fields["model"], "codex/gpt-5.6-sol")
        self.assertEqual(fields["source"], "rate_limit")
        self.assertIn("until", fields)
        self.assertEqual(fields["repeat"], 1)

    def test_clear_lock_skip_raises_instead_of_reporting_the_entry(self):
        health.flag("claude/opus", "skip-clear", source="manual")
        with mock.patch.object(health, "_locked_mutate",
                               return_value=health._LOCK_SKIPPED), \
                mock.patch.object(health, "notify") as note:
            with self.assertRaises(health.HealthLockUnavailable):
                health.clear("claude/opus", "skip-clear")
        self.assertEqual(_notify_calls(note, "seat_cleared"), [])
        self.assertTrue(health.active("claude/opus", "skip-clear"))

    def test_clear_keep_sources_keeps_only_the_listed_source(self):
        health.flag("claude", "keep", source="credit_exhausted")
        self.assertIsNone(health.clear("claude", "keep", keep_sources=health.SUCCESS_KEEP_SOURCES))
        self.assertTrue(health.active("claude", "keep"))
        health.flag("claude", "keep2", source="rate_limit")
        self.assertIsNotNone(health.clear("claude", "keep2", keep_sources=health.SUCCESS_KEEP_SOURCES))
        self.assertIsNotNone(health.clear("claude", "keep"))

    def test_clear_models_names_the_road_only_for_road_strings(self):
        self.assertEqual(health.clear_models("claude"), ["claude"])
        self.assertEqual(health.clear_models("claude/m"), ["claude/m", "claude"])
        self.assertEqual(health.clear_models("gemini-x"), ["gemini-x"])

    def test_clear_missing_does_not_emit(self):
        with mock.patch.object(health, "notify") as note:
            health.clear("no-such-model", "default")
        self.assertEqual(_notify_calls(note, "seat_cleared"), [])

    def test_repeat_fourth_hit_within_hour(self):
        key_model = "cursor/sol"
        for _ in range(3):
            health.flag(key_model, "work", source="rate_limit")
            health.clear(key_model, "work")
        with mock.patch.object(health, "notify") as note:
            health.flag(key_model, "work", source="rate_limit")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["repeat"], 4)

    def test_repeat_window_file_survives_restart(self):
        health.flag("grok-4.6", "a", source="rate_limit")
        path = health._flag_window_path()
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        key = health.pool_key("grok-4.6", "a")
        self.assertIn(key, stored)
        self.assertEqual(len(stored[key]), 1)
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["OMNI_LLM_STATE_DIR"] = os.environ["OMNI_LLM_STATE_DIR"]
        env.pop("OMNI_LLM_NOTIFY_CMD", None)
        extra = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo if not extra else repo + os.pathsep + extra
        # Assert the SEAM's repeat in the fresh process, not the internal
        # helper: an implementation whose notify path read a separate
        # in-memory counter would satisfy the helper and still be wrong.
        script = (
            "from providers import health\n"
            "seen = []\n"
            "health.notify = lambda event, **f: seen.append((event, f.get('repeat')))\n"
            "health.flag(%r, %r, source='rate_limit')\n"
            "print(next(r for e, r in seen if e == 'seat_flagged'))\n"
        ) % ("grok-4.6", "a")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "2")
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(len(stored[key]), 2)

    def test_expiry_prune_emits_seat_cleared_once(self):
        now = 1_800_000_000.0
        until = int(now + 60)
        with mock.patch.object(health.time, "time", return_value=now):
            health.flag("claude/haiku", "exp-a", until=until, source="quota")
            # flagged twice, so the expiry event must carry repeat=2; asserting
            # repeat==1 here would pass against a hard-coded literal.
            health.flag("claude/haiku", "exp-a", until=until, source="quota")
            health.flag("codex/gpt-5.4", "exp-b", until=until, source="rate_limit")
        with mock.patch.object(health.time, "time", return_value=now + 120), \
                mock.patch.object(health, "notify") as note:
            health.maintain()
        calls = _notify_calls(note, "seat_cleared")
        self.assertEqual(len(calls), 2)
        by_model = {c.kwargs["model"]: c.kwargs for c in calls}
        claude = by_model["claude/haiku"]
        self.assertEqual(claude["road"], "claude")
        self.assertEqual(claude["profile"], "exp-a")
        self.assertEqual(claude["source"], "expired")
        self.assertEqual(claude["until"], until)
        self.assertEqual(claude["repeat"], 2)
        codex = by_model["codex/gpt-5.4"]
        self.assertEqual(codex["road"], "codex")
        self.assertEqual(codex["profile"], "exp-b")
        self.assertEqual(codex["source"], "expired")
        with mock.patch.object(health.time, "time", return_value=now + 180), \
                mock.patch.object(health, "notify") as note:
            health.maintain()
            health.active("claude/haiku", "exp-a")
            health.snapshot()
        self.assertEqual(_notify_calls(note, "seat_cleared"), [])
        self.assertEqual(note.call_args_list, [])

    def test_read_that_prunes_nothing_does_not_notify(self):
        until = int(time.time()) + 3600
        health.flag("pi/xai/grok-4.5", "live-a", until=until, source="rate_limit")
        with mock.patch.object(health, "notify") as note:
            self.assertTrue(health.active("pi/xai/grok-4.5", "live-a"))
            health.snapshot()
            health.seat_kind("pi/xai/grok-4.5", "live-a")
            health.maintain()
        self.assertEqual(note.call_args_list, [])

    def test_expired_unread_prune_does_not_notify(self):
        health.flag("agy/x", "idle-exp", until=time.time() - 5, source="rate_limit")
        with mock.patch.object(health, "notify") as note:
            self.assertFalse(health.active("agy/x", "idle-exp"))
            health.snapshot()
            health.seat_kind("agy/x", "idle-exp")
        self.assertEqual(note.call_args_list, [])

    def test_flag_survives_notify_sink_failure(self):
        os.environ["OMNI_LLM_NOTIFY_CMD"] = "/no/such/omni-notify-sink"
        try:
            health.flag("pi/xai/grok-4.6", "default", source="rate_limit")
        finally:
            os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
        self.assertTrue(health.active("pi/xai/grok-4.6", "default"))

    def test_note_failure_emits_seat_flagged_only_on_transition(self):
        model, profile = "claude/opus", "crash-a"
        # Prior flag/clear so the crash-flag repeat is not the default 1;
        # a hard-coded repeat=1 would pass if we asserted the first hit.
        health.flag(model, profile, source="rate_limit")
        health.clear(model, profile)
        with mock.patch.object(health, "notify") as note:
            health.note_failure(model, profile, "runtime_error")
            health.note_failure(model, profile, "runtime_error")
            self.assertEqual(_notify_calls(note, "seat_flagged"), [])
            health.note_failure(model, profile, "runtime_error")
            calls = _notify_calls(note, "seat_flagged")
            self.assertEqual(len(calls), 1)
            fields = calls[0].kwargs
            self.assertEqual(fields["road"], "claude")
            self.assertEqual(fields["profile"], profile)
            self.assertEqual(fields["model"], model)
            self.assertEqual(fields["source"], "crash")
            self.assertEqual(fields["repeat"], 2)
            self.assertGreater(fields["until"], time.time() + 250)
            self.assertLess(fields["until"], time.time() + 350)
            health.note_failure(model, profile, "runtime_error")
            self.assertEqual(len(_notify_calls(note, "seat_flagged")), 1)

    def test_note_failure_emits_when_crash_overwrites_other_source(self):
        model, profile = "claude/opus", "overwrite-a"
        health.note_failure(model, profile, "runtime_error")
        health.note_failure(model, profile, "runtime_error")
        health.flag(model, profile, source="rate_limit")
        with mock.patch.object(health, "notify") as note:
            health.note_failure(model, profile, "runtime_error")
            calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["source"], "crash")

    def test_note_failure_lock_skip_does_not_emit(self):
        for _ in range(2):
            health.note_failure("claude/opus", "skip-crash", "runtime_error")
        with mock.patch.object(health, "_locked_mutate",
                               return_value=health._LOCK_SKIPPED), \
                mock.patch.object(health, "notify") as note:
            health.note_failure("claude/opus", "skip-crash", "runtime_error")
        self.assertEqual(_notify_calls(note, "seat_flagged"), [])

    def test_clear_all_emits_seat_cleared_per_removed_entry(self):
        until_a = 1_800_000_600
        # Flag one seat twice so its clear_all event must carry repeat=2;
        # asserting repeat==1 here would pass against a hard-coded literal.
        health.flag("claude/haiku", "wipe-a", until=until_a, source="quota")
        health.flag("claude/haiku", "wipe-a", until=until_a, source="quota")
        health.flag("codex/gpt-5.4", "wipe-b", source="rate_limit")
        with mock.patch.object(health, "notify") as note:
            health.clear_all()
        calls = _notify_calls(note, "seat_cleared")
        self.assertEqual(len(calls), 2)
        by_model = {c.kwargs["model"]: c.kwargs for c in calls}
        claude = by_model["claude/haiku"]
        self.assertEqual(claude["road"], "claude")
        self.assertEqual(claude["profile"], "wipe-a")
        self.assertEqual(claude["source"], "quota")
        self.assertEqual(claude["until"], until_a)
        self.assertEqual(claude["repeat"], 2)
        codex = by_model["codex/gpt-5.4"]
        self.assertEqual(codex["road"], "codex")
        self.assertEqual(codex["profile"], "wipe-b")
        self.assertEqual(codex["source"], "rate_limit")
        self.assertFalse(health.active("claude/haiku", "wipe-a"))
        self.assertFalse(health.active("codex/gpt-5.4", "wipe-b"))

    def test_clear_all_already_empty_does_not_emit(self):
        health.flag("pi/xai/grok-4.5", "empty-wipe", source="rate_limit")
        health.clear_all()
        with mock.patch.object(health, "notify") as note:
            health.clear_all()
        self.assertEqual(note.call_args_list, [])

    def test_clear_all_lock_skip_does_not_emit(self):
        health.flag("claude/opus", "skip-wipe-a", source="quota")
        health.flag("codex/gpt-5.4", "skip-wipe-b", source="rate_limit")
        with mock.patch.object(health, "_locked_mutate",
                               return_value=health._LOCK_SKIPPED), \
                mock.patch.object(health, "notify") as note:
            health.clear_all()
        self.assertEqual(_notify_calls(note, "seat_cleared"), [])
        self.assertTrue(health.active("claude/opus", "skip-wipe-a"))
        self.assertTrue(health.active("codex/gpt-5.4", "skip-wipe-b"))


class AccountIdentityTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def test_flag_writes_account_from_resolver(self):
        with mock.patch.object(
                health, "_account_identity", return_value="abc@example"):
            health.flag("claude/opus", "premium", source="rate_limit")
        key = health.pool_key("claude/opus", "premium")
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored[key]["account"], "abc@example")
        self.assertEqual(health.snapshot()[key]["account"], "abc@example")

    def test_flag_account_none_when_resolver_returns_none(self):
        with mock.patch.object(health, "_account_identity", return_value=None):
            health.flag("claude/opus", "premium", source="rate_limit")
        key = health.pool_key("claude/opus", "premium")
        self.assertIsNone(health.snapshot()[key]["account"])

    def test_flag_survives_resolver_raise(self):
        def boom(*a, **k):
            raise RuntimeError("identity exploded")

        with mock.patch.object(health, "_account_identity", side_effect=boom):
            health.flag("claude/opus", "premium", source="rate_limit")
        key = health.pool_key("claude/opus", "premium")
        self.assertTrue(health.active("claude/opus", "premium"))
        self.assertIsNone(health.snapshot()[key]["account"])

    def test_snapshot_legacy_entry_without_account(self):
        until = int(time.time()) + 3600
        key = health.pool_key("claude/opus", "legacy")
        payload = {
            key: {
                "model": "claude/opus",
                "profile": "legacy",
                "until": until,
                "source": "rate_limit",
                "flagged_at": time.time(),
                "probe_after": time.time() + 300,
                "probe_every": 600,
            }
        }
        with open(health._ledger_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        snap = health.snapshot()
        self.assertIn(key, snap)
        self.assertIsNone(snap[key].get("account"))

    def test_flag_emit_seat_omits_account(self):
        with mock.patch.object(
                health, "_account_identity", return_value="abc@example"):
            until = 1_800_000_600
            with mock.patch.object(health, "notify") as note:
                health.flag(
                    "claude/opus", "premium", until=until, source="quota")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("account", calls[0].kwargs)
        key = health.pool_key("claude/opus", "premium")
        self.assertEqual(health.snapshot()[key]["account"], "abc@example")

    def test_mcp_snapshot_strips_account(self):
        with mock.patch.object(
                health, "_account_identity", return_value="abc@example"):
            health.flag("claude/opus", "premium", source="rate_limit")
        key = health.pool_key("claude/opus", "premium")
        self.assertEqual(health.snapshot()[key]["account"], "abc@example")
        mcp = dispatch._health_snapshot()
        self.assertIn(key, mcp)
        self.assertNotIn("account", mcp[key])
        wrapped = dispatch._with_health({"ok": True})
        self.assertNotIn("account", wrapped["health"][key])

    def test_account_identity_per_road(self):
        rows_by_road = {
            "claude": [{
                "name": "default",
                "identity": {
                    "emailAddress": "op@example.test",
                    "kind": "nope",
                },
            }],
            "codex": [{
                "name": "default",
                "identity": {
                    "account_id_suffix": "xyz789",
                    "emailAddress": "nope",
                },
            }],
            "cursor": [{
                "name": "default",
                "identity": {
                    "kind": "api-key",
                    "emailAddress": "nope",
                },
            }],
            "grok": [{
                "name": "default",
                "identity": {"emailAddress": "nope@example.test"},
            }],
        }

        def fake_list(roads=None):
            if isinstance(roads, str):
                return list(rows_by_road.get(roads, []))
            return []

        with mock.patch.object(
                health, "_account_identity", new=_real_account_identity), \
                mock.patch("providers.profile_check.list_profiles",
                           side_effect=fake_list):
            self.assertEqual(
                health._account_identity("claude/opus", "default"),
                "op@example.test")
            self.assertEqual(
                health._account_identity("codex/gpt-5.4", "default"),
                "xyz789")
            self.assertIsNone(
                health._account_identity("cursor/sol", "default"))
            self.assertIsNone(
                health._account_identity("grok-4.6", "default"))

    def test_flag_none_does_not_erase_account(self):
        model, profile = "claude/opus", "keep-flag"
        with mock.patch.object(
                health, "_account_identity", return_value="x@example"):
            health.flag(model, profile, source="rate_limit")
        with mock.patch.object(
                health, "_account_identity", return_value=None):
            health.flag(model, profile, source="rate_limit")
        key = health.pool_key(model, profile)
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored[key]["account"], "x@example")
        self.assertEqual(health.snapshot()[key]["account"], "x@example")

    def test_note_failure_none_does_not_erase_account(self):
        model, profile = "claude/opus", "keep-acct"
        with mock.patch.object(
                health, "_account_identity", return_value="abc@example"):
            entry = health.note_failure(model, profile, "runtime_error")
        self.assertEqual(entry["account"], "abc@example")
        with mock.patch.object(
                health, "_account_identity", return_value=None):
            later = health.note_failure(model, profile, "runtime_error")
        self.assertEqual(later["account"], "abc@example")
        key = health.pool_key(model, profile)
        with open(health._ledger_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored[key]["account"], "abc@example")

    def test_streak_only_keeps_account(self):
        kept = health._streak_only({
            "crash_streak": 2,
            "last_crash_at": 1.0,
            "model": "claude/opus",
            "profile": "default",
            "account": "abc@example",
            "until": 99,
            "source": "rate_limit",
        })
        self.assertEqual(kept["account"], "abc@example")
        self.assertEqual(kept["crash_streak"], 2)
        self.assertNotIn("until", kept)


class RoadForPoolTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()

    def _flag_codex_account(self, model, profile="p"):
        def fake_list(roads=None):
            if roads == "codex":
                return [{
                    "name": profile,
                    "identity": {"account_id_suffix": "xyz789"},
                }]
            return []

        with mock.patch.object(
                health, "_account_identity", new=_real_account_identity), \
                mock.patch("providers.profile_check.list_profiles",
                           side_effect=fake_list), \
                mock.patch.object(health, "notify") as note:
            health.flag(model, profile, source="quota")
        return note

    def test_quota_flag_on_codex_emits_codex_road_and_account(self):
        note = self._flag_codex_account("codex")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["road"], "codex")
        self.assertNotIn("window", calls[0].kwargs)
        key = health.pool_key("codex", "p")
        self.assertEqual(health.snapshot()[key]["account"], "xyz789")

    def test_quota_flag_on_codex_week_emits_codex_road_and_account(self):
        note = self._flag_codex_account("codex#week")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["road"], "codex")
        self.assertEqual(calls[0].kwargs["window"], "week")
        key = health.pool_key("codex#week", "p")
        self.assertEqual(health.snapshot()[key]["account"], "xyz789")
        with mock.patch.object(health, "notify") as note2:
            health.clear("codex#week", "p")
        cleared = _notify_calls(note2, "seat_cleared")
        self.assertEqual(len(cleared), 1)
        self.assertEqual(cleared[0].kwargs["window"], "week")

    def test_prefixed_codex_model_still_emits_codex_road(self):
        with mock.patch.object(health, "notify") as note:
            health.flag("codex/gpt-5.6-sol", "p", source="quota")
        calls = _notify_calls(note, "seat_flagged")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["road"], "codex")

    def test_bare_unknown_falls_through_to_provider_for_model(self):
        real = health.provider_for_model
        with mock.patch.object(
                health, "provider_for_model", wraps=real) as pf:
            got = health._road_for_pool("not-a-road")
        pf.assert_called_once_with("not-a-road")
        self.assertEqual(got, real("not-a-road"))


class RedirectedLedgerPathTest(unittest.TestCase):
    def test_snapshot_maintain_never_compute_default_state_dir(self):
        seen = []
        real = health._ledger_path

        def record():
            path = real()
            seen.append(path)
            return path

        with mock.patch.object(health, "_ledger_path", side_effect=record):
            health.snapshot()
            health.maintain()
        self.assertTrue(seen)
        default_root = os.path.abspath(
            os.path.expanduser(health.DEFAULT_STATE_DIR))
        prefix = default_root + os.sep
        for path in seen:
            resolved = os.path.abspath(path)
            self.assertNotEqual(resolved, default_root, resolved)
            self.assertFalse(resolved.startswith(prefix), resolved)


if __name__ == "__main__":
    unittest.main()
