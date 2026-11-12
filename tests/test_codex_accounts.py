#!/usr/bin/env python3
# no-opencc
"""Hermetic account failover contract tests; no live state or vendor calls."""
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from providers import codex_accounts as accounts, health
from tests import offline_home

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True

_HOME = None


def setUpModule():
    # Direct runs admit real home paths; keep protected_home() on a synthetic home.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _seed_codex(*models, profiles=("default", "a", "b", "c")):
    """Write a v2 record for the current codex homes. Call after the env the attempt fingerprints."""
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    if not models:
        models = ("synthetic-model",)
    items = [{
        "id": model,
        "name": model,
        "routable_id": "codex/%s" % model,
        "supported_efforts": ["minimal", "low", "medium", "high", "xhigh", "max"],
    } for model in models]
    record, _warnings = validate_strict("codex", {
        "provider": "codex",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": items,
    })
    for profile in profiles:
        catalog_v2.save("codex", profile, record)


def event_error(message=None, code=None):
    error = {}
    if message is not None:
        error["message"] = message
    if code is not None:
        error["code"] = code
    return json.dumps({"type": "turn.failed", "error": error})


def available(used=10, secondary=None):
    info = {"used_percent": used, "reached": False, "primary_reached": False,
            "window_mins": 300, "resets_at": None}
    if secondary is not None:
        info["secondary"] = {"used_percent": secondary, "reached": False,
                             "window_mins": 10080, "resets_at": None}
    return info


class AccountCase(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        temp = tempfile.TemporaryDirectory(prefix="omni-accounts-")
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        self.homes = {}
        for label in ("a", "b", "c"):
            path = os.path.join(self.root, label)
            os.mkdir(path)
            self.homes[label] = path
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_LLM_STATE_DIR": self.root,
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_NOTIFY_QUIET": "1"}
        env.update({"CODEX_HOME_" + k.upper(): v for k, v in self.homes.items()})
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        _seed_codex("synthetic-model", "synthetic-a", "synthetic-second")

    def selection(self, **kwargs):
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        return accounts.AccountSelection(**kwargs)

    def test_priority_and_single_visit(self):
        selection = self.selection()
        self.assertEqual(selection.next()["account_profile"], "a")
        self.assertEqual(selection.next()["account_profile"], "b")
        self.assertIsNone(selection.next())
        self.assertIsNone(selection.next())
        self.assertEqual(selection.exhausted()["reason"], "account_exhausted")

    def test_explicit_pin_ignores_ring_and_quota(self):
        selection = self.selection(profile="b")
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=AssertionError("probe")):
            self.assertEqual(selection.next()["account_profile"], "b")
        self.assertFalse(selection.managed)
        self.assertIsNone(selection.next())

    def test_no_ring_retains_default(self):
        selection = accounts.AccountSelection()
        self.assertFalse(selection.managed)
        self.assertEqual(selection.next()["account_profile"], "default")

    def test_disabled_probe_is_explicit_unverified_eligibility(self):
        selection = self.selection()
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=AssertionError("RPC disabled")):
            self.assertEqual(selection.next()["account_profile"], "a")

    def test_primary_and_secondary_exhaustion_never_spawn(self):
        selection = self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        async def forbidden(*args):
            self.fail("exhausted accounts spawned")
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=[(available(100), None), (available(10, 100), None)]):
            result = asyncio.run(accounts.run_selected(selection, forbidden))
        self.assertEqual(result["reason"], "account_exhausted")
        self.assertEqual([t["reason"] for t in result["tried"]], ["quota_exhausted"] * 2)

    def test_unknown_and_malformed_quota_never_healthy(self):
        for info, err in [(None, {"reason": "timeout"}), ({"reached": False}, None),
                          (available(float("nan")), None), (available(1, None) | {"secondary": {}}, None)]:
            with self.subTest(info=info):
                selection = self.selection()
                os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
                with mock.patch("providers.codex_quota.read_rate_limits", return_value=(info, err)):
                    self.assertIsNone(selection.next())
                self.assertEqual([t["reason"] for t in selection.tried], ["quota_unknown"] * 2)

    def test_quota_warning_cold_account_is_skipped_after_apply(self):
        selection = self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=[(available(95), None), (available(10), None)]):
            self.assertEqual(selection.next()["account_profile"], "b")
        self.assertEqual(selection.tried[0]["reason"], "skipped_cooldown")

    def test_cooling_is_never_used_as_last_resort(self):
        health.flag("codex", "a")
        health.flag("codex", "b")
        selection = self.selection()
        self.assertIsNone(selection.next())
        self.assertEqual([t["reason"] for t in selection.tried], ["skipped_cooldown"] * 2)

    def test_missing_and_duplicate_home_skipped(self):
        os.environ["CODEX_HOME_A"] = os.path.join(self.root, "absent")
        selection = self.selection()
        self.assertEqual(selection.next()["account_profile"], "b")
        self.assertEqual(selection.tried[0]["reason"], "account_unavailable")
        os.environ["CODEX_HOME_A"] = self.homes["b"]
        selection = self.selection()
        self.assertEqual(selection.next()["account_profile"], "a")
        self.assertIsNone(selection.next())
        self.assertEqual(selection.tried[-1]["reason"], "duplicate_account_home")

    def test_probe_receives_bound_deadline_and_home(self):
        selection = self.selection(timeout_s=2)
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch("providers.codex_quota.read_rate_limits", return_value=(available(), None)) as probe:
            selected = selection.next()
        self.assertEqual(selected["account_profile"], "a")
        self.assertLessEqual(probe.call_args.kwargs["timeout_s"], 2)
        self.assertEqual(probe.call_args.kwargs["codex_home"], self.homes["a"])

    def test_deadline_before_and_after_probe_never_spawns(self):
        selection = self.selection()
        selection.deadline = 0
        self.assertIsNone(selection.next())
        self.assertEqual(selection.exhausted()["reason"], "timeout")
        selection = self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        def delayed(*args, **kwargs):
            selection.deadline = 0
            return available(), None
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=delayed):
            self.assertIsNone(selection.next())
        self.assertEqual(selection.exhausted()["reason"], "timeout")

    def test_auto_uses_registry_path_without_env_or_auth_json(self):
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "auto"
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        os.environ.pop("CODEX_HOME_A")
        from providers.profile_check import DISCOVERY_ENV
        os.environ[DISCOVERY_ENV] = "1"
        row = {"road": "codex", "name": "a", "path": self.homes["a"],
               "state": "not_logged_in", "problems": [], "source": "discovered"}
        before = dict(os.environ)
        with mock.patch("providers.profile_check.list_profiles", return_value=[row]):
            selection = accounts.AccountSelection()
        self.assertEqual(selection.next()["codex_home"], self.homes["a"])
        self.assertEqual(dict(os.environ), before)

    def test_resume_pin_and_default_home_drift(self):
        selection = self.selection(profile="a", pinned_home=self.homes["b"], resume=True)
        self.assertIsNone(selection.next())
        self.assertEqual(selection.exhausted()["reason"], "account_home_drift")
        os.environ["CODEX_HOME"] = self.homes["a"]
        selection = accounts.AccountSelection(pinned_home=self.homes["b"], resume=True)
        self.assertIsNone(selection.next())
        self.assertEqual(selection.exhausted()["reason"], "account_home_drift")

    def test_default_pin_uses_protected_home_not_ambient_home(self):
        os.environ["HOME"] = self.homes["b"]
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=self.homes["a"]):
            home, err = accounts.profile_home("default")
        self.assertIsNone(err)
        self.assertEqual(home, os.path.join(self.homes["a"], ".codex"))

    def test_sibling_and_api_key_isolation_preserves_parent(self):
        from providers.profiles import spawn_env
        parent = {**os.environ, "CODEX_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"}
        env = accounts.account_env(spawn_env(parent, {"CODEX_HOME": self.homes["a"]}, scrub_siblings=True))
        self.assertEqual(env["CODEX_HOME"], self.homes["a"])
        self.assertFalse(any(k.startswith("CODEX_HOME_") for k in env))
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertIn("CODEX_HOME_B", parent)

    def test_startup_quota_and_auth_stop_and_next_request_uses_b(self):
        for stream in (event_error(code="quota_exceeded"), event_error("401 Unauthorized"),
                       event_error("You've hit your usage limit")):
            with self.subTest(stream=stream):
                health.clear_all()
                selection = self.selection()
                called = []
                async def attempt(account, remaining, managed):
                    called.append(account["account_profile"])
                    if len(called) == 1:
                        return accounts.annotate_failure({"ok": False, "reason": "runtime_error"}, stream)
                    return {"ok": True, "text": "synthetic reply"}
                result = asyncio.run(accounts.run_selected(selection, attempt))
                self.assertFalse(result["ok"])
                self.assertEqual(called, ["a"])
                self.assertEqual(result["account_profile"], "a")
                self.assertIs(result["account_retry_safe"], False)
                again = asyncio.run(accounts.run_selected(self.selection(), attempt))
                self.assertTrue(again["ok"])
                self.assertEqual(called, ["a", "b"])
                self.assertEqual(again["account_profile"], "b")

    def test_unsafe_failures_never_replay_and_next_request_skips_quota(self):
        streams = ["", "not-json", event_error(code="quota_exceeded") + '\n{',
                   json.dumps({"type": "new.future.event"}) + '\n' + event_error(code="quota_exceeded"),
                   json.dumps({"type": "item.started", "item": {"type": "command_execution"}}) + '\n' + event_error(code="quota_exceeded")]
        for stream in streams:
            with self.subTest(stream=stream):
                health.clear_all()
                selection = self.selection()
                called = []
                async def attempt(account, remaining, managed):
                    called.append(account["account_profile"])
                    return accounts.annotate_failure({"ok": False, "reason": "rate_limit"}, stream)
                result = asyncio.run(accounts.run_selected(selection, attempt))
                self.assertFalse(result["ok"])
                self.assertEqual(called, ["a"])
                self.assertEqual(self.selection().next()["account_profile"], "b")

    def test_timeout_runtime_fence_and_input_never_replay(self):
        for reason in ("timeout", "runtime_error", "fence_path_rejected", "unsupported_files"):
            selection = self.selection()
            called = []
            async def attempt(account, remaining, managed):
                called.append(account["account_profile"])
                return {"ok": False, "reason": reason, "account_retry_safe": True}
            result = asyncio.run(accounts.run_selected(selection, attempt))
            self.assertEqual(called, ["a"])
            self.assertEqual(result["reason"], reason)

    def test_resume_and_workspace_writes_forbid_replay(self):
        for kwargs, writes in [({"resume": True}, []), ({}, ["synthetic.txt"])]:
            selection = self.selection(candidates=["a", "b"], **kwargs)
            called = []
            async def attempt(account, remaining, managed):
                called.append(account["account_profile"])
                return {"ok": False, "reason": "rate_limit", "account_retry_safe": True,
                        "workspace_writes": writes}
            asyncio.run(accounts.run_selected(selection, attempt))
            self.assertEqual(len(called), 1)

    def test_quota_auth_failure_is_typed_and_sanitized(self):
        selection = self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=[
                (None, {"reason": "auth_error", "detail": "synthetic identity secret"}),
                (available(), None)]):
            self.assertEqual(selection.next()["account_profile"], "b")
        self.assertEqual(selection.tried, [{"account_profile": "a", "reason": "auth_error"}])

    def test_fake_quota_vendor_env_argv_ttl_and_home_remapping(self):
        from pathlib import Path
        from providers import codex_quota
        binary = Path(self.root) / "fake-codex"
        capture = Path(self.root) / "capture.jsonl"
        binary.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
with open(os.environ["OMNI_FAKE_CAPTURE"], "a") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\n")
for line in sys.stdin:
    message = json.loads(line)
    if message.get("id") == 1:
        print(json.dumps({"id": 1, "result": {}}), flush=True)
    if message.get("id") == 2:
        percent = 10 if os.path.basename(os.environ["CODEX_HOME"]) == "a" else 20
        print(json.dumps({"id": 2, "result": {"rateLimits": {"primary": {
            "usedPercent": percent, "windowDurationMins": 300}, "accountId": "synthetic-private"}}}), flush=True)
        break
''', encoding="utf-8")
        binary.chmod(0o700)
        # The quota child gets an allowlisted environment (#138): the fake
        # vendor's capture variable is passed by name.
        os.environ.update({"OMNI_LLM_QUOTA_PROBE": "1", "OMNI_FAKE_CAPTURE": str(capture),
                           "OMNI_LLM_CHILD_ENV_PASS": "OMNI_FAKE_CAPTURE",
                           "CODEX_API_KEY": "synthetic-key", "OPENAI_API_KEY": "synthetic-key"})
        before = dict(os.environ)
        with mock.patch.object(codex_quota.shutil, "which", return_value=str(binary)):
            first, err = codex_quota.read_rate_limits("a", codex_home=self.homes["a"], timeout_s=3)
            self.assertIsNone(err)
            cached, err = codex_quota.read_rate_limits("a", codex_home=self.homes["a"], timeout_s=3)
            self.assertIsNone(err)
            os.environ["CODEX_HOME_A"] = self.homes["b"]
            changed, err = codex_quota.read_rate_limits("a", timeout_s=3)
            self.assertIsNone(err)
        self.assertEqual(first["used_percent"], 10)
        self.assertEqual(cached["used_percent"], 10)
        self.assertEqual(changed["used_percent"], 20)
        captures = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(captures), 2)
        self.assertEqual([row["argv"] for row in captures], [["app-server"], ["app-server"]])
        self.assertEqual([row["env"]["CODEX_HOME"] for row in captures], [self.homes["a"], self.homes["b"]])
        for row in captures:
            self.assertFalse(any(k.startswith("CODEX_HOME_") for k in row["env"]))
            self.assertNotIn("CODEX_API_KEY", row["env"])
            self.assertNotIn("OPENAI_API_KEY", row["env"])
        self.assertNotIn("synthetic-private", json.dumps(first))
        self.assertNotIn(self.homes["a"], json.dumps(health.read_quota_cache()))
        self.assertEqual(os.environ["OPENAI_API_KEY"], before["OPENAI_API_KEY"])

    def test_quota_rpc_error_sanitizes_message_preserving_auth_reason(self):
        from providers.codex_quota import _rpc_error
        result = _rpc_error({"error": {"code": 401, "message": "synthetic@example.invalid Unauthorized"}})
        self.assertEqual(result["reason"], "auth_error")
        self.assertNotIn("synthetic@", json.dumps(result))

    def test_error_diagnostics_only_contain_validated_labels(self):
        selection = self.selection()
        selection.record("synthetic@example.invalid", "quota_unknown")
        self.assertEqual(selection.tried[-1]["account_profile"], "invalid")
        self.assertNotIn(self.root, json.dumps(selection.exhausted()))


class IntegrationCase(unittest.TestCase):
    setUp = AccountCase.setUp
    selection = AccountCase.selection
    def test_malformed_secondary_survives_rpc_cache_and_blocks_attempt(self):
        from providers import codex_quota
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        selection = self.selection()
        replies = [{"id": 1, "result": {}}, {"id": 2, "result": {"rateLimits": {
            "primary": {"usedPercent": 10, "windowDurationMins": 300}, "secondary": []}}}]
        def transport(profile, timeout, overrides):
            with mock.patch.object(codex_quota, "_write_msg", return_value=True), mock.patch.object(codex_quota, "_read_until_id", side_effect=[(reply, None) for reply in replies]):
                return codex_quota._rpc_session(mock.Mock(), timeout)
        async def forbidden(*args):
            self.fail("malformed secondary spawned an account")
        with mock.patch.object(codex_quota.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex_quota, "_spawn_rate_limits", side_effect=transport) as spawned:
            result = asyncio.run(accounts.run_selected(selection, forbidden))
            self.assertEqual(result["reason"], "account_exhausted")
            self.assertEqual([t["reason"] for t in result["tried"]], ["quota_unknown", "quota_unknown"])
            cached = self.selection()
            self.assertIsNone(cached.next())
            self.assertEqual(spawned.call_count, 2)

    def test_deadline_expires_in_health_check_never_calls_quota(self):
        selection = self.selection(timeout_s=1)
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        def slow_health(*args):
            selection.deadline = 0
            return "healthy"
        with mock.patch.object(health, "seat_kind", side_effect=slow_health), mock.patch("providers.codex_quota.read_rate_limits", return_value=(available(), None)) as quota:
            self.assertIsNone(selection.next())
            self.assertEqual(selection.exhausted()["reason"], "timeout")
            quota.assert_not_called()

    def test_non_string_event_type_is_fail_closed_without_exception(self):
        for kind in ({}, [], None, 1):
            health.clear_all()
            selection = self.selection()
            called = []
            stream = json.dumps({"type": kind}) + "\n" + event_error("401 Unauthorized")
            async def attempt(account, *args):
                called.append(account["account_profile"])
                return accounts.annotate_failure({"ok": False, "reason": "auth_error"}, stream)
            result = asyncio.run(accounts.run_selected(selection, attempt))
            self.assertFalse(result["account_retry_safe"])
            self.assertEqual(called, ["a"])

    def test_tool_activity_guard_rejects_even_an_erroneous_safe_annotation(self):
        selection = self.selection()
        called = []
        async def attempt(account, *args):
            called.append(account["account_profile"])
            return {"ok": False, "reason": "rate_limit", "account_retry_safe": True,
                    "tool_activity": True, "workspace_writes": 0}
        asyncio.run(accounts.run_selected(selection, attempt))
        self.assertEqual(called, ["a"])

    def test_discovered_job_resume_unchanged_mapping_and_path_drift(self):
        from providers import jobs
        from providers.profile_check import DISCOVERY_ENV
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "auto"
        os.environ[DISCOVERY_ENV] = "1"
        os.environ.pop("CODEX_HOME_A")
        row = {"road": "codex", "name": "a", "path": self.homes["a"],
               "state": "not_logged_in", "problems": []}
        with mock.patch("providers.profile_check.list_profiles", return_value=[row]):
            result = self.job_start()
            self.assertEqual(self.spawn_pin, ("a", self.homes["a"]))
            data = jobs._read_job(result["job_dir"])
            data.update(state="failed", session_id="synthetic-discovered-session")
            jobs._write_job(result["job_dir"], data)
            with mock.patch.object(jobs, "_workdir_has_other_live", return_value=False), mock.patch.object(jobs, "start", return_value={"job_id": "synthetic-child"}) as start:
                resumed = jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)
                self.assertEqual(resumed["job_id"], "synthetic-child")
                self.assertEqual(start.call_args.kwargs["codex_home"], self.homes["a"])
                unchanged = accounts.AccountSelection("a", pinned_home=self.homes["a"], resume=True)
                self.assertEqual(unchanged.next()["codex_home"], self.homes["a"])
                row["path"] = self.homes["b"]
                drifted = jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)
                self.assertEqual(drifted["refused"]["reason"], "account_home_drift")
                row["problems"] = ["collision"]
                missing = jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)
                self.assertEqual(missing["refused"]["reason"], "account_home_drift")
                self.assertEqual(start.call_count, 1)

    def fake_runner(self, failure_stream=None, returncode=1, timed_out=False):
        called = []
        async def run(argv, **kwargs):
            called.append((list(argv), dict(kwargs)))
            if len(called) == 1 and failure_stream is not None:
                return {"returncode": returncode, "stdout": failure_stream,
                        "stderr": "synthetic@example.invalid quota identity", "timed_out": timed_out}
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w") as handle:
                    handle.write("synthetic success")
            return {"returncode": 0, "stdout": json.dumps({"type": "item.completed",
                    "item": {"type": "agent_message", "text": "synthetic success"}}), "stderr": ""}
        return called, run

    def agent_patches(self, runner):
        from providers import agent_lane
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(agent_lane, "run_cli", side_effect=runner))
        stack.enter_context(mock.patch.object(agent_lane.shutil, "which", return_value="/synthetic/codex"))
        stack.enter_context(mock.patch.object(agent_lane, "_workdir_fence", return_value=None))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_snapshot", return_value={"entries": {}, "head": "synthetic"}))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)))
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=
                            lambda provider, argv, env, *args, **kwargs: (argv, env, "vendor", {}, None)))
        stack.enter_context(mock.patch.object(agent_lane.sandbox_canary, "gate", return_value=None))
        return stack

    def test_sync_agent_startup_failure_stops_and_next_request_preserves_model_effort_and_env(self):
        from providers import agent_lane
        self.selection()
        called, runner = self.fake_runner(event_error("401 Unauthorized"))
        with self.agent_patches(runner):
            stopped = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root,
                                                        effort="high", timeout_s=5, read_only=True))
            self.assertFalse(stopped["ok"])
            self.assertEqual(stopped["account_profile"], "a")
            self.assertEqual(len(called), 1)
            result = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root,
                                                       effort="high", timeout_s=5, read_only=True))
        self.assertTrue(result["ok"])
        self.assertEqual(result["account_profile"], "b")
        self.assertEqual([c[1]["env"]["CODEX_HOME"] for c in called], [self.homes["a"], self.homes["b"]])
        for argv, kwargs in called:
            self.assertEqual(argv[argv.index("-m") + 1], "synthetic-model")
            self.assertIn("model_reasoning_effort=high", argv)
            self.assertFalse(any(k.startswith("CODEX_HOME_") for k in kwargs["env"]))
        self.assertNotIn("synthetic@", json.dumps(result))

    def test_resume_backend_failure_never_switches_and_flags_original(self):
        from providers import agent_lane
        self.selection()
        called, runner = self.fake_runner(event_error("You've hit your usage limit"))
        with self.agent_patches(runner):
            result = asyncio.run(agent_lane.run_agent("synthetic continuation", "codex/synthetic-model", self.root,
                                                       account_profile="a", codex_home=self.homes["a"],
                                                       resume_session="synthetic-original-session", timeout_s=5))
        self.assertFalse(result["ok"])
        self.assertEqual(len(called), 1)
        argv, kwargs = called[0]
        self.assertEqual(argv[1:4], ["exec", "resume", "synthetic-original-session"])
        self.assertEqual(kwargs["env"]["CODEX_HOME"], self.homes["a"])
        self.assertEqual(self.selection().next()["account_profile"], "b")

    def test_sync_post_tool_quota_and_unknown_snapshot_stop(self):
        from providers import agent_lane
        tool_stream = json.dumps({"type": "item.started", "item": {"type": "command_execution"}}) + "\n" + event_error("You've hit your usage limit")
        for stream, writes in [(tool_stream, 0), (event_error("401 Unauthorized"), None)]:
            health.clear_all()
            self.selection()
            called, runner = self.fake_runner(stream)
            with self.agent_patches(runner), mock.patch.object(agent_lane, "_workspace_writes", return_value=(writes, "synthetic unknown")):
                result = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root, timeout_s=5))
            self.assertFalse(result["ok"])
            self.assertEqual(len(called), 1)

    def test_consult_real_fake_vendor_stops_then_next_request_switches_env_argv_and_no_key_leak(self):
        from pathlib import Path
        from providers import codex
        self.selection()
        binary = Path(self.root) / "fake-codex-exec"
        capture = Path(self.root) / "exec-capture.jsonl"
        binary.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
sys.stdin.read()
with open(os.environ["OMNI_FAKE_CAPTURE"], "a") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\n")
if os.path.basename(os.environ["CODEX_HOME"]) == "a":
    print(json.dumps({"type": "thread.started", "thread_id": "synthetic-a"}))
    print(json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}))
    sys.exit(1)
with open(sys.argv[sys.argv.index("-o") + 1], "w") as handle:
    handle.write("synthetic success")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "synthetic success"}}))
''', encoding="utf-8")
        binary.chmod(0o700)
        os.environ.update({"OMNI_FAKE_CAPTURE": str(capture), "CODEX_API_KEY": "synthetic-key", "OPENAI_API_KEY": "synthetic-key"})
        def prepared(argv, **kwargs):
            return argv, kwargs["env"], {"fence": "none", "fence_detail": {}}
        with mock.patch.object(codex.shutil, "which", return_value=str(binary)), mock.patch.object(codex, "prepare_host_fence", side_effect=prepared), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None):
            stopped = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5, effort="high"))
            self.assertFalse(stopped["ok"])
            self.assertEqual(stopped["account_profile"], "a")
            self.assertEqual(len(capture.read_text().splitlines()), 1)
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5, effort="high"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["account_profile"], "b")
        captures = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([c["env"]["CODEX_HOME"] for c in captures], [self.homes["a"], self.homes["b"]])
        for row in captures:
            self.assertIn("--json", row["argv"])
            self.assertIn("model_reasoning_effort=high", row["argv"])
            self.assertNotIn("OPENAI_API_KEY", row["env"])
            self.assertNotIn("CODEX_API_KEY", row["env"])
            self.assertFalse(any(k.startswith("CODEX_HOME_") for k in row["env"]))

    def test_managed_default_keeps_default_fence_admission(self):
        from providers import codex
        home = os.path.join(self.root, ".codex")
        os.mkdir(home)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "default"
        called, runner = self.fake_runner()
        def prepared(argv, **kwargs):
            self.assertIsNone(kwargs["config_dir"])
            return argv, kwargs["env"], {"fence": "none"}
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=self.root), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=prepared), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            _seed_codex(profiles=("default",))
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"])
        self.assertEqual(called[0][1]["env"]["CODEX_HOME"], home)

    def test_consult_policy_denies_sibling_homes_but_not_winner(self):
        from providers import codex
        protected = os.path.join(self.root, "protected")
        os.mkdir(protected)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        policies = []
        def prepared(argv, **kwargs):
            policies.append(kwargs["policy"])
            return argv, kwargs["env"], {"fence": "none"}
        called, runner = self.fake_runner()
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=protected), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=prepared), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["account_profile"], "a")
        self.assertEqual(len(policies), 1)
        wrap_kw = policies[0]["wrap_kw"]
        self.assertTrue(wrap_kw["read_wall"])
        denied = {path for _kind, path in wrap_kw["extra_deny_read"]}
        default = os.path.realpath(os.path.join(protected, ".codex"))
        # A named account is walled off from the default ~/.codex as well.
        self.assertEqual(denied, {self.homes["b"], self.homes["c"], default})
        self.assertNotIn(self.homes["a"], denied)
        self.assertIn(("subpath", default), wrap_kw["extra_deny_write"])

    def test_default_account_consult_policy_has_protected_config_final_denies(self):
        from providers import codex
        protected = os.path.join(self.root, "protected")
        default = os.path.realpath(os.path.join(protected, ".codex"))
        os.makedirs(default)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "default"
        policies = []
        def prepared(argv, **kwargs):
            self.assertIsNone(kwargs["config_dir"])
            policies.append(kwargs["policy"])
            return argv, kwargs["env"], {"fence": "none"}
        called, runner = self.fake_runner()
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=protected), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=prepared), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            _seed_codex()
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"], result)
        wrap_kw = policies[0]["wrap_kw"]
        finals = set(wrap_kw["extra_deny_write_final"])
        for name, kind in (("config.toml", "literal"), ("hooks.json", "literal"),
                           ("AGENTS.md", "literal"), ("AGENTS.override.md", "literal"),
                           ("agents", "subpath"), ("skills", "subpath"),
                           ("plugins", "subpath"), ("rules", "subpath")):
            self.assertIn((kind, os.path.join(default, name)), finals, name)
        # The home is write-denied whole and only the state set is re-allowed:
        # auth.json and sessions/ writable, hooks/ and custom layers not.
        self.assertIn(("subpath", default), set(wrap_kw["extra_deny_write"]))
        post = set(wrap_kw["extra_post_write"])
        self.assertIn(("literal", os.path.join(default, "auth.json")), post)
        self.assertIn(("subpath", os.path.join(default, "sessions")), post)
        for rel in ("hooks", "custom.config.toml", ".tmp"):
            self.assertFalse(any(path == os.path.join(default, rel) or path.startswith(os.path.join(default, rel) + os.sep) for _k, path in post), rel)
        self.assertFalse(any(path == os.path.join(default, "auth.json") for _k, path in finals))

    def test_default_account_consult_refuses_aliased_config(self):
        from providers import codex
        protected = os.path.join(self.root, "protected")
        default = os.path.realpath(os.path.join(protected, ".codex"))
        os.makedirs(os.path.join(default, "sessions"))
        with open(os.path.join(default, "sessions", "payload"), "w") as handle:
            handle.write("x")
        os.symlink(os.path.join(default, "sessions", "payload"), os.path.join(default, "AGENTS.md"))
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "default"
        called, runner = self.fake_runner()
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=protected), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence") as prepared, mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            _seed_codex()
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertEqual(result.get("reason"), "fence_path_rejected", result)
        prepared.assert_not_called()
        self.assertEqual(called, [])

    def _consult_policy(self, protected):
        from providers import codex
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        policies = []
        def prepared(argv, **kwargs):
            policies.append(kwargs["policy"])
            return argv, kwargs["env"], {"fence": "none"}
        called, runner = self.fake_runner()
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=protected), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=prepared), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(policies), 1)
        return policies[0]["wrap_kw"]

    def test_named_account_consult_walls_off_default_home_and_locks_own_config(self):
        protected = os.path.join(self.root, "protected")
        os.mkdir(protected)
        default = os.path.realpath(os.path.join(protected, ".codex"))
        os.mkdir(default)
        wrap_kw = self._consult_policy(protected)
        home = self.homes["a"]
        deny_write = set(wrap_kw["extra_deny_write"])
        # The default account tree: no read, no write.
        self.assertIn(("subpath", default), deny_write)
        self.assertIn(("subpath", default), set(wrap_kw["extra_deny_read"]))
        # The selected home: whole-home deny, then only vendor state back.
        self.assertIn(("subpath", home), deny_write)
        for name, kind in (("config.toml", "literal"), ("hooks.json", "literal"),
                           ("AGENTS.md", "literal"), ("AGENTS.override.md", "literal"),
                           ("agents", "subpath"), ("skills", "subpath"),
                           ("plugins", "subpath"), ("rules", "subpath")):
            self.assertIn((kind, os.path.join(home, name)), deny_write, name)
        post = {path for _kind, path in wrap_kw["extra_post_write"]}
        self.assertIn(os.path.join(home, "sessions"), post)
        self.assertIn(os.path.join(home, "log"), post)
        for name in ("config.toml", "hooks.json", "AGENTS.md", "AGENTS.override.md",
                     "agents", "skills", "plugins", "rules"):
            self.assertNotIn(os.path.join(home, name), post, name)
        self.assertTrue(all(path == home or path.startswith(home + os.sep) for path in post))

    def test_named_account_consult_refuses_a_planted_state_symlink(self):
        protected = os.path.join(self.root, "protected")
        os.mkdir(protected)
        other = os.path.join(self.root, "elsewhere")
        os.mkdir(other)
        os.symlink(other, os.path.join(self.homes["a"], "sessions"))
        from providers import codex
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        called, runner = self.fake_runner()
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=protected), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence") as prepared, mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner):
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
        self.assertEqual(result.get("reason"), "fence_path_rejected", result)
        prepared.assert_not_called()
        self.assertEqual(called, [])

    def host_fence_patches(self):
        from providers import host_fence, kernel_sandbox
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(host_fence.platform, "system", return_value="Darwin"))
        stack.enter_context(mock.patch.object(kernel_sandbox, "protected_home", return_value=self.root))
        stack.enter_context(mock.patch.object(kernel_sandbox, "available", return_value=True))
        stack.enter_context(mock.patch.object(kernel_sandbox, "verify_fence", return_value=None))
        stack.enter_context(mock.patch.object(kernel_sandbox, "wrap_argv", side_effect=lambda argv, *args, **kwargs: list(argv)))
        stack.enter_context(mock.patch.object(kernel_sandbox, "is_wrapped", return_value=True))
        stack.enter_context(mock.patch.object(kernel_sandbox, "disable_inner_codex_sandbox", side_effect=lambda argv: argv))
        stack.enter_context(mock.patch.object(kernel_sandbox, "inner_sandbox_disable_failed", return_value=False))
        return stack

    def test_managed_default_agent_actual_host_policy_keeps_default_admission(self):
        from providers import agent_lane, kernel_sandbox
        default_home = os.path.join(self.root, ".codex")
        os.mkdir(default_home)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "default"
        called, runner = self.fake_runner()
        def policy(provider, **kwargs):
            self.assertIsNone(kwargs["config_dir"])
            return {"ok": True, "vendor_state_writable": True}
        with self.host_fence_patches(), mock.patch.object(agent_lane, "run_cli", side_effect=runner), mock.patch.object(agent_lane.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(agent_lane, "_workdir_fence", return_value=None), mock.patch.object(agent_lane, "_workspace_snapshot", return_value={}), mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)), mock.patch.object(kernel_sandbox, "agent_fence_policy", side_effect=policy) as admission, mock.patch.object(kernel_sandbox, "wrap_kwargs_from_policy", return_value={"read_wall": True}):
            _seed_codex(profiles=("default",))
            result = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root,
                                                       read_only=True, timeout_s=5))
        self.assertTrue(result["ok"])
        self.assertEqual(admission.call_count, 1)
        self.assertEqual(called[0][1]["env"]["CODEX_HOME"], default_home)

    def test_managed_default_consult_actual_host_fence_restores_pinned_home(self):
        from providers import codex, kernel_sandbox, host_fence
        default_home = os.path.join(self.root, ".codex")
        os.mkdir(default_home)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "default"
        called, runner = self.fake_runner()
        with self.host_fence_patches(), mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "run_cli", side_effect=runner), mock.patch.object(kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "prepare_host_fence", wraps=host_fence.prepare_host_fence) as prepared:
            _seed_codex(profiles=("default",))
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"])
        self.assertIsNone(prepared.call_args.kwargs["config_dir"])
        self.assertNotIn("CODEX_HOME", prepared.call_args.kwargs["env"])
        self.assertEqual(called[0][1]["env"]["CODEX_HOME"], default_home)

    def test_dispatch_never_generic_retries_codex_rate_limit(self):
        from providers import dispatch
        async def failure(*args, **kwargs):
            return {"ok": False, "reason": "rate_limit", "tool_activity": True}
        with mock.patch.object(dispatch, "route_to_provider", side_effect=failure) as routed, mock.patch.object(dispatch, "_flag_rate_limit"), mock.patch.object(dispatch, "_health_note_failure"):
            result = asyncio.run(dispatch.dispatch_single("synthetic task", "codex/synthetic-model", 5, [], "default"))
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(routed.call_count, 1)

    def test_no_ring_startup_failure_ends_the_request_without_backoff_retry(self):
        from providers import dispatch
        calls = []
        async def outcome(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return accounts.annotate_failure({"ok": False, "reason": "rate_limit"}, event_error("You've hit your usage limit"))
            return {"ok": True, "text": "synthetic success"}
        sleep = mock.AsyncMock()
        with mock.patch.object(dispatch, "route_to_provider", side_effect=outcome), mock.patch.object(dispatch.asyncio, "sleep", new=sleep), mock.patch.object(dispatch, "_flag_rate_limit"), mock.patch.object(dispatch, "_health_clear"):
            result = asyncio.run(dispatch.dispatch_single("synthetic task", "codex/synthetic-model", 5, [], "default"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_no_ring_success_keeps_original_env_and_argv(self):
        from providers import codex
        os.environ["CODEX_API_KEY"] = "synthetic-key"
        called, runner = self.fake_runner()
        with mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", side_effect=runner), mock.patch("providers.codex_quota.read_rate_limits") as quota:
            result = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(result["ok"])
        self.assertNotIn("account_profile", result)
        self.assertEqual(called[0][1]["env"]["CODEX_API_KEY"], "synthetic-key")
        self.assertIn("CODEX_HOME_B", called[0][1]["env"])
        self.assertIn("--ephemeral", called[0][0])
        quota.assert_not_called()

    def test_no_ring_pinned_default_job_preserves_sibling_and_api_env(self):
        from providers import agent_lane
        os.environ["CODEX_HOME"] = self.homes["a"]
        os.environ["OPENAI_API_KEY"] = "synthetic-key"
        called, runner = self.fake_runner()
        with self.agent_patches(runner):
            result = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root,
                                                       codex_home=self.homes["a"], timeout_s=5))
        self.assertTrue(result["ok"])
        env = called[0][1]["env"]
        self.assertEqual(env["CODEX_HOME"], self.homes["a"])
        self.assertEqual(env["CODEX_HOME_B"], self.homes["b"])
        self.assertEqual(env["OPENAI_API_KEY"], "synthetic-key")

    def test_detached_deadline_expires_before_backend_without_spawn(self):
        from providers import jobs, job_runner
        self.selection()
        started = self.job_start()
        job = jobs._read_job(started["job_dir"])
        job["codex_deadline"] = 0
        with mock.patch.object(job_runner, "run_agent", new=mock.AsyncMock()) as agent:
            result = asyncio.run(job_runner._execute(started["job_dir"], job))
        self.assertEqual(result["reason"], "timeout")
        agent.assert_not_called()

    def test_persisted_deadline_includes_supervisor_delay_across_clock_reset(self):
        from providers import jobs, job_runner
        self.selection()
        started = self.job_start()
        job = jobs._read_job(started["job_dir"])
        import time
        self.assertGreater(job["codex_deadline"], time.time())
        self.assertLessEqual(job["codex_deadline"], time.time() + 5)
        with mock.patch.object(job_runner.time, "time", return_value=job["codex_deadline"] + 1), mock.patch.object(job_runner, "run_agent", new=mock.AsyncMock()) as agent:
            result = asyncio.run(job_runner._execute(started["job_dir"], job))
        self.assertEqual(result["reason"], "timeout")
        agent.assert_not_called()

    def test_seed_account_cooling_during_supervisor_delay_skips_once(self):
        selection = self.selection(candidates=["b"], profile="a", pinned_home=self.homes["a"],
                                   initial_account={"account_profile": "a", "codex_home": self.homes["a"]})
        health.flag("codex", "a")
        called = []
        async def attempt(account, *args):
            called.append(account["account_profile"])
            return {"ok": True, "text": "synthetic success"}
        result = asyncio.run(accounts.run_selected(selection, attempt))
        self.assertEqual(called, ["b"])
        self.assertEqual(result["tried"], [{"account_profile": "a", "reason": "skipped_cooldown"}])
        pinned = accounts.AccountSelection("a", pinned_home=self.homes["a"], resume=True)
        self.assertEqual(pinned.next()["account_profile"], "a")

    def test_exhausted_ring_is_not_repeated_across_model_chain(self):
        from providers import dispatch
        self.selection()
        called = []
        async def outcome(prompt, model, *args, **kwargs):
            called.append(model)
            if model.startswith("codex/"):
                return {"ok": False, "reason": "account_exhausted", "tried": []}
            return {"ok": True, "text": "synthetic fallback"}
        with mock.patch.object(dispatch, "dispatch_single", side_effect=outcome), mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)), mock.patch.object(dispatch, "_health_maintain"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-a,codex/synthetic-b,claude/synthetic-c", 5, [], "default"))
        self.assertTrue(result["ok"])
        self.assertEqual(called, ["codex/synthetic-a", "claude/synthetic-c"])

    def test_real_wrapper_expired_selector_stops_chain_before_second_probe_or_spawn(self):
        from providers import codex, dispatch, codex_quota
        self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        clock = [10.0]
        probes = []
        def delayed(profile, **kwargs):
            probes.append(profile)
            clock[0] = 12.0  # one request's 1s budget expires inside the quota seam
            return available(), None
        with mock.patch.object(accounts.time, "monotonic", side_effect=lambda: clock[0]), mock.patch.object(codex_quota, "read_rate_limits", side_effect=delayed), mock.patch.object(codex, "_run_codex_once", new=mock.AsyncMock(return_value={"ok": True, "text": "synthetic success", "model": "synthetic-two"})) as backend, mock.patch.object(dispatch, "_health_maintain"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-one,codex/synthetic-two", 1, [], "default"))
        self.assertEqual(probes, ["a"])
        backend.assert_not_called()
        self.assertEqual(result.get("reason"), "timeout")
        self.assertIs(result.get("account_retry_safe"), False)

    def test_selection_terminal_refusals_are_never_chain_replay_safe(self):
        from providers import codex, agent_lane
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,,b"
        with mock.patch.object(codex, "_run_codex_once", new=mock.AsyncMock()) as consult, mock.patch.object(agent_lane, "_run_agent_once", new=mock.AsyncMock()) as agent:
            results = [asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5)),
                       asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root, timeout_s=5))]
        self.assertEqual([result["reason"] for result in results], ["profile_config"] * 2)
        self.assertEqual([result["account_retry_safe"] for result in results], [False] * 2)
        consult.assert_not_called()
        agent.assert_not_called()
        selection = self.selection()
        attempt = mock.AsyncMock()
        def expire_after_selection(account, selected):
            selected.deadline = 0
        result = asyncio.run(accounts.run_selected(selection, attempt, expire_after_selection))
        self.assertEqual(result["reason"], "timeout")
        self.assertFalse(result["account_retry_safe"])
        attempt.assert_not_called()

    def test_real_wrapper_exhausted_ring_still_allows_other_provider_fallback(self):
        from providers import codex, dispatch, codex_quota
        self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch.object(codex_quota, "read_rate_limits", return_value=(None, {"reason": "timeout"})) as probe, mock.patch.object(codex, "_run_codex_once", new=mock.AsyncMock()) as backend, mock.patch.object(dispatch.claude_mod, "run_claude_subprocess", new=mock.AsyncMock(return_value={"ok": True, "text": "synthetic fallback"})) as fallback, mock.patch.object(dispatch, "_health_maintain"), mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-one,codex/synthetic-two,claude/synthetic-three", 5, [], "default"))
        self.assertTrue(result["ok"])
        self.assertEqual(probe.call_count, 2)  # once for a, once for b; no second ring
        backend.assert_not_called()
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual([entry for entry in result["tried"] if "model" not in entry],
                         [{"account_profile": label, "reason": "quota_unknown"} for label in ("a", "b")])

    def test_chain_terminal_exhaustion_preserves_account_reasons(self):
        from providers import codex, dispatch, codex_quota
        self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch.object(codex_quota, "read_rate_limits", return_value=(None, {"reason": "auth_error"})), mock.patch.object(codex, "_run_codex_once", new=mock.AsyncMock()) as backend, mock.patch.object(dispatch, "_health_maintain"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-one,codex/synthetic-two", 5, [], "default"))
        self.assertEqual(result["reason"], "account_exhausted")
        self.assertEqual([entry for entry in result["tried"] if "model" not in entry],
                         [{"account_profile": label, "reason": "auth_error"} for label in ("a", "b")])
        backend.assert_not_called()

    def test_chain_stops_at_codex_startup_failure_and_preserves_inner_account_and_outer_seat_reasons(self):
        from providers import codex, dispatch
        self.selection()
        outcomes = [{"ok": False, "reason": "rate_limit", "account_retry_safe": True},
                    {"ok": True, "text": "synthetic success"}]
        backend = mock.AsyncMock(side_effect=outcomes)
        with mock.patch.object(codex, "_run_codex_once", new=backend), mock.patch.object(dispatch.claude_mod, "run_claude_subprocess", new=mock.AsyncMock(return_value={"ok": False, "reason": "runtime_error"})), mock.patch.object(dispatch, "_health_maintain"), mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "claude/synthetic-first,codex/synthetic-second", 5, [], "default"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(result["account_profile"], "a")
        self.assertEqual(backend.call_count, 1)
        self.assertIn({"account_profile": "a", "reason": "rate_limit"}, result["tried"])
        self.assertTrue(any(entry.get("model") == "claude/synthetic-first" and entry.get("reason") == "runtime_error" for entry in result["tried"]))

    def test_direct_sync_entry_malformed_ring_never_spawns(self):
        from providers import codex, agent_lane
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,,b"
        with mock.patch.object(codex, "_run_codex_once", new=mock.AsyncMock()) as consult, mock.patch.object(agent_lane, "_run_agent_once", new=mock.AsyncMock()) as agent:
            c = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
            a = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root, timeout_s=5))
        self.assertEqual(c["reason"], "profile_config")
        self.assertEqual(a["reason"], "profile_config")
        consult.assert_not_called()
        agent.assert_not_called()

    def test_no_ring_unknown_backend_failure_through_real_wrapper_stops_chain(self):
        from providers import codex, dispatch
        outcome = {"returncode": 0, "stdout": "", "stderr": ""}
        with mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), mock.patch.object(codex, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})), mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None), mock.patch.object(codex, "run_cli", new=mock.AsyncMock(return_value=outcome)) as cli, mock.patch.object(dispatch.claude_mod, "run_claude_subprocess", new=mock.AsyncMock(return_value={"ok": True, "text": "synthetic fallback"})) as fallback, mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)), mock.patch.object(dispatch, "_health_maintain"), mock.patch.object(dispatch, "_health_note_failure"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-a,claude/synthetic-b", 5, [], "default"))
        self.assertEqual(result["reason"], "empty_response")
        self.assertFalse(result["account_retry_safe"])
        self.assertEqual(cli.call_count, 1)
        fallback.assert_not_called()

    def test_dispatch_chain_stops_after_codex_tool_failure(self):
        from providers import dispatch
        self.selection()
        async def failure(*args, **kwargs):
            return {"ok": False, "reason": "rate_limit", "tool_activity": True,
                    "account_retry_safe": False, "account_profile": "a"}
        with mock.patch.object(dispatch, "dispatch_single", side_effect=failure) as routed, mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)), mock.patch.object(dispatch, "_health_maintain"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", "codex/synthetic-model,claude/synthetic-model", 5, [], "default"))
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(routed.call_count, 1)

    def job_start(self, profile="default", resume_session=None, codex_home=None, codex_auth_isolated=None):
        from providers import jobs
        def spawn(argv, **kwargs):
            data = jobs._read_job(argv[-1])
            self.spawn_pin = (data.get("account_profile"), data.get("codex_home"))
            return mock.Mock(pid=99999999)
        with mock.patch.object(jobs, "_preflight", return_value=(None, self.root, {"provider": "codex", "model": "synthetic-model"})), mock.patch.object(jobs, "_drift_pin", return_value={"repo_root": self.root, "branch": "synthetic-work", "head": "synthetic-head"}), mock.patch.object(jobs.subprocess, "Popen", side_effect=spawn), mock.patch.object(jobs, "_record_supervisor", side_effect=lambda data, proc: data), mock.patch.object(jobs, "_wait_for_supervisor_lock"), mock.patch.object(jobs.threading, "Thread"):
            return jobs.start("synthetic task", "codex/synthetic-model", self.root, timeout_s=5,
                              account_profile=profile, resume_session=resume_session, codex_home=codex_home,
                              codex_auth_isolated=codex_auth_isolated)

    def test_job_persists_selected_account_before_supervisor_spawn(self):
        from providers import jobs
        self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch("providers.codex_quota.read_rate_limits", side_effect=[(available(100), None), (available(), None)]):
            result = self.job_start()
        self.assertNotIn("refused", result)
        self.assertEqual(self.spawn_pin, ("b", self.homes["b"]))
        data = jobs._read_job(result["job_dir"])
        self.assertEqual(data["codex_home"], self.homes["b"])
        self.assertEqual(data["account_profile"], "b")

    def test_exhausted_job_never_supervisor_spawns(self):
        self.selection()
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        with mock.patch("providers.codex_quota.read_rate_limits", return_value=(available(100), None)):
            result = self.job_start()
        self.assertEqual(result["refused"]["reason"], "account_exhausted")
        self.assertFalse(hasattr(self, "spawn_pin"))

    def test_resume_pins_original_account_and_refuses_mapping_drift(self):
        from providers import jobs
        self.selection()
        result = self.job_start(profile="b")
        data = jobs._read_job(result["job_dir"])
        data.update(state="failed", session_id="synthetic-session")
        jobs._write_job(result["job_dir"], data)
        with mock.patch.object(jobs, "_workdir_has_other_live", return_value=False), mock.patch.object(jobs, "start", return_value={"job_id": "synthetic-child"}) as start:
            resumed = jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)
            self.assertEqual(resumed["job_id"], "synthetic-child")
            self.assertEqual(start.call_args.kwargs["account_profile"], "b")
            self.assertEqual(start.call_args.kwargs["codex_home"], self.homes["b"])
            os.environ["CODEX_HOME_B"] = self.homes["c"]
            drifted = jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)
            self.assertEqual(drifted["refused"]["reason"], "account_home_drift")
            self.assertEqual(start.call_count, 1)

    def test_resume_default_home_drift_and_legacy_pin_refused(self):
        from providers import jobs
        os.environ["CODEX_HOME"] = self.homes["a"]
        result = self.job_start()
        data = jobs._read_job(result["job_dir"])
        data.update(state="failed", session_id="synthetic-session")
        jobs._write_job(result["job_dir"], data)
        with mock.patch.object(jobs, "_workdir_has_other_live", return_value=False), mock.patch.object(jobs, "start") as start:
            os.environ["CODEX_HOME"] = self.homes["b"]
            self.assertEqual(jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)["refused"]["reason"], "account_home_drift")
            data.pop("codex_home")
            jobs._write_job(result["job_dir"], data)
            self.assertEqual(jobs.resume(result["job_dir"], "synthetic continuation", allow_drift=True)["refused"]["reason"], "account_home_drift")
            start.assert_not_called()

    def subscription_vendor(self):
        from pathlib import Path
        binary = Path(self.root) / "fake-subscription-vendor"
        capture = Path(self.root) / "subscription-env.jsonl"
        binary.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
sys.stdin.read()
with open(os.environ["OMNI_SYNTHETIC_CAPTURE"], "a") as handle:
    handle.write(json.dumps({"home": os.environ.get("CODEX_HOME"), "argv": sys.argv[1:],
        "keys": [key for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE") if key in os.environ]}) + "\n")
print(json.dumps({"type": "thread.started", "thread_id": "synthetic-subscription-session"}))
print(json.dumps({"type": "item.started", "item": {"type": "command_execution"}}))
print(json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}))
sys.exit(1)
''', encoding="utf-8")
        binary.chmod(0o700)
        os.environ.update(OMNI_SYNTHETIC_CAPTURE=str(capture), CODEX_API_KEY="synthetic-key",
                          OPENAI_API_KEY="synthetic-key", OPENAI_API_BASE="https://synthetic.invalid")
        return binary, capture

    def real_vendor_patches(self, binary):
        from providers import agent_lane, job_runner
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(agent_lane, "resolve_agent_binary", return_value=str(binary)))
        stack.enter_context(mock.patch.object(agent_lane, "_workdir_fence", return_value=None))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_snapshot", return_value={"state": "ok", "entries": {}, "head": None}))
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=
                            lambda provider, argv, env, *args, **kwargs: (argv, env, "vendor", {}, None)))
        stack.enter_context(mock.patch.object(agent_lane.sandbox_canary, "gate", return_value=None))
        stack.enter_context(mock.patch.object(job_runner, "read_successor_attest", return_value=None))
        return stack

    def resume_without_supervisor(self, job_dir):
        from providers import jobs
        with mock.patch.object(jobs, "_preflight", return_value=(None, self.root, {"provider": "codex", "model": "synthetic-model"})), mock.patch.object(jobs, "_drift_pin", return_value={"repo_root": self.root, "branch": "synthetic-work", "head": "synthetic-head"}), mock.patch.object(jobs.subprocess, "Popen", return_value=mock.Mock(pid=99999999)), mock.patch.object(jobs, "_record_supervisor", side_effect=lambda data, proc: data), mock.patch.object(jobs, "_wait_for_supervisor_lock"), mock.patch.object(jobs.threading, "Thread"), mock.patch.object(jobs, "_workdir_has_other_live", return_value=False):
            return jobs.resume(job_dir, "synthetic continuation", allow_drift=True)

    def test_managed_auth_isolation_survives_default_and_named_resume_chain(self):
        from providers import jobs, job_runner
        binary, capture = self.subscription_vendor()
        for profile in ("default", "a"):
            health.clear_all()
            os.environ["CODEX_HOME"] = self.homes["a"]
            os.environ["OMNI_LLM_PROFILES_CODEX"] = ("default,b" if profile == "default" else "a,b")
            _seed_codex()
            capture.write_text("", encoding="utf-8")
            origin = self.job_start()
            self.assertNotIn("refused", origin)
            source = jobs._read_job(origin["job_dir"])
            self.assertTrue(source["codex_auth_isolated"])
            self.assertEqual(source["account_profile"], profile)
            with self.real_vendor_patches(binary):
                normal = asyncio.run(job_runner._execute(origin["job_dir"], source))
            self.assertEqual(normal["reason"], "rate_limit")
            os.environ.pop("OMNI_LLM_PROFILES_CODEX")
            for _ in range(2):
                child = self.resume_without_supervisor(origin["job_dir"])
                self.assertNotIn("refused", child)
                source = jobs._read_job(child["job_dir"])
                self.assertFalse(source["codex_managed"])
                self.assertTrue(source["codex_auth_isolated"])
                self.assertEqual(source["account_profile"], profile)
                self.assertEqual(source["codex_home"], self.homes["a"])
                with self.real_vendor_patches(binary):
                    resumed = asyncio.run(job_runner._execute(child["job_dir"], source))
                self.assertEqual(resumed["reason"], "rate_limit")
                origin = child
            rows = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(rows), 3)  # exactly one vendor per original/resume
            self.assertEqual([row["keys"] for row in rows], [[], [], []])
            self.assertEqual([row["home"] for row in rows], [self.homes["a"]] * 3)
            self.assertNotIn("resume", rows[0]["argv"])
            self.assertTrue(all("resume" in row["argv"] for row in rows[1:]))

    def test_original_no_ring_job_api_auth_survives_resume_with_new_ring(self):
        from providers import jobs, job_runner
        binary, capture = self.subscription_vendor()
        os.environ["CODEX_HOME"] = self.homes["a"]
        origin = self.job_start()
        source = jobs._read_job(origin["job_dir"])
        self.assertFalse(source["codex_auth_isolated"])
        with self.real_vendor_patches(binary):
            result = asyncio.run(job_runner._execute(origin["job_dir"], source))
        self.assertEqual(result["reason"], "rate_limit")
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "b"
        child = self.resume_without_supervisor(origin["job_dir"])
        self.assertNotIn("refused", child)
        source = jobs._read_job(child["job_dir"])
        self.assertFalse(source["codex_auth_isolated"])
        self.assertEqual(source["codex_home"], self.homes["a"])
        with self.real_vendor_patches(binary):
            result = asyncio.run(job_runner._execute(child["job_dir"], source))
        self.assertEqual(result["reason"], "rate_limit")
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual([set(row["keys"]) for row in rows], [{"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_API_BASE"}] * 2)
        self.assertEqual([row["home"] for row in rows], [self.homes["a"]] * 2)

    def test_managed_auth_provenance_cannot_be_disabled_by_internal_false(self):
        from providers import jobs, job_runner, agent_lane
        binary, capture = self.subscription_vendor()
        self.selection()
        origin = self.job_start(codex_auth_isolated=False)
        source = jobs._read_job(origin["job_dir"])
        self.assertTrue(source["codex_managed"])
        self.assertTrue(source["codex_auth_isolated"])
        with self.real_vendor_patches(binary):
            result = asyncio.run(job_runner._execute(origin["job_dir"], source))
        self.assertEqual(result["reason"], "rate_limit")
        health.clear_all()
        with self.real_vendor_patches(binary):
            result = asyncio.run(agent_lane.run_agent("synthetic task", "codex/synthetic-model", self.root,
                                                       timeout_s=5, codex_auth_isolated=False))
        self.assertEqual(result["reason"], "rate_limit")
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([row["keys"] for row in rows], [[], []])
        os.environ.pop("OMNI_LLM_PROFILES_CODEX")
        no_ring = self.job_start(codex_auth_isolated=False)
        self.assertFalse(jobs._read_job(no_ring["job_dir"])["codex_auth_isolated"])

    def test_direct_sync_resume_ring_isolates_keys_and_never_switches(self):
        from providers import agent_lane
        binary, capture = self.subscription_vendor()
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        with self.real_vendor_patches(binary):
            result = asyncio.run(agent_lane.run_agent("synthetic continuation", "codex/synthetic-model", self.root,
                                                       account_profile="a", codex_home=self.homes["a"],
                                                       resume_session="synthetic-original-session", timeout_s=5))
        self.assertEqual(result["reason"], "rate_limit")
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["keys"], [])
        self.assertEqual(rows[0]["home"], self.homes["a"])

    def test_runner_startup_failure_keeps_job_pin_and_session_on_first_account(self):
        from pathlib import Path
        from providers import jobs, job_runner, agent_lane
        self.selection()
        started = self.job_start()
        job_dir = started["job_dir"]
        job = jobs._read_job(job_dir)
        called = []
        async def runner(argv, **kwargs):
            called.append(kwargs["env"]["CODEX_HOME"])
            label = os.path.basename(called[-1])
            stream = json.dumps({"type": "thread.started", "thread_id": "synthetic-" + label}) + "\n"
            with open(kwargs["log_path"], "a") as handle:
                handle.write(stream)
            self.assertEqual(jobs._read_job(job_dir)["account_profile"], label)
            self.assertEqual(jobs._read_job(job_dir)["codex_home"], called[-1])
            job_runner._try_capture_session_id(job_dir)
            self.assertEqual(jobs._read_job(job_dir)["session_id"], "synthetic-a")
            return {"returncode": 1, "stdout": stream + event_error("401 Unauthorized"), "stderr": ""}
        with self.agent_patches(runner), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(job_dir, job))
        self.assertFalse(result["ok"])
        self.assertEqual(called, [self.homes["a"]])
        final = jobs._read_job(job_dir)
        self.assertEqual(final["account_profile"], "a")
        self.assertEqual(final["session_id"], "synthetic-a")


if __name__ == "__main__":
    unittest.main()
