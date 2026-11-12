#!/usr/bin/env python3
# no-opencc
"""A Codex request is never re-run on another account within the same request.

A definite startup quota/auth failure stops the request, cools and flags the
account, and leaves the failure envelope intact; the NEXT request selects the
next account. Hermetic: no vendor binary, no network.
"""
import asyncio
import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

from providers import codex_accounts as accounts, health
from tests import offline_home

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

MODEL = "codex/synthetic-model"


def quota_stream():
    return json.dumps({"type": "turn.failed", "error": {"code": "usage_limit_reached"}})


def auth_stream():
    return json.dumps({"type": "error", "message": "401 Unauthorized"})


class NoReplayCase(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        temp = tempfile.TemporaryDirectory(prefix="omni-noreplay-")
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        self.homes = {}
        for label in ("a", "b"):
            path = os.path.join(self.root, label)
            os.mkdir(path)
            self.homes[label] = path
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_LLM_STATE_DIR": self.root,
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_NOTIFY_QUIET": "1",
               "OMNI_LLM_PROFILES_CODEX": "a,b"}
        env.update({"CODEX_HOME_" + k.upper(): v for k, v in self.homes.items()})
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        self.spawns = []

    def runner(self, first_failure):
        """First spawn fails with the given JSONL stream; any later spawn succeeds."""
        async def run(argv, **kwargs):
            self.spawns.append(os.path.basename(kwargs["env"]["CODEX_HOME"]))
            if len(self.spawns) == 1:
                return {"returncode": 1, "stdout": first_failure,
                        "stderr": "synthetic quota", "timed_out": False}
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w") as handle:
                    handle.write("synthetic success")
            return {"returncode": 0, "stdout": json.dumps({"type": "item.completed",
                    "item": {"type": "agent_message", "text": "synthetic success"}}), "stderr": ""}
        return run

    def consult_patches(self, runner):
        from providers import codex
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"))
        stack.enter_context(mock.patch.object(codex, "prepare_host_fence",
                            side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})))
        stack.enter_context(mock.patch.object(codex.kernel_sandbox, "admit_vendor_binary", return_value=None))
        stack.enter_context(mock.patch.object(codex, "run_cli", side_effect=runner))
        return stack

    def agent_patches(self, runner):
        from providers import agent_lane
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(agent_lane, "run_cli", side_effect=runner))
        stack.enter_context(mock.patch.object(agent_lane.shutil, "which", return_value="/synthetic/codex"))
        stack.enter_context(mock.patch.object(agent_lane, "_workdir_fence", return_value=None))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_snapshot",
                            return_value={"entries": {}, "head": "synthetic"}))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)))
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=
                            lambda provider, argv, env, *args, **kwargs: (argv, env, "vendor", {}, None)))
        stack.enter_context(mock.patch.object(agent_lane.sandbox_canary, "gate", return_value=None))
        return stack

    def assert_cooled(self, label):
        entry = health.snapshot()[health.pool_key("codex", label)]
        self.assertEqual(entry["source"], "account_startup")
        self.assertEqual(health.seat_kind(MODEL, label), "cooling")

    def assert_failure_envelope(self, result, reason):
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["reason"], reason)
        self.assertEqual(result["account_profile"], "a")
        self.assertIs(result["account_retry_safe"], False)
        self.assertEqual(result["tried"], [{"account_profile": "a", "reason": reason}])

    def test_consult_startup_failure_spawns_once_then_next_request_uses_b(self):
        from providers import codex
        for stream, reason in ((quota_stream(), "rate_limit"), (auth_stream(), "auth_error")):
            with self.subTest(reason=reason):
                health.clear_all()
                self.spawns.clear()
                with self.consult_patches(self.runner(stream)):
                    first = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
                    self.assertEqual(self.spawns, ["a"], "the request was replayed on another account")
                    self.assert_failure_envelope(first, reason)
                    self.assert_cooled("a")
                    second = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
                self.assertTrue(second["ok"], second)
                self.assertEqual(second["account_profile"], "b")
                self.assertEqual(self.spawns, ["a", "b"])

    def test_chain_does_not_reach_another_codex_seat_after_startup_failure(self):
        from providers import dispatch
        with self.consult_patches(self.runner(quota_stream())), \
                mock.patch.object(dispatch, "_health_maintain"), \
                mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)):
            result = asyncio.run(dispatch.dispatch_llm_request(
                "synthetic task", "codex/synthetic-one,codex/synthetic-two", 5, [], "default"))
        self.assertEqual(self.spawns, ["a"], "the chain replayed the request on another account")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")

    def test_agent_startup_failure_spawns_once_then_next_request_uses_b(self):
        from providers import agent_lane
        for stream, reason in ((quota_stream(), "rate_limit"), (auth_stream(), "auth_error")):
            with self.subTest(reason=reason):
                health.clear_all()
                self.spawns.clear()
                with self.agent_patches(self.runner(stream)):
                    first = asyncio.run(agent_lane.run_agent(
                        "synthetic task", MODEL, self.root, read_only=True, timeout_s=5))
                    self.assertEqual(self.spawns, ["a"], "the agent run was replayed on another account")
                    self.assert_failure_envelope(first, reason)
                    self.assert_cooled("a")
                    second = asyncio.run(agent_lane.run_agent(
                        "synthetic task", MODEL, self.root, read_only=True, timeout_s=5))
                self.assertTrue(second["ok"], second)
                self.assertEqual(second["account_profile"], "b")
                self.assertEqual(self.spawns, ["a", "b"])

    def test_job_style_run_with_remaining_candidates_spawns_once(self):
        """The shape a detached job runner passes: pinned first account plus the rest."""
        from providers import agent_lane
        selected = []
        with self.agent_patches(self.runner(quota_stream())):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", MODEL, self.root, read_only=True, timeout_s=5,
                account_profile="a", codex_home=self.homes["a"], codex_homes=dict(self.homes),
                codex_candidates=["b"], codex_tried=[],
                codex_initial={"account_profile": "a", "codex_home": self.homes["a"]},
                codex_auth_isolated=True,
                on_account_selected=lambda account, selection: selected.append(account["account_profile"])))
        self.assertEqual(self.spawns, ["a"], "the job run was replayed on another account")
        self.assertEqual(selected, ["a"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(result["account_profile"], "a")
        self.assert_cooled("a")

    def test_run_selected_never_calls_attempt_twice_for_a_startup_failure(self):
        """The selector itself: a typed startup failure on a managed ring ends the walk."""
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        selection = accounts.AccountSelection()
        calls = []
        async def attempt(account, remaining, managed):
            calls.append(account["account_profile"])
            return accounts.annotate_failure({"ok": False, "reason": "runtime_error"}, quota_stream())
        result = asyncio.run(accounts.run_selected(selection, attempt))
        self.assertEqual(calls, ["a"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertIs(result["account_retry_safe"], False)
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,b"
        self.assertEqual(accounts.AccountSelection().next()["account_profile"], "b")

    def test_failed_cools_the_account_and_leaves_the_result_untouched(self):
        selection = accounts.AccountSelection()
        result = {"reason": "rate_limit", "account_retry_safe": False}
        selection.failed({"account_profile": "a"}, result)
        self.assertEqual(result, {"reason": "rate_limit", "account_retry_safe": False})
        self.assertEqual(health.snapshot()[health.pool_key("codex", "a")]["source"], "account_startup")
        self.assertEqual(selection.tried, [{"account_profile": "a", "reason": "rate_limit"}])

    def test_no_stream_ever_marks_a_codex_failure_retry_safe(self):
        item = json.dumps({"type": "item.started", "item": {"type": "command_execution"}})
        streams = {
            "clean_quota": quota_stream(),
            "clean_auth": auth_stream(),
            "item_activity": item + "\n" + quota_stream(),
            "unknown_event": json.dumps({"type": "new.future.event"}) + "\n" + quota_stream(),
            "truncated_json": quota_stream() + "\n{",
            "empty": "",
            "not_json": "not-json",
        }
        for name, stream in streams.items():
            for reason in ("runtime_error", "rate_limit", "auth_error"):
                with self.subTest(stream=name, reason=reason):
                    annotated = accounts.annotate_failure({"ok": False, "reason": reason}, stream)
                    self.assertIs(annotated["account_retry_safe"], False)
        # The stream still names the failure and the tool-activity fact.
        clean = accounts.annotate_failure({"ok": False, "reason": "runtime_error"}, quota_stream())
        self.assertEqual((clean["reason"], clean["tool_activity"]), ("rate_limit", False))
        active = accounts.annotate_failure({"ok": False, "reason": "runtime_error"}, item + "\n" + quota_stream())
        self.assertIs(active["tool_activity"], True)

    # Pinned and no-ring Codex seats: the same single attempt, no fall-through.
    def marker_runner(self, stream):
        """Every spawn records a side effect, then reports a startup quota failure."""
        marker = os.path.join(self.root, "side-effects.txt")
        async def run(argv, **kwargs):
            self.spawns.append(os.path.basename(kwargs["env"].get("CODEX_HOME", "default")))
            with open(marker, "a") as handle:
                handle.write("ran\n")
            return {"returncode": 1, "stdout": stream, "stderr": "synthetic quota", "timed_out": False}
        return run, marker

    def chain(self, model, family=None):
        from providers import dispatch
        if family:
            os.environ["OMNI_LLM_FAMILY_SOL"] = family
        cursor = mock.AsyncMock(return_value={"ok": True, "text": "synthetic cursor"})
        sleep = mock.AsyncMock()
        with mock.patch.object(dispatch.cursor_mod, "run_cursor_subprocess", new=cursor), \
                mock.patch.object(dispatch.asyncio, "sleep", new=sleep), \
                mock.patch.object(dispatch, "_health_maintain"), \
                mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)):
            result = asyncio.run(dispatch.dispatch_llm_request(
                "synthetic task", model, 5, [], "default"))
        return result, cursor, sleep

    def test_pinned_family_seat_then_codex_seat_runs_once_and_writes_the_marker_once(self):
        runner, marker = self.marker_runner(quota_stream())
        with self.consult_patches(runner):
            result, cursor, sleep = self.chain("sol,codex/gpt-5.6-sol", family="codex:a")
        self.assertEqual(self.spawns, ["a"], "the chain re-ran the request")
        with open(marker) as handle:
            self.assertEqual(handle.read().splitlines(), ["ran"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertIs(result["account_retry_safe"], False)
        cursor.assert_not_called()
        sleep.assert_not_called()

    def test_no_ring_default_codex_quota_failure_has_no_backoff_retry(self):
        del os.environ["OMNI_LLM_PROFILES_CODEX"]
        runner, marker = self.marker_runner(quota_stream())
        with self.consult_patches(runner):
            result, cursor, sleep = self.chain("codex/synthetic-model")
        self.assertEqual(len(self.spawns), 1, "the default account was run again")
        with open(marker) as handle:
            self.assertEqual(handle.read().splitlines(), ["ran"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertIs(result["account_retry_safe"], False)
        sleep.assert_not_called()

    def test_family_codex_then_cursor_never_reaches_cursor_without_a_ring(self):
        del os.environ["OMNI_LLM_PROFILES_CODEX"]
        runner, marker = self.marker_runner(quota_stream())
        with self.consult_patches(runner):
            result, cursor, sleep = self.chain("sol", family="codex,cursor")
        self.assertEqual(len(self.spawns), 1)
        self.assertFalse(result["ok"])
        cursor.assert_not_called()
        sleep.assert_not_called()

    def test_family_codex_then_cursor_never_reaches_cursor_with_a_ring(self):
        runner, marker = self.marker_runner(quota_stream())
        with self.consult_patches(runner):
            result, cursor, sleep = self.chain("sol", family="codex,cursor")
        self.assertEqual(self.spawns, ["a"])
        self.assertFalse(result["ok"])
        cursor.assert_not_called()
        sleep.assert_not_called()

    def test_explicit_profile_consult_failure_is_not_retry_safe(self):
        from providers import codex
        runner, marker = self.marker_runner(quota_stream())
        with self.consult_patches(runner):
            result = asyncio.run(codex.run_codex_subprocess(
                "synthetic task", "synthetic-model", 5, account_profile="b", account_pinned=True))
        self.assertEqual(self.spawns, ["b"])
        self.assertIs(result["account_retry_safe"], False)
        self.assertEqual(result["reason"], "rate_limit")

    def test_pinned_auth_failure_cools_the_account_and_the_next_ring_request_uses_b(self):
        from providers import codex
        with self.consult_patches(self.runner(auth_stream())):
            first = asyncio.run(codex.run_codex_subprocess(
                "synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
            self.assertEqual(self.spawns, ["a"])
            self.assertFalse(first["ok"])
            self.assertEqual(first["reason"], "auth_error")
            self.assertIs(first["account_retry_safe"], False)
            self.assert_cooled("a")
            second = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(second["ok"], second)
        self.assertEqual(second["account_profile"], "b")
        self.assertEqual(self.spawns, ["a", "b"])

    def test_pinned_runtime_failure_does_not_cool_the_account(self):
        from providers import codex
        failure = json.dumps({"type": "turn.failed", "error": {"message": "synthetic internal fault"}})
        with self.consult_patches(self.runner(failure)):
            first = asyncio.run(codex.run_codex_subprocess(
                "synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
        self.assertFalse(first["ok"])
        self.assertEqual(first["reason"], "runtime_error")
        self.assertNotIn(health.pool_key("codex", "a"), health.snapshot())

    def chain_over(self, results, model="codex/synthetic-one,cursor/synthetic-two"):
        """Dispatch a chain whose first seat returns each given result in turn."""
        from providers import dispatch
        cursor = mock.AsyncMock(return_value={"ok": True, "text": "synthetic cursor"})
        sleep = mock.AsyncMock()
        route = mock.AsyncMock(side_effect=list(results))
        with mock.patch.object(dispatch, "route_to_provider", new=route), \
                mock.patch.object(dispatch.cursor_mod, "run_cursor_subprocess", new=cursor), \
                mock.patch.object(dispatch.asyncio, "sleep", new=sleep), \
                mock.patch.object(dispatch, "_flag_rate_limit"), \
                mock.patch.object(dispatch, "_health_note_failure"), \
                mock.patch.object(dispatch, "_health_maintain"), \
                mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", model, 5, [], "default"))
        return result, route, cursor, sleep

    def test_tool_activity_alone_stops_the_chain_without_a_retry_flag(self):
        result, route, cursor, sleep = self.chain_over(
            [{"ok": False, "reason": "runtime_error", "tool_activity": True}])
        self.assertEqual(route.call_count, 1)
        self.assertFalse(result["ok"])
        cursor.assert_not_called()

    def test_a_retry_safe_codex_result_still_stops_the_chain_and_the_backoff(self):
        """A producer regression (account_retry_safe True) must not reopen a re-run."""
        regressed = {"ok": False, "reason": "rate_limit", "account_retry_safe": True}
        result, route, cursor, sleep = self.chain_over([regressed, regressed])
        self.assertEqual(route.call_count, 1, "codex was run again")
        self.assertEqual(result["reason"], "rate_limit")
        cursor.assert_not_called()
        sleep.assert_not_called()

    def test_account_exhausted_is_the_only_codex_result_that_moves_on(self):
        exhausted = {"ok": False, "reason": "account_exhausted", "account_retry_safe": False}
        result, route, cursor, sleep = self.chain_over([exhausted, {"ok": True, "text": "synthetic cursor"}])
        self.assertTrue(result["ok"], result)
        self.assertEqual(route.call_count, 2)

    def test_codex_failure_before_the_child_starts_ends_the_request(self):
        from providers import codex
        for ring in (True, False):
            with self.subTest(ring=ring):
                if not ring:
                    os.environ.pop("OMNI_LLM_PROFILES_CODEX", None)
                self.spawns.clear()
                runner, marker = self.marker_runner(quota_stream())
                with self.consult_patches(runner), \
                        mock.patch.object(codex.shutil, "which", return_value=None):
                    result, cursor, sleep = self.chain("codex/synthetic-one,cursor/synthetic-two")
                self.assertEqual(self.spawns, [])
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "cli_missing")
                cursor.assert_not_called()

    def test_pinned_success_clears_the_account_flag_a_pinned_failure_raised(self):
        from providers import codex
        with self.consult_patches(self.runner(auth_stream())):
            first = asyncio.run(codex.run_codex_subprocess(
                "synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
            self.assertFalse(first["ok"])
            self.assert_cooled("a")
            second = asyncio.run(codex.run_codex_subprocess(
                "synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
        self.assertTrue(second["ok"], second)
        self.assertEqual(self.spawns, ["a", "a"])
        self.assertNotIn(health.pool_key("codex", "a"), health.snapshot())

    def test_pinned_success_keeps_quota_and_credit_evidence(self):
        import time
        from providers import codex
        for source in ("quota", "credit_exhausted"):
            with self.subTest(source=source):
                health.clear_all()
                self.spawns.clear()
                health.flag("codex", "a", until=time.time() + 3600, source=source)
                self.spawns.append("warm")  # makes the shared runner succeed on its first spawn
                with self.consult_patches(self.runner(None)):
                    result = asyncio.run(codex.run_codex_subprocess(
                        "synthetic task", "synthetic-model", 5, account_profile="a", account_pinned=True))
                self.assertTrue(result["ok"], result)
                self.assertEqual(health.snapshot()[health.pool_key("codex", "a")]["source"], source)


if __name__ == "__main__":
    unittest.main()
