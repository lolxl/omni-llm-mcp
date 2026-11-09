#!/usr/bin/env python3
# no-opencc
"""Issue #156: a failure omni raised itself does not cool a seat.

A local exception (missing task file, failed spawn, a refusal by omni's own
gate) is returned as ``runtime_error`` like a crashed vendor run, but no vendor
ran, so it must not count toward the crash streak (3 in 10 minutes -> 5 minute
cooldown, source=crash). The envelope carries an internal origin stamp set only
at omni's own exception boundaries; the streak skips stamped envelopes (neither
incrementing nor resetting) and the stamp never leaves dispatch.

Hermetic: ``run_cli`` and the vendor binary lookup are mocked, nothing spawns.
"""
import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from providers import claude, dispatch, errors, health

MODEL = "claude/synthetic-model"
PROFILE = "default"
MARKER = errors.LOCAL_ORIGIN_KEY


def _run(coro):
    return asyncio.run(coro)


class Env(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-156-")
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_LLM_STATE_DIR": self.root,
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_NOTIFY_QUIET": "1"}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)

    def entry(self):
        return health.snapshot().get(health.pool_key(MODEL, PROFILE))

    def raw_entry(self):
        """The ledger record, active or not (``snapshot`` lists active ones)."""
        return health._read_ledger_unlocked().get(health.pool_key(MODEL, PROFILE))

    def dispatch_claude(self, run_cli):
        """One real dispatch_single -> route_to_provider -> claude, CLI mocked."""
        with mock.patch.object(claude.shutil, "which", return_value="/synthetic/claude"), \
                mock.patch.object(claude, "run_cli", side_effect=run_cli):
            return _run(dispatch.dispatch_single(
                "synthetic task", MODEL, 5, [], PROFILE))

    def local_failure(self):
        async def spawn_fails(argv, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "/synthetic/task.txt")
        return self.dispatch_claude(spawn_fails)

    def vendor_failure(self, stdout="", stderr="boom segfault"):
        async def crashed(argv, **kwargs):
            return {"returncode": 1, "stdout": stdout, "stderr": stderr}
        return self.dispatch_claude(crashed)


class TestLocalFailuresDoNotCool(Env):
    def test_three_local_exceptions_leave_the_seat_alone(self):
        for _ in range(3):
            res = self.local_failure()
            self.assertIs(res.get("ok"), False, res)
            self.assertEqual(res.get("reason"), "runtime_error", res)
            self.assertTrue(res["detail"].startswith("FileNotFoundError: "), res)
        self.assertIsNone(self.raw_entry())
        self.assertFalse(health.active(MODEL, PROFILE))

    def test_local_failures_neither_raise_nor_reset_a_vendor_streak(self):
        self.vendor_failure()
        self.vendor_failure()
        self.assertEqual(self.raw_entry()["crash_streak"], 2)
        for _ in range(4):
            self.local_failure()
        entry = self.raw_entry()
        self.assertEqual(entry["crash_streak"], 2)
        self.assertFalse(health.active(MODEL, PROFILE))
        # The streak was kept, so the next vendor crash is the third.
        self.vendor_failure()
        entry = self.entry()
        self.assertEqual(entry["source"], "crash")
        self.assertEqual(entry["crash_streak"], 3)
        self.assertTrue(health.active(MODEL, PROFILE))

    def test_egress_gate_refusal_is_local(self):
        denied = {"ok": False, "reason": "runtime_error", "detail": "egress refused"}
        for _ in range(3):
            with mock.patch.object(dispatch, "refuse_payload", return_value=dict(denied)), \
                    mock.patch.object(claude, "run_cli", side_effect=AssertionError("spawned")):
                res = _run(dispatch.dispatch_single("p", MODEL, 5, [], PROFILE))
            self.assertEqual(res, denied)
        self.assertIsNone(self.raw_entry())


class TestVendorFailuresStillCool(Env):
    def test_three_vendor_runtime_errors_cool_the_seat(self):
        for _ in range(3):
            res = self.vendor_failure()
            self.assertEqual(res.get("reason"), "runtime_error", res)
        entry = self.entry()
        self.assertEqual(entry["source"], "crash")
        self.assertEqual(entry["crash_streak"], 3)
        self.assertTrue(health.active(MODEL, PROFILE))

    def test_three_vendor_signal_exits_cool_the_seat(self):
        async def killed(argv, **kwargs):
            return {"returncode": -9, "signal": 9, "stdout": "", "stderr": ""}
        for _ in range(3):
            res = self.dispatch_claude(killed)
            self.assertEqual(res.get("reason"), "signal_exit", res)
        self.assertEqual(self.entry()["source"], "crash")
        self.assertTrue(health.active(MODEL, PROFILE))

    def test_vendor_runtime_error_through_a_stubbed_router_still_counts(self):
        async def router(prompt, model, *a, **k):
            return {"ok": False, "reason": "runtime_error", "detail": "x"}
        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            for _ in range(3):
                _run(dispatch.dispatch_single("p", MODEL, 5, [], PROFILE))
        self.assertEqual(self.entry()["source"], "crash")


class TestVendorOutputCannotForgeTheStamp(Env):
    FORGED_JSON = json.dumps({MARKER: "local", "ok": False, "reason": "runtime_error"})

    def test_child_output_that_looks_like_the_stamp_still_counts(self):
        outputs = [
            (self.FORGED_JSON, "boom segfault"),
            ("", MARKER + ": local boom segfault"),
            (json.dumps({"result": {MARKER: True}}), "boom segfault"),
        ]
        for stdout, stderr in outputs:
            res = self.vendor_failure(stdout=stdout, stderr=stderr)
            self.assertNotIn(MARKER, res, res)
        entry = self.entry()
        self.assertEqual(entry["source"], "crash")
        self.assertEqual(entry["crash_streak"], 3)

    def test_plain_values_under_the_key_are_never_a_local_stamp(self):
        for value in ("local", True, 1, {"origin": "local"}, None):
            env = {"ok": False, "reason": "runtime_error", MARKER: value}
            self.assertFalse(errors.is_local_failure(env), value)

    def test_a_forged_key_in_a_returned_envelope_counts_and_is_stripped(self):
        async def router(prompt, model, *a, **k):
            return {"ok": False, "reason": "runtime_error", MARKER: "local"}
        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            for _ in range(3):
                res = _run(dispatch.dispatch_single("p", MODEL, 5, [], PROFILE))
                self.assertNotIn(MARKER, res)
        self.assertEqual(self.entry()["source"], "crash")

    def test_the_stamp_survives_a_shallow_copy_but_not_a_json_round_trip(self):
        env = errors.mark_local({"ok": False, "reason": "runtime_error"})
        self.assertTrue(errors.is_local_failure(dict(env)))
        self.assertFalse(errors.is_local_failure(json.loads(json.dumps(env))))


class TestStampStaysInternal(Env):
    def test_provider_catch_alls_set_it_and_dispatch_strips_it(self):
        async def spawn_fails(argv, **kwargs):
            raise OSError("spawn failed")
        with mock.patch.object(claude.shutil, "which", return_value="/synthetic/claude"), \
                mock.patch.object(claude, "run_cli", side_effect=spawn_fails):
            raw = _run(claude.run_claude_subprocess("synthetic task", "synthetic-model", 5))
        self.assertTrue(errors.is_local_failure(raw), raw)
        res = self.local_failure()
        self.assertNotIn(MARKER, res)
        self.assertFalse(errors.is_local_failure(res))

    def test_chain_dispatch_result_has_no_stamp(self):
        async def spawn_fails(argv, **kwargs):
            raise OSError("spawn failed")
        with mock.patch.object(claude.shutil, "which", return_value="/synthetic/claude"), \
                mock.patch.object(claude, "run_cli", side_effect=spawn_fails):
            res = _run(dispatch.dispatch_llm_request("synthetic task", MODEL, 5, [], PROFILE))
        self.assertEqual(res.get("reason"), "runtime_error", res)
        self.assertNotIn(MARKER, json.dumps(res))

    def test_call_tool_local_refusals_carry_no_stamp(self):
        cases = [
            ("llm_generate", {"prompt": "  "}),
            ("llm_big_context", {"instruction": "i", "text": ""}),
            ("llm_image", {"prompt": ""}),
            ("llm_image", {"prompt": "p", "backend": "nope"}),
            ("no_such_tool", {}),
        ]
        for name, args in cases:
            res = _run(dispatch.call_tool(name, args))
            self.assertIs(res.get("ok"), False, res)
            self.assertEqual(res.get("reason"), "runtime_error", res)
            self.assertNotIn(MARKER, json.dumps(res))
        # The stamp is real inside: the inner function carries it.
        inner = _run(dispatch._call_tool_inner("no_such_tool", {}))
        self.assertTrue(errors.is_local_failure(inner), inner)

    def test_mcp_response_has_no_stamp(self):
        import server
        sent = []

        async def capture(id_, result):
            sent.append(result)

        with mock.patch.object(server, "_result", side_effect=capture):
            _run(server.handle_tool_call(1, "no_such_tool", {}))
        self.assertEqual(len(sent), 1)
        self.assertNotIn(MARKER, sent[0]["content"][0]["text"])
        self.assertNotIn(MARKER, sent[0]["structuredContent"])

    def test_cli_output_has_no_stamp(self):
        import omni_cli
        envelope = omni_cli._call("no_such_tool", {})
        self.assertEqual(envelope.get("reason"), "runtime_error", envelope)
        self.assertNotIn(MARKER, json.dumps(envelope))

    def test_strip_returns_the_same_object_when_there_is_nothing_to_strip(self):
        env = {"ok": True}
        self.assertIs(errors.strip_local_marker(env), env)
        stamped = errors.mark_local({"ok": False})
        self.assertNotIn(MARKER, errors.strip_local_marker(stamped))
        self.assertIn(MARKER, stamped)


DEEP = "[" * 1500 + "0" + "]" * 1500        # json decoder: RecursionError
CLAUDE_BAD_STATUS = json.dumps(
    {"type": "result", "is_error": True, "api_error_status": []})

# (seat model, provider module name). Every seat runs the real
# dispatch_single -> route_to_provider -> provider path with only the CLI
# runner and the binary lookup replaced.
SEATS = [
    ("claude/synthetic-model", "claude"),
    ("codex/synthetic-model", "codex"),
    ("grok-synthetic", "grok"),
    ("pi/synthetic-model", "pi"),
    ("cursor/synthetic-model", "cursor"),
    ("Synthetic Gemini", "agy"),
]


class SeatEnv(Env):
    """Run any one seat through dispatch with the vendor CLI replaced."""

    def run_seat(self, model, modname, run_cli, files=None, prompt="synthetic task"):
        from providers import agy, codex, cursor, grok, kernel_sandbox, pi
        mod = {"claude": claude, "codex": codex, "grok": grok, "pi": pi,
               "cursor": cursor, "agy": agy}[modname]
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            enter(mock.patch.object(shutil, "which", return_value="/synthetic/bin"))
            enter(mock.patch.object(agy, "HAS_SDK", False))
            enter(mock.patch.object(mod, "run_cli", side_effect=run_cli))
            enter(mock.patch.object(cursor, "resolve_bin", return_value="/synthetic/cursor"))
            enter(mock.patch.object(kernel_sandbox, "admit_vendor_binary", return_value=None))
            for module in (claude, codex, grok, pi, cursor):
                enter(mock.patch.object(
                    module, "prepare_host_fence",
                    side_effect=lambda argv, **kw: (argv, kw["env"], {})))
            return _run(dispatch.dispatch_single(
                prompt, model, 5, files or [], PROFILE))

    def raw(self, model):
        return health._read_ledger_unlocked().get(health.pool_key(model, PROFILE))


def _write_codex_reply(argv, data):
    with open(argv[argv.index("-o") + 1], "wb") as handle:
        handle.write(data)


class TestPostSpawnParsingFailuresStillCount(SeatEnv):
    """An exception while reading what the child printed is vendor-side."""

    def _three(self, model, modname, run_cli):
        results = [self.run_seat(model, modname, run_cli) for _ in range(3)]
        for res in results:
            self.assertEqual(res.get("reason"), "runtime_error", res)
            self.assertNotIn(MARKER, res)
        entry = self.raw(model)
        self.assertEqual(entry["crash_streak"], 3, entry)
        self.assertEqual(entry["source"], "crash", entry)

    def test_deeply_nested_json_counts_on_every_provider(self):
        for model, modname in SEATS:
            for rc in (0, 1):
                with self.subTest(provider=modname, rc=rc):
                    health.clear_all()

                    async def run_cli(argv, **kwargs):
                        return {"returncode": rc, "stdout": DEEP, "stderr": ""}
                    self._three(model, modname, run_cli)

    def test_claude_and_cursor_unhashable_api_error_status_counts(self):
        for model, modname in (("claude/synthetic-model", "claude"),
                               ("cursor/synthetic-model", "cursor")):
            with self.subTest(provider=modname):
                health.clear_all()

                async def run_cli(argv, **kwargs):
                    return {"returncode": 0, "stdout": CLAUDE_BAD_STATUS, "stderr": ""}
                self._three(model, modname, run_cli)

    def test_codex_reply_file_with_invalid_utf8_counts(self):
        async def run_cli(argv, **kwargs):
            _write_codex_reply(argv, b"\xff\xfe\x80 not utf-8")
            return {"returncode": 0, "stdout": "", "stderr": ""}
        self._three("codex/synthetic-model", "codex", run_cli)

    def test_codex_signal_exit_with_unparsable_output_still_counts(self):
        async def run_cli(argv, **kwargs):
            return {"returncode": -9, "signal": 9, "stdout": DEEP, "stderr": ""}
        for _ in range(3):
            res = self.run_seat("codex/synthetic-model", "codex", run_cli)
            self.assertIn(res.get("reason"), ("runtime_error", "signal_exit"), res)
            self.assertNotIn(MARKER, res)
        self.assertEqual(self.raw("codex/synthetic-model")["source"], "crash")

    def test_run_cli_raising_stays_local_on_every_provider(self):
        for model, modname in SEATS:
            with self.subTest(provider=modname):
                health.clear_all()

                async def run_cli(argv, **kwargs):
                    raise FileNotFoundError(2, "No such file", "/synthetic/task.txt")
                for _ in range(3):
                    res = self.run_seat(model, modname, run_cli)
                    self.assertEqual(res.get("reason"), "runtime_error", res)
                    self.assertNotIn(MARKER, res)
                self.assertIsNone(self.raw(model))


class TestPreSpawnRefusalsAreLocal(SeatEnv):
    def seed_streak(self, model, modname, n=2):
        async def crashed(argv, **kwargs):
            return {"returncode": 1, "stdout": "", "stderr": "boom segfault"}
        for _ in range(n):
            self.run_seat(model, modname, crashed)
        self.assertEqual(self.raw(model)["crash_streak"], n)

    def test_denied_grok_attachment_with_allowed_prompt_never_cools(self):
        src = os.path.join(self.root, "src")
        os.mkdir(src)
        deny = os.path.join(self.root, "deny.txt")
        with open(deny, "w", encoding="utf-8") as handle:
            handle.write("SYNTHETIC-DENIED-LITERAL\n")
        attachment = os.path.join(src, "note.txt")
        with open(attachment, "w", encoding="utf-8") as handle:
            handle.write("body with SYNTHETIC-DENIED-LITERAL inside")
        calls = []

        async def run_cli(argv, **kwargs):
            calls.append(argv)
            return {"returncode": 0, "stdout": "", "stderr": ""}
        env = {"OMNI_EGRESS_GATE": "1", "OMNI_EGRESS_DENYLIST": deny,
               "OMNI_LLM_SOURCE_ROOTS": src}
        with mock.patch.dict(os.environ, env):
            for _ in range(3):
                res = self.run_seat("grok-synthetic", "grok", run_cli, files=[attachment])
                self.assertEqual(res.get("reason"), "runtime_error", res)
                self.assertIn("OMNI-EGRESS-DENY", res.get("detail", ""), res)
                self.assertNotIn(MARKER, res)
        self.assertEqual(calls, [])
        self.assertIsNone(self.raw("grok-synthetic"))

    def test_agy_prompt_plus_files_gate_refusal_never_cools(self):
        deny = os.path.join(self.root, "deny.txt")
        with open(deny, "w", encoding="utf-8") as handle:
            handle.write("SYNTHETIC-DENIED-FILENAME\n")
        calls = []

        async def run_cli(argv, **kwargs):
            calls.append(argv)
            return {"returncode": 0, "stdout": "", "stderr": ""}
        env = {"OMNI_EGRESS_GATE": "1", "OMNI_EGRESS_DENYLIST": deny}
        with mock.patch.dict(os.environ, env):
            for _ in range(3):
                res = self.run_seat("Synthetic Gemini", "agy", run_cli,
                                    files=["/synthetic/SYNTHETIC-DENIED-FILENAME.txt"])
                self.assertEqual(res.get("reason"), "runtime_error", res)
                self.assertNotIn(MARKER, res)
        self.assertEqual(calls, [])
        self.assertIsNone(self.raw("Synthetic Gemini"))

    def test_provider_side_gate_refusals_never_cool(self):
        deny = os.path.join(self.root, "deny.txt")
        with open(deny, "w", encoding="utf-8") as handle:
            handle.write("SYNTHETIC-DENIED-LITERAL\n")
        calls = []

        async def run_cli(argv, **kwargs):
            calls.append(argv)
            return {"returncode": 0, "stdout": "", "stderr": ""}
        env = {"OMNI_EGRESS_GATE": "1", "OMNI_EGRESS_DENYLIST": deny}
        for model, modname in SEATS:
            with self.subTest(provider=modname), mock.patch.dict(os.environ, env):
                for _ in range(3):
                    # Skip dispatch's own gate: the provider's must hold alone.
                    with mock.patch.object(dispatch, "refuse_payload", return_value=None):
                        res = self.run_seat(model, modname, run_cli,
                                            prompt="has SYNTHETIC-DENIED-LITERAL")
                    self.assertEqual(res.get("reason"), "runtime_error", res)
                    self.assertIn("OMNI-EGRESS-DENY", res.get("detail", ""), res)
                    self.assertNotIn(MARKER, res)
                self.assertIsNone(self.raw(model))
        self.assertEqual(calls, [])

    def test_unsupported_files_and_cli_missing_keep_a_vendor_streak(self):
        model, modname = "claude/synthetic-model", "claude"
        self.seed_streak(model, modname)

        async def never(argv, **kwargs):
            raise AssertionError("no child may run")
        for _ in range(3):
            res = self.run_seat(model, modname, never, files=["/synthetic/a.png"])
            self.assertEqual(res.get("reason"), "unsupported_files", res)
            self.assertNotIn(MARKER, res)
        self.assertEqual(self.raw(model)["crash_streak"], 2)
        for _ in range(3):
            with mock.patch.object(shutil, "which", return_value=None), \
                    mock.patch.object(claude, "run_cli", side_effect=never):
                res = _run(dispatch.dispatch_single("p", model, 5, [], PROFILE))
            self.assertEqual(res.get("reason"), "cli_missing", res)
            self.assertNotIn(MARKER, res)
        self.assertEqual(self.raw(model)["crash_streak"], 2)

    def test_codex_missing_file_keeps_a_vendor_streak(self):
        model = "codex/synthetic-model"
        self.seed_streak(model, "codex")

        async def never(argv, **kwargs):
            raise AssertionError("no child may run")
        for _ in range(3):
            res = self.run_seat(model, "codex", never,
                                files=[os.path.join(self.root, "missing.png")])
            self.assertEqual(res.get("reason"), "unsupported_files", res)
            self.assertNotIn(MARKER, res)
        self.assertEqual(self.raw(model)["crash_streak"], 2)

    def test_recursion_guard_keeps_a_vendor_streak(self):
        model, modname = "claude/synthetic-model", "claude"
        self.seed_streak(model, modname)

        async def never(argv, **kwargs):
            raise AssertionError("no child may run")
        with mock.patch.object(dispatch, "env_depth", return_value=2):
            res = self.run_seat(model, modname, never)
        self.assertEqual(res.get("reason"), "recursion_guard", res)
        self.assertNotIn(MARKER, res)
        self.assertEqual(self.raw(model)["crash_streak"], 2)

    def test_late_pre_spawn_refusal_creates_no_child_and_is_local(self):
        # run_cli answers without a returncode when the pre_spawn hook refuses.
        async def refused(argv, **kwargs):
            return {"ok": False, "reason": "runtime_error", "detail": "pre-spawn refusal"}
        for _ in range(3):
            res = self.run_seat("claude/synthetic-model", "claude", refused)
            self.assertEqual(res.get("reason"), "runtime_error", res)
            self.assertNotIn(MARKER, res)
        self.assertIsNone(self.raw("claude/synthetic-model"))


class TestAccountLayerRefusals(SeatEnv):
    """Refusals made by account selection happen before any attempt spawns."""

    def seed_streak(self, model, modname, n=2):
        async def crashed(argv, **kwargs):
            return {"returncode": 1, "stdout": "", "stderr": "boom segfault"}
        for _ in range(n):
            self.run_seat(model, modname, crashed)
        self.assertEqual(self.raw(model)["crash_streak"], n)

    def test_bad_profile_config_keeps_a_vendor_streak(self):
        for model, modname in (("claude/synthetic-model", "claude"),
                               ("codex/synthetic-model", "codex"),
                               ("grok-synthetic", "grok")):
            with self.subTest(provider=modname):
                health.clear_all()
                self.seed_streak(model, modname)
                calls = []

                async def run_cli(argv, **kwargs):
                    calls.append(argv)
                    return {"returncode": 0, "stdout": "", "stderr": ""}
                env = {"OMNI_LLM_PROFILES_%s" % modname.upper(): "bad/name"}
                with mock.patch.dict(os.environ, env):
                    for _ in range(3):
                        res = self.run_seat(model, modname, run_cli)
                        self.assertIs(res.get("ok"), False, res)
                        self.assertEqual(res.get("reason"), "profile_config", res)
                        self.assertNotIn(MARKER, res)
                self.assertEqual(calls, [])
                self.assertEqual(self.raw(model)["crash_streak"], 2)

    def test_request_stamp_follows_whether_any_attempt_spawned(self):
        exhausted = {"ok": False, "reason": "account_exhausted", "detail": "none"}

        @errors.local_request_before_spawn()
        async def refused_before_any_attempt():
            return dict(exhausted)

        @errors.local_request_before_spawn()
        async def refused_after_a_spawn():
            errors.note_spawned(None)
            return dict(exhausted)

        @errors.local_request_before_spawn()
        async def succeeded():
            return {"ok": True, "text": "t"}

        self.assertTrue(errors.is_local_failure(_run(refused_before_any_attempt())))
        self.assertFalse(errors.is_local_failure(_run(refused_after_a_spawn())))
        self.assertFalse(errors.is_local_failure(_run(succeeded())))

    def test_attempt_that_spawned_makes_the_whole_request_vendor_side(self):
        @errors.local_before_spawn
        async def attempt(_spawn=None):
            errors.note_spawned(_spawn)
            return {"ok": False, "reason": "rate_limit", "detail": "x"}

        @errors.local_request_before_spawn()
        async def request():
            await attempt()
            return {"ok": False, "reason": "account_exhausted", "detail": "x"}

        self.assertFalse(errors.is_local_failure(_run(request())))

    def test_codex_selection_deadline_is_not_stamped(self):
        # Account selection runs the codex binary for its quota check, so a
        # selection deadline can be vendor time; other providers stamp it.
        timeout = {"ok": False, "reason": "timeout",
                   "detail": "account selection deadline exceeded"}

        @errors.local_request_before_spawn(unstamped_reasons=("timeout",))
        async def codex_like():
            return dict(timeout)

        @errors.local_request_before_spawn()
        async def claude_like():
            return dict(timeout)

        self.assertFalse(errors.is_local_failure(_run(codex_like())))
        self.assertTrue(errors.is_local_failure(_run(claude_like())))


class TestAgySdkGate(Env):
    def test_sdk_lane_gate_refusal_never_cools(self):
        from providers import agy
        deny = os.path.join(self.root, "deny.txt")
        with open(deny, "w", encoding="utf-8") as handle:
            handle.write("SYNTHETIC-DENIED-FILENAME\n")
        started = []

        class FakeAgent:
            def __init__(self, *a, **k):
                started.append("agent")

        model = "Synthetic Gemini"
        env = {"OMNI_EGRESS_GATE": "1", "OMNI_EGRESS_DENYLIST": deny}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(agy, "HAS_SDK", True), \
                mock.patch.object(agy, "Agent", FakeAgent, create=True), \
                mock.patch.object(agy, "build_agy_sdk_config", return_value=object()):
            for _ in range(3):
                res = _run(dispatch.dispatch_single(
                    "allowed prompt", model, 5,
                    ["/synthetic/SYNTHETIC-DENIED-FILENAME.txt"], PROFILE))
                self.assertEqual(res.get("reason"), "runtime_error", res)
                self.assertIn("OMNI-EGRESS-DENY", res.get("detail", ""), res)
                self.assertNotIn(MARKER, res)
        self.assertEqual(started, [])
        self.assertIsNone(health._read_ledger_unlocked().get(health.pool_key(model, PROFILE)))


class TestSpawnBoundaryIsChildCreation(SeatEnv):
    def test_every_lane_reports_the_child_to_run_cli(self):
        for model, modname in SEATS:
            with self.subTest(provider=modname):
                seen = []

                async def run_cli(argv, **kwargs):
                    seen.append(kwargs.get("on_spawn"))
                    return {"returncode": 0, "stdout": "", "stderr": ""}
                self.run_seat(model, modname, run_cli)
                self.assertTrue(callable(seen[0]), seen)

    def test_exception_after_the_child_exists_counts_as_vendor(self):
        for model, modname in SEATS:
            with self.subTest(provider=modname):
                health.clear_all()

                async def run_cli(argv, **kwargs):
                    kwargs["on_spawn"](object())     # the child now exists
                    raise MemoryError("buffering child output")
                for _ in range(3):
                    res = self.run_seat(model, modname, run_cli)
                    self.assertEqual(res.get("reason"), "runtime_error", res)
                    self.assertNotIn(MARKER, res)
                entry = self.raw(model)
                self.assertEqual(entry["crash_streak"], 3, entry)
                self.assertEqual(entry["source"], "crash", entry)

    def test_spawn_oserror_before_the_hook_stays_local(self):
        for model, modname in SEATS:
            with self.subTest(provider=modname):
                health.clear_all()

                async def run_cli(argv, **kwargs):
                    raise PermissionError(13, "cannot exec", argv[0])
                for _ in range(3):
                    self.run_seat(model, modname, run_cli)
                self.assertIsNone(self.raw(model))


if __name__ == "__main__":
    unittest.main()
