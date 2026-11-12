#!/usr/bin/env python3
# no-opencc
"""Grok stall watchdog and timeout output (agent lane + run_cli).

Hermetic: the "grok" is a small python child that prints canned lines and then
idles. The real run_cli drives it (agent_lane.run_cli is replaced by a bridge
that swaps argv for the fake child and forwards every other keyword), so the
watchdog kwargs, the kill and the envelope are the production ones. No vendor
CLI, no network.
"""
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from providers import agent_lane, cli_run, config as config_mod  # noqa: E402
from providers import known_secrets, stream_contract  # noqa: E402
from tests import offline_home  # noqa: E402

_REAL_WHICH = shutil.which
_REAL_RUN_CLI = cli_run.run_cli
# Timeout envelope keys at base: nothing grok-specific.
BASE_TIMEOUT_KEYS = {"ok", "reason", "detail", "workspace_writes", "fence",
                     "fence_detail", "account_retry_safe",
                     "effort", "effort_source", "effort_note"}
BASE_RUN_CLI_KEYS = {"returncode", "stdout", "stderr", "timed_out", "watched",
                     "signal"}
SECRET = "mZ4qT9vL" + "2xW7pR5n" + "K8dJ3hB6"

_STATE_PATCH = None
_STATE_DIR = None
_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module. It goes
    # first (and closes last): the state-dir patch below must not be scrubbed.
    global _HOME
    _HOME = offline_home.use_synthetic_home()
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    try:
        _STATE_PATCH.stop()
        shutil.rmtree(_STATE_DIR, ignore_errors=True)
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _fake_which(name, *a, **kw):
    if os.path.basename(str(name)) in ("codex", "claude", "grok", "pi"):
        return "/fake/bin/" + os.path.basename(str(name))
    return _REAL_WHICH(name, *a, **kw)


def assistant(text):
    return json.dumps({"type": "assistant",
                       "message": {"content": [{"type": "text", "text": text}]}})


def result(**extra):
    return json.dumps(dict({"type": "result", "subtype": "success",
                            "is_error": False}, **extra))


def child(*, lines=(), tail="", pre_sleep=None, close_pipes=False, drip=None,
          sleep=60):
    """Python source of a fake grok: print ``lines`` (one per line) then
    ``tail`` unterminated, optionally drip one byte every 0.5 s for ``drip``
    seconds, optionally close its pipes, then idle for ``sleep`` seconds."""
    return (
        "import os, sys, time\n"
        "out = sys.stdout\n"
        "for l in %r:\n    out.write(l + '\\n'); out.flush()\n"
        "out.write(%r); out.flush()\n"
        "if %r:\n"
        "    end = time.time() + %r\n"
        "    while time.time() < end:\n"
        "        out.write('.'); out.flush(); time.sleep(0.5)\n"
        "if %r:\n    os.close(1); os.close(2)\n"
        "time.sleep(%r)\n"
    ) % (list(lines), tail, bool(drip), drip or 0, close_pipes, sleep)


class Bridge:
    """agent_lane.run_cli stand-in: runs the fake child under the real run_cli."""

    def __init__(self, code):
        self.code = code
        self.calls = []

    async def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return await _REAL_RUN_CLI([sys.executable, "-u", "-c", self.code], **kwargs)


class Canned:
    """agent_lane.run_cli stand-in that returns a fixed result dict."""

    def __init__(self, res):
        self.res = res
        self.calls = []

    async def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return dict(self.res)


@mock.patch("platform.system", new=lambda: "Linux")
class LaneCase(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        known_secrets.clear()
        self.addCleanup(known_secrets.clear)

    def mkdir(self):
        d = os.path.realpath(tempfile.mkdtemp(prefix="omni_stall_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def mkrepo(self):
        """A checkout on a work branch with one empty commit (writes countable)."""
        d = self.mkdir()
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="t",
                   GIT_AUTHOR_EMAIL="t@test.invalid", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@test.invalid")
        for args in (["init", "-b", "work"], ["commit", "--allow-empty", "-m", "x"]):
            subprocess.run(["git"] + args, cwd=d, env=env, check=True,
                           capture_output=True)
        return d

    def agent(self, runner, *, model="grok-4.6", job_lane=True, wall=30,
              grace=None, workdir=None, **kw):
        env = {"OMNI_LLM_DEPTH": ""}
        if grace is not None:
            env["OMNI_LLM_GROK_STALL_GRACE_S"] = str(grace)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, env))
            os.environ.pop("OMNI_LLM_DEPTH", None)
            if grace is None:
                os.environ.pop("OMNI_LLM_GROK_STALL_GRACE_S", None)
            stack.enter_context(mock.patch.object(agent_lane, "run_cli", runner))
            stack.enter_context(mock.patch.object(shutil, "which", _fake_which))
            stack.enter_context(mock.patch.object(
                agent_lane, "_workdir_fence", return_value=None))
            t = time.monotonic()
            res = asyncio.run(agent_lane.run_agent(
                task="t", model=model, workdir=workdir or self.mkdir(),
                timeout_s=wall,
                job_lane=job_lane or None, fence_requested=False, **kw))
            self.wall_seen = time.monotonic() - t
        return res


class TestStall(LaneCase):
    def test_success_result_then_hang_is_ok_with_stall_termination(self):
        bridge = Bridge(child(lines=[assistant("the answer"), result()]))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertTrue(env["ok"], env)
        self.assertEqual(env["text"], "the answer")
        self.assertEqual(env["termination"]["reason"], "stall")
        self.assertIs(env["termination"]["marker"], True)
        self.assertIsInstance(env["termination"]["elapsed_s"], int)
        self.assertLess(self.wall_seen, 10)
        self.assertLess(env["termination"]["elapsed_s"], 10)
        kw = bridge.calls[0][1]
        self.assertEqual(kw["stall_grace_s"], 1.0)
        self.assertTrue(callable(kw["stall_marker"]))

    def test_error_result_then_hang_is_a_stall_failure_with_partial_reply(self):
        bridge = Bridge(child(lines=[
            assistant("half"), result(is_error=True, result="boom")]))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertFalse(env["ok"], env)
        self.assertEqual(env["reason"], "stall")
        self.assertEqual(env["partial_reply"], "half")
        self.assertIn("workspace_writes", env)
        self.assertIn("boom", env["detail"])
        self.assertEqual(env["termination"]["reason"], "stall")
        self.assertLess(self.wall_seen, 10)

    def test_non_success_subtype_is_a_failure(self):
        bridge = Bridge(child(lines=[
            assistant("x"), result(subtype="error_max_turns")]))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertFalse(env["ok"], env)
        self.assertEqual(env["reason"], "stall")

    def test_null_subtype_is_a_failure(self):
        # An explicit "subtype": null is not "success"; only an absent key is.
        bridge = Bridge(child(lines=[assistant("x"), result(subtype=None)]))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertFalse(env["ok"], env)
        self.assertEqual(env["reason"], "stall")
        self.assertLess(self.wall_seen, 10)

    def test_result_prefix_extended_by_garbage_is_not_a_stall(self):
        # `{"type":"result"}` alone on the pending line matches, but the same
        # line then continues with `garbage` before its newline: it was never a
        # result line, so the run is not stall-killed and ends at the wall.
        code = (
            "import sys, time\n"
            "sys.stdout.write(%r + '\\n'); sys.stdout.flush()\n"
            "sys.stdout.write('{\"type\":\"result\"}'); sys.stdout.flush()\n"
            "time.sleep(0.3)\n"
            "sys.stdout.write('garbage\\n'); sys.stdout.flush()\n"
            "time.sleep(60)\n") % assistant("a")
        env = self.agent(Bridge(code), grace=1, wall=4)
        self.assertEqual(env["reason"], "timeout", env)
        self.assertIs(env["termination"]["marker"], False)
        self.assertGreaterEqual(self.wall_seen, 3.5)

    def test_result_without_subtype_counts_as_success(self):
        line = json.dumps({"type": "result", "result": "done text"})
        env = self.agent(Bridge(child(lines=[line])), grace=1, wall=30)
        self.assertTrue(env["ok"], env)
        # no assistant message and no deltas: the result field is the reply
        self.assertEqual(env["text"], "done text")

    def test_activity_after_the_result_line_postpones_the_kill(self):
        bridge = Bridge(child(lines=[assistant("a"), result()], drip=3))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertTrue(env["ok"], env)
        self.assertGreaterEqual(env["termination"]["elapsed_s"], 3, env)
        self.assertLess(self.wall_seen, 15)

    def test_result_line_without_trailing_newline_still_stalls(self):
        bridge = Bridge(child(lines=[assistant("a")], tail=result()))
        env = self.agent(bridge, grace=1, wall=30)
        self.assertTrue(env["ok"], env)
        self.assertEqual(env["termination"]["reason"], "stall")
        self.assertLess(self.wall_seen, 10)

    def test_stall_success_still_honours_expect_writes(self):
        bridge = Bridge(child(lines=[assistant("a"), result()]))
        env = self.agent(bridge, grace=1, wall=30, expect_writes=True,
                         workdir=self.mkrepo())
        self.assertFalse(env["ok"], env)
        self.assertEqual(env["reason"], "no_workspace_writes")
        self.assertEqual(env["termination"]["reason"], "stall")

    def test_stall_success_with_empty_reply_is_empty_response(self):
        line = json.dumps({"type": "result", "subtype": "success"})
        env = self.agent(Bridge(child(lines=[line])), grace=1, wall=30)
        self.assertEqual(env["reason"], "empty_response", env)
        self.assertEqual(env["termination"]["reason"], "stall")

    def test_watchdog_trace_is_emitted_once(self):
        bridge = Bridge(child(lines=[assistant("a"), result()]))
        with mock.patch.object(agent_lane, "log_event") as log:
            self.agent(bridge, grace=1, wall=30)
        traces = [c for c in log.call_args_list if c.args[0] == "omni-watchdog"]
        self.assertEqual(len(traces), 1, log.call_args_list)
        self.assertEqual(traces[0].kwargs["provider"], "grok")
        self.assertEqual(traces[0].kwargs["reason"], "stall")
        self.assertEqual(traces[0].kwargs["marker"], "yes")
        self.assertIsInstance(traces[0].kwargs["elapsed"], int)


LS = "\u2028"  # JSON.stringify / ensure_ascii=False leave it unescaped


def raw_json(obj):
    return json.dumps(obj, ensure_ascii=False)


class TestLfFraming(LaneCase):
    """The grok helpers frame stdout on LF only, like the watchdog: a line
    whose JSON string holds U+2028 (or another splitlines() separator) is one
    line, not fragments."""

    def stalled_stdout(self):
        return "\n".join([
            raw_json({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "first%ssecond" % LS}]}}),
            raw_json({"type": "result", "subtype": "success",
                      "is_error": False, "result": "a%sb" % LS}),
        ]) + "\n"

    def test_result_frame_with_a_literal_u2028_is_found(self):
        out = self.stalled_stdout()
        self.assertTrue(agent_lane._grok_result_line(out.split("\n")[1]))
        frame = agent_lane._last_grok_result_frame(out)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["result"], "a%sb" % LS)
        self.assertTrue(agent_lane._grok_result_ok(frame))

    def test_result_frame_with_other_splitlines_separators(self):
        for sep in ("\u2029", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e",
                    "\x85", "\r"):
            frame = agent_lane._last_grok_result_frame(
                raw_json({"type": "result", "result": "a%sb" % sep}) + "\n")
            self.assertIsNotNone(frame, repr(sep))
            self.assertEqual(frame["result"], "a%sb" % sep)

    def test_crlf_output_still_frames(self):
        out = "\r\n".join([assistant("x"), result()]) + "\r\n"
        self.assertTrue(agent_lane._grok_result_ok(
            agent_lane._last_grok_result_frame(out)))
        self.assertEqual(agent_lane._partial_reply(True, out)["partial_reply"], "x")

    def test_partial_reply_keeps_the_whole_text_around_a_u2028(self):
        out = raw_json({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "before%safter" % LS}]}}) + "\n"
        self.assertEqual(agent_lane._partial_reply(True, out),
                         {"partial_reply": "before%safter" % LS})

    def test_stalled_run_with_u2028_in_its_frames_is_not_a_stall_failure(self):
        # End to end, stall classification only: the success frame was
        # recognised, so the run is not reason "stall". Reply recovery with
        # these separators is TestJobLaneReplyLfFraming.
        code = (
            "import sys, time\n"
            "sys.stdout.buffer.write(%r); sys.stdout.flush()\n"
            "time.sleep(60)\n") % self.stalled_stdout().encode("utf-8")
        env = self.agent(Bridge(code), grace=1, wall=30)
        self.assertNotEqual(env.get("reason"), "stall", env)
        self.assertEqual(env["termination"]["reason"], "stall")
        self.assertLess(self.wall_seen, 10)


class TestJobLaneReplyLfFraming(LaneCase):
    """Shared job-lane reply recovery frames stdout on LF only (issue #210)."""

    def finished(self, stdout):
        return Canned({"returncode": 0, "stdout": stdout, "stderr": "",
                       "timed_out": False, "signal": None})

    def test_stdout_has_events_sees_a_line_holding_u2028(self):
        self.assertTrue(agent_lane._stdout_has_events(
            raw_json({"type": "result", "result": "a%sb" % LS}) + "\n"))

    def test_grok_reply_with_splitlines_separators_is_recovered(self):
        for sep in (LS, "\u2029", "\x0b", "\x0c", "\x1c", "\x85"):
            with self.subTest(sep=repr(sep)):
                out = "\n".join([
                    raw_json({"type": "assistant", "message": {"content": [
                        {"type": "text", "text": "a%sb" % sep}]}}),
                    raw_json({"type": "result", "subtype": "success",
                              "is_error": False, "result": "a%sb" % sep}),
                ]) + "\n"
                env = self.agent(self.finished(out))
                self.assertTrue(env.get("ok"), env)
                self.assertEqual(env["text"], "a%sb" % sep)

    def test_crlf_stdout_still_recovers_the_reply(self):
        env = self.agent(self.finished("\r\n".join([assistant("x"), result()]) + "\r\n"))
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn('"type"', env["text"])


def codex_message(text):
    return raw_json({"type": "item.completed",
                     "item": {"id": "item_1", "type": "agent_message",
                              "text": text}})


class TestCodexReplyLfFraming(LaneCase):
    """codex reply recovery frames stdout on LF only (issue #213). codex's
    serde_json writes U+2028, U+2029 and U+0085 raw inside JSON strings."""

    def finished(self, stdout):
        return Canned({"returncode": 0, "stdout": stdout, "stderr": "",
                       "timed_out": False, "signal": None})

    def codex(self, stdout, **kw):
        return self.agent(self.finished(stdout), model="codex/gpt-5.6-sol", **kw)

    def assertRecovered(self, sep):
        out = "\n".join([
            raw_json({"type": "thread.started", "thread_id": "t1"}),
            codex_message("a%sb" % sep),
        ]) + "\n"
        env = self.codex(out)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], "a%sb" % sep)

    def test_reply_holding_u2028_is_recovered(self):
        self.assertRecovered("\u2028")

    def test_reply_holding_u2029_is_recovered(self):
        self.assertRecovered("\u2029")

    def test_reply_holding_u0085_is_recovered(self):
        self.assertRecovered("\x85")

    def test_reply_holding_the_escaped_separators_is_recovered(self):
        for sep in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e"):
            with self.subTest(sep=repr(sep)):
                self.assertRecovered(sep)

    def test_crlf_stdout_still_recovers_the_reply(self):
        env = self.codex("\r\n".join([
            raw_json({"type": "thread.started", "thread_id": "t1"}),
            codex_message("x")]) + "\r\n")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], "x")


class TestTimeout(LaneCase):
    def test_timeout_without_result_keeps_the_partial_reply(self):
        bridge = Bridge(child(lines=[assistant("partial text so far")]))
        with mock.patch.object(agent_lane, "log_event") as log:
            env = self.agent(bridge, wall=2)
        self.assertFalse(env["ok"], env)
        self.assertEqual(env["reason"], "timeout")
        self.assertIn("partial text so far", env["partial_reply"])
        self.assertEqual(env["termination"]["reason"], "timeout")
        self.assertIs(env["termination"]["marker"], False)
        self.assertGreaterEqual(env["termination"]["elapsed_s"], 1)
        self.assertNotIn("partial_reply_truncated", env)
        traces = [c for c in log.call_args_list if c.args[0] == "omni-watchdog"]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].kwargs["marker"], "no")

    def test_timeout_after_closed_pipes_still_has_the_text(self):
        bridge = Bridge(child(lines=[assistant("said before closing")],
                              close_pipes=True))
        env = self.agent(bridge, wall=2)
        self.assertEqual(env["reason"], "timeout", env)
        self.assertIn("said before closing", env["partial_reply"])
        self.assertLess(self.wall_seen, 15)

    def test_sync_lane_grok_has_no_watchdog_and_keeps_stripped_stdout(self):
        bridge = Bridge(child(tail="plain words\n"))
        env = self.agent(bridge, job_lane=False, wall=2, grace=1)
        self.assertEqual(env["reason"], "timeout", env)
        self.assertEqual(env["partial_reply"], "plain words")
        self.assertEqual(env["termination"],
                         {"reason": "timeout",
                          "elapsed_s": env["termination"]["elapsed_s"],
                          "marker": False})
        kw = bridge.calls[0][1]
        self.assertNotIn("stall_marker", kw)
        self.assertNotIn("stall_grace_s", kw)
        self.assertGreaterEqual(self.wall_seen, 1.5)

    def test_marker_seen_is_reported_when_the_wall_wins(self):
        # grace far beyond the wall: the wall fires first, with the marker seen.
        bridge = Bridge(child(lines=[assistant("a"), result()]))
        env = self.agent(bridge, wall=2, grace=100)
        self.assertEqual(env["reason"], "timeout", env)
        self.assertIs(env["termination"]["marker"], True)

    def test_partial_reply_over_the_cap_keeps_the_tail(self):
        bridge = Bridge(child(tail="H" * 70000 + "TAILEND\n"))
        env = self.agent(bridge, job_lane=False, wall=3)
        cap = agent_lane.PARTIAL_REPLY_MAX_CHARS
        self.assertEqual(cap, 65536)
        self.assertEqual(len(env["partial_reply"]), cap)
        self.assertTrue(env["partial_reply"].endswith("TAILEND"))
        self.assertIs(env["partial_reply_truncated"], True)

    def test_known_secret_is_masked_in_partial_reply(self):
        known_secrets.register(SECRET)
        bridge = Bridge(child(lines=[assistant("key is %s ok" % SECRET)]))
        env = self.agent(bridge, wall=2)
        self.assertEqual(env["reason"], "timeout", env)
        self.assertNotIn(SECRET, env["partial_reply"])
        self.assertNotIn(SECRET[:8], env["partial_reply"])
        self.assertIn("<redacted>", env["partial_reply"])
        # and the same through the egress every envelope leaves by
        from providers.errors import mask_envelope
        self.assertNotIn(SECRET, json.dumps(mask_envelope(env)))

    def test_secret_straddling_the_cap_is_masked_before_the_cut(self):
        known_secrets.register(SECRET)
        cap = agent_lane.PARTIAL_REPLY_MAX_CHARS
        text = "x" * 10 + SECRET + "y" * (cap - 12)
        out = agent_lane._partial_reply(False, text)
        self.assertNotIn(SECRET[-8:], out["partial_reply"])
        self.assertLessEqual(len(out["partial_reply"]), cap)

    def test_timeout_partial_reply_is_the_final_text_of_the_events(self):
        # last whole assistant message wins over earlier ones
        bridge = Bridge(child(lines=[assistant("first"), assistant("second")]))
        env = self.agent(bridge, wall=2)
        self.assertEqual(env["partial_reply"], "second")


class TestOtherProvidersUnchanged(LaneCase):
    def test_non_grok_timeout_envelopes_have_exactly_the_base_keys(self):
        for model in ("claude/sonnet", "codex/gpt-5.6-sol"):
            with self.subTest(model=model):
                canned = Canned({"returncode": -1, "stdout": "partial", "stderr": "",
                                 "timed_out": True, "signal": None})
                env = self.agent(canned, model=model, wall=30)
                self.assertEqual(env["reason"], "timeout", env)
                self.assertEqual(set(env), BASE_TIMEOUT_KEYS, env)
                for _argv, kw in canned.calls:
                    self.assertNotIn("stall_marker", kw)
                    self.assertNotIn("stall_grace_s", kw)

    def test_grok_timeout_adds_only_the_new_keys(self):
        canned = Canned({"returncode": -1, "stdout": "partial", "stderr": "",
                         "timed_out": True, "signal": None})
        env = self.agent(canned, job_lane=False, wall=30)
        self.assertEqual(set(env) - BASE_TIMEOUT_KEYS,
                         {"partial_reply", "termination"}, env)


class TestRunCli(unittest.TestCase):
    def run_cli(self, code, **kw):
        return asyncio.run(cli_run.run_cli(
            [sys.executable, "-u", "-c", code], **kw))

    def test_default_result_has_exactly_the_base_keys(self):
        ok = self.run_cli("print('hi')", timeout_s=10)
        self.assertEqual(set(ok), BASE_RUN_CLI_KEYS)
        slow = self.run_cli("import time; time.sleep(30)", timeout_s=0.5)
        self.assertEqual(set(slow), BASE_RUN_CLI_KEYS)
        self.assertTrue(slow["timed_out"])
        self.assertEqual(slow["returncode"], -1)
        self.assertIsNone(slow["signal"])

    def test_timeout_keeps_what_was_read(self):
        res = self.run_cli(
            "import sys, time\nprint('out-bytes'); sys.stdout.flush()\n"
            "sys.stderr.write('err-bytes'); sys.stderr.flush()\ntime.sleep(30)",
            timeout_s=1)
        self.assertTrue(res["timed_out"])
        self.assertIn("out-bytes", res["stdout"])
        self.assertIn("err-bytes", res["stderr"])

    def test_timeout_after_the_child_closed_its_pipes(self):
        t = time.monotonic()
        res = self.run_cli(
            "import os, sys, time\nprint('before-close'); sys.stdout.flush()\n"
            "os.close(1); os.close(2); time.sleep(30)", timeout_s=1)
        self.assertTrue(res["timed_out"])
        self.assertIn("before-close", res["stdout"])
        self.assertLess(time.monotonic() - t, 15)

    def test_marker_kwargs_add_the_status_keys(self):
        res = self.run_cli("print('hi')", timeout_s=10,
                           stall_marker=lambda line: False, stall_grace_s=1)
        self.assertEqual(set(res) - BASE_RUN_CLI_KEYS,
                         {"stalled", "marker_seen", "elapsed_s"})
        self.assertIs(res["stalled"], False)
        self.assertIs(res["marker_seen"], False)
        self.assertIsInstance(res["elapsed_s"], float)
        slow = self.run_cli("import time; time.sleep(30)", timeout_s=0.5,
                            stall_marker=lambda line: False, stall_grace_s=1)
        self.assertTrue(slow["timed_out"])
        self.assertIs(slow["stalled"], False)
        self.assertIs(slow["marker_seen"], False)

    def test_stall_fires_on_silence_after_the_marker(self):
        res = self.run_cli(
            "import time\nprint('x'); print('MARK', flush=True)\ntime.sleep(60)",
            timeout_s=30, stall_marker=lambda line: line == b"MARK",
            stall_grace_s=1)
        self.assertIs(res["stalled"], True)
        self.assertIs(res["marker_seen"], True)
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], -1)
        self.assertIsNone(res["signal"])
        self.assertLess(res["elapsed_s"], 10)
        self.assertEqual(res["stdout"], "x\nMARK\n")

    def test_marker_line_split_across_reads(self):
        res = self.run_cli(
            "import sys, time\nsys.stdout.write('MA'); sys.stdout.flush()\n"
            "time.sleep(0.3)\nsys.stdout.write('RK\\n'); sys.stdout.flush()\n"
            "time.sleep(60)",
            timeout_s=30, stall_marker=lambda line: line == b"MARK",
            stall_grace_s=1)
        self.assertIs(res["stalled"], True)
        self.assertLess(res["elapsed_s"], 10)

    def test_marker_on_final_line_at_eof(self):
        res = self.run_cli("import sys\nsys.stdout.write('MARK')", timeout_s=10,
                           stall_marker=lambda line: line == b"MARK",
                           stall_grace_s=5)
        self.assertIs(res["marker_seen"], True)
        self.assertIs(res["stalled"], False)
        self.assertEqual(res["returncode"], 0)

    def test_wall_wins_when_it_comes_first(self):
        res = self.run_cli(
            "import time\nprint('MARK', flush=True)\ntime.sleep(60)",
            timeout_s=1, stall_marker=lambda line: line == b"MARK",
            stall_grace_s=50)
        self.assertTrue(res["timed_out"])
        self.assertIs(res["stalled"], False)
        self.assertIs(res["marker_seen"], True)

    def test_marker_and_grace_go_together(self):
        with self.assertRaises(ValueError):
            self.run_cli("pass", stall_marker=lambda line: True)
        with self.assertRaises(ValueError):
            self.run_cli("pass", stall_grace_s=5)
        with self.assertRaises(ValueError):
            self.run_cli("pass", stall_marker=lambda line: True, stall_grace_s=0)


class TestMarkerScan(unittest.TestCase):
    """_MarkerScan on its own: a match on an unterminated line is provisional."""

    def scan(self):
        async def make():
            return cli_run._MarkerScan(lambda line: line == b"MARK")
        return asyncio.run(make())

    def test_partial_match_is_undone_when_the_line_grows(self):
        sc = self.scan()
        sc.feed(b"MARK")
        self.assertTrue(sc.seen)
        self.assertTrue(sc.event.is_set())
        sc.feed(b"garbage")
        self.assertFalse(sc.seen)
        self.assertFalse(sc.event.is_set())
        sc.feed(b"\n")
        sc.finish()
        self.assertFalse(sc.seen)

    def test_partial_match_stays_when_the_line_completes_as_is(self):
        sc = self.scan()
        sc.feed(b"MARK")
        sc.feed(b"\n")
        self.assertTrue(sc.seen)
        self.assertTrue(sc.event.is_set())
        sc.feed(b"more output\n")          # final now: later bytes change nothing
        self.assertTrue(sc.seen)

    def test_partial_match_at_eof_is_final(self):
        sc = self.scan()
        sc.feed(b"MARK")
        sc.finish()
        self.assertTrue(sc.seen)
        sc.feed(b"x")
        self.assertTrue(sc.seen)

    def test_garbled_line_then_a_real_marker_line_matches(self):
        sc = self.scan()
        sc.feed(b"MARK")
        sc.feed(b"garbage\nMARK\n")
        self.assertTrue(sc.seen)

    def test_oversized_line_is_dropped_up_to_its_newline(self):
        with mock.patch.object(cli_run, "_STALL_LINE_CAP", 64):
            sc = self.scan()
            sc.feed(b"x" * 65)               # over the cap: dropped
            sc.feed(b"MARK")                 # still the same long line
            self.assertFalse(sc.seen)
            sc.feed(b"MARK\n")               # same long line, ends here
            self.assertFalse(sc.seen)
            sc.feed(b"MARK\n")               # a new line
            self.assertTrue(sc.seen)


class TestRunCliProvisionalMarker(unittest.TestCase):
    def test_marker_prefix_extended_by_more_bytes_does_not_stall(self):
        code = ("import sys, time\n"
                "sys.stdout.write('MARK'); sys.stdout.flush()\n"
                "time.sleep(0.3)\n"
                "sys.stdout.write('garbage\\n'); sys.stdout.flush()\n"
                "time.sleep(60)\n")
        res = asyncio.run(cli_run.run_cli(
            [sys.executable, "-u", "-c", code], timeout_s=3,
            stall_marker=lambda line: line == b"MARK", stall_grace_s=1))
        self.assertTrue(res["timed_out"], res)
        self.assertIs(res["stalled"], False)
        self.assertIs(res["marker_seen"], False)


class TestGraceEnv(unittest.TestCase):
    def grace(self, value):
        with mock.patch.dict(os.environ, {"OMNI_LLM_GROK_STALL_GRACE_S": value}), \
                mock.patch.object(agent_lane, "log_event") as log:
            return agent_lane._grok_stall_grace(), log

    def test_default_is_720(self):
        self.assertEqual(config_mod.GROK_STALL_GRACE_S, 720)
        with mock.patch.dict(os.environ):
            os.environ.pop("OMNI_LLM_GROK_STALL_GRACE_S", None)
            self.assertEqual(agent_lane._grok_stall_grace(), 720)

    def test_valid_value_is_used(self):
        got, log = self.grace("2.5")
        self.assertEqual(got, 2.5)
        log.assert_not_called()

    def test_invalid_values_fall_back_to_720_and_are_noted_once(self):
        for bad in ("abc", "0", "-5", "inf", "nan", "-inf"):
            with self.subTest(value=bad):
                got, log = self.grace(bad)
                self.assertEqual(got, 720)
                self.assertEqual(log.call_count, 1)
                self.assertEqual(log.call_args.kwargs["value"], bad)

    def test_long_invalid_value_is_not_logged(self):
        got, log = self.grace("z" * 40)
        self.assertEqual(got, 720)
        self.assertEqual(log.call_count, 1)
        self.assertIsNone(log.call_args.kwargs["value"])
        self.assertNotIn("z" * 10, repr(log.call_args))


class TestFinalText(unittest.TestCase):
    RESULT = json.dumps({"type": "result", "result": "from result"})

    def test_result_field_only_when_nothing_else(self):
        self.assertEqual(stream_contract.final_text("grok", [self.RESULT]),
                         "from result")

    def test_assistant_message_outranks_result(self):
        self.assertEqual(stream_contract.final_text(
            "grok", [assistant("whole"), self.RESULT]), "whole")

    def test_deltas_outrank_result(self):
        delta = json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "dd"}}})
        self.assertEqual(stream_contract.final_text(
            "grok", [delta, self.RESULT]), "dd")

    def test_last_result_wins_and_non_string_is_ignored(self):
        other = json.dumps({"type": "result", "result": "later"})
        bad = json.dumps({"type": "result", "result": {"x": 1}})
        self.assertEqual(stream_contract.final_text(
            "grok", [self.RESULT, other]), "later")
        # Only the LAST result event counts: when its "result" is not a string
        # the fallback is empty, not an earlier event's string.
        self.assertEqual(stream_contract.final_text(
            "grok", [self.RESULT, other, bad]), "")
        missing = json.dumps({"type": "result"})
        self.assertEqual(stream_contract.final_text(
            "grok", [self.RESULT, missing]), "")

    def test_claude_and_cursor_are_unchanged(self):
        self.assertEqual(stream_contract.final_text("claude", [self.RESULT]), "")
        self.assertEqual(stream_contract.final_text("cursor", [self.RESULT]),
                         "from result")


if __name__ == "__main__":
    unittest.main()
