"""Grok consult lane: no tools, and nothing released before the verdict.

A fake grok CLI interprets the consult argv the way grok 1.0.41 does (live
2026-09-30; its stream shape was re-captured on 1.0.52, where the ``end``
event also reports ``cwd``): an empty ``--tools`` is unset (every tool); a non-empty
allowlist replaces the default set but grok still injects ``search_tool`` /
``use_tool``; ``--disallowed-tools`` runs last. It answers in the requested
``--output-format`` and, like grok, writes a per-session tool record under
``$GROK_HOME/sessions/<quoted cwd>/<session id>/``. A ``vendor`` knob lets a
test model a grok release that ignores the tool flags, and ``record`` lets it
model a record that is missing, stale, symlinked or rewritten by the child.
"""
import asyncio
import json
import os
import re
import shutil
import tempfile
import unittest
import urllib.parse
from unittest import mock

import providers.cli_run as cli_run
from providers import grok as grok_mod

GROK_DEFAULT_TOOLS = [
    "run_terminal_command", "read_file", "search_replace", "write",
    "todo_write", "search_tool", "use_tool",
]
SECRET = "SECRET-REPLY-7Q2"


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def offered_tools(argv, vendor="honest"):
    """The tool set grok offers for *argv* (see module docstring)."""
    tools = list(GROK_DEFAULT_TOOLS)
    allow = _flag(argv, "--tools")
    if allow and vendor != "ignores_tool_flags":
        tools = [t for t in allow.split(",") if t] + ["search_tool", "use_tool"]
    deny = _flag(argv, "--disallowed-tools")
    if deny and vendor not in ("ignores_tool_flags", "ignores_denylist"):
        tools = [t for t in tools if t not in deny.split(",")]
    return tools


def grok_stdout(argv, tools, text=SECRET):
    fmt = _flag(argv, "--output-format")
    sid = _flag(argv, "--session-id") or "no-session-id"
    usage = {"input_tokens": 7, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0, "output_tokens": 2}
    if fmt == "streaming-json":
        lines = [{"type": "available_commands", "tools": tools},
                 {"type": "thought", "data": "hidden reasoning"},
                 {"type": "text", "data": text},
                 {"type": "end", "stopReason": "end_turn", "sessionId": sid,
                  "cwd": _flag(argv, "--cwd") or "", "usage": usage}]
        return "".join(json.dumps(line) + "\n" for line in lines)
    return json.dumps({"text": text, "usage": usage})


def write_record(grok_home, cwd, sid, tools, record):
    sessions = os.path.join(grok_home, "sessions")
    here = os.path.join(sessions, urllib.parse.quote(os.path.realpath(cwd), safe=""), sid)
    if record == "missing":
        return
    if record == "stale_other_cwd":
        # A [] left by another run under the same id; this run wrote nothing.
        stale = os.path.join(sessions, "%2Fsome%2Fother%2Fcwd", sid)
        os.makedirs(stale)
        with open(os.path.join(stale, "tool_definitions.json"), "w") as f:
            json.dump([], f)
        return
    os.makedirs(here)
    path = os.path.join(here, "tool_definitions.json")
    if record == "symlink_to_empty":
        decoy = os.path.join(grok_home, "decoy.json")
        with open(decoy, "w") as f:
            json.dump([], f)
        os.symlink(decoy, path)
        return
    with open(path, "w") as f:
        json.dump([] if record == "rewritten_empty" else tools, f)


class FakeGrok:
    def __init__(self, grok_home, vendor="honest", record="honest",
                 returncode=0, stderr=""):
        self.grok_home = grok_home
        self.vendor = vendor
        self.record = record
        self.returncode = returncode
        self.stderr = stderr
        self.argv = None

    async def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.argv = argv
        tools = offered_tools(argv, self.vendor)
        write_record(self.grok_home, kwargs.get("cwd") or _flag(argv, "--cwd"),
                     _flag(argv, "--session-id") or "none", tools, self.record)
        return {"returncode": self.returncode, "stdout": grok_stdout(argv, tools),
                "stderr": self.stderr, "timed_out": False,
                "signal": -self.returncode if self.returncode < 0 else None}


class ConsultContainmentBase(unittest.TestCase):
    def setUp(self):
        self.grok_home = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_home_"))
        self.addCleanup(shutil.rmtree, self.grok_home, True)
        for patcher in (
                mock.patch.dict(os.environ, {"GROK_HOME": self.grok_home}),
                # Hermetic: no host wrap, so argv reaches the fake verbatim.
                mock.patch("providers.host_fence.platform.system", new=lambda: "Linux"),
                mock.patch.object(grok_mod.shutil, "which", return_value="/fake/bin/grok")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def consult(self, fake):
        with mock.patch.object(grok_mod, "run_cli", fake):
            return asyncio.run(grok_mod.run_grok_subprocess("synthetic", "grok-4.5", 30))

    def assertWithheld(self, res):
        self.assertFalse(res["ok"], res)
        self.assertNotIn(SECRET, json.dumps(res))
        self.assertNotIn("hidden reasoning", json.dumps(res))


class TestConsultToolEvidence(ConsultContainmentBase):
    def test_large_complete_prompt_uses_verbatim_and_preserves_tail(self):
        tail = "SOURCE_RECEIPT: complete-prompt-tail-7Q2"
        prompt = "complete public source\n" + ("source line\n" * 48000) + tail
        seen = {}

        class CompletePrompt(FakeGrok):
            async def __call__(self, argv, **kwargs):
                with open(_flag(argv, "--prompt-file"), "rb") as f:
                    seen["prompt"] = f.read()
                return await super().__call__(argv, **kwargs)

        fake = CompletePrompt(self.grok_home)
        with mock.patch.object(grok_mod, "run_cli", fake):
            res = asyncio.run(grok_mod.run_grok_subprocess(prompt, "grok-4.7", 30))
        self.assertTrue(res["ok"], res)
        self.assertGreater(len(seen["prompt"]), 526330)
        self.assertEqual(seen["prompt"], prompt.encode("utf-8"))
        self.assertTrue(seen["prompt"].endswith(tail.encode("utf-8")))
        self.assertIn("--verbatim", fake.argv)
        self.assertEqual(offered_tools(fake.argv), [])
        self.assertIn("--no-memory", fake.argv)
        self.assertIn("--disable-web-search", fake.argv)
        self.assertEqual(_flag(fake.argv, "--output-format"), "streaming-json")
        self.assertEqual(res["text"], SECRET)

    def test_structured_prompt_preserves_content_and_uses_verbatim(self):
        prompt = "Review the complete source fixture"
        source = "# complete fixture\nlast-line-marker-7Q2\n"
        with tempfile.TemporaryDirectory(prefix="omni_grok_source_") as root:
            path = os.path.join(root, "source.md")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(source)
            fake = FakeGrok(self.grok_home)
            with mock.patch.dict(os.environ, {"OMNI_LLM_SOURCE_ROOTS": root}), \
                    mock.patch.object(grok_mod, "run_cli", fake):
                res = asyncio.run(grok_mod.run_grok_subprocess(
                    prompt, "grok-4.7", 30, files=[path]))
        self.assertTrue(res["ok"], res)
        self.assertIn("--verbatim", fake.argv)
        self.assertNotIn("--prompt-file", fake.argv)
        payload = json.loads(_flag(fake.argv, "--prompt-json"))
        self.assertEqual(payload["content"][0], {"type": "text", "text": prompt})
        self.assertEqual(payload["content"][1]["resource"]["text"], source)
        self.assertEqual(offered_tools(fake.argv), [])
        self.assertIn("--disable-web-search", fake.argv)
        self.assertEqual(res["text"], SECRET)

    def test_f11_honest_grok_given_consult_flags_offers_no_tools(self):
        fake = FakeGrok(self.grok_home)
        res = self.consult(fake)
        self.assertEqual(offered_tools(fake.argv), [])
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["text"], SECRET)
        self.assertNotIn("hidden reasoning", res["text"])

    def test_f11_blank_tools_flag_would_offer_every_tool(self):
        # The emulator is not tautological: the pre-fix argv exposes tools.
        self.assertEqual(offered_tools(["grok", "--tools", ""]), GROK_DEFAULT_TOOLS)

    def test_vendor_ignoring_the_denylist_is_refused(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_denylist",
                                    record="honest"))
        self.assertWithheld(res)
        self.assertEqual(res["reason"], "tool_containment_failed")

    def test_f1_stale_empty_record_from_another_cwd_is_not_evidence(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_tool_flags",
                                    record="stale_other_cwd"))
        self.assertWithheld(res)

    def test_f2_symlinked_empty_record_is_not_evidence(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_tool_flags",
                                    record="symlink_to_empty"))
        self.assertWithheld(res)

    def test_f3_record_rewritten_by_the_child_is_not_evidence(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_tool_flags",
                                    record="rewritten_empty"))
        self.assertWithheld(res)


class TestConsultNothingReleasedEarly(ConsultContainmentBase):
    def test_f4_nonzero_exit_never_puts_stdout_in_detail(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_tool_flags",
                                    returncode=1))
        self.assertWithheld(res)

    def test_f4_signal_exit_never_puts_stdout_in_detail(self):
        res = self.consult(FakeGrok(self.grok_home, vendor="ignores_tool_flags",
                                    returncode=-9))
        self.assertWithheld(res)
        self.assertEqual(res["reason"], "signal_exit")

    def test_f4_auth_reason_survives_without_stdout(self):
        res = self.consult(FakeGrok(self.grok_home, returncode=1,
                                    stderr="error: please run `grok login` first"))
        self.assertEqual(res["reason"], "auth_error")
        self.assertIn("grok login", res["detail"])
        self.assertNotIn(SECRET, json.dumps(res))

    def test_f5_watch_pane_never_sees_stdout(self):
        seen = {}

        async def fake_open(session, tab_id, log_path, done_path, pane_name, max_s):
            seen["log"] = log_path
            return "terminal_1"

        def keep_workdir(session, pane_id, work_dir):
            seen["work_dir"] = work_dir

        async def fake_spawn(*argv, **kwargs):
            argv = [str(a) for a in argv]
            tools = offered_tools(argv, "ignores_tool_flags")
            from tests.test_server import _FakeGrokProc
            return _FakeGrokProc(stdout=grok_stdout(argv, tools).encode(),
                                 stderr=b"", returncode=0)

        token = cli_run.set_watch_context(
            watch_pane=True, zellij_session="fixture", zellij_tab="fixture")
        try:
            with mock.patch.object(cli_run, "list_zellij_sessions", return_value=["fixture"]), \
                    mock.patch.object(cli_run, "list_zellij_tabs",
                                      return_value=[{"name": "fixture", "tab_id": 0}]), \
                    mock.patch.object(cli_run, "_open_watch_pane", fake_open), \
                    mock.patch.object(cli_run, "_schedule_watch_finish", keep_workdir), \
                    mock.patch.object(cli_run, "_spawn", fake_spawn):
                res = asyncio.run(grok_mod.run_grok_subprocess("synthetic", "grok-4.5", 30))
        finally:
            cli_run.reset_watch_context(token)
            if seen.get("work_dir"):
                self.addCleanup(shutil.rmtree, seen["work_dir"], True)
        self.assertIn("log", seen, "watch pane was not opened")
        with open(seen["log"], encoding="utf-8") as f:
            pane = f.read()
        self.assertNotIn(SECRET, pane)
        self.assertNotIn("hidden reasoning", pane)
        self.assertWithheld(res)



class TestConsultStreamVerdict(unittest.TestCase):
    SID = "11111111-2222-4333-8444-555555555555"

    def verdict(self, *events, raw=None):
        out = raw if raw is not None else "".join(json.dumps(e) + "\n" for e in events)
        return grok_mod._consult_result(out, self.SID)

    def good(self, **end):
        return [{"type": "available_commands", "tools": []},
                {"type": "thought", "data": "hidden reasoning"},
                {"type": "text", "data": "PO"}, {"type": "text", "data": "NG"},
                {"type": "usage", "usage": {"input_tokens": 1}},
                {"type": "end", "stopReason": "end_turn", "sessionId": self.SID,
                 "usage": {"input_tokens": 3, "output_tokens": 1}, **end}]

    def test_clean_stream_releases_only_text(self):
        text, usage, refusal = self.verdict(*self.good())
        self.assertIsNone(refusal)
        self.assertEqual(text, "PONG")
        self.assertEqual(usage["output_tokens"], 1)

    def test_refusals(self):
        good = self.good()
        cases = {
            "tool call": (good[:2] + [{"type": "tool_call", "toolName": "read_file"}] + good[2:],
                          "tool_containment_failed"),
            "tool update": (good[:2] + [{"type": "tool_call_update"}] + good[2:],
                            "tool_containment_failed"),
            "late tool announcement": (good[:3] + [{"type": "available_commands",
                                                    "tools": ["write"]}] + good[3:],
                                       "tool_containment_failed"),
            "no announcement": (good[1:], "tool_containment_unverified"),
            "unknown event": (good[:2] + [{"type": "plan"}] + good[2:],
                              "tool_containment_unverified"),
            "other session": (good[:-1] + [dict(good[-1], sessionId="other")],
                              "tool_containment_unverified"),
            "no end": (good[:-1], "tool_containment_unverified"),
            "event after end": (good + [{"type": "text", "data": "x"}],
                                "tool_containment_unverified"),
            "two ends": (good + [good[-1]], "tool_containment_unverified"),
            "announcement not a list": ([{"type": "available_commands"}] + good[1:],
                                        "tool_containment_unverified"),
        }
        for label, (events, reason) in cases.items():
            with self.subTest(case=label):
                text, _usage, refusal = self.verdict(*events)
                self.assertIsNone(text)
                self.assertEqual(refusal["reason"], reason)
                self.assertNotIn("PO", refusal["detail"])
        _t, _u, refusal = self.verdict(raw='{"text": "PONG"}\nnot json\n')
        self.assertEqual(refusal["reason"], "tool_containment_unverified")


# Every refusal detail the verdict may emit: fixed wording and integers only.
_REFUSAL_DETAIL = re.compile(
    r"\Agrok consult (?:stream not verified: (?:non-JSON stdout line"
    r"|non-object stdout line|unexpected event type|tool announcement without a list"
    r"|no tool-set announcement|stream does not end with one end event"
    r"|end event names another session|duplicate key in stdout line"
    r"|unexpected event fields|event data is not a string|usage is not an object"
    r"|command announcement is not a name list"
    r"|end working directory is not a string"
    r"|end event stop reason is not end_turn)"
    r"|stream carried [0-9]+ tool event\(s\)|was offered [0-9]+ tool\(s\))"
    r"; reply withheld\Z")


class TestConsultStreamVerdictRound2(TestConsultStreamVerdict):
    LEAK = "SECRET-REPLY-7Q2"

    def assertFixedDetail(self, refusal):
        self.assertEqual(set(refusal), {"ok", "reason", "detail"}, refusal)
        self.assertIn(refusal["reason"], ("tool_containment_failed",
                                          "tool_containment_unverified"))
        self.assertRegex(refusal["detail"], _REFUSAL_DETAIL)

    def test_r3_3_every_refusal_detail_is_fixed_wording(self):
        good = self.good()
        leak = self.LEAK
        cases = [
            good[:2] + [{"type": "tool_call", "title": leak, "toolName": leak}] + good[2:],
            good[:2] + [{"type": "tool_" + leak}] + good[2:],
            [{"type": "available_commands", "tools": [leak, "write"]}] + good[1:],
            good[:2] + [{"type": leak}] + good[2:],
            [{"type": "available_commands", "tools": leak}] + good[1:],
            good[:-1] + [dict(good[-1], sessionId=leak)],
            good[:-1] + [dict(good[-1], sessionId="Z" + self.SID)],
            good[1:], good[:-1], good + [good[-1]], good + [{"type": "text", "data": leak}],
        ]
        for events in cases:
            with self.subTest(events=events[:3]):
                _t, _u, refusal = self.verdict(*events)
                self.assertFixedDetail(refusal)
        for raw in (leak + "\n", json.dumps([leak]) + "\n"):
            _t, _u, refusal = self.verdict(raw=raw)
            self.assertFixedDetail(refusal)

    def test_r2_1_refusal_detail_carries_no_stream_string(self):
        good = self.good()
        leak = self.LEAK
        cases = [
            good[:2] + [{"type": "tool_call", "title": leak, "toolName": leak}] + good[2:],
            good[:2] + [{"type": "tool_" + leak}] + good[2:],
            [{"type": "available_commands", "tools": [leak, "write"]}] + good[1:],
            good[:2] + [{"type": leak}] + good[2:],
            [{"type": "available_commands", "tools": leak}] + good[1:],
            good[:-1] + [dict(good[-1], sessionId=leak)],
        ]
        for events in cases:
            with self.subTest(events=events[:3]):
                _t, _u, refusal = self.verdict(*events)
                self.assertIsNotNone(refusal)
                self.assertNotIn(leak, json.dumps(refusal))
        _t, _u, refusal = self.verdict(raw=leak + "\n")
        self.assertNotIn(leak, json.dumps(refusal))

    def test_r2_5_unicode_line_separators_inside_json_strings_are_data(self):
        good = self.good()
        good[2] = {"type": "text", "data": "A\u2028B\u2029C\x85D\x1cE"}
        out = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in good)
        text, _usage, refusal = grok_mod._consult_result(out, self.SID)
        self.assertIsNone(refusal)
        self.assertEqual(text, "A\u2028B\u2029C\x85D\x1cENG")

    def test_r2_6_released_text_is_the_exact_concatenation(self):
        good = self.good()
        good[2] = {"type": "text", "data": " \nOK"}
        good[3] = {"type": "text", "data": "\n "}
        text, _usage, refusal = self.verdict(*good)
        self.assertIsNone(refusal)
        self.assertEqual(text, " \nOK\n ")

    def test_r2_6_whitespace_only_reply_is_still_empty_response(self):
        fake_home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, fake_home, True)
        good = self.good()
        good[2] = {"type": "text", "data": " \n"}
        good[3] = {"type": "text", "data": ""}

        async def fake_run(argv, **kwargs):
            sid = argv[argv.index("--session-id") + 1]
            events = good[:-1] + [dict(good[-1], sessionId=sid)]
            return {"returncode": 0, "stderr": "",
                    "stdout": "".join(json.dumps(e) + "\n" for e in events)}

        with mock.patch.dict(os.environ, {"GROK_HOME": fake_home}), \
                mock.patch("providers.host_fence.platform.system", new=lambda: "Linux"), \
                mock.patch.object(grok_mod.shutil, "which", return_value="/fake/bin/grok"), \
                mock.patch.object(grok_mod, "run_cli", fake_run):
            res = asyncio.run(grok_mod.run_grok_subprocess("synthetic", "grok-4.5", 30))
        self.assertEqual(res["reason"], "empty_response", res)

    def test_r2_6_consult_releases_exact_text_end_to_end(self):
        class Exact(FakeGrok):
            async def __call__(self, argv, **kwargs):
                res = await FakeGrok.__call__(self, argv, **kwargs)
                res["stdout"] = res["stdout"].replace(json.dumps(SECRET), json.dumps(" \n" + SECRET + "\n "))
                return res
        base = ConsultContainmentBase("run")
        base.setUp()
        self.addCleanup(base.doCleanups)
        res = base.consult(Exact(base.grok_home))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["text"], " \n" + SECRET + "\n ")


class TestConsultStreamSchemaRound5(TestConsultStreamVerdictRound2):
    """Grok Reviewer-B findings G1-G3 and U3 (review of aa4f82c)."""

    def lines(self, *raw):
        return grok_mod._consult_result("".join(r + "\n" for r in raw), self.SID)

    def end_line(self, **extra):
        end = {"type": "end", "stopReason": "end_turn", "sessionId": self.SID,
               "usage": {"input_tokens": 3, "output_tokens": 1}}
        end.update(extra)
        return json.dumps(end)

    def test_g1_duplicate_keys_refuse_at_any_depth(self):
        announce = json.dumps({"type": "available_commands", "tools": []})
        cases = [
            '{"type":"tool_call","toolName":"run_terminal_command",'
            '"rawInput":{"command":"id"},"type":"text","data":"PWNED"}',
            '{"type":"text","data":"a","data":"PWNED"}',
            '{"type":"text","data":"ok","extra":{"k":1,"k":2}}',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                text, _usage, refusal = self.lines(announce, raw, self.end_line())
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)
                self.assertNotIn("PWNED", json.dumps(refusal))
        # duplicate inside end.usage, and a top-level non-object line
        text, _u, refusal = self.lines(
            announce, '{"type":"text","data":"ok"}',
            '{"type":"end","stopReason":"end_turn","sessionId":"%s",'
            '"usage":{"input_tokens":1,"input_tokens":2}}' % self.SID)
        self.assertIsNone(text)
        self.assertFixedDetail(refusal)
        for raw in ('"just a string"', "[1, 2]", "3"):
            _t, _u, refusal = self.lines(announce, raw, self.end_line())
            self.assertFixedDetail(refusal)

    def test_g2_announcement_shape_is_explicit(self):
        text_line = json.dumps({"type": "text", "data": "ok"})
        bad = [
            {"type": "available_commands", "tools": [], "commands": [{"name": "shell"}]},
            {"type": "available_commands", "tools": [], "mcp": ["x"]},
            {"type": "available_commands", "tools": [], "commands": "shell"},
            {"type": "available_commands", "tools": [], "commands": [1]},
            {"type": "available_commands", "commands": []},
        ]
        for announce in bad:
            with self.subTest(announce=announce):
                text, _u, refusal = self.lines(json.dumps(announce), text_line, self.end_line())
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)
        # Live shape (grok 1.0.41 consult, inference blocked locally): tools
        # empty, commands = slash-command names. That is released.
        ok = {"type": "available_commands", "tools": [], "commands": ["compact", "goal"]}
        text, _u, refusal = self.lines(json.dumps(ok), text_line, self.end_line())
        self.assertIsNone(refusal)
        self.assertEqual(text, "ok")

    def test_g3_usage_carries_only_integer_counters(self):
        announce = json.dumps({"type": "available_commands", "tools": []})
        end = self.end_line(usage={"input_tokens": 1, "output_tokens": 2,
                                   "reasoning_tokens": 3, "rawOutput": "TOOL-SECRET",
                                   "total_tokens": "7"})
        text, usage, refusal = self.lines(announce, '{"type":"text","data":"ok"}', end)
        self.assertIsNone(refusal)
        self.assertNotIn("TOOL-SECRET", json.dumps(usage))
        raw = usage.get("raw") or {}
        self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool)
                            for v in raw.values()), raw)
        self.assertEqual(raw, {"input_tokens": 1, "output_tokens": 2, "reasoning_tokens": 3})

    def test_u3_stop_reason_other_than_end_turn_refuses(self):
        announce = json.dumps({"type": "available_commands", "tools": []})
        text_line = '{"type":"text","data":"ok"}'
        for end in (self.end_line(stopReason="tool_use"), self.end_line(stopReason=None),
                    json.dumps({"type": "end", "sessionId": self.SID})):
            with self.subTest(end=end):
                text, _u, refusal = self.lines(announce, text_line, end)
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)
        usage_event = json.dumps({"type": "usage", "stopReason": "tool_use",
                                  "usage": {"input_tokens": 1}})
        text, _u, refusal = self.lines(announce, usage_event, text_line, self.end_line())
        self.assertIsNone(text)
        self.assertFixedDetail(refusal)

    def test_end_cwd_is_accepted_only_as_a_string(self):
        # grok 1.0.52 adds `cwd` to `end`: the run's working directory, an
        # echo of --cwd. A string is inert; any other type is refused.
        announce = json.dumps({"type": "available_commands", "tools": []})
        text_line = '{"type":"text","data":"ok"}'
        text, _u, refusal = self.lines(
            announce, text_line, self.end_line(cwd="/tmp/grok_cwd_placeholder"))
        self.assertIsNone(refusal)
        self.assertEqual(text, "ok")
        for bad in (None, 7, ["/tmp/x"], {"path": "/tmp/x"}, True):
            with self.subTest(cwd=bad):
                text, _u, refusal = self.lines(announce, text_line, self.end_line(cwd=bad))
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)
        # `cwd` is an `end` field only: no other event may carry it.
        for raw in ('{"type":"text","data":"ok","cwd":"/tmp/x"}',
                    '{"type":"usage","usage":{"input_tokens":1},"cwd":"/tmp/x"}'):
            with self.subTest(raw=raw):
                text, _u, refusal = self.lines(announce, raw, self.end_line())
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)

    def test_1052_stream_shape_is_released(self):
        # Shape captured on grok 1.0.52 (placeholders for ids/paths/signature).
        announce = json.dumps({"type": "available_commands", "tools": [],
                               "commands": ["compact", "goal"]})
        events = [announce, announce,
                  '{"type":"text","data":"P"}', '{"type":"text","data":"ONG"}',
                  announce,
                  json.dumps({"type": "usage", "signature": "sig-placeholder",
                              "usage": {"input_tokens": 1, "output_tokens": 1,
                                        "cache_read_input_tokens": 0,
                                        "cache_creation_input_tokens": 0,
                                        "reasoning_tokens": 0}}),
                  self.end_line(cwd="/tmp/grok_cwd_placeholder", requestId="req-placeholder",
                                num_turns=1, total_cost_usd=0.0, total_cost_usd_ticks=0,
                                modelUsage={"m": {"inputTokens": 1}})]
        text, _u, refusal = self.lines(*events)
        self.assertIsNone(refusal)
        self.assertEqual(text, "PONG")

    def test_nonempty_tools_still_refuses_with_cwd_present(self):
        text_line = '{"type":"text","data":"ok"}'
        end = self.end_line(cwd="/tmp/grok_cwd_placeholder")
        for tools in (["shell"], [{"name": "shell"}], ["todo_write"]):
            with self.subTest(tools=tools):
                announce = json.dumps({"type": "available_commands", "tools": tools,
                                       "commands": ["compact"]})
                text, _u, refusal = self.lines(announce, text_line, end)
                self.assertIsNone(text)
                self.assertEqual(refusal["reason"], "tool_containment_failed")

    def test_unknown_end_field_still_refuses_beside_cwd(self):
        announce = json.dumps({"type": "available_commands", "tools": []})
        text_line = '{"type":"text","data":"ok"}'
        for extra in ({"toolCalls": []}, {"workdir": "/tmp/x"}, {"env": {}}):
            with self.subTest(extra=extra):
                text, _u, refusal = self.lines(
                    announce, text_line, self.end_line(cwd="/tmp/x", **extra))
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)

    def test_unknown_fields_on_known_events_refuse(self):
        announce = json.dumps({"type": "available_commands", "tools": []})
        for raw in ('{"type":"text","data":"ok","toolName":"x"}',
                    '{"type":"thought","data":"x","rawInput":{}}',
                    '{"type":"text","data":7}'):
            with self.subTest(raw=raw):
                text, _u, refusal = self.lines(announce, raw, self.end_line())
                self.assertIsNone(text)
                self.assertFixedDetail(refusal)

if __name__ == "__main__":
    unittest.main()
