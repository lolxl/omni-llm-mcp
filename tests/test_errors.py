# no-opencc
"""classify_cli_error: signal_exit is decided from the integer alone."""
import asyncio
import json
import unittest
from unittest import mock

from providers import codex as codex_mod
from providers.errors import (
    classify_cli_error,
    codex_error_text,
    codex_failure_detail,
    signal_exit_detail,
    signal_name,
)


_LOGIN_AGENT_TEXT = ("login credential " * 16).strip()
_LOGIN_AGENT_STDOUT = json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": _LOGIN_AGENT_TEXT},
}) + "\n"
_SKILLS_STDERR = (
    "ERROR codex_skills_extension::loader::host: failed to walk skills root: "
    "Operation not permitted (os error 1)"
)
_AUTH_EVENT_STDOUT = json.dumps({
    "type": "error",
    "message": "401 Unauthorized",
}) + "\n"
_QUOTA_AGENT_STDOUT = json.dumps({
    "type": "item.completed",
    "item": {
        "type": "agent_message",
        "text": "we hit the quota and rate limit",
    },
}) + "\n"


def _codex_fail_envelope(stdout, stderr, returncode=1):
    """Drive providers.codex's non-zero-exit classifier through run_cli."""
    async def fake_run_cli(argv, **kwargs):
        return {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
        }

    with mock.patch.object(codex_mod.shutil, "which",
                           return_value="/fake/bin/codex"), \
            mock.patch.object(codex_mod, "prepare_host_fence",
                                  side_effect=lambda argv, **kw: (argv, kw.get("env") or {}, {})), \
            mock.patch.object(codex_mod, "run_cli", fake_run_cli):
        return asyncio.run(codex_mod.run_codex_subprocess(
            "hi", "gpt-5.6-sol", 30))


class ClassifyCliErrorTest(unittest.TestCase):
    def test_negative_rc_is_signal_exit_before_any_text(self):
        self.assertEqual(
            classify_cli_error("login please", "", -9), "signal_exit")
        self.assertEqual(
            classify_cli_error("", "quota exceeded 429", -11), "signal_exit")

    def test_rc_1_with_login_is_still_auth_error(self):
        self.assertEqual(
            classify_cli_error("login please", "", 1), "auth_error")

    def test_minus_one_is_not_signal_exit(self):
        self.assertEqual(classify_cli_error("boom", "", -1), "runtime_error")
        self.assertEqual(
            classify_cli_error("login please", "", -1), "auth_error")

    def test_none_and_zero_fall_through_to_text(self):
        self.assertEqual(classify_cli_error("", "boom", None), "runtime_error")
        self.assertEqual(classify_cli_error("", "boom", 0), "runtime_error")


class SignalNameTest(unittest.TestCase):
    def test_known_signals(self):
        self.assertEqual(signal_name(-9), "SIGKILL(9)")
        self.assertEqual(signal_name(-11), "SIGSEGV(11)")

    def test_unknown_signal_falls_back(self):
        self.assertEqual(signal_name(-99), "signal 99")

    def test_none_and_non_negative_are_signal_question(self):
        self.assertEqual(signal_name(None), "signal ?")
        self.assertEqual(signal_name(0), "signal ?")
        self.assertEqual(signal_name(1), "signal ?")
        self.assertEqual(signal_name("nope"), "signal ?")

    def test_detail_names_cli_and_keeps_partial_output(self):
        detail = signal_exit_detail("claude", -9, "partial-out", "")
        self.assertIn("SIGKILL(9)", detail)
        self.assertIn("partial-out", detail)
        self.assertTrue(detail.startswith("claude killed by"))
        detail_err = signal_exit_detail("grok", -9, "stdout-bit", "stderr-bit")
        self.assertIn("stderr-bit", detail_err)
        self.assertNotIn("stdout-bit", detail_err)

    def test_detail_caps_whole_string_at_300(self):
        long_out = "x" * 400
        detail = signal_exit_detail("claude", -9, long_out, "")
        self.assertEqual(len(detail), 300)
        self.assertTrue(detail.startswith("claude killed by SIGKILL(9)"))
        self.assertIn("x", detail)


class CodexCallSiteClassificationTest(unittest.TestCase):
    def test_agent_message_login_with_skills_stderr_is_runtime_error(self):
        res = _codex_fail_envelope(_LOGIN_AGENT_STDOUT, _SKILLS_STDERR, 1)
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "runtime_error")

    def test_agent_message_login_with_empty_stderr_is_runtime_error(self):
        res = _codex_fail_envelope(_LOGIN_AGENT_STDOUT, "", 1)
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "runtime_error")

    def test_error_event_401_is_auth_error(self):
        res = _codex_fail_envelope(_AUTH_EVENT_STDOUT, "", 1)
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "auth_error")

    def test_agent_message_quota_is_not_rate_limit(self):
        res = _codex_fail_envelope(_QUOTA_AGENT_STDOUT, "", 1)
        self.assertFalse(res.get("ok"), res)
        self.assertNotEqual(res.get("reason"), "rate_limit")
        self.assertEqual(res.get("reason"), "runtime_error")


class CodexFailureDetailTest(unittest.TestCase):
    # stderr can carry the caller's prompt echoed back, and a prompt can
    # imitate a codex log line, so no failure detail may come from stderr.
    _PROMPT = "SYNTHETIC-PROMPT-SENTINEL"
    _KEY = "sk-" + "k" * 40
    _STDERR = (
        _PROMPT + " fix the auth bug\n"
        "2026-10-10T03:55:02.1Z ERROR codex_core::client: " + _PROMPT + " " + _KEY + "\n")

    def envelope(self, stdout, returncode=1, signal=None, stderr=None, prompt="hi"):
        """codex consult through run_cli, host fence passed through."""
        async def fake_run_cli(argv, **kwargs):
            res = {"returncode": returncode, "stdout": stdout,
                   "stderr": self._STDERR if stderr is None else stderr,
                   "timed_out": False}
            if signal is not None:
                res["signal"] = signal
            return res
        with mock.patch.object(codex_mod.shutil, "which", return_value="/fake/bin/codex"), \
                mock.patch.object(codex_mod, "prepare_host_fence",
                                  side_effect=lambda argv, **kw: (argv, kw.get("env") or {}, {})), \
                mock.patch.object(codex_mod, "run_cli", fake_run_cli):
            return asyncio.run(codex_mod.run_codex_subprocess(prompt, "gpt-5.6-sol", 30))

    def assert_clean(self, res):
        self.assertFalse(res.get("ok"), res)
        detail = res.get("detail") or ""
        self.assertNotIn(self._PROMPT, detail)
        self.assertNotIn("sk-k", detail)
        self.assertLessEqual(len(detail), 300)
        return detail

    def test_error_event_is_the_detail_and_stderr_never_is(self):
        self.assertEqual(self.assert_clean(self.envelope(_AUTH_EVENT_STDOUT)), "401 Unauthorized")
        self.assertEqual(self.assert_clean(self.envelope("")), "exit 1")

    def test_signal_exit_carries_no_stderr(self):
        detail = self.assert_clean(self.envelope("", returncode=-9, signal=9))
        self.assertIn("SIGKILL", detail)

    def test_empty_response_carries_no_stderr(self):
        res = self.envelope("", returncode=0)
        self.assertEqual(res.get("reason"), "empty_response", res)
        self.assertEqual(self.assert_clean(res), "codex returned no text")

    def test_secrets_are_masked_before_the_cut(self):
        # A key straddling the cut: cutting first leaves "sk-" plus a few
        # characters, too short for the mask, so part of the key would leak.
        event = json.dumps({"type": "error", "message": "x" * 290 + " sk-" + "a" * 40})
        detail = codex_failure_detail(event, 1)
        self.assertNotIn("sk-a", detail)
        self.assertLessEqual(len(detail), 300)

    def test_stderr_classifies_only_a_run_without_events(self):
        # Once codex has spoken in events they decide: stderr text (a prompt,
        # a tool's output) must not turn a transport failure into an account
        # failure and a cooldown, however it is phrased.
        event = json.dumps({"type": "error", "message": "transport failed"})
        for stderr in ("fix the login bug in auth.json\n", "user: quota exceeded\n",
                       "Error: not logged in, run codex login\n"):
            with self.subTest(stderr=stderr):
                res = self.envelope(event, stderr=stderr)
                self.assertEqual(res.get("reason"), "runtime_error", res)
        # Codex's structured error codes classify without any message text
        # (after tool activity this is the only source: the agent-lane test
        # test_actual_valid_activity_then_error_remains_terminal pins it).
        for code, reason in (("usage_limit_reached", "rate_limit"),
                             ("rate_limit_exceeded", "rate_limit"),
                             ("authentication_failed", "auth_error")):
            with self.subTest(code=code):
                coded = json.dumps({"type": "error", "code": code})
                self.assertEqual(self.envelope(coded, stderr="").get("reason"), reason)
        # Refresh-token failures arrive as message-only events.
        for message in ("Your access token could not be refreshed because your refresh "
                        "token was already used. Please log out and sign in again.",
                        "Your access token could not be refreshed because your refresh "
                        "token was revoked. Please log out and sign in again."):
            with self.subTest(message=message[60:90]):
                ev = json.dumps({"type": "error", "message": message})
                self.assertEqual(self.envelope(ev, stderr="").get("reason"), "auth_error")
        # Legacy msg-wrapped events count as events: stderr stays out.
        legacy = json.dumps({"id": 1, "msg": {"type": "error", "message": "transport failed"}})
        res = self.envelope(legacy, stderr="Error: not logged in, run codex login\n")
        self.assertEqual(res.get("reason"), "runtime_error", res)
        # A run that failed before any event (not signed in) still classifies.
        res = self.envelope("", stderr="Error: not logged in, run codex login\n")
        self.assertEqual(res.get("reason"), "auth_error", res)

    def test_no_event_names_the_exit_code(self):
        self.assertEqual(codex_failure_detail("", 7), "exit 7")
        self.assertEqual(codex_failure_detail("", None), "exit -1")


class CodexErrorTextProvenanceTest(unittest.TestCase):
    """Only error nodes are read: never the event's own message/text when
    an error key or an error-typed item is what made it an error."""

    def test_sibling_message_of_an_error_payload_is_not_read(self):
        event = {"type": "turn.failed", "message": "PROMPT_SENTINEL",
                 "error": {"message": "transport failed"}}
        self.assertEqual(codex_error_text(json.dumps(event)), "transport failed")

    def test_sibling_message_of_an_error_item_is_not_read(self):
        event = {"type": "item.completed", "message": "PROMPT_SENTINEL",
                 "item": {"type": "error", "message": "command failed"}}
        self.assertEqual(codex_error_text(json.dumps(event)), "command failed")


class CodexErrorTextTest(unittest.TestCase):
    def test_codex_0154_top_level_error_event_keeps_401(self):
        text = codex_error_text(json.dumps({
            "type": "error",
            "message": "request failed: 401 Unauthorized",
        }))
        self.assertTrue(text)
        self.assertIn("401", text)

    def test_codex_0154_item_error_keeps_401(self):
        text = codex_error_text(json.dumps({
            "type": "item.completed",
            "item": {"type": "error", "message": "upstream 401 Unauthorized"},
        }))
        self.assertTrue(text)
        self.assertIn("401", text)

    def test_codex_0154_turn_failed_error_object_keeps_401(self):
        text = codex_error_text(json.dumps({
            "type": "turn.failed",
            "error": {"message": "401 Unauthorized"},
        }))
        self.assertTrue(text)
        self.assertIn("401", text)

    def test_old_proto_wrapper_msg_error_is_boom(self):
        self.assertEqual(
            codex_error_text(json.dumps({
                "id": 1,
                "msg": {"type": "error", "message": "boom"},
            })),
            "boom",
        )

    def test_null_error_on_reasoning_item_is_empty(self):
        self.assertEqual(
            codex_error_text(json.dumps({
                "type": "item.completed",
                "error": None,
                "item": {
                    "type": "reasoning",
                    "text": "the default login directory",
                },
            })),
            "",
        )

    def test_error_event_does_not_harvest_sibling_item_text(self):
        text = codex_error_text(json.dumps({
            "type": "error",
            "message": "boom",
            "item": {
                "type": "assistant_message",
                "text": "please login",
            },
        }))
        self.assertEqual(text, "boom")
        self.assertNotIn("login", text)

    def test_agent_message_quota_rate_limit_is_empty(self):
        self.assertEqual(
            codex_error_text(json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "quota 429 rate limit",
                },
            })),
            "",
        )


if __name__ == "__main__":
    unittest.main()
