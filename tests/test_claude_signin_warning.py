#!/usr/bin/env python3
# no-opencc
"""A claude sign-in expiry WARNING is not an auth failure.

The claude CLI prints a notice on stderr while the sign-in is still valid:
"this sign-in ends between <date> and <date> ... Run claude auth login before
then". The text carries the auth markers "auth" and "login", so classifying
stderr verbatim turned any non-zero exit that happened to carry the notice
into ``auth_error`` (with the notice as the detail), and an ``auth_error``
benches the seat as ``account_startup``.

Pinned here, with a stubbed CLI only:
  (a) warning on stderr + exit 0 + a valid result  -> ok, seat not benched
  (b) warning + a genuine auth failure              -> still auth_error
  (c) warning + a different genuine failure         -> that failure's reason,
      and the detail is not the warning

Hermetic: ``run_cli`` and the vendor binary lookup are mocked, nothing spawns.
"""
import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from providers import claude, errors, health
from tests import offline_home
import tests.test_provider_accounts as tpa

MODEL = "claude/synthetic-model"
PROFILE = "default"

WARNING = (
    "Warning: this sign-in ends between 2026-11-16 and 2026-11-19 "
    "(in 6 days at the earliest). Renewing it does not move those dates. "
    "Run claude auth login before then, or use claude setup-token for "
    "unattended jobs."
)
# The same notice wrapped over several lines.
WARNING_WRAPPED = (
    "Warning: this sign-in ends between 2026-11-16 and 2026-11-19\n"
    "(in 6 days at the earliest). Renewing it does not move those dates.\n"
    "Run claude auth login before then, or use claude setup-token for\n"
    "unattended jobs."
)

HARD_EXPIRY = ("Failed to authenticate: this login reached its time limit and "
               "cannot be renewed. Run claude auth login to sign in again.")

_HOME = None


def setUpModule():
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _ok_stdout(text="synthetic answer"):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "result": text, "usage": {"input_tokens": 3, "output_tokens": 2}})


class Env(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-signin-warn-")
        self.addCleanup(temp.cleanup)
        env = {"PATH": os.environ.get("PATH", ""),
               "OMNI_LLM_STATE_DIR": os.path.realpath(temp.name),
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_NOTIFY_QUIET": "1"}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def call(self, rc, stdout, stderr):
        async def fake_cli(argv, **kwargs):
            return {"returncode": rc, "stdout": stdout, "stderr": stderr,
                    "timed_out": False, "signal": None}
        with mock.patch.object(claude, "resolve_agent_binary",
                               return_value="/synthetic/claude"), \
                mock.patch.object(claude, "prepare_host_fence",
                                  side_effect=lambda argv, **k: (argv, k["env"], {})), \
                mock.patch.object(claude, "run_cli", side_effect=fake_cli):
            return asyncio.run(claude.run_claude_subprocess(
                "synthetic task", "synthetic-model", 5))

    def benched(self):
        return health.pool_key("claude", PROFILE) in health.snapshot()


class TestSignInWarning(Env):
    def test_a_warning_with_a_valid_result_is_ok_and_does_not_bench(self):
        for warning in (WARNING, WARNING_WRAPPED):
            res = self.call(0, _ok_stdout(), warning)
            self.assertIs(res.get("ok"), True, res)
            self.assertEqual(res.get("text"), "synthetic answer")
            self.assertNotEqual(res.get("reason"), "auth_error", res)
            self.assertFalse(self.benched(), health.snapshot())

    def test_a_warning_with_an_empty_reply_does_not_show_the_warning(self):
        res = self.call(0, json.dumps({"result": ""}), WARNING)
        self.assertEqual(res.get("reason"), "empty_response", res)
        self.assertNotIn("sign-in", res.get("detail", ""), res)
        self.assertFalse(self.benched(), health.snapshot())

    def test_a_warning_alone_on_a_failed_exit_is_not_an_auth_error(self):
        for warning in (WARNING, WARNING_WRAPPED):
            res = self.call(1, "", warning)
            self.assertIs(res.get("ok"), False, res)
            self.assertEqual(res.get("reason"), "runtime_error", res)
            self.assertNotIn("sign-in", res.get("detail", ""), res)
            self.assertFalse(self.benched(), health.snapshot())

    def test_b_a_real_auth_failure_with_the_warning_is_still_auth_error(self):
        stdout = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                             "api_error_status": 401,
                             "result": "Invalid authentication credentials"})
        for out, err in ((stdout, WARNING), ("", "Not logged in - run /login\n" + WARNING)):
            health.clear_all()
            res = self.call(1, out, err)
            self.assertEqual(res.get("reason"), "auth_error", res)
            self.assertNotIn("sign-in ends", res.get("detail", ""), res)
            self.assertTrue(self.benched())

    def test_c_a_different_failure_with_the_warning_keeps_its_own_reason(self):
        cases = (
            # plain-text stderr failure after the notice
            (1, "", WARNING + "\nsegfault in renderer", "runtime_error", "segfault"),
            # in-band rate limit
            (1, json.dumps({"type": "result", "is_error": True, "api_error_status": 429,
                            "result": "Usage limit reached"}), WARNING, "rate_limit", "Usage limit"),
            # in-band runtime failure
            (1, json.dumps({"type": "result", "is_error": True,
                            "result": "boom in the model"}), WARNING, "runtime_error", "boom"),
        )
        for rc, out, err, reason, cause in cases:
            health.clear_all()
            res = self.call(rc, out, err)
            self.assertEqual(res.get("reason"), reason, res)
            self.assertIn(cause, res.get("detail", ""), res)
            self.assertNotIn("sign-in ends", res.get("detail", ""), res)
            if reason == "runtime_error":
                self.assertFalse(self.benched(), health.snapshot())


class TestStripHelper(unittest.TestCase):
    def test_only_the_notice_is_removed(self):
        self.assertEqual(errors.strip_signin_expiry_warning(WARNING), "")
        self.assertEqual(errors.strip_signin_expiry_warning(WARNING_WRAPPED), "")
        self.assertEqual(
            errors.strip_signin_expiry_warning("before\n" + WARNING + "\nafter"),
            "before\nafter")
        self.assertEqual(errors.strip_signin_expiry_warning("Not logged in"), "Not logged in")
        self.assertEqual(errors.strip_signin_expiry_warning(None), "")


class TestStripFailsSafe(Env):
    """The strip removes only an exact notice; when unsure it keeps the text."""

    def test_i_a_real_error_line_quoting_the_phrase_is_kept(self):
        err = ("Error: this sign-in ends between 2026-11-16 and 2026-11-19. "
               "Authentication failed.")
        self.assertEqual(errors.strip_signin_expiry_warning(err), err)
        res = self.call(1, "", err)
        self.assertEqual(res.get("reason"), "auth_error", res)
        self.assertTrue(self.benched())

    def test_ii_a_real_failure_line_inside_the_wrapped_notice_is_kept(self):
        # The notice is complete (it ends with "unattended jobs."), but a real
        # failure line sits inside it: nothing matches, so nothing is removed.
        err = ("Warning: this sign-in ends between 2026-11-16 and 2026-11-19\n"
               "Not logged in - run /login\n"
               "(in 6 days at the earliest). Renewing it does not move those dates.\n"
               "Run claude auth login before then, or use claude setup-token for\n"
               "unattended jobs.")
        self.assertEqual(errors.strip_signin_expiry_warning(err), err)
        res = self.call(1, "", err)
        self.assertEqual(res.get("reason"), "auth_error", res)
        self.assertTrue(self.benched())

    def test_ii_b_a_reworded_notice_does_not_eat_a_later_real_failure(self):
        reworded = ("Warning: this sign-in ends between 2026-11-16 and 2026-11-19. "
                    "Run claude auth login before then.")
        err = (reworded + "\nNot logged in - run /login\n"
               "hint: use setup-token for unattended jobs and cron")
        kept = errors.strip_signin_expiry_warning(err)
        self.assertIn("Not logged in", kept)
        res = self.call(1, "", err)
        self.assertEqual(res.get("reason"), "auth_error", res)
        self.assertTrue(self.benched())

    def test_an_extra_word_or_line_inside_the_notice_keeps_the_text(self):
        words = WARNING.split(" ")
        cases = [" ".join(words[:i] + ["extra"] + words[i:])
                 for i in range(1, len(words))]
        cases.append(WARNING.replace("Renewing it", "Renewing\n\nit"))
        cases.append(WARNING.replace("in 6 days", "in six days"))
        cases.append(WARNING.replace("2026-11-16", "2026-11"))
        cases.append(WARNING + " Please retry.")
        for err in cases:
            self.assertEqual(errors.strip_signin_expiry_warning(err), err)

    def test_the_real_one_line_notice_and_a_wrap_at_any_word_are_stripped(self):
        self.assertEqual(errors.strip_signin_expiry_warning(WARNING), "")
        words = WARNING.split(" ")
        for i in range(1, len(words)):
            wrapped = " ".join(words[:i]) + "\n" + " ".join(words[i:])
            self.assertEqual(errors.strip_signin_expiry_warning(wrapped), "", wrapped)
        self.assertEqual(
            errors.strip_signin_expiry_warning(WARNING.replace("6 days", "1 day")), "")

    def test_ansi_colour_codes_between_tokens_are_stripped(self):
        sgr = ("Warning: \x1b[33mthis sign-in ends between \x1b[1m2026-11-16\x1b[0m "
               "and 2026-11-19 (in 6 days at the earliest). Renewing it does not "
               "move those dates. Run \x1b[36mclaude auth login\x1b[0m before "
               "then, or use claude setup-token for unattended jobs.")
        self.assertEqual(errors.strip_signin_expiry_warning(sgr), "")
        self.assertEqual(
            errors.strip_signin_expiry_warning("before\n" + sgr + "\nafter"),
            "before\nafter")

    def test_iii_a_notice_with_ansi_colour_codes_is_stripped(self):
        coloured = "\x1b[33m" + WARNING + "\x1b[0m"
        inner = "\x1b[1mWarning:\x1b[0m" + WARNING[len("Warning:"):]
        split = "Warn\x1b[33ming:" + WARNING[len("Warning:"):]
        for notice in (coloured, inner, split):
            self.assertEqual(errors.strip_signin_expiry_warning(notice), "")
            self.assertEqual(
                errors.strip_signin_expiry_warning("before\n" + notice + "\nafter"),
                "before\nafter")
        res = self.call(1, "", coloured)
        self.assertEqual(res.get("reason"), "runtime_error", res)
        self.assertFalse(self.benched(), health.snapshot())

    def test_iv_an_over_long_or_blank_line_split_notice_is_kept(self):
        head = "Warning: this sign-in ends between 2026-11-16 and 2026-11-19 "
        too_long = (head + "Run claude auth login before then. " + ("padding " * 80)
                    + "unattended jobs.")
        split = (head + "(in 6 days)\n\nRun claude auth login before then, or use "
                 "claude setup-token for unattended jobs.")
        for notice in (too_long, split):
            err = notice + "\nNot logged in"
            self.assertEqual(errors.strip_signin_expiry_warning(err), err)
            res = self.call(1, "", err)
            self.assertEqual(res.get("reason"), "auth_error", res)

    def test_v_the_remaining_stderr_keeps_its_blank_lines_and_crlf(self):
        text = "first\r\n\r\n" + WARNING + "\r\nsecond\n\n\nthird\r\n"
        self.assertEqual(errors.strip_signin_expiry_warning(text),
                         "first\r\n\r\nsecond\n\n\nthird\r\n")
        wrapped = "a\n\n" + WARNING_WRAPPED.replace("\n", "\r\n") + "\r\n\r\nb"
        self.assertEqual(errors.strip_signin_expiry_warning(wrapped), "a\n\n\r\nb")
        # The text without a notice is returned untouched.
        plain = "x\r\n\r\n  y \n\n"
        self.assertEqual(errors.strip_signin_expiry_warning(plain), plain)

    def test_vi_the_hard_expiry_sentence_is_still_auth_error(self):
        for err in (HARD_EXPIRY, WARNING + "\n" + HARD_EXPIRY,
                    HARD_EXPIRY + "\n" + WARNING):
            health.clear_all()
            res = self.call(1, "", err)
            self.assertEqual(res.get("reason"), "auth_error", res)
            self.assertIn("Failed to authenticate", res.get("detail", ""), res)
            self.assertNotIn("sign-in ends", res.get("detail", ""), res)
            self.assertTrue(self.benched())


class TestAgentLane(tpa.ApiFixture):
    """The agent lane classifies the same stderr, so it reads the same way."""

    def setUp(self):
        super().setUp()
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_a_warning_with_a_valid_result_is_ok(self):
        self.respond(a={"stderr": WARNING})
        res = self.agent()
        self.assertIs(res.get("ok"), True, res)
        self.assertEqual(self.ledger("a"), [])

    def test_a_warning_alone_on_a_failed_exit_is_not_an_auth_error(self):
        self.respond(a={"stdout": "", "rc": 1, "stderr": WARNING})
        res = self.agent()
        self.assertEqual(res.get("reason"), "runtime_error", res)
        self.assertNotIn("sign-in ends", res.get("detail", ""), res)

    def test_b_a_real_auth_failure_with_the_warning_is_still_auth_error(self):
        self.respond(a={"stdout": "", "rc": 1,
                        "stderr": "Not logged in - run /login\n" + WARNING})
        res = self.agent()
        self.assertEqual(res.get("reason"), "auth_error", res)
        self.assertNotIn("sign-in ends", res.get("detail", ""), res)

    def test_c_a_different_failure_with_the_warning_keeps_its_own_reason(self):
        self.respond(a={"stdout": "", "rc": 1,
                        "stderr": WARNING + "\nsegfault in renderer"})
        res = self.agent()
        self.assertEqual(res.get("reason"), "runtime_error", res)
        self.assertIn("segfault", res.get("detail", ""), res)
        self.assertNotIn("sign-in ends", res.get("detail", ""), res)


if __name__ == "__main__":
    unittest.main()
