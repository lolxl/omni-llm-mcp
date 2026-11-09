#!/usr/bin/env python3
# no-opencc
"""Offline tests for the optional egress refuse layer. No network, no CLIs."""
import asyncio
import contextlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

from providers import agy as agy_mod
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import dispatch
from providers import egress_gate
from providers import grok as grok_mod
from providers import openai_compat
from providers import pi as pi_mod
from providers import cursor as cursor_mod
from providers.egress_gate import check_strings, refuse_payload


# Fixture values used as payloads. Assertions must never see these in error
# text (except the reserved-domain email, which is a pass, not a deny).
TW_MOBILE = "0912345678"
TW_NATIONAL_ID = "A123456789"
LIVE_EMAIL = "alice@gmail.com"
RESERVED_EMAILS = (
    "user@example.com",
    "user@example.net",
    "user@example.org",
    "user@foo.test",
    "user@bar.example",
    "user@baz.invalid",
    "user@qux.localhost",
)
DENY_LITERAL = "super-secret-token"


def _env(gate=None, denylist=None):
    """Patch OMNI_EGRESS_* only. None = unset that var."""
    env = os.environ.copy()
    env.pop("OMNI_EGRESS_GATE", None)
    env.pop("OMNI_EGRESS_DENYLIST", None)
    if gate is not None:
        env["OMNI_EGRESS_GATE"] = gate
    if denylist is not None:
        env["OMNI_EGRESS_DENYLIST"] = denylist
    return mock.patch.dict(os.environ, env, clear=True)


def _detail(envelope):
    return str((envelope or {}).get("detail") or "")


def _fake_which(wanted):
    real = shutil.which

    def which(cmd, *args, **kwargs):
        if cmd == wanted:
            return "/usr/bin/%s" % cmd
        return real(cmd, *args, **kwargs)

    return which


class TestCheckStringsPass(unittest.TestCase):
    def test_clean_payload_has_no_hits(self):
        with _env():
            self.assertEqual(check_strings(["hello world", "no pii here"]), [])
        with _env(gate="1"):
            self.assertIsNone(refuse_payload("hello world", "no pii here"))

    def test_reserved_domain_email_is_not_a_hit(self):
        with _env():
            self.assertEqual(check_strings(RESERVED_EMAILS), [])
        with _env(gate="1"):
            self.assertIsNone(refuse_payload(*RESERVED_EMAILS))


class TestRefuseDenySignature(unittest.TestCase):
    def _assert_deny(self, payload, family, full_value):
        with _env(gate="1"):
            env = refuse_payload(payload)
            hits = check_strings([payload])
        self.assertIsNotNone(env)
        self.assertFalse(env.get("ok"))
        detail = _detail(env)
        self.assertTrue(
            detail.startswith("OMNI-EGRESS-DENY [%s]" % family),
            detail,
        )
        self.assertNotIn(full_value, detail)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], family)
        self.assertNotIn(full_value, hits[0][1])

    def test_deny_tw_mobile(self):
        self._assert_deny("call me at %s please" % TW_MOBILE, "tw_mobile", TW_MOBILE)

    def test_deny_tw_national_id(self):
        self._assert_deny("id=%s" % TW_NATIONAL_ID, "tw_national_id", TW_NATIONAL_ID)

    def test_deny_non_reserved_email(self):
        self._assert_deny("mail %s" % LIVE_EMAIL, "email", LIVE_EMAIL)

    def test_deny_denylist_literal(self):
        fd, path = tempfile.mkstemp(prefix="omni_deny_", suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# comment\n\n%s\n" % DENY_LITERAL)
            with _env(gate="1", denylist=path):
                env = refuse_payload("payload carries %s inside" % DENY_LITERAL)
            self.assertIsNotNone(env)
            detail = _detail(env)
            self.assertTrue(detail.startswith("OMNI-EGRESS-DENY [denylist]"), detail)
            self.assertNotIn(DENY_LITERAL, detail)
        finally:
            os.remove(path)


class TestDefaultOff(unittest.TestCase):
    def test_unset_gate_does_not_refuse_tw_mobile(self):
        # this test documents the honest bare-run gap
        sent = []

        def fake_urlopen(*_a, **_k):
            sent.append(True)
            raise RuntimeError("stubbed send reached")

        with _env(gate=None):
            self.assertIsNone(refuse_payload("call %s" % TW_MOBILE))
            with mock.patch("urllib.request.urlopen", fake_urlopen):
                res = asyncio.run(openai_compat.run_openai_compatible(
                    "call %s" % TW_MOBILE, "gpt-4o", 5, "default"))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(sent, [True], "bare-run (gate off) must still reach send")


class TestFailClosedDenylist(unittest.TestCase):
    def test_missing_denylist_file_refuses(self):
        missing = os.path.join(
            tempfile.gettempdir(), "omni-egress-deny-missing-no-such-file.txt")
        if os.path.exists(missing):
            os.remove(missing)
        with _env(gate="1", denylist=missing):
            env = refuse_payload("clean payload")
        self.assertIsNotNone(env)
        self.assertFalse(env.get("ok"))
        self.assertIn("OMNI-EGRESS-DENY", _detail(env))


class TestSeamBeforeSend(unittest.TestCase):
    # Every gated consult CLI/HTTP seam. Agent/job lanes stay declared-ungated.
    _SEAMS = (
        ("grok",
         "providers.grok.run_cli", "providers.grok.shutil.which", "grok",
         lambda p: grok_mod.run_grok_subprocess(p, "grok-4.6", 5, "default")),
        ("codex",
         "providers.codex.run_cli", "providers.codex.shutil.which", "codex",
         lambda p: codex_mod.run_codex_subprocess(p, "gpt-5.6-sol", 5, "default")),
        ("claude",
         "providers.claude.run_cli", "providers.claude.shutil.which", "claude",
         lambda p: claude_mod.run_claude_subprocess(p, "opus", 5, "default")),
        ("pi",
         "providers.pi.run_cli", "providers.pi.shutil.which", "pi",
         lambda p: pi_mod.run_pi_subprocess(p, "xai/grok-4.6", 5, "default")),
        ("cursor",
         "providers.cursor.run_cli", "providers.cursor.shutil.which",
         "cursor-agent",
         lambda p: cursor_mod.run_cursor_subprocess(p, "composer-2.5", 5, "default")),
        ("agy_subprocess",
         "providers.agy.run_cli", "providers.agy.vendor_bin.which", "agy",
         lambda p: agy_mod.run_agy_subprocess(p, "Gemini 3.5 Flash", 5, "default")),
        ("agy_sdk",
         "providers.agy.Agent", None, None,
         lambda p: agy_mod.run_agy_sdk(p, "Gemini 3.5 Flash", 5)),
        ("openai_compat",
         "urllib.request.urlopen", None, None,
         lambda p: openai_compat.run_openai_compatible(p, "gpt-4o", 5, "default")),
    )

    def test_gated_cli_seams_refuse_before_send(self):
        prompt = "reach me at %s" % TW_MOBILE
        for name, send_target, which_target, which_bin, runner in self._SEAMS:
            with self.subTest(seam=name):
                sent = []

                def boom(*_a, **_k):
                    sent.append(name)
                    raise AssertionError("%s send reached" % name)

                with _env(gate="1"):
                    os.environ.pop("OMNI_LLM_DEPTH", None)
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(mock.patch(send_target, boom))
                        if which_target:
                            stack.enter_context(
                                mock.patch(which_target, _fake_which(which_bin)))
                        if name == "codex":
                            stack.enter_context(mock.patch(
                                "providers.codex.prepare_host_fence",
                                side_effect=lambda argv, **kw: (
                                    argv, kw.get("env") or {}, {}),
                            ))
                        if name == "cursor":
                            stack.enter_context(mock.patch.object(
                                cursor_mod, "resolve_bin",
                                return_value="/fake/bin/cursor-agent",
                            ))
                        res = asyncio.run(runner(prompt))
                self.assertFalse(res.get("ok"), (name, res))
                self.assertTrue(
                    _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
                    (name, _detail(res)),
                )
                self.assertNotIn(TW_MOBILE, _detail(res))
                self.assertEqual(sent, [], "gate must fire before %s send" % name)


class TestPayloadCoverage(unittest.TestCase):
    def test_llm_image_denies_source_image_path_only(self):
        sent = []

        def boom(*_a, **_k):
            sent.append(True)
            raise AssertionError("llm_image backend send reached")

        with _env(gate="1"):
            os.environ["OMNI_LLM_METRICS"] = "0"
            with mock.patch(
                "providers.grok_image_cli.run_grok_image_edit", boom
            ), mock.patch(
                "providers.xai_image.run_xai_image", boom
            ):
                res = asyncio.run(dispatch.call_tool("llm_image", {
                    "prompt": "make it blue",
                    "source_image": "/tmp/ref-%s.png" % TW_MOBILE,
                    "backend": "cli",
                }))
        self.assertFalse(res.get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "gate must fire before image backend")

    def test_grok_denies_attachment_filename_uri(self):
        sent = []
        root = tempfile.mkdtemp(prefix="omni_gate_roots_")
        path = os.path.join(root, "%s.txt" % TW_MOBILE)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("clean body only\n")

            def boom(*_a, **_k):
                sent.append(True)
                raise AssertionError("grok send reached")

            with _env(gate="1"):
                os.environ["OMNI_LLM_SOURCE_ROOTS"] = os.path.realpath(root)
                with mock.patch("providers.grok.shutil.which", _fake_which("grok")):
                    with mock.patch("providers.grok.run_cli", boom):
                        res = asyncio.run(grok_mod.run_grok_subprocess(
                            "summarize the attachment",
                            "grok-4.6",
                            5,
                            "default",
                            files=[path],
                        ))
        finally:
            shutil.rmtree(root, ignore_errors=True)
        self.assertFalse(res.get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "gate must fire before grok send")


class TestEnabledPredicate(unittest.TestCase):
    def test_only_exact_one_enables(self):
        with _env(gate="1"):
            self.assertTrue(egress_gate.enabled())
        for value in (None, "", "0", "true", "yes"):
            with _env(gate=value):
                self.assertFalse(egress_gate.enabled(), repr(value))


if __name__ == "__main__":
    unittest.main()
