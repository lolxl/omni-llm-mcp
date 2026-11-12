#!/usr/bin/env python3
# no-opencc
"""Egress residual seam (4/4): attachment content (codex consult `-i`).

CURRENT-BEHAVIOR: even with OMNI_EGRESS_GATE=1, refuse_payload is called with
the prompt only (providers/codex.py:240). File bytes handed to `codex exec -i`
are never scanned. Filename is clean so this pins CONTENT, not the path.

Gap-description note (code wins): claude consult does not accept files= at
all (dispatch unsupported_files). grok consult DOES scan resource text/uri
and is therefore not this residual. Agent/job attachments are the sibling
ungated lane in test_egress_seam_agent_job.py.

No network: run_cli is intercepted. Synthetic tokens only.
"""
import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import dispatch as dispatch_mod
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


TW_MOBILE = "0912345678"


def _env(gate="1"):
    env = os.environ.copy()
    env.pop("OMNI_EGRESS_GATE", None)
    env.pop("OMNI_EGRESS_DENYLIST", None)
    env["OMNI_LLM_METRICS"] = "0"
    env.pop("OMNI_LLM_DEPTH", None)
    if gate is not None:
        env["OMNI_EGRESS_GATE"] = gate
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


class TestCodexConsultAttachmentContentUngated(unittest.TestCase):
    """Entry: providers/codex.py:240 refuse_payload(prompt) only.

    `-i` paths are appended at lines 192-193. Server never reads file bytes
    (docstring at 149-151).
    """

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_codex_file_body_not_scanned_with_gate_on(self):
        # CURRENT-BEHAVIOR: consult `-i` file content is not gated.
        sent = []
        d = tempfile.mkdtemp(prefix="omni_seam_attach_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "notes.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("fixture contact %s\n" % TW_MOBILE)

        def boom(*_a, **_k):
            sent.append("codex")
            raise RuntimeError("stubbed send reached")

        with _env(gate="1"):
            with mock.patch("providers.codex.shutil.which", _fake_which("codex")):
                with mock.patch(
                    "providers.codex.prepare_host_fence",
                    side_effect=lambda argv, **kw: (
                        argv, kw.get("env") or {}, {}),
                ):
                    with mock.patch("providers.codex.run_cli", boom):
                        res = asyncio.run(codex_mod.run_codex_subprocess(
                            "summarize the attachment",
                            "gpt-5.6-sol",
                            5,
                            "default",
                            files=[path],
                        ))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(
            sent, ["codex"],
            "codex consult must still reach send; file body was not scanned",
        )

    def test_codex_file_path_string_not_scanned_with_gate_on(self):
        # CURRENT-BEHAVIOR: the `-i` path string itself is not passed to
        # refuse_payload. Filename carries the fixture token; prompt is clean.
        sent = []
        d = tempfile.mkdtemp(prefix="omni_seam_attach_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "ref-%s.txt" % TW_MOBILE)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("clean body only\n")

        def boom(*_a, **_k):
            sent.append("codex")
            raise RuntimeError("stubbed send reached")

        with _env(gate="1"):
            with mock.patch("providers.codex.shutil.which", _fake_which("codex")):
                with mock.patch(
                    "providers.codex.prepare_host_fence",
                    side_effect=lambda argv, **kw: (
                        argv, kw.get("env") or {}, {}),
                ):
                    with mock.patch("providers.codex.run_cli", boom):
                        res = asyncio.run(codex_mod.run_codex_subprocess(
                            "summarize the attachment",
                            "gpt-5.6-sol",
                            5,
                            "default",
                            files=[path],
                        ))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(
            sent, ["codex"],
            "codex consult must still reach send; -i path was not scanned",
        )


class TestClaudeConsultHasNoFilesEntry(unittest.TestCase):
    """Code wins over the gap blurb: claude consult does not take files=."""

    def test_claude_consult_files_rejected_before_provider(self):
        rec = []

        async def boom(*_a, **_k):
            rec.append("claude")
            raise AssertionError("claude consult must not be called")

        d = tempfile.mkdtemp(prefix="omni_seam_claude_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "notes.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("fixture contact %s\n" % TW_MOBILE)

        with _env(gate="1"):
            with mock.patch.object(claude_mod, "run_claude_subprocess", boom):
                res = asyncio.run(dispatch_mod.route_to_provider(
                    "summarize", "claude/opus", 5, [path], "default"))
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "unsupported_files")
        self.assertEqual(rec, [])
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))


if __name__ == "__main__":
    unittest.main()
