#!/usr/bin/env python3
# no-opencc
"""Issue #148: a LOCAL exception never picks the failure reason.

The envelope reason for an exception raised by omni's own code (a missing task
file, a failed spawn, a bug) is always ``runtime_error``; the detail is the
exception type name plus the message with secrets masked before truncation.
Hermetic: ``run_cli`` and the vendor binaries are mocked, nothing is spawned.
"""
import asyncio
import contextlib
import os
import io
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from providers import (agent_lane, agy, artifacts, claude, dispatch, metrics, codex, cursor, errors, grok,
                       health, kernel_sandbox, host_fence, openai_compat, pi,
                       sandbox_canary, xai_image)
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

KEY = "sk-" + "A1b2C3d4E5" * 2          # sk- + 20 alphanumerics
PATH_WORD = "login-auth-token"           # AUTH_MARKERS fodder inside a path


def _assert_redacted(case, res, key=KEY):
    case.assertIs(res.get("ok"), False, res)
    case.assertEqual(res.get("reason"), "runtime_error", res)
    detail = res.get("detail", "")
    case.assertIn("<redacted>", detail)
    case.assertNotIn(key, detail)
    case.assertNotIn("sk-", detail)


class Env(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        temp = tempfile.TemporaryDirectory(prefix="omni-148-")
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

    def host_fence_patches(self):
        stack = contextlib.ExitStack()
        enter = stack.enter_context
        enter(mock.patch.object(host_fence.platform, "system", return_value="Darwin"))
        enter(mock.patch.object(kernel_sandbox, "protected_home", return_value=self.root))
        enter(mock.patch.object(kernel_sandbox, "available", return_value=True))
        enter(mock.patch.object(kernel_sandbox, "verify_fence", return_value=None))
        enter(mock.patch.object(kernel_sandbox, "wrap_argv",
                                side_effect=lambda argv, *a, **k: list(argv)))
        enter(mock.patch.object(kernel_sandbox, "is_wrapped", return_value=True))
        enter(mock.patch.object(kernel_sandbox, "disable_inner_codex_sandbox",
                                side_effect=lambda argv: argv))
        enter(mock.patch.object(kernel_sandbox, "inner_sandbox_disable_failed",
                                return_value=False))
        return stack


class TestExceptionDetail(unittest.TestCase):
    def test_type_name_prefix_and_redaction(self):
        d = errors.exception_detail(RuntimeError("boom " + KEY + " Bearer abc.def"))
        self.assertTrue(d.startswith("RuntimeError: boom "), d)
        self.assertIn("<redacted>", d)
        self.assertNotIn(KEY, d)
        self.assertNotIn("abc.def", d)

    def test_key_straddling_the_limit_is_redacted_before_truncation(self):
        # The key starts before char 300 and ends after it: truncating first
        # would leave a 16+ char "sk-..." prefix that no longer matches.
        prefix = "x" * (300 - len("RuntimeError: ") - 10)
        d = errors.exception_detail(RuntimeError(prefix + KEY + " tail"))
        self.assertLessEqual(len(d), 300)
        self.assertNotIn("sk-", d)
        self.assertIn("<redacted>", d)

    def test_classifier_itself_is_unchanged(self):
        self.assertEqual(errors.classify_exception_message("429 too many requests"),
                         "rate_limit")
        self.assertEqual(errors.classify_exception_message("login expired"),
                         "auth_error")


class TestAgentLane(Env):
    def test_missing_task_file_is_runtime_error_and_redacted(self):
        task_file = os.path.join(self.root, PATH_WORD + "-" + KEY, "task.txt")

        async def runner(argv, **kwargs):
            # What the real run_cli does with stdin_file: open it.
            with open(kwargs["stdin_file"], "rb"):
                pass
            return {"returncode": 0, "stdout": "", "stderr": ""}

        with self.host_fence_patches(), \
                mock.patch.object(agent_lane, "run_cli", side_effect=runner), \
                mock.patch.object(agent_lane.shutil, "which", return_value="/synthetic/codex"), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane, "_workspace_snapshot", return_value={}), \
                mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value={"ok": True, "vendor_state_writable": True}), \
                mock.patch.object(kernel_sandbox, "wrap_kwargs_from_policy",
                                  return_value={"read_wall": True}):
            res = asyncio.run(agent_lane.run_agent(
                "synthetic task", "codex/synthetic-model", self.root,
                read_only=True, timeout_s=5, task_file=task_file))
        _assert_redacted(self, res)
        self.assertTrue(res["detail"].startswith("FileNotFoundError: "), res)
        self.assertIn(PATH_WORD, res["detail"])


class TestCodexConsult(Env):
    def test_run_cli_exception_is_runtime_error_and_redacted(self):
        async def boom(argv, **kwargs):
            raise RuntimeError("login expired " + KEY)

        with self.host_fence_patches(), \
                mock.patch.object(codex.shutil, "which", return_value="/synthetic/codex"), \
                mock.patch.object(codex, "run_cli", side_effect=boom), \
                mock.patch.object(kernel_sandbox, "admit_vendor_binary", return_value=None):
            res = asyncio.run(codex.run_codex_subprocess("synthetic task", "synthetic-model", 5))
        _assert_redacted(self, res)
        self.assertTrue(res["detail"].startswith("RuntimeError: "), res)
        self.assertIs(res.get("account_retry_safe"), False)


class TestWriteCanary(Env):
    def test_probe_exception_is_runtime_error_and_redacted(self):
        async def boom(argv, **kwargs):
            raise RuntimeError("login expired 429 " + KEY)

        with mock.patch.object(sandbox_canary.shutil, "which", return_value="/fake/bin/codex"), \
                mock.patch.object(sandbox_canary, "is_vendor_readonly_argv", return_value=True), \
                mock.patch.object(sandbox_canary, "run_cli", side_effect=boom):
            res = asyncio.run(sandbox_canary._run_vendor_readonly_probe("p", 5))
        _assert_redacted(self, res)
        self.assertTrue(res["detail"].startswith("RuntimeError: "), res)


class TestOtherConsultLanes(Env):
    def test_claude_consult(self):
        async def boom(argv, **kwargs):
            raise RuntimeError("401 unauthorized login " + KEY)

        with mock.patch.object(claude.shutil, "which", return_value="/synthetic/claude"), \
                mock.patch.object(claude, "run_cli", side_effect=boom):
            res = asyncio.run(claude.run_claude_subprocess("synthetic task", "synthetic-model", 5))
        _assert_redacted(self, res)
        self.assertTrue(res["detail"].startswith("RuntimeError: "), res)


class TestAgySdkKeepsTextClassification(Env):
    def test_sdk_remote_429_is_still_rate_limit(self):
        class FakeAgent:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                raise RuntimeError("429 too many requests " + KEY)

            async def __aexit__(self, *a):
                return False

        with mock.patch.object(agy, "build_agy_sdk_config", return_value=object()), \
                mock.patch.object(agy, "Agent", FakeAgent, create=True):
            res = asyncio.run(agy.run_agy_sdk("p", "synthetic-model", 5))
        self.assertEqual(res.get("reason"), "rate_limit", res)
        self.assertNotIn(KEY, res.get("detail", ""))
        self.assertTrue(res["detail"].startswith("RuntimeError: "), res)


class TestEnvelopeTextIsRedacted(Env):
    def test_openai_compat_http_error_body_redacted_reason_from_status(self):
        body = ('{"error": "bad key %s"}' % KEY).encode()
        err = urllib.error.HTTPError(
            "http://x/chat/completions", 429, "Too Many Requests", {}, io.BytesIO(body))
        with mock.patch.object(openai_compat.urllib.request, "urlopen", side_effect=err):
            res = asyncio.run(openai_compat.run_openai_compatible(
                "p", "synthetic-model", 5, "default"))
        self.assertEqual(res.get("reason"), "rate_limit", res)   # from the status
        self.assertIn("<redacted>", res["detail"])
        self.assertNotIn("sk-", res["detail"])

    def test_artifact_read_root_open_error_redacted(self):
        with mock.patch.object(artifacts, "_is_confined_id", return_value=True), \
                mock.patch.object(artifacts, "_ensure_and_open_root",
                                  side_effect=OSError("denied for " + KEY)):
            res = artifacts.read("a" * 64)
        self.assertEqual(res.get("reason"), "runtime_error", res)
        self.assertIn("<redacted>", res["detail"])
        self.assertNotIn("sk-", res["detail"])


class TestFixRound1(Env):
    def test_fence_path_rejected_masks_before_the_300_cut(self):
        key = "sk-" + "A" * 24
        label = "extra"
        head = "fence path rejected (%s does not exist): " % label
        # Components stay under NAME_MAX so lstat says ENOENT; the key starts
        # at char 290 of the message and ends past the 300 cut.
        pad = 290 - len(head) - len(self.root) - 1
        raw = os.path.join(self.root, "x" * pad + key)
        with self.assertRaises(kernel_sandbox.FencePathRejected) as ctx:
            kernel_sandbox.admit_path(raw, "subpath", label=label)
        detail = ctx.exception.detail
        self.assertIn("<redacted>", detail)
        self.assertNotIn("sk-", detail)
        self.assertLessEqual(len(detail), 300)
        self.assertNotIn("sk-", str(ctx.exception))

    def test_grok_scan_refusal_masks_before_the_cut(self):
        key = "sk-" + "B" * 24
        exc = kernel_sandbox._GrokScanRefusal("y" * 290 + key)
        self.assertIn("<redacted>", exc.detail)
        self.assertNotIn("sk-", exc.detail)

    def test_metrics_error_on_returned_envelope_is_masked(self):
        key = "sk-" + "C" * 24
        with mock.patch.object(dispatch.metrics_mod, "build_record",
                               side_effect=RuntimeError("x" * 280 + key)):
            out = dispatch._finish_metrics(
                {"ok": True, "text": "t", "model": "m"}, tool="t", model="m",
                account_profile="default", request_bytes=1, t0=time.time())
        self.assertIn("<redacted>", out["metrics_error"])
        self.assertNotIn("sk-", out["metrics_error"])

    def test_attach_metrics_error_is_masked(self):
        key = "sk-" + "D" * 24
        with mock.patch.object(metrics, "metrics_enabled", return_value=True), \
                mock.patch.object(metrics, "append_record",
                                  side_effect=OSError("y" * 280 + key)):
            out = metrics.attach_metrics({"ok": True}, {})
        self.assertIn("<redacted>", out["metrics_error"])
        self.assertNotIn("sk-", out["metrics_error"])

    def test_xai_image_non_http_failure_is_runtime_error(self):
        os.environ["XAI_API_KEY"] = "synthetic-test-key"
        os.environ["XAI_API_BASE"] = "bogus-scheme-login-auth://host"
        res = asyncio.run(xai_image.run_xai_image("p"))
        self.assertEqual(res.get("reason"), "runtime_error", res)


class TestFixRound2(Env):
    """`_collect_agent_fence` texts are masked once at each exit, then cut."""

    def _straddling_config_dir(self):
        # CODEX_HOME resolves <inside_home>: <cfg> -> <rp>. Measure where the
        # first path starts, then name the dir so a synthetic key starts at
        # char 290 (only a few key chars survive the cut) and runs past the 300 cut.
        probe = os.path.join(self.root, "QQQQQQQQQQQQQQQQ")
        os.mkdir(probe)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=self.root):
            msg = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.root, read_only=True, config_dir=probe)["detail"]
        start = msg.index(probe)
        key = "sk-" + "E" * 24
        pad = 290 - start - len(self.root) - 1
        self.assertGreater(pad, 0)
        cfg = os.path.join(self.root, "x" * pad + key)
        os.mkdir(cfg)
        return cfg

    def _assert_masked(self, detail):
        self.assertIn("<redacted>", detail)
        self.assertNotIn("sk-", detail)
        self.assertLessEqual(len(detail), 300)

    def test_agent_fence_policy_masks_rejected_before_the_cut(self):
        cfg = self._straddling_config_dir()
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=self.root):
            res = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.root, read_only=True, config_dir=cfg)
        self.assertEqual(res.get("reason"), "fence_path_rejected", res)
        self._assert_masked(res["detail"])

    def test_codex_consult_account_fence_masks_rejected_before_the_cut(self):
        cfg = self._straddling_config_dir()
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=self.root):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as ctx:
                kernel_sandbox.codex_consult_account_fence(cfg, self.root)
        self._assert_masked(ctx.exception.detail)
        self._assert_masked(str(ctx.exception))

    def test_grok_scan_refusal_str_is_masked(self):
        exc = kernel_sandbox._GrokScanRefusal("z" * 40 + "sk-" + "F" * 24)
        self.assertIn("<redacted>", str(exc))
        self.assertNotIn("sk-", str(exc))
        self.assertEqual(str(exc), exc.detail)


if __name__ == "__main__":
    unittest.main()
