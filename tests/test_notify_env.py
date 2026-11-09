#!/usr/bin/env python3
# no-opencc
"""Notify-sink strip at spawn_env. No processes, no network."""
import asyncio
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-notify-env-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
os.environ["GIT_CONFIG_SYSTEM"] = os.devnull
os.environ["GIT_TERMINAL_PROMPT"] = "0"

from providers import agent_lane  # noqa: E402
from providers import claude as claude_mod  # noqa: E402
from providers import codex as codex_mod  # noqa: E402
from providers import codex_quota as quota_mod  # noqa: E402
from providers import cursor as cursor_mod  # noqa: E402
from providers import grok as grok_mod  # noqa: E402
from providers.profiles import spawn_env  # noqa: E402
from tests.test_agent import FakeRunCli, depth, patched_lane  # noqa: E402

_REAL_WHICH = shutil.which
_SINK = "/bin/echo"
_CLIS = ("codex", "claude", "grok", "pi")


def _fake_which(name, *a, **kw):
    base = os.path.basename(str(name))
    if base in _CLIS:
        return "/fake/bin/" + base
    return _REAL_WHICH(name, *a, **kw)


class TestNotifyEnv(unittest.TestCase):
    def _assert_stripped(self, env):
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", env)
        self.assertEqual(env.get("OMNI_LLM_NOTIFY_QUIET"), "1")

    def _sink_env(self):
        return mock.patch.dict(os.environ, {"OMNI_LLM_NOTIFY_CMD": _SINK},
                               clear=False)

    def test_codex(self):
        fake = FakeRunCli()
        with self._sink_env(), \
                mock.patch.object(codex_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(codex_mod, "prepare_host_fence",
                           side_effect=lambda argv, **kw: (argv, kw.get("env") or {}, {})):
            asyncio.run(codex_mod.run_codex_subprocess("hi", "gpt-5.6-sol", 30))
        self.assertEqual(len(fake.calls), 1)
        self._assert_stripped(fake.kwargs.get("env") or {})

    def test_claude(self):
        fake = FakeRunCli()
        with self._sink_env(), depth(None), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            asyncio.run(claude_mod.run_claude_subprocess("hi", "sonnet", 30))
        self.assertEqual(len(fake.calls), 1)
        self._assert_stripped(fake.kwargs.get("env") or {})

    def test_cursor(self):
        fake = FakeRunCli(stdout='{"type":"result","is_error":false,"result":"HI"}')
        with self._sink_env(), \
                mock.patch.object(cursor_mod, "run_cli", fake), \
                mock.patch.object(cursor_mod, "resolve_bin",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0):
            asyncio.run(cursor_mod.run_cursor_subprocess(
                "hi", "composer-2.5", 30))
        self.assertEqual(len(fake.calls), 1)
        self._assert_stripped(fake.kwargs.get("env") or {})

    def test_codex_quota(self):
        captured = {}

        def fake_popen(*args, **kwargs):
            captured["env"] = dict(kwargs.get("env") or {})
            raise FileNotFoundError("codex")

        with self._sink_env(), \
                tempfile.TemporaryDirectory(prefix="omni-quota-") as tmp, \
                mock.patch.object(quota_mod.shutil, "which",
                                  return_value="/fake/bin/codex"), \
                mock.patch.object(quota_mod.tempfile, "mkdtemp",
                                  return_value=tmp), \
                mock.patch.object(quota_mod.subprocess, "Popen", fake_popen):
            quota_mod._spawn_rate_limits("default", 5, overrides={})
        self.assertIn("env", captured)
        self._assert_stripped(captured["env"] or {})

    def test_grok(self):
        fake = FakeRunCli()
        with self._sink_env(), \
                mock.patch.object(grok_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            asyncio.run(grok_mod.run_grok_subprocess("hi", "grok-4.5", 30))
        self.assertEqual(len(fake.calls), 1)
        self._assert_stripped(fake.kwargs.get("env") or {})

    def _run_claude_lane(self, **kwargs):
        fake = FakeRunCli()
        root = tempfile.mkdtemp(prefix="omni_notify_wd_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        workdir = os.path.join(root, "wd")
        os.makedirs(workdir)
        git_env = dict(os.environ)
        git_env.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        })
        subprocess.run(
            ["git", "init", "-q", "-b", "work", workdir],
            check=True, env=git_env,
        )
        log_path = os.path.join(root, "stream.raw.log")
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet", workdir=workdir,
                timeout_s=30, job_lane=True, log_path=log_path, **kwargs))
        self.assertTrue(fake.calls, env)
        return env, fake

    def test_agent_lane(self):
        with self._sink_env():
            _env, fake = self._run_claude_lane(fence_requested=False)
        self._assert_stripped(fake.kwargs.get("env") or {})

    def test_agent_lane_foreman(self):
        old_quiet = os.environ.get("OMNI_LLM_NOTIFY_QUIET")
        os.environ["OMNI_LLM_NOTIFY_CMD"] = _SINK
        os.environ.pop("OMNI_LLM_NOTIFY_QUIET", None)
        try:
            _env, fake = self._run_claude_lane(foreman=True)
        finally:
            os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
            if old_quiet is None:
                os.environ.pop("OMNI_LLM_NOTIFY_QUIET", None)
            else:
                os.environ["OMNI_LLM_NOTIFY_QUIET"] = old_quiet
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("OMNI_LLM_NOTIFY_CMD"), _SINK)
        self.assertNotEqual(child.get("OMNI_LLM_NOTIFY_QUIET"), "1")

    def test_spawn_env_os_environ_untouched(self):
        os.environ["OMNI_LLM_NOTIFY_CMD"] = _SINK
        try:
            out = spawn_env(os.environ, {})
            self.assertIn("OMNI_LLM_NOTIFY_CMD", os.environ)
            self.assertEqual(os.environ["OMNI_LLM_NOTIFY_CMD"], _SINK)
            self.assertIsNot(out, os.environ)
            self.assertNotIn("OMNI_LLM_NOTIFY_CMD", out)
            self.assertEqual(out.get("OMNI_LLM_NOTIFY_QUIET"), "1")
        finally:
            os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

    def test_pi_child_env(self):
        from providers import pi as pi_mod
        with self._sink_env():
            child = pi_mod.child_env(0)
        self._assert_stripped(child)
        self.assertEqual(child.get(agent_lane.ENV_DEPTH), "1")

    def test_spawn_env_keep_notify_still_applies_overrides(self):
        base = {"OMNI_LLM_NOTIFY_CMD": _SINK, "CODEX_HOME": "/from/base"}
        out = spawn_env(base, {"CODEX_HOME": "/from/override"}, keep_notify=True)
        self.assertIsNot(out, base)
        self.assertEqual(out.get("CODEX_HOME"), "/from/override")
        self.assertEqual(out.get("OMNI_LLM_NOTIFY_CMD"), _SINK)
        self.assertNotEqual(out.get("OMNI_LLM_NOTIFY_QUIET"), "1")


if __name__ == "__main__":
    unittest.main()
