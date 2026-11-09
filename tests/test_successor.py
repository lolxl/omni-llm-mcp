#!/usr/bin/env python3
# no-opencc
"""Successor launcher: permission-mode mapping, attest, foreman, fake binaries."""
import asyncio
import json
import os
import platform
import shutil
import subprocess
import tempfile
import unittest
import uuid
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-successor-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
os.environ.pop("OMNI_LLM_DEPTH", None)

from providers import agent_lane, jobs, kernel_sandbox
from providers.agent_lane import (
    CODEX_BYPASS_SANDBOX_FLAG,
    UnsupportedPermissionMode,
    successor_permission_argv,
)


def _git_env():
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    })
    return env


def _make_workdir(root):
    wd = os.path.join(root, "wd")
    os.makedirs(wd)
    subprocess.run(
        ["git", "init", "-q", "-b", "work", wd],
        check=True, env=_git_env(),
    )
    return wd


class SuccessorPermissionArgvTest(unittest.TestCase):
    def test_successor_permission_argv_claude(self):
        self.assertEqual(
            successor_permission_argv("claude", "acceptEdits"),
            ["--permission-mode", "acceptEdits"],
        )
        self.assertEqual(
            successor_permission_argv("claude", "default"),
            [],
        )
        self.assertEqual(
            successor_permission_argv("claude", None),
            [],
        )

    def test_successor_permission_argv_codex(self):
        self.assertEqual(
            successor_permission_argv("codex", "plan"),
            ["--sandbox", "read-only"],
        )
        self.assertEqual(
            successor_permission_argv("codex", "bypassPermissions"),
            [CODEX_BYPASS_SANDBOX_FLAG],
        )
        self.assertEqual(
            CODEX_BYPASS_SANDBOX_FLAG,
            "--dangerously-bypass-approvals-and-sandbox",
        )
        for mode in ("default", "acceptEdits", "auto", "manual", "dontAsk"):
            self.assertEqual(
                successor_permission_argv("codex", mode),
                ["--sandbox", "workspace-write"],
                mode,
            )

    def test_successor_permission_argv_grok(self):
        self.assertEqual(
            successor_permission_argv("grok", "bypassPermissions"),
            ["--sandbox", "worker", "--always-approve"],
        )
        for mode in ("default", "acceptEdits", "auto", "manual", "dontAsk"):
            self.assertEqual(
                successor_permission_argv("grok", mode),
                ["--sandbox", "strict"],
                mode,
            )
        with self.assertRaises(UnsupportedPermissionMode):
            successor_permission_argv("grok", "plan")

    def test_successor_permission_argv_unknown_provider_raises(self):
        with self.assertRaises(UnsupportedPermissionMode):
            successor_permission_argv("pi", "default")


class SuccessorPermissionJobsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        cls._git_env_prev = {}
        for key, value in _git_env().items():
            if key.startswith("GIT_"):
                cls._git_env_prev[key] = os.environ.get(key)
                os.environ[key] = value
        cls.root = tempfile.mkdtemp(prefix="omni-successor-jobs-")
        cls.workdir = _make_workdir(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        for key, value in cls._git_env_prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        os.environ.pop("OMNI_LLM_DEPTH", None)

    def test_permission_mode_requires_foreman(self):
        started = jobs.start(
            task="hello", model="claude/sonnet", workdir=self.workdir,
            timeout_s=30, permission_mode="acceptEdits",
        )
        self.assertIn("refused", started)
        env = started["refused"]
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "permission_mode_requires_foreman")
        with open(os.path.join(started["job_dir"], "result.json")) as handle:
            res = json.load(handle)
        self.assertEqual(res.get("reason"), "permission_mode_requires_foreman")
        self.assertIsNone(jobs.status(started["job_dir"]).get("pid"))

    def test_unsupported_permission_mode(self):
        started = jobs.start(
            task="hello", model="claude/sonnet", workdir=self.workdir,
            timeout_s=30, permission_mode="not-a-mode", foreman=True,
        )
        self.assertIn("refused", started)
        self.assertEqual(
            started["refused"].get("reason"), "unsupported_permission_mode")

    def test_grok_plan_foreman_is_unsupported_permission_mode(self):
        started = jobs.start(
            task="hello", model="grok-4.6", workdir=self.workdir,
            timeout_s=30, permission_mode="plan", foreman=True,
        )
        self.assertIn("refused", started)
        env = started["refused"]
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_permission_mode")
        self.assertIsNone(jobs.status(started["job_dir"]).get("pid"))

    def test_foreman_requires_write_lane(self):
        started = jobs.start(
            task="hello", model="claude/sonnet", workdir=self.workdir,
            timeout_s=30, foreman=True, read_only=True,
        )
        self.assertIn("refused", started)
        self.assertEqual(
            started["refused"].get("reason"), "foreman_requires_write_lane")
        self.assertIsNone(jobs.status(started["job_dir"]).get("pid"))

    def test_pi_foreman_is_unsupported_provider(self):
        started = jobs.start(
            task="hello", model="pi/anthropic/claude-sonnet-5",
            workdir=self.workdir, timeout_s=30, foreman=True,
        )
        self.assertIn("refused", started)
        env = started["refused"]
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "foreman_unsupported_provider")
        self.assertIsNone(jobs.status(started["job_dir"]).get("pid"))

    def test_cursor_foreman_is_unsupported_provider(self):
        started = jobs.start(
            task="hello", model="cursor/composer-2.5",
            workdir=self.workdir, timeout_s=30, foreman=True,
        )
        self.assertIn("refused", started)
        env = started["refused"]
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "foreman_unsupported_provider")
        self.assertIsNone(jobs.status(started["job_dir"]).get("pid"))

    def test_foreman_start_does_not_pop_depth(self):
        os.environ[agent_lane.ENV_DEPTH] = "1"
        real_pop = os.environ.pop

        def guarded_pop(key, *args, **kwargs):
            if key == agent_lane.ENV_DEPTH:
                self.fail("os.environ.pop called with %r" % (key,))
            return real_pop(key, *args, **kwargs)

        try:
            with mock.patch.object(os.environ, "pop", side_effect=guarded_pop):
                started = jobs.start(
                    task="hello", model="claude/sonnet",
                    workdir=self.workdir, timeout_s=30,
                    effort="not-a-real-effort-token",
                    foreman=True,
                )
            self.assertIn("refused", started)
            self.assertNotEqual(
                started["refused"].get("reason"), "recursion_guard",
                started)
            self.assertEqual(os.environ.get(agent_lane.ENV_DEPTH), "1")
        finally:
            real_pop(agent_lane.ENV_DEPTH, None)

    def test_default_claude_job_stays_fenced(self):
        # 3d(i): no --foreman → planned fence stays seatbelt-fs on Darwin
        # with sandbox-exec advertised available. Mutation m2 (foreman
        # default True) must turn this red.
        from providers import kernel_sandbox
        with mock.patch.object(kernel_sandbox, "available", return_value=True):
            started = jobs.start(
                task="hello", model="claude/sonnet", workdir=self.workdir,
                timeout_s=30, effort="not-a-real-effort-token",
            )
        with open(os.path.join(started["job_dir"], "job.json")) as handle:
            job = json.load(handle)
        self.assertTrue(job.get("fence_requested"), job)
        self.assertEqual(job.get("fence"), "seatbelt-fs", job)
        self.assertFalse(job.get("foreman"), job)

    def test_foreman_claude_job_is_unfenced(self):
        from providers import kernel_sandbox
        with mock.patch.object(kernel_sandbox, "available", return_value=True):
            started = jobs.start(
                task="hello", model="claude/sonnet", workdir=self.workdir,
                timeout_s=30, effort="not-a-real-effort-token",
                foreman=True, permission_mode="acceptEdits",
            )
        with open(os.path.join(started["job_dir"], "job.json")) as handle:
            job = json.load(handle)
        self.assertTrue(job.get("foreman"), job)
        self.assertFalse(job.get("fence_requested"), job)
        self.assertEqual(job.get("fence"), "none", job)

    def _run_claude_argv(self, **kwargs):
        from tests.test_agent import FakeRunCli, depth, patched_lane
        fake = FakeRunCli()
        log_path = os.path.join(self.root, "stream.raw.log")
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet", workdir=self.workdir,
                timeout_s=30, job_lane=True, log_path=log_path, **kwargs))
        self.assertTrue(fake.calls, env)
        return env, fake

    def test_foreman_claude_argv_has_permission_mode_not_dropped_flags(self):
        # 3d(ii)
        _env, fake = self._run_claude_argv(
            foreman=True, permission_mode="acceptEdits")
        argv = fake.argv
        self.assertIn("--permission-mode", argv)
        self.assertEqual(
            argv[argv.index("--permission-mode") + 1], "acceptEdits")
        self.assertNotIn("--setting-sources", argv)
        self.assertNotIn("--dangerously-skip-permissions", argv)
        self.assertNotIn("--strict-mcp-config", argv)

    def test_no_fence_alone_keeps_old_claude_argv(self):
        # 3d(iii): --no-fence is not --foreman
        _env, fake = self._run_claude_argv(fence_requested=False)
        argv = fake.argv
        self.assertIn("--dangerously-skip-permissions", argv)
        self.assertIn("--setting-sources", argv)
        self.assertIn("--strict-mcp-config", argv)
        self.assertNotIn("--permission-mode", argv)

    def test_foreman_child_env_keeps_notify_and_drops_depth(self):
        # 3d(iv) foreman
        old_quiet = os.environ.get("OMNI_LLM_NOTIFY_QUIET")
        os.environ["OMNI_LLM_NOTIFY_CMD"] = "/tmp/operator-sink"
        os.environ.pop("OMNI_LLM_NOTIFY_QUIET", None)
        os.environ.pop("OMNI_LLM_DEPTH", None)
        try:
            _env, fake = self._run_claude_argv(foreman=True)
        finally:
            os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
            if old_quiet is None:
                os.environ.pop("OMNI_LLM_NOTIFY_QUIET", None)
            else:
                os.environ["OMNI_LLM_NOTIFY_QUIET"] = old_quiet
        child = fake.kwargs.get("env") or {}
        self.assertNotIn("OMNI_LLM_DEPTH", child)
        self.assertEqual(child.get("OMNI_LLM_NOTIFY_CMD"), "/tmp/operator-sink")
        self.assertNotEqual(child.get("OMNI_LLM_NOTIFY_QUIET"), "1")

    def test_default_child_env_sets_depth_and_silences_notify(self):
        # 3d(iv) default
        os.environ["OMNI_LLM_NOTIFY_CMD"] = "/tmp/operator-sink"
        try:
            _env, fake = self._run_claude_argv(fence_requested=False)
        finally:
            os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("OMNI_LLM_DEPTH"), "1")
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", child)
        self.assertEqual(child.get("OMNI_LLM_NOTIFY_QUIET"), "1")


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX = os.path.join(_REPO, "tests", "fixtures", "successor")

FAKE_BACKEND = r'''#!/usr/bin/env python3
import os
import shutil
import sys

provider = os.environ.get("OMNI_FAKE_PROVIDER", "")
log = os.environ.get("OMNI_FAKE_ARGV_LOG")
if log:
    with open(log, "w", encoding="utf-8") as handle:
        for arg in sys.argv:
            handle.write(arg + "\n")

env_dump = os.environ.get("OMNI_FAKE_ENV_DUMP")
if env_dump:
    with open(env_dump, "w", encoding="utf-8") as handle:
        handle.write("OMNI_LLM_DEPTH=%s\n" % os.environ.get("OMNI_LLM_DEPTH", ""))
        handle.write("OMNI_LLM_NOTIFY_QUIET=%s\n" % os.environ.get(
            "OMNI_LLM_NOTIFY_QUIET", ""))

session_id = os.environ.get("OMNI_FAKE_SESSION_ID") or ""
if not session_id:
    for flag in ("--session-id", "-s"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                session_id = sys.argv[i + 1]
                break

record_src = os.environ.get("OMNI_FAKE_RECORD_SRC") or ""
if record_src and session_id:
    if provider == "claude":
        home = os.environ.get("CLAUDE_CONFIG_DIR") or ""
        if home:
            dest_dir = os.path.join(
                home, "projects",
                os.environ.get("OMNI_FAKE_CLAUDE_PROJECT") or "probe")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(record_src, os.path.join(dest_dir, session_id + ".jsonl"))
    elif provider == "codex":
        home = os.environ.get("CODEX_HOME") or ""
        if home:
            dest_dir = os.path.join(home, "sessions", "2026", "09", "15")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(
                record_src,
                os.path.join(dest_dir, "rollout-probe-%s.jsonl" % session_id),
            )
    elif provider == "grok":
        home = os.environ.get("GROK_HOME") or ""
        if home:
            dest_dir = os.path.join(home, "sessions", "probe", session_id)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(record_src, os.path.join(dest_dir, "summary.json"))

stdout_src = os.environ.get("OMNI_FAKE_STDOUT_SRC") or ""
if stdout_src:
    with open(stdout_src, encoding="utf-8") as handle:
        text = handle.read()
    if session_id:
        text = text.replace("__SESSION_ID__", session_id)
        text = text.replace(
            "01a0a2ea-2b61-7c60-8ea8-feb3e90d818f", session_id)
        text = text.replace(
            "b6e7e486-5940-45eb-972f-9ccc40374f0b", session_id)
        text = text.replace(
            "bcdfc3dd-04de-4055-b559-bd67070a1869", session_id)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()

if "-o" in sys.argv:
    outp = sys.argv[sys.argv.index("-o") + 1]
    parent = os.path.dirname(outp)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as handle:
        handle.write("ok\n")

if not sys.stdin.isatty():
    sys.stdin.read()
'''


def _read_argv_log(path):
    with open(path, encoding="utf-8") as handle:
        return [line[:-1] if line.endswith("\n") else line
                for line in handle]


class SuccessorFakeBinaryTest(unittest.TestCase):
    """Item 4: fake PATH binaries pin argv byte-for-byte and attest."""

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        os.environ.pop("OMNI_LLM_DEPTH", None)
        self.root = tempfile.mkdtemp(prefix="omni-successor-fake-")
        self.bindir = os.path.realpath(os.path.join(self.root, "bin"))
        os.makedirs(self.bindir)
        self.workdir = _make_workdir(self.root)
        self.vendor_home = os.path.realpath(
            os.path.join(self.root, "vendor-home"))
        os.makedirs(self.vendor_home)
        self.argv_log = os.path.join(self.root, "argv.log")
        # The host fence's read wall covers the checkout (it lives under
        # home), so a fenced fake vendor binary cannot read fixtures from
        # there. Serve a copy from the temp root the fence admits.
        self.fix = os.path.join(self.root, "fixtures")
        shutil.copytree(_FIX, self.fix)
        self._old = {
            "PATH": os.environ.get("PATH"),
            "OMNI_LLM_CLAUDE_BIN": os.environ.get("OMNI_LLM_CLAUDE_BIN"),
            "CLAUDE_CONFIG_DIR": os.environ.get("CLAUDE_CONFIG_DIR"),
            "CODEX_HOME": os.environ.get("CODEX_HOME"),
            "GROK_HOME": os.environ.get("GROK_HOME"),
            "OMNI_FAKE_PROVIDER": os.environ.get("OMNI_FAKE_PROVIDER"),
            "OMNI_FAKE_ARGV_LOG": os.environ.get("OMNI_FAKE_ARGV_LOG"),
            "OMNI_FAKE_RECORD_SRC": os.environ.get("OMNI_FAKE_RECORD_SRC"),
            "OMNI_FAKE_STDOUT_SRC": os.environ.get("OMNI_FAKE_STDOUT_SRC"),
            "OMNI_FAKE_SESSION_ID": os.environ.get("OMNI_FAKE_SESSION_ID"),
            "OMNI_FAKE_ENV_DUMP": os.environ.get("OMNI_FAKE_ENV_DUMP"),
            "OMNI_FAKE_CLAUDE_PROJECT": os.environ.get(
                "OMNI_FAKE_CLAUDE_PROJECT"),
            "OMNI_LLM_CHILD_ENV_PASS": os.environ.get(
                "OMNI_LLM_CHILD_ENV_PASS"),
        }
        # The fenced child gets an allowlisted environment (#138): the fake
        # vendor's control variables are passed by name, as an operator would.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = ",".join((
            "OMNI_FAKE_PROVIDER", "OMNI_FAKE_ARGV_LOG", "OMNI_FAKE_RECORD_SRC",
            "OMNI_FAKE_STDOUT_SRC", "OMNI_FAKE_SESSION_ID",
            "OMNI_FAKE_ENV_DUMP", "OMNI_FAKE_CLAUDE_PROJECT"))
        for key in (
            "OMNI_LLM_CLAUDE_BIN", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
            "GROK_HOME", "OMNI_FAKE_SESSION_ID", "OMNI_FAKE_ENV_DUMP",
            "OMNI_FAKE_CLAUDE_PROJECT",
        ):
            os.environ.pop(key, None)
        os.environ["PATH"] = self.bindir + os.pathsep + (self._old["PATH"] or "")
        for key, value in _git_env().items():
            if key.startswith("GIT_"):
                os.environ[key] = value

    def tearDown(self):
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.root, ignore_errors=True)

    def _install(self, name):
        path = os.path.join(self.bindir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(FAKE_BACKEND)
        os.chmod(path, 0o755)
        return path

    def _write_stdout(self, name, body):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def _wait_ok(self, started):
        self.assertNotIn("refused", started, started)
        job_dir = started["job_dir"]
        res = jobs.result(job_dir, wait_s=40)
        self.assertTrue(res.get("ok"), res)
        return job_dir, res

    def test_fake_claude_argv_and_attest(self):
        bin_path = self._install("claude")
        stdout_src = self._write_stdout("claude.stdout.jsonl", "\n".join((
            json.dumps({"type": "system", "subtype": "init",
                        "session_id": "__SESSION_ID__",
                        "model": "claude-sonnet-5"}),
            json.dumps({"type": "assistant",
                        "message": {"role": "assistant",
                                    "model": "claude-sonnet-5",
                                    "content": [{"type": "text",
                                                 "text": "ok"}]},
                        "session_id": "__SESSION_ID__"}),
            json.dumps({"type": "result", "subtype": "success",
                        "is_error": False, "result": "ok",
                        "session_id": "__SESSION_ID__"}),
        )) + "\n")
        os.environ["CLAUDE_CONFIG_DIR"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "claude"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "claude-transcript-first-assistant.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        started = jobs.start(
            task="ok", model="claude/claude-sonnet-5",
            workdir=self.workdir, timeout_s=60, effort="low",
            foreman=True, permission_mode="acceptEdits",
        )
        job_dir, res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        expected = [
            os.path.realpath(bin_path), "-p",
            "--model", "claude-sonnet-5",
            "--session-id", sid,
            "--permission-mode", "acceptEdits",
            "--effort", "low",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        self.assertEqual(recorded, expected)
        planted = os.path.join(
            self.vendor_home, "projects", "probe", "%s.jsonl" % sid)
        self.assertEqual(res.get("attest"), {
            "session_id": sid,
            "model": "claude-sonnet-5",
            "effort": "low",
            "source": os.path.abspath(planted),
        })
        self.assertEqual(job.get("session_id"), sid)

    def test_fake_codex_argv_and_attest(self):
        bin_path = self._install("codex")
        sid = str(uuid.uuid4())
        with open(os.path.join(self.fix, "codex-exec-json-events.jsonl"),
                  encoding="utf-8") as handle:
            stdout_src = self._write_stdout("codex.stdout.jsonl", handle.read())
        os.environ["CODEX_HOME"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "codex"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "codex-rollout-session_meta-turn_context.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        os.environ["OMNI_FAKE_SESSION_ID"] = sid
        started = jobs.start(
            task="ok", model="codex/gpt-5.6-sol",
            workdir=self.workdir, timeout_s=60, effort="low",
            foreman=True, permission_mode="bypassPermissions",
        )
        job_dir, res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        recorded = _read_argv_log(self.argv_log)
        self.assertIn("-o", recorded)
        tmp_out = recorded[recorded.index("-o") + 1]
        expected = [
            bin_path, "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check", "--json",
            "-c", "model_reasoning_effort=low",
            "-m", "gpt-5.6-sol",
            "-o", tmp_out,
            "-",
        ]
        self.assertEqual(recorded, expected)
        planted = os.path.join(
            self.vendor_home, "sessions", "2026", "09", "15",
            "rollout-probe-%s.jsonl" % sid)
        self.assertEqual(res.get("attest"), {
            "session_id": sid,
            "model": "gpt-5.6-sol",
            "effort": "low",
            "source": os.path.abspath(planted),
        })
        self.assertEqual(job.get("session_id"), sid)

    def test_fake_grok_argv_and_attest(self):
        bin_path = self._install("grok")
        with open(os.path.join(
                self.fix, "grok-streaming-messages-json-events.jsonl"),
                encoding="utf-8") as handle:
            stdout_src = self._write_stdout("grok.stdout.jsonl", handle.read())
        os.environ["GROK_HOME"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "grok"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "grok-session-summary.json")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        started = jobs.start(
            task="ok", model="grok-4.6",
            workdir=self.workdir, timeout_s=60, effort="low",
            foreman=True, permission_mode="acceptEdits",
        )
        job_dir, res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        self.assertIn("--prompt-file", recorded)
        tmp_prompt = recorded[recorded.index("--prompt-file") + 1]
        expected = [
            bin_path, "--prompt-file", tmp_prompt,
            "--model", "grok-4.6",
            "--reasoning-effort", "low",
            "--output-format", "streaming-messages-json",
            "--no-memory",
            "--sandbox", "strict",
            "--cwd", os.path.realpath(self.workdir),
            "--include-partial-messages",
            "--session-id", sid,
        ]
        self.assertEqual(recorded, expected)
        planted = os.path.join(
            self.vendor_home, "sessions", "probe", sid, "summary.json")
        self.assertEqual(res.get("attest"), {
            "session_id": sid,
            "model": "grok-4.6",
            "effort": "low",
            "source": os.path.abspath(planted),
        })
        self.assertEqual(job.get("session_id"), sid)

    def _claude_stream_stdout(self, name):
        return self._write_stdout(name, "\n".join((
            json.dumps({"type": "system", "subtype": "init",
                        "session_id": "__SESSION_ID__",
                        "model": "claude-sonnet-5"}),
            json.dumps({"type": "assistant",
                        "message": {"role": "assistant",
                                    "model": "claude-sonnet-5",
                                    "content": [{"type": "text",
                                                 "text": "ok"}]},
                        "session_id": "__SESSION_ID__"}),
            json.dumps({"type": "result", "subtype": "success",
                        "is_error": False, "result": "ok",
                        "session_id": "__SESSION_ID__"}),
        )) + "\n")

    def _read_env_dump(self, path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_fake_claude_foreman_omitted_mode_has_no_permission_flag(self):
        bin_path = self._install("claude")
        stdout_src = self._claude_stream_stdout("claude-omit.stdout.jsonl")
        os.environ["CLAUDE_CONFIG_DIR"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "claude"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "claude-transcript-first-assistant.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        started = jobs.start(
            task="ok", model="claude/claude-sonnet-5",
            workdir=self.workdir, timeout_s=60, effort="low",
            foreman=True,
        )
        job_dir, _res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        expected = [
            os.path.realpath(bin_path), "-p",
            "--model", "claude-sonnet-5",
            "--session-id", sid,
            "--effort", "low",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        self.assertEqual(recorded, expected)
        self.assertNotIn("--permission-mode", recorded)

    def test_fake_claude_foreman_plan_mode_has_permission_mode_plan(self):
        bin_path = self._install("claude")
        stdout_src = self._claude_stream_stdout("claude-plan.stdout.jsonl")
        os.environ["CLAUDE_CONFIG_DIR"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "claude"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "claude-transcript-first-assistant.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        started = jobs.start(
            task="ok", model="claude/claude-sonnet-5",
            workdir=self.workdir, timeout_s=60, effort="low",
            foreman=True, permission_mode="plan",
        )
        job_dir, _res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        expected = [
            os.path.realpath(bin_path), "-p",
            "--model", "claude-sonnet-5",
            "--session-id", sid,
            "--permission-mode", "plan",
            "--effort", "low",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        self.assertEqual(recorded, expected)

    def test_fake_claude_default_lane_argv(self):
        bin_path = self._install("claude")
        stdout_src = self._claude_stream_stdout("claude-default.stdout.jsonl")
        env_dump = os.path.join(self.root, "env.dump")
        os.environ["CLAUDE_CONFIG_DIR"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "claude"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "claude-transcript-first-assistant.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        os.environ["OMNI_FAKE_ENV_DUMP"] = env_dump
        # The fence grants write only to the ENCODED workdir project dir.
        os.environ["OMNI_FAKE_CLAUDE_PROJECT"] = (
            kernel_sandbox.claude_encoded_workdir(
                os.path.realpath(self.workdir)))
        started = jobs.start(
            task="ok", model="claude/claude-sonnet-5",
            workdir=self.workdir, timeout_s=60, effort="low",
            fence_requested=False,
        )
        job_dir, _res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        self.assertFalse(job.get("foreman"), job)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        expected = [
            os.path.realpath(bin_path), "-p",
            "--setting-sources", "project",
            "--model", "claude-sonnet-5",
            "--session-id", sid,
            "--dangerously-skip-permissions", "--strict-mcp-config",
            "--effort", "low",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        self.assertEqual(recorded, expected)
        dumped = self._read_env_dump(env_dump)
        self.assertIn("OMNI_LLM_DEPTH=1\n", dumped)
        self.assertIn("OMNI_LLM_NOTIFY_QUIET=1\n", dumped)

    def test_fake_codex_default_lane_argv(self):
        bin_path = self._install("codex")
        sid = str(uuid.uuid4())
        with open(os.path.join(self.fix, "codex-exec-json-events.jsonl"),
                  encoding="utf-8") as handle:
            stdout_src = self._write_stdout(
                "codex-default.stdout.jsonl", handle.read())
        env_dump = os.path.join(self.root, "env.dump")
        os.environ["CODEX_HOME"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "codex"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "codex-rollout-session_meta-turn_context.jsonl")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        os.environ["OMNI_FAKE_SESSION_ID"] = sid
        os.environ["OMNI_FAKE_ENV_DUMP"] = env_dump
        started = jobs.start(
            task="ok", model="codex/gpt-5.6-sol",
            workdir=self.workdir, timeout_s=60, effort="low",
            fence_requested=False,
        )
        job_dir, _res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        self.assertFalse(job.get("foreman"), job)
        recorded = _read_argv_log(self.argv_log)
        self.assertIn("-o", recorded)
        tmp_out = recorded[recorded.index("-o") + 1]
        # On Darwin the verified outer kernel wrap is the sandbox: the host
        # fence replaces codex's inner "--sandbox workspace-write" with the
        # bypass flag. Elsewhere the vendor flags are kept as built.
        inner = (["--dangerously-bypass-approvals-and-sandbox"]
                 if platform.system() == "Darwin"
                 else ["--sandbox", "workspace-write"])
        expected = [
            bin_path, "exec",
            *inner,
            "--skip-git-repo-check", "--json",
            "-c", "model_reasoning_effort=low",
            "-m", "gpt-5.6-sol",
            "-o", tmp_out,
            "-",
        ]
        self.assertEqual(recorded, expected)
        dumped = self._read_env_dump(env_dump)
        self.assertIn("OMNI_LLM_DEPTH=1\n", dumped)
        self.assertIn("OMNI_LLM_NOTIFY_QUIET=1\n", dumped)

    def test_fake_grok_default_lane_argv(self):
        bin_path = self._install("grok")
        with open(os.path.join(
                self.fix, "grok-streaming-messages-json-events.jsonl"),
                encoding="utf-8") as handle:
            stdout_src = self._write_stdout(
                "grok-default.stdout.jsonl", handle.read())
        env_dump = os.path.join(self.root, "env.dump")
        os.environ["GROK_HOME"] = self.vendor_home
        os.environ["OMNI_FAKE_PROVIDER"] = "grok"
        os.environ["OMNI_FAKE_ARGV_LOG"] = self.argv_log
        os.environ["OMNI_FAKE_RECORD_SRC"] = os.path.join(
            self.fix, "grok-session-summary.json")
        os.environ["OMNI_FAKE_STDOUT_SRC"] = stdout_src
        os.environ["OMNI_FAKE_ENV_DUMP"] = env_dump
        started = jobs.start(
            task="ok", model="grok-4.6",
            workdir=self.workdir, timeout_s=60, effort="low",
            fence_requested=False,
        )
        job_dir, _res = self._wait_ok(started)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            job = json.load(handle)
        self.assertFalse(job.get("foreman"), job)
        sid = job.get("session_id")
        self.assertTrue(sid)
        recorded = _read_argv_log(self.argv_log)
        self.assertIn("--prompt-file", recorded)
        tmp_prompt = recorded[recorded.index("--prompt-file") + 1]
        rest = [
            "--prompt-file", tmp_prompt,
            "--model", "grok-4.6",
            "--reasoning-effort", "low",
            "--output-format", "streaming-messages-json",
            "--no-memory",
        ]
        tail = [
            "--cwd", os.path.realpath(self.workdir),
            "--include-partial-messages",
            "--session-id", sid,
        ]
        if platform.system() == "Darwin":
            # Verified outer kernel wrap: the host fence moves grok's
            # nested sandbox to "off", right after the binary.
            expected = [bin_path, "--sandbox", "off", *rest, *tail]
        else:
            expected = [bin_path, *rest, "--sandbox", "strict", *tail]
        self.assertEqual(recorded, expected)
        dumped = self._read_env_dump(env_dump)
        self.assertIn("OMNI_LLM_DEPTH=1\n", dumped)
        self.assertIn("OMNI_LLM_NOTIFY_QUIET=1\n", dumped)


if __name__ == "__main__":
    unittest.main()


