# no-opencc
"""Issue #65: agent envelopes count workspace writes; expect_writes makes zero LOUD."""
import asyncio
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from providers import agent_lane


def _run(coro):
    return asyncio.run(coro)


def _git_workdir():
    wd = tempfile.mkdtemp(prefix="omni65_wd_")
    subprocess.run(["git", "init", "-q", "-b", "work", wd], check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "--allow-empty", "-q", "-m", "seed"], cwd=wd, check=True)
    return os.path.realpath(wd)


_STATE_PATCH = None
_STATE_DIR = None


def setUpModule():
    # #168: a write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    import shutil
    _STATE_PATCH.stop()
    shutil.rmtree(_STATE_DIR, ignore_errors=True)



class TestWorkspaceWrites(unittest.TestCase):
    def setUp(self):
        self.workdir = _git_workdir()
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)
        self.old_depth = os.environ.pop(agent_lane.ENV_DEPTH, None)
        self.old_fence = os.environ.get("OMNI_LLM_AGENT_FENCE")
        os.environ["OMNI_LLM_AGENT_FENCE"] = "0"  # hermetic: no sandbox-exec
        self.addCleanup(self._restore)

    def _restore(self):
        if self.old_depth is not None:
            os.environ[agent_lane.ENV_DEPTH] = self.old_depth
        if self.old_fence is None:
            os.environ.pop("OMNI_LLM_AGENT_FENCE", None)
        else:
            os.environ["OMNI_LLM_AGENT_FENCE"] = self.old_fence

    def _run_agent(self, fake_run_cli, workdir=None, **kwargs):
        with mock.patch.object(agent_lane, "run_cli", fake_run_cli), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane.shutil, "which",
                                  lambda n, *a, **k: "/fake/bin/" + str(n)), \
                mock.patch.object(agent_lane.kernel_sandbox, "resolve_agent_binary",
                                  lambda p: "/fake/bin/" + p):
            return _run(agent_lane.run_agent(
                task="do it", model="claude/sonnet",
                workdir=workdir or self.workdir, timeout_s=30, **kwargs))

    def _fake(self, writes_file):
        workdir = self.workdir

        async def fake_run_cli(argv, **kwargs):
            if writes_file:
                with open(os.path.join(workdir, "new.txt"), "w") as handle:
                    handle.write("x")
            return {"returncode": 0, "stdout": "done", "stderr": "", "timed_out": False}
        return fake_run_cli

    def test_writes_are_counted(self):
        env = self._run_agent(self._fake(True))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("workspace_writes"), 1)

    def test_zero_writes_without_flag_stays_ok(self):
        env = self._run_agent(self._fake(False))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("workspace_writes"), 0)

    def test_zero_writes_with_flag_is_loud_and_keeps_text(self):
        env = self._run_agent(self._fake(False), expect_writes=True)
        self.assertFalse(env.get("ok"))
        self.assertEqual(env.get("reason"), "no_workspace_writes")
        self.assertEqual(env.get("text"), "done")
        self.assertEqual(env.get("workspace_writes"), 0)

    def test_writes_with_flag_is_ok(self):
        env = self._run_agent(self._fake(True), expect_writes=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("workspace_writes"), 1)

    def test_non_git_workdir_is_null_and_unknown_with_flag(self):
        plain = tempfile.mkdtemp(prefix="omni65_plain_")
        self.addCleanup(shutil.rmtree, plain, ignore_errors=True)

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": "done", "stderr": "", "timed_out": False}
        env = self._run_agent(fake_run_cli, workdir=plain)
        self.assertTrue(env.get("ok"), env)
        self.assertIsNone(env.get("workspace_writes"))
        env = self._run_agent(fake_run_cli, workdir=plain, expect_writes=True)
        self.assertEqual(env.get("reason"), "workspace_writes_unknown")


class TestWorkspaceWritesCounting(unittest.TestCase):
    """Path-level counting per the review: letters may change, paths count once."""

    def setUp(self):
        self.workdir = _git_workdir()
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)

    def _snap(self):
        return agent_lane._workspace_snapshot(self.workdir)

    def _git(self, *args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t"] + list(args),
                       cwd=self.workdir, check=True, capture_output=True)

    def test_status_letter_change_is_one_path(self):
        open(os.path.join(self.workdir, "f"), "w").write("1")
        before = self._snap()                  # ?? f
        self._git("add", "f")                  # A  f
        count, _ = agent_lane._workspace_writes(self.workdir, before, self._snap())
        self.assertEqual(count, 1)

    def test_staged_rename_is_one_path(self):
        open(os.path.join(self.workdir, "old"), "w").write("1")
        self._git("add", "old"); self._git("commit", "-q", "-m", "add old")
        before = self._snap()
        self._git("mv", "old", "new")
        count, _ = agent_lane._workspace_writes(self.workdir, before, self._snap())
        self.assertEqual(count, 1)

    def test_modify_then_revert_is_zero(self):
        before = self._snap()
        path = os.path.join(self.workdir, "tmp")
        open(path, "w").write("x"); os.remove(path)
        count, _ = agent_lane._workspace_writes(self.workdir, before, self._snap())
        self.assertEqual(count, 0)

    def test_write_then_commit_counts_committed_paths(self):
        before = self._snap()
        open(os.path.join(self.workdir, "done.txt"), "w").write("x")
        self._git("add", "done.txt"); self._git("commit", "-q", "-m", "agent commit")
        after = self._snap()
        self.assertNotEqual(before["head"], after["head"])
        count, _ = agent_lane._workspace_writes(self.workdir, before, after)
        self.assertEqual(count, 1)

    def test_status_failure_is_unknown_not_non_git(self):
        # Corrupt the index so git status fails with a non-128 error.
        with open(os.path.join(self.workdir, ".git", "index"), "wb") as handle:
            handle.write(b"not an index")
        snap = self._snap()
        self.assertEqual(snap["state"], "error")
        self.assertNotIn("not a git tree", snap["detail"])

    def test_non_git_is_not_git(self):
        plain = tempfile.mkdtemp(prefix="omni65_plain_")
        self.addCleanup(shutil.rmtree, plain, ignore_errors=True)
        self.assertEqual(agent_lane._workspace_snapshot(plain)["state"], "not_git")


class TestFailureEnvelopesCarryField(TestWorkspaceWrites):
    def test_timeout_and_cli_error_and_empty_carry_workspace_writes(self):
        async def timed_out(argv, **kwargs):
            return {"returncode": None, "stdout": "", "stderr": "", "timed_out": True}

        async def cli_error(argv, **kwargs):
            return {"returncode": 2, "stdout": "", "stderr": "boom", "timed_out": False}

        async def empty(argv, **kwargs):
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}
        for fake, reason in ((timed_out, "timeout"), (cli_error, None), (empty, "empty_response")):
            env = self._run_agent(fake)
            self.assertFalse(env.get("ok"))
            if reason:
                self.assertEqual(env.get("reason"), reason)
            self.assertIn("workspace_writes", env, env)
            self.assertEqual(env["workspace_writes"], 0)


class TestJobLanePersistsExpectWrites(unittest.TestCase):
    def test_job_json_carries_expect_writes_even_when_refused(self):
        from providers import jobs
        root = tempfile.mkdtemp(prefix="omni65_job_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        wd = _git_workdir()
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": root,
                                          "PATH": "/usr/bin:/bin"}):
            os.environ.pop(agent_lane.ENV_DEPTH, None)
            # grok LOUD-refuses read_only, so no supervisor is spawned; the
            # job record is still written and must carry the flag.
            out = jobs.start(task="t", model="grok-4.6", workdir=wd,
                             timeout_s=5, read_only=True, expect_writes=True)
        self.assertIn("refused", out)
        import json
        with open(os.path.join(out["job_dir"], "job.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertIs(data.get("expect_writes"), True)


if __name__ == "__main__":
    unittest.main()
