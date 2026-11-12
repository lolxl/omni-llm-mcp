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
from tests import offline_home


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
_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module. It goes
    # first (and closes last): the state-dir patch below must not be scrubbed.
    global _HOME
    _HOME = offline_home.use_synthetic_home()
    # #168: a write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    import shutil
    try:
        _STATE_PATCH.stop()
        shutil.rmtree(_STATE_DIR, ignore_errors=True)
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()



class TestWorkspaceWrites(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
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

    def _write(self, name, text):
        with open(os.path.join(self.workdir, name), "w") as handle:
            handle.write(text)

    def test_content_change_of_already_modified_tracked_file_counts(self):
        self._write("t.txt", "one")
        self._git("add", "t.txt"); self._git("commit", "-q", "-m", "add t")
        self._write("t.txt", "two")             # " M" before the run
        before = self._snap()
        self._write("t.txt", "three!")          # still " M", new content
        after = self._snap()
        self.assertEqual(before["entries"], after["entries"])
        count, _ = agent_lane._workspace_writes(self.workdir, before, after)
        self.assertEqual(count, 1)

    def test_rewritten_untracked_file_counts(self):
        self._write("u.txt", "one")             # "??" before the run
        before = self._snap()
        self._write("u.txt", "two")
        count, _ = agent_lane._workspace_writes(self.workdir, before, self._snap())
        self.assertEqual(count, 1)

    def test_untouched_dirty_file_counts_zero(self):
        self._write("t.txt", "one")
        self._git("add", "t.txt"); self._git("commit", "-q", "-m", "add t")
        self._write("t.txt", "two")
        self._write("u.txt", "untracked")
        before = self._snap()
        count, _ = agent_lane._workspace_writes(self.workdir, before, self._snap())
        self.assertEqual(count, 0)

    def test_symlink_is_fingerprinted_by_target_name_not_bytes(self):
        outside = tempfile.mkdtemp(prefix="omni243_out_")
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        target = os.path.join(outside, "secret.txt")
        with open(target, "w") as handle:
            handle.write("outside bytes")
        os.symlink(target, os.path.join(self.workdir, "lnk"))
        self.assertEqual(agent_lane._path_fingerprint(self.workdir, "lnk"),
                         "link:" + target)
        self.assertEqual(self._snap()["digests"]["lnk"], "link:" + target)
        # Never read through the link: no open of any kind during the call.
        with mock.patch("builtins.open", side_effect=AssertionError("open")), \
                mock.patch("os.open", side_effect=AssertionError("os.open")):
            self.assertEqual(agent_lane._path_fingerprint(self.workdir, "lnk"),
                             "link:" + target)

    def test_regular_file_fingerprint_is_stat_tuple(self):
        self._write("f.txt", "abc")
        st = os.lstat(os.path.join(self.workdir, "f.txt"))
        self.assertEqual(
            agent_lane._path_fingerprint(self.workdir, "f.txt"),
            (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino))

    def test_quoted_path_rewrite_counts(self):
        names = ("a b.txt", "café.txt")
        for name in names:
            self._write(name, "one")
            self._git("add", name)
        self._git("commit", "-q", "-m", "add quoted")
        for name in names:
            self._write(name, "two")            # " M" before the run
        before = self._snap()
        for name in names:
            self._write(name, "three!")         # size changes, same letters
        after = self._snap()
        self.assertEqual(before["entries"], after["entries"])
        self.assertEqual(len(before["entries"]), 2)
        for fp in before["digests"].values():
            self.assertIsNotNone(fp)            # quoted names were unquoted
        count, _ = agent_lane._workspace_writes(self.workdir, before, after)
        self.assertEqual(count, 2)

    def test_quoted_path_rewrite_counts_with_quotepath_false(self):
        self._git("config", "core.quotePath", "false")
        names = ("中 文.txt", "café x.txt")
        for name in names:
            self._write(name, "one")
            self._git("add", name)
        self._git("commit", "-q", "-m", "add quoted")
        for name in names:
            self._write(name, "two")
        before = self._snap()
        for name in names:
            self._write(name, "three!")
        after = self._snap()
        self.assertEqual(before["entries"], after["entries"])
        self.assertEqual(len(before["entries"]), 2)
        for fp in before["digests"].values():
            self.assertIsNotNone(fp)
        count, _ = agent_lane._workspace_writes(self.workdir, before, after)
        self.assertEqual(count, 2)

    def test_unquote_porcelain_path(self):
        self.assertEqual(agent_lane._unquote_porcelain_path('"a b.txt"'), "a b.txt")
        self.assertEqual(agent_lane._unquote_porcelain_path(r'"caf\303\251.txt"'),
                         "café.txt")
        self.assertEqual(agent_lane._unquote_porcelain_path("plain.txt"), "plain.txt")
        self.assertEqual(agent_lane._unquote_porcelain_path(r'"a\tb.txt"'), "a\tb.txt")
        self.assertEqual(agent_lane._unquote_porcelain_path(r'"a\"b.txt"'), 'a"b.txt')
        self.assertEqual(agent_lane._unquote_porcelain_path(r'"a\\b.txt"'), "a\\b.txt")
        self.assertEqual(agent_lane._unquote_porcelain_path('"中 文.txt"'), "中 文.txt")

    def test_symlinked_parent_dir_is_not_read(self):
        os.mkdir(os.path.join(self.workdir, "d"))
        self._write(os.path.join("d", "x"), "one")
        self._git("add", "d/x")
        self._git("commit", "-q", "-m", "add d/x")
        outside = tempfile.mkdtemp(prefix="omni243_out_")
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        with open(os.path.join(outside, "x"), "w") as handle:
            handle.write("outside")
        os.rename(os.path.join(self.workdir, "d"),
                  os.path.join(self.workdir, "d_moved"))
        os.symlink(outside, os.path.join(self.workdir, "d"))
        with mock.patch("builtins.open", side_effect=AssertionError("open")), \
                mock.patch("os.open", side_effect=AssertionError("os.open")):
            snap = agent_lane._workspace_snapshot(self.workdir)
        self.assertEqual(snap["state"], "ok")

    def test_unsafe_or_unreadable_paths_have_no_fingerprint(self):
        self._write("f.txt", "x")
        os.mkdir(os.path.join(self.workdir, "d"))
        for path in ("../f.txt", "a/../f.txt", os.path.join(self.workdir, "f.txt"),
                     "d", "missing.txt", ""):
            with self.subTest(path=path):
                self.assertIsNone(agent_lane._path_fingerprint(self.workdir, path))
        self.assertIsNotNone(agent_lane._path_fingerprint(self.workdir, "f.txt"))

    def test_old_shape_snapshots_without_digests_compare_by_letters(self):
        before = {"state": "ok", "entries": {"a": " M", "b": "??"}, "head": None}
        same = {"state": "ok", "entries": {"a": " M", "b": "??"}, "head": None}
        self.assertEqual(agent_lane._workspace_writes(self.workdir, before, same),
                         (0, None))
        letters = {"state": "ok", "entries": {"a": "M ", "b": "??"}, "head": None}
        self.assertEqual(agent_lane._workspace_writes(self.workdir, before, letters),
                         (1, None))
        # One side with digests, the other without: still letters only.
        digested = dict(same, digests={"a": "x", "b": "y"})
        self.assertEqual(agent_lane._workspace_writes(self.workdir, before, digested),
                         (0, None))

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
