#!/usr/bin/env python3
"""Protected refs: the list lives outside the fence, a run is verified after (#168).

Real Git in temp repositories; the real kernel fence wherever the fence is
what matters. The fake vendor CLI is a shell script that does what a hostile
worker would do to the repository; ``run_agent`` spawns it exactly as it
spawns a real one.

  * The protected branch names are recorded the first time omni sees a
    repository (keyed by the common dir's identity) and later runs read the
    record, never the refs a child could have rewritten.
  * Every protected ref is read from outside the fence before the spawn and
    again after the child exits; any difference fails the run loudly.
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import io
import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import unicodedata
import unittest
from unittest import mock

from providers import agent_lane, host_fence, job_runner, jobs, kernel_sandbox
from providers.agent_lane import ENV_DEPTH
from tests import offline_home

DARWIN = platform.system() == "Darwin"
OVERRIDE_ENV = "OMNI_LLM_PROTECTED_BRANCHES"
_IDENTITY = ["-c", "user.email=t@example.invalid", "-c", "user.name=t"]


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def _write_exec(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, stat.S_IRWXU)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


class _Base(unittest.TestCase):
    """A temp state dir, a fake ``claude`` binary and repository builders."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="omni168_", dir="/private/tmp")
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.state = os.path.join(self.root, "state")
        os.makedirs(self.state)
        self.bindir = os.path.realpath(tempfile.mkdtemp(prefix="omni168_bin_"))
        self.addCleanup(shutil.rmtree, self.bindir, ignore_errors=True)
        self.claude = os.path.join(self.bindir, "claude")
        patcher = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state,
            kernel_sandbox.ENV_CLAUDE_BIN: self.claude,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in (ENV_DEPTH, OVERRIDE_ENV):
            os.environ.pop(key, None)

    # -- repositories -------------------------------------------------

    def repo(self, default="main", name="repo", pack=False):
        """A repo on branch ``work`` whose origin default branch is *default*.

        ``main``, ``master`` and the default branch all exist as local
        branches at the seed commit; ``self.seed`` / ``self.other`` are two
        distinct commit ids.
        """
        repo = os.path.join(self.root, name)
        os.makedirs(repo)
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "seed")
        self.seed = _git(repo, "rev-parse", "HEAD")
        _git(repo, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "other")
        self.other = _git(repo, "rev-parse", "HEAD")
        _git(repo, "update-ref", "refs/heads/main", self.seed)
        for branch in {"master", default}:
            _git(repo, "update-ref", "refs/heads/" + branch, self.seed)
        _git(repo, "update-ref", "refs/heads/work", self.seed)
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/work")
        for branch in {"main", default, "other"}:
            _git(repo, "update-ref", "refs/remotes/origin/" + branch, self.seed)
        _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD",
             "refs/remotes/origin/" + default)
        if pack:
            _git(repo, "pack-refs", "--all", "--prune")
        return repo

    def records(self):
        return sorted(glob.glob(os.path.join(self.state, "protected-refs", "*.json")))

    def ledger(self):
        return os.path.join(self.state, "protected-refs", "ledger")

    # -- running a hostile worker ---------------------------------------

    def _agent(self, repo, script, **kwargs):
        _write_exec(self.claude, "#!/bin/sh\nR=%s\n%s\necho AGENT OK\n" % (
            _sh_quote(repo), script))
        return agent_lane.run_agent(
            task="do it", model="claude/sonnet", workdir=repo, timeout_s=60,
            **kwargs)

    def run_child(self, repo, script, *, foreman=False, read_only=False,
                  allow_main=False, **extra):
        """Run ``run_agent`` with a fake ``claude`` that executes *script*."""
        return asyncio.run(self._agent(
            repo, script, foreman=foreman, read_only=read_only,
            allow_main=allow_main, **extra))

    def run_child_cancelled(self, repo, script, after_s=2.0, **kwargs):
        """Like ``run_child``, but the task is cancelled *after_s* seconds in."""
        async def go():
            task = asyncio.ensure_future(self._agent(repo, script, **kwargs))
            await asyncio.sleep(after_s)
            task.cancel()
            return await task
        return asyncio.run(go())


def _sh_quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


class RecordedProtectedList(_Base):
    """Issue repro 1: the default branch is read once, not from child-writable refs."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_child_symbolic_ref_on_the_remote_default_does_not_move_protection(self):
        repo = self.repo(default="trunk")
        tamper = ("printf 'ref: refs/remotes/origin/other\\n' "
                  "> \"$R/.git/refs/remotes/origin/trunk\"")
        first = self.run_child(repo, tamper)
        # The child really did rewrite the ref the old derivation followed.
        self.assertEqual(_read(os.path.join(repo, ".git/refs/remotes/origin/trunk")),
                         "ref: refs/remotes/origin/other\n")
        self.assertEqual(_git(repo, "symbolic-ref", "-q", "refs/remotes/origin/HEAD"),
                         "refs/remotes/origin/other")
        self.assertTrue(first.get("ok"), first)
        # Next run, parked on the recorded default branch: still refused.
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
        marker = os.path.join(repo, "ran-on-trunk")
        second = self.run_child(repo, ": > \"$R/ran-on-trunk\"")
        self.assertFalse(second.get("ok"), second)
        self.assertEqual(second.get("reason"), "workdir_fence", second)
        self.assertFalse(os.path.exists(marker))

    def test_record_survives_a_rewritten_origin_head_without_any_fence(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/other")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertIn("protected branch", caught.exception.envelope["detail"])

    def test_first_sight_writes_one_record_bound_to_the_common_dir(self):
        repo = self.repo(default="trunk")
        self.assertEqual(self.records(), [])
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        files = self.records()
        self.assertEqual(len(files), 1, files)
        data = json.loads(_read(files[0]))
        st = os.stat(os.path.join(repo, ".git"))
        self.assertEqual(data["common_dir"], os.path.join(repo, ".git"))
        self.assertEqual((data["dev"], data["ino"]), (st.st_dev, st.st_ino))
        self.assertEqual(sorted(data["protected"]), ["main", "master", "trunk"])
        # A second sight (any entry point) reads it back and adds nothing.
        host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertEqual(self.records(), files)

    def test_linked_worktree_shares_the_common_dir_record(self):
        repo = self.repo(default="trunk")
        linked = os.path.join(self.root, "linked")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        linked = os.path.realpath(linked)
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertIsNone(agent_lane._workdir_fence(linked, False))
        self.assertEqual(len(self.records()), 1)

    def test_operator_override_names_the_protected_list(self):
        repo = self.repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/release")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release, hotfix"}):
            env = agent_lane._workdir_fence(repo, False)
            self.assertIsNotNone(env)
            self.assertEqual(env["reason"], "workdir_fence")
            _git(repo, "symbolic-ref", "HEAD", "refs/heads/work")
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            # main / master stay protected whatever the override says.
            _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
            self.assertIsNotNone(agent_lane._workdir_fence(repo, False))


class RecordIdentity(_Base):
    """Issue: a record that does not fit is never rebuilt from refs."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _tamper(self, repo, mutate):
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        files = self.records()
        self.assertEqual(len(files), 1, files)
        raw = _read(files[0])
        os.chmod(files[0], 0o600)
        with open(files[0], "w", encoding="utf-8") as handle:
            handle.write(mutate(json.loads(raw)))
        return files[0], _read(files[0])

    def _assert_refused_everywhere(self, repo, record, content):
        marker = os.path.join(repo, "ran")
        env = self.run_child(repo, ": > \"$R/ran\"")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertIn("protected", env.get("detail", "").lower())
        self.assertFalse(os.path.exists(marker))
        env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")
        # Nothing was rebuilt from the refs.
        self.assertEqual(self.records(), [record])
        self.assertEqual(_read(record), content)

    def test_inode_mismatch_refuses(self):
        repo = self.repo(default="trunk")
        record, content = self._tamper(repo, lambda d: json.dumps(
            dict(d, ino=d["ino"] + 1)))
        self._assert_refused_everywhere(repo, record, content)

    def test_device_mismatch_refuses(self):
        repo = self.repo(default="trunk")
        record, content = self._tamper(repo, lambda d: json.dumps(
            dict(d, dev=d["dev"] + 1)))
        self._assert_refused_everywhere(repo, record, content)

    def test_recorded_path_mismatch_refuses(self):
        repo = self.repo(default="trunk")
        record, content = self._tamper(repo, lambda d: json.dumps(
            dict(d, common_dir=d["common_dir"] + "-elsewhere")))
        self._assert_refused_everywhere(repo, record, content)

    def test_unreadable_record_refuses(self):
        repo = self.repo(default="trunk")
        record, content = self._tamper(repo, lambda d: "{not json")
        self._assert_refused_everywhere(repo, record, content)

    def test_record_with_a_wrong_shape_refuses(self):
        repo = self.repo(default="trunk")
        record, content = self._tamper(repo, lambda d: json.dumps(
            dict(d, protected="main")))
        self._assert_refused_everywhere(repo, record, content)

    def test_read_only_run_does_not_depend_on_the_record(self):
        repo = self.repo(default="trunk")
        self._tamper(repo, lambda d: "{not json")
        env = self.run_child(repo, "true", read_only=True)
        self.assertTrue(env.get("ok"), env)


class PostRunVerification(_Base):
    """A run that changed a protected ref fails, whatever route it took."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _assert_changed(self, env, ref):
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn(ref, env["detail"])
        self.assertIn("reflog", env["detail"])

    def test_changed_ref(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\"",
            foreman=True)
        self._assert_changed(env, "refs/heads/main")
        self.assertIn(self.seed, env["detail"])
        self.assertIn(self.other, env["detail"])

    def test_created_ref(self):
        repo = self.repo()
        _git(repo, "update-ref", "-d", "refs/heads/master")
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/master\"",
            foreman=True)
        self._assert_changed(env, "refs/heads/master")

    def test_deleted_ref(self):
        repo = self.repo()
        env = self.run_child(repo, "rm \"$R/.git/refs/heads/main\"", foreman=True)
        self._assert_changed(env, "refs/heads/main")

    def test_ref_turned_symbolic(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf 'ref: refs/heads/work\\n' > \"$R/.git/refs/heads/main\"",
            foreman=True)
        self._assert_changed(env, "refs/heads/main")

    def test_packed_ref_rewritten(self):
        repo = self.repo(pack=True)
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads/main")))
        script = ("sed 's/^%s refs\\/heads\\/main$/%s refs\\/heads\\/main/' "
                  "\"$R/.git/packed-refs\" > \"$R/.git/packed-refs.new\" && "
                  "mv \"$R/.git/packed-refs.new\" \"$R/.git/packed-refs\"") % (
                      self.seed, self.other)
        env = self.run_child(repo, script, foreman=True)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.other)
        self._assert_changed(env, "refs/heads/main")

    def test_a_failed_run_still_reports_the_change(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other
            + " > \"$R/.git/refs/heads/main\"; exit 3", foreman=True)
        self._assert_changed(env, "refs/heads/main")

    def test_remote_default_branch_is_protected_too(self):
        repo = self.repo(default="trunk")
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/trunk\"",
            foreman=True)
        self._assert_changed(env, "refs/heads/trunk")

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_issue_repro_2_directory_moved_over_a_packed_ref(self):
        # ``release/v1`` lives only in packed-refs and ``refs/heads/release``
        # does not exist, so no directory deny covers it; the leaf deny does
        # not stop a whole directory (holding a loose ``v1``) being moved in.
        repo = self.repo(default="release/v1", pack=True)
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads/release")))
        script = ("mkdir \"$R/plant\"; printf '%s\\n' " + self.other
                  + " > \"$R/plant/v1\"; mv \"$R/plant\" \"$R/.git/refs/heads/release\"")
        env = self.run_child(repo, script)
        planted = os.path.join(repo, ".git/refs/heads/release/v1")
        if os.path.exists(planted):
            self._assert_changed(env, "refs/heads/release/v1")
            self.assertEqual(_git(repo, "rev-parse", "refs/heads/release/v1"), self.other)
        else:
            self.assertEqual(_git(repo, "rev-parse", "refs/heads/release/v1"), self.seed)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_issue_repro_2_created_loose_ref_is_denied_or_detected(self):
        repo = self.repo(default="release/v1", pack=True)
        script = ("mkdir -p \"$R/.git/refs/heads/release\" && printf '%s\\n' " + self.other
                  + " > \"$R/.git/refs/heads/release/v1\"")
        env = self.run_child(repo, script)
        if _git(repo, "rev-parse", "refs/heads/release/v1") != self.seed:
            self._assert_changed(env, "refs/heads/release/v1")

    # -- controls ------------------------------------------------------

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_ordinary_work_branch_commit_and_fetched_ref_stay_ok(self):
        repo = self.repo()
        env = self.run_child(
            repo, ": > \"$R/new.txt\"; printf '%s\\n' " + self.other
            + " > \"$R/.git/refs/heads/work\"; printf '%s\\n' " + self.other
            + " > \"$R/.git/refs/remotes/origin/feature\"")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/work"), self.other)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_real_git_commit_and_ref_update_on_a_work_branch_stay_ok(self):
        repo = self.repo()
        script = ("/usr/bin/git -C \"$R\" -c user.name=t -c user.email=t@example.invalid "
                  "commit --allow-empty -qm child && "
                  "/usr/bin/git -C \"$R\" update-ref refs/remotes/origin/feature HEAD")
        env = self.run_child(repo, script)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "log", "-1", "--format=%s", "refs/heads/work"), "child")
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_remote_set_head_works_inside_the_fence(self):
        # The recorded names make a deny on origin/HEAD unnecessary, and it
        # used to make this fail (side effect 4 of the issue).
        repo = self.repo(default="trunk")
        _git(repo, "remote", "add", "origin", "/nonexistent")
        env = self.run_child(
            repo, "/usr/bin/git -C \"$R\" remote set-head origin other")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "symbolic-ref", "refs/remotes/origin/HEAD"),
                         "refs/remotes/origin/other")
        # ... and the protection did not follow it.
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
        self.assertEqual(self.run_child(repo, "true").get("reason"), "workdir_fence")

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_run_that_does_nothing_stays_ok(self):
        repo = self.repo()
        env = self.run_child(repo, "true")
        self.assertTrue(env.get("ok"), env)

    def test_unfenced_work_branch_commit_stays_ok(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/work\"",
            foreman=True)
        self.assertTrue(env.get("ok"), env)

    def test_allow_main_exempts_only_the_checked_out_branch(self):
        repo = self.repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\"",
            foreman=True, allow_main=True)
        self.assertTrue(env.get("ok"), env)

    def test_allow_main_still_verifies_every_other_protected_ref(self):
        repo = self.repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\"; "
            "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/master\"",
            foreman=True, allow_main=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn("refs/heads/master", env["detail"])
        self.assertNotIn("refs/heads/main:", env["detail"])

    def test_allow_main_on_a_work_branch_verifies_all_protected_refs(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\"",
            foreman=True, allow_main=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)

    def test_non_git_workdir_is_unaffected(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        env = self.run_child(plain, ": > \"$R/x\"", foreman=True)
        self.assertTrue(env.get("ok"), env)

    def test_operator_override_names_are_verified(self):
        repo = self.repo()
        _git(repo, "update-ref", "refs/heads/release", self.seed)
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            env = self.run_child(
                repo, "printf '%s\\n' " + self.other
                + " > \"$R/.git/refs/heads/release\"", foreman=True)
        self._assert_changed(env, "refs/heads/release")


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class RecordIsOutsideTheChildsReach(_Base):
    """A fenced child cannot write the protected-branch record (live probe)."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _fenced(self, repo, script):
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", script], provider="claude", workdir=repo,
            env=dict(os.environ))
        return subprocess.run(argv, cwd=repo, env=env, capture_output=True,
                              text=True, check=False)

    def _probe(self, repo):
        files = self.records()
        self.assertEqual(len(files), 1, files)
        record = files[0]
        before = _read(record)
        probe = self._fenced(
            repo, "printf x > %s; rm -f %s; : > %s; mv %s %s.moved" % (
                _sh_quote(record), _sh_quote(record),
                _sh_quote(os.path.join(os.path.dirname(record), "planted")),
                _sh_quote(record), _sh_quote(record)))
        self.assertEqual(self.records(), [record])
        self.assertEqual(_read(record), before)
        self.assertNotEqual(probe.returncode, 0, probe.stderr)

    def test_default_state_dir_is_not_writable(self):
        repo = self.repo(default="trunk")
        host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self._probe(repo)

    def test_state_dir_inside_the_workdir_is_still_not_writable(self):
        repo = self.repo(default="trunk")
        inside = os.path.join(repo, "state-inside")
        os.makedirs(inside)
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": inside}):
            self.state = inside
            self.assertEqual(self.records(), [])
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
            self._probe(repo)


class ResidualItems(_Base):
    """The two items the issue's comment left open."""

    def test_symbolic_ref_exit_other_than_1_is_not_an_absent_default(self):
        common = os.path.join(self.root, "common")
        os.makedirs(common)
        name, why = host_fence.origin_default_branch(128, "", common)
        self.assertIsNone(name)
        self.assertTrue(why, "exit 128 with no file must not read as 'no default branch'")
        self.assertEqual(host_fence.origin_default_branch(1, "", common), (None, None))

    def test_reftable_repo_is_refused_by_the_agent_lane_fence(self):
        repo = os.path.join(self.root, "rt")
        subprocess.run(["git", "init", "-q", "--ref-format=reftable", "-b", "work", repo],
                       check=True)
        _git(repo, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "seed")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("files ref storage", env["detail"])
        self.assertEqual(self.records(), [])

    def test_bare_repo_as_workdir_is_refused_in_write_mode(self):
        bare = os.path.join(self.root, "bare.git")
        subprocess.run(["git", "init", "-q", "--bare", bare], check=True)
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(bare, required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")

    def test_separate_git_dir_as_workdir_is_refused_in_write_mode(self):
        tree = os.path.join(self.root, "tree")
        gitdir = os.path.join(self.root, "elsewhere.d")
        os.makedirs(tree)
        subprocess.run(["git", "init", "-q", "--separate-git-dir", gitdir, tree],
                       check=True)
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(
                os.path.realpath(gitdir), required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")

    def test_plain_non_git_directory_is_still_admitted(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.assertEqual(host_fence._repo_write_roots(
            plain, required=False, allow_main=False), ((), (), (), (), ()))


# ---------------------------------------------------------------------------
# Fix round 1
# ---------------------------------------------------------------------------

class LedgerEnrollment(_Base):
    """A missing record is not "first sight": enrolled repositories are listed."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _enroll(self, repo):
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertEqual(len(self.records()), 1)
        return self.records()[0]

    def _assert_refused_everywhere(self, repo, needle):
        env = self.run_child(repo, ": > \"$R/ran\"")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertIn(needle, env.get("detail", ""))
        self.assertIn("delete", env["detail"])
        self.assertNotIn("OMNI_LLM_PROTECTED_BRANCHES", env["detail"])
        self.assertFalse(os.path.exists(os.path.join(repo, "ran")))
        env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")

    def test_first_sight_enrolls_the_record_and_a_ledger_line(self):
        repo = self.repo(default="trunk")
        self.assertFalse(os.path.exists(self.ledger()))
        record = self._enroll(repo)
        key = os.path.basename(record)[:-len(".json")]
        self.assertIn(key, _read(self.ledger()))

    def test_record_missing_while_its_key_is_enrolled_refuses_and_is_not_rebuilt(self):
        repo = self.repo(default="trunk")
        os.unlink(self._enroll(repo))
        ledger = _read(self.ledger())
        self._assert_refused_everywhere(repo, "ledger")
        self.assertEqual(self.records(), [])
        self.assertEqual(_read(self.ledger()), ledger)

    def test_ledger_missing_while_records_exist_refuses(self):
        repo = self.repo(default="trunk")
        record = self._enroll(repo)
        content = _read(record)
        os.unlink(self.ledger())
        self._assert_refused_everywhere(repo, "ledger")
        self.assertFalse(os.path.exists(self.ledger()))
        self.assertEqual(_read(record), content)

    def test_ledger_missing_refuses_another_repository_too(self):
        first = self.repo(default="trunk", name="one")
        self._enroll(first)
        os.unlink(self.ledger())
        second = self.repo(default="trunk", name="two")
        self._assert_refused_everywhere(second, "ledger")

    def test_explicit_reset_enrolls_again(self):
        repo = self.repo(default="trunk")
        self._enroll(repo)
        shutil.rmtree(os.path.dirname(self.ledger()))
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertEqual(len(self.records()), 1)

    def test_operator_override_does_not_bypass_a_missing_record(self):
        repo = self.repo(default="trunk")
        os.unlink(self._enroll(repo))
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "trunk"}):
            self._assert_refused_everywhere(repo, "ledger")
        self.assertEqual(self.records(), [])

    def test_read_only_run_does_not_need_the_ledger(self):
        repo = self.repo(default="trunk")
        os.unlink(self._enroll(repo))
        self.assertTrue(self.run_child(repo, "true", read_only=True).get("ok"))


class DecoyCommonDir(_Base):
    """A linked worktree cannot point the verification at a copy of the repo."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _linked(self, default="release/v1"):
        repo = self.repo(default=default, pack=True)
        linked = os.path.join(self.root, "lw")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        return repo, os.path.realpath(linked)

    def test_commondir_rewritten_to_a_decoy_is_still_detected(self):
        repo, linked = self._linked()
        admin = os.path.join(repo, ".git", "worktrees", "lw")
        decoy = os.path.join(self.root, "decoy.git")
        script = " && ".join([
            "cp -R %s %s" % (_sh_quote(os.path.join(repo, ".git")), _sh_quote(decoy)),
            "printf '%%s\\n' %s > %s" % (_sh_quote(decoy), _sh_quote(
                os.path.join(admin, "commondir"))),
            "mkdir %s" % _sh_quote(os.path.join(self.root, "plant")),
            "printf '%%s\\n' %s > %s" % (self.other, _sh_quote(
                os.path.join(self.root, "plant", "v1"))),
            "mv %s %s" % (_sh_quote(os.path.join(self.root, "plant")),
                         _sh_quote(os.path.join(repo, ".git", "refs", "heads", "release"))),
        ])
        env = self.run_child(linked, script, foreman=True)
        self.assertEqual(_read(os.path.join(admin, "commondir")), decoy + "\n")
        self.assertTrue(os.path.exists(os.path.join(repo, ".git/refs/heads/release/v1")))
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn("refs/heads/release/v1", env["detail"])

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_fenced_child_cannot_write_commondir_or_gitdir(self):
        repo, linked = self._linked(default="trunk")
        admin = os.path.join(repo, ".git", "worktrees", "lw")
        before = {name: _read(os.path.join(admin, name)) for name in ("commondir", "gitdir")}
        script = ("printf x > %s; printf x > %s; : > %s; rm -f %s" % tuple(
            _sh_quote(path) for path in (
                os.path.join(admin, "commondir"), os.path.join(admin, "gitdir"),
                os.path.join(repo, ".git", "commondir"), os.path.join(admin, "gitdir"))))
        self.run_child(linked, script)
        for name, content in before.items():
            self.assertEqual(_read(os.path.join(admin, name)), content, name)
        self.assertFalse(os.path.lexists(os.path.join(repo, ".git", "commondir")))

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_main_checkout_commondir_cannot_be_planted(self):
        repo = self.repo(default="trunk")
        self.run_child(repo, ": > %s" % _sh_quote(os.path.join(repo, ".git", "commondir")))
        self.assertFalse(os.path.lexists(os.path.join(repo, ".git", "commondir")))

    def test_planted_commondir_in_the_real_common_dir_is_not_followed(self):
        repo = self.repo(default="trunk")
        decoy = os.path.join(self.root, "decoy.git")
        script = "cp -R %s %s; printf '%%s\\n' %s > %s" % (
            _sh_quote(os.path.join(repo, ".git")), _sh_quote(decoy), _sh_quote(decoy),
            _sh_quote(os.path.join(repo, ".git", "commondir")))
        env = self.run_child(repo, script, foreman=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "protected_ref_unverified", env)


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class RecordContainment(_Base):
    """The record is out of reach in EVERY fenced profile, ancestors included."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _fenced(self, workdir, script):
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", script], provider="claude", workdir=workdir,
            env=dict(os.environ))
        return subprocess.run(argv, cwd=workdir, env=env, capture_output=True,
                              text=True, check=False)

    def _tree(self, top):
        found = []
        for current, dirs, files in os.walk(top):
            found.extend(os.path.join(current, name) for name in dirs + files)
        return sorted(found)

    def test_state_dir_inside_a_writable_non_git_workdir(self):
        plain = os.path.join(self.root, "plain")
        inside = os.path.join(plain, "outer", "state")
        os.makedirs(inside)
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": inside}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            record, = glob.glob(os.path.join(inside, "protected-refs", "*.json"))
            content = _read(record)
            ledger = _read(os.path.join(inside, "protected-refs", "ledger"))
            tree = self._tree(plain)
            probe = self._fenced(plain, "; ".join([
                "printf x > %s" % _sh_quote(record),
                "rm -f %s" % _sh_quote(record),
                ": > %s" % _sh_quote(os.path.join(inside, "protected-refs", "planted")),
                "printf x > %s" % _sh_quote(os.path.join(inside, "protected-refs", "ledger")),
                "rm -f %s" % _sh_quote(os.path.join(inside, "protected-refs", "ledger")),
            ]))
            self.assertNotEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(_read(record), content)
            self.assertEqual(_read(os.path.join(inside, "protected-refs", "ledger")), ledger)
            self.assertEqual(self._tree(plain), tree)

    def test_ancestor_of_the_record_dir_cannot_be_renamed_away(self):
        plain = os.path.join(self.root, "plain")
        inside = os.path.join(plain, "outer", "state")
        os.makedirs(inside)
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": inside}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            tree = self._tree(plain)
            probe = self._fenced(plain, "; ".join([
                "mv %s %s" % (_sh_quote(os.path.join(inside, "protected-refs")),
                              _sh_quote(os.path.join(inside, "pr.moved"))),
                "mv %s %s" % (_sh_quote(inside), _sh_quote(inside + ".moved")),
                "mv %s %s" % (_sh_quote(os.path.join(plain, "outer")),
                              _sh_quote(os.path.join(plain, "outer.moved"))),
            ]))
            self.assertNotEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(self._tree(plain), tree)
            self.assertTrue(glob.glob(os.path.join(inside, "protected-refs", "*.json")))

    def test_git_workdir_with_the_state_dir_one_level_up(self):
        repo = self.repo(default="trunk")
        outer = os.path.join(repo, "work-state")
        os.makedirs(outer)
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": os.path.join(outer, "s")}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            tree = self._tree(outer)
            probe = self._fenced(repo, "mv %s %s" % (
                _sh_quote(outer), _sh_quote(outer + ".moved")))
            self.assertNotEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(self._tree(outer), tree)


class CaseVariants(_Base):
    """Every ref spelled like a protected name counts (case, NFC/NFD)."""

    def _append_packed(self, line):
        return "printf '%%s\\n' %s >> \"$R/.git/packed-refs\"" % _sh_quote(line)

    def test_variant_spelling_created_in_packed_refs_is_a_change(self):
        repo = self.repo(pack=True)
        env = self.run_child(
            repo, self._append_packed(self.other + " refs/heads/MAIN"), foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn("refs/heads/MAIN", env["detail"])

    def test_mixed_case_default_is_recorded_as_spelled_and_watched(self):
        repo = self.repo(default="ReLeAsE/v1", pack=True)
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        data = json.loads(_read(self.records()[0]))
        self.assertIn("ReLeAsE/v1", data["protected"])
        env = self.run_child(
            repo, self._append_packed(self.other + " refs/heads/RELEASE/v1"), foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn("refs/heads/RELEASE/v1", env["detail"])

    def test_nfd_spelling_of_a_protected_name_is_a_change(self):
        nfc = unicodedata.normalize("NFC", "Rélease")
        nfd = unicodedata.normalize("NFD", "Rélease")
        repo = self.repo(default="x", name="repo")
        _git(repo, "config", "core.precomposeunicode", "false")
        _git(repo, "update-ref", "refs/heads/" + nfc, self.seed)
        _git(repo, "update-ref", "refs/remotes/origin/" + nfc, self.seed)
        _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/" + nfc)
        _git(repo, "pack-refs", "--all", "--prune")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        env = self.run_child(
            repo, self._append_packed(self.other + " refs/heads/" + nfd), foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)

    def test_a_default_branch_spelled_like_the_floor_is_not_dropped(self):
        repo = self.repo(default="Main")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        data = json.loads(_read(self.records()[0]))
        self.assertIn("Main", data["protected"])
        self.assertIn("main", data["protected"])

    def test_unrelated_branches_never_count(self):
        repo = self.repo(pack=True)
        env = self.run_child(
            repo, self._append_packed(self.other + " refs/heads/mainline")
            + "; printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/feature\"",
            foreman=True)
        self.assertTrue(env.get("ok"), env)


class CancelAndStop(_Base):
    """Cancellation and exceptions after the spawn still compare the refs."""

    def _plant(self):
        return "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\""

    def test_cancel_after_a_change_reports_the_change(self):
        repo = self.repo()
        env = self.run_child_cancelled(
            repo, self._plant() + "; sleep 30", foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)

    def test_cancel_without_a_change_still_cancels(self):
        repo = self.repo()
        with self.assertRaises(asyncio.CancelledError):
            self.run_child_cancelled(repo, "sleep 30", foreman=True)

    def test_exception_after_the_spawn_reports_the_change(self):
        repo = self.repo()

        async def boom(argv, **kwargs):
            _git(repo, "update-ref", "refs/heads/main", self.other)
            raise RuntimeError("boom")

        with mock.patch.object(agent_lane, "run_cli", boom):
            env = self.run_child(repo, "true", foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)

    def _stopped_job(self, fake_agent):
        repo = self.repo()
        _job_id, job_dir, _data = jobs._new_job(
            "t", "claude/sonnet", repo, 60, "default", False)

        async def go():
            task = asyncio.ensure_future(
                job_runner._execute(job_dir, jobs._read_job(job_dir)))
            await asyncio.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)
            return await task

        with mock.patch.object(job_runner, "run_agent", fake_agent), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            return asyncio.run(go()), job_dir

    def test_stop_does_not_overwrite_a_protected_ref_failure(self):
        async def fake(**kwargs):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return {"ok": False, "reason": "protected_ref_changed",
                        "detail": "the run changed a protected local ref (x)"}

        result, job_dir = self._stopped_job(fake)
        self.assertEqual(result["reason"], "protected_ref_changed", result)
        self.assertIn("stopped", result["detail"])
        self.assertEqual(jobs._read_job(job_dir)["state"], "failed")

    def test_stop_does_not_overwrite_an_unverified_failure(self):
        async def fake(**kwargs):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return {"ok": False, "reason": "protected_ref_unverified", "detail": "x"}

        result, _job_dir = self._stopped_job(fake)
        self.assertEqual(result["reason"], "protected_ref_unverified", result)

    def test_stop_still_wins_over_an_ordinary_failure(self):
        async def fake(**kwargs):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return {"ok": False, "reason": "timeout", "detail": "x"}

        result, job_dir = self._stopped_job(fake)
        self.assertEqual(result["reason"], "stopped", result)
        self.assertEqual(jobs._read_job(job_dir)["state"], "stopped")


class AllowMainForwarding(_Base):
    def test_start_persists_allow_main_and_the_runner_forwards_it(self):
        repo = self.repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        with mock.patch.object(jobs.subprocess, "Popen", side_effect=OSError("no spawn")):
            started = jobs.start("t", model="claude/sonnet", workdir=repo, allow_main=True)
        job = jobs._read_job(started["job_dir"])
        self.assertIs(job.get("allow_main"), True)
        fake = mock.AsyncMock(return_value={"ok": True, "text": "x"})
        with mock.patch.object(job_runner, "run_agent", fake), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(started["job_dir"], job))
        self.assertIs(fake.call_args.kwargs.get("allow_main"), True)

    def test_default_job_does_not_allow_main(self):
        repo = self.repo()
        with mock.patch.object(jobs.subprocess, "Popen", side_effect=OSError("no spawn")):
            started = jobs.start("t", model="claude/sonnet", workdir=repo)
        job = jobs._read_job(started["job_dir"])
        self.assertFalse(job.get("allow_main"))
        fake = mock.AsyncMock(return_value={"ok": True, "text": "x"})
        with mock.patch.object(job_runner, "run_agent", fake), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(started["job_dir"], job))
        self.assertNotIn("allow_main", fake.call_args.kwargs)


class MaskedRefusals(_Base):
    SECRET = "sk-ant-api03-" + "Zq9xW3kL" * 6

    def test_invalid_override_value_is_masked(self):
        repo = self.repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "main," + self.SECRET + ":x"}):
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
        detail = caught.exception.envelope["detail"]
        self.assertIn(OVERRIDE_ENV, detail)
        self.assertNotIn("Zq9xW3kL", detail)

    def test_agent_lane_refusal_is_masked(self):
        repo = self.repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET + ":x"}):
            env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertNotIn("Zq9xW3kL", env["detail"])

    @unittest.skipUnless(DARWIN, "host-exec only reaches the repository roots on macOS")
    def test_host_exec_prints_no_credential(self):
        repo = self.repo()
        err = io.StringIO()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET + ":x"}), \
                contextlib.redirect_stderr(err):
            rc = host_fence.main(["--provider", "claude", "--workdir", repo,
                                  "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self.assertIn("fence_path_rejected", err.getvalue())
        self.assertNotIn("Zq9xW3kL", err.getvalue())


# ---------------------------------------------------------------------------
# Fix round 2
# ---------------------------------------------------------------------------

class DirectoryMoveHole(_Base):
    """The kernel profile closes the directory-move hole; the compare is not the control."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _plant_script(self, spelling="release"):
        return ("mkdir \"$R/plant\"; printf '%%s\\n' %s > \"$R/plant/v1\"; "
                "mv \"$R/plant\" \"$R/.git/refs/heads/%s\"; echo mv=$? > \"$R/rc.txt\"; "
                "mkdir \"$R/.git/refs/heads/%s\"; echo mkdir=$? >> \"$R/rc.txt\"") % (
                    self.other, spelling, spelling)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_rename_and_mkdir_into_a_missing_parent_of_a_protected_name_are_denied(self):
        repo = self.repo(default="release/v1", pack=True)
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads/release")))
        env = self.run_child(repo, self._plant_script())
        rcs = _read(os.path.join(repo, "rc.txt"))
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads/release")), rcs)
        self.assertNotIn("mv=0", rcs)
        self.assertNotIn("mkdir=0", rcs)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/release/v1"), self.seed)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_the_deny_holds_for_the_case_spellings(self):
        repo = self.repo(default="release/v1", pack=True)
        self.run_child(repo, self._plant_script("RELEASE"))
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads/release")))

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_slash_work_branches_still_commit(self):
        repo = self.repo(default="release/v1", pack=True)
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/feature/x")
        _git(repo, "update-ref", "refs/heads/feature/x", self.seed)
        script = ("/usr/bin/git -C \"$R\" -c user.name=t -c user.email=t@example.invalid "
                  "commit --allow-empty -qm child && "
                  "/usr/bin/git -C \"$R\" update-ref refs/heads/topic/y HEAD")
        env = self.run_child(repo, script)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "log", "-1", "--format=%s", "refs/heads/feature/x"), "child")
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/topic/y"),
                         _git(repo, "rev-parse", "refs/heads/feature/x"))

    def test_same_group_descendant_cannot_plant_after_the_leader_exits(self):
        repo = self.repo()
        script = ("( sleep 1.5; printf '%s\\n' " + self.other
                  + " > \"$R/.git/refs/heads/main\" ) >/dev/null 2>&1 &")
        env = self.run_child(repo, script, foreman=True)
        self.assertTrue(env.get("ok"), env)
        time.sleep(3.0)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)

    def test_a_change_made_while_the_reply_is_read_is_still_caught(self):
        repo = self.repo()

        def late_attest(*args, **kwargs):
            _git(repo, "update-ref", "refs/heads/main", self.other)
            return None

        with mock.patch.object(agent_lane, "read_successor_attest", late_attest):
            env = self.run_child(repo, "true", foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class FirmlinkSpelling(_Base):
    """A state dir spelled through /System/Volumes/Data is denied under its plain spelling too."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def test_record_and_ancestors_are_denied_under_the_plain_spelling(self):
        plain = os.path.join(self.root, "plain")
        inside = os.path.join(plain, "outer", "state")
        os.makedirs(inside)
        firm = "/System/Volumes/Data" + inside
        self.assertTrue(os.path.samestat(os.stat(firm), os.stat(inside)))
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": firm}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            record, = glob.glob(os.path.join(inside, "protected-refs", "*.json"))
            content = _read(record)
            tree = sorted(os.path.join(c, n) for c, d, f in os.walk(plain) for n in d + f)
            argv, env, _meta = host_fence.prepare_host_fence(
                ["/bin/sh", "-c", "; ".join([
                    "printf x > %s" % _sh_quote(record),
                    "rm -f %s" % _sh_quote(record),
                    "mv %s %s" % (_sh_quote(os.path.join(plain, "outer")),
                                  _sh_quote(os.path.join(plain, "outer.moved"))),
                ])],
                provider="claude", workdir=plain, env=dict(os.environ))
            probe = subprocess.run(argv, cwd=plain, env=env, capture_output=True, text=True)
            self.assertNotEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(_read(record), content)
            self.assertEqual(
                sorted(os.path.join(c, n) for c, d, f in os.walk(plain) for n in d + f), tree)

    def test_plain_spelling_helper_matches_prefix_case_insensitively(self):
        rows = host_fence.record_dir_denies()
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": "/system/volumes/DATA" + self.state}):
            spelled = host_fence.record_dir_denies()
        self.assertIn(("subpath", os.path.join(self.state, "protected-refs")), spelled)
        self.assertTrue(rows)


class AllowMainExactRef(_Base):
    def test_only_the_exact_checked_out_ref_is_exempt(self):
        repo = self.repo(pack=True)
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        script = "printf '%s\\n' '" + self.other + " refs/heads/MAIN' >> \"$R/.git/packed-refs\""
        env = self.run_child(repo, script, foreman=True, allow_main=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertIn("refs/heads/MAIN", env["detail"])

    def test_the_exact_checked_out_ref_itself_is_still_exempt(self):
        repo = self.repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        env = self.run_child(
            repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/main\"",
            foreman=True, allow_main=True)
        self.assertTrue(env.get("ok"), env)


class UnlistableRefs(_Base):
    """A protected ref `for-each-ref` cannot represent is not "absent".

    git skips a loose ref whose content is not an object id (and a dangling
    symbolic ref), so creating or rewriting one left both snapshots empty.
    """

    GARBAGE = "zzzz not an object id"

    def test_creating_an_unlistable_ref_is_unverified(self):
        repo = self.repo()
        _git(repo, "update-ref", "-d", "refs/heads/master")
        env = self.run_child(
            repo, "printf '%s\\n' '" + self.GARBAGE + "' > \"$R/.git/refs/heads/master\"",
            foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_unverified", env)
        self.assertIn("refs/heads/master", env["detail"])

    def test_rewriting_a_ref_into_an_unlistable_one_is_unverified(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' '" + self.GARBAGE + "' > \"$R/.git/refs/heads/main\"",
            foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_unverified", env)

    def test_a_dangling_symbolic_ref_is_unverified(self):
        repo = self.repo()
        _git(repo, "update-ref", "-d", "refs/heads/master")
        env = self.run_child(
            repo, "printf 'ref: refs/heads/nowhere\\n' > \"$R/.git/refs/heads/master\"",
            foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_unverified", env)

    def test_an_unlistable_variant_spelling_in_packed_refs_is_unverified(self):
        repo = self.repo(pack=True)
        env = self.run_child(
            repo, "printf '%s\\n' 'refs/heads/MAIN' >> \"$R/.git/packed-refs\"",
            foreman=True)
        self.assertFalse(env.get("ok"), env)
        self.assertIn(env.get("reason"), ("protected_ref_unverified", "protected_ref_changed"))

    def test_an_unlistable_protected_ref_before_the_run_refuses_it(self):
        repo = self.repo()
        with open(os.path.join(repo, ".git/refs/heads/main"), "w") as handle:
            handle.write(self.GARBAGE + "\n")
        env = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
        self.assertFalse(env.get("ok"), env)
        self.assertFalse(os.path.exists(os.path.join(repo, "ran")))

    def test_genuine_absence_is_not_unverified(self):
        repo = self.repo()
        _git(repo, "update-ref", "-d", "refs/heads/master")
        env = self.run_child(repo, "true", foreman=True)
        self.assertTrue(env.get("ok"), env)

    def test_a_listable_ref_with_a_missing_object_is_still_a_plain_change(self):
        repo = self.repo()
        env = self.run_child(
            repo, "printf '%s\\n' 0123456789abcdef0123456789abcdef01234567 "
            "> \"$R/.git/refs/heads/main\"", foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)


class LedgerPathEncoding(_Base):
    def test_a_carriage_return_in_the_common_dir_path_does_not_corrupt_the_ledger(self):
        weird = self.repo(default="trunk", name="re\rpo")
        plain = self.repo(default="trunk", name="plain")
        self.assertIsNone(agent_lane._workdir_fence(weird, False))
        self.assertIsNone(agent_lane._workdir_fence(plain, False))
        self.assertEqual(len(self.records()), 2)
        # ... and both are read back afterwards.
        self.assertIsNone(agent_lane._workdir_fence(weird, False))
        self.assertIsNone(agent_lane._workdir_fence(plain, False))
        for line in _read(self.ledger()).split("\n"):
            if line:
                self.assertEqual(sorted(json.loads(line)), ["common_dir", "key"])


# ---------------------------------------------------------------------------
# Fix round 4
# ---------------------------------------------------------------------------

class OverrideStillEnrolls(_Base):
    """The override changes the answer, not the record: an unseen repository is enrolled."""

    def test_override_run_on_an_unseen_repository_writes_record_and_ledger(self):
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertEqual(len(self.records()), 1)
        self.assertTrue(os.path.exists(self.ledger()))
        data = json.loads(_read(self.records()[0]))
        self.assertIn("trunk", data["protected"])
        self.assertIn("release", data["protected"])
        # The child retargets origin/HEAD; an unset run uses the recorded names.
        _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/other")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")

    def test_override_run_with_a_broken_ledger_is_refused_before_the_spawn(self):
        repo = self.repo(default="trunk")
        os.makedirs(os.path.dirname(self.ledger()))
        with open(self.ledger(), "w") as handle:
            handle.write("not json at all\n")
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            env = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
            self.assertEqual(env.get("reason"), "fence_path_rejected", env)
            self.assertIn("delete", env["detail"])
            self.assertFalse(os.path.exists(os.path.join(repo, "ran")))
            with self.assertRaises(host_fence.HostFenceError):
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertEqual(self.records(), [])
        self.assertEqual(_read(self.ledger()), "not json at all\n")

    def _override_with_an_unreadable_origin_head(self, content):
        repo = self.repo(default="trunk")
        with open(os.path.join(repo, ".git/refs/remotes/origin/HEAD"), "w") as handle:
            handle.write(content)
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertEqual(len(self.records()), 1)
        data = json.loads(_read(self.records()[0]))
        self.assertEqual(sorted(data["protected"]), ["main", "master", "release"])
        # The child retargets origin/HEAD; an unset run still refuses the recorded name.
        os.unlink(os.path.join(repo, ".git/refs/remotes/origin/HEAD"))
        _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/other")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/release")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")

    def test_override_with_a_garbage_origin_head_records_the_override_names(self):
        self._override_with_an_unreadable_origin_head("garbage\n")

    def test_override_with_a_direct_oid_origin_head_records_the_override_names(self):
        self._override_with_an_unreadable_origin_head(
            "0123456789abcdef0123456789abcdef01234567\n")

    def test_override_never_overwrites_an_existing_record(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        record, = self.records()
        before = _read(record)
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertEqual(self.records(), [record])
        self.assertEqual(_read(record), before)


class IdentityKeyedRecord(_Base):
    """The record is keyed by the common dir's device/inode, not by its path."""

    @unittest.skipUnless(DARWIN, "the firmlink spelling exists on macOS only")
    def test_firmlink_and_plain_spellings_share_one_record(self):
        repo = self.repo(default="trunk")
        firm = "/System/Volumes/Data" + repo
        self.assertTrue(os.path.samestat(os.stat(firm), os.stat(repo)))
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertIsNone(agent_lane._workdir_fence(firm, False))
        self.assertEqual(len(self.records()), 1)
        data = json.loads(_read(self.records()[0]))
        self.assertEqual(data["common_dir"], os.path.join(repo, ".git"))

    @unittest.skipUnless(DARWIN, "needs a case-insensitive volume")
    def test_a_recased_parent_component_is_the_same_repository(self):
        repo = self.repo(default="trunk")
        parent, leaf = os.path.split(repo)
        recased = os.path.join(parent.upper().replace("/PRIVATE/TMP", "/private/tmp"), leaf)
        if recased == repo or not os.path.exists(recased):
            self.skipTest("volume is case-sensitive")
        self.assertTrue(os.path.samestat(os.stat(recased), os.stat(repo)))
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertIsNone(agent_lane._workdir_fence(recased, False))
        self.assertEqual(len(self.records()), 1)

    def test_a_missing_stored_path_refuses(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        record, = self.records()
        data = json.loads(_read(record))
        data["common_dir"] = os.path.join(self.root, "nowhere", ".git")
        with open(record, "w") as handle:
            json.dump(data, handle)
        env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected", env)

    def test_a_stored_path_that_is_another_directory_refuses(self):
        repo = self.repo(default="trunk")
        other = self.repo(default="trunk", name="other")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        record, = self.records()
        data = json.loads(_read(record))
        data["common_dir"] = os.path.join(other, ".git")
        with open(record, "w") as handle:
            json.dump(data, handle)
        env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected", env)
        self.assertIn("moved", env["detail"])

    def test_a_moved_common_dir_refuses_instead_of_enrolling(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        record, ledger = self.records(), _read(self.ledger())
        moved = repo + "-moved"
        os.rename(repo, moved)
        env = agent_lane._workdir_fence(moved, False)
        self.assertEqual(env["reason"], "fence_path_rejected", env)
        self.assertIn("reset", env["detail"])
        with self.assertRaises(host_fence.HostFenceError):
            host_fence._repo_write_roots(moved, required=False, allow_main=False)
        self.assertEqual(self.records(), record)
        self.assertEqual(_read(self.ledger()), ledger)

    def test_a_new_repository_at_an_old_path_is_a_new_first_sight(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        os.rename(repo, repo + "-old")
        again = self.repo(default="trunk")
        self.assertEqual(again, repo)
        self.assertIsNone(agent_lane._workdir_fence(again, False))
        self.assertEqual(len(self.records()), 2)

    def test_a_legacy_path_keyed_record_is_refused_with_the_reset_hint(self):
        import hashlib
        repo = self.repo(default="trunk")
        common = os.path.join(repo, ".git")
        st = os.stat(common)
        key = hashlib.sha256(os.fsencode(common)).hexdigest()[:32]
        directory = os.path.join(self.state, "protected-refs")
        os.makedirs(directory)
        with open(os.path.join(directory, key + ".json"), "w") as handle:
            json.dump({"version": 1, "common_dir": common, "dev": st.st_dev,
                       "ino": st.st_ino, "protected": ["main", "master", "trunk"]}, handle)
        with open(os.path.join(directory, "ledger"), "w") as handle:
            handle.write(json.dumps({"key": key, "common_dir": common}) + "\n")
        before = sorted(n for n in os.listdir(directory) if not n.startswith("."))
        env = agent_lane._workdir_fence(repo, False)
        self.assertEqual(env["reason"], "fence_path_rejected", env)
        self.assertIn("legacy", env["detail"])
        self.assertIn("reset", env["detail"])
        self.assertEqual(sorted(n for n in os.listdir(directory) if not n.startswith(".")),
                         before)


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class RepoRootRename(_Base):
    """The repository root and every ancestor cannot be renamed away in the fence.

    A destination outside the repository is not writable in the fence, so the
    move that works without a deny goes into the child's own temp dir. That
    dir outlives the test, hence unique destination names and a cleanup.
    """

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()
        self.tag = os.path.basename(self.root)

    def _dest(self, tmpdir, name):
        path = os.path.join(tmpdir, "%s-%s" % (name, self.tag))
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def _host_exec(self, workdir, script):
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", script], provider="claude", workdir=workdir,
            env=dict(os.environ))
        proc = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True)
        return proc, env.get("TMPDIR")

    def _tree(self, top):
        return sorted(os.path.join(c, n) for c, d, f in os.walk(top) for n in d + f)

    def test_agent_lane_cannot_move_the_repository_root_into_its_own_temp_dir(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        tree = self._tree(repo)
        env = self.run_child(
            repo, "echo \"$TMPDIR\" > \"$R/tmpdir.txt\"; "
            "mv \"$R\" \"$TMPDIR/moved-%s\"; echo mv=$? > \"$R/rc.txt\"" % self.tag)
        self.assertTrue(env.get("ok"), env)
        tmpdir = _read(os.path.join(repo, "tmpdir.txt")).strip()
        self.assertTrue(tmpdir)
        dest = self._dest(tmpdir, "moved")
        self.assertNotIn("mv=0", _read(os.path.join(repo, "rc.txt")))
        self.assertFalse(os.path.exists(dest))
        self.assertTrue(os.path.isdir(os.path.join(repo, ".git")))
        self.assertEqual([t for t in self._tree(repo)
                          if not t.endswith(("tmpdir.txt", "rc.txt"))], tree)

    def test_host_exec_cannot_move_the_repository_root_or_its_parent(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        tree = self._tree(self.root)
        probe, tmpdir = self._host_exec(
            repo, "mv %s \"$TMPDIR/repo-%s\"; mv %s \"$TMPDIR/up-%s\"" % (
                _sh_quote(repo), self.tag, _sh_quote(self.root), self.tag))
        self.assertTrue(tmpdir)
        repo_dest, up_dest = self._dest(tmpdir, "repo"), self._dest(tmpdir, "up")
        self.assertNotEqual(probe.returncode, 0, probe.stderr)
        self.assertFalse(os.path.exists(repo_dest))
        self.assertFalse(os.path.exists(up_dest))
        self.assertEqual(self._tree(self.root), tree)

    def test_linked_worktree_root_and_main_checkout_root_cannot_be_moved(self):
        repo = self.repo(default="trunk")
        linked = os.path.join(self.root, "lw")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        linked = os.path.realpath(linked)
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        tree = self._tree(self.root)
        probe, tmpdir = self._host_exec(
            linked, "mv %s \"$TMPDIR/lw-%s\"; mv %s \"$TMPDIR/main-%s\"" % (
                _sh_quote(linked), self.tag, _sh_quote(repo), self.tag))
        lw_dest, main_dest = self._dest(tmpdir, "lw"), self._dest(tmpdir, "main")
        self.assertNotEqual(probe.returncode, 0, probe.stderr)
        self.assertFalse(os.path.exists(lw_dest))
        self.assertFalse(os.path.exists(main_dest))
        self.assertEqual(self._tree(self.root), tree)


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class RefsHeadsAlwaysDenied(_Base):
    """``refs`` and ``refs/heads`` of the common dir are denied even when missing."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _staged_move(self, repo):
        staged = _sh_quote(os.path.join(repo, "staged"))
        return ("mkdir %s; printf '%%s\\n' %s > %s; mv %s %s; echo mv=$? > %s" % (
            staged, self.other, _sh_quote(os.path.join(repo, "staged", "main")), staged,
            _sh_quote(os.path.join(repo, ".git", "refs", "heads")),
            _sh_quote(os.path.join(repo, "rc.txt"))))

    def _repo_without_refs_heads(self):
        repo = self.repo(pack=True)
        heads = os.path.join(repo, ".git", "refs", "heads")
        for current, dirs, files in os.walk(heads, topdown=False):
            for name in files:
                os.unlink(os.path.join(current, name))
            os.rmdir(current)
        self.assertFalse(os.path.exists(heads))
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)
        return repo

    def test_agent_lane_cannot_move_a_staged_directory_into_a_missing_refs_heads(self):
        repo = self._repo_without_refs_heads()
        env = self.run_child(repo, self._staged_move(repo))
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("mv=0", _read(os.path.join(repo, "rc.txt")))
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads")), env)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)

    def test_host_exec_cannot_either(self):
        repo = self._repo_without_refs_heads()
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", self._staged_move(repo)],
            provider="claude", workdir=repo, env=dict(os.environ))
        subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True)
        self.assertNotIn("mv=0", _read(os.path.join(repo, "rc.txt")))
        self.assertFalse(os.path.exists(os.path.join(repo, ".git/refs/heads")))

    def test_the_rows_are_there_whether_or_not_the_directories_exist(self):
        repo = self._repo_without_refs_heads()
        denies = host_fence._repo_write_roots(repo, required=False, allow_main=False)[1]
        common = os.path.join(repo, ".git")
        self.assertIn(("literal", os.path.join(common, "refs", "heads")), denies)
        self.assertIn(("literal", os.path.join(common, "refs")), denies)

    def test_a_slash_work_branch_still_commits_under_the_always_denied_directory(self):
        repo = self.repo(default="trunk")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/feature/x")
        _git(repo, "update-ref", "refs/heads/feature/x", self.seed)
        denies = host_fence._repo_write_roots(repo, required=False, allow_main=False)[1]
        self.assertIn(("literal", os.path.join(repo, ".git", "refs", "heads")), denies)
        env = self.run_child(
            repo, "/usr/bin/git -C \"$R\" -c user.name=t -c user.email=t@example.invalid "
            "commit --allow-empty -qm child")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(_git(repo, "log", "-1", "--format=%s", "refs/heads/feature/x"), "child")


class SymbolicProtectedName(_Base):
    """A protected local name that is a symbolic ref is refused at admission."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _symbolic(self, repo, kind):
        if kind == "single":
            _git(repo, "update-ref", "refs/heads/alias", self.seed)
            _git(repo, "symbolic-ref", "refs/heads/main", "refs/heads/alias")
        elif kind == "chain":
            _git(repo, "update-ref", "refs/heads/end", self.seed)
            _git(repo, "symbolic-ref", "refs/heads/mid", "refs/heads/end")
            _git(repo, "symbolic-ref", "refs/heads/main", "refs/heads/mid")
        else:
            _git(repo, "symbolic-ref", "refs/heads/main", "refs/heads/nowhere")

    def test_every_shape_is_refused_in_both_lanes(self):
        for kind in ("single", "chain", "dangling"):
            for allow_main in (False, True):
                with self.subTest(kind=kind, allow_main=allow_main):
                    repo = self.repo(default="trunk", name="r-%s-%s" % (kind, allow_main))
                    self._symbolic(repo, kind)
                    env = agent_lane._workdir_fence(repo, allow_main)
                    self.assertIsNotNone(env)
                    self.assertEqual(env["reason"], "fence_path_rejected", env)
                    self.assertIn("symbolic", env["detail"])
                    with self.assertRaises(host_fence.HostFenceError) as caught:
                        host_fence._repo_write_roots(
                            repo, required=False, allow_main=allow_main)
                    self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")
                    self.assertIn("symbolic", caught.exception.envelope["detail"])

    def test_the_run_is_refused_before_the_spawn_even_with_allow_main(self):
        repo = self.repo(default="trunk")
        self._symbolic(repo, "single")
        env = self.run_child(repo, ": > \"$R/ran\"", foreman=True, allow_main=True)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertFalse(os.path.exists(os.path.join(repo, "ran")))

    def test_detached_head_does_not_skip_the_check(self):
        repo = self.repo(default="trunk")
        self._symbolic(repo, "dangling")
        _git(repo, "update-ref", "--no-deref", "HEAD", self.seed)
        env = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)

    def test_the_remote_head_symbolic_ref_is_normal(self):
        repo = self.repo(default="trunk")
        self.assertEqual(_git(repo, "symbolic-ref", "refs/remotes/origin/HEAD"),
                         "refs/remotes/origin/trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertTrue(host_fence._repo_write_roots(
            repo, required=False, allow_main=False)[0])

    def test_read_only_runs_are_not_refused(self):
        repo = self.repo(default="trunk")
        self._symbolic(repo, "single")
        self.assertIsNone(agent_lane._workdir_fence(repo, False, True))


class LateCompare(_Base):
    """The compare after the reply is read is not gated on a successful envelope."""

    def _flip_on_second_call(self):
        calls = []

        def failure(self_, workspace_writes=None, note=None):
            calls.append(1)
            if len(calls) < 2:
                return None
            return {"ok": False, "reason": "protected_ref_changed", "detail": "late",
                    "workspace_writes": workspace_writes}
        return calls, failure

    def test_an_integrity_reason_wins_over_no_workspace_writes(self):
        repo = self.repo()
        calls, failure = self._flip_on_second_call()
        with mock.patch.object(agent_lane._ProtectedRefs, "failure", failure):
            env = self.run_child(repo, "true", foreman=True, expect_writes=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self.assertEqual(len(calls), 2)

    def test_an_integrity_reason_wins_over_unknown_workspace_writes(self):
        repo = self.repo()
        calls, failure = self._flip_on_second_call()
        with mock.patch.object(agent_lane._ProtectedRefs, "failure", failure), \
                mock.patch.object(agent_lane, "_workspace_writes",
                                  return_value=(None, "cannot count")):
            env = self.run_child(repo, "true", foreman=True, expect_writes=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)

    def test_job_runner_stop_keeps_the_late_integrity_reason(self):
        late = {"ok": False, "reason": "protected_ref_changed", "detail": "late"}
        kept = job_runner._after_stop(late)
        self.assertEqual(kept["reason"], "protected_ref_changed")
        self.assertIn("stopped", kept["detail"])


class ProcessGroupsClearedOnce(_Base):
    def test_exactly_one_clear_per_run_and_before_the_workspace_walk(self):
        repo = self.repo()
        log = []
        real_clear, real_walk = agent_lane._clear_process_groups, agent_lane._workspace_writes

        def clear(*args, **kwargs):
            log.append("clear")
            return real_clear(*args, **kwargs)

        def walk(*args, **kwargs):
            log.append("walk")
            return real_walk(*args, **kwargs)

        with mock.patch.object(agent_lane, "_clear_process_groups", clear), \
                mock.patch.object(agent_lane, "_workspace_writes", walk):
            env = self.run_child(repo, "true", foreman=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(log, ["clear", "walk"])

    def test_descendants_are_cleared_after_a_failed_leader_too(self):
        repo = self.repo()
        script = ("( sleep 1.5; printf '%s\\n' " + self.other
                  + " > \"$R/.git/refs/heads/main\" ) >/dev/null 2>&1 & exit 3")
        env = self.run_child(repo, script, foreman=True)
        self.assertFalse(env.get("ok"), env)
        time.sleep(3.0)
        self.assertEqual(_git(repo, "rev-parse", "refs/heads/main"), self.seed)


# ---------------------------------------------------------------------------
# Fix round 5
# ---------------------------------------------------------------------------

class NoSymlinkUnderRefs(_Base):
    """A symlink anywhere under <common dir>/refs refuses write mode, in both lanes."""

    def _alias_file(self, repo):
        main = os.path.join(repo, ".git/refs/heads/main")
        os.unlink(main)
        os.symlink("work", main)

    def _alias_dir(self, repo):
        elsewhere = os.path.join(self.root, "elsewhere")
        os.makedirs(elsewhere)
        os.symlink(elsewhere, os.path.join(repo, ".git/refs/heads/sub"))

    def _assert_refused(self, repo):
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "fence_path_rejected", env)
        self.assertIn("symlink", env["detail"])
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertIn("symlink", caught.exception.envelope["detail"])
        run = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
        self.assertEqual(run.get("reason"), "fence_path_rejected", run)
        self.assertFalse(os.path.exists(os.path.join(repo, "ran")))
        # read-only runs are not refused
        self.assertIsNone(agent_lane._workdir_fence(repo, False, True))

    def test_a_symlinked_ref_file_is_refused(self):
        repo = self.repo(default="trunk")
        self._alias_file(repo)
        self._assert_refused(repo)

    def test_a_symlinked_ref_directory_is_refused(self):
        repo = self.repo(default="trunk")
        self._alias_dir(repo)
        self._assert_refused(repo)

    def test_snapshot_refuses_too(self):
        repo = self.repo(default="trunk")
        self._alias_dir(repo)
        snap, refusal = agent_lane._snapshot_protected_refs(repo, False)
        self.assertIsNone(snap)
        self.assertIn("symlink", refusal["detail"])

    def test_a_clean_refs_tree_is_not_refused(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertIsNone(host_fence.ref_alias_refusal(os.path.join(repo, ".git")))


class SymrefUnderEverySpelling(_Base):
    """Any spelling of a protected name that is a symbolic ref refuses write mode."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    @staticmethod
    def _fake_git(listing):
        def run(*args):
            if args[0] == "for-each-ref":
                return subprocess.CompletedProcess(args, 0, stdout=listing, stderr="")
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return run

    def test_host_sweep_folds_case_and_normalization(self):
        nfd = unicodedata.normalize("NFD", "Rélease")
        for name, spelled in (("main", "MaIn"), ("master", "MAIN".replace("MAIN", "MASTER")),
                              ("Rélease", nfd)):
            with self.subTest(spelled=spelled):
                listing = "refs/heads/other\0\nrefs/heads/%s\0refs/heads/alias\n" % spelled
                why = host_fence.protected_symref_refusal(self._fake_git(listing), [name])
                self.assertTrue(why and "symbolic" in why, why)

    def test_host_sweep_ignores_unrelated_and_non_symbolic_refs(self):
        listing = ("refs/heads/mainline\0refs/heads/alias\nrefs/heads/MaIn\0\n"
                   "refs/remotes/origin/HEAD\0refs/remotes/origin/main\n")
        self.assertIsNone(host_fence.protected_symref_refusal(
            self._fake_git(listing), ["main", "master"]))

    def test_agent_lane_snapshot_refuses_a_symref_in_any_spelling(self):
        repo = self.repo(default="trunk")
        real = agent_lane.git_safe.run_git

        def fake(workdir, args, **kwargs):
            proc = real(workdir, args, **kwargs)
            if list(args)[:1] == ["for-each-ref"] and "%(objectname)" in list(args)[1]:
                extra = "refs/heads/MaIn\0%s\0refs/heads/alias\n" % self.seed
                return subprocess.CompletedProcess(
                    proc.args, 0, stdout=proc.stdout + extra, stderr=proc.stderr)
            return proc

        with mock.patch.object(agent_lane.git_safe, "run_git", fake):
            snap, refusal = agent_lane._snapshot_protected_refs(repo, False)
        self.assertIsNone(snap)
        self.assertEqual(refusal["reason"], "fence_path_rejected")
        self.assertIn("symbolic", refusal["detail"])

    def test_agent_lane_snapshot_refuses_when_the_exempt_branch_is_a_symref(self):
        repo = self.repo(default="trunk")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        real = agent_lane.git_safe.run_git

        def fake(workdir, args, **kwargs):
            proc = real(workdir, args, **kwargs)
            if list(args)[:1] == ["for-each-ref"] and "%(objectname)" in list(args)[1]:
                text = proc.stdout.replace(
                    "refs/heads/main\0%s\0\n" % self.seed,
                    "refs/heads/main\0%s\0refs/heads/master\n" % self.seed)
                return subprocess.CompletedProcess(proc.args, 0, stdout=text, stderr="")
            return proc

        with mock.patch.object(agent_lane.git_safe, "run_git", fake):
            snap, refusal = agent_lane._snapshot_protected_refs(repo, True)
        self.assertIsNone(snap)
        self.assertIn("symbolic", refusal["detail"])

    def _case_sensitive_root(self):
        probe = os.path.join(self.root, "probe-a")
        os.makedirs(probe)
        if os.path.exists(os.path.join(self.root, "PROBE-A")):
            self.skipTest("volume is case-insensitive")

    def test_a_mixed_case_symref_refuses_on_a_case_sensitive_volume(self):
        self._case_sensitive_root()
        repo = self.repo(default="trunk")
        _git(repo, "symbolic-ref", "refs/heads/MaIn", "refs/heads/work")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertIn("symbolic", env["detail"])
        with self.assertRaises(host_fence.HostFenceError):
            host_fence._repo_write_roots(repo, required=False, allow_main=False)

    def test_fence_refusal_text_is_masked(self):
        text = agent_lane._fence_refusal("branch sk-ant-api03-" + "Zq9xW3kL" * 6 + " is odd")
        self.assertNotIn("Zq9xW3kL", text["detail"])

    def test_the_remote_head_symref_stays_normal(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertTrue(host_fence._repo_write_roots(
            repo, required=False, allow_main=False)[0])


class StateDirHops(_Base):
    """Every hop of a symlinked state dir path (and its ancestors) is denied."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _hops(self):
        wd = os.path.join(self.root, "wd")
        os.makedirs(os.path.join(wd, "real"))
        os.symlink("real", os.path.join(wd, "hop"))
        os.symlink("hop", os.path.join(wd, "state"))
        return wd

    def test_every_visited_path_has_a_literal_row(self):
        wd = self._hops()
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": os.path.join(wd, "state")}):
            rows = host_fence.record_dir_denies()
        for path in (os.path.join(wd, "hop"), os.path.join(wd, "state"),
                     os.path.join(wd, "real"), wd):
            self.assertIn(("literal", path), rows, path)

    def test_a_cycle_refuses(self):
        wd = os.path.join(self.root, "loop")
        os.makedirs(wd)
        os.symlink("b", os.path.join(wd, "a"))
        os.symlink("a", os.path.join(wd, "b"))
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": os.path.join(wd, "a")}):
            with self.assertRaises(host_fence.ProtectedListError):
                host_fence.record_dir_denies()

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_a_fenced_child_cannot_rename_a_hop(self):
        wd = self._hops()
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": os.path.join(wd, "state")}):
            self.assertIsNone(agent_lane._workdir_fence(repo, False))
            ledger = _read(os.path.join(wd, "real", "protected-refs", "ledger"))
            argv, env, _meta = host_fence.prepare_host_fence(
                ["/bin/sh", "-c", "mv %s %s; mv %s %s" % (
                    _sh_quote(os.path.join(wd, "hop")), _sh_quote(os.path.join(wd, "hop2")),
                    _sh_quote(os.path.join(wd, "state")), _sh_quote(os.path.join(wd, "state2")))],
                provider="claude", workdir=wd, env=dict(os.environ))
            probe = subprocess.run(argv, cwd=wd, env=env, capture_output=True, text=True)
            self.assertNotEqual(probe.returncode, 0, probe.stderr)
            self.assertFalse(os.path.exists(os.path.join(wd, "hop2")))
            self.assertFalse(os.path.exists(os.path.join(wd, "state2")))
            self.assertEqual(_read(os.path.join(wd, "state", "protected-refs", "ledger")), ledger)
            self.assertIsNone(agent_lane._workdir_fence(repo, False))


class MaskedSymrefRefusal(_Base):
    SECRET = "sk-ant-api03-" + "Zq9xW3kL" * 6

    def _symref_repo(self):
        repo = self.repo(default="trunk")
        _git(repo, "update-ref", "refs/heads/" + self.SECRET, self.seed)
        _git(repo, "symbolic-ref", "refs/heads/" + self.SECRET, "refs/heads/work")
        return repo

    def test_agent_envelope_and_host_error_do_not_carry_the_name(self):
        repo = self._symref_repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET}):
            env = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
            self.assertEqual(env.get("reason"), "fence_path_rejected", env)
            self.assertNotIn("Zq9xW3kL", json.dumps(env))
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
            self.assertNotIn("Zq9xW3kL", json.dumps(caught.exception.envelope))

    @unittest.skipUnless(DARWIN, "host-exec only reaches the repository roots on macOS")
    def test_host_exec_prints_no_credential(self):
        repo = self._symref_repo()
        err = io.StringIO()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET}), \
                contextlib.redirect_stderr(err):
            rc = host_fence.main(["--provider", "claude", "--workdir", repo,
                                  "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self.assertIn("symbolic", err.getvalue())
        self.assertNotIn("Zq9xW3kL", err.getvalue())


# ---------------------------------------------------------------------------
# Fix round 6
# ---------------------------------------------------------------------------

class MetadataSymlinks(_Base):
    """A symlinked file the protect denies name is a filesystem alias: write mode refuses."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()

    def _link(self, repo, leaf):
        path = os.path.join(repo, ".git", leaf)
        if leaf == "HEAD":
            # git only accepts a HEAD symlink that points into refs/ (legacy form).
            os.unlink(path)
            os.symlink("refs/heads/work", path)
            return
        real = os.path.join(self.root, "real-" + leaf)
        os.rename(path, real)
        os.symlink(real, path)

    def test_every_metadata_path_the_denies_name_is_checked_in_both_lanes(self):
        for leaf in ("packed-refs", "config", "hooks", "HEAD"):
            with self.subTest(leaf=leaf):
                repo = self.repo(default="trunk", name="r-" + leaf, pack=True)
                self._link(repo, leaf)
                env = agent_lane._workdir_fence(repo, False)
                self.assertIsNotNone(env, leaf)
                self.assertEqual(env["reason"], "fence_path_rejected", env)
                self.assertIn("symlink", env["detail"])
                with self.assertRaises(host_fence.HostFenceError) as caught:
                    host_fence._repo_write_roots(repo, required=False, allow_main=False)
                self.assertIn("symlink", caught.exception.envelope["detail"])

    def test_the_snapshot_refuses_too(self):
        repo = self.repo(default="trunk", pack=True)
        self._link(repo, "packed-refs")
        snap, refusal = agent_lane._snapshot_protected_refs(repo, False)
        self.assertIsNone(snap)
        self.assertIn("symlink", refusal["detail"])

    def test_a_symlinked_packed_refs_does_not_refuse_a_read_only_run(self):
        repo = self.repo(default="trunk", pack=True)
        self._link(repo, "packed-refs")
        self.assertIsNone(agent_lane._workdir_fence(repo, False, True))

    def test_a_clean_repository_passes(self):
        repo = self.repo(default="trunk", pack=True)
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertTrue(host_fence._repo_write_roots(
            repo, required=False, allow_main=False)[0])


class EmptyOverrideIsUnset(_Base):
    def test_a_list_that_strips_to_no_names_is_treated_as_unset(self):
        for value in (",", " , ,", ",,,"):
            with self.subTest(value=value):
                repo = self.repo(default="trunk", name="r%d" % len(value))
                with open(os.path.join(repo, ".git/refs/remotes/origin/HEAD"), "w") as handle:
                    handle.write("garbage\n")
                with mock.patch.dict(os.environ, {OVERRIDE_ENV: value}):
                    env = agent_lane._workdir_fence(repo, False)
                self.assertIsNotNone(env)
                self.assertIn("origin default branch", env["detail"])
                self.assertEqual(self.records(), [])

    def test_the_parser_returns_none_for_such_lists(self):
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: " , ,"}):
            self.assertIsNone(host_fence._override_names())


class MaskedFailureEgress(_Base):
    SECRET = "sk-ant-api03-" + "Zq9xW3kL" * 6

    def _repo(self):
        repo = self.repo(default="trunk")
        _git(repo, "update-ref", "refs/heads/" + self.SECRET, self.seed)
        return repo

    def _assert_clean(self, text):
        self.assertNotIn("Zq9xW3kL", text)
        self.assertNotIn("zq9xw3kl", text)

    def test_protected_ref_changed_detail_is_masked(self):
        repo = self._repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET}):
            env = self.run_child(
                repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/" + self.SECRET + "\"",
                foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_changed", env)
        self._assert_clean(json.dumps(env))

    def test_protected_ref_unverified_detail_is_masked(self):
        repo = self._repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET}):
            env = self.run_child(
                repo, "printf 'zzzz not an object id\\n' > \"$R/.git/refs/heads/" + self.SECRET + "\"",
                foreman=True)
        self.assertEqual(env.get("reason"), "protected_ref_unverified", env)
        self._assert_clean(json.dumps(env))

    def test_envelope_masker_covers_every_detail_the_lane_returns(self):
        out = agent_lane._stamp_fence({"ok": False, "reason": "x",
                                       "detail": "ref " + self.SECRET + " odd"})
        self._assert_clean(out["detail"])

    @unittest.skipUnless(DARWIN, "host-exec only reaches the repository roots on macOS")
    def test_host_exec_allow_repo_write_refusal_is_masked(self):
        repo = self._repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/" + self.SECRET)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.SECRET}), \
                contextlib.redirect_stderr(err), mock.patch.object(
                host_fence.os, "execvpe", side_effect=AssertionError("exec reached")):
            rc = host_fence.main(["--provider", "claude", "--workdir", repo,
                                  "--allow-repo-write", "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self.assertIn("protected branch", err.getvalue())
        self._assert_clean(err.getvalue())

    def test_host_main_masks_every_string_of_the_envelope(self):
        err = io.StringIO()
        boom = host_fence.HostFenceError({"ok": False, "reason": "fence_path_rejected",
                                          "detail": "branch " + self.SECRET})
        with mock.patch.object(host_fence, "prepare_host_fence", side_effect=boom), \
                mock.patch.object(host_fence, "_vendor_exec_path", side_effect=lambda c, *a: c), \
                contextlib.redirect_stderr(err):
            rc = host_fence.main(["--provider", "claude", "--workdir", self.root,
                                  "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self._assert_clean(err.getvalue())


class HardlinksOnEveryWritePath(_Base):
    def _protected_repo(self):
        repo = self.repo(default="trunk")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        os.link(os.path.join(repo, ".git/refs/heads/main"),
                os.path.join(repo, ".git/refs/heads/main-link"))
        return repo

    def test_ordinary_write_mode_on_a_protected_branch_refuses_a_hardlinked_ref(self):
        repo = self._protected_repo()
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertIn("hard-linked", caught.exception.envelope["detail"])

    def test_allow_repo_write_on_a_protected_branch_still_refuses(self):
        repo = self._protected_repo()
        with self.assertRaises(host_fence.HostFenceError):
            host_fence._repo_write_roots(repo, required=True, allow_main=False)

    @unittest.skipUnless(DARWIN, "needs the macOS host fence")
    def test_host_exec_refuses_before_the_spawn(self):
        repo = self._protected_repo()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(
                host_fence.os, "execvpe", side_effect=AssertionError("exec reached")):
            rc = host_fence.main(["--provider", "claude", "--workdir", repo,
                                  "--", "/usr/bin/touch", os.path.join(repo, "ran")])
        self.assertEqual(rc, 126)
        self.assertFalse(os.path.exists(os.path.join(repo, "ran")))
        self.assertIn("hard-linked", err.getvalue())

    def test_a_protected_branch_without_hardlinks_still_returns_its_denies(self):
        repo = self.repo(default="trunk")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        roots, denies, _meta, _post, _final = host_fence._repo_write_roots(
            repo, required=False, allow_main=False)
        self.assertEqual(roots, ())
        self.assertTrue(denies)


class LooseSymrefFile(_Base):
    """A loose ref file that says ``ref:`` under a watched name is a symbolic ref.

    ``for-each-ref`` omits a dangling one and the exact-name probe asks for
    main / master only, so the files under refs/heads are read.
    """

    def _tree(self, name, content):
        self._trees = getattr(self, "_trees", 0) + 1
        common = os.path.join(self.root, "tree%d" % self._trees, ".git")
        heads = os.path.join(common, "refs", "heads")
        os.makedirs(heads)
        with open(os.path.join(heads, name), "w") as handle:
            handle.write(content)
        return common

    def test_unit_a_dangling_symref_in_a_folded_spelling_is_refused(self):
        common = self._tree("MaIn", "ref: refs/heads/unborn\n")
        why = host_fence.loose_symref_refusal(common, ["main", "master"])
        self.assertTrue(why and "symbolic" in why, why)

    def test_unit_nested_and_normalized_spellings(self):
        nfd = unicodedata.normalize("NFD", "Rélease")
        common = self._tree(nfd, "ref: refs/heads/unborn\n")
        self.assertIn("symbolic", host_fence.loose_symref_refusal(
            common, [unicodedata.normalize("NFC", "Rélease")]))

    def test_unit_ordinary_and_unrelated_files_pass(self):
        common = self._tree("MaIn", "0123456789abcdef0123456789abcdef01234567\n")
        self.assertIsNone(host_fence.loose_symref_refusal(common, ["main", "master"]))
        other = self._tree("other", "ref: refs/heads/unborn\n")
        self.assertIsNone(host_fence.loose_symref_refusal(other, ["main", "master"]))

    def test_real_repository_both_lanes(self):
        repo = self.repo(default="trunk", pack=True)
        with open(os.path.join(repo, ".git/refs/heads/MaIn"), "w") as handle:
            handle.write("ref: refs/heads/unborn\n")
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "fence_path_rejected", env)
        self.assertIn("symbolic", env["detail"])
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertIn("symbolic", caught.exception.envelope["detail"])
        snap, refusal = agent_lane._snapshot_protected_refs(repo, True)
        self.assertIsNone(snap)
        self.assertIn("symbolic", refusal["detail"])

    def test_read_only_is_not_refused(self):
        repo = self.repo(default="trunk", pack=True)
        with open(os.path.join(repo, ".git/refs/heads/MaIn"), "w") as handle:
            handle.write("ref: refs/heads/unborn\n")
        self.assertIsNone(agent_lane._workdir_fence(repo, False, True))

    def test_case_sensitive_volume(self):
        probe = os.path.join(self.root, "probe-a")
        os.makedirs(probe)
        if os.path.exists(os.path.join(self.root, "PROBE-A")):
            self.skipTest("volume is case-insensitive")
        repo = self.repo(default="trunk")
        with open(os.path.join(repo, ".git/refs/heads/MaIn"), "w") as handle:
            handle.write("ref: refs/heads/unborn\n")
        self.assertIn("symbolic", agent_lane._workdir_fence(repo, False)["detail"])


class UnlistableRecordDir(_Base):
    def _hide_listing(self):
        real = os.listdir
        target = host_fence.protected_record_dir()

        def fake(path="."):
            if os.fspath(path) == target:
                raise PermissionError(13, "Permission denied")
            return real(path)

        return mock.patch.object(host_fence.os, "listdir", fake)

    def _assert_refused_without_writing(self, repo):
        os.makedirs(os.path.dirname(self.ledger()), exist_ok=True)
        with self._hide_listing(), \
                mock.patch.object(host_fence, "_enroll", side_effect=AssertionError("enrolled")), \
                mock.patch.object(host_fence, "_write_record",
                                  side_effect=AssertionError("recorded")):
            env = agent_lane._workdir_fence(repo, False)
            self.assertIsNotNone(env)
            self.assertEqual(env["reason"], "fence_path_rejected", env)
            self.assertIn("delete", env["detail"])
            with self.assertRaises(host_fence.HostFenceError):
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertEqual(self.records(), [])

    def test_an_unlistable_record_dir_refuses_instead_of_enrolling(self):
        self._assert_refused_without_writing(self.repo(default="trunk"))

    def test_the_override_does_not_change_that(self):
        repo = self.repo(default="trunk")
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "release"}):
            self._assert_refused_without_writing(repo)

    def test_a_missing_record_dir_is_still_a_first_sight(self):
        repo = self.repo(default="trunk")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))


class MaskedGoogleKeyShape(_Base):
    """A branch named like a Google key: the mask is case-sensitive, so no message
    may carry the lowercased (folded) spelling, only the original one, masked."""

    KEY = "AIza" + "abcdefghij0123456789klmnopqrstuvwxy"[:35]

    def _assert_clean(self, text):
        self.assertNotIn(self.KEY, text)
        self.assertNotIn(self.KEY.lower(), text)

    def _repo(self):
        repo = self.repo(default="trunk")
        _git(repo, "update-ref", "refs/heads/" + self.KEY, self.seed)
        return repo

    def test_the_key_shape_is_a_valid_branch_name_the_masker_knows_in_its_own_case(self):
        self.assertEqual(len(self.KEY), 39)
        self.assertNotIn(self.KEY, host_fence.redact_secrets("x " + self.KEY + " y"))

    @unittest.skipUnless(DARWIN, "host-exec only reaches the repository roots on macOS")
    def test_host_exec_allow_repo_write_refusal_carries_neither_spelling(self):
        repo = self._repo()
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/" + self.KEY)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.KEY}), \
                contextlib.redirect_stderr(err), mock.patch.object(
                host_fence.os, "execvpe", side_effect=AssertionError("exec reached")):
            rc = host_fence.main(["--provider", "claude", "--workdir", repo,
                                  "--allow-repo-write", "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self.assertIn("protected branch", err.getvalue())
        self._assert_clean(err.getvalue())

    def test_agent_envelopes_carry_neither_spelling(self):
        repo = self._repo()
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.KEY}):
            changed = self.run_child(
                repo, "printf '%s\\n' " + self.other + " > \"$R/.git/refs/heads/" + self.KEY + "\"",
                foreman=True)
            self.assertEqual(changed.get("reason"), "protected_ref_changed", changed)
            self._assert_clean(json.dumps(changed))
        _git(repo, "update-ref", "refs/heads/" + self.KEY, self.seed)
        _git(repo, "symbolic-ref", "refs/heads/" + self.KEY, "refs/heads/work")
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: self.KEY}):
            env = self.run_child(repo, ": > \"$R/ran\"", foreman=True)
            self.assertEqual(env.get("reason"), "fence_path_rejected", env)
            self._assert_clean(json.dumps(env))
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
            self._assert_clean(json.dumps(caught.exception.envelope))

    def test_a_lowercase_probe_hit_names_the_recorded_spelling(self):
        # a symref found through the lowercase probe is reported under the name
        # as recorded, never under the folded one
        def run(*args):
            if args[0] == "for-each-ref":
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            ref = args[-1]
            if ref == "refs/heads/" + self.KEY.lower():
                return subprocess.CompletedProcess(args, 0, stdout="refs/heads/x\n", stderr="")
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

        why = host_fence.protected_symref_refusal(run, [self.KEY])
        self.assertTrue(why and "symbolic" in why, why)
        self._assert_clean(why)


if __name__ == "__main__":
    unittest.main()
