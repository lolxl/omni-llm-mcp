#!/usr/bin/env python3
"""Repo-write mode refuses a repository whose refs are not loose files (#164).

The protected-branch denies name loose ref files. A reftable repository keeps
its refs in ``<common dir>/reftable/`` tables that no deny reaches, so the
fence must read the ref storage format from outside the child and refuse
anything other than ``files`` -- including a format it cannot determine.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import unittest
from unittest import mock

from providers import git_safe, host_fence
from tests import offline_home

_IDENTITY = ["-c", "user.email=t@example.invalid", "-c", "user.name=t"]


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True)


def _init(root, name, ref_format):
    repo = os.path.join(root, name)
    subprocess.run(["git", "init", "-q", "--ref-format=" + ref_format, repo],
                   check=True)
    _git(repo, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "seed")
    # A work branch, so the protected-branch refusal is not what fires.
    _git(repo, "checkout", "-q", "-b", "work")
    return repo


def _ref_format(repo):
    return _git(repo, "rev-parse", "--show-ref-format").stdout.strip()


_STATE_PATCH = None
_STATE_DIR = None
_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module.
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



class _Base(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        tmp = tempfile.TemporaryDirectory(prefix="omni_refstore_", dir="/private/tmp")
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)


class RefStorageRefusal(_Base):
    def test_reftable_repo_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        self.assertEqual(_ref_format(repo), "reftable")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=True, allow_main=True)
        env = caught.exception.envelope
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("files ref storage", env["detail"])
        self.assertIn("reftable", env["detail"])

    def test_reftable_repo_is_refused_on_a_work_branch_too(self):
        repo = _init(self.root, "rt", "reftable")
        with self.assertRaises(host_fence.HostFenceError):
            host_fence._repo_write_roots(repo, required=True, allow_main=False)

    def test_reftable_linked_worktree_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        linked = os.path.join(self.root, "rt-linked")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        linked = os.path.realpath(linked)
        self.assertEqual(_ref_format(linked), "reftable")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(linked, required=True, allow_main=False)
        self.assertIn("files ref storage", caught.exception.envelope["detail"])

    def test_files_repo_is_unchanged(self):
        repo = _init(self.root, "files", "files")
        self.assertEqual(_ref_format(repo), "files")
        roots, denies, meta, _post, _final = host_fence._repo_write_roots(
            repo, required=True, allow_main=False)
        self.assertTrue(roots)
        self.assertTrue(denies)
        self.assertEqual(meta, (os.path.join(repo, ".git"),) * 2)

    def test_files_linked_worktree_is_unchanged(self):
        repo = _init(self.root, "files", "files")
        linked = os.path.join(self.root, "files-linked")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        roots = host_fence._repo_write_roots(
            os.path.realpath(linked), required=True, allow_main=False)[0]
        self.assertTrue(roots)


class OrdinaryWriteMode(_Base):
    """required=False (ordinary write mode): a non-files ref store refuses."""

    def test_reftable_plain_clone_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=False, allow_main=False)
        env = caught.exception.envelope
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("write mode requires files ref storage", env["detail"])
        self.assertNotIn("--allow-repo-write", env["detail"])

    def test_reftable_linked_worktree_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        linked = os.path.join(self.root, "rt-linked")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(
                os.path.realpath(linked), required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")
        self.assertIn("write mode requires files ref storage",
                      caught.exception.envelope["detail"])

    def test_undeterminable_format_plain_clone_is_refused(self):
        repo = _init(self.root, "files", "files")
        real = git_safe.run_git

        def fake(workdir, args, **kw):
            args = list(args)
            if "--show-ref-format" in args or "extensions.refStorage" in args:
                return subprocess.CompletedProcess(["git"], 128, stdout="", stderr="")
            return real(workdir, args, **kw)

        with mock.patch.object(git_safe, "run_git", side_effect=fake):
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(repo, required=False, allow_main=False)
        self.assertIn("found undetermined", caught.exception.envelope["detail"])

    def test_files_repo_is_unchanged(self):
        repo = _init(self.root, "files", "files")
        roots, denies, meta, _post, _final = host_fence._repo_write_roots(
            repo, required=False, allow_main=False)
        self.assertTrue(roots)
        self.assertTrue(denies)
        self.assertEqual(meta, (os.path.join(repo, ".git"),) * 2)

    def test_non_git_directory_is_unchanged(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.assertEqual(host_fence._repo_write_roots(
            plain, required=False, allow_main=False), ((), (), (), (), ()))


class DiscoveryFailure(_Base):
    """Ordinary write mode keeps the workdir writable: failed discovery is not
    "not a repository" when the workdir is, or sits in, a Git directory."""

    def _fail_discovery(self):
        real = git_safe.run_git

        def fake(workdir, args, **kw):
            if "--show-toplevel" in list(args):
                return subprocess.CompletedProcess(["git"], 128, stdout="", stderr="")
            return real(workdir, args, **kw)

        return mock.patch.object(git_safe, "run_git", side_effect=fake)

    def _refused(self, workdir, **kw):
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(workdir, required=False, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")

    def test_workdir_inside_a_real_repos_reftable_dir_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        self._refused(os.path.join(repo, ".git", "reftable"))

    def test_workdir_inside_a_real_repos_git_dir_is_refused(self):
        repo = _init(self.root, "files", "files")
        self._refused(os.path.join(repo, ".git", "refs", "heads"))

    def test_repo_root_with_failed_discovery_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        with self._fail_discovery():
            self._refused(repo)

    def test_gitdir_component_is_refused_in_any_case(self):
        for index, name in enumerate((".git", ".GIT", ".Git")):
            with self.subTest(name):
                workdir = os.path.join(self.root, "d%d" % index, name, "sub")
                os.makedirs(workdir)
                self._refused(os.path.realpath(workdir))

    def test_plain_non_git_directory_is_unchanged(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.assertEqual(host_fence._repo_write_roots(
            plain, required=False, allow_main=False), ((), (), (), (), ()))
        with self._fail_discovery():
            self.assertEqual(host_fence._repo_write_roots(
                plain, required=False, allow_main=False), ((), (), (), (), ()))

    def test_a_dotgit_lookalike_name_is_not_a_git_directory(self):
        other = os.path.join(self.root, ".gitx", "dotgit", "sub")
        os.makedirs(other)
        self.assertEqual(host_fence._repo_write_roots(
            os.path.realpath(other), required=False, allow_main=False),
            ((), (), (), (), ()))


class UndeterminableFormat(_Base):
    """Every way the format read can fail refuses; none falls through."""

    def _run_with_format_answer(self, repo, answer, config=None, required=True):
        """Answer the two format reads from the seam; every other call is real.

        ``config`` answers ``config --get extensions.refStorage`` (the fallback
        read); by default that read fails too, so only ``answer`` decides.
        """
        real = git_safe.run_git
        config = self._proc(128, "") if config is None else config

        def fake(workdir, args, **kw):
            args = list(args)
            if "--show-ref-format" in args:
                return answer
            if args[:1] == ["config"] and "extensions.refStorage" in args:
                return config
            return real(workdir, args, **kw)

        with mock.patch.object(git_safe, "run_git", side_effect=fake):
            return host_fence._repo_write_roots(repo, required=required, allow_main=False)

    def _proc(self, rc, out):
        return subprocess.CompletedProcess(["git"], rc, stdout=out, stderr="")

    # A Git that does not know --show-ref-format either fails the call or, before
    # 2.45, echoes the flag on stdout with exit 0 (rc 0, "--show-ref-format").
    # Either way the declared extension is then the answer.
    ECHO = (0, "--show-ref-format\n")

    def test_old_git_echoing_the_flag_with_refstorage_unset_behaves_as_files(self):
        repo = _init(self.root, "files", "files")
        for required in (True, False):
            with self.subTest(required=required):
                roots = self._run_with_format_answer(
                    repo, self._proc(*self.ECHO), config=self._proc(1, ""),
                    required=required)[0]
                self.assertTrue(roots)

    def test_old_git_echoing_the_flag_with_refstorage_reftable_is_refused(self):
        repo = _init(self.root, "files", "files")
        for required in (True, False):
            with self.subTest(required=required):
                with self.assertRaises(host_fence.HostFenceError) as caught:
                    self._run_with_format_answer(
                        repo, self._proc(*self.ECHO), config=self._proc(0, "reftable\n"),
                        required=required)
                self.assertIn("found reftable", caught.exception.envelope["detail"])

    def test_old_git_echoing_the_flag_with_failing_config_is_refused(self):
        repo = _init(self.root, "files", "files")
        with self.assertRaises(host_fence.HostFenceError):
            self._run_with_format_answer(
                repo, self._proc(*self.ECHO), config=self._proc(128, ""))

    def test_old_git_fallback_reads_the_real_extension_on_the_echo_shape(self):
        files = _init(self.root, "files", "files")
        reftable = _init(self.root, "rt", "reftable")
        real = git_safe.run_git

        def fake(workdir, args, **kw):
            if "--show-ref-format" in list(args):
                return self._proc(*self.ECHO)
            return real(workdir, args, **kw)

        with mock.patch.object(git_safe, "run_git", side_effect=fake):
            self.assertTrue(host_fence._repo_write_roots(
                files, required=False, allow_main=False)[0])
            with self.assertRaises(host_fence.HostFenceError):
                host_fence._repo_write_roots(reftable, required=False, allow_main=False)

    def test_old_git_with_refstorage_unset_behaves_as_files(self):
        repo = _init(self.root, "files", "files")
        roots, denies, _meta, _post, _final = self._run_with_format_answer(
            repo, self._proc(129, ""), config=self._proc(1, ""))
        self.assertTrue(roots)
        self.assertTrue(denies)

    def test_old_git_with_refstorage_unset_ordinary_write_is_unchanged(self):
        repo = _init(self.root, "files", "files")
        roots = self._run_with_format_answer(
            repo, self._proc(129, ""), config=self._proc(1, ""), required=False)[0]
        self.assertTrue(roots)

    def test_old_git_with_refstorage_files_behaves_as_files(self):
        repo = _init(self.root, "files", "files")
        roots = self._run_with_format_answer(
            repo, self._proc(129, ""), config=self._proc(0, "files\n"))[0]
        self.assertTrue(roots)

    def test_old_git_with_refstorage_reftable_is_refused(self):
        repo = _init(self.root, "files", "files")
        for required in (True, False):
            with self.subTest(required=required):
                with self.assertRaises(host_fence.HostFenceError) as caught:
                    self._run_with_format_answer(
                        repo, self._proc(129, ""), config=self._proc(0, "reftable\n"),
                        required=required)
                self.assertIn("found reftable", caught.exception.envelope["detail"])

    def test_old_git_with_failing_config_read_is_refused(self):
        repo = _init(self.root, "files", "files")
        for label, config in (("rc 128", self._proc(128, "")),
                              ("rc 2", self._proc(2, "")),
                              ("rc 1 with output", self._proc(1, "files\n")),
                              ("rc 0 empty", self._proc(0, "")),
                              ("rc 0 two lines", self._proc(0, "files\nfiles\n"))):
            for required in (True, False):
                with self.subTest(label, required=required):
                    with self.assertRaises(host_fence.HostFenceError) as caught:
                        self._run_with_format_answer(
                            repo, self._proc(129, ""), config=config, required=required)
                    self.assertIn("files ref storage", caught.exception.envelope["detail"])

    def _only_show_ref_format_fails(self):
        real = git_safe.run_git

        def fake(workdir, args, **kw):
            if "--show-ref-format" in list(args):
                return self._proc(129, "")
            return real(workdir, args, **kw)

        return mock.patch.object(git_safe, "run_git", side_effect=fake)

    def test_old_git_fallback_reads_the_real_extension(self):
        files = _init(self.root, "files", "files")
        reftable = _init(self.root, "rt", "reftable")
        with self._only_show_ref_format_fails():
            self.assertTrue(host_fence._repo_write_roots(
                files, required=True, allow_main=False)[0])
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(reftable, required=True, allow_main=False)
        self.assertIn("found reftable", caught.exception.envelope["detail"])

    def test_show_ref_format_answer_wins_over_the_config_fallback(self):
        repo = _init(self.root, "files", "files")
        with self.assertRaises(host_fence.HostFenceError):
            self._run_with_format_answer(
                repo, self._proc(0, "reftable\n"), config=self._proc(1, ""))

    def test_failure_empty_unknown_and_garbage_all_refuse(self):
        repo = _init(self.root, "files", "files")
        cases = {
            "nonzero exit": self._proc(128, ""),
            "nonzero exit with echoed option": self._proc(128, "--show-ref-format\n"),
            "empty output": self._proc(0, ""),
            "blank output": self._proc(0, "\n"),
            "unknown format": self._proc(0, "future\n"),
            "files with a suffix": self._proc(0, "files-extra\n"),
            "uppercase": self._proc(0, "FILES\n"),
            "garbage": self._proc(0, "x\x00y \u202e\n"),
            "two lines": self._proc(0, "files\nfiles\n"),
        }
        for label, answer in cases.items():
            with self.subTest(label):
                with self.assertRaises(host_fence.HostFenceError) as caught:
                    self._run_with_format_answer(repo, answer)
                self.assertEqual(caught.exception.envelope["reason"],
                                 "fence_path_rejected")
                self.assertIn("files ref storage", caught.exception.envelope["detail"])

    def test_refusal_text_never_echoes_unbounded_git_output(self):
        repo = _init(self.root, "files", "files")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            self._run_with_format_answer(repo, self._proc(0, "secret-token-123\n"))
        self.assertNotIn("secret-token-123", caught.exception.envelope["detail"])
        self.assertIn("undetermined", caught.exception.envelope["detail"])

    def test_git_unable_to_run_still_refuses_loudly(self):
        repo = _init(self.root, "files", "files")
        real = git_safe.run_git

        def fake(workdir, args, **kw):
            if "--show-ref-format" in list(args):
                raise git_safe.GitSafeError("cannot build a safe environment")
            return real(workdir, args, **kw)

        with mock.patch.object(git_safe, "run_git", side_effect=fake):
            with self.assertRaises(host_fence.HostFenceError) as caught:
                host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")


@unittest.skipUnless(platform.system() == "Darwin", "host fence is Darwin-only")
class ThroughPrepareHostFence(_Base):
    def test_repo_write_spawn_on_reftable_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence.prepare_host_fence(
                ["/usr/bin/true"], provider="claude", workdir=repo,
                allow_repo_write=True)
        self.assertIn("files ref storage", caught.exception.envelope["detail"])

    def test_read_only_spawn_on_reftable_is_unaffected(self):
        repo = _init(self.root, "rt", "reftable")
        argv, _env, _meta = host_fence.prepare_host_fence(
            ["/usr/bin/true"], provider="claude", workdir=repo, read_only=True)
        self.assertTrue(argv)

    def test_ordinary_write_spawn_on_reftable_plain_clone_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence.prepare_host_fence(
                ["/usr/bin/true"], provider="claude", workdir=repo)
        self.assertIn("write mode requires files ref storage",
                      caught.exception.envelope["detail"])

    def test_ordinary_write_spawn_in_reftable_linked_worktree_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        linked = os.path.join(self.root, "rt-linked")
        _git(repo, "worktree", "add", "-q", "-b", "feature", linked)
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence.prepare_host_fence(
                ["/usr/bin/true"], provider="claude",
                workdir=os.path.realpath(linked))
        self.assertIn("write mode requires files ref storage",
                      caught.exception.envelope["detail"])

    def test_ordinary_write_spawn_with_workdir_inside_reftable_dir_is_refused(self):
        repo = _init(self.root, "rt", "reftable")
        with self.assertRaises(host_fence.HostFenceError):
            host_fence.prepare_host_fence(
                ["/usr/bin/true"], provider="claude",
                workdir=os.path.join(repo, ".git", "reftable"))

    def test_ordinary_write_spawn_on_files_repo_still_writes_git_metadata(self):
        repo = _init(self.root, "files", "files")
        probe = os.path.join(repo, ".git", "probe")
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/usr/bin/touch", probe], provider="claude", workdir=repo)
        result = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(probe))


if __name__ == "__main__":
    unittest.main()
