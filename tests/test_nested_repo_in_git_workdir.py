#!/usr/bin/env python3
"""Write mode refuses a git workdir that holds another repository (#199).

Real Git in temp repositories and the real macOS kernel fence. The refusal
cases call ``host_fence.prepare_host_fence(..., provider="claude",
workdir=<outer repo>)`` and expect ``fence_path_rejected`` before any child
launches. The controls (the outer repository alone, a linked worktree of it
under the workdir, a workdir that is itself a linked worktree) must still be
admitted, and the fenced child of the first two still cannot move the outer
``main``.

Before the fix the nested-repo cases were live probes: a fenced child moved
``main`` of an untracked clone, an ignored clone and a submodule.
"""
from __future__ import annotations

import contextlib
import errno
import io
import os
import platform
import shutil
import subprocess
import unittest
from unittest import mock

import omni_cli
from providers import host_fence, known_secrets
from tests.test_protected_refs import _IDENTITY, _Base, _git, _sh_quote

DARWIN = platform.system() == "Darwin"
_NO_USER_CONFIG = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


class _NestedBase(_Base):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, _NO_USER_CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fenced(self, repo, script):
        argv, env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", script], provider="claude", workdir=repo,
            env=dict(os.environ))
        return subprocess.run(argv, cwd=repo, env=env, capture_output=True,
                              text=True, check=False)

    def _two_commit_source(self, name):
        """A repository on ``main`` with two commits; returns path, old, new."""
        src = os.path.join(self.root, name)
        os.makedirs(src)
        _git(src, "init", "-q", "-b", "main")
        _git(src, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "one")
        old = _git(src, "rev-parse", "HEAD")
        _git(src, *_IDENTITY, "commit", "-q", "--allow-empty", "-m", "two")
        return src, old, _git(src, "rev-parse", "HEAD")

    def _prepare(self, repo):
        return host_fence.prepare_host_fence(
            ["/bin/sh", "-c", "true"], provider="claude", workdir=repo,
            env=dict(os.environ))

    def assertRefused(self, repo, *needles):
        with self.assertRaises(host_fence.HostFenceError) as ctx:
            self._prepare(repo)
        env = ctx.exception.envelope
        self.assertEqual(env["reason"], "fence_path_rejected")
        for needle in needles:
            self.assertIn(needle, env["detail"])
        return env

    def _clone(self, repo, rel):
        src, _old, _new = self._two_commit_source("src")
        os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
        # file:// so objects are copied, not hard-linked: the fence already
        # refuses a work tree holding pre-existing hard-linked files, which a
        # plain local-path clone would trip before the nested scan ran.
        subprocess.run(["git", "clone", "-q", "file://" + src,
                        os.path.join(repo, rel)], check=True)
        return os.path.join(repo, rel)

    def _common(self, repo):
        return os.path.realpath(os.path.join(
            repo, _git(repo, "rev-parse", "--git-common-dir")))

    def _add_submodule(self, repo):
        src, _old, _new = self._two_commit_source("subsrc")
        bare = os.path.join(self.root, "sub.git")
        subprocess.run(["git", "clone", "-q", "--bare", "file://" + src, bare],
                       check=True)
        _git(repo, "-c", "protocol.file.allow=always", *_IDENTITY,
             "submodule", "add", "-q", "file://" + bare, "sub")
        _git(repo, *_IDENTITY, "commit", "-q", "-m", "add sub")
        self.assertTrue(os.path.isdir(
            os.path.join(self._common(repo), "modules", "sub")))


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class NestedRepoInGitWorkdir(_NestedBase):

    def test_untracked_clone_is_refused(self):
        """#199 shape 1: an untracked clone at <repo>/vendor/lib."""
        repo = self.repo()
        self._clone(repo, os.path.join("vendor", "lib"))
        self.assertRefused(repo, "another Git repository",
                           os.path.join("vendor", "lib", ".git"))

    def test_ignored_clone_is_refused(self):
        """#199 shape 2: the same clone, listed in the outer .gitignore."""
        repo = self.repo()
        self._clone(repo, os.path.join("vendor", "lib"))
        with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as fh:
            fh.write("vendor/lib/\n")
        _git(repo, "check-ignore", "-q", "vendor/lib")
        self.assertRefused(repo, "another Git repository")

    def test_submodule_is_refused(self):
        """#199 shape 3: an initialized submodule at <repo>/sub.

        Its git dir is <common_dir>/modules/sub; its work tree holds a
        ``.git`` pointer file that is not a linked worktree.
        """
        repo = self.repo()
        self._add_submodule(repo)
        self.assertRefused(repo, "another Git repository",
                           os.path.join("sub", ".git"))

    def test_retained_submodule_git_dir_is_refused(self):
        """The submodule work tree is gone, its git dir under modules/ stays."""
        repo = self.repo()
        self._add_submodule(repo)
        shutil.rmtree(os.path.join(repo, "sub"))
        self.assertTrue(os.listdir(os.path.join(self._common(repo), "modules")))
        self.assertRefused(repo, "submodule Git directories")

    def test_nested_bare_repository_is_refused(self):
        repo = self.repo()
        os.makedirs(os.path.join(repo, "vendor"))
        subprocess.run(["git", "init", "-q", "--bare",
                        os.path.join(repo, "vendor", "bare.git")], check=True)
        self.assertRefused(repo, "bare repository",
                           os.path.join("vendor", "bare.git"))

    def test_forged_worktree_pointer_is_refused(self):
        """A .git file naming a real worktrees/<id> dir whose backlink does
        not point back at it is not a linked worktree."""
        repo = self.repo()
        wt = os.path.join(repo, ".claude", "worktrees", "x")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        meta = os.path.join(self._common(repo), "worktrees", "x")
        forged = os.path.join(repo, "forged")
        os.makedirs(forged)
        with open(os.path.join(forged, ".git"), "w", encoding="utf-8") as fh:
            fh.write("gitdir: %s\n" % meta)
        self.assertRefused(repo, "another Git repository",
                           os.path.join("forged", ".git"))

    def test_pointer_to_missing_worktree_dir_is_refused(self):
        repo = self.repo()
        forged = os.path.join(repo, "forged")
        os.makedirs(forged)
        with open(os.path.join(forged, ".git"), "w", encoding="utf-8") as fh:
            fh.write("gitdir: %s\n" % os.path.join(
                self._common(repo), "worktrees", "nope"))
        self.assertRefused(repo, "another Git repository")

    def test_pointer_into_another_repository_is_refused(self):
        """A linked worktree of a DIFFERENT repository is not this one's."""
        repo = self.repo()
        other = self.repo(name="other")
        wt = os.path.join(repo, "foreign")
        _git(other, "worktree", "add", "-q", wt, "-b", "side")
        self.assertRefused(repo, "another Git repository",
                           "foreign")

    def test_independent_repo_inside_linked_worktree_is_refused(self):
        repo = self.repo()
        wt = os.path.join(repo, ".claude", "worktrees", "x")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        self._clone(wt, os.path.join("vendor", "lib"))
        self.assertRefused(repo, os.path.join(
            ".claude", "worktrees", "x", "vendor", "lib", ".git"))

    def test_outer_repo_alone_is_admitted(self):
        self._prepare(self.repo())

    def test_linked_worktree_under_workdir_is_admitted(self):
        repo = self.repo()
        _git(repo, "worktree", "add", "-q",
             os.path.join(repo, ".claude", "worktrees", "x"), "-b", "side")
        self._prepare(repo)

    def test_workdir_that_is_a_linked_worktree_is_admitted(self):
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        self._prepare(wt)

    def test_workdir_linked_worktree_with_sibling_worktree_inside_is_admitted(self):
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        _git(repo, "worktree", "add", "-q",
             os.path.join(wt, "inner"), "-b", "side2")
        self._prepare(wt)

    def test_nested_clone_in_linked_worktree_workdir_is_refused(self):
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        self._clone(wt, "lib")
        self.assertRefused(wt, "another Git repository", "lib")

    def test_read_only_run_is_not_scanned(self):
        repo = self.repo()
        self._clone(repo, "lib")
        argv, _env, _meta = host_fence.prepare_host_fence(
            ["/bin/sh", "-c", "true"], provider="claude", workdir=repo,
            env=dict(os.environ), read_only=True)
        self.assertTrue(argv)

    # -- fix round 1: failed modules lookup, relative links, credentials ----

    def test_failed_modules_lookup_fails_closed(self):
        """An EIO on the modules/ lookup is not "modules absent"."""
        repo = self.repo()
        self._add_submodule(repo)
        shutil.rmtree(os.path.join(repo, "sub"))
        modules = os.path.join(self._common(repo), "modules")
        self.assertTrue(os.listdir(modules))
        real_lstat, calls = os.lstat, []

        def flaky(path, *args, **kwargs):
            if os.fspath(path) == modules and not calls:
                calls.append(path)
                raise OSError(errno.EIO, "injected", os.fspath(path))
            return real_lstat(path, *args, **kwargs)

        with mock.patch.object(host_fence.os, "lstat", side_effect=flaky):
            env = self.assertRefused(repo, "could not verify", "fails closed")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("holds", env["detail"])

    def _relative_worktree(self, repo, rel=os.path.join(".claude", "worktrees", "x"),
                           branch="side"):
        wt = os.path.join(repo, rel)
        _git(repo, "-c", "worktree.useRelativePaths=true", "worktree", "add",
             "-q", wt, "-b", branch)
        with open(os.path.join(wt, ".git"), encoding="utf-8") as fh:
            pointer = fh.read().strip()
        self.assertFalse(os.path.isabs(pointer[len("gitdir: "):]), pointer)
        meta = os.path.join(self._common(repo), "worktrees", os.path.basename(wt))
        with open(os.path.join(meta, "gitdir"), encoding="utf-8") as fh:
            self.assertFalse(os.path.isabs(fh.read().strip()))
        return wt, meta

    def _write_pointer(self, repo, rel, target):
        forged = os.path.join(repo, rel)
        os.makedirs(forged)
        with open(os.path.join(forged, ".git"), "w", encoding="utf-8") as fh:
            fh.write("gitdir: %s\n" % target)
        return forged

    def test_relative_link_linked_worktree_under_workdir_is_admitted(self):
        repo = self.repo()
        self._relative_worktree(repo)
        self._prepare(repo)

    def test_relative_pointer_outside_direct_child_of_worktrees_is_refused(self):
        repo = self.repo()
        _wt, meta = self._relative_worktree(repo)
        os.makedirs(os.path.join(meta, "inner"))
        self._write_pointer(repo, "forged", "../.git/worktrees/x/inner")
        self.assertRefused(repo, "another Git repository",
                           os.path.join("forged", ".git"))

    def test_relative_pointer_with_backlink_elsewhere_is_refused(self):
        """A second pointer at the real worktree metadata: the backlink names
        the first worktree's .git file, not this one."""
        repo = self.repo()
        self._relative_worktree(repo)
        self._write_pointer(repo, "forged", "../.git/worktrees/x")
        self.assertRefused(repo, "another Git repository",
                           os.path.join("forged", ".git"))

    def test_relative_link_worktree_of_another_repository_is_refused(self):
        repo = self.repo()
        other = self.repo(name="other")
        wt = os.path.join(repo, "foreign")
        _git(other, "-c", "worktree.useRelativePaths=true", "worktree", "add",
             "-q", wt, "-b", "side")
        self.assertRefused(repo, "another Git repository", "foreign")

    def test_host_exec_refusal_masks_an_unregistered_credential(self):
        """Through ``omni_cli.main``: the credential is primed before the
        host-exec diagnostic, so a nested directory named with it is masked."""
        secret = "zQ7vK2mXp9LrT4wBn6"
        repo = self.repo()
        os.makedirs(os.path.join(repo, secret, ".git"))
        known_secrets.clear()
        self.addCleanup(known_secrets.clear)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"CURSOR_API_KEY": secret}), \
                mock.patch.object(known_secrets, "register_auth_files"), \
                mock.patch.object(host_fence.os, "execvpe",
                                  side_effect=AssertionError("exec reached")), \
                contextlib.redirect_stderr(err):
            rc = omni_cli.main(["host-exec", "--provider", "claude",
                                "--workdir", repo, "--", "/usr/bin/true"])
        self.assertEqual(rc, 126)
        self.assertIn("another Git repository", err.getvalue())
        self.assertNotIn(secret, err.getvalue())
        self.assertNotIn(secret[:10], err.getvalue())

    # -- fix round 2: submodule git dirs under the worktree's own git dir ----

    def _linked_workdir_with_submodule(self, deinit):
        """A linked worktree on a feature branch with a submodule added there.

        Returns the worktree and its own ``modules`` directory, asserting the
        submodule git dir really lives under ``<git_dir>/modules`` and not
        under ``<common_dir>/modules``.
        """
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "feature")
        self._add_submodule_in(wt)
        git_dir = os.path.realpath(_git(wt, "rev-parse", "--absolute-git-dir"))
        modules = os.path.realpath(os.path.join(
            wt, _git(wt, "rev-parse", "--git-path", "modules")))
        self.assertEqual(modules, os.path.join(git_dir, "modules"))
        self.assertNotEqual(git_dir, self._common(wt))
        self.assertTrue(os.path.isdir(os.path.join(modules, "sub")))
        self.assertFalse(os.path.exists(
            os.path.join(self._common(wt), "modules")))
        if deinit:
            _git(wt, "submodule", "deinit", "-q", "-f", "sub")
            self.assertFalse(os.path.lexists(os.path.join(wt, "sub", ".git")))
            self.assertTrue(os.path.isdir(os.path.join(modules, "sub")))
        return wt, modules

    def _add_submodule_in(self, workdir):
        src, _old, _new = self._two_commit_source("subsrc")
        bare = os.path.join(self.root, "sub.git")
        subprocess.run(["git", "clone", "-q", "--bare", "file://" + src, bare],
                       check=True)
        _git(workdir, "-c", "protocol.file.allow=always", *_IDENTITY,
             "submodule", "add", "-q", "file://" + bare, "sub")
        _git(workdir, *_IDENTITY, "commit", "-q", "-m", "add sub")

    def test_deinitialized_submodule_in_linked_worktree_is_refused(self):
        wt, _modules = self._linked_workdir_with_submodule(deinit=True)
        self.assertRefused(wt, "submodule Git directories",
                           "modules/ of the worktree Git directory")

    def test_initialized_submodule_in_linked_worktree_is_refused(self):
        wt, _modules = self._linked_workdir_with_submodule(deinit=False)
        self.assertRefused(wt, "another Git repository",
                           os.path.join("sub", ".git"))

    def test_empty_worktree_modules_dir_is_admitted(self):
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "feature")
        git_dir = os.path.realpath(_git(wt, "rev-parse", "--absolute-git-dir"))
        os.makedirs(os.path.join(git_dir, "modules"))
        self._prepare(wt)

    def test_failed_worktree_modules_lookup_fails_closed(self):
        repo = self.repo()
        wt = os.path.join(self.root, "wt")
        _git(repo, "worktree", "add", "-q", wt, "-b", "feature")
        git_dir = os.path.realpath(_git(wt, "rev-parse", "--absolute-git-dir"))
        modules = os.path.join(git_dir, "modules")
        os.makedirs(modules)
        real_lstat, calls = os.lstat, []

        def flaky(path, *args, **kwargs):
            if os.fspath(path) == modules and not calls:
                calls.append(path)
                raise OSError(errno.EIO, "injected", os.fspath(path))
            return real_lstat(path, *args, **kwargs)

        with mock.patch.object(host_fence.os, "lstat", side_effect=flaky):
            env = self.assertRefused(wt, "could not verify", "fails closed")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("holds", env["detail"])


@unittest.skipUnless(DARWIN, "needs the macOS host fence")
class OuterRepoControls(_NestedBase):

    def _assert_outer_main_holds(self, repo, scripts):
        for name, script in scripts:
            with self.subTest(probe=name):
                probe = self._fenced(repo, script)
                now = _git(repo, "rev-parse", "refs/heads/main")
                if now != self.seed:
                    _git(repo, "update-ref", "refs/heads/main", self.seed)
                self.assertEqual(
                    now, self.seed,
                    "%s moved the outer main (%s -> %s); rc=%s stderr=%r" % (
                        name, self.seed[:8], now[:8], probe.returncode,
                        probe.stderr.strip()))

    def test_control_outer_main_cannot_be_moved(self):
        """Control: the existing protection of the outer repo's own main."""
        repo = self.repo()
        self._assert_outer_main_holds(repo, (
            ("git update-ref", "git update-ref refs/heads/main %s" % self.other),
            ("direct ref rewrite", "printf '%%s\\n' %s > .git/refs/heads/main"
             % self.other),
        ))

    def test_control_linked_worktree_cannot_move_outer_main(self):
        """Control: a linked worktree of the outer repo under the workdir."""
        repo = self.repo()
        wt = os.path.join(repo, ".claude", "worktrees", "x")
        _git(repo, "worktree", "add", "-q", wt, "-b", "side")
        common = os.path.realpath(os.path.join(
            repo, _git(repo, "rev-parse", "--git-common-dir")))
        self._assert_outer_main_holds(repo, (
            ("git update-ref via worktree",
             "git -C .claude/worktrees/x update-ref refs/heads/main %s"
             % self.other),
            ("direct rewrite in common dir",
             "printf '%%s\\n' %s > %s" % (
                 self.other, _sh_quote(os.path.join(
                     common, "refs", "heads", "main")))),
        ))


if __name__ == "__main__":
    unittest.main()
