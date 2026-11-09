#!/usr/bin/env python3
"""pi / cursor state: symlink chains above the vendor root, hard-linked state.

Issue #149 follow-up. The vendor root (``~/.pi``, ``~/.cursor``) may itself be
a symlink (a stow-managed dotfile tree); that is fine while every link in the
chain sits where the child cannot write and the target is outside every write
allow of the spawn. A link the child CAN rewrite steers the late write allows
anywhere, so it refuses. A hard-linked state file is a writable alias that
survives across runs, so it refuses too, and the vendor root carries a
``file-link`` deny like the codex home does.

Hermetic: synthetic home in a temp dir, no vendor CLI, no network.
"""
from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from providers import agent_lane, host_fence, kernel_sandbox

_LEAVES = {
    "pi": (".pi", (("agent", "sessions"),), ("agent", "auth.json")),
    "cursor": (".cursor", (("projects",),), None),
}


def _isolated_write_roots(workdir, read_only, extra_write_paths=None):
    """Write allow-set without the host's temp roots.

    The synthetic home lives in a temp dir, which the real allow-set always
    covers; keep only the workdir and the caller's extras so "where the child
    can write" is exactly what each test builds.
    """
    out = []
    if workdir is not None and not read_only:
        out.append(os.path.realpath(str(workdir)))
    out.extend(os.fspath(p) for p in (extra_write_paths or ()))
    return tuple(out)


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



class _Base(unittest.TestCase):
    def setUp(self):
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="omni_pcl_home_"))
        self.wd = os.path.realpath(tempfile.mkdtemp(prefix="omni_pcl_wd_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.addCleanup(shutil.rmtree, self.wd, True)

    def _make_root(self, provider, where):
        """A real vendor tree at *where* with every state leaf present."""
        _dot, dirs, file_leaf = _LEAVES[provider]
        for parts in dirs:
            os.makedirs(os.path.join(where, *parts))
        if file_leaf:
            with open(os.path.join(where, *file_leaf), "w") as fh:
                fh.write("{}")
        return where

    def _policy(self, provider):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            return kernel_sandbox.agent_fence_policy(
                provider, workdir=self.wd, read_only=False)

    def _assert_refused(self, policy):
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)

    def _assert_spawns(self, policy):
        self.assertNotIn("ok", policy, policy)
        self.assertIn("wrap_kw", policy)


class VendorRootLinkChain(_Base):
    def test_middle_link_in_a_child_writable_dir_refuses(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                real = self._make_root(
                    provider, os.path.join(self.home, "real", dot))
                hop = os.path.join(self.wd, "hop")
                os.makedirs(hop)
                os.symlink(real, os.path.join(hop, "l2"))
                os.symlink(os.path.join(hop, "l2"),
                           os.path.join(self.home, dot))
                self._assert_refused(self._policy(provider))

    def test_relative_two_hop_chain_through_a_writable_dir_refuses(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                real = self._make_root(
                    provider, os.path.join(self.home, "real", dot))
                hop = os.path.join(self.wd, "hop")
                os.makedirs(hop)
                os.symlink(real, os.path.join(hop, "l2"))
                rel = os.path.relpath(os.path.join(hop, "l2"), self.home)
                os.symlink(rel, os.path.join(self.home, dot))
                self._assert_refused(self._policy(provider))

    def test_target_inside_a_write_allow_refuses(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                real = self._make_root(
                    provider, os.path.join(self.wd, "vendor", dot))
                os.symlink(real, os.path.join(self.home, dot))
                self._assert_refused(self._policy(provider))

    def test_spawn_decision_refuses_and_leaves_argv_unwrapped(self):
        real = self._make_root("pi", os.path.join(self.home, "real", ".pi"))
        hop = os.path.join(self.wd, "hop")
        os.makedirs(hop)
        os.symlink(real, os.path.join(hop, "l2"))
        os.symlink(os.path.join(hop, "l2"), os.path.join(self.home, ".pi"))
        argv = ["/fake/pi", "-p", "x"]
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            out_argv, _env, _fence, _detail, refusal = (
                agent_lane._apply_kernel_fence(
                    "pi", argv, {"PATH": "/usr/bin"}, self.wd, False, True))
        self.assertIs(out_argv, argv)
        self.assertIsNotNone(refusal)

    # -- negative controls: legitimate layouts keep spawning --------------

    def test_plain_home_spawns(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                self._make_root(provider, os.path.join(self.home, dot))
                self._assert_spawns(self._policy(provider))

    def test_root_linked_to_an_unwritable_dotfiles_dir_spawns(self):
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                real = self._make_root(
                    provider, os.path.join(self.home, "dotfiles", dot))
                os.symlink(real, os.path.join(self.home, dot))
                policy = self._policy(provider)
                self._assert_spawns(policy)
                leaf = os.path.join(os.path.realpath(real), *dirs[0])
                self.assertIn(("subpath", leaf),
                              policy["wrap_kw"]["extra_post_write"])

    def test_two_hop_chain_entirely_outside_write_allows_spawns(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                real = self._make_root(
                    provider, os.path.join(self.home, "dotfiles", dot))
                mid = os.path.join(self.home, "stow")
                os.makedirs(mid)
                os.symlink(real, os.path.join(mid, "l2"))
                os.symlink(os.path.join(mid, "l2"),
                           os.path.join(self.home, dot))
                self._assert_spawns(self._policy(provider))


class HardLinkedState(_Base):
    def test_pi_hard_linked_auth_json_refuses(self):
        root = self._make_root("pi", os.path.join(self.home, ".pi"))
        os.link(os.path.join(root, "agent", "auth.json"),
                os.path.join(self.home, "alias.json"))
        self._assert_refused(self._policy("pi"))

    def test_pi_hard_linked_session_file_refuses(self):
        root = self._make_root("pi", os.path.join(self.home, ".pi"))
        sess = os.path.join(root, "agent", "sessions", "s.jsonl")
        with open(sess, "w") as fh:
            fh.write("{}")
        os.link(sess, os.path.join(self.home, "alias.jsonl"))
        self._assert_refused(self._policy("pi"))

    def test_cursor_hard_linked_state_file_refuses(self):
        root = self._make_root("cursor", os.path.join(self.home, ".cursor"))
        state = os.path.join(root, "projects", "p", "state.json")
        os.makedirs(os.path.dirname(state))
        with open(state, "w") as fh:
            fh.write("{}")
        os.link(state, os.path.join(self.home, "alias.json"))
        self._assert_refused(self._policy("cursor"))

    def test_single_link_state_files_spawn(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                state = os.path.join(root, *_LEAVES[provider][1][0], "s.json")
                with open(state, "w") as fh:
                    fh.write("{}")
                self._assert_spawns(self._policy(provider))


class FileLinkDeny(_Base):
    def test_vendor_root_carries_a_file_link_deny(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                policy = self._policy(provider)
                self._assert_spawns(policy)
                wrap_kw = policy["wrap_kw"]
                self.assertIn(
                    os.path.realpath(root),
                    {path for _k, path in wrap_kw["extra_deny_link"]},
                )
                with mock.patch.object(kernel_sandbox, "protected_home",
                                       return_value=self.home), \
                     mock.patch.object(kernel_sandbox, "_write_allow_paths",
                                       _isolated_write_roots):
                    profile = kernel_sandbox.build_profile(
                        self.wd, False, **wrap_kw)
                self.assertIn("(deny file-link", profile)
                self.assertGreater(profile.index("(deny file-link"),
                                   profile.index("(allow default)"))


class ChainThroughTheVendorsOwnStateLeaf(_Base):
    """A link of the chain inside a leaf the child writes (post-deny write)."""

    def test_cursor_chain_through_projects_refuses(self):
        real = self._make_root("cursor", os.path.join(self.home, "real", ".cursor"))
        os.symlink(real, os.path.join(real, "projects", "hop"))
        os.symlink(os.path.join(real, "projects", "hop"),
                   os.path.join(self.home, ".cursor"))
        self._assert_refused(self._policy("cursor"))

    def test_pi_chain_through_sessions_refuses(self):
        real = self._make_root("pi", os.path.join(self.home, "real", ".pi"))
        os.symlink(real, os.path.join(real, "agent", "sessions", "hop"))
        os.symlink(os.path.join(real, "agent", "sessions", "hop"),
                   os.path.join(self.home, ".pi"))
        self._assert_refused(self._policy("pi"))


class RootBinding(_Base):
    def test_root_that_is_not_the_one_the_leaves_used_refuses(self):
        self._make_root("pi", os.path.join(self.home, ".pi"))
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            other = kernel_sandbox.pi_cursor_root_link_refusal(
                "pi", self.wd, False, (), (), os.path.join(self.home, "other"))
            gone = kernel_sandbox.pi_cursor_root_link_refusal(
                "cursor", self.wd, False, (), (),
                os.path.join(self.home, ".cursor"))
            same = kernel_sandbox.pi_cursor_root_link_refusal(
                "pi", self.wd, False, (), (), os.path.join(self.home, ".pi"))
        self.assertIn("fence path rejected", other or "")
        self.assertIn("fence path rejected", gone or "")
        self.assertIsNone(same)


@unittest.skipUnless(platform.system() == "Darwin", "host fence is Darwin-only")
class ComposedWriteRoots(_Base):
    """Repository roots are added by prepare_host_fence, after the policy."""

    def _vcs(self, cwd, *args):
        subprocess.run(
            ["git", "-C", cwd, *args], check=True, capture_output=True,
            text=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "Fence test",
                 "GIT_AUTHOR_EMAIL": "fence@example.invalid",
                 "GIT_COMMITTER_NAME": "Fence test",
                 "GIT_COMMITTER_EMAIL": "fence@example.invalid"})

    def _prepare(self, provider, workdir):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(
            kernel_sandbox, "protected_home", return_value=self.home))
        stack.enter_context(mock.patch.object(
            kernel_sandbox, "_write_allow_paths", _isolated_write_roots))
        stack.enter_context(mock.patch.object(
            kernel_sandbox, "verify_fence", return_value=None))
        stack.enter_context(mock.patch.object(
            kernel_sandbox, "wrap_argv",
            side_effect=lambda argv, *a, **kw: [
                "/usr/bin/sandbox-exec", "-p", "fixture", *argv]))
        return host_fence.prepare_host_fence(
            ["/fake/" + provider], provider=provider, workdir=workdir, env={})

    def _repo(self):
        """``(toplevel, subdir)`` of a linked worktree on a feature branch.

        A linked worktree is the shape the host fence admits for a workdir
        below its toplevel (its Git paths are absolute).
        """
        base = os.path.realpath(tempfile.mkdtemp(prefix="omni_pcl_repo_"))
        self.addCleanup(shutil.rmtree, base, True)
        main = os.path.join(base, "main")
        os.mkdir(main)
        self._vcs(main, "init", "-q")
        self._vcs(main, "commit", "--allow-empty", "-qm", "seed")
        linked = os.path.join(base, "linked")
        self._vcs(main, "worktree", "add", "-qb", "work", linked)
        linked = os.path.realpath(linked)
        sub = os.path.join(linked, "sub")
        os.mkdir(sub)
        return linked, sub

    def test_resolved_root_under_the_repo_toplevel_refuses(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                repo, sub = self._repo()
                real = self._make_root(
                    provider, os.path.join(repo, "dotfiles", dot))
                os.symlink(real, os.path.join(self.home, dot))
                # The policy alone (workdir = repo/sub) cannot see the repo.
                with mock.patch.object(kernel_sandbox, "protected_home",
                                       return_value=self.home), \
                     mock.patch.object(kernel_sandbox, "_write_allow_paths",
                                       _isolated_write_roots):
                    self._assert_spawns(kernel_sandbox.agent_fence_policy(
                        provider, workdir=sub, read_only=False))
                with self.assertRaises(host_fence.HostFenceError) as caught:
                    self._prepare(provider, sub)
                self.assertEqual(caught.exception.envelope["reason"],
                                 "fence_path_rejected")
                self.assertIn("resolves into a write root",
                              caught.exception.envelope["detail"])

    def test_dotfiles_root_outside_the_repo_still_spawns(self):
        for provider, (dot, _d, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                _repo, sub = self._repo()
                real = self._make_root(
                    provider, os.path.join(self.home, "dotfiles", dot))
                os.symlink(real, os.path.join(self.home, dot))
                argv, _env, _meta = self._prepare(provider, sub)
                self.assertEqual(argv[0], "/usr/bin/sandbox-exec")


class HardLinkScanFailsClosed(_Base):
    def test_unreadable_directory_refuses(self):
        if os.geteuid() == 0:
            self.skipTest("permission bits do not bind root")
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                hidden = os.path.join(root, *dirs[0], "hidden")
                os.makedirs(hidden)
                os.chmod(hidden, 0)
                self.addCleanup(os.chmod, hidden, 0o700)
                self._assert_refused(self._policy(provider))

    def test_tree_over_the_cap_refuses(self):
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                for i in range(5):
                    with open(os.path.join(root, *dirs[0], "f%d" % i), "w") as fh:
                        fh.write("{}")
                with mock.patch.object(kernel_sandbox, "_STATE_LINK_SCAN_CAP", 3):
                    policy = self._policy(provider)
                self._assert_refused(policy)
                self.assertIn("too large", policy["detail"])

    def test_tree_at_the_cap_spawns(self):
        root = self._make_root("pi", os.path.join(self.home, ".pi"))
        for i in range(3):
            with open(os.path.join(root, "agent", "sessions", "f%d" % i), "w") as fh:
                fh.write("{}")
        with mock.patch.object(kernel_sandbox, "_STATE_LINK_SCAN_CAP", 3):
            self._assert_spawns(self._policy("pi"))

    def test_stat_error_refuses_but_a_vanished_file_does_not(self):
        root = self._make_root("pi", os.path.join(self.home, ".pi"))
        leaf = os.path.join(root, "agent", "sessions")
        for name in ("bad.json", "gone.json"):
            with open(os.path.join(leaf, name), "w") as fh:
                fh.write("{}")
        real_lstat = os.lstat

        def fake(path, *a, **kw):
            if str(path).endswith("bad.json"):
                raise PermissionError(13, "Permission denied", str(path))
            if str(path).endswith("gone.json"):
                raise FileNotFoundError(2, "No such file", str(path))
            return real_lstat(path, *a, **kw)

        with mock.patch.object(kernel_sandbox.os, "lstat", fake):
            why = kernel_sandbox._hard_linked_state_refusal("pi", leaf)
        self.assertIn("cannot be read", why or "")
        os.unlink(os.path.join(leaf, "bad.json"))
        with mock.patch.object(kernel_sandbox.os, "lstat", fake):
            self.assertIsNone(kernel_sandbox._hard_linked_state_refusal("pi", leaf))


_DATA_VOLUME = "/System/Volumes/Data"


class FirmlinkSpelledChain(_Base):
    """A chain link spelled through the Data-volume alias of a write root."""

    def _alias(self, path):
        alias = _DATA_VOLUME + path
        try:
            same = os.path.samefile(alias, path)
        except OSError:
            same = False
        if not same:
            self.skipTest("%s does not alias %s on this host" % (alias, path))
        return alias

    def test_cursor_chain_link_spelled_through_the_data_volume_refuses(self):
        real = self._make_root("cursor", os.path.join(self.home, "real", ".cursor"))
        hop = os.path.join(real, "projects", "hop")
        os.symlink(real, hop)
        os.symlink(self._alias(real) + "/projects/hop",
                   os.path.join(self.home, ".cursor"))
        self._assert_refused(self._policy("cursor"))

    def test_pi_chain_link_spelled_through_the_data_volume_refuses(self):
        real = self._make_root("pi", os.path.join(self.home, "real", ".pi"))
        hop = os.path.join(real, "agent", "sessions", "hop")
        os.symlink(real, hop)
        os.symlink(self._alias(real) + "/agent/sessions/hop",
                   os.path.join(self.home, ".pi"))
        self._assert_refused(self._policy("pi"))


class RootCapturedOnce(_Base):
    def test_root_swapped_between_leaf_steps_never_yields_mixed_allowances(self):
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                a = self._make_root(
                    provider, os.path.join(self.home, "dotfiles", "a", dot))
                b = self._make_root(
                    provider, os.path.join(self.home, "dotfiles", "b", dot))
                link = os.path.join(self.home, dot)
                os.symlink(a, link)
                original = kernel_sandbox._claude_run_leaf
                calls = []

                def swapping(root, *parts, **kw):
                    out = original(root, *parts, **kw)
                    if not calls:
                        os.unlink(link)
                        os.symlink(b, link)
                    calls.append(parts)
                    return out

                with mock.patch.object(kernel_sandbox, "_claude_run_leaf",
                                       swapping):
                    policy = self._policy(provider)
                if policy.get("reason") == "fence_path_rejected":
                    continue
                kw = policy["wrap_kw"]
                under_a = [p for _k, p in kw["extra_post_write"]
                           if p.startswith(os.path.realpath(a))]
                self.assertEqual(len(under_a), len(kw["extra_post_write"]), kw)
                self.assertEqual(policy["vendor_root"], os.path.realpath(a))
                self.assertIn(os.path.realpath(a),
                              {p for _k, p in kw["extra_deny_link"]})


class HostFenceMissingVendorRoot(_Base):
    @unittest.skipUnless(platform.system() == "Darwin", "host fence is Darwin-only")
    def test_vendor_state_rows_without_a_vendor_root_refuse(self):
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                leaf = os.path.join(os.path.realpath(root), *dirs[0])
                policy = {"vendor_state_writable": True, "wrap_kw": {
                    "read_wall": True,
                    "extra_post_write": (("subpath", leaf),)}}
                with mock.patch.object(kernel_sandbox, "protected_home",
                                       return_value=self.home), \
                     mock.patch.object(kernel_sandbox, "verify_fence",
                                       return_value=None), \
                     mock.patch.object(
                         kernel_sandbox, "wrap_argv",
                         side_effect=lambda argv, *a, **kw: [
                             "/usr/bin/sandbox-exec", "-p", "fixture", *argv]):
                    with self.assertRaises(host_fence.HostFenceError) as caught:
                        host_fence.prepare_host_fence(
                            ["/fake/" + provider], provider=provider,
                            workdir=self.wd, env={}, policy=policy)
                self.assertEqual(caught.exception.envelope["reason"],
                                 "fence_path_rejected")
                self.assertIn("without its vendor root",
                              caught.exception.envelope["detail"])


class DirectoriesCountTowardTheCap(_Base):
    def test_directory_only_tree_over_the_cap_refuses(self):
        for provider, (dot, dirs, _f) in _LEAVES.items():
            with self.subTest(provider=provider):
                self.setUp()
                root = self._make_root(provider, os.path.join(self.home, dot))
                for i in range(5):
                    os.makedirs(os.path.join(root, *dirs[0], "d%d" % i))
                with mock.patch.object(kernel_sandbox, "_STATE_LINK_SCAN_CAP", 3):
                    policy = self._policy(provider)
                self._assert_refused(policy)
                self.assertIn("too large", policy["detail"])


if __name__ == "__main__":
    unittest.main()
