#!/usr/bin/env python3
"""claude DEFAULT home reached through a symlink chain the run can write.

Issue #196. With no CLAUDE_CONFIG_DIR, ``~/.claude`` may be a link (a
stow-managed dotfile tree) while every link of the chain sits where the child
cannot write and the target is outside every write allow of the spawn. A link
the run CAN rewrite steers the late write allows (spelled at the RESOLVED
root) anywhere, so it refuses, the same rule ``~/.pi`` / ``~/.cursor`` /
agy already get. A named CLAUDE_CONFIG_DIR home is not judged by this rule.

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

from providers import host_fence, kernel_sandbox


def _isolated_write_roots(workdir, read_only, extra_write_paths=None):
    """Write allow-set without the host's temp roots (workdir plus extras)."""
    out = []
    if workdir is not None and not read_only:
        out.append(os.path.realpath(str(workdir)))
    out.extend(os.fspath(p) for p in (extra_write_paths or ()))
    return tuple(out)


_STATE_PATCH = None
_STATE_DIR = None


def setUpModule():
    # A write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    _STATE_PATCH.stop()
    shutil.rmtree(_STATE_DIR, ignore_errors=True)


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="omni_cl_home_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.wd = os.path.join(self.home, "proj")      # writable workdir inside HOME
        os.makedirs(self.wd)

    def _root(self, where):
        """A real claude tree at *where* with every state leaf present."""
        os.makedirs(where, exist_ok=True)
        for d in ("projects", "todos", "debug", "shell-snapshots",
                  "sessions", "log"):
            os.makedirs(os.path.join(where, d), exist_ok=True)
        with open(os.path.join(where, "settings.json"), "w") as fh:
            fh.write("{}")
        return where

    def _policy(self, provider="claude", **kw):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            return kernel_sandbox.agent_fence_policy(
                provider, workdir=self.wd, read_only=False, **kw)

    def _assert_refused(self, policy):
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)

    def _assert_spawns(self, policy):
        self.assertNotIn("ok", policy, policy)
        self.assertIn("wrap_kw", policy)

    # Layouts of the probe: HOME/.claude reached through ...
    def _l1(self):      # HOME/.claude -> wd/cfg -> HOME/elsewhere
        real = self._root(os.path.join(self.home, "elsewhere"))
        os.symlink(real, os.path.join(self.wd, "cfg"))
        os.symlink(os.path.join(self.wd, "cfg"),
                   os.path.join(self.home, ".claude"))

    def _l2(self):      # HOME/.claude -> wd/hop/l2 -> HOME/real/x
        real = self._root(os.path.join(self.home, "real", "x"))
        hop = os.path.join(self.wd, "hop")
        os.makedirs(hop)
        os.symlink(real, os.path.join(hop, "l2"))
        os.symlink(os.path.join(hop, "l2"), os.path.join(self.home, ".claude"))


class DefaultHomeChain(_Base):
    def test_l1_link_in_the_workdir_refuses(self):
        self._l1()
        self._assert_refused(self._policy())

    def test_l2_middle_link_in_a_writable_dir_refuses(self):
        self._l2()
        self._assert_refused(self._policy())

    def test_repointed_link_refuses(self):
        self._l1()
        victim = os.path.join(self.home, "victim")
        os.makedirs(victim)
        link = os.path.join(self.wd, "cfg")
        os.unlink(link)
        os.symlink(victim, link)        # what the child could do between runs
        self._assert_refused(self._policy())

    def test_l3_link_to_an_unwritable_dotfiles_dir_spawns(self):
        real = self._root(os.path.join(self.home, "dotfiles", ".claude"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        self._assert_spawns(self._policy())

    def test_l4_plain_dir_spawns(self):
        self._root(os.path.join(self.home, ".claude"))
        self._assert_spawns(self._policy())

    def test_resolved_root_inside_a_write_root_refuses(self):
        real = self._root(os.path.join(self.wd, "realcfg"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        self._assert_refused(self._policy())

    def test_resolved_root_containing_a_write_root_refuses(self):
        # The workdir sits inside the dir HOME/.claude resolves to.
        real = self._root(os.path.join(self.home, "dotfiles", ".claude"))
        wd = os.path.join(real, "work")
        os.makedirs(wd)
        os.symlink(real, os.path.join(self.home, ".claude"))
        self.wd = wd
        self._assert_refused(self._policy())

    def test_policy_records_the_resolved_root_for_the_host_fence(self):
        real = self._root(os.path.join(self.home, "dotfiles", ".claude"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        self.assertEqual(self._policy().get("vendor_root"), real)

    def test_missing_default_home_spawns(self):
        # Nothing to judge: no root, no state rows.
        self._assert_spawns(self._policy())

    def test_codex_default_home_rule_is_unchanged(self):
        # codex keeps its own, stricter rule: any symlinked ~/.codex refuses,
        # and it never carries a vendor root.
        real = os.path.join(self.home, "dotfiles", ".codex")
        os.makedirs(os.path.join(real, "sessions"))
        with open(os.path.join(real, "config.toml"), "w") as fh:
            fh.write("")
        os.symlink(real, os.path.join(self.home, ".codex"))
        policy = self._policy("codex")
        self.assertIsNone(policy.get("vendor_root"))
        self._assert_refused(policy)
        # A plain ~/.codex still spawns.
        os.unlink(os.path.join(self.home, ".codex"))
        shutil.copytree(real, os.path.join(self.home, ".codex"))
        policy = self._policy("codex")
        self.assertIsNone(policy.get("vendor_root"))
        self._assert_spawns(policy)


class DirectChainCheck(_Base):
    def test_composed_write_set_with_a_link_inside_refuses(self):
        extra = os.path.join(self.home, "extra")
        os.makedirs(extra)
        real = self._root(os.path.join(self.home, "elsewhere"))
        os.symlink(real, os.path.join(extra, "cfg"))
        os.symlink(os.path.join(extra, "cfg"),
                   os.path.join(self.home, ".claude"))
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            before = kernel_sandbox.pi_cursor_root_link_refusal(
                "claude", self.wd, False, (), (), real)
            after = kernel_sandbox.pi_cursor_root_link_refusal(
                "claude", self.wd, False, (extra,), (), real)
        self.assertIsNone(before)
        self.assertIn("fence path rejected", after or "")
        self.assertIn("symlink the run can write", after)


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

    def _prepare(self, workdir, env=None, policy=None, config_dir=None):
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
            ["/fake/claude"], provider="claude", workdir=workdir,
            env=env if env is not None else {}, policy=policy,
            config_dir=config_dir)

    def _repo(self):
        """``(toplevel, subdir)`` of a linked worktree on a feature branch."""
        base = os.path.realpath(tempfile.mkdtemp(prefix="omni_cl_repo_"))
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

    def _policy_for(self, workdir):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            return kernel_sandbox.agent_fence_policy(
                "claude", workdir=workdir, read_only=False)

    def test_resolved_root_under_the_repo_toplevel_refuses(self):
        repo, sub = self._repo()
        real = self._root(os.path.join(repo, "dotfiles", ".claude"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        # The policy alone (workdir = repo/sub) cannot see the repo.
        self._assert_spawns(self._policy_for(sub))
        with self.assertRaises(host_fence.HostFenceError) as caught:
            self._prepare(sub)
        self.assertEqual(caught.exception.envelope["reason"],
                         "fence_path_rejected")
        self.assertIn("resolves into a write root",
                      caught.exception.envelope["detail"])

    def test_link_inside_the_repo_toplevel_refuses(self):
        repo, sub = self._repo()
        real = self._root(os.path.join(self.home, "elsewhere"))
        os.symlink(real, os.path.join(repo, "cfg"))
        os.symlink(os.path.join(repo, "cfg"), os.path.join(self.home, ".claude"))
        self._assert_spawns(self._policy_for(sub))
        with self.assertRaises(host_fence.HostFenceError) as caught:
            self._prepare(sub)
        self.assertEqual(caught.exception.envelope["reason"],
                         "fence_path_rejected")
        self.assertIn("symlink the run can write",
                      caught.exception.envelope["detail"])

    def test_dotfiles_root_outside_the_repo_still_spawns(self):
        _repo, sub = self._repo()
        real = self._root(os.path.join(self.home, "dotfiles", ".claude"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        argv, _env, _meta = self._prepare(sub)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_plain_default_home_still_spawns(self):
        _repo, sub = self._repo()
        self._root(os.path.join(self.home, ".claude"))
        argv, _env, _meta = self._prepare(sub)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_named_config_dir_home_still_spawns(self):
        # vendor_root is None for a named home and its state rows sit under
        # the profile root, not under ~/.claude: the host fence must not
        # refuse it as "vendor state rows without the root".
        _repo, sub = self._repo()
        named = self._root(os.path.join(self.home, ".claude-work"))
        argv, _env, _meta = self._prepare(
            sub, env={"CLAUDE_CONFIG_DIR": named})
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_named_home_with_default_home_aliasing_home_spawns(self):
        # HOME/.claude -> HOME: realpath(~/.claude) is an ancestor of the
        # named profile root, so its state rows sit "inside" the default
        # home. The named home is judged by the caller-config-dir rule, not
        # by the default-home vendor-root guard.
        named = self._root(os.path.join(self.home, ".claude-work"))
        os.symlink(self.home, os.path.join(self.home, ".claude"))
        argv, _env, _meta = self._prepare(
            self.wd, env={"CLAUDE_CONFIG_DIR": named})
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_named_home_with_default_home_into_the_profile_spawns(self):
        # HOME/.claude -> HOME/.claude-work/projects: the default home
        # resolves into the named profile root.
        named = self._root(os.path.join(self.home, ".claude-work"))
        os.symlink(os.path.join(named, "projects"),
                   os.path.join(self.home, ".claude"))
        argv, _env, _meta = self._prepare(
            self.wd, env={"CLAUDE_CONFIG_DIR": named})
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_default_home_policy_without_its_vendor_root_still_refuses(self):
        # The guard still covers a default-home claude run: a caller-supplied
        # policy whose state rows sit under ~/.claude but which dropped the
        # vendor root cannot be checked against the composed write set.
        self._root(os.path.join(self.home, ".claude"))
        policy = self._policy_for(self.wd)
        self.assertTrue(policy.get("vendor_root"), policy)
        policy = dict(policy)
        policy.pop("vendor_root")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            self._prepare(self.wd, policy=policy)
        self.assertEqual(caught.exception.envelope["reason"],
                         "fence_path_rejected")
        self.assertIn("vendor state write rows without its vendor root",
                      caught.exception.envelope["detail"])

    # --- the config-dir binding (fix round 2) -------------------------------

    def _assert_prepare_refused(self, *args, detail=None, **kw):
        with self.assertRaises(host_fence.HostFenceError) as caught:
            self._prepare(*args, **kw)
        self.assertEqual(caught.exception.envelope["reason"],
                         "fence_path_rejected")
        if detail:
            self.assertIn(detail, caught.exception.envelope["detail"])

    def test_keyword_config_dir_reaches_the_child_environment(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        argv, env, _meta = self._prepare(self.wd, config_dir=named)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), named)

    def test_keyword_config_dir_with_a_different_env_home_refuses(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        other = self._root(os.path.join(self.home, ".claude-other"))
        self._assert_prepare_refused(
            self.wd, env={"CLAUDE_CONFIG_DIR": other}, config_dir=named,
            detail="differs from the fenced claude config dir")

    def test_keyword_config_dir_with_the_same_env_home_spawns(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        _argv, env, _meta = self._prepare(
            self.wd, env={"CLAUDE_CONFIG_DIR": named}, config_dir=named)
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), named)

    def test_keyword_named_home_never_leaves_the_child_on_the_default_home(self):
        # HOME/.claude -> wd/cfg -> N/projects: the default home's chain link
        # (wd/cfg) is writable, and a keyword-only config dir used to build
        # the policy without ever reaching the child, which then ran on the
        # default home with both vendor-root guards skipped.
        named = self._root(os.path.join(self.home, ".claude-work"))
        os.symlink(os.path.join(named, "projects"),
                   os.path.join(self.wd, "cfg"))
        os.symlink(os.path.join(self.wd, "cfg"),
                   os.path.join(self.home, ".claude"))
        # The run is admitted, and the child runs on the keyword's home. Any
        # refusal (this one or an unrelated one) fails the test.
        _argv, env, _meta = self._prepare(self.wd, config_dir=named)
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), named)

    # --- a relative home means another path in the child (fix round 3) ------

    def _relative_home(self):
        """A real profile root that the server's cwd reaches as a relative
        name, so only the relative spelling can make the run refuse."""
        named = self._root(os.path.join(self.home, ".claude-work"))
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.home)
        self.assertEqual(os.path.abspath(".claude-work"), named)
        return ".claude-work", named

    def test_relative_keyword_config_dir_refuses(self):
        rel, _named = self._relative_home()
        self._assert_prepare_refused(
            self.wd, config_dir=rel, detail="must be an absolute path")

    def test_relative_env_config_dir_refuses(self):
        rel, _named = self._relative_home()
        self._assert_prepare_refused(
            self.wd, env={"CLAUDE_CONFIG_DIR": rel},
            detail="must be an absolute path")

    def test_relative_keyword_with_a_matching_absolute_env_refuses(self):
        # The server's cwd resolves the keyword to the env home, but
        # host-exec chdirs into the workdir before the child resolves it.
        rel, named = self._relative_home()
        self._assert_prepare_refused(
            self.wd, env={"CLAUDE_CONFIG_DIR": named}, config_dir=rel,
            detail="must be an absolute path")

    def _default_policy_without_root(self, workdir):
        policy = dict(self._policy_for(workdir))
        self.assertTrue(policy.get("vendor_root"), policy)
        policy.pop("vendor_root")
        return policy

    def _linked_default_home_in_a_repo(self):
        repo, sub = self._repo()
        real = self._root(os.path.join(self.home, "elsewhere"))
        os.symlink(real, os.path.join(repo, "cfg"))
        os.symlink(os.path.join(repo, "cfg"),
                   os.path.join(self.home, ".claude"))
        return sub

    def test_default_home_policy_with_a_keyword_config_dir_refuses(self):
        sub = self._linked_default_home_in_a_repo()
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = self._default_policy_without_root(sub)
        self._assert_prepare_refused(
            sub, policy=policy, config_dir=named,
            detail="built for a different config home")

    def test_default_home_policy_with_an_env_config_dir_refuses(self):
        sub = self._linked_default_home_in_a_repo()
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = self._default_policy_without_root(sub)
        self._assert_prepare_refused(
            sub, policy=policy, env={"CLAUDE_CONFIG_DIR": named},
            detail="built for a different config home")

    def test_named_home_policy_with_an_empty_config_dir_refuses(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = self._policy(config_dir=named)
        self.assertEqual(policy.get("claude_config_root"), named)
        self._assert_prepare_refused(
            self.wd, policy=policy, config_dir="",
            detail="built for a different config home")
        self._assert_prepare_refused(
            self.wd, policy=policy,
            detail="built for a different config home")

    def test_named_home_policy_for_another_named_home_refuses(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        other = self._root(os.path.join(self.home, ".claude-other"))
        policy = self._policy(config_dir=named)
        self._assert_prepare_refused(
            self.wd, policy=policy, config_dir=other,
            detail="built for a different config home")

    def test_named_run_with_a_policy_without_a_recorded_home_refuses(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = dict(self._policy(config_dir=named))
        policy.pop("claude_config_root")
        self._assert_prepare_refused(
            self.wd, policy=policy, config_dir=named,
            detail="does not record the config home")

    def test_default_run_with_a_policy_without_a_recorded_home_spawns(self):
        # A hand-built default-home policy (no key) is still judged by the
        # default-home guards; only a named run needs the key.
        self._root(os.path.join(self.home, ".claude"))
        policy = dict(self._policy_for(self.wd))
        policy.pop("claude_config_root")
        argv, _env, _meta = self._prepare(self.wd, policy=policy)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")

    def test_matching_named_home_policy_spawns(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = self._policy(config_dir=named)
        _argv, env, _meta = self._prepare(
            self.wd, policy=policy, config_dir=named)
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), named)

    def test_default_home_policy_rows_sit_under_the_recorded_root(self):
        real = self._root(os.path.join(self.home, "dotfiles", ".claude"))
        os.symlink(real, os.path.join(self.home, ".claude"))
        policy = self._policy_for(self.wd)
        root = policy["vendor_root"]
        self.assertEqual(root, real)
        rows = policy["wrap_kw"]["extra_post_write"]
        self.assertTrue(rows)
        for entry in rows:
            _kind, path = kernel_sandbox._write_entry_bound(*entry)
            self.assertTrue(path == root or path.startswith(root + os.sep),
                            (path, root))


class NamedHome(_Base):
    def test_plain_named_config_dir_home_spawns(self):
        named = self._root(os.path.join(self.home, ".claude-work"))
        policy = self._policy(config_dir=named)
        self._assert_spawns(policy)
        self.assertIsNone(policy.get("vendor_root"))

    def test_default_home_chain_is_not_judged_on_a_named_run(self):
        # Named run: ~/.claude is not its home, even when it is a writable link.
        self._l1()
        named = self._root(os.path.join(self.home, ".claude-work"))
        self._assert_spawns(self._policy(config_dir=named))


if __name__ == "__main__":
    unittest.main()
