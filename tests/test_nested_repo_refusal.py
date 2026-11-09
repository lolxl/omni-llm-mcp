"""Write mode refuses a non-git workdir that holds a nested Git repository (#187)."""
import contextlib
import io
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

import omni_cli
from providers import host_fence as h
from providers import known_secrets

SECRET_LOOKING = "sk-" + "A" * 30


class NestedRepoRefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.realpath(self.tmp.name)
        self.outside = tempfile.TemporaryDirectory()
        self.addCleanup(self.outside.cleanup)
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        stack = __import__("contextlib").ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": self.state.name}))
        stack.enter_context(mock.patch.object(h.platform, "system", return_value="Darwin"))
        stack.enter_context(mock.patch.object(h.kernel_sandbox, "available", return_value=True))
        stack.enter_context(mock.patch.object(
            h.kernel_sandbox, "agent_fence_policy", return_value={"vendor_state_writable": True}))
        stack.enter_context(mock.patch.object(
            h.kernel_sandbox, "wrap_kwargs_from_policy",
            return_value={"read_wall": True, "extra_deny_read": tuple(h._grok_instruction_rows(None))}))
        stack.enter_context(mock.patch.object(h.kernel_sandbox, "verify_fence", return_value=None))
        stack.enter_context(mock.patch.object(
            h.kernel_sandbox, "wrap_argv",
            side_effect=lambda argv, *a, **kw: ["/usr/bin/sandbox-exec", "-p", "fixture", *argv]))
        stack.enter_context(mock.patch.object(
            h.kernel_sandbox, "agent_child_env", side_effect=lambda env, **kw: env))

    def prepare(self, **kw):
        return h.prepare_host_fence(
            ["/fake/grok", "-p", "fixture"], provider="grok", workdir=self.work, env={}, **kw)

    def refusal(self, **kw):
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(**kw)
        return ctx.exception.envelope

    def mkdirs(self, *parts):
        path = os.path.join(self.work, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def test_nested_dot_git_directory_is_refused(self):
        self.mkdirs("tools", "vendor", "lib", ".git")
        env = self.refusal()
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("nested Git repository", env["detail"])
        self.assertIn(os.path.join("tools", "vendor", "lib", ".git"), env["detail"])
        self.assertNotIn(self.work, env["detail"])

    def test_nested_dot_git_pointer_file_is_refused(self):
        sub = self.mkdirs("clone")
        with open(os.path.join(sub, ".git"), "w") as f:
            f.write("gitdir: ../elsewhere/.git/worktrees/clone\n")
        env = self.refusal()
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn(os.path.join("clone", ".git"), env["detail"])

    def test_nested_bare_looking_directory_is_refused(self):
        bare = self.mkdirs("mirror.git")
        os.mkdir(os.path.join(bare, "objects"))
        os.mkdir(os.path.join(bare, "refs"))
        with open(os.path.join(bare, "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")
        env = self.refusal()
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("bare repository", env["detail"])
        self.assertIn("mirror.git", env["detail"])

    def test_bare_look_is_found_in_any_case_spelling(self):
        # A case-insensitive volume lets Git open "head" or "Objects", so each
        # of the three names counts in any spelling (one variant altered each).
        variants = [("head", "objects", "refs"), ("HEAD", "Objects", "refs"),
                    ("HEAD", "objects", "REFS"), ("Head", "OBJECTS", "Refs")]
        for i, (head, objects, refs) in enumerate(variants):
            with self.subTest(head=head, objects=objects, refs=refs):
                name = "v%d" % i
                bare = self.mkdirs(name)
                self.addCleanup(shutil.rmtree, bare, True)
                os.mkdir(os.path.join(bare, objects))
                os.mkdir(os.path.join(bare, refs))
                with open(os.path.join(bare, head), "w") as f:
                    f.write("ref: refs/heads/main\n")
                env = self.refusal()
                self.assertEqual(env["reason"], "fence_path_rejected")
                self.assertIn("bare repository", env["detail"])
                self.assertIn(name, env["detail"])
                shutil.rmtree(bare)

    def test_partial_bare_look_is_not_a_hit(self):
        part = self.mkdirs("data")
        os.mkdir(os.path.join(part, "objects"))
        with open(os.path.join(part, "HEAD"), "w") as f:
            f.write("x")
        self.prepare()  # no refs/ directory: not a bare repository

    def test_symlinked_directory_is_not_descended(self):
        repo = os.path.join(self.outside.name, "repo")
        os.makedirs(os.path.join(repo, ".git"))
        os.symlink(repo, os.path.join(self.work, "link"))
        argv, _env, _meta = self.prepare()  # contents of the target are not a hit
        self.assertEqual(argv[:3], ["/usr/bin/sandbox-exec", "-p", "fixture"])

    def test_symlink_named_dot_git_is_a_hit(self):
        sub = self.mkdirs("proj")
        os.symlink(self.outside.name, os.path.join(sub, ".git"))
        env = self.refusal()
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn(os.path.join("proj", ".git"), env["detail"])

    def test_case_variant_of_dot_git_is_a_hit(self):
        self.mkdirs("proj", ".GIT")
        self.assertIn("nested Git repository", self.refusal()["detail"])

    def test_cap_exceeded_refuses_with_incomplete_scan_reason(self):
        for i in range(60):
            open(os.path.join(self.work, "f%03d" % i), "w").close()
        with mock.patch.object(h, "NESTED_REPO_SCAN_CAP", 50):
            env = self.refusal()
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("could not verify", env["detail"])
        self.assertIn("scan cap 50 entries", env["detail"])
        self.assertNotIn("nested Git repository (", env["detail"])

    def test_default_cap_is_named_and_stated(self):
        self.assertEqual(h.NESTED_REPO_SCAN_CAP, 20000)

    def test_unreadable_directory_fails_closed(self):
        if os.geteuid() == 0:
            self.skipTest("root reads every directory")
        locked = self.mkdirs("locked")
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, stat.S_IRWXU)
        env = self.refusal()
        self.assertIn("could not verify", env["detail"])
        self.assertIn("could not be read", env["detail"])

    def test_clean_non_git_workdir_is_admitted(self):
        self.mkdirs("src", "pkg")
        open(os.path.join(self.work, "src", "pkg", "mod.py"), "w").close()
        argv, _env, meta = self.prepare()
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        self.assertTrue(meta["fence_detail"]["workdir_writable"])

    def test_read_only_mode_is_not_refused_by_the_check(self):
        self.mkdirs("clone", ".git")
        argv, _env, meta = self.prepare(read_only=True)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        self.assertFalse(meta["fence_detail"]["workdir_writable"])

    def assertNestedRefusal(self, env):
        # The nested scan fired, not some unrelated refusal.
        self.assertEqual(env["reason"], "fence_path_rejected")
        self.assertIn("nested Git repository", env["detail"])

    def assertNoRun(self, secret, detail, width=8):
        for i in range(len(secret) - width + 1):
            self.assertNotIn(secret[i:i + width], detail)

    def test_refusal_masks_secret_looking_names(self):
        self.mkdirs(SECRET_LOOKING, ".git")
        env = self.refusal()
        self.assertNestedRefusal(env)
        self.assertNotIn(SECRET_LOOKING, env["detail"])

    def test_credential_shaped_name_spanning_the_cut_is_masked_whole(self):
        body = "B" * 40
        # "sk-" starts at offset 181 of the path, so the 200 character cut
        # would keep 19 characters of it if the cut came before the mask.
        self.mkdirs(*(["a"] * 90 + ["b" + "sk-" + body, ".git"]))
        env = self.refusal()
        self.assertNestedRefusal(env)
        self.assertNoRun(body, env["detail"])

    def test_known_value_spanning_the_cut_is_masked_whole(self):
        value = "qZ7rT2wK" * 8
        self.addCleanup(known_secrets.clear)
        known_secrets.register_many([value], authoritative=True)
        self.mkdirs(*(["a"] * 95 + [value, ".git"]))
        env = self.refusal()
        self.assertNestedRefusal(env)
        self.assertNoRun(value, env["detail"])

    def test_refusal_happens_before_any_launch(self):
        self.mkdirs("clone", ".git")
        with mock.patch.object(h.kernel_sandbox, "wrap_argv") as wrap:
            with self.assertRaises(h.HostFenceError) as ctx:
                self.prepare()
        self.assertNestedRefusal(ctx.exception.envelope)
        wrap.assert_not_called()

    def test_host_exec_refusal_masks_an_unregistered_credential(self):
        """Through ``omni_cli.main``: credentials are primed before the
        host-exec diagnostic prints a path named with one."""
        secret = "zQ7vK2mXp9LrT4wBn6"
        self.mkdirs(secret, ".git")
        known_secrets.clear()
        self.addCleanup(known_secrets.clear)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"CURSOR_API_KEY": secret}), \
                mock.patch.object(known_secrets, "register_auth_files"), \
                mock.patch.object(h.os, "execvpe",
                                  side_effect=AssertionError("exec reached")), \
                contextlib.redirect_stderr(err):
            rc = omni_cli.main(["host-exec", "--provider", "grok",
                                "--workdir", self.work, "--", "/fake/grok"])
        self.assertEqual(rc, 126)
        self.assertIn("nested Git repository", err.getvalue())
        self.assertNotIn(secret, err.getvalue())
        self.assertNotIn(secret[:10], err.getvalue())


class NestedRepoExemptTests(unittest.TestCase):
    """The ``exempt`` hook of ``_nested_repo_hit`` (#199)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)

    def mk(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def test_exempt_hit_is_skipped_and_walk_continues(self):
        self.mk(".git")
        self.mk("a", "b", ".git")
        seen = []

        def exempt(rel, entry):
            seen.append(rel)
            return rel == ".git"

        self.assertEqual(h._nested_repo_hit(self.root, exempt),
                         ("repo", os.path.join("a", "b", ".git")))
        self.assertEqual(seen, [".git", os.path.join("a", "b", ".git")])

    def test_exempt_git_directory_is_not_descended(self):
        # A bare-looking layout inside an exempt .git must not be reported.
        for name in ("objects", "refs"):
            self.mk(".git", name)
        with open(os.path.join(self.root, ".git", "HEAD"), "w") as fh:
            fh.write("ref: refs/heads/main\n")
        self.assertEqual(h._nested_repo_hit(self.root, lambda rel, e: True),
                         (None, None))

    def test_everything_exempt_is_clean_and_default_is_unchanged(self):
        self.mk("x", ".git")
        self.assertEqual(h._nested_repo_hit(self.root, lambda rel, e: True),
                         (None, None))
        self.assertEqual(h._nested_repo_hit(self.root),
                         ("repo", os.path.join("x", ".git")))
        self.assertEqual(h._nested_repo_hit(self.root, lambda rel, e: False),
                         ("repo", os.path.join("x", ".git")))

    def test_exempt_does_not_hide_bare_or_cap(self):
        self.mk("bare", "objects")
        self.mk("bare", "refs")
        with open(os.path.join(self.root, "bare", "HEAD"), "w") as fh:
            fh.write("ref: refs/heads/main\n")
        self.assertEqual(h._nested_repo_hit(self.root, lambda rel, e: True),
                         ("bare", "bare"))
        with mock.patch.object(h, "NESTED_REPO_SCAN_CAP", 1):
            kind, _ = h._nested_repo_hit(self.root, lambda rel, e: True)
        self.assertEqual(kind, "incomplete")


if __name__ == "__main__":
    unittest.main()
