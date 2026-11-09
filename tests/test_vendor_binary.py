"""Vendor binary admission shared by the agent policy and the codex consult.

A vendor CLI installed inside the protected home (for example
``~/.codex/packages/standalone/current/bin/codex`` reached through the
``~/.local/bin/codex`` symlink) must be exec'd by its realpath and admitted as
a literal read: Seatbelt refuses an execvp through the home-tree symlink even
when the target file is admitted, and the codex consult passes its own policy,
so ``agent_fence_policy`` never admitted the binary there.
"""
import asyncio
import os
import tempfile
import unittest
from unittest import mock

from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import host_fence
from providers import kernel_sandbox
from providers import pi as pi_mod
from providers.host_fence import HostFenceError


class _FakeHome(unittest.TestCase):
    def setUp(self):
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="omni_vbin_home_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.home, True))
        patcher = mock.patch.object(
            kernel_sandbox, "protected_home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.leaf_dir = os.path.join(
            self.home, ".codex", "packages", "standalone", "current", "bin")
        os.makedirs(self.leaf_dir)
        self.leaf = os.path.join(self.leaf_dir, "codex")
        with open(self.leaf, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(self.leaf, 0o755)
        link_dir = os.path.join(self.home, ".local", "bin")
        os.makedirs(link_dir)
        self.link = os.path.join(link_dir, "codex")
        os.symlink(self.leaf, self.link)


class AdmitVendorBinaryTest(_FakeHome):
    def test_outside_home_needs_no_entry(self):
        self.assertIsNone(kernel_sandbox.admit_vendor_binary("/usr/bin/true"))
        self.assertIsNone(kernel_sandbox.admit_vendor_binary(None))

    def test_inside_home_is_a_literal_on_the_leaf(self):
        ap = kernel_sandbox.admit_vendor_binary(self.leaf)
        self.assertIsNotNone(ap)
        self.assertEqual(ap.path, self.leaf)
        self.assertEqual(ap.kind, "literal")

    def test_inside_home_directory_is_rejected(self):
        with self.assertRaises(kernel_sandbox.FencePathRejected) as ctx:
            kernel_sandbox.admit_vendor_binary(self.leaf_dir)
        self.assertIn("vendor binary is not a regular file under home",
                      ctx.exception.detail)

    def test_home_itself_is_rejected(self):
        with self.assertRaises(kernel_sandbox.FencePathRejected):
            kernel_sandbox.admit_vendor_binary(self.home)


class CodexConsultBinaryTest(_FakeHome):
    def _run(self, which_result):
        seen = {}

        def _prepare(argv, **kw):
            seen["argv"] = list(argv)
            seen["policy"] = kw.get("policy")
            raise HostFenceError({"ok": False, "reason": "fence_unavailable",
                                  "detail": "captured"})

        with mock.patch.object(codex_mod.shutil, "which",
                               return_value=which_result), \
                mock.patch.object(codex_mod, "prepare_host_fence", _prepare):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hello", "fixture-codex", 5, "default"))
        return res, seen

    def test_symlinked_in_home_binary_execs_the_leaf_and_admits_it(self):
        res, seen = self._run(self.link)
        self.assertEqual(res.get("detail"), "captured", res)
        self.assertEqual(seen["argv"][0], self.leaf)
        reads = seen["policy"]["wrap_kw"]["extra_read_paths"]
        paths = [getattr(ap, "path", ap) for ap in reads]
        self.assertIn(self.leaf, paths)
        self.assertNotIn(self.link, paths)
        self.assertNotIn(self.leaf_dir, paths)

    def test_outside_home_binary_adds_no_read_entry(self):
        res, seen = self._run("/usr/bin/true")
        self.assertEqual(res.get("detail"), "captured", res)
        self.assertEqual(seen["argv"][0], "/usr/bin/true")
        self.assertEqual(list(seen["policy"]["wrap_kw"]["extra_read_paths"]), [])


class HostExecBinaryTest(_FakeHome):
    """``omni host-exec`` callers pass the CLI by name or through the symlink."""

    def _main(self, head, workdir=None, provider="codex", which=None):
        which = (which or {"codex": self.link, "git": "/usr/bin/git"}).get
        with mock.patch.object(host_fence.shutil, "which",
                               side_effect=lambda name, **_kw: which(name)), \
                mock.patch.object(host_fence, "prepare_host_fence",
                                  return_value=(["/fake"], {}, {})) as prepare, \
                mock.patch.object(host_fence.os, "chdir"), \
                mock.patch.object(host_fence.os, "execvpe"):
            host_fence.main(["--provider", provider,
                             "--workdir", workdir or self.home,
                             "--read-only", "--", head, "exec", "-"])
        return prepare.call_args.args[0]

    def test_bare_name_execs_the_leaf(self):
        self.assertEqual(self._main("codex"), [self.leaf, "exec", "-"])

    def test_symlink_path_execs_the_leaf(self):
        self.assertEqual(self._main(self.link), [self.leaf, "exec", "-"])

    def test_other_command_is_unchanged(self):
        self.assertEqual(self._main("/bin/cat"), ["/bin/cat", "exec", "-"])
        self.assertEqual(self._main("cat"), ["cat", "exec", "-"])

    def test_bare_name_of_another_binary_is_unchanged(self):
        self.assertEqual(self._main("git"), ["git", "exec", "-"])

    def test_relative_path_resolves_against_workdir(self):
        link_dir = os.path.dirname(self.link)
        # ./codex in the workdir is the vendor symlink: exec the leaf.
        self.assertEqual(self._main("./codex", workdir=link_dir),
                         [self.leaf, "exec", "-"])
        # The invoker cwd holding the vendor symlink must not matter when
        # the workdir's ./codex is another program.
        other = os.path.join(self.home, "repo")
        os.makedirs(other)
        with open(os.path.join(other, "codex"), "w") as f:
            f.write("#!/bin/sh\n")
        cwd = os.getcwd()
        os.chdir(link_dir)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(self._main("./codex", workdir=other),
                         ["./codex", "exec", "-"])

    def test_bare_name_via_relative_path_entry_is_unchanged(self):
        # PATH=bin: which() answers relative to the invoker cwd, where the
        # vendor symlink sits; exec happens in the workdir after chdir.
        local = os.path.dirname(os.path.dirname(self.link))
        cwd = os.getcwd()
        os.chdir(local)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(
            self._main("codex", which={"codex": os.path.join("bin", "codex")}),
            ["codex", "exec", "-"])

    def test_override_binary_does_not_capture_a_different_path_hit(self):
        # OMNI_LLM_CLAUDE_BIN names the vendor; PATH's claude is another file.
        with mock.patch.dict(os.environ,
                             {kernel_sandbox.ENV_CLAUDE_BIN: self.leaf}):
            self.assertEqual(
                self._main("claude", provider="claude",
                           which={"claude": "/usr/bin/true"}),
                ["claude", "exec", "-"])


# Parent-env keys that would divert resolve_agent_binary or add a profile
# read. The consult copies os.environ; the fence also scans it directly.
_CONSULT_ENV_EXACT = (
    "OMNI_LLM_CLAUDE_BIN",
    "OMNI_LLM_PI_BIN",
    "OMNI_LLM_CURSOR_BIN",
    "OMNI_LLM_DEPTH",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
)
_CONSULT_ENV_PREFIXES = ("CLAUDE_CONFIG_DIR_", "CODEX_HOME_")


class _ConsultThroughFence(_FakeHome):
    """Claude and pi consults do not pass a policy.

    ``prepare_host_fence`` therefore builds one with ``agent_fence_policy``,
    which already admits the resolved leaf. These cases exec by realpath and
    capture the read set at ``verify_fence`` — the set the fence actually
    checks — rather than a policy argument the consult never supplies.
    """

    module = None
    runner = None
    bin_name = None
    env_key = None
    model = None
    rel_parts = None

    def setUp(self):
        super().setUp()
        self.leaf_dir = os.path.join(self.home, *self.rel_parts)
        os.makedirs(self.leaf_dir, exist_ok=True)
        self.leaf = os.path.join(self.leaf_dir, self.bin_name)
        with open(self.leaf, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(self.leaf, 0o755)
        self.link = os.path.join(self.home, ".local", "bin", self.bin_name)
        os.symlink(self.leaf, self.link)

    def _quiet_environ(self):
        saved = {}
        for key in list(os.environ):
            if key in _CONSULT_ENV_EXACT or key.startswith(_CONSULT_ENV_PREFIXES):
                saved[key] = os.environ.pop(key)
        return saved

    def _run(self, which_result, override=None):
        seen = {}

        def _which(name, *args, **kwargs):
            if name == self.bin_name:
                return which_result
            return None

        def _verify(workdir, read_only=False, **kw):
            seen["reads"] = list(kw.get("extra_read_paths") or ())
            return {"ok": False, "reason": "fence_unavailable",
                    "detail": "captured"}

        original = self.module.prepare_host_fence

        def _prepare(argv, **kw):
            seen["argv"] = list(argv)
            seen["policy_arg"] = kw.get("policy")
            return original(argv, **kw)

        saved_env = self._quiet_environ()
        if override is not None:
            os.environ[self.env_key] = override
        try:
            with mock.patch("shutil.which", side_effect=_which), \
                    mock.patch.object(host_fence.platform, "system",
                                      return_value="Darwin"), \
                    mock.patch.object(kernel_sandbox, "available",
                                      return_value=True), \
                    mock.patch.object(kernel_sandbox, "claude_scratch_root",
                                      return_value=(None, None)), \
                    mock.patch.object(kernel_sandbox, "verify_fence", _verify), \
                    mock.patch.object(self.module, "prepare_host_fence",
                                      _prepare):
                res = asyncio.run(self.runner(
                    "hello", self.model, 5, "default"))
        finally:
            os.environ.pop(self.env_key, None)
            os.environ.update(saved_env)
        return res, seen

    def _read_paths(self, seen):
        return [getattr(ap, "path", ap) for ap in seen["reads"]]

    def _assert_in_home(self):
        res, seen = self._run(self.link)
        self.assertEqual(res.get("detail"), "captured", res)
        self.assertIsNone(seen["policy_arg"])
        self.assertEqual(seen["argv"][0], self.leaf)
        paths = self._read_paths(seen)
        self.assertIn(self.leaf, paths)
        self.assertNotIn(self.link, paths)
        self.assertNotIn(self.leaf_dir, paths)
        admitted = [ap for ap in seen["reads"]
                    if getattr(ap, "path", ap) == self.leaf]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0].kind, "literal")

    def _assert_override_is_the_single_source(self):
        # The override names another in-home version than the PATH symlink:
        # exec and admission must both follow the override, never split.
        other_dir = os.path.join(os.path.dirname(self.leaf_dir), "1.0.0")
        os.makedirs(other_dir)
        other = os.path.join(other_dir, self.bin_name)
        with open(other, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(other, 0o755)
        res, seen = self._run(self.link, override=other)
        self.assertEqual(res.get("detail"), "captured", res)
        self.assertEqual(seen["argv"][0], other)
        paths = self._read_paths(seen)
        self.assertIn(other, paths)
        self.assertNotIn(self.leaf, paths)

    def _assert_outside(self):
        res, seen = self._run("/usr/bin/true")
        self.assertEqual(res.get("detail"), "captured", res)
        self.assertIsNone(seen["policy_arg"])
        self.assertEqual(seen["argv"][0], "/usr/bin/true")
        self.assertEqual(self._read_paths(seen), [])


class ClaudeConsultBinaryTest(_ConsultThroughFence):
    module = claude_mod
    runner = staticmethod(claude_mod.run_claude_subprocess)
    bin_name = "claude"
    env_key = kernel_sandbox.ENV_CLAUDE_BIN
    model = "fixture-claude"
    rel_parts = (".local", "share", "claude", "versions", "9.9.9")

    def test_symlinked_in_home_binary_execs_the_leaf_and_admits_it(self):
        self._assert_in_home()

    def test_outside_home_binary_adds_no_read_entry(self):
        self._assert_outside()

    def test_override_is_the_single_source(self):
        self._assert_override_is_the_single_source()


class PiConsultBinaryTest(_ConsultThroughFence):
    module = pi_mod
    runner = staticmethod(pi_mod.run_pi_subprocess)
    bin_name = "pi"
    env_key = kernel_sandbox.ENV_PI_BIN
    model = "fixture-pi"
    rel_parts = (".local", "share", "pi", "versions", "9.9.9")

    def test_symlinked_in_home_binary_execs_the_leaf_and_admits_it(self):
        self._assert_in_home()

    def test_outside_home_binary_adds_no_read_entry(self):
        self._assert_outside()

    def test_override_is_the_single_source(self):
        self._assert_override_is_the_single_source()


if __name__ == "__main__":
    unittest.main()
