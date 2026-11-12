#!/usr/bin/env python3
"""Tests for the kernel Seatbelt fence (issues #26 write, #45 read wall).

Most cases are hermetic (argv / profile text / injected fakes). The live read
probe exercises real sandbox-exec when available and skips otherwise.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import unittest
from unittest import mock

from providers import agent_lane, kernel_sandbox, sandbox_canary
from tests import offline_home


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


class TestKernelSandboxProfile(unittest.TestCase):
    """S1 / S5.1–S5.2: profile text and realpath params."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="omni_ksb_wd_")
        self.addCleanup(_rmtree, self.workdir)

    def test_build_profile_clause_order_last_match_wins(self):
        # F4: Seatbelt is last-match-wins; pin exact relative order.
        profile_ro = kernel_sandbox.build_profile(self.workdir, read_only=True)
        i_allow_default = profile_ro.index("(allow default)")
        i_deny_write = profile_ro.index("(deny file-write*)")
        i_allow_write = profile_ro.index("(allow file-write*")
        self.assertLess(i_allow_default, i_deny_write)
        self.assertLess(i_deny_write, i_allow_write)

        self.assertIn('(literal "/dev/null")', profile_ro)
        for rx in kernel_sandbox.ALLOW_WRITE_REGEX:
            self.assertIn('(regex "%s")' % rx, profile_ro)

        params_ro = kernel_sandbox.profile_params(self.workdir, read_only=True)
        params_rw = kernel_sandbox.profile_params(self.workdir, read_only=False)
        real_wd = os.path.realpath(self.workdir)

        # Paths appear as params, never interpolated into profile text.
        for path in list(params_ro.values()) + list(params_rw.values()):
            self.assertNotIn(path, profile_ro)
        profile_rw = kernel_sandbox.build_profile(self.workdir, read_only=False)
        for path in list(params_ro.values()) + list(params_rw.values()):
            self.assertNotIn(path, profile_rw)
        i_allow_default_rw = profile_rw.index("(allow default)")
        i_deny_write_rw = profile_rw.index("(deny file-write*)")
        i_allow_write_rw = profile_rw.index("(allow file-write*")
        self.assertLess(i_allow_default_rw, i_deny_write_rw)
        self.assertLess(i_deny_write_rw, i_allow_write_rw)

        # RO omits cwd from write ALLOWS and explicitly binds its final deny.
        self.assertNotIn(real_wd, [v for k, v in params_ro.items() if k.startswith("P")])
        self.assertEqual(params_ro["Q0"], real_wd)
        # read_only=False DOES.
        self.assertIn(real_wd, params_rw.values())

        # Param keys in profile match profile_params.
        for key in params_ro:
            self.assertIn('(subpath (param "%s"))' % key, profile_ro)
        for key in params_rw:
            self.assertIn('(subpath (param "%s"))' % key, profile_rw)

    def test_workdir_symlink_resolves_in_params(self):
        base = tempfile.mkdtemp(prefix="omni_ksb_sym_")
        self.addCleanup(_rmtree, base)
        real = os.path.join(base, "real")
        link = os.path.join(base, "link")
        os.mkdir(real)
        os.symlink(real, link)
        params = kernel_sandbox.profile_params(link, read_only=False)
        self.assertIn(os.path.realpath(real), params.values())
        self.assertNotIn(link, params.values())

    def test_file_link_deny_is_after_allow_default_and_omitted_when_empty(self):
        entries = (("subpath", "/private/tmp/omni108-codex-home"),)
        profile = kernel_sandbox.build_profile(
            self.workdir, read_only=False, extra_deny_link=entries,
        )
        i_default = profile.index("(allow default)")
        i_allow_write = profile.index("(allow file-write*")
        i_link = profile.index("(deny file-link")
        self.assertGreater(i_link, i_default)
        self.assertGreater(i_link, i_allow_write)
        self.assertNotIn("/private/tmp/omni108-codex-home", profile)
        params = kernel_sandbox.profile_params(
            self.workdir, False, extra_deny_link=entries,
        )
        self.assertEqual(params.get("L0"), "/private/tmp/omni108-codex-home")
        self.assertIn('(subpath (param "L0"))', profile)
        bare = kernel_sandbox.build_profile(self.workdir, read_only=False)
        self.assertNotIn("file-link", bare)
        self.assertNotIn("L0", kernel_sandbox.profile_params(
            self.workdir, read_only=False))


class TestKernelSandboxWrapArgv(unittest.TestCase):
    """S5.3–S5.4: wrap_argv shape and identity when unavailable."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="omni_ksb_wrap_")
        self.addCleanup(_rmtree, self.workdir)
        self.old_available = kernel_sandbox.available
        self.old_which = kernel_sandbox.shutil.which
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox.shutil, "which", self.old_which)

    def test_wrap_argv_prefixes_sandbox_exec_and_preserves_tail(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.shutil.which = (
            lambda name, *a, **k: "/usr/bin/sandbox-exec"
            if name == "sandbox-exec" else self.old_which(name, *a, **k)
        )
        original = ["/fake/codex", "exec", "--sandbox", "read-only", "-"]
        wrapped = kernel_sandbox.wrap_argv(
            original, self.workdir, read_only=True,
        )
        self.assertEqual(wrapped[0], "/usr/bin/sandbox-exec")
        self.assertIn("-p", wrapped)
        p_idx = wrapped.index("-p")
        profile = wrapped[p_idx + 1]
        self.assertIn("(allow default)", profile)
        self.assertIn("(deny file-write*)", profile)

        params = kernel_sandbox.profile_params(self.workdir, read_only=True)
        for key, value in params.items():
            flag = "%s=%s" % (key, value)
            self.assertIn(flag, wrapped)
            d_idx = wrapped.index(flag) - 1
            self.assertEqual(wrapped[d_idx], "-D")

        # Original argv is the unbroken tail.
        self.assertEqual(wrapped[-len(original):], original)

    def test_wrap_argv_identity_when_unavailable(self):
        kernel_sandbox.available = lambda: False
        original = ["/fake/codex", "exec", "-"]
        result = kernel_sandbox.wrap_argv(original, self.workdir, read_only=True)
        self.assertIs(result, original)


class TestIsWrapped(unittest.TestCase):
    """F2: is_wrapped true/false cases."""

    def setUp(self):
        self.old_which = kernel_sandbox.shutil.which
        self.addCleanup(setattr, kernel_sandbox.shutil, "which", self.old_which)

    def test_is_wrapped_true_when_matches_which(self):
        kernel_sandbox.shutil.which = (
            lambda name, *a, **k: "/usr/bin/sandbox-exec"
            if name == "sandbox-exec" else self.old_which(name, *a, **k)
        )
        self.assertTrue(
            kernel_sandbox.is_wrapped(["/usr/bin/sandbox-exec", "-p", "x", "cmd"])
        )

    def test_is_wrapped_true_for_abs_basename_sandbox_exec(self):
        # PATH churn: which may return None or a different path.
        kernel_sandbox.shutil.which = lambda name, *a, **k: None
        self.assertTrue(
            kernel_sandbox.is_wrapped(["/opt/custom/sandbox-exec", "-p", "x"])
        )

    def test_is_wrapped_false_empty(self):
        self.assertFalse(kernel_sandbox.is_wrapped([]))

    def test_is_wrapped_false_unwrapped_codex(self):
        kernel_sandbox.shutil.which = (
            lambda name, *a, **k: "/usr/bin/sandbox-exec"
            if name == "sandbox-exec" else self.old_which(name, *a, **k)
        )
        self.assertFalse(kernel_sandbox.is_wrapped(["/fake/bin/codex", "exec", "-"]))

    def test_is_wrapped_false_relative_sandbox_exec(self):
        kernel_sandbox.shutil.which = lambda name, *a, **k: None
        self.assertFalse(kernel_sandbox.is_wrapped(["sandbox-exec", "-p", "x"]))


class TestKernelSandboxVerifyFence(unittest.TestCase):
    """F1: verify_fence two-part proof via the subprocess seam."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="omni_ksb_verify_wd_")
        self.addCleanup(_rmtree, self.workdir)
        self.old_available = kernel_sandbox.available
        self.old_run = kernel_sandbox.subprocess.run
        self.old_wrap = kernel_sandbox.wrap_argv
        self.old_tmpdir_rejected = kernel_sandbox._tmpdir_rejected
        self.old_codex_rejected = kernel_sandbox._codex_home_rejected
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox.subprocess, "run", self.old_run)
        self.addCleanup(setattr, kernel_sandbox, "wrap_argv", self.old_wrap)
        self.addCleanup(
            setattr, kernel_sandbox, "_tmpdir_rejected", self.old_tmpdir_rejected,
        )
        self.addCleanup(
            setattr, kernel_sandbox, "_codex_home_rejected", self.old_codex_rejected,
        )
        kernel_sandbox.available = lambda: True
        # Default: allow-roots accepted so F1 cases isolate probe logic.
        kernel_sandbox._tmpdir_rejected = lambda: None
        kernel_sandbox._codex_home_rejected = lambda: None
        # wrap_argv identity so fenced argv is still runnable under the fake.
        kernel_sandbox.wrap_argv = lambda argv, *a, **k: list(argv)
        self._call_n = 0

    def _fake_run_sequence(self, *stdout_list):
        """Return a subprocess.run fake that yields successive stdout strings."""
        outputs = list(stdout_list)

        def _run(argv, **kwargs):
            idx = self._call_n
            self._call_n += 1
            out = outputs[idx] if idx < len(outputs) else outputs[-1]
            # Token-only evidence (no home-dir writes in the hermetic harness).
            return subprocess.CompletedProcess(argv, 0, out, "")

        return _run

    def test_probe_tokens_require_exact_single_output(self):
        self.assertEqual(kernel_sandbox._parse_probe_stdout("WROTE\n"), "WROTE")
        self.assertIsNone(kernel_sandbox._parse_probe_stdout("prefix WROTE"))
        self.assertIsNone(kernel_sandbox._parse_probe_stdout("ERRNO=1\nextra"))
        self.assertEqual(kernel_sandbox._parse_read_probe_stdout("READ_OK\n"), "READ_OK")
        self.assertIsNone(kernel_sandbox._parse_read_probe_stdout("prefix READ_OK"))

    def test_probe_nonzero_exit_refuses(self):
        kernel_sandbox.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(
            a[0], 7, "ERRNO=1", ""
        )
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIn("exited 7", refusal["detail"])

    def test_probe_commands_keep_python_isolation_through_wrapping(self):
        wrapped = []
        kernel_sandbox.wrap_argv = lambda argv, *a, **k: (
            wrapped.append(list(argv)) or list(argv)
        )
        kernel_sandbox.subprocess.run = self._fake_run_sequence(
            "WROTE\n", "ERRNO=1\n", "ERRNO=1\n",
            "READ_OK\n", "ERRNO=1\n", "ERRNO=1\n",
        )
        self.assertIsNone(kernel_sandbox.verify_fence(
            self.workdir, read_only=True, read_wall=True,
        ))
        self.assertEqual(len(wrapped), 4)
        self.assertTrue(all(
            argv[1:3] == ["-I", "-S"] for argv in wrapped
        ))

    def test_verify_fence_control_write_fails_refuses(self):
        # Control (unwrapped) does not report WROTE → refuse, not pass.
        kernel_sandbox.subprocess.run = self._fake_run_sequence("ERRNO=13\n")
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertFalse(refusal["ok"])
        self.assertIn("control write failed", refusal["detail"])

    def test_verify_fence_wrapped_wrote_refuses(self):
        kernel_sandbox.subprocess.run = self._fake_run_sequence(
            "WROTE\n", "WROTE\n",
        )
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("fence did not deny", refusal["detail"])

    def test_verify_fence_wrapped_non_eperm_refuses(self):
        kernel_sandbox.subprocess.run = self._fake_run_sequence(
            "WROTE\n", "ERRNO=13\n",
        )
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("ERRNO=13", refusal["detail"])

    def test_verify_fence_wrapped_neither_token_refuses(self):
        kernel_sandbox.subprocess.run = self._fake_run_sequence(
            "WROTE\n", "sandbox_apply: Operation not permitted\n",
        )
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("neither WROTE nor ERRNO", refusal["detail"])

    def test_verify_fence_happy_path_returns_none(self):
        kernel_sandbox.subprocess.run = self._fake_run_sequence(
            "WROTE\n", "ERRNO=1\n", "ERRNO=1\n",
        )
        self.assertIsNone(kernel_sandbox.verify_fence(self.workdir, read_only=True))

    def test_verify_fence_refuses_when_default_home_write_deny_is_removed(self):
        observed = {"profiles": []}

        def _wrap(argv, workdir, read_only, **kwargs):
            profile = kernel_sandbox.build_profile(
                workdir, read_only, **kwargs,
            ).replace("(deny file-write*)", "")
            observed["profiles"].append(profile)
            return ["sandbox-exec", "-p", profile, *argv]

        def _run(argv, **kwargs):
            profile = argv[argv.index("-p") + 1] if "-p" in argv else None
            # This fake models the policy result: (allow default) permits the
            # canary when the profile's broad home write deny is removed.
            token = "WROTE\n" if profile is None or "(deny file-write*)" not in profile else "ERRNO=1\n"
            return subprocess.CompletedProcess(argv, 0, token, "")

        kernel_sandbox.wrap_argv = _wrap
        kernel_sandbox.subprocess.run = _run
        home_target = os.path.join(
            kernel_sandbox.protected_home(), "omni_ksb_removed_default_deny",
        )
        fd = os.open(os.devnull, os.O_RDONLY)
        with mock.patch.object(
            kernel_sandbox.tempfile, "mkstemp", return_value=(fd, home_target),
        ), mock.patch.object(kernel_sandbox.os, "remove"):
            refusal = kernel_sandbox.verify_fence(None, read_only=True)

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("fence did not deny", refusal["detail"])
        self.assertTrue(observed["profiles"])
        self.assertTrue(all("(deny file-write*)" not in p
                            for p in observed["profiles"]))

    def test_verify_fence_refuses_composed_write_post_allow(self):
        target = os.path.join(
            kernel_sandbox.protected_home(), "omni_ksb_composed_write_probe",
        )
        fd = os.open(os.devnull, os.O_RDONLY)
        observed = {"calls": 0, "profiles": []}
        post_write = (("literal", target),)

        def _run(argv, **kwargs):
            tokens = ("WROTE\n", "ERRNO=1\n", "WROTE\n")
            idx = observed["calls"]
            observed["calls"] += 1
            return subprocess.CompletedProcess(argv, 0, tokens[idx], "")

        def _wrap(argv, workdir, read_only, **kwargs):
            profile = kernel_sandbox.build_profile(
                workdir, read_only, **kwargs,
            )
            observed["profiles"].append(profile)
            return ["sandbox-exec", "-p", profile, *argv]

        kernel_sandbox.subprocess.run = _run
        kernel_sandbox.wrap_argv = _wrap
        with mock.patch.object(
            kernel_sandbox.tempfile, "mkstemp", return_value=(fd, target),
        ), mock.patch.object(kernel_sandbox.os, "remove"):
            refusal = kernel_sandbox.verify_fence(
                None, read_only=True, extra_post_write=post_write,
            )

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("composed-profile write probe expected ERRNO=1",
                      refusal["detail"])
        self.assertEqual(observed["calls"], 3)
        self.assertIn('(allow file-write* (literal (param "T0")))',
                      observed["profiles"][-1])

    def test_verify_fence_probe_checks_read_only_workdir_deny(self):
        observed = {}

        def _run(argv, **kwargs):
            observed.setdefault("target", kwargs["env"]["OMNI_KSB_TARGET"])
            token = "WROTE\n" if self._call_n == 0 else "ERRNO=1\n"
            self._call_n += 1
            return subprocess.CompletedProcess(argv, 0, token, "")

        def _wrap(argv, workdir, read_only, **kwargs):
            observed["extra_deny_write"] = kwargs.get("extra_deny_write", ())
            profile = kernel_sandbox.build_profile(
                workdir, read_only,
                extra_deny_write=observed["extra_deny_write"],
            )
            observed["profile"] = profile
            return list(argv)

        kernel_sandbox.subprocess.run = _run
        kernel_sandbox.wrap_argv = _wrap
        self.assertIsNone(kernel_sandbox.verify_fence(
            self.workdir, read_only=True,
        ))

        target = observed["target"]
        self.assertTrue(kernel_sandbox._path_inside(target, self.workdir))
        self.assertFalse(any(
            kernel_sandbox._path_inside(target, path)
            for _kind, path in (observed["extra_deny_write"] or ())
        ))
        self.assertIn('(deny file-write* (subpath (param "Q0")))',
                      observed["profile"])

    def test_verify_fence_consult_probe_uses_default_home_deny(self):
        home = kernel_sandbox.protected_home()
        target = os.path.join(home, "omni_ksb_verify_probe_test")
        fd = os.open(os.devnull, os.O_RDONLY)
        observed = {}

        def _run(argv, **kwargs):
            token = "WROTE\n" if self._call_n == 0 else "ERRNO=1\n"
            self._call_n += 1
            return subprocess.CompletedProcess(argv, 0, token, "")

        def _wrap(argv, workdir, read_only, **kwargs):
            observed["extra_deny_write"] = kwargs.get("extra_deny_write", ())
            observed["profile"] = kernel_sandbox.build_profile(
                workdir, read_only,
                extra_deny_write=observed["extra_deny_write"],
            )
            return list(argv)

        kernel_sandbox.subprocess.run = _run
        kernel_sandbox.wrap_argv = _wrap
        with mock.patch.object(
            kernel_sandbox.tempfile, "mkstemp", return_value=(fd, target),
        ), mock.patch.object(kernel_sandbox.os, "remove"):
            self.assertIsNone(kernel_sandbox.verify_fence(
                None, read_only=True,
            ))

        self.assertTrue(kernel_sandbox._path_inside(target, home))
        self.assertFalse(any(
            kernel_sandbox._path_inside(target, path)
            for _kind, path in (observed["extra_deny_write"] or ())
        ))
        self.assertIn("(deny file-write*)", observed["profile"])

    def test_verify_fence_read_wall_lists_home_without_canary_write(self):
        observed = {"calls": 0}

        def _run(argv, **kwargs):
            env = kwargs["env"]
            if "os.listdir(t)" in " ".join(argv):
                observed["read_target"] = env["OMNI_KSB_TARGET"]
                token = "READ_OK\n" if observed["calls"] == 3 else "ERRNO=1\n"
            else:
                observed["write_target"] = env["OMNI_KSB_TARGET"]
                token = "WROTE\n" if observed["calls"] == 0 else "ERRNO=1\n"
            observed["calls"] += 1
            return subprocess.CompletedProcess(argv, 0, token, "")

        kernel_sandbox.subprocess.run = _run
        self.assertIsNone(kernel_sandbox.verify_fence(
            self.workdir, read_only=True, read_wall=True,
        ))
        self.assertEqual(observed["calls"], 6)
        self.assertEqual(observed["read_target"], kernel_sandbox.protected_home())
        self.assertTrue(kernel_sandbox._path_inside(
            observed["write_target"], self.workdir,
        ))

    def test_verify_fence_composed_profile_refuses_home_read_reallow(self):
        home = kernel_sandbox.protected_home()
        target = os.path.join(home, "omni_ksb_composed_read_probe")
        fd = os.open(os.devnull, os.O_RDONLY)
        observed = {"calls": 0, "profiles": []}
        post_read = (("subpath", home),)

        def _run(argv, **kwargs):
            idx = observed["calls"]
            observed["calls"] += 1
            tokens = (
                "WROTE\n", "ERRNO=1\n", "ERRNO=1\n",
                "READ_OK\n", "ERRNO=1\n", "READ_OK\n",
            )
            return subprocess.CompletedProcess(argv, 0, tokens[idx], "")

        def _wrap(argv, workdir, read_only, **kwargs):
            profile = kernel_sandbox.build_profile(
                workdir, read_only, **kwargs,
            )
            observed["profiles"].append(profile)
            return ["sandbox-exec", "-p", profile, *argv]

        kernel_sandbox.subprocess.run = _run
        kernel_sandbox.wrap_argv = _wrap
        with mock.patch.object(
            kernel_sandbox.tempfile, "mkstemp", return_value=(fd, target),
        ), mock.patch.object(kernel_sandbox.os, "remove"):
            refusal = kernel_sandbox.verify_fence(
                None, read_only=True, read_wall=True,
                extra_post_read=post_read,
            )

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("composed-profile read-probe succeeded", refusal["detail"])
        self.assertEqual(observed["calls"], 6)
        self.assertIn('(allow file-read* (subpath (param "S0")))',
                      observed["profiles"][-1])

    def test_verify_fence_timeout_is_unverified(self):
        def _run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 5)

        kernel_sandbox.subprocess.run = _run
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")

    def test_verify_fence_never_raises(self):
        def _run(argv, **kwargs):
            raise RuntimeError("boom")

        kernel_sandbox.subprocess.run = _run
        refusal = kernel_sandbox.verify_fence(self.workdir, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")


class TestAllowRootValidation(unittest.TestCase):
    """F3: TMPDIR / ~/.codex allow-root validation."""

    def setUp(self):
        self.old_available = kernel_sandbox.available
        self.old_run = kernel_sandbox.subprocess.run
        self.old_wrap = kernel_sandbox.wrap_argv
        self.old_env_tmpdir = os.environ.get("TMPDIR")
        self.old_codex_rejected = kernel_sandbox._codex_home_rejected
        self.old_expanduser = kernel_sandbox.os.path.expanduser
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox.subprocess, "run", self.old_run)
        self.addCleanup(setattr, kernel_sandbox, "wrap_argv", self.old_wrap)
        self.addCleanup(self._restore_tmpdir)
        self.addCleanup(
            setattr, kernel_sandbox, "_codex_home_rejected", self.old_codex_rejected,
        )
        self.addCleanup(
            setattr, kernel_sandbox.os.path, "expanduser", self.old_expanduser,
        )
        kernel_sandbox.available = lambda: True
        kernel_sandbox.wrap_argv = lambda argv, *a, **k: list(argv)

        def _happy_run(argv, **kwargs):
            # Token-only: evidence is stdout, not filesystem side effects.
            n = getattr(_happy_run, "n", 0)
            _happy_run.n = n + 1
            if n == 0:
                return subprocess.CompletedProcess(argv, 0, "WROTE\n", "")
            return subprocess.CompletedProcess(argv, 0, "ERRNO=1\n", "")

        kernel_sandbox.subprocess.run = _happy_run

    def _restore_tmpdir(self):
        if self.old_env_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = self.old_env_tmpdir

    def test_tmpdir_outside_approved_absent_and_verify_refuses(self):
        # Sensitive path that is not under approved temp roots.
        bad = os.path.realpath(os.path.expanduser("~/.ssh"))
        os.environ["TMPDIR"] = bad
        # Allow-set must not contain the bad path.
        params = kernel_sandbox.profile_params(None, read_only=True)
        self.assertNotIn(bad, params.values())
        refusal = kernel_sandbox.verify_fence(None, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn(bad, refusal["detail"])
        self.assertIn("TMPDIR", refusal["detail"])

    def test_tmpdir_inside_approved_root_present(self):
        # Writable subdir under an approved root (/private/tmp is always listed).
        approved = kernel_sandbox._approved_temp_roots()
        self.assertTrue(approved, "host must have at least one approved temp root")
        parent = None
        for root in approved:
            if os.access(root, os.W_OK):
                parent = root
                break
        self.assertIsNotNone(parent, "no writable approved temp root on host")
        sub = tempfile.mkdtemp(prefix="omni_ksb_ok_tmp_", dir=parent)
        self.addCleanup(_rmtree, sub)
        os.environ["TMPDIR"] = sub
        # Clear tempfile cache so nothing stale interferes with other helpers.
        old_cache = tempfile.tempdir
        tempfile.tempdir = None
        self.addCleanup(setattr, tempfile, "tempdir", old_cache)

        params = kernel_sandbox.profile_params(None, read_only=True)
        self.assertIn(os.path.realpath(sub), params.values())
        # verify_fence should not refuse for TMPDIR reasons (happy probe faked).
        self.assertIsNone(kernel_sandbox.verify_fence(sub, read_only=True))

    def test_codex_home_outside_home_refused(self):
        # Clear inherited TMPDIR so this pin isolates ~/.codex rejection only.
        os.environ.pop("TMPDIR", None)
        outside = "/private/tmp/omni_fake_codex_home"
        # Force ~/.codex to resolve outside ~ via the validation seam.
        kernel_sandbox._codex_home_rejected = lambda: outside
        refusal = kernel_sandbox.verify_fence(None, read_only=True)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn(outside, refusal["detail"])
        self.assertIn("codex", refusal["detail"].lower())


class TestRunAgentKernelFenceWiring(unittest.TestCase):
    """Both macOS Codex lanes require host fencing; legacy cases pin Linux."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.workdir = tempfile.mkdtemp(prefix="omni_ksb_agent_")
        self.addCleanup(_rmtree, self.workdir)
        self.old_depth = os.environ.get(agent_lane.ENV_DEPTH)
        os.environ.pop(agent_lane.ENV_DEPTH, None)
        self.addCleanup(self._restore_depth)

        self.old_run_cli = agent_lane.run_cli
        self.old_which = agent_lane.shutil.which
        self.old_available = kernel_sandbox.available
        self.old_verify = kernel_sandbox.verify_fence
        self.old_wrap = kernel_sandbox.wrap_argv
        self.old_is_wrapped = kernel_sandbox.is_wrapped
        self.addCleanup(setattr, agent_lane, "run_cli", self.old_run_cli)
        self.addCleanup(setattr, agent_lane.shutil, "which", self.old_which)
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox, "verify_fence", self.old_verify)
        self.addCleanup(setattr, kernel_sandbox, "wrap_argv", self.old_wrap)
        self.addCleanup(setattr, kernel_sandbox, "is_wrapped", self.old_is_wrapped)

        # Stand the vendor canary down so read_only codex reaches argv build.
        self.old_gate = sandbox_canary.gate
        self.addCleanup(setattr, sandbox_canary, "gate", self.old_gate)
        sandbox_canary.gate = lambda *a, **k: None

        agent_lane.shutil.which = (
            lambda name, *a, **k: "/fake/bin/" + os.path.basename(str(name))
            if os.path.basename(str(name)) in ("codex", "claude", "grok", "pi",
                                               "cursor-agent", "sandbox-exec")
            else self.old_which(name, *a, **k)
        )

    def _restore_depth(self):
        if self.old_depth is None:
            os.environ.pop(agent_lane.ENV_DEPTH, None)
        else:
            os.environ[agent_lane.ENV_DEPTH] = self.old_depth

    def _real_sandbox_exec_which(self):
        """PATH where sandbox-exec resolves absolute, so is_wrapped is real."""
        def _which(name, *a, **k):
            base = os.path.basename(str(name))
            if base == "sandbox-exec":
                return "/usr/bin/sandbox-exec"
            if base in ("codex", "claude", "grok", "pi", "cursor-agent"):
                return "/fake/bin/" + base
            return self.old_which(name, *a, **k)

        agent_lane.shutil.which = _which
        kernel_sandbox.shutil.which = _which
        self.addCleanup(setattr, kernel_sandbox.shutil, "which", self.old_which)

    def _recorder(self):
        recorded = {"calls": 0}

        async def _fake_run_cli(argv, **kwargs):
            recorded["calls"] += 1
            recorded["argv"] = list(argv)
            if "-o" in argv:
                path = argv[argv.index("-o") + 1]
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("DONE")
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "timed_out": False}

        agent_lane.run_cli = _fake_run_cli
        return recorded

    def test_run_agent_wraps_codex_write_argv_and_disables_inner(self):
        # Issue #94 piece 3: the write lane rides the kernel wrap, and the
        # inner vendor sandbox is dropped for the bypass -- one Seatbelt per
        # spawn, because a nested sandbox_apply is rc=71.
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        self._real_sandbox_exec_which()
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        self.assertTrue(kernel_sandbox.is_wrapped(argv))
        self.assertFalse(kernel_sandbox.inner_vendor_sandbox_present(argv))
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        # The bypass is a flag of `exec`, and the prompt sentinel stays last.
        self.assertEqual(
            argv[argv.index("exec") + 1],
            "--dangerously-bypass-approvals-and-sandbox",
        )
        self.assertEqual(argv[-1], "-")
        self.assertEqual(env.get("fence"), "seatbelt-fs")
        self.assertTrue(env["fence_detail"]["read_wall"])
        self.assertTrue(env["fence_detail"]["workdir_writable"])

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_run_agent_codex_write_no_fence_falls_back_to_vendor(self):
        # --no-fence / OMNI_LLM_AGENT_FENCE=0: the kernel wrap is DECLINED,
        # so the argv keeps --sandbox workspace-write and the envelope must
        # say `vendor`, not `none` -- that spawn really is on the vendor
        # sandbox, and the stamp is what a reader would act on.
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        self._real_sandbox_exec_which()
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
            fence_requested=False,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertEqual(argv[0], "/fake/bin/codex")
        self.assertEqual(argv[argv.index("--sandbox") + 1], "workspace-write")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertEqual(env.get("fence"), "vendor")
        self.assertFalse(env["fence_detail"]["read_wall"])

    def test_run_agent_codex_write_wrap_postcondition_failure_no_spawn(self):
        # A wrap that does not take must REFUSE, never fall back to the
        # vendor sandbox (the silent third state the design review found).
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        kernel_sandbox.wrap_argv = lambda argv, *a, **k: list(argv)
        self._real_sandbox_exec_which()
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_unavailable")
        self.assertEqual(recorded["calls"], 0)

    def test_run_agent_codex_write_disable_postcondition_failure_no_spawn(self):
        # Same rule one step later: a proven wrap whose inner-sandbox disable
        # does not hold is a refusal, not a dual-wrapped spawn.
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        self.addCleanup(setattr, kernel_sandbox, "disable_inner_codex_sandbox",
                        kernel_sandbox.disable_inner_codex_sandbox)
        kernel_sandbox.disable_inner_codex_sandbox = lambda argv: None
        self._real_sandbox_exec_which()
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_unavailable")
        self.assertIn("inner vendor sandbox disable", env.get("detail", ""))
        self.assertEqual(recorded["calls"], 0)

    @mock.patch("platform.system", new=lambda: "Darwin")
    def test_run_agent_read_only_wraps_and_disables_inner_sandbox(self):
        # Read-only workers have the same host boundary without cwd writes.
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        recorded = {}

        async def _fake_run_cli(argv, **kwargs):
            recorded["argv"] = list(argv)
            if "-o" in argv:
                path = argv[argv.index("-o") + 1]
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("DONE")
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "timed_out": False}

        agent_lane.run_cli = _fake_run_cli
        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=True,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertTrue(kernel_sandbox.is_wrapped(argv))
        self.assertNotIn("--sandbox", argv)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertEqual(env["fence"], "seatbelt-fs")
        self.assertFalse(env["fence_detail"]["workdir_writable"])

    @mock.patch("platform.system", new=lambda: "Darwin")
    def test_run_agent_cursor_wraps_and_disables_inner_sandbox(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        self._real_sandbox_exec_which()
        recorded = {"calls": 0}

        async def _fake_run_cli(argv, **kwargs):
            recorded["calls"] += 1
            recorded["argv"] = list(argv)
            payload = json.dumps({
                "type": "result", "subtype": "success",
                "result": "DONE", "session_id": "s1",
            })
            return {"returncode": 0, "stdout": payload, "stderr": "",
                    "timed_out": False}

        agent_lane.run_cli = _fake_run_cli
        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="cursor/composer-2.5",
            workdir=self.workdir,
            timeout_s=30,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertTrue(kernel_sandbox.is_wrapped(argv))
        self.assertEqual(argv[argv.index("--sandbox") + 1], "disabled")
        self.assertEqual(argv.count("--sandbox"), 1)
        self.assertEqual(env["fence"], "seatbelt-fs")

    def test_run_agent_codex_write_verify_refusal_refuses_no_spawn(self):
        # The write lane now DOES consult the kernel fence: a verify refusal
        # blocks the spawn instead of being ignored (issue #94 piece 3).
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: {
            "ok": False,
            "reason": "kernel_fence_unverified",
            "detail": "probe failed",
        }
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "kernel_fence_unverified")
        self.assertEqual(recorded["calls"], 0)

    @mock.patch("platform.system", new=lambda: "Darwin")
    def test_run_agent_codex_read_only_verify_refusal_refuses_no_spawn(self):
        # Read-only host verification failure must prevent vendor execution.
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: {
            "ok": False,
            "reason": "kernel_fence_unverified",
            "detail": "probe failed",
        }
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=True,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "kernel_fence_unverified")
        self.assertEqual(recorded["calls"], 0)

    def test_run_agent_codex_write_off_darwin_refuses_without_optin(self):
        # Decision B: off Darwin the write lane refuses rather than resting
        # on a vendor sandbox with two recorded failures.
        kernel_sandbox.available = lambda: False
        self.addCleanup(setattr, kernel_sandbox, "unfenced_fallback_allowed",
                        kernel_sandbox.unfenced_fallback_allowed)
        self.addCleanup(setattr, agent_lane.platform, "system",
                        agent_lane.platform.system)
        agent_lane.platform.system = lambda: "Linux"
        kernel_sandbox.unfenced_fallback_allowed = lambda: False
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_unavailable")
        self.assertEqual(recorded["calls"], 0)

    def test_run_agent_codex_write_off_darwin_optin_spawns_vendor(self):
        # ... and with OMNI_LLM_AGENT_UNFENCED=1 it spawns on the vendor
        # sandbox, stamped `vendor` because that is what argv still carries.
        kernel_sandbox.available = lambda: False
        self.addCleanup(setattr, kernel_sandbox, "unfenced_fallback_allowed",
                        kernel_sandbox.unfenced_fallback_allowed)
        self.addCleanup(setattr, agent_lane.platform, "system",
                        agent_lane.platform.system)
        agent_lane.platform.system = lambda: "Linux"
        kernel_sandbox.unfenced_fallback_allowed = lambda: True
        recorded = self._recorder()

        env = asyncio.run(agent_lane.run_agent(
            task="do it",
            model="codex/gpt-5.6-sol",
            workdir=self.workdir,
            read_only=False,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertEqual(argv[0], "/fake/bin/codex")
        self.assertNotIn("-p", argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "workspace-write")
        self.assertEqual(env.get("fence"), "vendor")

class TestSandboxCanaryNoKernelStandDown(unittest.TestCase):
    """Vendor canary must not stand down when a kernel fence merely verifies.

    Agent no longer wraps (issue #51); the canary gates the fence that is
    actually applied (vendor --sandbox read-only).
    """

    def setUp(self):
        self.old_available = kernel_sandbox.available
        self.old_verify = kernel_sandbox.verify_fence
        self.old_version = sandbox_canary.codex_cli_version
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox, "verify_fence", self.old_verify)
        self.addCleanup(
            setattr, sandbox_canary, "codex_cli_version", self.old_version,
        )

    def test_no_stand_down_when_kernel_fence_verifies(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        refusal = sandbox_canary.gate(True, "codex")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_no_stand_down_when_verify_fence_refuses(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: {
            "ok": False,
            "reason": "kernel_fence_unverified",
            "detail": "probe failed",
        }
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        refusal = sandbox_canary.gate(True, "codex")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "sandbox_breached")


class TestReadWallProfile(unittest.TestCase):
    """Issue #45: read_wall profile clauses and re-allow params."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.workdir = tempfile.mkdtemp(prefix="omni_ksb_rw_")
        self.addCleanup(_rmtree, self.workdir)

    def test_read_wall_false_profile_has_no_file_read_deny(self):
        profile = kernel_sandbox.build_profile(self.workdir, read_only=True)
        self.assertNotIn("file-read", profile)
        params = kernel_sandbox.profile_params(self.workdir, read_only=True)
        self.assertTrue(all(k.startswith("P") or k == "Q0" for k in params), params)
        self.assertEqual(params["Q0"], os.path.realpath(self.workdir))
        self.assertNotIn("H0", params)

    def test_read_wall_true_denies_home_and_reallows_codex(self):
        profile = kernel_sandbox.build_profile(
            self.workdir, read_only=True, read_wall=True,
        )
        i_deny_write = profile.index("(deny file-write*)")
        i_deny_read = profile.index("(deny file-read*")
        i_allow_read = profile.index("(allow file-read*")
        self.assertLess(i_deny_write, i_deny_read)
        self.assertLess(i_deny_read, i_allow_read)
        self.assertIn('(subpath (param "H0"))', profile)

        params = kernel_sandbox.profile_params(
            self.workdir, read_only=True, read_wall=True,
        )
        home = kernel_sandbox.protected_home()
        self.assertEqual(params.get("H0"), home)
        codex = kernel_sandbox.codex_config_dir()
        if os.path.isdir(codex):
            self.assertIn(codex, params.values())
        # Paths never interpolated into profile text.
        for path in params.values():
            self.assertNotIn(path, profile)

    def test_extra_read_paths_under_home_become_literal_params(self):
        home = kernel_sandbox.protected_home()
        canary_dir = tempfile.mkdtemp(prefix=".omni_ksb_extra_", dir=home)
        self.addCleanup(_rmtree, canary_dir)
        canary = os.path.join(canary_dir, "in.txt")
        with open(canary, "w", encoding="utf-8") as fh:
            fh.write("in\n")
        params = kernel_sandbox.profile_params(
            self.workdir, read_only=True, read_wall=True,
            extra_read_paths=[canary],
        )
        self.assertIn(os.path.realpath(canary), params.values())
        profile = kernel_sandbox.build_profile(
            self.workdir, read_only=True, read_wall=True,
            extra_read_paths=[canary],
        )
        self.assertIn("(literal (param ", profile)

    def test_metadata_reallow_covers_exact_ancestor_chain(self):
        """realpath(3) lstats every ancestor; M* must cover exactly that
        chain (home inclusive, leaf exclusive) for a re-allowed path several
        directories deep — and nothing more, nothing less."""
        home = kernel_sandbox.protected_home()
        deep = tempfile.mkdtemp(prefix=".omni_ksb_meta_", dir=home)
        self.addCleanup(_rmtree, deep)
        mid = os.path.join(deep, "a", "b")
        os.makedirs(mid)
        leaf = os.path.join(mid, "c")
        os.makedirs(leaf)
        params = kernel_sandbox.profile_params(
            self.workdir, read_only=True, read_wall=True,
            extra_read_paths=[leaf],
        )
        m_values = {v for k, v in params.items() if k.startswith("M")}
        expected = {
            os.path.realpath(home),
            os.path.realpath(deep),
            os.path.realpath(os.path.join(deep, "a")),
            os.path.realpath(mid),
        }
        # leaf itself must NOT be in the metadata set (it already has full
        # read access from its own R* entry).
        self.assertNotIn(os.path.realpath(leaf), m_values)
        self.assertTrue(expected.issubset(m_values), m_values)
        profile = kernel_sandbox.build_profile(
            self.workdir, read_only=True, read_wall=True,
            extra_read_paths=[leaf],
        )
        self.assertIn("(allow file-read-metadata ", profile)
        i_allow_read = profile.index("(allow file-read*", profile.index("(deny file-read*"))
        i_meta = profile.index("(allow file-read-metadata ")
        self.assertLess(i_allow_read, i_meta,
                        "metadata re-allow must come after the R* re-allow")
        for path in m_values:
            self.assertNotIn(path, profile, "paths must ride via -D params only")

    def test_metadata_reallow_empty_when_no_reallow_entries(self):
        """No R*/S* entries -> no M* clause (bare read wall stays minimal)."""
        empty_home_child = kernel_sandbox.protected_home()
        params = kernel_sandbox.profile_params(
            "/does/not/matter", read_only=True, read_wall=True,
            extra_read_paths=[],
        )
        # Only H0 (and maybe R* for ~/.codex if it exists) are expected;
        # assert the mutation-catching shape directly via the helper.
        entries = kernel_sandbox._metadata_reallow_entries(
            "/does/not/matter", extra_read_paths=[],
        )
        if not kernel_sandbox._read_reallow_entries("/does/not/matter", []):
            self.assertEqual(entries, ())
            self.assertFalse(any(k.startswith("M") for k in params), params)

    def test_h0_is_protected_home_not_inherited_HOME(self):
        """F1: spoofed $HOME must not redefine the deny-set target."""
        decoy = tempfile.mkdtemp(prefix="omni_ksb_fakehome_")
        self.addCleanup(_rmtree, decoy)
        # The product function itself (the offline run swaps in a synthetic
        # home): it reads the passwd database only, so a spoofed $HOME must
        # not change its answer and it opens nothing under the home.
        original = offline_home.original_protected_home
        real = original()
        self.assertEqual(real, offline_home.real_home())
        old = os.environ.get("HOME")
        try:
            os.environ["HOME"] = decoy
            self.assertEqual(original(), real)
            self.assertNotEqual(original(), os.path.realpath(decoy))
            # The profile built under the (possibly synthetic) home follows
            # protected_home(), never the spoofed variable.
            params = kernel_sandbox.profile_params(
                self.workdir, read_only=True, read_wall=True,
            )
            self.assertEqual(params.get("H0"), kernel_sandbox.protected_home())
            self.assertNotEqual(params.get("H0"), os.path.realpath(decoy))
            # expanduser would follow the spoof — pin that we do not.
            self.assertEqual(
                os.path.realpath(os.path.expanduser("~")),
                os.path.realpath(decoy),
            )
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old


class TestCursorFenceGuards(unittest.TestCase):
    """Fix round 3 (C, D): a home-tree binary/config override must never
    turn into a re-allow of home itself or a direct child of it."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.old_bin = os.environ.get(kernel_sandbox.ENV_CURSOR_BIN)
        self.addCleanup(self._restore_bin)

    def _restore_bin(self):
        if self.old_bin is None:
            os.environ.pop(kernel_sandbox.ENV_CURSOR_BIN, None)
        else:
            os.environ[kernel_sandbox.ENV_CURSOR_BIN] = self.old_bin

    def _fake_home(self):
        # Synthetic home: the lane cannot mkdir under the protected home.
        home = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_cursor_home_"))
        self.addCleanup(_rmtree, home)
        return home

    def test_binary_directly_under_home_is_rejected(self):
        home = self._fake_home()
        fake_bin = os.path.join(home, "omni_ksb_cursor_agent_test-%d" % os.getpid())
        with open(fake_bin, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho fake\n")
        os.chmod(fake_bin, 0o755)
        self.addCleanup(os.remove, fake_bin)
        os.environ[kernel_sandbox.ENV_CURSOR_BIN] = fake_bin
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=None, read_only=False,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("direct child of home", policy.get("detail", ""))

    def test_binary_dir_one_level_under_home_is_also_rejected(self):
        # bin_dir itself (not the binary) is a DIRECT CHILD of home — the
        # distinct third guard clause, not just "bin_dir == home".
        home = self._fake_home()
        bin_dir = os.path.join(home, "omni_ksb_cursor_bindir_test-%d" % os.getpid())
        os.mkdir(bin_dir)
        self.addCleanup(_rmtree, bin_dir)
        fake_bin = os.path.join(bin_dir, "cursor-agent")
        with open(fake_bin, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho fake\n")
        os.chmod(fake_bin, 0o755)
        os.environ[kernel_sandbox.ENV_CURSOR_BIN] = fake_bin
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=None, read_only=False,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("direct child of home", policy.get("detail", ""))

    def test_config_dir_directly_under_home_is_rejected(self):
        home = self._fake_home()
        fake_cfg = os.path.join(home, "omni_ksb_cursor_cfg_test-%d" % os.getpid())
        os.mkdir(fake_cfg)
        self.addCleanup(_rmtree, fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("direct child of home", policy.get("detail", ""))

    def test_config_dir_one_level_deeper_still_admitted(self):
        # Job-lane pin: <jobs-root>/<id>/cursor-config lives inside home
        # and is admitted only because jobs-root is a launcher allowlist
        # entry — not because it is "one level deeper".
        home = self._fake_home()
        token = uuid.uuid4().hex
        jobs_root = os.path.join(home, "omni_ksb_jobs_%s" % token)
        os.mkdir(jobs_root)
        deeper = os.path.join(jobs_root, token, "cursor-config")
        os.makedirs(deeper)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=None, read_only=False, config_dir=deeper,
                launcher_roots=(jobs_root,),
            )
        self.assertNotIn("ok", policy, policy)
        self.assertIn(os.path.realpath(deeper), policy["wrap_kw"]["extra_write_paths"])

    def test_config_dir_home_grandchild_without_launcher_root_is_refused(self):
        home = self._fake_home()
        token = uuid.uuid4().hex
        parent = os.path.join(home, token)
        os.mkdir(parent)
        deeper = os.path.join(parent, "cursor-config")
        os.mkdir(deeper)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            rej = kernel_sandbox._caller_config_dir_rejection(
                deeper, label="cursor config_dir",
            )
        self.assertIsNotNone(rej, deeper)
        self.assertIn("inside the home tree", rej)

    def test_config_dir_library_keychains_is_refused(self):
        # Measured hole: real ~/Library/Keychains, no mkdir.
        home = kernel_sandbox.protected_home()
        keychains = os.path.join(home, "Library", "Keychains")
        rej = kernel_sandbox._caller_config_dir_rejection(
            keychains, label="cursor config_dir",
        )
        self.assertIsNotNone(rej, keychains)

    def test_config_dir_dedicated_cursor_dotdir_is_admitted(self):
        home = self._fake_home()
        token = uuid.uuid4().hex
        fake_cfg = os.path.join(home, ".cursor-%s" % token)
        os.mkdir(fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertNotIn("ok", policy, policy)
        self.assertIn(
            os.path.realpath(fake_cfg), policy["wrap_kw"]["extra_write_paths"])

    def test_launcher_root_equal_to_home_or_library_is_refused(self):
        # Real home / ~/Library — both are widening roots; Keychains exists.
        home = kernel_sandbox.protected_home()
        lib = os.path.join(home, "Library")
        keychains = os.path.join(lib, "Keychains")
        self.assertIsNotNone(
            kernel_sandbox._caller_config_dir_rejection(
                keychains, label="cursor config_dir", launcher_roots=(home,),
            ),
            "launcher root equal to home must not admit a home descendant",
        )
        self.assertIsNotNone(
            kernel_sandbox._caller_config_dir_rejection(
                keychains, label="cursor config_dir", launcher_roots=(lib,),
            ),
            "launcher root equal to ~/Library must not admit a descendant",
        )

    def _firmlink_home_or_skip(self):
        home = kernel_sandbox.protected_home()
        firm = "/System/Volumes/Data" + home
        if not os.path.isdir(firm):
            self.skipTest(
                "firmlink spelling of home does not exist on this host: %s"
                % firm)
        return home, firm

    def test_stat_permission_error_on_firmlink_alias_fails_closed(self):
        home, firm = self._firmlink_home_or_skip()
        probe = os.path.join(firm, "Library", "Keychains")
        if not os.path.isdir(probe):
            self.skipTest(
                "firmlink-spelled Keychains does not exist: %s" % probe)
        real_stat = os.stat

        def wrapped(path, *args, **kwargs):
            if os.path.normpath(os.fspath(path)) == os.path.normpath(firm):
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(kernel_sandbox.os, "stat", wrapped):
            rej = kernel_sandbox._caller_config_dir_rejection(
                probe, label="cursor config_dir",
            )
        self.assertIsNotNone(
            rej,
            "permission failure on a firmlink-spelled home ancestor must "
            "fail closed (refused), not fall to lexical admission",
        )

    def test_launcher_root_firmlink_state_dir_not_usable(self):
        _home, firm = self._firmlink_home_or_skip()
        jobs_root = os.path.join(firm, "jobs")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": firm}):
            self.assertFalse(
                kernel_sandbox._launcher_root_usable(
                    os.path.realpath(jobs_root)),
                "OMNI_LLM_STATE_DIR spelled through the firmlink home must "
                "not yield a usable launcher root",
            )

    def test_launcher_root_unreadable_state_dir_not_usable(self):
        fd, parent = tempfile.mkstemp(prefix="omni_ksb_state_parent_")
        os.close(fd)
        self.addCleanup(os.remove, parent)
        state = os.path.join(parent, "omni-state")
        launcher = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_lroot_"))
        self.addCleanup(_rmtree, launcher)
        real_realpath = os.path.realpath
        parent_names = {
            os.path.normpath(parent),
            os.path.normpath(real_realpath(parent)),
        }

        def wrapped(path):
            p = os.path.normpath(os.fspath(path))
            for name in parent_names:
                if p == name or p.startswith(name + os.sep):
                    raise OSError(errno.ENOTDIR, "Not a directory", path)
            return real_realpath(path)

        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}), \
                mock.patch.object(kernel_sandbox.os.path, "realpath", wrapped):
            self.assertFalse(
                kernel_sandbox._launcher_root_usable(launcher),
                "unreadable state dir must not yield a usable launcher root",
            )

    def test_launcher_root_identity_unavailable_on_state_dir_not_usable(self):
        state = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_state_"))
        self.addCleanup(_rmtree, state)
        launcher = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_lroot_"))
        self.addCleanup(_rmtree, launcher)
        orig = kernel_sandbox._same_dir
        raised = {"once": False}

        def wrapped(path, root):
            if os.path.realpath(os.fspath(path)) == state:
                if not raised["once"]:
                    raised["once"] = True
                    raise kernel_sandbox._IdentityUnavailable(root)
            return orig(path, root)

        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}), \
                mock.patch.object(kernel_sandbox, "_same_dir", wrapped):
            self.assertFalse(
                kernel_sandbox._launcher_root_usable(launcher),
                "identity-unavailable state-dir comparison must not yield "
                "a usable launcher root",
            )

    def test_dotcodex_firmlink_spelling_admitted_by_producer_and_fence(self):
        home, firm = self._firmlink_home_or_skip()
        token = uuid.uuid4().hex
        probe = os.path.join(firm, ".codex-%s" % token)
        try:
            os.mkdir(probe)
        except OSError as exc:
            self.skipTest("cannot mkdir under firmlink home: %s" % exc)
        self.addCleanup(_rmtree, probe)
        why = kernel_sandbox.home_tree_rejection(
            os.path.realpath(probe), home, ".codex-")
        self.assertIsNone(why, why)
        rej = kernel_sandbox._caller_config_dir_rejection(
            probe, label="codex config_dir", allow_dotdir_prefix=".codex-",
        )
        self.assertIsNone(rej, rej)

    def test_config_dir_firmlink_identity(self):
        home, firm = self._firmlink_home_or_skip()
        token = uuid.uuid4().hex
        probe = os.path.join(firm, token, "cursor-config")
        try:
            os.makedirs(probe)
        except OSError as exc:
            self.skipTest(
                "firmlink spelling of home cannot host the probe "
                "(%s): %s" % (probe, exc))
        self.addCleanup(_rmtree, os.path.join(firm, token))
        rej = kernel_sandbox._caller_config_dir_rejection(
            probe, label="cursor config_dir",
        )
        self.assertIsNotNone(
            rej,
            "firmlink-spelled home grandchild must be refused by identity",
        )
        self.assertIn("inside the home tree", rej)

    def test_config_dir_case_identity(self):
        # Lexical casefold fallback: a path that does not exist has no
        # inode, so identity walks cannot run. A differently-cased
        # spelling of a home descendant must still be refused.
        home = self._fake_home()
        folded = home.swapcase()
        if folded == home:
            self.skipTest("fake home spelling has no case difference")
        probe = os.path.join(folded, "no-such-%s" % uuid.uuid4().hex)
        self.assertFalse(os.path.lexists(probe), probe)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            rej = kernel_sandbox._caller_config_dir_rejection(
                probe, label="cursor config_dir",
            )
        self.assertIsNotNone(
            rej,
            "non-existent differently-cased spelling of a home descendant "
            "must still be refused via the lexical casefold fallback",
        )

    def test_eperm_on_keychains_plus_launcher_root_is_refused(self):
        home = kernel_sandbox.protected_home()
        keychains = os.path.join(home, "Library", "Keychains")
        if not os.path.isdir(keychains):
            self.skipTest("~/Library/Keychains does not exist on this host")
        launcher = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_lroot_"))
        self.addCleanup(_rmtree, launcher)
        real_stat = os.stat

        def wrapped(path, *args, **kwargs):
            if os.path.normpath(os.fspath(path)) == os.path.normpath(keychains):
                raise PermissionError(errno.EPERM, "Operation not permitted", path)
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(kernel_sandbox.os, "stat", wrapped):
            rej = kernel_sandbox._caller_config_dir_rejection(
                keychains, label="cursor config_dir",
                launcher_roots=(launcher,),
            )
        self.assertIsNotNone(
            rej,
            "EPERM on ~/Library/Keychains must not admit via a launcher root",
        )

    def test_eperm_on_victim_dotdir_parent_is_not_exempt(self):
        victim = self._fake_home()
        probe = os.path.join(victim, ".cursor-evil")
        os.mkdir(probe)
        real_stat = os.stat

        def wrapped(path, *args, **kwargs):
            if os.path.normpath(os.fspath(path)) == os.path.normpath(victim):
                raise PermissionError(errno.EPERM, "Operation not permitted", path)
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=victim), \
                mock.patch.object(kernel_sandbox.os, "stat", wrapped):
            rej = kernel_sandbox._caller_config_dir_rejection(
                probe, label="cursor config_dir",
                allow_dotdir_prefix=".cursor-",
            )
            why = kernel_sandbox.home_tree_rejection(
                os.path.realpath(probe), victim, ".cursor-")
        self.assertIsNotNone(
            rej,
            "EPERM on the parent of <victim>/.cursor-evil must not exempt it",
        )
        self.assertIsNotNone(
            why,
            "home_tree_rejection must not exempt <victim>/.cursor-evil "
            "when the parent cannot be identified",
        )

    def test_stat_error_on_launcher_root_identity_message(self):
        home = self._fake_home()
        cfg = os.path.join(home, "direct-child")
        os.mkdir(cfg)
        launcher = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_lstat_"))
        self.addCleanup(_rmtree, launcher)
        real_stat = os.stat

        def wrapped(path, *args, **kwargs):
            if os.path.normpath(os.fspath(path)) == os.path.normpath(launcher):
                raise PermissionError(errno.EPERM, "Operation not permitted", path)
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=home), \
                mock.patch.object(kernel_sandbox.os, "stat", wrapped):
            rej = kernel_sandbox._caller_config_dir_rejection(
                cfg, label="cursor config_dir",
                launcher_roots=(launcher,),
            )
        self.assertIsNotNone(rej, "stat error on the launcher root must refuse")
        self.assertIn("identity could not be established", rej)
        self.assertNotIn("is a direct child of home", rej)

    def test_config_dir_firmlink_data_volume_spellings_are_refused(self):
        probes = (
            "/System/Volumes/Data/Users",
            "/System/Volumes/Data",
        )
        present = [p for p in probes if os.path.isdir(p)]
        if not present:
            self.skipTest(
                "firmlink data-volume spellings do not exist on this host: "
                "%s" % ", ".join(probes)
            )
        labels = (
            "CURSOR_CONFIG_DIR",
            "CLAUDE_CONFIG_DIR",
            "CODEX_HOME",
        )
        for probe in present:
            for label in labels:
                with self.subTest(probe=probe, label=label):
                    rej = kernel_sandbox._caller_config_dir_rejection(
                        probe, label=label,
                    )
                    self.assertIsNotNone(
                        rej,
                        "%s as %s must be refused" % (probe, label),
                    )
            for provider in ("cursor", "claude", "codex"):
                with self.subTest(probe=probe, provider=provider):
                    policy = kernel_sandbox.agent_fence_policy(
                        provider, workdir=None, read_only=False,
                        config_dir=probe,
                    )
                    self.assertEqual(policy.get("ok"), False, policy)
                    self.assertEqual(
                        policy.get("reason"), "fence_path_rejected", policy)

    def test_cursor_firmlink_users_config_dir_is_refused(self):
        probe = "/System/Volumes/Data/Users"
        if not os.path.isdir(probe):
            self.skipTest(
                "firmlink spelling does not exist on this host: %s" % probe)
        policy = kernel_sandbox.agent_fence_policy(
            "cursor", workdir=None, read_only=False, config_dir=probe,
        )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)


class TestClaudeFenceGuards(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _fake_home(self):
        home = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_claude_home_"))
        self.addCleanup(_rmtree, home)
        return home

    def test_agent_child_env_keeps_profile_config_dir(self):
        kept = kernel_sandbox.agent_child_env(
            {"CLAUDE_CONFIG_DIR": "/profile", "PATH": "/bin"},
            keep_config_dir=True,
        )
        self.assertEqual(kept.get("CLAUDE_CONFIG_DIR"), "/profile")
        stripped = kernel_sandbox.agent_child_env(
            {"CLAUDE_CONFIG_DIR": "/profile", "PATH": "/bin"},
        )
        self.assertNotIn("CLAUDE_CONFIG_DIR", stripped)

    def test_config_dir_directly_under_home_is_rejected(self):
        home = self._fake_home()
        fake_cfg = os.path.join(home, "omni_ksb_claude_cfg_test")
        os.mkdir(fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("direct child of home", policy.get("detail", ""))

    def test_config_dir_dotclaude_under_home_is_admitted(self):
        home = self._fake_home()
        fake_cfg = os.path.join(home, ".claude-omni72")
        os.mkdir(fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertNotIn("ok", policy, policy)
        self.assertIn(
            os.path.realpath(fake_cfg), policy["wrap_kw"]["extra_write_paths"])

    def test_default_dotclaude_under_home_is_rejected(self):
        home = self._fake_home()
        fake_cfg = os.path.join(home, ".claude")
        os.mkdir(fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("default login directory", policy.get("detail", ""))
        self.assertIn("cannot be a profile", policy.get("detail", ""))

    def test_dotclaude_alt_under_home_is_admitted(self):
        home = self._fake_home()
        fake_cfg = os.path.join(home, ".claude-alt")
        os.mkdir(fake_cfg)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=fake_cfg,
            )
        self.assertNotIn("ok", policy, policy)
        self.assertIn(
            os.path.realpath(fake_cfg), policy["wrap_kw"]["extra_write_paths"])

    def test_dotclaudeX_under_home_is_rejected(self):
        home = self._fake_home()
        for name in (".claudeX", ".claude_backup"):
            fake_cfg = os.path.join(home, name)
            os.mkdir(fake_cfg)
            with mock.patch.object(kernel_sandbox, "protected_home",
                                   return_value=home):
                policy = kernel_sandbox.agent_fence_policy(
                    "claude", workdir=None, read_only=False,
                    config_dir=fake_cfg,
                )
            self.assertEqual(policy.get("reason"), "fence_path_rejected", name)

    def test_config_dir_temp_outside_home_is_admitted(self):
        d = tempfile.mkdtemp(prefix="omni_ksb_claude_prof_")
        self.addCleanup(_rmtree, d)
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=None, read_only=False, config_dir=d,
        )
        self.assertNotIn("ok", policy, policy)
        rp = os.path.realpath(d)
        self.assertIn(rp, policy["wrap_kw"]["extra_write_paths"])
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        self.assertNotIn(default_root, policy["wrap_kw"]["extra_read_paths"])

    def test_profile_root_projects_is_deny_write(self):
        d = tempfile.mkdtemp(prefix="omni_ksb_claude_proj_")
        self.addCleanup(_rmtree, d)
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=None, read_only=False, config_dir=d,
        )
        self.assertNotIn("ok", policy, policy)
        projects = os.path.realpath(os.path.join(d, "projects"))
        wrap_kw = policy["wrap_kw"]
        self.assertIn(("subpath", projects), wrap_kw["extra_deny_write"])
        self.assertIn(("literal", projects), wrap_kw["extra_post_write"])

    def test_config_dir_home_itself_is_rejected(self):
        home = kernel_sandbox.protected_home()
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=None, read_only=False, config_dir=home,
        )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_default_session_dir_unchanged_without_config_dir(self):
        wd = tempfile.mkdtemp(prefix="omni_ksb_claude_wd_")
        self.addCleanup(_rmtree, wd)
        session = kernel_sandbox.claude_session_dir(wd)
        self.assertEqual(
            session,
            os.path.join(
                kernel_sandbox.protected_home(),
                ".claude", "projects",
                kernel_sandbox.claude_encoded_workdir(wd),
            ),
        )

    def _deny_read_paths(self, policy):
        return [p for _k, p in policy["wrap_kw"].get("extra_deny_read") or ()]

    def test_default_root_sibling_not_denied(self):
        home = self._fake_home()
        default = os.path.join(home, ".claude")
        os.mkdir(default)
        winner = tempfile.mkdtemp(prefix="omni_ksb_sib_win_")
        self.addCleanup(_rmtree, winner)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=home), \
                mock.patch.dict(os.environ, {
                    "CLAUDE_CONFIG_DIR_MAIN": default,
                }, clear=False):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=winner)
        self.assertNotIn("ok", policy, policy)
        self.assertNotIn(os.path.realpath(default), self._deny_read_paths(policy))

    def test_sibling_inside_winner_is_refused(self):
        # A registered home nested in the winner's own home is another
        # account's tree. It used to be read-denied inside the allowed root;
        # that deny had to win a last-match fight with the root's re-allow,
        # so the layout is refused outright, naming the label only.
        winner = tempfile.mkdtemp(prefix="omni_ksb_sib_win_")
        self.addCleanup(_rmtree, winner)
        inner = os.path.join(winner, "inner")
        os.mkdir(inner)
        with mock.patch.dict(os.environ, {
            "CLAUDE_CONFIG_DIR_INNER": inner,
        }, clear=False):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=winner)
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("CLAUDE_CONFIG_DIR_INNER", policy["detail"])
        self.assertNotIn(os.path.realpath(inner), policy["detail"])

    def test_outside_home_sibling_denied(self):
        winner = tempfile.mkdtemp(prefix="omni_ksb_sib_win_")
        other = tempfile.mkdtemp(prefix="omni_ksb_sib_other_")
        self.addCleanup(_rmtree, winner)
        self.addCleanup(_rmtree, other)
        with mock.patch.dict(os.environ, {
            "CLAUDE_CONFIG_DIR_OTHER": other,
        }, clear=False):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=winner)
        self.assertNotIn("ok", policy, policy)
        self.assertIn(os.path.realpath(other), self._deny_read_paths(policy))

    def test_sibling_containing_winner_is_refused(self):
        # It cannot be denied without punching out the winner and leaving it
        # open exposes the other account: fail closed, naming the label only.
        parent = tempfile.mkdtemp(prefix="omni_ksb_sib_par_")
        self.addCleanup(_rmtree, parent)
        winner = os.path.join(parent, "winner")
        os.mkdir(winner)
        with mock.patch.dict(os.environ, {
            "CLAUDE_CONFIG_DIR_PARENT": parent,
        }, clear=False):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=winner)
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("CLAUDE_CONFIG_DIR_PARENT", policy["detail"])
        self.assertNotIn(os.path.realpath(parent), policy["detail"])


class TestCodexGrokSiblingHomeDeny(unittest.TestCase):
    """Mirror of the Claude sibling-ring rule for CODEX_HOME_<P> / GROK_HOME_<P>."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    PROVIDERS = (("codex", "CODEX_HOME_", ".codex"), ("grok", "GROK_HOME_", ".grok"))

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _policy(self, provider, env, home=None):
        winner = self._dir("omni_ksb_sib_win_")
        env = {key: (value(winner) if callable(value) else value) for key, value in env.items()}
        patches = [mock.patch.dict(os.environ, env, clear=False)]
        if home is not None:
            patches.append(mock.patch.object(kernel_sandbox, "protected_home", return_value=home))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            policy = kernel_sandbox.agent_fence_policy(
                provider, workdir=None, read_only=False, config_dir=winner)
        self.assertNotIn("ok", policy, policy)
        return policy, [p for _k, p in policy["wrap_kw"].get("extra_deny_read") or ()], winner

    def test_default_root_sibling_not_denied(self):
        for provider, prefix, dotdir in self.PROVIDERS:
            with self.subTest(provider=provider):
                home = self._dir("omni_ksb_sib_home_")
                default = os.path.join(home, dotdir)
                os.mkdir(default)
                _policy, denied, _winner = self._policy(provider, {prefix + "MAIN": default}, home)
                # Codex already denies the default login to a profile run; the
                # sibling rule must neither add a second row nor deny it for Grok.
                self.assertEqual(denied.count(os.path.realpath(default)),
                                 1 if provider == "codex" else 0)

    def test_sibling_inside_winner_is_refused(self):
        for provider, prefix, _dotdir in self.PROVIDERS:
            with self.subTest(provider=provider):
                winner = self._dir("omni_ksb_sib_win_")
                inner = os.path.join(winner, "inner")
                os.mkdir(inner)
                with mock.patch.dict(os.environ, {prefix + "INNER": inner}, clear=False):
                    policy = kernel_sandbox.agent_fence_policy(
                        provider, workdir=None, read_only=False, config_dir=winner)
                self.assertEqual(policy.get("ok"), False, policy)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn(prefix + "INNER", policy["detail"])
                self.assertNotIn(os.path.realpath(inner), policy["detail"])

    def test_outside_home_sibling_denied(self):
        for provider, prefix, _dotdir in self.PROVIDERS:
            with self.subTest(provider=provider):
                other = self._dir("omni_ksb_sib_other_")
                _policy, denied, _winner = self._policy(provider, {prefix + "OTHER": other})
                self.assertIn(os.path.realpath(other), denied)

    def test_sibling_containing_winner_is_refused(self):
        for provider, prefix, _dotdir in self.PROVIDERS:
            with self.subTest(provider=provider):
                parent = self._dir("omni_ksb_sib_par_")
                winner = os.path.join(parent, "winner")
                os.mkdir(winner)
                with mock.patch.dict(os.environ, {prefix + "PARENT": parent}, clear=False):
                    policy = kernel_sandbox.agent_fence_policy(
                        provider, workdir=None, read_only=False, config_dir=winner)
                self.assertEqual(policy.get("ok"), False, policy)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn(prefix + "PARENT", policy["detail"])
                self.assertNotIn(os.path.realpath(parent), policy["detail"])


class TestAccountHomeIsolation(unittest.TestCase):
    """Named claude homes, per-run symlinks, and the registry-wide read walls."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    CONFIG_NAMES = (
        ("settings.json", "literal"), ("settings.local.json", "literal"),
        ("hooks", "subpath"), ("plugins", "subpath"), ("commands", "subpath"),
        ("agents", "subpath"), ("skills", "subpath"), ("CLAUDE.md", "literal"),
        (".claude.json", "literal"),
    )

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _claude(self, config_dir=None, workdir=None, read_only=False, home=None):
        patches = []
        if home is not None:
            patches.append(mock.patch.object(kernel_sandbox, "protected_home",
                                             return_value=home))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return kernel_sandbox.agent_fence_policy(
                "claude", workdir=workdir, read_only=read_only, config_dir=config_dir)

    def test_named_claude_home_denies_config_and_hook_writes(self):
        profile = self._dir("omni_ksb_acct_")
        policy = self._claude(profile)
        self.assertNotIn("ok", policy, policy)
        deny = set(policy["wrap_kw"]["extra_deny_write"])
        for name, kind in self.CONFIG_NAMES:
            self.assertIn((kind, os.path.join(profile, name)), deny, name)
        # The profile root itself stays writable for the vendor's own state.
        self.assertIn(profile, [str(p) for p in policy["wrap_kw"]["extra_write_paths"]])

    def test_named_claude_home_symlinked_run_leaf_is_refused(self):
        for leaf in ("todos", "shell-snapshots", "debug"):
            with self.subTest(leaf=leaf):
                profile = self._dir("omni_ksb_acct_")
                other = self._dir("omni_ksb_acct_other_")
                os.symlink(other, os.path.join(profile, leaf))
                policy = self._claude(profile)
                self.assertEqual(policy.get("ok"), False, policy)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn("symlink", policy["detail"])
                self.assertNotIn(other, policy["detail"])

    def test_symlinked_run_leaf_is_ignored_for_read_only_lane_but_session_is_not(self):
        # Read-only never emits the todos/debug leaves, so only the session
        # path (and projects/) can be refused there.
        profile = self._dir("omni_ksb_acct_")
        other = self._dir("omni_ksb_acct_other_")
        wd = self._dir("omni_ksb_acct_wd_")
        os.symlink(other, os.path.join(profile, "todos"))
        self.assertNotIn("ok", self._claude(profile, workdir=wd, read_only=True))
        os.mkdir(os.path.join(profile, "projects"))
        os.symlink(other, kernel_sandbox.claude_session_dir(wd, config_dir=profile))
        policy = self._claude(profile, workdir=wd, read_only=True)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_symlinked_projects_dir_is_refused_and_never_resolved(self):
        profile = self._dir("omni_ksb_acct_")
        other = self._dir("omni_ksb_acct_other_")
        os.symlink(other, os.path.join(profile, "projects"))
        policy = self._claude(profile)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_ordinary_run_leaves_are_allowed_lexically(self):
        profile = self._dir("omni_ksb_acct_")
        wd = self._dir("omni_ksb_acct_wd_")
        for leaf in ("todos", "shell-snapshots", "debug", "projects"):
            os.mkdir(os.path.join(profile, leaf))
        policy = self._claude(profile, workdir=wd)
        self.assertNotIn("ok", policy, policy)
        post = {path for _kind, path in policy["wrap_kw"]["extra_post_write"]}
        for leaf in ("todos", "shell-snapshots", "debug"):
            self.assertIn(os.path.join(profile, leaf), post, leaf)
        self.assertIn(kernel_sandbox.claude_session_dir(wd, config_dir=profile), post)

    def test_default_home_symlinked_run_leaf_is_refused_too(self):
        home = self._dir("omni_ksb_acct_home_")
        other = self._dir("omni_ksb_acct_other_")
        default = os.path.join(home, ".claude")
        os.mkdir(default)
        os.symlink(other, os.path.join(default, "todos"))
        policy = self._claude(home=home)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def _denied(self, provider, winner, env, home=None):
        patches = [mock.patch.dict(os.environ, env, clear=False)]
        if home is not None:
            patches.append(mock.patch.object(kernel_sandbox, "protected_home",
                                             return_value=home))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            policy = kernel_sandbox.agent_fence_policy(
                provider, workdir=None, read_only=False, config_dir=winner)
        self.assertNotIn("ok", policy, policy)
        return [p for _k, p in policy["wrap_kw"].get("extra_deny_read") or ()]

    def test_every_provider_registry_is_denied_to_every_child(self):
        for provider in ("claude", "codex", "grok", "pi", "cursor"):
            with self.subTest(provider=provider):
                winner = self._dir("omni_ksb_acct_win_")
                homes = {key: self._dir("omni_ksb_acct_") for key in
                         ("CLAUDE_CONFIG_DIR_B", "CODEX_HOME_B", "GROK_HOME_B")}
                own = {"claude": "CLAUDE_CONFIG_DIR_A", "codex": "CODEX_HOME_A",
                       "grok": "GROK_HOME_A"}.get(provider)
                env = dict(homes)
                if own:
                    env[own] = winner
                denied = self._denied(provider, winner if own else None, env)
                for key, path in homes.items():
                    self.assertIn(path, denied, key)
                self.assertNotIn(winner, denied)

    def test_winner_registered_under_other_label_is_not_denied(self):
        winner = self._dir("omni_ksb_acct_win_")
        denied = self._denied("claude", winner, {"CLAUDE_CONFIG_DIR_ALIAS": winner})
        self.assertNotIn(winner, denied)

    def test_other_provider_default_outside_home_is_denied(self):
        home = self._dir("omni_ksb_acct_home_")
        outside = self._dir("omni_ksb_acct_outside_")
        os.symlink(outside, os.path.join(home, ".grok"))
        winner = self._dir("omni_ksb_acct_win_")
        denied = self._denied("claude", winner, {"CLAUDE_CONFIG_DIR_A": winner}, home=home)
        self.assertIn(outside, denied)

    def test_only_the_running_providers_default_is_exempt(self):
        home = self._dir("omni_ksb_acct_home_")
        env = {"CLAUDE_CONFIG_DIR_MAIN": os.path.join(home, ".claude"),
               "GROK_HOME_MAIN": os.path.join(home, ".grok")}
        with mock.patch.dict(os.environ, env, clear=False):
            claude = [p for _k, p in kernel_sandbox._sibling_home_denies("claude", None, home)]
            codex = [p for _k, p in kernel_sandbox._sibling_home_denies("codex", None, home)]
        self.assertNotIn(os.path.join(home, ".claude"), claude)
        self.assertIn(os.path.join(home, ".grok"), claude)
        self.assertIn(os.path.join(home, ".claude"), codex)
        self.assertIn(os.path.join(home, ".grok"), codex)

    def test_disjoint_registry_is_unchanged(self):
        winner = self._dir("omni_ksb_acct_win_")
        other = self._dir("omni_ksb_acct_other_")
        denied = self._denied("claude", winner, {"CLAUDE_CONFIG_DIR_O": other})
        self.assertEqual(denied.count(other), 1)


@unittest.skipUnless(kernel_sandbox.available(), "sandbox-exec not available")
class TestAccountHomeIsolationLive(unittest.TestCase):
    """Real sandbox-exec: the account-isolation rows hold at the kernel."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _run(self, kw, cwd, read_only, script):
        argv = kernel_sandbox.wrap_argv(
            ["/bin/sh", "-c", script], cwd, read_only, **kw)
        return subprocess.run(argv, capture_output=True, text=True, timeout=10, cwd=cwd)

    def test_named_claude_home_config_is_unwritable_but_state_is_writable(self):
        profile = self._dir("omni_ksb_live_acct_")
        other = self._dir("omni_ksb_live_other_")
        wd = self._dir("omni_ksb_live_wd_")
        for leaf in ("todos", "debug", "shell-snapshots"):
            os.mkdir(os.path.join(profile, leaf))
        with open(os.path.join(other, "secret"), "w", encoding="utf-8") as fh:
            fh.write("OTHER_ACCOUNT\n")
        env = {"GROK_HOME_B": other}
        with mock.patch.dict(os.environ, env, clear=False):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=wd, read_only=False, config_dir=profile)
        self.assertNotIn("ok", policy, policy)
        kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        ok = self._run(kw, wd, False,
                       "echo x > %s/todos/t && echo x > %s/debug/d" % (profile, profile))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        for name in ("settings.json", ".claude.json", "hooks/h.sh", "CLAUDE.md"):
            target = os.path.join(profile, name)
            os.makedirs(os.path.dirname(target), exist_ok=True) if "/" in name else None
            bad = self._run(kw, wd, False, "echo x > %s" % target)
            self.assertNotEqual(bad.returncode, 0, name)
            self.assertIn("Operation not permitted", bad.stderr, name)
        read = self._run(kw, wd, False, "cat %s/secret" % other)
        self.assertNotEqual(read.returncode, 0)
        self.assertNotIn("OTHER_ACCOUNT", read.stdout)

    def test_named_codex_consult_cannot_touch_default_home_or_own_config(self):
        home = self._dir("omni_ksb_live_home_")
        default = os.path.join(home, ".codex")
        os.mkdir(default)
        with open(os.path.join(default, "auth.json"), "w", encoding="utf-8") as fh:
            fh.write("DEFAULT_ACCOUNT\n")
        profile = self._dir("omni_ksb_live_codex_")
        cwd = self._dir("omni_ksb_live_cwd_")
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            ap = kernel_sandbox.admit_path(profile, "subpath", label="CODEX_HOME")
            rows = kernel_sandbox.codex_consult_account_fence(ap.path, cwd)
            kw = dict(read_wall=True, extra_read_paths=[ap],
                      extra_write_paths=[ap], **rows)
            ok = self._run(kw, cwd, True,
                           "mkdir -p %s/sessions/2026 && echo x > %s/sessions/2026/r.jsonl"
                           % (profile, profile))
            self.assertEqual(ok.returncode, 0, ok.stderr)
            for target in ("config.toml", "hooks.json", "AGENTS.md", "skills/s.md"):
                path = os.path.join(profile, target)
                if "/" in target:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                bad = self._run(kw, cwd, True, "echo x > %s" % path)
                self.assertNotEqual(bad.returncode, 0, target)
            write = self._run(kw, cwd, True, "echo x > %s/planted" % default)
            self.assertNotEqual(write.returncode, 0)
            read = self._run(kw, cwd, True, "cat %s/auth.json" % default)
            self.assertNotEqual(read.returncode, 0)
            self.assertNotIn("DEFAULT_ACCOUNT", read.stdout)
        self.assertFalse(os.path.exists(os.path.join(default, "planted")))


class TestFenceOrderingFinalRows(unittest.TestCase):
    """Seatbelt is last-match-wins: protected-config and sibling-home denies
    are FINAL rows, and a protected name that aliases is refused up front."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _policy(self, provider, config_dir=None, workdir=None, read_only=False,
                home=None, env=None):
        with contextlib.ExitStack() as stack:
            if home is not None:
                stack.enter_context(mock.patch.object(
                    kernel_sandbox, "protected_home", return_value=home))
            stack.enter_context(mock.patch.dict(os.environ, env or {}, clear=False))
            return kernel_sandbox.agent_fence_policy(
                provider, workdir=workdir, read_only=read_only,
                config_dir=config_dir)

    def _assert_all_after_allows(self, profile, params, markers):
        """Every *marker* param's deny appears after the last allow row."""
        last_allow = max(profile.rfind(tok) for tok in (
            "(allow file-write*", "(allow file-read*",
            "(allow file-read-metadata"))
        self.assertGreater(last_allow, 0)
        for key, path in markers.items():
            self.assertIn(key, params)
            self.assertEqual(params[key], path)
            self.assertGreater(profile.index('(param "%s")' % key), last_allow, key)

    # --- build_profile row order -------------------------------------
    def test_final_read_clause_follows_every_read_allow(self):
        wd = self._dir("omni_ksb_fin_wd_")
        kw = dict(
            read_wall=True,
            extra_read_paths=(), extra_post_read=(("subpath", "/private/tmp/s"),),
            extra_deny_read=(("subpath", "/private/tmp/s/inner"),),
            extra_deny_read_final=(("subpath", "/private/tmp/s/inner"),),
            extra_post_write=(("subpath", "/private/tmp/s/w"),),
            extra_deny_write_final=(("literal", "/private/tmp/s/w/cfg"),),
        )
        profile = kernel_sandbox.build_profile(wd, False, **kw)
        params = kernel_sandbox.profile_params(wd, False, **kw)
        self._assert_all_after_allows(
            profile, params,
            {"Y0": "/private/tmp/s/inner", "F0": "/private/tmp/s/w/cfg"})
        self.assertNotIn("/private/tmp/s", profile)
        bare = kernel_sandbox.build_profile(wd, False, read_wall=True)
        self.assertNotIn('"Y0"', bare)
        self.assertNotIn("Y0", kernel_sandbox.profile_params(wd, False, read_wall=True))

    def test_named_claude_policy_rows_are_final_after_all_allows(self):
        profile_home = self._dir("omni_ksb_fin_claude_")
        wd = self._dir("omni_ksb_fin_wd_")
        other = self._dir("omni_ksb_fin_other_")
        for leaf in ("todos", "debug", "shell-snapshots"):
            os.mkdir(os.path.join(profile_home, leaf))
        policy = self._policy("claude", profile_home, wd, env={"GROK_HOME_B": other})
        self.assertNotIn("ok", policy, policy)
        kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        text = kernel_sandbox.build_profile(wd, False, **kw)
        params = kernel_sandbox.profile_params(wd, False, **kw)
        finals = {"F%d" % i: p for i, (_k, p) in enumerate(kw["extra_deny_write_final"])}
        reads = {"Y%d" % i: p for i, (_k, p) in enumerate(kw["extra_deny_read_final"])}
        self.assertIn(os.path.join(profile_home, "settings.json"), finals.values())
        self.assertIn(os.path.join(profile_home, "hooks"), finals.values())
        self.assertIn(other, reads.values())
        self._assert_all_after_allows(text, params, {**finals, **reads})
        # The pre-existing W / X rows are untouched (never weakened).
        self.assertIn(("literal", os.path.join(profile_home, "settings.json")),
                      kw["extra_deny_write"])
        self.assertIn(("subpath", other), kw["extra_deny_read"])

    def test_named_codex_consult_rows_are_final_after_all_allows(self):
        home = self._dir("omni_ksb_fin_home_")
        os.mkdir(os.path.join(home, ".codex"))
        profile_home = self._dir("omni_ksb_fin_codex_")
        other = self._dir("omni_ksb_fin_other_")
        cwd = self._dir("omni_ksb_fin_cwd_")
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_B": other}, clear=False):
            ap = kernel_sandbox.admit_path(profile_home, "subpath", label="CODEX_HOME")
            rows = kernel_sandbox.codex_consult_account_fence(ap.path, cwd)
            kw = dict(read_wall=True, extra_read_paths=[ap],
                      extra_write_paths=[ap], **rows)
            text = kernel_sandbox.build_profile(cwd, True, **kw)
            params = kernel_sandbox.profile_params(cwd, True, **kw)
        finals = {"F%d" % i: p for i, (_k, p) in enumerate(rows["extra_deny_write_final"])}
        reads = {"Y%d" % i: p for i, (_k, p) in enumerate(rows["extra_deny_read_final"])}
        for name in ("config.toml", "hooks.json", "AGENTS.md", "AGENTS.override.md",
                     "agents", "skills", "plugins", "rules"):
            self.assertIn(os.path.join(profile_home, name), finals.values(), name)
        self.assertIn(other, reads.values())
        self._assert_all_after_allows(text, params, {**finals, **reads})

    # --- symlinked protected names are refused ------------------------
    def test_codex_named_home_config_symlink_into_state_is_refused(self):
        for name in ("config.toml", "hooks.json", "AGENTS.md", "AGENTS.override.md",
                     "agents", "skills", "plugins", "rules"):
            with self.subTest(name=name):
                home = self._dir("omni_ksb_sym_home_")
                profile_home = self._dir("omni_ksb_sym_codex_")
                cwd = self._dir("omni_ksb_sym_cwd_")
                os.makedirs(os.path.join(profile_home, "sessions"))
                with open(os.path.join(profile_home, "sessions", "payload"), "w") as fh:
                    fh.write("x")
                os.symlink(os.path.join("sessions", "payload"),
                           os.path.join(profile_home, name))
                with mock.patch.object(kernel_sandbox, "protected_home",
                                       return_value=home):
                    ap = kernel_sandbox.admit_path(
                        profile_home, "subpath", label="CODEX_HOME")
                    with self.assertRaises(kernel_sandbox.FencePathRejected) as ctx:
                        kernel_sandbox.codex_consult_account_fence(ap.path, cwd)
                self.assertIn(name, ctx.exception.detail)
                self.assertIn("named codex home", ctx.exception.detail)
                self.assertNotIn(profile_home, ctx.exception.detail)

    def test_claude_named_home_config_symlink_into_state_is_refused(self):
        for name in ("settings.json", "settings.local.json", "hooks", "plugins",
                     "commands", "agents", "skills", "CLAUDE.md", ".claude.json"):
            with self.subTest(name=name):
                profile_home = self._dir("omni_ksb_sym_claude_")
                wd = self._dir("omni_ksb_sym_wd_")
                os.makedirs(os.path.join(profile_home, "debug"))
                with open(os.path.join(profile_home, "debug", "payload"), "w") as fh:
                    fh.write("x")
                os.symlink(os.path.join("debug", "payload"),
                           os.path.join(profile_home, name))
                policy = self._policy("claude", profile_home, wd)
                self.assertEqual(policy.get("ok"), False, policy)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn(name, policy["detail"])
                if name != ".claude.json":
                    # (the older literal-admission refusal for .claude.json
                    # echoes its path; the new one never does)
                    self.assertNotIn(profile_home, policy["detail"])

    def test_named_home_refuses_a_symlink_even_to_a_harmless_place(self):
        profile_home = self._dir("omni_ksb_sym_claude_")
        elsewhere = self._dir("omni_ksb_sym_else_")
        os.symlink(elsewhere, os.path.join(profile_home, "skills"))
        policy = self._policy("claude", profile_home, self._dir("omni_ksb_sym_wd_"))
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_default_home_symlink_into_state_or_writable_root_is_refused(self):
        for provider, rel, name in (
                ("claude", os.path.join("debug", "payload"), "settings.json"),
                ("codex", os.path.join("sessions", "payload"), "config.toml")):
            with self.subTest(provider=provider):
                home = self._dir("omni_ksb_sym_dhome_")
                root = os.path.join(home, ".%s" % provider)
                os.makedirs(os.path.dirname(os.path.join(root, rel)))
                with open(os.path.join(root, rel), "w") as fh:
                    fh.write("x")
                os.symlink(os.path.join(root, rel), os.path.join(root, name))
                policy = self._policy(provider, None, self._dir("omni_ksb_sym_wd_"),
                                      home=home)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn(name, policy["detail"])
        # A target inside the (writable) workdir is refused as well.
        home = self._dir("omni_ksb_sym_dhome_")
        wd = self._dir("omni_ksb_sym_wd_")
        root = os.path.join(home, ".claude")
        os.mkdir(root)
        target = os.path.join(wd, "CLAUDE.md")
        with open(target, "w") as fh:
            fh.write("x")
        os.symlink(target, os.path.join(root, "CLAUDE.md"))
        policy = self._policy("claude", None, wd, home=home)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_default_home_dotfile_symlink_to_unwritable_place_still_works(self):
        # The operator's own CLAUDE.md / AGENTS.md are usually links into a
        # dotfiles tree the child cannot write. That stays usable.
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outside = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_dot_", dir=repo))
        self.addCleanup(_rmtree, outside)
        for provider, name in (("claude", "CLAUDE.md"), ("codex", "AGENTS.md")):
            with self.subTest(provider=provider):
                home = self._dir("omni_ksb_sym_dhome_")
                root = os.path.join(home, ".%s" % provider)
                os.mkdir(root)
                target = os.path.join(outside, provider + "-" + name)
                with open(target, "w") as fh:
                    fh.write("x")
                os.symlink(target, os.path.join(root, name))
                policy = self._policy(provider, None, self._dir("omni_ksb_sym_wd_"),
                                      home=home)
                self.assertNotIn("ok", policy, policy)
                finals = {p for _k, p in policy["wrap_kw"]["extra_deny_write_final"]}
                self.assertIn(target, finals)
                self.assertIn(os.path.join(root, name), finals)

    def test_ordinary_state_and_disjoint_homes_still_work(self):
        profile_home = self._dir("omni_ksb_ok_claude_")
        wd = self._dir("omni_ksb_ok_wd_")
        other = self._dir("omni_ksb_ok_other_")
        for leaf in ("todos", "debug", "shell-snapshots", "projects"):
            os.mkdir(os.path.join(profile_home, leaf))
        with open(os.path.join(profile_home, "settings.json"), "w") as fh:
            fh.write("{}")
        policy = self._policy("claude", profile_home, wd,
                              env={"CODEX_HOME_B": other})
        self.assertNotIn("ok", policy, policy)
        post = {p for _k, p in policy["wrap_kw"]["extra_post_write"]}
        for leaf in ("todos", "debug", "shell-snapshots"):
            self.assertIn(os.path.join(profile_home, leaf), post)

    # --- cursor config dir nesting -------------------------------------
    def test_cursor_config_dir_containing_a_sibling_home_is_refused(self):
        cursor_dir = self._dir("omni_ksb_cur_")
        inner = os.path.join(cursor_dir, "grok-b")
        os.mkdir(inner)
        wd = self._dir("omni_ksb_cur_wd_")
        policy = self._policy("cursor", cursor_dir, wd, env={"GROK_HOME_B": inner})
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("GROK_HOME_B", policy["detail"])
        self.assertNotIn(inner, policy["detail"])

    def test_cursor_config_dir_equal_to_a_sibling_home_is_refused(self):
        cursor_dir = self._dir("omni_ksb_cur_")
        wd = self._dir("omni_ksb_cur_wd_")
        policy = self._policy("cursor", cursor_dir, wd,
                              env={"CODEX_HOME_B": cursor_dir})
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("CODEX_HOME_B", policy["detail"])

    def test_cursor_disjoint_sibling_is_a_final_read_deny(self):
        cursor_dir = self._dir("omni_ksb_cur_")
        other = self._dir("omni_ksb_cur_other_")
        wd = self._dir("omni_ksb_cur_wd_")
        policy = self._policy("cursor", cursor_dir, wd, env={"GROK_HOME_B": other})
        self.assertNotIn("ok", policy, policy)
        self.assertIn(("subpath", other), policy["wrap_kw"]["extra_deny_read_final"])
        kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        text = kernel_sandbox.build_profile(wd, False, **kw)
        params = kernel_sandbox.profile_params(wd, False, **kw)
        reads = {"Y%d" % i: p for i, (_k, p) in enumerate(kw["extra_deny_read_final"])}
        self._assert_all_after_allows(text, params, reads)

    def test_pi_and_default_roots_refuse_a_nested_sibling(self):
        home = self._dir("omni_ksb_pi_home_")
        os.mkdir(os.path.join(home, ".pi"))
        inner = os.path.join(home, ".pi", "grok-b")
        os.mkdir(inner)
        policy = self._policy("pi", None, self._dir("omni_ksb_pi_wd_"), home=home,
                              env={"GROK_HOME_B": inner})
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("GROK_HOME_B", policy["detail"])


@unittest.skipUnless(kernel_sandbox.available(), "sandbox-exec not available")
class TestFenceOrderingLive(unittest.TestCase):
    """Real sandbox-exec: a later allow cannot reopen a final deny."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _run(self, kw, cwd, read_only, script):
        argv = kernel_sandbox.wrap_argv(
            ["/bin/sh", "-c", script], cwd, read_only, **kw)
        return subprocess.run(argv, capture_output=True, text=True, timeout=10, cwd=cwd)

    def test_claude_settings_alias_through_debug_stays_unwritable(self):
        profile_home = self._dir("omni_ksb_lf_claude_")
        wd = self._dir("omni_ksb_lf_wd_")
        os.mkdir(os.path.join(profile_home, "debug"))
        payload = os.path.join(profile_home, "debug", "payload")
        with open(payload, "w", encoding="utf-8") as fh:
            fh.write("ORIGINAL\n")
        os.symlink(os.path.join("debug", "payload"),
                   os.path.join(profile_home, "settings.json"))
        # The spawn is refused up front; bypass that gate to prove the rows
        # alone hold at the kernel (defence in depth).
        with mock.patch.object(kernel_sandbox, "_protected_config_refusal",
                               return_value=None):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=wd, read_only=False, config_dir=profile_home)
        self.assertNotIn("ok", policy, policy)
        kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        # Control: with the final rows stripped the old ordering lets the
        # alias be written, so this test discriminates.
        stripped = dict(kw, extra_deny_write_final=())
        leak = self._run(stripped, wd, False,
                         "echo LEAK > %s/settings.json" % profile_home)
        self.assertEqual(leak.returncode, 0, leak.stderr)
        with open(payload, "w", encoding="utf-8") as fh:
            fh.write("ORIGINAL\n")
        bad = self._run(kw, wd, False, "echo PWNED > %s/settings.json" % profile_home)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("Operation not permitted", bad.stderr)
        direct = self._run(kw, wd, False, "echo PWNED > %s" % payload)
        self.assertNotEqual(direct.returncode, 0)
        with open(payload, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "ORIGINAL\n")
        ok = self._run(kw, wd, False, "echo x > %s/debug/other" % profile_home)
        self.assertEqual(ok.returncode, 0, ok.stderr)

    def test_codex_named_config_alias_through_sessions_stays_unwritable(self):
        home = self._dir("omni_ksb_lf_home_")
        profile_home = self._dir("omni_ksb_lf_codex_")
        cwd = self._dir("omni_ksb_lf_cwd_")
        os.mkdir(os.path.join(profile_home, "sessions"))
        payload = os.path.join(profile_home, "sessions", "payload")
        with open(payload, "w", encoding="utf-8") as fh:
            fh.write("ORIGINAL\n")
        os.symlink(os.path.join("sessions", "payload"),
                   os.path.join(profile_home, "config.toml"))
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home), \
                mock.patch.object(kernel_sandbox, "_protected_config_refusal",
                                  return_value=None):
            ap = kernel_sandbox.admit_path(profile_home, "subpath", label="CODEX_HOME")
            rows = kernel_sandbox.codex_consult_account_fence(ap.path, cwd)
            kw = dict(read_wall=True, extra_read_paths=[ap],
                      extra_write_paths=[ap], **rows)
            stripped = dict(kw, extra_deny_write_final=())
            leak = self._run(stripped, cwd, True,
                             "echo LEAK > %s/config.toml" % profile_home)
            self.assertEqual(leak.returncode, 0, leak.stderr)
            with open(payload, "w", encoding="utf-8") as fh:
                fh.write("ORIGINAL\n")
            bad = self._run(kw, cwd, True, "echo PWNED > %s/config.toml" % profile_home)
            self.assertNotEqual(bad.returncode, 0)
            with open(payload, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "ORIGINAL\n")
            ok = self._run(kw, cwd, True, "echo x > %s/sessions/new" % profile_home)
            self.assertEqual(ok.returncode, 0, ok.stderr)

    def test_sibling_home_nested_in_allowed_root_is_unreadable(self):
        home = self._dir("omni_ksb_lf_rhome_")
        root = os.path.join(home, "cursor-root")
        inner = os.path.join(root, "grok-b")
        os.makedirs(inner)
        with open(os.path.join(inner, "secret"), "w", encoding="utf-8") as fh:
            fh.write("SIBLING_ACCOUNT\n")
        with open(os.path.join(root, "own"), "w", encoding="utf-8") as fh:
            fh.write("OWN_STATE\n")
        cwd = self._dir("omni_ksb_lf_rcwd_")
        rows = (("subpath", inner),)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            base = dict(read_wall=True, extra_post_read=(("subpath", root),),
                        extra_deny_read=rows)
            # Control: the X deny alone is reopened by the root's S re-allow.
            leak = self._run(base, cwd, True, "cat %s/secret" % inner)
            self.assertEqual(leak.returncode, 0, leak.stderr)
            self.assertIn("SIBLING_ACCOUNT", leak.stdout)
            kw = dict(base, extra_deny_read_final=rows)
            bad = self._run(kw, cwd, True, "cat %s/secret" % inner)
            self.assertNotEqual(bad.returncode, 0)
            self.assertNotIn("SIBLING_ACCOUNT", bad.stdout)
            own = self._run(kw, cwd, True, "cat %s/own" % root)
            self.assertEqual(own.returncode, 0, own.stderr)
            self.assertIn("OWN_STATE", own.stdout)


@unittest.skipUnless(kernel_sandbox.available(), "sandbox-exec not available")
class TestApiSettingsDenyLive(unittest.TestCase):
    """Real sandbox-exec: a final-position deny pins the generated settings file."""

    def setUp(self):
        self.wd = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_apiset_"))
        self.addCleanup(_rmtree, self.wd)
        self.dir = os.path.join(self.wd, "settings")
        os.mkdir(self.dir)
        self.path = os.path.join(self.dir, "api-settings.json")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("ORIGINAL")
        rows = (kernel_sandbox._deny_entries("literal", self.path)
                + kernel_sandbox._deny_entries("subpath", self.dir))
        self.kw = {"extra_deny_write_final": tuple(rows)}

    def _run(self, script):
        argv = kernel_sandbox.wrap_argv(
            ["/bin/sh", "-c", script], self.wd, False, **self.kw)
        return subprocess.run(argv, capture_output=True, text=True, timeout=10, cwd=self.wd)

    def test_content_unlink_and_rename_are_all_refused(self):
        spare = os.path.join(self.wd, "spare.json")
        with open(spare, "w", encoding="utf-8") as fh:
            fh.write("PLANTED")
        for name, script in (
                ("overwrite", "echo PLANTED > '%s'" % self.path),
                ("append", "echo PLANTED >> '%s'" % self.path),
                ("unlink", "rm -f '%s'" % self.path),
                ("rename over", "mv -f '%s' '%s'" % (spare, self.path)),
                ("rename away", "mv -f '%s' '%s.moved'" % (self.path, self.path)),
                ("rename directory", "mv '%s' '%s.moved'" % (self.dir, self.dir)),
                ("sibling", "echo x > '%s/other'" % self.dir)):
            with self.subTest(name):
                res = self._run(script)
                self.assertNotEqual(res.returncode, 0, name)
                self.assertIn("not permitted", res.stderr, name)
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "ORIGINAL")
        self.assertEqual(os.listdir(self.dir), ["api-settings.json"])

    def test_the_rest_of_the_workdir_stays_writable_and_the_file_readable(self):
        res = self._run("echo ok > '%s/work.txt' && cat '%s'" % (self.wd, self.path))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip().endswith("ORIGINAL"), True)


class TestReadWallLiveProbe(unittest.TestCase):
    """Deterministic live probe: fail-before (unwrapped) / pass-after (fenced)."""

    def test_live_write_probe_refuses_when_default_home_deny_is_removed(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        original_wrap = kernel_sandbox.wrap_argv
        observed = {"profiles": [], "commands": [], "target": None}
        original_mkstemp = kernel_sandbox.tempfile.mkstemp

        def _tracked_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            observed["target"] = path
            return fd, path

        def _without_default_deny(argv, workdir, read_only, **kwargs):
            wrapped = original_wrap(argv, workdir, read_only, **kwargs)
            profile_index = wrapped.index("-p") + 1
            profile = wrapped[profile_index]
            self.assertIn("(deny file-write*)", profile)
            self.assertFalse(kwargs.get("extra_deny_write"))
            profile = profile.replace("(deny file-write*)", "", 1)
            observed["profiles"].append(profile)
            wrapped[profile_index] = profile
            observed["commands"].append(list(wrapped))
            return wrapped

        with mock.patch.object(kernel_sandbox, "wrap_argv", _without_default_deny), \
                mock.patch.object(kernel_sandbox.tempfile, "mkstemp",
                                  _tracked_mkstemp):
            refusal = kernel_sandbox.verify_fence(None, read_only=True)

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("fence did not deny", refusal["detail"])
        self.assertTrue(observed["target"])
        self.assertTrue(observed["profiles"])
        self.assertTrue(all(observed["target"] not in profile
                            for profile in observed["profiles"]))
        self.assertTrue(all(observed["target"] not in argv
                            for argv in observed["commands"]))

    def test_live_readonly_probe_refuses_when_q0_deny_is_removed(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        cwd = tempfile.mkdtemp(prefix="omni_ksb_q0_mutation_cwd_")
        self.addCleanup(_rmtree, cwd)
        original_wrap = kernel_sandbox.wrap_argv
        original_mkstemp = kernel_sandbox.tempfile.mkstemp
        observed = {"profiles": [], "commands": [], "target": None}

        def _tracked_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            observed["target"] = path
            return fd, path

        def _without_q0_deny(argv, workdir, read_only, **kwargs):
            wrapped = original_wrap(argv, workdir, read_only, **kwargs)
            profile_index = wrapped.index("-p") + 1
            profile = wrapped[profile_index]
            q0_deny = '(deny file-write* (subpath (param "Q0")))'
            self.assertIn(q0_deny, profile)
            self.assertFalse(kwargs.get("extra_deny_write"))
            profile = profile.replace(q0_deny, "", 1)
            observed["profiles"].append(profile)
            wrapped[profile_index] = profile
            observed["commands"].append(list(wrapped))
            return wrapped

        with mock.patch.object(kernel_sandbox, "wrap_argv", _without_q0_deny), \
                mock.patch.object(kernel_sandbox.tempfile, "mkstemp",
                                  _tracked_mkstemp):
            refusal = kernel_sandbox.verify_fence(cwd, read_only=True)

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("fence did not deny", refusal["detail"])
        self.assertTrue(kernel_sandbox._path_inside(observed["target"], cwd))
        self.assertTrue(all(observed["target"] not in profile
                            for profile in observed["profiles"]))
        self.assertTrue(all(observed["target"] not in argv
                            for argv in observed["commands"]))

    def test_readonly_home_workdir_passes_without_home_root_read_canary_write(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        home = kernel_sandbox.protected_home()
        cwd = tempfile.mkdtemp(prefix="omni_ksb_home_workdir_", dir=home)
        self.addCleanup(_rmtree, cwd)
        original_mkstemp = kernel_sandbox.tempfile.mkstemp
        created = []

        def _tracked_mkstemp(*args, **kwargs):
            self.assertNotEqual(os.path.realpath(kwargs.get("dir", "/")), home)
            fd, path = original_mkstemp(*args, **kwargs)
            created.append(path)
            return fd, path

        with mock.patch.object(kernel_sandbox.tempfile, "mkstemp",
                               _tracked_mkstemp):
            refusal = kernel_sandbox.verify_fence(
                cwd, read_only=True, read_wall=True,
            )

        self.assertIsNone(refusal, refusal)
        # Q0 write control is intentionally created in the assigned workdir,
        # which may itself be under protected home; no separate home-root
        # read canary is created.
        self.assertEqual(len(created), 1)
        self.assertTrue(kernel_sandbox._path_inside(created[0], cwd))

    def test_home_workdir_read_wall_refuses_when_home_deny_is_removed(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        home = kernel_sandbox.protected_home()
        cwd = tempfile.mkdtemp(prefix="omni_ksb_home_q0_mutation_", dir=home)
        self.addCleanup(_rmtree, cwd)
        original_wrap = kernel_sandbox.wrap_argv
        observed = {"read_target": None}
        original_mkstemp = kernel_sandbox.tempfile.mkstemp

        def _tracked_mkstemp(*args, **kwargs):
            self.assertNotEqual(os.path.realpath(kwargs.get("dir", "/")), home)
            fd, path = original_mkstemp(*args, **kwargs)
            return fd, path

        def _observe_read_target(argv, **kwargs):
            if "os.listdir(t)" in " ".join(argv):
                observed["read_target"] = kwargs["env"]["OMNI_KSB_TARGET"]
                wrapped = original_run(argv, **kwargs)
                return wrapped
            return original_run(argv, **kwargs)

        def _without_home_read_deny(argv, workdir, read_only, **kwargs):
            wrapped = original_wrap(argv, workdir, read_only, **kwargs)
            profile_index = wrapped.index("-p") + 1
            profile = wrapped[profile_index]
            home_deny = '(deny file-read* (subpath (param "H0")))'
            self.assertIn(home_deny, profile)
            self.assertFalse(kwargs.get("extra_deny_read"))
            wrapped[profile_index] = profile.replace(home_deny, "", 1)
            return wrapped

        original_run = kernel_sandbox.subprocess.run
        with mock.patch.object(kernel_sandbox, "wrap_argv",
                               _without_home_read_deny), \
                mock.patch.object(kernel_sandbox.tempfile, "mkstemp",
                                  _tracked_mkstemp), \
                mock.patch.object(kernel_sandbox.subprocess, "run",
                                  _observe_read_target):
            refusal = kernel_sandbox.verify_fence(
                cwd, read_only=True, read_wall=True,
            )

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("read wall did not deny", refusal["detail"])
        self.assertEqual(observed["read_target"], home)

    def test_composed_home_content_reallow_is_refused_live(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        home = kernel_sandbox.protected_home()
        cwd = tempfile.mkdtemp(prefix="omni_ksb_composed_read_cwd_")
        self.addCleanup(_rmtree, cwd)

        refusal = kernel_sandbox.verify_fence(
            cwd, read_only=True, read_wall=True,
            extra_post_read=(("subpath", home),),
        )

        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "kernel_fence_unverified")
        self.assertIn("composed-profile read-probe succeeded", refusal["detail"])

    def test_home_canary_readable_unwrapped_denied_under_read_wall(self):
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        home = kernel_sandbox.protected_home()
        canary = os.path.join(
            home, ".omni_ksb_live_read_%d_%d" % (os.getpid(), os.getpid()),
        )
        cwd = tempfile.mkdtemp(prefix="omni_ksb_live_cwd_")
        self.addCleanup(_rmtree, cwd)
        try:
            with open(canary, "w", encoding="utf-8") as fh:
                fh.write("SECRET_LIVE_CANARY\n")
            # Fail-before: unwrapped read must succeed (canary location usable).
            bare = subprocess.run(
                [sys.executable, "-c",
                 "print(open(%r).read())" % canary],
                capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(bare.returncode, 0, bare.stderr)
            self.assertIn("SECRET_LIVE_CANARY", bare.stdout)

            # Under read wall: home canary denied.
            fenced = kernel_sandbox.wrap_argv(
                ["/bin/cat", canary], cwd, read_only=True, read_wall=True,
            )
            denied = subprocess.run(
                fenced, capture_output=True, text=True, timeout=5, cwd=cwd,
            )
            self.assertNotEqual(denied.returncode, 0, denied.stdout)
            self.assertNotIn("SECRET_LIVE_CANARY", denied.stdout)
            self.assertIn("Operation not permitted", denied.stderr)

            # Cwd read still allowed (temp cwd is outside home).
            ok_path = os.path.join(cwd, "ok.txt")
            with open(ok_path, "w", encoding="utf-8") as fh:
                fh.write("CWD_OK\n")
            fenced_ok = kernel_sandbox.wrap_argv(
                ["/bin/cat", ok_path], cwd, read_only=True, read_wall=True,
            )
            allowed = subprocess.run(
                fenced_ok, capture_output=True, text=True, timeout=5, cwd=cwd,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertEqual(allowed.stdout, "CWD_OK\n")

            # verify_fence with read_wall must pass (write + read proofs).
            self.assertIsNone(
                kernel_sandbox.verify_fence(cwd, read_only=True, read_wall=True)
            )
        finally:
            try:
                os.remove(canary)
            except OSError:
                pass

    def test_spoofed_HOME_still_denies_real_home_canary(self):
        """F1 live: HOME=/decoy must not neuter the wall against real home."""
        if not kernel_sandbox.available():
            self.skipTest("sandbox-exec not available")
        real_home = kernel_sandbox.protected_home()
        canary = os.path.join(
            real_home,
            ".omni_ksb_home_spoof_%d_%d" % (os.getpid(), time_ns_safe()),
        )
        decoy = tempfile.mkdtemp(prefix="omni_ksb_spoof_home_")
        cwd = tempfile.mkdtemp(prefix="omni_ksb_spoof_cwd_")
        self.addCleanup(_rmtree, decoy)
        self.addCleanup(_rmtree, cwd)
        old_home = os.environ.get("HOME")
        try:
            with open(canary, "w", encoding="utf-8") as fh:
                fh.write("SPOOF_HOME_CANARY\n")
            os.environ["HOME"] = decoy
            # Deny-set still names the real home.
            params = kernel_sandbox.profile_params(
                cwd, read_only=True, read_wall=True,
            )
            self.assertEqual(params.get("H0"), real_home)

            fenced = kernel_sandbox.wrap_argv(
                ["/bin/cat", canary], cwd, read_only=True, read_wall=True,
            )
            denied = subprocess.run(
                fenced, capture_output=True, text=True, timeout=5, cwd=cwd,
                env={**os.environ, "HOME": decoy},
            )
            self.assertNotEqual(
                denied.returncode, 0,
                "read wall neutered by HOME spoof: stdout=%r" % denied.stdout,
            )
            self.assertNotIn("SPOOF_HOME_CANARY", denied.stdout)
            self.assertEqual(len(denied.stdout), 0)

            # Cwd read still OK under the same spoofed env.
            ok_path = os.path.join(cwd, "ok.txt")
            with open(ok_path, "w", encoding="utf-8") as fh:
                fh.write("CWD_OK\n")
            fenced_ok = kernel_sandbox.wrap_argv(
                ["/bin/cat", ok_path], cwd, read_only=True, read_wall=True,
            )
            allowed = subprocess.run(
                fenced_ok, capture_output=True, text=True, timeout=5, cwd=cwd,
                env={**os.environ, "HOME": decoy},
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertEqual(allowed.stdout, "CWD_OK\n")

            # verify_fence under spoofed HOME still passes (probes protected home).
            self.assertIsNone(
                kernel_sandbox.verify_fence(cwd, read_only=True, read_wall=True)
            )
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            try:
                os.remove(canary)
            except OSError:
                pass


class TestConsultCodexFenceWiring(unittest.TestCase):
    """Behavioral pins for providers.codex consult-lane fence (issue #45)."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        from providers import codex as codex_mod
        self.codex_mod = codex_mod
        self.old_run_cli = codex_mod.run_cli
        self.old_available = kernel_sandbox.available
        self.old_verify = kernel_sandbox.verify_fence
        self.old_wrap = kernel_sandbox.wrap_argv
        self.old_is_wrapped = kernel_sandbox.is_wrapped
        self.old_which = codex_mod.shutil.which
        self.addCleanup(setattr, codex_mod, "run_cli", self.old_run_cli)
        self.addCleanup(setattr, kernel_sandbox, "available", self.old_available)
        self.addCleanup(setattr, kernel_sandbox, "verify_fence", self.old_verify)
        self.addCleanup(setattr, kernel_sandbox, "wrap_argv", self.old_wrap)
        self.addCleanup(setattr, kernel_sandbox, "is_wrapped", self.old_is_wrapped)
        self.addCleanup(setattr, codex_mod.shutil, "which", self.old_which)

        def _which(name, *a, **k):
            base = os.path.basename(str(name))
            if base == "sandbox-exec":
                return "/usr/bin/sandbox-exec"
            if base == "codex":
                return "/fake/bin/codex"
            return self.old_which(name, *a, **k)

        codex_mod.shutil.which = _which

    def test_consult_wraps_with_read_wall_true(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        seen = {}

        def _wrap(argv, workdir, read_only, read_wall=False, extra_read_paths=None,
                  extra_deny_read=None, extra_deny_read_final=None,
                  extra_deny_write_final=None, **_policy_rows):
            seen["read_wall"] = read_wall
            seen["extra"] = list(extra_read_paths or [])
            return ["/usr/bin/sandbox-exec", "-p", "PROF"] + list(argv)

        kernel_sandbox.wrap_argv = _wrap
        kernel_sandbox.is_wrapped = lambda argv: (
            bool(argv) and os.path.basename(argv[0]) == "sandbox-exec"
        )
        recorded = {}

        async def _fake_run_cli(argv, **kwargs):
            recorded["argv"] = list(argv)
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as fh:
                    fh.write("PONG")
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        self.codex_mod.run_cli = _fake_run_cli
        env = asyncio.run(self.codex_mod.run_codex_subprocess(
            "hi", "gpt-5.6-sol", 30,
        ))
        self.assertTrue(env.get("ok"), env)
        self.assertIs(seen.get("read_wall"), True)
        self.assertEqual(recorded["argv"][0], "/usr/bin/sandbox-exec")
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", recorded["argv"])
        self.assertNotIn("--sandbox", recorded["argv"])

    def test_consult_passes_files_as_extra_read_paths(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        seen = {}

        def _wrap(argv, workdir, read_only, read_wall=False, extra_read_paths=None,
                  extra_deny_read=None, extra_deny_read_final=None,
                  extra_deny_write_final=None, **_policy_rows):
            seen["extra"] = list(extra_read_paths or [])
            return ["/usr/bin/sandbox-exec", "-p", "PROF"] + list(argv)

        kernel_sandbox.wrap_argv = _wrap
        kernel_sandbox.is_wrapped = lambda argv: True
        # Also pin verify_fence kwargs
        def _verify(workdir=None, read_only=True, read_wall=False,
                    extra_read_paths=None, extra_deny_read=None,
                    extra_deny_read_final=None,
                    extra_deny_write_final=None, **_policy_rows):
            seen["verify_extra"] = list(extra_read_paths or [])
            seen["verify_rw"] = read_wall
            return None

        kernel_sandbox.verify_fence = _verify

        async def _fake_run_cli(argv, **kwargs):
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as fh:
                    fh.write("OK")
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        self.codex_mod.run_cli = _fake_run_cli
        # Plant a real file so the files= existence check passes.
        d = tempfile.mkdtemp(prefix="omni_ksb_img_")
        self.addCleanup(_rmtree, d)
        img = os.path.join(d, "x.png")
        with open(img, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        env = asyncio.run(self.codex_mod.run_codex_subprocess(
            "hi", "gpt-5.6-sol", 30, files=[img],
        ))
        self.assertTrue(env.get("ok"), env)
        want = kernel_sandbox.admit_path(
            os.path.realpath(img), "literal", label="consult files",
        )
        self.assertEqual(seen.get("extra"), [want])
        self.assertEqual(seen.get("verify_extra"), [want])
        self.assertIs(seen.get("verify_rw"), True)

    def test_consult_fail_closed_when_wrap_returns_unwrapped(self):
        kernel_sandbox.available = lambda: True
        kernel_sandbox.verify_fence = lambda *a, **k: None
        kernel_sandbox.wrap_argv = lambda argv, *a, **k: argv
        kernel_sandbox.is_wrapped = lambda argv: (
            bool(argv) and os.path.basename(argv[0]) == "sandbox-exec"
        )

        def _bomb(*_a, **_k):
            raise AssertionError(
                "run_cli must not be called when wrap postcondition fails"
            )

        self.codex_mod.run_cli = _bomb
        env = asyncio.run(self.codex_mod.run_codex_subprocess(
            "hi", "gpt-5.6-sol", 30,
        ))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_unavailable")

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_consult_unwrapped_never_bypasses(self):
        kernel_sandbox.available = lambda: False
        recorded = {}

        async def _fake_run_cli(argv, **kwargs):
            recorded["argv"] = list(argv)
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as fh:
                    fh.write("OK")
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        self.codex_mod.run_cli = _fake_run_cli
        env = asyncio.run(self.codex_mod.run_codex_subprocess(
            "hi", "gpt-5.6-sol", 30,
        ))
        self.assertTrue(env.get("ok"), env)
        argv = recorded["argv"]
        self.assertEqual(argv[0], "/fake/bin/codex")
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)

    def test_disable_inner_sandbox_inserts_bypass_after_exec(self):
        wrapped = [
            "/usr/bin/sandbox-exec", "-p", "PROF",
            "/fake/bin/codex", "exec", "--json",
            "--sandbox", "read-only", "--ephemeral", "-",
        ]
        out = kernel_sandbox.disable_inner_codex_sandbox(wrapped)
        self.assertEqual(out[0], "/usr/bin/sandbox-exec")
        self.assertNotIn("--sandbox", out)
        self.assertNotIn("read-only", out)
        self.assertEqual(
            out[out.index("exec") + 1],
            "--dangerously-bypass-approvals-and-sandbox",
        )
        self.assertEqual(
            out.count("--dangerously-bypass-approvals-and-sandbox"), 1,
        )

    def test_disable_inner_refuses_unwrapped_argv(self):
        unwrapped = [
            "/fake/bin/codex", "exec", "--json",
            "--sandbox", "read-only", "--ephemeral", "-",
        ]
        self.assertIsNone(kernel_sandbox.disable_inner_codex_sandbox(unwrapped))

    def test_disable_inner_on_exec_resume_argv(self):
        """Job-lane resume spelling: bypass stays a flag of ``exec``."""
        wrapped = [
            "/usr/bin/sandbox-exec", "-p", "PROF",
            "/fake/bin/codex", "exec", "resume", "sess-123",
            "--sandbox", "workspace-write", "--json", "-",
        ]
        out = kernel_sandbox.disable_inner_codex_sandbox(wrapped)
        self.assertIsNotNone(out)
        self.assertFalse(kernel_sandbox.inner_sandbox_disable_failed(out))
        self.assertEqual(
            out[out.index("exec") + 1],
            "--dangerously-bypass-approvals-and-sandbox",
        )
        self.assertEqual(out[out.index("exec") + 2], "resume")
        self.assertEqual(out[out.index("exec") + 3], "sess-123")

    def test_inner_sandbox_disable_failed_catches_each_conjunct(self):
        good = [
            "/usr/bin/sandbox-exec", "-p", "PROF",
            "/fake/bin/codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox", "-",
        ]
        self.assertFalse(kernel_sandbox.inner_sandbox_disable_failed(good))
        self.assertTrue(kernel_sandbox.inner_sandbox_disable_failed(None))
        # vendor sandbox flag still present
        self.assertTrue(kernel_sandbox.inner_sandbox_disable_failed(
            good[:5] + ["--sandbox=read-only"] + good[5:]))
        # bypass token missing
        self.assertTrue(kernel_sandbox.inner_sandbox_disable_failed(
            [tok for tok in good
             if tok != "--dangerously-bypass-approvals-and-sandbox"]))
        # not wrapped
        self.assertTrue(kernel_sandbox.inner_sandbox_disable_failed(good[3:]))

    def test_disable_inner_strips_short_sandbox_flag(self):
        wrapped = [
            "/usr/bin/sandbox-exec", "-p", "PROF",
            "/fake/bin/codex", "exec", "--json",
            "-s", "read-only", "--ephemeral", "-",
        ]
        out = kernel_sandbox.disable_inner_codex_sandbox(wrapped)
        self.assertIsNotNone(out)
        self.assertNotIn("-s", out)
        self.assertNotIn("read-only", out)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", out)



def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def time_ns_safe():
    try:
        return time.time_ns()
    except AttributeError:
        return int(time.time() * 1e9)


_PROTECTED_CODEX_NAMES = (
    ("config.toml", "literal"), ("hooks.json", "literal"),
    ("AGENTS.md", "literal"), ("AGENTS.override.md", "literal"),
    ("agents", "subpath"), ("skills", "subpath"),
    ("plugins", "subpath"), ("rules", "subpath"),
)


class TestCodexDefaultConsultFence(unittest.TestCase):
    """The default-account codex consult gets the agent lane's protected
    configuration rows: final write denies and the alias refusal."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix, base=None):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix, dir=base))
        self.addCleanup(_rmtree, path)
        return path

    def _fence(self, home, cwd):
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            return kernel_sandbox.codex_consult_default_fence(cwd)

    def test_final_write_denies_follow_every_allow(self):
        home = self._dir("omni_ksb_dc_home_")
        root = os.path.join(home, ".codex")
        os.mkdir(root)
        cwd = self._dir("omni_ksb_dc_cwd_")
        rows = self._fence(home, cwd)
        self.assertIn("extra_deny_write_final", rows)
        for name, kind in _PROTECTED_CODEX_NAMES:
            self.assertIn((kind, os.path.join(root, name)),
                          rows["extra_deny_write_final"], name)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            kw = dict(read_wall=True, **rows)
            text = kernel_sandbox.build_profile(cwd, True, **kw)
            params = kernel_sandbox.profile_params(cwd, True, **kw)
        finals = {"F%d" % i: p for i, (_k, p) in enumerate(rows["extra_deny_write_final"])}
        last_allow = max(text.rfind(tok) for tok in (
            "(allow file-write*", "(allow file-read*", "(allow file-read-metadata"))
        for key, path in finals.items():
            self.assertEqual(params[key], path)
            self.assertGreater(text.index('(param "%s")' % key), last_allow, key)

    def test_default_consult_denies_the_whole_home_and_reallows_only_state(self):
        home = self._dir("omni_ksb_dc_home_")
        root = os.path.join(home, ".codex")
        os.makedirs(os.path.join(root, "sessions"))
        os.makedirs(os.path.join(root, "hooks"))
        os.makedirs(os.path.join(root, ".tmp", "plugin"))
        for rel in ("auth.json", "hooks/start.sh", "custom.config.toml",
                    ".tmp/plugin/payload", "logs_2.sqlite"):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                fh.write("x")
        cwd = self._dir("omni_ksb_dc_cwd_")
        rows = self._fence(home, cwd)
        self.assertIn(("subpath", root), rows["extra_deny_write"])
        self.assertIn(("subpath", root), rows["extra_deny_link"])
        reallowed = rows["extra_post_write"]

        def covered(path):
            for kind, allowed in reallowed:
                if kind == "subpath" and (path == allowed
                                          or path.startswith(allowed + os.sep)):
                    return True
                if kind == "literal" and path == allowed:
                    return True
            return False

        for rel in ("sessions/2026/10/09/rollout.jsonl", "sessions", "auth.json",
                    "log/codex.log", "cache/x", "tmp/arg0/helper",
                    "logs_2.sqlite", "logs_2.sqlite-wal", "history.jsonl"):
            self.assertTrue(covered(os.path.join(root, rel)), rel)
        for rel in ("hooks/start.sh", "custom.config.toml", ".tmp/plugin/payload",
                    "new-at-root", "config.toml", "hooks.json", "AGENTS.md",
                    "AGENTS.override.md", "skills/s.md", "agents/a.md",
                    "plugins/p", "rules/r", ".tmp"):
            self.assertFalse(covered(os.path.join(root, rel)), rel)
        # Every re-allow stays inside the codex home.
        for _kind, allowed in reallowed:
            self.assertTrue(allowed == root or allowed.startswith(root + os.sep))
        # The protected names stay FINAL denies on top of the whole-home deny.
        for name, kind in _PROTECTED_CODEX_NAMES:
            self.assertIn((kind, os.path.join(root, name)),
                          rows["extra_deny_write_final"], name)

    def test_default_consult_read_denies_registered_sibling_homes(self):
        home = self._dir("omni_ksb_dc_home_")
        os.mkdir(os.path.join(home, ".codex"))
        other = self._dir("omni_ksb_dc_other_")
        cwd = self._dir("omni_ksb_dc_cwd_")
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_B": other}, clear=False):
            rows = self._fence(home, cwd)
        self.assertEqual(rows["extra_deny_read"], rows["extra_deny_read_final"])
        self.assertIn(other, [p for _k, p in rows["extra_deny_read_final"]])
        # The default home itself stays readable.
        self.assertNotIn(os.path.join(home, ".codex"),
                         [p for _k, p in rows["extra_deny_read"]])

    def test_default_consult_refuses_a_symlinked_state_leaf(self):
        home = self._dir("omni_ksb_dc_home_")
        root = os.path.join(home, ".codex")
        os.mkdir(root)
        cwd = self._dir("omni_ksb_dc_cwd_")
        os.symlink(cwd, os.path.join(root, "sessions"))
        with self.assertRaises(kernel_sandbox.FencePathRejected):
            self._fence(home, cwd)

    def test_protected_name_linked_into_the_home_or_a_writable_root_is_refused(self):
        for name, _kind in _PROTECTED_CODEX_NAMES:
            for where in ("sessions", "cwd"):
                with self.subTest(name=name, where=where):
                    home = self._dir("omni_ksb_dc_home_")
                    root = os.path.join(home, ".codex")
                    os.makedirs(os.path.join(root, "sessions"))
                    cwd = self._dir("omni_ksb_dc_cwd_")
                    target = os.path.join(
                        root if where == "sessions" else cwd,
                        "sessions/payload" if where == "sessions" else "payload")
                    with open(target, "w", encoding="utf-8") as fh:
                        fh.write("x")
                    os.symlink(target, os.path.join(root, name))
                    with self.assertRaises(kernel_sandbox.FencePathRejected) as ctx:
                        self._fence(home, cwd)
                    self.assertIn(name, ctx.exception.detail)
                    self.assertNotIn(home, ctx.exception.detail)

    def test_intentional_link_into_an_unwritable_shared_tree_is_allowed(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outside = self._dir("omni_ksb_dc_dot_", base=repo)
        home = self._dir("omni_ksb_dc_home_")
        root = os.path.join(home, ".codex")
        os.mkdir(root)
        target = os.path.join(outside, "codex-AGENTS.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.symlink(target, os.path.join(root, "AGENTS.md"))
        rows = self._fence(home, self._dir("omni_ksb_dc_cwd_"))
        finals = {p for _k, p in rows["extra_deny_write_final"]}
        self.assertIn(target, finals)
        self.assertIn(os.path.join(root, "AGENTS.md"), finals)


@unittest.skipUnless(kernel_sandbox.available(), "sandbox-exec not available")
class TestCodexDefaultConsultLive(unittest.TestCase):
    """Real sandbox-exec over the default-account consult policy."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix, base=None):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix, dir=base))
        self.addCleanup(_rmtree, path)
        return path

    def _run(self, kw, cwd, script):
        argv = kernel_sandbox.wrap_argv(["/bin/sh", "-c", script], cwd, True, **kw)
        return subprocess.run(argv, capture_output=True, text=True, timeout=10, cwd=cwd)

    def test_default_consult_cannot_rewrite_config_or_replace_agents(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shared = self._dir("omni_ksb_dcl_shared_", base=repo)
        home = self._dir("omni_ksb_dcl_home_")
        root = os.path.join(home, ".codex")
        os.makedirs(os.path.join(root, "sessions"))
        cwd = self._dir("omni_ksb_dcl_cwd_")
        link_target = os.path.join(shared, "codex-AGENTS.md")
        for path, text in ((link_target, "SHARED\n"),
                           (os.path.join(root, "config.toml"), "ORIGINAL\n"),
                           (os.path.join(root, "auth.json"), "AUTH\n")):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        os.symlink(link_target, os.path.join(root, "AGENTS.md"))
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            rows = kernel_sandbox.codex_consult_default_fence(cwd)
            kw = dict(read_wall=True, **rows)
            # Control: without the final rows the shared allowance lets the
            # child unlink the link and plant a regular file.
            # (the old shared whole-home allowance: no whole-home deny, no
            # state-only re-allow, no final rows).
            leak = self._run(
                dict(kw, extra_deny_write_final=(), extra_deny_write=(),
                     extra_post_write=(), extra_deny_link=()), cwd,
                "rm -f %s/AGENTS.md && echo LEAK > %s/AGENTS.md" % (root, root))
            self.assertEqual(leak.returncode, 0, leak.stderr)
            os.unlink(os.path.join(root, "AGENTS.md"))
            os.symlink(link_target, os.path.join(root, "AGENTS.md"))
            ok = self._run(kw, cwd,
                           "mkdir -p %s/sessions/2026 && echo x > %s/sessions/2026/r.jsonl"
                           " && echo x > %s/auth.json" % (root, root, root))
            self.assertEqual(ok.returncode, 0, ok.stderr)
            for script in ("echo PWNED > %s/config.toml" % root,
                           "rm -f %s/config.toml" % root,
                           "echo PWNED > %s/hooks.json" % root,
                           "rm -f %s/AGENTS.md" % root,
                           "mv %s/sessions %s/AGENTS.md" % (root, root),
                           "ln -sf %s/sessions/2026/r.jsonl %s/AGENTS.md" % (root, root),
                           "echo PWNED > %s/AGENTS.md" % root,
                           "echo PWNED > %s" % link_target,
                           "echo PWNED > %s/AGENTS.override.md" % root,
                           "mkdir -p %s/skills && echo x > %s/skills/s.md" % (root, root)):
                bad = self._run(kw, cwd, script)
                self.assertNotEqual(bad.returncode, 0, script)
        with open(os.path.join(root, "config.toml"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "ORIGINAL\n")
        self.assertTrue(os.path.islink(os.path.join(root, "AGENTS.md")))
        with open(link_target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "SHARED\n")
        for name in ("hooks.json", "AGENTS.override.md", "skills"):
            self.assertFalse(os.path.lexists(os.path.join(root, name)), name)

    def test_default_consult_cannot_poison_later_runs_outside_state(self):
        home = self._dir("omni_ksb_dcl_home_")
        root = os.path.join(home, ".codex")
        os.makedirs(os.path.join(root, "sessions"))
        os.makedirs(os.path.join(root, "hooks"))
        os.makedirs(os.path.join(root, ".tmp", "plugin"))
        cwd = self._dir("omni_ksb_dcl_cwd_")
        seeded = {"hooks/start.sh": "ORIGINAL-HOOK\n",
                  "custom.config.toml": "ORIGINAL-LAYER\n",
                  ".tmp/plugin/payload": "ORIGINAL-PLUGIN\n",
                  "auth.json": "AUTH\n"}
        for rel, text in seeded.items():
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            rows = kernel_sandbox.codex_consult_default_fence(cwd)
            kw = dict(read_wall=True, **rows)
            # Control: the shared whole-home allowance (the pre-fix policy)
            # lets the child rewrite every one of them.
            old = dict(kw, extra_deny_write=(), extra_post_write=(),
                       extra_deny_link=())
            for rel in ("hooks/start.sh", "custom.config.toml",
                        ".tmp/plugin/payload"):
                leak = self._run(old, cwd, "echo LEAK > %s/%s" % (root, rel))
                self.assertEqual(leak.returncode, 0, (rel, leak.stderr))
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(seeded[rel])
            ok = self._run(kw, cwd,
                           "mkdir -p %s/sessions/2026 && echo x > %s/sessions/2026/r.jsonl"
                           " && echo x > %s/auth.json && echo x > %s/history.jsonl"
                           % (root, root, root, root))
            self.assertEqual(ok.returncode, 0, ok.stderr)
            for script in ("echo PWNED > %s/hooks/start.sh" % root,
                           "rm -f %s/hooks/start.sh" % root,
                           "echo PWNED > %s/custom.config.toml" % root,
                           "echo PWNED > %s/.tmp/plugin/payload" % root,
                           "echo x > %s/.tmp/plugin/new" % root,
                           "echo x > %s/new-at-root" % root,
                           "mkdir %s/newdir" % root):
                bad = self._run(kw, cwd, script)
                self.assertNotEqual(bad.returncode, 0, script)
        for rel in ("hooks/start.sh", "custom.config.toml", ".tmp/plugin/payload"):
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), seeded[rel], rel)
        for rel in ("new-at-root", "newdir", ".tmp/plugin/new"):
            self.assertFalse(os.path.lexists(os.path.join(root, rel)), rel)
        with open(os.path.join(root, "auth.json"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x\n")


class TestCodexModelsCacheReadOnly(unittest.TestCase):
    """models_cache.json feeds later sessions on the same account, so no
    fenced codex child may write it (consult and agent, default and named)."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def _assert_final_deny_no_later_allow(self, rows, target):
        finals = list(rows["extra_deny_write_final"])
        self.assertIn(("literal", target), finals)
        self.assertNotIn(("literal", target), list(rows["extra_post_write"]))
        for kind, allowed in rows["extra_post_write"]:
            if kind == "subpath":
                self.assertFalse(target.startswith(allowed + os.sep), allowed)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home):
            kw = dict(read_wall=True, **rows)
            cwd = self.cwd
            text = kernel_sandbox.build_profile(cwd, True, **kw)
            params = kernel_sandbox.profile_params(cwd, True, **kw)
        key = "F%d" % finals.index(("literal", target))
        self.assertEqual(params[key], target)
        last_allow = max(text.rfind(tok) for tok in (
            "(allow file-write*", "(allow file-read*", "(allow file-read-metadata"))
        self.assertGreater(text.index('(param "%s")' % key), last_allow)

    def test_default_consult_policy(self):
        self.home = self._dir("omni_ksb_mc_home_")
        root = os.path.join(self.home, ".codex")
        os.makedirs(root)
        self.cwd = self._dir("omni_ksb_mc_cwd_")
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home):
            rows = kernel_sandbox.codex_consult_default_fence(self.cwd)
        self._assert_final_deny_no_later_allow(
            rows, os.path.join(root, "models_cache.json"))

    def test_named_consult_policy(self):
        self.home = self._dir("omni_ksb_mc_home_")
        os.makedirs(os.path.join(self.home, ".codex"))
        prof = self._dir("omni_ksb_mc_prof_")
        self.cwd = self._dir("omni_ksb_mc_cwd_")
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home):
            ap = kernel_sandbox.admit_path(prof, "subpath", label="CODEX_HOME")
            rows = kernel_sandbox.codex_consult_account_fence(ap.path, self.cwd)
        self._assert_final_deny_no_later_allow(
            rows, os.path.join(prof, "models_cache.json"))

    def test_agent_lane_policy_default_and_named(self):
        for named in (False, True):
            with self.subTest(named=named):
                self.home = self._dir("omni_ksb_mc_home_")
                root = os.path.join(self.home, ".codex")
                os.makedirs(root)
                self.cwd = self._dir("omni_ksb_mc_cwd_")
                cfg = None
                if named:
                    root = self._dir("omni_ksb_mc_prof_")
                    cfg = root
                with mock.patch.object(kernel_sandbox, "protected_home",
                                       return_value=self.home):
                    pol = kernel_sandbox.agent_fence_policy(
                        "codex", workdir=self.cwd, read_only=False,
                        config_dir=cfg)
                self.assertNotIn("reason", pol, pol)
                wk = pol["wrap_kw"]
                rows = {k: wk[k] for k in (
                    "extra_deny_write_final", "extra_post_write")}
                target = os.path.join(root, "models_cache.json")
                self.assertIn(("literal", target), rows["extra_deny_write_final"])
                self.assertNotIn(("literal", target), list(rows["extra_post_write"]))

    def test_live_fenced_shell_cannot_touch_models_cache(self):
        home = self._dir("omni_ksb_mc_home_")
        root = os.path.join(home, ".codex")
        os.makedirs(os.path.join(root, "sessions"))
        cwd = self._dir("omni_ksb_mc_cwd_")
        cache = os.path.join(root, "models_cache.json")
        for path, text in ((cache, "ORIGINAL\n"),
                           (os.path.join(root, "auth.json"), "AUTH\n")):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)

        def run(kw, script):
            argv = kernel_sandbox.wrap_argv(["/bin/sh", "-c", script], cwd, True, **kw)
            return subprocess.run(argv, capture_output=True, text=True,
                                  timeout=10, cwd=cwd)

        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            rows = kernel_sandbox.codex_consult_default_fence(cwd)
            kw = dict(read_wall=True, **rows)
            ok = run(kw, "mkdir -p %s/sessions/2026 && echo x > %s/sessions/2026/r"
                         " && echo x > %s/auth.json" % (root, root, root))
            self.assertEqual(ok.returncode, 0, ok.stderr)
            for script in ("echo PWNED > %s" % cache,
                           "echo PWNED >> %s" % cache,
                           "rm -f %s" % cache,
                           "echo PWNED > %s/m.tmp && mv %s/m.tmp %s" % (root, root, cache)):
                bad = run(kw, script)
                self.assertNotEqual(bad.returncode, 0, script)
        with open(cache, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "ORIGINAL\n")
        with open(os.path.join(root, "auth.json"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x\n")


def _other_case(path):
    """The same directory on APFS, spelled with the leaf's case swapped."""
    return os.path.join(os.path.dirname(path), os.path.basename(path).swapcase())


class TestRefusalsIgnoreCaseAndNormalization(unittest.TestCase):
    """realpath keeps the caller's spelling; APFS and Seatbelt do not care."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _dir(self, prefix):
        path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(_rmtree, path)
        return path

    def test_fold_covers_case_and_nfc_nfd(self):
        import unicodedata
        nfc = unicodedata.normalize("NFC", "/x/caf\u00e9")
        nfd = unicodedata.normalize("NFD", nfc)
        self.assertNotEqual(nfc, nfd)
        fold = kernel_sandbox._fold_path
        self.assertEqual(fold(nfc), fold(nfd))
        self.assertEqual(fold("/x/CAF\u00c9"), fold(nfd))
        self.assertNotEqual(fold("/x/cafe"), fold(nfd))

    def test_nested_sibling_in_other_case_is_refused(self):
        outer = self._dir("omni_ksb_case_outer_")
        inner = os.path.join(outer, "inner")
        os.mkdir(inner)
        wd = self._dir("omni_ksb_case_wd_")
        for env in ({"GROK_HOME_B": _other_case(outer)},                        # contains the run's home
                    {"GROK_HOME_B": os.path.join(_other_case(inner), "sub")}):  # inside its root
            if "sub" in env["GROK_HOME_B"]:
                os.makedirs(os.path.join(inner, "sub"), exist_ok=True)
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=False):
                policy = kernel_sandbox.agent_fence_policy(
                    "cursor", workdir=wd, read_only=False, config_dir=inner)
                self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
                self.assertIn("GROK_HOME_B", policy["detail"])

    def test_protected_symlink_into_other_case_writable_root_is_refused(self):
        home_root = self._dir("omni_ksb_case_home_")
        writable = self._dir("omni_ksb_case_w_")
        os.makedirs(os.path.join(writable, "sub"))
        payload = os.path.join(writable, "sub", "payload")
        with open(payload, "w", encoding="utf-8") as fh:
            fh.write("x\n")
        name = os.path.join(home_root, "CLAUDE.md")
        os.symlink(payload, name)
        entries = [("CLAUDE.md", name)]
        for roots in ([writable], [_other_case(writable)]):
            with self.subTest(roots=roots):
                self.assertIsNotNone(kernel_sandbox._protected_config_refusal(
                    "default claude home", entries, named=False,
                    writable_roots=roots, home_root=home_root))
        self.assertIsNone(kernel_sandbox._protected_config_refusal(
            "default claude home", entries, named=False,
            writable_roots=[self._dir("omni_ksb_case_other_")], home_root=home_root))

    def test_identity_arm_catches_aliases_folding_misses(self):
        # Stand-in for an alias Python's folding cannot see (APFS's own case
        # table, a newer Unicode version): disable folding, keep the inode.
        root = self._dir("omni_ksb_case_id_")
        os.makedirs(os.path.join(root, "acct", "x"))
        alias = os.path.join(root, "ACCT")
        with mock.patch.object(kernel_sandbox, "_fold_path", side_effect=lambda t: t):
            self.assertTrue(kernel_sandbox._path_inside_any_case(
                os.path.join(alias, "x"), os.path.join(root, "acct")))
            self.assertTrue(kernel_sandbox._same_path_any_case(alias, os.path.join(root, "acct")))
            self.assertFalse(kernel_sandbox._path_inside_any_case(
                os.path.join(root, "other"), os.path.join(root, "acct")))

    def test_exemptions_never_fold(self):
        # Two folded-equal names are two directories on a case-sensitive
        # volume: an exemption decided by folding would widen access.
        with mock.patch.object(kernel_sandbox, "_same_dir", return_value=False):
            self.assertFalse(kernel_sandbox._same_dir_exact("/v/Account", "/v/account"))
            self.assertTrue(kernel_sandbox._same_dir_exact("/v/a", "/v/a"))
        with mock.patch.object(kernel_sandbox, "_same_dir",
                               side_effect=kernel_sandbox._IdentityUnavailable("/v")):
            self.assertFalse(kernel_sandbox._same_dir_exact("/v/A", "/v/a"))

    def test_missing_path_fallback_folds_normalization(self):
        import unicodedata
        nfc = "/nonexistent/" + unicodedata.normalize("NFC", "Caf\u00e9")
        nfd = "/nonexistent/" + unicodedata.normalize("NFD", "caf\u00e9") + "/Library"
        self.assertTrue(kernel_sandbox._lexical_ci_same_or_inside(nfd, nfc))

    def test_default_login_dir_in_other_case_is_refused(self):
        home = self._dir("omni_ksb_case_home2_")
        os.mkdir(os.path.join(home, ".claude"))
        for spelled in (".claude", ".CLAUDE"):
            with self.subTest(spelled=spelled):
                why = kernel_sandbox._caller_config_dir_rejection(
                    os.path.join(home, spelled), label="CLAUDE_CONFIG_DIR", home=home,
                    allow_dotdir_prefix=".claude-", refuse_default_login=".claude",
                    launcher_roots=(os.path.join(home, ".claude"),))
                self.assertIsNotNone(why)
                self.assertIn("default login directory", why)

    def test_pi_state_symlink_into_other_case_codex_home_is_dropped(self):
        home = self._dir("omni_ksb_case_pi_")
        os.makedirs(os.path.join(home, ".codex"))
        os.makedirs(os.path.join(home, ".pi", "agent"))
        os.symlink(os.path.join(home, ".CoDeX"), os.path.join(home, ".pi", "agent", "sessions"))
        wd = self._dir("omni_ksb_case_piwd_")
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy("pi", workdir=wd, read_only=False)
        rows = (policy.get("wrap_kw") or policy).get("extra_post_write") or ()
        codex = os.path.join(home, ".codex")
        self.assertFalse([r for r in rows if kernel_sandbox._path_inside_any_case(r[1], codex)], rows)

class TestVendorStateLeavesRefuseSymlinks(unittest.TestCase):
    """A child can replace its own writable state leaf with a symlink; the
    next run must refuse it, never allow writes at the link's target."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _home(self):
        home = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_leaf_home_"))
        self.addCleanup(_rmtree, home)
        target = os.path.join(home, "Library", "LaunchAgents")
        os.makedirs(target)
        return home, target

    def _policy(self, provider, home):
        wd = os.path.realpath(tempfile.mkdtemp(prefix="omni_ksb_leaf_wd_"))
        self.addCleanup(_rmtree, wd)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            return kernel_sandbox.agent_fence_policy(provider, workdir=wd, read_only=False)

    def _assert_refused(self, policy, target):
        rows = (policy.get("wrap_kw") or {}).get("extra_post_write") or ()
        self.assertFalse([r for r in rows if r[1].startswith(target)], rows)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)

    def test_pi_leaves(self):
        for leaf, make in ((("sessions",), "dir"), (("settings.json.lock",), "dir"),
                           (("auth.json.lock",), "dir"), (("auth.json",), "file")):
            with self.subTest(leaf=leaf):
                home, target = self._home()
                agent = os.path.join(home, ".pi", "agent")
                os.makedirs(agent)
                dest = target if make == "dir" else os.path.join(target, "evil.plist")
                if make == "file":
                    open(dest, "w").close()
                os.symlink(dest, os.path.join(agent, *leaf))
                self._assert_refused(self._policy("pi", home), target)

    def test_pi_agent_dir_symlink(self):
        home, target = self._home()
        os.makedirs(os.path.join(home, ".pi"))
        os.symlink(target, os.path.join(home, ".pi", "agent"))
        self._assert_refused(self._policy("pi", home), target)

    def test_cursor_projects(self):
        home, target = self._home()
        os.makedirs(os.path.join(home, ".cursor"))
        os.symlink(target, os.path.join(home, ".cursor", "projects"))
        self._assert_refused(self._policy("cursor", home), target)

    def test_plain_leaves_are_still_allowed(self):
        home, _target = self._home()
        os.makedirs(os.path.join(home, ".pi", "agent", "sessions"))
        os.makedirs(os.path.join(home, ".cursor", "projects"))
        for provider, leaf in (("pi", os.path.join(home, ".pi", "agent", "sessions")),
                               ("cursor", os.path.join(home, ".cursor", "projects"))):
            with self.subTest(provider=provider):
                policy = self._policy(provider, home)
                self.assertNotIn("ok", policy, policy)
                self.assertIn(("subpath", leaf), policy["wrap_kw"]["extra_post_write"])


if __name__ == "__main__":
    unittest.main()
