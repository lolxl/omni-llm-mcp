#!/usr/bin/env python3
# no-opencc
"""Fence extras carry filesystem identity, not spelling (issue #120).

No processes, no network. Home-shaped fixtures live in a temp dir with
``protected_home`` patched.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from providers import agent_lane, kernel_sandbox


def _rmtree(path):
    shutil.rmtree(path, ignore_errors=True)


def _canonical_tempdir(prefix):
    return os.path.realpath(tempfile.mkdtemp(prefix=prefix))


def _p_values(params):
    return {v for k, v in params.items() if k.startswith("P")}


def _r_values(params):
    return {v for k, v in params.items() if k.startswith("R")}


def _plant_same_inode_ancestor(mid):
    """Replace *mid* with a symlink to the renamed original (identity kept)."""
    moved = mid + ".real"
    os.rename(mid, moved)
    os.symlink(moved, mid)
    return moved


class TestFenceIdentity(unittest.TestCase):
    def _fake_home(self):
        home = _canonical_tempdir("omni120a_home_")
        self.addCleanup(_rmtree, home)
        return home

    def _dir(self, prefix="omni120a_d_"):
        d = _canonical_tempdir(prefix)
        self.addCleanup(_rmtree, d)
        return d

    def _file_in(self, parent, name="leaf.txt"):
        path = os.path.join(parent, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("x\n")
        return os.path.realpath(path)

    def _admit_dir(self, path, label="test"):
        return kernel_sandbox.admit_path(path, "subpath", label=label)

    def _admit_file(self, path, label="test"):
        return kernel_sandbox.admit_path(path, "literal", label=label)

    def _assert_rejected(self, exc, *needles):
        detail = exc.exception.detail
        self.assertLessEqual(len(detail), 300)
        for needle in needles:
            self.assertIn(needle, detail, detail)

    # --- r1: admitted dir swapped for a symlink to C ---

    def test_r1_write(self):
        d = self._dir("omni120a_r1w_d_")
        c = self._dir("omni120a_r1w_c_")
        ap = self._admit_dir(d)
        os.rename(d, d + ".moved")
        os.symlink(c, d)
        self.addCleanup(_rmtree, d)
        with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
            kernel_sandbox.build_profile(
                None, read_only=True, extra_write_paths=[ap],
            )
        self._assert_rejected(caught, d, c)

    def test_r1_read(self):
        home = self._fake_home()
        d = os.path.join(home, "r1r_d")
        os.mkdir(d)
        d = os.path.realpath(d)
        c = self._dir("omni120a_r1r_c_")
        ap = self._admit_dir(d)
        os.rename(d, d + ".moved")
        os.symlink(c, d)
        self.addCleanup(_rmtree, d)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox.build_profile(
                    None, read_only=True, read_wall=True, extra_read_paths=[ap],
                )
        self._assert_rejected(caught, d, c)

    # --- r2: same spelling, new real dir (identity differs) ---

    def test_r2_write(self):
        d = self._dir("omni120a_r2w_d_")
        ap = self._admit_dir(d)
        os.rename(d, d + ".moved")
        os.mkdir(d)
        with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
            kernel_sandbox.build_profile(
                None, read_only=True, extra_write_paths=[ap],
            )
        self._assert_rejected(caught, d)

    def test_r2_read(self):
        home = self._fake_home()
        d = os.path.join(home, "r2r_d")
        os.mkdir(d)
        d = os.path.realpath(d)
        ap = self._admit_dir(d)
        os.rename(d, d + ".moved")
        os.mkdir(d)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox.build_profile(
                    None, read_only=True, read_wall=True, extra_read_paths=[ap],
                )
        self._assert_rejected(caught, d)

    # --- r3: symlink ancestor planted inside the fake home ---

    def test_r3_write(self):
        home = self._fake_home()
        mid = os.path.join(home, "mid")
        leaf = os.path.join(mid, "leaf")
        os.makedirs(leaf)
        leaf = os.path.realpath(leaf)
        ap = self._admit_dir(leaf)
        _plant_same_inode_ancestor(mid)
        self.addCleanup(_rmtree, mid)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox.build_profile(
                    None, read_only=True, extra_write_paths=[ap],
                )
        self._assert_rejected(caught, mid)

    def test_r3_read(self):
        home = self._fake_home()
        mid = os.path.join(home, "mid")
        leaf = os.path.join(mid, "leaf")
        os.makedirs(leaf)
        leaf = os.path.realpath(leaf)
        ap = self._admit_dir(leaf)
        _plant_same_inode_ancestor(mid)
        self.addCleanup(_rmtree, mid)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox.build_profile(
                    None, read_only=True, read_wall=True, extra_read_paths=[ap],
                )
        self._assert_rejected(caught, mid)

    # --- r4: symlink ancestor planted in a user-writable dir outside home ---

    def test_r4_write(self):
        base = self._dir("omni120a_r4w_")
        mid = os.path.join(base, "mid")
        leaf = os.path.join(mid, "leaf")
        os.makedirs(leaf)
        leaf = os.path.realpath(leaf)
        ap = self._admit_dir(leaf)
        _plant_same_inode_ancestor(mid)
        self.addCleanup(_rmtree, mid)
        with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
            kernel_sandbox.build_profile(
                None, read_only=True, extra_write_paths=[ap],
            )
        self._assert_rejected(caught, mid)

    def test_r4_read(self):
        base = self._dir("omni120a_r4r_")
        mid = os.path.join(base, "mid")
        leaf = os.path.join(mid, "leaf")
        os.makedirs(leaf)
        leaf = os.path.realpath(leaf)
        ap = self._admit_dir(leaf)
        _plant_same_inode_ancestor(mid)
        self.addCleanup(_rmtree, mid)
        with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
            kernel_sandbox.build_profile(
                None, read_only=True, read_wall=True, extra_read_paths=[ap],
            )
        self._assert_rejected(caught, mid)

    # --- r5: root-owned /var/folders ancestor must BUILD, canonical entry ---

    def test_r5_write(self):
        d = tempfile.mkdtemp(prefix="omni120a_r5w_")
        self.addCleanup(_rmtree, d)
        self.assertTrue(d.startswith("/var/") or os.path.realpath(d).startswith(
            "/private/var/"), d)
        kernel_sandbox.build_profile(
            None, read_only=True, extra_write_paths=[d],
        )
        params = kernel_sandbox.profile_params(
            None, read_only=True, extra_write_paths=[d],
        )
        canonical = os.path.realpath(d)
        self.assertTrue(canonical.startswith("/private/var/"), canonical)
        self.assertIn(canonical, _p_values(params))
        if d != canonical:
            self.assertNotIn(d, _p_values(params))

    def test_r5_read(self):
        d = tempfile.mkdtemp(prefix="omni120a_r5r_")
        self.addCleanup(_rmtree, d)
        kernel_sandbox.build_profile(
            None, read_only=True, read_wall=True, extra_read_paths=[d],
        )
        params = kernel_sandbox.profile_params(
            None, read_only=True, read_wall=True, extra_read_paths=[d],
        )
        canonical = os.path.realpath(d)
        self.assertTrue(canonical.startswith("/private/var/"), canonical)
        if canonical in _r_values(params):
            if d != canonical:
                self.assertNotIn(d, _r_values(params))

    # --- r6: symlink at admission; caller admits the validated target ---

    def test_r6_write(self):
        target = self._dir("omni120a_r6w_t_")
        base = self._dir("omni120a_r6w_b_")
        link = os.path.join(base, "link")
        os.symlink(target, link)
        ap = self._admit_dir(target)
        kernel_sandbox.build_profile(
            None, read_only=True, extra_write_paths=[ap],
        )
        params = kernel_sandbox.profile_params(
            None, read_only=True, extra_write_paths=[ap],
        )
        self.assertIn(target, _p_values(params))
        self.assertNotIn(link, _p_values(params))

    def test_r6_read(self):
        home = self._fake_home()
        target = os.path.join(home, "target")
        os.mkdir(target)
        target = os.path.realpath(target)
        link = os.path.join(home, "link")
        os.symlink(target, link)
        ap = self._admit_dir(target)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            kernel_sandbox.build_profile(
                None, read_only=True, read_wall=True, extra_read_paths=[ap],
            )
            params = kernel_sandbox.profile_params(
                None, read_only=True, read_wall=True, extra_read_paths=[ap],
            )
        self.assertIn(target, _r_values(params))
        self.assertNotIn(link, _r_values(params))

    # --- r7: .claude.json replaced by a symlink to a dir ---

    def test_r7_read(self):
        home = self._fake_home()
        profile = os.path.join(home, ".claude-x")
        os.mkdir(profile)
        victim = self._dir("omni120a_r7_victim_")
        marker = os.path.join(victim, "secret")
        os.mkdir(marker)
        claude_json = os.path.join(profile, ".claude.json")
        os.symlink(victim, claude_json)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=profile,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)
        # If a profile were emitted, the victim dir must not be a subpath read.
        extras = (policy.get("wrap_kw") or {}).get("extra_read_paths") or ()
        victim_rp = os.path.realpath(victim)
        for item in extras:
            path = item.path if isinstance(item, kernel_sandbox.AdmittedPath) else item
            kind = item.kind if isinstance(item, kernel_sandbox.AdmittedPath) else None
            self.assertFalse(
                kind == "subpath" and os.path.realpath(path) == victim_rp,
                extras,
            )

    # --- r8: H1 race — rejection plants a symlink to C ---

    def test_r8_write(self):
        home = self._fake_home()
        d = os.path.join(home, ".claude-x")
        os.mkdir(d)
        d = os.path.realpath(d)
        c = self._dir("omni120a_r8_c_")

        def _swap(config_dir, *args, **kwargs):
            if os.path.isdir(d) and not os.path.islink(d):
                os.rename(d, d + ".moved")
                os.symlink(c, d)
            return None

        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home), \
                mock.patch.object(
                    kernel_sandbox, "_caller_config_dir_rejection",
                    side_effect=_swap,
                ):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=d,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)
        detail = policy.get("detail") or ""
        self.assertTrue(d in detail or c in detail, detail)

    # --- r9: consult files through a symlinked parent ---

    def test_r9_read(self):
        # Root-owned /var spelling: admitted.
        var_dir = tempfile.mkdtemp(prefix="omni120a_r9_var_")
        self.addCleanup(_rmtree, var_dir)
        var_file = self._file_in(var_dir, "in.bin")
        raw = var_file if not os.path.islink(var_file) else os.path.realpath(var_file)
        ap_ok = kernel_sandbox.admit_path(raw, "literal", label="consult files")
        self.assertTrue(ap_ok.path.startswith("/private/var/"), ap_ok.path)
        kernel_sandbox.build_profile(
            None, read_only=True, read_wall=True, extra_read_paths=[ap_ok],
        )

        # User-writable symlink parent: refused.
        base = self._dir("omni120a_r9_uw_")
        realdir = os.path.join(base, "real")
        os.mkdir(realdir)
        leaf = self._file_in(realdir, "in.bin")
        link = os.path.join(base, "link")
        os.symlink(realdir, link)
        via = os.path.join(link, "in.bin")
        self.assertFalse(os.path.islink(via))
        self.assertTrue(os.path.isfile(via))
        with self.assertRaises(kernel_sandbox.FencePathRejected):
            kernel_sandbox.admit_path(via, "literal", label="consult files")

    # --- r10: raw str extra link/../x admits the true target, not lexical ---

    def _r10_layout(self):
        base = self._dir("omni120a_r10_base_")
        target_root = self._dir("omni120a_r10_tgt_")
        nested = os.path.join(target_root, "nested")
        os.mkdir(nested)
        true_x = os.path.join(target_root, "x")
        os.mkdir(true_x)
        lexical_x = os.path.join(base, "x")
        os.mkdir(lexical_x)
        link = os.path.join(base, "link")
        os.symlink(nested, link)
        raw = os.path.join(link, "..", "x")
        true_rp = os.path.realpath(raw)
        lexical = os.path.normpath(raw)
        self.assertEqual(true_rp, os.path.realpath(true_x))
        self.assertEqual(lexical, lexical_x)
        self.assertNotEqual(true_rp, lexical)
        return raw, true_rp, lexical

    def test_r10_write(self):
        raw, true_rp, lexical = self._r10_layout()
        kernel_sandbox.build_profile(
            None, read_only=True, extra_write_paths=[raw],
        )
        params = kernel_sandbox.profile_params(
            None, read_only=True, extra_write_paths=[raw],
        )
        self.assertIn(true_rp, _p_values(params))
        self.assertNotIn(lexical, _p_values(params))

    def test_r10_read(self):
        raw, true_rp, lexical = self._r10_layout()
        kernel_sandbox.build_profile(
            None, read_only=True, read_wall=True, extra_read_paths=[raw],
        )
        params = kernel_sandbox.profile_params(
            None, read_only=True, read_wall=True, extra_read_paths=[raw],
        )
        # Outside home: may not appear in R*, but the admitted extra is
        # the true target (assert via a direct last-resolver call).
        entries = kernel_sandbox._read_reallow_entries(None, extra_read_paths=[raw])
        admitted = [
            path for _kind, path in entries if path in (true_rp, lexical)
        ]
        self.assertNotIn(lexical, admitted)
        if admitted:
            self.assertEqual(admitted, [true_rp])
        # And the write-side last resolver (always emits the extra) names
        # the true target; read-side identity check used the same admit.
        self.assertEqual(os.path.realpath(raw), true_rp)
        self.assertNotIn(lexical, _r_values(params) | _p_values(
            kernel_sandbox.profile_params(
                None, read_only=True, extra_write_paths=[raw],
            )
        ))

    # --- S4(c) per-call-site no-spawn mapping ---

    def test_wrap_argv_raises_fence_path_rejected(self):
        d = self._dir("omni120a_wrap_d_")
        c = self._dir("omni120a_wrap_c_")
        os.rename(d, d + ".moved")
        os.symlink(c, d)
        self.addCleanup(_rmtree, d)
        if not kernel_sandbox.available():
            with mock.patch.object(kernel_sandbox, "available", return_value=True), \
                    mock.patch.object(
                        kernel_sandbox.shutil, "which",
                        return_value="/usr/bin/sandbox-exec",
                    ):
                with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                    kernel_sandbox.wrap_argv(
                        ["/bin/true"], None, True, extra_write_paths=[d],
                    )
        else:
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox.wrap_argv(
                    ["/bin/true"], None, True, extra_write_paths=[d],
                )
        self._assert_rejected(caught, d)

    def test_verify_fence_maps_fence_path_rejected(self):
        d = self._dir("omni120a_vf_d_")
        c = self._dir("omni120a_vf_c_")
        os.rename(d, d + ".moved")
        os.symlink(c, d)
        self.addCleanup(_rmtree, d)

        class _Proc:
            def __init__(self, stdout, returncode=0):
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = ""

        def _fake_probe(_argv, _env):
            return _Proc("WROTE\n")

        with mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(kernel_sandbox, "_run_probe", _fake_probe), \
                mock.patch.object(
                    kernel_sandbox, "_tmpdir_rejected", return_value=None
                ), \
                mock.patch.object(
                    kernel_sandbox, "_codex_home_rejected", return_value=None
                ):
            refusal = kernel_sandbox.verify_fence(
                None, read_only=True, extra_write_paths=[d],
            )
        self.assertIsInstance(refusal, dict)
        self.assertEqual(refusal.get("ok"), False)
        self.assertEqual(refusal.get("reason"), "fence_path_rejected")
        self.assertIn(d, refusal.get("detail") or "")

    def test_apply_kernel_fence_no_spawn_on_fence_path_rejected(self):
        base = self._dir("omni120a_akf_")
        real = os.path.join(base, "real")
        os.mkdir(real)
        link = os.path.join(base, "link")
        os.symlink(real, link)
        argv = ["/fake/claude", "-p"]
        env = {"PATH": "/usr/bin"}
        home = self._fake_home()
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home), \
                mock.patch.object(kernel_sandbox, "available", return_value=True):
            out_argv, _out_env, fence, _detail, refusal = (
                agent_lane._apply_kernel_fence(
                    "claude", argv, env, base, False, True,
                    config_dir=link,
                )
            )
        self.assertIs(out_argv, argv)
        self.assertFalse(kernel_sandbox.is_wrapped(out_argv))
        self.assertEqual(fence, "none")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.get("ok"), False)
        self.assertEqual(refusal.get("reason"), "fence_path_rejected")

    def test_consult_no_spawn_on_fence_path_rejected(self):
        from providers import codex as codex_mod

        d = self._dir("omni120a_consult_d_")
        c = self._dir("omni120a_consult_c_")
        os.rename(d, d + ".moved")
        os.symlink(c, d)
        self.addCleanup(_rmtree, d)
        spawned = []

        async def _fake_run_cli(*_a, **_k):
            spawned.append(True)
            return {
                "returncode": 0, "stdout": "NO", "stderr": "", "timed_out": False,
            }

        def _which(name, *a, **k):
            base = os.path.basename(str(name))
            if base == "codex":
                return "/fake/bin/codex"
            if base == "sandbox-exec":
                return "/usr/bin/sandbox-exec"
            return None

        with mock.patch.object(
                codex_mod, "resolve_profile",
                return_value=({"CODEX_HOME": d}, None),
            ), \
                mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(codex_mod, "run_cli", _fake_run_cli), \
                mock.patch.object(codex_mod.shutil, "which", _which), \
                mock.patch.object(codex_mod, "refuse_payload", return_value=None):
            env = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30,
            ))
        self.assertEqual(env.get("ok"), False, env)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertEqual(spawned, [])

    def test_agent_fence_policy_maps_fence_path_rejected(self):
        home = self._fake_home()
        d = os.path.join(home, ".claude-x")
        os.mkdir(d)
        os.symlink(self._dir("omni120a_pol_c_"), os.path.join(d, ".claude.json"))
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=d,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)

    # --- F6(a): wrap_argv raises after verify_fence returned None; no spawn ---

    def test_apply_kernel_fence_no_spawn_when_wrap_raises_after_verify(self):
        spawned = []

        def _fake_run_cli(*_a, **_k):
            spawned.append(True)
            raise AssertionError("run_cli must not be called")

        with mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(
                    kernel_sandbox, "agent_fence_policy",
                    return_value={
                        "wrap_kw": {"read_wall": True},
                        "tmpdir": None,
                        "vendor_state_writable": False,
                    },
                ), \
                mock.patch.object(
                    kernel_sandbox, "verify_fence", return_value=None,
                ), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    side_effect=kernel_sandbox.FencePathRejected(
                        "fence path rejected (changed): /tmp/admitted "
                        "-> /tmp/now"
                    ),
                ), \
                mock.patch.object(agent_lane, "run_cli", _fake_run_cli):
            out_argv, _out_env, fence, _detail, refusal = (
                agent_lane._apply_kernel_fence(
                    "claude", ["/fake/claude", "-p"], {"PATH": "/usr/bin"},
                    "/tmp/wd", False, True,
                )
            )
        self.assertEqual(out_argv, ["/fake/claude", "-p"])
        self.assertFalse(kernel_sandbox.is_wrapped(out_argv))
        self.assertEqual(fence, "none")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.get("ok"), False)
        self.assertEqual(refusal.get("reason"), "fence_path_rejected")
        self.assertEqual(spawned, [])

    def test_consult_no_spawn_when_wrap_raises_after_verify(self):
        from providers import codex as codex_mod

        spawned = []

        async def _fake_run_cli(*_a, **_k):
            spawned.append(True)
            return {
                "returncode": 0, "stdout": "NO", "stderr": "", "timed_out": False,
            }

        def _which(name, *a, **k):
            base = os.path.basename(str(name))
            if base == "codex":
                return "/fake/bin/codex"
            if base == "sandbox-exec":
                return "/usr/bin/sandbox-exec"
            return None

        with mock.patch.object(
                codex_mod, "resolve_profile", return_value=({}, None),
            ), \
                mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(
                    kernel_sandbox, "verify_fence", return_value=None,
                ), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    side_effect=kernel_sandbox.FencePathRejected(
                        "fence path rejected (changed): /tmp/admitted "
                        "-> /tmp/now"
                    ),
                ), \
                mock.patch.object(codex_mod, "run_cli", _fake_run_cli), \
                mock.patch.object(codex_mod.shutil, "which", _which), \
                mock.patch.object(codex_mod, "refuse_payload", return_value=None):
            env = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30,
            ))
        self.assertEqual(env.get("ok"), False, env)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertEqual(spawned, [])

    # --- F6(b): subpath given-spelling plantable ancestor ---

    def test_subpath_given_spelling_plantable_ancestor(self):
        base = self._dir("omni120a_f6b_")
        real = os.path.join(base, "real")
        os.mkdir(real)
        profile = os.path.join(real, "profile")
        os.mkdir(profile)
        link = os.path.join(base, "link")
        os.symlink(real, link)
        via = os.path.join(link, "profile")
        self.assertFalse(os.path.islink(via))
        with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
            kernel_sandbox.admit_path(
                via, "subpath", label="CLAUDE_CONFIG_DIR",
            )
        self._assert_rejected(caught, link)

        var_dir = tempfile.mkdtemp(prefix="omni120a_f6b_var_")
        self.addCleanup(_rmtree, var_dir)
        ap = kernel_sandbox.admit_path(
            var_dir, "subpath", label="CLAUDE_CONFIG_DIR",
        )
        self.assertTrue(ap.path.startswith("/private/var/"), ap.path)

    # --- F6(c): bare str extra calls admit_path exactly once ---

    def test_coerce_admitted_str_calls_admit_path_once(self):
        d = self._dir("omni120a_f6c_")
        with mock.patch.object(
                kernel_sandbox, "admit_path",
                wraps=kernel_sandbox.admit_path,
        ) as wrapped:
            ap = kernel_sandbox._coerce_admitted(d, label="test")
        self.assertEqual(wrapped.call_count, 1, wrapped.call_args_list)
        self.assertIsInstance(ap, kernel_sandbox.AdmittedPath)
        self.assertEqual(ap.kind, "subpath")
        self.assertEqual(ap.path, os.path.realpath(d))

    # --- F6(d): .claude.json symlink to a regular file ---

    def test_claude_json_symlink_to_regular_file_refused(self):
        home = self._fake_home()
        profile = os.path.join(home, ".claude-x")
        os.mkdir(profile)
        victim = self._file_in(self._dir("omni120a_f6d_"), "secret")
        claude_json = os.path.join(profile, ".claude.json")
        os.symlink(victim, claude_json)
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=None, read_only=False, config_dir=profile,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)

    # --- F6(e): confirmation kind mismatch with patched identity ---

    def test_confirm_admitted_kind_mismatch_refused(self):
        d = self._dir("omni120a_f6e_")
        ap = self._admit_dir(d)
        os.rmdir(d)
        with open(d, "w", encoding="utf-8") as fh:
            fh.write("x\n")
        self.addCleanup(lambda: os.path.isfile(d) and os.unlink(d))
        orig_stat = kernel_sandbox.os.stat

        def fake_stat(path, *args, **kwargs):
            st = orig_stat(path, *args, **kwargs)
            p = os.fspath(path)
            if isinstance(p, bytes):
                p = os.fsdecode(p)
            if os.path.normpath(p) == ap.path:
                seq = list(st)
                seq[1] = ap.ino
                seq[2] = ap.dev
                return os.stat_result(seq)
            return st

        with mock.patch.object(kernel_sandbox.os, "stat", side_effect=fake_stat):
            with self.assertRaises(kernel_sandbox.FencePathRejected) as caught:
                kernel_sandbox._confirm_admitted(ap, label="extra_write_paths")
        self._assert_rejected(caught, ap.path)

    # --- F6(f): firmlink spelling of ~/.codex is inside home ---

    def test_firmlink_codex_home_consult_refused_as_inside_home(self):
        home = kernel_sandbox.protected_home()
        firm = "/System/Volumes/Data" + home + "/.codex"
        try:
            st = os.stat(firm)
        except OSError as exc:
            self.skipTest("no firmlink spelling of ~/.codex (%s)" % exc)
        if not stat.S_ISDIR(st.st_mode):
            self.skipTest("firmlink spelling of ~/.codex is not a directory")
        ap = kernel_sandbox.admit_path(firm, "subpath", label="CODEX_HOME")
        self.assertEqual(ap.kind, "subpath")

        from providers import codex as codex_mod

        spawned = []

        async def _fake_run_cli(*_a, **_k):
            spawned.append(True)
            return {
                "returncode": 0, "stdout": "NO", "stderr": "", "timed_out": False,
            }

        def _which(name, *a, **k):
            base = os.path.basename(str(name))
            if base == "codex":
                return "/fake/bin/codex"
            if base == "sandbox-exec":
                return "/usr/bin/sandbox-exec"
            return None

        with mock.patch.object(
                codex_mod, "resolve_profile",
                return_value=({"CODEX_HOME": firm}, None),
            ), \
                mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(
                    kernel_sandbox, "verify_fence", return_value=None,
                ), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    side_effect=lambda argv, *a, **k: argv,
                ), \
                mock.patch.object(codex_mod, "run_cli", _fake_run_cli), \
                mock.patch.object(codex_mod.shutil, "which", _which), \
                mock.patch.object(codex_mod, "refuse_payload", return_value=None):
            env = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30,
            ))
        self.assertEqual(env.get("ok"), False, env)
        self.assertEqual(env.get("reason"), "fence_path_rejected", env)
        self.assertIn("inside the home tree", env.get("detail") or "")
        self.assertEqual(spawned, [])


if __name__ == "__main__":
    unittest.main()
