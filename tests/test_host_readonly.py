"""RO workdirs override containing temp grants, including in tool children."""
import errno
import json
import os
import subprocess
import sys
import tempfile
import unittest

from providers import host_fence, kernel_sandbox


_WRITE = """
import json, os, sys
try:
    with open(sys.argv[1], 'w') as handle:
        handle.write('synthetic fixture')
    print(json.dumps({'errno': 0}))
except OSError as exc:
    print(json.dumps({'errno': exc.errno}))
"""
_CHILD_WRITE = """
import subprocess, sys
result = subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]],
                        capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""


class ReadonlyProfileTests(unittest.TestCase):
    def test_final_readonly_deny_binds_canonical_workdir_after_postallow(self):
        with tempfile.TemporaryDirectory() as workdir:
            extra = {"extra_post_write": (("subpath", workdir),)}
            profile = kernel_sandbox.build_profile(workdir, True, **extra)
            params = kernel_sandbox.profile_params(workdir, True, **extra)
            self.assertEqual(params["Q0"], os.path.realpath(workdir))
            self.assertTrue(profile.endswith(
                '(deny file-write* (subpath (param "Q0")))'))
            self.assertLess(profile.index('(allow file-write* (subpath (param "T0"))'),
                            profile.index('(deny file-write* (subpath (param "Q0"))'))
            self.assertNotIn("Q0", kernel_sandbox.profile_params(workdir, False))
            self.assertNotIn('param "Q0"', kernel_sandbox.build_profile(workdir, False))


@unittest.skipUnless(kernel_sandbox.available(), "macOS Seatbelt required")
class ReadonlyHostTests(unittest.TestCase):
    def run_write(self, workdir, target, *, read_only, child=False, policy=None):
        command = [sys.executable, "-c", _WRITE, target]
        if child:
            command = [sys.executable, "-c", _CHILD_WRITE, _WRITE, target]
        argv, env, metadata = host_fence.prepare_host_fence(
            command, provider="pi", workdir=workdir, read_only=read_only,
            policy=policy)
        result = subprocess.run(argv, cwd=workdir, env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["errno"], metadata

    def test_temp_workdir_readonly_and_inherited_child_deny_with_rw_control(self):
        with tempfile.TemporaryDirectory() as workdir:
            target = os.path.join(workdir, "fixture.txt")
            # The exact same location is writable without the fence.
            control = subprocess.run([sys.executable, "-c", _WRITE, target],
                                     capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(control.stdout)["errno"], 0)
            for child in (False, True):
                with self.subTest(child=child):
                    denied, meta = self.run_write(
                        workdir, target, read_only=True, child=child)
                    self.assertEqual(denied, errno.EPERM)
                    self.assertFalse(meta["fence_detail"]["workdir_writable"])
                    allowed, meta = self.run_write(
                        workdir, target, read_only=False, child=child)
                    self.assertEqual(allowed, 0)
                    self.assertTrue(meta["fence_detail"]["workdir_writable"])

    def test_readonly_wins_over_overlapping_runtime_postallow(self):
        with tempfile.TemporaryDirectory() as workdir, \
                tempfile.TemporaryDirectory() as runtime:
            policy = {"wrap_kw": {"extra_post_write": (
                ("subpath", os.path.realpath(workdir)),
                ("subpath", os.path.realpath(runtime)),
            )}}
            denied, _meta = self.run_write(
                workdir, os.path.join(workdir, "denied.txt"),
                read_only=True, policy=policy)
            self.assertEqual(denied, errno.EPERM)
            allowed, _meta = self.run_write(
                workdir, os.path.join(runtime, "allowed.txt"),
                read_only=True, policy=policy)
            self.assertEqual(allowed, 0)


if __name__ == "__main__":
    unittest.main()
