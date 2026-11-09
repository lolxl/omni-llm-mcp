"""host-exec resolves a bare vendor name past harness launcher shims."""

import io
import json
import os
import tempfile
import unittest
from unittest import mock

from providers import host_fence
from providers import kernel_sandbox


def _exe(path, body):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.chmod(path, 0o755)


class HostExecShimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="omni_hexec_home_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.home, True))
        patcher = mock.patch.object(
            kernel_sandbox, "protected_home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.shims = os.path.join(self.tmp.name, "shims")
        self.real = os.path.join(self.tmp.name, "real")
        os.mkdir(self.shims)
        os.mkdir(self.real)
        _exe(os.path.join(self.shims, "grok"),
             "#!/bin/bash\n# mofuu-harness-shim\nexec launcher \"$@\"\n")
        self.workdir = os.path.join(self.tmp.name, "work")
        os.mkdir(self.workdir)

    def _run(self, head, path):
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"PATH": path}), \
                mock.patch.object(host_fence, "prepare_host_fence",
                                  return_value=(["/fake"], {}, {})) as prepare, \
                mock.patch.object(host_fence.os, "chdir"), \
                mock.patch.object(host_fence.os, "execvpe") as execvpe, \
                mock.patch.object(host_fence.sys, "stderr", err):
            rc = host_fence.main(["--provider", "grok",
                                  "--workdir", self.workdir,
                                  "--read-only", "--", head, "--version"])
        return rc, prepare, execvpe, err.getvalue()

    def test_shim_ahead_execs_the_real_binary(self):
        real = os.path.join(self.real, "grok")
        _exe(real, "#!/bin/sh\necho real\n")
        _rc, prepare, _execvpe, _err = self._run(
            "grok", os.pathsep.join([self.shims, self.real]))
        self.assertEqual(
            prepare.call_args.args[0],
            [os.path.realpath(real), "--version"])

    def test_only_shim_refuses_before_exec(self):
        rc, prepare, execvpe, err = self._run("grok", self.shims)
        self.assertEqual(rc, 126)
        self.assertFalse(prepare.called)
        self.assertFalse(execvpe.called)
        payload = json.loads(err)
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertEqual(
            payload["detail"],
            "grok not found on PATH outside harness launcher shims")
        self.assertNotIn(self.home, payload["detail"])

    def test_bare_other_command_is_unchanged(self):
        _rc, prepare, _execvpe, _err = self._run("cat", self.shims)
        self.assertEqual(prepare.call_args.args[0], ["cat", "--version"])


if __name__ == "__main__":
    unittest.main()
