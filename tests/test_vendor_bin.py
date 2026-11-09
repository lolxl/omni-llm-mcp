"""vendor_bin.which skips harness launcher shims on PATH."""

import os
import tempfile
import unittest

from providers import vendor_bin


def _exe(path, body):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.chmod(path, 0o755)


class VendorBinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shims = os.path.join(self.tmp.name, "shims")
        self.real = os.path.join(self.tmp.name, "real")
        os.mkdir(self.shims)
        os.mkdir(self.real)
        _exe(os.path.join(self.shims, "codex"), "#!/bin/bash\n# mofuu-harness-shim\nexec launcher \"$@\"\n")

    def test_a_shim_ahead_on_path_is_skipped(self):
        _exe(os.path.join(self.real, "codex"), "#!/bin/sh\necho real\n")
        found = vendor_bin.which("codex", os.pathsep.join([self.shims, self.real]))
        self.assertEqual(found, os.path.join(self.real, "codex"))

    def test_only_a_shim_finds_nothing(self):
        self.assertIsNone(vendor_bin.which("codex", self.shims))

    def test_relative_and_empty_entries_are_ignored(self):
        _exe(os.path.join(self.real, "codex"), "#!/bin/sh\necho real\n")
        search = os.pathsep.join(["", ".", "real", self.real])
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(vendor_bin.which("codex", search), os.path.join(self.real, "codex"))

    def test_a_marker_anywhere_in_the_first_block_counts(self):
        _exe(os.path.join(self.real, "claude"), "#!/bin/bash\n" + "# x\n" * 50 + "# mofuu-harness-shim\n")
        self.assertTrue(vendor_bin.is_harness_shim(os.path.join(self.real, "claude")))
        self.assertIsNone(vendor_bin.which("claude", self.real))


if __name__ == "__main__":
    unittest.main()
