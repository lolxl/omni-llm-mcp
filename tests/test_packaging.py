"""pyproject.toml keeps installing the ``omni`` CLI from a wheel.

No TOML parser is available on Python 3.7-3.10 without a dependency, so these checks
read the file as text. They pin the facts that break an install silently: the console
script target, the single-sourced version, and the sibling layout that
``omni_cli.py`` and the job runner assume.
"""
import os
import re
import unittest

import omni_cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        return fh.read()


class TestPackaging(unittest.TestCase):
    def test_console_script_targets_main(self):
        self.assertRegex(_pyproject(), r'(?m)^omni = "omni_cli:main"$')
        self.assertTrue(callable(omni_cli.main))

    def test_version_is_single_sourced(self):
        text = _pyproject()
        self.assertIn('dynamic = ["version"]', text)
        self.assertIn('version = { attr = "providers.version.__version__" }', text)
        self.assertNotRegex(text, r'(?m)^version = "')

    def test_flat_layout_keeps_cli_beside_providers(self):
        # dispatch_exec, model_catalog and jobs locate omni_cli.py as a sibling of the
        # providers package, so both must install at the same level.
        text = _pyproject()
        self.assertIn('py-modules = ["omni_cli"]', text)
        self.assertIn('packages = ["providers"]', text)

    def test_server_py_stays_out_of_the_wheel(self):
        # A top-level module named `server` would collide with unrelated packages;
        # the plugins and clones launch server.py by path.
        modules = re.search(r"(?m)^py-modules = \[(.*)\]$", _pyproject()).group(1)
        self.assertNotIn("server", modules)


if __name__ == "__main__":
    unittest.main()
