#!/usr/bin/env python3
"""Run provider contract tests without invoking logged-in model services.

Run from any cwd: python3 tests/run_host_consumer_tests.py
The ordinary test_server suite contains explicit live-provider tests; those
remain available separately and are not counted as verified by this runner.
macOS local Seatbelt tests require a host environment permitting sandbox-exec.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "tests.test_server", "tests.test_cursor", "tests.test_images",
        "tests.test_host_fence", "tests.test_host_worker_admission",
        "tests.test_grok_host_consumers"])
    selected, excluded = [], []
    for case in flatten(suite):
        if case.id().endswith("_live"):
            excluded.append(case.id())
        else:
            selected.append(case)
    print("Excluded live-provider cases: " + ", ".join(excluded), flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(selected))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
