#!/usr/bin/env python3
"""Run every offline test in the repository.

Run from any cwd: python3 tests/run_offline_tests.py
Discovers every tests/test_*.py module and runs them in one process. Cases
whose id ends in "_live" may call real vendor services and are excluded
(listed before the run), except the ones in OFFLINE_LIVE_NAMED, which carry
the suffix but run only local code or mocks.
Everything selected needs no network and no logged-in model service.
macOS fence tests need a host environment permitting sandbox-exec.
Exit status is 0 only when every selected case passed AND no case touched the
real account home: the run swaps in a synthetic home (tests/offline_home.py)
and records any admission or credential-file open that still resolves under
the real one.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import offline_home  # noqa: E402

# "_live" in these names means a mocked live probe or a local sandbox-exec probe,
# not a vendor call. Any other "_live" case stays excluded.
OFFLINE_LIVE_NAMED = (
    "tests.test_kernel_sandbox.TestReadWallLiveProbe.test_composed_home_content_reallow_is_refused_live",
    "tests.test_profiles.TestCheckProfiles.test_strict_quota_fails_cooling_and_skips_live",
    "tests.test_usage.TestPerBackendMapping.test_openai_shape_unverified_live",
)


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


class _TrackingResult(unittest.TextTestResult):
    """Tell the real-home recorder which case is running."""

    def startTest(self, test):
        offline_home.set_current(test.id())
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        offline_home.set_current(None)


def main(run=None):
    # Tells tests/test_offline_home_guard.py that the guard must be armed.
    os.environ["OMNI_OFFLINE_RUNNER"] = "1"
    rc = 1
    try:
        offline_home.install()
        rc = (run or _run)()
    finally:
        problems = offline_home.uninstall()
    if problems:
        print("FAILED: synthetic home cleanup:\n  " + "\n  ".join(problems),
              file=sys.stderr, flush=True)
        return 1
    return rc


def _run():
    suite = unittest.defaultTestLoader.discover(
        str(ROOT), pattern="test_*.py", top_level_dir=str(ROOT))
    selected, excluded = [], []
    for case in flatten(suite):
        if case.id().endswith("_live") and case.id() not in OFFLINE_LIVE_NAMED:
            excluded.append(case.id())
        else:
            selected.append(case)
    print("Excluded live-provider cases: " + ", ".join(excluded), flush=True)
    result = unittest.TextTestRunner(
        verbosity=1, resultclass=_TrackingResult,
    ).run(unittest.TestSuite(selected))
    leaked = offline_home.violations()
    if not offline_home.active():
        print("FAILED: the real-home guard was not armed during the run",
              file=sys.stderr, flush=True)
        return 1
    if leaked:
        print("FAILED: the run touched the real account home:\n"
              + offline_home.format_violations(leaked), file=sys.stderr, flush=True)
    return 0 if result.wasSuccessful() and not leaked else 1


if __name__ == "__main__":
    sys.exit(main())
