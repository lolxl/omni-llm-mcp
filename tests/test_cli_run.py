# no-opencc
"""P0 acceptance tests — run_cli must not deadlock on large stdin payloads.

Seam for issue #18 WP-0: cli_run._drive() historically wrote the whole stdin
payload (write + drain) BEFORE starting the stdout/stderr readers; a child that
echoes as it reads fills its stdout pipe, nobody drains it, and both sides block
until the wall-clock kill. These tests pin the fixed contract:

  * a 1 MiB payload through an echo child completes well inside the budget with
    every byte intact in BOTH directions;
  * a child that closes stdin early and exits cleanly must not surface a feeder
    error — its envelope stays rc=0, not an exception.
"""
import asyncio
import os
import resource
import sys
import tempfile
import time
import unittest
from unittest import mock

from providers.cli_run import run_cli


PUMP = "import sys, shutil; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)"
EARLY_EXIT = "import sys; sys.stdin.buffer.read(1); sys.stdout.write('ok')"
SLOW_COUNT = (
    "import sys,time; total=0; "
    "\nwhile True:\n"
    "    chunk=sys.stdin.buffer.read(1024 * 1024)\n"
    "    if not chunk: break\n"
    "    total += len(chunk)\n"
    "    time.sleep(0.05)\n"
    "sys.stdout.write(str(total))"
)


def _max_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss * 1024)


def _run(coro):
    return asyncio.run(coro)


class LargePayloadTest(unittest.TestCase):
    def test_1mib_echo_roundtrip_no_deadlock(self):
        payload = b"x" * (1024 * 1024)
        res = _run(run_cli(
            [sys.executable, "-u", "-c", PUMP],
            stdin_data=payload,
            timeout_s=30,
        ))
        self.assertFalse(res["timed_out"], "1 MiB echo hit the wall-clock kill "
                         "— stdin feed and stdout read are not concurrent")
        self.assertEqual(res["returncode"], 0)
        self.assertEqual(len(res["stdout"].encode()), len(payload))

    def test_child_closing_stdin_early_is_not_an_error(self):
        payload = b"y" * (256 * 1024)
        res = _run(run_cli(
            [sys.executable, "-u", "-c", EARLY_EXIT],
            stdin_data=payload,
            timeout_s=30,
        ))
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], 0)
        self.assertEqual(res["stdout"], "ok")


class StdinFileTest(unittest.TestCase):
    def test_stdin_file_echo_roundtrip(self):
        payload = b"z" * (1024 * 1024)
        with tempfile.NamedTemporaryFile() as input_file:
            input_file.write(payload)
            input_file.flush()
            res = _run(run_cli(
                [sys.executable, "-u", "-c", PUMP],
                stdin_file=input_file.name,
                timeout_s=30,
            ))
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], 0)
        self.assertEqual(len(res["stdout"].encode()), len(payload))
        self.assertEqual(res["stdout"].encode(), payload)

    def test_missing_stdin_file_is_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "missing-input")
            started = time.monotonic()
            with mock.patch("providers.cli_run._spawn", new_callable=mock.AsyncMock) as spawn:
                with self.assertRaises(FileNotFoundError):
                    _run(run_cli(
                        [sys.executable, "-u", "-c", "import time; time.sleep(10)"],
                        stdin_file=missing,
                        timeout_s=30,
                    ))
                elapsed = time.monotonic() - started
                self.assertEqual(spawn.await_count, 0)
        self.assertLess(elapsed, 2.0)

    def test_stdin_data_and_stdin_file_are_mutually_exclusive(self):
        with tempfile.NamedTemporaryFile() as input_file:
            with self.assertRaises(ValueError):
                _run(run_cli(
                    [sys.executable, "-u", "-c", PUMP],
                    stdin_data=b"data",
                    stdin_file=input_file.name,
                ))


class StdinFileBackpressureTest(unittest.TestCase):
    def setUp(self):
        fd, self.input_path = tempfile.mkstemp(prefix="cli-run-", suffix=".bin")
        with os.fdopen(fd, "wb") as input_file:
            chunk = b"b" * (1024 * 1024)
            for _ in range(64):
                input_file.write(chunk)

    def tearDown(self):
        if self.input_path and os.path.exists(self.input_path):
            os.unlink(self.input_path)

    def test_stdin_file_backpressure_memory_is_bounded(self):
        expected_bytes = 64 * 1024 * 1024
        started = time.monotonic()
        before = _max_rss_bytes()
        res = _run(run_cli(
            [sys.executable, "-u", "-c", SLOW_COUNT],
            stdin_file=self.input_path,
            timeout_s=15,
        ))
        elapsed = time.monotonic() - started
        after = _max_rss_bytes()

        self.assertLess(elapsed, 20.0)
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], 0)
        self.assertEqual(res["stdout"], str(expected_bytes))
        self.assertLess(after - before, 32 * 1024 * 1024)


class SignalFieldTest(unittest.TestCase):
    def test_killed_child_reports_signal_and_returncode(self):
        res = _run(run_cli(
            [sys.executable, "-u", "-c",
             "import os,sys; sys.stdout.write('partial-out'); "
             "sys.stdout.flush(); os.kill(os.getpid(), 9)"],
            timeout_s=10,
        ))
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], -9)
        self.assertEqual(res["signal"], 9)
        self.assertIn("partial-out", res["stdout"])

    def test_clean_exit_has_null_signal(self):
        res = _run(run_cli(
            [sys.executable, "-u", "-c", "print('ok')"],
            timeout_s=10,
        ))
        self.assertEqual(res["returncode"], 0)
        self.assertIsNone(res["signal"])

    def test_timeout_sentinel_has_null_signal(self):
        res = _run(run_cli(
            [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            timeout_s=0.3,
        ))
        self.assertTrue(res["timed_out"])
        self.assertEqual(res["returncode"], -1)
        self.assertIsNone(res["signal"])

    def test_sighup_is_not_the_timeout_sentinel(self):
        res = _run(run_cli(
            [sys.executable, "-u", "-c",
             "import os; os.kill(os.getpid(), 1)"],
            timeout_s=10,
        ))
        self.assertFalse(res["timed_out"])
        self.assertEqual(res["returncode"], -1)
        self.assertEqual(res["signal"], 1)


if __name__ == "__main__":
    unittest.main()

