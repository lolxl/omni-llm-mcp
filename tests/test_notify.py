# no-opencc
"""Acceptance tests — notify seam (issue #88).

Pinned contract:
  * notify() always calls log_event first (stderr JSON); stdout is never written;
  * an optional OMNI_LLM_NOTIFY_CMD sink receives the JSON payload on stdin;
  * missing binary / non-zero exit / timeout / any exception is swallowed after
    one notify_failed log_event; notify() never raises.
"""
import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-notify-")
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
os.environ.pop("OMNI_LLM_NOTIFY_QUIET", None)

from providers import notify  # noqa: E402


class NotifySeamTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(notify.ENV_NOTIFY_CMD, None)
        os.environ.pop(notify.ENV_NOTIFY_QUIET, None)

    def tearDown(self):
        os.environ.pop(notify.ENV_NOTIFY_CMD, None)
        os.environ.pop(notify.ENV_NOTIFY_QUIET, None)

    def test_log_event_always_first(self):
        order = []

        def fake_log(event, **fields):
            order.append(("log", event))

        def fake_run(*args, **kwargs):
            order.append("run")
            return mock.Mock(returncode=0)

        os.environ[notify.ENV_NOTIFY_CMD] = "/bin/true"
        with mock.patch.object(notify, "log_event", side_effect=fake_log), \
                mock.patch.object(notify.subprocess, "run", side_effect=fake_run):
            notify.notify("seat_flagged", model="m", repeat=1)
        self.assertEqual(order[0], ("log", "seat_flagged"))
        self.assertIn("run", order)
        self.assertLess(order.index(("log", "seat_flagged")), order.index("run"))

    def test_no_cmd_skips_sink(self):
        with mock.patch.object(notify, "log_event") as logged, \
                mock.patch.object(notify.subprocess, "run") as run:
            notify.notify("seat_flagged", model="m")
        logged.assert_called_once()
        self.assertEqual(logged.call_args.args[0], "seat_flagged")
        run.assert_not_called()

    def test_quiet_skips_sink_but_still_logs(self):
        os.environ[notify.ENV_NOTIFY_CMD] = "/tmp/fake-notify"
        os.environ[notify.ENV_NOTIFY_QUIET] = "1"
        with mock.patch.object(notify, "log_event") as logged, \
                mock.patch.object(notify.subprocess, "run") as run:
            notify.notify("seat_flagged", model="m")
        logged.assert_called_once()
        self.assertEqual(logged.call_args.args[0], "seat_flagged")
        run.assert_not_called()

    def test_sink_receives_json_on_stdin(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            captured["timeout"] = kwargs.get("timeout")
            captured["stdout"] = kwargs.get("stdout")
            return mock.Mock(returncode=0)

        os.environ[notify.ENV_NOTIFY_CMD] = "/tmp/fake-notify"
        with mock.patch.object(notify, "log_event"), \
                mock.patch.object(notify.subprocess, "run", side_effect=fake_run):
            notify.notify("seat_flagged", model="claude/opus", profile="p",
                          road="claude", source="quota", until=1, repeat=1)
        self.assertEqual(captured["argv"], ["/tmp/fake-notify"])
        self.assertEqual(captured["timeout"], notify.NOTIFY_TIMEOUT_S)
        self.assertIs(captured["stdout"], subprocess.DEVNULL)
        payload = json.loads(captured["input"].decode("utf-8"))
        self.assertEqual(payload["event"], "seat_flagged")
        self.assertEqual(payload["model"], "claude/opus")
        self.assertEqual(payload["repeat"], 1)

    def test_missing_binary_is_swallowed(self):
        os.environ[notify.ENV_NOTIFY_CMD] = "/no/such/omni-notify-sink"
        with mock.patch.object(notify, "log_event") as logged:
            notify.notify("seat_flagged", model="m")
        events = [c.args[0] for c in logged.call_args_list]
        self.assertEqual(events[0], "seat_flagged")
        self.assertIn("notify_failed", events)
        failed = [c.kwargs for c in logged.call_args_list
                  if c.args[0] == "notify_failed"]
        self.assertEqual(failed[0]["reason"], "missing")

    def test_nonzero_exit_is_swallowed(self):
        os.environ[notify.ENV_NOTIFY_CMD] = "/tmp/fake-notify"
        with mock.patch.object(notify, "log_event") as logged, \
                mock.patch.object(notify.subprocess, "run",
                                  return_value=mock.Mock(returncode=2)):
            notify.notify("seat_cleared", model="m")
        events = [c.args[0] for c in logged.call_args_list]
        self.assertIn("notify_failed", events)

    def test_timeout_is_swallowed(self):
        os.environ[notify.ENV_NOTIFY_CMD] = "/tmp/fake-notify"
        exc = subprocess.TimeoutExpired(cmd="/tmp/fake-notify",
                                        timeout=notify.NOTIFY_TIMEOUT_S)
        with mock.patch.object(notify, "log_event") as logged, \
                mock.patch.object(notify.subprocess, "run", side_effect=exc):
            notify.notify("seat_flagged", model="m")
        failed = [c.kwargs for c in logged.call_args_list
                  if c.args and c.args[0] == "notify_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["reason"], "timeout")

    def test_sink_exception_is_swallowed(self):
        os.environ[notify.ENV_NOTIFY_CMD] = "/tmp/fake-notify"
        with mock.patch.object(notify, "log_event") as logged, \
                mock.patch.object(notify.subprocess, "run",
                                  side_effect=RuntimeError("boom")):
            notify.notify("seat_flagged", model="m", repeat=1)
        failed = [c.kwargs for c in logged.call_args_list
                  if c.args and c.args[0] == "notify_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["reason"], "RuntimeError")

    def test_notify_writes_nothing_to_stdout(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notify.notify("seat_flagged", model="m", profile="default",
                          road="claude", source="rate_limit", until=1,
                          repeat=1)
        self.assertEqual(buf.getvalue(), "")

    def test_real_script_gets_payload(self):
        out = os.path.join(_TMP, "sink-out.json")
        script = os.path.join(_TMP, "sink.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(
                "import sys\n"
                "open(sys.argv[1], 'w', encoding='utf-8').write(sys.stdin.read())\n"
            )
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        os.environ[notify.ENV_NOTIFY_CMD] = "%s %s %s" % (
            sys.executable, script, out)
        notify.notify("seat_cleared", model="codex/x", profile="default",
                      road="codex", source="rate_limit", until=9, repeat=2)
        with open(out, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
        self.assertEqual(payload["event"], "seat_cleared")
        self.assertEqual(payload["model"], "codex/x")
        self.assertEqual(payload["repeat"], 2)


if __name__ == "__main__":
    unittest.main()
