#!/usr/bin/env python3
# no-opencc
"""omni dispatch-exec: header, canary, grok argv/trace, watchdog (grok stage of D).

Hermetic: the "grok" and the host-exec wrapper are small python scripts driven
by a JSON scenario. No vendor CLI, no network, no sandbox-exec; where the
Darwin wrapper is involved, ``dispatch_exec.host_exec_prefix`` is replaced by a
fake. ``D`` = the external dispatch-external.sh script, ``T`` =
tests/dispatch_external/selftest.sh; comments cite the D lines each assertion
mirrors.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import omni_cli  # noqa: E402
from providers import (dispatch_canary, dispatch_exec, dispatch_result,  # noqa: E402
                       dispatch_supervision)

_FAKE_CHILD = r'''
import json, os, subprocess, sys, time

GRANDCHILD = (
    "import os, signal, sys, time\n"
    "def term(*a):\n"
    "    open(sys.argv[2], 'a').write('term\\n')\n"
    "signal.signal(signal.SIGTERM, term)\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)

steps = json.load(open(sys.argv[1]))
rest = sys.argv[3:]
for st in steps:
    op = st["op"]
    if op == "out":
        os.write(1, bytes.fromhex(st["hex"]))
    elif op == "err":
        os.write(2, bytes.fromhex(st["hex"]))
    elif op == "sleep":
        time.sleep(st["s"])
    elif op == "exit":
        os._exit(st["code"])
    elif op == "hang":
        while True:
            time.sleep(60)
    elif op == "kill_self":
        os.kill(os.getpid(), st["sig"])
        time.sleep(60)
    elif op == "ignore_term":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif op == "term_exit":
        import signal
        def _bye(*a, path=st["path"]):
            open(path, "w").write("term")
            os._exit(0)
        signal.signal(signal.SIGTERM, _bye)
    elif op == "pidfile":
        open(st["path"], "w").write(str(os.getpid()))
    elif op == "argv":
        json.dump({"argv": rest, "cwd": os.getcwd(), "grok_home": os.environ.get("GROK_HOME"),
                   "env": dict(os.environ),
                   "stdin_null": os.fstat(0).st_rdev == os.stat(os.devnull).st_rdev},
                  open(st["path"], "w"))
    elif op == "err_every":
        for _ in range(st["count"]):
            os.write(2, bytes.fromhex(st["hex"]))
            time.sleep(st["every"])
    elif op == "out_every":
        while True:
            os.write(1, bytes.fromhex(st["hex"]))
            time.sleep(st["every"])
    elif op == "stop_group":
        import signal
        open(st["path"], "w").write(str(os.getpid()))
        os.killpg(os.getpgrp(), signal.SIGSTOP)
        while True:
            time.sleep(60)
    elif op == "grandchild":
        kw = {}
        if st.get("own_group"):
            kw["preexec_fn"] = lambda: os.setpgid(0, 0)
        subprocess.Popen([sys.executable, "-c", GRANDCHILD, st["pidfile"], st["termfile"]], **kw)
        while not os.path.exists(st["pidfile"]) or not open(st["pidfile"]).read():
            time.sleep(0.02)
'''

_FAKE_WRAPPER = r'''
import json, os, re, subprocess, sys, time

mode = sys.argv[1]
rest = sys.argv[3:]
if "DISPATCH_RO_CANARY" not in rest:
    # The vendor launch through the fake host-exec: a finished run.
    os.write(1, b'{"type":"result","is_error":false}\n')
    sys.exit(0)
at = rest.index("DISPATCH_RO_CANARY")
script, target = rest[at - 1], rest[at + 1]
nonce = re.search(r"nonce='([0-9a-f]+)'", open(script).read()).group(1)
info = os.environ.get("FAKE_CANARY_INFO")
if info:
    json.dump({"script_mode": oct(os.stat(script).st_mode & 0o777),
               "dir_mode": oct(os.stat(os.path.dirname(script)).st_mode & 0o777),
               "script": script, "target": target, "rest": rest,
               "pid": os.getpid()}, open(info, "w"))

def say(text):
    os.write(2, (text + "\n").encode())

ready = "dispatch-ro-canary: writing " + nonce
denial = "sh: %s: Operation not permitted" % target
if mode == "pass":
    say(ready); say(denial); sys.exit(23)
elif mode == "write":
    say(ready); open(target, "w").close(); sys.exit(0)
elif mode == "no_ready":
    say(denial); sys.exit(23)
elif mode == "no_denial":
    say(ready); sys.exit(23)
elif mode == "wrong_rc":
    say(ready); say(denial); sys.exit(1)
elif mode == "exit0_denied":
    say(ready); say(denial); sys.exit(0)
elif mode == "other_path":
    say(ready); say("sh: /elsewhere/file: Operation not permitted"); sys.exit(23)
elif mode == "marker_missing":
    say("dispatch-ro-canary: marker missing"); sys.exit(2)
elif mode == "hang":
    say(ready)
    while True:
        time.sleep(60)
'''

_ROOT = None
_FAKE_CHILD_PATH = None
_FAKE_WRAPPER_PATH = None

RESULT_LINE = b'{"type":"result","is_error":false}\n'
TIMINGS = dict(wall_s=60, grace_s=60, kill_grace_s=1, poll_s=1)
HEADER = b"dispatch-containment: layer=omni-host-exec sandbox=workspace canary=n/a canary_at=-\n"
INVOCATION_ID = "0123456789abcdef0123456789abcdef"
# The real seams, captured before any test patches them.
_REAL_PINNED_ENV = dispatch_exec.pinned_env
_REAL_PREFIX = dispatch_exec.host_exec_prefix
RECORD_KEYS = {"version", "invocation_id", "outcome", "detail", "spawned", "vendor_rc",
               "vendor_signal", "signal", "exit_code", "elapsed_ms", "marker"}


_TERMINATION = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
_HANDLERS_AT_START = {}


def _handlers():
    return {sig: signal.getsignal(sig) for sig in _TERMINATION}


def setUpModule():
    global _ROOT, _FAKE_CHILD_PATH, _FAKE_WRAPPER_PATH
    _HANDLERS_AT_START.update(_handlers())
    _ROOT = tempfile.mkdtemp(prefix="dispatch_exec_test_")
    _FAKE_CHILD_PATH = os.path.join(_ROOT, "fake_child.py")
    with open(_FAKE_CHILD_PATH, "w") as handle:
        handle.write(_FAKE_CHILD)
    _FAKE_WRAPPER_PATH = os.path.join(_ROOT, "fake_wrapper.py")
    with open(_FAKE_WRAPPER_PATH, "w") as handle:
        handle.write(_FAKE_WRAPPER)


def tearDownModule():
    shutil.rmtree(_ROOT, ignore_errors=True)
    # Guard: no test may leave SIG_IGN (or a gate handler) behind for later suites.
    leaked = {sig.name: _handlers()[sig] for sig in _TERMINATION
              if _handlers()[sig] != _HANDLERS_AT_START[sig]}
    for sig in _TERMINATION:
        signal.signal(sig, _HANDLERS_AT_START[sig])
    if leaked:
        raise AssertionError("HUP/INT/TERM handlers leaked by the suite: %r" % leaked)


def hx(data):
    return data.hex() if isinstance(data, bytes) else data.encode().hex()


def out(data):
    return {"op": "out", "hex": hx(data)}


def err(data):
    return {"op": "err", "hex": hx(data)}


def sleep(seconds):
    return {"op": "sleep", "s": seconds}


def exit_(code):
    return {"op": "exit", "code": code}


HANG = {"op": "hang"}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="case_", dir=_ROOT)
        self.work = os.path.join(self.dir, "work")
        os.mkdir(self.work)
        self.outfile = os.path.join(self.dir, "out.txt")
        self.prompt = os.path.join(self.dir, "prompt.txt")
        self.n = 0
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(self._restore_handlers, _handlers())
        # The pinned credential home, the scanned prompt and the record target.
        self.cfg = os.path.join(self.dir, "grokhome")
        os.mkdir(self.cfg)
        self.write_auth({"xai": {"key": "k-test-not-a-secret"}})
        self.prompt_raw = b"clean work order"
        with open(self.prompt, "wb") as handle:
            handle.write(self.prompt_raw)
        self.result = os.path.join(self.dir, "result.json")

    def _restore_handlers(self, before):
        # A test that installs handlers (a real SignalGate: arm, ignore) must not leak
        # them: put the previous HUP/INT/TERM back, then fail loudly if they differed.
        after = _handlers()
        for sig, old in before.items():
            signal.signal(sig, old)
        self.assertEqual(after, before, "test leaked HUP/INT/TERM handlers")

    def path(self, name):
        return os.path.join(self.dir, name)

    def write_auth(self, data, home=None):
        with open(os.path.join(home or self.cfg, "auth.json"), "w") as handle:
            json.dump(data, handle)

    def set_prompt(self, raw):
        self.prompt_raw = raw
        with open(self.prompt, "wb") as handle:
            handle.write(raw)

    def args(self, *extra, mode="workspace", parent=None, drop=()):
        """Every required flag; a later duplicate in ``extra`` overrides, ``drop`` removes."""
        base = {
            "--provider": "grok", "--model": "grok-4.6", "--effort": "xhigh",
            "--workdir": self.work, "--prompt-file": self.prompt,
            "--prompt-sha256": hashlib.sha256(self.prompt_raw).hexdigest(),
            "--outfile": self.outfile, "--invocation-id": INVOCATION_ID,
            "--result-file": self.result, "--login": "xai", "--config-dir": self.cfg,
            "--parent-pid": str(os.getppid() if parent is None else parent),
            "--wall-s": "60", "--stall-grace-s": "60", "--kill-grace-s": "1",
            "--poll-s": "1", "--canary-timeout-s": "20",
        }
        if mode is not None:
            base["--mode"] = mode
        for name in drop:
            del base[name]
        argv = [part for pair in base.items() for part in pair]
        return argv + list(extra)

    def record(self):
        return json.loads(self.slurp(self.result))

    def assertRecord(self, rc, outcome, detail, spawned, *, vendor_rc=None,
                     vendor_signal=None, signum=None, marker=False):
        """The published record has exactly the schema keys and these values, and its
        exit_code is the process rc; no temp file is left beside it."""
        record = self.record()
        self.assertEqual(set(record), RECORD_KEYS)
        elapsed = record.pop("elapsed_ms")
        self.assertIsInstance(elapsed, int)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(record, {
            "version": 1, "invocation_id": INVOCATION_ID, "outcome": outcome,
            "detail": detail, "spawned": spawned, "vendor_rc": vendor_rc,
            "vendor_signal": vendor_signal, "signal": signum, "exit_code": rc,
            "marker": marker})
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])
        self.assertEqual(os.stat(self.result).st_mode & 0o777, 0o600)

    def scenario(self, steps):
        self.n += 1
        path = self.path("scenario%d.json" % self.n)
        with open(path, "w") as handle:
            json.dump(steps, handle)
        return path

    def fake_argv(self, steps):
        return [sys.executable, _FAKE_CHILD_PATH, self.scenario(steps), "--"]

    def read_out(self):
        with open(self.outfile, "rb") as handle:
            return handle.read()

    def wait_dead(self, pid, within=5.0):
        end = time.monotonic() + within
        while time.monotonic() < end:
            try:
                os.kill(pid, 0)
            except OSError:
                return True
            time.sleep(0.05)
        return False

    def wait_file(self, path, within=10.0):
        end = time.monotonic() + within
        while time.monotonic() < end:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            time.sleep(0.02)
        return False

    def supervise(self, steps, header=HEADER, **kw):
        with open(self.outfile, "wb") as handle:
            handle.write(header)
        params = dict(wall_s=60, grace_s=60, kill_grace_s=1, poll_s=1)
        params.update(kw)
        errs = io.StringIO()
        self.ending = dispatch_supervision.supervise(
            self.fake_argv(steps), cwd=self.work, env=dict(os.environ),
            outfile=self.outfile, stderr=errs, **params)
        return self.ending.rc, self.read_out(), errs.getvalue()

    def watchdog_lines(self, data):
        # D appends the line with no leading newline, so it can follow an
        # unterminated vendor line (D 2028).
        return re.findall(r"dispatch-watchdog: [^\n]*", data.decode("utf-8", "replace"))

    def slurp(self, path, mode="r"):
        with open(path, mode) as handle:
            return handle.read()


class TestWatchdog(Base):
    def test_natural_exit_passes_status_and_raw_bytes_through(self):
        # D 1886-1891: the vendor status wins; nothing is added or masked. B2: the
        # child holds the outfile directly (D 1966 `> O 2>&1`), bytes untouched.
        for code in (0, 4):
            with self.subTest(code=code):
                rc, data, errs = self.supervise(
                    [out(b"hello \xff\x00 raw\n"), err(b"warn\n"), exit_(code)])
                self.assertEqual(rc, code)
                self.assertEqual(data, HEADER + b"hello \xff\x00 raw\nwarn\n")
                self.assertNotIn(b"dispatch-watchdog", data)
                self.assertEqual(errs, "")

    def test_signalled_child_returns_128_plus_signal(self):
        # D 1886: `if ($status & 127) { exit(128 + ($status & 127)); }`, no line.
        rc, data, errs = self.supervise([out(b"x\n"), {"op": "kill_self", "sig": 9}])
        self.assertEqual(rc, 128 + 9)
        self.assertEqual(self.watchdog_lines(data), [])
        self.assertEqual(errs, "")

    def test_vendor_exit_9_and_10_are_passthrough_without_a_line(self):
        # D 1888-1891, 2018-2020: a vendor that exits 9/10 by itself is not rewritten,
        # whatever the marker state.
        for code in (9, 10):
            for body in ([], [out(RESULT_LINE)]):
                with self.subTest(code=code, marker=bool(body)):
                    rc, data, errs = self.supervise(body + [exit_(code)])
                    self.assertEqual(rc, code)
                    self.assertEqual(self.watchdog_lines(data), [])
                    self.assertEqual(errs, "")

    def test_natural_exit_takes_no_descendant_snapshot(self):
        # D 1873-1875: no ps snapshot is rooted at a reaped child.
        with mock.patch.object(dispatch_supervision._Watch, "snapshot_descendants",
                               side_effect=AssertionError("snapshot on natural exit")):
            rc, _data, _errs = self.supervise([out(b"a\n"), exit_(0)])
        self.assertEqual(rc, 0)

    def test_marker_then_hang_fires_completed_then_hung_once(self):
        # D 1948-1950, 2019-2029, T 987-1006: rc 9, one identical line in the
        # outfile and on stderr.
        rc, data, errs = self.supervise([out(RESULT_LINE), HANG], grace_s=1)
        self.assertEqual(rc, 9)
        lines = self.watchdog_lines(data)
        self.assertEqual(len(lines), 1, data)
        self.assertRegex(lines[0] + "\n", r"^dispatch-watchdog: reason=completed-then-hung "
                                          r"elapsed=[0-9]+ marker=yes "
                                          r"active=[0-9]+ paused=[0-9]+\n$")
        self.assertTrue(data.endswith((lines[0] + "\n").encode()))
        self.assertTrue(data.startswith(HEADER + RESULT_LINE))
        self.assertEqual(errs, lines[0] + "\n")
        self.assertEqual(data.count(b"dispatch-watchdog:"), 1)

    def test_watchdog_line_keeps_the_prefix_and_appends_active_time(self):
        # D 2019/2027-2029: the prefix is still reason/elapsed/marker. Active time,
        # paused time and the ceiling mark are appended.
        line = dispatch_supervision.watchdog_line
        self.assertEqual(line("wall-deadline", 7, "no", 7, 0, False),
                         "dispatch-watchdog: reason=wall-deadline elapsed=7 "
                         "marker=no active=7 paused=0\n")
        self.assertEqual(line("completed-then-hung", 0, "yes", 0, 0, False),
                         "dispatch-watchdog: reason=completed-then-hung elapsed=0 "
                         "marker=yes active=0 paused=0\n")
        self.assertEqual(line("wall-deadline", 13, "no", 2, 11, True),
                         "dispatch-watchdog: reason=wall-deadline elapsed=13 "
                         "marker=no active=2 paused=11 ceiling=yes\n")

    def test_wall_deadline_without_marker_is_10(self):
        # D 1955, 2005-2006: no marker at the wall -> rc 10.
        rc, data, errs = self.supervise([out(b"progress, no result\n"), HANG], wall_s=2)
        self.assertEqual(rc, 10)
        lines = self.watchdog_lines(data)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^dispatch-watchdog: reason=wall-deadline "
                                  r"elapsed=[0-9]+ marker=no "
                                  r"active=[0-9]+ paused=[0-9]+$")
        self.assertEqual(errs, lines[0] + "\n")

    def test_marker_then_growth_past_wall_is_9_wall_deadline_yes(self):
        # D 1951-1954: growth keeps restarting the grace, the wall still fires, and a
        # marker seen means 9.
        rc, data, errs = self.supervise(
            [out(RESULT_LINE), {"op": "out_every", "hex": hx(b"tick\n"), "every": 0.3}],
            wall_s=2, grace_s=1)
        self.assertEqual(rc, 9)
        lines = self.watchdog_lines(data)
        self.assertEqual(len(lines), 1, data)
        self.assertRegex(lines[0], r"^dispatch-watchdog: reason=wall-deadline "
                                  r"elapsed=[0-9]+ marker=yes "
                                  r"active=[0-9]+ paused=[0-9]+$")
        self.assertEqual(errs, lines[0] + "\n")

    def test_grace_wins_a_simultaneous_wall_decision(self):
        # D 1948-1956: the grace test comes first. Both elapse before the first
        # boundary check that can see the marker (poll 3 s, grace 1 s, wall 2 s).
        rc, data, _errs = self.supervise([out(RESULT_LINE), HANG],
                                         wall_s=2, grace_s=1, poll_s=3)
        self.assertEqual(rc, 9)
        self.assertIn("reason=completed-then-hung", self.watchdog_lines(data)[0])

    def test_stderr_growth_after_marker_restarts_the_grace(self):
        # D 1856-1859: the clock is the outfile size, so stderr growth counts.
        # Without a restart the grace fires at ~2 s; with it, after the last write
        # (~2.4 s) plus the grace and the next poll boundary.
        rc, data, _errs = self.supervise(
            [out(RESULT_LINE),
             {"op": "err_every", "hex": hx(b"still working\n"), "every": 0.4, "count": 6},
             HANG], grace_s=1, poll_s=1)
        self.assertEqual(rc, 9)
        elapsed = int(re.search(r"elapsed=([0-9]+)", self.watchdog_lines(data)[0]).group(1))
        self.assertGreaterEqual(elapsed, 3)

    def test_growth_during_the_run_keeps_a_live_child_alive(self):
        # D 1856: a vendor that keeps writing after its marker is never fired on,
        # so its own exit (rc 0, no line) is what ends the run.
        rc, data, _errs = self.supervise(
            [out(RESULT_LINE),
             {"op": "err_every", "hex": hx(b"x\n"), "every": 0.4, "count": 8},
             exit_(0)], grace_s=1, poll_s=1)
        self.assertEqual(rc, 0)
        self.assertEqual(self.watchdog_lines(data), [])

    def test_natural_exit_in_the_poll_sleep_beats_an_elapsed_grace(self):
        # D 1963, T 1344-1366 (T12): the child exits after the grace has elapsed but
        # before the poll boundary; it is reaped inside the sleep and its status wins.
        rc, data, errs = self.supervise([out(RESULT_LINE), sleep(2.0), exit_(3)],
                                        grace_s=1, poll_s=3, wall_s=60)
        self.assertEqual(rc, 3)
        self.assertEqual(self.watchdog_lines(data), [])
        self.assertEqual(errs, "")

    def test_non_markers(self):
        # D 1843-1851: the merged outfile is scanned line by line; only a strictly
        # decoded JSON object with a top-level type "result" counts.
        cases = {
            "torn by interleaved stderr": [out(b'{"type":"re'), err(b"WARN: x\n"),
                                           out(b'sult"}\n')],
            "invalid utf-8": [out(b'{"type":"result","x":"\xff"}\n')],
            "nested type": [out(b'{"msg":{"type":"result"}}\n')],
            "unterminated final line": [out(b'{"type":"result"}')],
            "array": [out(b'[{"type":"result"}]\n')],
            "type not a string": [out(b'{"type":["result"]}\n')],
            "other type": [out(b'{"type":"assistant"}\n')],
            "not json": [out(b"type result\n")],
            # JSON::PP utf8 rejects an unmatched surrogate escape (D 1846).
            "lone surrogate": [out(b'{"type":"result","x":"\\ud800"}\n')],
        }
        for name, body in cases.items():
            with self.subTest(name):
                rc, data, errs = self.supervise(body + [HANG], wall_s=1, grace_s=1)
                self.assertEqual(rc, 10, data)
                self.assertIn("marker=no", self.watchdog_lines(data)[0])

    def test_lone_surrogate_anywhere_is_not_a_marker_but_a_pair_is(self):
        # D 1846 (JSON::PP utf8): an unmatched \\uD800-style escape fails the decode, in a
        # value, a key, or at depth; a properly paired escape is one code point.
        marker = dispatch_supervision.is_marker_line
        self.assertFalse(marker(b'{"type":"result","x":"\\ud800"}'))
        self.assertFalse(marker(b'{"type":"result","x":"\\udc00"}'))
        self.assertFalse(marker(b'{"type":"result","\\ud800":1}'))
        self.assertFalse(marker(b'{"type":"result","x":[{"y":["a\\ud83dz"]}]}'))
        self.assertFalse(marker(b'{"type":"result","x":"\\ude00\\ud83d"}'))
        self.assertTrue(marker(b'{"type":"result","x":"\\ud83d\\ude00"}'))
        self.assertTrue(marker('{"type":"result","x":"\U0001F600"}'.encode("utf-8")))

    def test_lone_surrogate_in_an_overwritten_duplicate_value_is_not_a_marker(self):
        # D 1846 (JSON::PP utf8): the decode of the line fails on the unmatched escape
        # even when a later duplicate key overwrites that value; a dict that collapses
        # duplicates first would hide it. JSON::PP keeps the last duplicate otherwise.
        marker = dispatch_supervision.is_marker_line
        self.assertFalse(marker(b'{"type":"result","x":"\\ud800","x":"ok"}'))
        self.assertFalse(marker(b'{"type":"result","x":["a\\ud800"],"x":1}'))
        self.assertFalse(marker(b'{"type":"result","x":{"y":"\\udc00"},"x":1}'))
        self.assertFalse(marker(b'{"type":"result","\\ud800":1,"\\ud800":2}'))
        self.assertFalse(marker(b'{"type":"result","x":[{"y":"\\ud800"}],"x":null}'))
        self.assertTrue(marker(b'{"type":"x","type":"result"}'))
        self.assertFalse(marker(b'{"type":"result","type":"x"}'))
        self.assertTrue(marker(b'{"type":"result","x":"\\ud83d\\ude00","x":"ok"}'))

    def test_lone_surrogate_line_then_hang_is_wall_deadline_without_marker(self):
        # End to end: the only result-looking line is invalid to D, so the hang is
        # rc 10 marker=no, never rc 9.
        rc, data, _errs = self.supervise(
            [out(b'{"type":"result","x":"\\ud800"}\n'), HANG], wall_s=1, grace_s=30)
        self.assertEqual(rc, 10, data)
        self.assertIn("reason=wall-deadline", self.watchdog_lines(data)[0])
        self.assertIn("marker=no", self.watchdog_lines(data)[0])

    def test_marker_after_other_lines_and_a_terminated_line_counts(self):
        # Positive control for test_non_markers (D 1843-1852), including a result
        # line completed by a later write.
        rc, data, _errs = self.supervise(
            [out(b'{"type":"assistant"}\n'), out(b'{"type":"resu'), sleep(0.3),
             out(b'lt"}\n'), HANG], grace_s=1)
        self.assertEqual(rc, 9)
        self.assertIn("marker=yes", self.watchdog_lines(data)[0])

    def test_header_bytes_are_not_scanned_as_vendor_output(self):
        # B3: the scan starts after the header.
        rc, data, _errs = self.supervise(
            [HANG], header=RESULT_LINE + HEADER, wall_s=1)
        self.assertEqual(rc, 10)
        self.assertIn("marker=no", self.watchdog_lines(data)[0])

    def test_descendant_ignoring_term_is_killed_after_the_kill_grace(self):
        # D 1789-1819 (terminate_group, watchdog firing): TERM the group and the
        # identity-checked descendants, KILL the survivors after the kill grace. The
        # grandchild sits in its own group and does not die on TERM.
        pidfile, termfile = self.path("gc.pid"), self.path("gc.term")
        rc, data, _errs = self.supervise(
            [{"op": "grandchild", "pidfile": pidfile, "termfile": termfile,
              "own_group": True}, out(RESULT_LINE), HANG],
            grace_s=1, kill_grace_s=1, poll_s=1)
        self.assertEqual(rc, 9)
        grandchild = int(self.slurp(pidfile))
        self.assertTrue(self.wait_dead(grandchild), "descendant survived the KILL")
        self.assertIn("term", self.slurp(termfile))  # TERM came first
        elapsed = int(re.search(r"elapsed=([0-9]+)", self.watchdog_lines(data)[0]).group(1))
        self.assertGreaterEqual(elapsed, 3)  # the kill grace is inside elapsed (D 1865)

    def test_cooperative_child_ends_on_term_before_the_kill_grace(self):
        # D 1789-1819 (terminate_group): TERM first, poll every 0.05 s while reaping,
        # stop early once the group is gone. A child that exits on TERM needs no KILL
        # and the firing does not wait out the kill grace. (Round 0 KILLed the group at
        # once: the child never saw TERM and group_kill ran.)
        termfile = self.path("child.term")
        with mock.patch.object(dispatch_supervision, "group_kill",
                               wraps=dispatch_supervision.group_kill) as killed:
            began = time.monotonic()
            rc, data, _errs = self.supervise(
                [{"op": "term_exit", "path": termfile}, out(RESULT_LINE), HANG],
                grace_s=1, kill_grace_s=20, poll_s=1)
            took = time.monotonic() - began
        self.assertEqual(rc, 9)
        self.assertEqual(self.slurp(termfile), "term")
        killed.assert_not_called()
        self.assertLess(took, 20 - 5, "waited out the kill grace for a child that had exited")
        elapsed = int(re.search(r"elapsed=([0-9]+)", self.watchdog_lines(data)[0]).group(1))
        self.assertLess(elapsed, 15)

    def test_term_ignoring_child_in_the_group_is_killed_after_the_kill_grace(self):
        # D 1789-1819: the group gets TERM, the child ignores it, and only after the
        # kill grace does the group KILL (and the pid KILL) land. A same-group
        # TERM-ignoring descendant is reached by the same group KILL.
        pidfile, termfile = self.path("c.pid"), self.path("gc.pid")
        rc, data, _errs = self.supervise(
            [{"op": "ignore_term"}, {"op": "pidfile", "path": pidfile},
             {"op": "grandchild", "pidfile": termfile, "termfile": self.path("gc.term")},
             out(RESULT_LINE), HANG], grace_s=1, kill_grace_s=2, poll_s=1)
        self.assertEqual(rc, 9)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived the KILL")
        self.assertTrue(self.wait_dead(int(self.slurp(termfile))), "group member survived")
        self.assertIn("term", self.slurp(self.path("gc.term")))  # TERM reached the group
        elapsed = int(re.search(r"elapsed=([0-9]+)", self.watchdog_lines(data)[0]).group(1))
        self.assertGreaterEqual(elapsed, 3)  # grace 1 + the whole kill grace 2 (D 1865)

    def test_signal_teardown_kills_at_once_but_terminate_group_waits_the_grace(self):
        # D 1894-1919 (on_signal: KILL the group at once) versus D 1789-1819
        # (terminate_group: TERM, then wait for the kill grace). Same TERM-ignoring child.
        def watch_for(argv):
            with open(self.outfile, "wb") as handle:
                handle.write(HEADER)
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, preexec_fn=dispatch_supervision.child_setup)
            self.addCleanup(lambda: proc.poll() is None and proc.kill())
            return dispatch_supervision._Watch(
                proc, mock.Mock(), self.outfile, 0, 60, 60, 2, 1,
                time.monotonic)

        for name, expect_wait in (("signal_teardown", False), ("terminate_group", True)):
            with self.subTest(teardown=name):
                pidfile = self.path("%s.pid" % name)
                watch = watch_for(self.fake_argv(
                    [{"op": "ignore_term"}, {"op": "pidfile", "path": pidfile}, HANG]))
                self.assertTrue(self.wait_file(pidfile))
                time.sleep(0.2)
                began = time.monotonic()
                getattr(watch, name)()
                took = time.monotonic() - began
                self.assertTrue(self.wait_dead(int(self.slurp(pidfile))))
                if expect_wait:
                    self.assertGreaterEqual(took, 1.8)  # kill grace 2
                else:
                    self.assertLess(took, 1.5)

    def test_natural_exit_kills_the_group_with_no_term(self):
        # D 1880: after a natural exit the group is KILLed outright, no TERM.
        pidfile, termfile = self.path("gc.pid"), self.path("gc.term")
        rc, _data, _errs = self.supervise(
            [{"op": "grandchild", "pidfile": pidfile, "termfile": termfile}, exit_(0)])
        self.assertEqual(rc, 0)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))))
        self.assertFalse(os.path.exists(termfile))

    def test_group_seen_then_gone_is_never_signalled_again(self):
        # D 1673-1683: a reused process-group id must not be killed.
        watch = dispatch_supervision._Watch(
            SimpleNamespace(pid=424242, returncode=None), mock.Mock(), self.outfile, 0,
            1, 1, 1, 1, time.monotonic)
        calls = []

        def killpg(pgid, sig):
            calls.append((pgid, sig))
            if len(calls) == 2:
                raise ProcessLookupError()

        with mock.patch("os.killpg", side_effect=killpg):
            self.assertTrue(watch.group_alive())
            self.assertFalse(watch.group_alive())
            self.assertFalse(watch.group_alive())  # would be True if the id was reused
        self.assertEqual(len(calls), 2)

    def test_reaped_child_pid_is_not_walked_or_signalled(self):
        # D 1534-1535, 1711, 1789, 1894: after the reap the pid is not snapshotted or killed.
        watch = dispatch_supervision._Watch(
            SimpleNamespace(pid=424242, returncode=0), mock.Mock(), self.outfile, 0,
            1, 1, 1, 1, time.monotonic)
        with mock.patch("subprocess.run", side_effect=AssertionError("ps")), \
                mock.patch("os.kill") as kill, \
                mock.patch("os.killpg", side_effect=ProcessLookupError()):
            self.assertEqual(watch.snapshot_descendants(), [])
            watch.terminate_group()
            watch.signal_teardown()
        kill.assert_not_called()

    def test_descendant_identity_mismatch_retires_the_record(self):
        # D 1745-1764: a pid whose lstart changed is never signalled.
        watch = dispatch_supervision._Watch(
            SimpleNamespace(pid=424242, returncode=None), mock.Mock(), self.outfile, 0,
            1, 1, 1, 1, time.monotonic)
        rec = dispatch_supervision._Descendant(999991, 999991, "Mon Jan  1 00:00:00 2001")
        with mock.patch.object(dispatch_supervision, "_ps_lstart",
                               return_value="Tue Feb  2 00:00:00 2002"), \
                mock.patch("os.kill") as kill, mock.patch("os.killpg") as killpg:
            watch._signal_descendant(signal.SIGTERM, rec)
            watch._signal_descendant(signal.SIGKILL, rec)
        kill.assert_not_called()
        killpg.assert_not_called()
        self.assertTrue(rec.retired)

    def test_signals_are_held_across_spawn_then_recorded(self):
        # D 1630, 1934: HUP/INT/TERM stay blocked until the handlers are installed, so
        # one sent in the spawn window is recorded instead of lost.
        with dispatch_supervision.SignalGate() as gate:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            self.assertIsNone(gate.pending)
            gate.arm()
            time.sleep(0.05)
            self.assertEqual(gate.pending, signal.SIGTERM)

    def signal_during_scan(self, steps, need_marker, **kw):
        """Deliver a pending TERM from the scan() of the iteration that would fire.

        The fake clock jumps past both the grace and the wall at that moment, so the
        round-2 loop (scan, then decide) fires the watchdog instead of the signal.
        """
        pidfile = self.path("scan-child-%d.pid" % self.n)
        steps = [{"op": "pidfile", "path": pidfile}] + steps
        state = {"sleeps": 0, "last": 0, "delivered": False, "offset": 0.0}
        real_sleep, real_scan = time.sleep, dispatch_supervision._Watch.scan

        def fake_sleep(seconds):
            state["sleeps"] += 1
            real_sleep(seconds)

        def scan(watch):
            real_scan(watch)
            # A scan with no sleep since the previous one is the top of the loop,
            # right before the grace and wall decisions; the others sit in the sleep.
            top = state["sleeps"] == state["last"]
            state["last"] = state["sleeps"]
            if (top and not state["delivered"] and state["sleeps"] >= 1
                    and (watch.saw_marker or not need_marker)):
                state["delivered"] = True
                state["offset"] = 10000.0
                watch.gate.pending = signal.SIGTERM

        fake_time = SimpleNamespace(sleep=fake_sleep, monotonic=time.monotonic)
        with mock.patch.object(dispatch_supervision, "time", fake_time), \
                mock.patch.object(dispatch_supervision._Watch, "scan", scan):
            result = self.supervise(
                steps, clock=lambda: time.monotonic() + state["offset"], **kw)
        self.assertTrue(state["delivered"])
        self.assertTrue(self.wait_file(pidfile))
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived")
        return result

    def test_signal_recorded_during_scan_beats_the_grace_decision(self):
        # D 1894-1920: the handler interrupts the loop at once, so a TERM recorded while
        # scanning wins over a grace that fires in the same iteration (round 2: rc 9).
        rc, data, errs = self.signal_during_scan(
            [out(RESULT_LINE), HANG], True, wall_s=60, grace_s=60, poll_s=1)
        self.assertEqual(rc, 143)
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertNotIn("dispatch-watchdog", errs)

    def test_signal_recorded_during_scan_beats_the_wall_decision(self):
        # Same, for the wall decision (round 2: rc 10).
        rc, data, errs = self.signal_during_scan(
            [out(b'{"type":"assistant"}\n'), HANG], False, wall_s=60, grace_s=60, poll_s=1)
        self.assertEqual(rc, 143)
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertNotIn("dispatch-watchdog", errs)

    @staticmethod
    def sigmask():
        return signal.pthread_sigmask(signal.SIG_BLOCK, [])

    def hung_after_marker(self, **kw):
        """A child that printed its result and hangs; the 1 s grace fires _fire()."""
        params = dict(wall_s=60, grace_s=1, kill_grace_s=1, poll_s=1)
        params.update(kw)
        return self.supervise([out(RESULT_LINE), HANG], **params)

    def test_signal_during_the_commit_window_is_held_then_discarded_like_d(self):
        # D 1789-1792, 1894-1920: once IGNORE is set a signal is discarded. _fire() blocks
        # HUP/INT/TERM across "read pending, switch to ignored", so a TERM sent inside
        # that window (hook on the switch) stays undelivered, never reaches the recorder,
        # and the run commits to the watchdog (rc 9, one line). Round 3 left the handlers
        # armed there: TERM was not blocked and the recorder ran at once.
        seen = {}
        real_ignore = dispatch_supervision.SignalGate.ignore

        def ignore(gate):
            if not seen:
                seen["blocked"] = set(self.sigmask())
                os.kill(os.getpid(), signal.SIGTERM)
                seen["pending_after_kill"] = gate.pending
                seen["kernel_pending"] = set(signal.sigpending())
            real_ignore(gate)

        before = self.sigmask()
        with mock.patch.object(dispatch_supervision.SignalGate, "ignore", ignore):
            rc, data, errs = self.hung_after_marker()
        self.assertTrue(set(dispatch_supervision.SIGNALS) <= seen["blocked"])
        self.assertIsNone(seen["pending_after_kill"])
        self.assertIn(signal.SIGTERM, seen["kernel_pending"])
        self.assertEqual(rc, 9)
        self.assertEqual(len(self.watchdog_lines(data)), 1)
        self.assertEqual(errs.count("dispatch-watchdog:"), 1)
        self.assertIn("reason=completed-then-hung", errs)
        self.assertNotIn(signal.SIGTERM, signal.sigpending())  # discarded, not delivered later
        self.assertEqual(self.sigmask(), before)

    def test_signal_landing_at_entry_to_the_commit_window_wins(self):
        # D 1894-1920: a TERM whose handler has not run yet when the window opens is run
        # by the blocking call itself, so the pending check sees it: rc 143, no line.
        # Round 3 never took the mask, so the hook never fired and it returned 9.
        fired = []
        real_mask = signal.pthread_sigmask

        def mask(how, sigs):
            handler = signal.getsignal(signal.SIGTERM)
            armed = getattr(handler, "__name__", "") == "_record"
            if armed and how == signal.SIG_BLOCK and not fired:
                fired.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return real_mask(how, sigs)

        before = self.sigmask()
        pidfile = self.path("entry-child.pid")
        with mock.patch.object(signal, "pthread_sigmask", mask):
            rc, data, errs = self.supervise(
                [{"op": "pidfile", "path": pidfile}, out(RESULT_LINE), HANG],
                wall_s=60, grace_s=1, kill_grace_s=1, poll_s=1)
        self.assertEqual(fired, [True])
        self.assertEqual(rc, 143)
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertNotIn("dispatch-watchdog", errs)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived")
        self.assertEqual(self.sigmask(), before)

    def test_signal_pending_before_the_check_wins_and_the_mask_is_restored(self):
        # The pending-before-the-check half of the same race: rc 143, no watchdog line,
        # and the process mask is what it was before supervise() started.
        before = self.sigmask()
        rc, data, errs = self.signal_during_scan(
            [out(RESULT_LINE), HANG], True, wall_s=60, grace_s=60, poll_s=1)
        self.assertEqual(rc, 143)
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertNotIn("dispatch-watchdog", errs)
        self.assertEqual(self.sigmask(), before)

    def test_mask_is_restored_when_the_commit_window_raises(self):
        # try/finally: an error raised while the signals are blocked must not leak the mask.
        real_ignore = dispatch_supervision.SignalGate.ignore
        state = {"raised": False}

        def ignore(gate):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("boom in the window")
            real_ignore(gate)

        before = self.sigmask()
        pidfile = self.path("raise-child.pid")
        with mock.patch.object(dispatch_supervision.SignalGate, "ignore", ignore):
            with self.assertRaises(RuntimeError):
                self.supervise(
                    [{"op": "pidfile", "path": pidfile}, out(RESULT_LINE), HANG],
                    wall_s=60, grace_s=1, kill_grace_s=1, poll_s=1)
        self.assertEqual(self.sigmask(), before)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived")

    def test_mask_equals_the_starting_mask_after_every_outcome(self):
        before = self.sigmask()
        for steps, kw in (([exit_(0)], {}),
                          ([out(RESULT_LINE), HANG], {"grace_s": 1}),
                          ([out(b'{"type":"x"}\n'), HANG], {"wall_s": 1})):
            with self.subTest(steps=steps):
                self.supervise(steps, **kw)
                self.assertEqual(self.sigmask(), before)

    def test_gate_ignore_then_exit_restores_the_previous_handlers(self):
        # Hygiene: ignore() installs SIG_IGN; leaving the gate must put back what was
        # there (the Base cleanup would also catch and undo a leak).
        before = _handlers()
        with dispatch_supervision.SignalGate() as gate:
            gate.arm()
            gate.ignore()
            self.assertEqual(set(_handlers().values()), {signal.SIG_IGN})
        self.assertEqual(_handlers(), before)

    def test_unlaunchable_child_raises_and_leaves_nothing(self):
        with open(self.outfile, "wb") as handle:
            handle.write(HEADER)
        with self.assertRaises(OSError):
            dispatch_supervision.supervise(
                [os.path.join(self.dir, "missing-binary")], cwd=self.work,
                env=dict(os.environ), outfile=self.outfile, **TIMINGS)
        self.assertEqual(self.read_out(), HEADER)


    def test_devnull_open_failure_closes_the_outfile_fd(self):
        # The second os.open (/dev/null) fails after the outfile fd is open: that fd must
        # be closed before the error propagates (round 2 leaked it).
        with open(self.outfile, "wb") as handle:
            handle.write(HEADER)
        real_open, real_close = os.open, os.close
        opened, closed = {}, []

        def fake_open(path, *args, **kw):
            if path == os.devnull:
                raise OSError(24, "Too many open files")
            fd = real_open(path, *args, **kw)
            opened[path] = fd
            return fd

        def fake_close(fd):
            closed.append(fd)
            real_close(fd)

        with mock.patch.object(os, "open", fake_open), mock.patch.object(os, "close", fake_close):
            with self.assertRaises(OSError):
                dispatch_supervision.supervise(
                    self.fake_argv([exit_(0)]), cwd=self.work, env=dict(os.environ),
                    outfile=self.outfile, **TIMINGS)
        self.assertIn(opened[self.outfile], closed)
        with self.assertRaises(OSError):
            os.fstat(opened[self.outfile])
        self.assertEqual(self.read_out(), HEADER)


    def test_ending_names_what_ended_the_run(self):
        # The Ending is what dispatch_exec turns into the result record.
        ended = dispatch_supervision.Ending
        for steps, kw, want in (
                ([exit_(0)], {}, ended("vendor_exit", 0, 0, None, None, False, True)),
                ([out(RESULT_LINE), exit_(4)], {},
                 ended("vendor_exit", 4, 4, None, None, True, True)),
                ([out(b"x\n"), {"op": "kill_self", "sig": 9}], {},
                 ended("vendor_exit", 137, None, 9, None, False, True)),
                ([exit_(9)], {}, ended("vendor_exit", 9, 9, None, None, False, True)),
                ([out(RESULT_LINE), HANG], {"grace_s": 1},
                 ended("stall", 9, None, None, None, True, True)),
                ([out(b"x\n"), HANG], {"wall_s": 1},
                 ended("wall_timeout", 10, None, None, None, False, True)),
                ([out(RESULT_LINE), out_every_steps()], {"wall_s": 2},
                 ended("wall_timeout", 9, None, None, None, True, True))):
            with self.subTest(steps=steps):
                self.supervise(steps, **kw)
                self.assertEqual(self.ending, want)

    def test_parent_death_tears_the_child_down_and_writes_no_line(self):
        # E3: the caller is gone (getppid differs from --parent-pid): same teardown as a
        # received signal, no watchdog line, outcome parent_died, rc 125.
        pidfile = self.path("pd-child.pid")
        grandfile = self.path("pd-gc.pid")
        real_ppid = os.getppid()

        def getppid():
            # The "parent" dies once the child tree is up.
            gone = os.path.exists(grandfile) and os.path.getsize(grandfile) > 0
            return real_ppid + 1 if gone else real_ppid

        with mock.patch.object(os, "getppid", getppid):
            rc, data, errs = self.supervise(
                [{"op": "pidfile", "path": pidfile},
                 {"op": "grandchild", "pidfile": grandfile,
                  "termfile": self.path("pd-gc.term"), "own_group": True}, HANG],
                parent_pid=real_ppid)
        self.assertEqual(self.ending, dispatch_supervision.Ending(
            "parent_died", 125, None, None, None, False, True))
        self.assertEqual(rc, 125)
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertNotIn("dispatch-watchdog", errs)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived")
        self.assertTrue(self.wait_dead(int(self.slurp(grandfile))), "descendant survived")

    def test_matching_parent_pid_changes_nothing(self):
        rc, data, _errs = self.supervise([out(RESULT_LINE), exit_(0)], parent_pid=os.getppid())
        self.assertEqual((rc, self.ending.outcome), (0, "vendor_exit"))
        self.assertEqual(data, HEADER + RESULT_LINE)

    def test_shared_gate_with_a_recorded_signal_launches_nothing(self):
        # E6: a signal recorded before the launch (between canary and launch) stops the
        # run before any child exists; the mask is back to what it was.
        pidfile = self.path("sg-child.pid")
        before = self.sigmask()
        with dispatch_supervision.SignalGate() as gate:
            gate.arm()
            gate.pending = signal.SIGHUP
            rc, data, errs = self.supervise(
                [{"op": "pidfile", "path": pidfile}, HANG], gate=gate)
        self.assertEqual(self.ending, dispatch_supervision.Ending(
            "signal", 129, None, None, signal.SIGHUP, False, False))
        self.assertEqual(data, HEADER)
        self.assertFalse(os.path.exists(pidfile))
        self.assertEqual(self.sigmask(), before)

    def test_shared_gate_with_a_dead_parent_launches_nothing(self):
        pidfile = self.path("sp-child.pid")
        with dispatch_supervision.SignalGate() as gate:
            gate.arm()
            self.supervise([{"op": "pidfile", "path": pidfile}, HANG], gate=gate,
                           parent_pid=os.getppid() + 1)
        self.assertEqual(self.ending, dispatch_supervision.Ending(
            "parent_died", 125, None, None, None, False, False))
        self.assertFalse(os.path.exists(pidfile))

    def test_shared_gate_run_ends_the_same_way_and_a_signal_wins(self):
        before = self.sigmask()
        with dispatch_supervision.SignalGate() as gate:
            gate.arm()
            self.supervise([out(RESULT_LINE), exit_(3)], gate=gate)
            self.assertEqual(self.ending.outcome, "vendor_exit")
            self.assertEqual(self.ending.vendor_rc, 3)
        pidfile = self.path("sw-child.pid")
        with dispatch_supervision.SignalGate() as gate:
            gate.arm()
            threading_kill = lambda: (self.wait_file(pidfile),
                                      os.kill(os.getpid(), signal.SIGTERM))
            import threading
            thread = threading.Thread(target=threading_kill)
            thread.start()
            rc, data, _errs = self.supervise(
                [{"op": "pidfile", "path": pidfile}, HANG], gate=gate)
            thread.join()
        self.assertEqual((rc, self.ending.outcome, self.ending.signal),
                         (143, "signal", signal.SIGTERM))
        self.assertNotIn(b"dispatch-watchdog", data)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "child survived")
        self.assertEqual(self.sigmask(), before)


def out_every_steps():
    return {"op": "out_every", "hex": hx(b"progress\n"), "every": 0.2}


DRIVER = r'''
import os, sys
sys.path.insert(0, %(root)r)
from providers import dispatch_exec
dispatch_exec.is_darwin = lambda: True
dispatch_exec.resolve_grok = lambda: "/x/grok"
dispatch_exec.host_exec_prefix = lambda provider, workdir, read_only: %(prefix)r
# The fake wrapper's side channel is injected after the env is built: the operator pass
# list is closed, so it cannot travel through it.
_real_pinned_env = dispatch_exec.pinned_env
def _pinned_env(home):
    env = _real_pinned_env(home)
    if "FAKE_CANARY_INFO" in os.environ:
        env["FAKE_CANARY_INFO"] = os.environ["FAKE_CANARY_INFO"]
    return env
dispatch_exec.pinned_env = _pinned_env
sys.exit(dispatch_exec.main(sys.argv[1:]))
'''

PARENT = r'''
import os, signal, subprocess, sys, time
root, driver, waitfile, logbase = sys.argv[1:5]
rest = sys.argv[5:]
proc = subprocess.Popen([sys.executable, driver] + rest + ["--parent-pid", str(os.getpid())],
                        stdout=open(logbase + ".out", "w"), stderr=open(logbase + ".err", "w"),
                        cwd=root)
print(proc.pid, flush=True)
end = time.monotonic() + 30
while time.monotonic() < end and not (os.path.exists(waitfile) and os.path.getsize(waitfile) > 0):
    time.sleep(0.02)
os.kill(os.getpid(), signal.SIGKILL)
'''

ENDED_OK = dispatch_supervision.Ending("vendor_exit", 0, 0, None, None, False, True)


_WATCHDOG_LINE = re.compile(
    r"^dispatch-watchdog: reason=(?P<reason>\S+) elapsed=(?P<elapsed>[0-9]+) "
    r"marker=(?P<marker>yes|no) active=(?P<active>[0-9]+) paused=(?P<paused>[0-9]+)"
    r"(?: ceiling=(?P<ceiling>yes))?$")


class _ScriptedClock:
    """Injected monotonic clock. ``sleep`` advances it by the requested slice and,
    once, by ``jump_by`` after a decision has observed ``t >= jump_after``."""

    def __init__(self, jump_after=None, jump_by=0.0, on_sleep=None, slice_s=0.005):
        self.t = 0.0
        self.jump_after = jump_after
        self.jump_by = jump_by
        self.jump_armed = False
        self.jumped = False
        self.on_sleep = on_sleep
        self.slice_s = slice_s
        self._real_sleep = time.sleep

    def __call__(self):
        if (self.jump_after is not None and not self.jump_armed and not self.jumped
                and self.t >= self.jump_after):
            self.jump_armed = True
        return self.t

    def sleep(self, seconds):
        self._real_sleep(self.slice_s)
        if self.on_sleep is not None:
            self.on_sleep(self)
        if self.jump_armed and not self.jumped:
            self.t += self.jump_by
            self.jumped = True
        self.t += seconds


def _bsd_short_status(pid):
    """``proc_bsdshortinfo.pbsi_status`` for ``pid`` (macOS). SSTOP is 4.

    Used instead of ``ps`` so a stopped child can be told apart from a sleeping one.
    """
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    class ShortInfo(ctypes.Structure):
        _fields_ = [
            ("pbsi_pid", ctypes.c_uint32),
            ("pbsi_ppid", ctypes.c_uint32),
            ("pbsi_pgid", ctypes.c_uint32),
            ("pbsi_status", ctypes.c_uint32),
            ("pbsi_comm", ctypes.c_char * 16),
            ("pbsi_flags", ctypes.c_uint32),
            ("pbsi_uid", ctypes.c_uint32),
            ("pbsi_gid", ctypes.c_uint32),
            ("pbsi_ruid", ctypes.c_uint32),
            ("pbsi_rgid", ctypes.c_uint32),
            ("pbsi_svuid", ctypes.c_uint32),
            ("pbsi_svgid", ctypes.c_uint32),
            ("pbsi_rfu", ctypes.c_uint32),
        ]

    proc_pidinfo = libc.proc_pidinfo
    proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                             ctypes.c_void_p, ctypes.c_int]
    proc_pidinfo.restype = ctypes.c_int
    info = ShortInfo()
    size = ctypes.sizeof(info)
    got = proc_pidinfo(pid, 13, 0, ctypes.byref(info), size)  # PROC_PIDT_SHORTBSDINFO
    if got != size:
        return None
    return info.pbsi_status


class TestActiveTime(Base):
    """Active-time budget. poll_s is 1, so a gap's cap is 2 seconds."""

    def parse_watchdog(self, data):
        lines = self.watchdog_lines(data)
        self.assertEqual(len(lines), 1, data)
        found = _WATCHDOG_LINE.match(lines[0])
        self.assertIsNotNone(found, lines[0])
        return found.groupdict()

    def scripted(self, steps, clock, **kw):
        log = []
        real_credit = dispatch_supervision._Watch._credit

        def credit(watch, now):
            prev = (watch.active, watch.paused, watch.quiet_active)
            real_credit(watch, now)
            log.append({"now": now, "active": watch.active, "paused": watch.paused,
                        "quiet": watch.quiet_active, "prev": prev, "t0": watch.t0})

        params = dict(poll_s=1, grace_s=60, kill_grace_s=0.2)
        params.update(kw)
        with mock.patch.object(dispatch_supervision.time, "sleep", clock.sleep), \
                mock.patch.object(dispatch_supervision._Watch, "_credit", credit):
            rc, data, errs = self.supervise(steps, clock=clock, **params)
        return rc, data, errs, log

    def test_whole_tree_pause_is_excluded_until_a_later_wall(self):
        # (a) WALL 6. After t≈1 the clock jumps 9 s between polls (the whole tree
        # was stopped). That poll must not fire. A later poll fires the wall, rc 10.
        clock = _ScriptedClock(jump_after=0.95, jump_by=9.0)
        rc, data, _errs, log = self.scripted([HANG], clock, wall_s=6)
        self.assertEqual(rc, 10)
        fields = self.parse_watchdog(data)
        self.assertEqual(fields["reason"], "wall-deadline")
        self.assertIsNone(fields["ceiling"])
        self.assertGreaterEqual(int(fields["paused"]), 6)
        self.assertGreaterEqual(int(fields["elapsed"]), 12)
        self.assertGreaterEqual(int(fields["active"]), 6)
        jumps = [i for i, row in enumerate(log) if row["paused"] - row["prev"][1] >= 6]
        self.assertEqual(len(jumps), 1, log)
        self.assertLess(jumps[0], len(log) - 1, log)
        self.assertLess(log[jumps[0]]["active"], 6)

    def test_worker_stopped_in_its_own_group_still_counts_as_active(self):
        # (b) SIGSTOP on the child's own group. The supervisor keeps running, so a
        # short wall fires and paused stays about 0.
        pidfile = self.path("stopped.pid")
        seen = []

        def watch_state():
            deadline = time.monotonic() + 8
            pid = None
            while time.monotonic() < deadline and pid is None:
                try:
                    text = self.slurp(pidfile).strip()
                except OSError:
                    text = ""
                if text:
                    pid = int(text)
                    break
                time.sleep(0.02)
            if pid is None:
                return
            while time.monotonic() < deadline:
                status = _bsd_short_status(pid)
                if status == 4:  # SSTOP
                    seen.append(status)
                    return
                if status is None:
                    return
                time.sleep(0.05)

        thread = threading.Thread(target=watch_state)
        thread.start()
        try:
            rc, data, _errs = self.supervise(
                [{"op": "stop_group", "path": pidfile}, HANG],
                wall_s=2, grace_s=30, kill_grace_s=0.3, poll_s=1)
        finally:
            thread.join(timeout=5)
        self.assertEqual(rc, 10)
        fields = self.parse_watchdog(data)
        self.assertEqual(fields["reason"], "wall-deadline")
        self.assertLessEqual(int(fields["paused"]), 1)
        self.assertTrue(seen, "child group was never observed stopped")

    def test_no_pause_fires_at_the_wall_with_paused_zero(self):
        # (c) No gap over the cap. The wall fires and paused is 0.
        clock = _ScriptedClock()
        rc, data, _errs, log = self.scripted([HANG], clock, wall_s=6)
        self.assertEqual(rc, 10)
        fields = self.parse_watchdog(data)
        self.assertEqual(fields["reason"], "wall-deadline")
        self.assertEqual(int(fields["paused"]), 0)
        self.assertGreaterEqual(int(fields["active"]), 6)
        self.assertIsNone(fields["ceiling"])
        self.assertTrue(log)
        self.assertEqual(log[-1]["paused"], 0.0)

    def test_ceiling_fires_on_the_jump_before_the_active_wall(self):
        # (d) WALL 3. Active is already within one cap of the wall, then the clock
        # jumps 13 s. Both predicates hold; the ceiling is first, so the line is
        # ceiling=yes on that poll.
        clock = _ScriptedClock(jump_after=1.95, jump_by=13.0)
        rc, data, _errs, log = self.scripted([HANG], clock, wall_s=3)
        self.assertEqual(rc, 10)
        fields = self.parse_watchdog(data)
        self.assertEqual(fields["reason"], "wall-deadline")
        self.assertEqual(fields["ceiling"], "yes")
        self.assertGreaterEqual(log[-1]["now"] - log[-1]["t0"], 4 * 3)
        self.assertGreaterEqual(log[-1]["active"], 3)
        self.assertEqual(clock.jumped, True)
        earlier = log[:-1]
        self.assertTrue(earlier)
        self.assertLess(earlier[-1]["active"], 3)
        self.assertLess(earlier[-1]["now"] - earlier[-1]["t0"], 4 * 3)

    def test_grace_ignores_a_jump_longer_than_the_grace(self):
        # (e) Marker seen, then a jump longer than the grace. That poll is not
        # completed-then-hung; the grace fires only after enough later active time.
        holder = []
        real_init = dispatch_supervision._Watch.__init__

        def init(watch, *args, **kwargs):
            real_init(watch, *args, **kwargs)
            holder.append(watch)

        def on_sleep(clock):
            if holder and holder[0].saw_marker and not clock.jumped and clock.t >= 0.5:
                clock.t += 10.0
                clock.jumped = True

        clock = _ScriptedClock(on_sleep=on_sleep)
        with mock.patch.object(dispatch_supervision._Watch, "__init__", init):
            rc, data, _errs, log = self.scripted(
                [out(RESULT_LINE), HANG], clock, wall_s=100, grace_s=5)
        self.assertEqual(rc, 9)
        fields = self.parse_watchdog(data)
        self.assertEqual(fields["reason"], "completed-then-hung")
        self.assertEqual(fields["marker"], "yes")
        self.assertIsNone(fields["ceiling"])
        self.assertTrue(clock.jumped)
        jumps = [i for i, row in enumerate(log) if row["paused"] - row["prev"][1] >= 6]
        self.assertEqual(len(jumps), 1, log)
        self.assertLess(log[jumps[0]]["quiet"], 5)
        self.assertLess(jumps[0], len(log) - 1)
        self.assertGreaterEqual(log[-1]["quiet"], 5)

    def test_backward_clock_step_adds_nothing(self):
        # (f) A poll whose clock is behind the previous poll adds nothing.
        pidfile = self.path("back.pid")

        class Backward(_ScriptedClock):
            def __init__(self):
                super().__init__(slice_s=0.01)
                self.n = 0
                self.retreated = False
                self.killed = False

            def __call__(self):
                self.n += 1
                if self.n == 4:
                    self.t -= 3.0
                    self.retreated = True
                return self.t

            def sleep(self, seconds):
                self._real_sleep(self.slice_s)
                self.t += seconds
                if self.retreated and not self.killed and os.path.exists(pidfile):
                    try:
                        with open(pidfile) as handle:
                            pid = int(handle.read())
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ValueError):
                        return
                    self.killed = True

        clock = Backward()
        rc, data, errs, log = self.scripted(
            [{"op": "pidfile", "path": pidfile}, HANG], clock, wall_s=60, grace_s=60)
        self.assertEqual(rc, 128 + 9, (data, errs))
        self.assertEqual(self.watchdog_lines(data), [])
        self.assertTrue(clock.killed)
        backs = [i for i in range(1, len(log)) if log[i]["now"] < log[i - 1]["now"]]
        self.assertEqual(len(backs), 1, log)
        row = log[backs[0]]
        self.assertEqual(row["active"], row["prev"][0])
        self.assertEqual(row["paused"], row["prev"][1])
        self.assertEqual(row["active"], log[backs[0] - 1]["active"])
        self.assertEqual(row["paused"], log[backs[0] - 1]["paused"])


class Driven(Base):
    """Runs dispatch_exec.main in a child python with the fake wrapper as host-exec."""

    def driver(self, prefix):
        path = self.path("driver-%d.py" % self.n)
        self.n += 1
        with open(path, "w") as handle:
            handle.write(DRIVER % {"root": HERE, "prefix": prefix})
        return path

    def vendor_prefix(self, steps):
        return [sys.executable, _FAKE_CHILD_PATH, self.scenario(steps), "--"]

    def canary_prefix(self, mode="hang"):
        return [sys.executable, _FAKE_WRAPPER_PATH, mode, "--"]

    def env(self):
        info = self.path("canary-info.json")
        env = dict(os.environ, FAKE_CANARY_INFO=info)
        return env, info


class TestSignals(Driven):
    def test_received_signal_tears_the_child_down_without_a_watchdog_line(self):
        # D 1894-1919, 1581: HUP/INT/TERM -> child gone, 128+signal, no note line.
        for signum, expect in ((signal.SIGTERM, 143), (signal.SIGHUP, 129),
                               (signal.SIGINT, 130)):
            with self.subTest(signal=signum.name):
                pidfile = self.path("child-%d.pid" % signum)
                grandfile = self.path("gc-%d.pid" % signum)
                steps = [
                    {"op": "pidfile", "path": pidfile},
                    {"op": "grandchild", "pidfile": grandfile,
                     "termfile": self.path("gc-%d.term" % signum), "own_group": True},
                    HANG]
                self.outfile = self.path("sig-out-%d.txt" % signum)
                self.result = self.path("sig-result-%d.json" % signum)
                proc = subprocess.Popen(
                    [sys.executable, self.driver(self.vendor_prefix(steps))]
                    + self.args(parent=os.getpid()),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE)
                try:
                    self.assertTrue(self.wait_file(pidfile) and self.wait_file(grandfile))
                    time.sleep(0.3)
                    child = int(self.slurp(pidfile))
                    grandchild = int(self.slurp(grandfile))
                    proc.send_signal(signum)
                    stdout, stderr = proc.communicate(timeout=30)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.communicate()
                self.assertEqual(proc.returncode, expect, stderr)
                self.assertTrue(self.wait_dead(child), "child survived")
                self.assertTrue(self.wait_dead(grandchild), "descendant survived")
                data = self.slurp(self.outfile, "rb")
                self.assertNotIn(b"dispatch-watchdog", data)
                self.assertNotIn(b"dispatch-watchdog", stderr)
                self.assertNotIn(b"dispatch-watchdog", stdout)
                # B2: the supervisor's own stdio never reaches the outfile.
                self.assertNotIn(b"dispatch-external", data)
                # E1: a received signal is a signal ending; the wrapper may have started
                # the vendor, so spawned is null.
                self.assertRecord(expect, "signal", "none", None, signum=int(signum))

    def test_term_during_the_canary_is_a_signal_ending_with_nothing_spawned(self):
        # E6: the first signal decides, even while the canary probe runs. The probe is
        # torn down, no header is written, the vendor is never launched, spawned false.
        env, info = self.env()
        proc = subprocess.Popen(
            [sys.executable, self.driver(self.canary_prefix("hang"))]
            + self.args(mode="read-only", parent=os.getpid()),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE, env=env)
        try:
            self.assertTrue(self.wait_file(info))
            probe = json.loads(self.slurp(info))["pid"]
            time.sleep(0.3)
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 143, stderr)
        self.assertTrue(self.wait_dead(probe), "canary probe survived")
        self.assertFalse(os.path.exists(self.outfile))
        self.assertEqual(stdout, b"")
        self.assertEqual([n for n in os.listdir(self.work)], [])
        self.assertRecord(143, "signal", "none", False, signum=int(signal.SIGTERM))


class TestParentDeath(Driven):
    def fake_parent(self, driver, waitfile, argv, env=None):
        """Starts omni under a parent that SIGKILLs itself once ``waitfile`` has content."""
        script = self.path("parent.py")
        with open(script, "w") as handle:
            handle.write(PARENT)
        parent = subprocess.Popen(
            [sys.executable, script, HERE, driver, waitfile, self.path("omni-log")] + argv,
            stdout=subprocess.PIPE, env=env)
        self.addCleanup(parent.stdout.close)
        omni_pid = int(parent.stdout.readline())
        self.addCleanup(self.kill_quietly, omni_pid)
        self.assertEqual(parent.wait(timeout=60), -signal.SIGKILL)
        return omni_pid

    @staticmethod
    def kill_quietly(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    def test_parent_death_mid_run_tears_the_vendor_group_down(self):
        # E3: the caller is SIGKILLed while the vendor runs. omni notices at its next poll
        # slice, kills the group and descendants, writes no watchdog line and records
        # parent_died (spawned null: the wrapper had started the vendor).
        pidfile, grandfile = self.path("pd.pid"), self.path("pdgc.pid")
        steps = [{"op": "pidfile", "path": pidfile},
                 {"op": "grandchild", "pidfile": grandfile,
                  "termfile": self.path("pdgc.term"), "own_group": True}, HANG]
        omni = self.fake_parent(self.driver(self.vendor_prefix(steps)), grandfile, self.args())
        self.assertTrue(self.wait_file(self.result, within=30), "no record published")
        self.assertRecord(125, "supervisor_failure", "parent_died", None)
        self.assertTrue(self.wait_dead(int(self.slurp(pidfile))), "vendor group survived")
        self.assertTrue(self.wait_dead(int(self.slurp(grandfile))), "descendant survived")
        self.assertTrue(self.wait_dead(omni), "omni itself did not exit")
        self.assertNotIn(b"dispatch-watchdog", self.slurp(self.outfile, "rb"))
        self.assertEqual(self.slurp(self.path("omni-log.err")), "")

    def test_parent_death_during_the_canary(self):
        # E3: the caller dies while the canary probe runs: the probe is torn down, nothing
        # is spawned (false), no header, no probe leftovers.
        env, info = self.env()
        omni = self.fake_parent(self.driver(self.canary_prefix("hang")), info,
                                self.args(mode="read-only"), env=env)
        self.assertTrue(self.wait_file(self.result, within=30), "no record published")
        self.assertRecord(125, "supervisor_failure", "parent_died", False)
        self.assertTrue(self.wait_dead(json.loads(self.slurp(info))["pid"]),
                        "canary probe survived")
        self.assertTrue(self.wait_dead(omni))
        self.assertFalse(os.path.exists(self.outfile))
        self.assertEqual(os.listdir(self.work), [])


class TestHeaderArgvTrace(Base):
    def test_header_bytes_match_ds_printf(self):
        # D 1496: 'dispatch-containment: layer=%s sandbox=%s canary=%s canary_at=%s\n';
        # layer omni-host-exec (D 1491-1495; the only layer left, off macOS is refused);
        # T/test_routing 1566, 1618 expectations.
        self.assertEqual(
            dispatch_exec.header_line("read-only", "pass", "2026-01-02T03:04:05Z"),
            "dispatch-containment: layer=omni-host-exec sandbox=read-only canary=pass "
            "canary_at=2026-01-02T03:04:05Z\n")
        self.assertEqual(
            dispatch_exec.header_line("workspace", "n/a", "-"),
            "dispatch-containment: layer=omni-host-exec sandbox=workspace canary=n/a "
            "canary_at=-\n")

    def test_grok_argv(self):
        # D 2078 (host-contained: --sandbox off), T 333, 1167; argv[0] is the checked path.
        self.assertEqual(
            dispatch_exec.grok_argv("/abs/grok", "grok-4.6", "xhigh", "clean work order"),
            ["/abs/grok", "--sandbox", "off", "--always-approve", "--reasoning-effort",
             "xhigh", "-m", "grok-4.6", "--output-format", "streaming-messages-json",
             "-p", "clean work order"])

    def test_trace_darwin(self):
        # D 2085-2101: omni host-exec ... [--read-only ]-- every arg quoted but the
        # prompt, which is replaced by "$(cat "P")".
        argv = dispatch_exec.grok_argv("grok", "grok-4.6", "xhigh", "ignored")
        quoted = ("'grok' '--sandbox' 'off' '--always-approve' '--reasoning-effort' "
                  "'xhigh' '-m' 'grok-4.6' '--output-format' 'streaming-messages-json' '-p' ")
        tail = '"$(cat "/tmp/p.txt")" </dev/null > "/tmp/o.txt" 2>&1\n'
        self.assertEqual(
            dispatch_exec.trace_text("workspace", "/w/dir", argv, "/tmp/p.txt", "/tmp/o.txt"),
            "omni host-exec --provider 'grok' --workdir '/w/dir' -- " + quoted + tail)
        self.assertEqual(
            dispatch_exec.trace_text("read-only", "/w/dir", argv, "/tmp/p.txt", "/tmp/o.txt"),
            "omni host-exec --provider 'grok' --workdir '/w/dir' --read-only -- "
            + quoted + tail)

    def test_quoting_helpers(self):
        # D 1012-1019: `'` -> `'\''`; ", \, $ and ` escaped inside double quotes.
        self.assertEqual(dispatch_exec.sh_quote("o'neil"), "'o'\\''neil'")
        self.assertEqual(dispatch_exec.dq_escape('a"b$c`d\\e'), 'a\\"b\\$c\\`d\\\\e')

    def test_prompt_is_returned_byte_for_byte(self):
        # E4: nothing is dropped or stripped; inner newlines, trailing spaces and
        # non-ASCII text survive, and the raw bytes come back with the text.
        for raw in (b"line1\n\nline2 ", "日本語 'q' \"d\" $x `y`\nz".encode(),
                    b"", b"-p"):
            with self.subTest(raw=raw):
                path = self.path("p")
                with open(path, "wb") as handle:
                    handle.write(raw)
                got = dispatch_exec.read_prompt(path, hashlib.sha256(raw).hexdigest())
                self.assertEqual(got.raw, raw)
                self.assertEqual(os.fsencode(got.text), raw)

    def test_prompt_that_is_not_the_scanned_one_or_not_canonical_is_refused(self):
        # E4: a digest mismatch, a NUL, a trailing LF and invalid UTF-8 are each refused
        # (the old code dropped NULs and stripped newlines instead).
        good = b"scanned prompt"
        for name, raw, sha in (
                ("sha mismatch", good, hashlib.sha256(b"other").hexdigest()),
                ("nul", b"a\0b", None), ("nul last", b"task\n\0", None),
                ("trailing lf", b"task\n", None), ("trailing lfs", b"a\n\n", None),
                ("invalid utf-8", b"bad \xff byte", None),
                ("encoded surrogate", b"\xed\xa0\x80", None)):
            with self.subTest(name):
                path = self.path("p")
                with open(path, "wb") as handle:
                    handle.write(raw)
                with self.assertRaises(dispatch_exec.PromptMismatch):
                    dispatch_exec.read_prompt(path, sha or hashlib.sha256(raw).hexdigest())

    def test_host_exec_prefix_is_ds_wrapper(self):
        # D 2104-2106, B1: <python> omni_cli.py host-exec --provider grok --workdir W
        # [--read-only] --
        # G1c: -E (PYTHON* ignored) and -s (no user site) come before the script.
        cli = os.path.join(HERE, "omni_cli.py")
        self.assertEqual(
            _REAL_PREFIX("grok", "/w", False),
            [sys.executable, "-E", "-s", cli, "host-exec", "--provider", "grok",
             "--workdir", "/w", "--"])
        self.assertEqual(
            _REAL_PREFIX("grok", "/w", True),
            [sys.executable, "-E", "-s", cli, "host-exec", "--provider", "grok",
             "--workdir", "/w", "--read-only", "--"])

    def test_resolve_grok_is_the_function_host_exec_rewrites_its_head_with(self):
        # host_fence._vendor_exec_path calls kernel_sandbox.resolve_agent_binary.
        with mock.patch.object(dispatch_exec.kernel_sandbox, "resolve_agent_binary",
                               return_value="/real/grok") as resolver:
            self.assertEqual(dispatch_exec.resolve_grok(), "/real/grok")
        resolver.assert_called_once_with("grok")


class MainBase(Base):
    """Drives dispatch_exec.main with the fake wrapper standing in for host-exec."""

    def setUp(self):
        super().setUp()
        self.prefix_calls = []
        self.wrapper_mode = "pass"
        self.steps = [out(RESULT_LINE), exit_(0)]
        self.info = self.path("canary-info.json")
        self.scripts = self.path("script-parent")
        os.mkdir(self.scripts)
        self.darwin = True
        self.grok = "/x/grok"  # what resolve_grok answers
        for patcher in (
                mock.patch.object(dispatch_exec, "resolve_grok", lambda: self.grok),
                mock.patch.object(dispatch_exec, "is_darwin", lambda: self.darwin),
                mock.patch.object(dispatch_exec, "host_exec_prefix", self.fake_prefix),
                mock.patch.object(dispatch_exec, "pinned_env", self.seam_pinned_env),
                mock.patch.object(dispatch_canary, "_script_parents",
                                  lambda: [self.scripts])):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_prefix(self, provider, workdir, read_only):
        self.prefix_calls.append((provider, workdir, read_only))
        return [sys.executable, _FAKE_WRAPPER_PATH, self.wrapper_mode, "--"]

    def seam_pinned_env(self, home):
        # The fake wrapper's side channel is injected after the env is built. The child
        # env is cut to the allowlist with the operator pass list closed, so the channel
        # cannot (and must not) go through OMNI_LLM_CHILD_ENV_PASS.
        env = _REAL_PINNED_ENV(home)
        env["FAKE_CANARY_INFO"] = self.info
        return env

    def run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = dispatch_exec.main(argv)
        return rc, stdout.getvalue(), stderr.getvalue()

    def canary_leftovers(self):
        return [name for name in os.listdir(self.work) if name.startswith("dispatch-ro-canary")]

    def use_child(self, steps):
        # The vendor launch through the fake wrapper ignores scenarios; for the
        # launch tests the wrapper itself is the fake child.
        self.prefix_scenario = self.scenario(steps)

        def prefix(provider, workdir, read_only):
            self.prefix_calls.append((provider, workdir, read_only))
            return [sys.executable, _FAKE_CHILD_PATH, self.prefix_scenario, "--"]
        patcher = mock.patch.object(dispatch_exec, "host_exec_prefix", prefix)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertNothingSpawned(self):
        self.assertEqual(self.prefix_calls, [])
        self.assertFalse(os.path.exists(self.outfile))


class TestMainLaunch(MainBase):
    def test_darwin_workspace_launch_header_trace_and_argv(self):
        # D 1496, 2085-2106: header first, trace on stdout, child = wrapper + grok argv
        # with cwd = workdir, stdin /dev/null, raw output after the header.
        argvfile = self.path("argv.json")
        self.use_child([{"op": "argv", "path": argvfile}, out(RESULT_LINE), exit_(0)])
        rc, stdout, stderr = self.run_main(self.args())
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(self.read_out(), HEADER + RESULT_LINE)
        seen = json.loads(self.slurp(argvfile))
        # E2: argv[0] is the resolved absolute binary, so host-exec execs the checked file.
        self.assertEqual(seen["argv"], dispatch_exec.grok_argv(
            "/x/grok", "grok-4.6", "xhigh", "clean work order"))
        self.assertEqual(os.path.realpath(seen["cwd"]), os.path.realpath(self.work))
        self.assertEqual(seen["grok_home"], os.path.realpath(self.cfg))
        self.assertTrue(seen["stdin_null"])
        self.assertEqual(self.prefix_calls, [("grok", os.path.abspath(self.work), False)])
        self.assertEqual(stdout, dispatch_exec.trace_text(
            "workspace", os.path.abspath(self.work), ["grok"] + seen["argv"][1:],
            self.prompt, self.outfile))
        self.assertRecord(0, "vendor_exit", "none", None, vendor_rc=0, marker=True)

    def test_relative_config_dir_resolves_against_the_callers_cwd(self):
        # The child runs with cwd = workdir, so GROK_HOME must name the caller-cwd
        # directory, not a same-named one under the workdir.
        os.mkdir(os.path.join(self.dir, "cfg"))
        os.mkdir(os.path.join(self.work, "cfg"))  # the decoy a relative path would hit
        self.write_auth({"xai": {"key": "k-test"}}, os.path.join(self.dir, "cfg"))
        self.write_auth({"xai": {"key": "k-decoy"}}, os.path.join(self.work, "cfg"))
        before = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, before)
        seen_env = {}

        def fake_supervise(argv, *, cwd, env, **_kw):
            seen_env.update(env)
            return ENDED_OK

        with mock.patch.object(dispatch_exec, "supervise", fake_supervise):
            rc, _stdout, stderr = self.run_main(self.args("--config-dir", "cfg"))
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(seen_env["GROK_HOME"], os.path.realpath(os.path.join(self.dir, "cfg")))
        self.assertTrue(os.path.isabs(seen_env["GROK_HOME"]))
        self.assertNotEqual(seen_env["GROK_HOME"],
                            os.path.realpath(os.path.join(self.work, "cfg")))

    def test_mode_defaults_read_only_on_darwin(self):
        # D 1433-1436.
        self.use_child([out(RESULT_LINE), exit_(0)])
        calls = []
        with mock.patch.object(dispatch_exec, "run_read_only_canary",
                               side_effect=lambda **kw: calls.append(kw) or
                               dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")):
            rc, _out, stderr = self.run_main(self.args(mode=None))
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.read_out().startswith(
            b"dispatch-containment: layer=omni-host-exec sandbox=read-only canary=pass "
            b"canary_at=2026-01-02T03:04:05Z\n"))
        self.assertEqual(self.prefix_calls[-1][2], True)  # launch carries --read-only

    def test_watchdog_through_main_adds_one_line_and_returns_9(self):
        # D 1948, 2019-2029 end to end: header + raw + one trailer, rc 9, stderr copy.
        self.use_child([out(RESULT_LINE), HANG])
        rc, _stdout, stderr = self.run_main(self.args("--stall-grace-s", "1", "--poll-s", "1"))
        self.assertEqual(rc, 9)
        lines = [ln for ln in self.read_out().decode().splitlines()
                 if ln.startswith("dispatch-watchdog:")]
        self.assertEqual(len(lines), 1)
        self.assertEqual(stderr, lines[0] + "\n")
        self.assertRecord(9, "stall", "none", None, marker=True)

    def test_record_for_every_supervised_ending(self):
        # E1: one record per run; outcome/detail/spawned/vendor_rc/vendor_signal/marker and
        # exit_code (== the rc main returns) for each way the supervised child can end.
        cases = (
            ("natural 0", [out(RESULT_LINE), exit_(0)], [], 0,
             ("vendor_exit", "none", None), dict(vendor_rc=0, marker=True)),
            ("natural 4", [out(b"x\n"), exit_(4)], [], 4,
             ("vendor_exit", "none", None), dict(vendor_rc=4)),
            ("vendor 9 is passthrough", [exit_(9)], [], 9,
             ("vendor_exit", "none", None), dict(vendor_rc=9)),
            ("vendor 10 is passthrough", [exit_(10)], [], 10,
             ("vendor_exit", "none", None), dict(vendor_rc=10)),
            ("vendor killed by 9", [out(b"x\n"), {"op": "kill_self", "sig": 9}], [], 137,
             ("vendor_exit", "none", None), dict(vendor_signal=9)),
            ("stall", [out(RESULT_LINE), HANG], ["--stall-grace-s", "1"], 9,
             ("stall", "none", None), dict(marker=True)),
            ("wall, no marker", [out(b"x\n"), HANG], ["--wall-s", "1"], 10,
             ("wall_timeout", "none", None), dict()),
            ("wall, marker", [out(RESULT_LINE), out_every_steps()], ["--wall-s", "2"], 9,
             ("wall_timeout", "none", None), dict(marker=True)),
            ("wrapper 126", [exit_(126)], [], 126,
             ("supervisor_failure", "start_unknown", None), dict()),
            ("wrapper 127", [exit_(127)], [], 127,
             ("supervisor_failure", "start_unknown", None), dict()),
        )
        for name, steps, extra, rc_expect, (outcome, detail, spawned), kw in cases:
            with self.subTest(name):
                self.result = self.path("res-%s.json" % name.replace(" ", "-").replace(",", ""))
                self.use_child(steps)
                rc, _out, stderr = self.run_main(self.args(*extra, "--poll-s", "1"))
                self.assertEqual(rc, rc_expect, stderr)
                self.assertRecord(rc, outcome, detail, spawned, **kw)

    def test_launch_oserror_is_start_unknown_with_rc_127(self):
        # D 1639: `exec @ARGV or exit 127`; the wrapper never started, so spawned is null.
        patcher = mock.patch.object(dispatch_exec, "host_exec_prefix",
                                    lambda *a: [os.path.join(self.dir, "nope"), "--"])
        patcher.start()
        self.addCleanup(patcher.stop)
        rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 127)
        self.assertIn("cannot launch grok", stderr)
        self.assertRecord(127, "supervisor_failure", "start_unknown", None)

    def test_unexpected_exception_is_an_internal_failure_with_rc_125(self):
        # E1: spawned is false before the vendor spawn and null from the spawn on.
        with mock.patch.object(dispatch_exec, "pinned_env", side_effect=RuntimeError("boom")):
            rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 125)
        self.assertIn("internal error: RuntimeError: boom", stderr)
        self.assertRecord(125, "supervisor_failure", "internal", False)
        os.unlink(self.result)
        with mock.patch.object(dispatch_exec, "supervise", side_effect=RuntimeError("late")):
            rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 125)
        self.assertRecord(125, "supervisor_failure", "internal", None)

    def test_missing_grok_is_rc_4_before_canary_and_header(self):
        # D 1004-1006: `command -v "$binary"` is checked before the header and before
        # the canary; the capacity-incident message, rc 4, launch_unavailable.
        self.grok = None
        for mode in ("read-only", "workspace"):
            with self.subTest(mode=mode):
                with open(self.outfile, "w") as handle:
                    handle.write("keep")
                if os.path.exists(self.result):
                    os.unlink(self.result)
                with mock.patch.object(dispatch_exec, "run_read_only_canary") as canary:
                    rc, stdout, stderr = self.run_main(self.args(mode=mode))
                self.assertEqual(rc, 4)
                self.assertEqual(
                    stderr, "capacity incident: grok binary not found; model=grok-4.6; "
                            "foreman fallback required per 51 §9\n")
                self.assertEqual(stdout, "")
                canary.assert_not_called()
                self.assertEqual(self.prefix_calls, [])
                self.assertEqual(self.slurp(self.outfile), "keep")  # untouched
                self.assertRecord(4, "refused_before_spawn", "launch_unavailable", False)

    def test_a_relative_resolution_is_not_a_resolved_binary(self):
        self.grok = "grok"
        rc, _out, _err = self.run_main(self.args())
        self.assertEqual(rc, 4)
        self.assertRecord(4, "refused_before_spawn", "launch_unavailable", False)
        self.assertNothingSpawned()


class TestCredentialPin(MainBase):
    UNPROVEN = "omni dispatch-exec: credential source unproven for resolved login\n"

    def assertUnproven(self, argv):
        rc, stdout, stderr = self.run_main(argv)
        self.assertEqual(rc, 6, stderr)
        self.assertEqual(stderr, self.UNPROVEN)
        self.assertEqual(stdout, "")
        self.assertNothingSpawned()
        self.assertRecord(6, "refused_before_spawn", "credential_unproven", False)

    def test_non_darwin_refuses_every_mode_before_anything_is_spawned(self):
        # E2: no direct launch of grok off macOS, whatever the mode; the old rc 7
        # read-only message is gone.
        self.darwin = False
        for mode in ("read-only", "workspace", "strict", None):
            with self.subTest(mode=mode):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                with mock.patch.object(dispatch_exec, "supervise") as launched, \
                        mock.patch.object(subprocess, "Popen") as popen:
                    self.assertUnproven(self.args(mode=mode))
                launched.assert_not_called()
                popen.assert_not_called()

    def test_login_other_than_xai_is_refused(self):
        for login in ("openai", "XAI", "", "xai "):
            with self.subTest(login=login):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.assertUnproven(self.args("--login", login))

    def test_config_dir_without_a_provable_login_is_refused(self):
        # E2 step 1: existing directory, auth.json a regular file holding a JSON object with
        # a non-empty string key / refresh_token (known_secrets' grok shape).
        empty = os.path.join(self.dir, "empty-home")
        os.mkdir(empty)
        as_dir = os.path.join(self.dir, "dir-home")
        os.makedirs(os.path.join(as_dir, "auth.json"))
        linked = os.path.join(self.dir, "link-home")
        os.mkdir(linked)
        with open(os.path.join(self.dir, "real-auth.json"), "w") as handle:
            json.dump({"xai": {"key": "k"}}, handle)
        os.symlink(os.path.join(self.dir, "real-auth.json"), os.path.join(linked, "auth.json"))
        bad = {"missing dir": os.path.join(self.dir, "nope"), "no auth.json": empty,
               "auth.json is a dir": as_dir, "auth.json is a symlink": linked,
               "a file, not a dir": self.prompt}
        for text in ("{not json", "[]", "null", '"key"', "{}", '{"xai": {}}',
                     '{"xai": {"key": ""}}', '{"xai": {"key": "  "}}',
                     '{"xai": {"key": 5}}', '{"xai": {"refresh_token": null}}',
                     '{"xai": ["key"]}', '{"key": ""}', "\xff"):
            home = self.path("home-%d" % len(bad))
            os.mkdir(home)
            with open(os.path.join(home, "auth.json"), "w", errors="surrogateescape") as handle:
                handle.write(text)
            bad["auth.json " + repr(text)] = home
        for name, home in bad.items():
            with self.subTest(name):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.assertUnproven(self.args("--config-dir", home))

    def test_either_login_shape_is_accepted(self):
        for data in ({"xai": {"key": "k"}}, {"https://auth.x.ai": {"refresh_token": "r"}},
                     {"key": "k"}, {"refresh_token": "r"}):
            with self.subTest(data=data):
                self.write_auth(data)
                if os.path.exists(self.result):
                    os.unlink(self.result)
                with mock.patch.object(dispatch_exec, "supervise", return_value=ENDED_OK):
                    self.assertEqual(self.run_main(self.args())[0], 0)

    def test_oversized_or_foreign_auth_json_is_not_a_proof(self):
        with open(os.path.join(self.cfg, "auth.json"), "w") as handle:
            handle.write('{"xai": {"key": "k"}, "pad": "%s"}' % ("x" * (1 << 20)))
        self.assertUnproven(self.args())
        os.unlink(self.result)
        self.write_auth({"xai": {"key": "k"}})
        with mock.patch.object(dispatch_exec.os, "getuid", return_value=os.getuid() + 1):
            self.assertIsNone(dispatch_exec.credential_home(self.cfg))

    def test_credential_is_proven_again_just_before_the_spawn(self):
        # E2 step 4: the file is gone (or unreadable) by the time the canary has finished.
        real = dispatch_exec.credential_home
        calls = []

        def second_check_fails(config_dir):
            calls.append(config_dir)
            return real(config_dir) if len(calls) == 1 else None

        with mock.patch.object(dispatch_exec, "credential_home", second_check_fails), \
                mock.patch.object(dispatch_exec, "supervise") as launched:
            rc, _out, stderr = self.run_main(self.args())
        self.assertEqual((rc, len(calls)), (6, 2))
        self.assertEqual(stderr, self.UNPROVEN)
        launched.assert_not_called()
        self.assertFalse(os.path.exists(self.outfile))  # refused before the header
        self.assertRecord(6, "refused_before_spawn", "credential_unproven", False)

    def test_child_env_is_the_allowlist_without_vendor_auth_and_with_the_pinned_home(self):
        # E2 steps 2-3: auth/endpoint variables in the parent env never reach the child
        # (even when OMNI_LLM_CHILD_ENV_PASS lists them), the pass channel itself is gone,
        # GROK_HOME is the realpath of --config-dir, argv[0] is the resolved binary.
        link = self.path("home-link")
        os.symlink(self.cfg, link)
        argvfile = self.path("env.json")
        self.use_child([{"op": "argv", "path": argvfile}, out(RESULT_LINE), exit_(0)])
        polluted = {
            "XAI_API_KEY": "xai-secret", "GROK_API_KEY": "grok-secret",
            "XAI_BASE_URL": "https://evil.example", "GROK_BASE_URL": "https://evil.example",
            "GROK_CLI_CHAT_PROXY_BASE_URL": "https://evil.example",
            "GROK_AUTH_PROVIDER_COMMAND": "evil", "XAI_API_KEY_BACKUP": "x",
            "OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a", "SOME_OTHER_SECRET": "s",
            "GROK_HOME": "/somewhere/else", "OMNI_LLM_NOTIFY_CMD": "notify",
            "OMNI_LLM_CHILD_ENV_PASS": "FAKE_CANARY_INFO,XAI_API_KEY,GROK_API_KEY,"
                                       "OPENAI_API_KEY,OMNI_LLM_CHILD_ENV_PASS",
        }
        with mock.patch.dict(os.environ, polluted):
            captured = []
            real = dispatch_exec.pinned_env

            def spy(home):
                captured.append(real(home))
                return captured[-1]

            with mock.patch.object(dispatch_exec, "pinned_env", spy):
                rc, _stdout, stderr = self.run_main(self.args("--config-dir", link))
        self.assertEqual(rc, 0, stderr)
        seen = json.loads(self.slurp(argvfile))
        child = seen["env"]
        for name in ("XAI_API_KEY", "GROK_API_KEY", "XAI_BASE_URL", "GROK_BASE_URL",
                     "GROK_CLI_CHAT_PROXY_BASE_URL", "GROK_AUTH_PROVIDER_COMMAND",
                     "XAI_API_KEY_BACKUP", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "SOME_OTHER_SECRET", "OMNI_LLM_NOTIFY_CMD", "OMNI_LLM_CHILD_ENV_PASS"):
            self.assertNotIn(name, child, name)
        # The test seam adds its channel after the env is built; the operator pass list
        # (which named XAI_API_KEY and others) opened nothing.
        self.assertIn("FAKE_CANARY_INFO", child)
        self.assertEqual(child["GROK_HOME"], os.path.realpath(self.cfg))
        self.assertNotEqual(child["GROK_HOME"], link)
        self.assertIn("PATH", child)
        self.assertEqual(seen["argv"][0], "/x/grok")
        self.assertEqual(captured[0].get("GROK_HOME"), os.path.realpath(self.cfg))

    def test_canary_runs_with_the_same_pinned_env(self):
        calls = []
        polluted = {"XAI_API_KEY": "xai-secret", "GROK_HOME": "/somewhere/else"}
        with mock.patch.dict(os.environ, polluted), \
                mock.patch.object(dispatch_exec, "supervise", return_value=ENDED_OK), \
                mock.patch.object(dispatch_exec, "run_read_only_canary",
                                  side_effect=lambda **kw: calls.append(kw) or
                                  dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")):
            rc, _out, stderr = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 0, stderr)
        env = calls[0]["env"]
        self.assertEqual(env, dispatch_exec.pinned_env(os.path.realpath(self.cfg)))
        self.assertNotIn("XAI_API_KEY", env)
        self.assertEqual(env["GROK_HOME"], os.path.realpath(self.cfg))
        self.assertNotIn("OMNI_LLM_CHILD_ENV_PASS", env)

    def test_binary_resolution_failure_is_rc_4_not_6(self):
        self.grok = None
        rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 4)
        self.assertIn("capacity incident", stderr)
        self.assertRecord(4, "refused_before_spawn", "launch_unavailable", False)


class TestWrapperHardening(MainBase):
    """G1: the host-exec wrapper is a Python process that runs before fencing."""

    HOSTILE = {
        "PYTHONPATH": "/evil/site", "PYTHONSTARTUP": "/evil/startup.py",
        "PYTHONHOME": "/evil/home", "PYTHONINSPECT": "1", "FOO": "bar",
        "OMNI_LLM_CHILD_ENV_PASS": "PYTHONPATH,FOO,PYTHONSTARTUP",
    }

    def assertClean(self, env):
        self.assertEqual([name for name in env if name.startswith("PYTHON")], [])
        self.assertNotIn("FOO", env)
        self.assertNotIn("OMNI_LLM_CHILD_ENV_PASS", env)

    def test_operator_pass_list_and_python_names_never_reach_the_wrapper_env(self):
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = _REAL_PINNED_ENV(os.path.realpath(self.cfg))
            # The process's own environment is only borrowed for the call.
            self.assertEqual(os.environ["OMNI_LLM_CHILD_ENV_PASS"],
                             self.HOSTILE["OMNI_LLM_CHILD_ENV_PASS"])
        self.assertClean(env)
        self.assertEqual(env["GROK_HOME"], os.path.realpath(self.cfg))

    def test_python_names_are_dropped_even_if_the_allowlist_lets_them_through(self):
        leaky = {"PATH": "/usr/bin", "PYTHONPATH": "/evil", "PYTHONSTARTUP": "/evil.py",
                 "PYTHONWARNINGS": "x", "KEEP": "1"}
        with mock.patch.object(dispatch_exec.host_fence, "allowlisted_child_env",
                               return_value=dict(leaky)):
            env = _REAL_PINNED_ENV(os.path.realpath(self.cfg))
        self.assertEqual([name for name in env if name.startswith("PYTHON")], [])
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_vendor_and_canary_children_get_the_clean_env(self):
        argvfile = self.path("hostile-env.json")
        self.use_child([{"op": "argv", "path": argvfile}, out(RESULT_LINE), exit_(0)])
        seen = []
        with mock.patch.dict(os.environ, self.HOSTILE), \
                mock.patch.object(dispatch_exec, "run_read_only_canary",
                                  side_effect=lambda **kw: seen.append(kw["env"]) or
                                  dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")):
            rc, _out, stderr = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 0, stderr)
        self.assertClean(json.loads(self.slurp(argvfile))["env"])
        self.assertClean(seen[0])

    def test_both_wrappers_start_with_E_and_s(self):
        # G1c/G1d: the vendor wrapper and the canary wrapper are launched the same way.
        cli = os.path.join(HERE, "omni_cli.py")
        children, canaries = [], []
        with mock.patch.object(dispatch_exec, "host_exec_prefix", _REAL_PREFIX), \
                mock.patch.object(dispatch_exec, "supervise",
                                  side_effect=lambda child, **kw: children.append(child) or
                                  ENDED_OK), \
                mock.patch.object(dispatch_exec, "run_read_only_canary",
                                  side_effect=lambda **kw: canaries.append(
                                      kw["wrapper_prefix"]) or
                                  dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")):
            rc, _out, stderr = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 0, stderr)
        for argv in (children[0], canaries[0]):
            self.assertEqual(argv[:4], [sys.executable, "-E", "-s", cli])
            self.assertNotIn("-I", argv[:4])

    def test_wrapper_imports_under_E_s_and_runs_no_sitecustomize_from_pythonpath(self):
        site = self.path("evil-site")
        os.mkdir(site)
        marker = self.path("sitecustomize-ran")
        with open(os.path.join(site, "sitecustomize.py"), "w") as handle:
            handle.write("open(%r, 'w').write('x')\n" % marker)
        env = dict(os.environ, PYTHONPATH=site)
        cli = os.path.join(HERE, "omni_cli.py")
        # Control: without -E the planted sitecustomize does run.
        subprocess.run([sys.executable, cli, "host-exec", "--help"], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.dir)
        self.assertTrue(os.path.exists(marker), "control: sitecustomize did not run")
        os.unlink(marker)
        argv = _REAL_PREFIX("grok", self.work, True)[:-1] + ["--help"]
        done = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=self.dir)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(b"host-exec", done.stdout)
        self.assertFalse(os.path.exists(marker), "wrapper ran PYTHONPATH code")
        # The imports host-exec needs once it is past argument parsing still resolve.
        probe = subprocess.run(
            [sys.executable, "-E", "-s", "-c",
             "import sys; sys.path.insert(0, %r); import omni_cli; "
             "from providers import host_fence, kernel_sandbox, vendor_bin" % HERE],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.dir)
        self.assertEqual(probe.returncode, 0, probe.stderr)


class TestPromptBytes(MainBase):
    def test_refusals_are_rc_2_prompt_mismatch_before_anything_runs(self):
        # E4 through main: each refusal is rc 2 / prompt_mismatch / spawned false, with
        # nothing launched and no outfile.
        cases = (("sha mismatch", b"clean work order", hashlib.sha256(b"x").hexdigest()),
                 ("nul", b"a\0b", None), ("trailing lf", b"clean work order\n", None),
                 ("invalid utf-8", b"bad \xff", None))
        for name, raw, sha in cases:
            with self.subTest(name):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.set_prompt(raw)
                argv = self.args("--prompt-sha256", sha or hashlib.sha256(raw).hexdigest())
                rc, stdout, stderr = self.run_main(argv)
                self.assertEqual(rc, 2, stderr)
                self.assertIn("prompt refused", stderr)
                self.assertEqual(stdout, "")
                self.assertNothingSpawned()
                self.assertRecord(2, "refused_before_spawn", "prompt_mismatch", False)

    def test_the_prompt_file_changing_after_the_scan_is_a_mismatch(self):
        scanned = hashlib.sha256(self.prompt_raw).hexdigest()
        self.set_prompt(b"swapped after the scan")
        rc, _out, _err = self.run_main(self.args("--prompt-sha256", scanned))
        self.assertEqual(rc, 2)
        self.assertRecord(2, "refused_before_spawn", "prompt_mismatch", False)

    def test_exact_bytes_reach_argv(self):
        # E4: what the child receives, re-encoded, is the file's bytes.
        raw = "日本語 \"q\" 'r' $x `y` \\ \n  indented\n\n\tend ".encode()
        self.set_prompt(raw)
        argvfile = self.path("bytes.json")
        self.use_child([{"op": "argv", "path": argvfile}, out(RESULT_LINE), exit_(0)])
        rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 0, stderr)
        prompt = json.loads(self.slurp(argvfile))["argv"][-1]
        self.assertEqual(prompt.encode("utf-8"), raw)

    def test_argv_bytes_that_do_not_round_trip_are_refused_before_spawn(self):
        # E4: if the argv encoding would change a byte the launch is refused.
        with mock.patch.object(dispatch_exec.os, "fsencode",
                               side_effect=lambda text: text.encode("utf-8") + b"!"):
            rc, _out, _err = self.run_main(self.args())
        self.assertEqual(rc, 2)
        self.assertRecord(2, "refused_before_spawn", "prompt_mismatch", False)
        self.assertNothingSpawned()


class TestResultRecord(MainBase):
    def test_record_is_published_once_atomically_with_mode_0600(self):
        order = []
        real_replace, real_fsync = os.replace, os.fsync
        with mock.patch.object(os, "replace", lambda a, b: (order.append(("replace", a, b)),
                                                             real_replace(a, b))[1]), \
                mock.patch.object(os, "fsync", lambda fd: (order.append(("fsync",)),
                                                            real_fsync(fd))[1]):
            rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 0, stderr)
        self.assertRecord(0, "vendor_exit", "none", None, vendor_rc=0, marker=True)
        kinds = [item[0] for item in order]
        self.assertEqual(kinds.count("replace"), 1)
        self.assertLess(kinds.index("fsync"), kinds.index("replace"))  # file before rename
        self.assertIn("fsync", kinds[kinds.index("replace"):])  # then the directory
        _tag, tmp, final = next(item for item in order if item[0] == "replace")
        self.assertEqual(os.path.dirname(tmp), os.path.dirname(final))
        self.assertFalse(os.path.exists(tmp))

    def test_nothing_is_written_before_the_run_ends(self):
        # No provisional record: the file does not exist while the vendor runs.
        seen = []

        def fake_supervise(*_a, **_kw):
            seen.append(os.path.lexists(self.result))
            seen.append([n for n in os.listdir(self.dir) if "dispatch-result" in n])
            return ENDED_OK

        with mock.patch.object(dispatch_exec, "supervise", fake_supervise):
            self.assertEqual(self.run_main(self.args())[0], 0)
        self.assertEqual(seen, [False, []])
        self.assertTrue(os.path.exists(self.result))

    def test_an_existing_result_file_is_refused_and_left_alone(self):
        with open(self.result, "w") as handle:
            handle.write("someone else's file")
        rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 2)
        self.assertIn("--result-file already exists", stderr)
        self.assertEqual(self.slurp(self.result), "someone else's file")
        self.assertNothingSpawned()
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_a_result_file_appearing_during_the_run_is_not_replaced(self):
        def fake_supervise(*_a, **_kw):
            with open(self.result, "w") as handle:
                handle.write("raced in")
            return ENDED_OK

        with mock.patch.object(dispatch_exec, "supervise", fake_supervise):
            rc, _out, stderr = self.run_main(self.args())
        self.assertEqual(rc, 125)
        self.assertEqual(stderr.splitlines()[-1],
                         "omni dispatch-exec: result record not published")
        self.assertEqual(self.slurp(self.result), "raced in")
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_publication_failure_is_rc_125_with_one_stderr_line_and_no_record(self):
        for name in ("replace", "fsync", "open"):
            with self.subTest(name):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                real_open = os.open

                def open_(path, *a, **kw):
                    if ".dispatch-result." in str(path):
                        raise OSError(28, "No space left on device")
                    return real_open(path, *a, **kw)

                patch = {"replace": mock.patch.object(os, "replace", side_effect=OSError(5, "io")),
                         "fsync": mock.patch.object(dispatch_result.os, "fsync",
                                                    side_effect=OSError(5, "io")),
                         "open": mock.patch.object(os, "open", open_)}[name]
                with mock.patch.object(dispatch_exec, "supervise", return_value=ENDED_OK), \
                        patch:
                    rc, _out, stderr = self.run_main(self.args())
                self.assertEqual(rc, 125)
                self.assertEqual(stderr, "omni dispatch-exec: result record not published\n")
                self.assertFalse(os.path.lexists(self.result))
                self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_record_for_every_refusal(self):
        # E1 mapping of the endings that happen before a spawn.
        from_args = self.args(drop=("--stall-grace-s",))
        rc, _out, _err = self.run_main(from_args)
        self.assertEqual(rc, 2)
        self.assertRecord(2, "refused_before_spawn", "arguments", False)
        os.unlink(self.result)
        argv = self.args()
        argv[argv.index("--workdir") + 1] = self.path("no-such-workdir")
        self.assertEqual(self.run_main(argv)[0], 2)
        self.assertRecord(2, "refused_before_spawn", "arguments", False)
        os.unlink(self.result)
        self.wrapper_mode = "write"
        rc, _out, _err = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 8)
        self.assertRecord(8, "refused_before_spawn", "containment", False)

    def test_dispatch_result_publish_and_build(self):
        record = dispatch_result.build(INVOCATION_ID, "vendor_exit", "none", None, 0, None,
                                       None, 0, 12, True)
        path = self.path("direct.json")
        dispatch_result.publish(path, record)
        self.assertEqual(json.loads(self.slurp(path)), record)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with self.assertRaises(dispatch_result.PublishError):
            dispatch_result.publish(path, record)  # never replaces another file
        self.assertEqual(json.loads(self.slurp(path)), record)
        with self.assertRaises(dispatch_result.PublishError):
            dispatch_result.publish(os.path.join(self.dir, "no", "dir.json"), record)
        for bad in (dict(outcome="bogus"), dict(detail="bogus"), dict(spawned="no"),
                    dict(vendor_rc=256), dict(exit_code="0"), dict(elapsed_ms=-1),
                    dict(invocation_id="XYZ")):
            good = dict(invocation_id=INVOCATION_ID, outcome="vendor_exit", detail="none",
                        spawned=None, vendor_rc=0, vendor_signal=None, signum=None,
                        exit_code=0, elapsed_ms=1, marker=False)
            good.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dispatch_result.build(**good)


class TestParentAtStart(MainBase):
    def test_a_parent_pid_that_is_not_ours_refuses_before_spawn(self):
        # E3: getppid() != --parent-pid at start -> supervisor_failure/parent_died, spawned
        # false, rc 125, nothing run.
        rc, stdout, stderr = self.run_main(self.args(parent=os.getppid() + 1))
        self.assertEqual(rc, 125, stderr)
        self.assertEqual(stdout, "")
        self.assertNothingSpawned()
        self.assertRecord(125, "supervisor_failure", "parent_died", False)

    def test_a_parent_that_dies_between_canary_and_launch_is_caught(self):
        real_ppid = os.getppid()
        state = {"canary_done": False}

        def canary(**_kw):
            state["canary_done"] = True
            return dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")

        with mock.patch.object(dispatch_exec, "run_read_only_canary", canary), \
                mock.patch.object(os, "getppid",
                                  lambda: real_ppid + 1 if state["canary_done"] else real_ppid), \
                mock.patch.object(dispatch_exec, "supervise") as launched:
            rc, _out, _err = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 125)
        launched.assert_not_called()
        self.assertFalse(os.path.exists(self.outfile))
        self.assertRecord(125, "supervisor_failure", "parent_died", False)

    def test_parent_death_during_preflight_wins_over_the_refusal_decided_with_it(self):
        # G4: the caller vanishes while a check runs; the check then fails. Before any
        # refusal (or failure) is published the parent is looked at again: the record is
        # parent_died / rc 125, never the refusal.
        real_ppid = os.getppid()
        real_read, real_cred = dispatch_exec.read_prompt, dispatch_exec.credential_home
        gone = {"now": False}

        def vanish(real=None, result=None, effect=None):
            def run(*args, **kwargs):
                gone["now"] = True
                if effect:
                    raise effect
                return real(*args, **kwargs) if real else result
            return run

        cases = (
            ("prompt mismatch", "read_prompt", vanish(real_read), ("--prompt-sha256", "0" * 64),
             "workspace", False),
            ("credential failure", "credential_home", vanish(result=None), (), "workspace",
             False),
            ("grok missing", "resolve_grok", vanish(result=None), (), "workspace", False),
            ("containment refused", "run_read_only_canary",
             vanish(result=dispatch_canary.CanaryResult("refuse", "inconclusive")), (),
             "read-only", False),
            ("start unknown", "supervise", vanish(effect=OSError("exec failed")), (),
             "workspace", None),
        )
        for name, target, replacement, extra, mode, spawned in cases:
            with self.subTest(name):
                gone["now"] = False
                self.result = self.path("preflight-%s.json" % name.replace(" ", "-"))
                with mock.patch.object(os, "getppid",
                                       lambda: real_ppid + 1 if gone["now"] else real_ppid), \
                        mock.patch.object(dispatch_exec, target, replacement):
                    rc, _out, _err = self.run_main(
                        self.args(*extra, mode=mode, parent=real_ppid))
                self.assertEqual(rc, 125)
                self.assertRecord(125, "supervisor_failure", "parent_died", spawned)

    def test_parent_pid_must_be_above_one(self):
        for bad in ("1", "0", "-5", "abc", "99999999999"):
            with self.subTest(bad=bad):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.assertEqual(self.run_main(self.args("--parent-pid", bad))[0], 2)
                self.assertRecord(2, "refused_before_spawn", "arguments", False)


class TestCancellationGate(MainBase):
    def test_a_signal_between_canary_and_launch_wins_and_nothing_launches(self):
        # E6: the first HUP/INT/TERM decides, also in the gap after the canary: rc 128+n,
        # outcome signal, spawned false, no header, the vendor never started.
        for signum, expect in ((signal.SIGTERM, 143), (signal.SIGHUP, 129),
                               (signal.SIGINT, 130)):
            with self.subTest(signal=signum.name):
                if os.path.exists(self.result):
                    os.unlink(self.result)

                def canary(**_kw):
                    os.kill(os.getpid(), signum)
                    time.sleep(0.05)  # let the recorder run
                    return dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")

                with mock.patch.object(dispatch_exec, "run_read_only_canary", canary), \
                        mock.patch.object(dispatch_exec, "supervise") as launched:
                    rc, _out, _err = self.run_main(self.args(mode="read-only"))
                self.assertEqual(rc, expect)
                launched.assert_not_called()
                self.assertFalse(os.path.exists(self.outfile))
                self.assertRecord(expect, "signal", "none", False, signum=int(signum))

    def test_the_first_signal_is_the_one_retained(self):
        def canary(**_kw):
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            return dispatch_canary.CanaryResult("pass", "2026-01-02T03:04:05Z")

        with mock.patch.object(dispatch_exec, "run_read_only_canary", canary):
            rc, _out, _err = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 129)
        self.assertRecord(129, "signal", "none", False, signum=int(signal.SIGHUP))

    def test_a_signal_during_a_refusal_still_decides_the_outcome(self):
        # The gate covers the whole invocation: a refusal being recorded is not a safe
        # harbour either.
        def prompt_read(*_a):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
            raise dispatch_exec.PromptMismatch("late")

        with mock.patch.object(dispatch_exec, "read_prompt", prompt_read):
            rc, _out, _err = self.run_main(self.args())
        self.assertEqual(rc, 130)
        self.assertRecord(130, "signal", "none", False, signum=int(signal.SIGINT))

    def guard_signals(self):
        """Make a signal that escapes the gate a test failure instead of killing the run."""
        saved = _handlers()

        def escaped(signum, _frame):
            raise AssertionError("signal %d escaped the cancellation gate" % signum)

        for sig in _TERMINATION:
            signal.signal(sig, escaped)

        def put_back():
            for sig, old in saved.items():
                signal.signal(sig, old)
        self.addCleanup(put_back)

    @contextlib.contextmanager
    def deliver_from(self, name, signum):
        """Patch dispatch_exec.<name> so that it delivers ``signum`` to us, then runs on."""
        real = getattr(dispatch_exec, name)

        def hit(*args, **kwargs):
            os.kill(os.getpid(), signum)
            time.sleep(0.05)  # let the recorder run
            return real(*args, **kwargs)

        with mock.patch.object(dispatch_exec, name, hit):
            yield

    def test_a_signal_during_argument_parsing_is_retained_and_decides_the_ending(self):
        # G3: the gate is the first thing main enters; the parse/validation window is
        # covered. Usable identity: a signal record, nothing spawned, rc 128+n.
        self.guard_signals()
        for signum, expect in ((signal.SIGTERM, 143), (signal.SIGHUP, 129),
                               (signal.SIGINT, 130)):
            for hook in ("_parser", "_scan"):
                with self.subTest(signal=signum.name, hook=hook):
                    self.result = self.path("parse-%s-%s.json" % (signum.name, hook))
                    with self.deliver_from(hook, signum):
                        try:
                            rc, _out, stderr = self.run_main(self.args())
                        except KeyboardInterrupt:
                            self.fail("KeyboardInterrupt escaped")
                    self.assertEqual(rc, expect, stderr)
                    self.assertNotIn("Traceback", stderr)
                    self.assertNothingSpawned()
                    self.assertRecord(expect, "signal", "none", False, signum=int(signum))

    def test_a_signal_during_validation_without_a_usable_identity_has_no_record(self):
        self.guard_signals()
        for signum, expect in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=signum.name):
                self.result = self.path("noid-%s.json" % signum.name)
                with self.deliver_from("_parser", signum):
                    try:
                        rc, _out, stderr = self.run_main(
                            self.args("--invocation-id", "nope", drop=("--invocation-id",)))
                    except KeyboardInterrupt:
                        self.fail("KeyboardInterrupt escaped")
                self.assertEqual(rc, expect)
                self.assertNotIn("Traceback", stderr)
                self.assertFalse(os.path.lexists(self.result))
                self.assertNothingSpawned()

    def test_a_signal_during_a_failed_parse_still_decides_the_ending(self):
        # The parse itself fails (rc 2 / arguments) while a signal is already retained.
        self.guard_signals()
        self.result = self.path("failed-parse.json")
        with self.deliver_from("_parser", signal.SIGTERM):
            rc, _out, _err = self.run_main(self.args("--wall-s", "0"))
        self.assertEqual(rc, 143)
        self.assertRecord(143, "signal", "none", False, signum=int(signal.SIGTERM))

    def test_the_gate_is_released_after_the_run(self):
        before = _handlers()
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        self.run_main(self.args())
        self.assertEqual(_handlers(), before)
        self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, []), mask)


class TestAliasAndFlags(MainBase):
    def assertRefusedBeforeTruncation(self, argv):
        rc, stdout, stderr = self.run_main(argv)
        self.assertEqual(rc, 2, stderr)
        self.assertEqual(stderr, "dispatch-external: outfile aliases the work order\n")
        self.assertEqual(self.prefix_calls, [])  # no canary, no launch
        self.assertEqual(stdout, "")
        self.assertRecord(2, "refused_before_spawn", "arguments", False)
        os.unlink(self.result)

    def test_outfile_aliasing_the_prompt_file_is_refused(self):
        # D 1464-1490: same path, symlink or hard link; nothing is truncated.
        original = self.slurp(self.prompt, "rb")
        link = self.path("link.txt")
        os.symlink(self.prompt, link)
        hard = self.path("hard.txt")
        os.link(self.prompt, hard)
        for name, target in (("same path", self.prompt), ("symlink", link),
                             ("hard link", hard)):
            with self.subTest(name):
                self.assertRefusedBeforeTruncation(
                    self.args("--outfile", target, mode="read-only"))
                self.assertEqual(self.slurp(self.prompt, "rb"), original)

    def test_outfile_aliasing_the_trace_prompt_is_refused(self):
        trace = self.path("trace.txt")
        with open(trace, "w") as handle:
            handle.write("trace body")
        self.assertRefusedBeforeTruncation(
            self.args("--trace-prompt", trace, "--outfile", trace))
        self.assertEqual(self.slurp(trace), "trace body")

    def test_result_file_aliasing_another_path_is_unusable_and_rc_2(self):
        # E1: a result file that aliases the prompt, the outfile or the trace path cannot
        # be written without destroying that file: rc 2 and no record anywhere.
        trace = self.path("trace-later.txt")
        for flag, target in (("--prompt-file", self.prompt), ("--outfile", self.outfile),
                             ("--trace-prompt", trace)):
            with self.subTest(flag):
                before = self.slurp(self.prompt, "rb")
                link = self.path("alias-link-%s" % flag.strip("-"))
                os.symlink(target, link)
                for result in (target, link):
                    rc, _out, stderr = self.run_main(
                        self.args("--trace-prompt", trace, "--result-file", result))
                    self.assertEqual(rc, 2, stderr)
                    self.assertNothingSpawned()
                    self.assertEqual(self.slurp(self.prompt, "rb"), before)
        self.assertFalse(os.path.exists(self.result))
        self.assertFalse(os.path.exists(trace))

    def test_invalid_identity_flags_exit_2_without_a_record(self):
        # The id and result file decide whether a record is possible at all.
        sub = os.path.join(self.dir, "sub")
        for name, extra in (
                ("short id", ["--invocation-id", "0123456789abcdef"]),
                ("upper id", ["--invocation-id", "0123456789ABCDEF0123456789ABCDEF"]),
                ("non-hex id", ["--invocation-id", "g" * 32]),
                ("id with newline", ["--invocation-id", "0" * 32 + "\n"]),
                ("relative result", ["--result-file", "result.json"]),
                ("result parent missing", ["--result-file", os.path.join(sub, "r.json")]),
                ("result is a dir", ["--result-file", self.dir])):
            with self.subTest(name):
                rc, stdout, stderr = self.run_main(self.args(*extra))
                self.assertEqual(rc, 2, stderr)
                self.assertNothingSpawned()
                self.assertFalse(os.path.exists(self.result))
        # ... and also when the rest of the command line is broken: nowhere to record.
        rc, _o, _e = self.run_main(self.args("--invocation-id", "nope", "--wall-s", "0"))
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(self.result))

    def test_argument_errors_with_a_usable_identity_always_publish_the_record(self):
        # G2: the id and file are recovered from the tokens, so any argparse failure
        # (a value-less flag, an unknown flag, a flag swallowing the next one) still
        # publishes refused_before_spawn/arguments, spawned false, rc 2.
        cases = {
            "value-less trailing --prompt-file": (("--prompt-file",), ["--prompt-file"]),
            "value-less trailing --wall-s": (("--wall-s",), ["--wall-s"]),
            "value-less trailing --outfile": (("--outfile",), ["--outfile"]),
            "value-less trailing --trace-prompt": ((), ["--trace-prompt"]),
            "value-less trailing --result-file after a good one": ((), ["--result-file"]),
            "unknown flag": ((), ["--no-such-flag"]),
            "unknown flag with value": ((), ["--no-such-flag", "x"]),
        }
        for name, (drop, tail) in cases.items():
            with self.subTest(name):
                self.result = self.path("argerr-%d.json" % self.n)
                self.n += 1
                rc, _out, stderr = self.run_main(self.args(drop=drop) + tail)
                self.assertEqual(rc, 2, stderr)
                self.assertNothingSpawned()
                self.assertRecord(2, "refused_before_spawn", "arguments", False)
        with self.subTest("a flag in front swallows nothing: next token is a flag"):
            self.result = self.path("argerr-lead.json")
            rc, _out, _err = self.run_main(["--prompt-file"] + self.args(drop=("--prompt-file",)))
            self.assertEqual(rc, 2)
            self.assertRecord(2, "refused_before_spawn", "arguments", False)
        with self.subTest("equals form"):
            self.result = self.path("argerr-eq.json")
            argv = self.args(drop=("--invocation-id", "--result-file", "--wall-s"))
            rc, _out, _err = self.run_main(
                argv + ["--invocation-id=" + INVOCATION_ID, "--result-file=" + self.result,
                        "--wall-s"])
            self.assertEqual(rc, 2)
            self.assertRecord(2, "refused_before_spawn", "arguments", False)
        with self.subTest("unusable identity stays unrecorded"):
            self.result = self.path("argerr-bad.json")
            rc, _out, _err = self.run_main(
                self.args("--invocation-id", "nope", drop=("--invocation-id",)) + ["--prompt-file"])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.lexists(self.result))

    def test_every_timing_flag_is_required(self):
        # E5: no defaults; an omitted flag is argparse's usage error, rc 2 (and, since the
        # id and file are usable, an `arguments` record).
        for flag in ("--wall-s", "--stall-grace-s", "--kill-grace-s", "--poll-s",
                     "--canary-timeout-s"):
            with self.subTest(flag):
                self.result = self.path("timing-%s.json" % flag.strip("-"))
                self.assertFalse(os.path.lexists(self.result))
                rc, _out, stderr = self.run_main(self.args(drop=(flag,)))
                self.assertEqual(rc, 2)
                self.assertIn("the following arguments are required: " + flag, stderr)
                self.assertNothingSpawned()
                self.assertRecord(2, "refused_before_spawn", "arguments", False)

    def test_every_new_required_flag_is_required(self):
        # Every iteration starts from a fresh, absent result file, so the rc 2 cannot come
        # from an existing-file refusal left by an earlier iteration: it is argparse's
        # required-flag error, and (id and file usable) the record names this run.
        for flag in ("--invocation-id", "--result-file", "--login", "--config-dir",
                     "--parent-pid", "--prompt-sha256"):
            with self.subTest(flag):
                self.result = self.path("required-%s.json" % flag.strip("-"))
                self.assertFalse(os.path.lexists(self.result))
                rc, _out, stderr = self.run_main(self.args(drop=(flag,)))
                self.assertEqual(rc, 2)
                self.assertIn("the following arguments are required: " + flag, stderr)
                self.assertNothingSpawned()
                if flag in ("--invocation-id", "--result-file"):
                    # No usable identity: nowhere to record, and nothing was written.
                    self.assertFalse(os.path.lexists(self.result))
                else:
                    self.assertRecord(2, "refused_before_spawn", "arguments", False)

    def test_invalid_flags_exit_2(self):
        # B: bounds wall 1-604800, grace 1-86400, kill grace 1-60, poll 1-60,
        # canary 1-120; unknown provider; missing paths. All rc 2 with an `arguments`
        # record, nothing written to the outfile.
        bad = [
            ["--wall-s", "0"], ["--wall-s", "604801"], ["--wall-s", "abc"],
            ["--wall-s", "-1"], ["--wall-s", "1.5"], ["--stall-grace-s", "0"],
            ["--stall-grace-s", "86401"], ["--kill-grace-s", "0"], ["--kill-grace-s", "61"],
            ["--poll-s", "0"], ["--poll-s", "61"], ["--canary-timeout-s", "0"],
            ["--canary-timeout-s", "121"], ["--mode", "write"],
            ["--prompt-sha256", "ABC"], ["--prompt-sha256", "0" * 63],
        ]
        for extra in bad:
            with self.subTest(extra=extra):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                rc, _out, _err = self.run_main(self.args(*extra))
                self.assertEqual(rc, 2)
                self.assertFalse(os.path.exists(self.outfile))
                self.assertRecord(2, "refused_before_spawn", "arguments", False)
        os.unlink(self.result)
        rc, _o, _e = self.run_main(self.args("--provider", "codex"))
        self.assertEqual(rc, 2)
        for flag in ("--model", "--effort", "--workdir", "--prompt-file", "--outfile"):
            argv = self.args()
            at = argv.index(flag)
            del argv[at:at + 2]
            with self.subTest(missing=flag):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.assertEqual(self.run_main(argv)[0], 2)
        for flag, value in (("--prompt-file", self.path("missing.txt")),
                            ("--outfile", os.path.join(self.dir, "no-parent", "o.txt")),
                            ("--model", " ")):
            with self.subTest(bad=flag):
                if os.path.exists(self.result):
                    os.unlink(self.result)
                argv = self.args()
                argv[argv.index(flag) + 1] = value
                self.assertEqual(self.run_main(argv)[0], 2)
                self.assertRecord(2, "refused_before_spawn", "arguments", False)
        self.assertEqual(self.prefix_calls, [])

    def test_help_is_not_an_error(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            self.assertEqual(dispatch_exec.main(["--help"]), 0)
        for flag in ("--invocation-id", "--result-file", "--login", "--parent-pid",
                     "--prompt-sha256", "--config-dir"):
            self.assertIn(flag, shown.getvalue())
        self.assertFalse(os.path.exists(self.result))

    def test_omni_cli_routes_dispatch_exec(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(omni_cli.main(["dispatch-exec", "--provider", "nope"]), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(omni_cli.main(["dispatch-exec", "--help"]), 0)


class TestCanary(MainBase):
    def canary(self, mode, timeout=20):
        self.wrapper_mode = mode
        return dispatch_canary.run_read_only_canary(
            wrapper_prefix=self.fake_prefix("grok", self.work, True), provider="grok",
            workdir=self.work, env=dict(os.environ, FAKE_CANARY_INFO=self.info),
            timeout_s=timeout)

    def test_pass_records_the_probe_contract(self):
        # D 1021-1093, 1350: ready line + target-specific denial + rc 23 + target
        # absent; probe script private (0700) and outside the workdir.
        result = self.canary("pass")
        self.assertEqual(result.status, "pass")
        self.assertRegex(result.detail, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        info = json.loads(self.slurp(self.info))
        self.assertEqual(info["script_mode"], "0o700")
        self.assertEqual(info["dir_mode"], "0o700")
        self.assertNotEqual(os.path.commonpath(
            [os.path.realpath(info["script"]), os.path.realpath(self.work)]),
            os.path.realpath(self.work))
        self.assertEqual(os.path.dirname(info["target"]), self.work)
        self.assertRegex(os.path.basename(info["target"]),
                         r"^dispatch-ro-canary\.[0-9a-f]{32}\.")
        self.assertFalse(os.path.basename(info["target"]).startswith("."))
        self.assertIn("DISPATCH_RO_CANARY", info["rest"])
        self.assertEqual(self.canary_leftovers(), [])
        self.assertFalse(os.path.exists(info["script"]))

    def test_refusals(self):
        # D 1337-1356, 1424-1426: each of these refuses with D's reason text.
        cases = {
            "write": "write succeeded",
            "no_ready": "inconclusive",
            "no_denial": "inconclusive",
            "wrong_rc": "inconclusive",
            "exit0_denied": "inconclusive",
            "other_path": "inconclusive",
            "marker_missing": "grok canary argv rewrite unsurvivable",
        }
        for mode, reason in cases.items():
            with self.subTest(mode):
                result = self.canary(mode)
                self.assertEqual((result.status, result.detail), ("refuse", reason))
                self.assertEqual(self.canary_leftovers(), [])

    def test_timeout_refuses_and_kills_the_probe_group(self):
        # D 1250-1258, 1280-1287: the bound expires -> group KILL -> inconclusive.
        start = time.monotonic()
        result = self.canary("hang", timeout=1)
        self.assertEqual((result.status, result.detail), ("refuse", "inconclusive"))
        self.assertLess(time.monotonic() - start, 15)
        self.assertTrue(self.wait_dead(json.loads(self.slurp(self.info))["pid"]))
        self.assertEqual(self.canary_leftovers(), [])

    def test_probe_slot_cleanup_removes_what_mkstemp_created(self):
        # D 1386-1403: no dispatch-ro-canary.* entry of this run may remain in the
        # workdir. mkstemp created it, then unlink (or close) fails: the slot is
        # inconclusive and the created path is removed best effort (round 2 left it).
        real_unlink, real_close, real_mkstemp = os.unlink, os.close, tempfile.mkstemp
        made = {}

        def mkstemp(*args, **kw):
            fd, path = real_mkstemp(*args, **kw)
            made["fd"] = fd
            return fd, path

        def unlink(path, *args, **kw):
            if (os.path.basename(str(path)).startswith("dispatch-ro-canary.")
                    and not made.get("unlink_failed")):
                made["unlink_failed"] = True
                raise OSError(5, "Input/output error")
            return real_unlink(path, *args, **kw)

        def close(fd):
            real_close(fd)
            if fd == made.get("fd") and not made.get("close_failed"):
                made["close_failed"] = True
                raise OSError(5, "Input/output error")

        for name, patches in (("unlink", [mock.patch.object(os, "unlink", unlink)]),
                              ("close", [mock.patch.object(os, "close", close)])):
            with self.subTest(name):
                made.clear()
                if os.path.exists(self.result):
                    os.unlink(self.result)
                self.wrapper_mode = "pass"
                with patches[0], mock.patch.object(tempfile, "mkstemp", mkstemp):
                    rc, _out, stderr = self.run_main(self.args(mode="read-only"))
                self.assertTrue(made.get(name + "_failed"))
                self.assertEqual(rc, 8)
                self.assertEqual(
                    stderr, "dispatch-external: read-only containment refused: inconclusive\n")
                self.assertEqual(self.canary_leftovers(), [])
                self.assertEqual(os.listdir(self.work), [])
                self.assertFalse(os.path.exists(self.outfile))
                self.assertFalse(os.path.exists(self.info))

    def test_unwritable_workdir_cannot_prove_a_denial(self):
        # D 1386-1388: the parent must be able to create and remove a non-hidden
        # name first, otherwise inconclusive and the wrapper is never run.
        if os.geteuid() == 0:
            self.skipTest("root ignores directory modes")
        os.chmod(self.work, 0o500)
        self.addCleanup(os.chmod, self.work, 0o700)
        result = self.canary("pass")
        self.assertEqual((result.status, result.detail), ("refuse", "inconclusive"))
        self.assertFalse(os.path.exists(self.info))

    def test_probe_script_skips_inserted_vendor_flags_and_exits_23_on_denial(self):
        # D 1088-1110: host-exec inserts `--sandbox off` after argv[0]; the real
        # script skips to the marker, prints the ready line and exits 23 when the
        # write is denied. Real sh error text is classified as a denial.
        if os.geteuid() == 0:
            self.skipTest("root ignores directory modes")
        nonce = "ab" * 16
        script_dir, script = dispatch_canary._install_script(self.work, nonce)
        self.addCleanup(shutil.rmtree, script_dir, True)
        ro = self.path("ro")
        os.mkdir(ro)
        os.chmod(ro, 0o500)
        self.addCleanup(os.chmod, ro, 0o700)
        target = os.path.join(ro, "dispatch-ro-canary.%s.X1" % nonce)
        done = subprocess.run([script, "--sandbox", "off", "DISPATCH_RO_CANARY", target],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        self.assertEqual(done.returncode, 23)
        self.assertIn("dispatch-ro-canary: writing " + nonce + "\n", done.stderr)
        errfile, outfile = self.path("e.txt"), self.path("o.txt")
        for name, text in ((errfile, done.stderr), (outfile, done.stdout)):
            with open(name, "w") as handle:
                handle.write(text)
        self.assertEqual(dispatch_canary.classify(nonce, target, outfile, errfile), 0)
        # No marker at all -> the script says so and the classifier names it.
        missing = subprocess.run([script, "--sandbox", "off"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("dispatch-ro-canary: marker missing", missing.stderr)

    def test_denial_phrase_is_read_only_after_the_path_is_removed(self):
        # D 1331-1333: a path that itself says "permission denied" is no evidence.
        nonce = "cd" * 16
        target = "/w/permission denied/dispatch-ro-canary.%s.X" % nonce
        errfile, outfile = self.path("e2.txt"), self.path("o2.txt")
        with open(outfile, "w"):
            pass
        with open(errfile, "w") as handle:
            handle.write("dispatch-ro-canary: writing %s\nsh: %s: no such file\n"
                         % (nonce, target))
        self.assertEqual(dispatch_canary.classify(nonce, target, outfile, errfile), 2)

    def test_main_read_only_pass_runs_canary_then_launches_with_the_same_wrapper(self):
        # D 1456, 1491-1500, 2103-2104: canary=pass header, then the launch under the
        # same wrapper arguments (grok, workdir, --read-only).
        self.wrapper_mode = "pass"
        rc, stdout, stderr = self.run_main(self.args(mode="read-only"))
        self.assertEqual(rc, 0, stderr)
        workdir = os.path.abspath(self.work)
        self.assertEqual(self.prefix_calls, [("grok", workdir, True)] * 2)
        data = self.read_out()
        self.assertRegex(
            data.decode().splitlines()[0],
            r"^dispatch-containment: layer=omni-host-exec sandbox=read-only canary=pass "
            r"canary_at=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertTrue(data.endswith(RESULT_LINE))
        self.assertIn("--read-only -- ", stdout)
        self.assertEqual(self.canary_leftovers(), [])
        self.assertRecord(0, "vendor_exit", "none", None, vendor_rc=0, marker=True)

    def test_main_canary_failure_is_rc_8_with_ds_message_and_no_outfile(self):
        # D 1359-1362 and T/test_routing 1580-1600: refuse, nothing launched, the
        # outfile never created.
        for mode, reason in (("write", "write succeeded"), ("no_ready", "inconclusive")):
            with self.subTest(mode):
                self.wrapper_mode = mode
                self.prefix_calls.clear()
                rc, stdout, stderr = self.run_main(self.args(mode="read-only"))
                self.assertEqual(rc, 8)
                self.assertEqual(
                    stderr, "dispatch-external: read-only containment refused: %s\n" % reason)
                self.assertEqual(stdout, "")
                self.assertFalse(os.path.exists(self.outfile))
                self.assertEqual(len(self.prefix_calls), 1)  # the canary only
                self.assertEqual(self.canary_leftovers(), [])
                self.assertRecord(8, "refused_before_spawn", "containment", False)
                os.unlink(self.result)

    def test_main_canary_timeout_is_rc_8(self):
        self.wrapper_mode = "hang"
        rc, _out, stderr = self.run_main(self.args("--canary-timeout-s", "1", mode="read-only"))
        self.assertEqual(rc, 8)
        self.assertIn("read-only containment refused: inconclusive", stderr)
        self.assertFalse(os.path.exists(self.outfile))
        self.assertRecord(8, "refused_before_spawn", "containment", False)

    def test_infrastructure_oserror_is_the_refusal_not_a_crash(self):
        # Capture-dir creation or opening canary.err fails: rc 8 with D's refusal text,
        # no traceback, and nothing of this run (probe dir, capture dir) left behind.
        scratch = self.path("scratch-tmp")
        os.mkdir(scratch)
        real_mkdtemp, real_open = tempfile.mkdtemp, open

        def mkdtemp(*args, **kw):
            if kw.get("prefix") == "dispatch-ro-canary-io.":
                raise OSError(28, "No space left on device")
            return real_mkdtemp(*args, **kw)

        def open_(file, *args, **kw):
            if str(file).endswith("canary.err"):
                raise OSError(24, "Too many open files")
            return real_open(file, *args, **kw)

        injections = {
            "capture dir": mock.patch.object(tempfile, "mkdtemp", mkdtemp),
            "canary.err": mock.patch("builtins.open", open_),
        }
        for name, patcher in injections.items():
            with self.subTest(name):
                self.wrapper_mode = "pass"
                self.prefix_calls.clear()
                if os.path.exists(self.result):
                    os.unlink(self.result)
                with mock.patch.object(tempfile, "tempdir", scratch), patcher:
                    rc, stdout, stderr = self.run_main(self.args(mode="read-only"))
                self.assertEqual(rc, 8)
                self.assertEqual(
                    stderr, "dispatch-external: read-only containment refused: inconclusive\n")
                self.assertEqual(stdout, "")
                self.assertFalse(os.path.exists(self.outfile))
                self.assertEqual(os.listdir(scratch), [])  # capture dir gone
                self.assertEqual(os.listdir(self.scripts), [])  # probe dir gone
                self.assertEqual(self.canary_leftovers(), [])
                self.assertFalse(os.path.exists(self.info))  # the wrapper never ran

    def test_main_workspace_runs_no_canary_and_records_na(self):
        # D 1370-1372: only read-only runs the canary; header canary=n/a canary_at=-.
        rc, _out, stderr = self.run_main(self.args(mode="workspace"))
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(self.read_out(), HEADER + RESULT_LINE)
        self.assertEqual(len(self.prefix_calls), 1)
        self.assertFalse(os.path.exists(self.info))
        self.assertRecord(0, "vendor_exit", "none", None, vendor_rc=0, marker=True)


if __name__ == "__main__":
    unittest.main()
