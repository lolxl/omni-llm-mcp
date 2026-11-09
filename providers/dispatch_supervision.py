# no-opencc
"""Synchronous supervisor for one dispatch child that writes the outfile itself.

``omni dispatch-exec`` reproduces dispatch-external.sh's grok execution stage
(``D`` below). This module is its watchdog. It is deliberately independent of
``run_cli``: the child's stdout and stderr are opened straight onto the outfile
(``> O 2>&1``), so the bytes are raw, unmasked, unordered by any reader and
carry no footer, and the supervisor's own stdio can never reach the file.

Watchdog contract, matched to D's perl ``_launch_grok_watchdog``:
- marker: a newline-terminated line of the merged outfile that strictly decodes
  as UTF-8 JSON and is an object whose top-level ``type`` is ``result``. A line
  torn by an interleaved stderr write, invalid UTF-8, or a ``type`` that is not
  top-level is no marker. An unterminated last line is not one yet.
- growth clock: the outfile size. Any growth after a marker (stderr included)
  restarts the grace.
- decisions are taken on the poll boundary; the child is reaped inside the poll
  sleep, so a natural exit that lands after the grace has already elapsed still
  wins and returns the vendor status with no watchdog line.
- natural exit: KILL the child's group (no TERM, no snapshot), return the vendor
  status (128+signal when signalled).
- watchdog firing (D ``terminate_group``): TERM the group and the identity-checked
  descendants (``ps`` lstart must still match), poll every 0.05 s while reaping the
  child until the group and the descendants are gone or the kill grace is spent,
  then KILL whatever is left (group, child pid, descendants). A cooperative child
  therefore ends on TERM without a KILL. Firing returns 9 after a marker and 10
  without one and appends one ``dispatch-watchdog:`` line to the outfile and to
  stderr.
- received HUP/INT/TERM (D ``on_signal``): KILL the group at once, TERM the
  descendants, KILL them after the kill grace. Returns 128+signal and writes no
  line.
  A group seen and then gone, and a reaped pid, are never signalled again.
- parent death (``parent_pid`` given): ``getppid`` is checked at every poll slice;
  a mismatch tears the child down like a received signal, writes no line and ends
  with ``parent_died`` (rc 125).

Pause contract. A host test-window pause stops the whole dispatch tree (the calling
dispatch shell, this supervisor and the worker) and continues it afterwards. The wall
and the grace spend active time only. Each poll keeps the previous poll's clock
reading (``last``, starting at ``t0``) and adds ``max(0, min(now - last, cap))`` to
the active budget and ``max(0, (now - last) - cap)`` to paused time, where
``cap = poll_s + 1``. A gap between polls longer than one poll plus one second is
excluded from the active budget: that covers such a pause, host sleep and a long
scheduling delay. A backward clock step adds nothing and does not move ``last``.
A stop this supervisor does not share (the worker stopping itself, a ptrace stop, a
pause of only the worker's group) counts as active time, because this process keeps
polling. Exclusions are bounded by an absolute ceiling of four times the wall: on
every poll, before the grace and wall checks, ``now - t0 >= 4 * wall_s`` fires
``wall-deadline`` (9 with a marker, 10 without) and the trailer carries
``ceiling=yes``. Each resume costs at most one cap of active budget, so a resume
close to the wall or the grace can still fire on the first poll after it. The quiet
clock advances only with active time.

``supervise`` returns an ``Ending``: the process rc plus what ended the run, so the
caller can publish a result record. A caller may pass one ``SignalGate`` that covers
its whole invocation; the child is then spawned with HUP/INT/TERM blocked and a
signal already recorded stops the launch.

Known limit (same as D): descendants that called ``setsid`` or were reparented
away before the snapshot are not reached.
"""
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import namedtuple

SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)

# outcome: vendor_exit | stall | wall_timeout | signal | parent_died.
# vendor_rc / vendor_signal: how the child itself ended (vendor_exit only, one of them).
# signal: the HUP/INT/TERM received. launched: False when no child was created.
Ending = namedtuple("Ending", "outcome rc vendor_rc vendor_signal signal marker launched")
PARENT_DIED_RC = 125

# D 2019/2022/2024 prints "dispatch-watchdog: reason=%s elapsed=%d marker=%s".
# Active time, paused time and the ceiling mark are appended after that prefix.
def watchdog_line(reason, elapsed, marker, active, paused, ceiling):
    """One trailer. ``marker`` is ``yes`` or ``no``; ``ceiling`` appends ``ceiling=yes``."""
    line = ("dispatch-watchdog: reason=%s elapsed=%d marker=%s active=%d paused=%d"
            % (reason, elapsed, marker, active, paused))
    if ceiling:
        line += " ceiling=yes"
    return line + "\n"

_SLICE_S = 0.1  # reap/scan cadence inside the poll sleep (D 1959 uses 0.2)
_GROUP_SETTLE_POLLS = 50  # D 1881, 1914: at most 50 * 0.02 s for the group to go
_GROUP_SETTLE_S = 0.02
_PS_TIMEOUT_S = 10


def child_setup():
    """Runs in the forked child before exec (D 1634-1638).

    Default dispositions and an empty mask for HUP/INT/TERM, then its own
    process group so the whole tree can be signalled as one.
    """
    for sig in SIGNALS:
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGNALS)
    os.setpgid(0, 0)


class SignalGate:
    """HUP/INT/TERM stay blocked from entry until ``arm`` (D 1630, 1934).

    A signal that arrives while the child is being spawned and put in its group
    stays pending and is recorded as soon as the handlers are installed, so it
    is never lost and never reaches the child half set up. Main thread only.
    """

    def __init__(self):
        self.pending = None
        self._old_mask = None
        self._old_handlers = None

    def __enter__(self):
        self._old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        return self

    def _record(self, signum, _frame):
        if self.pending is None:
            self.pending = signum

    def arm(self):
        self._old_handlers = {sig: signal.signal(sig, self._record) for sig in SIGNALS}
        signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGNALS)

    def ignore(self):
        for sig in SIGNALS:
            signal.signal(sig, signal.SIG_IGN)

    @contextlib.contextmanager
    def hold(self):
        """Block HUP/INT/TERM (a spawn window) on an armed gate; a signal that arrives
        stays pending and is recorded when the block is lifted. Blocking runs a
        handler that already tripped, so ``pending`` is current inside the block."""
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        try:
            yield self
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def commit(self):
        """Read ``pending`` and switch to ignored as one step; returns the signal or None.

        HUP/INT/TERM are blocked across both, so a signal cannot land between them and
        be recorded with nobody left to read it. One that arrives while blocked is
        discarded by the SIG_IGN, as D does once terminate_group has set IGNORE
        (D 1789-1792, 1894-1920). Main thread only.
        """
        with self.hold():
            pending = self.pending
            if pending is None:
                self.ignore()
        return pending

    def __exit__(self, *_exc):
        if self._old_handlers:
            for sig, old in self._old_handlers.items():
                signal.signal(sig, old if old is not None else signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, self._old_mask)
        return False


def group_signal(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


def group_kill(pgid):
    group_signal(pgid, signal.SIGKILL)


def _reject_constant(name):
    raise ValueError(name)


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _has_lone_surrogate(value):
    """True when a key or string anywhere in the parsed value holds U+D800-U+DFFF.

    json.loads joins a paired escape into one code point, so what is left here
    came from an unmatched escape, which JSON::PP utf8 rejects (D 1846).
    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            for key, child in item.items():
                if _LONE_SURROGATE.search(key):
                    return True
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _pairs_hook(pairs):
    """Check every decoded key and string value, then collapse duplicates last-wins.

    A plain dict would drop the overwritten value of a duplicate key before
    ``_has_lone_surrogate`` could see it; JSON::PP rejects the whole line (D 1846)
    and otherwise keeps the last duplicate.
    """
    for key, value in pairs:
        if _LONE_SURROGATE.search(key):
            raise ValueError("lone surrogate")
        # Dict values were checked by their own call; strings and lists are not.
        if isinstance(value, (str, list)) and _has_lone_surrogate(value):
            raise ValueError("lone surrogate")
    return dict(pairs)


def is_marker_line(line):
    """True for one complete line (no newline) that is a result frame (D 1846-1851)."""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        data = json.loads(text, parse_constant=_reject_constant,
                          object_pairs_hook=_pairs_hook)
    except (ValueError, RecursionError):
        return False
    if not isinstance(data, dict):
        return False
    kind = data.get("type")
    return isinstance(kind, str) and kind == "result"


def _normalize_lstart(text):
    return (text or "").strip()


def _ps_lstart(pid):
    if not pid or pid <= 0:
        return ""
    try:
        done = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=_PS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return ""
    return _normalize_lstart(done.stdout.splitlines()[0] if done.stdout.strip() else "")


def _proc_alive(pid):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class _Descendant:
    __slots__ = ("pid", "pgid", "lstart", "retired")

    def __init__(self, pid, pgid, lstart):
        self.pid = pid
        self.pgid = pgid
        self.lstart = lstart
        self.retired = False


class _Watch:
    """State of one supervised child; every method mirrors a D perl sub."""

    def __init__(self, proc, gate, outfile, offset, wall_s, grace_s, kill_grace_s,
                 poll_s, clock, parent_pid=None):
        self.proc = proc
        self.parent_pid = parent_pid
        self.pid = proc.pid
        self.gate = gate
        self.outfile = outfile
        self.offset = offset
        self.wall_s = wall_s
        self.grace_s = grace_s
        self.kill_grace_s = kill_grace_s
        self.poll_s = poll_s
        self.clock = clock
        self.t0 = clock()  # D 1654: after the fork
        self.group_seen = False
        self.group_gone = False
        self.saw_marker = False
        self.quiet_at = None
        self.quiet_size = -1
        self.last = self.t0  # previous poll; a backward step does not move this
        self.active = 0.0
        self.paused = 0.0
        self.quiet_active = 0.0  # active seconds since the quiet clock last restarted
        self.ceiling = False
        self._partial = bytearray()
        self._nl_from = 0
        self._in = None

    # -- liveness helpers ---------------------------------------------------
    @property
    def reaped(self):
        return self.proc.returncode is not None

    def group_alive(self):
        """D 1675-1683: a group seen and then gone is never signalled again."""
        if self.group_gone:
            return False
        try:
            os.killpg(self.pid, 0)
        except OSError:
            if self.group_seen:
                self.group_gone = True
            return False
        self.group_seen = True
        return True

    def parent_gone(self):
        """The process that started omni is no longer our parent (it died or was killed)."""
        return self.parent_pid is not None and os.getppid() != self.parent_pid

    def reap_if_exited(self):
        """Vendor status (128+signal when signalled) once the child has exited."""
        if self.reaped:
            return None
        code = self.proc.poll()
        if code is None:
            return None
        return 128 - code if code < 0 else code

    # -- descendants ----------------------------------------------------------
    def snapshot_descendants(self):
        """D 1709-1743: a reaped child pid may belong to someone else; do not walk it."""
        if self.reaped:
            return []
        try:
            done = subprocess.run(
                ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "lstart="],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, universal_newlines=True,
                timeout=_PS_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return []
        kids = {}
        meta = {}
        for line in done.stdout.splitlines():
            found = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)$", line)
            if not found:
                continue
            started = _normalize_lstart(found.group(4))
            if not started:
                continue
            pid, ppid, pgid = int(found.group(1)), int(found.group(2)), int(found.group(3))
            kids.setdefault(ppid, []).append(pid)
            meta[pid] = (pgid, started)
        out = []
        seen = set()
        stack = list(kids.get(self.pid, []))
        while stack:
            pid = stack.pop()
            if pid <= 0 or pid in seen:
                continue
            seen.add(pid)
            if pid == self.pid:
                continue
            if pid in meta:
                out.append(_Descendant(pid, meta[pid][0], meta[pid][1]))
            stack.extend(kids.get(pid, []))
        return out

    @staticmethod
    def _confirmed(rec):
        """D 1746-1755: a mismatch or a missing pid retires the record for good."""
        if rec.retired:
            return False
        current = _ps_lstart(rec.pid)
        if not current or current != rec.lstart:
            rec.retired = True
            return False
        return True

    def _signal_descendant(self, sig, rec):
        if not self._confirmed(rec):
            return
        try:
            os.kill(rec.pid, sig)
        except OSError:
            pass
        if rec.pgid == rec.pid:
            try:
                os.killpg(rec.pid, sig)
            except OSError:
                pass

    @staticmethod
    def _descendants_alive(desc):
        return any(not rec.retired and _proc_alive(rec.pid) for rec in desc)

    def _settle_group(self):
        n = 0
        while n < _GROUP_SETTLE_POLLS and self.group_alive():
            time.sleep(_GROUP_SETTLE_S)
            n += 1

    # -- teardown -------------------------------------------------------------
    def natural_exit(self, code):
        """D 1871-1892: the child exited by itself; its status wins."""
        self.gate.ignore()
        if self.group_alive():
            group_kill(self.pid)
        self._settle_group()
        return code

    def terminate_group(self):
        """D 1789-1819 (watchdog firing): TERM, wait the kill grace, then KILL."""
        self.gate.ignore()
        desc = self.snapshot_descendants()
        if self.group_alive():
            group_signal(self.pid, signal.SIGTERM)
        for rec in desc:
            self._signal_descendant(signal.SIGTERM, rec)
        deadline = time.monotonic() + self.kill_grace_s
        while time.monotonic() < deadline:
            if not self.reaped:
                self.proc.poll()  # reap inside the wait (D 1801-1804)
            if not self.group_alive() and not self._descendants_alive(desc):
                break
            time.sleep(0.05)
        if self.group_alive():
            group_kill(self.pid)
        if not self.reaped and _proc_alive(self.pid):
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass
        for rec in desc:
            self._signal_descendant(signal.SIGKILL, rec)
        if not self.reaped:
            self.proc.wait()
        self._settle_group()

    def signal_teardown(self):
        """D 1894-1919 (received HUP/INT/TERM): KILL the group now, TERM descendants,
        KILL them after the grace."""
        self.gate.ignore()
        desc = self.snapshot_descendants()
        if self.group_alive():
            group_kill(self.pid)
        if not self.reaped and _proc_alive(self.pid):
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass
        for rec in desc:
            self._signal_descendant(signal.SIGTERM, rec)
        deadline = time.monotonic() + self.kill_grace_s
        while time.monotonic() < deadline and self._descendants_alive(desc):
            time.sleep(0.05)
        for rec in desc:
            self._signal_descendant(signal.SIGKILL, rec)
        if not self.reaped:
            self.proc.wait()
        self._settle_group()

    # -- outfile scan ---------------------------------------------------------
    def _take(self, buf):
        self._partial += buf
        while True:
            at = self._partial.find(b"\n", self._nl_from)
            if at < 0:
                self._nl_from = len(self._partial)
                return
            line = bytes(self._partial[:at])
            del self._partial[:at + 1]
            self._nl_from = 0
            if is_marker_line(line):
                self.saw_marker = True

    def scan(self):
        """D 1829-1860: read new bytes; growth (any writer) restarts the quiet clock."""
        try:
            size = os.stat(self.outfile).st_size
        except OSError:
            return
        if self._in is None:
            try:
                self._in = open(self.outfile, "rb")
            except OSError:
                self._in = None
        if self._in is not None and size > self.offset:
            try:
                self._in.seek(self.offset)
                want = size - self.offset
                while want > 0:
                    buf = self._in.read(min(want, 1 << 20))
                    if not buf:
                        break
                    self.offset += len(buf)
                    want -= len(buf)
                    self._take(buf)
            except OSError:
                pass
        if self.saw_marker and size > self.quiet_size:
            self.quiet_at = self.clock()
            self.quiet_size = size
            self.quiet_active = 0.0

    def close(self):
        if self._in is not None:
            self._in.close()
            self._in = None

    # -- decisions --------------------------------------------------------------
    def _ending(self, outcome, rc, vendor_rc=None, vendor_signal=None, signum=None):
        return Ending(outcome, rc, vendor_rc, vendor_signal, signum, self.saw_marker, True)

    def finish(self, reason, code, out):
        self.terminate_group()
        # D 1865: read after teardown, so the kill grace is included in elapsed.
        # active/paused are the budget at the decision, not the teardown.
        elapsed = max(0, int(self.clock() - self.t0))
        line = watchdog_line(reason, elapsed, "yes" if self.saw_marker else "no",
                             max(0, int(self.active)), max(0, int(self.paused)),
                             self.ceiling)
        _append_line(self.outfile, line)
        out = out if out is not None else sys.stderr
        try:
            out.write(line)
            out.flush()
        except (OSError, ValueError):
            pass
        return self._ending("stall" if reason == "completed-then-hung" else "wall_timeout",
                            code)

    def _fire(self, reason, code, out):
        """A signal recorded before the commit wins over a watchdog decision.

        ``gate.commit`` checks ``pending`` and switches the handlers to ignored as one
        step, so a signal cannot land between them and be recorded with nobody left to
        read it (D 1789-1792, 1894-1920). Main thread only.
        """
        if self.gate.commit() is not None:
            return self.on_signal()
        return self.finish(reason, code, out)

    def on_signal(self):
        self.signal_teardown()
        return self._ending("signal", 128 + self.gate.pending, signum=self.gate.pending)

    def on_parent_death(self):
        """The caller is gone: nobody reads a watchdog line, so tear down and say why."""
        self.signal_teardown()
        return self._ending("parent_died", PARENT_DIED_RC)

    def _vendor_exit(self, code):
        """Natural exit: the child's own status wins (128+signal when signalled)."""
        self.natural_exit(code)
        raw = self.proc.returncode
        return self._ending("vendor_exit", code,
                            vendor_rc=raw if raw >= 0 else None,
                            vendor_signal=-raw if raw < 0 else None)

    def run(self, stderr):
        """D 1944-1965."""
        while True:
            if self.gate.pending is not None:
                return self.on_signal()
            code = self.reap_if_exited()
            if code is not None:
                return self._vendor_exit(code)
            if self.parent_gone():
                return self.on_parent_death()
            self.scan()
            if self.gate.pending is not None:
                # Recorded while scanning: D's handler would have interrupted the
                # loop at once (D 1894-1920), so the signal beats any decision.
                return self.on_signal()
            now = self.clock()
            self._credit(now)
            # Ceiling before grace and wall: a long exclusion cannot run forever.
            if now - self.t0 >= 4 * self.wall_s:
                self.ceiling = True
                return self._fire("wall-deadline", 9 if self.saw_marker else 10, stderr)
            if (self.saw_marker and self.quiet_at is not None
                    and self.quiet_active >= self.grace_s):
                return self._fire("completed-then-hung", 9, stderr)
            if self.active >= self.wall_s:
                # D 1951-1955: a marker seen means 9 even at the wall.
                return self._fire("wall-deadline", 9 if self.saw_marker else 10, stderr)
            remain = float(self.poll_s)
            while remain > 0:
                step = min(remain, _SLICE_S)
                time.sleep(step)
                remain -= step
                self.scan()
                if self.gate.pending is not None:
                    return self.on_signal()
                code = self.reap_if_exited()
                if code is not None:
                    return self._vendor_exit(code)
                if self.parent_gone():
                    return self.on_parent_death()

    def _credit(self, now):
        """Add one poll gap to the active budget. The first ``poll_s + 1`` seconds
        count; the rest is paused. A backward step adds nothing."""
        if now < self.last:
            return
        delta = now - self.last
        cap = self.poll_s + 1.0
        if delta <= cap:
            add_active = delta
            add_paused = 0.0
        else:
            add_active = cap
            add_paused = delta - cap
        if self.quiet_at is not None:
            # The active part of this gap is the leading interval [last, last+cap].
            active_hi = self.last + add_active
            lo = self.quiet_at if self.quiet_at > self.last else self.last
            if active_hi > lo:
                self.quiet_active += active_hi - lo
        self.active += add_active
        self.paused += add_paused
        self.last = now


def _append_line(path, line):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError:
        return
    try:
        os.write(fd, line.encode("utf-8"))
    except OSError:
        pass
    finally:
        os.close(fd)


def supervise(argv, *, cwd, env, outfile, wall_s, grace_s, kill_grace_s, poll_s,
              stderr=None, clock=time.monotonic, gate=None, parent_pid=None):
    """Run ``argv`` with stdout and stderr on ``outfile`` and watch it; return an Ending.

    ``outfile`` must already hold the header; marker scanning starts after it.
    Raises OSError when the child cannot be launched (nothing is left running).
    The caller owns exit-code policy for that case.

    ``gate`` is the caller's armed SignalGate for the whole invocation; without one a
    private gate covers this call. A signal already recorded, or a parent that is
    already gone, ends the call before any child exists (``launched`` False).
    ``parent_pid`` makes the loop check ``getppid`` every poll slice.
    """
    if gate is None:
        with SignalGate() as own:
            return _supervise(argv, cwd, env, outfile, wall_s, grace_s, kill_grace_s,
                              poll_s, stderr, clock, own, True, parent_pid)
    return _supervise(argv, cwd, env, outfile, wall_s, grace_s, kill_grace_s, poll_s,
                      stderr, clock, gate, False, parent_pid)


def _not_launched(outcome, rc, signum=None):
    return Ending(outcome, rc, None, None, signum, False, False)


def _supervise(argv, cwd, env, outfile, wall_s, grace_s, kill_grace_s, poll_s, stderr,
               clock, gate, own, parent_pid):
    """``own``: the gate is private, entered (blocked) and armed here after the spawn."""
    out_fd = os.open(outfile, os.O_WRONLY | os.O_APPEND)
    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
    except BaseException:
        os.close(out_fd)  # nothing else owns it yet: do not leak it past the error
        raise
    watch = None
    try:
        offset = os.fstat(out_fd).st_size
        # A shared gate is armed already: block HUP/INT/TERM across the spawn (D 1630).
        with (contextlib.nullcontext() if own else gate.hold()):
            if not own:
                if gate.pending is not None:
                    return _not_launched("signal", 128 + gate.pending, gate.pending)
                if parent_pid is not None and os.getppid() != parent_pid:
                    return _not_launched("parent_died", PARENT_DIED_RC)
            proc = subprocess.Popen(
                argv, stdin=null_fd, stdout=out_fd, stderr=out_fd, cwd=cwd, env=env,
                preexec_fn=child_setup)
            try:
                os.setpgid(proc.pid, proc.pid)  # D 1655: both sides, whoever wins
            except OSError:
                pass
            watch = _Watch(proc, gate, outfile, offset, wall_s, grace_s,
                           kill_grace_s, poll_s, clock, parent_pid)
            if own:
                gate.arm()
        # Vendor output stays with the child (D 1935-1938): drop our copies.
        os.close(out_fd)
        out_fd = -1
        try:
            return watch.run(stderr)
        except BaseException:
            watch.signal_teardown()  # never leave the child behind on an internal error
            raise
    finally:
        if watch is not None:
            watch.close()
        if out_fd >= 0:
            os.close(out_fd)
        os.close(null_fd)
