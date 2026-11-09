# no-opencc
"""Read-only containment canary for ``omni dispatch-exec`` (D = dispatch-external.sh).

A read-only dispatch launches only after the same ``omni host-exec --read-only``
wrapper, with the same --workdir string, has denied a write in this workdir
during this run. Nothing is cached. The probe is a private 0700 script outside
the workdir, never the vendor CLI: host-exec inserts provider flags after
argv[0], so the script skips arguments until its marker, prints a ready line
carrying a per-run nonce, then tries to create the target.

Pass needs all of: the exact ready line, a denial line naming the target path
(the path holds the nonce, denial phrases are read only after the path is
removed from the line), adapter status 23, and the target still absent. Any
other status, a timeout, a launch failure or ambiguous evidence refuses. A
target that was created is removed, and that refuses too.

Plain subprocess with a group KILL bounds the probe (D uses perl alarm); the
probe's stdout and stderr go to separate private files. A descendant that left
the probe's process group is outside that kill (D 1051-1053).

The invocation's SignalGate may be passed in: a HUP/INT/TERM recorded on it before
or during the probe ends the canary with ``signal`` (the probe is torn down and no
vendor is ever started). ``parent_pid`` makes the wait loop end with ``parent_died``
when the process that started omni is gone.
"""
import contextlib
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
from collections import namedtuple

from providers.dispatch_supervision import SignalGate, child_setup, group_kill

CanaryResult = namedtuple("CanaryResult", "status detail")  # pass | refuse | signal | parent_died

MARKER = "DISPATCH_RO_CANARY"
READY_PREFIX = "dispatch-ro-canary: writing "
PROBE_EXIT = 23  # D 1109: the probe's own denial status
TIMEOUT_RC = 142  # D 1257: the bound expired
_DENIAL_PHRASES = (  # D 1312-1320
    "operation not permitted",
    "permission denied",
    "read-only file system",
    "read-only filesystem",
    "operation not allowed",
    "eperm",
    "deny file-write",
)

# D 1093-1110. Marker and nonce are baked in because grok makes argv[1] a vendor
# flag. The write runs in a subshell so a failed redirect cannot abort dash.
_SCRIPT = """#!/bin/sh
marker='@MARKER@'
nonce='@NONCE@'
while [ "$#" -gt 0 ] && [ "$1" != "$marker" ]; do
    shift
done
if [ "${1:-}" != "$marker" ] || [ "$#" -lt 2 ]; then
    printf '%s\\n' 'dispatch-ro-canary: marker missing' >&2
    exit 2
fi
shift
printf '%s\\n' "dispatch-ro-canary: writing $nonce" >&2
if ( printf '' > "$1" ); then
    exit 0
fi
exit 23
"""


def _script_parents():
    """D 1067: /tmp first (host-exec is known to exec scripts there), then TMPDIR."""
    return ["/tmp", os.environ.get("TMPDIR") or "/tmp"]


def _inside(path, base):
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def _install_script(workdir, nonce):
    """A 0700 script in a fresh 0700 directory outside the workdir, or None (D 1062-1121)."""
    phys_workdir = os.path.realpath(workdir)
    for parent in _script_parents():
        if not parent or not os.path.isdir(parent) or not os.access(parent, os.W_OK):
            continue
        if _inside(os.path.realpath(parent), phys_workdir):
            continue
        try:
            script_dir = tempfile.mkdtemp(prefix="dispatch-ro-canary.", dir=parent)
        except OSError:
            continue
        try:
            if _inside(os.path.realpath(script_dir), phys_workdir):
                raise OSError("inside workdir")
            os.chmod(script_dir, 0o700)
            script = os.path.join(script_dir, "probe")
            fd = os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
            with os.fdopen(fd, "w") as handle:
                handle.write(_SCRIPT.replace("@MARKER@", MARKER).replace("@NONCE@", nonce))
            os.chmod(script, 0o700)
        except OSError:
            shutil.rmtree(script_dir, ignore_errors=True)
            continue
        return script_dir, script
    return None


def _empty_slot(workdir, prefix):
    """A non-hidden exclusive name in the workdir root, created then removed (D 1386-1403).

    A sandbox that guards only dotfiles must not be able to produce a pass, and
    an unwritable workdir cannot prove an adapter denial. None means inconclusive.
    """
    path = None
    try:
        fd, path = tempfile.mkstemp(prefix=prefix, dir=workdir)
        os.close(fd)
        os.unlink(path)
    except OSError:
        # mkstemp may have created the entry before close or unlink failed: leave
        # nothing of this run in the workdir (best effort).
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        return None
    if os.path.lexists(path):
        return None  # reappeared after this run's unlink
    return path


def _drop(path):
    """True when nothing is left at path."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def classify(nonce, target, out_path, err_path):
    """0 denial proven, 3 marker never reached, 2 inconclusive (D 1295-1338)."""
    ready = READY_PREFIX + nonce
    chunks = []
    for name in (out_path, err_path):
        try:
            with open(name, encoding="utf-8", errors="replace") as handle:
                chunks.append(handle.read())
        except OSError:
            pass
    ready_seen = False
    denial = False
    marker = False
    target_low = target.lower()
    nonce_low = nonce.lower()
    for line in "\n".join(chunks).splitlines():
        if line == ready:
            ready_seen = True
            continue
        low = line.lower()
        if "dispatch-ro-canary: marker missing" in low:
            marker = True
        if target_low in low and nonce_low in low:
            rest = low.replace(target_low, "")
            if any(phrase in rest for phrase in _DENIAL_PHRASES):
                denial = True
    if marker and not ready_seen:
        return 3
    if ready_seen and denial:
        return 0
    return 2


def _run_probe(argv, workdir, env, out_path, err_path, timeout_s, gate=None, parent_pid=None):
    """('exit', status) | ('timeout', 142) | ('signal', signum) | ('parent_died', None)
    | ('error', None). ``gate``: the caller's armed gate (else a private one)."""
    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
    except OSError:
        return ("error", None)
    try:
        try:
            out = open(out_path, "wb")
        except OSError:
            return ("error", None)
        try:
            err = open(err_path, "wb")
        except OSError:
            out.close()
            return ("error", None)
        with out, err, contextlib.ExitStack() as stack:
            shared = gate is not None
            if not shared:
                gate = stack.enter_context(SignalGate())
            try:
                # Shared: block HUP/INT/TERM across the spawn. Private: the gate is
                # blocked from entry until arm.
                with (gate.hold() if shared else contextlib.nullcontext()):
                    if shared:
                        if gate.pending is not None:
                            return ("signal", gate.pending)
                        if parent_pid is not None and os.getppid() != parent_pid:
                            return ("parent_died", None)
                    proc = subprocess.Popen(argv, stdin=null_fd, stdout=out, stderr=err,
                                            cwd=workdir, env=env, preexec_fn=child_setup)
                    try:
                        os.setpgid(proc.pid, proc.pid)
                    except OSError:
                        pass
                    if not shared:
                        gate.arm()
            except OSError:
                return ("error", None)
            deadline = time.monotonic() + timeout_s
            outcome = None
            while outcome is None:
                if gate.pending is not None:
                    outcome = ("signal", gate.pending)
                    break
                if parent_pid is not None and os.getppid() != parent_pid:
                    outcome = ("parent_died", None)
                    break
                code = proc.poll()
                if code is not None:
                    outcome = ("exit", 128 - code if code < 0 else code)
                    break
                if time.monotonic() >= deadline:
                    outcome = ("timeout", TIMEOUT_RC)
                    break
                time.sleep(0.02)
            if not shared:
                gate.ignore()
            # D 1236-1244, 1255: the group dies with the probe, then is awaited.
            group_kill(proc.pid)
            if proc.returncode is None:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait()
            settle = time.monotonic() + 1.0
            while time.monotonic() < settle:
                try:
                    os.killpg(proc.pid, 0)
                except OSError:
                    break
                time.sleep(0.02)
            return outcome
    finally:
        os.close(null_fd)


def run_read_only_canary(*, wrapper_prefix, provider, workdir, env, timeout_s,
                         gate=None, parent_pid=None):
    """Prove the read-only wrapper denies a write in ``workdir`` right now.

    ``wrapper_prefix`` is the complete ``host-exec ... --`` argv up to and
    including ``--``. Returns CanaryResult: ``pass`` (detail = UTC canary_at),
    ``refuse`` (detail = D's reason text), ``signal`` (detail = signum) or
    ``parent_died`` (detail None).
    """
    nonce = secrets.token_hex(16)
    if _empty_slot(workdir, "dispatch-ro-canary.") is None:
        return CanaryResult("refuse", "inconclusive")
    target = _empty_slot(workdir, "dispatch-ro-canary.%s." % nonce)
    if target is None:
        return CanaryResult("refuse", "inconclusive")
    installed = _install_script(workdir, nonce)
    if installed is None:
        return CanaryResult("refuse", "inconclusive")
    script_dir, script = installed
    io_dir = None
    try:
        # An infrastructure OSError (capture dir, canary.out/err) is a refusal, not a crash.
        io_dir = tempfile.mkdtemp(prefix="dispatch-ro-canary-io.")
        out_path = os.path.join(io_dir, "canary.out")
        err_path = os.path.join(io_dir, "canary.err")
        kind, value = _run_probe(list(wrapper_prefix) + [script, MARKER, target],
                                 workdir, env, out_path, err_path, timeout_s,
                                 gate=gate, parent_pid=parent_pid)
        shutil.rmtree(script_dir, ignore_errors=True)
        if kind == "signal":
            _drop(target)
            return CanaryResult("signal", value)
        if kind == "parent_died":
            _drop(target)
            return CanaryResult("parent_died", None)
        if kind in ("timeout", "error"):
            # The group is dead. A target it already created is removed first.
            _drop(target)
            return CanaryResult("refuse", "inconclusive")
        if os.path.lexists(target):
            _drop(target)
            return CanaryResult("refuse", "write succeeded")
        verdict = classify(nonce, target, out_path, err_path)
        if os.path.lexists(target):
            _drop(target)
            return CanaryResult("refuse", "write succeeded")
        if verdict == 3:
            return CanaryResult("refuse", "%s canary argv rewrite unsurvivable" % provider)
        # Output alone is not a pass. The probe must exit 23 (D 1350).
        if verdict == 0 and value == PROBE_EXIT:
            return CanaryResult("pass", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return CanaryResult("refuse", "inconclusive")
    except OSError:
        _drop(target)
        return CanaryResult("refuse", "inconclusive")
    except BaseException:
        _drop(target)
        raise
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)
        if io_dir is not None:
            shutil.rmtree(io_dir, ignore_errors=True)
