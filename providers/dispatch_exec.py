# no-opencc
"""``omni dispatch-exec``: the grok execution stage of dispatch-external.sh (D).

Synchronous. It reproduces D's header, read-only canary, grok argv, launch through
``omni host-exec``, watchdog, exit codes and note lines so that D's perl watchdog can
be replaced by one call. Its caller is an external dispatch script, dispatch-external.sh
(opt-in ``DISPATCH_GROK_VIA_OMNI=1``, on by default via dispatch-chain.sh). Seat
resolution, lease, PII, quota, payload freezing, account selection and the freeze
note stay in D.

Owner ruling: the caller gets a result record bound to this invocation
(``--invocation-id``, ``--result-file``; see dispatch_result), the credential source is
pinned (macOS only; ``--login xai`` plus ``--config-dir`` holding ``auth.json``, a
child environment cut down to the host-exec allowlist with every vendor auth/endpoint
variable removed, ``GROK_HOME`` set to the realpath, the vendor exec'd by the absolute
path that was checked), the prompt sent is the prompt scanned (``--prompt-sha256``),
every timing is explicit, and a dead caller (``--parent-pid``) stops the run.

Exit codes: the vendor's (128+signal when signalled) plus the ``EXIT_CODES`` table
below, which ``omni contract`` reports. The fence code is not touched: the child is
exactly ``omni host-exec`` as D launches it.
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections import namedtuple
from types import SimpleNamespace

from providers import dispatch_result, host_fence, kernel_sandbox, provider_accounts
from providers.dispatch_canary import run_read_only_canary
from providers.dispatch_supervision import SignalGate, supervise

PROVIDER = "grok"
# The omni-owned exit codes (every other code is the vendor's, 128+signal when
# signalled). ``omni contract`` reports this table verbatim.
EXIT_CODES = {
    "2": "bad arguments, an alias or a prompt that is not the one scanned",
    "4": "grok missing",
    "6": "credential source unproven (also every non-macOS run)",
    "8": "read-only containment refused",
    "9": "watchdog fired after a result marker",
    "10": "watchdog fired with no result marker",
    "125": "omni's own failure (parent died, internal error, record not published)",
    "126": "the host-exec wrapper could not start the vendor (recorded as supervisor_failure/start_unknown)",
    "127": "the host-exec wrapper could not start the vendor (recorded as supervisor_failure/start_unknown)",
    "129": "HUP received",
    "130": "INT received",
    "143": "TERM received",
}
_OMNI_CLI = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                         "omni_cli.py")
_MODES = ("read-only", "workspace", "strict")
LOGIN = "xai"
_AUTH_MAX_BYTES = 1 << 20
_MSG_UNPROVEN = "omni dispatch-exec: credential source unproven for resolved login\n"
_MSG_NOT_PUBLISHED = "omni dispatch-exec: result record not published\n"


def is_darwin():
    """Seam: D's ``host_contained`` is true on Darwin only (D 1433-1443)."""
    return sys.platform == "darwin"


def host_exec_prefix(provider, workdir, read_only):
    """Seam: the wrapper argv up to and including ``--`` (D 2104-2106).

    The wrapper is a Python process that runs before the fence is applied, so it starts
    with ``-E`` (PYTHON* variables ignored: no PYTHONPATH/PYTHONSTARTUP code) and ``-s``
    (no user site). Not ``-I``: that would drop the script directory from sys.path, which
    the ``providers`` imports need.
    """
    prefix = [sys.executable, "-E", "-s", _OMNI_CLI, "host-exec", "--provider", provider,
              "--workdir", workdir]
    if read_only:
        prefix.append("--read-only")
    return prefix + ["--"]


def resolve_grok():
    """Seam: the absolute realpath ``omni host-exec`` rewrites the vendor head to.

    host_fence._vendor_exec_path execs ``kernel_sandbox.resolve_agent_binary``'s answer,
    so passing that same path as argv[0] makes host-exec exec exactly the checked file.
    """
    return kernel_sandbox.resolve_agent_binary(PROVIDER)


def _bounded(lo, hi):
    def parse(text):
        if not re.fullmatch(r"[0-9]{1,9}", text):
            raise argparse.ArgumentTypeError("expected an integer %d..%d" % (lo, hi))
        value = int(text)
        if value < lo or value > hi:
            raise argparse.ArgumentTypeError("expected an integer %d..%d" % (lo, hi))
        return value
    return parse


def _nonempty(text):
    if not text.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _sha256(text):
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise argparse.ArgumentTypeError("expected 64 lowercase hex characters")
    return text


def _parser():
    parser = argparse.ArgumentParser(
        prog="omni dispatch-exec", allow_abbrev=False,
        description="Run one grok dispatch: header, read-only canary, launch via "
                    "host-exec, completion watchdog. Output goes raw to --outfile; "
                    "the ending is published to --result-file.")
    parser.add_argument("--provider", required=True, choices=(PROVIDER,))
    parser.add_argument("--model", required=True, type=_nonempty)
    parser.add_argument("--effort", required=True, type=_nonempty)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=_MODES, default=None,
                        help="default: read-only on macOS")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-sha256", required=True, type=_sha256,
                        help="sha256 of the prompt file's raw bytes")
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--trace-prompt", default=None,
                        help="path shown in the launch trace (default: --prompt-file)")
    parser.add_argument("--invocation-id", required=True,
                        help="32 lowercase hex characters naming this invocation")
    parser.add_argument("--result-file", required=True,
                        help="absolute path of the result record; must not exist yet")
    parser.add_argument("--login", required=True,
                        help="the login the credential must belong to (only 'xai')")
    parser.add_argument("--config-dir", required=True, type=_nonempty,
                        help="grok home holding auth.json; becomes GROK_HOME")
    parser.add_argument("--parent-pid", required=True, type=_bounded(2, 1 << 22),
                        help="pid of the caller; the run stops when it is gone")
    parser.add_argument("--wall-s", required=True, type=_bounded(1, 604800))
    parser.add_argument("--stall-grace-s", required=True, type=_bounded(1, 86400))
    parser.add_argument("--kill-grace-s", required=True, type=_bounded(1, 60))
    parser.add_argument("--poll-s", required=True, type=_bounded(1, 60))
    parser.add_argument("--canary-timeout-s", required=True, type=_bounded(1, 120))
    return parser


def sh_quote(text):
    """POSIX single quote (D 1012). Command substitution drops trailing newlines."""
    return "'%s'" % text.rstrip("\n").replace("'", "'\\''")


def dq_escape(text):
    """Keep a path literal inside the trace's double quotes (D 1017)."""
    text = text.rstrip("\n")
    for char in ("\\", '"', "$", "`"):
        text = text.replace(char, "\\" + char)
    return text


Prompt = namedtuple("Prompt", "text raw")


class PromptMismatch(Exception):
    """The bytes read are not the bytes the caller scanned, or not canonical."""


def read_prompt(path, expected_sha256):
    """The prompt file read once, as the exact text that goes into argv.

    The sha256 of the raw bytes must equal ``expected_sha256`` (the digest of what was
    scanned), and the bytes must be canonical: no NUL, no trailing LF, valid UTF-8, and
    they must survive the argv encoding unchanged. Nothing is dropped or stripped.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PromptMismatch("sha256 differs from --prompt-sha256")
    if b"\0" in raw or raw.endswith(b"\n"):
        raise PromptMismatch("prompt holds a NUL or ends with a newline")
    try:
        text = raw.decode("utf-8")
        same = os.fsencode(text) == raw
    except UnicodeError:
        raise PromptMismatch("prompt is not valid UTF-8") from None
    if not same:
        raise PromptMismatch("prompt does not survive the argv encoding")
    return Prompt(text, raw)


def grok_argv(binary, model, effort, prompt):
    """D 2078 (host-contained, sandbox off); ``binary`` is the checked absolute path."""
    return [binary, "--sandbox", "off", "--always-approve", "--reasoning-effort",
            effort, "-m", model, "--output-format", "streaming-messages-json",
            "-p", prompt]


def header_line(mode, canary, canary_at):
    """D 1491-1497; the layer token stays despite the new command."""
    return "dispatch-containment: layer=omni-host-exec sandbox=%s canary=%s canary_at=%s\n" % (
        mode, canary, canary_at)


def trace_text(mode, workdir, argv, trace_prompt, outfile):
    """The launch line D prints for grok (D 2085-2101). Not a replay: it keeps `>`."""
    tail = '"$(cat "%s")" </dev/null > "%s" 2>&1\n' % (dq_escape(trace_prompt),
                                                      dq_escape(outfile))
    text = "omni host-exec --provider %s --workdir %s " % (
        sh_quote(PROVIDER), sh_quote(workdir))
    if mode == "read-only":
        text += "--read-only "
    text += "-- "
    for arg in argv[:-1]:
        text += sh_quote(arg) + " "
    return text + tail


def _same_file(left, right):
    """D 1475-1485: same realpath, or the same device and inode (hard link)."""
    if os.path.realpath(left) == os.path.realpath(right):
        return True
    if os.path.exists(left) and os.path.exists(right):
        a, b = os.stat(left), os.stat(right)
        return a.st_dev == b.st_dev and a.st_ino == b.st_ino
    return False


def _has_login(data):
    """known_secrets' grok shape: top-level entries that are objects holding a non-empty
    string ``key`` or ``refresh_token`` (a flat top-level pair is accepted too)."""
    if not isinstance(data, dict):
        return False
    for holder in [data] + [entry for entry in data.values() if isinstance(entry, dict)]:
        for name in ("key", "refresh_token"):
            value = holder.get(name)
            if isinstance(value, str) and value.strip():
                return True
    return False


def credential_home(config_dir):
    """Realpath of ``config_dir`` when its auth.json proves a login, else None.

    ``auth.json`` must be a regular file (no final symlink) owned by this user whose
    JSON object holds a non-empty ``key`` or ``refresh_token``.
    """
    try:
        home = os.path.realpath(config_dir)
        if not os.path.isdir(home):
            return None
        fd = os.open(os.path.join(home, "auth.json"),
                     os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_size > _AUTH_MAX_BYTES):
            return None
        raw = os.read(fd, _AUTH_MAX_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    return home if _has_login(data) else None


@contextlib.contextmanager
def _pass_list_closed():
    """``allowlisted_child_env`` reads the operator's pass list from os.environ itself, so
    the only way to compute the env without it is to take it out for the call."""
    saved = os.environ.pop(host_fence.CHILD_ENV_PASS_VAR, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[host_fence.CHILD_ENV_PASS_VAR] = saved


def pinned_env(home):
    """The child environment: host-exec's allowlist computed with the operator's pass list
    closed (nothing listed there reaches the wrapper), every vendor auth/endpoint name
    removed, every PYTHON* name removed, GROK_HOME pinned to the checked home.

    The child is a Python wrapper that runs before fencing: a name that reaches it (a
    listed ``PYTHONPATH``, ``PYTHONSTARTUP``) could run code that restores a credential.
    """
    with _pass_list_closed():
        env = host_fence.allowlisted_child_env(dict(os.environ), PROVIDER)
    env = provider_accounts.account_env(env, PROVIDER)
    env.pop(host_fence.CHILD_ENV_PASS_VAR, None)
    for name in [name for name in env if name.startswith("PYTHON")]:
        del env[name]
    env["GROK_HOME"] = home
    return env


class _Usage(Exception):
    pass


def _err(text):
    sys.stderr.write(text)
    sys.stderr.flush()


_PATH_FLAGS = ("prompt_file", "outfile", "trace_prompt")


def _check_paths(ns):
    if not os.path.isdir(ns.workdir):
        raise _Usage("--workdir is not a directory")
    if not os.path.isfile(ns.prompt_file) or not os.access(ns.prompt_file, os.R_OK):
        raise _Usage("--prompt-file is not a readable file")
    if not os.path.isdir(os.path.dirname(os.path.abspath(ns.outfile))):
        raise _Usage("--outfile parent is not a directory")
    # Absolute against the caller's cwd now; the child runs with cwd = workdir.
    ns.config_dir = os.path.abspath(ns.config_dir)


def _result_problem(result_file, paths):
    """Why the record cannot go to ``result_file`` (unusable: nothing can be recorded)."""
    why = dispatch_result.result_file_problem(result_file)
    if why:
        return "--result-file " + why
    for other in paths:
        if other is not None and _same_file(result_file, other):
            return "--result-file aliases another path"
    return None


def _scan(args):
    """Id, result file and the paths it must not alias, read straight from the tokens.

    No argparse involved, so it works whatever else is wrong with the command line (a
    value-less trailing flag, an unknown flag). The last flag that carries a value wins.
    """
    found = dict.fromkeys(("invocation_id", "result_file") + _PATH_FLAGS)
    for at, token in enumerate(args):
        for name in found:
            flag = "--" + name.replace("_", "-")
            if token == flag:
                following = args[at + 1] if at + 1 < len(args) else None
                if following is not None and not following.startswith("-"):
                    found[name] = following
            elif token.startswith(flag + "="):
                found[name] = token[len(flag) + 1:]
    return SimpleNamespace(**found)


def _identity_usable(seen):
    return (dispatch_result.valid_invocation_id(seen.invocation_id)
            and _result_problem(seen.result_file,
                                [getattr(seen, name) for name in _PATH_FLAGS]) is None)


class _Record:
    """The one terminal record of this invocation (see dispatch_result)."""

    _KEEP = object()

    def __init__(self, invocation_id, result_file, started, gate, parent_pid=None):
        self.invocation_id = invocation_id
        self.result_file = result_file
        self.started = started
        self.gate = gate
        self.parent_pid = parent_pid
        self.spawned = False  # becomes None once a vendor-capable process may exist

    def finish(self, outcome, detail, exit_code, *, spawned=_KEEP, vendor_rc=None,
               vendor_signal=None, signum=None, marker=False, decided=False):
        """Publish and return omni's exit status (125 when the record is not published).

        Before the run is decided, a signal already received wins and ignoring starts
        here (one step), so the first HUP/INT/TERM fixes outcome and rc. Otherwise a
        caller that is gone wins over any refusal or failure decided in the same window.
        """
        if not decided:
            pending = self.gate.commit() if self.gate is not None else None
            if pending is not None:
                outcome, detail, exit_code = "signal", "none", 128 + pending
                vendor_rc = vendor_signal = None
                signum = pending
            elif (self.parent_pid is not None and detail != "parent_died"
                    and os.getppid() != self.parent_pid):
                outcome, detail, exit_code = "supervisor_failure", "parent_died", 125
                vendor_rc = vendor_signal = None
        if spawned is self._KEEP:
            spawned = self.spawned
        try:
            record = dispatch_result.build(
                self.invocation_id, outcome, detail, spawned, vendor_rc, vendor_signal,
                signum, exit_code, int((time.monotonic() - self.started) * 1000), marker)
            dispatch_result.publish(self.result_file, record)
        except (dispatch_result.PublishError, ValueError):
            _err(_MSG_NOT_PUBLISHED)
            return 125
        return exit_code

    def refuse(self, detail, exit_code):
        return self.finish("refused_before_spawn", detail, exit_code)

    def signalled(self, signum):
        return self.finish("signal", "none", 128 + signum, signum=signum)

    def parent_died(self, spawned=_KEEP):
        return self.finish("supervisor_failure", "parent_died", 125, spawned=spawned)


def main(argv=None):
    started = time.monotonic()
    args = list(sys.argv[1:] if argv is None else argv)
    # One gate for the whole invocation, from the first parse step to publication: a
    # HUP/INT/TERM is recorded, never fatal, and decides the ending.
    with SignalGate() as gate:
        gate.arm()
        return _main(args, gate, started)


def _unrecorded(gate, code):
    """Exit status for an ending that has no record: a signal received meanwhile wins."""
    pending = gate.commit()
    return 128 + pending if pending is not None else code


def _main(args, gate, started):
    seen = _scan(args)
    try:
        ns = _parser().parse_args(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        if code != 0 and _identity_usable(seen):
            # The id and file are usable, so the caller still gets its record.
            return _Record(seen.invocation_id, seen.result_file, started, gate).refuse(
                "arguments", code)
        return _unrecorded(gate, code)
    if not dispatch_result.valid_invocation_id(ns.invocation_id):
        _err("omni dispatch-exec: error: --invocation-id must be 32 lowercase hex "
             "characters\n")
        return _unrecorded(gate, 2)
    trace_prompt = ns.trace_prompt if ns.trace_prompt is not None else ns.prompt_file
    problem = _result_problem(ns.result_file, [ns.prompt_file, ns.outfile, trace_prompt])
    if problem:
        _err("omni dispatch-exec: error: %s\n" % problem)
        return _unrecorded(gate, 2)
    record = _Record(ns.invocation_id, ns.result_file, started, gate, ns.parent_pid)
    try:
        return _invoke(ns, gate, record, trace_prompt)
    except Exception as exc:  # a bug after validation: say so in the record
        _err("omni dispatch-exec: internal error: %s: %s\n" % (type(exc).__name__, exc))
        return record.finish("supervisor_failure", "internal", 125)


def _interrupted(ns, gate, record):
    """Exit status when a signal arrived or the caller is gone, else None."""
    if gate.pending is not None:
        return record.signalled(gate.pending)
    if os.getppid() != ns.parent_pid:
        return record.parent_died()
    return None


def _invoke(ns, gate, record, trace_prompt):
    stopped = _interrupted(ns, gate, record)  # a signal during validation, or no caller
    if stopped is not None:
        return stopped
    try:
        _check_paths(ns)
    except _Usage as exc:
        _err("omni dispatch-exec: error: %s\n" % exc)
        return record.refuse("arguments", 2)

    darwin = is_darwin()
    mode = ns.mode or "read-only"
    workdir = os.path.abspath(ns.workdir)

    # The credential source is provable on macOS for the xai login only: any other
    # platform or login is refused before anything is read or spawned.
    if not darwin or ns.login != LOGIN:
        _err(_MSG_UNPROVEN)
        return record.refuse("credential_unproven", 6)

    try:
        prompt = read_prompt(ns.prompt_file, ns.prompt_sha256)
    except PromptMismatch as exc:
        _err("omni dispatch-exec: error: prompt refused: %s\n" % exc)
        return record.refuse("prompt_mismatch", 2)
    except OSError as exc:
        _err("omni dispatch-exec: error: --prompt-file unreadable: %s\n" % exc)
        return record.refuse("arguments", 2)

    # An outfile that is the work order or its trace path would be replaced by
    # the header (D 1464-1490). Refuse before anything is truncated.
    for source in (ns.prompt_file, trace_prompt):
        if _same_file(source, ns.outfile):
            _err("dispatch-external: outfile aliases the work order\n")
            return record.refuse("arguments", 2)

    home = credential_home(ns.config_dir)
    if home is None:
        _err(_MSG_UNPROVEN)
        return record.refuse("credential_unproven", 6)
    env = pinned_env(home)
    binary = resolve_grok()
    if not isinstance(binary, str) or not os.path.isabs(binary):
        # D 1004-1006: checked before the header is written.
        _err("capacity incident: grok binary not found; model=%s; "
             "foreman fallback required per 51 §9\n" % ns.model)
        return record.refuse("launch_unavailable", 4)

    canary, canary_at = "n/a", "-"
    if mode == "read-only":
        result = run_read_only_canary(
            wrapper_prefix=host_exec_prefix(PROVIDER, workdir, True),
            provider=PROVIDER, workdir=workdir, env=env, timeout_s=ns.canary_timeout_s,
            gate=gate, parent_pid=ns.parent_pid)
        if result.status == "signal":
            return record.signalled(result.detail)
        if result.status == "parent_died":
            return record.parent_died()
        if result.status != "pass":
            _err("dispatch-external: read-only containment refused: %s\n" % result.detail)
            return record.refuse("containment", 8)
        canary, canary_at = "pass", result.detail

    # Between canary and launch: a signal or a dead caller ends it here, and the
    # credential is proven once more, as close to the spawn as the header allows.
    stopped = _interrupted(ns, gate, record)
    if stopped is not None:
        return stopped
    if credential_home(ns.config_dir) != home:
        _err(_MSG_UNPROVEN)
        return record.refuse("credential_unproven", 6)

    vendor = grok_argv(binary, ns.model, ns.effort, prompt.text)
    if os.fsencode(vendor[-1]) != prompt.raw:
        _err("omni dispatch-exec: error: prompt refused: argv bytes differ\n")
        return record.refuse("prompt_mismatch", 2)
    try:
        with open(ns.outfile, "w", encoding="utf-8") as handle:
            handle.write(header_line(mode, canary, canary_at))
    except OSError:
        _err("dispatch-external: cannot write containment header\n")
        return record.refuse("arguments", 2)

    sys.stdout.write(trace_text(mode, workdir, ["grok"] + vendor[1:], trace_prompt,
                                ns.outfile))
    sys.stdout.flush()
    child = host_exec_prefix(PROVIDER, workdir, mode == "read-only") + vendor
    record.spawned = None  # from here the wrapper may have started the vendor
    try:
        ending = supervise(child, cwd=workdir, env=env, outfile=ns.outfile,
                           wall_s=ns.wall_s, grace_s=ns.stall_grace_s,
                           kill_grace_s=ns.kill_grace_s, poll_s=ns.poll_s,
                           gate=gate, parent_pid=ns.parent_pid)
    except OSError as exc:
        _err("dispatch-exec: cannot launch grok: %s\n" % exc)
        return record.finish("supervisor_failure", "start_unknown", 127)  # D 1639
    return _record_ending(record, ending)


def _record_ending(record, ending):
    """Publish what ended the supervised run (already decided: signals are ignored)."""
    common = dict(spawned=None if ending.launched else False, marker=ending.marker,
                  decided=True)
    if ending.outcome == "vendor_exit":
        if ending.vendor_rc in (126, 127):
            # host-exec could not start the vendor (fence refusal or exec failure).
            return record.finish("supervisor_failure", "start_unknown", ending.rc, **common)
        return record.finish("vendor_exit", "none", ending.rc, vendor_rc=ending.vendor_rc,
                             vendor_signal=ending.vendor_signal, **common)
    if ending.outcome in ("stall", "wall_timeout"):
        return record.finish(ending.outcome, "none", ending.rc, **common)
    if ending.outcome == "signal":
        return record.finish("signal", "none", ending.rc, signum=ending.signal, **common)
    return record.finish("supervisor_failure", "parent_died", ending.rc, **common)
