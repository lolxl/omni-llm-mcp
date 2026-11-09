# no-opencc
"""Shared CLI runner with optional zellij watch pane (viewer + PIPE).

Watch is opt-in. Opening a pane requires an explicit tab (tool arg or env).
Invalid high-priority values fail closed (no downgrade). Zero hold: pane closes
as soon as the CLI finishes. MCP returns as soon as CLI I/O is collected.

Env:
  OMNI_LLM_WATCH_PANE=1|true|yes|on
  OMNI_LLM_ZELLIJ_SESSION=<name>
  OMNI_LLM_ZELLIJ_TAB=<tab name>
  OMNI_LLM_ZELLIJ_TAB_ID=<id>   # advanced; tool schema does not expose id

Tool overrides (via set_watch_context): watch_pane, zellij_session, zellij_tab.
"""
import asyncio
import contextvars
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from providers import known_secrets
from providers.config import WATCH_FINISH_DRAIN_S, WATCH_PANE_MAX_AGE_PAD_S
from providers.logutil import log_event

ENV_WATCH = "OMNI_LLM_WATCH_PANE"
ENV_SESSION = "OMNI_LLM_ZELLIJ_SESSION"
ENV_TAB = "OMNI_LLM_ZELLIJ_TAB"
ENV_TAB_ID = "OMNI_LLM_ZELLIJ_TAB_ID"

_UNSET = object()
_STDIN_CHUNK_SIZE = 64 * 1024
# After a kill, the pipes get this long to hand over what the child had
# already written; a descendant that still holds them open must not stall the
# return.
_POST_KILL_DRAIN_S = 2.0
# stall_marker line buffering: a line longer than the cap is dropped (it is
# no marker line), and a pending partial line is probed only while it is
# small, so one endless unterminated line cannot make every read quadratic.
_STALL_LINE_CAP = 8 * 1024 * 1024
_STALL_PARTIAL_PROBE_MAX = 1024 * 1024
_watch_ctx = contextvars.ContextVar("omni_cli_watch_ctx", default=None)
_watch_finish_tasks = set()

# Test seams (patch these in hermetic tests).
_which = shutil.which
_run_sync = subprocess.run


async def _spawn(*args, **kwargs):
    """Wrapper so tests can patch asyncio.create_subprocess_exec at runtime."""
    return await asyncio.create_subprocess_exec(*args, **kwargs)


def set_watch_context(watch_pane=_UNSET, zellij_session=_UNSET, zellij_tab=_UNSET):
    """Install per-call overrides. Use finally + reset_watch_context(token)."""
    return _watch_ctx.set({
        "watch_pane": watch_pane,
        "zellij_session": zellij_session,
        "zellij_tab": zellij_tab,
    })


def reset_watch_context(token):
    _watch_ctx.reset(token)


def _ctx_get(key, default=_UNSET):
    ctx = _watch_ctx.get()
    if not ctx:
        return default
    return ctx.get(key, default)


def _truthy(value):
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def watch_enabled():
    wp = _ctx_get("watch_pane", _UNSET)
    if wp is not _UNSET:
        return _truthy(wp)
    return _truthy(os.environ.get(ENV_WATCH, ""))


def list_zellij_sessions():
    zellij = _which("zellij")
    if not zellij:
        return []
    try:
        proc = _run_sync(
            [zellij, "list-sessions", "-s"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def list_zellij_tabs(session):
    """Return list of {name, tab_id} for session, or []."""
    zellij = _which("zellij")
    if not zellij or not session:
        return []
    try:
        proc = _run_sync(
            [zellij, "--session", session, "action", "list-tabs", "-j"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return []
    out = []
    if not isinstance(data, list):
        return []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        tid = item.get("tab_id")
        if name is None or tid is None:
            continue
        out.append({"name": str(name), "tab_id": tid})
    return out


def resolve_zellij_session():
    """Fail-closed session resolve. None => skip watch."""
    sessions = list_zellij_sessions()
    provided = _ctx_get("zellij_session", _UNSET)
    if provided is not _UNSET:
        name = str(provided or "").strip()
        if name and name in sessions:
            return name
        return None
    env = (os.environ.get(ENV_SESSION) or "").strip()
    if env:
        if env in sessions:
            return env
        return None
    # Weak inference only when nothing was explicitly provided.
    current = (os.environ.get("ZELLIJ_SESSION_NAME") or "").strip()
    if current and current in sessions:
        return current
    if len(sessions) == 1:
        return sessions[0]
    return None


def resolve_zellij_tab_id(session):
    """Fail-closed tab resolve. Requires explicit tab; None => skip watch."""
    if not session:
        return None
    tabs = list_zellij_tabs(session)
    by_name = {}
    by_id = {}
    for t in tabs:
        by_name.setdefault(t["name"], []).append(t["tab_id"])
        by_id[str(t["tab_id"])] = t["tab_id"]

    provided = _ctx_get("zellij_tab", _UNSET)
    if provided is not _UNSET:
        name = str(provided or "").strip()
        ids = by_name.get(name) or []
        if len(ids) == 1:
            return ids[0]
        return None

    env_name = (os.environ.get(ENV_TAB) or "").strip()
    if env_name:
        ids = by_name.get(env_name) or []
        if len(ids) == 1:
            return ids[0]
        return None

    env_id = (os.environ.get(ENV_TAB_ID) or "").strip()
    if env_id:
        if env_id in by_id:
            return by_id[env_id]
        return None

    return None


def _parse_pane_id(text):
    if not text:
        return None
    for line in reversed(str(text).splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("terminal_") or line.startswith("plugin_"):
            return line
        if line.isdigit():
            return "terminal_%s" % line
    return None


def _viewer_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_viewer.py")


async def _open_watch_pane(session, tab_id, log_path, done_path, pane_name, max_s):
    zellij = _which("zellij")
    if not zellij:
        return None
    token = uuid.uuid4().hex[:8]
    name = ("%s-%s" % (pane_name or "omni", token))[:40]
    viewer = _viewer_path()
    py = sys.executable or "python3"
    # Ensure log exists before viewer attaches.
    with open(log_path, "ab"):
        pass
    proc = None
    try:
        proc = await _spawn(
            zellij, "--session", session, "run",
            "--tab-id", str(tab_id),
            "-n", name,
            "-f",
            "--width", "50%",
            "--height", "40%",
            "--x", "48%",
            "--y", "8%",
            "-c",
            "--",
            py, viewer, log_path, done_path, str(max_s),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=8)
        except asyncio.TimeoutError:
            await _kill_proc_tree(proc)
            log_event("watch_pane_open_error", detail="zellij run timed out")
            return None
        except asyncio.CancelledError:
            await _kill_proc_tree(proc)
            raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if proc is not None:
            await _kill_proc_tree(proc)
        log_event("watch_pane_open_error", detail=str(e)[:200])
        return None
    text = ((out_b or b"") + b"\n" + (err_b or b"")).decode(errors="replace")
    pane_id = _parse_pane_id(text)
    log_event(
        "watch_pane_open",
        session=session,
        tab_id=tab_id,
        pane_id=pane_id,
        pane=name,
        ok=bool(pane_id),
    )
    return pane_id


async def _close_pane(session, pane_id):
    if not session or not pane_id:
        return
    zellij = _which("zellij")
    if not zellij:
        return
    try:
        proc = await _spawn(
            zellij, "--session", session, "action", "close-pane", "-p", str(pane_id),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        pass


async def _finish_watch_async(session, pane_id, work_dir):
    """Background: force-close pane and drop temp dir. Must not gate MCP return."""
    try:
        await _close_pane(session, pane_id)
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def _schedule_watch_finish(session, pane_id, work_dir):
    """Fire-and-forget watch teardown so run_cli can return immediately."""
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_finish_watch_async(session, pane_id, work_dir))
        _watch_finish_tasks.add(task)
        task.add_done_callback(_watch_finish_tasks.discard)
    except Exception:
        # No running loop (tests) or schedule failed — best-effort sync cleanup.
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


async def drain_watch_finishers(timeout_s=WATCH_FINISH_DRAIN_S):
    """Wait briefly for scheduled watch teardowns before a one-shot loop closes."""
    try:
        pending = {task for task in _watch_finish_tasks if not task.done()}
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, float(timeout_s)))
    except BaseException:
        pass


def _killpg_if_unreaped(proc):
    """SIGKILL the whole process group, but only while the leader looks unreaped.

    POSIX does not reuse a process ID while a process group with that ID
    exists, so the number cannot be recycled while any member of our group (a
    lingering descendant or an unreaped zombie leader) is alive. The recycle
    window only opens once the group is empty, when there is nothing of ours
    left to kill. ``proc.returncode is None`` does not prove the leader is
    unreaped: asyncio's default watcher on macOS (ThreadedChildWatcher) calls
    waitpid on its own thread and sets returncode on the loop later, so the pid
    can already be free while returncode is still None. This check therefore
    has the same small window as ``agent_lane._clear_process_groups`` and only
    narrows it. macOS hands out pids sequentially and wraps at 99999, so a pid
    freed by the reap is reused only after the counter has gone round the whole
    range, which is not a realistic event within the milliseconds before
    killpg; that is the accepted residual of #186. Hosts with randomized pid
    allocation would need an owned anchor that kills its own group (a member
    calling ``kill(0, SIGKILL)``, e.g. a tiny POSIX sh launcher); not built.
    After a clean reap we deliberately do nothing: the paths that strand
    children (timeout, CancelledError, exception) all arrive here unreaped, and
    on the clean path both pipes already hit EOF, so no descendant still holds
    them. A descendant that closed its pipes survives a clean exit, and one
    that called ``setsid`` is not reached.

    Returns True if the group signal was delivered.
    """
    if proc is None or proc.returncode is not None:
        return False
    pid = getattr(proc, "pid", None)
    if not pid:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        # Not a group leader (start_new_session unsupported) or already gone.
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return False


async def _kill_proc_tree(proc):
    """Reap the CLI: group-kill while that is still safe, then wait."""
    if proc is None:
        return
    _killpg_if_unreaped(proc)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass


class _MarkerScan:
    """Finds the first stdout line that ``marker`` accepts, and tracks when the
    last stdout byte arrived (the stall watchdog's inactivity clock).

    Lines are split on newlines across reads. A pending line with no newline yet
    is offered to ``marker`` too (and again at EOF): a child that prints its
    final line unterminated and then idles would otherwise never be seen. The
    callable must therefore return False for a line that is merely incomplete.
    A raising marker counts as no match.

    A match on a newline-terminated line, or at EOF, is final: only the clock
    keeps moving afterwards. A match on the unterminated pending line is
    provisional: more bytes on that same line re-judge it as extended, and if it
    no longer matches ``seen`` and ``event`` are cleared again. ``seen`` and
    ``event`` always agree. A line over the cap is dropped up to its next
    newline, so its tail is not judged as if it were a new line.
    """

    def __init__(self, marker):
        self._marker = marker
        self._buf = b""
        self._final = False
        self._skipping = False
        self.seen = False
        self.last_byte = time.monotonic()
        self.event = asyncio.Event()

    def _hit(self, line, final):
        try:
            matched = bool(self._marker(line))
        except Exception:
            matched = False
        if matched:
            self.seen = True
            self.event.set()
            if final:
                self._final = True
                self._buf = b""
        return matched

    def feed(self, data):
        self.last_byte = time.monotonic()
        if self._final:
            return
        if self.seen:
            # Only a provisional match gets here: judge the line again below.
            self.seen = False
            self.event.clear()
        if self._skipping:
            _, newline, data = data.partition(b"\n")
            if not newline:
                return
            self._skipping = False
        *lines, self._buf = (self._buf + data).split(b"\n")
        for line in lines:
            if self._hit(line, True):
                return
        if len(self._buf) > _STALL_LINE_CAP:
            self._buf = b""
            self._skipping = True
        elif self._buf and len(self._buf) <= _STALL_PARTIAL_PROBE_MAX:
            self._hit(self._buf, False)

    def finish(self):
        if not self._final and self._buf:
            self._hit(self._buf, True)


async def _cancel_tasks(*tasks):
    live = [t for t in tasks if t is not None]
    for t in live:
        t.cancel()
    if live:
        await asyncio.gather(*live, return_exceptions=True)


def decode_output(data):
    """The child's output as text. Bytes decode with ``errors="replace"``,
    exactly as before, so what a caller parses is unchanged; text passes
    through.

    A known credential holding bytes that are not UTF-8 comes out spelled by
    the decoder in a way that depends on the neighbouring bytes, so how it was
    spelled here is registered (``known_secrets.register_decoded_spans``) and
    the failure text built from this output is masked against it.
    """
    if isinstance(data, str):
        return data
    data = data or b""
    known_secrets.register_decoded_spans(data)
    return data.decode(errors="replace")


async def run_cli(
    argv,
    *,
    stdin_data=None,
    stdin_file=None,
    cwd=None,
    env=None,
    timeout_s=180,
    pane_name="omni-cli",
    log_path=None,
    err_log_path=None,
    on_spawn=None,
    watch_stdout=True,
    pre_spawn=None,
    stall_marker=None,
    stall_grace_s=None,
):
    """Run argv under PIPE; optionally mirror output to a zellij floating pane.

    Returns:
      {
        "returncode": int|None,
        "stdout": str,
        "stderr": str,
        "timed_out": bool,
        "watched": bool,
      }

    `timeout_s` is the whole wall-clock budget: the process group is hard-killed
    at exactly that point, with no hidden grace added on top. A CLI that owns an
    internal deadline should be given one strictly inside this budget so it can
    self-report first (see CLI_SELF_TIMEOUT_MARGIN_S).

    `stdin_file` streams a readable regular file to child stdin and is mutually
    exclusive with `stdin_data`; it is validated before the child is spawned.

    `log_path` appends collected output incrementally to a caller-supplied file.
    When `err_log_path` is unset (default), stdout and stderr chunks both go to
    `log_path` and a ``[omni-llm-mcp] done ...`` footer is appended — today's
    merged behaviour, byte-identical for the sync lane. When `err_log_path` is
    set, stdout chunks go only to `log_path`, stderr chunks only to
    `err_log_path`, and the done-footer is written to NEITHER (callers that set
    `err_log_path` own their meta, e.g. supervisor.log). `on_spawn`, when
    supplied, is called with the live process immediately after it is created.
    ``watch_stdout=False`` keeps stdout out of the watch pane (stderr and the
    done-footer still reach it): for callers that must judge the output
    before any of it is shown.

    ``pre_spawn`` (default None: no behaviour change) is a zero-argument
    callable run as the very last step before the child is created, after
    every await in this function. A truthy return is a refusal dict: it is
    returned as-is (no ``returncode`` key) and nothing is spawned. A raising
    callable fails closed with a typed ``pre_spawn_error`` refusal.

    A wall timeout keeps what was read: ``stdout`` / ``stderr`` carry every
    byte received before the kill (plus whatever the pipes still held, drained
    for at most ``_POST_KILL_DRAIN_S``), also when the pipes had already hit
    EOF and only the child's exit hung.

    ``stall_marker`` / ``stall_grace_s`` (both or neither; default None keeps
    the result dict and behaviour exactly as without them) arm a stall
    watchdog for children that finish their work and then never exit.
    ``stall_marker`` takes one stdout line as bytes (no newline) and returns
    whether it is the child's final line. After the first match, a run that
    receives no stdout byte for ``stall_grace_s`` seconds (any byte restarts
    the clock) is group-killed and returned with ``stalled`` True; the wall
    timeout still applies, whichever fires first wins. With it, every result
    also carries ``stalled`` (bool), ``marker_seen`` (bool) and ``elapsed_s``
    (float, seconds since spawn at the moment the run ended or was cut).

    MCP return is not delayed by pane close (done sentinel + -c; close-pane is
    scheduled in the background). CLI process-group reap still runs before return.
    """
    if stdin_data is not None and stdin_file is not None:
        raise ValueError("stdin_data and stdin_file are mutually exclusive")
    if stdin_file is not None:
        file_stat = os.stat(stdin_file)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("stdin_file must be a readable regular file")
        with open(stdin_file, "rb"):
            pass

    if (stall_marker is None) != (stall_grace_s is None):
        raise ValueError("stall_marker and stall_grace_s go together")
    if stall_marker is not None:
        stall_grace_s = float(stall_grace_s)
        if not stall_grace_s > 0 or stall_grace_s == float("inf"):
            raise ValueError("stall_grace_s must be a finite number > 0")

    requested_log_path = log_path
    requested_err_log_path = err_log_path
    log_path = None
    done_path = None
    log_f = None
    direct_log_f = None
    err_log_f = None
    pane_id = None
    session = None
    watched = False
    proc = None
    out_b = b""
    err_b = b""
    out_chunks = []
    err_chunks = []
    timed_out = False
    stalled = False
    scan = None
    t_spawn = None
    elapsed_s = None
    work_dir = None
    watch_finish_scheduled = False
    split_err = requested_err_log_path is not None

    # Entire body under try/finally so cancel mid-open still reaps temp/pane state.
    try:
        # The child's final environment is known here, before anything is
        # written: every log sink below masks the credential values in it.
        known_secrets.register_child_env(env)
        if watch_enabled():
            session = resolve_zellij_session()
            tab_id = resolve_zellij_tab_id(session) if session else None
            if not session:
                log_event("watch_pane_skip", reason="no_session", pane=pane_name)
            elif tab_id is None:
                log_event("watch_pane_skip", reason="no_tab", pane=pane_name,
                          session=session)
            else:
                try:
                    work_dir = tempfile.mkdtemp(prefix="omni_watch_")
                    try:
                        os.chmod(work_dir, 0o700)
                    except OSError:
                        pass
                    log_path = os.path.join(work_dir, "stream.log")
                    done_path = os.path.join(work_dir, "done")
                    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                    os.close(fd)
                    with open(log_path, "ab") as f:
                        # argv only (no prompt body); avoid shell-join ambiguity for
                        # display — space-separated is best-effort status text.
                        cmd_s = known_secrets.mask_known(
                            " ".join(str(a) for a in argv))
                        f.write(("[omni-llm-mcp] watch %s @ %s\ncmd: %s\n\n" % (
                            pane_name, time.strftime("%H:%M:%S"), cmd_s)).encode())
                    max_s = float(timeout_s) + WATCH_PANE_MAX_AGE_PAD_S
                    pane_id = await _open_watch_pane(
                        session, tab_id, log_path, done_path, pane_name, max_s)
                    watched = bool(pane_id)
                    if not pane_id:
                        log_event("watch_pane_skip", reason="open_failed",
                                  session=session, tab_id=tab_id, pane=pane_name)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log_event("watch_pane_error", detail=str(e)[:200], pane=pane_name)
                    pane_id = None
                    watched = False

        if log_path:
            log_f = open(log_path, "ab")
        if requested_log_path is not None:
            direct_path = os.fspath(requested_log_path)
            if log_path is not None and os.path.abspath(direct_path) == os.path.abspath(log_path):
                direct_path = None
            if direct_path is not None:
                parent = os.path.dirname(os.path.abspath(direct_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                direct_log_f = open(direct_path, "ab")
        if requested_err_log_path is not None:
            err_path = os.fspath(requested_err_log_path)
            parent = os.path.dirname(os.path.abspath(err_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            err_log_f = open(err_path, "ab")

        kwargs = dict(
            stdin=(asyncio.subprocess.PIPE
                   if stdin_data is not None or stdin_file is not None
                   else asyncio.subprocess.DEVNULL),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        if pre_spawn is not None:
            # Last synchronous step before process creation: nothing below
            # awaits ahead of _spawn, so this verdict cannot go stale.
            try:
                refusal = pre_spawn()
            except Exception as exc:  # fail closed on a broken gate
                return {"ok": False, "reason": "pre_spawn_error",
                        "detail": "pre-spawn check failed: %s" % exc.__class__.__name__,
                        "account_retry_safe": False}
            if refusal:
                return refusal
        # New session => process group for clean kill of descendants (Unix).
        try:
            proc = await _spawn(
                *[str(a) for a in argv], start_new_session=True, **kwargs)
        except TypeError:
            proc = await _spawn(*[str(a) for a in argv], **kwargs)

        t_spawn = time.monotonic()
        if stall_marker is not None:
            scan = _MarkerScan(stall_marker)

        if on_spawn is not None:
            callback_result = on_spawn(proc)
            if hasattr(callback_result, "__await__"):
                await callback_result

        def _write_sinks(sinks, data):
            if not data:
                return
            for output_log in sinks:
                if output_log is not None:
                    try:
                        output_log.write(data)
                        output_log.flush()
                    except Exception:
                        pass

        async def _read(stream, sinks, chunks, line_scan=None):
            # The returned bytes stay raw (callers parse them); what reaches a
            # log sink is masked, with a value split across two reads held back
            # until it can be judged. ``chunks`` belongs to the caller so a
            # cancelled read leaves what it had collected.
            masker = known_secrets.StreamMasker()
            try:
                while True:
                    data = await stream.read(4096)
                    if not data:
                        break
                    chunks.append(data)
                    if line_scan is not None:
                        line_scan.feed(data)
                    _write_sinks(sinks, masker.feed(data))
                if line_scan is not None:
                    line_scan.finish()
            finally:
                _write_sinks(sinks, masker.flush())
            return b"".join(chunks)

        # split_err: stdout -> log_path only; stderr -> err_log_path only.
        # Unset: both streams mirror to the same sinks (byte-identical sync).
        pane_out = log_f if watch_stdout else None
        if split_err:
            out_sinks = (pane_out, direct_log_f)
            err_sinks = (err_log_f,)
        else:
            out_sinks = (pane_out, direct_log_f)
            err_sinks = (log_f, direct_log_f)

        async def _feed_stdin():
            if stdin_data is None and stdin_file is None:
                return
            try:
                if stdin_data is not None:
                    raw = (stdin_data if isinstance(stdin_data, (bytes, bytearray))
                           else str(stdin_data).encode())
                    for offset in range(0, len(raw), _STDIN_CHUNK_SIZE):
                        proc.stdin.write(raw[offset:offset + _STDIN_CHUNK_SIZE])
                        await proc.stdin.drain()
                else:
                    with open(stdin_file, "rb") as f:
                        while True:
                            chunk = f.read(_STDIN_CHUNK_SIZE)
                            if not chunk:
                                break
                            proc.stdin.write(chunk)
                            await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                wait_closed = getattr(proc.stdin, "wait_closed", None)
                if wait_closed is not None:
                    try:
                        await wait_closed()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        async def _drive():
            # Readers must run concurrently with feeding: the wall-clock bound
            # and wait_closed() rely on draining the pipes, or a non-empty buffer
            # can block forever if feed/read are sequenced again.
            stdout, stderr, _ = await asyncio.gather(
                _read(proc.stdout, out_sinks, out_chunks, scan),
                _read(proc.stderr, err_sinks, err_chunks),
                _feed_stdin())
            await proc.wait()
            return stdout, stderr

        async def _stall_watch():
            # Returns once the marker line is out and no stdout byte has come
            # for stall_grace_s; every new byte pushes the deadline back. A
            # provisional match that later bytes undo is waited for again.
            while True:
                await scan.event.wait()
                if not scan.seen:
                    continue
                idle = time.monotonic() - scan.last_byte
                if idle >= stall_grace_s:
                    return
                await asyncio.sleep(stall_grace_s - idle)

        async def _drain_after_kill():
            # The killed child's pipes close, so what it had already written
            # but this side had not yet read is still there; bounded, because
            # a descendant outside the group may hold the pipes open.
            try:
                await asyncio.wait_for(asyncio.gather(
                    _read(proc.stdout, out_sinks, out_chunks, scan),
                    _read(proc.stderr, err_sinks, err_chunks)),
                    timeout=_POST_KILL_DRAIN_S)
            except Exception:
                pass
            return b"".join(out_chunks), b"".join(err_chunks)

        drive_task = asyncio.ensure_future(_drive())
        stall_task = (asyncio.ensure_future(_stall_watch())
                      if scan is not None else None)
        try:
            # timeout_s is the caller's whole budget — no hidden grace on top.
            await asyncio.wait(
                [t for t in (drive_task, stall_task) if t is not None],
                timeout=float(timeout_s),
                return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            await _cancel_tasks(drive_task, stall_task)
            raise
        elapsed_s = round(time.monotonic() - t_spawn, 3)
        if drive_task.done():
            await _cancel_tasks(stall_task)
            out_b, err_b = drive_task.result()
        else:
            if stall_task is not None and stall_task.done():
                stall_task.result()
                stalled = True
            else:
                timed_out = True
            await _cancel_tasks(drive_task, stall_task)
            await _kill_proc_tree(proc)
            out_b, err_b = await _drain_after_kill()
        # A login file the child refreshed holds new tokens now: the failure
        # text built from this result is masked against them too.
        known_secrets.register_child_env(env)

        # Done-footer only for the merged log_path path. Callers that set
        # err_log_path own their meta (supervisor.log), so neither stream file
        # receives the footer.
        if not split_err:
            for output_log in (log_f, direct_log_f):
                if output_log is not None:
                    try:
                        output_log.write(("\n[omni-llm-mcp] done rc=%s timed_out=%s @ %s\n" % (
                            None if proc is None else proc.returncode,
                            timed_out, time.strftime("%H:%M:%S"))).encode())
                        output_log.flush()
                    except Exception:
                        pass

        raw_rc = None if proc is None else proc.returncode
        if timed_out or stalled:
            rc = -1
            sig = None
        else:
            rc = raw_rc
            if (isinstance(rc, int) and not isinstance(rc, bool) and rc < 0):
                sig = -rc
            else:
                sig = None
        result = {
            "returncode": rc,
            "stdout": decode_output(out_b),
            "stderr": decode_output(err_b),
            "timed_out": timed_out,
            "watched": bool(watched and pane_id),
            "signal": sig,
        }
        if scan is not None:
            result.update({"stalled": stalled, "marker_seen": scan.seen,
                           "elapsed_s": elapsed_s})
        return result
    finally:
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        if direct_log_f is not None:
            try:
                direct_log_f.close()
            except Exception:
                pass
        if err_log_f is not None:
            try:
                err_log_f.close()
            except Exception:
                pass
        # 1) Reap CLI process group first (no-op once the child was reaped cleanly).
        await _kill_proc_tree(proc)
        # 2) Signal viewer to exit immediately (zero user-facing hold).
        if done_path:
            try:
                fd = os.open(done_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                os.write(fd, b"1")
                os.close(fd)
            except OSError:
                pass
        # 3) Pane close + temp rmtree in background — must not gate MCP return.
        if session and pane_id:
            _schedule_watch_finish(session, pane_id, work_dir)
            watch_finish_scheduled = True
            work_dir = None
        if work_dir and os.path.isdir(work_dir) and not watch_finish_scheduled:
            shutil.rmtree(work_dir, ignore_errors=True)
