# no-opencc
"""Detached agent jobs backed by a small, file-based state machine.

The caller performs the agent-lane refusal checks before a supervisor is
detached.  A supervisor then owns ``job.lock`` for its whole lifetime and is
the sole writer of a successful or failed ``result.json``.  ``stop`` may write
the distinct ``stopped`` result only after the supervisor has gone away.
"""
import fcntl
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from providers import agent_lane, git_safe, kernel_sandbox, sandbox_canary, codex_accounts, provider_accounts
from providers.attachments import _attach_block, stage_images
from providers.egress_gate import refuse_payload
from providers.config import (
    AGENT_TIMEOUT_S,
    DEFAULT_AGENT_MODEL,
    effort_for_provider,
)
from providers.errors import (
    exception_detail, mask_envelope, redact_secrets, safe_detail)
from providers.logutil import log_event
from providers.profiles import resolve_profile, resolve_codex_home, resolve_account_home
from providers.route import resolve_route
from providers.seat_name import take_effort


ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"
TERMINAL_STATES = frozenset(("done", "failed", "stopped"))
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PS_PATH = ("/bin/ps" if os.path.isfile("/bin/ps") and
            os.access("/bin/ps", os.X_OK) else shutil.which("ps"))
STARTUP_GRACE_S = 15.0
BACKEND_REAP_GRACE_S = 2.0


def _state_dir():
    """Resolve the state root on every call, matching the health ledger."""
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR
    ))


def _jobs_root():
    return os.path.join(_state_dir(), "jobs")


def _now():
    return round(time.time(), 3)


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".omni-job-", suffix=".tmp",
                                     dir=directory)
    open_fd = fd
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            open_fd = None
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if open_fd is not None:
            os.close(open_fd)
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON object required: %s" % path)
    return data


def _read_job(job_dir):
    return _read_json(os.path.join(job_dir, "job.json"))


RAW_LOG = "stream.raw.log"


def _discard_raw_log(job_dir):
    """Delete the raw backend event log; None when gone, else the error text."""
    try:
        os.unlink(os.path.join(job_dir, RAW_LOG))
    except FileNotFoundError:
        pass
    except OSError as exc:
        return "%s: %s" % (type(exc).__name__, exc.strerror or "unlink failed")
    return None


def _write_job(job_dir, data):
    if isinstance(data, dict) and data.get("state") in TERMINAL_STATES:
        # Every path to a terminal state (the runner, stop, the stale/crashed
        # repair) comes through here. The raw log is only the in-run channel
        # between the child and the supervisor; the masked stream.log is what
        # stays (#170). It goes before the state is published, and a delete
        # that fails is recorded in job.json (and logged) so the repair paths
        # can retry it; the refusal to serve it follows the state, not the file.
        error = _discard_raw_log(job_dir)
        if error:
            data = dict(data, raw_log_discard_error=error)
            log_event("raw_log_discard_failed", job_dir=job_dir, detail=error)
        elif "raw_log_discard_error" in data:
            data = {k: v for k, v in data.items() if k != "raw_log_discard_error"}
    _atomic_json(os.path.join(job_dir, "job.json"), data)


def _retry_raw_discard(job_dir, data):
    """Delete a raw log a terminal job still has (an earlier delete failed)."""
    if (isinstance(data, dict) and data.get("state") in TERMINAL_STATES
            and os.path.lexists(os.path.join(job_dir, RAW_LOG))):
        _write_job(job_dir, data)


class RawLogRefused(Exception):
    """``tail(raw=True)`` of a job whose raw log may not be served."""

    def __init__(self, envelope):
        super().__init__(envelope.get("reason"))
        self.envelope = envelope


def raw_log_refusal(job_dir_or_id):
    """The typed refusal for the raw log of a job, or None when it may be served.

    The raw log is served only while the job state is ``starting`` or
    ``running`` and no result exists. A terminal state is refused as not
    retained; a state that is missing, null, unknown or unreadable (any error
    reading the job's metadata) fails closed. It never depends on whether the
    file is still there.
    """
    try:
        job_dir = _resolve_job_dir(job_dir_or_id)
        state = _read_job(job_dir).get("state")
        ended = _read_result(job_dir) is not None
    except Exception:
        state, ended = None, False
        unreadable = True
    else:
        unreadable = not isinstance(state, str)    # a non-string state is no state
    if not unreadable and not ended and state in ("starting", "running"):
        return None
    if not unreadable and (ended or state in TERMINAL_STATES):
        return {
            "ok": False,
            "reason": "raw_log_not_retained",
            "detail": "stream.raw.log is deleted when a job ends; read the "
                      "masked stream.log (omni agent tail <job>) instead",
        }
    return {
        "ok": False,
        "reason": "raw_log_state_unreadable",
        "detail": "the job's state is not known to be running, so "
                  "stream.raw.log is not served; read the masked stream.log "
                  "(omni agent tail <job>) instead",
    }


def _result_path(job_dir):
    return os.path.join(job_dir, "result.json")


def _read_result(job_dir):
    path = _result_path(job_dir)
    if not os.path.isfile(path):
        return None
    return _read_json(path)


def _write_result(job_dir, result):
    path = _result_path(job_dir)
    lock_path = os.path.join(job_dir, "result.lock")
    if isinstance(result, dict):
        # The file egress: failure text is masked once more on the way to
        # disk. In place, so a caller comparing the winner by identity (stop)
        # still sees its own dict.
        masked = mask_envelope(result)
        result.clear()
        result.update(masked)
    with open(lock_path, "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(path):
                return _read_json(path)
            _atomic_json(path, result)
            return result
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _resolve_job_dir(job_dir_or_id):
    raw = os.fspath(job_dir_or_id)
    if not isinstance(raw, str) or not raw:
        raise ValueError("job must be a job directory or job id")
    if (os.path.isabs(raw) or os.sep in raw or
            (os.altsep is not None and os.altsep in raw)):
        job_dir = os.path.abspath(raw)
    else:
        if not _JOB_ID_RE.fullmatch(raw):
            raise ValueError("invalid job id")
        job_dir = os.path.join(_jobs_root(), raw)
    if not os.path.isdir(job_dir):
        raise FileNotFoundError("job directory not found: %s" % job_dir)
    if not os.path.isfile(os.path.join(job_dir, "job.json")):
        raise FileNotFoundError("job.json not found: %s" % job_dir)
    return job_dir


def _pid_identity(pid):
    """Return the process start token used for identity-safe signalling."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if not _PS_PATH:
        return None
    try:
        proc = subprocess.run(
            [_PS_PATH, "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip()
    return token or None


def _process_record(data, kind):
    if kind == "supervisor":
        pid = data.get("pid", data.get("supervisor_pid"))
        pgid = data.get("pgid", data.get("supervisor_pgid"))
        token = (data.get("pid_identity") or data.get("supervisor_identity") or
                 data.get("supervisor_start_identity"))
    else:
        pid = data.get("backend_pid")
        pgid = data.get("backend_pgid")
        token = (data.get("backend_pid_identity") or
                 data.get("backend_identity") or
                 data.get("backend_start_identity"))
    return pid, pgid, token


def _identity_status_record(data, kind):
    """Return match, mismatch, or unknown for one recorded process."""
    pid, _pgid, token = _process_record(data, kind)
    if pid is None or not token:
        return "unknown"
    current = _pid_identity(pid)
    if current is None:
        return "unknown"
    return "match" if current == token else "mismatch"


def _identity_status(data, include_backend=True):
    records = ["supervisor"]
    if include_backend and data.get("backend_pid") is not None:
        records.append("backend")
    statuses = []
    for kind in records:
        statuses.append(_identity_status_record(data, kind))
    if "mismatch" in statuses:
        return "mismatch"
    if "unknown" in statuses:
        return "unknown"
    return "match"


def _identity_matches(data, include_backend=True):
    return _identity_status(data, include_backend) == "match"


def _identity_matches_record(data, kind):
    return _identity_status_record(data, kind) == "match"


def _lock_is_held(job_dir):
    """Probe liveness by trying to acquire the supervisor's exclusive lock."""
    path = os.path.join(job_dir, "job.lock")
    try:
        handle = open(path, "a+b")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in (11, 35):
                return True
            return False
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    finally:
        handle.close()


def _startup_over(job_dir, data, now=None):
    """Whether a free lock proves that supervisor startup has ended."""
    if os.path.isfile(os.path.join(job_dir, "runner.active")):
        return True
    if data.get("state") != "starting":
        return True
    try:
        created_at = float(data.get("created_at"))
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    return current - created_at > STARTUP_GRACE_S


def _state_for_result(job_dir, data, result=None):
    result = _read_result(job_dir) if result is None else result
    if result is None:
        return data
    reason = result.get("reason") if isinstance(result, dict) else None
    state = "stopped" if reason == "stopped" else (
        "done" if result.get("ok") else "failed")
    if data.get("state") != state:
        data = dict(data)
        data["state"] = state
        _write_job(job_dir, data)
    return data


def _record_stop_metadata(job_dir, data, reason):
    """Keep the caller's stop reason when the runner wins result arbitration."""
    if (reason is None or data.get("state") != "stopped" or
            data.get("stop_reason") is not None):
        return data
    data = dict(data)
    data.update({"stopped_by": os.getpid(), "stop_reason": reason})
    _write_job(job_dir, data)
    return data


def _repair_failed(job_dir, data, detail):
    if _lock_is_held(job_dir):
        return data
    _reap_backend_if_live(job_dir, data)
    result = _read_result(job_dir)
    if result is None:
        result = _write_result(job_dir, {
            "ok": False,
            "reason": "runtime_error",
            "detail": safe_detail(detail),
        })
    return _state_for_result(job_dir, data, result)


def _status_view(job_dir, data):
    result = _read_result(job_dir)
    if result is not None:
        return _state_for_result(job_dir, data, result)
    if data.get("state") in TERMINAL_STATES:
        return data
    if _lock_is_held(job_dir):
        return data
    if not _startup_over(job_dir, data):
        return data
    return _repair_failed(job_dir, data, "supervisor lock is free")


def status(job_dir_or_id):
    """Return the current view; repair and write a result for a dead job."""
    job_dir = _resolve_job_dir(job_dir_or_id)
    data = _read_job(job_dir)
    _retry_raw_discard(job_dir, data)
    return _status_view(job_dir, data)


def _preflight(model, workdir, read_only, allow_main, effort=None,
               account_profile="default", ignore_depth=False, codex_home=None, account_home=None):
    """Mirror run_agent's refusal order without spawning a child process."""
    # Only the separately owned foreman start sets ignore_depth.
    host_required = platform.system() == "Darwin" and not ignore_depth
    if not ignore_depth:
        depth = agent_lane.env_depth()
        if depth >= 1:
            return ({"ok": False, "reason": "recursion_guard",
                     "detail": "%s=%d (no nested agents)" % (
                         agent_lane.ENV_DEPTH, depth)}, None, None)
    normalized = os.path.realpath(str(workdir))
    fence = agent_lane._workdir_fence(normalized, allow_main, read_only)
    if fence:
        return fence, normalized, None
    model, effort, split_err = take_effort(model, effort)
    if split_err:
        return split_err, normalized, None
    route = resolve_route(model)
    provider = route["provider"]
    if provider not in agent_lane.AGENT_PROVIDERS:
        return ({"ok": False, "reason": "unsupported_agent",
                 "detail": "agent lane supports codex/claude/grok/pi/cursor; model %r routes to %s"
                 % (str(model), provider)}, normalized, None)
    if provider == "grok" and read_only and not host_required:
        return ({"ok": False, "reason": "unsupported_read_only",
                 "detail": "grok read-only lane not implemented (tools disallow-list "
                           "unverified); use codex, claude, or pi for the read-only lane"},
                normalized, None)
    # Effort before which(): unknown tokens stay unsupported_effort even when
    # the backend CLI is missing (otherwise cli_missing masks the token error).
    _resolved, effort_err = effort_for_provider(provider, effort)
    if effort_err:
        return effort_err, normalized, route
    # Same ordering for profiles: unsupported_profile must not be masked by
    # cli_missing, and jobs.start must refuse before creating a runner.
    _overrides, profile_err = (resolve_codex_home(account_profile, codex_home)
                               if provider == "codex" and codex_home
                               else resolve_account_home(provider, account_profile, account_home)
                               if provider in ("claude", "grok") and account_home
                               else resolve_profile(provider, account_profile))
    if profile_err:
        return profile_err, normalized, route
    if provider in ("claude", "pi", "cursor"):
        executable = agent_lane.resolve_agent_binary(provider)
    else:
        executable = shutil.which(provider)
    if not executable:
        # cursor's binary name is cursor-agent, never the bare provider key
        # "cursor" (that name collides with Grok Build CLI) — match the
        # message agent_lane.py's own cli_missing refusal uses.
        bin_name = "cursor-agent" if provider == "cursor" else provider
        return ({"ok": False, "reason": "cli_missing",
                 "detail": "%s not on PATH" % bin_name}, normalized, None)
    sandbox_refusal = None if host_required else sandbox_canary.gate(read_only, provider)
    if sandbox_refusal:
        return sandbox_refusal, normalized, None
    return None, normalized, route


def _git_pin_query(workdir, *args):
    """Return stripped stdout on success, or None on any failure/indeterminate."""
    try:
        proc = git_safe.run_git(workdir, args)
    except (OSError, subprocess.SubprocessError, git_safe.GitSafeError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _drift_pin(workdir):
    """Record repo_root / branch / head for resume drift checks.

    Indeterminate git state stores nulls for all three fields — resume then
    refuses with resume_drift unless allow_drift.
    """
    nulls = {"repo_root": None, "branch": None, "head": None}
    if not workdir:
        return dict(nulls)
    toplevel = _git_pin_query(workdir, "rev-parse", "--show-toplevel")
    if not toplevel:
        return dict(nulls)
    head = _git_pin_query(workdir, "rev-parse", "HEAD")
    if not head:
        return dict(nulls)
    # symbolic-ref fails (rc 1) on detached HEAD — that is determinate.
    try:
        proc = git_safe.run_git(workdir, ("symbolic-ref", "--short", "-q", "HEAD"))
    except (OSError, subprocess.SubprocessError, git_safe.GitSafeError):
        return dict(nulls)
    if proc.returncode == 0 and (proc.stdout or "").strip():
        branch = (proc.stdout or "").strip()
    elif proc.returncode == 1:
        branch = "detached:%s" % head
    else:
        return dict(nulls)
    return {
        "repo_root": os.path.realpath(toplevel),
        "branch": branch,
        "head": head,
    }


def _pin_all_null(pin):
    """True when repo_root/branch/head are all null (non-git / indeterminate)."""
    pin = pin or {}
    return all(pin.get(k) is None for k in ("repo_root", "branch", "head"))


def _pin_differs(stored, current):
    """True when pins differ, including all-null vs all-null (unverified).

    All-null workdirs have no git identity to verify — treat as drift so
    resume refuses unless allow_drift. Transient null-vs-real already differs.
    """
    if _pin_all_null(stored) and _pin_all_null(current):
        return True
    for key in ("repo_root", "branch", "head"):
        if (stored or {}).get(key) != (current or {}).get(key):
            return True
    return False


def _workdir_has_other_live(workdir, exclude_job_dir=None):
    """True if any other non-terminal job shares workdir's realpath."""
    if not workdir:
        return False
    target = os.path.realpath(str(workdir))
    exclude = (os.path.abspath(exclude_job_dir)
               if exclude_job_dir is not None else None)
    root = _jobs_root()
    if not os.path.isdir(root):
        return False
    for name in os.listdir(root):
        if not _JOB_ID_RE.fullmatch(name):
            continue
        job_dir = os.path.join(root, name)
        if exclude is not None and os.path.abspath(job_dir) == exclude:
            continue
        if not os.path.isfile(os.path.join(job_dir, "job.json")):
            continue
        try:
            view = status(job_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if view.get("state") in TERMINAL_STATES:
            continue
        other_wd = view.get("workdir")
        if other_wd and os.path.realpath(str(other_wd)) == target:
            return True
    return False


def _resume_refused(reason, detail):
    """Typed resume refusal: no job is created."""
    return {"refused": {"ok": False, "reason": reason, "detail": detail}}


def _new_job(task, model, workdir, timeout_s, account_profile, read_only):
    job_id = uuid.uuid4().hex
    root = _jobs_root()
    os.makedirs(root, mode=0o700, exist_ok=True)
    job_dir = os.path.join(root, job_id)
    os.mkdir(job_dir, 0o700)
    for name, mode in (("job.lock", "ab"), ("supervisor.log", "ab")):
        with open(os.path.join(job_dir, name), mode):
            pass
    task_text = str(task)
    with open(os.path.join(job_dir, "prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(task_text)
    try:
        timeout_value = int(timeout_s)
    except (TypeError, ValueError):
        timeout_value = AGENT_TIMEOUT_S
    data = {
        "job_id": job_id,
        "state": "starting",
        "task": task_text,
        "model": str(model),
        "workdir": workdir,
        "timeout_s": timeout_value,
        "account_profile": str(account_profile),
        "read_only": bool(read_only),
        "created_at": _now(),
        "pid": None,
        "pgid": None,
        "pid_identity": None,
        "supervisor_pid": None,
        "supervisor_pgid": None,
        "supervisor_identity": None,
        "supervisor_start_identity": None,
        "backend_pid": None,
        "backend_pgid": None,
        "backend_pid_identity": None,
        "backend_identity": None,
        "backend_start_identity": None,
        "stop_reason": None,
        "stopped_by": None,
        # Resume-lane fields (session_id / drift pin filled by start()).
        "session_id": None,
        "resume_session": None,
        "resumed_from": None,
        "repo_root": None,
        "branch": None,
        "head": None,
        # Backend name for stream_contract.normalize_lines (set in start()).
        "provider": None,
        # Staged image copy paths (issue #18 P4); empty unless start(images=...).
        "images": [],
        # Reasoning effort; None = backend default. Resume inherits this field.
        "effort": None,
    }
    _write_job(job_dir, data)
    return job_id, job_dir, data


def _set_refusal(job_dir, data, result):
    result = dict(result) if isinstance(result, dict) else {
        "ok": False, "reason": "runtime_error", "detail": safe_detail(result),
    }
    result.setdefault("fence", "none")
    result.setdefault("fence_detail", dict(kernel_sandbox.FENCE_NONE_DETAIL))
    _write_result(job_dir, result)
    data = dict(data)
    data["state"] = "failed"
    _write_job(job_dir, data)


def _record_supervisor(data, proc):
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = pid
    token = _pid_identity(pid)
    data = dict(data)
    data.update({
        "pid": pid,
        "pgid": pgid,
        "pid_identity": token,
        "supervisor_pid": pid,
        "supervisor_pgid": pgid,
        "supervisor_identity": token,
        "supervisor_start_identity": token,
        "state": "running",
    })
    return data


def _record_backend(job_dir, proc):
    data = _read_job(job_dir)
    pid = getattr(proc, "pid", None)
    try:
        pgid = os.getpgid(pid) if pid is not None else None
    except OSError:
        pgid = pid
    token = _pid_identity(pid)
    data.update({
        "backend_pid": pid,
        "backend_pgid": pgid,
        "backend_pid_identity": token,
        "backend_identity": token,
        "backend_start_identity": token,
    })
    _write_job(job_dir, data)


def _reap_process(proc):
    try:
        proc.wait()
    except Exception:
        pass


def _wait_for_supervisor_lock(job_dir, proc, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        active = os.path.join(job_dir, "runner.active")
        if _lock_is_held(job_dir) and os.path.isfile(active):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.01)
    return (_lock_is_held(job_dir) and
            os.path.isfile(os.path.join(job_dir, "runner.active")))


def start(task, model=DEFAULT_AGENT_MODEL, workdir=None, timeout_s=AGENT_TIMEOUT_S,
          account_profile="default", read_only=False, allow_main=False,
          resume_session=None, resumed_from=None, images=None, effort=None,
          fence_requested=None, expect_writes=False, foreman=False,
          permission_mode=None, codex_home=None, codex_auth_isolated=None,
          account_home=None, account_auth_isolated=None, account_home_explicit=None):
    """Create a job and detach its supervisor after caller-side fences pass.

    Optional ``resume_session`` / ``resumed_from`` are set by ``resume()`` for
    child jobs; callers of the public start path leave them None.

    Optional ``images``: list of source paths to stage as validated copies under
    the workdir's ``.omni-attach/<job_id>/``. Resume does NOT restage parent
    images — a child job only gets a non-empty list when the resume caller
    passes new ones (resume() itself never passes images).
    """
    # Fail-closed before prompt.txt / job.json persist the task, and before
    # supervisor spawn. Gate-off (enabled() is False) is a no-op.
    denied = refuse_payload(task)
    if denied:
        return {"refused": denied}

    # Split `@<token>` once, before anything persists: job.json must carry the
    # bare model on every path, refusals included. A split error is left for
    # _preflight to surface so the refusal order stays identical.
    bare, merged, split_err = take_effort(model, effort)
    if not split_err:
        model, effort = bare, merged
    perm_err = None
    early = resolve_route(model)
    if early.get("provider") in agent_lane.AGENT_PROVIDERS:
        perm_err = agent_lane.permission_mode_refusal(
            early["provider"], permission_mode, foreman=bool(foreman),
            read_only=bool(read_only))
    # The keyword is passed only for a foreman start: existing callers and
    # test doubles of _preflight keep their pre-#117 signature.
    preflight_kwargs = {"ignore_depth": True} if foreman else {}
    if codex_home is not None:
        preflight_kwargs["codex_home"] = codex_home
    if account_home is not None:
        preflight_kwargs["account_home"] = account_home
    refusal, normalized, route = _preflight(
        model, workdir, read_only, allow_main, effort=effort,
        account_profile=account_profile, **preflight_kwargs)
    if perm_err is not None:
        refusal = perm_err
        if route is None:
            route = early
    if normalized is None:
        normalized = os.path.realpath(str(workdir))
    selection = None
    selected = None
    if refusal is None and early.get("provider") == "codex":
        selection = codex_accounts.AccountSelection(
            account_profile, model, timeout_s, pinned_home=codex_home,
            resume=bool(resume_session))
        if codex_home and not codex_accounts.home_matches(account_profile, codex_home):
            refusal = {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
        else:
            selected = selection.next()
            if selected is None:
                refusal = selection.exhausted()
            else:
                account_profile = selected["account_profile"]
    if refusal is None and early.get("provider") in ("claude", "grok"):
        provider = early["provider"]
        selection = provider_accounts.AccountSelection(
            account_profile, model, timeout_s, pinned_home=account_home,
            resume=bool(resume_session), provider=provider)
        if account_home and not provider_accounts.home_matches(provider, account_profile, account_home):
            refusal = {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
        else:
            selected = selection.next()
            if selected is None:
                refusal = selection.exhausted()
            else:
                account_profile = selected["account_profile"]
                provenance = (selected["account_home_explicit"]
                              if account_home_explicit is None else account_home_explicit)
                refusal = provider_accounts.home_provenance_error(
                    provider, account_profile, selected["account_home"], provenance)
                if refusal:
                    selected = None
    job_id, job_dir, data = _new_job(
        task, model, normalized, timeout_s, account_profile, read_only)

    # Drift pin: realpath repo root, branch (or detached:<sha>), head.
    data.update(_drift_pin(normalized))
    data["resume_session"] = (
        str(resume_session) if resume_session is not None else None)
    data["resumed_from"] = (
        str(resumed_from) if resumed_from is not None else None)
    # Pre-assign session_id for claude/grok/pi NEW runs; codex captures later.
    # On resume: grok gets a fresh child id (fork); claude/pi child stays null
    # (chained capture is P3); codex stays null until thread.started.
    provider = route["provider"] if route else None
    data["provider"] = provider
    if selected is not None and provider == "codex":
        data["codex_home"] = selected["codex_home"]
        data["codex_managed"] = selection.managed
        if codex_auth_isolated is None:
            codex_auth_isolated = provider_accounts.ring_pinned(
                "codex", account_profile, bool(resume_session))
        data["codex_auth_isolated"] = selection.managed or bool(codex_auth_isolated)
        data["codex_candidates"] = (list(selection.names[selection._index:])
                                     if selection.managed else None)
        data["codex_homes"] = dict(selection.homes)
        data["codex_tried"] = list(selection.tried)
        data["codex_deadline"] = time.time() + selection.remaining()
        data["codex_session_offset"] = 0
    if selected is not None and provider in ("claude", "grok"):
        data["account_home"] = selected["account_home"]
        if provider == "claude":
            data["account_kind"] = selected.get("account_kind", "subscription")
        data["account_home_explicit"] = (
            selected["account_home_explicit"] if account_home_explicit is None
            else bool(account_home_explicit))
        data["account_managed"] = selection.managed
        if account_auth_isolated is None:
            account_auth_isolated = provider_accounts.ring_pinned(
                provider, account_profile, bool(resume_session))
        data["account_auth_isolated"] = selection.managed or bool(account_auth_isolated)
        data["account_candidates"] = (list(selection.names[selection._index:]) if selection.managed else None)
        data["account_homes"] = dict(selection.homes)
        data["account_tried"] = list(selection.tried)
        data["account_deadline"] = time.time() + selection.remaining()
        data["account_session_offset"] = 0
    if foreman:
        fence_requested = False
    host_required = platform.system() == "Darwin" and not foreman
    requested = (
        kernel_sandbox.agent_fence_switch_on()
        if fence_requested is None else bool(fence_requested)
    )
    if host_required:
        requested = True
    data["fence_requested"] = requested
    data["expect_writes"] = bool(expect_writes)
    data["foreman"] = bool(foreman)
    data["allow_main"] = bool(allow_main)
    if permission_mode is not None and str(permission_mode).strip():
        data["permission_mode"] = str(permission_mode).strip()
    else:
        data["permission_mode"] = None
    # PLANNED fence, not an observed one. It has to track run_agent's own
    # choice or job.json lies before the run even starts. All ordinary
    # macOS workers require the host fence, including read-only workers.
    # It is NOT re-stamped on finish: nothing writes the executed fence back
    # into job.json, so a run whose wrap was refused still reads `seatbelt-fs`
    # here. The EXECUTED fence lives in result.json, which is what an audit
    # must read. Stated rather than fixed: the re-stamp is a job-lifecycle
    # change, and the planned/observed split predates this issue.
    kernel_lane = host_required or provider in ("claude", "pi", "cursor") or (
        provider == "codex" and not read_only
    )
    if kernel_lane and requested:
        data["fence"] = "seatbelt-fs"
    elif provider in ("codex", "grok"):
        data["fence"] = "vendor"
    else:
        data["fence"] = "none"
    if resume_session is not None:
        if provider == "grok":
            data["session_id"] = str(uuid.uuid4())
        else:
            data["session_id"] = None
    elif provider in ("claude", "grok", "pi"):
        data["session_id"] = str(uuid.uuid4())
    else:
        # codex and cursor capture their own id from the event stream.
        data["session_id"] = None
    data["images"] = []
    if effort is not None and str(effort).strip():
        data["effort"] = str(effort).strip()
    _write_job(job_dir, data)

    if refusal:
        _set_refusal(job_dir, data, refusal)
        return {"job_id": job_id, "job_dir": job_dir, "refused": refusal}

    if data.get("effort") is not None and route is not None:
        _resolved, effort_err = effort_for_provider(route["provider"], data["effort"])
        if effort_err:
            _set_refusal(job_dir, data, effort_err)
            return {"job_id": job_id, "job_dir": job_dir, "refused": effort_err}
        data["effort"] = _resolved
        _write_job(job_dir, data)

    # Stage images only after preflight passes and the job dir exists.
    # Refusal: typed envelope as result.json, state failed, no spawn.
    if images:
        staged, img_refusal = stage_images(normalized, job_id, images)
        if img_refusal is not None:
            _set_refusal(job_dir, data, img_refusal)
            return {"job_id": job_id, "job_dir": job_dir, "refused": img_refusal}
        data["images"] = list(staged)
        _write_job(job_dir, data)
        # task_file path: keep prompt.txt the single source for claude (and
        # any backend that feeds stdin from the file). Append the attach block
        # here so run_agent does not mutate the file again.
        if staged:
            prompt_path = os.path.join(job_dir, "prompt.txt")
            with open(prompt_path, "a", encoding="utf-8") as handle:
                handle.write(_attach_block(staged))

    if selected is not None and selection.managed and selection.remaining() <= 0:
        refusal = {"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"}
        _set_refusal(job_dir, data, refusal)
        return {"job_id": job_id, "job_dir": job_dir, "refused": refusal}
    supervisor_log = open(os.path.join(job_dir, "supervisor.log"), "ab")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "providers.job_runner", job_dir],
            cwd=_REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=supervisor_log,
            stderr=supervisor_log,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        supervisor_log.close()
        _set_refusal(job_dir, data, {
            "ok": False,
            "reason": "runtime_error",
            "detail": exception_detail(exc),
        })
        refusal = {
            "ok": False,
            "reason": "runtime_error",
            "detail": exception_detail(exc),
        }
        return {"job_id": job_id, "job_dir": job_dir, "refused": refusal}
    finally:
        if not supervisor_log.closed:
            supervisor_log.close()

    data = _record_supervisor(data, proc)
    _write_job(job_dir, data)
    with open(os.path.join(job_dir, "runner.ready"), "w", encoding="utf-8") as handle:
        handle.write("1\n")
    threading.Thread(target=_reap_process, args=(proc,), daemon=True).start()
    _wait_for_supervisor_lock(job_dir, proc)
    return {"job_id": job_id, "job_dir": job_dir}


def resume(source, prompt, allow_drift=False):
    """Start a child job that continues ``source``'s backend session.

    Refusal checks (each returns ``{"refused": envelope}`` and creates no job),
    in order: resume_busy, resume_not_terminal, resume_unsupported,
    resume_drift, resume_workdir_busy. On pass, fences re-run via start().
    """
    try:
        source_dir = _resolve_job_dir(source)
    except (OSError, ValueError, FileNotFoundError) as exc:
        return _resume_refused("resume_unsupported", redact_secrets(str(exc))[:300])

    lock_path = os.path.join(source_dir, "resume.lock")
    try:
        lock_handle = open(lock_path, "a+b")
    except OSError as exc:
        return _resume_refused("resume_busy", redact_secrets(str(exc))[:300])

    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if (isinstance(exc, BlockingIOError) or
                    getattr(exc, "errno", None) in (11, 35)):
                return _resume_refused(
                    "resume_busy", "source resume.lock is held")
            return _resume_refused("resume_busy", redact_secrets(str(exc))[:300])
        try:
            data = status(source_dir)
            if data.get("state") not in TERMINAL_STATES:
                return _resume_refused(
                    "resume_not_terminal",
                    "source state is %r (need done/failed/stopped)"
                    % data.get("state"))
            session_id = data.get("session_id")
            if not session_id:
                return _resume_refused(
                    "resume_unsupported",
                    "source session_id is null")
            if not allow_drift:
                stored = {k: data.get(k) for k in ("repo_root", "branch", "head")}
                current = _drift_pin(data.get("workdir") or "")
                if _pin_all_null(stored) and _pin_all_null(current):
                    return _resume_refused(
                        "resume_drift",
                        "workdir has no git identity to verify")
                if _pin_differs(stored, current):
                    return _resume_refused(
                        "resume_drift",
                        "workdir git pin differs from source job")
            if _workdir_has_other_live(data.get("workdir"),
                                       exclude_job_dir=source_dir):
                return _resume_refused(
                    "resume_workdir_busy",
                    "another non-terminal job shares this workdir")

            if data.get("provider") == "codex":
                if not codex_accounts.home_matches(
                        data.get("account_profile") or "default", data.get("codex_home")):
                    return _resume_refused(
                        "account_home_drift", "original account home is missing or changed")
            if data.get("provider") in ("claude", "grok"):
                if not provider_accounts.home_matches(
                        data["provider"], data.get("account_profile") or "default", data.get("account_home")):
                    return _resume_refused(
                        "account_home_drift", "original account home is missing or changed")
            if data.get("provider") == "claude" and (
                    provider_accounts.account_kind("claude", data.get("account_profile") or "default")
                    != data.get("account_kind", "subscription")):
                return _resume_refused(
                    "account_home_drift", "original account kind changed")
            # Child creation while holding resume.lock (single-flight).
            return start(
                task=prompt,
                model=data.get("model") or DEFAULT_AGENT_MODEL,
                workdir=data.get("workdir"),
                timeout_s=data.get("timeout_s", AGENT_TIMEOUT_S),
                account_profile=data.get("account_profile") or "default",
                read_only=bool(data.get("read_only", False)),
                resume_session=session_id,
                resumed_from=data.get("job_id"),
                effort=data.get("effort"),
                **({"codex_home": data["codex_home"],
                    "codex_auth_isolated": bool(data.get("codex_auth_isolated",
                                                        data.get("codex_managed", False)))}
                   if data.get("provider") == "codex" else
                   {"account_home": data["account_home"],
                    # Legacy records exported their concrete pin; never infer
                    # different provenance from the resume caller's environment.
                    "account_home_explicit": data.get("account_home_explicit", True),
                    "account_auth_isolated": bool(data.get("account_auth_isolated", False))}
                   if data.get("provider") in ("claude", "grok") else {}),
            )
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        lock_handle.close()


def _send_group(data, kind, sig):
    _pid, pgid, _token = _process_record(data, kind)
    try:
        pgid = int(pgid)
    except (TypeError, ValueError):
        return False
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _wait_for_unlock(job_dir, timeout_s):
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() < deadline:
        if not _lock_is_held(job_dir):
            return True
        time.sleep(0.05)
    return not _lock_is_held(job_dir)


def _signal_backend_if_new(job_dir, sig, sent):
    data = _read_job(job_dir)
    pid = data.get("backend_pid")
    key = (pid, sig)
    if pid is None or key in sent:
        return False
    if _identity_matches_record(data, "backend"):
        sent.add(key)
        _send_group(data, "backend", sig)
        return True
    return False


def _reap_backend_if_live(job_dir, data=None, sent=None):
    """Reap an identity-matching backend after its supervisor is gone."""
    if _lock_is_held(job_dir):
        return
    sent = set() if sent is None else sent
    if data is None:
        data = _read_job(job_dir)
    if not _signal_backend_if_new(job_dir, signal.SIGTERM, sent):
        return
    deadline = time.monotonic() + BACKEND_REAP_GRACE_S
    while time.monotonic() < deadline:
        current = _read_job(job_dir)
        if _identity_status_record(current, "backend") != "match":
            return
        time.sleep(0.05)
    _signal_backend_if_new(job_dir, signal.SIGKILL, sent)


def _unverifiable_stop_view(job_dir, detail, grace_s):
    """Wait for a possibly live supervisor without signalling its pid/group."""
    _wait_for_unlock(job_dir, grace_s)
    data = _read_job(job_dir)
    if _read_result(job_dir) is not None:
        return _state_for_result(job_dir, data)
    if not _lock_is_held(job_dir) and _startup_over(job_dir, data):
        return _repair_failed(job_dir, data, detail)
    view = dict(data)
    view["detail"] = detail
    return view


def stop(job_dir_or_id, grace_s=10, reason=None):
    """Stop a live job, preserving any result already written by its runner."""
    job_dir = _resolve_job_dir(job_dir_or_id)
    data = _read_job(job_dir)
    result = _read_result(job_dir)
    if result is not None:
        if not _lock_is_held(job_dir):
            _reap_backend_if_live(job_dir, data)
        view = _state_for_result(job_dir, data, result)
        return _record_stop_metadata(job_dir, view, reason)
    if data.get("state") in TERMINAL_STATES:
        if not _lock_is_held(job_dir):
            _reap_backend_if_live(job_dir, data)
        return data
    if not _lock_is_held(job_dir):
        if not _startup_over(job_dir, data):
            return data
        return _repair_failed(job_dir, data, "supervisor lock is free")
    try:
        grace = max(0.0, float(grace_s))
    except (TypeError, ValueError):
        grace = 10.0

    supervisor_identity = _identity_status_record(data, "supervisor")
    if supervisor_identity != "match":
        return _unverifiable_stop_view(
            job_dir,
            "supervisor identity unverifiable (%s); no process group signalled"
            % supervisor_identity,
            grace,
        )

    data = _read_job(job_dir)
    # Re-check the supervisor identity immediately before signalling.  A pid
    # can be recycled between the first probe and this point.
    supervisor_identity = _identity_status_record(data, "supervisor")
    if supervisor_identity != "match":
        return _unverifiable_stop_view(
            job_dir,
            "supervisor identity changed before stop (%s); no process group "
            "signalled" % supervisor_identity,
            grace,
        )
    _send_group(data, "supervisor", signal.SIGTERM)

    backend_sent = set()
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and _lock_is_held(job_dir):
        _signal_backend_if_new(job_dir, signal.SIGTERM, backend_sent)
        if _read_result(job_dir) is not None:
            view = _state_for_result(job_dir, _read_job(job_dir))
            return _record_stop_metadata(job_dir, view, reason)
        time.sleep(0.05)
    if _read_result(job_dir) is not None:
        view = _state_for_result(job_dir, _read_job(job_dir))
        return _record_stop_metadata(job_dir, view, reason)

    if _lock_is_held(job_dir):
        data = _read_job(job_dir)
        supervisor_identity = _identity_status_record(data, "supervisor")
        if supervisor_identity == "match":
            _send_group(data, "supervisor", signal.SIGKILL)
            _signal_backend_if_new(job_dir, signal.SIGKILL, backend_sent)
        else:
            return _unverifiable_stop_view(
                job_dir,
                "supervisor identity changed before kill (%s); no SIGKILL "
                "signalled" % supervisor_identity,
                0,
            )
        if not _wait_for_unlock(job_dir, 5):
            return _read_job(job_dir)
    result = _read_result(job_dir)
    if result is not None:
        view = _state_for_result(job_dir, _read_job(job_dir), result)
        return _record_stop_metadata(job_dir, view, reason)

    _reap_backend_if_live(job_dir, _read_job(job_dir), backend_sent)
    data = _read_job(job_dir)
    if not _startup_over(job_dir, data):
        return data

    detail = str(reason) if reason is not None else "stopped"
    result = {
        "ok": False,
        "reason": "stopped",
        "detail": safe_detail(detail),
    }
    winner = _write_result(job_dir, result)
    if winner is not result:
        view = _state_for_result(job_dir, _read_job(job_dir), winner)
        return _record_stop_metadata(job_dir, view, reason)
    data = _read_job(job_dir)
    data.update({
        "state": "stopped",
        "stopped_by": os.getpid(),
        "stop_reason": reason,
    })
    _write_job(job_dir, data)
    return data


def result(job_dir_or_id, wait_s=0):
    """Read a result, optionally waiting for the job to become terminal."""
    job_dir = _resolve_job_dir(job_dir_or_id)
    try:
        wait = max(0.0, float(wait_s))
    except (TypeError, ValueError):
        wait = 0.0
    deadline = time.monotonic() + wait
    while True:
        value = _read_result(job_dir)
        if value is not None:
            return value
        view = status(job_dir)
        if view.get("state") in TERMINAL_STATES or time.monotonic() >= deadline:
            return {
                "ok": False,
                "reason": "not_finished",
                "state": view.get("state"),
            }
        time.sleep(0.1)


def tail(job_dir_or_id, lines=10, raw=False, err=False):
    """Return the last ``lines`` of a job stream file.

    Default: ``stream.log``. ``raw=True`` → ``stream.raw.log``, only while the
    job runs: otherwise ``RawLogRefused`` is raised (see ``raw_log_refusal``).
    ``err=True`` → ``stream.err.log`` (takes precedence over ``raw``).
    CLI exposes ``--raw`` / ``--err`` as mutually exclusive.
    """
    job_dir = _resolve_job_dir(job_dir_or_id)
    if raw and not err:
        refusal = raw_log_refusal(job_dir)
        if refusal:
            raise RawLogRefused(refusal)
    try:
        count = max(0, int(lines))
    except (TypeError, ValueError):
        count = 10
    if err:
        name = "stream.err.log"
    elif raw:
        name = "stream.raw.log"
    else:
        name = "stream.log"
    path = os.path.join(job_dir, name)
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            values = handle.readlines()
    except FileNotFoundError:
        return ""
    return "".join(values[-count:]) if count else ""


def sweep_orphans():
    """Repair every non-terminal job whose supervisor lock is free.

    Skips jobs whose lock is held or whose startup grace has not ended.
    Idempotent: a second call finds terminal state and returns []. Logs
    ``job_repaired`` with the job id for each repair.
    """
    repaired = []
    root = _jobs_root()
    if not os.path.isdir(root):
        return repaired
    for name in sorted(os.listdir(root)):
        if not _JOB_ID_RE.fullmatch(name):
            continue
        job_dir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(job_dir, "job.json")):
            continue
        try:
            data = _read_job(job_dir)
        except (OSError, ValueError):
            log_event("job_sweep_skipped", job_dir=job_dir)
            continue
        if data.get("state") in TERMINAL_STATES:
            _retry_raw_discard(job_dir, data)
            continue
        if _lock_is_held(job_dir):
            continue
        if not _startup_over(job_dir, data):
            continue
        job_id = data.get("job_id") or name
        updated = _repair_failed(job_dir, data, "supervisor lock is free")
        if updated.get("state") in TERMINAL_STATES:
            repaired.append(job_id)
            log_event("job_repaired", job_id=job_id)
    return repaired


def list_jobs():
    """Sweep orphans once, then list every job directory; no GC.

    Returns ``{"jobs": [...], "repaired": [...]}`` from that single pass.
    """
    repaired = sweep_orphans()
    root = _jobs_root()
    if not os.path.isdir(root):
        return {"jobs": [], "repaired": repaired}
    out = []
    for name in sorted(os.listdir(root)):
        if not _JOB_ID_RE.fullmatch(name):
            continue
        job_dir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(job_dir, "job.json")):
            continue
        try:
            out.append(status(job_dir))
        except (OSError, ValueError):
            log_event("job_sweep_skipped", job_dir=job_dir)
            continue
    return {"jobs": out, "repaired": repaired}
