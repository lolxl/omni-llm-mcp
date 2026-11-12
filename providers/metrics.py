# no-opencc
"""Content-free request metrics (JSONL) for consult calls.

One line per tools/call and per CLI omni gen/big/img consult. Never stores
prompt or output content — only operational sizes, seats, and usage envelopes.
Write failures must not alter the model result: the envelope gains
``metrics_error`` and processing continues.
"""
import contextvars
import json
import os
import stat
import sys
import time

from providers.errors import redact_secrets
from providers.logutil import log_event

ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
ENV_METRICS = "OMNI_LLM_METRICS"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"
METRICS_BASENAME = "metrics.jsonl"
ROTATE_BYTES = 5 * 1024 * 1024  # 5 MiB
DIR_MODE = 0o700
FILE_MODE = 0o600

# Caller-controlled model/profile strings written to metrics.jsonl.
# Cap length so a hostile/oversized seat never bloats the log; truncated
# values carry an explicit marker so they are never mistaken for real ones.
FIELD_MAX_LEN = 256
FIELD_TRUNC_MARKER = "…[truncated]"

# Per-call attempt trace. None = no open trace (outside call_tool metrics path).
_attempt_trace = contextvars.ContextVar("omni_metrics_attempt_trace", default=None)
_seat_calls = contextvars.ContextVar("omni_metrics_seat_calls", default=None)

# Warn once per DIRECTORY when a pre-existing state dir is looser than 0700.
# Keyed by path: _state_dir() is re-resolved every call, so one process can
# legitimately meet several roots and each deserves its own diagnostic.
_loose_dirs_warned = set()


def _state_dir():
    # Resolve every call: tests and long-lived processes may change the root.
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR
    ))


def metrics_path():
    return os.path.join(_state_dir(), METRICS_BASENAME)


def metrics_enabled():
    """Writes are on unless OMNI_LLM_METRICS is exactly \"0\"."""
    return os.environ.get(ENV_METRICS, "1") != "0"


_entry_override = None


def set_entry(kind):
    """Pin the entry kind for this process (``None`` restores detection).

    `omni mcp` runs the MCP server inside the CLI executable, so the
    ``__main__`` / argv[0] heuristic below would log it as \"cli\".
    """
    global _entry_override
    _entry_override = kind


def detect_entry():
    """Return \"cli\" when invoked via omni_cli, else \"mcp\"."""
    if _entry_override:
        return _entry_override
    main = sys.modules.get("__main__")
    if main is not None:
        main_file = getattr(main, "__file__", None) or ""
        base = os.path.basename(main_file)
        if base in ("omni_cli.py", "omni") or base.startswith("omni_cli"):
            return "cli"
    if sys.argv:
        argv0 = os.path.basename(sys.argv[0] or "")
        if argv0 in ("omni", "omni_cli.py") or argv0.startswith("omni_cli"):
            return "cli"
    return "mcp"


def begin_trace():
    """Open a per-call attempt/seat trace; return tokens for reset."""
    t_attempts = _attempt_trace.set([])
    t_seats = _seat_calls.set([])
    return (t_attempts, t_seats)


def end_trace(tokens):
    if not tokens:
        return
    t_attempts, t_seats = tokens
    _attempt_trace.reset(t_attempts)
    _seat_calls.reset(t_seats)


# Allowlist for usage projected onto disk. Envelope keeps full usage (incl. raw).
_USAGE_PERSIST_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "source",
)


def project_usage(usage):
    """Project envelope usage to the metrics-disk allowlist (never includes raw)."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return None
    return {k: usage.get(k) for k in _USAGE_PERSIST_KEYS}


def bound_field(value, max_len=FIELD_MAX_LEN):
    """Cap a caller-controlled string for metrics.jsonl.

    Values at or under ``max_len`` are unchanged. Longer values are truncated
    and end with ``FIELD_TRUNC_MARKER`` so a truncated write is never mistaken
    for a real model/profile name. Total length is always <= max_len.
    """
    if value is None:
        return None
    s = str(value)
    if len(s) <= max_len:
        return s
    marker = FIELD_TRUNC_MARKER
    # Keep marker; shrink head so total length == max_len.
    head = max_len - len(marker)
    if head < 0:
        return marker[:max_len]
    return s[:head] + marker


def note_provider_attempt(model, provider, res):
    """Record one real provider invocation (including rate-limit retries).

    Skipped_cooldown seats never reach this (they never call the provider).
    """
    lst = _attempt_trace.get()
    if lst is None:
        return
    entry = {
        "model": bound_field(model),
        "provider": str(provider),
        "ok": bool(res.get("ok")),
    }
    if not res.get("ok") and res.get("reason") is not None:
        entry["reason"] = res.get("reason")
    # Per-attempt usage only when already present on the provider result.
    # Project to allowlist so raw never reaches the metrics file.
    if "usage" in res and res.get("usage") is not None:
        entry["usage"] = project_usage(res.get("usage"))
    lst.append(entry)


def note_seat_call(model):
    """Record one dispatch_single seat entry (rate-limit retries share one seat)."""
    lst = _seat_calls.get()
    if lst is None:
        return
    lst.append(bound_field(model))


def current_attempts():
    return list(_attempt_trace.get() or [])


def current_fallback_count():
    seats = _seat_calls.get() or []
    return max(0, len(seats) - 1)


def utf8_len(text):
    if text is None:
        return 0
    if isinstance(text, bytes):
        return len(text)
    return len(str(text).encode("utf-8"))


def result_bytes_of(envelope):
    """UTF-8 length of the envelope reply text (content-free size only)."""
    if not isinstance(envelope, dict):
        return 0
    text = envelope.get("text")
    if text is None:
        return 0
    return utf8_len(text)


def _ensure_state_dir(directory):
    """Create the metrics state dir with 0700 when we create it.

    When the directory already exists, leave its mode alone. If it is looser
    than 0700, record that once via the LOUD local-error channel — never chmod
    a directory this module does not own.
    """
    global _loose_dirs_warned
    existed = os.path.isdir(directory)
    os.makedirs(directory, mode=DIR_MODE, exist_ok=True)
    if not existed:
        # Metrics created this directory (or won the create race): set 0700.
        # makedirs mode is umask-masked, so an explicit chmod on *our* create
        # is required to honor DIR_MODE.
        try:
            os.chmod(directory, DIR_MODE)
        except OSError:
            pass
        return
    # Pre-existing: do not touch mode. Warn once if looser than 0700.
    try:
        mode = stat.S_IMODE(os.stat(directory).st_mode)
    except OSError:
        return
    if (mode & 0o077) == 0:
        return
    if directory in _loose_dirs_warned:
        return
    _loose_dirs_warned.add(directory)
    log_event(
        "metrics_state_dir_loose_mode",
        path=directory,
        mode=oct(mode),
        expected=oct(DIR_MODE),
    )


def _rotate_if_needed(path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size <= ROTATE_BYTES:
        return
    # TOCTOU: a concurrent racer may have already rotated; never surface that as
    # metrics_error, and never clobber a fresh (small) file over .1.
    try:
        size = os.path.getsize(path)
        if size <= ROTATE_BYTES:
            return
        # Single-generation rotate: overwrite any prior .1.
        os.replace(path, path + ".1")
    except OSError:
        return


def _bound_seat_fields(record):
    """Re-apply the field cap at the write boundary.

    build_record() already bounds these, but append_record is public: the
    content-free invariant is a property of what reaches disk, not of how a
    caller happened to build the dict.
    """
    if not isinstance(record, dict):
        return record
    seat = record.get("seat")
    if isinstance(seat, dict):
        seat = dict(seat)
        for key in ("model", "account_profile"):
            if key in seat:
                seat[key] = bound_field(seat[key])
        record = dict(record)
        record["seat"] = seat
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        bounded = []
        for att in attempts:
            if isinstance(att, dict) and "model" in att:
                att = dict(att)
                att["model"] = bound_field(att["model"])
            bounded.append(att)
        record = dict(record)
        record["attempts"] = bounded
    return record


def append_record(record):
    """Append one JSONL metrics line. Raises on failure (caller maps to metrics_error)."""
    if not metrics_enabled():
        return
    path = metrics_path()
    directory = os.path.dirname(path)
    _ensure_state_dir(directory)
    # _rotate_if_needed already no-ops on missing path (OSError); no exists guard.
    _rotate_if_needed(path)
    record = _bound_seat_fields(record)
    line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    data = line.encode("utf-8")
    # O_APPEND + write loop (or truncate-back) so a short write never leaves a torn line.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_MODE)
    try:
        try:
            os.fchmod(fd, FILE_MODE)
        except OSError:
            pass
        try:
            pre_size = os.fstat(fd).st_size
        except OSError:
            pre_size = 0
        offset = 0
        try:
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written <= 0:
                    raise OSError(
                        "short metrics write: %d of %d" % (offset, len(data))
                    )
                offset += written
        except Exception:
            # Undo any partial bytes so the file never retains a torn line.
            try:
                os.ftruncate(fd, pre_size)
            except OSError:
                pass
            raise
    finally:
        os.close(fd)


def build_record(
    *,
    entry,
    tool,
    seat,
    attempts,
    fallback_count,
    request_bytes,
    result_bytes,
    latency_ms,
    usage,
    ok,
    reason=None,
    ts=None,
    original_text_bytes=None,
):
    seat = seat or {}
    rec = {
        "ts": time.time() if ts is None else ts,
        "entry": entry,
        "tool": tool,
        "seat": {
            "provider": seat.get("provider"),
            "model": bound_field(seat.get("model")),
            "account_profile": bound_field(seat.get("account_profile")),
        },
        "attempts": attempts,
        "fallback_count": int(fallback_count),
        "request_bytes": int(request_bytes),
        "result_bytes": int(result_bytes),
        "latency_ms": int(latency_ms),
        # Allowlist projection — raw never reaches disk; envelope keeps raw.
        "usage": project_usage(usage),
        "ok": bool(ok),
    }
    if reason is not None:
        rec["reason"] = reason
    # Spilled results only: raw text size before preview truncation (P1.4).
    if original_text_bytes is not None:
        rec["original_text_bytes"] = int(original_text_bytes)
    return rec


def attach_metrics(envelope, record):
    """Write metrics for this call; on failure set metrics_error, keep result."""
    if not isinstance(envelope, dict):
        return envelope
    if not metrics_enabled():
        return envelope
    try:
        append_record(record)
        return envelope
    except Exception as exc:
        out = dict(envelope)
        out["metrics_error"] = redact_secrets(str(exc))[:300]
        return out
