# no-opencc
"""Best-effort structured notification seam.

Always records via log_event (stderr; stdout is the MCP wire and is never
touched). An optional external sink named by OMNI_LLM_NOTIFY_CMD then
receives the same JSON object on stdin. OMNI_LLM_NOTIFY_QUIET=1 keeps the
stderr line and skips that external command. A missing binary, non-zero
exit, timeout, or any exception is swallowed after one log_event recording
the delivery failure. This module never raises and never blocks beyond the
hard timeout.
"""
import json
import os
import shlex
import subprocess

from providers.logutil import log_event

ENV_NOTIFY_CMD = "OMNI_LLM_NOTIFY_CMD"
ENV_NOTIFY_QUIET = "OMNI_LLM_NOTIFY_QUIET"
NOTIFY_TIMEOUT_S = 2.0


def notify(event, **fields):
    """Deliver one structured event. Never raises. Never writes stdout."""
    try:
        log_event(event, **fields)
    except Exception:
        pass
    if os.environ.get(ENV_NOTIFY_QUIET) == "1":
        return
    raw = os.environ.get(ENV_NOTIFY_CMD)
    if raw is None:
        return
    cmd = str(raw).strip()
    if not cmd:
        return
    payload = {"event": event}
    payload.update(fields)
    try:
        argv = shlex.split(cmd, posix=True)
        if not argv:
            return
        _run_sink(argv, payload)
    except Exception as exc:
        try:
            log_event(
                "notify_failed",
                event=str(event),
                reason=_fail_reason(exc),
                detail=str(exc)[:300],
            )
        except Exception:
            pass


def _fail_reason(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, FileNotFoundError):
        return "missing"
    return type(exc).__name__


def _run_sink(argv, payload):
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    proc = subprocess.run(
        argv,
        input=blob,
        timeout=NOTIFY_TIMEOUT_S,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError("exit %s" % proc.returncode)
