# no-opencc
"""Stderr logging helpers (never touch stdout — that is the MCP wire)."""
import json
import sys
import time

from providers import known_secrets


def log(*a):
    """Free-text stderr log (startup, rare diagnostics). Prefer log_event for tool traffic."""
    print("[omni-llm-mcp]", *(known_secrets.mask_known(str(x)) for x in a),
          file=sys.stderr, flush=True)


def _mask_fields(value, depth=0):
    """Every string field masked ALONE first, so a value a caller cut short
    (``str(exc)[:300]``) is caught at the end of its field too."""
    if isinstance(value, str):
        return known_secrets.mask_known(value)
    if depth < 8 and isinstance(value, dict):
        return {k: _mask_fields(v, depth + 1) for k, v in value.items()}
    if depth < 8 and isinstance(value, (list, tuple)):
        return [_mask_fields(v, depth + 1) for v in value]
    return value


def log_event(event, **fields):
    """One structured JSON line on stderr for observability."""
    payload = {"event": event, "ts": round(time.time(), 3)}
    payload.update(_mask_fields(fields) if known_secrets.active() else fields)
    line = json.dumps(payload, ensure_ascii=False)
    print("[omni-llm-mcp]", known_secrets.mask_known(line),
          file=sys.stderr, flush=True)
