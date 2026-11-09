# no-opencc
"""Read successor model/effort from the vendor's own session record.

Never fall back to the requested model or effort. Missing file, unreadable
JSON, missing key, or a null session_id yields ``"unknown"`` for that field.
"""
import glob
import json
import os
import re

from providers.logutil import log_event

UNKNOWN = "unknown"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _expand_tilde(path, env_home):
    """Expand a leading ``~`` against *env_home* when set, else process HOME."""
    if path == "~" or path.startswith("~/") or path.startswith("~" + os.sep):
        if env_home:
            if path == "~":
                return env_home
            return os.path.join(env_home, path[2:])
        return os.path.expanduser(path)
    return path


def _home(env, env_key, default_dotdir, workdir=None):
    """Vendor home for *env_key*, or None when it cannot be resolved.

    An explicit empty ``HOME`` on the child env means the child had no home:
    a ``~`` path then resolves to nothing rather than to the runner's home.
    """
    env = env or {}
    env_home = env.get("HOME")
    if env_home is not None:
        env_home = str(env_home).strip()
        if not env_home:
            raw_override = env.get(env_key)
            text = str(raw_override).strip() if raw_override is not None else ""
            if not text or text.startswith("~"):
                return None
            env_home = None
    raw = env.get(env_key)
    if raw is not None and str(raw).strip():
        path = _expand_tilde(str(raw).strip(), env_home)
        if not os.path.isabs(path) and workdir:
            path = os.path.join(workdir, path)
        return os.path.abspath(path)
    default = _expand_tilde(os.path.join("~", default_dotdir), env_home)
    return os.path.abspath(default)


def _unknown(session_id):
    return {
        "session_id": session_id,
        "model": UNKNOWN,
        "effort": UNKNOWN,
        "source": UNKNOWN,
    }


def _as_token(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNKNOWN


def _load_jsonl_first(path, type_name):
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(obj, dict) and obj.get("type") == type_name:
                    return obj
    except OSError:
        return None
    return None


def _first_match(pattern):
    matches = sorted(p for p in glob.glob(pattern, recursive=True)
                     if os.path.isfile(p))
    return matches[0] if matches else None


def _unreadable(provider, pattern, session_id):
    log_event("attest_unreadable", provider=provider, pattern=pattern)
    return _unknown(session_id)


def read_successor_attest(provider, session_id, env, workdir=None):
    """Return ``{session_id, model, effort, source}`` from the vendor record.

    Paths honour ``CLAUDE_CONFIG_DIR`` / ``CODEX_HOME`` / ``GROK_HOME`` on
    *env*, else ``~/.claude`` / ``~/.codex`` / ``~/.grok``. ``~`` expands
    against ``env["HOME"]`` when present. A relative override resolves
    against *workdir* when given, else the process cwd.
    """
    sid = str(session_id) if session_id else None
    if not sid:
        pattern = {
            "claude": "<claude home>/projects/*/<session_id>.jsonl",
            "codex": "<codex home>/sessions/**/rollout-*-<session_id>.jsonl",
            "grok": "<grok home>/sessions/*/<session_id>/summary.json",
        }.get(provider, "<none>")
        if provider in ("claude", "codex", "grok"):
            log_event("attest_unreadable", provider=provider, pattern=pattern)
        return _unknown(None)
    if not _SESSION_ID_RE.fullmatch(sid):
        # Not a vendor session id shape: never glob for it, whatever it is.
        log_event("attest_unreadable", provider=provider,
                  reason="session_id_invalid")
        return _unknown(sid)
    if provider in ("pi", "cursor"):
        return {
            "session_id": sid,
            "model": UNKNOWN,
            "effort": UNKNOWN,
            "source": UNKNOWN,
        }
    env = env if isinstance(env, dict) else {}
    sid_glob = glob.escape(sid)
    if provider == "claude":
        home = _home(env, "CLAUDE_CONFIG_DIR", ".claude", workdir=workdir)
        if home is None:
            return _unreadable(provider, "<child HOME empty>", sid)
        pattern = os.path.join(home, "projects", "*", "%s.jsonl" % sid_glob)
        path = _first_match(pattern)
        if not path:
            return _unreadable(provider, pattern, sid)
        row = _load_jsonl_first(path, "assistant")
        if not isinstance(row, dict):
            return _unreadable(provider, pattern, sid)
        message = row.get("message")
        model = UNKNOWN
        if isinstance(message, dict):
            model = _as_token(message.get("model"))
        effort = UNKNOWN
        if "effort" in row:
            effort = _as_token(row.get("effort"))
        return {
            "session_id": sid,
            "model": model,
            "effort": effort,
            "source": os.path.abspath(path),
        }
    if provider == "codex":
        home = _home(env, "CODEX_HOME", ".codex", workdir=workdir)
        if home is None:
            return _unreadable(provider, "<child HOME empty>", sid)
        pattern = os.path.join(
            home, "sessions", "**", "rollout-*-%s.jsonl" % sid_glob)
        path = _first_match(pattern)
        if not path:
            return _unreadable(provider, pattern, sid)
        row = _load_jsonl_first(path, "turn_context")
        if not isinstance(row, dict):
            return _unreadable(provider, pattern, sid)
        payload = row.get("payload")
        model = UNKNOWN
        effort = UNKNOWN
        if isinstance(payload, dict):
            model = _as_token(payload.get("model"))
            if "effort" in payload:
                effort = _as_token(payload.get("effort"))
        return {
            "session_id": sid,
            "model": model,
            "effort": effort,
            "source": os.path.abspath(path),
        }
    if provider == "grok":
        home = _home(env, "GROK_HOME", ".grok", workdir=workdir)
        if home is None:
            return _unreadable(provider, "<child HOME empty>", sid)
        pattern = os.path.join(
            home, "sessions", "*", sid_glob, "summary.json")
        path = _first_match(pattern)
        if not path:
            return _unreadable(provider, pattern, sid)
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return _unreadable(provider, pattern, sid)
        if not isinstance(doc, dict):
            return _unreadable(provider, pattern, sid)
        return {
            "session_id": sid,
            "model": _as_token(doc.get("current_model_id")),
            "effort": _as_token(doc.get("reasoning_effort")),
            "source": os.path.abspath(path),
        }
    return _unknown(sid)
