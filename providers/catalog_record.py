"""Strict catalog records and display-name parsers.

Nothing in the package calls this yet. Parsing and validation are local:
they do not spawn a process, use the network, or read a cache.
"""
from __future__ import annotations

import re

from providers.model_catalog import (
    MAX_MODEL_COUNT,
    ROAD_PREFIX,
    _DISCOVERY_POSTURES,
    _ID_RE,
    _utc_datetime,
    _utc_now,
)


# Known effort tokens first, then any other accepted token alphabetically.
_KNOWN_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
EFFORT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_VERSION_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+")
_SOURCE_NOTES = (
    "vendor_cache_not_live",
    "unauthenticated_vendor_registry",
    "vendor_registry_not_entitlement",
)
_VOCABULARY_SOURCES = ("recorded", "owner_ruled")


class StrictRecordError(Exception):
    """The value cannot be used as catalog evidence."""

    def __init__(self, reason, detail):
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(self.detail)


def grok_cli_version(text):
    """Return the first whitespace token matching ``\\d+(\\.\\d+)+``, or None."""
    if not isinstance(text, str):
        return None
    for token in text.split():
        if _VERSION_TOKEN_RE.fullmatch(token):
            return token
    return None


def validate_strict(provider, value):
    """Return ``(record, warnings)`` or raise ``StrictRecordError``.

    Manual records are not evidence. The raw ``supported_efforts`` field is
    the effort state: omitted when absent, ``[]`` when known-empty, or the
    tokens that matched ``EFFORT_TOKEN_RE``. A bad token is dropped with a
    warning. If every token is dropped, the key is omitted, not written as
    ``[]``. ``effort_field`` and ``efforts_known`` repeat that state for
    readers that still branch on them. They are not a second input.
    """
    if provider not in ROAD_PREFIX or not isinstance(value, dict):
        _unreadable("catalog record is missing or not an object")
    source = value.get("source")
    if source == "manual":
        _unreadable("manual catalog records are not evidence")
    if source not in ("live", "bundled"):
        _unreadable("catalog source is not live or bundled evidence")
    observed_at = value.get("observed_at")
    if _utc_datetime(observed_at) is None:
        _unreadable("catalog observed_at is missing or not UTC")
    if value.get("provider") not in (None, provider):
        _unreadable("catalog provider does not match the record")
    version = value.get("cli_version")
    if version is not None and (not isinstance(version, str) or len(version) > 100):
        _unreadable("catalog cli_version is invalid")
    availability = value.get("availability")
    if availability not in ("provider_advertised", "unknown"):
        _unreadable("catalog availability is invalid")
    models = value.get("models")
    if not isinstance(models, list) or not models or len(models) > MAX_MODEL_COUNT:
        _unreadable("catalog model list is missing or empty")

    warnings = []
    normalized = []
    seen = set()
    for item in models:
        normalized.append(_normalize_row(provider, item, seen, warnings))
    result = {
        "provider": provider,
        "source": source,
        "observed_at": observed_at,
        "availability": availability,
        "models": normalized,
    }
    if version is not None:
        result["cli_version"] = version
    source_note = value.get("source_note")
    if source_note in _SOURCE_NOTES:
        result["source_note"] = source_note
    if "discovery" in value:
        result["discovery"] = _discovery(value["discovery"])
    if "grok_default" in value:
        result["grok_default"] = _grok_default(value["grok_default"], seen)
    if "effort_vocabulary" in value:
        vocabulary = _vocabulary(value["effort_vocabulary"], warnings)
        if vocabulary is not None:
            result["effort_vocabulary"] = vocabulary
    return result, warnings


def parse_claude_models(models, *, observed_at=None, cli_version=None):
    """Parse a claude model list, keeping each row's display name.

    Effort tokens are kept for ``validate_strict``. A token ``validate_strict``
    accepts, including one outside today's fixed set, does not fail the parse.
    """
    if not isinstance(models, list) or not models:
        _unreadable("claude model list is empty")
    rows = []
    for item in models:
        if not isinstance(item, dict):
            _unreadable("claude model entry is not an object")
        rows.append(_vendor_row(
            "claude", item,
            ("value", "id"),
            ("displayName", "display_name", "name"),
            ("supportedEffortLevels",),
        ))
    return _assemble("claude", rows, observed_at=observed_at, cli_version=cli_version)


def parse_codex_result(result, *, observed_at=None, cli_version=None):
    """Parse a codex ``model/list`` result.

    Any effort token matching the strict pattern is kept. Other tokens stay
    in the row so ``validate_strict`` can drop them one at a time.
    """
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        _unreadable("codex model list is missing")
    if not result["data"]:
        _unreadable("codex model list is empty")
    rows = []
    for item in result["data"]:
        if not isinstance(item, dict):
            _unreadable("codex model entry is not an object")
        row = _vendor_row(
            "codex", item,
            ("model", "id"),
            ("displayName", "name"),
            ("supportedReasoningEfforts",),
            default_keys=("defaultReasoningEffort",),
        )
        if isinstance(item.get("hidden"), bool):
            row["visibility"] = "hidden" if item["hidden"] else "list"
        rows.append(row)
    return _assemble("codex", rows, observed_at=observed_at, cli_version=cli_version)


def parse_grok_output(output, *, observed_at=None, cli_version=None):
    """Parse grok ``models`` output, keeping both default markers.

    ``grok_default.line`` is the ``Default model:`` id. ``grok_default.row``
    is the row marked ``(default)``. Routing does not read either field.
    """
    if not isinstance(output, str):
        _unreadable("grok model list is missing")
    if output and not output.endswith("\n"):
        _unreadable("grok model list is incomplete")
    rows = []
    saw_header = False
    line_default = None
    row_default = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "You are not authenticated.":
            _unreadable("grok model list is unauthenticated")
        if stripped == "You are logged in with grok.com.":
            continue
        if not stripped:
            continue
        if stripped.startswith("Default model: "):
            line_default = stripped.split(":", 1)[1].strip()
            continue
        if stripped == "Available models:":
            if saw_header:
                _unreadable("grok model list repeats its header")
            saw_header = True
            continue
        if not saw_header:
            _unreadable("grok model list has no header")
        marked = re.fullmatch(r"[*-]\s+([^\s]+)\s+\(default\)", stripped)
        plain = re.fullmatch(r"[*-]\s+([^\s]+)", stripped)
        match = marked or plain
        if not match:
            _unreadable("grok model line is malformed")
        model_id = match.group(1)
        if marked:
            if row_default is not None:
                _unreadable("grok model list marks more than one default row")
            row_default = model_id
        rows.append({
            "id": model_id,
            "name": model_id,
            "routable_id": ROAD_PREFIX["grok"] + model_id,
            "efforts_known": False,
        })
    ids = {row["id"] for row in rows}
    if not saw_header or not rows or (line_default and line_default not in ids):
        _unreadable("grok model list is incomplete")
    record = _assemble(
        "grok", rows, observed_at=observed_at, cli_version=cli_version,
        availability="unknown", source_note="vendor_registry_not_entitlement",
    )
    record["grok_default"] = {"line": line_default, "row": row_default}
    return record


def _assemble(provider, rows, *, observed_at, cli_version, availability="provider_advertised",
              source_note=None):
    record = {
        "provider": provider,
        "source": "live",
        "observed_at": observed_at or _utc_now(),
        "availability": availability,
        "models": rows,
    }
    if cli_version:
        record["cli_version"] = cli_version
    if source_note:
        record["source_note"] = source_note
    return record


def _vendor_row(provider, value, id_keys, name_keys, effort_keys, default_keys=()):
    model_id = _first_text(value, id_keys)
    if not model_id:
        _unreadable("%s model id is missing" % provider)
    display_name = _first_text(value, name_keys) or model_id
    present, tokens, default = _raw_efforts(value, effort_keys, default_keys, provider)
    row = {
        "id": model_id,
        "name": display_name,
        "display_name": display_name,
        "routable_id": ROAD_PREFIX[provider] + model_id,
        "efforts_known": present,
    }
    if present:
        row["supported_efforts"] = tokens
    if default is not None:
        row["default_effort"] = default
    return row


def _first_text(value, keys):
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _raw_efforts(metadata, keys, default_keys, provider):
    """Return ``(present, tokens, default)``.

    ``present`` is false when every effort key is absent. A present empty
    list is known-empty. Tokens are not filtered here.
    """
    present = False
    tokens = []
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, list):
            _unreadable("%s effort field is not a list" % provider)
        present = True
        for entry in value:
            tokens.append(_token_text(entry, provider))
        break
    defaults = [metadata[key] for key in default_keys if key in metadata]
    if any(not isinstance(item, str) for item in defaults) or len(set(defaults)) > 1:
        _unreadable("%s default effort is invalid" % provider)
    return present, tokens, (defaults[0] if defaults else None)


def _token_text(entry, provider):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        declared = [entry[name] for name in ("reasoningEffort", "effort", "value") if name in entry]
        if len(declared) == 1 and isinstance(declared[0], str):
            return declared[0]
        _unreadable("%s effort entry is not a token" % provider)
    # A non-string element, such as 42, reaches validate_strict and is dropped.
    return entry


def _normalize_row(provider, item, seen, warnings):
    if not isinstance(item, dict):
        _unreadable("catalog model entry is not an object")
    model_id = item.get("id")
    if not isinstance(model_id, str) or not _ID_RE.fullmatch(model_id):
        _unreadable("catalog model id is invalid")
    if model_id in seen:
        _unreadable("catalog model id is duplicated")
    seen.add(model_id)
    name = item.get("name", model_id)
    if not isinstance(name, str) or not _ID_RE.fullmatch(name):
        _unreadable("catalog model name is invalid")
    routable_id = item.get("routable_id", ROAD_PREFIX[provider] + model_id)
    if routable_id != ROAD_PREFIX[provider] + model_id:
        _unreadable("catalog routable id is invalid")
    state, tokens = _input_efforts(item)
    if state == "listed":
        tokens = _keep_tokens(model_id, tokens, warnings)
        state = "listed" if tokens else "absent"
    # Absent omits the key. ``[]`` is only the known-empty state, so a second
    # validate_strict call does not turn an absent row into known-empty.
    row = {
        "id": model_id,
        "name": name,
        "routable_id": routable_id,
        "efforts_known": state != "absent",
        "effort_field": state,
    }
    if state == "listed":
        row["supported_efforts"] = list(tokens)
    elif state == "empty":
        row["supported_efforts"] = []
    if "display_name" in item:
        display_name = item["display_name"]
        if not isinstance(display_name, str) or not _ID_RE.fullmatch(display_name):
            _unreadable("catalog display name is invalid")
        row["display_name"] = display_name
    default = item.get("default_effort")
    if isinstance(default, str) and state == "listed" and default in row["supported_efforts"]:
        row["default_effort"] = default
    elif default is not None and not (isinstance(default, str) and EFFORT_TOKEN_RE.fullmatch(default)):
        warnings.append({
            "reason": "dropped_effort_token",
            "model_id": model_id,
            "token": default if isinstance(default, str) else repr(default),
        })
    visibility = item.get("visibility")
    if visibility is not None:
        if not isinstance(visibility, str) or not _ID_RE.fullmatch(visibility):
            _unreadable("catalog visibility is invalid")
        row["visibility"] = visibility
    return row


def _input_efforts(item):
    """Return ``(state, tokens)`` from ``supported_efforts`` alone.

    The field is absent, ``[]`` (known-empty), or a list. ``effort_field``
    and ``efforts_known`` are not a second state and cannot override it.
    Bad tokens are dropped by the caller; a list whose tokens are all
    dropped is absent, not known-empty.
    """
    if "supported_efforts" not in item or item.get("supported_efforts") is None:
        return "absent", []
    raw = item["supported_efforts"]
    if not isinstance(raw, list):
        _unreadable("catalog effort field is not a list")
    if raw == []:
        return "empty", []
    return "listed", list(raw)


def _keep_tokens(model_id, raw_tokens, warnings):
    kept = []
    seen = set()
    for token in raw_tokens:
        if isinstance(token, str) and token not in seen and EFFORT_TOKEN_RE.fullmatch(token):
            kept.append(token)
            seen.add(token)
            continue
        if isinstance(token, str) and token in seen and EFFORT_TOKEN_RE.fullmatch(token):
            continue
        warnings.append({
            "reason": "dropped_effort_token",
            "model_id": model_id,
            "token": token if isinstance(token, str) else repr(token),
        })
    kept.sort(key=_effort_sort_key)
    return kept


def _effort_sort_key(token):
    try:
        return (0, _KNOWN_EFFORT_ORDER.index(token))
    except ValueError:
        return (1, token)


def _discovery(value):
    posture = value.get("posture") if isinstance(value, dict) else None
    system = value.get("platform") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or set(value) != {"posture", "platform"}
            or posture not in _DISCOVERY_POSTURES or not isinstance(system, str)):
        _unreadable("catalog discovery is invalid")
    return {"posture": posture, "platform": system}


def _grok_default(value, ids):
    if not isinstance(value, dict) or set(value) - {"line", "row"}:
        _unreadable("grok default marker is invalid")
    cleaned = {}
    for key in ("line", "row"):
        if key not in value:
            cleaned[key] = None
            continue
        item = value[key]
        if item is None:
            cleaned[key] = None
            continue
        if not isinstance(item, str) or item not in ids:
            _unreadable("grok default marker is not a listed model id")
        cleaned[key] = item
    return {"line": cleaned["line"], "row": cleaned["row"]}


def _vocabulary(value, warnings):
    if not isinstance(value, dict):
        _unreadable("grok effort vocabulary is invalid")
    source = value.get("source")
    version = value.get("cli_version")
    tokens = value.get("tokens")
    if source not in _VOCABULARY_SOURCES:
        _unreadable("grok effort vocabulary source is invalid")
    if not isinstance(version, str) or not version or len(version) > 100:
        _unreadable("grok effort vocabulary cli version is invalid")
    if not isinstance(tokens, list):
        _unreadable("grok effort vocabulary tokens are invalid")
    kept = _keep_tokens(None, tokens, warnings)
    if not kept:
        return None
    return {"source": source, "cli_version": version, "tokens": kept}


def _unreadable(detail):
    raise StrictRecordError("catalog_unreadable", detail)
