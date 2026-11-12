"""Admit one model request against one validated catalog record.

``decide`` is pure: it does not spawn a process, use the network, or read a
cache. The caller passes the record, or None when there is no record.
``gate`` chooses today's behavior or a catalog decision for one attempt.
"""
from __future__ import annotations

import datetime as _datetime
import json
import os
import re

from providers.catalog_record import EFFORT_TOKEN_RE
from providers.config import (
    CURSOR_EFFORT_SLUG_OVERRIDES,
    CURSOR_MODEL_ALIASES,
    EFFORT_LEVELS,
)
from providers.model_catalog import DEFAULT_STALE_AFTER_S, _utc_datetime


# Owner ruling, Cut 1. Used only when the record has no effort vocabulary.
GROK_RULED_CLI_VERSION = "1.0.79"
_GROK_RULED_EFFORTS = ("low", "medium", "high", "xhigh")
_BUILTIN_EFFORT = "medium"
# Cursor's closed effort words. ``none`` is a listed suffix. ``fast`` is not a word.
_CURSOR_WORDS = tuple(EFFORT_LEVELS) + ("none",)
_PIN_RE = re.compile(r"^claude-([a-z]+)-([0-9]+)-([0-9]+)$")
_DATED_RE = re.compile(r"^claude-([a-z]+)-([0-9]+)-([0-9]+)-([0-9]{8})$")
_MAJOR_ONLY_RE = re.compile(r"^claude-([a-z]+)-([0-9]+)$")
_DISPLAY_RE = re.compile(r"^([A-Z][a-z]+) ([0-9]+)\.([0-9]+)$")
TERMINAL_CATALOG_REFUSALS = frozenset((
    "no_catalog",
    "catalog_unreadable",
    "unsupported_model",
    "unsupported_effort",
    "effort_required",
    "effort_already_encoded",
    "ambiguous_model",
    "conflicting_evidence",
    "alias_file",
))


# Providers whose attempts never consult a v2 record.
_TODAY_PROVIDERS = frozenset(("pi", "openai", "ollama", "lmstudio", "xai_image"))
# Cursor and agy have no today's gate.
_CATALOG_PROVIDERS = frozenset(("cursor", "agy"))
_HOME_KEYS = {"codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR", "grok": "GROK_HOME"}
_CLAUDE_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "AWS_", "GOOGLE_")
_TODAY_NOTE = "effort: (today's gate)"


def gate(provider, profile, *, isolated, host_env, settings_files):
    """Return ``today`` or ``catalog``.

    ``today`` keeps the attempt's current effort behavior and skips the
    catalog. ``catalog`` means the attempt is admitted against the v2 record.
    Cursor and agy are always ``catalog``. pi and the API providers are
    always ``today``.
    """
    name = str(provider or "").strip().lower()
    if name in _CATALOG_PROVIDERS:
        return "catalog"
    if name in _TODAY_PROVIDERS or name not in ("claude", "codex", "grok"):
        return "today"
    if _unlabeled_home(name, profile, host_env):
        return "today"
    if _auth_env_differs(name, profile, isolated, host_env):
        return "today"
    if name == "claude" and settings_account_affecting(settings_files):
        return "today"
    return "catalog"


def claude_settings_files(config_home, workdir):
    """User and project settings, both ``.json`` and ``.local.json``.

    Missing paths are not account-affecting. Every lane passes both roots.
    """
    files = []
    if config_home:
        root = os.fspath(config_home)
        files.append(os.path.join(root, "settings.json"))
        files.append(os.path.join(root, "settings.local.json"))
    if workdir:
        root = os.path.join(os.fspath(workdir), ".claude")
        files.append(os.path.join(root, "settings.json"))
        files.append(os.path.join(root, "settings.local.json"))
    return files


def settings_account_affecting(paths):
    """True when a claude settings file the lane may load changes the account.

    A missing file does not. A file that cannot be read, decoded, or parsed
    does, including invalid UTF-8.
    """
    for path in paths or ():
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            data = json.loads(raw)
        except FileNotFoundError:
            continue
        except Exception:
            return True
        if not isinstance(data, dict):
            continue
        if "apiKeyHelper" in data:
            return True
        env = data.get("env")
        if not isinstance(env, dict):
            continue
        from providers.host_fence import _VENDOR_AUTH_ENV
        auth_names = _VENDOR_AUTH_ENV.get("claude", frozenset())
        for key in env:
            if (not isinstance(key, str) or key in auth_names
                    or key.startswith(_CLAUDE_ENV_PREFIXES)):
                return True
    return False


def consider(provider, profile, model, effort, *, isolated=False,
             host_env=None, settings_files=None, now=None,
             apply_aliases=False, alias_image=False):
    """Gate one candidate, then decide when the gate is ``catalog``.

    ``effort`` is the caller's effort or None. ``apply_aliases`` is the
    dispatch opt-in: only then does a catalog attempt read the alias file.
    Today's gate returns before that read and keeps the built-ins.
    ``alias_image`` is the grok image admission, whose model is ``image.grok``.
    """
    from providers.family import logical_target
    mode = gate(provider, profile, isolated=bool(isolated), host_env=host_env,
                settings_files=settings_files)
    logical = logical_target(provider, model)
    if mode == "today":
        # The selected account's gate. The alias file is not read.
        decision = _result(effort_source="today_gate", effort_note=_TODAY_NOTE, refusal=None,
                           warnings=[]) | {"gate": "today"}
        if logical:
            decision["model"] = logical[1]
        return decision
    alias_map = None
    if apply_aliases:
        from providers.aliases import AliasError, effective
        try:
            alias_map = effective()
        except AliasError as exc:
            decision = _result(refusal="alias_file", effort_note=exc.detail)
            decision["gate"] = "catalog"
            return decision
        if logical:
            model = alias_map.get("family.%s.%s" % (logical[0], provider))
        elif alias_image:
            model = alias_map.get("image.grok")
    elif logical:
        model = logical[1]
    from providers import catalog_v2
    record, reason = catalog_v2.load(provider, profile or "default")
    moment = now or _datetime.datetime.now(_datetime.timezone.utc)
    request = {"provider": provider, "model": model, "effort": effort}
    if alias_map is not None:
        request["alias_map"] = alias_map.values
        if provider == "codex" and _clean_effort(effort) is None:
            request["omitted_effort"] = alias_map.get("effort.codex")
            request["omitted_effort_from_file"] = (
                alias_map.origins.get("effort.codex") == "file")
    if reason:
        decision = _result(refusal=reason)
    else:
        decision = decide(request, record, now=moment)
    decision = dict(decision)
    decision["gate"] = "catalog"
    return decision


def refusal_envelope(provider, profile, decision):
    """Catalog refusal. The detail names the refresh command for this profile."""
    if isinstance(decision, dict) and decision.get("refusal") == "alias_file":
        return {
            "ok": False,
            "reason": "alias_file",
            "detail": decision.get("effort_note") or "alias file refused",
            "account_retry_safe": False,
        }
    label = profile or "default"
    detail = "models refresh --provider %s --profile %s" % (provider, label)
    note = decision.get("effort_note") if isinstance(decision, dict) else None
    if note:
        detail = "%s. %s" % (note, detail)
    envelope = {
        "ok": False,
        "reason": (decision or {}).get("refusal") or "no_catalog",
        "detail": detail,
        "account_retry_safe": False,
    }
    envelope.update(disclosure(decision))
    return envelope


def disclosure(decision, *, sent_effort=None):
    """Fields every consult and agent envelope carries."""
    if not isinstance(decision, dict):
        return {"effort": None, "effort_source": None, "effort_note": None}
    if decision.get("gate") == "today" or decision.get("effort_source") == "today_gate":
        return {
            "effort": sent_effort,
            "effort_source": "today_gate",
            "effort_note": decision.get("effort_note") or _TODAY_NOTE,
        }
    return {
        "effort": decision.get("effort"),
        "effort_source": decision.get("effort_source"),
        "effort_note": decision.get("effort_note"),
    }


def stamp(envelope, decision, *, sent_effort=None):
    if not isinstance(envelope, dict):
        return envelope
    out = dict(envelope)
    out.update(disclosure(decision, sent_effort=sent_effort))
    return out


def ensure_disclosure(envelope):
    """Consult and agent envelopes always carry the three disclosure keys.

    When no admission decision exists, each value is None.
    """
    if not isinstance(envelope, dict):
        return envelope
    if ("effort" in envelope and "effort_source" in envelope
            and "effort_note" in envelope):
        return envelope
    out = dict(envelope)
    out.setdefault("effort", None)
    out.setdefault("effort_source", None)
    out.setdefault("effort_note", None)
    return out


def is_terminal(decision):
    return (isinstance(decision, dict) and decision.get("gate") == "catalog"
            and decision.get("refusal") in TERMINAL_CATALOG_REFUSALS)


def sent_effort(provider, decision, requested):
    """Effort value the attempt actually puts on argv."""
    if not isinstance(decision, dict) or decision.get("gate") != "today":
        return None if not isinstance(decision, dict) else decision.get("effort")
    text = requested if isinstance(requested, str) else (
        None if requested is None else str(requested))
    if text is not None:
        text = text.strip().lower()
    if provider == "codex":
        from providers.config import DEFAULT_CODEX_REASONING_EFFORT
        return text or DEFAULT_CODEX_REASONING_EFFORT
    return text or None


def bound_call(model, effort, decision):
    """``(model, effort, catalog)`` the adapter sends.

    A catalog admission sends the decision unchanged, including no effort.
    Today's gate keeps the caller's model and effort, except a family logical
    id, which the decision has already replaced with the built-in.
    """
    if (isinstance(decision, dict) and decision.get("gate") == "catalog"
            and not decision.get("refusal")):
        return decision.get("model"), decision.get("effort"), True
    if isinstance(decision, dict) and decision.get("gate") == "today":
        builtin = decision.get("model")
        if isinstance(builtin, str) and builtin:
            return builtin, effort, False
    return model, effort, False


def install_ring_skip(selection, admit_account):
    """Skip catalog refusals inside one account ring, like an ineligible account.

    ``admit_account(account)`` returns one decision. The holder records the
    first refusal. ``selection.next`` then yields only an account the catalog
    (or today's gate) allows.
    """
    holder = {"first": None, "profile": None}
    original = selection.next

    def wrapped():
        while True:
            account = original()
            if account is None:
                return None
            decision = admit_account(account)
            if isinstance(account, dict):
                account["_admission"] = decision
            if is_terminal(decision):
                if holder["first"] is None:
                    holder["first"] = decision
                    holder["profile"] = account.get("account_profile")
                selection.record(account.get("account_profile"), decision.get("refusal"))
                continue
            return account

    selection.next = wrapped
    return holder


def ring_exhausted(provider, result, holder):
    """When every account in the ring catalog-refused, use the first reason."""
    if not isinstance(result, dict) or result.get("reason") != "account_exhausted":
        return None
    tried = result.get("tried") or []
    if not holder.get("first") or not tried:
        return None
    if not all(isinstance(entry, dict) and entry.get("reason") in TERMINAL_CATALOG_REFUSALS
               for entry in tried):
        return None
    envelope = refusal_envelope(provider, holder.get("profile"), holder["first"])
    envelope["tried"] = list(tried)
    return envelope


def _unlabeled_home(provider, profile, host_env):
    """A profile with no ``<VENDOR>_HOME_<LABEL>`` that still has a home.

    The home comes from discovery or from a persisted pin on the attempt.
    The default profile is the normal login, not this case.
    """
    key = _HOME_KEYS.get(provider)
    if not key:
        return False
    name = "default" if profile in (None, "") else str(profile)
    if name == "default":
        return False
    from providers.profiles import PROFILE_NAME_RE
    if not PROFILE_NAME_RE.fullmatch(name):
        return False
    label = "%s_%s" % (key, name.upper())
    attempt = host_env if host_env is not None else os.environ
    if label in os.environ or label in attempt:
        return False
    from providers.provider_accounts import profile_home
    home, err = profile_home(provider, name)
    if err is None and home:
        return True
    pinned = attempt.get(key) if hasattr(attempt, "get") else None
    return bool(pinned)


def _auth_env_differs(provider, profile, isolated, host_env):
    """True when attempt and discovery disagree on a vendor auth variable."""
    from providers.host_fence import _VENDOR_AUTH_ENV
    from providers.model_catalog import _catalog_child_environment
    names = _VENDOR_AUTH_ENV.get(provider)
    if not names:
        return False
    try:
        _overrides, _child, allowed, _home, _bindings = _catalog_child_environment(
            provider, profile or "default")
    except Exception:
        # Includes endpoint_secret_refused. A discovery env that cannot be
        # computed is not a difference, so the gate stays catalog.
        return False
    attempt = dict(host_env if host_env is not None else os.environ)
    if isolated:
        from providers import provider_accounts
        attempt = provider_accounts.account_env(attempt, provider)
    if not isinstance(allowed, dict):
        return True
    for name in names:
        discovered = name in allowed
        attempted = name in attempt
        if discovered != attempted:
            return True
        if discovered and allowed.get(name) != attempt.get(name):
            return True
    return False


def decide(request, record, *, now, stale_after_s=DEFAULT_STALE_AFTER_S):
    """Return the admission result for one request.

    ``request`` carries ``provider``, ``model``, and ``effort`` (or None).
    ``record`` is a validated record,
    or None. A stale record is admitted; ``warnings`` then contains ``stale``.
    """
    if record is None:
        return _result(refusal="no_catalog")
    if not isinstance(record, dict) or not isinstance(request, dict):
        return _result(refusal="catalog_unreadable")
    provider = request.get("provider")
    if not isinstance(provider, str) or not provider:
        return _result(refusal="unsupported_model")
    if record.get("provider") not in (None, provider):
        return _result(refusal="catalog_unreadable")
    warnings = _age_warnings(record, now, stale_after_s)
    rows = [row for row in record.get("models") or [] if isinstance(row, dict)]
    if not rows:
        return _result(refusal="catalog_unreadable", warnings=warnings)
    if provider == "cursor":
        return _decide_cursor(request, rows, warnings)
    if provider == "agy":
        return _decide_agy(request, record, rows, warnings)
    if provider == "claude":
        return _decide_claude(request, rows, warnings)
    if provider in ("codex", "grok"):
        return _decide_flagged(provider, request, record, rows, warnings)
    return _result(refusal="unsupported_model", warnings=warnings)


def _decide_cursor(request, rows, warnings):
    bare, token, split_error = _cursor_request(request)
    if split_error:
        return _result(refusal=split_error, warnings=warnings)
    if not bare:
        return _result(refusal="unsupported_model", warnings=warnings)
    by_id, duplicates = _index(rows)
    ids = set(by_id)
    # A listed id is sent as given. Alias expansion is only for an unlisted name.
    # A dispatch opt-in carries the effective sol/fable targets; an explicit
    # concrete id is never rewritten.
    alias_map = request.get("alias_map") if isinstance(request.get("alias_map"), dict) else None
    if alias_map is not None:
        short = {
            "sol": alias_map.get("family.sol.cursor"),
            "fable": alias_map.get("family.fable.cursor"),
        }
    else:
        short = CURSOR_MODEL_ALIASES
    name = bare if bare in ids else short.get(bare.lower(), bare)
    if name in ids:
        if name in duplicates:
            return _result(refusal="ambiguous_model", warnings=warnings)
        if token is None:
            return _admitted(name, None, "model_id", by_id[name], warnings)
        folded = _cursor_fold(name, token, ids, override=False)
        if folded:
            if folded in duplicates:
                return _result(refusal="ambiguous_model", warnings=warnings)
            return _admitted(folded, token, "request", by_id[folded], warnings)
        if token not in _CURSOR_WORDS:
            return _result(refusal="unsupported_effort", warnings=warnings,
                           effort_note="effort: %s is not an effort" % token)
        # An exact listed id whose `<id>-<effort>` is not listed. No override.
        return _result(refusal="effort_already_encoded", warnings=warnings,
                       effort_note="effort is already encoded in the model id")
    if not _cursor_base(name, ids):
        return _result(refusal="unsupported_model", warnings=warnings)
    if token is None:
        folded = _cursor_fold(name, _BUILTIN_EFFORT, ids)
        if folded:
            if folded in duplicates:
                return _result(refusal="ambiguous_model", warnings=warnings)
            return _admitted(folded, _BUILTIN_EFFORT, "omni_default", by_id[folded], warnings)
        return _result(
            refusal="effort_required", warnings=warnings,
            effort_note=_required_note(_cursor_admitted_words(name, ids)),
        )
    if token not in _CURSOR_WORDS:
        return _result(refusal="unsupported_effort", warnings=warnings,
                       effort_note="effort: %s is not an effort" % token)
    folded = _cursor_fold(name, token, ids)
    if folded:
        if folded in duplicates:
            return _result(refusal="ambiguous_model", warnings=warnings)
        return _admitted(folded, token, "request", by_id[folded], warnings)
    return _result(
        refusal="unsupported_effort", warnings=warnings,
        effort_note="effort: %s is not admitted; admitted efforts: %s"
        % (token, ", ".join(_cursor_admitted_words(name, ids))),
    )


def _cursor_request(request):
    """Return ``(bare, effort_or_None, refusal_or_None)``."""
    raw = request.get("model")
    if not isinstance(raw, str):
        return "", None, "unsupported_model"
    effort = _clean_effort(request.get("effort"))
    if "@" not in raw:
        return raw.strip(), effort, None
    bare, suffix = raw.rsplit("@", 1)
    token = suffix.strip().lower()
    if not bare.strip() or not token:
        return "", None, "unsupported_effort"
    if effort is not None:
        return "", None, "unsupported_effort"
    return bare.strip(), token, None


def _cursor_fold(base, word, ids, *, override=True):
    """Listed ``<base>-<word>``. Overrides apply only when ``override`` is true."""
    if word not in _CURSOR_WORDS:
        return None
    mechanical = "%s-%s" % (base, word)
    if mechanical in ids:
        return mechanical
    if not override:
        return None
    override_slug = CURSOR_EFFORT_SLUG_OVERRIDES.get((base, word))
    if isinstance(override_slug, str) and override_slug in ids:
        return override_slug
    return None


def _cursor_base(name, ids):
    return any(_cursor_fold(name, word, ids) for word in _CURSOR_WORDS)


def _cursor_admitted_words(name, ids):
    return [word for word in _CURSOR_WORDS if _cursor_fold(name, word, ids)]


def _decide_agy(request, record, rows, warnings):
    del record
    raw = request.get("model")
    if not isinstance(raw, str) or not raw.strip():
        return _result(refusal="unsupported_model", warnings=warnings)
    model = raw.strip()
    effort = _clean_effort(request.get("effort"))
    id_hits = [row for row in rows if row.get("id") == model]
    name_hits = [row for row in rows if _row_name(row) == model and row not in id_hits]
    id_ids = {row.get("id") for row in id_hits}
    name_ids = {row.get("id") for row in name_hits}
    if id_ids and name_ids - id_ids:
        return _result(refusal="ambiguous_model", warnings=warnings)
    if len(id_ids | name_ids) > 1:
        return _result(refusal="ambiguous_model", warnings=warnings)
    if not id_ids and not name_ids:
        return _result(refusal="unsupported_model", warnings=warnings)
    if effort is not None:
        return _result(refusal="effort_already_encoded", warnings=warnings,
                       effort_note="effort is already encoded in the model id")
    row = id_hits[0] if id_hits else name_hits[0]
    emitted = row["id"] if id_hits else model
    return _admitted(emitted, None, "model_id", row, warnings)


def _decide_claude(request, rows, warnings):
    model, effort, error = _flagged_request(request)
    if error:
        return _result(refusal=error, warnings=warnings)
    by_id, duplicates = _index(rows)
    # R5 step 1. The vendor listed this exact id, whatever its display name says.
    if model in by_id:
        if model in duplicates:
            return _result(refusal="ambiguous_model", warnings=warnings)
        return _from_row(model, by_id[model], effort, warnings)
    kind, requested = _pin_request(model)
    if kind != "pin":
        return _result(refusal="unsupported_model", warnings=warnings)
    if _conflicting_pin(rows, requested):
        return _result(refusal="conflicting_evidence", warnings=warnings)
    floating = [
        row for row in rows
        if _is_floating(row.get("id")) and _display_version(row) == requested
    ]
    if len(floating) > 1:
        return _result(refusal="ambiguous_model", warnings=warnings)
    if len(floating) != 1:
        return _result(refusal="unsupported_model", warnings=warnings)
    # The emitted id is the requested pin. Effort authority is the floating row.
    return _from_row(model, floating[0], effort, warnings)


def _decide_flagged(provider, request, record, rows, warnings):
    model, effort, error = _flagged_request(request)
    if error:
        return _result(refusal=error, warnings=warnings)
    by_id, duplicates = _index(rows)
    row = by_id.get(model)
    if row is None:
        return _result(refusal="unsupported_model", warnings=warnings)
    if model in duplicates:
        return _result(refusal="ambiguous_model", warnings=warnings)
    omitted = request.get("omitted_effort") if provider == "codex" else None
    omitted_from_file = bool(request.get("omitted_effort_from_file")) if provider == "codex" else False
    if provider == "grok":
        return _from_grok_row(model, row, record, effort, warnings)
    return _from_row(model, row, effort, warnings, omitted=omitted,
                     omitted_from_file=omitted_from_file)


def _from_row(emitted, row, effort, warnings, omitted=None, omitted_from_file=False):
    state, tokens = _effort_state(row)
    if omitted_from_file and effort is None and state in ("empty", "absent"):
        # The file named an effort the row cannot evidence. Do not admit with none.
        return _result(refusal="unsupported_effort", warnings=warnings,
                       effort_note="effort: %s is not admitted" % omitted)
    if state == "empty":
        if effort is None:
            return _admitted(emitted, None, "not_applicable", row, warnings)
        return _result(refusal="unsupported_effort", warnings=warnings,
                       effort_note="effort: %s is not admitted" % effort)
    if state == "absent":
        if effort is None:
            return _admitted(emitted, None, "vendor", row, warnings)
        return _result(refusal="unsupported_effort", warnings=warnings,
                       effort_note="effort: %s is not admitted" % effort)
    return _from_tokens(emitted, row, tokens, effort, warnings, omitted=omitted)


def _from_grok_row(emitted, row, record, effort, warnings):
    """Grok evidence is the row, else the recorded vocabulary, else the Owner set."""
    state, tokens = _effort_state(row)
    if state == "empty":
        if effort is None:
            return _admitted(emitted, None, "not_applicable", row, warnings)
        return _result(refusal="unsupported_effort", warnings=warnings,
                       effort_note="effort: %s is not admitted" % effort)
    if state == "listed":
        return _from_tokens(emitted, row, tokens, effort, warnings)
    vocabulary = record.get("effort_vocabulary") if isinstance(record, dict) else None
    recorded = vocabulary.get("tokens") if isinstance(vocabulary, dict) else None
    if isinstance(recorded, list) and recorded:
        tokens = tuple(token for token in recorded if isinstance(token, str))
    else:
        tokens = _GROK_RULED_EFFORTS
    return _from_tokens(emitted, row, tokens, effort, warnings)


def _from_tokens(emitted, row, tokens, effort, warnings, omitted=None):
    token_set = set(tokens)
    if effort is None:
        default = omitted or _BUILTIN_EFFORT
        if default in token_set:
            return _admitted(emitted, default, "omni_default", row, warnings)
        return _result(
            refusal="effort_required", warnings=warnings,
            effort_note=_required_note(_ordered(tokens)),
        )
    if effort in token_set:
        return _admitted(emitted, effort, "request", row, warnings)
    return _result(
        refusal="unsupported_effort", warnings=warnings,
        effort_note="effort: %s is not admitted; admitted efforts: %s"
        % (effort, ", ".join(_ordered(tokens))),
    )


def _conflicting_pin(rows, requested):
    """R5 step 2. A display name with no parseable version never conflicts."""
    for row in rows:
        own = _row_pin_version(row.get("id"))
        display = _display_version(row)
        if own is None or display is None:
            continue
        if display != own and display == requested:
            return True
    return False


def _pin_request(model):
    """Return ``(kind, version)``. Dated ids and major-only ids match by exact id."""
    if _DATED_RE.fullmatch(model) or _MAJOR_ONLY_RE.fullmatch(model):
        return "exact_only", None
    match = _PIN_RE.fullmatch(model)
    if not match:
        return "other", None
    return "pin", (match.group(1), int(match.group(2)), int(match.group(3)))


def _row_pin_version(model_id):
    if not isinstance(model_id, str):
        return None
    match = _PIN_RE.fullmatch(model_id) or _DATED_RE.fullmatch(model_id)
    if not match:
        return None
    return (match.group(1), int(match.group(2)), int(match.group(3)))


def _display_version(row):
    text = row.get("display_name")
    if not isinstance(text, str) or not text:
        text = row.get("name")
    if not isinstance(text, str):
        return None
    match = _DISPLAY_RE.fullmatch(text)
    if not match:
        return None
    family = match.group(1)
    if family != family.lower().capitalize():
        return None
    return (family.lower(), int(match.group(2)), int(match.group(3)))


def _is_floating(model_id):
    return isinstance(model_id, str) and bool(model_id) and not any(ch.isdigit() for ch in model_id)


def _effort_state(row):
    field = row.get("effort_field")
    raw = row.get("supported_efforts", None)
    known = row.get("efforts_known", None)
    if field == "absent" or (field is None and known is False and not raw):
        return "absent", ()
    if field == "empty" or (field is None and known is True and not raw):
        return "empty", ()
    if field == "listed" or (field is None and known is True and raw):
        return "listed", tuple(raw or ())
    if raw is None or raw == []:
        return "absent", ()
    return "listed", tuple(raw)


def _flagged_request(request):
    raw = request.get("model")
    if not isinstance(raw, str) or not raw.strip():
        return "", None, "unsupported_model"
    effort = _clean_effort(request.get("effort"))
    if effort is not None and not EFFORT_TOKEN_RE.fullmatch(effort):
        return raw.strip(), None, "unsupported_effort"
    return raw.strip(), effort, None


def _clean_effort(value):
    if value is None:
        return None
    if not isinstance(value, str):
        text = str(value).strip().lower()
    else:
        text = value.strip().lower()
    return text or None


def _index(rows):
    by_id = {}
    duplicates = set()
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if model_id in by_id:
            duplicates.add(model_id)
        by_id[model_id] = row
    return by_id, duplicates


def _row_name(row):
    name = row.get("name")
    return name if isinstance(name, str) else ""


def _row_label(row):
    display = row.get("display_name")
    if isinstance(display, str) and display:
        return display
    return _row_name(row)


def _ordered(tokens):
    return sorted(tokens, key=lambda token: (
        0, ("minimal", "low", "medium", "high", "xhigh", "max", "ultra").index(token)
    ) if token in ("minimal", "low", "medium", "high", "xhigh", "max", "ultra") else (1, token))


def _required_note(tokens):
    shown = ", ".join(tokens) if tokens else "(none)"
    return "effort required; admitted efforts: %s" % shown


def _note(effort, source):
    if source == "omni_default":
        return "effort: %s (chosen by omni; no effort was given)" % effort
    if source == "request":
        return "effort: %s (as requested)" % effort
    if source == "model_id":
        return "effort: (encoded in the model id)"
    if source == "not_applicable":
        return "effort: (not applicable; the listing gives an empty effort field)"
    if source == "vendor":
        return "effort: (listing has no effort info)"
    return None


def _admitted(model, effort, source, row, warnings):
    return _result(
        model=model,
        effort=effort,
        effort_source=source,
        effort_note=_note(effort, source),
        evidence_row=dict(row),
        warnings=list(warnings),
        refusal=None,
    )


def _age_warnings(record, now, stale_after_s):
    parsed = _utc_datetime(record.get("observed_at")) if isinstance(record.get("observed_at"), str) else None
    if parsed is None or not isinstance(now, _datetime.datetime):
        return []
    moment = now if now.tzinfo is not None else now.replace(tzinfo=_datetime.timezone.utc)
    moment = moment.astimezone(_datetime.timezone.utc)
    if (moment - parsed).total_seconds() > float(stale_after_s):
        return ["stale"]
    return []


def _result(*, model=None, effort=None, effort_source=None, effort_note=None,
            evidence_row=None, warnings=None, refusal):
    return {
        "model": model,
        "effort": effort,
        "effort_source": effort_source,
        "effort_note": effort_note,
        "evidence_row": evidence_row,
        "warnings": list(warnings or []),
        "refusal": refusal,
    }
