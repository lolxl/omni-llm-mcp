# no-opencc
"""Family short names: a bare family expands to an ordered list of seats.

A family name (``sol``, ``fable``) is matched case-insensitively and only
when the whole bare name — after an ``@effort`` split — equals a key in
``config.FAMILIES``. ``cursor/sol`` is cursor's own alias, not a family.

Each seat is ``{family, provider, model, account_profile}``. ``model`` is
the logical id (``codex/sol``, ``cursor/sol``, ``claude/fable``). The alias
file is not read here. Admission, after the selected account's gate, maps
that id to the built-in or to the file. Env cannot change it.

Env override ``OMNI_LLM_FAMILY_<NAME.upper()>`` is a comma list of
``road[:profile]``. Every listed road must appear in that family's static
tuple. Unknown road, duplicate road, empty list, or a malformed entry is
LOUD ``family_config`` naming the variable.

Profile precedence:
  * a non-default call-level ``account_profile`` applies to EVERY seat
    and overrides any ``:profile`` in the env entries;
  * when the call-level profile is default (``None`` / ``""`` /
    ``"default"``), each env entry's ``:profile`` applies to its own
    seat (default when absent).
"""
import os

from providers.config import FAMILIES

_ENV_PREFIX = "OMNI_LLM_FAMILY_"
_DEFAULT_PROFILES = (None, "", "default")


def _family_key(name):
    return str(name or "").strip().lower()


def is_family(name):
    """True when ``name`` is a family key (case-insensitive, whole name)."""
    key = _family_key(name)
    return bool(key) and key in FAMILIES


def _is_default_profile(account_profile):
    if account_profile is None:
        return True
    return str(account_profile).strip() in ("", "default")


def _norm_profile(account_profile):
    if _is_default_profile(account_profile):
        return "default"
    return str(account_profile)


def _config_error(var, detail):
    return None, {
        "ok": False,
        "reason": "family_config",
        "detail": "%s: %s" % (var, detail),
    }


def _parse_override(raw, table, var):
    """Parse ``road[:profile]`` list. ``table`` is the family's static tuple."""
    allowed = {road: model_id for road, model_id in table}
    parts = [p.strip() for p in str(raw).split(",")]
    if not parts or any(p == "" for p in parts):
        return _config_error(
            var, "empty list or malformed entry (need comma list of road[:profile])")
    seen = []
    out = []
    for part in parts:
        if ":" in part:
            road, profile = part.split(":", 1)
            road = road.strip().lower()
            profile = profile.strip()
            if not road or not profile or ":" in profile:
                return _config_error(
                    var,
                    "malformed entry %r (need road[:profile])" % part,
                )
        else:
            road = part.strip().lower()
            profile = None
            if not road:
                return _config_error(
                    var,
                    "malformed entry %r (need road[:profile])" % part,
                )
        if road not in allowed:
            return _config_error(
                var,
                "unknown road %r (not in family table)" % road,
            )
        if road in seen:
            return _config_error(
                var, "duplicate road %r" % road)
        seen.append(road)
        out.append((road, profile, allowed[road]))
    return out, None


def logical_target(provider, model):
    """``(family, builtin_model_id)`` when ``model`` is that seat's logical id.

    ``claude/fable`` arrives here as ``fable`` after the road prefix is
    stripped. A concrete catalog id is not a family key and returns None.
    This does not read the alias file.
    """
    if not isinstance(model, str):
        return None
    key = _family_key(model)
    for road, model_id in FAMILIES.get(key) or ():
        if road == provider:
            return key, model_id
    return None


def _attempt_gate(provider, profile):
    """The gate ``admission.gate`` returns for this seat's attempt."""
    from providers import admission, dispatch
    name = "default" if profile in (None, "") else str(profile)
    return admission.gate(
        provider, name,
        isolated=dispatch._attempt_isolated(provider, name, False),
        host_env=os.environ,
        settings_files=dispatch._consult_settings(provider, name),
    )


def health_model(model, profile=None):
    """Concrete ``road/id`` a Codex or Claude logical seat shares with health.

    The id is the model admission sends for that attempt. Today's gate sends
    the built-in and does not read the alias file. The catalog gate sends
    the alias file's target, or the built-in when that file is absent or
    refused. A concrete catalog id, and every Cursor seat, returns None.
    Cursor keeps its billing-pool key.
    """
    text = "" if model is None else str(model)
    if "/" not in text:
        return None
    provider, remainder = text.split("/", 1)
    if provider not in ("codex", "claude"):
        return None
    logical = logical_target(provider, remainder)
    if not logical:
        return None
    family, builtin = logical
    target = builtin
    if _attempt_gate(provider, profile) != "today":
        try:
            from providers.aliases import AliasError, effective
            target = effective().get("family.%s.%s" % (family, provider))
        except AliasError:
            target = builtin
        if not isinstance(target, str) or not target.strip():
            target = builtin
    return "%s/%s" % (provider, target)


def expand(name, account_profile):
    """Expand a family name into ``(seats, err)``.

    ``seats`` is a list of dicts; ``err`` is a LOUD envelope or ``None``.
    Seats carry the logical id. They do not read the alias file.
    """
    key = _family_key(name)
    table = FAMILIES.get(key)
    var = _ENV_PREFIX + key.upper()
    if not table:
        return _config_error(var if key else _ENV_PREFIX, "not a family name")

    raw = os.environ.get(var)
    if raw is None:
        entries = [(road, None, model_id) for road, model_id in table]
    else:
        entries, err = _parse_override(raw, table, var)
        if err:
            return None, err

    call_default = _is_default_profile(account_profile)
    call_profile = _norm_profile(account_profile)
    seats = []
    for road, env_profile, _model_id in entries:
        if call_default:
            profile = env_profile if env_profile else "default"
        else:
            profile = call_profile
        seats.append({
            "family": key,
            "provider": road,
            "model": "%s/%s" % (road, key),
            "account_profile": profile,
        })
    return seats, None
