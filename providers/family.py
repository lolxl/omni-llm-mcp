# no-opencc
"""Family short names: a bare family expands to an ordered list of seats.

A family name (``sol``, ``fable``) is matched case-insensitively and only
when the whole bare name — after an ``@effort`` split — equals a key in
``config.FAMILIES``. ``cursor/sol`` is cursor's own alias, not a family.

Each seat is ``{family, provider, model, account_profile}``. ``model`` is
the routable slug (``codex/gpt-5.6-sol``, ``cursor/gpt-5.6-sol``,
``claude/claude-fable-5-1``). The model-id always comes from the static
table; env cannot change it.

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


def expand(name, account_profile):
    """Expand a family name into ``(seats, err)``.

    ``seats`` is a list of dicts; ``err`` is a LOUD envelope or ``None``.
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
    for road, env_profile, model_id in entries:
        if call_default:
            profile = env_profile if env_profile else "default"
        else:
            profile = call_profile
        seats.append({
            "family": key,
            "provider": road,
            "model": "%s/%s" % (road, model_id),
            "account_profile": profile,
        })
    return seats, None
