# no-opencc
"""Split a trailing ``@<effort>`` token off a model name.

Consult and agent-lane call sites peel the token BEFORE ``resolve_route``,
so routing always sees the bare name. Cursor consumes the same helper
inside ``resolve_model`` so its argv / envelope stay identical.
"""
from providers.config import EFFORT_LEVELS


def is_agy_opus_alias(name):
    return str(name).strip().lower().startswith("agy-opus-")


def split_effort(name):
    """Return ``(bare_name, token_or_None, error_or_None)``.

    Splits on the last ``@``. The token is stripped and lower-cased and
    must be in ``EFFORT_LEVELS``; otherwise the error is the LOUD
    ``unsupported_effort`` envelope. Empty bare name or empty token use
    the same envelope. A name with no ``@`` is returned unchanged.
    """
    raw = "" if name is None else str(name)
    if "@" not in raw:
        return raw, None, None
    bare, suffix = raw.rsplit("@", 1)
    token = suffix.strip().lower()
    if token == "med" and is_agy_opus_alias(bare):
        token = "medium"
    if not bare.strip() or not token or token not in EFFORT_LEVELS:
        return None, None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": (
                "@%s is not a valid effort; pass --effort with one of %s"
                % (suffix, ", ".join(EFFORT_LEVELS))
            ),
        }
    return bare.strip(), token, None


def take_effort(name, effort):
    """Split ``@<token>`` then merge with ``effort=``.

    Both spellings present (including identical values) is LOUD
    ``unsupported_effort``. Blank ``effort=`` is omitted, matching
    ``normalize_effort``. Returns ``(bare_name, effort_or_None, error)``.
    """
    bare, token, err = split_effort(name)
    if err:
        return None, None, err
    passed = effort is not None and str(effort).strip()
    if token is not None and passed:
        other = str(effort).strip().lower()
        return None, None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": (
                "both @%s on the model and --effort %s were given"
                % (token, other)
            ),
        }
    merged = token if token is not None else effort
    if is_agy_opus_alias(bare):
        from providers.model_catalog import agy_opus_alias_model
        normalized = str(merged or "").strip().lower()
        if normalized == "med":
            normalized = "medium"
        canonical, err = agy_opus_alias_model(bare, normalized)
        return (None, None, err) if err else (canonical, normalized, None)
    return bare, merged, None
