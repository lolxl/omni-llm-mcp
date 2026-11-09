# no-opencc
"""Optional, default-OFF egress privacy gate (thin refuse layer).

When ``OMNI_EGRESS_GATE=1``, outgoing provider payloads are scanned with
deterministic regex families (and an optional literal deny-list) and refused
before send. Any other value or an unset env leaves the gate disabled — that
bare-run gap is intentional.
"""
import os
import re

from providers.errors import redact_secrets

ENV_GATE = "OMNI_EGRESS_GATE"
ENV_DENYLIST = "OMNI_EGRESS_DENYLIST"

# RFC 2606 reserved names. An email whose domain is under one of these is
# treated as a fixture, not a hit.
_RESERVED_EXACT = frozenset({
    "example.com", "example.net", "example.org",
    "test", "example", "invalid", "localhost",
})
_RESERVED_SUFFIXES = (
    ".example.com", ".example.net", ".example.org",
    ".test", ".example", ".invalid", ".localhost",
)

PATTERNS = {
    "tw_mobile": re.compile(r"09\d{8}"),
    "tw_national_id": re.compile(r"[A-Z][12]\d{8}"),
    "email": re.compile(
        r"(?i)[A-Z0-9._%+\-]+@(?:[A-Z0-9\-]+\.)+[A-Z]{2,}"
    ),
}

_DENY_PREFIX = "OMNI-EGRESS-DENY"


class EgressGateError(Exception):
    """Fail-closed gate error (missing/unreadable deny-list, etc.)."""


def enabled():
    """True only when OMNI_EGRESS_GATE is exactly ``1``."""
    return os.environ.get(ENV_GATE) == "1"


def _mask(value):
    """At most the first 2 characters plus length. Never the full match."""
    text = str(value)
    return "%s…(len=%d)" % (text[:2], len(text))


def _domain_reserved(domain):
    d = (domain or "").lower().rstrip(".")
    if d in _RESERVED_EXACT:
        return True
    return any(d.endswith(sfx) for sfx in _RESERVED_SUFFIXES)


def _email_is_reserved(addr):
    if "@" not in addr:
        return False
    return _domain_reserved(addr.rsplit("@", 1)[1])


def _load_denylist():
    """Return literal strings, or None when the env is unset.

    Missing / unreadable file with the env set is fail-closed.
    """
    path = os.environ.get(ENV_DENYLIST)
    if path is None or path == "":
        return None
    if not os.path.isfile(path):
        raise EgressGateError(
            "%s [denylist] deny-list file missing" % _DENY_PREFIX
        )
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except (OSError, UnicodeError):
        raise EgressGateError(
            "%s [denylist] deny-list file unreadable" % _DENY_PREFIX
        )
    literals = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        literals.append(stripped)
    return literals


def check_strings(iterable_of_str):
    """Scan strings; return ``[(family, masked_excerpt), ...]``.

    The matched value itself is never returned. Raises ``EgressGateError``
    when ``OMNI_EGRESS_DENYLIST`` is set but the file cannot be used.
    """
    literals = _load_denylist()
    hits = []
    for item in iterable_of_str or ():
        if item is None:
            continue
        text = item if isinstance(item, str) else str(item)
        for family, regex in PATTERNS.items():
            for match in regex.finditer(text):
                value = match.group(0)
                if family == "email" and _email_is_reserved(value):
                    continue
                hits.append((family, _mask(value)))
        if literals:
            for literal in literals:
                if literal and literal in text:
                    hits.append(("denylist", _mask(literal)))
    return hits


def refuse_payload(*strings):
    """If the gate is on and *strings hit, return a LOUD envelope; else None."""
    if not enabled():
        return None
    try:
        hits = check_strings(strings)
    except EgressGateError as exc:
        return {"ok": False, "reason": "runtime_error", "detail": redact_secrets(str(exc))[:300]}
    if not hits:
        return None
    family, excerpt = hits[0]
    detail = "%s [%s] %s" % (_DENY_PREFIX, family, excerpt)
    return {"ok": False, "reason": "runtime_error", "detail": detail[:300]}
