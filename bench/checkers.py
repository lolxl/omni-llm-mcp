# no-opencc
"""Deterministic path checkers for the hermetic baseline.

Each checker asserts a verifiable property of a path envelope. test_bench.py
proves every checker rejects a deliberately wrong output (falsifier).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class CheckerError(AssertionError):
    """Raised when a path checker rejects the envelope."""


def check_consultation(envelope: Dict[str, Any], fixture: Dict[str, Any]) -> None:
    if not envelope.get("ok"):
        raise CheckerError("consultation: expected ok=true, got %r" % envelope.get("ok"))
    expected = fixture.get("expected_text")
    text = envelope.get("text")
    if text != expected:
        raise CheckerError(
            "consultation: expected text %r, got %r" % (expected, text)
        )


def check_big_context(envelope: Dict[str, Any], fixture: Dict[str, Any]) -> None:
    if not envelope.get("ok"):
        raise CheckerError("big_context: expected ok=true, got %r" % envelope.get("ok"))
    expected = fixture.get("expected_text")
    text = envelope.get("text")
    if text != expected:
        raise CheckerError(
            "big_context: expected text %r, got %r" % (expected, text)
        )


def check_fallback_chain(envelope: Dict[str, Any], fixture: Dict[str, Any]) -> None:
    if not envelope.get("ok"):
        raise CheckerError("fallback_chain: expected ok=true, got %r" % envelope.get("ok"))
    expected = fixture.get("expected_text")
    text = envelope.get("text")
    if text != expected:
        raise CheckerError(
            "fallback_chain: expected text %r, got %r" % (expected, text)
        )
    # Breadcrumb of the failed first seat must be present (dispatch contract).
    tried = envelope.get("tried")
    if not isinstance(tried, list) or not tried:
        raise CheckerError("fallback_chain: expected non-empty tried breadcrumb")
    first = tried[0]
    if not isinstance(first, dict) or first.get("model") != "bench/fail":
        raise CheckerError(
            "fallback_chain: expected first tried model bench/fail, got %r" % first
        )


def check_large_result(envelope: Dict[str, Any], fixture: Dict[str, Any]) -> None:
    if not envelope.get("ok"):
        raise CheckerError("large_result: expected ok=true, got %r" % envelope.get("ok"))
    text = envelope.get("text")
    if not isinstance(text, str):
        raise CheckerError("large_result: text must be str, got %r" % type(text).__name__)
    marker = fixture.get("expected_marker")
    if not marker or not text.endswith(marker):
        raise CheckerError(
            "large_result: text must end with marker %r" % marker
        )
    min_chars = int(fixture.get("expected_min_chars") or 0)
    if len(text) < min_chars:
        raise CheckerError(
            "large_result: expected >= %d chars, got %d" % (min_chars, len(text))
        )


CHECKERS = {
    "consultation": check_consultation,
    "big_context": check_big_context,
    "fallback_chain": check_fallback_chain,
    "large_result": check_large_result,
}


def run_checker(path: str, envelope: Dict[str, Any], fixture: Dict[str, Any]) -> str:
    """Return 'pass' or raise CheckerError."""
    fn = CHECKERS.get(path)
    if fn is None:
        raise CheckerError("unknown path %r" % path)
    fn(envelope, fixture)
    return "pass"


def falsifier_envelope(path: str, fixture: Dict[str, Any]) -> Dict[str, Any]:
    """A deliberately wrong but well-formed envelope for falsifier tests."""
    if path == "consultation":
        return {"ok": True, "text": "CONSULT_OK:WRONG", "model": "bench/consult"}
    if path == "big_context":
        return {"ok": True, "text": "BIG_OK:wrongtoken", "model": "bench/big"}
    if path == "fallback_chain":
        # ok + right text but missing tried breadcrumb
        return {"ok": True, "text": fixture.get("expected_text"), "model": "bench/ok"}
    if path == "large_result":
        return {
            "ok": True,
            "text": "tiny" + str(fixture.get("expected_marker") or ""),
            "model": "bench/large",
        }
    raise KeyError(path)
