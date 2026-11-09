# no-opencc
"""Fixture-driven fake provider backend for hermetic path runs.

Patches providers.dispatch.route_to_provider (the same seam used by the hermetic
suite in test_health.py / test_server.py) so no network and no real CLIs run.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


def materialize_reply(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a fixture fake_replies entry into a provider envelope.

    Usage is normalized via providers.usage.normalize_usage (import, don't
    reimplement). Fixtures may supply either:
      * usage_raw + optional usage_backend (default "grok") — preferred
      * usage — already-normalized envelope (pass-through for tests)
    A seat with neither yields no usage key (honest absence).
    """
    if "text" in spec:
        out = {
            "ok": bool(spec.get("ok", True)),
            "text": spec["text"],
            "model": spec.get("model"),
            **({"reason": spec["reason"]} if "reason" in spec else {}),
            **({"detail": spec["detail"]} if "detail" in spec else {}),
        }
    else:
        # large_result style: synthesize a long body from compact fixture fields
        prefix = str(spec.get("text_prefix", ""))
        unit = str(spec.get("repeat_unit", "x"))
        count = int(spec.get("repeat_count", 0))
        suffix = str(spec.get("text_suffix", ""))
        text = prefix + (unit * count) + suffix
        out = {
            "ok": bool(spec.get("ok", True)),
            "text": text,
            "model": spec.get("model"),
        }

    usage = _materialize_usage(spec)
    if usage is not None:
        out["usage"] = usage
    return out


def _materialize_usage(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "usage" in spec and isinstance(spec["usage"], dict):
        # Already-normalized (or test-injected) envelope — shallow copy.
        return copy.deepcopy(spec["usage"])
    if "usage_raw" not in spec:
        return None
    # Import normalize_usage — do not reimplement provider field mapping.
    from providers.usage import normalize_usage  # noqa: WPS433

    backend = str(spec.get("usage_backend") or "grok")
    return normalize_usage(backend, spec["usage_raw"])


class CountingFakeBackend:
    """Async route_to_provider stand-in that counts real seat attempts."""

    def __init__(self, replies_by_model: Dict[str, Dict[str, Any]]):
        self.replies_by_model = {
            m: materialize_reply(copy.deepcopy(spec))
            for m, spec in replies_by_model.items()
        }
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, prompt, model, timeout_s, files, account_profile):
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "timeout_s": timeout_s,
            "files": files,
            "account_profile": account_profile,
        })
        if model not in self.replies_by_model:
            return {
                "ok": False,
                "reason": "runtime_error",
                "detail": "no fake reply for model %r" % model,
                "model": model,
            }
        # Return a fresh dict so callers can mutate without poisoning the fixture.
        return copy.deepcopy(self.replies_by_model[model])

    @property
    def attempts(self) -> int:
        """Real seat calls (cooldown skips never hit this fake)."""
        return len(self.calls)

    @property
    def fallback_count(self) -> int:
        """Seats called after the first real call (P1.2a pre-instrumentation def)."""
        return max(0, self.attempts - 1)


def backend_from_fixture(fixture: Dict[str, Any]) -> CountingFakeBackend:
    replies = fixture.get("fake_replies") or {}
    if not isinstance(replies, dict) or not replies:
        raise ValueError("fixture missing fake_replies")
    return CountingFakeBackend(replies)
