# no-opencc
"""Request-local Codex account selection and conservative startup-failure classification.

Only labels and typed reasons leave this module. Homes are internal pins, never
public diagnostics. This module does not inspect credentials or copy state.
"""
import json
import math
import os
import time

from providers import health, provider_accounts
from providers.profiles import codex_account_ring, resolve_profile, resolve_codex_home, PROFILE_NAME_RE

_FAILURE_CODES = {
    "rate_limit_exceeded": "rate_limit", "usage_limit_reached": "rate_limit",
    "quota_exceeded": "rate_limit", "authentication_failed": "auth_error",
    "unauthorized": "auth_error", "account_unavailable": "account_unavailable",
}
_STARTUP_REASONS = frozenset(_FAILURE_CODES.values())
_SAFE_EVENTS = frozenset(("thread.started", "turn.started", "turn.failed", "error"))


def profile_home(profile, discovered_home=None):
    overrides, err = (resolve_codex_home(profile, discovered_home)
                      if discovered_home else resolve_profile("codex", profile))
    if err and os.environ.get("OMNI_LLM_PROFILES_CODEX", "").strip() == "auto":
        _names, homes, ring_err = codex_account_ring()
        discovered = homes.get(profile)
        if discovered and not ring_err:
            overrides, err = resolve_codex_home(profile, discovered)
    if err:
        return None, err
    raw = overrides.get("CODEX_HOME") or os.environ.get("CODEX_HOME")
    from providers.kernel_sandbox import protected_home
    return os.path.realpath(raw or os.path.join(protected_home(), ".codex")), None


def home_matches(profile, pinned):
    current, err = profile_home(profile)
    if err and os.environ.get("OMNI_LLM_PROFILES_CODEX", "").strip() == "auto":
        _names, homes, ring_err = codex_account_ring()
        current = homes.get(profile)
        err = ring_err or (None if current else {"reason": "unsupported_profile"})
    return not err and bool(pinned) and current == pinned


def fence_home(home):
    """Default state keeps its existing fence admission; named homes do not."""
    from providers.kernel_sandbox import protected_home
    default = os.path.realpath(os.path.join(protected_home(), ".codex"))
    return None if home == default else home


def account_env(env):
    """A managed subscription account must not inherit API-key authentication."""
    return provider_accounts.account_env(env, "codex")


def _unique_json_object(pairs):
    event = {}
    for name, value in pairs:
        if name in event:
            raise ValueError("duplicate JSON object name")
        event[name] = value
    return event


def _reject_json_constant(value):
    raise ValueError("non-standard JSON constant")


def startup_evidence(stdout):
    """Return (startup reason, tool activity); absent/unknown evidence fails closed.

    A startup reason needs a complete known-event stream and a structured
    startup error code or a definite message within an error-typed event. Any
    item event denotes possible activity, including unknown future tools.
    Text-only stderr classification never yields one. The reason names the
    failure and cools the account; it never permits running the request again
    on another account (the stream is the child's own output, so it is not
    proof that nothing ran).
    """
    reason = None
    valid = bool(str(stdout or "").strip())
    tool_activity = False
    for line in str(stdout or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line, object_pairs_hook=_unique_json_object,
                               parse_constant=_reject_json_constant)
        except (TypeError, ValueError, RecursionError):
            valid = False
            continue
        if not isinstance(event, dict):
            valid = False
            continue
        kind = event.get("type")
        if not isinstance(kind, str):
            valid = False
            continue
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in (
                    "agent_message", "reasoning"):
                tool_activity = True
            valid = False  # a startup proof has no model item activity
        elif kind not in _SAFE_EVENTS:
            valid = False
        if kind in ("error", "turn.failed"):
            error = event.get("error")
            if not isinstance(error, dict):
                error = event
            code = error.get("code")
            matched = _FAILURE_CODES.get(code) if isinstance(code, str) else None
            if not matched:
                from providers.errors import codex_error_text, classify_cli_error
                from providers.config import CODEX_EXTRA_AUTH_MARKERS
                classified = classify_cli_error(
                    codex_error_text(json.dumps(event)), "", 1,
                    extra_auth_markers=CODEX_EXTRA_AUTH_MARKERS)
                if classified in _STARTUP_REASONS:
                    matched = classified
            if matched:
                if reason is not None and reason != matched:
                    valid = False
                reason = matched
            else:
                valid = False
    return (reason if valid and not tool_activity else None), tool_activity


def annotate_failure(result, stdout):
    """Attach only bounded facts, preserving existing provider envelopes.

    The stream is the child's own output, so it names the failure reason but
    never permits another run: every Codex failure carries
    ``account_retry_safe: false`` and ends the request. No same-account retry,
    no other account, no later seat; the next request selects an account.
    """
    result = dict(result)
    safe_reason, tools = startup_evidence(stdout)
    if safe_reason and result.get("reason") in ("runtime_error", safe_reason):
        result["reason"] = safe_reason
    result["account_retry_safe"] = False
    result["tool_activity"] = tools
    return result


def _quota_state(info):
    if not isinstance(info, dict) or info.get("quota_unknown"):
        return "quota_unknown"
    windows = [info]
    if isinstance(info.get("secondary"), dict):
        windows.append(info["secondary"])
    if info.get("reached") or any(w.get("reached") for w in windows):
        return "quota_exhausted"
    for window in windows:
        used = window.get("used_percent")
        if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used):
            return "quota_unknown"
        if used >= 100:
            return "quota_exhausted"
        if used < 0:
            return "quota_unknown"
    return "available"


class AccountSelection(provider_accounts.AccountSelection):
    """Codex adapter: existing RPC quota and startup-failure contract."""
    def _ring(self):
        return provider_accounts.account_ring("codex")

    def _home(self, profile, discovered_home=None):
        return profile_home(profile, discovered_home)

    def _home_matches(self, profile, home):
        return home_matches(profile, home)

    def _quota(self, label, home):
        from providers.codex_quota import read_rate_limits, ENV_QUOTA_PROBE
        if os.environ.get(ENV_QUOTA_PROBE) == "0":
            return "available"
        remaining = self.remaining()
        if remaining <= 0:
            return "quota_unknown"
        info, quota_err = read_rate_limits(label, timeout_s=min(15, remaining), codex_home=home)
        state = _quota_state(info) if not quota_err else (
            quota_err.get("reason") if isinstance(quota_err, dict) and
            quota_err.get("reason") in ("auth_error", "account_unavailable", "cli_missing")
            else "quota_exhausted" if isinstance(quota_err, dict) and quota_err.get("reason") == "rate_limit"
            else "quota_unknown")
        if not quota_err:
            health.apply_codex_quota("codex", label, info, None)
        if state == "available" and health.seat_kind(self.model, label) in ("cooling", "probing"):
            state = "skipped_cooldown"
        return state


run_selected = provider_accounts.run_selected
