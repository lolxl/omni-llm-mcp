# no-opencc
"""Common request-local account selection. Quota oracles remain provider adapters."""
import math
import os
import re
import time
from providers import health
from providers.errors import contains_key_literal, mask_text
from providers.logutil import log_event
from providers.notify import notify
from providers.profiles import (ACCOUNT_HOME_KEYS, PROFILE_NAME_RE, account_ring,
                                claude_api_labels, resolve_profile, resolve_account_home,
                                _profile_config)

_ACCOUNT_FAILURE_REASONS = frozenset(("rate_limit", "auth_error", "account_unavailable"))
_DETAIL_MAX = 300
API_FALLBACK_VAR = "OMNI_LLM_API_FALLBACK_CLAUDE"
API_BUDGET_VAR = "OMNI_LLM_API_MAX_BUDGET_USD"
CREDIT_COOLDOWN_VAR = "OMNI_LLM_CREDIT_COOLDOWN_S"
_DEFAULT_API_BUDGET = "1.00"
_DEFAULT_CREDIT_COOLDOWN_S = 21600.0  # non-API accounts only
_CREDIT_HOLD_S = 100 * 365 * 24 * 3600.0  # "until cleared": far-future expiry
_CREDIT_WINDOW_RE = re.compile(r"^[0-9]{1,12}$")
_credit_invalid_logged = set()
_BUDGET_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
# Launch-context variables of a parent interactive session; user preferences
# (CLAUDE_CODE_MAX_OUTPUT_TOKENS and friends) are deliberately not listed.
_CLAUDE_LAUNCH_CONTEXT = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SSE_PORT")

def api_labels(provider):
    """Declared API-key labels; only Claude has the concept."""
    return claude_api_labels() if provider == "claude" else (frozenset(), None)

def account_kind(provider, label):
    """``api`` only for a declared Claude label; everything else is a subscription."""
    labels, err = api_labels(provider)
    return "api" if not err and label in labels else "subscription"

def api_budget():
    """Positive decimal spend cap for API-key children, or a LOUD profile_config."""
    raw = os.environ.get(API_BUDGET_VAR)
    value = _DEFAULT_API_BUDGET if raw is None else raw.strip()
    if not _BUDGET_RE.fullmatch(value) or float(value) <= 0 or not math.isfinite(float(value)):
        return None, _profile_config(API_BUDGET_VAR, "need a positive decimal such as 1.00")
    return value, None

def api_disabled_refusal(label):
    """Typed refusal for any use of an API label without the explicit opt-in."""
    return {"ok": False, "reason": "api_disabled",
            "detail": "API account %s needs %s=1" % (label, API_FALLBACK_VAR),
            "account_retry_safe": False}

def api_child_policy(provider, label):
    """Return ``(kind, budget, err)`` for one Claude child; budget only for api."""
    if provider != "claude":
        return "subscription", None, None
    labels, err = api_labels(provider)
    if err:
        return "subscription", None, err
    if label not in labels:
        return "subscription", None, None
    if os.environ.get(API_FALLBACK_VAR) != "1":
        return "api", None, api_disabled_refusal(label)
    budget, err = api_budget()
    return "api", budget, err

_HELPER_MAX_BYTES = 65536
_HELPER_COMMAND_RE = re.compile(r"[A-Za-z0-9_./~][A-Za-z0-9_./~+:@%=,-]*(?:\s|$)")

def _helper_acceptable(helper):
    """A one-line command string that starts with a command word and carries no key."""
    return (isinstance(helper, str) and helper.strip() and helper == helper.strip()
            and "\n" not in helper and "\r" not in helper and "\0" not in helper
            and _HELPER_COMMAND_RE.match(helper) is not None
            and not contains_key_literal(helper))

def read_api_key_helper(home):
    """The ``apiKeyHelper`` command string from ``<home>/settings.json``, or None.

    Only the helper command is read; the key it prints is never touched. The
    file must be a regular file owned by the current user, opened without
    following a final symlink, directly inside the canonical selected home.
    """
    import json
    import stat
    if not isinstance(home, str) or not home:
        return None
    try:
        real_home = os.path.realpath(home)
        if real_home != os.path.normpath(home):
            return None
        target = os.path.join(real_home, "settings.json")
        if os.path.dirname(os.path.realpath(target)) != real_home:
            return None
        dir_fd = os.open(real_home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open("settings.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return None
            raw = os.read(fd, _HELPER_MAX_BYTES + 1)
        finally:
            os.close(fd)
        if len(raw) > _HELPER_MAX_BYTES:
            return None
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    helper = data.get("apiKeyHelper") if isinstance(data, dict) else None
    return helper if _helper_acceptable(helper) else None

def write_api_settings(label, home, directory, name="api-settings.json"):
    """Write ``{"apiKeyHelper": ...}`` (0600) into *directory*; return ``(path, err)``."""
    import json
    helper = read_api_key_helper(home) if home else None
    if helper is None:
        return None, {"ok": False, "reason": "account_unavailable",
                      "detail": "API account %s has no usable apiKeyHelper setting (missing, unreadable, not a plain one-line command, or it embeds a key-shaped literal)" % label}
    path = os.path.join(directory, name)
    try:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump({"apiKeyHelper": helper}, handle)
    except OSError as exc:
        return None, {"ok": False, "reason": "runtime_error",
                      "detail": "could not write API settings for %s: %s" % (label, exc.__class__.__name__)}
    return path, None

def credit_cooldown_s():
    """Automatic retry window in seconds, or ``None`` for a manual hold.

    Purchased API credits never come back on their own, so the default is a
    hold that only ``omni health clear`` ends. A positive integer in
    ``OMNI_LLM_CREDIT_COOLDOWN_S`` opts into an automatic window (keys backed by
    monthly credits). Any other value fails closed to the hold and is logged once.
    """
    raw = os.environ.get(CREDIT_COOLDOWN_VAR)
    if raw is None:
        return None
    text = raw.strip()
    if _CREDIT_WINDOW_RE.fullmatch(text) and int(text) > 0:
        return int(text)
    if raw not in _credit_invalid_logged:
        _credit_invalid_logged.add(raw)
        log_event("credit_cooldown_invalid", var=CREDIT_COOLDOWN_VAR,
                  detail="need a positive integer; holding until cleared")
    return None

def credit_hint(label):
    """Label-only guidance for a calling agent to relay to the user."""
    return ("API credits for account '%s' are exhausted. To keep using it, add credits in the "
            "Claude Console (Settings > Billing) - purchased credits are charged - then run "
            "`omni health clear claude --profile %s`. Until then omni skips this account."
            % (label, label))

def managed_failure_detail(selection, account, result):
    """Generic prefix plus the vendor cause, with account homes and keys redacted."""
    reason = result.get("reason") or "runtime_error"
    text = "%s account failed: %s" % (selection.provider, reason)
    inner = result.get("detail")
    if not isinstance(inner, str) or not inner.strip():
        return text
    sibling_homes = [profile_home(selection.provider, label, selection.homes.get(label))[0]
                     for label in selection.names]
    homes = set()
    for value in [account.get(selection.home_key), *selection.homes.values(), *sibling_homes]:
        if isinstance(value, str) and value:
            homes.update((value, os.path.realpath(value)))
    ordered = sorted(homes, key=len, reverse=True)
    for home in ordered:
        inner = re.sub(re.escape(home), "<account home>", inner, flags=re.IGNORECASE)
    inner = mask_text(inner)
    inner = _mask_cut_tail(inner, ordered).strip()
    # Everything above ran on the full text; the cut comes last, so it can only
    # shorten what is already masked.
    return (text + ": " + inner)[:_DETAIL_MAX] if inner else text

def _mask_cut_tail(inner, homes):
    """Mask a home path or email that an earlier cut left half-written at the end.

    A provider cuts its vendor text before this function sees it; a path or
    address straddling that cut no longer matches its full pattern.
    """
    # Only the tail can hold a cut address; scanning a bounded slice keeps
    # this linear whatever the length of the detail.
    keep, tail = inner[:-128], inner[-128:]
    tail = re.sub(r"(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{0,64}$", "<email>", tail)
    inner = keep + tail
    low = inner.lower()
    for home in homes:
        head = home.lower()
        for n in range(len(head) - 1, 3, -1):
            if low.endswith(head[:n]):
                return inner[:len(inner) - n] + "<account home>"
    return inner

def profile_home(provider, profile, discovered_home=None):
    overrides, err = (resolve_account_home(provider, profile, discovered_home)
                      if discovered_home else resolve_profile(provider, profile))
    if (err and os.environ.get("OMNI_LLM_PROFILE_DISCOVERY") == "1"
            and isinstance(profile, str) and PROFILE_NAME_RE.fullmatch(profile)
            and ACCOUNT_HOME_KEYS[provider] + "_" + profile.upper() not in os.environ):
        from providers.profile_check import _discovered_path, _shape_discovered
        path = _discovered_path(provider, profile)
        if path and not _shape_discovered(provider, profile, path):
            overrides, err = resolve_account_home(provider, profile, path)
    if err:
        return None, err
    from providers.kernel_sandbox import protected_home
    key = ACCOUNT_HOME_KEYS[provider]
    return os.path.realpath(overrides.get(key) or os.environ.get(key) or
                            os.path.join(protected_home(), "." + provider)), None

def home_matches(provider, profile, pinned):
    current, err = profile_home(provider, profile)
    return not err and bool(pinned) and current == pinned

def fence_home(provider, home):
    from providers.kernel_sandbox import protected_home
    default = os.path.realpath(os.path.join(protected_home(), "." + provider))
    return None if home == default else home

def home_explicit(provider, profile):
    """A concrete routing pin does not imply an exported native-home override."""
    return profile not in (None, "", "default") or ACCOUNT_HOME_KEYS[provider] in os.environ

def home_provenance_error(provider, profile, home, explicit):
    """Unset native config can only address the canonical default login."""
    if not isinstance(explicit, bool):
        return {"ok": False, "reason": "account_home_drift",
                "detail": "invalid account home provenance"}
    if not explicit and (profile not in (None, "", "default")
                         or fence_home(provider, home) is not None):
        return {"ok": False, "reason": "account_home_drift",
                "detail": "unset account home requires native default profile and home"}
    return None

def ring_pinned(provider, profile, resume=False):
    """An explicit or resumed profile under a configured ring is auth-isolated."""
    explicit = bool(resume) or profile not in (None, "", "default")
    return explicit and os.environ.get("OMNI_LLM_PROFILES_" + provider.upper()) is not None

def apply_home_env(env, provider, home, explicit=True):
    """Restore persisted home provenance after host environment rewriting."""
    key = ACCOUNT_HOME_KEYS[provider]
    if explicit:
        env[key] = home
    else:
        env.pop(key, None)

def account_env(env, provider):
    """Managed login isolation; apply only to a child copy."""
    env = dict(env)
    keys = {
        "codex": ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL"),
        "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                   "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                   "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"),
        "grok": ("XAI_API_KEY", "GROK_API_KEY", "XAI_BASE_URL", "XAI_API_BASE",
                 "GROK_BASE_URL", "GROK_API_BASE", "GROK_CLI_CHAT_PROXY_BASE_URL",
                 "GROK_WS_URL", "GROK_AUTH_PROVIDER_COMMAND", "GROK_MODELS_BASE_URL",
                 "GROK_MODELS_LIST_URL", "GROK_XAI_API_BASE_URL", "GROK_OIDC_ISSUER",
                 "GROK_OIDC_CLIENT_ID", "GROK_OIDC_AUDIENCE", "GROK_OIDC_SCOPES",
                 "GROK_OAUTH2_ISSUER", "GROK_OAUTH2_CLIENT_ID", "GROK_OAUTH2_PRINCIPAL_ID",
                 "GROK_OAUTH2_PRINCIPAL_TYPE", "GROK_OAUTH2_REFERRER", "GROK_OAUTH2_SCOPES",
                 "GROK_WS_ORIGIN"),
    }
    denied = tuple(key for names in keys.values() for key in names)
    for key in list(env):
        if key in denied or any(key.startswith(name + "_") for name in denied):
            env.pop(key, None)
    # The parent session's launch context never reaches an isolated child of any
    # provider; a child must not inherit the parent's session identity.
    for key in _CLAUDE_LAUNCH_CONTEXT:
        env.pop(key, None)
    return env

def process_failure(result, provider):
    """Process status precedes stdout evidence; unknown statuses fail closed."""
    from providers.errors import signal_exit_detail
    rc, signum = result.get("returncode"), result.get("signal")
    if signum is not None:
        if isinstance(signum, int) and not isinstance(signum, bool) and signum > 0:
            return {"ok": False, "reason": "signal_exit",
                    "detail": signal_exit_detail(provider, -signum,
                                                  result.get("stdout"), result.get("stderr")),
                    "account_retry_safe": False}
        return {"ok": False, "reason": "runtime_error",
                "detail": "invalid process signal status", "account_retry_safe": False}
    if not isinstance(rc, int) or isinstance(rc, bool):
        return {"ok": False, "reason": "runtime_error",
                "detail": "unknown process exit status", "account_retry_safe": False}
    if rc < 0:
        return {"ok": False, "reason": "signal_exit",
                "detail": signal_exit_detail(provider, rc, result.get("stdout"), result.get("stderr")),
                "account_retry_safe": False}
    return None


_ERROR_SUBTYPES = frozenset(("error", "error_during_execution", "error_max_turns",
                            "error_max_budget_usd", "error_max_structured_output_retries"))

def _runtime_frames(stdout):
    """Only outer vendor envelopes are evidence; embedded answer JSON is not."""
    import json
    text = str(stdout or "").strip()
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        objects = []
        for line in text.splitlines():
            try:
                obj = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(obj, dict):
                objects.append(obj)
        return objects
    return [obj] if isinstance(obj, dict) else []

def _error_frame(obj):
    kind = obj.get("type")
    if kind not in ("error", "turn.failed", "result"):
        return False
    if "is_error" in obj and obj["is_error"] is not True:
        return False
    if kind != "result" or obj.get("is_error") is True:
        return True
    subtype = obj.get("subtype")
    return isinstance(subtype, str) and subtype in _ERROR_SUBTYPES

def account_error_text(stdout):
    """Exclude success/assistant metadata from legacy nonzero-exit classifiers."""
    import json
    objects = _runtime_frames(stdout)
    if objects:
        return "\n".join(json.dumps(obj) for obj in objects if _error_frame(obj))
    # Existing plain-text vendor error streams remain supported; an unknown
    # or malformed JSON-looking payload never becomes text-only evidence.
    text = str(stdout or "").strip()
    return "" if text.startswith(("{", "[")) else text

def _untyped_error_envelope(obj):
    """A bare ``{"error": {...}}`` body; a typed non-error frame never qualifies."""
    return "type" not in obj and obj.get("is_error") is not False and isinstance(obj.get("error"), dict)

def _claude_error_text(obj, error):
    """Vendor error text of one frame already known to be a genuine error."""
    # A result frame reaching here is an error frame (is_error true or an error
    # subtype); success and assistant frames were already excluded upstream.
    flagged = obj.get("is_error") is True or obj.get("type") == "result"
    texts = [error.get("message"), obj.get("error")]
    if flagged:
        texts.append(obj.get("result"))
    errors = obj.get("errors")
    if isinstance(errors, list):
        texts.extend(errors)
    return " ".join(text for text in texts if isinstance(text, str)).lower()

def structured_failure(provider, stdout):
    """Quota/auth facts require a genuine, uncontradicted error envelope."""
    import json
    from providers.usage import inband_error
    error_seen = False
    for obj in _runtime_frames(stdout):
        if not (_error_frame(obj) or (provider == "claude" and _untyped_error_envelope(obj))):
            continue
        error_seen = True
        error = obj.get("error")
        error = error if isinstance(error, dict) else obj
        if provider == "claude":
            # Only genuine error envelopes reach here; a successful answer that
            # quotes the vendor text is never an error frame.
            said = _claude_error_text(obj, error)
            if "credit balance is too low" in said:
                return "credit_exhausted"
            if "not scoped to a workspace" in said:
                return "account_unavailable"
        code = error.get("code")
        codes = {"rate_limit_exceeded": "rate_limit", "usage_limit_reached": "rate_limit",
                 "quota_exceeded": "rate_limit", "authentication_failed": "auth_error",
                 "unauthorized": "auth_error", "account_unavailable": "account_unavailable"}
        if isinstance(code, str) and code in codes:
            return codes[code]
        status = error.get("api_error_status", error.get("status"))
        if status == 429:
            return "rate_limit"
        if status in (401, 403):
            return "auth_error"
        if provider == "claude" and obj.get("is_error") is True:
            reported = inband_error(provider, json.dumps(obj))
            if reported:
                from providers.errors import classify_cli_error
                classified = classify_cli_error(reported[1], "", 1)
                if classified in _ACCOUNT_FAILURE_REASONS:
                    return classified
    return "runtime_error" if error_seen else None

class AccountSelection:
    """One ordered walk, one total deadline, never retry a candidate twice."""
    def __init__(self, profile="default", model="codex", timeout_s=120,
                 candidates=None, pinned_home=None, resume=False, homes=None,
                 initial_account=None, tried=None, provider="codex"):
        self.provider = provider
        self.home_key = "codex_home" if provider == "codex" else "account_home"
        self.model = model
        self.profile = profile or "default"
        self.tried = list(tried or [])
        self.error = None
        self.pinned_home = pinned_home
        self.resume = bool(resume)
        try:
            seconds = float(timeout_s)
            if not math.isfinite(seconds) or seconds <= 0:
                seconds = 120
        except (TypeError, ValueError):
            seconds = 120
        self.deadline = time.monotonic() + seconds
        self.managed = False
        self.homes = {}
        if resume or self.profile != "default":
            self.names = [self.profile]
            # Persisted jobs can have a concrete non-default selected profile.
            self.managed = candidates is not None and not resume
            if self.managed:
                self.names = list(candidates)
        else:
            names, self.homes, self.error = (self._ring() if candidates is None
                                             else (list(candidates), {}, None))
            self.managed = names is not None
            self.names = list(names or ["default"])
        self.homes.update(homes or {})
        self.api_labels, kind_err = api_labels(provider)
        if kind_err and not self.error:
            self.error = kind_err
        self.api_enabled = provider == "claude" and os.environ.get(API_FALLBACK_VAR) == "1"
        # R3 gate state: subscriptions skipped for cooldown open the API path,
        # any other outcome for one closes it. Replayed from persisted history.
        self._sub_cooled = 0
        self._sub_blocked = False
        for entry in self.tried:
            self._note_kind(entry.get("account_profile"), entry.get("reason"))
        self.initial_account = initial_account
        self._pin_checked = False
        self._index = 0
        self._homes_seen = set()
        self._seen = set()

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def kind_of(self, label):
        return "api" if label in self.api_labels else "subscription"

    def _note_kind(self, label, reason):
        if self.provider != "claude" or self.kind_of(label) == "api":
            return
        if reason == "skipped_cooldown":
            self._sub_cooled += 1
        else:
            self._sub_blocked = True

    def _cooling_states(self, label):
        # Half-open probing is a subscription recovery path; an API account
        # stays out while any covering cooldown is active.
        if self.provider == "codex" or self.kind_of(label) == "api":
            return ("cooling", "probing")
        return ("cooling",)

    def _api_refusal(self, label):
        """Typed reason when automatic selection may not use this API label."""
        if self.provider != "claude" or self.kind_of(label) != "api":
            return None
        if not self.api_enabled:
            # Explicit pins and resumes need the opt-in too.
            return "api_disabled"
        if not self.managed:
            return None
        if self._sub_blocked or self._sub_cooled == 0:
            return "api_not_permitted"
        return None

    def late_api_refusal(self, account):
        """Final gate re-check, run as the last step before the child is created.

        Selection and launch preparation (fence, snapshot) take time; an
        earlier subscription may have recovered meanwhile. The refusal is
        typed and final for this request: no re-selection at this point.
        """
        label = account.get("account_profile") if isinstance(account, dict) else None
        if self._api_gate_stale(label):
            return {"ok": False, "reason": "api_not_permitted",
                    "detail": "earlier subscription account recovered",
                    "account_retry_safe": False}
        return None

    def _api_gate_stale(self, label):
        """A preselected API account whose permitting evidence has expired.

        The persisted gate says every earlier subscription was cooling when the
        job was submitted; that is only a claim until re-read at execution.
        """
        if self.provider != "claude" or not self.managed or self.kind_of(label) != "api":
            return False
        return any(entry.get("reason") == "skipped_cooldown"
                   and self.kind_of(entry.get("account_profile")) != "api"
                   and health.seat_kind(self.model, entry.get("account_profile")) not in self._cooling_states(
                       entry.get("account_profile"))
                   for entry in self.tried)

    def _rewalk(self, label):
        """Restart the ordered walk from the earlier labels, against current health."""
        earlier = [entry.get("account_profile") for entry in self.tried]
        rest = [name for name in self.names if name not in earlier and name != label]
        self.names = earlier + [label] + rest
        self.tried = []
        self._sub_cooled = 0
        self._sub_blocked = False
        self._seen = set()
        self._homes_seen = set()
        self._index = 0
        self.initial_account = None
        self.pinned_home = None
        self._pin_checked = True

    def _with_kind(self, account, label):
        if self.provider == "claude":
            account = dict(account, account_kind=self.kind_of(label))
        return account

    def record(self, label, reason):
        # Caller labels are validated before any diagnostics become public.
        label = label if isinstance(label, str) and PROFILE_NAME_RE.fullmatch(label) else "invalid"
        entry = {"account_profile": label, "reason": reason}
        if self.provider == "claude":
            entry["account_kind"] = self.kind_of(label)
        self.tried.append(entry)
        self._note_kind(label, reason)

    def next(self):
        if self.error:
            return None
        if self.remaining() <= 0:
            self.error = {"ok": False, "reason": "timeout", "detail": "account selection deadline exceeded"}
            return None
        if self.initial_account is not None:
            account, self.initial_account = self.initial_account, None
            label, home = account["account_profile"], account[self.home_key]
            if not self._home_matches(label, home):
                self.error = {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
                return None
            if self._api_gate_stale(label):
                self._rewalk(label)
                return self.next()
            self._pin_checked = True
            self._seen.add(label)
            self._homes_seen.add(home)
            refusal = self._api_refusal(label)
            if refusal:
                return self._refuse(label, refusal)
            if self.managed and health.seat_kind(self.model, label) in self._cooling_states(label):
                self.record(label, "skipped_cooldown")
                return self.next()
            return self._with_kind(account, label)
        while self._index < len(self.names):
            if self.remaining() <= 0:
                self.error = {"ok": False, "reason": "timeout", "detail": "account selection deadline exceeded"}
                return None
            label = self.names[self._index]
            self._index += 1
            if label in self._seen:
                continue
            self._seen.add(label)
            refusal = self._api_refusal(label)
            if refusal:
                if not self.managed:
                    return self._refuse(label, refusal)
                self.record(label, refusal)
                continue
            home, err = self._home(label, self.homes.get(label))
            if err:
                if not self.managed:
                    self.error = err
                    return None
                self.record(label, "account_unavailable")
                continue
            if self.pinned_home and not self._pin_checked and home != self.pinned_home:
                self.error = {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
                return None
            self._pin_checked = True
            if self.managed and home in self._homes_seen:
                self.record(label, "duplicate_account_home")
                continue
            self._homes_seen.add(home)
            if self.managed:
                kind = health.seat_kind(self.model, label)
                if kind in self._cooling_states(label):
                    self.record(label, "skipped_cooldown")
                    continue
                state = self._quota(label, home)
                if self.remaining() <= 0:
                    self.error = {"ok": False, "reason": "timeout", "detail": "account selection deadline exceeded"}
                    return None
                if state != "available":
                    self.record(label, state)
                    continue
            account = {"account_profile": label, self.home_key: home}
            if self.provider != "codex":
                account["account_home_explicit"] = home_explicit(self.provider, label)
            return self._with_kind(account, label)
        return None

    def _refuse(self, label, refusal):
        """Managed walks record and move on; an explicit pin or resume stops here."""
        if self.managed:
            self.record(label, refusal)
            return self.next()
        self.error = (api_disabled_refusal(label) if refusal == "api_disabled"
                      else {"ok": False, "reason": refusal,
                            "detail": "API account %s is not permitted" % label})
        return None

    def exhausted(self):
        if self.error:
            return {**self.error, "account_retry_safe": False}
        detail = "no eligible %s account" % self.provider
        if self.provider == "claude":
            hints = [credit_hint(label) for label in self._credit_held_labels()]
            if hints:
                detail = " ".join([detail + "."] + hints)
        return {"ok": False, "reason": "account_exhausted", "detail": detail,
                "tried": list(self.tried), "account_retry_safe": False}

    def failed(self, account, result, raw_detail=None):
        """Flag the account for the failure and record it. Never permits a re-run:
        the request ends here for every provider and the next request selects
        another account."""
        reason = result.get("reason")
        if reason in _ACCOUNT_FAILURE_REASONS:
            # Label the cooldown with what actually failed; the source only
            # names the cause here, it does not change cooldown or probing.
            flag = {"source": reason if result.get("tool_activity") else "account_startup"}
            if reason == "rate_limit" and self.provider != "codex":
                # The managed result carries the vendor cause as text; reuse the
                # parser the unmanaged dispatch path applies to the same detail.
                text = raw_detail if isinstance(raw_detail, str) else result.get("detail")
                until = health.vendor_reset_from_output(
                    self.provider, text if isinstance(text, str) else "",
                    result.get("stderr") if isinstance(result.get("stderr"), str) else "")
                if until is not None:
                    flag = {"until": until, "source": "vendor_reset"}
            health.flag(self.provider, account["account_profile"], **flag)
        elif reason == "credit_exhausted" and self.provider == "claude":
            self._flag_credit_exhausted(account["account_profile"])
        self.record(account["account_profile"], reason or "runtime_error")

    def _flag_credit_exhausted(self, label):
        """Hold the account out; tell the user once when the hold is newly raised."""
        api = self.kind_of(label) == "api"
        window = credit_cooldown_s()
        if window is None:
            window = _CREDIT_HOLD_S if api else _DEFAULT_CREDIT_COOLDOWN_S
        # "Newly raised" is decided inside the locked ledger mutation, so two
        # overlapping raises cannot both see "no hold yet".
        _entry, newly_raised = health.flag_credit_hold(
            self.provider, label, until=time.time() + window)
        if api and newly_raised:
            notify("api_credit_exhausted", provider="claude", account_profile=label,
                   action="add credits in the Claude Console, then omni health clear "
                          "claude --profile %s" % label)

    def _credit_held_labels(self):
        """API labels this request skipped because of a credit hold."""
        held = health.snapshot()
        labels = []
        for entry in self.tried:
            label = entry.get("account_profile")
            if (entry.get("reason") == "skipped_cooldown" and entry.get("account_kind") == "api"
                    and label not in labels
                    and health.credit_hold(
                        held.get(health.pool_key(self.provider, label))) is not None):
                labels.append(label)
        return labels

    def _ring(self):
        return account_ring(self.provider)

    def _home(self, profile, discovered_home=None):
        return profile_home(self.provider, profile, discovered_home)

    def _home_matches(self, profile, home):
        return home_matches(self.provider, profile, home)

    def _quota(self, label, home):
        # No oracle exists for Claude/Grok. Selection promises only local
        # eligibility; runtime failure is never replayed within this request.
        return "available"


def _credit_envelope(selection, account, result):
    """Swap the vendor text for label-only guidance on an API credit failure."""
    if (selection.provider == "claude" and result.get("reason") == "credit_exhausted"
            and account.get("account_kind") == "api"):
        result["detail"] = credit_hint(account["account_profile"])


async def run_selected(selection, attempt, on_selected=None):
    """One attempt on one account. A failure ends the request; it is never re-run."""
    started = time.time()
    account = selection.next()
    if account is None:
        return selection.exhausted()
    if on_selected:
        on_selected(account, selection)
    if selection.remaining() <= 0:
        return {"ok": False, "reason": "timeout", "detail": "account selection deadline exceeded",
                "account_retry_safe": False}
    result = await attempt(account, selection.remaining(), selection.managed)
    if account.get("account_kind") == "api" and not selection.managed:
        result = dict(result, account_kind="api")
    if result.get("ok") and selection.provider != "codex":
        # A flag raised after this request began is newer evidence than
        # this success and survives it.
        for cleared in (selection.provider, selection.model):
            try:
                health.clear(cleared, account["account_profile"],
                             unless_flagged_after=started,
                             keep_sources=health.SUCCESS_KEEP_SOURCES)
            except health.HealthLockUnavailable:
                pass  # best effort: a success must not fail on the ledger lock
    if not selection.managed:
        if not result.get("ok"):
            result.setdefault("account_retry_safe", False)
            selection.failed(account, result)
            _credit_envelope(selection, account, result)
        return result
    result = dict(result)
    raw_detail = result.get("detail")
    if not result.get("ok"):
        # A managed failure is final for this request, so no later seat in a
        # fallback chain may re-run it on another account either.
        result["account_retry_safe"] = False
        result["detail"] = managed_failure_detail(selection, account, result)
        _credit_envelope(selection, account, result)
    result["account_profile"] = account["account_profile"]
    if selection.provider == "claude":
        result["account_kind"] = account.get("account_kind", "subscription")
    if selection.provider != "codex":
        result["quota_verified"] = False
    if result.get("ok"):
        if selection.tried:
            result["tried"] = list(selection.tried)
        return result
    selection.failed(account, result, raw_detail)
    result["tried"] = list(selection.tried)
    return result
