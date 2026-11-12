"""Local, attributable model catalogs and metadata-only refresh adapters.

Catalog state is scoped to a provider profile and never changes model defaults.
Refresh writes a v2 record. Status, check, and admit read that record.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _datetime
import fcntl
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from providers import vendor_bin


PROVIDERS = ("cursor", "agy", "codex", "claude", "grok")
ROAD_PREFIX = {name: name + "/" for name in PROVIDERS}
EFFORTS = frozenset(("minimal", "low", "medium", "high", "xhigh", "max", "ultra"))
CURSOR_EFFORTS = frozenset(("minimal", "low", "medium", "high", "xhigh", "max"))
AGY_EFFORTS = frozenset(("low", "medium", "high", "xhigh", "max"))
DEFAULT_STALE_AFTER_S = 48 * 60 * 60
MAX_STDOUT_BYTES = 1024 * 1024
MAX_MODEL_COUNT = 5000
MAX_CODEX_PAGES = 100
CODEX_RPC_METHODS = frozenset(("initialize", "initialized", "model/list"))
CODEX_SAFE_NOTIFICATIONS = frozenset(("remoteControl/status/changed",))
_POSTURE_SEATBELT = "seatbelt-fs"
_POSTURE_UNFENCED = "metadata-unfenced"
_DISCOVERY_POSTURES = (_POSTURE_SEATBELT, _POSTURE_UNFENCED)
# Off Darwin, refresh discovery may spawn only these argv tails, and only when
# argv[0] realpath-matches the binary that provider's discoverer looks up.
# Claude's full argv is _claude_discovery_argv. Any other argv is refused
# before a child starts.
_METADATA_ARGV_TAILS = {
    "cursor": ("--list-models",),
    "agy": ("models",),
    "grok": ("models",),
    "codex": ("app-server",),
}
_ACTIVE_PROFILE = contextvars.ContextVar("omni_catalog_profile", default="default")
# Raw vendor output from the fetch that the current refresh is parsing.
# A patched discoverer leaves this unset, so refresh validates that return
# value instead of reparsing vendor output.
_FETCHED_OUTPUT = contextvars.ContextVar("omni_catalog_fetched_output", default=None)
_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_EFFORT_SUFFIXES = (
    ("extra-high", "xhigh"), ("minimal", "minimal"), ("medium", "medium"),
    ("xhigh", "xhigh"), ("high", "high"), ("low", "low"), ("max", "max"),
)


class CatalogError(Exception):
    """A bounded catalog failure with a stable reason and no vendor output."""

    def __init__(self, reason, provider=None):
        self.reason = str(reason)
        self.provider = provider
        super().__init__(self.reason)

    def envelope(self):
        value = {"ok": False, "reason": self.reason,
                 "detail": "model catalog %s" % self.reason}
        if self.provider:
            value["provider"] = self.provider
        return value


def _utc_now():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_datetime(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        return None
    return parsed.astimezone(_datetime.timezone.utc)


def _profile_name(profile):
    value = "default" if profile is None or str(profile).strip() in ("", "default") else str(profile)
    if not re.fullmatch(r"[a-z0-9_]+", value):
        raise CatalogError("unsupported_profile")
    return value


def profile_scope(profile):
    """Set the profile used by legacy Cursor resolution call sites."""
    return _ACTIVE_PROFILE.set(_profile_name(profile))


@contextlib.contextmanager
def using_profile(profile):
    token = profile_scope(profile)
    try:
        yield
    finally:
        _ACTIVE_PROFILE.reset(token)


def active_profile():
    return _ACTIVE_PROFILE.get()


def _state_dir():
    raw = os.environ.get("OMNI_LLM_STATE_DIR") or "~/.local/state/omni-llm"
    return Path(os.path.abspath(os.path.expanduser(raw)))


def _catalog_root(create=False):
    root = _state_dir() / "model-catalogs"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise CatalogError("cache_path_invalid")
    return root


def _safe_endpoint(raw):
    if not raw:
        return ""
    try:
        parts = urlsplit(str(raw).strip())
        if not parts.hostname:
            return ""
        host = parts.hostname.lower()
        if parts.port:
            host += ":%d" % parts.port
        path = parts.path.rstrip("/")
        return "%s://%s%s" % (parts.scheme.lower() or "https", host, path)
    except (TypeError, ValueError):
        return ""


_ENDPOINT_ENV_KEYS = {
    "cursor": ("CURSOR_API_BASE", "CURSOR_BASE_URL"),
    "agy": ("AGY_API_BASE", "AGY_ENDPOINT"),
    "codex": ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "claude": ("ANTHROPIC_BASE_URL",),
    "grok": ("XAI_BASE_URL", "XAI_API_BASE", "GROK_BASE_URL", "GROK_API_BASE",
             "GROK_CLI_CHAT_PROXY_BASE_URL", "GROK_WS_URL", "GROK_MODELS_BASE_URL",
             "GROK_MODELS_LIST_URL", "GROK_XAI_API_BASE_URL", "GROK_OIDC_ISSUER",
             "GROK_OAUTH2_ISSUER", "GROK_WS_ORIGIN"),
}
_HOME_ENV_KEYS = {"cursor": "CURSOR_CONFIG_DIR", "agy": "AGY_CONFIG_DIR",
                  "codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR",
                  "grok": "GROK_HOME"}


def _catalog_child_environment(provider, profile, environ=None):
    """Mirror the metadata child's selected, account-scrubbed allowlisted env."""
    from providers.profiles import resolve_profile, spawn_env
    from providers.host_fence import allowlisted_child_env

    name = _profile_name(profile)
    source = dict(os.environ if environ is None else environ)
    overrides, error = resolve_profile(provider, name)
    if error:
        raise CatalogError(error.get("reason") or "unsupported_profile", provider)
    child_env = spawn_env(source, overrides, scrub_siblings=True)
    if provider == "codex":
        from providers import codex_accounts
        child_env = codex_accounts.account_env(child_env)
    elif provider == "claude":
        from providers.claude import scrub_read_only_env
        scrub_read_only_env(child_env)
    elif provider == "agy" and name != "default":
        selected = source.get("AGY_CONFIG_DIR_%s" % name.upper())
        if not selected or not os.path.isdir(os.path.expanduser(selected)):
            raise CatalogError("unsupported_profile", provider)
        child_env["AGY_CONFIG_DIR"] = selected
    selected_home = None
    if provider == "cursor":
        from providers.config import CURSOR_CONFIG_DIR_ENV
        from providers.cursor import resolve_config_dir
        selected_home = resolve_config_dir(child_env)
        if selected_home:
            child_env[CURSOR_CONFIG_DIR_ENV] = selected_home
        else:
            # A metadata run uses a throwaway directory, not the inherited vendor key.
            child_env.pop(CURSOR_CONFIG_DIR_ENV, None)
    allowed_env = allowlisted_child_env(child_env, provider)
    endpoint_bindings = _validate_endpoint_bindings(provider, allowed_env, source)
    return overrides, child_env, allowed_env, selected_home, endpoint_bindings


def _validate_endpoint_bindings(provider, child_env, source_env):
    """Reject credential-bearing endpoint values before they enter identity hashes."""
    from providers import known_secrets

    known_secrets.register_env(source_env)
    known_secrets.register_env(child_env)
    result = {}
    for key in _ENDPOINT_ENV_KEYS[provider]:
        value = child_env.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            parts = urlsplit(value)
        except ValueError:
            raise CatalogError("endpoint_invalid", provider)
        sensitive_query = any(re.search(
            r"(?:api[-_]?key|access[-_]?token|token|secret|password|passwd|auth|credential|session)",
            query_key, re.IGNORECASE)
            for query_key, _query_value in parse_qsl(parts.query, keep_blank_values=True))
        if (parts.username is not None or parts.password is not None
                or sensitive_query or parts.fragment
                or known_secrets.mask_known(value) != value):
            raise CatalogError("endpoint_secret_refused", provider)
        result[key] = value
    return result


def _catalog_endpoint_values(bindings):
    """Fingerprint actual operational endpoint values without retaining them."""
    return {key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in bindings.items()}


def _profile_identity_inputs(provider, profile):
    name = _profile_name(profile)
    overrides, child_env, allowed_env, selected_home, endpoint_bindings = (
        _catalog_child_environment(provider, name))
    home = selected_home if provider == "cursor" else allowed_env.get(_HOME_ENV_KEYS[provider])
    home_explicit = bool(home)
    config_home = os.path.realpath(os.path.expanduser(str(home))) if home else ""
    if provider == "agy":
        agy_key = "AGY_CONFIG_DIR" if name == "default" else "AGY_CONFIG_DIR_%s" % name.upper()
        agy_home = os.environ.get(agy_key)
        if name != "default" and (not agy_home or not os.path.isdir(os.path.expanduser(agy_home))):
            raise CatalogError("unsupported_profile", provider)
        if agy_home:
            config_home = os.path.realpath(os.path.expanduser(agy_home))
            home_explicit = True
    if not config_home and name == "default" and provider in ("claude", "codex", "grok"):
        if provider == "codex":
            from providers.codex_accounts import profile_home
            native_home, error = profile_home("default")
        else:
            from providers.provider_accounts import profile_home
            native_home, error = profile_home(provider, "default")
        if error or not native_home:
            raise CatalogError((error or {}).get("reason") or "unsupported_profile", provider)
        config_home = os.path.realpath(native_home)
    suffix = "" if name == "default" else "_" + name.upper()
    catalog_key = "OMNI_LLM_CATALOG_ENDPOINT_%s" % provider.upper()
    catalog_endpoint = _safe_endpoint(
        os.environ.get(catalog_key + suffix) or
        (os.environ.get(catalog_key) if not suffix else None))
    catalog_raw = (os.environ.get(catalog_key + suffix) or
                   (os.environ.get(catalog_key) if not suffix else None))
    if catalog_raw:
        from providers import known_secrets
        if known_secrets.mask_known(catalog_raw) != catalog_raw:
            raise CatalogError("endpoint_secret_refused", provider)
    return {"profile": name, "config_home": config_home,
            "home_explicit": home_explicit,
            "endpoints": _catalog_endpoint_values(endpoint_bindings),
            "catalog_endpoint": catalog_endpoint}, allowed_env


def _identity_key(root, create=False):
    path = root / ".identity-key"
    if path.is_symlink():
        raise CatalogError("cache_path_invalid")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = None
    except OSError:
        raise CatalogError("cache_identity_invalid")
    if data is None:
        try:
            prefixes = tuple("%s-" % provider for provider in PROVIDERS)
            cached_state_exists = any(
                entry.name.startswith(prefixes) and entry.name.endswith(".json")
                for entry in root.iterdir())
        except FileNotFoundError:
            cached_state_exists = False
        except OSError:
            raise CatalogError("cache_identity_invalid")
        if cached_state_exists:
            raise CatalogError("cache_identity_invalid")
        if not create:
            raise CatalogError("cache_identity_unavailable")
        candidate = secrets.token_bytes(32)
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                data = path.read_bytes()
            except OSError:
                raise CatalogError("cache_identity_invalid")
        except OSError:
            raise CatalogError("cache_identity_unavailable")
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(candidate)
                stream.flush()
                os.fsync(stream.fileno())
            data = candidate
    if len(data) != 32:
        raise CatalogError("cache_identity_invalid")
    return data


def _scope_fingerprint(provider, profile, create=False):
    root = _catalog_root(create=create)
    identity, _overrides = _profile_identity_inputs(provider, profile)
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_identity_key(root, create=create), payload, hashlib.sha256).hexdigest()[:32]

def _atomic_write(path, data, mode=0o600):
    parent = path.parent
    if path.is_symlink() or parent.is_symlink() or not parent.is_dir():
        raise CatalogError("cache_path_invalid")
    fd, temp_name = tempfile.mkstemp(prefix=".%s-" % path.name, dir=str(parent))
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        try:
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def _provider_lock(path):
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _json_loads(data):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=reject_duplicates)

def status(provider="all", profile="default", stale_after_s=DEFAULT_STALE_AFTER_S,
           details=False):
    """Read v2 records only. A missing record is a row with null freshness."""
    try:
        selected = PROVIDERS if provider in (None, "all") else (provider,)
        if any(item not in PROVIDERS for item in selected):
            raise CatalogError("unknown_provider")
        from providers import catalog_v2
        rows = []
        now = _datetime.datetime.now(_datetime.timezone.utc)
        for name in selected:
            record, load_error = catalog_v2.load(name, profile)
            observed = _utc_datetime(record.get("observed_at")) if record else None
            age = max(0, int((now - observed).total_seconds())) if observed else None
            entry = {
                "provider": name,
                "observed_at": record.get("observed_at") if record else None,
                "age_seconds": age,
                "stale": age > int(stale_after_s) if age is not None else None,
                "model_count": len(record.get("models", [])) if record else 0,
            }
            if load_error:
                entry["reason"] = load_error
            if details:
                entry["models"] = [
                    {"id": row.get("id"), "name": row.get("name")}
                    for row in (record.get("models", []) if record else [])
                ]
            rows.append(entry)
        return {"ok": True, "profile": _profile_name(profile),
                "details": bool(details), "catalogs": rows}
    except CatalogError as exc:
        return exc.envelope()


def _shipped_defaults(provider):
    """Models ``models check`` admits for one provider. Grok also admits the image id."""
    from providers import config
    from providers.grok_image_cli import GROK_IMAGE_MODEL
    return {
        "claude": (config.DEFAULT_CLAUDE_MODEL,),
        "codex": (config.DEFAULT_CODEX_MODEL,),
        "grok": (config.DEFAULT_GROK_MODEL, GROK_IMAGE_MODEL),
        "cursor": (config.DEFAULT_CURSOR_MODEL,),
        "agy": (config.DEFAULT_MODEL,),
    }[provider]


def _admission_report(provider, profile):
    """One admission outcome per shipped default, against this profile's v2 record."""
    from providers import admission, catalog_v2
    # The record itself is checked; today's gate depends on a request's
    # environment, so it does not apply here.
    record, load_error = catalog_v2.load(provider, profile)
    now = _datetime.datetime.now(_datetime.timezone.utc)
    rows = []
    for model in _shipped_defaults(provider):
        if load_error:
            decision = {"refusal": load_error}
        else:
            decision = admission.decide(
                {"provider": provider, "model": model, "effort": None},
                record, now=now)
        refusal = decision.get("refusal") if isinstance(decision, dict) else "no_catalog"
        # effort_required is reported and is not a failing admission.
        hard = refusal not in (None, "effort_required")
        rows.append({
            "model": model,
            "ok": not hard,
            "effort": decision.get("effort") if isinstance(decision, dict) else None,
            "refusal": refusal,
        })
    return rows


def admit(provider, model, effort=None, profile="default"):
    """Admit one model and effort against one v2 record.

    Loads that record and calls ``admission.decide``. Does not run the gate,
    read aliases, spawn a vendor CLI, or write.
    """
    from providers import admission, catalog_v2
    name = _profile_name(profile)
    if provider not in PROVIDERS:
        raise CatalogError("unknown_provider")
    record, load_error = catalog_v2.load(provider, name)
    if load_error:
        decision = {
            "model": None,
            "effort": None,
            "effort_source": None,
            "warnings": [],
            "refusal": load_error,
        }
    else:
        decision = admission.decide(
            {"provider": provider, "model": model, "effort": effort},
            record, now=_datetime.datetime.now(_datetime.timezone.utc))
    refusal = decision.get("refusal") if isinstance(decision, dict) else "no_catalog"
    admitted = refusal is None
    decided_model = decision.get("model") if isinstance(decision, dict) else None
    decided_effort = decision.get("effort") if isinstance(decision, dict) else None
    warnings = decision.get("warnings") if isinstance(decision, dict) else []
    if not isinstance(warnings, list):
        warnings = []
    requested = model.strip() if isinstance(model, str) else ""
    return {
        "ok": admitted,
        "admitted": admitted,
        "provider": provider,
        "profile": name,
        "model": decided_model if isinstance(decided_model, str) and decided_model else requested,
        "effort": decided_effort if decided_effort is not None else effort,
        "effort_source": decision.get("effort_source") if isinstance(decision, dict) else None,
        "refusal": refusal,
        "warnings": list(warnings),
    }


def check(provider="all", profile="default"):
    """Admit the shipped defaults against v2 records. A hard refusal is exit 2."""
    try:
        selected = PROVIDERS if provider in (None, "all") else (provider,)
        if any(item not in PROVIDERS for item in selected):
            raise CatalogError("unknown_provider")
        rows = []
        blocked = False
        for name in selected:
            admission_rows = _admission_report(name, profile)
            hard = any(not item["ok"] for item in admission_rows)
            if hard:
                blocked = True
            rows.append({"provider": name, "ok": not hard,
                         "admission": admission_rows})
        return {"ok": not blocked, "profile": _profile_name(profile),
                "providers": rows, "exit_code": 2 if blocked else 0}
    except CatalogError as exc:
        out = exc.envelope()
        out["exit_code"] = 2
        return out


def _parse_model_line(line, pattern, provider):
    match = pattern.fullmatch(line)
    if not match:
        raise CatalogError("malformed_output", provider)
    model_id, name = match.group(1).strip(), match.group(2).strip()
    if not model_id or not name:
        raise CatalogError("malformed_output", provider)
    return {"id": model_id, "name": name, "routable_id": ROAD_PREFIX[provider] + model_id,
            "supported_efforts": [], "efforts_known": False}

def parse_cursor_output(output, *, observed_at=None, cli_version=None):
    if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_STDOUT_BYTES:
        raise CatalogError("output_cap", "cursor")
    if not output.strip():
        raise CatalogError("empty_output", "cursor")
    if not output.endswith("\n"):
        raise CatalogError("incomplete_output", "cursor")
    rows = []
    pattern = re.compile(r"^\s*([^\s]+)\s+-\s+(.+?)\s*$")
    lines = [line for line in output.splitlines() if line.strip()]
    saw_header = False
    saw_footer = False
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line == "Available models":
            if saw_header or rows or saw_footer:
                raise CatalogError("malformed_output", "cursor")
            saw_header = True
            continue
        if line.startswith("Tip:"):
            if not saw_header or saw_footer or index != len(lines) - 1:
                raise CatalogError("malformed_output", "cursor")
            saw_footer = True
            continue
        if not saw_header or saw_footer:
            raise CatalogError("malformed_output", "cursor")
        rows.append(_parse_model_line(line, pattern, "cursor"))
    if not saw_header or not saw_footer:
        raise CatalogError("incomplete_output", "cursor")
    if not rows:
        raise CatalogError("empty_output", "cursor")
    return _strict_live_record("cursor", rows, observed_at=observed_at,
                               cli_version=cli_version)


def parse_agy_output(output, *, observed_at=None, cli_version=None):
    if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_STDOUT_BYTES:
        raise CatalogError("output_cap", "agy")
    if not output.strip():
        raise CatalogError("empty_output", "agy")
    if not output.endswith("\n"):
        raise CatalogError("incomplete_output", "agy")
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t", 1)
        if len(fields) != 2:
            raise CatalogError("malformed_output", "agy")
        rows.append({"id": fields[0].strip(), "name": fields[1].strip(),
                     "routable_id": ROAD_PREFIX["agy"] + fields[0].strip(),
                     "supported_efforts": [], "efforts_known": False})
    if not rows:
        raise CatalogError("empty_output", "agy")
    return _strict_live_record("agy", rows, observed_at=observed_at,
                               cli_version=cli_version)


def _strict_live_record(provider, rows, *, observed_at, cli_version):
    """Build one strict live record from rows that already passed format checks."""
    from providers import catalog_record
    record = {
        "provider": provider,
        "source": "live",
        "observed_at": observed_at or _utc_now(),
        "availability": "provider_advertised",
        "models": rows,
    }
    if cli_version:
        record["cli_version"] = cli_version
    try:
        strict, _warnings = catalog_record.validate_strict(provider, record)
    except catalog_record.StrictRecordError as exc:
        raise CatalogError(exc.reason, provider)
    return strict


def parse_agy_effort_tokens(help_output):
    """Return only the effort vocabulary printed by the installed agy CLI."""
    if not isinstance(help_output, str) or len(help_output.encode("utf-8")) > MAX_STDOUT_BYTES:
        raise CatalogError("output_cap", "agy")
    match = re.search(r"--effort\b(?:(?!\n\s*--)[\s\S]){0,500}?\(([^()]+)\)",
                      help_output)
    if not match:
        raise CatalogError("unsupported_effort", "agy")
    tokens = [part.strip().lower() for part in match.group(1).replace(",", "|").split("|")
              if part.strip()]
    if not tokens or any(token not in AGY_EFFORTS for token in tokens):
        raise CatalogError("unsupported_effort", "agy")
    return frozenset(tokens)

def _claude_models_from_control_events(output):
    """Return the model list from one claude initialize stream.

    This is the walk ``parse_claude_control_events`` has always done, before
    today's effort gate. A v2 parse can keep tokens that gate rejects.
    """
    if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_STDOUT_BYTES:
        raise CatalogError("output_cap", "claude")
    found = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = _json_loads(line)
        except (ValueError, TypeError):
            raise CatalogError("malformed_output", "claude")
        if not isinstance(event, dict):
            raise CatalogError("malformed_output", "claude")
        if event.get("type") == "control_request":
            raise CatalogError("unexpected_server_request", "claude")
        if event.get("type") == "control_response":
            response = event.get("response")
            if isinstance(response, dict) and response.get("request_id") == "catalog-init":
                payload = response.get("response")
                if not isinstance(payload, dict):
                    raise CatalogError("malformed_response", "claude")
                models = payload.get("models", payload.get("supported_models"))
                found.append(models)
        elif str(event.get("type", "")).startswith(("tool", "user")):
            raise CatalogError("unexpected_event", "claude")
    if len(found) != 1:
        raise CatalogError("incomplete_output", "claude")
    return found[0]


def parse_claude_control_events(output, *, observed_at=None, cli_version=None):
    from providers import catalog_record
    models = _claude_models_from_control_events(output)
    try:
        record = catalog_record.parse_claude_models(
            models, observed_at=observed_at, cli_version=cli_version)
        strict, _warnings = catalog_record.validate_strict("claude", record)
    except catalog_record.StrictRecordError as exc:
        raise CatalogError(exc.reason, "claude")
    return strict

def _decode_output(raw, provider):
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise CatalogError("invalid_utf8", provider)


def _terminate_process(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, name, None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _run_bounded(argv, *, env, cwd, timeout_s, provider, input_bytes=None,
                 combine_stderr=False):
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if combine_stderr else subprocess.DEVNULL,
            start_new_session=True, bufsize=0,
        )
    except FileNotFoundError:
        raise CatalogError("cli_missing", provider)
    except OSError:
        raise CatalogError("spawn_failed", provider)
    try:
        if input_bytes is not None:
            try:
                proc.stdin.write(input_bytes)
                proc.stdin.flush()
            except (OSError, BrokenPipeError):
                raise CatalogError("stdin_closed", provider)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        data = bytearray()
        deadline = time.monotonic() + max(1, int(timeout_s))
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogError("timeout", provider)
            events = selector.select(min(0.2, remaining))
            if not events and proc.poll() is not None:
                break
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                data.extend(chunk)
                if len(data) > MAX_STDOUT_BYTES:
                    raise CatalogError("output_cap", provider)
        try:
            code = proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise CatalogError("timeout", provider)
        if code != 0:
            raise CatalogError("discovery_exit", provider)
        return bytes(data)
    finally:
        _terminate_process(proc)


class _RpcReader:
    def __init__(self, proc, provider, deadline):
        self.proc, self.provider, self.deadline = proc, provider, deadline
        self.buffer = bytearray()
        self.total = 0

    def line(self):
        while b"\n" not in self.buffer:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogError("timeout", self.provider)
            try:
                ready, _write, _error = __import__("select").select(
                    [self.proc.stdout], [], [], min(0.2, remaining))
            except (OSError, ValueError):
                raise CatalogError("protocol_io", self.provider)
            if not ready:
                if self.proc.poll() is not None:
                    raise CatalogError("incomplete_output", self.provider)
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise CatalogError("incomplete_output", self.provider)
            self.total += len(chunk)
            if self.total > MAX_STDOUT_BYTES:
                raise CatalogError("output_cap", self.provider)
            self.buffer.extend(chunk)
        line, _sep, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        return _decode_output(line, self.provider)


def _send_codex_rpc(proc, method, params=None, request_id=None, *, deadline):
    if method not in CODEX_RPC_METHODS:
        raise CatalogError("protocol_method_denied", "codex")
    message = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        fd = proc.stdin.fileno()
        os.set_blocking(fd, False)
        offset = 0
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CatalogError("timeout", "codex")
                if not selector.select(remaining):
                    continue
                if time.monotonic() >= deadline:
                    raise CatalogError("timeout", "codex")
                try:
                    written = os.write(fd, payload[offset:])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise OSError("stdin closed")
                offset += written
    except CatalogError:
        raise
    except (AttributeError, OSError, ValueError):
        raise CatalogError("stdin_closed", "codex")


def _read_codex_response(reader, wanted_id):
    while True:
        try:
            message = _json_loads(reader.line())
        except CatalogError:
            raise
        except (ValueError, TypeError):
            raise CatalogError("malformed_response", "codex")
        if not isinstance(message, dict):
            raise CatalogError("malformed_response", "codex")
        if isinstance(message.get("method"), str):
            if "id" in message:
                raise CatalogError("server_request_denied", "codex")
            if message["method"] not in CODEX_SAFE_NOTIFICATIONS:
                raise CatalogError("unexpected_notification", "codex")
            # This known control-plane notification has no response channel.
            continue
        if message.get("id") != wanted_id:
            raise CatalogError("unexpected_response", "codex")
        if "error" in message:
            raise CatalogError("discovery_error", "codex")
        if "result" not in message or not isinstance(message["result"], dict):
            raise CatalogError("malformed_response", "codex")
        return message["result"]


def _claude_discovery_argv(binary):
    """The metadata argv `_discover_claude` has always built. No prompt is sent."""
    return [binary, "--print", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose", "--tools", "",
            "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}",
            "--setting-sources", "", "--no-session-persistence",
            "--disable-slash-commands"]


def _discovery_posture():
    """The platform chooses the posture. Callers do not."""
    if platform.system() == "Darwin":
        return _POSTURE_SEATBELT
    return _POSTURE_UNFENCED


def _resolved_metadata_binary(provider):
    """The executable the discoverer looks up, or None.

    Cursor, agy, and claude return the realpath their lookups already return.
    Grok and codex are ``vendor_bin.which`` plus ``os.path.realpath``, which is
    what ``_discover_grok`` and ``_discover_codex`` pass to the spawn.
    """
    if provider == "cursor":
        from providers.cursor import resolve_bin
        return resolve_bin()
    if provider == "agy":
        from providers.agy import find_agy
        return find_agy()
    if provider == "grok":
        found = vendor_bin.which("grok")
        return os.path.realpath(found) if found else None
    if provider == "codex":
        found = vendor_bin.which("codex")
        return os.path.realpath(found) if found else None
    if provider == "claude":
        from providers.kernel_sandbox import resolve_agent_binary
        return resolve_agent_binary("claude")
    return None


def _metadata_argv_allowed(provider, argv):
    try:
        parts = tuple(os.fsdecode(os.fspath(part)) for part in argv)
    except (TypeError, ValueError):
        return False
    if not parts:
        return False
    resolved = _resolved_metadata_binary(provider)
    if not resolved:
        return False
    try:
        if os.path.realpath(parts[0]) != os.path.realpath(resolved):
            return False
    except (OSError, ValueError):
        return False
    if provider == "claude":
        return list(parts) == _claude_discovery_argv(parts[0])
    tail = _METADATA_ARGV_TAILS.get(provider)
    return tail is not None and len(parts) == 1 + len(tail) and parts[1:] == tail


def _require_verified_metadata_fence(metadata, provider):
    """Refuse metadata discovery unless the shared host fence proved its wrap."""
    if not isinstance(metadata, dict) or metadata.get("fence") != "seatbelt-fs":
        raise CatalogError("discovery_unavailable", provider)


def _accept_discovery_posture(metadata, provider, argv):
    """Admit one refresh discovery spawn under the platform posture.

    Darwin still requires ``seatbelt-fs``. Everywhere else the shared fence
    reports ``none``; only the fixed metadata argv may then start.
    """
    if _discovery_posture() == _POSTURE_SEATBELT:
        _require_verified_metadata_fence(metadata, provider)
        return
    if not isinstance(metadata, dict) or metadata.get("fence") != "none":
        raise CatalogError("discovery_unavailable", provider)
    if not _metadata_argv_allowed(provider, argv):
        raise CatalogError("discovery_unavailable", provider)


def _attach_discovery(record):
    """Record the posture that just ran. The platform is read, not chosen."""
    stamped = dict(record)
    stamped["discovery"] = {
        "posture": _discovery_posture(),
        "platform": platform.system(),
    }
    return stamped


def _query_codex_models(binary, *, env, cwd, timeout_s):
    raw = _fetch_codex_pages(binary, env=env, cwd=cwd, timeout_s=timeout_s)
    return _strict_vendor_record("codex", raw)


def _fetch_codex_pages(binary, *, env, cwd, timeout_s):
    """Run the codex ``model/list`` exchange and return ``{"data": models}``."""
    try:
        argv, child_env, metadata = __import__("providers.host_fence", fromlist=["prepare_host_fence"]).prepare_host_fence(
            [binary, "app-server"], provider="codex", workdir=cwd,
            read_only=True, env=env,
            config_dir=env.get("CODEX_HOME"))
    except Exception as exc:
        reason = getattr(getattr(exc, "envelope", None), "get", lambda *_: None)("reason")
        raise CatalogError(reason or "fence_unavailable", "codex")
    _accept_discovery_posture(metadata, "codex", argv)
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=child_env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, start_new_session=True,
                                bufsize=0)
    except OSError:
        raise CatalogError("spawn_failed", "codex")
    try:
        deadline = time.monotonic() + max(1, int(timeout_s))
        reader = _RpcReader(proc, "codex", deadline)
        _send_codex_rpc(proc, "initialize", {
            "clientInfo": {"name": "omni-model-catalog", "title": "omni-model-catalog", "version": "1"},
            "capabilities": {},
        }, request_id=1, deadline=deadline)
        _read_codex_response(reader, 1)
        _send_codex_rpc(proc, "initialized", {}, deadline=deadline)
        models = []
        cursor = None
        cursors_seen = set()
        for page_no in range(MAX_CODEX_PAGES):
            params = {"limit": 100, "includeHidden": False}
            if cursor is not None:
                params["cursor"] = cursor
            _send_codex_rpc(proc, "model/list", params, request_id=2 + page_no,
                            deadline=deadline)
            page = _read_codex_response(reader, 2 + page_no)
            data = page.get("data")
            if not isinstance(data, list) or not data:
                raise CatalogError("malformed_response", "codex")
            models.extend(data)
            if len(models) > MAX_MODEL_COUNT:
                raise CatalogError("model_count_cap", "codex")
            next_cursor = page.get("nextCursor")
            if next_cursor in (None, ""):
                break
            if not isinstance(next_cursor, str) or next_cursor in cursors_seen:
                raise CatalogError("pagination_invalid", "codex")
            cursors_seen.add(next_cursor)
            cursor = next_cursor
        else:
            raise CatalogError("pagination_limit", "codex")
        return _remember_fetch({"data": models})
    finally:
        _terminate_process(proc)


def _spawn_one_shot(provider, argv, profile, timeout_s, *, config_dir=None,
                    cursor_config=False, combine_stderr=False,
                    seatbelt_required=False):
    from providers.host_fence import HostFenceError, prepare_host_fence

    profile = _profile_name(profile)
    _overrides, child_env, _allowed_env, selected_home, _endpoints = (
        _catalog_child_environment(provider, profile))
    temp_config = None
    cwd = tempfile.mkdtemp(prefix="omni-catalog-")
    try:
        if cursor_config:
            from providers.config import CURSOR_CONFIG_DIR_ENV, ENV_CURSOR_CONFIG_DIR
            from providers.cursor import resolve_config_dir
            config_dir = resolve_config_dir(child_env)
            if config_dir is None:
                temp_config = tempfile.mkdtemp(prefix="omni-catalog-cursor-")
                config_dir = temp_config
            child_env[ENV_CURSOR_CONFIG_DIR] = config_dir
            child_env[CURSOR_CONFIG_DIR_ENV] = config_dir
        try:
            fenced_argv, child_env, metadata = prepare_host_fence(
                argv, provider=provider, workdir=cwd, read_only=True,
                env=child_env, config_dir=config_dir)
        except HostFenceError as exc:
            reason = exc.envelope.get("reason") or "fence_unavailable"
            raise CatalogError(reason, provider)
        if seatbelt_required:
            # Dispatch (agy effort) keeps the pre-posture check on every host.
            _require_verified_metadata_fence(metadata, provider)
        else:
            _accept_discovery_posture(metadata, provider, fenced_argv)
        raw = _run_bounded(fenced_argv, env=child_env, cwd=cwd,
                           timeout_s=timeout_s, provider=provider,
                           combine_stderr=combine_stderr)
        return _decode_output(raw, provider)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
        if temp_config:
            shutil.rmtree(temp_config, ignore_errors=True)


def _remember_fetch(value):
    _FETCHED_OUTPUT.set(value)
    return value


def _fetch_cursor(profile, timeout_s):
    from providers.cursor import resolve_bin
    binary = resolve_bin()
    if not binary:
        raise CatalogError("cli_missing", "cursor")
    output = _spawn_one_shot("cursor", [binary, "--list-models"], profile,
                             timeout_s, cursor_config=True)
    return _remember_fetch(output)


def _discover_cursor(profile, timeout_s):
    return _attach_discovery(parse_cursor_output(_fetch_cursor(profile, timeout_s)))


def _fetch_agy(profile, timeout_s):
    from providers.agy import find_agy
    binary = find_agy()
    if not binary:
        raise CatalogError("cli_missing", "agy")
    output = _spawn_one_shot("agy", [binary, "models"], profile, timeout_s)
    return _remember_fetch(output)


def _discover_agy(profile, timeout_s):
    return _attach_discovery(parse_agy_output(_fetch_agy(profile, timeout_s)))


def agy_effort_tokens(profile="default", timeout_s=30):
    """Read effort capability from this installed CLI through the shared fence."""
    from providers.agy import find_agy
    binary = find_agy()
    if not binary:
        raise CatalogError("cli_missing", "agy")
    output = _spawn_one_shot("agy", [binary, "--help"], profile, timeout_s,
                             combine_stderr=True, seatbelt_required=True)
    return parse_agy_effort_tokens(output)


def _fetch_grok(profile, timeout_s):
    binary = vendor_bin.which("grok")
    if not binary:
        raise CatalogError("cli_missing", "grok")
    output = _spawn_one_shot("grok", [os.path.realpath(binary), "models"], profile, timeout_s)
    return _remember_fetch(output)


def _discover_grok(profile, timeout_s):
    return _attach_discovery(_strict_vendor_record("grok", _fetch_grok(profile, timeout_s)))


def _fetch_claude(profile, timeout_s):
    from providers.kernel_sandbox import resolve_agent_binary
    binary = resolve_agent_binary("claude")
    if not binary:
        raise CatalogError("cli_missing", "claude")
    argv = _claude_discovery_argv(binary)
    from providers.host_fence import HostFenceError, prepare_host_fence
    _overrides, child_env, _allowed_env, _selected_home, _endpoints = (
        _catalog_child_environment("claude", profile))
    config_dir = child_env.get("CLAUDE_CONFIG_DIR")
    cwd = tempfile.mkdtemp(prefix="omni-catalog-claude-")
    proc = None
    try:
        try:
            fenced, child_env, metadata = prepare_host_fence(
                argv, provider="claude", workdir=cwd, read_only=True,
                env=child_env, config_dir=config_dir)
        except HostFenceError as exc:
            raise CatalogError(exc.envelope.get("reason") or "fence_unavailable", "claude")
        _accept_discovery_posture(metadata, "claude", fenced)
        try:
            proc = subprocess.Popen(fenced, cwd=cwd, env=child_env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, start_new_session=True,
                                    bufsize=0)
        except OSError:
            raise CatalogError("spawn_failed", "claude")
        request = {"type": "control_request", "request_id": "catalog-init",
                   "request": {"subtype": "initialize", "hooks": {}, "agents": {}}}
        try:
            proc.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (OSError, BrokenPipeError):
            raise CatalogError("stdin_closed", "claude")
        deadline = time.monotonic() + max(1, int(timeout_s))
        reader = _RpcReader(proc, "claude", deadline)
        lines = []
        while time.monotonic() < deadline:
            line = reader.line()
            lines.append(line)
            if len(lines) > 10000:
                raise CatalogError("event_count_cap", "claude")
            try:
                event = _json_loads(line)
            except (ValueError, TypeError):
                raise CatalogError("malformed_output", "claude")
            if isinstance(event, dict) and event.get("type") == "control_request":
                raise CatalogError("unexpected_server_request", "claude")
            response = event.get("response") if isinstance(event, dict) else None
            if (isinstance(event, dict) and event.get("type") == "control_response"
                    and isinstance(response, dict)
                    and response.get("request_id") == "catalog-init"):
                break
        else:
            raise CatalogError("timeout", "claude")
        return _remember_fetch("\n".join(lines))
    finally:
        _terminate_process(proc)
        shutil.rmtree(cwd, ignore_errors=True)


def _discover_claude(profile, timeout_s):
    return _attach_discovery(parse_claude_control_events(_fetch_claude(profile, timeout_s)))


def _fetch_codex(profile, timeout_s):
    _overrides, child_env, _allowed_env, _selected_home, _endpoints = (
        _catalog_child_environment("codex", profile))
    binary = vendor_bin.which("codex")
    if not binary:
        raise CatalogError("cli_missing", "codex")
    cwd = tempfile.mkdtemp(prefix="omni-catalog-codex-")
    try:
        return _fetch_codex_pages(
            os.path.realpath(binary), env=child_env, cwd=cwd, timeout_s=timeout_s)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def _discover_codex(profile, timeout_s):
    return _attach_discovery(_strict_vendor_record(
        "codex", _fetch_codex(profile, timeout_s)))


def _strict_vendor_record(provider, raw):
    """Parse codex or grok vendor output into a strict record."""
    from providers import catalog_record
    try:
        if provider == "codex":
            record = catalog_record.parse_codex_result(raw, observed_at=_utc_now())
        elif provider == "grok":
            record = catalog_record.parse_grok_output(raw, observed_at=_utc_now())
        else:
            raise CatalogError("unknown_provider", provider)
        strict, _warnings = catalog_record.validate_strict(provider, record)
    except catalog_record.StrictRecordError as exc:
        raise CatalogError(exc.reason, provider)
    return strict


_DISCOVERERS = {"cursor": _discover_cursor, "agy": _discover_agy,
                "codex": _discover_codex, "claude": _discover_claude,
                "grok": _discover_grok}


def _best_effort_cli_version(provider, profile, timeout_s):
    """Return an installed CLI version through the shared read-only fence."""
    try:
        if provider == "cursor":
            from providers.cursor import resolve_bin
            binary = resolve_bin()
        elif provider == "agy":
            from providers.agy import find_agy
            binary = find_agy()
        elif provider == "codex":
            binary = vendor_bin.which("codex")
        elif provider == "claude":
            from providers.kernel_sandbox import resolve_agent_binary
            binary = resolve_agent_binary("claude")
        else:
            binary = vendor_bin.which("grok")
        if not binary:
            return None
        argv = [os.path.realpath(binary), "--version"]
        output = _spawn_one_shot(
            provider, argv, profile, timeout_s,
            cursor_config=(provider == "cursor"),
            combine_stderr=True)
        match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)+(?:-[A-Za-z0-9][A-Za-z0-9._-]*)?)",
                          output)
        return match.group(1) if match else None
    except Exception:
        # Version is optional metadata. Its failure must not discard a valid listing.
        return None


def _refresh_failure_result(provider, profile, reason):
    """Report one provider failure. This does not read or write a catalog."""
    del profile
    return {"ok": False, "provider": provider, "reason": reason,
            "detail": "model catalog %s" % reason}


def _v2_record_from_raw(provider, raw, cli_version):
    """Parse one fetched payload with the strict parsers and validate it."""
    from providers import catalog_record

    if provider == "claude":
        models = _claude_models_from_control_events(raw)
        record = catalog_record.parse_claude_models(
            models, cli_version=cli_version)
    elif provider == "codex":
        record = catalog_record.parse_codex_result(raw, cli_version=cli_version)
    elif provider == "grok":
        record = catalog_record.parse_grok_output(raw, cli_version=cli_version)
    elif provider == "cursor":
        record = parse_cursor_output(raw, cli_version=cli_version)
    elif provider == "agy":
        record = parse_agy_output(raw, cli_version=cli_version)
    else:
        raise CatalogError("unknown_provider", provider)
    record = _attach_discovery(record)
    strict, _warnings = catalog_record.validate_strict(provider, record)
    return strict


def refresh(provider="all", profile="default", timeout_s=30):
    """Discover each provider and write only the v2 record. Success follows that write."""
    from providers import catalog_record, catalog_v2

    selected = PROVIDERS if provider in (None, "all") else (provider,)
    if any(item not in PROVIDERS for item in selected):
        return {"ok": False, "reason": "unknown_provider", "providers": {}}
    try:
        timeout_s = int(timeout_s)
        if timeout_s < 1 or timeout_s > 3600:
            raise ValueError()
        profile = _profile_name(profile)
    except (TypeError, ValueError, CatalogError):
        return {"ok": False, "reason": "invalid_arguments", "providers": {}}
    results = {}
    for name in selected:
        fetched = _FETCHED_OUTPUT.set(None)
        applied_version = None
        try:
            try:
                deadline = time.monotonic() + timeout_s
                discovered = _DISCOVERERS[name](profile, timeout_s)
                remaining = deadline - time.monotonic()
                version_budget = min(2, int(remaining - 0.25))
                if version_budget > 0:
                    try:
                        version = _best_effort_cli_version(name, profile, version_budget)
                    except Exception:
                        version = None
                    if version:
                        applied_version = version
                raw = _FETCHED_OUTPUT.get()
                if raw is not None:
                    record = _v2_record_from_raw(name, raw, applied_version)
                else:
                    if applied_version and isinstance(discovered, dict):
                        discovered = dict(discovered)
                        discovered["cli_version"] = applied_version
                    record, _warnings = catalog_record.validate_strict(name, discovered)
                catalog_v2.save(name, profile, record)
                results[name] = {"ok": True, "provider": name,
                                 "source": record["source"],
                                 "observed_at": record["observed_at"],
                                 "model_count": len(record["models"])}
            except CatalogError as exc:
                results[name] = _refresh_failure_result(name, profile, exc.reason)
            except catalog_record.StrictRecordError as exc:
                results[name] = _refresh_failure_result(name, profile, exc.reason)
            except Exception:
                results[name] = _refresh_failure_result(name, profile, "runtime_error")
            finally:
                # Per-provider failures must never abort later roads.
                if name not in results:
                    results[name] = _refresh_failure_result(name, profile, "runtime_error")
        finally:
            _FETCHED_OUTPUT.reset(fetched)
    return {"ok": all(item.get("ok") for item in results.values()),
            "profile": profile, "providers": results}

def _default_path(provider, profile):
    root = _catalog_root(create=False)
    return root / ("model-catalogs-%s-%s.log" % (provider, _profile_name(profile)))


def _schedule_endpoint_bindings(provider, child_env, source_env):
    return _validate_endpoint_bindings(provider, child_env, source_env)


def _registered_schedule_homes(source_env):
    from providers.kernel_sandbox import (registered_account_homes,
                                          registered_cursor_config_dirs)

    result = {}
    for registry in (registered_account_homes(source_env),
                     registered_cursor_config_dirs(source_env)):
        for label, _provider, path in registry:
            result[label] = os.path.realpath(os.path.expanduser(path))
    return result


def _schedule_home_binding(provider, profile):
    if provider == "cursor":
        base = "OMNI_LLM_CURSOR_CONFIG_DIR"
    else:
        base = _HOME_ENV_KEYS[provider]
    return base if profile == "default" else base + "_" + profile.upper()


def _schedule_environment(provider, profile):
    source_env = os.environ.copy()
    result = {"PATH": source_env.get("PATH", "/usr/bin:/bin"),
              "OMNI_LLM_STATE_DIR": str(_state_dir())}
    result.update(_registered_schedule_homes(source_env))
    required_selectors = set()
    selected = PROVIDERS if provider == "all" else (provider,)
    supported_profiles = 0
    for road in selected:
        try:
            identity, child_env = _profile_identity_inputs(road, profile)
        except CatalogError as exc:
            if provider != "all" or exc.reason != "unsupported_profile":
                raise
            continue
        supported_profiles += 1
        config_home = identity["config_home"]
        if config_home and identity.get("home_explicit"):
            result[_schedule_home_binding(road, profile)] = config_home
        endpoint_bindings = _schedule_endpoint_bindings(road, child_env, source_env)
        result.update(endpoint_bindings)
        from providers.host_fence import _child_env_pass_names
        required_selectors.update(set(endpoint_bindings) & _child_env_pass_names(source_env))
    if provider == "all" and supported_profiles == 0:
        raise CatalogError("unsupported_profile")
    if required_selectors:
        result["OMNI_LLM_CHILD_ENV_PASS"] = ",".join(sorted(required_selectors))
    return result


def schedule_plist(profile, interval=86400, timeout_s=120, provider="all"):
    try:
        profile = _profile_name(profile)
        if provider not in (*PROVIDERS, "all"):
            raise CatalogError("unknown_provider")
        interval, timeout_s = int(interval), int(timeout_s)
        if interval < 60 or timeout_s < 1 or timeout_s > 3600:
            raise CatalogError("invalid_arguments")
        if provider != "all":
            _profile_identity_inputs(provider, profile)
        entry = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "omni_cli.py"))
        python = os.path.realpath(sys.executable)
        log_path = _default_path(provider, profile)
        import plistlib
        suffix = profile if provider == "all" else provider + "." + profile
        label = "com.omni-llm.model-catalogs." + suffix
        value = {
            "Label": label,
            "ProgramArguments": [python, entry, "models", "refresh", "--provider", provider,
                                 "--profile", profile, "--timeout", str(timeout_s)],
            "StartInterval": interval,
            "EnvironmentVariables": _schedule_environment(provider, profile),
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(log_path),
            "ProcessType": "Background",
        }
        return {"ok": True, "label": label,
                "plist": plistlib.dumps(value, fmt=plistlib.FMT_XML).decode("utf-8")}
    except (CatalogError, ValueError, OSError) as exc:
        reason = exc.reason if isinstance(exc, CatalogError) else "invalid_arguments"
        return {"ok": False, "reason": reason,
                "detail": "model catalog %s" % reason}
