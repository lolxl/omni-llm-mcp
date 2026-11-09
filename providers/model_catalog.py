"""Local, attributable model catalogs and metadata-only refresh adapters.

Catalog state is scoped to a provider profile, never changes model defaults,
and is used only as additive metadata by Cursor effort resolution.
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


SCHEMA_VERSION = 1
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
_ACTIVE_PROFILE = contextvars.ContextVar("omni_catalog_profile", default="default")
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


def _cache_paths(provider, profile, create=False):
    if provider not in PROVIDERS:
        raise CatalogError("unknown_provider", provider)
    root = _catalog_root(create=create)
    fingerprint = _scope_fingerprint(provider, profile, create=create)
    stem = "%s-%s" % (provider, fingerprint)
    return root / (stem + ".json"), root / (stem + ".status.json"), root / (stem + ".lock")


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


def validate_record(provider, value, *, allow_manual_timestamp=True):
    if provider not in PROVIDERS or not isinstance(value, dict):
        raise CatalogError("schema_invalid", provider)
    source = value.get("source")
    if source not in ("live", "bundled", "manual"):
        raise CatalogError("schema_invalid", provider)
    observed_at = value.get("observed_at")
    if observed_at is not None and _utc_datetime(observed_at) is None:
        raise CatalogError("timestamp_invalid", provider)
    if observed_at is None and source != "manual" and not allow_manual_timestamp:
        raise CatalogError("timestamp_missing", provider)
    if source != "manual" and observed_at is None:
        raise CatalogError("timestamp_missing", provider)
    version = value.get("cli_version")
    if version is not None and (not isinstance(version, str) or len(version) > 100):
        raise CatalogError("schema_invalid", provider)
    availability = value.get("availability")
    if availability not in ("provider_advertised", "unknown"):
        raise CatalogError("schema_invalid", provider)
    models = value.get("models")
    if not isinstance(models, list) or not models or len(models) > MAX_MODEL_COUNT:
        raise CatalogError("model_list_invalid", provider)
    normalized = []
    seen = set()
    for item in models:
        if not isinstance(item, dict):
            raise CatalogError("model_entry_invalid", provider)
        model_id = item.get("id")
        if not isinstance(model_id, str) or not _ID_RE.fullmatch(model_id):
            raise CatalogError("model_id_invalid", provider)
        if model_id in seen:
            raise CatalogError("duplicate_model_id", provider)
        seen.add(model_id)
        name = item.get("name", model_id)
        if not isinstance(name, str) or not _ID_RE.fullmatch(name):
            raise CatalogError("model_name_invalid", provider)
        routable_id = item.get("routable_id", ROAD_PREFIX[provider] + model_id)
        if routable_id != ROAD_PREFIX[provider] + model_id:
            raise CatalogError("routable_id_invalid", provider)
        efforts = item.get("supported_efforts", [])
        if not isinstance(efforts, list) or any(e not in EFFORTS for e in efforts):
            raise CatalogError("effort_metadata_invalid", provider)
        efforts = sorted(set(efforts), key=lambda e: ("minimal", "low", "medium", "high", "xhigh", "max", "ultra").index(e))
        default_effort = item.get("default_effort")
        if default_effort is not None and default_effort not in efforts:
            raise CatalogError("effort_metadata_invalid", provider)
        effort_known = item.get("efforts_known", bool(efforts))
        if not isinstance(effort_known, bool) or (efforts and not effort_known):
            raise CatalogError("effort_metadata_invalid", provider)
        row = {"id": model_id, "name": name, "routable_id": routable_id,
               "supported_efforts": efforts, "efforts_known": effort_known}
        if default_effort is not None:
            row["default_effort"] = default_effort
        visibility = item.get("visibility")
        if visibility is not None:
            if not isinstance(visibility, str) or not _ID_RE.fullmatch(visibility):
                raise CatalogError("model_entry_invalid", provider)
            row["visibility"] = visibility
        normalized.append(row)
    result = {"provider": provider, "source": source, "observed_at": observed_at,
              "availability": availability, "models": normalized}
    if version is not None:
        result["cli_version"] = version
    source_note = value.get("source_note")
    if source_note in ("vendor_cache_not_live", "unauthenticated_vendor_registry",
                       "vendor_registry_not_entitlement"):
        result["source_note"] = source_note
    if provider == "cursor":
        mappings = value.get("effort_slugs")
        if mappings is not None:
            if not isinstance(mappings, dict):
                raise CatalogError("effort_metadata_invalid", provider)
            exact_ids = {row["id"] for row in normalized}
            checked = {}
            for base, by_effort in mappings.items():
                if not isinstance(base, str) or not _ID_RE.fullmatch(base) or not isinstance(by_effort, dict):
                    raise CatalogError("effort_metadata_invalid", provider)
                checked[base] = {}
                for effort, slug in by_effort.items():
                    if effort not in CURSOR_EFFORTS or slug not in exact_ids:
                        raise CatalogError("effort_metadata_invalid", provider)
                    checked[base][effort] = slug
            result["effort_slugs"] = checked
    return result


def _load_bundle():
    path = Path(__file__).with_name("model_catalog.json")
    try:
        data = _json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raise CatalogError("bundled_catalog_invalid")
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError("bundled_catalog_invalid")
    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise CatalogError("bundled_catalog_invalid")
    result = {}
    for provider, record in providers.items():
        if provider not in PROVIDERS:
            raise CatalogError("bundled_catalog_invalid", provider)
        result[provider] = validate_record(provider, record)
    return result


def bundled_catalog():
    return _load_bundle()


def agy_opus_alias_model(name, effort):
    """Resolve only an explicit AGY Opus/version alias from bundled metadata.

    This is local name normalization, not a discovery or entitlement probe.
    Missing and ambiguous rows refuse instead of choosing another seat.
    """
    match = re.fullmatch(r"agy-opus-([0-9]+(?:\.[0-9]+)*)", str(name).strip(), re.I)
    if not match:
        return None, {"ok": False, "reason": "unsupported_model",
                      "detail": "agy Opus shorthand requires an exact version"}
    token = str(effort or "").strip().lower()
    if token not in AGY_EFFORTS:
        return None, {"ok": False, "reason": "unsupported_effort",
                      "detail": "agy Opus shorthand requires a supported explicit effort"}
    try:
        record = bundled_catalog().get("agy") or {}
    except CatalogError as exc:
        return None, exc.envelope()
    version = match.group(1)
    family = []
    for row in record.get("models", []):
        label = re.fullmatch(r"Claude Opus ([0-9]+(?:\.[0-9]+)*) \(([^()]+)\)",
                             row.get("name", ""), re.I)
        if label and label.group(1) == version:
            family.append((row, label.group(2).lower()))
    matches = [row for row, level in family if level == token]
    if len(matches) != 1:
        return None, {"ok": False, "reason": "unsupported_effort" if family and not matches else "unsupported_model",
                      "detail": "agy Opus shorthand has no unique bundled version/effort match"}
    row = matches[0]
    if row.get("routable_id") != ROAD_PREFIX["agy"] + row.get("id", ""):
        return None, {"ok": False, "reason": "unsupported_model",
                      "detail": "agy Opus shorthand has an invalid bundled route"}
    return row["routable_id"], None


def _load_cached(provider, profile):
    try:
        path, _status_path, _lock_path = _cache_paths(provider, profile)
        if not path.exists():
            return None, None
        wrapper = _json_loads(path.read_text(encoding="utf-8"))
        expected = _scope_fingerprint(provider, profile)
        if (not isinstance(wrapper, dict) or wrapper.get("provider") != provider
                or wrapper.get("scope_fingerprint") != expected):
            raise CatalogError("cache_scope_mismatch", provider)
        record = validate_record(provider, wrapper.get("record"))
        last_success = wrapper.get("last_success_at")
        if _utc_datetime(last_success) is None:
            raise CatalogError("cache_invalid", provider)
        return record, {"last_success_at": last_success}
    except CatalogError as exc:
        return None, {"reason": exc.reason}
    except (OSError, ValueError, TypeError):
        return None, {"reason": "cache_invalid"}


def _read_last_result(provider, profile):
    try:
        _path, status_path, _lock_path = _cache_paths(provider, profile)
        status = _json_loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict) or status.get("provider") != provider:
            return None
        return {key: status.get(key) for key in
                ("status", "reason", "finished_at", "last_success_at") if key in status}
    except (CatalogError, OSError, ValueError, TypeError):
        return None


def _write_status(provider, profile, status, reason=None, last_success_at=None):
    _path, status_path, _lock_path = _cache_paths(provider, profile, create=True)
    value = {"schema_version": SCHEMA_VERSION, "provider": provider,
             "status": status, "finished_at": _utc_now()}
    if reason:
        value["reason"] = str(reason)
    if last_success_at:
        value["last_success_at"] = last_success_at
    _atomic_write(status_path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _save_last_good(provider, profile, record, diagnostics=None):
    path, status_path, lock_path = _cache_paths(provider, profile, create=True)
    fingerprint = _scope_fingerprint(provider, profile, create=True)
    last_success = _utc_now()
    wrapper = {"schema_version": SCHEMA_VERSION, "provider": provider,
               "scope_fingerprint": fingerprint, "last_success_at": last_success,
               "record": validate_record(provider, record)}
    encoded = (json.dumps(wrapper, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    with _provider_lock(lock_path):
        _atomic_write(path, encoded)
        try:
            _write_status(provider, profile, "ok", last_success_at=last_success)
        except Exception:
            if diagnostics is None:
                raise
            diagnostics["status_diagnostic"] = "status_write_failed"
    return last_success


def _availability_for_default(provider, record):
    from providers import config
    default_ids = {
        "cursor": config.DEFAULT_CURSOR_MODEL,
        "agy": config.DEFAULT_MODEL,
        "codex": config.DEFAULT_CODEX_MODEL,
        "claude": config.DEFAULT_CLAUDE_MODEL,
        "grok": config.DEFAULT_GROK_MODEL,
    }
    wanted = default_ids[provider]
    return any(wanted in (row["id"], row["name"], row["routable_id"])
               for row in record.get("models", []))


def _drift(baseline, current, provider):
    before = {row["id"]: row for row in baseline.get("models", [])}
    after = {row["id"]: row for row in current.get("models", [])}
    effort_changes = {}
    for model_id in sorted(set(before) & set(after)):
        old = before[model_id].get("supported_efforts", [])
        new = after[model_id].get("supported_efforts", [])
        if old != new:
            effort_changes[model_id] = {"bundled": old, "local": new}
    defaults = {"bundled_available": _availability_for_default(provider, baseline),
                "local_available": _availability_for_default(provider, current)}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    return {"added": added, "removed": removed, "effort_changes": effort_changes,
            "default_availability": defaults,
            "drift": bool(added or removed or effort_changes
                          or defaults["bundled_available"] != defaults["local_available"])}


def status(provider="all", profile="default", stale_after_s=DEFAULT_STALE_AFTER_S,
           details=False):
    try:
        selected = PROVIDERS if provider in (None, "all") else (provider,)
        if any(item not in PROVIDERS for item in selected):
            raise CatalogError("unknown_provider")
        bundle = _load_bundle()
        rows = []
        now = _datetime.datetime.now(_datetime.timezone.utc)
        for name in selected:
            local, cache_info = _load_cached(name, profile)
            record = local or bundle.get(name)
            observed = _utc_datetime(record.get("observed_at")) if record else None
            age = max(0, int((now - observed).total_seconds())) if observed else None
            last = _read_last_result(name, profile)
            base = bundle.get(name)
            entry = {
                "provider": name,
                "source": record.get("source") if record else None,
                "origin": "local" if local else "bundled" if record else None,
                "observed_at": record.get("observed_at") if record else None,
                "age_seconds": age,
                "stale": age > int(stale_after_s) if age is not None else None,
                "last_success_at": (cache_info or {}).get("last_success_at"),
                "last_refresh": last,
                "model_count": len(record.get("models", [])) if record else 0,
                "availability": record.get("availability") if record else "unknown",
                "baseline_drift": _drift(base, local, name) if base and local else None,
                "cache_diagnostic": (cache_info or {}).get("reason"),
                "discovery": adapter_coverage(name),
            }
            if details:
                entry["models"] = list(record.get("models", [])) if record else []
            rows.append(entry)
        return {"ok": True, "profile": _profile_name(profile),
                "details": bool(details), "catalogs": rows}
    except CatalogError as exc:
        return exc.envelope()


def check(provider="all", profile="default"):
    try:
        selected = PROVIDERS if provider in (None, "all") else (provider,)
        if any(item not in PROVIDERS for item in selected):
            raise CatalogError("unknown_provider")
        bundle = _load_bundle()
        rows, unusable = [], []
        for name in selected:
            local, cache_info = _load_cached(name, profile)
            if local is None:
                reason = (cache_info or {}).get("reason") or "last_good_missing"
                rows.append({"provider": name, "ok": False, "reason": reason})
                unusable.append(name)
                continue
            rows.append({"provider": name, "ok": True,
                         **_drift(bundle[name], local, name)})
        drifted = any(row.get("drift") for row in rows)
        return {"ok": not unusable, "profile": _profile_name(profile),
                "unusable": unusable, "drift": drifted, "providers": rows,
                "exit_code": 2 if unusable else 3 if drifted else 0}
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


def _cursor_effort_candidates(model_ids, legacy_overrides=None):
    ids = set(model_ids)
    result = {}
    overrides = legacy_overrides or {}
    for (base, effort), slug in overrides.items():
        if slug in ids and effort in CURSOR_EFFORTS:
            result.setdefault(base, {})[effort] = slug
    candidates = {}
    for slug in sorted(ids):
        base_id = slug[:-5] if slug.endswith("-fast") else slug
        fast = slug.endswith("-fast")
        matches = []
        for spelling, effort in _EFFORT_SUFFIXES:
            suffix = "-" + spelling
            if base_id.endswith(suffix) and len(base_id) > len(suffix):
                matches.append((base_id[:-len(suffix)], effort, fast))
        if len(matches) != 1:
            continue
        base, effort, fast = matches[0]
        if base in ids:
            continue
        candidates.setdefault(base, {}).setdefault(effort, []).append((slug, fast))
    for base, by_effort in candidates.items():
        if len(by_effort) < 2:
            continue
        for effort, variants in by_effort.items():
            nonfast = [slug for slug, fast in variants if not fast]
            chosen = nonfast if nonfast else [slug for slug, _fast in variants]
            if len(chosen) == 1:
                result.setdefault(base, {}).setdefault(effort, chosen[0])
    return result


def cursor_effort_map(record, legacy_overrides=None):
    explicit = record.get("effort_slugs") if isinstance(record, dict) else None
    if isinstance(explicit, dict):
        return {base: dict(tokens) for base, tokens in explicit.items()}
    ids = [row.get("id") for row in record.get("models", []) if isinstance(row, dict)]
    return _cursor_effort_candidates(ids, legacy_overrides=legacy_overrides)


def cursor_effort_slug(base, effort, profile=None, legacy_overrides=None):
    """Return an exact observed Cursor effort slug from local ∪ bundle data."""
    if effort is None:
        return base, None
    profile = active_profile() if profile is None else profile
    try:
        bundle = _load_bundle().get("cursor")
    except CatalogError:
        bundle = None
    try:
        local, _info = _load_cached("cursor", profile)
    except CatalogError:
        local = None
    maps = []
    if bundle:
        maps.append(cursor_effort_map(bundle, legacy_overrides))
    if local:
        maps.append(cursor_effort_map(local, legacy_overrides))
    for mapping in maps:
        slug = mapping.get(base, {}).get(effort)
        if slug:
            return slug, None
    return None, {"ok": False, "reason": "unsupported_effort",
                  "detail": "cursor model %s has no observed %s slug" % (base, effort)}


def _record_from_rows(provider, rows, *, source, observed_at, cli_version=None,
                      availability="provider_advertised", source_note=None):
    record = {"provider": provider, "source": source, "observed_at": observed_at,
              "availability": availability, "models": rows}
    if cli_version:
        record["cli_version"] = cli_version
    if source_note:
        record["source_note"] = source_note
    if provider == "cursor":
        from providers.config import CURSOR_EFFORT_SLUG_OVERRIDES
        effort_slugs = _cursor_effort_candidates(
            [row["id"] for row in rows],
            legacy_overrides=CURSOR_EFFORT_SLUG_OVERRIDES)
        record["effort_slugs"] = effort_slugs
        by_slug = {}
        for by_effort in effort_slugs.values():
            for effort, slug in by_effort.items():
                by_slug.setdefault(slug, set()).add(effort)
        for row in rows:
            row["supported_efforts"] = sorted(
                by_slug.get(row["id"], ()),
                key=lambda e: ("minimal", "low", "medium", "high", "xhigh", "max").index(e))
            row["efforts_known"] = row["id"] in by_slug
    return validate_record(provider, record)


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
    record = _record_from_rows("cursor", rows, source="live",
                               observed_at=observed_at or _utc_now(),
                               cli_version=cli_version)
    return record


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
    return _record_from_rows("agy", rows, source="live",
                              observed_at=observed_at or _utc_now(),
                              cli_version=cli_version)


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


def parse_grok_output(output, *, observed_at=None, cli_version=None):
    if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_STDOUT_BYTES:
        raise CatalogError("output_cap", "grok")
    if output and not output.endswith("\n"):
        raise CatalogError("incomplete_output", "grok")
    rows = []
    saw_header = False
    default_id = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "You are not authenticated.":
            raise CatalogError("auth_error", "grok")
        if stripped == "You are logged in with grok.com.":
            continue
        if not stripped:
            continue
        if stripped.startswith("Default model: "):
            default_id = stripped.split(":", 1)[1].strip()
            continue
        if stripped == "Available models:":
            if saw_header:
                raise CatalogError("malformed_output", "grok")
            saw_header = True
            continue
        if not saw_header:
            raise CatalogError("malformed_output", "grok")
        match = re.fullmatch(r"[*-]\s+([^\s]+)(?:\s+\(default\))?", stripped)
        if not match:
            raise CatalogError("malformed_output", "grok")
        model_id = match.group(1)
        rows.append({"id": model_id, "name": model_id,
                     "routable_id": ROAD_PREFIX["grok"] + model_id,
                     "supported_efforts": [], "efforts_known": False})
    if not saw_header or not rows or (default_id and default_id not in {r["id"] for r in rows}):
        raise CatalogError("incomplete_output", "grok")
    return _record_from_rows("grok", rows, source="live",
                              observed_at=observed_at or _utc_now(),
                              cli_version=cli_version, availability="unknown",
                               source_note="vendor_registry_not_entitlement")


def _effort_values(metadata, keys, provider):
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, list):
            raise CatalogError("effort_metadata_invalid", provider)
        result = []
        for entry in value:
            if isinstance(entry, str):
                token = entry
            elif isinstance(entry, dict):
                token_keys = ("reasoningEffort", "effort", "value")
                declared = [entry[name] for name in token_keys if name in entry]
                if len(declared) != 1 or not isinstance(declared[0], str):
                    raise CatalogError("effort_metadata_invalid", provider)
                token = declared[0]
            else:
                raise CatalogError("effort_metadata_invalid", provider)
            if token not in EFFORTS:
                raise CatalogError("effort_metadata_invalid", provider)
            result.append(token)
        ordered = sorted(set(result), key=lambda effort: (
            "minimal", "low", "medium", "high", "xhigh", "max", "ultra").index(effort))
        return ordered, True
    return [], False


def _metadata_row(provider, value, id_keys, name_keys, effort_keys, default_keys=(),
                  visibility_key=None):
    model_id = next((value.get(key) for key in id_keys
                     if isinstance(value.get(key), str) and value.get(key)), None)
    name = next((value.get(key) for key in name_keys
                 if isinstance(value.get(key), str) and value.get(key)), None)
    if model_id is None:
        raise CatalogError("model_id_missing", provider)
    efforts, efforts_known = _effort_values(value, effort_keys, provider)
    supplied_defaults = [value[key] for key in default_keys if key in value]
    if any(not isinstance(default, str) or default not in EFFORTS
           for default in supplied_defaults):
        raise CatalogError("effort_metadata_invalid", provider)
    if len(set(supplied_defaults)) > 1:
        raise CatalogError("effort_metadata_invalid", provider)
    default = supplied_defaults[0] if supplied_defaults else None
    row = {"id": model_id, "name": name or model_id,
           "routable_id": ROAD_PREFIX[provider] + model_id,
           "supported_efforts": efforts, "efforts_known": efforts_known}
    if default is not None:
        if not efforts_known or default not in efforts:
            raise CatalogError("effort_metadata_invalid", provider)
        row["default_effort"] = default
    if visibility_key and isinstance(value.get(visibility_key), bool):
        row["visibility"] = "hidden" if value[visibility_key] else "list"
    return row


def parse_codex_result(result, *, observed_at=None, cli_version=None):
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise CatalogError("malformed_response", "codex")
    if not result["data"]:
        raise CatalogError("empty_output", "codex")
    if any(not isinstance(item, dict) for item in result["data"]):
        raise CatalogError("model_entry_invalid", "codex")
    rows = [_metadata_row("codex", item, ("model", "id"), ("displayName", "name"),
                          ("supportedReasoningEfforts",),
                          ("defaultReasoningEffort",), "hidden")
            for item in result["data"]]
    return _record_from_rows("codex", rows, source="live",
                              observed_at=observed_at or _utc_now(),
                              cli_version=cli_version)


def parse_claude_models(models, *, observed_at=None, cli_version=None):
    if not isinstance(models, list) or not models:
        raise CatalogError("empty_output", "claude")
    rows = [_metadata_row("claude", item, ("value", "id"),
                          ("displayName", "display_name", "name"),
                          ("supportedEffortLevels",), visibility_key=None)
            for item in models if isinstance(item, dict)]
    if len(rows) != len(models):
        raise CatalogError("model_entry_invalid", "claude")
    return _record_from_rows("claude", rows, source="live",
                              observed_at=observed_at or _utc_now(),
                              cli_version=cli_version)


def parse_claude_control_events(output, *, observed_at=None, cli_version=None):
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
    return parse_claude_models(found[0], observed_at=observed_at,
                               cli_version=cli_version)


def adapter_coverage(provider):
    if provider in PROVIDERS:
        return {"status": "automatic_available", "manual_import": True}
    return {"status": "discovery_unavailable", "manual_import": True}


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


def _require_verified_metadata_fence(metadata, provider):
    """Refuse metadata discovery unless the shared host fence proved its wrap."""
    if not isinstance(metadata, dict) or metadata.get("fence") != "seatbelt-fs":
        raise CatalogError("discovery_unavailable", provider)


def _query_codex_models(binary, *, env, cwd, timeout_s):
    try:
        argv, child_env, metadata = __import__("providers.host_fence", fromlist=["prepare_host_fence"]).prepare_host_fence(
            [binary, "app-server"], provider="codex", workdir=cwd,
            read_only=True, env=env,
            config_dir=env.get("CODEX_HOME"))
    except Exception as exc:
        reason = getattr(getattr(exc, "envelope", None), "get", lambda *_: None)("reason")
        raise CatalogError(reason or "fence_unavailable", "codex")
    _require_verified_metadata_fence(metadata, "codex")
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
        cli_version = None
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
        return parse_codex_result({"data": models}, observed_at=_utc_now(),
                                  cli_version=cli_version)
    finally:
        _terminate_process(proc)


def _spawn_one_shot(provider, argv, profile, timeout_s, *, config_dir=None,
                    cursor_config=False, combine_stderr=False):
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
        _require_verified_metadata_fence(metadata, provider)
        raw = _run_bounded(fenced_argv, env=child_env, cwd=cwd,
                           timeout_s=timeout_s, provider=provider,
                           combine_stderr=combine_stderr)
        return _decode_output(raw, provider)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
        if temp_config:
            shutil.rmtree(temp_config, ignore_errors=True)


def _discover_cursor(profile, timeout_s):
    from providers.cursor import resolve_bin
    binary = resolve_bin()
    if not binary:
        raise CatalogError("cli_missing", "cursor")
    output = _spawn_one_shot("cursor", [binary, "--list-models"], profile,
                             timeout_s, cursor_config=True)
    return parse_cursor_output(output)


def _discover_agy(profile, timeout_s):
    from providers.agy import find_agy
    binary = find_agy()
    if not binary:
        raise CatalogError("cli_missing", "agy")
    output = _spawn_one_shot("agy", [binary, "models"], profile, timeout_s)
    return parse_agy_output(output)


def agy_effort_tokens(profile="default", timeout_s=30):
    """Read effort capability from this installed CLI through the shared fence."""
    from providers.agy import find_agy
    binary = find_agy()
    if not binary:
        raise CatalogError("cli_missing", "agy")
    output = _spawn_one_shot("agy", [binary, "--help"], profile, timeout_s,
                             combine_stderr=True)
    return parse_agy_effort_tokens(output)


def _discover_grok(profile, timeout_s):
    binary = vendor_bin.which("grok")
    if not binary:
        raise CatalogError("cli_missing", "grok")
    output = _spawn_one_shot("grok", [os.path.realpath(binary), "models"], profile, timeout_s)
    return parse_grok_output(output)


def _discover_claude(profile, timeout_s):
    from providers.kernel_sandbox import resolve_agent_binary
    binary = resolve_agent_binary("claude")
    if not binary:
        raise CatalogError("cli_missing", "claude")
    argv = [binary, "--print", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose", "--tools", "",
            "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}",
            "--setting-sources", "", "--no-session-persistence",
            "--disable-slash-commands"]
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
        _require_verified_metadata_fence(metadata, "claude")
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
        return parse_claude_control_events("\n".join(lines))
    finally:
        _terminate_process(proc)
        shutil.rmtree(cwd, ignore_errors=True)


def _discover_codex(profile, timeout_s):
    _overrides, child_env, _allowed_env, _selected_home, _endpoints = (
        _catalog_child_environment("codex", profile))
    binary = vendor_bin.which("codex")
    if not binary:
        raise CatalogError("cli_missing", "codex")
    cwd = tempfile.mkdtemp(prefix="omni-catalog-codex-")
    try:
        return _query_codex_models(os.path.realpath(binary), env=child_env,
                                   cwd=cwd, timeout_s=timeout_s)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


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
    previous, info = False, None
    try:
        previous, info = _load_cached(provider, profile)
        previous = previous is not None
    except Exception:
        info = {"reason": "cache_unavailable"}
    result = {"ok": False, "reason": reason,
              "last_success_at": (info or {}).get("last_success_at"),
              "last_good_preserved": previous}
    diagnostic = (info or {}).get("reason")
    if diagnostic:
        result["cache_diagnostic"] = diagnostic
    try:
        _write_status(provider, profile, "error", reason=reason,
                      last_success_at=(info or {}).get("last_success_at"))
    except Exception:
        result["status_diagnostic"] = "status_write_failed"
    return result


def refresh(provider="all", profile="default", timeout_s=30):
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
        try:
            deadline = time.monotonic() + timeout_s
            record = validate_record(name, _DISCOVERERS[name](profile, timeout_s))
            remaining = deadline - time.monotonic()
            version_budget = min(2, int(remaining - 0.25))
            if version_budget > 0:
                try:
                    version = _best_effort_cli_version(name, profile, version_budget)
                except Exception:
                    version = None
                if version:
                    record["cli_version"] = version
            save_diagnostics = {}
            last_success = _save_last_good(name, profile, record,
                                           diagnostics=save_diagnostics)
            results[name] = {"ok": True, "source": record["source"],
                             "observed_at": record["observed_at"],
                             "last_success_at": last_success,
                             "model_count": len(record["models"]),
                             **save_diagnostics}
        except CatalogError as exc:
            results[name] = _refresh_failure_result(name, profile, exc.reason)
        except Exception:
            results[name] = _refresh_failure_result(name, profile, "runtime_error")
        finally:
            # Per-provider profile/cache/status failures must never abort later roads.
            if name not in results:
                results[name] = {"ok": False, "reason": "runtime_error",
                                 "last_good_preserved": False}
    return {"ok": all(item.get("ok") for item in results.values()),
            "profile": profile, "providers": results}


def manual_import(provider, profile, path):
    if provider not in PROVIDERS:
        return {"ok": False, "reason": "unknown_provider", "provider": provider}
    try:
        profile = _profile_name(profile)
        raw = Path(path).read_text(encoding="utf-8")
        value = _json_loads(raw)
        if isinstance(value, dict) and isinstance(value.get("providers"), dict):
            value = value["providers"].get(provider)
        if not isinstance(value, dict) or value.get("provider", provider) != provider:
            raise CatalogError("schema_invalid", provider)
        rows = value.get("models")
        if not isinstance(rows, list):
            raise CatalogError("model_list_invalid", provider)
        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                raise CatalogError("model_entry_invalid", provider)
            model_id = row.get("id", row.get("value", row.get("model")))
            name = row.get("name", row.get("displayName", row.get("display_name", model_id)))
            normalized_rows.append({
                "id": model_id, "name": name,
                "routable_id": row.get(
                    "routable_id", ROAD_PREFIX[provider] + str(model_id)),
                "supported_efforts": row.get("supported_efforts", []),
                "efforts_known": row.get("efforts_known", "supported_efforts" in row),
                **({"default_effort": row["default_effort"]} if "default_effort" in row else {}),
                **({"visibility": row["visibility"]} if "visibility" in row else {}),
            })
        record = {"provider": provider, "source": "manual",
                  "observed_at": value.get("observed_at"),
                  "availability": "unknown", "models": normalized_rows}
        if value.get("cli_version"):
            record["cli_version"] = value["cli_version"]
        if provider == "cursor" and value.get("effort_slugs") is not None:
            record["effort_slugs"] = value["effort_slugs"]
        record = validate_record(provider, record)
        save_diagnostics = {}
        last_success = _save_last_good(provider, profile, record,
                                       diagnostics=save_diagnostics)
        return {"ok": True, "provider": provider, "source": "manual",
                "availability": "unknown", "observed_at": record["observed_at"],
                "last_success_at": last_success, "model_count": len(record["models"]),
                **save_diagnostics}
    except CatalogError as exc:
        return exc.envelope()
    except (OSError, ValueError, TypeError):
        return {"ok": False, "provider": provider, "reason": "import_invalid",
                "detail": "model catalog import_invalid"}


def _output_path(path):
    if not path:
        raise CatalogError("output_required")
    requested = Path(os.path.abspath(os.path.expanduser(str(path))))
    if requested.is_symlink() or (requested.exists() and not requested.is_file()):
        raise CatalogError("output_path_invalid")
    current = requested.parent
    while True:
        if current.is_symlink():
            alias = str(current) in ("/tmp", "/var")
            resolved_alias = os.path.realpath(current)
            if not alias or resolved_alias not in ("/private/tmp", "/private/var"):
                raise CatalogError("output_path_invalid")
        if current.parent == current:
            break
        current = current.parent
    parent = Path(os.path.realpath(requested.parent))
    if not parent.is_dir():
        raise CatalogError("output_path_invalid")
    target = parent / requested.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise CatalogError("output_path_invalid")
    return target


def snapshot(path, provider="all", profile="default"):
    try:
        target = _output_path(path)
        profile = _profile_name(profile)
        selected = PROVIDERS if provider in (None, "all") else (provider,)
        if any(item not in PROVIDERS for item in selected):
            raise CatalogError("unknown_provider")
        bundle = _load_bundle()
        providers = dict(bundle)
        if target.exists():
            try:
                existing = _json_loads(target.read_text(encoding="utf-8"))
                if existing.get("schema_version") != SCHEMA_VERSION or not isinstance(existing.get("providers"), dict):
                    raise CatalogError("snapshot_invalid")
                for name, record in existing["providers"].items():
                    if name not in PROVIDERS:
                        raise CatalogError("snapshot_invalid")
                    providers[name] = validate_record(name, record)
            except CatalogError:
                raise
            except (OSError, ValueError, TypeError):
                raise CatalogError("snapshot_invalid")
        for name in selected:
            local, cache_info = _load_cached(name, profile)
            if cache_info and cache_info.get("reason") not in (None, "cache_identity_unavailable"):
                raise CatalogError("cache_invalid", name)
            record = local or bundle.get(name)
            if record is None:
                raise CatalogError("catalog_unavailable", name)
            providers[name] = record
        output = {"schema_version": SCHEMA_VERSION, "exported_at": _utc_now(),
                  "providers": {name: providers[name] for name in PROVIDERS if name in providers}}
        encoded = (json.dumps(output, sort_keys=True, ensure_ascii=False,
                              indent=2) + "\n").encode("utf-8")
        _atomic_write(target, encoded, mode=0o644)
        return {"ok": True, "output": str(target), "providers": list(output["providers"]),
                "exported_at": output["exported_at"]}
    except CatalogError as exc:
        return exc.envelope()


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
