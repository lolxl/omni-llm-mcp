# no-opencc
"""Account-profile resolution: a non-default name must select a real login.

``None``, ``""`` and ``"default"`` keep today's environment. Any other name
either maps onto a provider-specific env var (codex / claude / cursor) or is
refused LOUD so a road without a profile mechanism cannot silently use the
default login.
"""
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar

from providers import known_secrets

PROFILE_NAME_RE = re.compile(r"^[a-z0-9_]+$")

_DEFAULT_NAMES = (None, "", "default")
_NO_PROFILE_ROADS = ("pi", "ollama", "lmstudio")
_DROP_EXACT = (
    "CODEX_HOME",
    "GROK_HOME",
    "CLAUDE_CONFIG_DIR",
    "CURSOR_API_KEY",
    "CURSOR_CONFIG_DIR",
    "OMNI_LLM_CURSOR_CONFIG_DIR",
)
_DROP_SUFFIX_RE = re.compile(
    r"^(CODEX_HOME|GROK_HOME|CLAUDE_CONFIG_DIR|CURSOR_API_KEY|"
    r"OMNI_LLM_CURSOR_CONFIG_DIR|AGY_CONFIG_DIR|"
    r"OPENAI_API_KEY|OPENAI_API_BASE)_[A-Z0-9_]+$"
)

RING_ROADS = ("claude", "codex", "grok", "cursor")
_RING_PREFIX = "OMNI_LLM_PROFILES_"
_ROAD_SIBLING_PREFIXES = {
    "claude": ("CLAUDE_CONFIG_DIR_",),
    "grok": ("GROK_HOME_",),
    "codex": ("CODEX_HOME_",),
    "cursor": ("CURSOR_API_KEY_", "OMNI_LLM_CURSOR_CONFIG_DIR_"),
}


def _unsupported(detail):
    return {"ok": False, "reason": "unsupported_profile", "detail": detail}


def _normalize_name(account_profile):
    """Return (name, err). name is None for the default profile."""
    if account_profile is None or account_profile in _DEFAULT_NAMES:
        return None, None
    name = str(account_profile)
    if name in _DEFAULT_NAMES:
        return None, None
    if not PROFILE_NAME_RE.match(name):
        return None, _unsupported(
            "account_profile %r does not match %s"
            % (account_profile, PROFILE_NAME_RE.pattern)
        )
    return name, None


def resolve_profile(provider, account_profile):
    """Map ``account_profile`` onto child-env overrides for one provider.

    Returns ``(overrides, err)``. Default -> ``({}, None)``. agy and openai
    keep their own mechanisms and are not resolved here (valid names yield
    empty overrides).
    """
    name, err = _normalize_name(account_profile)
    if err:
        return {}, err
    if name is None:
        return {}, None

    provider = str(provider or "").strip().lower()
    if provider in _NO_PROFILE_ROADS:
        return {}, _unsupported(
            "road %s has no profile mechanism" % provider
        )
    if provider == "codex":
        return _resolve_codex(name)
    if provider == "claude":
        return _resolve_claude(name)
    if provider == "grok":
        return _resolve_directory("grok", name)
    if provider == "cursor":
        return _resolve_cursor(name)
    # agy / openai / anything else with its own mechanism: not this module.
    return {}, None


def _config_dir_unsupported(var, resolved, dotdir_prefix, default_leaf):
    """Refusal for a caller-supplied vendor config dir, or None.

    A path inside the home tree must be a dedicated ``~/<dotdir_prefix><name>``:
    the home tree itself and the default login dir are refused outright, and so
    is any deeper path — ``~/Library/Keychains`` is not a config directory, and
    admitting it would hand a fenced run write access to it (issue #102).

    Codex, cursor and claude.
    """
    from providers.kernel_sandbox import (
        _same_path_any_case, home_tree_rejection, protected_home)
    guarded = protected_home()
    reason = home_tree_rejection(resolved, guarded, dotdir_prefix)
    if reason == "is_home":
        return _unsupported(
            "%s must be a directory inside or beside the home tree, not the "
            "home tree itself" % var
        )
    if _same_path_any_case(os.path.join(guarded, default_leaf), resolved):
        return _unsupported(
            "%s is the default login directory and cannot be a profile; use "
            "a dedicated ~/%s<name> directory, or one outside home"
            % (var, dotdir_prefix)
        )
    if reason == "inside_home":
        return _unsupported(
            "%s is inside the home tree; use a dedicated ~/%s<name> "
            "directory, or one outside home" % (var, dotdir_prefix)
        )
    return None


def _nested(a, b):
    """True when realpath *a* is *b* or inside it, in any spelling."""
    from providers.kernel_sandbox import _path_inside_any_case
    return _path_inside_any_case(a, b)


def nested_home_conflict(label, home, provider=None):
    """LOUD ``profile_config`` when *home* collides with another account home.

    *label* is the env var of the home being resolved and *home* its
    realpath. Every other registered home of EVERY provider
    (``CLAUDE_CONFIG_DIR_*``, ``CODEX_HOME_*``, ``GROK_HOME_*``) must be
    disjoint from it, and so must every cursor config dir
    (``OMNI_LLM_CURSOR_CONFIG_DIR[_*]``; a cursor *provider* is compared
    against the account homes only): equal or nested homes cannot be fenced apart (a deny
    on the outer one removes the inner one's allowance, and an allow on the
    outer one exposes the inner). The refusal names the two labels, never
    the paths. ``None`` when disjoint.

    Two labels of the SAME provider that resolve to the very same directory
    are one account listed twice (the ring dedups them to the first), so
    they are not a conflict; the same directory under two providers, or any
    strict containment, is.
    """
    from providers.kernel_sandbox import (
        registered_account_homes, registered_cursor_config_dirs,
    )
    others = list(registered_account_homes())
    if provider != "cursor":
        # Cursor's config dir is fenced as a writable, readable root; an
        # account home equal to or nested with it cannot be walled off.
        others += registered_cursor_config_dirs()
    for other, other_provider, other_home in others:
        if other == label or not other_home:
            continue
        if other_home == home and other_provider == provider:
            continue
        if _nested(home, other_home) or _nested(other_home, home):
            return _profile_config(
                label,
                "account homes %s and %s are equal or nested; every account "
                "needs its own disjoint directory" % (label, other),
            )
    return None


def _resolve_codex(name, raw=None):
    var = "CODEX_HOME_%s" % name.upper()
    if raw is None:
        raw = os.environ.get(var)
    if not raw or not os.path.isdir(raw):
        return {}, _unsupported(
            "%s is unset or not an existing directory" % var
        )
    home = os.path.realpath(raw)
    # The fence re-allows this one directory for READS under the home read
    # wall and for WRITES (providers/codex.py passes it as both), so this
    # admission is the whole containment story: whatever it names becomes
    # writable by a fenced codex run (issue #102).
    err = _config_dir_unsupported(var, home, ".codex-", ".codex")
    if err:
        return {}, err
    err = nested_home_conflict(var, home, "codex")
    if err:
        return {}, err
    return {"CODEX_HOME": home}, None


def _resolve_claude(name, raw=None):
    var = "CLAUDE_CONFIG_DIR_%s" % name.upper()
    if raw is None:
        raw = os.environ.get(var)
    if not raw or not os.path.isdir(raw):
        return {}, _unsupported(
            "%s is unset or not an existing directory" % var
        )
    home = os.path.realpath(raw)
    # The claude rule is now the codex rule (issue #120, Owner 2026-09-15).
    # Home-tree profiles are accepted only as a dedicated ~/.claude-<name>.
    err = _config_dir_unsupported(var, home, ".claude-", ".claude")
    if err:
        return {}, err
    err = nested_home_conflict(var, home, "claude")
    if err:
        return {}, err
    return {"CLAUDE_CONFIG_DIR": home}, None


def _resolve_cursor(name):
    suffix = name.upper()
    config_var = "OMNI_LLM_CURSOR_CONFIG_DIR_%s" % suffix
    key_var = "CURSOR_API_KEY_%s" % suffix
    config_dir = os.environ.get(config_var)
    api_key = os.environ.get(key_var)
    config_set = config_dir is not None and str(config_dir).strip()
    key_set = api_key is not None and str(api_key).strip()
    if config_set:
        raw = os.path.expanduser(str(config_dir).strip())
        if not os.path.isdir(raw):
            return {}, _unsupported(
                "%s is unset or not an existing directory" % config_var
            )
        resolved = os.path.realpath(raw)
        err = _config_dir_unsupported(
            config_var, resolved, ".cursor-", ".cursor")
        if err:
            return {}, err
        err = nested_home_conflict(config_var, resolved, "cursor")
        if err:
            return {}, err
        overrides = {"OMNI_LLM_CURSOR_CONFIG_DIR": raw}
        if key_set:
            overrides["CURSOR_API_KEY"] = api_key
        return overrides, None
    if key_set:
        return {"CURSOR_API_KEY": api_key}, None
    return {}, _unsupported(
        "neither %s nor %s is set" % (config_var, key_var)
    )


def _profile_config(var, detail):
    return {"ok": False, "reason": "profile_config", "detail": "%s: %s" % (var, detail)}


_AUTO_SKIP_STATES = ("not_logged_in", "keychain_missing", "unconfigured")


def _parse_ring_names(raw, var):
    """Parse a comma list of profile names. Lone ``auto`` is a sentinel."""
    parts = [p.strip() for p in str(raw).split(",")]
    if not parts or any(p == "" for p in parts):
        return None, _profile_config(
            var, "empty list or malformed entry (need comma list of profile names)")
    if any(p == "auto" for p in parts):
        if parts == ["auto"]:
            return ["auto"], None
        return None, _profile_config(
            var, "malformed: auto cannot be mixed with explicit names")
    seen = []
    out = []
    for part in parts:
        if not PROFILE_NAME_RE.match(part):
            return None, _profile_config(
                var,
                "name %r does not match %s" % (part, PROFILE_NAME_RE.pattern),
            )
        if part in seen:
            return None, _profile_config(var, "duplicate name %r" % part)
        seen.append(part)
        out.append(part)
    return out, None


def _resolve_auto_ring(road, var):
    """Expand ``auto`` through the registry. Discovery must be on."""
    from providers.profile_check import DISCOVERY_ENV, list_profiles
    if os.environ.get(DISCOVERY_ENV) != "1":
        return None, _profile_config(
            var, "auto needs %s=1 (discovery is off)" % DISCOVERY_ENV)
    usable = []
    for row in list_profiles(roads=(road,)):
        if row.get("road") != road:
            continue
        name = row.get("name")
        if not name:
            continue
        if row.get("state") in _AUTO_SKIP_STATES:
            continue
        if row.get("problems"):
            continue
        usable.append(name)
    names = []
    if "default" in usable:
        names.append("default")
    names.extend(sorted(n for n in usable if n != "default"))
    if not names:
        return None, _profile_config(
            var, "auto: no usable profile in the registry")
    return names, None


def profile_ring(road):
    """Return ``(names, err)`` for ``OMNI_LLM_PROFILES_<ROAD>``.

    Unset on a known road (claude / codex / cursor) → ``(None, None)`` (no
    ring). Unknown road, duplicates, empty entries, mixed ``auto``, or
    ``auto`` with discovery off / an empty usable registry → LOUD
    ``profile_config`` naming the variable. Lone ``auto`` expands through
    ``providers.profile_check.list_profiles`` (convenience, not the
    production contract).
    """
    road = str(road or "").strip().lower()
    var = _RING_PREFIX + road.upper()
    if road not in RING_ROADS:
        return None, _profile_config(
            var, "unknown road (ring is claude/codex/grok/cursor only)")
    raw = os.environ.get(var)
    if raw is None:
        return None, None
    names, err = _parse_ring_names(raw, var)
    if err:
        return None, err
    if names == ["auto"]:
        return _resolve_auto_ring(road, var)
    return names, None


def _load_profile_rings():
    """Return ``(road → names, err)``. One ``profile_ring`` per set env var.

    Unknown roads, malformed lists, and unresolvable ``auto`` (discovery
    off or an empty usable registry) are LOUD regardless of which roads
    have seats on this call.
    """
    rings = {}
    for key in list(os.environ):
        if not key.startswith(_RING_PREFIX) or key == _RING_PREFIX:
            continue
        suffix = key[len(_RING_PREFIX):]
        road = suffix.lower()
        if road not in RING_ROADS:
            return None, _profile_config(
                key, "unknown road (ring is claude/codex/grok/cursor only)")
        if road in ACCOUNT_HOME_KEYS:
            names, _homes, err = account_ring(road)
        else:
            names, err = profile_ring(road)
        if err:
            return None, err
        rings[road] = names
    return rings, None


def validate_profile_ring_env():
    """Scan ``OMNI_LLM_PROFILES_*``. Unknown, malformed, or unresolvable
    ``auto`` → LOUD, else None. Lone ``auto`` is resolved so discovery-off
    is LOUD on every call, not only when that road has a seat.
    """
    _rings, err = _load_profile_rings()
    return err


def _sibling_prefixes(roads):
    prefixes = []
    for road in roads:
        prefixes.extend(_ROAD_SIBLING_PREFIXES.get(road) or ())
    return tuple(prefixes)


def child_env_for_profile(base_env, overrides, *, scrub_siblings=False):
    """Apply profile overrides onto a child env.

    Non-empty *overrides*: copy *base_env*, drop default-login keys and every
    ``<ENV>_<PROFILE>`` sibling, then apply *overrides*. Empty *overrides*
    (the default profile): return *base_env* unchanged unless *scrub_siblings*
    is true — then drop every road's ``<VAR>_<X>`` siblings so a default ring
    seat never carries sibling profile variables.
    """
    if overrides:
        env = dict(base_env)
        for key in list(env):
            if key in _DROP_EXACT or _DROP_SUFFIX_RE.match(key):
                del env[key]
        env.update(overrides)
        known_secrets.register_overrides(overrides)
        return env
    if not scrub_siblings:
        return base_env
    prefixes = _sibling_prefixes(RING_ROADS)
    env = dict(base_env)
    for key in list(env):
        if any(key.startswith(p) for p in prefixes):
            del env[key]
    return env


def spawn_env(base_env, overrides, *, scrub_siblings=False, keep_notify=False):
    """Child env for a spawned process: profile overrides plus notify policy.

    Always returns a new dict (never *base_env* itself) so callers may pass
    ``os.environ`` without in-place mutation. Unless *keep_notify* is true,
    drops ``OMNI_LLM_NOTIFY_CMD`` and sets ``OMNI_LLM_NOTIFY_QUIET=1``.
    """
    env = dict(child_env_for_profile(
        base_env, overrides, scrub_siblings=scrub_siblings))
    if not keep_notify:
        env.pop("OMNI_LLM_NOTIFY_CMD", None)
        env["OMNI_LLM_NOTIFY_QUIET"] = "1"
    return env


def list_profiles(roads=None):
    """List account profiles for *roads* (default: claude/codex/cursor)."""
    from providers.profile_check import list_profiles as _list
    return _list(roads)


def check(profiles, live=False, strict_quota=False):
    """Verify profile rows. Returns ``{"ok": bool, "profiles": [...]}``."""
    from providers.profile_check import check as _check
    return _check(profiles, live=live, strict_quota=strict_quota)


def resolve_codex_home(name, path):
    """Admit a discovered/persisted Codex home without changing os.environ."""
    normalized, err = _normalize_name(name)
    if err:
        return {}, err
    if normalized is None:
        return {"CODEX_HOME": os.path.realpath(path)}, None
    return _resolve_codex(normalized, raw=path)


def codex_account_ring():
    """Codex ring plus discovered home pins, with no auth.json prerequisite.

    Registry shape/path errors still exclude a row. Login/quota readiness is
    decided by the quota RPC, which supports keyring authentication.
    """
    raw = os.environ.get("OMNI_LLM_PROFILES_CODEX")
    if raw is None or raw.strip() != "auto":
        names, err = profile_ring("codex")
        return names, {}, err
    from providers.profile_check import DISCOVERY_ENV, list_profiles
    if os.environ.get(DISCOVERY_ENV) != "1":
        return None, {}, _profile_config(
            "OMNI_LLM_PROFILES_CODEX", "auto needs %s=1 (discovery is off)" % DISCOVERY_ENV)
    homes = {}
    for row in list_profiles(roads=("codex",)):
        name, path = row.get("name"), row.get("path")
        if (row.get("road") != "codex" or not isinstance(name, str)
                or not PROFILE_NAME_RE.fullmatch(name) or not path
                or row.get("problems") or row.get("state") == "unconfigured"):
            continue
        overrides, err = resolve_codex_home(name, path)
        if not err:
            homes[name] = overrides["CODEX_HOME"]
    names = (["default"] if "default" in homes else []) + sorted(
        name for name in homes if name != "default")
    if not names:
        return None, {}, _profile_config("OMNI_LLM_PROFILES_CODEX", "auto: no usable profile in the registry")
    return names, homes, None


ACCOUNT_HOME_KEYS = {"codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR", "grok": "GROK_HOME"}

API_PROFILES_VAR = "OMNI_LLM_API_PROFILES_CLAUDE"


def claude_api_labels():
    """Return ``(labels, err)`` for ``OMNI_LLM_API_PROFILES_CLAUDE``.

    The list marks ring labels whose Claude home authenticates with an API key
    (kind ``api``); every other label is a subscription. The kind is declared
    here, never through a ``CLAUDE_CONFIG_DIR_<LABEL>_*`` spelling, because the
    registry regex would read that as another profile. Unset -> no api labels.
    Malformed, duplicate, ``default`` or unknown labels are LOUD ``profile_config``.
    """
    raw = os.environ.get(API_PROFILES_VAR)
    if raw is None:
        return frozenset(), None
    parts = [part.strip() for part in raw.split(",")]
    if any(part == "" for part in parts):
        return frozenset(), _profile_config(
            API_PROFILES_VAR, "empty list or malformed entry (need comma list of profile names)")
    seen = []
    for part in parts:
        if not PROFILE_NAME_RE.match(part):
            return frozenset(), _profile_config(
                API_PROFILES_VAR, "name %r does not match %s" % (part, PROFILE_NAME_RE.pattern))
        if part == "default":
            return frozenset(), _profile_config(
                API_PROFILES_VAR, "default cannot be declared an API account")
        if part in seen:
            return frozenset(), _profile_config(API_PROFILES_VAR, "duplicate name %r" % part)
        seen.append(part)
    unregistered = [part for part in seen if "CLAUDE_CONFIG_DIR_" + part.upper() not in os.environ]
    if unregistered:
        names, homes, _err = account_ring("claude")
        known = set(names or ()) | set(homes)
        for part in unregistered:
            if part not in known:
                return frozenset(), _profile_config(
                    API_PROFILES_VAR, "unknown profile %r (not in the ring or registry)" % part)
    return frozenset(seen), None

def _resolve_directory(provider, name, raw=None):
    key = ACCOUNT_HOME_KEYS[provider]
    var = "%s_%s" % (key, name.upper())
    raw = os.environ.get(var) if raw is None else raw
    if not raw or not os.path.isdir(raw):
        return {}, _unsupported("%s is unset or not an existing directory" % var)
    home = os.path.realpath(raw)
    err = (_config_dir_unsupported(var, home, ".%s-" % provider, ".%s" % provider)
           or nested_home_conflict(var, home, provider))
    return ({}, err) if err else ({key: home}, None)

def resolve_account_home(provider, name, path):
    """Admit an internal canonical pin without mutating the registry."""
    if provider == "codex":
        return resolve_codex_home(name, path)
    normalized, err = _normalize_name(name)
    if err:
        return {}, err
    if normalized is None:
        return {ACCOUNT_HOME_KEYS[provider]: os.path.realpath(path)}, None
    return _resolve_directory(provider, normalized, raw=path)

def _account_ring_uncached(provider):
    if provider == "codex":
        return codex_account_ring()
    raw = os.environ.get("OMNI_LLM_PROFILES_" + provider.upper())
    if raw is None or raw.strip() != "auto":
        names, err = profile_ring(provider)
        return names, {}, err
    from providers.profile_check import DISCOVERY_ENV, list_profiles
    var = "OMNI_LLM_PROFILES_" + provider.upper()
    if os.environ.get(DISCOVERY_ENV) != "1":
        return None, {}, _profile_config(var, "auto needs %s=1 (discovery is off)" % DISCOVERY_ENV)
    homes = {}
    for row in list_profiles(roads=(provider,)):
        name, path = row.get("name"), row.get("path")
        if (row.get("road") != provider or not isinstance(name, str)
                or not PROFILE_NAME_RE.fullmatch(name) or not path or row.get("problems")):
            continue
        overrides, err = resolve_account_home(provider, name, path)
        if not err:
            homes[name] = overrides[ACCOUNT_HOME_KEYS[provider]]
    names = (["default"] if "default" in homes else []) + sorted(n for n in homes if n != "default")
    if not names:
        return None, {}, _profile_config(var, "auto: no usable profile in the registry")
    return names, homes, None


_ACCOUNT_REGISTRY = ContextVar("omni_account_registry", default=None)

@contextmanager
def account_registry_scope():
    """Snapshot discovery once per request, with no cross-request cache."""
    token = _ACCOUNT_REGISTRY.set({})
    try:
        yield
    finally:
        _ACCOUNT_REGISTRY.reset(token)

def account_ring(provider):
    cache = _ACCOUNT_REGISTRY.get()
    if cache is None:
        return _account_ring_uncached(provider)
    if provider not in cache:
        cache[provider] = _account_ring_uncached(provider)
    names, homes, err = cache[provider]
    return list(names) if names is not None else None, dict(homes), dict(err) if err else None
