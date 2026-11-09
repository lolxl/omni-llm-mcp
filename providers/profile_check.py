# no-opencc
"""Account-profile registry, opt-in discovery, identity, and check.

Pure functions returning dicts. Printing lives in omni_cli.py. Not an MCP
tool. Discovery of ``~/.claude-<name>`` / ``~/.codex-<name>`` is opt-in
(``OMNI_LLM_PROFILE_DISCOVERY=1``) and fail-closed.
"""
import asyncio
import json
import os
import shlex
import shutil
import stat
import sys

from providers.profiles import (
    PROFILE_NAME_RE, RING_ROADS, nested_home_conflict, resolve_profile,
)


DISCOVERY_ENV = "OMNI_LLM_PROFILE_DISCOVERY"
LIVE_PROMPT = "Reply with the single word: pong"
LIVE_TIMEOUT_S = 30

CLAUDE_IDENTITY_KEYS = (
    "emailAddress",
    "organizationName",
    "seatTier",
    "userRateLimitTier",
    "organizationRateLimitTier",
    "hasExtraUsageEnabled",
    "accountUuid",
)

_ENV_PREFIXES = {
    "claude": ("CLAUDE_CONFIG_DIR_",),
    "codex": ("CODEX_HOME_",),
    "grok": ("GROK_HOME_",),
    "cursor": ("OMNI_LLM_CURSOR_CONFIG_DIR_", "CURSOR_API_KEY_"),
}

_DOTDIR = {"claude": ".claude", "codex": ".codex", "grok": ".grok"}
_NESTED_CHECK_ROADS = tuple(_DOTDIR) + ("cursor",)


def _home():
    return os.path.expanduser("~")


def _discovery_on():
    return os.environ.get(DISCOVERY_ENV) == "1"


def _shell_quote(value):
    return shlex.quote(str(value))


def _scalar(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return None


def _row(road, name, source, path, identity=None, state=None, problems=None,
         cooldown=None):
    return {
        "road": road,
        "name": name,
        "source": source,
        "path": path,
        "identity": identity if identity is not None else {},
        "state": state,
        "problems": list(problems or []),
        "cooldown": cooldown,
    }


def _seat_model(road):
    from providers.config import (
        DEFAULT_CLAUDE_MODEL,
        DEFAULT_CURSOR_MODEL,
    )
    if road == "claude":
        return "claude/%s" % DEFAULT_CLAUDE_MODEL
    if road == "codex":
        return "codex"
    if road == "grok":
        from providers.config import DEFAULT_GROK_MODEL
        return DEFAULT_GROK_MODEL
    if road == "cursor":
        return "cursor/%s" % DEFAULT_CURSOR_MODEL
    return None


def _cooldown(road, name):
    model = _seat_model(road)
    if not model:
        return None
    from providers import health
    return health.seat_kind(model, name)


def _is_safe_regular_file(path):
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()


def _read_json(path):
    if not _is_safe_regular_file(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _oauth_from_claude_json(path):
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    oauth = data.get("oauthAccount")
    if not isinstance(oauth, dict):
        return {}
    out = {}
    for key in CLAUDE_IDENTITY_KEYS:
        if key not in oauth:
            continue
        value = _scalar(oauth.get(key))
        if value is not None:
            out[key] = value
    return out


def read_claude_identity(config_dir):
    """Read allowed ``oauthAccount`` keys from ``<dir>/.claude.json``.

    The default login also keeps that object at ``~/.claude.json`` (home
    root); try that file only when *config_dir* is the default ``~/.claude``
    and the in-dir file has no oauthAccount.
    """
    if not config_dir:
        return {}
    ident = _oauth_from_claude_json(os.path.join(config_dir, ".claude.json"))
    if ident:
        return ident
    default_root = os.path.join(_home(), ".claude")
    try:
        same = os.path.realpath(config_dir) == os.path.realpath(default_root)
    except OSError:
        same = False
    if same:
        return _oauth_from_claude_json(os.path.join(_home(), ".claude.json"))
    return {}


def read_codex_identity(codex_home):
    """Read ``tokens.account_id`` only; return last-6 suffix, never tokens."""
    if not codex_home:
        return {}
    data = _read_json(os.path.join(codex_home, "auth.json"))
    if not isinstance(data, dict):
        return {}
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return {}
    account_id = tokens.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        return {}
    suffix = account_id[-6:] if len(account_id) >= 6 else account_id
    return {"account_id_suffix": suffix}


def read_cursor_identity(name):
    """Identity is which cursor selector is set; never the key value."""
    if name in (None, "", "default"):
        config = os.environ.get("OMNI_LLM_CURSOR_CONFIG_DIR")
        key = os.environ.get("CURSOR_API_KEY")
    else:
        suffix = str(name).upper()
        config = os.environ.get("OMNI_LLM_CURSOR_CONFIG_DIR_%s" % suffix)
        key = os.environ.get("CURSOR_API_KEY_%s" % suffix)
    config_set = config is not None and str(config).strip()
    key_set = key is not None and str(key).strip()
    if config_set:
        return {"kind": "config-dir"}, config
    if key_set:
        return {"kind": "api-key"}, None
    return {}, None


def identity_label(road, identity):
    """Return the ledger identity string for *road*, or None.

    Claude: ``emailAddress``. Codex: ``account_id_suffix``. Cursor and
    every other road: None (cursor ``kind`` is a category, not a login).
    Empty or missing values become None.
    """
    ident = identity if isinstance(identity, dict) else {}
    if road == "claude":
        value = ident.get("emailAddress")
    elif road == "codex":
        value = ident.get("account_id_suffix")
    else:
        return None
    if value is None or value == "":
        return None
    return str(value)


def _state_from_identity(road, identity):
    identity = identity or {}
    if road == "grok":
        return "quota_unknown"
    if road == "claude":
        if identity.get("emailAddress") or identity.get("accountUuid"):
            return "identity_cached"
        return "not_logged_in"
    if road == "codex":
        if identity.get("account_id_suffix"):
            return "identity_cached"
        return "not_logged_in"
    if road == "cursor":
        if identity.get("kind"):
            return "identity_cached"
        return "unconfigured"
    return None


def _identity_for(road, name, path):
    if road == "claude":
        return read_claude_identity(path)
    if road == "codex":
        return read_codex_identity(path)
    if road == "cursor":
        ident, _cfg = read_cursor_identity(name)
        return ident
    return {}


def _default_path(road):
    dot = _DOTDIR.get(road)
    if not dot:
        return None
    return os.path.join(_home(), dot)


def _discovered_path(road, name):
    if road not in _DOTDIR:
        return None
    return os.path.join(_home(), ".%s-%s" % (road, name))


def _profile_vars(road, name):
    suffix = str(name).upper()
    if road == "claude":
        return ["CLAUDE_CONFIG_DIR_%s" % suffix]
    if road == "codex":
        return ["CODEX_HOME_%s" % suffix]
    if road == "grok":
        return ["GROK_HOME_%s" % suffix]
    if road == "cursor":
        return [
            "OMNI_LLM_CURSOR_CONFIG_DIR_%s" % suffix,
            "CURSOR_API_KEY_%s" % suffix,
        ]
    return []


def _scan_env_names(road):
    """Return ordered unique profile names declared in env for *road*."""
    found = []
    seen = set()
    prefixes = _ENV_PREFIXES.get(road) or ()
    for key in sorted(os.environ):
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if not suffix or suffix != suffix.upper():
                continue
            name = suffix.lower()
            if name in ("default",) or not PROFILE_NAME_RE.match(name):
                continue
            raw = os.environ.get(key)
            if raw is None or not str(raw).strip():
                continue
            if name not in seen:
                seen.add(name)
                found.append(name)
    return found


def _env_path(road, name):
    if road == "cursor":
        _ident, cfg = read_cursor_identity(name)
        return cfg
    for var in _profile_vars(road, name):
        raw = os.environ.get(var)
        if raw is None or not str(raw).strip():
            continue
        if os.path.isdir(raw):
            return os.path.realpath(raw)
        return raw
    return None


def _shape_discovered(road, name, path):
    """Fail-closed discovery rules. Stop at the first problem code."""
    if not PROFILE_NAME_RE.match(name or ""):
        return ["invalid_name"]
    try:
        st = os.lstat(path)
    except OSError:
        return ["missing"]
    if stat.S_ISLNK(st.st_mode):
        return ["symlink"]
    if not stat.S_ISDIR(st.st_mode):
        return ["not_a_dir"]
    if st.st_uid != os.getuid():
        return ["not_owned"]
    if stat.S_IMODE(st.st_mode) & 0o022:
        return ["writable"]
    home = os.path.realpath(_home())
    rp = os.path.realpath(path)
    default_root = os.path.realpath(os.path.join(home, ".%s" % road))
    if rp == default_root:
        return ["default_root"]
    from providers.kernel_sandbox import _caller_config_dir_rejection
    prefix = ".%s-" % road
    rej = _caller_config_dir_rejection(
        path,
        label="discovered",
        home=home,
        allow_dotdir_prefix=prefix,
        refuse_contains_home=True,
        must_exist=True,
        refuse_default_login=".%s" % road,
    )
    if rej:
        return ["widening"]
    return []


def _shape_env(road, name):
    overrides, err = resolve_profile(road, name)
    if err:
        if err.get("reason") == "profile_config":
            return ["nested_home"]
        return ["unsupported_profile"]
    if road == "cursor":
        return []
    if not overrides:
        return ["unsupported_profile"]
    return []


def _listdirs(home, prefix):
    try:
        names = os.listdir(home)
    except OSError:
        return []
    out = []
    for base in sorted(names):
        if not base.startswith(prefix) or len(base) <= len(prefix):
            continue
        out.append(base)
    return out


def _list_road(road):
    rows = []
    taken_names = set()
    accepted_rp = {}
    cooldown_default = _cooldown(road, "default")

    default_path = _default_path(road)
    default_ident = _identity_for(road, "default", default_path)
    default_state = _state_from_identity(road, default_ident)
    if road == "cursor":
        default_ident, cfg = read_cursor_identity("default")
        default_path = cfg
        default_state = _state_from_identity(road, default_ident)
    rows.append(_row(
        road, "default", "default", default_path,
        identity=default_ident, state=default_state,
        cooldown=cooldown_default,
    ))
    taken_names.add("default")
    if default_path and os.path.isdir(default_path):
        accepted_rp[os.path.realpath(default_path)] = "default"

    for name in _scan_env_names(road):
        path = _env_path(road, name)
        ident = _identity_for(road, name, path)
        if road == "cursor":
            ident, path = read_cursor_identity(name)
        problems = []
        if path and os.path.isdir(path):
            rp = os.path.realpath(path)
            if rp in accepted_rp:
                problems.append("collision")
            else:
                accepted_rp[rp] = name
            if not problems and road in _NESTED_CHECK_ROADS and nested_home_conflict(
                    _profile_vars(road, name)[0], rp, road):
                problems.append("nested_home")
        row = _row(
            road, name, "env", path,
            identity=ident, state=_state_from_identity(road, ident),
            problems=problems, cooldown=_cooldown(road, name),
        )
        rows.append(row)
        taken_names.add(name)

    if not _discovery_on() or road not in _DOTDIR:
        return rows

    home = _home()
    prefix = ".%s-" % road
    for base in _listdirs(home, prefix):
        name = base[len(prefix):]
        path = os.path.join(home, base)
        problems = _shape_discovered(road, name, path)
        if name in taken_names:
            # Env-declared (or default) wins; report collision on that row.
            for row in rows:
                if row["name"] == name:
                    if "collision" not in row["problems"]:
                        row["problems"].append("collision")
                    break
            continue
        path_out = path
        if "not_owned" in problems:
            path_out = None
        ident = {}
        state = None
        if not problems:
            rp = os.path.realpath(path)
            if rp in accepted_rp:
                problems = ["collision"]
            else:
                accepted_rp[rp] = name
                ident = _identity_for(road, name, path)
                state = _state_from_identity(road, ident)
        row = _row(
            road, name, "discovered", path_out,
            identity=ident, state=state, problems=problems,
            cooldown=_cooldown(road, name),
        )
        rows.append(row)
        taken_names.add(name)
    return rows


def list_profiles(roads=None):
    """Return one row dict per profile on *roads* (default: RING_ROADS)."""
    if roads is None:
        wanted = RING_ROADS
    elif isinstance(roads, str):
        wanted = (roads,)
    else:
        wanted = tuple(roads)
    rows = []
    for road in wanted:
        road = str(road or "").strip().lower()
        if road not in RING_ROADS:
            continue
        rows.extend(_list_road(road))
    return rows


def _inject_env(road, name, path):
    """Temporarily declare a discovered profile so resolve_profile can see it."""
    if not name or name == "default" or not path:
        return None, None
    if road == "claude":
        var = "CLAUDE_CONFIG_DIR_%s" % name.upper()
    elif road == "codex":
        var = "CODEX_HOME_%s" % name.upper()
    elif road == "grok":
        var = "GROK_HOME_%s" % name.upper()
    else:
        return None, None
    old = os.environ.get(var)
    missing = var not in os.environ
    os.environ[var] = path
    return var, (None if missing else old)


def _restore_env(var, old):
    if var is None:
        return
    if old is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = old


def _live_one(road, name):
    async def _go():
        if road == "claude":
            from providers.claude import run_claude_subprocess
            from providers.config import DEFAULT_CLAUDE_MODEL
            return await run_claude_subprocess(
                LIVE_PROMPT, DEFAULT_CLAUDE_MODEL, LIVE_TIMEOUT_S,
                account_profile=name, account_pinned=True)
        if road == "codex":
            from providers.codex import run_codex_subprocess
            from providers.config import DEFAULT_CODEX_MODEL
            return await run_codex_subprocess(
                LIVE_PROMPT, DEFAULT_CODEX_MODEL, LIVE_TIMEOUT_S,
                account_profile=name, account_pinned=True)
        if road == "grok":
            from providers.grok import run_grok_subprocess
            from providers.config import DEFAULT_GROK_MODEL
            return await run_grok_subprocess(
                LIVE_PROMPT, DEFAULT_GROK_MODEL, LIVE_TIMEOUT_S, account_profile=name, account_pinned=True)
        if road == "cursor":
            from providers.cursor import run_cursor_subprocess
            from providers.config import DEFAULT_CURSOR_MODEL
            return await run_cursor_subprocess(
                LIVE_PROMPT, DEFAULT_CURSOR_MODEL, LIVE_TIMEOUT_S,
                account_profile=name)
        return {"ok": False, "reason": "unsupported_profile",
                "detail": "no live check for road %s" % road}
    return asyncio.run(_go())


def _shape_one(row):
    road, name, source = row["road"], row["name"], row["source"]
    path = row.get("path")
    if source == "discovered":
        cand = path or _discovered_path(road, name)
        if not cand:
            return ["missing"]
        return _shape_discovered(road, name, cand)
    if source == "env":
        return _shape_env(road, name)
    return []


def _quota_one(row, strict_quota=False):
    """Apply Codex quota (codex only) and refresh cooldown. Return problems.

    Cooling (reached / used_percent>=95) is informational unless *strict_quota*.
    Only an unreadable quota (*err*, other than disabled) is always a problem.
    """
    from providers import health
    road, name = row["road"], row["name"]
    row["cooldown"] = _cooldown(road, name)
    if road != "codex":
        return []
    from providers.codex_quota import read_rate_limits
    info, err = read_rate_limits(name)
    health.apply_codex_quota("codex", name, info, err)
    row["cooldown"] = health.seat_kind("codex", name)
    if err:
        reason = err.get("reason") if isinstance(err, dict) else None
        if reason == "disabled":
            return []
        return ["quota"]
    if strict_quota and isinstance(info, dict):
        used = info.get("used_percent")
        if info.get("reached") or (used is not None and used >= 95):
            return ["quota"]
    return []


def _live_result_state(road, identity, res):
    if res.get("ok"):
        return "session_ok", []
    detail = res.get("detail") or ""
    auth = res.get("reason") == "auth_error" and "Not logged in" in detail
    if auth:
        # keychain_missing is claude-only by design (macOS Keychain concept).
        if road == "claude" and (
                identity.get("emailAddress") or identity.get("accountUuid")):
            return "keychain_missing", ["keychain_missing"]
        return "not_logged_in", ["not_logged_in"]
    return None, ["live_error"]


def _check_one(row, live, strict_quota=False):
    checked = _row(
        row.get("road"), row.get("name"), row.get("source"), row.get("path"),
        identity=dict(row.get("identity") or {}),
        state=row.get("state"),
        problems=[],
        cooldown=row.get("cooldown"),
    )
    shape = _shape_one(checked)
    if shape:
        checked["problems"] = shape
        if "not_owned" in shape:
            checked["path"] = None
        return checked

    ident = _identity_for(
        checked["road"], checked["name"], checked.get("path"))
    if checked["road"] == "cursor":
        ident, cfg = read_cursor_identity(checked["name"])
        if cfg:
            checked["path"] = cfg
    checked["identity"] = ident
    state = _state_from_identity(checked["road"], ident)
    checked["state"] = state
    if state == "not_logged_in":
        checked["problems"] = ["not_logged_in"]
        return checked

    qprobs = _quota_one(checked, strict_quota=strict_quota)
    if qprobs:
        checked["problems"] = qprobs
        return checked

    if not live:
        return checked

    var, old = None, None
    if checked["source"] == "discovered":
        var, old = _inject_env(
            checked["road"], checked["name"], checked.get("path"))
    try:
        res = _live_one(checked["road"], checked["name"])
    finally:
        _restore_env(var, old)
    new_state, lprobs = _live_result_state(
        checked["road"], ident, res or {})
    if new_state:
        checked["state"] = new_state
    checked["problems"] = lprobs
    return checked


def check(profiles, live=False, strict_quota=False):
    """Validate *profiles* rows. Stop at the first failure per profile.

    Order: shape → identity → quota (codex) → live (only if *live*).
    Cooling is informational unless *strict_quota*; live still runs on a
    cooling profile. Returns ``{"ok": bool, "profiles": [...]}``.
    """
    out = []
    ok = True
    for row in profiles or []:
        checked = _check_one(
            row, live=bool(live), strict_quota=bool(strict_quota))
        out.append(checked)
        if checked.get("problems"):
            ok = False
    return {"ok": ok, "profiles": out}


def _quota_payload(info, err):
    if isinstance(info, dict):
        return {
            "used_percent": info.get("used_percent"),
            "resets_at": info.get("resets_at"),
            "window_mins": info.get("window_mins"),
            "reached": info.get("reached"),
            "plan": info.get("plan"),
            "checked_at": info.get("checked_at"),
        }
    entry = {
        "used_percent": None,
        "resets_at": None,
        "window_mins": None,
        "reached": False,
        "plan": None,
        "checked_at": None,
    }
    if isinstance(err, dict):
        entry["reason"] = err.get("reason")
    return entry


def _codex_probe_pool():
    """Ledger pool a codex quota result lands on: the road itself."""
    return "codex"


# value = the ledger pool a road's quota result lands on (the road itself
# for account-level oracles). ``probe_all`` branches on this table and
# takes the pool out of it; ``health`` reads the same table to decide
# ``will_recheck``. Coverage is :func:`probe_covered_roads` (table keys
# intersect ``RING_ROADS``), a separate question.
#
# Scope, so the next road is not added on a false promise: this generalizes
# WHICH KEY a result lands on, not who answers. ``probe_all`` still asks
# ``read_rate_limits``, which is codex's oracle — it resolves a codex profile
# and runs the codex binary. A second road needs its own oracle wired here as
# well as an entry below; only then is the table's half of it one edit.
PROBE_POOLS = {"codex": _codex_probe_pool}


def probe_covered_roads():
    """Roads whose scheduled probe runs the vendor quota oracle.

    Derived from :data:`PROBE_POOLS` so coverage cannot be declared in one
    place and acted on in another, then narrowed to the roads that can
    actually reach ``probe_all``: it iterates ``list_profiles()`` rows, and
    that call drops every road outside ``RING_ROADS``. A table entry for a
    road off the ring names a pool nothing ever visits, so it is not
    coverage. Both facts are read here, so no caller carries the second one.
    """
    return frozenset(road for road in PROBE_POOLS if road in RING_ROADS)


def probe_pool_model(road):
    """Pool named for *road* in :data:`PROBE_POOLS`, else ``None``.

    This resolves a table entry. It does **not** answer coverage: an entry for
    a road outside ``RING_ROADS`` resolves here while
    :func:`probe_covered_roads` excludes it, because ``probe_all`` can never
    receive a row for that road. Anything that needs to know what the
    scheduled probe visits must ask both, the way ``health`` does.

    ``None`` means the road has no entry, or its entry cannot name a pool
    right now — a config import that failed. Neither is a statement about
    coverage. The second is why the read path treats ``None`` as "no key" and
    fails closed; a writer must not inherit that and record a quota result
    against nothing.
    """
    resolve = PROBE_POOLS.get(road)
    if resolve is None:
        return None
    return resolve()


def probe_all(roads=None):
    """Refresh the ledger for every known profile. Never dispatches a prompt.

    Roads in :data:`PROBE_POOLS`: ``read_rate_limits`` + ``apply_codex_quota``
    per profile, applied to the pool that table names for the road. Other
    roads: no request; ``health.maintain`` expires stale flags once.
    Returns ``{road: {name: quota_entry}}`` (names and quota only).
    """
    from providers import health
    from providers.codex_quota import read_rate_limits

    rows = list_profiles(roads=roads)
    quota = {}
    pending = []
    for row in rows:
        road = row.get("road")
        name = row.get("name")
        if not road or not name:
            continue
        quota.setdefault(road, {})
        pool = probe_pool_model(road)
        if pool:
            info, err = read_rate_limits(name)
            health.apply_codex_quota(pool, name, info, err)
            quota[road][name] = _quota_payload(info, err)
        else:
            pending.append((road, name))
    health.maintain()
    for road, name in pending:
        quota[road][name] = {
            "signal": "none",
            "cooldown": _cooldown(road, name),
        }
    return quota


SCHEDULE_LABEL = "com.omni-llm.profiles-probe"
SCHEDULE_PLIST_NAME = "com.omni-llm.profiles-probe.plist"
_PROBE_LOG_NAME = "profiles-probe.log"


def probe_log_path():
    """Absolute log path under ``$OMNI_LLM_STATE_DIR/logs/``."""
    from providers.health import DEFAULT_STATE_DIR, ENV_STATE_DIR
    root = os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR))
    return os.path.join(root, "logs", _PROBE_LOG_NAME)


def schedule_plist_path():
    return os.path.join(
        os.path.expanduser("~"), "Library", "LaunchAgents",
        SCHEDULE_PLIST_NAME)


def _xml_escape(value):
    return (str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


_LAUNCHD_PATH_TAIL = "/usr/bin:/bin:/usr/sbin:/sbin"
_PLIST_ENV_PREFIXES = (
    "CODEX_HOME_",
    "CLAUDE_CONFIG_DIR_",
    "GROK_HOME_",
    "OMNI_LLM_CURSOR_CONFIG_DIR_",
)
_PLIST_ENV_EXACT = ("OMNI_LLM_PROFILE_DISCOVERY", "OMNI_LLM_STATE_DIR")
_CURSOR_KEY_PREFIX = "CURSOR_API_KEY_"
_VENDOR_BINARIES = ("codex", "claude", "grok", "cursor-agent")


def _vendor_bin_dirs():
    """Dirnames of resolved vendor CLIs, first-seen order, realpath'd."""
    dirs = []
    for name in _VENDOR_BINARIES:
        found = shutil.which(name)
        if not found:
            continue
        directory = os.path.dirname(os.path.realpath(found))
        if directory and directory not in dirs:
            dirs.append(directory)
    return dirs


def _schedule_env_pairs():
    """``(key, value)`` pairs for launchd EnvironmentVariables.

    PATH is the vendor-bin dirs prepended to the launchd default. Profile
    prefixes and ``OMNI_LLM_PROFILE_DISCOVERY`` / ``OMNI_LLM_STATE_DIR``
    are copied from the current environment. ``CURSOR_API_KEY_*`` is never
    copied (the plist is mode 0644).
    """
    path_value = ":".join(_vendor_bin_dirs() + [_LAUNCHD_PATH_TAIL])
    pairs = [("PATH", path_value)]
    skipped_cursor_key = False
    extras = []
    for key in os.environ:
        if key.startswith(_CURSOR_KEY_PREFIX):
            skipped_cursor_key = True
            continue
        if key in _PLIST_ENV_EXACT or any(
                key.startswith(prefix) for prefix in _PLIST_ENV_PREFIXES):
            extras.append(key)
    for key in sorted(extras):
        pairs.append((key, os.environ.get(key)))
    return pairs, skipped_cursor_key


def schedule_plist(interval_s, omni_argv, log_path=None):
    """Return launchd plist text for a periodic advisory probe.

    *omni_argv* is the full ProgramArguments list (absolute python +
    absolute omni_cli.py + flags). *interval_s* becomes StartInterval.
    Emits EnvironmentVariables so launchd can find vendor CLIs and profile
    env. Warns on stderr when ``CURSOR_API_KEY_*`` is set (not copied).
    """
    try:
        interval = int(interval_s)
    except (TypeError, ValueError):
        interval = 0
    log_path = log_path or probe_log_path()
    args_xml = []
    for arg in list(omni_argv or []):
        args_xml.append("\t\t<string>%s</string>" % _xml_escape(arg))
    env_pairs, skipped_cursor_key = _schedule_env_pairs()
    env_xml = ["\t<key>EnvironmentVariables</key>", "\t<dict>"]
    for key, value in env_pairs:
        env_xml.append("\t\t<key>%s</key>" % _xml_escape(key))
        env_xml.append("\t\t<string>%s</string>" % _xml_escape(
            "" if value is None else value))
    env_xml.append("\t</dict>")
    if skipped_cursor_key:
        sys.stderr.write(
            "warning: CURSOR_API_KEY_* profiles are not probed by "
            "the scheduled job\n")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
        '<plist version="1.0">',
        "<dict>",
        "\t<key>Label</key>",
        "\t<string>%s</string>" % _xml_escape(SCHEDULE_LABEL),
        "\t<key>ProgramArguments</key>",
        "\t<array>",
    ]
    lines.extend(args_xml)
    lines.extend([
        "\t</array>",
        "\t<key>StartInterval</key>",
        "\t<integer>%d</integer>" % interval,
        "\t<key>StandardOutPath</key>",
        "\t<string>%s</string>" % _xml_escape(log_path),
        "\t<key>StandardErrorPath</key>",
        "\t<string>%s</string>" % _xml_escape(log_path),
    ])
    lines.extend(env_xml)
    lines.extend([
        "</dict>",
        "</plist>",
        "",
    ])
    return "\n".join(lines)


def _is_empty_dir(path):
    """True only for a real (non-symlink) directory with no entries."""
    if not path:
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode):
        return False
    try:
        return len(os.listdir(path)) == 0
    except OSError:
        return False


def _name_taken(road, name):
    for var in _profile_vars(road, name):
        raw = os.environ.get(var)
        if raw is not None and str(raw).strip():
            return True
    if road != "cursor":
        dest = _discovered_path(road, name)
        if dest and os.path.lexists(dest) and not _is_empty_dir(dest):
            return True
    if _discovery_on():
        for row in list_profiles(roads=(road,)):
            if row.get("name") == name and row.get("source") in (
                    "env", "discovered"):
                if _is_empty_dir(row.get("path")):
                    continue
                return True
    return False


def profiles_env_path():
    return os.path.join(_home(), ".config", "omni-llm", "profiles.env")


def persist_env_line(var, value, path=None):
    """Append ``export VAR=value``, deduped by var name. Returns (path, line)."""
    path = path or profiles_env_path()
    line = "export %s=%s" % (var, _shell_quote(value))
    config_root = os.path.join(_home(), ".config")
    if not os.path.isdir(config_root):
        os.mkdir(config_root, 0o755)
    omni_dir = os.path.join(config_root, "omni-llm")
    if not os.path.isdir(omni_dir):
        os.mkdir(omni_dir, 0o700)
    os.chmod(omni_dir, 0o700)
    existing = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = handle.read().splitlines()
    prefix = "export %s=" % var
    out_lines = [ln for ln in existing if not ln.startswith(prefix)]
    out_lines.append(line)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out_lines) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path, line


def add_profile(road, name, persist=False):
    """Create ``~/.<road>-<name>`` (0700) or refuse. Never prints. Never execs."""
    road = str(road or "").strip().lower()
    name = str(name or "").strip()
    if road not in RING_ROADS:
        return {"ok": False, "reason": "profile_config",
                "detail": "unknown road (ring is claude/codex/grok/cursor only)"}
    if not PROFILE_NAME_RE.match(name) or name == "default":
        return {"ok": False, "reason": "unsupported_profile",
                "detail": "account_profile %r does not match %s"
                          % (name, PROFILE_NAME_RE.pattern)}
    if _name_taken(road, name):
        return {"ok": False, "reason": "unsupported_profile",
                "detail": "profile %s/%s already exists" % (road, name)}

    if road == "cursor":
        var = "CURSOR_API_KEY_%s" % name.upper()
        envelope = {
            "ok": True,
            "road": road,
            "name": name,
            "path": None,
            "created": False,
            "login_command": None,
            "login_argv": None,
            "login_env": None,
            "detail": "only %s is supported; no profile directory is created"
                      % var,
            "mcp_env": {var: ""},
            "var": var,
        }
        if persist:
            envelope["ok"] = False
            envelope["reason"] = "unsupported_persist"
            envelope["detail"] = (
                "cursor profiles are CURSOR_API_KEY_<NAME>; nothing to persist")
        return envelope

    path = _discovered_path(road, name)
    created = False
    if os.path.lexists(path):
        if not _is_empty_dir(path):
            return {"ok": False, "reason": "unsupported_profile",
                    "detail": "profile %s/%s already exists" % (road, name)}
    else:
        os.mkdir(path)
        created = True
    os.chmod(path, 0o700)
    real = os.path.realpath(path)
    if road == "claude":
        var = "CLAUDE_CONFIG_DIR_%s" % name.upper()
        login_env = {"CLAUDE_CONFIG_DIR": real}
        login_argv = ["claude", "auth", "login"]
        login_command = "CLAUDE_CONFIG_DIR=%s claude auth login" % _shell_quote(real)
    elif road == "grok":
        var = "GROK_HOME_%s" % name.upper()
        login_env = {"GROK_HOME": real}
        login_argv = ["grok", "login"]
        login_command = "GROK_HOME=%s grok login" % _shell_quote(real)
    else:
        var = "CODEX_HOME_%s" % name.upper()
        login_env = {"CODEX_HOME": real}
        login_argv = ["codex", "login"]
        login_command = "CODEX_HOME=%s codex login" % _shell_quote(real)
    envelope = {
        "ok": True,
        "road": road,
        "name": name,
        "path": real,
        "created": created,
        "login_command": login_command,
        "login_argv": login_argv,
        "login_env": login_env,
        "mcp_env": {var: real},
        "var": var,
    }
    if persist:
        persist_path, persist_line = persist_env_line(var, real)
        envelope["persist_path"] = persist_path
        envelope["persist_line"] = persist_line
    return envelope
