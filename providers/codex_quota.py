# no-opencc
"""Codex app-server quota probe (account/rateLimits/read).

Spawns ``codex app-server`` over stdio NDJSON, never a model request. The
child is not kernel-fenced and must never be given ``--dangerously-*`` flags.
``accountId``, ``rateLimitUpsell`` and ``rateLimitResetCredits`` are dropped
before anything is returned or logged.
"""
import hashlib
import json
import math
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time

from providers import health
from providers import config as _config
from providers.profiles import resolve_profile, resolve_codex_home, spawn_env
from providers import vendor_bin
from providers.host_fence import allowlisted_child_env


QUOTA_PROBE_TTL_S = 60
STDOUT_CAP_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_S = 15
_CLIENT_NAME = "omni-llm-mcp"
_CLIENT_VERSION = str(getattr(_config, "VERSION", None) or "0")
_QUOTA_LOCK_NAME = "quota.lock"
ENV_QUOTA_PROBE = "OMNI_LLM_QUOTA_PROBE"  # "0" disables the RPC (CI, air-gapped hosts)


def parse_rate_limits(result):
    """Pure parse of a JSON-RPC ``result`` object into the public info dict.

    Type errors yield ``None`` on that field, never an exception. A past
    ``resetsAt`` becomes ``None``. ``primary is None`` leaves used/reset/window
    unset; top-level ``reached`` is still ``rateLimitReachedType is not None``.
    A type naming both windows (``week``/``secondary`` and
    ``primary``/``5h``/``hour``) attributes to both. A type containing
    ``week`` (case-insensitive) is attributed to whichever window is
    weekly from ``window_mins`` (primary or secondary); if neither is
    weekly, to primary. A type containing ``secondary`` is attributed to
    the secondary window. Any other type is attributed to primary.
    ``limits.secondary`` (same three fields plus ``reached``) becomes
    ``info["secondary"]`` when the vendor sends a dict; omitted otherwise.
    """
    info = {
        "used_percent": None,
        "resets_at": None,
        "window_mins": None,
        "reached": False,
        "primary_reached": False,
        "plan": None,
    }
    if not isinstance(result, dict):
        return info
    limits = result.get("rateLimits")
    if not isinstance(limits, dict):
        limits = {}
    reached_type = limits.get("rateLimitReachedType")
    info["reached"] = reached_type is not None
    plan = limits.get("planType")
    if isinstance(plan, str):
        info["plan"] = plan
    primary = limits.get("primary")
    if isinstance(primary, dict):
        info["used_percent"] = _as_float(primary.get("usedPercent"))
        info["window_mins"] = _as_int(primary.get("windowDurationMins"))
        info["resets_at"] = _as_epoch(primary.get("resetsAt"))
    secondary = limits.get("secondary")
    secondary_mins = None
    if isinstance(secondary, dict):
        secondary_mins = _as_int(secondary.get("windowDurationMins"))
    primary_reached, secondary_reached = _attribute_reached(
        reached_type, info["window_mins"], secondary_mins)
    info["primary_reached"] = primary_reached
    if isinstance(secondary, dict):
        info["secondary"] = {
            "used_percent": _as_float(secondary.get("usedPercent")),
            "resets_at": _as_epoch(secondary.get("resetsAt")),
            "window_mins": secondary_mins,
            "reached": secondary_reached,
        }
    return info


def _attribute_reached(reached_type, primary_mins=None, secondary_mins=None):
    """Return ``(primary_reached, secondary_reached)`` from the vendor type.

    Weekly-ness is taken from each window's ``window_mins``, not from
    whether the vendor stored it as primary or secondary. A type that
    names both windows returns ``(True, True)``.
    """
    if reached_type is None:
        return False, False
    text = str(reached_type).lower()
    names_secondary = "week" in text or "secondary" in text
    names_primary = "primary" in text or "5h" in text or "hour" in text
    if names_secondary and names_primary:
        return True, True
    if "week" in text:
        primary_weekly = health._window_is_weekly(primary_mins)
        secondary_weekly = health._window_is_weekly(secondary_mins)
        if secondary_weekly and not primary_weekly:
            return False, True
        if primary_weekly and secondary_weekly:
            return True, True
        return True, False
    if "secondary" in text:
        return False, True
    return True, False


def read_rate_limits(profile="default", timeout_s=DEFAULT_TIMEOUT_S, force=False,
                     codex_home=None):
    """Return ``(info, err)`` with exactly one side set. Never raises.

    Cache hits within ``QUOTA_PROBE_TTL_S`` skip the spawn. Errors are never
    cached. ``force=True`` bypasses and then refreshes the cache.
    """
    try:
        timeout_s = float(timeout_s) if timeout_s is not None else DEFAULT_TIMEOUT_S
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            timeout_s = DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S
    profile_text = "default" if profile is None else str(profile)
    if os.environ.get(ENV_QUOTA_PROBE) == "0":
        return None, _err("disabled", "quota_probe_disabled")
    deadline = time.monotonic() + timeout_s
    overrides, profile_err = (resolve_codex_home(profile_text, codex_home)
                              if codex_home else resolve_profile("codex", profile_text))
    if profile_err:
        return None, _err(
            profile_err.get("reason") or "runtime_error",
            "profile_unresolved",
        )
    from providers.kernel_sandbox import protected_home
    home = os.path.realpath(overrides.get("CODEX_HOME") or os.environ.get("CODEX_HOME")
                            or os.path.join(protected_home(), ".codex"))
    # A label may be remapped within the TTL. Keep only a digest internally;
    # the path never enters quota results or diagnostics.
    home_id = hashlib.sha256(home.encode("utf-8")).hexdigest()
    overrides = {**overrides, "CODEX_HOME": home}
    if not force:
        cached = _cache_get(profile_text, home_id)
        if cached is not None:
            return cached, None
    if not vendor_bin.which("codex"):
        return None, _err("cli_missing", "codex_cli_missing")
    lock_path = os.path.join(health._state_dir(), _QUOTA_LOCK_NAME)
    handle = health.try_exclusive_lock(lock_path, timeout_s)
    if handle is None:
        return None, _err("timeout", "quota_lock_timeout")
    try:
        if not force:
            cached = _cache_get(profile_text, home_id)
            if cached is not None:
                return cached, None
        else:
            _cache_drop(profile_text)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, _err("timeout", "quota_rpc_deadline")
        info, err = _spawn_rate_limits(profile_text, remaining, overrides)
        if info is not None:
            info = _cache_put(profile_text, info, home_id)
            return info, None
        return None, err
    except Exception:
        return None, _err("runtime_error", "quota_probe_failed")
    finally:
        health.release_lock(handle)


def _err(reason, detail):
    return {"ok": False, "reason": reason, "detail": detail}


def _as_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        as_int = int(value)
        if float(as_int) != value:
            return as_int
        return as_int
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_epoch(value):
    ts = health._as_timestamp(value)
    if ts is None:
        return None
    now = time.time()
    if ts <= now:
        return None
    as_int = int(ts)
    return as_int


def _cache_fresh(entry, now=None):
    if not isinstance(entry, dict):
        return False
    checked = health._as_timestamp(entry.get("checked_at"))
    if checked is None:
        return False
    now = time.time() if now is None else now
    return (now - checked) < QUOTA_PROBE_TTL_S


def _info_from_cache_entry(entry):
    info = parse_rate_limits({})
    info["used_percent"] = _as_float(entry.get("used_percent"))
    if entry.get("quota_unknown"):
        info["quota_unknown"] = True
    resets = entry.get("resets_at")
    if resets is not None:
        try:
            info["resets_at"] = int(resets)
        except (TypeError, ValueError):
            info["resets_at"] = None
        else:
            if info["resets_at"] <= time.time():
                info["resets_at"] = None
    info["window_mins"] = _as_int(entry.get("window_mins"))
    info["reached"] = bool(entry.get("reached"))
    if "primary_reached" in entry:
        info["primary_reached"] = bool(entry.get("primary_reached"))
    else:
        info["primary_reached"] = info["reached"]
    plan = entry.get("plan")
    info["plan"] = plan if isinstance(plan, str) else None
    checked = health._as_timestamp(entry.get("checked_at"))
    if checked is not None:
        info["checked_at"] = checked
    secondary = entry.get("secondary")
    if isinstance(secondary, dict):
        sec_resets = secondary.get("resets_at")
        if sec_resets is not None:
            try:
                sec_resets = int(sec_resets)
            except (TypeError, ValueError):
                sec_resets = None
            else:
                if sec_resets <= time.time():
                    sec_resets = None
        info["secondary"] = {
            "used_percent": _as_float(secondary.get("used_percent")),
            "resets_at": sec_resets,
            "window_mins": _as_int(secondary.get("window_mins")),
            "reached": bool(secondary.get("reached")),
        }
    return info


def _cache_get(profile, home_id=None):
    data = health.read_quota_cache()
    entry = data.get(profile)
    if not _cache_fresh(entry) or (home_id is not None and entry.get("home_id") != home_id):
        return None
    return _info_from_cache_entry(entry)


def _cache_drop(profile):
    data = health.read_quota_cache()
    if profile in data:
        data = dict(data)
        data.pop(profile, None)
        health.write_quota_cache(data)


def _cache_put(profile, info, home_id=None):
    stored = {
        "used_percent": info.get("used_percent"),
        "resets_at": info.get("resets_at"),
        "window_mins": info.get("window_mins"),
        "reached": bool(info.get("reached")),
        "primary_reached": bool(info.get("primary_reached", info.get("reached"))),
        "plan": info.get("plan") if isinstance(info.get("plan"), str) else None,
        "checked_at": time.time(),
    }
    if info.get("quota_unknown"):
        stored["quota_unknown"] = True
    if home_id is not None:
        stored["home_id"] = home_id
    secondary = info.get("secondary")
    if isinstance(secondary, dict):
        stored["secondary"] = {
            "used_percent": secondary.get("used_percent"),
            "resets_at": secondary.get("resets_at"),
            "window_mins": secondary.get("window_mins"),
            "reached": bool(secondary.get("reached")),
        }
    data = health.read_quota_cache()
    data[profile] = stored
    health.write_quota_cache(data)
    out = dict(info)
    out["checked_at"] = stored["checked_at"]
    return out


def _spawn_rate_limits(profile, timeout_s, overrides=None):
    if overrides is None:
        overrides, profile_err = resolve_profile("codex", profile)
        if profile_err:
            return None, _err(
                profile_err.get("reason") or "runtime_error",
                "profile_unresolved",
            )
    binary = vendor_bin.which("codex")
    if not binary:
        return None, _err("cli_missing", "codex_cli_missing")
    from providers.codex_accounts import account_env
    env = account_env(spawn_env(os.environ, overrides, scrub_siblings=True))
    # Not fenced, but a vendor child all the same: allowlisted environment.
    env = allowlisted_child_env(env, "codex")
    tmpdir = tempfile.mkdtemp(prefix="omni-quota-")
    proc = None
    try:
        proc = subprocess.Popen(
            [binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=tmpdir,
            start_new_session=True,
            bufsize=0,
        )
        return _rpc_session(proc, timeout_s)
    except FileNotFoundError:
        return None, _err("cli_missing", "codex_cli_missing")
    except Exception:
        return None, _err("runtime_error", "quota_spawn_failed")
    finally:
        _kill_group(proc)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _rpc_session(proc, timeout_s):
    deadline = time.time() + timeout_s
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {
                "name": _CLIENT_NAME,
                "title": _CLIENT_NAME,
                "version": _CLIENT_VERSION,
            },
            "capabilities": {},
        },
    }
    initialized = {
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {},
    }
    read_limits = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "account/rateLimits/read",
        "params": {},
    }
    if not _write_msg(proc, init):
        return None, _err("runtime_error", "quota_stdin_closed")
    init_msg, init_err = _read_until_id(proc, 1, deadline)
    if init_err:
        return None, init_err
    if isinstance(init_msg, dict) and "error" in init_msg:
        return None, _rpc_error(init_msg)
    if not _write_msg(proc, initialized):
        return None, _err("runtime_error", "quota_stdin_closed")
    if not _write_msg(proc, read_limits):
        return None, _err("runtime_error", "quota_stdin_closed")
    reply, reply_err = _read_until_id(proc, 2, deadline)
    if reply_err:
        return None, reply_err
    if isinstance(reply, dict) and "error" in reply:
        return None, _rpc_error(reply)
    if not isinstance(reply, dict) or "result" not in reply:
        return None, _err("runtime_error", "malformed_reply")
    info = parse_rate_limits(reply.get("result"))
    result = reply.get("result")
    limits = result.get("rateLimits") if isinstance(result, dict) else None
    if isinstance(limits, dict) and "secondary" in limits and limits["secondary"] is not None and not isinstance(limits["secondary"], dict):
        info["quota_unknown"] = True
    return info, None


def _rpc_error(msg):
    error = msg.get("error") if isinstance(msg, dict) else None
    code = None
    if isinstance(error, dict) and "code" in error:
        try:
            code = int(error.get("code"))
        except (TypeError, ValueError):
            code = None
    if code is None:
        detail = "jsonrpc_error"
    else:
        detail = "jsonrpc_error code=%s" % code
    from providers.errors import classify_cli_error
    message = error.get("message") if isinstance(error, dict) else None
    reason = classify_cli_error(message if isinstance(message, str) else "", "", 1)
    if code == 401:
        reason = "auth_error"
    elif code == 429:
        reason = "rate_limit"
    elif isinstance(error, dict) and error.get("code") == "account_unavailable":
        reason = "account_unavailable"
    return _err(reason, detail)


def _write_msg(proc, obj):
    if proc.stdin is None:
        return False
    try:
        raw = (json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
        proc.stdin.write(raw.encode("utf-8"))
        proc.stdin.flush()
        return True
    except Exception:
        return False


def _read_until_id(proc, want_id, deadline):
    buf = b""
    stdout = proc.stdout
    if stdout is None:
        return None, _err("timeout", "quota_rpc_deadline")
    fd = stdout.fileno()
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None, _err("timeout", "quota_rpc_deadline")
        try:
            ready, _, _ = select.select([fd], [], [], remaining)
        except (OSError, ValueError, select.error):
            return None, _err("timeout", "quota_rpc_deadline")
        if not ready:
            return None, _err("timeout", "quota_rpc_deadline")
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return None, _err("timeout", "quota_rpc_deadline")
        if not chunk:
            return None, _err("timeout", "quota_rpc_deadline")
        buf += chunk
        if len(buf) > STDOUT_CAP_BYTES:
            return None, _err("timeout", "quota_stdout_cap")
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            msg = _parse_line(line)
            if msg is None:
                continue
            if "id" not in msg:
                continue
            if msg.get("id") != want_id:
                continue
            return msg, None


def _parse_line(line):
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _kill_group(proc):
    if proc is None or proc.pid is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait()
    except Exception:
        pass
