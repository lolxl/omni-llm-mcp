# no-opencc
"""Best-effort seat health ledger, Codex quota probe, and half-open recovery."""
import contextlib
import errno
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from providers.config import CURSOR_MODELS_POOL_STEMS, ROUTE_RULES
from providers.cursor import alias_model as _cursor_alias_model
from providers.logutil import log_event
from providers.notify import notify
from providers.route import provider_for_model


ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
ENV_CODEX_SESSIONS_DIR = "OMNI_LLM_CODEX_SESSIONS_DIR"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"
DEFAULT_CODEX_SESSIONS_DIR = "~/.codex/sessions"
DEFAULT_COOLDOWN_S = 45 * 60
HALF_OPEN_MIN_S = 5 * 60
HALF_OPEN_EVERY_S = 10 * 60
CRASH_STREAK_LIMIT = 3
CRASH_STREAK_WINDOW_S = 10 * 60
CRASH_COOLDOWN_S = 5 * 60
_CRASH_REASONS = frozenset(("runtime_error", "signal_exit"))
FLAG_WINDOW_S = 60 * 60
HEALTH_LOCK_WAIT_S = 2.0
QUOTA_LOCK_NAME = "quota.lock"
_LOCK_POLL_S = 0.02
_LOCK_SKIPPED = object()
_RECOVERABLE_SOURCES = frozenset((
    "rate_limit", "quota", "probe", "codex_passive", "codex_quota",
))
# seat_kind never consults probe_after for these; flag() therefore must not
# claim an early probe time. vendor_reset is absent from _RECOVERABLE_SOURCES
# on purpose — apply_codex_quota will not clear it.
# credit_exhausted is held until the user adds credits and clears it (or until
# its optional configured window ends); nothing probes it back early.
_NO_EARLY_PROBE_SOURCES = frozenset((
    "manual", "quota", "vendor_reset", "credit_exhausted"))
_malformed_logged = set()
_probe_coverage_logged = set()
_account_identity_logged = set()
# Cooldown keys always contain `@`; `_leases` never does — collision guard.
_LEASES_KEY = "_leases"
# Weekly quota window: 7 days, minus 60 minutes of slack so a vendor
# window just under 10080 minutes still names ``week``.
_WEEK_WINDOW_MINS = 7 * 24 * 60
_WEEK_WINDOW_SLACK_MINS = 60
_QUOTA_WINDOW_RE = re.compile(r"^(?:week|m[0-9]+)$")
# Roads provider_for_model can return, plus "codex": a bare account key
# is the road itself (provider_for_model("codex") is openai).
_KNOWN_ROADS = frozenset(
    rule["provider"] for rule in ROUTE_RULES) | frozenset(("agy", "codex"))


def _profile_value(profile):
    return "default" if profile is None else str(profile)


def _cursor_slug_for_pool(remainder):
    """Billing slug. ``sol`` and ``fable`` follow the alias file's target."""
    text = str(remainder or "")
    key = text.strip().lower()
    if key in ("sol", "fable"):
        try:
            from providers.aliases import AliasError, effective
            return effective().get("family.%s.cursor" % key)
        except AliasError:
            pass
    return _cursor_alias_model(text)


def pool_key(model, profile):
    """Return the health-ledger key ``<pool>@<profile>`` (issue #72).

    *profile* is the given profile with ``None`` normalized to ``"default"``.
    It is never collapsed: a non-default profile that can reach the ledger
    is a real separate login (unsupported profiles are refused upstream).

    *pool* is the model string, except for two rewrites. A Codex or Claude
    logical family seat (``codex/sol``, ``claude/fable``) uses the same key
    as the concrete model admission sends: the built-in on today's gate, or
    the alias file's target on the catalog gate (the built-in when that file
    is absent or refused). ``cursor/<remainder>``
    (prefix matched case-insensitively) is resolved through the alias file
    first (``sol`` and ``fable``). The resolved slug ``auto`` →
    ``cursor:auto``; a slug that starts with a stem in
    ``CURSOR_MODELS_POOL_STEMS`` → ``cursor:cursor-models``; every other
    ``cursor/`` slug → ``cursor:other-models``.
    """
    model_text = str(model)
    profile_text = _profile_value(profile)
    lower = model_text.lower()
    if lower.startswith("cursor/"):
        remainder = model_text[len("cursor/"):]
        slug = _cursor_slug_for_pool(remainder).lower()
        if slug == "auto":
            pool = "cursor:auto"
        elif slug.startswith(CURSOR_MODELS_POOL_STEMS):
            pool = "cursor:cursor-models"
        else:
            pool = "cursor:other-models"
    else:
        from providers.family import health_model
        pool = health_model(model_text, profile_text) or model_text
    return "%s@%s" % (pool, profile_text)


def _is_quota_window_name(window):
    return isinstance(window, str) and _QUOTA_WINDOW_RE.match(window) is not None


def split_pool_key(key):
    """Return ``(road, kind, name, window, profile)`` for a ledger key.

    ``#<window>`` is stripped from the pool first. Then a three-way shape
    rule: ``:`` → bucket, ``/`` → model, else account. *name* is the
    remainder after the road separator, or ``None`` for an account key.
    *window* is ``None`` when the pool has no suffix. A suffix that is not
    ``week`` or ``m<digits>`` is rejected: *window* is ``None`` and *kind*
    is ``unknown``.
    """
    text = str(key)
    if "@" in text:
        pool, profile = text.rsplit("@", 1)
    else:
        pool, profile = text, "default"
    window = None
    invalid_window = False
    if "#" in pool:
        pool, raw_window = pool.split("#", 1)
        if _is_quota_window_name(raw_window):
            window = raw_window
        else:
            invalid_window = True
    if ":" in pool:
        road, name = pool.split(":", 1)
        kind = "bucket"
    elif "/" in pool:
        road, name = pool.split("/", 1)
        kind = "model"
    else:
        road, name, kind = pool, None, "account"
    if invalid_window:
        kind = "unknown"
        window = None
    return road, kind, name, window, profile


def _is_quota_window_key(key, road, profile):
    """True when *key* is exactly ``<road>#<week|m<digits>>@<profile>``."""
    key_road, _kind, _name, window, key_profile = split_pool_key(key)
    if not window or key_road != road or key_profile != _profile_value(profile):
        return False
    return str(key) == pool_key("%s#%s" % (road, window), profile)


def _quota_window_name(window_mins):
    """Name a non-primary quota window: ``week`` or ``m<mins>``.

    Missing or unparsable duration is unnamed (empty), never ``week``.
    """
    mins = None
    if window_mins is not None and not isinstance(window_mins, bool):
        try:
            mins = int(window_mins)
        except (TypeError, ValueError):
            mins = None
    if mins is None:
        return ""
    if mins >= (_WEEK_WINDOW_MINS - _WEEK_WINDOW_SLACK_MINS):
        return "week"
    return "m%s" % mins


def _window_is_weekly(window_mins):
    """True when *window_mins* is a weekly quota duration."""
    if window_mins is None or isinstance(window_mins, bool):
        return False
    try:
        mins = int(window_mins)
    except (TypeError, ValueError):
        return False
    return mins >= (_WEEK_WINDOW_MINS - _WEEK_WINDOW_SLACK_MINS)


def _covering_keys(model, profile, ledger=None):
    """Ledger keys whose active records cool *model* on *profile*.

    Always includes ``pool_key(model, profile)`` (the cursor bucket key
    when *model* is ``cursor/…``). When *model* has a ``<road>/…`` prefix,
    also the account key ``<road>@profile`` and every present
    ``<road>#<window>@profile`` key whose window is ``week`` or
    ``m<digits>`` and whose parsed road and profile match exactly. A bare
    road name (no ``/``, ``:``, or ``#``) is the account key; window keys
    for that road still cover. Profile isolation is exact.
    """
    profile_text = _profile_value(profile)
    model_text = str(model)
    ordered = []
    seen = set()

    def add(key):
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    add(pool_key(model_text, profile_text))
    road = None
    if "/" in model_text:
        road = model_text.split("/", 1)[0]
        add(pool_key(road, profile_text))
    elif ":" not in model_text and "#" not in model_text:
        road = "grok" if provider_for_model(model_text) == "grok" else model_text
        if road == "grok":
            add(pool_key(road, profile_text))
    if not road:
        return ordered
    if ledger is None:
        ledger = _read_ledger_unlocked()
    prefix = "%s#" % road
    for key in ledger:
        if key == _LEASES_KEY:
            continue
        text = str(key)
        if not text.startswith(prefix):
            continue
        if _is_quota_window_key(text, road, profile_text):
            add(text)
    return ordered


def _state_dir():
    # Resolve this on every call: tests and long-lived servers can change the
    # state root after importing this module.
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR
    ))


def _ledger_path():
    return os.path.join(_state_dir(), "health.json")


def _lock_path():
    return os.path.join(_state_dir(), "health.lock")


def _quota_dir():
    """Directory of the per-profile Codex quota cache files."""
    return os.path.join(_state_dir(), "quota")


def _flag_window_path():
    return os.path.join(_state_dir(), "flag_window.json")


def _as_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    try:
        iso = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _quiet_log_event(event, **fields):
    """Write one diagnostic line; a failing write drops the line.

    The cache paths log while they hold a row's fate (a drop's unlink, a put's
    result): a stderr pipe with no reader must not change what they return.
    """
    try:
        log_event(event, **fields)
    except Exception:
        pass


def _log_malformed(path):
    key = os.path.abspath(path)
    if key in _malformed_logged:
        return
    _malformed_logged.add(key)
    _quiet_log_event("health_error", operation="malformed_file",
                     detail="malformed_state_json")


def _log_probe_coverage_error(detail):
    # Keyed like _log_malformed: a second, different breakage still gets one
    # line, instead of being silent for the life of a long-running server.
    if detail in _probe_coverage_logged:
        return
    _probe_coverage_logged.add(detail)
    log_event("health_error", operation="probe_coverage", detail=detail)


def _log_account_identity_error(detail):
    if detail in _account_identity_logged:
        return
    _account_identity_logged.add(detail)
    log_event("health_error", operation="account_identity", detail=detail)


def _read_json_object(path):
    """Load a JSON object. Missing -> {}; malformed -> {} + log; OSError raises."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        _log_malformed(path)
        return {}
    except TypeError:
        _log_malformed(path)
        return {}
    if not isinstance(data, dict):
        _log_malformed(path)
        return {}
    return data


def _read_ledger():
    data = _read_json_object(_ledger_path())
    return {
        str(key): dict(value)
        for key, value in data.items()
        if isinstance(value, dict)
    }


def _read_ledger_unlocked():
    """Side-effect-free read. Permission errors become an empty view."""
    try:
        return _read_ledger()
    except OSError:
        log_event("health_error", operation="read", detail="state_unreadable")
        return {}


def _atomic_write_json(path, data, allow_nan=True):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".omni-", suffix=".tmp", dir=directory)
    open_fd = fd
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            open_fd = None
            json.dump(data, handle, ensure_ascii=False, sort_keys=True,
                      allow_nan=allow_nan)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if open_fd is not None:
            os.close(open_fd)
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _write_ledger(data):
    _atomic_write_json(_ledger_path(), data)


def try_exclusive_lock(path, timeout_s):
    """Acquire an exclusive flock. Return the open handle, or None on failure."""
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        handle = open(path, "a+b")
    except OSError:
        return None
    deadline = time.time() + max(0.0, float(timeout_s))
    busy = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.time() >= deadline:
                try:
                    handle.close()
                except OSError:
                    pass
                return None
            time.sleep(_LOCK_POLL_S)
        except OSError as exc:
            if exc.errno in busy:
                if time.time() >= deadline:
                    try:
                        handle.close()
                    except OSError:
                        pass
                    return None
                time.sleep(_LOCK_POLL_S)
                continue
            try:
                handle.close()
            except OSError:
                pass
            return None


def release_lock(handle):
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass


def _locked_mutate(operation, mutator):
    handle = try_exclusive_lock(_lock_path(), HEALTH_LOCK_WAIT_S)
    if handle is None:
        log_event("health_skipped", operation=operation)
        return _LOCK_SKIPPED
    dropped = []
    result = _LOCK_SKIPPED
    try:
        try:
            data = _read_ledger()
        except OSError:
            log_event("health_error", operation="read", detail="state_unreadable")
            return _LOCK_SKIPPED
        dropped = _prune(data)
        result = mutator(data)
        _write_ledger(data)
    finally:
        release_lock(handle)
    _log_expired(dropped)
    return result


class StrictReadError(Exception):
    """A state file the quota decision needs could not be read exactly.

    ``reason`` is ``ledger_unreadable`` or ``telemetry_malformed``. No path,
    value or exception text is kept: the message is the reason alone.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _strict_reject_constant(_name):
    raise ValueError("non-finite number")


def _strict_reject_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _strict_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def strict_json_loads(text):
    """Decode *text* as JSON for every strict read of the quota decision.

    A duplicate key at any depth, ``NaN``/``Infinity``/``-Infinity`` and a
    number that is not finite after decode (``1e999``) raise ValueError, as
    does any malformed JSON. Callers map that to their own reason.
    """
    return json.loads(text, object_pairs_hook=_strict_pairs,
                      parse_constant=_strict_reject_constant,
                      parse_float=_strict_reject_float)


class _ReadTooLarge(StrictReadError):
    """A bounded strict read found more bytes than its limit allows."""


def _strict_read_maybe(path, reason, dir_fd=None, max_bytes=None):
    """Return the JSON object at *path*, or ``None`` when the file is absent.

    Unlike :func:`_read_json_object` nothing is forgiven: a read error,
    malformed JSON, a duplicate key, NaN/Infinity or a root that is not an
    object raises :class:`StrictReadError`. A symbolic link at *path*, dangling
    or not, is unreadable: only a regular file is read, and it is opened
    without following a link, so a link swapped in after a check cannot be
    followed. Reads without a lock: every writer replaces the file
    atomically.

    With *dir_fd*, *path* is a name relative to that open directory. With
    *max_bytes* the read is bounded: a file with more bytes raises
    :class:`_ReadTooLarge` (a StrictReadError) without being read whole.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError:
        raise StrictReadError(reason)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise StrictReadError(reason)
        if max_bytes is not None and info.st_size > max_bytes:
            raise _ReadTooLarge(reason)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        if max_bytes is not None and len(raw) > max_bytes:
            raise _ReadTooLarge(reason)
        data = strict_json_loads(raw.decode("utf-8"))
    except StrictReadError:
        raise
    except (OSError, ValueError, RecursionError):
        # ValueError covers JSONDecodeError and UnicodeDecodeError.
        raise StrictReadError(reason)
    finally:
        os.close(fd)
    if not isinstance(data, dict):
        raise StrictReadError(reason)
    return data


def _strict_read_object(path, reason):
    """Return the JSON object at *path*; ``{}`` only when the file is absent.

    See :func:`_strict_read_maybe` for what is refused.
    """
    data = _strict_read_maybe(path, reason)
    return {} if data is None else data


_STRICT_FLAG_MARKERS = ("source", "flagged_at", "probe_after")


def _strict_timestamp_ok(value):
    """True for a present expiry that parses to a finite timestamp."""
    ts = _as_timestamp(value)
    return ts is not None and math.isfinite(ts)


def strict_ledger_snapshot():
    """Return the whole ledger exactly as stored, or raise StrictReadError.

    A missing ``health.json`` is an empty ledger (first run). Every row must
    be an object. A present ``until`` (null included) or a ``credit_hold.until``
    must parse to a finite timestamp; a row with no ``until`` and no flag
    marker (a crash-streak counter) is valid. A symbolic link is unreadable.
    The lenient readers keep their callers; this is a separate path.
    """
    data = _strict_read_object(_ledger_path(), "ledger_unreadable")
    for key, row in data.items():
        if not isinstance(row, dict):
            raise StrictReadError("ledger_unreadable")
        if key == _LEASES_KEY:
            continue
        # A flag row always carries a finite expiry: a present ``until`` of
        # null or any other bad value, or flag markers without one, are
        # unreadable. Only a row with neither (a crash-streak counter) has none.
        if "until" in row:
            if not _strict_timestamp_ok(row["until"]):
                raise StrictReadError("ledger_unreadable")
        elif any(marker in row for marker in _STRICT_FLAG_MARKERS):
            raise StrictReadError("ledger_unreadable")
        if "credit_hold" in row:
            hold = row["credit_hold"]
            if not isinstance(hold, dict) or not _strict_timestamp_ok(
                    hold.get("until")):
                raise StrictReadError("ledger_unreadable")
    return data


def _strict_number(value):
    """True for a finite int or float (a bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _strict_cache_window_ok(window):
    """Basic types of one quota-cache window (the row itself or ``secondary``)."""
    for name in ("used_percent", "resets_at"):
        if window.get(name) is not None and not _strict_number(window[name]):
            return False
    mins = window.get("window_mins")
    if mins is not None and (isinstance(mins, bool) or not isinstance(mins, int)):
        return False
    for name in ("reached", "primary_reached"):
        if name in window and not isinstance(window[name], bool):
            return False
    return True


def _strict_cache_row_ok(row):
    """True for a well-typed quota-cache row.

    An object with a finite-number ``checked_at``, and finite number or null
    ``used_percent`` and ``resets_at``, int or null ``window_mins``, bool
    ``reached``/``primary_reached``, str ``home_id``, str or null ``plan`` and
    bool ``quota_unknown`` when present; ``secondary`` is an object of the
    same types, or absent.
    """
    if not isinstance(row, dict) or not _strict_number(row.get("checked_at")):
        return False
    if "home_id" in row and not isinstance(row["home_id"], str):
        return False
    # The writer stores ``plan`` as a str or null and ``quota_unknown`` only as true.
    if "plan" in row and row["plan"] is not None and not isinstance(row["plan"], str):
        return False
    if "quota_unknown" in row and not isinstance(row["quota_unknown"], bool):
        return False
    windows = [row]
    if "secondary" in row:
        if not isinstance(row["secondary"], dict):
            return False
        windows.append(row["secondary"])
    return all(_strict_cache_window_ok(window) for window in windows)


# One cache file per Codex profile: <state_dir>/quota/<sha256 of the label>.json.
# The file holds the row plus ``"profile": <label>``. A write touches only its
# own file; nothing on these paths takes the ledger lock.

QUOTA_ROW_MAX_BYTES = 64 * 1024     # a row is a few hundred bytes; more is malformed


def _quota_row_path(profile):
    """Return ``(label, name)`` of one profile's cache file.

    *name* is relative to the cache directory. Raises ValueError when the
    label cannot be encoded.
    """
    label = _profile_value(profile)
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return label, digest + ".json"


def _quota_dir_checked():
    """Test seam: called once the cache directory descriptor is verified."""


def _open_quota_dir(create=False):
    """Open the cache directory and return its descriptor, or None when absent.

    The directory is opened once, without following a link at its own name,
    and checked on the descriptor: a real directory owned by the current user.
    Every later read, create, rename and unlink of one operation goes through
    that descriptor, so replacing the name afterwards changes nothing. A link,
    a file or a directory of another user at that name raises
    :class:`StrictReadError`. With *create* an absent directory is made
    (mode 0700) and opened.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(_quota_dir(), flags)
    except FileNotFoundError:
        if not create:
            return None
        os.makedirs(_state_dir(), exist_ok=True)
        try:
            os.mkdir(_quota_dir(), 0o700)
        except FileExistsError:
            pass
        try:
            fd = os.open(_quota_dir(), flags)
        except OSError:
            raise StrictReadError("telemetry_malformed")
    except OSError:
        raise StrictReadError("telemetry_malformed")
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise StrictReadError("telemetry_malformed")
        _quota_dir_checked()
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextlib.contextmanager
def _quota_dir_fd(create=False):
    """Yield the verified cache directory descriptor (None when absent)."""
    fd = _open_quota_dir(create)
    try:
        yield fd
    finally:
        if fd is not None:
            os.close(fd)


def _write_file_at(dir_fd, name, payload):
    """Atomically create or replace *name* in the open directory with *payload*.

    The temp file is created exclusively, without following a link, relative
    to *dir_fd*; it is fsynced, renamed over *name* inside that directory and
    the directory is fsynced; a failed directory fsync raises OSError. A
    failure before the rename leaves no temp file behind.
    """
    temp = ".omni-%s.tmp" % os.urandom(8).hex()
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=dir_fd)
    try:
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(temp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    # The rename is not durable until the directory is synced: a failed sync
    # is a failed write (OSError), whatever state the name is left in.
    os.fsync(dir_fd)


def _write_row_at(dir_fd, name, data):
    """Write the JSON object *data* as one cache file; NaN or infinity raise ValueError."""
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _write_file_at(dir_fd, name, (text + "\n").encode("utf-8"))


def _read_quota_row_at(dir_fd, label, name):
    """Return the row in *name* of the open directory, ``None`` when absent.

    The read is bounded by :data:`QUOTA_ROW_MAX_BYTES`; a larger file is malformed.
    """
    row = _strict_read_maybe(name, "telemetry_malformed", dir_fd=dir_fd,
                             max_bytes=QUOTA_ROW_MAX_BYTES)
    if row is None:
        return None
    if not _strict_cache_row_ok(row) or row.get("profile") != label:
        raise StrictReadError("telemetry_malformed")
    return row


def _is_regular_at(dir_fd, name):
    try:
        return stat.S_ISREG(os.stat(name, dir_fd=dir_fd,
                                    follow_symlinks=False).st_mode)
    except OSError:
        return False


def _note_malformed_row_at(dir_fd, label, name):
    """Log (once) a regular cache file that is malformed; refuse anything else.

    Returns normally when the file is absent, well-formed or a malformed
    regular file (after logging it). A link or a non-regular file raises
    :class:`StrictReadError`. An oversized or unreadable regular file counts
    as malformed: whatever the read does, a regular file is logged (one fixed
    reason) and the caller may go on to replace or unlink it.
    """
    try:
        _read_quota_row_at(dir_fd, label, name)
    except (StrictReadError, OSError, ValueError, RecursionError):
        if not _is_regular_at(dir_fd, name):
            raise StrictReadError("telemetry_malformed")
        _log_malformed(os.path.join(_quota_dir(), name))


def _read_quota_row(profile):
    """Return one profile's row, or ``None`` when its file is absent.

    A present file that is undecodable, not an object, fails the row checks or
    names another label raises ``StrictReadError("telemetry_malformed")``, as
    does a cache directory or file that is not a real directory or regular file.
    """
    try:
        label, name = _quota_row_path(profile)
    except ValueError:
        raise StrictReadError("telemetry_malformed")
    with _quota_dir_fd() as dir_fd:
        if dir_fd is None:
            return None
        return _read_quota_row_at(dir_fd, label, name)


def strict_quota_cache_row(profile):
    """Return one profile's cache row exactly as stored, or raise StrictReadError.

    ``None`` means the profile's file is truly absent. Only this profile's file
    is read: a malformed file of another profile does not matter. The old
    one-file cache is never read.
    """
    return _read_quota_row(profile)


def get_quota_row(profile):
    """Lenient read of one profile's row: ``None`` when absent or unusable."""
    try:
        return _read_quota_row(profile)
    except StrictReadError:
        return None


def _log_quota_cache_error(detail):
    _quiet_log_event("health_error", operation="quota_cache", detail=detail)


def put_quota_row(profile, row):
    """Replace one profile's cache file with *row*; other files are untouched.

    Returns True when written and the rename is durable (the directory was
    fsynced); a failed directory fsync returns False. A NaN or infinity in
    *row*, an unusable cache directory or a profile file that is a link or not
    a regular file writes nothing and logs one fixed line. An existing file
    that is malformed is logged once and then replaced. After a published row
    the old one-file cache, if any, is deleted (best effort, never read).
    """
    try:
        label, name = _quota_row_path(profile)
        with _quota_dir_fd(create=True) as dir_fd:
            _note_malformed_row_at(dir_fd, label, name)
            _write_row_at(dir_fd, name, dict(row, profile=label))
    except StrictReadError:
        _log_quota_cache_error("quota_cache_unsafe_path")
        return False
    except (OSError, ValueError, TypeError):
        _log_quota_cache_error("quota_cache_write_failed")
        return False
    _remove_old_cache_file()
    return True


def drop_quota_row(profile):
    """Unlink one profile's cache file; True when it is gone and the unlink is durable.

    A missing file (or directory) is a success. Any failure, an unlink that
    is refused, an unsafe directory or a link at the file's name, or a failed
    directory fsync after the unlink, returns False: the caller must not
    trust that the old row is gone. The file is never read to decide whether
    to unlink it.
    """
    try:
        label, name = _quota_row_path(profile)
        with _quota_dir_fd() as dir_fd:
            if dir_fd is None:     # a directory absent at the open: nothing to drop
                return True
            # A malformed regular file is logged once before it goes; a link
            # or another non-regular file is refused and left alone.
            _note_malformed_row_at(dir_fd, label, name)
            try:
                os.unlink(name, dir_fd=dir_fd)
            except FileNotFoundError:
                return True
            # An error here, ENOENT included, is not a drop: the unlink is
            # not known to be durable.
            os.fsync(dir_fd)
        return True
    except StrictReadError:
        _log_quota_cache_error("quota_cache_unsafe_path")
    except (OSError, ValueError):
        _log_quota_cache_error("quota_cache_write_failed")
    return False


_OLD_CACHE_NAME = "quota.json"
_old_cache_logged = set()


def _remove_old_cache_file():
    """Best effort: delete the one-file cache this layout replaced.

    Nothing ever reads it. It is unlinked by name relative to an open
    descriptor of the state directory (opened without following a link), so a
    link at that name is removed, never followed. A failure is logged once per
    process and ignored.
    """
    try:
        fd = os.open(_state_dir(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.unlink(_OLD_CACHE_NAME, dir_fd=fd)
        finally:
            os.close(fd)
    except FileNotFoundError:
        return
    except OSError:
        if "removed" not in _old_cache_logged:
            _old_cache_logged.add("removed")
            _log_quota_cache_error("quota_cache_old_file_not_removed")


def credit_hold(entry, now=None):
    """Return the active credit-exhaustion hold of *entry*, or None.

    The hold is evidence kept apart from the generic ``until``/``source``
    pair, which a later or longer flag may take over. It is the
    ``credit_hold`` field ``{"until", "flagged_at"}``; an entry written
    before that field existed (``source`` credit_exhausted, no field) is
    read as a hold lasting until its own ``until``. Expired -> None.
    """
    if not isinstance(entry, dict):
        return None
    now = time.time() if now is None else now
    hold = entry.get("credit_hold")
    if isinstance(hold, dict):
        until = _as_timestamp(hold.get("until"))
        stamp = hold.get("flagged_at")
    elif entry.get("source") == "credit_exhausted":
        until = _as_timestamp(entry.get("until"))
        stamp = entry.get("flagged_at")
    else:
        return None
    if until is None or until <= now:
        return None
    return {"until": until, "flagged_at": stamp}


def _is_active(entry, now=None):
    if not isinstance(entry, dict):
        return False
    now = time.time() if now is None else now
    until = _as_timestamp(entry.get("until"))
    if until is not None and until > now:
        return True
    return credit_hold(entry, now) is not None


def _streak_value(entry):
    if not isinstance(entry, dict):
        return 0
    try:
        n = int(entry.get("crash_streak") or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _streak_only(entry):
    """Keep crash-streak counters after an expired flag is dropped."""
    kept = {"crash_streak": _streak_value(entry)}
    if entry.get("last_crash_at") is not None:
        kept["last_crash_at"] = entry.get("last_crash_at")
    if entry.get("model") is not None:
        kept["model"] = entry.get("model")
    if entry.get("profile") is not None:
        kept["profile"] = entry.get("profile")
    if entry.get("account") is not None:
        kept["account"] = entry.get("account")
    return kept


def _prune(data, now=None):
    """Drop expired flags in place.

    An expired ``source=crash`` flag resets the streak to 0 (the whole
    entry is dropped). Streak counters on any other expired flag survive
    so a later crash can still count inside ``CRASH_STREAK_WINDOW_S``.
    """
    now = time.time() if now is None else now
    dropped = []
    for key in list(data):
        if key == _LEASES_KEY:
            # No `until` on this bucket, so `_is_active` is False and the
            # loop would otherwise pop the whole map. Drop expired lease
            # entries here; never emit for them. A non-dict bucket is
            # left alone.
            leases = data.get(key)
            if isinstance(leases, dict):
                _prune_expired_leases(leases, now)
                _store_lease_bucket(data, leases)
            continue
        value = data.get(key)
        if _is_active(value, now):
            if (isinstance(value, dict) and "credit_hold" in value
                    and credit_hold(value, now) is None):
                value.pop("credit_hold", None)
            continue
        if isinstance(value, dict) and value.get("source") == "crash":
            dropped.append((key, data.pop(key)))
            continue
        streak = _streak_value(value)
        had_until = isinstance(value, dict) and value.get("until") is not None
        if streak > 0:
            if had_until:
                dropped.append((key, value))
            data[key] = _streak_only(value)
            continue
        dropped.append((key, data.pop(key)))
    return dropped


def _log_expired(dropped):
    """Announce pruned entries after a real ledger write. No-op when empty."""
    if not dropped:
        return
    now = time.time()
    for key, entry in dropped:
        model = None
        profile = None
        until = None
        if isinstance(entry, dict):
            model = entry.get("model")
            profile = entry.get("profile")
            until = entry.get("until")
        if model is None or profile is None:
            model_text, profile_text = str(key).rsplit("@", 1) if "@" in str(key) \
                else (str(key), "default")
            if model is None:
                model = model_text
            if profile is None:
                profile = profile_text
        # Two events for one occurrence, deliberately. seat_recovered predates
        # the seam and goes to stderr, where callers already grep for it;
        # seat_cleared is the machine seam. Do not deduplicate them.
        log_event("seat_recovered", model=model, profile=profile, source="expired")
        repeat = _flag_window_count(str(key), now)
        _emit_seat("seat_cleared", model, profile, "expired", until, repeat,
                   window=split_pool_key(key)[3])


def _whole_timestamp(value):
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    as_float = float(value)
    as_int = int(as_float)
    if as_float == as_int:
        return as_int
    return as_float


_RESET_ABS_KEYS = ("resets_at", "resetsAt", "resets-at")
_RESET_DELAY_KEYS = ("retry_after", "retryAfter", "retry-after")
_RETRY_AFTER_RE = re.compile(r"(?i)retry-after:\s*(\S.*?)(?:\r?\n|$)")
_RESETS_AT_RE = re.compile(r"(?i)resets\s+at\s+(\S+)")
_TRY_AGAIN_RE = re.compile(
    r"(?i)try\s+again\s+in\s+(\d+)\s*(seconds?|minutes?)\b"
)
_HM_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def _iter_json_objects(text):
    decoder = json.JSONDecoder()
    idx = 0
    found = 0
    n = len(text)
    while idx < n and found < 8:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            found += 1
            yield obj
        idx = end if end > start else start + 1


def _delay_seconds(value, now):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return now + float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        amount = float(text)
    except (TypeError, ValueError):
        return _as_timestamp(text)
    if amount < 0:
        return None
    return now + amount


def _timestamp_from_mapping(obj, now):
    if not isinstance(obj, dict):
        return None
    for key in _RESET_ABS_KEYS:
        if key not in obj:
            continue
        ts = _as_timestamp(obj[key])
        if ts is not None:
            return ts
    for key in _RESET_DELAY_KEYS:
        if key not in obj:
            continue
        ts = _delay_seconds(obj[key], now)
        if ts is not None:
            return ts
    return None


def _walk_reset_mapping(obj, now, depth=0):
    if depth > 5 or not isinstance(obj, dict):
        return None
    ts = _timestamp_from_mapping(obj, now)
    if ts is not None:
        return ts
    for value in obj.values():
        if isinstance(value, dict):
            ts = _walk_reset_mapping(value, now, depth + 1)
            if ts is not None:
                return ts
        elif isinstance(value, list) and depth < 3:
            for item in value:
                if isinstance(item, dict):
                    ts = _walk_reset_mapping(item, now, depth + 1)
                    if ts is not None:
                        return ts
    return None


def _hhmm_timestamp(token, now):
    match = _HM_RE.match(token)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return None
    # Vendor "resets at HH:MM" is local wall clock, not UTC.
    now_dt = datetime.fromtimestamp(now).astimezone()
    candidate = now_dt.replace(
        hour=hour, minute=minute, second=second, microsecond=0)
    if candidate.timestamp() < now:
        candidate = candidate + timedelta(days=1)
    return candidate.timestamp()


def _reset_from_retry_after(text, now):
    match = _RETRY_AFTER_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1).strip()
    if re.match(r"^\d+$", raw):
        return now + float(raw)
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _reset_from_text(text, now):
    match = _RESETS_AT_RE.search(text or "")
    if match:
        token = match.group(1).rstrip(".,;")
        ts = _as_timestamp(token)
        if ts is not None:
            return ts
        ts = _hhmm_timestamp(token, now)
        if ts is not None:
            return ts
    match = _TRY_AGAIN_RE.search(text or "")
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("minute"):
            amount *= 60
        return now + amount
    return None


def _accept_vendor_reset(ts, now):
    """Keep a parsed reset only when ``now < ts <= now + 24h``.

    Numeric values greater than ``1e11`` are treated as milliseconds.
    Anything else (elapsed, zero delay, more than a day ahead) → None
    so the caller falls back to the default cooldown.
    """
    if ts is None:
        return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts > 1e11:
        ts = ts / 1000.0
    if now < ts <= now + 24 * 3600:
        return ts
    return None


def vendor_reset_from_output(backend, stdout, stderr):
    """Return a Unix timestamp parsed from vendor output, or None.

    Structured fields win when present (JSON ``resetsAt`` / ``resets_at`` /
    ``retry_after`` / ``retryAfter``, HTTP ``Retry-After: N``). Text forms
    (``resets at <ISO or HH:MM>``, ``try again in N minutes/seconds``) are
    best-effort. A parsed value is kept only when ``now < ts <= now + 24h``
    (epoch-ms first divided by 1000); otherwise None. Nothing parsed →
    None; never guess. *backend* is accepted for the call site; the same
    forms apply to claude and cursor.
    """
    del backend
    text = "%s\n%s" % (stdout or "", stderr or "")
    if not text.strip():
        return None
    now = time.time()
    for obj in _iter_json_objects(text):
        ts = _walk_reset_mapping(obj, now)
        if ts is not None:
            return _accept_vendor_reset(ts, now)
    ts = _reset_from_retry_after(text, now)
    if ts is not None:
        return _accept_vendor_reset(ts, now)
    return _accept_vendor_reset(_reset_from_text(text, now), now)


def _road_for_pool(model_text):
    """Return the road for a ledger model, including a bare account key."""
    remainder = str(model_text)
    if "#" in remainder:
        remainder = remainder.split("#", 1)[0]
    if "/" not in remainder and ":" not in remainder and remainder in _KNOWN_ROADS:
        return remainder
    return provider_for_model(remainder)


def _account_identity(model, profile):
    """Return the vendor login identity for this seat, or None.

    Claude ``emailAddress``, Codex ``account_id_suffix``. Cursor and every
    other road return None (cursor ``kind`` is a category, not a login).
    Never raises.
    """
    try:
        from providers.profile_check import identity_label, list_profiles
        road = _road_for_pool(model)
        name = _profile_value(profile)
        for row in list_profiles(road):
            if not isinstance(row, dict) or row.get("name") != name:
                continue
            return identity_label(road, row.get("identity"))
        return None
    except Exception as exc:
        _log_account_identity_error(
            "account_identity_unreadable:%s" % type(exc).__name__)
        return None


def note_failure(model, profile, reason):
    """Increment or reset the per-seat crash streak; flag at the limit.

    ``runtime_error`` / ``signal_exit`` increment ``crash_streak`` when
    ``last_crash_at`` is within ``CRASH_STREAK_WINDOW_S`` of this crash
    (older, or missing, starts at 1). Any other reason that reaches this
    function resets the streak without touching an existing (non-crash)
    flag. ``rate_limit`` never arrives here. At ``CRASH_STREAK_LIMIT``
    the seat is flagged ``source=crash`` for ``CRASH_COOLDOWN_S`` with
    the same half-open probe as ``rate_limit``. Success is ``clear()``
    (dispatch calls it on ok, including ``probing=True``).

    Dispatch does not call this for a failure omni raised itself (stamped by
    ``errors.mark_local``): no vendor ran, so it neither adds to nor resets
    the streak.
    """
    key = pool_key(model, profile)
    reason_text = "" if reason is None else str(reason)
    now = time.time()
    did_flag = False
    repeat = 1
    try:
        account = _account_identity(model, profile)
    except Exception:
        account = None

    def mutator(data):
        nonlocal did_flag, repeat
        raw = data.get(key)
        entry = dict(raw) if isinstance(raw, dict) else {}
        if account is not None:
            entry["account"] = account
        if reason_text not in _CRASH_REASONS:
            entry.pop("crash_streak", None)
            entry.pop("last_crash_at", None)
            if _is_active(entry, now):
                data[key] = entry
            else:
                data.pop(key, None)
            return dict(data[key]) if key in data else None
        last = _as_timestamp(entry.get("last_crash_at"))
        if last is not None and (now - last) <= CRASH_STREAK_WINDOW_S:
            n = _streak_value(entry) + 1
        else:
            n = 1
        entry["crash_streak"] = n
        entry["last_crash_at"] = now
        entry["model"] = str(model)
        entry["profile"] = key.rsplit("@", 1)[1]
        if n >= CRASH_STREAK_LIMIT and (
                not _is_active(entry, now) or entry.get("source") != "crash"):
            expiry = _whole_timestamp(now + CRASH_COOLDOWN_S)
            entry["until"] = expiry
            entry["source"] = "crash"
            entry["flagged_at"] = now
            entry["probe_after"] = now + HALF_OPEN_MIN_S
            entry["probe_every"] = HALF_OPEN_EVERY_S
            did_flag = True
        data[key] = entry
        if did_flag:
            repeat = _record_flag_hit(key, now)
        return dict(entry)

    result = _locked_mutate("note_failure", mutator)
    if result is _LOCK_SKIPPED:
        existing = _read_ledger_unlocked().get(key)
        return dict(existing) if isinstance(existing, dict) else None
    if did_flag:
        _emit_seat(
            "seat_flagged",
            result["model"],
            result["profile"],
            result["source"],
            result.get("until"),
            repeat,
            window=split_pool_key(key)[3],
        )
    return result


def _load_flag_window(now):
    """Return pool_key -> hit timestamps still inside the rolling hour."""
    cutoff = now - FLAG_WINDOW_S
    try:
        data = _read_json_object(_flag_window_path())
    except OSError:
        data = {}
    cleaned = {}
    for raw_key, raw_hits in data.items():
        if not isinstance(raw_hits, list):
            continue
        kept = []
        for item in raw_hits:
            try:
                ts = float(item)
            except (TypeError, ValueError):
                continue
            if ts > cutoff:
                kept.append(ts)
        if kept:
            cleaned[str(raw_key)] = kept
    return cleaned


def _record_flag_hit(key, now):
    """Append *now* for *key* and persist. Return the in-window hit count."""
    cleaned = _load_flag_window(now)
    hits = list(cleaned.get(key, []))
    hits.append(now)
    cleaned[key] = hits
    try:
        _atomic_write_json(_flag_window_path(), cleaned)
    except OSError:
        pass
    return len(hits)


def _flag_window_count(key, now):
    return len(_load_flag_window(now).get(key, []))


def _coerce_repeat(repeat):
    """Non-int / non-finite *repeat* becomes 0 so notify still fires."""
    try:
        number = float(repeat)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(number):
        return 0
    try:
        return int(number)
    except (OverflowError, ValueError):
        return 0


def _emit_seat(event, model, profile, source, until, repeat, window=None):
    fields = {
        "road": _road_for_pool(model),
        "profile": _profile_value(profile),
        "model": str(model),
        "source": str(source),
        "until": until,
        "repeat": _coerce_repeat(repeat),
    }
    if window:
        fields["window"] = window
    notify(event, **fields)


def _flag_entry(model, profile, until=None, source="rate_limit",
                record_model=None, now=None):
    """Return ``(key, entry)`` with the ledger flag schema. Does not write.

    *until* is parsed as in :func:`flag`: missing → now + default cooldown;
    unparsable raises ``ValueError``.
    """
    if now is None:
        now = time.time()
    expiry = _as_timestamp(until)
    if until is not None and expiry is None:
        raise ValueError("until must be a Unix timestamp or ISO-8601 timestamp")
    if expiry is None:
        expiry = now + DEFAULT_COOLDOWN_S
    expiry = _whole_timestamp(expiry)
    key = pool_key(model, profile)
    stored_model = str(record_model if record_model is not None else model)
    source_text = str(source)
    if source_text in _NO_EARLY_PROBE_SOURCES:
        probe_after = expiry
    else:
        probe_after = now + HALF_OPEN_MIN_S
    try:
        # Already inside a locked mutator (apply_codex_quota / flag).
        # Nothing beneath this call may take the ledger lock.
        account = _account_identity(stored_model, profile)
    except Exception:
        account = None
    entry = {
        "model": stored_model,
        "profile": key.rsplit("@", 1)[1],
        "until": expiry,
        "source": source_text,
        "flagged_at": now,
        "probe_after": probe_after,
        "probe_every": HALF_OPEN_EVERY_S,
        "account": account,
    }
    if source_text == "credit_exhausted":
        entry["credit_hold"] = {"until": expiry, "flagged_at": now}
    return key, entry


def _install_flag(data, key, entry, now):
    """Write *entry* at *key*, keeping crash-streak fields. Return hit count."""
    existing = data.get(key)
    if isinstance(existing, dict):
        n = _streak_value(existing)
        if n:
            entry["crash_streak"] = n
        if existing.get("last_crash_at") is not None:
            entry["last_crash_at"] = existing.get("last_crash_at")
        if entry.get("account") is None and existing.get("account") is not None:
            entry["account"] = existing.get("account")
        # Credit evidence outlives whatever generic flag replaces the entry:
        # keep it, and let a credit raise extend it but never shorten it.
        prior_hold = credit_hold(existing, now)
        if prior_hold is not None:
            hold = entry.get("credit_hold")
            if isinstance(hold, dict):
                if prior_hold["until"] > hold["until"]:
                    hold["until"] = _whole_timestamp(prior_hold["until"])
            else:
                entry["credit_hold"] = {
                    "until": _whole_timestamp(prior_hold["until"]),
                    "flagged_at": prior_hold["flagged_at"]}
    data[key] = entry
    return _record_flag_hit(key, now)


def _flag_core(model, profile, until, source, record_model):
    """Write the flag; return ``(result, prior, newly_raised)``.

    *newly_raised* is True when no credit hold was active before the write
    and one is active after it, both judged at one instant taken inside the
    locked mutation (False when the lock was skipped).

    *prior* is the ledger entry that existed inside the locked mutation
    before this write, or None. A skipped lock returns the unlocked
    snapshot for both, so a caller never mistakes it for a new raise.
    """
    now = time.time()
    key, entry = _flag_entry(
        model, profile, until=until, source=source,
        record_model=record_model, now=now)
    repeat = 1
    prior = None
    newly_raised = False

    def mutator(data):
        nonlocal repeat, prior, newly_raised
        # An active cooldown is never shortened: a slower older request's
        # short flag must not cut a newer, longer one (credit exhaustion).
        # Operators who want a shorter window use clear.
        current = data.get(key)
        if isinstance(current, dict):
            prior = dict(current)
            kept = _as_timestamp(current.get("until"))
            if kept is not None and kept > now and kept > entry["until"]:
                entry["until"] = _whole_timestamp(kept)
                entry["source"] = str(current.get("source", entry["source"]))
                if current.get("probe_after") is not None:
                    entry["probe_after"] = current["probe_after"]
        stamp = time.time()
        held_before = credit_hold(current, stamp) is not None
        repeat = _install_flag(data, key, entry, now)
        newly_raised = (not held_before
                        and credit_hold(data.get(key), stamp) is not None)
        return dict(entry)

    result = _locked_mutate("flag", mutator)
    if result is _LOCK_SKIPPED:
        existing = _read_ledger_unlocked().get(key)
        existing = dict(existing) if isinstance(existing, dict) else None
        return existing, existing, False
    _emit_seat("seat_flagged", entry["model"], entry["profile"],
               entry["source"], entry["until"], repeat,
               window=split_pool_key(key)[3])
    return result, prior, newly_raised


def flag(model, profile, until=None, source="rate_limit", record_model=None):
    """Flag a seat until an explicit timestamp or the default cooldown.

    *record_model* overrides the stored ``model`` field (the ledger key
    still comes from *model* via :func:`pool_key`). Window records use
    this so a ``codex#week`` key still stores the road ``codex``.
    """
    return _flag_core(model, profile, until, source, record_model)[0]


def flag_with_prior(model, profile, until=None, source="rate_limit",
                    record_model=None):
    """Like :func:`flag`, returning ``(entry, prior)``.

    *prior* is the entry that was already in the ledger when the locked
    mutation ran (None when there was none). Deciding "newly raised" from
    it is race-free, unlike a snapshot taken before the lock.
    """
    return _flag_core(model, profile, until, source, record_model)[:2]


def flag_credit_hold(model, profile, until=None, record_model=None):
    """Raise a ``credit_exhausted`` flag; return ``(entry, newly_raised)``.

    *newly_raised* is decided inside the locked mutation with one
    timestamp, so a slow ``seat_flagged`` sink cannot change the answer.
    """
    result, _prior, newly_raised = _flag_core(
        model, profile, until, "credit_exhausted", record_model)
    return result, newly_raised


class HealthLockUnavailable(RuntimeError):
    """The ledger lock (or the ledger itself) could not be taken or read."""


# A success is not evidence that credits were added: only an explicit
# clear (or the configured window expiring) ends these holds.
SUCCESS_KEEP_SOURCES = ("credit_exhausted",)


def clear_models(model):
    """Model strings ``omni health clear <model>`` clears, in order.

    *model* itself (the same pool :func:`flag` writes), plus the bare road
    when *model* names a road (``claude``) or is ``<road>/<model>``. Never
    another road: a bare ``claude`` is a road even though routing would
    send it elsewhere.
    """
    text = str(model)
    models = [text]
    road = text.split("/", 1)[0].split("#", 1)[0]
    if road in _KNOWN_ROADS and road not in models:
        models.append(road)
    return models


def clear(model, profile, unless_flagged_after=None, keep_sources=()):
    """Remove a seat flag, pruning expired entries while writing.

    *unless_flagged_after* (Unix seconds) keeps a flag raised after that
    instant: an older success must not erase newer failure evidence.
    *keep_sources* keeps a flag whose ``source`` is listed, and, when
    ``credit_exhausted`` is listed, any entry with an active credit hold
    whatever its ``source`` now says (a success passes
    :data:`SUCCESS_KEEP_SOURCES`). Raises
    :class:`HealthLockUnavailable` when the removal could not be made;
    returns None when nothing was flagged or the flag was kept.
    """
    key = pool_key(model, profile)
    repeat = 0

    def mutator(data):
        nonlocal repeat
        current = data.get(key)
        if keep_sources and isinstance(current, dict) and (
                current.get("source") in keep_sources
                or ("credit_exhausted" in keep_sources
                    and credit_hold(current) is not None)):
            return None
        if unless_flagged_after is not None:
            stamp = _as_timestamp(current.get("flagged_at")) if isinstance(current, dict) else None
            if stamp is not None and stamp > unless_flagged_after:
                return None
        popped = data.pop(key, None)
        if not isinstance(popped, dict):
            return None
        repeat = _flag_window_count(key, time.time())
        return dict(popped)

    result = _locked_mutate("clear", mutator)
    if result is _LOCK_SKIPPED:
        raise HealthLockUnavailable("health ledger lock unavailable")
    if not isinstance(result, dict):
        return result
    _emit_seat(
        "seat_cleared",
        result.get("model") or str(model),
        result.get("profile") or _profile_value(profile),
        result.get("source") or "",
        result.get("until"),
        repeat,
        window=split_pool_key(key)[3],
    )
    return result


def clear_all():
    """Clear all flags in the current call-time state directory."""
    removed = []

    def mutator(data):
        now = time.time()
        leases = data.get(_LEASES_KEY)
        keep_leases = _LEASES_KEY in data and isinstance(leases, dict)
        for key, raw in list(data.items()):
            if key == _LEASES_KEY:
                continue
            if not isinstance(raw, dict):
                continue
            removed.append(
                (str(key), dict(raw), _flag_window_count(str(key), now)))
        data.clear()
        if keep_leases:
            data[_LEASES_KEY] = leases
        return None

    result = _locked_mutate("clear_all", mutator)
    if result is _LOCK_SKIPPED:
        return None
    for key, entry, repeat in removed:
        model = entry.get("model")
        profile = entry.get("profile")
        if model is None or profile is None:
            model_text, profile_text = str(key).rsplit("@", 1) if "@" in str(key) \
                else (str(key), "default")
            if model is None:
                model = model_text
            if profile is None:
                profile = profile_text
        _emit_seat(
            "seat_cleared",
            model,
            profile,
            entry.get("source") or "",
            entry.get("until"),
            repeat,
            window=split_pool_key(key)[3],
        )
    return result


def reprobe(model, profile):
    """Bump ``probe_after`` on an active flag; keep ``until`` unchanged."""
    key = pool_key(model, profile)

    def mutator(data):
        entry = data.get(key)
        if not _is_active(entry):
            return None
        every = entry.get("probe_every", HALF_OPEN_EVERY_S)
        try:
            every = float(every)
        except (TypeError, ValueError):
            every = HALF_OPEN_EVERY_S
        updated = dict(entry)
        updated["probe_after"] = time.time() + every
        data[key] = updated
        return dict(updated)

    result = _locked_mutate("reprobe", mutator)
    if result is _LOCK_SKIPPED:
        existing = _read_ledger_unlocked().get(key)
        return dict(existing) if isinstance(existing, dict) else None
    return result


def maintain():
    """Prune expired ledger entries under the lock. Does not spawn the RPC."""
    def mutator(data):
        return None

    result = _locked_mutate("maintain", mutator)
    if result is _LOCK_SKIPPED:
        return None
    return result


def _is_lease_active(entry, now=None):
    if not isinstance(entry, dict):
        return False
    expires = _as_timestamp(entry.get("expires_at"))
    return expires is not None and expires > (time.time() if now is None else now)


def _expires_at_iso(expires):
    ts = _as_timestamp(expires)
    if ts is None:
        return None
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _lease_view(record):
    expires = record.get("expires_at")
    return {
        "token": record.get("token"),
        "pool": record.get("pool"),
        "order_id": record.get("order_id"),
        "ttl_s": record.get("ttl_s"),
        "acquired_at": record.get("acquired_at"),
        "renewed_at": record.get("renewed_at"),
        "expires_at": expires,
        "expires_at_iso": _expires_at_iso(expires),
    }


def _lease_bucket(data):
    leases = data.get(_LEASES_KEY)
    if not isinstance(leases, dict):
        leases = {}
        data[_LEASES_KEY] = leases
    return leases


def _prune_expired_leases(leases, now):
    for token in list(leases):
        if not _is_lease_active(leases.get(token), now):
            leases.pop(token, None)


def _store_lease_bucket(data, leases):
    if leases:
        data[_LEASES_KEY] = leases
    else:
        data.pop(_LEASES_KEY, None)


def _lease_views(leases, now):
    views = [
        _lease_view(record)
        for record in leases.values()
        if _is_lease_active(record, now)
    ]
    views.sort(key=lambda row: _as_timestamp(row.get("expires_at")) or 0.0)
    return views


def _require_lease_write(result):
    if result is _LOCK_SKIPPED:
        raise RuntimeError("health_skipped")
    return result


def _require_ttl(ttl_s):
    """Return a positive int TTL, or raise ValueError('bad_ttl')."""
    try:
        number = float(ttl_s)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("bad_ttl")
    if not math.isfinite(number) or number <= 0:
        raise ValueError("bad_ttl")
    try:
        ttl = int(number)
    except (OverflowError, ValueError):
        raise ValueError("bad_ttl")
    if ttl <= 0:
        raise ValueError("bad_ttl")
    return ttl


def lease_acquire(token, pool, order_id, ttl_s):
    """Record a lease. Active duplicate token → ValueError('lease_exists')."""
    token = str(token)
    pool = str(pool)
    order_id = str(order_id)
    ttl = _require_ttl(ttl_s)

    def mutator(data):
        now = time.time()
        leases = _lease_bucket(data)
        _prune_expired_leases(leases, now)
        if _is_lease_active(leases.get(token), now):
            _store_lease_bucket(data, leases)
            return ("lease_exists", None)
        record = {
            "token": token,
            "pool": pool,
            "order_id": order_id,
            "ttl_s": ttl,
            "acquired_at": now,
            "renewed_at": now,
            "expires_at": _whole_timestamp(now + ttl),
        }
        leases[token] = record
        _store_lease_bucket(data, leases)
        return ("ok", record)

    status, record = _require_lease_write(
        _locked_mutate("lease_acquire", mutator))
    if status == "lease_exists":
        raise ValueError("lease_exists")
    return _lease_view(record)


def lease_renew(token, ttl_s=None):
    """Extend a lease. Unknown or expired → KeyError('unknown_lease')."""
    token = str(token)
    if ttl_s is not None:
        ttl_s = _require_ttl(ttl_s)

    def mutator(data):
        now = time.time()
        leases = _lease_bucket(data)
        _prune_expired_leases(leases, now)
        existing = leases.get(token)
        if not _is_lease_active(existing, now):
            _store_lease_bucket(data, leases)
            return ("unknown_lease", None)
        if ttl_s is None:
            ttl = int(existing.get("ttl_s"))
        else:
            ttl = ttl_s
        updated = dict(existing)
        updated["ttl_s"] = ttl
        updated["renewed_at"] = now
        updated["expires_at"] = _whole_timestamp(now + ttl)
        leases[token] = updated
        _store_lease_bucket(data, leases)
        return ("ok", updated)

    status, record = _require_lease_write(
        _locked_mutate("lease_renew", mutator))
    if status == "unknown_lease":
        raise KeyError("unknown_lease")
    return _lease_view(record)


def lease_release(token):
    """Drop a lease. Unknown or expired → KeyError('unknown_lease')."""
    now = time.time()
    token = str(token)

    def mutator(data):
        leases = _lease_bucket(data)
        _prune_expired_leases(leases, now)
        if token not in leases:
            _store_lease_bucket(data, leases)
            return ("unknown_lease", None)
        leases.pop(token, None)
        _store_lease_bucket(data, leases)
        return ("ok", token)

    status, released = _require_lease_write(
        _locked_mutate("lease_release", mutator))
    if status == "unknown_lease":
        raise KeyError("unknown_lease")
    return released


def leases():
    """Return active lease view records, sorted by expires_at."""
    now = time.time()
    raw = _read_ledger_unlocked().get(_LEASES_KEY)
    if not isinstance(raw, dict):
        return []
    return _lease_views(raw, now)


def active(model, profile):
    """Return whether any covering record is a non-expired flag."""
    ledger = _read_ledger_unlocked()
    now = time.time()
    for key in _covering_keys(model, profile, ledger):
        if _is_active(ledger.get(key), now):
            return True
    return False


def _scheduled_probe_keys(profile):
    """Ledger keys ``probe_all`` visits for *profile*, on every covered road.

    Asked of ``profile_check``, which owns both the coverage and the pool each
    road's result lands on; this module states neither, so a road added there
    is reflected here with no edit. Unreadable → no key is visited.
    Codex: ``codex@<profile>``, every present ``codex#<window>@<profile>``
    whose window is ``week`` or ``m<digits>`` and whose parsed road and
    profile match exactly, and ``codex#week@<profile>`` (a not-yet-written
    week key is still advertised).

    The question is what the probe does when it runs for *profile*, not
    whether ``list_profiles()`` still enumerates it: an unlisted profile is
    answered as if it were listed. Deciding otherwise would pull account
    enumeration into every ledger read to catch a case that is wrong by one
    optimistic flag, and that expires at its ``until`` regardless.
    """
    try:
        from providers.profile_check import (
            probe_covered_roads,
            probe_pool_model,
        )
        roads = probe_covered_roads()
    except Exception as exc:
        # Fail closed, and do not raise: this runs on every ledger read, and
        # coverage that cannot be read must cost a pessimistic field (someone
        # clears a flag that would have cleared itself), never a broken
        # ``omni health``. Logged once per kind so it is not silent.
        _log_probe_coverage_error(
            "probe_coverage_unreadable:%s" % type(exc).__name__)
        return frozenset()
    keys = set()
    for road in roads:
        try:
            model = probe_pool_model(road)
            if model:
                keys.add(pool_key(model, profile))
                if road == "codex":
                    keys.add(pool_key("codex#week", profile))
                    profile_text = _profile_value(profile)
                    for key in _read_ledger_unlocked():
                        if key == _LEASES_KEY:
                            continue
                        text = str(key)
                        if _is_quota_window_key(text, "codex", profile_text):
                            keys.add(text)
        except Exception as exc:
            # Fail closed for THIS road only. One road that cannot name its
            # pool must not erase the roads that did: those keys really are
            # re-checked on every scheduled run, and reporting them as
            # abandoned is the answer that gets a live flag cleared by hand.
            _log_probe_coverage_error(
                "probe_pool_unreadable:%s:%s" % (road, type(exc).__name__))
    return frozenset(keys)


def _will_recheck(entry, key=None):
    """Whether anything (half-open dispatch or the scheduled probe) will re-check.

    Half-open: any source ``seat_kind`` still consults ``probe_after`` for.
    Scheduled probe: a recoverable source whose ledger key is in the
    write set :func:`_scheduled_probe_keys` (for codex: ``codex@p``,
    present ``codex#<window>@p``, and ``codex#week@p``). A model-level
    429 is not rechecked because of a quota probe.
    ``apply_codex_quota`` will not clear ``manual``.
    """
    if credit_hold(entry) is not None:
        return False
    source = str(entry.get("source") or "")
    if source not in _NO_EARLY_PROBE_SOURCES:
        return True
    if source not in _RECOVERABLE_SOURCES:
        return False
    model = entry.get("model")
    if not model:
        return False
    profile = entry.get("profile")
    if key is None:
        key = pool_key(model, profile)
    return str(key) in _scheduled_probe_keys(profile)


def snapshot():
    """Return active ledger entries keyed by :func:`pool_key`.

    Each copy carries ``will_recheck``, derived at read time from the
    entry's source and from whether this ledger key is one ``probe_all``
    visits on any road it covers. It is not stored.
    """
    now = time.time()
    out = {}
    for key, value in _read_ledger_unlocked().items():
        if not _is_active(value, now):
            continue
        entry = dict(value)
        entry["will_recheck"] = _will_recheck(entry, key)
        if "account" not in entry:
            entry["account"] = None
        out[key] = entry
    return out


def seat_kind(model, profile, now=None):
    """Return ``healthy``, ``cooling``, or ``probing`` without writing.

    Looks at every covering record. Cooling beats probing: ``probing``
    only when every covering active record is past its ``probe_after``
    and none has a source in ``_NO_EARLY_PROBE_SOURCES``. ``healthy``
    only when none is active.
    """
    now = time.time() if now is None else now
    ledger = _read_ledger_unlocked()
    actives = []
    for key in _covering_keys(model, profile, ledger):
        entry = ledger.get(key)
        if _is_active(entry, now):
            actives.append(entry)
    if not actives:
        return "healthy"
    for entry in actives:
        if credit_hold(entry, now) is not None:
            return "cooling"
        source = str(entry.get("source") or "")
        if source in _NO_EARLY_PROBE_SOURCES:
            return "cooling"
        probe_after = entry.get("probe_after")
        ts = _as_timestamp(probe_after)
        if ts is None or ts > now:
            return "cooling"
    return "probing"


def _pop_quota_source(data, events, key, now):
    """Remove *key* if it is a quota record. Append a seat_cleared event."""
    entry = data.get(key)
    if not isinstance(entry, dict):
        return False
    if str(entry.get("source") or "") != "quota":
        return False
    popped = data.pop(key)
    events.append((
        "seat_cleared",
        dict(popped),
        key,
        _flag_window_count(str(key), now if now is not None else time.time()),
    ))
    return True


def _apply_quota_window(data, events, pool, profile, used, reached, resets_at,
                        record_model=None, now=None, observed_at=None):
    """Flag or clear one quota window at *pool* on the locked *data* view.

    A healthy reading clears only a ``source=quota`` record at this key.
    Does not call :func:`flag` or :func:`clear`.
    """
    if now is None:
        now = time.time()
    exhausted = bool(reached) or (used is not None and used >= 95)
    stored = record_model if record_model is not None else pool
    if exhausted:
        key, entry = _flag_entry(
            pool, profile, until=resets_at, source="quota",
            record_model=stored, now=now)
        if observed_at is not None:
            entry["observed_at"] = observed_at
        repeat = _install_flag(data, key, entry, now)
        events.append(("seat_flagged", entry, key, repeat))
        return True
    if used is None and not reached:
        return False
    key = pool_key(pool, profile)
    entry = data.get(key)
    if not _is_active(entry, now):
        return False
    source = str(entry.get("source") or "")
    if source == "manual":
        return True
    if source == "quota":
        _pop_quota_source(data, events, key, now)
    return False


def _newest_quota_observed_at(data, model, profile, road):
    """Newest ``observed_at`` on this profile's account and window quota records."""
    newest = None
    profile_text = _profile_value(profile)
    account_key = pool_key(model, profile_text)
    for key, entry in data.items():
        if key == _LEASES_KEY or not isinstance(entry, dict):
            continue
        if str(entry.get("source") or "") != "quota":
            continue
        if key != account_key and not _is_quota_window_key(key, road, profile_text):
            continue
        observed = _as_timestamp(entry.get("observed_at"))
        if observed is None:
            continue
        if newest is None or observed > newest:
            newest = observed
    return newest


def apply_codex_quota(model, profile, info, err):
    """Apply a quota-probe result to the account key and optional week.

    Primary lands on ``pool_key(model, profile)`` (writers pass a road).
    When ``info["secondary"]`` is present it is applied independently to
    ``<road>#<window>@profile`` with the record's ``model`` field set to
    that road. *road* is :func:`_road_for_pool` of *model*. One
    observation is one locked write: primary flag/clear, secondary
    flag/clear, and — when the observation carries window information
    (a secondary dict, or a weekly primary) — deletion of every other
    ``<road>#<window>@profile`` quota record of this profile (active or
    not). An observation without window information leaves window keys
    alone. Manual and model-level records are left alone. If
    ``info["reached"]`` is true and no window would be flagged after
    attribution and window-name validation, the primary window is
    flagged (fail-closed). Each written quota record stores
    ``observed_at`` from ``info["checked_at"]`` (else now); an older
    observation than the newest stored ``observed_at`` on this
    profile's account and window quota records is a no-op.
    """
    if err:
        reason = "runtime_error"
        detail = "quota_probe_failed"
        if isinstance(err, dict):
            reason = err.get("reason") or reason
            if reason == "disabled":
                return False
            raw_detail = err.get("detail")
            if isinstance(raw_detail, str) and raw_detail:
                detail = raw_detail
        log_event("health_error", operation="quota_probe", reason=reason,
                  detail=str(detail)[:300])
        return False
    if not isinstance(info, dict):
        return False
    road = _road_for_pool(model)
    if "primary_reached" in info:
        primary_reached = bool(info.get("primary_reached"))
    else:
        primary_reached = bool(info.get("reached"))
    secondary = info.get("secondary")
    secondary_reached = (
        isinstance(secondary, dict) and bool(secondary.get("reached"))
    )
    now = time.time()
    observed_at = _as_timestamp(info.get("checked_at"))
    if observed_at is None:
        observed_at = now
    events = []
    keep_windows = set()
    secondary_suffix = None
    if isinstance(secondary, dict):
        candidate = _quota_window_name(secondary.get("window_mins"))
        if candidate and _QUOTA_WINDOW_RE.match(candidate):
            secondary_suffix = candidate
            keep_windows.add(secondary_suffix)
        else:
            log_event(
                "health_error",
                operation="quota_window",
                detail="invalid window name %s" % candidate,
            )
    # Fail-closed after window-name validation: reached with no window
    # that this observation will actually flag.
    valid_secondary_reached = (
        secondary_suffix is not None and secondary_reached
    )
    if bool(info.get("reached")) and not primary_reached and not valid_secondary_reached:
        primary_reached = True
        log_event(
            "health_error",
            operation="quota_reached_unattributed",
            model=str(model),
            profile=_profile_value(profile),
            detail="reached with no attributed window",
        )
    primary_weekly = _window_is_weekly(info.get("window_mins"))
    if primary_reached and primary_weekly:
        keep_windows.add("week")
    window_authoritative = (
        secondary_suffix is not None or primary_weekly
    )

    def mutator(data):
        # _flag_entry calls _account_identity; this mutator already holds
        # the ledger lock, so nothing beneath that call may take it.
        newest = _newest_quota_observed_at(data, model, profile, road)
        if newest is not None and observed_at < newest:
            log_event(
                "health_error",
                operation="quota_stale_observation",
                model=str(model),
                profile=_profile_value(profile),
            )
            return False
        flagged = _apply_quota_window(
            data, events, model, profile,
            used=info.get("used_percent"),
            reached=primary_reached,
            resets_at=info.get("resets_at"),
            record_model=str(model),
            now=now,
            observed_at=observed_at,
        )
        if secondary_suffix is not None:
            flagged = _apply_quota_window(
                data, events, "%s#%s" % (road, secondary_suffix), profile,
                used=secondary.get("used_percent"),
                reached=bool(secondary.get("reached")),
                resets_at=secondary.get("resets_at"),
                record_model=road,
                now=now,
                observed_at=observed_at,
            ) or flagged
        if window_authoritative:
            _reconcile_codex_quota_windows(
                data, events, profile, keep_windows, now, road=road)
        if primary_reached and primary_weekly:
            week_pool = "%s#week" % road
            week_key = pool_key(week_pool, profile)
            existing = data.get(week_key)
            if (isinstance(existing, dict)
                    and str(existing.get("source") or "") == "quota"):
                flagged = _apply_quota_window(
                    data, events, week_pool, profile,
                    used=info.get("used_percent"),
                    reached=True,
                    resets_at=info.get("resets_at"),
                    record_model=road,
                    now=now,
                    observed_at=observed_at,
                ) or flagged
        return bool(flagged)

    result = _locked_mutate("apply_codex_quota", mutator)
    if result is _LOCK_SKIPPED:
        return False
    for event, entry, key, repeat in events:
        model_text = entry.get("model") or str(model)
        profile_text = entry.get("profile") or _profile_value(profile)
        _emit_seat(
            event,
            model_text,
            profile_text,
            entry.get("source") or "",
            entry.get("until"),
            repeat,
            window=split_pool_key(key)[3],
        )
        if event == "seat_cleared":
            log_event(
                "seat_recovered",
                model=model_text,
                profile=profile_text,
                source="quota",
            )
    return bool(result)


def _reconcile_codex_quota_windows(data, events, profile, keep_windows,
                                  now=None, road="codex"):
    """Delete stale ``<road>#<window>@profile`` quota records not in *keep_windows*.

    Runs on the locked *data* view of one observation. Expired quota
    windows are removed too. Manual records, model-keyed records, and
    keys of other profiles are not touched. *road* is the pool road
    (from :func:`_road_for_pool`), not a hardcoded ``codex``.
    """
    if now is None:
        now = time.time()
    profile_text = _profile_value(profile)
    keep = set(keep_windows)
    for key in list(data):
        if key == _LEASES_KEY:
            continue
        if not _is_quota_window_key(key, road, profile_text):
            continue
        window = split_pool_key(key)[3]
        if window in keep:
            continue
        _pop_quota_source(data, events, key, now)


def _latest_session_file():
    """Deprecated: session-file walk kept for existing tests that patch it."""
    root = os.path.expanduser(
        os.environ.get(ENV_CODEX_SESSIONS_DIR) or DEFAULT_CODEX_SESSIONS_DIR
    )
    paths = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _rate_limits_values(value):
    """Deprecated: session-file walker kept for existing tests."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "rate_limits":
                yield child
            yield from _rate_limits_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _rate_limits_values(child)


def _codex_session_file_probe(model, profile, session_path):
    if active(model, profile):
        return True
    if not session_path:
        return False
    with open(session_path, encoding="utf-8", errors="replace") as handle:
        lines = deque(handle, maxlen=50)
    latest = None
    for line in lines:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        for candidate in _rate_limits_values(payload):
            latest = candidate
    if not isinstance(latest, dict):
        return False
    primary = latest.get("primary")
    if not isinstance(primary, dict):
        return False
    try:
        used_percent = float(primary.get("used_percent"))
    except (TypeError, ValueError):
        return False
    if used_percent < 95:
        return False
    reset_at = _as_timestamp(primary.get("resets_at"))
    if reset_at is not None and reset_at <= time.time():
        return False
    flag(model, profile, until=reset_at, source="codex_passive")
    return True


def codex_passive_probe(model, profile):
    """Flag or recover a Codex seat from the app-server quota RPC.

    The RPC result is applied to the account key ``codex``. The deprecated
    session-file walk (``OMNI_LLM_CODEX_SESSIONS_DIR``) still flags the
    given model with ``source=codex_passive``. The probe logs and swallows
    every error.
    """
    if not str(model).lower().startswith("codex/"):
        return False
    try:
        try:
            session_path = _latest_session_file()
        except Exception as exc:
            log_event("health_error", operation="passive_probe",
                      detail=str(exc)[:300])
            return False
        if os.environ.get(ENV_CODEX_SESSIONS_DIR):
            return _codex_session_file_probe(model, profile, session_path)
        from providers.codex_quota import read_rate_limits
        info, err = read_rate_limits(_profile_value(profile))
        return apply_codex_quota("codex", profile, info, err)
    except Exception as exc:
        log_event("health_error", operation="passive_probe", detail=str(exc)[:300])
        return False
