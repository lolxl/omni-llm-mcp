# no-opencc
"""``quota check`` / ``quota policy show``: quota-pool eligibility, ``quota-decision/v1``.

One decision for one (login, account): is this money still spendable? It layers on
what the tool already keeps: the seat-health ledger, the Codex account/rateLimits/read
RPC (through ``codex_quota.read_rate_limits``) and a caller-supplied Anthropic usage
snapshot. It fails closed, and every value it prints or audits is built from an
allowlist with value validation; profile or identity objects are never copied in.

The decision is only about quota. Seat validity, capacity leases and fallback order
belong to the caller.

``quota report-incident`` records a hard rate limit for one (quota pool, account)
as a marker in the state dir, interface ``quota-incident/v1``; ``check`` refuses
that quota pool and account until the marker expires, and refuses as well when a
marker is present but cannot be read exactly.
"""
import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from providers import health

INTERFACE_VERSION = "quota-decision/v1"
POLICY_SCHEMA = "quota-policy/v1"
POLICY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "quota_policy.json")

EXIT_ELIGIBLE = 0
EXIT_INELIGIBLE = 10
EXIT_USAGE = 2
EXIT_RECORD = 73   # report-incident could not write its marker
EXIT_OUTPUT = 74   # the result could not be written to stdout (closed pipe)

DECISIONS = ("eligible", "ineligible", "advisory")
# The closed reason enum. The first group keeps the caller's spelling; the second
# group is new in this interface (see docs/reference.md for each caller equivalent).
CALLER_REASONS = (
    "ok",
    "quota_reserve",
    "stale_telemetry",
    "missing_telemetry",
    "malformed_telemetry",
    "unmonitored_pool",
    "unknown_login",
    "registry_invalid",
    "hard_rate_limit",
)
NEW_REASONS = (
    "telemetry_unavailable",
    "telemetry_malformed",
    "ledger_cooling",
    "ledger_unreadable",
    "unknown_account",
)
REASONS = CALLER_REASONS + NEW_REASONS
# Only a valid, correctly bound observation at or over the threshold is a warning
# for a short task (decision advisory, exit 0). Every other refusal, above all
# invalid, stale or unverifiable evidence, refuses for long and short tasks alike.
# The caller's softer mapping of missing, stale or malformed telemetry is the
# caller's own rule (see docs/reference.md, "caller advisory / tool ineligible").
SOFT_REASONS = frozenset(("quota_reserve",))
# A decision built on one of these is never reusable: it expires when it is made.
NO_REUSE_REASONS = frozenset((
    "stale_telemetry", "missing_telemetry", "malformed_telemetry",
    "telemetry_unavailable", "telemetry_malformed", "ledger_unreadable",
    "registry_invalid",
))

TASK_CLASSES = ("long", "short")
CONTEXTS = ("primary", "fallback")
MONITORING = ("codex_rpc", "anthropic_snapshot", "unmonitored")
ROADS = ("codex", "grok", "agy", "cursor", "claude")

AUDIT_FILE = "quota-audit.jsonl"
AUDIT_ROTATE_BYTES = 5 * 1024 * 1024
SNAPSHOT_MAX_BYTES = 1024 * 1024
SNAPSHOT_MAX_WINDOWS = 16
_DIR_MODE = 0o700
_FILE_MODE = 0o600

_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_REF_RE = re.compile(r"[A-Za-z0-9_.:/+-]{1,128}")
_WINDOW_RE = re.compile(r"[A-Za-z0-9_.-]{1,32}")
_ISO_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_REVISION_RE = re.compile(r"[0-9a-f]{64}")

DOC_KEYS = (
    "interface_version", "decision_id", "decision", "reasons", "quota_pool",
    "login", "account", "used_bp", "window", "observed_at", "freshness_s",
    "policy_revision", "checked_at", "expires_at",
)

INCIDENT_INTERFACE = "quota-incident/v1"
INCIDENT_DIR = "quota-incidents"
INCIDENT_LOG = "quota-incidents.jsonl"
INCIDENT_KINDS = ("hard_rate_limit",)
INCIDENT_MAX_BYTES = 64 * 1024
# The marker and the incident log line, in this order. The report output adds
# ``ok`` and ``login``.
INCIDENT_KEYS = (
    "interface_version", "quota_pool", "account", "kind", "until", "recorded_at",
    "incident_id",
)
REPORT_KEYS = (
    "interface_version", "ok", "quota_pool", "login", "account", "kind", "until",
    "recorded_at", "incident_id",
)
# The only audit-only field a decision line may carry, and its values.
INCIDENT_STATES = ("active", "expired", "unreadable")


class PolicyError(Exception):
    """The shipped policy is missing or invalid. The message is fixed text."""


class _OutputInvalid(Exception):
    pass


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Policy: schema validation and revision
# ---------------------------------------------------------------------------
def _int_in(value, low, high):
    return (isinstance(value, int) and not isinstance(value, bool)
            and low <= value <= high)


def _id_ok(value):
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def _keys(obj, expected):
    return isinstance(obj, dict) and set(obj) == set(expected)


def validate_policy(doc):
    """Raise :class:`PolicyError` unless *doc* is a valid ``quota-policy/v1`` document."""
    def bad(msg):
        raise PolicyError(msg)

    if not _keys(doc, ("policy_schema", "reserve_threshold_bp", "decision_ttl_cap_s",
                       "codex_rpc", "anthropic_snapshot", "pools")):
        bad("policy keys")
    if doc["policy_schema"] != POLICY_SCHEMA:
        bad("policy schema")
    if not _int_in(doc["reserve_threshold_bp"], 1, 10000):
        bad("reserve_threshold_bp")
    if not _int_in(doc["decision_ttl_cap_s"], 1, 3600):
        bad("decision_ttl_cap_s")
    codex = doc["codex_rpc"]
    if not _keys(codex, ("cache_freshness_s", "timeout_s", "future_tolerance_seconds")):
        bad("codex_rpc keys")
    if not (_int_in(codex["cache_freshness_s"], 1, 3600)
            and _int_in(codex["timeout_s"], 1, 30)
            and _int_in(codex["future_tolerance_seconds"], 0, 300)):
        bad("codex_rpc values")
    anth = doc["anthropic_snapshot"]
    if not _keys(anth, ("max_age_seconds", "future_tolerance_seconds")):
        bad("anthropic_snapshot keys")
    if not (_int_in(anth["max_age_seconds"], 1, 86400)
            and _int_in(anth["future_tolerance_seconds"], 0, 3600)):
        bad("anthropic_snapshot values")
    pools = doc["pools"]
    if not isinstance(pools, dict) or not pools:
        bad("pools")
    logins = set()
    for pool_id, pool in pools.items():
        if not _id_ok(pool_id):
            bad("pool id")
        if not _keys(pool, ("login", "road", "accounts", "profile", "monitoring")):
            bad("pool keys")
        if not _id_ok(pool["login"]) or pool["login"] in logins:
            bad("pool login")
        logins.add(pool["login"])
        if pool["road"] not in ROADS:
            bad("pool road")
        if not _id_ok(pool["profile"]):
            bad("pool profile")
        if pool["monitoring"] not in MONITORING:
            bad("pool monitoring")
        accounts = pool["accounts"]
        if (not isinstance(accounts, list) or not accounts
                or not all(_id_ok(a) for a in accounts)
                or len(set(accounts)) != len(accounts)
                or "default" not in accounts):
            bad("pool accounts")
        # One telemetry binding per pool: a monitored pool serves one account.
        if pool["monitoring"] != "unmonitored" and accounts != ["default"]:
            bad("monitored pool accounts")
        if pool["monitoring"] == "codex_rpc" and pool["road"] != "codex":
            bad("codex_rpc road")
        if pool["monitoring"] == "anthropic_snapshot" and pool["road"] != "claude":
            bad("anthropic_snapshot road")
    return doc


def policy_revision(doc):
    """sha256 (hex) of the canonical JSON of the policy content."""
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def load_policy(path=None):
    """Return ``(policy, revision)`` for the shipped policy, or raise PolicyError.

    *path* exists for tests. There is no environment or state-dir overlay: the
    shipped file is the only authority.
    """
    target = path or POLICY_FILE
    try:
        with open(target, encoding="utf-8") as handle:
            doc = health.strict_json_loads(handle.read())
    except (OSError, ValueError, RecursionError):
        raise PolicyError("policy unreadable")
    validate_policy(doc)
    return doc, policy_revision(doc)


def _pool_for_login(policy, login):
    for pool_id, pool in policy["pools"].items():
        if pool["login"] == login:
            return pool_id, pool
    return None, None


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------
def used_percent_to_bp(used_percent):
    """Decimal half-up conversion, as the caller does (``str`` of the number)."""
    bp = (Decimal(str(used_percent)) * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP)
    return int(bp)


def _percent_ok(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 100)


def _parse_iso(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError("timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_label(kind):
    if isinstance(kind, str) and _WINDOW_RE.fullmatch(kind):
        return kind
    return "other"


class _Evidence(object):
    """What one telemetry read established. Plain values only."""

    def __init__(self):
        self.reasons = []        # [(reason, soft)] in order
        self.used_bp = None
        self.window = None
        self.observed_at = None  # aware datetime
        self.freshness_limit_s = None

    def fail(self, reason):
        self.reasons.append((reason, reason in SOFT_REASONS))


def _read_snapshot_file(path):
    """Return ``(data, reason)`` for the Anthropic snapshot at *path*."""
    if not path:
        return None, "missing_telemetry"
    try:
        if not os.path.isfile(path):
            return None, "missing_telemetry"
        with open(path, "rb") as handle:
            raw = handle.read(SNAPSHOT_MAX_BYTES + 1)
        if len(raw) > SNAPSHOT_MAX_BYTES:
            return None, "malformed_telemetry"
        data = health.strict_json_loads(raw.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None, "malformed_telemetry"
    return data, None


def _evaluate_anthropic(data, read_reason, policy, reserve_bp, now, account):
    """Match the caller's ``evaluate_subscription_telemetry``, plus the account
    binding: a snapshot that names an account must name the requested one, and
    one that names none is accepted only for ``default``."""
    ev = _Evidence()
    ev.freshness_limit_s = policy["anthropic_snapshot"]["max_age_seconds"]
    if read_reason:
        ev.fail(read_reason)
        return ev
    windows = data.get("windows") if isinstance(data, dict) else None
    if (not isinstance(data, dict) or not isinstance(windows, list) or not windows
            or len(windows) > SNAPSHOT_MAX_WINDOWS
            or not all(isinstance(w, dict) and isinstance(w.get("kind"), str)
                       and _percent_ok(w.get("used_percent")) for w in windows)):
        ev.fail("malformed_telemetry")
        return ev
    if "account" in data:
        if not isinstance(data["account"], str) or data["account"] != account:
            ev.fail("malformed_telemetry")
            return ev
    elif account != "default":
        ev.fail("malformed_telemetry")
        return ev
    try:
        generated_at = _parse_iso(data.get("generated_at"))
    except (ValueError, OverflowError):
        ev.fail("malformed_telemetry")
        return ev
    ev.observed_at = generated_at
    age = (now - generated_at).total_seconds()
    tolerance = policy["anthropic_snapshot"]["future_tolerance_seconds"]
    if age < -tolerance or age > ev.freshness_limit_s:
        ev.fail("stale_telemetry")
        return ev
    best_bp, best_kind = None, None
    for window in windows:
        bp = used_percent_to_bp(window["used_percent"])
        if best_bp is None or bp > best_bp:
            best_bp, best_kind = bp, window["kind"]
    ev.used_bp = best_bp
    ev.window = _window_label(best_kind)
    if best_bp >= reserve_bp:
        ev.fail("quota_reserve")
    return ev


def _number_ok(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _int_or_none(value):
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _window_fields_ok(window):
    """The side fields every window carries; a bad type makes the window malformed."""
    return (isinstance(window.get("reached"), bool)
            and (window.get("resets_at") is None or _number_ok(window.get("resets_at")))
            and _int_or_none(window.get("window_mins")))


def _evaluate_codex(info, err, policy, reserve_bp, now):
    """Judge one Codex observation (an RPC reply or an already validated cache
    row). Both windows, vendor ``reached``, ``quota_unknown``, finite percentages,
    the types of every field and the observation time all count; any malformed
    window refuses the whole observation."""
    ev = _Evidence()
    ev.freshness_limit_s = policy["codex_rpc"]["cache_freshness_s"]
    if err or not isinstance(info, dict):
        ev.fail("telemetry_unavailable")
        return ev
    observed = info.get("checked_at")
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not math.isfinite(observed)):
        ev.fail("malformed_telemetry")
        return ev
    try:
        ev.observed_at = datetime.fromtimestamp(observed, timezone.utc)
    except (OverflowError, OSError, ValueError):
        ev.fail("malformed_telemetry")
        return ev
    unknown = info.get("quota_unknown", False)
    if not isinstance(unknown, bool) or unknown:
        ev.fail("malformed_telemetry")
        return ev
    primary = info.get("used_percent")
    primary_reached = info.get("primary_reached", info.get("reached"))
    plan = info.get("plan")
    if (not _window_fields_ok(info) or not isinstance(primary_reached, bool)
            or not (plan is None or isinstance(plan, str))
            or not (primary is None or _percent_ok(primary))):
        ev.fail("malformed_telemetry")
        return ev
    values = []   # (label, bp)
    reached = info["reached"] or primary_reached
    if primary is not None:
        values.append(("primary", used_percent_to_bp(primary)))
    secondary = info.get("secondary")
    if secondary is not None:
        if (not isinstance(secondary, dict) or not _window_fields_ok(secondary)
                or not _percent_ok(secondary.get("used_percent"))
                # A healthy secondary must not stand in for a primary window
                # that is missing or was dropped as malformed upstream.
                or primary is None):
            ev.fail("malformed_telemetry")
            return ev
        reached = reached or secondary["reached"]
        values.append(("secondary", used_percent_to_bp(secondary["used_percent"])))
    age = (now - ev.observed_at).total_seconds()
    if (age < -policy["codex_rpc"]["future_tolerance_seconds"]
            or age > ev.freshness_limit_s):
        ev.fail("stale_telemetry")
        return ev
    if values:
        best = values[0]
        for item in values[1:]:
            if item[1] > best[1]:
                best = item
        ev.window, ev.used_bp = best
    if not values and not reached:
        ev.fail("missing_telemetry")
        return ev
    if reached or ev.used_bp >= reserve_bp:
        ev.fail("quota_reserve")
    return ev


def _codex_home_id(profile):
    """sha256 of the profile's resolved Codex home, as the RPC reader keys its
    cache (``None`` when the profile cannot be resolved)."""
    from providers.kernel_sandbox import protected_home
    from providers.profiles import resolve_profile
    overrides, profile_err = resolve_profile("codex", profile)
    if profile_err:
        return None
    home = os.path.realpath(overrides.get("CODEX_HOME") or os.environ.get("CODEX_HOME")
                            or os.path.join(protected_home(), ".codex"))
    return hashlib.sha256(home.encode("utf-8")).hexdigest()


def _info_from_cache_row(row):
    """The observation a cache row holds, field for field and unconverted:
    ``_evaluate_codex`` judges the types."""
    info = {key: row.get(key) for key in (
        "used_percent", "resets_at", "window_mins", "reached", "plan")}
    info["primary_reached"] = row.get("primary_reached", row.get("reached"))
    info["checked_at"] = health._as_timestamp(row.get("checked_at"))
    info["quota_unknown"] = row.get("quota_unknown", False)
    if row.get("secondary") is not None:
        info["secondary"] = row["secondary"]
    return info


def _codex_observation(row, pool, policy, now):
    """Return ``(info, err)`` for the Codex pool.

    *row* is this profile's cache row as the strict read returned it, or
    ``None`` when its file is absent. A fresh row bound to this profile's home
    is judged exactly as read: that object is the only thing decided on, it is
    never read again. Anything else (no row, stale, a future time beyond
    tolerance, another home, a disabled probe) goes to the RPC with ``force`` so
    its own cache read is not consulted either.
    """
    from providers import codex_quota
    profile = pool["profile"]
    settings = policy["codex_rpc"]
    if os.environ.get(codex_quota.ENV_QUOTA_PROBE) != "0":
        try:
            home_id = _codex_home_id(profile)
        except Exception:
            home_id = None
        if isinstance(row, dict) and home_id is not None and row.get("home_id") == home_id:
            checked = health._as_timestamp(row.get("checked_at"))
            if checked is not None:
                age = now.timestamp() - checked
                if -settings["future_tolerance_seconds"] <= age <= settings["cache_freshness_s"]:
                    return _info_from_cache_row(row), None
    return codex_quota.read_rate_limits(profile, timeout_s=settings["timeout_s"],
                                        force=True)


def _canonical_ledger_key(key):
    """The ledger key as ``health.pool_key`` spells it (``CURSOR/auto`` and
    ``cursor/auto`` both ``cursor:auto``). A key with no profile is ``default``."""
    pool, sep, profile = str(key).rpartition("@")
    if not sep:
        pool, profile = str(key), "default"
    return health.pool_key(pool, profile)


# provider_for_model answers ``openai`` for the road the policy calls ``codex``.
_PROVIDER_ROAD = {"openai": "codex"}


def _bare_model_is_other_road(model, road):
    """True when a bare model name unmistakably belongs to another quota road
    (``grok-4.6`` is grok, ``gpt-6.2-sol`` is codex). ``agy`` is the route's
    fallthrough for any unrecognised name, so it never proves another road."""
    from providers.route import provider_for_model
    provider = provider_for_model(model)
    if provider == "agy":
        return False
    return _PROVIDER_ROAD.get(provider, provider) != road


def _ledger_cooling(ledger, pool, model, now_ts):
    """True when an active ledger flag covers this quota pool.

    Only flags of the billed road match: its account key, a window of it
    (``week`` or ``m<minutes>``) and the requested model's own key. Ledger keys
    and the requested model are compared in ``health.pool_key`` form. A model
    that names another road (``codex/...`` or a bare ``grok-4.6`` on a claude
    pool) contributes no key of its own.
    """
    road, profile = pool["road"], pool["profile"]
    wanted = {health.pool_key(road, profile)}
    head = re.split(r"[/:]", model, maxsplit=1)[0] if re.search(r"[/:]", model) else None
    if head is None:
        bare = model
    elif head.lower() == road:
        # ``grok/grok-4.6`` and ``grok-4.6`` are one billed model: judge the
        # remainder exactly as a bare name on this road.
        bare = model[len(head) + 1:]
        wanted.add(health.pool_key(model, profile))
    else:
        bare = None  # another road's prefix contributes no key of its own
    if bare:
        # The transport fixes the billed road, so the model's key on that road
        # always counts (a bare vendor slug on cursor is a cursor pool key).
        wanted.add(health.pool_key("%s/%s" % (road, bare), profile))
        # The bare spelling counts only when it cannot be another road's model
        # or another road's own name.
        if bare == road or (bare not in health._KNOWN_ROADS
                            and not _bare_model_is_other_road(bare, road)):
            wanted.add(health.pool_key(bare, profile))
    for key, row in ledger.items():
        if key == health._LEASES_KEY or not health._is_active(row, now_ts):
            continue
        canonical = _canonical_ledger_key(key)
        if canonical in wanted or health._is_quota_window_key(canonical, road, profile):
            return True
    return False


# ---------------------------------------------------------------------------
# Incidents: one marker per (quota pool, account, kind) under the state dir
# ---------------------------------------------------------------------------
def _parse_instant(value):
    """An ISO 8601 instant that names its timezone (``Z`` or an offset), as UTC."""
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("instant")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("instant")
    return parsed.astimezone(timezone.utc)


def _ceil_second(moment):
    """*moment* rounded up to a whole second (stored and printed instants are
    whole seconds; rounding down would end an incident early)."""
    floor = moment.replace(microsecond=0)
    if moment.microsecond:
        try:
            return floor + timedelta(seconds=1)
        except OverflowError:
            return floor   # the last representable second; the cap still applies
    return moment


def incident_path(pool_id, account, kind):
    return os.path.join(health._state_dir(), INCIDENT_DIR,
                        "%s.%s.%s.json" % (pool_id, account, kind))


def _uuid_ok(value):
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
        return parsed is not None and parsed.version == 4 and str(parsed) == value
    except ValueError:
        return False


def _incident_ok(doc, pool_id, account, kind):
    """True when *doc* is a ``quota-incident/v1`` marker for exactly this key."""
    if not (isinstance(doc, dict) and set(doc) == set(INCIDENT_KEYS)):
        return False
    if (doc["interface_version"] != INCIDENT_INTERFACE or doc["quota_pool"] != pool_id
            or doc["account"] != account or doc["kind"] != kind
            or not _uuid_ok(doc["incident_id"])):
        return False
    try:
        _parse_instant(doc["recorded_at"])
        _parse_instant(doc["until"])
    except (ValueError, OverflowError):
        return False
    return True


def read_incident(pool_id, account, now, kind="hard_rate_limit"):
    """Return ``(state, until)`` for the marker of this quota pool and account.

    ``state`` is ``none`` (no marker file), ``active`` (``until`` after *now*),
    ``expired`` or ``unreadable``. Unreadable covers a symbolic link (opened
    without following it), a file that is not regular, more than 64 KiB, bad
    JSON (the strict decoder), a root that is not a marker for exactly this key,
    and a ``recorded_at`` or ``until`` that is not an instant with a timezone.
    """
    try:
        fd = os.open(incident_path(pool_id, account, kind),
                     os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return "none", None
    except OSError:
        return "unreadable", None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return "unreadable", None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(INCIDENT_MAX_BYTES + 1)
        if len(raw) > INCIDENT_MAX_BYTES:
            return "unreadable", None
        doc = health.strict_json_loads(raw.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return "unreadable", None
    finally:
        os.close(fd)
    if not _incident_ok(doc, pool_id, account, kind):
        return "unreadable", None
    until = _parse_instant(doc["until"])
    return ("active" if until > now else "expired"), until


def _append_line(path, doc):
    """Append one JSON line with the decision audit's rules: mode 0600, one
    previous generation above 5 MiB. Best effort, never raises."""
    try:
        os.makedirs(os.path.dirname(path), mode=_DIR_MODE, exist_ok=True)
        try:
            if os.path.getsize(path) > AUDIT_ROTATE_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        line = json.dumps(doc, separators=(",", ":"), ensure_ascii=True) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
        try:
            os.write(fd, line.encode("ascii"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _write_marker(path, doc):
    """Write *doc* to *path* atomically (temp file in the same directory, fsync,
    replace). Raises OSError (or ValueError) on any failure; the temp file is
    removed."""
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=_DIR_MODE, exist_ok=True)
    data = (json.dumps(doc, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".incident-", suffix=".tmp")
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_incident(login, account, kind, until):
    """Record one incident. Returns ``(report, exit_code, error_text)``.

    *until* is an aware datetime. An unknown login or an account the policy does
    not list is a usage error. A past *until* records an expired marker, which
    is how an active incident is cleared.
    """
    account = account or "default"
    try:
        policy, _revision = load_policy()
    except PolicyError:
        return None, EXIT_RECORD, "quota: cannot record the incident\n"
    pool_id, pool = _pool_for_login(policy, login)
    if pool is None:
        return None, EXIT_USAGE, ("omni quota: usage error: --login is not a login "
                                  "of the quota policy\n")
    if account not in pool["accounts"]:
        return None, EXIT_USAGE, ("omni quota: usage error: --account is not listed "
                                  "for this login\n")
    now = _now()
    # A future until is rounded up (the incident never ends early); a past one
    # is rounded down, so a clearing report clears at once.
    stored = _ceil_second(until) if until > now else until.replace(microsecond=0)
    marker = {
        "interface_version": INCIDENT_INTERFACE,
        "quota_pool": pool_id,
        "account": account,
        "kind": kind,
        "until": _iso(stored),
        "recorded_at": _iso(now),
        "incident_id": str(uuid.uuid4()),
    }
    if not _incident_ok(marker, pool_id, account, kind):
        return None, EXIT_RECORD, "quota: cannot record the incident\n"
    try:
        _write_marker(incident_path(pool_id, account, kind), marker)
    except Exception:
        return None, EXIT_RECORD, "quota: cannot record the incident\n"
    _append_line(os.path.join(health._state_dir(), INCIDENT_LOG), marker)
    report = {"interface_version": INCIDENT_INTERFACE, "ok": True}
    for key in REPORT_KEYS[2:]:
        report[key] = login if key == "login" else marker[key]
    return report, EXIT_ELIGIBLE, None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def _build_document(decision, reasons, pool_id, login, account, ev, revision,
                    now, cap_s, expires_at=None):
    """Build the one allowlisted output object, validating every value.
    *expires_at*, when given, is the already computed expiry (an aware datetime)."""
    remaining = float(cap_s)
    used_bp = window = observed_at = freshness_s = None
    if ev is not None:
        used_bp, window = ev.used_bp, ev.window
        if ev.observed_at is not None:
            observed_at = _iso(ev.observed_at)
            age = (now - ev.observed_at).total_seconds()
            freshness_s = max(0, int(age))
            if ev.freshness_limit_s is not None:
                remaining = min(remaining, max(0.0, ev.freshness_limit_s - max(age, 0.0)))
    if any(reason in NO_REUSE_REASONS for reason in reasons):
        # Invalid, stale or unverifiable evidence is never reusable, whatever
        # the observation time says (a future one included).
        remaining = 0.0
    expires = expires_at or now + timedelta(seconds=int(remaining))
    doc = {
        "interface_version": INTERFACE_VERSION,
        "decision_id": str(uuid.uuid4()),
        "decision": decision,
        "reasons": list(reasons),
        "quota_pool": pool_id,
        "login": login,
        "account": account,
        "used_bp": used_bp,
        "window": window,
        "observed_at": observed_at,
        "freshness_s": freshness_s,
        "policy_revision": revision,
        "checked_at": _iso(now),
        "expires_at": _iso(expires),
    }
    return validate_document(doc)


def validate_document(doc):
    """Raise ``_OutputInvalid`` unless every field holds an allowlisted value."""
    def need(ok):
        if not ok:
            raise _OutputInvalid()

    need(isinstance(doc, dict) and tuple(doc) == DOC_KEYS)
    need(doc["interface_version"] == INTERFACE_VERSION)
    ident = doc["decision_id"]
    try:
        need(isinstance(ident, str) and str(uuid.UUID(ident)) == ident)
    except ValueError:
        raise _OutputInvalid()
    need(doc["decision"] in DECISIONS)
    reasons = doc["reasons"]
    need(isinstance(reasons, list) and 1 <= len(reasons) <= 8
         and len(set(reasons)) == len(reasons)
         and all(isinstance(r, str) and r in REASONS for r in reasons))
    for key in ("quota_pool", "login", "account"):
        need(doc[key] is None or _id_ok(doc[key]))
    used = doc["used_bp"]
    need(used is None or _int_in(used, 0, 10000))
    need(doc["window"] is None or (isinstance(doc["window"], str)
                                   and _WINDOW_RE.fullmatch(doc["window"])))
    for key in ("observed_at", "checked_at", "expires_at"):
        value = doc[key]
        need(value is None or (isinstance(value, str) and _ISO_RE.fullmatch(value)))
    need(doc["checked_at"] is not None and doc["expires_at"] is not None)
    need(doc["freshness_s"] is None or _int_in(doc["freshness_s"], 0, 10 ** 9))
    need(doc["policy_revision"] is None
         or (isinstance(doc["policy_revision"], str)
             and _REVISION_RE.fullmatch(doc["policy_revision"])))
    return doc


def decide(login, account, transport, model, task_class, context,
           snapshot_path=None, policy=None, revision=None, notes=None):
    """Return ``(document, exit_code)`` for one request. Never raises.

    *transport* is validated by the caller and reserved: the decision is about the
    billed (login, account), not the transport. *policy* and *revision* exist for
    tests; a normal call loads the shipped policy. *notes*, when a dict, receives
    audit-only fields (``incident_state``); they are never printed.
    """
    try:
        return _decide(login, account, model, task_class, context,
                       snapshot_path, policy, revision, notes)
    except Exception:
        # Anything unforeseen refuses: never an eligible decision by accident.
        try:
            doc = _build_document("ineligible", ["registry_invalid"], None, None,
                                  None, None, None, _now(), 300)
        except Exception:
            doc = {key: None for key in DOC_KEYS}
            doc.update({"interface_version": INTERFACE_VERSION,
                        "decision": "ineligible", "reasons": ["registry_invalid"]})
        return doc, EXIT_INELIGIBLE


def _decide(login, account, model, task_class, context, snapshot_path, policy,
            revision, notes=None):
    if policy is None:
        try:
            policy, revision = load_policy()
        except PolicyError:
            return _refusal("registry_invalid", None, None, None, None)
    reserve_bp = policy["reserve_threshold_bp"]
    cap_s = policy["decision_ttl_cap_s"]
    account = account or "default"
    pool_id, pool = _pool_for_login(policy, login)
    if pool is None:
        return _refusal("unknown_login", None, None, None, revision, cap_s)
    if account not in pool["accounts"]:
        # Never borrows another account's telemetry: refuse before reading any.
        return _refusal("unknown_account", pool_id, login, None, revision, cap_s)

    # A recorded hard rate limit is checked before any telemetry and before the
    # unmonitored rule, as in the caller, and is never softened for a short task.
    now = _now()
    incident, until = read_incident(pool_id, account, now)
    if incident != "none" and isinstance(notes, dict):
        notes["incident_state"] = incident
    if incident == "active":
        # One clock sample: expires at the earlier of until (rounded up to the
        # second, so a printed expiry never precedes it) and checked_at + cap.
        capped = now.replace(microsecond=0) + timedelta(seconds=cap_s)
        return _refusal("hard_rate_limit", pool_id, login, account, revision, cap_s,
                        now=now, expires_at=min(_ceil_second(until), capped))
    if incident == "unreadable":
        # Fail closed, as the caller does with a corrupt marker; never reusable.
        return _refusal("hard_rate_limit", pool_id, login, account, revision, 0,
                        now=now)

    try:
        ledger = health.strict_ledger_snapshot()
    except health.StrictReadError as exc:
        return _refusal(exc.reason, pool_id, login, account, revision, cap_s)

    monitoring = pool["monitoring"]
    info = err = snapshot = snap_reason = None
    if monitoring == "codex_rpc":
        try:
            row = health.strict_quota_cache_row(pool["profile"])
        except health.StrictReadError as exc:
            return _refusal(exc.reason, pool_id, login, account, revision, cap_s)
        try:
            info, err = _codex_observation(row, pool, policy, _now())
        except Exception:
            info, err = None, {"ok": False}
    elif monitoring == "anthropic_snapshot":
        snapshot, snap_reason = _read_snapshot_file(snapshot_path)

    now = _now()
    failures = []   # [(reason, soft)]
    try:
        if _ledger_cooling(ledger, pool, model, now.timestamp()):
            failures.append(("ledger_cooling", False))
    except Exception:
        return _refusal("ledger_unreadable", pool_id, login, account, revision, cap_s)

    ev = None
    if monitoring == "codex_rpc":
        ev = _evaluate_codex(info, err, policy, reserve_bp, now)
    elif monitoring == "anthropic_snapshot":
        ev = _evaluate_anthropic(snapshot, snap_reason, policy, reserve_bp, now,
                                 account)
    if ev is not None:
        failures.extend(ev.reasons)
    elif context == "fallback":
        # An unmonitored pool is never a fallback target (as in the caller).
        failures.append(("unmonitored_pool", False))

    if not failures:
        reasons = ["unmonitored_pool"] if monitoring == "unmonitored" else ["ok"]
        decision, code = "eligible", EXIT_ELIGIBLE
    else:
        reasons = []
        for reason, _soft in failures:
            if reason not in reasons:
                reasons.append(reason)
        if task_class == "short" and all(reason in SOFT_REASONS for reason, _soft in failures):
            decision, code = "advisory", EXIT_ELIGIBLE
        else:
            decision, code = "ineligible", EXIT_INELIGIBLE
    doc = _build_document(decision, reasons, pool_id, login, account, ev, revision,
                          now, cap_s)
    return doc, code


def _refusal(reason, pool_id, login, account, revision, cap_s=300, now=None,
             expires_at=None):
    doc = _build_document("ineligible", [reason], pool_id, login, account, None,
                          revision, now or _now(), cap_s, expires_at)
    return doc, EXIT_INELIGIBLE


# ---------------------------------------------------------------------------
# Audit: one allowlisted line per decision, size-capped rotation
# ---------------------------------------------------------------------------
def audit_path():
    return os.path.join(health._state_dir(), AUDIT_FILE)


def append_audit(doc, notes=None):
    """Append *doc* (already validated) to the audit log, plus the allowlisted
    audit-only ``incident_state`` from *notes*. Best effort, never raises."""
    try:
        validate_document(doc)
        line = dict(doc)
        state = notes.get("incident_state") if isinstance(notes, dict) else None
        if state in INCIDENT_STATES:
            line["incident_state"] = state
        _append_line(audit_path(), line)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_FIXED_ERRORS = {
    "argument --login: must match [A-Za-z0-9_-]+ (at most 64 characters)":
        "--login must match [A-Za-z0-9_-]+ (at most 64 characters)",
    "argument --account: must match [A-Za-z0-9_-]+ (at most 64 characters)":
        "--account must match [A-Za-z0-9_-]+ (at most 64 characters)",
    "argument --transport: invalid value": "--transport is invalid",
    "argument --model: invalid value": "--model is invalid",
    "argument --task-class: must be long or short": "--task-class must be long or short",
    "argument --context: must be primary or fallback":
        "--context must be primary or fallback",
    "argument --anthropic-snapshot: invalid value": "--anthropic-snapshot is invalid",
    "argument --kind: must be hard_rate_limit": "--kind must be hard_rate_limit",
    "argument --until: must be an ISO 8601 instant with a timezone":
        "--until must be an ISO 8601 instant with a timezone",
}
_REQUIRED_RE = re.compile(r"the following arguments are required: [-A-Za-z_, ]+")


class _QuotaParser(argparse.ArgumentParser):
    """Usage errors print fixed text and never echo an argument value."""

    def error(self, message):
        if message in _FIXED_ERRORS:
            text = _FIXED_ERRORS[message]
        elif _REQUIRED_RE.fullmatch(message):
            text = message
        else:
            text = "invalid arguments"
        self.exit(EXIT_USAGE, "omni quota: usage error: %s\n" % text)


def _id_arg(name):
    def parse(value):
        if not _id_ok(value):
            raise argparse.ArgumentTypeError(
                "must match [A-Za-z0-9_-]+ (at most 64 characters)")
        return value
    parse.__name__ = name
    return parse


def _ref_arg(value):
    if not isinstance(value, str) or _REF_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid value")
    return value


def _path_arg(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        raise argparse.ArgumentTypeError("invalid value")
    return value


def _task_class_arg(value):
    if value not in TASK_CLASSES:
        raise argparse.ArgumentTypeError("must be long or short")
    return value


def _context_arg(value):
    if value not in CONTEXTS:
        raise argparse.ArgumentTypeError("must be primary or fallback")
    return value


def _kind_arg(value):
    if value not in INCIDENT_KINDS:
        raise argparse.ArgumentTypeError("must be hard_rate_limit")
    return value


def _until_arg(value):
    try:
        moment = _parse_instant(value)
        if _ISO_RE.fullmatch(_iso(moment)) is None:
            raise ValueError("instant")
    except (ValueError, OverflowError, TypeError):
        raise argparse.ArgumentTypeError(
            "must be an ISO 8601 instant with a timezone")
    return moment


def build_parser(prog="omni quota"):
    parser = _QuotaParser(
        prog=prog,
        description="Quota-pool eligibility (quota-decision/v1). Exit 0 eligible or "
                    "advisory, 10 ineligible, 2 usage error; report-incident exits "
                    "73 when its marker cannot be written.")
    sub = parser.add_subparsers(dest="quota_cmd", required=True, parser_class=_QuotaParser)
    check = sub.add_parser("check", allow_abbrev=False,
                           help="decide quota eligibility for one login/account")
    check.add_argument("--login", required=True, type=_id_arg("login"))
    check.add_argument("--account", default=None, type=_id_arg("account"),
                       help="account alias; omitted means 'default'")
    check.add_argument("--transport", required=True, type=_ref_arg)
    check.add_argument("--model", required=True, type=_ref_arg)
    check.add_argument("--task-class", required=True, type=_task_class_arg)
    check.add_argument("--context", required=True, type=_context_arg)
    check.add_argument("--anthropic-snapshot", default=None, type=_path_arg,
                       help="path of the normalized Anthropic usage snapshot")
    check.add_argument("--json", action="store_true", required=True)
    policy = sub.add_parser("policy", help="inspect the shipped quota policy")
    psub = policy.add_subparsers(dest="policy_cmd", required=True,
                                 parser_class=_QuotaParser)
    show = psub.add_parser("show", help="print the effective login/account map")
    show.add_argument("--json", action="store_true", required=True)
    report = sub.add_parser("report-incident", allow_abbrev=False,
                            help="record a hard rate limit for one login/account")
    report.add_argument("--login", required=True, type=_id_arg("login"))
    report.add_argument("--account", default=None, type=_id_arg("account"),
                        help="account alias; omitted means 'default'")
    report.add_argument("--kind", required=True, type=_kind_arg)
    report.add_argument("--until", required=True, type=_until_arg,
                        help="ISO 8601 instant with a timezone; a past one clears")
    report.add_argument("--json", action="store_true", required=True)
    return parser


def policy_show_document(policy, revision):
    rows = []
    for pool_id in sorted(policy["pools"]):
        pool = policy["pools"][pool_id]
        for account in pool["accounts"]:
            rows.append({"login": pool["login"], "account": account,
                         "quota_pool": pool_id, "monitoring": pool["monitoring"]})
    rows.sort(key=lambda r: (r["login"], r["account"]))
    return {
        "interface_version": INTERFACE_VERSION,
        "policy_revision": revision,
        "reserve_threshold_bp": policy["reserve_threshold_bp"],
        "map": rows,
    }


def _output_failed():
    """Say once, in fixed text, that the result could not be written, and park
    stdout on the null device so the interpreter's exit flush stays silent."""
    try:
        sys.stderr.write("quota: cannot write the result\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, sys.stdout.fileno())
        finally:
            os.close(null)
    except Exception:
        pass


def _emit(doc):
    """Write one JSON line to stdout. False (after a fixed stderr message) when
    stdout is closed or fails: no traceback, no path."""
    try:
        sys.stdout.write(json.dumps(doc, separators=(",", ":"), ensure_ascii=True) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        _output_failed()
        return False
    return True


def run(args):
    """Run an already parsed ``quota`` command. Returns the exit code."""
    if args.quota_cmd == "policy":
        try:
            policy, revision = load_policy()
        except PolicyError:
            if not _emit({"interface_version": INTERFACE_VERSION,
                          "policy_revision": None, "reserve_threshold_bp": None,
                          "map": [], "reasons": ["registry_invalid"]}):
                return EXIT_OUTPUT
            return EXIT_INELIGIBLE
        if not _emit(policy_show_document(policy, revision)):
            return EXIT_OUTPUT
        return EXIT_ELIGIBLE
    if args.quota_cmd == "report-incident":
        try:
            report, code, error = record_incident(args.login, args.account,
                                                  args.kind, args.until)
        except Exception:
            report, code, error = None, EXIT_RECORD, "quota: cannot record the incident\n"
        if report is None:
            try:
                sys.stderr.write(error)
                sys.stderr.flush()
            except Exception:
                pass
            return code
        if not _emit(report):
            return EXIT_OUTPUT
        return code
    notes = {}
    doc, code = decide(args.login, args.account, args.transport, args.model,
                       args.task_class, args.context, args.anthropic_snapshot,
                       notes=notes)
    append_audit(doc, notes)
    if not _emit(doc):
        return EXIT_OUTPUT
    return code


def main(argv=None):
    """Entry for ``omni quota ...``: parse *argv* (without ``quota``) and run it."""
    try:
        args = build_parser().parse_args(argv)
    except OSError:
        # Unbuffered stdout (``python3 -u``): argparse's own help or usage write
        # raises here instead of at the flush below.
        _output_failed()
        return EXIT_OUTPUT
    except SystemExit as exc:
        # ``--help`` and usage errors write through argparse and leave the text
        # buffered: flush it here so a closed stdout is reported, not raised at
        # interpreter exit.
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            _output_failed()
            return EXIT_OUTPUT
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    return run(args)
