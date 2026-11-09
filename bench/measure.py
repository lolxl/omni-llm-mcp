# no-opencc
"""Hermetic measurements for the pre-instrumentation baseline (P1.2a/P1.2b)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
BASELINE_PATH = os.path.join(HERE, "baseline.json")
LIVE_REPORT_PATH = os.path.join(HERE, "live-report.json")

PATH_NAMES = ("consultation", "big_context", "fallback_chain", "large_result")

USAGE_UNAVAILABLE = {
    "input_tokens": None,
    "cached_input_tokens": None,
    "cache_write_tokens": None,
    "output_tokens": None,
    "source": "unavailable",
}

# Immutable P1.2a pre-instrumentation reference (seeded once; never mutated).
# Numbers from commit 0cc6639 baseline, measured at f045286.
P12A_REFERENCE = {
    "tools_compact_bytes": 9625,
    "tools_list_wire_bytes": {
        "legacy": 10111,
        "modern": 10111,
    },
    "commit": "f04528638f42181aafe10ad5ff229bd10ee13a50",
}

# Modern-era keys (must match server.py; read-only consumption).
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
SUPPORTED_MODERN = "2026-07-28"
SUPPORTED_LEGACY = "2025-06-18"


class BaselineImmutabilityError(ValueError):
    """Typed refusal when an update would mutate immutable baseline fields."""


class FixtureLoadError(ValueError):
    """Typed harness failure when a fixture cannot be loaded during comparison.

    Harness/internal error (exit 2). Never silently fall back to baseline values.
    """


def _ensure_root_on_path():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, "%s.json" % name)


def load_fixture(name: str) -> Dict[str, Any]:
    path = fixture_path(name)
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                "fixture %s (%s): %s" % (name, path, e.msg),
                e.doc,
                e.pos,
            ) from e
    if not isinstance(data, dict):
        raise ValueError("fixture %s is not an object" % name)
    return data


def fixture_hashes() -> Dict[str, str]:
    return {name: sha256_file(fixture_path(name)) for name in PATH_NAMES}


def git_commit() -> str:
    """Resolve HEAD for the environment manifest; fail loud if unresolvable."""
    env = os.environ.copy()
    # Avoid sandbox trips on inaccessible user gitconfig.
    env.setdefault("GIT_CONFIG_GLOBAL", os.devnull)
    env.setdefault("GIT_CONFIG_SYSTEM", os.devnull)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            env=env,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        commit = out.strip()
        if not commit:
            raise RuntimeError("git rev-parse HEAD returned empty output")
        return commit
    except Exception as e:
        raise RuntimeError(
            "git_commit failed: unable to resolve HEAD at %s: %s" % (ROOT, e)
        ) from e


def tools_compact_bytes() -> int:
    _ensure_root_on_path()
    from providers.dispatch import TOOLS  # read-only import
    return len(json.dumps(TOOLS, separators=(",", ":")))


def _modern_meta() -> Dict[str, Any]:
    return {
        META_PROTOCOL_VERSION: SUPPORTED_MODERN,
        META_CLIENT_CAPABILITIES: {},
    }


def tools_list_wire_bytes() -> Dict[str, int]:
    """Full JSON-RPC tools/list response byte lengths for both protocol eras.

    Consumes tests.test_protocol_2026.exchange (parameterized for raw lines) — M7;
    no copied capture block.
    """
    _ensure_root_on_path()
    from tests.test_protocol_2026 import exchange  # noqa: WPS433

    def _run(messages: List[Dict[str, Any]], list_id: int) -> int:
        lines = exchange(messages, raw_lines=True)
        for ln in lines:
            try:
                msg = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == list_id and "result" in msg:
                # Wire bytes = UTF-8 of the JSON-RPC response line body.
                return len(ln.encode("utf-8"))
        raise RuntimeError(
            "tools/list response id=%s not found in %r" % (list_id, lines)
        )

    legacy = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": SUPPORTED_LEGACY,
                    "capabilities": {},
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
        list_id=2,
    )
    modern = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": {"_meta": _modern_meta()},
            },
        ],
        list_id=3,
    )
    return {"legacy": legacy, "modern": modern}


def _project_usage(usage: Any) -> Dict[str, Any]:
    """Baseline usage slot: allowlisted fields only (no raw).

    Uses providers.usage semantics for source/token meaning; projects to the
    five baseline keys. Honest nulls when the envelope has no usage.
    """
    if not isinstance(usage, dict):
        return dict(USAGE_UNAVAILABLE)
    out = {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "cache_write_tokens": usage.get("cache_write_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "source": usage.get("source") or "unavailable",
    }
    # If every token field is None and source is missing/empty, stay unavailable.
    token_keys = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
    )
    if all(out[k] is None for k in token_keys) and out["source"] not in (
        "provider",
        "estimated",
    ):
        out["source"] = "unavailable"
    return out


def run_path(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute one hermetic path under the fake backend. Returns (record, envelope)."""
    from bench.checkers import run_checker
    from bench.fake_backend import backend_from_fixture

    _ensure_root_on_path()
    from providers import dispatch  # noqa: WPS433
    from providers import health  # noqa: WPS433

    fixture = load_fixture(name)
    backend = backend_from_fixture(fixture)
    tool = fixture["tool"]
    arguments = dict(fixture.get("arguments") or {})

    # Isolate seat-health ledger so chain runs stay deterministic.
    try:
        health.clear_all()
    except Exception as e:
        raise RuntimeError(
            "health.clear_all() failed; isolation broken: %s" % e
        ) from e

    t0 = time.perf_counter()
    with mock.patch.object(dispatch, "route_to_provider", backend), \
            mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0):
        envelope = asyncio_run_call_tool(dispatch, tool, arguments)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    attempts = backend.attempts
    fallback_count = backend.fallback_count

    # P1.4: large_result exercises ref mode; rehydrate full body for the
    # path checker (which asserts marker + min length on the raw text).
    check_envelope = envelope
    if name == "large_result" and isinstance(envelope, dict):
        ref = envelope.get("artifact_ref")
        if isinstance(ref, dict) and ref.get("id"):
            from providers import artifacts as artifacts_mod  # noqa: WPS433
            body = artifacts_mod.read(ref["id"])
            if body.get("ok") and isinstance(body.get("text"), str):
                check_envelope = dict(envelope)
                check_envelope["text"] = body["text"]

    checker_status = run_checker(name, check_envelope, fixture)

    # M4 single truth: expected_attempts / expected_fallback_count are NOT
    # asserted here. They participate in compare_baseline (REGRESSION line,
    # exit 1) instead of raising AssertionError before the comparator.

    record = {
        "latency_ms": latency_ms,
        "attempts": attempts,
        "fallback_count": fallback_count,
        "ok": bool(envelope.get("ok")),
        "checker": checker_status,
    }
    return record, envelope


def asyncio_run_call_tool(dispatch, tool, arguments):
    import asyncio
    return asyncio.run(dispatch.call_tool(tool, arguments))


def measure_all() -> Dict[str, Any]:
    """Produce a full baseline-shaped measurement of the CURRENT tree."""
    _ensure_root_on_path()
    # Import server BEFORE any asyncio.run(): on Python 3.9, server module-level
    # asyncio.Semaphore() needs a living event-loop policy at import time.
    import server  # noqa: F401,WPS433

    # Schema / wire metrics first (no provider calls).
    compact = tools_compact_bytes()
    wire = tools_list_wire_bytes()
    hashes = fixture_hashes()

    paths = {}
    usage = {}
    for name in PATH_NAMES:
        record, envelope = run_path(name)
        paths[name] = record
        usage[name] = _project_usage(envelope.get("usage"))

    return {
        "baseline_schema_version": 1,
        "commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "fixture_hashes": hashes,
        "tools_compact_bytes": compact,
        "tools_list_wire_bytes": wire,
        "paths": paths,
        "usage": usage,
    }


# --------------------------------------------------------------------------- #
# Baseline I/O + comparison
# --------------------------------------------------------------------------- #

# Fields compared for regression. latency_ms is observational only (wall clock).
# commit / python_version / platform are environment manifest, not gates.
# p12a_reference / history are durability metadata, not metric gates.
GATED_METRIC_PATHS = [
    ("baseline_schema_version",),
    ("fixture_hashes", "consultation"),
    ("fixture_hashes", "big_context"),
    ("fixture_hashes", "fallback_chain"),
    ("fixture_hashes", "large_result"),
    ("tools_compact_bytes",),
    ("tools_list_wire_bytes", "legacy"),
    ("tools_list_wire_bytes", "modern"),
]
for _p in PATH_NAMES:
    GATED_METRIC_PATHS.extend([
        ("paths", _p, "attempts"),
        ("paths", _p, "fallback_count"),
        ("paths", _p, "ok"),
        ("paths", _p, "checker"),
        ("usage", _p, "input_tokens"),
        ("usage", _p, "cached_input_tokens"),
        ("usage", _p, "cache_write_tokens"),
        ("usage", _p, "output_tokens"),
        ("usage", _p, "source"),
    ])

_FIXTURE_EXPECTED_KEYS = {
    "attempts": "expected_attempts",
    "fallback_count": "expected_fallback_count",
}


def _metric_name(path: Tuple[str, ...]) -> str:
    return ".".join(str(p) for p in path)


def _lookup(doc: Dict[str, Any], path: Tuple[str, ...]):
    cur: Any = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _validate_p12a_reference(ref: Any, *, context: str) -> Dict[str, Any]:
    """Require ref to be exactly the immutable P12A_REFERENCE constant.

    Divergence, wrong type, or otherwise corrupt block → BaselineImmutabilityError.
    """
    if not isinstance(ref, dict) or ref != P12A_REFERENCE:
        raise BaselineImmutabilityError(
            "p12a_reference is immutable and must match P12A_REFERENCE; "
            "refusing %s (divergent or corrupt)" % context
        )
    return ref


def _validate_history(history: Any, *, context: str) -> List[Any]:
    """History must be a list when present. Corrupt non-list → typed error."""
    if not isinstance(history, list):
        raise BaselineImmutabilityError(
            "history must be a list; refusing %s (present but non-list)" % context
        )
    return history


def load_baseline(path: str = BASELINE_PATH) -> Dict[str, Any]:
    """Load baseline JSON with loud validation of durability fields.

    When ``p12a_reference`` is present it must equal ``P12A_REFERENCE``.
    When ``history`` is present it must be a list (corrupt non-list refused).
    Missing history is legal (pre-P1.2b empty-start); missing p12a_reference
    is legal on load so a first write can seed it — the gate helper refuses
    absence separately.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise BaselineImmutabilityError(
            "baseline at %s is not a JSON object" % path
        )
    if "p12a_reference" in doc:
        _validate_p12a_reference(doc["p12a_reference"], context="load of %s" % path)
    if "history" in doc:
        _validate_history(doc["history"], context="load of %s" % path)
    return doc


def p12a_reference_tools_compact_bytes(path: str = BASELINE_PATH) -> int:
    """P1.3 gate helper: reference tools_compact_bytes from p12a_reference.

    Returns the immutable P1.2a figure, NOT the current baseline top-level value.
    When the baseline file exists, ``p12a_reference`` must be present and equal
    ``P12A_REFERENCE`` — absent/corrupt/divergent blocks raise
    BaselineImmutabilityError (no silent fallback). Missing file still returns
    the seeded constant (pre-seed path).
    """
    if not os.path.isfile(path):
        return int(P12A_REFERENCE["tools_compact_bytes"])
    doc = load_baseline(path)
    if "p12a_reference" not in doc:
        raise BaselineImmutabilityError(
            "p12a_reference block absent in %s; refusing silent fallback" % path
        )
    ref = _validate_p12a_reference(
        doc["p12a_reference"], context="gate helper on %s" % path
    )
    return int(ref["tools_compact_bytes"])


def _headline_metrics(measured: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tools_compact_bytes": measured.get("tools_compact_bytes"),
        "tools_list_wire_bytes": measured.get("tools_list_wire_bytes"),
    }


def write_baseline(measured: Dict[str, Any], reason: str, path: str = BASELINE_PATH) -> None:
    """Write baseline, seeding p12a_reference once and appending history.

    Refuses (typed BaselineImmutabilityError) if an update would mutate
    p12a_reference or if the on-disk reference diverges from P12A_REFERENCE.
    History is append-only — prior entries are never dropped. Corrupt history
    (present but non-list) is refused; missing history is a legal empty-start.
    """
    existing = None
    if os.path.isfile(path):
        existing = load_baseline(path)

    doc = dict(measured)
    doc["update_reason"] = reason

    # --- p12a_reference: seed once, never mutate; always == P12A_REFERENCE ---
    incoming = measured.get("p12a_reference") if isinstance(measured, dict) else None
    if incoming is not None:
        _validate_p12a_reference(incoming, context="--update-baseline incoming seed")

    if existing and "p12a_reference" in existing:
        existing_ref = _validate_p12a_reference(
            existing["p12a_reference"],
            context="--update-baseline of existing %s" % path,
        )
        if incoming is not None and incoming != existing_ref:
            raise BaselineImmutabilityError(
                "p12a_reference is immutable; refusing to modify it via --update-baseline"
            )
        doc["p12a_reference"] = dict(existing_ref)
    else:
        # First seed (or pre-P1.2b file without the block).
        doc["p12a_reference"] = dict(P12A_REFERENCE)

    # --- history: append-only; corrupt non-list refused; missing = empty-start ---
    if existing is not None and "history" in existing:
        prior = list(
            _validate_history(
                existing["history"],
                context="--update-baseline of existing %s" % path,
            )
        )
    else:
        prior = []
    entry = {
        "commit": measured.get("commit"),
        "reason": reason,
    }
    entry.update(_headline_metrics(measured))
    doc["history"] = prior + [entry]

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def compare_baseline(
    measured: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    path: str = BASELINE_PATH,
) -> List[Dict[str, Any]]:
    """Return list of regressions {name, baseline, measured}. Empty = pass.

    M4 single truth for attempts/fallback_count: when a fixture declares
    expected_attempts / expected_fallback_count, that value is the comparison
    baseline for the corresponding gated path field (REGRESSION line, not
    AssertionError).
    """
    if baseline is None:
        baseline = load_baseline(path)
    regressions = []
    for mpath in GATED_METRIC_PATHS:
        bval = _lookup(baseline, mpath)
        mval = _lookup(measured, mpath)
        # Single-truth override: fixture expected_* wins for attempt gates.
        if (
            len(mpath) == 3
            and mpath[0] == "paths"
            and mpath[2] in _FIXTURE_EXPECTED_KEYS
        ):
            try:
                fixture = load_fixture(mpath[1])
            except (OSError, ValueError, json.JSONDecodeError) as e:
                # LOUD typed failure — never fixture={} silent baseline fallback.
                raise FixtureLoadError(
                    "fixture %s failed to load during comparison: %s"
                    % (mpath[1], e)
                ) from e
            exp_key = _FIXTURE_EXPECTED_KEYS[mpath[2]]
            if exp_key in fixture and fixture[exp_key] is not None:
                bval = fixture[exp_key]
        if bval != mval:
            regressions.append({
                "name": _metric_name(mpath),
                "baseline": bval,
                "measured": mval,
            })
    return regressions


def format_pass_lines(measured: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    lines = []
    for mpath in GATED_METRIC_PATHS:
        name = _metric_name(mpath)
        val = _lookup(measured, mpath)
        lines.append("PASS %s = %s" % (name, json.dumps(val)))
    return lines


# --------------------------------------------------------------------------- #
# Live lane (observational smoke — never merges into baseline.json)
# --------------------------------------------------------------------------- #

_CLI_NAMES = ("agy", "grok", "claude", "codex")


def collect_cli_versions(*, execute: bool = True) -> Dict[str, Optional[str]]:
    """Best-effort CLI version map for live-report context. No network.

    When ``execute`` is False (hermetic / fake-backend path), do not spawn local
    provider CLIs — return an all-None map so the suite stays offline.
    """
    if not execute:
        return {name: None for name in _CLI_NAMES}
    versions: Dict[str, Optional[str]] = {}
    for name in _CLI_NAMES:
        path = shutil.which(name)
        if not path:
            versions[name] = None
            continue
        try:
            out = subprocess.check_output(
                [path, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            versions[name] = (out or "").strip().splitlines()[0][:200] or None
        except Exception:
            # Probe failed: typed marker, never a fake version string like "present".
            versions[name] = "unavailable"
    return versions


def run_live_path(
    name: str,
    seat: str,
    *,
    fake_backend=None,
) -> Dict[str, Any]:
    """Run one path against a real (or injected fake) seat. Observational only."""
    import asyncio

    _ensure_root_on_path()
    from providers import dispatch  # noqa: WPS433
    from providers import health  # noqa: WPS433

    fixture = load_fixture(name)
    tool = fixture["tool"]
    arguments = dict(fixture.get("arguments") or {})
    fixture_model = arguments.get("model")
    # Live: drive the path's tool with the requested seat as model.
    # This collapses multi-seat fixtures (fallback_chain) to a single seat —
    # label that path honestly so it is never mistaken for a real chain run.
    arguments["model"] = seat

    try:
        health.clear_all()
    except Exception as e:
        raise RuntimeError(
            "health.clear_all() failed; isolation broken: %s" % e
        ) from e

    t0 = time.perf_counter()
    if fake_backend is not None:
        with mock.patch.object(dispatch, "route_to_provider", fake_backend), \
                mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0):
            envelope = asyncio.run(dispatch.call_tool(tool, arguments))
    else:
        envelope = asyncio.run(dispatch.call_tool(tool, arguments))
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    # R6: honest label when live seat override collapses a multi-seat fixture.
    path_label = name
    seat_override = None
    if name == "fallback_chain":
        path_label = "fallback_chain(single-seat)"
        seat_override = seat

    entry = {
        "path": path_label,
        "seat": seat,
        "ok": bool(envelope.get("ok")),
        "latency_ms": latency_ms,
        "usage": _project_usage(envelope.get("usage")),
        "model": envelope.get("model") or seat,
        "reason": envelope.get("reason") if not envelope.get("ok") else None,
    }
    if seat_override is not None:
        entry["seat_override"] = seat_override
        entry["fixture_model"] = fixture_model
    return entry


def measure_live(
    seats: List[str],
    *,
    fake_backend=None,
    cli_versions: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Run all four paths against each live seat. Never gates, never baselines.

    When ``fake_backend`` is injected, local provider CLIs are not executed for
    version collection (F6); pass ``cli_versions`` to supply an explicit map.
    """
    _ensure_root_on_path()
    import server  # noqa: F401,WPS433

    if not seats:
        raise ValueError("measure_live requires at least one seat")

    if cli_versions is not None:
        versions = dict(cli_versions)
    elif fake_backend is not None:
        # Hermetic path: never spawn provider CLIs.
        versions = collect_cli_versions(execute=False)
    else:
        versions = collect_cli_versions(execute=True)
    # R7: same normalized (stripped) seat value for iteration and report["seats"].
    normalized_seats: List[str] = []
    for seat in seats:
        seat = str(seat).strip()
        if not seat:
            continue
        normalized_seats.append(seat)
    results = []
    for seat in normalized_seats:
        for name in PATH_NAMES:
            entry = run_live_path(name, seat, fake_backend=fake_backend)
            entry["cli_versions"] = dict(versions)
            results.append(entry)

    return {
        "schema_version": 1,
        "commit": git_commit(),
        "seats": list(normalized_seats),
        "cli_versions": versions,
        "results": results,
    }


def write_live_report(
    report: Dict[str, Any],
    path: str = LIVE_REPORT_PATH,
    *,
    baseline_path: str = BASELINE_PATH,
) -> None:
    """Write live-report.json. NEVER merges into baseline.json."""
    if os.path.abspath(path) == os.path.abspath(baseline_path):
        raise ValueError("refusing to write live report over baseline.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
