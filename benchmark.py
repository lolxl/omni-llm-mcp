#!/usr/bin/env python3
# no-opencc
"""Hermetic + live baseline harness (issue #22 P1.2a/P1.2b).

Usage:
  python3 benchmark.py --hermetic --update-baseline "reason"
  python3 benchmark.py --hermetic
  python3 benchmark.py --live --live-seats <model[,model...]>
  python3 benchmark.py --live          # typed refusal: needs --live-seats

Hermetic mode: no network, no CLI logins, isolated OMNI_LLM_STATE_DIR (tempdir).
Baseline is written only with --update-baseline; otherwise compared for regressions.
Live mode: observational smoke against real backends; writes bench/live-report.json
and NEVER merges into baseline.json / never gates.

Exit codes:
  0 — no regression (or baseline/live report written successfully)
  1 — metric regression vs baseline
  2 — harness/internal error (git pin, isolation, checker raise, bad fixture,
      live without seats, immutability refusal, …)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback


def _isolate_state_dir():
    """Point OMNI_LLM_STATE_DIR at a fresh tempdir for this process."""
    state = tempfile.mkdtemp(prefix="omni-bench-state-")
    os.environ["OMNI_LLM_STATE_DIR"] = state
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="omni-llm-mcp hermetic/live baseline harness (P1.2b)",
    )
    parser.add_argument(
        "--hermetic",
        action="store_true",
        help="Run hermetic baseline measurement (fake backends, no network).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live lane: observational smoke against real backends (needs --live-seats).",
    )
    parser.add_argument(
        "--live-seats",
        metavar="MODELS",
        default=None,
        help="Comma-separated model seats for --live (required with --live).",
    )
    parser.add_argument(
        "--update-baseline",
        metavar="REASON",
        default=None,
        help="Write/refresh bench/baseline.json with REASON (required to update).",
    )
    args = parser.parse_args(argv)

    if args.live and args.hermetic:
        print(
            "error: --live and --hermetic are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    if args.live:
        return _run_live(args)

    if not args.hermetic:
        parser.error("specify --hermetic or --live")

    # Import after env isolation setup so health/dispatch see the temp state root.
    state_dir = _isolate_state_dir()
    try:
        return _run_hermetic(args)
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


def _run_live(
    args,
    *,
    fake_backend=None,
    live_report_path=None,
    baseline_path=None,
) -> int:
    """Live observational lane.

    ``fake_backend``, ``live_report_path``, and ``baseline_path`` are injectable
    for hermetic tests. Production CLI leaves them None (real dispatch, default
    bench/live-report.json, default bench/baseline.json identity guard).
    Live flow NEVER writes baseline.json.
    """
    seats_raw = args.live_seats
    if seats_raw is None or not str(seats_raw).strip():
        # Typed refusal — must not be silently conflated with hermetic results.
        print(
            "error: --live requires --live-seats <model[,model...]>",
            file=sys.stderr,
        )
        return 2

    seats = [s.strip() for s in str(seats_raw).split(",") if s.strip()]
    if not seats:
        print(
            "error: --live requires --live-seats <model[,model...]>",
            file=sys.stderr,
        )
        return 2

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    state_dir = _isolate_state_dir()
    try:
        from bench.measure import (  # noqa: WPS433
            BASELINE_PATH,
            LIVE_REPORT_PATH,
            measure_live,
            write_live_report,
        )

        report_path = (
            live_report_path if live_report_path is not None else LIVE_REPORT_PATH
        )
        base_path = baseline_path if baseline_path is not None else BASELINE_PATH

        try:
            report = measure_live(seats, fake_backend=fake_backend)
            write_live_report(
                report, path=report_path, baseline_path=base_path
            )
        except Exception as e:
            print("harness error: %s" % e, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2

        n_ok = sum(1 for r in report.get("results") or [] if r.get("ok"))
        n_total = len(report.get("results") or [])
        print(
            "live report written: seats=%s results=%d ok=%d path=%s"
            % (",".join(seats), n_total, n_ok, report_path)
        )
        # Observational only — never exit 1 on path failure.
        return 0
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


def _run_hermetic(args) -> int:
    # Ensure repo root is importable when invoked as `python3 benchmark.py`.
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    from bench.measure import (  # noqa: WPS433
        BASELINE_PATH,
        BaselineImmutabilityError,
        compare_baseline,
        format_pass_lines,
        load_baseline,
        measure_all,
        write_baseline,
    )

    # metric regression = 1; harness/internal error = 2
    try:
        measured = measure_all()
    except Exception as e:
        print("harness error: %s" % e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2

    if args.update_baseline is not None:
        reason = args.update_baseline
        if not str(reason).strip():
            print("error: --update-baseline requires a non-empty reason", file=sys.stderr)
            return 2
        try:
            write_baseline(measured, reason=reason)
        except BaselineImmutabilityError as e:
            print("error: %s" % e, file=sys.stderr)
            return 2
        except Exception as e:
            print("harness error: %s" % e, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2
        print(
            "baseline updated (%s): tools_compact_bytes=%s paths=%s"
            % (
                reason,
                measured["tools_compact_bytes"],
                ",".join(sorted(measured["paths"])),
            )
        )
        print("wrote %s" % BASELINE_PATH)
        return 0

    if not os.path.isfile(BASELINE_PATH):
        print(
            "error: no baseline at %s; run with --update-baseline \"reason\" first"
            % BASELINE_PATH,
            file=sys.stderr,
        )
        return 2

    try:
        baseline = load_baseline()
        regressions = compare_baseline(measured, baseline)
    except Exception as e:
        print("harness error: %s" % e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2

    if regressions:
        for reg in regressions:
            print(
                "REGRESSION %s: baseline=%s measured=%s"
                % (reg["name"], reg["baseline"], reg["measured"])
            )
        print("regression: %d metric(s) diverged" % len(regressions))
        return 1

    for line in format_pass_lines(measured, baseline):
        print(line)
    print("no regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
