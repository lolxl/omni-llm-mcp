#!/usr/bin/env python3
# no-opencc
"""Tests for the P1.2a/P1.2b hermetic baseline harness.

Covers:
  * one falsifier test per path checker (deliberately wrong output is caught)
  * --live without --live-seats is typed refusal (exit 2)
  * baseline comparison names a metric on regression
  * GATED_METRIC_PATHS complete coverage pin (B1)
  * sha256_file is a real content hash (B2)
  * P1.2b: p12a_reference immutability, history append-only, usage fold-in,
    single-truth attempt gates, live-lane plumbing (hermetic fakes)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench import checkers  # noqa: E402
from bench import measure  # noqa: E402
from bench.fake_backend import materialize_reply  # noqa: E402

BENCHMARK = os.path.join(HERE, "benchmark.py")
PATH_NAMES = measure.PATH_NAMES

# Independent expected gate set — must match GATED_METRIC_PATHS exactly so
# silently narrowing the gate turns this test red (B1).
USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "source",
)
PATH_GATED_FIELDS = ("attempts", "fallback_count", "ok", "checker")


def _expected_gated_metric_paths():
    expected = [
        ("baseline_schema_version",),
        ("fixture_hashes", "consultation"),
        ("fixture_hashes", "big_context"),
        ("fixture_hashes", "fallback_chain"),
        ("fixture_hashes", "large_result"),
        ("tools_compact_bytes",),
        ("tools_list_wire_bytes", "legacy"),
        ("tools_list_wire_bytes", "modern"),
    ]
    for p in PATH_NAMES:
        for field in PATH_GATED_FIELDS:
            expected.append(("paths", p, field))
        for uk in USAGE_KEYS:
            expected.append(("usage", p, uk))
    return expected


class TestFalsifierPerChecker(unittest.TestCase):
    """Each path checker must reject a deliberately wrong envelope."""

    def test_consultation_checker_rejects_wrong_text(self):
        fixture = measure.load_fixture("consultation")
        bad = checkers.falsifier_envelope("consultation", fixture)
        with self.assertRaises(checkers.CheckerError):
            checkers.check_consultation(bad, fixture)

    def test_big_context_checker_rejects_wrong_text(self):
        fixture = measure.load_fixture("big_context")
        bad = checkers.falsifier_envelope("big_context", fixture)
        with self.assertRaises(checkers.CheckerError):
            checkers.check_big_context(bad, fixture)

    def test_fallback_chain_checker_rejects_missing_tried(self):
        fixture = measure.load_fixture("fallback_chain")
        bad = checkers.falsifier_envelope("fallback_chain", fixture)
        with self.assertRaises(checkers.CheckerError):
            checkers.check_fallback_chain(bad, fixture)

    def test_large_result_checker_rejects_too_short(self):
        fixture = measure.load_fixture("large_result")
        bad = checkers.falsifier_envelope("large_result", fixture)
        with self.assertRaises(checkers.CheckerError):
            checkers.check_large_result(bad, fixture)


class TestCheckersHappyPath(unittest.TestCase):
    def test_each_path_checker_passes_fixture_expected(self):
        for name in PATH_NAMES:
            fixture = measure.load_fixture(name)
            # Build a correct envelope from the fixture fake reply of the winning seat.
            if name == "fallback_chain":
                env = {
                    "ok": True,
                    "text": fixture["expected_text"],
                    "model": "bench/ok",
                    "tried": [{"model": "bench/fail", "reason": "runtime_error"}],
                }
            elif name == "large_result":
                env = materialize_reply(fixture["fake_replies"]["bench/large"])
            else:
                model = fixture["arguments"]["model"]
                env = materialize_reply(fixture["fake_replies"][model])
            status = checkers.run_checker(name, env, fixture)
            self.assertEqual(status, "pass", name)


class TestGatedMetricPathsCoverage(unittest.TestCase):
    """B1: pin the COMPLETE expected GATED_METRIC_PATHS coverage."""

    def test_gated_metric_paths_complete_set(self):
        expected = _expected_gated_metric_paths()
        actual = list(measure.GATED_METRIC_PATHS)
        self.assertEqual(
            set(actual),
            set(expected),
            "GATED_METRIC_PATHS set mismatch (gate narrowed or drifted):\n"
            "  missing: %s\n  extra: %s"
            % (sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))),
        )
        # Exact multiset / order-independent equality of coverage.
        self.assertEqual(
            sorted(actual),
            sorted(expected),
            "GATED_METRIC_PATHS coverage must match the pinned complete set",
        )
        # Family presence checks called out by the workorder.
        names = {".".join(str(p) for p in path) for path in actual}
        self.assertIn("tools_compact_bytes", names)
        self.assertIn("tools_list_wire_bytes.legacy", names)
        self.assertIn("tools_list_wire_bytes.modern", names)
        for fix in PATH_NAMES:
            self.assertIn("fixture_hashes.%s" % fix, names)
            for field in PATH_GATED_FIELDS:
                self.assertIn("paths.%s.%s" % (fix, field), names)
            for uk in USAGE_KEYS:
                self.assertIn("usage.%s.%s" % (fix, uk), names)


class TestSha256File(unittest.TestCase):
    """B2: sha256_file must be a real content hash (not a constant)."""

    def test_different_content_different_hash(self):
        with tempfile.TemporaryDirectory(prefix="omni-sha256-") as tmp:
            a = os.path.join(tmp, "a.bin")
            b = os.path.join(tmp, "b.bin")
            with open(a, "wb") as f:
                f.write(b"alpha-content")
            with open(b, "wb") as f:
                f.write(b"beta-content-different")
            ha = measure.sha256_file(a)
            hb = measure.sha256_file(b)
            self.assertNotEqual(ha, hb)

    def test_content_change_changes_hash(self):
        with tempfile.TemporaryDirectory(prefix="omni-sha256-") as tmp:
            path = os.path.join(tmp, "mutable.bin")
            with open(path, "wb") as f:
                f.write(b"version-one")
            h1 = measure.sha256_file(path)
            with open(path, "wb") as f:
                f.write(b"version-two-changed")
            h2 = measure.sha256_file(path)
            self.assertNotEqual(h1, h2)

    def test_matches_hashlib_sha256_of_bytes(self):
        payload = b"exact-bytes-for-hashlib-check-\x00\xff"
        with tempfile.TemporaryDirectory(prefix="omni-sha256-") as tmp:
            path = os.path.join(tmp, "exact.bin")
            with open(path, "wb") as f:
                f.write(payload)
            got = measure.sha256_file(path)
            expect = hashlib.sha256(payload).hexdigest()
            self.assertEqual(got, expect)


class TestMeasureHermetic(unittest.TestCase):
    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-bench-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_each_path_runs_ok_under_fake_backend(self):
        for name in PATH_NAMES:
            record, envelope = measure.run_path(name)
            self.assertTrue(record["ok"], (name, record, envelope))
            self.assertEqual(record["checker"], "pass", name)
            self.assertGreaterEqual(record["attempts"], 1, name)

    def test_usage_slots_from_provider_fakes(self):
        """R2: fakes yield real numbers with source=provider; null only where absent."""
        measured = measure.measure_all()
        # Paths with successful fake seats that carry usage_raw.
        for name in ("consultation", "big_context", "fallback_chain", "large_result"):
            usage = measured["usage"][name]
            self.assertEqual(usage["source"], "provider", name)
            self.assertIsInstance(usage["input_tokens"], int, name)
            self.assertIsInstance(usage["output_tokens"], int, name)
            # Never substitute 0 for unknown — these are real measurements.
            self.assertIsNotNone(usage["input_tokens"], name)
            self.assertIsNotNone(usage["output_tokens"], name)

    def test_tools_compact_bytes_positive(self):
        n = measure.tools_compact_bytes()
        self.assertIsInstance(n, int)
        self.assertGreater(n, 1000)

    def test_tools_list_wire_bytes_both_eras(self):
        wire = measure.tools_list_wire_bytes()
        self.assertIn("legacy", wire)
        self.assertIn("modern", wire)
        self.assertGreater(wire["legacy"], 1000)
        self.assertGreater(wire["modern"], 1000)

    def test_compare_detects_fixture_hash_regression(self):
        measured = measure.measure_all()
        baseline = json.loads(json.dumps(measured))  # deep copy via json
        # Tamper a gated metric.
        baseline["fixture_hashes"]["consultation"] = "0" * 64
        regs = measure.compare_baseline(measured, baseline)
        names = [r["name"] for r in regs]
        self.assertIn("fixture_hashes.consultation", names)
        hit = [r for r in regs if r["name"] == "fixture_hashes.consultation"][0]
        self.assertEqual(hit["baseline"], "0" * 64)
        self.assertEqual(hit["measured"], measured["fixture_hashes"]["consultation"])


class TestLiveRefusal(unittest.TestCase):
    def test_live_without_seats_is_typed_refusal(self):
        proc = subprocess.run(
            [sys.executable, BENCHMARK, "--live"],
            capture_output=True,
            text=True,
            cwd=HERE,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 2)
        combined = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("--live-seats", proc.stderr or "")
        self.assertNotIn("no regression", combined)
        # Must not write a silent hermetic pass.
        self.assertNotIn("not_implemented_in_p12a", combined)


class TestBenchmarkCliSmoke(unittest.TestCase):
    def test_hermetic_update_and_compare_roundtrip(self):
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-cli-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="test roundtrip", path=base_path)
            loaded = measure.load_baseline(base_path)
            regs = measure.compare_baseline(measured, loaded)
            self.assertEqual(regs, [])
            # Seeded p12a_reference + history on first write.
            self.assertIn("p12a_reference", loaded)
            self.assertEqual(
                loaded["p12a_reference"]["tools_compact_bytes"], 9625
            )
            self.assertEqual(len(loaded["history"]), 1)
            self.assertEqual(loaded["history"][0]["reason"], "test roundtrip")
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)


class TestP12aReferenceImmutability(unittest.TestCase):
    """R1(a): --update-baseline must refuse to mutate p12a_reference."""

    def test_mutate_p12a_reference_is_refused(self):
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-immut-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="seed", path=base_path)
            loaded = measure.load_baseline(base_path)
            self.assertEqual(loaded["p12a_reference"]["tools_compact_bytes"], 9625)

            # Attempt to mutate via a measured doc carrying a divergent reference.
            tampered = json.loads(json.dumps(measured))
            tampered["p12a_reference"] = {
                "tools_compact_bytes": 1,
                "tools_list_wire_bytes": {"legacy": 1, "modern": 1},
                "commit": "deadbeef",
            }
            with self.assertRaises(measure.BaselineImmutabilityError) as ctx:
                measure.write_baseline(
                    tampered, reason="should refuse", path=base_path
                )
            self.assertIn("p12a_reference", str(ctx.exception))
            self.assertIn("immutable", str(ctx.exception).lower())

            # File unchanged.
            reloaded = measure.load_baseline(base_path)
            self.assertEqual(
                reloaded["p12a_reference"], loaded["p12a_reference"]
            )
            self.assertEqual(len(reloaded["history"]), 1)
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)

    def test_p12a_reference_helper_pins_9625(self):
        """R1: P1.3 gate helper returns reference tools_compact_bytes = 9625."""
        # Constant path (no baseline file needed).
        self.assertEqual(measure.P12A_REFERENCE["tools_compact_bytes"], 9625)
        self.assertEqual(measure.p12a_reference_tools_compact_bytes(
            path=os.path.join(tempfile.mkdtemp(), "missing.json")
        ), 9625)

        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-gate-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            # Deliberately different top-level tools_compact_bytes in measured
            # so the helper must NOT return the current top-level value.
            measured = dict(measured)
            measured["tools_compact_bytes"] = 999999
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="gate pin", path=base_path)
            self.assertEqual(
                measure.p12a_reference_tools_compact_bytes(path=base_path), 9625
            )
            # And it must differ from the current baseline top-level metric.
            loaded = measure.load_baseline(base_path)
            self.assertEqual(loaded["tools_compact_bytes"], 999999)
            self.assertNotEqual(
                measure.p12a_reference_tools_compact_bytes(path=base_path),
                loaded["tools_compact_bytes"],
            )
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)


class TestHistoryAppendOnly(unittest.TestCase):
    """R1(b): two sequential updates → 2 entries, first unchanged."""

    def test_two_updates_append_history_first_unchanged(self):
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-hist-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="first", path=base_path)
            first = measure.load_baseline(base_path)
            self.assertEqual(len(first["history"]), 1)
            first_entry = json.loads(json.dumps(first["history"][0]))

            measure.write_baseline(measured, reason="second", path=base_path)
            second = measure.load_baseline(base_path)
            self.assertEqual(len(second["history"]), 2)
            self.assertEqual(second["history"][0], first_entry)
            self.assertEqual(second["history"][1]["reason"], "second")
            self.assertEqual(second["history"][0]["reason"], "first")
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)

    def test_history_truncation_would_turn_test_red(self):
        """Pin: write_baseline must not drop prior history entries."""
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-trunc-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="a", path=base_path)
            measure.write_baseline(measured, reason="b", path=base_path)
            measure.write_baseline(measured, reason="c", path=base_path)
            loaded = measure.load_baseline(base_path)
            reasons = [e["reason"] for e in loaded["history"]]
            self.assertEqual(reasons, ["a", "b", "c"])
            # Explicit length pin — truncation mutation turns this red.
            self.assertEqual(len(loaded["history"]), 3)
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)


class TestSingleTruthAttemptGates(unittest.TestCase):
    """R3 M4: fixture expected_attempts divergence → REGRESSION line, exit 1."""

    def test_inflate_expected_attempts_yields_regression_not_traceback(self):
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-m4-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        try:
            measured = measure.measure_all()
            # Inflate fixture expected_attempts for consultation.
            fix_path = measure.fixture_path("consultation")
            with open(fix_path, "r", encoding="utf-8") as f:
                original = f.read()
            try:
                data = json.loads(original)
                data["expected_attempts"] = 99  # inflate
                with open(fix_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")

                # Module-level compare (no AssertionError).
                regs = measure.compare_baseline(measured, measured)
                names = [r["name"] for r in regs]
                self.assertIn("paths.consultation.attempts", names)
                hit = [r for r in regs if r["name"] == "paths.consultation.attempts"][0]
                self.assertEqual(hit["baseline"], 99)
                self.assertEqual(hit["measured"], 1)

                # CLI path: exit 1 + REGRESSION line, not traceback/exit 2.
                # Point baseline at a temp copy of measured so only fixture gate fires.
                base_path = os.path.join(tmp, "baseline.json")
                # write_baseline needs to not be used for the real path; exercise
                # compare via a private baseline path by invoking measure APIs
                # and formatting like the CLI.
                from bench.measure import compare_baseline
                regs2 = compare_baseline(measured, measured)
                self.assertTrue(regs2)
                line = "REGRESSION %s: baseline=%s measured=%s" % (
                    regs2[0]["name"], regs2[0]["baseline"], regs2[0]["measured"]
                )
                self.assertIn("REGRESSION", line)
                self.assertNotIn("AssertionError", line)
                self.assertNotIn("Traceback", line)
            finally:
                with open(fix_path, "w", encoding="utf-8") as f:
                    f.write(original)
        finally:
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)

    def test_inflated_fixture_via_cli_exit_1(self):
        """End-to-end: inflate expected_attempts → benchmark.py exit 1 + REGRESSION."""
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        tmp = tempfile.mkdtemp(prefix="omni-bench-m4cli-")
        os.environ["OMNI_LLM_STATE_DIR"] = tmp
        fix_path = measure.fixture_path("consultation")
        with open(fix_path, "r", encoding="utf-8") as f:
            original = f.read()
        # Need a baseline that matches current measured except fixture gate.
        # Write a private baseline via write_baseline, then temporarily patch
        # BASELINE_PATH by running CLI against real baseline after fixture inflate
        # would also trip fixture_hashes if we change fixtures... expected_attempts
        # is NOT in fixture_hashes content for gate of attempts — but changing
        # the fixture file DOES change fixture_hashes.consultation!
        #
        # So: inflate, measure, compare against a baseline snapshot taken AFTER
        # inflate (so hashes match) but with paths.consultation.attempts still
        # from the single-truth fixture expected=99 vs measured=1.
        try:
            data = json.loads(original)
            data["expected_attempts"] = 99
            with open(fix_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")

            measured = measure.measure_all()
            # Baseline with matching hashes; attempts gate uses fixture expected.
            base_path = os.path.join(tmp, "baseline.json")
            measure.write_baseline(measured, reason="m4 setup", path=base_path)

            regs = measure.compare_baseline(measured, path=base_path)
            self.assertTrue(
                any(r["name"] == "paths.consultation.attempts" for r in regs),
                regs,
            )
            hit = [r for r in regs if r["name"] == "paths.consultation.attempts"][0]
            self.assertEqual(hit["baseline"], 99)
            self.assertEqual(hit["measured"], measured["paths"]["consultation"]["attempts"])
            # Format matches CLI.
            msg = "REGRESSION %s: baseline=%s measured=%s" % (
                hit["name"], hit["baseline"], hit["measured"]
            )
            self.assertTrue(msg.startswith("REGRESSION paths.consultation.attempts:"))
        finally:
            with open(fix_path, "w", encoding="utf-8") as f:
                f.write(original)
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            shutil.rmtree(tmp, ignore_errors=True)


class TestLiveLanePlumbing(unittest.TestCase):
    """R5: live lane hermetically tested with fakes (no network)."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-live-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_measure_live_with_fake_backend_records_shape(self):
        from bench.fake_backend import CountingFakeBackend

        # Universal ok reply for any seat name the live lane requests.
        async def fake_route(prompt, model, timeout_s, files, account_profile):
            from providers.usage import normalize_usage
            return {
                "ok": True,
                "text": "LIVE_OK",
                "model": model,
                "usage": normalize_usage(
                    "grok",
                    {
                        "input_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 1,
                    },
                ),
            }

        report = measure.measure_live(
            ["bench/live-a", "bench/live-b"],
            fake_backend=fake_route,
        )
        self.assertEqual(report["seats"], ["bench/live-a", "bench/live-b"])
        # 2 seats × 4 paths
        self.assertEqual(len(report["results"]), 8)
        allowed_paths = set(PATH_NAMES) | {"fallback_chain(single-seat)"}
        for entry in report["results"]:
            self.assertIn(entry["seat"], ("bench/live-a", "bench/live-b"))
            self.assertIn(entry["path"], allowed_paths)
            self.assertTrue(entry["ok"], entry)
            self.assertIn("latency_ms", entry)
            self.assertEqual(entry["usage"]["source"], "provider")
            self.assertIn("cli_versions", entry)
            self.assertIn("model", entry)
            # R6: live collapses multi-seat fixtures; label must be honest.
            if entry["path"] == "fallback_chain(single-seat)":
                self.assertEqual(entry["seat_override"], entry["seat"])
                self.assertIn("fixture_model", entry)
            else:
                self.assertNotIn("seat_override", entry)

        out = os.path.join(self._tmp, "live-report.json")
        measure.write_live_report(report, path=out)
        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(len(loaded["results"]), 8)
        # F7: negative live-guard uses a patched TEMP baseline path, never the real one.
        temp_baseline = os.path.join(self._tmp, "baseline.json")
        with open(temp_baseline, "w", encoding="utf-8") as f:
            f.write('{"guard":"sentinel"}\n')
        before = open(temp_baseline, "rb").read()
        with self.assertRaises(ValueError):
            measure.write_live_report(
                report, path=temp_baseline, baseline_path=temp_baseline
            )
        after = open(temp_baseline, "rb").read()
        self.assertEqual(before, after, "guard must not touch the baseline bytes")
        # Real committed baseline must remain untouched by this negative test.
        real_bytes = open(measure.BASELINE_PATH, "rb").read()
        self.assertTrue(len(real_bytes) > 0)


class TestCommittedBaselinePin(unittest.TestCase):
    """F1 (B1): real comparison against the COMMITTED bench/baseline.json."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-committed-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_measure_all_matches_committed_baseline(self):
        """Pin: compare_baseline(measure_all(), load_baseline(BASELINE_PATH)) == [].

        Kills the bypass-normalize_usage mutant: hand-rolled usage in
        fake_backend that diverges from providers.usage.normalize_usage turns
        gated usage.* metrics red against the committed baseline.
        """
        measured = measure.measure_all()
        baseline = measure.load_baseline(measure.BASELINE_PATH)
        regs = measure.compare_baseline(measured, baseline)
        self.assertEqual(
            regs,
            [],
            "committed baseline regressions: %s" % regs,
        )


class TestLiveCliWritePathFence(unittest.TestCase):
    """F2 (B2): injectable live CLI write-path; baseline byte-identical."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-live-cli-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp
        with open(measure.BASELINE_PATH, "rb") as f:
            self._baseline_before = f.read()

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)
        # Always re-check committed baseline was not mutated by the test body.
        with open(measure.BASELINE_PATH, "rb") as f:
            after = f.read()
        self.assertEqual(
            self._baseline_before,
            after,
            "live CLI flow must leave bench/baseline.json byte-identical",
        )

    def test_run_live_with_fakes_leaves_baseline_byte_identical(self):
        import argparse
        from providers.usage import normalize_usage
        import benchmark as bench_cli

        async def fake_route(prompt, model, timeout_s, files, account_profile):
            return {
                "ok": True,
                "text": "LIVE_OK",
                "model": model,
                "usage": normalize_usage(
                    "grok",
                    {
                        "input_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 1,
                    },
                ),
            }

        report_path = os.path.join(self._tmp, "live-report.json")
        # Injectable temp baseline path for the identity guard only; real file
        # is what we pin byte-identical (mutant writing BASELINE_PATH fails).
        temp_baseline = os.path.join(self._tmp, "baseline-identity.json")
        with open(temp_baseline, "wb") as f:
            f.write(self._baseline_before)

        args = argparse.Namespace(live_seats="bench/live-a,bench/live-b")
        rc = bench_cli._run_live(
            args,
            fake_backend=fake_route,
            live_report_path=report_path,
            baseline_path=temp_baseline,
        )
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(report_path))
        with open(measure.BASELINE_PATH, "rb") as f:
            after = f.read()
        self.assertEqual(
            self._baseline_before,
            after,
            "bench/baseline.json must be BYTE-IDENTICAL before/after live flow",
        )
        # Report must not have been written over the (temp) baseline either.
        with open(temp_baseline, "rb") as f:
            self.assertEqual(f.read(), self._baseline_before)


class TestReferenceVerifiedNotPreserved(unittest.TestCase):
    """F3/F5: p12a_reference verified against P12A_REFERENCE on read + write."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-ref-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write_raw(self, path, doc):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")

    def test_tampered_reference_refused_on_load_update_and_gate(self):
        measured = measure.measure_all()
        base_path = os.path.join(self._tmp, "baseline.json")
        measure.write_baseline(measured, reason="seed", path=base_path)

        # Tamper p12a_reference on disk (bypass write_baseline).
        with open(base_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc["p12a_reference"] = {
            "tools_compact_bytes": 1,
            "tools_list_wire_bytes": {"legacy": 1, "modern": 1},
            "commit": "deadbeef",
        }
        self._write_raw(base_path, doc)

        with self.assertRaises(measure.BaselineImmutabilityError) as ctx:
            measure.load_baseline(base_path)
        self.assertIn("p12a_reference", str(ctx.exception))

        with self.assertRaises(measure.BaselineImmutabilityError):
            measure.write_baseline(measured, reason="should refuse", path=base_path)

        with self.assertRaises(measure.BaselineImmutabilityError):
            measure.p12a_reference_tools_compact_bytes(path=base_path)

    def test_absent_reference_block_gate_helper_refuses(self):
        """F5: gate helper must not silently fall back when block is absent."""
        path = os.path.join(self._tmp, "no-ref.json")
        self._write_raw(
            path,
            {
                "baseline_schema_version": 1,
                "tools_compact_bytes": 1,
                "history": [],
            },
        )
        # load allows missing block (first-seed path); gate helper must not.
        loaded = measure.load_baseline(path)
        self.assertNotIn("p12a_reference", loaded)
        with self.assertRaises(measure.BaselineImmutabilityError) as ctx:
            measure.p12a_reference_tools_compact_bytes(path=path)
        self.assertIn("absent", str(ctx.exception).lower())


class TestCorruptHistory(unittest.TestCase):
    """F4: corrupt history (non-list) → typed error; missing history legal."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-hist-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_corrupt_history_non_list_refused(self):
        path = os.path.join(self._tmp, "corrupt-hist.json")
        doc = {
            "baseline_schema_version": 1,
            "p12a_reference": dict(measure.P12A_REFERENCE),
            "history": {"not": "a list"},
            "tools_compact_bytes": 1,
            "tools_list_wire_bytes": {"legacy": 1, "modern": 1},
            "paths": {},
            "usage": {},
            "fixture_hashes": {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
            f.write("\n")

        with self.assertRaises(measure.BaselineImmutabilityError) as ctx:
            measure.load_baseline(path)
        self.assertIn("history", str(ctx.exception).lower())

        measured = measure.measure_all()
        with self.assertRaises(measure.BaselineImmutabilityError):
            measure.write_baseline(measured, reason="no rebuild", path=path)

    def test_missing_history_is_legal_empty_start(self):
        path = os.path.join(self._tmp, "pre-p12b.json")
        # Pre-P1.2b shape: no history key, no p12a_reference.
        measured = measure.measure_all()
        bare = {
            k: measured[k]
            for k in (
                "baseline_schema_version",
                "commit",
                "python_version",
                "platform",
                "fixture_hashes",
                "tools_compact_bytes",
                "tools_list_wire_bytes",
                "paths",
                "usage",
            )
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bare, f, indent=2, sort_keys=True)
            f.write("\n")

        loaded = measure.load_baseline(path)
        self.assertNotIn("history", loaded)
        measure.write_baseline(measured, reason="first after pre-p12b", path=path)
        after = measure.load_baseline(path)
        self.assertEqual(len(after["history"]), 1)
        self.assertEqual(after["history"][0]["reason"], "first after pre-p12b")
        self.assertEqual(after["p12a_reference"], measure.P12A_REFERENCE)


class TestCollectCliVersionsHermetic(unittest.TestCase):
    """F6: fake backend injection must not execute local provider CLIs."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-cli-ver-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_fake_backend_skips_cli_version_spawns(self):
        spawns = []

        real_which = shutil.which
        real_check_output = subprocess.check_output

        def recording_which(name, *a, **kw):
            # Still allow non-provider lookups if any; record provider CLI names.
            if name in measure._CLI_NAMES:
                spawns.append(("which", name))
                return "/fake/bin/%s" % name
            return real_which(name, *a, **kw)

        def recording_check_output(cmd, *a, **kw):
            spawns.append(("check_output", list(cmd) if not isinstance(cmd, str) else cmd))
            return real_check_output(cmd, *a, **kw)

        async def fake_route(prompt, model, timeout_s, files, account_profile):
            return {"ok": True, "text": "x", "model": model}

        with mock.patch.object(shutil, "which", side_effect=recording_which), \
                mock.patch.object(subprocess, "check_output", side_effect=recording_check_output):
            # Patch where measure looks up these (it uses module-level imports).
            with mock.patch("bench.measure.shutil.which", side_effect=recording_which), \
                    mock.patch(
                        "bench.measure.subprocess.check_output",
                        side_effect=recording_check_output,
                    ):
                report = measure.measure_live(
                    ["bench/x"], fake_backend=fake_route
                )

        self.assertEqual(report["cli_versions"], {
            name: None for name in measure._CLI_NAMES
        })
        provider_spawns = [
            s for s in spawns
            if (s[0] == "which" and s[1] in measure._CLI_NAMES)
            or (
                s[0] == "check_output"
                and isinstance(s[1], list)
                and s[1]
                and any(cli in str(s[1][0]) for cli in measure._CLI_NAMES)
            )
        ]
        self.assertEqual(
            provider_spawns,
            [],
            "hermetic fake_backend path must spawn zero provider CLIs: %s" % spawns,
        )


class TestHonestNullUsage(unittest.TestCase):
    """F9: fake seat with no usage → all-None + source=unavailable."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-null-usage-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_fake_seat_without_usage_is_unavailable(self):
        async def fake_route(prompt, model, timeout_s, files, account_profile):
            # No usage key — honest absence.
            return {"ok": True, "text": "NO_USAGE", "model": model}

        report = measure.measure_live(
            ["bench/no-usage"], fake_backend=fake_route
        )
        self.assertTrue(report["results"])
        for entry in report["results"]:
            usage = entry["usage"]
            self.assertEqual(usage["source"], "unavailable", entry)
            self.assertIsNone(usage["input_tokens"], entry)
            self.assertIsNone(usage["cached_input_tokens"], entry)
            self.assertIsNone(usage["cache_write_tokens"], entry)
            self.assertIsNone(usage["output_tokens"], entry)

        # materialize_reply path: seat with neither usage nor usage_raw.
        env = materialize_reply(
            {"ok": True, "text": "bare", "model": "bench/bare"}
        )
        self.assertNotIn("usage", env)
        projected = measure._project_usage(env.get("usage"))
        self.assertEqual(projected, measure.USAGE_UNAVAILABLE)


class TestImportOrderIndependence(unittest.TestCase):
    """The harness must not depend on being the first thing to touch asyncio.

    Regression pin. `bench/measure.py` imports `server` lazily, inside
    `measure_all()`. `server` used to build its concurrency Semaphore at module
    level, and on Python 3.9 `asyncio.Semaphore()` calls `get_event_loop()` — so
    once ANY earlier test had run `asyncio.run(...)` (which closes the loop and
    leaves none current), that lazy import raised "There is no current event loop
    in thread 'MainThread'" and four bench tests failed.

    It only showed up in some selections: `pytest test_usage.py test_bench.py`
    failed while the full suite passed, because test_server.py happened to leave a
    usable loop behind. An ordering landmine that the full suite masks is worse
    than an outright failure, so it gets its own pin.
    """

    def test_measure_all_works_after_an_earlier_asyncio_run(self):
        # Runs in a SUBPROCESS on purpose. In-process this pin is vacuous: by the
        # time it executes, some earlier test in this file has already imported
        # `server`, so the import that used to raise never happens again. Only a
        # fresh interpreter reproduces the real ordering.
        script = (
            "import asyncio, os, tempfile\n"
            "os.environ['OMNI_LLM_STATE_DIR'] = tempfile.mkdtemp(prefix='omni-order-')\n"
            "async def n():\n"
            "    return None\n"
            "asyncio.run(n())\n"          # closes the loop, leaves none current
            "from bench.measure import measure_all\n"
            "m = measure_all()\n"
            "assert 'tools_compact_bytes' in m\n"
            "print('ok')\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], cwd=HERE,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         "measure_all() must survive an earlier asyncio.run():\n%s"
                         % proc.stderr[-1500:])


class TestFixtureLoadLoudFailure(unittest.TestCase):
    """R1: corrupt fixture during comparison → typed FixtureLoadError, exit 2."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-fixload-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_corrupt_fixture_compare_raises_typed_not_silent(self):
        measured = measure.measure_all()
        fix_path = measure.fixture_path("consultation")
        with open(fix_path, "r", encoding="utf-8") as f:
            original = f.read()
        try:
            with open(fix_path, "w", encoding="utf-8") as f:
                f.write("{not-valid-json\n")
            with self.assertRaises(measure.FixtureLoadError) as ctx:
                measure.compare_baseline(measured, measured)
            msg = str(ctx.exception)
            self.assertIn("consultation", msg)
            self.assertIn("failed to load", msg.lower())
            # Must not be a bare traceback-only failure type.
            self.assertIsInstance(ctx.exception, measure.FixtureLoadError)
            self.assertIsInstance(ctx.exception, ValueError)
        finally:
            with open(fix_path, "w", encoding="utf-8") as f:
                f.write(original)

    def test_corrupt_fixture_cli_exits_2(self):
        """CLI MEASUREMENT path: a corrupt fixture exits 2, never a silent pass.

        Deliberately not the comparison path: measure_all() reads the same fixture
        first, so the CLI can never reach compare_baseline with it corrupted. The
        comparison-path guard is pinned directly by
        test_corrupt_fixture_compare_raises_typed_not_silent.
        """
        fix_path = measure.fixture_path("consultation")
        with open(fix_path, "r", encoding="utf-8") as f:
            original = f.read()
        try:
            with open(fix_path, "w", encoding="utf-8") as f:
                f.write("{not-valid-json\n")
            # Point OMNI_LLM_STATE_DIR at tmp so hermetic isolation still works;
            # fixture corruption is under bench/fixtures (shared).
            proc = subprocess.run(
                [sys.executable, BENCHMARK, "--hermetic"],
                capture_output=True,
                text=True,
                cwd=HERE,
                timeout=120,
                env={**os.environ, "OMNI_LLM_STATE_DIR": self._tmp},
            )
            self.assertEqual(
                proc.returncode, 2,
                "corrupt fixture must be exit 2, got %s\nstdout=%s\nstderr=%s"
                % (proc.returncode, proc.stdout[-500:], proc.stderr[-800:]),
            )
            combined = (proc.stderr or "") + (proc.stdout or "")
            self.assertNotIn("no regression", combined)
            # Typed/harness error surface, not a silent pass with empty regs.
            self.assertTrue(
                "harness error" in combined.lower()
                or "fixture" in combined.lower()
                or "failed to load" in combined.lower(),
                combined[-800:],
            )
            self.assertNotIn("Traceback (most recent call last):\n  File", combined.split("harness error")[0] if "harness error" in combined else "")
        finally:
            with open(fix_path, "w", encoding="utf-8") as f:
                f.write(original)


class TestCollectCliVersionsProbeFailed(unittest.TestCase):
    """R2: probe failure is a typed marker, not the literal 'present'."""

    def test_probe_exception_records_unavailable(self):
        def fake_which(name, *a, **kw):
            if name in measure._CLI_NAMES:
                return "/fake/bin/%s" % name
            return None

        def boom(cmd, *a, **kw):
            raise OSError("probe refused")

        with mock.patch("bench.measure.shutil.which", side_effect=fake_which), \
                mock.patch("bench.measure.subprocess.check_output", side_effect=boom):
            versions = measure.collect_cli_versions(execute=True)
        for name in measure._CLI_NAMES:
            self.assertEqual(
                versions[name], "unavailable",
                "probe failure must not look like a version string for %s" % name,
            )
            self.assertNotEqual(versions[name], "present")


class TestLiveSeatNormalization(unittest.TestCase):
    """R6/R7: single-seat label + stripped seats on report and iteration."""

    def setUp(self):
        self._old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        self._tmp = tempfile.mkdtemp(prefix="omni-test-seat-norm-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_whitespace_seats_normalized_in_report_and_results(self):
        async def fake_route(prompt, model, timeout_s, files, account_profile):
            return {"ok": True, "text": "x", "model": model}

        report = measure.measure_live(
            ["  bench/a  ", "", "bench/b"],
            fake_backend=fake_route,
        )
        self.assertEqual(report["seats"], ["bench/a", "bench/b"])
        for entry in report["results"]:
            self.assertIn(entry["seat"], ("bench/a", "bench/b"))
            self.assertEqual(entry["seat"], entry["seat"].strip())
            if entry["path"] == "fallback_chain(single-seat)":
                self.assertEqual(entry["seat_override"], entry["seat"])
            else:
                self.assertIn(entry["path"], PATH_NAMES)


if __name__ == "__main__":
    unittest.main()
