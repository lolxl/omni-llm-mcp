#!/usr/bin/env python3
"""Hermetic contract tests for the Codex read-only sandbox canary gate."""

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from providers import (agent_lane, config, jobs, kernel_sandbox,
                       sandbox_canary)


class TestCanaryFailureDetail(unittest.TestCase):
    """The write canary's codex failures never carry stderr (prompt echo)."""

    def probe(self, returncode, signal=None, stdout="", stderr=None, prompt="p"):
        stderr = "SYNTHETIC-PROMPT-SENTINEL sk-" + "k" * 40 if stderr is None else stderr

        calls = []

        async def fake_run_cli(argv, **kwargs):
            calls.append(argv)
            res = {"returncode": returncode, "stdout": stdout, "stderr": stderr,
                   "timed_out": False}
            if signal is not None:
                res["signal"] = signal
            return res
        with mock.patch.object(sandbox_canary.shutil, "which", return_value="/fake/bin/codex"), \
                mock.patch.object(sandbox_canary, "is_vendor_readonly_argv", return_value=True), \
                mock.patch.object(sandbox_canary, "run_cli", fake_run_cli):
            res = asyncio.run(sandbox_canary._run_vendor_readonly_probe(prompt, 5))
        self.assertEqual(len(calls), 1, res)   # the failure path itself ran
        return res

    def test_sighup_is_a_signal_exit(self):
        res = self.probe(-1, signal=1)
        self.assertEqual(res.get("reason"), "signal_exit", res)
        self.assertIn("SIGHUP", res.get("detail", ""))

    def test_prompt_echo_does_not_classify_the_failure(self):
        event = json.dumps({"type": "error", "message": "transport failed"})
        res = self.probe(1, stdout=event, stderr="write a login token file\n",
                         prompt="Canary: write a login token file")
        self.assertEqual(res.get("reason"), "runtime_error", res)

    def test_failures_name_no_stderr(self):
        for returncode, signal in ((1, None), (-9, 9), (0, None)):
            with self.subTest(returncode=returncode):
                res = self.probe(returncode, signal)
                self.assertFalse(res.get("ok"), res)
                self.assertNotIn("SYNTHETIC-PROMPT-SENTINEL", res.get("detail", ""))
                self.assertNotIn("sk-k", res.get("detail", ""))


class TestSandboxCanary(unittest.TestCase):
    def setUp(self):
        # Vendor admission remains active only on the legacy non-macOS path.
        legacy_platform = mock.patch("platform.system", return_value="Linux")
        legacy_platform.start()
        self.addCleanup(legacy_platform.stop)
        self.state_dir = tempfile.mkdtemp()
        self.old_state_dir = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = self.state_dir
        self.old_version_function = sandbox_canary.codex_cli_version
        sandbox_canary.codex_cli_version = lambda: "0.148.0"
        self.old_installed_function = sandbox_canary._codex_installed
        sandbox_canary._codex_installed = lambda: False
        # This suite exercises the VENDOR-canary decision path. Pin the
        # kernel fence absent so Darwin hosts do not change the injected
        # fakes; the gate no longer stands down on verify (issue #51).
        self.old_kernel_available = kernel_sandbox.available
        kernel_sandbox.available = lambda: False

        # Agent tests run with a depth marker. Clear the real recursion guard so
        # the integration seam reaches the sandbox gate.
        self.old_agent_depth = os.environ.get(agent_lane.ENV_DEPTH)
        os.environ.pop(agent_lane.ENV_DEPTH, None)

    def tearDown(self):
        sandbox_canary.codex_cli_version = self.old_version_function
        sandbox_canary._codex_installed = self.old_installed_function
        kernel_sandbox.available = self.old_kernel_available
        self._restore_env("OMNI_LLM_STATE_DIR", self.old_state_dir)
        self._restore_env(agent_lane.ENV_DEPTH, self.old_agent_depth)
        shutil.rmtree(self.state_dir)

    @staticmethod
    def _restore_env(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _record(self, version="0.148.0", verdict="verified", runs=None,
                verified_at=1_000_000.0):
        if runs is None:
            runs = sandbox_canary.DEFAULT_RUNS
        return {
            "cli": "codex",
            "version": version,
            "verified_at": verified_at,
            "runs": runs,
            "clean": runs if verdict == "verified" else 0,
            "breaches": 1 if verdict == "breached" else 0,
            "inconclusive": 1 if verdict == "inconclusive" else 0,
            "verdict": verdict,
        }

    def _write_record_directly(self, record):
        path = os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle)

    def test_non_read_only_lane_is_allowed(self):
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        self.assertIsNone(sandbox_canary.gate(False, "codex"))

    def test_non_codex_read_only_lane_is_allowed(self):
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        self.assertIsNone(sandbox_canary.gate(True, "claude"))

    def test_known_breached_version_cannot_be_whitewashed(self):
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        self._write_record_directly(self._record(version="0.147.0"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_measured_breached_version_cannot_be_whitewashed(self):
        # 0.154.0 was measured breached (issue #94): a verified record for it,
        # stale or minted before the upgrade, must not reopen the lane.
        sandbox_canary.codex_cli_version = lambda: "0.154.0"
        self._write_record_directly(self._record(version="0.154.0"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertIsNotNone(refusal, "the gate let a known-breached version through")
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_breach_refusal_does_not_attribute_every_version_to_one_issue(self):
        # The message named issue #26 for every listed version, which is
        # 0.147.0's. Unpinned, restoring that text passes the whole suite.
        sandbox_canary.codex_cli_version = lambda: "0.154.0"
        self._write_record_directly(self._record(version="0.154.0"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "sandbox_breached")
        self.assertNotIn("#26", refusal["detail"])
        self.assertIn("0.154.0", refusal["detail"])

    def test_a_version_outside_the_list_still_rests_on_its_record(self):
        # The list refuses named versions, not every version: an unlisted one
        # is decided by measurement, or this becomes a deny-all by accident —
        # including a version *above* every listed one, which is what a
        # floor-style check ("anything from 0.154.0 up") would get wrong.
        # Derived, not hardcoded: naming next release would need editing the
        # day that release is legitimately listed.
        version = "%d.0.0" % (1 + max(
            int(v.split(".")[0]) for v in sandbox_canary.KNOWN_BREACHED))
        self.assertNotIn(version, sandbox_canary.KNOWN_BREACHED)
        sandbox_canary.codex_cli_version = lambda: version
        self._write_record_directly(self._record(version=version))
        self.assertIsNone(sandbox_canary.gate(True, "codex", now=1_000_001.0))

    def test_battery_spends_no_probe_on_a_listed_version(self):
        # Issue #99: the gate decides a listed version on its name, so a
        # record cannot reopen the lane and the probes buy nothing. The
        # refusal the operator sees must not send them here either.
        sandbox_canary.codex_cli_version = lambda: "0.154.0"
        spawned = []

        async def spy(prompt, timeout_s):
            spawned.append(prompt)
            return {"ok": True, "text": "BLOCKED"}

        old = sandbox_canary._run_vendor_readonly_probe
        sandbox_canary._run_vendor_readonly_probe = spy
        try:
            out = sandbox_canary.run_write_canary(runs=8, timeout_s=1)
        finally:
            sandbox_canary._run_vendor_readonly_probe = old
        self.assertFalse(out.get("ok", False))
        self.assertEqual(out["reason"], "canary_error")
        self.assertIn("--remeasure", out["detail"])
        self.assertEqual(spawned, [])
        self.assertIsNone(sandbox_canary.read_record())

    def test_a_measured_breach_is_the_one_case_that_says_upgrade(self):
        # An unlisted version whose own record found a write is the problem
        # itself; there the upgrade is the fix and re-measuring after it is
        # the follow-up. This is the branch that keeps the old remedy.
        sandbox_canary.codex_cli_version = lambda: "0.148.0"
        self._write_record_directly(
            self._record(version="0.148.0", verdict="breached"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")
        self.assertIn("upgrade", refusal["detail"])
        self.assertIn("omni canary codex-write", refusal["detail"])

    def test_battery_matches_the_list_the_way_the_gate_does(self):
        # A membership test without normalization passes every other test
        # here (they all use the exact string) and spends eight probes on a
        # build-suffixed build of the same listed version.
        sandbox_canary.codex_cli_version = lambda: "0.154.0+build"
        spawned = []

        async def spy(prompt, timeout_s):
            spawned.append(prompt)
            return {"ok": True, "text": "BLOCKED"}

        old = sandbox_canary._run_vendor_readonly_probe
        sandbox_canary._run_vendor_readonly_probe = spy
        try:
            out = sandbox_canary.run_write_canary(runs=8, timeout_s=1)
        finally:
            sandbox_canary._run_vendor_readonly_probe = old
        self.assertEqual(out["reason"], "canary_error")
        self.assertEqual(spawned, [])

    def test_remeasure_returns_evidence_and_writes_no_record(self):
        # The way to disprove a listed entry is to measure it, but the
        # measurement must not mint the "verified" artifact the gate ignores.
        sandbox_canary.codex_cli_version = lambda: "0.154.0"

        async def clean(prompt, timeout_s):
            return {"ok": True, "text": "BLOCKED"}

        old = sandbox_canary._run_vendor_readonly_probe
        sandbox_canary._run_vendor_readonly_probe = clean
        try:
            out = sandbox_canary.run_write_canary(
                runs=sandbox_canary.MIN_RUNS, timeout_s=1, remeasure=True)
        finally:
            sandbox_canary._run_vendor_readonly_probe = old
        # Eight clean probes on an intermittent breach are a sample, not a
        # disproof, so the evidence must not wear the gate's allow-token —
        # that word is what makes someone reach for the list entry.
        self.assertNotIn("verdict", out)
        self.assertEqual(out["measurement"], "clean")
        self.assertEqual(out["clean"], sandbox_canary.MIN_RUNS)
        self.assertIs(out["recorded"], False)
        self.assertEqual(out["gate"], "refused_on_version")
        self.assertIsNone(sandbox_canary.read_record())
        # And the lane is still shut, which is the whole point.
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_the_two_refusals_do_not_share_one_remedy(self):
        # A listed version is not fixed by measuring it again; an unverified
        # one is measured by exactly that. One remedy for both sent the
        # operator to burn eight probes for nothing (issue #99).
        sandbox_canary.codex_cli_version = lambda: "0.154.0"
        self._write_record_directly(self._record(version="0.154.0"))
        breached = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(breached["reason"], "sandbox_breached")
        self.assertNotIn("omni canary codex-write", breached["detail"])

        sandbox_canary.codex_cli_version = lambda: "0.148.0"
        os.remove(os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME))
        unverified = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(unverified["reason"], "sandbox_unverified")
        self.assertIn("omni canary codex-write", unverified["detail"])

        # A version the gate cannot read is one it will not read off a record
        # either, so these two cannot be answered by running the battery —
        # and for an unparseable version the battery would spend probes and
        # then disagree with the gate, which is this issue all over again.
        # And an unlisted version whose record is merely missing must not be
        # told to upgrade first: it is verifiable as it stands, and the
        # upgrade could land on a listed version that neither the gate nor
        # the battery will touch.
        self.assertNotIn("upgrade", unverified["detail"])

        sandbox_canary.codex_cli_version = lambda: "banana.1"
        unreadable = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(unreadable["reason"], "sandbox_unverified")
        self.assertNotIn("omni canary codex-write", unreadable["detail"])

        sandbox_canary.codex_cli_version = lambda: None
        sandbox_canary._codex_installed = lambda: True
        unknown = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(unknown["reason"], "sandbox_unverified")
        self.assertNotIn("omni canary codex-write", unknown["detail"])

    def test_build_suffixed_known_breached_version_is_breached(self):
        sandbox_canary.codex_cli_version = lambda: "0.147.0+build"
        self._write_record_directly(self._record(version="0.147.0+build"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_zero_padded_known_breached_version_is_breached(self):
        sandbox_canary.codex_cli_version = lambda: "00.147.0"
        self._write_record_directly(self._record(version="00.147.0"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_unnormalizable_version_is_unverified(self):
        sandbox_canary.codex_cli_version = lambda: "banana.1"
        self._write_record_directly(self._record(version="banana.1"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_normalized_safe_version_is_allowed(self):
        self._write_record_directly(self._record(version="0.148.0"))
        self.assertIsNone(
            sandbox_canary.gate(True, "codex", now=1_000_001.0)
        )

    def test_missing_record_is_unverified(self):
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_record_for_other_version_is_unverified(self):
        self._write_record_directly(self._record(version="0.147.9"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_breached_record_is_rejected_as_breached(self):
        self._write_record_directly(self._record(verdict="breached"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_inconclusive_record_is_unverified(self):
        self._write_record_directly(self._record(verdict="inconclusive"))
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_too_few_runs_is_unverified(self):
        record = self._record(runs=sandbox_canary.MIN_RUNS - 1)
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_zero_clean_with_eight_runs_is_unverified(self):
        record = self._record()
        record["clean"] = 0
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_seven_clean_with_eight_runs_is_unverified(self):
        record = self._record()
        record["clean"] = record["runs"] - 1
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_missing_clean_counter_is_unverified(self):
        record = self._record()
        del record["clean"]
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_missing_cli_field_is_unverified(self):
        record = self._record()
        del record["cli"]
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_non_codex_cli_field_is_unverified(self):
        record = self._record()
        record["cli"] = "claude"
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_boolean_counter_is_unverified(self):
        record = self._record()
        record["clean"] = True
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_consistent_counter_schema_is_allowed(self):
        record = self._record()
        self._write_record_directly(record)
        self.assertIsNone(
            sandbox_canary.gate(True, "codex", now=1_000_001.0)
        )

    def test_stale_verified_record_is_unverified(self):
        now = 2_000_000.0
        record = self._record(
            verified_at=now - sandbox_canary.MAX_AGE_S - 1,
        )
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=now)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_future_verified_record_is_unverified(self):
        now = 2_000_000.0
        self._write_record_directly(
            self._record(verified_at=now + 10_000),
        )
        refusal = sandbox_canary.gate(True, "codex", now=now)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_nan_verified_record_is_unverified(self):
        self._write_record_directly(
            self._record(verified_at=float("nan")),
        )
        refusal = sandbox_canary.gate(True, "codex", now=2_000_000.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_infinite_verified_record_is_unverified(self):
        self._write_record_directly(
            self._record(verified_at=float("inf")),
        )
        refusal = sandbox_canary.gate(True, "codex", now=2_000_000.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_nonrepresentable_verified_time_is_unverified(self):
        self._write_record_directly(
            self._record(verified_at=10**10000),
        )
        refusal = sandbox_canary.gate(True, "codex", now=2_000_000.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_gate_never_raises_on_an_unforeseen_error(self):
        # run_agent promises a typed envelope and never raises, so the gate's
        # outer guard must convert ANY unexpected failure into a fail-closed
        # refusal — not just the specific cases with their own checks.
        def _boom(_now=None):
            raise RuntimeError("unforeseen")
        self.addCleanup(setattr, sandbox_canary, "_gate_codex",
                        sandbox_canary._gate_codex)
        sandbox_canary._gate_codex = _boom
        refusal = sandbox_canary.gate(True, "codex")
        self.assertEqual(refusal["reason"], "sandbox_unverified")
        # and the non-codex / write-mode short circuits still cannot raise
        self.assertIsNone(sandbox_canary.gate(False, "codex"))
        self.assertIsNone(sandbox_canary.gate(True, "claude"))

    def test_fresh_verified_record_is_allowed(self):
        self._write_record_directly(self._record())
        self.assertIsNone(
            sandbox_canary.gate(True, "codex", now=1_000_001.0)
        )

    def test_missing_codex_binary_is_left_to_downstream_refusal(self):
        sandbox_canary.codex_cli_version = lambda: None
        self.assertIsNone(sandbox_canary.gate(True, "codex"))

    def test_installed_unversionable_codex_is_unverified(self):
        sandbox_canary.codex_cli_version = lambda: None
        sandbox_canary._codex_installed = lambda: True
        refusal = sandbox_canary.gate(True, "codex")
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_absent_unversionable_codex_is_left_to_cli_missing(self):
        sandbox_canary.codex_cli_version = lambda: None
        sandbox_canary._codex_installed = lambda: False
        self.assertIsNone(sandbox_canary.gate(True, "codex"))

    def test_codex_cli_version_parser_rejects_non_version_token(self):
        class _ProcessResult:
            returncode = 0
            stderr = ""

            def __init__(self, stdout):
                self.stdout = stdout

        output = {"stdout": "codex-cli banana\n"}
        old_which = sandbox_canary.shutil.which
        old_run = sandbox_canary.subprocess.run
        self.addCleanup(setattr, sandbox_canary.shutil, "which", old_which)
        self.addCleanup(setattr, sandbox_canary.subprocess, "run", old_run)
        sandbox_canary.shutil.which = lambda _name: "/fake/codex"
        sandbox_canary.subprocess.run = (
            lambda *_args, **_kwargs: _ProcessResult(output["stdout"])
        )

        self.assertIsNone(self.old_version_function())
        output["stdout"] = "codex-cli 0.147.0\n"
        self.assertEqual(self.old_version_function(), "0.147.0")

    def test_verified_label_with_breaches_is_breached(self):
        record = self._record()
        record["breaches"] = 1
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_verified_label_with_inconclusive_is_unverified(self):
        record = self._record()
        record["breaches"] = 0
        record["inconclusive"] = 1
        self._write_record_directly(record)
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_corrupt_record_is_none_and_gate_refuses(self):
        path = os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        self.assertIsNone(sandbox_canary.read_record())
        refusal = sandbox_canary.gate(True, "codex", now=1_000_001.0)
        self.assertEqual(refusal["reason"], "sandbox_unverified")

    def test_record_round_trip_and_state_dir_override(self):
        record = self._record()
        sandbox_canary.write_record(record)
        self.assertEqual(sandbox_canary.read_record(), record)
        self.assertEqual(
            sandbox_canary.record_path(),
            os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME),
        )

    # Pins the CODEX write-canary gate, so the model is spelled explicitly:
    # the agent-lane default is not a codex model and read-only would be
    # refused by an earlier, unrelated gate.
    CODEX_MODEL = "codex/" + config.DEFAULT_CODEX_MODEL

    def test_jobs_preflight_default_model_refuses_read_only(self):
        """Read-only on the lane default is a LOUD refusal, not a reroute.

        The default is a grok id and grok's strict sandbox cannot separate
        reads from writes, so preflight must say so instead of substituting
        another backend.
        """
        with tempfile.TemporaryDirectory() as workdir:
            refusal, _normalized, _route = jobs._preflight(
                config.DEFAULT_AGENT_MODEL, workdir, True, False,
            )
            self.assertIsNotNone(refusal, "read-only default must be refused")
            self.assertEqual(refusal["reason"], "unsupported_read_only")

    def test_jobs_preflight_uses_sandbox_gate(self):
        with tempfile.TemporaryDirectory() as workdir:
            sandbox_canary.codex_cli_version = lambda: "0.147.0"
            refusal, _normalized, _route = jobs._preflight(
                self.CODEX_MODEL, workdir, True, False,
            )
            self.assertEqual(refusal["reason"], "sandbox_breached")

            # The production path checks freshness against real time, so this
            # positive twin needs a genuinely fresh record (arbitration fix:
            # the helper's 1970 default is stale by contract).
            self._write_record_directly(
                self._record(verified_at=time.time()))
            sandbox_canary.codex_cli_version = lambda: "0.148.0"
            refusal, _normalized, _route = jobs._preflight(
                self.CODEX_MODEL, workdir, True, False,
            )
            if refusal is not None:
                self.assertNotIn(
                    refusal["reason"],
                    {"sandbox_breached", "sandbox_unverified"},
                )

    def test_sync_agent_lane_has_sandbox_gate_wiring_pin(self):
        old_run_cli = agent_lane.run_cli
        old_which = agent_lane.shutil.which
        self.addCleanup(setattr, agent_lane, "run_cli", old_run_cli)
        self.addCleanup(setattr, agent_lane.shutil, "which", old_which)
        bomb_calls = {"count": 0}

        def _bomb(*_args, **_kwargs):
            bomb_calls["count"] += 1
            raise AssertionError("run_cli must not be called after gate refusal")

        agent_lane.run_cli = _bomb
        agent_lane.shutil.which = lambda _name: "/fake/codex"
        sandbox_canary.codex_cli_version = lambda: "0.147.0"
        with tempfile.TemporaryDirectory() as workdir:
            envelope = asyncio.run(agent_lane.run_agent(
                task="read only",
                model="codex/gpt-5.6-sol",
                workdir=workdir,
                read_only=True,
            ))
        self.assertEqual(envelope["reason"], "sandbox_breached")
        self.assertEqual(bomb_calls["count"], 0)

    def test_write_canary_inconclusive_probes_never_verify(self):
        # A probe battery where codex neither writes the file nor replies
        # BLOCKED is inconclusive; the verdict must NOT be "verified" (a weak
        # sample must never mint a record the gate would later trust).
        async def _fake(*_a, **_k):
            return {"ok": True, "text": "sure, done"}  # no BLOCKED, no write
        self.addCleanup(
            setattr, sandbox_canary, "_run_vendor_readonly_probe",
            sandbox_canary._run_vendor_readonly_probe,
        )
        sandbox_canary._run_vendor_readonly_probe = _fake
        sandbox_canary.codex_cli_version = lambda: "0.148.0"
        record = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1)
        self.assertEqual(record["verdict"], "inconclusive")
        self.assertEqual(record["inconclusive"], sandbox_canary.MIN_RUNS)
        # And a record born inconclusive is refused by the gate.
        self.assertEqual(
            sandbox_canary.gate(True, "codex", now=record["verified_at"])["reason"],
            "sandbox_unverified")

    def test_write_canary_breach_probe_sets_breached(self):
        # If any probe finds the file written, the verdict is "breached".
        # Hermetic: redirect HOME so the canary target (built from
        # expanduser("~")) lands in a tempdir, never the real home.
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        old_home = os.environ.get("HOME")
        self.addCleanup(self._restore_env, "HOME", old_home)
        os.environ["HOME"] = home
        target_seen = {"n": 0}
        async def _fake(prompt, *_a, **_k):
            # Emulate a breach on the first probe: create the target file.
            import re as _re
            m = _re.search(r"at (\S+) containing", prompt)
            if m and target_seen["n"] == 0:
                target_seen["n"] += 1
                with open(m.group(1), "w") as fh:
                    fh.write("PWNED")
            return {"ok": True, "text": "ok"}
        self.addCleanup(
            setattr, sandbox_canary, "_run_vendor_readonly_probe",
            sandbox_canary._run_vendor_readonly_probe,
        )
        sandbox_canary._run_vendor_readonly_probe = _fake
        sandbox_canary.codex_cli_version = lambda: "0.148.0"
        record = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1)
        self.assertEqual(record["verdict"], "breached")
        self.assertGreaterEqual(record["breaches"], 1)

    def test_write_canary_write_outranks_probe_error(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        old_home = os.environ.get("HOME")
        self.addCleanup(self._restore_env, "HOME", old_home)
        os.environ["HOME"] = home

        async def _fake(prompt, *_args, **_kwargs):
            import re as _re
            match = _re.search(r"at (\S+) containing", prompt)
            with open(match.group(1), "w", encoding="utf-8") as handle:
                handle.write("PWNED")
            return {"ok": False, "reason": "rate_limit", "detail": "late"}

        old_probe = sandbox_canary._run_vendor_readonly_probe
        self.addCleanup(
            setattr, sandbox_canary, "_run_vendor_readonly_probe", old_probe,
        )
        sandbox_canary._run_vendor_readonly_probe = _fake
        record = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1,
        )
        self.assertEqual(record["verdict"], "breached")
        self.assertEqual(record["runs"], 1)
        self.assertEqual(record["clean"] + record["breaches"] +
                         record["inconclusive"], record["runs"])
        self.assertEqual(sandbox_canary.read_record(), record)
        refusal = sandbox_canary.gate(
            True, "codex", now=record["verified_at"],
        )
        self.assertEqual(refusal["reason"], "sandbox_breached")

    def test_write_canary_version_change_writes_no_record(self):
        versions = iter(("0.148.0", "9.9.9"))
        sandbox_canary.codex_cli_version = lambda: next(versions)
        probe_calls = {"count": 0}

        async def _fake(*_args, **_kwargs):
            probe_calls["count"] += 1
            return {"ok": True, "text": "BLOCKED"}

        old_probe = sandbox_canary._run_vendor_readonly_probe
        self.addCleanup(
            setattr, sandbox_canary, "_run_vendor_readonly_probe", old_probe,
        )
        sandbox_canary._run_vendor_readonly_probe = _fake
        result = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1,
        )
        self.assertEqual(result["reason"], "canary_error")
        self.assertEqual(probe_calls["count"], sandbox_canary.MIN_RUNS)
        self.assertFalse(os.path.exists(
            os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME),
        ))

    def test_write_canary_missing_initial_version_runs_no_probes(self):
        sandbox_canary.codex_cli_version = lambda: None
        probe_calls = {"count": 0}

        async def _fake(*_args, **_kwargs):
            probe_calls["count"] += 1
            return {"ok": True, "text": "BLOCKED"}

        old_probe = sandbox_canary._run_vendor_readonly_probe
        self.addCleanup(
            setattr, sandbox_canary, "_run_vendor_readonly_probe", old_probe,
        )
        sandbox_canary._run_vendor_readonly_probe = _fake
        result = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1,
        )
        self.assertEqual(result["reason"], "canary_error")
        self.assertEqual(probe_calls["count"], 0)
        self.assertFalse(os.path.exists(
            os.path.join(self.state_dir, sandbox_canary.RECORD_BASENAME),
        ))

    def test_write_canary_argv_is_vendor_readonly_when_kernel_available(self):
        # The battery that mints sandbox_canary.json must measure vendor
        # --sandbox read-only even on Darwin (consult wrap must not
        # whitewash the record that authorises agent --read-only).
        from providers import codex as codex_mod

        kernel_sandbox.available = lambda: True
        wrap_calls = []
        old_wrap = kernel_sandbox.wrap_argv

        def _wrap(*a, **k):
            wrap_calls.append(1)
            return old_wrap(*a, **k)

        self.addCleanup(setattr, kernel_sandbox, "wrap_argv", old_wrap)
        kernel_sandbox.wrap_argv = _wrap

        def _bomb(*_a, **_k):
            raise AssertionError(
                "consult run_codex_subprocess must not mint the canary"
            )

        old_consult = codex_mod.run_codex_subprocess
        self.addCleanup(
            setattr, codex_mod, "run_codex_subprocess", old_consult,
        )
        codex_mod.run_codex_subprocess = _bomb

        recorded = []

        async def _fake_run_cli(argv, **kwargs):
            recorded.append(list(argv))
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as fh:
                    fh.write("BLOCKED")
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "timed_out": False}

        old_run = sandbox_canary.run_cli
        self.addCleanup(setattr, sandbox_canary, "run_cli", old_run)
        sandbox_canary.run_cli = _fake_run_cli

        old_which = sandbox_canary.shutil.which
        self.addCleanup(setattr, sandbox_canary.shutil, "which", old_which)
        sandbox_canary.shutil.which = (
            lambda name, *a, **k: "/fake/bin/codex"
            if os.path.basename(str(name)) == "codex"
            else old_which(name, *a, **k)
        )

        record = sandbox_canary.run_write_canary(
            runs=sandbox_canary.MIN_RUNS, timeout_s=1,
        )
        self.assertEqual(record.get("verdict"), "verified", record)
        self.assertEqual(len(recorded), sandbox_canary.MIN_RUNS)
        self.assertEqual(wrap_calls, [])
        for argv in recorded:
            self.assertTrue(
                sandbox_canary.is_vendor_readonly_argv(argv), argv,
            )
            self.assertNotIn("sandbox-exec", argv)
            self.assertNotIn(
                "--dangerously-bypass-approvals-and-sandbox", argv,
            )


if __name__ == "__main__":
    unittest.main()
