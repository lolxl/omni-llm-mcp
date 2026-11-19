# no-opencc
"""``quota check``: decisions, fail-closed reads, allowlisted output, audit, PII.

Every case runs against a throwaway state dir and a pinned clock. The Codex RPC is
replaced at ``codex_quota.read_rate_limits`` (or disabled with
OMNI_LLM_QUOTA_PROBE=0); nothing here spawns a vendor binary.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from providers import codex_quota, health, quota

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)
EMAIL = "noreply@example.invalid"
LONG, SHORT = "long", "short"


def _argv(login="openai", account=None, task=LONG, context="primary",
          model="gpt-6.2-sol", transport="codex:gpt-6.2-sol", snapshot=None):
    argv = ["check", "--login", login, "--transport", transport, "--model", model,
            "--task-class", task, "--context", context, "--json"]
    if account is not None:
        argv += ["--account", account]
    if snapshot is not None:
        argv += ["--anthropic-snapshot", snapshot]
    return argv


def _info(primary=40.0, secondary=None, age=10, reached=False, **extra):
    info = {"used_percent": primary, "resets_at": None,
            "window_mins": 300 if primary is not None else None,
            "reached": reached, "primary_reached": reached, "plan": "pro",
            "checked_at": NOW.timestamp() - age}
    if secondary is not None:
        info["secondary"] = {"used_percent": secondary, "resets_at": None,
                             "window_mins": 10080, "reached": False}
    info.update(extra)
    return info


class QuotaCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="omni-quota-test-")
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "state")
        os.makedirs(self.state)
        env = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "0",
            "OMNI_LLM_NOTIFY_QUIET": "1"})
        env.start()
        self.addCleanup(env.stop)
        clock = mock.patch.object(quota, "_now", lambda: NOW)
        clock.start()
        self.addCleanup(clock.stop)
        self.rpc_calls = []
        self.scoped_patchers = []      # a scenario loop stops these between scenarios

    # -- fixtures ---------------------------------------------------------
    def rpc(self, info=None, err=None, exc=None):
        def fake(profile="default", timeout_s=15, **kwargs):
            self.rpc_calls.append((profile, timeout_s))
            if exc is not None:
                raise exc
            return info, err
        patcher = mock.patch.object(codex_quota, "read_rate_limits", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scoped_patchers.append(patcher)

    def write(self, name, value, raw=False):
        path = os.path.join(self.state, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value if raw else json.dumps(value))
        return path

    @staticmethod
    def row_name(profile="default"):
        """State-dir-relative path of one profile's cache file."""
        return os.path.join("quota", hashlib.sha256(
            profile.encode("utf-8")).hexdigest() + ".json")

    def write_row(self, value, profile="default", raw=False):
        """Write one profile's cache file; a dict gets its ``profile`` label."""
        if not raw and isinstance(value, dict):
            value = dict(value, profile=value.get("profile", profile))
        return self.write(self.row_name(profile), value, raw=raw)

    def snapshot(self, windows=None, age=60, **overrides):
        doc = {"generated_at": (NOW - timedelta(seconds=age)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "windows": windows if windows is not None else [
                   {"kind": "five_hour", "used_percent": 31.2},
                   {"kind": "seven_day", "used_percent": 55.0}]}
        doc.update(overrides)
        return self.write("anthropic-snapshot.json", doc)

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = quota.main(list(argv))
        doc = json.loads(out.getvalue()) if out.getvalue() else None
        return code, doc, out.getvalue(), err.getvalue()

    def check(self, **kwargs):
        code, doc, _out, _err = self.run_cli(_argv(**kwargs))
        self.assertIsNotNone(doc)
        return code, doc

    def audit_lines(self):
        path = os.path.join(self.state, quota.AUDIT_FILE)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()


class CodexTests(QuotaCase):
    def test_eligible_uses_both_windows_and_records_the_observation(self):
        self.rpc(_info(primary=40.0, secondary=61.5, age=10))
        code, doc = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(doc["decision"], "eligible")
        self.assertEqual(doc["reasons"], ["ok"])
        self.assertEqual(doc["quota_pool"], "openai-sub")
        self.assertEqual((doc["login"], doc["account"]), ("openai", "default"))
        self.assertEqual(doc["used_bp"], 6150)
        self.assertEqual(doc["window"], "secondary")
        self.assertEqual(doc["observed_at"], "2026-11-18T11:59:50Z")
        self.assertEqual(doc["freshness_s"], 10)
        self.assertEqual(doc["checked_at"], "2026-11-18T12:00:00Z")
        # expires_at = checked_at + min(remaining freshness 60 - 10, 300 s)
        self.assertEqual(doc["expires_at"], "2026-11-18T12:00:50Z")

    def test_rpc_is_called_for_the_default_profile_with_a_5_second_budget(self):
        self.rpc(_info())
        self.check()
        self.assertEqual(self.rpc_calls, [("default", 5)])

    def test_threshold_boundary_uses_decimal_half_up(self):
        for percent, decision, reason in (
                (89.99, "eligible", "ok"), (89.994, "eligible", "ok"),
                (89.995, "ineligible", "quota_reserve"),
                (90.0, "ineligible", "quota_reserve"), (100, "ineligible", "quota_reserve")):
            self.rpc(_info(primary=percent))
            code, doc = self.check()
            self.assertEqual((doc["decision"], doc["reasons"][0]), (decision, reason),
                             percent)
            self.assertEqual(code, 0 if decision == "eligible" else 10)

    def test_the_higher_window_decides(self):
        self.rpc(_info(primary=10.0, secondary=93.0))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"], doc["used_bp"], doc["window"]),
                         (10, ["quota_reserve"], 9300, "secondary"))

    def test_vendor_reached_refuses_even_with_a_low_percentage(self):
        self.rpc(_info(primary=3.0, reached=True))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["quota_reserve"]))
        self.rpc(_info(primary=3.0, secondary=4.0,
                       secondary_extra=None))
        info = _info(primary=3.0, secondary=4.0)
        info["secondary"]["reached"] = True
        self.rpc(info)
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["quota_reserve"]))

    def test_reached_without_any_percentage_still_refuses(self):
        self.rpc(_info(primary=None, reached=True))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"], doc["used_bp"]),
                         (10, ["quota_reserve"], None))

    def test_no_window_at_all_is_missing_telemetry(self):
        self.rpc(_info(primary=None))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["missing_telemetry"]))

    def test_unknown_or_non_finite_percentages_are_malformed(self):
        bad = [float("nan"), float("inf"), -1.0, 100.5, True, "40"]
        for value in bad:
            self.rpc(_info(primary=value))
            code, doc = self.check()
            self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]),
                             repr(value))
        info = _info(primary=10.0, secondary=None)
        info["secondary"] = {"used_percent": None, "reached": False}
        self.rpc(info)
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]))
        self.rpc(_info(primary=10.0, quota_unknown=True))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]))

    def test_missing_or_future_or_old_observation_is_not_healthy(self):
        info = _info()
        info.pop("checked_at")
        self.rpc(info)
        self.assertEqual(self.check()[1]["reasons"], ["malformed_telemetry"])
        self.rpc(_info(age=61))
        self.assertEqual(self.check()[1]["reasons"], ["stale_telemetry"])
        self.rpc(_info(age=-30))
        self.assertEqual(self.check()[1]["reasons"], ["stale_telemetry"])
        self.rpc(_info(age=60))
        self.assertEqual(self.check()[0], 0)

    def test_rpc_error_timeout_or_exception_is_telemetry_unavailable(self):
        for err in ({"ok": False, "reason": "timeout", "detail": "quota_rpc_deadline"},
                    {"ok": False, "reason": "disabled", "detail": "quota_probe_disabled"},
                    {"ok": False, "reason": "auth_error", "detail": EMAIL}):
            self.rpc(None, err)
            code, doc = self.check()
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["telemetry_unavailable"]))
            self.assertEqual((doc["used_bp"], doc["observed_at"], doc["freshness_s"]),
                             (None, None, None))
        self.rpc(exc=RuntimeError(EMAIL))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["telemetry_unavailable"]))

    def test_probe_disabled_without_a_stub_is_telemetry_unavailable(self):
        code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["telemetry_unavailable"]))

    def test_a_fresh_account_keyed_cache_is_used_without_spawning(self):
        home = os.path.join(self.tmp.name, "codex-home")
        os.makedirs(home)
        digest = hashlib.sha256(os.path.realpath(home).encode("utf-8")).hexdigest()
        self.write_row({
            "used_percent": 12.0, "resets_at": None, "window_mins": 300,
            "reached": False, "primary_reached": False, "plan": "pro",
            "checked_at": NOW.timestamp() - 20, "home_id": digest,
            "secondary": {"used_percent": 33.0, "resets_at": None,
                          "window_mins": 10080, "reached": False}})
        env = {"CODEX_HOME": home, "OMNI_LLM_QUOTA_PROBE": "1"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(codex_quota, "_spawn_rate_limits",
                                  side_effect=AssertionError("spawned")), \
                mock.patch.object(codex_quota.time, "time",
                                  lambda: NOW.timestamp()):
            code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["used_bp"], doc["freshness_s"]),
                         (0, "eligible", 3300, 20))

    def test_short_task_softens_only_a_valid_observation_over_the_threshold(self):
        self.rpc(_info(primary=95.0))
        code, doc = self.check(task=SHORT)
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (0, "advisory", ["quota_reserve"]))
        self.rpc(_info(primary=3.0, reached=True))
        code, doc = self.check(task=SHORT)
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (0, "advisory", ["quota_reserve"]))

    def test_short_task_refuses_a_failed_or_timed_out_rpc(self):
        for err in ({"ok": False, "reason": "timeout"},
                    {"ok": False, "reason": "cli_missing"},
                    {"ok": False, "reason": "disabled"}):
            self.rpc(None, err)
            code, doc = self.check(task=SHORT)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["telemetry_unavailable"]))
        self.rpc(exc=RuntimeError("x"))
        self.assertEqual(self.check(task=SHORT)[0], 10)

    def test_short_task_refuses_every_invalid_observation(self):
        bad = {
            "stale": _info(age=61),
            "future": _info(age=-30),
            "no observation time": {k: v for k, v in _info().items() if k != "checked_at"},
            "nan percent": _info(primary=float("nan")),
            "unknown quota": _info(quota_unknown=True),
            "no windows": _info(primary=None),
        }
        for label, info in bad.items():
            self.rpc(info)
            code, doc = self.check(task=SHORT)
            self.assertEqual((code, doc["decision"]), (10, "ineligible"), label)
            self.assertNotIn("quota_reserve", doc["reasons"], label)


class AnthropicTests(QuotaCase):
    def test_eligible_takes_the_highest_window(self):
        path = self.snapshot()
        code, doc = self.check(login="anthropic", snapshot=path, model="claude-fable-5-2")
        self.assertEqual((code, doc["decision"], doc["reasons"]), (0, "eligible", ["ok"]))
        self.assertEqual((doc["quota_pool"], doc["used_bp"], doc["window"]),
                         ("anthropic-sub", 5500, "seven_day"))
        self.assertEqual((doc["observed_at"], doc["freshness_s"]),
                         ("2026-11-18T11:59:00Z", 60))
        # remaining freshness 7200 - 60 is far above the 300 s cap
        self.assertEqual(doc["expires_at"], "2026-11-18T12:05:00Z")

    def test_expiry_follows_remaining_freshness_when_it_is_short(self):
        path = self.snapshot(age=7190)
        _code, doc = self.check(login="anthropic", snapshot=path)
        self.assertEqual(doc["expires_at"], "2026-11-18T12:00:10Z")

    def test_missing_snapshot_argument_or_file(self):
        code, doc = self.check(login="anthropic")
        self.assertEqual((code, doc["reasons"]), (10, ["missing_telemetry"]))
        code, doc = self.check(login="anthropic",
                               snapshot=os.path.join(self.state, "absent.json"))
        self.assertEqual((code, doc["reasons"]), (10, ["missing_telemetry"]))
        code, doc = self.check(login="anthropic", snapshot=self.state)  # a directory
        self.assertEqual((code, doc["reasons"]), (10, ["missing_telemetry"]))

    def test_malformed_snapshots(self):
        good = [{"kind": "five_hour", "used_percent": 10}]
        cases = {
            "not json": ("{", True),
            "list root": ("[]", True),
            "nan": ('{"generated_at": "2026-11-18T11:59:00Z", "windows": '
                    '[{"kind": "a", "used_percent": NaN}]}', True),
            "dup key": ('{"windows": [], "windows": []}', True),
        }
        for label, (text, raw) in cases.items():
            path = self.write("snap.json", text, raw=raw)
            code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]), label)
        for label, windows in (
                ("empty", []), ("not list", {}), ("percent high", [{"kind": "a", "used_percent": 100.1}]),
                ("percent low", [{"kind": "a", "used_percent": -0.1}]),
                ("percent bool", [{"kind": "a", "used_percent": True}]),
                ("percent str", [{"kind": "a", "used_percent": "5"}]),
                ("kind missing", [{"used_percent": 5}]),
                ("row not object", [5])):
            path = self.snapshot(windows=windows)
            code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]), label)
        for label, extra in (("no generated_at", None), ("bad generated_at", "yesterday"),
                             ("number generated_at", 5)):
            path = self.snapshot(windows=good, generated_at=extra)
            if extra is None:
                with open(path, encoding="utf-8") as handle:
                    value = json.load(handle)
                value.pop("generated_at")
                self.write("anthropic-snapshot.json", value)
            code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]), label)

    def test_age_and_future_tolerance(self):
        for age, reasons in ((7200, ["ok"]), (7201, ["stale_telemetry"]),
                             (-300, ["ok"]), (-301, ["stale_telemetry"])):
            path = self.snapshot(age=age)
            _code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual(doc["reasons"], reasons, age)
        _code, doc = self.check(login="anthropic", snapshot=self.snapshot(age=7201))
        self.assertEqual((doc["used_bp"], doc["window"]), (None, None))
        self.assertEqual(doc["freshness_s"], 7201)

    def test_threshold_boundary_matches_the_caller(self):
        for percent, reasons in ((89.99, ["ok"]), (89.995, ["quota_reserve"]),
                                 (90, ["quota_reserve"])):
            path = self.snapshot(windows=[{"kind": "five_hour", "used_percent": percent}])
            code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual(doc["reasons"], reasons, percent)
            self.assertEqual(code, 0 if reasons == ["ok"] else 10)

    def test_only_the_default_account_is_bound(self):
        path = self.snapshot()
        code, doc = self.check(login="anthropic", account="work", snapshot=path)
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["unknown_account"]))
        self.assertEqual((doc["quota_pool"], doc["login"], doc["account"]),
                         ("anthropic-sub", "anthropic", None))
        self.assertEqual((doc["observed_at"], doc["used_bp"]), (None, None))
        code, doc = self.check(login="anthropic", account="default", snapshot=path)
        self.assertEqual(code, 0)

    def test_unknown_account_is_never_softened_for_a_short_task(self):
        code, doc = self.check(login="anthropic", account="work", task=SHORT,
                               snapshot=self.snapshot())
        self.assertEqual((code, doc["decision"]), (10, "ineligible"))

    def test_short_task_softens_only_a_valid_snapshot_over_the_threshold(self):
        path = self.snapshot(windows=[{"kind": "five_hour", "used_percent": 95}])
        code, doc = self.check(login="anthropic", task=SHORT, snapshot=path)
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (0, "advisory", ["quota_reserve"]))

    def test_short_task_refuses_a_missing_stale_or_future_snapshot(self):
        for label, path, reason in (
                ("stale", self.snapshot(age=9000), "stale_telemetry"),
                ("future", self.snapshot(age=-301), "stale_telemetry"),
                ("missing", None, "missing_telemetry")):
            code, doc = self.check(login="anthropic", task=SHORT, snapshot=path)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", [reason]), label)

    def test_short_task_refuses_every_malformed_snapshot(self):
        bad = {
            "empty file": "", "not json": "{", "list root": "[]", "empty object": "{}",
            "nan": '{"generated_at": "2026-11-18T11:59:00Z", "windows": '
                   '[{"kind": "a", "used_percent": NaN}]}',
            "duplicate key": '{"generated_at": "2026-11-18T11:59:00Z", "windows": [], '
                             '"windows": [{"kind": "a", "used_percent": 99}]}',
        }
        for label, text in bad.items():
            path = self.write("snap.json", text, raw=True)
            code, doc = self.check(login="anthropic", task=SHORT, snapshot=path)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["malformed_telemetry"]), label)

    def test_the_snapshot_path_is_ignored_for_other_logins(self):
        self.rpc(_info())
        code, doc = self.check(login="openai", snapshot=self.snapshot(age=99999))
        self.assertEqual(code, 0)

    def test_window_label_is_validated(self):
        path = self.snapshot(windows=[{"kind": EMAIL, "used_percent": 12}])
        code, doc = self.check(login="anthropic", snapshot=path)
        self.assertEqual((code, doc["window"]), (0, "other"))
        self.assertNotIn(EMAIL, json.dumps(doc))


class UnmonitoredTests(QuotaCase):
    def test_primary_context_is_eligible_with_its_own_reason(self):
        for login, pool in (("xai", "xai-sub"), ("google", "google-sub"),
                            ("cursor", "cursor-sub")):
            code, doc = self.check(login=login, model="m")
            self.assertEqual((code, doc["decision"], doc["reasons"], doc["quota_pool"]),
                             (0, "eligible", ["unmonitored_pool"], pool))
            self.assertEqual((doc["used_bp"], doc["window"], doc["observed_at"],
                              doc["freshness_s"]), (None, None, None, None))
            self.assertEqual(doc["expires_at"], "2026-11-18T12:05:00Z")

    def test_fallback_context_is_ineligible_for_long_and_short(self):
        for task in (LONG, SHORT):
            code, doc = self.check(login="xai", context="fallback", task=task)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["unmonitored_pool"]), task)

    def test_monitored_pools_do_not_change_with_context(self):
        self.rpc(_info())
        self.assertEqual(self.check(context="fallback")[0], 0)

    def test_unknown_login_and_unknown_account(self):
        code, doc = self.check(login="mistral")
        self.assertEqual((code, doc["reasons"], doc["quota_pool"], doc["login"],
                          doc["account"]), (10, ["unknown_login"], None, None, None))
        code, doc = self.check(login="xai", account="team")
        self.assertEqual((code, doc["reasons"], doc["account"]),
                         (10, ["unknown_account"], None))


class LedgerTests(QuotaCase):
    def flag(self, key, until=3600, **extra):
        row = {"model": key.split("@")[0], "profile": "default",
               "until": NOW.timestamp() + until, "source": "quota",
               "flagged_at": NOW.timestamp(), "account": None}
        row.update(extra)
        ledger_path = os.path.join(self.state, "health.json")
        ledger = {}
        if os.path.exists(ledger_path):
            with open(ledger_path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        ledger[key] = row
        self.write("health.json", ledger)

    def test_missing_ledger_is_an_empty_ledger(self):
        self.rpc(_info())
        self.assertFalse(os.path.exists(os.path.join(self.state, "health.json")))
        self.assertEqual(self.check()[0], 0)

    def test_active_account_week_and_model_flags_cool_the_pool(self):
        for key in ("codex@default", "codex#week@default",
                    health.pool_key("codex/gpt-6.2-sol", "default"),
                    health.pool_key("gpt-6.2-sol", "default")):
            os.path.exists(os.path.join(self.state, "health.json")) and os.remove(
                os.path.join(self.state, "health.json"))
            self.rpc(_info())
            self.flag(key)
            code, doc = self.check()
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["ledger_cooling"]), key)

    def test_cooling_and_a_quota_refusal_report_both_reasons(self):
        self.flag("codex@default")
        self.rpc(_info(primary=95.0))
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_cooling", "quota_reserve"]))

    def test_expired_or_other_profile_flags_do_not_cool(self):
        self.rpc(_info())
        self.flag("codex@default", until=-5)
        self.flag("codex@other", until=3600)
        self.assertEqual(self.check()[0], 0)

    def test_unmonitored_pools_are_cooled_too(self):
        self.flag("grok@default")
        code, doc = self.check(login="xai", model="grok-4.5")
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_cooling"]))

    def test_cooling_is_never_softened_for_a_short_task(self):
        for source in ("quota", "manual", "crash", "rate_limit"):
            self.flag("codex@default", source=source)
            self.rpc(_info())
            code, doc = self.check(task=SHORT)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["ledger_cooling"]), source)
        # cooling next to a softenable refusal still refuses
        self.rpc(_info(primary=95.0))
        code, doc = self.check(task=SHORT)
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["ledger_cooling", "quota_reserve"]))

    def test_streak_only_rows_and_leases_are_valid(self):
        self.write("health.json", {
            "gpt-6.2-sol@default": {"crash_streak": 2, "last_crash_at": NOW.timestamp()},
            "_leases": {"tok": {"pool": "p", "expires": NOW.timestamp() + 5}}})
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)

    def test_unreadable_ledgers_refuse_and_never_reach_the_rpc(self):
        bad = {
            "malformed": "{not json",
            "list root": "[]",
            "scalar row": '{"codex@default": 5}',
            "null row": '{"codex@default": null}',
            "invalid expiry": '{"codex@default": {"until": "soon"}}',
            "bool expiry": '{"codex@default": {"until": true}}',
            "nan expiry": '{"codex@default": {"until": NaN}}',
            "bad credit hold": '{"codex@default": {"credit_hold": {"until": "x"}}}',
            "hold not object": '{"codex@default": {"credit_hold": 5}}',
            "duplicate key": '{"codex@default": {}, "codex@default": {}}',
            "empty file": "",
        }
        for label, text in bad.items():
            self.write("health.json", text, raw=True)
            self.rpc(_info())
            before = len(self.rpc_calls)
            for task in (LONG, SHORT):
                code, doc = self.check(task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (10, "ineligible", ["ledger_unreadable"]), (label, task))
            self.assertEqual(len(self.rpc_calls), before, label)

    def test_a_ledger_that_cannot_be_opened_refuses(self):
        os.mkdir(os.path.join(self.state, "health.json"))
        self.rpc(_info())
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_unreadable"]))

    def test_unmonitored_pool_also_needs_a_readable_ledger(self):
        self.write("health.json", "{", raw=True)
        code, doc = self.check(login="xai")
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["ledger_unreadable"]))

    def test_a_failure_while_matching_flags_refuses(self):
        with mock.patch.object(quota, "_ledger_cooling", side_effect=RuntimeError(EMAIL)):
            code, doc = self.check(login="xai")
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_unreadable"]))


class QuotaCacheTests(QuotaCase):
    @staticmethod
    def labelled(profile, inner):
        """One cache file's text: a row that carries its own profile label."""
        return '{"profile": "%s", %s}' % (profile, inner)

    def malformed_rows(self):
        """Each row defect, as the inner text of a row that is otherwise labelled."""
        return {
            "no checked_at": '"used_percent": 1',
            "bad checked_at": '"checked_at": "x"',
            "nan percent": '"checked_at": 1, "used_percent": NaN',
            "string percent": '"checked_at": 1, "used_percent": "1"',
            "bad secondary": '"checked_at": 1, "secondary": 5',
            "bad secondary pct": '"checked_at": 1, "secondary": {"used_percent": true}',
            "int plan": '"checked_at": 1, "plan": 17',
            "string quota_unknown": '"checked_at": 1, "quota_unknown": "false"',
            "checked_at string": '"checked_at": "1"',
            "checked_at bool": '"checked_at": true',
            "checked_at null": '"checked_at": null',
            "used_percent bool": '"checked_at": 1, "used_percent": true',
            "resets_at string": '"checked_at": 1, "resets_at": "x"',
            "window_mins string": '"checked_at": 1, "window_mins": "300"',
            "window_mins float": '"checked_at": 1, "window_mins": 300.5',
            "reached string": '"checked_at": 1, "reached": "no"',
            "primary_reached int": '"checked_at": 1, "primary_reached": 1',
            "home_id number": '"checked_at": 1, "home_id": 7',
            "secondary null": '"checked_at": 1, "secondary": null',
            "secondary list": '"checked_at": 1, "secondary": []',
            "secondary used_percent string":
                '"checked_at": 1, "secondary": {"used_percent": "9"}',
            "secondary resets_at bool":
                '"checked_at": 1, "secondary": {"resets_at": false}',
            "secondary window_mins string":
                '"checked_at": 1, "secondary": {"window_mins": "10080"}',
            "secondary reached string":
                '"checked_at": 1, "secondary": {"reached": "x"}',
        }

    def test_malformed_quota_cache_refuses_without_calling_the_rpc(self):
        bad = {
            "malformed": "{",
            "list root": "[]",
            "scalar root": "5",
            "null root": "null",
            "names another label": '{"profile": "work", "checked_at": 1}',
            "no label": '{"checked_at": 1}',
            "label not a string": '{"profile": 5, "checked_at": 1}',
        }
        for label, inner in self.malformed_rows().items():
            bad[label] = self.labelled("default", inner)
        self.rpc(_info())
        for label, text in bad.items():
            self.write_row(text, raw=True)
            before = len(self.rpc_calls)
            for task in (LONG, SHORT):
                code, doc = self.check(task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (10, "ineligible", ["telemetry_malformed"]), (label, task))
            self.assertEqual(len(self.rpc_calls), before, label)

    def test_a_malformed_sibling_file_does_not_block_the_requested_profile(self):
        self.write_row({"checked_at": NOW.timestamp(), "used_percent": 4.0})
        self.rpc(_info())
        siblings = {"unlabelled": "{", "list": "[]"}
        for label, inner in self.malformed_rows().items():
            siblings[label] = self.labelled("work", inner)
        for label, text in siblings.items():
            self.write_row(text, profile="work", raw=True)
            for task in (LONG, SHORT):
                code, doc = self.check(task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (0, "eligible", ["ok"]), (label, task))
            with self.assertRaises(health.StrictReadError, msg=label):
                health.strict_quota_cache_row("work")

    def test_a_well_typed_sibling_row_does_not_block(self):
        self.write_row({"checked_at": NOW.timestamp(), "used_percent": 4.0})
        self.write_row({"checked_at": 1, "used_percent": None, "resets_at": 5.5,
                        "window_mins": 300, "reached": False, "primary_reached": False,
                        "home_id": "abc", "plan": "pro",
                        "secondary": {"used_percent": 3, "resets_at": None,
                                      "window_mins": 10080, "reached": False}},
                       profile="work")
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)

    def test_a_valid_quota_cache_does_not_block(self):
        self.write_row({"checked_at": NOW.timestamp(), "used_percent": 4.0})
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)

    def test_unmonitored_and_anthropic_pools_do_not_read_the_quota_cache(self):
        self.write_row("{", raw=True)
        self.assertEqual(self.check(login="xai")[0], 0)
        self.assertEqual(self.check(login="anthropic", snapshot=self.snapshot())[0], 0)

    def test_the_strict_readers_leave_the_lenient_ones_alone(self):
        self.write("health.json", "{", raw=True)
        self.write_row("[]", raw=True)
        self.assertEqual(health._read_ledger_unlocked(), {})
        self.assertIsNone(health.get_quota_row("default"))
        with self.assertRaises(health.StrictReadError) as cm:
            health.strict_ledger_snapshot()
        self.assertEqual(cm.exception.reason, "ledger_unreadable")
        with self.assertRaises(health.StrictReadError) as cm:
            health.strict_quota_cache_row("default")
        self.assertEqual(cm.exception.reason, "telemetry_malformed")
        self.assertEqual(str(cm.exception), "telemetry_malformed")


class NeverEligibleOnUnreadableEvidenceTests(QuotaCase):
    """The falsifier: no path prints eligible while evidence is unreadable."""

    def test_sweep(self):
        scenarios = []
        for text in ("{", "[]", '{"codex@default": 5}', '{"codex@default": {"until": "x"}}'):
            scenarios.append(("ledger", "health.json", text, "openai", None))
            scenarios.append(("ledger-unmon", "health.json", text, "xai", None))
        for text in ("{", "[]", '{"profile": "default", "checked_at": "x"}'):
            scenarios.append(("cache", self.row_name(), text, "openai", None))
        for text in ("{", "[]", "{}", '{"generated_at": "x", "windows": []}'):
            scenarios.append(("snapshot", "snap.json", text, "anthropic", "snap.json"))
        for label, name, text, login, snap in scenarios:
            for task in (LONG, SHORT):
                for context in ("primary", "fallback"):
                    self.rpc(_info())
                    self.write(name, text, raw=True)
                    path = os.path.join(self.state, snap) if snap else None
                    code, doc = self.check(login=login, task=task, context=context,
                                           snapshot=path)
                    self.assertNotEqual(doc["decision"], "eligible", (label, text))
                    self.assertEqual(code, 10, (label, text, task, context))
                    os.remove(os.path.join(self.state, name))


class OutputTests(QuotaCase):
    def test_document_has_exactly_the_allowlisted_fields_and_values(self):
        self.rpc(_info())
        _code, doc = self.check()
        self.assertEqual(tuple(doc), quota.DOC_KEYS)
        self.assertEqual(doc["interface_version"], "quota-decision/v1")
        self.assertEqual(str(uuid.UUID(doc["decision_id"], version=4)), doc["decision_id"])
        self.assertEqual(doc["policy_revision"], quota.load_policy()[1])
        quota.validate_document(doc)

    def test_every_decision_gets_its_own_id(self):
        ids = {self.check(login="xai")[1]["decision_id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_validate_document_rejects_values_outside_the_allowlist(self):
        self.rpc(_info())
        _code, doc = self.check()

        def mutated(**changes):
            copy = dict(doc)
            copy.update(changes)
            return copy

        for changes in (
                {"decision": "maybe"}, {"reasons": ["something_else"]}, {"reasons": []},
                {"reasons": ["ok", "ok"]}, {"quota_pool": EMAIL}, {"login": EMAIL},
                {"account": "a b"}, {"used_bp": 10001}, {"used_bp": True},
                {"used_bp": 1.5}, {"window": EMAIL}, {"observed_at": "yesterday"},
                {"freshness_s": -1}, {"policy_revision": "abc"},
                {"checked_at": None}, {"expires_at": "2026-11-18 12:00:00"},
                {"decision_id": "not-a-uuid"}, {"interface_version": "v2"}):
            with self.assertRaises(quota._OutputInvalid, msg=str(changes)):
                quota.validate_document(mutated(**changes))
        extra = dict(doc)
        extra["email"] = EMAIL
        with self.assertRaises(quota._OutputInvalid):
            quota.validate_document(extra)
        short = dict(doc)
        short.pop("window")
        with self.assertRaises(quota._OutputInvalid):
            quota.validate_document(short)

    def test_invalid_policy_refuses_with_registry_invalid(self):
        with mock.patch.object(quota, "load_policy", side_effect=quota.PolicyError("x")):
            code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["registry_invalid"]))
        self.assertIsNone(doc["policy_revision"])
        self.assertIsNone(doc["quota_pool"])

    def test_an_unexpected_failure_refuses(self):
        with mock.patch.object(quota, "_pool_for_login", side_effect=RuntimeError(EMAIL)):
            code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["registry_invalid"]))

    def test_omitted_account_means_default(self):
        _code, omitted = self.check(login="xai")
        _code, explicit = self.check(login="xai", account="default")
        self.assertEqual(omitted["account"], "default")
        self.assertEqual(explicit["account"], "default")

    def test_exit_codes(self):
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(self.check(login="mistral")[0], 10)
        self.assertEqual(self.run_cli(["check"])[0], 2)


class ContractKeyTests(unittest.TestCase):
    def test_contract_advertises_only_the_static_capability(self):
        env = dict(os.environ)
        env.pop("OMNI_LLM_STATE_DIR", None)
        env["OMNI_LLM_NOTIFY_QUIET"] = "1"
        cli = os.path.join(ROOT, "omni_cli.py")
        proc = subprocess.run([sys.executable, cli, "contract"], capture_output=True,
                              text=True, env=env, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(doc["quota"], "quota-decision/v1")
        revision = quota.load_policy()[1]
        for needle in (revision, "openai-sub", "reserve_threshold_bp", "9000"):
            self.assertNotIn(needle, proc.stdout)


class UsageTests(QuotaCase):
    def test_login_and_account_are_validated_at_parse_time_without_echo(self):
        for flag in ("--login", "--account"):
            for value in (EMAIL, "a b", "a/b", "", "x" * 65, "lé", "a\nb", "a.b"):
                argv = _argv(account="work") if flag == "--account" else _argv()
                argv[argv.index(flag) + 1] = value
                code, doc, out, err = self.run_cli(argv)
                self.assertEqual(code, 2, (flag, value))
                self.assertEqual(out, "")
                self.assertIn("usage error", err)
                self.assertNotIn(value or "\0", err)
        self.assertEqual(self.audit_lines(), [])

    def test_other_bad_values_are_refused_without_echo(self):
        cases = (
            ("--task-class", "forever"), ("--context", EMAIL), ("--model", EMAIL),
            ("--transport", "codex:" + EMAIL), ("--model", "a b"),
            ("--anthropic-snapshot", ""), ("--anthropic-snapshot", "x\0y"))
        for flag, value in cases:
            argv = _argv(snapshot="/tmp/x.json")
            argv[argv.index(flag) + 1] = value
            code, _doc, out, err = self.run_cli(argv)
            self.assertEqual((code, out), (2, ""), (flag, value))
            self.assertNotIn("example.invalid", err)
            if value:
                self.assertNotIn(value, err)

    def test_unknown_options_and_missing_options_are_usage_errors(self):
        for argv in (_argv() + ["--surprise=" + EMAIL], _argv() + [EMAIL],
                     ["check", "--login", "openai"], [], ["bogus"],
                     [a for a in _argv() if a != "--json"]):
            code, _doc, out, err = self.run_cli(argv)
            self.assertEqual((code, out), (2, ""), argv)
            self.assertNotIn("example.invalid", err)

    def test_abbreviated_options_are_not_accepted(self):
        argv = [("--acc" if a == "--account" else a) for a in _argv(account="work")]
        self.assertEqual(self.run_cli(argv)[0], 2)

    def test_check_reads_nothing_when_arguments_are_invalid(self):
        self.rpc(_info())
        self.run_cli(_argv(login=EMAIL))
        self.assertEqual(self.rpc_calls, [])


class AuditTests(QuotaCase):
    def test_one_allowlisted_line_per_decision_equal_to_the_output(self):
        self.rpc(_info())
        code1, doc1, out1, _ = self.run_cli(_argv())
        code2, doc2, out2, _ = self.run_cli(_argv(login="xai", context="fallback"))
        self.assertEqual((code1, code2), (0, 10))
        lines = self.audit_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], [doc1, doc2])
        self.assertEqual(lines[0] + "\n", out1)
        for line in lines:
            self.assertEqual(tuple(json.loads(line)), quota.DOC_KEYS)

    def test_the_audit_log_is_append_only_and_private(self):
        self.run_cli(_argv(login="xai"))
        first = self.audit_lines()
        self.run_cli(_argv(login="xai"))
        second = self.audit_lines()
        self.assertEqual(second[:1], first)
        self.assertEqual(len(second), 2)
        mode = os.stat(os.path.join(self.state, quota.AUDIT_FILE)).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_rotation_keeps_one_previous_generation(self):
        with mock.patch.object(quota, "AUDIT_ROTATE_BYTES", 300):
            for _ in range(6):
                self.run_cli(_argv(login="xai"))
        names = sorted(n for n in os.listdir(self.state) if n.startswith("quota-audit"))
        self.assertEqual(names, ["quota-audit.jsonl", "quota-audit.jsonl.1"])
        for name in names:
            with open(os.path.join(self.state, name), encoding="utf-8") as handle:
                for line in handle:
                    json.loads(line)

    def test_usage_errors_write_no_audit_line(self):
        self.run_cli(_argv(login=EMAIL))
        self.assertEqual(self.audit_lines(), [])

    def test_a_non_allowlisted_document_is_never_written(self):
        quota.append_audit({"login": EMAIL, "decision": "eligible"})
        quota.append_audit({key: None for key in quota.DOC_KEYS})
        self.assertEqual(self.audit_lines(), [])

    def test_an_audit_failure_does_not_change_the_decision(self):
        os.mkdir(os.path.join(self.state, quota.AUDIT_FILE))
        code, doc, out, _err = self.run_cli(_argv(login="xai"))
        self.assertEqual((code, doc["decision"]), (0, "eligible"))


class PiiTests(QuotaCase):
    """No e-mail address and no home path in stdout, stderr or the audit file."""

    def setUp(self):
        super().setUp()
        self.home = os.path.expanduser("~")

    def run_all(self, argv):
        code, doc, out, err = self.run_cli(argv)
        audit = ""
        path = os.path.join(self.state, quota.AUDIT_FILE)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                audit = handle.read()
        return code, out, err, audit

    def assert_clean(self, label, *streams):
        for name, text in zip(("stdout", "stderr", "audit"), streams):
            for needle in (EMAIL, "example.invalid", "noreply@", self.state,
                           self.tmp.name):
                self.assertNotIn(needle, text, (label, name, needle))
            if self.home not in ("/", ""):
                self.assertNotIn(self.home, text, (label, name, "home"))

    def test_email_shaped_values_never_reach_any_output(self):
        identity = {"email": EMAIL, "emailAddress": EMAIL, "accountId": EMAIL,
                    "account": EMAIL}
        email_dir = os.path.join(self.state, EMAIL)
        os.makedirs(email_dir)
        scenarios = []

        def scenario(label, argv, setup=None):
            scenarios.append((label, argv, setup))

        # success paths with identity-bearing fixtures
        scenario("codex ok", _argv(),
                 lambda: self.rpc(_info(**identity)))
        scenario("codex err", _argv(),
                 lambda: self.rpc(None, {"ok": False, "reason": "auth_error",
                                         "detail": EMAIL, "identity": identity}))
        scenario("codex raises", _argv(), lambda: self.rpc(exc=RuntimeError(EMAIL)))
        scenario("codex secondary identity", _argv(),
                 lambda: self.rpc(_info(secondary=3.0, plan=EMAIL)))
        # (a snapshot naming another account is refused, see "anthropic account
        # identity"; the two below carry the other identity keys only)
        snap_identity = {k: v for k, v in identity.items() if k != "account"}
        scenario("anthropic ok", _argv(login="anthropic", snapshot=os.path.join(
                 self.state, "snap.json")),
                 lambda: self.snapshot_to("snap.json", extra=snap_identity))
        scenario("anthropic account identity", _argv(login="anthropic",
                 snapshot=os.path.join(self.state, "snap.json")),
                 lambda: self.snapshot_to("snap.json", extra=identity))
        scenario("anthropic window kind", _argv(login="anthropic", snapshot=os.path.join(
                 self.state, "snap.json")),
                 lambda: self.snapshot_to("snap.json", kind=EMAIL))
        scenario("anthropic malformed", _argv(login="anthropic", snapshot=os.path.join(
                 self.state, "snap.json")),
                 lambda: self.write("snap.json", '{"email": "%s"' % EMAIL, raw=True))
        scenario("anthropic stale", _argv(login="anthropic", snapshot=os.path.join(
                 self.state, "snap.json")),
                 lambda: self.snapshot_to("snap.json", age=99999, extra=snap_identity))
        scenario("anthropic path with email", _argv(login="anthropic",
                 snapshot=os.path.join(email_dir, "snap.json")))
        scenario("anthropic unknown account", _argv(login="anthropic", account="work"))
        scenario("unmonitored", _argv(login="xai"))
        scenario("unmonitored fallback", _argv(login="xai", context="fallback"))
        scenario("unknown login", _argv(login="mistral"))
        # ledger and cache with identity-bearing rows
        row = {"model": "codex", "profile": "default", "until": NOW.timestamp() + 600,
               "source": "quota", "account": EMAIL, "identity": identity}
        scenario("ledger cooling with identity", _argv(login="xai"),
                 lambda: self.write("health.json", {"grok@default": dict(row, model="grok")}))
        scenario("ledger malformed", _argv(login="xai"),
                 lambda: self.write("health.json", '{"a": "%s"' % EMAIL, raw=True))
        scenario("ledger bad row", _argv(login="xai"),
                 lambda: self.write("health.json", {"k": EMAIL}))
        scenario("ledger bad expiry", _argv(login="xai"),
                 lambda: self.write("health.json", {"k": {"until": EMAIL}}))
        scenario("cache malformed", _argv(),
                 lambda: self.write_row(
                     '{"profile": "default", "checked_at": "%s"}' % EMAIL, raw=True))
        scenario("cache identity row", _argv(),
                 lambda: (self.write_row(dict(
                     checked_at=NOW.timestamp(), used_percent=1.0, **identity)),
                     self.rpc(_info())))
        # internal failures
        scenario("policy invalid", _argv(),
                 lambda: self.patch(quota, "load_policy", mock.Mock(
                     side_effect=quota.PolicyError(EMAIL))))
        scenario("unexpected failure", _argv(),
                 lambda: self.patch(quota, "_pool_for_login", mock.Mock(
                     side_effect=RuntimeError(EMAIL))))
        scenario("cooling check failure", _argv(login="xai"),
                 lambda: self.patch(quota, "_ledger_cooling", mock.Mock(
                     side_effect=RuntimeError(EMAIL))))
        # usage errors
        scenario("usage login", _argv(login=EMAIL))
        scenario("usage account", _argv(account=EMAIL))
        scenario("usage model", _argv(model=EMAIL))
        scenario("usage transport", _argv(transport=EMAIL))
        scenario("usage task", _argv(task=EMAIL))
        scenario("usage context", _argv(context=EMAIL))
        scenario("usage unknown flag", _argv() + ["--" + EMAIL])
        scenario("usage stray", _argv() + [EMAIL])
        scenario("usage home path model", _argv(model=self.home + "/x"))
        scenario("usage snapshot", _argv(login="anthropic", snapshot=""))
        scenario("policy show", ["policy", "show", "--json"])
        scenario("policy show bad", ["policy", "show", "--json", EMAIL])

        # label -> (exit code, first reason or None for a usage error, RPC ran)
        expected = {
            "codex ok": (0, "ok", True),
            "codex err": (10, "telemetry_unavailable", True),
            "codex raises": (10, "telemetry_unavailable", True),
            "codex secondary identity": (0, "ok", True),
            "anthropic ok": (0, "ok", False),
            "anthropic account identity": (10, "malformed_telemetry", False),
            "anthropic window kind": (0, "ok", False),
            "anthropic malformed": (10, "malformed_telemetry", False),
            "anthropic stale": (10, "stale_telemetry", False),
            "anthropic path with email": (10, "missing_telemetry", False),
            "anthropic unknown account": (10, "unknown_account", False),
            "unmonitored": (0, "unmonitored_pool", False),
            "unmonitored fallback": (10, "unmonitored_pool", False),
            "unknown login": (10, "unknown_login", False),
            "ledger cooling with identity": (10, "ledger_cooling", False),
            "ledger malformed": (10, "ledger_unreadable", False),
            "ledger bad row": (10, "ledger_unreadable", False),
            "ledger bad expiry": (10, "ledger_unreadable", False),
            # a malformed cache file refuses before any RPC; a row with no
            # home binding goes to the (replaced) RPC
            "cache malformed": (10, "telemetry_malformed", False),
            "cache identity row": (0, "ok", True),
            "policy invalid": (10, "registry_invalid", False),
            "unexpected failure": (10, "registry_invalid", False),
            "cooling check failure": (10, "ledger_unreadable", False),
            "usage login": (2, None, False),
            "usage account": (2, None, False),
            "usage model": (2, None, False),
            "usage transport": (2, None, False),
            "usage task": (2, None, False),
            "usage context": (2, None, False),
            "usage unknown flag": (2, None, False),
            "usage stray": (2, None, False),
            "usage home path model": (10, "telemetry_unavailable", False),
            "usage snapshot": (2, None, False),
            "policy show": (0, None, False),
            "policy show bad": (2, None, False),
        }
        for label, argv, setup in scenarios:
            self.reset_state()
            if setup is not None:
                setup()
            code, out, err, audit = self.run_all(argv)
            self.assert_clean(label, out, err, audit)
            self.assertIn(code, (0, 2, 10), label)
            # Each scenario must have reached the path it is named for: its exit
            # code, its first reason and whether the RPC ran are all pinned.
            self.assertIn(label, expected, label)
            want_code, want_reason, want_rpc = expected[label]
            self.assertEqual(code, want_code, label)
            if want_reason is not None:
                self.assertEqual(json.loads(out)["reasons"][0], want_reason, label)
            self.assertEqual(bool(self.rpc_calls), want_rpc, label)
        self.assertEqual(sorted(expected), sorted(label for label, _a, _s in scenarios))
        self.assertGreater(len(scenarios), 30)

    def test_a_subprocess_run_is_clean_too(self):
        env = dict(os.environ)
        env.update({"OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "0",
                    "OMNI_LLM_NOTIFY_QUIET": "1"})
        code = ("import sys; from providers import quota; "
                "sys.exit(quota.main(sys.argv[1:]))")
        for argv in (_argv(), _argv(login=EMAIL), _argv(login="xai")):
            proc = subprocess.run([sys.executable, "-c", code] + argv, cwd=ROOT,
                                  env=env, capture_output=True, text=True, timeout=60)
            audit = "\n".join(self.audit_lines())
            self.assert_clean(" ".join(argv[:3]), proc.stdout, proc.stderr, audit)
        self.assertEqual(
            subprocess.run([sys.executable, "-c", code] + _argv(login=EMAIL), cwd=ROOT,
                           env=env, capture_output=True, text=True).returncode, 2)

    def test_the_omni_cli_routes_quota_to_the_decision_layer(self):
        env = dict(os.environ)
        env.update({"OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "0",
                    "OMNI_LLM_NOTIFY_QUIET": "1"})
        cli = os.path.join(ROOT, "omni_cli.py")
        proc = subprocess.run([sys.executable, cli, "quota"] + _argv(), cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 10, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(doc["interface_version"], "quota-decision/v1")
        self.assertEqual(doc["reasons"], ["telemetry_unavailable"])
        bad = subprocess.run([sys.executable, cli, "quota"] + _argv(login=EMAIL), cwd=ROOT,
                             env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(bad.returncode, 2)
        self.assert_clean("cli", bad.stdout, bad.stderr, "\n".join(self.audit_lines()))

    def test_contract_style_text_carries_no_state_dir(self):
        self.rpc(_info())
        _code, out, _err, audit = self.run_all(_argv())
        self.assertNotIn(self.state, out + audit)

    # -- helpers ----------------------------------------------------------
    def reset_state(self):
        """Back to an empty state dir (the e-mail named directory stays) and no
        patch of the previous scenario."""
        for patcher in reversed(self.scoped_patchers):
            patcher.stop()
        self.scoped_patchers.clear()
        for name in os.listdir(self.state):
            path = os.path.join(self.state, name)
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif name == "quota" and os.path.isdir(path):
                shutil.rmtree(path)
        self.assertFalse(os.path.lexists(os.path.join(self.state, "quota")))
        self.rpc_calls = []

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scoped_patchers.append(patcher)

    def snapshot_to(self, name, extra=None, kind="five_hour", age=60):
        doc = {"generated_at": (NOW - timedelta(seconds=age)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "windows": [{"kind": kind, "used_percent": 20}]}
        doc.update(extra or {})
        self.write(name, doc)


class SymlinkAndFlagRowTests(QuotaCase):
    """A symbolic link and a flag without an expiry are unreadable evidence."""

    def link(self, name, target):
        os.symlink(target, os.path.join(self.state, name))

    def test_a_dangling_ledger_symlink_refuses(self):
        self.link("health.json", os.path.join(self.tmp.name, "nowhere.json"))
        self.rpc(_info())
        for task in (LONG, SHORT):
            code, doc = self.check(task=task)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["ledger_unreadable"]), task)
        self.assertEqual(self.rpc_calls, [])
        code, doc = self.check(login="xai")
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_unreadable"]))

    def test_a_ledger_symlink_to_a_healthy_file_refuses(self):
        target = os.path.join(self.tmp.name, "real-health.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("{}")
        self.link("health.json", target)
        self.rpc(_info())
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_unreadable"]))

    def link_row(self, target):
        os.makedirs(os.path.join(self.state, "quota"), exist_ok=True)
        self.link(self.row_name(), target)

    def test_a_dangling_cache_symlink_refuses_without_the_rpc(self):
        self.link_row(os.path.join(self.tmp.name, "nowhere.json"))
        self.rpc(_info())
        for task in (LONG, SHORT):
            code, doc = self.check(task=task)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["telemetry_malformed"]), task)
        self.assertEqual(self.rpc_calls, [])

    def test_a_cache_symlink_to_a_healthy_file_refuses(self):
        target = os.path.join(self.tmp.name, "real-quota.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"profile": "default", "checked_at": NOW.timestamp(),
                       "used_percent": 4.0}, handle)
        self.link_row(target)
        self.rpc(_info())
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["telemetry_malformed"]))

    def test_a_cache_directory_symlink_to_healthy_files_refuses(self):
        real = os.path.join(self.tmp.name, "real-quota-dir")
        os.makedirs(real)
        with open(os.path.join(real, os.path.basename(self.row_name())), "w",
                  encoding="utf-8") as handle:
            json.dump({"profile": "default", "checked_at": NOW.timestamp(),
                       "used_percent": 4.0}, handle)
        self.link("quota", real)
        self.rpc(_info())
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["telemetry_malformed"]))

    def test_a_fifo_is_not_a_readable_ledger(self):
        path = os.path.join(self.state, "health.json")
        os.mkfifo(path)
        reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        writer = os.open(path, os.O_WRONLY)
        self.addCleanup(os.close, reader)
        self.addCleanup(os.close, writer)
        os.write(writer, b"{}")   # a reader that follows the FIFO would take this
        self.rpc(_info())

        def blocked(_signum, _frame):
            raise AssertionError("the strict reader blocked on a FIFO")
        previous = signal.signal(signal.SIGALRM, blocked)
        signal.alarm(5)
        try:
            code, doc = self.check()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        self.assertEqual((code, doc["reasons"]), (10, ["ledger_unreadable"]))

    def test_an_absent_ledger_and_cache_are_still_empty(self):
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(health.strict_ledger_snapshot(), {})
        self.assertIsNone(health.strict_quota_cache_row("default"))

    def test_a_flag_row_without_a_usable_expiry_is_unreadable(self):
        flagged = {"model": "codex", "profile": "default", "source": "manual",
                   "flagged_at": NOW.timestamp()}
        bad = {
            "until null": dict(flagged, until=None),
            "bare until null": {"until": None},
            "until null on a model row": {"model": "codex", "profile": "default",
                                          "until": None},
            "until missing on a flag": flagged,
            "until empty": dict(flagged, until=""),
            "probe_after only": {"probe_after": NOW.timestamp() + 5},
            "credit hold until null": dict(flagged, until=NOW.timestamp() + 5,
                                           credit_hold={"until": None}),
            "credit hold null": dict(flagged, until=NOW.timestamp() + 5,
                                     credit_hold=None),
            "credit hold list": dict(flagged, until=NOW.timestamp() + 5,
                                     credit_hold=[]),
        }
        for label, row in bad.items():
            self.write("health.json", {"codex@default": row})
            self.rpc(_info())
            for task in (LONG, SHORT):
                code, doc = self.check(task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (10, "ineligible", ["ledger_unreadable"]), (label, task))

    def test_streak_only_rows_stay_valid(self):
        self.write("health.json", {"codex@default": {
            "crash_streak": 1, "last_crash_at": NOW.timestamp(), "model": "codex",
            "profile": "default", "account": None}})
        self.rpc(_info())
        self.assertEqual(self.check()[0], 0)

    def test_nested_duplicate_keys_and_non_finite_numbers_are_unreadable(self):
        for label, name, text, reason in (
                ("ledger nested duplicate", "health.json",
                 '{"codex@default": {"until": 1, "until": 99999999999}}', "ledger_unreadable"),
                ("ledger unvalidated nan", "health.json",
                 '{"codex@default": {"crash_streak": NaN}}', "ledger_unreadable"),
                ("ledger infinity", "health.json",
                 '{"codex@default": {"x": Infinity}}', "ledger_unreadable"),
                ("cache nested duplicate", self.row_name(),
                 '{"profile": "default", "checked_at": 1, "checked_at": 2}',
                 "telemetry_malformed"),
                ("cache unvalidated nan", self.row_name(),
                 '{"profile": "default", "checked_at": 1, "plan": NaN}',
                 "telemetry_malformed"),
                ("cache infinity", self.row_name(),
                 '{"profile": "default", "checked_at": 1, "x": -Infinity}',
                 "telemetry_malformed")):
            self.write(name, text, raw=True)
            self.rpc(_info())
            code, doc = self.check()
            self.assertEqual((code, doc["reasons"]), (10, [reason]), label)
            os.remove(os.path.join(self.state, name))


class StrictDecoderTests(QuotaCase):
    """One decoder serves the policy, the snapshot, the ledger and the quota cache."""

    BAD = (("duplicate key", '"x": 1, "x": 2'),
           ("NaN", '"x": NaN'),
           ("Infinity", '"x": Infinity'),
           ("-Infinity", '"x": -Infinity'),
           ("1e999", '"x": 1e999'),
           ("-1e999", '"x": -1e999'),
           ("nested 1e999", '"x": {"y": [1e999]}'))

    def test_the_decoder_refuses_each_bad_input_and_keeps_good_numbers(self):
        for label, member in self.BAD:
            with self.assertRaises(ValueError, msg=label):
                health.strict_json_loads("{%s}" % member)
        self.assertEqual(health.strict_json_loads('{"a": [1, 2.5, 1e5, -0.0]}'),
                         {"a": [1, 2.5, 100000.0, -0.0]})

    def test_every_source_refuses_each_bad_input(self):
        policy_text = open(quota.POLICY_FILE, encoding="utf-8").read()
        self.rpc(_info())
        for label, member in self.BAD:
            # policy: an extra member in the shipped document
            doc = policy_text.rstrip().rstrip("}") + ",\n  %s}" % member
            if label == "duplicate key":
                doc = policy_text.replace('"profile": "default",',
                                          '"profile": "default",\n"profile": "default",', 1)
            path = self.write("policy.json", doc, raw=True)
            with self.assertRaises(quota.PolicyError, msg=("policy", label)) as cm:
                quota.load_policy(path)
            self.assertEqual(str(cm.exception), "policy unreadable", ("policy", label))
            # snapshot
            snap = ('{"generated_at": "2026-11-18T11:59:00Z", "windows": '
                    '[{"kind": "a", "used_percent": 10}], %s}' % member)
            path = self.write("snap.json", snap, raw=True)
            for task in (LONG, SHORT):
                code, check = self.check(login="anthropic", snapshot=path, task=task)
                self.assertEqual((code, check["reasons"]),
                                 (10, ["malformed_telemetry"]), ("snapshot", label, task))
            # ledger and quota cache
            for name, text, reason in (
                    ("health.json", '{"codex@default": {%s}}' % member, "ledger_unreadable"),
                    (self.row_name(),
                     '{"profile": "default", "checked_at": 1, %s}' % member,
                     "telemetry_malformed")):
                self.write(name, text, raw=True)
                for task in (LONG, SHORT):
                    code, check = self.check(task=task)
                    self.assertEqual((code, check["reasons"]), (10, [reason]),
                                     (name, label, task))
                os.remove(os.path.join(self.state, name))

    def test_all_four_sources_go_through_the_one_helper(self):
        self.rpc(_info())
        failing = mock.patch.object(health, "strict_json_loads",
                                    side_effect=ValueError("refused"))
        snap = self.snapshot()
        self.write("health.json", {}, raw=False)
        # a valid row, so only the patched decoder can make the cache refuse
        self.write_row({"checked_at": NOW.timestamp(), "used_percent": 4.0})
        with failing:
            with self.assertRaises(quota.PolicyError):
                quota.load_policy()
            self.assertEqual(self.check(login="anthropic", snapshot=snap)[1]["reasons"],
                             ["registry_invalid"])
            self.assertEqual(self.check()[1]["reasons"], ["registry_invalid"])
            with self.assertRaises(health.StrictReadError) as cm:
                health.strict_ledger_snapshot()
            self.assertEqual(cm.exception.reason, "ledger_unreadable")
            with self.assertRaises(health.StrictReadError) as cm:
                health.strict_quota_cache_row("default")
            self.assertEqual(cm.exception.reason, "telemetry_malformed")
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 4.0)
        # the snapshot reader maps a decoder refusal to its own reason
        policy, revision = quota.load_policy()
        self.assertEqual(quota._read_snapshot_file(snap)[1], None)
        with failing:
            self.assertEqual(quota._read_snapshot_file(snap), (None, "malformed_telemetry"))


class AccountBindingTests(QuotaCase):
    """A snapshot is evidence for the account it names, and only that one."""

    def snap(self, path_name="snap.json", **extra):
        doc = {"generated_at": (NOW - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "windows": [{"kind": "five_hour", "used_percent": 12.0}]}
        doc.update(extra)
        return self.write(path_name, doc)

    def test_a_snapshot_for_another_account_is_not_evidence_for_default(self):
        path = self.snap(account="work")
        for task in (LONG, SHORT):
            for account in (None, "default"):
                code, doc = self.check(login="anthropic", account=account, task=task,
                                       snapshot=path)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (10, "ineligible", ["malformed_telemetry"]), (task, account))
                self.assertEqual((doc["used_bp"], doc["window"]), (None, None))

    def test_the_same_snapshot_through_a_symlink_is_refused_too(self):
        real = self.snap("real-work-snapshot.json", account="work")
        link = os.path.join(self.state, "default-snapshot.json")
        os.symlink(real, link)
        code, doc = self.check(login="anthropic", snapshot=link)
        self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]))

    def test_conflicting_or_unverifiable_bindings_are_refused(self):
        for label, value in (("null", None), ("number", 5), ("list", ["default"]),
                             ("empty", ""), ("case", "Default"), ("bool", True)):
            path = self.snap(account=value)
            code, doc = self.check(login="anthropic", snapshot=path)
            self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]), label)

    def test_the_named_default_account_and_no_account_field_are_accepted(self):
        self.assertEqual(self.check(login="anthropic", snapshot=self.snap(account="default"))[0], 0)
        self.assertEqual(self.check(login="anthropic", snapshot=self.snap())[0], 0)

    def test_an_unnamed_snapshot_is_only_for_default(self):
        with open(self.snap(), encoding="utf-8") as handle:
            data = json.load(handle)
        reason = quota._evaluate_anthropic(data, None, quota.load_policy()[0], 9000,
                                           NOW, "work").reasons
        self.assertEqual(reason, [("malformed_telemetry", False)])
        reason = quota._evaluate_anthropic(data, None, quota.load_policy()[0], 9000,
                                           NOW, "default").reasons
        self.assertEqual(reason, [])

    def test_an_unknown_account_is_still_refused_before_any_read(self):
        code, doc = self.check(login="anthropic", account="work", snapshot=self.snap(account="work"))
        self.assertEqual((code, doc["reasons"], doc["observed_at"]),
                         (10, ["unknown_account"], None))


class CodexCacheBindingTests(QuotaCase):
    """The strict cache read is the only read: what it validated is what decides."""

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp.name, "codex-home")
        os.makedirs(self.home)
        self.digest = hashlib.sha256(os.path.realpath(self.home).encode("utf-8")).hexdigest()
        env = mock.patch.dict(os.environ, {"CODEX_HOME": self.home,
                                           "OMNI_LLM_QUOTA_PROBE": "1"})
        env.start()
        self.addCleanup(env.stop)
        spawn = mock.patch.object(codex_quota, "_spawn_rate_limits",
                                  side_effect=AssertionError("spawned"))
        spawn.start()
        self.addCleanup(spawn.stop)

    def row(self, age=20, home_id=None, **over):
        row = {"used_percent": 33.0, "resets_at": None, "window_mins": 300,
               "reached": False, "primary_reached": False, "plan": "pro",
               "checked_at": NOW.timestamp() - age, "home_id": home_id or self.digest}
        row.update(over)
        return row

    def test_a_valid_bound_row_decides_without_any_second_read(self):
        self.write_row(self.row())
        with mock.patch.object(codex_quota, "read_rate_limits",
                               side_effect=AssertionError("rpc")), \
                mock.patch.object(health, "get_quota_row",
                                  side_effect=AssertionError("lenient reread")):
            code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["used_bp"]), (0, "eligible", 3300))

    def test_an_atomic_replacement_after_the_strict_read_changes_nothing(self):
        self.write_row(self.row())
        real = health.strict_quota_cache_row

        def swap_after_read(profile):
            data = real(profile)
            # duplicate percentages 99 then 1: a lenient reader keeps the 1
            self.write_row('{"profile": "default", "checked_at": %s, "home_id": "%s", '
                           '"reached": false, "used_percent": 99.0, "used_percent": 1.0}'
                           % (NOW.timestamp() - 5, self.digest), raw=True)
            return data
        with mock.patch.object(health, "strict_quota_cache_row", swap_after_read), \
                mock.patch.object(codex_quota, "read_rate_limits",
                                  side_effect=AssertionError("second read")):
            code, doc = self.check()
        self.assertEqual((code, doc["decision"], doc["used_bp"]), (0, "eligible", 3300))

    def test_a_refresh_bypasses_the_cache_and_judges_the_reply(self):
        self.write_row(self.row(age=61))
        with mock.patch.object(codex_quota, "read_rate_limits",
                               return_value=(_info(primary=20.0), None)) as rpc:
            code, doc = self.check()
        self.assertEqual((code, doc["used_bp"]), (0, 2000))
        self.assertEqual(rpc.call_args.kwargs.get("force"), True)

    def test_a_malformed_matching_row_refuses_for_long_and_short_without_a_refresh(self):
        STRICT_CACHE_LABELS = ("percent string", "reached list", "reached string",
                               "primary_reached int", "resets_at string", "resets_at bool",
                               "window_mins string", "secondary reached list",
                               "secondary resets_at", "plan list", "quota_unknown list")
        bad = {
            "reached list": self.row(reached=[]),
            "reached string": self.row(reached="no"),
            "reached missing": {k: v for k, v in self.row().items() if k != "reached"},
            "primary_reached int": self.row(primary_reached=0),
            "resets_at string": self.row(resets_at="bad"),
            "resets_at bool": self.row(resets_at=True),
            "window_mins string": self.row(window_mins="300"),
            "plan list": self.row(plan=["pro"]),
            "quota_unknown": self.row(quota_unknown=True),
            "quota_unknown list": self.row(quota_unknown=[]),
            "percent string": self.row(used_percent="33"),
            "primary missing, secondary healthy": self.row(
                used_percent=None, secondary={"used_percent": 5.0, "reached": False,
                                              "resets_at": None, "window_mins": 10080}),
            "secondary reached list": self.row(secondary={
                "used_percent": 5.0, "reached": [], "resets_at": None, "window_mins": 10080}),
            "secondary resets_at": self.row(secondary={
                "used_percent": 5.0, "reached": False, "resets_at": "x", "window_mins": 1}),
            "secondary percent missing": self.row(secondary={
                "reached": False, "resets_at": None, "window_mins": 10080}),
        }
        for label, row in bad.items():
            self.write_row(row)
            for task in (LONG, SHORT):
                with mock.patch.object(codex_quota, "read_rate_limits",
                                       side_effect=AssertionError("refresh")):
                    code, doc = self.check(task=task)
                self.assertEqual((code, doc["decision"]), (10, "ineligible"), (label, task))
                # the strict cache read catches a mistyped field itself
                reason = ("telemetry_malformed" if label in STRICT_CACHE_LABELS
                          else "malformed_telemetry")
                self.assertEqual(doc["reasons"], [reason], (label, task))

    def test_stale_future_and_foreign_home_rows_refresh_and_a_failed_refresh_refuses(self):
        rows = {
            "stale": self.row(age=61),
            "future": self.row(age=-6),
            "other home": self.row(home_id="0" * 64),
            "no home": {k: v for k, v in self.row().items() if k != "home_id"},
        }
        for label, row in rows.items():
            self.write_row(row)
            for task in (LONG, SHORT):
                for err in ({"ok": False, "reason": "timeout"}, {"ok": False}):
                    with mock.patch.object(codex_quota, "read_rate_limits",
                                           return_value=(None, err)) as rpc:
                        code, doc = self.check(task=task)
                    self.assertEqual((code, doc["decision"], doc["reasons"]),
                                     (10, "ineligible", ["telemetry_unavailable"]),
                                     (label, task))
                    self.assertEqual(rpc.call_count, 1, label)

    def test_the_rpc_normalizer_cannot_drop_a_malformed_primary(self):
        reply = codex_quota.parse_rate_limits({"rateLimits": {
            "primary": {"usedPercent": "not a number", "windowDurationMins": 300},
            "secondary": {"usedPercent": 5.0, "windowDurationMins": 10080}}})
        reply["checked_at"] = NOW.timestamp() - 5
        self.assertIsNone(reply["used_percent"])
        for task in (LONG, SHORT):
            self.rpc(reply)
            code, doc = self.check(task=task)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["malformed_telemetry"]), task)

    def test_a_primary_that_is_absent_entirely_does_not_vouch_for_a_secondary(self):
        info = _info(primary=None, secondary=5.0)
        self.rpc(info)
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["malformed_telemetry"]))

    def test_a_probe_that_is_disabled_never_uses_the_cache(self):
        self.write_row(self.row())
        with mock.patch.dict(os.environ, {"OMNI_LLM_QUOTA_PROBE": "0"}):
            code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (10, ["telemetry_unavailable"]))


class LedgerMatchTests(QuotaCase):
    """A flag cools a quota pool only for its own road, account and window."""

    def put(self, key, **extra):
        row = {"model": key.split("@")[0], "profile": "default",
               "until": NOW.timestamp() + 3600, "source": "quota",
               "flagged_at": NOW.timestamp(), "account": None}
        row.update(extra)
        path = os.path.join(self.state, "health.json")
        ledger = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        ledger[key] = row
        self.write("health.json", ledger)

    def anthropic(self, **kwargs):
        kwargs.setdefault("model", "claude-fable-5-2")
        return self.check(login="anthropic", snapshot=self.snapshot(), **kwargs)

    def test_a_codex_flag_does_not_cool_an_anthropic_check_with_a_codex_prefixed_model(self):
        for key in ("codex@default", "codex#week@default", "codex/gpt-6.2-sol@default"):
            self.put(key)
        for model in ("codex/gpt-6.2-sol", "codex/sol", "codex", "codex:x"):
            for task in (LONG, SHORT):
                code, doc = self.anthropic(model=model, task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (0, "eligible", ["ok"]), (model, task))

    def test_the_billed_roads_account_and_window_flags_do_cool(self):
        for key in ("claude@default", "claude#week@default", "claude#m300@default"):
            path = os.path.join(self.state, "health.json")
            if os.path.exists(path):
                os.remove(path)
            self.put(key)
            for model in ("claude-fable-5-2", "claude/claude-fable-5-2", "codex/gpt-6.2-sol"):
                code, doc = self.anthropic(model=model)
                self.assertEqual((code, doc["reasons"]), (10, ["ledger_cooling"]), (key, model))

    def test_a_models_own_flag_cools_only_a_model_of_this_road(self):
        self.put("claude-fable-5-2@default")
        self.assertEqual(self.anthropic(model="claude-fable-5-2")[1]["reasons"],
                         ["ledger_cooling"])
        self.assertEqual(self.anthropic(model="claude-other")[1]["reasons"], ["ok"])
        self.assertEqual(self.anthropic(model="codex/claude-fable-5-2")[1]["reasons"], ["ok"])

    def test_another_roads_window_or_other_profiles_flag_does_not_cool(self):
        self.put("grok#week@default")
        self.put("claude@other")
        self.put("claude#week@other")
        self.put("cursor:auto@default")
        self.assertEqual(self.anthropic()[1]["reasons"], ["ok"])
        self.rpc(_info())
        self.assertEqual(self.check(model="cursor:auto")[1]["reasons"], ["ok"])

    def test_manual_and_crash_holds_on_the_right_road_still_refuse_short_tasks(self):
        for source in ("manual", "crash"):
            self.put("claude@default", source=source)
            code, doc = self.anthropic(task=SHORT)
            self.assertEqual((code, doc["decision"], doc["reasons"]),
                             (10, "ineligible", ["ledger_cooling"]), source)

    def test_a_bare_model_flag_of_another_road_does_not_cool_this_road(self):
        for key, model in (("grok-4.6@default", "grok-4.6"),
                           ("gpt-6.2-sol@default", "gpt-6.2-sol")):
            path = os.path.join(self.state, "health.json")
            if os.path.exists(path):
                os.remove(path)
            self.put(key)
            for task in (LONG, SHORT):
                code, doc = self.anthropic(model=model, task=task)
                self.assertEqual((code, doc["decision"], doc["reasons"]),
                                 (0, "eligible", ["ok"]), (key, task))
            code, doc = self.check(login="cursor", model=model)
            self.assertEqual((code, doc["reasons"]), (0, ["unmonitored_pool"]), key)

    def test_a_flag_is_matched_in_pool_key_form(self):
        for key in ("CURSOR/auto@default", "cursor/auto@default", "cursor:auto@default"):
            path = os.path.join(self.state, "health.json")
            if os.path.exists(path):
                os.remove(path)
            self.put(key)
            for model in ("cursor:auto", "cursor/auto", "CURSOR/auto"):
                code, doc = self.check(login="cursor", model=model)
                self.assertEqual((code, doc["reasons"]), (10, ["ledger_cooling"]),
                                 (key, model))
        self.assertEqual(self.anthropic()[1]["reasons"], ["ok"])

    def test_equivalent_spellings_of_the_billed_model_honor_its_flag(self):
        # (flag key, login, model spellings that must all be cooled)
        cases = (
            ("grok-4.6@default", "xai", ("grok-4.6", "grok/grok-4.6", "GROK/grok-4.6")),
            ("grok/grok-4.6@default", "xai", ("grok-4.6", "grok/grok-4.6")),
            ("cursor/gpt-6.2@default", "cursor", ("gpt-6.2", "cursor/gpt-6.2")),
        )
        for key, login, models in cases:
            path = os.path.join(self.state, "health.json")
            if os.path.exists(path):
                os.remove(path)
            self.put(key)
            for model in models:
                for task in (LONG, SHORT):
                    code, doc = self.check(login=login, model=model, task=task)
                    self.assertEqual((code, doc["reasons"]), (10, ["ledger_cooling"]),
                                     (key, model, task))
            # The flag still never crosses to another road.
            self.assertEqual(self.anthropic(model=models[0])[1]["reasons"], ["ok"], key)


class ExpiryTests(QuotaCase):
    """Invalid or stale evidence is never reusable: it expires when it is made."""

    def assert_expires_now(self, doc, label):
        self.assertEqual(doc["expires_at"], doc["checked_at"], label)

    def test_a_future_snapshot_beyond_tolerance_expires_at_checked_at(self):
        for age in (-301, -3600):
            _code, doc = self.check(login="anthropic", snapshot=self.snapshot(age=age))
            self.assertEqual(doc["reasons"], ["stale_telemetry"])
            self.assert_expires_now(doc, age)

    def test_a_future_codex_observation_beyond_tolerance_expires_at_checked_at(self):
        self.rpc(_info(age=-6))
        _code, doc = self.check()
        self.assertEqual(doc["reasons"], ["stale_telemetry"])
        self.assert_expires_now(doc, "codex future")

    def test_every_invalid_stale_or_unavailable_result_expires_at_checked_at(self):
        cases = []
        cases.append(("stale snapshot", dict(login="anthropic",
                                             snapshot=self.snapshot(age=7201))))
        cases.append(("missing snapshot", dict(login="anthropic")))
        cases.append(("malformed snapshot", dict(
            login="anthropic", snapshot=self.write("bad.json", "{", raw=True))))
        for label, kwargs in cases:
            for task in (LONG, SHORT):
                _code, doc = self.check(task=task, **kwargs)
                self.assert_expires_now(doc, (label, task))
        self.rpc(None, {"ok": False, "reason": "timeout"})
        self.assert_expires_now(self.check()[1], "unavailable")
        self.rpc(_info(age=61))
        self.assert_expires_now(self.check()[1], "stale codex")
        self.write("health.json", "{", raw=True)
        self.assert_expires_now(self.check()[1], "ledger unreadable")
        os.remove(os.path.join(self.state, "health.json"))
        self.write_row("{", raw=True)
        self.assert_expires_now(self.check()[1], "cache malformed")
        with mock.patch.object(quota, "load_policy", side_effect=quota.PolicyError("x")):
            self.assert_expires_now(self.check()[1], "registry invalid")

    def test_valid_evidence_keeps_its_ttl(self):
        _code, doc = self.check(login="anthropic",
                                snapshot=self.snapshot(windows=[
                                    {"kind": "five_hour", "used_percent": 95}]))
        self.assertEqual(doc["reasons"], ["quota_reserve"])
        self.assertEqual(doc["expires_at"], "2026-11-18T12:05:00Z")
        self.assertEqual(self.check(login="xai")[1]["expires_at"], "2026-11-18T12:05:00Z")


class OutputFailureTests(QuotaCase):
    """A closed or failing stdout is a fixed message and a nonzero exit."""

    class Broken(object):
        def __init__(self, exc):
            self.exc = exc

        def write(self, _text):
            raise self.exc

        def flush(self):
            raise self.exc

    def test_a_broken_stdout_gives_a_fixed_message_and_a_nonzero_exit(self):
        for exc in (BrokenPipeError(32, "Broken pipe", self.state), OSError(5, "io", self.state),
                    ValueError("I/O operation on closed file")):
            err = io.StringIO()
            with mock.patch.object(sys, "stdout", self.Broken(exc)), \
                    contextlib.redirect_stderr(err):
                code = quota.main(_argv(login="xai"))
                policy = quota.main(["policy", "show", "--json"])
            self.assertEqual((code, policy), (quota.EXIT_OUTPUT, quota.EXIT_OUTPUT))
            self.assertEqual(err.getvalue(), "quota: cannot write the result\n" * 2)
            self.assertNotIn(self.state, err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_a_closed_pipe_prints_no_traceback_and_no_path(self):
        env = dict(os.environ)
        env.update({"OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "0",
                    "OMNI_LLM_NOTIFY_QUIET": "1"})
        code = ("import sys; from providers import quota; "
                "sys.exit(quota.main(sys.argv[1:]))")
        for argv in (_argv(login="xai"), ["policy", "show", "--json"]):
            reader, writer = os.pipe()
            os.close(reader)
            try:
                proc = subprocess.run([sys.executable, "-c", code] + argv, cwd=ROOT,
                                      env=env, stdout=writer, stderr=subprocess.PIPE,
                                      text=True, timeout=60)
            finally:
                os.close(writer)
            self.assertEqual(proc.returncode, quota.EXIT_OUTPUT, proc.stderr)
            self.assertEqual(proc.stderr, "quota: cannot write the result\n")
            for needle in ("Traceback", ROOT, self.state, os.path.expanduser("~")):
                self.assertNotIn(needle, proc.stderr)

    def test_a_closed_pipe_during_help_gives_the_fixed_message(self):
        env = dict(os.environ)
        env.update({"OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "0",
                    "OMNI_LLM_NOTIFY_QUIET": "1"})
        code = ("import sys; from providers import quota; "
                "sys.exit(quota.main(sys.argv[1:]))")
        cases = [(flags, argv) for flags in ([], ["-u"])
                 for argv in (["--help"], ["check", "--help"], ["policy", "show", "--help"])]
        for flags, argv in cases:
            reader, writer = os.pipe()
            os.close(reader)
            try:
                proc = subprocess.run([sys.executable] + flags + ["-c", code] + argv, cwd=ROOT,
                                      env=env, stdout=writer, stderr=subprocess.PIPE,
                                      text=True, timeout=60)
            finally:
                os.close(writer)
            self.assertEqual(proc.returncode, quota.EXIT_OUTPUT, (flags, argv, proc.stderr))
            self.assertEqual(proc.stderr, "quota: cannot write the result\n", (flags, argv))
            for needle in ("Traceback", ROOT, self.state, os.path.expanduser("~")):
                self.assertNotIn(needle, proc.stderr)

    def test_help_to_a_healthy_stdout_still_exits_zero(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = quota.main(["--help"])
        self.assertEqual((code, err.getvalue()), (0, ""))
        self.assertIn("usage", out.getvalue())

    def test_a_healthy_stdout_still_exits_with_the_decision(self):
        code, doc, _out, err = self.run_cli(_argv(login="xai"))
        self.assertEqual((code, doc["decision"], err), (0, "eligible", ""))


class SnapshotGuardTests(QuotaCase):
    """Each snapshot reader guard fails alone when it is removed."""

    GENERATED = "2026-11-18T11:59:00Z"

    def text(self, windows='[{"kind": "five_hour", "used_percent": 10}]', extra=""):
        return '{"generated_at": "%s", "windows": %s%s}' % (self.GENERATED, windows, extra)

    def verdict(self, text, task=LONG):
        path = self.write("snap.json", text, raw=True)
        code, doc = self.check(login="anthropic", snapshot=path, task=task)
        return code, doc["decision"], doc["reasons"]

    def test_the_plain_text_is_eligible(self):
        self.assertEqual(self.verdict(self.text()), (0, "eligible", ["ok"]))

    def test_a_duplicate_key_is_malformed_even_when_the_last_value_is_healthy(self):
        for label, text in (
                ("top level", '{"generated_at": "%s", "generated_at": "%s", "windows": '
                              '[{"kind": "a", "used_percent": 1}]}' % (self.GENERATED,
                                                                        self.GENERATED)),
                ("nested", self.text(windows='[{"kind": "a", "used_percent": 99, '
                                             '"used_percent": 1}]')),
                ("window kind", self.text(windows='[{"kind": "a", "kind": "b", '
                                                  '"used_percent": 1}]'))):
            json.loads(text)   # plain JSON keeps the last value, which is healthy
            for task in (LONG, SHORT):
                self.assertEqual(self.verdict(text, task),
                                 (10, "ineligible", ["malformed_telemetry"]), (label, task))

    def test_a_non_finite_number_is_malformed_even_in_a_field_nobody_validates(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            text = self.text(extra=', "note": %s' % constant)
            for task in (LONG, SHORT):
                self.assertEqual(self.verdict(text, task),
                                 (10, "ineligible", ["malformed_telemetry"]),
                                 (constant, task))

    def test_the_size_cap_is_exact(self):
        text = self.text()
        size = len(text.encode("utf-8"))
        with mock.patch.object(quota, "SNAPSHOT_MAX_BYTES", size):
            self.assertEqual(self.verdict(text), (0, "eligible", ["ok"]))
        with mock.patch.object(quota, "SNAPSHOT_MAX_BYTES", size - 1):
            for task in (LONG, SHORT):
                self.assertEqual(self.verdict(text, task),
                                 (10, "ineligible", ["malformed_telemetry"]), task)

    def test_a_snapshot_over_one_mebibyte_is_malformed(self):
        text = self.text(extra=",\n" + " " * quota.SNAPSHOT_MAX_BYTES + '"x": 1')
        self.assertGreater(len(text), quota.SNAPSHOT_MAX_BYTES)
        self.assertEqual(self.verdict(text), (10, "ineligible", ["malformed_telemetry"]))

    def test_the_window_count_cap_is_exact(self):
        def windows(count):
            return "[%s]" % ", ".join(
                '{"kind": "w%d", "used_percent": %d}' % (i, i) for i in range(count))
        self.assertEqual(quota.SNAPSHOT_MAX_WINDOWS, 16)
        self.assertEqual(self.verdict(self.text(windows(16))), (0, "eligible", ["ok"]))
        for task in (LONG, SHORT):
            self.assertEqual(self.verdict(self.text(windows(17)), task),
                             (10, "ineligible", ["malformed_telemetry"]), task)


if __name__ == "__main__":
    unittest.main()
