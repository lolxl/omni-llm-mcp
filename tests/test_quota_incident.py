# no-opencc
"""``quota report-incident`` and the incident markers ``quota check`` reads.

Runs against a throwaway state dir and a pinned clock (see test_quota_check).
"""
import json
import os
import stat
import unittest
import uuid
from datetime import timedelta
from unittest import mock

from providers import quota
from tests.test_quota_check import EMAIL, LONG, NOW, SHORT, QuotaCase, _info


def _iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_argv(login="xai", account=None, kind="hard_rate_limit", until=None):
    argv = ["report-incident", "--login", login, "--kind", kind,
            "--until", until or _iso(NOW + timedelta(seconds=600)), "--json"]
    if account is not None:
        argv += ["--account", account]
    return argv


class IncidentCase(QuotaCase):
    def report(self, **kwargs):
        return self.run_cli(_report_argv(**kwargs))

    def marker_path(self, pool="xai-sub", account="default"):
        return os.path.join(self.state, quota.INCIDENT_DIR,
                            "%s.%s.hard_rate_limit.json" % (pool, account))

    def write_marker(self, content, pool="xai-sub", account="default", raw=False):
        path = self.marker_path(pool, account)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content if raw else json.dumps(content))
        return path

    def marker(self, until_s=600, **overrides):
        doc = {"interface_version": "quota-incident/v1", "quota_pool": "xai-sub",
               "account": "default", "kind": "hard_rate_limit",
               "until": _iso(NOW + timedelta(seconds=until_s)),
               "recorded_at": _iso(NOW), "incident_id": str(uuid.uuid4())}
        doc.update(overrides)
        return doc

    def xai(self, task=LONG, context="primary"):
        return self.check(login="xai", task=task, context=context,
                          model="grok-4.6", transport="grok:grok-4.6")

    def assert_refused(self, doc, code):
        self.assertEqual(code, 10)
        self.assertEqual(doc["decision"], "ineligible")
        self.assertEqual(doc["reasons"], ["hard_rate_limit"])


class ReportTests(IncidentCase):
    def test_report_writes_the_marker_and_prints_no_path(self):
        code, doc, out, err = self.report()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(tuple(doc), quota.REPORT_KEYS)
        self.assertEqual(doc["interface_version"], "quota-incident/v1")
        self.assertIs(doc["ok"], True)
        self.assertEqual((doc["quota_pool"], doc["login"], doc["account"], doc["kind"]),
                         ("xai-sub", "xai", "default", "hard_rate_limit"))
        self.assertEqual(doc["until"], "2026-11-18T12:10:00Z")
        self.assertEqual(doc["recorded_at"], "2026-11-18T12:00:00Z")
        uuid.UUID(doc["incident_id"])
        self.assertNotIn(self.tmp.name, out)
        with open(self.marker_path(), encoding="utf-8") as handle:
            marker = json.load(handle)
        self.assertEqual(tuple(marker), quota.INCIDENT_KEYS)
        self.assertEqual(marker["incident_id"], doc["incident_id"])
        directory = os.path.dirname(self.marker_path())
        self.assertEqual(stat.S_IMODE(os.stat(directory).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.marker_path()).st_mode), 0o600)
        leftovers = [n for n in os.listdir(directory) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_each_report_appends_one_allowlisted_log_line(self):
        self.report()
        self.report(until=_iso(NOW - timedelta(seconds=1)))
        with open(os.path.join(self.state, quota.INCIDENT_LOG), encoding="utf-8") as h:
            lines = [json.loads(line) for line in h.read().splitlines()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(tuple(line), quota.INCIDENT_KEYS)
        self.assertEqual(lines[1]["until"], "2026-11-18T11:59:59Z")

    def test_an_offset_until_is_stored_in_utc(self):
        code, doc, _out, _err = self.report(until="2026-11-18T21:10:00+09:00")
        self.assertEqual(code, 0)
        self.assertEqual(doc["until"], "2026-11-18T12:10:00Z")

    def test_an_omitted_account_and_default_write_the_same_marker(self):
        self.report()
        first = self.marker_path()
        self.report(account="default", until=_iso(NOW - timedelta(seconds=5)))
        self.assertEqual(os.listdir(os.path.dirname(first)), [os.path.basename(first)])

    def test_usage_errors_never_echo_the_value(self):
        secret = "zz-secret-login"
        cases = (
            (_report_argv(login=secret), "--login is not a login of the quota policy"),
            (_report_argv(login="bad login@x"), "--login must match"),
            (_report_argv(account=secret), "--account is not listed for this login"),
            (_report_argv(account="x@y"), "--account must match"),
            (_report_argv(kind="soft-" + secret), "--kind must be hard_rate_limit"),
            (_report_argv(until="2026-11-18T12:10:00"), "--until must be an ISO 8601"),
            (_report_argv(until="2026-11-18T12:10:00.123"), "--until must be an ISO 8601"),
            # Converting to UTC overflows: a usage error, not a traceback.
            (_report_argv(until="9999-12-31T23:59:59-01:00"), "--until must be an ISO 8601"),
            (_report_argv(until="tomorrow-" + secret), "--until must be an ISO 8601"),
            (["report-incident", "--login", "xai", "--json"], "required"),
        )
        for argv, needle in cases:
            with self.subTest(argv=argv):
                code, doc, out, err = self.run_cli(argv)
                self.assertEqual(code, 2)
                self.assertIsNone(doc)
                self.assertEqual(out, "")
                self.assertIn(needle, err)
                self.assertNotIn(secret, err)
                self.assertNotIn("bad login", err)
                self.assertNotIn("x@y", err)
                self.assertNotIn("12:10:00", err)
                self.assertNotIn("Traceback", err)
        self.assertFalse(os.path.exists(os.path.dirname(self.marker_path())))

    def test_ids_are_uuid4_in_the_report_the_marker_and_the_log(self):
        _code, doc, _out, _err = self.report()
        with open(self.marker_path(), encoding="utf-8") as handle:
            marker = json.load(handle)
        with open(os.path.join(self.state, quota.INCIDENT_LOG), encoding="utf-8") as h:
            logged = json.loads(h.read().splitlines()[-1])
        for value in (doc["incident_id"], marker["incident_id"], logged["incident_id"]):
            self.assertEqual(uuid.UUID(value).version, 4)

    def test_a_sub_second_until_is_rounded_up_and_still_refuses(self):
        later = NOW + timedelta(milliseconds=750)
        with mock.patch.object(quota, "_now", lambda: later):
            code, doc, _out, _err = self.report(until="2026-11-18T12:00:00.900Z")
            self.assertEqual((code, doc["until"]), (0, "2026-11-18T12:00:01Z"))
            for task in (LONG, SHORT):
                code, check = self.xai(task=task)
                self.assert_refused(check, code)
                self.assertEqual(check["expires_at"], "2026-11-18T12:00:01Z")

    def test_the_marker_is_written_beside_itself_fsynced_then_replaced(self):
        events = []
        real_mkstemp, real_fsync, real_replace = (quota.tempfile.mkstemp, quota.os.fsync,
                                                  quota.os.replace)

        def mkstemp(**kwargs):
            events.append(("mkstemp", kwargs.get("dir")))
            return real_mkstemp(**kwargs)

        def fsync(fd):
            events.append(("fsync", None))
            return real_fsync(fd)

        def replace(src, dst):
            events.append(("replace", os.path.dirname(src) == os.path.dirname(dst)))
            return real_replace(src, dst)
        real_write = os.write

        def write(fd, data):
            if not events or events[-1] != ("write", None):
                events.append(("write", None))
            return real_write(fd, data)
        with mock.patch.object(quota.tempfile, "mkstemp", mkstemp), \
                mock.patch.object(quota.os, "fsync", fsync), \
                mock.patch.object(quota.os, "write", write), \
                mock.patch.object(quota.os, "replace", replace):
            code, _doc, _out, _err = self.report()
        self.assertEqual(code, 0)
        # The payload, then fsync, then replace (the log line is written after).
        self.assertEqual(events[:4], [("mkstemp", os.path.dirname(self.marker_path())),
                                      ("write", None), ("fsync", None), ("replace", True)])

    def test_short_writes_still_publish_a_complete_marker(self):
        real_write = os.write

        def one_byte(fd, data):
            return real_write(fd, bytes(data[:1]))
        with mock.patch.object(quota.os, "write", one_byte):
            code, doc, _out, _err = self.report()
        self.assertEqual(code, 0)
        with open(self.marker_path(), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["incident_id"], doc["incident_id"])

    def test_a_write_failure_exits_73_with_fixed_text(self):
        # A regular file where the marker directory belongs: nothing can be written.
        with open(os.path.join(self.state, quota.INCIDENT_DIR), "w") as handle:
            handle.write("x")
        code, doc, out, err = self.report()
        self.assertEqual(code, 73)
        self.assertIsNone(doc)
        self.assertEqual(out, "")
        self.assertEqual(err, "quota: cannot record the incident\n")
        self.assertFalse(os.path.exists(os.path.join(self.state, quota.INCIDENT_LOG)))

    def test_a_failed_replace_leaves_no_temp_file_and_exits_73(self):
        with mock.patch.object(quota.os, "replace", side_effect=OSError("disk full")):
            code, _doc, out, err = self.report()
        self.assertEqual((code, out), (73, ""))
        self.assertEqual(err, "quota: cannot record the incident\n")
        directory = os.path.dirname(self.marker_path())
        self.assertEqual(os.listdir(directory), [])

    def test_a_closed_stdout_exits_74(self):
        with mock.patch.object(quota, "_emit", return_value=False):
            code, _doc, _out, _err = self.report()
        self.assertEqual(code, 74)


class CheckTests(IncidentCase):
    def test_an_active_incident_refuses_long_and_short(self):
        self.report(until=_iso(NOW + timedelta(seconds=120)))
        for task in (LONG, SHORT):
            with self.subTest(task=task):
                code, doc = self.xai(task=task)
                self.assert_refused(doc, code)
                self.assertEqual(doc["quota_pool"], "xai-sub")
                # expires_at = min(until, the 300 s cap)
                self.assertEqual(doc["expires_at"], "2026-11-18T12:02:00Z")

    def test_the_expiry_uses_one_clock_sample(self):
        self.report(until=_iso(NOW + timedelta(seconds=120)))
        with mock.patch.object(quota, "_now", lambda: NOW + timedelta(milliseconds=600)):
            code, doc = self.xai()
        self.assert_refused(doc, code)
        self.assertEqual((doc["checked_at"], doc["expires_at"]),
                         ("2026-11-18T12:00:00Z", "2026-11-18T12:02:00Z"))

    def test_an_incident_ends_exactly_at_until(self):
        self.report(until=_iso(NOW + timedelta(seconds=120)))
        with mock.patch.object(quota, "_now", lambda: NOW + timedelta(seconds=120)):
            code, doc = self.xai()
        self.assertEqual((code, doc["decision"]), (0, "eligible"))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root reads anything")
    def test_a_marker_that_cannot_be_opened_refuses(self):
        path = self.write_marker(self.marker(until_s=-600))
        os.chmod(path, 0)
        self.addCleanup(os.chmod, path, 0o600)
        code, doc = self.xai(task=SHORT)
        self.assert_refused(doc, code)

    def test_an_incident_is_judged_before_the_ledger(self):
        self.report()
        self.write("health.json", "{not json", raw=True)
        code, doc = self.xai()
        self.assert_refused(doc, code)

    def test_the_printed_decision_keeps_its_key_set_and_the_audit_gets_the_state(self):
        self.report()
        code, doc, out, _err = self.run_cli(["check", "--login", "xai", "--transport",
                                             "grok:grok-4.6", "--model", "grok-4.6",
                                             "--task-class", "long", "--context",
                                             "primary", "--json"])
        self.assertEqual(tuple(doc), quota.DOC_KEYS)
        self.assertNotIn("incident_state", out)
        self.assertEqual(json.loads(self.audit_lines()[-1])["incident_state"], "active")

    def test_an_unlisted_incident_state_never_reaches_the_audit(self):
        doc, _code = quota.decide("xai", None, "grok:grok-4.6", "grok-4.6", LONG,
                                  "primary")
        quota.append_audit(doc, {"incident_state": "/home/x"})
        self.assertNotIn("incident_state", json.loads(self.audit_lines()[-1]))

    def test_a_past_sub_second_until_clears_at_once(self):
        self.report()
        later = NOW + timedelta(milliseconds=750)
        with mock.patch.object(quota, "_now", lambda: later):
            code, doc, _out, _err = self.report(until="2026-11-18T12:00:00.600Z")
            self.assertEqual((code, doc["until"]), (0, "2026-11-18T12:00:00Z"))
            code, check = self.xai(task=SHORT)
        self.assertEqual((code, check["decision"]), (0, "eligible"))

    def test_the_last_representable_instant_is_recordable_and_refuses(self):
        code, doc, _out, _err = self.report(until="9999-12-31T23:59:59.999999Z")
        self.assertEqual((code, doc["until"]), (0, "9999-12-31T23:59:59Z"))
        self.write_marker(self.marker(until="9999-12-31T23:59:59.999999Z"))
        code, check = self.xai()
        self.assert_refused(check, code)
        self.assertEqual(check["expires_at"], "2026-11-18T12:05:00Z")

    def test_a_fractional_marker_expiry_is_rounded_up(self):
        self.write_marker(self.marker(until="2026-11-18T12:01:00.400Z"))
        code, doc = self.xai()
        self.assert_refused(doc, code)
        self.assertEqual(doc["expires_at"], "2026-11-18T12:01:01Z")

    def test_an_incident_decision_reads_the_clock_once(self):
        self.report()
        calls = []

        def clock():
            calls.append(1)
            return NOW + timedelta(seconds=len(calls))
        with mock.patch.object(quota, "_now", clock):
            doc, code = quota.decide("xai", None, "grok:grok-4.6", "grok-4.6", LONG,
                                     "primary")
        self.assert_refused(doc, code)
        self.assertEqual(len(calls), 1)
        self.assertEqual(doc["checked_at"], "2026-11-18T12:00:01Z")

    def test_the_expiry_is_capped(self):
        self.report(until=_iso(NOW + timedelta(hours=5)))
        code, doc = self.xai()
        self.assert_refused(doc, code)
        self.assertEqual(doc["expires_at"], "2026-11-18T12:05:00Z")

    def test_after_until_the_check_admits_again(self):
        self.report(until=_iso(NOW + timedelta(seconds=120)))
        with mock.patch.object(quota, "_now", lambda: NOW + timedelta(seconds=121)):
            code, doc = self.xai()
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (0, "eligible", ["unmonitored_pool"]))
        self.assertEqual(json.loads(self.audit_lines()[-1])["incident_state"], "expired")

    def test_a_past_until_clears_an_active_incident(self):
        self.report()
        self.assert_refused(*reversed(self.xai()))
        self.report(until=_iso(NOW - timedelta(seconds=1)))
        code, doc = self.xai()
        self.assertEqual((code, doc["decision"]), (0, "eligible"))

    def test_an_active_incident_refuses_a_monitored_pool_before_the_rpc(self):
        self.rpc(_info(primary=10.0))
        self.report(login="openai")
        for task in (LONG, SHORT):
            with self.subTest(task=task):
                code, doc = self.check(task=task)
                self.assert_refused(doc, code)
                self.assertEqual(doc["quota_pool"], "openai-sub")
        self.assertEqual(self.rpc_calls, [])

    def test_a_monitored_pool_without_an_incident_is_unchanged(self):
        self.rpc(_info(primary=10.0))
        self.report()   # xai only
        code, doc = self.check()
        self.assertEqual((code, doc["reasons"]), (0, ["ok"]))
        self.assertNotIn("incident_state", json.loads(self.audit_lines()[-1]))

    def test_an_incident_on_another_pool_has_no_effect(self):
        self.report(login="google")
        code, doc = self.xai()
        self.assertEqual((code, doc["decision"]), (0, "eligible"))

    def test_an_incident_on_another_account_has_no_effect(self):
        policy, revision = quota.load_policy()
        policy = json.loads(json.dumps(policy))
        policy["pools"]["xai-sub"]["accounts"] = ["default", "work"]
        self.write_marker(self.marker(account="work"), account="work")

        def decide(account):
            return quota.decide("xai", account, "grok:grok-4.6", "grok-4.6", LONG,
                                "primary", policy=policy, revision=revision)
        doc, code = decide("default")
        self.assertEqual((code, doc["decision"]), (0, "eligible"))
        doc, code = decide("work")
        self.assert_refused(doc, code)

    def test_a_marker_for_another_key_in_this_file_is_unreadable(self):
        for field, value in (("quota_pool", "google-sub"), ("account", "work"),
                             ("kind", "soft_limit"),
                             ("interface_version", "quota-incident/v2")):
            with self.subTest(field=field):
                # Expired once accepted: only the binding check refuses it.
                self.write_marker(self.marker(until_s=-600, **{field: value}))
                code, doc = self.xai()
                self.assert_refused(doc, code)

    def test_unreadable_markers_refuse_and_are_never_reusable(self):
        target = os.path.join(self.tmp.name, "elsewhere.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.marker(until_s=-600), handle)
        cases = {
            "corrupt": lambda: self.write_marker('{"interface_version": "quo', raw=True),
            "empty": lambda: self.write_marker("", raw=True),
            "non-object": lambda: self.write_marker([self.marker()]),
            "duplicate key": lambda: self.write_marker(
                '{"until": "x", ' + json.dumps(self.marker())[1:], raw=True),
            "bad until": lambda: self.write_marker(self.marker(until="soon")),
            "naive until": lambda: self.write_marker(
                self.marker(until="2026-11-18T12:10:00")),
            "number until": lambda: self.write_marker(self.marker(until=1e12)),
            # Converting to UTC overflows: unreadable, not registry_invalid.
            "overflowing until": lambda: self.write_marker(
                self.marker(until="0001-01-01T00:00:00+14:00")),
            "bad recorded_at": lambda: self.write_marker(self.marker(recorded_at="")),
            "bad incident_id": lambda: self.write_marker(self.marker(incident_id="1")),
            "uuid5 incident_id": lambda: self.write_marker(self.marker(
                until_s=-600, incident_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "x")))),
            "oversized trailing": lambda: self.write_marker(
                json.dumps(self.marker(until_s=-600))
                + " " * (quota.INCIDENT_MAX_BYTES + 1), raw=True),
            "extra key": lambda: self.write_marker(dict(self.marker(), note="x")),
            "missing key": lambda: self.write_marker(
                {k: v for k, v in self.marker().items() if k != "recorded_at"}),
            # Valid and expired once read whole: only the size cap refuses it.
            "oversized": lambda: self.write_marker(
                " " * (quota.INCIDENT_MAX_BYTES + 1)
                + json.dumps(self.marker(until_s=-600)), raw=True),
            "symlink": lambda: os.symlink(target, self.marker_path()),
            "directory": lambda: os.makedirs(self.marker_path()),
        }
        for label, make in cases.items():
            with self.subTest(marker=label):
                directory = os.path.dirname(self.marker_path())
                os.makedirs(directory, exist_ok=True)
                path = self.marker_path()
                if os.path.islink(path) or os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    os.rmdir(path)
                make()
                for task in (LONG, SHORT):
                    code, doc = self.xai(task=task)
                    self.assert_refused(doc, code)
                    self.assertEqual(doc["expires_at"], doc["checked_at"])
                    line = json.loads(self.audit_lines()[-1])
                    self.assertEqual(line["incident_state"], "unreadable")

    def test_no_path_email_or_traceback_in_any_output_or_audit(self):
        outputs = []
        outputs.extend(self.report(until="2026-11-18T12:10:00Z")[2:])
        outputs.extend(self.run_cli(_report_argv(login=EMAIL))[2:])
        outputs.extend(self.run_cli(["check", "--login", "xai", "--transport",
                                     "grok:grok-4.6", "--model", "grok-4.6",
                                     "--task-class", "long", "--context", "primary",
                                     "--json"])[2:])
        self.write_marker("{", raw=True)
        outputs.extend(self.run_cli(["check", "--login", "xai", "--transport",
                                     "grok:grok-4.6", "--model", "grok-4.6",
                                     "--task-class", "short", "--context", "primary",
                                     "--json"])[2:])
        with open(os.path.join(self.state, quota.INCIDENT_DIR, "x"), "w"):
            pass
        os.chmod(os.path.join(self.state, quota.INCIDENT_DIR), 0o500)
        self.addCleanup(os.chmod, os.path.join(self.state, quota.INCIDENT_DIR), 0o700)
        outputs.extend(self.report()[2:])
        outputs.extend(self.audit_lines())
        with open(os.path.join(self.state, quota.INCIDENT_LOG), encoding="utf-8") as h:
            outputs.append(h.read())
        blob = "\n".join(outputs)
        for needle in (self.tmp.name, self.state, os.path.expanduser("~"), "@",
                       "Traceback", ".json"):
            self.assertNotIn(needle, blob)


if __name__ == "__main__":
    unittest.main()
