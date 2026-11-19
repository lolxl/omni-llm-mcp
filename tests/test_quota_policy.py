# no-opencc
"""The shipped quota policy: schema, revision, no overlay, ``quota policy show``."""
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from providers import quota

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGINS = ("openai", "xai", "google", "cursor", "anthropic")
EMAIL = "noreply@example.invalid"


def _shipped():
    with open(quota.POLICY_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def _show(argv=("policy", "show", "--json")):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = quota.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ShippedPolicyTests(unittest.TestCase):
    def test_policy_loads_and_is_valid(self):
        policy, revision = quota.load_policy()
        self.assertEqual(policy["policy_schema"], "quota-policy/v1")
        self.assertRegex(revision, r"^[0-9a-f]{64}$")

    def test_one_quota_pool_per_vendor_login_with_a_default_account(self):
        policy, _ = quota.load_policy()
        logins = sorted(pool["login"] for pool in policy["pools"].values())
        self.assertEqual(logins, sorted(LOGINS))
        for pool in policy["pools"].values():
            self.assertEqual(pool["accounts"], ["default"])
            self.assertEqual(pool["profile"], "default")

    def test_monitoring_kinds(self):
        policy, _ = quota.load_policy()
        kinds = {p["login"]: p["monitoring"] for p in policy["pools"].values()}
        self.assertEqual(kinds, {
            "openai": "codex_rpc", "anthropic": "anthropic_snapshot",
            "xai": "unmonitored", "google": "unmonitored", "cursor": "unmonitored",
        })

    def test_numbers_match_the_plan(self):
        policy, _ = quota.load_policy()
        self.assertEqual(policy["reserve_threshold_bp"], 9000)
        self.assertEqual(policy["anthropic_snapshot"]["max_age_seconds"], 7200)
        self.assertEqual(policy["anthropic_snapshot"]["future_tolerance_seconds"], 300)
        self.assertEqual(policy["codex_rpc"]["cache_freshness_s"], 60)
        self.assertEqual(policy["codex_rpc"]["timeout_s"], 5)
        self.assertEqual(policy["decision_ttl_cap_s"], 300)

    def test_package_data_ships_the_policy(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertRegex(text, r"(?m)^\[tool\.setuptools\.package-data\]$")
        self.assertRegex(text, r'(?m)^providers = \[.*"quota_policy\.json".*\]$')
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "providers", "quota_policy.json")))


class RevisionTests(unittest.TestCase):
    def test_revision_is_sha256_of_canonical_content(self):
        doc = _shipped()
        canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        self.assertEqual(quota.policy_revision(doc),
                         hashlib.sha256(canonical.encode("ascii")).hexdigest())

    def test_revision_ignores_key_order_and_whitespace(self):
        doc = _shipped()
        reordered = json.loads(json.dumps(doc, sort_keys=True))
        self.assertEqual(quota.policy_revision(doc), quota.policy_revision(reordered))

    def test_revision_changes_with_any_content_change(self):
        doc = _shipped()
        base = quota.policy_revision(doc)
        changed = copy.deepcopy(doc)
        changed["reserve_threshold_bp"] = 8999
        self.assertNotEqual(base, quota.policy_revision(changed))
        changed = copy.deepcopy(doc)
        changed["pools"]["xai-sub"]["monitoring"] = "codex_rpc"
        self.assertNotEqual(base, quota.policy_revision(changed))


class SchemaTests(unittest.TestCase):
    def _bad(self, mutate):
        doc = _shipped()
        mutate(doc)
        with self.assertRaises(quota.PolicyError):
            quota.validate_policy(doc)

    def test_shipped_document_validates(self):
        quota.validate_policy(_shipped())

    def test_unknown_or_missing_top_level_key(self):
        self._bad(lambda d: d.update(extra=1))
        self._bad(lambda d: d.pop("pools"))
        self._bad(lambda d: d.update(policy_schema="quota-policy/v2"))

    def test_threshold_must_be_an_integer_in_range(self):
        for value in (0, 10001, -1, 9000.5, "9000", True, None):
            self._bad(lambda d, v=value: d.update(reserve_threshold_bp=v))

    def test_numeric_sections_are_bounded(self):
        self._bad(lambda d: d["codex_rpc"].update(timeout_s=0))
        self._bad(lambda d: d["codex_rpc"].update(cache_freshness_s=True))
        self._bad(lambda d: d["anthropic_snapshot"].update(max_age_seconds=86401))
        self._bad(lambda d: d["anthropic_snapshot"].update(future_tolerance_seconds=-1))
        self._bad(lambda d: d["codex_rpc"].update(extra=1))

    def test_pool_rows_are_closed(self):
        self._bad(lambda d: d["pools"]["xai-sub"].update(extra=1))
        self._bad(lambda d: d["pools"]["xai-sub"].pop("profile"))
        self._bad(lambda d: d["pools"]["xai-sub"].update(road="nowhere"))
        self._bad(lambda d: d["pools"]["xai-sub"].update(monitoring="metered"))

    def test_login_is_claimed_by_one_pool_and_is_an_id(self):
        self._bad(lambda d: d["pools"]["xai-sub"].update(login="openai"))
        self._bad(lambda d: d["pools"]["xai-sub"].update(login=EMAIL))
        self._bad(lambda d: d["pools"].update({EMAIL: d["pools"].pop("xai-sub")}))

    def test_accounts_need_default_and_unique_ids(self):
        self._bad(lambda d: d["pools"]["xai-sub"].update(accounts=[]))
        self._bad(lambda d: d["pools"]["xai-sub"].update(accounts=["work"]))
        self._bad(lambda d: d["pools"]["xai-sub"].update(accounts=["default", "default"]))
        self._bad(lambda d: d["pools"]["xai-sub"].update(accounts=["default", "a b"]))

    def test_a_monitored_pool_serves_one_account(self):
        # One telemetry binding per pool: a second account needs its own binding.
        self._bad(lambda d: d["pools"]["anthropic-sub"].update(accounts=["default", "work"]))
        self._bad(lambda d: d["pools"]["openai-sub"].update(accounts=["default", "work"]))
        doc = _shipped()
        doc["pools"]["xai-sub"]["accounts"] = ["default", "work"]
        quota.validate_policy(doc)

    def test_monitoring_kind_must_match_the_road(self):
        self._bad(lambda d: d["pools"]["openai-sub"].update(road="grok"))
        self._bad(lambda d: d["pools"]["anthropic-sub"].update(road="codex"))

    def test_load_policy_refuses_unreadable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                "missing.json": None,
                "garbage.json": "{not json",
                "nan.json": '{"reserve_threshold_bp": NaN}',
                "dup.json": '{"policy_schema": "quota-policy/v1", "policy_schema": "x"}',
                "list.json": "[]",
            }
            for name, text in cases.items():
                path = os.path.join(tmp, name)
                if text is not None:
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(text)
                with self.assertRaises(quota.PolicyError, msg=name):
                    quota.load_policy(path)


class LoaderGuardTests(unittest.TestCase):
    """Each loader and validator guard fails alone when it is removed."""

    def _load_text(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return quota.load_policy(path)

    def _shipped_text(self):
        with open(quota.POLICY_FILE, encoding="utf-8") as handle:
            return handle.read()

    def test_the_shipped_text_loads(self):
        self.assertEqual(self._load_text(self._shipped_text())[0], _shipped())

    def test_a_duplicate_key_is_unreadable_even_when_the_last_value_is_valid(self):
        text = self._shipped_text()
        top = text.replace('"reserve_threshold_bp": 9000,',
                           '"reserve_threshold_bp": 1,\n  "reserve_threshold_bp": 9000,', 1)
        nested = text.replace('"profile": "default",',
                              '"profile": "x",\n      "profile": "default",', 1)
        for label, doc in (("top level", top), ("nested", nested)):
            self.assertNotEqual(doc, text, label)
            json.loads(doc)   # plain JSON takes the last value and is valid
            with self.assertRaises(quota.PolicyError, msg=label) as cm:
                self._load_text(doc)
            self.assertEqual(str(cm.exception), "policy unreadable", label)

    def test_a_non_finite_number_is_unreadable(self):
        text = self._shipped_text()
        for constant in ("NaN", "Infinity", "-Infinity"):
            for field in ('"decision_ttl_cap_s": 300', '"reserve_threshold_bp": 9000'):
                key = field.split(":")[0]
                doc = text.replace(field, "%s: %s" % (key, constant), 1)
                self.assertNotEqual(doc, text)
                with self.assertRaises(quota.PolicyError, msg=(constant, field)) as cm:
                    self._load_text(doc)
                # not merely invalid as a number: the loader refused the constant
                self.assertEqual(str(cm.exception), "policy unreadable", (constant, field))

    def test_a_number_that_overflows_to_infinity_is_unreadable(self):
        text = self._shipped_text()
        for number in ("1e999", "-1e999", "1E+999"):
            for field in ('"decision_ttl_cap_s": 300', '"reserve_threshold_bp": 9000'):
                key = field.split(":")[0]
                doc = text.replace(field, "%s: %s" % (key, number), 1)
                self.assertNotEqual(doc, text)
                with self.assertRaises(quota.PolicyError, msg=(number, field)) as cm:
                    self._load_text(doc)
                self.assertEqual(str(cm.exception), "policy unreadable", (number, field))

    def test_the_profile_must_be_a_plain_id(self):
        for value in ("a b", EMAIL, "", "x" * 65, 5, None, ["default"], "a/b", "lé"):
            self.assertRaises(quota.PolicyError, quota.validate_policy,
                              self._with(lambda d, v=value: d["pools"]["xai-sub"].update(
                                  profile=v)))

    def test_the_decision_ttl_cap_is_an_integer_between_1_and_3600(self):
        for value in (0, -1, 3601, 10 ** 6, True, 1.5, "300", None, [300]):
            self.assertRaises(quota.PolicyError, quota.validate_policy,
                              self._with(lambda d, v=value: d.update(decision_ttl_cap_s=v)))
        for value in (1, 300, 3600):
            quota.validate_policy(self._with(lambda d, v=value: d.update(
                decision_ttl_cap_s=v)))

    def test_codex_and_snapshot_bounds_are_enforced_at_both_ends(self):
        for section, field, bad, good in (
                ("codex_rpc", "cache_freshness_s", (0, 3601), (1, 3600)),
                ("codex_rpc", "timeout_s", (0, 31), (1, 30)),
                ("codex_rpc", "future_tolerance_seconds", (-1, 301), (0, 300)),
                ("anthropic_snapshot", "max_age_seconds", (0, 86401), (1, 86400)),
                ("anthropic_snapshot", "future_tolerance_seconds", (-1, 3601), (0, 3600))):
            for value in bad:
                self.assertRaises(quota.PolicyError, quota.validate_policy, self._with(
                    lambda d, s=section, f=field, v=value: d[s].update({f: v})))
            for value in good:
                quota.validate_policy(self._with(
                    lambda d, s=section, f=field, v=value: d[s].update({f: v})))

    def _with(self, mutate):
        doc = _shipped()
        mutate(doc)
        return doc


class NoOverlayTests(unittest.TestCase):
    def test_state_dir_and_environment_cannot_change_the_policy(self):
        _, baseline = quota.load_policy()
        with tempfile.TemporaryDirectory() as state:
            hostile = _shipped()
            hostile["reserve_threshold_bp"] = 1
            for name in ("quota_policy.json", "pools.local.json", "quota-policy.json"):
                with open(os.path.join(state, name), "w", encoding="utf-8") as handle:
                    json.dump(hostile, handle)
            env = {"OMNI_LLM_STATE_DIR": state, "QUOTA_POOLS_FILE":
                   os.path.join(state, "quota_policy.json"),
                   "QUOTA_DIR": state, "MOFUU_STATE_HOME": state}
            with mock.patch.dict(os.environ, env):
                _, revision = quota.load_policy()
                code, out, _err = _show()
        self.assertEqual(revision, baseline)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["policy_revision"], baseline)
        self.assertEqual(json.loads(out)["reserve_threshold_bp"], 9000)


class PolicyShowTests(unittest.TestCase):
    def test_show_prints_the_effective_map_and_revision(self):
        code, out, err = _show()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        doc = json.loads(out)
        _, revision = quota.load_policy()
        self.assertEqual(doc["interface_version"], "quota-decision/v1")
        self.assertEqual(doc["policy_revision"], revision)
        self.assertEqual(doc["reserve_threshold_bp"], 9000)
        by_login = {row["login"]: row for row in doc["map"]}
        self.assertEqual(sorted(by_login), sorted(LOGINS))
        self.assertEqual(by_login["openai"], {
            "login": "openai", "account": "default", "quota_pool": "openai-sub",
            "monitoring": "codex_rpc"})
        for row in doc["map"]:
            self.assertEqual(set(row), {"login", "account", "quota_pool", "monitoring"})
            self.assertEqual(row["account"], "default")

    def test_show_needs_json_and_never_echoes_an_argument(self):
        code, out, err = _show(("policy", "show"))
        self.assertEqual((code, out), (2, ""))
        code, out, err = _show(("policy", "show", "--json", "--who=" + EMAIL))
        self.assertEqual((code, out), (2, ""))
        for needle in (EMAIL, "example.invalid", "noreply@", "--who"):
            self.assertNotIn(needle, err)

    def test_show_refuses_when_the_policy_is_invalid(self):
        with mock.patch.object(quota, "load_policy", side_effect=quota.PolicyError("x")):
            code, out, _err = _show()
        self.assertEqual(code, 10)
        doc = json.loads(out)
        self.assertEqual(doc["reasons"], ["registry_invalid"])
        self.assertIsNone(doc["policy_revision"])
        self.assertEqual(doc["map"], [])

    def test_reason_enum_is_closed_and_has_no_duplicates(self):
        self.assertEqual(len(set(quota.REASONS)), len(quota.REASONS))
        for reason in ("telemetry_unavailable", "ledger_cooling", "ledger_unreadable",
                       "unknown_account", "telemetry_malformed"):
            self.assertIn(reason, quota.NEW_REASONS)
        for reason in ("ok", "quota_reserve", "stale_telemetry", "missing_telemetry",
                       "malformed_telemetry", "unmonitored_pool", "unknown_login",
                       "registry_invalid"):
            self.assertIn(reason, quota.CALLER_REASONS)
        for reason in quota.REASONS:
            self.assertRegex(reason, re.compile(r"^[a-z_]+$"))


if __name__ == "__main__":
    unittest.main()
