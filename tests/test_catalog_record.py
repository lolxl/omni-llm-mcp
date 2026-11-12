"""Strict records and the claude, codex, and grok parsers. No vendor process."""
import datetime
import json
import unittest
from pathlib import Path

from providers.admission import decide
from providers.catalog_record import (
    StrictRecordError,
    grok_cli_version,
    parse_claude_models,
    parse_codex_result,
    parse_grok_output,
    validate_strict,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
OBSERVED = "2026-11-10T00:00:00Z"
CLAUDE_IDS = (
    "default",
    "opus",
    "fable",
    "sonnet",
    "haiku",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
CODEX_IDS = (
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)
GROK_IDS = (
    "grok-4.7",
    "grok-4.7-build-fast",
    "grok-4.6",
    "grok-4.5",
)


def _claude_models():
    payload = json.loads((FIXTURES / "model_catalog_claude_initialize.json").read_text(encoding="utf-8"))
    return payload["models"]


def _codex_result():
    return json.loads((FIXTURES / "model_catalog_codex_model_list.json").read_text(encoding="utf-8"))


def _grok_text():
    return (FIXTURES / "model_catalog_grok_models.txt").read_text(encoding="utf-8")


def _live(provider, rows, **extra):
    models = []
    for row in rows:
        item = {
            "id": row["id"],
            "name": row.get("name", row["id"]),
            "routable_id": "%s/%s" % (provider, row["id"]),
        }
        if "display_name" in row:
            item["display_name"] = row["display_name"]
        if "efforts" in row:
            item["supported_efforts"] = list(row["efforts"])
            item["efforts_known"] = row.get("known", True)
        elif row.get("absent"):
            item["efforts_known"] = False
        models.append(item)
    value = {
        "provider": provider,
        "source": extra.pop("source", "live"),
        "observed_at": extra.pop("observed_at", OBSERVED),
        "availability": extra.pop("availability", "provider_advertised"),
        "models": models,
    }
    value.update(extra)
    return value


class CatalogRecordTests(unittest.TestCase):
    def test_fixture_parsers_keep_todays_ids_and_order(self):
        claude = parse_claude_models(_claude_models(), observed_at=OBSERVED, cli_version="2.1.301")
        codex = parse_codex_result(_codex_result(), observed_at=OBSERVED, cli_version="0.162.1")
        grok = parse_grok_output(_grok_text(), observed_at=OBSERVED, cli_version="1.0.52")
        self.assertEqual([row["id"] for row in claude["models"]], list(CLAUDE_IDS))
        self.assertEqual([row["id"] for row in codex["models"]], list(CODEX_IDS))
        self.assertEqual([row["id"] for row in grok["models"]], list(GROK_IDS))

    def test_claude_parser_keeps_display_names(self):
        parsed = parse_claude_models(_claude_models(), observed_at=OBSERVED)
        record, warnings = validate_strict("claude", parsed)
        by_id = {row["id"]: row for row in record["models"]}
        self.assertEqual(warnings, [])
        self.assertEqual(by_id["fable"]["display_name"], "Fable 5.1")
        self.assertEqual(by_id["opus"]["display_name"], "Opus 5.5")
        self.assertEqual(by_id["claude-haiku-4-5-20251001"]["display_name"], "Haiku 4.5")
        self.assertEqual(by_id["claude-haiku-4-5-20251001"]["effort_field"], "absent")
        self.assertEqual(by_id["fable"]["effort_field"], "listed")
        self.assertEqual(by_id["fable"]["supported_efforts"],
                         ["low", "medium", "high", "xhigh", "max"])

    def test_codex_parser_keeps_known_efforts_in_todays_order(self):
        parsed = parse_codex_result(_codex_result(), observed_at=OBSERVED)
        record, warnings = validate_strict("codex", parsed)
        first = record["models"][0]
        self.assertEqual(warnings, [])
        self.assertEqual(first["id"], "gpt-6.1-sol")
        self.assertEqual(first["display_name"], "GPT-6.1-Sol")
        self.assertEqual(first["supported_efforts"],
                         ["low", "medium", "high", "xhigh", "max", "ultra"])
        self.assertEqual(first["default_effort"], "low")
        self.assertEqual(first["effort_field"], "listed")

    def test_grok_parser_keeps_both_default_markers(self):
        text = (
            "You are logged in with grok.com.\n"
            "\n"
            "Default model: grok-4.6\n"
            "\n"
            "Available models:\n"
            "  * grok-4.7 (default)\n"
            "  - grok-4.6\n"
            "  - grok-4.5\n"
        )
        parsed = parse_grok_output(text, observed_at=OBSERVED)
        record, warnings = validate_strict("grok", parsed)
        self.assertEqual(warnings, [])
        self.assertEqual(record["grok_default"]["line"], "grok-4.6")
        self.assertEqual(record["grok_default"]["row"], "grok-4.7")
        self.assertEqual(record["source_note"], "vendor_registry_not_entitlement")
        self.assertEqual(record["availability"], "unknown")
        self.assertEqual(record["models"][0]["effort_field"], "absent")

    def test_grok_cli_version_reads_the_first_dotted_token(self):
        self.assertEqual(
            grok_cli_version("1.0.52 (0920c8169e27) [stable] 2.0.0"),
            "1.0.52",
        )
        self.assertEqual(grok_cli_version("grok 1.0.79"), "1.0.79")
        self.assertIsNone(grok_cli_version("no version here"))

    def test_deep_is_kept_and_a_malformed_token_is_dropped(self):
        parsed = parse_codex_result({
            "data": [{
                "model": "gpt-test",
                "displayName": "Test",
                "supportedReasoningEfforts": ["deep", "NOT A TOKEN", "medium"],
            }],
        }, observed_at=OBSERVED)
        record, warnings = validate_strict("codex", parsed)
        self.assertEqual(record["models"][0]["supported_efforts"], ["medium", "deep"])
        self.assertEqual(record["models"][0]["effort_field"], "listed")
        self.assertEqual(warnings, [{
            "reason": "dropped_effort_token",
            "model_id": "gpt-test",
            "token": "NOT A TOKEN",
        }])

    def test_empty_effort_field_is_known_empty(self):
        value = _live("codex", [{"id": "gpt-test", "efforts": [], "known": True}])
        record, warnings = validate_strict("codex", value)
        row = record["models"][0]
        self.assertEqual(warnings, [])
        self.assertEqual(row["effort_field"], "empty")
        self.assertEqual(row["supported_efforts"], [])
        self.assertIs(row["efforts_known"], True)

    def test_absent_effort_field_stays_absent(self):
        value = _live("claude", [{"id": "haiku", "absent": True}])
        record, warnings = validate_strict("claude", value)
        row = record["models"][0]
        self.assertEqual(warnings, [])
        self.assertEqual(row["effort_field"], "absent")
        self.assertIs(row["efforts_known"], False)
        self.assertNotIn("supported_efforts", row)

    def test_effort_label_does_not_override_the_raw_field(self):
        labeled_absent = _live("codex", [{"id": "gpt-test", "absent": True}])
        labeled_absent["models"][0]["effort_field"] = "empty"
        record, warnings = validate_strict("codex", labeled_absent)
        row = record["models"][0]
        self.assertEqual(warnings, [])
        self.assertEqual(row["effort_field"], "absent")
        self.assertIs(row["efforts_known"], False)
        self.assertNotIn("supported_efforts", row)

        labeled_bad = _live("codex", [{"id": "gpt-test", "efforts": ["BAD"]}])
        labeled_bad["models"][0]["effort_field"] = "empty"
        record, warnings = validate_strict("codex", labeled_bad)
        row = record["models"][0]
        self.assertEqual(row["effort_field"], "absent")
        self.assertIs(row["efforts_known"], False)
        self.assertNotIn("supported_efforts", row)
        self.assertEqual(warnings, [{
            "reason": "dropped_effort_token",
            "model_id": "gpt-test",
            "token": "BAD",
        }])

    def test_non_string_effort_element_is_dropped_with_a_warning(self):
        observed = {"observed_at": OBSERVED}
        codex = parse_codex_result({
            "data": [{
                "model": "gpt-test",
                "displayName": "Test",
                "supportedReasoningEfforts": ["medium", 42],
            }],
        }, **observed)
        claude = parse_claude_models([{
            "value": "opus",
            "displayName": "Opus",
            "supportedEffortLevels": ["medium", 42],
        }], **observed)
        for provider, parsed, model_id in (
            ("codex", codex, "gpt-test"),
            ("claude", claude, "opus"),
        ):
            record, warnings = validate_strict(provider, parsed)
            self.assertEqual(record["models"][0]["id"], model_id)
            self.assertEqual(record["models"][0]["effort_field"], "listed")
            self.assertEqual(record["models"][0]["supported_efforts"], ["medium"])
            self.assertEqual(warnings, [{
                "reason": "dropped_effort_token",
                "model_id": model_id,
                "token": repr(42),
            }])

    def test_invalid_only_efforts_are_absent_not_known_empty(self):
        value = _live("codex", [{"id": "gpt-test", "efforts": ["NOT_A_TOKEN", "9nope"]}])
        record, warnings = validate_strict("codex", value)
        row = record["models"][0]
        self.assertEqual(row["effort_field"], "absent")
        self.assertIs(row["efforts_known"], False)
        self.assertNotIn("supported_efforts", row)
        self.assertEqual(warnings, [
            {"reason": "dropped_effort_token", "model_id": "gpt-test", "token": "NOT_A_TOKEN"},
            {"reason": "dropped_effort_token", "model_id": "gpt-test", "token": "9nope"},
        ])

    def test_mixed_efforts_keep_the_valid_tokens(self):
        value = _live("claude", [{
            "id": "opus",
            "efforts": ["medium", "NOT_A_TOKEN", "low"],
        }])
        record, warnings = validate_strict("claude", value)
        self.assertEqual(record["models"][0]["effort_field"], "listed")
        self.assertEqual(record["models"][0]["supported_efforts"], ["low", "medium"])
        self.assertEqual(warnings, [{
            "reason": "dropped_effort_token",
            "model_id": "opus",
            "token": "NOT_A_TOKEN",
        }])

    def test_revalidation_keeps_each_effort_state(self):
        absent = _live("grok", [{"id": "grok-4.6", "absent": True}])
        empty = _live("grok", [{"id": "grok-4.6", "efforts": []}])
        listed = _live("grok", [{"id": "grok-4.6", "efforts": ["high", "medium", "low"]}])
        now = datetime.datetime(2026, 11, 10, 12, 0, tzinfo=datetime.timezone.utc)
        checked = {}
        for name, value, expected in (
            ("absent", absent, None),
            ("empty", empty, []),
            ("listed", listed, ["low", "medium", "high"]),
        ):
            record, warnings = validate_strict("grok", value)
            again, again_warnings = validate_strict("grok", record)
            self.assertEqual(warnings, [])
            self.assertEqual(again_warnings, [])
            self.assertEqual(again, record)
            row = record["models"][0]
            if expected is None:
                self.assertNotIn("supported_efforts", row)
            else:
                self.assertEqual(row["supported_efforts"], expected)
            checked[name] = again
        admitted = decide(
            {"provider": "grok", "model": "grok-4.6", "effort": None},
            checked["absent"],
            now=now,
        )
        self.assertIsNone(admitted["refusal"])
        self.assertEqual(admitted["model"], "grok-4.6")
        self.assertEqual(admitted["effort"], "medium")
        self.assertEqual(admitted["effort_source"], "omni_default")

    def test_manual_record_raises(self):
        value = _live("claude", [{"id": "opus", "efforts": ["medium"]}], source="manual")
        with self.assertRaises(StrictRecordError) as caught:
            validate_strict("claude", value)
        self.assertEqual(caught.exception.reason, "catalog_unreadable")
        self.assertEqual(caught.exception.detail, "manual catalog records are not evidence")

    def test_discovery_field_is_kept(self):
        value = _live("claude", [{"id": "opus", "efforts": ["medium"]}])
        value["discovery"] = {"posture": "metadata-unfenced", "platform": "Linux"}
        record, warnings = validate_strict("claude", value)
        self.assertEqual(warnings, [])
        self.assertEqual(record["discovery"], {"posture": "metadata-unfenced", "platform": "Linux"})


if __name__ == "__main__":
    unittest.main()
