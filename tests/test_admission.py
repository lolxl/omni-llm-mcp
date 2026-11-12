"""Admission decisions from synthetic records and the 2026-11-10 cursor ids."""
import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from providers import model_catalog
from providers.admission import GROK_RULED_CLI_VERSION, TERMINAL_CATALOG_REFUSALS, decide
from providers.catalog_record import validate_strict
from providers.config import CURSOR_EFFORT_SLUGS, CURSOR_MODEL_ALIASES, EFFORT_LEVELS
from providers.cursor import resolve_model

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True


NOW = datetime.datetime(2026, 11, 10, 12, 0, tzinfo=datetime.timezone.utc)
FRESH = "2026-11-10T00:00:00Z"
STALE_AT = "2026-11-01T00:00:00Z"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "model_catalog_cursor_ids_20261110.txt"
CURSOR_IDS = (
    "claude-fable-5-2-low",
    "claude-fable-5-2-medium",
    "claude-fable-5-2-high",
    "claude-fable-5-2-xhigh",
    "claude-fable-5-2-max",
    "gpt-5.2",
    "gpt-5.2-low",
    "gpt-5.2-high",
    "gpt-5.2-xhigh",
    "gpt-5.2-fast",
    "gpt-5.5-low",
    "gpt-5.5-medium",
    "gpt-5.5-high",
    "gpt-5.5-extra-high",
    "composer-2.5",
)


def _record(provider, rows, **extra):
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
        "source": "live",
        "observed_at": extra.pop("observed_at", FRESH),
        "availability": "provider_advertised",
        "models": models,
    }
    value.update(extra)
    record, _warnings = validate_strict(provider, value)
    return record


def _cursor(ids, **extra):
    return _record("cursor", [{"id": model_id} for model_id in ids], **extra)


def _call(provider, model, record, effort=None, now=NOW):
    request = {"provider": provider, "model": model, "effort": effort}
    return decide(request, record, now=now)


class AdmissionTests(unittest.TestCase):
    def test_terminal_refusal_codes(self):
        self.assertEqual(TERMINAL_CATALOG_REFUSALS, frozenset((
            "no_catalog",
            "catalog_unreadable",
            "unsupported_model",
            "unsupported_effort",
            "effort_required",
            "effort_already_encoded",
            "ambiguous_model",
            "conflicting_evidence",
            "alias_file",
        )))

    def test_no_record_is_no_catalog(self):
        result = _call("claude", "opus", None)
        self.assertEqual(result["refusal"], "no_catalog")
        self.assertIsNone(result["model"])
        self.assertEqual(result["warnings"], [])

    def test_r5_floating_match(self):
        record = _record("claude", [{
            "id": "fable",
            "name": "Fable 5.2",
            "display_name": "Fable 5.2",
            "efforts": ["low", "medium", "high"],
        }])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "claude-fable-5-2")
        self.assertEqual(result["effort"], "medium")
        self.assertEqual(result["effort_source"], "omni_default")
        self.assertEqual(result["effort_note"],
                         "effort: medium (chosen by omni; no effort was given)")
        self.assertEqual(result["evidence_row"]["id"], "fable")
        self.assertEqual(result["warnings"], [])

    def test_r5_lagging_display_name(self):
        record = _record("claude", [{
            "id": "fable",
            "name": "Fable 5.1",
            "display_name": "Fable 5.1",
            "efforts": ["medium"],
        }])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertEqual(result["refusal"], "unsupported_model")
        self.assertIsNone(result["model"])

    def test_r5_conflicting_pinned_row(self):
        record = _record("claude", [{
            "id": "claude-fable-5-1",
            "name": "Fable 5.2",
            "display_name": "Fable 5.2",
            "efforts": ["medium"],
        }])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertEqual(result["refusal"], "conflicting_evidence")
        self.assertIsNone(result["model"])
        self.assertIsNone(result["evidence_row"])

    def test_r5_ambiguous_names(self):
        record = _record("claude", [
            {"id": "fable", "name": "Fable 5.2", "display_name": "Fable 5.2",
             "efforts": ["medium"]},
            {"id": "fable-alias", "name": "Fable 5.2", "display_name": "Fable 5.2",
             "efforts": ["medium"]},
        ])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertEqual(result["refusal"], "ambiguous_model")
        self.assertIsNone(result["model"])

    def test_r5_dated_id_matches_only_the_exact_row(self):
        floating = _record("claude", [{
            "id": "haiku",
            "name": "Haiku 4.5",
            "display_name": "Haiku 4.5",
            "efforts": ["medium"],
        }])
        refused = _call("claude", "claude-haiku-4-5-20251001", floating)
        self.assertEqual(refused["refusal"], "unsupported_model")
        self.assertIsNone(refused["model"])

        listed = _record("claude", [{
            "id": "claude-haiku-4-5-20251001",
            "name": "Haiku 4.5",
            "display_name": "Haiku 4.5",
            "efforts": ["low", "medium"],
        }])
        admitted = _call("claude", "claude-haiku-4-5-20251001", listed)
        self.assertIsNone(admitted["refusal"])
        self.assertEqual(admitted["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(admitted["effort"], "medium")
        self.assertEqual(admitted["effort_source"], "omni_default")
        self.assertEqual(admitted["evidence_row"]["id"], "claude-haiku-4-5-20251001")

    def test_r5_exact_row_with_mismatched_display_name_is_admitted(self):
        record = _record("claude", [{
            "id": "claude-fable-5-2",
            "name": "Fable 5.1",
            "display_name": "Fable 5.1",
            "efforts": ["low", "medium", "high"],
        }])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "claude-fable-5-2")
        self.assertEqual(result["effort"], "medium")
        self.assertEqual(result["effort_source"], "omni_default")
        self.assertEqual(result["evidence_row"]["id"], "claude-fable-5-2")
        self.assertEqual(result["evidence_row"]["display_name"], "Fable 5.1")

    def test_r5_padded_display_name_is_not_evidence(self):
        record = _record("claude", [{
            "id": "fable",
            "name": "Fable 5.2",
            "display_name": " Fable 5.2 ",
            "efforts": ["medium"],
        }])
        result = _call("claude", "claude-fable-5-2", record)
        self.assertEqual(result["refusal"], "unsupported_model")
        self.assertIsNone(result["model"])

    def test_cursor_fable_without_effort_folds_medium(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "fable", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "claude-fable-5-2-medium")
        self.assertEqual(result["effort"], "medium")
        self.assertEqual(result["effort_source"], "omni_default")
        self.assertEqual(result["effort_note"],
                         "effort: medium (chosen by omni; no effort was given)")
        self.assertEqual(result["evidence_row"]["id"], "claude-fable-5-2-medium")

    def test_cursor_fable_without_medium_listed_requires_effort(self):
        record = _cursor(("claude-fable-5-2-low", "claude-fable-5-2-high"))
        result = _call("cursor", "fable", record)
        self.assertEqual(result["refusal"], "effort_required")
        self.assertIsNone(result["model"])
        self.assertEqual(result["effort_note"], "effort required; admitted efforts: low, high")

    def test_cursor_gpt_5_2_at_high(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "gpt-5.2@high", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "gpt-5.2-high")
        self.assertEqual(result["effort"], "high")
        self.assertEqual(result["effort_source"], "request")
        self.assertEqual(result["effort_note"], "effort: high (as requested)")

    def test_cursor_gpt_5_5_at_xhigh_uses_the_override(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "gpt-5.5@xhigh", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "gpt-5.5-extra-high")
        self.assertEqual(result["effort"], "xhigh")
        self.assertEqual(result["effort_source"], "request")

    def test_cursor_composer_is_sent_verbatim(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "composer-2.5", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "composer-2.5")
        self.assertIsNone(result["effort"])
        self.assertEqual(result["effort_source"], "model_id")
        self.assertEqual(result["effort_note"], "effort: (encoded in the model id)")

    def test_cursor_listed_fable_is_sent_verbatim(self):
        record = _cursor(("fable", "claude-fable-5-2-medium"))
        result = _call("cursor", "fable", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "fable")
        self.assertIsNone(result["effort"])
        self.assertEqual(result["effort_source"], "model_id")

    def test_cursor_listed_gpt_5_5_xhigh_is_already_encoded(self):
        record = _cursor(("gpt-5.5", "gpt-5.5-extra-high"))
        result = _call("cursor", "gpt-5.5@xhigh", record)
        self.assertEqual(result["refusal"], "effort_already_encoded")
        self.assertIsNone(result["model"])
        self.assertEqual(result["effort_note"], "effort is already encoded in the model id")

    def test_cursor_exact_id_with_unlisted_fold_is_already_encoded(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "composer-2.5", record, effort="high")
        self.assertEqual(result["refusal"], "effort_already_encoded")
        self.assertIsNone(result["model"])
        self.assertEqual(result["effort_note"], "effort is already encoded in the model id")

    def test_cursor_fast_is_never_an_effort(self):
        record = _cursor(CURSOR_IDS)
        verbatim = _call("cursor", "gpt-5.2-fast", record)
        self.assertIsNone(verbatim["refusal"])
        self.assertEqual(verbatim["model"], "gpt-5.2-fast")
        self.assertIsNone(verbatim["effort"])
        self.assertEqual(verbatim["effort_source"], "model_id")
        refused = _call("cursor", "gpt-5.2", record, effort="fast")
        self.assertEqual(refused["refusal"], "unsupported_effort")
        self.assertIsNone(refused["model"])

    def test_cursor_unknown_name(self):
        record = _cursor(CURSOR_IDS)
        result = _call("cursor", "not-a-cursor-model", record)
        self.assertEqual(result["refusal"], "unsupported_model")
        self.assertIsNone(result["model"])

    def test_agy_id_name_collision_and_effort(self):
        named = _record("agy", [
            {"id": "gemini-3.8-flash-high", "name": "Gemini 3.8 Flash (High)",
             "display_name": "Gemini 3.8 Flash (High)"},
            {"id": "claude-opus-5-5-high", "name": "Claude Opus 5.5 (High)",
             "display_name": "Claude Opus 5.5 (High)"},
            {"id": "claude-opus-5-5-low", "name": "Claude Opus 5.5 (Low)",
             "display_name": "Claude Opus 5.5 (Low)"},
        ])
        by_id = _call("agy", "gemini-3.8-flash-high", named)
        self.assertIsNone(by_id["refusal"])
        self.assertEqual(by_id["model"], "gemini-3.8-flash-high")
        self.assertIsNone(by_id["effort"])
        self.assertEqual(by_id["effort_source"], "model_id")
        self.assertEqual(by_id["evidence_row"]["id"], "gemini-3.8-flash-high")

        by_name = _call("agy", "Gemini 3.8 Flash (High)", named)
        self.assertIsNone(by_name["refusal"])
        self.assertEqual(by_name["model"], "Gemini 3.8 Flash (High)")
        self.assertIsNone(by_name["effort"])
        self.assertEqual(by_name["effort_source"], "model_id")
        self.assertEqual(by_name["evidence_row"]["id"], "gemini-3.8-flash-high")

        collision = _record("agy", [
            {"id": "shared-id", "name": "Unique Name"},
            {"id": "other-id", "name": "shared-id"},
        ])
        collided = _call("agy", "shared-id", collision)
        self.assertEqual(collided["refusal"], "ambiguous_model")
        self.assertIsNone(collided["model"])

        encoded = _call("agy", "gemini-3.8-flash-high", named, effort="high")
        self.assertEqual(encoded["refusal"], "effort_already_encoded")
        self.assertIsNone(encoded["model"])

    def test_codex_deep_is_admitted_when_listed(self):
        record = _record("codex", [{
            "id": "gpt-test",
            "name": "Test",
            "display_name": "Test",
            "efforts": ["deep", "medium"],
        }])
        result = _call("codex", "gpt-test", record, effort="deep")
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "gpt-test")
        self.assertEqual(result["effort"], "deep")
        self.assertEqual(result["effort_source"], "request")
        self.assertEqual(result["effort_note"], "effort: deep (as requested)")
        self.assertEqual(result["evidence_row"]["supported_efforts"], ["medium", "deep"])

    def test_grok_recorded_vocabulary(self):
        record = _record("grok", [{"id": "grok-4.6", "absent": True}], effort_vocabulary={
            "source": "recorded",
            "cli_version": "1.0.52",
            "tokens": ["low", "medium", "high"],
        })
        self.assertEqual(record["effort_vocabulary"], {
            "source": "recorded",
            "cli_version": "1.0.52",
            "tokens": ["low", "medium", "high"],
        })
        omitted = _call("grok", "grok-4.6", record)
        self.assertIsNone(omitted["refusal"])
        self.assertEqual(omitted["model"], "grok-4.6")
        self.assertEqual(omitted["effort"], "medium")
        self.assertEqual(omitted["effort_source"], "omni_default")
        explicit = _call("grok", "grok-4.6", record, effort="high")
        self.assertIsNone(explicit["refusal"])
        self.assertEqual(explicit["effort"], "high")
        self.assertEqual(explicit["effort_source"], "request")
        refused = _call("grok", "grok-4.6", record, effort="xhigh")
        self.assertEqual(refused["refusal"], "unsupported_effort")
        self.assertIsNone(refused["model"])

    def test_grok_owner_ruled_vocabulary(self):
        self.assertEqual(GROK_RULED_CLI_VERSION, "1.0.79")
        record = _record("grok", [{"id": "grok-4.6", "absent": True}])
        self.assertNotIn("effort_vocabulary", record)
        admitted = _call("grok", "grok-4.6", record, effort="xhigh")
        self.assertIsNone(admitted["refusal"])
        self.assertEqual(admitted["model"], "grok-4.6")
        self.assertEqual(admitted["effort"], "xhigh")
        self.assertEqual(admitted["effort_source"], "request")
        self.assertEqual(admitted["effort_note"], "effort: xhigh (as requested)")
        refused = _call("grok", "grok-4.6", record, effort="max")
        self.assertEqual(refused["refusal"], "unsupported_effort")
        empty = _record("grok", [{"id": "grok-4.6", "efforts": [], "known": True}])
        empty_omitted = _call("grok", "grok-4.6", empty)
        self.assertIsNone(empty_omitted["refusal"])
        self.assertIsNone(empty_omitted["effort"])
        self.assertEqual(empty_omitted["effort_source"], "not_applicable")
        empty_explicit = _call("grok", "grok-4.6", empty, effort="medium")
        self.assertEqual(empty_explicit["refusal"], "unsupported_effort")
        omitted = _call("grok", "grok-4.6", record)
        self.assertEqual(omitted["effort"], "medium")
        self.assertEqual(omitted["effort_source"], "omni_default")
        self.assertEqual(omitted["effort_note"],
                         "effort: medium (chosen by omni; no effort was given)")

    def test_effort_field_states_with_and_without_an_explicit_effort(self):
        empty = _record("codex", [{"id": "gpt-test", "efforts": [], "known": True}])
        absent = _record("codex", [{"id": "gpt-test", "absent": True}])
        invalid_only = _record("codex", [{"id": "gpt-test", "efforts": ["NOT_A_TOKEN"]}])
        mixed = _record("codex", [{"id": "gpt-test", "efforts": ["low", "NOT_A_TOKEN", "medium"]}])

        self.assertEqual(empty["models"][0]["effort_field"], "empty")
        self.assertEqual(absent["models"][0]["effort_field"], "absent")
        self.assertEqual(invalid_only["models"][0]["effort_field"], "absent")
        self.assertEqual(mixed["models"][0]["supported_efforts"], ["low", "medium"])

        empty_omitted = _call("codex", "gpt-test", empty)
        self.assertIsNone(empty_omitted["refusal"])
        self.assertIsNone(empty_omitted["effort"])
        self.assertEqual(empty_omitted["effort_source"], "not_applicable")
        self.assertEqual(empty_omitted["effort_note"],
                         "effort: (not applicable; the listing gives an empty effort field)")
        empty_explicit = _call("codex", "gpt-test", empty, effort="high")
        self.assertEqual(empty_explicit["refusal"], "unsupported_effort")

        absent_omitted = _call("codex", "gpt-test", absent)
        self.assertIsNone(absent_omitted["refusal"])
        self.assertIsNone(absent_omitted["effort"])
        self.assertEqual(absent_omitted["effort_source"], "vendor")
        self.assertEqual(absent_omitted["effort_note"], "effort: (listing has no effort info)")
        absent_explicit = _call("codex", "gpt-test", absent, effort="high")
        self.assertEqual(absent_explicit["refusal"], "unsupported_effort")

        invalid_omitted = _call("codex", "gpt-test", invalid_only)
        self.assertIsNone(invalid_omitted["refusal"])
        self.assertIsNone(invalid_omitted["effort"])
        self.assertEqual(invalid_omitted["effort_source"], "vendor")
        invalid_explicit = _call("codex", "gpt-test", invalid_only, effort="medium")
        self.assertEqual(invalid_explicit["refusal"], "unsupported_effort")

        mixed_omitted = _call("codex", "gpt-test", mixed)
        self.assertIsNone(mixed_omitted["refusal"])
        self.assertEqual(mixed_omitted["effort"], "medium")
        self.assertEqual(mixed_omitted["effort_source"], "omni_default")
        mixed_explicit = _call("codex", "gpt-test", mixed, effort="low")
        self.assertIsNone(mixed_explicit["refusal"])
        self.assertEqual(mixed_explicit["effort"], "low")
        self.assertEqual(mixed_explicit["effort_source"], "request")
        mixed_refused = _call("codex", "gpt-test", mixed, effort="high")
        self.assertEqual(mixed_refused["refusal"], "unsupported_effort")

        no_medium = _record("claude", [{"id": "opus", "efforts": ["low", "high"]}])
        self.assertEqual(no_medium["models"][0]["supported_efforts"], ["low", "high"])
        required = _call("claude", "opus", no_medium)
        self.assertEqual(required["refusal"], "effort_required")
        self.assertIsNone(required["model"])
        self.assertIsNone(required["effort"])

    def test_stale_record_is_admitted_with_a_warning(self):
        record = _record("claude", [{
            "id": "opus",
            "name": "Opus 5.5",
            "display_name": "Opus 5.5",
            "efforts": ["low", "medium", "high"],
        }], observed_at=STALE_AT)
        result = _call("claude", "opus", record)
        self.assertIsNone(result["refusal"])
        self.assertEqual(result["model"], "opus")
        self.assertEqual(result["effort"], "medium")
        self.assertEqual(result["effort_source"], "omni_default")
        self.assertEqual(result["warnings"], ["stale"])

    def test_cursor_fixture_matches_resolve_model_when_the_id_is_listed(self):
        ids = [
            line.strip() for line in FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(ids[0], "auto")
        self.assertEqual(ids[-1], "deepseek-v41-flash-1017-max")
        self.assertEqual(len(ids), 268)
        lines = ["Available models"]
        for model_id in ids:
            lines.append("%s - %s" % (model_id, model_id))
        lines.append("Tip: fixture")
        text = "\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory(prefix="omni-d2b1-") as state:
            with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}):
                parsed = model_catalog.parse_cursor_output(text, observed_at=FRESH)
                from providers import catalog_v2
                catalog_v2.save("cursor", "default", parsed)
                loaded, info = catalog_v2.load("cursor", "default")
                self.assertIsNotNone(loaded, info)
                self.assertEqual([row["id"] for row in loaded["models"]], ids)
                listed = set(ids)
                bases = set(CURSOR_EFFORT_SLUGS) | set(CURSOR_MODEL_ALIASES)
                for model_id in ids:
                    for word in EFFORT_LEVELS:
                        suffix = "-" + word
                        if model_id.endswith(suffix) and model_id[:-len(suffix)]:
                            bases.add(model_id[:-len(suffix)])
                checked = 0
                for base in sorted(bases):
                    for word in EFFORT_LEVELS:
                        slug, err = resolve_model(base, word)
                        if err or slug not in listed:
                            continue
                        result = _call("cursor", base, loaded, effort=word)
                        self.assertIsNone(result["refusal"], (base, word, result))
                        self.assertEqual(result["model"], slug, (base, word))
                        checked += 1
        self.assertEqual(checked, 141)


class FileEffortWithoutEvidenceTests(unittest.TestCase):
    """A file ``effort.codex`` with no row evidence is a catalog refusal."""

    def _consider(self, efforts):
        from providers import admission, catalog_v2
        row = {"id": "gpt-bare"}
        if efforts is not None:
            row["efforts"] = efforts
        with tempfile.TemporaryDirectory(prefix="omni-effort-") as state:
            with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}):
                for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
                    os.environ.pop(key, None)
                catalog_v2.save("codex", "default", _record("codex", [row]))
                path = os.path.join(state, "aliases.json")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write('{"effort.codex": "xhigh"}\n')
                filed = admission.consider(
                    "codex", "default", "gpt-bare", None, host_env={},
                    settings_files=(), apply_aliases=True, now=NOW)
                os.remove(path)
                builtin = admission.consider(
                    "codex", "default", "gpt-bare", None, host_env={},
                    settings_files=(), apply_aliases=True, now=NOW)
        return filed, builtin

    def test_absent_and_empty_rows_refuse_the_file_effort(self):
        for efforts in (None, []):
            filed, builtin = self._consider(efforts)
            self.assertEqual(filed.get("refusal"), "unsupported_effort", filed)
            self.assertIsNone(filed.get("effort"))
            self.assertIn("xhigh", filed.get("effort_note") or "")
            self.assertIsNone(builtin.get("refusal"), builtin)
            self.assertIsNone(builtin.get("effort"))


if __name__ == "__main__":
    unittest.main()
