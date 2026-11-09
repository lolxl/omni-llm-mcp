"""Hermetic AGY Opus shorthand checks; no CLI, cache, fence, or model request."""
import asyncio
import copy
import unittest
from unittest import mock

from providers import dispatch
from providers import model_catalog
from providers.seat_name import take_effort


def catalog_target():
    return next(row for row in model_catalog.bundled_catalog()["agy"]["models"]
                if row["name"] == "Claude Opus 5.5 (Medium)")


class TestAgyAlias(unittest.TestCase):
    def test_exact_requested_shorthand_uses_existing_bundled_canonical_id(self):
        canonical = catalog_target()["routable_id"]
        self.assertEqual(take_effort("agy-opus-5.5@med", None),
                         (canonical, "medium", None))

    def test_explicit_effort_and_normalization_are_idempotent(self):
        canonical = catalog_target()["routable_id"]
        first = take_effort("agy-opus-5.5", "medium")
        self.assertEqual(first, (canonical, "medium", None))
        self.assertEqual(take_effort(first[0], first[1]), first)

    def test_original_canonical_and_other_provider_names_stay_unchanged(self):
        canonical = catalog_target()["routable_id"]
        self.assertEqual(take_effort(canonical, None), (canonical, None, None))
        self.assertEqual(take_effort("codex/example@high", None),
                         ("codex/example", "high", None))
        self.assertEqual(take_effort("agy-sonnet-5.5@medium", None),
                         ("agy-sonnet-5.5", "medium", None))

    def test_duplicate_suffix_and_explicit_effort_is_still_rejected(self):
        for effort in ("medium", "high"):
            with self.subTest(effort=effort):
                self.assertEqual(take_effort("agy-opus-5.5@med", effort)[2]["reason"],
                                 "unsupported_effort")

    def exercise_dispatch(self, model, effort=None):
        seen = []
        async def fake_dispatch(prompt, selected, timeout_s, files, profile, **kwargs):
            seen.append((selected, kwargs.get("effort")))
            return {"ok": True, "text": "synthetic", "model": selected}
        with mock.patch.object(dispatch, "_health_maintain") as health, \
             mock.patch.object(dispatch, "_apply_profile_rings", side_effect=lambda seats, profile: (seats, None)) as admission, \
             mock.patch.object(dispatch, "_want_read_wall", return_value=False), \
             mock.patch.object(dispatch, "backends_snapshot") as snapshot, \
             mock.patch.object(dispatch, "dispatch_single", side_effect=fake_dispatch) as spawn, \
             mock.patch.object(dispatch.health, "lease_acquire") as lease:
            result = asyncio.run(dispatch._dispatch_llm_request(
                "synthetic harmless prompt", model, 30, [], "default", effort=effort))
            return result, seen, (health.call_count, admission.call_count, snapshot.call_count,
                                  spawn.call_count, lease.call_count), admission.call_args

    def test_seats_and_dispatch_see_only_the_canonical_model(self):
        canonical = catalog_target()["routable_id"]
        result, seen, counts, admission = self.exercise_dispatch("agy-opus-5.5@med")
        self.assertTrue(result["ok"], result)
        self.assertEqual(admission.args[0][0]["model"], canonical)
        self.assertEqual(seen, [(canonical, "medium")])
        self.assertEqual(result["seat"]["model"], canonical)
        self.assertEqual(counts, (1, 1, 0, 1, 0))

    def test_invalid_aliases_stop_before_admission_health_or_spawn(self):
        for name, effort in [("agy-opus-99.99@med", None), ("agy-opus-latest@med", None),
                             ("agy-opus-5.5@minimal", None), ("agy-opus-5.5@turbo", None),
                             ("agy-opus-5.5@med", "medium"), ("agy-opus-5.5", None),
                             ("agy-opus-99.99@med,agy/known-control", None),
                             ("agy/known-control,agy-opus-99.99@med", None)]:
            with self.subTest(name=name, effort=effort):
                result, seen, counts, _ = self.exercise_dispatch(name, effort)
                self.assertFalse(result["ok"])
                self.assertIn(result["reason"], ("unsupported_model", "unsupported_effort"))
                self.assertEqual(seen, [])
                self.assertEqual(counts, (0, 0, 0, 0, 0))

    def test_missing_effort_variant_never_selects_another_version_or_provider(self):
        bundle = copy.deepcopy(model_catalog.bundled_catalog())
        bundle["agy"]["models"] = [row for row in bundle["agy"]["models"]
                                    if row["name"] != "Claude Opus 5.5 (Medium)"]
        with mock.patch.object(model_catalog, "bundled_catalog", return_value=bundle):
            result, seen, counts, _ = self.exercise_dispatch("agy-opus-5.5@med")
        self.assertFalse(result["ok"])
        self.assertEqual(seen, [])
        self.assertEqual(counts, (0, 0, 0, 0, 0))

    def test_ambiguous_bundled_matches_fail_before_admission(self):
        bundle = copy.deepcopy(model_catalog.bundled_catalog())
        duplicate = copy.deepcopy(catalog_target())
        duplicate["id"] = "synthetic-ambiguous-opus-medium"
        duplicate["routable_id"] = "agy/" + duplicate["id"]
        bundle["agy"]["models"].append(duplicate)
        with mock.patch.object(model_catalog, "bundled_catalog", return_value=bundle):
            result, seen, counts, _ = self.exercise_dispatch("agy-opus-5.5@med")
        self.assertFalse(result["ok"])
        self.assertEqual(seen, [])
        self.assertEqual(counts, (0, 0, 0, 0, 0))

    def test_alias_honors_existing_canonical_cooldown_in_a_chain(self):
        canonical = catalog_target()["routable_id"]
        seen, health_models = [], []
        async def fake_dispatch(prompt, model, timeout_s, files, profile, **kwargs):
            seen.append(model)
            return {"ok": True, "text": "synthetic", "model": model}
        def kind(model, profile):
            health_models.append(model)
            return "cooling" if model == canonical else "healthy"
        with mock.patch.object(dispatch, "_health_maintain"), \
             mock.patch.object(dispatch, "_apply_profile_rings", side_effect=lambda seats, profile: (seats, None)), \
             mock.patch.object(dispatch, "_want_read_wall", return_value=False), \
             mock.patch.object(dispatch, "resolve_profile", return_value=({}, None)), \
             mock.patch.object(dispatch, "_health_passive_probe"), \
             mock.patch.object(dispatch, "_health_kind", side_effect=kind), \
             mock.patch.object(dispatch, "log_event"), \
             mock.patch.object(dispatch, "dispatch_single", side_effect=fake_dispatch):
            result = asyncio.run(dispatch._dispatch_llm_request(
                "synthetic harmless prompt", "agy-opus-5.5@med,agy/synthetic-healthy-control",
                30, [], "default"))
        self.assertTrue(result["ok"], result)
        self.assertIn(canonical, health_models)
        self.assertNotIn("agy-opus-5.5", health_models)
        self.assertEqual(seen, ["agy/synthetic-healthy-control"])
        self.assertEqual(result["tried"][0]["model"], canonical)
        self.assertEqual(result["tried"][0]["reason"], "skipped_cooldown")

    def test_canonical_control_reaches_vendor_with_exact_model_and_effort(self):
        canonical = catalog_target()
        async def fake_agy(prompt, model, timeout_s, files, profile, **kwargs):
            return {"ok": True, "text": "synthetic", "model": model, "effort": kwargs.get("effort")}
        with mock.patch.object(dispatch.agy_mod, "run_agy", side_effect=fake_agy), \
             mock.patch.object(dispatch, "env_depth", return_value=0), \
             mock.patch.object(dispatch, "refuse_payload", return_value=None):
            result = asyncio.run(dispatch.route_to_provider(
                "synthetic harmless prompt", "agy-opus-5.5@med", 30, [], "default"))
        self.assertEqual(result.get("model"), canonical["id"])
        self.assertEqual(result.get("effort"), "medium")


if __name__ == "__main__":
    unittest.main(verbosity=2)
