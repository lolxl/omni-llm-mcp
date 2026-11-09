"""Hermetic model catalog parsers, cache operations, profiles, and protocols."""
import datetime
import asyncio
import io
import json
import os
import plistlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from providers import config, model_catalog
from providers.route import resolve_route


HERE = Path(__file__).resolve().parent.parent
FIXTURES = HERE / "tests" / "fixtures"


def _verified_read_only_fence_metadata():
    return {"fence": "seatbelt-fs",
            "fence_detail": {"read_wall": True,
                             "workdir_writable": False,
                             "vendor_state_writable": False}}


class PipeCaptureStdin:
    """Capture writes through a real selectable OS pipe descriptor."""

    def __init__(self):
        self._read_fd, write_fd = os.pipe()
        self._writer = os.fdopen(write_fd, "wb", buffering=0)

    def fileno(self):
        return self._writer.fileno()

    def close(self):
        self._writer.close()

    def read_all(self):
        self.close()
        chunks = []
        while True:
            chunk = os.read(self._read_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        self.close_capture()
        return b"".join(chunks)

    def close_capture(self):
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp(prefix="omni-model-catalog-")
        self.work = tempfile.mkdtemp(prefix="omni-model-work-")
        self.env_patch = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state,
            "OMNI_LLM_NOTIFY_QUIET": "1",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.addCleanup(shutil.rmtree, self.state, True)
        self.addCleanup(shutil.rmtree, self.work, True)

    def write_json(self, name, value):
        path = Path(self.work) / name
        with open(path, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return path

    def manual_payload(self, provider, model_id, *, observed_at=None, efforts=None):
        value = {"provider": provider, "models": [{
            "id": model_id,
            "name": model_id,
            "supported_efforts": list(efforts or []),
        }]}
        if observed_at is not None:
            value["observed_at"] = observed_at
        return value

    def test_sanitized_captures_parse_for_all_five_providers(self):
        cursor = model_catalog.parse_cursor_output(
            (FIXTURES / "model_catalog_cursor_models.txt").read_text(encoding="utf-8"),
            observed_at="2026-10-16T01:51:28.408634Z", cli_version="2026.10.01-e373342")
        agy = model_catalog.parse_agy_output(
            (FIXTURES / "model_catalog_agy_models.tsv").read_text(encoding="utf-8"),
            observed_at="2026-10-16T01:51:29.564387Z", cli_version="1.3.3")
        codex = model_catalog.parse_codex_result(json.loads(
            (FIXTURES / "model_catalog_codex_model_list.json").read_text(encoding="utf-8")),
            observed_at="2026-10-16T02:12:36.020149Z", cli_version="0.162.1")
        claude = model_catalog.parse_claude_models(json.loads(
            (FIXTURES / "model_catalog_claude_initialize.json").read_text(encoding="utf-8"))["models"],
            observed_at="2026-10-16T02:06:25.779546Z", cli_version="2.1.301")
        grok = model_catalog.parse_grok_output(
            (FIXTURES / "model_catalog_grok_models.txt").read_text(encoding="utf-8"),
            observed_at="2026-10-16T02:06:26.123907Z", cli_version="1.0.52")

        self.assertEqual([len(item["models"]) for item in
                          (cursor, agy, codex, claude, grok)], [265, 13, 7, 13, 4])
        self.assertEqual(grok["availability"], "unknown")
        self.assertEqual(grok["source_note"], "vendor_registry_not_entitlement")
        self.assertEqual(claude["availability"], "provider_advertised")
        self.assertEqual(codex["models"][0]["default_effort"], "low")
        self.assertIn("gpt-oss-120b-medium", {row["id"] for row in agy["models"]})

    def test_unauthenticated_grok_listing_fails_without_replacing_last_good(self):
        unauthenticated = (FIXTURES / "model_catalog_grok_unauthenticated.txt").read_text(
            encoding="utf-8")
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog.parse_grok_output(unauthenticated)
        self.assertEqual(caught.exception.reason, "auth_error")

        manual = self.write_json("grok-last-good.json", self.manual_payload(
            "grok", "grok-last-good", observed_at="2026-10-01T00:00:00Z"))
        self.assertTrue(model_catalog.manual_import("grok", "default", manual)["ok"])
        cache_path = model_catalog._cache_paths("grok", "default")[0]
        before = cache_path.read_bytes()

        def unauthenticated_discovery(_profile, _timeout):
            return model_catalog.parse_grok_output(unauthenticated)

        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {"grok": unauthenticated_discovery}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
            result = model_catalog.refresh("grok", "default", 5)
        self.assertEqual(result["providers"]["grok"]["reason"], "auth_error")
        self.assertEqual(cache_path.read_bytes(), before)

    def test_refresh_all_returns_every_road_when_profiles_and_cache_fail(self):
        bundle = model_catalog.bundled_catalog()
        home = Path(self.work) / ".codex-work"
        home.mkdir()

        def resolve(provider, profile):
            if provider == "codex" and profile == "work":
                return {"CODEX_HOME": str(home)}, None
            return {}, {"reason": "unsupported_profile"}

        discoverers = {}
        for name in model_catalog.PROVIDERS:
            if name == "codex":
                discoverers[name] = lambda _profile, _timeout: bundle["codex"]
            else:
                discoverers[name] = lambda _profile, _timeout, road=name: (_ for _ in ()).throw(
                    model_catalog.CatalogError("unsupported_profile", road))
        with mock.patch("providers.profiles.resolve_profile", side_effect=resolve), \
                mock.patch.dict(model_catalog._DISCOVERERS, discoverers), \
                mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
            result = model_catalog.refresh("all", "work", 5)
        self.assertEqual(set(result["providers"]), set(model_catalog.PROVIDERS))
        self.assertTrue(result["providers"]["codex"]["ok"])
        for name in ("cursor", "agy", "claude", "grok"):
            with self.subTest(provider=name):
                self.assertFalse(result["providers"][name]["ok"])
                self.assertEqual(result["providers"][name]["reason"], "unsupported_profile")

    def test_missing_agy_named_profile_refuses_before_process_spawn(self):
        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": ""}), \
                mock.patch("providers.profiles.resolve_profile", return_value=({}, None)), \
                mock.patch.object(model_catalog.subprocess, "Popen") as popen, \
                mock.patch("providers.host_fence.prepare_host_fence") as prepare:
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._spawn_one_shot("agy", ["/usr/bin/agy", "models"],
                                              "work", 2)
        self.assertEqual(caught.exception.reason, "unsupported_profile")
        popen.assert_not_called()
        prepare.assert_not_called()

    def test_metadata_adapters_refuse_unverified_fences_before_launch(self):
        child_env = {"PATH": "/usr/bin:/bin"}
        identity_result = ({}, child_env, {}, "", {})
        invalid_metadata = (
            ("missing", {"fence_detail": {"read_wall": True}}),
            ("none", {"fence": "none"}),
            ("unknown", {"fence": "sandbox-exec"}),
        )
        for label, metadata in invalid_metadata:
            with self.subTest(metadata=label):
                def prepare(argv, **kwargs):
                    return list(argv), kwargs.get("env", child_env), metadata

                with mock.patch.object(model_catalog, "_catalog_child_environment",
                                       return_value=identity_result), \
                        mock.patch("providers.host_fence.prepare_host_fence",
                                   side_effect=prepare), \
                        mock.patch.object(model_catalog, "_run_bounded") as bounded:
                    with self.assertRaises(model_catalog.CatalogError) as caught:
                        model_catalog._spawn_one_shot(
                            "agy", ["/usr/bin/agy", "models"], "default", 2)
                    self.assertEqual(caught.exception.reason, "discovery_unavailable")
                    bounded.assert_not_called()

                with mock.patch("providers.host_fence.prepare_host_fence",
                                side_effect=prepare), \
                        mock.patch.object(model_catalog.subprocess, "Popen") as popen:
                    with self.assertRaises(model_catalog.CatalogError) as caught:
                        model_catalog._query_codex_models(
                            "/usr/bin/codex", env=child_env, cwd=self.work,
                            timeout_s=2)
                    self.assertEqual(caught.exception.reason, "discovery_unavailable")
                    popen.assert_not_called()

                with mock.patch.object(model_catalog, "_catalog_child_environment",
                                       return_value=identity_result), \
                        mock.patch("providers.kernel_sandbox.resolve_agent_binary",
                                   return_value="/usr/bin/claude"), \
                        mock.patch("providers.host_fence.prepare_host_fence",
                                   side_effect=prepare), \
                        mock.patch.object(model_catalog.subprocess, "Popen") as popen:
                    with self.assertRaises(model_catalog.CatalogError) as caught:
                        model_catalog._discover_claude("default", 2)
                    self.assertEqual(caught.exception.reason, "discovery_unavailable")
                    popen.assert_not_called()

    def test_unverified_metadata_fence_preserves_cache_and_manual_import_works(self):
        initial = self.write_json("cursor-fenced-last-good.json", self.manual_payload(
            "cursor", "cursor-fenced-last-good"))
        self.assertTrue(model_catalog.manual_import(
            "cursor", "default", initial)["ok"])
        cache_path = model_catalog._cache_paths("cursor", "default")[0]
        last_good = cache_path.read_bytes()
        cursor_home = Path(self.work) / "cursor-config"
        cursor_home.mkdir()

        def prepare(argv, *, env, **_kwargs):
            return list(argv), env, {"fence": "none"}

        with mock.patch.object(model_catalog, "_catalog_child_environment",
                               return_value=({}, {"PATH": "/usr/bin",
                                                 "CURSOR_CONFIG_DIR": str(cursor_home)},
                                             {}, str(cursor_home), {})), \
                mock.patch("providers.cursor.resolve_bin",
                           return_value="/usr/bin/cursor-agent"), \
                mock.patch("providers.cursor.resolve_config_dir",
                           return_value=str(cursor_home)), \
                mock.patch("providers.host_fence.prepare_host_fence",
                           side_effect=prepare), \
                mock.patch.object(model_catalog, "_run_bounded") as bounded, \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  return_value=None):
            refreshed = model_catalog.refresh("cursor", "default", 5)
        self.assertFalse(refreshed["ok"], refreshed)
        self.assertEqual(refreshed["providers"]["cursor"]["reason"],
                         "discovery_unavailable")
        bounded.assert_not_called()
        self.assertEqual(cache_path.read_bytes(), last_good)

        manual = self.write_json("cursor-manual-after-fence-refusal.json",
                                 self.manual_payload("cursor", "cursor-manual-after-refusal"))
        imported = model_catalog.manual_import("cursor", "default", manual)
        self.assertTrue(imported["ok"], imported)
        cached, _info = model_catalog._load_cached("cursor", "default")
        self.assertEqual(cached["models"][0]["id"], "cursor-manual-after-refusal")

    def test_cursor_metadata_child_keeps_vendor_config_env_after_allowlist(self):
        from providers.host_fence import allowlisted_child_env
        config_home = Path(self.work) / "cursor-home"
        config_home.mkdir()
        captured = {}

        def fenced(argv, *, env, **_kwargs):
            return argv, allowlisted_child_env(env, "cursor"), \
                _verified_read_only_fence_metadata()

        def bounded(argv, *, env, **_kwargs):
            captured.update(argv=argv, env=env)
            return b"cursor metadata\n"

        with mock.patch.dict(os.environ, {
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": str(config_home)}), \
                mock.patch("providers.host_fence.prepare_host_fence", side_effect=fenced), \
                mock.patch.object(model_catalog, "_run_bounded", side_effect=bounded):
            output = model_catalog._spawn_one_shot(
                "cursor", ["/usr/bin/cursor-agent", "--list-models"],
                "work", 2, cursor_config=True)
        self.assertEqual(output, "cursor metadata\n")
        self.assertEqual(captured["env"]["CURSOR_CONFIG_DIR"], str(config_home))

    def test_named_cursor_home_precedes_global_vendor_config_for_cache_and_child(self):
        from providers.host_fence import allowlisted_child_env

        global_home = Path(self.work) / "cursor-global-home"
        named_home_one = Path(self.work) / "cursor-work-one"
        named_home_two = Path(self.work) / "cursor-work-two"
        for path in (global_home, named_home_one, named_home_two):
            path.mkdir()
        payload = self.write_json("cursor-work-catalog.json", self.manual_payload(
            "cursor", "named-work-only"))
        captured = {}

        def fenced(argv, *, env, **_kwargs):
            return argv, allowlisted_child_env(env, "cursor"), \
                _verified_read_only_fence_metadata()

        def bounded(argv, *, env, **_kwargs):
            captured.update(argv=argv, env=env)
            return b"cursor metadata\n"

        with mock.patch.dict(os.environ, {
                "CURSOR_CONFIG_DIR": str(global_home),
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": str(named_home_one)}):
            imported = model_catalog.manual_import("cursor", "work", payload)
            first_path = model_catalog._cache_paths("cursor", "work")[0]
            first_status = model_catalog.status("cursor", "work", details=True)
            self.assertTrue(imported["ok"], imported)
            self.assertEqual(first_status["catalogs"][0]["origin"], "local")
            self.assertEqual(first_status["catalogs"][0]["models"][0]["id"],
                             "named-work-only")

            os.environ["OMNI_LLM_CURSOR_CONFIG_DIR_WORK"] = str(named_home_two)
            second_path = model_catalog._cache_paths("cursor", "work")[0]
            identity, _overrides = model_catalog._profile_identity_inputs("cursor", "work")
            second_status = model_catalog.status("cursor", "work", details=True)
            with mock.patch("providers.host_fence.prepare_host_fence", side_effect=fenced), \
                    mock.patch.object(model_catalog, "_run_bounded", side_effect=bounded):
                output = model_catalog._spawn_one_shot(
                    "cursor", ["/usr/bin/cursor-agent", "--list-models"],
                    "work", 2, cursor_config=True)

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(identity["config_home"], os.path.realpath(named_home_two))
        self.assertEqual(second_status["catalogs"][0]["origin"], "bundled")
        self.assertNotIn("named-work-only", {
            row["id"] for row in second_status["catalogs"][0]["models"]})
        self.assertEqual(output, "cursor metadata\n")
        self.assertEqual(captured["env"]["CURSOR_CONFIG_DIR"], str(named_home_two))

    def test_default_cursor_identity_tracks_child_opt_in_precedence(self):
        from providers.cursor import resolve_config_dir
        from providers.profiles import resolve_profile, spawn_env

        vendor_home = Path(self.work) / "cursor-vendor-home"
        opt_in_one = Path(self.work) / "cursor-opt-in-one"
        opt_in_two = Path(self.work) / "cursor-opt-in-two"
        for path in (vendor_home, opt_in_one, opt_in_two):
            path.mkdir()
        payload = self.write_json("cursor-default-catalog.json", self.manual_payload(
            "cursor", "default-opt-in-only"))

        def selected_child_home():
            overrides, error = resolve_profile("cursor", "default")
            self.assertIsNone(error)
            child_env = spawn_env(os.environ.copy(), overrides, scrub_siblings=True)
            return resolve_config_dir(child_env)

        with mock.patch.dict(os.environ, {
                "CURSOR_CONFIG_DIR": str(vendor_home),
                "OMNI_LLM_CURSOR_CONFIG_DIR": str(opt_in_one),
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": ""}):
            imported = model_catalog.manual_import("cursor", "default", payload)
            first_path = model_catalog._cache_paths("cursor", "default")[0]
            first_identity, _ = model_catalog._profile_identity_inputs("cursor", "default")
            first_status = model_catalog.status("cursor", "default", details=True)
            first_child_home = selected_child_home()
            os.environ["OMNI_LLM_CURSOR_CONFIG_DIR"] = str(opt_in_two)
            second_path = model_catalog._cache_paths("cursor", "default")[0]
            second_identity, _ = model_catalog._profile_identity_inputs("cursor", "default")
            second_status = model_catalog.status("cursor", "default", details=True)
            second_child_home = selected_child_home()

        self.assertTrue(imported["ok"], imported)
        self.assertEqual(first_identity["config_home"], os.path.realpath(first_child_home))
        self.assertEqual(first_child_home, str(opt_in_one))
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(second_identity["config_home"], os.path.realpath(second_child_home))
        self.assertEqual(second_child_home, str(opt_in_two))
        self.assertEqual(first_status["catalogs"][0]["origin"], "local")
        self.assertEqual(second_status["catalogs"][0]["origin"], "bundled")
        self.assertNotIn("default-opt-in-only", {
            row["id"] for row in second_status["catalogs"][0]["models"]})

    def test_cursor_api_key_only_profile_identity_matches_child_without_inherited_home(self):
        from providers.cursor import resolve_config_dir
        from providers.profiles import resolve_profile, spawn_env

        global_home = Path(self.work) / "cursor-api-key-global"
        global_home.mkdir()
        with mock.patch.dict(os.environ, {
                "CURSOR_CONFIG_DIR": str(global_home),
                "OMNI_LLM_CURSOR_CONFIG_DIR": str(global_home),
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": "",
                "CURSOR_API_KEY_WORK": "test-only-key"}):
            overrides, error = resolve_profile("cursor", "work")
            self.assertIsNone(error)
            child_env = spawn_env(os.environ.copy(), overrides, scrub_siblings=True)
            selected_home = resolve_config_dir(child_env)
            identity, _ = model_catalog._profile_identity_inputs("cursor", "work")
        self.assertIsNone(selected_home)
        self.assertEqual(identity["config_home"], "")

    def test_status_check_and_effort_reads_do_not_create_catalog_state(self):
        state = Path(self.work) / "read-only-state"
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": str(state)}):
            status = model_catalog.status("cursor", "default")
            checked = model_catalog.check("cursor", "default")
            slug, error = config.cursor_effort_slug("grok-4.7", "high", "default")
        self.assertTrue(status["ok"])
        row = status["catalogs"][0]
        self.assertEqual(row["origin"], "bundled")
        self.assertEqual(row["cache_diagnostic"], "cache_identity_unavailable")
        self.assertEqual(checked["providers"][0]["reason"], "cache_identity_unavailable")
        self.assertEqual((slug, error), ("grok-4.7-high", None))
        self.assertFalse(state.exists())

    def test_list_models_keeps_legacy_envelope_when_catalog_read_fails(self):
        from providers import dispatch

        def attach_health(value):
            value["health"] = {"probe": "retained"}
            return value

        with mock.patch("providers.agy.find_agy", return_value=None), \
                mock.patch.object(model_catalog, "status", side_effect=PermissionError("denied")), \
                mock.patch.object(dispatch, "_with_health", side_effect=attach_health):
            result = asyncio.run(dispatch.list_models())
        self.assertEqual(result["reason"], "cli_missing")
        self.assertIn("backends", result)
        self.assertEqual(result["health"], {"probe": "retained"})
        self.assertEqual(result["catalogs"]["reason"], "cache_unavailable")

    def test_refresh_records_best_effort_cli_version_without_losing_listing(self):
        record = dict(model_catalog.bundled_catalog()["codex"])
        record.pop("cli_version", None)
        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {"codex": lambda _profile, _timeout: record}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  return_value="9.8.7"):
            result = model_catalog.refresh("codex", "default", 5)
        self.assertTrue(result["providers"]["codex"]["ok"])
        cached, _info = model_catalog._load_cached("codex", "default")
        self.assertEqual(cached["cli_version"], "9.8.7")

        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {"codex": lambda _profile, _timeout: record}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  side_effect=model_catalog.CatalogError("timeout")):
            result = model_catalog.refresh("codex", "default", 5)
        self.assertTrue(result["providers"]["codex"]["ok"])
        cached, _info = model_catalog._load_cached("codex", "default")
        self.assertNotIn("cli_version", cached)

    def test_snapshot_accepts_os_tmp_alias_but_still_rejects_target_symlink(self):
        temp_root = Path(tempfile.gettempdir())
        root = temp_root / "path-alias-test"
        root.mkdir(parents=True, exist_ok=True)
        tmp_real = Path(os.path.realpath("/tmp"))
        try:
            relative = temp_root.relative_to(tmp_real)
            requested_root = Path("/tmp") / relative
        except ValueError:
            requested_root = temp_root
        requested = requested_root / "path-alias-test" / "export.json"
        result = model_catalog.snapshot(requested, "cursor", "default")
        self.assertTrue(result["ok"], result)
        self.assertTrue(Path(result["output"]).is_file())

        target = root / "real.json"
        target.write_text("{}\n", encoding="utf-8")
        link = root / "link.json"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        refused = model_catalog.snapshot(link, "cursor", "default")
        self.assertEqual(refused["reason"], "output_path_invalid")

    def test_codex_only_profile_schedule_pins_road_and_safe_bindings(self):
        home = Path(self.work) / ".codex-work"
        home.mkdir()
        env = {
            "PATH": "/usr/bin:/bin",
            "OMNI_LLM_STATE_DIR": self.state,
            "CODEX_HOME_WORK": str(home),
            "CODEX_API_BASE": "https://api.example/v1",
            "CODEX_API_BASE_WORK": "https://alice:password@work.example/v1?token=hidden",
            "OPENAI_API_KEY_WORK": "secret-key",
            "XAI_API_KEY": "another-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = model_catalog.schedule_plist(
                "work", 86400, 120, provider="codex")
        self.assertTrue(result["ok"], result)
        plist = plistlib.loads(result["plist"].encode("utf-8"))
        self.assertEqual(plist["Label"], "com.omni-llm.model-catalogs.codex.work")
        self.assertEqual(plist["ProgramArguments"][5], "codex")
        bindings = plist["EnvironmentVariables"]
        self.assertEqual(bindings["CODEX_HOME_WORK"], os.path.realpath(home))
        self.assertNotIn("CODEX_API_BASE", bindings)
        self.assertNotIn("CODEX_API_BASE_WORK", bindings)
        self.assertNotIn("CURSOR_CONFIG_DIR", bindings)
        self.assertNotIn("OPENAI_API_KEY_WORK", bindings)
        self.assertNotIn("secret-key", result["plist"])
        self.assertNotIn("password", result["plist"])
        self.assertNotIn("hidden", result["plist"])
        self.assertNotIn("another-secret", result["plist"])

    def test_cursor_effort_union_uses_exact_observed_slugs(self):
        expected = (
            ("claude-opus-5-5", "high", "claude-opus-5-5-high"),
            ("gemini-4-argon", "high", "gemini-4-argon-high"),
            ("grok-4.7", "xhigh", "grok-4.7-xhigh"),
            ("muse-wonder-1.0", "high", "muse-wonder-1.0-high"),
        )
        for base, effort, wanted in expected:
            slug, error = config.cursor_effort_slug(base, effort, "default")
            self.assertIsNone(error, (base, effort))
            self.assertEqual(slug, wanted)

        self.assertEqual(
            model_catalog._cursor_effort_candidates(["gpt-oss-120b-medium"]), {})
        thinking = model_catalog._cursor_effort_candidates(
            ["family-thinking-low", "family-thinking-high"])
        self.assertEqual(thinking["family-thinking"], {
            "low": "family-thinking-low", "high": "family-thinking-high"})

    def test_cursor_local_catalog_adds_without_revoking_legacy_mapping(self):
        payload = {"provider": "cursor", "models": [
            {"id": "manual-family-low", "name": "Manual low"},
            {"id": "manual-family-high", "name": "Manual high"},
        ]}
        path = self.write_json("cursor.json", payload)
        imported = model_catalog.manual_import("cursor", "default", path)
        self.assertTrue(imported["ok"], imported)
        slug, error = config.cursor_effort_slug("manual-family", "high", "default")
        self.assertIsNone(error)
        self.assertEqual(slug, "manual-family-high")
        self.assertEqual(config.cursor_effort_slug("gpt-5.6-sol", "xhigh", "default"),
                         ("gpt-5.6-sol-xhigh", None))

        ambiguous = self.write_json("ambiguous.json", {
            "provider": "cursor",
            "models": [{"id": "gpt-oss-120b-medium", "name": "gpt-oss-120b-medium"}],
        })
        self.assertTrue(model_catalog.manual_import("cursor", "default", ambiguous)["ok"])
        slug, error = config.cursor_effort_slug("gpt-oss-120b", "medium", "default")
        self.assertIsNone(slug)
        self.assertEqual(error["reason"], "unsupported_effort")

    def test_route_prefix_is_additive_and_bare_gpt_stays_openai(self):
        self.assertEqual(resolve_route("agy/gpt-oss-120b-medium"), {
            "provider": "agy", "model": "gpt-oss-120b-medium"})
        self.assertEqual(resolve_route("gpt-oss-120b-medium")["provider"], "openai")

    def test_all_five_manual_imports_round_trip_as_unknown(self):
        imported_ids = {}
        for provider in model_catalog.PROVIDERS:
            model_id = "manual-%s-model" % provider
            path = self.write_json(provider + ".json", self.manual_payload(
                provider, model_id, observed_at="2020-01-02T03:04:05Z"))
            result = model_catalog.manual_import(provider, "default", path)
            self.assertTrue(result["ok"], (provider, result))
            self.assertEqual(result["source"], "manual")
            self.assertEqual(result["availability"], "unknown")
            imported_ids[provider] = model_id

        state = model_catalog.status("all", "default")
        self.assertEqual([row["provider"] for row in state["catalogs"]],
                         list(model_catalog.PROVIDERS))
        self.assertTrue(all(row["source"] == "manual" for row in state["catalogs"]))
        self.assertTrue(all(row["stale"] is True for row in state["catalogs"]))
        self.assertEqual(model_catalog.check("all", "default")["exit_code"], 3)

        output = Path(self.work) / "snapshot.json"
        exported = model_catalog.snapshot(output, "all", "default")
        self.assertTrue(exported["ok"], exported)
        result = json.loads(output.read_text(encoding="utf-8"))
        for provider, model_id in imported_ids.items():
            record = result["providers"][provider]
            self.assertEqual(record["source"], "manual")
            self.assertEqual(record["availability"], "unknown")
            self.assertEqual(record["observed_at"], "2020-01-02T03:04:05Z")
            self.assertEqual(record["models"][0]["id"], model_id)

    def test_manual_import_without_observation_time_does_not_invent_one(self):
        path = self.write_json("manual-no-time.json", self.manual_payload(
            "claude", "claude-manual"))
        result = model_catalog.manual_import("claude", "default", path)
        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["observed_at"])
        state = model_catalog.status("claude", "default", details=True)
        self.assertIsNone(state["catalogs"][0]["observed_at"])
        self.assertIsNone(state["catalogs"][0]["age_seconds"])
        self.assertEqual(state["catalogs"][0]["origin"], "local")
        self.assertEqual(state["catalogs"][0]["models"][0]["id"], "claude-manual")

    def test_manual_import_preserves_supplied_route_and_effort_known_for_validation(self):
        seed = self.write_json("claude-last-good.json", self.manual_payload(
            "claude", "claude-last-good"))
        self.assertTrue(model_catalog.manual_import("claude", "default", seed)["ok"])
        cache_path = model_catalog._cache_paths("claude", "default")[0]
        last_good_bytes = cache_path.read_bytes()

        cases = (
            ({"id": "claude-route-conflict", "routable_id": "grok/claude-route-conflict",
              "supported_efforts": []}, "routable_id_invalid"),
            ({"id": "claude-known-string", "routable_id": "claude/claude-known-string",
              "supported_efforts": [], "efforts_known": "false"},
             "effort_metadata_invalid"),
            ({"id": "claude-unknown-with-efforts",
              "routable_id": "claude/claude-unknown-with-efforts",
              "supported_efforts": ["low"], "efforts_known": False},
             "effort_metadata_invalid"),
        )
        for row, reason in cases:
            with self.subTest(reason=reason):
                source = self.write_json("invalid-%s.json" % row["id"], {
                    "provider": "claude", "models": [row]})
                result = model_catalog.manual_import("claude", "default", source)
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], reason)
                self.assertEqual(cache_path.read_bytes(), last_good_bytes)

    def test_bundled_snapshot_has_no_machine_paths_and_uses_road_prefixes(self):
        path = Path(__file__).resolve().parent.parent / "providers" / "model_catalog.json"
        encoded = path.read_bytes()
        self.assertNotIn(b"/Users/", encoded)
        self.assertNotIn(b"/private/tmp/", encoded)
        for provider, record in model_catalog.bundled_catalog().items():
            self.assertEqual(record["provider"], provider)
            for row in record["models"]:
                self.assertTrue(row["routable_id"].startswith(provider + "/"))

    def test_profile_endpoint_and_home_changes_isolate_cache_without_secrets(self):
        keys = ("OMNI_LLM_CURSOR_CONFIG_DIR_WORK", "CURSOR_API_KEY_WORK",
                "OMNI_LLM_CATALOG_ENDPOINT_CURSOR_WORK")
        previous = {key: os.environ.get(key) for key in keys}
        def restore_profile_env():
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(restore_profile_env)
        home_one = Path(self.work) / "cursor-home-one"
        home_two = Path(self.work) / "cursor-home-two"
        home_one.mkdir()
        home_two.mkdir()
        os.environ["OMNI_LLM_CURSOR_CONFIG_DIR_WORK"] = str(home_one)
        os.environ["CURSOR_API_KEY_WORK"] = "test-secret-one"
        os.environ["OMNI_LLM_CATALOG_ENDPOINT_CURSOR_WORK"] = "https://one.example/v1?token=hidden"
        first_path = model_catalog._cache_paths("cursor", "work", create=True)[0]
        first_fingerprint = model_catalog._scope_fingerprint("cursor", "work", create=True)

        os.environ["CURSOR_API_KEY_WORK"] = "test-secret-two"
        self.assertEqual(model_catalog._scope_fingerprint("cursor", "work", create=True),
                         first_fingerprint)
        os.environ["OMNI_LLM_CATALOG_ENDPOINT_CURSOR_WORK"] = "https://two.example/v1"
        second_fingerprint = model_catalog._scope_fingerprint("cursor", "work", create=True)
        self.assertNotEqual(first_fingerprint, second_fingerprint)
        os.environ["OMNI_LLM_CURSOR_CONFIG_DIR_WORK"] = str(home_two)
        third_path = model_catalog._cache_paths("cursor", "work", create=True)[0]
        self.assertNotEqual(first_path, third_path)

        path = self.write_json("private-profile.json", self.manual_payload(
            "cursor", "private-model"))
        self.assertTrue(model_catalog.manual_import("cursor", "work", path)["ok"])
        snapshot_path = Path(self.work) / "profile-snapshot.json"
        self.assertTrue(model_catalog.snapshot(snapshot_path, "cursor", "work")["ok"])
        for file in Path(self.state).rglob("*"):
            if file.is_file():
                self.assertNotIn(b"test-secret", file.read_bytes())
        self.assertNotIn(b"one.example", snapshot_path.read_bytes())
        self.assertNotIn(b"hidden", snapshot_path.read_bytes())

    def test_agy_named_profile_requires_and_fingerprints_its_config_home(self):
        first = Path(self.work) / "agy-profile-one"
        second = Path(self.work) / "agy-profile-two"
        first.mkdir()
        second.mkdir()
        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": str(first)}):
            first_fingerprint = model_catalog._scope_fingerprint("agy", "work", create=True)
        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": str(second)}):
            second_fingerprint = model_catalog._scope_fingerprint("agy", "work", create=True)
        self.assertNotEqual(first_fingerprint, second_fingerprint)
        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": ""}):
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._scope_fingerprint("agy", "work", create=True)
        self.assertEqual(caught.exception.reason, "unsupported_profile")

    def test_refresh_output_cap_and_malformed_result_preserve_last_good_bytes(self):
        path = self.write_json("cursor-good.json", self.manual_payload(
            "cursor", "cursor-good"))
        self.assertTrue(model_catalog.manual_import("cursor", "default", path)["ok"])
        cache_path = model_catalog._cache_paths("cursor", "default")[0]
        original = cache_path.read_bytes()

        def too_large(_profile, _timeout):
            return model_catalog.parse_cursor_output("x" * (model_catalog.MAX_STDOUT_BYTES + 1))

        with mock.patch.dict(model_catalog._DISCOVERERS, {"cursor": too_large}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
            result = model_catalog.refresh("cursor", "default", 3)
        self.assertFalse(result["ok"])
        self.assertEqual(result["providers"]["cursor"]["reason"], "output_cap")
        self.assertEqual(cache_path.read_bytes(), original)
        self.assertEqual(model_catalog.status("cursor", "default")["catalogs"][0]
                         ["last_refresh"]["reason"], "output_cap")

        def partial(_profile, _timeout):
            return model_catalog.parse_cursor_output("cursor-id - label")

        with mock.patch.dict(model_catalog._DISCOVERERS, {"cursor": partial}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
            result = model_catalog.refresh("cursor", "default", 3)
        self.assertEqual(result["providers"]["cursor"]["reason"], "incomplete_output")
        self.assertEqual(cache_path.read_bytes(), original)

    def test_cache_commit_reports_success_when_status_write_fails(self):
        initial = self.write_json("cursor-status-initial.json", self.manual_payload(
            "cursor", "cursor-before-status-failure"))
        self.assertTrue(model_catalog.manual_import("cursor", "default", initial)["ok"])
        cache_path, status_path, _lock_path = model_catalog._cache_paths(
            "cursor", "default")
        original_cache = cache_path.read_bytes()
        status_target = Path(self.work) / "status-target.json"
        target_bytes = b"status target sentinel\n"
        status_target.write_bytes(target_bytes)
        status_path.unlink()
        status_path.symlink_to(status_target)

        refreshed_record = model_catalog.bundled_catalog()["cursor"]
        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {"cursor": lambda _profile, _timeout: refreshed_record}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  return_value=None):
            refreshed = model_catalog.refresh("cursor", "default", 5)
        provider_result = refreshed["providers"]["cursor"]
        self.assertTrue(refreshed["ok"], refreshed)
        self.assertTrue(provider_result["ok"], provider_result)
        self.assertEqual(provider_result["status_diagnostic"], "status_write_failed")
        self.assertNotIn("last_good_preserved", provider_result)
        self.assertNotEqual(cache_path.read_bytes(), original_cache)
        cached, _info = model_catalog._load_cached("cursor", "default")
        self.assertEqual(cached["models"], refreshed_record["models"])
        self.assertEqual(status_target.read_bytes(), target_bytes)
        self.assertTrue(status_path.is_symlink())

        manual = self.write_json("cursor-status-manual.json", self.manual_payload(
            "cursor", "cursor-after-status-failure"))
        before_manual = cache_path.read_bytes()
        with mock.patch.object(model_catalog, "_write_status",
                               side_effect=PermissionError("synthetic status failure")):
            imported = model_catalog.manual_import("cursor", "default", manual)
        self.assertTrue(imported["ok"], imported)
        self.assertEqual(imported["status_diagnostic"], "status_write_failed")
        self.assertNotIn("last_good_preserved", imported)
        self.assertNotEqual(cache_path.read_bytes(), before_manual)
        cached, _info = model_catalog._load_cached("cursor", "default")
        self.assertEqual(cached["models"][0]["id"], "cursor-after-status-failure")
        self.assertEqual(status_target.read_bytes(), target_bytes)

    def test_refresh_all_keeps_success_when_another_provider_fails(self):
        bundle = model_catalog.bundled_catalog()
        discoverers = {name: (lambda _profile, _timeout, record=record: record)
                       for name, record in bundle.items()}
        def agy_failure(_profile, _timeout):
            raise model_catalog.CatalogError("auth_error", "agy")
        discoverers["agy"] = agy_failure
        with mock.patch.dict(model_catalog._DISCOVERERS, discoverers), \
                mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
            result = model_catalog.refresh("all", "default", 5)
        self.assertFalse(result["ok"])
        self.assertTrue(result["providers"]["cursor"]["ok"])
        self.assertEqual(result["providers"]["agy"]["reason"], "auth_error")
        cursor, _info = model_catalog._load_cached("cursor", "default")
        agy, _info = model_catalog._load_cached("agy", "default")
        self.assertIsNotNone(cursor)
        self.assertIsNone(agy)

    def test_atomic_cache_writers_leave_only_complete_records_for_readers(self):
        from threading import Thread
        first = self.write_json("first.json", self.manual_payload("cursor", "cursor-first"))
        second = self.write_json("second.json", self.manual_payload("cursor", "cursor-second"))
        self.assertTrue(model_catalog.manual_import("cursor", "default", first)["ok"])
        failures = []
        def writer():
            for index in range(30):
                path = first if index % 2 else second
                result = model_catalog.manual_import("cursor", "default", path)
                if not result.get("ok"):
                    failures.append(result)
        thread = Thread(target=writer)
        thread.start()
        reads = 0
        while thread.is_alive() or reads < 30:
            record, _info = model_catalog._load_cached("cursor", "default")
            if record is None or len(record["models"]) != 1:
                failures.append("partial cache")
            else:
                self.assertIn(record["models"][0]["id"],
                              ("cursor-first", "cursor-second"))
            reads += 1
        thread.join()
        self.assertEqual(failures, [])

    def test_status_and_check_are_local_and_name_unusable_cache(self):
        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {name: mock.Mock(side_effect=AssertionError("network"))
                              for name in model_catalog.PROVIDERS}):
            value = model_catalog.status("all", "default")
            checked = model_catalog.check("all", "default")
        self.assertTrue(value["ok"])
        self.assertEqual(len(value["catalogs"]), 5)
        self.assertEqual(checked["exit_code"], 2)
        self.assertEqual(checked["unusable"], list(model_catalog.PROVIDERS))

    def test_status_details_lists_normalized_ids_without_refreshing(self):
        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {name: mock.Mock(side_effect=AssertionError("network"))
                              for name in model_catalog.PROVIDERS}):
            value = model_catalog.status("cursor", "default", details=True)
        self.assertTrue(value["details"])
        models = value["catalogs"][0]["models"]
        self.assertEqual(len(models), 265)
        self.assertIn("grok-4.7-high", {row["id"] for row in models})

    def test_check_reports_added_removed_efforts_and_default_availability(self):
        changed = self.write_json("cursor-drift.json", {
            "provider": "cursor",
            "models": [{"id": "new-cursor-model", "name": "new-cursor-model"}],
        })
        self.assertTrue(model_catalog.manual_import("cursor", "default", changed)["ok"])
        result = model_catalog.check("cursor", "default")
        self.assertEqual(result["exit_code"], 3)
        row = result["providers"][0]
        self.assertIn("new-cursor-model", row["added"])
        self.assertIn("auto", row["removed"])
        self.assertFalse(row["default_availability"]["local_available"])

        effort_file = self.write_json("effort-drift.json", {
            "provider": "cursor",
            "models": [{"id": "claude-opus-5-5-high",
                         "supported_efforts": ["low"]}],
        })
        self.assertTrue(model_catalog.manual_import("cursor", "default", effort_file)["ok"])
        effort_check = model_catalog.check("cursor", "default")
        self.assertIn("claude-opus-5-5-high",
                      effort_check["providers"][0]["effort_changes"])

    def test_snapshot_preserves_original_observation_and_checks_output_symlinks(self):
        observed = "2019-12-31T23:59:59Z"
        path = self.write_json("manual-grok.json", self.manual_payload(
            "grok", "grok-manual", observed_at=observed))
        self.assertTrue(model_catalog.manual_import("grok", "default", path)["ok"])
        output = Path(self.work) / "promoted.json"
        self.assertTrue(model_catalog.snapshot(output, "grok", "default")["ok"])
        exported = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exported["providers"]["grok"]["observed_at"], observed)
        self.assertEqual(exported["providers"]["grok"]["source"], "manual")
        self.assertNotEqual(exported["exported_at"], observed)
        self.assertEqual(exported["providers"]["cursor"]["source"],
                         model_catalog.bundled_catalog()["cursor"]["source"])

        link = Path(self.work) / "snapshot-link.json"
        link.symlink_to(output)
        refused = model_catalog.snapshot(link, "grok", "default")
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["reason"], "output_path_invalid")

        corrupt = Path(self.work) / "corrupt.json"
        corrupt.write_text('{"schema_version":1,"providers":{"unknown":{}}}',
                           encoding="utf-8")
        refused = model_catalog.snapshot(corrupt, "cursor", "default")
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["reason"], "snapshot_invalid")

    def test_snapshot_refuses_broken_identity_without_changing_destination(self):
        bundle = model_catalog.bundled_catalog()
        previous_snapshot = {
            "schema_version": model_catalog.SCHEMA_VERSION,
            "providers": bundle,
        }
        destination_bytes = (json.dumps(previous_snapshot, sort_keys=True,
                                        ensure_ascii=False) + "\n").encode("utf-8")
        original_read_bytes = Path.read_bytes
        for state_name, failure in (("truncated", "truncated"),
                                    ("unreadable", "unreadable"),
                                    ("missing", "missing")):
            with self.subTest(identity_state=state_name):
                state = Path(self.work) / ("state-" + state_name)
                source = self.write_json("identity-%s.json" % state_name,
                                         self.manual_payload("cursor", "cursor-seed"))
                destination = Path(self.work) / ("destination-%s.json" % state_name)
                destination.write_bytes(destination_bytes)
                with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": str(state)}):
                    self.assertTrue(model_catalog.manual_import(
                        "cursor", "default", source)["ok"])
                    root = model_catalog._catalog_root()
                    cache_path, _status_path, _lock_path = model_catalog._cache_paths(
                        "cursor", "default")
                    self.assertTrue(cache_path.is_file())
                    identity_path = root / ".identity-key"
                    if failure == "truncated":
                        identity_path.write_bytes(b"x")
                    elif failure == "missing":
                        identity_path.unlink()

                    if failure == "unreadable":
                        def denied_identity_read(path):
                            if path == identity_path:
                                raise PermissionError("synthetic identity read denial")
                            return original_read_bytes(path)

                        with mock.patch.object(Path, "read_bytes", denied_identity_read):
                            with self.assertRaises(model_catalog.CatalogError) as caught:
                                model_catalog._scope_fingerprint(
                                    "cursor", "default", create=True)
                            self.assertEqual(caught.exception.reason,
                                             "cache_identity_invalid")
                            refused = model_catalog.snapshot(
                                destination, "cursor", "default")
                    else:
                        with self.assertRaises(model_catalog.CatalogError) as caught:
                            model_catalog._scope_fingerprint(
                                "cursor", "default", create=True)
                        self.assertEqual(caught.exception.reason,
                                         "cache_identity_invalid")
                        refused = model_catalog.snapshot(
                            destination, "cursor", "default")
                    self.assertFalse(refused["ok"], refused)
                    self.assertEqual(refused["reason"], "cache_invalid")
                    self.assertEqual(destination.read_bytes(), destination_bytes)
                    if failure == "truncated":
                        self.assertEqual(identity_path.read_bytes(), b"x")
                    elif failure == "missing":
                        self.assertFalse(identity_path.exists())

        fresh_state = Path(self.work) / "fresh-state"
        fresh_destination = Path(self.work) / "fresh-export.json"
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": str(fresh_state)}):
            fresh = model_catalog.snapshot(fresh_destination, "cursor", "default")
        self.assertTrue(fresh["ok"], fresh)
        exported = json.loads(fresh_destination.read_text(encoding="utf-8"))
        self.assertEqual(exported["providers"]["cursor"], bundle["cursor"])
        self.assertFalse(fresh_state.exists())

    def test_corrupt_local_cache_falls_back_to_bundle_with_diagnostic(self):
        cache_path = model_catalog._cache_paths("cursor", "default", create=True)[0]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b'{"partial":')
        result = model_catalog.status("cursor", "default")
        self.assertTrue(result["ok"])
        self.assertEqual(result["catalogs"][0]["source"], "live")
        self.assertEqual(result["catalogs"][0]["origin"], "bundled")
        self.assertEqual(result["catalogs"][0]["cache_diagnostic"], "cache_invalid")
        slug, error = config.cursor_effort_slug("grok-4.7", "xhigh", "default")
        self.assertIsNone(error)
        self.assertEqual(slug, "grok-4.7-xhigh")

    def test_launchd_plist_is_explicit_and_profile_scoped(self):
        cursor_profile = Path(self.work) / "cursor-profile"
        cursor_profile.mkdir()
        os.environ["OMNI_LLM_CURSOR_CONFIG_DIR_WORK"] = str(cursor_profile)
        result = model_catalog.schedule_plist("work", 86400, 120)
        self.assertTrue(result["ok"], result)
        plist = plistlib.loads(result["plist"].encode("utf-8"))
        self.assertEqual(plist["Label"], "com.omni-llm.model-catalogs.work")
        args = plist["ProgramArguments"]
        self.assertTrue(os.path.isabs(args[0]))
        self.assertTrue(os.path.isabs(args[1]))
        self.assertEqual(args[2:], ["models", "refresh", "--provider", "all",
                                    "--profile", "work", "--timeout", "120"])
        self.assertTrue(plist["EnvironmentVariables"]["PATH"])
        self.assertTrue(os.path.isabs(plist["StandardOutPath"]))

    def test_schedule_all_default_profile_remains_supported(self):
        with mock.patch.dict(os.environ, {
                "HOME": self.work, "PATH": "/usr/bin:/bin",
                "OMNI_LLM_STATE_DIR": self.state}, clear=True):
            result = model_catalog.schedule_plist(
                "default", 86400, 120, provider="all")
        self.assertTrue(result["ok"], result)
        plist = plistlib.loads(result["plist"].encode("utf-8"))
        self.assertEqual(plist["ProgramArguments"][5], "all")
        self.assertEqual(plist["ProgramArguments"][7], "default")

    def test_schedule_all_refuses_valid_unconfigured_profile(self):
        with mock.patch.dict(os.environ, {
                "HOME": self.work, "PATH": "/usr/bin:/bin",
                "OMNI_LLM_STATE_DIR": self.state}, clear=True):
            result = model_catalog.schedule_plist(
                "unconfigured", 86400, 120, provider="all")
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["reason"], "unsupported_profile")

    def test_schedule_all_allows_partially_supported_named_profile(self):
        cursor_profile = Path(self.work) / "cursor-profile-work"
        cursor_profile.mkdir()
        with mock.patch.dict(os.environ, {
                "HOME": self.work, "PATH": "/usr/bin:/bin",
                "OMNI_LLM_STATE_DIR": self.state,
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": str(cursor_profile)},
                clear=True):
            result = model_catalog.schedule_plist(
                "work", 86400, 120, provider="all")
        self.assertTrue(result["ok"], result)
        plist = plistlib.loads(result["plist"].encode("utf-8"))
        self.assertEqual(plist["ProgramArguments"][5], "all")
        self.assertEqual(plist["ProgramArguments"][7], "work")
        self.assertEqual(
            plist["EnvironmentVariables"]["OMNI_LLM_CURSOR_CONFIG_DIR_WORK"],
            os.path.realpath(cursor_profile))

    def test_print_only_schedule_does_not_create_local_state(self):
        state = Path(self.work) / "schedule-state-not-created"
        with mock.patch.dict(os.environ, {
                "HOME": self.work, "PATH": "/usr/bin:/bin",
                "OMNI_LLM_STATE_DIR": str(state)}, clear=True):
            result = model_catalog.schedule_plist(
                "default", 86400, 120, provider="cursor")
        self.assertTrue(result["ok"], result)
        plist = plistlib.loads(result["plist"].encode("utf-8"))
        self.assertEqual(Path(plist["StandardOutPath"]).parent,
                         state / "model-catalogs")
        self.assertFalse(state.exists())

    def test_schedule_keeps_all_registered_homes_canonicalized(self):
        home = Path(self.work) / "user-home"
        codex_home = Path(self.work) / "codex-work"
        claude_home = Path(self.work) / "claude-alt"
        grok_home = Path(self.work) / "grok-alt"
        cursor_home = Path(self.work) / "cursor-shared"
        for path in (home, codex_home, claude_home, grok_home, cursor_home):
            path.mkdir()
        env = {
            "HOME": str(home), "PATH": "/usr/bin:/bin",
            "OMNI_LLM_STATE_DIR": self.state,
            "CODEX_HOME_WORK": str(codex_home),
            "CLAUDE_CONFIG_DIR_ALT": str(claude_home),
            "GROK_HOME_ALT": str(grok_home),
            "OMNI_LLM_CURSOR_CONFIG_DIR_SHARED": str(cursor_home),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = model_catalog.schedule_plist("work", 86400, 120, provider="codex")
        self.assertTrue(result["ok"], result)
        bindings = plistlib.loads(result["plist"].encode("utf-8"))["EnvironmentVariables"]
        for key, path in (("CODEX_HOME_WORK", codex_home),
                          ("CLAUDE_CONFIG_DIR_ALT", claude_home),
                          ("GROK_HOME_ALT", grok_home),
                          ("OMNI_LLM_CURSOR_CONFIG_DIR_SHARED", cursor_home)):
            with self.subTest(home=key):
                self.assertEqual(bindings[key], os.path.realpath(path))

    def test_schedule_canonicalizes_relative_codex_home_across_working_directories(self):
        from providers.profiles import resolve_profile

        user_home = Path(self.work) / "user-home"
        first_cwd = Path(self.work) / "cwd-first"
        second_cwd = Path(self.work) / "cwd-second"
        target = first_cwd / "profiles" / "codex-work"
        user_home.mkdir()
        first_cwd.mkdir()
        second_cwd.mkdir()
        target.mkdir(parents=True)
        relative_home = "profiles/codex-work"
        original_cwd = os.getcwd()
        env = {"HOME": str(user_home), "PATH": "/usr/bin:/bin",
               "OMNI_LLM_STATE_DIR": self.state, "CODEX_HOME_WORK": relative_home}
        try:
            os.chdir(first_cwd)
            with mock.patch.dict(os.environ, env, clear=True):
                result = model_catalog.schedule_plist(
                    "work", 86400, 120, provider="codex")
                self.assertTrue(result["ok"], result)
                bindings = plistlib.loads(result["plist"].encode("utf-8"))[
                    "EnvironmentVariables"]
                selected_home = bindings["CODEX_HOME_WORK"]
            os.chdir(second_cwd)
            with mock.patch.dict(os.environ, {
                    "HOME": str(user_home), "CODEX_HOME_WORK": selected_home}, clear=True):
                overrides, error = resolve_profile("codex", "work")
            self.assertIsNone(error)
            self.assertEqual(overrides["CODEX_HOME"], os.path.realpath(target))
            self.assertEqual(selected_home, os.path.realpath(target))
        finally:
            os.chdir(original_cwd)

    def test_native_default_home_remap_changes_identity_without_explicit_override(self):
        from providers import kernel_sandbox

        home_keys = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME",
                     "grok": "GROK_HOME"}
        for provider in ("claude", "codex", "grok"):
            with self.subTest(provider=provider):
                native_home = Path(self.work) / (provider + "-native-home")
                first_root = Path(self.work) / (provider + "-native-one")
                second_root = Path(self.work) / (provider + "-native-two")
                for path in (native_home, first_root, second_root):
                    path.mkdir()
                leaf = native_home / ("." + provider)
                leaf.symlink_to(first_root, target_is_directory=True)
                env = {"HOME": str(native_home), "PATH": "/usr/bin:/bin",
                       "OMNI_LLM_STATE_DIR": self.state}
                with mock.patch.dict(os.environ, env, clear=True), \
                        mock.patch.object(kernel_sandbox, "protected_home",
                                          return_value=str(native_home)):
                    first_identity, first_child = model_catalog._profile_identity_inputs(
                        provider, "default")
                    first_fingerprint = model_catalog._scope_fingerprint(
                        provider, "default", create=True)
                    first_plist = model_catalog.schedule_plist(
                        "default", 86400, 120, provider=provider)
                    leaf.unlink()
                    leaf.symlink_to(second_root, target_is_directory=True)
                    second_identity, second_child = model_catalog._profile_identity_inputs(
                        provider, "default")
                    second_fingerprint = model_catalog._scope_fingerprint(
                        provider, "default", create=True)
                    second_plist = model_catalog.schedule_plist(
                        "default", 86400, 120, provider=provider)

                self.assertFalse(first_identity["home_explicit"])
                self.assertFalse(second_identity["home_explicit"])
                self.assertEqual(first_identity["config_home"], os.path.realpath(first_root))
                self.assertEqual(second_identity["config_home"], os.path.realpath(second_root))
                self.assertNotEqual(first_fingerprint, second_fingerprint)
                self.assertNotIn(home_keys[provider], first_child)
                self.assertTrue(first_plist["ok"], first_plist)
                self.assertTrue(second_plist["ok"], second_plist)
                first_env = plistlib.loads(first_plist["plist"].encode("utf-8"))[
                    "EnvironmentVariables"]
                second_env = plistlib.loads(second_plist["plist"].encode("utf-8"))[
                    "EnvironmentVariables"]
                self.assertNotIn(home_keys[provider], first_env)
                self.assertNotIn(home_keys[provider], second_env)

    def test_schedule_retains_only_required_endpoint_passthrough_selectors(self):
        env = {
            "HOME": self.work, "PATH": "/usr/bin:/bin",
            "OMNI_LLM_STATE_DIR": self.state,
            "OMNI_LLM_CHILD_ENV_PASS": "CURSOR_BASE_URL,CLAUDE_CODE_PLUGIN_DIRS,UNRELATED_SELECTOR",
            "CURSOR_BASE_URL": "https://cursor.example/v1",
            "CLAUDE_CODE_PLUGIN_DIRS": "/temporary/plugin-dir",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            source_identity, _source_child = model_catalog._profile_identity_inputs(
                "cursor", "default")
            result = model_catalog.schedule_plist("default", 86400, 120, provider="cursor")
        self.assertTrue(result["ok"], result)
        bindings = plistlib.loads(result["plist"].encode("utf-8"))["EnvironmentVariables"]
        self.assertEqual(bindings["OMNI_LLM_CHILD_ENV_PASS"], "CURSOR_BASE_URL")
        self.assertEqual(bindings["CURSOR_BASE_URL"], "https://cursor.example/v1")
        self.assertNotIn("CLAUDE_CODE_PLUGIN_DIRS", bindings)
        self.assertNotIn("UNRELATED_SELECTOR", bindings["OMNI_LLM_CHILD_ENV_PASS"])

        with mock.patch.dict(os.environ, bindings, clear=True):
            _overrides, child_env, allowed_env, _selected, _endpoints = (
                model_catalog._catalog_child_environment("cursor", "default"))
            scheduled_identity, _scheduled_child = model_catalog._profile_identity_inputs(
                "cursor", "default")
        self.assertEqual(child_env["CURSOR_BASE_URL"], "https://cursor.example/v1")
        self.assertEqual(allowed_env["CURSOR_BASE_URL"], "https://cursor.example/v1")
        self.assertEqual(source_identity["endpoints"], scheduled_identity["endpoints"])

    def test_claude_read_only_spawn_uses_scrubbed_effective_environment(self):
        from providers.host_fence import allowlisted_child_env

        capture = json.loads((FIXTURES / "model_catalog_claude_initialize.json").read_text(
            encoding="utf-8"))
        event = {"type": "control_response", "response": {
            "request_id": "catalog-init", "response": {"models": capture["models"]}}}
        read_fd, write_fd = os.pipe()
        os.write(write_fd, (json.dumps(event) + "\n").encode("utf-8"))
        os.close(write_fd)
        captured = {}

        class FakeProc:
            pid = 99999999
            def __init__(self, stdout):
                self.stdin = io.BytesIO()
                self.stdout = stdout
                self.killed = False
                self.waited = False
            def poll(self):
                return None if not self.killed else -9
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                self.waited = True
                return -9

        proc = FakeProc(os.fdopen(read_fd, "rb", buffering=0))

        def prepare(argv, *, env, **_kwargs):
            captured["prepare_env"] = dict(env)
            return list(argv), allowlisted_child_env(env, "claude"), {
                **_verified_read_only_fence_metadata()}

        def popen(argv, *, env, **kwargs):
            captured["spawn_env"] = dict(env)
            return proc

        with mock.patch.dict(os.environ, {
                "OMNI_LLM_CHILD_ENV_PASS": "CLAUDE_CODE_PLUGIN_DIRS",
                "CLAUDE_CODE_PLUGIN_DIRS": "/synthetic/plugin-dir"}), \
                mock.patch("providers.kernel_sandbox.resolve_agent_binary",
                           return_value="/usr/bin/claude"), \
                mock.patch("providers.host_fence.prepare_host_fence", side_effect=prepare), \
                mock.patch.object(model_catalog.subprocess, "Popen", side_effect=popen):
            record = model_catalog._discover_claude("default", 5)
        self.assertEqual(len(record["models"]), 13)
        self.assertNotIn("CLAUDE_CODE_PLUGIN_DIRS", captured["prepare_env"])
        self.assertNotIn("CLAUDE_CODE_PLUGIN_DIRS", captured["spawn_env"])
        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_claude_and_grok_identity_use_only_actual_allowlisted_endpoints(self):
        claude_home = Path(self.work) / ".claude-work"
        grok_home = Path(self.work) / "grok-work"
        claude_home.mkdir()
        grok_home.mkdir()
        env = {
            "CLAUDE_CONFIG_DIR_WORK": str(claude_home),
            "ANTHROPIC_BASE_URL": "https://claude.example/v1",
            "ANTHROPIC_BASE_URL_WORK": "https://claude-sibling.example/v1",
            "GROK_HOME_WORK": str(grok_home),
            "XAI_API_BASE": "https://xai.example/v1",
            "GROK_MODELS_LIST_URL": "https://grok.example/models",
            "GROK_CLI_CHAT_PROXY_BASE_URL": "https://proxy.example/v1",
            "XAI_API_BASE_WORK": "https://xai-sibling.example/v1",
            "OMNI_LLM_CATALOG_ENDPOINT_GROK_WORK": "https://catalog-only.example/v1",
        }
        with mock.patch.dict(os.environ, env):
            _overrides, _claude_env, claude_allowed, _home, _endpoints = (
                model_catalog._catalog_child_environment("claude", "work"))
            claude_identity, _claude_child = model_catalog._profile_identity_inputs(
                "claude", "work")
            claude_plist = model_catalog.schedule_plist(
                "work", 86400, 120, provider="claude")
            os.environ["ANTHROPIC_BASE_URL_WORK"] = "https://claude-sibling-two.example/v1"
            claude_suffix_identity, _claude_child = model_catalog._profile_identity_inputs(
                "claude", "work")
            os.environ["ANTHROPIC_BASE_URL"] = "https://claude-two.example/v1"
            claude_generic_identity, _claude_child = model_catalog._profile_identity_inputs(
                "claude", "work")
            initial_grok_identity, _grok_child = model_catalog._profile_identity_inputs(
                "grok", "work")
            grok_plist = model_catalog.schedule_plist("work", 86400, 120, provider="grok")
            os.environ["XAI_API_BASE_WORK"] = "https://xai-sibling-two.example/v1"
            grok_suffix_identity, _grok_child = model_catalog._profile_identity_inputs(
                "grok", "work")
            os.environ["GROK_MODELS_LIST_URL"] = "https://grok.example/v2/models"
            changed_grok_identity, _grok_child = model_catalog._profile_identity_inputs(
                "grok", "work")

        self.assertEqual(claude_allowed["ANTHROPIC_BASE_URL"], "https://claude.example/v1")
        self.assertNotIn("ANTHROPIC_BASE_URL_WORK", claude_allowed)
        self.assertEqual(set(claude_identity["endpoints"]), {"ANTHROPIC_BASE_URL"})
        self.assertEqual(claude_identity["endpoints"], claude_suffix_identity["endpoints"])
        self.assertNotEqual(claude_identity["endpoints"], claude_generic_identity["endpoints"])
        self.assertTrue(claude_plist["ok"], claude_plist)
        claude_bindings = plistlib.loads(claude_plist["plist"].encode("utf-8"))[
            "EnvironmentVariables"]
        self.assertEqual(claude_bindings["ANTHROPIC_BASE_URL"],
                         "https://claude.example/v1")
        self.assertNotIn("ANTHROPIC_BASE_URL_WORK", claude_bindings)

        self.assertEqual(set(initial_grok_identity["endpoints"]), {
            "XAI_API_BASE", "GROK_MODELS_LIST_URL", "GROK_CLI_CHAT_PROXY_BASE_URL"})
        self.assertEqual(initial_grok_identity["endpoints"],
                         grok_suffix_identity["endpoints"])
        self.assertNotEqual(initial_grok_identity["endpoints"],
                            changed_grok_identity["endpoints"])
        self.assertTrue(grok_plist["ok"], grok_plist)
        grok_bindings = plistlib.loads(grok_plist["plist"].encode("utf-8"))[
            "EnvironmentVariables"]
        self.assertEqual(grok_bindings["XAI_API_BASE"], "https://xai.example/v1")
        self.assertEqual(grok_bindings["GROK_MODELS_LIST_URL"],
                         "https://grok.example/models")
        self.assertEqual(grok_bindings["GROK_CLI_CHAT_PROXY_BASE_URL"],
                         "https://proxy.example/v1")
        self.assertNotIn("XAI_API_BASE_WORK", grok_bindings)
        self.assertNotIn("OMNI_LLM_CATALOG_ENDPOINT_GROK_WORK", grok_bindings)

    def test_schedule_refuses_secret_bearing_endpoint_without_echoing_it(self):
        synthetic_key = "synthetic-astra-endpoint-key-20261017"
        unsafe_urls = (
            "https://api.example/v1/" + synthetic_key,
            "https://api.example/v1?access_token=" + synthetic_key,
            "https://user:" + synthetic_key + "@api.example/v1",
            "https://api.example/v1#" + synthetic_key,
        )
        with mock.patch.dict(os.environ, {
                "ANTHROPIC_API_KEY": synthetic_key}):
            refused = []
            refused_all = []
            for unsafe_url in unsafe_urls:
                os.environ["ANTHROPIC_BASE_URL"] = unsafe_url
                refused.append(model_catalog.schedule_plist(
                    "default", 86400, 120, provider="claude"))
                refused_all.append(model_catalog.schedule_plist(
                    "default", 86400, 120, provider="all"))
            os.environ["ANTHROPIC_BASE_URL"] = "https://api.example/v1?api-version=2024-11"
            safe = model_catalog.schedule_plist(
                "default", 86400, 120, provider="claude")
        for result in refused:
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "endpoint_secret_refused")
            self.assertNotIn(synthetic_key, json.dumps(result))
            self.assertNotIn("api.example", json.dumps(result))
        for result in refused_all:
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "endpoint_secret_refused")
            self.assertNotIn(synthetic_key, json.dumps(result))
            self.assertNotIn("api.example", json.dumps(result))
        self.assertTrue(safe["ok"], safe)
        safe_bindings = plistlib.loads(safe["plist"].encode("utf-8"))["EnvironmentVariables"]
        self.assertEqual(safe_bindings["ANTHROPIC_BASE_URL"],
                         "https://api.example/v1?api-version=2024-11")

    def test_secret_endpoint_refuses_before_hash_store_and_read_side_writes(self):
        safe_payload = self.write_json("claude-secret-endpoint-last-good.json",
                                       self.manual_payload("claude", "claude-last-good"))
        self.assertTrue(model_catalog.manual_import(
            "claude", "default", safe_payload)["ok"])
        cache_path = model_catalog._cache_paths("claude", "default")[0]
        previous_bytes = cache_path.read_bytes()
        synthetic_key = "synthetic-astra-hash-boundary-key-20261017"
        unsafe_url = "https://api.example/v1/path/" + synthetic_key

        with mock.patch.dict(os.environ, {
                "ANTHROPIC_API_KEY": synthetic_key,
                "ANTHROPIC_BASE_URL": unsafe_url}):
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._scope_fingerprint("claude", "default")
            self.assertEqual(caught.exception.reason, "endpoint_secret_refused")

            imported = model_catalog.manual_import("claude", "default", safe_payload)
            self.assertFalse(imported["ok"])
            self.assertEqual(imported["reason"], "endpoint_secret_refused")
            with mock.patch.dict(model_catalog._DISCOVERERS, {
                    "claude": lambda _profile, _timeout:
                    model_catalog.bundled_catalog()["claude"]}), \
                    mock.patch.object(model_catalog, "_best_effort_cli_version",
                                      return_value=None):
                refreshed = model_catalog.refresh("claude", "default", 5)
            self.assertEqual(refreshed["providers"]["claude"]["reason"],
                             "endpoint_secret_refused")
        self.assertEqual(cache_path.read_bytes(), previous_bytes)

        read_only_state = Path(self.work) / "secret-read-only-state"
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": str(read_only_state),
                "ANTHROPIC_API_KEY": synthetic_key,
                "ANTHROPIC_BASE_URL": unsafe_url}):
            status = model_catalog.status("claude", "default")
            checked = model_catalog.check("claude", "default")
        self.assertTrue(status["ok"])
        self.assertEqual(status["catalogs"][0]["cache_diagnostic"],
                         "endpoint_secret_refused")
        self.assertEqual(checked["providers"][0]["reason"], "endpoint_secret_refused")
        self.assertFalse(read_only_state.exists())
        self.assertNotIn(synthetic_key, json.dumps(status))
        self.assertNotIn(synthetic_key, json.dumps(checked))

    def test_agy_effort_help_parser_uses_fenced_installed_vocabulary(self):
        help_text = (FIXTURES / "model_catalog_agy_help.txt").read_text(encoding="utf-8")
        self.assertEqual(model_catalog.parse_agy_effort_tokens(help_text),
                         frozenset(("low", "medium", "high", "xhigh", "max")))
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog.parse_agy_effort_tokens("Usage: agy --help\n--effort\n")
        self.assertEqual(caught.exception.reason, "unsupported_effort")

    def test_codex_rpc_paginates_in_order_and_always_reaps_child(self):
        class FakeProc:
            pid = 99999999
            def __init__(self, stdout):
                self.stdin = PipeCaptureStdin()
                self.stdout = stdout
                self.killed = False
                self.waited = False
            def poll(self):
                return None if not self.killed else -9
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                self.waited = True
                return -9

        messages = [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {
                "data": [{"model": "gpt-test-a", "displayName": "A",
                          "supportedReasoningEfforts": [{"reasoningEffort": "low"}],
                          "defaultReasoningEffort": "low"}], "nextCursor": "page-2"}},
            {"jsonrpc": "2.0", "id": 3, "result": {
                "data": [{"model": "gpt-test-b", "displayName": "B",
                          "supportedReasoningEfforts": []}], "nextCursor": None}},
        ]
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"".join((json.dumps(item) + "\n").encode("utf-8")
                                     for item in messages))
        os.close(write_fd)
        proc = FakeProc(os.fdopen(read_fd, "rb", buffering=0))
        prepare_calls = []
        def fake_prepare(argv, **kwargs):
            prepare_calls.append((list(argv), kwargs))
            return list(argv), kwargs["env"], \
                _verified_read_only_fence_metadata()

        with mock.patch("providers.host_fence.prepare_host_fence", fake_prepare), \
                mock.patch.object(model_catalog.subprocess, "Popen", return_value=proc):
            record = model_catalog._query_codex_models(
                "/usr/bin/codex", env={"PATH": "/usr/bin"}, cwd=self.work,
                timeout_s=5)
        self.assertEqual([row["id"] for row in record["models"]],
                         ["gpt-test-a", "gpt-test-b"])
        argv, kwargs = prepare_calls[0]
        self.assertEqual(argv, ["/usr/bin/codex", "app-server"])
        self.assertEqual(kwargs["provider"], "codex")
        self.assertTrue(kwargs["read_only"])
        sent = [json.loads(line) for line in proc.stdin.read_all().decode("utf-8").splitlines()]
        self.assertEqual([item["method"] for item in sent],
                         ["initialize", "initialized", "model/list", "model/list"])
        self.assertEqual(sent[-1]["params"]["cursor"], "page-2")
        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_codex_rpc_completes_partial_pipe_writes(self):
        server_code = (
            "import sys; data = sys.stdin.buffer.read(); "
            "sys.stdout.write(str(len(data)) + '\\n'); sys.stdout.flush()")
        params = {"cursor": "x" * (256 * 1024)}
        expected = (json.dumps({"jsonrpc": "2.0", "method": "model/list",
                                "id": 7, "params": params},
                               separators=(",", ":")) + "\n").encode("utf-8")
        self.assertGreater(len(expected), 65536)
        proc = model_catalog.subprocess.Popen(
            [sys.executable, "-c", server_code], stdin=model_catalog.subprocess.PIPE,
            stdout=model_catalog.subprocess.PIPE,
            stderr=model_catalog.subprocess.DEVNULL,
            start_new_session=True, bufsize=0)
        try:
            model_catalog._send_codex_rpc(
                proc, "model/list", params, request_id=7,
                deadline=time.monotonic() + 3)
            proc.stdin.close()
            self.assertEqual(int(proc.stdout.readline()), len(expected))
            self.assertEqual(proc.wait(timeout=1), 0)
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=1)
            proc.stdout.close()

    def test_codex_rpc_full_pipe_times_out_and_reaps_nonreading_child(self):
        fake_server = Path(self.work) / "fake-codex-app-server"
        fake_server.write_text("""#!%s
import json, time

def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)

initialize = {"jsonrpc": "2.0", "id": 1, "result": {}}
page = {"jsonrpc": "2.0", "id": 2, "result": {
    "data": [{"model": "gpt-rpc-test"}], "nextCursor": ""}}
limit = 1024 * 1024
cursor_size = limit - len((
    json.dumps(initialize, separators=(",", ":")) + "\\n"
).encode()) - len((
    json.dumps(page, separators=(",", ":")) + "\\n"
).encode())
page["result"]["nextCursor"] = "x" * cursor_size
emit(initialize)
emit(page)
time.sleep(60)
""" % sys.executable, encoding="utf-8")
        fake_server.chmod(0o700)

        real_popen = model_catalog.subprocess.Popen
        spawned = []
        outcome = []
        completed = threading.Event()

        def capture_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        def prepare(argv, **kwargs):
            return list(argv), kwargs["env"], \
                _verified_read_only_fence_metadata()

        def query():
            try:
                outcome.append(model_catalog._query_codex_models(
                    str(fake_server), env={"PATH": "/usr/bin:/bin"},
                    cwd=self.work, timeout_s=1))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                completed.set()

        started = time.monotonic()
        watchdog_fired = False
        with mock.patch("providers.host_fence.prepare_host_fence", prepare), \
                mock.patch.object(model_catalog.subprocess, "Popen", side_effect=capture_popen):
            worker = threading.Thread(target=query, daemon=True)
            worker.start()
            if not completed.wait(2.75):
                watchdog_fired = True
                if spawned and spawned[0].poll() is None:
                    spawned[0].kill()
                completed.wait(0.75)
            worker.join(0.01)

        if spawned and spawned[0].poll() is None:
            spawned[0].kill()
            spawned[0].wait(timeout=0.5)
        elapsed = time.monotonic() - started
        self.assertFalse(watchdog_fired, "query did not return before the bounded watchdog")
        self.assertTrue(completed.is_set(), "query worker remained blocked")
        self.assertEqual(len(spawned), 1)
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], model_catalog.CatalogError)
        self.assertEqual(outcome[0].reason, "timeout")
        self.assertLess(elapsed, 2.5)
        self.assertIsNotNone(spawned[0].poll(), "query did not reap the fake child")

    def test_codex_live_metadata_rejects_bad_rows_and_efforts_without_cache_replace(self):
        previous_payload = self.write_json("codex-last-good.json", self.manual_payload(
            "codex", "codex-last-good"))
        self.assertTrue(model_catalog.manual_import(
            "codex", "default", previous_payload)["ok"])
        cache_path = model_catalog._cache_paths("codex", "default")[0]
        last_good_bytes = cache_path.read_bytes()

        class FakeProc:
            pid = 99999999
            def __init__(self, stdout):
                self.stdin = PipeCaptureStdin()
                self.stdout = stdout
                self.killed = False
                self.waited = False
            def poll(self):
                return None if not self.killed else -9
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                self.waited = True
                return -9

        def refresh_pages(pages):
            messages = [{"jsonrpc": "2.0", "id": 1, "result": {}}]
            for index, rows in enumerate(pages):
                next_cursor = "page-%d" % (index + 2) if index + 1 < len(pages) else None
                messages.append({"jsonrpc": "2.0", "id": index + 2,
                                 "result": {"data": rows, "nextCursor": next_cursor}})
            read_fd, write_fd = os.pipe()
            os.write(write_fd, b"".join(
                (json.dumps(message) + "\n").encode("utf-8") for message in messages))
            os.close(write_fd)
            proc = FakeProc(os.fdopen(read_fd, "rb", buffering=0))

            def discover(_profile, timeout_s):
                return model_catalog._query_codex_models(
                    "/usr/bin/codex", env={"PATH": "/usr/bin"},
                    cwd=self.work, timeout_s=timeout_s)

            def prepare(argv, **kwargs):
                return list(argv), kwargs["env"], \
                    _verified_read_only_fence_metadata()

            with mock.patch("providers.host_fence.prepare_host_fence", side_effect=prepare), \
                    mock.patch.object(model_catalog.subprocess, "Popen", return_value=proc), \
                    mock.patch.dict(model_catalog._DISCOVERERS, {"codex": discover}), \
                    mock.patch.object(model_catalog, "_best_effort_cli_version", return_value=None):
                result = model_catalog.refresh("codex", "default", 5)
            try:
                self.assertTrue(proc.killed)
                self.assertTrue(proc.waited)
                return result
            finally:
                proc.stdin.close_capture()

        valid_row = {"model": "codex-page-one", "displayName": "Page one"}
        invalid_cases = (
            ("non_dict_later_page", [[valid_row], [None]], "model_entry_invalid"),
            ("unknown_effort_string", [[{"model": "codex-none-string",
                                         "supportedReasoningEfforts": ["none"]}]],
             "effort_metadata_invalid"),
            ("unknown_effort_object", [[{"model": "codex-none-object",
                                         "supportedReasoningEfforts": [
                                             {"reasoningEffort": "none"}]}]],
             "effort_metadata_invalid"),
            ("wrong_effort_shape", [[{"model": "codex-wrong-shape",
                                      "supportedReasoningEfforts": "low"}]],
             "effort_metadata_invalid"),
            ("conflicting_default", [[{"model": "codex-default-conflict",
                                        "supportedReasoningEfforts": ["low"],
                                        "defaultReasoningEffort": "high"}]],
             "effort_metadata_invalid"),
            ("unknown_default", [[{"model": "codex-default-unknown",
                                    "supportedReasoningEfforts": ["low"],
                                    "defaultReasoningEffort": "none"}]],
             "effort_metadata_invalid"),
            ("wrong_type_default", [[{"model": "codex-default-wrong-type",
                                       "supportedReasoningEfforts": ["low"],
                                       "defaultReasoningEffort": 7}]],
             "effort_metadata_invalid"),
        )
        for label, pages, reason in invalid_cases:
            with self.subTest(case=label):
                result = refresh_pages(pages)
                self.assertFalse(result["ok"])
                self.assertEqual(result["providers"]["codex"]["reason"], reason)
                self.assertEqual(cache_path.read_bytes(), last_good_bytes)

        valid_result = refresh_pages([[
            {"model": "codex-effort-absent"},
            {"model": "codex-effort-empty", "supportedReasoningEfforts": []},
            {"model": "codex-effort-string", "supportedReasoningEfforts": ["low"]},
            {"model": "codex-effort-object", "supportedReasoningEfforts": [
                {"reasoningEffort": "high", "description": "Higher reasoning"}]},
            {"model": "codex-effort-default",
             "supportedReasoningEfforts": ["low"], "defaultReasoningEffort": "low"},
        ]])
        self.assertTrue(valid_result["ok"], valid_result)
        cached, _info = model_catalog._load_cached("codex", "default")
        rows = {row["id"]: row for row in cached["models"]}
        self.assertFalse(rows["codex-effort-absent"]["efforts_known"])
        self.assertTrue(rows["codex-effort-empty"]["efforts_known"])
        self.assertEqual(rows["codex-effort-string"]["supported_efforts"], ["low"])
        self.assertEqual(rows["codex-effort-object"]["supported_efforts"], ["high"])
        self.assertEqual(rows["codex-effort-default"]["default_effort"], "low")

    def test_codex_rpc_rejects_server_requests_and_non_metadata_methods(self):
        class Reader:
            def __init__(self, lines):
                self.lines = iter(lines)
            def line(self):
                return next(self.lines)
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog._read_codex_response(Reader([
                json.dumps({"jsonrpc": "2.0", "id": 9,
                            "method": "item/command/requestApproval"})]), 9)
        self.assertEqual(caught.exception.reason, "server_request_denied")
        trace = json.loads((FIXTURES / "model_catalog_codex_protocol_framing.json").read_text(encoding="utf-8"))
        notification = next(item["method"] for item in trace
                            if item.get("method") is not None and not item.get("has_id"))
        response = json.dumps({"jsonrpc": "2.0", "id": 9, "result": {"data": []}})
        result = model_catalog._read_codex_response(Reader([
            json.dumps({"jsonrpc": "2.0", "method": notification,
                        "params": {"type": "metadata"}}), response]), 9)
        self.assertEqual(result, {"data": []})
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog._read_codex_response(Reader([
                json.dumps({"jsonrpc": "2.0", "method": "unknown/notification"})]), 9)
        self.assertEqual(caught.exception.reason, "unexpected_notification")
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog._send_codex_rpc(None, "turn/start", {},
                                          deadline=time.monotonic() + 1)
        self.assertEqual(caught.exception.reason, "protocol_method_denied")

    def test_claude_initialize_parser_requires_one_catalog_response(self):
        capture = json.loads((FIXTURES / "model_catalog_claude_initialize.json").read_text(encoding="utf-8"))
        response = {"type": "control_response", "response": {
            "request_id": "catalog-init", "response": {"models": capture["models"]}}}
        parsed = model_catalog.parse_claude_control_events(json.dumps(response) + "\n")
        self.assertEqual(len(parsed["models"]), 13)
        self.assertNotIn("description", parsed["models"][0])
        rejected = {"type": "control_request", "request_id": "approval",
                    "request": {"subtype": "can_use_tool"}}
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog.parse_claude_control_events(json.dumps(rejected) + "\n")
        self.assertEqual(caught.exception.reason, "unexpected_server_request")

    def test_cursor_and_agy_truncated_captures_fail_closed(self):
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog.parse_cursor_output("model-id - Label")
        self.assertEqual(caught.exception.reason, "incomplete_output")
        with self.assertRaises(model_catalog.CatalogError) as caught:
            model_catalog.parse_agy_output("model-id\tLabel")
        self.assertEqual(caught.exception.reason, "incomplete_output")


if __name__ == "__main__":
    unittest.main()
