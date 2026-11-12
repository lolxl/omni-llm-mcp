# no-opencc
"""Admission for grok image editing, live probes, the sandbox canary, and models check.

No vendor CLI and no network. Spawn primitives are patched.
"""
import asyncio
import os
import tempfile
import unittest
from unittest import mock

from providers import admission
from providers import catalog_v2, model_catalog, profile_check, sandbox_canary
from providers import grok_image_cli
from providers.catalog_record import validate_strict
from providers.host_fence import _VENDOR_AUTH_ENV
from tests import offline_home

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True

_HOME = None
FRESH = "2026-11-10T00:00:00Z"


def setUpModule():
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _record(provider, rows):
    models = []
    for row in rows:
        item = {
            "id": row["id"],
            "name": row.get("name", row["id"]),
            "routable_id": "%s/%s" % (provider, row["id"]),
        }
        if "efforts" in row:
            item["supported_efforts"] = list(row["efforts"])
        models.append(item)
    value = {
        "provider": provider,
        "source": "live",
        "observed_at": FRESH,
        "availability": "provider_advertised",
        "models": models,
    }
    record, _warnings = validate_strict(provider, value)
    return record


def _save(provider, profile, rows):
    catalog_v2.save(provider, profile, _record(provider, rows))


def _fence(argv, **kwargs):
    return list(argv), kwargs.get("env") or {}, {"fence": "none"}


class ProbeCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory(prefix="omni-probe-")
        self.addCleanup(self.state.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state.name
        for names in _VENDOR_AUTH_ENV.values():
            for name in names:
                os.environ.pop(name, None)

    def _image(self, prompt, source, out_dir, **kwargs):
        calls = []

        def fake_run(argv, **_kw):
            calls.append(list(argv))
            path = os.path.join(out_dir, "image_edit.png")
            with open(path, "wb") as handle:
                handle.write(b"PNG")
            return mock.Mock(returncode=0, stdout=b"", stderr=b"")

        with mock.patch.object(grok_image_cli.vendor_bin, "which",
                                return_value="/fake/bin/grok"), \
                mock.patch.object(grok_image_cli, "prepare_host_fence",
                                  side_effect=_fence), \
                mock.patch.object(grok_image_cli.subprocess, "run",
                                  side_effect=fake_run), \
                mock.patch.object(grok_image_cli.subprocess, "Popen",
                                  side_effect=AssertionError("popen")):
            result = asyncio.run(grok_image_cli.run_grok_image_edit(
                prompt, source, out_dir=out_dir, **kwargs))
        return result, calls


class ImageAdmissionTests(ProbeCase):
    def test_listed_model_and_effort_are_on_argv(self):
        _save("grok", "default", [{"id": "grok-4.7", "efforts": ["low", "medium", "high"]}])
        out = tempfile.mkdtemp(prefix="omni-img-")
        source = os.path.join(out, "source.png")
        with open(source, "wb") as handle:
            handle.write(b"src")
        result, calls = self._image("make it sunset", source, out)
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("model"), "grok-image-edit")
        self.assertEqual(result.get("effort"), "medium")
        self.assertEqual(result.get("effort_source"), "omni_default")
        prompt = calls[0][2]
        self.assertEqual(calls[0], [
            "/fake/bin/grok", "-p", prompt,
            "-m", "grok-4.7", "--reasoning-effort", "medium",
        ])

    def test_missing_record_does_not_spawn(self):
        out = tempfile.mkdtemp(prefix="omni-img-")
        source = os.path.join(out, "source.png")
        result, calls = self._image("make it sunset", source, out)
        self.assertEqual(calls, [])
        self.assertFalse(result.get("ok"), result)
        self.assertIn("models refresh", result.get("detail") or "")

    def test_record_without_the_image_model_does_not_spawn(self):
        _save("grok", "default", [{"id": "grok-4.6", "efforts": ["medium"]}])
        out = tempfile.mkdtemp(prefix="omni-img-")
        source = os.path.join(out, "source.png")
        result, calls = self._image("make it sunset", source, out)
        self.assertEqual(calls, [])
        self.assertFalse(result.get("ok"), result)
        self.assertIn("models refresh", result.get("detail") or "")

    def test_named_account_still_reads_the_default_record(self):
        _save("grok", "default", [{"id": "grok-4.7", "efforts": ["medium"]}])
        # A named grok profile exists only when GROK_HOME_<NAME> is a directory.
        work = os.path.join(self.state.name, "grok-work")
        os.makedirs(work)
        os.environ["GROK_HOME_WORK"] = work
        _save("grok", "work", [{"id": "grok-4.6", "efforts": ["medium"]}])
        out = tempfile.mkdtemp(prefix="omni-img-")
        source = os.path.join(out, "source.png")
        result, calls = self._image(
            "make it sunset", source, out, account_profile="work")
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(calls[0][3:], [
            "-m", "grok-4.7", "--reasoning-effort", "medium",
        ])


class ProfileCheckAdmissionTests(ProbeCase):
    # The runners admit by themselves (test_admission_wiring proves zero
    # spawns on a refusal); the probe hands them the request unchanged and
    # reports their catalog refusal as not_admitted.
    def test_runner_catalog_refusal_is_not_admitted(self):
        async def fake(*_args, **_kwargs):
            return admission.refusal_envelope(
                "codex", "default", {"refusal": "no_catalog"})

        with mock.patch("providers.codex.run_codex_subprocess", fake):
            result = profile_check._live_one("codex", "default")
        state, problems = profile_check._live_result_state("codex", {}, result)
        self.assertEqual(state, "not_admitted")
        self.assertEqual(problems, ["no_catalog"])

    def test_probe_passes_the_shipped_model_without_effort(self):
        calls = []

        async def fake(prompt, model, timeout_s, account_profile="default", **kwargs):
            calls.append((model, account_profile, "effort" in kwargs))
            return {"ok": True, "text": "pong"}

        with mock.patch("providers.cursor.run_cursor_subprocess", fake):
            result = profile_check._live_one("cursor", "work")
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(calls, [("composer-2.5", "work", False)])


class SandboxCanaryAdmissionTests(ProbeCase):
    def test_refusal_does_not_run_or_write_a_record(self):
        calls = []

        async def fake_run_cli(argv, **_kwargs):
            calls.append(list(argv))
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "timed_out": False, "signal": None}

        with mock.patch.object(sandbox_canary, "codex_cli_version",
                                return_value="0.148.0"), \
                mock.patch.object(sandbox_canary.vendor_bin, "which",
                                  return_value="/fake/bin/codex"), \
                mock.patch.object(sandbox_canary, "run_cli", fake_run_cli):
            result = sandbox_canary.run_write_canary(runs=1, timeout_s=5)
        self.assertEqual(calls, [])
        self.assertEqual(result.get("reason"), "not_admitted")
        self.assertIn("no_catalog", result.get("detail") or "")
        self.assertFalse(os.path.exists(
            os.path.join(self.state.name, "sandbox_canary.json")))

    def test_admitted_argv_carries_the_decided_model(self):
        _save("codex", "default", [{
            "id": "gpt-6.1-sol", "efforts": ["low", "medium", "high"],
        }])
        calls = []

        async def fake_run_cli(argv, **_kwargs):
            calls.append(list(argv))
            return {"returncode": 1, "stdout": "", "stderr": "nope",
                    "timed_out": False, "signal": None}

        with mock.patch.object(sandbox_canary.vendor_bin, "which",
                                return_value="/fake/bin/codex"), \
                mock.patch.object(sandbox_canary, "run_cli", fake_run_cli):
            asyncio.run(sandbox_canary._run_vendor_readonly_probe("p", 5))
        argv = calls[0]
        out = argv[argv.index("-o") + 1]
        self.assertEqual(argv, [
            "/fake/bin/codex", "exec", "--json",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "-c", "model_reasoning_effort=medium",
            "-m", "gpt-6.1-sol",
            "-o", out,
            "-",
        ])


class ModelsCheckAdmissionTests(ProbeCase):
    def test_admitted_default_exits_zero(self):
        _save("codex", "default", [{
            "id": "gpt-6.1-sol", "efforts": ["low", "medium", "high"],
        }])
        result = model_catalog.check("codex", "default")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["providers"][0]["admission"], [{
            "model": "gpt-6.1-sol",
            "ok": True,
            "effort": "medium",
            "refusal": None,
        }])

    def test_grok_check_admits_the_image_model(self):
        _save("grok", "default", [
            {"id": "grok-4.6", "efforts": ["medium"]},
            {"id": "grok-4.7", "efforts": ["medium"]},
        ])
        result = model_catalog.check("grok", "default")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["providers"][0]["admission"], [
            {"model": "grok-4.6", "ok": True, "effort": "medium", "refusal": None},
            {"model": "grok-4.7", "ok": True, "effort": "medium", "refusal": None},
        ])

    def test_effort_required_stays_exit_zero(self):
        _save("codex", "default", [{"id": "gpt-6.1-sol", "efforts": ["high"]}])
        result = model_catalog.check("codex", "default")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["providers"][0]["ok"])
        item = result["providers"][0]["admission"][0]
        self.assertEqual(item["model"], "gpt-6.1-sol")
        self.assertEqual(item["refusal"], "effort_required")
        self.assertTrue(item["ok"])

    def test_missing_default_model_exits_two(self):
        _save("codex", "default", [{"id": "other-model", "efforts": ["medium"]}])
        result = model_catalog.check("codex", "default")
        self.assertEqual(result["exit_code"], 2)
        self.assertFalse(result["ok"])
        self.assertFalse(result["providers"][0]["ok"])
        item = result["providers"][0]["admission"][0]
        self.assertEqual(item["model"], "gpt-6.1-sol")
        self.assertFalse(item["ok"])
        self.assertEqual(item["refusal"], "unsupported_model")

    def test_host_auth_env_does_not_skip_the_record(self):
        # A host API key would put a request on today's gate; the check still
        # reads the record.
        _save("codex", "default", [{"id": "other-model", "efforts": ["medium"]}])
        with mock.patch.dict(os.environ, {"CODEX_API_KEY": "probe-test-codex-key"}):
            result = model_catalog.check("codex", "default")
        self.assertEqual(result["exit_code"], 2)
        item = result["providers"][0]["admission"][0]
        self.assertEqual(item["refusal"], "unsupported_model")


if __name__ == "__main__":
    unittest.main()
