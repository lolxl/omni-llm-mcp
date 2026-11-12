"""Refresh writes a v2 catalog. Fake vendor output only."""
import contextlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import omni_cli
from providers import catalog_v2, kernel_sandbox, model_catalog

FIXED_HOME = "/omni-d2b2a-fixed-home"
WHEN = model_catalog._datetime.datetime(
    2026, 11, 11, 12, 0, 0, tzinfo=model_catalog._datetime.timezone.utc)
IDENTITY = b"\x22" * 32
CLAUDE_ARGV = (
    "--print", "--input-format", "stream-json",
    "--output-format", "stream-json", "--verbose", "--tools", "",
    "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}",
    "--setting-sources", "", "--no-session-persistence",
    "--disable-slash-commands",
)

def _encode(messages):
    return b"".join(
        (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")
        for item in messages)


def _codex_stdout(rows):
    return _encode([
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 2, "result": {"data": rows, "nextCursor": None}},
    ])


def _claude_stdout(models):
    return _encode([{"type": "control_response", "response": {
        "request_id": "catalog-init",
        "response": {"models": models},
    }}])


_DEFAULT_STDOUT = {
    "cursor": b"Available models\nplain-cursor - Plain Cursor\nTip: listed\n",
    "agy": b"agy-one\tAgy One\n",
    "grok": b"Available models:\n- grok-one\n",
    "codex": _codex_stdout([{
        "model": "codex-one",
        "displayName": "Codex One",
        "supportedReasoningEfforts": ["medium", "low"],
        "defaultReasoningEffort": "low",
    }]),
    "claude": _claude_stdout([{
        "value": "claude-one",
        "displayName": "Claude One",
        "supportedEffortLevels": ["high", "low"],
    }]),
}


class _FakeProc:
    def __init__(self, stdout_bytes):
        self.pid = 99999999
        self.returncode = None
        out_read, out_write = os.pipe()
        os.write(out_write, stdout_bytes)
        os.close(out_write)
        self.stdout = os.fdopen(out_read, "rb", buffering=0)
        in_read, in_write = os.pipe()
        self._stdin_read = in_read
        self.stdin = os.fdopen(in_write, "wb", buffering=0)
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9
        if self._stdin_read is not None:
            os.close(self._stdin_read)
            self._stdin_read = None


class CatalogV2Tests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp(prefix="omni-catalog-v2-")
        self.bindir = tempfile.mkdtemp(prefix="omni-catalog-v2-bins-")
        self.addCleanup(shutil.rmtree, self.state, True)
        self.addCleanup(shutil.rmtree, self.bindir, True)
        for name in ("cursor-agent", "agy", "grok", "codex", "claude"):
            path = os.path.join(self.bindir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)
        removed = {}
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_CATALOG_ENDPOINT_"):
                removed[key] = os.environ.pop(key)
        for keys in model_catalog._ENDPOINT_ENV_KEYS.values():
            for key in keys:
                if key in os.environ:
                    removed[key] = os.environ.pop(key)
        for key in ("CURSOR_CONFIG_DIR", "OMNI_LLM_CURSOR_CONFIG_DIR",
                    "AGY_CONFIG_DIR", "CODEX_HOME", "CLAUDE_CONFIG_DIR",
                    "GROK_HOME", "OMNI_LLM_CURSOR_BIN", "OMNI_LLM_CLAUDE_BIN"):
            if key in os.environ:
                removed[key] = os.environ.pop(key)
        self.addCleanup(os.environ.update, removed)
        self.env = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state,
            "OMNI_LLM_NOTIFY_QUIET": "1",
            "PATH": self.bindir + os.pathsep + os.environ.get("PATH", ""),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        # Fingerprint inputs stay fixed for save and for the later load.
        self.platform = mock.patch("platform.system", return_value="Linux")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.home = mock.patch.object(
            kernel_sandbox, "protected_home", lambda: FIXED_HOME)
        self.home.start()
        self.addCleanup(self.home.stop)
        root = Path(self.state) / "model-catalogs"
        root.mkdir(mode=0o700)
        key_path = root / ".identity-key"
        key_path.write_bytes(IDENTITY)
        os.chmod(key_path, 0o600)

    @contextlib.contextmanager
    def _patched_discovery(self, stdout=None, version=None):
        """The vendor CLIs the refresh fakes already use."""
        payloads = dict(_DEFAULT_STDOUT)
        if stdout:
            payloads.update(stdout)
        calls = []

        def popen(argv, **kwargs):
            calls.append({"argv": list(argv), "env": kwargs.get("env")})
            if "app-server" in argv:
                payload = payloads["codex"]
            elif "--list-models" in argv:
                payload = payloads["cursor"]
            elif "stream-json" in argv:
                payload = payloads["claude"]
            elif argv[-1:] == ["models"] and "agy" in " ".join(map(str, argv)):
                payload = payloads["agy"]
            elif argv[-1:] == ["models"]:
                payload = payloads["grok"]
            else:
                payload = b"9.9.9\n"
            return _FakeProc(payload)

        real_dt = model_catalog._datetime.datetime

        class Frozen(real_dt):
            @classmethod
            def now(cls, tz=None):
                return WHEN

        with mock.patch.object(model_catalog._datetime, "datetime", Frozen), \
                mock.patch.object(model_catalog.subprocess, "Popen",
                                  side_effect=popen), \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  return_value=version):
            yield calls

    def _refresh(self, provider="all", stdout=None, version=None):
        with self._patched_discovery(stdout, version) as calls:
            result = model_catalog.refresh(provider, "default", 30)
        return result, calls

    def _cli_refresh(self, provider="all", stdout=None, version=None):
        """Exit code from the real ``models refresh`` handler."""
        stdout_buf = io.StringIO()
        with self._patched_discovery(stdout, version) as calls:
            with contextlib.redirect_stdout(stdout_buf):
                code = omni_cli.main([
                    "models", "refresh",
                    "--provider", provider,
                    "--profile", "default",
                    "--timeout", "30",
                ])
        return code, json.loads(stdout_buf.getvalue()), calls

    def _today_files(self, provider):
        folder = Path(self.state) / "model-catalogs"
        return sorted(
            path for path in folder.glob(provider + "-*.json")
            if not path.name.endswith(".status.json"))

    def _v2_files(self, provider):
        folder = Path(self.state) / "model-catalogs-v2"
        return sorted(folder.glob(provider + "-*.json")) if folder.is_dir() else []

    def test_refresh_writes_only_v2(self):
        result, calls = self._refresh()
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(calls), 5, [call["argv"] for call in calls])
        expected_tails = {
            "cursor": ("--list-models",),
            "agy": ("models",),
            "codex": ("app-server",),
            "claude": CLAUDE_ARGV,
            "grok": ("models",),
        }
        for call, provider in zip(calls, model_catalog.PROVIDERS):
            self.assertEqual(tuple(call["argv"][1:]), expected_tails[provider])
            self.assertNotIn("v2_warning", result["providers"][provider])
            self.assertTrue(result["providers"][provider]["ok"])
            self.assertEqual(self._today_files(provider), [])
            copied = self._v2_files(provider)
            self.assertEqual(len(copied), 1, copied)
            self.assertEqual(stat.S_IMODE(copied[0].stat().st_mode), 0o600)
            loaded, reason = catalog_v2.load(provider, "default")
            self.assertIsNone(reason)
            self.assertEqual(loaded, json.loads(copied[0].read_text(encoding="utf-8")))

        cursor, _reason = catalog_v2.load("cursor", "default")
        agy, _reason = catalog_v2.load("agy", "default")
        codex, _reason = catalog_v2.load("codex", "default")
        claude, _reason = catalog_v2.load("claude", "default")
        grok, _reason = catalog_v2.load("grok", "default")
        self.assertEqual(cursor["models"][0]["effort_field"], "empty")
        self.assertEqual(agy["models"][0]["effort_field"], "empty")
        self.assertEqual(codex["models"][0]["display_name"], "Codex One")
        self.assertEqual(codex["models"][0]["supported_efforts"], ["low", "medium"])
        self.assertEqual(codex["models"][0]["default_effort"], "low")
        self.assertEqual(claude["models"][0]["display_name"], "Claude One")
        self.assertEqual(claude["models"][0]["supported_efforts"], ["low", "high"])
        self.assertEqual(grok["models"][0]["effort_field"], "absent")
        self.assertEqual(grok["grok_default"], {"line": None, "row": None})
        self._v2_files("agy")[0].unlink()
        missing, missing_reason = catalog_v2.load("agy", "default")
        self.assertIsNone(missing)
        self.assertIsNone(missing_reason)

        copied = self._v2_files("codex")[0]
        copied.write_bytes(b"{not json\n")
        loaded, reason = catalog_v2.load("codex", "default")
        self.assertIsNone(loaded)
        self.assertEqual(reason, "catalog_unreadable")
        copied.write_text(
            '{"provider":"codex","source":"manual","observed_at":'
            '"2026-11-11T12:00:00Z","availability":"unknown","models":'
            '[{"id":"codex-one","name":"codex-one",'
            '"routable_id":"codex/codex-one"}]}\n',
            encoding="utf-8")
        loaded, reason = catalog_v2.load("codex", "default")
        self.assertIsNone(loaded)
        self.assertEqual(reason, "catalog_unreadable")

    def test_cli_version_is_stored_on_v2(self):
        result, calls = self._refresh("codex", version="1.2.3")
        self.assertTrue(result["providers"]["codex"]["ok"], result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._today_files("codex"), [])
        loaded, reason = catalog_v2.load("codex", "default")
        self.assertIsNone(reason)
        self.assertEqual(loaded["cli_version"], "1.2.3")

    def test_codex_deep_is_a_successful_v2_write(self):
        result, calls = self._refresh("codex", stdout={"codex": _codex_stdout([{
            "model": "codex-deep",
            "displayName": "Codex Deep",
            "supportedReasoningEfforts": ["deep"],
        }])})
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["providers"]["codex"]["ok"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["argv"][1:], ["app-server"])
        self.assertEqual(self._today_files("codex"), [])
        loaded, reason = catalog_v2.load("codex", "default")
        self.assertIsNone(reason)
        row = loaded["models"][0]
        self.assertEqual(row["id"], "codex-deep")
        self.assertEqual(row["supported_efforts"], ["deep"])
        self.assertEqual(row["effort_field"], "listed")

    def test_claude_empty_effort_list_is_known_empty_in_v2(self):
        result, _calls = self._refresh("claude", stdout={"claude": _claude_stdout([{
            "value": "claude-empty",
            "displayName": "Claude Empty",
            "supportedEffortLevels": [],
        }])})
        self.assertTrue(result["providers"]["claude"]["ok"], result)
        self.assertEqual(self._today_files("claude"), [])
        loaded, reason = catalog_v2.load("claude", "default")
        self.assertIsNone(reason)
        row = loaded["models"][0]
        self.assertEqual(row["id"], "claude-empty")
        self.assertEqual(row["effort_field"], "empty")
        self.assertEqual(row["supported_efforts"], [])
        self.assertIs(row["efforts_known"], True)

    def test_v2_save_failure_is_a_provider_failure(self):
        with mock.patch.object(catalog_v2, "save",
                               side_effect=model_catalog.CatalogError(
                                   "cache_path_invalid")):
            result, _calls = self._refresh()
        self.assertFalse(result["ok"], result)
        self.assertEqual(self._v2_files("cursor"), [])
        for provider in model_catalog.PROVIDERS:
            self.assertEqual(self._today_files(provider), [])
            row = result["providers"][provider]
            self.assertEqual(row["ok"], False)
            self.assertEqual(row["reason"], "cache_path_invalid")
            self.assertNotIn("preserved", json.dumps(row))

    def test_models_refresh_handler_exits_zero_on_success(self):
        code, result, calls = self._cli_refresh()
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(calls), len(model_catalog.PROVIDERS))
        for provider in model_catalog.PROVIDERS:
            self.assertNotIn("v2_warning", result["providers"][provider])
        self.assertEqual(code, 0)

    def test_models_refresh_handler_exits_two_on_v2_failure(self):
        code, result, calls = self._cli_refresh("cursor", stdout={
            "cursor": b"not a listing\n",
        })
        self.assertFalse(result["ok"])
        self.assertEqual(result["providers"]["cursor"]["reason"], "malformed_output")
        self.assertEqual(len(calls), 1)
        self.assertEqual(code, 2)
        self.assertEqual(self._v2_files("cursor"), [])

    def test_models_refresh_handler_exits_two_when_v2_save_fails(self):
        with mock.patch.object(
                catalog_v2, "save",
                side_effect=model_catalog.CatalogError("cache_path_invalid")):
            code, result, calls = self._cli_refresh()
        self.assertFalse(result["ok"], result)
        self.assertEqual(len(calls), len(model_catalog.PROVIDERS))
        for provider in model_catalog.PROVIDERS:
            self.assertEqual(result["providers"][provider]["reason"], "cache_path_invalid")
        self.assertEqual(code, 2)

    def test_patched_discoverer_writes_v2_from_the_returned_record(self):
        record = {
            "provider": "grok",
            "source": "live",
            "observed_at": "2026-11-11T12:00:00Z",
            "availability": "unknown",
            "source_note": "vendor_registry_not_entitlement",
            "models": [{
                "id": "grok-patched",
                "name": "grok-patched",
                "routable_id": "grok/grok-patched",
            }],
        }
        with mock.patch.dict(model_catalog._DISCOVERERS,
                             {"grok": lambda _profile, _timeout: record}), \
                mock.patch.object(model_catalog, "_best_effort_cli_version",
                                  return_value=None):
            result = model_catalog.refresh("grok", "default", 5)
        self.assertTrue(result["providers"]["grok"]["ok"], result)
        self.assertEqual(self._today_files("grok"), [])
        loaded, reason = catalog_v2.load("grok", "default")
        self.assertIsNone(reason)
        self.assertEqual(loaded["models"][0]["id"], "grok-patched")

    def test_status_check_list_and_refresh_work_without_the_bundle(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "providers" / "model_catalog.json").is_file())
        before = model_catalog.status("all", "default")
        self.assertTrue(before["ok"], before)
        self.assertEqual(len(before["catalogs"]), 5)
        self.assertTrue(all(row["model_count"] == 0 for row in before["catalogs"]))
        checked = model_catalog.check("all", "default")
        self.assertEqual(checked["exit_code"], 2)
        result, calls = self._refresh()
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(calls), 5)
        status = model_catalog.status("agy", "default", details=True)
        self.assertEqual(status["catalogs"][0]["models"],
                         [{"id": "agy-one", "name": "Agy One"}])
        from providers import dispatch
        import asyncio
        with mock.patch("providers.agy.find_agy", return_value=None):
            listed = asyncio.run(dispatch.list_models())
        self.assertEqual(listed["reason"], "cli_missing")
        self.assertTrue(listed["catalogs"]["ok"], listed["catalogs"])
        self.assertEqual(
            next(row["model_count"] for row in listed["catalogs"]["catalogs"]
                 if row["provider"] == "agy"),
            1)
