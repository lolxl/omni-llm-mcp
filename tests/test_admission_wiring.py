# no-opencc
"""Admission wired into consult and the agent lane. No vendor CLI, no network."""
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from providers import admission, agent_lane, agy as agy_mod, catalog_v2
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import cursor as cursor_mod
from providers import dispatch
from providers import grok as grok_mod
from providers.catalog_record import validate_strict
from providers.config import DEFAULT_CODEX_REASONING_EFFORT
from tests import offline_home

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True

# A synthetic mobile number the egress gate refuses; split so the literal
# does not appear in source.
_SYNTHETIC_MOBILE = "0912" + "345678"

_HOME = None
FRESH = "2026-11-10T00:00:00Z"
NOTE_DEFAULT = "effort: medium (chosen by omni; no effort was given)"
BIN = {
    "codex": "/fake/bin/codex",
    "claude": "/fake/bin/claude",
    "grok": "/fake/bin/grok",
    "cursor": "/fake/bin/cursor-agent",
    "agy": "/fake/bin/agy",
}


def setUpModule():
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    _HOME.close()


def _record(provider, rows):
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


class FakeRun:
    def __init__(self, stdout="OK"):
        self.stdout = stdout
        self.calls = []

    async def __call__(self, argv=None, **kwargs):
        argv = list(argv or [])
        self.calls.append((argv, kwargs))
        if "-o" in argv:
            path = argv[argv.index("-o") + 1]
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("OK")
        return {"returncode": 0, "stdout": self.stdout, "stderr": "",
                "timed_out": False, "signal": None}


def _which(name, *args, **kwargs):
    base = os.path.basename(str(name))
    for provider, path in BIN.items():
        if base == provider or path.endswith("/" + base):
            return path
    if base in ("codex", "claude", "grok", "agy", "cursor-agent"):
        return BIN["cursor" if base == "cursor-agent" else base]
    return None


def _fence(argv, **kwargs):
    return list(argv), kwargs.get("env") or {}, {"fence": "none", "fence_detail": {}}


def _agy_fence(argv, env, prompt_bytes=None):
    return list(argv), env, tempfile.mkdtemp(prefix="agy_cwd_")


class _SDKAgent:
    calls = []

    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def chat(self, prompt):
        type(self).calls.append((self.config, prompt))

        async def tokens():
            yield "OK"
        return tokens()


class _SDKConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class WiringCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory(prefix="omni-admit-")
        self.addCleanup(self.state.cleanup)
        self.workdir = tempfile.TemporaryDirectory(prefix="omni-admit-wd-")
        self.addCleanup(self.workdir.cleanup)
        self.work = os.path.realpath(self.workdir.name)
        # Every variable a test sets is undone, not only the ones named here.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state.name
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "0"
        for key in ("OMNI_LLM_DEPTH", "OPENAI_BASE_URL", "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY", "CLAUDE_CONFIG_DIR"):
            os.environ.pop(key, None)
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_PROFILES_"):
                os.environ.pop(key, None)
        _SDKAgent.calls = []

    def _patches(self, fake):
        return [
            mock.patch.object(codex_mod, "run_cli", fake),
            mock.patch.object(claude_mod, "run_cli", fake),
            mock.patch.object(grok_mod, "run_cli", fake),
            mock.patch.object(cursor_mod, "run_cli", fake),
            mock.patch.object(agy_mod, "run_cli", fake),
            mock.patch.object(agent_lane, "run_cli", fake),
            mock.patch.object(codex_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(claude_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(grok_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(cursor_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(agent_lane, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(codex_mod.kernel_sandbox, "admit_vendor_binary", return_value=None),
            mock.patch.object(codex_mod.vendor_bin, "which", side_effect=_which),
            mock.patch.object(grok_mod.vendor_bin, "which", side_effect=_which),
            mock.patch.object(agy_mod, "find_agy", return_value=BIN["agy"]),
            mock.patch.object(agent_lane.vendor_bin, "which", side_effect=_which),
            mock.patch.object(agy_mod, "fence_agy_spawn", side_effect=_agy_fence),
            mock.patch.object(claude_mod, "resolve_agent_binary", return_value=BIN["claude"]),
            mock.patch.object(agent_lane, "resolve_agent_binary", side_effect=lambda provider: BIN.get(provider)),
            mock.patch.object(cursor_mod, "resolve_bin", return_value=BIN["cursor"]),
            mock.patch.object(grok_mod, "_consult_result",
                              side_effect=lambda out, sid: ("OK", {}, None)),
            mock.patch("shutil.which", side_effect=_which),
        ]

    def _run(self, fake, coro):
        patches = self._patches(fake)
        for patch in patches:
            patch.start()
        try:
            return asyncio.run(coro)
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_consult_listed_argv(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["low", "medium", "high"]}])
        _save("claude", "default", [{"id": "opus", "efforts": ["low", "medium", "high"]}])
        _save("grok", "default", [{"id": "grok-4.5", "efforts": ["low", "medium", "high"]}])
        _save("cursor", "default", [{"id": "composer-2.5"}])
        cases = (
            ("codex", codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30, effort="high")),
            ("claude", claude_mod.run_claude_subprocess("hi", "opus", 30, effort="high")),
            ("grok", grok_mod.run_grok_subprocess("hi", "grok-4.5", 30, effort="high")),
            ("cursor", cursor_mod.run_cursor_subprocess("hi", "composer-2.5", 30)),
        )
        for name, coro in cases:
            fake = FakeRun('{"result": "OK"}' if name in ("claude", "cursor") else "OK")
            result = self._run(fake, coro)
            self.assertTrue(result.get("ok"), (name, result))
            self.assertEqual(result.get("effort"), None if name == "cursor" else "high")
            self.assertEqual(result.get("effort_source"),
                             "model_id" if name == "cursor" else "request")
            argv = fake.calls[0][0]
            if name == "codex":
                self.assertEqual(argv, [
                    BIN["codex"], "exec", "--json", "--sandbox", "read-only",
                    "--skip-git-repo-check", "--ephemeral",
                    "-c", "model_reasoning_effort=high",
                    "-m", "gpt-5.4", "-o", argv[argv.index("-o") + 1], "-",
                ])
            elif name == "claude":
                self.assertEqual(argv, [
                    BIN["claude"], "-p", "--output-format", "json", "--model", "opus",
                    "--strict-mcp-config", "--tools", claude_mod.READ_ONLY_TOOLS,
                    "--disallowedTools", claude_mod.READ_ONLY_DISALLOWED,
                    "--effort", "high",
                ])
            elif name == "grok":
                self.assertEqual(argv, [
                    BIN["grok"], "--prompt-file", argv[2], "--model", "grok-4.5",
                    "--reasoning-effort", "high", "--output-format", "streaming-json",
                    "--no-memory", "--verbatim", "--disable-web-search",
                    "--tools", "todo_write",
                    "--disallowed-tools", "todo_write,search_tool,use_tool",
                    "--session-id", argv[argv.index("--session-id") + 1],
                    "--sandbox", "strict", "--cwd", argv[argv.index("--cwd") + 1],
                ])
            else:
                self.assertEqual(argv, [
                    BIN["cursor"], "-p", "--mode", "ask", "--output-format", "json",
                    "--trust", "--sandbox", "enabled",
                    "--workspace", argv[argv.index("--workspace") + 1],
                    "--model", "composer-2.5",
                ])

    def test_consult_unlisted_never_spawns(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["medium"]}])
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        _save("grok", "default", [{"id": "grok-4.5", "efforts": ["medium"]}])
        _save("cursor", "default", [{"id": "composer-2.5"}])
        _save("agy", "default", [{"id": "gemini-3", "name": "Gemini 3"}])
        calls = (
            codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30, effort="high"),
            codex_mod.run_codex_subprocess("hi", "missing", 30, effort="medium"),
            claude_mod.run_claude_subprocess("hi", "opus", 30, effort="high"),
            claude_mod.run_claude_subprocess("hi", "missing", 30, effort="medium"),
            grok_mod.run_grok_subprocess("hi", "grok-4.5", 30, effort="high"),
            grok_mod.run_grok_subprocess("hi", "missing", 30, effort="medium"),
            cursor_mod.run_cursor_subprocess("hi", "missing", 30),
            agy_mod.run_agy("hi", "missing", 30),
        )
        for coro in calls:
            fake = FakeRun()
            _SDKAgent.calls = []
            with mock.patch.object(agy_mod, "Agent", _SDKAgent), \
                    mock.patch.object(agy_mod, "HAS_SDK", True):
                result = self._run(fake, coro)
            self.assertFalse(result.get("ok"), result)
            self.assertIn(result.get("reason"), ("unsupported_model", "unsupported_effort"))
            self.assertEqual(fake.calls, [])
            self.assertEqual(_SDKAgent.calls, [])
            self.assertIn("models refresh --provider", result.get("detail", ""))

    def test_agent_listed_argv(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["low", "medium", "high"]}])
        _save("claude", "default", [{"id": "opus", "efforts": ["low", "medium", "high"]}])
        _save("grok", "default", [{"id": "grok-4.5", "efforts": ["low", "medium", "high"]}])
        _save("cursor", "default", [{"id": "composer-2.5-high"}])
        specs = (
            ("codex", "codex/gpt-5.4", "high"),
            ("claude", "claude/opus", "high"),
            ("grok", "grok-4.5", "high"),
            ("cursor", "cursor/composer-2.5", "high"),
        )
        for provider, model, effort in specs:
            fake = FakeRun('{"result": "OK"}')
            result = self._run(fake, agent_lane.run_agent(
                "task", model, self.work, read_only=False, effort=effort, timeout_s=30))
            self.assertTrue(result.get("ok"), (provider, result))
            self.assertEqual(fake.calls and len(fake.calls), 1, provider)
            argv = fake.calls[0][0]
            if provider == "codex":
                self.assertEqual(argv, [
                    BIN["codex"], "exec", "--sandbox", "workspace-write",
                    "--skip-git-repo-check", "--json", "--ephemeral",
                    "-c", "model_reasoning_effort=high",
                    "-m", "gpt-5.4", "-o", argv[argv.index("-o") + 1], "-",
                ])
            elif provider == "claude":
                self.assertEqual(argv, [
                    BIN["claude"], "-p", "--setting-sources", "project",
                    "--model", "opus", "--dangerously-skip-permissions",
                    "--strict-mcp-config", "--effort", "high",
                ])
            elif provider == "grok":
                self.assertEqual(argv, [
                    BIN["grok"], "--prompt-file", argv[2], "--model", "grok-4.5",
                    "--reasoning-effort", "high", "--output-format", "plain",
                    "--no-memory", "--sandbox", "strict", "--cwd", self.work,
                ])
            else:
                self.assertEqual(argv, [
                    BIN["cursor"], "-p", "--trust", "--sandbox", "enabled",
                    "--workspace", self.work, "--model", "composer-2.5-high",
                    "--force", "--output-format", "json",
                ])
                self.assertEqual(result.get("effort"), "high")
                self.assertEqual(result.get("effort_source"), "request")

    def test_agent_unlisted_never_spawns(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["medium"]}])
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        _save("grok", "default", [{"id": "grok-4.5", "efforts": ["medium"]}])
        _save("cursor", "default", [{"id": "composer-2.5"}])
        _save("agy", "default", [
            {"id": "claude-opus-5-5-high", "name": "Claude Opus 5.5 (High)",
             "display_name": "Claude Opus 5.5 (High)"},
        ])
        specs = (
            ("codex/gpt-5.4", "high"),
            ("codex/missing", "medium"),
            ("claude/opus", "high"),
            ("claude/missing", "medium"),
            ("grok-4.5", "high"),
            ("grok-missing", "medium"),
            ("cursor/missing", None),
            ("agy-opus-5.5", "low"),
        )
        for model, effort in specs:
            fake = FakeRun()
            _SDKAgent.calls = []
            result = self._run(fake, agent_lane.run_agent(
                "task", model, self.work, read_only=False, effort=effort, timeout_s=30))
            self.assertFalse(result.get("ok"), (model, result))
            self.assertEqual(fake.calls, [], model)
            self.assertEqual(_SDKAgent.calls, [], model)
            self.assertIn(result.get("reason"), (
                "unsupported_model", "unsupported_effort", "unsupported_agent",
                "effort_already_encoded", "no_catalog"), model)

    def test_omitted_effort_becomes_medium(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["medium", "high"]}])
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        _save("grok", "default", [{"id": "grok-4.5", "efforts": ["medium"]}])
        _save("cursor", "default", [{"id": "claude-fable-5-2-medium"}])
        specs = (
            ("codex", codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30),
             "model_reasoning_effort=medium"),
            ("claude", claude_mod.run_claude_subprocess("hi", "opus", 30), "--effort"),
            ("grok", grok_mod.run_grok_subprocess("hi", "grok-4.5", 30), "--reasoning-effort"),
        )
        for name, coro, flag in specs:
            fake = FakeRun('{"result": "OK"}')
            result = self._run(fake, coro)
            self.assertTrue(result.get("ok"), (name, result))
            self.assertEqual(result.get("effort"), "medium")
            self.assertEqual(result.get("effort_source"), "omni_default")
            self.assertEqual(result.get("effort_note"), NOTE_DEFAULT)
            argv = fake.calls[0][0]
            if name == "codex":
                self.assertIn(flag, argv)
            else:
                self.assertEqual(argv[argv.index(flag) + 1], "medium")
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, cursor_mod.run_cursor_subprocess("hi", "fable", 30))
        self.assertTrue(result.get("ok"), result)
        argv = fake.calls[0][0]
        self.assertEqual(argv[argv.index("--model") + 1], "claude-fable-5-2-medium")
        self.assertEqual(result.get("effort"), "medium")
        self.assertEqual(result.get("effort_source"), "omni_default")
        self.assertEqual(result.get("effort_note"), NOTE_DEFAULT)

    def test_codex_empty_and_absent_efforts_send_no_flag(self):
        for rows in (
            [{"id": "gpt-5.4", "efforts": []}],
            [{"id": "gpt-5.4"}],
        ):
            state = tempfile.TemporaryDirectory(prefix="omni-admit-empty-")
            self.addCleanup(state.cleanup)
            os.environ["OMNI_LLM_STATE_DIR"] = state.name
            _save("codex", "default", rows)
            fake = FakeRun()
            result = self._run(fake, codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30))
            self.assertTrue(result.get("ok"), result)
            argv = fake.calls[0][0]
            self.assertFalse(any(str(part).startswith("model_reasoning_effort=") for part in argv))
            self.assertIsNone(result.get("effort"))
            self.assertIn(result.get("effort_source"), ("not_applicable", "vendor"))

    def test_agy_catalog_id_dispatches_and_opus_form_refuses(self):
        _save("agy", "default", [{"id": "agy-one", "name": "Agy One"}])
        fake = FakeRun()
        _SDKAgent.calls = []
        admitted = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "agy/agy-one", 30, None, "default"))
        self.assertTrue(admitted.get("ok"), admitted)
        self.assertTrue(fake.calls)

        fake = FakeRun()
        _SDKAgent.calls = []
        refused = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "agy-opus-5.5@high", 30, None, "default"))
        self.assertEqual(refused.get("reason"), "unsupported_model")
        self.assertIn("agy/<id>", refused.get("detail", ""))
        self.assertEqual(fake.calls, [])
        self.assertEqual(_SDKAgent.calls, [])

    def test_retired_opus_effort_forms_are_unsupported_model(self):
        from providers.seat_name import take_effort
        cases = (
            take_effort("agy-opus-5.5@med", None),
            take_effort("agy-opus-5.5@high", "high"),
        )
        for bare, _effort, err in cases:
            self.assertIsNone(bare)
            self.assertEqual(err.get("reason"), "unsupported_model")
            self.assertIn("agy/<id>", err.get("detail", ""))

    def test_retired_opus_chain_refuses_before_health_or_next_seat(self):
        ledger = os.path.join(self.state.name, "health.json")
        payload = (
            b'{ "ollama/safe@default" : {"until": 1, "source": "rate_limit"} }\n'
        )
        with open(ledger, "wb") as handle:
            handle.write(payload)
        fake = FakeRun()
        spawned = mock.AsyncMock(return_value={"ok": True, "text": "spawned"})
        with mock.patch.object(dispatch.openai_mod, "run_openai_compatible", spawned):
            refused = self._run(fake, dispatch.dispatch_llm_request(
                "hi", "agy-opus-5.5@high,ollama/safe", 30, None, "default"))
        self.assertEqual(refused.get("reason"), "unsupported_model")
        self.assertIn("agy/<id>", refused.get("detail", ""))
        self.assertEqual(fake.calls, [])
        spawned.assert_not_called()
        with open(ledger, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_agy_id_plus_effort_is_effort_already_encoded(self):
        _save("agy", "default", [
            {"id": "claude-opus-5-5-high", "name": "Claude Opus 5.5 (High)",
             "display_name": "Claude Opus 5.5 (High)"},
        ])
        fake = FakeRun()
        _SDKAgent.calls = []
        with mock.patch.object(agy_mod, "Agent", _SDKAgent), \
                mock.patch.object(agy_mod, "HAS_SDK", True):
            result = self._run(fake, agy_mod.run_agy(
                "hi", "claude-opus-5-5-high", 30, effort="high"))
        self.assertEqual(result.get("reason"), "effort_already_encoded")
        self.assertEqual(fake.calls, [])
        self.assertEqual(_SDKAgent.calls, [])
        self.assertIn("models refresh --provider agy --profile default", result["detail"])

    def test_ring_skips_account_missing_the_model(self):
        home = tempfile.TemporaryDirectory(prefix="omni-admit-homes-")
        self.addCleanup(home.cleanup)
        root = os.path.realpath(home.name)
        homes = {}
        for label in ("a", "b"):
            path = os.path.join(root, label)
            os.mkdir(path)
            homes[label] = path
            os.environ["CODEX_HOME_" + label.upper()] = path
        self.addCleanup(lambda: [os.environ.pop("CODEX_HOME_" + label.upper(), None)
                                 for label in ("a", "b")])
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "b,a"
        _save("codex", "b", [{"id": "other", "efforts": ["high"]}])
        _save("codex", "a", [{"id": "gpt-5.4", "efforts": ["high"]}])
        fake = FakeRun()
        result = self._run(fake, codex_mod.run_codex_subprocess(
            "hi", "gpt-5.4", 30, effort="high"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(len(fake.calls), 1)
        env = fake.calls[0][1].get("env") or {}
        self.assertEqual(os.path.realpath(env.get("CODEX_HOME", "")), homes["a"])
        argv = fake.calls[0][0]
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.4")
        self.assertIn("model_reasoning_effort=high", argv)

    def test_comma_chain_catalog_refusal_stops(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["medium", "high"]}])
        fake = FakeRun()
        result = self._run(fake, dispatch._dispatch_llm_request(
            "hi", "codex/missing,codex/gpt-5.4", 30, None, "default", effort="high"))
        self.assertEqual(result.get("reason"), "unsupported_model")
        self.assertEqual(fake.calls, [])
        self.assertIn("models refresh --provider codex --profile default", result["detail"])

    def test_chain_runs_a_today_attempt_the_old_precheck_refused(self):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "a"
        home = tempfile.TemporaryDirectory(prefix="omni-admit-claude-a-")
        self.addCleanup(home.cleanup)
        os.environ["CLAUDE_CONFIG_DIR_A"] = os.path.realpath(home.name)
        self.addCleanup(lambda: os.environ.pop("CLAUDE_CONFIG_DIR_A", None))
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, dispatch._dispatch_llm_request(
            "hi", "claude/opus,claude/sonnet", 30, None, "a"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("--model", fake.calls[0][0])
        self.assertEqual(fake.calls[0][0][fake.calls[0][0].index("--model") + 1], "opus")

    def test_secret_discovery_endpoint_refuses_with_zero_spawns(self):
        os.environ["GROK_BASE_URL"] = "https://example.invalid/?token=synthetic"
        self.addCleanup(lambda: os.environ.pop("GROK_BASE_URL", None))
        fake = FakeRun()
        result = self._run(fake, grok_mod.run_grok_subprocess("hi", "grok-4.6", 30))
        self.assertEqual(fake.calls, [])
        self.assertEqual(result.get("reason"), "catalog_unreadable")
        self.assertNotEqual(result.get("effort_source"), "today_gate")
        self.assertIn("models refresh --provider grok --profile default", result["detail"])

    def test_no_record_names_refresh(self):
        fake = FakeRun()
        result = self._run(fake, codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30))
        self.assertEqual(result.get("reason"), "no_catalog")
        self.assertEqual(fake.calls, [])
        self.assertIn("models refresh --provider codex --profile default", result["detail"])

    def test_today_gate_codex_host_base_url(self):
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
        fake = FakeRun()
        result = self._run(fake, codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(result.get("effort"), DEFAULT_CODEX_REASONING_EFFORT)
        argv = fake.calls[0][0]
        self.assertIn("model_reasoning_effort=%s" % DEFAULT_CODEX_REASONING_EFFORT, argv)

    def test_today_gate_claude_isolated_api_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        home = tempfile.TemporaryDirectory(prefix="omni-admit-claude-")
        self.addCleanup(home.cleanup)
        path = os.path.realpath(home.name)
        os.environ["CLAUDE_CONFIG_DIR_A"] = path
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "a"
        self.addCleanup(lambda: os.environ.pop("CLAUDE_CONFIG_DIR_A", None))
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, claude_mod.run_claude_subprocess("hi", "opus", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("--effort", fake.calls[0][0])

    def test_today_gate_claude_project_settings(self):
        project = os.path.join(self.work, ".claude")
        os.makedirs(project)
        with open(os.path.join(project, "settings.json"), "w", encoding="utf-8") as handle:
            json.dump({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}}, handle)
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, agent_lane.run_agent(
            "task", "claude/opus", self.work, read_only=False, timeout_s=30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertNotIn("--effort", fake.calls[0][0])

    def test_today_gate_unparseable_settings(self):
        path = os.path.join(_HOME.home, ".claude", "settings.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, claude_mod.run_claude_subprocess("hi", "opus", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)

    def test_today_gate_invalid_utf8_settings(self):
        path = os.path.join(_HOME.home, ".claude", "settings.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"\xff\xfe{")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, claude_mod.run_claude_subprocess("hi", "opus", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)

    def test_read_only_agent_checks_user_settings_local(self):
        path = os.path.join(_HOME.home, ".claude", "settings.local.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"apiKeyHelper": "printf test"}, handle)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        fake = FakeRun('{"result": "OK"}')
        result = self._run(fake, agent_lane.run_agent(
            "task", "claude/opus", self.work, read_only=True, timeout_s=30))
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)

    def test_consult_checks_project_settings_local(self):
        project = os.path.join(self.work, ".claude")
        os.makedirs(project)
        with open(os.path.join(project, "settings.local.json"), "w", encoding="utf-8") as handle:
            json.dump({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}}, handle)
        _save("claude", "default", [{"id": "opus", "efforts": ["medium"]}])
        fake = FakeRun('{"result": "OK"}')
        previous = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(lambda: os.chdir(previous))
        result = self._run(fake, claude_mod.run_claude_subprocess("hi", "opus", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertEqual(len(fake.calls), 1)

    def test_early_refusal_carries_empty_disclosure(self):
        os.environ["OMNI_EGRESS_GATE"] = "1"
        self.addCleanup(lambda: os.environ.pop("OMNI_EGRESS_GATE", None))
        fake = FakeRun()
        consult = self._run(fake, dispatch.dispatch_llm_request(
            "call " + _SYNTHETIC_MOBILE, "codex/gpt-5.4", 30, None, "default"))
        self.assertIn("OMNI-EGRESS-DENY", consult.get("detail", ""))
        self.assertEqual(fake.calls, [])
        for key in ("effort", "effort_source", "effort_note"):
            self.assertIn(key, consult)
            self.assertIsNone(consult[key])
        agent = self._run(fake, agent_lane.run_agent(
            "call " + _SYNTHETIC_MOBILE, "codex/gpt-5.4", self.work, timeout_s=30))
        self.assertIn("OMNI-EGRESS-DENY", agent.get("detail", ""))
        self.assertEqual(fake.calls, [])
        for key in ("effort", "effort_source", "effort_note"):
            self.assertIn(key, agent)
            self.assertIsNone(agent[key])
        profile = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "claude/opus", 30, None, "nope"))
        self.assertEqual(profile.get("reason"), "unsupported_profile")
        self.assertEqual(fake.calls, [])
        for key in ("effort", "effort_source", "effort_note"):
            self.assertIn(key, profile)
            self.assertIsNone(profile[key])

    def test_mcp_result_is_one_json_text_item(self):
        _save("codex", "default", [{"id": "gpt-5.4", "efforts": ["medium"]}])
        sent = []

        async def capture(obj):
            sent.append(obj)

        fake = FakeRun()
        patches = self._patches(fake)
        patches.append(mock.patch("providers.mcp_server._send", capture))
        for patch in patches:
            patch.start()
        try:
            from providers.mcp_server import handle_tool_call
            asyncio.run(handle_tool_call("1", "llm_generate", {
                "prompt": "hi", "model": "codex/gpt-5.4"}))
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual(len(sent), 1)
        content = sent[0]["result"]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        payload = json.loads(content[0]["text"])
        self.assertEqual(payload.get("effort"), "medium")
        self.assertEqual(payload.get("effort_source"), "omni_default")
        self.assertEqual(payload.get("effort_note"), NOTE_DEFAULT)
        self.assertEqual(sent[0]["result"]["structuredContent"]["effort_note"], NOTE_DEFAULT)
        self.assertNotIn(NOTE_DEFAULT, payload.get("text", ""))

    def test_plain_writes_effort_note_to_stderr_only(self):
        from io import StringIO
        import omni_cli
        envelope = {
            "ok": True, "text": "hello\n", "model": "codex/gpt-5.4",
            "effort": "medium", "effort_source": "omni_default",
            "effort_note": NOTE_DEFAULT,
        }
        stdout, stderr = StringIO(), StringIO()
        with mock.patch.object(omni_cli, "_call", return_value=envelope), \
                mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            code = omni_cli.main(["gen", "-m", "codex/gpt-5.4", "--plain", "hi"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "hello\n")
        self.assertEqual(stderr.getvalue(), NOTE_DEFAULT + "\n")
        self.assertNotIn("effort_source", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
