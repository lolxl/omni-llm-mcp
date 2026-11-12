#!/usr/bin/env python3
# no-opencc
"""Protocol + loud-failure tests for omni-llm-mcp. No agy / no grok / no network needed.

These run anywhere (CI included): they exercise the JSON-RPC handshake, tools/list,
and the failure envelopes by pointing PATH at nothing so `agy` is absent. The happy
path (a real agy round-trip) is verified manually, not here, to keep tests hermetic.
TestGrokBackend covers the grok-* provider branch in-process with mocked subprocess
calls. Live vendor canaries are in tests.test_server_live; run that module
directly. The offline runner skips its ``_live`` ids.
"""
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-server-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

import server  # noqa: E402
from tests import offline_home  # noqa: E402

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True

_HOME = None
_STATE_PATCH = None


def _seed_agy(profile, model):
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    record, _warnings = validate_strict("agy", {
        "provider": "agy",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [{
            "id": model,
            "name": model,
            "routable_id": "agy/%s" % model,
        }],
    })
    catalog_v2.save("agy", profile, record)


def _seed_server_catalogs():
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    efforts = ["minimal", "low", "medium", "high", "xhigh", "max"]
    rows = []
    for model in ("gpt-5.6-sol",):
        rows.append({
            "id": model,
            "name": model,
            "routable_id": "codex/%s" % model,
            "supported_efforts": efforts,
        })
    record, _warnings = validate_strict("codex", {
        "provider": "codex",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": rows,
    })
    catalog_v2.save("codex", "default", record)
    grok, _warnings = validate_strict("grok", {
        "provider": "grok",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [{
            "id": "grok-4.5",
            "name": "grok-4.5",
            "routable_id": "grok/grok-4.5",
            "supported_efforts": [],
        }, {
            "id": "grok-4.6",
            "name": "grok-4.6",
            "routable_id": "grok/grok-4.6",
            "supported_efforts": [],
        }, {
            "id": "grok-4.7",
            "name": "grok-4.7",
            "routable_id": "grok/grok-4.7",
            "supported_efforts": ["medium"],
        }],
    })
    catalog_v2.save("grok", "default", grok)
    agy_ids = (
        "Gemini 3.8 Flash (High)",
        "Gemini 3.5 Flash",
        "Gemini 3.5 Flash (High)",
        "Model A",
        "Model B",
        "My Model",
        "Solo Model",
        # Timeout-budget tests call run_agy_subprocess with this id.
        "m",
    )
    agy, _warnings = validate_strict("agy", {
        "provider": "agy",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [{
            "id": model,
            "name": model,
            "routable_id": "agy/%s" % model,
        } for model in agy_ids],
    })
    catalog_v2.save("agy", "default", agy)


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module. It goes
    # first (and closes last): that helper scrubs the import-time state-dir pin.
    global _HOME, _STATE_PATCH
    _HOME = offline_home.use_synthetic_home()
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _TMP})
    _STATE_PATCH.start()
    _seed_server_catalogs()


def tearDownModule():
    try:
        _STATE_PATCH.stop()
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(HERE, "server.py")
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}


def run_server(messages, no_agy=False):
    env = dict(os.environ)
    if no_agy:
        env["PATH"] = "/nonexistent"          # shutil.which('agy') -> None
    inp = "".join(json.dumps(m) + "\n" for m in messages)
    proc = subprocess.run([sys.executable, SERVER], input=inp, text=True,
                          capture_output=True, env=env, timeout=30)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _by_id(out, id_):
    return [m for m in out if m.get("id") == id_][0]


class TestProtocol(unittest.TestCase):
    def test_initialize_echoes_protocol_version(self):
        out = run_server([INIT], no_agy=True)
        r = _by_id(out, 1)["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")
        self.assertIn("tools", r["capabilities"])
        self.assertEqual(r["serverInfo"]["name"], "omni-llm-mcp")

    def test_tools_list_exposes_expected_tools(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}], no_agy=True)
        tools = _by_id(out, 2)["result"]["tools"]
        self.assertEqual({t["name"] for t in tools},
                         {"llm_generate", "llm_big_context", "list_models", "llm_image"})
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_cli_missing_is_loud(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "llm_generate", "arguments": {"prompt": "hi"}}}], no_agy=True)
        call = _by_id(out, 3)["result"]
        self.assertTrue(call["isError"])
        payload = json.loads(call["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")

    def test_list_models_cli_missing_is_loud(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                                 "params": {"name": "list_models", "arguments": {}}}], no_agy=True)
        call = _by_id(out, 8)["result"]
        self.assertTrue(call["isError"])
        payload = json.loads(call["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")
        # backends stay attached even on LOUD failure (static routing map)
        self.assertIn("backends", payload)
        self.assertIn("agy", payload["backends"])
        self.assertIn("grok", payload["backends"])
        self.assertIn("openai", payload["backends"])

    def test_unknown_tool_is_error(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                 "params": {"name": "nope", "arguments": {}}}], no_agy=True)
        self.assertTrue(_by_id(out, 4)["result"]["isError"])

    def test_empty_prompt_rejected_without_calling_agy(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                 "params": {"name": "llm_generate", "arguments": {"prompt": "   "}}}], no_agy=True)
        payload = json.loads(_by_id(out, 5)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])  # rejected before any subprocess

    def test_big_context_requires_text(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                 "params": {"name": "llm_big_context",
                                            "arguments": {"instruction": "summarize", "text": ""}}}], no_agy=True)
        payload = json.loads(_by_id(out, 6)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])

    def test_unknown_method_returns_jsonrpc_error(self):
        out = run_server([{"jsonrpc": "2.0", "id": 9, "method": "bogus/method"}], no_agy=True)
        self.assertEqual(_by_id(out, 9)["error"]["code"], -32601)

    def test_notification_gets_no_response(self):
        # a lone notification (no id) must NOT produce a response line
        out = run_server([{"jsonrpc": "2.0", "method": "notifications/initialized"}], no_agy=True)
        self.assertEqual(out, [])

    def test_malformed_line_returns_parse_error_and_loop_survives(self):
        # a bad JSON line -> -32700 (id null), AND the loop must keep going (next request answered)
        raw = "{bad json not valid\n" + json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/list"}) + "\n"
        env = dict(os.environ)
        env["PATH"] = "/nonexistent"
        proc = subprocess.run([sys.executable, SERVER], input=raw, text=True,
                              capture_output=True, env=env, timeout=30)
        out = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(out[0]["error"]["code"], -32700)
        self.assertIsNone(out[0]["id"])
        self.assertTrue(any(m.get("id") == 7 and "result" in m for m in out),
                        "loop must survive a parse error and still answer the next request")

    def test_payload_over_old_64kb_wall_gets_answer_and_loop_survives(self):
        # regression: asyncio.StreamReader's default 64 KiB limit used to kill the
        # whole stdio session on one oversized tools/call line — readline raised
        # ValueError, the loop broke, and the client saw silence instead of an
        # envelope (the one hole in the LOUD contract).
        big_text = "x" * (200 * 1024)  # 200 KiB, well past the old 64 KiB wall
        out = run_server([
            INIT,
            {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
             "params": {"name": "llm_big_context",
                        "arguments": {"instruction": "summarize", "text": big_text}}},
            {"jsonrpc": "2.0", "id": 11, "method": "tools/list"},
        ], no_agy=True)
        payload = json.loads(_by_id(out, 10)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])                  # structured failure, not a dead session
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertTrue(any(m.get("id") == 11 and "result" in m for m in out),
                        "loop must survive a >64KiB line and still answer the next request")

    def test_fallback_chain_tries_all_and_reports_tried(self):
        # comma-separated model = opt-in fallback. Both names route to agy (no prefix),
        # so with agy absent both fail cli_missing — hermetic, no network. The final
        # loud failure must carry a `tried` breadcrumb naming every model attempted.
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                                 "params": {"name": "llm_generate",
                                            "arguments": {"prompt": "hi", "model": "Model A,Model B"}}}], no_agy=True)
        payload = json.loads(_by_id(out, 12)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertEqual([t["model"] for t in payload["tried"]], ["Model A", "Model B"])

    def test_single_model_has_no_tried_breadcrumb(self):
        # a single model = no fallback: the envelope must stay identical to before
        # (no `tried` key) so existing callers see unchanged behavior.
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                                 "params": {"name": "llm_generate",
                                            "arguments": {"prompt": "hi", "model": "Solo Model"}}}], no_agy=True)
        payload = json.loads(_by_id(out, 13)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertNotIn("tried", payload)


class _FakeStream:
    """Minimal asyncio StreamReader stand-in for run_cli's concurrent drain."""

    def __init__(self, data=b""):
        self._data = data if isinstance(data, (bytes, bytearray)) else bytes(data or b"")
        self._pos = 0

    async def read(self, n=-1):
        if self._pos >= len(self._data):
            return b""
        if n is None or n < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return bytes(chunk)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return bytes(chunk)


class _FakeStdin:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    async def drain(self):
        return None

    def close(self):
        self.closed = True


class _FakeGrokProc:
    """Stand-in for the asyncio subprocess handle the grok CLI would return."""

    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._stdout_bytes = stdout
        self._stderr_bytes = stderr
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.stdin = _FakeStdin()
        self.returncode = None
        self._final_rc = returncode
        self.pid = 0x7FFFFFF0  # unlikely real pgid; killpg → ProcessLookupError
        self.killed = False

    async def communicate(self, input=None):
        self.returncode = self._final_rc
        return self._stdout_bytes, self._stderr_bytes

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = self._final_rc
        return self.returncode


def _grok_stream(argv, text, usage, tools=()):
    """grok --output-format streaming-json for one consult turn."""
    sid = argv[argv.index("--session-id") + 1]
    lines = [{"type": "available_commands", "tools": list(tools)},
             {"type": "text", "data": text},
             {"type": "end", "stopReason": "end_turn", "sessionId": sid, "usage": usage}]
    return "".join(json.dumps(line) + "\n" for line in lines)


class TestGrokBackend(unittest.TestCase):
    """In-process tests for the grok-* provider branch (_route_to_provider +
    _run_grok_subprocess). Every subprocess is mocked — no real Grok CLI, no
    network — keeping the suite as hermetic as TestProtocol above."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        # Isolated GROK_HOME: the consult lane must never read or need ~/.grok.
        self.grok_home = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_home_"))
        self.addCleanup(shutil.rmtree, self.grok_home, True)
        patcher = mock.patch.dict(os.environ, {"GROK_HOME": self.grok_home})
        patcher.start()
        self.addCleanup(patcher.stop)
        # The module seed was fingerprinted before this home existed.
        _seed_server_catalogs()

    def test_grok_prefix_routes_to_grok_backend_and_others_do_not(self):
        called = []

        async def fake_grok(prompt, model, timeout_s, account_profile="default", files=None):
            called.append(("grok", model))
            return {"ok": True, "text": "stub", "model": model}

        async def fake_agy(prompt, model, timeout_s, files=None, account_profile="default"):
            called.append(("agy", model))
            return {"ok": True, "text": "stub", "model": model}

        with mock.patch("providers.grok.run_grok_subprocess", fake_grok), \
             mock.patch("providers.agy.run_agy", fake_agy):
            asyncio.run(server._route_to_provider("hi", "grok-4.5", 30, [], "default"))
            asyncio.run(server._route_to_provider("hi", "Gemini 3.5 Flash", 30, [], "default"))

        self.assertEqual(called, [("grok", "grok-4.5"), ("agy", "Gemini 3.5 Flash")])

    def test_grok_cli_missing_is_loud(self):
        with mock.patch("shutil.which", return_value=None):
            res = asyncio.run(server._run_grok_subprocess("hi", "grok-4.5", 30))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "cli_missing")

    # Legacy transport flags are unchanged off macOS; host coverage is separate.
    @mock.patch("providers.host_fence.platform.system", new=lambda: "Linux")
    def test_grok_hardened_flags_and_success_envelope(self):
        captured = {}

        async def fake_exec(*argv, **kwargs):
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            with open(argv[argv.index("--prompt-file") + 1]) as f:
                captured["prompt_content"] = f.read()
            # streaming-json: reply from `text` events, counts from `end.usage`
            # (the same field names json mode carried since #22 P1.1).
            payload = _grok_stream(argv, "OK", {"input_tokens": 12,
                                                "cache_read_input_tokens": 4,
                                                "cache_creation_input_tokens": 0,
                                                "output_tokens": 3})
            return _FakeGrokProc(stdout=payload.encode(), stderr=b"", returncode=0)

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            res = asyncio.run(server._run_grok_subprocess("hello grok", "grok-4.5", 30))

        argv = captured["argv"]
        # hardened pipe: every flag, in order — --sandbox strict is the load-bearing wall
        # off macOS; the tool flags empty the tool set (see the tool tests below)
        flags = [a for a in argv if a.startswith("--")]
        self.assertEqual(flags, ["--prompt-file", "--model", "--output-format",
                                 "--no-memory", "--verbatim", "--disable-web-search", "--tools",
                                 "--disallowed-tools", "--session-id",
                                 "--sandbox", "--cwd"])
        # streaming-json: the tool verdict is read from grok's own stdout
        self.assertEqual(argv[argv.index("--output-format") + 1], "streaming-json")
        self.assertIsNotNone(__import__("uuid").UUID(argv[argv.index("--session-id") + 1]))
        self.assertEqual(argv[argv.index("--tools") + 1], "todo_write")
        self.assertEqual(argv[argv.index("--disallowed-tools") + 1],
                         "todo_write,search_tool,use_tool")
        self.assertEqual(argv[argv.index("--sandbox") + 1], "strict")  # kernel wall — must never regress
        tmp_cwd = argv[argv.index("--cwd") + 1]
        self.assertEqual(captured["kwargs"]["cwd"], tmp_cwd)   # launched inside the throwaway dir
        self.assertEqual(captured["prompt_content"], "hello grok")
        self.assertEqual(res["ok"], True)
        self.assertEqual(res["text"], "OK")
        self.assertEqual(res["model"], "grok-4.5")
        # summed: grok reports the UNCACHED part, so logical input is 12+4+0
        self.assertEqual(res["usage"]["input_tokens"], 16)
        self.assertEqual(res["usage"]["cached_input_tokens"], 4)
        self.assertEqual(res["usage"]["cache_write_tokens"], 0)
        self.assertEqual(res["usage"]["output_tokens"], 3)
        self.assertIsNone(res["usage"]["reasoning_tokens"])     # not in this payload
        self.assertEqual(res["usage"]["source"], "provider")
        # temp prompt file and throwaway cwd must be cleaned up afterwards
        self.assertFalse(os.path.exists(argv[argv.index("--prompt-file") + 1]))
        self.assertFalse(os.path.isdir(tmp_cwd))

    def test_grok_auth_error_is_typed(self):
        async def fake_exec(*argv, **kwargs):
            return _FakeGrokProc(stdout=b"", stderr=b"error: please run `grok login` first",
                                 returncode=1)

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            res = asyncio.run(server._run_grok_subprocess("hi", "grok-4.5", 30))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_grok_model_via_full_pipe_hits_grok_branch(self):
        # end-to-end through the JSON-RPC pipe, same hermetic style as TestProtocol:
        # with PATH pointing at nothing the grok CLI is absent, so a grok-4.5 call
        # must fail cli_missing with the GROK detail — proving tools/call routing
        # picked the grok branch, not the agy default.
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 14, "method": "tools/call",
                                 "params": {"name": "llm_generate",
                                            "arguments": {"prompt": "hi", "model": "grok-4.5"}}}], no_agy=True)
        payload = json.loads(_by_id(out, 14)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertIn("Grok Build CLI", payload["detail"])


class _FakeCodexProc:
    """Stand-in for the asyncio subprocess handle the codex CLI would return."""

    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._stdout_bytes = stdout
        self._stderr_bytes = stderr
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.stdin = _FakeStdin()
        self.returncode = None
        self._final_rc = returncode
        self.received_input = None
        self.pid = 0x7FFFFFF1
        self.killed = False

    async def communicate(self, input=None):
        self.received_input = input
        self.returncode = self._final_rc
        return self._stdout_bytes, self._stderr_bytes

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        # Mirror stdin capture used by older communicate()-based callers.
        if self.received_input is None and self.stdin.written:
            self.received_input = self.stdin.written
        if self.returncode is None:
            self.returncode = self._final_rc
        return self.returncode


class _FakeRaisingStream:
    def __init__(self, exc):
        self._exc = exc

    async def read(self, n=-1):
        raise self._exc


class _FakeAgyProc:
    """agy subprocess handle whose stream read raises, to exercise the orphan
    guard: returncode stays None (still running) until kill() reaps it."""

    def __init__(self, exc):
        self._exc = exc
        self.returncode = None
        self.killed = False
        self.stdout = _FakeRaisingStream(exc)
        self.stderr = _FakeStream(b"")
        self.stdin = _FakeStdin()
        self.pid = 0x7FFFFFF2

    async def communicate(self, input=None):
        raise self._exc

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


class TestCodexBackend(unittest.TestCase):
    """In-process tests for the codex/ provider branch (_route_to_provider +
    _run_codex_subprocess). Every subprocess is mocked — no real Codex CLI, no
    network — hermetic like TestGrokBackend. The live vendor canaries are in
    tests.test_server_live."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_codex_route_snapshot(self):
        # acceptance #3: codex/<model> -> codex backend with the prefix stripped;
        # gpt-4o stays on openai (UNCHANGED); default names stay agy.
        cases = [
            ("codex/gpt-5.6-sol", "codex", "gpt-5.6-sol"),
            ("codex/o4-mini", "codex", "o4-mini"),
            ("gpt-4o", "openai", "gpt-4o"),
            ("Gemini 3.5 Flash", "agy", "Gemini 3.5 Flash"),
        ]
        for name, provider, routed in cases:
            route = server.resolve_route(name)
            self.assertEqual(route["provider"], provider, name)
            self.assertEqual(route["model"], routed, name)

    def test_codex_prefix_routes_to_codex_backend_and_others_do_not(self):
        called = []

        async def fake_codex(prompt, model, timeout_s, account_profile="default", files=None,
                             apply_aliases=False):
            called.append(("codex", model))
            return {"ok": True, "text": "stub", "model": model}

        async def fake_openai(prompt, model, timeout_s, account_profile, api_base=None):
            called.append(("openai", model))
            return {"ok": True, "text": "stub", "model": model}

        async def fake_agy(prompt, model, timeout_s, files=None, account_profile="default"):
            called.append(("agy", model))
            return {"ok": True, "text": "stub", "model": model}

        with mock.patch("providers.codex.run_codex_subprocess", fake_codex), \
             mock.patch("providers.openai_compat.run_openai_compatible", fake_openai), \
             mock.patch("providers.agy.run_agy", fake_agy):
            asyncio.run(server._route_to_provider("hi", "codex/gpt-5.6-sol", 30, [], "default"))
            asyncio.run(server._route_to_provider("hi", "gpt-4o", 30, [], "default"))
            asyncio.run(server._route_to_provider("hi", "Gemini 3.5 Flash", 30, [], "default"))

        # codex/ peels the prefix (codex sees "gpt-5.6-sol"); gpt-4o still openai; default agy
        self.assertEqual(called, [("codex", "gpt-5.6-sol"),
                                  ("openai", "gpt-4o"),
                                  ("agy", "Gemini 3.5 Flash")])

    def test_codex_cli_missing_is_loud(self):
        with mock.patch("shutil.which", return_value=None):
            res = asyncio.run(server._run_codex_subprocess("hi", "gpt-5.6-sol", 30))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "cli_missing")

    def test_codex_hardened_flags_and_success_envelope(self):
        captured = {}

        async def fake_exec(*argv, **kwargs):
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            # codex writes the FINAL agent message to the -o file; mirror that so the
            # provider parses the reply from the file, not the (stderr) banner.
            out_path = argv[argv.index("-o") + 1]
            with open(out_path, "w") as f:
                f.write("OK")
            # --json (issue #22 P1.1) puts an event stream on stdout while the reply
            # still comes from the -o file. usage rides the final turn.completed.
            events = "".join(json.dumps(e) + "\n" for e in (
                {"type": "thread.started", "thread_id": "t1"},
                {"type": "turn.completed",
                 "usage": {"input_tokens": 21, "cached_input_tokens": 7,
                           "cache_write_input_tokens": 0, "output_tokens": 2,
                           "reasoning_output_tokens": 0}},
            ))
            proc = _FakeCodexProc(stdout=events.encode(),
                                  stderr=b"OpenAI Codex banner...", returncode=0)
            captured["proc"] = proc
            return proc

        with mock.patch("shutil.which", return_value="/fake/bin/codex"), \
             mock.patch("providers.codex.kernel_sandbox.available", return_value=False), \
             mock.patch("providers.host_fence.platform.system", return_value="Linux"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            res = asyncio.run(server._run_codex_subprocess("hello codex", "gpt-5.6-sol", 30))

        argv = captured["argv"]
        self.assertEqual(argv[1], "exec")
        # Non-macOS retains the pre-migration vendor invocation contract.
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertIn("--skip-git-repo-check", argv)   # REQUIRED for the empty tempdir cwd
        self.assertIn("--ephemeral", argv)
        # NEVER any bypass / write / full-access escape hatch
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertNotIn("--dangerously-bypass-hook-trust", argv)
        self.assertNotIn("workspace-write", argv)
        self.assertNotIn("danger-full-access", argv)
        # model + omitted-effort default (medium, when listed) + reply capture
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-sol")
        self.assertEqual(argv[argv.index("-c") + 1], "model_reasoning_effort=medium")
        self.assertEqual(argv[-1], "-")                                  # prompt via stdin sentinel
        self.assertEqual(captured["proc"].received_input, b"hello codex")
        out_path = argv[argv.index("-o") + 1]
        tmp_cwd = captured["kwargs"]["cwd"]
        self.assertIn("--json", argv)          # event stream carries usage (#22 P1.1)
        # reply is the final agent message from the -o file, not the banner and not
        # the event stream — --json must not leak JSONL into the user-facing text
        self.assertEqual(res["ok"], True)
        self.assertEqual(res["text"], "OK")
        self.assertEqual(res["model"], "gpt-5.6-sol")
        self.assertEqual(res["usage"]["input_tokens"], 21)
        self.assertEqual(res["usage"]["cached_input_tokens"], 7)
        self.assertEqual(res["usage"]["cache_write_tokens"], 0)
        self.assertEqual(res["usage"]["output_tokens"], 2)
        self.assertIsNone(res["usage"]["total_tokens"])   # codex reports no total
        self.assertEqual(res["usage"]["source"], "provider")
        # throwaway cwd and reply file must be cleaned up afterwards
        self.assertFalse(os.path.isdir(tmp_cwd))
        self.assertFalse(os.path.exists(out_path))

    def test_codex_reasoning_effort_passthrough(self):
        captured = {}

        async def fake_exec(*argv, **kwargs):
            captured["argv"] = list(argv)
            out_path = argv[argv.index("-o") + 1]
            with open(out_path, "w") as f:
                f.write("OK")
            return _FakeCodexProc(returncode=0)

        with mock.patch("shutil.which", return_value="/fake/bin/codex"), \
             mock.patch("providers.codex.kernel_sandbox.available", return_value=False), \
             mock.patch("providers.host_fence.platform.system", return_value="Linux"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            # Admitted consults honor effort=, not the legacy reasoning_effort default.
            res = asyncio.run(server._run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, effort="xhigh"))
        argv = captured["argv"]
        self.assertEqual(argv[argv.index("-c") + 1], "model_reasoning_effort=xhigh")
        self.assertTrue(res["ok"])

    def test_codex_auth_error_is_typed(self):
        async def fake_exec(*argv, **kwargs):
            return _FakeCodexProc(stdout=b"", stderr=b"error: run `codex login` first",
                                  returncode=1)

        with mock.patch("shutil.which", return_value="/fake/bin/codex"), \
             mock.patch("providers.codex.kernel_sandbox.available", return_value=False), \
             mock.patch("providers.host_fence.platform.system", return_value="Linux"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            res = asyncio.run(server._run_codex_subprocess("hi", "gpt-5.6-sol", 30))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_codex_model_via_full_pipe_hits_codex_branch(self):
        # end-to-end through the JSON-RPC pipe, same hermetic style as TestProtocol:
        # with PATH pointing at nothing the codex CLI is absent, so a codex/... call
        # must fail cli_missing with the Codex detail — proving tools/call routing
        # picked the codex branch, not the agy default.
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 15, "method": "tools/call",
                                 "params": {"name": "llm_generate",
                                            "arguments": {"prompt": "hi",
                                                          "model": "codex/gpt-5.6-sol"}}}], no_agy=True)
        payload = json.loads(_by_id(out, 15)["result"]["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "cli_missing")
        self.assertIn("Codex CLI", payload["detail"])

    def test_codex_consult_wraps_with_kernel_read_wall(self):
        """Hermetic: when the kernel fence is available, consult argv is wrapped
        with a read_wall profile (sandbox-exec prefix + deny file-read* of home)."""
        captured = {}

        async def fake_run_cli(argv, **kwargs):
            captured["argv"] = list(argv)
            if "-o" in argv:
                path = argv[argv.index("-o") + 1]
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("OK")
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        with mock.patch("shutil.which", side_effect=lambda n, *a, **k: {
                "codex": "/fake/bin/codex",
                "sandbox-exec": "/usr/bin/sandbox-exec",
            }.get(os.path.basename(str(n)) if n else "", None)), \
             mock.patch("providers.codex.kernel_sandbox.available", return_value=True), \
             mock.patch("providers.codex.kernel_sandbox.verify_fence", return_value=None), \
             mock.patch("providers.codex.run_cli", fake_run_cli):
            # Use real wrap_argv / build_profile so the profile text is asserted.
            res = asyncio.run(server._run_codex_subprocess("hi", "gpt-5.6-sol", 30))
        self.assertTrue(res.get("ok"), res)
        argv = captured["argv"]
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        self.assertIn("-p", argv)
        profile = argv[argv.index("-p") + 1]
        self.assertIn("(deny file-read*", profile)
        self.assertIn("(deny file-write*)", profile)
        self.assertIn("exec", argv)
        self.assertNotIn("--sandbox", argv)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertNotIn("--dangerously-bypass-hook-trust", argv)
        self.assertNotIn("workspace-write", argv)
        self.assertNotIn("danger-full-access", argv)
        self.assertEqual(argv[-1], "-")

    def test_codex_consult_fail_closed_when_verify_fence_refuses(self):
        async def bomb(*_a, **_k):
            raise AssertionError("run_cli must not be called after fence refusal")

        with mock.patch("shutil.which", return_value="/fake/bin/codex"), \
             mock.patch("providers.codex.kernel_sandbox.available", return_value=True), \
             mock.patch("providers.codex.kernel_sandbox.verify_fence", return_value={
                 "ok": False,
                 "reason": "kernel_fence_unverified",
                 "detail": "probe failed",
             }), \
             mock.patch("providers.codex.run_cli", bomb):
            res = asyncio.run(server._run_codex_subprocess("hi", "gpt-5.6-sol", 30))
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "kernel_fence_unverified")

    def test_codex_consult_read_wall_public_llm_generate(self):
        """Composition: JSON-RPC tools/call → real fence → real stub subprocess.

        Enters via handle_message tools/call (same wire dispatch as the protocol
        tests). A stub `codex` on PATH is the child: it actually opens the
        planted home canary and a cwd marker, so the tool result depends on the
        kernel profile — not a hardcoded BLOCKED string. An unfenced control
        proves the same stub can read the canary; the fenced run must attempt the
        open and report EPERM. Fence-apply failure is a hard assert (never skip).
        Skip only when sandbox-exec is absent.
        """
        import io
        import errno
        from providers import kernel_sandbox as ksb

        if not ksb.available():
            self.skipTest("sandbox-exec not available (read wall not enforced)")

        home = ksb.protected_home()
        canary_dir = tempfile.mkdtemp(prefix=".codex_pub_canary_", dir=home)
        canary_path = os.path.join(canary_dir, "secret.txt")
        token = "CANARY_CODEX_READ_9Q4ZK7_END"
        marker = "CWD_MARK_9Q4ZK7_OK"
        probe_cwd = tempfile.mkdtemp(prefix="codex_pub_cwd_")
        bindir = tempfile.mkdtemp(prefix="codex_stub_bin_")
        try:
            with open(canary_path, "w", encoding="utf-8") as fh:
                fh.write(token + "\n")

            # Fence-apply failure is a hard failure (not skip).
            refusal = ksb.verify_fence(
                probe_cwd, read_only=True, read_wall=True,
            )
            self.assertIsNone(
                refusal,
                "fence-apply failure must fail the test, not skip: %s" % refusal,
            )

            # Stub CLI standing in for codex: output depends on what the kernel
            # profile actually allows the child to read.
            stub_path = os.path.join(bindir, "codex")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env python3\n"
                    "import os, sys\n"
                    "try:\n"
                    "    sys.stdin.read()\n"
                    "except Exception:\n"
                    "    pass\n"
                    "argv = sys.argv\n"
                    "out = None\n"
                    "if '-o' in argv:\n"
                    "    out = argv[argv.index('-o') + 1]\n"
                    "canary = os.environ.get('OMNI_TEST_CANARY_PATH', '')\n"
                    "marker_token = os.environ.get('OMNI_TEST_CWD_MARKER', 'CWD_MARK')\n"
                    "mp = os.path.join(os.getcwd(), 'cwd_marker.txt')\n"
                    "with open(mp, encoding='utf-8') as mf:\n"
                    "    got_marker = mf.read().strip()\n"
                    "if got_marker != marker_token:\n"
                    "    sys.exit(66)\n"
                    "canary_text = ''\n"
                    "canary_attempted = 0\n"
                    "canary_errno = 0\n"
                    "if canary:\n"
                    "    canary_attempted = 1\n"
                    "    try:\n"
                    "        with open(canary, encoding='utf-8') as cf:\n"
                    "            canary_text = cf.read()\n"
                    "    except OSError as exc:\n"
                    "        canary_errno = exc.errno or 0\n"
                    "        canary_text = ''\n"
                    "reply = ('MARKER=%s\\nREAD_ATTEMPTED=%d\\n'\n"
                    "         'CANARY_ERRNO=%d\\nCANARY=%s') % (\n"
                    "             got_marker, canary_attempted, canary_errno,\n"
                    "             canary_text.strip())\n"
                    "if out:\n"
                    "    with open(out, 'w', encoding='utf-8') as of:\n"
                    "        of.write(reply)\n"
                    "else:\n"
                    "    sys.stdout.write(reply)\n"
                    "sys.exit(0)\n"
                )
            os.chmod(stub_path, 0o755)

            env_patch = {
                "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
                "OMNI_TEST_CANARY_PATH": canary_path,
                "OMNI_TEST_CWD_MARKER": marker,
                "OMNI_LLM_CHILD_ENV_PASS": (
                    "OMNI_TEST_CANARY_PATH,OMNI_TEST_CWD_MARKER"),
            }

            async def _tools_call_via_jsonrpc():
                # Same capture idiom as test_protocol_2026.exchange: drive
                # handle_message so tools/call schedules handle_tool_call.
                server._write_lock = None
                buf = io.StringIO()
                old_out = sys.stdout
                sys.stdout = buf
                try:
                    await server.handle_message({
                        "jsonrpc": "2.0",
                        "id": 45,
                        "method": "tools/call",
                        "params": {
                            "name": "llm_generate",
                            "arguments": {
                                "prompt": "composition canary probe",
                                "model": "codex/gpt-5.6-sol",
                                "timeout_s": 30,
                            },
                        },
                    })
                    pending = {t for t in server._pending_tasks if not t.done()}
                    if pending:
                        await asyncio.wait(pending, timeout=60)
                finally:
                    sys.stdout = old_out
                lines = [
                    json.loads(ln)
                    for ln in buf.getvalue().splitlines() if ln.strip()
                ]
                return lines

            # The consult cwd is read-only. Plant its readable marker before
            # launch; a child write here would violate the boundary under test.
            with open(os.path.join(probe_cwd, "cwd_marker.txt"), "w") as fh:
                fh.write(marker)
            control_env = {
                "PATH": env_patch["PATH"],
                "HOME": home,
                "OMNI_TEST_CANARY_PATH": canary_path,
                "OMNI_TEST_CWD_MARKER": marker,
            }
            unfenced = subprocess.run(
                [stub_path], cwd=probe_cwd, env=control_env, input="",
                text=True, capture_output=True, timeout=10)
            self.assertEqual(unfenced.returncode, 0, unfenced.stderr)
            self.assertIn("READ_ATTEMPTED=1", unfenced.stdout)
            self.assertIn("CANARY_ERRNO=0", unfenced.stdout)
            self.assertIn(token, unfenced.stdout)
            self.assertIn(marker, unfenced.stdout)
            original_mkdtemp = tempfile.mkdtemp
            def fixture_mkdtemp(*args, **kwargs):
                if kwargs.get("prefix") == "codex_cwd_":
                    return probe_cwd
                return original_mkdtemp(*args, **kwargs)
            with mock.patch.dict(os.environ, env_patch, clear=False), \
                 mock.patch("providers.codex.tempfile.mkdtemp", side_effect=fixture_mkdtemp):
                out = asyncio.run(_tools_call_via_jsonrpc())

            msg = _by_id(out, 45)
            self.assertIn("result", msg, msg)
            result = msg["result"]
            # Prefer structuredContent (typed envelope); fall back to text block.
            payload = result.get("structuredContent")
            if not isinstance(payload, dict):
                payload = json.loads(result["content"][0]["text"])
            self.assertTrue(payload.get("ok"), payload)
            text = payload.get("text") or ""
            self.assertIn(
                marker, text,
                "cwd-readable marker must appear (stub ran under fence): %r" % text,
            )
            self.assertIn("READ_ATTEMPTED=1", text)
            self.assertIn("CANARY_ERRNO=%d" % errno.EPERM, text)
            self.assertNotIn(
                token, text,
                "read wall breached: tools/call result leaked home canary: %r" % text,
            )
        finally:
            shutil.rmtree(canary_dir, ignore_errors=True)
            shutil.rmtree(probe_cwd, ignore_errors=True)
            shutil.rmtree(bindir, ignore_errors=True)


class TestBackendsAndObservability(unittest.TestCase):

    def test_backends_snapshot_keys(self):
        snap = server._backends_snapshot()
        self.assertEqual(set(snap), {"agy", "grok", "codex", "claude", "pi", "cursor", "openai", "ollama", "lmstudio"})
        self.assertIn("files", snap["agy"])
        self.assertTrue(snap["agy"]["files"])
        self.assertTrue(snap["grok"]["files"])

    def test_backends_snapshot_read_wall_flags(self):
        """Issue #45: every backend advertises a boolean read_wall flag.

        F2/F3: grok tracks Seatbelt mechanism availability (not unconditional
        True); codex and claude require verify_fence(read_wall=True).
        """
        from providers import dispatch as dispatch_mod
        from providers import kernel_sandbox as ksb

        # --- matrix: codex true only with verified fence ---
        with mock.patch.object(ksb, "available", return_value=True), \
             mock.patch.object(ksb, "verify_fence", return_value=None):
            snap = dispatch_mod.backends_snapshot()
            self.assertIs(snap["codex"]["read_wall"], True)
            self.assertIs(snap["claude"]["read_wall"], True)
            self.assertIs(snap["agy"]["read_wall"], True)    # verified consult wall (#161)
            self.assertIs(snap["grok"]["read_wall"], True)

        with mock.patch.object(ksb, "available", return_value=True), \
             mock.patch.object(ksb, "verify_fence", return_value={
                 "ok": False,
                 "reason": "kernel_fence_unverified",
                 "detail": "simulated refuse",
             }):
            snap = dispatch_mod.backends_snapshot()
            self.assertIs(snap["codex"]["read_wall"], False)
            self.assertIs(snap["claude"]["read_wall"], False)
            self.assertIs(snap["agy"]["read_wall"], False)
            # grok still follows mechanism presence, not codex verify.
            self.assertIs(snap["grok"]["read_wall"], True)

        with mock.patch.object(ksb, "available", return_value=False):
            snap = dispatch_mod.backends_snapshot()
            self.assertIs(snap["codex"]["read_wall"], False)
            self.assertIs(snap["claude"]["read_wall"], False)
            self.assertIs(snap["agy"]["read_wall"], False)
            self.assertIs(snap["grok"]["read_wall"], False)

        # Live host: flags match real mechanism / verified fence.
        snap = server._backends_snapshot()
        for name, entry in snap.items():
            with self.subTest(backend=name):
                self.assertIn("read_wall", entry)
                self.assertIsInstance(entry["read_wall"], bool)
        self.assertIs(
            snap["grok"]["read_wall"],
            bool(ksb.available()),
        )
        if ksb.available() and ksb.verify_fence(
            None, read_only=True, read_wall=True,
        ) is None:
            self.assertIs(snap["codex"]["read_wall"], True)
            self.assertIs(snap["claude"]["read_wall"], True)
            self.assertIs(snap["agy"]["read_wall"], True)
        else:
            self.assertIs(snap["codex"]["read_wall"], False)
            self.assertIs(snap["claude"]["read_wall"], False)
            self.assertIs(snap["agy"]["read_wall"], False)
        for name in ("pi", "cursor", "openai", "ollama", "lmstudio"):
            self.assertIs(snap[name]["read_wall"], False, name)

    def test_backends_snapshot_probe_exception_degrades_not_raises(self):
        """F3: availability/verify exceptions degrade flags; never escape."""
        from providers import dispatch as dispatch_mod
        from providers import kernel_sandbox as ksb

        with mock.patch.object(
            ksb, "available", side_effect=RuntimeError("boom"),
        ):
            snap = dispatch_mod.backends_snapshot()
        self.assertIs(snap["grok"]["read_wall"], False)
        self.assertIs(snap["codex"]["read_wall"], False)
        self.assertIs(snap["claude"]["read_wall"], False)

        with mock.patch.object(ksb, "available", return_value=True), \
             mock.patch.object(
                 ksb, "verify_fence", side_effect=RuntimeError("probe boom"),
             ):
            snap = dispatch_mod.backends_snapshot()
        self.assertIs(snap["codex"]["read_wall"], False)
        self.assertIs(snap["claude"]["read_wall"], False)
        # grok only needs mechanism; available() returned True.
        self.assertIs(snap["grok"]["read_wall"], True)

        # list_models still returns a LOUD envelope (no exception escape).
        with mock.patch.object(
            ksb, "available", side_effect=RuntimeError("boom"),
        ), mock.patch("shutil.which", return_value=None):
            res = asyncio.run(dispatch_mod.list_models(5))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "cli_missing")
        self.assertIs(res["backends"]["codex"]["read_wall"], False)
        self.assertIs(res["backends"]["grok"]["read_wall"], False)

    def test_list_models_reports_whether_the_default_is_listed(self):
        from providers import dispatch as dispatch_mod
        from providers import agy as agy_mod
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fake = os.path.join(tmp, "agy")

        def run(listing):
            with open(fake, "w") as handle:
                handle.write("#!/bin/sh\nprintf '%s'\n" % listing)
            os.chmod(fake, 0o755)
            def fake_fence(argv, env):
                cwd = tempfile.mkdtemp(prefix="omni_agy_list_")
                return list(argv), env, cwd
            with mock.patch("shutil.which", return_value=fake), \
                    mock.patch.object(agy_mod, "fence_agy_spawn", fake_fence):
                return asyncio.run(dispatch_mod.list_models(5))

        # `agy models` as captured 2026-10-10 (header line included). A fixed
        # catalog, so reverting DEFAULT_MODEL to a retired slug fails here.
        listed = run(
            "Fetching available models...\\n"
            "gemini-3.8-flash-high\\tGemini 3.8 Flash (High)\\n"
            "gemini-3.8-flash-medium\\tGemini 3.8 Flash (Medium)\\n"
            "gemini-3.7-flash-high\\tGemini 3.7 Flash (High)\\n"
            "gemini-3.1-pro-high\\tGemini 3.1 Pro (High)")
        self.assertTrue(listed["ok"], listed)
        self.assertIs(listed["default_available"], True)
        self.assertNotIn("warning", listed)
        retired = run("gemini-y-high\\tSome Newer Flash (High)")
        self.assertTrue(retired["ok"], retired)
        self.assertIs(retired["default_available"], False)
        self.assertIn(dispatch_mod.DEFAULT_MODEL, retired["warning"])

    def test_list_models_in_process_includes_backends_on_cli_missing(self):
        with mock.patch("shutil.which", return_value=None):
            res = asyncio.run(server._list_models(5))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "cli_missing")
        self.assertEqual(set(res["backends"]), {"agy", "grok", "codex", "claude", "pi", "cursor", "openai", "ollama", "lmstudio"})

    def test_log_event_emits_json_on_stderr(self):
        import io
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            server._log_event("unit_test", foo=1, bar="x")
        finally:
            sys.stderr = old
        line = buf.getvalue().strip()
        self.assertTrue(line.startswith("[omni-llm-mcp] "))
        payload = json.loads(line[len("[omni-llm-mcp] "):])
        self.assertEqual(payload["event"], "unit_test")
        self.assertEqual(payload["foo"], 1)
        self.assertIn("ts", payload)

    def test_drain_pending_waits_for_inflight_task(self):
        async def _run():
            server._pending_tasks.clear()
            finished = []

            async def slow():
                await asyncio.sleep(0.05)
                finished.append(True)

            t = asyncio.create_task(slow())
            server._track_task(t)
            await server._drain_pending(timeout_s=2)
            self.assertEqual(finished, [True])
            self.assertEqual(len(server._pending_tasks), 0)

        asyncio.run(_run())

    def test_drain_pending_cancels_on_timeout(self):
        async def _run():
            server._pending_tasks.clear()
            cancelled = []

            async def hang():
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

            t = asyncio.create_task(hang())
            server._track_task(t)
            await server._drain_pending(timeout_s=0.05)
            self.assertEqual(cancelled, [True])

        asyncio.run(_run())


class TestRouteTable(unittest.TestCase):
    """resolve_route / ROUTE_RULES snapshot — single source for model→provider."""

    def test_route_snapshot(self):
        cases = [
            ("ollama/llama3", "ollama", "llama3"),
            ("lmstudio/qwen", "lmstudio", "qwen"),
            ("gpt-4o", "openai", "gpt-4o"),
            ("o1-preview", "openai", "o1-preview"),
            ("o3-mini", "openai", "o3-mini"),
            ("codex", "openai", "codex"),
            ("openai-codex-mini", "openai", "openai-codex-mini"),
            ("grok-4.5", "grok", "grok-4.5"),
            ("Gemini 3.5 Flash", "agy", "Gemini 3.5 Flash"),
            ("anything-else", "agy", "anything-else"),
        ]
        for name, provider, routed in cases:
            route = server.resolve_route(name)
            self.assertEqual(route["provider"], provider, name)
            self.assertEqual(route["model"], routed, name)

    def test_openai_rules_carry_default_api_base(self):
        self.assertEqual(
            server.resolve_route("gpt-4o")["default_api_base"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            server.resolve_route("ollama/x")["default_api_base"],
            "http://localhost:11434/v1",
        )
        self.assertNotIn("default_api_base", server.resolve_route("grok-1"))


class TestErrorClassification(unittest.TestCase):
    def test_cli_auth_and_rate_limit(self):
        self.assertEqual(
            server.classify_cli_error("", "please run login first", 1), "auth_error")
        self.assertEqual(
            server.classify_cli_error("", "quota exceeded 429", 1), "rate_limit")
        self.assertEqual(
            server.classify_cli_error("", "boom segfault", 1), "runtime_error")

    def test_codex_usage_limit_is_rate_limit(self):
        self.assertEqual(
            server.classify_cli_error(
                "", "You've hit your usage limit. Try again at Aug 20th.", 1),
            "rate_limit")

    def test_auth_wins_over_rate_limit_when_both_present(self):
        self.assertEqual(
            server.classify_cli_error("", "auth expired; quota also hit", 1),
            "auth_error",
        )

    def test_grok_extra_403_auth_marker(self):
        self.assertEqual(
            server.classify_cli_error(
                "", "HTTP 403", 1, extra_auth_markers=server.GROK_EXTRA_AUTH_MARKERS),
            "auth_error",
        )

    def test_exception_message_classification(self):
        self.assertEqual(server.classify_exception_message("429 too many requests"),
                         "rate_limit")
        self.assertEqual(server.classify_exception_message("credential expired"),
                         "auth_error")
        self.assertEqual(server.classify_exception_message("connection reset"),
                         "runtime_error")

    def test_http_status_reason_map(self):
        self.assertEqual(server.reason_from_http_status(401), "auth_error")
        self.assertEqual(server.reason_from_http_status(403), "auth_error")
        self.assertEqual(server.reason_from_http_status(429), "rate_limit")
        self.assertEqual(server.reason_from_http_status(500), "runtime_error")
        self.assertEqual(server.reason_from_http_status(None), "runtime_error")

    def test_openai_http_401_is_auth_error(self):
        async def _fake_to_thread(fn):
            return {"error": True, "code": 401, "body": "unauthorized"}

        with mock.patch("providers.openai_compat.asyncio.to_thread", _fake_to_thread):
            res = asyncio.run(server._run_openai_compatible(
                "hi", "gpt-4o", 10, "default"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_openai_http_429_is_rate_limit(self):
        async def _fake_to_thread(fn):
            return {"error": True, "code": 429, "body": "slow down"}

        with mock.patch("providers.openai_compat.asyncio.to_thread", _fake_to_thread):
            res = asyncio.run(server._run_openai_compatible(
                "hi", "gpt-4o", 10, "default"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "rate_limit")


class TestXaiImageBackend(unittest.TestCase):
    """llm_image -> xAI image REST API. HTTP is mocked; no network / no key needed."""

    def test_no_key_is_auth_error(self):
        from providers import xai_image
        env = {k: v for k, v in os.environ.items() if not k.startswith("XAI_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = asyncio.run(xai_image.run_xai_image("a red cube"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_http_401_is_auth_error(self):
        from providers import xai_image

        async def _fake_to_thread(fn):
            return {"error": True, "code": 401, "body": "unauthorized"}

        with mock.patch.dict(os.environ, {"XAI_API_KEY": "k"}), \
                mock.patch("providers.xai_image.asyncio.to_thread", _fake_to_thread):
            res = asyncio.run(xai_image.run_xai_image("a red cube"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_http_429_is_rate_limit(self):
        from providers import xai_image

        async def _fake_to_thread(fn):
            return {"error": True, "code": 429, "body": "slow down"}

        with mock.patch.dict(os.environ, {"XAI_API_KEY": "k"}), \
                mock.patch("providers.xai_image.asyncio.to_thread", _fake_to_thread):
            res = asyncio.run(xai_image.run_xai_image("a red cube"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "rate_limit")

    def test_success_b64_writes_file(self):
        import base64
        import tempfile
        from providers import xai_image

        payload = base64.b64encode(b"PNGDATA").decode()

        async def _fake_to_thread(fn):
            return {"data": [{"b64_json": payload, "revised_prompt": "a bright red cube"}]}

        out = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"XAI_API_KEY": "k"}), \
                mock.patch("providers.xai_image.asyncio.to_thread", _fake_to_thread):
            res = asyncio.run(xai_image.run_xai_image(
                "a red cube", response_format="b64_json", out_dir=out))
        self.assertTrue(res["ok"])
        self.assertEqual(res["n"], 1)
        path = res["images"][0]["path"]
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"PNGDATA")
        self.assertEqual(res["images"][0]["revised_prompt"], "a bright red cube")


class _FakeCompleted:
    """Stand-in for the subprocess.CompletedProcess the grok CLI would return."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# These process-result fixtures cover the unchanged non-macOS transport.
# test_grok_host_consumers separately verifies the macOS host boundary.
@mock.patch("providers.host_fence.platform.system", new=lambda: "Linux")
class TestGrokImageEditBackend(unittest.TestCase):
    """llm_image backend='cli' -> keyless grok CLI image_edit provider. subprocess.run
    is mocked — no real grok CLI, no network — hermetic like TestGrokBackend. Mirrors
    tools/batch_generate.py's invocation shape (grok -p <image_edit prompt>, cwd=out_dir,
    then pick the produced PNG off disk)."""

    def test_missing_source_image_is_clear_error(self):
        from providers import grok_image_cli
        res = asyncio.run(grok_image_cli.run_grok_image_edit("make it sunset", None))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")
        self.assertIn("source_image", res["detail"])

    def test_cli_missing_is_loud(self):
        from providers import grok_image_cli
        with mock.patch("shutil.which", return_value=None):
            res = asyncio.run(grok_image_cli.run_grok_image_edit("x", "/refs/a.png"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "cli_missing")

    def test_success_returns_image_path_and_mirrors_invocation(self):
        from providers import grok_image_cli
        out = tempfile.mkdtemp()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            # grok's image_edit writes the PNG to the path named in the prompt and
            # prints it; mirror that so the provider finds the file on disk.
            p = os.path.join(kwargs["cwd"], "image_edit.png")
            with open(p, "wb") as f:
                f.write(b"PNGDATA")
            return _FakeCompleted(returncode=0, stdout=p, stderr="")

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
                mock.patch("providers.grok_image_cli.subprocess.run", side_effect=fake_run):
            res = asyncio.run(grok_image_cli.run_grok_image_edit(
                "make it sunset", "/refs/base.png", out_dir=out))

        self.assertTrue(res["ok"])
        self.assertEqual(res["model"], "grok-image-edit")
        self.assertEqual(res["n"], 1)
        path = res["images"][0]["path"]
        self.assertEqual(path, os.path.join(out, "image_edit.png"))
        self.assertTrue(os.path.exists(path))
        # invocation shape: grok -p <prompt>, cwd == output dir, image_edit + refs embedded
        self.assertEqual(captured["cmd"][0], "/fake/bin/grok")
        self.assertEqual(captured["cmd"][1], "-p")
        cli_prompt = captured["cmd"][2]
        self.assertIn("Use the image_edit tool", cli_prompt)
        self.assertIn("/refs/base.png", cli_prompt)   # source reference path
        self.assertIn(path, cli_prompt)               # save-to path
        self.assertEqual(captured["kwargs"]["cwd"], out)   # cwd is the output dir
        self.assertTrue(captured["kwargs"]["capture_output"])
        # bytes, decoded in one place (cli_run.decode_output), never text=True
        self.assertFalse(captured["kwargs"].get("text"))

    def test_nonzero_exit_auth_is_classified(self):
        from providers import grok_image_cli

        def fake_run(cmd, **kwargs):
            return _FakeCompleted(returncode=1, stdout="", stderr="error: please run `grok login`")

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
                mock.patch("providers.grok_image_cli.subprocess.run", side_effect=fake_run):
            res = asyncio.run(grok_image_cli.run_grok_image_edit("x", "/refs/a.png"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "auth_error")

    def test_nonzero_exit_generic_is_runtime_error(self):
        from providers import grok_image_cli

        def fake_run(cmd, **kwargs):
            return _FakeCompleted(returncode=2, stdout="", stderr="boom: renderer crashed")

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
                mock.patch("providers.grok_image_cli.subprocess.run", side_effect=fake_run):
            res = asyncio.run(grok_image_cli.run_grok_image_edit("x", "/refs/a.png"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")
        self.assertIn("renderer crashed", res["detail"])

    def test_timeout_is_typed(self):
        from providers import grok_image_cli

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
                mock.patch("providers.grok_image_cli.subprocess.run", side_effect=fake_run):
            res = asyncio.run(grok_image_cli.run_grok_image_edit(
                "x", "/refs/a.png", out_dir=tempfile.mkdtemp(), timeout_s=5))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "timeout")

    def test_zero_exit_but_no_file_is_empty_response(self):
        from providers import grok_image_cli

        def fake_run(cmd, **kwargs):
            # grok exits 0 but produced no image (e.g. refused / wrote nowhere)
            return _FakeCompleted(returncode=0, stdout="I could not edit that", stderr="")

        with mock.patch("shutil.which", return_value="/fake/bin/grok"), \
                mock.patch("providers.grok_image_cli.subprocess.run", side_effect=fake_run):
            res = asyncio.run(grok_image_cli.run_grok_image_edit(
                "x", "/refs/a.png", out_dir=tempfile.mkdtemp()))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "empty_response")

    # --- backend routing through call_tool (default = cli) ---

    def test_default_backend_routes_to_cli(self):
        called = {}

        async def fake_cli(prompt, source_image, timeout_s=180, out_dir=None, account_profile="default",
                           apply_aliases=False):
            called["cli"] = (prompt, source_image)
            return {"ok": True, "model": "grok-image-edit", "images": [{"path": "/x.png"}], "n": 1}

        async def fake_rest(prompt, **kw):
            called["rest"] = True
            return {"ok": True}

        with mock.patch("providers.grok_image_cli.run_grok_image_edit", fake_cli), \
                mock.patch("providers.xai_image.run_xai_image", fake_rest):
            res = asyncio.run(server._call_tool(
                "llm_image", {"prompt": "edit it", "source_image": "/r.png"}))
        self.assertIn("cli", called)          # no backend -> CLI (subscription default)
        self.assertNotIn("rest", called)
        self.assertEqual(called["cli"], ("edit it", "/r.png"))
        self.assertTrue(res["ok"])

    def test_backend_cli_routes_to_cli(self):
        called = {}

        async def fake_cli(prompt, source_image, timeout_s=180, out_dir=None, account_profile="default",
                           apply_aliases=False):
            called["cli"] = True
            return {"ok": True, "images": [{"path": "/x.png"}], "n": 1}

        with mock.patch("providers.grok_image_cli.run_grok_image_edit", fake_cli):
            res = asyncio.run(server._call_tool(
                "llm_image", {"prompt": "edit it", "backend": "cli", "source_image": "/r.png"}))
        self.assertIn("cli", called)
        self.assertTrue(res["ok"])

    def test_backend_rest_routes_to_rest(self):
        called = {}

        async def fake_cli(prompt, source_image, **kw):
            called["cli"] = True
            return {"ok": True}

        async def fake_rest(prompt, **kw):
            called["rest"] = prompt
            return {"ok": True, "model": "grok-2-image", "images": [], "n": 0}

        with mock.patch("providers.grok_image_cli.run_grok_image_edit", fake_cli), \
                mock.patch("providers.xai_image.run_xai_image", fake_rest):
            res = asyncio.run(server._call_tool(
                "llm_image", {"prompt": "a red cube", "backend": "rest"}))
        self.assertIn("rest", called)         # backend=rest -> xAI REST path
        self.assertNotIn("cli", called)
        self.assertEqual(called["rest"], "a red cube")
        self.assertTrue(res["ok"])

    def test_unknown_backend_is_loud(self):
        res = asyncio.run(server._call_tool(
            "llm_image", {"prompt": "x", "backend": "bogus"}))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")
        self.assertIn("backend", res["detail"])

    def test_cli_backend_missing_source_image_surfaces_through_dispatch(self):
        # end-to-end through call_tool with the REAL provider (source_image check is
        # first, so this is hermetic whether or not a grok CLI is installed).
        res = asyncio.run(server._call_tool(
            "llm_image", {"prompt": "edit it", "backend": "cli"}))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")
        self.assertIn("source_image", res["detail"])

    def test_tools_list_llm_image_exposes_backend_and_source_image(self):
        out = run_server([INIT, {"jsonrpc": "2.0", "id": 30, "method": "tools/list"}], no_agy=True)
        tools = _by_id(out, 30)["result"]["tools"]
        img = [t for t in tools if t["name"] == "llm_image"][0]
        props = img["inputSchema"]["properties"]
        self.assertIn("backend", props)
        self.assertEqual(props["backend"]["enum"], ["cli", "rest"])
        self.assertIn("source_image", props)


class TestFilesSupport(unittest.TestCase):
    """files= runs on agy/codex/grok; other providers must LOUD-reject (not silently drop)."""

    def test_files_on_grok_without_roots_fails_closed(self):
        # grok now supports image files, gated by OMNI_LLM_SOURCE_ROOTS —
        # with no allowlist configured the fence must fail closed, not spawn.
        env = {k: v for k, v in os.environ.items() if k != "OMNI_LLM_SOURCE_ROOTS"}
        with mock.patch.dict(os.environ, env, clear=True):
            res = asyncio.run(server._route_to_provider(
                "hi", "grok-4.5", 30, ["/tmp/a.png"], "default"))
        self.assertFalse(res["ok"])
        self.assertEqual(res.get("error_code"), "source_path_disabled")

    def test_files_on_openai_is_unsupported_files(self):
        res = asyncio.run(server._route_to_provider(
            "hi", "gpt-4o", 30, ["/tmp/a.png"], "default"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "unsupported_files")

    def test_files_on_ollama_is_unsupported_files(self):
        res = asyncio.run(server._route_to_provider(
            "hi", "ollama/llama3", 30, ["https://youtu.be/x"], "default"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "unsupported_files")

    def test_files_on_agy_is_forwarded_not_rejected(self):
        # agy path accepts files (appended into prompt). With agy absent we still
        # must NOT get unsupported_files — the failure is cli_missing further down.
        called = {}

        async def fake_agy(prompt, model, timeout_s, files=None, account_profile="default"):
            called["files"] = files
            called["prompt"] = prompt
            return {"ok": True, "text": "ok", "model": model}

        with mock.patch("providers.agy.run_agy", fake_agy):
            res = asyncio.run(server._route_to_provider(
                "describe", "Gemini 3.5 Flash", 30, ["/tmp/x.png"], "default"))
        self.assertTrue(res["ok"])
        self.assertEqual(called["files"], ["/tmp/x.png"])

    def test_fallback_skips_unsupported_files_to_next_model(self):
        # chain: gpt-4o (rejects files) → agy model (accepts). Must land on agy.
        called = []

        async def fake_route(prompt, model, timeout_s, files, account_profile):
            called.append(model)
            if model.startswith("gpt-"):
                return {"ok": False, "reason": "unsupported_files", "detail": "nope"}
            return {"ok": True, "text": "from-agy", "model": model}

        with mock.patch("providers.dispatch.route_to_provider", fake_route):
            res = asyncio.run(server._dispatch_llm_request(
                "hi", "gpt-4o,Gemini 3.5 Flash", 30, ["/tmp/a.png"], "default"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["text"], "from-agy")
        self.assertEqual(called, ["gpt-4o", "Gemini 3.5 Flash"])
        self.assertEqual(res["tried"], [{"model": "gpt-4o", "reason": "unsupported_files"}])


class TestAgySdkFidelity(unittest.TestCase):
    """SDK path must honor model/timeout/profile or fall back to subprocess — never
    silently answer with a different model (P0 from architecture review)."""

    def test_sdk_unavailable_model_falls_back_to_subprocess(self):
        subprocess_calls = []
        agy_profile_dir = tempfile.mkdtemp(prefix="omni-agy-work-profile-")
        self.addCleanup(shutil.rmtree, agy_profile_dir, True)

        async def fake_sub(prompt, model, timeout_s, account_profile):
            subprocess_calls.append({
                "model": model, "timeout_s": timeout_s, "account_profile": account_profile,
                "prompt": prompt,
            })
            return {"ok": True, "text": "via-sub", "model": model}

        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": agy_profile_dir}), \
             mock.patch("providers.agy.HAS_SDK", True), \
             mock.patch("providers.agy.build_agy_sdk_config", return_value=None), \
             mock.patch("providers.agy.run_agy_subprocess", fake_sub):
            _seed_agy("work", "Gemini 3.5 Flash (High)")
            res = asyncio.run(server._run_agy(
                "hello", "Gemini 3.5 Flash (High)", 42, None, "work"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["text"], "via-sub")
        self.assertEqual(len(subprocess_calls), 1)
        self.assertEqual(subprocess_calls[0]["model"], "Gemini 3.5 Flash (High)")
        self.assertEqual(subprocess_calls[0]["timeout_s"], 42)
        self.assertEqual(subprocess_calls[0]["account_profile"], "work")

    def test_sdk_success_honors_timeout_and_skips_subprocess(self):
        subprocess_calls = []

        async def fake_sdk(prompt, model, timeout_s, files=None, account_profile="default"):
            return {"ok": True, "text": "via-sdk", "model": model}

        async def fake_sub(prompt, model, timeout_s, account_profile):
            subprocess_calls.append(model)
            return {"ok": True, "text": "via-sub", "model": model}

        # Off macOS: on macOS run_agy never calls the SDK (it is outside the host fence).
        with mock.patch("providers.agy.HAS_SDK", True), \
             mock.patch("providers.agy.platform.system", return_value="Linux"), \
             mock.patch("providers.agy.run_agy_sdk", fake_sdk), \
             mock.patch("providers.agy.run_agy_subprocess", fake_sub):
            self.addCleanup(_seed_server_catalogs)
            _seed_agy("default", "My Model")
            res = asyncio.run(server._run_agy("hi", "My Model", 30, None, "default"))
        self.assertEqual(res["text"], "via-sdk")
        self.assertEqual(subprocess_calls, [])

    def test_build_agy_sdk_config_returns_none_without_sdk_types(self):
        # When LocalAgentConfig rejects all model kwargs, every TypeError path yields None.
        with mock.patch("providers.agy.LocalAgentConfig", side_effect=TypeError("nope")):
            self.assertIsNone(server._build_agy_sdk_config("any-model"))


class TestAgyOrphanGuard(unittest.TestCase):
    """A mid-communicate error (or client-disconnect CancelledError) must not leave
    the agy subprocess running. Without the finally, the `except Exception` branch
    returns WITHOUT killing proc (and CancelledError isn't caught at all) -> orphan
    leak. Mirrors the grok.py / codex.py guard from #7/#8."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_exception_during_communicate_reaps_agy_subprocess(self):
        proc = _FakeAgyProc(RuntimeError("boom"))

        async def fake_exec(*argv, **kwargs):
            return proc

        with mock.patch("shutil.which", return_value="/fake/bin/agy"), \
             mock.patch("asyncio.create_subprocess_exec", fake_exec):
            res = asyncio.run(
                server._run_agy_subprocess("hi", "Gemini 3.5 Flash", 30, "default"))

        self.assertFalse(res["ok"])       # error surfaced as a loud envelope
        self.assertTrue(proc.killed)      # orphan reaped by the finally

    def test_cancelled_error_still_reaps_and_propagates(self):
        proc = _FakeAgyProc(asyncio.CancelledError())

        async def fake_exec(*argv, **kwargs):
            return proc

        async def drive():
            with mock.patch("shutil.which", return_value="/fake/bin/agy"), \
                 mock.patch("asyncio.create_subprocess_exec", fake_exec):
                await server._run_agy_subprocess("hi", "Gemini 3.5 Flash", 30, "default")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(drive())
        self.assertTrue(proc.killed)      # reaped even though CancelledError propagates


class TestAgySdkProfileIsolation(unittest.TestCase):
    """Concurrent SDK calls with different account_profiles must not clobber the
    process-global AGY_CONFIG_DIR — that would route a call to the wrong account
    (a privacy boundary, not a cosmetic glitch). Regression guard for the _sdk_lock."""

    def test_concurrent_profiles_do_not_leak_config_dir(self):
        import providers.agy as agy_mod

        class _FakeResponse:
            def __init__(self):
                self._done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                # yield control so a sibling coroutine could (wrongly) mutate env mid-chat
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return os.environ.get("AGY_CONFIG_DIR", "<unset>")

        class _FakeAgent:
            def __init__(self, config):
                pass

            async def __aenter__(self):
                await asyncio.sleep(0)
                return self

            async def __aexit__(self, *exc):
                return False

            async def chat(self, prompt):
                await asyncio.sleep(0)
                return _FakeResponse()

        async def _run():
            agy_mod._sdk_lock = None  # fresh lock bound to this loop
            saved = os.environ.get("AGY_CONFIG_DIR")
            os.environ["AGY_CONFIG_DIR_WORK"] = "/cfg/work"
            os.environ["AGY_CONFIG_DIR_HOME"] = "/cfg/home"
            os.environ.pop("AGY_CONFIG_DIR", None)
            try:
                with mock.patch("providers.agy.Agent", _FakeAgent), \
                     mock.patch("providers.agy.build_agy_sdk_config", return_value=object()):
                    work, home = await asyncio.gather(
                        agy_mod.run_agy_sdk("hi", "m", 30, None, "work"),
                        agy_mod.run_agy_sdk("hi", "m", 30, None, "home"),
                    )
                # each call must have observed ITS OWN profile dir during chat
                self.assertEqual(work["text"], "/cfg/work")
                self.assertEqual(home["text"], "/cfg/home")
                # env fully restored to its pre-call state (was unset)
                self.assertNotIn("AGY_CONFIG_DIR", os.environ)
            finally:
                for k in ("AGY_CONFIG_DIR_WORK", "AGY_CONFIG_DIR_HOME"):
                    os.environ.pop(k, None)
                if saved is not None:
                    os.environ["AGY_CONFIG_DIR"] = saved
                else:
                    os.environ.pop("AGY_CONFIG_DIR", None)

        asyncio.run(_run())


class TestStdoutWriteLock(unittest.TestCase):
    """Concurrent _send must not interleave JSON-RPC lines on stdout."""

    def test_concurrent_sends_produce_one_json_object_per_line(self):
        async def hammer():
            # Reset lock so this test owns a fresh one bound to this loop.
            server._write_lock = None
            n = 40
            payload = {"jsonrpc": "2.0", "id": 0, "result": {"blob": "x" * 200}}

            async def one(i):
                obj = dict(payload, id=i)
                await server._send(obj)

            # Capture stdout
            import io
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                await asyncio.gather(*[one(i) for i in range(n)])
            finally:
                sys.stdout = old

            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            self.assertEqual(len(lines), n)
            ids = sorted(json.loads(ln)["id"] for ln in lines)
            self.assertEqual(ids, list(range(n)))
            # every line must be independently parseable (no partial merges)
            for ln in lines:
                json.loads(ln)

        asyncio.run(hammer())


class TestBigContextSourcePath(unittest.TestCase):
    """Hermetic tests for llm_big_context source_path (allowlist + XOR text)."""

    def tearDown(self):
        os.environ.pop("OMNI_LLM_SOURCE_ROOTS", None)

    def _call_big(self, arguments, fake_dispatch=None):
        import providers.dispatch as d

        async def run():
            return await d.call_tool("llm_big_context", arguments)

        if fake_dispatch is not None:
            with mock.patch.object(d, "dispatch_llm_request", fake_dispatch):
                return asyncio.run(run())
        return asyncio.run(run())

    def test_legacy_empty_text_still_fails(self):
        res = self._call_big({"instruction": "sum", "text": ""})
        self.assertFalse(res["ok"])
        self.assertEqual(res["detail"], "empty text")
        self.assertNotIn("error_code", res)

    def test_path_without_roots_disabled(self):
        os.environ.pop("OMNI_LLM_SOURCE_ROOTS", None)
        res = self._call_big({
            "instruction": "sum",
            "source_path": "/tmp/nope.txt",
        })
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_path_disabled")

    def test_path_only_reads_file(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "doc.txt")
        with open(path, "w") as f:
            f.write("hello-from-file")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        seen = []

        async def fake_dispatch(prompt, model, timeout_s, files, profile):
            seen.append(prompt)
            return {"ok": True, "text": "ok", "model": model}

        res = self._call_big(
            {"instruction": "summarize", "source_path": path},
            fake_dispatch=fake_dispatch,
        )
        self.assertTrue(res["ok"])
        self.assertEqual(len(seen), 1)
        self.assertIn("hello-from-file", seen[0])
        self.assertIn("summarize", seen[0])
        self.assertIn("--- SOURCE (public) ---", seen[0])

    def test_path_and_text_mutually_exclusive(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "doc.txt")
        with open(path, "w") as f:
            f.write("x")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        res = self._call_big({
            "instruction": "sum",
            "text": "inline",
            "source_path": path,
        })
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "invalid_arguments")

    def test_relative_path_rejected(self):
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = "/tmp"
        res = self._call_big({
            "instruction": "sum",
            "source_path": "relative/doc.txt",
        })
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_path_not_absolute")

    def test_outside_roots_not_allowed(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        other = tempfile.mkdtemp(prefix="omni_out_")
        path = os.path.join(other, "secret.txt")
        with open(path, "w") as f:
            f.write("secret")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        res = self._call_big({"instruction": "sum", "source_path": path})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_path_not_allowed")
        # detail must not echo full path
        self.assertNotIn(path, res.get("detail", ""))

    def test_missing_file_under_root(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        path = os.path.join(root, "missing.txt")
        res = self._call_big({"instruction": "sum", "source_path": path})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_not_found")

    def test_directory_not_regular(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        res = self._call_big({"instruction": "sum", "source_path": root})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_not_regular")

    def test_too_large(self):
        import providers.source_load as sl
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "big.txt")
        with open(path, "wb") as f:
            f.write(b"a" * 100)
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        with mock.patch.object(sl, "SOURCE_PATH_MAX_BYTES", 50), \
             mock.patch("providers.source_load.SOURCE_PATH_MAX_BYTES", 50):
            # read_source_file default arg binds at def time — pass via patch on module used by resolve
            body, err = sl.read_source_file(path, max_bytes=50, roots=[root])
        self.assertIsNone(body)
        self.assertEqual(err["error_code"], "source_too_large")

    def test_invalid_utf8(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "bin.txt")
        with open(path, "wb") as f:
            f.write(b"\xff\xfe not utf8")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        res = self._call_big({"instruction": "sum", "source_path": path})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_invalid_utf8")

    def test_symlink_file_to_inside_resolves_via_openat(self):
        """Symlink leaf under root pointing inside: realpath + openat opens the target."""
        root = tempfile.mkdtemp(prefix="omni_src_")
        real = os.path.join(root, "real.txt")
        link = os.path.join(root, "link.txt")
        with open(real, "w") as f:
            f.write("via-link")
        os.symlink(real, link)
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        seen = []

        async def fake_dispatch(prompt, model, timeout_s, files, profile):
            seen.append(prompt)
            return {"ok": True, "text": "ok", "model": model}

        res = self._call_big(
            {"instruction": "sum", "source_path": link},
            fake_dispatch=fake_dispatch,
        )
        self.assertTrue(res["ok"], res)
        self.assertIn("via-link", seen[0])

    def test_symlink_file_to_outside_rejected(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        outside = tempfile.mkdtemp(prefix="omni_out_")
        secret = os.path.join(outside, "secret.txt")
        with open(secret, "w") as f:
            f.write("escaped")
        link = os.path.join(root, "link.txt")
        os.symlink(secret, link)
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        res = self._call_big({"instruction": "sum", "source_path": link})
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "source_path_not_allowed")

    def test_schema_has_source_path_instruction_only_required(self):
        tool = next(t for t in server.TOOLS if t["name"] == "llm_big_context")
        props = tool["inputSchema"]["properties"]
        self.assertIn("source_path", props)
        self.assertEqual(tool["inputSchema"]["required"], ["instruction"])

    def test_provider_not_called_on_path_error(self):
        os.environ.pop("OMNI_LLM_SOURCE_ROOTS", None)
        called = []

        async def fake_dispatch(*a, **k):
            called.append(1)
            return {"ok": True, "text": "x", "model": "m"}

        res = self._call_big(
            {"instruction": "sum", "source_path": "/abs/x"},
            fake_dispatch=fake_dispatch,
        )
        self.assertFalse(res["ok"])
        self.assertEqual(called, [])

    def test_intermediate_symlink_dir_rejected(self):
        """openat O_NOFOLLOW must refuse a symlink directory under the root."""
        root = tempfile.mkdtemp(prefix="omni_src_")
        outside = tempfile.mkdtemp(prefix="omni_out_")
        with open(os.path.join(outside, "secret.txt"), "w") as f:
            f.write("escaped")
        # root/sub -> outside; path root/sub/secret.txt would escape if followed
        os.symlink(outside, os.path.join(root, "sub"))
        path = os.path.join(root, "sub", "secret.txt")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        called = []

        async def fake_dispatch(*a, **k):
            called.append(1)
            return {"ok": True, "text": "x", "model": "m"}

        res = self._call_big(
            {"instruction": "sum", "source_path": path},
            fake_dispatch=fake_dispatch,
        )
        self.assertFalse(res["ok"])
        self.assertIn(res["error_code"], (
            "source_path_not_allowed", "source_not_found", "source_io_error"))
        self.assertEqual(called, [])

    def test_nested_file_under_root_ok(self):
        root = tempfile.mkdtemp(prefix="omni_src_")
        sub = os.path.join(root, "a", "b")
        os.makedirs(sub)
        path = os.path.join(sub, "doc.txt")
        with open(path, "w") as f:
            f.write("nested-ok")
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        seen = []

        async def fake_dispatch(prompt, model, timeout_s, files, profile):
            seen.append(prompt)
            return {"ok": True, "text": "ok", "model": model}

        res = self._call_big(
            {"instruction": "sum", "source_path": path},
            fake_dispatch=fake_dispatch,
        )
        self.assertTrue(res["ok"], res)
        self.assertIn("nested-ok", seen[0])

    def test_prompt_size_boundary_with_instruction(self):
        """Assembled prompt cap: body alone may fit while instruction pushes over."""
        import providers.source_load as sl
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "body.txt")
        with open(path, "w") as f:
            f.write("x" * 100)
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = root
        tiny = 80
        with mock.patch.object(sl, "SOURCE_PROMPT_MAX_BYTES", tiny):
            out, err = sl.resolve_big_context_source(
                "", path, instruction="I" * 50)
        self.assertIsNone(out)
        self.assertEqual(err["error_code"], "source_too_large")

    def test_file_size_exact_max_ok(self):
        import providers.source_load as sl
        root = tempfile.mkdtemp(prefix="omni_src_")
        path = os.path.join(root, "exact.txt")
        data = b"z" * 64
        with open(path, "wb") as f:
            f.write(data)
        body, err = sl.read_source_file(path, max_bytes=64, roots=[root])
        self.assertIsNone(err)
        self.assertEqual(body, "z" * 64)
        body2, err2 = sl.read_source_file(path, max_bytes=63, roots=[root])
        self.assertIsNone(body2)
        self.assertEqual(err2["error_code"], "source_too_large")


class TestCliWatchResolve(unittest.TestCase):
    """Hermetic routing for optional zellij watch panes (no real zellij)."""

    def tearDown(self):
        for k in ("OMNI_LLM_WATCH_PANE", "OMNI_LLM_ZELLIJ_SESSION",
                  "OMNI_LLM_ZELLIJ_TAB", "OMNI_LLM_ZELLIJ_TAB_ID",
                  "ZELLIJ_SESSION_NAME"):
            os.environ.pop(k, None)

    def test_watch_false_overrides_env_true(self):
        import providers.cli_run as cr
        os.environ["OMNI_LLM_WATCH_PANE"] = "1"
        token = cr.set_watch_context(watch_pane=False)
        try:
            self.assertFalse(cr.watch_enabled())
        finally:
            cr.reset_watch_context(token)

    def test_watch_true_from_tool(self):
        import providers.cli_run as cr
        os.environ.pop("OMNI_LLM_WATCH_PANE", None)
        token = cr.set_watch_context(watch_pane=True)
        try:
            self.assertTrue(cr.watch_enabled())
        finally:
            cr.reset_watch_context(token)

    def test_env_zero_is_not_truthy(self):
        import providers.cli_run as cr
        os.environ["OMNI_LLM_WATCH_PANE"] = "0"
        self.assertFalse(cr.watch_enabled())

    def test_no_tab_skips_even_when_watch_and_session_ok(self):
        import providers.cli_run as cr
        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork", zellij_tab=cr._UNSET)
        try:
            with mock.patch.object(cr, "list_zellij_sessions", return_value=["mywork"]), \
                 mock.patch.object(cr, "list_zellij_tabs", return_value=[
                     {"name": "work", "tab_id": 0}]):
                self.assertEqual(cr.resolve_zellij_session(), "mywork")
                self.assertIsNone(cr.resolve_zellij_tab_id("mywork"))
        finally:
            cr.reset_watch_context(token)

    def test_invalid_tool_tab_fail_closed_no_env_downgrade(self):
        import providers.cli_run as cr
        os.environ["OMNI_LLM_ZELLIJ_TAB"] = "work"
        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork", zellij_tab="no-such-tab")
        try:
            with mock.patch.object(cr, "list_zellij_sessions", return_value=["mywork"]), \
                 mock.patch.object(cr, "list_zellij_tabs", return_value=[
                     {"name": "work", "tab_id": 0}]):
                # Explicit invalid tool tab must NOT fall back to env work.
                self.assertIsNone(cr.resolve_zellij_tab_id("mywork"))
        finally:
            cr.reset_watch_context(token)

    def test_tab_name_resolves_to_id(self):
        import providers.cli_run as cr
        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork", zellij_tab="notes")
        try:
            with mock.patch.object(cr, "list_zellij_sessions", return_value=["mywork"]), \
                 mock.patch.object(cr, "list_zellij_tabs", return_value=[
                     {"name": "work", "tab_id": 0},
                     {"name": "notes", "tab_id": 3}]):
                self.assertEqual(cr.resolve_zellij_tab_id("mywork"), 3)
        finally:
            cr.reset_watch_context(token)

    def test_duplicate_tab_name_fail_closed(self):
        import providers.cli_run as cr
        token = cr.set_watch_context(zellij_tab="dup")
        try:
            with mock.patch.object(cr, "list_zellij_tabs", return_value=[
                    {"name": "dup", "tab_id": 1},
                    {"name": "dup", "tab_id": 2}]):
                self.assertIsNone(cr.resolve_zellij_tab_id("mywork"))
        finally:
            cr.reset_watch_context(token)

    def test_invalid_tool_session_fail_closed(self):
        import providers.cli_run as cr
        os.environ["OMNI_LLM_ZELLIJ_SESSION"] = "mywork"
        token = cr.set_watch_context(zellij_session="stale-name")
        try:
            with mock.patch.object(cr, "list_zellij_sessions",
                                   return_value=["mywork"]):
                self.assertIsNone(cr.resolve_zellij_session())
        finally:
            cr.reset_watch_context(token)

    def test_run_cli_without_watch_is_plain_pipe(self):
        import providers.cli_run as cr
        captured = {}

        async def fake_spawn(*argv, **kwargs):
            captured["argv"] = list(argv)
            return _FakeGrokProc(stdout=b"hello", stderr=b"", returncode=0)

        async def drive():
            with mock.patch.object(cr, "_spawn", fake_spawn), \
                 mock.patch.object(cr, "watch_enabled", return_value=False):
                return await cr.run_cli(["/bin/echo", "x"], timeout_s=5)

        res = asyncio.run(drive())
        self.assertEqual(res["stdout"].strip(), "hello")
        self.assertFalse(res["watched"])
        self.assertFalse(res["timed_out"])
        self.assertEqual(captured["argv"][0], "/bin/echo")

    def test_run_cli_watch_without_tab_does_not_open_pane(self):
        import providers.cli_run as cr
        opens = []

        async def fake_open(*a, **k):
            opens.append(1)
            return "terminal_9"

        async def fake_spawn(*argv, **kwargs):
            return _FakeGrokProc(stdout=b"ok", stderr=b"", returncode=0)

        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork")  # no tab
        try:
            async def drive():
                with mock.patch.object(cr, "list_zellij_sessions",
                                       return_value=["mywork"]), \
                     mock.patch.object(cr, "list_zellij_tabs",
                                       return_value=[{"name": "work", "tab_id": 0}]), \
                     mock.patch.object(cr, "_open_watch_pane", fake_open), \
                     mock.patch.object(cr, "_spawn", fake_spawn):
                    return await cr.run_cli(["/bin/true"], timeout_s=5)

            res = asyncio.run(drive())
        finally:
            cr.reset_watch_context(token)
        self.assertEqual(opens, [])
        self.assertFalse(res["watched"])
        self.assertEqual(res["returncode"], 0)

    def test_schema_exposes_watch_fields(self):
        for tool_name in ("llm_generate", "llm_big_context"):
            tool = next(t for t in server.TOOLS if t["name"] == tool_name)
            props = tool["inputSchema"]["properties"]
            self.assertIn("watch_pane", props)
            self.assertIn("zellij_session", props)
            self.assertIn("zellij_tab", props)

    def test_return_not_blocked_by_slow_close_pane(self):
        """MCP must return without waiting on backup close-pane (Codex BLOCK #1)."""
        import providers.cli_run as cr
        close_started = []

        async def slow_close(session, pane_id):
            close_started.append(1)
            await asyncio.sleep(2.0)

        async def fake_open(*a, **k):
            return "terminal_42"

        async def fake_spawn(*argv, **kwargs):
            return _FakeGrokProc(stdout=b"ok", stderr=b"", returncode=0)

        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork", zellij_tab="work")
        try:
            async def drive():
                with mock.patch.object(cr, "list_zellij_sessions",
                                       return_value=["mywork"]), \
                     mock.patch.object(cr, "list_zellij_tabs", return_value=[
                         {"name": "work", "tab_id": 0}]), \
                     mock.patch.object(cr, "_open_watch_pane", fake_open), \
                     mock.patch.object(cr, "_close_pane", slow_close), \
                     mock.patch.object(cr, "_spawn", fake_spawn):
                    t0 = time.monotonic()
                    res = await cr.run_cli(["/bin/true"], timeout_s=5,
                                           pane_name="omni:t")
                    elapsed = time.monotonic() - t0
                    return res, elapsed

            res, elapsed = asyncio.run(drive())
        finally:
            cr.reset_watch_context(token)
        self.assertTrue(res["watched"])
        self.assertLess(elapsed, 0.5, "return blocked by close-pane: %.2fs" % elapsed)

    def test_kill_proc_tree_group_kills_while_unreaped(self):
        """Leader still unreaped => pgid cannot be recycled => killpg is safe."""
        import providers.cli_run as cr
        calls = []

        class LiveProc:
            def __init__(self):
                self.pid = 0x7FFFFFF3
                self.returncode = None  # not reaped yet

            def kill(self):
                calls.append("kill")
                self.returncode = -9

            async def wait(self):
                self.returncode = -9
                return self.returncode

        def fake_killpg(pid, sig):
            calls.append(("killpg", pid, sig))

        async def drive():
            with mock.patch.object(cr.os, "killpg", fake_killpg):
                await cr._kill_proc_tree(LiveProc())

        asyncio.run(drive())
        self.assertEqual(calls[0], ("killpg", 0x7FFFFFF3, signal.SIGKILL))

    def test_kill_proc_tree_never_signals_after_reap(self):
        """Regression: a reaped pid is free for reuse — never killpg it.

        Blindly signalling the old pgid on the clean path could land on an
        unrelated process group that inherited the recycled pid.
        """
        import providers.cli_run as cr
        calls = []

        class DoneProc:
            def __init__(self):
                self.pid = 0x7FFFFFF3
                self.returncode = 0  # already reaped by _drive()

            def kill(self):
                calls.append("kill")

            async def wait(self):
                calls.append("wait")
                return self.returncode

        def fake_killpg(pid, sig):
            calls.append(("killpg", pid, sig))

        async def drive():
            with mock.patch.object(cr.os, "killpg", fake_killpg):
                await cr._kill_proc_tree(DoneProc())

        asyncio.run(drive())
        self.assertEqual(calls, [])

    def test_killpg_failure_falls_back_to_direct_kill(self):
        """No process group (start_new_session unsupported) => kill the child."""
        import providers.cli_run as cr
        calls = []

        class LiveProc:
            def __init__(self):
                self.pid = 0x7FFFFFF4
                self.returncode = None

            def kill(self):
                calls.append("kill")
                self.returncode = -9

            async def wait(self):
                self.returncode = -9
                return self.returncode

        def fake_killpg(pid, sig):
            calls.append("killpg")
            raise PermissionError()

        async def drive():
            with mock.patch.object(cr.os, "killpg", fake_killpg):
                await cr._kill_proc_tree(LiveProc())

        asyncio.run(drive())
        self.assertEqual(calls, ["killpg", "kill"])

    def test_contextvar_reset_isolates_calls(self):
        import providers.cli_run as cr
        # After reset, env-only path must not see prior tool tab.
        os.environ.pop("OMNI_LLM_ZELLIJ_TAB", None)
        token = cr.set_watch_context(zellij_tab="work")
        cr.reset_watch_context(token)
        with mock.patch.object(cr, "list_zellij_tabs", return_value=[
                {"name": "work", "tab_id": 0}]):
            self.assertIsNone(cr.resolve_zellij_tab_id("mywork"))

    def test_log_mode_0600_when_watch_opens(self):
        import providers.cli_run as cr
        modes = []

        async def fake_open(session, tab_id, log_path, done_path, pane_name, max_s):
            modes.append(os.stat(log_path).st_mode & 0o777)
            return "terminal_7"

        async def fake_spawn(*argv, **kwargs):
            return _FakeGrokProc(stdout=b"x", stderr=b"", returncode=0)

        token = cr.set_watch_context(
            watch_pane=True, zellij_session="mywork", zellij_tab="work")
        try:
            async def drive():
                with mock.patch.object(cr, "list_zellij_sessions",
                                       return_value=["mywork"]), \
                     mock.patch.object(cr, "list_zellij_tabs", return_value=[
                         {"name": "work", "tab_id": 0}]), \
                     mock.patch.object(cr, "_open_watch_pane", fake_open), \
                     mock.patch.object(cr, "_spawn", fake_spawn), \
                     mock.patch.object(cr, "_schedule_watch_finish", lambda *a, **k: None):
                    return await cr.run_cli(["/bin/true"], timeout_s=5)

            asyncio.run(drive())
        finally:
            cr.reset_watch_context(token)
        self.assertEqual(modes, [0o600])

    def test_drain_watch_finishers_awaits_scheduled_teardown(self):
        import providers.cli_run as cr
        closed = []

        async def fake_close(session, pane_id):
            await asyncio.sleep(0)
            closed.append((session, pane_id))

        async def drive():
            with mock.patch.object(cr, "_close_pane", fake_close):
                cr._schedule_watch_finish("mywork", "terminal_42", None)
                await cr.drain_watch_finishers(timeout_s=1)
                await asyncio.sleep(0)

        asyncio.run(drive())
        self.assertEqual(closed, [("mywork", "terminal_42")])
        self.assertEqual(cr._watch_finish_tasks, set())


class TestCliTimeoutBudget(unittest.TestCase):
    """timeout_s is the whole budget — no hidden grace on top of it."""

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_run_cli_kills_at_timeout_s_not_later(self):
        import providers.cli_run as cr

        async def drive():
            t0 = time.monotonic()
            res = await cr.run_cli(["/bin/sh", "-c", "sleep 60"], timeout_s=1)
            return res, time.monotonic() - t0

        res, elapsed = asyncio.run(drive())
        self.assertTrue(res["timed_out"])
        self.assertEqual(res["returncode"], -1)
        # Old behaviour waited timeout_s + 30; anything near that fails here.
        self.assertLess(elapsed, 5)

    def test_agy_inner_deadline_sits_inside_the_budget(self):
        """agy self-reports before run_cli SIGKILLs it."""
        import providers.agy as agy_mod
        from providers.config import CLI_SELF_TIMEOUT_MARGIN_S
        seen = {}

        async def fake_run_cli(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["timeout_s"] = kwargs.get("timeout_s")
            return {"returncode": 0, "stdout": "ok", "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 100, "default"))

        argv = seen["argv"]
        inner = argv[argv.index("--print-timeout") + 1]
        self.assertEqual(inner, "%ds" % (100 - CLI_SELF_TIMEOUT_MARGIN_S))
        self.assertEqual(seen["timeout_s"], 100)
        self.assertLess(int(inner.rstrip("s")), seen["timeout_s"])

    def test_agy_inner_deadline_never_goes_below_one(self):
        import providers.agy as agy_mod
        seen = {}

        async def fake_run_cli(argv, **kwargs):
            seen["argv"] = list(argv)
            return {"returncode": 0, "stdout": "ok", "stderr": "",
                    "timed_out": False, "watched": False}

        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 2, "default"))

        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--print-timeout") + 1], "1s")

    def test_agy_prompt_is_positional_and_not_sent_on_stdin(self):
        import providers.agy as agy_mod
        seen = {}

        async def fake_run_cli(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["stdin_data"] = kwargs.get("stdin_data")
            # agy print mode now runs --output-format json (issue #22 P1.1), so the
            # fake speaks the real shape: reply in .response, counts in .usage.
            return {"returncode": 0,
                    "stdout": json.dumps({"response": "ok\n",
                                          "usage": {"input_tokens": 11,
                                                    "output_tokens": 2,
                                                    "cache_read_tokens": 0}}),
                    "stderr": "", "timed_out": False, "watched": False}

        prompt = "What is 17 times 3? Answer with digits only."
        # The host fence wraps argv on macOS; this test pins the vendor argv.
        with mock.patch.object(agy_mod.vendor_bin, "which", return_value="/usr/bin/agy"), \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli), \
             mock.patch.object(
                 agy_mod, "prepare_host_fence",
                 side_effect=lambda argv, **kw: (argv, kw["env"], {})):
            res = asyncio.run(
                agy_mod.run_agy_subprocess(prompt, "gemini-test", 60, "default"))

        self.assertTrue(res["ok"], res)
        self.assertEqual(
            seen["argv"],
            ["/usr/bin/agy", "-p", prompt, "--model", "gemini-test",
             "--print-timeout", "55s", "--output-format", "json"])
        self.assertIsNone(seen["stdin_data"])
        self.assertEqual(res["text"], "ok")
        self.assertEqual(res["usage"]["input_tokens"], 11)
        self.assertEqual(res["usage"]["cached_input_tokens"], 0)   # measured zero
        self.assertIsNone(res["usage"]["cache_write_tokens"])      # agy has no such field
        self.assertEqual(res["usage"]["source"], "provider")

    def test_agy_oversized_prompt_fails_without_spawning(self):
        import providers.agy as agy_mod
        spawned = []

        async def fake_run_cli(*args, **kwargs):
            spawned.append((args, kwargs))
            return {}

        with mock.patch.object(agy_mod, "AGY_PROMPT_MAX_ARGV_BYTES", 3), \
             mock.patch.object(agy_mod.vendor_bin, "which",
                               return_value="/usr/bin/agy") as sh_which, \
             mock.patch.object(agy_mod, "run_cli", fake_run_cli):
            res = asyncio.run(
                agy_mod.run_agy_subprocess("éé", "gemini-test", 60, "default"))

        self.assertEqual(res["ok"], False)
        self.assertEqual(res["reason"], "payload_too_large")
        self.assertEqual(
            res["detail"],
            "agy prompt cap 3 bytes exceeded (4); "
            "use codex/<model> for large payloads")
        self.assertEqual(spawned, [])
        sh_which.assert_not_called()


class TestDrainBudget(unittest.TestCase):
    """Shutdown drain must not cancel a call that is still inside its own budget."""

    def tearDown(self):
        server._pending_tasks.clear()
        server._task_budgets.clear()

    def test_big_context_budget_exceeds_its_own_timeout(self):
        """Regression: flat DEFAULT_TIMEOUT_S + 30 (210s) cut big_context (240s) short."""
        from providers.config import BIG_CONTEXT_TIMEOUT_S
        budget = server.call_budget_s("llm_big_context", {})
        self.assertGreater(budget, BIG_CONTEXT_TIMEOUT_S)

    def test_model_chain_multiplies_the_budget(self):
        from providers.config import DEFAULT_TIMEOUT_S, RATE_LIMIT_RETRIES
        one = server.call_budget_s("llm_generate", {"model": "grok-4.5"})
        three = server.call_budget_s(
            "llm_generate", {"model": "grok-4.5,codex/gpt-5.6-sol,Gemini 3.5 Flash"})
        self.assertGreater(three, one * 2)
        # one model may still burn timeout twice: the rate_limit retry.
        self.assertGreaterEqual(one, DEFAULT_TIMEOUT_S * (RATE_LIMIT_RETRIES + 1))

    def test_caller_timeout_is_honoured_and_invalid_falls_back(self):
        from providers.config import DEFAULT_TIMEOUT_S
        self.assertGreaterEqual(
            server.call_budget_s("llm_generate", {"timeout_s": 600}), 600)
        for bad in ({"timeout_s": "abc"}, {"timeout_s": 0}, {"timeout_s": -5}):
            self.assertGreaterEqual(
                server.call_budget_s("llm_generate", bad), DEFAULT_TIMEOUT_S)

    def test_drain_wait_tracks_the_slowest_pending_call(self):
        async def drive():
            slow = asyncio.create_task(asyncio.sleep(0))
            server._track_task(slow, budget_s=500)
            fast = asyncio.create_task(asyncio.sleep(0))
            server._track_task(fast, budget_s=10)
            wait_s = server._drain_budget_s()
            await asyncio.gather(slow, fast)
            return wait_s

        self.assertEqual(asyncio.run(drive()), 500 + server.DRAIN_MARGIN_S)

    def test_drain_wait_is_capped(self):
        async def drive():
            t = asyncio.create_task(asyncio.sleep(0))
            server._track_task(t, budget_s=10 ** 6)
            wait_s = server._drain_budget_s()
            await t
            return wait_s

        self.assertEqual(asyncio.run(drive()), server.DRAIN_MAX_S)

    def test_finished_task_budget_is_released(self):
        async def drive():
            t = asyncio.create_task(asyncio.sleep(0))
            server._track_task(t, budget_s=500)
            await t
            await asyncio.sleep(0)  # let done callbacks run

        asyncio.run(drive())
        self.assertEqual(server._task_budgets, {})
        self.assertEqual(server._drain_budget_s(),
                         server.DEFAULT_TIMEOUT_S + server.DRAIN_MARGIN_S)

    def test_drain_does_not_cancel_a_task_inside_its_budget(self):
        finished = []

        async def drive():
            async def work():
                await asyncio.sleep(0.2)
                finished.append(True)

            server._track_task(asyncio.create_task(work()), budget_s=5)
            await server._drain_pending()

        asyncio.run(drive())
        self.assertEqual(finished, [True])

    def test_drain_still_cancels_a_task_past_its_budget(self):
        async def drive():
            async def stuck():
                await asyncio.sleep(60)

            t = asyncio.create_task(stuck())
            server._track_task(t)
            await server._drain_pending(timeout_s=0.1)
            return t

        t = asyncio.run(drive())
        self.assertTrue(t.cancelled())


if __name__ == "__main__":
    unittest.main()
