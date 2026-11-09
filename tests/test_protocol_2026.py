#!/usr/bin/env python3
# no-opencc
"""Executable seam for issue #22 P0: dual-era MCP protocol contract (2026-07-28).

Hermetic unittest pins for legacy initialize + modern per-request `_meta`.
ALL new pins are intentionally RED against today's server (see REPORT.md).
No production code lives here; Order 2/2 implements against this file.

Convention (repo): unittest, sync `def test_...` + `asyncio.run(...)`, stdlib only.
Harness: drive `server.handle_message` in-process and capture JSON-RPC lines written
via `server._send` (same capture idea as TestStdoutWriteLock in test_server.py).
Never spawn a subprocess; never touch the network.
"""
import asyncio
import io
import json
import sys
import unittest
from unittest import mock

import server
from tests.test_server import _by_id

# Supported protocol versions this dual-era server is expected to advertise.
SUPPORTED_LEGACY = "2025-06-18"
SUPPORTED_MODERN = "2026-07-28"
SUPPORTED_VERSIONS = (SUPPORTED_MODERN, SUPPORTED_LEGACY)

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

# Real agy stderr fragment (dispatcher-captured live) for unknown-model typing.
AGY_UNKNOWN_MODEL_STDERR = (
    "model Gemini 3.5 Flsh (High) is not recognized as a known model "
    "or custom model in settings"
)


def modern_meta(protocol_version=SUPPORTED_MODERN, caps=None, client_info=None,
                include_protocol_version=True, include_capabilities=True):
    """Build a modern-era `_meta` object with the pinned key strings."""
    meta = {}
    if include_protocol_version:
        meta[META_PROTOCOL_VERSION] = protocol_version
    if include_capabilities:
        meta[META_CLIENT_CAPABILITIES] = {} if caps is None else caps
    if client_info is not None:
        meta[META_CLIENT_INFO] = client_info
    return meta


def exchange(messages, raw_lines=False):
    """Drive `handle_message` in-process; capture JSON-RPC lines from `_send`.

    Reuses the capture pattern from test_server.TestStdoutWriteLock (stdout
    redirect around `_send`) and `_by_id` for id lookup. tools/call schedules a
    background task — we drain `_pending_tasks` before restoring stdout.

    raw_lines=False (default): return parsed JSON-RPC objects (existing callers).
    raw_lines=True: return the raw response line strings so consumers can measure
    wire bytes without reimplementing the capture (bench/measure.py M7).
    """
    async def _run():
        server._write_lock = None
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            for msg in messages:
                await server.handle_message(msg)
            pending = {t for t in server._pending_tasks if not t.done()}
            if pending:
                await asyncio.wait(pending, timeout=30)
        finally:
            sys.stdout = old
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        if raw_lines:
            return lines
        return [json.loads(ln) for ln in lines]

    return asyncio.run(_run())


def _primitive_type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_against_output_schema(instance, schema):
    """Small local JSON-Schema subset checker (stdlib only, no jsonschema).

    Checks declared `type`, `required` keys present, and each declared
    property's primitive type when given. Returns a list of error strings
    (empty == ok).
    """
    errors = []
    if not isinstance(schema, dict):
        return ["schema is not a dict"]
    expected_type = schema.get("type")
    if expected_type is not None:
        actual = _primitive_type_name(instance)
        # JSON Schema "number" accepts int; "integer" is strict.
        if expected_type == "number" and actual in ("integer", "number"):
            pass
        elif actual != expected_type:
            errors.append("type: expected %r got %r" % (expected_type, actual))
    if expected_type == "object" or isinstance(instance, dict):
        if not isinstance(instance, dict):
            return errors
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                errors.append("missing required key %r" % key)
        properties = schema.get("properties") or {}
        for key, prop_schema in properties.items():
            if key not in instance:
                continue
            if not isinstance(prop_schema, dict):
                continue
            prop_type = prop_schema.get("type")
            if prop_type is None:
                continue
            actual = _primitive_type_name(instance[key])
            if prop_type == "number" and actual in ("integer", "number"):
                continue
            if actual != prop_type:
                errors.append(
                    "property %r: expected type %r got %r" % (key, prop_type, actual)
                )
    return errors


class TestALegacyPath(unittest.TestCase):
    """A. Legacy path stays intact (regression guard)."""

    def test_a1_initialize_echoes_legacy_protocol_version(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": SUPPORTED_LEGACY, "capabilities": {}},
        }])
        r = _by_id(out, 1)["result"]
        self.assertEqual(r["protocolVersion"], SUPPORTED_LEGACY)
        self.assertIn("capabilities", r)
        self.assertIn("serverInfo", r)
        self.assertIn("name", r["serverInfo"])
        self.assertIn("version", r["serverInfo"])

    def test_a2_initialize_without_protocol_version_uses_server_legacy(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"capabilities": {}},
        }])
        r = _by_id(out, 1)["result"]
        pv = r.get("protocolVersion")
        self.assertTrue(pv, "protocolVersion must not be null/empty")
        self.assertIn(pv, SUPPORTED_VERSIONS)

    def test_a3_initialize_unsupported_version_must_not_echo_as_accepted(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "1900-01-01", "capabilities": {}},
        }])
        r = _by_id(out, 1)["result"]
        self.assertNotEqual(
            r.get("protocolVersion"), "1900-01-01",
            "unsupported version must not be echoed as accepted",
        )
        self.assertIn(r.get("protocolVersion"), SUPPORTED_VERSIONS)


class TestBModernPath(unittest.TestCase):
    """B. Modern path (per-request `_meta`) + dual-era selection."""

    def test_b4_server_discover_shape(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "server/discover",
            "params": {"_meta": modern_meta()},
        }])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        r = msg["result"]
        self.assertEqual(r.get("resultType"), "complete")
        versions = r.get("supportedVersions") or []
        self.assertIn(SUPPORTED_MODERN, versions)
        self.assertIn(SUPPORTED_LEGACY, versions)
        self.assertIn("tools", (r.get("capabilities") or {}))
        meta = r.get("_meta") or {}
        sinfo = meta.get(META_SERVER_INFO) or {}
        self.assertTrue(sinfo.get("name"))
        self.assertTrue(sinfo.get("version"))
        self.assertIsInstance(r.get("ttlMs"), int)
        self.assertGreaterEqual(r["ttlMs"], 0)
        self.assertEqual(r.get("cacheScope"), "public")

    def test_b5_tools_list_modern_meta_shape(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": modern_meta()},
        }])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        r = msg["result"]
        self.assertEqual(r.get("resultType"), "complete")
        self.assertIn("tools", r)
        self.assertIsInstance(r.get("ttlMs"), int)
        self.assertGreaterEqual(r["ttlMs"], 0)
        self.assertEqual(r.get("cacheScope"), "public")

    def test_b6_unsupported_modern_protocol_version_is_32022(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": modern_meta(protocol_version="1900-01-01")},
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, "unsupported modern protocolVersion must be a JSON-RPC error")
        err = msg["error"]
        self.assertEqual(err["code"], -32022)
        data = err.get("data") or {}
        supported = data.get("supported") or []
        self.assertTrue(supported, "data.supported must be non-empty")
        self.assertIn(SUPPORTED_MODERN, supported)
        self.assertEqual(data.get("requested"), "1900-01-01")

    def test_b7_modern_meta_missing_protocol_version_is_32602(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {
                "_meta": modern_meta(include_protocol_version=False),
            },
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, "missing required _meta protocolVersion must be -32602")
        self.assertEqual(msg["error"]["code"], -32602)

    def test_b8_modern_meta_missing_client_capabilities_is_32602(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {
                "_meta": modern_meta(include_capabilities=False),
            },
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, "missing required _meta clientCapabilities must be -32602")
        self.assertEqual(msg["error"]["code"], -32602)

    def test_b9_tools_list_without_meta_still_succeeds_legacy(self):
        """Dual-era pin: pure legacy client (post-initialize) must not get -32602."""
        out = exchange([
            {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": SUPPORTED_LEGACY, "capabilities": {}},
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        msg = _by_id(out, 2)
        self.assertNotIn("error", msg, msg)
        if "error" in msg:
            self.assertNotEqual(msg["error"].get("code"), -32602)
        self.assertIn("tools", msg["result"])


class TestCToolsCallResultContract(unittest.TestCase):
    """C. tools/call result contract + outputSchema on tool definitions."""

    def _tools_list(self):
        out = exchange([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
        return _by_id(out, 1)["result"]["tools"]

    def test_c10_every_tool_declares_object_output_schema(self):
        tools = self._tools_list()
        self.assertTrue(tools, "tools/list must return tools")
        for t in tools:
            schema = t.get("outputSchema")
            self.assertIsInstance(
                schema, dict,
                "tool %r missing outputSchema dict" % t.get("name"),
            )
            self.assertEqual(
                schema.get("type"), "object",
                "tool %r outputSchema.type" % t.get("name"),
            )

    def _mock_ok_call(self):
        async def _ok(name, args):
            return {
                "ok": True,
                "text": "hello-from-mock",
                "model": "mock-model",
                "reason": None,
            }
        return _ok

    def test_c11_successful_tools_call_has_result_type_and_structured(self):
        with mock.patch.object(server, "_call_tool", self._mock_ok_call()):
            out = exchange([{
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "llm_generate", "arguments": {"prompt": "hi"}},
            }])
        r = _by_id(out, 1)["result"]
        self.assertEqual(r.get("resultType"), "complete")
        self.assertIs(r.get("isError"), False)
        content = r.get("content") or []
        self.assertTrue(content and content[0].get("type") == "text")
        self.assertIsInstance(r.get("structuredContent"), dict)

    def test_c12_compatibility_mirror_matches_structured_content(self):
        with mock.patch.object(server, "_call_tool", self._mock_ok_call()):
            out = exchange([{
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "llm_generate", "arguments": {"prompt": "hi"}},
            }])
        r = _by_id(out, 1)["result"]
        text_payload = json.loads(r["content"][0]["text"])
        self.assertEqual(text_payload, r.get("structuredContent"))

    def test_c13_failing_tools_call_has_structured_envelope(self):
        # Empty prompt is rejected before any CLI — hermetic, no network.
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "llm_generate", "arguments": {"prompt": "   "}},
        }])
        r = _by_id(out, 1)["result"]
        self.assertIs(r.get("isError"), True)
        sc = r.get("structuredContent")
        self.assertIsInstance(sc, dict, "failing call must still carry structuredContent")
        self.assertIs(sc.get("ok"), False)
        self.assertIn("reason", sc)

    def test_c14_tools_call_has_no_caching_hints(self):
        with mock.patch.object(server, "_call_tool", self._mock_ok_call()):
            out = exchange([{
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "llm_generate", "arguments": {"prompt": "hi"}},
            }])
        r = _by_id(out, 1)["result"]
        # Spec: caching hints only on list/discover complete results, NOT tools/call.
        self.assertNotIn("ttlMs", r)
        self.assertNotIn("cacheScope", r)

    def test_c15_structured_content_validates_against_output_schema(self):
        tools = self._tools_list()
        by_name = {t["name"]: t for t in tools}
        self.assertIn("llm_generate", by_name)
        schema = by_name["llm_generate"].get("outputSchema")
        self.assertIsInstance(schema, dict, "llm_generate must declare outputSchema")

        with mock.patch.object(server, "_call_tool", self._mock_ok_call()):
            out = exchange([{
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "llm_generate", "arguments": {"prompt": "hi"}},
            }])
        r = _by_id(out, 1)["result"]
        sc = r.get("structuredContent")
        self.assertIsInstance(sc, dict)
        errors = validate_against_output_schema(sc, schema)
        self.assertEqual(errors, [], "structuredContent vs outputSchema: %s" % errors)


class TestDUnknownModelTyping(unittest.TestCase):
    """D. unknown model typing via classify_cli_error."""

    def test_d16_agy_unrecognized_model_is_unknown_model(self):
        reason = server.classify_cli_error("", AGY_UNKNOWN_MODEL_STDERR, 1)
        self.assertEqual(reason, "unknown_model")

    def test_d17_ordinary_error_still_runtime_error(self):
        # Negative twin from test_server.TestErrorClassification — must not widen.
        reason = server.classify_cli_error("", "boom segfault", 1)
        self.assertEqual(reason, "runtime_error")



class TestLegacyMetaIsNotModern(unittest.TestCase):
    """_meta is not by itself a modern marker (foreman gap-fill).

    2025-06-18 clients put `progressToken` in `_meta`. If any `_meta` counted as
    modern, every legacy client using progress would start getting -32602.
    """

    def test_legacy_progress_token_meta_is_served_as_legacy(self):
        res = exchange([{"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                         "params": {"_meta": {"progressToken": "abc123"}}}])[0]
        self.assertNotIn("error", res,
                         "legacy _meta.progressToken must not be treated as modern")
        self.assertIn("tools", res["result"])

    def test_meta_with_only_client_capabilities_is_modern_and_rejected(self):
        res = exchange([{"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                         "params": {"_meta": {
                             "io.modelcontextprotocol/clientCapabilities": {}}}}])[0]
        self.assertEqual(res["error"]["code"], -32602)


class TestF1NonDictParams(unittest.TestCase):
    """F1: non-dict params must not break tools/list (era detection only)."""

    def test_f1_tools_list_params_list_returns_tools(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 40, "method": "tools/list",
            "params": [1, 2],
        }])
        msg = _by_id(out, 40)
        self.assertNotIn("error", msg, msg)
        self.assertIn("tools", msg["result"])

    def test_f1_tools_list_params_string_returns_tools(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 41, "method": "tools/list",
            "params": "not-an-object",
        }])
        msg = _by_id(out, 41)
        self.assertNotIn("error", msg, msg)
        self.assertIn("tools", msg["result"])

    def test_f1_tools_list_params_absent_returns_tools(self):
        out = exchange([{"jsonrpc": "2.0", "id": 42, "method": "tools/list"}])
        msg = _by_id(out, 42)
        self.assertNotIn("error", msg, msg)
        self.assertIn("tools", msg["result"])


class TestF2InitializeHandshakeVersions(unittest.TestCase):
    """F2: initialize must not negotiate a handshake-less version."""

    def test_f2_initialize_modern_version_not_echoed(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": SUPPORTED_MODERN, "capabilities": {}},
        }])
        r = _by_id(out, 1)["result"]
        self.assertNotEqual(r.get("protocolVersion"), SUPPORTED_MODERN)
        self.assertEqual(r.get("protocolVersion"), SUPPORTED_LEGACY)

    def test_f2_initialize_legacy_version_still_echoed(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": SUPPORTED_LEGACY, "capabilities": {}},
        }])
        r = _by_id(out, 1)["result"]
        self.assertEqual(r.get("protocolVersion"), SUPPORTED_LEGACY)


class TestF3ModernMetaTypes(unittest.TestCase):
    """F3: modern _meta value types must be validated."""

    def test_f3_protocol_version_non_string_is_32602_not_32022(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": {
                META_PROTOCOL_VERSION: 123,
                META_CLIENT_CAPABILITIES: {},
            }},
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, msg)
        self.assertEqual(msg["error"]["code"], -32602)
        self.assertNotEqual(msg["error"]["code"], -32022)

    def test_f3_client_capabilities_non_object_is_32602(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": {
                META_PROTOCOL_VERSION: SUPPORTED_MODERN,
                META_CLIENT_CAPABILITIES: "notanobject",
            }},
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, msg)
        self.assertEqual(msg["error"]["code"], -32602)

    def test_f3_valid_types_still_served(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
            "params": {"_meta": modern_meta()},
        }])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        self.assertIn("tools", msg["result"])


class TestF4ModernNotification(unittest.TestCase):
    """F4: modern-marked notification must fall through to dispatch (no response)."""

    def test_f4_modern_notification_produces_no_response(self):
        out = exchange([{
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {"_meta": modern_meta()},
        }])
        self.assertEqual(out, [], "notification must not produce any response line: %r" % out)


class TestF5DiscoverModernValidation(unittest.TestCase):
    """F5: server/discover validates modern _meta only when it carries a modern marker."""

    def test_f5_discover_unsupported_modern_version_is_32022(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "server/discover",
            "params": {"_meta": modern_meta(protocol_version="1900-01-01")},
        }])
        msg = _by_id(out, 1)
        self.assertIn("error", msg, msg)
        self.assertEqual(msg["error"]["code"], -32022)
        data = msg["error"].get("data") or {}
        self.assertTrue(data.get("supported"), "data.supported must be present")
        self.assertIn(SUPPORTED_MODERN, data["supported"])

    def test_f5_discover_no_meta_is_era_probe(self):
        out = exchange([{"jsonrpc": "2.0", "id": 1, "method": "server/discover"}])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        r = msg["result"]
        self.assertEqual(r.get("resultType"), "complete")
        self.assertIn(SUPPORTED_MODERN, r.get("supportedVersions") or [])

    def test_f5_discover_valid_modern_meta_succeeds(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "server/discover",
            "params": {"_meta": modern_meta()},
        }])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        r = msg["result"]
        self.assertEqual(r.get("resultType"), "complete")
        self.assertIn(SUPPORTED_MODERN, r.get("supportedVersions") or [])


class TestF6ResultTypeAndServerInfo(unittest.TestCase):
    """F6: resultType + serverInfo _meta applied centrally via _result()."""

    def _assert_server_info(self, result):
        meta = result.get("_meta") or {}
        sinfo = meta.get(META_SERVER_INFO) or {}
        self.assertTrue(sinfo.get("name"), "missing serverInfo name in %r" % result)
        self.assertTrue(sinfo.get("version"), "missing serverInfo version in %r" % result)

    def test_f6_modern_ping_has_result_type_and_server_info(self):
        out = exchange([{
            "jsonrpc": "2.0", "id": 1, "method": "ping",
            "params": {"_meta": modern_meta()},
        }])
        msg = _by_id(out, 1)
        self.assertNotIn("error", msg, msg)
        r = msg["result"]
        self.assertEqual(r.get("resultType"), "complete")
        self._assert_server_info(r)

    def test_f6_tools_list_call_discover_carry_result_type_and_server_info(self):
        async def _ok(name, args):
            return {"ok": True, "text": "hi", "model": "mock", "reason": None}

        with mock.patch.object(server, "_call_tool", _ok):
            out = exchange([
                {
                    "jsonrpc": "2.0", "id": 1, "method": "tools/list",
                    "params": {"_meta": modern_meta()},
                },
                {
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {
                        "name": "llm_generate",
                        "arguments": {"prompt": "hi"},
                        "_meta": modern_meta(),
                    },
                },
                {
                    "jsonrpc": "2.0", "id": 3, "method": "server/discover",
                    "params": {"_meta": modern_meta()},
                },
            ])
        for rid in (1, 2, 3):
            msg = _by_id(out, rid)
            self.assertNotIn("error", msg, msg)
            r = msg["result"]
            self.assertEqual(r.get("resultType"), "complete", "id=%s" % rid)
            self._assert_server_info(r)


if __name__ == "__main__":
    unittest.main()
