#!/usr/bin/env python3
# no-opencc
"""Issue #43: modern MCP path e2e + notification dispatch behavioral pins.

(1) Modern path e2e — drive a real `server.py` subprocess over stdio (same
    house idiom as test_server.run_server / test_jsonrpc_robustness.run_raw)
    with a multi-step 2026-07-28 flow and NO legacy `initialize` handshake.
    Hermetic: PATH blanked so tool backends fail fast with cli_missing.

    Scope honesty (issue #43 item 1): this is a protocol-level real client on
    the wire, not an external ecosystem MCP SDK. A genuine third-party 2026
    client remains blocked on the ecosystem; this closes the in-repo gap of
    "modern path never exercised end-to-end through the real stdio entry."

(2) Notification dispatch pins — internal seam (`_notification_dispatch_log`
    + stderr `notification_dispatched`) so "dispatched and intentionally
    silent" is distinguishable from "never dispatched" (issue #43 item 2).
    Pins ordering, target method, and recorded payload shape summary.
"""
import asyncio
import json
import os
import subprocess
import sys
import unittest

import server
from tests.test_protocol_2026 import (
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    SUPPORTED_LEGACY,
    SUPPORTED_MODERN,
    exchange,
    modern_meta,
)
from tests.test_server import _by_id

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(HERE, "server.py")


def run_raw(raw_lines, timeout=30):
    """Feed literal stdin lines to a real server process; return (msgs, stderr).

    Same shape as test_jsonrpc_robustness.run_raw (stdio integration idiom).
    Fails if the process exits non-zero after writing responses (post-response
    crash must not be silent).
    """
    env = dict(os.environ)
    env["PATH"] = "/nonexistent"
    inp = "".join(line if line.endswith("\n") else line + "\n" for line in raw_lines)
    proc = subprocess.run(
        [sys.executable, SERVER],
        input=inp,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    msgs = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert proc.returncode == 0, (
        "server.py exited %s after responses (stdout=%r stderr=%r)"
        % (proc.returncode, proc.stdout[-500:], (proc.stderr or "")[-500:])
    )
    return msgs, proc.stderr


def run_server(messages, timeout=30):
    """JSON-object list → real server.py over stdio. PATH blanked (hermetic)."""
    return run_raw([json.dumps(m) for m in messages], timeout=timeout)


def _modern_params(**extra):
    """params object carrying a valid modern `_meta` plus any extra fields."""
    p = {"_meta": modern_meta()}
    p.update(extra)
    return p


def _stderr_events(stderr, event_name):
    """Parse structured log_event lines that carry the given event name."""
    found = []
    for line in (stderr or "").splitlines():
        if event_name not in line:
            continue
        # log_event format: "[omni-llm-mcp] {json...}"
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            payload = json.loads(line[brace:])
        except Exception:
            continue
        if payload.get("event") == event_name:
            found.append(payload)
    return found


class TestModernPathE2ESubprocess(unittest.TestCase):
    """Real stdio client exercises the 2026-07-28 path without initialize."""

    def test_modern_session_discover_list_ping_call_without_initialize(self):
        """Full modern session: discover → tools/list → ping → tools/call
        → unsupported-version probe.

        No `initialize` / `notifications/initialized` — 2026-07-28 has no
        handshake. Each request carries its own modern `_meta`. The final
        unsupported-version request is the era-detection canary: if the modern
        path is broken (requests served as legacy), it returns a success result
        instead of -32022 and this test fails loudly.
        """
        msgs, _stderr = run_server([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": _modern_params(),
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": _modern_params(),
            },
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "ping",
                "params": _modern_params(),
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": _modern_params(
                    name="list_models",
                    arguments={},
                ),
            },
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/list",
                "params": {
                    "_meta": modern_meta(protocol_version="1900-01-01"),
                },
            },
        ])
        self.assertEqual(len(msgs), 5, "expected 5 responses, got %r" % msgs)

        # --- server/discover ---
        d = _by_id(msgs, 1)
        self.assertNotIn("error", d, d)
        dr = d["result"]
        self.assertEqual(dr.get("resultType"), "complete")
        versions = dr.get("supportedVersions") or []
        self.assertIn(SUPPORTED_MODERN, versions)
        self.assertIn(SUPPORTED_LEGACY, versions)
        self.assertIsInstance(dr.get("ttlMs"), int)
        self.assertEqual(dr.get("cacheScope"), "public")
        sinfo = (dr.get("_meta") or {}).get(META_SERVER_INFO) or {}
        self.assertTrue(sinfo.get("name"))
        self.assertTrue(sinfo.get("version"))

        # --- tools/list ---
        tl = _by_id(msgs, 2)
        self.assertNotIn("error", tl, tl)
        tr = tl["result"]
        self.assertEqual(tr.get("resultType"), "complete")
        self.assertIsInstance(tr.get("tools"), list)
        self.assertGreater(len(tr["tools"]), 0)
        self.assertIsInstance(tr.get("ttlMs"), int)
        self.assertEqual(tr.get("cacheScope"), "public")

        # --- ping ---
        p = _by_id(msgs, 3)
        self.assertNotIn("error", p, p)
        self.assertEqual(p["result"].get("resultType"), "complete")

        # --- tools/call (cli missing → LOUD structured envelope) ---
        c = _by_id(msgs, 4)
        self.assertNotIn("error", c, c)
        cr = c["result"]
        self.assertEqual(cr.get("resultType"), "complete")
        self.assertIn("structuredContent", cr)
        sc = cr["structuredContent"]
        self.assertIsInstance(sc, dict)
        self.assertIn("ok", sc)
        # Claimed outcome under blank PATH: list_models fails with cli_missing.
        # An always-{"ok": true} stub must fail here.
        self.assertIs(sc.get("ok"), False)
        self.assertEqual(sc.get("reason"), "cli_missing")
        self.assertTrue(cr.get("isError"))
        # compatibility mirror: text block == structuredContent
        content = cr.get("content") or []
        self.assertTrue(content and content[0].get("type") == "text")
        mirrored = json.loads(content[0]["text"])
        self.assertEqual(mirrored, sc)
        # tools/call must NOT carry list/discover caching hints
        self.assertNotIn("ttlMs", cr)
        self.assertNotIn("cacheScope", cr)

        # --- era-detection canary: unsupported modern version → -32022 ---
        bad = _by_id(msgs, 5)
        self.assertIn("error", bad, "modern path must reject unsupported version: %r" % bad)
        self.assertEqual(bad["error"]["code"], -32022)
        data = bad["error"].get("data") or {}
        self.assertIn(SUPPORTED_MODERN, data.get("supported") or [])
        self.assertEqual(data.get("requested"), "1900-01-01")

    def test_valid_modern_request_takes_modern_branch(self):
        """Positive pin: valid supported modern `_meta` is classified modern
        and a real request completes through that branch.

        Falsifier: if `_is_modern_request` returns False only for valid supported
        modern metadata (still True for missing/malformed/unsupported), this
        assertion fails while the unsupported-version canaries may still pass.
        """
        params = _modern_params()
        self.assertTrue(
            server._is_modern_request(params),
            "valid modern _meta must take the modern era branch",
        )
        # Through real stdio: success + modern discover observables.
        msgs, _ = run_server([{
            "jsonrpc": "2.0",
            "id": 20,
            "method": "server/discover",
            "params": params,
        }])
        msg = _by_id(msgs, 20)
        self.assertNotIn("error", msg, msg)
        versions = msg["result"].get("supportedVersions") or []
        self.assertIn(SUPPORTED_MODERN, versions)
        self.assertEqual(msg["result"].get("resultType"), "complete")
        self.assertIsInstance(msg["result"].get("ttlMs"), int)
        # Contrasts: missing / unsupported still rejected (modern branch active).
        bad_missing, _ = run_server([{
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/list",
            "params": {"_meta": modern_meta(include_protocol_version=False)},
        }])
        self.assertEqual(_by_id(bad_missing, 21)["error"]["code"], -32602)
        bad_ver, _ = run_server([{
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/list",
            "params": {"_meta": modern_meta(protocol_version="1900-01-01")},
        }])
        self.assertEqual(_by_id(bad_ver, 22)["error"]["code"], -32022)

    def test_modern_unsupported_version_is_32022_with_supported_list(self):
        """A real client asking for an unknown modern version gets -32022 + data.supported."""
        msgs, _ = run_server([{
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/list",
            "params": {
                "_meta": modern_meta(protocol_version="1900-01-01"),
            },
        }])
        msg = _by_id(msgs, 10)
        self.assertIn("error", msg, msg)
        self.assertEqual(msg["error"]["code"], -32022)
        data = msg["error"].get("data") or {}
        supported = data.get("supported") or []
        self.assertIn(SUPPORTED_MODERN, supported)
        self.assertIn(SUPPORTED_LEGACY, supported)
        self.assertEqual(data.get("requested"), "1900-01-01")

    def test_modern_missing_required_meta_fields_are_32602(self):
        """Real client omitting a required modern field is rejected at the wire."""
        msgs, _ = run_server([
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/list",
                "params": {"_meta": modern_meta(include_protocol_version=False)},
            },
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/list",
                "params": {"_meta": modern_meta(include_capabilities=False)},
            },
        ])
        self.assertEqual(_by_id(msgs, 11)["error"]["code"], -32602)
        self.assertEqual(_by_id(msgs, 12)["error"]["code"], -32602)

    def test_discover_with_unsupported_modern_version_is_32022(self):
        """Unsupported modern version on server/discover → -32022.

        Falsifier note: if era detection stops treating modern fields as modern,
        unsupported-version requests would be served as legacy discover instead
        of -32022. (Demonstrated live under a temporary mutation in the work-order
        report; this test is the permanent green pin of the correct behavior.)
        """
        msgs, _ = run_server([{
            "jsonrpc": "2.0",
            "id": 99,
            "method": "server/discover",
            "params": {
                "_meta": {
                    META_PROTOCOL_VERSION: "1900-01-01",
                    META_CLIENT_CAPABILITIES: {},
                },
            },
        }])
        msg = _by_id(msgs, 99)
        self.assertIn("error", msg)
        self.assertEqual(msg["error"]["code"], -32022)


class TestNotificationDispatchPin(unittest.TestCase):
    """Issue #43 item 2: dispatch is observable independent of handler body."""

    def setUp(self):
        server._notification_dispatch_log.clear()

    def tearDown(self):
        server._notification_dispatch_log.clear()

    def test_dispatch_records_target_and_order(self):
        """Three known notification methods are recorded in arrival order,
        each bound to its own params shape summary (method + param_keys).
        """
        out = exchange([
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "initialized",
             "params": {"_meta": {"progressToken": "t1"}}},
            {"jsonrpc": "2.0", "method": "notifications/cancelled",
             "params": {"requestId": 7}},
        ])
        self.assertEqual(out, [], "notifications must produce no wire response: %r" % out)
        log = list(server._notification_dispatch_log)
        self.assertEqual(
            [(r["method"], r["param_keys"]) for r in log],
            [
                ("notifications/initialized", []),
                ("initialized", ["_meta"]),
                ("notifications/cancelled", ["requestId"]),
            ],
        )
        # has_params True for dict (incl. normalized empty) params
        self.assertTrue(all(r["has_params"] for r in log))

    def test_dispatch_payload_shape_records_params_object(self):
        """Recorded payload is a shape summary (param_keys), not the live dict."""
        params = {"requestId": 42, "reason": "user"}
        out = exchange([{
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": params,
        }])
        self.assertEqual(out, [])
        self.assertEqual(len(server._notification_dispatch_log), 1)
        rec = server._notification_dispatch_log[0]
        self.assertEqual(rec["method"], "notifications/cancelled")
        self.assertTrue(rec["has_params"])
        self.assertEqual(rec["param_keys"], ["reason", "requestId"])
        # Must not retain the client-controlled live object.
        self.assertNotIn("params", rec)

    def test_dispatch_absent_params_is_empty_object(self):
        """Absent params normalize to {} before the seam (same as other methods)."""
        out = exchange([{"jsonrpc": "2.0", "method": "notifications/initialized"}])
        self.assertEqual(out, [])
        self.assertEqual(len(server._notification_dispatch_log), 1)
        rec = server._notification_dispatch_log[0]
        self.assertTrue(rec["has_params"])
        self.assertEqual(rec["param_keys"], [])

    def test_dispatch_non_dict_params_has_params_false(self):
        """List params still record the method with has_params False."""
        out = exchange([{
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": ["not", "an", "object"],
        }])
        self.assertEqual(out, [])
        self.assertEqual(len(server._notification_dispatch_log), 1)
        rec = server._notification_dispatch_log[0]
        self.assertEqual(rec["method"], "notifications/cancelled")
        self.assertFalse(rec["has_params"])
        self.assertIsNone(rec["param_keys"])

    def test_unknown_method_notification_is_not_recorded_as_known(self):
        """Only the three known notification methods hit the dispatch seam."""
        out = exchange([
            {"jsonrpc": "2.0", "method": "notifications/something_else"},
            {"jsonrpc": "2.0", "method": "ping"},  # notification-shaped ping
        ])
        self.assertEqual(out, [])
        self.assertEqual(
            list(server._notification_dispatch_log), [],
            "unknown / non-notification methods must not use the known-notif seam",
        )

    def test_modern_marked_notification_still_dispatches(self):
        """Modern `_meta` on a notification skips validation and still dispatches (F4+)."""
        out = exchange([{
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {"_meta": modern_meta()},
        }])
        self.assertEqual(out, [])
        self.assertEqual(len(server._notification_dispatch_log), 1)
        rec = server._notification_dispatch_log[0]
        self.assertEqual(rec["method"], "notifications/initialized")
        self.assertTrue(rec["has_params"])
        self.assertEqual(rec["param_keys"], ["_meta"])

    def test_subprocess_notification_emits_log_event_and_no_stdout(self):
        """Subprocess: stderr carries notification_dispatched; stdout stays empty for them."""
        msgs, stderr = run_server([
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled",
             "params": {"requestId": 1}},
            # keep-alive probe so we know the process answered after the notifs
            {"jsonrpc": "2.0", "id": 50, "method": "ping"},
        ])
        # Only the ping is answered.
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["id"], 50)
        events = _stderr_events(stderr, "notification_dispatched")
        methods = [e.get("method") for e in events]
        self.assertEqual(
            methods,
            [
                "notifications/initialized",
                "initialized",
                "notifications/cancelled",
            ],
            "stderr events must pin target + order: %r" % events,
        )

    def test_handle_message_records_dispatch_directly(self):
        """handle_message records known notifications on the seam.

        Falsifier note: if the record helper is never called, the log stays
        empty — the pin that distinguishes drop from intentional silence.
        (Green path: called.)
        """
        asyncio.run(server.handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }))
        self.assertEqual(
            [r["method"] for r in server._notification_dispatch_log],
            ["notifications/initialized"],
        )

    def test_notification_dispatch_log_bounded_maxlen(self):
        """deque maxlen=64: oldest records are evicted after 65+ appends."""
        self.assertEqual(server._notification_dispatch_log.maxlen, 64)
        for i in range(65):
            server._record_notification_dispatch(
                "notifications/cancelled",
                {"requestId": i},
            )
        log = list(server._notification_dispatch_log)
        self.assertEqual(len(log), 64)
        # First append (requestId 0) evicted; oldest remaining is requestId 1.
        self.assertEqual(log[0]["param_keys"], ["requestId"])
        # Reconstruct which ids remain via a fresh record with unique keys would
        # not work — param_keys only. Re-record with distinct key sets:
        server._notification_dispatch_log.clear()
        for i in range(65):
            server._record_notification_dispatch(
                "notifications/cancelled",
                {"k%d" % i: True},
            )
        log = list(server._notification_dispatch_log)
        self.assertEqual(len(log), 64)
        self.assertEqual(log[0]["param_keys"], ["k1"])
        self.assertEqual(log[-1]["param_keys"], ["k64"])
        # k0 was the oldest and must be gone
        self.assertNotIn(["k0"], [r["param_keys"] for r in log])


if __name__ == "__main__":
    unittest.main()
