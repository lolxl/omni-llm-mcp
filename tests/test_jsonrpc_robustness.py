#!/usr/bin/env python3
# no-opencc
"""Executable seam for issue #31: JSON-RPC framing robustness.

Three PRE-EXISTING defects (measured identical on main before and after #22 P0):

  1. A batch array kills the read loop. `handle_message` assumes a dict; the outer
     exception handler in `main_loop` then calls `msg.get("id")` on the list and
     raises AGAIN, so the loop dies and the process stops answering for the rest of
     its lifetime.
  2. Malformed `params` / `arguments` surface as -32603 with raw AttributeError text
     instead of -32602 (Invalid params).
  3. Notification-shaped requests (no `id`) get `{"id": null, ...}` responses.
     JSON-RPC forbids responding to a notification at all.

ALL pins here are intentionally RED against server.py @ 72fbae4. No production code
lives in this file.

Harness: these defects live in `main_loop`'s read loop and in the id/params handling
that precedes task creation, so they CANNOT be pinned by driving `handle_message`
in-process (the way test_protocol_2026.py does). Every test below spawns the real
server over stdio, exactly like `test_server.run_server`, and several also assert on
stderr — hence the local `run_raw` rather than importing that helper.
No network: PATH is blanked so `shutil.which('agy')` returns None and every tool call
fails fast with `cli_missing`.
"""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(HERE, "server.py")

PING = {"jsonrpc": "2.0", "id": 99, "method": "ping"}


def run_raw(raw_lines, timeout=30):
    """Feed literal stdin lines to a real server process; return (msgs, stderr).

    `raw_lines` are strings, not objects, because several pins must send payloads
    that `json.dumps` of a dict cannot express (a top-level array).
    """
    env = dict(os.environ)
    env["PATH"] = "/nonexistent"          # shutil.which('agy') -> None
    inp = "".join(line if line.endswith("\n") else line + "\n" for line in raw_lines)
    proc = subprocess.run([sys.executable, SERVER], input=inp, text=True,
                          capture_output=True, env=env, timeout=timeout)
    msgs = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    return msgs, proc.stderr


def answered(msgs, id_):
    return [m for m in msgs if m.get("id") == id_]


class TestBatchArrayDoesNotKillTheLoop(unittest.TestCase):
    """Defect 1 — the worst of the three: one batch array ends the session."""

    def test_batch_array_gets_an_error_and_loop_survives(self):
        batch = json.dumps([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        msgs, _ = run_raw([batch, json.dumps(PING)])

        # The batch itself must be refused LOUDLY rather than crashing the reader.
        # Batches are legal JSON-RPC 2.0 but MCP stdio does not require them, so
        # -32600 (Invalid Request) is the honest answer. id is null: a batch has no
        # single id to answer under.
        errs = [m for m in msgs if "error" in m and m.get("id") is None]
        self.assertTrue(errs, "batch array must produce a LOUD error response")
        self.assertEqual(errs[0]["error"]["code"], -32600)

        # ...and the loop must still be alive for the NEXT request. This is the
        # assertion that actually matters.
        self.assertTrue(answered(msgs, 99),
                        "read loop died: request after a batch array was never answered")
        self.assertIn("result", answered(msgs, 99)[0])

    def test_empty_array_does_not_kill_the_loop(self):
        msgs, _ = run_raw(["[]", json.dumps(PING)])
        self.assertTrue(answered(msgs, 99),
                        "read loop died on an empty array")

    def test_scalar_toplevel_message_does_not_kill_the_loop(self):
        # Valid JSON, not an object: `"hello"` and `42` decode fine and then hit the
        # same `msg.get` assumption.
        msgs, _ = run_raw(['"hello"', "42", json.dumps(PING)])
        self.assertTrue(answered(msgs, 99),
                        "read loop died on a non-object top-level JSON value")

    def test_null_toplevel_message_does_not_kill_the_loop(self):
        msgs, _ = run_raw(["null", json.dumps(PING)])
        self.assertTrue(answered(msgs, 99),
                        "read loop died on a top-level null")


class TestMalformedParamsAreInvalidParams(unittest.TestCase):
    """Defect 2 — -32602, and no internal exception text on the wire."""

    def _assert_invalid_params(self, msgs, id_):
        got = answered(msgs, id_)
        self.assertTrue(got, "no response for id %r" % (id_,))
        self.assertIn("error", got[0], "expected a JSON-RPC error, got %r" % (got[0],))
        err = got[0]["error"]
        self.assertEqual(err["code"], -32602,
                         "expected -32602 Invalid params, got %r" % (err,))
        # Internal exception text must never reach the client.
        blob = json.dumps(err)
        for leak in ("has no attribute", "AttributeError", "Traceback"):
            self.assertNotIn(leak, blob,
                             "internal exception text leaked to the client: %r" % (err,))

    def test_string_arguments_is_invalid_params(self):
        msgs, stderr = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 50, "method": "tools/call",
             "params": {"name": "llm_generate", "arguments": "notanobject"}})])
        self._assert_invalid_params(msgs, 50)
        # The defect also left an unhandled task exception behind.
        self.assertNotIn("Task exception was never retrieved", stderr)

    def test_list_arguments_is_invalid_params(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 51, "method": "tools/call",
             "params": {"name": "llm_generate", "arguments": ["prompt", "hi"]}})])
        self._assert_invalid_params(msgs, 51)

    def test_string_params_is_invalid_params(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 52, "method": "tools/call",
             "params": "notanobject"})])
        self._assert_invalid_params(msgs, 52)

    def test_list_params_is_invalid_params(self):
        # JSON-RPC positional params. We accept objects only; say so with -32602.
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 53, "method": "tools/call",
             "params": ["llm_generate", {}]})])
        self._assert_invalid_params(msgs, 53)

    def test_non_string_tool_name_is_invalid_params(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 54, "method": "tools/call",
             "params": {"name": {"a": 1}, "arguments": {}}})])
        self._assert_invalid_params(msgs, 54)

    def test_loop_survives_malformed_params(self):
        msgs, _ = run_raw([
            json.dumps({"jsonrpc": "2.0", "id": 55, "method": "tools/call",
                        "params": {"name": "llm_generate", "arguments": "notanobject"}}),
            json.dumps(PING),
        ])
        self.assertTrue(answered(msgs, 99),
                        "read loop must survive malformed params")

    def test_uncomputable_budget_does_not_leak_or_answer_twice(self):
        # Found by the sol review seat and reproduced verbatim before fixing.
        # `1e309` decodes to float('inf'); `call_budget_s` does int(inf) and raises
        # OverflowError. Because Python evaluates `create_task(...)` BEFORE the
        # `budget_s=call_budget_s(...)` keyword argument, the task was already
        # scheduled when the raise happened -- so id 77 got BOTH a -32603 carrying
        # "cannot convert float infinity to integer" AND, later, a normal tool
        # result. Two responses for one id, plus the exception-text leak this issue
        # exists to close.
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 77, "method": "tools/call",
             "params": {"name": "llm_generate",
                        "arguments": {"prompt": "x", "timeout_s": 1e309}}})])
        got = answered(msgs, 77)
        self.assertEqual(len(got), 1,
                         "id 77 must be answered exactly once, got %d responses"
                         % len(got))
        self._assert_invalid_params(msgs, 77)

    def test_uncomputable_budget_variants(self):
        for i, bad in enumerate(("-1e309", "1e400"), start=78):
            msgs, _ = run_raw([
                '{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":'
                '{"name":"llm_generate","arguments":{"prompt":"x","timeout_s":%s}}}'
                % (i, bad)])
            self.assertEqual(len(answered(msgs, i)), 1, "bad=%s" % bad)
            self._assert_invalid_params(msgs, i)

    def test_initialize_with_non_object_params_is_invalid_params(self):
        # Also from the sol review: MCP's InitializeRequest.params is a REQUIRED
        # object. A list was silently coerced to {} and the handshake succeeded.
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 71, "method": "initialize", "params": [1, 2]})])
        self._assert_invalid_params(msgs, 71)

    def test_initialize_with_object_params_still_succeeds(self):
        # Falsifier guard for the pin above.
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 72, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})])
        got = answered(msgs, 72)
        self.assertTrue(got and "result" in got[0], "%r" % (got,))
        self.assertEqual(got[0]["result"]["protocolVersion"], "2025-06-18")

    def test_absent_params_still_works_for_paramless_methods(self):
        # Falsifier guard: the new params guard must NOT reject legitimate
        # params-less requests. If this test ever needs the guard relaxed, the
        # guard is wrong.
        msgs, _ = run_raw([json.dumps({"jsonrpc": "2.0", "id": 56, "method": "tools/list"})])
        got = answered(msgs, 56)
        self.assertTrue(got and "result" in got[0],
                        "tools/list with no params must still succeed: %r" % (got,))

    def test_null_params_still_works(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "id": 57, "method": "tools/list", "params": None})])
        got = answered(msgs, 57)
        self.assertTrue(got and "result" in got[0],
                        "explicit null params must be treated as absent: %r" % (got,))


class TestNotificationsGetNoResponse(unittest.TestCase):
    """Defect 3 — a message with no `id` must never be answered."""

    def _assert_silent(self, msgs, label):
        stray = [m for m in msgs if m.get("id") is None and m.get("jsonrpc")]
        self.assertEqual(stray, [],
                         "%s: a notification must produce NO response, got %r"
                         % (label, stray))

    def test_tools_call_without_id_is_silent(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "method": "tools/call",
             "params": {"name": "list_models", "arguments": {}}})])
        self._assert_silent(msgs, "tools/call notification")

    def test_dropped_tools_call_notification_is_recorded_in_the_log(self):
        # From the fable review: the delivery claimed a red-committed pin for every
        # fix, but nothing asserted the log event -- deleting the _log_event line
        # left the whole suite green. Silence on the wire is required; silence in
        # the log violates this server's LOUD STRUCTURED FAILURE contract.
        msgs, stderr = run_raw([json.dumps(
            {"jsonrpc": "2.0", "method": "tools/call",
             "params": {"name": "list_models", "arguments": {}}})])
        self._assert_silent(msgs, "tools/call notification")
        self.assertIn("tool_call_notification_dropped", stderr)

    def test_dropped_notification_with_bad_budget_is_also_recorded(self):
        # The budget-exception path returned before reaching the drop event, so a
        # notification carrying an unusable timeout vanished from the wire AND the
        # log at once -- the one combination that leaves no trace anywhere.
        msgs, stderr = run_raw([json.dumps(
            {"jsonrpc": "2.0", "method": "tools/call",
             "params": {"name": "llm_generate",
                        "arguments": {"prompt": "x", "timeout_s": 1e309}}})])
        self._assert_silent(msgs, "bad-budget notification")
        self.assertIn("tool_call_bad_budget", stderr)

    def test_ping_without_id_is_silent(self):
        msgs, _ = run_raw([json.dumps({"jsonrpc": "2.0", "method": "ping"})])
        self._assert_silent(msgs, "ping notification")

    def test_initialize_without_id_is_silent(self):
        msgs, _ = run_raw([json.dumps(
            {"jsonrpc": "2.0", "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})])
        self._assert_silent(msgs, "initialize notification")

    def test_tools_list_without_id_is_silent(self):
        msgs, _ = run_raw([json.dumps({"jsonrpc": "2.0", "method": "tools/list"})])
        self._assert_silent(msgs, "tools/list notification")

    def test_server_discover_without_id_is_silent(self):
        msgs, _ = run_raw([json.dumps({"jsonrpc": "2.0", "method": "server/discover"})])
        self._assert_silent(msgs, "server/discover notification")

    def test_malformed_notification_is_still_silent(self):
        # A notification whose params are malformed must not be answered EITHER --
        # the -32602 guard must stay id-aware. The loop must still survive it.
        msgs, _ = run_raw([
            json.dumps({"jsonrpc": "2.0", "method": "tools/call",
                        "params": {"name": "llm_generate", "arguments": "notanobject"}}),
            json.dumps(PING),
        ])
        self._assert_silent(msgs, "malformed notification")
        self.assertTrue(answered(msgs, 99),
                        "read loop must survive a malformed notification")

    def test_explicit_null_id_is_a_bad_request_not_a_notification(self):
        # From the sol review. JSON-RPC 2.0 defines a notification as a request that
        # OMITS the id member. `"id": null` is a request whose id is invalid -- and
        # MCP narrows RequestId to string|number, so null is doubly illegal. It must
        # be refused, not silently swallowed the way an absent id is.
        msgs, _ = run_raw(['{"jsonrpc":"2.0","id":null,"method":"ping"}',
                           json.dumps(PING)])
        errs = [m for m in msgs
                if m.get("id") is None and m.get("error", {}).get("code") == -32600]
        self.assertTrue(errs,
                        "explicit null id must be refused with -32600, got %r" % (msgs,))
        self.assertTrue(answered(msgs, 99), "loop must survive it")

    def test_non_scalar_ids_are_refused_like_null(self):
        # From the fable review: the -32600 message promises "id must be a string
        # or number", but booleans, objects and arrays were accepted and echoed
        # back verbatim. An error message must not describe a rule the code does
        # not enforce.
        for i, bad in enumerate(("true", "false", '{"a":1}', "[1,2]")):
            msgs, _ = run_raw(['{"jsonrpc":"2.0","id":%s,"method":"ping"}' % bad,
                               json.dumps(PING)])
            errs = [m for m in msgs if m.get("error", {}).get("code") == -32600]
            self.assertTrue(errs, "id=%s must be refused, got %r" % (bad, msgs))
            echoed = [m for m in msgs if m.get("id") not in (None, 99)]
            self.assertEqual(echoed, [], "id=%s was echoed back: %r" % (bad, echoed))
            self.assertTrue(answered(msgs, 99), "loop must survive id=%s" % bad)

    def test_string_and_number_ids_still_work(self):
        # Falsifier guard: the two legal id types must keep working, including 0
        # and a negative number, which are falsy but perfectly valid ids.
        for id_ in ('"abc"', "0", "-3", "12"):
            msgs, _ = run_raw(['{"jsonrpc":"2.0","id":%s,"method":"ping"}' % id_])
            self.assertTrue(msgs and "result" in msgs[0],
                            "id=%s must be answered: %r" % (id_, msgs))

    def test_absent_id_stays_a_notification(self):
        # Falsifier guard for the pin above: the fix must distinguish "id member
        # absent" from "id member present and null", not collapse them again.
        msgs, _ = run_raw([json.dumps({"jsonrpc": "2.0", "method": "ping"}),
                           json.dumps(PING)])
        self._assert_silent([m for m in msgs if m.get("id") is None],
                            "absent id")
        self.assertTrue(answered(msgs, 99))

    def test_parse_error_still_answers_with_null_id(self):
        # Falsifier guard: -32700 for an unparseable line is the ONE legitimate
        # null-id response (JSON-RPC 2.0 s5.1) and must NOT be silenced by the
        # notification fix. Pinned green today; it must stay green.
        msgs, _ = run_raw(["{bad json not valid", json.dumps(PING)])
        parse_errs = [m for m in msgs
                      if m.get("id") is None and m.get("error", {}).get("code") == -32700]
        self.assertTrue(parse_errs, "parse error must still be reported with id null")
        self.assertTrue(answered(msgs, 99))


if __name__ == "__main__":
    unittest.main(verbosity=2)
