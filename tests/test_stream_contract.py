# no-opencc
"""Acceptance tests — stream contract module (issue #18 P3, plan R2, WO-5a).

The contract is CODE, not prose: every line of a job's stream.log is either
(a) an Anthropic Messages STREAM EVENT — type in {message_start,
    content_block_start, content_block_delta, content_block_stop,
    message_delta, message_stop} — or
(b) an omni wrapper {"type": "x_omni_<kind>", "backend": <name>, ...}.

Fixtures under tests/fixtures/ are REAL scrubbed backend captures
(regenerate with the commands in their header comments + scrub.py):
  grok_stream.jsonl            grok 1.0.0  streaming-messages-json +partial
  claude_stream_partial.jsonl  claude 2.1.226 stream-json +partial
  claude_stream.jsonl          claude stream-json (no partial — whole msgs)
  codex_stream.jsonl           codex 0.147.0 --json
  pi_stream.jsonl              pi 0.84.2 --mode json (xai/grok-4.6, --no-tools)

Pinned:
  * normalize_lines(backend, lines) consumes EVERY input line — nothing is
    silently dropped: each yields >=1 contract line, unknown shapes come out
    as x_omni_passthrough (never an exception on fixture input);
  * every output line satisfies is_contract_line;
  * grok/claude stream_event envelopes UNWRAP to their inner Messages events;
  * final_text(backend, lines) recovers the assistant's final answer ("PONG")
    from every fixture, including the codex event schema and the no-partial
    claude capture.
"""
import json
import os
import unittest

from providers import stream_contract

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "fixtures")

MESSAGES_EVENT_TYPES = {
    "message_start", "content_block_start", "content_block_delta",
    "content_block_stop", "message_delta", "message_stop",
}

BACKEND_FIXTURES = {
    "grok": "grok_stream.jsonl",
    "claude": "claude_stream_partial.jsonl",
    "codex": "codex_stream.jsonl",
    "pi": "pi_stream.jsonl",
}


def _lines(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


class ContractLineTest(unittest.TestCase):
    def test_messages_events_are_contract_lines(self):
        for t in MESSAGES_EVENT_TYPES:
            self.assertTrue(stream_contract.is_contract_line({"type": t}), t)

    def test_wrappers_are_contract_lines_and_raw_backend_lines_are_not(self):
        self.assertTrue(stream_contract.is_contract_line(
            {"type": "x_omni_passthrough", "backend": "codex", "event": {}}))
        for raw in ({"type": "system"}, {"type": "assistant"},
                    {"type": "result"}, {"type": "thread.started"},
                    {"type": "stream_event"}, {"kind": "no-type"}):
            self.assertFalse(stream_contract.is_contract_line(raw), raw)


class NormalizeFixtureTest(unittest.TestCase):
    def _normalize(self, backend, fixture):
        raw = _lines(fixture)
        out = list(stream_contract.normalize_lines(backend, raw))
        self.assertGreaterEqual(len(out), len(raw),
                                "%s: fewer contract lines than raw lines — "
                                "something was dropped" % backend)
        for line in out:
            obj = json.loads(line) if isinstance(line, str) else line
            self.assertTrue(stream_contract.is_contract_line(obj),
                            "non-contract output for %s: %r" % (backend, obj))
        return [json.loads(l) if isinstance(l, str) else l for l in out]

    def test_grok_fixture_normalizes_and_unwraps(self):
        out = self._normalize("grok", BACKEND_FIXTURES["grok"])
        kinds = {o["type"] for o in out}
        self.assertIn("message_start", kinds)
        self.assertIn("content_block_delta", kinds)
        self.assertIn("message_stop", kinds)

    def test_claude_fixture_normalizes_and_unwraps(self):
        out = self._normalize("claude", BACKEND_FIXTURES["claude"])
        kinds = {o["type"] for o in out}
        self.assertIn("content_block_delta", kinds)

    def test_codex_fixture_maps_or_wraps_every_event(self):
        out = self._normalize("codex", BACKEND_FIXTURES["codex"])
        wrapped = [o for o in out if str(o["type"]).startswith("x_omni_")]
        self.assertTrue(wrapped, "codex unmappable events must be wrapped, "
                                 "not dropped")
        for o in wrapped:
            self.assertEqual(o.get("backend"), "codex")

    def test_garbage_line_becomes_passthrough_not_exception(self):
        out = list(stream_contract.normalize_lines(
            "codex", ["not json at all", '{"type": "mystery"}']))
        self.assertEqual(len(out), 2)
        for line in out:
            obj = json.loads(line) if isinstance(line, str) else line
            self.assertTrue(str(obj["type"]).startswith("x_omni_"))


class FinalTextTest(unittest.TestCase):
    def test_final_text_recovers_pong_from_every_fixture(self):
        for backend, fixture in BACKEND_FIXTURES.items():
            got = stream_contract.final_text(backend, _lines(fixture))
            self.assertEqual(got, "PONG", "%s/%s -> %r"
                             % (backend, fixture, got))

    def test_final_text_claude_whole_messages_no_partial(self):
        got = stream_contract.final_text("claude",
                                         _lines("claude_stream.jsonl"))
        self.assertEqual(got, "PONG")

    def test_final_text_empty_stream_is_empty(self):
        self.assertEqual(stream_contract.final_text("codex", []), "")


class CursorFixtureTest(unittest.TestCase):
    """cursor_agent_stream.jsonl (live 2026-09-08 capture, claude-like shape:
    system/init, user, thinking delta/completed, assistant, result)."""

    def test_cursor_fixture_normalizes_every_line_to_a_contract_line(self):
        raw = _lines("cursor_agent_stream.jsonl")
        out = list(stream_contract.normalize_lines("cursor", raw))
        self.assertGreaterEqual(len(out), len(raw))
        for line in out:
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)

    def test_final_text_recovers_pong_from_cursor_fixture(self):
        got = stream_contract.final_text(
            "cursor", _lines("cursor_agent_stream.jsonl"))
        self.assertEqual(got, "pong")

    def test_final_text_prefers_result_event_over_last_assistant(self):
        # Two assistant events (partial, as a real run streams them) plus a
        # trailing result event whose `result` is their concatenation — the
        # authoritative full text must win, not just the LAST assistant.
        lines = [
            json.dumps({"type": "assistant",
                       "message": {"role": "assistant",
                                   "content": [{"type": "text", "text": "foo"}]}}),
            json.dumps({"type": "assistant",
                       "message": {"role": "assistant",
                                   "content": [{"type": "text", "text": "bar"}]}}),
            json.dumps({"type": "result", "subtype": "success",
                       "result": "foobar", "session_id": "s1"}),
        ]
        got = stream_contract.final_text("cursor", lines)
        self.assertEqual(got, "foobar")
        self.assertNotEqual(got, "bar",
                            "must not fall back to last-assistant when a "
                            "result event is present")


class ForemanCrashPathPins(unittest.TestCase):
    """Truncated-stream and synthesis-completeness pins (crash honesty)."""

    def test_final_text_from_deltas_when_stream_truncated(self):
        # Simulate a crash before the whole assistant/result lines landed:
        # only stream_event deltas survive. final_text must still recover.
        raw = [l for l in _lines(BACKEND_FIXTURES["grok"])
               if json.loads(l).get("type") == "stream_event"]
        self.assertTrue(raw)
        self.assertEqual(stream_contract.final_text("grok", raw), "PONG")

    def test_codex_synthesis_is_a_complete_messages_sequence(self):
        out = [json.loads(l) for l in stream_contract.normalize_lines(
            "codex", _lines(BACKEND_FIXTURES["codex"]))]
        kinds = [o["type"] for o in out if not str(o["type"]).startswith("x_omni_")]
        for required in ("message_start", "content_block_start",
                         "content_block_delta", "content_block_stop",
                         "message_stop"):
            self.assertIn(required, kinds)
        self.assertLessEqual(kinds.index("message_start"),
                             kinds.index("message_stop"))


def _no_constant(name):
    raise ValueError("non-finite constant " + name)


def _no_overflow(text):
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise ValueError("overflowing number " + text)
    return value


def strict_json(text):
    """The parse of a strict JSON consumer: no Infinity, -Infinity, NaN, and
    no number that overflows to infinity."""
    return json.loads(text, parse_constant=_no_constant, parse_float=_no_overflow)


# Raw child text for a number JSON cannot hold, as a child might print it.
NON_FINITE = ("1e309", "-1e309", "NaN", "Infinity", "-Infinity")
NAMES = {"1e309": "Infinity", "-1e309": "-Infinity", "NaN": "NaN",
         "Infinity": "Infinity", "-Infinity": "-Infinity"}

# Where the bad number sits: a top-level event value, a nested value, a key-less
# array, and the shapes each backend passes through, wraps or synthesizes.
SHAPES = {
    "event_value": '{"type":"ping","value":%s}',
    "nested": '{"type":"ping","a":{"b":[1,{"c":%s}],"d":"x"}}',
    "keyless_array": '[1,%s,"y"]',
    "messages_event": ('{"type":"content_block_delta","index":0,'
                       '"delta":{"type":"text_delta","text":"hi"},"w":%s}'),
    "stream_event": '{"type":"stream_event","event":{"type":"ping","v":%s}}',
    "pi_update": ('{"type":"message_update","assistantMessageEvent":'
                  '{"type":"text_delta","contentIndex":0,"delta":"hi","w":%s}}'),
    "codex_event": '{"type":"turn.completed","usage":{"n":%s}}',
    "codex_agent_message": ('{"type":"item.completed","item":'
                            '{"type":"agent_message","text":"ok","w":%s}}'),
}


class StrictJsonOutputTest(unittest.TestCase):
    """Issue #184: a line of a job stream is strict JSON whatever numbers the
    child prints. Python's json.dumps writes Infinity / NaN by default and
    json.loads reads them back, so the round trip hid the problem."""

    def test_every_backend_and_shape_writes_strict_json(self):
        for backend in ("claude", "grok", "pi", "cursor", "codex"):
            for shape, template in SHAPES.items():
                for raw in NON_FINITE:
                    line = template % raw
                    with self.subTest(backend=backend, shape=shape, raw=raw):
                        # The input is what the bug needs: Python reads it.
                        json.loads(line)
                        out = list(stream_contract.normalize_lines(backend, [line]))
                        self.assertTrue(out)
                        for text in out:
                            self.assertIsInstance(strict_json(text), dict)

    def test_a_non_finite_number_is_written_as_its_name_in_place(self):
        # The event keeps its shape; only the number becomes a string.
        for raw, name in NAMES.items():
            with self.subTest(raw=raw):
                out = list(stream_contract.normalize_lines(
                    "claude", ['{"type":"ping","a":{"b":[1,{"c":%s}]}}' % raw]))
                self.assertEqual(len(out), 1)
                self.assertEqual(strict_json(out[0]), {
                    "type": "x_omni_ping", "backend": "claude",
                    "event": {"type": "ping", "a": {"b": [1, {"c": name}]}}})
        out = list(stream_contract.normalize_lines(
            "claude", ['{"type":"content_block_delta","index":0,'
                       '"delta":{"type":"text_delta","text":"hi"},"w":1e309}']))
        event = strict_json(out[0])
        self.assertEqual(event["type"], "content_block_delta")
        self.assertEqual(event["delta"], {"type": "text_delta", "text": "hi"})
        self.assertEqual(event["w"], "Infinity")

    def test_a_key_less_array_is_kept_as_raw_text(self):
        out = list(stream_contract.normalize_lines("claude", ["[1e309,NaN]"]))
        self.assertEqual(strict_json(out[0]), {
            "type": "x_omni_passthrough", "backend": "claude", "raw": "[1e309,NaN]"})

    def test_dumps_never_emits_a_non_finite_constant(self):
        inf, nan = float("inf"), float("nan")
        text = stream_contract._dumps({"a": inf, "b": [-inf, nan, (inf,)], "c": 1.5})
        self.assertEqual(strict_json(text),
                         {"a": "Infinity", "b": ["-Infinity", "NaN", ["Infinity"]],
                          "c": 1.5})
        self.assertIsInstance(strict_json(stream_contract._dumps(inf)), str)

    def test_a_deeply_nested_non_finite_number_is_written_strict(self):
        # json.loads / json.dumps take this depth; a recursive copy of the
        # event ran out of frames at about 498 levels and raised.
        for depth in (498, 600, 900):
            line = '{"type":"ping","v":' + '{"a":' * depth + "1e309" + "}" * depth + "}"
            with self.subTest(depth=depth):
                json.loads(line)
                out = list(stream_contract.normalize_lines("claude", [line]))
                self.assertEqual(len(out), 1)
                node = strict_json(out[0])["event"]["v"]
                for _ in range(depth):
                    node = node["a"]
                self.assertEqual(node, "Infinity")

    def test_nesting_past_the_encoder_is_a_fixed_notice_not_a_raise(self):
        node = float("inf")
        for _ in range(5000):
            node = {"a": node}
        text = stream_contract._dumps({"type": "ping", "v": node})
        self.assertEqual(strict_json(text), {
            "type": "x_omni_omitted", "reason": "nesting_too_deep"})

    def test_a_line_too_deep_to_parse_keeps_nothing_of_the_child(self):
        deep = '{"type":"ping","v":' + '{"a":' * 1500 + '"SENTINEL-TEXT"' + "}" * 1500 + "}"
        with self.assertRaises(RecursionError):
            json.loads(deep)
        self.assertEqual(stream_contract._parse_line(deep), (None, None))
        for backend in ("claude", "grok", "cursor", "pi", "codex", "other"):
            with self.subTest(backend=backend):
                out = list(stream_contract.normalize_lines(
                    backend, [deep, '{"type":"ping","n":1}']))
                self.assertEqual(len(out), 2)
                self.assertEqual(strict_json(out[0]), {
                    "type": "x_omni_omitted", "backend": backend,
                    "reason": "nesting_too_deep"})
                self.assertNotIn("SENTINEL", out[0])
                self.assertIn('"n":1', out[1])
        # The final-text readers skip it and do not raise.
        for backend in ("claude", "grok", "cursor", "pi", "codex"):
            with self.subTest(final_text=backend):
                self.assertEqual(stream_contract.final_text(backend, [deep]), "")

    def test_a_cycle_is_a_fixed_notice_and_does_not_loop(self):
        import signal
        notice = {"type": "x_omni_omitted", "reason": "nesting_too_deep"}
        cyclic_list = [1]
        cyclic_list.append(cyclic_list)
        cyclic_dict = {"type": "ping"}
        cyclic_dict["self"] = {"inner": [cyclic_dict]}
        with_inf = {"a": float("inf"), "c": cyclic_dict}

        def alarm(*_):
            raise AssertionError("_dumps did not return (loop)")
        old = signal.signal(signal.SIGALRM, alarm)
        signal.alarm(10)
        try:
            for obj in (cyclic_list, cyclic_dict, with_inf):
                self.assertEqual(strict_json(stream_contract._dumps(obj)), notice)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def test_shared_acyclic_containers_serialize_unchanged(self):
        shared = {"k": [1, 2], "inf": float("inf")}
        obj = {"a": shared, "b": shared, "c": [shared, shared, [shared]]}
        text = stream_contract._dumps(obj)
        inner = {"k": [1, 2], "inf": "Infinity"}
        self.assertEqual(strict_json(text), {
            "a": inner, "b": inner, "c": [inner, inner, [inner]]})
        finite = {"k": [1, 2]}
        both = {"a": finite, "b": finite}
        self.assertEqual(stream_contract._dumps(both),
                         json.dumps(both, separators=(",", ":")))

    def test_the_strict_parser_refuses_what_json_loads_accepts(self):
        for text in ("Infinity", "-Infinity", "NaN", "1e309", "-1e309",
                     '{"a":[NaN]}', '{"a":{"b":1e999}}'):
            with self.subTest(text=text):
                json.loads(text)
                with self.assertRaises(ValueError):
                    stream_contract.loads_strict(text)
        self.assertEqual(stream_contract.loads_strict('{"a":[1,1.5,1e308,-2]}'),
                         {"a": [1, 1.5, 1e308, -2]})

    def test_ordinary_numbers_and_strings_are_written_unchanged(self):
        for value in (0, 1, -2, 1.5, -0.0, 0.1, 1e308, -1e308, 1e-320, 5e-324,
                      2 ** 70, 10 ** 30, True, None, "Infinity", "NaN", "1e309"):
            obj = {"type": "ping", "v": value, "l": [value, {"k": value}]}
            with self.subTest(value=value):
                self.assertEqual(
                    stream_contract._dumps(obj),
                    json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        # The words as strings are not read as numbers and stay strings.
        out = list(stream_contract.normalize_lines(
            "claude", ['{"type":"ping","a":"Infinity","b":"NaN","c":1e308}']))
        self.assertEqual(strict_json(out[0])["event"],
                         {"type": "ping", "a": "Infinity", "b": "NaN", "c": 1e308})

    def test_the_fixtures_are_unchanged_by_the_strict_dump(self):
        for backend, name in BACKEND_FIXTURES.items():
            for line in _lines(name):
                obj, _raw = stream_contract._parse_line(line)
                with self.subTest(backend=backend, name=name):
                    self.assertEqual(
                        stream_contract._dumps(obj),
                        json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
