# no-opencc
"""Issue #170: child output is persisted only after reassembly and masking.

#160 masks the byte stream of the child's stdout. A model that prints a
credential token by token splits it across two JSON delta events, with JSON
wrapping between the halves, so the byte layer never sees the whole value.
The job lane therefore holds each content block until it is complete, masks
the assembled value, and only then writes the contract ``stream.log``;
``stream.raw.log`` (the in-run IPC file between the child and the supervisor)
is served only while the job runs and is deleted once it reaches a terminal
state.

Hermetic: the "vendor" is a script of this file with a synthetic credential
built at runtime; nothing leaves the machine and none of it is a real key.
"""
import contextlib
import io
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-170-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP

import omni_cli  # noqa: E402
from tests.test_stream_contract import NAMES, NON_FINITE, SHAPES, strict_json  # noqa: E402
from providers import job_runner, jobs, known_secrets, stream_contract  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# No key-shaped prefix: only the known-value layer can catch it. 32 characters.
SECRET = "Zq8vK2mX" + "p5Lw9RtY" + "c3Hn7BdF" + "j6GsA4eU"
CRED_VAR = "OMNI_TEST_STREAM_TOKEN"
FRAGMENT = 8

CONTROL = "hello world, plain control text"


def forms_of(secret):
    """The value and every form it is registered in (escaped, URL, base64...)."""
    return {secret} | set(known_secrets._variants(secret))


def leaked(blob, secret=SECRET, size=FRAGMENT):
    """First ``size``-char slice of any form of ``secret`` found in ``blob``."""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    for form in sorted(forms_of(secret)):
        for i in range(len(form) - size + 1):
            if form[i:i + size] in blob:
                return form[i:i + size]
    return None


def delta_texts(stream_log):
    """{(index, delta type): joined text} over the delta events of a stream.log."""
    out = {}
    for line in stream_log.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") != "content_block_delta":
            continue
        delta = obj["delta"]
        field = {"text_delta": "text", "thinking_delta": "thinking",
                 "input_json_delta": "partial_json"}[delta["type"]]
        key = (obj["index"], delta["type"])
        out[key] = out.get(key, "") + delta[field]
    return out


def stream_event(inner):
    return json.dumps({"type": "stream_event", "event": inner}) + "\n"


def raw_lines(*inners):
    return "".join(stream_event(inner) for inner in inners).encode("utf-8")


def text_delta(index, text):
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": text}}


def thinking_delta(index, text):
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "thinking_delta", "thinking": text}}


def block_start(index, kind="text"):
    return {"type": "content_block_start", "index": index,
            "content_block": {"type": kind, kind: ""}}


def block_stop(index):
    return {"type": "content_block_stop", "index": index}


FAKE_SPLIT_GROK = r'''#!/usr/bin/env python3
import json, os, sys, time
if not sys.stdin.isatty():
    sys.stdin.read()
SECRET = %(secret)r
CONTROL = %(control)r
HANG = os.environ.get("FAKE_HANG") == "1"

def ev(inner):
    sys.stdout.write(json.dumps({"type": "stream_event", "event": inner}) + "\n")
    sys.stdout.flush()

def delta(index, kind, field, text):
    ev({"type": "content_block_delta", "index": index,
        "delta": {"type": kind, field: text}})

ev({"type": "message_start"})
ev({"type": "content_block_start", "index": 0,
    "content_block": {"type": "text", "text": ""}})
delta(0, "text_delta", "text", "key is " + SECRET[:13])
if HANG:
    time.sleep(60)
delta(0, "text_delta", "text", SECRET[13:] + " end")
ev({"type": "content_block_stop", "index": 0})
ev({"type": "content_block_start", "index": 1,
    "content_block": {"type": "thinking", "thinking": ""}})
delta(1, "thinking_delta", "thinking", "mull " + SECRET[:9])
delta(1, "thinking_delta", "thinking", SECRET[9:])
ev({"type": "content_block_stop", "index": 1})
ev({"type": "content_block_start", "index": 2,
    "content_block": {"type": "text", "text": ""}})
delta(2, "text_delta", "text", CONTROL[:9])
delta(2, "text_delta", "text", CONTROL[9:])
ev({"type": "content_block_stop", "index": 2})
if os.environ.get("FAKE_SECOND") == "1":
    ev({"type": "content_block_start", "index": 3,
        "content_block": {"type": "text", "text": ""}})
    delta(3, "text_delta", "text", SECRET[:13])
    delta(3, "text_delta", "text", SECRET[13:])
    ev({"type": "content_block_stop", "index": 3})
if os.environ.get("FAKE_SHORTS") == "1":
    ev({"type": "content_block_start", "index": 6,
        "content_block": {"type": "text", "text": ""}})
    delta(6, "text_delta", "text", "key " + SECRET[:24])
    delta(6, "text_delta", "text", SECRET[24:] + " end")
    ev({"type": "content_block_stop", "index": 6})
if os.environ.get("FAKE_SNAP") == "1":
    # A snapshot after 24 characters of a block whose list position (0) is not
    # the stream index (5): a claude partial snapshot is a block-local envelope.
    ev({"type": "content_block_start", "index": 5,
        "content_block": {"type": "text", "text": ""}})
    delta(5, "text_delta", "text", "key is " + SECRET[:24])
    sys.stdout.write(json.dumps({"type": "assistant", "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "key is " + SECRET[:24]}]}}) + "\n")
    sys.stdout.flush()
    delta(5, "text_delta", "text", SECRET[24:] + " end")
    ev({"type": "content_block_stop", "index": 5})
if os.environ.get("FAKE_NONFINITE") == "1":
    # Numbers JSON cannot hold, as text a child might print.
    for raw in ("1e309", "-1e309", "NaN"):
        sys.stdout.write('{"type":"ping","value":' + raw + ',"n":{"v":[' + raw + ']}}' + chr(10))
    sys.stdout.write('[1e309,NaN]' + chr(10))
    sys.stdout.write('{"type":"stream_event","event":{"type":"content_block_delta",'
                     '"index":9,"delta":{"type":"text_delta","text":"t"},"w":1e309}}' + chr(10))
    sys.stdout.flush()
BLOCKS = os.environ.get("FAKE_BLOCKS") == "1"
if BLOCKS:
    ev({"type": "content_block_start", "index": 4,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                          "input": {}}})
    for a, b in ((0, 10), (10, 21), (21, len(SECRET))):
        delta(4, "input_json_delta", "partial_json",
              ('{"command": "echo ' if a == 0 else "") + SECRET[a:b]
              + ('"}' if b == len(SECRET) else ""))
    ev({"type": "content_block_stop", "index": 4})
ev({"type": "message_stop"})
content = [{"type": "text", "text": "ANSWER " + SECRET}]
if BLOCKS:
    content = [{"type": "tool_use", "id": "toolu_1", "name": "Bash",
                "input": {"command": "echo " + SECRET}}]
sys.stdout.write(json.dumps({"type": "assistant", "message": {
    "role": "assistant", "content": content}}) + "\n")
sys.stdout.flush()
''' % {"secret": SECRET, "control": CONTROL}


# Environment variables the fake CLI reads to pick its scenario.
FAKE_SWITCHES = ("FAKE_HANG", "FAKE_SECOND", "FAKE_SHORTS", "FAKE_SNAP", "FAKE_BLOCKS",
                 "FAKE_NONFINITE")


def _wait_terminal(job_dir, timeout=40):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = jobs.status(job_dir)["state"]
        if state in ("done", "failed", "stopped"):
            return state
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _parser_gives_up(text):
    """True when json.loads runs out of frames on *text* on this interpreter."""
    try:
        json.loads(text)
    except RecursionError:
        return True
    except ValueError:
        return False
    return False


_PROBE_CAP = 1 << 19
_PAST_PARSER = {}


def _depth_past_parser(opener="[", closer="]"):
    """A nesting depth at which json.loads raises RecursionError on this
    interpreter, or None when it never does up to _PROBE_CAP. The limit is
    ~1000 on 3.9-3.11 and far higher (and stack-dependent) on 3.12+, so a
    fixed depth cannot stand in for "deeper than the parser can read"."""
    key = (opener, closer)
    if key not in _PAST_PARSER:
        found = None
        depth = 1000
        while depth <= _PROBE_CAP:
            if _parser_gives_up(opener * depth + "1" + closer * depth):
                found = depth
                break
            depth *= 2
        _PAST_PARSER[key] = found
    return _PAST_PARSER[key]


class _Base(unittest.TestCase):
    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        known_secrets.clear()
        self.addCleanup(known_secrets.clear)
        self.root = tempfile.mkdtemp(prefix="omni-170-test-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


def _fixture_obj(name, kind):
    """First object of a tests/fixtures stream for which kind(obj) is true."""
    path = os.path.join(_REPO, "tests", "fixtures", name)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if kind(obj):
                return obj
    raise AssertionError("no matching line in %s" % name)


def _pi_line(kind, index=0, **fields):
    obj = _fixture_obj("pi_stream.jsonl", lambda o: (
        o.get("type") == "message_update"
        and o["assistantMessageEvent"].get("type") == "thinking_delta"))
    obj["assistantMessageEvent"] = dict(fields, type=kind, contentIndex=index)
    return json.dumps(obj) + "\n"


def _cursor_thinking(text, subtype="delta"):
    obj = _fixture_obj("cursor_agent_stream.jsonl", lambda o: (
        o.get("type") == "thinking" and o.get("subtype") == "delta"))
    obj["subtype"] = subtype
    if subtype == "delta":
        obj["text"] = text
    else:
        obj.pop("text", None)
    return json.dumps(obj) + "\n"


def _lines(stream_log):
    return [json.loads(l) for l in stream_log.splitlines() if l.strip()]


def _joined_texts(stream_log):
    """Joined delta text per (shape, index, kind), across the delta shapes."""
    texts = {}
    for obj in _lines(stream_log):
        event = obj.get("event") if isinstance(obj.get("event"), dict) else {}
        ame = event.get("assistantMessageEvent") or {}
        if obj.get("type") == "content_block_delta":
            delta = obj["delta"]
            field = {"text_delta": "text", "thinking_delta": "thinking",
                     "input_json_delta": "partial_json"}.get(delta["type"])
            if field:
                key = ("block", obj["index"], delta["type"])
                texts[key] = texts.get(key, "") + delta[field]
        elif obj.get("type") == "x_omni_message_update" and "delta" in ame:
            key = ("pi", ame["contentIndex"], ame["type"])
            texts[key] = texts.get(key, "") + ame["delta"]
        elif obj.get("type") == "x_omni_thinking" and "text" in event:
            texts[("cursor",)] = texts.get(("cursor",), "") + event["text"]
    return texts


class _NormalizerCase(_Base):
    def _normalizer(self, backend="claude", register=True):
        job_dir = tempfile.mkdtemp(dir=self.root)
        if register:
            known_secrets.register(SECRET, authoritative=True)
        norm = job_runner._StreamNormalizer(job_dir, backend)
        return norm

    @staticmethod
    def _write(norm, data):
        with open(norm.raw_path, "ab") as handle:
            handle.write(data.encode("utf-8") if isinstance(data, str) else data)
        norm.poll_once()

    def _run(self, data, backend="claude", register=True, final=True):
        norm = self._normalizer(backend, register)
        self._write(norm, data)
        if final:
            norm.final_drain()
        return _read(norm.out_path)


class BlockMaskingTests(_NormalizerCase):
    """A content block is persisted once, complete, after mask_known on the
    whole assembled value."""

    def test_secret_split_across_deltas_inside_one_block_is_masked(self):
        out = self._run(raw_lines(
            block_start(0),
            text_delta(0, "key is " + SECRET[:13]),
            text_delta(0, SECRET[13:] + " end"), block_stop(0)))
        self.assertIsNone(leaked(out), out)
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "key is <redacted> end")
        # One assembled delta per block.
        kinds = [o["type"] for o in _lines(out)]
        self.assertEqual(kinds, ["content_block_start", "content_block_delta",
                                 "content_block_stop"])

    def test_one_character_deltas_and_the_other_delta_kinds(self):
        out = self._run(raw_lines(
            block_start(0), *[text_delta(0, ch) for ch in "a " + SECRET + " z"],
            block_stop(0),
            block_start(1, "thinking"), thinking_delta(1, "mull " + SECRET[:9]),
            thinking_delta(1, SECRET[9:]), block_stop(1),
            {"type": "content_block_delta", "index": 2, "delta": {
                "type": "input_json_delta", "partial_json": '{"cmd":"echo ' + SECRET[:11]}},
            {"type": "content_block_delta", "index": 2, "delta": {
                "type": "input_json_delta", "partial_json": SECRET[11:] + '"}'}},
            block_stop(2)))
        self.assertIsNone(leaked(out), out)
        texts = delta_texts(out)
        self.assertEqual(texts[(0, "text_delta")], "a <redacted> z")
        self.assertEqual(texts[(1, "thinking_delta")], "mull <redacted>")
        self.assertEqual(texts[(2, "input_json_delta")], '{"cmd":"echo <redacted>"}')

    def test_nothing_of_an_open_block_is_persisted_before_its_stop(self):
        norm = self._normalizer()
        self._write(norm, raw_lines(
            {"type": "message_start"}, block_start(0), text_delta(0, "ok " + SECRET[:3])))
        out = _read(norm.out_path)
        self.assertEqual([o["type"] for o in _lines(out)], ["message_start"])
        self._write(norm, raw_lines(block_stop(0)))
        out = _read(norm.out_path)
        self.assertEqual([o["type"] for o in _lines(out)],
                         ["message_start", "content_block_start",
                          "content_block_delta", "content_block_stop"])
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "ok " + SECRET[:3])

    def test_non_secret_text_split_across_deltas_is_reproduced_exactly(self):
        out = self._run(raw_lines(
            block_start(2), text_delta(2, CONTROL[:9]), text_delta(2, CONTROL[9:]),
            block_stop(2)))
        self.assertEqual(delta_texts(out)[(2, "text_delta")], CONTROL)

    def test_text_after_a_redaction_marker_is_judged_as_a_value_start(self):
        # A shorter registered value was replaced by the byte masker inside the
        # longer one; the rest of the longer one follows the marker whole.
        known_secrets.register(SECRET[:13], authoritative=True)
        out = self._run(raw_lines(
            block_start(0), text_delta(0, "x <redacted>"),
            text_delta(0, SECRET[13:] + " end"), block_stop(0)))
        self.assertIsNone(leaked(out), out)
        self.assertEqual(delta_texts(out)[(0, "text_delta")],
                         "x <redacted><redacted> end")

    def test_a_cut_off_front_at_the_end_of_a_block_is_masked(self):
        out = self._run(raw_lines(
            block_start(0), text_delta(0, "tail " + SECRET[:15]), block_stop(0)))
        self.assertIsNone(leaked(out), out)
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "tail <redacted>")

    def test_partial_snapshot_between_blocks_is_not_duplicated_or_unmasked(self):
        def snapshot(*texts):
            return json.dumps({"type": "assistant", "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": t} for t in texts]}}) + "\n"
        data = (raw_lines({"type": "message_start"}, block_start(0),
                          text_delta(0, "a " + SECRET[:13]))
                + snapshot("a " + SECRET[:13]).encode()       # partial, block 0 open
                + raw_lines(text_delta(0, SECRET[13:] + " b"), block_stop(0))
                + snapshot("a " + SECRET + " b").encode()     # completed block
                + raw_lines({"type": "message_stop"}))
        out = self._run(data)
        self.assertIsNone(leaked(out), out)
        objs = _lines(out)
        snaps = [o for o in objs if o["type"] == "x_omni_assistant"]
        # The in-progress entry is gone from the first snapshot ...
        self.assertEqual(snaps[0]["event"]["message"]["content"], [])
        # ... and the completed one is persisted masked.
        self.assertEqual(snaps[1]["event"]["message"]["content"][0]["text"],
                         "a <redacted> b")
        # Order is kept: the block comes out before the snapshot that waited.
        kinds = [o["type"] for o in objs]
        self.assertLess(kinds.index("content_block_stop"), kinds.index("x_omni_assistant"))
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "a <redacted> b")

    def test_snapshot_never_keeps_content_of_an_open_block_whatever_its_position(self):
        # The real fixture shape: the snapshot's content has one entry at
        # position 0 while the open block is stream index 1. Both the value and
        # its first 13 characters are registered, so the byte layer has already
        # replaced the head and left 11 characters of the rest in the entry.
        known_secrets.register(SECRET[:13], authoritative=True)
        snapshot = _fixture_obj("claude_stream_partial.jsonl",
                                lambda o: o.get("type") == "assistant")
        snapshot["message"]["content"] = [
            {"type": "text", "text": "key is <redacted>" + SECRET[13:24]}]
        data = (raw_lines({"type": "message_start"}, block_start(0), block_stop(0),
                          block_start(1),
                          text_delta(1, "key is <redacted>" + SECRET[13:24]))
                + (json.dumps(snapshot) + "\n").encode()
                + raw_lines(text_delta(1, SECRET[24:] + " end"), block_stop(1)))
        out = self._run(data)
        self.assertIsNone(leaked(out), out)
        snaps = [o for o in _lines(out) if o["type"] == "x_omni_assistant"]
        self.assertEqual(snaps[0]["event"]["message"]["content"], [])

    def test_block_scoped_events_that_are_not_assembled_keep_their_order(self):
        sig = {"type": "content_block_delta", "index": 0,
               "delta": {"type": "signature_delta", "signature": "AAAA"}}
        out = self._run(raw_lines(
            block_start(0, "thinking"), thinking_delta(0, "mull " + SECRET[:9]),
            sig, thinking_delta(0, SECRET[9:]), block_stop(0)))
        self.assertIsNone(leaked(out), out)
        self.assertEqual(
            [(o["type"], o.get("delta", {}).get("type")) for o in _lines(out)],
            [("content_block_start", None), ("content_block_delta", "thinking_delta"),
             ("content_block_delta", "signature_delta"), ("content_block_stop", None)])

    def test_the_unchanged_claude_fixture_keeps_its_order_with_masking_active(self):
        with open(os.path.join(_REPO, "tests", "fixtures",
                               "claude_stream_partial.jsonl"), "rb") as handle:
            data = handle.read()
        expect = [json.loads(l) for l in stream_contract.normalize_lines(
            "claude", data.decode().splitlines())]

        def shape(objs):
            out = []
            for o in objs:
                item = (o["type"], (o.get("delta") or {}).get("type"))
                if out and out[-1] == item and item[1] in (
                        "text_delta", "thinking_delta", "input_json_delta"):
                    continue    # assembled: one delta per kind
                out.append(item)
            return out
        out = self._run(data)

        def blocks_only(items):
            # A snapshot that arrives inside a block waits behind it; every
            # block-scoped event keeps its place.
            return [i for i in items if i[0] != "x_omni_assistant"]
        self.assertEqual(blocks_only(shape(_lines(out))), blocks_only(shape(expect)))
        self.assertIn(("content_block_delta", "signature_delta"), shape(_lines(out)))

    def test_events_of_unexpected_field_types_are_ordinary_masked_lines(self):
        odd = [
            {"type": "content_block_delta", "index": 0, "delta": {"type": []}},
            {"type": "content_block_delta", "index": [], "delta": {"type": {}}},
            {"type": []},
            {"type": "content_block_start", "index": {}, "content_block": []},
            {"type": "x_omni_message_update", "event": {"assistantMessageEvent": {
                "type": [], "contentIndex": {}, "delta": SECRET}}},
            {"type": "ping", "note": "n " + SECRET},
        ]
        data = (raw_lines(block_start(0), text_delta(0, "a"), block_stop(0))
                + "".join(json.dumps(o) + "\n" for o in odd).encode()
                + raw_lines(block_start(1), text_delta(1, "b " + SECRET), block_stop(1)))
        out = self._run(data)
        self.assertIsNone(leaked(out), out)
        texts = {}
        for obj in _lines(out):
            delta = obj.get("delta")
            if (obj.get("type") == "content_block_delta" and isinstance(delta, dict)
                    and delta.get("type") == "text_delta"):
                texts[obj["index"]] = delta["text"]
        self.assertEqual(texts[0], "a")
        self.assertEqual(texts[1], "b <redacted>")
        self.assertIn('"x_omni_ping"', out)

    # Short registered values next to the long one: the byte layer replaces them
    # inside each delta and leaves the interior of the long value between markers.
    PIECE_A = "key <redacted>" + SECRET[8:24]
    PIECE_B = SECRET[24:-8] + "<redacted> end"

    def _register_shorts(self):
        known_secrets.register(SECRET[:8], authoritative=True)
        known_secrets.register(SECRET[-8:], authoritative=True)

    def test_interior_between_redaction_markers_is_masked_for_every_backend(self):
        self._register_shorts()
        cases = [
            ("claude", raw_lines(block_start(0), text_delta(0, self.PIECE_A),
                                 text_delta(0, self.PIECE_B), block_stop(0))),
            ("grok", raw_lines(block_start(0, "thinking"),
                               thinking_delta(0, self.PIECE_A),
                               thinking_delta(0, self.PIECE_B), block_stop(0))),
            ("pi", (_pi_line("thinking_start") + _pi_line("thinking_delta", delta=self.PIECE_A)
                    + _pi_line("thinking_delta", delta=self.PIECE_B)
                    + _pi_line("thinking_end", content="x")).encode()),
            ("cursor", (_cursor_thinking(self.PIECE_A) + _cursor_thinking(self.PIECE_B)
                        + _cursor_thinking(None, "completed")).encode()),
        ]
        for backend, data in cases:
            with self.subTest(backend=backend):
                out = self._run(data, backend)
                self.assertIsNone(leaked(out), out)
                self.assertIn("end", out)

    def test_a_middle_slice_is_masked_in_every_persisted_string(self):
        middle = SECRET[10:22]
        start = {"type": "content_block_start", "index": 0, "content_block": {
            "type": "text", "text": "x " + middle + " y"}}
        thinking = {"type": "content_block_start", "index": 1, "content_block": {
            "type": "thinking", "thinking": middle}}
        tool = {"type": "content_block_start", "index": 2, "content_block": {
            "type": "tool_use", "id": "t", "name": "Bash", "input": {"c": "e " + middle}}}
        lines = (raw_lines(start, block_stop(0), thinking, block_stop(1), tool, block_stop(2))
                 + (json.dumps({"type": "assistant", "message": {
                     "role": "assistant", "content": "z " + middle}}) + "\n").encode()
                 + (json.dumps({"type": "result", "result": middle}) + "\n").encode())
        out = self._run(lines)
        self.assertIsNone(leaked(out), out)
        self.assertNotIn(middle, out)

    def test_mask_fragments_only_adds_masking_for_runs_of_eight_or_more(self):
        known_secrets.register(SECRET, authoritative=True)
        self.assertEqual(known_secrets.mask_fragments("a " + SECRET[3:10] + " b"),
                         "a " + SECRET[3:10] + " b")           # 7 characters
        self.assertEqual(known_secrets.mask_fragments("a " + SECRET[3:11] + " b"),
                         "a <redacted> b")                      # 8 characters
        self.assertEqual(known_secrets.mask_fragments(SECRET[:9] + "!" + SECRET[20:30]),
                         "<redacted>!<redacted>")
        self.assertEqual(known_secrets.mask_fragments(CONTROL), CONTROL)
        known_secrets.clear()
        self.assertEqual(known_secrets.mask_fragments("a " + SECRET[3:11]), "a " + SECRET[3:11])

    def test_ordinary_text_with_no_eight_character_overlap_is_untouched(self):
        text = "The quick brown fox jumps over the lazy dog; 0123456789 and Zq8vK2 plus p5Lw9."
        out = self._run(raw_lines(block_start(0), text_delta(0, text), block_stop(0))
                        + (json.dumps({"type": "ping", "note": text}) + "\n").encode())
        self.assertEqual(delta_texts(out)[(0, "text_delta")], text)
        self.assertIn(text, [o.get("event", {}).get("note") for o in _lines(out)])

    def test_overflowed_block_still_suppresses_snapshots_until_its_stop(self):
        known_secrets.register(SECRET[:13], authoritative=True)
        snapshot = _fixture_obj("claude_stream_partial.jsonl",
                                lambda o: o.get("type") == "assistant")
        snapshot["message"]["content"] = [
            {"type": "text", "text": "key <redacted>" + SECRET[13:24] + " x"}]
        with mock.patch.object(job_runner, "_MAX_HELD_BYTES", 600):
            out = self._run(
                raw_lines({"type": "message_start"}, block_start(0), block_stop(0),
                          block_start(1), text_delta(1, "a" * 700))
                + (json.dumps(snapshot) + "\n").encode()
                + raw_lines(block_stop(1)))
        self.assertIsNone(leaked(out), out)
        objs = _lines(out)
        self.assertEqual([o["reason"] for o in objs if o["type"] == "x_omni_omitted"],
                         ["held_content_over_limit"])
        snaps = [o for o in objs if o["type"] == "x_omni_assistant"]
        self.assertEqual(snaps[0]["event"]["message"]["content"], [])

    def test_a_malformed_lifecycle_event_opens_no_block_and_stalls_nothing(self):
        norm = self._normalizer()
        malformed = [
            {"type": "content_block_start", "index": {}, "content_block": [],
             "note": "n " + SECRET},
            {"type": "content_block_start", "index": 0, "content_block": "x"},
            {"type": "content_block_start", "index": True, "content_block": {}},
            {"type": "content_block_stop", "index": []},
            {"type": "x_omni_message_update", "event": {"assistantMessageEvent": {
                "type": "text_start", "contentIndex": {}}}},
        ]
        self._write(norm, "".join(json.dumps(o) + "\n" for o in malformed)
                    + json.dumps({"type": "ping", "n": 1}) + "\n")
        out = _read(norm.out_path)           # live: no final drain yet
        kinds = [o["type"] for o in _lines(out)]
        self.assertEqual(kinds.count("content_block_start"), 3)
        self.assertEqual(kinds.count("content_block_stop"), 1)
        self.assertIn("x_omni_ping", kinds)
        self.assertIsNone(leaked(out), out)
        self.assertIn("<redacted>", out)      # the malformed line was kept, masked
        # And a valid block afterwards still works.
        self._write(norm, raw_lines(block_start(0), text_delta(0, "ok"), block_stop(0)))
        self.assertEqual(delta_texts(_read(norm.out_path))[(0, "text_delta")], "ok")

    def test_the_shape_guards_reject_unexpected_field_types_without_raising(self):
        # What the type checks alone prove (the catch-all in the next test is a
        # second, separate guard).
        bad_delta = {"type": "content_block_delta", "index": 0, "delta": {"type": []}}
        self.assertIsNone(job_runner._delta_ref(bad_delta))
        self.assertIsNone(job_runner._delta_ref(
            {"type": "content_block_delta", "index": True,
             "delta": {"type": "text_delta", "text": "x"}}))
        self.assertIsNone(job_runner._lifecycle({"type": []}))
        self.assertIsNone(job_runner._lifecycle({"type": "content_block_stop", "index": {}}))
        self.assertIsNone(job_runner._member_key(
            {"type": "content_block_delta", "index": [], "delta": {}}))

    def test_an_unexpected_failure_while_classifying_is_an_ordinary_masked_line(self):
        norm = self._normalizer()
        with mock.patch.object(job_runner, "_delta_ref", side_effect=RuntimeError("boom")):
            self._write(norm, raw_lines(text_delta(0, "n " + SECRET),
                                        {"type": "ping", "n": 2}))
        out = _read(norm.out_path)
        self.assertIsNone(leaked(out), out)
        kinds = [o["type"] for o in _lines(out)]
        self.assertEqual(kinds, ["content_block_delta", "x_omni_stream_event"])

    def test_tool_use_input_value_is_masked(self):
        tool = {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
        out = self._run(raw_lines(
            {"type": "content_block_start", "index": 1, "content_block": tool},
            {"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": '{"command": "echo ' + SECRET[:7]}},
            {"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": SECRET[7:] + '"}'}},
            block_stop(1)))
        self.assertIsNone(leaked(out), out)
        self.assertEqual(json.loads(delta_texts(out)[(1, "input_json_delta")]),
                         {"command": "echo <redacted>"})

    def test_no_string_leaf_is_skipped_not_even_a_structure_key(self):
        envelope = json.dumps({"type": "assistant", "message": {
            "role": "assistant", "id": SECRET, "model": SECRET, "content": [
                {"type": "tool_use", "id": SECRET, "name": SECRET, "signature": SECRET,
                 "input": {"command": "echo " + SECRET, "type": SECRET, "id": SECRET}}]}}) + "\n"
        system = json.dumps({"type": "system", "subtype": "init", "session_id": SECRET,
                             "tools": [SECRET], "cwd": "/x/" + SECRET}) + "\n"
        out = self._run(envelope + system)
        self.assertIsNone(leaked(out), out)
        self.assertNotIn(SECRET, out)

    def test_other_lines_get_the_deep_mask_cut_off_edges_included(self):
        # A value cut off at the end (or the start) of a string leaf is masked
        # too: the line-level pass alone only sees whole values.
        line = json.dumps({"type": "system", "subtype": "init",
                           "note": "tail " + SECRET[:15], "head": SECRET[13:] + " x",
                           "tools": [{"input": {"k": "z " + SECRET[:11]}}]}) + "\n"
        out = self._run(line)
        self.assertIsNone(leaked(out), out)

    def test_a_message_edge_completes_the_blocks_still_open(self):
        out = self._run(raw_lines(
            {"type": "message_start"}, block_start(0), text_delta(0, "x " + SECRET),
            {"type": "message_stop"}), final=False)
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "x <redacted>")

    def test_a_stream_that_ends_inside_a_block_leaves_an_omission_notice(self):
        norm = self._normalizer()
        self._write(norm, raw_lines(
            block_start(0), text_delta(0, "half " + SECRET[:13])))
        tail = json.dumps({"type": "result", "result": "R " + SECRET}) + "\n"
        self._write(norm, tail)
        norm.final_drain()
        out = _read(norm.out_path)
        self.assertIsNone(leaked(out), out)
        objs = _lines(out)
        self.assertEqual([o["type"] for o in objs], ["x_omni_omitted", "x_omni_result"])
        self.assertEqual(objs[0]["reason"], "unfinished_block")
        self.assertEqual(objs[0]["blocks"], ["0"])
        self.assertTrue(stream_contract.is_contract_line(objs[0]))
        self.assertNotIn("half", out)

    def test_held_content_is_bounded_and_overflow_drops_it_with_a_notice(self):
        with mock.patch.object(job_runner, "_MAX_HELD_BYTES", 600):
            data = raw_lines(
                block_start(0), text_delta(0, "a" * 200), text_delta(0, SECRET[:13]),
                text_delta(0, "b" * 400), text_delta(0, SECRET[13:]),
                block_stop(0),
                block_start(1), text_delta(1, "fine"), block_stop(1))
            out = self._run(data)
        self.assertIsNone(leaked(out), out)
        objs = _lines(out)
        omitted = [o for o in objs if o["type"] == "x_omni_omitted"]
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]["reason"], "held_content_over_limit")
        self.assertNotIn("a" * 20, out)
        # The dropped block's later events are dropped; the next block is intact.
        self.assertNotIn((0, "text_delta"), delta_texts(out))
        self.assertEqual(delta_texts(out)[(1, "text_delta")], "fine")

    def test_lines_waiting_behind_an_open_block_are_bounded_too(self):
        with mock.patch.object(job_runner, "_MAX_HELD_BYTES", 600):
            filler = [json.dumps({"type": "ping", "pad": "x" * 100}) + "\n" for _ in range(20)]
            out = self._run(raw_lines(block_start(0), text_delta(0, "t"))
                            + "".join(filler).encode())
        kinds = [o["type"] for o in _lines(out)]
        self.assertIn("x_omni_omitted", kinds)
        self.assertLess(kinds.count("x_omni_ping"), 20)

    def test_without_registered_values_lines_pass_through_as_they_arrive(self):
        norm = self._normalizer(register=False)
        data = raw_lines(block_start(0), text_delta(0, "ab"), text_delta(0, "cd"))
        self._write(norm, data)
        expect = "\n".join(stream_contract.normalize_lines(
            "claude", data.decode().splitlines())) + "\n"
        self.assertEqual(_read(norm.out_path), expect)

    def test_codex_whole_message_is_masked(self):
        line = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": "see " + SECRET}}) + "\n"
        out = self._run(line, backend="codex")
        self.assertIsNone(leaked(out), out)
        self.assertEqual(delta_texts(out)[(0, "text_delta")], "see <redacted>")

    def test_the_real_fixtures_keep_their_text_and_shape(self):
        for backend, name in (("grok", "grok_stream.jsonl"),
                              ("claude", "claude_stream_partial.jsonl"),
                              ("pi", "pi_stream.jsonl"),
                              ("cursor", "cursor_agent_stream.jsonl")):
            with open(os.path.join(_REPO, "tests", "fixtures", name), "rb") as handle:
                data = handle.read()
            expect = "\n".join(stream_contract.normalize_lines(
                backend, data.decode().splitlines())) + "\n"
            known_secrets.clear()
            self.assertEqual(self._run(data, backend, register=False), expect, backend)
            known_secrets.clear()
            out = self._run(data, backend, register=True)
            self.assertEqual(_joined_texts(out), _joined_texts(expect), backend)


QUOTED = 'k3Lq"9vXz2Mw'         # a registered value whose JSON form has a backslash


def _nest(depth, leaf, key="a"):
    obj = leaf
    for _ in range(depth):
        obj = {key: obj}
    return obj


class OutputGuaranteeTests(_NormalizerCase):
    """No 8-character window of any registered form of any value is in the
    bytes written to stream.log, whatever path the line took, and every line
    written is JSON."""

    VALUES = (SECRET, QUOTED)

    def _register(self):
        for value in self.VALUES:
            known_secrets.register(value, authoritative=True)

    def _assert_clean(self, out, why=""):
        for value in self.VALUES:
            self.assertIsNone(leaked(out, value), "%s: %s" % (why, out[:400]))
        for line in out.splitlines():
            if line.strip():
                json.loads(line)

    def _through(self, data, backend="claude", cap=None, notice_ok=False):
        """Run *data* through the real append path; check after every poll.

        Unless *notice_ok*, no line may have fallen back to the unmaskable
        notice: masking is what removed the windows, the notice is only the
        last resort.
        """
        self._register()
        norm = self._normalizer(backend, register=False)
        ctx = mock.patch.object(job_runner, "_MAX_HELD_BYTES", cap or 4 * 1024 * 1024)
        with ctx:
            for line in data.splitlines(True):
                self._write(norm, line)
                self._assert_clean(_read(norm.out_path), "after a poll")
            norm.final_drain()
        out = _read(norm.out_path)
        self._assert_clean(out, "after the final drain")
        if not notice_ok:
            self.assertNotIn("unmaskable_line", out)
        return out

    @staticmethod
    def _slice(a=3, b=20):
        return SECRET[a:b]

    def test_a_string_index_that_is_part_of_a_value_never_reaches_a_notice(self):
        key = self._slice(5, 30)
        start = {"type": "content_block_start", "index": key,
                 "content_block": {"type": "text", "text": ""}}
        delta = {"type": "content_block_delta", "index": key,
                 "delta": {"type": "text_delta", "text": "half"}}
        out = self._through(raw_lines(start, delta))     # unfinished at EOF
        self.assertIn("unfinished_block", out)
        out = self._through(raw_lines(start, delta, {"type": "content_block_stop",
                                                     "index": key}))
        self.assertIn("text_delta", out)

    def test_an_overflow_notice_names_no_window(self):
        key = self._slice(2, 28)
        start = {"type": "content_block_start", "index": key,
                 "content_block": {"type": "text", "text": ""}}
        big = {"type": "content_block_delta", "index": key,
               "delta": {"type": "text_delta", "text": "a" * 800}}
        out = self._through(raw_lines(start, big), cap=600)
        self.assertIn("held_content_over_limit", out)

    def test_an_interior_slice_of_a_value_in_a_json_key_is_masked(self):
        key = self._slice(4, 24)
        cases = {
            "claude": json.dumps({"type": "ping", key: 1}) + "\n",
            "grok": json.dumps({"type": "ping", "x": {key: [1, {key: 2}]}}) + "\n",
            "pi": json.dumps({"type": "agent_settled", key: True}) + "\n",
            "cursor": json.dumps({"type": "system", "subtype": "init", key: "x"}) + "\n",
        }
        for backend, data in cases.items():
            with self.subTest(backend=backend):
                self._through(data, backend)

    def test_a_slice_beyond_the_deep_mask_depth_is_masked(self):
        leaf = self._slice(6, 22)
        for depth in (0, 50, 205, 210):
            for backend in ("claude", "grok", "pi", "cursor"):
                with self.subTest(depth=depth, backend=backend):
                    line = json.dumps({"type": "ping", "n": _nest(depth, leaf)}) + "\n"
                    self._through(line, backend)

    def test_a_seven_character_leaf_that_becomes_an_escaped_window_is_masked(self):
        leaf = QUOTED[:7]            # k3Lq"9v: 7 characters, but \" in the JSON text
        self.assertEqual(len(leaf), 7)
        for backend in ("claude", "grok", "pi", "cursor"):
            with self.subTest(backend=backend):
                data = json.dumps({"type": "ping", "note": leaf}) + "\n"
                self._through(data, backend)
        self._through(raw_lines(block_start(0), text_delta(0, leaf), block_stop(0)))

    def test_keys_values_depth_and_block_indexes_for_the_other_shapes(self):
        piece = self._slice(1, 19)
        self._through(_pi_line("thinking_start", index=piece), "pi")
        self._through(_pi_line("thinking_start", index=piece)
                      + _pi_line("thinking_delta", index=piece, delta="d")
                      + _pi_line("thinking_end", index=piece, content="d"), "pi")
        cursor = _cursor_thinking(piece) + _cursor_thinking(None, "completed")
        self._through(cursor, "cursor")

    def test_a_secret_free_fixture_is_written_byte_for_byte(self):
        self._register()
        for backend, name in (("claude", "claude_stream_partial.jsonl"),
                              ("grok", "grok_stream.jsonl"),
                              ("pi", "pi_stream.jsonl"),
                              ("cursor", "cursor_agent_stream.jsonl")):
            with open(os.path.join(_REPO, "tests", "fixtures", name), "rb") as handle:
                data = handle.read().decode()
            norm = self._normalizer(backend, register=False)
            for line in stream_contract.normalize_lines(backend, data.splitlines()):
                self.assertEqual(norm._persistable(line), line, backend)

    def test_a_line_that_cannot_be_masked_becomes_the_fixed_notice(self):
        # A registered value shaped like envelope syntax: masking it breaks JSON.
        known_secrets.register('text_delta","text":"hi"}}', authoritative=True)
        norm = self._normalizer("claude", register=False)
        line = json.dumps({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": "hi"}},
                          separators=(",", ":"))
        self.assertEqual(norm._persistable(line),
                         '{"type":"x_omni_omitted","backend":"claude","reason":"unmaskable_line"}')
        self._write(norm, raw_lines(block_start(0), text_delta(0, "hi"), block_stop(0)))
        out = _read(norm.out_path)
        self.assertIn("unmaskable_line", out)
        self.assertNotIn("hi", out)
        self.assertIsNone(leaked(out, 'text_delta","text":"hi"}}'))
        for text in out.splitlines():
            json.loads(text)

    def test_when_even_the_notice_collides_nothing_is_written_and_the_next_line_is(self):
        known_secrets.register('"type":"', authoritative=True)    # in every event
        norm = self._normalizer("claude", register=False)
        norm._append_contract(['{"type":"a","n":1}', '{"k": 1}'])
        self.assertEqual(_read(norm.out_path), '{"k": 1}\n')

    def test_a_value_made_of_record_framing_is_a_documented_residual(self):
        # The writer's own framing (`}` + newline + the next record's `{"type`)
        # is outside the record the gate scans: such a value, which looks like
        # envelope syntax and is no real credential, matches between two
        # lines. The records themselves are written unchanged and parse.
        framing = '}\n{"type'
        known_secrets.register(framing, authoritative=True)
        norm = self._normalizer("claude", register=False)
        first, second = '{"type":"a","n":1}', '{"type":"a","n":2}'
        norm._append_contract([first, second])
        out = _read(norm.out_path)
        self.assertEqual(out, first + "\n" + second + "\n")
        self.assertEqual([json.loads(line) for line in out.splitlines()],
                         [{"type": "a", "n": 1}, {"type": "a", "n": 2}])
        self.assertIn(framing, out)         # the documented residual

    def test_a_gap_in_the_masking_is_caught_by_the_independent_scan(self):
        # If the masking pass missed something, the check still keeps it out.
        self._register()
        key = self._slice(4, 24)
        norm = self._normalizer("claude", register=False)
        with mock.patch.object(known_secrets, "mask_fragments", lambda text: text):
            self._write(norm, json.dumps({"type": "ping", key: 1}) + "\n")
        out = _read(norm.out_path)
        self._assert_clean(out, "masking disabled")
        self.assertIn("unmaskable_line", out)

    def test_the_notice_carries_nothing_the_child_wrote(self):
        known_secrets.register('text_delta","text":"hi"}}', authoritative=True)
        norm = self._normalizer("pi", register=False)
        line = '{"type":"content_block_delta","index":"CHILD-INDEX","delta":{"type":"text_delta","text":"hi"}}'
        notice = norm._persistable(line)
        self.assertEqual(json.loads(notice), {
            "type": "x_omni_omitted", "backend": "pi", "reason": "unmaskable_line"})

    def test_fuzz_a_random_slice_anywhere_never_reaches_the_file(self):
        import random
        rng = random.Random(170)
        long_secret = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                              for _ in range(48))
        self.VALUES = (long_secret,)
        known_secrets.register(long_secret, authoritative=True)
        for trial in range(200):
            length = rng.randint(8, 40)
            at = rng.randint(0, len(long_secret) - length)
            piece = long_secret[at:at + length]
            depth = rng.choice((0, 50, 210))
            where = rng.choice(("key", "value", "index", "notice", "block-text", "start"))
            if where == "key":
                data = json.dumps({"type": "ping", "x": _nest(depth, 1, piece)}) + "\n"
            elif where == "value":
                data = json.dumps({"type": "ping", "x": _nest(depth, piece)}) + "\n"
            elif where == "index":
                data = raw_lines(
                    {"type": "content_block_start", "index": piece,
                     "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": piece,
                     "delta": {"type": "text_delta", "text": "t"}},
                    {"type": "content_block_stop", "index": piece}).decode()
            elif where == "notice":
                data = raw_lines(
                    {"type": "content_block_start", "index": piece,
                     "content_block": {"type": "text", "text": ""}}).decode()
            elif where == "block-text":
                data = raw_lines(block_start(0), text_delta(0, "a " + piece[:length // 2]),
                                 text_delta(0, piece[length // 2:] + " b"),
                                 block_stop(0)).decode()
            else:
                data = raw_lines({"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": piece}},
                                 block_stop(0)).decode()
            with self.subTest(trial=trial, where=where, depth=depth, length=length):
                norm = self._normalizer("claude", register=False)
                self._write(norm, data)
                norm.final_drain()
                self._assert_clean(_read(norm.out_path), where)
                self.assertNotIn("unmaskable_line", _read(norm.out_path))

    def test_held_bytes_are_a_running_count_that_returns_to_zero(self):
        # 60 blocks of ~150 bytes pass through one after another: more than the
        # cap in all, never more than one block at a time.
        with mock.patch.object(job_runner, "_MAX_HELD_BYTES", 600):
            data = b""
            for i in range(60):
                data += raw_lines(block_start(i), text_delta(i, "x" * 60), block_stop(i))
            out = self._run(data)
        self.assertNotIn("x_omni_omitted", out)
        self.assertEqual(len(delta_texts(out)), 60)

    def test_queued_lines_are_released_from_the_count(self):
        with mock.patch.object(job_runner, "_MAX_HELD_BYTES", 600):
            data = b""
            for i in range(40):
                data += (raw_lines(block_start(i), text_delta(i, "y"))
                         + (json.dumps({"type": "ping", "n": i}) + "\n").encode()
                         + raw_lines(block_stop(i)))
            out = self._run(data)
        self.assertNotIn("x_omni_omitted", out)

    def test_never_closed_blocks_scale_linearly(self):
        known_secrets.register(SECRET, authoritative=True)

        def run(count):
            best = None
            for _ in range(2):
                masker = job_runner._BlockMasker("claude")
                lines = [json.dumps({"type": "content_block_start", "index": i,
                                     "content_block": {"type": "text"}})
                         for i in range(count)]
                began = time.perf_counter()
                masker.apply(lines)
                took = time.perf_counter() - began
                best = took if best is None else min(best, took)
            return best
        t10, t30 = run(10000), run(30000)
        self.assertLess(t30, 4 * t10 + 0.2, (t10, t30))


class StrictJsonStreamLogTests(_NormalizerCase):
    """Issue #184: every line of stream.log parses under strict JSON, with or
    without a known value in play, whatever numbers the child prints."""

    BACKENDS = ("claude", "grok", "pi", "cursor", "codex")

    @staticmethod
    def _strict_lines(out):
        return [strict_json(line) for line in out.splitlines() if line.strip()]

    def test_every_backend_and_shape_is_strict_with_and_without_a_known_value(self):
        for register in (False, True):
            for backend in self.BACKENDS:
                for shape, template in SHAPES.items():
                    for raw in NON_FINITE:
                        data = template % raw + "\n"
                        with self.subTest(register=register, backend=backend,
                                          shape=shape, raw=raw):
                            out = self._run(data, backend, register=register)
                            self.assertTrue(self._strict_lines(out))
                            self.assertNotIn("unmaskable_line", out)

    def test_a_held_block_with_a_non_finite_number_is_written_strict(self):
        marker = "@@NON-FINITE@@"
        events = []
        for event in (block_start(0), text_delta(0, "alpha "),
                      text_delta(0, "beta"), block_stop(0)):
            events.append(dict(event, w=marker))
        data = raw_lines(*events).decode()
        for raw in NON_FINITE:
            tainted = data.replace('"%s"' % marker, raw)
            with self.subTest(raw=raw):
                json.loads(tainted.splitlines()[0])
                out = self._run(tainted, "claude")
                self._strict_lines(out)
                self.assertEqual(delta_texts(out)[(0, "text_delta")], "alpha beta")
                self.assertIn('"w":"%s"' % NAMES[raw], out)
                self.assertNotIn("unmaskable_line", out)

    def test_a_deep_non_finite_event_does_not_stall_the_lines_after_it(self):
        deep = ('{"type":"ping","v":' + '{"a":' * 600 + "1e309" + "}" * 600 + "}")
        for register in (False, True):
            norm = self._normalizer("claude", register=register)
            self._write(norm, deep + "\n")
            self._write(norm, '{"type":"ping","after":1}\n')
            # No final_drain: both lines are already in the file.
            lines = self._strict_lines(_read(norm.out_path))
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["type"], "x_omni_ping")    # not the depth notice
            self.assertEqual(lines[1]["event"], {"type": "ping", "after": 1})
            self.assertNotIn("unmaskable_line", _read(norm.out_path))

    def test_a_line_nested_past_the_parser_is_replaced_and_the_next_line_arrives(self):
        # Built as text, so the strict dump is not what runs out of frames: the
        # line is JSON when it reaches the gate, which cannot verify it.
        for register in (False, True):
            # The depth where each stage runs out of frames moves with the
            # stack the test runs on, so sweep across all of them.
            for depth in tuple(range(900, 1010)) + (1500,):
                deep = '{"type":"ping","v":' + '{"a":' * depth + "1" + "}" * depth + "}"
                with self.subTest(register=register, depth=depth):
                    norm = self._normalizer("claude", register=register)
                    self._write(norm, deep + "\n" + '{"type":"ping","after":1}\n')
                    lines = self._strict_lines(_read(norm.out_path))
                    self.assertEqual(len(lines), 2)
                    self.assertEqual(lines[1]["event"], {"type": "ping", "after": 1})
                    # later chunks are still read
                    self._write(norm, '{"type":"ping","later":2}\n')
                    self.assertEqual(len(self._strict_lines(_read(norm.out_path))), 3)

    def test_a_deep_line_with_an_escaped_credential_leaves_no_trace_in_stream_log(self):
        # The production path: normalize_lines -> block masker -> gate, through
        # poll_once. The credential is only ever \uXXXX escapes, inside a line
        # nested past the parser (so it is never decoded, never masked as text).
        escaped = "".join("\\u%04x" % ord(c) for c in SECRET)
        past = _depth_past_parser("{\"a\":", "}")
        if past is None:
            self.skipTest("json.loads never runs out of frames on this interpreter")
        for depth in sorted({1500, 5000, past}):
            deep = ('{"type":"ping","v":' + '{"a":' * depth + '"' + escaped + '"'
                    + "}" * depth + "}")
            if depth == past:
                # the line really is one this interpreter's parser cannot read
                self.assertTrue(_parser_gives_up(deep))
            for backend in self.BACKENDS + ("unknown-backend",):
                with self.subTest(depth=depth, backend=backend):
                    norm = self._normalizer(backend, register=True)
                    self._write(norm, deep + "\n" + '{"type":"ping","after":1}\n')
                    out = _read(norm.out_path)
                    lines = self._strict_lines(out)
                    self.assertEqual(len(lines), 2)
                    if _parser_gives_up(deep):
                        self.assertEqual(lines[0], {
                            "type": "x_omni_omitted", "backend": backend,
                            "reason": "nesting_too_deep"})
                    else:
                        # 3.12+ reads 1500 and 5000 levels: the line goes
                        # through (masked), not as the depth notice.
                        self.assertIn(lines[0]["type"],
                                      ("x_omni_ping", "x_omni_passthrough"))
                    self.assertIn('"after":1', out)
                    self.assertNotIn("\\u", out)
                    self.assertNotIn(escaped, out)
                    self.assertIsNone(leaked(out))
                    self.assertNotIn(SECRET, out)

    def test_the_gate_cannot_verify_a_line_nested_past_the_parser_so_it_replaces_it(self):
        # Straight at the gate, so the earlier stages cannot be what handles it.
        past = _depth_past_parser("{\"a\":", "}")
        if past is None:
            self.skipTest("json.loads never runs out of frames on this interpreter")
        deep = '{"type":"ping","v":' + '{"a":' * past + "1" + "}" * past + "}"
        after = '{"type":"ping","after":1}'
        for register in (False, True):
            with self.subTest(register=register):
                norm = self._normalizer("claude", register=register)
                self.assertTrue(_parser_gives_up(deep))
                # The block masker treats it as an ordinary text line, no raise.
                self.assertEqual(norm._logical.apply([deep]), [deep])
                norm._append_contract([deep, after])
                lines = self._strict_lines(_read(norm.out_path))
                self.assertEqual(lines, [
                    {"type": "x_omni_omitted", "backend": "claude",
                     "reason": "unmaskable_line"},
                    {"type": "ping", "after": 1}])

    def test_an_exception_in_the_gate_costs_one_line_not_the_stream(self):
        norm = self._normalizer("claude", register=False)
        real = norm._persistable
        calls = []

        def flaky(line):
            calls.append(line)
            if "boom" in line:
                raise RuntimeError("injected")
            return real(line)
        norm._persistable = flaky
        self._write(norm, '{"type":"ping","n":"boom"}\n{"type":"ping","n":2}\n')
        lines = self._strict_lines(_read(norm.out_path))
        self.assertEqual([o.get("reason", o.get("event", {}).get("n")) for o in lines],
                         ["unmaskable_line", 2])

    def test_the_gate_replaces_a_line_that_is_json_to_python_but_not_strict(self):
        notice = '{"type":"x_omni_omitted","backend":"claude","reason":"unmaskable_line"}'
        bad = ('{"type":"ping","v":Infinity}', '{"type":"ping","v":-Infinity}',
               '{"type":"ping","v":NaN}', '{"type":"ping","v":1e309}',
               '{"type":"ping","v":-1e309}', '{"type":"ping","a":{"b":[NaN]}}',
               '[1e309,NaN]')
        for register in (False, True):
            norm = self._normalizer("claude", register=register)
            for line in bad:
                with self.subTest(register=register, line=line):
                    json.loads(line)
                    self.assertEqual(norm._persistable(line), notice)
            good = '{"type":"ping","v":1.5,"w":1e308,"s":"Infinity"}'
            self.assertEqual(norm._persistable(good), good)

    def test_append_contract_never_writes_a_non_strict_line(self):
        for register in (False, True):
            norm = self._normalizer("claude", register=register)
            norm._append_contract(['{"type":"ping","v":Infinity}',
                                   '{"type":"ping","n":1}', '{"type":"p","v":1e309}'])
            lines = self._strict_lines(_read(norm.out_path))
            self.assertEqual(
                [o.get("reason", o.get("n")) for o in lines],
                ["unmaskable_line", 1, "unmaskable_line"])

    def test_a_line_that_is_json_only_to_python_is_still_checked(self):
        # was_json comes from the permissive parse: a strict-only parse there
        # would call this line "not JSON" and skip the check.
        norm = self._normalizer("claude", register=True)
        self.assertTrue(norm._parses('{"v":Infinity}'))
        self.assertFalse(norm._parses('{"v":Infinity}', strict=True))
        self.assertIn("unmaskable_line", norm._persistable('{"v":NaN}'))

    def test_ordinary_numbers_are_written_unchanged(self):
        line = ('{"type":"ping","a":0,"b":-2,"c":1.5,"d":1e308,"e":-0.0,'
                '"f":5e-324,"g":1000000000000000000000,"h":"NaN"}')
        for register in (False, True):
            norm = self._normalizer("claude", register=register)
            self.assertEqual(norm._persistable(line), line)


class EscapedValueStreamLogTests(_NormalizerCase):
    """Issue #191: a registered value written as JSON \\uXXXX escapes (all of
    it or some of it, either hex case) is not persisted in a form that decodes
    back to 8+ characters of it, on the lines the dict path does not decode."""

    BACKENDS = ("claude", "codex")
    NOTICE = "unmaskable_line"
    AFTER = '{"type":"ping","after":1}'

    @staticmethod
    def _esc(text, fmt="\\u%04x"):
        return "".join(fmt % ord(c) for c in text)

    def _forms(self, secret=SECRET):
        """Spellings of *secret* made of escapes: full, partial, either case."""
        half = len(secret) // 2
        return {
            "full-lower": self._esc(secret),
            "full-upper": self._esc(secret, "\\u%04X"),
            "tail": secret[:10] + self._esc(secret[10:]),
            "head": self._esc(secret[:10]) + secret[10:],
            "middle": secret[:6] + self._esc(secret[6:-6], "\\u%04X") + secret[-6:],
            "mixed-case": self._esc(secret[:half]) + self._esc(secret[half:], "\\u%04X"),
            "alternating": "".join(
                self._esc(c) if i % 2 else c for i, c in enumerate(secret)),
        }

    @staticmethod
    def _unescape_once(text):
        """One level of JSON string unescaping, written out character by
        character (an oracle of its own: nothing shared with the writer gate).
        A backslash a decoder keeps stays; surrogate halves that end up side
        by side are joined."""
        simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                  "n": "\n", "r": "\r", "t": "\t"}
        hexdigits = "0123456789abcdefABCDEF"
        out = []
        i = 0
        while i < len(text):
            c = text[i]
            if c == "\\" and i + 1 < len(text):
                n = text[i + 1]
                if n in simple:
                    out.append(simple[n])
                    i += 2
                    continue
                digits = text[i + 2:i + 6]
                if n == "u" and len(digits) == 4 and all(
                        d in hexdigits for d in digits):
                    out.append(chr(int(digits, 16)))
                    i += 6
                    continue
            out.append(c)
            i += 1
        joined = []
        for c in out:
            if (joined and 0xDC00 <= ord(c) <= 0xDFFF
                    and 0xD800 <= ord(joined[-1]) <= 0xDBFF):
                high = ord(joined.pop())
                c = chr(0x10000 + ((high - 0xD800) << 10) + (ord(c) - 0xDC00))
            joined.append(c)
        return "".join(joined)

    @classmethod
    def _strings(cls, value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from cls._strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._strings(item)

    @classmethod
    def _decoded(cls, blob):
        """What a reader that decodes the file sees: every line read as JSON
        when it is JSON (its string values), and each text decoded again, level
        by level, with every level kept."""
        views = []
        for line in blob.splitlines():
            texts = [line]
            try:
                texts += list(cls._strings(json.loads(line)))
            except ValueError:
                pass
            for text in texts:
                for _ in range(12):
                    views.append(text)
                    decoded = cls._unescape_once(text)
                    if decoded == text:
                        break
                    text = decoded
        return "\n".join(views)

    def _shapes(self, spelled):
        return {
            "non-json text": "note: " + spelled,
            "top-level array": '["' + spelled + '"]',
            "top-level string": '"' + spelled + '"',
            "array of values": '[1,"x","' + spelled + '",{"k":2}]',
            "parser rejects": '{"type":"ping","v":"' + spelled + '"} trailing',
        }

    def _assert_gone(self, out, label, secret=SECRET):
        self.assertIsNone(leaked(out, secret), label)
        self.assertIsNone(leaked(self._decoded(out), secret), label)
        self.assertNotIn(secret, self._decoded(out), label)

    def test_escaped_values_are_replaced_on_every_passthrough_shape(self):
        for backend in self.BACKENDS:
            for form, spelled in self._forms().items():
                for shape, line in self._shapes(spelled).items():
                    with self.subTest(backend=backend, form=form, shape=shape):
                        out = self._run(line + "\n" + self.AFTER + "\n", backend)
                        self._assert_gone(out, (backend, form, shape))
                        self.assertIn(self.NOTICE, out)
                        self.assertIn('"after":1', out)     # the stream goes on

    def test_the_gate_replaces_the_text_it_is_handed(self):
        notice = ('{"type":"x_omni_omitted","backend":"claude",'
                  '"reason":"unmaskable_line"}')
        norm = self._normalizer("claude")
        for form, spelled in self._forms().items():
            for shape, line in self._shapes(spelled).items():
                with self.subTest(form=form, shape=shape):
                    self.assertEqual(norm._persistable(line), notice)

    def test_a_line_nested_past_the_parser_is_replaced_by_the_gate(self):
        # What is asserted is what reaches stream.log, not that json.loads
        # raises: 3.12+ reads 1500 and 5000 levels, so the depth the parser
        # gives up at is probed on the running interpreter.
        past = _depth_past_parser()
        if past is None:
            self.skipTest("json.loads never runs out of frames on this interpreter")
        for form in ("full-lower", "full-upper", "tail"):
            spelled = self._forms()[form]
            for depth in sorted({1500, 5000, past}):
                deep = '[' * depth + '"' + spelled + '"' + ']' * depth
                with self.subTest(form=form, depth=depth):
                    if depth == past:
                        self.assertTrue(_parser_gives_up(deep))
                    norm = self._normalizer("claude")
                    norm._append_contract([deep, self.AFTER])
                    out = _read(norm.out_path)
                    self._assert_gone(out, (form, depth))
                    self.assertIn(self.NOTICE, out)
                    self.assertIn('"after":1', out)

    def test_a_line_nested_past_the_parser_through_the_real_path(self):
        past = _depth_past_parser()
        if past is None:
            self.skipTest("json.loads never runs out of frames on this interpreter")
        depth = max(5000, past)
        for backend in self.BACKENDS:
            for form, spelled in self._forms().items():
                deep = '[' * depth + '"' + spelled + '"' + ']' * depth
                with self.subTest(backend=backend, form=form):
                    self.assertTrue(_parser_gives_up(deep))
                    out = self._run(deep + "\n" + self.AFTER + "\n", backend)
                    self._assert_gone(out, (backend, form))
                    self.assertIn('"after":1', out)

    @unittest.skipUnless(hasattr(sys, "get_int_max_str_digits")
                         and sys.get_int_max_str_digits() > 0,
                         "the int digit limit needs Python 3.11+")
    def test_a_line_the_parser_refuses_for_its_digits_is_replaced(self):
        digits = "9" * (sys.get_int_max_str_digits() + 10)
        value = self.BACKSLASH_VALUES[0]
        known_secrets.register(value, authoritative=True)
        cases = [(form, spelled, SECRET) for form, spelled in self._forms().items()]
        cases += [("bs-" + form, spelled, value)
                  for form, spelled in self._backslash_forms(value).items()]
        for backend in self.BACKENDS:
            for form, spelled, secret in cases:
                line = '{"a":' + digits + ',"b":"' + spelled + '"}'
                with self.subTest(backend=backend, form=form):
                    with self.assertRaises(ValueError):
                        json.loads(line)
                    out = self._run(line + "\n" + self.AFTER + "\n", backend)
                    self._assert_gone(out, (backend, form), secret)
                    self.assertIn('"after":1', out)

    def test_surrogate_pairs_and_escapes_of_the_backslash_that_starts_an_escape(self):
        wide = "Zq8vK2mX\U0001F600p5Lw9RtY\U00010437c3Hn7BdF"
        known_secrets.register(wide, authoritative=True)

        def pair(c, fmt):
            v = ord(c) - 0x10000
            return (fmt + fmt) % (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
        spelled = {
            "pairs-lower": "".join(
                pair(c, "\\u%04x") if ord(c) > 0xFFFF else "\\u%04x" % ord(c)
                for c in wide),
            "pairs-upper": "".join(
                pair(c, "\\u%04X") if ord(c) > 0xFFFF else "\\u%04X" % ord(c)
                for c in wide),
            "escaped-escape": "".join("\\u005cu%04x" % ord(c) for c in SECRET),
            "three-levels": "".join("\\u005cu005cu%04x" % ord(c) for c in SECRET),
        }
        for form, text in spelled.items():
            for shape, line in self._shapes(text).items():
                with self.subTest(form=form, shape=shape):
                    out = self._run(line + "\n" + self.AFTER + "\n", "claude")
                    self._assert_gone(out, (form, shape))
                    self._assert_gone(out, (form, shape), wide)
                    self.assertIn(self.NOTICE, out)

    # No run of ASCII reaches 8 characters, so every 8-character window of the
    # value holds an emoji and only the joined pair can show it.
    WIDE_EVERY_WINDOW = "Zq8vK2m\U0001F600p5Lw9Rt\U00010437c3Hn7B\U0001F600"
    BACKSLASH_VALUES = ("Ab3\\Cd4Ef", "ab\\cdefg")

    @staticmethod
    def _spell_backslash(value, backslash, fmt="\\u%04x"):
        """*value* with every character an escape and each backslash written
        as *backslash*."""
        return "".join(backslash if c == "\\" else fmt % ord(c) for c in value)

    def _backslash_forms(self, value):
        """Spellings of a value that holds a backslash, as a decoder keeps it:
        an escaped pair is ONE backslash, so the text holds 2, 4, or an escape."""
        spell = self._spell_backslash
        return {
            "pair": spell(value, "\\\\"),
            "pair-upper": spell(value, "\\\\", "\\u%04X"),
            "u005c": spell(value, "\\u005c"),
            "twice": spell(value, "\\\\\\\\"),
            "pair-after-pair": spell(value, "\\\\\\\\\\\\"),
            "partial": value[:value.index("\\")] + spell(
                value[value.index("\\"):], "\\\\"),
        }

    def _wide_forms(self, wide):
        """Spellings of an emoji value whose surrogate halves are revealed by
        different passes: the high half is escaped once (upper-case hex, which
        is no registered spelling) and the low half sits
        behind one or two more levels (an escaped backslash, or two). Every
        other character is spelled as deep as the low half, so no view before
        the pair is joined shows a window of any registered spelling."""
        def spell(low, depth):
            deep = ("\\u%04x" if depth == 1 else
                    "\\u005c" + "u005c" * (depth - 2) + "u%04x")
            out = []
            for c in wide:
                if ord(c) > 0xFFFF:
                    v = ord(c) - 0x10000
                    out.append("\\u%04X" % (0xD800 + (v >> 10)) + low
                               + "%04x" % (0xDC00 + (v & 0x3FF)))
                else:
                    out.append(deep % ord(c))
            return "".join(out)
        return {
            "high-escaped-low-nested": spell("\\u005cu", 2),
            "high-escaped-low-twice-nested": spell("\\u005cu005cu", 3),
            "high-escaped-low-pair-nested": spell("\\\\u", 2),
        }

    def _all_shapes(self, spelled):
        shapes = self._shapes(spelled)
        shapes["dict value"] = '{"type":"ping","v":"' + spelled + '"}'
        return shapes

    def test_values_that_hold_a_backslash_are_replaced_on_every_shape(self):
        for value in self.BACKSLASH_VALUES:
            known_secrets.register(value, authoritative=True)
        for backend in self.BACKENDS:
            for value in self.BACKSLASH_VALUES:
                for form, spelled in self._backslash_forms(value).items():
                    for shape, line in self._all_shapes(spelled).items():
                        with self.subTest(backend=backend, value=value,
                                          form=form, shape=shape):
                            out = self._run(line + "\n" + self.AFTER + "\n", backend)
                            self._assert_gone(out, (backend, form, shape), value)
                            self.assertIn('"after":1', out)
                            if shape != "dict value":
                                self.assertIn(self.NOTICE, out)

    def test_the_reported_spelling_of_a_value_with_a_backslash(self):
        # \u0041\u0062\u0033 + an escaped backslash pair + the rest as
        # escapes: a JSON decoder reads it as Ab3 + backslashes + Cd4Ef.
        value = "Ab3\\Cd4Ef"
        known_secrets.register(value, authoritative=True)
        for text in (
                r"\u0041\u0062\u0033\\\u0043\u0064\u0034\u0045\u0066",
                r"\u0041\u0062\u0033\\\\\u0043\u0064\u0034\u0045\u0066"):
            for backend in self.BACKENDS:
                for shape, line in self._shapes(text).items():
                    with self.subTest(backend=backend, shape=shape, text=text):
                        out = self._run(line + "\n", backend)
                        self.assertIsNone(
                            leaked(self._decoded(out), value), out)
                        self.assertIsNone(leaked(out, value), out)
                        self.assertIn(self.NOTICE, out)

    def test_surrogate_halves_revealed_by_different_passes_are_joined(self):
        wide = self.WIDE_EVERY_WINDOW
        known_secrets.register(wide, authoritative=True)
        for backend in self.BACKENDS:
            for form, spelled in self._wide_forms(wide).items():
                for shape, line in self._all_shapes(spelled).items():
                    with self.subTest(backend=backend, form=form, shape=shape):
                        out = self._run(line + "\n" + self.AFTER + "\n", backend)
                        self._assert_gone(out, (backend, form, shape), wide)
                        self.assertIn('"after":1', out)

    def test_paired_mutation_without_the_faithful_view_backslash_spellings_are_written(self):
        # The whole-run rule as the only view reads an escaped backslash pair
        # as nothing, which is the leak this round closes.
        for value in self.BACKSLASH_VALUES:
            known_secrets.register(value, authoritative=True)
        with mock.patch.object(job_runner, "_faithful_pass", lambda text: text):
            for value in self.BACKSLASH_VALUES:
                written = 0
                for form, spelled in self._backslash_forms(value).items():
                    if form not in ("pair", "pair-upper", "partial"):
                        continue
                    for shape, line in self._shapes(spelled).items():
                        with self.subTest(value=value, form=form, shape=shape):
                            out = self._run(line + "\n", "claude")
                            self.assertIsNotNone(
                                leaked(self._decoded(out), value))
                            written += 1
                self.assertGreater(written, 0)

    def test_paired_mutation_without_the_surrogate_join_the_emoji_is_written(self):
        wide = self.WIDE_EVERY_WINDOW
        known_secrets.register(wide, authoritative=True)
        spelled = self._wide_forms(wide)["high-escaped-low-nested"]
        with mock.patch.object(job_runner, "_join_surrogates", lambda text: text):
            for shape, line in self._shapes(spelled).items():
                with self.subTest(shape=shape):
                    out = self._run(line + "\n", "claude")
                    self.assertIsNotNone(
                        leaked(self._decoded(out), wide), out)

    def test_a_run_of_backslashes_no_pass_budget_can_read_is_replaced(self):
        # Documented over-inclusion: a text still changing after the last pass
        # cannot be verified, so it is the notice while a value is registered.
        norm = self._normalizer("claude")
        self.assertIn(self.NOTICE, norm._persistable("note " + "\\" * 600))
        self.assertEqual(
            norm._persistable("note " + "\\" * 100), "note " + "\\" * 100)

    def test_a_text_still_escaped_after_the_last_pass_is_replaced(self):
        nested = "\\" + "u005c" * 12 + "u0041"
        norm = self._normalizer("claude")
        self.assertIn(self.NOTICE, norm._persistable("note " + nested))

    def test_escapes_in_a_dict_line_are_masked_not_replaced(self):
        # The dict path decodes first, so a value there is masked as before.
        for form, spelled in self._forms().items():
            line = '{"type":"ping","v":"' + spelled + '"}'
            with self.subTest(form=form):
                out = self._run(line + "\n", "claude")
                self._assert_gone(out, form)
                self.assertNotIn(self.NOTICE, out)
                self.assertIn(known_secrets.MARKER, out)

    def test_a_dict_line_that_prints_the_escapes_as_text_is_replaced(self):
        # The child printed backslash-u text; the line holds it as doubled
        # backslashes, which the view reads as the escapes they imitate.
        line = json.dumps({"type": "ping", "v": self._esc(SECRET)})
        out = self._run(line + "\n", "claude")
        self._assert_gone(out, "text")
        self.assertIn(self.NOTICE, out)

    def test_ordinary_lines_are_written_as_before(self):
        # Nothing about the value is in these lines: the decoded view must not
        # change them, with a value known, whatever escapes they hold.
        lines = [
            '{"type":"ping","t":"caf\\u00e9 \\u4e2d\\u6587 \\ud83d\\ude00 \\u003cb\\u003e"}',
            '{"type":"ping","t":"lone \\ud800 and bad \\uZZZZ and \\u12"}',
            "plain text with \\u0041\\u0042\\u0043 and a path C:\\users\\me",
            '["\\u0041\\u0042\\u0043\\u0044\\u0045\\u0046\\u0047\\u0048\\u0049"]',
            '"\\u3053\\u3093\\u306b\\u3061\\u306f\\u4e16\\u754c\\u3053\\u3093"',
            '{"type":"ping","n":1}',
            '{"type":"ping","t":"two\\nlines, a \\"quote\\", a\\\\b, c\\/d, \\t\\r"}',
            "windows path C:\\\\Users\\\\me\\\\file.txt and \\n \\/ \\\"x\\\"",
        ]
        for backend in self.BACKENDS:
            norm = self._normalizer(backend)
            for line in lines:
                with self.subTest(backend=backend, line=line):
                    self.assertEqual(norm._persistable(line), line)

    def test_the_real_fixtures_are_written_the_same_with_the_gate_on_and_off(self):
        names = {"claude": "claude_stream.jsonl", "codex": "codex_stream.jsonl",
                 "grok": "grok_stream.jsonl", "pi": "pi_stream.jsonl",
                 "cursor": "cursor_agent_stream.jsonl"}
        for backend, name in names.items():
            path = os.path.join(_REPO, "tests", "fixtures", name)
            with open(path, encoding="utf-8") as handle:
                data = handle.read()
            with self.subTest(backend=backend):
                on = self._run(data, backend)
                with mock.patch.object(
                        job_runner, "_decoded_view_hits", return_value=False):
                    off = self._run(data, backend)
                self.assertEqual(on, off)
                self.assertNotIn(self.NOTICE, on)
                self.assertTrue(on.strip())

    def test_the_cost_stays_linear_and_nothing_is_decoded_without_an_escape(self):
        known_secrets.register(SECRET, authoritative=True)
        calls = []
        real = known_secrets.find_fragment

        def counting(text):
            calls.append(len(text))
            return real(text)
        with mock.patch.object(known_secrets, "find_fragment", counting):
            self.assertFalse(job_runner._decoded_view_hits("x" * 2_000_000))
        self.assertEqual(calls, [])             # no backslash-u: no decode, no scan
        started = time.monotonic()
        for text in ("\\u0041" * 200_000, "\\u00" * 300_000,
                     "\\\\u0041" * 150_000, "\\n" * 400_000,
                     "\\\\" * 50 + "x" * 100_000, ("\\\\" * 100 + "x") * 5_000):
            self.assertFalse(job_runner._decoded_view_hits(text))
        # Texts that never settle in the pass budget are answered, not hung on.
        for text in ("\\" * 1_000_000, "\\" * 500_000 + "u12" * 100_000):
            self.assertTrue(job_runner._decoded_view_hits(text))
        self.assertLess(time.monotonic() - started, 20)

    def test_paired_mutation_with_the_gate_off_the_escapes_are_written(self):
        # The same lines as the first test with the decoded view disabled: they
        # reach the file, which is what proves the tests above can fail.
        with mock.patch.object(job_runner, "_decoded_view_hits", return_value=False):
            for backend in self.BACKENDS:
                for form, spelled in self._forms().items():
                    for shape, line in self._shapes(spelled).items():
                        with self.subTest(backend=backend, form=form, shape=shape):
                            out = self._run(line + "\n", backend)
                            self.assertIsNotNone(leaked(self._decoded(out)))


class OtherBackendBlockTests(_NormalizerCase):
    def test_pi_thinking_block_is_one_masked_value_and_appears_once(self):
        data = (_pi_line("thinking_start")
                + _pi_line("thinking_delta", delta="mull " + SECRET[:16])
                + _pi_line("thinking_delta", delta=SECRET[16:] + " done")
                + _pi_line("thinking_end", content="mull " + SECRET + " done"))
        out = self._run(data, "pi")
        self.assertIsNone(leaked(out), out)
        self.assertEqual(_joined_texts(out)[("pi", 0, "thinking_delta")],
                         "mull <redacted> done")
        # The block appears once: the end event's copy is dropped.
        self.assertEqual(out.count("mull"), 1)
        end = [o for o in _lines(out) if o["type"] == "x_omni_message_update"
               and o["event"]["assistantMessageEvent"]["type"] == "thinking_end"]
        self.assertEqual(len(end), 1)
        self.assertNotIn("content", end[0]["event"]["assistantMessageEvent"])

    def test_pi_block_end_without_deltas_keeps_its_content_masked(self):
        out = self._run(_pi_line("text_end", index=1, content="t " + SECRET), "pi")
        self.assertIsNone(leaked(out), out)
        self.assertIn("t <redacted>", out)

    def test_pi_text_and_tool_call_deltas_and_envelopes(self):
        text = []
        for piece in ("t " + SECRET[:16], SECRET[16:] + " e"):
            obj = _fixture_obj("pi_stream.jsonl", lambda o: (
                o.get("type") == "message_update"
                and o["assistantMessageEvent"].get("type") == "text_delta"))
            obj["assistantMessageEvent"].update(delta=piece, contentIndex=1)
            text.append(json.dumps(obj) + "\n")
        envelope = _fixture_obj("pi_stream.jsonl", lambda o: o.get("type") == "message_end")
        envelope["message"]["content"] = [{"type": "text", "text": "m " + SECRET}]
        data = ("".join(text) + _pi_line("text_end", index=1, content="t " + SECRET + " e")
                + _pi_line("toolcall_delta", index=2, delta='{"a":"' + SECRET[:16])
                + _pi_line("toolcall_delta", index=2, delta=SECRET[16:] + '"}')
                + _pi_line("toolcall_end", index=2, content="x")
                + json.dumps(envelope) + "\n")
        out = self._run(data, "pi")
        self.assertIsNone(leaked(out), out)
        texts = _joined_texts(out)
        self.assertEqual(texts[("block", 1, "text_delta")], "t <redacted> e")
        self.assertEqual(texts[("pi", 2, "toolcall_delta")], '{"a":"<redacted>"}')

    def test_cursor_thinking_block_is_one_masked_value(self):
        out = self._run(_cursor_thinking("mull " + SECRET[:16])
                        + _cursor_thinking(SECRET[16:] + " done")
                        + _cursor_thinking(None, "completed"), "cursor")
        self.assertIsNone(leaked(out), out)
        self.assertEqual(_joined_texts(out)[("cursor",)], "mull <redacted> done")
        self.assertEqual(len(_lines(out)), 2)       # one delta, the completed event

    def test_cursor_and_grok_envelopes_and_results_are_masked(self):
        grok = _fixture_obj("grok_stream.jsonl", lambda o: o.get("type") == "assistant")
        grok["message"]["content"] = [{"type": "text", "text": "e " + SECRET}]
        cursor = _fixture_obj("cursor_agent_stream.jsonl", lambda o: o.get("type") == "result")
        cursor["result"] = "r " + SECRET
        self.assertIsNone(leaked(self._run(json.dumps(grok) + "\n", "grok")))
        self.assertIsNone(leaked(self._run(json.dumps(cursor) + "\n", "cursor")))


class _JobCase(_Base):
    """Run the fake vendor as a real detached job."""

    def setUp(self):
        super().setUp()
        self.bindir = os.path.join(self.root, "bin")
        os.makedirs(self.bindir)
        self.workdir = os.path.join(self.root, "wd")
        os.makedirs(self.workdir)
        subprocess.run(["git", "init", "-q", "-b", "work", self.workdir],
                       check=True)
        fake = os.path.join(self.bindir, "grok")
        with open(fake, "w", encoding="utf-8") as handle:
            handle.write(FAKE_SPLIT_GROK)
        os.chmod(fake, 0o755)
        saved = {k: os.environ.get(k) for k in (
            "PATH", CRED_VAR, CRED_VAR + "_2", "FAKE_HANG", "FAKE_SECOND", "FAKE_BLOCKS",
            "FAKE_SNAP", "FAKE_SHORTS", CRED_VAR + "_3", CRED_VAR + "_4",
            "OMNI_LLM_CHILD_ENV_PASS", "FAKE_NONFINITE")}

        def restore():
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(restore)
        os.environ["PATH"] = self.bindir + os.pathsep + os.environ["PATH"]
        os.environ[CRED_VAR] = SECRET
        # The fenced fake CLI gets an allowlisted environment (#138): its
        # scenario switches are passed by name, as an operator would.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = ",".join(FAKE_SWITCHES)

    def start(self, hang=False, second=False, blocks=False, snap=False, shorts=False,
              nonfinite=False):
        os.environ.pop("FAKE_NONFINITE", None)
        if nonfinite:
            os.environ["FAKE_NONFINITE"] = "1"
        for name in ("FAKE_SHORTS", CRED_VAR + "_3", CRED_VAR + "_4"):
            os.environ.pop(name, None)
        if shorts:
            # Short registered values that sit inside the long one.
            os.environ["FAKE_SHORTS"] = "1"
            os.environ[CRED_VAR + "_3"] = SECRET[:8]
            os.environ[CRED_VAR + "_4"] = SECRET[-8:]
        os.environ.pop("FAKE_SNAP", None)
        if snap:
            os.environ["FAKE_SNAP"] = "1"
        os.environ.pop("FAKE_BLOCKS", None)
        if blocks:
            os.environ["FAKE_BLOCKS"] = "1"
        os.environ.pop("FAKE_SECOND", None)
        os.environ.pop(CRED_VAR + "_2", None)
        if second or snap:
            # A second credential equal to the first 13 characters of the first.
            os.environ["FAKE_SECOND"] = "1" if second else ""
            os.environ[CRED_VAR + "_2"] = SECRET[:13]
        if hang:
            os.environ["FAKE_HANG"] = "1"
        else:
            os.environ.pop("FAKE_HANG", None)
        started = jobs.start(task="split", model="grok-4.5",
                             workdir=self.workdir, timeout_s=90)
        return started["job_dir"]

    def tail_cli(self, job_dir, *args, interrupt_after=None):
        """Run ``omni agent tail``; with *interrupt_after* follow, then Ctrl-C.

        The follow runs until it is interrupted, so the helper waits (bounded)
        for the output that proves it has started and printed the job's lines,
        pauses *interrupt_after* seconds, then sends SIGINT. The child gets the
        default SIGINT disposition: a suite started in the background inherits
        SIGINT as ignored, and a bare timeout after the signal would then fail
        whatever the load is.
        """
        env = {k: v for k, v in os.environ.items() if k != CRED_VAR}
        env["OMNI_LLM_STATE_DIR"] = _TMP
        argv = [sys.executable, os.path.join(_REPO, "omni_cli.py"),
                "agent", "tail", job_dir, "-n", "500"] + list(args)
        if interrupt_after is None:
            return subprocess.run(argv, capture_output=True, text=True, env=env)
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        seen = b""
        deadline = time.time() + 60
        while (time.time() < deadline and b"message_stop" not in seen
               and proc.poll() is None):
            if select.select([proc.stdout], [], [], 0.2)[0]:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                seen += chunk
        time.sleep(interrupt_after)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            out, err = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        return subprocess.CompletedProcess(
            argv, proc.returncode, (seen + out).decode("utf-8", "replace"),
            err.decode("utf-8", "replace"))


class SplitSecretJobTests(_JobCase):
    def test_split_credential_never_reaches_stream_log_or_follow(self):
        job_dir = self.start()
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        self.assertTrue(stream)
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")
        texts = delta_texts(stream)
        self.assertEqual(texts[(0, "text_delta")], "key is <redacted> end")
        self.assertEqual(texts[(1, "thinking_delta")], "mull <redacted>")
        # The control text split across deltas is reproduced exactly.
        self.assertEqual(texts[(2, "text_delta")], CONTROL)
        # Every line still honours the contract.
        for line in stream.splitlines():
            if line.strip():
                self.assertTrue(stream_contract.is_contract_line(json.loads(line)))
        # `agent tail --follow` runs in a process whose registry is empty: it
        # shows only what the file holds.
        followed = self.tail_cli(job_dir, "--follow", interrupt_after=1.5)
        self.assertEqual(followed.returncode, 0, followed.stderr[-300:])
        self.assertIn("content_block_delta", followed.stdout)
        self.assertIsNone(leaked(followed.stdout), "tail --follow leaked")
        plain = self.tail_cli(job_dir)
        self.assertIsNone(leaked(plain.stdout), "tail leaked")

    def test_second_credential_that_is_a_prefix_of_the_first_leaves_no_fragment(self):
        job_dir = self.start(second=True)
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")
        followed = self.tail_cli(job_dir, "--follow", interrupt_after=1.5)
        self.assertIsNone(leaked(followed.stdout), "tail --follow leaked")
        self.assertEqual(delta_texts(stream)[(3, "text_delta")],
                         "<redacted><redacted>")

    def test_interior_between_markers_of_short_registered_values_leaves_no_fragment(self):
        job_dir = self.start(shorts=True)
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")
        followed = self.tail_cli(job_dir, "--follow", interrupt_after=0.5)
        self.assertIsNone(leaked(followed.stdout), "tail --follow leaked")
        self.assertIn(" end", delta_texts(stream)[(6, "text_delta")])

    def test_snapshot_whose_position_is_not_the_block_index_leaves_no_fragment(self):
        job_dir = self.start(snap=True)
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")
        followed = self.tail_cli(job_dir, "--follow", interrupt_after=0.5)
        self.assertIsNone(leaked(followed.stdout), "tail --follow leaked")
        snaps = [o for o in _lines(stream) if o["type"] == "x_omni_assistant"]
        self.assertTrue(snaps)
        for snap in snaps[:-1]:
            self.assertEqual(snap["event"]["message"]["content"], [])

    def test_tool_use_input_split_across_deltas_and_in_the_envelope_is_masked(self):
        job_dir = self.start(blocks=True)
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")
        self.assertEqual(json.loads(delta_texts(stream)[(4, "input_json_delta")]),
                         {"command": "echo <redacted>"})
        envelope = [o for o in _lines(stream) if o["type"] == "x_omni_assistant"][0]
        self.assertEqual(
            envelope["event"]["message"]["content"][0]["input"]["command"],
            "echo <redacted>")
        followed = self.tail_cli(job_dir, "--follow", interrupt_after=1.5)
        self.assertIsNone(leaked(followed.stdout), "tail --follow leaked")

    def test_a_child_that_prints_non_finite_numbers_leaves_a_strict_stream_log(self):
        job_dir = self.start(nonfinite=True)
        self.assertEqual(_wait_terminal(job_dir), "done")
        stream = _read(os.path.join(job_dir, "stream.log"))
        parsed = [strict_json(line) for line in stream.splitlines() if line.strip()]
        self.assertNotIn("unmaskable_line", stream)
        pings = [o for o in parsed if o["type"] == "x_omni_ping"]
        self.assertEqual([p["event"]["value"] for p in pings],
                         ["Infinity", "-Infinity", "NaN"])
        self.assertEqual([p["event"]["n"]["v"] for p in pings],
                         [["Infinity"], ["-Infinity"], ["NaN"]])
        self.assertIn({"type": "x_omni_passthrough", "backend": "grok",
                       "raw": "[1e309,NaN]"}, parsed)
        # The credential is still masked next to them.
        self.assertIsNone(leaked(stream), "stream.log leaked a fragment")

    def test_final_result_text_keeps_its_masking(self):
        job_dir = self.start()
        self.assertEqual(_wait_terminal(job_dir), "done")
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as handle:
            res = json.load(handle)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["text"].startswith("ANSWER"), res["text"])
        self.assertIsNone(leaked(res["text"]))

    def test_raw_log_is_gone_when_the_job_is_done(self):
        job_dir = self.start()
        self.assertEqual(_wait_terminal(job_dir), "done")
        self.assertFalse(os.path.exists(os.path.join(job_dir, "stream.raw.log")))
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "stream.err.log")))
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "stream.log")))
        # No other file in the job directory keeps the raw event text either.
        for name in os.listdir(job_dir):
            path = os.path.join(job_dir, name)
            if os.path.isfile(path) and name != "stream.log":
                with open(path, "rb") as handle:
                    self.assertIsNone(
                        leaked(handle.read(), SECRET[:13]),
                        "%s kept a raw fragment" % name)

    def test_tail_raw_after_the_end_says_it_is_not_retained(self):
        job_dir = self.start()
        self.assertEqual(_wait_terminal(job_dir), "done")
        for extra in ((), ("--follow",)):
            shown = self.tail_cli(job_dir, "--raw", *extra,
                                  interrupt_after=1.0 if extra else None)
            self.assertEqual(shown.returncode, 1, (extra, shown.stderr[-300:]))
            envelope = json.loads(shown.stdout)
            self.assertFalse(envelope["ok"])
            self.assertEqual(envelope["reason"], "raw_log_not_retained")
        # stream.log and stream.err.log stay servable.
        self.assertEqual(self.tail_cli(job_dir).returncode, 0)
        self.assertEqual(self.tail_cli(job_dir, "--err").returncode, 0)


class InRunRawLogTests(_JobCase):
    def test_raw_log_is_private_and_served_while_running_then_gone_on_stop(self):
        job_dir = self.start(hang=True)
        raw = os.path.join(job_dir, "stream.raw.log")
        stream = os.path.join(job_dir, "stream.log")
        deadline = time.time() + 30
        while time.time() < deadline and not (
                os.path.isfile(stream) and "message_start" in _read(stream)):
            time.sleep(0.1)
        self.assertIn("message_start", _read(stream))
        time.sleep(0.5)     # the first block is open: none of it is persisted
        self.assertNotIn("content_block", _read(stream))
        self.assertTrue(os.path.isfile(raw))
        self.assertEqual(stat.S_IMODE(os.stat(raw).st_mode), 0o600)
        live = self.tail_cli(job_dir, "--raw")
        self.assertEqual(live.returncode, 0, live.stderr[-300:])
        self.assertIn("content_block_delta", live.stdout)
        # The raw log has the half-written block; stream.log has nothing of it.
        self.assertIsNone(leaked(_read(stream)))
        try:
            jobs.stop(job_dir, grace_s=3, reason="test")
            state = _wait_terminal(job_dir, timeout=15)
            self.assertIn(state, ("stopped", "failed"), state)
            self.assertFalse(os.path.exists(raw))
            # The unfinished block is omitted with a notice, never partial.
            self.assertIsNone(leaked(_read(stream)), "stop path leaked")
            self.assertIn("unfinished_block", _read(stream))
        finally:
            for key in ("supervisor_pgid", "backend_pgid"):
                try:
                    pgid = jobs._read_job(job_dir).get(key)
                    if pgid:
                        os.killpg(int(pgid), signal.SIGKILL)
                except (OSError, ValueError, ProcessLookupError):
                    pass


class RawTailRefusalTests(_Base):
    """The raw log is served only while the job state is starting or running."""

    def _job(self, name, state=...):
        job_dir = os.path.join(self.root, name)
        os.makedirs(job_dir)
        if state is ...:
            data = {"job_id": name}                  # no state key at all
        elif state == "{}":
            data = {}
        else:
            data = {"job_id": name, "state": state}
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        with open(os.path.join(job_dir, "stream.raw.log"), "wb") as handle:
            handle.write(raw_lines(text_delta(0, SECRET[:13])))
        return job_dir

    def _cli(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = omni_cli._main_agent_job(argv)
        return code, out.getvalue()

    def test_every_state_but_starting_and_running_is_refused_everywhere(self):
        cases = [("done", "done"), ("failed", "failed"), ("stopped", "stopped"),
                 ("empty", "{}"), ("null", None), ("missing", ...), ("weird", "weird"),
                 ("list", []), ("dict", {})]
        for name, state in cases:
            with self.subTest(state=name):
                job_dir = self._job("raw170-" + name, state)
                with self.assertRaises(jobs.RawLogRefused) as caught:
                    jobs.tail(job_dir, lines=5, raw=True)
                self.assertFalse(caught.exception.envelope["ok"])
                code, shown = self._cli(["tail", job_dir, "--raw"])
                self.assertEqual(code, 1)
                self.assertIn(json.loads(shown)["reason"],
                              ("raw_log_not_retained", "raw_log_state_unreadable"))
                self.assertNotIn(SECRET[:13], shown)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = omni_cli._tail_follow(job_dir, raw=True, n=5)
                self.assertEqual(code, 1)
                self.assertNotIn(SECRET[:13], out.getvalue())
                # The other files stay servable.
                self.assertEqual(jobs.tail(job_dir, lines=5), "")

    def test_a_state_that_is_not_a_string_is_unreadable_not_an_error(self):
        for name, state in (("list", []), ("dict", {}), ("number", 7)):
            with self.subTest(state=name):
                job_dir = self._job("raw170n-" + name, state)
                with self.assertRaises(jobs.RawLogRefused) as caught:
                    jobs.tail(job_dir, lines=5, raw=True)
                self.assertEqual(caught.exception.envelope["reason"],
                                 "raw_log_state_unreadable")
                code, shown = self._cli(["tail", job_dir, "--raw"])
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(shown)["reason"], "raw_log_state_unreadable")

    def test_a_terminal_state_is_not_retained_and_the_rest_is_unreadable(self):
        reasons = {name: jobs.raw_log_refusal(self._job("raw170r-" + name, state))["reason"]
                   for name, state in (("done", "done"), ("null", None), ("missing", ...))}
        self.assertEqual(reasons, {"done": "raw_log_not_retained",
                                   "null": "raw_log_state_unreadable",
                                   "missing": "raw_log_state_unreadable"})

    def test_starting_and_running_are_served(self):
        for state in ("starting", "running"):
            job_dir = self._job("raw170s-" + state, state)
            self.assertIsNone(jobs.raw_log_refusal(job_dir))
            self.assertIn(SECRET[:13], jobs.tail(job_dir, lines=5, raw=True))

    def test_follow_stops_with_the_refusal_when_the_state_flips_mid_follow(self):
        job_dir = self._job("raw170-flip", "running")
        raw = os.path.join(job_dir, "stream.raw.log")
        job_json = os.path.join(job_dir, "job.json")
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            if len(polls) == 1:
                with open(raw, "ab") as handle:
                    handle.write(b"LIVE-LINE\n")
            elif len(polls) == 2:
                with open(job_json, "w", encoding="utf-8") as handle:
                    json.dump({"job_id": "raw170-flip", "state": "done"}, handle)
                with open(raw, "ab") as handle:
                    handle.write(b"AFTER-THE-FLIP\n")
            elif len(polls) > 6:
                raise KeyboardInterrupt      # would follow forever: fail the test

        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                mock.patch.object(omni_cli.time, "sleep", fake_sleep):
            code = omni_cli._tail_follow(job_dir, raw=True, n=10)
        shown = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("LIVE-LINE", shown)
        self.assertNotIn("AFTER-THE-FLIP", shown)
        self.assertIn("raw_log_not_retained", shown)
        self.assertLessEqual(len(polls), 3)


class RepairedJobTests(_Base):
    def _orphan(self, job_id):
        job_dir = os.path.join(_TMP, "jobs", job_id)
        os.makedirs(job_dir)
        for name in ("job.lock", "result.lock"):
            open(os.path.join(job_dir, name), "ab").close()
        jobs._write_job(job_dir, {
            "job_id": job_id, "state": "running", "created_at": jobs._now() - 600,
            "pid": None, "pgid": None, "pid_identity": None,
            "supervisor_pid": None, "supervisor_pgid": None,
            "supervisor_identity": None, "supervisor_start_identity": None,
            "backend_pid": None, "backend_pgid": None,
            "backend_pid_identity": None, "backend_identity": None,
            "backend_start_identity": None, "stop_reason": None, "stopped_by": None,
        })
        raw = os.path.join(job_dir, "stream.raw.log")
        with open(raw, "wb") as handle:
            handle.write(raw_lines(text_delta(0, SECRET[:13])))
        return job_dir, raw

    def test_sweep_of_a_crashed_job_deletes_the_raw_log(self):
        job_dir, raw = self._orphan("crash170a")
        self.assertEqual(jobs.sweep_orphans(), ["crash170a"])
        self.assertEqual(jobs.status(job_dir)["state"], "failed")
        self.assertFalse(os.path.exists(raw))

    def test_status_of_a_crashed_job_deletes_the_raw_log(self):
        job_dir, raw = self._orphan("crash170b")
        self.assertEqual(jobs.status(job_dir)["state"], "failed")
        self.assertFalse(os.path.exists(raw))

    def test_stop_of_a_crashed_job_deletes_the_raw_log(self):
        job_dir, raw = self._orphan("crash170c")
        jobs.stop(job_dir, grace_s=0, reason="test")
        self.assertFalse(os.path.exists(raw))

    def _failing_unlink(self):
        real = os.unlink

        def unlink(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == "stream.raw.log":
                raise PermissionError(13, "injected")
            return real(path, *args, **kwargs)
        return mock.patch.object(jobs.os, "unlink", unlink)

    def _tail_raw(self, job_dir):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = omni_cli._main_agent_job(["tail", job_dir, "--raw"])
        return code, out.getvalue()

    def test_a_failed_delete_is_still_refused_recorded_and_retried(self):
        job_dir, raw = self._orphan("crash170e")
        with self._failing_unlink():
            self.assertEqual(jobs.sweep_orphans(), ["crash170e"])
            self.assertTrue(os.path.exists(raw))
            # The terminal state is what refuses, not the absence of the file.
            refusal = jobs.raw_log_refusal(job_dir)
            self.assertEqual(refusal["reason"], "raw_log_not_retained")
            code, shown = self._tail_raw(job_dir)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(shown)["reason"], "raw_log_not_retained")
            self.assertNotIn(SECRET[:13], shown)
            # The failure is on record in job.json.
            self.assertTrue(jobs._read_job(job_dir).get("raw_log_discard_error"))
            # And the repair path retries while the fault persists: still there.
            jobs.sweep_orphans()
            self.assertTrue(os.path.exists(raw))
        # Once the fault is gone the next repair pass removes it and clears the flag.
        jobs.sweep_orphans()
        self.assertFalse(os.path.exists(raw))
        self.assertFalse(jobs._read_job(job_dir).get("raw_log_discard_error"))

    def test_status_also_retries_a_failed_delete(self):
        job_dir, raw = self._orphan("crash170f")
        with self._failing_unlink():
            self.assertEqual(jobs.status(job_dir)["state"], "failed")
            self.assertTrue(os.path.exists(raw))
        jobs.status(job_dir)
        self.assertFalse(os.path.exists(raw))

    def test_unreadable_job_metadata_fails_closed(self):
        job_dir, raw = self._orphan("crash170h")
        data = jobs._read_job(job_dir)
        data.update(state="running", created_at=jobs._now())
        jobs._write_job(job_dir, data)
        self.assertIsNone(jobs.raw_log_refusal(job_dir))    # live: served
        for name, fault in (
                ("permission", mock.patch.object(
                    jobs, "_read_job", side_effect=PermissionError(13, "denied"))),
                ("value", mock.patch.object(
                    jobs, "_read_job", side_effect=ValueError("bad json"))),
                ("type", mock.patch.object(
                    jobs, "_read_job", side_effect=TypeError("bad")))):
            with self.subTest(fault=name), fault:
                self.assertEqual(jobs.raw_log_refusal(job_dir)["reason"],
                                 "raw_log_state_unreadable")
                code, shown = self._tail_raw(job_dir)
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(shown)["reason"], "raw_log_state_unreadable")
                self.assertNotIn(SECRET[:13], shown)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = omni_cli._tail_follow(job_dir, raw=True, n=5)
                self.assertEqual(code, 1)
                self.assertNotIn(SECRET[:13], out.getvalue())

    def test_corrupt_or_unknown_state_fails_closed(self):
        job_dir, raw = self._orphan("crash170i")
        with open(os.path.join(job_dir, "job.json"), "w") as handle:
            handle.write("{not json")
        code, shown = self._tail_raw(job_dir)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(shown)["reason"], "raw_log_state_unreadable")
        job_dir2, raw2 = self._orphan("crash170j")
        data = jobs._read_job(job_dir2)
        data["state"] = "weird"
        jobs._write_job(job_dir2, data)
        code, shown = self._tail_raw(job_dir2)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(shown)["reason"], "raw_log_state_unreadable")
        code, shown = self._tail_raw("no-such-job-170")
        self.assertEqual(code, 1)

    def test_refusal_follows_the_state_even_with_a_stray_file(self):
        job_dir, raw = self._orphan("crash170g")
        jobs.sweep_orphans()
        with open(raw, "wb") as handle:
            handle.write(b"stray")
        self.assertEqual(jobs.raw_log_refusal(job_dir)["reason"], "raw_log_not_retained")

    def test_a_running_job_keeps_its_raw_log(self):
        job_dir, raw = self._orphan("live170d")
        data = jobs._read_job(job_dir)
        data.update(state="starting", created_at=jobs._now())  # inside the grace
        jobs._write_job(job_dir, data)
        self.assertEqual(jobs.sweep_orphans(), [])
        self.assertTrue(os.path.exists(raw))


if __name__ == "__main__":
    unittest.main()
