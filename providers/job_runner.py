# no-opencc
"""Supervisor process for one detached agent job."""
import asyncio
import fcntl
import json
import os
import re
import signal
import sys
import time

from providers import cli_run, jobs, known_secrets, stream_contract
from providers.agent_lane import INTEGRITY_REASONS, run_agent
from providers.attest import read_successor_attest
from providers.errors import exception_detail
from providers.logutil import log_event
from providers.profiles import resolve_profile
from providers.route import resolve_route


def _wait_for_parent_ready(job_dir, timeout_s=10):
    path = os.path.join(job_dir, "runner.ready")
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.01)
    return os.path.exists(path)


def _backend_name(job):
    """Provider key for stream_contract (from job.json, else model route)."""
    provider = job.get("provider")
    if isinstance(provider, str) and provider:
        return provider
    try:
        return resolve_route(job.get("model")).get("provider") or "unknown"
    except Exception:
        return "unknown"


def _session_id_from_event(evt):
    """Return the session/thread id an event carries, or None.

    Codex: ``{"type":"thread.started","thread_id":"<uuid>"}``. Cursor: its
    ``system``/``init`` event carries ``session_id`` directly (cursor-agent
    has no ``--session-id`` flag; ``--resume`` needs the id it assigned).
    """
    if evt.get("type") == "thread.started":
        thread_id = evt.get("thread_id")
        return str(thread_id) if thread_id else None
    if evt.get("type") == "system" and evt.get("subtype") == "init":
        session_id = evt.get("session_id")
        return str(session_id) if session_id else None
    return None


def _try_capture_session_id(job_dir):
    """If stream.raw.log has a session-bearing event, write it into job.json once.

    Capture must survive a mid-run crash/stop — that is resume's whole
    point — so we write as soon as the line is seen. Reads stream.raw.log
    (not the normalized stream.log).
    """
    path = os.path.join(job_dir, "stream.raw.log")
    if not os.path.isfile(path):
        return False
    try:
        data = jobs._read_job(job_dir)
        offset = int(data.get("account_session_offset") or data.get("codex_session_offset") or 0)
        with open(path, "rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(evt, dict):
            continue
        session_id = _session_id_from_event(evt)
        if not session_id:
            continue
        try:
            data = jobs._read_job(job_dir)
            if data.get("session_id"):
                return True
            data["session_id"] = session_id
            jobs._write_job(job_dir, data)
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    return False


def _job_session_id_set(job_dir):
    """True when job.json already has a non-empty session_id."""
    try:
        data = jobs._read_job(job_dir)
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False
    return bool(data.get("session_id"))


async def _poll_session_id(job_dir, stop_event):
    """Poll stream.raw.log every 0.5s until session_id is captured or run ends.

    Claude/grok pre-assign session_id in job.json before spawn — exit as soon
    as that field is set so we never reread the raw log for the job lifetime.
    Codex and cursor leave it null and rely on event capture from the log.
    """
    codex = jobs._read_job(job_dir).get("provider") == "codex"
    if _job_session_id_set(job_dir) and not codex:
        return
    while not stop_event.is_set():
        if _job_session_id_set(job_dir) and not codex:
            return
        if _try_capture_session_id(job_dir) and not codex:
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
    # One last pass after the run ends (event may land in the final flush).
    if not _job_session_id_set(job_dir):
        _try_capture_session_id(job_dir)


# Upper bound, in bytes, of what the block assembler holds for blocks that are
# still open (their lines and the lines that arrive behind them). Past it the
# held content is dropped and one omission notice is persisted instead.
_MAX_HELD_BYTES = 4 * 1024 * 1024

# Delta kinds that carry model-written text, and the field holding it.
_DELTA_FIELD = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "input_json_delta": "partial_json",
}
# Backend events that end a message where the contract has no message stop of
# its own (pi): blocks still open are complete there.
_END_WRAPPERS = frozenset(
    "x_omni_" + kind for kind in ("message_end", "turn_end", "agent_end"))


# The ways a text can be read with its escapes decoded. Two views are kept,
# and a hit in either counts: over-inclusion is the safe side (a harmless
# deeply nested escape may become the notice when a value is registered).
#
# Faithful view: what a JSON string decoder does to the text, one level per
# pass. Left to right, \\ -> \, \" -> ", \/ -> /, \b \f \n \r \t, and \uXXXX
# (either hex case). A backslash a decoder keeps (one for each escaped pair)
# is kept, so a value that holds a backslash reads as it decodes.
_JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))')
_JSON_SIMPLE = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                "n": "\n", "r": "\r", "t": "\t"}
# A high surrogate directly followed by a low one: the two halves of a
# character past U+FFFF. Joined after each pass, so a pair whose halves were
# revealed by different passes (one escaped, the other nested) reads as one
# character.
_SURROGATE_PAIR = re.compile(r"[\ud800-\udbff][\udc00-\udfff]")
# Aggressive view: a JSON \uXXXX escape with an optional second one right
# behind it so a surrogate pair decodes as one character. The run of
# backslashes in front is consumed whole: a value spelled with backslash-u
# text must still read as the escape it imitates (a reader that decodes
# \uXXXX wherever it sees it). The lookbehind starts the match at the head of
# a run, so a long run of backslashes costs one scan, not one per position.
_UNICODE_ESCAPE = re.compile(
    r"(?<!\\)\\+u([0-9a-fA-F]{4})(?:\\+u([dD][c-fC-F][0-9a-fA-F]{2}))?")
# Each pass removes one level of escaping (\u0041 -> A -> A); a text
# that still holds an escape after this many is not one a real line has.
_DECODE_PASSES = 8


def _join_surrogates(text):
    return _SURROGATE_PAIR.sub(
        lambda m: chr(0x10000 + ((ord(m.group(0)[0]) - 0xD800) << 10)
                      + (ord(m.group(0)[1]) - 0xDC00)), text)


def _json_unescape(match):
    code = match.group(1)
    if code is not None:
        return chr(int(code, 16))
    return _JSON_SIMPLE[match.group(2)]


def _faithful_pass(text):
    decoded = _JSON_ESCAPE.sub(_json_unescape, text)
    if decoded == text:
        return text
    return _join_surrogates(decoded)


def _unescape_unicode(match):
    high = int(match.group(1), 16)
    low = match.group(2)
    if low is None:
        return chr(high)
    low = int(low, 16)
    if 0xD800 <= high <= 0xDBFF:
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    return chr(high) + chr(low)


def _aggressive_pass(text):
    decoded = _UNICODE_ESCAPE.sub(_unescape_unicode, text)
    if decoded == text:
        return text
    return _join_surrogates(decoded)


def _view_hits(text, one_pass):
    """True when a view of *text* made by repeating *one_pass* shows a known
    value in some pass, or is still changing after the last pass (it cannot be
    verified)."""
    for _ in range(_DECODE_PASSES):
        decoded = one_pass(text)
        if decoded == text:
            return False
        if known_secrets.find_fragment(decoded) >= 0:
            return True
        text = decoded
    return one_pass(text) != text


def _decoded_view_hits(text):
    """True when *text*, read with its escapes decoded, holds an 8-character
    window of a known value.

    A registered value is spelled the ways ``json.dumps`` writes it, which
    leaves ASCII alone; a child that writes the same value as escapes (all of
    it or some of it, either hex case, a surrogate pair for a character past
    U+FFFF, a doubled backslash, or an escape of the backslash that starts the
    next escape) has spelled it in a form that is registered nowhere, and a
    reader that decodes the line sees the value. Each view is the decoded
    text, repeated until it is stable; every pass is checked, and a text still
    changing after the last pass counts as a hit. Linear in the text for each
    pass, and skipped when the text has no backslash or nothing is known.
    """
    if "\\" not in text or not known_secrets.active():
        return False
    return (_view_hits(text, _faithful_pass)
            or ("\\u" in text and _view_hits(text, _aggressive_pass)))


def _create_private(path):
    """Create *path* owner-only when it is absent; an existing file is kept."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    os.close(fd)


def _mask_value(text):
    """``mask_known`` on an assembled block value.

    A ``<redacted>`` marker the byte masker left (a shorter registered value
    replaced inside a longer one) ends a stretch of the value, so the text
    after it is judged as the start of a value, and the text before it as an
    end. The block is then written through ``_mask_obj``, which also masks
    every run of 8+ characters that lies inside a known value.
    """
    marker = known_secrets.MARKER
    return marker.join(known_secrets.mask_known(part) for part in text.split(marker))


def _mask_fragments_deep(value, _depth=0):
    """*value* with ``mask_fragments`` applied to every string value, in place.

    Dict keys are schema and are left alone, as ``mask_known_deep`` does.
    """
    if _depth > 200:
        return value
    if isinstance(value, str):
        return known_secrets.mask_fragments(value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _mask_fragments_deep(item, _depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _mask_fragments_deep(item, _depth + 1)
    return value


def _ok_index(value):
    """A block index as a backend writes it: an int or a string, never a bool."""
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _size(line):
    return len(line.encode("utf-8", "backslashreplace"))


def _delta_ref(obj):
    """(block key, delta kind, container, field) of a text-bearing delta, or None.

    Shapes: the contract's ``content_block_delta`` (also what pi's text deltas
    become), pi's wrapped ``message_update`` deltas of any other kind (thinking,
    tool-call arguments) and cursor's wrapped ``thinking`` delta. The block key
    is the block index as text.
    """
    kind = obj.get("type")
    if kind == "content_block_delta":
        delta = obj.get("delta")
        dtype = delta.get("type") if isinstance(delta, dict) else None
        field = _DELTA_FIELD.get(dtype) if isinstance(dtype, str) else None
        if field and isinstance(delta.get(field), str) and _ok_index(obj.get("index")):
            return str(obj.get("index")), delta["type"], delta, field
        return None
    event = obj.get("event")
    if not isinstance(event, dict):
        return None
    if kind == "x_omni_message_update":
        ame = event.get("assistantMessageEvent")
        if (isinstance(ame, dict) and isinstance(ame.get("type"), str)
                and ame["type"].endswith("_delta")
                and isinstance(ame.get("delta"), str)
                and _ok_index(ame.get("contentIndex"))):
            return str(ame.get("contentIndex")), ame["type"], ame, "delta"
        return None
    if (kind == "x_omni_thinking" and event.get("subtype") == "delta"
            and isinstance(event.get("text"), str)):
        return "thinking", "thinking_delta", event, "text"
    return None


def _lifecycle(obj):
    """("open"|"close", block key), ("all", None) for a message edge, or None."""
    kind = obj.get("type")
    if not isinstance(kind, str):
        return None
    if kind == "content_block_start":
        if _ok_index(obj.get("index")) and isinstance(obj.get("content_block"), dict):
            return "open", str(obj.get("index"))
        return None
    if kind == "content_block_stop":
        if _ok_index(obj.get("index")):
            return "close", str(obj.get("index"))
        return None
    if kind in ("message_start", "message_stop") or kind in _END_WRAPPERS:
        return "all", None
    event = obj.get("event")
    if kind == "x_omni_message_update" and isinstance(event, dict):
        ame = event.get("assistantMessageEvent")
        etype = ame.get("type") if isinstance(ame, dict) else None
        if (isinstance(etype, str) and not etype.endswith("_delta")
                and _ok_index(ame.get("contentIndex"))):
            if etype.endswith("_start"):
                return "open", str(ame.get("contentIndex"))
            if etype.endswith("_end"):
                return "close", str(ame.get("contentIndex"))
        return None
    if (kind == "x_omni_thinking" and isinstance(event, dict)
            and event.get("subtype") != "delta"):
        return "close", "thinking"
    return None


def _member_key(obj):
    """Block key of an event that belongs to a block but is not assembled.

    The deltas of other kinds (a signature, citations) and pi's other
    ``message_update`` events: they stay among their block's held events, in
    their original order.
    """
    kind = obj.get("type")
    if kind == "content_block_delta" and _ok_index(obj.get("index")):
        return str(obj.get("index"))
    event = obj.get("event")
    if kind == "x_omni_message_update" and isinstance(event, dict):
        ame = event.get("assistantMessageEvent")
        if isinstance(ame, dict) and _ok_index(ame.get("contentIndex")):
            return str(ame.get("contentIndex"))
    return None


class _Block:
    """The lines of one content block that is still open."""

    __slots__ = ("items", "texts", "templates", "size")

    def __init__(self):
        self.items = []         # [("obj", event)] and [("delta", kind)] in order
        self.texts = {}         # delta kind -> text pieces
        self.templates = {}     # delta kind -> the first delta event of that kind
        self.size = 0           # bytes held for this block


class _BlockMasker:
    """Persist a content block only after it is complete and masked.

    The byte masker on the child's stdout (#160) cannot see a value that the
    child prints in two events, because JSON wrapping sits between the halves.
    Here the granularity is the content block. Its start, deltas and stop are
    held (nothing of an open block is persisted); at the stop the delta texts
    of each kind are joined into one value, ``mask_known`` runs on the whole
    value, and the block is written as its start, one delta per kind with the
    masked text, the other held events and the stop. A tool-call input is a
    block of ``input_json_delta`` events, assembled the same way. Every other
    line gets ``mask_known_deep`` on its string values; dict keys are schema
    here and are covered by the writer pass (``_StreamNormalizer
    ._append_contract``), which checks the exact text of every line it writes.

    A snapshot of the message (the assistant envelope) that arrives while a
    block is open loses all its content entries (their position says nothing
    about which block they copy): only completed, masked blocks are persisted.
    Lines that arrive while a block is open wait behind it, in order. A message start or stop (or a pi message/turn/agent
    end) completes the blocks still open. A pi block end whose block had
    deltas drops its ``content`` copy, so a block appears once.

    What is held is bounded by ``_MAX_HELD_BYTES``. Past it, or when the stream
    ends inside a block, the held content is dropped and one omission notice
    is persisted instead; nothing partial is written. A secret split across
    two separate blocks is not reassembled (the bar of result.json).
    """

    def __init__(self, backend):
        self.backend = backend
        self._blocks = {}
        self._dead = set()      # keys whose content was dropped for size
        self._queue = []        # masked lines waiting behind open blocks
        self._queued = 0        # their bytes
        self._held = 0          # bytes held in all, kept as a running count

    def _notice(self, reason, keys):
        keys = sorted(keys)
        return stream_contract._dumps({
            "type": "x_omni_omitted", "backend": self.backend,
            "reason": reason, "blocks": [k[:64] for k in keys[:16]],
            "count": len(keys)})

    def _mask_obj(self, obj):
        masked = known_secrets.mask_known_deep(obj)
        return stream_contract._dumps(_mask_fragments_deep(masked))

    def _mask_text(self, line):
        return known_secrets.mask_fragments(known_secrets.mask_known(line))

    def apply(self, lines):
        out = []
        for line in lines:
            out.extend(self._one(line))
        return out

    def flush_all(self):
        """The stream has ended: blocks still open are omitted, never partial."""
        out = []
        if self._blocks:
            out.append(self._notice("unfinished_block", self._blocks))
        self._blocks.clear()
        self._dead.clear()
        out.extend(self._queue)
        self._queue, self._queued, self._held = [], 0, 0
        return out

    def _one(self, line):
        if not self._blocks and not self._queue and not known_secrets.active():
            return [line]       # nothing is known, nothing to hide
        try:
            obj = json.loads(line)
        except (TypeError, ValueError, RecursionError):
            obj = None
        if not isinstance(obj, dict):
            return self._other(self._mask_text(line), None)
        try:
            ref = _delta_ref(obj)
            action, key = (None, None) if ref else (_lifecycle(obj) or (None, None))
            member = None if ref or action else _member_key(obj)
        except Exception:
            # An event of an unexpected shape is an ordinary line: it still gets
            # mask_known_deep, and it must never stall the stream.
            ref, action, key, member = None, None, None, None
        if ref is not None:
            return self._delta(obj, line, ref)
        if member is not None and (member in self._blocks or member in self._dead):
            if member in self._dead:
                return []
            block = self._blocks[member]
            block.items.append(("obj", obj))
            return self._hold(line, block)
        if action == "open":
            if key in self._dead:
                return []
            block = self._blocks.setdefault(key, _Block())
            block.items.append(("obj", obj))
            return self._hold(line, block)
        if action == "close":
            if key in self._dead:
                self._dead.discard(key)
                return []
            block = self._blocks.pop(key, None)
            if block is None:
                return self._other(self._mask_obj(obj), obj)
            block.items.append(("obj", obj))
            return self._render(block) + self._release()
        if action == "all":
            out = []
            for key in list(self._blocks):
                out.extend(self._render(self._blocks.pop(key)))
            self._dead.clear()
            return out + self._release() + [self._mask_obj(obj)]
        return self._other(self._mask_obj(self._snapshot(obj)), obj)

    def _snapshot(self, obj):
        """Drop the model content of a message snapshot while a block is open.

        A snapshot is a block-local envelope: its list position says nothing
        about which stream block an entry copies, so while any block is open no
        entry (text, thinking, tool call with its input) is persisted. That
        holds for a block whose content was dropped for size until its stop or
        the message edge. Only snapshots that arrive between blocks, whole,
        are kept (masked).
        """
        message = obj.get("event", {}).get("message") if obj.get("type") == "x_omni_assistant" \
            and isinstance(obj.get("event"), dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if (self._blocks or self._dead) and isinstance(content, list):
            message["content"] = []
        return obj

    def _other(self, masked, obj):
        if not self._blocks:
            return [masked]
        self._queue.append(masked)
        return self._hold(masked, None)

    def _delta(self, obj, line, ref):
        key, kind, container, field = ref
        if key in self._dead:
            return []
        block = self._blocks.setdefault(key, _Block())
        if kind not in block.texts:
            block.texts[kind] = []
            block.templates[kind] = obj
            block.items.append(("delta", kind))
        block.texts[kind].append(container[field])
        return self._hold(line, block)

    def _hold(self, line, block):
        size = _size(line)
        if block is None:
            self._queued += size
        else:
            block.size += size
        self._held += size
        if self._held <= _MAX_HELD_BYTES:
            return []
        keys = set(self._blocks)
        self._dead |= keys
        self._blocks.clear()
        self._queue, self._queued, self._held = [], 0, 0
        return [self._notice("held_content_over_limit", keys)]

    def _release(self):
        if self._blocks:
            return []
        out, self._queue = self._queue, []
        self._held -= self._queued
        self._queued = 0
        return out

    def _render(self, block):
        self._held -= block.size
        had_text = bool(block.texts)
        out = []
        for what, value in block.items:
            if what == "delta":
                template = block.templates[value]
                _key, _kind, container, field = _delta_ref(template)
                container[field] = _mask_value("".join(block.texts[value]))
                out.append(self._mask_obj(template))
                continue
            if had_text:
                self._drop_copy(value)
            out.append(self._mask_obj(value))
        return out

    @staticmethod
    def _drop_copy(obj):
        """A pi block end repeats the block's text in ``content``: once is enough."""
        event = obj.get("event")
        if obj.get("type") == "x_omni_message_update" and isinstance(event, dict):
            ame = event.get("assistantMessageEvent")
            if isinstance(ame, dict):
                ame.pop("content", None)


class _StreamNormalizer:
    """Tail stream.raw.log → normalize → append stream.log (complete lines).

    The delta text is masked as reassembled logical text before it is written
    (see _BlockMasker); stream.raw.log is only the IPC file feeding this
    and is deleted when the job reaches a terminal state (jobs._write_job).

    Buffers partial *bytes* across reads and only decodes complete lines
    (split on b"\\n"). final_drain() flushes any remainder after the child
    exits so stream.log is complete before result.json.

    Policy: the normalizer must NEVER affect job outcome — write failures and
    decode/normalize exceptions stay isolated (see _normalize_loop / _execute).
    """

    def __init__(self, job_dir, backend):
        self.raw_path = os.path.join(job_dir, "stream.raw.log")
        self.out_path = os.path.join(job_dir, "stream.log")
        self.backend = backend
        self._offset = 0
        self._buf = b""
        self._logical = _BlockMasker(backend)
        # Ensure both files exist so readers never race on missing paths.
        for path in (self.raw_path, self.out_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not os.path.isfile(path):
                if path == self.raw_path:
                    # The in-run IPC file: owner-only, deleted when the job ends.
                    _create_private(path)
                else:
                    with open(path, "ab"):
                        pass

    @staticmethod
    def _parses(text, strict=False):
        """True when *text* is JSON. Not strict, the Python parser's own view
        (it also takes Infinity / NaN / 1e309); strict, the view of a consumer
        that accepts only RFC JSON.

        A text nested past what the parser can walk (RecursionError) cannot be
        verified: it counts as JSON that must be checked (not strict: True) and
        as failing the check (strict: False), so the gate replaces it."""
        try:
            if strict:
                stream_contract.loads_strict(text)
            else:
                json.loads(text)
        except RecursionError:
            return not strict
        except (TypeError, ValueError):
            return False
        return True

    def _clean(self, text, was_json):
        """True when *text* is safe to write: it parses strictly (if the input
        parsed at all), no 8-character window of any known value is left in
        it, and none is left in it once its \\uXXXX escapes are decoded."""
        if was_json and not self._parses(text, strict=True):
            return False
        if known_secrets.find_fragment(text) >= 0:
            return False
        return not _decoded_view_hits(text)

    def _persistable(self, line):
        """The exact text to write for *line*, or None to write nothing.

        The last gate before the file: the text is taken as the file will hold
        it (lone surrogates become backslash escapes), every run of 8+
        characters inside a known value is masked in it, and the result is
        checked on its own (it still parses when the input did, and an
        independent scan finds no window). The parse is strict: a line that
        holds Infinity, -Infinity, NaN or an overflowing number is not JSON to
        the readers of stream.log. *was_json* is taken with the permissive
        parse, so such a line counts as one that must pass the strict check
        rather than as text that is exempt from it. A line that fails is
        replaced by a fixed notice that carries nothing the child wrote; if
        even that fails, nothing is written. Nothing is repaired twice and
        nothing is re-serialized after the check.
        """
        was_json = self._parses(line)
        line = known_secrets.mask_known(line, edges=False)
        text = line.encode("utf-8", "backslashreplace").decode("utf-8")
        masked = known_secrets.mask_fragments(text)
        if self._clean(masked, was_json):
            return masked
        return self._unmaskable_notice()

    def _unmaskable_notice(self):
        notice = stream_contract._dumps({
            "type": "x_omni_omitted", "backend": self.backend,
            "reason": "unmaskable_line"})
        if self._clean(notice, True):
            return notice
        return None

    def _gated(self, line):
        """``_persistable``, but whatever it raises for one line, that line
        becomes the fixed notice and the lines after it are still processed:
        the writer must never stall the stream for the rest of the job."""
        try:
            return self._persistable(line)
        except Exception:
            try:
                return self._unmaskable_notice()
            except Exception:
                return None

    def _append_contract(self, lines):
        if not lines:
            return
        try:
            with open(self.out_path, "a", encoding="utf-8",
                      errors="backslashreplace") as handle:
                for line in lines:
                    text = self._gated(line)
                    if text is None:
                        continue
                    handle.write(text)
                    if not text.endswith("\n"):
                        handle.write("\n")
                handle.flush()
        except OSError:
            pass

    def _consume_complete(self, data, flush_remainder=False):
        """Feed newly read bytes; decode+normalize complete lines only."""
        if data:
            self._buf += data
        if not self._buf:
            return
        parts = self._buf.split(b"\n")
        if flush_remainder:
            complete = parts
            self._buf = b""
        else:
            # Last element is incomplete (no trailing newline yet).
            complete, self._buf = parts[:-1], parts[-1]
        # Drop empty segments from consecutive newlines / leading splits, but
        # still normalize non-empty logical lines (including whitespace-only).
        to_norm = []
        for part in complete:
            if part == b"":
                continue
            # One decode per complete line; errors="replace" is the final guard.
            to_norm.append(part.decode("utf-8", errors="replace"))
        if not to_norm:
            return
        out_lines = list(stream_contract.normalize_lines(self.backend, to_norm))
        self._append_contract(self._logical.apply(out_lines))

    def poll_once(self):
        """Read new bytes from raw log and normalize complete lines."""
        try:
            if not os.path.isfile(self.raw_path):
                return
            size = os.path.getsize(self.raw_path)
            if size < self._offset:
                # Truncated/rewritten — restart from beginning.
                self._offset = 0
                self._buf = b""
            if size == self._offset:
                return
            with open(self.raw_path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return
        if not chunk:
            return
        self._consume_complete(chunk, flush_remainder=False)

    def final_drain(self):
        """Read raw to EOF, flush partial buffer, normalize remaining lines."""
        self.poll_once()
        if self._buf:
            self._consume_complete(b"", flush_remainder=True)
        # Text still held back is judged now: the child has ended (or been
        # stopped), so no later event can complete it.
        self._append_contract(self._logical.flush_all())


async def _normalize_loop(normalizer, stop_event):
    """Poll raw log ~0.2s until stop_event; caller still runs final_drain.

    ANY exception here degrades to "stream.log stops growing" + one
    log_event — never propagates into _execute (job outcome stays intact).
    """
    if normalizer is None:
        try:
            await stop_event.wait()
        except Exception:
            pass
        return
    while not stop_event.is_set():
        try:
            normalizer.poll_once()
        except Exception as exc:
            log_event("stream_normalizer_error",
                      detail=str(exc)[:300], phase="poll")
            # Stop growing: wait for the run to end, then exit cleanly.
            try:
                await stop_event.wait()
            except Exception:
                pass
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
    # One more poll after the child is done; final_drain follows in _execute.
    try:
        normalizer.poll_once()
    except Exception as exc:
        log_event("stream_normalizer_error",
                  detail=str(exc)[:300], phase="poll")


def _after_stop(result):
    """What a cancelled / stopped run reports.

    A finished success stays. A verdict about the repository's integrity (a
    protected ref changed, or could not be verified) outranks the stop: it is
    kept, with the stop noted in its detail. Everything else becomes "stopped".
    """
    if isinstance(result, dict) and result.get("ok"):
        return result
    if isinstance(result, dict) and result.get("reason") in INTEGRITY_REASONS:
        kept = dict(result)
        kept["detail"] = "%s (the job was also being stopped)" % (kept.get("detail") or "")
        return kept
    return {"ok": False, "reason": "stopped", "detail": "supervisor cancelled"}


async def _execute(job_dir, job):
    current = asyncio.current_task()
    backend_proc = None
    result = None
    stop_requested = False
    backend = _backend_name(job)
    raw_log = os.path.join(job_dir, "stream.raw.log")
    err_log = os.path.join(job_dir, "stream.err.log")
    try:
        _create_private(raw_log)
    except OSError:
        pass
    # Construction is outside the main try; any failure must not kill the job.
    try:
        normalizer = _StreamNormalizer(job_dir, backend)
    except Exception as exc:
        log_event("stream_normalizer_error",
                  detail=str(exc)[:300], phase="init")
        normalizer = None

    def on_spawn(proc):
        nonlocal backend_proc
        backend_proc = proc
        jobs._record_backend(job_dir, proc)

    async def shutdown_backend():
        if backend_proc is not None:
            await cli_run._kill_proc_tree(backend_proc)

    def handle_sigterm(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True
        if current is not None:
            current.cancel()

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_sigterm)
    with open(os.path.join(job_dir, "runner.active"), "w", encoding="utf-8") as handle:
        handle.write("1\n")
    stop_event = asyncio.Event()
    poll_task = asyncio.create_task(_poll_session_id(job_dir, stop_event))
    norm_stop = asyncio.Event()
    norm_task = asyncio.create_task(_normalize_loop(normalizer, norm_stop))
    try:
        agent_kwargs = {}
        if job.get("effort") is not None:
            agent_kwargs["effort"] = job.get("effort")
        if "fence_requested" in job:
            agent_kwargs["fence_requested"] = bool(job.get("fence_requested"))
        if job.get("expect_writes"):
            agent_kwargs["expect_writes"] = True
        if job.get("foreman"):
            agent_kwargs["foreman"] = True
        if job.get("allow_main"):
            agent_kwargs["allow_main"] = True
        if job.get("permission_mode"):
            agent_kwargs["permission_mode"] = job.get("permission_mode")
        if backend == "codex" and job.get("codex_home"):
            def selected_account(account, selection):
                # Synchronous callback precedes the backend spawn. The old
                # thread id and its stream prefix cannot bind the next account.
                current_job = jobs._read_job(job_dir)
                current_job["account_profile"] = account["account_profile"]
                current_job["codex_home"] = account["codex_home"]
                current_job["session_id"] = None
                current_job["codex_session_offset"] = (
                    os.path.getsize(raw_log) if os.path.isfile(raw_log) else 0)
                jobs._write_job(job_dir, current_job)
            agent_kwargs.update(
                codex_home=job["codex_home"],
                codex_auth_isolated=bool(job.get("codex_auth_isolated", job.get("codex_managed", False))),
                codex_homes=job.get("codex_homes") or {},
                codex_initial={"account_profile": job.get("account_profile") or "default",
                               "codex_home": job["codex_home"]},
                on_account_selected=selected_account)
            if job.get("codex_managed") and not job.get("resume_session"):
                agent_kwargs["codex_candidates"] = job.get("codex_candidates") or []
                agent_kwargs["codex_tried"] = job.get("codex_tried") or []
        if backend == "codex" and job.get("codex_managed") and job.get("codex_deadline") is not None:
            remaining = float(job["codex_deadline"]) - time.time()
            if remaining <= 0:
                return _finalize(job_dir, {"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"})
            agent_kwargs["codex_deadline"] = time.monotonic() + remaining
        if backend in ("claude", "grok") and job.get("account_home"):
            def selected_provider_account(account, selection):
                # The final selector may skip a newly cooled preselection.
                # Publish its pin atomically before spawn; keep the preassigned
                # Claude/Grok session id because no attempt has started yet.
                current_job = jobs._read_job(job_dir)
                current_job.update(
                    account_profile=account["account_profile"],
                    account_home=account["account_home"],
                    account_home_explicit=account.get("account_home_explicit", True))
                if "account_kind" in account:
                    current_job["account_kind"] = account["account_kind"]
                    # The walk may have been redone against current health.
                    current_job["account_tried"] = list(selection.tried)
                jobs._write_job(job_dir, current_job)
                job.update(current_job)
            agent_kwargs.update(
                account_home=job["account_home"],
                account_home_explicit=job.get("account_home_explicit", True),
                account_auth_isolated=bool(job.get("account_auth_isolated", False)),
                account_homes=job.get("account_homes") or {},
                account_initial={"account_profile": job.get("account_profile") or "default",
                                 "account_home": job["account_home"],
                                 "account_home_explicit": job.get("account_home_explicit", True)},
                on_account_selected=selected_provider_account)
            if job.get("account_managed") and not job.get("resume_session"):
                agent_kwargs["account_candidates"] = job.get("account_candidates") or []
                agent_kwargs["account_tried"] = job.get("account_tried") or []
            if job.get("account_deadline") is not None:
                remaining = float(job["account_deadline"]) - time.time()
                if remaining <= 0:
                    return _finalize(job_dir, {"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"})
                agent_kwargs["account_deadline"] = time.monotonic() + remaining
        result = await run_agent(
            task=job.get("task", ""),
            task_file=os.path.join(job_dir, "prompt.txt"),
            model=job.get("model"),
            workdir=job.get("workdir"),
            read_only=bool(job.get("read_only", False)),
            timeout_s=job.get("timeout_s"),
            account_profile=job.get("account_profile", "default"),
            log_path=raw_log,
            err_log_path=err_log,
            job_lane=True,
            on_spawn=on_spawn,
            session_id=job.get("session_id"),
            resume_session=job.get("resume_session"),
            images=job.get("images") or None,
            **agent_kwargs,
        )
        await asyncio.sleep(0)
        if stop_requested:
            result = _after_stop(result)
    except asyncio.CancelledError:
        if stop_requested:
            await shutdown_backend()
        result = _after_stop(result)
    except Exception as exc:  # run_agent normally returns envelopes, but never leak a traceback
        result = {
            "ok": False,
            "reason": "runtime_error",
            "detail": exception_detail(exc),
        }
    finally:
        # Stop normalizer loop, then FINAL DRAIN so stream.log is complete
        # before result.json / terminal state are published. Broaden the
        # await guard: ANY exception logs and continues to _finalize.
        norm_stop.set()
        try:
            await asyncio.wait_for(norm_task, timeout=2.0)
        except Exception as exc:
            log_event("stream_normalizer_error",
                      detail=str(exc)[:300], phase="await_loop")
            norm_task.cancel()
            try:
                await norm_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if normalizer is not None:
                normalizer.final_drain()
        except Exception as exc:
            log_event("stream_normalizer_error",
                      detail=str(exc)[:300], phase="final_drain")
        stop_event.set()
        try:
            await asyncio.wait_for(poll_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            poll_task.cancel()
            try:
                await poll_task
            except (asyncio.CancelledError, Exception):
                pass
        signal.signal(signal.SIGTERM, previous_sigterm)

    if result is None:
        result = {
            "ok": False,
            "reason": "runtime_error",
            "detail": "agent returned no result",
        }
    if isinstance(result, dict):
        result = dict(result)
        try:
            job = jobs._read_job(job_dir)
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            job = {}
        provider = job.get("provider") or backend
        sid = job.get("session_id")
        env = dict(os.environ)
        try:
            overrides, err = resolve_profile(
                provider, job.get("account_profile") or "default")
            if not err and overrides:
                env.update(overrides)
            if provider == "codex" and job.get("codex_home"):
                env["CODEX_HOME"] = job["codex_home"]
            if provider in ("claude", "grok") and job.get("account_home"):
                from providers import provider_accounts
                provider_accounts.apply_home_env(env, provider, job["account_home"],
                                                job.get("account_home_explicit", True))
        except Exception:
            pass
        result["attest"] = read_successor_attest(
            provider, sid, env, workdir=job.get("workdir"))
    return _finalize(job_dir, result)


def _finalize(job_dir, result):
    """Write the result before publishing the matching terminal state."""
    winner = jobs._write_result(job_dir, result)
    job = jobs._read_job(job_dir)
    if winner.get("reason") == "stopped":
        job["state"] = "stopped"
    else:
        job["state"] = "done" if winner.get("ok") else "failed"
    jobs._write_job(job_dir, job)
    return winner


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        return 2
    job_dir = os.path.abspath(argv[0])
    known_secrets.prime()
    lock_path = os.path.join(job_dir, "job.lock")
    try:
        lock_handle = open(lock_path, "a+b")
        job = jobs._read_job(job_dir)
    except OSError:
        return 1
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not _wait_for_parent_ready(job_dir):
            return 1
        job = jobs._read_job(job_dir)
        asyncio.run(_execute(job_dir, job))
        return 0
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
