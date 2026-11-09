# no-opencc
"""Normalize provider usage objects into one stable envelope."""
import copy
import json
import re

from providers.errors import safe_detail


USAGE_SOURCES = ("provider", "estimated", "unavailable")

_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)

_BACKEND_FIELDS = {
    "agy": {
        "input_tokens": "input_tokens",
        "cached_input_tokens": "cache_read_tokens",
        "cache_write_tokens": None,
        "output_tokens": "output_tokens",
        "reasoning_tokens": "thinking_tokens",
        "total_tokens": "total_tokens",
    },
    "grok": {
        "input_tokens": "input_tokens",
        "cached_input_tokens": "cache_read_input_tokens",
        "cache_write_tokens": "cache_creation_input_tokens",
        "output_tokens": "output_tokens",
        "reasoning_tokens": "reasoning_tokens",
        "total_tokens": "total_tokens",
    },
    "claude": {
        "input_tokens": "input_tokens",
        "cached_input_tokens": "cache_read_input_tokens",
        "cache_write_tokens": "cache_creation_input_tokens",
        "output_tokens": "output_tokens",
        "reasoning_tokens": None,
        "total_tokens": None,
    },
    "codex": {
        "input_tokens": "input_tokens",
        "cached_input_tokens": "cached_input_tokens",
        "cache_write_tokens": "cache_write_input_tokens",
        "output_tokens": "output_tokens",
        "reasoning_tokens": "reasoning_output_tokens",
        "total_tokens": None,
    },
    "openai": {
        "input_tokens": "prompt_tokens",
        "cached_input_tokens": ("prompt_tokens_details", "cached_tokens"),
        # Read the cache-write count rather than declaring it unreportable:
        # hardcoding None turned a reported 0 into unknown, inverting the one rule
        # this module exists to enforce. An absent field still yields None, so
        # reading it is safe on providers that do not send it. UNVERIFIED-LIVE.
        "cache_write_tokens": ("prompt_tokens_details", "cache_write_tokens"),
        "output_tokens": "completion_tokens",
        "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens"),
        "total_tokens": "total_tokens",
    },
    # Live capture pi 0.84.2 --mode json (xai/grok-4.6): message.usage uses
    # camelCase input/output/cacheRead/cacheWrite/reasoning/totalTokens.
    # xAI reports cacheRead separately from input (490 + 128 = 618 toward
    # totalTokens 620 with output 2), so input is the uncached part and is
    # summed in _LOGICAL_INPUT_PARTS. Other pi providers may differ — this
    # mapping is pinned to the committed fixture, not to every catalog.
    "pi": {
        "input_tokens": "input",
        "cached_input_tokens": "cacheRead",
        "cache_write_tokens": "cacheWrite",
        "output_tokens": "output",
        "reasoning_tokens": "reasoning",
        "total_tokens": "totalTokens",
    },
    # Cursor `--output-format json` docs (2026-08) expose no token fields.
    # Unmapped until a live capture lands a fixture. All None, never 0.
    "cursor": {
        "input_tokens": None,
        "cached_input_tokens": None,
        "cache_write_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    },
}

# `input_tokens` must mean the same thing on every backend: TOTAL logical input,
# cached and uncached together (#22 P1 scoping ruling). Some providers report only
# the UNCACHED part, so their logical input is the sum of these raw fields.
# Measured live on this host 2026-08-11 rather than read off documentation:
#   grok   8305 + 4864 + 29(output) == 13198, its own reported total.
#   agy    same ~65.5k prompt twice: cold input=65497 cache_read=0; warm
#          input=45048 cache_read=20451 (== the same prompt), and agy's own
#          total_tokens is input+output in both runs.
#   claude 9 uncached against 13525 read + 12954 written.
# codex and openai already include cached input in their raw field, so summing
# them would double-count. Absence from this table means "already inclusive".
_LOGICAL_INPUT_PARTS = {
    "agy": ("input_tokens", "cache_read_tokens"),
    "grok": ("input_tokens", "cache_read_input_tokens",
             "cache_creation_input_tokens"),
    "claude": ("input_tokens", "cache_read_input_tokens",
               "cache_creation_input_tokens"),
    "pi": ("input", "cacheRead"),
}

_CLI_REPLY_FIELDS = {
    "agy": "response",
    "grok": "text",
    "claude": "result",
    "cursor": "result",
}


def _lookup(raw, path):
    if path is None or not isinstance(raw, dict):
        return None
    if isinstance(path, tuple):
        value = raw.get(path[0])
        if not isinstance(value, dict):
            return None
        return value.get(path[1])
    return raw.get(path)


def _measured_count(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _unavailable(backend):
    return normalize_usage(backend, None)


def normalize_usage(backend, raw):
    """Return a stable usage envelope for one supported provider.

    Only actual integer counts are measurements. Missing, null, and malformed
    values remain ``None`` so measured zero is never confused with unknown.
    """
    fields = _BACKEND_FIELDS.get(backend)
    if fields is None:
        raise ValueError("unknown usage backend: %s" % backend)

    source_raw = raw if isinstance(raw, dict) else None
    envelope = {
        field: _measured_count(_lookup(source_raw, fields[field]))
        for field in _TOKEN_FIELDS
    }
    parts = _LOGICAL_INPUT_PARTS.get(backend)
    if parts is not None:
        # Every part must be present. Returning the uncached part alone would
        # UNDERSTATE the total while still looking like a measurement -- the same
        # class of lie as substituting 0 for unknown.
        counts = [_measured_count(_lookup(source_raw, part)) for part in parts]
        envelope["input_tokens"] = (None if any(c is None for c in counts)
                                    else sum(counts))
    envelope["source"] = (
        "provider" if any(envelope[field] is not None for field in _TOKEN_FIELDS)
        else "unavailable"
    )
    envelope["raw"] = copy.deepcopy(source_raw) if source_raw is not None else None
    return envelope


def _text_value(payload, field):
    value = payload.get(field, "")
    return value.strip() if isinstance(value, str) else ""


_AGY_STREAM_EVENTS = frozenset(("init", "step_update", "result"))
_AGY_STREAM_PREFIX_RE = re.compile(
    r'^\s*\{\s*"event"\s*:\s*"(?:init|step_update|result)"'
    r'(?=\s*(?:,|\}|$))'
)


def _decode_agy_stream(stdout):
    """Decode AGY's stream-json framing and select its last terminal result.

    Ordinary AGY JSON documents and plain-text output deliberately remain on
    their existing recovery paths. Once an event frame identifies stream-json,
    however, malformed or incomplete framing is a failure and never a reply.
    """
    text = stdout if isinstance(stdout, str) else ""
    frames = []
    malformed = False
    recognized = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
        except (TypeError, ValueError):
            malformed = True
            frames.append(None)
            # Recognize only an object prefix whose first key is a complete,
            # known AGY event value. Earlier/ambiguous text remains eligible
            # for the legacy plain-text fallback.
            if _AGY_STREAM_PREFIX_RE.search(line):
                recognized = True
            continue
        if isinstance(frame, dict) and frame.get("event") in _AGY_STREAM_EVENTS:
            recognized = True
        frames.append(frame)
    if not recognized:
        return {"recognized": False, "valid": False, "result": None}

    def invalid(reason):
        return {"recognized": True, "valid": False, "result": None,
                "reason": reason}

    if malformed or any(not isinstance(frame, dict) for frame in frames):
        return invalid("malformed AGY stream-json frame")
    if any(frame.get("event") not in _AGY_STREAM_EVENTS for frame in frames):
        return invalid("unexpected AGY stream-json event")
    init_frames = [frame for frame in frames if frame.get("event") == "init"]
    result_frames = [frame for frame in frames if frame.get("event") == "result"]
    if len(init_frames) != 1:
        return invalid("incomplete AGY stream-json terminal results")
    if frames[0].get("event") != "init":
        return invalid("AGY stream-json init must be the first frame")
    if not isinstance(init_frames[0].get("init"), dict):
        return invalid("malformed AGY stream-json init payload")
    if not result_frames:
        return invalid("incomplete AGY stream-json terminal results")
    if frames[-1].get("event") != "result":
        return invalid("AGY stream-json ended after a terminal result without completing the next turn")
    if any(not isinstance(frame.get("step_update"), dict)
           for frame in frames if frame.get("event") == "step_update"):
        return invalid("malformed AGY stream-json step update")

    results = []
    previous_turns = None
    for frame in result_frames:
        result = frame.get("result")
        if not isinstance(result, dict):
            return invalid("malformed AGY stream-json result")
        turns = result.get("num_turns")
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
            return invalid("invalid AGY stream-json terminal turn count")
        if previous_turns is not None and turns <= previous_turns:
            return invalid("unordered AGY stream-json terminal turn sequence")
        previous_turns = turns
        if not isinstance(result.get("status"), str):
            return invalid("missing AGY stream-json terminal status")
        results.append(result)
    terminal = results[-1]
    if terminal.get("status") == "SUCCESS" and not isinstance(terminal.get("response"), str):
        return invalid("missing AGY stream-json terminal response")
    return {"recognized": True, "valid": True, "result": terminal,
            "results": results}


def _agy_stream_failure(decoded):
    """Return a bounded typed failure for an invalid or failed AGY stream."""
    if not decoded.get("recognized"):
        return None
    if not decoded.get("valid"):
        return "runtime_error", safe_detail(decoded.get("reason") or
                                            "invalid AGY stream-json output")
    result = decoded.get("result") or {}
    status = result.get("status")
    if status == "SUCCESS":
        return None
    cause = result.get("error")
    if isinstance(cause, dict):
        cause = (cause.get("message") or cause.get("detail") or
                 cause.get("error"))
    if not isinstance(cause, str) or not cause.strip():
        cause = "agy status=%s" % status
    return "runtime_error", safe_detail(cause)


def extract_cli_result(backend, stdout):
    """Extract reply text and usage from a provider CLI's JSON output.

    A malformed or empty output is returned as text with unavailable usage so a
    provider reply is never discarded merely because its usage was unavailable.
    Codex emits JSONL events and reports usage on its final ``turn.completed``
    event; its reply normally comes from the separate ``-o`` file.
    """
    if backend not in _BACKEND_FIELDS:
        raise ValueError("unknown usage backend: %s" % backend)

    text = stdout if isinstance(stdout, str) else ""
    stripped = text.strip()
    if not stripped:
        return "", _unavailable(backend)

    if backend == "codex":
        # Line-tolerant on purpose. An all-or-nothing parse meant one banner line
        # discarded every readable event AND returned the raw stream as the reply,
        # which providers/codex.py then hands back as a model answer when the -o
        # file is empty. Unreadable lines are skipped; the stream is NEVER the reply.
        events = []
        for line in stripped.splitlines():
            try:
                events.append(json.loads(line))
            except (TypeError, ValueError):
                continue

        reply = ""
        usage = None
        for event in events:
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if (event.get("type") == "item.completed"
                    and isinstance(item, dict)
                    and item.get("type") == "agent_message"):
                value = item.get("text")
                if isinstance(value, str):
                    reply = value.strip()
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
        return reply, normalize_usage(backend, usage)

    if backend == "pi":
        # pi --mode json is a JSONL event stream (session / message_update /
        # message_end / turn_end / agent_end). The stream is NEVER the reply.
        return _extract_pi_result(stripped)

    if backend == "agy":
        decoded = _decode_agy_stream(stripped)
        if decoded["recognized"]:
            if not decoded["valid"] or decoded["result"].get("status") != "SUCCESS":
                return "", _unavailable(backend)
            result = decoded["result"]
            return _text_value(result, "response"), normalize_usage(
                backend, result.get("usage")
            )

    payload = _parse_document(backend, stripped)
    if payload is None:
        if _embedded_objects(stripped):
            # Readable JSON is present but none of it is an envelope: protocol
            # debris (a truncated document, a stray fragment). Handing it back
            # would give the caller half a wire message as a model reply.
            # Keyed on "readable JSON exists", NOT on the first character: a reply
            # that merely begins with "{" is still a reply.
            return "", _unavailable(backend)
        # Genuinely not JSON: the CLI printed plain text. Hand the reply back
        # rather than losing it to a usage failure.
        return text, _unavailable(backend)
    return _text_value(payload, _CLI_REPLY_FIELDS.get(backend, "")), normalize_usage(
        backend, payload.get("usage")
    )


def _parse_document(backend, stripped):
    """Recover the JSON envelope from a CLI's stdout, or None if there isn't one.

    A whole-document parse alone was too brittle: a CLI that prints an update
    banner before its JSON failed it, so the ENTIRE protocol document — session
    ids, cost fields and all — was handed back as the model's answer while its
    perfectly readable usage was discarded.

    The fallback scans for embedded objects with `raw_decode` rather than by line,
    because these CLIs pretty-print: grok's real output spans many lines, so a
    line-based scan would have recovered nothing from exactly the shape that
    matters. The LAST object carrying `usage` wins, falling back to the last
    object found — a banner or progress object that happens to repeat the reply
    field must not outrank the result envelope.
    """
    try:
        payload = json.loads(stripped)
    except (TypeError, ValueError):
        payload = None
    # The result-like check applies to a lone whole document too, not just to
    # mixed output: a single progress object is still not an answer.
    if isinstance(payload, dict):
        return payload if _is_result_like(payload) else None

    found = _embedded_objects(stripped)
    if not found:
        return None
    # Only an object that actually looks like a result envelope may stand in for
    # one. Without this, a nested fragment (`{"input_tokens": 9}` lifted out of a
    # truncated document) could be mistaken for the envelope and quietly answer
    # for it. The LAST qualifying object wins, and one carrying `usage` outranks
    # one that merely repeats the reply field -- a trailing progress line must not
    # outrank the real result.
    reply_field = _CLI_REPLY_FIELDS.get(backend)
    found = [c for c in found if _is_result_like(c)]
    with_usage = [c for c in found if "usage" in c]
    if with_usage:
        return with_usage[-1]
    if reply_field:
        with_reply = [c for c in found if reply_field in c]
        if with_reply:
            return with_reply[-1]
    return None


def _assistant_text_blocks(message):
    """Join text blocks from a pi assistant message. Thinking is not the reply."""
    if not isinstance(message, dict):
        return ""
    if message.get("role") != "assistant":
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return "".join(parts).strip()


def _pi_usage_is_placeholder(raw):
    """True when every numeric usage field is 0 — the live mid-stream stub.

    A later non-zero ``message_update`` or a ``message_end.usage`` outranks this.
    An all-zero object that is the ONLY usage on the stream is still treated as
    unknown: collapsing it to measured zeros would invent a measurement.
    """
    if not isinstance(raw, dict):
        return True
    counts = []
    for key in ("input", "output", "cacheRead", "cacheWrite", "reasoning",
                "totalTokens"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts.append(value)
    if not counts:
        return True
    return all(c == 0 for c in counts)


def _extract_pi_result(stripped):
    """Reply + usage from a pi --mode json event stream.

    Last assistant ``message_end`` / ``turn_end`` wins for text. Usage is taken
    from that message when present, else from the last ``message_update.usage``
    that is not the all-zero placeholder (live capture starts at zeros and
    those zeros are not a measurement).
    Unreadable lines are skipped; the raw stream is never the reply.
    """
    reply = ""
    usage = None
    last_update_usage = None
    saw_event = False
    for line in stripped.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind is None:
            continue
        saw_event = True
        if kind == "message_update" and isinstance(event.get("usage"), dict):
            if not _pi_usage_is_placeholder(event["usage"]):
                last_update_usage = event["usage"]
        if kind in ("message_end", "turn_end"):
            text = _assistant_text_blocks(event.get("message"))
            if text:
                reply = text
                msg = event.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
                    if not _pi_usage_is_placeholder(msg["usage"]):
                        usage = msg["usage"]
    if usage is None:
        usage = last_update_usage
    if not reply and not saw_event:
        # Genuinely not a JSONL event stream: keep the reply rather than
        # losing a plain-text fallback (hermetic fakes, --mode text).
        return stripped, _unavailable("pi")
    return reply, normalize_usage("pi", usage)


def _is_result_like(obj):
    """Could this object be a result envelope, or does it announce it is not?

    The discriminator comes from the committed fixtures, not from guesswork:
    claude's envelope is `type: "result"`, agy's and grok's carry no `type` at
    all, and progress/banner objects declare something else. Without this, an
    object was ranked purely on carrying `usage`, so a progress line with a token
    count beat the real answer.
    """
    kind = obj.get("type")
    return kind is None or kind == "result"


def _embedded_objects(stripped):
    """Every JSON object embedded in `stripped`, in order.

    Scans with `raw_decode` rather than line by line because these CLIs
    pretty-print: grok's real output spans many lines, so a line-based scan would
    recover nothing from exactly the shape that matters.
    """
    decoder = json.JSONDecoder()
    found = []
    idx = stripped.find("{")
    while idx != -1:
        try:
            candidate, end = decoder.raw_decode(stripped, idx)
        except ValueError:
            idx = stripped.find("{", idx + 1)
            continue
        if isinstance(candidate, dict):
            found.append(candidate)
        idx = stripped.find("{", max(end, idx + 1))
    return found


# In-band failure: these CLIs report their own errors INSIDE a JSON envelope that
# still exits 0. Reading only the reply field turned such a run into either a
# successful-looking answer or a misfiled `empty_response` that hid the real cause.
_INBAND_HTTP_REASONS = {429: "rate_limit", 401: "auth_error", 403: "auth_error"}


def inband_error(backend, stdout):
    """Return (reason, detail) if the CLI reported a failure in-band, else None."""
    if backend not in _CLI_REPLY_FIELDS:
        return None
    text = stdout if isinstance(stdout, str) else ""
    if backend == "agy":
        stream_failure = _agy_stream_failure(_decode_agy_stream(text.strip()))
        if stream_failure is not None:
            return stream_failure
    candidates = _embedded_objects(text.strip())
    if not candidates:
        return None
    # Scan EVERY embedded object, not just the one chosen as the envelope: a
    # trailing progress line was enough to hide a committed error envelope when
    # only a single object was inspected.
    for payload in candidates:
        if not _is_result_like(payload):
            # A progress object's own status must not fail a run that succeeded.
            continue
        if backend in ("claude", "cursor") and payload.get("is_error"):
            status = payload.get("api_error_status")
            reason = _INBAND_HTTP_REASONS.get(status, "runtime_error")
            # The human-readable cause is `result` ("Not logged in · Please
            # run /login"); `subtype` is "success" even on that failure, so
            # it is only the fallback when result is empty.
            result_text = payload.get("result")
            if not (isinstance(result_text, str) and result_text.strip()):
                result_text = None
            detail = safe_detail(result_text or payload.get("subtype")
                                 or ("%s reported is_error" % backend))
            return reason, detail
        if backend == "agy":
            status = payload.get("status")
            if status is not None and status != "SUCCESS":
                if not isinstance(status, str):
                    status = json.dumps(status, ensure_ascii=False, default=str)
                return "runtime_error", safe_detail("agy status=%s" % status)
    return None
