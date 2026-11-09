# no-opencc
"""P3 stream contract: normalize backend JSONL into Messages stream events.

Every job stream line is either an Anthropic Messages STREAM EVENT
(message_start / content_block_start / content_block_delta /
content_block_stop / message_delta / message_stop) or an omni wrapper
{"type": "x_omni_<kind>", "backend": <backend>, ...}.

normalize_lines never drops a line and never raises on garbage.
final_text recovers the assistant answer. Ground truth: tests/fixtures/*.jsonl.
"""
import json
import math

MESSAGES_EVENT_TYPES = frozenset({
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
})


def is_contract_line(obj):
    """True iff obj is a Messages stream event or an x_omni_* wrapper."""
    if not isinstance(obj, dict):
        return False
    t = obj.get("type")
    if not isinstance(t, str):
        return False
    if t in MESSAGES_EVENT_TYPES:
        return True
    if t.startswith("x_omni_"):
        return True
    return False


def _reject_constant(name):
    raise ValueError("non-finite JSON constant: %s" % name)


def _reject_overflow(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON number out of float range: %s" % text)
    return value


def loads_strict(text):
    """json.loads that refuses Infinity, -Infinity, NaN and numbers (1e309)
    that overflow to infinity: the parse a strict JSON consumer would do.
    Raises ValueError (JSONDecodeError is one) on any of them."""
    return json.loads(text, parse_constant=_reject_constant,
                      parse_float=_reject_overflow)


_TOO_DEEP = '{"type":"x_omni_omitted","reason":"nesting_too_deep"}'


def _finite_leaf(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _is_container(value):
    return isinstance(value, (dict, list, tuple))


def _finite(obj):
    """A copy of *obj* with each non-finite float replaced by its name as a
    JSON string ("Infinity", "-Infinity", "NaN"); everything else is kept.

    Iterative (an explicit stack), so it reaches every depth json.loads and
    json.dumps do; a recursive copy runs out of frames at about half that
    depth (two Python frames per container)."""
    if not _is_container(obj):
        return _finite_leaf(obj)
    root = {} if isinstance(obj, dict) else []
    path = {id(obj)}        # the containers from the root down to the current one
    stack = [(_items(obj), root, id(obj))]
    while stack:
        items, dst, here = stack[-1]
        for key, value in items:
            if _is_container(value):
                if id(value) in path:
                    raise ValueError("circular reference")
                copy = {} if isinstance(value, dict) else []
                _put(dst, key, copy)
                path.add(id(value))
                stack.append((_items(value), copy, id(value)))
                break       # go down into it; resume this one after
            _put(dst, key, _finite_leaf(value))
        else:
            path.discard(here)
            stack.pop()
    return root


def _items(container):
    return iter(container.items() if isinstance(container, dict)
                else enumerate(container))


def _put(dst, key, value):
    if isinstance(dst, dict):
        dst[key] = value
    else:
        dst.append(value)


def _dumps(obj):
    """Compact JSON text of *obj*, strict: never Infinity / -Infinity / NaN.

    A child number that overflows (1e309) or a NaN / Infinity literal parses
    to a non-finite float; JSON has no such number, so it is written as the
    string "Infinity" / "-Infinity" / "NaN". The event keeps its shape (its
    type, its block index and the other fields stay what they were), unlike a
    switch to passthrough, which would turn the whole event into one raw
    string. Every line this module and the job writer produce goes through
    here, so the representation is the same for all of them.
    """
    try:
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False)
        except ValueError:
            return json.dumps(_finite(obj), ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False)
    except (RecursionError, ValueError):
        # Nested past what the encoder can walk, or a cycle (ValueError from
        # the encoder or from _finite; a child's JSON cannot make one): a fixed
        # notice instead of a raise that would drop the chunk the line came from.
        return _TOO_DEEP


def _passthrough(backend, raw=None, event=None):
    out = {"type": "x_omni_passthrough", "backend": backend}
    if raw is not None:
        out["raw"] = raw
    if event is not None:
        out["event"] = event
    return out


def _wrapper(backend, kind, event):
    """Wrap a backend envelope line as x_omni_<kind> with original under event."""
    safe = kind if isinstance(kind, str) and kind else "unknown"
    # Dots (e.g. thread.started) are not used as wrapper kinds here; callers
    # that want passthrough for codex use _passthrough instead.
    return {"type": "x_omni_%s" % safe, "backend": backend, "event": event}


def _parse_line(line):
    """Return (obj_or_None, raw_str). obj is None when line is not a JSON object."""
    if line is None:
        return None, ""
    if isinstance(line, dict):
        return line, _dumps(line)
    raw = line if isinstance(line, str) else str(line)
    raw = raw.strip() if isinstance(raw, str) else raw
    if not raw:
        return None, raw if isinstance(raw, str) else ""
    try:
        obj = json.loads(raw)
    except RecursionError:
        # Nested past the parser: it cannot be read, so none of it is kept.
        # raw None (a line, even an empty one, gives a str) marks this case; a
        # raw copy would carry whatever the child put inside, uninspected.
        return None, None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, raw
    if not isinstance(obj, dict):
        return None, raw
    return obj, raw


def _assistant_text_from_message(msg):
    """Join text blocks from an assistant envelope message dict."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _delta_text(event):
    """Extract text_delta text from a content_block_delta event."""
    if not isinstance(event, dict):
        return ""
    if event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return ""
    if delta.get("type") != "text_delta":
        return ""
    t = delta.get("text")
    return t if isinstance(t, str) else ""


def _codex_agent_message_sequence(text):
    """Minimal Messages event sequence for one codex agent_message."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [
        {"type": "message_start"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]


def _normalize_one_pi(backend, obj, raw):
    """Yield contract objects for one pi --mode json line.

    Live capture (pi 0.84.2): session / agent_start / turn_start /
    message_start / message_update / message_end / turn_end / agent_end.
    ``message_update.assistantMessageEvent`` of type ``text_delta`` becomes a
    Messages ``content_block_delta``. Everything else is wrapped.
    """
    if obj is None:
        yield _passthrough(backend, raw=raw)
        return

    t = obj.get("type")
    if isinstance(t, str) and t in MESSAGES_EVENT_TYPES:
        yield obj
        return

    if t == "message_update":
        ev = obj.get("assistantMessageEvent")
        if isinstance(ev, dict) and ev.get("type") == "text_delta":
            delta = ev.get("delta")
            text = delta if isinstance(delta, str) else ""
            yield {
                "type": "content_block_delta",
                "index": ev.get("contentIndex", 0),
                "delta": {"type": "text_delta", "text": text},
            }
            return
        yield _wrapper(backend, "message_update", obj)
        return

    if isinstance(t, str) and t:
        yield _wrapper(backend, t, obj)
        return
    yield _passthrough(backend, event=obj)


def _normalize_one_grok_claude(backend, obj, raw):
    """Yield one or more contract objects for a grok/claude raw line."""
    if obj is None:
        yield _passthrough(backend, raw=raw)
        return

    t = obj.get("type")

    # Already a contract Messages event — pass through.
    if isinstance(t, str) and t in MESSAGES_EVENT_TYPES:
        yield obj
        return

    # stream_event: unwrap inner Anthropic Messages event when present.
    if t == "stream_event":
        inner = obj.get("event")
        if isinstance(inner, dict) and isinstance(inner.get("type"), str) \
                and inner.get("type") in MESSAGES_EVENT_TYPES:
            yield inner
            return
        yield _wrapper(backend, "stream_event", obj)
        return

    # Other envelope lines → x_omni_<type> with original under "event".
    if isinstance(t, str) and t:
        yield _wrapper(backend, t, obj)
        return

    yield _passthrough(backend, event=obj)


def _normalize_one_codex(backend, obj, raw):
    """Yield one or more contract objects for a codex raw line."""
    if obj is None:
        yield _passthrough(backend, raw=raw)
        return

    t = obj.get("type")

    # item.completed agent_message → synthesize Messages stream once.
    if t == "item.completed":
        item = obj.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            for ev in _codex_agent_message_sequence(text):
                yield ev
            return

    # Everything else (including mystery types) → passthrough.
    yield _passthrough(backend, event=obj)


def normalize_lines(backend, lines):
    """Normalize backend JSONL lines into contract JSON strings.

    backend in {"grok","claude","codex","pi","cursor"}. Every input line
    yields >=1 output line. Never raises on garbage (x_omni_passthrough with
    raw).
    """
    backend = backend if isinstance(backend, str) else str(backend or "")
    if lines is None:
        return
    for line in lines:
        obj, raw = _parse_line(line)
        if raw is None:
            # Too deep to read (see _parse_line): a fixed notice for every
            # backend, nothing of the line, and the lines after it go on.
            yield _dumps({"type": "x_omni_omitted", "backend": backend,
                          "reason": "nesting_too_deep"})
            continue
        if backend == "codex":
            for out in _normalize_one_codex(backend, obj, raw):
                yield _dumps(out)
        elif backend == "pi":
            for out in _normalize_one_pi(backend, obj, raw):
                yield _dumps(out)
        elif backend in ("grok", "claude", "cursor"):
            # cursor's stream-json shape is claude-like (system/init, user,
            # thinking delta/completed, assistant{message.content[].text},
            # result{result,session_id,usage}) — reuse rather than duplicate.
            for out in _normalize_one_grok_claude(backend, obj, raw):
                yield _dumps(out)
        else:
            # Unknown backend: still never drop / never raise.
            if obj is None:
                yield _dumps(_passthrough(backend or "unknown", raw=raw))
            else:
                yield _dumps(_passthrough(backend or "unknown", event=obj))


def _final_text_grok_claude(lines, result_fallback=False):
    """Last assistant whole message, else the reassembled deltas.

    ``result_fallback`` (grok only) adds a last resort: when both are empty,
    the last ``type:"result"`` event's string ``result`` field (nothing when
    that event has none). It never outranks an assistant message or deltas.
    """
    last_assistant = None
    delta_parts = []
    result_text = None

    for line in lines or ():
        obj, _raw = _parse_line(line)
        if obj is None:
            continue
        t = obj.get("type")
        if t == "result":
            # Only the last result event counts; a later one without a string
            # "result" does not hand the fallback an earlier one's.
            r = obj.get("result")
            result_text = r if isinstance(r, str) else None
            continue
        if t == "assistant":
            text = _assistant_text_from_message(obj.get("message"))
            # Keep last assistant that has extractable whole-message text.
            if text:
                last_assistant = text
            continue
        if t == "stream_event":
            inner = obj.get("event")
            piece = _delta_text(inner)
            if piece:
                delta_parts.append(piece)
            continue
        if t in MESSAGES_EVENT_TYPES:
            piece = _delta_text(obj)
            if piece:
                delta_parts.append(piece)

    if last_assistant is not None:
        return last_assistant
    joined = "".join(delta_parts)
    if not joined and result_fallback and result_text:
        return result_text
    return joined


def _final_text_cursor(lines):
    """Like ``_final_text_grok_claude``, but a ``type:"result"`` event's own
    ``result`` field — the authoritative full answer, same field the sync
    lane reads off the single-object ``--output-format json`` shape — wins
    over the last assistant message when both are present. Falls back to
    the shared grok/claude logic (last assistant, else deltas) when no
    result event carries a string ``result``.
    """
    last_assistant = None
    delta_parts = []
    result_text = None

    for line in lines or ():
        obj, _raw = _parse_line(line)
        if obj is None:
            continue
        t = obj.get("type")
        if t == "result":
            r = obj.get("result")
            if isinstance(r, str):
                result_text = r
            continue
        if t == "assistant":
            text = _assistant_text_from_message(obj.get("message"))
            if text:
                last_assistant = text
            continue
        if t == "stream_event":
            inner = obj.get("event")
            piece = _delta_text(inner)
            if piece:
                delta_parts.append(piece)
            continue
        if t in MESSAGES_EVENT_TYPES:
            piece = _delta_text(obj)
            if piece:
                delta_parts.append(piece)

    if result_text is not None:
        return result_text
    if last_assistant is not None:
        return last_assistant
    return "".join(delta_parts)


def _final_text_pi(lines):
    """Last assistant message_end / turn_end text; else assembled text_delta."""
    last_assistant = None
    delta_parts = []
    for line in lines or ():
        obj, _raw = _parse_line(line)
        if obj is None:
            continue
        t = obj.get("type")
        if t in ("message_end", "turn_end"):
            text = _assistant_text_from_message(obj.get("message"))
            if text and isinstance(obj.get("message"), dict) \
                    and obj["message"].get("role") == "assistant":
                last_assistant = text
            continue
        if t == "message_update":
            ev = obj.get("assistantMessageEvent")
            if isinstance(ev, dict) and ev.get("type") == "text_delta":
                delta = ev.get("delta")
                if isinstance(delta, str) and delta:
                    delta_parts.append(delta)
            continue
        if t in MESSAGES_EVENT_TYPES:
            piece = _delta_text(obj)
            if piece:
                delta_parts.append(piece)
    if last_assistant is not None:
        return last_assistant
    return "".join(delta_parts)


def _final_text_codex(lines):
    last = ""
    for line in lines or ():
        obj, _raw = _parse_line(line)
        if obj is None:
            continue
        if obj.get("type") != "item.completed":
            continue
        item = obj.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str):
            last = text
        elif text is None:
            last = ""
        else:
            last = str(text)
    return last


def final_text(backend, lines):
    """Return the assistant's final answer text from raw backend lines.

    grok/claude: last assistant whole-message text if present, else reassemble
    content_block text deltas; grok alone then falls back to the last result
    event's string ``result``. codex: last item.completed agent_message text.
    Empty input -> "".
    """
    if not lines:
        return ""
    backend = backend if isinstance(backend, str) else str(backend or "")
    if backend == "codex":
        return _final_text_codex(lines)
    if backend == "pi":
        return _final_text_pi(lines)
    if backend == "cursor":
        return _final_text_cursor(lines)
    if backend in ("grok", "claude"):
        return _final_text_grok_claude(lines, result_fallback=backend == "grok")
    # Unknown backend: best-effort empty rather than raise.
    return ""
