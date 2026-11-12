#!/usr/bin/env python3
# no-opencc
"""omni-llm-mcp - async MCP stdio server wrapping agy, OpenAI, Grok, and Local LLMs.

Runs as `omni mcp` (installed CLI) or `python3 server.py` (plugins and clones;
server.py is a thin shim over this module).

A POLICY-FREE multi-model pipe: exposes models from `agy`, OpenAI, Grok Build CLI,
or local OpenAI-compatible servers as MCP tools.

Provider implementations live under `providers/` (still zero third-party deps).
This file owns JSON-RPC framing, concurrency, and stdout serialization.

DESIGN CONTRACT
- POLICY-FREE PIPE: text in -> text out. The server does NOT inspect or filter content.
- LOUD STRUCTURED FAILURE: every failure returns {ok:false, reason:<typed>, detail}. Never silent.
- ADVISORY + UNTRUSTED: output is never authoritative; consumers must re-sanitize and ground it.
"""
import asyncio
import collections
import json
import sys
import time

from providers import known_secrets, metrics
from providers.config import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    HAS_SDK,
    LocalAgentConfig,
    ROUTE_RULES,
    AUTH_MARKERS,
    RATE_LIMIT_MARKERS,
    GROK_EXTRA_AUTH_MARKERS,
    HTTP_STATUS_REASONS,
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_BACKOFF_S,
)
from providers.errors import (
    classify_cli_error,
    classify_exception_message,  # noqa: F401 -- re-exported (test_server)
    exception_detail,
    mask_envelope,
    redact_secrets,
    reason_from_http_status,
)
from providers.route import resolve_route, provider_for_model as _provider_for_model
from providers.route import provider_supports_files as _provider_supports_files
from providers.logutil import log as _log, log_event as _log_event
from providers.openai_compat import run_openai_compatible as _run_openai_compatible
from providers.agy import (
    run_agy as _run_agy,
    run_agy_subprocess as _run_agy_subprocess,
    run_agy_sdk as _run_agy_sdk,
    build_agy_sdk_config as _build_agy_sdk_config,
    apply_agy_account_profile as _apply_agy_account_profile,
    restore_agy_account_profile as _restore_agy_account_profile,
)
from providers.grok import run_grok_subprocess as _run_grok_subprocess
from providers.codex import run_codex_subprocess as _run_codex_subprocess
from providers.dispatch import (
    TOOLS,
    call_tool as _call_tool,
    call_budget_s,
    route_to_provider as _route_to_provider,
    dispatch_single as _dispatch_single,
    dispatch_llm_request as _dispatch_llm_request,
    list_models as _list_models,
    backends_snapshot as _backends_snapshot,
)

from providers.version import SERVER_NAME, __version__ as SERVER_VERSION  # noqa: E402
PROTOCOL_VERSION = "2025-06-18"
# Newest first. PROTOCOL_VERSION stays the legacy default for existing callers.
# Modern `_meta` path + server/discover advertise this full set.
SUPPORTED_PROTOCOL_VERSIONS = ("2026-07-28", "2025-06-18")
# Handshake-era versions `initialize` may negotiate. 2026-07-28 removed the
# handshake, so it must never be echoed as an accepted initialize version.
LEGACY_NEGOTIABLE_VERSIONS = ("2025-06-18",)
# tools/list and server/discover are static for a process lifetime.
RESULT_TTL_MS = 300000  # 5 minutes
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
STDIN_LIMIT_BYTES = 16 * 1024 * 1024
MAX_CONCURRENT_TASKS = 5
# Shutdown drain. The wait is derived per-call from call_budget_s() rather than
# fixed, because `timeout_s` is caller-supplied and a comma model chain multiplies
# it — a flat constant silently cancelled long calls that were still within budget.
DRAIN_MARGIN_S = 30    # teardown headroom on top of the slowest in-flight budget
DRAIN_MAX_S = 900      # hard ceiling so one huge timeout_s cannot wedge shutdown

# Both are built lazily, on the running loop. On Python 3.9 `asyncio.Semaphore()`
# calls `get_event_loop()`, so constructing one at MODULE level made importing this
# file fail outright if anything in the process had already run and closed a loop —
# `import server` inside a function, after an `asyncio.run(...)`, raised "There is
# no current event loop". `_write_lock` was already deferred for the same reason;
# the semaphore now matches it.
_semaphore = None
_write_lock = None
_pending_tasks = set()
_task_budgets = {}
# Issue #43: observability seam for no-op notification handlers. Distinguishes
# "dispatched and intentionally silent" from "never reached the handler". Tests
# clear and assert this deque; production traffic only appends a fixed-size
# shape summary (no live client params retained).
_notification_dispatch_log = collections.deque(maxlen=64)


def _get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    return _semaphore


def _get_write_lock():
    global _write_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    return _write_lock


def _record_notification_dispatch(method, params):
    """Pin that a known notification method was dispatched (issue #43).

    Handler bodies stay no-ops; this records target + params shape so a routing
    bug cannot look like correct silence. Also LOUD on stderr via _log_event so
    subprocess tests can assert the same seam without in-process state.
    """
    record = {
        "method": method,
        "has_params": isinstance(params, dict),
        "param_keys": sorted(params) if isinstance(params, dict) else None,
    }
    _notification_dispatch_log.append(record)
    _log_event("notification_dispatched", method=method)


def _track_task(task, budget_s=DEFAULT_TIMEOUT_S):
    """Register an in-flight tools/call task (with its budget) for EOF drain."""
    _pending_tasks.add(task)
    _task_budgets[task] = budget_s
    task.add_done_callback(_pending_tasks.discard)
    task.add_done_callback(lambda t: _task_budgets.pop(t, None))
    return task


def _drain_budget_s():
    """How long the drain should wait: slowest in-flight budget + teardown margin."""
    budgets = [b for t, b in _task_budgets.items() if not t.done()]
    base = max(budgets) if budgets else DEFAULT_TIMEOUT_S
    return min(base + DRAIN_MARGIN_S, DRAIN_MAX_S)


async def _send(obj):
    """Write one JSON-RPC line to stdout under a lock (stdio framing must not interleave)."""
    line = json.dumps(obj) + "\n"
    async with _get_write_lock():
        sys.stdout.write(line)
        sys.stdout.flush()


async def _result(id_, result):
    # Central defaults: every result MUST carry resultType; servers SHOULD put
    # serverInfo in every result's _meta. Never clobber an explicit caller value.
    if "resultType" not in result or "_meta" not in result:
        result = dict(result)
        if "resultType" not in result:
            result["resultType"] = "complete"
        if "_meta" not in result:
            result["_meta"] = _server_info_meta()
    await _send({"jsonrpc": "2.0", "id": id_, "result": result})


async def _error(id_, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    await _send({"jsonrpc": "2.0", "id": id_, "error": err})


def _server_info_meta():
    return {
        META_SERVER_INFO: {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _is_modern_request(params):
    """Modern era when `_meta` carries one of the modern protocol fields.

    Presence of `_meta` alone is NOT the discriminator: `_meta.progressToken` is
    ordinary legacy usage, and treating it as modern would reject every legacy
    client that opts into progress notifications. Once a modern field appears the
    request is modern, so a `_meta` carrying only one of the required pair still
    fails validation with -32602 rather than being served as legacy.

    Non-dict params are not modern (era detection is not the place to reject
    malformed JSON-RPC params; leave existing downstream behavior alone).
    """
    if not isinstance(params, dict):
        return False
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return False
    return META_PROTOCOL_VERSION in meta or META_CLIENT_CAPABILITIES in meta


async def _validate_modern_meta(id_, params):
    """Modern validation order. Returns True if the request may proceed."""
    meta = params.get("_meta") or {}
    # 1. protocolVersion missing → -32602 (only when _meta exists without it —
    #    no-_meta is legacy and never reaches here).
    if META_PROTOCOL_VERSION not in meta:
        await _error(id_, -32602, "Invalid params: missing %s" % META_PROTOCOL_VERSION)
        return False
    # 1b. protocolVersion present but wrong type → -32602 (malformed required
    #     field, not an unsupported version — must not reach -32022).
    requested = meta.get(META_PROTOCOL_VERSION)
    if not isinstance(requested, str):
        await _error(id_, -32602, "Invalid params: %s must be a string" % META_PROTOCOL_VERSION)
        return False
    # 2. clientCapabilities missing → -32602
    if META_CLIENT_CAPABILITIES not in meta:
        await _error(id_, -32602, "Invalid params: missing %s" % META_CLIENT_CAPABILITIES)
        return False
    # 2b. clientCapabilities present but wrong type → -32602
    caps = meta.get(META_CLIENT_CAPABILITIES)
    if not isinstance(caps, dict):
        await _error(
            id_, -32602,
            "Invalid params: %s must be an object" % META_CLIENT_CAPABILITIES,
        )
        return False
    # 3. unsupported version → -32022
    if requested not in SUPPORTED_PROTOCOL_VERSIONS:
        await _error(
            id_, -32022, "Unsupported protocol version",
            data={
                "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
                "requested": known_secrets.mask_known(requested),
            },
        )
        return False
    return True


async def handle_tool_call(id_, name, args):
    """Background task to handle a tool call concurrently without blocking the stdin reader."""
    t0 = time.monotonic()
    async with _get_semaphore():
        try:
            payload = await _call_tool(name, args)
        except Exception as e:
            payload = {
                "ok": False,
                "reason": "runtime_error",
                "detail": exception_detail(e),
            }
    # The egress: failure text is masked once more where the envelope leaves
    # the process, whichever site built it. Successful `text` is untouched.
    payload = mask_envelope(payload)

    latency_ms = int((time.monotonic() - t0) * 1000)
    _log_event(
        "tool_call",
        tool=name,
        ok=bool(payload.get("ok")),
        reason=payload.get("reason"),
        model=payload.get("model") or (args or {}).get("model"),
        latency_ms=latency_ms,
    )
    # structuredContent is the same typed envelope serialized into the text block
    # so the compatibility mirror is exact by construction.
    await _result(id_, {
        "resultType": "complete",
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "structuredContent": payload,
        "isError": not payload.get("ok", False),
    })


async def handle_message(msg):
    """Process one decoded JSON-RPC message."""
    # MCP stdio is one object per line. Batches / scalars / null are not objects:
    # refuse with -32600 and keep the read loop alive (never let msg.get raise).
    if not isinstance(msg, dict):
        await _error(None, -32600, "Invalid Request: expected a JSON object")
        return

    method = msg.get("method")
    id_ = msg.get("id")
    # JSON-RPC 2.0 defines a notification as a request that OMITS the id member.
    # `"id": null` is therefore NOT a notification — it is a request carrying an
    # illegal id (MCP narrows RequestId further, to string|number). Refuse it
    # rather than swallowing it the way an absent id is swallowed.
    # ...and MCP narrows RequestId further, to string|number. Enforce exactly what
    # the error message promises: booleans, objects and arrays were previously
    # accepted and echoed straight back. `isinstance(True, int)` is why bool is
    # excluded explicitly; 0 and negative numbers are legal ids and must pass.
    if "id" in msg and not (isinstance(id_, (str, int))
                            and not isinstance(id_, bool)):
        await _error(None, -32600, "Invalid Request: id must be a string or number")
        return
    has_id = id_ is not None

    # Absent / explicit-null params → {}. Non-object params are left as-is so era
    # detection stays a no-op (#22 F1: tools/list with list/string params still
    # works). Methods that actually read fields (tools/call) reject non-objects
    # with -32602 below — do not global-reject here.
    raw_params = msg.get("params")
    if raw_params is None:
        params = {}
    else:
        params = raw_params

    # Per-request era: modern iff _meta carries a modern protocol field. Stateless —
    # never infer era from a previous message on the same connection.
    # server/discover is the stdio backward-compat probe: no _meta → DiscoverResult;
    # when it carries a modern marker, run the same modern validation as other methods.
    if _is_modern_request(params):
        # Notifications cannot be answered with an error: skip validation and fall
        # through to normal dispatch instead of returning early.
        if has_id:
            if not await _validate_modern_meta(id_, params):
                return

    if method == "initialize":
        if not has_id:
            return
        # MCP's InitializeRequest.params is a REQUIRED object. Coercing a list or a
        # string to {} used to complete the handshake against a client that never
        # sent one. Reject at the method that reads the field, not globally —
        # test_protocol_2026 F1 pins that era detection must not break tools/list.
        if not isinstance(params, dict):
            await _error(id_, -32602, "Invalid params: params must be an object")
            return
        # Echo only a handshake-era negotiable version; otherwise answer with our
        # legacy default. 2026-07-28 removed the handshake and must not be negotiated here.
        client_pv = params.get("protocolVersion")
        if client_pv in LEGACY_NEGOTIABLE_VERSIONS:
            accepted = client_pv
        else:
            accepted = PROTOCOL_VERSION
        await _result(id_, {
            "protocolVersion": accepted,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    elif method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        # No-op body (MCP defines no reply). Record dispatch so tests can tell
        # arrival from silent drop (issue #43).
        _record_notification_dispatch(method, params)
    elif method == "ping":
        if not has_id:
            return
        await _result(id_, {})
    elif method == "server/discover":
        if not has_id:
            return
        await _result(id_, {
            "resultType": "complete",
            "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
            "capabilities": {"tools": {}},
            "_meta": _server_info_meta(),
            "ttlMs": RESULT_TTL_MS,
            "cacheScope": "public",
        })
    elif method == "tools/list":
        if not has_id:
            return
        await _result(id_, {
            "resultType": "complete",
            "tools": TOOLS,
            "ttlMs": RESULT_TTL_MS,
            "cacheScope": "public",
        })
    elif method == "tools/call":
        # tools/call requires object params (and object arguments). Reject with
        # -32602 rather than letting .get / call_budget_s raise -32603 + leak text.
        if not isinstance(params, dict):
            if has_id:
                await _error(id_, -32602, "Invalid params: params must be an object")
            return
        name = params.get("name")
        # Absent / null arguments → {}; non-object → -32602 (never leak AttributeError).
        raw_args = params.get("arguments")
        if raw_args is None:
            args = {}
        elif not isinstance(raw_args, dict):
            if has_id:
                await _error(id_, -32602, "Invalid params: arguments must be an object")
            return
        else:
            args = raw_args
        if not isinstance(name, str):
            if has_id:
                await _error(id_, -32602, "Invalid params: name must be a string")
            return
        # The budget MUST be computed before the task exists. Python evaluates
        # `create_task(...)` before the `budget_s=` keyword, so a raise in
        # call_budget_s used to leave an already-scheduled task behind: the client
        # got a -32603 leaking the exception text AND, later, a second response
        # under the same id. `timeout_s: 1e309` decodes to inf and int(inf) raises.
        try:
            budget = call_budget_s(name, args)
        except Exception as e:
            # LOUD in the log either way. A notification with an unusable budget
            # used to return here without a wire response AND without a log line,
            # which is the one combination that leaves no trace anywhere.
            _log_event("tool_call_bad_budget", tool=name,
                       detail=type(e).__name__)
            if has_id:
                await _error(id_, -32602,
                             "Invalid params: arguments contain an unusable value")
            return
        # Notification-shaped tools/call: validated, but MCP defines no such
        # notification, so nothing is spawned. LOUD in the log, silent on the wire.
        if not has_id:
            _log_event("tool_call_notification_dropped", tool=name)
            return
        _track_task(asyncio.create_task(handle_tool_call(id_, name, args)),
                    budget_s=budget)
    else:
        if has_id:
            await _error(id_, -32601, "method not found: %s" % method)


async def _drain_pending(timeout_s=None):
    """Wait for in-flight tools/call tasks after stdin EOF (or cancel leftovers)."""
    pending = {t for t in _pending_tasks if not t.done()}
    if not pending:
        return
    if timeout_s is None:
        timeout_s = _drain_budget_s()
    _log_event("drain_start", pending=len(pending), timeout_s=timeout_s)
    done, still = await asyncio.wait(pending, timeout=timeout_s)
    if still:
        _log_event("drain_timeout", cancelling=len(still))
        for t in still:
            t.cancel()
        await asyncio.gather(*still, return_exceptions=True)
    _log_event("drain_done", completed=len(done), cancelled=len(still))


async def main_loop():
    _log("ready (model default: %s) [SDK: %s]" % (DEFAULT_MODEL, "enabled" if HAS_SDK else "fallback"))
    _log_event("ready", default_model=DEFAULT_MODEL, sdk=bool(HAS_SDK), version=SERVER_VERSION)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=STDIN_LIMIT_BYTES)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line_bytes = await reader.readline()
        except ValueError as e:
            _log("stdin line exceeded %d bytes; closing session:" % STDIN_LIMIT_BYTES, e)
            break
        except Exception as e:
            _log("stdin read error:", e)
            break

        if not line_bytes:
            break

        line = line_bytes.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except Exception as e:
            _log("parse error:", e)
            await _error(None, -32700, "parse error: %s" % redact_secrets(str(e))[:100])
            continue

        try:
            await handle_message(msg)
        except Exception as e:
            _log("handler error:", e)
            # Guard: msg may be a non-dict here. handle_message's own type check
            # returns early, but it can still RAISE while handling a non-dict — the
            # `_error(None, -32600, ...)` it writes goes through _send, and a broken
            # stdout pipe raises from there. Without this isinstance, that raise met
            # `msg.get("id")` on a list and killed the read loop before
            # `_drain_pending` ever ran. (Kept deliberately: a 5-mutant replay left
            # it alive, which measures the pin, not the guard's value.)
            if isinstance(msg, dict) and msg.get("id") is not None:
                await _error(msg.get("id"), -32603, "internal error: %s" % redact_secrets(str(e))[:200])

    await _drain_pending()


def main():
    # Explicit entry kind: under `omni mcp` the process looks like the CLI to
    # metrics.detect_entry(), but it is the MCP server.
    previous = metrics._entry_override
    metrics.set_entry("mcp")
    try:
        known_secrets.prime()
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
    finally:
        metrics.set_entry(previous)


if __name__ == "__main__":
    main()
