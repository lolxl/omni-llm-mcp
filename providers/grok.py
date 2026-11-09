# no-opencc
"""Grok Build CLI provider — host-fenced text pipe (+ inline vision/text files)."""
import base64
import json
import os
import time
import re
import shutil
import tempfile
import uuid

from providers.cli_run import run_cli
from providers import provider_accounts
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.egress_gate import refuse_payload
from providers.config import (
    DEFAULT_GROK_MODEL,
    DEFAULT_TIMEOUT_S,
    ENV_DEPTH,
    FILES_IMAGE_MIME,
    FILES_TEXT_MIME,
    GROK_EXTRA_AUTH_MARKERS,
    GROK_FILES_MAX_PAYLOAD_BYTES,
    GROK_FILES_TEXT_MAX_BYTES,
    GROK_FILES_TEXT_TOTAL_MAX_BYTES,
    env_depth,
)
from providers.errors import (
    classify_cli_error,
    exception_detail,
    safe_detail,
    local_before_spawn,
    local_request_before_spawn,
    note_spawned,
    signal_exit_detail,
)
from providers.profiles import resolve_profile, resolve_account_home, spawn_env
from providers.source_load import open_binary_under_roots, read_source_file
from providers.usage import extract_cli_result
from providers import vendor_bin

# Resource-block filename sanitizer. The uri is rendered to the model, so a hostile
# source basename must not carry prompt lines into it — same reasoning as the
# synthesized names in providers/attachments.py.
_SAFE_URI_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_URI_STEM_LEN = 64
_MAX_URI_EXT_LEN = 16

# ONE stable error_code for every size-cap refusal this module raises (per-file
# inline text, total inline text, whole argv payload). The per-file case gets it
# from the shared source_load taxonomy, so the other two reuse it rather than
# inventing a second code for the same meaning — `reason` already distinguishes
# files_too_large from the source_path caps.
_SIZE_ERROR_CODE = "source_too_large"

# Consult lane is text-only. grok treats `--tools ""` as unset (all ~25 tools,
# shell and write included). A non-empty allowlist switches off default tool
# injection; the denylist then runs after it and also strips the two MCP
# discovery tools grok injects regardless. Live-verified on grok 1.0.41.
GROK_CONSULT_TOOL_ARGS = (
    "--tools", "todo_write",
    "--disallowed-tools", "todo_write,search_tool,use_tool",
)

# Evidence comes from grok's own NDJSON stdout, which omni reads as it is
# written: grok announces the offered tool set (`available_commands`) and
# every tool call before running it, and nothing the child does later can
# take a line back. Session files under $GROK_HOME are child-writable and
# are not used as evidence. Every other event type refuses (fail closed).
_CONSULT_SAFE_EVENTS = ("available_commands", "thought", "text", "usage", "end")

# The only fields each safe event may carry (grok 1.0.41 streaming-json, as
# captured from a real consult with inference blocked locally and from a
# tool-bearing run; re-captured on 1.0.52 through the fenced read-wall path
# and the plain path, which differ only in whether `thought` events appear).
# Anything else refuses: a new field is new behaviour.
# `commands` is grok's slash-command catalogue (plain names); it is present
# in every consult and is not a tool grant, so it may be non-empty, but only
# as a list of strings. `tools` must be present and empty.
# `cwd` (on `end`, since 1.0.52) is the working directory grok reports for the
# run, an echo of the --cwd this lane passed; it is a plain string, is never
# released or rendered, and names no tool, so it may be present but only as str.
_CONSULT_EVENT_FIELDS = {
    "available_commands": ({"type", "tools"}, {"commands"}),
    "thought": ({"type", "data"}, set()),
    "text": ({"type", "data"}, set()),
    "usage": ({"type", "usage"}, {"signature"}),
    "end": ({"type", "sessionId", "stopReason"},
            {"usage", "requestId", "num_turns", "total_cost_usd",
             "total_cost_usd_ticks", "modelUsage", "cwd"}),
}
# Integer counters copied into the envelope's usage; nothing else is.
_CONSULT_USAGE_COUNTERS = (
    "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "output_tokens", "reasoning_tokens", "total_tokens",
)


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_keys(pairs):
    """json object_pairs_hook: a repeated key at any depth is ambiguous
    (json.loads would silently keep the last one), so it refuses."""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateKey()
        obj[key] = value
    return obj


def _unverified(why):
    # `why` is always one of the fixed phrases below: stream content never
    # reaches a refusal (it is unverified reply material).
    return {"ok": False, "reason": "tool_containment_unverified",
            "detail": "grok consult stream not verified: %s; reply withheld" % why}


def _tools_failed(why):
    return {"ok": False, "reason": "tool_containment_failed",
            "detail": "grok consult %s; reply withheld" % why}


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _consult_result(out, session_id):
    """``(text, usage, refusal)`` from a consult's streaming-json stdout.

    Released only when every line is one JSON object without duplicate keys,
    every event is a known type carrying only its known fields, the stream
    announced an empty tool set, made no tool call, and ends with exactly one
    ``end`` event naming this run's pinned session with ``stopReason`` equal
    to ``end_turn``. Reasoning (``thought``) is never part of the text. The
    text is the exact concatenation of the ``text`` events (no strip). Events
    are split on ``\\n`` only: JSON strings may carry U+2028 and friends raw.
    Usage carries only whitelisted integer counters. Refusal details carry
    fixed phrases and counts, never stream strings.
    """
    events = []
    for line in (out or "").split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        except _DuplicateKey:
            return None, None, _unverified("duplicate key in stdout line")
        except ValueError:
            return None, None, _unverified("non-JSON stdout line")
        if not isinstance(event, dict):
            return None, None, _unverified("non-object stdout line")
        events.append(event)
    tool_events = sum(1 for e in events
                      if isinstance(e.get("type"), str) and e["type"].startswith("tool"))
    if tool_events:
        return None, None, _tools_failed("stream carried %d tool event(s)" % tool_events)
    announced = False
    for event in events:
        kind = event.get("type")
        if kind not in _CONSULT_SAFE_EVENTS:
            return None, None, _unverified("unexpected event type")
        required, optional = _CONSULT_EVENT_FIELDS[kind]
        if not required <= set(event) or set(event) - required - optional:
            return None, None, _unverified("unexpected event fields")
        if kind in ("text", "thought") and not isinstance(event["data"], str):
            return None, None, _unverified("event data is not a string")
        if kind == "end" and "cwd" in event and not isinstance(event["cwd"], str):
            return None, None, _unverified("end working directory is not a string")
        if kind in ("usage", "end") and "usage" in event \
                and not isinstance(event["usage"], dict):
            return None, None, _unverified("usage is not an object")
        if kind == "available_commands":
            tools = event["tools"]
            if not isinstance(tools, list):
                return None, None, _unverified("tool announcement without a list")
            if tools:
                return None, None, _tools_failed(
                    "was offered %d tool(s)" % len(tools))
            commands = event.get("commands", [])
            if not isinstance(commands, list) or \
                    not all(isinstance(c, str) for c in commands):
                return None, None, _unverified("command announcement is not a name list")
            announced = True
    if not announced:
        return None, None, _unverified("no tool-set announcement")
    ends = [e for e in events if e.get("type") == "end"]
    if len(ends) != 1 or events[-1] is not ends[0]:
        return None, None, _unverified("stream does not end with one end event")
    end = ends[0]
    if end["sessionId"] != session_id:
        return None, None, _unverified("end event names another session")
    if end["stopReason"] != "end_turn":
        return None, None, _unverified("end event stop reason is not end_turn")
    text = "".join(e["data"] for e in events if e.get("type") == "text")
    counters = {key: value for key, value in (end.get("usage") or {}).items()
                if key in _CONSULT_USAGE_COUNTERS and _is_count(value)}
    _ignored, usage = extract_cli_result(
        "grok", json.dumps({"text": text, "usage": counters}))
    return text, usage, None


def _accepted_extensions_detail():
    """Human-readable list of BOTH accepted classes (never drifts from the tables)."""
    return "images (%s) or text (%s)" % (
        "/".join(sorted(FILES_IMAGE_MIME)),
        "/".join(sorted(FILES_TEXT_MIME)),
    )


def _resource_uri(index, path):
    """`file:///<n>-<safe-name>` for a text resource block.

    The 1-based attachment index ALWAYS leads, so two files that share a basename
    but live in different directories stay distinguishable in the payload. The
    basename is sanitized (the uri is rendered to the model, so a hostile filename
    must not carry prompt lines) and truncated on the STEM only — the suffix
    survives, because it is the model's type hint for the block.
    """
    stem, ext = os.path.splitext(os.path.basename(str(path)))
    stem = _SAFE_URI_NAME_RE.sub("_", stem)[:_MAX_URI_STEM_LEN]
    ext = _SAFE_URI_NAME_RE.sub("_", ext)[:_MAX_URI_EXT_LEN]
    if not stem.strip("._-"):
        stem = "file"
    return "file:///%d-%s%s" % (index, stem, ext)


def _build_acp_payload(prompt, files):
    """prompt + image/text files -> (json_payload, None) or (None, error_envelope).

    ACP content blocks: one text block, then one block per file in caller order —
    images as base64 `image` blocks, whitelisted text as `resource` blocks with the
    decoded body inline. Every file is read HERE (server side, under the
    OMNI_LLM_SOURCE_ROOTS fence) because the grok sandbox cannot reach them —
    `--sandbox strict` blocks reads outside cwd.

    Extension classification runs BEFORE the fenced read, so an unusable type is
    reported as such instead of as a path/allowlist problem.
    """
    content = [{"type": "text", "text": str(prompt)}]
    text_total = 0
    for index, f in enumerate(files, start=1):
        ext = os.path.splitext(str(f))[1].lower()
        image_mime = FILES_IMAGE_MIME.get(ext)
        text_mime = FILES_TEXT_MIME.get(ext)
        if image_mime:
            data, err = open_binary_under_roots(f, GROK_FILES_MAX_PAYLOAD_BYTES)
            if err:
                # Allowlist unset / outside roots / symlink / too large: surface the
                # source_path taxonomy unchanged (stable error_code for callers).
                return None, err
            content.append({
                "type": "image",
                "data": base64.b64encode(data).decode("ascii"),
                "mimeType": image_mime,
            })
            continue
        if text_mime:
            body, err = read_source_file(f, max_bytes=GROK_FILES_TEXT_MAX_BYTES)
            if err:
                code = err.get("error_code")
                if code == _SIZE_ERROR_CODE:
                    # Distinct reason: this is OUR inline-text cap, not the shared
                    # 12 MiB source_path cap the taxonomy usually means.
                    return None, {
                        "ok": False, "reason": "files_too_large",
                        "error_code": code,
                        "detail": "grok inline text cap %d bytes per file exceeded "
                                  "by %r; split the file or use codex"
                                  % (GROK_FILES_TEXT_MAX_BYTES, f),
                    }
                if code == "source_invalid_utf8":
                    # Loud, never a silent skip: the extension promised text.
                    return None, {
                        "ok": False, "reason": "unsupported_files",
                        "error_code": code,
                        "detail": "grok text attachment %r is not valid UTF-8" % f,
                    }
                return None, err
            text_total += len(body.encode("utf-8"))
            if text_total > GROK_FILES_TEXT_TOTAL_MAX_BYTES:
                return None, {
                    "ok": False, "reason": "files_too_large",
                    "error_code": _SIZE_ERROR_CODE,
                    "detail": "grok inline text total cap %d bytes exceeded; "
                              "send fewer/smaller text files or use codex"
                              % GROK_FILES_TEXT_TOTAL_MAX_BYTES,
                }
            content.append({
                "type": "resource",
                "resource": {
                    "uri": _resource_uri(index, f),
                    "mimeType": text_mime,
                    "text": body,
                },
            })
            continue
        return None, {"ok": False, "reason": "unsupported_files",
                      "detail": "grok files= accepts %s; got %r"
                                % (_accepted_extensions_detail(), f)}

    # ensure_ascii=False (the house style everywhere else in this repo): with the
    # default ASCII escaping every non-ASCII character becomes a 6-byte \uXXXX
    # sequence, so CJK text inflates ~2x on the wire and two 249 KB zh-TW files
    # that pass BOTH inline-text caps still blow the argv cap. The value is passed
    # to the CLI as an argv element, not a byte stream, so no encoding is lost.
    payload = json.dumps({"type": "acp", "content": content}, ensure_ascii=False)
    size = len(payload.encode("utf-8"))
    if size > GROK_FILES_MAX_PAYLOAD_BYTES:
        kinds = {b.get("type") for b in content}
        if "resource" in kinds:
            # Text-bearing payload: this is a size problem, not a type problem, and
            # the advice must not talk about images that may not be there.
            return None, {
                "ok": False, "reason": "files_too_large",
                "error_code": _SIZE_ERROR_CODE,
                "detail": "grok inline payload cap %d bytes exceeded (%d) with "
                          "prompt + attachments; send fewer/smaller files or "
                          "use codex" % (GROK_FILES_MAX_PAYLOAD_BYTES, size),
            }
        return None, {"ok": False, "reason": "unsupported_files",
                      "error_code": _SIZE_ERROR_CODE,
                      "detail": "grok inline payload cap %d bytes exceeded (%d); "
                                "use agy or codex for large images"
                                % (GROK_FILES_MAX_PAYLOAD_BYTES, size)}
    return payload, None


def _texts_for_gate(prompt, prompt_json):
    """Prompt plus inlined ACP text/resource bodies and resource uris.

    Resource ``uri`` is rendered to the model and is built from the file
    basename, so a pattern in the filename is part of the egress payload.
    Image block bytes are never scanned.
    """
    parts = [str(prompt)]
    if not prompt_json:
        return parts
    try:
        obj = json.loads(prompt_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        parts.append(str(prompt_json))
        return parts
    for block in obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
        resource = block.get("resource")
        if isinstance(resource, dict):
            ruri = resource.get("uri")
            if isinstance(ruri, str) and ruri:
                parts.append(ruri)
            rtext = resource.get("text")
            if isinstance(rtext, str) and rtext:
                parts.append(rtext)
    return parts


@local_before_spawn
async def _run_grok_once(prompt, model, timeout_s, account_profile="default",
                              files=None, effort=None, scrub_siblings=False, account_home=None, account_auth_isolated=False, account_deadline=None,
                              _spawn=None):
    """Pipe prompt -> `grok --prompt-file` (headless single-turn) -> typed envelope.

    On macOS, the shared host fence is the process boundary. After that wrap
    verifies, the helper changes Grok's nested --sandbox flag to off; on other
    platforms the original --sandbox strict argv and environment are retained.
    Tools: GROK_CONSULT_TOOL_ARGS empty the tool set (empty --tools alone is
    unset). Output is streaming-json with a pinned --session-id; nothing from
    stdout is released (envelope, error detail or watch pane) unless
    _consult_result verifies the stream, else the reply is withheld (LOUD).
    Global instructions ($GROK_HOME/AGENTS.md, ~/.claude/CLAUDE.md, rules/)
    are kept out by the host fence's read deny, not by argv.

    `files` (attachments): local IMAGE or whitelisted TEXT files under
    OMNI_LLM_SOURCE_ROOTS. The server reads them (TOCTOU-safe openat + O_NOFOLLOW)
    and ships prompt + attachments as ONE inline `--prompt-json` ACP payload —
    images base64'd into `image` blocks, text decoded into `resource` blocks — with
    no temp prompt file, since --prompt-file and --prompt-json are mutually
    exclusive in the grok CLI. The payload rides argv, hence the
    GROK_FILES_MAX_PAYLOAD_BYTES cap (plus the tighter inline-text caps).
    No files -> byte-identical to the text-only `--prompt-file` path.
    """
    files = list(files or [])
    prompt_json = None
    if files:
        # Argument validation first: environment errors (cli_missing) come after.
        prompt_json, files_err = _build_acp_payload(prompt, files)
        if files_err:
            return files_err

    overrides, profile_err = (resolve_account_home("grok", account_profile, account_home)
                              if account_home else resolve_profile("grok", account_profile))
    if profile_err:
        return profile_err
    grok = vendor_bin.which("grok")
    if not grok:
        return {"ok": False, "reason": "cli_missing",
                "detail": "grok not on PATH - install Grok Build CLI, then `grok login`"}
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    model = str(model or "").strip() or DEFAULT_GROK_MODEL
    tmp_prompt = None
    tmp_cwd = None
    try:
        if prompt_json is not None:
            prompt_args = ["--prompt-json", prompt_json]
        else:
            fd, tmp_prompt = tempfile.mkstemp(suffix=".txt", prefix="grok_prompt_")
            with os.fdopen(fd, "w") as f:
                f.write(str(prompt))
            prompt_args = ["--prompt-file", tmp_prompt]
        tmp_cwd = tempfile.mkdtemp(prefix="grok_cwd_")
        session_id = str(uuid.uuid4())

        argv = [
            grok, *prompt_args,
            "--model", model,
        ]
        if effort:
            argv += ["--reasoning-effort", str(effort)]
        argv += [
            "--output-format", "streaming-json",
            "--no-memory",
            # Consult has no file tools to recover offloaded prompt content.
            "--verbatim",
            "--disable-web-search",
            *GROK_CONSULT_TOOL_ARGS,
            "--session-id", session_id,
            "--sandbox", "strict",
            "--cwd", tmp_cwd,
        ]
        denied = refuse_payload(*_texts_for_gate(prompt, prompt_json))
        if denied:
            return denied
        child_env = spawn_env({**os.environ, ENV_DEPTH: str(env_depth() + 1)}, overrides,
                              scrub_siblings=scrub_siblings)
        if account_auth_isolated:
            child_env = provider_accounts.account_env(child_env, "grok")
        if account_home:
            child_env["GROK_HOME"] = account_home
        config_root = provider_accounts.fence_home("grok", account_home) if account_home else overrides.get("GROK_HOME")
        if account_home and config_root is None:
            child_env.pop("GROK_HOME", None)
        try:
            argv, child_env, _fence_meta = prepare_host_fence(
                argv, provider="grok", workdir=tmp_cwd,
                read_only=True, env=child_env,
                extra_read_paths=(tmp_prompt,) if tmp_prompt else (),
                config_dir=config_root,
            )
        except HostFenceError as exc:
            return exc.envelope
        if account_home:
            if not provider_accounts.home_matches("grok", account_profile, account_home):
                return {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
            child_env["GROK_HOME"] = account_home
        if account_deadline is not None:
            timeout_s = max(0.0, account_deadline - time.monotonic())
            if timeout_s <= 0:
                return {"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"}
        res = await run_cli(
            argv,
            cwd=tmp_cwd,
            env=child_env,
            timeout_s=timeout_s,
            pane_name="omni:grok",
            watch_stdout=False,
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "grok exceeded %ss" % timeout_s}

        out = res.get("stdout") or ""   # unstripped: the verdict owns parsing
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")

        # stdout is the unverified reply stream: on failure it may only steer
        # the reason enum, never reach the detail, including signal failures.
        process_error = provider_accounts.process_failure({**res, "stdout": ""}, "grok")
        if process_error:
            return process_error
        structured = provider_accounts.structured_failure("grok", out)
        if structured:
            return {"ok": False, "reason": structured, "detail": "grok account failure",
                    "account_retry_safe": False}
        if rc != 0:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = signal_exit_detail("grok", -res["signal"], "", err)
            else:
                reason = classify_cli_error(
                    provider_accounts.account_error_text(out), err, rc,
                    extra_auth_markers=GROK_EXTRA_AUTH_MARKERS,
                )
                if reason == "signal_exit":
                    detail = signal_exit_detail("grok", rc, "", err)
                else:
                    detail = safe_detail(err or "exit %d" % (
                        rc if rc is not None else -1))
            return {"ok": False, "reason": reason, "detail": detail}
        text, usage, refusal = _consult_result(out, session_id)
        if refusal:
            return refusal
        # See providers/agy.py: judge emptiness after extraction, not on raw
        # stdout. Whitespace-only is empty; otherwise the text is released exact.
        if not text.strip():
            return {"ok": False, "reason": "empty_response",
                    "detail": safe_detail(err or "grok returned no text")}
        return {"ok": True, "text": text, "model": model, "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}
    finally:
        if tmp_prompt and os.path.exists(tmp_prompt):
            try:
                os.remove(tmp_prompt)
            except OSError:
                pass
        if tmp_cwd and os.path.isdir(tmp_cwd):
            shutil.rmtree(tmp_cwd, ignore_errors=True)


@local_request_before_spawn()
async def run_grok_subprocess(prompt, model, timeout_s, account_profile="default",
                              files=None, effort=None, scrub_siblings=False, account_pinned=False):
    selection = provider_accounts.AccountSelection(
        account_profile, str(model) if str(model).startswith("grok") else "grok/" + str(model),
        timeout_s, provider="grok", resume=account_pinned)
    if selection.error:
        return selection.exhausted()
    isolated = selection.managed or provider_accounts.ring_pinned(
        "grok", account_profile, account_pinned)
    async def attempt(account, remaining, managed):
        return await _run_grok_once(
            prompt, model, remaining, account["account_profile"], files=files, effort=effort,
            scrub_siblings=scrub_siblings or isolated,
            account_home=account["account_home"] if managed or isolated else None,
            account_auth_isolated=isolated, account_deadline=selection.deadline)
    return await provider_accounts.run_selected(selection, attempt)
