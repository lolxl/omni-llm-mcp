# no-opencc
"""Tool schemas, provider routing, fallback chain, and list_models."""
import asyncio
import os
import shutil
import time

from providers import agy as agy_mod
from providers import known_secrets
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import grok as grok_mod
from providers import pi as pi_mod
from providers import cursor as cursor_mod
from providers import grok_image_cli as grok_image_cli_mod
from providers import openai_compat as openai_mod
from providers import xai_image as xai_image_mod
from providers import artifacts as artifacts_mod
from providers import health
from providers.host_fence import HostFenceError
from providers import kernel_sandbox
from providers import metrics as metrics_mod
from providers.config import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    BIG_CONTEXT_TIMEOUT_S,
    DEFAULT_MODELS_TIMEOUT_S,
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_BACKOFF_S,
    HAS_SDK,
    ENV_SOURCE_ROOTS,
    ENV_DEPTH,
    FILES_IMAGE_MIME,
    FILES_TEXT_MIME,
    GROK_FILES_MAX_PAYLOAD_BYTES,
    GROK_FILES_TEXT_MAX_BYTES,
    GROK_FILES_TEXT_TOTAL_MAX_BYTES,
    env_depth,
    effort_for_provider,
    cursor_effort_slug,
)
from providers.cli_run import set_watch_context, reset_watch_context, _UNSET
from providers.egress_gate import refuse_payload
from providers.errors import (
    classify_cli_error,
    exception_detail,
    is_local_failure,
    mark_local,
    redact_secrets,
    safe_detail,
    strip_local_marker,
)
from providers.logutil import log_event
from providers.family import expand as expand_family, is_family
from providers.profiles import (
    RING_ROADS,
    resolve_profile,
    _load_profile_rings,
    account_registry_scope,
)
from providers.route import resolve_route, provider_supports_files
from providers.seat_name import retired_agy_opus, retired_model_error, take_effort
from providers import admission
from providers import provider_accounts
from providers.source_load import resolve_big_context_source
from providers import vendor_bin


def _health_snapshot():
    try:
        snap = health.snapshot()
    except Exception as exc:
        log_event("health_error", operation="snapshot", detail=str(exc)[:300])
        return {}
    stripped = {}
    for key, value in snap.items():
        if not isinstance(value, dict):
            stripped[key] = value
            continue
        entry = dict(value)
        entry.pop("account", None)
        stripped[key] = entry
    return stripped


def _with_health(envelope):
    result = dict(envelope)
    result["health"] = _health_snapshot()
    return result


def _health_flag(model, profile, until=None, source="rate_limit"):
    try:
        health.flag(model, profile, until=until, source=source)
    except Exception as exc:
        log_event("health_error", operation="flag", model=model,
                  detail=str(exc)[:300])


def _health_clear(model, profile):
    try:
        health.clear(model, profile, keep_sources=health.SUCCESS_KEEP_SOURCES)
    except Exception as exc:
        log_event("health_error", operation="clear", model=model,
                  detail=str(exc)[:300])


def _health_note_failure(model, profile, reason):
    try:
        health.note_failure(model, profile, reason)
    except Exception as exc:
        log_event("health_error", operation="note_failure", model=model,
                  detail=str(exc)[:300])


def _health_active(model, profile):
    try:
        return bool(health.active(model, profile))
    except Exception as exc:
        log_event("health_error", operation="active", model=model,
                  detail=str(exc)[:300])
        return False


def _health_passive_probe(model, profile):
    try:
        return bool(health.codex_passive_probe(model, profile))
    except Exception as exc:
        log_event("health_error", operation="passive_probe", model=model,
                  detail=str(exc)[:300])
        return False


def _health_maintain():
    try:
        health.maintain()
    except Exception as exc:
        log_event("health_error", operation="maintain", detail=str(exc)[:300])


def _health_reprobe(model, profile):
    try:
        health.reprobe(model, profile)
    except Exception as exc:
        log_event("health_error", operation="reprobe", model=model,
                  detail=str(exc)[:300])


def _health_kind(model, profile):
    try:
        return health.seat_kind(model, profile)
    except Exception as exc:
        log_event("health_error", operation="seat_kind", model=model,
                  detail=str(exc)[:300])
        return "healthy"


def _walk_will_dispatch(kind, probe_taken):
    if kind == "healthy":
        return True
    if kind == "probing" and not probe_taken:
        return True
    return False

# Shared optional args for CLI watch panes (llm_generate / llm_big_context).
# Static zellij setup prose lives in README; keep only call-time semantics here.
_WATCH_PANE_PROPS = {
    "watch_pane": {
        "type": "boolean",
        "description": (
            "If true, open a zellij floating pane that tails CLI stdout/stderr "
            "(agy/grok/codex). Requires zellij_tab or OMNI_LLM_ZELLIJ_TAB. "
            "false overrides OMNI_LLM_WATCH_PANE=1. PUBLIC stream only — output may hit a temp log. HTTP/SDK no-op."
        ),
    },
    "zellij_session": {
        "type": "string",
        "description": (
            "zellij session name for watch_pane. Fail-closed if invalid (no downgrade). "
            "Else OMNI_LLM_ZELLIJ_SESSION / weak single-session inference."
        ),
    },
    "zellij_tab": {
        "type": "string",
        "description": (
            "zellij tab NAME for watch_pane. Required to open a pane (no active-tab fallback). "
            "Fail-closed if missing/invalid. Else OMNI_LLM_ZELLIJ_TAB."
        ),
    },
}


class _BackendCap(dict):
    """Per-backend capability map.

    ``agent_fence`` / ``agent_fence_switch`` are advertised via item access
    and JSON; historical schema-pin tests iterate the original keys.
    """
    _ADVERTISED_EXTRA = frozenset({"agent_fence", "agent_fence_switch"})

    def __iter__(self):
        for key in dict.__iter__(self):
            if key not in self._ADVERTISED_EXTRA:
                yield key


def _seatbelt_mechanism_available() -> bool:
    """True when macOS Seatbelt (sandbox-exec) can enforce a read wall.

    Shared evidence class for backends whose isolation is Seatbelt-backed
    (grok ``--sandbox strict``, codex kernel fence). Never raises — degrades
    to False so ``list_models`` / ``backends_snapshot`` stay LOUD envelopes.
    """
    try:
        return bool(kernel_sandbox.available())
    except Exception as exc:  # noqa: BLE001 — flag must not escape list_models
        log_event(
            "read_wall_probe_error",
            backend="seatbelt",
            detail=str(exc)[:200],
        )
        return False


def _consult_read_wall_verified() -> bool:
    """True only when the consult fence verify path succeeds (issue #45 F3).

    Presence of sandbox-exec alone is not enough: ``kernel_fence_unverified``
    still means there is no read wall. Exceptions during the probe degrade the
    flag to False; they never propagate out of ``backends_snapshot``.
    """
    try:
        if not kernel_sandbox.available():
            return False
        refusal = kernel_sandbox.verify_fence(
            None, read_only=True, read_wall=True,
        )
        if refusal is not None:
            log_event(
                "read_wall_unverified",
                backend="consult",
                detail=str(refusal.get("detail") or refusal.get("reason") or "")[:200],
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — degrade, never raise
        log_event(
            "read_wall_probe_error",
            backend="consult",
            detail=str(exc)[:200],
        )
        return False


def backends_snapshot():
    """Static routing/capability map + CLI presence (no network probes).

    ``read_wall`` is the machine-readable isolation flag (issue #45): True when
    this backend's consult path enforces a kernel-level deny of home-tree (or
    stricter) file reads. Callers on private-data hosts should gate seats on
    this flag rather than on docstring lore. Flags never overstate.

      - grok: True only when the Seatbelt mechanism is available (Darwin +
        sandbox-exec). ``--sandbox strict`` may warn-and-run-unsandboxed
        elsewhere; that host must not advertise a read wall.
      - codex, claude: True only after
        ``verify_fence(..., read_wall=True)`` succeeds for the consult-shaped
        fence. Each provider's actual consult then verifies its provider-specific
        policy before spawning. sandbox-exec present but verify refused → False.
      - others: False — no consult-lane read confinement.

    Probe exceptions degrade the relevant flag to False and never escape.

    ``agent_fence`` is the agent-lane isolation class advertised as if the
    fence switch were ON (``seatbelt-fs`` / ``vendor`` / ``none``). It
    describes the WRITE lane, which is the one a caller picking a seat is
    choosing. codex is the one backend whose two lanes differ (issue #94):
    its write lane is ``seatbelt-fs`` where the mechanism exists and falls
    back to ``vendor`` where it does not, while ``--read-only`` is always
    vendor and additionally gated by the write-canary.
    ``agent_fence_switch`` is the live ``OMNI_LLM_AGENT_FENCE`` switch.
    ``read_wall`` keeps its consult-lane meaning.
    """
    openai_key = bool(os.environ.get("OPENAI_API_KEY") or any(
        k.startswith("OPENAI_API_KEY_") and os.environ.get(k)
        for k in os.environ
    ))
    # Each flag probe is independently fail-soft.
    grok_read_wall = _seatbelt_mechanism_available()
    consult_read_wall = _consult_read_wall_verified()
    codex_read_wall = consult_read_wall
    claude_read_wall = consult_read_wall
    switch = "on" if kernel_sandbox.agent_fence_switch_on() else "off"
    seatbelt = "seatbelt-fs" if _seatbelt_mechanism_available() else "none"
    # codex degrades to the vendor sandbox rather than to nothing: without
    # the Seatbelt mechanism its argv keeps --sandbox workspace-write.
    codex_fence = "seatbelt-fs" if _seatbelt_mechanism_available() else "vendor"

    def _cap(entry, agent_fence):
        out = _BackendCap(entry)
        out["agent_fence"] = agent_fence
        out["agent_fence_switch"] = switch
        return out

    return {
        "agy": _cap({
            "cli": bool(shutil.which("agy")),
            "sdk": bool(HAS_SDK),
            "prefix": "(default — any name not matched below)",
            "files": True,
            # The agy consult lane runs inside the host fence on macOS with
            # the same read wall as codex/claude: the verified consult probe
            # (verify_fence read_wall=True), not just sandbox-exec presence.
            # False off macOS or when the probe fails. agy is not an agent
            # backend, so its agent_fence stays "none".
            "read_wall": consult_read_wall,
            "profiles": True,
        }, "none"),
        "grok": _cap({
            "cli": bool(vendor_bin.which("grok")),
            "prefix": "grok*",
            "files": True,
            "read_wall": grok_read_wall,
            "profiles": True,
        }, "vendor"),
        "codex": _cap({
            "cli": bool(vendor_bin.which("codex")),
            "prefix": "codex/*",
            "files": True,
            "read_wall": codex_read_wall,
            "profiles": True,
        }, codex_fence),
        "claude": _cap({
            "cli": bool(vendor_bin.which("claude")),
            "prefix": "claude/*",
            "files": False,
            "read_wall": claude_read_wall,
            "profiles": True,
        }, seatbelt),
        "pi": _cap({
            "cli": bool(vendor_bin.which("pi")),
            "prefix": "pi/*",
            "files": False,
            "read_wall": False,
            "profiles": False,
        }, seatbelt),
        "cursor": _cap({
            "cli": bool(cursor_mod.resolve_bin()),
            "prefix": "cursor/*",
            "files": False,
            "read_wall": False,
            "profiles": True,
        }, "none"),
        "openai": _cap({
            "api_key_env": openai_key,
            "prefix": "gpt-*/o1*/o3*/*codex*",
            "files": False,
            "read_wall": False,
            "profiles": True,
            "default_api_base": "https://api.openai.com/v1",
        }, "none"),
        "ollama": _cap({
            "prefix": "ollama/",
            "files": False,
            "read_wall": False,
            "profiles": False,
            "default_api_base": "http://localhost:11434/v1",
        }, "none"),
        "lmstudio": _cap({
            "prefix": "lmstudio/",
            "files": False,
            "read_wall": False,
            "profiles": False,
            "default_api_base": "http://localhost:1234/v1",
        }, "none"),
    }


async def list_models(timeout_s=DEFAULT_MODELS_TIMEOUT_S):
    """Wrap `agy models` + attach static backends map."""
    def done(value):
        from providers import model_catalog
        try:
            value["catalogs"] = model_catalog.status()
        except model_catalog.CatalogError as exc:
            value["catalogs"] = exc.envelope()
        except Exception:
            value["catalogs"] = {"ok": False, "reason": "cache_unavailable",
                                  "detail": "model catalog cache unavailable"}
        return _with_health(value)

    backends = backends_snapshot()
    agy = agy_mod.find_agy()
    if not agy:
        return done({
            "ok": False, "reason": "cli_missing",
            "detail": "agy not on PATH - install Antigravity CLI / run `agy install`",
            "backends": backends,
        })
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_MODELS_TIMEOUT_S

    known_secrets.register_child_env(None)
    # Same fenced spawn as the agy consult lane: host fence (provider agy,
    # read-only), allowlisted environment, empty throwaway cwd.
    cwd = None
    try:
        try:
            argv, child_env, cwd = agy_mod.fence_agy_spawn(
                [agy, "models"], os.environ.copy())
        except HostFenceError as exc:
            return done(dict(
                exc.envelope, backends=backends))
        proc = await asyncio.create_subprocess_exec(
            *argv, env=child_env, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_bytes, err_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return done({
                "ok": False, "reason": "timeout",
                "detail": "agy models exceeded %ds" % timeout_s,
                "backends": backends,
            })

        out = (out_bytes.decode() or "").strip()
        err = (err_bytes.decode() or "").strip()

        if proc.returncode != 0:
            reason = classify_cli_error(out, err, proc.returncode)
            return done({
                "ok": False, "reason": reason,
                "detail": safe_detail(err or out or "exit %d" % proc.returncode),
                "backends": backends,
            })

        models = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not models:
            return done({
                "ok": False, "reason": "empty_response",
                "detail": safe_detail(err or "agy models returned nothing"),
                "backends": backends,
            })
        # Lines are "<slug>\t<display name>"; either spelling selects it.
        default_available = any(
            DEFAULT_MODEL in (field.strip() for field in ln.split("\t"))
            for ln in models
        )
        res = {
            "ok": True, "models": models, "default": DEFAULT_MODEL,
            "default_available": default_available,
            "routable_models": sorted({
                "agy/" + line.split("\t", 1)[0].strip()
                for line in models if line.split("\t", 1)[0].strip()
            }),
            "backends": backends,
        }
        if not default_available:
            res["warning"] = (
                "default model %r is not in `agy models`; calls without a "
                "model will fail unknown_model" % DEFAULT_MODEL)
        return done(res)
    except Exception as e:
        return done({
            "ok": False, "reason": "runtime_error",
            "detail": exception_detail(e), "backends": backends,
        })
    finally:
        if cwd:
            shutil.rmtree(cwd, ignore_errors=True)


# Typed LOUD envelopes: `ok` is always present; optional fields only when set.
# `reason` is failure-only (do not require). `text`/`model` are success-path for
# text tools; the strict checker only validates present declared property types.
_OUTPUT_SCHEMA_LLM_TEXT = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "text": {"type": "string"},
        "model": {"type": "string"},
        "detail": {"type": "string"},
        # reason is failure-only string; not typed here because success paths may
        # omit it or leave it null (strict checker rejects null vs string).
        "tried": {"type": "array"},
        "seat": {"type": "object"},
        "usage": {
            "type": "object",
            "properties": {
                "input_tokens": {"type": ["integer", "null"]},
                "cached_input_tokens": {"type": ["integer", "null"]},
                "cache_write_tokens": {"type": ["integer", "null"]},
                "output_tokens": {"type": ["integer", "null"]},
                "reasoning_tokens": {"type": ["integer", "null"]},
                "total_tokens": {"type": ["integer", "null"]},
                "source": {"type": "string", "enum": ["provider", "estimated", "unavailable"]},
                "raw": {},
            },
            "required": [
                "input_tokens", "cached_input_tokens", "cache_write_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens",
                "source", "raw",
            ],
        },
        # Present only when result_mode ref|auto spilled the full body (P1.4).
        "artifact_ref": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "sha256": {"type": "string"},
                "text_bytes": {"type": "integer"},
                "expires": {"type": "number"},
            },
        },
    },
    "required": ["ok"],
}

# Optional. Omitted = backend default. Static menus live in README.
_EFFORT_PROP = {
    "effort": {
        "type": "string",
        "description": (
            "agy/grok/claude/codex/pi/cursor; else unsupported_effort"
        ),
    },
}

_REQUIRE_READ_WALL_PROP = {
    "require_read_wall": {
        "type": "boolean",
        "description": "LOUD read_wall_required unless fenced",
    },
}

# Optional opt-in for llm_generate / llm_big_context only (default = inline).
_RESULT_MODE_PROP = {
    "result_mode": {
        "type": "string",
        "enum": ["inline", "ref", "auto"],
        "description": (
            "inline (default) | ref (spill) | auto (spill > 65536 B). "
            "Spilled: preview + artifact_ref; fetch via omni artifact read. "
            "Rehydrated body is untrusted."
        ),
    },
}

_OUTPUT_SCHEMA_LIST_MODELS = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "models": {"type": "array"},
        "default": {"type": "string"},
        "backends": {"type": "object"},
        "health": {"type": "object"},
        "detail": {"type": "string"},
    },
    "required": ["ok"],
}

_OUTPUT_SCHEMA_LLM_IMAGE = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "model": {"type": "string"},
        "n": {"type": "integer"},
        "images": {"type": "array"},
        "detail": {"type": "string"},
    },
    "required": ["ok"],
}


TOOLS = [
    {
        "name": "llm_generate",
        "description": (
            "PUBLIC text pipe (agy/OpenAI/Grok/Codex/Claude/Pi/Cursor/local). "
            "`files` only on agy/codex/grok (different semantics; see files param); "
            "other models LOUD {ok:false, reason:unsupported_files}. "
            "Comma-separated model = opt-in fallback chain L→R; single model = no fallback. "
            "Short backoff-retry on transient rate_limit. Total failure carries tried. LOUD {ok:false, reason} on failure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt text. PUBLIC data only."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "agy / codex / grok only. agy: absolute paths or URLs appended into prompt. "
                        "codex: local `-i` vision; no URLs. "
                        "grok: %s base64 or %s inlined as text under %s; "
                        "per-file/total/payload %d/%d/%d-byte caps. "
                        "files_too_large on text-cap; unsupported_files on "
                        "ext mismatch or image-only overflow. "
                        "must be UTF-8 (source_invalid_utf8)."
                    ) % (
                        "/".join(e.lstrip(".") for e in sorted(FILES_IMAGE_MIME)),
                        "/".join(e.lstrip(".") for e in sorted(FILES_TEXT_MIME)),
                        ENV_SOURCE_ROOTS,
                        GROK_FILES_TEXT_MAX_BYTES,
                        GROK_FILES_TEXT_TOTAL_MAX_BYTES,
                        GROK_FILES_MAX_PAYLOAD_BYTES,
                    )
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Model name. Default agy (e.g. 'Gemini 3.5 Flash'). "
                        "Prefix: gpt-/o1/o3/*codex*→OpenAI; ollama/… lmstudio/…→local; "
                        "grok*→Grok; claude/…→Claude; pi/…→Pi; cursor/…→Cursor; "
                        "codex/…→Codex CLI "
                        "(⚠️ read-only ≠ confinement: codex can READ any local file — PUBLIC data only). "
                        "Comma-separated = opt-in fallback chain L→R "
                        "(e.g. 'grok-4.5,Gemini 3.5 Flash'); single model = no fallback."
                    ),
                },
                "account_profile": {
                    "type": "string",
                    "description": (
                        "Optional account profile (e.g. 'default', 'premium'). "
                        "Selects credentials/seat."
                    ),
                },
                **_REQUIRE_READ_WALL_PROP,
                "timeout_s": {"type": "integer", "description": "Seconds to wait. Default %d." % DEFAULT_TIMEOUT_S},
                **_EFFORT_PROP,
                **_RESULT_MODE_PROP,
                **_WATCH_PANE_PROPS,
            },
            "required": ["prompt"],
        },
        "outputSchema": _OUTPUT_SCHEMA_LLM_TEXT,
    },
    {
        "name": "llm_big_context",
        "description": (
            "Digest a large PUBLIC document. `instruction` = what to extract/do. "
            "EITHER `text` (inline) OR `source_path` (server reads one UTF-8 file under "
            "OMNI_LLM_SOURCE_ROOTS) — mutually exclusive. "
            "source_path is NOT sent to providers; server loads it into SOURCE before routing "
            "(sandboxes do not cover this read). Plain text for all backends (unlike agy files=). "
            "Same LOUD envelope as llm_generate; path failures add error_code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "What to extract or do with the source."},
                "text": {
                    "type": "string",
                    "description": (
                        "Inline PUBLIC source. Mutually exclusive with source_path."
                    ),
                },
                "source_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to a PUBLIC UTF-8 text file the MCP server reads. "
                        "Requires env %s (allowlist). Mutually exclusive with text. "
                        "Not a capability grant — only selects within roots. "
                        "Symlinks rejected. Max 12MiB. Path failures: LOUD ok:false with error_code."
                    ) % ENV_SOURCE_ROOTS,
                },
                "model": {
                    "type": "string",
                    "description": "Model name. Default agy. Comma-list = opt-in fallback.",
                },
                "account_profile": {
                    "type": "string",
                    "description": "Account profile (credentials/seat).",
                },
                **_REQUIRE_READ_WALL_PROP,
                "timeout_s": {"type": "integer", "description": "Seconds to wait. Default 240."},
                **_EFFORT_PROP,
                **_RESULT_MODE_PROP,
                **_WATCH_PANE_PROPS,
            },
            "required": ["instruction"],
        },
        "outputSchema": _OUTPUT_SCHEMA_LLM_TEXT,
    },
    {
        "name": "list_models",
        "description": (
            "List models from `agy models` plus static `backends` map and seat `health`. "
            "On agy failure still LOUD {ok:false} but keeps backends and health."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_s": {"type": "integer", "description": "Seconds to wait. Default %d." % DEFAULT_MODELS_TIMEOUT_S},
            },
        },
        "outputSchema": _OUTPUT_SCHEMA_LIST_MODELS,
    },
    {
        "name": "llm_image",
        "description": (
            "Image generation/edit. PUBLIC data only. "
            "backend='cli' (default, subscription): grok CLI image_edit — EDITS a reference, REQUIRES source_image; returns PNG path. "
            "backend='rest' (per-image billing): xAI images API (XAI_API_KEY); text→image; honors n/response_format; "
            "ignores source_image. "
            "Returns {ok, model, n, images:[{path|url, revised_prompt?}]}. "
            "LOUD {ok:false, reason} on failure "
            "(cli_missing/auth_error/rate_limit/timeout/empty_response/runtime_error)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Image prompt / edit instruction. PUBLIC data only.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["cli", "rest"],
                    "description": (
                        "'cli' (default, subscription): grok CLI image_edit — EDITS a reference, REQUIRES source_image. "
                        "'rest': xAI image REST API; requires XAI_API_KEY; text→image."
                    ),
                },
                "source_image": {
                    "type": "string",
                    "description": (
                        "cli only (required there): filesystem path to the reference image "
                        "for image_edit."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "rest only: xAI image model. Default 'grok-2-image'.",
                },
                "n": {
                    "type": "integer",
                    "description": "rest only: image count (1-10). Default 1.",
                },
                "response_format": {
                    "type": "string",
                    "enum": ["url", "b64_json"],
                    "description": (
                        "rest only: 'b64_json' (default) → PNG path; 'url' → hosted URL."
                    ),
                },
                "out_dir": {
                    "type": "string",
                    "description": (
                        "Directory for PNG output (cli, or rest b64_json). Default: temp dir."
                    ),
                },
                "account_profile": {
                    "type": "string",
                    "description": (
                        "rest: selects XAI_API_KEY_<PROFILE> (else XAI_API_KEY). "
                        "cli: single grok login."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Seconds to wait. Default %d." % DEFAULT_TIMEOUT_S,
                },
            },
            "required": ["prompt"],
        },
        "outputSchema": _OUTPUT_SCHEMA_LLM_IMAGE,
    },
]


def _provider_kwargs(effort):
    """Keyword extras for a provider call. Omitted effort stays off the call."""
    if effort is None:
        return {}
    return {"effort": effort}


def _peel_request(model, effort):
    """Logical model and effort after ``@`` resolution."""
    return take_effort(model, effort)


def _ring_managed(provider, profile):
    return (provider in ("codex", "claude", "grok")
            and _norm_seat_profile(profile) == "default"
            and os.environ.get("OMNI_LLM_PROFILES_" + str(provider).upper()) is not None)


def _attempt_isolated(provider, profile, scrub_siblings):
    """Isolation the provider will apply, so the gate sees the same env."""
    if scrub_siblings or _ring_managed(provider, profile):
        return True
    return provider_accounts.ring_pinned(provider, profile, False)


def _consult_settings(provider, profile):
    if provider != "claude":
        return ()
    return claude_mod.consult_settings_files(profile)


async def _route_to_provider(prompt, model, timeout_s, files, account_profile,
                             effort=None, scrub_siblings=False):
    """Pick a provider for ONE model via ROUTE_RULES and call it exactly once."""
    depth = env_depth()
    if depth >= 2:
        return mark_local({
            "ok": False, "reason": "recursion_guard",
            "detail": "omni consultation refused at %s=%d (chain bound)"
                      % (ENV_DEPTH, depth)})
    model, effort, split_err = _peel_request(model, effort)
    if split_err:
        return mark_local(dict(split_err))
    route = resolve_route(model)
    provider = route["provider"]
    routed_model = route["model"]
    files = files or []

    if files and not provider_supports_files(provider):
        return mark_local({
            "ok": False,
            "reason": "unsupported_files",
            "detail": (
                "files= is supported on agy/codex/grok; model %r routes to %s. "
                "Omit files or pick a supporting model."
            ) % (model, provider),
        })

    if provider in ("grok", "claude", "pi", "ollama", "lmstudio"):
        _overrides, profile_err = resolve_profile(provider, account_profile)
        if profile_err:
            return mark_local(dict(profile_err))

    isolated = _attempt_isolated(provider, account_profile, scrub_siblings)
    settings = _consult_settings(provider, account_profile)
    mode = admission.gate(
        provider, account_profile, isolated=isolated, host_env=os.environ,
        settings_files=settings)
    # Catalog attempts still admit inside the provider. The egress check
    # stays here for every attempt, catalog or today, before that call.
    if mode != "catalog":
        resolved, effort_err = effort_for_provider(provider, effort)
        if effort_err:
            return mark_local(dict(effort_err))
        extra = _provider_kwargs(resolved)
    else:
        extra = _provider_kwargs(effort)
    denied = refuse_payload(prompt)
    if denied:
        # omni's own egress gate refused: no vendor ran, so it cannot cool a seat.
        if mode != "catalog":
            decision = admission.consider(
                provider, account_profile, routed_model, effort, isolated=isolated,
                host_env=os.environ, settings_files=settings)
            return mark_local(admission.stamp(
                denied, decision,
                sent_effort=admission.sent_effort(provider, decision, effort)))
        return mark_local(denied)
    if scrub_siblings and provider in RING_ROADS:
        extra["scrub_siblings"] = True

    if provider in ("ollama", "lmstudio", "openai"):
        if provider in ("ollama", "lmstudio"):
            api_base = os.environ.get(
                "OPENAI_API_BASE_%s" % str(account_profile).upper(),
                route.get("default_api_base", "http://localhost:11434/v1"),
            )
        else:
            api_base = route.get("default_api_base", "https://api.openai.com/v1")
        result = await openai_mod.run_openai_compatible(
            prompt, routed_model, timeout_s, account_profile, api_base=api_base)
        decision = admission.consider(
            provider, account_profile, routed_model, effort, isolated=False,
            host_env=os.environ, settings_files=())
        return admission.stamp(
            result, decision, sent_effort=admission.sent_effort(provider, decision, effort))

    if provider == "grok":
        # routed_model is the name AFTER a `grok/` prefix is stripped
        # (bare `grok-4.6` is unchanged). files: local images under
        # OMNI_LLM_SOURCE_ROOTS, inlined as --prompt-json.
        return await grok_mod.run_grok_subprocess(
            prompt, routed_model, timeout_s, account_profile, files=files,
            **extra)

    if provider == "codex":
        # routed_model is the name AFTER the `codex/` prefix is stripped (what codex -m wants).
        # files: existing local paths handed to codex as repeated `-i` (native vision).
        return await codex_mod.run_codex_subprocess(
            prompt, routed_model, timeout_s, account_profile, files=files,
            apply_aliases=True, **extra)

    if provider == "claude":
        # routed_model is the name AFTER the `claude/` prefix is stripped.
        return await claude_mod.run_claude_subprocess(
            prompt, routed_model, timeout_s, account_profile,
            apply_aliases=True, **extra)

    if provider == "pi":
        # routed_model is the name AFTER the `pi/` prefix is stripped
        # (what `pi --model` wants, e.g. xai/grok-4.6). pi is always today's gate.
        result = await pi_mod.run_pi_subprocess(
            prompt, routed_model, timeout_s, account_profile, **extra)
        decision = admission.consider(
            "pi", account_profile, routed_model, effort, isolated=False,
            host_env=os.environ, settings_files=())
        return admission.stamp(
            result, decision, sent_effort=admission.sent_effort("pi", decision, effort))

    if provider == "cursor":
        # routed_model is the name AFTER the `cursor/` prefix is stripped;
        # alias_model inside the provider expands fable/sol.
        return await cursor_mod.run_cursor_subprocess(
            prompt, routed_model, timeout_s, account_profile,
            apply_aliases=True, **extra)

    return await agy_mod.run_agy(
        prompt, routed_model, timeout_s, files, account_profile, **extra)


async def route_to_provider(prompt, model, timeout_s, files, account_profile,
                            effort=None, scrub_siblings=False):
    """One provider call. The result always carries the disclosure keys."""
    result = await _route_to_provider(
        prompt, model, timeout_s, files, account_profile,
        effort=effort, scrub_siblings=scrub_siblings)
    return admission.ensure_disclosure(result)


def _chain_stops_after(provider, res):
    """True when a failed seat ends the request instead of moving to the next seat.

    Codex: every failure ends it except ``account_exhausted`` (no task child
    was run); the decision does not depend on ``account_retry_safe``.
    Claude and Grok: tool activity, or a result carrying terminal no-retry
    provenance, ends it.
    """
    if provider == "codex":
        return bool(res.get("tool_activity")) or res.get("reason") != "account_exhausted"
    if provider in ("claude", "grok"):
        return bool(res.get("tool_activity") or (
            res.get("reason") != "account_exhausted"
            and res.get("account_retry_safe") is False))
    return False


async def dispatch_single(prompt, model, timeout_s, files, account_profile,
                          effort=None, probing=False, scrub_siblings=False):
    """One model, with bounded backoff-retry on a transient rate_limit."""
    model, effort, split_err = _peel_request(model, effort)
    if split_err:
        return split_err
    metrics_mod.note_seat_call(model)
    attempt = 0
    extra = _provider_kwargs(effort)
    if scrub_siblings:
        extra["scrub_siblings"] = True
    while True:
        res = await route_to_provider(
            prompt, model, timeout_s, files, account_profile, **extra)
        # A failure omni raised itself (task file, spawn, refusal) says nothing
        # about the vendor seat: it must not feed the crash streak. The stamp is
        # internal; nothing past this point sees it.
        local_failure = is_local_failure(res)
        res = strip_local_marker(res)
        provider = resolve_route(model)["provider"]
        metrics_mod.note_provider_attempt(model, provider, res)
        # Codex never backs off and retries: whatever its result says, a
        # failure ends the request. Claude and Grok stop on provenance.
        codex_stop = provider == "codex" or (provider in ("claude", "grok") and (
            res.get("account_profile") is not None
            or res.get("account_retry_safe") is not True or res.get("tool_activity")))
        if (codex_stop or res.get("reason") != "rate_limit"
                or attempt >= RATE_LIMIT_RETRIES):
            if res.get("reason") == "rate_limit":
                if not probing and not res.get("account_profile"):
                    _flag_rate_limit(model, account_profile, result=res)
            elif res.get("ok"):
                _health_clear(model, res.get("account_profile") or account_profile)
            elif not local_failure:
                _health_note_failure(
                    model, res.get("account_profile") or account_profile, res.get("reason"))
            return res
        attempt += 1
        log_event("rate_limit_retry", model=model, backoff_s=RATE_LIMIT_BACKOFF_S,
                  attempt=attempt, max_retries=RATE_LIMIT_RETRIES)
        await asyncio.sleep(RATE_LIMIT_BACKOFF_S)


def _flag_rate_limit(model, account_profile, result=None):
    """Flag a rate-limited seat; Codex uses the quota RPC for ``until``.

    Claude / cursor consult parse vendor reset times from the failure
    envelope (``stdout`` / ``stderr`` when present, else ``detail`` — raw
    CLI output is not reachable at this layer). Unparsed → default cooldown.
    """
    until = None
    source = "rate_limit"
    provider = resolve_route(model)["provider"]
    if provider == "codex":
        info = None
        try:
            from providers.codex_quota import read_rate_limits
            info, _err = read_rate_limits(
                _norm_seat_profile(account_profile), force=True)
        except Exception as exc:
            log_event("health_error", operation="quota_probe", model=model,
                      detail=str(exc)[:300])
            info = None
        if isinstance(info, dict) and info.get("resets_at") is not None:
            until = info.get("resets_at")
            source = "quota"
    elif provider in ("claude", "cursor"):
        result = result if isinstance(result, dict) else {}
        stdout = result.get("stdout") or result.get("detail") or ""
        stderr = result.get("stderr") or ""
        until = health.vendor_reset_from_output(provider, stdout, stderr)
        if until is not None:
            source = "vendor_reset"
            log_event(
                "vendor_reset_applied",
                model=model,
                profile=_norm_seat_profile(account_profile),
                until=until,
            )
    _health_flag(model, account_profile, until=until, source=source)


def _norm_seat_profile(account_profile):
    if account_profile is None or str(account_profile).strip() in ("", "default"):
        return "default"
    return str(account_profile)


def _want_read_wall(require_read_wall):
    """Parameter wins; env OMNI_LLM_REQUIRE_READ_WALL=1 is the omit-default."""
    if require_read_wall is None:
        return os.environ.get("OMNI_LLM_REQUIRE_READ_WALL") == "1"
    return bool(require_read_wall)


def _seat_view(seat):
    return {
        "family": seat.get("family"),
        "provider": seat.get("provider"),
        "model": seat.get("model"),
        "account_profile": _norm_seat_profile(seat.get("account_profile")),
    }


def _tried_entry(seat, reason):
    """Family and ring seats name the profile; other explicit seats keep {model, reason}."""
    if seat.get("family") is not None or seat.get("ring"):
        return {
            "model": seat["model"],
            "account_profile": _norm_seat_profile(seat.get("account_profile")),
            "reason": reason,
        }
    return {"model": seat["model"], "reason": reason}


def _read_wall_required(seat):
    return {
        "ok": False,
        "reason": "read_wall_required",
        "detail": (
            "model %s routes to %s which has no consult-lane read wall"
            % (seat.get("model"), seat.get("provider"))
        ),
    }


def _expand_chain(chain, links, account_profile):
    """Turn chain links into seats. Family config errors abort the whole request."""
    seats = []
    for index, raw in enumerate(chain):
        bare, eff, err = links[index]
        if err:
            seats.append({
                "family": None,
                "provider": None,
                "model": raw,
                "account_profile": account_profile,
                "effort": None,
                "split_err": err,
            })
            continue
        if is_family(bare):
            fam_seats, fam_err = expand_family(bare, account_profile)
            if fam_err:
                return None, fam_err
            for fam in fam_seats:
                seat = dict(fam)
                seat["effort"] = eff
                seat["split_err"] = None
                seats.append(seat)
            continue
        seat = {
            "family": None,
            "provider": resolve_route(bare)["provider"],
            "model": bare,
            "account_profile": account_profile,
            "effort": eff,
            "split_err": None,
        }
        seats.append(seat)
    return seats, None


def _apply_profile_rings(seats, account_profile):
    """Expand default-profile seats on a road that has ``OMNI_LLM_PROFILES_<ROAD>``.

    A call-level non-default profile pins every seat and disables expansion.
    Malformed / unknown-road ring env is LOUD ``profile_config`` for the
    whole call. No ring → the same *seats* list object. Each road's ring is
    resolved once per call (dict *road* → *names*) and reused across seats.
    """
    rings, env_err = _load_profile_rings()
    if env_err:
        return None, env_err
    if _norm_seat_profile(account_profile) != "default":
        return seats, None
    out = None
    for seat in seats:
        if _norm_seat_profile(seat.get("account_profile")) != "default":
            if out is not None:
                out.append(seat)
            continue
        provider = seat.get("provider")
        if provider not in RING_ROADS:
            if out is not None:
                out.append(seat)
            continue
        # Subscription providers own one request-local account walk.
        # Expanding these here would bypass their replay boundary.
        if provider in ("codex", "claude", "grok"):
            if out is not None:
                out.append(seat)
            continue
        names = (rings or {}).get(provider)
        if not names:
            if out is not None:
                out.append(seat)
            continue
        if out is None:
            out = []
            # Copy seats already walked that did not expand.
            for prior in seats:
                if prior is seat:
                    break
                out.append(prior)
        for name in names:
            child = dict(seat)
            child["account_profile"] = name
            child["ring"] = True
            if name == "default":
                child["scrub_siblings"] = True
            out.append(child)
    return (seats if out is None else out), None


def _seat_gate(seat):
    """The gate this seat's provider attempt will see."""
    provider = seat.get("provider")
    profile = seat.get("account_profile")
    isolated = bool(seat.get("scrub_siblings")) or _ring_managed(provider, profile)
    return admission.gate(
        provider, profile, isolated=isolated, host_env=os.environ,
        settings_files=_consult_settings(provider, profile))


def _classify_seat(seat, want_wall, snap):
    """Return (kind, reason): ineligible, cooling, probing, or healthy.

    Catalog admission is decided inside the provider. A catalog seat is
    classified only by profile, health, and the read wall. A catalog refusal
    on the provider result ends the walk.
    """
    if seat.get("split_err"):
        return "ineligible", seat["split_err"].get("reason") or "unsupported_effort"
    provider = seat["provider"]
    profile = seat["account_profile"]
    _overrides, profile_err = resolve_profile(provider, profile)
    if profile_err:
        return "ineligible", "unsupported_profile"
    mode = _seat_gate(seat)
    if mode == "today":
        resolved, effort_err = effort_for_provider(provider, seat.get("effort"))
        if effort_err:
            return "ineligible", "unsupported_effort"
        if provider == "cursor":
            base = cursor_mod.alias_model(resolve_route(seat["model"])["model"])
            _slug, slug_err = cursor_effort_slug(base, resolved, profile)
            if slug_err:
                return "ineligible", "unsupported_effort"
    if want_wall:
        info = (snap or {}).get(provider) or {}
        if not info.get("read_wall"):
            return "ineligible", "read_wall_required"
    if (provider in ("codex", "claude", "grok") and _norm_seat_profile(profile) == "default"
            and os.environ.get("OMNI_LLM_PROFILES_" + provider.upper()) is not None):
        return "healthy", None  # eligibility is evaluated inside the account selector
    _health_passive_probe(seat["model"], profile)
    kind = _health_kind(seat["model"], profile)
    if kind in ("cooling", "probing"):
        return kind, None
    return "healthy", None


def _attach_seat(res, seat):
    if res.get("ok"):
        res["seat"] = _seat_view(seat)
        if res.get("account_profile") is not None:
            res["seat"]["account_profile"] = res["account_profile"]
    return res


def _retired_agy_chain(model):
    """True when any comma-chain seat uses the retired ``agy-opus-*`` form."""
    text = "" if model is None else str(model)
    return any(retired_agy_opus(part) for part in text.split(","))


async def _dispatch_llm_request(prompt, model, timeout_s, files, account_profile,
                               effort=None, require_read_wall=None):
    """OPT-IN FALLBACK: comma-separated model chain tried left-to-right.

    A family link expands to its ordered seats. Healthy seats dispatch first;
    cooling seats dispatch only after every healthy seat has failed;
    ineligible seats are never dispatched. A retired ``agy-opus-*`` name
    anywhere in the chain refuses before expansion, health maintenance,
    or any seat.
    """
    if _retired_agy_chain(model):
        return retired_model_error()
    chain = [m.strip() for m in str(model).split(",") if m.strip()] or [DEFAULT_MODEL]
    links = [_peel_request(m, effort) for m in chain]
    seats, cfg_err = _expand_chain(chain, links, account_profile)
    if cfg_err:
        return cfg_err
    seats, ring_err = _apply_profile_rings(seats, account_profile)
    if ring_err:
        return ring_err

    want_wall = _want_read_wall(require_read_wall)
    snap = backends_snapshot() if want_wall else None
    _health_maintain()

    # Single explicit model: today's dispatch_single path + seat + read-wall gate.
    # Any ring (even one name) disables this path; no ring keeps it
    # byte-identical to a chain of one explicit model.
    single_explicit = (
        len(chain) == 1
        and len(seats) == 1
        and seats[0].get("family") is None
        and not seats[0].get("ring")
    )
    if single_explicit:
        seat = seats[0]
        if seat.get("split_err"):
            return seat["split_err"]
        if want_wall:
            info = (snap or {}).get(seat["provider"]) or {}
            if not info.get("read_wall"):
                return _read_wall_required(seat)
        kwargs = _provider_kwargs(seat.get("effort"))
        res = await dispatch_single(
            prompt, seat["model"], timeout_s, files,
            seat["account_profile"], **kwargs)
        return _attach_seat(res, seat)

    states = [_classify_seat(s, want_wall, snap) for s in seats]

    tried = []
    skipped = []
    res = None
    dispatched = False
    probe_taken = False
    dispatched_indices = set()
    exhausted_roads = set()
    def managed_codex(seat):
        return (seat.get("provider") in ("codex", "claude", "grok")
                and _norm_seat_profile(seat.get("account_profile")) == "default"
                and os.environ.get("OMNI_LLM_PROFILES_" + seat["provider"].upper()) is not None)
    for index, seat in enumerate(seats):
        if seat.get("provider") in exhausted_roads and managed_codex(seat):
            tried.append(_tried_entry(seat, "account_exhausted"))
            continue
        kind, reason = states[index]
        if kind == "ineligible":
            tried.append(_tried_entry(seat, reason))
            continue
        effective = kind
        if kind == "probing" and probe_taken:
            effective = "cooling"
        later_live = any(
            _walk_will_dispatch(
                states[j][0],
                probe_taken or effective == "probing",
            )
            for j in range(index + 1, len(seats)))
        if effective == "cooling" and later_live:
            tried.append(_tried_entry(seat, "skipped_cooldown"))
            skipped.append(index)
            log_event("skipped_cooldown", model=seat["model"])
            continue
        is_probe = effective == "probing"
        if is_probe:
            probe_taken = True
            log_event(
                "seat_probe",
                model=seat["model"],
                profile=_norm_seat_profile(seat.get("account_profile")),
            )
        res = await dispatch_single(
            prompt, seat["model"], timeout_s, files,
            seat["account_profile"], probing=is_probe,
            scrub_siblings=bool(seat.get("scrub_siblings")),
            **_provider_kwargs(seat.get("effort")))
        dispatched = True
        dispatched_indices.add(index)
        if is_probe:
            if res.get("ok"):
                log_event(
                    "seat_recovered",
                    model=seat["model"],
                    profile=_norm_seat_profile(seat.get("account_profile")),
                    source="probe",
                )
            else:
                _health_reprobe(seat["model"], seat["account_profile"])
        if res.get("ok"):
            if tried:
                inner = res.get("tried") if seat.get("provider") in ("codex", "claude", "grok") else []
                res["tried"] = list(inner or []) + tried
            return _attach_seat(res, seat)
        if res.get("reason") in admission.TERMINAL_CATALOG_REFUSALS:
            if tried:
                inner = res.get("tried") if seat.get("provider") in ("codex", "claude", "grok") else []
                res["tried"] = list(inner or []) + tried
            return res
        if managed_codex(seat) and res.get("reason") == "account_exhausted":
            exhausted_roads.add(seat["provider"])
        if _chain_stops_after(seat.get("provider"), res):
            if tried:
                res["tried"] = list(res.get("tried") or []) + tried
            return res
        if seat.get("provider") in ("codex", "claude", "grok"):
            tried.extend(res.get("tried") or [])
        tried.append(_tried_entry(seat, res.get("reason")))

    for index in skipped:
        if index in dispatched_indices:
            continue
        seat = seats[index]
        if seat.get("provider") in exhausted_roads and managed_codex(seat):
            tried.append(_tried_entry(seat, "account_exhausted"))
            continue
        res = await dispatch_single(
            prompt, seat["model"], timeout_s, files,
            seat["account_profile"],
            scrub_siblings=bool(seat.get("scrub_siblings")),
            **_provider_kwargs(seat.get("effort")))
        dispatched = True
        if res.get("ok"):
            if tried:
                inner = res.get("tried") if seat.get("provider") in ("codex", "claude", "grok") else []
                res["tried"] = list(inner or []) + tried
            return _attach_seat(res, seat)
        if res.get("reason") in admission.TERMINAL_CATALOG_REFUSALS:
            if tried:
                inner = res.get("tried") if seat.get("provider") in ("codex", "claude", "grok") else []
                res["tried"] = list(inner or []) + tried
            return res
        if managed_codex(seat) and res.get("reason") == "account_exhausted":
            exhausted_roads.add(seat["provider"])
        if _chain_stops_after(seat.get("provider"), res):
            if tried:
                res["tried"] = list(res.get("tried") or []) + tried
            return res
        if seat.get("provider") in ("codex", "claude", "grok"):
            tried.extend(res.get("tried") or [])
        tried.append(_tried_entry(seat, res.get("reason")))

    if not dispatched:
        return {
            "ok": False,
            "reason": "no_eligible_seat",
            "detail": "no seat could be dispatched: %s" % "; ".join(
                "%s (%s)" % (t.get("model"), t.get("reason")) for t in tried),
            "tried": tried,
        }
    res["tried"] = tried
    return res


def _push_watch_from_args(args):
    """Install watch overrides for this call; returns context token for reset."""
    args = args or {}
    return set_watch_context(
        watch_pane=args["watch_pane"] if "watch_pane" in args else _UNSET,
        zellij_session=args["zellij_session"] if "zellij_session" in args else _UNSET,
        zellij_tab=args["zellij_tab"] if "zellij_tab" in args else _UNSET,
    )


def _first_chain_model(model):
    chain = [m.strip() for m in str(model or "").split(",") if m.strip()]
    return chain[0] if chain else DEFAULT_MODEL


def _seat_for_result(res, model, account_profile):
    """Winning (or last) seat for the metrics record — not a routing change."""
    attempts = metrics_mod.current_attempts()
    if res.get("ok") and res.get("model") is not None:
        seat_model = res.get("model")
    elif attempts:
        seat_model = attempts[-1].get("model") or _first_chain_model(model)
    else:
        seat_model = _first_chain_model(model)
    # Prefer the provider recorded on the last matching attempt; else resolve.
    provider = None
    for entry in reversed(attempts):
        if entry.get("model") == seat_model:
            provider = entry.get("provider")
            break
    if provider is None:
        provider = resolve_route(seat_model)["provider"]
    profile = account_profile
    env_seat = res.get("seat")
    if isinstance(env_seat, dict) and env_seat.get("account_profile") is not None:
        profile = env_seat.get("account_profile")
    elif res.get("account_profile") is not None:
        profile = res.get("account_profile")
    return {
        "provider": provider,
        "model": str(seat_model),
        "account_profile": (
            "default" if profile is None else str(profile)
        ),
    }


def _finish_metrics(envelope, *, tool, model, account_profile, request_bytes, t0):
    """Append one content-free metrics line; never rewrite model result fields."""
    if not isinstance(envelope, dict):
        return envelope
    # Seat/record construction sits inside the metrics failure guard so any
    # metrics-side exception becomes metrics_error and leaves the result intact.
    try:
        latency_ms = int(max(0.0, (time.time() - t0) * 1000.0))
        seat = _seat_for_result(envelope, model, account_profile)
        # Spilled envelopes keep preview size in result_bytes; original_text_bytes
        # records the pre-spill raw UTF-8 length (P1.4 / issue #22).
        original_text_bytes = None
        ref = envelope.get("artifact_ref")
        if isinstance(ref, dict) and ref.get("text_bytes") is not None:
            try:
                original_text_bytes = int(ref["text_bytes"])
            except (TypeError, ValueError):
                original_text_bytes = None
        record = metrics_mod.build_record(
            entry=metrics_mod.detect_entry(),
            tool=tool,
            seat=seat,
            attempts=metrics_mod.current_attempts(),
            fallback_count=metrics_mod.current_fallback_count(),
            request_bytes=request_bytes,
            result_bytes=metrics_mod.result_bytes_of(envelope),
            latency_ms=latency_ms,
            usage=envelope.get("usage"),
            ok=bool(envelope.get("ok")),
            reason=envelope.get("reason") if not envelope.get("ok") else None,
            original_text_bytes=original_text_bytes,
        )
        return metrics_mod.attach_metrics(envelope, record)
    except Exception as exc:
        out = dict(envelope)
        out["metrics_error"] = redact_secrets(str(exc))[:300]
        return out


def _apply_result_mode(envelope, args):
    """Opt-in spill for llm_generate / llm_big_context only.

    UnicodeEncodeError from create() (e.g. lone surrogates in provider text)
    becomes a typed runtime_error — never an escaped exception out of call_tool.
    ArtifactError (size/quota) becomes a typed LOUD reason.
    Rehydrated artifact bodies are untrusted prior LLM output.
    """
    args = args or {}
    if "result_mode" not in args:
        return envelope  # no opt-in → byte-identical envelope
    try:
        return artifacts_mod.apply_result_mode(envelope, args.get("result_mode"))
    except UnicodeEncodeError as exc:
        return {
            "ok": False,
            "reason": "runtime_error",
            "detail": redact_secrets("text is not encodable as utf-8: %s" % exc),
        }
    except artifacts_mod.ArtifactError as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "detail": exc.detail or exc.reason,
        }


def call_budget_s(name, args):
    """Worst-case wall clock one tools/call can legitimately occupy, in seconds.

    The shutdown drain uses this so an in-flight call is not cancelled before its
    own budget is up. Mirrors dispatch_llm_request: every model in a comma chain
    may burn the full timeout, and each may additionally retry once on a
    transient rate_limit (plus its backoff). A flat constant cannot express that
    — a 3-model chain at the default timeout is already ~18x a single call.
    """
    args = args or {}
    is_big = name in ("llm_big_context", "gemini_big_context")
    default_s = BIG_CONTEXT_TIMEOUT_S if is_big else DEFAULT_TIMEOUT_S
    try:
        timeout_s = int(args.get("timeout_s", default_s))
    except (TypeError, ValueError):
        timeout_s = default_s
    if timeout_s <= 0:
        timeout_s = default_s

    if not (is_big or name in ("llm_generate", "gemini_generate")):
        return timeout_s  # llm_image / list_models: no fallback chain

    chain = [m for m in str(args.get("model", "")).split(",") if m.strip()] or [""]
    attempts = len(chain) * (RATE_LIMIT_RETRIES + 1)
    return timeout_s * attempts + RATE_LIMIT_BACKOFF_S * (attempts - 1)


_CONSULT_TOOLS = (
    "llm_generate", "gemini_generate", "llm_big_context", "gemini_big_context",
)


async def call_tool(name, args):
    """One tool call -> envelope; the internal local-origin stamp never leaves."""
    result = strip_local_marker(await _call_tool_inner(name, args))
    if name in _CONSULT_TOOLS:
        return admission.ensure_disclosure(result)
    return result


async def _call_tool_inner(name, args):
    if name in ("llm_generate", "gemini_generate"):
        prompt = args.get("prompt", "")
        files = args.get("files", [])
        model = str(args.get("model") or DEFAULT_MODEL)
        profile = args.get("account_profile", "default")
        timeout_s = args.get("timeout_s", DEFAULT_TIMEOUT_S)
        extra = _provider_kwargs(args.get("effort") if "effort" in args else None)
        if "require_read_wall" in args:
            extra["require_read_wall"] = args["require_read_wall"]
        request_bytes = metrics_mod.utf8_len(prompt)
        t0 = time.time()
        trace_tokens = metrics_mod.begin_trace()
        try:
            if not str(prompt).strip():
                res = mark_local({"ok": False, "reason": "runtime_error",
                                  "detail": "empty prompt"})
                return _finish_metrics(
                    res, tool=name, model=model, account_profile=profile,
                    request_bytes=request_bytes, t0=t0)

            token = _push_watch_from_args(args)
            try:
                res = await dispatch_llm_request(
                    prompt, model, timeout_s, files, profile, **extra)
            finally:
                reset_watch_context(token)
            res = _apply_result_mode(res, args)
            return _finish_metrics(
                res, tool=name, model=model, account_profile=profile,
                request_bytes=request_bytes, t0=t0)
        finally:
            metrics_mod.end_trace(trace_tokens)

    if name in ("llm_big_context", "gemini_big_context"):
        instruction = str(args.get("instruction", "")).strip()
        # Preserve legacy text handling when not in path mode (including strip).
        raw_text = args.get("text", "")
        text = str(raw_text).strip() if raw_text is not None else ""
        # Do not strip source_path (path may legally contain edge whitespace).
        source_path = args.get("source_path")
        if source_path is not None:
            source_path = str(source_path)
        model = str(args.get("model") or DEFAULT_MODEL)
        profile = args.get("account_profile", "default")
        timeout_s = args.get("timeout_s", BIG_CONTEXT_TIMEOUT_S)
        extra = _provider_kwargs(args.get("effort") if "effort" in args else None)
        if "require_read_wall" in args:
            extra["require_read_wall"] = args["require_read_wall"]
        t0 = time.time()
        # request_bytes = UTF-8 of instruction + source inputs (content-free size).
        request_bytes = metrics_mod.utf8_len(instruction)
        trace_tokens = metrics_mod.begin_trace()
        try:
            path_mode = source_path is not None and source_path != ""
            if not path_mode:
                if not text:
                    res = mark_local({"ok": False, "reason": "runtime_error",
                                      "detail": "empty text"})
                    return _finish_metrics(
                        res, tool=name, model=model, account_profile=profile,
                        request_bytes=request_bytes, t0=t0)
                source_body = text
            else:
                source_body, load_err = resolve_big_context_source(
                    text, source_path, instruction=instruction)
                if load_err:
                    log_event(
                        "source_path_error",
                        error_code=load_err.get("error_code"),
                    )
                    return _finish_metrics(
                        load_err, tool=name, model=model, account_profile=profile,
                        request_bytes=request_bytes, t0=t0)

            request_bytes = metrics_mod.utf8_len(instruction) + metrics_mod.utf8_len(
                source_body)
            prompt = "%s\n\n--- SOURCE (public) ---\n%s" % (instruction, source_body)
            token = _push_watch_from_args(args)
            try:
                res = await dispatch_llm_request(
                    prompt, model, timeout_s, [], profile, **extra)
            finally:
                reset_watch_context(token)
            res = _apply_result_mode(res, args)
            return _finish_metrics(
                res, tool=name, model=model, account_profile=profile,
                request_bytes=request_bytes, t0=t0)
        finally:
            metrics_mod.end_trace(trace_tokens)

    if name == "llm_image":
        prompt = args.get("prompt", "")
        profile = args.get("account_profile", "default")
        request_bytes = metrics_mod.utf8_len(prompt)
        t0 = time.time()
        # Image backends are a single seat (no fallback chain).
        model = str(args.get("model") or "grok-2-image")
        trace_tokens = metrics_mod.begin_trace()
        try:
            if not str(prompt).strip():
                res = mark_local({"ok": False, "reason": "runtime_error",
                                  "detail": "empty prompt"})
                return _finish_metrics(
                    res, tool=name, model=model, account_profile=profile,
                    request_bytes=request_bytes, t0=t0)
            denied = refuse_payload(
                prompt, args.get("source_image"), args.get("out_dir"))
            if denied:
                return _finish_metrics(
                    denied, tool=name, model=model, account_profile=profile,
                    request_bytes=request_bytes, t0=t0)
            backend = str(args.get("backend") or "cli").lower()
            if backend not in ("cli", "rest"):
                res = mark_local({
                    "ok": False, "reason": "runtime_error",
                    "detail": "unknown backend %r (use 'cli' or 'rest')" % backend})
                return _finish_metrics(
                    res, tool=name, model=model, account_profile=profile,
                    request_bytes=request_bytes, t0=t0)
            if backend == "cli":
                # DEFAULT: keyless grok CLI image_edit (subscription) — edits source_image.
                # The request model is not the CLI model. Admit the R11 id
                # against the process-environment record before any spawn.
                model = "grok-image-edit"
                decision = admission.consider(
                    "grok", "default", grok_image_cli_mod.GROK_IMAGE_MODEL, None,
                    host_env=os.environ, apply_aliases=True, alias_image=True)
                if decision.get("refusal"):
                    res = admission.refusal_envelope("grok", "default", decision)
                    return _finish_metrics(
                        res, tool=name, model=model, account_profile=profile,
                        request_bytes=request_bytes, t0=t0)
                metrics_mod.note_seat_call(model)
                res = await grok_image_cli_mod.run_grok_image_edit(
                    prompt,
                    args.get("source_image"),
                    timeout_s=args.get("timeout_s", DEFAULT_TIMEOUT_S),
                    out_dir=args.get("out_dir"),
                    account_profile=profile,
                    apply_aliases=True,
                )
                metrics_mod.note_provider_attempt(model, "grok", res)
            else:
                # backend == "rest": existing xAI image REST path (needs XAI_API_KEY).
                metrics_mod.note_seat_call(model)
                res = await xai_image_mod.run_xai_image(
                    prompt,
                    model=model,
                    n=args.get("n", 1),
                    response_format=args.get("response_format", "b64_json"),
                    timeout_s=args.get("timeout_s", DEFAULT_TIMEOUT_S),
                    account_profile=profile,
                    out_dir=args.get("out_dir"),
                )
                metrics_mod.note_provider_attempt(model, "xai_image", res)
            return _finish_metrics(
                res, tool=name, model=model, account_profile=profile,
                request_bytes=request_bytes, t0=t0)
        finally:
            metrics_mod.end_trace(trace_tokens)

    if name == "list_models":
        return await list_models(args.get("timeout_s", DEFAULT_MODELS_TIMEOUT_S))

    return mark_local({"ok": False, "reason": "runtime_error",
                       "detail": "unknown tool %s" % name})


async def dispatch_llm_request(prompt, model, timeout_s, files, account_profile,
                               effort=None, require_read_wall=None):
    with account_registry_scope():
        result = await _dispatch_llm_request(
            prompt, model, timeout_s, files, account_profile,
            effort=effort, require_read_wall=require_read_wall)
    return admission.ensure_disclosure(result)
