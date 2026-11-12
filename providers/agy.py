# no-opencc
"""Google Antigravity (agy) provider — SDK when model-capable, else CLI subprocess.

The subprocess lane runs inside the shared host fence on macOS (provider
"agy" in ``providers.host_fence``): Seatbelt profile, home read wall, vendor
state limited to diagnostics, the GEMINI.md / AGENTS.md instruction family and
agy's hooks / MCP configuration denied for read, and the allowlisted child
environment. agy starts in an owned throwaway directory; large CLI requests
place their complete read-only input there before fence construction. The in-process SDK path is not a child process
omni can wrap and stays outside the fence, so on macOS ``run_agy`` never calls
it: the fenced subprocess is the only agy path there (a missing or failing
fence is a loud refusal, never a fallback to the SDK).
"""
import asyncio
import hashlib
import json
import os
import platform
import re
import tempfile
from shutil import rmtree as _rmtree

from providers import vendor_bin
from providers.cli_run import run_cli
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.config import (
    AGY_PROMPT_MAX_ARGV_BYTES,
    AGY_PROMPT_FILE_THRESHOLD_BYTES,
    SOURCE_PROMPT_MAX_BYTES,
    CLI_SELF_TIMEOUT_MARGIN_S,
    DEFAULT_TIMEOUT_S,
    ENV_DEPTH,
    HAS_SDK,
    Agent,
    LocalAgentConfig,
    CapabilitiesConfig,
    env_depth,
)
from providers.errors import (
    classify_cli_error,
    classify_exception_message,
    exception_detail,
    safe_detail,
    local_before_spawn,
    mark_local,
    note_spawned,
    signal_exit_detail,
)
from providers.logutil import log
from providers.usage import extract_cli_result, inband_error, normalize_usage

# AGY_CONFIG_DIR is process-global and the SDK reads it from the environment, so two
# concurrent SDK calls cannot each hold a different profile's config dir at once —
# apply/restore around one call would clobber another's account (a privacy boundary,
# not a cosmetic glitch). Serialize the whole SDK critical section (apply→chat→restore)
# behind one lock. The subprocess path (per-call env= copy) and every non-agy provider
# stay fully concurrent; only the in-process SDK path is one-at-a-time. Lazy-init so the
# lock binds to the running loop (py3.9), mirroring server._get_write_lock.
_sdk_lock = None


def _agy_profile_dir(account_profile):
    name = "" if account_profile is None else str(account_profile).strip()
    if name in ("", "default"):
        return None, None
    if not re.fullmatch(r"[a-z0-9_]+", name):
        return None, {"ok": False, "reason": "unsupported_profile",
                      "detail": "account_profile is not a valid agy profile name"}
    key = "AGY_CONFIG_DIR_%s" % name.upper()
    raw = os.environ.get(key)
    raw = raw.strip() if raw else ""
    expanded = os.path.realpath(os.path.expanduser(raw)) if raw else None
    if not expanded:
        return None, {"ok": False, "reason": "unsupported_profile",
                      "detail": "%s is unset" % key}
    return expanded, None


def find_agy():
    """The agy file to exec, or None.

    ``vendor_bin.which`` skips harness launcher shims, and the fence admits
    ``kernel_sandbox.resolve_agent_binary("agy")`` (the same lookup, resolved),
    so exec and admission name one file. The result is resolved too: Seatbelt
    refuses an execvp through a symlink inside the home read wall.
    """
    found = vendor_bin.which("agy")
    return os.path.realpath(found) if found else None


def fence_agy_spawn(argv, env, prompt_bytes=None):
    """``(argv, env, cwd)`` for one fenced agy child (shared by every agy spawn).

    Runs the shared host fence for provider ``agy`` (read-only, allowlisted
    environment) on a fresh owned directory that becomes the child's cwd. An
    optional complete input is staged read-only before policy construction. The
    caller removes ``cwd`` when the child is done; on a refusal it is removed
    here and ``HostFenceError`` propagates.
    """
    cwd = tempfile.mkdtemp(prefix="agy_cwd_")
    try:
        if prompt_bytes is not None:
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is None:
                raise RuntimeError("agy input staging requires no-follow file creation")
            path = os.path.join(cwd, "omni-input.txt")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
            try:
                remaining = memoryview(prompt_bytes)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("agy input file write made no progress")
                    remaining = remaining[written:]
                os.fchmod(fd, 0o400)
            finally:
                os.close(fd)
            instruction = (
                "Read the complete UTF-8 user request from the owned local file below "
                "using read-only file tools. Read bounded offset/limit pages until EOF; "
                "do not stop at a preview or infer omitted content. Process every byte, "
                "including the final lines, then follow the complete contained request. "
                "If any part cannot be read, report incomplete input instead of guessing.\n"
                "INPUT_FILE: " + json.dumps(path) + "\n"
                "INPUT_BYTES: " + str(len(prompt_bytes)) + "\n"
                "INPUT_SHA256: " + hashlib.sha256(prompt_bytes).hexdigest()
            )
            argv = list(argv)
            argv[argv.index("-p") + 1] = instruction
        wrapped, child_env, _meta = prepare_host_fence(
            argv, provider="agy", workdir=cwd, read_only=True, env=env)
    except BaseException:
        _rmtree(cwd, ignore_errors=True)
        raise
    return wrapped, child_env, cwd


def _get_sdk_lock():
    global _sdk_lock
    if _sdk_lock is None:
        _sdk_lock = asyncio.Lock()
    return _sdk_lock


@local_before_spawn
async def run_agy_subprocess(prompt, model, timeout_s, account_profile,
                            effort=None, _spawn=None):
    """Use inline text or one exact-byte file request; return a typed envelope."""
    profile_dir, profile_error = _agy_profile_dir(account_profile)
    if profile_error:
        return profile_error
    prompt_text = str(prompt)
    try:
        prompt_bytes = prompt_text.encode("utf-8")
    except UnicodeError:
        return {"ok": False, "reason": "runtime_error",
                "detail": "agy prompt is not valid UTF-8"}
    try:
        denied = refuse_payload(prompt_text)
    except Exception as e:
        return {"ok": False, "reason": "runtime_error", "detail": exception_detail(e)}
    if denied:
        return denied
    size = len(prompt_bytes)
    use_file = size > AGY_PROMPT_FILE_THRESHOLD_BYTES
    limit = SOURCE_PROMPT_MAX_BYTES if use_file else AGY_PROMPT_MAX_ARGV_BYTES
    if size > limit:
        if use_file:
            detail = "agy file prompt cap %d bytes exceeded (%d)" % (limit, size)
        else:
            detail = ("agy prompt cap %d bytes exceeded (%d); "
                      "use codex/<model> for large payloads" % (limit, size))
        return {"ok": False, "reason": "payload_too_large", "detail": detail}

    agy = find_agy()
    if not agy:
        return {"ok": False, "reason": "cli_missing",
                "detail": "agy not on PATH - install Antigravity CLI / run `agy install`"}
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    env = os.environ.copy()
    env[ENV_DEPTH] = str(env_depth() + 1)
    if profile_dir:
        env["AGY_CONFIG_DIR"] = profile_dir

    # Let agy hit its own deadline just before run_cli's hard kill, so a slow
    # model comes back as agy's own timeout message rather than a SIGKILL.
    inner_timeout_s = max(1, timeout_s - CLI_SELF_TIMEOUT_MARGIN_S)

    if effort is not None:
        from providers.model_catalog import CatalogError, agy_effort_tokens
        try:
            supported = await asyncio.to_thread(
                agy_effort_tokens, account_profile, min(timeout_s, 30))
        except CatalogError as exc:
            return {"ok": False, "reason": exc.reason,
                    "detail": "agy effort capability check failed (%s)" % exc.reason}
        if str(effort).strip().lower() not in supported:
            return {"ok": False, "reason": "unsupported_effort",
                    "detail": "agy installed CLI does not advertise effort %r"
                              % str(effort).strip().lower()}

    tmp_cwd = None
    try:
        argv = [agy, "-p", "" if use_file else prompt_text, "--model", str(model),
                "--print-timeout", "%ds" % inner_timeout_s,
                "--output-format", "json"]
        if effort is not None:
            argv += ["--effort", str(effort).strip().lower()]
        try:
            if use_file:
                argv, env, tmp_cwd = fence_agy_spawn(argv, env, prompt_bytes=prompt_bytes)
            else:
                argv, env, tmp_cwd = fence_agy_spawn(argv, env)
        except HostFenceError as exc:
            return exc.envelope
        res = await run_cli(
            argv,
            cwd=tmp_cwd,
            env=env,
            timeout_s=timeout_s,
            pane_name="omni:agy",
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "agy exceeded %ds" % timeout_s}

        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")

        if rc != 0:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = signal_exit_detail("agy", -res["signal"], out, err)
            else:
                reason = classify_cli_error(out, err, rc)
                if reason == "signal_exit":
                    detail = signal_exit_detail("agy", rc, out, err)
                else:
                    detail = safe_detail(err or out or "exit %d" % (
                        rc if rc is not None else -1))
            return {"ok": False, "reason": reason, "detail": detail}
        # In-band failure first: agy can exit 0 while reporting status != SUCCESS,
        # and reading only .response would file that as an empty reply.
        failure = inband_error("agy", out)
        if failure:
            return {"ok": False, "reason": failure[0], "detail": failure[1]}
        # Emptiness is judged AFTER extraction: under --output-format json a
        # reply-less run still prints a full JSON document, so checking raw stdout
        # would let an empty answer through as a silent success.
        text, usage = extract_cli_result("agy", out)
        if not text:
            return {"ok": False, "reason": "empty_response",
                    "detail": safe_detail(err or "agy returned no text")}
        return {"ok": True, "text": text, "model": str(model), "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}
    finally:
        if tmp_cwd:
            _rmtree(tmp_cwd, ignore_errors=True)


def apply_agy_account_profile(account_profile):
    """Mirror subprocess profile env into the process for SDK use.

    Returns (env_applied, previous_AGY_CONFIG_DIR or None).
    """
    if not account_profile or account_profile == "default":
        return False, None
    target_dir = os.environ.get("AGY_CONFIG_DIR_%s" % str(account_profile).upper())
    if not target_dir:
        return False, None
    previous = os.environ.get("AGY_CONFIG_DIR")
    os.environ["AGY_CONFIG_DIR"] = target_dir
    return True, previous


def restore_agy_account_profile(applied, previous):
    if not applied:
        return
    if previous is None:
        os.environ.pop("AGY_CONFIG_DIR", None)
    else:
        os.environ["AGY_CONFIG_DIR"] = previous


def build_agy_sdk_config(model):
    """Build LocalAgentConfig that honors `model`, or None if the SDK cannot."""
    if LocalAgentConfig is None or CapabilitiesConfig is None:
        return None
    try:
        caps = CapabilitiesConfig()
    except TypeError:
        return None
    base = dict(
        system_instructions="You are a helpful assistant.",
        capabilities=caps,
    )
    for key in ("model", "default_model", "model_name"):
        try:
            return LocalAgentConfig(**dict(base, **{key: model}))
        except TypeError:
            continue
    return None


async def run_agy_sdk(prompt, model, timeout_s, files=None, account_profile="default",
                      effort=None):
    """Run via google.antigravity SDK. Honors model/timeout/profile or signals fallback."""
    _profile_dir, profile_error = _agy_profile_dir(account_profile)
    if profile_error:
        return profile_error
    if effort is not None:
        return {"ok": False, "reason": "unsupported_effort",
                "detail": "agy SDK cannot honor an explicit effort; use the CLI path"}
    if files:
        prompt += "\n\nFiles to analyze:\n" + "\n".join(files)

    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    denied = refuse_payload(prompt)
    if denied:
        # omni's own gate (it also sees the file paths dispatch does not):
        # the SDK never started, so this cannot cool the seat.
        return mark_local(denied)

    config = build_agy_sdk_config(str(model))
    if config is None:
        return {"ok": False, "reason": "sdk_unavailable",
                "detail": "google.antigravity LocalAgentConfig cannot accept model=%r; use subprocess" % model}

    # Hold the lock across apply→chat→restore: the profile env must stay pinned for
    # the whole chat (the SDK may read AGY_CONFIG_DIR lazily), and restore() runs in
    # the finally — inside the lock — so the next waiter always sees a clean env.
    async with _get_sdk_lock():
        applied, previous = apply_agy_account_profile(account_profile)
        try:
            async def _chat():
                async with Agent(config) as agent:
                    response = await agent.chat(prompt)
                    out = ""
                    async for token in response:
                        out += token
                    return out

            try:
                out = await asyncio.wait_for(_chat(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return {"ok": False, "reason": "timeout",
                        "detail": "agy SDK exceeded %ds" % timeout_s}

            if not (out or "").strip():
                return {"ok": False, "reason": "empty_response",
                        "detail": "agy SDK returned no text"}
            return {"ok": True, "text": out.strip(), "model": str(model),
                    "usage": normalize_usage("agy", None)}
        except Exception as e:
            return {"ok": False, "reason": classify_exception_message(e),
                    "detail": exception_detail(e)}
        finally:
            restore_agy_account_profile(applied, previous)


async def run_agy(prompt, model, timeout_s, files=None, account_profile="default",
                  effort=None):
    """agy provider: prefer SDK only when it can honor model/timeout/profile.

    On macOS the SDK is never used: it runs inside this process, outside the
    host fence, while the subprocess lane is fenced. Off macOS nothing is
    fenced and the SDK preference is unchanged.
    """
    _profile_dir, profile_error = _agy_profile_dir(account_profile)
    if profile_error:
        return profile_error
    from providers import admission
    decision = admission.consider(
        "agy", account_profile, model, effort,
        isolated=False, host_env=os.environ, settings_files=())
    if admission.is_terminal(decision):
        return admission.refusal_envelope("agy", account_profile, decision)
    model, effort, _catalog = admission.bound_call(model, effort, decision)
    if effort is not None:
        if files:
            prompt = str(prompt) + "\n\nFiles to analyze:\n" + "\n".join(files)
        result = await run_agy_subprocess(
            prompt, model, timeout_s, account_profile, effort=effort)
        return admission.stamp(result, decision)

    if HAS_SDK and platform.system() == "Darwin":
        log("agy SDK skipped on macOS: the subprocess lane runs inside the host fence")
    elif HAS_SDK:
        res = await run_agy_sdk(prompt, model, timeout_s, files, account_profile)
        if res.get("reason") != "sdk_unavailable":
            return admission.stamp(res, decision)
        log("SDK cannot honor model=%r; falling back to agy subprocess" % model)

    if files:
        prompt += "\n\nFiles to analyze:\n" + "\n".join(files)
    result = await run_agy_subprocess(prompt, model, timeout_s, account_profile)
    return admission.stamp(result, decision)
