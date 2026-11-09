# no-opencc
"""Pi coding-agent CLI provider — subscription-auth, near-pure text pipe.

Rides the local `pi` login (OAuth / auth.json, no API key in this process),
the same shape as claude/codex/grok. This module is the CONSULTATION lane
only; the agent lane builds its own pi command in providers/agent_lane.py.

Confinement: `-p --no-tools` headless with context/extension/skill discovery
off, empty tempdir cwd, `--no-session` so consult does not write ~/.pi
sessions. On macOS, the shared host fence enforces the existing home read
wall and provider allow-set; reads outside protected home remain allowed, so
the caller still supplies only public data. RECURSION GUARD: refuses when
OMNI_LLM_DEPTH >= 1 and exports depth+1 to the child, so a pi->omni->pi
loop dies at the first hop. Parent PI_* session markers are stripped so
the child does not inherit the caller's session identity.
"""
import os
import shutil
import tempfile

from providers.cli_run import run_cli
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.kernel_sandbox import resolve_agent_binary
from providers.config import (
    DEFAULT_TIMEOUT_S,
    DEFAULT_PI_MODEL,
    ENV_DEPTH,
    env_depth,
)
from providers.errors import (
    classify_cli_error,
    exception_detail,
    safe_detail,
    local_before_spawn,
    note_spawned,
    signal_exit_detail,
)
from providers.profiles import spawn_env
from providers.usage import extract_cli_result

# Session identity inherited from a parent `pi` process. Never leak these
# into a child consult spawn — they would point at the caller's session.
_PARENT_PI_SESSION_KEYS = (
    "PI_SESSION_ID",
    "PI_SESSION_FILE",
    "PI_PROVIDER",
    "PI_MODEL",
    "PI_REASONING_LEVEL",
)


def child_env(depth):
    """Env for a spawned `pi`: bump OMNI_LLM_DEPTH, drop parent session markers."""
    env = spawn_env({**os.environ, ENV_DEPTH: str(depth + 1)}, {})
    for key in _PARENT_PI_SESSION_KEYS:
        env.pop(key, None)
    return env


def consult_argv(pi_bin, model, effort=None):
    """Headless single-turn argv. Prompt arrives on stdin."""
    argv = [
        pi_bin, "-p",
        "--mode", "json",
        "--no-tools",
        "--no-session",
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-approve",
        "--model", model,
    ]
    if effort:
        argv += ["--thinking", str(effort)]
    return argv


@local_before_spawn
async def run_pi_subprocess(prompt, model, timeout_s, account_profile="default",
                            effort=None, _spawn=None):
    """Pipe prompt -> `pi -p --mode json` (headless single-turn) -> typed envelope."""
    depth = env_depth()
    if depth >= 1:
        return {"ok": False, "reason": "recursion_guard",
                "detail": "pi backend refused at %s=%d "
                          "(pi->omni->pi loop guard)" % (ENV_DEPTH, depth)}
    pi_bin = resolve_agent_binary("pi")
    if not pi_bin:
        return {"ok": False, "reason": "cli_missing",
                "detail": "pi not on PATH - install Pi coding-agent CLI, then log in"}
    # Exec the same file the fence admits: agent_fence_policy (this consult
    # passes no policy of its own) admits resolve_agent_binary's result as a
    # literal read, and Seatbelt refuses an execvp through the ~/.local/bin
    # symlink. Resolving here from the same source keeps exec and admission
    # on one path, the OMNI_LLM_PI_BIN override included.
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    model = str(model or "").strip() or DEFAULT_PI_MODEL
    tmp_cwd = None
    try:
        tmp_cwd = tempfile.mkdtemp(prefix="pi_cwd_")
        argv = consult_argv(pi_bin, model, effort=effort)
        denied = refuse_payload(prompt)
        if denied:
            return denied
        child_env_for_cli = child_env(depth)
        try:
            argv, child_env_for_cli, _fence_meta = prepare_host_fence(
                argv, provider="pi", workdir=tmp_cwd,
                read_only=True, env=child_env_for_cli,
            )
        except HostFenceError as exc:
            return exc.envelope
        res = await run_cli(
            argv,
            stdin_data=str(prompt),
            cwd=tmp_cwd,
            env=child_env_for_cli,
            timeout_s=timeout_s,
            pane_name="omni:pi",
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "pi exceeded %ds" % timeout_s}

        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")

        if rc != 0:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = signal_exit_detail("pi", -res["signal"], out, err)
            else:
                reason = classify_cli_error(out, err, rc)
                if reason == "signal_exit":
                    detail = signal_exit_detail("pi", rc, out, err)
                else:
                    detail = safe_detail(err or out or "exit %d" % (
                        rc if rc is not None else -1))
            return {"ok": False, "reason": reason, "detail": detail}
        text, usage = extract_cli_result("pi", out)
        if not text:
            return {"ok": False, "reason": "empty_response",
                    "detail": safe_detail(err or "pi returned no text")}
        return {"ok": True, "text": text, "model": model, "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}
    finally:
        if tmp_cwd and os.path.isdir(tmp_cwd):
            shutil.rmtree(tmp_cwd, ignore_errors=True)
