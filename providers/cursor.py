# no-opencc
"""Cursor Agent CLI provider — subscription-auth, near-pure text pipe.

Rides the local `cursor-agent` login (Cursor account / CURSOR_API_KEY), the
same shape as claude/pi. This module is the CONSULTATION lane only; the
agent lane is a later PR.

Binary: NEVER `agent`. Grok Build CLI also installs that name. Resolution
is OMNI_LLM_CURSOR_BIN (must be an executable file) else
`shutil.which("cursor-agent")`, then ``realpath`` so argv[0] is the
versioned install leaf the host fence re-allows — a PATH symlink under
``~/.local/bin`` is inside the home read wall and ``execvp`` dies with
rc=71.

Confinement: `-p --mode ask` (read-only mode; --force/--yolo/--approve-mcps
are never passed), `--trust` so headless does not hang on the workspace
dialog, `--sandbox enabled` as the off-Darwin vendor baseline, empty
tempdir as both cwd and `--workspace`, and a per-run `CURSOR_CONFIG_DIR`
(see resolve_config_dir) so a consult can neither read the human's Cursor
state nor write to it. On macOS, the shared host fence is the process
boundary; after that wrap verifies, the helper rewrites the nested
`--sandbox` flag to ``disabled`` (cursor-agent's ``cursorsandbox`` is a
second ``sandbox_apply``, rc=71). Reads outside protected home remain
allowed.
RECURSION GUARD: refuses when OMNI_LLM_DEPTH >= 1 and exports depth+1.

Short names: after the `cursor/` prefix is stripped, exact aliases
`fable` → `claude-fable-5` and `sol` → `gpt-5.6-sol` (case-insensitive).
Any other remainder is passed to `--model` verbatim.

Effort: cursor-agent has no effort flag, so the public `effort` token is
mapped onto the model slug here (`sol` + xhigh → `gpt-5.6-sol-xhigh`,
table in config.CURSOR_EFFORT_SLUGS). A trailing `@<token>` on the
remainder is the same token (`cursor/sol@high`). The mapped slug is what
the CLI runs AND what the envelope reports, so a re-dispatch of the
envelope model lands on the same seat at the same strength. Omitted
effort changes nothing, except bases in config.CURSOR_DEFAULT_EFFORT
(sol/luna/terra → medium) whose catalog no longer lists a bare id.
"""
import os
import shutil
import tempfile

from providers.cli_run import run_cli
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.config import (
    CURSOR_CONFIG_DIR_ENV,
    CURSOR_DEFAULT_EFFORT,
    CURSOR_MODEL_ALIASES,
    DEFAULT_CURSOR_MODEL,
    DEFAULT_TIMEOUT_S,
    ENV_CURSOR_BIN,
    ENV_CURSOR_CONFIG_DIR,
    ENV_DEPTH,
    cursor_effort_slug,
    env_depth,
    normalize_effort,
)
from providers.errors import (
    classify_cli_error,
    exception_detail,
    safe_detail,
    local_before_spawn,
    note_spawned,
    signal_exit_detail,
)
from providers.logutil import log_event
from providers.seat_name import split_effort
from providers.profiles import resolve_profile, spawn_env
from providers.usage import extract_cli_result, inband_error


def _usable_bin(path):
    """Absolute realpath of an executable file, or None."""
    if not path:
        return None
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return os.path.realpath(path)
    return None


def resolve_bin():
    """Return the Cursor CLI path, or None.

    Never falls back to `agent` — that name is Grok Build CLI on this host.
    A set-but-unusable OMNI_LLM_CURSOR_BIN is fail-closed (None), not a
    fallback onto PATH. A PATH hit that is a symlink (the usual
    ``~/.local/bin/cursor-agent`` install) is resolved: the host fence
    re-allows the versioned install directory, not the home-tree link.
    """
    override = os.environ.get(ENV_CURSOR_BIN)
    if override is not None and str(override).strip():
        return _usable_bin(os.path.expanduser(str(override).strip()))
    return _usable_bin(shutil.which("cursor-agent"))


def alias_model(model):
    """Map a stripped cursor/ remainder to a --model slug."""
    raw = str(model or "").strip()
    if not raw:
        return DEFAULT_CURSOR_MODEL
    return CURSOR_MODEL_ALIASES.get(raw.lower(), raw)


def resolve_config_dir(env=None):
    """Where this run keeps its Cursor state, or None for a fresh temp dir.

    `cursor-agent` persists the selected model, its parameters and the chat
    transcript into its config dir, and reads the parameters back on the next
    run. Sharing the human's ~/.cursor therefore leaks BOTH ways: a consult
    rewrites their default model, and a bare base slug silently inherits the
    strength they last picked. A per-run directory makes the run hermetic;
    ENV_CURSOR_CONFIG_DIR opts back into a shared one. *env* (default:
    ``os.environ``) is the mapping consulted, so a profile override in the
    child env is treated exactly like the process-level opt-in.
    """
    source = os.environ if env is None else env
    override = source.get(ENV_CURSOR_CONFIG_DIR)
    if override is not None and str(override).strip():
        # Absolute, always: the child's cwd is the temp workspace this lane
        # deletes afterwards, so a relative override would be created inside
        # it and swept away with it. A relative value resolves against the
        # omni process, which is what an operator setting the env var means.
        return os.path.abspath(os.path.expanduser(str(override).strip()))
    return None


def resolve_model(model, effort, account_profile=None):
    """Alias, then fold effort into the slug. Returns (slug, error_envelope).

    A trailing ``@<token>`` on the remainder is effort shorthand. If
    ``effort`` is also set and the two tokens differ, the envelope is
    unsupported_effort. Omitted effort changes nothing, except bases in
    CURSOR_DEFAULT_EFFORT whose catalog no longer lists a bare id: those
    get the table's token.
    """
    raw = str(model or "").strip()
    raw, at_token, split_err = split_effort(raw)
    if split_err:
        return None, split_err
    token, err = normalize_effort(effort)
    if err:
        return None, err
    if at_token is not None and token is not None:
        return None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": (
                "both @%s on the model and --effort %s were given"
                % (at_token, token)
            ),
        }
    if token is None:
        token = at_token
    base = alias_model(raw)
    if token is None:
        token = CURSOR_DEFAULT_EFFORT.get(base)
    return cursor_effort_slug(base, token, account_profile)


def cleanup_tempdir(path):
    """Remove a temp dir this lane (consult or agent) made.

    ``FileNotFoundError`` is silent (already gone). Any other ``OSError``
    retries with ``ignore_errors=True`` and, if the path still exists
    afterward, logs ``cursor_tempdir_left_behind`` on stderr — never a
    silent ignore, and never raised (a leftover dir must not cost the
    caller an answer that already came back).
    """
    if not path:
        return
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        shutil.rmtree(path, ignore_errors=True)
        if os.path.exists(path):
            log_event("cursor_tempdir_left_behind", path=path,
                      error=str(e)[:200])


def envelope_model(model):
    """Public envelope id: always `cursor/<slug>` after alias.

    Bare `gpt-5.6-sol` / `claude-fable-5` are other providers' routes. Retry
    and metrics must stay on this backend.
    """
    return "cursor/%s" % alias_model(model)


def consult_argv(cursor_bin, model, workspace):
    """Headless Ask-mode argv. Prompt arrives on stdin.

    ``model`` already carries any requested effort (see resolve_model): the
    CLI has no flag for it.
    """
    return [
        cursor_bin, "-p",
        "--mode", "ask",
        "--output-format", "json",
        "--trust",
        "--sandbox", "enabled",
        "--workspace", workspace,
        "--model", model,
    ]


@local_before_spawn
async def run_cursor_subprocess(prompt, model, timeout_s, account_profile="default",
                                effort=None, scrub_siblings=False, _spawn=None):
    """Pipe prompt -> `cursor-agent -p --mode ask` -> typed envelope.

    ``effort`` is folded into the model slug before anything else runs, so a
    base/token pair the catalog cannot express stays LOUD unsupported_effort
    even when the CLI is missing (cli_missing must not mask a token error).
    """
    model, effort_err = resolve_model(model, effort, account_profile)
    if effort_err:
        return effort_err
    overrides, profile_err = resolve_profile("cursor", account_profile)
    if profile_err:
        return profile_err
    depth = env_depth()
    if depth >= 1:
        return {"ok": False, "reason": "recursion_guard",
                "detail": "cursor backend refused at %s=%d "
                          "(cursor->omni->cursor loop guard)" % (ENV_DEPTH, depth)}
    cursor_bin = resolve_bin()
    if not cursor_bin:
        return {"ok": False, "reason": "cli_missing",
                "detail": "cursor-agent not on PATH - install Cursor CLI "
                          "(curl https://cursor.com/install), then "
                          "`cursor-agent login`. Do not use the `agent` "
                          "name (Grok collision)."}
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    denied = refuse_payload(prompt)
    if denied:
        return denied

    tmp_cwd = None
    tmp_config = None
    try:
        tmp_cwd = tempfile.mkdtemp(prefix="cursor_cwd_")
        child_env = spawn_env(
            {**os.environ, ENV_DEPTH: str(depth + 1)}, overrides,
            scrub_siblings=scrub_siblings)
        config_dir = resolve_config_dir(child_env)
        if config_dir is None:
            tmp_config = tempfile.mkdtemp(prefix="cursor_cfg_")
            config_dir = tmp_config
        child_env[CURSOR_CONFIG_DIR_ENV] = config_dir
        argv = consult_argv(cursor_bin, model, tmp_cwd)
        try:
            argv, child_env, _fence_meta = prepare_host_fence(
                argv, provider="cursor", workdir=tmp_cwd,
                read_only=True, env=child_env, config_dir=config_dir,
            )
        except HostFenceError as exc:
            return exc.envelope
        res = await run_cli(
            argv,
            stdin_data=str(prompt),
            cwd=tmp_cwd,
            env=child_env,
            timeout_s=timeout_s,
            pane_name="omni:cursor",
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "cursor exceeded %ds" % timeout_s}

        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")

        if rc != 0:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = signal_exit_detail("cursor", -res["signal"], out, err)
            else:
                reason = classify_cli_error(out, err, rc)
                if reason == "signal_exit":
                    detail = signal_exit_detail("cursor", rc, out, err)
                else:
                    detail = safe_detail(err or out or "exit %d" % (
                        rc if rc is not None else -1))
            return {"ok": False, "reason": reason, "detail": detail}
        failure = inband_error("cursor", out)
        if failure:
            return {"ok": False, "reason": failure[0], "detail": failure[1]}
        text, usage = extract_cli_result("cursor", out)
        if not text:
            return {"ok": False, "reason": "empty_response",
                    "detail": safe_detail(err or "cursor returned no text")}
        return {"ok": True, "text": text, "model": envelope_model(model),
                "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}
    finally:
        # Only what this lane made: an OMNI_LLM_CURSOR_CONFIG_DIR directory is
        # the caller's and is never touched. tmp_config holds the prompt (the
        # CLI writes its chat store there), so a leftover is worth a LOUD line
        # on stderr rather than a silent ignore_errors — but it must not cost
        # the caller an answer that already came back.
        for path in (tmp_cwd, tmp_config):
            cleanup_tempdir(path)
