# no-opencc
"""Claude Code CLI provider — subscription-auth, near-pure text pipe.

Rides the local `claude` login (Max subscription, no API key), the same shape
as codex/grok. This module is the CONSULTATION lane only; the agent lane builds
its own claude command in providers/agent_lane.py.

Confinement: `-p` headless auto-denies un-granted tool permissions; we
additionally --disallowedTools the write/exec/web tools, run in an empty
tempdir cwd, and pass --strict-mcp-config so the spawned claude loads NO MCP
servers (it must never load omni-llm again). On macOS, the shared host fence
protects the child process with the existing home read wall and profile
allow-set. Reads outside protected home remain allowed by the current profile,
so callers still supply only public data. User-installed hooks also still run.
RECURSION GUARD:
refuses when OMNI_LLM_DEPTH >= 1 and exports depth+1 to the child, so a
claude->omni->claude loop dies at the first hop.
"""
import os
import time
import shutil
import tempfile

from providers.cli_run import run_cli
from providers import provider_accounts
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.kernel_sandbox import resolve_agent_binary
from providers.config import (
    DEFAULT_TIMEOUT_S,
    DEFAULT_CLAUDE_MODEL,
    ENV_DEPTH,
    env_depth,
)
from providers.errors import (
    classify_cli_error,
    exception_detail,
    safe_detail,
    strip_signin_expiry_warning,
    local_before_spawn,
    local_request_before_spawn,
    note_spawned,
    signal_exit_detail,
)
from providers.profiles import resolve_profile, resolve_account_home, spawn_env
from providers.usage import extract_cli_result, inband_error

# Read-only Claude lanes (consult and agent/job --read-only) run with an
# explicit built-in tool allowlist, so a built-in tool added later does not
# appear in them by default. The disallow list stays as a second layer; both
# lanes import these two constants (single source).
READ_ONLY_TOOLS = "Read,Grep,Glob"
READ_ONLY_DISALLOWED = "Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Task,Agent"


def claude_config_home(account_profile="default", account_home=None):
    """Config directory whose user settings this attempt may load."""
    if account_home:
        return account_home
    overrides, err = resolve_profile("claude", account_profile or "default")
    if not err and overrides.get("CLAUDE_CONFIG_DIR"):
        return overrides["CLAUDE_CONFIG_DIR"]
    pinned = os.environ.get("CLAUDE_CONFIG_DIR")
    if pinned:
        return pinned
    from providers.kernel_sandbox import protected_home
    return os.path.join(protected_home(), ".claude")


def consult_settings_files(account_profile="default", account_home=None, workdir=None):
    """The four claude settings files. Consult's project root is the process cwd."""
    from providers.admission import claude_settings_files
    if workdir is None:
        workdir = os.getcwd()
    return claude_settings_files(
        claude_config_home(account_profile, account_home), workdir)


# Env selectors that enable plugins (and their hooks) independently of
# --setting-sources; a read-only child never inherits them.
READ_ONLY_SCRUBBED_ENV = ("CLAUDE_CODE_PLUGIN_DIRS",)


def scrub_read_only_env(env):
    """Drop the plugin selectors from a read-only claude child env, in place."""
    for key in READ_ONLY_SCRUBBED_ENV:
        env.pop(key, None)
    return env


def _profile_detail(detail, account_profile, overrides):
    """Prefix *detail* with ``CLAUDE_CONFIG_DIR_<P>`` when a profile is active."""
    if not overrides:
        return detail
    var = "CLAUDE_CONFIG_DIR_%s" % str(account_profile).upper()
    if var in (detail or ""):
        return detail
    return safe_detail("%s: %s" % (var, detail))


@local_before_spawn
async def _run_claude_once(prompt, model, timeout_s, account_profile="default",
                                effort=None, scrub_siblings=False, account_home=None, account_auth_isolated=False,
                                account_deadline=None, account_home_explicit=True, pre_spawn=None,
                                _spawn=None):
    """Pipe prompt -> `claude -p` (headless single-turn) -> typed envelope.

    ``pre_spawn`` is the late API-gate check: a callable returning a refusal
    envelope or None, handed to ``run_cli``, which runs it as the last step before the child is created.
    """
    depth = env_depth()
    if depth >= 1:
        return {"ok": False, "reason": "recursion_guard",
                "detail": "claude backend refused at %s=%d "
                          "(claude->omni->claude loop guard)" % (ENV_DEPTH, depth)}
    # Profile before which(): a bad name stays unsupported_profile even when
    # the CLI is missing (same ordering rule as effort / codex).
    overrides, profile_err = (resolve_account_home("claude", account_profile, account_home)
                              if account_home else resolve_profile("claude", account_profile))
    if profile_err:
        return profile_err
    if account_home or account_home_explicit is not True:
        provenance_home = account_home or overrides.get("CLAUDE_CONFIG_DIR")
        if not provenance_home:
            provenance_home, profile_err = provider_accounts.profile_home("claude", account_profile)
            if profile_err:
                return profile_err
        provenance_err = provider_accounts.home_provenance_error(
            "claude", account_profile, provenance_home, account_home_explicit)
        if provenance_err:
            return provenance_err
        if not account_home_explicit:
            # Keep selected native-default provenance even when this unmanaged
            # call originally needed no concrete home override.
            account_home = provenance_home
    claude = resolve_agent_binary("claude")
    if not claude:
        return {"ok": False, "reason": "cli_missing",
                "detail": "claude not on PATH - install Claude Code CLI, then log in"}
    # Exec the same file the fence admits: agent_fence_policy (this consult
    # passes no policy of its own) admits resolve_agent_binary's result as a
    # literal read, and Seatbelt refuses an execvp through the ~/.local/bin
    # symlink. Resolving here from the same source keeps exec and admission
    # on one path, the OMNI_LLM_CLAUDE_BIN override included.
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    model = str(model or "").strip() or DEFAULT_CLAUDE_MODEL
    tmp_cwd = None
    try:
        tmp_cwd = tempfile.mkdtemp(prefix="claude_cwd_")
        argv = [
            claude, "-p",
            "--output-format", "json",
            "--model", model,
            "--strict-mcp-config",
            "--tools", READ_ONLY_TOOLS,
            "--disallowedTools", READ_ONLY_DISALLOWED,
        ]
        if effort:
            argv += ["--effort", str(effort)]
        # API-key children carry a hard spend cap; subscription children never do.
        _kind, budget, policy_err = provider_accounts.api_child_policy("claude", account_profile)
        if policy_err:
            return policy_err
        if budget:
            argv += ["--max-budget-usd", budget]
        denied = refuse_payload(prompt)
        if denied:
            return denied
        child_env = spawn_env(
            {**os.environ, ENV_DEPTH: str(depth + 1)}, overrides,
            scrub_siblings=scrub_siblings)
        if account_auth_isolated:
            child_env = provider_accounts.account_env(child_env, "claude")
        if account_home:
            provider_accounts.apply_home_env(child_env, "claude", account_home, account_home_explicit)
        config_root = provider_accounts.fence_home("claude", account_home) if account_home else overrides.get("CLAUDE_CONFIG_DIR")
        if account_home and config_root is None:
            child_env.pop("CLAUDE_CONFIG_DIR", None)
        # The consult lane is read-only by design: no plugin selector survives.
        scrub_read_only_env(child_env)
        try:
            argv, child_env, _fence_meta = prepare_host_fence(
                argv, provider="claude", workdir=tmp_cwd,
                read_only=True, env=child_env,
                config_dir=config_root,
            )
        except HostFenceError as exc:
            return exc.envelope
        if account_home:
            if not provider_accounts.home_matches("claude", account_profile, account_home):
                return {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
            provider_accounts.apply_home_env(child_env, "claude", account_home, account_home_explicit)
        if account_deadline is not None:
            timeout_s = max(0.0, account_deadline - time.monotonic())
            if timeout_s <= 0:
                return {"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"}
        res = await run_cli(
            argv,
            stdin_data=str(prompt),
            cwd=tmp_cwd,
            env=child_env,
            timeout_s=timeout_s,
            pane_name="omni:claude",
            pre_spawn=pre_spawn,
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        if "returncode" not in res:
            return res  # pre_spawn refusal: the child was never created
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "claude exceeded %ss" % timeout_s}

        out = (res.get("stdout") or "").strip()
        # The sign-in expiry notice is advice, not a failure: it must neither
        # type the call as auth_error nor stand in as the failure's detail.
        err = strip_signin_expiry_warning((res.get("stderr") or "").strip())
        res = dict(res, stderr=err)
        rc = res.get("returncode")

        process_error = provider_accounts.process_failure(res, "claude")
        if process_error:
            return process_error
        account_out = provider_accounts.account_error_text(out)
        structured = provider_accounts.structured_failure("claude", out)
        if structured:
            reported = inband_error("claude", account_out)
            detail = reported[1] if reported else "claude account failure"
            return {"ok": False, "reason": structured,
                    "detail": _profile_detail(detail, account_profile, overrides),
                    "account_retry_safe": False}
        if rc != 0:
            if res.get("signal") is not None:
                return {"ok": False, "reason": "signal_exit",
                        "detail": signal_exit_detail(
                            "claude", -res["signal"], out, err)}
            if isinstance(rc, int) and not isinstance(rc, bool) and rc >= 0:
                failure = inband_error("claude", account_out)
                if failure:
                    reason, detail = failure
                    # inband maps a missing api_error_status to runtime_error;
                    # "Not logged in" still has to type as auth_error.
                    classified = classify_cli_error(detail, err, rc)
                    if classified != "runtime_error":
                        reason = classified
                else:
                    reason = classify_cli_error(account_out, err, rc)
                    detail = safe_detail(err or out or "exit %d" % (
                        rc if rc is not None else -1))
                return {"ok": False, "reason": reason,
                        "detail": _profile_detail(
                            detail, account_profile, overrides)}
            reason = classify_cli_error(account_out, err, rc)
            if reason == "signal_exit":
                detail = signal_exit_detail("claude", rc, out, err)
            else:
                detail = safe_detail(err or out or "exit %d" % (
                    rc if rc is not None else -1))
            return {"ok": False, "reason": reason, "detail": detail}
        # claude exits 0 while reporting is_error in its JSON result envelope;
        # api_error_status carries the real cause (429/401/...).
        failure = inband_error("claude", account_out)
        if failure:
            return {"ok": False, "reason": failure[0],
                    "detail": _profile_detail(failure[1], account_profile, overrides)}
        # See providers/agy.py: judge emptiness after extraction, not on raw stdout.
        text, usage = extract_cli_result("claude", out)
        if not text:
            return {"ok": False, "reason": "empty_response",
                    "detail": safe_detail(err or "claude returned no text")}
        return {"ok": True, "text": text, "model": model, "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}
    finally:
        if tmp_cwd and os.path.isdir(tmp_cwd):
            shutil.rmtree(tmp_cwd, ignore_errors=True)


@local_request_before_spawn()
async def run_claude_subprocess(prompt, model, timeout_s, account_profile="default",
                              effort=None, scrub_siblings=False, account_pinned=False,
                              apply_aliases=False):
    selection = provider_accounts.AccountSelection(
        account_profile, "claude/" + str(model),
        timeout_s, provider="claude", resume=account_pinned)
    if selection.error:
        return selection.exhausted()
    isolated = selection.managed or provider_accounts.ring_pinned(
        "claude", account_profile, account_pinned)
    from providers import admission
    holder = admission.install_ring_skip(selection, lambda account: admission.consider(
        "claude", account.get("account_profile"), model, effort, isolated=isolated,
        host_env=os.environ,
        settings_files=consult_settings_files(
            account.get("account_profile"), account.get("account_home")),
        apply_aliases=apply_aliases))

    async def attempt(account, remaining, managed):
        decision = account.get("_admission")
        run_model, run_effort, _catalog = admission.bound_call(model, effort, decision)
        result = await _run_claude_once(
            prompt, run_model, remaining, account["account_profile"], effort=run_effort,
            scrub_siblings=scrub_siblings or isolated,
            account_home=account["account_home"] if managed or isolated else None,
            account_home_explicit=account.get("account_home_explicit", True),
            account_auth_isolated=isolated, account_deadline=selection.deadline,
            pre_spawn=(lambda: selection.late_api_refusal(account)) if managed else None)
        return admission.stamp(
            result, decision, sent_effort=admission.sent_effort("claude", decision, effort))

    result = await provider_accounts.run_selected(selection, attempt)
    replaced = admission.ring_exhausted("claude", result, holder)
    return replaced if replaced is not None else result
