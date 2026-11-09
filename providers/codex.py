# no-opencc
"""OpenAI Codex CLI provider — subscription-auth, host-fenced pure text pipe.

The consult lane preserves its protected-home read wall and declared file and
CODEX_HOME access. On macOS, every spawn goes through providers.host_fence for
one verified host wrap; that shared operation disables Codex's nested Seatbelt
only after wrapping succeeds. Other platforms keep the existing vendor
--sandbox read-only behavior. The agent lane has its own call path and policy.

The protected home is the passwd pw_dir, never inherited HOME; declared files
under that tree and the existing Codex auth/config state are read exceptions.
The current profile allows reads outside protected home, including sibling temp
paths. It also permits provider network traffic; disabling the vendor sandbox
does not add a network isolation guarantee. Do not put secrets in temp directories
and assume the home read wall hides them. The child HOME remains protected-home
based so inherited HOME cannot redirect that boundary.

The reply is captured via `-o <file>` (output-last-message = the final agent
message ONLY); codex prints its banner/metadata to stderr, not to the reply.
stdout is the fallback if the file is empty.

`files` (native vision): each entry must be an EXISTING local file and is
passed as a repeated `-i <path>` arg. Codex opens the file itself — the server
never reads it, so no OMNI_LLM_SOURCE_ROOTS fence applies and URLs are not
supported. Missing paths are rejected LOUDLY before anything is spawned.
Declared paths under the protected home are added to the host read allow-set.
No files -> argv is byte-identical to the text-only path on non-macOS.
"""
import os
import shutil
import tempfile
import time

from providers.cli_run import run_cli
from providers.egress_gate import refuse_payload
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.config import (
    DEFAULT_TIMEOUT_S,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CODEX_EXTRA_AUTH_MARKERS,
    ENV_DEPTH,
    env_depth,
)
from providers.errors import (
    classify_cli_error,
    exception_detail,
    local_before_spawn,
    local_request_before_spawn,
    note_spawned,
    codex_error_text,
    codex_failure_detail,
    codex_safe_output,
    codex_classification_stderr,
    signal_exit_detail,
)
from providers import kernel_sandbox
from providers.profiles import resolve_profile, resolve_codex_home, spawn_env
from providers import codex_accounts, health, provider_accounts
from providers.usage import extract_cli_result
from providers import vendor_bin


@local_before_spawn
async def _run_codex_once(prompt, model, timeout_s, account_profile="default",
                               reasoning_effort=DEFAULT_CODEX_REASONING_EFFORT,
                               files=None, effort=None, scrub_siblings=False,
                               account_home=None, account_deadline=None, _spawn=None):
    """Pipe prompt -> `codex exec` (non-interactive) -> typed envelope.

    Rides the local ChatGPT/Codex CLI login (subscription auth, no API key), the
    same shape as grok/agy. Runs in an isolated empty tempdir cwd with:
      --skip-git-repo-check   REQUIRED — empty tempdir cwd is not a git repo
      --ephemeral             no session/rollout files persisted to disk

    On Darwin with sandbox-exec, the argv is wrapped by ``kernel_sandbox``
    with ``read_wall=True`` after ``verify_fence`` (fail-closed). That wrap
    is the load-bearing isolation. After the wrap postcondition holds, the
    inner ``--sandbox read-only`` is dropped and
    ``--dangerously-bypass-approvals-and-sandbox`` is added so Codex does
    not apply a second Seatbelt (issue #51). Vendor command-network deny
    does not survive that swap; Darwin consult isolation is the kernel
    profile (home-tree read deny + write allow-set), not vendor
    ``read-only``. Unwrapped consult (fence mechanism absent) keeps
    ``--sandbox read-only`` and never bypasses.
    ``verify_fence`` refusal still fail-closes when the fence is claimed
    available. ``backends_snapshot()["codex"]["read_wall"]`` is True only
    after the verify path succeeds so callers can gate programmatically.

    The reply is captured via `-o <file>` (output-last-message = the final agent
    message ONLY); codex prints its banner/metadata to stderr, not to the reply.
    stdout is the fallback if the file is empty.

    `files` (native vision): each entry must be an EXISTING local file and is
    passed as a repeated `-i <path>` arg. Codex opens the file itself — the
    server never reads it, so no OMNI_LLM_SOURCE_ROOTS fence applies and URLs
    are not supported. Missing paths are rejected LOUDLY before anything is
    spawned. Declared paths under the protected home are added to the kernel
    read re-allow set so vision inputs remain readable under the read wall.
    No files -> argv is byte-identical to the text-only path (aside from the
    kernel wrap prefix when the fence is active).
    """
    files = list(files or [])
    for f in files:
        # Argument validation first: environment errors (cli_missing) come after.
        if not os.path.isfile(f):
            return {"ok": False, "reason": "unsupported_files",
                    "detail": "codex files= wants an existing local file; "
                              "%r is not (URLs unsupported)" % f}

    # Profile before which(): a bad name stays unsupported_profile even when
    # the CLI is missing (same ordering rule as effort).
    overrides, profile_err = (resolve_codex_home(account_profile, account_home)
                              if account_home else resolve_profile("codex", account_profile))
    if profile_err:
        return profile_err

    codex = vendor_bin.which("codex")
    if not codex:
        return {"ok": False, "reason": "cli_missing",
                "detail": "codex not on PATH - install OpenAI Codex CLI, then `codex login`"}
    # Exec the resolved file: the fence admits the in-home install leaf, and
    # Seatbelt refuses an execvp through the ~/.local/bin symlink.
    codex = os.path.realpath(codex)
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    model = str(model) or DEFAULT_CODEX_MODEL
    if effort is not None and str(effort).strip():
        reasoning_effort = effort
    effort = str(reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT)
    tmp_cwd = None
    tmp_out = None
    try:
        tmp_cwd = tempfile.mkdtemp(prefix="codex_cwd_")
        out_fd, tmp_out = tempfile.mkstemp(suffix=".txt", prefix="codex_out_")
        os.close(out_fd)

        argv = [
            codex, "exec", "--json",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
        ]
        for f in files:
            argv += ["-i", f]
        argv += [
            "-c", "model_reasoning_effort=%s" % effort,
            "-m", model,
            "-o", tmp_out,
            "-",
        ]

        # Kernel Seatbelt fence (issue #45 consult read wall + #26 write wall).
        # verify+apply is one fail-closed operation; never spawn unfenced when
        # the fence is claimed available.
        child_env = spawn_env(
            {**os.environ, ENV_DEPTH: str(env_depth() + 1)}, overrides,
            scrub_siblings=scrub_siblings)
        if account_home:
            child_env = codex_accounts.account_env(child_env)
        admitted_files = []
        for f in files:
            # Admit the given spelling. A leaf symlink is LOUD
            # fence_path_rejected (the caller passes the validated target,
            # not the link); a plantable symlink parent is still visible
            # to the ancestor rule. AdmittedPath.path is the single
            # realpath inside admit_path.
            try:
                admitted_files.append(
                    kernel_sandbox.admit_path(
                        f, "literal", label="consult files",
                    )
                )
            except kernel_sandbox.FencePathRejected as exc:
                return {
                    "ok": False,
                    "reason": "fence_path_rejected",
                    "detail": exc.detail,
                }
        # This consult passes its own policy, so agent_fence_policy's vendor
        # binary admission does not run; admit the binary here.
        try:
            ap_bin = kernel_sandbox.admit_vendor_binary(codex)
        except kernel_sandbox.FencePathRejected as exc:
            return {
                "ok": False,
                "reason": "fence_path_rejected",
                "detail": exc.detail,
            }
        fence_reads = admitted_files + ([ap_bin] if ap_bin else [])
        fence_kw = dict(read_wall=True, extra_read_paths=fence_reads)
        codex_home = overrides.get("CODEX_HOME")
        if account_home:
            codex_home = codex_accounts.fence_home(codex_home)
        if codex_home:
            # Re-check what this resolves to NOW. resolve_profile admitted a
            # realpath; admit_path evaluates once and the AdmittedPath is
            # what enters extras. Containment uses the same identity
            # helpers as the agent branch (_ci_same_or_inside /
            # _is_widening_or_home_ancestor) on ap.path, so a firmlink
            # spelling of home is the same object. The last resolver
            # checks identity, so a directory swapped for a symlink in
            # between cannot enter the allow-set (issue #102 / #120).
            try:
                ap_home = kernel_sandbox.admit_path(
                    codex_home, "subpath", label="CODEX_HOME",
                )
            except kernel_sandbox.FencePathRejected as exc:
                return {
                    "ok": False,
                    "reason": "fence_path_rejected",
                    "detail": exc.detail,
                }
            home = kernel_sandbox.protected_home()
            rp_cfg = ap_home.path
            try:
                widening = kernel_sandbox._is_widening_or_home_ancestor(
                    rp_cfg, home,
                )
            except kernel_sandbox._IdentityUnavailable:
                widening = True
            try:
                inside = kernel_sandbox._ci_same_or_inside(rp_cfg, home)
            except kernel_sandbox._IdentityUnavailable:
                inside = True
            exempt = False
            if inside and not widening:
                parent = os.path.dirname(str(rp_cfg).rstrip(os.sep))
                base = os.path.basename(str(rp_cfg).rstrip(os.sep))
                try:
                    parent_is_home = kernel_sandbox._same_dir(parent, home)
                except kernel_sandbox._IdentityUnavailable:
                    parent_is_home = False
                if (parent_is_home
                        and base.casefold().startswith(".codex-")
                        and len(base) > len(".codex-")):
                    exempt = True
            if widening or (inside and not exempt):
                return {
                    "ok": False,
                    "reason": "fence_path_rejected",
                    "detail": "CODEX_HOME now resolves inside the home tree "
                              "without being a dedicated ~/.codex-<name> "
                              "directory: %s -> %s" % (codex_home, ap_home.path),
                }
            fence_kw["extra_read_paths"] = list(fence_reads) + [ap_home]
            fence_kw["extra_write_paths"] = [ap_home]
        # This policy replaces agent_fence_policy's, so its account isolation
        # is rebuilt here. A consult on a NON-default account takes the codex
        # agent fence's profile rows: the default ~/.codex is write- and
        # read-denied (the shared defaults admit it), the selected home is
        # write-denied whole with only vendor state re-allowed (config.toml,
        # hooks.json, AGENTS files and the agents/skills/plugins/rules trees
        # stay read-only; auth.json stays writable for token refresh, the
        # agent fence's accepted exposure), and every other registered account
        # home is read-denied.
        # The default-account consult takes the same structure for the default
        # ~/.codex (codex_consult_default_fence): the whole home is
        # write-denied with only the validated state set re-allowed
        # (sessions, log, cache, tmp, sqlite stores and the named state files
        # including auth.json), the protected configuration names are final
        # write denies, and every registered account home is read-denied.
        try:
            if codex_home:
                fence_kw.update(
                    kernel_sandbox.codex_consult_account_fence(
                        ap_home.path, tmp_cwd,
                    )
                )
            else:
                fence_kw.update(
                    kernel_sandbox.codex_consult_default_fence(tmp_cwd)
                )
        except kernel_sandbox.FencePathRejected as exc:
            return {
                "ok": False,
                "reason": "fence_path_rejected",
                "detail": exc.detail,
            }
        fence_env = child_env
        if account_home and codex_home is None:
            fence_env = dict(child_env)
            fence_env.pop("CODEX_HOME", None)
        try:
            argv, child_env, _fence_meta = prepare_host_fence(
                argv,
                provider="codex",
                workdir=tmp_cwd,
                read_only=True,
                env=fence_env,
                config_dir=codex_home,
                policy={
                    "wrap_kw": fence_kw,
                    "tmpdir": None,
                    "vendor_state_writable": True,
                },
            )
        except HostFenceError as exc:
            return exc.envelope

        if account_home:
            child_env["CODEX_HOME"] = account_home
        denied = refuse_payload(prompt)
        if denied:
            return denied
        if account_home and not codex_accounts.home_matches(account_profile, account_home):
            return {"ok": False, "reason": "account_home_drift", "detail": "original account home changed"}
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
            pane_name="omni:codex",
            on_spawn=lambda _proc: note_spawned(_spawn),
        )
        note_spawned(_spawn)
        if res.get("timed_out"):
            return {"ok": False, "reason": "timeout",
                    "detail": "codex exceeded %ds" % timeout_s, "account_retry_safe": False}

        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")

        if rc != 0:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = signal_exit_detail(
                    "codex", -res["signal"], codex_safe_output(out), "")
            else:
                reason = classify_cli_error(
                    codex_error_text(out, with_codes=True),
                    codex_classification_stderr(out, err), rc,
                    extra_auth_markers=CODEX_EXTRA_AUTH_MARKERS,
                )
                if reason == "signal_exit":
                    detail = signal_exit_detail(
                        "codex", rc, codex_safe_output(out), "")
                else:
                    detail = codex_failure_detail(out, rc)
            return codex_accounts.annotate_failure(
                {"ok": False, "reason": reason, "detail": detail}, out)

        event_reason = classify_cli_error(codex_error_text(out, with_codes=True), "", 1,
                                          extra_auth_markers=CODEX_EXTRA_AUTH_MARKERS)
        if event_reason in ("rate_limit", "auth_error"):
            return codex_accounts.annotate_failure(
                {"ok": False, "reason": event_reason, "detail": "Codex account failure"}, out)
        event_text, usage = extract_cli_result("codex", out)
        reply = ""
        if tmp_out and os.path.exists(tmp_out):
            with open(tmp_out) as f:
                reply = f.read().strip()
        if not reply:
            # event_text only. NEVER fall back to raw stdout here: with --json that
            # is a JSONL event stream, and handing it back would dress protocol
            # noise as a model answer.
            reply = event_text
        if not reply:
            return {"ok": False, "reason": "empty_response",
                    "detail": (codex_safe_output(out) or "codex returned no text")[:300]}
        return {"ok": True, "text": reply, "model": model, "usage": usage}
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e), "account_retry_safe": False}
    finally:
        if tmp_out and os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        if tmp_cwd and os.path.isdir(tmp_cwd):
            shutil.rmtree(tmp_cwd, ignore_errors=True)


# A selection deadline stays unstamped: account selection runs the codex
# binary for its quota check, so that time can be vendor time.
@local_request_before_spawn(unstamped_reasons=("timeout",))
async def run_codex_subprocess(prompt, model, timeout_s, account_profile="default",
                               reasoning_effort=DEFAULT_CODEX_REASONING_EFFORT,
                               files=None, effort=None, scrub_siblings=False, account_pinned=False):
    selection = codex_accounts.AccountSelection(account_profile, "codex/" + str(model), timeout_s, resume=account_pinned)
    if selection.error:
        return selection.exhausted()
    if not selection.managed:
        ring_pin = provider_accounts.ring_pinned("codex", account_profile, account_pinned)
        scrub_siblings = scrub_siblings or ring_pin
        want_pin = ring_pin or (account_pinned and scrub_siblings)
        pin = selection.next() if want_pin else None
        if want_pin and pin is None:
            return selection.exhausted()
        started = time.time()
        result = await _run_codex_once(prompt, model, timeout_s, account_profile,
                                      reasoning_effort, files, effort, scrub_siblings,
                                      **({"account_home": pin["codex_home"],
                                          "account_deadline": selection.deadline} if pin else {}))
        if result.get("ok"):
            # A failure above flags the account key; dispatch only clears the
            # model key on success, so clear the account key here. A flag
            # raised after this request began, and quota/credit evidence,
            # survive the success.
            try:
                health.clear("codex", account_profile or "default",
                             unless_flagged_after=started,
                             keep_sources=health.SUCCESS_KEEP_SOURCES + ("quota",))
            except health.HealthLockUnavailable:
                pass  # best effort: a success must not fail on the ledger lock
        else:
            result.setdefault("account_retry_safe", False)
            # Quota/auth/account failures cool the account so the next request
            # selects another; any other reason only follows the crash streak.
            selection.failed({"account_profile": account_profile or "default"}, result)
        return result
    async def attempt(account, remaining, managed):
        return await _run_codex_once(
            prompt, model, remaining, account["account_profile"],
            reasoning_effort, files, effort, True,
            account_home=account["codex_home"], account_deadline=selection.deadline)
    return await codex_accounts.run_selected(selection, attempt)
