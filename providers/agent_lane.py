# no-opencc
"""Delegation lane — run an agentic CLI (codex / claude / grok / pi / cursor) inside a worktree.

CLI-only by design (no MCP tool): synchronous callers retain the direct typed
envelope, while long runs may ride a detached jobs/ supervisor with a lock,
stream log, result file, and identity-safe stop semantics.

Ordinary macOS workers, including read-only Codex and Grok, use the shared
host verify+wrap operation. The host fence is mandatory on these paths;
vendor sandbox flags are compatibility details. Read-only workers do not
add the workdir to the write allow-set. Vendor state allowances remain
provider-specific in kernel_sandbox.agent_fence_policy: session/runtime
state is writable, while configuration/hooks/install files stay protected.
Codex, Grok and Cursor inner sandboxes are disabled only after a verified host wrap.

Non-macOS behavior retains the existing provider split and fallback opt-ins.
The separately owned foreman=True role retains its prior permission and
admission contract and is not advertised as host-contained by this change.

Fences, in order: recursion guard (OMNI_LLM_DEPTH >= 1 refuses — no nested
agents); workdir is realpath-normalized ONCE and that single value feeds both
the fence and the run (no check/run divergence via symlinks or `..`); a git
workdir whose current branch is main/master (or the origin default branch) is
refused unless allow_main — main checkout or linked worktree alike, and any
INDETERMINATE git state fails CLOSED. Known accepted limits: the fence is
check-then-run (an agent handed the repo can switch branches mid-run); the
envelope text is the agent's own claim — verify against `git diff`, never the
story. Output stays ADVISORY + UNTRUSTED. Darwin write-mode git worktrees
admit repository metadata through the host fence so a linked worktree
can commit; hooks, config, packed-refs, HEAD, and protected-branch
refs stay deny-write. Git remote helpers are process-exec denied. Path denies
cannot express every way Git stores a ref, so a write run also reads each
protected local ref before the spawn and after the exit, from outside the
fence, and a difference fails the run (``protected_ref_changed``).

Success semantics for codex (issue #51): `codex exec` exits 0 even when every
shell command it ran was denied, because a failing tool call is a recoverable
event to codex. The outer return code therefore does not decide ``ok`` — see
``_codex_run_failure``, which judges from the ``--json`` event stream and is
never given a channel carrying caller- or agent-authored text.
"""
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import unicodedata

from providers.attachments import _attach_block
from providers.cli_run import run_cli
from providers.claude import (
    READ_ONLY_DISALLOWED, READ_ONLY_TOOLS, scrub_read_only_env,
)
from providers.egress_gate import refuse_payload
from providers import kernel_sandbox
from providers import known_secrets
from providers import git_safe
from providers import host_fence as host_fence_mod
from providers.host_fence import HostFenceError, prepare_host_fence
from providers.logutil import log_event
from providers.config import (
    AGENT_TIMEOUT_S,
    CODEX_EXTRA_AUTH_MARKERS,
    CURSOR_CONFIG_DIR_ENV,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    DEFAULT_GROK_MODEL,
    DEFAULT_PI_MODEL,
    ENV_DEPTH,
    ENV_GROK_STALL_GRACE,
    GROK_EXTRA_AUTH_MARKERS,
    GROK_STALL_GRACE_S,
    env_depth,
    effort_for_provider,
)
from providers import cursor as cursor_mod
from providers import usage as usage_mod
from providers.pi import child_env as pi_child_env
from providers.errors import (
    classify_cli_error,
    exception_detail,
    redact_secrets,
    safe_detail,
    codex_error_text,
    codex_failure_detail,
    codex_safe_output,
    codex_classification_stderr,
    signal_exit_detail,
)
from providers.profiles import resolve_account_home, ACCOUNT_HOME_KEYS, resolve_profile, resolve_codex_home, spawn_env
from providers import codex_accounts, provider_accounts
from providers.route import resolve_route
from providers.seat_name import take_effort
from providers import sandbox_canary, stream_contract
from providers.attest import read_successor_attest
from providers import vendor_bin

AGENT_PROVIDERS = ("codex", "claude", "grok", "pi", "cursor")

# Claude's --permission-mode choice list (claude.help.txt lines 144-147),
# plus "default" — the omitted-flag spelling (accepted token, not a claude
# CLI choice). For claude, omitted and explicit default emit no
# --permission-mode flag. For codex/grok, default uses the same mapping as
# every other non-plan, non-bypass mode.
PERMISSION_MODES = (
    "acceptEdits",
    "auto",
    "bypassPermissions",
    "manual",
    "dontAsk",
    "plan",
    "default",
)
CODEX_BYPASS_SANDBOX_FLAG = "--dangerously-bypass-approvals-and-sandbox"


class UnsupportedPermissionMode(ValueError):
    """Harness cannot map this permission mode (e.g. grok + plan)."""


def successor_permission_argv(provider, mode):
    """Map a successor permission mode onto vendor argv (foreman mode only).

    Pure: returns a list of flag tokens. grok+plan raises
    ``UnsupportedPermissionMode`` (strict sandbox cannot split read from write).
    For claude, omitted and ``default`` emit no flag (claude CLI has no
    ``default`` choice). Unknown providers raise
    ``UnsupportedPermissionMode``.
    """
    token = (
        "default" if mode is None or str(mode).strip() == ""
        else str(mode).strip()
    )
    if token not in PERMISSION_MODES:
        raise UnsupportedPermissionMode(
            "permission mode must be one of %s; got %r"
            % (", ".join(PERMISSION_MODES), mode)
        )
    if provider == "claude":
        if token == "default":
            return []
        return ["--permission-mode", token]
    if provider == "codex":
        if token == "plan":
            return ["--sandbox", "read-only"]
        if token == "bypassPermissions":
            return [CODEX_BYPASS_SANDBOX_FLAG]
        return ["--sandbox", "workspace-write"]
    if provider == "grok":
        if token == "plan":
            raise UnsupportedPermissionMode(
                "grok cannot map permission mode %r (strict sandbox "
                "cannot separate read from write)" % token
            )
        if token == "bypassPermissions":
            return ["--sandbox", "worker", "--always-approve"]
        return ["--sandbox", "strict"]
    raise UnsupportedPermissionMode(
        "permission mode mapping is not supported for provider %r" % provider
    )


def permission_mode_refusal(provider, permission_mode, foreman=False,
                            read_only=False):
    """LOUD envelope or None. Call before spawn."""
    token = None
    if permission_mode is not None and str(permission_mode).strip():
        token = str(permission_mode).strip()
        if not foreman:
            return {
                "ok": False,
                "reason": "permission_mode_requires_foreman",
                "detail": "--permission-mode requires --foreman",
            }
        if token not in PERMISSION_MODES:
            return {
                "ok": False,
                "reason": "unsupported_permission_mode",
                "detail": "permission mode must be one of %s; got %r"
                          % (", ".join(PERMISSION_MODES), permission_mode),
            }
    if foreman and read_only:
        return {
            "ok": False,
            "reason": "foreman_requires_write_lane",
            "detail": "--foreman cannot combine with --read-only",
        }
    if not foreman:
        return None
    if provider not in ("claude", "codex", "grok"):
        return {
            "ok": False,
            "reason": "foreman_unsupported_provider",
            "detail": "--foreman is supported for claude, codex, and grok only",
        }
    if token is None:
        token = "default"
    try:
        successor_permission_argv(provider, token)
    except UnsupportedPermissionMode as exc:
        return {
            "ok": False,
            "reason": "unsupported_permission_mode",
            "detail": redact_secrets(str(exc))[:300],
        }
    return None


def _foreman_mode_token(permission_mode):
    if permission_mode is None or not str(permission_mode).strip():
        return "default"
    return str(permission_mode).strip()



def _codex_json_event_lines(stdout_text):
    """Yield parseable codex --json event dicts from a stdout dump. Framed on
    LF only: codex writes U+2028, U+2029 and U+0085 raw inside JSON strings,
    and str.splitlines() would break the event line at them (issue #213)."""
    for line in (stdout_text or "").split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(evt, dict) and evt.get("type") is not None:
            yield evt


def _last_codex_agent_message(stdout_text):
    """Extract the last agent_message text from a codex --json event stream.

    Looks for events shaped like
    {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
    and returns the last non-empty text, or "" if none.
    """
    last = ""
    for evt in _codex_json_event_lines(stdout_text):
        if evt.get("type") != "item.completed":
            continue
        item = evt.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            last = text.strip()
    return last


def _stdout_looks_like_codex_json_events(stdout_text):
    """True when stdout carries at least one codex --json event object."""
    for _ in _codex_json_event_lines(stdout_text):
        return True
    return False


# Issue #51: `codex exec` exits 0 even when every shell command it ran died,
# because a failing tool call is a recoverable event to codex. The signature of
# the environment fault this detector exists for is sysexits EX_OSERR — what
# sandbox-exec returns when it cannot apply a profile, which is what every
# command returns when codex applies its own Seatbelt policy inside an outer
# Seatbelt sandbox (macOS refuses a nested sandbox_apply).
_SANDBOX_APPLY_RC = 71

_CODEX_COMMAND_ITEM_TYPES = ("command_execution", "local_shell_call")
_CODEX_FAILED_STATUSES = ("failed", "error")


def _codex_exit_code(item):
    """A command item's exit code as an int, or None when it cannot be read.

    codex event fields are caller-shaped JSON and their types drift between
    versions: ``exit_code`` has been seen as an int and as a decimal string,
    and sibling fields such as ``output`` arrive as dicts. Anything that is not
    an int or a digit-string yields None rather than raising — an exception
    here would be laundered by the outer handler into a wrong envelope. ``bool``
    is rejected explicitly because ``True == 1``.
    """
    code = item.get("exit_code")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        try:
            return int(code.strip())
        except ValueError:
            return None
    return None


def _codex_command_outcomes(stdout_text):
    """(total, failed, env_faults) over codex command events.

    THE INVARIANT: only machine-generated fields are read — the event type, the
    item type, ``status`` and ``exit_code``. The ``command`` and ``output``
    fields are never inspected, so no part of this verdict can be driven by
    text the caller or the agent can author. That is a structural property of
    what this function reads, not a list of exclusions: an agent may print, and
    a caller may ask about, anything at all without moving these counters.

    ``env_faults`` counts failures whose exit code is exactly 71 — the
    objective signal that a command never ran. A command that merely returns
    non-zero (a red test, a grep with no match) is a legitimate task RESULT, so
    it lands in ``failed`` but never in ``env_faults``.
    """
    total = failed = env_faults = 0
    for evt in _codex_json_event_lines(stdout_text):
        if evt.get("type") != "item.completed":
            continue
        item = evt.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") not in _CODEX_COMMAND_ITEM_TYPES:
            continue
        total += 1
        code = _codex_exit_code(item)
        status = str(item.get("status") or "").lower()
        if status not in _CODEX_FAILED_STATUSES and code in (0, None):
            continue
        failed += 1
        if code == _SANDBOX_APPLY_RC:
            env_faults += 1
    return total, failed, env_faults


def _codex_run_failure(stdout_text):
    """Refusal envelope when a codex run that exited 0 did no real work.

    `codex exec` treats a failing shell tool call as recoverable and still
    exits 0, so the outer return code alone cannot decide ``ok`` (issue #51):
    a run whose every command was denied returned ok:true with the agent's
    account of the work as ``text``.

    The verdict is read from the ``--json`` event stream and NOTHING else.
    Both lanes pass ``--json``, so both produce that stream; there is no
    free-text arm to fall back to, and deliberately so. Earlier revisions
    pattern-matched the denial string in codex's stderr transcript, which
    turned out to carry the CALLER'S OWN PROMPT echoed verbatim before the
    model is even called — asking this lane to "fix the sandbox_apply bug"
    killed its own correct answer. Free text near this decision is a standing
    false-positive generator; the signature of this function is one argument
    for that reason.

    Disqualifying shape: at least one command ran, EVERY command failed, and at
    least one of those failures exited 71 (see ``_codex_command_outcomes``).

    Two accepted limits, both erring toward returning the agent's answer:

      * ``failed == total`` means a single incidental success (a ``true``, an
        ``echo``) masks an otherwise wholesale denial. Kept deliberately: the
        opposite error — discarding a correct answer because one command in a
        long run hit rc 71 — destroys real work, which is the failure mode
        review has caught here repeatedly.
      * a run with NO command events is never disqualified. That covers a
        legitimate question-answering run that ran nothing, and it also means a
        denial that happens before the first command event is reported as
        success. Detecting that needs a machine-readable signal codex does not
        currently emit; a text scan is not an acceptable substitute.

    Returns None when nothing disqualifying was seen. Never raises.
    """
    total, failed, env_faults = _codex_command_outcomes(stdout_text)
    if total and failed == total and env_faults:
        return {
            "ok": False,
            "reason": "runtime_error",
            "detail": (
                "codex exited 0 but all %d shell command(s) failed, %d of them "
                "with an environment fault (exit 71: the command never ran); "
                "the reply is not backed by any successful command"
                % (total, env_faults)
            ),
        }
    return None


def _git(workdir, *args):
    """One git query; returns (rc, stdout, stderr) — both streams stripped.

    Any exception -> (None, "", "<message>"), which callers MUST treat as an
    indeterminate git state (fail closed), never as "not a repository".

    The SYNCHRONOUS subprocess.run is deliberate, not an oversight of the
    run_cli-only rule: these are sub-second local read-only git queries with a
    hard 10s timeout, and run() kills the child on timeout. Accepted exception
    — the fence must answer before any async CLI run is started.

    Runs in the PARENT, outside the child fence, so it goes through
    ``git_safe.run_git``: repo-chosen commands (fsmonitor, filter drivers,
    hooks) must not run (issue #157).
    """
    try:
        p = git_safe.run_git(os.fspath(workdir), args)
        return (p.returncode, (p.stdout or "").strip(),
                redact_secrets((p.stderr or "").strip()))
    except Exception as e:
        return None, "", redact_secrets(str(e))[:120]


def _git_ref(workdir, *args):
    """``_git`` for a ref name: only Git's own trailing newline is removed.

    A ref name may end in other whitespace (a no-break space is a valid
    branch name); ``_git`` strips both streams and would name a different ref.
    """
    try:
        p = git_safe.run_git(os.fspath(workdir), args)
        return (p.returncode, host_fence_mod.ref_line(p.stdout),
                redact_secrets((p.stderr or "").strip()))
    except Exception as e:
        return None, "", redact_secrets(str(e))[:120]


def _fence_refusal(detail):
    # Masked here, once: a refusal can quote a branch name that looks like a key.
    return {"ok": False, "reason": "workdir_fence", "detail": redact_secrets(detail)}


def _parse_porcelain(out):
    """``{path: status}`` from ``git status --porcelain`` output.

    A rename line (``R  old -> new``) is keyed by the new path with the old
    path folded into the status, so a rename counts as one changed path.
    """
    entries = {}
    for line in out.splitlines():
        if len(line) < 4:
            continue
        xy, rest = line[:2], line[3:]
        if xy[0] in "RC" and " -> " in rest:
            old, new = rest.split(" -> ", 1)
            entries[new] = "%s:%s" % (xy, old)
        else:
            entries[rest] = xy
    return entries


def _workspace_snapshot(workdir):
    """Porcelain entries plus HEAD, or why they could not be read (issue #65).

    Returns ``{"state": "ok", "entries": {...}, "head": sha|None}``,
    ``{"state": "not_git"}`` when the workdir is not a git tree, or
    ``{"state": "error", "detail": ...}`` when git could not answer (timeout,
    corrupt index). Only ``not_git`` and ``error`` become
    ``workspace_writes: null``; they are reported apart so a slow or broken
    git is never described as "not a git tree".
    """
    # --ignore-submodules=dirty: a submodule's own status would be a child git
    # reading the submodule's config and attributes; its moved HEAD is still
    # reported, its work-tree content is not (issue #157).
    rc, out, err = _git(workdir, "status", "--porcelain", "--untracked-files=all",
                        "--ignore-submodules=dirty")
    if rc == 0:
        hrc, head, _herr = _git(workdir, "rev-parse", "--verify", "-q", "HEAD")
        return {"state": "ok", "entries": _parse_porcelain(out),
                "head": head if hrc == 0 and head else None}
    if rc == 128 and "not a git repository" in (err or "").lower():
        return {"state": "not_git", "detail": "%s is not a git tree" % workdir}
    return {"state": "error",
            "detail": ("git status failed (rc=%s): %s" % (rc, err))[:200]}


def _workspace_writes(workdir, before, after):
    """``(count, unknown_detail)``: paths whose status changed, or why unknown.

    A path counts once whatever happened to its status letters. When HEAD
    moved during the run (the agent committed), the files in that commit
    range are added, so "write then commit" is not mistaken for a no-op.
    """
    for snap in (before, after):
        if snap.get("state") != "ok":
            return None, snap.get("detail") or "workspace state unknown"
    b, a = before["entries"], after["entries"]
    changed = {path for path in set(b) | set(a) if b.get(path) != a.get(path)}
    if before.get("head") and after.get("head") and before["head"] != after["head"]:
        rc, out, _err = _git(workdir, "diff", "--name-only",
                             "--no-ext-diff", "--no-textconv",
                             before["head"], after["head"])
        if rc != 0:
            # Without the commit range the count would silently miss committed
            # work (git unavailable, refused by the safe wrapper, objects gone).
            return None, ("git diff of the commit range failed (rc=%s): %s"
                          % (rc, (_err or "")))[:200]
        changed |= {line for line in out.splitlines() if line.strip()}
    return len(changed), None


def _stamp_fence(env, fence="none", detail=None):
    """Attach honest ``fence`` / ``fence_detail``; never overwrite a set value."""
    if not isinstance(env, dict):
        return env
    out = dict(env)
    if isinstance(out.get("detail"), str):
        # The one place agent envelopes leave: a detail can quote a branch or
        # ref name that looks like a credential.
        out["detail"] = redact_secrets(out["detail"])
    out.setdefault("fence", fence)
    out.setdefault(
        "fence_detail",
        dict(detail or kernel_sandbox.FENCE_NONE_DETAIL),
    )
    return out


def _fence_unavailable(detail):
    return _stamp_fence({
        "ok": False,
        "reason": "fence_unavailable",
        "detail": detail,
    })


def _vendor_fence_detail(provider, read_only):
    if provider == "grok":
        return {
            "read_wall": False,
            "workdir_writable": True,
            "vendor_state_writable": False,
        }
    return {
        "read_wall": False,
        "workdir_writable": not read_only,
        "vendor_state_writable": False,
    }


def resolve_agent_binary(provider):
    """Absolute vendor CLI path (override env, else PATH)."""
    return kernel_sandbox.resolve_agent_binary(provider)


def _composed_write_roots(wrap_kw, workdir_abs, read_only):
    """Every write root of the composed policy (raises FencePathRejected)."""
    roots = list(kernel_sandbox._write_allow_paths(
        workdir_abs, read_only,
        extra_write_paths=wrap_kw.get("extra_write_paths")))
    for entry in wrap_kw.get("extra_post_write") or ():
        bound = kernel_sandbox._write_entry_bound(*entry)
        if bound and bound[0] == "subpath":
            roots.append(bound[1])
    return roots


API_SETTINGS_PREFIX = "omni_agent_api_"
API_SETTINGS_STALE_S = 24 * 3600


def _base_identity(info):
    """What must stay the same between validating a base and creating in it."""
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode)


class _PinnedBase(str):
    """A validated base path that remembers the identity it validated as."""
    ident = None


def _validated_private_base(path, create=True):
    """Canonical *path* once the whole chain is trustworthy, else None.

    Every ancestor from the filesystem root down must be a directory owned by
    the current user or root and not group/other-writable (unless sticky and
    root-owned, like /tmp). Below the first user-owned component no symlink is
    allowed; a root-owned symlink above it (``/var``, ``/tmp``) is followed.
    The base itself must be a real directory owned by the user with mode
    exactly 0700. Missing components are created one by one, 0700, only after
    their parent was judged (never with ``create=False``). Any mismatch fails
    closed.
    """
    real, _ident = _validated_private_base_ident(path, create)
    return real


def _validated_private_base_ident(path, create=True):
    """``(canonical path, identity)`` of a validated base, else ``(None, None)``."""
    uid = os.getuid()

    def walk(target, create):
        user_seen = False
        cur = os.sep
        for comp in [c for c in target.split(os.sep) if c]:
            cur = os.path.join(cur, comp)
            try:
                info = os.lstat(cur)
            except FileNotFoundError:
                if not create:
                    return False
                os.mkdir(cur, 0o700)
                info = os.lstat(cur)
            if stat.S_ISLNK(info.st_mode):
                if user_seen or info.st_uid != 0:
                    return False
                info = os.stat(cur)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, uid):
                return False
            if info.st_uid == uid and uid != 0:
                user_seen = True
            if info.st_mode & 0o022 and not (
                    info.st_mode & stat.S_ISVTX and info.st_uid == 0):
                return False
        return True

    try:
        target = os.path.abspath(path)
        if not walk(target, create):
            return None, None
        real = os.path.realpath(target)
        if real != target and not walk(real, False):
            return None, None
        info = os.lstat(real)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != uid
                or stat.S_IMODE(info.st_mode) != 0o700):
            return None, None
        return real, _base_identity(info)
    except OSError:
        return None, None


_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _new_run_settings_dir(base):
    """Fresh per-run directory inside the validated *base*: ``(path, fd)`` or None.

    The base is opened O_DIRECTORY|O_NOFOLLOW and its (device, inode, owner,
    mode) must equal what validation saw, so a base swapped in between is
    refused. The run directory is made relative to that descriptor with a
    random name and checked through a no-follow open of its own; the returned
    descriptor is the one to write the settings through. The path reported is
    built from the validated base. The caller closes the descriptor.
    """
    ident = getattr(base, "ident", None)
    base_fd = run_fd = None
    name = None
    try:
        if ident is None:
            info = os.lstat(base)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                return None
            ident = _base_identity(info)
        base_fd = os.open(base, _DIR_FLAGS)
        if _base_identity(os.fstat(base_fd)) != ident:
            return None
        name = API_SETTINGS_PREFIX + secrets.token_hex(8)
        os.mkdir(name, 0o700, dir_fd=base_fd)
        run_fd = os.open(name, _DIR_FLAGS, dir_fd=base_fd)
        info = os.fstat(run_fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise OSError("run directory is not private")
        # The reported path goes through the base spelling: it must still be
        # the directory this descriptor pinned.
        if _base_identity(os.lstat(base)) != ident:
            raise OSError("base changed during creation")
        done = (os.path.join(base, name), run_fd)
        run_fd = None
        return done
    except OSError:
        if name is not None and base_fd is not None:
            try:
                if run_fd is not None:
                    for leaf in os.listdir(run_fd):
                        os.unlink(leaf, dir_fd=run_fd)
                os.rmdir(name, dir_fd=base_fd)
            except OSError:
                pass
        return None
    finally:
        if run_fd is not None:
            os.close(run_fd)
        if base_fd is not None:
            os.close(base_fd)


def _write_api_settings_in(label, home, run_fd, directory, name="api-settings.json"):
    """Write the API settings (0600) relative to *run_fd*: ``(path, err)``.

    Same contract and errors as ``provider_accounts.write_api_settings``, but
    the file is created O_EXCL|O_NOFOLLOW inside the pinned directory, never
    through the path.
    """
    import json
    helper = provider_accounts.read_api_key_helper(home) if home else None
    if helper is None:
        return None, {"ok": False, "reason": "account_unavailable",
                      "detail": "API account %s has no usable apiKeyHelper setting (missing, unreadable, not a plain one-line command, or it embeds a key-shaped literal)" % label}
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=run_fd)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump({"apiKeyHelper": helper}, handle)
    except OSError as exc:
        return None, {"ok": False, "reason": "runtime_error",
                      "detail": "could not write API settings for %s: %s" % (label, exc.__class__.__name__)}
    return os.path.join(directory, name), None


def _sweep_all_api_settings():
    """Best-effort: sweep leftovers in whichever settings base already exists.

    Runs at the start of any fenced spawn. A base that is absent or does not
    validate is skipped; nothing is ever created here.
    """
    try:
        bases = host_fence_mod.api_settings_bases()
    except Exception:  # noqa: BLE001 - nothing to sweep without a home
        return
    for base in bases:
        try:
            real = _validated_private_base(base, create=False)
            if real is not None:
                _sweep_stale_api_settings(real)
        except Exception:  # noqa: BLE001 - housekeeping must never block a spawn
            continue


def _sweep_stale_api_settings(base, now=None):
    """Best-effort removal of per-run directories older than 24 h.

    Only names with the omni prefix, only directories owned by the user, only
    regular files inside; nothing is followed through a symlink.
    """
    horizon = (time.time() if now is None else now) - API_SETTINGS_STALE_S
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        base_fd = os.open(base, flags)
    except OSError:
        return
    try:
        for name in os.listdir(base_fd):
            if not name.startswith(API_SETTINGS_PREFIX):
                continue
            try:
                info = os.stat(name, dir_fd=base_fd, follow_symlinks=False)
                if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mtime > horizon):
                    continue
                run_fd = os.open(name, flags, dir_fd=base_fd)
                try:
                    for leaf in os.listdir(run_fd):
                        leaf_info = os.stat(leaf, dir_fd=run_fd, follow_symlinks=False)
                        if stat.S_ISREG(leaf_info.st_mode):
                            os.unlink(leaf, dir_fd=run_fd)
                finally:
                    os.close(run_fd)
                os.rmdir(name, dir_fd=base_fd)
            except OSError:
                continue
    except OSError:
        pass
    finally:
        os.close(base_fd)


def _api_settings_base(roots=None):
    """Validated private 0700 directory outside the child's write roots, or None.

    The generated API settings must not sit where the child can write: a
    literal deny covers the file, but a writable ancestor could still be
    renamed away and rebuilt. Candidates are the two fixed bases of
    ``host_fence.api_settings_bases`` (state dir area, then a per-user area
    under the real home); the first whose whole chain validates and that no
    write root can reach wins. *roots* is the composed write-root set (default
    roots only when omitted); ``prepare_host_fence`` re-checks the final set
    fail-closed and is authoritative.
    """
    try:
        if roots is None:
            roots = kernel_sandbox._write_allow_paths(None, True)
        candidates = host_fence_mod.api_settings_bases()
    except Exception:  # noqa: BLE001 - no private area can be proven
        return None
    entries = [("subpath", root) for root in roots]
    for base in candidates:
        try:
            # Judge the spelling before creating anything, then again once it
            # exists (a symlink planted meanwhile resolves elsewhere).
            if host_fence_mod.settings_base_exposed(base, entries):
                continue
            real, ident = _validated_private_base_ident(base)
            if real is None or host_fence_mod.settings_base_exposed(real, entries):
                continue
            pinned = _PinnedBase(real)
            pinned.ident = ident
            return pinned
        except OSError:
            continue
    return None


def _api_settings_fence(wrap_kw, workdir_abs, read_only, settings_path):
    """Final-position deny-write rows for the generated settings, or a refusal.

    Returns ``(rows, None)`` or ``(None, detail)``. Rows cover the exact file
    (content write, unlink, rename-over) and its private directory (rename of
    the directory itself). The settings directory must lie outside every
    write root of the composed policy, so no writable ancestor can be renamed
    away and rebuilt.
    """
    directory = os.path.dirname(settings_path)
    try:
        roots = _composed_write_roots(wrap_kw, workdir_abs, read_only)
    except kernel_sandbox.FencePathRejected as exc:
        return None, exc.detail
    if host_fence_mod.settings_base_exposed(
            directory, [("subpath", root) for root in roots]):
        return None, "API settings directory lies inside a child write root"
    rows = tuple(kernel_sandbox._deny_entries("literal", settings_path)
                 + kernel_sandbox._deny_entries("subpath", directory))
    return rows, None


def _apply_kernel_fence(provider, argv, env, workdir_abs, read_only,
                        fence_requested, config_dir=None, launcher_roots=(),
                        allow_main=False, extra_read_paths=(), api_settings=None):
    """Shared host operation with this lane's existing fallback contract.

    Returns ``(argv, env, fence, fence_detail, refusal)``. *refusal* is an
    envelope when the spawn must not happen. ``config_dir`` is this run's
    vendor config root (cursor ``CURSOR_CONFIG_DIR`` or a claude profile
    ``CLAUDE_CONFIG_DIR``). ``launcher_roots`` is the explicit allowlist
    for launcher-owned home descendants (jobs root / temp root).
    """
    requested = (
        kernel_sandbox.agent_fence_switch_on()
        if fence_requested is None else bool(fence_requested)
    )
    none_detail = dict(kernel_sandbox.FENCE_NONE_DETAIL)
    if not requested:
        log_event("agent_unfenced", provider=provider)
        return argv, env, "none", none_detail, None

    if not kernel_sandbox.available():
        if platform.system() == "Darwin" or not kernel_sandbox.unfenced_fallback_allowed():
            if platform.system() == "Darwin":
                detail = (
                    "kernel fence unavailable "
                    "(/usr/bin/sandbox-exec missing or not executable)"
                )
            else:
                detail = (
                    "kernel fence unavailable off Darwin; set "
                    "OMNI_LLM_AGENT_UNFENCED=1 to spawn unfenced"
                )
            return argv, env, "none", none_detail, _fence_unavailable(detail)
        log_event("agent_unfenced", provider=provider, cause="off_darwin_optin")
        return argv, env, "none", none_detail, None

    _sweep_all_api_settings()
    policy = None
    if api_settings:
        # An API spawn's generated settings must be unwritable to the child:
        # build the usual policy, append the final-position deny, hand it on.
        policy = kernel_sandbox.agent_fence_policy(
            provider, workdir=workdir_abs, read_only=read_only,
            config_dir=config_dir, launcher_roots=launcher_roots)
        if policy.get("ok") is False:
            return argv, env, "none", none_detail, policy
        rows, detail = _api_settings_fence(
            policy["wrap_kw"], workdir_abs, read_only, api_settings)
        if detail:
            return argv, env, "none", none_detail, {
                "ok": False, "reason": "fence_path_rejected", "detail": detail}
        policy["wrap_kw"]["extra_deny_write_final"] = tuple(
            policy["wrap_kw"].get("extra_deny_write_final") or ()) + rows
    try:
        argv, env, metadata = prepare_host_fence(
            argv, provider=provider, workdir=workdir_abs, read_only=read_only,
            env=env, config_dir=config_dir, launcher_roots=launcher_roots,
            allow_main=allow_main, extra_read_paths=extra_read_paths,
            **({"policy": policy} if policy is not None else {}),
        )
    except HostFenceError as exc:
        return argv, env, "none", none_detail, exc.envelope
    return argv, env, metadata["fence"], metadata["fence_detail"], None


def _protected_names(workdir, common, record=True):
    """``(names, None)`` or ``(None, refusal envelope)`` for the protected branches.

    *common* is what ``rev-parse --git-common-dir`` printed (relative to the
    workdir). See ``host_fence.protected_branches`` for where the names come from.
    """
    common_dir = os.path.realpath(os.path.join(os.fspath(workdir), common))

    def origin_probe():
        rc, ref, _err = _git_ref(workdir, "symbolic-ref", "-q",
                                 "refs/remotes/origin/HEAD")
        return rc, ref

    try:
        names, why = host_fence_mod.protected_branches(
            common_dir, origin_probe, record=record)
    except host_fence_mod.ProtectedListError as exc:
        return None, {"ok": False, "reason": "fence_path_rejected",
                      "detail": exc.detail}
    if why:
        return None, _fence_refusal("%s - fence fails closed" % why)
    return names, None


_REFS_HEADS = "refs/heads/"
_HEX_OID = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")
PROTECTED_REF_CHANGED = "protected_ref_changed"
PROTECTED_REF_UNVERIFIED = "protected_ref_unverified"
# A finished run whose verdict is about the repository's integrity: stop
# handling must not turn these into a plain "stopped".
INTEGRITY_REASONS = frozenset((PROTECTED_REF_CHANGED, PROTECTED_REF_UNVERIFIED))


def _protected_ref_names(workdir):
    """``(names, common_dir, None)``, ``(None, None, None)`` outside a repository,
    or ``(None, None, refusal)``. *common_dir* is canonical and pinned here."""
    if not os.path.isdir(workdir):
        return None, None, None
    rc, common, err = _git(workdir, "rev-parse", "--git-common-dir")
    if rc != 0 or not common:
        if rc is not None and "not a git repository" in err.lower():
            return None, None, None
        return None, None, _fence_refusal(
            "git state could not be determined (%s) - protected refs cannot be "
            "verified, fence fails closed" % ((err or "git exit %s" % rc)[:200]))
    names, refusal = _protected_names(workdir, common)
    if refusal:
        return None, None, refusal
    return names, os.path.realpath(os.path.join(os.fspath(workdir), common)), None


def _watched(rest, names):
    """*rest* (a name under refs/heads) is a protected name, or lies below one.

    Compared the way admission compares (case, NFC/NFD folded). A ref BELOW a
    protected name is a directory/file clash that can shadow it.
    """
    return host_fence_mod.ref_is_watched(rest, names)


def _read_protected_refs(common_dir, names, exclude=None, symrefs=None):
    """``({ref: (oid, symbolic target or None)}, None)`` or ``(None, why)``.

    Every branch ref whose name matches a protected name under admission's own
    folding is listed, so a variant spelling created during a run shows up as
    a new key. All refs are listed and filtered here: a prefix pattern relies
    on ``packed-refs`` being sorted, and a rewritten file need not be. The repository is named explicitly (``--git-dir`` on the
    canonical common dir pinned before the spawn), never found again from the
    workdir, where a linked worktree's ``commondir`` / ``gitdir`` files or a
    ``.git`` pointer could name a decoy. A ``commondir`` file in the common
    dir itself would redirect even that read, so its presence is a failure.
    *exclude* is a branch the run is allowed to work on (``--allow-main``).
    *symrefs*, when a list, receives every watched ref that is a symbolic ref,
    the excluded one included.
    """
    if os.path.lexists(os.path.join(common_dir, "commondir")):
        return None, "the common dir carries a commondir file"
    try:
        proc = git_safe.run_git(
            None, ("for-each-ref", "--format=%(refname)%00%(objectname)%00%(symref)"),
            git_dir=common_dir)
    except Exception as exc:
        return None, redact_secrets(str(exc))[:120]
    if proc.returncode != 0:
        return None, ("git exit %s: %s" % (
            proc.returncode, redact_secrets((proc.stderr or "").strip())))[:160]
    refs = {}
    listed = set()
    for line in (proc.stdout or "").split("\n"):
        if not line:
            continue
        fields = line.split("\0")
        if len(fields) != 3 or not _HEX_OID.fullmatch(fields[1]):
            return None, "unexpected for-each-ref line"
        ref, oid, symref = fields
        if not ref.startswith(_REFS_HEADS):
            continue
        rest = ref[len(_REFS_HEADS):]
        listed.add(unicodedata.normalize("NFC", rest))
        if not _watched(rest, names):
            continue
        if symrefs is not None and symref:
            symrefs.append(ref)
        # Only the exact checked-out ref: a variant spelling of it is another ref.
        if exclude and rest == exclude:
            continue
        refs[ref] = (oid, symref or None)
    # A ref whose object is missing is skipped by for-each-ref: absent from the
    # listing is not absent from the repository. A watched name that the files
    # themselves hold (a loose file, a packed entry) but the listing lacks
    # cannot be compared, so the whole read fails instead of passing as empty.
    stored, why = _stored_branch_names(common_dir)
    if stored is None:
        return None, why
    for rest in sorted(stored):
        if (unicodedata.normalize("NFC", rest) not in listed and _watched(rest, names)
                and not (exclude and rest == exclude)):
            return None, "%s%s exists but cannot be listed (missing object or broken ref)" % (
                _REFS_HEADS, rest)
    return refs, None


def _stored_branch_names(common_dir):
    """``(names under refs/heads held by loose files or packed-refs, None)`` or ``(None, why)``."""
    found = set()
    failed = []

    def onerror(exc):
        failed.append(exc)

    heads = os.path.join(common_dir, "refs", "heads")
    # A packed-only repository may have no refs/heads at all.
    for current, _dirs, files in (os.walk(heads, onerror=onerror)
                                  if os.path.isdir(heads) else ()):
        for name in files:
            if name.endswith(".lock"):
                continue
            found.add(os.path.relpath(os.path.join(current, name), heads).replace(os.sep, "/"))
    if failed:
        return None, "refs/heads cannot be walked (%s)" % type(failed[0]).__name__
    try:
        with open(os.path.join(common_dir, "packed-refs"), "rb") as stream:
            text = stream.read().decode("utf-8", "surrogateescape")
    except FileNotFoundError:
        text = ""
    except OSError as exc:
        return None, "packed-refs cannot be read (%s)" % type(exc).__name__
    for line in text.split("\n"):
        if not line or line[0] in "#^":
            continue
        _oid, _sep, ref = line.partition(" ")
        if ref.startswith(_REFS_HEADS):
            found.add(ref[len(_REFS_HEADS):])
    return found, None


def _clear_process_groups(pids, wait_s=1.0):
    """SIGKILL the child's process group once the leader has exited; best effort.

    The child leads its own session, so its pid is its pgid. POSIX does not
    reuse a process ID while a process group with that ID exists, so the number
    cannot be recycled while any member (a lingering descendant or an unreaped
    zombie) is alive, and ``killpg`` can only reach our own group. The recycle
    window only opens once the group is empty, when there is nothing of ours
    left to kill: ``killpg`` then finds nothing (the normal case), but a pid
    freed by the reap could in principle have been handed out again before
    this call and the signal would reach an unrelated group. macOS hands out
    pids sequentially and wraps at 99999, so that needs the counter to go round
    the whole range within the milliseconds after the reap; it is not a
    realistic event and is accepted as the residual of #186. Hosts with
    randomized pid allocation would need an owned anchor that kills its own
    group (a member calling ``kill(0, SIGKILL)``, e.g. a tiny POSIX sh
    launcher); not built. ``cli_run`` skips the kill after a clean reap, but
    its ``returncode is None`` check only narrows the same window. A descendant
    that called ``setsid`` is not reached. One call per run, right after
    ``run_cli`` returned (or raised).
    """
    for pid in pids:
        if not pid or pid <= 1:
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            continue
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except OSError:
                break
            time.sleep(0.01)


def _describe_ref(state):
    if state is None:
        return "(absent)"
    oid, target = state
    return "symbolic ref to %s" % target if target else oid


class _ProtectedRefs:
    """The protected refs as read before the spawn; ``failure()`` compares."""

    def __init__(self, common_dir, names, identity, exclude, before):
        self.common_dir = common_dir
        self.names = names
        self.identity = identity
        self.exclude = exclude
        self.before = before

    def failure(self, workspace_writes=None, note=None):
        """The loud envelope for a run that changed a protected ref, or None."""
        try:
            st = os.stat(self.common_dir)
            same = (st.st_dev, st.st_ino) == self.identity
        except OSError:
            same = False
        after, why = (_read_protected_refs(self.common_dir, self.names, self.exclude)
                      if same else (None, "the common dir was replaced"))
        suffix = " (%s)" % note if note else ""
        if after is None:
            return {"ok": False, "reason": PROTECTED_REF_UNVERIFIED,
                    "detail": redact_secrets(
                        "protected refs could not be read after the run (%s); "
                        "check them before trusting this repository%s" % (why, suffix)),
                    "workspace_writes": workspace_writes}
        changed = sorted(ref for ref in set(self.before) | set(after)
                         if self.before.get(ref) != after.get(ref))
        if not changed:
            return None
        shown = "; ".join("%s: %s -> %s" % (
            ref, _describe_ref(self.before.get(ref)), _describe_ref(after.get(ref)))
            for ref in changed[:6])
        detail = (
            "the run changed a protected local ref (%s). The previous value is "
            "above and in the reflog (git reflog show <branch>); restore it with "
            "git update-ref <ref> <old value>. An update made by someone else "
            "while the run was going cannot be told apart from the run's own.%s"
            % (shown, suffix))
        return {"ok": False, "reason": PROTECTED_REF_CHANGED,
                "detail": redact_secrets(detail), "workspace_writes": workspace_writes}


def _symref_refusal(workdir, names, common_dir=None):
    """A ``fence_path_rejected`` envelope when a protected local name is a symbolic ref."""
    def run(*args):
        return git_safe.run_git(os.fspath(workdir), args)
    try:
        why = host_fence_mod.protected_symref_refusal(run, names) or (
            host_fence_mod.loose_symref_refusal(common_dir, names) if common_dir else None)
    except Exception as exc:
        why = "protected branches cannot be inspected (%s)" % redact_secrets(str(exc))[:120]
    if why:
        return {"ok": False, "reason": "fence_path_rejected", "detail": redact_secrets(why)}
    return None


def _alias_refusal(workdir, common_dir):
    """A ``fence_path_rejected`` envelope when a ref or a git metadata path is a symlink."""
    rc, git_dir, _err = _git(workdir, "rev-parse", "--absolute-git-dir")
    rc2, root, _err2 = _git(workdir, "rev-parse", "--show-toplevel")
    why = (host_fence_mod.ref_alias_refusal(common_dir)
           or host_fence_mod.metadata_alias_refusal(
               os.path.realpath(git_dir) if rc == 0 and git_dir else common_dir,
               common_dir, os.path.realpath(root) if rc2 == 0 and root else None))
    if why:
        return {"ok": False, "reason": "fence_path_rejected", "detail": redact_secrets(why)}
    return None


def _snapshot_protected_refs(workdir, allow_main):
    """``(_ProtectedRefs or None, refusal or None)``, read before the spawn.

    With *allow_main* only the branch the workdir has checked out is exempt:
    that is the one a run may legitimately move.
    """
    names, common_dir, refusal = _protected_ref_names(workdir)
    if refusal:
        return None, refusal
    if names is None:
        return None, None
    alias = _alias_refusal(workdir, common_dir)
    if alias:
        return None, alias
    symref = _symref_refusal(workdir, names, common_dir)
    if symref:
        return None, symref
    exclude = None
    if allow_main:
        rc, ref, _err = _git_ref(workdir, "symbolic-ref", "-q", "HEAD")
        if rc == 0 and ref.startswith(_REFS_HEADS):
            exclude = ref[len(_REFS_HEADS):]
    try:
        st = os.stat(common_dir)
    except OSError as exc:
        return None, _fence_refusal("protected refs cannot be read (%s) - fence "
                                    "fails closed" % (exc.strerror or "stat"))
    symrefs = []
    before, why = _read_protected_refs(common_dir, names, exclude, symrefs)
    if before is None:
        return None, _fence_refusal("protected refs cannot be read (%s) - fence "
                                    "fails closed" % why)
    if symrefs:
        # Any spelling, the exempt branch included: a symbolic ref reaches its
        # target under another name.
        return None, {"ok": False, "reason": "fence_path_rejected",
                      "detail": redact_secrets(
                          "protected branch %s is a symbolic ref" % symrefs[0])}
    return _ProtectedRefs(common_dir, names, (st.st_dev, st.st_ino), exclude, before), None


def _workdir_fence(workdir, allow_main, read_only=False):
    """Return a LOUD refusal envelope, or None when workdir is acceptable.

    The fence is BRANCH-based: agents never work on main/master (nor the origin
    default branch) — in the main checkout OR a linked worktree — unless
    allow_main says so. (A linked worktree parked on main is still
    direct-to-main work.) Branch comparison folds case and NFC/NFD.

    The protected names come from the record omni made the first time it saw
    the repository (``host_fence.protected_branches``), not from refs a
    writable child could have rewritten; a record that does not fit refuses
    (``fence_path_rejected``). A write run also needs files ref storage. A
    read-only run cannot change a ref, so it keeps deriving the names from the
    refs and neither reads nor writes the record.

    It FAILS CLOSED: the only state that passes as "not a git repo" is git
    saying so in as many words. A missing/broken/timing-out git, a repo that
    errors for any other reason, or an undeterminable branch all REFUSE.
    """
    if not os.path.isdir(workdir):
        return _fence_refusal("workdir %r is not a directory" % str(workdir))
    rc, inside, err = _git(workdir, "rev-parse", "--is-inside-work-tree")
    if rc is None:
        return _fence_refusal("git state could not be determined (%s) - fence "
                              "fails closed" % (err or "git did not run"))
    if rc != 0:
        if "not a git repository" in (err + inside).lower():
            return None  # genuinely not a git dir: allowed
        return _fence_refusal(
            "git state could not be determined (%s) - fence fails closed"
            % ((err or inside or "git exit %d" % rc)[:200]))
    if inside != "true":
        # Inside .git metadata (rc 0, "false") or anything unexpected.
        return _fence_refusal(
            "workdir is not inside a git work tree (is-inside-work-tree=%r) - "
            "fence fails closed" % inside)
    if not read_only:
        # Protected-branch handling covers files ref storage only (#164); the
        # host fence refuses the rest too, but read this far first so no
        # record of a repository like that is ever made.
        try:
            ref_format = host_fence_mod._ref_storage_format(
                lambda *a: git_safe.run_git(os.fspath(workdir), a))
        except Exception as e:
            return _fence_refusal("git state could not be determined (%s) - fence "
                                  "fails closed" % redact_secrets(str(e))[:120])
        if ref_format != "files":
            return {"ok": False, "reason": "fence_path_rejected",
                    "detail": host_fence_mod._ref_storage_refusal(
                        ref_format, "write mode")}
        # Before anything reads HEAD or a ref: a symlink where the protect
        # denies expect a file (HEAD itself included) must not be read through.
        rc0, common0, _err0 = _git(workdir, "rev-parse", "--git-common-dir")
        if rc0 == 0 and common0:
            refusal = _alias_refusal(
                workdir, os.path.realpath(os.path.join(os.fspath(workdir), common0)))
            if refusal:
                return refusal

    # Full ref names, never ``--short``: git prints the shortest UNAMBIGUOUS
    # name, so a tag named like a branch turns ``refs/heads/main`` into
    # ``heads/main``, and a fenced child can create that tag.
    rc, head_ref, err = _git_ref(workdir, "symbolic-ref", "-q", "HEAD")
    if rc == 1 and head_ref == "":
        return None  # detached HEAD: no branch to protect
    branch = (head_ref[len("refs/heads/"):]
              if head_ref.startswith("refs/heads/") else "")
    if rc != 0 or branch == "":
        return _fence_refusal(
            "current branch could not be determined (%s) - fence fails closed"
            % ((err or "git exit %s" % rc)[:200]))

    rc3, common, _err3 = _git(workdir, "rev-parse", "--git-common-dir")
    if rc3 != 0 or not common:
        return _fence_refusal("git state could not be determined (common "
                              "directory) - fence fails closed")
    names, refusal = _protected_names(workdir, common, record=not read_only)
    if refusal:
        return refusal
    if not read_only:
        # Whatever allow_main says: a protected name that is a symbolic ref
        # reaches its target under another name.
        refusal = _symref_refusal(
            workdir, names, os.path.realpath(os.path.join(os.fspath(workdir), common)))
        if refusal:
            return refusal
    fenced = {name.lower() for name in names}
    if not kernel_sandbox.branch_name_in(branch, fenced) or allow_main:
        return None  # feature branch, or explicit override
    return _fence_refusal(("refusing to run an agent on branch %r - check out a "
                           "work branch (or pass --allow-main)") % branch)


# Longest partial_reply an envelope carries; a longer one keeps its tail.
PARTIAL_REPLY_MAX_CHARS = 65536


def _grok_result_line(line):
    """run_cli stall marker: a stdout line that parses to a dict whose
    top-level ``type`` is ``result`` (grok's last event of a turn)."""
    try:
        obj = json.loads(line)
    except (ValueError, RecursionError):
        return False
    return isinstance(obj, dict) and obj.get("type") == "result"


def _grok_stall_grace():
    """Seconds of stdout silence, after grok's result line, before the stall
    watchdog kills the run: OMNI_LLM_GROK_STALL_GRACE_S when it is a finite
    float > 0, else GROK_STALL_GRACE_S (an invalid value is noted once)."""
    raw = os.environ.get(ENV_GROK_STALL_GRACE)
    if raw is None or not raw.strip():
        return GROK_STALL_GRACE_S
    try:
        grace = float(raw)
    except ValueError:
        grace = None
    if grace is not None and math.isfinite(grace) and grace > 0:
        return grace
    log_event("grok_stall_grace_invalid", env=ENV_GROK_STALL_GRACE,
              value=raw if len(raw) <= 32 else None, chars=len(raw),
              default=GROK_STALL_GRACE_S)
    return GROK_STALL_GRACE_S


def _stdout_has_events(raw):
    """True when some stdout line is a JSON object with a ``type`` (a real
    event stream, as opposed to the plain text hermetic fakes print). Framed
    on LF only: a JSON string may hold U+2028 or another splitlines() break."""
    for line in (raw or "").split("\n"):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and obj.get("type") is not None:
            return True
    return False


def _last_grok_result_frame(stdout):
    """The last stdout line that is a ``type:"result"`` object, else None."""
    frame = None
    for line in (stdout or "").split("\n"):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            frame = obj
    return frame


def _grok_result_ok(frame):
    """A result frame is a success unless it says is_error, or carries a
    subtype that is not "success" (a present null is not "success" either)."""
    return (isinstance(frame, dict) and frame.get("is_error") is not True
            and ("subtype" not in frame or frame["subtype"] == "success"))


def _partial_reply(job_lane, stdout):
    """The reply a killed grok run had produced so far, as envelope fields.

    Known credential values are masked BEFORE the cap, so a value straddling
    the cut cannot survive as a visible fragment; the envelope egress masks
    again. A longer reply keeps its tail.
    """
    raw = stdout or ""
    text = ""
    if job_lane:
        text = stream_contract.final_text("grok", raw.split("\n")).strip()
        if not text and not _stdout_has_events(raw):
            text = raw.strip()
    else:
        text = raw.strip()
    text = known_secrets.mask_known(text)
    if len(text) > PARTIAL_REPLY_MAX_CHARS:
        return {"partial_reply": text[-PARTIAL_REPLY_MAX_CHARS:],
                "partial_reply_truncated": True}
    return {"partial_reply": text}


def _grok_termination(provider, res, t0):
    """``termination`` record for a grok run omni ended (wall timeout or the
    stall watchdog), else None. Emits the one ``omni-watchdog`` trace line."""
    if provider != "grok":
        return None
    if res.get("stalled"):
        reason = "stall"
    elif res.get("timed_out"):
        reason = "timeout"
    else:
        return None
    elapsed = res.get("elapsed_s")
    if not isinstance(elapsed, (int, float)):
        elapsed = time.monotonic() - t0
    termination = {"reason": reason, "elapsed_s": int(elapsed),
                   "marker": bool(res.get("marker_seen"))}
    log_event("omni-watchdog", provider="grok", reason=reason,
              elapsed=termination["elapsed_s"],
              marker="yes" if termination["marker"] else "no")
    return termination


async def _run_agent_once(task, model, workdir, read_only=False,
                    timeout_s=AGENT_TIMEOUT_S, account_profile="default",
                    allow_main=False, task_file=None, log_path=None,
                    err_log_path=None, job_lane=None, on_spawn=None,
                    session_id=None, resume_session=None, images=None,
                    effort=None, fence_requested=None, expect_writes=False,
                    foreman=False, permission_mode=None, account_home=None,
                    account_deadline=None, codex_auth_isolated=False, account_auth_isolated=False,
                    account_home_explicit=True, pre_spawn=None):
    """Run one delegated agentic job; returns a typed envelope, never raises.

    ``pre_spawn`` (managed API selections) is the late gate check: a callable
    returning a refusal envelope or None, run immediately before the child is
    created.

    ``session_id`` / ``resume_session`` are job-lane pass-throughs (both None
    keeps the synchronous lane byte-identical). Codex captures its own id from
    events; job-lane codex drops ``--ephemeral`` and adds ``--json``.

    ``images`` is a list of already-staged COPY paths (job lane only in this
    batch). Codex gets repeated ``-i <copy>`` before the ``-`` stdin arg;
    claude/grok get a trailing ``[attached images]`` path block on the prompt
    payload. ``images=None`` / empty keeps the sync lane byte-identical.

    ``job_lane`` is preferred over the implicit ``log_path is not None``
    coupling: when None, falls back to ``log_path is not None``. Sync lane
    (unset log_path / job_lane) stays byte-identical.
    """
    foreman = bool(foreman)
    host_required = platform.system() == "Darwin" and not foreman
    if foreman:
        fence_requested = False
    elif host_required:
        fence_requested = True
    depth = env_depth()
    if depth >= 1 and not foreman:
        return _stamp_fence({"ok": False, "reason": "recursion_guard",
                "detail": "agent lane refused at %s=%d (no nested agents)"
                          % (ENV_DEPTH, depth)})
    # Normalize ONCE: this single value feeds the fence, run_cli's cwd, grok's
    # --cwd and the envelope, so the fence can never check a different path
    # than the one the agent runs in (symlinks / `..` / trailing slashes).
    # str() also makes workdir=None safe: "None" is not a directory, so the
    # fence returns a workdir_fence envelope instead of raising.
    workdir = os.path.realpath(str(workdir))
    fence = _workdir_fence(workdir, allow_main, read_only)
    if fence:
        return _stamp_fence(fence)
    model, effort, split_err = take_effort(model, effort)
    if split_err:
        return _stamp_fence(split_err)
    route = resolve_route(model)
    provider = route["provider"]
    routed_model = route["model"]
    if provider not in AGENT_PROVIDERS:
        return _stamp_fence({"ok": False, "reason": "unsupported_agent",
                "detail": "agent lane supports codex/claude/grok/pi/cursor; "
                          "model %r routes to %s" % (str(model), provider)})
    perm_err = permission_mode_refusal(
        provider, permission_mode, foreman=bool(foreman),
        read_only=bool(read_only))
    if perm_err:
        return _stamp_fence(perm_err)
    resolved_effort, effort_err = effort_for_provider(provider, effort)
    if effort_err:
        return _stamp_fence(effort_err)
    overrides, profile_err = (resolve_codex_home(account_profile, account_home)
                              if provider == "codex" and account_home
                              else resolve_account_home(provider, account_profile, account_home)
                              if provider in ("claude", "grok") and account_home
                              else resolve_profile(provider, account_profile))
    if profile_err:
        return _stamp_fence(profile_err)
    if provider in ("claude", "grok") and account_home:
        provenance_err = provider_accounts.home_provenance_error(
            provider, account_profile, account_home, account_home_explicit)
        if provenance_err:
            return _stamp_fence(provenance_err)
    api_kind, api_budget, api_policy_err = provider_accounts.api_child_policy(provider, account_profile)
    if api_policy_err:
        return _stamp_fence(api_policy_err)
    if provider in ACCOUNT_HOME_KEYS and account_home and account_profile in (None, "", "default"):
        # A default job's persisted home is a pin, not a named profile.
        # Keep default environment behavior; apply the home after spawn_env.
        overrides = {}
    if provider == "grok" and read_only and not host_required:
        return _stamp_fence({"ok": False, "reason": "unsupported_read_only",
                "detail": "grok read-only lane not implemented (tools "
                          "disallow-list unverified); use codex, claude, or pi "
                          "for the read-only lane"})
    sandbox_refusal = None if host_required else sandbox_canary.gate(read_only, provider)
    if sandbox_refusal:
        return _stamp_fence(sandbox_refusal)

    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = AGENT_TIMEOUT_S

    images = [os.fspath(p) for p in (images or []) if p]
    if foreman:
        base_env = dict(os.environ)
        base_env.pop(ENV_DEPTH, None)
        env = spawn_env(base_env, overrides, scrub_siblings=codex_auth_isolated or account_auth_isolated, keep_notify=True)
    else:
        env = spawn_env(
            {**os.environ, ENV_DEPTH: str(depth + 1)}, overrides, scrub_siblings=codex_auth_isolated or account_auth_isolated)
    if provider == "codex" and account_home:
        env["CODEX_HOME"] = account_home
    if provider == "codex" and codex_auth_isolated:
        env = codex_accounts.account_env(env)
    if provider in ("claude", "grok"):
        if account_home:
            provider_accounts.apply_home_env(env, provider, account_home, account_home_explicit)
        if account_auth_isolated:
            env = provider_accounts.account_env(env, provider)
    workdir_abs = workdir  # already realpath-normalized above (fence == run)
    t0 = time.monotonic()
    tmp_out = None
    tmp_prompt = None
    tmp_cursor_config = None
    tmp_api_settings = None
    api_settings_path = None
    protected = None      # the pre-spawn read of the protected refs, if any
    child_groups = []     # process groups of the spawned child (verified runs)
    observed_fence, observed_detail = "none", None
    extra_auth = ()
    stdin_data = None
    stdin_file = None
    # Explicit job_lane preferred; fall back to log_path coupling for callers
    # that have not migrated.
    if job_lane is None:
        job_lane = log_path is not None
    else:
        job_lane = bool(job_lane)
    try:
        # Scan the caller-authored task string before stdin / --prompt-file
        # materialization. Attachment trailers are seam4 and are not scanned.
        denied = refuse_payload(task)
        if denied:
            return _stamp_fence(denied)
        if provider == "codex":
            # Realpath like claude / pi: agent_fence_policy admits the
            # resolved leaf, and exec through a home-tree symlink is refused.
            codex = resolve_agent_binary("codex") or vendor_bin.which("codex")
            if not codex:
                return _stamp_fence({"ok": False, "reason": "cli_missing",
                        "detail": "codex not on PATH - install OpenAI Codex CLI, then `codex login`"})
            m = str(routed_model or "").strip() or DEFAULT_CODEX_MODEL
            out_fd, tmp_out = tempfile.mkstemp(suffix=".txt", prefix="omni_agent_out_")
            os.close(out_fd)
            # Resume rewrites the subcommand; session_id is ignored (events).
            if resume_session:
                argv = [codex, "exec", "resume", str(resume_session)]
            else:
                argv = [codex, "exec"]
            # codex-cli 0.145 deprecated the --full-auto alias; the explicit
            # sandbox pair is the supported spelling for the write lane.
            if foreman:
                try:
                    argv += successor_permission_argv(
                        "codex", _foreman_mode_token(permission_mode))
                except UnsupportedPermissionMode as exc:
                    return _stamp_fence({
                        "ok": False,
                        "reason": "unsupported_permission_mode",
                        "detail": redact_secrets(str(exc))[:300],
                    })
            else:
                argv += ["--sandbox", "read-only"] if read_only \
                    else ["--sandbox", "workspace-write"]
            # Job lane: durable session + JSON event stream for thread_id.
            # Sync lane: keep --ephemeral (no session files persisted).
            # BOTH lanes pass --json (issue #51). It is what makes the run
            # machine-readable: without it codex emits only a human transcript,
            # on stderr, that opens with the CALLER'S PROMPT echoed verbatim —
            # the one stream a success/failure verdict must never be read from.
            # The flags are orthogonal (codex-cli 0.147.0 `exec --help`:
            # --json prints events to stdout, -o writes the last message to a
            # file) and the job lane has run this exact combination since #22.
            codex_effort = resolved_effort or DEFAULT_CODEX_REASONING_EFFORT
            argv += ["--skip-git-repo-check", "--json"]
            if not job_lane:
                argv += ["--ephemeral"]
            argv += ["-c", "model_reasoning_effort=%s" % codex_effort,
                     "-m", m, "-o", tmp_out]
            # Staged copies only — never source paths (jobs.start stages first).
            for img in images:
                argv += ["-i", img]
            argv += ["-"]
            if task_file is None:
                stdin_data = str(task)
            else:
                stdin_file = task_file
            extra_auth = CODEX_EXTRA_AUTH_MARKERS
            # The vendor --sandbox pair above is what a DECLINED or off-Darwin
            # spawn rests on. On the write lane the kernel wrap replaces it
            # below and this flag is dropped before the spawn: one Seatbelt
            # per spawn, because a nested sandbox_apply is rc=71 (issue #51).
        elif provider == "claude":
            claude = resolve_agent_binary("claude") or vendor_bin.which("claude")
            if not claude:
                return _stamp_fence({"ok": False, "reason": "cli_missing",
                        "detail": "claude not on PATH - install Claude Code CLI, then log in"})
            m = str(routed_model or "").strip() or DEFAULT_CLAUDE_MODEL
            argv = [claude, "-p"]
            if not foreman:
                # Write lane keeps the project settings; a read-only run loads
                # none, because a workdir settings file can carry hooks that
                # execute (and reach the network) despite the tool allowlist.
                argv += ["--setting-sources", "" if read_only else "project"]
                if api_kind == "api":
                    # User-level settings never load here, so an API-key home
                    # needs its helper delivered by a per-spawn file that holds
                    # only the helper command (never the key).
                    # A private directory outside the child's write roots, for
                    # job and sync runs alike; removed when the run ends.
                    api_cfg = (provider_accounts.fence_home(provider, account_home)
                               if account_home else overrides.get(ACCOUNT_HOME_KEYS[provider]))
                    api_policy = kernel_sandbox.agent_fence_policy(
                        provider, workdir=workdir_abs, read_only=read_only,
                        config_dir=api_cfg or None)
                    if api_policy.get("ok") is False:
                        return _stamp_fence(api_policy)
                    try:
                        api_roots = _composed_write_roots(
                            api_policy["wrap_kw"], workdir_abs, read_only)
                    except kernel_sandbox.FencePathRejected as exc:
                        return _stamp_fence({"ok": False, "reason": "fence_path_rejected",
                                             "detail": exc.detail})
                    if not read_only:
                        # The host fence adds the repository roots later; judge
                        # the base against them now (superset, pre-filter only).
                        api_roots = list(api_roots) + list(
                            host_fence_mod.repo_root_candidates(workdir_abs))
                    settings_base = _api_settings_base(api_roots)
                    if settings_base is None:
                        return _stamp_fence({
                            "ok": False, "reason": "fence_path_rejected",
                            "detail": "no private directory outside the child's "
                                      "write roots for the API settings"})
                    _sweep_stale_api_settings(settings_base)
                    made = _new_run_settings_dir(settings_base)
                    if made is None:
                        return _stamp_fence({
                            "ok": False, "reason": "fence_path_rejected",
                            "detail": "API settings directory could not be created "
                                      "and verified as private"})
                    tmp_api_settings, settings_fd = made
                    try:
                        api_settings_path, settings_err = _write_api_settings_in(
                            account_profile,
                            account_home or overrides.get("CLAUDE_CONFIG_DIR"),
                            settings_fd, tmp_api_settings)
                    finally:
                        os.close(settings_fd)
                    if settings_err:
                        return _stamp_fence(settings_err)
                    argv += ["--settings", api_settings_path]
            argv += ["--model", m]
            if resume_session:
                argv += ["-r", str(resume_session)]
            elif session_id:
                argv += ["--session-id", str(session_id)]
            if foreman:
                try:
                    argv += successor_permission_argv(
                        "claude", _foreman_mode_token(permission_mode))
                except UnsupportedPermissionMode as exc:
                    return _stamp_fence({
                        "ok": False,
                        "reason": "unsupported_permission_mode",
                        "detail": redact_secrets(str(exc))[:300],
                    })
            elif read_only:
                argv += ["--strict-mcp-config",
                         "--tools", READ_ONLY_TOOLS,
                         "--disallowedTools", READ_ONLY_DISALLOWED]
            else:
                argv += ["--dangerously-skip-permissions", "--strict-mcp-config"]
            if resolved_effort:
                argv += ["--effort", resolved_effort]
            if api_budget:
                argv += ["--max-budget-usd", api_budget]
            # Job lane: stream-json event JSONL for the normalizer / final_text.
            if job_lane:
                argv += ["--output-format", "stream-json",
                         "--include-partial-messages", "--verbose"]
            if task_file is None:
                # Inline task: append attach block here. Job-lane task_file
                # path already has the block written into prompt.txt by start().
                stdin_data = str(task)
                if images:
                    stdin_data = stdin_data + _attach_block(images)
            else:
                stdin_file = task_file
        elif provider == "grok":
            grok = vendor_bin.which("grok")
            if not grok:
                return _stamp_fence({"ok": False, "reason": "cli_missing",
                        "detail": "grok not on PATH - install Grok Build CLI, then `grok login`"})
            m = str(routed_model or "").strip() or DEFAULT_GROK_MODEL
            # Grok resume must fork into a pre-assigned child session_id —
            # never fall back to same-id (-r -s without --fork-session errors).
            if resume_session and not session_id:
                return _stamp_fence({"ok": False, "reason": "resume_unsupported",
                        "detail": "grok resume requires a pre-assigned child "
                                  "session_id with --fork-session"})
            # Grok always materializes a --prompt-file from `task` (not
            # task_file). Append the attach block so job-lane copies are
            # visible even though prompt.txt is not the stdin source.
            prompt_text = str(task)
            if images:
                prompt_text = prompt_text + _attach_block(images)
            fd, tmp_prompt = tempfile.mkstemp(suffix=".txt", prefix="omni_agent_task_")
            with os.fdopen(fd, "w") as f:
                f.write(prompt_text)
            # Agent lane: tools + web search stay ENABLED (unlike grok.py's
            # consultation pipe) — the Seatbelt fence at workdir is the control.
            # Job lane: streaming-messages-json (replaces plain) so the runner
            # can normalize + recover final_text; sync keeps --output-format plain.
            if job_lane:
                out_fmt = "streaming-messages-json"
            else:
                out_fmt = "plain"
            argv = [grok, "--prompt-file", tmp_prompt, "--model", m]
            if resolved_effort:
                argv += ["--reasoning-effort", resolved_effort]
            argv += ["--output-format", out_fmt, "--no-memory"]
            if foreman:
                try:
                    argv += successor_permission_argv(
                        "grok", _foreman_mode_token(permission_mode))
                except UnsupportedPermissionMode as exc:
                    return _stamp_fence({
                        "ok": False,
                        "reason": "unsupported_permission_mode",
                        "detail": redact_secrets(str(exc))[:300],
                    })
            else:
                argv += ["--sandbox", "strict"]
            argv += ["--cwd", workdir_abs]
            if job_lane:
                argv += ["--include-partial-messages"]
            if resume_session:
                argv += ["--resume", str(resume_session), "--fork-session",
                         "--session-id", str(session_id)]
            elif session_id:
                argv += ["--session-id", str(session_id)]
            extra_auth = GROK_EXTRA_AUTH_MARKERS
        elif provider == "pi":
            pi_bin = resolve_agent_binary("pi") or vendor_bin.which("pi")
            if not pi_bin:
                return _stamp_fence({"ok": False, "reason": "cli_missing",
                        "detail": "pi not on PATH - install Pi coding-agent CLI, then log in"})
            m = str(routed_model or "").strip() or DEFAULT_PI_MODEL
            argv = [pi_bin, "-p", "--no-approve", "--no-extensions",
                    "--model", m]
            if resume_session:
                argv += ["--session", str(resume_session)]
            elif session_id:
                argv += ["--session-id", str(session_id)]
            if read_only:
                argv += ["--tools", "read,grep,find,ls"]
            # else: tools stay ON (read/bash/edit/write); kernel fence is the wall.
            if resolved_effort:
                argv += ["--thinking", resolved_effort]
            if job_lane:
                argv += ["--mode", "json"]
            if task_file is None:
                stdin_data = str(task)
                if images:
                    stdin_data = stdin_data + _attach_block(images)
            else:
                stdin_file = task_file
            env = pi_child_env(depth)
            if foreman:
                env.pop(ENV_DEPTH, None)
        else:  # cursor
            cursor_bin = resolve_agent_binary("cursor") or cursor_mod.resolve_bin()
            if not cursor_bin:
                return _stamp_fence({"ok": False, "reason": "cli_missing",
                        "detail": "cursor-agent not on PATH - install Cursor "
                                  "CLI (curl https://cursor.com/install), "
                                  "then `cursor-agent login`. Do not use the "
                                  "`agent` name (Grok collision)."})
            slug, cursor_err = cursor_mod.resolve_model(
                routed_model, resolved_effort, account_profile)
            if cursor_err:
                return _stamp_fence(cursor_err)
            m = slug
            argv = [cursor_bin, "-p", "--trust", "--sandbox", "enabled",
                    "--workspace", workdir_abs, "--model", m]
            if read_only:
                argv += ["--mode", "plan"]
            else:
                # Headless -p has no TTY to approve Shell. Default Smart Auto
                # blocks unittest/review commands before spawn. The host fence
                # is the write wall; --force does not admit MCP (--approve-mcps
                # still never passed).
                argv += ["--force"]
            if resume_session:
                argv += ["--resume", str(resume_session)]
            argv += ["--output-format", "stream-json" if job_lane else "json"]
            if task_file is None:
                stdin_data = str(task)
                if images:
                    stdin_data = stdin_data + _attach_block(images)
            else:
                stdin_file = task_file
            # CURSOR_CONFIG_DIR: an account-profile override (already applied
            # to `env` by child_env_for_profile) wins; else a durable
            # <job_dir>/cursor-config for the job lane (so --resume finds
            # the chat), else a fresh per-run temp dir for the sync lane.
            # Launcher-owned paths pass their owning root as launcher_roots
            # so the fence admits them by allowlist; a profile override
            # must pass the dedicated-dotdir rule instead.
            cursor_config_dir = cursor_mod.resolve_config_dir(env)
            launcher_roots = ()
            if cursor_config_dir is None:
                if job_lane and log_path:
                    cursor_config_dir = os.path.join(
                        os.path.dirname(log_path), "cursor-config")
                    os.makedirs(cursor_config_dir, exist_ok=True)
                    from providers import jobs
                    launcher_roots = (os.path.realpath(jobs._jobs_root()),)
                else:
                    tmp_cursor_config = tempfile.mkdtemp(
                        prefix="omni_agent_cursor_cfg_")
                    cursor_config_dir = tmp_cursor_config
                    launcher_roots = (
                        os.path.dirname(os.path.realpath(tmp_cursor_config)),
                    )
            env = dict(env)
            env[CURSOR_CONFIG_DIR_ENV] = cursor_config_dir

        # Every ordinary macOS worker uses the host operation. Preserve the
        # existing provider split for non-macOS and the separate foreman role.
        kernel_fenced = host_required or provider in ("claude", "pi", "cursor") or (
            provider == "codex" and not read_only
        )
        if not kernel_fenced:
            observed_fence = "vendor"
            observed_detail = _vendor_fence_detail(provider, read_only)
        else:
            observed_fence = "none"
            observed_detail = dict(kernel_sandbox.FENCE_NONE_DETAIL)
            fence_kwargs = {}
            if provider == "cursor":
                fence_kwargs["config_dir"] = cursor_config_dir
                if launcher_roots:
                    fence_kwargs["launcher_roots"] = launcher_roots
            elif provider in ("claude", "grok"):
                config_root = (provider_accounts.fence_home(provider, account_home)
                               if account_home else overrides.get(ACCOUNT_HOME_KEYS[provider]))
                if config_root:
                    fence_kwargs["config_dir"] = config_root
            elif provider == "codex":
                config_root = overrides.get("CODEX_HOME")
                if account_home:
                    config_root = codex_accounts.fence_home(account_home)
                if config_root:
                    fence_kwargs["config_dir"] = config_root
            if api_settings_path:
                fence_kwargs["api_settings"] = api_settings_path
                try:
                    fence_kwargs["extra_read_paths"] = (kernel_sandbox.admit_path(
                        api_settings_path, "literal", label="API settings file"),)
                except kernel_sandbox.FencePathRejected as exc:
                    return _stamp_fence({"ok": False, "reason": "fence_path_rejected",
                                         "detail": exc.detail})
            if provider == "codex" and account_home and codex_accounts.fence_home(account_home) is None:
                # host_fence infers config_dir from env. Standard default state
                # must keep default admission, then regain its concrete pin.
                env = dict(env)
                env.pop("CODEX_HOME", None)
            if provider in ("claude", "grok") and account_home and provider_accounts.fence_home(provider, account_home) is None:
                env = dict(env)
                env.pop(ACCOUNT_HOME_KEYS[provider], None)
            argv, env, observed_fence, observed_detail, fence_err = (
                _apply_kernel_fence(
                    provider, argv, env, workdir_abs, read_only,
                    fence_requested, allow_main=allow_main, **fence_kwargs,
                )
            )
            if fence_err:
                return fence_err
            if provider == "codex" and observed_fence != "seatbelt-fs":
                # An allowed legacy fallback still carries the vendor flags.
                # The shared operation owns disabling the inner sandbox only
                # after proving the host wrapper.
                observed_fence = "vendor"
                observed_detail = _vendor_fence_detail(provider, read_only)

        if provider == "codex" and account_home:
            env["CODEX_HOME"] = account_home
        if provider in ("claude", "grok") and account_home:
            provider_accounts.apply_home_env(env, provider, account_home, account_home_explicit)
            if not provider_accounts.home_matches(provider, account_profile, account_home):
                return _stamp_fence({"ok": False, "reason": "account_home_drift", "detail": "original account home changed"})
        if provider == "claude" and read_only:
            # Last step after every env merge and the fence: a plugin selector
            # inherited from the parent would load plugin hooks even though
            # --setting-sources is empty.
            env = scrub_read_only_env(dict(env))
        cli_kwargs = {
            "stdin_data": stdin_data,
            "stdin_file": stdin_file,
            "cwd": workdir_abs,
            "env": env,
            "timeout_s": timeout_s,
            "pane_name": "omni:agent:%s" % provider,
        }
        if log_path is not None:
            cli_kwargs["log_path"] = log_path
        if err_log_path is not None:
            cli_kwargs["err_log_path"] = err_log_path
        if on_spawn is not None:
            cli_kwargs["on_spawn"] = on_spawn
        writes_before = _workspace_snapshot(workdir_abs)
        # Every protected local ref, read from outside the fence right before
        # the spawn (a read-only run cannot change one; --allow-main exempts the
        # checked-out branch only).
        if not read_only:
            protected, refused = _snapshot_protected_refs(workdir_abs, allow_main)
            if refused:
                return _stamp_fence(refused)
        if protected is not None:
            # Remember the child's process group (it leads its own session) so
            # the compare below can first clear out what outlived the leader.
            original_on_spawn = cli_kwargs.get("on_spawn")

            def _note_spawn(proc):
                child_groups.append(getattr(proc, "pid", None))
                return original_on_spawn(proc) if original_on_spawn else None

            cli_kwargs["on_spawn"] = _note_spawn
        if provider == "codex" and account_home:
            if not codex_accounts.home_matches(account_profile, account_home):
                return _stamp_fence({"ok": False, "reason": "account_home_drift", "detail": "original account home changed"})
        if account_deadline is not None:
            cli_kwargs["timeout_s"] = max(0.0, account_deadline - time.monotonic())
            if cli_kwargs["timeout_s"] <= 0:
                return _stamp_fence({"ok": False, "reason": "timeout", "detail": "account dispatch deadline exceeded"})
        if pre_spawn is not None:
            cli_kwargs["pre_spawn"] = pre_spawn
        if provider == "grok" and job_lane:
            # grok can print its final result line and then never exit. The
            # sync lane (plain output, no marker line) gets no watchdog.
            cli_kwargs["stall_marker"] = _grok_result_line
            cli_kwargs["stall_grace_s"] = _grok_stall_grace()
        try:
            res = await run_cli(argv, **cli_kwargs)
        except BaseException:
            # Cancellation (a stop) or any error AFTER the spawn: run_cli has
            # reaped the child by now, and the comparison must still happen.
            # A change wins over the cancellation (the envelope is returned);
            # otherwise the original exception goes on.
            _clear_process_groups(child_groups)
            failure = protected.failure(note="the run was interrupted") if protected else None
            if failure:
                return _stamp_fence(failure, fence=observed_fence, detail=observed_detail)
            raise
        # The one clearing of this run, before anything else looks at the
        # workspace or the refs: a clean exit leaves descendants that closed
        # their pipes running, and the walk and the compare are only a
        # quiescence point once they are gone (best effort, see the helper).
        _clear_process_groups(child_groups)
        if "returncode" not in res:
            # pre_spawn refusal from inside run_cli: the child was never created.
            return _stamp_fence(res, fence=observed_fence, detail=observed_detail)
        workspace_writes, writes_unknown = _workspace_writes(
            workdir_abs, writes_before, _workspace_snapshot(workdir_abs))
        termination = _grok_termination(provider, res, t0)

        def _finish(envelope, fence="none", detail=None):
            # Every envelope that leaves from here on says how omni ended a
            # grok run (no-op for any other run).
            if termination is not None and isinstance(envelope, dict):
                envelope = dict(envelope, termination=termination)
            return _stamp_fence(envelope, fence=fence, detail=detail)

        if protected is not None:
            # Whatever the child did, whatever it exited with: the protected
            # refs are compared first, so the loud reason wins over the run's.
            failure = protected.failure(workspace_writes)
            if failure:
                return _finish(failure, fence=observed_fence,
                                    detail=observed_detail)
        if res.get("timed_out"):
            timeout_env = {"ok": False, "reason": "timeout",
                           "detail": "%s agent exceeded %ds" % (provider, timeout_s),
                           "workspace_writes": workspace_writes}
            if provider == "grok":
                timeout_env.update(_partial_reply(job_lane, res.get("stdout")))
            return _finish(timeout_env, fence=observed_fence, detail=observed_detail)

        # The watchdog killed a grok run that had printed its result line.
        stalled_ok = False
        if provider == "grok" and res.get("stalled"):
            frame = _last_grok_result_frame(res.get("stdout"))
            if _grok_result_ok(frame):
                stalled_ok = True
            else:
                said = frame.get("result") if isinstance(frame, dict) else None
                stall_env = {"ok": False, "reason": "stall",
                             "detail": safe_detail(
                                 "grok reported an error result, then stopped "
                                 "producing output and was ended%s"
                                 % (": %s" % said if isinstance(said, str) and said else "")),
                             "workspace_writes": workspace_writes}
                stall_env.update(_partial_reply(job_lane, res.get("stdout")))
                return _finish(stall_env, fence=observed_fence, detail=observed_detail)

        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        rc = res.get("returncode")
        if provider in ("claude", "grok"):
            process_error = (None if stalled_ok else
                             provider_accounts.process_failure(res, provider))
            if process_error:
                process_error["workspace_writes"] = workspace_writes
                return _finish(process_error, fence=observed_fence, detail=observed_detail)
            # Plain stdout is the assistant's answer, including JSON examples.
            # Job formats and nonzero process status give errors provenance.
            structured = provider_accounts.structured_failure(provider, out) if job_lane or rc != 0 else None
            if structured:
                reported = usage_mod.inband_error(provider, provider_accounts.account_error_text(out))
                detail = reported[1] if reported else "provider account failure"
                if provider == "claude" and overrides.get("CLAUDE_CONFIG_DIR"):
                    detail = safe_detail("CLAUDE_CONFIG_DIR_%s: %s" % (str(account_profile).upper(), detail))
                return _finish({"ok": False, "reason": structured,
                                     "detail": detail,
                                     "account_retry_safe": False, "workspace_writes": workspace_writes},
                                    fence=observed_fence, detail=observed_detail)
        if rc != 0 and not stalled_ok:
            if res.get("signal") is not None:
                reason = "signal_exit"
                detail = (signal_exit_detail(
                    provider, -res["signal"], codex_safe_output(out), "")
                    if provider == "codex" else
                    signal_exit_detail(provider, -res["signal"], out, err))
            else:
                reason = classify_cli_error(
                    codex_error_text(out, with_codes=True) if provider == "codex" else
                    provider_accounts.account_error_text(out) if provider in ("claude", "grok") else out,
                    codex_classification_stderr(out, err) if provider == "codex" else err,
                    rc, extra_auth_markers=extra_auth)
                if reason == "signal_exit":
                    detail = (signal_exit_detail(
                        provider, rc, codex_safe_output(out), "")
                        if provider == "codex" else
                        signal_exit_detail(provider, rc, out, err))
                elif provider == "codex":
                    detail = codex_failure_detail(out, rc)
                else:
                    detail = safe_detail(err or out or "exit %d" % (
                        rc if rc is not None else -1))
                    if (provider == "claude"
                            and isinstance(rc, int)
                            and not isinstance(rc, bool)
                            and rc >= 0):
                        failure = usage_mod.inband_error("claude", provider_accounts.account_error_text(out))
                        if failure:
                            reason, detail = failure
                            classified = classify_cli_error(
                                detail, err, rc, extra_auth_markers=extra_auth)
                            if classified != "runtime_error":
                                reason = classified
                        if overrides.get("CLAUDE_CONFIG_DIR"):
                            var = "CLAUDE_CONFIG_DIR_%s" % str(
                                account_profile).upper()
                            if var not in detail:
                                detail = safe_detail("%s: %s" % (var, detail))
            failure = {"ok": False, "reason": reason, "detail": detail,
                       "workspace_writes": workspace_writes}
            if provider == "codex":
                failure = codex_accounts.annotate_failure(failure, out)
            return _finish(failure, fence=observed_fence, detail=observed_detail)

        if provider == "codex":
            event_reason = classify_cli_error(codex_error_text(out, with_codes=True), "", 1,
                                              extra_auth_markers=extra_auth)
            if event_reason in ("rate_limit", "auth_error"):
                failure = codex_accounts.annotate_failure(
                    {"ok": False, "reason": event_reason, "detail": "Codex account failure",
                     "workspace_writes": workspace_writes}, out)
                return _finish(failure, fence=observed_fence, detail=observed_detail)
            # Outer rc 0 is not proof of work: codex swallows failing shell
            # tool calls (issue #51). Check before any reply is extracted, so
            # a run whose commands were all denied never returns ok:true.
            # Only the event stream is passed in — never `err`, which carries
            # the caller's own prompt echoed back.
            run_failure = _codex_run_failure(out)
            if run_failure:
                run_failure = dict(run_failure)
                run_failure["workspace_writes"] = workspace_writes
                return _finish(run_failure, fence=observed_fence,
                                    detail=observed_detail)

        reply = ""
        if tmp_out and os.path.exists(tmp_out):
            with open(tmp_out) as f:
                reply = f.read().strip()
        if not reply:
            # codex: -o may be empty while --json stdout still has the
            # agent_message event. Never return raw JSONL as text. Both lanes
            # pass --json now (issue #51), so both recover the reply the same
            # way — previously only the job lane could, and the sync lane fell
            # through to the raw-stdout branch below.
            if provider == "codex":
                reply = _last_codex_agent_message(out)
                if not reply and (_stdout_looks_like_codex_json_events(out)
                                  or any(line.lstrip().startswith("{")
                                         for line in out.split("\n"))):
                    # A JSON-shaped stdout that yields no reply is never the
                    # reply, malformed or not; keep codex's own error text.
                    return _finish({"ok": False, "reason": "empty_response",
                            "detail": (codex_safe_output(out) or "empty -o reply at %s"
                                       % (tmp_out or "(missing)"))[:300],
                            "workspace_writes": workspace_writes},
                            fence=observed_fence, detail=observed_detail)
                if not reply:
                    # No JSON events (hermetic plain-stdout fakes): keep
                    # prior stdout fallback.
                    reply = out
            elif job_lane and provider in ("grok", "claude", "pi", "cursor"):
                # Streaming JSONL on stdout — recover assistant final answer.
                raw_out = res.get("stdout") or ""
                # LF framing, as in _stdout_has_events (issue #210).
                reply = stream_contract.final_text(
                    provider, raw_out.split("\n")).strip()
                if not reply:
                    # Hermetic plain-stdout fakes (no event JSONL): keep prior
                    # stdout fallback so resume/jobs fakes stay green. Real
                    # streaming JSONL that yields empty final_text still hits
                    # empty_response below (looks like events → no fallback).
                    if not _stdout_has_events(raw_out):
                        reply = out
            elif provider == "cursor":
                # Sync lane: one JSON object on stdout, `result` is the reply
                # (same shape as the consult lane). Never drop the answer —
                # a non-JSON stdout (hermetic fakes) falls back verbatim.
                # Same in-band check as the consult lane (providers/cursor.py):
                # rc==0 does not mean the CLI succeeded — an `is_error: true`
                # document must never become an ok:true envelope with the
                # error text handed back as the answer.
                inband = usage_mod.inband_error("cursor", out)
                if inband:
                    reason, idetail = inband
                    return _finish({"ok": False, "reason": reason,
                            "detail": idetail, "workspace_writes": workspace_writes},
                            fence=observed_fence, detail=observed_detail)
                try:
                    doc = json.loads(out)
                except (TypeError, ValueError, json.JSONDecodeError):
                    doc = None
                if isinstance(doc, dict) and isinstance(doc.get("result"), str):
                    reply = doc["result"].strip()
                else:
                    reply = out
            else:
                # Sync lane (and non-streaming): fall back to stdout text.
                reply = out
        if not reply:
            return _finish({"ok": False, "reason": "empty_response",
                    "detail": safe_detail((codex_safe_output(out) if provider == "codex" else err)
                                          or "%s agent returned no text" % provider),
                    "workspace_writes": workspace_writes},
                    fence=observed_fence, detail=observed_detail)
        # cursor folds effort into the model slug (no CLI effort flag): the
        # envelope model must be the slug actually run, like the consult
        # lane, so a re-dispatch of this envelope lands on the same seat at
        # the same strength. Other backends echo the caller's own id.
        envelope_model = cursor_mod.envelope_model(m) if provider == "cursor" \
            else str(model)
        envelope = {"ok": True, "text": reply, "model": envelope_model,
                    "workdir": workdir_abs,
                    "duration_s": round(time.monotonic() - t0, 1),
                    "workspace_writes": workspace_writes,
                    "attest": read_successor_attest(
                        provider, session_id, env, workdir=workdir_abs)}
        if expect_writes and workspace_writes is None:
            envelope.update({
                "ok": False, "reason": "workspace_writes_unknown",
                "detail": "expect_writes set but writes cannot be counted: %s"
                          % writes_unknown})
        elif expect_writes and workspace_writes == 0:
            # Issue #65: a write-type task that changed nothing is not a
            # success. The reply is kept so the caller can see the agent's
            # own account of what happened.
            envelope.update({
                "ok": False, "reason": "no_workspace_writes",
                "detail": "%s agent returned ok with zero workspace writes "
                          "in %s" % (provider, workdir_abs)})
        if protected is not None:
            # The reply (and the attest) has been read by now; whatever moved
            # while that happened is caught before the result goes out, and an
            # integrity reason outranks no_workspace_writes and the like.
            late = protected.failure(workspace_writes)
            if late:
                return _finish(late, fence=observed_fence, detail=observed_detail)
        return _finish(envelope, fence=observed_fence, detail=observed_detail)
    except Exception as e:
        failure = protected.failure(note="the run also failed: %s" % exception_detail(e)
                                    ) if protected else None
        if failure:
            return _stamp_fence(failure, fence=observed_fence, detail=observed_detail)
        return _stamp_fence({"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)})
    finally:
        for p in (tmp_out, tmp_prompt):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        cursor_mod.cleanup_tempdir(tmp_cursor_config)
        if tmp_api_settings:
            shutil.rmtree(tmp_api_settings, ignore_errors=True)


async def run_agent(task, model, workdir, read_only=False,
                    timeout_s=AGENT_TIMEOUT_S, account_profile="default",
                    allow_main=False, task_file=None, log_path=None,
                    err_log_path=None, job_lane=None, on_spawn=None,
                    session_id=None, resume_session=None, images=None,
                    effort=None, fence_requested=None, expect_writes=False,
                    foreman=False, permission_mode=None, codex_home=None,
                    codex_candidates=None, codex_homes=None,
                    codex_initial=None, codex_tried=None, on_account_selected=None,
                    codex_deadline=None, codex_auth_isolated=None,
                    account_home=None, account_auth_isolated=None, account_initial=None,
                    account_deadline=None, account_homes=None, account_candidates=None, account_tried=None,
                    account_home_explicit=None):
    kwargs = dict(task=task, model=model, workdir=workdir, read_only=read_only,
                  timeout_s=timeout_s, account_profile=account_profile,
                  allow_main=allow_main, task_file=task_file, log_path=log_path,
                  err_log_path=err_log_path, job_lane=job_lane, on_spawn=on_spawn,
                  session_id=session_id, resume_session=resume_session, images=images,
                  effort=effort, fence_requested=fence_requested, expect_writes=expect_writes,
                  foreman=foreman, permission_mode=permission_mode)
    # Refuse the caller-authored task before account selection and any git query.
    denied = refuse_payload(task)
    if denied:
        return _stamp_fence(denied)
    bare, _eff, split_err = take_effort(model, effort)
    provider = resolve_route(bare)["provider"] if not split_err else None
    if provider in ("claude", "grok"):
        selection = provider_accounts.AccountSelection(
            account_profile, bare, timeout_s, candidates=account_candidates,
            pinned_home=account_home, resume=bool(resume_session), homes=account_homes,
            initial_account=account_initial, tried=account_tried, provider=provider)
        if selection.error:
            return _stamp_fence(selection.exhausted())
        if account_auth_isolated is None:
            account_auth_isolated = selection.managed or provider_accounts.ring_pinned(
                provider, account_profile, resume_session)
        if account_deadline is not None:
            selection.deadline = min(selection.deadline, float(account_deadline))
        async def provider_attempt(account, remaining, managed):
            return await _run_agent_once(**dict(kwargs,
                account_profile=account["account_profile"], timeout_s=remaining,
                account_home=account["account_home"] if managed or account_home else None,
                account_home_explicit=(account.get("account_home_explicit", True)
                    if managed or account_initial is not None else
                    True if account_home_explicit is None else account_home_explicit),
                account_auth_isolated=managed or bool(account_auth_isolated),
                account_deadline=selection.deadline,
                pre_spawn=(lambda: selection.late_api_refusal(account)) if managed else None))
        return await provider_accounts.run_selected(selection, provider_attempt, on_account_selected)
    if split_err or provider != "codex":
        return await _run_agent_once(**kwargs)
    selection = codex_accounts.AccountSelection(
        account_profile, bare, timeout_s, candidates=codex_candidates,
        pinned_home=codex_home, resume=bool(resume_session), homes=codex_homes,
        initial_account=codex_initial, tried=codex_tried)
    if selection.error:
        return _stamp_fence(selection.exhausted())
    # Authentication provenance survives resume even though replay is disabled.
    # Direct sync resume has no job record; a configured ring establishes the
    # subscription isolation unless the caller supplies original provenance.
    if codex_auth_isolated is None:
        codex_auth_isolated = selection.managed or provider_accounts.ring_pinned(
            "codex", account_profile, resume_session)
    kwargs["codex_auth_isolated"] = selection.managed or bool(codex_auth_isolated)
    if codex_deadline is not None:
        selection.deadline = min(selection.deadline, float(codex_deadline))
    if not selection.managed and not codex_home:
        result = await _run_agent_once(**kwargs)
        if not result.get("ok"):
            result.setdefault("account_retry_safe", False)
            selection.failed({"account_profile": account_profile or "default"}, result)
        return result
    async def attempt(account, remaining, managed):
        call = dict(kwargs, account_profile=account["account_profile"],
                    timeout_s=remaining, account_home=account["codex_home"],
                    account_deadline=selection.deadline)
        return await _run_agent_once(**call)
    return await codex_accounts.run_selected(selection, attempt, on_account_selected)
