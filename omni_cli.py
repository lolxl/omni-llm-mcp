#!/usr/bin/env python3
# no-opencc
"""omni — CLI entry point over the same provider core as the MCP server.

Same envelope, same routing, same LOUD-failure contract as the MCP tools
(llm_generate / llm_big_context / llm_image / list_models), callable from any
shell, harness, or scheduler without an MCP client. Zero third-party deps.

DESIGN CONTRACT (inherited from server.py)
- POLICY-FREE PIPE: text in -> text out. Content is never inspected.
- LOUD STRUCTURED FAILURE: stdout carries exactly one JSON envelope
  {ok, ...}; exit code 0 iff ok. Never silent.
- ADVISORY + UNTRUSTED: output is never authoritative; callers re-ground it.

Output modes (gen / big):
  default      envelope JSON on stdout
  --out FILE   reply text is written to FILE; the envelope's `text` is replaced
               by `text_path` + `text_bytes` (keeps stdout small for callers
               that only want the verdict — the fan-out / agent-lane pattern).
               For agent, the file is also written on a failed run that has
               report text (the envelope stays ok: false, exit code 1).
  --out-mode   ref|auto — opt-in artifact spill (NEW flag; never overloads --out).
               ref: any non-empty success text becomes preview + artifact_ref;
               auto: spill only when UTF-8 text > 65536 bytes. Default = inline.
               Rehydrated artifact content is untrusted prior LLM output.
  --plain      print the reply text only; on failure the envelope goes to
               stderr and the exit code is still non-zero

Artifact store (`omni artifact read|rm|gc`): content-addressed spilled bodies
under $OMNI_LLM_STATE_DIR/artifacts/ (TTL 7d; OMNI_LLM_ARTIFACT_TTL_S override).
Integrity is anchored by the sha256 id held by the caller, outside the store.

The `agent` subcommand is the DELEGATION lane (CLI-only; deliberately not an
MCP tool): bare `omni agent` remains synchronous, while `agent start` creates a
detached job under the state directory. `status`, `result`, `tail`, `stop`, and
`jobs` inspect or control those supervisors. See providers/agent_lane.py and
providers/jobs.py for fences, locks, result ordering, and stop semantics.

Zellij watch panes use the same env vars as the MCP server:
OMNI_LLM_WATCH_PANE / OMNI_LLM_ZELLIJ_SESSION / OMNI_LLM_ZELLIJ_TAB.
"""
import argparse
import asyncio
import codecs
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from providers import known_secrets  # noqa: E402
from providers.version import __version__  # noqa: E402
from providers.dispatch import call_tool  # noqa: E402
from providers import artifacts as artifacts_mod  # noqa: E402
from providers import health, sandbox_canary  # noqa: E402
from providers.agent_lane import run_agent  # noqa: E402
from providers.cli_run import drain_watch_finishers  # noqa: E402
from providers import jobs as jobs_mod  # noqa: E402
from providers.errors import (  # noqa: E402
    exception_detail, mask_envelope, mask_text, redact_secrets, safe_detail)
from providers.config import (  # noqa: E402
    DEFAULT_TIMEOUT_S,
    BIG_CONTEXT_TIMEOUT_S,
    DEFAULT_MODELS_TIMEOUT_S,
    DEFAULT_AGENT_MODEL,
    AGENT_TIMEOUT_S,
)


async def _run_with_drain(coro):
    try:
        return await coro
    finally:
        await drain_watch_finishers()


def _call(name, args):
    """One tool call -> envelope. Mirrors server.handle_tool_call's catch-all."""
    try:
        return asyncio.run(_run_with_drain(call_tool(name, args)))
    except Exception as e:  # LOUD, never a traceback on stdout
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}


def _call_agent(**kwargs):
    """One agent-lane run -> envelope; same LOUD catch-all as _call."""
    try:
        return asyncio.run(_run_with_drain(run_agent(**kwargs)))
    except Exception as e:
        return {"ok": False, "reason": "runtime_error",
                "detail": exception_detail(e)}


def _emit(envelope, out=None, plain=False, out_on_failure=False):
    """Apply the output mode and return the process exit code.

    ``out_on_failure`` (agent only): also write --out when the envelope is not
    ok but carries report text; the envelope itself stays a failure.
    """
    if (out and (envelope.get("ok") or out_on_failure)
            and isinstance(envelope.get("text"), str)):
        # Known credential values never reach the file, even the answer's.
        text = known_secrets.mask_known(envelope["text"])
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        envelope = dict(envelope)
        del envelope["text"]
        envelope["text_path"] = os.path.abspath(out)
        envelope["text_bytes"] = len(text.encode("utf-8"))

    envelope = mask_envelope(envelope)
    if plain:
        if envelope.get("ok"):
            text = envelope.get("text", "")
            sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")
            note = envelope.get("effort_note")
            if isinstance(note, str) and note:
                sys.stderr.write(note if note.endswith("\n") else note + "\n")
        else:
            sys.stderr.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    return 0 if envelope.get("ok") else 1


# Agent-lane `-m` help. The lane defaults to an EXECUTION seat, so --help has to
# disclose the model, and that grok has no read-only mode (see --read-only).
AGENT_MODEL_HELP = (
    "model name or comma-separated fallback chain (same routing as the MCP "
    "tools); default %s. --read-only needs an explicit codex/, claude/, pi/ "
    "or cursor/ model — the default refuses it." % DEFAULT_AGENT_MODEL
)


def _add_common(parser, default_timeout, model_help=None):
    parser.add_argument("-m", "--model", default=None,
                        help=model_help or
                        "model name or comma-separated fallback chain "
                        "(same routing as the MCP tools)")
    parser.add_argument("--profile", default="default", dest="profile",
                        help="account profile (multi-quota-pool scheduling)")
    parser.add_argument("--timeout", type=int, default=default_timeout,
                        help="seconds to wait (default %d)" % default_timeout)


def _add_effort(parser):
    parser.add_argument(
        "--effort", default=None,
        help="reasoning effort: grok low|medium|high|xhigh; "
             "claude also max; codex/pi also minimal|max. "
             "Omitted = backend default. Else LOUD unsupported_effort.",
    )


def _add_require_read_wall(parser):
    parser.add_argument(
        "--require-read-wall",
        action="store_true",
        default=False,
        dest="require_read_wall",
        help="only dispatch seats whose backend advertises a consult-lane "
             "read wall (else LOUD read_wall_required). "
             "Default: OMNI_LLM_REQUIRE_READ_WALL=1.",
    )


def _add_output_modes(parser, *, with_out_mode=False):
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--out", default=None,
                      help="write reply text to FILE; envelope keeps only "
                           "text_path/text_bytes")
    mode.add_argument("--plain", action="store_true",
                      help="print reply text only (envelope to stderr on failure)")
    # --out-mode is gen/big only (P1.4 / P2); never on agent (S2).
    if with_out_mode:
        parser.add_argument(
            "--out-mode",
            choices=["ref", "auto"],
            default=None,
            dest="out_mode",
            help="artifact spill: ref (always) or auto (UTF-8 text > 65536). "
                 "Default inline (full text). Maps to MCP result_mode.",
        )


def build_parser():
    p = argparse.ArgumentParser(
        prog="omni",
        description="Multi-model LLM pipe (agy / codex / grok / claude / pi / openai / local) — "
                    "CLI twin of the omni-llm MCP tools. PUBLIC data only.")
    p.add_argument("--version", action="version", version="omni " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("contract", help="print the interfaces this omni offers as one JSON "
                                    "object (version, subcommands, MCP tools, dispatch-exec)")

    sub.add_parser("mcp", help="run the MCP stdio server (JSON-RPC on stdin/stdout; "
                               "register it in an MCP client as command `omni`, args `mcp`)")

    sub.add_parser("session", help="local session table and peer routing")

    sub.add_parser("host-exec", help="exec a task command under the macOS host fence",
                   add_help=False)

    sub.add_parser("dispatch-exec", help="run one grok dispatch: canary, host-exec "
                                         "launch, completion watchdog", add_help=False)

    g = sub.add_parser("gen", help="single-shot generation (llm_generate)")
    g.add_argument("prompt", help="prompt text, or '-' to read it from stdin")
    g.add_argument("-f", "--file", action="append", dest="files", default=None,
                   metavar="PATH_OR_URL",
                   help="repeatable; agy: path/URL into the prompt, codex: local "
                        "file as native vision (-i), grok: local image under "
                        "OMNI_LLM_SOURCE_ROOTS inlined as base64; others LOUD-reject")
    _add_common(g, DEFAULT_TIMEOUT_S)
    _add_effort(g)
    _add_require_read_wall(g)
    _add_output_modes(g, with_out_mode=True)

    b = sub.add_parser("big", help="big-context digestion (llm_big_context)")
    b.add_argument("instruction", help="what to extract or do with the source")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("-t", "--text", default=None, help="inline PUBLIC source text")
    src.add_argument("-s", "--source-path", default=None, dest="source_path",
                     help="server-side read of one UTF-8 file under "
                          "OMNI_LLM_SOURCE_ROOTS")
    src.add_argument("--text-stdin", action="store_true",
                     help="read the source text from stdin")
    _add_common(b, BIG_CONTEXT_TIMEOUT_S)
    _add_effort(b)
    _add_require_read_wall(b)
    _add_output_modes(b, with_out_mode=True)

    i = sub.add_parser("img", help="image generation/edit (llm_image)")
    i.add_argument("prompt", help="image prompt / edit instruction")
    i.add_argument("--backend", choices=["cli", "rest"], default=None,
                   help="cli (default): keyless grok image_edit, needs "
                        "--source-image; rest: xAI REST, needs XAI_API_KEY")
    i.add_argument("--source-image", default=None, dest="source_image",
                   help="cli backend: path to the reference image")
    i.add_argument("-n", type=int, default=None, help="rest backend: image count")
    i.add_argument("--response-format", choices=["url", "b64_json"], default=None,
                   dest="response_format", help="rest backend only")
    i.add_argument("--out-dir", default=None, dest="out_dir",
                   help="directory for produced PNGs")
    _add_common(i, DEFAULT_TIMEOUT_S)

    a = sub.add_parser("agent",
                       help="delegated agentic run inside a worktree (CLI-only "
                            "lane; codex / claude / grok / pi / cursor backends)")
    a.add_argument("task", nargs="?", default=None,
                   help="task text, or '-' to read it from stdin")
    a.add_argument("--task-file", default=None, dest="task_file",
                   help="read the task from FILE (mutually exclusive with TASK)")
    a.add_argument("--workdir", required=True,
                   help="directory the agent works in — use a git worktree on a "
                        "work branch; any git dir on main/master is refused "
                        "unless --allow-main")
    a.add_argument("--read-only", action="store_true", dest="read_only",
                   help="analysis/review mode: codex --sandbox read-only, "
                        "claude write tools disallowed, pi --tools read-only, "
                        "cursor --mode plan; grok LOUD-refuses")
    a.add_argument("--allow-main", action="store_true", dest="allow_main",
                   help="explicitly permit a workdir on main/master")
    a.add_argument("--no-fence", action="store_true", dest="no_fence",
                   help="spawn claude/pi/cursor unfenced "
                        "(OMNI_LLM_AGENT_FENCE=0); on the codex write lane it "
                        "falls back to the vendor sandbox instead of the "
                        "kernel wrap; grok is unchanged")
    a.add_argument("--expect-writes", action="store_true", dest="expect_writes",
                   help="write-type task: zero workspace writes is LOUD "
                        "no_workspace_writes instead of ok:true (issue #65)")
    _add_common(a, AGENT_TIMEOUT_S, model_help=AGENT_MODEL_HELP)
    _add_effort(a)
    # S2: --out/--plain only; --out-mode is gen/big (P2), not agent.
    _add_output_modes(a, with_out_mode=False)

    mo = sub.add_parser("models", help="list agy models + backends map (list_models)")
    mo.add_argument("--timeout", type=int, default=DEFAULT_MODELS_TIMEOUT_S,
                    help="seconds to wait (default %d)" % DEFAULT_MODELS_TIMEOUT_S)
    model_ops = mo.add_subparsers(dest="model_catalog_cmd")

    def _catalog_args(command, *, provider_default="all", provider_required=False,
                      profile_required=False):
        choices = ("cursor", "agy", "codex", "claude", "grok") if provider_required else (
            "all", "cursor", "agy", "codex", "claude", "grok")
        command.add_argument("--provider", choices=choices,
                             default=provider_default, required=provider_required)
        command.add_argument("--profile", default=None, required=profile_required,
                             help="provider account profile (explicit for scoped catalogs)")
        command.add_argument("--json", action="store_true",
                             help="emit the stable JSON result")

    status_cmd = model_ops.add_parser("status", help="show v2 catalog freshness")
    _catalog_args(status_cmd)
    status_cmd.add_argument("--details", action="store_true",
                            help="list model ids and names from the v2 record")

    refresh_cmd = model_ops.add_parser("refresh", help="refresh local metadata through fenced CLI adapters")
    _catalog_args(refresh_cmd)
    refresh_cmd.add_argument("--timeout", type=int, default=120,
                             help="per-provider deadline in seconds (default 120)")

    check_cmd = model_ops.add_parser("check", help="admit the shipped defaults against the v2 record")
    _catalog_args(check_cmd)

    admit_cmd = model_ops.add_parser(
        "admit", help="admit one model and effort against the v2 record")
    admit_cmd.add_argument("--provider", required=True,
                           choices=("cursor", "agy", "codex", "claude", "grok"))
    admit_cmd.add_argument("--model", required=True)
    admit_cmd.add_argument("--effort", default=None)
    admit_cmd.add_argument("--profile", default="default",
                           help="provider account profile (default: default)")
    admit_cmd.add_argument("--json", action="store_true",
                           help="emit the stable JSON result")

    alias_cmd = model_ops.add_parser(
        "alias", help="show or edit the user alias file (catalog dispatch only)")
    alias_ops = alias_cmd.add_subparsers(dest="alias_cmd", required=True)
    alias_ops.add_parser("show", help="print each key, its effective value, and its origin")
    alias_set = alias_ops.add_parser(
        "set", help="set one key; a record refusal warns and still writes")
    alias_set.add_argument("key")
    alias_set.add_argument("value")
    alias_unset = alias_ops.add_parser("unset", help="remove one key from the alias file")
    alias_unset.add_argument("key")

    schedule_cmd = model_ops.add_parser("schedule", help="print a launchd catalog refresh plist")
    schedule_cmd.add_argument("--print", action="store_true", required=True,
                              dest="schedule_print", help="print the plist; do not install it")
    schedule_cmd.add_argument("--profile", required=True,
                              help="provider account profile (one job per profile)")
    schedule_cmd.add_argument("--provider", choices=("all", "cursor", "agy", "codex", "claude", "grok"),
                              default="all", help="provider road selected by this scheduled job")
    schedule_cmd.add_argument("--interval", type=int, default=86400)
    schedule_cmd.add_argument("--timeout", type=int, default=120)

    health_shared = argparse.ArgumentParser(add_help=False)
    health_shared.add_argument(
        "--probe", action="store_true",
        help="query Codex quota RPC, print a quota object, and apply flag/clear")
    health_shared.add_argument(
        "--all-profiles", action="store_true", dest="all_profiles",
        help="with --probe, refresh every known profile (requires --probe)")
    health_shared.add_argument(
        "--json", action="store_true", dest="json",
        help="JSON output (health is always JSON)")
    health_shared.add_argument(
        "--quiet", action="store_true", dest="quiet",
        help="one compact JSON line: ok / probed / cooling / repaired")
    health_shared.add_argument(
        "--profile", default="default", dest="profile",
        help="account profile (default: default)")
    h = sub.add_parser("health", parents=[health_shared],
                       help="show or update seat health cooldowns")
    hs = h.add_subparsers(dest="health_cmd")
    hf = hs.add_parser("flag", parents=[health_shared],
                       help="manually flag a model pool")
    hf.add_argument("model", help="model name")
    hf.add_argument("--until", default=None,
                    help="Unix or ISO-8601 timestamp; defaults to 45 minutes")
    hc = hs.add_parser("clear", parents=[health_shared],
                       help="remove the flag for a model pool and its provider key")
    hc.add_argument("model", help="model name")
    hl = hs.add_parser("lease", help="acquire, renew, or release a seat lease")
    hls = hl.add_subparsers(dest="lease_cmd", required=True)
    hla = hls.add_parser("acquire", help="record a seat lease")
    hla.add_argument("--token", required=True, dest="token")
    hla.add_argument("--pool", required=True, dest="pool")
    hla.add_argument("--order", required=True, dest="order")
    hla.add_argument("--ttl", type=int, required=True, dest="ttl")
    hlr = hls.add_parser("renew", help="extend a seat lease")
    hlr.add_argument("--token", required=True, dest="token")
    hlr.add_argument("--ttl", type=int, default=None, dest="ttl")
    hlx = hls.add_parser("release", help="drop a seat lease")
    hlx.add_argument("--token", required=True, dest="token")

    c = sub.add_parser("canary", help="run live sandbox verification canaries")
    cs = c.add_subparsers(dest="canary_cmd", required=True)
    cw = cs.add_parser("codex-write",
                       help="verify that Codex read-only blocks writes")
    cw.add_argument("--runs", type=int, default=sandbox_canary.DEFAULT_RUNS,
                    help="number of write probes (default %d)"
                         % sandbox_canary.DEFAULT_RUNS)
    cw.add_argument("--timeout", type=int, default=120,
                    help="per-probe timeout in seconds (default 120)")
    cw.add_argument("--remeasure", action="store_true",
                    help="measure a version that is already on the "
                         "known-breached list; prints the counts and writes "
                         "no record, since the gate decides that version on "
                         "its name")

    art = sub.add_parser(
        "artifact",
        help="read / remove / GC spilled result artifacts under "
             "$OMNI_LLM_STATE_DIR/artifacts",
    )
    art_sub = art.add_subparsers(dest="artifact_cmd", required=True)
    art_read = art_sub.add_parser("read", help="read one artifact by id")
    art_read.add_argument("id", help="artifact id")
    art_rm = art_sub.add_parser("rm", help="remove one artifact by id")
    art_rm.add_argument("id", help="artifact id")
    art_sub.add_parser(
        "gc",
        help="delete expired artifacts (only under artifacts/)",
    )

    # --json / --road / --name belong ONLY on list/check/add. Putting them
    # on the profiles parent lets subparser defaults overwrite the parsed
    # values (`omni profiles --json check` silently dropped --json).
    profiles_json = argparse.ArgumentParser(add_help=False)
    profiles_json.add_argument(
        "--json", action="store_true", dest="json",
        help="JSON output (default when stdout is not a tty)",
    )
    profiles_filter = argparse.ArgumentParser(add_help=False)
    profiles_filter.add_argument(
        "--road", default=None, dest="profiles_road",
        help="limit to one road (claude / codex / grok / cursor)",
    )
    profiles_filter.add_argument(
        "--name", default=None, dest="profiles_name",
        help="limit to one profile name",
    )
    pr = sub.add_parser(
        "profiles",
        help="list / check / add account profiles (CLI-only; not an MCP tool)",
    )
    pr_sub = pr.add_subparsers(dest="profiles_cmd")
    pr_sub.add_parser(
        "list", parents=[profiles_json, profiles_filter],
        help="list account profiles",
    )
    chk = pr_sub.add_parser(
        "check", parents=[profiles_json, profiles_filter],
        help="verify profiles; exit 1 if any problem (2 with --strict-quota)",
    )
    chk.add_argument(
        "--live", action="store_true", dest="live",
        help="probe the vendor CLI session (claude/codex/grok/cursor consult)",
    )
    chk.add_argument(
        "--strict-quota", action="store_true", dest="strict_quota",
        help="fail check (exit 2) when a Codex profile is cooling on quota",
    )
    ad = pr_sub.add_parser(
        "add", parents=[profiles_json],
        help="prepare ~/.<road>-<name> and print the vendor login command",
    )
    ad.add_argument("add_road", metavar="road",
                    help="claude / codex / grok / cursor")
    ad.add_argument("add_name", metavar="name",
                    help="profile name matching [a-z0-9_]+")
    ad.add_argument(
        "--exec", action="store_true", dest="exec_login",
        help="run the vendor login command in the foreground (tty required)",
    )
    ad.add_argument(
        "--persist", action="store_true", dest="persist",
        help="append export VAR=dir to ~/.config/omni-llm/profiles.env",
    )
    sch = pr_sub.add_parser(
        "schedule", parents=[profiles_json],
        help="print or install a launchd plist for the advisory profile probe",
    )
    sch_mode = sch.add_mutually_exclusive_group(required=True)
    sch_mode.add_argument(
        "--print", action="store_true", dest="schedule_print",
        help="print the launchd plist to stdout (does not install)",
    )
    sch_mode.add_argument(
        "--install", action="store_true", dest="schedule_install",
        help="write ~/Library/LaunchAgents and launchctl bootstrap",
    )
    sch_mode.add_argument(
        "--uninstall", action="store_true", dest="schedule_uninstall",
        help="launchctl bootout and remove the plist",
    )
    sch.add_argument(
        "--interval", type=int, default=900, dest="schedule_interval",
        help="StartInterval seconds (default 900)",
    )
    sch.add_argument(
        "--force", action="store_true", dest="schedule_force",
        help="with --install, replace an existing plist",
    )

    return p


def _args_gen(ns):
    prompt = sys.stdin.read() if ns.prompt == "-" else ns.prompt
    args = {"prompt": prompt, "account_profile": ns.profile,
            "timeout_s": ns.timeout}
    if ns.model:
        args["model"] = ns.model
    if ns.files:
        args["files"] = ns.files
    if getattr(ns, "out_mode", None):
        args["result_mode"] = ns.out_mode
    if getattr(ns, "effort", None):
        args["effort"] = ns.effort
    if getattr(ns, "require_read_wall", False):
        args["require_read_wall"] = True
    return args


def _args_big(ns):
    args = {"instruction": ns.instruction, "account_profile": ns.profile,
            "timeout_s": ns.timeout}
    if ns.text_stdin:
        args["text"] = sys.stdin.read()
    elif ns.source_path is not None:
        args["source_path"] = ns.source_path
    else:
        args["text"] = ns.text
    if ns.model:
        args["model"] = ns.model
    if getattr(ns, "out_mode", None):
        args["result_mode"] = ns.out_mode
    if getattr(ns, "effort", None):
        args["effort"] = ns.effort
    if getattr(ns, "require_read_wall", False):
        args["require_read_wall"] = True
    return args


def _args_img(ns):
    args = {"prompt": ns.prompt, "account_profile": ns.profile,
            "timeout_s": ns.timeout}
    for key in ("backend", "source_image", "n", "response_format",
                "out_dir", "model"):
        val = getattr(ns, key)
        if val is not None:
            args[key] = val
    return args


def _agent_task(ns, parser):
    """Resolve the agent task text; exactly one of TASK / --task-file."""
    if (ns.task is None) == (ns.task_file is None):
        parser.error("agent needs exactly one of TASK or --task-file")
    if ns.task_file is not None:
        with open(ns.task_file, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read() if ns.task == "-" else ns.task


_AGENT_JOB_COMMANDS = {
    "start", "status", "result", "tail", "stop", "jobs", "resume",
}


# `agent tail` is a log view, not the answer channel, but a key in it is as
# sensitive as one in an envelope. Only whole lines are masked and printed; the
# unfinished tail of the file is held back until its newline arrives, and what
# is held is bounded.
_TAIL_PENDING_MAX = 64 * 1024
_TAIL_PENDING_KEEP = 8 * 1024
_TAIL_READ_BLOCK = 1 << 20
_BLANKS = " \t\r\f\v"


def _mask_unfinished_line(text):
    """Mask a line that has no newline yet, and withhold its last token.

    Whatever ends the text may be the front half of a key too short for any
    pattern, so a final token of 8+ characters is replaced by a marker.
    """
    masked = mask_text(text)
    start = max(masked.rfind(c) for c in _BLANKS) + 1
    if len(masked) - start >= 8:
        masked = masked[:start] + "<cut>"
    return masked


def _mask_log_text(text):
    """Mask log text: complete lines whole, an unfinished last line guarded."""
    head, newline, rest = text.rpartition("\n")
    out = mask_text(head + newline) if newline else ""
    return out + (_mask_unfinished_line(rest) if rest else "")


def _tail_feed(text, pending):
    """Print the complete lines of ``pending + text``; return the new remainder."""
    cut = text.rfind("\n")
    if cut >= 0:
        sys.stdout.write(mask_text(pending + text[:cut + 1]))
        pending = text[cut + 1:]
    else:
        pending += text
    if len(pending) > _TAIL_PENDING_MAX:
        body, keep = pending[:-_TAIL_PENDING_KEEP], pending[-_TAIL_PENDING_KEEP:]
        blank = max(body.rfind(c) for c in _BLANKS)
        if blank >= 0:
            body, keep = body[:blank + 1], body[blank + 1:] + keep
        sys.stdout.write(mask_text(body))
        pending = keep
    sys.stdout.flush()
    return pending


def _tail_follow(job, raw=False, err=False, n=10, poll_s=0.2):
    """Print the last n lines, then poll-append until Ctrl-C (exit 0)."""
    if err:
        name = "stream.err.log"
    elif raw:
        name = "stream.raw.log"
    else:
        name = "stream.log"
    try:
        job_dir = jobs_mod._resolve_job_dir(job)
    except (OSError, ValueError, FileNotFoundError) as exc:
        envelope = {"ok": False, "reason": "runtime_error",
                    "detail": redact_secrets(str(exc))[:300]}
        sys.stdout.write(json.dumps(mask_envelope(envelope), ensure_ascii=False) + "\n")
        return 1
    def refused():
        # --raw is only for a running job: checked here and on every poll.
        refusal = jobs_mod.raw_log_refusal(job_dir) if raw and not err else None
        if refusal:
            sys.stdout.write(json.dumps(mask_envelope(refusal), ensure_ascii=False) + "\n")
            sys.stdout.flush()
        return bool(refusal)

    if refused():
        return 1
    path = os.path.join(job_dir, name)
    # Seed with the same last-n view as one-shot tail.
    try:
        count = max(0, int(n))
    except (TypeError, ValueError):
        count = 10
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    offset = 0
    try:
        if os.path.isfile(path):
            # One read decides both what is shown and where following starts,
            # so nothing is skipped or shown twice; the unfinished last line
            # is held back, not printed.
            with open(path, "rb") as handle:
                data = handle.read()
            offset = len(data)
            head, newline, pending = decoder.decode(data).rpartition("\n")
            lines = (head + newline).splitlines(True)
            shown = lines[-count:] if count else lines
            sys.stdout.write(mask_text("".join(shown)))
            sys.stdout.flush()
            # Only `pending` and `offset` survive into the polling loop.
            del data, head, newline, lines, shown
    except OSError:
        offset = 0
        pending = ""
    try:
        while True:
            if raw and refused():
                return 1
            try:
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    if size < offset:
                        offset = 0
                        pending = ""
                        decoder.reset()
                    if size > offset:
                        with open(path, "rb") as handle:
                            handle.seek(offset)
                            while True:
                                chunk = handle.read(_TAIL_READ_BLOCK)
                                if not chunk:
                                    break
                                offset += len(chunk)
                                pending = _tail_feed(decoder.decode(chunk), pending)
            except OSError:
                pass
            time.sleep(poll_s)
    except KeyboardInterrupt:
        rest = pending + decoder.decode(b"", final=True)
        if rest:
            sys.stdout.write(_mask_unfinished_line(rest))
            sys.stdout.flush()
        return 0


def _build_agent_jobs_parser():
    parser = argparse.ArgumentParser(
        prog="omni agent",
        description="Start and inspect detached agent jobs.",
    )
    sub = parser.add_subparsers(dest="agent_job_cmd", required=True)

    start = sub.add_parser("start", help="detach an agent run and return its job id")
    start.add_argument("task", nargs="?", default=None,
                       help="task text, or '-' to read it from stdin")
    start.add_argument("--task-file", default=None, dest="task_file",
                       help="read the task from FILE (mutually exclusive with TASK)")
    start.add_argument("--workdir", required=True,
                       help="directory the agent works in")
    start.add_argument("--read-only", action="store_true", dest="read_only",
                       help="analysis/review mode")
    start.add_argument("--allow-main", action="store_true", dest="allow_main",
                       help="explicitly permit a workdir on main/master")
    start.add_argument("--no-fence", action="store_true", dest="no_fence",
                       help="spawn claude/pi/cursor unfenced (persisted as "
                            "fence_requested: false); on the codex write lane "
                            "it falls back to the vendor sandbox instead of "
                            "the kernel wrap; grok is unchanged")
    start.add_argument("--expect-writes", action="store_true",
                       dest="expect_writes",
                       help="write-type task: zero workspace writes is LOUD "
                            "no_workspace_writes (persisted in job.json)")
    start.add_argument("--image", action="append", dest="images", default=None,
                       metavar="PATH",
                       help="stage a local image into the job attach dir "
                            "(repeatable; job-lane only — not on bare "
                            "`omni agent`; paths must be under "
                            "OMNI_LLM_SOURCE_ROOTS)")
    _add_common(start, AGENT_TIMEOUT_S, model_help=AGENT_MODEL_HELP)
    _add_effort(start)
    start.add_argument(
        "--permission-mode", default=None, dest="permission_mode",
        metavar="MODE",
        help="successor permission mode (claude choice list: acceptEdits, "
             "auto, bypassPermissions, manual, dontAsk, plan; omitted or "
             "default is the omitted-flag spelling — claude emits no "
             "--permission-mode, codex/grok keep their default mapping). "
             "Requires --foreman; else LOUD "
             "permission_mode_requires_foreman. Unknown token → LOUD "
             "unsupported_permission_mode.",
    )
    start.add_argument(
        "--foreman", action="store_true", dest="foreman", default=False,
        help="unfenced successor session (handoff root). claude/codex/grok "
             "only; other providers LOUD foreman_unsupported_provider. "
             "Implies no kernel fence (claude fence=none, codex/grok "
             "fence=vendor, same as --no-fence). Drops claude "
             "--setting-sources project, --dangerously-skip-permissions, "
             "and --strict-mcp-config. Child env does not carry "
             "OMNI_LLM_DEPTH and does not silence notify. Default stays "
             "fenced. With --read-only: LOUD foreman_requires_write_lane.",
    )

    for command, help_text in (
        ("status", "show one job's current state"),
        ("result", "show one job's terminal result"),
        ("tail", "show the end of one job's stream log"),
        ("stop", "stop one running job"),
    ):
        sub.add_parser(command, help=help_text).add_argument("job")
    result = sub.choices["result"]
    result.add_argument("--wait", type=float, default=0,
                        help="wait up to S seconds for a result")
    tail = sub.choices["tail"]
    tail.add_argument("-n", type=int, default=10,
                      help="number of lines to print (default 10)")
    tail_src = tail.add_mutually_exclusive_group()
    tail_src.add_argument("--raw", action="store_true",
                          help="read stream.raw.log (backend JSONL) instead of "
                               "normalized stream.log; only while the job "
                               "runs, the file is deleted when it ends")
    tail_src.add_argument("--err", action="store_true",
                          help="read stream.err.log (backend stderr)")
    tail.add_argument("-f", "--follow", action="store_true",
                      help="follow appends until Ctrl-C (exits 0)")
    stop = sub.choices["stop"]
    stop.add_argument("--grace", type=float, default=10,
                      help="seconds between SIGTERM and SIGKILL")
    stop.add_argument("--reason", default=None,
                      help="optional reason recorded on the stopped job")
    sub.add_parser(
        "resume",
        help="resume a terminal job's backend session with a new prompt")
    resume = sub.choices["resume"]
    resume.add_argument("job", help="source job id or job directory")
    resume.add_argument("prompt", help="continuation prompt text")
    resume.add_argument("--allow-drift", action="store_true", dest="allow_drift",
                        help="skip workdir git drift pin check")
    sub.add_parser("jobs", help="list all existing jobs")
    return parser


def _main_agent_job(argv):
    parser = _build_agent_jobs_parser()
    ns = parser.parse_args(argv)
    try:
        if ns.agent_job_cmd == "start":
            task = _agent_task(ns, parser)
            start_kwargs = {}
            if getattr(ns, "effort", None):
                start_kwargs["effort"] = ns.effort
            if getattr(ns, "no_fence", False):
                start_kwargs["fence_requested"] = False
            if getattr(ns, "expect_writes", False):
                start_kwargs["expect_writes"] = True
            if getattr(ns, "permission_mode", None):
                start_kwargs["permission_mode"] = ns.permission_mode
            if getattr(ns, "foreman", False):
                start_kwargs["foreman"] = True
            value = jobs_mod.start(
                task=task,
                model=ns.model or DEFAULT_AGENT_MODEL,
                workdir=ns.workdir,
                read_only=ns.read_only,
                timeout_s=ns.timeout,
                account_profile=ns.profile,
                allow_main=ns.allow_main,
                images=getattr(ns, "images", None),
                **start_kwargs,
            )
            sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
            return 1 if "refused" in value else 0
        if ns.agent_job_cmd == "status":
            value = jobs_mod.status(ns.job)
            sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
            return 0
        if ns.agent_job_cmd == "result":
            value = jobs_mod.result(ns.job, wait_s=ns.wait)
            sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
            return 0 if value.get("reason") != "not_finished" else 1
        if ns.agent_job_cmd == "tail":
            raw = bool(getattr(ns, "raw", False))
            err = bool(getattr(ns, "err", False))
            if getattr(ns, "follow", False):
                return _tail_follow(ns.job, raw=raw, err=err, n=ns.n)
            try:
                text = jobs_mod.tail(ns.job, lines=ns.n, raw=raw, err=err)
            except jobs_mod.RawLogRefused as refused:
                sys.stdout.write(json.dumps(
                    mask_envelope(refused.envelope), ensure_ascii=False) + "\n")
                return 1
            sys.stdout.write(_mask_log_text(text))
            return 0
        if ns.agent_job_cmd == "stop":
            value = jobs_mod.stop(ns.job, grace_s=ns.grace, reason=ns.reason)
            sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
            return 0
        if ns.agent_job_cmd == "resume":
            value = jobs_mod.resume(
                ns.job, ns.prompt, allow_drift=ns.allow_drift)
            sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
            return 1 if "refused" in value else 0
        value = jobs_mod.list_jobs()
        sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        envelope = {"ok": False, "reason": "runtime_error", "detail": redact_secrets(str(exc))[:300]}
        sys.stdout.write(json.dumps(mask_envelope(envelope), ensure_ascii=False) + "\n")
        return 1


def _is_agent_job_invocation(argv):
    return (len(argv) >= 2 and argv[0] == "agent" and
            argv[1] in _AGENT_JOB_COMMANDS)


_PROFILE_TABLE_COLS = (
    "ROAD", "NAME", "SOURCE", "IDENTITY", "STATE", "COOLDOWN", "PROBLEMS",
)


def _profile_identity_cell(row):
    from providers.profile_check import identity_label
    ident = row.get("identity") or {}
    road = row.get("road")
    label = identity_label(road, ident)
    if label is not None:
        return label
    if road == "cursor":
        return ident.get("kind") or "-"
    return "-"


def _format_profiles_table(rows):
    cells = []
    for row in rows:
        problems = row.get("problems") or []
        cells.append([
            str(row.get("road") or ""),
            str(row.get("name") or ""),
            str(row.get("source") or ""),
            str(_profile_identity_cell(row)),
            str(row.get("state") or "-"),
            str(row.get("cooldown") or "-"),
            ",".join(str(p) for p in problems) if problems else "-",
        ])
    widths = [len(c) for c in _PROFILE_TABLE_COLS]
    for line in cells:
        for i, val in enumerate(line):
            widths[i] = max(widths[i], len(val))

    def fmt(parts):
        return "  ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    out = [fmt(_PROFILE_TABLE_COLS)]
    for line in cells:
        out.append(fmt(line))
    return "\n".join(out) + "\n"


def _want_profiles_json(ns):
    if getattr(ns, "json", False):
        return True
    return not sys.stdout.isatty()


def _emit_profiles(payload, as_json):
    if as_json:
        return _emit(payload)
    sys.stdout.write(known_secrets.mask_known(
        _format_profiles_table(payload.get("profiles") or [])))
    return 0 if payload.get("ok") else 1


def _filter_profile_rows(rows, road=None, name=None):
    out = list(rows)
    if road:
        road = str(road).strip().lower()
        out = [r for r in out if r.get("road") == road]
    if name:
        out = [r for r in out if r.get("name") == name]
    return out


def _remove_empty_login_dir(path):
    """Remove *path* if it is an empty directory. Return 'removed', 'left', or None."""
    if not path:
        return None
    if not os.path.isdir(path):
        return None
    try:
        if os.listdir(path):
            return "left"
        os.rmdir(path)
        return "removed"
    except OSError:
        return "left"


def _probe_omni_argv():
    return [
        os.path.realpath(sys.executable),
        os.path.realpath(__file__),
        "health", "--probe", "--all-profiles", "--json", "--quiet",
    ]


def _linked_worktree_checkout(cli_path=None):
    """True when *cli_path*'s directory holds a ``.git`` file (linked worktree)."""
    path = os.path.realpath(cli_path or __file__)
    return os.path.isfile(os.path.join(os.path.dirname(path), ".git"))


_PROBE_LOG_ROTATE_BYTES = 1024 * 1024


def _rotate_probe_log(log_path):
    """If *log_path* is larger than 1 MiB, rename it to ``.1``."""
    try:
        if os.path.getsize(log_path) > _PROBE_LOG_ROTATE_BYTES:
            os.replace(log_path, log_path + ".1")
    except OSError:
        return


def _launchctl(argv, loud=True):
    import subprocess
    try:
        completed = subprocess.run(argv, capture_output=True, text=True)
    except OSError as exc:
        return {"ok": False, "reason": "runtime_error",
                "detail": redact_secrets(str(exc))[:300], "rc": None}
    rc = completed.returncode
    if rc == 0:
        return {"ok": True, "rc": 0}
    if not loud:
        return {"ok": False, "rc": rc}
    detail = (completed.stderr or completed.stdout or "").strip()
    text = "launchctl %s rc=%s" % (argv[1] if len(argv) > 1 else "", rc)
    if detail:
        text = "%s %s" % (text, detail)
    return {"ok": False, "reason": "runtime_error",
            "detail": safe_detail(text), "rc": rc}


def _write_text_file(path, text, mode):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ensure_dir(path, mode):
    if not os.path.isdir(path):
        os.makedirs(path)
    os.chmod(path, mode)


def _gui_target():
    return "gui/%d" % os.getuid()


def _main_profiles_schedule(ns):
    """profiles schedule --print/--install/--uninstall. Never an MCP tool."""
    from providers.profile_check import (
        SCHEDULE_LABEL, probe_log_path, schedule_plist, schedule_plist_path,
    )

    interval = getattr(ns, "schedule_interval", 900)
    if interval is None or int(interval) < 1:
        return _emit({
            "ok": False,
            "reason": "profile_config",
            "detail": "--interval must be a positive number of seconds",
        })
    interval = int(interval)
    omni_argv = _probe_omni_argv()
    log_path = probe_log_path()
    text = schedule_plist(interval, omni_argv, log_path=log_path)
    plist_path = schedule_plist_path()
    gui = _gui_target()

    if getattr(ns, "schedule_print", False):
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return 0

    if getattr(ns, "schedule_uninstall", False):
        boot = _launchctl(
            ["launchctl", "bootout", gui, plist_path], loud=False)
        removed = False
        if os.path.isfile(plist_path):
            try:
                os.remove(plist_path)
                removed = True
            except OSError as exc:
                return _emit({
                    "ok": False,
                    "reason": "runtime_error",
                    "detail": redact_secrets(str(exc))[:300],
                    "path": plist_path,
                    "rc": boot.get("rc"),
                })
        if not removed and not boot.get("ok"):
            return _emit({
                "ok": False,
                "reason": "runtime_error",
                "detail": "launchctl bootout rc=%s" % boot.get("rc"),
                "path": plist_path,
                "rc": boot.get("rc"),
            })
        return _emit({
            "ok": True,
            "path": plist_path,
            "label": SCHEDULE_LABEL,
            "removed": removed,
        })

    exists = os.path.isfile(plist_path)
    if exists and not getattr(ns, "schedule_force", False):
        return _emit({
            "ok": False,
            "reason": "already_exists",
            "detail": "plist exists; pass --force to replace",
            "path": plist_path,
        })
    if (_linked_worktree_checkout(__file__)
            and not getattr(ns, "schedule_force", False)):
        return _emit({
            "ok": False,
            "reason": "worktree_checkout",
            "detail": "install from the main checkout; "
                      "re-run --install --force when the checkout moves",
            "path": os.path.realpath(__file__),
        })
    if exists:
        _launchctl(["launchctl", "bootout", gui, plist_path], loud=False)
    _ensure_dir(os.path.dirname(log_path), 0o700)
    _rotate_probe_log(log_path)
    _write_text_file(plist_path, text, 0o644)
    loaded = _launchctl(["launchctl", "bootstrap", gui, plist_path], loud=True)
    if not loaded.get("ok"):
        loaded["path"] = plist_path
        loaded["label"] = SCHEDULE_LABEL
        return _emit(loaded)
    return _emit({
        "ok": True,
        "path": plist_path,
        "label": SCHEDULE_LABEL,
        "loaded": True,
        "interval": interval,
    })


def _main_profiles(ns):
    """profiles / profiles check / profiles add — CLI-only, never MCP."""
    from providers.profile_check import add_profile, check, list_profiles
    from providers.profiles import RING_ROADS

    cmd = getattr(ns, "profiles_cmd", None)
    if cmd == "schedule":
        return _main_profiles_schedule(ns)
    as_json = _want_profiles_json(ns)
    road = getattr(ns, "profiles_road", None)
    if road:
        road = str(road).strip().lower()
        if road not in RING_ROADS:
            return _emit({"ok": False, "reason": "profile_config",
                          "detail": "unknown road (ring is claude/codex/grok/cursor "
                                    "only)"})

    if cmd == "add":
        if getattr(ns, "exec_login", False) and not sys.stdin.isatty():
            return _emit({
                "ok": False,
                "reason": "not_a_tty",
                "detail": "--exec requires a tty; refusing to run login",
            })
        envelope = add_profile(
            ns.add_road, ns.add_name, persist=bool(getattr(ns, "persist", False)))
        if not envelope.get("ok"):
            return _emit(envelope)
        if getattr(ns, "exec_login", False):
            argv = envelope.get("login_argv")
            if not argv:
                return _emit({
                    "ok": False,
                    "reason": "unsupported_profile",
                    "detail": envelope.get("detail") or (
                        "no vendor login command for this road"),
                    "road": envelope.get("road"),
                    "name": envelope.get("name"),
                })
            import subprocess
            child_env = dict(os.environ)
            child_env.update(envelope.get("login_env") or {})
            try:
                login_rc = subprocess.call(argv, env=child_env)
            except OSError as exc:
                return _emit({"ok": False, "reason": "runtime_error",
                              "detail": redact_secrets(str(exc))[:300]})
            var = envelope.get("var")
            path = envelope.get("path")
            if login_rc != 0:
                fate = _remove_empty_login_dir(path)
                detail = "login exited %s" % login_rc
                if path:
                    detail += "; directory %s" % path
                    if fate == "removed":
                        detail += " was empty and removed"
                    elif fate == "left":
                        detail += " was left in place"
                return _emit({
                    "ok": False,
                    "reason": "login_failed",
                    "detail": detail,
                    "road": envelope.get("road"),
                    "name": envelope.get("name"),
                    "path": path,
                    "rc": login_rc,
                })
            if var and path:
                os.environ[var] = path
            rows = _filter_profile_rows(
                list_profiles(roads=(envelope["road"],)),
                name=envelope["name"])
            if not rows:
                rows = [{
                    "road": envelope["road"],
                    "name": envelope["name"],
                    "source": "env",
                    "path": path,
                    "identity": {},
                    "state": None,
                    "problems": [],
                    "cooldown": None,
                }]
            checked = check(rows, live=True)
            envelope["check"] = checked
            envelope["ok"] = bool(checked.get("ok"))
        if as_json:
            return _emit(envelope)
        login = envelope.get("login_command")
        if login:
            sys.stdout.write(login + "\n")
        detail = envelope.get("detail")
        if detail:
            sys.stdout.write(redact_secrets(detail) + "\n")
        mcp_env = envelope.get("mcp_env")
        if mcp_env and getattr(ns, "persist", False):
            sys.stdout.write(json.dumps(mcp_env, ensure_ascii=False) + "\n")
        return 0 if envelope.get("ok") else 1

    roads = (road,) if road else None
    rows = list_profiles(roads=roads)
    name = getattr(ns, "profiles_name", None)
    if name:
        rows = _filter_profile_rows(rows, name=name)
    if cmd == "check":
        strict = bool(getattr(ns, "strict_quota", False))
        payload = check(
            rows, live=bool(getattr(ns, "live", False)),
            strict_quota=strict)
        rc = _emit_profiles(payload, as_json)
        if rc and strict and any(
                "quota" in (row.get("problems") or [])
                for row in payload.get("profiles") or []):
            return 2
        return rc
    payload = {"ok": True, "profiles": rows}
    return _emit_profiles(payload, as_json)


def _main_canary(ns):
    value = sandbox_canary.run_write_canary(
        runs=ns.runs, timeout_s=ns.timeout, remeasure=ns.remeasure,
    )
    sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False, indent=2) + "\n")
    # Success means a record the gate will read. A --remeasure of a listed
    # version returns evidence and writes nothing, so it is never exit 0
    # however clean the sample looked (issue #99).
    return 0 if value.get("verdict") == "verified" else 1


def _main_artifact(ns):
    """artifact read|rm|gc — LOUD JSON, exit 1 on ok:false."""
    try:
        if ns.artifact_cmd == "read":
            value = artifacts_mod.read(ns.id)
        elif ns.artifact_cmd == "rm":
            value = artifacts_mod.remove(ns.id)
        elif ns.artifact_cmd == "gc":
            value = artifacts_mod.gc(force=True)
        else:
            value = {"ok": False, "reason": "runtime_error",
                     "detail": "unknown artifact command"}
    except Exception as e:
        value = {"ok": False, "reason": "runtime_error", "detail": redact_secrets(str(e))[:300]}
    sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
    return 0 if value.get("ok") else 1


def _main_alias(ns):
    """``omni models alias show|set|unset``. Writes refuse at OMNI_LLM_DEPTH >= 1."""
    from providers import aliases
    from providers.config import env_depth
    command = ns.alias_cmd
    if command in ("set", "unset") and env_depth() >= 1:
        return _emit({
            "ok": False,
            "reason": "recursion_guard",
            "detail": "omni models alias %s refused at OMNI_LLM_DEPTH=%d" % (
                command, env_depth()),
        })
    try:
        if command == "show":
            return _emit({"ok": True, "aliases": aliases.show()})
        if command == "set":
            warning = aliases.admission_warning(ns.key, ns.value)
            aliases.set_alias(ns.key, ns.value)
            payload = {"ok": True, "key": ns.key, "value": ns.value}
            if warning:
                payload["warning"] = warning
                sys.stderr.write(warning if warning.endswith("\n") else warning + "\n")
            return _emit(payload)
        if command == "unset":
            aliases.unset_alias(ns.key)
            return _emit({"ok": True, "key": ns.key, "unset": True})
        return _emit({"ok": False, "reason": "unknown_command",
                      "detail": "alias unknown_command"})
    except aliases.AliasError as exc:
        return _emit({"ok": False, "reason": "alias_file", "detail": exc.detail})


def _format_models_admit(value):
    """One line for ``omni models admit`` without ``--json``."""
    def show(item):
        if item is None:
            return "-"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, list):
            return ",".join(str(part) for part in item) or "-"
        return str(item)

    return (
        "ok=%s admitted=%s provider=%s profile=%s model=%s effort=%s "
        "effort_source=%s refusal=%s warnings=%s\n"
        % (
            show(value.get("ok")),
            show(value.get("admitted")),
            show(value.get("provider")),
            show(value.get("profile")),
            show(value.get("model")),
            show(value.get("effort")),
            show(value.get("effort_source")),
            show(value.get("refusal")),
            show(value.get("warnings")),
        )
    )


def _main_model_catalog(ns):
    from providers import model_catalog

    command = ns.model_catalog_cmd
    try:
        if command == "status":
            value = model_catalog.status(
                ns.provider, ns.profile, details=bool(getattr(ns, "details", False)))
        elif command == "refresh":
            value = model_catalog.refresh(ns.provider, ns.profile, ns.timeout)
        elif command == "check":
            value = model_catalog.check(ns.provider, ns.profile)
        elif command == "admit":
            value = model_catalog.admit(
                ns.provider, ns.model, effort=getattr(ns, "effort", None),
                profile=getattr(ns, "profile", None))
        elif command == "schedule":
            value = model_catalog.schedule_plist(
                ns.profile, ns.interval, ns.timeout, provider=ns.provider)
        else:
            value = {"ok": False, "reason": "unknown_command",
                     "detail": "model catalog unknown_command"}
    except model_catalog.CatalogError as exc:
        # A profile name that fails the name rule is argv usage, ordered exit 2.
        # Other catalog failures stay a runtime error, as before.
        if command == "admit" and exc.reason == "unsupported_profile":
            value = {"ok": False, "reason": "usage_error",
                     "detail": "model catalog unsupported_profile",
                     "exit_code": 2}
        else:
            value = {"ok": False, "reason": "runtime_error",
                     "detail": "model catalog runtime_error"}
    except Exception as exc:
        value = {"ok": False, "reason": "runtime_error",
                 "detail": "model catalog runtime_error"}

    if command == "schedule" and value.get("ok"):
        sys.stdout.write(value["plist"])
        return 0
    if command == "admit" and isinstance(value, dict) and "admitted" in value:
        payload = mask_envelope(value)
        if getattr(ns, "json", False):
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(_format_models_admit(payload))
        return 0 if payload.get("admitted") else 1
    sys.stdout.write(json.dumps(mask_envelope(value), ensure_ascii=False) + "\n")
    if command in ("check", "admit") and isinstance(value.get("exit_code"), int):
        return value["exit_code"]
    if command == "refresh" and not value.get("ok"):
        return 2
    return 0 if value.get("ok") else 1


def _cmd_health_clear(ns):
    """omni health clear <model> [--profile P] — drop the pool and road flags.

    Exit 0 even when nothing was flagged. This is how a credit_exhausted hold
    ends after the user adds credits. A removal that could not be made
    (ledger lock unavailable) is ok:false with exit 1, never a quiet success.
    """
    from providers.seat_name import take_effort
    clear_model, _effort, split_err = take_effort(ns.model, None)
    if split_err:
        return _emit(split_err)
    cleared = []
    try:
        for model in health.clear_models(clear_model):
            if health.clear(model, ns.profile):
                cleared.append(health.pool_key(model, ns.profile))
    except health.HealthLockUnavailable:
        return _emit({"ok": False, "reason": "health_lock_unavailable",
                      "cleared": cleared})
    return _emit({"ok": True, "cleared": cleared})


def _cmd_health_lease(ns):
    """omni health lease acquire|renew|release — one JSON line via _emit."""
    cmd = getattr(ns, "lease_cmd", None)
    try:
        if cmd == "acquire":
            rec = health.lease_acquire(ns.token, ns.pool, ns.order, ns.ttl)
            return _emit({"ok": True, "lease": rec})
        if cmd == "renew":
            rec = health.lease_renew(ns.token, ttl_s=ns.ttl)
            return _emit({"ok": True, "lease": rec})
        if cmd == "release":
            health.lease_release(ns.token)
            return _emit({"ok": True, "released": ns.token})
    except ValueError as exc:
        reason = str(exc) or "lease_exists"
        return _emit({"ok": False, "reason": reason, "detail": reason})
    except KeyError as exc:
        reason = exc.args[0] if exc.args else "unknown_lease"
        if not isinstance(reason, str):
            reason = "unknown_lease"
        return _emit({"ok": False, "reason": reason, "detail": reason})
    except RuntimeError as exc:
        reason = str(exc)
        if reason == "health_skipped":
            return _emit({"ok": False, "reason": "health_skipped",
                          "detail": "health_skipped"})
        raise
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["session"]:
        from providers.session_router_cli import main as session_main
        return session_main(argv[1:])
    # Primed before any dispatch: the host-exec admission diagnostics print
    # paths and names, which can quote a credential the process holds.
    known_secrets.prime()
    if argv[:1] == ["host-exec"]:
        from providers.host_fence import main as host_exec
        return host_exec(argv[1:])
    if argv[:1] == ["dispatch-exec"]:
        from providers.dispatch_exec import main as dispatch_exec
        return dispatch_exec(argv[1:])
    if _is_agent_job_invocation(argv):
        return _main_agent_job(argv[1:])
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.cmd == "mcp":
        # Raw stdio server: no JSON envelope, no extra event loop. main() owns
        # its own asyncio.run and stdout carries JSON-RPC only. prime() above
        # already ran; the server's own prime() is an idempotent re-register.
        from providers.mcp_server import main as mcp_main
        mcp_main()
        return 0
    if ns.cmd == "contract":
        from providers.contract import build_contract
        sys.stdout.write(json.dumps(build_contract(parser), indent=2, sort_keys=True) + "\n")
        return 0
    if ns.cmd == "gen":
        return _emit(_call("llm_generate", _args_gen(ns)),
                     out=ns.out, plain=ns.plain)
    if ns.cmd == "big":
        return _emit(_call("llm_big_context", _args_big(ns)),
                     out=ns.out, plain=ns.plain)
    if ns.cmd == "agent":
        # --task-file may not exist / may not be readable: that is a LOUD
        # envelope on stdout, not a traceback. parser.error paths still exit 2.
        try:
            task = _agent_task(ns, parser)
        except OSError as e:
            return _emit({"ok": False, "reason": "runtime_error",
                          "detail": redact_secrets(str(e))[:300]}, out=ns.out, plain=ns.plain)
        agent_kwargs = {}
        if getattr(ns, "effort", None):
            agent_kwargs["effort"] = ns.effort
        if getattr(ns, "no_fence", False):
            agent_kwargs["fence_requested"] = False
        if getattr(ns, "expect_writes", False):
            agent_kwargs["expect_writes"] = True
        return _emit(_call_agent(task=task,
                                 model=ns.model or DEFAULT_AGENT_MODEL,
                                 workdir=ns.workdir,
                                 read_only=ns.read_only,
                                 timeout_s=ns.timeout,
                                 account_profile=ns.profile,
                                 allow_main=ns.allow_main,
                                 **agent_kwargs),
                     out=ns.out, plain=ns.plain, out_on_failure=True)
    if ns.cmd == "img":
        return _emit(_call("llm_image", _args_img(ns)))
    if ns.cmd == "models":
        if getattr(ns, "model_catalog_cmd", None) == "alias":
            return _main_alias(ns)
        if getattr(ns, "model_catalog_cmd", None):
            return _main_model_catalog(ns)
        return _emit(_call("list_models", {"timeout_s": ns.timeout}))
    if ns.cmd == "health":
        try:
            if getattr(ns, "all_profiles", False) and not getattr(ns, "probe", False):
                parser.error("--all-profiles requires --probe")
            if ns.health_cmd == "lease":
                return _cmd_health_lease(ns)
            if ns.health_cmd == "flag":
                # Strip a trailing @effort the way dispatch does, so a manual
                # flag lands on the same ledger key as the seat it names.
                from providers.seat_name import take_effort
                flag_model, _effort, split_err = take_effort(ns.model, None)
                if split_err:
                    return _emit(split_err)
                health.flag(flag_model, ns.profile, until=ns.until, source="manual")
            if ns.health_cmd == "clear":
                return _cmd_health_clear(ns)
            payload = health.snapshot()
            probed = 0
            jobs_repaired = []
            if getattr(ns, "probe", False):
                jobs_repaired = jobs_mod.sweep_orphans()
                if getattr(ns, "all_profiles", False):
                    from providers.profile_check import probe_all
                    quota = probe_all()
                    payload = health.snapshot()
                    payload["quota"] = quota
                    if isinstance(quota, dict):
                        for group in quota.values():
                            if isinstance(group, dict):
                                probed += len(group)
                else:
                    from providers.codex_quota import read_rate_limits
                    profile = getattr(ns, "profile", None) or "default"
                    info, err = read_rate_limits(profile)
                    health.apply_codex_quota("codex", profile, info, err)
                    payload = health.snapshot()
                    if isinstance(info, dict):
                        quota_entry = {
                            "used_percent": info.get("used_percent"),
                            "resets_at": info.get("resets_at"),
                            "window_mins": info.get("window_mins"),
                            "reached": info.get("reached"),
                            "plan": info.get("plan"),
                            "checked_at": info.get("checked_at"),
                        }
                    else:
                        quota_entry = {
                            "used_percent": None,
                            "resets_at": None,
                            "window_mins": None,
                            "reached": False,
                            "plan": None,
                            "checked_at": None,
                        }
                        if isinstance(err, dict):
                            quota_entry["reason"] = err.get("reason")
                    payload["quota"] = {"codex": {str(profile): quota_entry}}
                    probed = 1
                payload["jobs_repaired"] = jobs_repaired
            if getattr(ns, "quiet", False):
                compact = {
                    "ok": True,
                    "probed": probed,
                    "cooling": sorted(health.snapshot().keys()),
                    "repaired": len(jobs_repaired),
                }
                sys.stdout.write(
                    json.dumps(compact, ensure_ascii=False) + "\n")
                return 0
            lease_rows = health.leases()
            if lease_rows:
                payload["leases"] = lease_rows
            sys.stdout.write(json.dumps(mask_envelope(payload), ensure_ascii=False) + "\n")
            return 0
        except Exception as e:
            return _emit({"ok": False, "reason": "runtime_error",
                          "detail": redact_secrets(str(e))[:300]})
    if ns.cmd == "canary":
        return _main_canary(ns)
    if ns.cmd == "artifact":
        return _main_artifact(ns)
    if ns.cmd == "profiles":
        try:
            return _main_profiles(ns)
        except Exception as e:
            return _emit({"ok": False, "reason": "runtime_error",
                          "detail": redact_secrets(str(e))[:300]})
    return 2  # unreachable: argparse enforces a subcommand


if __name__ == "__main__":
    sys.exit(main())
