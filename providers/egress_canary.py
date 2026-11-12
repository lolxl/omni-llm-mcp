# no-opencc
"""Per-seam egress canary. One process, one seam.

Run as ``python3 -m providers.egress_canary``. Each seam is the packaged
form of the omni-egress transport stub: the same patches, the same
sentinels, and the same exit codes. The caller still supplies the
environment that wraps the process (API base, fake keys). If ``TMPDIR``
is unset or empty, this module sets it to ``<state>/tmp`` and creates
that directory before importing any seam module. A ``TMPDIR`` that is
already set is left as the caller provided it.

A ``--text`` value that starts with ``-`` must be passed as
``--text=VALUE``. ``--gate`` is exactly ``0`` or ``1``. ``--state``,
``--workdir`` and ``--image`` are turned into absolute real paths before
use.

B must run this module with ``cwd`` or ``PYTHONPATH`` set to the checkout
it tests, and confirm the copy with ``--where``.

``HOME`` is not changed by the module. The caller must run it under a
home it is willing to have read. The unit tests use the synthetic home.

``jobs_start`` refuses a state dir whose ``jobs`` directory is not empty,
so every ``jobs_start`` run needs its own fresh state dir.

Before its lane call, a seam that sends to codex, claude, grok, agy, or cursor
writes a synthetic v2 catalog for that profile. The model id, and for codex
the effort the adapter fills in, are read at that moment from the same
constants the seam call uses. Missing catalog_v2 exits 43.

Exit codes match the stub contract:
  1  egress refusal on stdout (detail starts with OMNI-EGRESS-DENY)
  2  usage error, STATE-NOT-CLEAN, or a jobs refusal that wrote prompt.txt
  3  the lane returned something else
  4  agy SDK has no local SDK (SDK-UNAVAILABLE-FALLBACK)
 42  a stubbed transport ran (FAIL-OPEN-REACHED-TRANSPORT, via os._exit)
 43  a missing patch target or a catalog seed failure (STUB-TARGET-MISSING:<name>, via os._exit)
"""
import argparse
import asyncio
import datetime
import os
import subprocess
import sys


SEAM_IDS = (
    "agent_lane",
    "jobs_start",
    "grok_image",
    "xai_image",
    "cli_grok",
    "cli_codex",
    "cli_claude",
    "cli_pi",
    "cli_agy",
    "cli_cursor",
    "agy_sdk",
    "openai_compat",
    "dispatch_route",
    "dispatch_tool",
)

# Fixture model ids. One spelling each, shared by the seam spec and the seed.
_FIXTURE_CODEX = "fixture-codex"
_FIXTURE_CLAUDE = "fixture-claude"
_FIXTURE_PI = "fixture-pi"
_FIXTURE_AGY = "fixture-agy"
_FIXTURE_CURSOR = "fixture-cursor"

# Literal in dispatch.call_tool's llm_image branch
# (args.get("model") or this value) and the default argument of
# xai_image.run_xai_image. providers/ exports no constant for it.
_GROK_2_IMAGE_MODEL = "grok-2-image"

# Success label on grok_image_cli.run_grok_image_edit stays "grok-image-edit".
# The CLI is admitted as grok_image_cli.GROK_IMAGE_MODEL, imported below.


def _grok_model():
    """config.DEFAULT_GROK_MODEL at this call, not at import."""
    from providers import config
    return config.DEFAULT_GROK_MODEL


def _agent_model():
    """config.DEFAULT_AGENT_MODEL at this call, not at import."""
    from providers import config
    return config.DEFAULT_AGENT_MODEL


def _codex_efforts():
    """``medium`` plus the effort codex fills in when none was given.

    Read at this call and deduped. Order is the known-token order
    catalog_record stores, so the seed matches the written row.
    """
    from providers import config
    tokens = []
    for token in ("medium", config.DEFAULT_CODEX_REASONING_EFFORT):
        if token not in tokens:
            tokens.append(token)
    order = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    tokens.sort(key=lambda token: (
        (0, order.index(token)) if token in order else (1, token)))
    return tuple(tokens)


# cli_<provider> -> (module, function, binary, model, bin resolver).
# grok's model is _grok_model, called when the seam runs.
# cursor's binary is cursor-agent and its lookup is resolve_bin.
_CLI_SEAMS = {
    "cli_grok": ("grok", "run_grok_subprocess", "grok", _grok_model, "which"),
    "cli_codex": ("codex", "run_codex_subprocess", "codex", _FIXTURE_CODEX, "which"),
    "cli_claude": ("claude", "run_claude_subprocess", "claude", _FIXTURE_CLAUDE, "which"),
    "cli_pi": ("pi", "run_pi_subprocess", "pi", _FIXTURE_PI, "which"),
    "cli_agy": ("agy", "run_agy_subprocess", "agy", _FIXTURE_AGY, "which"),
    "cli_cursor": (
        "cursor", "run_cursor_subprocess", "cursor-agent", _FIXTURE_CURSOR,
        "resolve_bin",
    ),
}

_NEEDS_WORKDIR = ("agent_lane", "jobs_start")
_NEEDS_IMAGE = ("grok_image", "dispatch_tool")


def _cli_call(seam):
    """One cli spec. A callable model is read now, not from import."""
    module, function, binary, model, resolver = _CLI_SEAMS[seam]
    if callable(model):
        model = model()
    return module, function, binary, model, resolver


def _catalog_seeds():
    """Synthetic rows, built when the seam runs.

    Model ids come from the same constants the seam call uses.
    ``efforts`` is supported_efforts; () omits the key.
    codex, claude, and grok list "medium" so an omitted effort is admitted.
    Codex also lists config.DEFAULT_CODEX_REASONING_EFFORT: the adapter
    fills that in before refuse_payload. agy and cursor carry the level
    in the id; decide admits effort=None with no supported_efforts.
    pi, openai_compat, and xai_image are absent: they keep today's gate.
    """
    from providers.grok_image_cli import GROK_IMAGE_MODEL
    medium = ("medium",)
    agent = _agent_model()
    grok = _grok_model()
    return {
        "agent_lane": (
            ("grok", "default", ((agent, medium),)),
        ),
        "jobs_start": (
            ("grok", "default", ((agent, medium),)),
        ),
        "grok_image": (
            ("grok", "default", ((GROK_IMAGE_MODEL, medium),)),
        ),
        "cli_grok": (
            ("grok", "default", ((grok, medium),)),
        ),
        "cli_codex": (
            ("codex", "default", ((_FIXTURE_CODEX, _codex_efforts()),)),
        ),
        "cli_claude": (
            ("claude", "default", ((_FIXTURE_CLAUDE, medium),)),
        ),
        "cli_agy": (
            ("agy", "default", ((_FIXTURE_AGY, ()),)),
        ),
        "cli_cursor": (
            ("cursor", "default", ((_FIXTURE_CURSOR, ()),)),
        ),
        "agy_sdk": (
            ("agy", "default", ((_FIXTURE_AGY, ()),)),
        ),
        "dispatch_route": (
            ("grok", "default", ((grok, medium),)),
        ),
        # call_tool's gate still holds _GROK_2_IMAGE_MODEL; the cli branch
        # admits GROK_IMAGE_MODEL.
        "dispatch_tool": (
            ("grok", "default", (
                (_GROK_2_IMAGE_MODEL, medium),
                (GROK_IMAGE_MODEL, medium),
            )),
        ),
    }


def _fail_open(*_a, **_k):
    sys.stdout.write("FAIL-OPEN-REACHED-TRANSPORT\n")
    sys.stdout.flush()
    os._exit(42)


async def _fail_open_async(*_a, **_k):
    _fail_open()


def _stub_attr(obj, name, value):
    if not hasattr(obj, name):
        sys.stdout.write("STUB-TARGET-MISSING:%s\n" % name)
        sys.stdout.flush()
        os._exit(43)
    setattr(obj, name, value)


def _argv0(*a, **k):
    argv = a[0] if a else k.get("args")
    if isinstance(argv, (list, tuple)) and argv:
        return os.path.basename(str(argv[0]))
    if isinstance(argv, str) and argv.split():
        return os.path.basename(argv.split()[0])
    return ""


def _install_backstop(scan_mods):
    """Popen/run backstop. git passes through so a worktree fence can run."""
    def _backstop(orig):
        def _wrapped(*a, **k):
            if _argv0(*a, **k) in ("git", "git.exe"):
                return orig(*a, **k)
            _fail_open()
        return _wrapped

    orig_popen = subprocess.Popen
    orig_run = subprocess.run
    _stub_attr(subprocess, "Popen", _backstop(orig_popen))
    _stub_attr(subprocess, "run", _backstop(orig_run))
    for mod in scan_mods:
        if hasattr(mod, "create_subprocess_exec"):
            _stub_attr(mod, "create_subprocess_exec", _fail_open_async)
        if hasattr(mod, "Popen"):
            _stub_attr(mod, "Popen", _backstop(orig_popen))
        if hasattr(mod, "subprocess"):
            _stub_attr(mod.subprocess, "Popen", _backstop(orig_popen))
            _stub_attr(mod.subprocess, "run", _backstop(orig_run))
        if hasattr(mod, "asyncio"):
            _stub_attr(mod.asyncio, "create_subprocess_exec", _fail_open_async)


def _pop_prefix(prefix):
    for key in list(os.environ):
        if key.startswith(prefix):
            os.environ.pop(key, None)


def _arm(state, gate, pop_depth=True):
    if pop_depth:
        os.environ.pop("OMNI_LLM_DEPTH", None)
    os.environ["OMNI_LLM_STATE_DIR"] = state
    os.environ["OMNI_EGRESS_GATE"] = gate


def _catalog_observed_at():
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _missing_catalog():
    sys.stdout.write("STUB-TARGET-MISSING:catalog_v2\n")
    sys.stdout.flush()
    os._exit(43)


def _short_seed_code(value):
    """True when *value* is a short token, not an endpoint or a message."""
    if not isinstance(value, str) or not value or len(value) > 64:
        return False
    for ch in value:
        if not (ch.isalnum() or ch in "_-"):
            return False
    return True


def _seed_failure(exc):
    """Building or saving a seed record failed. Exit 43.

    Print the class and a short ``code`` or ``reason`` only. The message
    can carry an endpoint value.
    """
    code = ""
    for attr in ("code", "reason"):
        value = getattr(exc, attr, None)
        if _short_seed_code(value):
            code = value
            break
    line = "STUB-TARGET-MISSING:catalog_seed:%s" % type(exc).__name__
    if code:
        line = "%s:%s" % (line, code)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    os._exit(43)


def _synthetic_record(provider, models, observed_at):
    """Validated v2 record. ``models`` is ``((id, efforts), ...)``.

    An empty ``efforts`` tuple omits ``supported_efforts``. A non-empty
    tuple is that field, unchanged.
    """
    from providers import catalog_record

    payload_models = []
    for model_id, efforts in models:
        row = {"id": model_id, "name": model_id}
        if efforts:
            row["supported_efforts"] = list(efforts)
        payload_models.append(row)
    record, _warnings = catalog_record.validate_strict(provider, {
        "provider": provider,
        "source": "live",
        "observed_at": observed_at,
        "availability": "provider_advertised",
        "models": payload_models,
    })
    return record


def _seed_catalog(seam):
    """Write this seam's v2 records into the armed state dir.

    No-op when the seam sends to a provider that keeps today's gate.
    The fingerprint reads the process environment, so this runs after
    ``_arm`` and after any profile env the lane call will see.
    """
    specs = _catalog_seeds().get(seam)
    if not specs:
        return
    try:
        from providers import catalog_record, catalog_v2
    except ImportError:
        _missing_catalog()
    if (not hasattr(catalog_record, "validate_strict")
            or not hasattr(catalog_v2, "save")):
        _missing_catalog()
    try:
        observed_at = _catalog_observed_at()
        for provider, profile, models in specs:
            catalog_v2.save(
                provider, profile,
                _synthetic_record(provider, models, observed_at))
    except Exception as exc:
        _seed_failure(exc)


def _abspath(value):
    """Absolute real path. Empty values are rejected before this is called."""
    return os.path.realpath(os.path.expanduser(value))


def _pin_tmpdir(state):
    """Keep a gate-0 ``os._exit`` from leaving the raw prompt in the system temp.

    A caller that already set ``TMPDIR`` (B's wrapper does) is left alone.
    This runs before any seam module is imported, so the first
    ``tempfile`` lookup in that seam sees the pin.
    """
    if os.environ.get("TMPDIR"):
        return
    path = os.path.join(state, "tmp")
    os.makedirs(path, exist_ok=True)
    os.environ["TMPDIR"] = path


def _require_fresh_jobs_dir(state):
    """Refuse a ``jobs_start`` whose ``<state>/jobs`` already holds a run.

    The gate-1 check walks ``<state>/jobs`` for a ``prompt.txt``, so any
    file left there by an earlier run would be reported as a leak.
    """
    jobs_dir = os.path.join(state, "jobs")
    if os.path.isdir(jobs_dir) and os.listdir(jobs_dir):
        print("STATE-NOT-CLEAN %s" % jobs_dir)
        raise SystemExit(2)


def _image_out(state):
    # The shell stub took a separate out dir. This CLI has --image and
    # --state only, so the out dir lives under the caller's state dir.
    out = os.path.join(state, "image-out")
    os.makedirs(out, exist_ok=True)
    return out


def _refused_detail(res):
    """Flat detail, or the detail nested under a jobs/agent ``refused`` key."""
    d = ""
    if isinstance(res, dict):
        inner = res.get("refused")
        if isinstance(inner, dict):
            d = inner.get("detail") or ""
        else:
            d = res.get("detail") or ""
    return d


def _finish_deny(detail):
    if isinstance(detail, str) and detail.startswith("OMNI-EGRESS-DENY"):
        print(detail)
        raise SystemExit(1)
    print("UNEXPECTED-LANE-RESULT")
    raise SystemExit(3)


def _which_stub(orig, names, prefix):
    def _which(cmd, *a, **k):
        if cmd in names:
            return "%s%s" % (prefix, cmd)
        return orig(cmd, *a, **k)
    return _which


def _seam_agent_lane(text, workdir, state, gate):
    _arm(state, gate)
    import providers.cli_run as cli_run
    import providers.agent_lane as agent_lane

    _stub_attr(cli_run, "_spawn", _fail_open_async)
    _stub_attr(cli_run, "run_cli", _fail_open_async)
    _stub_attr(agent_lane, "run_cli", _fail_open_async)
    _stub_attr(asyncio, "create_subprocess_exec", _fail_open_async)
    _install_backstop((agent_lane, cli_run))
    if not hasattr(agent_lane, "shutil") or not hasattr(agent_lane.shutil, "which"):
        sys.stdout.write("STUB-TARGET-MISSING:which\n")
        sys.stdout.flush()
        os._exit(43)
    orig_which = agent_lane.shutil.which
    _stub_attr(
        agent_lane.shutil, "which",
        _which_stub(orig_which, ("grok", "codex", "claude", "pi"), "/o38-stub-"),
    )
    _seed_catalog("agent_lane")
    res = asyncio.run(agent_lane.run_agent(text, _agent_model(), workdir))
    _finish_deny(_refused_detail(res))


def _seam_jobs_start(text, workdir, state, gate):
    _arm(state, gate)
    import providers.jobs as jobs

    if not hasattr(jobs, "subprocess"):
        sys.stdout.write("STUB-TARGET-MISSING:subprocess\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(jobs.subprocess, "Popen", _fail_open)
    _seed_catalog("jobs_start")
    res = jobs.start(text, model=_agent_model(), workdir=workdir)
    d = _refused_detail(res)
    if isinstance(d, str) and d.startswith("OMNI-EGRESS-DENY"):
        print(d)
        leaked = []
        if state and os.path.isdir(state):
            for root, _dirs, files in os.walk(state):
                if "prompt.txt" in files:
                    leaked.append(os.path.join(root, "prompt.txt"))
        if leaked:
            print("jobs-refuse-leaked-prompt.txt")
            raise SystemExit(2)
        print("jobs-refuse-no-prompt.txt")
        raise SystemExit(1)
    print("UNEXPECTED-LANE-RESULT")
    raise SystemExit(3)


def _seam_grok_image(text, image, state, gate):
    _arm(state, gate, pop_depth=False)
    import providers.grok_image_cli as gic

    if not hasattr(gic, "subprocess"):
        sys.stdout.write("STUB-TARGET-MISSING:subprocess\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(gic.subprocess, "run", _fail_open)
    _stub_attr(gic.subprocess, "Popen", _fail_open)
    if not hasattr(gic, "shutil") or not hasattr(gic.shutil, "which"):
        sys.stdout.write("STUB-TARGET-MISSING:which\n")
        sys.stdout.flush()
        os._exit(43)
    orig_which = gic.shutil.which

    def _which(cmd, *a, **k):
        if cmd == "grok":
            return "/o38-stub-grok"
        return orig_which(cmd, *a, **k)

    _stub_attr(gic.shutil, "which", _which)
    out_dir = _image_out(state)
    _seed_catalog("grok_image")
    res = asyncio.run(gic.run_grok_image_edit(text, image, out_dir=out_dir))
    _finish_deny(res.get("detail") or "" if isinstance(res, dict) else "")


def _seam_xai_image(text, state, gate):
    # Profile keys win over XAI_API_KEY. Drop them here so a live
    # XAI_API_KEY_<PROFILE> cannot aim a stub miss at the real API.
    # XAI_API_BASE is also pinned inside the process; the caller still
    # exports the same names around the process.
    _pop_prefix("XAI_API_KEY_")
    os.environ["XAI_API_KEY"] = "o38-fixture-not-a-key"
    os.environ["XAI_API_BASE"] = "http://127.0.0.1:1/v1"
    _arm(state, gate, pop_depth=False)
    import urllib.request
    import providers.xai_image as xi

    _stub_attr(urllib.request, "urlopen", _fail_open)
    _seed_catalog("xai_image")
    res = asyncio.run(xi.run_xai_image(text))
    _finish_deny(res.get("detail") or "" if isinstance(res, dict) else "")


def _import_cli_mod(mod_name):
    if mod_name == "grok":
        import providers.grok as mod
    elif mod_name == "codex":
        import providers.codex as mod
    elif mod_name == "claude":
        import providers.claude as mod
    elif mod_name == "pi":
        import providers.pi as mod
    elif mod_name == "agy":
        import providers.agy as mod
    elif mod_name == "cursor":
        import providers.cursor as mod
    else:
        sys.stdout.write("STUB-TARGET-MISSING:providers.%s\n" % mod_name)
        sys.stdout.flush()
        os._exit(43)
    return mod


def _seam_cli(mod_name, func_name, bin_name, model, text, state, gate, bin_resolver):
    _arm(state, gate)
    if mod_name == "agy":
        # Not read on the default-profile subprocess path. Kept so import
        # cannot see a live SDK config dir.
        os.environ["AGY_CONFIG_DIR"] = state
    import providers.cli_run as cli_run
    mod = _import_cli_mod(mod_name)
    _stub_attr(cli_run, "_spawn", _fail_open_async)
    _stub_attr(cli_run, "run_cli", _fail_open_async)
    if not hasattr(mod, func_name):
        sys.stdout.write("STUB-TARGET-MISSING:%s\n" % func_name)
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(mod, "run_cli", _fail_open_async)
    _stub_attr(asyncio, "create_subprocess_exec", _fail_open_async)
    _install_backstop((mod, cli_run))
    # Codex applies prepare_host_fence before refuse_payload and has no
    # unwrapped fallback. Pass the fence through so this case proves the
    # gate; the run_cli / _spawn / exec backstops still catch a transport.
    if mod_name == "codex":
        def _fence_passthrough(argv, *_a, **k):
            return argv, k.get("env"), {}
        _stub_attr(mod, "prepare_host_fence", _fence_passthrough)
    if bin_resolver == "resolve_bin":
        def _resolve_bin(*_a, **_k):
            return "/o46-stub-%s" % bin_name
        _stub_attr(mod, "resolve_bin", _resolve_bin)
    elif bin_resolver == "which":
        if mod_name == "agy":
            # find_agy calls vendor_bin.which. agy.py imports rmtree only,
            # so a shutil.which stub cannot target it. A renamed
            # vendor_bin.which stays loud (exit 43).
            if not hasattr(mod, "vendor_bin") or not hasattr(mod.vendor_bin, "which"):
                sys.stdout.write("STUB-TARGET-MISSING:vendor_bin.which\n")
                sys.stdout.flush()
                os._exit(43)
            orig_which = mod.vendor_bin.which

            def _agy_which(name, *a, **k):
                if name == bin_name:
                    return "/o42-stub-%s" % name
                return orig_which(name, *a, **k)

            _stub_attr(mod.vendor_bin, "which", _agy_which)
        elif not hasattr(mod, "shutil") or not hasattr(mod.shutil, "which"):
            sys.stdout.write("STUB-TARGET-MISSING:which\n")
            sys.stdout.flush()
            os._exit(43)
        else:
            orig_which = mod.shutil.which

            def _which(cmd, *a, **k):
                if cmd == bin_name:
                    return "/o42-stub-%s" % cmd
                return orig_which(cmd, *a, **k)

            _stub_attr(mod.shutil, "which", _which)
    else:
        sys.stdout.write("STUB-TARGET-MISSING:bin_resolver:%s\n" % bin_resolver)
        sys.stdout.flush()
        os._exit(43)
    _seed_catalog("cli_%s" % mod_name)
    res = asyncio.run(getattr(mod, func_name)(text, model, 5, "default"))
    detail = ""
    if isinstance(res, dict):
        detail = res.get("detail") or ""
    _finish_deny(detail)


def _seam_agy_sdk(text, state, gate):
    _arm(state, gate)
    os.environ["AGY_CONFIG_DIR"] = state
    import providers.cli_run as cli_run
    import providers.agy as mod

    if not hasattr(mod, "run_agy_sdk"):
        sys.stdout.write("STUB-TARGET-MISSING:run_agy_sdk\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(cli_run, "_spawn", _fail_open_async)
    _stub_attr(cli_run, "run_cli", _fail_open_async)
    _stub_attr(mod, "run_cli", _fail_open_async)
    _stub_attr(asyncio, "create_subprocess_exec", _fail_open_async)
    # Agent may be None when the SDK is absent. hasattr still holds.
    # Replacing None does not fabricate a transport: config build keys off
    # LocalAgentConfig, which this fixture does not stub.
    _stub_attr(mod, "Agent", _fail_open)
    _install_backstop((mod, cli_run))
    _seed_catalog("agy_sdk")
    res = asyncio.run(mod.run_agy_sdk(text, _FIXTURE_AGY, 5, None, "default"))
    detail = ""
    if isinstance(res, dict):
        detail = res.get("detail") or ""
    if isinstance(detail, str) and detail.startswith("OMNI-EGRESS-DENY"):
        print(detail)
        raise SystemExit(1)
    # No local SDK: LocalAgentConfig is None and the lane returns
    # sdk_unavailable. That is a fallback signal, not a transport.
    if isinstance(res, dict) and res.get("reason") == "sdk_unavailable":
        print("SDK-UNAVAILABLE-FALLBACK")
        raise SystemExit(4)
    print("UNEXPECTED-LANE-RESULT")
    raise SystemExit(3)


def _seam_openai_compat(text, state, gate):
    # openai_compat.py does not read OPENAI_API_BASE; api_base is a kwarg.
    # The env pin is leftover isolation, same as the stub. Profile keys are
    # dropped so they cannot override the fixture key.
    _pop_prefix("OPENAI_API_KEY_")
    os.environ["OPENAI_API_KEY"] = "o42-fixture-not-a-key"
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:1/v1"
    _arm(state, gate)
    import urllib.request
    import providers.openai_compat as oc

    if not hasattr(oc, "run_openai_compatible"):
        sys.stdout.write("STUB-TARGET-MISSING:run_openai_compatible\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(urllib.request, "urlopen", _fail_open)
    _seed_catalog("openai_compat")
    res = asyncio.run(oc.run_openai_compatible(
        text, "gpt-4.1", 5, "default", api_base="http://127.0.0.1:1/v1"))
    detail = ""
    if isinstance(res, dict):
        detail = res.get("detail") or ""
    _finish_deny(detail)


def _seam_dispatch_route(text, state, gate):
    _arm(state, gate)
    import urllib.request
    import providers.dispatch as dispatch

    if not hasattr(dispatch, "route_to_provider"):
        sys.stdout.write("STUB-TARGET-MISSING:route_to_provider\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(asyncio, "create_subprocess_exec", _fail_open_async)
    if hasattr(dispatch, "asyncio"):
        _stub_attr(dispatch.asyncio, "create_subprocess_exec", _fail_open_async)
    _stub_attr(subprocess, "Popen", _fail_open)
    _stub_attr(subprocess, "run", _fail_open)
    _stub_attr(urllib.request, "urlopen", _fail_open)
    pairs = (
        (dispatch.grok_mod, "run_grok_subprocess"),
        (dispatch.codex_mod, "run_codex_subprocess"),
        (dispatch.claude_mod, "run_claude_subprocess"),
        (dispatch.pi_mod, "run_pi_subprocess"),
        (dispatch.agy_mod, "run_agy"),
        (dispatch.agy_mod, "run_agy_subprocess"),
        (dispatch.openai_mod, "run_openai_compatible"),
        (dispatch.cursor_mod, "run_cursor_subprocess"),
    )
    for obj, name in pairs:
        _stub_attr(obj, name, _fail_open_async)
    _seed_catalog("dispatch_route")
    res = asyncio.run(dispatch.route_to_provider(
        text, _grok_model(), 5, [], "default"))
    detail = ""
    if isinstance(res, dict):
        detail = res.get("detail") or ""
    _finish_deny(detail)


def _seam_dispatch_tool(text, image, state, gate):
    _arm(state, gate)
    import urllib.request
    import providers.dispatch as dispatch

    if not hasattr(dispatch, "call_tool"):
        sys.stdout.write("STUB-TARGET-MISSING:call_tool\n")
        sys.stdout.flush()
        os._exit(43)
    if not hasattr(dispatch, "grok_image_cli_mod"):
        sys.stdout.write("STUB-TARGET-MISSING:grok_image_cli_mod\n")
        sys.stdout.flush()
        os._exit(43)
    if not hasattr(dispatch, "xai_image_mod"):
        sys.stdout.write("STUB-TARGET-MISSING:xai_image_mod\n")
        sys.stdout.flush()
        os._exit(43)
    _stub_attr(dispatch.grok_image_cli_mod, "run_grok_image_edit", _fail_open_async)
    _stub_attr(dispatch.xai_image_mod, "run_xai_image", _fail_open_async)
    _stub_attr(urllib.request, "urlopen", _fail_open)
    _stub_attr(asyncio, "create_subprocess_exec", _fail_open_async)
    _install_backstop((
        dispatch, dispatch.grok_image_cli_mod, dispatch.xai_image_mod))
    out_dir = _image_out(state)
    args = {"prompt": text, "source_image": image, "out_dir": out_dir}
    _seed_catalog("dispatch_tool")
    res = asyncio.run(dispatch.call_tool("llm_image", args))
    detail = ""
    if isinstance(res, dict):
        detail = res.get("detail") or ""
    _finish_deny(detail)


def _run_seam(seam, gate, text, state, workdir, image):
    if seam == "jobs_start":
        _require_fresh_jobs_dir(state)
    _pin_tmpdir(state)
    if seam == "agent_lane":
        _seam_agent_lane(text, workdir, state, gate)
    elif seam == "jobs_start":
        _seam_jobs_start(text, workdir, state, gate)
    elif seam == "grok_image":
        _seam_grok_image(text, image, state, gate)
    elif seam == "xai_image":
        _seam_xai_image(text, state, gate)
    elif seam in _CLI_SEAMS:
        mod_name, func_name, bin_name, model, resolver = _cli_call(seam)
        _seam_cli(mod_name, func_name, bin_name, model, text, state, gate, resolver)
    elif seam == "agy_sdk":
        _seam_agy_sdk(text, state, gate)
    elif seam == "openai_compat":
        _seam_openai_compat(text, state, gate)
    elif seam == "dispatch_route":
        _seam_dispatch_route(text, state, gate)
    elif seam == "dispatch_tool":
        _seam_dispatch_tool(text, image, state, gate)
    else:
        raise SystemExit(2)


def _print_where():
    here = os.path.realpath(__file__)
    print(here)
    print(os.path.realpath(os.path.dirname(here)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="providers.egress_canary")
    parser.add_argument("--list", action="store_true",
                        help="print seam ids, one per line")
    parser.add_argument("--where", action="store_true",
                        help="print this file and the providers directory")
    parser.add_argument("--seam", choices=SEAM_IDS)
    parser.add_argument("--gate", choices=("0", "1"))
    parser.add_argument("--text")
    parser.add_argument("--state")
    parser.add_argument("--workdir")
    parser.add_argument("--image")
    args = parser.parse_args(argv)
    modes = (bool(args.list), bool(args.where), bool(args.seam))
    if sum(modes) != 1:
        parser.error("pass exactly one of --list, --where or --seam")
    if args.list:
        for seam_id in SEAM_IDS:
            print(seam_id)
        return 0
    if args.where:
        _print_where()
        return 0
    missing = []
    if args.gate not in ("0", "1"):
        missing.append("--gate")
    if not args.text:
        missing.append("--text")
    if not args.state:
        missing.append("--state")
    if args.workdir == "" or (args.seam in _NEEDS_WORKDIR and not args.workdir):
        missing.append("--workdir")
    if args.image == "" or (args.seam in _NEEDS_IMAGE and not args.image):
        missing.append("--image")
    if missing:
        parser.error("missing %s" % ", ".join(missing))
    state = _abspath(args.state)
    workdir = _abspath(args.workdir) if args.workdir else None
    image = _abspath(args.image) if args.image else None
    _run_seam(args.seam, args.gate, args.text, state, workdir, image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
