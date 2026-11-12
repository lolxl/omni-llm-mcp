# no-opencc
"""Shared constants for omni-llm-mcp providers and routing."""
import os

# Must name a model `agy models` still lists (slug or display name); vendors
# retire them. `list_models` reports default_available so drift shows up.
DEFAULT_MODEL = "Gemini 3.8 Flash (High)"
DEFAULT_TIMEOUT_S = 180
BIG_CONTEXT_TIMEOUT_S = 240
DEFAULT_MODELS_TIMEOUT_S = 120

# Ruling retire-gpt-5-6-sol (2026-09-30): move GPT-5.6 Sol to GPT-6.1 Sol
# on Codex; the Cursor route stays on GPT-5.6 Sol until Cursor offers GPT-6.
SOL_CODEX_MODEL = "gpt-6.1-sol"
SOL_CURSOR_MODEL = "gpt-5.6-sol"
# Codex CLI (subscription-auth). Used when a `codex/<model>` name omits the model.
DEFAULT_CODEX_MODEL = SOL_CODEX_MODEL
DEFAULT_CODEX_REASONING_EFFORT = "high"  # omitted consult/agent path still sends this

# Public `effort` token. Omitted = backend default (codex consult/agent: high;
# grok/claude: no flag). Specified values are normalized (strip + lower).
# Sets are the live CLI enums (grok 1.0.4: low|medium|high|xhigh only).
# Invalid token or a backend with no effort flag → LOUD unsupported_effort.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")

# Cursor Agent CLI has NO effort flag. Effort is encoded in the model id, either
# as an exploded legacy slug (`<base>-<effort>[-fast]`) or as a full parameter
# string (`<base>[context=...,effort=...,fast=...]`, every parameter required,
# and the parameter id differs per family). The consult lane takes the legacy
# slug road: the public `effort` token is mapped onto the slug here, so callers
# keep ONE spelling of effort across every backend.
#
# Snapshot of `cursor-agent --list-models` (CLI 2026.09.08-6caf4ff). Per-account
# catalogs differ: a base the table does not carry is a LOUD unsupported_effort,
# and a slug the account happens to lack fails LOUD at the CLI rather than
# silently downgrading. Regenerate by stripping a trailing `-fast`, then a
# trailing effort word, off every id the CLI lists. `-none` variants are dropped
# on purpose: `none` is not a public effort token.
CURSOR_EFFORT_SLUGS = {
    "claude-4.5-opus": ("high",),
    "claude-4.5-opus-thinking": ("high",),
    "claude-4.6-opus": ("high", "max"),
    "claude-4.6-opus-thinking": ("high", "max"),
    "claude-4.6-sonnet": ("medium",),
    "claude-4.6-sonnet-thinking": ("medium",),
    "claude-fable-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-fable-5-thinking": ("low", "medium", "high", "xhigh", "max"),
    "claude-fable-5-1": ("low", "medium", "high", "xhigh", "max"),
    "claude-fable-5-1-thinking": ("low", "medium", "high", "xhigh", "max"),
    "claude-fable-5-2": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-4-7": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-4-7-thinking": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-4-8": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-4-8-thinking": ("low", "medium", "high", "xhigh", "max"),
    "claude-opus-5": ("low", "medium", "high"),
    "claude-opus-5-thinking": ("low", "medium", "high", "xhigh", "max"),
    "claude-sonnet-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-sonnet-5-thinking": ("low", "medium", "high", "xhigh", "max"),
    "cursor-grok-4.5": ("low", "medium", "high"),
    "cursor-grok-4.6": ("low", "medium", "high", "xhigh"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "glm-5.2": ("high", "max"),
    "gpt-5.1": ("low", "high"),
    "gpt-5.2": ("low", "high", "xhigh"),
    "gpt-5.3-codex": ("low", "high", "xhigh"),
    "gpt-5.4": ("low", "medium", "high", "xhigh"),
    "gpt-5.4-mini": ("low", "medium", "high", "xhigh"),
    "gpt-5.4-nano": ("low", "medium", "high", "xhigh"),
    "gpt-5.5": ("low", "medium", "high", "xhigh"),
    "gpt-5.6-luna": ("low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-sol": ("low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-terra": ("low", "medium", "high", "xhigh", "max"),
    "kimi-k3": ("low", "high", "max"),
    "muse-spark-1.3": ("minimal", "low", "medium", "high", "xhigh", "max"),
}
# Ids the `<base>-<token>` rule cannot spell: a different word for the token, or
# (pre-5 Claude families) `-thinking` sitting AFTER the effort word.
CURSOR_EFFORT_SLUG_OVERRIDES = {
    ("gpt-5.5", "xhigh"): "gpt-5.5-extra-high",
    ("claude-4.5-opus-thinking", "high"): "claude-4.5-opus-high-thinking",
    ("claude-4.6-opus-thinking", "high"): "claude-4.6-opus-high-thinking",
    ("claude-4.6-opus-thinking", "max"): "claude-4.6-opus-max-thinking",
    ("claude-4.6-sonnet-thinking", "medium"): "claude-4.6-sonnet-medium-thinking",
}
# Bases whose BARE id is absent from the cursor catalog (CLI 2026-09-03:
# `cursor-agent models` lists only effort-suffixed ids for these families,
# with `-medium` labelled as the plain one). Omitted effort on these is
# folded into the table token in resolve_model. Keep this explicit: do not
# infer membership from CURSOR_EFFORT_SLUGS.
CURSOR_DEFAULT_EFFORT = {
    "gpt-5.6-sol": "medium",
    "gpt-5.6-luna": "medium",
    "gpt-5.6-terra": "medium",
}
EFFORT_BY_PROVIDER = {
    "grok": frozenset(("low", "medium", "high", "xhigh")),
    # Installed agy 1.3.3 help explicitly lists low|medium|high|xhigh|max.
    # The provider checks the installed CLI again before forwarding the flag.
    "agy": frozenset(("low", "medium", "high", "xhigh", "max")),
    "claude": frozenset(("low", "medium", "high", "xhigh", "max")),
    "codex": frozenset(("minimal", "low", "medium", "high", "xhigh", "max")),
    # pi --thinking: off|minimal|low|medium|high|xhigh|max. `off` is not in
    # the public effort token set (none is refused); omit the flag instead.
    "pi": frozenset(("minimal", "low", "medium", "high", "xhigh", "max")),
    # cursor: no flag to carry the token, so this is only the COARSE gate (the
    # union over CURSOR_EFFORT_SLUGS). The per-family gate is
    # cursor_effort_model, which is what turns the token into a model slug.
    "cursor": frozenset(t for tokens in CURSOR_EFFORT_SLUGS.values()
                        for t in tokens),
}


def normalize_effort(value):
    """Return (token_or_None, error_envelope). None/blank → (None, None)."""
    if value is None:
        return None, None
    token = str(value).strip().lower()
    if not token:
        return None, None
    if token not in EFFORT_LEVELS:
        return None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": "effort must be one of %s; got %r"
                      % (", ".join(EFFORT_LEVELS), value),
        }
    return token, None


def effort_for_provider(provider, value):
    """Resolve effort for one routed provider. Unspecified stays (None, None)."""
    token, err = normalize_effort(value)
    if err or token is None:
        return token, err
    allowed = EFFORT_BY_PROVIDER.get(provider)
    if allowed is None:
        return None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": (
                "effort is supported on grok/claude/codex/pi/cursor; "
                "this model routes to %s" % provider
            ),
        }
    if token not in allowed:
        ordered = [t for t in EFFORT_LEVELS if t in allowed]
        return None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": "%s accepts %s; got %r"
                      % (provider, ", ".join(ordered), token),
        }
    return token, None


def cursor_effort_slug(base, token, account_profile=None):
    """Map (cursor base slug, public effort token) -> exploded legacy slug.

    Returns (slug, None) or (None, error_envelope). ``token`` None means the
    caller asked for no effort: the base is handed back untouched, so an
    omitted effort keeps today's argv byte for byte.
    """
    base = str(base or "").strip()
    if token is None:
        return base, None
    del account_profile
    tokens = CURSOR_EFFORT_SLUGS.get(base)
    if tokens is not None and token in tokens:
        return CURSOR_EFFORT_SLUG_OVERRIDES.get((base, token),
                                                "%s-%s" % (base, token)), None
    if tokens is None:
        return None, {
            "ok": False,
            "reason": "unsupported_effort",
            "detail": _cursor_effort_detail(base),
        }
    return None, {
        "ok": False,
        "reason": "unsupported_effort",
        "detail": "cursor model %s accepts %s; got %r"
                  % (base, ", ".join(tokens), token),
    }


def cursor_effort_variants(base):
    """Every catalog id this base can expand into, `-fast` twins included.

    `-none` is not a public effort token, so it never comes out of
    cursor_effort_slug -- but a caller CAN hand us `<base>-none`, and that id
    does carry a strength. It belongs in this set so the refusal says so.
    """
    ids = set()
    tokens = tuple(CURSOR_EFFORT_SLUGS.get(base, ())) + ("none",)
    default = CURSOR_DEFAULT_EFFORT.get(base)
    if default is not None and default not in tokens:
        tokens = tokens + (default,)
    for token in tokens:
        slug, err = cursor_effort_slug(base, token)
        if err or slug is None:
            slug = "%s-%s" % (base, token)
        ids.add(slug)
        ids.add("%s-fast" % slug)
    return ids


def _cursor_effort_detail(base):
    """Why this base cannot take an effort token."""
    if "[" in base:
        return ("cursor model %s already carries its parameters; the bracket "
                "form must spell every parameter itself, so pass the base slug "
                "with effort=, or drop effort" % base)
    for stem in CURSOR_EFFORT_SLUGS:
        if base in cursor_effort_variants(stem):
            return ("cursor model %s already encodes effort; pass cursor/%s "
                    "with effort=, or drop effort" % (base, stem))
    return ("cursor carries effort in the model slug; %r has no effort "
            "variants. Pass a base model that does (e.g. gpt-5.6-sol, "
            "claude-fable-5, cursor-grok-4.6), or drop effort" % base)

# Claude Code CLI (subscription-auth). Used when a `claude/<model>` name omits the model.
DEFAULT_CLAUDE_MODEL = "opus"

# Grok Build CLI (subscription-auth). Used when a grok model name is omitted.
DEFAULT_GROK_MODEL = "grok-4.6"

# Pi coding-agent CLI (subscription-auth). Used when a `pi/<model>` name omits the model.
# Must NOT be a grok* id: bare grok names stay on Grok Build CLI. Callers who
# want Grok-via-pi must spell `pi/xai/grok-4.6` explicitly.
DEFAULT_PI_MODEL = "anthropic/claude-sonnet-5"

# Cursor Agent CLI (subscription-auth). Used when a `cursor/<model>` name omits
# the model. Must NOT be a fable/sol/opus/grok id: those seats are named
# explicitly (cursor/fable, cursor/sol, …). composer-2.5 is the docs default
# for new Cursor CLI sessions (UNVERIFIED-LIVE on this host).
DEFAULT_CURSOR_MODEL = "composer-2.5"
# Current fable id. Alias lookup and both family seats read this one value.
FABLE_MODEL = "claude-fable-5-2"
# Exact (case-insensitive) remainder after stripping `cursor/`. Not a prefix.
CURSOR_MODEL_ALIASES = {
    "fable": FABLE_MODEL,
    "sol": SOL_CURSOR_MODEL,
}
# "Cursor Models" monthly pool stems (https://cursor.com/docs/account/pricing).
# Remaining cursor/ ids bill as "Other Models"; auto is a router, not a pool.
CURSOR_MODELS_POOL_STEMS = ("composer-", "cursor-grok-")
# Bare family names (consult lane). Tuple order is dispatch priority.
# Matched case-insensitively on the whole name after an `@effort` split;
# `cursor/sol` is cursor's own alias, not a family.
FAMILIES = {
    "sol": (("codex", SOL_CODEX_MODEL), ("cursor", SOL_CURSOR_MODEL)),
    "fable": (("claude", FABLE_MODEL), ("cursor", FABLE_MODEL)),
}
# Optional absolute path to the Cursor CLI. Never fall back to `agent`.
ENV_CURSOR_BIN = "OMNI_LLM_CURSOR_BIN"
# The Cursor CLI's own config-dir knob (it beats XDG_CONFIG_HOME and
# ~/.cursor). The consult lane points it at a fresh temp dir per run: without
# that, `cursor-agent -p --model X` REWRITES the human's ~/.cursor/cli-config.json
# default model, a bare base slug reads its parameters BACK from that file (so
# the effective strength depends on whatever was last picked interactively),
# and every consult prompt is persisted under ~/.cursor/chats. Credentials do
# not live in this directory (darwin keeps them in the Keychain), so an
# isolated run still rides the subscription.
CURSOR_CONFIG_DIR_ENV = "CURSOR_CONFIG_DIR"
# Escape hatch: set it to reuse one directory across runs (e.g. `~/.cursor` to
# get the pre-isolation behaviour back, or a dedicated omni profile that keeps
# its own MCP servers). Blank/unset = a temp dir per run.
ENV_CURSOR_CONFIG_DIR = "OMNI_LLM_CURSOR_CONFIG_DIR"

# Agent lane (`omni agent`, CLI-only — no MCP tool by design): delegated agentic
# runs inside a caller-supplied worktree.
# The agent lane EXECUTES; the consult lane (DEFAULT_MODEL) only advises, so the
# two lanes keep separate defaults. A silent default must land on a seat that is
# cleared for execution — hence a grok Build-CLI id, not a consult-only model.
# Explicit `-m <name>` is untouched: any routable model can still be asked for.
# Spelled BARE (no `grok/` prefix). `grok/<id>` also routes to grok and
# strips the prefix (`grok/grok-4.6` → `--model grok-4.6`); bare `grok-4.6`
# keeps today's route.
DEFAULT_AGENT_MODEL = "grok-4.6"
AGENT_TIMEOUT_S = 3600
# Grok job lane only: grok sometimes prints its final `type:"result"` line and
# then never exits. Once that line is out, omni kills the run after this many
# seconds without a single further stdout byte. OMNI_LLM_GROK_STALL_GRACE_S
# overrides it (finite float > 0; anything else falls back to this default).
GROK_STALL_GRACE_S = 720
ENV_GROK_STALL_GRACE = "OMNI_LLM_GROK_STALL_GRACE_S"
# Job-lane image staging (issue #18 P4): per-file and per-job total caps for
# validated-copy under <workdir>/.omni-attach/<job_id>/.
AGENT_IMAGE_MAX_BYTES = 5 * 1024 * 1024
AGENT_IMAGES_TOTAL_MAX_BYTES = 20 * 1024 * 1024
# Recursion guard: every CLI omni spawns (agy/codex/grok/claude/pi/cursor)
# gets OMNI_LLM_DEPTH=<depth+1> in env. The agent lane and the
# claude/pi/cursor consult backends refuse at depth >= 1; text consultation
# on other backends (generate/big-context) refuses at depth >= 2 (bounds any
# omni->CLI->omni chain to two hops). Cursor has no agent lane yet.
ENV_DEPTH = "OMNI_LLM_DEPTH"


def env_depth():
    """Current OMNI_LLM_DEPTH as a NON-NEGATIVE int; unset/garbage/negative -> 0.

    Clamping matters: a negative value would buy extra nesting hops through
    every >= guard downstream.
    """
    try:
        return max(0, int(os.environ.get(ENV_DEPTH, "0") or "0"))
    except ValueError:
        return 0

# CLI wall-clock discipline. `timeout_s` is the caller's whole budget: run_cli
# hard-kills the process group at exactly that, so the tool honours the number it
# advertises. A CLI that owns an internal deadline (agy --print-timeout) is given
# a slightly shorter one, so it can self-report a clean timeout before we kill it.
CLI_SELF_TIMEOUT_MARGIN_S = 5
# Watch-pane viewer max age. Backstop only — the viewer normally exits the moment
# the done sentinel appears; this just stops a wedged MCP leaving a pane forever.
WATCH_PANE_MAX_AGE_PAD_S = 60
WATCH_FINISH_DRAIN_S = 8

RATE_LIMIT_RETRIES = 1
RATE_LIMIT_BACKOFF_S = 2

# llm_big_context source_path: single-file and assembled-prompt hard caps (no raise-env).
SOURCE_PATH_MAX_BYTES = 12 * 1024 * 1024
SOURCE_PROMPT_MAX_BYTES = 12 * 1024 * 1024
# Colon- or comma-separated absolute directories. Unset/empty => source_path disabled.
ENV_SOURCE_ROOTS = "OMNI_LLM_SOURCE_ROOTS"

# llm_generate `files=` vision support (agy / codex / grok only).
# Extra keys in FILES_IMAGE_MIME / FILES_TEXT_MIME count against test_descriptions.py DECLARED_BUDGET_BYTES.
# Image extensions the grok inline ACP payload accepts, mapped to their MIME type.
# codex reads its own files (`-i <path>`) and is NOT limited to this table.
FILES_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Text extensions the grok inline ACP payload accepts, mapped to their MIME type.
# These ride the same `--prompt-json` envelope as images, but as `resource` blocks
# carrying the decoded UTF-8 body inline:
#   {"type":"resource","resource":{"uri":...,"mimeType":...,"text":...}}
# Verified against the grok CLI (2026-08-18): the body round-trips verbatim, so the
# older image-only filter was repo-side conservatism, not an upstream limit.
# Anything outside BOTH tables is still a LOUD unsupported_files refusal — no block
# type has been verified for pdf/office/other binaries.
FILES_TEXT_MIME = {
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".csv": "text/csv",
}
# Inline text is read server-side and embedded verbatim, so it costs BOTH model
# context and argv bytes. Per-file and total caps are deliberately far tighter than
# the job-lane attachment caps (5 MiB / 20 MiB), and both sit UNDER
# GROK_FILES_MAX_PAYLOAD_BYTES so each is actually reachable — the payload cap is
# the backstop for JSON-escape inflation and mixed image+text calls, not the first
# wall. Over either cap => LOUD `files_too_large`, never a silent truncation.
GROK_FILES_TEXT_MAX_BYTES = 256 * 1024
GROK_FILES_TEXT_TOTAL_MAX_BYTES = 512 * 1024
# grok ships prompt + base64 images as ONE inline `--prompt-json` argv value, so the
# whole payload rides the process argument list: cap it well under ARG_MAX (macOS
# 1MiB, and base64 inflates bytes ~4/3). Also the per-file read cap under the
# OMNI_LLM_SOURCE_ROOTS fence. Larger images belong on agy or codex.
GROK_FILES_MAX_PAYLOAD_BYTES = 600 * 1024
# Keep the legacy inline argv guard below ARG_MAX; large CLI requests instead
# use one owned input file under the shared assembled-prompt byte cap.
AGY_PROMPT_MAX_ARGV_BYTES = 600 * 1024
# Routing threshold only, not a vendor context guarantee. Large CLI requests
# travel in one owned read-only input file under SOURCE_PROMPT_MAX_BYTES.
AGY_PROMPT_FILE_THRESHOLD_BYTES = 32 * 1024

# First match wins. kind: "prefix" | "contains". Unmatched → "agy".
ROUTE_RULES = (
    {"kind": "prefix", "match": "ollama/", "provider": "ollama", "strip": 7,
     "default_api_base": "http://localhost:11434/v1"},
    {"kind": "prefix", "match": "lmstudio/", "provider": "lmstudio", "strip": 9,
     "default_api_base": "http://localhost:1234/v1"},
    # `codex/<model>` -> Codex CLI backend (subscription-auth). Must precede the
    # `contains codex` -> openai rule below (first match wins), so codex/* peels off
    # to the CLI while bare `codex` / `*codex*` API names still fall through to openai.
    {"kind": "prefix", "match": "codex/", "provider": "codex", "strip": 6},
    # `claude/<model>` -> Claude Code CLI backend (subscription-auth, `claude -p`).
    {"kind": "prefix", "match": "claude/", "provider": "claude", "strip": 7},
    # `pi/<model>` -> Pi coding-agent CLI (subscription-auth, `pi -p`).
    # Must precede grok* so `pi/xai/grok-4.6` stays on pi, not grok CLI.
    {"kind": "prefix", "match": "pi/", "provider": "pi", "strip": 3},
    # `cursor/<model>` -> Cursor Agent CLI. Must precede gpt-/grok* so
    # `cursor/gpt-5.6-sol` and `cursor/grok-4.6` stay on cursor.
    {"kind": "prefix", "match": "cursor/", "provider": "cursor", "strip": 7},
    # Explicit agy prefix wins over the generic gpt-* OpenAI API route.
    {"kind": "prefix", "match": "agy/", "provider": "agy", "strip": 4},
    {"kind": "prefix", "match": "gpt-", "provider": "openai", "strip": 0,
     "default_api_base": "https://api.openai.com/v1"},
    {"kind": "prefix", "match": "o1", "provider": "openai", "strip": 0,
     "default_api_base": "https://api.openai.com/v1"},
    {"kind": "prefix", "match": "o3", "provider": "openai", "strip": 0,
     "default_api_base": "https://api.openai.com/v1"},
    {"kind": "contains", "match": "codex", "provider": "openai", "strip": 0,
     "default_api_base": "https://api.openai.com/v1"},
    # `grok/<model>` strips the prefix so the CLI gets the bare id
    # (`grok/grok-4.6` → `--model grok-4.6`). Must precede the bare `grok`
    # rule, which matches the same names with strip 0.
    {"kind": "prefix", "match": "grok/", "provider": "grok", "strip": 5},
    {"kind": "prefix", "match": "grok", "provider": "grok", "strip": 0},
)

AUTH_MARKERS = ("login", "auth", "unauthor", "credential", "expired")
RATE_LIMIT_MARKERS = ("quota", "rate limit", "429", "too many requests", "usage limit")
GROK_EXTRA_AUTH_MARKERS = ("403",)
# Codex's refresh failures ("... your refresh token was already used /
# was revoked. Please log out and sign in again.") reach the classifier
# only as event text now that stderr no longer decides an evented run.
CODEX_EXTRA_AUTH_MARKERS = ("401", "refresh token", "sign in again")

HTTP_STATUS_REASONS = {
    401: "auth_error",
    403: "auth_error",
    429: "rate_limit",
}

try:
    from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    Agent = None            # type: ignore[assignment,misc]
    LocalAgentConfig = None  # type: ignore[assignment,misc]
    CapabilitiesConfig = None  # type: ignore[assignment,misc]
