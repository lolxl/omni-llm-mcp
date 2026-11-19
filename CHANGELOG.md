# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] — 2026-11-19

### Upgrade notes

- The Codex quota cache moves from one `quota.json` to one file per profile. The old file is not migrated: every Codex profile starts with a cold cache and probes its quota on demand (when probing is enabled), and the old file is deleted, best effort, after the first successful cache write. No manual step is needed.
- `omni profiles list` and `check` and the health probe now include an agy row. In JSON output, `list` and `check` include `{"road": "agy", "name": "default", "source": "default", "path": ..., "identity": {}, "state": "quota_unmonitored", "problems": [], "cooldown": ...}`, and `omni health --probe --all-profiles --json` includes `quota.agy.default` as `{"signal": "none", "cooldown": ...}` (`cooldown` is `healthy`, `probing` or `cooling`). The scheduled job runs with `--quiet`, so there agy only adds one to the `probed` count. Strict JSON consumers should expect these rows.

### Added

- `omni quota report-incident --json` records a hard rate limit for one login and account until a given time, as a `quota-incident/v1` marker under the state dir (written atomically, then one log line, best effort). Required flags: `--login`, `--kind hard_rate_limit`, `--until` (an ISO 8601 instant with a timezone) and `--json`; `--account` defaults to `default`. `omni quota check` then refuses that quota pool and account with `hard_rate_limit` for long and short tasks until the incident expires, and refuses as well when a marker is present but not exactly readable. A past `--until` writes an expired marker, which replaces and so clears an active incident. If the incident cannot be recorded (including an unreadable quota policy) it exits 73 with one fixed stderr line, and 74 if it was recorded but the result cannot be written; an unknown login or account exits 2. `omni contract` gains `quota_incident: "quota-incident/v1"` (refs #253).
- `omni quota check --json` returns a versioned `quota-decision/v1` decision on whether one login and account may still spend its quota. Required flags: `--login`, `--transport`, `--model`, `--task-class` (`long` or `short`), `--context` (`primary` or `fallback`) and `--json`; `--account` defaults to `default` and `--anthropic-snapshot` is optional (see `omni quota check --help`). Exit 0 is eligible or advisory, 10 ineligible, 2 usage error. `omni quota policy show --json` prints the shipped login/account to quota-pool map and its revision. The decision reads the seat-health ledger and, for the Codex pool, that profile's quota cache file through strict readers: a file that exists but cannot be read exactly (malformed, a symbolic link, not a regular file) refuses, while a missing ledger reads as empty and a missing cache file is no row. A Codex cache row that is absent, older than 60 s, more than 5 s in the future or bound to another Codex home is replaced by a live rate-limit RPC (5 s budget) when one succeeds; when it fails, or probing is disabled (`OMNI_LLM_QUOTA_PROBE=0`), the pool is refused as `telemetry_unavailable`. The Anthropic pool reads the caller-supplied snapshot, which refuses when missing, unreadable, malformed, bound to another account, older than 7200 s or more than 300 s in the future. All such refusals are `ineligible` for short tasks too. `advisory` (exit 0) is returned only for a short task whose only refusal is `quota_reserve`: a valid observation at or over the 90% threshold (9000 basis points after rounding), or a Codex window the vendor reports as reached even below it. An unmonitored pool (the `xai`, `google` and `cursor` logins) can be eligible as `primary` and is refused as `fallback`. A broken output pipe exits 74 with one fixed stderr line (a stdout closed before launch can still end with exit 1 and a traceback). The decision uses fixed fields and reason codes (a snapshot window label matching `[A-Za-z0-9_.-]{1,32}` passes through) and appends one audit line per decision, best effort: a failed log write does not change the decision. `omni contract` gains `quota: "quota-decision/v1"` (refs #253).
- `omni profiles list` and `check` list the agy login as one `default` row with `state: quota_unmonitored`, empty identity and a health cooldown; `--road agy` filters to it. `omni health --probe --all-profiles --json` reports `{"signal": "none", "cooldown": ...}` for agy (the scheduled `--quiet` run only counts it in `probed`). agy is not on the account ring and has no live check (#251).

### Changed

- The Codex quota cache is one file per profile, `<state_dir>/quota/<sha256 of the profile label>.json`, instead of one `quota.json`. A write replaces only its own profile's file, so a ledger-lock skip or a malformed file can no longer drop the other profiles' rows, and a malformed file is logged before it is replaced. `omni quota check` reads only the Codex profile its policy selects, so a malformed file of one profile refuses only that profile (`telemetry_malformed` for a file that cannot be read strictly; `malformed_telemetry` for a readable row with out-of-range values). The cache takes no ledger lock. The old `quota.json` is not migrated (rows live 60 seconds; a profile without a usable row probes again when needed): it is never read, and the first successful write of a profile file deletes it, best effort. A forced refresh, as `omni quota check` runs, first removes the profile's old row. If that fails (the unlink or the directory sync), the RPC does not run and `omni quota check` refuses with `telemetry_unavailable` (the internal detail is `quota_cache_drop_failed`). A failed directory sync when writing a row fails the cache write (logged as `quota_cache_write_failed`); the fresh RPC result is still used for that call (refs #254).

## [0.11.0] — 2026-11-12

### Breaking

- Catalog checks apply to claude, codex, grok, cursor, and agy before any vendor runs, so `omni models refresh` must run once per profile before first use, when that account is checked against its catalog. A Claude, Codex, or Grok run keeps the previous behavior, with no list check, when its profile home is not labeled, when its authentication environment differs from the profile's, or when Claude settings change the account or endpoint (for example a redirected endpoint or `apiKeyHelper`, #236) or cannot be read.
- When effort is omitted and the catalog row lists efforts, admission selects `medium` when that value is listed (a Codex omitted-effort alias is used instead when one is set and listed). That replaces Codex's previous `high` and the omitted effort flag for Claude and Grok. If that value is not listed, the request is refused until an effort is passed. Agy refuses a separate effort argument; put the effort in the `agy/<id>` model id.
- A catalog refusal ends a comma-separated chain, so later models in the chain do not run. Other skips (quota, health, login, and a Cursor timeout) still move on to the next model.
- The bundled model catalog, v1 records, `models import` and `models snapshot` are removed.
- The `agy-opus-<version>` shorthand is retired (use `agy/<id>`).

### Added

- `models refresh` writes a v2 catalog record for each selected provider and profile.
- `omni models check` admits the shipped defaults against that record and exits 2 when a default is refused. It reports `effort_required` for a default model and still exits 0; dispatch of that model then needs an explicit effort.
- `omni models admit` reports whether one profile's v2 record admits one model and effort, without calling a vendor or writing state.
- `<state>/aliases.json` retargets family seats, omitted codex effort and the grok image model; `omni models alias show|set|unset` reads and writes it.
- `python3 -m providers.egress_canary` runs one transport seam and reports whether the call was refused or the transport was reached.
- `pipx install git+https://github.com/lolxl/omni-llm-mcp` puts the `omni` CLI on `PATH`; the version is read from `providers/version.py`, and `server.py` stays out of the wheel (the server ships as `providers/mcp_server.py`).
- `omni mcp` runs the MCP stdio server from the installed CLI (`claude mcp add --scope user omni-llm -- omni mcp`); plugins and clones keep launching the `server.py` shim, and metrics log `entry: "mcp"` for `omni mcp`.

### Changed

- Off macOS, `models refresh` discovers catalogs under the `metadata-unfenced` posture and records that posture.
- fable routes to `claude-fable-5-2` through one constant.
- Codex sol and the Codex default route to `gpt-6.1-sol`; the Cursor sol route stays `gpt-5.6-sol`.

### Fixed

- The stated minimum Python is 3.9 (README, `docs/reference.md`, `pyproject.toml`); the code already required it.
- A Claude sign-in expiry warning on stderr no longer fails the call or benches the seat while the sign-in is still valid.
- `omni agent --expect-writes` counts a rewrite of an already-dirty file as a write, and `--out` is still written when a failed run has report text.

### Known limitation

- A Claude, Codex, or Grok run whose profile home is not labeled, whose authentication environment differs from the profile's, or whose Claude settings change the account or endpoint (for example a redirected endpoint or `apiKeyHelper`, #236) or cannot be read is not checked against a model list.

## [0.10.0] — first public release

This is the first public release. Earlier development happened in a private
repository; this entry summarizes what the release contains.

### Backends

- Google Antigravity CLI (`agy`), OpenAI Codex CLI (`codex/`), Claude Code CLI
  (`claude/`), xAI Grok Build CLI (`grok-*`), Cursor Agent CLI (`cursor/`), Pi
  coding-agent CLI (`pi/`), the OpenAI HTTP API, and local Ollama / LM Studio.

### Interfaces

- MCP stdio server with `llm_generate`, `llm_big_context`, `llm_image`, and
  `list_models`. Serves both the `2025-06-18` and the stateless `2026-07-28`
  protocol eras.
- `omni` CLI over the same provider core: `gen`, `big`, `img`, `models`,
  `health`, `profiles`, `agent` (synchronous and detached jobs), `artifact`,
  `dispatch-exec`, `--version`, and `contract`.
- Plugin manifests for Claude Code, Codex and Grok. The repository is a
  marketplace for all three, and each plugin runs the bundled `server.py` as the
  `omni-llm` MCP server.

### Routing and accounts

- Prefix-based model routing, opt-in fallback chains, and the `sol` / `fable`
  model families.
- Per-backend `effort`, validated against each backend's accepted values.
- Multiple accounts per backend, account rings with cooldown and quota awareness,
  and a local health ledger.
- Bundled and refreshable model catalogs.

### Safety

- macOS kernel Seatbelt sandbox for CLI backends, verified before each run;
  calls fail closed when the sandbox cannot be applied.
- Read-only consult lanes, a recursion guard, an allowlisted child environment,
  and secret redaction in errors and logs.
- Structured failure envelopes for every error path.
