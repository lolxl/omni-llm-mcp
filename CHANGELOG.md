# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
