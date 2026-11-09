# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
