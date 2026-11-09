# omni-llm-mcp

A zero-dependency **MCP stdio server** and **`omni` CLI** that put many LLM backends
behind one interface, so any MCP client (for example Claude Code) can consult other
models with a single tool call.

Supported backends:

| Backend | Model prefix | Auth |
|---|---|---|
| Google Antigravity CLI (`agy`) | default (no prefix), e.g. `Gemini 3.5 Flash` | your `agy` login (OAuth) |
| OpenAI Codex CLI | `codex/<model>` | ChatGPT / Codex subscription (`codex login`) |
| Claude Code CLI | `claude/<model>` | Claude subscription (`claude auth login`) |
| xAI Grok Build CLI | `grok-*` | SuperGrok / X Premium subscription (`grok login`) |
| Cursor Agent CLI | `cursor/<model>` | Cursor subscription or `CURSOR_API_KEY` |
| Pi coding-agent CLI | `pi/<provider>/<model>` | your local `pi` login |
| OpenAI HTTP API | e.g. `gpt-4o` | `OPENAI_API_KEY` |
| Ollama / LM Studio | `ollama/<model>`, `lmstudio/<model>` | none (local) |

## Why

- **Zero dependencies.** Pure standard library on the system `python3` (3.7+).
  No virtualenv, no `pip install`.
- **Ride your subscriptions.** CLI backends use the vendor CLI's own login, so no
  metered API key is needed for most of them.
- **Text in, text out.** The server does not inspect or police content. What may be
  sent, and how replies are used, is each calling project's policy.
- **Loud failures.** Every failure returns a structured `{ok: false, reason, detail}`
  envelope. Nothing fails silently.
- **Advisory output.** Replies are untrusted input; callers should validate them.

## Install

You need `python3` 3.7 or later. Log in once, in a real terminal, to each vendor CLI
you want to use.

### As a plugin

The repository is a plugin marketplace for Claude Code, Codex, and Grok. Each one
runs the bundled `server.py` as the `omni-llm` MCP server.

```bash
# Claude Code
claude plugin marketplace add lolxl/omni-llm-mcp
claude plugin install omni-llm@omni-llm

# Codex
codex plugin marketplace add lolxl/omni-llm-mcp
codex plugin add omni-llm@omni-llm

# Grok
grok plugin install https://github.com/lolxl/omni-llm-mcp --trust
```

Restart the client after installing. The four tools then appear under the
`omni-llm` MCP server. Ask for them in plain language ("ask grok-4.6 to review this
diff") or name the tool directly (`llm_generate` with `model: "codex/gpt-5.6-sol"`).

If you already registered the server by hand, remove that entry first so the tools
do not appear twice. For example, in Claude Code: `claude mcp remove omni-llm`.

To check that it works:

| Harness | Command |
|---|---|
| Claude Code | `claude mcp list` should show `plugin:omni-llm:omni-llm ... ✔ Connected` |
| Codex | `codex mcp list` should show `omni-llm` as `enabled` |
| Grok | `grok mcp doctor` should report `omni-llm` healthy with 4 tools |

To update or remove it:

| Harness | Update | Remove |
|---|---|---|
| Claude Code | `claude plugin marketplace update omni-llm`, then `claude plugin update omni-llm@omni-llm` | `claude plugin uninstall omni-llm@omni-llm` |
| Codex | `codex plugin marketplace upgrade omni-llm`, then `codex plugin add omni-llm@omni-llm` | `codex plugin remove omni-llm@omni-llm` |
| Grok | `grok plugin update omni-llm` | `grok plugin uninstall omni-llm` |

**Configuration.** The plugin server reads the same environment variables as a
manual install: vendor logins, `OMNI_LLM_*` settings, and account profiles (see
[docs/reference.md](docs/reference.md)). Set them in the environment the client
starts from. Each client decides which of its environment variables reach an MCP
server. If a setting has no effect under a plugin install, pass it through that
client's MCP server environment settings.

### From a clone

```bash
git clone https://github.com/lolxl/omni-llm-mcp ~/omni-llm-mcp

# Register the MCP server with Claude Code (user scope = all projects)
claude mcp add --scope user omni-llm -- python3 "$HOME/omni-llm-mcp/server.py"
```

### The `omni` CLI

The plugins do not install the CLI. Link it from a clone:

```bash
ln -sf "$HOME/omni-llm-mcp/omni_cli.py" ~/.local/bin/omni
```

Restart the MCP client after updating; a running server does not reload.

## MCP tools

| Tool | Purpose |
|---|---|
| `llm_generate(prompt, model?, account_profile?, effort?, files?, ...)` | General Q&A and multimodal analysis |
| `llm_big_context(instruction, text \| source_path, effort?, ...)` | Digest a large inline text or a local file |
| `llm_image(prompt, backend?, source_image?, ...)` | Image generation or editing |
| `list_models()` | Available models, backend routing map, and seat health |

Success returns `{ok: true, text, model, usage?, seat?}`. `usage` uses one
normalized shape for every backend; fields a backend cannot report are `null`,
never `0`.

### Model selection

- **Prefix routing.** The model name picks the backend (see the table above).
- **Fallback chains (opt-in).** A comma-separated list such as
  `"codex/gpt-5.6-sol,grok-4.6,ollama/llama3"` is tried left to right. A single model
  never falls back.
- **Families.** The bare names `sol` and `fable` expand to an ordered list of seats
  across backends (for example `sol` → `codex/gpt-5.6-sol`, then
  `cursor/gpt-5.6-sol`). Override the order with `OMNI_LLM_FAMILY_<NAME>`.
- **Effort.** `effort` accepts `low` / `medium` / `high` / `xhigh`; some backends
  also accept `minimal` or `max`. Unsupported values are rejected, not ignored.

### Image backends

- `backend='cli'` (default): edits `source_image` through the local `grok` CLI.
  Uses your subscription; no API key.
- `backend='rest'`: text-to-image through the xAI REST API. Needs `XAI_API_KEY`
  and is billed per image.

## CLI (`omni`)

The CLI is a second entry point to the same provider core, with the same envelopes
and failure contract. It is meant for shells, scripts, and schedulers.

| Command | What it does |
|---|---|
| `omni gen "prompt" -m grok-4.6 --effort high` | Same as `llm_generate` |
| `omni big "instruction" -s doc.txt` | Same as `llm_big_context` |
| `omni img "edit" --source-image a.png` | Same as `llm_image` |
| `omni models` | List models; `status` / `refresh` / `check` maintain the catalogs |
| `omni health [--probe]` | Show cooldowns and quota; flag or clear seats |
| `omni profiles list \| check \| add` | Manage multiple accounts per backend |
| `omni agent "task" --workdir <dir>` | Hand a whole task to an agentic CLI (CLI only) |
| `omni agent start "task" --workdir <dir>` | Same, as a detached job you can poll later |
| `omni artifact read <id>` | Read large result bodies spilled to disk |
| `omni --version`, `omni contract` | Release and interface-contract versions |

## Accounts and health

- **Multiple accounts.** `account_profile=<name>` selects a separate login, for
  example `CODEX_HOME_<NAME>`, `CLAUDE_CONFIG_DIR_<NAME>`, or `GROK_HOME_<NAME>`.
  A missing or invalid profile is refused, never silently replaced.
- **Account rings.** `OMNI_LLM_PROFILES_<ROAD>=a,b,c` rotates through accounts and
  skips ones that are cooling down or out of quota. A failed request is not retried
  on another account; the next request picks a new one.
  See [docs/account-failover.md](docs/account-failover.md).
- **Health ledger.** Rate limits, quota exhaustion, and crash streaks put a seat on
  cooldown. The ledger lives in `~/.local/state/omni-llm/` (override with
  `OMNI_LLM_STATE_DIR`).

## Security model

- On macOS, CLI backends run inside a kernel Seatbelt sandbox: the home directory is
  read-walled, and writes are limited to the working directory plus named vendor
  state. If the sandbox cannot be applied and verified, the call fails instead of
  running unsandboxed.
- Consult calls disable write, exec, and network tools wherever the vendor CLI
  allows it, and a recursion guard (`OMNI_LLM_DEPTH`) stops loops such as
  claude → omni → claude.
- Child processes get an allowlisted environment; known secret patterns are redacted
  from errors and logs.
- Outside macOS, the vendor sandboxes are the only protection, and some of them do
  not block reads of the whole machine. Send only data you would be comfortable
  sharing with that vendor.

## Tests

```bash
python3 -m unittest tests.test_server    # protocol and failure contract; no network needed
```

The full suite (`python3 -m unittest discover -p 'test_*.py'`) is much larger and
takes well over ten minutes.

Manual smoke test over stdio:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"llm_generate","arguments":{"prompt":"say hi","model":"ollama/llama3"}}}' \
  | python3 server.py
```

The server speaks both MCP protocol eras: `2025-06-18` (with `initialize`) and the
stateless `2026-07-28`.

## More documentation

- [docs/reference.md](docs/reference.md): full reference for every option,
  environment variable, and backend detail
- [docs/account-failover.md](docs/account-failover.md): account selection and failover
- [docs/model-catalogs.md](docs/model-catalogs.md): model catalog maintenance
- [docs/session-router.md](docs/session-router.md): session router
- [CHANGELOG.md](CHANGELOG.md): release history

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
