# omni-llm-mcp Architecture Analysis and Improvement Proposal

- **Date**: 2026-07-11
- **Baseline commit**: `8e753a8` (v0.5.0)
- **Worktree**: `omni-llm-mcp-analysis-arch` / branch `analysis/architecture-review`
- **Test status**: `python3 -m unittest tests.test_server` → 19 passed

---

## 1. Project positioning (current state)

A zero-dependency MCP stdio server: it folds multiple LLM backends (`agy` / OpenAI-compatible / Grok CLI / Ollama / LM Studio) into three tools:

| Tool | Role |
|---|---|
| `llm_generate` | Ordinary prompt (may attach a `files` path string) |
| `llm_big_context` | instruction + large document → compose a prompt, then the same dispatch path |
| `list_models` | Only wraps `agy models` |

**Three design contracts** (`server.py` file header):

1. **POLICY-FREE PIPE** — do not inspect content
2. **LOUD STRUCTURED FAILURE** — `{ok:false, reason, detail}`
3. **ADVISORY + UNTRUSTED** — the caller is responsible for sanitize / ground

These three are the boundaries this project must hold; improvements must not erode them.

---

## 2. Execution flow (data flow)

```
stdin (JSON-RPC lines, limit 16MiB)
        │
        ▼
  main_loop() ── readline → json.loads → handle_message()
        │
        ├─ initialize / tools/list / ping → synchronous reply
        ├─ notifications/* → no reply
        └─ tools/call → asyncio.create_task(handle_tool_call)
                              │
                              ▼
                     Semaphore(MAX=5)
                              │
                              ▼
                       _call_tool(name, args)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        llm_generate   llm_big_context   list_models
              │               │               │
              └───────┬───────┘               ▼
                      ▼                 _list_models (agy only)
           _dispatch_llm_request
             (comma-split chain)
                      │
                      ▼
              _dispatch_single  ← bounded rate_limit retry
                      │
                      ▼
            _route_to_provider (prefix routing)
                      │
     ┌────────┬───────┼────────┬────────┐
     ▼        ▼       ▼        ▼        ▼
  ollama/  lmstudio/ gpt-/o1  grok*    (default)
  OpenAI-  OpenAI-   OpenAI-  grok CLI  _run_agy
  compat   compat    compat             ├─ SDK if HAS_SDK
                                        └─ else subprocess
                      │
                      ▼
            envelope JSON → content[0].text → stdout
```

### 2.1 Routing rules (prefix)

| model form | provider |
|---|---|
| `ollama/...` | OpenAI-compat → `localhost:11434/v1` |
| `lmstudio/...` | OpenAI-compat → `localhost:1234/v1` |
| `gpt-*` / contains `codex` / `o1*` / `o3*` | OpenAI API |
| `grok*` | Grok Build CLI (sandbox strict) |
| everything else | Google `agy` (SDK or CLI) |

### 2.2 Fallback chain

- Single model: call `_dispatch_single` directly; the envelope does **not** include `tried`
- Comma-separated: try left to right; attach `tried` only on success after a prior failure; if all fail, attach the full `tried`

### 2.3 Error classification (heuristic)

`auth_error` / `rate_limit` / `runtime_error` / `timeout` / `cli_missing` / `empty_response`  
Mostly substring matches on stderr/stdout (`login`, `429`, `quota`, …).

---

## 3. Current strengths (should be kept)

| Strength | Description |
|---|---|
| Zero dependencies | stdlib + optional `google.antigravity`; any project can attach it with zero install |
| LOUD contract is testable | Protocol + cli_missing + fallback `tried` have hermetic tests |
| stdio hardening | 16MiB limit fixes the whole session dying at 64KiB; a parse error does not kill the loop |
| Grok sandbox | `--sandbox strict` + throwaway cwd; there is a live canary |
| Bounded concurrency | `Semaphore(5)` + tool-call background tasks, so the stdin reader is not blocked |
| Opt-in fallback | The caller controls whether to failover; single-model behavior stays the same |
| Policy kept outside | No content review inside the server — the boundary with the caller project stays clear |

---

## 4. Issue inventory (by severity)

### P0 — Correctness / contract holes

#### P0-1 SDK path drops model / timeout / account_profile

`_run_agy`, when `HAS_SDK`, goes through `_run_agy_sdk`, but the SDK path:

- **Does not pass model** (always uses the Agent default)
- **Does not use timeout_s**
- **Ignores account_profile** (multi-account switching does not work)
- `files` only concatenates path strings into the prompt; they are not real multi-modal attachments

```
_run_agy_sdk(prompt, model, files=None)  # model is only echoed in the envelope; it does not drive the Agent
```

**Impact**: in an environment where the SDK is installed, the caller thinks it is selecting a model / switching profile, but the actual behavior differs; this conflicts with the "LOUD / predictable" contract.

**Recommendation**: the SDK path must accept and use `model` / `timeout` / profile; if it cannot, **fall back to subprocess and say so in detail**, or explicitly document that "SDK mode ignores these parameters" and return `mode: sdk` in the envelope so the caller can notice.

#### P0-2 Concurrent stdout interleaving risk

Multiple `handle_tool_call`s `await _send(...)` at the same time. stdio MCP requires **one complete JSON-RPC message per line**. `print`/`write`+`flush` does not guarantee cross-task atomicity for large payloads.

**Recommendation**: wrap `_send` in a single `asyncio.Lock`, or use a queue plus a single writer task.

#### P0-3 `files` capability is over-promised

The TOOL description says multi-modal / YouTube / PDF, but the implementation is:

```python
prompt += "\n\nFiles to analyze:\n" + "\n".join(files)
```

The OpenAI / Grok / Ollama paths do not consume files at all; the agy subprocess likewise only gets path strings.

**Recommendation**:

1. Rewrite the docs and schema to "paths are appended to the prompt text; real multi-modal only when on the SDK and supported"; or
2. Actually implement attachments: if the agy CLI has a flag, pass it; otherwise, for non-agy providers, return `reason: unsupported_files` (LOUD).

### P1 — Architectural maintainability

#### P1-1 One 600-line file, providers coupled

`server.py` simultaneously owns: JSON-RPC framing, tool schema, routing, four providers, fallback, and retry.

**Suggested module split** (still a zero-dep single entry point):

```
server.py              # main_loop + JSON-RPC only
providers/
  base.py              # envelope helpers, classify_error()
  agy.py
  grok.py
  openai_compat.py
routing.py             # prefix table + fallback chain
tools.py               # TOOLS schema + _call_tool
```

If the project insists on "single-file copy-deploy", use **logical sections + clear function boundaries** as a transition; there is no need to split into a package immediately.

#### P1-2 Routing table is hardcoded and easy to misclassify

```python
"codex" in model_lower   # any name containing codex goes to OpenAI
startswith("o1") / "o3"  # too broad
```

**Recommendation**: an explicit `PREFIX_TABLE` / `EXACT_ALIASES` dict; unmatched → agy; unit tests cover every rule.

#### P1-3 `list_models` only knows agy

The tool description already admits that Local/OpenAI are not listed, but the omni name does not match the behavior.

**Staged recommendation**:

1. Statically return "known backends + routing rules + agy dynamic list"
2. Optional liveness probes (`ollama list` / query OpenAI only when a key is in the environment)
3. Add a `backends: {agy, openai, grok, ollama}` status to the envelope

#### P1-4 `llm_big_context` is almost a thin wrapper

It only changes the timeout default and the prompt template. The tool name can stay (caller semantics); internally force a shared `_dispatch_llm_request`, and extract the template into a constant so it does not drift.

#### P1-5 `account_profile` semantics are inconsistent

| provider | profile behavior |
|---|---|
| OpenAI | `OPENAI_API_KEY_{PROFILE}` |
| Ollama/LM Studio | `OPENAI_API_BASE_{PROFILE}` |
| agy subprocess | `AGY_CONFIG_DIR_{PROFILE}` → `AGY_CONFIG_DIR` |
| agy SDK | **ignored** |
| grok | **accepts the parameter but does not use it** |

**Recommendation**: document the matrix; for unsupported providers, mark `profile_ignored` in log/detail; or return an explicit error (stricter).

### P2 — Robustness / observability

| ID | Problem | Recommendation |
|---|---|---|
| P2-1 | Error classification relies on substrings | Centralize `classify_cli_error(stdout, stderr, rc)`; table-driven + tests |
| P2-2 | Shutdown does not cancel in-flight tasks | After stdin EOF, `gather(*pending, return_exceptions=True)` or a timeout drain |
| P2-3 | No request-id / structured log | stderr JSON lines: `ts, tool, model, reason, latency_ms` (still POLICY-FREE) |
| P2-4 | OpenAI only recognizes 429 | 401→auth, 5xx→runtime, timeout separated |
| P2-5 | No CI setup | GitHub Action: `python3 -m unittest tests.test_server` (zero dependencies) |
| P2-6 | README install path is stale | Still says `gemini-agy-mcp-omni` |

### P3 — Test gaps

The existing 19 tests lean toward protocol + the grok branch. Missing:

- OpenAI-compat path (mock `urlopen`): success envelope + usage, 429→rate_limit, 401→auth
- Fallback across providers (mock multiple backends)
- rate_limit retry count and backoff (mock clock / sleep)
- `_send` concurrency does not interleave (stress)
- SDK vs subprocess selection and parameter passing
- `account_profile` env injection
- `list_models` successful parse (mock subprocess)

---

## 5. Improvement roadmap (suggested PR split)

### PR-A — Correctness hotfixes (small, first) ✅ implemented (v0.5.1)

1. Add an asyncio.Lock to `_send` (lazy `_get_write_lock`)
2. SDK path: `_build_agy_sdk_config` must be able to pass model; otherwise `sdk_unavailable` → subprocess. timeout uses `wait_for`; profile uses an `AGY_CONFIG_DIR_*` env mirror
3. `files` for non-agy: LOUD `unsupported_files` (the fallback chain may skip to the next one)
4. README path fix + honest schema/description

**Acceptance**: 28 tests OK (original 19 + 9 new tests).

### PR-B — Table-driven routing and errors ✅ implemented (v0.5.2)

1. `ROUTE_RULES` / `resolve_route` + snapshot tests
2. Shared `classify_cli_error` / `classify_exception_message`
3. OpenAI HTTP status → reason (401/403/429)

### PR-C — Observability + honest list_models ✅ implemented (v0.5.3)

1. Structured stderr log (`tool_call` / `rate_limit_retry` / drain)
2. `list_models` includes `backends` (on failure still LOUD + backends)
3. After stdin EOF, drain in-flight tools/call

### PR-D — Structural refactor ✅ implemented (v0.6.0)

Split `providers/*`; `server.py` keeps only framing.  
Entry point remains `python3 server.py` (same-directory package, zero install).

### Explicitly out of scope (honor the contract)

- No content policy / PII filter (left to each project's CLAUDE.md)
- Do not introduce hard dependencies on `openai` / `litellm` / `httpx` (optional extras are a separate discussion)
- Do not persist prompt / usage to disk inside the server

---

## Amendment — 2026-08-11 (issue #22 P1.1b metrics)

The original rule **"do not persist prompt / usage to disk inside the server"** (no prompt/usage persistence on disk) is **narrowed**, not repealed:

- **Still forbidden:** persisting prompt or output **CONTENT** (or any substring of user/model text) to disk from the consult path.
- **Now allowed:** content-free **operational metrics** — one JSONL record per tools/call and per CLI `omni gen/big/img` consult — written under `$OMNI_LLM_STATE_DIR/metrics.jsonl` (same state-dir resolve rules as the health ledger).

Record fields (no content): `ts`, `entry` (`mcp`|`cli`), `tool`, `seat` `{provider, model, account_profile}`, `attempts` (real provider invocations including rate-limit retries; excludes `skipped_cooldown`), `fallback_count` (seats actually called after the first real seat), `request_bytes`, `result_bytes`, `latency_ms`, `usage` (normalized envelope usage of the **winning** seat only), `ok`, optional `reason`.

Write discipline: `O_APPEND` + single `os.write` per line; state dir mode `0700`, metrics file `0600`; when `metrics.jsonl` exceeds 5 MiB it is rotated to `metrics.jsonl.1` (single generation). `OMNI_LLM_METRICS=0` disables all metrics writes. A metrics write failure must not alter the model result — the envelope gains a LOUD local `metrics_error` field and processing continues.

Envelope `usage` remains winning-seat only; per-attempt usage may appear only inside the metrics record’s attempts detail when already present on the provider result.

---

## 6. Target architecture (after evolution)

```
                    ┌─────────────────────┐
   MCP Client ──────│  framing (stdio)    │  lock-protected writer
                    │  tools schema       │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  dispatch           │  chain + rate_limit retry
                    │  route(model)→Prov  │  pure function table
                    └─────────┬───────────┘
          ┌───────────┬───────┴────┬────────────┐
          ▼           ▼            ▼            ▼
       AgyProv    GrokProv   OpenAICompat    (future)
       SDK|CLI    sandboxed  ollama/lm/api
          │           │            │
          └───────────┴───────┬────┘
                              ▼
                     Envelope {ok, text|reason, ...}
```

Provider interface (conceptual):

```python
async def generate(prompt, model, timeout_s, *, files, account_profile) -> dict:
    """Return typed envelope. Never raise to caller for expected failures."""
```

---

## 7. Priority in one sentence

> **First fix "what we say does not match what we do" (SDK parameters, files, stdout lock), then table-driven routing/errors, and only then split files.**

Zero dependencies + LOUD + POLICY-FREE are the moat; any refactor uses not breaking these three as the merge gate.
