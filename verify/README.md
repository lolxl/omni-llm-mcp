# verify/

Per-repo end-to-end drive for omni-llm-mcp. Doctor and feature map are deferred
(pstack plan v2.1 P2-a).

## drive

`verify/drive` launches this repo's MCP stdio server (`server.py`) under an
isolated env, sends one user-facing `tools/call` (`list_models`), and exits 0
or 1 on an explicit envelope oracle.

```
python3 verify/drive           # launch → call → oracle → cleanup
python3 verify/drive --cleanup # idempotent teardown of verify/.run
```

Isolation (the child does not inherit operator credentials or state):

- `PATH=/nonexistent` — no agy/grok/codex/claude/pi/cursor-agent from the login
- `OMNI_LLM_STATE_DIR=verify/.run/state` — not `~/.local/state/omni-llm`
- `OMNI_LLM_CODEX_SESSIONS_DIR` pointed at the same private tree
- `HOME`/`TMPDIR` retargeted at `verify/.run`
- inherited `*KEY*`/`*TOKEN*`/`*SECRET*`/`*PASSWORD*` env vars are not passed
- stdout/stderr printed by drive is token-masked (`sk-` / `xai-` / `Bearer`)

The `list_models` call is the same hermetic path
`tests.test_server.TestProtocol.test_list_models_cli_missing_is_loud` already pins:
blank PATH → LOUD `{ok: false, reason: "cli_missing"}` plus a `backends` map.
That is a real JSON-RPC round-trip through `server.py`; it does not call a
live model and does not read the operator's CLI sessions.

Oracle (exit 1 if any clause fails):

- `initialize` → `serverInfo.name == omni-llm-mcp`, protocol `2025-06-18`
- `tools/call list_models` → `ok is false`, `reason == cli_missing`,
  `backends` contains `agy` and `grok`, `isError is true`

Cleanup is idempotent: SIGTERM/SIGKILL the pid in `verify/.run/server.pid` if
still alive, then `rmtree verify/.run`. Running `--cleanup` twice is a no-op
success. Drive never git add/commit/push/merge.

To confirm the oracle can go red: change `EXPECT_REASON` in `verify/drive` and
re-run; restore the constant afterwards.
