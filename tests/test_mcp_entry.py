#!/usr/bin/env python3
# no-opencc
"""`omni mcp` and `server.py` are two entries to one server (providers/mcp_server.py).

Pins: `server` and `providers.mcp_server` are the same module object; both
entries speak JSON-RPC only on stdout and exit cleanly on stdin EOF; metrics
log `entry: "mcp"` for both and `entry: "cli"` for an ordinary CLI call.
Everything runs in fresh subprocesses; metrics state goes to a temp dir.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(HERE, "server.py")
OMNI_CLI = os.path.join(HERE, "omni_cli.py")

ENTRIES = {
    "omni mcp": [sys.executable, OMNI_CLI, "mcp"],
    "server.py": [sys.executable, SERVER],
}

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
# An explicit model that routes to a CLI-subprocess backend: `claude/` resolves
# to provider "claude" (providers/config.py ROUTE_RULES), whose backend execs
# `claude -p` and returns cli_missing when the binary is not on PATH
# (providers/claude.py). The default model may route to the agy SDK path.
MODEL = "claude/haiku"
CALL = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "llm_generate",
                   "arguments": {"prompt": "hi", "model": MODEL}}}

# Vendor config homes the providers read from the env.
_VENDOR_VARS = ("CODEX_HOME", "AGY_CONFIG_DIR", "CLAUDE_CONFIG_DIR", "TMPDIR")


def _env(state_dir):
    """Hermetic child env: no OMNI_* setting, vendor key or config home of the
    developer's shell leaks in (OMNI_LLM_METRICS=0, OMNI_LLM_DEPTH=99 ...)."""
    # Keep the offline runner's child-arming variables. Dropping
    # OMNI_OFFLINE_HOME skips install_in_child, and the child then refuses
    # no_catalog before cli_missing. OMNI_OFFLINE_REAL_ADMISSION stays so a
    # module that seeds v2 records still reaches the real gate.
    env = {k: v for k, v in os.environ.items()
           if k.startswith("OMNI_OFFLINE_") or not (
               k.startswith("OMNI_") or "_API_KEY" in k
               or k.endswith("_TOKEN") or k in _VENDOR_VARS)}
    env["PATH"] = "/nonexistent"          # backends fail fast with cli_missing
    env["OMNI_LLM_STATE_DIR"] = state_dir
    return env


def _talk(cmd, messages, state_dir):
    inp = "".join(json.dumps(m) + "\n" for m in messages)
    proc = subprocess.run(cmd, input=inp, text=True, capture_output=True,
                          env=_env(state_dir), cwd=HERE, timeout=60)
    return proc


def _metrics_entries(state_dir):
    path = os.path.join(state_dir, "metrics.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["entry"] for line in fh if line.strip()]


class IdentityTests(unittest.TestCase):
    def _run(self, code):
        with tempfile.TemporaryDirectory() as state:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                  text=True, cwd=HERE, timeout=60,
                                  env=_env(state))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "same")

    def test_server_first(self):
        self._run("import server, providers.mcp_server as m\n"
                  "print('same' if server is m else 'different')")

    def test_package_first(self):
        self._run("import providers.mcp_server as m, server\n"
                  "print('same' if server is m else 'different')")

    def test_patching_the_shim_patches_the_implementation(self):
        self._run("import server, providers.mcp_server as m\n"
                  "server._send = lambda obj: None\n"
                  "print('same' if m._send is server._send else 'different')")


class StdioTests(unittest.TestCase):
    def test_initialize_and_tools_list_over_stdio(self):
        for label, cmd in ENTRIES.items():
            with self.subTest(entry=label), tempfile.TemporaryDirectory() as state:
                proc = _talk(cmd, [INIT, LIST], state)
                # Clean exit on stdin EOF, and nothing but JSON-RPC on stdout.
                self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                msgs = [json.loads(ln) for ln in lines]
                self.assertEqual(len(msgs), 2, proc.stdout)
                for m in msgs:
                    self.assertEqual(m["jsonrpc"], "2.0")
                    self.assertNotIn("error", m)
                by_id = {m["id"]: m for m in msgs}
                self.assertEqual(by_id[1]["result"]["serverInfo"]["name"],
                                 "omni-llm-mcp")
                self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-06-18")
                tools = {t["name"] for t in by_id[2]["result"]["tools"]}
                self.assertEqual(tools, {"llm_generate", "llm_big_context",
                                         "llm_image", "list_models"})

    def test_omni_mcp_rejects_extra_arguments(self):
        proc = subprocess.run([sys.executable, OMNI_CLI, "mcp", "--bogus"],
                              capture_output=True, text=True, cwd=HERE,
                              timeout=60, stdin=subprocess.DEVNULL,
                              env=_env(tempfile.gettempdir()))
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")


class MetricsEntryTests(unittest.TestCase):
    def test_mcp_entries_log_mcp(self):
        for label, cmd in ENTRIES.items():
            with self.subTest(entry=label), tempfile.TemporaryDirectory() as state:
                proc = _talk(cmd, [INIT, CALL], state)
                self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
                self.assertEqual(_metrics_entries(state), ["mcp"])

    def test_plain_cli_logs_cli(self):
        with tempfile.TemporaryDirectory() as state:
            proc = subprocess.run([sys.executable, OMNI_CLI, "gen", "hi", "-m", MODEL],
                                  capture_output=True, text=True, env=_env(state),
                                  cwd=HERE, timeout=60, stdin=subprocess.DEVNULL)
            self.assertEqual(json.loads(proc.stdout)["reason"], "cli_missing")
            self.assertEqual(_metrics_entries(state), ["cli"])

    def test_detect_entry_override_and_reset(self):
        from providers import metrics
        before = metrics.detect_entry()
        try:
            metrics.set_entry("mcp")
            self.assertEqual(metrics.detect_entry(), "mcp")
            metrics.set_entry("cli")
            self.assertEqual(metrics.detect_entry(), "cli")
        finally:
            metrics.set_entry(None)
        self.assertEqual(metrics.detect_entry(), before)

    def test_mcp_main_restores_the_entry_override(self):
        from providers import mcp_server, metrics
        seen = []

        async def fake_loop():
            seen.append(metrics.detect_entry())

        before_override = metrics._entry_override
        before = metrics.detect_entry()
        orig = mcp_server.main_loop
        mcp_server.main_loop = fake_loop
        try:
            mcp_server.main()
            after_override = metrics._entry_override
            after = metrics.detect_entry()
        finally:
            mcp_server.main_loop = orig
            metrics._entry_override = before_override
        self.assertEqual(seen, ["mcp"])           # pinned while the loop runs
        self.assertEqual(after, before)
        self.assertEqual(after_override, before_override)


if __name__ == "__main__":
    unittest.main()
