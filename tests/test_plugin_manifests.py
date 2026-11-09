"""Plugin manifests for Claude Code, Codex and Grok stay consistent with the release.

The manifests repeat the release version because each harness reads its own file;
this test pins every copy to ``providers.version.__version__`` and checks that each
harness's MCP entry still launches the repository's ``server.py``.
"""
import json
import os
import unittest

from providers.version import __version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


class TestPluginManifests(unittest.TestCase):
    def test_versions_match_release(self):
        for rel in (".claude-plugin/plugin.json", ".codex-plugin/plugin.json"):
            self.assertEqual(_load(rel)["version"], __version__, rel)
        for rel in (".claude-plugin/marketplace.json", ".grok-plugin/marketplace.json"):
            for entry in _load(rel)["plugins"]:
                self.assertEqual(entry["version"], __version__, rel)

    def test_one_plugin_name_everywhere(self):
        names = {
            _load(".claude-plugin/plugin.json")["name"],
            _load(".codex-plugin/plugin.json")["name"],
        }
        for rel in (".claude-plugin/marketplace.json", ".grok-plugin/marketplace.json",
                    ".agents/plugins/marketplace.json"):
            names.update(entry["name"] for entry in _load(rel)["plugins"])
        self.assertEqual(names, {"omni-llm"})

    def test_marketplaces_point_at_repo_root(self):
        for rel in (".claude-plugin/marketplace.json", ".grok-plugin/marketplace.json"):
            self.assertEqual([e["source"] for e in _load(rel)["plugins"]], ["./"], rel)
        codex = _load(".agents/plugins/marketplace.json")["plugins"]
        self.assertEqual([e["source"] for e in codex], [{"source": "local", "path": "."}])

    def test_claude_and_grok_launch_server_py(self):
        server = _load(".claude-plugin/plugin.json")["mcpServers"]["omni-llm"]
        self.assertEqual(server["command"], "python3")
        self.assertEqual(server["args"], ["${CLAUDE_PLUGIN_ROOT}/server.py"])

    def test_codex_launches_server_py(self):
        manifest = _load(".codex-plugin/plugin.json")
        rel = manifest["mcpServers"]
        self.assertTrue(rel.startswith("./"), rel)
        server = _load(rel[2:])["mcpServers"]["omni-llm"]
        self.assertEqual((server["command"], server["args"], server["cwd"]),
                         ("python3", ["./server.py"], "."))
        self.assertTrue(os.path.isfile(os.path.join(ROOT, server["cwd"], server["args"][0])))

    def test_no_root_mcp_json(self):
        # A root .mcp.json would also register the server as a project-scoped
        # server for anyone who opens a clone in Claude Code.
        self.assertFalse(os.path.exists(os.path.join(ROOT, ".mcp.json")))


if __name__ == "__main__":
    unittest.main()
