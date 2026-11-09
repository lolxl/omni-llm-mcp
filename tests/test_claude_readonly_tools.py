#!/usr/bin/env python3
# no-opencc
"""Hermetic tests: read-only Claude lanes run with a built-in tool allowlist (#137).

The agent/job `--read-only` branch and the consult lane must both carry
`--tools Read,Grep,Glob` and a `--disallowedTools` value that contains every
consult-lane denial. The write branch must not gain `--tools`. `run_cli` is
faked at the module seam, so no claude process is ever spawned.
"""
import asyncio
import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from providers import agent_lane  # noqa: E402
from providers import claude as claude_mod  # noqa: E402
from providers import health  # noqa: E402
import tests.test_provider_accounts as tpa  # noqa: E402

_REAL_WHICH = shutil.which
CONSULT_DENIALS = ("Bash", "Write", "Edit", "NotebookEdit",
                   "WebFetch", "WebSearch", "Task")
# "Agent" is the current name of "Task"; the second layer denies both.
READ_ONLY_DENIALS = CONSULT_DENIALS + ("Agent",)
READ_ONLY_TOOLS = "Read,Grep,Glob"


class FakeRunCli:
    def __init__(self):
        self.calls = []

    async def __call__(self, argv=None, **kwargs):
        self.calls.append((list(argv or []), dict(kwargs)))
        return {"returncode": 0, "stdout": "OK", "stderr": "",
                "timed_out": False, "signal": None}

    @property
    def argv(self):
        return self.calls[0][0]


def _fake_which(name, *a, **kw):
    base = os.path.basename(str(name))
    if base == "claude":
        return "/fake/bin/claude"
    return _REAL_WHICH(name, *a, **kw)


def _flag_value(argv, flag):
    argv = list(argv)
    if flag not in argv:
        return None
    i = argv.index(flag)
    return argv[i + 1] if i + 1 < len(argv) else None


@contextlib.contextmanager
def _no_depth():
    with mock.patch.dict(os.environ, {}):
        os.environ.pop("OMNI_LLM_DEPTH", None)
        yield


class ReadOnlyToolsTest(unittest.TestCase):
    def _workdir(self):
        d = os.path.realpath(tempfile.mkdtemp(prefix="omni_ro_tools_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def _job_argv(self, **kwargs):
        log_path = os.path.join(self._workdir(), "stream.raw.log")
        return self._agent_argv(job_lane=True, log_path=log_path, **kwargs)

    def _assert_read_only_contract(self, argv):
        self.assertEqual(_flag_value(argv, "--tools"), READ_ONLY_TOOLS)
        items = set((_flag_value(argv, "--disallowedTools") or "").split(","))
        for tool in READ_ONLY_DENIALS:
            self.assertIn(tool, items, "missing %s" % tool)
        self.assertIn("--setting-sources", argv)
        self.assertEqual(_flag_value(argv, "--setting-sources"), "")

    def _agent_argv(self, **kwargs):
        fake = FakeRunCli()
        with _no_depth(), \
                mock.patch.object(agent_lane, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            env = asyncio.run(agent_lane.run_agent(
                task="DO THE THING", model="claude/sonnet",
                workdir=self._workdir(), timeout_s=30, **kwargs))
        self.assertEqual(len(fake.calls), 1, env)
        return fake.argv

    def _consult_argv(self):
        fake = FakeRunCli()
        with _no_depth(), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            asyncio.run(claude_mod.run_claude_subprocess("hello", "sonnet", 30))
        self.assertEqual(len(fake.calls), 1)
        return fake.argv

    def test_read_only_agent_lane_carries_the_tool_allowlist(self):
        argv = self._agent_argv(read_only=True)
        self.assertIn("--tools", argv)
        self.assertEqual(_flag_value(argv, "--tools"), READ_ONLY_TOOLS)

    def test_read_only_agent_lane_disallows_every_consult_denial(self):
        argv = self._agent_argv(read_only=True)
        value = _flag_value(argv, "--disallowedTools") or ""
        items = set(value.split(","))
        for tool in CONSULT_DENIALS:
            self.assertIn(tool, items, "missing %s in %r" % (tool, value))

    def test_consult_lane_carries_the_same_tool_allowlist(self):
        argv = self._consult_argv()
        self.assertEqual(_flag_value(argv, "--tools"), READ_ONLY_TOOLS)
        items = set((_flag_value(argv, "--disallowedTools") or "").split(","))
        for tool in CONSULT_DENIALS:
            self.assertIn(tool, items)

    def test_read_only_loads_no_setting_sources_write_lane_keeps_project(self):
        ro = self._agent_argv(read_only=True)
        i = ro.index("--setting-sources")
        self.assertEqual(ro[i + 1], "")
        self.assertEqual(ro.count("--setting-sources"), 1)
        rw = self._agent_argv()
        self.assertEqual(_flag_value(rw, "--setting-sources"), "project")

    def test_read_only_job_lane_carries_the_full_contract(self):
        self._assert_read_only_contract(self._job_argv(read_only=True))

    def test_read_only_resume_carries_the_full_contract(self):
        argv = self._agent_argv(read_only=True, resume_session="sess-123")
        self.assertIn("-r", argv)
        self._assert_read_only_contract(argv)

    def test_read_only_job_resume_carries_the_full_contract(self):
        argv = self._job_argv(read_only=True, resume_session="sess-123")
        self.assertIn("-r", argv)
        self._assert_read_only_contract(argv)

    def test_variadic_flags_are_followed_by_a_flag_or_end_of_argv(self):
        # --tools / --disallowedTools are variadic in the claude CLI: a bare
        # positional token after them would be swallowed as one more tool name.
        for argv in (self._agent_argv(read_only=True),
                     self._job_argv(read_only=True),
                     self._consult_argv()):
            for flag in ("--tools", "--disallowedTools"):
                i = argv.index(flag)
                after = argv[i + 2:i + 3]
                self.assertTrue(
                    not after or after[0].startswith("-"),
                    "%s value followed by %r in %r" % (flag, after, argv[-8:]))

    def test_agent_is_in_the_disallow_list(self):
        self.assertIn("Agent", claude_mod.READ_ONLY_DISALLOWED.split(","))
        argv = self._agent_argv(read_only=True)
        self.assertIn("Agent", (_flag_value(argv, "--disallowedTools") or "").split(","))
        self.assertIn("Agent", (_flag_value(self._consult_argv(), "--disallowedTools") or "").split(","))

    def test_write_branch_does_not_gain_a_tools_flag(self):
        argv = self._agent_argv()
        self.assertNotIn("--tools", argv)
        self.assertIn("--dangerously-skip-permissions", argv)


class ReadOnlyApiAndEnvTest(tpa.ApiFixture):
    """API-key account and plugin-selector env, on the shared fake-binary fixture."""

    def test_api_key_read_only_run_carries_empty_sources_and_settings_file(self):
        health.flag(self.model, "a")
        result = self.agent(read_only=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["account_kind"], "api")
        argv = self.rows()[0]["argv"]
        self.assertEqual(_flag_value(argv, "--setting-sources"), "")
        self.assertIn("--settings", argv)
        self.assertEqual(_flag_value(argv, "--tools"), READ_ONLY_TOOLS)
        settings = self.rows()[0]["settings"]
        self.assertEqual(argv[argv.index("--settings") + 1], settings["path"])

    def test_plugin_dirs_env_is_scrubbed_from_read_only_children_only(self):
        os.environ["CLAUDE_CODE_PLUGIN_DIRS"] = "/hostile/plugins"
        self.assertTrue(self.agent(read_only=True)["ok"])
        self.assertTrue(self.consult()["ok"])
        for row in self.rows():
            self.assertNotIn("CLAUDE_CODE_PLUGIN_DIRS", row["env"], row)
        self.assertEqual(len(self.rows()), 2)
        self.capture.unlink()
        self.assertTrue(self.agent()["ok"])
        self.assertEqual(self.rows()[0]["env"].get("CLAUDE_CODE_PLUGIN_DIRS"),
                         "/hostile/plugins")

    def test_plugin_dirs_env_is_scrubbed_on_the_read_only_job_lane(self):
        os.environ["CLAUDE_CODE_PLUGIN_DIRS"] = "/hostile/plugins"
        _origin, result = self.run_job(read_only=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(_flag_value(self.rows()[0]["argv"], "--tools"), READ_ONLY_TOOLS)
        self.assertNotIn("CLAUDE_CODE_PLUGIN_DIRS", self.rows()[0]["env"])


if __name__ == "__main__":
    unittest.main()
