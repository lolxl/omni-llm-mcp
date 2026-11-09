#!/usr/bin/env python3
# no-opencc
"""Unit tests for successor attest — vendor records only, no processes."""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers.attest import read_successor_attest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX = os.path.join(_REPO, "tests", "fixtures", "successor")

CLAUDE_SID = "bcdfc3dd-04de-4055-b559-bd67070a1869"
HAIKU_SID = "83ce3990-5811-4569-89df-f4a60549256f"
CODEX_SID = "01a0a2ea-2b61-7c60-8ea8-feb3e90d818f"
GROK_SID = "b6e7e486-5940-45eb-972f-9ccc40374f0b"


class ReadSuccessorAttestTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="omni-attest-home-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def _claude_path(self, session_id):
        dest_dir = os.path.join(self.home, "projects", "probe")
        os.makedirs(dest_dir, exist_ok=True)
        return os.path.join(dest_dir, "%s.jsonl" % session_id)

    def test_claude_fixture_reads_top_level_effort_and_message_model(self):
        dest = self._claude_path(CLAUDE_SID)
        shutil.copy(
            os.path.join(_FIX, "claude-transcript-first-assistant.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "claude", CLAUDE_SID, {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(got["session_id"], CLAUDE_SID)
        self.assertEqual(got["model"], "claude-sonnet-5")
        self.assertEqual(got["effort"], "low")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_codex_fixture_reads_turn_context_payload(self):
        dest_dir = os.path.join(
            self.home, "sessions", "2026", "09", "15")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "rollout-probe-%s.jsonl" % CODEX_SID)
        shutil.copy(
            os.path.join(_FIX, "codex-rollout-session_meta-turn_context.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "codex", CODEX_SID, {"CODEX_HOME": self.home})
        self.assertEqual(got["session_id"], CODEX_SID)
        self.assertEqual(got["model"], "gpt-5.6-sol")
        self.assertEqual(got["effort"], "low")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_grok_fixture_reads_summary_json(self):
        dest_dir = os.path.join(self.home, "sessions", "encoded-cwd", GROK_SID)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "summary.json")
        shutil.copy(os.path.join(_FIX, "grok-session-summary.json"), dest)
        got = read_successor_attest(
            "grok", GROK_SID, {"GROK_HOME": self.home})
        self.assertEqual(got["session_id"], GROK_SID)
        self.assertEqual(got["model"], "grok-4.6")
        self.assertEqual(got["effort"], "low")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_missing_file_is_unknown(self):
        sid = "missing-session-id"
        got = read_successor_attest(
            "claude", sid, {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(got["session_id"], sid)
        self.assertEqual(got["model"], "unknown")
        self.assertEqual(got["effort"], "unknown")
        self.assertEqual(got["source"], "unknown")

    def test_haiku_no_effort_key_is_unknown(self):
        dest = self._claude_path(HAIKU_SID)
        shutil.copy(
            os.path.join(
                _FIX, "claude-transcript-first-assistant-haiku-no-effort.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "claude", HAIKU_SID, {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(got["session_id"], HAIKU_SID)
        self.assertEqual(got["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(got["effort"], "unknown")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_star_session_id_does_not_match_unrelated(self):
        dest = self._claude_path(CLAUDE_SID)
        shutil.copy(
            os.path.join(_FIX, "claude-transcript-first-assistant.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "claude", "*", {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(got["session_id"], "*")
        self.assertEqual(got["model"], "unknown")
        self.assertEqual(got["effort"], "unknown")
        self.assertEqual(got["source"], "unknown")
        still = read_successor_attest(
            "claude", CLAUDE_SID, {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(still["session_id"], CLAUDE_SID)
        self.assertEqual(still["model"], "claude-sonnet-5")
        self.assertEqual(still["effort"], "low")
        self.assertEqual(still["source"], os.path.abspath(dest))

    def test_home_env_resolves_codex_dotdir(self):
        dest_dir = os.path.join(
            self.home, ".codex", "sessions", "2026", "09", "15")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "rollout-probe-%s.jsonl" % CODEX_SID)
        shutil.copy(
            os.path.join(_FIX, "codex-rollout-session_meta-turn_context.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "codex", CODEX_SID, {"HOME": self.home})
        self.assertEqual(got["session_id"], CODEX_SID)
        self.assertEqual(got["model"], "gpt-5.6-sol")
        self.assertEqual(got["effort"], "low")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_relative_codex_home_resolves_against_workdir(self):
        dest_dir = os.path.join(
            self.home, ".acct", "sessions", "2026", "09", "15")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "rollout-probe-%s.jsonl" % CODEX_SID)
        shutil.copy(
            os.path.join(_FIX, "codex-rollout-session_meta-turn_context.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "codex", CODEX_SID, {"CODEX_HOME": ".acct"},
            workdir=self.home)
        self.assertEqual(got["session_id"], CODEX_SID)
        self.assertEqual(got["model"], "gpt-5.6-sol")
        self.assertEqual(got["effort"], "low")
        self.assertEqual(got["source"], os.path.abspath(dest))

    def test_slash_session_id_is_unknown(self):
        dest = self._claude_path(CLAUDE_SID)
        shutil.copy(
            os.path.join(_FIX, "claude-transcript-first-assistant.jsonl"),
            dest,
        )
        got = read_successor_attest(
            "claude", "foo/bar", {"CLAUDE_CONFIG_DIR": self.home})
        self.assertEqual(got["session_id"], "foo/bar")
        self.assertEqual(got["model"], "unknown")
        self.assertEqual(got["effort"], "unknown")
        self.assertEqual(got["source"], "unknown")

class ReadSuccessorAttestRoundTwoTest(unittest.TestCase):
    def test_empty_child_home_never_falls_back_to_runner_home(self):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        sid = "11111111-2222-4333-8444-555555555555"
        runner_home = os.path.join(td, "runner")
        day = os.path.join(runner_home, ".codex", "sessions", "2026", "09", "15")
        os.makedirs(day)
        with open(os.path.join(day, "rollout-x-%s.jsonl" % sid), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "turn_context",
                                 "payload": {"model": "gpt-x", "effort": "low"}}) + "\n")
        with mock.patch.dict(os.environ, {"HOME": runner_home}):
            got = read_successor_attest(
                "codex", sid, {"HOME": "", "CODEX_HOME": "~/.codex"})
        self.assertEqual(got["model"], "unknown")
        self.assertEqual(got["effort"], "unknown")
        self.assertEqual(got["source"], "unknown")

    def test_trailing_newline_session_id_is_unknown(self):
        got = read_successor_attest("codex", "abc\n", {})
        self.assertEqual(got["model"], "unknown")
        self.assertEqual(got["source"], "unknown")


if __name__ == "__main__":
    unittest.main()
