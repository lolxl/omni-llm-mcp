#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for per-road consult profile rings (issue #72).

Fake runners only. Ledger isolated under OMNI_LLM_STATE_DIR.
"""
import asyncio
import contextlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import dispatch, health
from tests.test_family import FamilyCase


class ProfileRingCase(FamilyCase):
    def setUp(self):
        super().setUp()
        self.claude_dirs = {}
        for name in ("a", "b", "c"):
            path = tempfile.mkdtemp(prefix="omni-ring-claude-%s-" % name)
            self.addCleanup(shutil.rmtree, path, ignore_errors=True)
            self.claude_dirs[name] = path
            os.environ["CLAUDE_CONFIG_DIR_%s" % name.upper()] = path
        self.addCleanup(self._drop_claude_profile_vars)

    def _dispatch(self, fakes, model, account_profile="default", effort=None,
                  require_read_wall=None, extra_env=None):
        # Keep the public account wrappers alive; fake only vendor attempts.
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, extra_env or {}, clear=False))
            stack.enter_context(mock.patch.object(dispatch.codex_mod, "run_codex_subprocess", fakes.codex))
            stack.enter_context(mock.patch.object(dispatch.cursor_mod, "run_cursor_subprocess", fakes.cursor))
            stack.enter_context(mock.patch.object(dispatch.claude_mod, "_run_claude_once", fakes.claude))
            stack.enter_context(mock.patch.object(dispatch.grok_mod, "_run_grok_once", fakes.grok))
            stack.enter_context(mock.patch.object(dispatch.agy_mod, "run_agy", fakes.agy))
            stack.enter_context(mock.patch.object(dispatch.openai_mod, "run_openai_compatible", fakes.openai))
            stack.enter_context(mock.patch.object(dispatch.pi_mod, "run_pi_subprocess", fakes.pi))
            stack.enter_context(mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0))
            return asyncio.run(dispatch.dispatch_llm_request(
                "synthetic prompt", model, 30, None, account_profile,
                effort=effort, require_read_wall=require_read_wall))

    def _drop_claude_profile_vars(self):
        for name in ("a", "b", "c"):
            os.environ.pop("CLAUDE_CONFIG_DIR_%s" % name.upper(), None)


class TestProfileRingDispatch(ProfileRingCase):
    def test_no_ring_single_explicit_seat(self):
        fakes = self._fakes()
        first = self._dispatch(fakes, "claude/sonnet")
        self.assertTrue(first.get("ok"), first)
        self.assertEqual(first["seat"]["account_profile"], "default")
        self.assertEqual(first["seat"]["model"], "claude/sonnet")
        self.assertEqual(first["seat"]["family"], None)
        self.assertNotIn("tried", first)
        self.assertEqual(
            [(c["model"], c["account_profile"]) for c in fakes.calls],
            [("sonnet", "default")],
        )
        fakes2 = self._fakes()
        second = self._dispatch(fakes2, "claude/sonnet")
        self.assertEqual(first, second)

    def test_ring_expands_three_seats_in_order(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["account_profile"], "a")
        self.assertEqual(res["seat"]["model"], "claude/sonnet")
        self.assertEqual(
            [(c["model"], c["account_profile"]) for c in fakes.calls],
            [("sonnet", "a")],
        )
        self.assertNotIn("tried", res)

    def test_cooling_first_ring_seat_skips_to_b(self):
        health.flag("claude/sonnet", "a")
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["account_profile"], "b")
        self.assertEqual(res["seat"]["model"], "claude/sonnet")
        self.assertEqual(res["tried"][0]["reason"], "skipped_cooldown")
        self.assertEqual(res["tried"][0]["account_profile"], "a")
        self.assertEqual(
            [c["account_profile"] for c in fakes.calls], ["b"])

    def test_missing_c_is_ineligible_call_succeeds_on_a(self):
        os.environ.pop("CLAUDE_CONFIG_DIR_C", None)
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "c,a,b"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["account_profile"], "a")
        self.assertEqual(res["tried"][0]["reason"], "account_unavailable")
        self.assertEqual(res["tried"][0]["account_profile"], "c")
        self.assertEqual(
            [c["account_profile"] for c in fakes.calls], ["a"])

    def test_all_three_missing_is_no_eligible_seat(self):
        for name in ("a", "b", "c"):
            os.environ.pop("CLAUDE_CONFIG_DIR_%s" % name.upper(), None)
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "account_exhausted")
        self.assertEqual(
            [t["reason"] for t in res["tried"] if t.get("account_profile")],
            ["account_unavailable", "account_unavailable", "account_unavailable"],
        )
        self.assertEqual(
            [t["account_profile"] for t in res["tried"] if t.get("account_profile")],
            ["a", "b", "c"],
        )
        self.assertEqual(fakes.calls, [])

    def test_call_level_profile_pins_and_disables_ring(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet", account_profile="b",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["account_profile"], "b")
        self.assertEqual(
            [c["account_profile"] for c in fakes.calls], ["b"])
        self.assertNotIn("tried", res)

    def test_fable_unknown_failure_stops_before_other_accounts_and_cursor(self):
        fakes = self._fakes(fail=("claude",))
        res = self._dispatch(
            fakes, "fable",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "runtime_error")
        self.assertEqual(
            [(c["provider"], c["account_profile"]) for c in fakes.calls],
            [("claude", "a")],
        )

    def test_malformed_empty_entry_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,,b"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_duplicate_name_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,a"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_unknown_road_var_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_PI": "a,b"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_PI", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_auto_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "auto"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("discovery", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_auto_on_other_road_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CURSOR": "auto"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("discovery", res["detail"])
        self.assertIn("OMNI_LLM_PROFILES_CURSOR", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_auto_list_profiles_once_per_road_with_three_seats(self):
        rows = [
            {"road": "claude", "name": "a", "path": self.claude_dirs["a"], "state": "identity_cached",
             "problems": []},
            {"road": "claude", "name": "b", "path": self.claude_dirs["b"], "state": "identity_cached",
             "problems": []},
            {"road": "claude", "name": "c", "path": self.claude_dirs["c"], "state": "identity_cached",
             "problems": []},
        ]
        fakes = self._fakes()
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows) as lp:
            res = self._dispatch(
                fakes, "claude/sonnet,claude/opus,claude/haiku",
                extra_env={
                    "OMNI_LLM_PROFILES_CLAUDE": "auto",
                    "OMNI_LLM_PROFILE_DISCOVERY": "1",
                })
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(lp.call_count, 1)

    def test_empty_string_is_profile_config(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": ""})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", res["detail"])
        self.assertEqual(fakes.calls, [])

    def test_one_entry_ring_missing_dir_is_no_eligible_seat(self):
        os.environ.pop("CLAUDE_CONFIG_DIR_A", None)
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "account_exhausted")
        self.assertEqual(res["tried"][0]["reason"], "account_unavailable")
        self.assertEqual(res["tried"][0]["account_profile"], "a")
        self.assertEqual(fakes.calls, [])

    def test_one_entry_ring_cooling_is_skipped_for_later_seat(self):
        health.flag("claude/sonnet", "a")
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "claude/sonnet,grok-4.6",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["provider"], "grok")
        self.assertEqual(res["tried"][0]["reason"], "skipped_cooldown")
        self.assertEqual(res["tried"][0]["account_profile"], "a")
        self.assertEqual(
            [(c["provider"], c["account_profile"]) for c in fakes.calls],
            [("grok", "default")],
        )

    def test_mixed_chain_unknown_failure_stops_before_grok(self):
        fakes = self._fakes(fail=("claude",))
        res = self._dispatch(
            fakes, "claude/sonnet,grok-4.6",
            extra_env={"OMNI_LLM_PROFILES_CLAUDE": "a,b,c"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "runtime_error")
        self.assertEqual(
            [(c["provider"], c["account_profile"]) for c in fakes.calls],
            [("claude", "a")],
        )


if __name__ == "__main__":
    unittest.main()
