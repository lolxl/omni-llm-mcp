#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for family short names, per-seat profile, read-wall gate.

Issue 72-C. Fake runners only. Ledger isolated under OMNI_LLM_STATE_DIR.
"""
import asyncio
import os
import shutil
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-family-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import dispatch
from providers import health
from providers.family import expand, is_family


def _run(coro):
    return asyncio.run(coro)


class _Fakes:
    """Record every provider runner call; optional fail set skips a success."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def _record(self, provider, model, account_profile, effort=None):
        self.calls.append({
            "provider": provider,
            "model": model,
            "account_profile": account_profile,
            "effort": effort,
        })

    def _result(self, provider, model, ok_model=None):
        if provider in self.fail:
            if provider == "codex":
                # Any other codex failure ends the request; account_exhausted
                # (no task child ran) is the one codex result that moves on.
                return {"ok": False, "reason": "account_exhausted", "detail": "fake"}
            return {"ok": False, "reason": "runtime_error", "detail": "fake"}
        return {
            "ok": True,
            "text": "from-%s" % provider,
            "model": ok_model if ok_model is not None else model,
        }

    async def codex(self, prompt, model, timeout_s, account_profile="default",
                    files=None, effort=None, **_k):
        self._record("codex", model, account_profile, effort)
        return self._result("codex", model, "gpt-5.6-sol")

    async def cursor(self, prompt, model, timeout_s, account_profile="default",
                     effort=None, **_k):
        self._record("cursor", model, account_profile, effort)
        return self._result("cursor", model, "cursor/%s" % model)

    async def claude(self, prompt, model, timeout_s, account_profile="default",
                     effort=None, **_k):
        self._record("claude", model, account_profile, effort)
        return self._result("claude", model)

    async def grok(self, prompt, model, timeout_s, account_profile="default",
                   files=None, effort=None, **_k):
        self._record("grok", model, account_profile, effort)
        return self._result("grok", model)

    async def agy(self, prompt, model, timeout_s, files=None,
                  account_profile="default", **_k):
        self._record("agy", model, account_profile, None)
        return self._result("agy", model)

    async def openai(self, prompt, model, timeout_s, account_profile,
                     api_base=None, **_k):
        self._record("openai", model, account_profile, None)
        return {"ok": False, "reason": "runtime_error", "detail": "openai must not run"}

    async def pi(self, prompt, model, timeout_s, account_profile="default",
                 effort=None, **_k):
        self._record("pi", model, account_profile, effort)
        return {"ok": False, "reason": "runtime_error", "detail": "pi must not run"}


class FamilyCase(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp(prefix="omni-family-state-")
        self.sessions_dir = tempfile.mkdtemp(prefix="omni-family-sessions-")
        self.addCleanup(shutil.rmtree, self.state_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.sessions_dir, ignore_errors=True)
        self._saved = {}
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_FAMILY_") or key.startswith(
                    "OMNI_LLM_PROFILES_") or key in (
                    "OMNI_LLM_REQUIRE_READ_WALL", "OMNI_LLM_STATE_DIR",
                    "OMNI_LLM_CODEX_SESSIONS_DIR", "CODEX_HOME_ALT",
                    "CURSOR_API_KEY_ALT"):
                self._saved[key] = os.environ.pop(key)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state_dir
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "0"  # never spawn the real RPC in unit tests
        os.environ["OMNI_LLM_CODEX_SESSIONS_DIR"] = self.sessions_dir
        self.codex_alt = tempfile.mkdtemp(prefix="omni-codex-alt-")
        self.addCleanup(shutil.rmtree, self.codex_alt, ignore_errors=True)
        os.environ["CODEX_HOME_ALT"] = self.codex_alt
        os.environ["CURSOR_API_KEY_ALT"] = "fake-cursor-alt"
        health.clear_all()

    def tearDown(self):
        health.clear_all()
        for key in ("OMNI_LLM_STATE_DIR", "OMNI_LLM_QUOTA_PROBE", "OMNI_LLM_CODEX_SESSIONS_DIR",
                    "OMNI_LLM_REQUIRE_READ_WALL", "CODEX_HOME_ALT",
                    "CURSOR_API_KEY_ALT"):
            os.environ.pop(key, None)
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_FAMILY_") or key.startswith(
                    "OMNI_LLM_PROFILES_"):
                os.environ.pop(key, None)
        os.environ.update(self._saved)

    def _fakes(self, fail=()):
        fakes = _Fakes(fail=fail)
        fakes.state_dir = self.state_dir
        fakes.sessions_dir = self.sessions_dir
        return fakes

    def _dispatch(self, fakes, model, account_profile="default", effort=None,
                  require_read_wall=None, extra_env=None):
        patches = [
            mock.patch.object(dispatch.codex_mod, "run_codex_subprocess", fakes.codex),
            mock.patch.object(dispatch.cursor_mod, "run_cursor_subprocess", fakes.cursor),
            mock.patch.object(dispatch.claude_mod, "run_claude_subprocess", fakes.claude),
            mock.patch.object(dispatch.grok_mod, "run_grok_subprocess", fakes.grok),
            mock.patch.object(dispatch.agy_mod, "run_agy", fakes.agy),
            mock.patch.object(dispatch.openai_mod, "run_openai_compatible", fakes.openai),
            mock.patch.object(dispatch.pi_mod, "run_pi_subprocess", fakes.pi),
            mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0),
        ]
        env = extra_env or {}
        entered = []
        if env:
            p = mock.patch.dict(os.environ, env, clear=False)
            p.start()
            entered.append(p)
        for p in patches:
            p.start()
            entered.append(p)
        try:
            kw = {}
            if effort is not None:
                kw["effort"] = effort
            if require_read_wall is not None:
                kw["require_read_wall"] = require_read_wall
            return _run(dispatch.dispatch_llm_request(
                "p", model, 30, None, account_profile, **kw))
        finally:
            for p in reversed(entered):
                p.stop()


class TestAgyReadWallFlag(FamilyCase):
    """agy's ``read_wall`` follows the verified consult fence probe (issue #161)."""

    AGY_MODEL = "Gemini 3.8 Flash (High)"

    def test_snapshot_flag_follows_the_probe(self):
        for verified in (True, False):
            with self.subTest(verified=verified), mock.patch.object(
                    dispatch, "_consult_read_wall_verified", return_value=verified):
                snap = dispatch.backends_snapshot()
                self.assertIs(snap["agy"]["read_wall"], verified)
                self.assertEqual(snap["agy"]["agent_fence"], "none")

    def test_probe_failure_is_false_not_an_exception(self):
        with mock.patch.object(dispatch.kernel_sandbox, "available",
                               side_effect=RuntimeError("boom")):
            self.assertIs(dispatch.backends_snapshot()["agy"]["read_wall"], False)

    def test_required_read_wall_selects_agy_when_verified_and_refuses_when_not(self):
        fakes = self._fakes()
        with mock.patch.object(dispatch, "_consult_read_wall_verified",
                               return_value=True):
            res = self._dispatch(fakes, self.AGY_MODEL, require_read_wall=True)
        self.assertTrue(res.get("ok"), res)
        self.assertEqual([c["provider"] for c in fakes.calls], ["agy"])
        fakes = self._fakes()
        with mock.patch.object(dispatch, "_consult_read_wall_verified",
                               return_value=False):
            res = self._dispatch(fakes, self.AGY_MODEL, require_read_wall=True)
        self.assertEqual(res.get("reason"), "read_wall_required", res)
        self.assertEqual(fakes.calls, [])


class TestFamilyExpand(FamilyCase):
    def test_default_sol_order_and_profiles(self):
        seats, err = expand("sol", "default")
        self.assertIsNone(err)
        self.assertEqual(
            [(s["provider"], s["model"], s["account_profile"]) for s in seats],
            [("codex", "codex/gpt-5.6-sol", "default"),
             ("cursor", "cursor/gpt-5.6-sol", "default")],
        )

    def test_env_reorder_and_per_road_profile(self):
        os.environ["OMNI_LLM_FAMILY_SOL"] = "cursor,codex:alt"
        seats, err = expand("sol", "default")
        self.assertIsNone(err)
        self.assertEqual(
            [(s["provider"], s["account_profile"]) for s in seats],
            [("cursor", "default"), ("codex", "alt")],
        )
        seats_team, err_team = expand("sol", "team")
        self.assertIsNone(err_team)
        self.assertEqual(
            [(s["provider"], s["account_profile"]) for s in seats_team],
            [("cursor", "team"), ("codex", "team")],
        )

    def test_unknown_road_is_family_config(self):
        os.environ["OMNI_LLM_FAMILY_SOL"] = "grok"
        seats, err = expand("sol", "default")
        self.assertIsNone(seats)
        self.assertEqual(err["reason"], "family_config")
        self.assertIn("OMNI_LLM_FAMILY_SOL", err["detail"])

    def test_duplicate_road_is_family_config(self):
        os.environ["OMNI_LLM_FAMILY_SOL"] = "codex,codex"
        seats, err = expand("sol", "default")
        self.assertIsNone(seats)
        self.assertEqual(err["reason"], "family_config")

    def test_is_family_whole_name_only(self):
        self.assertFalse(is_family("cursor/sol"))
        self.assertTrue(is_family("SOL"))
        self.assertTrue(is_family("sol"))
        self.assertFalse(is_family("sol@high"))


class TestFamilyDispatch(FamilyCase):
    def test_a_sol_high_skips_cooling_codex(self):
        health.flag("codex/gpt-5.6-sol", "default")
        fakes = self._fakes()
        res = self._dispatch(fakes, "sol@high")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["provider"], "cursor")
        self.assertEqual(res["seat"]["family"], "sol")
        self.assertEqual(res["seat"]["model"], "cursor/gpt-5.6-sol")
        self.assertEqual(res["tried"][0], {
            "model": "codex/gpt-5.6-sol",
            "account_profile": "default",
            "reason": "skipped_cooldown",
        })
        self.assertEqual([c["provider"] for c in fakes.calls], ["cursor"])

    def test_b_both_flagged_dispatches_cooling_in_order(self):
        health.flag("codex/gpt-5.6-sol", "default")
        health.flag("cursor/gpt-5.6-sol", "default")
        fakes = self._fakes(fail=("codex",))
        res = self._dispatch(fakes, "sol")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual([c["provider"] for c in fakes.calls], ["codex", "cursor"])
        self.assertEqual(res["seat"]["provider"], "cursor")

    def test_c_fable_minimal_no_eligible_no_agy(self):
        fakes = self._fakes()
        res = self._dispatch(fakes, "fable@minimal")
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "no_eligible_seat")
        reasons = [t["reason"] for t in res["tried"]]
        self.assertEqual(reasons, ["unsupported_effort", "unsupported_effort"])
        models = [t["model"] for t in res["tried"]]
        self.assertEqual(models, [
            "claude/claude-fable-5-1", "cursor/claude-fable-5-1",
        ])
        self.assertEqual(fakes.calls, [])

    def test_d_fable_require_read_wall(self):
        fakes = self._fakes()
        with mock.patch.object(dispatch, "backends_snapshot", return_value={
            "claude": {"read_wall": False},
            "cursor": {"read_wall": False},
        }):
            res = self._dispatch(fakes, "fable", require_read_wall=True)
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "no_eligible_seat")
        self.assertEqual(
            [t["reason"] for t in res["tried"]],
            ["read_wall_required", "read_wall_required"],
        )
        self.assertEqual(fakes.calls, [])

    def test_claude_explicit_model_dispatches_when_read_wall_is_verified(self):
        fakes = self._fakes()
        snapshot = {
            "claude": {"read_wall": True},
        }
        with mock.patch.object(dispatch, "backends_snapshot", return_value=snapshot):
            res = self._dispatch(
                fakes, "claude/opus", require_read_wall=True,
            )
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["provider"], "claude")
        self.assertEqual([call["provider"] for call in fakes.calls], ["claude"])

    def test_e_explicit_cursor_sol_loud_read_wall(self):
        fakes = self._fakes()
        res = self._dispatch(fakes, "cursor/sol", require_read_wall=True)
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "read_wall_required")
        self.assertNotEqual(res["reason"], "no_eligible_seat")
        self.assertEqual(fakes.calls, [])

    def test_f_call_level_alt_reaches_every_runner(self):
        fakes = self._fakes(fail=("codex",))
        res = self._dispatch(fakes, "sol", account_profile="alt")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(
            [c["account_profile"] for c in fakes.calls],
            ["alt", "alt"],
        )
        self.assertEqual(res["seat"]["account_profile"], "alt")

    def test_g_env_per_road_profile_on_default_call(self):
        fakes = self._fakes(fail=("codex",))
        res = self._dispatch(
            fakes, "sol",
            extra_env={"OMNI_LLM_FAMILY_SOL": "codex:alt,cursor"})
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(
            [(c["provider"], c["account_profile"]) for c in fakes.calls],
            [("codex", "alt"), ("cursor", "default")],
        )

    def test_h_plain_grok_carries_seat(self):
        fakes = self._fakes()
        res = self._dispatch(fakes, "grok-4.6")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"], {
            "family": None,
            "provider": "grok",
            "model": "grok-4.6",
            "account_profile": "default",
        })

    def test_i_family_then_explicit_three_seats(self):
        fakes = self._fakes(fail=("codex", "cursor"))
        res = self._dispatch(fakes, "sol@high,grok-4.6")
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res["seat"]["provider"], "grok")
        self.assertEqual(
            [t["model"] for t in res["tried"]],
            ["codex/gpt-5.6-sol", "cursor/gpt-5.6-sol"],
        )
        self.assertEqual(
            [c["provider"] for c in fakes.calls],
            ["codex", "cursor", "grok"],
        )

    def test_j_duplicate_env_family_config_no_runner(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "sol",
            extra_env={"OMNI_LLM_FAMILY_SOL": "codex,codex"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "family_config")
        self.assertEqual(fakes.calls, [])

    def test_k_env_require_read_wall_omitted_param(self):
        fakes = self._fakes()
        res = self._dispatch(
            fakes, "cursor/sol",
            extra_env={"OMNI_LLM_REQUIRE_READ_WALL": "1"})
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res["reason"], "read_wall_required")
        self.assertEqual(fakes.calls, [])


if __name__ == "__main__":
    unittest.main()


class TestNoEligibleSeatEnvelope(unittest.TestCase):
    def test_detail_names_every_seat_and_reason(self):
        from providers import dispatch as dispatch_mod
        import asyncio
        with mock.patch.dict(os.environ, {"OMNI_LLM_REQUIRE_READ_WALL": "1"}), \
             mock.patch.object(dispatch_mod, "backends_snapshot", return_value={
                 "claude": {"read_wall": False},
                 "cursor": {"read_wall": False},
             }):
            res = asyncio.run(dispatch_mod.dispatch_llm_request(
                "hi", "fable", 30, [], "default"))
        self.assertEqual(res["reason"], "no_eligible_seat")
        self.assertIn("claude/claude-fable-5-1 (read_wall_required)", res["detail"])
        self.assertIn("cursor/claude-fable-5-1 (read_wall_required)", res["detail"])
