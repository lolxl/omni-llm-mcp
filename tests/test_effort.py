#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for the public `effort` argument.

Pins:
- an admitted omitted effort is medium when that token is listed (grok's ruled set lists it)
- claude with no effort evidence still sends no flag; codex no longer injects high
- specified effort is passed as the backend's own flag when the record lists it
- unknown tokens and backends without an effort flag are LOUD unsupported_effort
- MCP / CLI / agent surfaces all accept the same name
"""
import asyncio
import inspect
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-effort-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

import omni_cli
from providers import agent_lane
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import config
from providers import dispatch
from providers import grok as grok_mod
from providers import jobs
from providers import pi as pi_mod
from providers import openai_compat as openai_mod
from tests import offline_home
from tests.test_cli import run_cli
from tests.test_agent import (
    FakeRunCli,
    _fake_which,
    depth,
    patched_lane,
    run_cli_main,
    LaneTestCase,
)

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True


_HOME = None
_WIDE = ("minimal", "low", "medium", "high", "xhigh", "max")


def _save_model(provider, model, efforts=None, profile="default"):
    """Write one v2 row so an admitted attempt can reach the patched spawn."""
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    item = {
        "id": model,
        "name": model,
        "routable_id": "%s/%s" % (provider, model),
    }
    if efforts is not None:
        item["supported_efforts"] = list(efforts)
    record, _warnings = validate_strict(provider, {
        "provider": provider,
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [item],
    })
    catalog_v2.save(provider, profile, record)


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()



class TestEffortHelpers(unittest.TestCase):
    def test_blank_is_omitted(self):
        self.assertEqual(config.normalize_effort(None), (None, None))
        self.assertEqual(config.normalize_effort(""), (None, None))
        self.assertEqual(config.normalize_effort("  "), (None, None))

    def test_case_and_whitespace(self):
        token, err = config.normalize_effort("  XHigh ")
        self.assertIsNone(err)
        self.assertEqual(token, "xhigh")

    def test_unknown_token_is_loud(self):
        token, err = config.normalize_effort("turbo")
        self.assertIsNone(token)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_agy_accepts_only_installed_cli_efforts(self):
        for value in ("low", "medium", "high", "xhigh", "max"):
            token, err = config.effort_for_provider("agy", value)
            self.assertEqual(token, value)
            self.assertIsNone(err)
        token, err = config.effort_for_provider("agy", "minimal")
        self.assertIsNone(token)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_cursor_accepts_every_mappable_token(self):
        for value in ("minimal", "low", "medium", "high", "xhigh", "max"):
            token, err = config.effort_for_provider("cursor", value)
            self.assertIsNone(err, value)
            self.assertEqual(token, value)

    def test_cursor_unknown_token_is_still_loud(self):
        token, err = config.effort_for_provider("cursor", "turbo")
        self.assertIsNone(token)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_cursor_every_mapped_slug_is_in_the_catalog_snapshot(self):
        """Falsifier for the table: a typo emits an id cursor-agent rejects."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "tests", "fixtures", "cursor_models.txt")
        with open(path, encoding="utf-8") as fh:
            catalog = set(
                line.strip() for line in fh
                if line.strip() and not line.startswith("#")
            )
        self.assertIn("gpt-5.6-sol-xhigh", catalog)  # fixture actually loaded
        checked = 0
        for base, tokens in config.CURSOR_EFFORT_SLUGS.items():
            for token in tokens:
                slug, err = config.cursor_effort_slug(base, token)
                self.assertIsNone(err, (base, token))
                self.assertIn(slug, catalog, "%s + %s -> %s" % (base, token, slug))
                checked += 1
        self.assertGreater(checked, 100)

    def test_cursor_overrides_are_all_load_bearing(self):
        """Every override must be an id the `<base>-<token>` rule cannot spell."""
        for (base, token), slug in config.CURSOR_EFFORT_SLUG_OVERRIDES.items():
            self.assertIn(token, config.CURSOR_EFFORT_SLUGS[base], base)
            self.assertNotEqual(slug, "%s-%s" % (base, token))

    def test_cursor_thinking_after_effort_families(self):
        # pre-5 Claude ids put -thinking AFTER the effort word
        self.assertEqual(
            config.cursor_effort_slug("claude-4.6-opus-thinking", "max"),
            ("claude-4.6-opus-max-thinking", None))
        self.assertEqual(
            config.cursor_effort_slug("claude-4.6-sonnet-thinking", "medium"),
            ("claude-4.6-sonnet-medium-thinking", None))

    def test_cursor_encoded_none_and_bracket_forms_are_named(self):
        for base in ("gpt-5.6-sol-none", "gpt-5.6-sol-none-fast",
                     "claude-4.6-opus-max-thinking"):
            slug, err = config.cursor_effort_slug(base, "low")
            self.assertIsNone(slug, base)
            self.assertIn("already encodes effort", err["detail"], base)
        slug, err = config.cursor_effort_slug(
            "gpt-5.6-sol[context=272k,reasoning=xhigh,fast=false]", "low")
        self.assertIsNone(slug)
        self.assertIn("already carries its parameters", err["detail"])

    def test_cursor_fast_twin_is_not_silently_stripped(self):
        # gpt-5.6-sol-fast is not a catalog id; turning it into gpt-5.6-sol-xhigh
        # would be a quiet variant swap.
        slug, err = config.cursor_effort_slug("gpt-5.6-sol-fast", "xhigh")
        self.assertIsNone(slug)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_cursor_slug_mapping(self):
        self.assertEqual(config.cursor_effort_slug("gpt-5.6-sol", "xhigh"),
                         ("gpt-5.6-sol-xhigh", None))
        self.assertEqual(config.cursor_effort_slug("claude-fable-5", "max"),
                         ("claude-fable-5-max", None))
        self.assertEqual(
            config.cursor_effort_slug("claude-opus-5-thinking", "medium"),
            ("claude-opus-5-thinking-medium", None))
        # gpt-5.5 is the family that spells xhigh differently.
        self.assertEqual(config.cursor_effort_slug("gpt-5.5", "xhigh"),
                         ("gpt-5.5-extra-high", None))

    def test_cursor_model_resolver_passes_profile_to_catalog_lookup(self):
        from providers import cursor as cursor_mod
        with mock.patch.object(cursor_mod, "cursor_effort_slug",
                               return_value=("profile-family-high", None)) as lookup:
            result = cursor_mod.resolve_model("profile-family", "high", "work")
        self.assertEqual(result, ("profile-family-high", None))
        self.assertEqual(lookup.call_args.args,
                         ("profile-family", "high", "work"))

    def test_cursor_slug_omitted_effort_is_passthrough(self):
        self.assertEqual(config.cursor_effort_slug("composer-2.5", None),
                         ("composer-2.5", None))
        self.assertEqual(config.cursor_effort_slug("gpt-5.6-sol-xhigh", None),
                         ("gpt-5.6-sol-xhigh", None))
        # defaulting lives in resolve_model, not here
        self.assertEqual(config.cursor_effort_slug("gpt-5.6-sol", None),
                         ("gpt-5.6-sol", None))

    def test_cursor_default_effort_is_explicit_and_mappable(self):
        from providers import cursor as cursor_mod
        self.assertEqual(set(config.CURSOR_DEFAULT_EFFORT),
                         {"gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"})
        for base, token in config.CURSOR_DEFAULT_EFFORT.items():
            self.assertIn(base, config.CURSOR_EFFORT_SLUGS, base)
            self.assertIn(token, config.CURSOR_EFFORT_SLUGS[base], base)
            slug, err = config.cursor_effort_slug(base, token)
            self.assertIsNone(err, base)
            self.assertIn(slug, config.cursor_effort_variants(base), base)
            resolved, rerr = cursor_mod.resolve_model(base, None)
            self.assertIsNone(rerr, base)
            self.assertEqual(resolved, slug)

    def test_cursor_family_without_that_token_is_loud(self):
        slug, err = config.cursor_effort_slug("claude-opus-5", "max")
        self.assertIsNone(slug)
        self.assertEqual(err["reason"], "unsupported_effort")
        self.assertIn("accepts low, medium, high", err["detail"])

    def test_cursor_model_without_effort_variants_is_loud(self):
        slug, err = config.cursor_effort_slug("composer-2.5", "high")
        self.assertIsNone(slug)
        self.assertEqual(err["reason"], "unsupported_effort")
        self.assertIn("no effort variants", err["detail"])

    def test_cursor_double_encoded_effort_is_loud(self):
        for base in ("gpt-5.6-sol-xhigh", "gpt-5.6-sol-xhigh-fast",
                     "gpt-5.5-extra-high"):
            slug, err = config.cursor_effort_slug(base, "low")
            self.assertIsNone(slug, base)
            self.assertEqual(err["reason"], "unsupported_effort")
            self.assertIn("already encodes effort", err["detail"])

    def test_cursor_coarse_gate_is_the_table_union(self):
        union = set()
        for tokens in config.CURSOR_EFFORT_SLUGS.values():
            union.update(tokens)
        self.assertEqual(set(config.EFFORT_BY_PROVIDER["cursor"]), union)

    def test_cursor_table_tokens_are_public_tokens(self):
        for base, tokens in config.CURSOR_EFFORT_SLUGS.items():
            self.assertTrue(tokens, base)
            for token in tokens:
                self.assertIn(token, config.EFFORT_LEVELS, base)

    def test_none_is_unknown(self):
        token, err = config.normalize_effort("none")
        self.assertIsNone(token)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_claude_rejects_minimal(self):
        token, err = config.effort_for_provider("claude", "minimal")
        self.assertIsNone(token)
        self.assertEqual(err["reason"], "unsupported_effort")
        self.assertIn("claude", err["detail"])

    def test_grok_rejects_none_minimal_max(self):
        for value in ("none", "minimal", "max"):
            token, err = config.effort_for_provider("grok", value)
            self.assertIsNone(token, value)
            self.assertEqual(err["reason"], "unsupported_effort", value)
            if value != "none":
                self.assertIn("grok", err["detail"], value)


class TestConsultArgv(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_grok_omitted_effort_has_no_flag(self):
        # The ruled vocabulary lists medium, so an omitted effort is disclosed
        # as omni's default and sent.
        _save_model("grok", "grok-4.5")
        fake = FakeRunCli(stdout=json.dumps({"text": "OK", "usage": {}}))
        with mock.patch.object(grok_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(grok_mod, "_consult_result",  # json fake, not a stream
                                  side_effect=lambda out, sid: (*grok_mod.extract_cli_result("grok", out), None)), \
                depth(None):
            res = asyncio.run(grok_mod.run_grok_subprocess("hi", "grok-4.5", 30))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res.get("effort"), "medium")
        self.assertEqual(res.get("effort_source"), "omni_default")
        i = fake.argv.index("--reasoning-effort")
        self.assertEqual(fake.argv[i + 1], "medium")

    def test_grok_passes_reasoning_effort(self):
        _save_model("grok", "grok-4.5")
        fake = FakeRunCli(stdout=json.dumps({"text": "OK", "usage": {}}))
        with mock.patch.object(grok_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(grok_mod, "_consult_result",  # json fake, not a stream
                                  side_effect=lambda out, sid: (*grok_mod.extract_cli_result("grok", out), None)), \
                depth(None):
            res = asyncio.run(grok_mod.run_grok_subprocess(
                "hi", "grok-4.5", 30, effort="xhigh"))
        self.assertTrue(res.get("ok"), res)
        i = fake.argv.index("--reasoning-effort")
        self.assertEqual(fake.argv[i + 1], "xhigh")

    def test_claude_omitted_effort_has_no_flag(self):
        _save_model("claude", "opus")
        fake = FakeRunCli(stdout=json.dumps({"result": "OK", "usage": {}}))
        with mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess("hi", "opus", 30))
        self.assertTrue(res.get("ok"), res)
        self.assertNotIn("--effort", fake.argv)

    def test_claude_passes_effort_flag(self):
        _save_model("claude", "opus", _WIDE)
        fake = FakeRunCli(stdout=json.dumps({"result": "OK", "usage": {}}))
        with mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "opus", 30, effort="low"))
        self.assertTrue(res.get("ok"), res)
        i = fake.argv.index("--effort")
        self.assertEqual(fake.argv[i + 1], "low")

    def test_pi_omitted_effort_has_no_flag(self):
        fake = FakeRunCli(stdout=json.dumps({
            "type": "message_end",
            "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "OK"}]},
        }))
        with mock.patch.object(pi_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(pi_mod, "env_depth", return_value=0):
            res = asyncio.run(pi_mod.run_pi_subprocess("hi", "xai/grok-4.6", 30))
        self.assertTrue(res.get("ok"), res)
        self.assertNotIn("--thinking", fake.argv)

    def test_pi_passes_thinking_flag(self):
        fake = FakeRunCli(stdout=json.dumps({
            "type": "message_end",
            "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "OK"}]},
        }))
        with mock.patch.object(pi_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(pi_mod, "env_depth", return_value=0):
            res = asyncio.run(pi_mod.run_pi_subprocess(
                "hi", "xai/grok-4.6", 30, effort="xhigh"))
        self.assertTrue(res.get("ok"), res)
        i = fake.argv.index("--thinking")
        self.assertEqual(fake.argv[i + 1], "xhigh")

    def test_codex_effort_kwarg_overrides_default(self):
        _save_model("codex", "gpt-5.6-sol", _WIDE)
        captured = {}

        async def fake_run_cli(argv, **kwargs):
            captured["argv"] = list(argv)
            if "-o" in argv:
                with open(argv[argv.index("-o") + 1], "w") as handle:
                    handle.write("OK")
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

        with mock.patch.object(codex_mod, "run_cli", fake_run_cli), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(codex_mod, "prepare_host_fence",
                                  side_effect=lambda argv, **kw: (argv, kw.get("env") or {}, {})), \
                depth(None):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, effort="max"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(
            captured["argv"][captured["argv"].index("-c") + 1],
            "model_reasoning_effort=max",
        )


class TestDispatchEffort(unittest.TestCase):
    def test_omitted_effort_is_not_forwarded(self):
        seen = {}

        async def fake_grok(*a, **k):
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": "grok-4.5"}

        with mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "grok-4.5", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertNotIn("effort", seen["kwargs"])

    def test_grok_effort_is_forwarded(self):
        seen = {}

        async def fake_grok(*a, **k):
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": "grok-4.5"}

        with mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "grok-4.5", 30, [], "default", effort="high"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen["kwargs"].get("effort"), "high")

    def test_agy_effort_is_forwarded_after_routing(self):
        seen = {}

        async def fake_agy(*a, **k):
            seen["args"] = a
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": "m"}

        with mock.patch.object(dispatch.agy_mod, "run_agy", fake_agy), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "agy/gpt-oss-120b-medium", 30, [], "default", effort="high"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen["args"][1], "gpt-oss-120b-medium")
        self.assertEqual(seen["kwargs"].get("effort"), "high")

    def test_grok_max_is_rejected_before_provider(self):
        # max is outside the ruled grok vocabulary. Admission refuses inside
        # the provider, before the CLI is spawned.
        _save_model("grok", "grok-4.5")
        fake = FakeRunCli()
        with mock.patch.object(grok_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "grok-4.5", 30, [], "default", effort="max"))
        self.assertEqual(res.get("reason"), "unsupported_effort")
        self.assertEqual(fake.calls, [])

    def test_openai_effort_is_rejected_before_provider(self):
        rec = []

        async def fake(*a, **k):
            rec.append(1)
            return {"ok": True, "text": "x", "model": "gpt-4o"}

        with mock.patch.object(openai_mod, "run_openai_compatible", fake), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "gpt-4o", 30, [], "default", effort="high"))
        self.assertEqual(res.get("reason"), "unsupported_effort")
        self.assertEqual(rec, [])

    def test_call_tool_forwards_effort(self):
        seen = {}

        async def fake_dispatch(*a, **k):
            seen["args"] = a
            seen["kwargs"] = k
            return {"ok": True, "text": "x", "model": "grok-4.5"}

        with mock.patch.object(dispatch, "dispatch_llm_request", fake_dispatch):
            res = asyncio.run(dispatch.call_tool("llm_generate", {
                "prompt": "hi", "model": "grok-4.5", "effort": "xhigh",
            }))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen["kwargs"].get("effort"), "xhigh")

    def test_call_tool_omits_effort_kwarg_when_unset(self):
        seen = {}

        async def fake_dispatch(*a, **k):
            seen["kwargs"] = k
            return {"ok": True, "text": "x", "model": "grok-4.5"}

        with mock.patch.object(dispatch, "dispatch_llm_request", fake_dispatch):
            res = asyncio.run(dispatch.call_tool("llm_generate", {
                "prompt": "hi", "model": "grok-4.5",
            }))
        self.assertTrue(res.get("ok"), res)
        self.assertNotIn("effort", seen["kwargs"])

    def test_route_signature_keeps_effort_optional(self):
        params = inspect.signature(dispatch.route_to_provider).parameters
        self.assertIn("effort", params)
        self.assertEqual(params["effort"].default, None)

    def test_cursor_classification_uses_each_seats_catalog_profile(self):
        # The slug pre-check is gone. The seat's own profile record admits it.
        seat = {"provider": "cursor", "model": "cursor/profile-family",
                "account_profile": "work", "effort": "high"}
        home = tempfile.mkdtemp(prefix="omni-effort-cursor-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        with mock.patch.dict(os.environ, {"OMNI_LLM_CURSOR_CONFIG_DIR_WORK": home}), \
                mock.patch.object(dispatch, "_health_passive_probe"), \
                mock.patch.object(dispatch, "_health_kind", return_value="healthy"):
            _save_model("cursor", "profile-family-high", profile="work")
            result = dispatch._classify_seat(seat, False, None)
        self.assertEqual(result, ("healthy", None))


class TestAgyEffortCli(unittest.TestCase):
    def test_installed_help_confirmation_precedes_effort_spawn(self):
        from providers import agy as agy_mod, model_catalog
        seen = {}
        cwd = tempfile.mkdtemp(prefix="omni-agy-effort-")
        self.addCleanup(shutil.rmtree, cwd, True)

        async def fake_run_cli(argv, **kwargs):
            seen["argv"] = list(argv)
            return {"returncode": 0,
                    "stdout": json.dumps({"response": "ok", "usage": {}}),
                    "stderr": "", "timed_out": False, "watched": False}

        def fake_fence(argv, env):
            seen["fence_argv"] = list(argv)
            return list(argv), env, cwd

        with mock.patch.object(agy_mod, "find_agy", return_value="/usr/bin/agy"), \
                mock.patch.object(agy_mod, "fence_agy_spawn", fake_fence), \
                mock.patch.object(agy_mod, "run_cli", fake_run_cli), \
                mock.patch.object(model_catalog, "agy_effort_tokens",
                                  return_value=frozenset(("low", "high", "xhigh", "max"))):
            result = asyncio.run(agy_mod.run_agy_subprocess(
                "hi", "gpt-oss-120b-medium", 60, "default", effort="high"))
        self.assertTrue(result["ok"], result)
        self.assertIn(["--effort", "high"], [seen["argv"][i:i + 2]
                                               for i in range(len(seen["argv"]) - 1)])
        self.assertIn("--effort", seen["fence_argv"])

    def test_requested_effort_missing_from_help_refuses_before_prompt_spawn(self):
        from providers import agy as agy_mod, model_catalog
        called = []
        with mock.patch.object(agy_mod, "find_agy", return_value="/usr/bin/agy"), \
                mock.patch.object(model_catalog, "agy_effort_tokens",
                                  return_value=frozenset(("low", "medium", "high"))), \
                mock.patch.object(agy_mod, "fence_agy_spawn") as fence:
            result = asyncio.run(agy_mod.run_agy_subprocess(
                "hi", "m", 60, "default", effort="xhigh"))
        self.assertEqual(result.get("reason"), "unsupported_effort")
        fence.assert_not_called()

    def test_explicit_effort_skips_sdk_path(self):
        from providers import agy as agy_mod
        calls = []

        async def fake_sdk(*args, **kwargs):
            calls.append("sdk")
            return {"ok": True}

        async def fake_subprocess(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True, "text": "ok"}

        _save_model("agy", "m")
        with mock.patch.object(agy_mod, "HAS_SDK", True), \
                mock.patch.object(agy_mod.platform, "system", return_value="Linux"), \
                mock.patch.object(agy_mod, "run_agy_sdk", fake_sdk), \
                mock.patch.object(agy_mod, "run_agy_subprocess", fake_subprocess):
            result = asyncio.run(agy_mod.run_agy("hi", "m", 30, effort="high"))
        # An id plus an effort is refused before either branch.
        self.assertEqual(result.get("reason"), "effort_already_encoded")
        self.assertEqual(calls, [])


class TestCliEffort(unittest.TestCase):
    def test_gen_maps_effort(self):
        _, _, _, calls = run_cli(
            ["gen", "hello", "-m", "grok-4.5", "--effort", "high"],
            {"ok": True, "text": "x"})
        self.assertEqual(calls[0][1]["effort"], "high")

    def test_gen_omits_effort_when_unset(self):
        _, _, _, calls = run_cli(["gen", "hello"], {"ok": True, "text": "x"})
        self.assertNotIn("effort", calls[0][1])

    def test_big_maps_effort(self):
        _, _, _, calls = run_cli(
            ["big", "summarize", "-t", "SRC", "--effort", "low"],
            {"ok": True, "text": "s"})
        self.assertEqual(calls[0][1]["effort"], "low")


class TestAgentEffort(LaneTestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_codex_agent_omitted_effort_sends_medium(self):
        _save_model("codex", "gpt-5.6-sol", _WIDE)
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="codex/gpt-5.6-sol",
                workdir=self.mkdir(), timeout_s=30))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("effort"), "medium")
        self.assertEqual(env.get("effort_source"), "omni_default")
        self.assertIn("model_reasoning_effort=medium", fake.argv)

    def test_codex_agent_passes_effort(self):
        _save_model("codex", "gpt-5.6-sol", _WIDE)
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="codex/gpt-5.6-sol",
                workdir=self.mkdir(), timeout_s=30, effort="xhigh"))
        self.assertTrue(env.get("ok"), env)
        self.assertIn("model_reasoning_effort=xhigh", fake.argv)

    def test_claude_agent_omitted_has_no_flag(self):
        _save_model("claude", "sonnet")
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet",
                workdir=self.mkdir(), timeout_s=30))
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("--effort", fake.argv)

    def test_claude_agent_passes_effort(self):
        _save_model("claude", "sonnet", _WIDE)
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet",
                workdir=self.mkdir(), timeout_s=30, effort="max"))
        self.assertTrue(env.get("ok"), env)
        i = fake.argv.index("--effort")
        self.assertEqual(fake.argv[i + 1], "max")

    def test_grok_agent_passes_reasoning_effort(self):
        _save_model("grok", "grok-4.5")
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="grok-4.5",
                workdir=self.mkdir(), timeout_s=30, effort="low"))
        self.assertTrue(env.get("ok"), env)
        i = fake.argv.index("--reasoning-effort")
        self.assertEqual(fake.argv[i + 1], "low")

    def test_pi_agent_omitted_has_no_flag(self):
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="pi/xai/grok-4.6",
                workdir=self.mkdir(), timeout_s=30))
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("--thinking", fake.argv)

    def test_pi_agent_passes_thinking(self):
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="pi/xai/grok-4.6",
                workdir=self.mkdir(), timeout_s=30, effort="max"))
        self.assertTrue(env.get("ok"), env)
        i = fake.argv.index("--thinking")
        self.assertEqual(fake.argv[i + 1], "max")

    def test_agent_rejects_unsupported_backend(self):
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = asyncio.run(agent_lane.run_agent(
                task="t", model="ollama/x",
                workdir=self.mkdir(), timeout_s=30, effort="high"))
        # unsupported_agent fires first for ollama; either refusal is LOUD
        # and nothing is spawned.
        self.assertFalse(env.get("ok"), env)
        self.assertIn(env.get("reason"), ("unsupported_agent", "unsupported_effort"))
        self.assertEqual(fake.calls, [])

    def test_cli_agent_maps_effort(self):
        _, _, _, calls = run_cli_main(
            ["agent", "task", "--workdir", "/repo/wt", "--effort", "high"])
        self.assertEqual(calls[0]["effort"], "high")

    def test_cli_agent_omits_effort_when_unset(self):
        _, _, _, calls = run_cli_main(
            ["agent", "task", "--workdir", "/repo/wt"])
        self.assertNotIn("effort", calls[0])


class TestJobEffort(unittest.TestCase):
    def _job_root(self):
        root = tempfile.mkdtemp(prefix="omni-effort-job-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        wd = os.path.join(root, "wd")
        os.makedirs(wd)
        import subprocess
        subprocess.run(["git", "init", "-q", "-b", "work", wd], check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "--allow-empty", "-q", "-m", "seed"],
            cwd=wd, check=True,
        )
        old = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = root
        self.addCleanup(
            lambda: os.environ.__setitem__("OMNI_LLM_STATE_DIR", old)
            if old is not None else os.environ.pop("OMNI_LLM_STATE_DIR", None))
        return wd

    def test_start_refuses_unknown_token_before_spawn(self):
        wd = self._job_root()
        # Hermetic: no fake codex on PATH. Token check must beat which().
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False):
            out = jobs.start(
                task="t", model="codex/gpt-5.6-sol", workdir=wd,
                timeout_s=5, effort="TURBO")
        self.assertIn("refused", out)
        self.assertEqual(out["refused"]["reason"], "unsupported_effort")
        with open(os.path.join(out["job_dir"], "job.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data.get("state"), "failed")
        self.assertFalse(os.path.isfile(os.path.join(out["job_dir"], "runner.ready")))

    def test_start_persists_bare_model_when_preflight_refuses(self):
        # grok LOUD-refuses --read-only, so _preflight refuses AFTER the
        # @token split; job.json must still carry the bare model + effort.
        wd = self._job_root()
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False):
            out = jobs.start(
                task="t", model="grok/grok-4.6@high", workdir=wd,
                timeout_s=5, read_only=True)
        self.assertIn("refused", out)
        with open(os.path.join(out["job_dir"], "job.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data.get("model"), "grok/grok-4.6")
        self.assertEqual(data.get("effort"), "high")
        self.assertEqual(data.get("state"), "failed")

    def test_start_refuses_bad_at_token_and_keeps_raw_name(self):
        wd = self._job_root()
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False):
            out = jobs.start(
                task="t", model="grok/grok-4.6@bogus", workdir=wd, timeout_s=5)
        self.assertEqual(out["refused"]["reason"], "unsupported_effort")
        with open(os.path.join(out["job_dir"], "job.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data.get("model"), "grok/grok-4.6@bogus")
        self.assertEqual(data.get("state"), "failed")

    def _supervisor_popen(self, *args, **kwargs):
        """Fake only the job_runner detach; leave git subprocess.run intact."""
        argv = args[0] if args else kwargs.get("args") or []
        if argv and "-m" in argv and "providers.job_runner" in argv:
            proc = mock.Mock()
            proc.pid = 12345
            proc.__enter__ = lambda s: s
            proc.__exit__ = lambda *a: False
            return proc
        return self._real_popen(*args, **kwargs)

    def _start_without_spawn(self, wd, **kwargs):
        """start() that passes fences but never detaches a supervisor."""
        self._real_popen = jobs.subprocess.Popen
        with mock.patch.object(shutil, "which", lambda n, *a, **k: "/fake/bin/" + str(n)), \
                mock.patch.object(jobs.sandbox_canary, "gate", return_value=None), \
                mock.patch.object(jobs.subprocess, "Popen",
                                  side_effect=self._supervisor_popen), \
                mock.patch.object(jobs, "_wait_for_supervisor_lock",
                                  return_value=True), \
                mock.patch.object(jobs, "_reap_process"):
            return jobs.start(
                task="t", model="codex/gpt-5.6-sol", workdir=wd,
                timeout_s=5, **kwargs)

    def test_start_persists_normalized_effort(self):
        wd = self._job_root()
        out = self._start_without_spawn(wd, effort="XHigh")
        self.assertNotIn("refused", out, out)
        with open(os.path.join(out["job_dir"], "job.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data.get("effort"), "xhigh")

    def test_resume_inherits_parent_effort(self):
        wd = self._job_root()
        parent = self._start_without_spawn(wd, effort="XHigh")
        parent_dir = parent["job_dir"]
        data = jobs._read_job(parent_dir)
        data["state"] = "done"
        data["session_id"] = "t-parent-effort"
        jobs._write_job(parent_dir, data)
        jobs._write_result(parent_dir, {"ok": True, "text": "done"})
        self._real_popen = jobs.subprocess.Popen
        with mock.patch.object(shutil, "which", lambda n, *a, **k: "/fake/bin/" + str(n)), \
                mock.patch.object(jobs.sandbox_canary, "gate", return_value=None), \
                mock.patch.object(jobs.subprocess, "Popen",
                                  side_effect=self._supervisor_popen), \
                mock.patch.object(jobs, "_wait_for_supervisor_lock",
                                  return_value=True), \
                mock.patch.object(jobs, "_reap_process"):
            child = jobs.resume(parent_dir, "continue")
        self.assertNotIn("refused", child, child)
        with open(os.path.join(child["job_dir"], "job.json"), encoding="utf-8") as handle:
            child_data = json.load(handle)
        self.assertEqual(child_data.get("effort"), "xhigh")
        self.assertEqual(child_data.get("resumed_from"), data["job_id"])

    def test_job_runner_forwards_effort_to_run_agent(self):
        from providers import job_runner
        wd = self._job_root()
        job_dir = os.path.join(os.environ["OMNI_LLM_STATE_DIR"], "jobs", "effortfwd")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "prompt.txt"), "w", encoding="utf-8") as handle:
            handle.write("task")
        job = {
            "task": "task", "model": "codex/gpt-5.6-sol", "workdir": wd,
            "read_only": False, "timeout_s": 5, "account_profile": "default",
            "session_id": None, "resume_session": None, "images": [],
            "effort": "xhigh", "provider": "codex",
        }
        jobs._write_job(job_dir, job)
        seen = {}

        async def fake_run_agent(**kwargs):
            seen.update(kwargs)
            return {"ok": True, "text": "x", "model": "m",
                    "workdir": wd, "duration_s": 0.1}

        async def fake_poll(*a, **k):
            return None

        class _FakeNorm:
            def __init__(self, *a, **k):
                pass

            def poll_once(self):
                return None

            def final_drain(self):
                return None

        with mock.patch.object(job_runner, "run_agent", fake_run_agent), \
                mock.patch.object(job_runner, "_StreamNormalizer", _FakeNorm), \
                mock.patch.object(job_runner, "_poll_session_id", fake_poll), \
                mock.patch.object(job_runner, "_normalize_loop", fake_poll), \
                mock.patch.object(job_runner.signal, "signal", lambda *a, **k: None), \
                mock.patch.object(job_runner.signal, "getsignal", lambda *a, **k: None):
            asyncio.run(job_runner._execute(job_dir, job))
        self.assertEqual(seen.get("effort"), "xhigh")

    def test_cli_agent_start_forwards_effort(self):
        import io
        from contextlib import redirect_stdout
        captured = {}

        def fake_start(**kwargs):
            captured.update(kwargs)
            return {"job_id": "j", "job_dir": "/tmp/j"}

        output = io.StringIO()
        with mock.patch.object(omni_cli.jobs_mod, "start", fake_start), \
                redirect_stdout(output):
            code = omni_cli._main_agent_job(
                ["start", "task", "--workdir", "/repo/wt", "--effort", "XHigh"])
        self.assertEqual(code, 0)
        self.assertEqual(captured.get("effort"), "XHigh")


class TestAtEffortShorthand(unittest.TestCase):
    """Trailing `@<token>` is split before route resolution on every consult seat."""

    def test_codex_at_high_reaches_runner(self):
        seen = {}

        async def fake_codex(prompt, model, *a, **k):
            seen["model"] = model
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": model}

        with mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "codex/gpt-5.6-sol@high", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen.get("model"), "gpt-5.6-sol")
        self.assertEqual(seen["kwargs"].get("effort"), "high")

    def test_claude_at_xhigh_reaches_runner(self):
        seen = {}

        async def fake_claude(prompt, model, *a, **k):
            seen["model"] = model
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": model}

        with mock.patch.object(claude_mod, "run_claude_subprocess", fake_claude), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "claude/x@xhigh", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen.get("model"), "x")
        self.assertEqual(seen["kwargs"].get("effort"), "xhigh")

    def test_grok_prefix_at_high_reaches_runner(self):
        seen = {}

        async def fake_grok(prompt, model, *a, **k):
            seen["model"] = model
            seen["kwargs"] = dict(k)
            return {"ok": True, "text": "x", "model": model}

        with mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "grok/grok-4.6@high", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen.get("model"), "grok-4.6")
        self.assertEqual(seen["kwargs"].get("effort"), "high")

    def test_at_token_plus_matching_effort_is_conflict(self):
        rec = []

        async def fake_codex(*a, **k):
            rec.append(1)
            return {"ok": True, "text": "x", "model": "x"}

        with mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "codex/x@high", 30, [], "default", effort="high"))
        self.assertEqual(res.get("reason"), "unsupported_effort")
        self.assertEqual(rec, [])

    def test_at_bogus_is_loud_and_skips_runner(self):
        rec = []

        async def fake_codex(*a, **k):
            rec.append(1)
            return {"ok": True, "text": "x", "model": "x"}

        with mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                depth(None):
            res = asyncio.run(dispatch.route_to_provider(
                "hi", "codex/x@bogus", 30, [], "default"))
        self.assertEqual(res.get("reason"), "unsupported_effort")
        self.assertEqual(rec, [])

    def test_chain_bad_token_records_tried_and_falls_through(self):
        rec = []

        async def fake_codex(*a, **k):
            rec.append("codex")
            return {"ok": False, "reason": "should_not_run"}

        async def fake_grok(prompt, model, *a, **k):
            rec.append("grok")
            return {"ok": True, "text": "x", "model": model}

        _save_model("grok", "grok-4.6")
        with mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            res = asyncio.run(dispatch.dispatch_llm_request(
                "hi", "codex/x@bogus,grok-4.6", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res.get("model"), "grok-4.6")
        self.assertEqual(res.get("tried")[0]["reason"], "unsupported_effort")
        self.assertNotIn("codex", rec)
        self.assertIn("grok", rec)

    def test_chain_cooldown_keys_on_bare_name(self):
        # A flag stored under the bare name must skip the `@token` link.
        from providers import health
        rec = []

        async def fake_codex(*a, **k):
            rec.append("codex")
            return {"ok": True, "text": "x", "model": "x"}

        async def fake_grok(prompt, model, *a, **k):
            rec.append("grok")
            return {"ok": True, "text": "x", "model": model}

        state = tempfile.mkdtemp(prefix="omni-effort-health-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}), \
                mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            _save_model("codex", "x", _WIDE)
            _save_model("grok", "grok-4.6")
            health.flag("codex/x", "default", source="test")
            res = asyncio.run(dispatch.dispatch_llm_request(
                "hi", "codex/x@high,grok-4.6", 30, [], "default"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res.get("tried")[0]["reason"], "skipped_cooldown")
        self.assertEqual(rec, ["grok"])

    def test_chain_link_token_beats_call_effort_per_link(self):
        seen = {}
        state = tempfile.mkdtemp(prefix="omni-effort-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        env_patch = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": state, "OMNI_LLM_QUOTA_PROBE": "0"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        _save_model("codex", "x", _WIDE)
        _save_model("grok", "grok-4.6")

        async def fake_codex(prompt, model, *a, **k):
            seen["codex"] = k.get("effort")
            # account_exhausted is the one codex failure that moves on to the next seat.
            return {"ok": False, "reason": "account_exhausted"}

        async def fake_grok(prompt, model, *a, **k):
            seen["grok"] = k.get("effort")
            return {"ok": True, "text": "x", "model": model}

        with mock.patch.object(codex_mod, "run_codex_subprocess", fake_codex), \
                mock.patch.object(grok_mod, "run_grok_subprocess", fake_grok), \
                depth(None):
            res = asyncio.run(dispatch.dispatch_llm_request(
                "hi", "codex/x@xhigh,grok-4.6", 30, [], "default", effort=None))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(seen, {"codex": "xhigh", "grok": None})

    def test_cursor_identical_token_and_effort_is_conflict(self):
        from providers import cursor as cursor_mod
        slug, err = cursor_mod.resolve_model("sol@high", "high")
        self.assertIsNone(slug)
        self.assertEqual(err["reason"], "unsupported_effort")

    def test_preflight_accepts_grok_prefix_at_high_rejects_at_max(self):
        with tempfile.TemporaryDirectory() as workdir, \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(shutil, "which",
                                  lambda n, *a, **k: "/fake/bin/" + str(n)), \
                mock.patch.object(jobs.sandbox_canary, "gate", return_value=None), \
                depth(None):
            refusal, _, route = jobs._preflight(
                "grok/grok-4.6@high", workdir, False, False)
            self.assertIsNone(refusal, refusal)
            self.assertEqual(route["provider"], "grok")
            self.assertEqual(route["model"], "grok-4.6")

            refusal, _, _ = jobs._preflight(
                "grok/grok-4.6@max", workdir, False, False)
            self.assertIsNotNone(refusal)
            self.assertEqual(refusal["reason"], "unsupported_effort")


if __name__ == "__main__":
    unittest.main()
