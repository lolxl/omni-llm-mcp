#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for the cursor/ consult backend. No network, no real CLI."""
import asyncio
import glob
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from providers import config as config_mod
from providers import cursor as cursor_mod
from providers import dispatch as dispatch_mod
from providers.route import resolve_route
from providers.usage import extract_cli_result, inband_error, normalize_usage
from tests.test_agent import FakeRunCli, depth

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True


_CURSOR_STATE = None
_STATE_PATCH = None


def _seed_cursor(profile="default"):
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    from providers.config import FABLE_MODEL
    ids = [
        "composer-2.5",
        "gpt-5.6-sol-medium",
        "gpt-5.6-sol-high",
        "gpt-5.6-sol-xhigh",
        FABLE_MODEL,
        FABLE_MODEL + "-max",
        "claude-opus-5-high",
        "gpt-5.5-extra-high",
        "claude-4.6-opus-max-thinking",
        "gpt-5.6-sol-none",
    ]
    record, _warnings = validate_strict("cursor", {
        "provider": "cursor",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [{
            "id": model_id,
            "name": model_id,
            "routable_id": "cursor/%s" % model_id,
        } for model_id in ids],
    })
    catalog_v2.save("cursor", profile, record)



def setUpModule():
    # Scoped to this module: the one-process runner imports every module first.
    global _CURSOR_STATE, _STATE_PATCH
    _CURSOR_STATE = tempfile.mkdtemp(prefix="omni-cursor-state-")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _CURSOR_STATE})
    _STATE_PATCH.start()
    try:
        _seed_cursor()
    except BaseException:
        # tearDownModule does not run when setUpModule raises.
        tearDownModule()
        raise


def tearDownModule():
    _STATE_PATCH.stop()
    shutil.rmtree(_CURSOR_STATE, ignore_errors=True)


CURSOR_OK = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "HI",
    "session_id": "scrubbed",
})


class TestCursorAliases(unittest.TestCase):
    def test_short_names(self):
        self.assertEqual(cursor_mod.alias_model("fable"), config_mod.FABLE_MODEL)
        self.assertEqual(cursor_mod.alias_model("sol"), "gpt-5.6-sol")
        self.assertEqual(cursor_mod.alias_model("Fable"), config_mod.FABLE_MODEL)
        self.assertEqual(cursor_mod.alias_model("SOL"), "gpt-5.6-sol")

    def test_verbatim_slugs(self):
        self.assertEqual(cursor_mod.alias_model("claude-fable-5"), "claude-fable-5")
        self.assertEqual(cursor_mod.alias_model("gpt-5.6-sol"), "gpt-5.6-sol")
        self.assertEqual(cursor_mod.alias_model("gpt-5.6-sol-high"),
                         "gpt-5.6-sol-high")
        self.assertEqual(cursor_mod.alias_model("composer-2.5"), "composer-2.5")

    def test_blank_uses_default(self):
        self.assertEqual(cursor_mod.alias_model(""), config_mod.DEFAULT_CURSOR_MODEL)
        self.assertEqual(cursor_mod.alias_model(None), config_mod.DEFAULT_CURSOR_MODEL)

    def test_envelope_model_stays_on_cursor_prefix(self):
        self.assertEqual(cursor_mod.envelope_model("sol"), "cursor/gpt-5.6-sol")
        self.assertEqual(cursor_mod.envelope_model("fable"),
                         "cursor/" + config_mod.FABLE_MODEL)
        self.assertEqual(cursor_mod.envelope_model("composer-2.5"),
                         "cursor/composer-2.5")
        self.assertEqual(resolve_route(cursor_mod.envelope_model("sol"))
                         ["provider"], "cursor")

    def test_default_is_not_a_family_seat(self):
        default = config_mod.DEFAULT_CURSOR_MODEL
        self.assertFalse(default.lower().startswith("grok"))
        self.assertNotIn(default.lower(), ("fable", "sol", "opus", "claude-fable-5",
                                           "gpt-5.6-sol"))


class TestCursorResolveBin(unittest.TestCase):
    def test_never_selects_agent(self):
        td = tempfile.mkdtemp(prefix="omni_agent_collision_")
        agent = os.path.join(td, "agent")
        with open(agent, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n")
        os.chmod(agent, stat.S_IRWXU)
        env = {"PATH": td, "OMNI_LLM_CURSOR_BIN": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(config_mod.ENV_CURSOR_BIN, None)
            self.assertIsNone(cursor_mod.resolve_bin())

    def test_override_must_be_executable(self):
        td = tempfile.mkdtemp(prefix="omni_cursor_bin_")
        missing = os.path.join(td, "nope")
        with mock.patch.dict(os.environ, {config_mod.ENV_CURSOR_BIN: missing}):
            self.assertIsNone(cursor_mod.resolve_bin())

    def test_override_wins(self):
        td = tempfile.mkdtemp(prefix="omni_cursor_bin_")
        path = os.path.join(td, "cursor-agent")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n")
        os.chmod(path, stat.S_IRWXU)
        with mock.patch.dict(os.environ, {config_mod.ENV_CURSOR_BIN: path,
                                          "PATH": td}):
            self.assertEqual(cursor_mod.resolve_bin(), os.path.realpath(path))

    def test_path_symlink_resolves_to_the_real_leaf(self):
        td = tempfile.mkdtemp(prefix="omni_cursor_link_")
        target = os.path.join(td, "versions", "v1", "cursor-agent")
        os.makedirs(os.path.dirname(target))
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n")
        os.chmod(target, stat.S_IRWXU)
        link = os.path.join(td, "bin", "cursor-agent")
        os.makedirs(os.path.dirname(link))
        os.symlink(target, link)
        with mock.patch.object(cursor_mod.shutil, "which", return_value=link), \
                mock.patch.dict(os.environ, {config_mod.ENV_CURSOR_BIN: ""},
                                clear=False):
            os.environ.pop(config_mod.ENV_CURSOR_BIN, None)
            self.assertEqual(cursor_mod.resolve_bin(), os.path.realpath(target))
            self.assertNotEqual(cursor_mod.resolve_bin(), link)


class TestCursorConsult(unittest.TestCase):
    def _run(self, fake, model="composer-2.5", prompt="hi", depth_value="0",
             effort=None, config_dir_env=None):
        env = dict(os.environ)
        env.pop(config_mod.ENV_CURSOR_CONFIG_DIR, None)
        env.pop(config_mod.CURSOR_CONFIG_DIR_ENV, None)
        if config_dir_env is not None:
            env[config_mod.ENV_CURSOR_CONFIG_DIR] = config_dir_env
        # Transport/argv fixtures pin the unchanged non-macOS contract.
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("providers.host_fence.platform.system", return_value="Linux"), \
                mock.patch.object(cursor_mod, "run_cli", fake), \
                mock.patch.object(cursor_mod, "resolve_bin",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(cursor_mod, "env_depth",
                                  return_value=int(depth_value)):
            _seed_cursor()
            return asyncio.run(
                cursor_mod.run_cursor_subprocess(prompt, model, 30,
                                                 effort=effort))

    def test_ask_mode_json_trust_no_force(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="fable")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "HI")
        self.assertEqual(env.get("model"), "cursor/" + config_mod.FABLE_MODEL)
        argv = fake.argv
        self.assertEqual(argv[0], "/fake/bin/cursor-agent")
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("--mode") + 1], "ask")
        self.assertEqual(argv[argv.index("--output-format") + 1], "json")
        self.assertIn("--trust", argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "enabled")
        workspace = argv[argv.index("--workspace") + 1]
        self.assertTrue(os.path.basename(workspace).startswith("cursor_cwd_"),
                        workspace)
        self.assertEqual(fake.kwargs.get("cwd"), workspace)
        self.assertEqual(
            fake.kwargs.get("env", {}).get(config_mod.ENV_DEPTH), "1")
        self.assertEqual(argv[argv.index("--model") + 1], config_mod.FABLE_MODEL)
        self.assertNotIn("--force", argv)
        self.assertNotIn("--yolo", argv)
        self.assertNotIn("--approve-mcps", argv)
        self.assertNotEqual(os.path.basename(argv[0]), "agent")

    def test_sol_alias_on_argv(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("model"), "cursor/gpt-5.6-sol-medium")
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         "gpt-5.6-sol-medium")
        self.assertEqual(resolve_route(env["model"])["provider"], "cursor")

    def test_config_dir_is_isolated_per_run(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake)
        self.assertTrue(env.get("ok"), env)
        config_dir = fake.kwargs.get("env", {}).get(
            config_mod.CURSOR_CONFIG_DIR_ENV)
        self.assertTrue(config_dir, fake.kwargs.get("env"))
        self.assertTrue(os.path.basename(config_dir).startswith("cursor_cfg_"),
                        config_dir)
        # the human's ~/.cursor is never the target, and the config dir is not
        # the workspace (cursor-agent writes state into it)
        self.assertNotEqual(
            os.path.realpath(config_dir),
            os.path.realpath(os.path.expanduser("~/.cursor")))
        self.assertNotEqual(config_dir, fake.kwargs.get("cwd"))
        self.assertNotIn(config_dir, fake.argv)
        # and it does not outlive the run
        self.assertFalse(os.path.isdir(config_dir), config_dir)

    def test_each_run_gets_a_fresh_config_dir(self):
        seen = []
        for _ in range(2):
            fake = FakeRunCli(stdout=CURSOR_OK)
            self._run(fake)
            seen.append(fake.kwargs["env"][config_mod.CURSOR_CONFIG_DIR_ENV])
        self.assertNotEqual(seen[0], seen[1])

    def test_config_dir_override_is_honoured_and_kept(self):
        with tempfile.TemporaryDirectory(prefix="omni_cursor_cfg_") as td:
            fake = FakeRunCli(stdout=CURSOR_OK)
            env = self._run(fake, config_dir_env=td)
            self.assertTrue(env.get("ok"), env)
            self.assertEqual(
                fake.kwargs["env"][config_mod.CURSOR_CONFIG_DIR_ENV], td)
            # a shared directory is the caller's: never deleted, on any path
            self.assertTrue(os.path.isdir(td))

    def test_caller_config_dir_survives_a_failed_run(self):
        with tempfile.TemporaryDirectory(prefix="omni_cursor_cfg_") as td:
            fake = FakeRunCli(stdout="", timed_out=True)
            env = self._run(fake, config_dir_env=td)
            self.assertEqual(env.get("reason"), "timeout")
            self.assertTrue(os.path.isdir(td))

    def test_relative_override_is_made_absolute(self):
        """A relative path would land inside the workspace this lane deletes."""
        fake = FakeRunCli(stdout=CURSOR_OK)
        self._run(fake, config_dir_env="omni-cursor-profile")
        config_dir = fake.kwargs["env"][config_mod.CURSOR_CONFIG_DIR_ENV]
        self.assertTrue(os.path.isabs(config_dir), config_dir)
        self.assertEqual(config_dir,
                         os.path.abspath("omni-cursor-profile"))
        self.assertFalse(
            config_dir.startswith(os.path.realpath(fake.kwargs["cwd"])),
            config_dir)

    def test_config_dir_override_expands_user(self):
        with mock.patch.dict(os.environ,
                             {config_mod.ENV_CURSOR_CONFIG_DIR: "~/.cursor"}):
            self.assertEqual(cursor_mod.resolve_config_dir(),
                             os.path.expanduser("~/.cursor"))

    def test_blank_config_dir_override_falls_back_to_temp(self):
        with mock.patch.dict(os.environ,
                             {config_mod.ENV_CURSOR_CONFIG_DIR: "   "}):
            self.assertIsNone(cursor_mod.resolve_config_dir())

    def test_inherited_cursor_config_dir_is_overridden(self):
        """A CURSOR_CONFIG_DIR already in the environment must not win."""
        fake = FakeRunCli(stdout=CURSOR_OK)
        with mock.patch.dict(
                os.environ,
                {config_mod.CURSOR_CONFIG_DIR_ENV: os.path.expanduser("~/.cursor")}):
            self._run(fake)
        config_dir = fake.kwargs["env"][config_mod.CURSOR_CONFIG_DIR_ENV]
        self.assertTrue(os.path.basename(config_dir).startswith("cursor_cfg_"),
                        config_dir)

    def test_refused_payload_creates_no_tempdirs(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        before = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                            "cursor_c*")))
        with mock.patch.object(cursor_mod, "refuse_payload",
                               return_value={"ok": False,
                                             "reason": "egress_refused"}):
            env = self._run(fake)
        self.assertEqual(env.get("reason"), "egress_refused")
        self.assertEqual(fake.calls, [])
        self.assertEqual(
            set(glob.glob(os.path.join(tempfile.gettempdir(),
                                       "cursor_c*"))), before)

    def test_recursion_guard(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, depth_value="1")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "recursion_guard")
        self.assertEqual(fake.calls, [])

    def test_cli_missing(self):
        with mock.patch.object(cursor_mod, "resolve_bin", return_value=None), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0):
            env = asyncio.run(
                cursor_mod.run_cursor_subprocess("hi", "composer-2.5", 30))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "cli_missing")
        self.assertIn("cursor-agent", env.get("detail", ""))

    def test_effort_is_mapped_onto_the_model_slug(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol", effort="xhigh")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         "gpt-5.6-sol-xhigh")
        self.assertEqual(env.get("model"), "cursor/gpt-5.6-sol-xhigh")
        self.assertEqual(resolve_route(env["model"])["provider"], "cursor")
        # no flag ever reaches the CLI
        self.assertNotIn("--effort", fake.argv)
        self.assertNotIn("--reasoning-effort", fake.argv)

    def test_sol_omitted_effort_uses_default_medium(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         "gpt-5.6-sol-medium")
        self.assertEqual(env.get("model"), "cursor/gpt-5.6-sol-medium")

    def test_sol_explicit_high_still_maps(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol", effort="high")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         "gpt-5.6-sol-high")

    def test_at_effort_shorthand_on_alias_and_slug(self):
        for model, slug in (
            ("sol@high", "gpt-5.6-sol-high"),
            ("gpt-5.6-sol@xhigh", "gpt-5.6-sol-xhigh"),
        ):
            fake = FakeRunCli(stdout=CURSOR_OK)
            env = self._run(fake, model=model)
            self.assertTrue(env.get("ok"), env)
            self.assertEqual(fake.argv[fake.argv.index("--model") + 1], slug)
            self.assertEqual(env.get("model"), "cursor/%s" % slug)

    def test_at_effort_and_flag_conflict_is_loud(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol@high", effort="low")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_effort")
        self.assertEqual(fake.calls, [])

    def test_at_bogus_effort_names_the_flag(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol@bogus")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_effort")
        self.assertIn("bogus", env.get("detail", ""))
        self.assertEqual(fake.calls, [])

    def test_at_effort_on_model_without_variants_is_loud(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="composer-2.5@high")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "effort_already_encoded")
        self.assertEqual(fake.calls, [])

    def test_effort_token_is_case_insensitive(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="fable", effort="  MAX ")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         config_mod.FABLE_MODEL + "-max")

    def test_omitted_effort_leaves_argv_untouched(self):
        # fable is not in CURSOR_DEFAULT_EFFORT; sol is the exception.
        plain = FakeRunCli(stdout=CURSOR_OK)
        self._run(plain, model="fable")
        blank = FakeRunCli(stdout=CURSOR_OK)
        self._run(blank, model="fable", effort="   ")
        self.assertEqual(plain.argv[plain.argv.index("--model") + 1],
                         config_mod.FABLE_MODEL)

        def scrub(argv):
            argv = list(argv)
            argv[argv.index("--workspace") + 1] = "<workspace>"
            return argv

        self.assertEqual(scrub(blank.argv), scrub(plain.argv))

    def test_effort_on_a_model_without_variants_is_loud(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="composer-2.5", effort="high")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "effort_already_encoded")
        self.assertEqual(fake.calls, [])

    def test_effort_on_an_already_encoded_slug_is_loud(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="gpt-5.6-sol-xhigh", effort="low")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "effort_already_encoded")
        self.assertEqual(fake.calls, [])

    def test_unknown_effort_token_is_loud(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="sol", effort="turbo")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_effort")
        self.assertEqual(fake.calls, [])

    def test_effort_error_is_not_masked_by_cli_missing(self):
        with mock.patch.object(cursor_mod, "resolve_bin", return_value=None), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0):
            env = asyncio.run(cursor_mod.run_cursor_subprocess(
                "hi", "composer-2.5", 30, effort="high"))
        self.assertEqual(env.get("reason"), "effort_already_encoded")

    def test_envelope_model_redispatches_at_the_same_strength(self):
        first = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(first, model="sol", effort="xhigh")
        self.assertEqual(env.get("model"), "cursor/gpt-5.6-sol-xhigh")
        # what a caller would do with that envelope: dispatch it again, no effort
        again = FakeRunCli(stdout=CURSOR_OK)
        route = resolve_route(env["model"])
        env2 = self._run(again, model=route["model"])
        self.assertTrue(env2.get("ok"), env2)
        self.assertEqual(again.argv[again.argv.index("--model") + 1],
                         "gpt-5.6-sol-xhigh")
        self.assertEqual(env2.get("model"), env.get("model"))

    def test_fast_twin_is_never_silently_swapped(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="gpt-5.6-sol-fast", effort="xhigh")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_model")
        self.assertEqual(fake.calls, [])

    def test_encoded_none_slug_is_named_as_already_encoded(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="gpt-5.6-sol-none", effort="low")
        self.assertFalse(env.get("ok"), env)
        self.assertIn("already encoded", env.get("detail", ""))

    def test_family_token_ceiling_at_the_subprocess_boundary(self):
        ok_run = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(ok_run, model="claude-opus-5", effort="high")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(ok_run.argv[ok_run.argv.index("--model") + 1],
                         "claude-opus-5-high")
        over = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(over, model="claude-opus-5", effort="max")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "unsupported_effort")
        self.assertEqual(over.calls, [])

    def test_irregular_slug_spellings_on_argv(self):
        for model, effort, slug in (
            ("gpt-5.5", "xhigh", "gpt-5.5-extra-high"),
            ("claude-4.6-opus-thinking", "max", "claude-4.6-opus-max-thinking"),
        ):
            fake = FakeRunCli(stdout=CURSOR_OK)
            env = self._run(fake, model=model, effort=effort)
            self.assertTrue(env.get("ok"), env)
            self.assertEqual(fake.argv[fake.argv.index("--model") + 1], slug)
            self.assertEqual(env.get("model"), "cursor/%s" % slug)

    def test_effort_error_is_not_masked_by_recursion_guard(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, model="composer-2.5", effort="high",
                        depth_value="1")
        self.assertEqual(env.get("reason"), "effort_already_encoded")

    def test_inband_error_is_not_success(self):
        payload = json.dumps({
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "rate limited",
        })
        fake = FakeRunCli(stdout=payload)
        env = self._run(fake)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error")

    def test_prompt_on_stdin(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        env = self._run(fake, prompt="hello cursor")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(fake.kwargs.get("stdin_data"), "hello cursor")


class TestCursorRoute(unittest.TestCase):
    def test_fable_and_sol_stay_on_cursor(self):
        self.assertEqual(resolve_route("cursor/fable"),
                         {"provider": "cursor", "model": "fable"})
        self.assertEqual(resolve_route("cursor/sol"),
                         {"provider": "cursor", "model": "sol"})
        self.assertEqual(resolve_route("cursor/claude-fable-5"),
                         {"provider": "cursor", "model": "claude-fable-5"})
        self.assertEqual(resolve_route("cursor/gpt-5.6-sol"),
                         {"provider": "cursor", "model": "gpt-5.6-sol"})

    def test_gpt_prefix_does_not_steal_sol(self):
        self.assertNotEqual(resolve_route("cursor/gpt-5.6-sol")["provider"],
                            "openai")
        self.assertEqual(resolve_route("gpt-5.6-sol")["provider"], "openai")

    def test_grok_prefix_does_not_steal(self):
        self.assertEqual(resolve_route("cursor/grok-4.6")["provider"], "cursor")
        self.assertEqual(resolve_route("grok-4.6")["provider"], "grok")

    def test_bare_cursor_is_not_hijacked(self):
        self.assertNotEqual(resolve_route("cursor")["provider"], "cursor")

    def test_route_to_provider_aliases_before_spawn(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        with mock.patch.object(cursor_mod, "run_cli", fake), \
                mock.patch.object(cursor_mod, "resolve_bin",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0), \
                depth(None):
            env = asyncio.run(dispatch_mod.route_to_provider(
                "hi", "cursor/fable", 30, None, "default"))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("model"), "cursor/" + config_mod.FABLE_MODEL)
        self.assertEqual(fake.argv[fake.argv.index("--model") + 1],
                         config_mod.FABLE_MODEL)
        self.assertEqual(resolve_route(env["model"])["provider"], "cursor")


class TestCursorMetricsSeat(unittest.TestCase):
    def test_sol_envelope_and_metrics_stay_on_cursor(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        td = tempfile.mkdtemp(prefix="omni_cursor_metrics_")
        with mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": td,
            "OMNI_LLM_METRICS": "1",
        }, clear=False), \
                mock.patch.object(cursor_mod, "run_cli", fake), \
                mock.patch.object(cursor_mod, "resolve_bin",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0), \
                depth(None):
            os.environ.pop("OMNI_EGRESS_GATE", None)
            _seed_cursor()
            env = asyncio.run(dispatch_mod.call_tool("llm_generate", {
                "prompt": "hi",
                "model": "cursor/sol",
            }))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("model"), "cursor/gpt-5.6-sol-medium")
        self.assertEqual(resolve_route(env["model"])["provider"], "cursor")
        # After call_tool the attempt ContextVar is gone; seat falls back to
        # envelope.model. Prefixing that field is what keeps the seat on cursor.
        seat = dispatch_mod._seat_for_result(env, "cursor/sol", "default")
        self.assertEqual(seat.get("provider"), "cursor", seat)
        self.assertEqual(seat.get("model"), "cursor/gpt-5.6-sol-medium")


class TestCursorUsage(unittest.TestCase):
    def test_json_result_text_usage_unavailable(self):
        text, env = extract_cli_result("cursor", CURSOR_OK)
        self.assertEqual(text, "HI")
        self.assertEqual(env["source"], "unavailable")
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.assertIsNone(env[key], key)

    def test_normalize_none_is_unavailable(self):
        env = normalize_usage("cursor", None)
        self.assertEqual(env["source"], "unavailable")

    def test_inband_is_error(self):
        stdout = json.dumps({
            "type": "result", "is_error": True, "result": "nope",
        })
        reason, detail = inband_error("cursor", stdout)
        self.assertEqual(reason, "runtime_error")
        self.assertIn("nope", detail)


if __name__ == "__main__":
    unittest.main()
