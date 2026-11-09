#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for account_profile login selection (issue 72).

No network. No real CLIs. Fake subprocess records argv and env.
"""
import asyncio
import io
import json
import os
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-profiles-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import agent_lane
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import cursor as cursor_mod
from providers import dispatch as dispatch_mod
from providers import jobs
from providers import kernel_sandbox
from providers.config import CURSOR_CONFIG_DIR_ENV, ENV_CURSOR_CONFIG_DIR, ENV_DEPTH
from providers.errors import classify_cli_error
from providers.profile_check import add_profile, persist_env_line, profiles_env_path
from providers.profiles import (
    check,
    child_env_for_profile,
    list_profiles,
    profile_ring,
    resolve_profile,
    validate_profile_ring_env,
)
from tests.test_agent import FakeRunCli, depth, patched_lane


CURSOR_OK = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "HI",
    "session_id": "scrubbed",
})

AMBIENT_CLAUDE = "/tmp/omni-profile-claude-ambient"
# Host fence passthrough for tests that mock run_cli: prepare_host_fence as
# bound in providers.codex (patching host_fence.* would not reach it).
_PASS_FENCE = lambda argv, **kw: (argv, kw.get("env") or {}, {})
CURSOR_KEY = "alt-cursor-key"


CLAUDE_OK = json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "PONG",
    "usage": {},
})

CLAUDE_NOT_LOGGED_IN = json.dumps({
    "type": "result",
    # Live shape (claude 2026-09-10): subtype is "success" even here, so a
    # detail that prefers subtype over result would read "success".
    "subtype": "success",
    "is_error": True,
    "result": "Not logged in · Please run /login",
}, separators=(",", ":"), ensure_ascii=False)


_REAL_WHICH = shutil.which


def _codex_which(name, *a, **k):
    base = os.path.basename(str(name))
    if base == "codex":
        return "/fake/bin/codex"
    return _REAL_WHICH(name)


def _claude_which(name, *a, **k):
    base = os.path.basename(str(name))
    if base == "claude":
        return "/fake/bin/claude"
    return _REAL_WHICH(name)


class TestResolveProfile(unittest.TestCase):
    def test_default_is_empty_overrides(self):
        for value in (None, "", "default"):
            self.assertEqual(resolve_profile("codex", value), ({}, None), value)
            self.assertEqual(resolve_profile("grok", value), ({}, None), value)

    def test_bad_name_is_loud(self):
        overrides, err = resolve_profile("codex", "Bad-Name")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")

    def test_codex_requires_existing_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODEX_HOME_NOPE", None)
            overrides, err = resolve_profile("codex", "nope")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("CODEX_HOME_NOPE", err["detail"])

    def test_codex_maps_realpath(self):
        d = tempfile.mkdtemp(prefix="omni_prof_codex_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": d}, clear=False):
            overrides, err = resolve_profile("codex", "alt")
        self.assertIsNone(err)
        self.assertEqual(list(overrides), ["CODEX_HOME"])
        self.assertEqual(overrides["CODEX_HOME"], os.path.realpath(d))

    def test_codex_direct_child_of_home_is_refused(self):
        # The reachable hole (issue #102): ~/.ssh was a valid codex profile,
        # and providers/codex.py hands CODEX_HOME to the fence as BOTH an
        # extra read path and an extra write path — so a valid profile was a
        # write allowance for whatever it named.
        home = tempfile.mkdtemp(prefix="omni_prof_codex_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        sensitive = os.path.join(home, ".ssh")
        os.mkdir(sensitive)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ,
                                {"CODEX_HOME_BAD": sensitive}):
            overrides, err = resolve_profile("codex", "bad")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("inside the home tree", err["detail"])
        self.assertIn("~/.codex-<name>", err["detail"])
        self.assertIn("CODEX_HOME_BAD", err["detail"])

    def test_default_dotcodex_cannot_be_a_profile(self):
        home = tempfile.mkdtemp(prefix="omni_prof_codex_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        default = os.path.join(home, ".codex")
        os.mkdir(default)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CODEX_HOME_BAD": default}):
            overrides, err = resolve_profile("codex", "bad")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("default login directory", err["detail"])

    def test_codex_grandchild_of_home_is_refused(self):
        # A "no direct children" rule waves ~/Library/Keychains through: its
        # parent is ~/Library, not home. Depth is not a defence, and a config
        # directory is not a reason to write-allow an arbitrary home subtree.
        home = tempfile.mkdtemp(prefix="omni_prof_codex_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        deep = os.path.join(home, "Library", "Keychains")
        os.makedirs(deep)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CODEX_HOME_BAD": deep}):
            overrides, err = resolve_profile("codex", "bad")
        self.assertEqual(overrides, {})
        self.assertIn("inside the home tree", err["detail"])

    def test_codex_case_variant_of_home_is_refused(self):
        # The default macOS volume is case-insensitive, so /users/alice/.ssh
        # and /Users/alice/.ssh are one directory. A case-sensitive string
        # comparison made them two answers, and the lower-cased spelling
        # walked past every guard.
        home = tempfile.mkdtemp(prefix="omni_prof_codex_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        sensitive = os.path.join(home, ".ssh")
        os.mkdir(sensitive)
        variant = os.path.join(home.upper(), ".ssh")
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CODEX_HOME_BAD": variant}):
            overrides, err = resolve_profile("codex", "bad")
        self.assertEqual(overrides, {}, "a case variant walked past the guard")
        self.assertIn("inside the home tree", err["detail"])

    def test_dotcodex_hyphen_name_under_home_is_accepted(self):
        # The admitted shape, so the guard refuses the dangerous cases
        # without refusing per-account profiles outright.
        home = tempfile.mkdtemp(prefix="omni_prof_codex_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        alt = os.path.join(home, ".codex-alt")
        os.mkdir(alt)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CODEX_HOME_ALT": alt}):
            overrides, err = resolve_profile("codex", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CODEX_HOME"], os.path.realpath(alt))

    def test_cursor_key_and_config_dir(self):
        d = tempfile.mkdtemp(prefix="omni_prof_cur_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CURSOR_API_KEY_ALT": CURSOR_KEY},
                             clear=False):
            os.environ.pop("OMNI_LLM_CURSOR_CONFIG_DIR_ALT", None)
            overrides, err = resolve_profile("cursor", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides, {"CURSOR_API_KEY": CURSOR_KEY})
        with mock.patch.dict(os.environ, {
            "OMNI_LLM_CURSOR_CONFIG_DIR_ALT": d,
            "CURSOR_API_KEY_ALT": CURSOR_KEY,
        }, clear=False):
            overrides, err = resolve_profile("cursor", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["OMNI_LLM_CURSOR_CONFIG_DIR"], d)
        self.assertEqual(overrides["CURSOR_API_KEY"], CURSOR_KEY)

    def test_cursor_grandchild_of_home_is_refused(self):
        # Same producer rule as CODEX_HOME: ~/Library/Keychains is not a
        # config directory. A "no direct children" rule waves it through.
        home = tempfile.mkdtemp(prefix="omni_prof_cursor_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        deep = os.path.join(home, "Library", "Keychains")
        os.makedirs(deep)
        name = uuid.uuid4().hex
        var = "OMNI_LLM_CURSOR_CONFIG_DIR_%s" % name.upper()
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {var: deep}):
            overrides, err = resolve_profile("cursor", name)
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("inside the home tree", err["detail"])
        self.assertIn("~/.cursor-<name>", err["detail"])
        self.assertIn(var, err["detail"])

    def test_cursor_tilde_dotdir_is_admitted(self):
        home = tempfile.mkdtemp(prefix="omni_prof_cursor_tilde_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        token = uuid.uuid4().hex
        alt = os.path.join(home, ".cursor-%s" % token)
        os.mkdir(alt)
        var = "OMNI_LLM_CURSOR_CONFIG_DIR_%s" % token.upper()
        tilde = "~/.cursor-%s" % token
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"HOME": home, var: tilde}):
            overrides, err = resolve_profile("cursor", token)
        self.assertIsNone(err, err)
        self.assertIn("OMNI_LLM_CURSOR_CONFIG_DIR", overrides)

    def test_unsupported_roads(self):
        for road in ("pi", "ollama", "lmstudio"):
            overrides, err = resolve_profile(road, "alt")
            self.assertEqual(overrides, {})
            self.assertEqual(err["reason"], "unsupported_profile", road)
            self.assertIn("road %s has no profile mechanism" % road, err["detail"])

    def test_claude_missing_var_is_loud(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_CONFIG_DIR_NOPE", None)
            overrides, err = resolve_profile("claude", "nope")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("CLAUDE_CONFIG_DIR_NOPE", err["detail"])

    def test_claude_not_a_dir_is_loud(self):
        fd, path = tempfile.mkstemp(prefix="omni_prof_claude_file_")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": path},
                             clear=False):
            overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("CLAUDE_CONFIG_DIR_ALT", err["detail"])

    def test_claude_maps_realpath(self):
        d = tempfile.mkdtemp(prefix="omni_prof_claude_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d},
                             clear=False):
            overrides, err = resolve_profile("claude", "alt")
        self.assertIsNone(err)
        self.assertEqual(list(overrides), ["CLAUDE_CONFIG_DIR"])
        self.assertEqual(overrides["CLAUDE_CONFIG_DIR"], os.path.realpath(d))

    def test_child_env_default_is_same_object(self):
        base = {"HOME": "/x", "CLAUDE_CONFIG_DIR": "/c"}
        self.assertIs(child_env_for_profile(base, {}), base)

    def test_child_env_default_scrubs_siblings_when_asked(self):
        base = {
            "HOME": "/x",
            "CLAUDE_CONFIG_DIR": "/c",
            "CLAUDE_CONFIG_DIR_A": "/a",
            "CODEX_HOME_ALT": "/alt",
            "CURSOR_API_KEY_B": "k",
            "OMNI_LLM_CURSOR_CONFIG_DIR_B": "/cur",
            "PATH": "/bin",
        }
        out = child_env_for_profile(base, {}, scrub_siblings=True)
        self.assertIsNot(out, base)
        self.assertEqual(out["CLAUDE_CONFIG_DIR"], "/c")
        self.assertEqual(out["PATH"], "/bin")
        self.assertNotIn("CLAUDE_CONFIG_DIR_A", out)
        self.assertNotIn("CODEX_HOME_ALT", out)
        self.assertNotIn("CURSOR_API_KEY_B", out)
        self.assertNotIn("OMNI_LLM_CURSOR_CONFIG_DIR_B", out)
        self.assertIs(child_env_for_profile(base, {}), base)

    def test_child_env_ring_var_alone_does_not_scrub(self):
        base = {
            "CLAUDE_CONFIG_DIR": "/c",
            "CLAUDE_CONFIG_DIR_A": "/a",
            "CODEX_HOME_ALT": "/keep-codex-sibling",
            "OMNI_LLM_PROFILES_CLAUDE": "default,a",
            "PATH": "/bin",
        }
        self.assertIs(child_env_for_profile(base, {}), base)

    def test_child_env_strips_default_login_and_siblings(self):
        base = {
            "CODEX_HOME": "/default-codex",
            "CLAUDE_CONFIG_DIR": AMBIENT_CLAUDE,
            "CURSOR_API_KEY": "keep-out",
            "CURSOR_CONFIG_DIR": "/default-cursor",
            "OMNI_LLM_CURSOR_CONFIG_DIR": "/opt-in",
            "CODEX_HOME_ALT": "/alt",
            "PATH": "/bin",
        }
        out = child_env_for_profile(base, {"CODEX_HOME": "/profile"})
        self.assertEqual(out["CODEX_HOME"], "/profile")
        self.assertNotIn("CODEX_HOME_ALT", out)
        self.assertNotIn("CLAUDE_CONFIG_DIR", out)
        self.assertNotIn("CURSOR_API_KEY", out)
        self.assertNotIn("CURSOR_CONFIG_DIR", out)
        self.assertEqual(out["PATH"], "/bin")


class TestProfileRingParse(unittest.TestCase):
    def tearDown(self):
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_PROFILES_"):
                os.environ.pop(key, None)
        os.environ.pop("OMNI_LLM_PROFILE_DISCOVERY", None)

    def test_unset_is_no_ring(self):
        os.environ.pop("OMNI_LLM_PROFILES_CLAUDE", None)
        names, err = profile_ring("claude")
        self.assertIsNone(names)
        self.assertIsNone(err)

    def test_list_order(self):
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "a, b, default"
        names, err = profile_ring("claude")
        self.assertIsNone(err)
        self.assertEqual(names, ["a", "b", "default"])

    def test_empty_string_is_profile_config(self):
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = ""
        names, err = profile_ring("claude")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", err["detail"])

    def test_empty_entry_is_profile_config(self):
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "a,,b"
        names, err = profile_ring("claude")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", err["detail"])

    def test_duplicate_is_profile_config(self):
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "a,a"
        names, err = profile_ring("codex")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CODEX", err["detail"])

    def test_auto_is_reserved(self):
        os.environ.pop("OMNI_LLM_PROFILE_DISCOVERY", None)
        os.environ["OMNI_LLM_PROFILES_CURSOR"] = "auto"
        names, err = profile_ring("cursor")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("discovery", err["detail"])
        self.assertIn("OMNI_LLM_PROFILE_DISCOVERY=1", err["detail"])
        self.assertIn("OMNI_LLM_PROFILES_CURSOR", err["detail"])

    def test_auto_mixed_is_malformed(self):
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "auto,b"
        names, err = profile_ring("codex")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("malformed", err["detail"])
        self.assertIn("OMNI_LLM_PROFILES_CODEX", err["detail"])

    def test_auto_discovery_on_usable_names(self):
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "auto"
        rows = [
            {"road": "codex", "name": "default", "state": "identity_cached",
             "problems": []},
            {"road": "codex", "name": "b", "state": "identity_cached",
             "problems": []},
            {"road": "codex", "name": "c", "state": "not_logged_in",
             "problems": []},
        ]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows):
            names, err = profile_ring("codex")
        self.assertIsNone(err)
        self.assertEqual(names, ["default", "b"])

    def test_auto_empty_usable_is_profile_config(self):
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "auto"
        rows = [
            {"road": "claude", "name": "default", "state": "not_logged_in",
             "problems": []},
        ]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows):
            names, err = profile_ring("claude")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_CLAUDE", err["detail"])

    def test_auto_unconfigured_is_no_usable_profile(self):
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        os.environ["OMNI_LLM_PROFILES_CURSOR"] = "auto"
        rows = [
            {"road": "cursor", "name": "default", "state": "unconfigured",
             "problems": []},
        ]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows):
            names, err = profile_ring("cursor")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("no usable profile", err["detail"])
        self.assertIn("OMNI_LLM_PROFILES_CURSOR", err["detail"])

    def test_unknown_road_is_profile_config(self):
        names, err = profile_ring("pi")
        self.assertIsNone(names)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_PI", err["detail"])

    def test_validate_flags_unknown_env_var(self):
        os.environ["OMNI_LLM_PROFILES_PI"] = "a,b"
        err = validate_profile_ring_env()
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("OMNI_LLM_PROFILES_PI", err["detail"])

    def test_validate_lone_auto_discovery_off_is_loud(self):
        os.environ.pop("OMNI_LLM_PROFILE_DISCOVERY", None)
        os.environ["OMNI_LLM_PROFILES_CURSOR"] = "auto"
        err = validate_profile_ring_env()
        self.assertIsNotNone(err)
        self.assertEqual(err["reason"], "profile_config")
        self.assertIn("discovery", err["detail"])
        self.assertIn("OMNI_LLM_PROFILES_CURSOR", err["detail"])


class TestCodexConsultProfile(unittest.TestCase):
    def test_a_alt_sets_codex_home_and_strips_siblings(self):
        fake = FakeRunCli(out_file_text="PONG")
        d = tempfile.mkdtemp(prefix="omni_prof_a_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        env = dict(os.environ)
        env["CODEX_HOME_ALT"] = d
        env["CLAUDE_CONFIG_DIR"] = AMBIENT_CLAUDE
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(codex_mod, "run_cli", fake), \
                mock.patch.object(codex_mod.shutil, "which", _codex_which), \
                mock.patch.object(codex_mod, "prepare_host_fence",
                                  side_effect=_PASS_FENCE):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, account_profile="alt"))
        self.assertTrue(res.get("ok"), res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CODEX_HOME"), os.path.realpath(d))
        self.assertNotIn("CODEX_HOME_ALT", child)
        self.assertNotIn("CLAUDE_CONFIG_DIR", child)

    def test_b_fence_gets_extra_write_paths(self):
        fake = FakeRunCli(out_file_text="PONG")
        d = tempfile.mkdtemp(prefix="omni_prof_b_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        seen = {}

        def _verify(workdir=None, read_only=True, read_wall=False,
                    extra_read_paths=None, extra_write_paths=None, **_fence_kw):
            seen["verify_write"] = extra_write_paths
            seen["verify_read"] = list(extra_read_paths or [])
            return None

        def _wrap(argv, workdir, read_only, read_wall=False,
                  extra_read_paths=None, extra_write_paths=None, **_fence_kw):
            seen["wrap_write"] = extra_write_paths
            seen["wrap_read"] = list(extra_read_paths or [])
            return ["/usr/bin/sandbox-exec", "-p", "PROF"] + list(argv)

        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": d}, clear=False), \
                mock.patch.object(codex_mod, "run_cli", fake), \
                mock.patch.object(codex_mod.shutil, "which", _codex_which), \
                mock.patch.object(codex_mod.kernel_sandbox, "available",
                                  return_value=True), \
                mock.patch.object(codex_mod.kernel_sandbox, "verify_fence",
                                  _verify), \
                mock.patch.object(codex_mod.kernel_sandbox, "wrap_argv", _wrap), \
                mock.patch.object(codex_mod.kernel_sandbox, "is_wrapped",
                                  lambda argv: (
                                      bool(argv) and os.path.basename(argv[0])
                                      == "sandbox-exec")):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, account_profile="alt"))
        self.assertTrue(res.get("ok"), res)
        want = [os.path.realpath(d)]
        self.assertEqual(seen.get("verify_write"), want)
        self.assertEqual(seen.get("wrap_write"), want)
        self.assertIn(os.path.realpath(d), seen.get("verify_read") or [])
        self.assertIn(os.path.realpath(d), seen.get("wrap_read") or [])

    def test_c_default_leaves_ambient_untouched(self):
        fake = FakeRunCli(out_file_text="PONG")
        d = tempfile.mkdtemp(prefix="omni_prof_c_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        env = dict(os.environ)
        env["CODEX_HOME_ALT"] = d
        env["CLAUDE_CONFIG_DIR"] = AMBIENT_CLAUDE
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(codex_mod, "run_cli", fake), \
                mock.patch.object(codex_mod.shutil, "which", _codex_which), \
                mock.patch.object(codex_mod, "prepare_host_fence",
                                  side_effect=_PASS_FENCE):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, account_profile="default"))
        self.assertTrue(res.get("ok"), res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), AMBIENT_CLAUDE)
        self.assertEqual(child.get("CODEX_HOME_ALT"), d)
        self.assertEqual(child.get(ENV_DEPTH), str(codex_mod.env_depth() + 1))

    def test_bad_profile_before_cli_missing(self):
        with mock.patch.object(codex_mod.shutil, "which", return_value=None):
            res = asyncio.run(codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30, account_profile="Bad-Name"))
        self.assertEqual(res.get("reason"), "unsupported_profile")


class TestCursorConsultProfile(unittest.TestCase):
    def _run(self, fake, account_profile, extra_env):
        env = dict(os.environ)
        env.update(extra_env)
        env.pop(ENV_CURSOR_CONFIG_DIR, None)
        env.pop(CURSOR_CONFIG_DIR_ENV, None)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(cursor_mod, "run_cli", fake), \
                mock.patch.object(cursor_mod, "resolve_bin",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(cursor_mod, "env_depth", return_value=0):
            return asyncio.run(cursor_mod.run_cursor_subprocess(
                "hi", "composer-2.5", 30, account_profile=account_profile))

    def test_d_api_key_in_env_not_argv_or_envelope(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        res = self._run(fake, "alt", {"CURSOR_API_KEY_ALT": CURSOR_KEY})
        self.assertTrue(res.get("ok"), res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CURSOR_API_KEY"), CURSOR_KEY)
        joined = " ".join(str(tok) for tok in fake.argv)
        self.assertNotIn(CURSOR_KEY, joined)
        self.assertNotIn(CURSOR_KEY, json.dumps(res, default=str))

    def test_e_config_dir_override_is_kept(self):
        td = tempfile.mkdtemp(prefix="omni_prof_e_")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        fake = FakeRunCli(stdout=CURSOR_OK)
        res = self._run(fake, "alt", {ENV_CURSOR_CONFIG_DIR + "_ALT": td})
        self.assertTrue(res.get("ok"), res)
        used = fake.kwargs["env"][CURSOR_CONFIG_DIR_ENV]
        self.assertEqual(used, os.path.abspath(td))
        self.assertTrue(os.path.isdir(td))
        self.assertFalse(os.path.basename(used).startswith("cursor_cfg_"))


class TestLoudRefuseOtherRoads(unittest.TestCase):
    def test_f_route_refuses_without_spawn(self):
        spawned = []

        async def boom(*_a, **_k):
            spawned.append("spawn")
            raise AssertionError("must not spawn")

        with depth(None), \
                mock.patch.object(dispatch_mod.claude_mod, "run_claude_subprocess",
                                  boom), \
                mock.patch.object(dispatch_mod.grok_mod, "run_grok_subprocess",
                                  boom), \
                mock.patch.object(dispatch_mod.pi_mod, "run_pi_subprocess",
                                  boom):
            for model in ("claude/x", "pi/x"):
                res = asyncio.run(dispatch_mod.route_to_provider(
                    "hi", model, 30, None, "alt"))
                self.assertEqual(res.get("reason"), "unsupported_profile", model)
                self.assertFalse(res.get("ok"), model)
        self.assertEqual(spawned, [])

    def test_g_jobs_preflight_refuses_claude_alt(self):
        wd = tempfile.mkdtemp(prefix="omni_prof_g_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        with depth(None), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            refusal, _normalized, _route = jobs._preflight(
                "claude/x", wd, False, False, account_profile="alt")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["reason"], "unsupported_profile")

    def test_h_agent_lane_codex_passes_codex_home(self):
        fake = FakeRunCli()
        d = tempfile.mkdtemp(prefix="omni_prof_h_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wd = tempfile.mkdtemp(prefix="omni_prof_h_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": d}, clear=False), \
                depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            res = asyncio.run(agent_lane.run_agent(
                task="t", model="codex/y", workdir=wd, timeout_s=30,
                account_profile="alt"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(fake.kwargs["env"]["CODEX_HOME"], os.path.realpath(d))
        self.assertNotIn("CODEX_HOME_ALT", fake.kwargs["env"])

    def test_i_chain_records_unsupported_then_answers_on_codex(self):
        d = tempfile.mkdtemp(prefix="omni_prof_i_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        spawned = []

        async def fake_codex(*_a, **_k):
            spawned.append("codex")
            return {"ok": True, "text": "FROM-CODEX", "model": "y"}

        async def fake_claude(*_a, **_k):
            spawned.append("claude")
            return {"ok": True, "text": "FROM-CLAUDE"}

        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": d}, clear=False), \
                depth(None), \
                mock.patch.object(dispatch_mod, "_health_passive_probe"), \
                mock.patch.object(dispatch_mod, "_health_active",
                                  return_value=False), \
                mock.patch.object(dispatch_mod.codex_mod, "run_codex_subprocess",
                                  fake_codex), \
                mock.patch.object(dispatch_mod.claude_mod,
                                  "run_claude_subprocess", fake_claude):
            res = asyncio.run(dispatch_mod.dispatch_llm_request(
                "hi", "claude/x,codex/y", 30, None, "alt"))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(res.get("text"), "FROM-CODEX")
        self.assertEqual(res.get("tried"),
                         [{"model": "claude/x", "reason": "unsupported_profile"}])
        self.assertEqual(spawned, ["codex"])


class TestWriteAllowPaths(unittest.TestCase):
    def test_j_extra_write_paths_grows_by_one(self):
        d = tempfile.mkdtemp(prefix="omni_prof_j_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        base = kernel_sandbox._write_allow_paths(None, True)
        grown = kernel_sandbox._write_allow_paths(
            None, True, extra_write_paths=[d])
        self.assertEqual(len(grown), len(base) + 1)
        self.assertEqual(set(grown) - set(base), {os.path.realpath(d)})


class TestBackendsSnapshotProfiles(unittest.TestCase):
    def test_profiles_flag_per_road(self):
        snap = dispatch_mod.backends_snapshot()
        for name in ("codex", "cursor", "agy", "openai", "claude", "grok"):
            self.assertIs(snap[name]["profiles"], True, name)
        for name in ("pi", "ollama", "lmstudio"):
            self.assertIs(snap[name]["profiles"], False, name)


if __name__ == "__main__":
    unittest.main()


class TestCodexHomeGuard(unittest.TestCase):
    def test_codex_home_equal_to_protected_home_is_refused(self):
        from providers import kernel_sandbox
        from providers.profiles import resolve_profile
        home = kernel_sandbox.protected_home()
        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": home}):
            overrides, err = resolve_profile("codex", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")

    def test_codex_home_parent_of_protected_home_is_refused(self):
        from providers import kernel_sandbox
        from providers.profiles import resolve_profile
        parent = os.path.dirname(kernel_sandbox.protected_home())
        with mock.patch.dict(os.environ, {"CODEX_HOME_ALT": parent}):
            _overrides, err = resolve_profile("codex", "alt")
        self.assertEqual(err["reason"], "unsupported_profile")

    def test_codex_home_sibling_dir_is_accepted(self):
        from providers.profiles import resolve_profile
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"CODEX_HOME_ALT": d}):
            overrides, err = resolve_profile("codex", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CODEX_HOME"], os.path.realpath(d))


class TestClaudeConsultProfile(unittest.TestCase):
    def test_alt_sets_config_dir_and_strips_siblings(self):
        fake = FakeRunCli(stdout=CLAUDE_OK)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_a_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        env = dict(os.environ)
        env["CLAUDE_CONFIG_DIR_ALT"] = d
        env["CLAUDE_CONFIG_DIR"] = AMBIENT_CLAUDE
        env["CODEX_HOME"] = "/default-codex"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(claude_mod.shutil, "which", _claude_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30, account_profile="alt"))
        self.assertTrue(res.get("ok"), res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), os.path.realpath(d))
        self.assertNotIn("CLAUDE_CONFIG_DIR_ALT", child)
        self.assertNotIn("CODEX_HOME", child)

    def test_default_leaves_ambient_untouched(self):
        fake = FakeRunCli(stdout=CLAUDE_OK)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_c_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ambient = os.path.realpath(
            tempfile.mkdtemp(prefix="omni_prof_claude_ambient_"))
        self.addCleanup(shutil.rmtree, ambient, ignore_errors=True)
        env = dict(os.environ)
        env["CLAUDE_CONFIG_DIR_ALT"] = d
        env["CLAUDE_CONFIG_DIR"] = ambient
        # The fence allowlists the child environment (#138): the ambient
        # sibling reaches the child only because the operator names it.
        env["OMNI_LLM_CHILD_ENV_PASS"] = "CLAUDE_CONFIG_DIR_ALT"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(claude_mod.shutil, "which", _claude_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30, account_profile="default"))
            self.assertTrue(res.get("ok"), res)
            child = fake.kwargs.get("env") or {}
            self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), ambient)
            self.assertEqual(child.get("CLAUDE_CONFIG_DIR_ALT"), d)
            self.assertEqual(child.get(ENV_DEPTH), str(claude_mod.env_depth() + 1))

    def test_default_scrub_siblings_strips_profile_vars(self):
        fake = FakeRunCli(stdout=CLAUDE_OK)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_scrub_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ambient = os.path.realpath(
            tempfile.mkdtemp(prefix="omni_prof_claude_ambient_"))
        self.addCleanup(shutil.rmtree, ambient, ignore_errors=True)
        env = dict(os.environ)
        env["CLAUDE_CONFIG_DIR_ALT"] = d
        env["CLAUDE_CONFIG_DIR"] = ambient
        env["OMNI_LLM_PROFILES_CLAUDE"] = "default,alt"
        # The fence allowlists the child environment (#138): the ring variable
        # reaches the child only because the operator names it.
        env["OMNI_LLM_CHILD_ENV_PASS"] = "OMNI_LLM_PROFILES_CLAUDE"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(claude_mod.shutil, "which", _claude_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30, account_profile="default",
                scrub_siblings=True))
        self.assertTrue(res.get("ok"), res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), ambient)
        self.assertNotIn("CLAUDE_CONFIG_DIR_ALT", child)
        self.assertEqual(child.get("OMNI_LLM_PROFILES_CLAUDE"), "default,alt")

    def test_bad_profile_before_cli_missing(self):
        with mock.patch.object(claude_mod.shutil, "which", return_value=None), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30, account_profile="Bad-Name"))
        self.assertEqual(res.get("reason"), "unsupported_profile")

    def test_not_logged_in_rc1_is_auth_error_naming_var(self):
        self.assertEqual(
            classify_cli_error(CLAUDE_NOT_LOGGED_IN, "", 1), "auth_error")
        self.assertIn('{"type"', CLAUDE_NOT_LOGGED_IN)
        fake = FakeRunCli(returncode=1, stdout=CLAUDE_NOT_LOGGED_IN)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_auth_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d},
                             clear=False), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(claude_mod.shutil, "which", _claude_which), \
                mock.patch.object(claude_mod, "env_depth", return_value=0):
            res = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30, account_profile="alt"))
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "auth_error")
        detail = res.get("detail") or ""
        self.assertIn("CLAUDE_CONFIG_DIR_ALT", detail)
        self.assertIn("Not logged in", detail)
        self.assertNotIn('{"type"', detail)


class TestClaudeAgentLaneProfile(unittest.TestCase):
    def test_agent_lane_keeps_config_dir(self):
        fake = FakeRunCli(stdout=CLAUDE_OK)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_h_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wd = tempfile.mkdtemp(prefix="omni_prof_claude_h_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        policy = {
            "wrap_kw": dict(
                read_wall=True, extra_read_paths=(), extra_write_paths=(),
            ),
            "tmpdir": None,
            "vendor_state_writable": True,
            "binary": "/fake/bin/claude",
            "claude_config_root": os.path.realpath(d),
        }
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d},
                             clear=False), \
                depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value=policy), \
                mock.patch.object(kernel_sandbox, "verify_fence",
                                  return_value=None), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    lambda argv, *a, **k: (
                        ["/usr/bin/sandbox-exec", "-p", "P"] + list(argv)
                    ),
                ), \
                mock.patch.object(kernel_sandbox, "is_wrapped",
                                  lambda argv: True):
            res = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet", workdir=wd, timeout_s=30,
                account_profile="alt"))
        self.assertTrue(res.get("ok") or res.get("reason") in (
            "empty_response", "runtime_error",
        ), res)
        self.assertTrue(fake.calls, res)
        child = fake.kwargs.get("env") or {}
        self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), os.path.realpath(d))
        self.assertNotIn("CLAUDE_CONFIG_DIR_ALT", child)

    def test_not_logged_in_rc1_is_auth_error_readable(self):
        fake = FakeRunCli(returncode=1, stdout=CLAUDE_NOT_LOGGED_IN)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_h_auth_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wd = tempfile.mkdtemp(prefix="omni_prof_claude_h_auth_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        policy = {
            "wrap_kw": dict(
                read_wall=True, extra_read_paths=(), extra_write_paths=(),
            ),
            "tmpdir": None,
            "vendor_state_writable": True,
            "binary": "/fake/bin/claude",
            "claude_config_root": os.path.realpath(d),
        }
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d},
                             clear=False), \
                depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(kernel_sandbox, "available", return_value=True), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value=policy), \
                mock.patch.object(kernel_sandbox, "verify_fence",
                                  return_value=None), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    lambda argv, *a, **k: (
                        ["/usr/bin/sandbox-exec", "-p", "P"] + list(argv)
                    ),
                ), \
                mock.patch.object(kernel_sandbox, "is_wrapped",
                                  lambda argv: True):
            res = asyncio.run(agent_lane.run_agent(
                task="t", model="claude/sonnet", workdir=wd, timeout_s=30,
                account_profile="alt"))
        self.assertFalse(res.get("ok"), res)
        self.assertEqual(res.get("reason"), "auth_error")
        detail = res.get("detail") or ""
        self.assertIn("CLAUDE_CONFIG_DIR_ALT", detail)
        self.assertIn("Not logged in", detail)
        self.assertNotIn('{"type"', detail)

    def test_fence_paths_use_profile_root(self):
        d = tempfile.mkdtemp(prefix="omni_prof_claude_fence_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wd = tempfile.mkdtemp(prefix="omni_prof_claude_fence_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        reads, writes = kernel_sandbox.agent_fence_paths(
            "claude", workdir=wd, read_only=False, config_dir=d)
        rp = os.path.realpath(d)
        self.assertIn(rp, list(reads) + list(writes))
        session = kernel_sandbox.claude_session_dir(wd, config_dir=d)
        self.assertTrue(session.startswith(rp + os.sep), session)
        self.assertIn(session, writes)
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        self.assertNotEqual(rp, default_root)
        self.assertNotIn(default_root, reads)

    def test_jobs_preflight_accepts_claude_alt_when_dir_exists(self):
        wd = tempfile.mkdtemp(prefix="omni_prof_claude_job_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        d = tempfile.mkdtemp(prefix="omni_prof_claude_job_cfg_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d},
                             clear=False), \
                depth(None), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane, "resolve_agent_binary",
                                  return_value="/fake/bin/claude"):
            refusal, _normalized, _route = jobs._preflight(
                "claude/x", wd, False, False, account_profile="alt")
        if refusal is not None:
            self.assertNotEqual(refusal.get("reason"), "unsupported_profile",
                                refusal)


class TestCursorAgentLaneProfile(unittest.TestCase):
    def test_profile_override_does_not_pass_launcher_roots(self):
        fake = FakeRunCli(stdout=CURSOR_OK)
        d = tempfile.mkdtemp(prefix="omni_prof_cursor_lane_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wd = tempfile.mkdtemp(prefix="omni_prof_cursor_lane_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        policy = {
            "wrap_kw": dict(
                read_wall=True, extra_read_paths=(), extra_write_paths=(),
            ),
            "tmpdir": None,
            "vendor_state_writable": True,
            "binary": "/fake/bin/cursor-agent",
        }
        seen = {}
        real_apply = agent_lane._apply_kernel_fence

        def capture_apply(*args, **kwargs):
            seen["kwargs"] = dict(kwargs)
            return real_apply(*args, **kwargs)

        with mock.patch.dict(os.environ, {
                "OMNI_LLM_CURSOR_CONFIG_DIR_ALT": d,
            }, clear=False), \
                depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_workdir_fence",
                                  return_value=None), \
                mock.patch.object(agent_lane, "resolve_agent_binary",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(kernel_sandbox, "available",
                                  return_value=True), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value=policy), \
                mock.patch.object(kernel_sandbox, "verify_fence",
                                  return_value=None), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    lambda argv, *a, **k: (
                        ["/usr/bin/sandbox-exec", "-p", "P"] + list(argv)
                    ),
                ), \
                mock.patch.object(kernel_sandbox, "is_wrapped",
                                  lambda argv: True), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=capture_apply):
            res = asyncio.run(agent_lane.run_agent(
                task="t", model="cursor/composer-2.5", workdir=wd,
                timeout_s=30, account_profile="alt"))
        self.assertTrue(res.get("ok") or res.get("reason") in (
            "empty_response", "runtime_error",
        ), res)
        self.assertNotIn("launcher_roots", seen.get("kwargs") or {})


class TestClaudeHomeGuard(unittest.TestCase):
    def test_claude_config_equal_to_protected_home_is_refused(self):
        home = kernel_sandbox.protected_home()
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": home}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("CLAUDE_CONFIG_DIR_ALT", err["detail"])

    def test_claude_config_parent_of_protected_home_is_refused(self):
        parent = os.path.dirname(kernel_sandbox.protected_home())
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": parent}):
            _overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(err["reason"], "unsupported_profile")

    def test_claude_config_sibling_dir_is_accepted(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CLAUDE_CONFIG_DIR"], os.path.realpath(d))

    def test_default_dotclaude_cannot_be_a_profile(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        default = os.path.join(home, ".claude")
        os.mkdir(default)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_BAD": default}):
            overrides, err = resolve_profile("claude", "bad")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("default login directory", err["detail"])
        self.assertIn("cannot be a profile", err["detail"])
        self.assertIn("CLAUDE_CONFIG_DIR_BAD", err["detail"])

    def test_dotclaude_hyphen_name_under_home_is_accepted(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        alt = os.path.join(home, ".claude-alt")
        os.mkdir(alt)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": alt}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CLAUDE_CONFIG_DIR"], os.path.realpath(alt))

    def test_dotclaudeX_under_home_is_refused(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        xyz = os.path.join(home, ".claudeX")
        os.mkdir(xyz)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_X": xyz}):
            overrides, err = resolve_profile("claude", "x")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn("CLAUDE_CONFIG_DIR_X", err["detail"])

    def test_temp_dir_outside_home_is_accepted(self):
        d = tempfile.mkdtemp(prefix="omni_prof_claude_outside_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": d}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CLAUDE_CONFIG_DIR"], os.path.realpath(d))

    def test_dotconfig_claude_under_home_is_refused(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        cfg = os.path.join(home, ".config", "claude")
        os.makedirs(cfg)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": cfg}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")
        self.assertIn(".claude-", err["detail"])

    def test_library_application_support_claude_under_home_is_refused(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        cfg = os.path.join(home, "Library", "Application Support", "Claude")
        os.makedirs(cfg)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {
                    "CLAUDE_CONFIG_DIR_ALT": cfg}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")

    def test_dotclaude_alt_under_patched_home_is_accepted(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        alt = os.path.join(home, ".claude-alt")
        os.mkdir(alt)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": alt}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertIsNone(err)
        self.assertEqual(overrides["CLAUDE_CONFIG_DIR"], os.path.realpath(alt))

    def test_claudecfg_direct_child_under_home_is_refused(self):
        home = tempfile.mkdtemp(prefix="omni_prof_claude_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        cfg = os.path.join(home, "claudecfg")
        os.mkdir(cfg)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=os.path.realpath(home)), \
                mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR_ALT": cfg}):
            overrides, err = resolve_profile("claude", "alt")
        self.assertEqual(overrides, {})
        self.assertEqual(err["reason"], "unsupported_profile")


SYNTH_OAUTH = {
    "emailAddress": "synth@example.test",
    "organizationName": "Synth Org",
    "seatTier": "pro",
    "userRateLimitTier": "default_claude_ai",
    "organizationRateLimitTier": "default",
    "hasExtraUsageEnabled": False,
    "accountUuid": "00000000-0000-4000-8000-000000000000",
}

_PROFILE_ENV_PREFIXES = (
    "CLAUDE_CONFIG_DIR_",
    "CODEX_HOME_",
    "CURSOR_API_KEY_",
    "OMNI_LLM_CURSOR_CONFIG_DIR_",
)


class _TempHomeCase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="omni_prof_reg_")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.state = os.path.join(self.home, "state")
        os.mkdir(self.state)
        self._patch = mock.patch.dict(os.environ, {
            "HOME": self.home,
            "OMNI_LLM_QUOTA_PROBE": "0",
            "OMNI_LLM_STATE_DIR": self.state,
        }, clear=False)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        os.environ.pop("OMNI_LLM_PROFILE_DISCOVERY", None)
        for key in list(os.environ):
            if key.startswith(_PROFILE_ENV_PREFIXES):
                os.environ.pop(key, None)

    def _write_claude(self, name, oauth, extra=None):
        d = os.path.join(self.home, ".claude-%s" % name)
        os.mkdir(d)
        os.chmod(d, 0o700)
        payload = dict(extra or {})
        if oauth is not None:
            payload["oauthAccount"] = oauth
        with open(os.path.join(d, ".claude.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return d

    def _row(self, road, name):
        return [r for r in list_profiles(roads=(road,)) if r.get("name") == name]


class TestListProfilesDiscovery(_TempHomeCase):
    def test_discovery_symlink_collision_and_off(self):
        a = self._write_claude("a", SYNTH_OAUTH, extra={"otherKey": "secret-other"})
        os.symlink(a, os.path.join(self.home, ".claude-b"))
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        rows = list_profiles(roads=("claude",))
        by_name = {r["name"]: r for r in rows}
        self.assertIn("default", by_name)
        self.assertEqual(by_name["a"]["source"], "discovered")
        self.assertEqual(by_name["a"]["identity"].get("emailAddress"),
                         "synth@example.test")
        self.assertNotIn("otherKey", by_name["a"]["identity"])
        self.assertEqual(by_name["b"]["source"], "discovered")
        self.assertIn("symlink", by_name["b"]["problems"])

        os.environ["CLAUDE_CONFIG_DIR_A"] = a
        rows = list_profiles(roads=("claude",))
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(by_name["a"]["source"], "env")
        self.assertIn("collision", by_name["a"]["problems"])
        self.assertEqual(by_name["a"]["identity"].get("emailAddress"),
                         "synth@example.test")

        os.environ.pop("OMNI_LLM_PROFILE_DISCOVERY", None)
        os.environ.pop("CLAUDE_CONFIG_DIR_A", None)
        names = {r["name"] for r in list_profiles(roads=("claude",))}
        self.assertNotIn("a", names)
        self.assertNotIn("b", names)
        self.assertIn("default", names)

    def test_codex_auth_suffix_never_leaks_tokens(self):
        d = os.path.join(self.home, ".codex-a")
        os.mkdir(d)
        os.chmod(d, 0o700)
        auth = {
            "tokens": {
                "account_id": "acct_synthetic_xyz789",
                "access_token": "MUST_NOT_LEAK_ACCESS",
                "refresh_token": "MUST_NOT_LEAK_REFRESH",
                "id_token": "MUST_NOT_LEAK_ID",
            },
            "OPENAI_API_KEY": "MUST_NOT_LEAK_KEY",
        }
        with open(os.path.join(d, "auth.json"), "w", encoding="utf-8") as handle:
            json.dump(auth, handle)
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        rows = self._row("codex", "a")
        self.assertEqual(len(rows), 1)
        blob = json.dumps(rows[0])
        self.assertEqual(rows[0]["identity"].get("account_id_suffix"), "xyz789")
        self.assertIn("xyz789", blob)
        for leak in ("MUST_NOT_LEAK_ACCESS", "MUST_NOT_LEAK_REFRESH",
                     "MUST_NOT_LEAK_ID", "MUST_NOT_LEAK_KEY",
                     "acct_synthetic_xyz789", "access_token", "refresh_token",
                     "id_token", "OPENAI_API_KEY"):
            self.assertNotIn(leak, blob)

    def test_default_reads_home_root_claude_json(self):
        default = os.path.join(self.home, ".claude")
        os.mkdir(default)
        os.chmod(default, 0o700)
        with open(os.path.join(self.home, ".claude.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"oauthAccount": SYNTH_OAUTH, "ignoreMe": "nope"}, handle)
        rows = self._row("claude", "default")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["identity"].get("emailAddress"),
                         "synth@example.test")
        self.assertNotIn("ignoreMe", rows[0]["identity"])
        self.assertEqual(rows[0]["state"], "identity_cached")

    def test_cursor_api_key_identity_does_not_leak_key(self):
        os.environ["CURSOR_API_KEY_ALT"] = "secret-cursor-key"
        rows = self._row("cursor", "alt")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "env")
        self.assertEqual(rows[0]["identity"], {"kind": "api-key"})
        self.assertIsNone(rows[0]["path"])
        blob = json.dumps(rows[0])
        self.assertNotIn("secret-cursor-key", blob)

    def test_cursor_default_unconfigured_is_named_state(self):
        rows = self._row("cursor", "default")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "unconfigured")
        self.assertEqual(rows[0]["problems"], [])


class TestCheckProfiles(_TempHomeCase):
    def test_no_oauth_is_not_logged_in(self):
        self._write_claude("a", None)
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        payload = check(self._row("claude", "a"), live=False)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["profiles"][0]["state"], "not_logged_in")
        self.assertIn("not_logged_in", payload["profiles"][0]["problems"])

    def test_oauth_without_live_is_identity_cached(self):
        self._write_claude("a", SYNTH_OAUTH)
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        payload = check(self._row("claude", "a"), live=False)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profiles"][0]["state"], "identity_cached")
        self.assertEqual(payload["profiles"][0]["problems"], [])

    def test_live_not_logged_in_is_keychain_missing(self):
        self._write_claude("a", SYNTH_OAUTH)
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"

        async def fake(*_a, **_k):
            return {"ok": False, "reason": "auth_error",
                    "detail": "Not logged in · Please run /login"}

        with mock.patch("providers.claude.run_claude_subprocess", fake):
            payload = check(self._row("claude", "a"), live=True)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["profiles"][0]["state"], "keychain_missing")
        self.assertIn("keychain_missing", payload["profiles"][0]["problems"])

    def test_live_ok_is_session_ok(self):
        self._write_claude("a", SYNTH_OAUTH)
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"

        async def fake(*_a, **_k):
            return {"ok": True, "text": "pong"}

        with mock.patch("providers.claude.run_claude_subprocess", fake):
            payload = check(self._row("claude", "a"), live=True)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profiles"][0]["state"], "session_ok")
        self.assertEqual(payload["profiles"][0]["problems"], [])

    def _write_codex(self, name, account_id="acct_synthetic_xyz789"):
        d = os.path.join(self.home, ".codex-%s" % name)
        os.mkdir(d)
        os.chmod(d, 0o700)
        with open(os.path.join(d, "auth.json"), "w", encoding="utf-8") as handle:
            json.dump({"tokens": {"account_id": account_id}}, handle)
        return d

    def test_cooling_quota_is_informational_and_live_still_runs(self):
        self._write_codex("a")
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        info = {"reached": True, "used_percent": 99, "resets_at": 1e12}
        live_calls = []

        def fake_live(road, name):
            live_calls.append((road, name))
            return {"ok": True}

        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)), \
                mock.patch("providers.profile_check._live_one", fake_live):
            payload = check(self._row("codex", "a"), live=True)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profiles"][0]["problems"], [])
        self.assertEqual(payload["profiles"][0]["cooldown"], "cooling")
        self.assertEqual(live_calls, [("codex", "a")])

    def test_strict_quota_fails_cooling_and_skips_live(self):
        self._write_codex("a")
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        info = {"reached": True, "used_percent": 99, "resets_at": 1e12}
        live_calls = []

        def fake_live(road, name):
            live_calls.append((road, name))
            return {"ok": True}

        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(info, None)), \
                mock.patch("providers.profile_check._live_one", fake_live):
            payload = check(self._row("codex", "a"), live=True,
                            strict_quota=True)
        self.assertFalse(payload["ok"])
        self.assertIn("quota", payload["profiles"][0]["problems"])
        self.assertEqual(payload["profiles"][0]["cooldown"], "cooling")
        self.assertEqual(live_calls, [])

    def test_unreadable_quota_is_always_a_problem(self):
        self._write_codex("a")
        os.environ["OMNI_LLM_PROFILE_DISCOVERY"] = "1"
        live_calls = []

        def fake_live(road, name):
            live_calls.append((road, name))
            return {"ok": True}

        with mock.patch("providers.codex_quota.read_rate_limits",
                        return_value=(None, {"reason": "runtime_error"})), \
                mock.patch("providers.profile_check._live_one", fake_live):
            payload = check(self._row("codex", "a"), live=True)
        self.assertFalse(payload["ok"])
        self.assertIn("quota", payload["profiles"][0]["problems"])
        self.assertEqual(live_calls, [])


class TestAddProfile(_TempHomeCase):
    def test_creates_0700_dir_and_login_command(self):
        env = add_profile("claude", "work")
        self.assertTrue(env.get("ok"), env)
        path = env["path"]
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o700)
        self.assertTrue(path.endswith(".claude-work"))
        self.assertIn("claude auth login", env["login_command"])
        self.assertIn("CLAUDE_CONFIG_DIR=", env["login_command"])

    def test_refuses_existing_name(self):
        first = add_profile("claude", "work")
        self.assertTrue(first.get("ok"), first)
        with open(os.path.join(first["path"], "keep"), "w", encoding="utf-8") as handle:
            handle.write("x")
        again = add_profile("claude", "work")
        self.assertFalse(again.get("ok"))
        self.assertEqual(again.get("reason"), "unsupported_profile")

    def test_persist_writes_one_line_per_var(self):
        env = add_profile("codex", "work", persist=True)
        self.assertTrue(env.get("ok"), env)
        path = profiles_env_path()
        self.assertTrue(os.path.isfile(path))
        persist_env_line(env["var"], env["path"])
        with open(path, encoding="utf-8") as handle:
            lines = [ln for ln in handle.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("export CODEX_HOME_WORK="))
        self.assertIn(env["path"], lines[0])
        self.assertEqual(env["mcp_env"], {"CODEX_HOME_WORK": env["path"]})
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_reuses_empty_dir_of_the_right_name(self):
        dest = os.path.join(self.home, ".claude-work")
        os.mkdir(dest)
        os.chmod(dest, 0o700)
        env = add_profile("claude", "work")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(os.path.realpath(dest), env["path"])
        self.assertTrue(os.path.isdir(dest))
        self.assertEqual(os.stat(dest).st_mode & 0o777, 0o700)

    def test_cursor_creates_nothing(self):
        env = add_profile("cursor", "work")
        self.assertTrue(env.get("ok"), env)
        self.assertFalse(env.get("created"))
        self.assertFalse(os.path.exists(os.path.join(self.home, ".cursor-work")))
        self.assertIn("CURSOR_API_KEY_WORK", env.get("detail") or "")

    def test_cursor_persist_is_loud_unsupported(self):
        env = add_profile("cursor", "work", persist=True)
        self.assertFalse(env.get("ok"))
        self.assertEqual(env.get("reason"), "unsupported_persist")
        self.assertEqual(
            env.get("detail"),
            "cursor profiles are CURSOR_API_KEY_<NAME>; nothing to persist")
        self.assertEqual(env.get("mcp_env"), {"CURSOR_API_KEY_WORK": ""})


class TestSchedulePlist(unittest.TestCase):
    def test_schedule_plist_contains_label_args_interval_log(self):
        from providers.profile_check import SCHEDULE_LABEL, schedule_plist
        argv = [
            "/abs/python3", "/abs/omni_cli.py",
            "health", "--probe", "--all-profiles", "--json",
        ]
        log_path = "/tmp/omni-state/logs/profiles-probe.log"
        text = schedule_plist(900, argv, log_path=log_path)
        self.assertIn(SCHEDULE_LABEL, text)
        self.assertIn("<key>ProgramArguments</key>", text)
        self.assertIn("/abs/python3", text)
        self.assertIn("/abs/omni_cli.py", text)
        self.assertIn("<integer>900</integer>", text)
        self.assertIn("<key>StandardOutPath</key>", text)
        self.assertIn(log_path, text)
        self.assertNotIn("identity", text)

    def test_schedule_plist_environment_variables(self):
        from providers.profile_check import schedule_plist
        fake_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(fake_dir, ignore_errors=True))
        fake_codex = os.path.join(fake_dir, "codex")
        with open(fake_codex, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(fake_codex, 0o755)

        def which(name, *args, **kwargs):
            if name == "codex":
                return fake_codex
            return None

        env = {
            "CODEX_HOME_B": "/tmp/codex-home-b",
            "CURSOR_API_KEY_B": "secret-cursor-key-b",
            "OMNI_LLM_CURSOR_CONFIG_DIR_B": "/tmp/cursor-config-b",
        }
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch("providers.profile_check.shutil.which",
                           side_effect=which), \
                mock.patch("sys.stderr", stderr):
            text = schedule_plist(
                900, ["/abs/python3", "/abs/omni_cli.py"],
                log_path="/tmp/profiles-probe.log")
        self.assertIn("<key>EnvironmentVariables</key>", text)
        self.assertIn("<key>PATH</key>", text)
        self.assertIn(fake_dir, text)
        self.assertIn("/usr/bin:/bin:/usr/sbin:/sbin", text)
        self.assertIn("CODEX_HOME_B", text)
        self.assertIn("/tmp/codex-home-b", text)
        self.assertIn("OMNI_LLM_CURSOR_CONFIG_DIR_B", text)
        self.assertNotIn("secret-cursor-key-b", text)
        self.assertNotIn("CURSOR_API_KEY_B", text)
        self.assertIn("not probed by the scheduled job", stderr.getvalue())

