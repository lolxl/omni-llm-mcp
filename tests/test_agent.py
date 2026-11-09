#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for the `omni agent` lane. No network, no real CLIs, no live git remotes.

Written against the FROZEN SPEC, not against an implementation: the units under
test (providers/agent_lane.py, providers/claude.py, the `agent` subcommand in
omni_cli.py, and the claude/* route + backends entry) may not exist yet. Missing
modules surface as ONE loud existence failure per unit plus granular skips, so a
partial implementation still produces a readable per-spec-point report.

Everything is faked at the seam the spec names:
- `run_cli` is patched at the MODULE level of the unit under test (agent_lane /
  providers.claude), so no subprocess is ever spawned for an agent CLI.
- `shutil.which` is patched to report codex/claude/grok as present, so command
  construction is asserted on this machine regardless of what is installed.
- git fixtures are local-only temp repos (init -b main, one empty commit,
  `git worktree add` for the linked-checkout case). No remote is ever contacted;
  the whole group skips when git is not on PATH.
"""
import asyncio
import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import omni_cli  # noqa: E402
from providers import codex as codex_mod  # noqa: E402
from providers import config as config_mod  # noqa: E402
from providers import dispatch as dispatch_mod  # noqa: E402
from providers import grok as grok_mod  # noqa: E402
from providers import agy as agy_mod  # noqa: E402
from providers import pi as pi_mod  # noqa: E402
from providers import sandbox_canary as sandbox_canary_mod  # noqa: E402
from providers.route import resolve_route  # noqa: E402

_REAL_WHICH = shutil.which
ENV_DEPTH_NAME = "OMNI_LLM_DEPTH"
AGENT_CLIS = ("codex", "claude", "grok", "pi")
HAS_GIT = _REAL_WHICH("git") is not None


def _try_import(name):
    """Import a not-yet-written module without killing collection of the whole suite."""
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001 - reported verbatim by the existence tests
        return None, "%s: %s" % (type(e).__name__, e)


agent_lane, AGENT_LANE_ERR = _try_import("providers.agent_lane")
claude_mod, CLAUDE_ERR = _try_import("providers.claude")
cursor_mod, CURSOR_ERR = _try_import("providers.cursor")
HAS_CLI_AGENT = hasattr(omni_cli, "run_agent")


_STATE_PATCH = None
_STATE_DIR = None


def setUpModule():
    # #168: a write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    import shutil
    _STATE_PATCH.stop()
    shutil.rmtree(_STATE_DIR, ignore_errors=True)



# --------------------------------------------------------------------------- #
# fakes / fixtures
# --------------------------------------------------------------------------- #
class FakeRunCli:
    """Async run_cli stand-in: records (argv, kwargs) and writes the `-o` reply file.

    stdout is deliberately DIFFERENT from the `-o` file body, so a codex test that
    reads "AGENT DONE" proves the file won over the banner on stdout.
    """

    def __init__(self, returncode=0, stdout="BANNER ON STDOUT", stderr="",
                 timed_out=False, out_file_text="AGENT DONE", signal=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.out_file_text = out_file_text
        self.signal = signal
        self.calls = []

    async def __call__(self, argv=None, **kwargs):
        argv = list(argv or [])
        self.calls.append((argv, dict(kwargs)))
        if "-o" in argv:
            path = argv[argv.index("-o") + 1]
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.out_file_text)
        return {"returncode": self.returncode, "stdout": self.stdout,
                "stderr": self.stderr, "timed_out": self.timed_out,
                "signal": self.signal}

    @property
    def argv(self):
        return self.calls[0][0]

    @property
    def kwargs(self):
        return self.calls[0][1]


def _fake_which(name, *a, **kw):
    base = os.path.basename(str(name))
    if base in AGENT_CLIS:
        return "/fake/bin/" + base
    return _REAL_WHICH(name, *a, **kw)


@contextlib.contextmanager
def patched_lane(fake):
    """Patch agent_lane.run_cli with `fake` and report every agent CLI as installed."""
    # Host wrapping may prefix argv on macOS; legacy vendor-only cases
    # explicitly pin Linux. Host admission also has dedicated tests.
    with mock.patch.object(agent_lane, "run_cli", fake), \
            mock.patch.object(shutil, "which", _fake_which):
        yield


@contextlib.contextmanager
def depth(value):
    """Set OMNI_LLM_DEPTH to `value`, or remove it entirely when value is None."""
    with mock.patch.dict(os.environ, {}):
        if value is None:
            os.environ.pop(ENV_DEPTH_NAME, None)
        else:
            os.environ[ENV_DEPTH_NAME] = value
        yield


def _flag_value(argv, flag):
    """The token following `flag`, or None when the flag is absent / trailing."""
    argv = list(argv)
    if flag not in argv:
        return None
    i = argv.index(flag)
    return argv[i + 1] if i + 1 < len(argv) else None


def _git_env():
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_GLOBAL": os.devnull,   # no user hooks / gpgsign / templates
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",        # never prompt, never reach a remote
        "GIT_AUTHOR_NAME": "omni test",
        "GIT_AUTHOR_EMAIL": "omni@test.invalid",
        "GIT_COMMITTER_NAME": "omni test",
        "GIT_COMMITTER_EMAIL": "omni@test.invalid",
    })
    return env


def git(cwd, *args):
    proc = subprocess.run(["git"] + list(args), cwd=cwd, env=_git_env(),
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError("git %s failed in %s: %s" % (" ".join(args), cwd, proc.stderr))
    return proc.stdout


class LaneTestCase(unittest.TestCase):
    """Shared temp-dir / argv helpers for every run_agent test."""

    def mkdir(self, prefix="omni_agent_"):
        """A realpath'd temp dir, so abspath(workdir) == workdir on macOS."""
        d = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def make_repo(self, branch="main"):
        """A checkout on `branch` with one empty commit. No remote, no global config."""
        d = self.mkdir("omni_agent_repo_")
        git(d, "init", "-b", branch)
        git(d, "config", "user.email", "omni@test.invalid")
        git(d, "config", "user.name", "omni test")
        git(d, "commit", "--allow-empty", "-m", "x")
        return d

    def run_agent(self, **kwargs):
        return asyncio.run(agent_lane.run_agent(**kwargs))

    def assertFlagValue(self, argv, flag, value):
        self.assertIn(flag, argv, "missing %s in %r" % (flag, argv))
        self.assertEqual(_flag_value(argv, flag), value,
                         "%s value in %r" % (flag, argv))

    def assertHasFlags(self, argv, *flags):
        for f in flags:
            self.assertIn(f, argv, "missing %s in %r" % (f, argv))

    def assertLacksFlags(self, argv, *flags):
        for f in flags:
            self.assertNotIn(f, argv, "unexpected %s in %r" % (f, argv))

    def assertRefusal(self, env, reason):
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), reason, env)
        self.assertIsInstance(env.get("detail"), str, env)


# --------------------------------------------------------------------------- #
# 0. existence — one loud failure per missing unit (the rest degrade to skips)
# --------------------------------------------------------------------------- #
class TestSpecUnitsExist(unittest.TestCase):
    def test_agent_lane_module_importable(self):
        self.assertIsNotNone(agent_lane, "providers/agent_lane.py: %s" % AGENT_LANE_ERR)
        self.assertTrue(callable(getattr(agent_lane, "run_agent", None)))
        self.assertTrue(hasattr(agent_lane, "run_cli"),
                        "agent_lane must import run_cli at module level (patch seam)")

    def test_claude_provider_module_importable(self):
        self.assertIsNotNone(claude_mod, "providers/claude.py: %s" % CLAUDE_ERR)
        self.assertTrue(callable(getattr(claude_mod, "run_claude_subprocess", None)))
        self.assertTrue(hasattr(claude_mod, "run_cli"),
                        "providers.claude must import run_cli at module level")

    def test_cursor_provider_module_importable(self):
        self.assertIsNotNone(cursor_mod, "providers/cursor.py: %s" % CURSOR_ERR)
        self.assertTrue(callable(getattr(cursor_mod, "run_cursor_subprocess", None)))
        self.assertTrue(hasattr(cursor_mod, "run_cli"),
                        "providers.cursor must import run_cli at module level")
        self.assertTrue(callable(getattr(cursor_mod, "resolve_bin", None)))
        agent_bin = shutil.which("agent")
        if agent_bin:
            self.assertNotEqual(cursor_mod.resolve_bin(), agent_bin)

    def test_omni_cli_exposes_run_agent(self):
        self.assertTrue(HAS_CLI_AGENT,
                        "omni_cli must import run_agent at module level (patch seam)")


# --------------------------------------------------------------------------- #
# 1. config / route / backends snapshot
# --------------------------------------------------------------------------- #
class TestConfigAndRouting(unittest.TestCase):
    def test_agent_lane_constants(self):
        self.assertEqual(getattr(config_mod, "DEFAULT_AGENT_MODEL", None),
                         "grok-4.6")
        self.assertEqual(getattr(config_mod, "ENV_DEPTH", None), ENV_DEPTH_NAME)

    def test_default_grok_model_is_the_runtime_fallback(self):
        default = getattr(config_mod, "DEFAULT_GROK_MODEL", None)
        self.assertEqual(default, "grok-4.6")
        self.assertIs(agent_lane.DEFAULT_GROK_MODEL, config_mod.DEFAULT_GROK_MODEL)
        fake = FakeRunCli(stdout=json.dumps({"text": "OK", "usage": {}}))
        with mock.patch.object(grok_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which), \
                mock.patch.object(grok_mod, "_consult_result",  # json fake, not a stream
                                  side_effect=lambda out, sid: (*grok_mod.extract_cli_result("grok", out), None)):
            res = asyncio.run(grok_mod.run_grok_subprocess("hi", "", 30))
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(_flag_value(fake.argv, "--model"), default)

    def test_agent_default_is_an_execution_seat_distinct_from_consult(self):
        """The silent `omni agent` default must land on the grok execution seat.

        The agent lane runs code; the consult lane only advises, so the two
        defaults are separate constants and must not collapse onto each other.
        """
        default = getattr(config_mod, "DEFAULT_AGENT_MODEL", "")
        self.assertTrue(default, "DEFAULT_AGENT_MODEL must be set")
        route = resolve_route(default)
        self.assertEqual(route["provider"], "grok",
                         "agent-lane default must route to grok: %r" % default)
        # Routed model is what reaches the CLI's --model: it must be the bare id.
        self.assertEqual(route["model"], default)
        self.assertIn(route["provider"],
                      getattr(agent_lane, "AGENT_PROVIDERS", ()) or ())
        self.assertNotEqual(default, getattr(config_mod, "DEFAULT_MODEL", None))

    def test_claude_prefix_routes_to_claude_provider(self):
        self.assertEqual(resolve_route("claude/opus"),
                         {"provider": "claude", "model": "opus"})

    def test_claude_prefix_strips_exactly_seven_chars(self):
        self.assertEqual(resolve_route("claude/sonnet")["model"], "sonnet")
        # bare `claude` (no slash) must NOT be hijacked / blind-stripped by that rule
        self.assertEqual(resolve_route("claude")["model"], "claude")

    def test_backends_snapshot_has_claude_entry(self):
        entry = dispatch_mod.backends_snapshot()["claude"]
        self.assertEqual(set(entry), {"cli", "prefix", "files", "read_wall", "profiles"})
        self.assertIsInstance(entry["cli"], bool)
        self.assertEqual(entry["prefix"], "claude/*")
        self.assertIs(entry["files"], False)
        self.assertIsInstance(entry["read_wall"], bool)
        self.assertIs(entry["profiles"], True)

    def test_pi_omitted_model_is_not_a_grok_id(self):
        default = getattr(config_mod, "DEFAULT_PI_MODEL", "")
        self.assertTrue(default, "DEFAULT_PI_MODEL must be set")
        self.assertFalse(
            default.lower().startswith("grok"),
            "omitted pi/ must not default to a grok* Build-CLI id: %r" % default,
        )
        self.assertNotEqual(resolve_route(default)["provider"], "grok")
        self.assertEqual(resolve_route("grok-4.6")["provider"], "grok")

    def test_pi_prefix_routes_to_pi_provider(self):
        self.assertEqual(resolve_route("pi/xai/grok-4.6"),
                         {"provider": "pi", "model": "xai/grok-4.6"})
        # Must not fall through to the grok* rule.
        self.assertNotEqual(resolve_route("pi/xai/grok-4.6")["provider"], "grok")

    def test_pi_prefix_strips_exactly_three_chars(self):
        self.assertEqual(resolve_route("pi/anthropic/claude-opus-5")["model"],
                         "anthropic/claude-opus-5")
        # bare `pi` (no slash) must NOT be hijacked
        self.assertEqual(resolve_route("pi")["provider"], "agy")

    def test_backends_snapshot_has_pi_entry(self):
        entry = dispatch_mod.backends_snapshot()["pi"]
        self.assertEqual(set(entry), {"cli", "prefix", "files", "read_wall", "profiles"})
        self.assertIsInstance(entry["cli"], bool)
        self.assertEqual(entry["prefix"], "pi/*")
        self.assertIs(entry["files"], False)
        self.assertIs(entry["read_wall"], False)
        self.assertIs(entry["profiles"], False)

    def test_cursor_prefix_routes_to_cursor_provider(self):
        self.assertEqual(resolve_route("cursor/composer-2.5"),
                         {"provider": "cursor", "model": "composer-2.5"})
        self.assertEqual(resolve_route("cursor/gpt-5.6-sol"),
                         {"provider": "cursor", "model": "gpt-5.6-sol"})
        self.assertEqual(resolve_route("cursor/claude-fable-5"),
                         {"provider": "cursor", "model": "claude-fable-5"})

    def test_cursor_prefix_strips_exactly_seven_chars(self):
        self.assertEqual(resolve_route("cursor/sol")["model"], "sol")
        self.assertNotEqual(resolve_route("cursor")["provider"], "cursor")

    def test_backends_snapshot_has_cursor_entry(self):
        entry = dispatch_mod.backends_snapshot()["cursor"]
        self.assertEqual(set(entry), {"cli", "prefix", "files", "read_wall", "profiles"})
        self.assertIsInstance(entry["cli"], bool)
        self.assertEqual(entry["prefix"], "cursor/*")
        self.assertIs(entry["files"], False)
        self.assertIs(entry["read_wall"], False)
        self.assertIs(entry["profiles"], True)

    def test_cursor_omitted_model_is_not_a_family_seat(self):
        default = getattr(config_mod, "DEFAULT_CURSOR_MODEL", "")
        self.assertTrue(default, "DEFAULT_CURSOR_MODEL must be set")
        self.assertFalse(
            default.lower().startswith("grok"),
            "omitted cursor/ must not default to a grok* id: %r" % default,
        )
        self.assertNotIn(default.lower(), ("fable", "sol", "opus"))

    def test_route_to_provider_refuses_at_depth_2(self):
        with depth("2"):
            env = asyncio.run(dispatch_mod.route_to_provider(
                "hi", "ollama/nonexistent-model", 5, None, "default"))
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "recursion_guard", env)


# --------------------------------------------------------------------------- #
# 2. run_agent refusal envelopes
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestRunAgentRefusals(LaneTestCase):
    def test_recursion_guard_at_depth_1_or_higher(self):
        wd = self.mkdir()
        for value in ("1", "2", "7"):
            with self.subTest(depth=value):
                fake = FakeRunCli()
                with depth(value), patched_lane(fake):
                    env = self.run_agent(task="t", model="codex/gpt-5.6-sol",
                                         workdir=wd, timeout_s=30)
                self.assertRefusal(env, "recursion_guard")
                self.assertEqual(fake.calls, [], "guard must fire before run_cli")

    def test_workdir_fence_on_missing_or_non_directory(self):
        wd = self.mkdir()
        a_file = os.path.join(wd, "not_a_dir.txt")
        with open(a_file, "w", encoding="utf-8") as f:
            f.write("x")
        for bad in (os.path.join(wd, "does", "not", "exist"), a_file):
            with self.subTest(workdir=bad):
                fake = FakeRunCli()
                with depth(None), patched_lane(fake):
                    env = self.run_agent(task="t", model="codex/gpt-5.6-sol",
                                         workdir=bad, timeout_s=30)
                self.assertRefusal(env, "workdir_fence")
                self.assertEqual(fake.calls, [])

    def test_unsupported_agent_for_non_agent_providers(self):
        wd = self.mkdir()
        for model in ("gpt-4o", "ollama/x", "Gemini 3.5 Flash (High)"):
            with self.subTest(model=model):
                fake = FakeRunCli()
                with depth(None), patched_lane(fake):
                    env = self.run_agent(task="t", model=model, workdir=wd, timeout_s=30)
                self.assertRefusal(env, "unsupported_agent")
                self.assertEqual(fake.calls, [],
                                 "refusal must precede any spawn")

    def test_cursor_is_a_supported_agent_provider(self):
        # cursor/composer-2.5 used to be pinned unsupported_agent (this was
        # the whole point of the pin: falsify it before trusting the flip).
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="cursor/composer-2.5",
                                 workdir=wd, timeout_s=30, fence_requested=False)
        self.assertNotEqual(env.get("reason"), "unsupported_agent", env)

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_grok_read_only_is_unsupported(self):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="grok-4.5", workdir=wd,
                                 read_only=True, timeout_s=30)
        self.assertRefusal(env, "unsupported_read_only")
        self.assertIn("pi", env.get("detail", ""))
        self.assertEqual(fake.calls, [])


# --------------------------------------------------------------------------- #
# 3. workdir fence — real (local-only) git fixtures
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestWorkdirGitFence(LaneTestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_GIT:
            raise unittest.SkipTest("git not on PATH")

    def call(self, workdir, **kwargs):
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="codex/gpt-5.6-sol",
                                 workdir=workdir, timeout_s=30, **kwargs)
        return env, fake

    def test_main_checkout_on_main_branch_is_fenced(self):
        env, fake = self.call(self.make_repo())
        self.assertRefusal(env, "workdir_fence")
        self.assertEqual(fake.calls, [])

    def test_allow_main_lifts_the_main_branch_fence(self):
        env, fake = self.call(self.make_repo(), allow_main=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(len(fake.calls), 1)

    def test_main_checkout_on_feature_branch_is_allowed(self):
        repo = self.make_repo()
        git(repo, "checkout", "-b", "feat")
        env, fake = self.call(repo)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(len(fake.calls), 1)

    def test_linked_worktree_on_feature_branch_is_allowed(self):
        repo = self.make_repo()
        sibling = os.path.join(os.path.dirname(repo), os.path.basename(repo) + "-wt")
        self.addCleanup(shutil.rmtree, sibling, ignore_errors=True)
        git(repo, "worktree", "add", sibling, "-b", "feat2")
        self.assertTrue(os.path.isfile(os.path.join(sibling, ".git")),
                        "fixture must be a LINKED worktree (.git is a file)")
        env, fake = self.call(os.path.realpath(sibling))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(len(fake.calls), 1)

    def test_linked_worktree_parked_on_main_is_fenced(self):
        """Foreman ruling (2026-07-28): the fence is BRANCH-based — a linked
        worktree sitting on main is still direct-to-main work and is refused."""
        repo = self.make_repo()
        git(repo, "checkout", "-b", "feat")  # free main so a worktree can take it
        sibling = os.path.join(os.path.dirname(repo), os.path.basename(repo) + "-wtmain")
        self.addCleanup(shutil.rmtree, sibling, ignore_errors=True)
        git(repo, "worktree", "add", sibling, "main")
        self.assertTrue(os.path.isfile(os.path.join(sibling, ".git")),
                        "fixture must be a LINKED worktree (.git is a file)")
        env, fake = self.call(os.path.realpath(sibling))
        self.assertRefusal(env, "workdir_fence")
        self.assertEqual(fake.calls, [])

    def test_plain_non_git_directory_is_allowed(self):
        env, fake = self.call(self.mkdir())
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(len(fake.calls), 1)


# --------------------------------------------------------------------------- #
# 4. codex command construction + envelope
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestCodexAgentCommand(LaneTestCase):
    @contextlib.contextmanager
    def verified_sandbox_record(self, version="0.148.0"):
        """Put the write-canary gate in its ALLOW state (issue #26)."""
        state_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, state_dir, ignore_errors=True)
        record = {"cli": "codex", "version": version, "verified_at": time.time(),
                  "runs": sandbox_canary_mod.DEFAULT_RUNS,
                  "clean": sandbox_canary_mod.DEFAULT_RUNS,
                  "breaches": 0, "inconclusive": 0, "verdict": "verified"}
        with open(os.path.join(state_dir,
                               sandbox_canary_mod.RECORD_BASENAME), "w") as fh:
            json.dump(record, fh)
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state_dir}), \
                mock.patch.object(sandbox_canary_mod, "codex_cli_version",
                                  lambda: version):
            yield

    def run_codex(self, **kwargs):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="codex/gpt-5.6-sol",
                                 workdir=wd, timeout_s=30, **kwargs)
        self.assertEqual(len(fake.calls), 1, env)
        return env, fake, wd

    def test_write_mode_argv_and_run_cli_kwargs(self):
        env, fake, wd = self.run_codex()
        argv = fake.argv
        self.assertHasFlags(argv, "exec", "--skip-git-repo-check", "--ephemeral")
        # Issue #94 piece 3: the write lane rides the kernel wrap, so the
        # vendor --sandbox pair is GONE and the bypass is in its place. The
        # declined-fence spelling is pinned by the next test.
        self.assertLacksFlags(argv, "--sandbox", "--full-auto")
        self.assertHasFlags(argv, "--dangerously-bypass-approvals-and-sandbox")
        self.assertEqual(env.get("fence"), "seatbelt-fs", env)
        self.assertFlagValue(argv, "-m", "gpt-5.6-sol")
        self.assertIn("-o", argv)
        self.assertTrue(_flag_value(argv, "-o"), "-o must be followed by a path")
        self.assertEqual(argv[-1], "-", "codex reads the task from stdin (trailing '-')")

        kwargs = fake.kwargs
        self.assertEqual(kwargs.get("cwd"), wd)
        self.assertEqual(kwargs.get("stdin_data"), "DO THE THING")
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_write_mode_declined_fence_keeps_the_vendor_sandbox_pair(self):
        # --no-fence: no kernel wrap, so the vendor pair is what isolates the
        # run. codex-cli 0.145 deprecated the --full-auto alias; the explicit
        # pair is the supported spelling.
        env, fake, _wd = self.run_codex(fence_requested=False)
        argv = fake.argv
        self.assertFlagValue(argv, "--sandbox", "workspace-write")
        self.assertLacksFlags(argv, "--full-auto",
                              "--dangerously-bypass-approvals-and-sandbox")
        self.assertEqual(env.get("fence"), "vendor", env)

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_read_only_mode_swaps_the_sandbox_value(self):
        # The read-only lane is gated by the write-canary (issue #26): under
        # the fake PATH the version probe cannot answer, so the gate fails
        # closed before argv is ever built. Give it a fresh verified record so
        # this argv pin exercises the gate's ALLOW state — the refusal states
        # have their own suite in test_sandbox_canary.py.
        with self.verified_sandbox_record():
            _, fake, _ = self.run_codex(read_only=True)
        argv = fake.argv
        self.assertFlagValue(argv, "--sandbox", "read-only")
        self.assertNotEqual(_flag_value(argv, "--sandbox"), "workspace-write")
        self.assertLacksFlags(argv, "--full-auto")

    def test_reply_text_comes_from_the_o_file_not_stdout(self):
        env, fake, wd = self.run_codex()
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], "AGENT DONE")
        self.assertNotEqual(env["text"], fake.stdout)

    def test_success_envelope_shape(self):
        env, _, wd = self.run_codex()
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["model"], "codex/gpt-5.6-sol")  # ORIGINAL model string
        self.assertEqual(env["workdir"], os.path.abspath(wd))
        self.assertIsInstance(env["duration_s"], (int, float))
        self.assertGreaterEqual(env["duration_s"], 0)


# --------------------------------------------------------------------------- #
# 4b. codex ok/failure semantics (issue #51)
# --------------------------------------------------------------------------- #
_SANDBOX_DENIAL = "sandbox-exec: sandbox_apply: Operation not permitted"

# What codex actually writes to STDERR without --json (codex-cli 0.147.0):
# a header, then the CALLER'S PROMPT echoed verbatim, before the model is even
# called. Any detector arm reading this stream is driven by caller text.
_STDERR_TRANSCRIPT = """OpenAI Codex v0.147.0
--------
workdir: /tmp/wd
model: gpt-5.6-sol
sandbox: read-only
--------
user
%s

codex
%s
"""


def _codex_command_event(command, exit_code, output, status=None):
    return json.dumps({
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": command,
            "aggregated_output": output,
            "exit_code": exit_code,
            "status": status or ("failed" if exit_code else "completed"),
        },
    })


def _codex_message_event(text):
    return json.dumps({
        "type": "item.completed",
        "item": {"id": "item_2", "type": "agent_message", "text": text},
    })


@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestCodexRunOkSemantics(LaneTestCase):
    """`codex exec` exits 0 even when every shell command it ran died.

    codex treats a failing tool call as a recoverable event, so the outer
    process exit code alone cannot decide ``ok``. These cases pin the ENVELOPE.
    The detector is a property of the lane's result handling, not of any
    sandbox arrangement.

    THE INVARIANT under test: no arm of the detector may be driven by text the
    caller or the agent can author. The verdict reads ``exit_code`` / ``status``
    off the ``--json`` event stream and nothing else, so the guards below feed
    the denial string through every writable channel — the task prompt, the
    stderr transcript, a command's output, a command's own argv, the agent's
    reply — and require ok:true from all of them.
    """

    def run_codex(self, stdout="", stderr="", reply="", **kwargs):
        wd = self.mkdir()
        fake = FakeRunCli(stdout=stdout, stderr=stderr, out_file_text=reply)
        with depth(None), patched_lane(fake):
            env = self.run_agent(task=kwargs.pop("task", "DO THE THING"),
                                 model="codex/gpt-5.6-sol",
                                 workdir=wd, timeout_s=30, **kwargs)
        self.assertEqual(len(fake.calls), 1, env)
        return env

    # -- the fault this detector exists for ---------------------------------- #

    def test_json_event_stream_of_rc71_commands_is_not_ok(self):
        # The issue #51 repro shape: outer rc 0, every internal command rc=71.
        stdout = "\n".join([
            _codex_command_event("bash -lc 'git status'", 71, _SANDBOX_DENIAL),
            _codex_command_event("bash -lc 'echo hello'", 71, _SANDBOX_DENIAL),
            _codex_message_event("Every command failed with rc=71."),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error")
        self.assertIn("exit 71", env.get("detail", ""))
        self.assertNotIn("text", env)

    def test_sync_lane_reaches_the_same_verdict(self):
        # The sync lane passes --json too, so the SAME event-stream verdict
        # covers it. Before that it had a free-text stderr arm of its own,
        # which is what B1 exploited.
        stdout = "\n".join([
            _codex_command_event("bash -lc 'cat needle.txt'", 71, _SANDBOX_DENIAL),
            _codex_message_event("I could not run anything."),
        ])
        env = self.run_codex(stdout=stdout)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error")

    def test_string_exit_code_is_still_an_environment_fault(self):
        # exit_code has been observed as a decimal string; a type drift must
        # not become a silent false negative.
        stdout = "\n".join([
            _codex_command_event("bash -lc 'echo hi'", "71", _SANDBOX_DENIAL),
            _codex_message_event("nothing ran"),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error")

    def test_dict_output_on_a_failed_command_does_not_crash_the_detector(self):
        # Some codex versions ship `output` as a dict. It used to reach
        # .lower() and raise AttributeError, which the outer handler laundered
        # into a wrong envelope. Now the field is never read at all, and the
        # rc=71 next to it is still counted.
        evt = json.dumps({
            "type": "item.completed",
            "item": {"id": "item_1", "type": "command_execution",
                     "command": "bash -lc 'echo hi'", "exit_code": 71,
                     "status": "failed", "output": {"chunks": [_SANDBOX_DENIAL]}},
        })
        env = self.run_codex(stdout="\n".join([evt, _codex_message_event("x")]),
                             job_lane=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error")

    def test_malformed_items_are_skipped_not_fatal(self):
        # Whole-item type drift: a non-dict item, a missing exit_code, a
        # non-numeric one. None of these may raise or flip a healthy run.
        stdout = "\n".join([
            json.dumps({"type": "item.completed", "item": "not-a-dict"}),
            json.dumps({"type": "item.completed",
                        "item": {"type": "command_execution",
                                 "command": "x", "exit_code": None}}),
            json.dumps({"type": "item.completed",
                        "item": {"type": "command_execution",
                                 "command": "x", "exit_code": "not-a-number"}}),
            _codex_message_event("all good"),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "all good")

    # -- guards: the caller's own text may never decide the verdict ---------- #

    def test_marker_in_the_callers_prompt_echoed_on_stderr_stays_ok(self):
        """B1: codex echoes the CALLER'S PROMPT into its stderr transcript.

        A healthy run asked to fix this very bug carries the denial string on
        stderr before the model is even called. Reading that stream made the
        lane kill its own correct answer — exactly the task this repo runs.
        """
        task = "fix issue #51, the symptom is " + _SANDBOX_DENIAL
        env = self.run_codex(
            task=task,
            stdout=_codex_message_event("Fixed it in providers/agent_lane.py."),
            stderr=_STDERR_TRANSCRIPT % (task, "Fixed it."),
            reply="Fixed it in providers/agent_lane.py.")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "Fixed it in providers/agent_lane.py.")

    def test_marker_on_stderr_with_no_command_events_stays_ok(self):
        # Same stream, question-answering run: no commands at all. The old
        # stderr arm fired on exactly this shape (total == 0).
        task = "explain " + _SANDBOX_DENIAL
        env = self.run_codex(
            task=task,
            stderr=_STDERR_TRANSCRIPT % (task, "It means the profile failed."),
            reply="It means the profile failed to apply.")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "It means the profile failed to apply.")

    def test_red_pytest_quoting_the_marker_stays_ok(self):
        """B2: a red test suite whose OUTPUT quotes the marker.

        The marker lives in providers/agent_lane.py and in this file, so
        `pytest -q test_agent.py` with one red test prints it. Counting that as
        an environment fault made a single-command run flip to ok:false.
        """
        red = ("=================================== FAILURES "
               "===================================\n"
               "_______________ TestCodexRunOkSemantics.test_marker _______________\n"
               "E       AssertionError: %r != ''\n"
               "=========================== short test summary info "
               "============================\n"
               "FAILED test_agent.py::TestCodexRunOkSemantics::test_marker\n"
               "1 failed, 9 passed in 0.14s" % _SANDBOX_DENIAL)
        stdout = "\n".join([
            _codex_command_event("bash -lc 'pytest -q test_agent.py'", 1, red),
            _codex_message_event("One test is red: the marker assertion."),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "One test is red: the marker assertion.")

    def test_ordinary_failing_command_still_returns_the_agent_answer(self):
        # Falsifier guard: "run the tests and report" with a red suite must NOT
        # be reclassified as a lane failure — rc != 0 is a legitimate RESULT.
        stdout = "\n".join([
            _codex_command_event("bash -lc 'pytest -q'", 1, "1 failed, 9 passed"),
            _codex_message_event("test_foo fails on the boundary case."),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "test_foo fails on the boundary case.")

    def test_denial_string_in_a_SUCCESSFUL_command_output_stays_ok(self):
        # An agent that greps this repository prints the denial string itself.
        stdout = "\n".join([
            _codex_command_event(
                "bash -lc 'grep -rn sandbox_apply providers/'", 0,
                "providers/agent_lane.py:118:    \"%s\"," % _SANDBOX_DENIAL),
            _codex_message_event("The marker lives in agent_lane.py:118."),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "The marker lives in agent_lane.py:118.")

    def test_marker_in_a_failed_commands_own_argv_stays_ok(self):
        # The `command` field is agent-authored too: it is whatever the agent
        # chose to run. A failing grep FOR the marker must stay a task result.
        stdout = "\n".join([
            _codex_command_event(
                "bash -lc 'grep -rn \"%s\" /etc'" % _SANDBOX_DENIAL, 1, ""),
            _codex_message_event("No match under /etc."),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "No match under /etc.")

    def test_agent_merely_quoting_the_denial_in_its_reply_stays_ok(self):
        # The -o reply is the agent's own words and is never read for a verdict.
        env = self.run_codex(
            stdout=_codex_message_event("about the bug"),
            reply="The bug prints %r for every command." % _SANDBOX_DENIAL)
        self.assertTrue(env.get("ok"), env)
        self.assertIn("sandbox_apply", env.get("text", ""))

    def test_clean_run_stays_ok(self):
        stdout = "\n".join([
            _codex_command_event("bash -lc 'echo hello'", 0, "hello"),
            _codex_message_event("hello, rc=0"),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "hello, rc=0")

    def test_one_success_among_denials_keeps_the_agent_answer(self):
        # Documented conservative limit: `failed == total` is required, so an
        # incidental success masks an otherwise wholesale denial. Pinned so the
        # tradeoff is visible if anyone tightens it.
        stdout = "\n".join([
            _codex_command_event("bash -lc 'true'", 0, ""),
            _codex_command_event("bash -lc 'git status'", 71, _SANDBOX_DENIAL),
            _codex_message_event("partial"),
        ])
        env = self.run_codex(stdout=stdout, job_lane=True)
        self.assertTrue(env.get("ok"), env)

    def test_non_codex_backends_are_untouched(self):
        # The detector is codex-specific: a grok run whose streams happen to
        # carry the marker must not be reclassified.
        wd = self.mkdir()
        fake = FakeRunCli(stderr=_SANDBOX_DENIAL)
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="grok-4.6",
                                 workdir=wd, timeout_s=30)
        self.assertTrue(env.get("ok"), env)

    # -- the argv change that removes the free-text arm ---------------------- #

    def test_sync_lane_passes_json_so_the_verdict_has_a_machine_signal(self):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="codex/gpt-5.6-sol",
                                 workdir=wd, timeout_s=30)
        self.assertHasFlags(fake.argv, "--json", "--ephemeral")
        self.assertEqual(env.get("text"), "AGENT DONE",
                         "-o reply must still win over stdout")

    def test_sync_lane_recovers_the_reply_from_events_when_o_is_empty(self):
        # --json means raw stdout is JSONL: the sync lane must never return it
        # as text, and must recover the agent_message instead.
        env = self.run_codex(
            stdout="\n".join([_codex_command_event("bash -lc 'ls'", 0, "a\nb"),
                              _codex_message_event("two files")]),
            reply="")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("text"), "two files")


# --------------------------------------------------------------------------- #
# 5. claude command construction (agent lane)
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestClaudeAgentCommand(LaneTestCase):
    def run_claude(self, **kwargs):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="claude/sonnet",
                                 workdir=wd, timeout_s=30, **kwargs)
        self.assertEqual(len(fake.calls), 1, env)
        return env, fake, wd

    def test_write_mode_argv_and_run_cli_kwargs(self):
        _, fake, wd = self.run_claude()
        argv = fake.argv
        self.assertHasFlags(argv, "-p", "--dangerously-skip-permissions",
                            "--strict-mcp-config")
        self.assertFlagValue(argv, "--model", "sonnet")  # claude/ prefix stripped
        kwargs = fake.kwargs
        self.assertEqual(kwargs.get("cwd"), wd)
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")

    def test_read_only_mode_disallows_write_and_bash(self):
        _, fake, _ = self.run_claude(read_only=True)
        argv = fake.argv
        self.assertIn("--disallowedTools", argv)
        value = _flag_value(argv, "--disallowedTools") or ""
        self.assertIn("Write", value)
        self.assertIn("Bash", value)
        self.assertLacksFlags(argv, "--dangerously-skip-permissions")

    def test_success_envelope_uses_stdout_and_original_model(self):
        env, fake, wd = self.run_claude()
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], fake.stdout)
        self.assertEqual(env["model"], "claude/sonnet")
        self.assertEqual(env["workdir"], os.path.abspath(wd))
        self.assertIsInstance(env["duration_s"], (int, float))


# --------------------------------------------------------------------------- #
# 6. grok command construction (agent lane)
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
@mock.patch("platform.system", new=lambda: "Linux")
class TestGrokAgentCommand(LaneTestCase):
    """Retain the non-macOS vendor-command contract; macOS has host guards."""
    def test_write_mode_argv_env_and_text(self):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="grok-4.5",
                                 workdir=wd, timeout_s=30)
        self.assertEqual(len(fake.calls), 1, env)
        argv = fake.argv
        self.assertFlagValue(argv, "--sandbox", "strict")
        self.assertFlagValue(argv, "--cwd", wd)
        self.assertIn("--prompt-file", argv)
        # the agent lane must NOT reuse the generate lane's clamps
        self.assertLacksFlags(argv, "--tools", "--disable-web-search")
        self.assertEqual((fake.kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], fake.stdout)
        self.assertEqual(env["model"], "grok-4.5")

    def test_zero_config_lane_spawns_grok_with_the_default_model(self):
        """No -m, no --effort: the bare id must reach grok's own --model.

        Pins the whole default path, not just the constant: a prefixed
        `grok/...` id would route to grok yet arrive here verbatim.
        """
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model=config_mod.DEFAULT_AGENT_MODEL,
                                 workdir=wd, timeout_s=30)
        self.assertEqual(len(fake.calls), 1, env)
        argv = fake.argv
        self.assertTrue(os.path.basename(argv[0]) == "grok", argv)
        self.assertFlagValue(argv, "--model", "grok-4.6")
        # Omitted effort sends NO flag: the backend's own default applies.
        self.assertLacksFlags(argv, "--reasoning-effort")
        self.assertTrue(env.get("ok"), env)

    def test_default_model_read_only_refuses_loudly(self):
        """Accepted regression: grok strict cannot separate reads from writes.

        `--read-only` on the default model is a LOUD refusal, never a silent
        reroute onto another backend.
        """
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model=config_mod.DEFAULT_AGENT_MODEL,
                                 workdir=wd, timeout_s=30, read_only=True)
        self.assertRefusal(env, "unsupported_read_only")
        self.assertEqual(fake.calls, [], "refusal must precede any spawn")


# --------------------------------------------------------------------------- #
# 6b. pi command construction (agent lane)
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestPiAgentCommand(LaneTestCase):
    def run_pi(self, **kwargs):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model="pi/xai/grok-4.6",
                                 workdir=wd, timeout_s=30, **kwargs)
        self.assertEqual(len(fake.calls), 1, env)
        return env, fake, wd

    def test_write_mode_argv_and_run_cli_kwargs(self):
        _, fake, wd = self.run_pi()
        argv = fake.argv
        self.assertHasFlags(argv, "-p", "--no-approve", "--no-extensions")
        self.assertFlagValue(argv, "--model", "xai/grok-4.6")
        self.assertLacksFlags(argv, "--no-tools")
        kwargs = fake.kwargs
        self.assertEqual(kwargs.get("cwd"), wd)
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")
        # Parent session markers must not leak into the child.
        env = kwargs.get("env") or {}
        self.assertNotIn("PI_SESSION_ID", env)
        self.assertNotIn("PI_SESSION_FILE", env)

    def test_read_only_mode_restricts_tools(self):
        _, fake, _ = self.run_pi(read_only=True)
        argv = fake.argv
        self.assertFlagValue(argv, "--tools", "read,grep,find,ls")
        self.assertLacksFlags(argv, "--no-tools")

    def test_success_envelope_uses_stdout_and_original_model(self):
        env, fake, wd = self.run_pi()
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], fake.stdout)
        self.assertEqual(env["model"], "pi/xai/grok-4.6")
        self.assertEqual(env["workdir"], os.path.abspath(wd))


# --------------------------------------------------------------------------- #
# 6c. cursor command construction (agent lane)
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestCursorAgentCommand(LaneTestCase):
    """Vendor argv on every platform; Darwin wrap rewrites --sandbox."""
    def run_cursor(self, model="cursor/composer-2.5", **kwargs):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="DO THE THING", model=model,
                                 workdir=wd, timeout_s=30, **kwargs)
        self.assertEqual(len(fake.calls), 1, env)
        return env, fake, wd

    def test_write_mode_argv_and_run_cli_kwargs(self):
        _, fake, wd = self.run_cursor()
        argv = fake.argv
        self.assertHasFlags(argv, "-p", "--trust")
        self.assertFlagValue(
            argv, "--sandbox",
            "disabled" if os.path.basename(argv[0]) == "sandbox-exec" else "enabled")
        self.assertFlagValue(argv, "--workspace", wd)
        self.assertFlagValue(argv, "--model", "composer-2.5")
        self.assertFlagValue(argv, "--output-format", "json")
        self.assertHasFlags(argv, "--force")
        self.assertLacksFlags(argv, "--mode", "--yolo",
                              "--approve-mcps", "--worktree")
        kwargs = fake.kwargs
        self.assertEqual(kwargs.get("cwd"), wd)
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")
        cfg = (kwargs.get("env") or {}).get("CURSOR_CONFIG_DIR")
        self.assertTrue(cfg, "sync lane must set a per-run CURSOR_CONFIG_DIR")
        self.assertNotEqual(os.path.realpath(cfg),
                            os.path.realpath(os.path.expanduser("~/.cursor")))

    def test_prompt_arrives_on_stdin_not_argv(self):
        _, fake, _ = self.run_cursor()
        self.assertNotIn("DO THE THING", fake.argv)
        self.assertEqual(fake.kwargs.get("stdin_data"), "DO THE THING")

    def test_read_only_mode_adds_plan_mode(self):
        _, fake, _ = self.run_cursor(read_only=True)
        self.assertFlagValue(fake.argv, "--mode", "plan")
        self.assertLacksFlags(fake.argv, "--force", "--yolo", "--approve-mcps")

    def test_resume_session_adds_resume_flag(self):
        _, fake, _ = self.run_cursor(resume_session="chat-123")
        self.assertFlagValue(fake.argv, "--resume", "chat-123")

    def test_job_lane_uses_stream_json_and_durable_config_dir(self):
        wd = self.mkdir()
        job_dir = self.mkdir("omni_cursor_job_")
        log_path = os.path.join(job_dir, "stream.raw.log")
        fake = FakeRunCli()
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="cursor/composer-2.5",
                                 workdir=wd, timeout_s=30, job_lane=True,
                                 log_path=log_path)
        self.assertEqual(len(fake.calls), 1, env)
        argv = fake.argv
        self.assertFlagValue(argv, "--output-format", "stream-json")
        cfg = (fake.kwargs.get("env") or {}).get("CURSOR_CONFIG_DIR")
        self.assertEqual(os.path.realpath(cfg),
                         os.path.realpath(os.path.join(job_dir, "cursor-config")))
        self.assertTrue(os.path.isdir(cfg))

    def test_effort_folds_into_model_slug_and_envelope(self):
        env, fake, _ = self.run_cursor(model="cursor/sol", effort="xhigh")
        self.assertFlagValue(fake.argv, "--model", "gpt-5.6-sol-xhigh")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["model"], "cursor/gpt-5.6-sol-xhigh")

    def test_effort_resolution_receives_the_named_account_profile(self):
        profile_home = self.mkdir("cursor-profile-")
        fake = FakeRunCli()
        def fake_fence(provider, argv, env, workdir_abs, read_only,
                       fence_requested, **kwargs):
            return argv, env, "seatbelt-fs", {
                "read_wall": True,
                "workdir_writable": not read_only,
                "vendor_state_writable": True,
            }, None
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": profile_home,
        }), mock.patch.object(cursor_mod, "resolve_model",
                              return_value=("profile-family-high", None)) as resolve, \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=fake_fence), \
                depth(None), patched_lane(fake):
            env = self.run_agent(
                task="DO THE THING", model="cursor/profile-family",
                workdir=self.mkdir(), timeout_s=30, account_profile="work",
                read_only=True,
                effort="high")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(resolve.call_args.args,
                         ("profile-family", "high", "work"))
        self.assertFlagValue(fake.argv, "--model", "profile-family-high")

    def test_success_envelope_uses_json_result_field(self):
        wd = self.mkdir()
        payload = json.dumps({"type": "result", "subtype": "success",
                              "result": "cursor answer", "session_id": "s1",
                              "usage": {"inputTokens": 1, "outputTokens": 2}})
        fake = FakeRunCli(stdout=payload)
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="cursor/composer-2.5",
                                 workdir=wd, timeout_s=30)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], "cursor answer")

    def test_unparseable_stdout_falls_back_verbatim(self):
        env, fake, _ = self.run_cursor()
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], fake.stdout)

    def test_inband_error_document_is_loud_not_ok_true(self):
        # rc==0 but the JSON envelope itself reports is_error:true — must
        # never become ok:true with the error text as the "answer" (same
        # in-band check as the consult lane, providers/usage.inband_error).
        wd = self.mkdir()
        payload = json.dumps({"type": "result", "is_error": True,
                              "result": "boom", "session_id": "x"})
        fake = FakeRunCli(stdout=payload)
        with depth(None), patched_lane(fake):
            env = self.run_agent(task="t", model="cursor/composer-2.5",
                                 workdir=wd, timeout_s=30)
        self.assertFalse(env.get("ok"), env)
        self.assertNotEqual(env.get("text"), "boom", env)
        self.assertIn("boom", env.get("detail") or "", env)


# --------------------------------------------------------------------------- #
# 7. providers/claude.py — the GENERATE lane (llm_generate, not the agent lane)
# --------------------------------------------------------------------------- #
@unittest.skipIf(claude_mod is None, "providers/claude.py not implemented yet")
class TestClaudeGenerateProvider(LaneTestCase):
    def call(self, fake, depth_value=None, **kwargs):
        with depth(depth_value), \
                mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            return asyncio.run(claude_mod.run_claude_subprocess(
                kwargs.pop("prompt", "hello"), kwargs.pop("model", "sonnet"),
                kwargs.pop("timeout_s", 30), **kwargs))

    def test_depth_1_refuses_without_spawning_anything(self):
        fake = FakeRunCli()
        env = self.call(fake, depth_value="1")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "recursion_guard", env)
        self.assertEqual(fake.calls, [], "guard must fire before run_cli")

    def test_success_sets_depth_1_and_pins_the_headless_flags(self):
        fake = FakeRunCli()
        env = self.call(fake)
        self.assertTrue(env.get("ok"), env)
        argv, kwargs = fake.calls[0]
        self.assertHasFlags(argv, "-p", "--strict-mcp-config")
        self.assertIn("--disallowedTools", argv)
        disallowed = _flag_value(argv, "--disallowedTools") or ""
        # write/exec AND the network + sub-agent escape hatches
        for tool in ("Bash", "Write", "Edit", "WebFetch", "WebSearch", "Task"):
            self.assertIn(tool, disallowed, "disallow-list %r" % disallowed)
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")

    def test_runs_in_a_temp_cwd_not_the_callers(self):
        fake = FakeRunCli()
        self.call(fake)
        cwd = fake.calls[0][1].get("cwd")
        self.assertTrue(cwd, "generate lane must pin an explicit cwd")
        self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(os.getcwd()))
        self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(HERE))
        self.assertTrue(os.path.realpath(cwd).startswith(
            os.path.realpath(tempfile.gettempdir())), cwd)

    def test_signal_kill_skips_inband_error(self):
        payload = json.dumps({
            "type": "result", "is_error": True, "result": "login please",
        })
        fake = FakeRunCli(returncode=-9, stdout=payload, stderr="")
        env = self.call(fake)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "signal_exit", env)
        self.assertIn("SIGKILL(9)", env.get("detail", ""), env)
        self.assertIn("login please", env.get("detail", ""), env)
        self.assertTrue(
            env.get("detail", "").startswith("claude killed by"), env)

    def test_sighup_signal_field_is_signal_exit_not_timeout(self):
        fake = FakeRunCli(returncode=-1, timed_out=False, signal=1,
                          stdout="login please", stderr="hangup-bit")
        env = self.call(fake)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "signal_exit", env)
        self.assertIn("SIGHUP(1)", env.get("detail", ""), env)
        self.assertTrue(
            env.get("detail", "").startswith("claude killed by"), env)


# --------------------------------------------------------------------------- #
# generate-lane consumers of run_cli's `signal` field
# --------------------------------------------------------------------------- #
@mock.patch("platform.system", new=lambda: "Linux")
class TestGenerateProvidersPreferSignalField(unittest.TestCase):
    """Signal classification independent of macOS host-fence admission."""
    def _which(self, name, *a, **kw):
        base = os.path.basename(str(name))
        if base in AGENT_CLIS or base in ("agy", "cursor-agent"):
            return "/fake/bin/" + base
        return _REAL_WHICH(name, *a, **kw)

    def test_sighup_signal_field_is_signal_exit_not_timeout(self):
        cases = (
            ("claude", claude_mod, lambda: claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30)),
            ("codex", codex_mod, lambda: codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30)),
            ("cursor", cursor_mod, lambda: cursor_mod.run_cursor_subprocess(
                "hi", "composer-2.5", 30)),
            ("grok", grok_mod, lambda: grok_mod.run_grok_subprocess(
                "hi", "grok-4.5", 30)),
            ("pi", pi_mod, lambda: pi_mod.run_pi_subprocess(
                "hi", "xai/grok-4.6", 30)),
            ("agy", agy_mod, lambda: agy_mod.run_agy_subprocess(
                "hi", "m", 30, "default")),
        )
        for name, module, call in cases:
            with self.subTest(provider=name):
                self.assertIsNotNone(module, name)
                fake = FakeRunCli(returncode=-1, timed_out=False, signal=1,
                                  stdout="", stderr="hangup-bit")
                with depth(None), mock.patch.object(module, "run_cli", fake), \
                        mock.patch.object(shutil, "which", self._which), \
                        mock.patch("providers.kernel_sandbox.available",
                                   return_value=False), \
                        mock.patch.object(
                            module, "resolve_bin",
                            return_value="/fake/bin/cursor-agent",
                        ) if name == "cursor" else contextlib.nullcontext():
                    env = asyncio.run(call())
                self.assertEqual(env.get("reason"), "signal_exit", env)
                self.assertIn("SIGHUP(1)", env.get("detail", ""), env)
                self.assertTrue(
                    env.get("detail", "").startswith("%s killed by" % name),
                    env)

    def test_timed_out_true_stays_timeout_even_with_signal(self):
        fake = FakeRunCli(returncode=-1, timed_out=True, signal=1)
        with depth(None), mock.patch.object(claude_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", self._which):
            env = asyncio.run(claude_mod.run_claude_subprocess(
                "hi", "sonnet", 30))
        self.assertEqual(env.get("reason"), "timeout", env)


# --------------------------------------------------------------------------- #
# 7b. providers/pi.py — the GENERATE lane
# --------------------------------------------------------------------------- #
class TestPiGenerateProvider(LaneTestCase):
    def call(self, fake, depth_value=None, **kwargs):
        with depth(depth_value), \
                mock.patch.object(pi_mod, "run_cli", fake), \
                mock.patch.object(shutil, "which", _fake_which):
            return asyncio.run(pi_mod.run_pi_subprocess(
                kwargs.pop("prompt", "hello"),
                kwargs.pop("model", "xai/grok-4.6"),
                kwargs.pop("timeout_s", 30), **kwargs))

    def test_depth_1_refuses_without_spawning_anything(self):
        fake = FakeRunCli()
        env = self.call(fake, depth_value="1")
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "recursion_guard", env)
        self.assertEqual(fake.calls, [], "guard must fire before run_cli")

    def test_success_sets_depth_1_and_pins_the_headless_flags(self):
        fake = FakeRunCli(stdout='{"type":"session"}\n'
                                 '{"type":"message_end","message":'
                                 '{"role":"assistant","content":'
                                 '[{"type":"text","text":"PONG"}]}}\n')
        env = self.call(fake)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["text"], "PONG")
        argv, kwargs = fake.calls[0]
        self.assertHasFlags(argv, "-p", "--no-tools", "--no-session",
                            "--no-context-files", "--no-approve")
        self.assertFlagValue(argv, "--mode", "json")
        self.assertFlagValue(argv, "--model", "xai/grok-4.6")
        self.assertEqual((kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")
        child_env = kwargs.get("env") or {}
        self.assertNotIn("PI_SESSION_ID", child_env)
        self.assertNotIn("PI_SESSION_FILE", child_env)

    def test_runs_in_a_temp_cwd_not_the_callers(self):
        fake = FakeRunCli(stdout='{"type":"message_end","message":'
                                 '{"role":"assistant","content":'
                                 '[{"type":"text","text":"ok"}]}}\n')
        self.call(fake)
        cwd = fake.calls[0][1].get("cwd")
        self.assertTrue(cwd, "generate lane must pin an explicit cwd")
        self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(os.getcwd()))
        self.assertTrue(os.path.realpath(cwd).startswith(
            os.path.realpath(tempfile.gettempdir())), cwd)


# --------------------------------------------------------------------------- #
# 8. omni_cli `agent` subcommand — argv -> run_agent kwargs
# --------------------------------------------------------------------------- #
_RUN_AGENT_POSITIONAL = ("task", "model", "workdir", "read_only", "timeout_s",
                         "account_profile", "allow_main")
OK_ENVELOPE = {"ok": True, "text": "t", "model": "m", "workdir": "w",
               "duration_s": 0.1}


class _AgentRecorder:
    """Async run_agent stand-in; normalizes positional args onto the spec signature."""

    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = []

    async def __call__(self, *args, **kwargs):
        bound = dict(zip(_RUN_AGENT_POSITIONAL, args))
        bound.update(kwargs)
        self.calls.append(bound)
        return self.envelope


def run_cli_main(argv, envelope=None, stdin_text=None):
    rec = _AgentRecorder(envelope or OK_ENVELOPE)
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch.object(omni_cli, "run_agent", rec):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if stdin_text is not None:
                with mock.patch.object(sys, "stdin", io.StringIO(stdin_text)):
                    code = omni_cli.main(argv)
            else:
                code = omni_cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue(), rec.calls


@unittest.skipUnless(HAS_CLI_AGENT, "omni_cli.run_agent not implemented yet")
class TestCliAgentMapping(unittest.TestCase):
    def test_maps_every_flag_onto_run_agent(self):
        code, out, _, calls = run_cli_main(
            ["agent", "fix the bug", "--workdir", "/repo/wt", "-m", "claude/opus",
             "--read-only", "--allow-main", "--profile", "premium",
             "--timeout", "900"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        kw = calls[0]
        self.assertEqual(kw["task"], "fix the bug")
        self.assertEqual(kw["workdir"], "/repo/wt")
        self.assertEqual(kw["model"], "claude/opus")
        self.assertIs(kw["read_only"], True)
        self.assertIs(kw["allow_main"], True)
        self.assertEqual(kw["account_profile"], "premium")
        self.assertEqual(kw["timeout_s"], 900)
        self.assertTrue(json.loads(out)["ok"])

    def test_defaults_model_to_default_agent_model_and_flags_to_false(self):
        _, _, _, calls = run_cli_main(["agent", "task", "--workdir", "/repo/wt"])
        kw = calls[0]
        self.assertEqual(kw["model"], config_mod.DEFAULT_AGENT_MODEL)
        self.assertIs(kw["read_only"], False)
        self.assertIs(kw["allow_main"], False)
        self.assertEqual(kw["account_profile"], "default")

    def test_task_file_supplies_the_task_body(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "task.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("TASK FROM FILE")
            _, _, _, calls = run_cli_main(
                ["agent", "--task-file", path, "--workdir", "/repo/wt"])
        self.assertEqual(calls[0]["task"], "TASK FROM FILE")

    def test_dash_reads_the_task_from_stdin(self):
        _, _, _, calls = run_cli_main(["agent", "-", "--workdir", "/repo/wt"],
                                      stdin_text="TASK FROM STDIN")
        self.assertEqual(calls[0]["task"], "TASK FROM STDIN")

    def test_exactly_one_task_source_is_required(self):
        for argv in (["agent", "--workdir", "/repo/wt"],
                     ["agent", "t", "--task-file", "/f", "--workdir", "/repo/wt"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as cm:
                    run_cli_main(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_workdir_is_required(self):
        with self.assertRaises(SystemExit) as cm:
            run_cli_main(["agent", "task"])
        self.assertEqual(cm.exception.code, 2)

    def test_out_writes_the_text_and_rewrites_the_envelope(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reply.md")
            code, out, _, _ = run_cli_main(
                ["agent", "task", "--workdir", "/repo/wt", "--out", path],
                {"ok": True, "text": "AGENT REPLY", "model": "m",
                 "workdir": "w", "duration_s": 1.0})
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "AGENT REPLY")
            env = json.loads(out)
            self.assertNotIn("text", env)
            self.assertEqual(env["text_path"], path)
            self.assertEqual(env["text_bytes"], len(b"AGENT REPLY"))

    def test_plain_prints_text_only_and_refusal_exits_1(self):
        code, out, _, _ = run_cli_main(
            ["agent", "task", "--workdir", "/repo/wt", "--plain"],
            {"ok": True, "text": "just text", "model": "m", "workdir": "w",
             "duration_s": 1.0})
        self.assertEqual((code, out), (0, "just text\n"))

        code, out, err, _ = run_cli_main(
            ["agent", "task", "--workdir", "/repo/wt"],
            {"ok": False, "reason": "workdir_fence", "detail": "on main"})
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["reason"], "workdir_fence")


# --------------------------------------------------------------------------- #
# 9. review hardening (2026-07-28 fix spec) — path normalization, fail-closed
#    fence, depth export, and the error envelopes the earlier suite left open
# --------------------------------------------------------------------------- #
@unittest.skipIf(agent_lane is None, "providers/agent_lane.py not implemented yet")
class TestReviewHardening(LaneTestCase):
    def call(self, model="codex/gpt-5.6-sol", fake=None, depth_value=None, **kwargs):
        fake = fake if fake is not None else FakeRunCli()
        with depth(depth_value), patched_lane(fake):
            env = self.run_agent(task="t", model=model, timeout_s=30, **kwargs)
        return env, fake

    # -- workdir normalization ------------------------------------------- #
    @unittest.skipUnless(HAS_GIT, "git not on PATH")
    def test_symlinked_workdir_is_fenced_via_its_realpath(self):
        """A symlink pointing at a repo on main must NOT slip past the fence."""
        repo = self.make_repo()
        link = os.path.join(self.mkdir("omni_agent_link_"), "repo_link")
        os.symlink(repo, link)
        env, fake = self.call(workdir=link)
        self.assertRefusal(env, "workdir_fence")
        self.assertEqual(fake.calls, [], "fence must fire before run_cli")

    @unittest.skipUnless(HAS_GIT, "git not on PATH")
    def test_fence_and_run_share_one_normalized_workdir(self):
        """Same symlink, allowed branch: cwd and envelope carry the REALPATH.

        This is the check/run divergence guard — abspath() would leave the link
        path on the exec side while the fence judged the target.
        """
        repo = self.make_repo()
        git(repo, "checkout", "-b", "feat")
        link = os.path.join(self.mkdir("omni_agent_link_"), "repo_link")
        os.symlink(repo, link)
        env, fake = self.call(workdir=link)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env["workdir"], repo)
        self.assertEqual(fake.kwargs.get("cwd"), repo)

    def test_workdir_none_is_a_fence_refusal_not_an_exception(self):
        env, fake = self.call(workdir=None)
        self.assertRefusal(env, "workdir_fence")
        self.assertEqual(fake.calls, [])

    # -- fail-closed git fence -------------------------------------------- #
    def test_indeterminate_git_state_fails_closed(self):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_git",
                                  lambda *a, **kw: (None, "", "boom")):
            env = self.run_agent(task="t", model="codex/gpt-5.6-sol",
                                 workdir=wd, timeout_s=30)
        self.assertRefusal(env, "workdir_fence")
        self.assertIn("boom", env["detail"])
        self.assertEqual(fake.calls, [], "an unreadable git state must not run")

    @unittest.skipUnless(HAS_GIT, "git not on PATH")
    def test_branch_fence_ignores_case(self):
        repo = self.make_repo(branch="Main")
        env, fake = self.call(workdir=repo)
        self.assertRefusal(env, "workdir_fence")
        self.assertEqual(fake.calls, [])

        env, fake = self.call(workdir=repo, allow_main=True)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(len(fake.calls), 1)

    # -- depth accounting -------------------------------------------------- #
    def test_negative_depth_clamps_and_the_child_still_gets_1(self):
        env, fake = self.call(workdir=self.mkdir(), depth_value="-5")
        self.assertTrue(env.get("ok"), env)
        self.assertEqual((fake.kwargs.get("env") or {}).get(ENV_DEPTH_NAME), "1")

    def test_consultation_lanes_export_the_child_depth(self):
        cases = (
            ("codex", codex_mod, lambda: codex_mod.run_codex_subprocess(
                "hi", "gpt-5.6-sol", 30)),
            ("grok", grok_mod, lambda: grok_mod.run_grok_subprocess(
                "hi", "grok-4.5", 30)),
        )
        for name, module, call in cases:
            with self.subTest(provider=name):
                fake = FakeRunCli()
                with depth(None), mock.patch.object(module, "run_cli", fake), \
                        mock.patch.object(shutil, "which", _fake_which):
                    asyncio.run(call())
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual((fake.kwargs.get("env") or {}).get(ENV_DEPTH_NAME),
                                 "1", "%s must export depth+1 to its child" % name)

    # -- failure envelopes ------------------------------------------------- #
    def test_timed_out_run_reports_timeout(self):
        env, _ = self.call(model="claude/sonnet", workdir=self.mkdir(),
                           fake=FakeRunCli(timed_out=True))
        self.assertRefusal(env, "timeout")

    def test_non_zero_exit_reports_a_classified_reason(self):
        env, _ = self.call(model="claude/sonnet", workdir=self.mkdir(),
                           fake=FakeRunCli(returncode=2, stdout="", stderr="boom"))
        self.assertFalse(env.get("ok"), env)
        self.assertIsInstance(env.get("reason"), str, env)
        self.assertTrue(env["reason"], "reason must not be empty")
        self.assertIn("boom", env.get("detail", ""))

    def test_codex_failure_detail_never_carries_stderr(self):
        # codex stderr can echo the caller's prompt; a prompt can imitate a
        # codex log line, so no codex failure detail may come from stderr.
        stderr = ("SYNTHETIC-PROMPT-SENTINEL\n2026-10-10T03:55:02.1Z ERROR "
                  "codex_core::client: SYNTHETIC-PROMPT-SENTINEL sk-" + "k" * 40 + "\n")
        for kwargs in ({"returncode": 1}, {"returncode": -11},
                       {"returncode": -1, "signal": 9}):
            with self.subTest(**kwargs):
                fake = FakeRunCli(stdout="", stderr=stderr, **kwargs)
                with mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                        depth(None), patched_lane(fake):
                    env = self.run_agent(
                        task="t", model="codex/gpt-5.6-sol",
                        workdir=self.mkdir(), timeout_s=30,
                        fence_requested=False)
                self.assertFalse(env.get("ok"), env)
                self.assertNotIn("SYNTHETIC-PROMPT-SENTINEL", env.get("detail", ""), env)
                self.assertNotIn("sk-k", env.get("detail", ""), env)

    def run_codex(self, fake, task="t"):
        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                depth(None), patched_lane(fake):
            return self.run_agent(task=task, model="codex/gpt-5.6-sol",
                                  workdir=self.mkdir(), timeout_s=30,
                                  fence_requested=False)

    def test_codex_prompt_echo_does_not_classify_the_failure(self):
        event = json.dumps({"type": "error", "message": "transport failed"})
        task = "Review.\nfix the login bug in auth.json"
        env = self.run_codex(FakeRunCli(returncode=1, stdout=event,
                                        stderr="fix the login bug in auth.json\n"), task)
        self.assertEqual(env.get("reason"), "runtime_error", env)

    def test_codex_json_shaped_stdout_is_never_the_reply(self):
        fragment = '{"type": "item.completed", "item": {"type": "reasoning", "text": "NON_ERROR_SENTINEL'
        for stdout in (fragment, "BANNER\n" + fragment):
            with self.subTest(stdout=stdout[:8]):
                env = self.run_codex(FakeRunCli(returncode=0, stdout=stdout, out_file_text=""))
                self.assertFalse(env.get("ok"), env)
                self.assertEqual(env.get("reason"), "empty_response", env)
                self.assertNotIn("NON_ERROR_SENTINEL", json.dumps(env))
        event = json.dumps({"type": "error", "message": "transport failed"})
        env = self.run_codex(FakeRunCli(returncode=0, stdout=event, out_file_text=""))
        self.assertEqual(env.get("reason"), "empty_response", env)
        self.assertEqual(env.get("detail"), "transport failed", env)

    def test_signal_exit_is_not_text_classified(self):
        fake = FakeRunCli(returncode=-11, stdout="login please",
                          stderr="partial-segv")
        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                depth(None), patched_lane(fake):
            env = self.run_agent(
                task="t", model="claude/sonnet",
                workdir=self.mkdir(), timeout_s=30,
                fence_requested=False)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "signal_exit", env)
        self.assertIn("SIGSEGV(11)", env.get("detail", ""), env)
        self.assertIn("partial-segv", env.get("detail", ""), env)
        self.assertTrue(
            env.get("detail", "").startswith("claude killed by"), env)

    def test_sighup_signal_field_is_signal_exit_not_timeout(self):
        fake = FakeRunCli(returncode=-1, timed_out=False, signal=1,
                          stdout="login please", stderr="hangup-bit")
        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                depth(None), patched_lane(fake):
            env = self.run_agent(
                task="t", model="claude/sonnet",
                workdir=self.mkdir(), timeout_s=30,
                fence_requested=False)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "signal_exit", env)
        self.assertIn("SIGHUP(1)", env.get("detail", ""), env)
        self.assertTrue(
            env.get("detail", "").startswith("claude killed by"), env)

    def test_timed_out_true_stays_timeout_even_with_signal(self):
        fake = FakeRunCli(returncode=-1, timed_out=True, signal=1)
        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                depth(None), patched_lane(fake):
            env = self.run_agent(
                task="t", model="claude/sonnet",
                workdir=self.mkdir(), timeout_s=30,
                fence_requested=False)
        self.assertRefusal(env, "timeout")

    def test_silent_success_reports_empty_response(self):
        env, _ = self.call(model="claude/sonnet", workdir=self.mkdir(),
                           fake=FakeRunCli(returncode=0, stdout="", stderr=""))
        self.assertRefusal(env, "empty_response")

    def test_missing_backend_cli_refuses_before_spawning(self):
        wd = self.mkdir()
        fake = FakeRunCli()
        with depth(None), mock.patch.object(agent_lane, "run_cli", fake), \
                mock.patch.object(shutil, "which", lambda *a, **kw: None):
            env = self.run_agent(task="t", model="codex/gpt-5.6-sol",
                                 workdir=wd, timeout_s=30)
        self.assertRefusal(env, "cli_missing")
        self.assertEqual(fake.calls, [])

    # -- CLI surface -------------------------------------------------------- #
    @unittest.skipUnless(HAS_CLI_AGENT, "omni_cli.run_agent not implemented yet")
    def test_unreadable_task_file_is_a_runtime_error_envelope(self):
        missing = os.path.join(self.mkdir(), "no_such_task.md")
        code, out, _, calls = run_cli_main(
            ["agent", "--task-file", missing, "--workdir", "/repo/wt"])
        self.assertEqual(code, 1)
        env = json.loads(out)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "runtime_error", env)
        self.assertEqual(calls, [], "run_agent must not be reached")


if __name__ == "__main__":
    unittest.main()
