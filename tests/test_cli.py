#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for the `omni` CLI entry (omni_cli.py). No network, no CLIs.

call_tool is patched at the omni_cli module level, so these exercise ONLY the
CLI's own responsibilities: argv -> tool-args mapping, output modes
(envelope / --out / --plain), exit codes, and stdin handling. Provider and
dispatch behavior is covered by test_server.py.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

import omni_cli

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Recorder:
    """Async call_tool stand-in that records (name, args) and returns a canned envelope."""

    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        return self.envelope


def run_cli(argv, envelope, stdin_text=None):
    rec = _Recorder(envelope)
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch.object(omni_cli, "call_tool", rec):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if stdin_text is not None:
                with mock.patch.object(sys, "stdin", io.StringIO(stdin_text)):
                    code = omni_cli.main(argv)
            else:
                code = omni_cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue(), rec.calls


class TestCanaryCli(unittest.TestCase):
    def _run(self, argv, returned):
        seen = {}

        def fake(runs, timeout_s, remeasure):
            seen.update(runs=runs, timeout_s=timeout_s, remeasure=remeasure)
            return returned

        stdout = io.StringIO()
        with mock.patch.object(
                omni_cli.sandbox_canary, "run_write_canary", fake):
            with contextlib.redirect_stdout(stdout):
                code = omni_cli.main(argv)
        return code, stdout.getvalue(), seen

    def test_remeasure_flag_reaches_the_battery(self):
        # Dropping `remeasure=ns.remeasure` in the handler kills the escape
        # hatch while every canary test still passes: the flag parses, the
        # battery never hears it, and a listed version can no longer be
        # measured at all (issue #99).
        code, out, seen = self._run(
            ["canary", "codex-write", "--remeasure"],
            {"measurement": "clean", "recorded": False,
             "gate": "refused_on_version"},
        )
        self.assertIs(seen["remeasure"], True)
        # Evidence is not success: nothing was recorded, the lane is shut.
        self.assertEqual(code, 1)
        self.assertIn("refused_on_version", out)

    def test_default_battery_is_not_a_remeasure(self):
        code, out, seen = self._run(
            ["canary", "codex-write"], {"verdict": "verified"})
        self.assertIs(seen["remeasure"], False)
        self.assertEqual(code, 0)


class TestGenMapping(unittest.TestCase):
    def test_gen_maps_all_args(self):
        code, out, _, calls = run_cli(
            ["gen", "hello", "-m", "grok-4.5", "-f", "/a.png", "-f", "/b.png",
             "--profile", "premium", "--timeout", "42"],
            {"ok": True, "text": "hi", "model": "grok-4.5"})
        self.assertEqual(code, 0)
        name, args = calls[0]
        self.assertEqual(name, "llm_generate")
        self.assertEqual(args["prompt"], "hello")
        self.assertEqual(args["model"], "grok-4.5")
        self.assertEqual(args["files"], ["/a.png", "/b.png"])
        self.assertEqual(args["account_profile"], "premium")
        self.assertEqual(args["timeout_s"], 42)
        self.assertTrue(json.loads(out)["ok"])

    def test_gen_omits_model_and_files_when_unset(self):
        _, _, _, calls = run_cli(["gen", "hello"], {"ok": True, "text": "x"})
        _, args = calls[0]
        self.assertNotIn("model", args)
        self.assertNotIn("files", args)
        self.assertEqual(args["account_profile"], "default")

    def test_gen_dash_reads_stdin(self):
        _, _, _, calls = run_cli(["gen", "-"], {"ok": True, "text": "x"},
                                 stdin_text="from stdin")
        self.assertEqual(calls[0][1]["prompt"], "from stdin")

    def test_failure_envelope_exits_1(self):
        code, out, _, _ = run_cli(["gen", "hello"],
                                  {"ok": False, "reason": "cli_missing", "detail": "x"})
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["ok"])


class TestOutputModes(unittest.TestCase):
    def test_out_writes_file_and_rewrites_envelope(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reply.md")
            code, out, _, _ = run_cli(["gen", "hello", "--out", path],
                                      {"ok": True, "text": "REPLY BODY"})
            self.assertEqual(code, 0)
            with open(path) as f:
                self.assertEqual(f.read(), "REPLY BODY")
            env = json.loads(out)
            self.assertNotIn("text", env)
            self.assertEqual(env["text_path"], path)
            self.assertEqual(env["text_bytes"], len(b"REPLY BODY"))

    def test_out_on_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reply.md")
            code, out, _, _ = run_cli(["gen", "hello", "--out", path],
                                      {"ok": False, "reason": "timeout", "detail": "x"})
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(path))
            self.assertEqual(json.loads(out)["reason"], "timeout")

    def test_plain_prints_text_only(self):
        code, out, err, _ = run_cli(["gen", "hello", "--plain"],
                                    {"ok": True, "text": "just text"})
        self.assertEqual(code, 0)
        self.assertEqual(out, "just text\n")
        self.assertEqual(err, "")

    def test_plain_failure_envelope_on_stderr(self):
        code, out, err, _ = run_cli(["gen", "hello", "--plain"],
                                    {"ok": False, "reason": "auth_error", "detail": "x"})
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["reason"], "auth_error")

    def test_out_and_plain_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as cm:
            run_cli(["gen", "hi", "--plain", "--out", "/tmp/x"], {"ok": True})
        self.assertEqual(cm.exception.code, 2)


class TestBigMapping(unittest.TestCase):
    def test_big_inline_text(self):
        _, _, _, calls = run_cli(["big", "summarize", "-t", "SRC", "-m", "agy-x"],
                                 {"ok": True, "text": "s"})
        name, args = calls[0]
        self.assertEqual(name, "llm_big_context")
        self.assertEqual(args["instruction"], "summarize")
        self.assertEqual(args["text"], "SRC")
        self.assertNotIn("source_path", args)

    def test_big_source_path(self):
        _, _, _, calls = run_cli(["big", "summarize", "-s", "/data/doc.txt"],
                                 {"ok": True, "text": "s"})
        _, args = calls[0]
        self.assertEqual(args["source_path"], "/data/doc.txt")
        self.assertNotIn("text", args)

    def test_big_text_stdin(self):
        _, _, _, calls = run_cli(["big", "summarize", "--text-stdin"],
                                 {"ok": True, "text": "s"}, stdin_text="PIPED SRC")
        self.assertEqual(calls[0][1]["text"], "PIPED SRC")

    def test_big_requires_exactly_one_source(self):
        with self.assertRaises(SystemExit) as cm:
            run_cli(["big", "summarize"], {"ok": True})
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit) as cm:
            run_cli(["big", "summarize", "-t", "a", "-s", "/b"], {"ok": True})
        self.assertEqual(cm.exception.code, 2)


class TestImgModelsMapping(unittest.TestCase):
    def test_img_maps_set_args_only(self):
        _, _, _, calls = run_cli(
            ["img", "a cat", "--backend", "rest", "-n", "2",
             "--response-format", "url", "--out-dir", "/tmp/imgs"],
            {"ok": True, "images": []})
        name, args = calls[0]
        self.assertEqual(name, "llm_image")
        self.assertEqual(args["backend"], "rest")
        self.assertEqual(args["n"], 2)
        self.assertEqual(args["response_format"], "url")
        self.assertEqual(args["out_dir"], "/tmp/imgs")
        self.assertNotIn("source_image", args)
        self.assertNotIn("model", args)

    def test_models_maps_timeout(self):
        code, _, _, calls = run_cli(["models", "--timeout", "7"],
                                    {"ok": True, "models": ["m"]})
        self.assertEqual(code, 0)
        self.assertEqual(calls[0], ("list_models", {"timeout_s": 7}))


class TestModelCatalogCli(unittest.TestCase):
    def run_catalog(self, argv, method, result):
        from providers import model_catalog
        stdout = io.StringIO()
        with mock.patch.object(model_catalog, method, return_value=result) as call, \
                contextlib.redirect_stdout(stdout):
            code = omni_cli.main(argv)
        return code, stdout.getvalue(), call

    def test_status_refresh_and_check_keep_typed_exit_codes(self):
        status_result = {"ok": True, "catalogs": []}
        code, out, call = self.run_catalog(
            ["models", "status", "--json"], "status", status_result)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), status_result)
        self.assertEqual(call.call_args.args, ("all", None))
        self.assertEqual(call.call_args.kwargs, {"details": False})

        detailed = {"ok": True, "details": True, "catalogs": [
            {"provider": "cursor", "models": [{"id": "cursor-model"}]},
        ]}
        code, out, call = self.run_catalog(
            ["models", "status", "--provider", "cursor", "--details", "--json"],
            "status", detailed)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), detailed)
        self.assertEqual(call.call_args.kwargs, {"details": True})

        refresh_result = {"ok": False, "providers": {"agy": {"ok": True}}}
        code, out, call = self.run_catalog(
            ["models", "refresh", "--provider", "agy", "--profile", "work",
             "--timeout", "9"], "refresh", refresh_result)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), refresh_result)
        self.assertEqual(call.call_args.args, ("agy", "work", 9))

        check_result = {"ok": True, "exit_code": 3, "drift": True}
        code, out, _call = self.run_catalog(
            ["models", "check", "--provider", "cursor", "--profile", "work"],
            "check", check_result)
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(out), check_result)

    def test_import_snapshot_and_schedule_use_explicit_paths_and_profiles(self):
        imported = {"ok": True, "source": "manual"}
        code, out, call = self.run_catalog(
            ["models", "import", "--provider", "claude", "--profile", "alt",
             "--file", "/tmp/catalog.json"], "manual_import", imported)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), imported)
        self.assertEqual(call.call_args.args,
                         ("claude", "alt", "/tmp/catalog.json"))

        snapshot = {"ok": True, "output": "/tmp/snapshot.json"}
        code, _out, call = self.run_catalog(
            ["models", "snapshot", "--output", "/tmp/snapshot.json"],
            "snapshot", snapshot)
        self.assertEqual(code, 0)
        self.assertEqual(call.call_args.args,
                         ("/tmp/snapshot.json", "all", None))

        code, out, call = self.run_catalog(
            ["models", "schedule", "--print", "--profile", "work",
             "--provider", "codex", "--interval", "86400"], "schedule_plist",
            {"ok": True, "plist": "<plist/>"})
        self.assertEqual(code, 0)
        self.assertEqual(out, "<plist/>")
        self.assertEqual(call.call_args.args, ("work", 86400, 120))
        self.assertEqual(call.call_args.kwargs, {"provider": "codex"})


class TestCallToolExceptionIsLoud(unittest.TestCase):
    def test_exception_becomes_envelope(self):
        async def boom(name, args):
            raise RuntimeError("kaput")
        stdout = io.StringIO()
        with mock.patch.object(omni_cli, "call_tool", boom):
            with contextlib.redirect_stdout(stdout):
                code = omni_cli.main(["gen", "hello"])
        self.assertEqual(code, 1)
        env = json.loads(stdout.getvalue())
        self.assertFalse(env["ok"])
        self.assertIn("kaput", env["detail"])


class TestProfilesCli(unittest.TestCase):
    def _parse_profiles(self, argv):
        parser = omni_cli.build_parser()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                return parser.parse_args(argv), None
            except SystemExit as exc:
                return None, exc.code

    def test_list_json_skips_mcp_and_emits_rows(self):
        rows = [{
            "road": "claude", "name": "default", "source": "default",
            "path": None, "identity": {}, "state": None, "problems": [],
            "cooldown": "healthy",
        }]
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=rows) as lp:
            code, out, err, calls = run_cli(["profiles", "list", "--json"],
                                            {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        lp.assert_called_once()
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profiles"], rows)

    def test_json_after_subcommand_is_honored(self):
        ns, err = self._parse_profiles(["profiles", "check", "--json"])
        self.assertIsNone(err)
        self.assertTrue(ns.json)
        self.assertEqual(ns.profiles_cmd, "check")

    def test_json_before_subcommand_is_not_silently_dropped(self):
        ns, err = self._parse_profiles(["profiles", "--json", "check"])
        if err is not None:
            self.assertNotEqual(err, 0)
            return
        self.assertTrue(
            ns.json,
            "--json before subcommand must be honored, not overwritten to False")

    def test_road_after_subcommand_is_honored(self):
        ns, err = self._parse_profiles(
            ["profiles", "check", "--road", "claude", "--live"])
        self.assertIsNone(err)
        self.assertEqual(ns.profiles_road, "claude")
        self.assertTrue(ns.live)

    def test_road_before_subcommand_is_not_silently_dropped(self):
        ns, err = self._parse_profiles(
            ["profiles", "--road", "claude", "check", "--live"])
        if err is not None:
            self.assertNotEqual(err, 0)
            return
        self.assertEqual(
            ns.profiles_road, "claude",
            "--road before subcommand must be honored, not overwritten to None")
        self.assertTrue(ns.live)

    def test_check_maps_live_road_name_and_exits_1(self):
        listed = [{
            "road": "claude", "name": "a", "source": "env", "path": "/x",
            "identity": {}, "state": "not_logged_in", "problems": [],
            "cooldown": "healthy",
        }]
        checked = {"ok": False, "profiles": [
            dict(listed[0], problems=["not_logged_in"])]}
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=listed) as lp, \
                mock.patch("providers.profile_check.check",
                           return_value=checked) as ck:
            code, out, _, calls = run_cli(
                ["profiles", "check", "--live", "--json",
                 "--road", "claude", "--name", "a"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(lp.call_args.kwargs.get("roads"), ("claude",))
        self.assertEqual(ck.call_args.kwargs.get("live"), True)
        self.assertFalse(json.loads(out)["ok"])

    def test_add_maps_persist_json(self):
        envelope = {
            "ok": True, "road": "claude", "name": "work",
            "path": "/tmp/.claude-work",
            "login_command": "CLAUDE_CONFIG_DIR=/tmp/.claude-work claude login",
            "mcp_env": {"CLAUDE_CONFIG_DIR_WORK": "/tmp/.claude-work"},
        }
        with mock.patch("providers.profile_check.add_profile",
                        return_value=envelope) as add:
            code, out, _, calls = run_cli(
                ["profiles", "add", "claude", "work", "--persist", "--json"],
                {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        add.assert_called_once_with("claude", "work", persist=True)
        payload = json.loads(out)
        self.assertIn("claude login", payload["login_command"])
        self.assertEqual(payload["mcp_env"]["CLAUDE_CONFIG_DIR_WORK"],
                         "/tmp/.claude-work")

    def test_add_exec_without_tty_refuses_loud(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False), \
                mock.patch("providers.profile_check.add_profile") as add:
            code, out, _, calls = run_cli(
                ["profiles", "add", "claude", "work", "--exec", "--json"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        add.assert_not_called()
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "not_a_tty")

    def test_add_exec_login_failed_removes_empty_dir(self):
        dest = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(dest, ignore_errors=True))
        envelope = {
            "ok": True, "road": "claude", "name": "work",
            "path": dest,
            "login_command": "CLAUDE_CONFIG_DIR=%s claude login" % dest,
            "login_argv": ["claude", "login"],
            "login_env": {"CLAUDE_CONFIG_DIR": dest},
            "var": "CLAUDE_CONFIG_DIR_WORK",
            "mcp_env": {"CLAUDE_CONFIG_DIR_WORK": dest},
        }
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
                mock.patch("providers.profile_check.add_profile",
                           return_value=envelope), \
                mock.patch("subprocess.call", return_value=7):
            code, out, _, calls = run_cli(
                ["profiles", "add", "claude", "work", "--exec", "--json"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "login_failed")
        self.assertEqual(payload["rc"], 7)
        self.assertIn(dest, payload["detail"])
        self.assertIn("removed", payload["detail"])
        self.assertFalse(os.path.exists(dest))

    def test_add_exec_login_failed_leaves_nonempty_dir(self):
        dest = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(dest, ignore_errors=True))
        with open(os.path.join(dest, "keep"), "w", encoding="utf-8") as handle:
            handle.write("x")
        envelope = {
            "ok": True, "road": "claude", "name": "work",
            "path": dest,
            "login_command": "CLAUDE_CONFIG_DIR=%s claude login" % dest,
            "login_argv": ["claude", "login"],
            "login_env": {"CLAUDE_CONFIG_DIR": dest},
            "var": "CLAUDE_CONFIG_DIR_WORK",
            "mcp_env": {"CLAUDE_CONFIG_DIR_WORK": dest},
        }
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
                mock.patch("providers.profile_check.add_profile",
                           return_value=envelope), \
                mock.patch("subprocess.call", return_value=3):
            code, out, _, calls = run_cli(
                ["profiles", "add", "claude", "work", "--exec", "--json"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "login_failed")
        self.assertEqual(payload["rc"], 3)
        self.assertIn(dest, payload["detail"])
        self.assertIn("left in place", payload["detail"])
        self.assertTrue(os.path.isdir(dest))
        self.assertTrue(os.path.isfile(os.path.join(dest, "keep")))

    def test_add_oserror_is_runtime_error_envelope(self):
        with mock.patch("providers.profile_check.add_profile",
                        side_effect=OSError("disk full")):
            code, out, err, calls = run_cli(
                ["profiles", "add", "claude", "work", "--json"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "runtime_error")
        self.assertIn("disk full", payload["detail"])

    def test_check_strict_quota_exits_2(self):
        listed = [{
            "road": "codex", "name": "a", "source": "env", "path": "/x",
            "identity": {"account_id_suffix": "xyz789"},
            "state": "identity_cached", "problems": [],
            "cooldown": "cooling",
        }]
        checked = {"ok": False, "profiles": [
            dict(listed[0], problems=["quota"])]}
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=listed), \
                mock.patch("providers.profile_check.check",
                           return_value=checked) as ck:
            code, out, _, calls = run_cli(
                ["profiles", "check", "--strict-quota", "--json"],
                {"ok": True})
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        self.assertEqual(ck.call_args.kwargs.get("strict_quota"), True)
        self.assertFalse(json.loads(out)["ok"])

    def test_check_default_does_not_pass_strict_quota(self):
        listed = [{
            "road": "codex", "name": "a", "source": "env", "path": "/x",
            "identity": {"account_id_suffix": "xyz789"},
            "state": "identity_cached", "problems": [],
            "cooldown": "cooling",
        }]
        checked = {"ok": True, "profiles": [dict(listed[0])]}
        with mock.patch("providers.profile_check.list_profiles",
                        return_value=listed), \
                mock.patch("providers.profile_check.check",
                           return_value=checked) as ck:
            code, out, _, calls = run_cli(
                ["profiles", "check", "--json"],
                {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(ck.call_args.kwargs.get("strict_quota"), False)
        self.assertTrue(json.loads(out)["ok"])

    def test_add_cursor_persist_emits_snippet(self):
        envelope = {
            "ok": False,
            "reason": "unsupported_persist",
            "detail": "cursor profiles are CURSOR_API_KEY_<NAME>; nothing to persist",
            "mcp_env": {"CURSOR_API_KEY_WORK": ""},
            "road": "cursor", "name": "work",
        }
        with mock.patch("providers.profile_check.add_profile",
                        return_value=envelope) as add:
            code, out, _, calls = run_cli(
                ["profiles", "add", "cursor", "work", "--persist", "--json"],
                {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        add.assert_called_once_with("cursor", "work", persist=True)
        payload = json.loads(out)
        self.assertEqual(payload["reason"], "unsupported_persist")
        self.assertEqual(payload["mcp_env"]["CURSOR_API_KEY_WORK"], "")

    def test_profiles_help_lists_subcommands(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "omni_cli.py"),
             "profiles", "--help"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("list", proc.stdout)
        self.assertIn("check", proc.stdout)
        self.assertIn("add", proc.stdout)
        self.assertIn("schedule", proc.stdout)

    def test_schedule_print_plist_shape(self):
        code, out, err, calls = run_cli(
            ["profiles", "schedule", "--print"], {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        self.assertIn("com.omni-llm.profiles-probe", out)
        self.assertIn("<key>ProgramArguments</key>", out)
        self.assertIn(os.path.realpath(sys.executable), out)
        self.assertIn(os.path.realpath(omni_cli.__file__), out)
        self.assertIn("<string>health</string>", out)
        self.assertIn("<string>--probe</string>", out)
        self.assertIn("<string>--all-profiles</string>", out)
        self.assertIn("<string>--json</string>", out)
        self.assertIn("<string>--quiet</string>", out)
        self.assertIn("<integer>900</integer>", out)
        self.assertIn("<key>StandardOutPath</key>", out)
        self.assertIn("/logs/", out)
        self.assertIn("profiles-probe.log", out)

    def test_schedule_install_writes_and_bootstraps(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch.object(omni_cli, "_linked_worktree_checkout",
                                  return_value=False), \
                mock.patch("subprocess.run", return_value=completed) as sp:
            code, out, err, calls = run_cli(
                ["profiles", "schedule", "--install"], {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        plist = os.path.join(
            home, "Library", "LaunchAgents",
            "com.omni-llm.profiles-probe.plist")
        self.assertTrue(os.path.isfile(plist))
        self.assertEqual(os.stat(plist).st_mode & 0o777, 0o644)
        sp.assert_called_once()
        argv = sp.call_args[0][0]
        self.assertEqual(argv[:3], [
            "launchctl", "bootstrap", "gui/%d" % os.getuid()])
        self.assertEqual(argv[3], plist)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        logs = os.path.join(state, "logs")
        self.assertTrue(os.path.isdir(logs))
        self.assertEqual(os.stat(logs).st_mode & 0o777, 0o700)

    def test_schedule_install_launchctl_failure_is_loud(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        failed = mock.Mock(returncode=5, stdout="", stderr="bootstrap failed")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch.object(omni_cli, "_linked_worktree_checkout",
                                  return_value=False), \
                mock.patch("subprocess.run", return_value=failed):
            code, out, err, calls = run_cli(
                ["profiles", "schedule", "--install"], {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["rc"], 5)
        self.assertIn("rc=5", payload["detail"])

    def test_schedule_install_without_force_refuses_existing(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        agents = os.path.join(home, "Library", "LaunchAgents")
        os.makedirs(agents)
        plist = os.path.join(agents, "com.omni-llm.profiles-probe.plist")
        with open(plist, "w", encoding="utf-8") as handle:
            handle.write("old\n")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch("subprocess.run") as sp:
            code, out, _, calls = run_cli(
                ["profiles", "schedule", "--install"], {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        sp.assert_not_called()
        payload = json.loads(out)
        self.assertEqual(payload["reason"], "already_exists")
        with open(plist, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "old\n")

    def test_schedule_install_refuses_linked_worktree(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        wt = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(wt, ignore_errors=True))
        with open(os.path.join(wt, ".git"), "w", encoding="utf-8") as handle:
            handle.write("gitdir: /somewhere\n")
        fake_cli = os.path.join(wt, "omni_cli.py")
        with open(fake_cli, "w", encoding="utf-8") as handle:
            handle.write("# fake\n")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch.object(omni_cli, "__file__", fake_cli), \
                mock.patch("subprocess.run") as sp:
            code, out, _, calls = run_cli(
                ["profiles", "schedule", "--install"], {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        sp.assert_not_called()
        payload = json.loads(out)
        self.assertEqual(payload["reason"], "worktree_checkout")

    def test_schedule_install_worktree_ok_with_force(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        wt = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(wt, ignore_errors=True))
        with open(os.path.join(wt, ".git"), "w", encoding="utf-8") as handle:
            handle.write("gitdir: /somewhere\n")
        fake_cli = os.path.join(wt, "omni_cli.py")
        with open(fake_cli, "w", encoding="utf-8") as handle:
            handle.write("# fake\n")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch.object(omni_cli, "__file__", fake_cli), \
                mock.patch("subprocess.run", return_value=completed) as sp:
            code, out, _, calls = run_cli(
                ["profiles", "schedule", "--install", "--force"], {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        sp.assert_called_once()

    def test_write_text_file_failure_leaves_old_intact(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        path = os.path.join(directory, "com.omni-llm.profiles-probe.plist")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("OLD-CONTENT\n")
        with mock.patch("os.fdopen", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                omni_cli._write_text_file(path, "NEW-CONTENT", 0o644)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "OLD-CONTENT\n")

    def test_schedule_install_rotates_large_log(self):
        home = tempfile.mkdtemp()
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        logs = os.path.join(state, "logs")
        os.makedirs(logs)
        log_path = os.path.join(logs, "profiles-probe.log")
        with open(log_path, "wb") as handle:
            handle.write(b"x" * (1024 * 1024 + 1))
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.dict(os.environ, {
                "HOME": home, "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch.object(omni_cli, "_linked_worktree_checkout",
                                  return_value=False), \
                mock.patch("subprocess.run", return_value=completed):
            code, out, _, calls = run_cli(
                ["profiles", "schedule", "--install"], {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertFalse(os.path.isfile(log_path))
        rotated = log_path + ".1"
        self.assertTrue(os.path.isfile(rotated))
        self.assertEqual(os.path.getsize(rotated), 1024 * 1024 + 1)
        self.assertTrue(json.loads(out)["ok"])


class TestHealthAllProfilesCli(unittest.TestCase):
    def test_all_profiles_requires_probe(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with contextlib.redirect_stderr(stderr), \
                contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                omni_cli.main(["health", "--all-profiles"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--all-profiles requires --probe", stderr.getvalue())

    def test_health_all_profiles_uses_probe_all(self):
        canned = {
            "codex": {"default": {"used_percent": 1, "reached": False}},
            "claude": {"default": {"signal": "none", "cooldown": "healthy"}},
        }
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch("providers.profile_check.probe_all",
                           return_value=canned) as pa:
            code, out, err, calls = run_cli(
                ["health", "--probe", "--all-profiles", "--json"],
                {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        pa.assert_called_once()
        payload = json.loads(out)
        self.assertEqual(payload["quota"]["claude"]["default"]["signal"],
                         "none")
        self.assertEqual(payload["quota"]["codex"]["default"]["used_percent"],
                         1)
        self.assertEqual(payload["jobs_repaired"], [])

    def test_health_quiet_payload_shape(self):
        canned = {
            "codex": {"default": {}, "b": {}},
            "claude": {"default": {}},
        }
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False), \
                mock.patch("providers.profile_check.probe_all",
                           return_value=canned):
            code, out, err, calls = run_cli(
                ["health", "--probe", "--all-profiles", "--quiet"],
                {"ok": True})
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(err, "")
        self.assertEqual(out.count("\n"), 1)
        payload = json.loads(out)
        self.assertEqual(set(payload.keys()),
                         {"ok", "probed", "cooling", "repaired"})
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["probed"], 3)
        self.assertIsInstance(payload["cooling"], list)
        self.assertEqual(payload["repaired"], 0)


class TestHealthLeaseCli(unittest.TestCase):
    def test_l7_health_json_leases_key_only_when_active(self):
        import uuid
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        token = str(uuid.uuid4())
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False):
            code, out, err, _calls = run_cli(["health", "--json"], {"ok": True})
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            payload = json.loads(out)
            self.assertNotIn("leases", payload)
            code, out, err, _calls = run_cli([
                "health", "lease", "acquire",
                "--token", token, "--pool", "codex-default",
                "--order", "wo-l7", "--ttl", "60",
            ], {"ok": True})
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            acquired = json.loads(out)
            self.assertTrue(acquired["ok"])
            self.assertEqual(acquired["lease"]["token"], token)
            code, out, err, _calls = run_cli(["health", "--json"], {"ok": True})
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertIn("leases", payload)
            self.assertGreaterEqual(len(payload["leases"]), 1)

    def test_health_json_ok_when_state_dir_read_only(self):
        import uuid
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        token = str(uuid.uuid4())
        try:
            with mock.patch.dict(os.environ, {
                    "OMNI_LLM_STATE_DIR": state}, clear=False):
                code, out, err, _calls = run_cli([
                    "health", "lease", "acquire",
                    "--token", token, "--pool", "codex-default",
                    "--order", "wo-ro", "--ttl", "60",
                ], {"ok": True})
                self.assertEqual(code, 0, out)
                os.chmod(state, 0o500)
                try:
                    code, out, err, _calls = run_cli(
                        ["health", "--json"], {"ok": True})
                finally:
                    os.chmod(state, 0o700)
            self.assertEqual(code, 0, out)
            self.assertEqual(err, "")
            payload = json.loads(out)
            self.assertIn("leases", payload)
            self.assertEqual(payload["leases"][0]["token"], token)
        finally:
            os.chmod(state, 0o700)

    def test_lease_acquire_bad_ttl_is_loud(self):
        import uuid
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        token = str(uuid.uuid4())
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False):
            code, out, err, _calls = run_cli([
                "health", "lease", "acquire",
                "--token", token, "--pool", "codex-default",
                "--order", "wo-ttl", "--ttl", "0",
            ], {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "bad_ttl")

    def test_lease_renew_bad_ttl_is_loud(self):
        import uuid
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        token = str(uuid.uuid4())
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False):
            code, out, err, _calls = run_cli([
                "health", "lease", "acquire",
                "--token", token, "--pool", "codex-default",
                "--order", "wo-ttl", "--ttl", "60",
            ], {"ok": True})
            self.assertEqual(code, 0, out)
            code, out, err, _calls = run_cli([
                "health", "lease", "renew",
                "--token", token, "--ttl", "-1",
            ], {"ok": True})
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "bad_ttl")

    def test_lease_acquire_lock_skip_is_health_skipped(self):
        import uuid
        state = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(state, ignore_errors=True))
        token = str(uuid.uuid4())
        seed = str(uuid.uuid4())
        ledger = os.path.join(state, "health.json")
        with mock.patch.dict(os.environ, {
                "OMNI_LLM_STATE_DIR": state}, clear=False):
            seed_code, seed_out, _seed_err, _calls = run_cli([
                "health", "lease", "acquire",
                "--token", seed, "--pool", "codex-default",
                "--order", "wo-seed", "--ttl", "60",
            ], {"ok": True})
            self.assertEqual(seed_code, 0, seed_out)
            before = None
            if os.path.exists(ledger):
                with open(ledger, encoding="utf-8") as handle:
                    before = handle.read()
            with mock.patch.object(
                    omni_cli.health, "try_exclusive_lock", return_value=None):
                code, out, err, _calls = run_cli([
                    "health", "lease", "acquire",
                    "--token", token, "--pool", "codex-default",
                    "--order", "wo-skip", "--ttl", "60",
                ], {"ok": True})
            after = None
            if os.path.exists(ledger):
                with open(ledger, encoding="utf-8") as handle:
                    after = handle.read()
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "health_skipped")
        self.assertEqual(after, before)


class TestAgentJobsCli(unittest.TestCase):
    def test_agent_jobs_payload_includes_repaired(self):
        listed = [{"job_id": "abc", "state": "failed"}]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(omni_cli.jobs_mod, "sweep_orphans",
                               side_effect=AssertionError(
                                   "sweep must go through list_jobs")), \
                mock.patch.object(omni_cli.jobs_mod, "list_jobs",
                                  return_value={"jobs": listed,
                                                "repaired": ["abc"]}), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = omni_cli.main(["agent", "jobs"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["repaired"], ["abc"])
        self.assertEqual(payload["jobs"], listed)


class TestExecutableSmoke(unittest.TestCase):
    def test_help_runs_from_a_symlink(self):
        """--help must work via a symlink (the ~/.local/bin install path)."""
        with tempfile.TemporaryDirectory() as d:
            link = os.path.join(d, "omni")
            os.symlink(os.path.join(HERE, "omni_cli.py"), link)
            proc = subprocess.run([sys.executable, link, "--help"],
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("gen", proc.stdout)
            self.assertIn("profiles", proc.stdout)


if __name__ == "__main__":
    unittest.main()
