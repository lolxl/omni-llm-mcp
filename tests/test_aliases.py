# no-opencc
"""User alias file. No vendor CLI, no network."""
import asyncio
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

import omni_cli
from providers import admission, agent_lane, catalog_v2, dispatch
from providers import cursor as cursor_mod
from providers import claude as claude_mod
from providers import codex as codex_mod
from providers import egress_canary, profile_check
from providers import grok_image_cli
from providers.catalog_record import validate_strict
from providers.config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    FABLE_MODEL,
    DEFAULT_CURSOR_MODEL,
    SOL_CODEX_MODEL,
    SOL_CURSOR_MODEL,
)
from providers.grok_image_cli import GROK_IMAGE_MODEL

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True

FRESH = "2026-11-10T00:00:00Z"
BIN = {
    "codex": "/fake/bin/codex",
    "cursor": "/fake/bin/cursor-agent",
    "grok": "/fake/bin/grok",
    "claude": "/fake/bin/claude",
}


def _record(provider, rows):
    models = []
    for row in rows:
        item = {"id": row["id"], "name": row["id"],
                "routable_id": "%s/%s" % (provider, row["id"])}
        if "efforts" in row:
            item["supported_efforts"] = list(row["efforts"])
        models.append(item)
    value = {
        "provider": provider,
        "source": "live",
        "observed_at": FRESH,
        "availability": "provider_advertised",
        "models": models,
    }
    record, _warnings = validate_strict(provider, value)
    return record


def _save(provider, profile, rows):
    catalog_v2.save(provider, profile, _record(provider, rows))


class FakeRun:
    def __init__(self):
        self.calls = []

    async def __call__(self, argv=None, **kwargs):
        argv = list(argv or [])
        self.calls.append(argv)
        self.stdout = "OK"
        if argv and os.path.basename(argv[0]) == "claude":
            self.stdout = '{"result":"OK"}'
        if "-o" in argv:
            path = argv[argv.index("-o") + 1]
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("OK")
        return {"returncode": 0, "stdout": self.stdout, "stderr": "",
                "timed_out": False, "signal": None}


def _which(name, *args, **kwargs):
    base = os.path.basename(str(name))
    if base in ("codex", "grok", "claude"):
        return BIN[base]
    if base == "cursor-agent":
        return BIN["cursor"]
    return None


def _fence(argv, **kwargs):
    return list(argv), kwargs.get("env") or {}, {"fence": "none", "fence_detail": {}}


class AliasCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory(prefix="omni-alias-")
        self.addCleanup(self.state.cleanup)
        self.work = tempfile.TemporaryDirectory(prefix="omni-alias-wd-")
        self.addCleanup(self.work.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["OMNI_LLM_STATE_DIR"] = self.state.name
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "0"
        for key in ("OMNI_LLM_DEPTH", "OPENAI_BASE_URL", "OPENAI_API_KEY",
                    "GROK_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_CONFIG_DIR"):
            os.environ.pop(key, None)
        for key in list(os.environ):
            if key.startswith("OMNI_LLM_FAMILY_") or key.startswith("OMNI_LLM_PROFILES_"):
                os.environ.pop(key, None)

    def _patches(self, fake):
        return [
            mock.patch.object(codex_mod, "run_cli", fake),
            mock.patch.object(cursor_mod, "run_cli", fake),
            mock.patch.object(claude_mod, "run_cli", fake),
            mock.patch.object(agent_lane, "run_cli", fake),
            mock.patch.object(codex_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(cursor_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(claude_mod, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(agent_lane, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(codex_mod.vendor_bin, "which", side_effect=_which),
            mock.patch.object(agent_lane.vendor_bin, "which", side_effect=_which),
            mock.patch.object(claude_mod, "resolve_agent_binary", return_value=BIN["claude"]),
            mock.patch.object(agent_lane, "resolve_agent_binary",
                              side_effect=lambda provider: BIN.get(provider)),
            mock.patch.object(cursor_mod, "resolve_bin", return_value=BIN["cursor"]),
            mock.patch("shutil.which", side_effect=_which),
        ]

    def _run(self, fake, coro):
        patches = self._patches(fake)
        for patch in patches:
            patch.start()
        try:
            return asyncio.run(coro)
        finally:
            for patch in reversed(patches):
                patch.stop()

    def _write(self, payload):
        path = os.path.join(self.state.name, "aliases.json")
        with open(path, "w", encoding="utf-8") as handle:
            if isinstance(payload, str):
                handle.write(payload)
            else:
                json.dump(payload, handle)
        return path

    def _seed_dispatch(self):
        _save("codex", "default", [
            {"id": SOL_CODEX_MODEL, "efforts": ["medium", "high", "xhigh"]},
            {"id": "gpt-alias", "efforts": ["medium", "high", "xhigh"]},
            {"id": "gpt-5.4", "efforts": ["low", "medium", "high", "xhigh"]},
        ])
        _save("cursor", "default", [
            {"id": SOL_CURSOR_MODEL + "-medium"},
            {"id": "composer-2.5"},
            {"id": "claude-fable-5-2-medium"},
        ])
        _save("grok", "default", [
            {"id": GROK_IMAGE_MODEL, "efforts": ["medium", "high"]},
            {"id": "grok-image-alias", "efforts": ["medium", "high"]},
        ])
        _save("claude", "default", [
            {"id": FABLE_MODEL, "efforts": ["medium", "high"]},
            {"id": "claude-alias", "efforts": ["medium", "high"]},
        ])

    def test_show_set_unset_round_trip_and_depth(self):
        code, out, err = _cli(["models", "alias", "show"])
        self.assertEqual(code, 0, out)
        self.assertEqual(err, "")
        rows = {row["key"]: row for row in json.loads(out)["aliases"]}
        self.assertEqual(rows["effort.codex"]["value"], "medium")
        self.assertEqual(rows["effort.codex"]["origin"], "built-in")
        self.assertEqual(rows["image.grok"]["value"], GROK_IMAGE_MODEL)
        self.assertEqual(rows["family.sol.cursor"]["value"], SOL_CURSOR_MODEL)
        self.assertTrue(all(row["origin"] == "built-in" for row in rows.values()))

        code, out, err = _cli(["models", "alias", "set", "family.sol.cursor", "composer-2.5"])
        self.assertEqual(code, 0, out)
        self.assertIn("warning:", err)
        self.assertTrue(json.loads(out)["ok"])
        shown = {row["key"]: row for row in json.loads(_cli(["models", "alias", "show"])[1])["aliases"]}
        self.assertEqual(shown["family.sol.cursor"]["value"], "composer-2.5")
        self.assertEqual(shown["family.sol.cursor"]["origin"], "file")
        self.assertEqual(shown["effort.codex"]["origin"], "built-in")

        code, out, err = _cli(["models", "alias", "unset", "family.sol.cursor"])
        self.assertEqual(code, 0, out)
        shown = {row["key"]: row for row in json.loads(_cli(["models", "alias", "show"])[1])["aliases"]}
        self.assertEqual(shown["family.sol.cursor"]["origin"], "built-in")
        self.assertEqual(shown["family.sol.cursor"]["value"], SOL_CURSOR_MODEL)
        self.assertFalse(os.path.exists(os.path.join(self.state.name, "aliases.json")))

        os.environ["OMNI_LLM_DEPTH"] = "1"
        code, out, err = _cli(["models", "alias", "set", "effort.codex", "xhigh"])
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(out)["reason"], "recursion_guard")
        self.assertFalse(os.path.exists(os.path.join(self.state.name, "aliases.json")))
        code, out, _err = _cli(["models", "alias", "unset", "effort.codex"])
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(out)["reason"], "recursion_guard")

    def test_refused_value_warns_and_still_writes(self):
        code, out, err = _cli(["models", "alias", "set", "image.grok", "not-listed"])
        self.assertEqual(code, 0, out)
        self.assertIn("not admitted", err)
        self.assertIn("warning", json.loads(out)["warning"])
        with open(os.path.join(self.state.name, "aliases.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["image.grok"], "not-listed")

    def test_unknown_key_does_not_write(self):
        code, out, _err = _cli(["models", "alias", "set", "effort.grok", "high"])
        self.assertNotEqual(code, 0)
        self.assertIn("unknown alias key", json.loads(out)["detail"])
        self.assertFalse(os.path.exists(os.path.join(self.state.name, "aliases.json")))

    def test_absent_file_matches_today_on_catalog_and_today_gates(self):
        self._seed_dispatch()
        fake = FakeRun()
        result = self._run(fake, codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort"), "medium")
        self.assertIn("model_reasoning_effort=medium", fake.calls[0])

        fake = FakeRun()
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
        result = self._run(fake, codex_mod.run_codex_subprocess("hi", "gpt-5.4", 30))
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertIn(
            "model_reasoning_effort=%s" % DEFAULT_CODEX_REASONING_EFFORT, fake.calls[0])
        os.environ.pop("OPENAI_BASE_URL")

        fake = FakeRun()
        result = self._run(fake, cursor_mod.run_cursor_subprocess("hi", "sol", 30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1],
                         SOL_CURSOR_MODEL + "-medium")

        argv = self._image(GROK_IMAGE_MODEL)
        self.assertIn(GROK_IMAGE_MODEL, argv)
        self.assertIn("-m", argv)
        argv = self._image(None, today=True)
        self.assertNotIn("-m", argv)

    def test_each_key_changes_catalog_dispatch_and_is_ignored_on_today(self):
        self._seed_dispatch()
        self._write({
            "family.sol.cursor": "composer-2.5",
            "family.sol.codex": "gpt-alias",
            "family.fable.claude": "claude-alias",
            "family.fable.cursor": "composer-2.5",
            "effort.codex": "xhigh",
            "image.grok": "grok-image-alias",
        })
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/gpt-5.4", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort"), "xhigh")
        self.assertEqual(result.get("effort_source"), "omni_default")
        self.assertIn("model_reasoning_effort=xhigh", fake.calls[0])
        self.assertEqual(fake.calls[0][fake.calls[0].index("-m") + 1], "gpt-5.4")

        fake = FakeRun()
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/gpt-5.4", 30, None, "default"))
        self.assertEqual(result.get("effort_source"), "today_gate")
        self.assertIn(
            "model_reasoning_effort=%s" % DEFAULT_CODEX_REASONING_EFFORT, fake.calls[0])
        self.assertNotIn("model_reasoning_effort=xhigh", fake.calls[0])
        os.environ.pop("OPENAI_BASE_URL")

        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "cursor/sol", 30, None, "default"))
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], "composer-2.5")

        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "cursor/" + SOL_CURSOR_MODEL, 30, None, "default"))
        self.assertEqual(
            fake.calls[0][fake.calls[0].index("--model") + 1], SOL_CURSOR_MODEL + "-medium")

        os.environ["OMNI_LLM_FAMILY_SOL"] = "codex"
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "sol", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("-m") + 1], "gpt-alias")

        fake = FakeRun()
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "sol", 30, None, "default"))
        self.assertEqual(result.get("effort_source"), "today_gate", result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("-m") + 1], SOL_CODEX_MODEL)
        self.assertIn(
            "model_reasoning_effort=%s" % DEFAULT_CODEX_REASONING_EFFORT, fake.calls[0])
        os.environ.pop("OPENAI_BASE_URL")
        os.environ.pop("OMNI_LLM_FAMILY_SOL")

        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/" + DEFAULT_CODEX_MODEL, 30, None, "default"))
        self.assertEqual(fake.calls[0][fake.calls[0].index("-m") + 1], DEFAULT_CODEX_MODEL)

        os.environ["OMNI_LLM_FAMILY_SOL"] = "cursor"
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "sol", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], "composer-2.5")
        os.environ.pop("OMNI_LLM_FAMILY_SOL")

        os.environ["OMNI_LLM_FAMILY_FABLE"] = "claude"
        # A clean config home and a cwd with no project settings keep this
        # seat on the catalog gate, so the file value is what the adapter gets.
        # The catalog file is keyed by that home, so seed after the switch.
        self._isolate_claude_catalog()
        self._seed_dispatch()
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "fable", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertNotEqual(result.get("effort_source"), "today_gate", result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], "claude-alias")
        os.environ.pop("OMNI_LLM_FAMILY_FABLE")

        os.environ["OMNI_LLM_FAMILY_FABLE"] = "cursor"
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "fable", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], "composer-2.5")
        os.environ.pop("OMNI_LLM_FAMILY_FABLE")

        argv = self._image("grok-image-alias")
        self.assertEqual(argv[argv.index("-m") + 1], "grok-image-alias")
        argv = self._image(None, today=True)
        self.assertNotIn("-m", argv)
        self.assertNotIn("grok-image-alias", argv)

    def test_hand_edited_unlisted_value_refuses_with_zero_spawns(self):
        self._seed_dispatch()
        self._write({"family.sol.codex": "no-such-model"})
        os.environ["OMNI_LLM_FAMILY_SOL"] = "codex"
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "sol", 30, None, "default"))
        self.assertEqual(result.get("reason"), "unsupported_model")
        self.assertEqual(fake.calls, [])

    def test_malformed_unknown_symlink_and_hardlink_refuse(self):
        self._seed_dispatch()
        path = self._write("{")
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/gpt-5.4", 30, None, "default"))
        self.assertEqual(result.get("reason"), "alias_file")
        self.assertIn("aliases.json", result.get("detail", ""))
        self.assertEqual(fake.calls, [])

        self._write({"effort.claude": "high"})
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "cursor/sol", 30, None, "default"))
        self.assertEqual(result.get("reason"), "alias_file")
        self.assertIn("unknown key", result.get("detail", ""))
        self.assertEqual(fake.calls, [])

        os.remove(path)
        os.symlink(os.path.join(self.state.name, "elsewhere"), path)
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request("hi", "sol", 30, None, "default"))
        self.assertEqual(result.get("reason"), "alias_file")
        self.assertIn("symlink", result.get("detail", ""))
        self.assertEqual(fake.calls, [])

        os.remove(path)
        self._write({"effort.codex": "high"})
        os.link(path, os.path.join(self.state.name, "aliases-hard.json"))
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/gpt-5.4", 30, None, "default"))
        self.assertEqual(result.get("reason"), "alias_file")
        self.assertIn("hard link", result.get("detail", ""))
        self.assertEqual(fake.calls, [])

    def test_probe_and_canary_ignore_alias_file(self):
        self._seed_dispatch()
        self._write({"effort.codex": "xhigh"})
        probed = self._probe()
        self.assertNotEqual(probed["result"].get("reason"), "alias_file", probed["result"])
        self.assertIn("model_reasoning_effort=medium", probed["argv"])
        self.assertNotIn("model_reasoning_effort=xhigh", probed["argv"])
        self._assert_canary_unchanged({"effort.codex": "xhigh"})

        self._write("{")
        probed = self._probe()
        self.assertNotEqual(probed["result"].get("reason"), "alias_file", probed["result"])
        self.assertIn("model_reasoning_effort=medium", probed["argv"])
        self._assert_canary_unchanged("{")

        self._write({"family.sol.cursor": "not-the-probe"})
        cursor = self._probe_cursor()
        self.assertNotEqual(cursor["result"].get("reason"), "alias_file", cursor["result"])
        self.assertIn(DEFAULT_CURSOR_MODEL, cursor["argv"])
        self.assertNotIn("not-the-probe", cursor["argv"])

        self._write("{")
        cursor = self._probe_cursor()
        self.assertNotEqual(cursor["result"].get("reason"), "alias_file", cursor["result"])
        self.assertIn(DEFAULT_CURSOR_MODEL, cursor["argv"])
        # The packaged cli_cursor seam calls this with no alias flag. On this
        # host the real fence rejects the temp config dir before transport, so
        # the seam's exit would stay 3 either way. Pass the fence through, as
        # the codex canary does, and require the transport stub.
        from providers.egress_canary import _FIXTURE_CURSOR
        _save("cursor", "default", [{"id": _FIXTURE_CURSOR}])
        result, argv = self._cursor_canary_call()
        self.assertNotEqual(result.get("reason"), "alias_file", result)
        self.assertIn(_FIXTURE_CURSOR, argv)

    def test_image_canary_ignores_a_malformed_alias_file(self):
        clean = self._canary("grok_image", None)
        dirty = self._canary("grok_image", "{")
        self.assertEqual(clean.returncode, 42, clean.stdout + clean.stderr)
        self.assertEqual(dirty.returncode, clean.returncode, dirty.stdout + dirty.stderr)
        self.assertIn("FAIL-OPEN-REACHED-TRANSPORT", dirty.stdout)
        self.assertNotIn("alias_file", dirty.stdout + dirty.stderr)
        self.assertEqual(clean.stdout, dirty.stdout)

    def test_managed_codex_ring_forwards_the_alias(self):
        self._seed_dispatch()
        self._statable_home()
        work = tempfile.mkdtemp(prefix="omni-alias-codex-")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        os.environ["CODEX_HOME_WORK"] = work
        os.environ["OMNI_LLM_PROFILES_CODEX"] = "work"
        _save("codex", "work", [
            {"id": "gpt-5.4", "efforts": ["low", "medium", "high", "xhigh"]},
        ])
        self._write({"effort.codex": "xhigh"})
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "codex/gpt-5.4", 30, None, "default"))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("account_profile"), "work")
        self.assertEqual(result.get("effort"), "xhigh")
        self.assertEqual(result.get("effort_source"), "omni_default")
        self.assertIn("model_reasoning_effort=xhigh", fake.calls[0])
        self.assertNotIn("model_reasoning_effort=medium", fake.calls[0])

    def test_dispatch_sees_alias_file_on_the_next_request(self):
        self._seed_dispatch()
        fake = FakeRun()
        first = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "cursor/sol", 30, None, "default"))
        self.assertTrue(first.get("ok"), first)
        self.assertEqual(
            fake.calls[0][fake.calls[0].index("--model") + 1], SOL_CURSOR_MODEL + "-medium")
        self._write({"family.sol.cursor": "composer-2.5"})
        fake = FakeRun()
        second = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "cursor/sol", 30, None, "default"))
        self.assertTrue(second.get("ok"), second)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], "composer-2.5")

    def test_refused_set_leaves_file_bytes_unchanged(self):
        from providers.aliases import AliasError, set_alias
        path = self._write({"image.grok": GROK_IMAGE_MODEL})
        with open(path, "rb") as handle:
            before = handle.read()
        for bad in ("line\nbreak", "line\rbreak", "line\x00break"):
            with self.assertRaises(AliasError):
                set_alias("image.grok", bad)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_agent_dispatch_uses_the_alias(self):
        self._seed_dispatch()
        self._write({"effort.codex": "xhigh"})
        fake = FakeRun()
        # The workdir fence asks git before spawn. This temp dir is not a
        # repository; answer that without a nested sandbox-exec.
        not_repo = subprocess.CompletedProcess(
            [], 128, "", "fatal: not a git repository")
        with mock.patch.object(agent_lane.git_safe, "run_git", return_value=not_repo):
            result = self._run(fake, agent_lane.run_agent(
                "task", "codex/gpt-5.4", self.work.name, read_only=True, timeout_s=30))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("effort"), "xhigh")
        self.assertIn("model_reasoning_effort=xhigh", fake.calls[0])

    def test_today_ring_account_keeps_builtin_fable(self):
        self._statable_home()
        self._seed_dispatch()
        alias = "not-listed-fable"
        self._write({"family.fable.claude": alias})
        work = self._claude_home({"apiKeyHelper": "printf test"})
        clean = self._claude_home({})
        os.environ["CLAUDE_CONFIG_DIR"] = clean
        os.environ["CLAUDE_CONFIG_DIR_WORK"] = work
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "work"
        os.environ["OMNI_LLM_FAMILY_FABLE"] = "claude"
        from providers.claude import consult_settings_files
        self.assertEqual(admission.gate(
            "claude", "default", isolated=True, host_env=os.environ,
            settings_files=consult_settings_files("default")), "catalog")
        self.assertEqual(admission.gate(
            "claude", "work", isolated=True, host_env=os.environ,
            settings_files=consult_settings_files("work", work)), "today")
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "fable", 30, None, "default"))
        self.assertTrue(fake.calls, result)
        sent = fake.calls[0][fake.calls[0].index("--model") + 1]
        self.assertEqual(sent, FABLE_MODEL)
        self.assertNotEqual(sent, alias)
        self.assertNotIn(alias, fake.calls[0])

    def test_malformed_file_does_not_block_today_family(self):
        self._statable_home()
        self._seed_dispatch()
        self._write("{")
        work = self._claude_home({"apiKeyHelper": "printf test"})
        clean = self._claude_home({})
        os.environ["CLAUDE_CONFIG_DIR"] = clean
        os.environ["CLAUDE_CONFIG_DIR_WORK"] = work
        os.environ["OMNI_LLM_PROFILES_CLAUDE"] = "work"
        os.environ["OMNI_LLM_FAMILY_FABLE"] = "claude"
        fake = FakeRun()
        result = self._run(fake, dispatch.dispatch_llm_request(
            "hi", "fable", 30, None, "default"))
        self.assertNotEqual(result.get("reason"), "alias_file", result)
        self.assertTrue(fake.calls, result)
        self.assertEqual(fake.calls[0][fake.calls[0].index("--model") + 1], FABLE_MODEL)

    def _claude_home(self, settings):
        path = tempfile.mkdtemp(prefix="omni-alias-claude-")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "settings.json"), "w", encoding="utf-8") as handle:
            json.dump(settings, handle)
        return path

    def _statable_home(self):
        """Home whose default vendor dirs can be statted.

        The seatbelt in this process cannot stat the real ``~/.claude``. Profile
        admission then treats every directory as that default login. A private
        home with a real ``.claude`` keeps the outside-home rule testable.
        """
        home = tempfile.mkdtemp(prefix="omni-alias-home-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        os.makedirs(os.path.join(home, ".claude"))
        patcher = mock.patch(
            "providers.kernel_sandbox.protected_home",
            return_value=os.path.realpath(home))
        patcher.start()
        self.addCleanup(patcher.stop)
        return os.path.realpath(home)

    def _isolate_claude_catalog(self):
        """Point consult at a readable empty home and a cwd with no project settings."""
        os.environ["CLAUDE_CONFIG_DIR"] = self._claude_home({})
        scratch = tempfile.mkdtemp(prefix="omni-alias-cwd-")
        self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        previous = os.getcwd()
        os.chdir(scratch)
        self.addCleanup(os.chdir, previous)

    def _probe_road(self, road):
        fake = FakeRun()
        patches = self._patches(fake)
        for patch in patches:
            patch.start()
        try:
            result = profile_check._live_one(road, "default")
        finally:
            for patch in reversed(patches):
                patch.stop()
        return {"result": result, "argv": fake.calls[0] if fake.calls else []}

    def _probe(self):
        return self._probe_road("codex")

    def _probe_cursor(self):
        return self._probe_road("cursor")

    def _cursor_canary_call(self):
        """``cli_cursor``'s call. The alias flag stays omitted."""
        seen = []

        async def transport(argv=None, **kwargs):
            seen.append(list(argv or []))
            return {"returncode": 0, "stdout": "OK", "stderr": "",
                    "timed_out": False, "signal": None}

        with mock.patch.object(cursor_mod, "resolve_bin",
                                return_value="/o46-stub-cursor-agent"), \
                mock.patch.object(cursor_mod, "prepare_host_fence", side_effect=_fence), \
                mock.patch.object(cursor_mod, "run_cli", side_effect=transport):
            result = asyncio.run(cursor_mod.run_cursor_subprocess(
                "hello", "fixture-cursor", 5, "default"))
        argv = seen[0] if seen else []
        return result, argv

    def _canary(self, seam, payload):
        state = tempfile.mkdtemp(prefix="omni-alias-canary-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        if payload is not None:
            with open(os.path.join(state, "aliases.json"), "w", encoding="utf-8") as handle:
                if isinstance(payload, str):
                    handle.write(payload)
                else:
                    json.dump(payload, handle)
        cmd = [sys.executable, "-m", "providers.egress_canary",
               "--seam", seam, "--gate", "0", "--text=hello",
               "--state", state]
        if seam == "grok_image":
            image = os.path.join(state, "src.png")
            with open(image, "wb") as handle:
                handle.write(b"png")
            cmd.extend(["--image", image])
        return subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, OMNI_LLM_QUOTA_PROBE="0", OMNI_LLM_STATE_DIR=state),
        )

    def _assert_canary_unchanged(self, payload, seam="cli_codex"):
        proc = self._canary(seam, payload)
        self.assertEqual(proc.returncode, 42, proc.stdout + proc.stderr)
        self.assertIn("FAIL-OPEN-REACHED-TRANSPORT", proc.stdout)
        self.assertNotIn("alias_file", proc.stdout + proc.stderr)

    def _image(self, expect_model, today=False):
        source = os.path.join(self.work.name, "src.png")
        with open(source, "wb") as handle:
            handle.write(b"png")
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            cwd = kwargs.get("cwd") or self.work.name
            with open(os.path.join(cwd, "image_edit.png"), "wb") as handle:
                handle.write(b"png")
            return subprocess.CompletedProcess(argv, 0, b"ok", b"")

        def consider_today(*args, **kwargs):
            # Discovery reads os.environ. An auth name only on the attempt
            # is today's gate, so the image entrypoint returns before the file.
            host = dict(kwargs.get("host_env") or os.environ)
            host["XAI_API_KEY"] = "synthetic-test"
            kwargs["host_env"] = host
            return _consider(*args, **kwargs)

        _consider = admission.consider
        stack = [
            mock.patch.object(grok_image_cli.vendor_bin, "which", return_value=BIN["grok"]),
            mock.patch.object(grok_image_cli, "prepare_host_fence", side_effect=_fence),
            mock.patch.object(grok_image_cli.subprocess, "run", side_effect=fake_run),
        ]
        if today:
            stack.append(mock.patch.object(admission, "consider", side_effect=consider_today))
        with contextlib.ExitStack() as stack_cm:
            for patch in stack:
                stack_cm.enter_context(patch)
            result = asyncio.run(grok_image_cli.run_grok_image_edit(
                "edit the picture", source, timeout_s=5, out_dir=self.work.name,
                apply_aliases=True))
        if expect_model is None:
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(len(calls), 1)
            return calls[0]
        if expect_model == "refuse":
            self.assertEqual(calls, [])
            return result
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(len(calls), 1)
        return calls[0]


def _cli(argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = omni_cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
