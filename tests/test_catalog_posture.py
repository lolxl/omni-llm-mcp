"""Posture for catalog discovery: seatbelt on Darwin, fixed argv elsewhere.

Fake binaries and a patched spawn only. No vendor CLI and no network.
"""
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from providers import catalog_record, catalog_v2, host_fence, model_catalog, vendor_bin
from providers.cursor import resolve_bin
from providers.agy import find_agy
from providers.kernel_sandbox import resolve_agent_binary
from tests import offline_home

# `from providers.host_fence import prepare_host_fence` binds the object that
# is current at first import. These modules do that at import time.
# model_catalog imports the name inside the functions that spawn, so it has
# no module global to restore.
_FENCE_CONSUMER_NAMES = (
    "providers.agent_lane",
    "providers.agy",
    "providers.claude",
    "providers.codex",
    "providers.cursor",
    "providers.grok",
    "providers.grok_image_cli",
    "providers.pi",
)
_ORIGINAL_PREPARE = host_fence.prepare_host_fence

_HOME = None

# Spawn prefix captured from the seatbelt wrap HEAD's catalog tests already use.
# Darwin discovery must hand this wrap to the child unchanged.
_HEAD_SEATBELT_WRAP = ["/usr/bin/sandbox-exec", "-p", "fixture"]
_PROVIDERS = ("cursor", "agy", "grok", "codex", "claude")
_TAILS = {
    "cursor": ("--list-models",),
    "agy": ("models",),
    "grok": ("models",),
    "codex": ("app-server",),
}
_CLAUDE_TAIL = (
    "--print", "--input-format", "stream-json",
    "--output-format", "stream-json", "--verbose", "--tools", "",
    "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}",
    "--setting-sources", "", "--no-session-persistence",
    "--disable-slash-commands",
)
_MODEL_IDS = {
    "cursor": "posture-cursor",
    "agy": "posture-agy",
    "grok": "posture-grok",
    "codex": "posture-codex",
    "claude": "posture-claude",
}
# Sibling profile names a default-profile child must not keep. Written here,
# not read from providers.profiles.
_SIBLING_PREFIXES = (
    "CLAUDE_CONFIG_DIR_",
    "CODEX_HOME_",
    "GROK_HOME_",
    "CURSOR_API_KEY_",
    "OMNI_LLM_CURSOR_CONFIG_DIR_",
)
# Names codex account_env removes from the child, including other vendors'
# auth and the parent Claude launch context. Written here, not read from
# providers.provider_accounts.
_CODEX_CHILD_DROPS = (
    "CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    "XAI_API_KEY", "GROK_API_KEY", "XAI_BASE_URL", "XAI_API_BASE",
    "GROK_BASE_URL", "GROK_API_BASE", "GROK_CLI_CHAT_PROXY_BASE_URL",
    "GROK_WS_URL", "GROK_AUTH_PROVIDER_COMMAND", "GROK_MODELS_BASE_URL",
    "GROK_MODELS_LIST_URL", "GROK_XAI_API_BASE_URL", "GROK_OIDC_ISSUER",
    "GROK_OIDC_CLIENT_ID", "GROK_OIDC_AUDIENCE", "GROK_OIDC_SCOPES",
    "GROK_OAUTH2_ISSUER", "GROK_OAUTH2_CLIENT_ID", "GROK_OAUTH2_PRINCIPAL_ID",
    "GROK_OAUTH2_PRINCIPAL_TYPE", "GROK_OAUTH2_REFERRER", "GROK_OAUTH2_SCOPES",
    "GROK_WS_ORIGIN",
)
_CODEX_LAUNCH_DROPS = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SSE_PORT",
)
# Present in the Darwin spawn test so a dropped or kept name is visible.
_DARWIN_SENTINELS = {
    "OMNI_LLM_NOTIFY_CMD": "/sentinel/notify",
    "OMNI_LLM_NOTIFY_QUIET": "0",
    "CLAUDE_CONFIG_DIR_WORK": "/sentinel/claude-work",
    "CODEX_HOME_WORK": "/sentinel/codex-work",
    "GROK_HOME_WORK": "/sentinel/grok-work",
    "CURSOR_API_KEY_WORK": "sentinel-cursor-key",
    "OMNI_LLM_CURSOR_CONFIG_DIR_WORK": "/sentinel/cursor-work",
    "OPENAI_API_KEY": "sentinel-openai-key",
    "OPENAI_API_KEY_WORK": "sentinel-openai-key-work",
    "CLAUDE_CODE_PLUGIN_DIRS": "/sentinel/plugins",
    "CLAUDECODE": "1",
}
# Record file HEAD 4b7185d writes for the inputs in
# test_non_darwin_codex_endpoint_refresh_reloads_under_same_fingerprint.
# Scratch run: identity key 32 bytes of 0x11, protected home
# /omni-d2f-codex-posture-home, profile default, no CODEX_HOME,
# OPENAI_BASE_URL https://codex-posture.example/v1.
_HEAD_CODEX_RECORD = "codex-73efb61ef9a27d6c17d59a77c9ae4b1f.json"


def setUpModule():
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def _assert_prepare_host_fence_restored():
    """The shared fence function, and every consumer from-import, is the original."""
    if host_fence.prepare_host_fence is not _ORIGINAL_PREPARE:
        raise AssertionError(
            "providers.host_fence.prepare_host_fence was not restored: %r"
            % (host_fence.prepare_host_fence,))
    leaked = []
    for name in _FENCE_CONSUMER_NAMES:
        module = sys.modules.get(name)
        if module is None or not hasattr(module, "prepare_host_fence"):
            continue
        bound = module.prepare_host_fence
        if bound is not _ORIGINAL_PREPARE:
            leaked.append("%s.prepare_host_fence is %r" % (name, bound))
    if leaked:
        raise AssertionError(
            "prepare_host_fence leaked into a consumer import: "
            + "; ".join(leaked))


@contextlib.contextmanager
def _prepare_patch(side_effect):
    """Patch the fence, then rebind consumers that imported the mock.

    The patcher restores ``providers.host_fence.prepare_host_fence`` on the
    way out. A consumer imported for the first time inside the patch keeps
    the mock in its own global, so rebind only after that restore.
    """
    try:
        with mock.patch("providers.host_fence.prepare_host_fence",
                        side_effect=side_effect):
            yield
    finally:
        # After the patcher has put the real function back. A consumer whose
        # first import happened inside the patch still holds the mock.
        real = host_fence.prepare_host_fence
        for name in _FENCE_CONSUMER_NAMES:
            module = sys.modules.get(name)
            if module is None or not hasattr(module, "prepare_host_fence"):
                continue
            if module.prepare_host_fence is not real:
                module.prepare_host_fence = real


def tearDownModule():
    try:
        _assert_prepare_host_fence_restored()
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _literal_discovery_env(provider):
    """Child environment a Darwin catalog spawn must receive.

    Built from the process environment and the sentinels this test sets.
    It does not call the catalog builder.
    """
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(_SIBLING_PREFIXES)
    }
    env.pop("OMNI_LLM_NOTIFY_CMD", None)
    env["OMNI_LLM_NOTIFY_QUIET"] = "1"
    if provider == "codex":
        for key in list(env):
            if (key in _CODEX_CHILD_DROPS or key in _CODEX_LAUNCH_DROPS
                    or any(key.startswith(name + "_") for name in _CODEX_CHILD_DROPS)):
                del env[key]
    elif provider == "claude":
        env.pop("CLAUDE_CODE_PLUGIN_DIRS", None)
    return env


def _table_argv(provider, binary):
    if provider == "claude":
        return [binary, *_CLAUDE_TAIL]
    return [binary, *_TAILS[provider]]


def _stdout_for(provider):
    if provider == "cursor":
        return b"Available models\nposture-cursor - Posture Cursor\nTip: listed\n"
    if provider == "agy":
        return b"posture-agy\tPosture Agy\n"
    if provider == "grok":
        return b"Available models:\n- posture-grok\n"
    if provider == "codex":
        messages = [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {
                "data": [{"model": "posture-codex", "displayName": "Posture Codex"}],
                "nextCursor": None}},
        ]
    else:
        messages = [{"type": "control_response", "response": {
            "request_id": "catalog-init",
            "response": {"models": [{
                "value": "posture-claude", "displayName": "Posture Claude"}]}}}]
    return b"".join(
        (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")
        for item in messages)


class _FakeProc:
    """A Popen stand-in with real pipes. It never execs."""

    def __init__(self, stdout_bytes):
        self.pid = 99999999
        self.returncode = None
        out_read, out_write = os.pipe()
        os.write(out_write, stdout_bytes)
        os.close(out_write)
        self.stdout = os.fdopen(out_read, "rb", buffering=0)
        in_read, in_write = os.pipe()
        self._stdin_read = in_read
        self.stdin = os.fdopen(in_write, "wb", buffering=0)
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9
        if self._stdin_read is not None:
            os.close(self._stdin_read)
            self._stdin_read = None


class CatalogPostureTests(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.state = tempfile.mkdtemp(prefix="omni-catalog-posture-")
        self.bindir = tempfile.mkdtemp(prefix="omni-catalog-bins-")
        self.addCleanup(shutil.rmtree, self.state, True)
        self.addCleanup(shutil.rmtree, self.bindir, True)
        self.bins = {}
        for name in ("cursor-agent", "agy", "grok", "codex", "claude"):
            path = os.path.join(self.bindir, name)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)
            self.bins[name] = os.path.realpath(path)
        removed = {}
        for key in ("CURSOR_CONFIG_DIR", "OMNI_LLM_CURSOR_CONFIG_DIR",
                    "OMNI_LLM_CURSOR_BIN", "OMNI_LLM_CLAUDE_BIN"):
            if key in os.environ:
                removed[key] = os.environ.pop(key)
        self.addCleanup(os.environ.update, removed)
        path = self.bindir + os.pathsep + os.environ.get("PATH", "")
        self.env = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state,
            "OMNI_LLM_NOTIFY_QUIET": "1",
            "PATH": path,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(_assert_prepare_host_fence_restored)

    def _binary(self, provider):
        if provider == "cursor":
            return resolve_bin()
        if provider == "agy":
            return find_agy()
        if provider == "grok":
            return os.path.realpath(vendor_bin.which("grok"))
        if provider == "codex":
            return os.path.realpath(vendor_bin.which("codex"))
        return resolve_agent_binary("claude")

    def _builder_env(self, provider):
        _overrides, child_env, _allowed, _home, _endpoints = (
            model_catalog._catalog_child_environment(provider, "default"))
        return dict(child_env)

    def _patch_spawn(self, provider):
        calls = []

        def popen(argv, **kwargs):
            cwd = kwargs.get("cwd")
            env = kwargs.get("env")
            config = env.get("CURSOR_CONFIG_DIR") if isinstance(env, dict) else None
            calls.append({
                "argv": list(argv),
                "cwd": cwd,
                "cwd_is_dir": bool(cwd) and os.path.isdir(cwd),
                "env": env,
                "kwargs": kwargs,
                "cursor_config_is_dir": bool(config) and os.path.isdir(config),
            })
            return _FakeProc(_stdout_for(provider))

        return calls, mock.patch.object(
            model_catalog.subprocess, "Popen", side_effect=popen)

    def _assert_spawn_contract(self, provider, call, expected_env, *, wrapped):
        binary = self._binary(provider)
        table = _table_argv(provider, binary)
        cwd = call["cwd"]
        self.assertTrue(call["cwd_is_dir"], cwd)
        if provider == "codex":
            self.assertTrue(os.path.basename(cwd).startswith("omni-catalog-codex-"))
        elif provider == "claude":
            self.assertTrue(os.path.basename(cwd).startswith("omni-catalog-claude-"))
        else:
            self.assertTrue(os.path.basename(cwd).startswith("omni-catalog-"))
        spawned_env = dict(call["env"])
        if provider == "cursor":
            config = spawned_env.pop("CURSOR_CONFIG_DIR")
            opt_in = spawned_env.pop("OMNI_LLM_CURSOR_CONFIG_DIR")
            self.assertEqual(config, opt_in)
            self.assertTrue(os.path.basename(config).startswith("omni-catalog-cursor-"))
            self.assertTrue(call["cursor_config_is_dir"], config)
        self.assertEqual(spawned_env, expected_env)
        kw = call["kwargs"]
        if provider in ("codex", "claude"):
            self.assertIs(kw["stdin"], subprocess.PIPE)
        else:
            self.assertIs(kw["stdin"], subprocess.DEVNULL)
        self.assertIs(kw["stdout"], subprocess.PIPE)
        self.assertIs(kw["stderr"], subprocess.DEVNULL)
        self.assertTrue(kw["start_new_session"])
        self.assertEqual(kw["bufsize"], 0)
        if wrapped:
            self.assertEqual(call["argv"], _HEAD_SEATBELT_WRAP + table)
        else:
            self.assertEqual(call["argv"], table)

    def test_non_darwin_table_argv_env_cwd_and_posture_roundtrip(self):
        """Guards: off Darwin, each refresh discoverer spawns only its table argv
        with the builder env and a temp cwd, and the posture survives validate,
        save, and reload.
        """
        for provider in _PROVIDERS:
            with self.subTest(provider=provider):
                expected_env = self._builder_env(provider)
                calls, popen_patch = self._patch_spawn(provider)
                with mock.patch("platform.system", return_value="Linux"), popen_patch:
                    result = model_catalog.refresh(provider, "default", 30)
                self.assertTrue(result["ok"], result)
                discovery = [call for call in calls
                             if call["argv"] == _table_argv(provider, self._binary(provider))]
                self.assertEqual(len(calls), 1, calls)
                self.assertEqual(discovery, calls)
                self._assert_spawn_contract(
                    provider, calls[0], expected_env, wrapped=False)
                loaded, info = catalog_v2.load(provider, "default")
                self.assertIsNotNone(loaded, info)
                self.assertEqual(loaded["models"][0]["id"], _MODEL_IDS[provider])
                checked, _warnings = catalog_record.validate_strict(provider, loaded)
                self.assertEqual(checked["discovery"], {
                    "posture": "metadata-unfenced", "platform": "Linux"})
                self.assertEqual(loaded["discovery"], checked["discovery"])

    def test_non_darwin_argv_outside_table_starts_no_child(self):
        """Guards: a metadata-posture argv outside the fixed table raises
        discovery_unavailable and does not start a child.
        """
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(_args)
            raise AssertionError("child started")

        binary = self._binary("cursor")
        with mock.patch("platform.system", return_value="Linux"), \
                mock.patch.object(model_catalog.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._spawn_one_shot(
                    "cursor", [binary, "--version"], "default", 5,
                    cursor_config=True)
        self.assertEqual(caught.exception.reason, "discovery_unavailable")
        self.assertEqual(calls, [])

        def widen(argv, **kwargs):
            return list(argv) + ["--extra"], kwargs.get("env"), {"fence": "none"}

        with mock.patch("platform.system", return_value="Linux"), \
                _prepare_patch(widen), \
                mock.patch.object(model_catalog.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._query_codex_models(
                    binary, env={"PATH": "/usr/bin", "CODEX_HOME": self.state},
                    cwd=self.state, timeout_s=5)
            self.assertEqual(caught.exception.reason, "discovery_unavailable")
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog._discover_claude("default", 5)
            self.assertEqual(caught.exception.reason, "discovery_unavailable")
        self.assertEqual(calls, [])

    def test_non_darwin_substituted_head_starts_no_child(self):
        """Guards: under metadata-unfenced, /bin/echo with a valid tail is
        discovery_unavailable for every spawn path, and no child starts.
        """
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(_args)
            raise AssertionError("child started")

        popen_patch = mock.patch.object(
            model_catalog.subprocess, "Popen", side_effect=popen)
        linux = mock.patch("platform.system", return_value="Linux")
        echo = "/bin/echo"
        with linux, popen_patch:
            for provider, tail in _TAILS.items():
                if provider == "codex":
                    continue
                with self.subTest(provider=provider):
                    with self.assertRaises(model_catalog.CatalogError) as caught:
                        model_catalog._spawn_one_shot(
                            provider, [echo, *tail], "default", 5,
                            cursor_config=(provider == "cursor"))
                    self.assertEqual(caught.exception.reason, "discovery_unavailable")

            with self.subTest(provider="codex"):
                with self.assertRaises(model_catalog.CatalogError) as caught:
                    model_catalog._query_codex_models(
                        echo, env={"PATH": "/usr/bin", "CODEX_HOME": self.state},
                        cwd=self.state, timeout_s=5)
                self.assertEqual(caught.exception.reason, "discovery_unavailable")

            def swap_head(argv, **kwargs):
                return [echo, *list(argv)[1:]], kwargs.get("env"), {"fence": "none"}

            with self.subTest(provider="claude"), \
                    _prepare_patch(swap_head):
                with self.assertRaises(model_catalog.CatalogError) as caught:
                    model_catalog._discover_claude("default", 5)
                self.assertEqual(caught.exception.reason, "discovery_unavailable")
        self.assertEqual(calls, [])

    def test_non_darwin_agy_effort_tokens_still_discovery_unavailable(self):
        """Guards: agy_effort_tokens still calls the seatbelt check, so dispatch
        stays discovery_unavailable when the fence is not seatbelt-fs.
        """
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(_args)
            raise AssertionError("child started")

        real_check = model_catalog._require_verified_metadata_fence

        def check(metadata, provider):
            calls.append(("check", metadata, provider))
            return real_check(metadata, provider)

        with mock.patch("platform.system", return_value="Linux"), \
                mock.patch.object(model_catalog.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(model_catalog, "_require_verified_metadata_fence",
                                  side_effect=check):
            with self.assertRaises(model_catalog.CatalogError) as caught:
                model_catalog.agy_effort_tokens()
        self.assertEqual(caught.exception.reason, "discovery_unavailable")
        self.assertEqual([item for item in calls if item and item[0] != "check"], [])
        checks = [item for item in calls if item and item[0] == "check"]
        self.assertEqual(len(checks), 1)
        self.assertNotEqual(checks[0][1].get("fence"), "seatbelt-fs")
        self.assertEqual(checks[0][2], "agy")

    def test_darwin_seatbelt_spawn_matches_head_and_records_posture(self):
        """Guards: on Darwin the argv, environment, and seatbelt metadata handling
        match HEAD (4b7185d): prepare receives the table argv, the child receives
        the seatbelt wrap unchanged, and a non-seatbelt fence starts nothing.
        The saved posture is seatbelt-fs.
        """
        def seatbelt(argv, **kwargs):
            return (_HEAD_SEATBELT_WRAP + list(argv), kwargs["env"],
                    {"fence": "seatbelt-fs",
                     "fence_detail": {"read_wall": True,
                                      "workdir_writable": False,
                                      "vendor_state_writable": False}})

        with mock.patch.dict(os.environ, _DARWIN_SENTINELS):
            for provider in _PROVIDERS:
                with self.subTest(provider=provider, phase="seatbelt"):
                    expected_env = _literal_discovery_env(provider)
                    calls, popen_patch = self._patch_spawn(provider)
                    with mock.patch("platform.system", return_value="Darwin"), \
                            _prepare_patch(seatbelt), \
                            popen_patch:
                        result = model_catalog.refresh(provider, "default", 30)
                    self.assertTrue(result["ok"], result)
                    table = _table_argv(provider, self._binary(provider))
                    discovery = [call for call in calls if call["argv"] == _HEAD_SEATBELT_WRAP + table]
                    version = [call for call in calls
                               if call["argv"] == _HEAD_SEATBELT_WRAP + [table[0], "--version"]]
                    self.assertEqual(len(discovery), 1, calls)
                    self.assertEqual(len(version), 1, calls)
                    self._assert_spawn_contract(
                        provider, discovery[0], expected_env, wrapped=True)
                    version_env = dict(version[0]["env"])
                    if provider == "cursor":
                        version_config = version_env.pop("CURSOR_CONFIG_DIR")
                        version_opt_in = version_env.pop("OMNI_LLM_CURSOR_CONFIG_DIR")
                        self.assertEqual(version_config, version_opt_in)
                        self.assertTrue(
                            os.path.basename(version_config).startswith(
                                "omni-catalog-cursor-"))
                        self.assertTrue(version[0]["cursor_config_is_dir"], version_config)
                    self.assertEqual(version_env, expected_env)
                    self.assertIs(version[0]["kwargs"]["stderr"], subprocess.STDOUT)
                    loaded, info = catalog_v2.load(provider, "default")
                    self.assertIsNotNone(loaded, info)
                    self.assertEqual(loaded["discovery"], {
                        "posture": "seatbelt-fs", "platform": "Darwin"})

        def unfenced(argv, **kwargs):
            return list(argv), kwargs.get("env"), {"fence": "none"}

        for provider in _PROVIDERS:
            with self.subTest(provider=provider, phase="unfenced-refused"):
                calls = []

                def popen(*_args, **_kwargs):
                    calls.append(_args)
                    raise AssertionError("child started")

                with mock.patch("platform.system", return_value="Darwin"), \
                        _prepare_patch(unfenced), \
                        mock.patch.object(model_catalog.subprocess, "Popen",
                                          side_effect=popen):
                    result = model_catalog.refresh(provider, "default", 5)
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["providers"][provider]["reason"],
                                 "discovery_unavailable")
                self.assertEqual(calls, [])

    def test_refused_version_query_saves_listing_without_cli_version(self):
        """Guards: a listing that already passed a valid posture is saved when
        the later --version query is refused, and that child never starts.
        Off macOS a version argv outside the table is refused before any child.
        """
        def seatbelt_for_discovery(argv, **kwargs):
            parts = list(argv)
            env = kwargs["env"]
            if parts[-1:] == ["--version"]:
                return parts, env, {"fence": "none"}
            return (_HEAD_SEATBELT_WRAP + parts, env,
                    {"fence": "seatbelt-fs",
                     "fence_detail": {"read_wall": True,
                                      "workdir_writable": False,
                                      "vendor_state_writable": False}})

        for provider in _PROVIDERS:
            with self.subTest(provider=provider, phase="darwin-version-refused"):
                prepared = []

                def selective(argv, **kwargs):
                    prepared.append(list(argv))
                    return seatbelt_for_discovery(argv, **kwargs)

                calls, popen_patch = self._patch_spawn(provider)
                with mock.patch("platform.system", return_value="Darwin"), \
                        _prepare_patch(selective), \
                        popen_patch:
                    result = model_catalog.refresh(provider, "default", 30)
                self.assertTrue(result["ok"], result)
                table = _table_argv(provider, self._binary(provider))
                self.assertIn(table, prepared)
                self.assertIn([table[0], "--version"], prepared)
                self.assertEqual(
                    [call["argv"] for call in calls],
                    [_HEAD_SEATBELT_WRAP + table])
                loaded, info = catalog_v2.load(provider, "default")
                self.assertIsNotNone(loaded, info)
                self.assertNotIn("cli_version", loaded)
                self.assertEqual(loaded["models"][0]["id"], _MODEL_IDS[provider])
                self.assertEqual(loaded["discovery"], {
                    "posture": "seatbelt-fs", "platform": "Darwin"})

        for provider in _PROVIDERS:
            with self.subTest(provider=provider, phase="linux-version-argv"):
                spawned = []

                def popen(argv, **kwargs):
                    spawned.append(list(argv))
                    return _FakeProc(b"9.9.9\n")

                binary = self._binary(provider)
                self.assertTrue(binary)
                with mock.patch("platform.system", return_value="Linux"), \
                        mock.patch.object(model_catalog.subprocess, "Popen",
                                          side_effect=popen):
                    version = model_catalog._best_effort_cli_version(
                        provider, "default", 5)
                self.assertIsNone(version)
                self.assertEqual(spawned, [])

    def _strict_grok(self):
        return catalog_record.parse_grok_output(
            "Available models:\n- posture-grok\n")

    def test_strict_record_without_discovery_is_valid(self):
        """Guards: a record that omits discovery stays valid."""
        record = self._strict_grok()
        self.assertNotIn("discovery", record)
        accepted, warnings = catalog_record.validate_strict("grok", record)
        self.assertEqual(warnings, [])
        self.assertNotIn("discovery", accepted)
        self.assertEqual(accepted["provider"], "grok")

    def test_strict_record_refuses_bad_posture(self):
        """Guards: a discovery.posture outside the two allowed values is refused."""
        record = self._strict_grok()
        record["discovery"] = {"posture": "none", "platform": "Linux"}
        with self.assertRaises(catalog_record.StrictRecordError) as caught:
            catalog_record.validate_strict("grok", record)
        self.assertEqual(caught.exception.reason, "catalog_unreadable")

    def test_strict_record_refuses_extra_discovery_key(self):
        """Guards: discovery with a key other than posture and platform is refused."""
        record = self._strict_grok()
        record["discovery"] = {
            "posture": "metadata-unfenced", "platform": "Linux", "fence": "none"}
        with self.assertRaises(catalog_record.StrictRecordError) as caught:
            catalog_record.validate_strict("grok", record)
        self.assertEqual(caught.exception.reason, "catalog_unreadable")

    def test_non_darwin_codex_endpoint_refresh_reloads_under_same_fingerprint(self):
        """Guards: an endpoint the builder reads does not move the v2 file.
        catalog_v2.load finds the refresh record under today's fingerprint.
        """
        endpoint = "https://codex-posture.example/v1"
        fixed_home = "/omni-d2f-codex-posture-home"
        catalog_root = Path(self.state) / "model-catalogs"
        catalog_root.mkdir(mode=0o700)
        key_path = catalog_root / ".identity-key"
        key_path.write_bytes(b"\x11" * 32)
        os.chmod(key_path, 0o600)
        from providers import kernel_sandbox
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": endpoint}), \
                mock.patch.object(kernel_sandbox, "protected_home",
                                  lambda: fixed_home):
            before = model_catalog._scope_fingerprint("codex", "default", create=True)
            identity, _child = model_catalog._profile_identity_inputs("codex", "default")
            calls, popen_patch = self._patch_spawn("codex")
            with mock.patch("platform.system", return_value="Linux"), popen_patch:
                result = model_catalog.refresh("codex", "default", 30)
            after = model_catalog._scope_fingerprint("codex", "default")
            path, _lock = catalog_v2._paths("codex", "default")
            loaded, info = catalog_v2.load("codex", "default")
        self.assertTrue(result["ok"], result)
        self.assertEqual(before, after)
        self.assertEqual(path.name, "codex-%s.json" % before)
        self.assertEqual(
            path, Path(self.state) / "model-catalogs-v2" / _HEAD_CODEX_RECORD)
        body = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("scope_fingerprint", body)
        self.assertEqual(identity["profile"], "default")
        self.assertIsNotNone(loaded, info)
        self.assertEqual(loaded["models"][0]["id"], "posture-codex")
        self.assertEqual(loaded["discovery"]["posture"], "metadata-unfenced")
