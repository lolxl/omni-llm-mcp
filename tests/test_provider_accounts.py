#!/usr/bin/env python3
# no-opencc
"""Three-account contracts through public consumers; temporary fake vendors only."""
import asyncio
import contextlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from providers import agent_lane, health, jobs, job_runner, provider_accounts

NATIVE = {"codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR", "grok": "GROK_HOME"}
AUTH = ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN", "XAI_API_KEY", "GROK_API_KEY", "GROK_API_BASE")


GROK_NATIVE_AUTH_ENDPOINTS = (
    "GROK_CLI_CHAT_PROXY_BASE_URL", "GROK_WS_URL", "GROK_AUTH_PROVIDER_COMMAND",
    "GROK_MODELS_BASE_URL", "GROK_MODELS_LIST_URL", "GROK_XAI_API_BASE_URL",
    "GROK_OIDC_ISSUER", "GROK_OIDC_CLIENT_ID", "GROK_OIDC_AUDIENCE", "GROK_OIDC_SCOPES",
    "GROK_OAUTH2_ISSUER", "GROK_OAUTH2_CLIENT_ID", "GROK_OAUTH2_PRINCIPAL_ID",
    "GROK_OAUTH2_PRINCIPAL_TYPE", "GROK_OAUTH2_REFERRER", "GROK_OAUTH2_SCOPES",
    "GROK_WS_ORIGIN")
ALL_AUTH = AUTH + GROK_NATIVE_AUTH_ENDPOINTS
GROK_RETAINED_CONTROLS = (
    "GROK_CONFIG", "GROK_AUTH_PROVIDER_LABEL", "GROK_AUTH_TOKEN_TTL",
    "GROK_AUTH_EXPIRED", "GROK_AUTH_EARLY_INVALIDATION_SECS")

# Independent local producer documentation snapshot (Grok 1.0.41). Rows are
# bound to the declared native auth/endpoint fields, not the implementation's
# deny tuple. The standalone proxy override comes from its environment table.
GROK_PRODUCER_AUTH_ENDPOINT_ROWS = """
GROK_AUTH_PROVIDER_COMMAND|string|auth.auth_provider_command
GROK_WS_ORIGIN|string|auth.grok_ws_origin
GROK_WS_URL|string|auth.grok_ws_url
GROK_OAUTH2_CLIENT_ID|string|auth.oauth2.client_id
GROK_OAUTH2_ISSUER|string|auth.oauth2.issuer
GROK_OAUTH2_PRINCIPAL_ID|string|auth.oauth2.principal_id
GROK_OAUTH2_PRINCIPAL_TYPE|string|auth.oauth2.principal_type
GROK_OAUTH2_REFERRER|string|auth.oauth2.referrer
GROK_OAUTH2_SCOPES|string[]|auth.oauth2.scopes
GROK_OIDC_AUDIENCE|string|auth.oidc.audience
GROK_OIDC_CLIENT_ID|string|auth.oidc.client_id
GROK_OIDC_ISSUER|string|auth.oidc.issuer
GROK_OIDC_SCOPES|string[]|auth.oidc.scopes
GROK_MODELS_BASE_URL|string|endpoints.models_base_url
GROK_MODELS_LIST_URL|string|endpoints.models_list_url
GROK_XAI_API_BASE_URL|string|endpoints.xai_api_base_url
GROK_CLI_CHAT_PROXY_BASE_URL|string|Override API proxy base URL
"""


def failure_stream(provider, tool=False, auth=False):
    """Documented vendor streams, including actual tool activity when requested."""
    if provider == "codex":
        events = [{"type": "thread.started", "thread_id": "synthetic-session"}]
        if tool:
            events.append({"type": "item.started", "item": {"type": "command_execution"}})
        events.append({"type": "turn.failed", "error": {
            "code": "authentication_failed" if auth else "usage_limit_reached",
            "message": "401 Unauthorized" if auth else "You've hit your usage limit"}})
    elif provider == "claude":
        events = [{"type": "system", "subtype": "init", "session_id": "synthetic-session"}]
        if tool:
            events.append({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "synthetic-tool", "name": "Bash", "input": {}}]}})
        events.append({"type": "result", "subtype": "error_during_execution",
                       "is_error": True, "result": "Not logged in" if auth else "usage limit reached",
                       "api_error_status": 401 if auth else 429})
    else:
        events = [{"type": "system", "subtype": "init", "session_id": "synthetic-session"}]
        if tool:
            events.append({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "synthetic-tool", "name": "shell", "input": {}}]}})
        events.append({"type": "error", "error": {
            "code": "authentication_failed" if auth else "usage_limit_reached",
            "message": "401 Unauthorized" if auth else "usage limit reached"}})
    return "\n".join(json.dumps(event) for event in events)


class ProviderContract:
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-provider-contract-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.homes = {}
        for label in ("a", "b", "c"):
            self.homes[label] = str(self.root / label)
            Path(self.homes[label]).mkdir()
        self.workdir = str(self.root / "work")
        Path(self.workdir).mkdir()
        self.model = self.provider + "/synthetic-model"
        self.native = NATIVE[self.provider]
        self.ring = "OMNI_LLM_PROFILES_" + self.provider.upper()
        env = {"PATH": os.environ.get("PATH", ""), "TMPDIR": "/private/tmp",
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_STATE_DIR": str(self.root / "state"),
               "OMNI_LLM_NOTIFY_QUIET": "1", self.ring: "a,b,c"}
        for native in NATIVE.values():
            # Registered homes of different providers must be disjoint
            # directories; only the provider under test uses self.homes.
            for label, home in self.homes.items():
                if native != self.native:
                    home = str(self.root / (native.lower() + "-" + label))
                    Path(home).mkdir()
                env[native + "_" + label.upper()] = home
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        self.calls = []

    def seed_auth_overrides(self):
        # The native auth provider accepts an executable command path. It must
        # remain uninvoked: fake vendors only record the environment they see.
        helper = self.root / "synthetic-auth-helper"
        marker = self.root / "auth-helper-invoked"
        helper.write_text("#!" + sys.executable + "\n" +
            "import json\nfrom pathlib import Path\n" +
            "Path(" + repr(str(marker)) + ").write_text('invoked')\n" +
            "print(json.dumps({'access_token': 'synthetic-other-account', 'expires_in': 3600}))\n")
        helper.chmod(0o700)
        values = {key: "synthetic-override" for key in AUTH}
        values.update({
            "GROK_CLI_CHAT_PROXY_BASE_URL": "https://synthetic.invalid/chat/v1",
            "GROK_WS_URL": "wss://synthetic.invalid/ws",
            "GROK_AUTH_PROVIDER_COMMAND": str(helper),
            "GROK_MODELS_BASE_URL": "https://synthetic.invalid/models",
            "GROK_MODELS_LIST_URL": "https://synthetic.invalid/models/list",
            "GROK_XAI_API_BASE_URL": "https://synthetic.invalid/xai/v1",
            "GROK_OIDC_ISSUER": "https://oidc.synthetic.invalid/tenant",
            "GROK_OIDC_CLIENT_ID": "synthetic-oidc-client",
            "GROK_OIDC_AUDIENCE": "https://synthetic.invalid/oidc-api",
            # JSON array serialization is a hermetic fixture choice; native
            # environment encoding is not a verified authentication contract.
            "GROK_OIDC_SCOPES": json.dumps(["openid", "profile"]),
            "GROK_OAUTH2_ISSUER": "https://oauth.synthetic.invalid/tenant",
            "GROK_OAUTH2_CLIENT_ID": "synthetic-oauth-client",
            "GROK_OAUTH2_PRINCIPAL_ID": "00000000-0000-4000-8000-000000000017",
            "GROK_OAUTH2_PRINCIPAL_TYPE": "Team",
            "GROK_OAUTH2_REFERRER": "https://synthetic.invalid/oauth-client",
            "GROK_OAUTH2_SCOPES": json.dumps(["openid", "profile"]),
            "GROK_WS_ORIGIN": "https://synthetic.invalid",
            "GROK_AUTH_PROVIDER_LABEL": "Synthetic fixture login",
            "GROK_AUTH_TOKEN_TTL": "3600",
            "GROK_AUTH_EXPIRED": "1",
            "GROK_AUTH_EARLY_INVALIDATION_SECS": "300",
            "GROK_CONFIG": json.dumps({"models": {"default_reasoning_effort": "low"}}),
        })
        os.environ.update(values)
        return values

    def assert_overlay_and_unused_helper(self, rows, values):
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["grok_config"], values["GROK_CONFIG"])
            self.assertEqual(row["retained_controls"], {key: values[key] for key in GROK_RETAINED_CONTROLS})
        self.assertFalse((self.root / "auth-helper-invoked").exists(), "fake vendor invoked auth helper")

    def lane_patches(self, runner=None, binary=None):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(agent_lane, "resolve_agent_binary", return_value=binary or "/synthetic/vendor"))
        stack.enter_context(mock.patch.object(agent_lane.shutil, "which", return_value=binary or "/synthetic/vendor"))
        stack.enter_context(mock.patch.object(agent_lane, "_workdir_fence", return_value=None))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_snapshot", return_value={"state": "ok", "entries": {}, "head": None}))
        stack.enter_context(mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)))
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=
            lambda provider, argv, env, *args, **kwargs: (argv, env, "vendor", {}, None)))
        stack.enter_context(mock.patch.object(agent_lane.sandbox_canary, "gate", return_value=None))
        if runner is not None:
            stack.enter_context(mock.patch.object(agent_lane, "run_cli", side_effect=runner))
        return stack

    def success(self, argv, kwargs):
        if "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_text("synthetic success")
        if self.provider == "codex":
            out = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "synthetic success"}})
        else:
            out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "synthetic success"})
        return {"returncode": 0, "stdout": out, "stderr": ""}

    async def recording_success(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return self.success(argv, kwargs)

    def invoke(self, **kwargs):
        return asyncio.run(agent_lane.run_agent("synthetic task", self.model,
                         self.workdir, timeout_s=5, **kwargs))

    def tried_row(self, label, reason):
        # Only Claude reports the account kind; Codex and Grok rows are unchanged.
        row = {"account_profile": label, "reason": reason}
        if self.provider == "claude":
            row["account_kind"] = "subscription"
        return row

    def selector(self, **kwargs):
        if self.provider == "codex":
            from providers.codex_accounts import AccountSelection
        else:
            from providers.provider_accounts import AccountSelection
            kwargs["provider"] = self.provider
        return AccountSelection(model=self.model, **kwargs)

    def test_ordered_three_candidates_visited_once_with_canonical_dedup(self):
        alias = self.root / "alias-a"
        alias.symlink_to(self.homes["a"], target_is_directory=True)
        os.environ[self.native + "_B"] = str(alias)
        selection = self.selector()
        chosen = [selection.next(), selection.next()]
        self.assertEqual([account["account_profile"] for account in chosen], ["a", "c"])
        self.assertIsNone(selection.next())
        self.assertIsNone(selection.next())
        self.assertEqual(selection.tried, [self.tried_row("b", "duplicate_account_home")])

    def test_all_three_missing_are_bounded_unavailable_attempts(self):
        for label in self.homes:
            os.environ[self.native + "_" + label.upper()] = str(self.root / ("missing-" + label))
        selection = self.selector()
        self.assertIsNone(selection.next())
        self.assertIsNone(selection.next())
        result = selection.exhausted()
        self.assertEqual(result["reason"], "account_exhausted")
        self.assertEqual(result["tried"], [self.tried_row(label, "account_unavailable") for label in ("a", "b", "c")])

    def test_one_deadline_expiration_never_starts_backend(self):
        from providers import provider_accounts, codex_accounts
        selection = self.selector(timeout_s=0.1)
        selection.deadline = 0
        async def forbidden(*args):
            self.fail("expired selection started a child")
        module = codex_accounts if self.provider == "codex" else provider_accounts
        result = asyncio.run(module.run_selected(selection, forbidden))
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(selection.tried, [])
        self.assertIsNone(selection.next())

    def managed_failure(self, inner):
        from providers import provider_accounts
        selection = self.selector()
        async def attempt(account, remaining, managed):
            result = {"ok": False, "reason": "runtime_error"}
            if inner is not None:
                result["detail"] = inner(account)
            return result
        return asyncio.run(provider_accounts.run_selected(selection, attempt))

    def test_managed_failure_keeps_redacted_vendor_detail(self):
        token = "sk-ant-" + "A1b2C3d4" * 3
        xai = "xai-" + "Z9y8X7w6" * 3
        def inner(account):
            home = account[next(k for k in account if k.endswith("_home"))]
            return ("400 This API key is not scoped to a workspace at %s (%s) key=%s %s sibling=%s %s"
                    % (home, os.path.realpath(home), token, xai, self.homes["c"], "x" * 400))
        result = self.managed_failure(inner)
        detail = result["detail"]
        self.assertTrue(detail.startswith(self.provider + " account failed: runtime_error: "), detail)
        self.assertIn("not scoped to a workspace", detail)
        self.assertIn("<account home>", detail)
        for secret in (token, xai, self.root.as_posix(), "A1b2C3d4"):
            self.assertNotIn(secret, detail)
        self.assertLessEqual(len(detail), 300)

    def test_managed_failure_without_inner_detail_stays_generic(self):
        for inner in (None, lambda account: "", lambda account: "   "):
            result = self.managed_failure(inner)
            self.assertEqual(result["detail"], self.provider + " account failed: runtime_error")

    def test_consult_real_child_receives_selected_native_env(self):
        from providers import dispatch, claude, codex, grok
        binary, capture = self.fake_vendor()
        os.environ[self.native + "_A"] = str(self.root / "missing")
        health.flag(self.model, "b")
        seeds = self.seed_auth_overrides()
        module = {"claude": claude, "codex": codex, "grok": grok}[self.provider]
        def prepared(argv, **kwargs):
            self.assertEqual(kwargs.get("config_dir"), self.homes["c"])
            return argv, kwargs["env"], {}
        with mock.patch.object(module.shutil, "which", return_value=str(binary)), mock.patch.object(module, "prepare_host_fence", side_effect=prepared):
            with contextlib.ExitStack() as stack:
                if self.provider == "claude":
                    stack.enter_context(mock.patch.object(module, "resolve_agent_binary", return_value=str(binary)))
                result = asyncio.run(dispatch.route_to_provider("synthetic task", self.model, 5, [], "default"))
        self.assertTrue(capture.exists(), result)
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["home"], self.homes["c"])
        self.assertEqual(rows[0]["auth"], [])
        self.assertEqual(rows[0]["siblings"], [])
        self.assert_overlay_and_unused_helper(rows, seeds)
        self.assertFalse(result["ok"])
        self.assertEqual(result["account_profile"], "c")
        if self.provider != "codex":
            self.assertIs(result.get("quota_verified"), False)

    def test_actual_fence_policy_admits_selected_home_and_excludes_other_accounts(self):
        from providers import kernel_sandbox
        binary, capture = self.fake_vendor()
        protected = self.root / "protected"
        protected.mkdir()
        policies = []
        def fenced(provider, argv, env, workdir, read_only, requested, **kwargs):
            self.assertEqual(kwargs.get("config_dir"), self.homes["a"])
            policy = kernel_sandbox.agent_fence_policy(provider, workdir=workdir,
                read_only=read_only, config_dir=kwargs.get("config_dir"))
            self.assertIn("wrap_kw", policy, policy)
            entries = list(policy["wrap_kw"]["extra_read_paths"])
            paths = [getattr(entry, "path", entry[1] if isinstance(entry, tuple) else entry) for entry in entries]
            self.assertIn(self.homes["a"], paths)
            self.assertNotIn(self.homes["b"], paths)
            self.assertNotIn(self.homes["c"], paths)
            policies.append(policy)
            return argv, env, "vendor", {}, None
        with self.lane_patches(binary=str(binary)), mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=fenced), mock.patch.object(kernel_sandbox, "protected_home", return_value=str(protected)), mock.patch.object(kernel_sandbox, "resolve_agent_binary", return_value=str(binary)):
            self.invoke()
        self.assertEqual(len(policies), 1)
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(rows[0]["home"], self.homes["a"])

    def test_quota_adapter_is_provider_specific_and_honest(self):
        from providers import codex_quota
        os.environ["OMNI_LLM_QUOTA_PROBE"] = "1"
        available = {"used_percent": 10, "reached": False, "primary_reached": False, "window_mins": 300, "resets_at": None}
        exhausted = dict(available, used_percent=100)
        second_exhausted = dict(available, secondary={"used_percent": 100, "reached": False})
        with mock.patch.object(codex_quota, "read_rate_limits", side_effect=[(exhausted, None), (second_exhausted, None), (available, None)]) as oracle, self.lane_patches(self.recording_success):
            result = self.invoke()
        self.assertTrue(result["ok"], result)
        if self.provider == "codex":
            self.assertEqual(oracle.call_count, 3)
            self.assertEqual(result["account_profile"], "c")
            self.assertEqual([entry["reason"] for entry in result["tried"]], ["quota_exhausted", "quota_exhausted"])
        else:
            oracle.assert_not_called()
            self.assertIs(result.get("quota_verified"), False)
            self.assertNotIn("used_percent", result)
            self.assertEqual(result["account_profile"], "a")

    def test_managed_default_resume_refuses_ambient_home_remap(self):
        os.environ[self.native] = self.homes["a"]
        os.environ[self.ring] = "default,b,c"
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual(data["account_profile"], "default")
        self.assertEqual(data[self.home_key()], self.homes["a"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        os.environ[self.native] = self.homes["c"]
        with self.job_patches():
            refused = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(refused.get("refused", {}).get("reason"), "account_home_drift")

    def test_dispatch_chain_cannot_replay_after_tool_activity(self):
        from providers import dispatch
        calls = []
        async def failed(*args, **kwargs):
            calls.append(args[1])
            return {"ok": False, "reason": "rate_limit", "account_profile": "a",
                    "account_retry_safe": False, "tool_activity": True}
        with mock.patch.object(dispatch, "dispatch_single", side_effect=failed), mock.patch.object(dispatch, "_classify_seat", return_value=("healthy", None)), mock.patch.object(dispatch, "_health_maintain"):
            result = asyncio.run(dispatch.dispatch_llm_request("synthetic task", self.model + ",cursor/synthetic-model", 5, [], "default"))
        self.assertEqual(calls, [self.model], "dispatch chained another vendor after tool activity")
        self.assertEqual(result["reason"], "rate_limit")

    def test_three_accounts_skip_missing_and_cooling_select_third(self):
        os.environ[self.native + "_A"] = str(self.root / "missing")
        health.flag(self.model, "b")
        with self.lane_patches(self.recording_success):
            result = self.invoke()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]["env"].get(self.native), self.homes["c"])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["account_profile"], "c")

    def test_all_three_cooling_returns_typed_exhaustion_without_child(self):
        for label in self.homes:
            health.flag(self.model, label)
        with self.lane_patches(self.recording_success):
            result = self.invoke()
        self.assertEqual(self.calls, [], "an exhausted account pool spawned a child")
        self.assertEqual(result["reason"], "account_exhausted")
        self.assertEqual([entry["account_profile"] for entry in result["tried"]], ["a", "b", "c"])

    def test_explicit_profile_stays_pinned_despite_ring_and_cooling(self):
        health.flag(self.model, "b")
        with self.lane_patches(self.recording_success):
            result = self.invoke(account_profile="b")
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]["env"][self.native], self.homes["b"])

    def test_post_tool_quota_never_reexecutes_and_fresh_job_switches(self):
        marker = self.root / "side-effects.txt"
        async def runner(argv, **kwargs):
            self.calls.append((list(argv), dict(kwargs)))
            home = kwargs["env"].get(self.native)
            if home == self.homes["a"]:
                with marker.open("a") as handle:
                    handle.write("tool ran\n")
                return {"returncode": 1, "stdout": failure_stream(self.provider, tool=True), "stderr": "usage limit reached"}
            return self.success(argv, kwargs)
        with self.lane_patches(runner):
            stopped = self.invoke()
            self.assertEqual(len(self.calls), 1, "quota after effects reran the current task")
            self.assertEqual(marker.read_text(), "tool ran\n")
            self.assertFalse(stopped["ok"])
            self.assertEqual(stopped["reason"], "rate_limit")
            self.calls.clear()
            fresh = self.invoke()
        self.assertTrue(fresh["ok"], fresh)
        self.assertEqual([row[1]["env"].get(self.native) for row in self.calls], [self.homes["b"]])
        self.assertEqual(marker.read_text(), "tool ran\n")

    def test_post_tool_auth_never_reexecutes(self):
        async def runner(argv, **kwargs):
            self.calls.append((list(argv), dict(kwargs)))
            if kwargs["env"].get(self.native) == self.homes["a"]:
                return {"returncode": 1, "stdout": failure_stream(self.provider, tool=True, auth=True), "stderr": "401 Unauthorized"}
            return self.success(argv, kwargs)
        with self.lane_patches(runner):
            result = self.invoke()
            self.assertEqual(len(self.calls), 1, "auth failure after a tool replayed the task")
            self.assertEqual(result["reason"], "auth_error")
            self.calls.clear()
            fresh = self.invoke()
        self.assertTrue(fresh["ok"], fresh)
        self.assertEqual([row[1]["env"].get(self.native) for row in self.calls], [self.homes["b"]])

    def test_unknown_failure_and_timeout_stop_after_one_account(self):
        for failure in ({"returncode": 1, "stdout": "unexpected vendor output", "stderr": "unknown failure"},
                        {"returncode": None, "stdout": "", "stderr": "", "timed_out": True}):
            with self.subTest(failure=failure):
                self.calls.clear()
                async def runner(argv, **kwargs):
                    self.calls.append((argv, kwargs))
                    return failure
                with self.lane_patches(runner):
                    result = self.invoke()
                self.assertFalse(result["ok"])
                self.assertEqual(len(self.calls), 1)

    def test_managed_native_env_scrubs_account_siblings_and_auth_overrides(self):
        seeds = self.seed_auth_overrides()
        before = dict(os.environ)
        with self.lane_patches(self.recording_success):
            result = self.invoke()
        self.assertTrue(result["ok"], result)
        env = self.calls[0][1]["env"]
        self.assertEqual(env.get(self.native), self.homes["a"])
        self.assertFalse(set(AUTH).intersection(env), "managed subscription inherited auth overrides")
        self.assertFalse(set(GROK_NATIVE_AUTH_ENDPOINTS).intersection(env), "managed child inherited native auth/endpoint overrides")
        self.assertEqual(env.get("GROK_CONFIG"), seeds["GROK_CONFIG"])
        for native in NATIVE.values():
            self.assertFalse(any(key.startswith(native + "_") for key in env))
            if native != self.native:
                self.assertNotIn(native, env)
        self.assertEqual(dict(os.environ), before)

    def test_actual_sync_child_scrubs_native_auth_and_keeps_ui_overlay(self):
        binary, capture = self.fake_vendor()
        seeds = self.seed_auth_overrides()
        parent = dict(os.environ)
        with self.lane_patches(binary=str(binary)):
            result = self.invoke()
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auth"], [], "forbidden auth/endpoint key reached actual sync child")
        self.assertEqual(rows[0]["native_auth_values"], {})
        self.assertEqual(rows[0]["home"], self.homes["a"])
        self.assert_overlay_and_unused_helper(rows, seeds)
        self.assertEqual(dict(os.environ), parent)
        self.assertEqual(result["reason"], "rate_limit")

    def test_unmanaged_real_consumers_preserve_native_auth_and_ui_overlay(self):
        from providers import dispatch, claude, codex, grok
        for lane in ("consult", "sync", "runner", "resume"):
            with self.subTest(lane=lane):
                health.clear_all()
                os.environ.pop(self.ring, None)
                os.environ[self.native] = self.homes["a"]
                binary, capture = self.fake_vendor()
                capture.write_text("")
                seeds = self.seed_auth_overrides()
                parent = dict(os.environ)
                if lane == "consult":
                    module = {"claude": claude, "codex": codex, "grok": grok}[self.provider]
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(mock.patch.object(module.shutil, "which", return_value=str(binary)))
                        stack.enter_context(mock.patch.object(module, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})))
                        if self.provider == "claude":
                            stack.enter_context(mock.patch.object(module, "resolve_agent_binary", return_value=str(binary)))
                        result = asyncio.run(dispatch.route_to_provider("synthetic task", self.model, 5, [], "default"))
                elif lane == "sync":
                    with self.lane_patches(binary=str(binary)):
                        result = self.invoke()
                else:
                    source = self.start_job()
                    data = jobs._read_job(source["job_dir"])
                    if lane == "resume":
                        data.update(state="failed", session_id="synthetic-parent-session")
                        jobs._write_job(source["job_dir"], data)
                        os.environ[self.ring] = "a,b,c"
                        with self.job_patches():
                            source = jobs.resume(source["job_dir"], "synthetic continuation", allow_drift=True)
                        self.assertNotIn("refused", source)
                        data = jobs._read_job(source["job_dir"])
                        parent = dict(os.environ)
                    with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
                        result = asyncio.run(job_runner._execute(source["job_dir"], data))
                rows = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(set(rows[0]["auth"]), set(ALL_AUTH), "unmanaged child lost inherited auth compatibility")
                self.assertEqual(rows[0]["native_auth_values"], {key: seeds[key] for key in GROK_NATIVE_AUTH_ENDPOINTS})
                self.assert_overlay_and_unused_helper(rows, seeds)
                self.assertEqual(dict(os.environ), parent)
                self.assertEqual(result["reason"], "rate_limit")

    def test_unmanaged_default_preserves_auth_environment(self):
        os.environ.pop(self.ring)
        os.environ[self.native] = self.homes["a"]
        seeds = self.seed_auth_overrides()
        with self.lane_patches(self.recording_success):
            result = self.invoke()
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(self.calls), 1)
        env = self.calls[0][1]["env"]
        for key in AUTH:
            self.assertEqual(env.get(key), "synthetic-override", key)

    def job_patches(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(jobs, "_preflight", return_value=(None, self.workdir, {"provider": self.provider, "model": "synthetic-model"})))
        stack.enter_context(mock.patch.object(jobs, "_drift_pin", return_value={"repo_root": self.workdir, "branch": "synthetic-work", "head": "synthetic-head"}))
        def spawn(argv, **kwargs):
            self.spawned_job = jobs._read_job(argv[-1])
            return mock.Mock(pid=99999999)
        stack.enter_context(mock.patch.object(jobs.subprocess, "Popen", side_effect=spawn))
        stack.enter_context(mock.patch.object(jobs, "_record_supervisor", side_effect=lambda data, proc: data))
        stack.enter_context(mock.patch.object(jobs, "_wait_for_supervisor_lock"))
        stack.enter_context(mock.patch.object(jobs.threading, "Thread"))
        stack.enter_context(mock.patch.object(jobs, "_workdir_has_other_live", return_value=False))
        return stack

    def start_job(self, **kwargs):
        with self.job_patches():
            return jobs.start("synthetic task", self.model, self.workdir, timeout_s=5, **kwargs)

    def test_detached_job_selects_third_before_supervisor_spawn(self):
        os.environ[self.native + "_A"] = str(self.root / "missing")
        health.flag(self.model, "b")
        result = self.start_job()
        self.assertNotIn("refused", result)
        self.assertEqual(self.spawned_job["account_profile"], "c")
        self.assertEqual(self.spawned_job.get(self.home_key()), self.homes["c"])

    def home_key(self):
        return "codex_home" if self.provider == "codex" else "account_home"

    def test_persisted_job_deadline_expires_before_real_child_spawn(self):
        binary, capture = self.fake_vendor()
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        key = "codex_deadline" if self.provider == "codex" else "account_deadline"
        data[key] = 0
        jobs._write_job(origin["job_dir"], data)
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        self.assertFalse(capture.exists(), "expired persisted job executed a backend")
        self.assertEqual(result["reason"], "timeout")

    def test_detached_exhaustion_never_spawns_supervisor(self):
        for label in self.homes:
            health.flag(self.model, label)
        result = self.start_job()
        self.assertFalse(hasattr(self, "spawned_job"), "exhausted ring detached a supervisor")
        self.assertEqual(result["refused"]["reason"], "account_exhausted")

    def test_resume_pins_home_after_ring_reorder_and_unset(self):
        started = self.start_job(account_profile="b")
        data = jobs._read_job(started["job_dir"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(started["job_dir"], data)
        for ring in ("c,a,b", None):
            with self.subTest(ring=ring):
                if ring is None:
                    os.environ.pop(self.ring, None)
                else:
                    os.environ[self.ring] = ring
                with self.job_patches():
                    resumed = jobs.resume(started["job_dir"], "synthetic continuation", allow_drift=True)
                self.assertNotIn("refused", resumed)
                self.assertEqual(self.spawned_job["account_profile"], "b")
                self.assertEqual(self.spawned_job.get(self.home_key()), self.homes["b"])
                self.assertEqual(self.spawned_job["resume_session"], "synthetic-parent-session")

    def test_resume_refuses_changed_or_missing_home_provenance(self):
        started = self.start_job(account_profile="b")
        data = jobs._read_job(started["job_dir"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(started["job_dir"], data)
        os.environ[self.native + "_B"] = self.homes["c"]
        with self.job_patches():
            changed = jobs.resume(started["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(changed.get("refused", {}).get("reason"), "account_home_drift")
        os.environ[self.native + "_B"] = self.homes["b"]
        data.pop(self.home_key(), None)
        jobs._write_job(started["job_dir"], data)
        with self.job_patches():
            missing = jobs.resume(started["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(missing.get("refused", {}).get("reason"), "account_home_drift")

    def fake_vendor(self):
        binary = self.root / "fake-vendor"
        capture = self.root / "vendor.jsonl"
        events = failure_stream(self.provider, tool=True)
        script = "#!" + sys.executable + "\n" + "import json, os, sys\n"
        if self.provider != "grok":
            script += "sys.stdin.read()\n"
        script += "native = " + repr(self.native) + "\n"
        script += "with open(os.environ['OMNI_SYNTHETIC_CAPTURE'], 'a') as h:\n"
        script += "    h.write(json.dumps({'home': os.environ.get(native), 'argv': sys.argv[1:], 'auth': [key for key in " + repr(ALL_AUTH) + " if key in os.environ], 'grok_config': os.environ.get('GROK_CONFIG'), 'native_auth_values': {key: os.environ[key] for key in " + repr(GROK_NATIVE_AUTH_ENDPOINTS) + " if key in os.environ}, 'retained_controls': {key: os.environ[key] for key in " + repr(GROK_RETAINED_CONTROLS) + " if key in os.environ}, 'siblings': [key for key in os.environ if key.startswith(('CODEX_HOME_', 'CLAUDE_CONFIG_DIR_', 'GROK_HOME_'))]}) + '\\n')\n"
        script += "print(" + repr(events) + ")\nsys.exit(1)\n"
        binary.write_text(script)
        binary.chmod(0o700)
        os.environ["OMNI_SYNTHETIC_CAPTURE"] = str(capture)
        return binary, capture

    def test_actual_resumed_child_scrubs_full_native_selector_inventory(self):
        binary, capture = self.fake_vendor()
        seeds = self.seed_auth_overrides()
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            original_result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        self.assertEqual(original_result["reason"], "rate_limit")
        original = jobs._read_job(origin["job_dir"])
        self.assertEqual(original["state"], "failed")
        self.assertTrue(original["session_id"])
        os.environ.pop(self.ring)
        parent = dict(os.environ)
        with self.job_patches():
            child = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertNotIn("refused", child)
        resumed_data = jobs._read_job(child["job_dir"])
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            resumed_result = asyncio.run(job_runner._execute(child["job_dir"], resumed_data))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["auth"], [], "forbidden native selector reached actual resumed child")
        self.assertEqual(rows[1]["native_auth_values"], {})
        self.assertEqual(rows[1]["home"], self.homes["a"])
        self.assert_overlay_and_unused_helper(rows, seeds)
        self.assertEqual(dict(os.environ), parent)
        self.assertEqual(resumed_result["reason"], "rate_limit")

    def test_launch_context_is_scrubbed_from_every_isolated_child(self):
        from providers import dispatch, claude, codex, grok
        for name in LAUNCH_CONTEXT:
            os.environ[name] = "synthetic-" + name
        os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "4096"
        seen = []

        def check(lane, env):
            seen.append(lane)
            self.assertFalse(set(LAUNCH_CONTEXT) & set(env), lane)
            if self.provider == "claude":
                self.assertEqual(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "4096", lane)

        with self.lane_patches(self.recording_success):
            self.invoke()
            check("managed sync agent", self.calls[-1][1]["env"])
            self.invoke(account_profile="b")
            check("ring-pinned sync agent", self.calls[-1][1]["env"])
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        with self.lane_patches(self.recording_success), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(origin["job_dir"], data))
        check("job", self.calls[-1][1]["env"])
        data = jobs._read_job(origin["job_dir"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        ring = os.environ.pop(self.ring)
        with self.job_patches():
            child = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertNotIn("refused", child)
        resumed = jobs._read_job(child["job_dir"])
        with self.lane_patches(self.recording_success), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(child["job_dir"], resumed))
        check("resumed job", self.calls[-1][1]["env"])
        os.environ[self.ring] = ring
        module = {"claude": claude, "codex": codex, "grok": grok}[self.provider]
        consult_envs = []

        async def fake_cli(argv, **kwargs):
            consult_envs.append(kwargs["env"])
            return {"returncode": 0, "stdout": "", "stderr": ""}
        for profile in ("default", "b"):
            with mock.patch.object(module.shutil, "which", return_value="/synthetic/vendor"), \
                    mock.patch.object(module, "run_cli", side_effect=fake_cli), \
                    mock.patch.object(module, "prepare_host_fence",
                                      side_effect=lambda argv, **kw: (argv, kw["env"], {})), \
                    contextlib.ExitStack() as stack:
                if self.provider == "claude":
                    stack.enter_context(mock.patch.object(
                        module, "resolve_agent_binary", return_value="/synthetic/vendor"))
                asyncio.run(dispatch.route_to_provider("synthetic task", self.model, 5, [], profile))
            self.assertTrue(consult_envs, profile)
            check("consult " + profile, consult_envs[-1])
        self.assertEqual(len(seen), 6)

    def test_real_child_and_job_runner_preserve_resume_auth_pin(self):
        binary, capture = self.fake_vendor()
        seeds = self.seed_auth_overrides()
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 1, "runner replayed the vendor after tool effects")
        self.assertEqual(rows[0]["home"], self.homes["a"])
        self.assertEqual(rows[0]["auth"], [])
        self.assertEqual(rows[0]["siblings"], [])
        self.assert_overlay_and_unused_helper(rows, seeds)
        self.assertEqual(result["reason"], "rate_limit")
        data = jobs._read_job(origin["job_dir"])
        # A failed session is resumable: retained native home is mandatory.
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        os.environ.pop(self.ring)
        with self.job_patches():
            child = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertNotIn("refused", child)
        child_data = jobs._read_job(child["job_dir"])
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            resumed = asyncio.run(job_runner._execute(child["job_dir"], child_data))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 2, "resume started more than one account")
        self.assertEqual([row["home"] for row in rows], [self.homes["a"]] * 2)
        self.assertEqual([row["auth"] for row in rows], [[], []])
        self.assert_overlay_and_unused_helper(rows, seeds)
        for key, value in seeds.items():
            self.assertEqual(os.environ.get(key), value, "managed child changed parent environment")
        self.assertEqual(resumed["reason"], "rate_limit")
        flag = "-r" if self.provider == "claude" else "--resume" if self.provider == "grok" else "resume"
        self.assertIn(flag, rows[1]["argv"])

    def test_explicit_profile_under_ring_is_auth_isolated_like_managed(self):
        """A pinned profile under a configured ring must not keep parent credentials."""
        from providers import dispatch, claude, codex, grok
        isolated_key = "codex_auth_isolated" if self.provider == "codex" else "account_auth_isolated"
        for ring in (True, False):
            for lane in ("consult", "sync", "runner", "resume"):
                with self.subTest(ring=ring, lane=lane):
                    health.clear_all()
                    if ring:
                        os.environ[self.ring] = "a,b"
                    else:
                        os.environ.pop(self.ring, None)
                    binary, capture = self.fake_vendor()
                    capture.write_text("")
                    seeds = self.seed_auth_overrides()
                    parent = dict(os.environ)
                    if lane == "consult":
                        module = {"claude": claude, "codex": codex, "grok": grok}[self.provider]
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(mock.patch.object(module.shutil, "which", return_value=str(binary)))
                            stack.enter_context(mock.patch.object(module, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})))
                            if self.provider == "claude":
                                stack.enter_context(mock.patch.object(module, "resolve_agent_binary", return_value=str(binary)))
                            result = asyncio.run(dispatch.route_to_provider("synthetic task", self.model, 5, [], "b"))
                    elif lane == "sync":
                        with self.lane_patches(binary=str(binary)):
                            result = self.invoke(account_profile="b")
                    else:
                        source = self.start_job(account_profile="b")
                        self.assertNotIn("refused", source)
                        data = jobs._read_job(source["job_dir"])
                        self.assertIs(data[isolated_key], ring)
                        self.assertEqual(data["account_profile"], "b")
                        if lane == "resume":
                            data.update(state="failed", session_id="synthetic-parent-session")
                            jobs._write_job(source["job_dir"], data)
                            with self.job_patches():
                                source = jobs.resume(source["job_dir"], "synthetic continuation", allow_drift=True)
                            self.assertNotIn("refused", source)
                            data = jobs._read_job(source["job_dir"])
                            self.assertIs(data[isolated_key], ring)
                            parent = dict(os.environ)
                        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
                            result = asyncio.run(job_runner._execute(source["job_dir"], data))
                    rows = [json.loads(line) for line in capture.read_text().splitlines()]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["home"], self.homes["b"])
                    if ring:
                        self.assertEqual(rows[0]["auth"], [], "pinned ring child kept parent credentials")
                        self.assertEqual(rows[0]["native_auth_values"], {})
                        self.assertEqual(rows[0]["siblings"], [])
                    else:
                        self.assertEqual(set(rows[0]["auth"]), set(ALL_AUTH), "legacy pin lost its environment")
                    self.assert_overlay_and_unused_helper(rows, seeds)
                    self.assertEqual(dict(os.environ), parent)
                    self.assertEqual(result["reason"], "rate_limit")

    def test_default_profile_resume_start_under_ring_is_auth_isolated(self):
        """jobs.start(default, resume_session) under a ring drops parent credentials."""
        isolated_key = "codex_auth_isolated" if self.provider == "codex" else "account_auth_isolated"
        binary, capture = self.fake_vendor()
        seeds = self.seed_auth_overrides()
        source = self.start_job(account_profile="default", resume_session="synthetic-parent-session")
        self.assertNotIn("refused", source)
        data = jobs._read_job(source["job_dir"])
        self.assertIs(data[isolated_key], True)
        with self.lane_patches(binary=str(binary)), mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(source["job_dir"], data))
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auth"], [], "resumed default child kept parent credentials")
        self.assertEqual(rows[0]["native_auth_values"], {})
        self.assert_overlay_and_unused_helper(rows, seeds)
        # No ring: unchanged legacy behavior.
        os.environ.pop(self.ring)
        legacy = self.start_job(account_profile="default", resume_session="synthetic-parent-session")
        self.assertNotIn("refused", legacy)
        self.assertIs(jobs._read_job(legacy["job_dir"])[isolated_key], False)

    def test_managed_failure_redacts_email_bearer_jwt_and_home_case(self):
        jwt = "eyJhbGciOi" + ".eyJzdWIiOiJ4" + ".c2lnbmF0dXJl"
        def inner(account):
            home = account[next(k for k in account if k.endswith("_home"))]
            return ("denied for user.name+tag@example.co.uk header Authorization: Bearer abc.DEF-123_x~y "
                    "jwt=%s at %s and %s" % (jwt, home.upper(), home.swapcase()))
        detail = self.managed_failure(inner)["detail"]
        self.assertIn("<email>", detail)
        self.assertIn("Bearer <redacted>", detail)
        self.assertIn("<account home>", detail)
        for secret in ("example.co.uk", "abc.DEF-123_x~y", "eyJhbGci", jwt, self.root.as_posix().upper(),
                       self.root.as_posix().swapcase()):
            self.assertNotIn(secret, detail)

    def test_managed_rate_limit_applies_vendor_reset_to_account_ledger(self):
        from datetime import datetime, timedelta, timezone
        from providers import provider_accounts
        if self.provider == "codex":
            self.skipTest("codex resets come from the quota RPC")
        for case in ("reset", "no_reset"):
            with self.subTest(case=case):
                health.clear_all()
                reset = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=2)
                selection = self.selector()
                async def attempt(account, remaining, managed):
                    detail = ("usage limit reached; resets at %s" % reset.isoformat()
                              if case == "reset" else "usage limit reached")
                    return {"ok": False, "reason": "rate_limit", "detail": detail}
                before = time.time()
                result = asyncio.run(provider_accounts.run_selected(selection, attempt))
                self.assertEqual(result["reason"], "rate_limit")
                entry = health._read_ledger_unlocked().get(health.pool_key(self.provider, "a"))
                self.assertIsNotNone(entry)
                if case == "reset":
                    self.assertEqual(entry["source"], "vendor_reset")
                    self.assertEqual(int(entry["until"]), int(reset.timestamp()))
                else:
                    self.assertNotEqual(entry["source"], "vendor_reset")
                    self.assertLess(entry["until"] - before, 2 * 3600)


for _provider in NATIVE:
    globals()["Test" + _provider.title() + "Accounts"] = type(
        "Test" + _provider.title() + "Accounts", (ProviderContract, unittest.TestCase), {"provider": _provider})


class TestGrokProducerInventory(unittest.TestCase):
    provider = "grok"
    setUp = ProviderContract.setUp
    seed_auth_overrides = ProviderContract.seed_auth_overrides

    def test_known_auth_endpoint_inventory_is_complete_and_fixture_shaped(self):
        from urllib.parse import urlsplit
        rows = [tuple(line.split("|")) for line in GROK_PRODUCER_AUTH_ENDPOINT_ROWS.strip().splitlines()]
        names = [row[0] for row in rows]
        self.assertEqual(len(rows), 17)
        self.assertEqual(len(set(names)), 17)
        self.assertEqual(set(names), set(GROK_NATIVE_AUTH_ENDPOINTS))
        self.assertEqual(len(GROK_NATIVE_AUTH_ENDPOINTS), 17)
        self.assertFalse(set(names).intersection(GROK_RETAINED_CONTROLS))
        values = self.seed_auth_overrides()
        for name, declared_type, _producer_field in rows:
            with self.subTest(name=name):
                self.assertIsInstance(values[name], str)
                self.assertTrue(values[name])
                if declared_type == "string[]":
                    # Shape only; the vendor's env-array encoding is unverified.
                    decoded = json.loads(values[name])
                    self.assertIsInstance(decoded, list)
                    self.assertTrue(decoded)
                    self.assertTrue(all(isinstance(value, str) and value for value in decoded))
                else:
                    self.assertEqual(declared_type, "string")
                if name.endswith(("_URL", "_ISSUER", "_AUDIENCE", "_REFERRER", "_ORIGIN")):
                    uri = urlsplit(values[name])
                    self.assertIn(uri.scheme, ("https", "wss"))
                    self.assertTrue(uri.hostname)
        self.assertTrue(os.access(values["GROK_AUTH_PROVIDER_COMMAND"], os.X_OK))
        self.assertFalse((self.root / "auth-helper-invoked").exists())


class ProcessStatusContract:
    setUp = ProviderContract.setUp
    lane_patches = ProviderContract.lane_patches
    invoke = ProviderContract.invoke

    def status_cases(self):
        # An explicit signal and signed exit status take precedence over text.
        # Only real nonnegative integers permit structured account failures.
        return [
            ({"returncode": -9}, "signal_exit"),
            ({"returncode": -1, "timed_out": False}, "signal_exit"),
            ({"returncode": 0, "signal": 9}, "signal_exit"),
            ({"returncode": None, "signal": 1}, "signal_exit"),
            ({"returncode": None}, "runtime_error"),
            ({"returncode": True}, "runtime_error"),
            ({"returncode": False}, "runtime_error"),
            ({"returncode": "1"}, "runtime_error"),
            ({"returncode": 0, "signal": True}, "runtime_error"),
            ({"returncode": 0, "signal": False}, "runtime_error"),
            ({"returncode": 0, "signal": 0}, "runtime_error"),
            ({"returncode": 1, "signal": "9"}, "runtime_error"),
            ({"returncode": -9, "timed_out": True}, "timeout"),
            ({"returncode": 0}, None),
            ({"returncode": 1}, None),
        ]

    def verify_status_matrix(self, lane):
        from providers import claude, grok
        module = claude if self.provider == "claude" else grok
        for status, expected in self.status_cases():
            for auth in (False, True):
                with self.subTest(lane=lane, status=status, auth=auth):
                    health.clear_all()
                    self.calls.clear()
                    envelope = {"returncode": 1, "stdout": failure_stream(self.provider, auth=auth), "stderr": ""}
                    envelope.update(status)
                    async def child(argv, **kwargs):
                        self.calls.append((list(argv), dict(kwargs)))
                        return envelope
                    if lane == "agent":
                        with self.lane_patches(child):
                            result = self.invoke(job_lane=True)
                    else:
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(mock.patch.object(module.shutil, "which", return_value="/synthetic/vendor"))
                            stack.enter_context(mock.patch.object(module, "run_cli", side_effect=child))
                            stack.enter_context(mock.patch.object(module, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})))
                            if self.provider == "claude":
                                stack.enter_context(mock.patch.object(module, "resolve_agent_binary", return_value="/synthetic/vendor"))
                            call = module.run_claude_subprocess if self.provider == "claude" else module.run_grok_subprocess
                            result = asyncio.run(call("synthetic task", "synthetic-model", 5))
                    self.assertEqual(len(self.calls), 1, "status failure replayed an account")
                    self.assertFalse(result["ok"], result)
                    self.assertEqual(result["reason"], expected or ("auth_error" if auth else "rate_limit"))
                    if expected is not None:
                        self.assertNotEqual(health.seat_kind(self.model, "a"), "cooling", "signal/unknown status falsely flagged account")
                    else:
                        self.assertEqual(health.seat_kind(self.model, "a"), "cooling", "valid process account failure did not flag account")

    def test_process_status_precedes_account_stream_in_agent(self):
        self.verify_status_matrix("agent")

    def test_process_status_precedes_account_stream_in_consult(self):
        self.verify_status_matrix("consult")


for _provider in ("claude", "grok"):
    globals()["Test" + _provider.title() + "ProcessStatus"] = type(
        "Test" + _provider.title() + "ProcessStatus", (ProcessStatusContract, unittest.TestCase), {"provider": _provider})


_MISSING = object()


class TerminalMetadataContract:
    setUp = ProviderContract.setUp
    lane_patches = ProviderContract.lane_patches
    invoke = ProviderContract.invoke

    def signals(self):
        return [({"code": "quota_exceeded"}, "rate_limit"),
                ({"code": "authentication_failed"}, "auth_error"),
                ({"status": 429}, "rate_limit"),
                ({"api_error_status": 401}, "auth_error"),
                ({"code": {"bad": "code"}}, None),
                ({"status": "429"}, None),
                ({"status": True}, None),
                ({"code": "unrelated"}, None)]

    def frame(self, kind, subtype, marker, metadata):
        obj = {"type": kind, "subtype": subtype, "result": "synthetic success", "error": metadata}
        if self.provider == "grok":
            obj["text"] = "synthetic success"
        if marker is not _MISSING:
            obj["is_error"] = marker
        return obj

    def test_terminal_metadata_discriminator_matrix(self):
        from providers.provider_accounts import structured_failure
        kinds = ("error", "turn.failed", "result", "assistant", "unknown", None, {"bad": "type"})
        subtypes = ("success", "error_during_execution", "unknown", None, {"bad": "subtype"})
        markers = (True, False, "true", 1, {}, None, _MISSING)
        for kind in kinds:
            for subtype in subtypes:
                for marker in markers:
                    for metadata, reason in self.signals():
                        with self.subTest(kind=kind, subtype=subtype, marker=repr(marker), metadata=metadata):
                            frame = self.frame(kind, subtype, marker, metadata)
                            actual = structured_failure(self.provider, json.dumps(frame))
                            valid_marker = marker is _MISSING or marker is True
                            known_terminal = isinstance(kind, str) and kind in ("error", "turn.failed", "result")
                            definite_result = kind != "result" or marker is True or (marker is _MISSING and subtype == "error_during_execution")
                            expected = reason if valid_marker and known_terminal and definite_result else None
                            if expected is None:
                                self.assertNotIn(actual, ("rate_limit", "auth_error", "account_unavailable"), "non-error metadata became definite account failure")
                            else:
                                self.assertEqual(actual, expected, "genuine terminal failure lost its structured signal")

    def consume(self, lane, frame, rc):
        from providers import claude, grok
        health.clear_all()
        self.calls.clear()
        module = claude if self.provider == "claude" else grok
        async def child(argv, **kwargs):
            self.calls.append((list(argv), dict(kwargs)))
            stdout = json.dumps(frame)
            if (lane == "agent" and rc == 0
                    and frame.get("type") == "result" and frame.get("subtype") == "success"
                    and frame.get("is_error") is False):
                # Both job streams release text from assistant message blocks.
                assistant = {"type": "assistant", "message": {
                    "role": "assistant", "content": [
                        {"type": "text", "text": frame["result"]}]}}
                stdout = json.dumps(assistant) + "\n" + stdout
            if (self.provider == "grok" and lane == "consult" and rc == 0
                    and frame.get("type") == "result" and frame.get("subtype") == "success"
                    and frame.get("is_error") is False):
                # Upstream consult releases only its native no-tool NDJSON
                # protocol. Successful error documentation is text data;
                # the hypothetical metadata object is not a vendor event.
                sid = argv[argv.index("--session-id") + 1]
                events = [{"type": "available_commands", "tools": []},
                          {"type": "text", "data": stdout},
                          {"type": "end", "stopReason": "end_turn", "sessionId": sid}]
                stdout = "".join(json.dumps(event) + "\n" for event in events)
            return {"returncode": rc, "stdout": stdout, "stderr": ""}
        if lane == "agent":
            with self.lane_patches(child):
                return self.invoke(job_lane=True)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module.shutil, "which", return_value="/synthetic/vendor"))
            stack.enter_context(mock.patch.object(module, "run_cli", side_effect=child))
            stack.enter_context(mock.patch.object(module, "prepare_host_fence", side_effect=lambda argv, **kwargs: (argv, kwargs["env"], {})))
            if self.provider == "claude":
                stack.enter_context(mock.patch.object(module, "resolve_agent_binary", return_value="/synthetic/vendor"))
            call = module.run_claude_subprocess if self.provider == "claude" else module.run_grok_subprocess
            return asyncio.run(call("synthetic task", "synthetic-model", 5))

    def test_public_consumers_treat_success_metadata_as_data(self):
        for metadata, _reason in self.signals()[:4]:
            for lane in ("agent", "consult"):
                with self.subTest(lane=lane, metadata=metadata):
                    frame = self.frame("result", "success", False, metadata)
                    result = self.consume(lane, frame, 0)
                    self.assertNotEqual(health.seat_kind(self.model, "a"), "cooling", "successful metadata falsely flagged the account")
                    self.assertNotIn(result.get("reason"), ("rate_limit", "auth_error", "account_unavailable"))
                    self.assertTrue(result["ok"], result)
                    self.assertEqual(len(self.calls), 1)

    def test_public_consumers_require_definite_failure_metadata(self):
        ambiguous = [("result", "success", _MISSING), ("result", "unknown", _MISSING),
                     ("result", "error_during_execution", False), ("result", "success", "true"),
                     ("result", "success", 1), ("result", "success", {}),
                     ("result", "success", None), ("assistant", "success", True),
                     ("unknown", "error_during_execution", True), ("error", "unknown", False)]
        genuine = [("result", "success", True), ("result", "error_during_execution", _MISSING),
                   ("error", None, _MISSING), ("turn.failed", None, True)]
        for frames, positive in ((ambiguous, False), (genuine, True)):
            for kind, subtype, marker in frames:
                for metadata, reason in self.signals()[:4]:
                    for lane in ("agent", "consult"):
                        for rc in (0, 1):
                            with self.subTest(lane=lane, kind=kind, subtype=subtype, marker=repr(marker), metadata=metadata, rc=rc):
                                result = self.consume(lane, self.frame(kind, subtype, marker, metadata), rc)
                                self.assertEqual(len(self.calls), 1, "metadata caused cross-account replay")
                                if positive:
                                    self.assertFalse(result["ok"], result)
                                    self.assertEqual(result["reason"], reason)
                                    self.assertEqual(health.seat_kind(self.model, "a"), "cooling")
                                else:
                                    self.assertNotIn(result.get("reason"), ("rate_limit", "auth_error", "account_unavailable"), "ambiguous metadata was reclassified as an account failure")
                                    self.assertNotEqual(health.seat_kind(self.model, "a"), "cooling", "ambiguous metadata falsely flagged the account")


for _provider in ("claude", "grok"):
    globals()["Test" + _provider.title() + "TerminalMetadata"] = type(
        "Test" + _provider.title() + "TerminalMetadata", (TerminalMetadataContract, unittest.TestCase), {"provider": _provider})


class TestGrokProfiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="omni-grok-profiles-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        env = {"HOME": str(self.root), "PATH": os.environ.get("PATH", ""), "TMPDIR": "/private/tmp",
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_STATE_DIR": str(self.root / "state"),
               "OMNI_LLM_NOTIFY_QUIET": "1"}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_add_login_argv_env_registry_and_private_modes(self):
        from providers.profile_check import add_profile, list_profiles
        result = add_profile("grok", "work", persist=True)
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result["login_argv"], ["grok", "login"])
        self.assertEqual(result["login_env"], {"GROK_HOME": result["path"]})
        self.assertEqual(result["var"], "GROK_HOME_WORK")
        self.assertEqual(result["mcp_env"], {"GROK_HOME_WORK": result["path"]})
        self.assertEqual(Path(result["path"]).name, ".grok-work")
        self.assertEqual(os.stat(result["path"]).st_mode & 0o777, 0o700)
        os.environ[result["var"]] = result["path"]
        rows = list_profiles(roads=("grok",))
        self.assertTrue(any(row["name"] == "work" and row["path"] == result["path"] for row in rows))

    def test_native_profile_resolution_and_unsafe_home_refused_before_spawn(self):
        from providers.profiles import resolve_profile
        safe = self.root / ".grok-work"
        safe.mkdir()
        os.environ["GROK_HOME_WORK"] = str(safe)
        with mock.patch("providers.kernel_sandbox.protected_home", return_value=str(self.root)):
            overrides, error = resolve_profile("grok", "work")
            self.assertIsNone(error)
            self.assertEqual(overrides, {"GROK_HOME": str(safe)})
            os.environ["GROK_HOME_WORK"] = str(self.root)
            from providers import grok
            with mock.patch.object(grok, "run_cli") as spawn:
                result = asyncio.run(grok.run_grok_subprocess("synthetic task", "synthetic-model", 5, account_profile="work"))
            spawn.assert_not_called()
        self.assertEqual(result["reason"], "unsupported_profile")

    def test_check_unknown_quota_and_live_probe_keep_requested_account(self):
        from providers.profile_check import check, list_profiles
        from providers import grok
        home = self.root / ".grok-work"
        home.mkdir()
        os.environ["GROK_HOME_WORK"] = str(home)
        os.environ["OMNI_LLM_PROFILES_GROK"] = "missing,work,other"
        rows = [row for row in list_profiles(roads=("grok",)) if row["name"] == "work"]
        self.assertEqual(len(rows), 1)
        offline = check(rows, live=False)
        self.assertEqual(offline["profiles"][0]["state"], "quota_unknown")
        self.assertNotIn("used_percent", offline["profiles"][0])
        seen = []
        async def fake_attempt(prompt, model, timeout_s, account_profile, **kwargs):
            seen.append(account_profile)
            return {"ok": True, "text": "pong", "model": model}
        with mock.patch.object(grok, "_run_grok_once", side_effect=fake_attempt):
            checked = check(rows, live=True)
        self.assertEqual(seen, ["work"])
        self.assertTrue(checked["ok"], checked)
        self.assertEqual(checked["profiles"][0]["state"], "session_ok")

    def test_numeric_oracle_coverage_does_not_claim_grok_or_claude(self):
        from providers.profile_check import probe_covered_roads
        covered = probe_covered_roads()
        self.assertIn("codex", covered)
        self.assertNotIn("grok", covered)
        self.assertNotIn("claude", covered)


class TestCodexStrictStartupJSON(unittest.TestCase):
    """A permissive JSON decoder must never provide permission to replay work."""
    provider = "codex"
    setUp = ProviderContract.setUp
    lane_patches = ProviderContract.lane_patches
    invoke = ProviderContract.invoke

    @staticmethod
    def ambiguous_streams():
        return {
            "root_activity_overwritten": '{"type":"item.started","item":{"type":"command_execution"},"type":"error","code":"usage_limit_reached"}',
            "root_equal_type": '{"type":"error","type":"error","code":"usage_limit_reached"}',
            "root_unknown_field": '{"type":"error","code":"usage_limit_reached","extra":1,"extra":1}',
            "escaped_equivalent_name": r'{"type":"error","code":"usage_limit_reached","\u0063ode":"usage_limit_reached"}',
            "nested_error_code": '{"type":"turn.failed","error":{"code":"unexpected_failure","code":"usage_limit_reached"}}',
            "nested_unknown_field": '{"type":"error","code":"usage_limit_reached","extra":{"label":"first","label":"last"}}',
            "array_nested_unknown_field": '{"type":"error","code":"usage_limit_reached","extra":[{"x":0,"x":0}]}',
        }

    @staticmethod
    def nonstandard_streams():
        streams = {}
        for token in ("NaN", "Infinity", "-Infinity"):
            streams["root_" + token] = '{"type":"error","code":"usage_limit_reached","extra":' + token + '}'
            streams["nested_" + token] = '{"type":"error","code":"usage_limit_reached","extra":{"values":[' + token + ']}}'
        return streams

    @staticmethod
    def recursive_stream():
        depth = sys.getrecursionlimit() + 100
        return '{"type":"error","code":"usage_limit_reached","extra":' + '[' * depth + '0' + ']' * depth + '}'

    def evidence_without_decoder_exception(self, stream):
        from providers.codex_accounts import startup_evidence
        try:
            return startup_evidence(stream)
        except RecursionError:
            self.fail("startup evidence leaked decoding recursion instead of denying replay")

    def test_duplicate_names_anywhere_cannot_authorize_retry(self):
        for name, stream in self.ambiguous_streams().items():
            with self.subTest(case=name):
                self.assertIsNone(self.evidence_without_decoder_exception(stream)[0])

    def test_nonstandard_numbers_anywhere_cannot_authorize_retry(self):
        for name, stream in self.nonstandard_streams().items():
            with self.subTest(case=name):
                self.assertIsNone(self.evidence_without_decoder_exception(stream)[0])

    def test_excessive_decoding_recursion_denies_retry_without_escaping(self):
        self.assertIsNone(self.evidence_without_decoder_exception(self.recursive_stream())[0])

    def test_valid_unique_json_and_structured_startup_errors_stay_compatible(self):
        streams = {
            "unique_values": '{"type":"error","code":"usage_limit_reached","extra":{"numbers":[0,-7,1.25,1e300],"strings":["NaN","Infinity","-Infinity"],"rows":[{"label":1},{"label":2}]}}',
            "nested_error": '{"type":"turn.failed","error":{"code":"usage_limit_reached","message":"synthetic limit"}}',
            "message_error": '{"type":"error","message":"You have hit your usage limit"}',
        }
        for name, stream in streams.items():
            with self.subTest(case=name):
                self.assertEqual(self.evidence_without_decoder_exception(stream), ("rate_limit", False))

    def fake_startup_vendor(self, stream):
        binary = self.root / "fake-startup-vendor"
        capture = self.root / "startup-children.jsonl"
        marker = self.root / "startup-side-effects"
        capture.write_text("", encoding="utf-8")
        marker.write_text("", encoding="utf-8")
        script = "#!" + sys.executable + "\nimport json, os, sys\nfrom pathlib import Path\nsys.stdin.read()\n"
        script += "with open(" + repr(str(capture)) + ", 'a') as out:\n"
        script += "    out.write(json.dumps({'home': os.environ.get('CODEX_HOME')}) + '\\n')\n"
        script += "with open(" + repr(str(marker)) + ", 'a') as out:\n    out.write('executed\\n')\n"
        script += "print(" + repr(stream) + ")\nprint('usage limit reached', file=sys.stderr)\nraise SystemExit(1)\n"
        binary.write_text(script, encoding="utf-8")
        binary.chmod(0o700)
        return binary, capture, marker

    def assert_actual_children(self, stream, expected_homes):
        health.clear_all()
        binary, capture, marker = self.fake_startup_vendor(stream)
        escaped = None
        result = None
        with self.lane_patches(binary=str(binary)), mock.patch.object(agent_lane, "read_successor_attest", return_value=None):
            try:
                result = self.invoke()
            except RecursionError as error:
                escaped = error
        rows = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), len(expected_homes), "startup evidence replayed the local fake task")
        self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["executed"] * len(expected_homes))
        self.assertEqual([row["home"] for row in rows], expected_homes)
        self.assertIsNone(escaped, "public run_agent leaked decoding recursion")
        self.assertFalse(result["ok"])
        return result

    def test_actual_ambiguous_and_nonstandard_startup_data_run_one_child(self):
        streams = {**self.ambiguous_streams(), **self.nonstandard_streams(), "recursion": self.recursive_stream()}
        for name, stream in streams.items():
            with self.subTest(case=name):
                self.assert_actual_children(stream, [self.homes["a"]])

    def test_actual_valid_activity_then_error_remains_terminal(self):
        stream = '{"type":"item.started","item":{"type":"command_execution"}}\n{"type":"error","code":"usage_limit_reached"}'
        result = self.assert_actual_children(stream, [self.homes["a"]])
        self.assertEqual(result["reason"], "rate_limit")

    def test_actual_definite_error_only_startup_runs_one_child_and_flags_the_account(self):
        stream = '{"type":"turn.failed","error":{"code":"usage_limit_reached"},"extra":{"numbers":[0,1.25],"label":"NaN"}}'
        result = self.assert_actual_children(stream, [self.homes["a"]])
        self.assertEqual(result["reason"], "rate_limit")
        self.assertEqual(result["account_profile"], "a")
        self.assertEqual(health.seat_kind(self.model, "a"), "cooling")


class TestDocumentedAccountExamples(unittest.TestCase):
    def test_registry_table_maps_through_production_resolver(self):
        from providers.profiles import account_ring, resolve_profile
        document = Path(__file__).resolve().parent.parent / "docs" / "account-failover.md"
        lines = document.read_text().splitlines()
        rows = [line for line in lines if line.startswith("| ") and "`" in line]
        self.assertEqual(len(rows), 3)
        with tempfile.TemporaryDirectory(prefix="omni-doc-account-") as root:
            for row in rows:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                provider = cells[0].lower()
                registry = re.findall(r"`([^`]+)`", cells[1])
                native = re.findall(r"`([^`]+)`", cells[2])[0]
                ring_var, labels = re.findall(r"`([^`]+)`", cells[3])[0].split("=", 1)
                labels = labels.split(",")
                self.assertEqual(len(registry), len(labels))
                env = {ring_var: ",".join(labels), "OMNI_LLM_QUOTA_PROBE": "0"}
                homes = {}
                for var, label in zip(registry, labels):
                    home = Path(root).resolve() / provider / label
                    home.mkdir(parents=True)
                    homes[label] = str(home)
                    env[var] = str(home)
                with mock.patch.dict(os.environ, env, clear=True):
                    names, _discovered, error = account_ring(provider)
                    self.assertIsNone(error, provider)
                    self.assertEqual(names, labels)
                    for label in labels:
                        overrides, error = resolve_profile(provider, label)
                        self.assertIsNone(error, provider)
                        self.assertEqual(overrides, {native: homes[label]})


class TestNestedAccountHomesRefused(unittest.TestCase):
    """Registered homes of any provider must be disjoint (LOUD profile_config)."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-nested-homes-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def _dir(self, *parts):
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _resolve(self, provider, label, env):
        from providers.profiles import resolve_profile
        with mock.patch.dict(os.environ, env, clear=True):
            return resolve_profile(provider, label)

    def test_nested_codex_homes_are_refused_for_both_labels(self):
        outer = self._dir("x", "a")
        inner = self._dir("x", "a", "b")
        env = {"CODEX_HOME_A": outer, "CODEX_HOME_B": inner}
        for label in ("a", "b"):
            with self.subTest(label=label):
                overrides, error = self._resolve("codex", label, env)
                self.assertEqual(overrides, {})
                self.assertEqual(error["reason"], "profile_config", error)
                self.assertIn("CODEX_HOME_A", error["detail"])
                self.assertIn("CODEX_HOME_B", error["detail"])
                self.assertNotIn(str(self.root), error["detail"])

    def test_nesting_in_other_case_or_normalization_is_refused(self):
        import unicodedata
        outer = self._dir("x", "Caf\u00e9")
        inner = self._dir("x", "Caf\u00e9", "b")
        spelled = os.path.join(os.path.dirname(outer), unicodedata.normalize(
            "NFD", os.path.basename(outer)).upper())
        env = {"CODEX_HOME_A": spelled, "CODEX_HOME_B": inner}
        _overrides, error = self._resolve("codex", "b", env)
        self.assertEqual(error["reason"], "profile_config", error)
        self.assertIn("CODEX_HOME_A", error["detail"])

    def test_nesting_across_providers_is_refused(self):
        outer = self._dir("claude-a")
        inner = self._dir("claude-a", "grok-b")
        env = {"CLAUDE_CONFIG_DIR_A": outer, "GROK_HOME_B": inner}
        for provider, label in (("claude", "a"), ("grok", "b")):
            with self.subTest(provider=provider):
                _overrides, error = self._resolve(provider, label, env)
                self.assertEqual(error["reason"], "profile_config", error)
                self.assertIn("CLAUDE_CONFIG_DIR_A", error["detail"])
                self.assertIn("GROK_HOME_B", error["detail"])

    def test_one_directory_registered_for_two_providers_is_refused(self):
        shared = self._dir("shared")
        env = {"CODEX_HOME_A": shared, "GROK_HOME_A": shared}
        for provider in ("codex", "grok"):
            with self.subTest(provider=provider):
                _overrides, error = self._resolve(provider, "a", env)
                self.assertEqual(error["reason"], "profile_config", error)

    def test_same_provider_alias_of_one_directory_keeps_ring_dedup(self):
        shared = self._dir("shared")
        env = {"CODEX_HOME_A": shared, "CODEX_HOME_B": shared}
        overrides, error = self._resolve("codex", "a", env)
        self.assertIsNone(error)
        self.assertEqual(overrides, {"CODEX_HOME": shared})

    def test_disjoint_homes_of_every_provider_are_unaffected(self):
        env = {"CODEX_HOME_A": self._dir("x", "a"), "CODEX_HOME_B": self._dir("x", "b"),
               "CLAUDE_CONFIG_DIR_A": self._dir("y", "a"), "GROK_HOME_A": self._dir("z", "a")}
        for provider, native, label in (("codex", "CODEX_HOME", "a"), ("codex", "CODEX_HOME", "b"),
                                        ("claude", "CLAUDE_CONFIG_DIR", "a"), ("grok", "GROK_HOME", "a")):
            with self.subTest(provider=provider, label=label):
                overrides, error = self._resolve(provider, label, env)
                self.assertIsNone(error)
                self.assertEqual(overrides, {native: env[native + "_" + label.upper()]})

    def test_cursor_config_dir_nested_with_an_account_home_is_refused(self):
        cursor = self._dir("outside", "cursor")
        inner = self._dir("outside", "cursor", "grok-b")
        env = {"OMNI_LLM_CURSOR_CONFIG_DIR_C": cursor, "GROK_HOME_B": inner}
        for provider, label in (("cursor", "c"), ("grok", "b")):
            with self.subTest(provider=provider):
                overrides, error = self._resolve(provider, label, env)
                self.assertEqual(overrides, {})
                self.assertEqual(error["reason"], "profile_config", error)
                self.assertIn("OMNI_LLM_CURSOR_CONFIG_DIR_C", error["detail"])
                self.assertIn("GROK_HOME_B", error["detail"])
                self.assertNotIn(str(self.root), error["detail"])

    def test_cursor_config_dir_equal_to_an_account_home_is_refused(self):
        shared = self._dir("shared")
        env = {"OMNI_LLM_CURSOR_CONFIG_DIR_C": shared, "CLAUDE_CONFIG_DIR_A": shared}
        for provider, label in (("cursor", "c"), ("claude", "a")):
            with self.subTest(provider=provider):
                _overrides, error = self._resolve(provider, label, env)
                self.assertEqual(error["reason"], "profile_config", error)

    def test_cursor_account_homes_that_contain_the_cursor_dir_are_refused(self):
        outer = self._dir("outer")
        cursor = self._dir("outer", "cursor")
        env = {"OMNI_LLM_CURSOR_CONFIG_DIR_C": cursor, "CODEX_HOME_A": outer}
        _overrides, error = self._resolve("cursor", "c", env)
        self.assertEqual(error["reason"], "profile_config", error)

    def test_disjoint_cursor_config_dir_is_unaffected(self):
        env = {"OMNI_LLM_CURSOR_CONFIG_DIR_C": self._dir("cur"),
               "GROK_HOME_B": self._dir("grok", "b"),
               "CODEX_HOME_A": self._dir("codex", "a")}
        overrides, error = self._resolve("cursor", "c", env)
        self.assertIsNone(error)
        self.assertEqual(overrides, {"OMNI_LLM_CURSOR_CONFIG_DIR": env["OMNI_LLM_CURSOR_CONFIG_DIR_C"]})
        for provider, label in (("grok", "b"), ("codex", "a")):
            _overrides, error = self._resolve(provider, label, env)
            self.assertIsNone(error, provider)

    def test_profile_check_reports_nested_home(self):
        from providers import profile_check
        env = {"CODEX_HOME_A": self._dir("x", "a"), "CODEX_HOME_B": self._dir("x", "a", "b")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(profile_check._shape_env("codex", "b"), ["nested_home"])
        env = {"CODEX_HOME_A": self._dir("x2", "a"), "CODEX_HOME_B": self._dir("x2", "b")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(profile_check._shape_env("codex", "b"), [])


HELPER = "/synthetic/bin/print-key --label fixture"
SUCCESS = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "result": "synthetic ok"})
CREDIT = ("Your credit balance is too low to access the Anthropic API. "
          "Please go to Plans & Billing to upgrade or purchase credits.")
CREDIT_ERROR = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                           "api_error_status": 400, "result": CREDIT})
LAUNCH_CONTEXT = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
                  "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_CHILD_SESSION",
                  "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
                  "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SSE_PORT")

FAKE = r'''import json, os, stat, sys
sys.stdin.read()
argv = sys.argv[1:]
home = os.environ.get("CLAUDE_CONFIG_DIR")
row = {"argv": argv, "home": home,
       "env": {k: v for k, v in os.environ.items() if k.startswith("CLAUDE_CODE") or k == "CLAUDECODE"}}
if "--settings" in argv:
    path = argv[argv.index("--settings") + 1]
    st = os.stat(path)
    row["settings"] = {"path": path, "mode": stat.S_IMODE(st.st_mode), "content": open(path).read()}
with open(os.environ["OMNI_SYNTHETIC_CAPTURE"], "a") as handle:
    handle.write(json.dumps(row) + "\n")
try:
    table = json.load(open(os.environ["OMNI_SYNTHETIC_RESPONSES"]))
except OSError:
    table = {}
reply = table.get(os.path.basename(home or ""), {})
if "stream-json" in argv and "stdout" not in reply:
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "synthetic ok"}]}}))
print(reply.get("stdout", os.environ["OMNI_SYNTHETIC_DEFAULT"]))
sys.exit(reply.get("rc", 0))
'''


class ApiFixture(unittest.TestCase):
    provider = "claude"
    lane_patches = ProviderContract.lane_patches
    job_patches = ProviderContract.job_patches
    start_job = ProviderContract.start_job
    success = ProviderContract.success

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-api-fallback-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.homes = {}
        for label in ("a", "api"):
            self.homes[label] = str(self.root / label)
            Path(self.homes[label]).mkdir()
        Path(self.homes["api"], "settings.json").write_text(
            json.dumps({"apiKeyHelper": HELPER, "permissions": {"allow": ["Bash"]}}))
        self.workdir = str(self.root / "work")
        Path(self.workdir).mkdir()
        self.model = "claude/synthetic-model"
        self.native = NATIVE["claude"]
        self.ring = "OMNI_LLM_PROFILES_CLAUDE"
        self.binary = self.root / "fake-claude"
        self.binary.write_text("#!" + sys.executable + "\n" + FAKE)
        self.binary.chmod(0o700)
        self.capture = self.root / "capture.jsonl"
        self.responses = self.root / "responses.json"
        env = {"PATH": os.environ.get("PATH", ""), "TMPDIR": "/private/tmp",
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_STATE_DIR": str(self.root / "state"),
               "OMNI_LLM_NOTIFY_QUIET": "1", self.ring: "a,api",
               "OMNI_LLM_API_PROFILES_CLAUDE": "api", "OMNI_LLM_API_FALLBACK_CLAUDE": "1",
               "CLAUDE_CONFIG_DIR_A": self.homes["a"], "CLAUDE_CONFIG_DIR_API": self.homes["api"],
               "OMNI_SYNTHETIC_CAPTURE": str(self.capture),
               "OMNI_SYNTHETIC_RESPONSES": str(self.responses),
               "OMNI_SYNTHETIC_DEFAULT": SUCCESS,
               # The fenced fake vendor gets an allowlisted environment
               # (#138): its control variables are passed by name.
               "OMNI_LLM_CHILD_ENV_PASS": ",".join((
                   "OMNI_SYNTHETIC_CAPTURE", "OMNI_SYNTHETIC_RESPONSES",
                   "OMNI_SYNTHETIC_DEFAULT"))}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(health, "_account_identity", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        # Generated API settings live in a private area outside the child's
        # write roots; keep the fixture's out of the real home.
        self.private_base = self.root / "private-api-settings"
        self.private_base.mkdir(mode=0o700)
        patch = mock.patch.object(agent_lane, "_api_settings_base",
                                  return_value=str(self.private_base))
        patch.start()
        self.addCleanup(patch.stop)
        # The suite's fake binary stands in for every `which` lookup, git
        # included; the repository-root pre-filter is exercised on its own.
        patch = mock.patch.object(agent_lane.host_fence_mod, "repo_root_candidates",
                                  return_value=())
        patch.start()
        self.addCleanup(patch.stop)

    def respond(self, **table):
        self.responses.write_text(json.dumps(table))

    def rows(self):
        if not self.capture.exists():
            return []
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def agent(self, **kwargs):
        with self.lane_patches(binary=str(self.binary)):
            return asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, **kwargs))

    def consult(self, **kwargs):
        from providers import claude
        with mock.patch.object(claude, "resolve_agent_binary", return_value=str(self.binary)), \
                mock.patch.object(claude, "prepare_host_fence",
                                  side_effect=lambda argv, **k: (argv, k["env"], {})):
            return asyncio.run(claude.run_claude_subprocess(
                "synthetic task", "synthetic-model", 5, **kwargs))

    def run_job(self, **kwargs):
        origin = self.start_job(**kwargs)
        self.assertNotIn("refused", origin)
        data = jobs._read_job(origin["job_dir"])
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        return origin, result

    def ledger(self, label):
        return [entry for entry in health._read_ledger().values() if entry.get("profile") == label]


class TestKindSelection(ApiFixture):
    def test_t1_cooled_subscription_opens_api_path_consult_and_agent(self):
        for lane in (self.consult, self.agent):
            with self.subTest(lane=lane.__name__):
                health.clear_all()
                self.capture.unlink(missing_ok=True)
                health.flag(self.model, "a")
                result = lane()
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["account_profile"], "api")
                self.assertEqual(result["account_kind"], "api")
                self.assertEqual(result["tried"], [{"account_profile": "a", "reason": "skipped_cooldown",
                                                    "account_kind": "subscription"}])
                self.assertEqual([row["home"] for row in self.rows()], [self.homes["api"]])

    def test_t2_fallback_disabled_never_spawns_api(self):
        for value in (None, "0", "true", ""):
            with self.subTest(value=value):
                health.clear_all()
                self.capture.unlink(missing_ok=True)
                if value is None:
                    os.environ.pop("OMNI_LLM_API_FALLBACK_CLAUDE")
                else:
                    os.environ["OMNI_LLM_API_FALLBACK_CLAUDE"] = value
                health.flag(self.model, "a")
                for lane in (self.consult, self.agent):
                    result = lane()
                    self.assertEqual(result["reason"], "account_exhausted", result)
                    self.assertEqual(result["tried"], [
                        {"account_profile": "a", "reason": "skipped_cooldown", "account_kind": "subscription"},
                        {"account_profile": "api", "reason": "api_disabled", "account_kind": "api"}])
                self.assertFalse(self.capture.exists(), "API child was spawned while fallback was off")

    def test_t3_non_cooldown_skip_closes_api_path(self):
        os.environ["CLAUDE_CONFIG_DIR_A"] = str(self.root / "missing")
        for lane in (self.consult, self.agent):
            result = lane()
            self.assertEqual(result["reason"], "account_exhausted", result)
            self.assertEqual(result["tried"], [
                {"account_profile": "a", "reason": "account_unavailable", "account_kind": "subscription"},
                {"account_profile": "api", "reason": "api_not_permitted", "account_kind": "api"}])
        self.assertFalse(self.capture.exists(), "API child was spawned after a non-cooldown skip")

    def test_api_first_in_ring_is_not_a_fallback(self):
        os.environ[self.ring] = "api,a"
        result = self.consult()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["account_profile"], "a")
        self.assertEqual(result["account_kind"], "subscription")
        self.assertEqual(result["tried"], [{"account_profile": "api", "reason": "api_not_permitted",
                                            "account_kind": "api"}])

    def test_eligible_subscription_wins_and_reports_subscription_kind(self):
        result = self.consult()
        self.assertEqual((result["account_profile"], result["account_kind"]), ("a", "subscription"))
        self.assertNotIn("--max-budget-usd", self.rows()[0]["argv"])

    def test_explicit_pin_runs_api_when_opted_in_and_reports_kind(self):
        health.flag(self.model, "a")
        for lane in (self.consult, self.agent):
            result = lane(account_profile="api")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["account_kind"], "api")
        self.assertEqual([row["home"] for row in self.rows()], [self.homes["api"]] * 2)

    def test_declaration_errors_are_loud_profile_config(self):
        for value in ("default", "ghost", "api,api", "a,", "BAD NAME", ""):
            with self.subTest(value=value):
                os.environ["OMNI_LLM_API_PROFILES_CLAUDE"] = value
                result = self.consult()
                self.assertEqual(result["reason"], "profile_config", result)
                self.assertIn("OMNI_LLM_API_PROFILES_CLAUDE", result["detail"])
        self.assertFalse(self.capture.exists())

    def test_kind_suffix_spelling_is_not_a_profile(self):
        # The kind is declared only through the list variable.
        os.environ["CLAUDE_CONFIG_DIR_API_KIND"] = "api"
        os.environ.pop("OMNI_LLM_API_PROFILES_CLAUDE")
        self.assertEqual(provider_accounts.account_kind("claude", "api"), "subscription")


class TestCreditClassification(ApiFixture):
    def test_t4_credit_envelope_holds_api_until_cleared(self):
        health.flag(self.model, "a")
        self.respond(api={"stdout": CREDIT_ERROR, "rc": 1})
        result = self.consult()
        self.assertEqual(result["reason"], "credit_exhausted", result)
        self.assertEqual(result["account_kind"], "api")
        entries = self.ledger("api")
        self.assertEqual([entry["source"] for entry in entries], ["credit_exhausted"])
        self.assertGreater(entries[0]["until"] - entries[0]["flagged_at"], 365 * 86400)
        self.assertEqual(len(self.rows()), 1)
        # A new request skips the cooled API account and never replays within the request.
        again = self.consult()
        self.assertEqual(again["reason"], "account_exhausted")
        self.assertEqual(again["tried"][-1], {"account_profile": "api", "reason": "skipped_cooldown",
                                              "account_kind": "api"})
        self.assertEqual(len(self.rows()), 1)

    def test_t4_cooldown_env_and_exit_zero_envelope(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "120"
        health.flag(self.model, "a")
        self.respond(api={"stdout": CREDIT_ERROR, "rc": 0})
        self.assertEqual(self.consult()["reason"], "credit_exhausted")
        entry = self.ledger("api")[0]
        self.assertAlmostEqual(entry["until"] - entry["flagged_at"], 120, delta=2)

    def test_t4_credit_hold_is_not_probed_early(self):
        health.flag(self.model, "a")
        self.respond(api={"stdout": CREDIT_ERROR, "rc": 1})
        self.consult()
        later = self.ledger("api")[0]["flagged_at"] + 6 * 3600 + 60
        self.assertEqual(health.seat_kind(self.model, "api", now=later), "cooling")
        # Selection reads the clock through health: run it far past the old
        # six hour window and past any half-open time, subscription still cooling.
        clock = mock.Mock(wraps=time)
        clock.time = lambda: later + 30 * 86400
        with mock.patch.object(health, "time", clock):
            health.flag(self.model, "a", source="manual")
            self.assertEqual(health.seat_kind(self.model, "a"), "cooling")
            self.assertEqual(health.seat_kind(self.model, "api"), "cooling")
            self.assertFalse(health.snapshot()["claude@api"]["will_recheck"])
            again = self.consult()
        self.assertEqual(again["reason"], "account_exhausted", again)
        self.assertEqual(again["tried"], [
            {"account_profile": "a", "reason": "skipped_cooldown", "account_kind": "subscription"},
            {"account_profile": "api", "reason": "skipped_cooldown", "account_kind": "api"}])
        self.assertEqual(len(self.rows()), 1, again)

    def test_t4_job_lane_credit_envelope(self):
        health.flag(self.model, "a")
        self.respond(api={"stdout": CREDIT_ERROR, "rc": 1})
        _origin, result = self.run_job()
        self.assertEqual(result["reason"], "credit_exhausted", result)
        self.assertEqual([entry["source"] for entry in self.ledger("api")], ["credit_exhausted"])

    def test_workspace_error_is_account_unavailable(self):
        body = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                           "api_error_status": 400,
                           "result": "This API key is not scoped to a workspace, so this request must "
                                     "include the anthropic-workspace-id header"})
        health.flag(self.model, "a")
        self.respond(api={"stdout": body, "rc": 1})
        self.assertEqual(self.consult()["reason"], "account_unavailable")

    def test_other_400_stays_runtime_error(self):
        body = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                           "api_error_status": 400, "result": "invalid_request_error: bad model"})
        health.flag(self.model, "a")
        self.respond(api={"stdout": body, "rc": 1})
        self.assertEqual(self.consult()["reason"], "runtime_error")
        self.assertEqual(self.ledger("api"), [])

    def test_t5_successful_answer_quoting_the_text_is_ok_and_cools_nothing(self):
        quote = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                            "result": "The docs say: " + CREDIT})
        health.flag(self.model, "a")
        self.respond(api={"stdout": quote, "rc": 0})
        result = self.consult()
        self.assertTrue(result["ok"], result)
        self.assertIn("credit balance is too low", result["text"])
        self.assertEqual(self.ledger("api"), [])
        self.assertEqual(health.seat_kind(self.model, "api"), "healthy")
        # Plain agent output that merely quotes the text is an answer too.
        self.respond(api={"stdout": CREDIT, "rc": 0})
        again = self.agent()
        self.assertTrue(again["ok"], again)
        self.assertEqual(self.ledger("api"), [])


class TestCreditUserDecision(ApiFixture):
    """The user, not omni, decides whether to add credits: hold, tell, clear."""
    HINT_WORDS = ("'api'", "Claude Console", "purchased credits are charged",
                  "omni health clear claude --profile api")

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(provider_accounts, "notify")
        self.notify = patch.start()
        self.addCleanup(patch.stop)
        provider_accounts._credit_invalid_logged.clear()
        health.flag(self.model, "a")

    def fail_credit(self):
        self.respond(api={"stdout": CREDIT_ERROR, "rc": 1})
        return self.consult()

    def select(self):
        return provider_accounts.AccountSelection("default", self.model, 5, provider="claude")

    def test_c1_envelope_names_label_only_and_how_to_continue(self):
        result = self.fail_credit()
        self.assertEqual(result["reason"], "credit_exhausted", result)
        for word in self.HINT_WORDS:
            self.assertIn(word, result["detail"])
        for leak in (str(self.root), self.homes["api"], "sk-", HELPER, "Plans & Billing"):
            self.assertNotIn(leak, result["detail"])

    def test_c1_account_exhausted_adds_the_same_hint(self):
        self.fail_credit()
        again = self.consult()
        self.assertEqual(again["reason"], "account_exhausted", again)
        self.assertEqual(again["tried"][-1]["reason"], "skipped_cooldown")
        for word in self.HINT_WORDS:
            self.assertIn(word, again["detail"])
        self.assertNotIn(str(self.root), again["detail"])

    def test_c1_plain_exhaustion_keeps_the_generic_detail(self):
        health.flag(self.model, "api")
        again = self.consult()
        self.assertEqual(again["reason"], "account_exhausted", again)
        self.assertEqual(again["detail"], "no eligible claude account")

    def test_c2_default_hold_has_no_expiry_for_practical_purposes(self):
        self.fail_credit()
        entry = self.ledger("api")[0]
        self.assertEqual(entry["source"], "credit_exhausted")
        self.assertGreater(entry["until"] - entry["flagged_at"], 365 * 86400)
        self.assertEqual(entry["probe_after"], entry["until"])
        clock = mock.Mock(wraps=time)
        clock.time = lambda: entry["flagged_at"] + 30 * 86400
        with mock.patch.object(health, "time", clock):
            self.assertEqual(health.seat_kind(self.model, "api"), "cooling")

    def test_c2_env_window_is_honoured(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "90"
        self.fail_credit()
        entry = self.ledger("api")[0]
        self.assertAlmostEqual(entry["until"] - entry["flagged_at"], 90, delta=2)
        self.assertEqual(health.seat_kind(self.model, "api", now=entry["until"] + 1), "healthy")

    def test_c2_invalid_env_holds_and_logs_once(self):
        for bad in ("abc", "0", "-5", "1.5", "", "inf", "nan", "1e3"):
            with self.subTest(value=bad):
                provider_accounts._credit_invalid_logged.clear()
                os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = bad
                with mock.patch.object(provider_accounts, "log_event") as logged:
                    self.assertIsNone(provider_accounts.credit_cooldown_s())
                    self.assertIsNone(provider_accounts.credit_cooldown_s())
                self.assertEqual(logged.call_count, 1)
                self.assertEqual(logged.call_args[0][0], "credit_cooldown_invalid")
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "bad-end-to-end"
        self.fail_credit()
        entry = self.ledger("api")[0]
        self.assertGreater(entry["until"] - entry["flagged_at"], 365 * 86400)

    def test_c3_one_notification_on_first_raise_only(self):
        self.fail_credit()
        self.assertEqual(self.notify.call_count, 1)
        args, kwargs = self.notify.call_args
        self.assertEqual(args, ("api_credit_exhausted",))
        self.assertEqual(kwargs, {
            "provider": "claude", "account_profile": "api",
            "action": "add credits in the Claude Console, then omni health clear claude --profile api"})
        self.consult()
        self.consult()
        self.assertEqual(self.notify.call_count, 1)

    def test_c3_kept_flags_do_not_notify(self):
        account = {"account_profile": "api"}
        result = {"reason": "credit_exhausted"}
        # A longer flag from another cause keeps the generic source, but the
        # credit evidence is recorded beside it and is a new raise.
        health.flag("claude", "api", until=time.time() + 90 * 86400, source="rate_limit")
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "120"
        self.select().failed(account, result)
        self.assertEqual([entry["source"] for entry in self.ledger("api")], ["rate_limit"])
        self.assertIsNotNone(health.credit_hold(self.ledger("api")[0]))
        self.assertEqual(self.notify.call_count, 1)
        # An existing credit hold is not a new raise either.
        health.clear("claude", "api")
        self.notify.reset_mock()
        self.select().failed(account, result)
        self.assertEqual(self.notify.call_count, 1)
        self.select().failed(account, result)
        self.assertEqual(self.notify.call_count, 1)

    def test_cooldown_source_names_the_actual_failure(self):
        # After tool activity the source is the failure reason itself (an
        # auth_error used to be recorded as rate_limit); without tool activity
        # it stays account_startup. Cooldown, half-open probing and the
        # quota-probe rule (only source=quota is cleared) are unchanged.
        cases = (
            ("claude", "a", {"reason": "auth_error", "tool_activity": True}, "auth_error"),
            ("claude", "b", {"reason": "account_unavailable", "tool_activity": True}, "account_unavailable"),
            ("claude", "c", {"reason": "rate_limit", "tool_activity": True}, "rate_limit"),
            ("claude", "d", {"reason": "auth_error", "tool_activity": False}, "account_startup"),
            ("codex", "e", {"reason": "auth_error", "tool_activity": True}, "auth_error"),
            ("codex", "f", {"reason": "rate_limit", "tool_activity": True}, "rate_limit"))
        for provider, profile, result, source in cases:
            with self.subTest(provider=provider, profile=profile):
                selection = (self.select() if provider == "claude" else
                             provider_accounts.AccountSelection("default", "codex", 5, provider="codex"))
                selection.failed({"account_profile": profile}, result)
                key = health.pool_key(provider, profile)
                entry = health.snapshot()[key]
                self.assertEqual(entry["source"], source)
                self.assertTrue(entry["will_recheck"])
                flagged = entry["flagged_at"]
                self.assertAlmostEqual(entry["until"] - flagged, health.DEFAULT_COOLDOWN_S, delta=2)
                self.assertAlmostEqual(entry["probe_after"] - flagged, health.HALF_OPEN_MIN_S, delta=2)
                if provider == "codex":
                    health.apply_codex_quota("codex", profile, {
                        "used_percent": 69.0, "reached": False, "window_mins": 10080,
                        "resets_at": time.time() + 86400}, None)
                    self.assertEqual(health.snapshot()[key]["source"], source)

    def test_c3_subscription_credit_failure_does_not_notify(self):
        self.select().failed({"account_profile": "a"}, {"reason": "credit_exhausted"})
        self.assertEqual(self.notify.call_count, 0)

    def run_clear(self, *argv):
        import io
        import omni_cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = omni_cli.main(["health", "clear", *argv])
        return code, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_c4_clear_reopens_the_api_account(self):
        self.fail_credit()
        self.assertEqual(self.consult()["reason"], "account_exhausted")
        code, payload = self.run_clear("claude", "--profile", "api")
        self.assertEqual(code, 0)
        self.assertEqual(payload, {"ok": True, "cleared": ["claude@api"]})
        self.assertEqual(self.ledger("api"), [])
        self.respond()
        after = self.consult()
        self.assertTrue(after["ok"], after)
        self.assertEqual(after["account_profile"], "api")
        self.assertEqual(after["account_kind"], "api")

    def test_c4_clear_with_a_model_name_and_nothing_flagged(self):
        code, payload = self.run_clear("claude/synthetic-model", "--profile", "api")
        self.assertEqual((code, payload), (0, {"ok": True, "cleared": []}))
        self.fail_credit()
        code, payload = self.run_clear("claude/synthetic-model", "--profile", "api")
        self.assertEqual(code, 0)
        self.assertEqual(payload["cleared"], ["claude@api"])
        self.assertEqual(self.ledger("api"), [])


class TestCreditReviewFixes(ApiFixture):
    """Review fixes: holds survive unrelated successes; clear is symmetric,
    honest about the lock; overlapping raises notify once."""

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(provider_accounts, "notify")
        self.notify = patch.start()
        self.addCleanup(patch.stop)

    def run_clear(self, *argv):
        import io
        import omni_cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = omni_cli.main(["health", "clear", *argv])
        return code, json.loads(out.getvalue().strip().splitlines()[-1])

    def select(self, profile):
        return provider_accounts.AccountSelection(profile, self.model, 5, provider="claude")

    def succeed(self, profile):
        async def attempt(account, remaining, managed):
            return {"ok": True, "text": "synthetic"}
        return asyncio.run(provider_accounts.run_selected(self.select(profile), attempt))

    def hold(self, label="api"):
        health.flag("claude", label, until=time.time() + 90 * 86400, source="credit_exhausted")

    def test_v1_explicit_pin_success_keeps_the_credit_hold(self):
        self.hold()
        result = self.succeed("api")
        self.assertTrue(result["ok"], result)
        self.assertEqual([e["source"] for e in self.ledger("api")], ["credit_exhausted"])
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")

    def test_v1_dispatch_success_clear_keeps_the_credit_hold(self):
        from providers import dispatch
        self.hold()
        dispatch._health_clear("claude", "api")
        self.assertEqual([e["source"] for e in self.ledger("api")], ["credit_exhausted"])
        health.flag("claude", "a", source="rate_limit")
        dispatch._health_clear("claude", "a")
        self.assertEqual(self.ledger("a"), [])

    def test_v1_sequential_subscription_success_still_clears_its_flag(self):
        health.flag("claude", "a", until=time.time() + 3600, source="rate_limit")
        time.sleep(0.01)
        self.hold()
        result = self.succeed("a")
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.ledger("a"), [])
        self.assertEqual([e["source"] for e in self.ledger("api")], ["credit_exhausted"])

    def test_v1_explicit_clear_still_ends_the_hold(self):
        self.hold()
        code, payload = self.run_clear("claude", "--profile", "api")
        self.assertEqual((code, payload), (0, {"ok": True, "cleared": ["claude@api"]}))
        self.assertEqual(self.ledger("api"), [])

    def test_v2_clear_bare_road_never_touches_another_road(self):
        health.flag("claude", "api", source="manual")
        health.flag("agy", "api", source="manual")
        code, payload = self.run_clear("claude", "--profile", "api")
        self.assertEqual((code, payload), (0, {"ok": True, "cleared": ["claude@api"]}))
        self.assertEqual(sorted(health._read_ledger()), ["agy@api"])

    def test_v2_clear_road_model_clears_pool_and_road_only(self):
        for model in ("claude/m", "claude", "agy", "claude/other"):
            health.flag(model, "api", source="manual")
        code, payload = self.run_clear("claude/m", "--profile", "api")
        self.assertEqual(code, 0)
        self.assertEqual(payload["cleared"], ["claude/m@api", "claude@api"])
        self.assertEqual(sorted(health._read_ledger()), ["agy@api", "claude/other@api"])

    def test_v2_clear_is_symmetric_with_flag_for_a_non_road_model(self):
        health.flag("gemini-x", "api", source="manual")
        health.flag("agy", "api", source="manual")
        code, payload = self.run_clear("gemini-x", "--profile", "api")
        self.assertEqual((code, payload), (0, {"ok": True, "cleared": ["gemini-x@api"]}))
        self.assertEqual(sorted(health._read_ledger()), ["agy@api"])

    def test_v3_overlapping_raises_notify_once(self):
        import threading
        barrier = threading.Barrier(2, timeout=5)
        real = health._flag_entry
        def meet(*args, **kwargs):
            barrier.wait()
            return real(*args, **kwargs)
        errors = []
        def raise_hold():
            try:
                self.select("default")._flag_credit_exhausted("api")
            except Exception as exc:  # surfaced below
                errors.append(exc)
        with mock.patch.object(health, "_flag_entry", side_effect=meet):
            threads = [threading.Thread(target=raise_hold) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(self.notify.call_count, 1)
        self.assertEqual([e["source"] for e in self.ledger("api")], ["credit_exhausted"])

    def _delayed_sink_raise(self, prior_s):
        """Raise the api hold while the seat_flagged sink 'blocks' 200 s."""
        if prior_s:
            health.flag("claude", "api", until=time.time() + prior_s,
                        source="credit_exhausted")
        offset = [0.0]
        real = time.time
        def slow_emit(*args, **kwargs):
            offset[0] += 200.0
        with mock.patch.object(time, "time", side_effect=lambda: real() + offset[0]), \
                mock.patch.object(health, "_emit_seat", side_effect=slow_emit):
            self.select("default")._flag_credit_exhausted("api")

    def test_v5_prior_hold_expiring_during_slow_sink_does_not_renotify(self):
        self._delayed_sink_raise(100)
        self.assertEqual(self.notify.call_count, 0)

    def test_v5_first_raise_notifies_exactly_once(self):
        self._delayed_sink_raise(0)
        self.assertEqual(self.notify.call_count, 1)

    def test_v4_clear_reports_an_unavailable_lock(self):
        self.hold()
        with mock.patch.object(health, "_locked_mutate", return_value=health._LOCK_SKIPPED):
            code, payload = self.run_clear("claude", "--profile", "api")
        self.assertEqual(code, 1)
        self.assertEqual(payload["ok"], False)
        self.assertEqual(payload["reason"], "health_lock_unavailable")
        self.assertEqual([e["source"] for e in self.ledger("api")], ["credit_exhausted"])

    def test_v4_nothing_flagged_stays_ok(self):
        code, payload = self.run_clear("claude", "--profile", "api")
        self.assertEqual((code, payload), (0, {"ok": True, "cleared": []}))

    def test_v4_success_path_survives_an_unavailable_lock(self):
        with mock.patch.object(health, "_locked_mutate", return_value=health._LOCK_SKIPPED):
            result = self.succeed("a")
        self.assertTrue(result["ok"], result)


class TestCreditHoldIndependentOfCooldown(ApiFixture):
    """Credit evidence is kept apart from the generic until/source precedence."""

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(provider_accounts, "notify")
        self.notify = patch.start()
        self.addCleanup(patch.stop)
        provider_accounts._credit_invalid_logged.clear()
        self.offset = 0.0
        real = time.time
        patch = mock.patch.object(time, "time", side_effect=lambda: real() + self.offset)
        patch.start()
        self.addCleanup(patch.stop)

    def select(self, profile="default"):
        return provider_accounts.AccountSelection(profile, self.model, 5, provider="claude")

    def raise_credit(self):
        self.select()._flag_credit_exhausted("api")

    def succeed(self):
        async def attempt(account, remaining, managed):
            return {"ok": True, "text": "synthetic"}
        return asyncio.run(provider_accounts.run_selected(self.select("api"), attempt))

    def entry(self):
        rows = self.ledger("api")
        return rows[0] if rows else None

    def run_clear(self):
        import io
        import omni_cli
        with contextlib.redirect_stdout(io.StringIO()):
            return omni_cli.main(["health", "clear", "claude", "--profile", "api"])

    def test_credit_raise_under_a_longer_rate_limit_is_recorded(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "120"
        health.flag("claude", "api", until=time.time() + 3600, source="rate_limit")
        self.raise_credit()
        self.assertEqual(self.notify.call_count, 1)
        entry = self.entry()
        self.assertEqual(entry["source"], "rate_limit")
        hold = health.credit_hold(entry)
        self.assertIsNotNone(hold)
        self.assertLess(hold["until"] - time.time(), 121)
        # A second credit failure inside the hold is not a new raise.
        self.raise_credit()
        self.assertEqual(self.notify.call_count, 1)
        # A success does not clear it.
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(self.entry())
        self.assertIsNotNone(health.credit_hold(self.entry()))
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")

    def test_cooling_continues_until_the_credit_window_ends(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "600"
        health.flag("claude", "api", until=time.time() + 60, source="rate_limit")
        self.raise_credit()
        self.offset = 120.0  # rate_limit over, credit window still open
        self.assertTrue(health.active("claude", "api"))
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")
        self.assertEqual(health.seat_kind("claude", "api", now=time.time() + 1000), "healthy")
        self.assertIsNotNone(health.credit_hold(self.entry()))
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(self.entry())
        self.assertFalse(health.snapshot()["claude@api"]["will_recheck"])
        self.offset = 700.0  # credit window over too
        self.assertEqual(health.seat_kind("claude", "api"), "healthy")
        health.maintain()
        self.assertEqual(self.ledger("api"), [])

    def test_longer_rate_limit_after_a_hold_keeps_the_hold(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "300"
        self.raise_credit()
        health.flag("claude", "api", until=time.time() + 600, source="rate_limit")
        entry = self.entry()
        self.assertEqual(entry["source"], "rate_limit")
        self.assertIsNotNone(health.credit_hold(entry))
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(health.credit_hold(self.entry()))
        self.offset = 400.0  # hold over, rate_limit still active
        self.assertIsNone(health.credit_hold(self.entry()))
        health.maintain()
        self.assertNotIn("credit_hold", self.entry())
        self.offset = 0.0
        self.assertEqual(self.run_clear(), 0)
        self.assertEqual(self.ledger("api"), [])

    def test_clear_removes_hold_and_longer_rate_limit_together(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "300"
        self.raise_credit()
        health.flag("claude", "api", until=time.time() + 600, source="rate_limit")
        self.assertEqual(self.entry()["source"], "rate_limit")
        self.assertIsNotNone(health.credit_hold(self.entry()))
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(self.entry())
        self.assertEqual(self.run_clear(), 0)
        self.assertEqual(self.ledger("api"), [])

    def test_hold_is_never_probed_half_open(self):
        os.environ["OMNI_LLM_CREDIT_COOLDOWN_S"] = "6000"
        health.flag("claude", "api", until=time.time() + 60, source="rate_limit")
        self.raise_credit()
        self.offset = 90.0  # past the rate_limit probe_after and until
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")

    def test_legacy_credit_source_without_the_field_is_still_a_hold(self):
        health.flag("claude", "api", until=time.time() + 90 * 86400, source="credit_exhausted")
        ledger = health._read_ledger()
        ledger["claude@api"].pop("credit_hold")
        health._write_ledger(ledger)
        entry = self.entry()
        self.assertNotIn("credit_hold", entry)
        self.assertIsNotNone(health.credit_hold(entry))
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(self.entry())
        self.assertEqual(health.seat_kind("claude", "api"), "cooling")
        # A longer non-credit flag converts it rather than losing it.
        health.flag("claude", "api", until=time.time() + 100 * 365 * 86400, source="rate_limit")
        entry = self.entry()
        self.assertEqual(entry["source"], "rate_limit")
        self.assertIsNotNone(health.credit_hold(entry))
        self.assertTrue(self.succeed()["ok"])
        self.assertIsNotNone(self.entry())


class TestCreditEnvelopeShapes(ApiFixture):
    """X2: the credit text is found in every genuine error envelope shape."""
    def classify(self, body, rc=1):
        health.clear_all()
        self.capture.unlink(missing_ok=True)
        health.flag(self.model, "a")
        self.respond(api={"stdout": json.dumps(body), "rc": rc})
        return self.consult()

    def test_errors_list_on_an_error_result(self):
        result = self.classify({"type": "result", "subtype": "error_during_execution",
                                "is_error": True, "errors": ["unrelated", CREDIT]})
        self.assertEqual(result["reason"], "credit_exhausted", result)
        self.assertEqual([entry["source"] for entry in self.ledger("api")], ["credit_exhausted"])

    def test_typed_error_events_without_is_error(self):
        shapes = (
            {"type": "error", "error": {"type": "billing_error", "message": CREDIT}},
            {"type": "error", "message": CREDIT},
            {"type": "error", "error": CREDIT},
            {"type": "error", "errors": [CREDIT]},
            {"type": "result", "subtype": "error_during_execution", "error": {"message": CREDIT}},
            {"error": {"type": "billing_error", "message": CREDIT}},
        )
        for body in shapes:
            with self.subTest(body=body):
                result = self.classify(body)
                self.assertEqual(result["reason"], "credit_exhausted", result)
                self.assertEqual([entry["source"] for entry in self.ledger("api")], ["credit_exhausted"])

    def test_workspace_text_in_new_shapes_is_account_unavailable(self):
        text = "This API key is not scoped to a workspace"
        for body in ({"type": "result", "is_error": True, "subtype": "error_during_execution",
                      "errors": [text]}, {"type": "error", "error": {"message": text}}):
            with self.subTest(body=body):
                self.assertEqual(self.classify(body)["reason"], "account_unavailable")

    def test_error_subtype_without_is_error_reads_errors_list(self):
        body = {"type": "result", "subtype": "error_during_execution", "errors": [CREDIT]}
        result = self.classify(body)
        self.assertEqual(result["reason"], "credit_exhausted", result)
        self.assertEqual([entry["source"] for entry in self.ledger("api")], ["credit_exhausted"])
        self.assertNotEqual(health.seat_kind(self.model, "api"), "healthy")
        workspace = {"type": "result", "subtype": "error_during_execution",
                     "errors": ["This API key is not scoped to a workspace"]}
        self.assertEqual(self.classify(workspace)["reason"], "account_unavailable")
        control = {"type": "result", "subtype": "success", "is_error": False, "errors": [CREDIT]}
        self.classify(control, rc=0)
        self.assertEqual(self.ledger("api"), [])
        self.assertEqual(health.seat_kind(self.model, "api"), "healthy")

    def test_successful_or_assistant_text_is_never_credit_evidence(self):
        controls = (
            {"type": "result", "subtype": "success", "is_error": False, "result": CREDIT},
            {"type": "result", "subtype": "success", "is_error": False, "errors": [CREDIT]},
            {"type": "result", "subtype": "success", "is_error": False,
             "error": {"message": CREDIT}, "result": "answer"},
            {"type": "result", "subtype": "success", "result": CREDIT},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": CREDIT}]}},
            {"type": "assistant", "error": CREDIT, "message": {"content": []}},
        )
        for body in controls:
            with self.subTest(body=body):
                self.classify(body, rc=0)
                self.assertEqual(self.ledger("api"), [])
        self.assertIsNone(provider_accounts.structured_failure("claude", json.dumps(controls[0])))


class TestOlderSuccessKeepsNewerFlag(ApiFixture):
    """X3: a success never erases a flag raised after its request began."""
    def select(self, profile="default"):
        return provider_accounts.AccountSelection(profile, self.model, 5, provider="claude")

    def test_flag_raised_during_the_request_survives_its_success(self):
        async def attempt(account, remaining, managed):
            # A second request observes the credit failure while this one runs.
            health.flag("claude", account["account_profile"], until=time.time() + 3600,
                        source="credit_exhausted")
            return {"ok": True, "text": "synthetic"}
        result = asyncio.run(provider_accounts.run_selected(self.select(), attempt))
        self.assertEqual(result["account_profile"], "a")
        self.assertEqual(health.seat_kind("claude", "a"), "cooling")
        self.assertEqual([entry["source"] for entry in self.ledger("a")], ["credit_exhausted"])

    def test_flag_set_before_the_request_is_cleared_by_its_success_as_before(self):
        health.flag("claude", "a", until=time.time() + 3600, source="rate_limit")
        health.flag(self.model, "a")
        time.sleep(0.01)
        async def attempt(account, remaining, managed):
            return {"ok": True, "text": "synthetic"}
        result = asyncio.run(provider_accounts.run_selected(self.select("a"), attempt))
        self.assertTrue(result["ok"])
        self.assertEqual(self.ledger("a"), [])
        self.assertEqual(health.seat_kind("claude", "a"), "healthy")


class TestStaleGateAtExecution(ApiFixture):
    """X1: the API permission is re-read from current health at execution."""
    def submit(self):
        origin = self.start_job()
        self.assertNotIn("refused", origin)
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual((data["account_profile"], data["account_kind"]), ("api", "api"))
        return origin, data

    def execute(self, origin, data):
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            return asyncio.run(job_runner._execute(origin["job_dir"], data))

    def test_subscription_recovered_before_execution_runs_on_the_subscription(self):
        health.flag(self.model, "a")
        origin, data = self.submit()
        health.clear(self.model, "a")
        result = self.execute(origin, data)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["account_profile"], result["account_kind"]), ("a", "subscription"))
        rows = self.rows()
        self.assertEqual([row["home"] for row in rows], [self.homes["a"]])
        self.assertNotIn("--settings", rows[0]["argv"])
        self.assertNotIn("--max-budget-usd", rows[0]["argv"])
        self.assertNotIn("settings", rows[0])
        state = jobs._read_job(origin["job_dir"])
        self.assertEqual((state["account_profile"], state["account_kind"], state["account_home"]),
                         ("a", "subscription", self.homes["a"]))
        self.assertEqual(state["account_tried"], [])

    def test_subscription_still_cooling_runs_on_the_api_account(self):
        health.flag(self.model, "a")
        origin, data = self.submit()
        result = self.execute(origin, data)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["account_profile"], result["account_kind"]), ("api", "api"))
        self.assertEqual([row["home"] for row in self.rows()], [self.homes["api"]])
        self.assertIn("--settings", self.rows()[0]["argv"])
        state = jobs._read_job(origin["job_dir"])
        self.assertEqual((state["account_profile"], state["account_kind"]), ("api", "api"))
        self.assertEqual(state["account_tried"], [{"account_profile": "a", "reason": "skipped_cooldown",
                                                   "account_kind": "subscription"}])

    def test_gate_closed_by_another_reason_refuses_without_spawning(self):
        health.flag(self.model, "a")
        origin, data = self.submit()
        health.clear(self.model, "a")
        os.environ["CLAUDE_CONFIG_DIR_A"] = str(self.root / "missing")
        result = self.execute(origin, data)
        self.assertEqual(result["reason"], "account_exhausted", result)
        self.assertEqual(result["tried"], [
            {"account_profile": "a", "reason": "account_unavailable", "account_kind": "subscription"},
            {"account_profile": "api", "reason": "api_not_permitted", "account_kind": "api"}])
        self.assertFalse(self.capture.exists(), "API child was spawned after the gate closed")


class TestBudgetBackstop(ApiFixture):
    def test_t6_api_children_carry_budget_and_subscription_children_do_not(self):
        health.flag(self.model, "a")
        for lane in (self.consult, self.agent):
            self.assertTrue(lane()["ok"])
        argvs = [row["argv"] for row in self.rows()]
        for argv in argvs:
            self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "1.00")
        os.environ["OMNI_LLM_API_MAX_BUDGET_USD"] = "2.5"
        self.assertTrue(self.consult()["ok"])
        argv = self.rows()[-1]["argv"]
        self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "2.5")
        health.clear_all()
        self.capture.unlink()
        for lane in (self.consult, self.agent):
            self.assertTrue(lane()["ok"])
        for row in self.rows():
            self.assertEqual(row["home"], self.homes["a"])
            self.assertNotIn("--max-budget-usd", row["argv"])

    def test_t6_bad_budget_is_profile_config_and_never_spawns(self):
        health.flag(self.model, "a")
        for value in ("abc", "0", "0.0", "-1", "1e3", "", "nan", "1.2.3"):
            with self.subTest(value=value):
                os.environ["OMNI_LLM_API_MAX_BUDGET_USD"] = value
                for lane in (self.consult, self.agent):
                    result = lane(account_profile="api")
                    self.assertEqual(result["reason"], "profile_config", result)
                    self.assertIn("OMNI_LLM_API_MAX_BUDGET_USD", result["detail"])
        self.assertFalse(self.capture.exists())

    def test_t6_bad_budget_does_not_touch_subscription_children(self):
        os.environ["OMNI_LLM_API_MAX_BUDGET_USD"] = "abc"
        self.assertTrue(self.consult()["ok"])

    def test_t6_job_and_resume_api_children_carry_budget(self):
        health.flag(self.model, "a")
        origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        argv = self.rows()[0]["argv"]
        self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "1.00")


class TestJobLaneKeyDelivery(ApiFixture):
    def test_t7_job_spawn_passes_settings_file_with_only_the_helper(self):
        health.flag(self.model, "a")
        origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual((data["account_profile"], data["account_kind"]), ("api", "api"))
        row = self.rows()[0]
        argv = row["argv"]
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project")
        settings = row["settings"]
        self.assertEqual(argv[argv.index("--settings") + 1], settings["path"])
        self.assertEqual(os.path.dirname(os.path.dirname(settings["path"])), str(self.private_base))
        self.assertFalse(os.path.exists(settings["path"]))
        self.assertEqual(settings["mode"], 0o600)
        self.assertEqual(json.loads(settings["content"]), {"apiKeyHelper": HELPER})
        self.assertEqual(row["home"], self.homes["api"])

    def test_t7_sync_agent_uses_private_temp_dir_removed_after_run(self):
        health.flag(self.model, "a")
        self.assertTrue(self.agent()["ok"])
        settings = self.rows()[0]["settings"]
        self.assertEqual(settings["mode"], 0o600)
        self.assertEqual(json.loads(settings["content"]), {"apiKeyHelper": HELPER})
        self.assertFalse(os.path.exists(settings["path"]))
        self.assertFalse(os.path.exists(os.path.dirname(settings["path"])))

    def test_t7_subscription_spawn_gets_no_settings_flag(self):
        _origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        row = self.rows()[0]
        self.assertEqual(row["home"], self.homes["a"])
        self.assertNotIn("--settings", row["argv"])
        self.assertNotIn("settings", row)
        self.assertIn("--setting-sources", row["argv"])

    def test_t7_missing_or_unsafe_helper_refuses_before_spawn(self):
        settings = Path(self.homes["api"], "settings.json")
        bad = ({}, {"apiKeyHelper": ""}, {"apiKeyHelper": "  "}, {"apiKeyHelper": ["x"]},
               {"apiKeyHelper": "a\nb"}, {"apiKeyHelper": "a\x00b"}, [])
        for content in bad:
            with self.subTest(content=content):
                health.clear_all()
                health.flag(self.model, "a")
                settings.write_text(json.dumps(content))
                _origin, result = self.run_job()
                self.assertEqual(result["reason"], "account_unavailable", result)
                self.assertIn("api", result["detail"])
                self.assertNotIn(self.root.as_posix(), result["detail"])
                self.assertEqual(self.rows(), [])
        settings.unlink()
        health.clear_all()
        health.flag(self.model, "a")
        _origin, result = self.run_job()
        self.assertEqual(result["reason"], "account_unavailable", result)
        self.assertEqual(self.rows(), [])

    def test_t7_settings_file_is_admitted_for_fenced_read(self):
        health.flag(self.model, "a")
        seen = []

        def fence(provider, argv, env, workdir, read_only, requested, **kwargs):
            seen.append(kwargs)
            return argv, env, "vendor", {}, None
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_apply_kernel_fence", side_effect=fence), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            asyncio.run(job_runner._execute(origin["job_dir"], data))
        admitted = seen[0]["extra_read_paths"]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(os.path.basename(admitted[0].path), "api-settings.json")
        self.assertEqual(os.path.dirname(os.path.dirname(admitted[0].path)), str(self.private_base))
        self.assertEqual(seen[0]["api_settings"], admitted[0].path)

    def test_t8_resume_regenerates_settings_and_keeps_home(self):
        health.flag(self.model, "a")
        origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        data = jobs._read_job(origin["job_dir"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        os.environ.pop(self.ring)
        with self.job_patches():
            child = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertNotIn("refused", child)
        resumed = jobs._read_job(child["job_dir"])
        self.assertEqual((resumed["account_profile"], resumed["account_kind"]), ("api", "api"))
        self.assertEqual(resumed["account_home"], self.homes["api"])
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            out = asyncio.run(job_runner._execute(child["job_dir"], resumed))
        self.assertTrue(out["ok"], out)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        second = rows[1]
        self.assertEqual(second["home"], self.homes["api"])
        self.assertIn("-r", second["argv"])
        self.assertIn("--max-budget-usd", second["argv"])
        self.assertEqual(os.path.dirname(os.path.dirname(second["settings"]["path"])), str(self.private_base))
        self.assertNotEqual(second["settings"]["path"], rows[0]["settings"]["path"])
        self.assertEqual(second["settings"]["mode"], 0o600)
        self.assertEqual(json.loads(second["settings"]["content"]), {"apiKeyHelper": HELPER})

    def test_t8_resume_refuses_kind_mismatch(self):
        health.flag(self.model, "a")
        origin, _result = self.run_job()
        data = jobs._read_job(origin["job_dir"])
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        os.environ["OMNI_LLM_API_PROFILES_CLAUDE"] = "a"
        with self.job_patches():
            refused = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(refused.get("refused", {}).get("reason"), "account_home_drift")
        self.assertIn("kind", refused["refused"]["detail"])
        os.environ["OMNI_LLM_API_PROFILES_CLAUDE"] = "api"
        data["account_kind"] = "subscription"
        jobs._write_job(origin["job_dir"], data)
        with self.job_patches():
            refused = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(refused.get("refused", {}).get("reason"), "account_home_drift")

    def test_t8_legacy_job_without_kind_reads_as_subscription(self):
        origin, _result = self.run_job()
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual(data["account_kind"], "subscription")
        data.pop("account_kind")
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        with self.job_patches():
            child = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertNotIn("refused", child)
        data["account_profile"] = "api"
        data["account_home"] = self.homes["api"]
        jobs._write_job(origin["job_dir"], data)
        with self.job_patches():
            refused = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(refused.get("refused", {}).get("reason"), "account_home_drift")


class TestLaunchContextScrub(ApiFixture):
    def test_t9_managed_children_drop_launch_context_and_keep_preferences(self):
        for name in LAUNCH_CONTEXT:
            os.environ[name] = "synthetic-" + name
        os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "4096"
        health.flag(self.model, "a")
        for lane in (self.consult, self.agent):
            self.assertTrue(lane()["ok"])
        origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        health.clear_all()
        for lane in (self.consult, self.agent):
            self.assertTrue(lane()["ok"])
        rows = self.rows()
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["home"] for row in rows}, {self.homes["a"], self.homes["api"]})
        for row in rows:
            self.assertEqual(row["env"], {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"}, row["home"])

    def test_t9_scrub_covers_every_provider_and_is_exact(self):
        env = {name: "x" for name in LAUNCH_CONTEXT}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "4096"
        for provider in ("claude", "codex", "grok"):
            scrubbed = provider_accounts.account_env(env, provider)
            self.assertEqual(scrubbed, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"}, provider)


def _recovering_fence(model, label="a"):
    """Stand-in for fence preparation during which an earlier account recovers."""
    def fence(provider, argv, env, *args, **kwargs):
        health.clear(model, label)
        return argv, env, "vendor", {}, None
    return fence


REAL_APPLY_KERNEL_FENCE = agent_lane._apply_kernel_fence
REAL_API_SETTINGS_BASE = agent_lane._api_settings_base


class TestApiSettingsFence(ApiFixture):
    """Q2: the child can neither rewrite nor replace the generated settings."""

    real_base = staticmethod(REAL_API_SETTINGS_BASE)

    def fence_policy(self, roots=(), config_dir=None):
        wrap_kw = {"extra_read_paths": (), "extra_write_paths": (), "extra_deny_write": (),
                   "extra_post_write": tuple(("subpath", root) for root in roots),
                   "extra_deny_read": (), "extra_post_read": (), "extra_deny_link": ()}
        # Like the real policy, record the claude home it was built for.
        return {"wrap_kw": wrap_kw, "tmpdir": None, "vendor_state_writable": False, "binary": None,
                "claude_config_root": os.path.realpath(config_dir) if config_dir else None}

    def real_fence(self, seen, roots=()):
        """Real _apply_kernel_fence with only the host-level calls faked."""
        def prepare(argv, **kwargs):
            seen.update(kwargs)
            return argv, kwargs["env"], {"fence": "seatbelt-fs", "fence_detail": {}}
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", REAL_APPLY_KERNEL_FENCE))
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "available", return_value=True))
        # The workdir is the only default-style write root here; the fixture's
        # own temp tree is not one.
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "_write_allow_paths",
                                              return_value=(os.path.realpath(self.workdir),)))
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "agent_fence_policy",
                                              side_effect=lambda *a, **k: self.fence_policy(roots)))
        stack.enter_context(mock.patch.object(agent_lane, "prepare_host_fence", side_effect=prepare))
        return stack

    def test_api_spawn_policy_denies_writes_to_the_settings_file_and_its_directory(self):
        seen = {}
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), self.real_fence(seen):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))
        self.assertTrue(result["ok"], result)
        path = self.rows()[0]["settings"]["path"]
        final = seen["policy"]["wrap_kw"]["extra_deny_write_final"]
        self.assertIn(("literal", path), final)
        self.assertIn(("subpath", os.path.dirname(path)), final)
        # State dir sits under TMPDIR, yet the file landed in the private area.
        self.assertEqual(os.path.dirname(os.path.dirname(path)), str(self.private_base))

    def test_subscription_spawn_keeps_the_default_policy(self):
        seen = {}
        with self.lane_patches(binary=str(self.binary)), self.real_fence(seen):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.rows()[0]["home"], self.homes["a"])
        self.assertNotIn("policy", seen)

    def subscription_spawn(self):
        """A non-API fenced spawn, home area kept inside the fixture tree."""
        seen = {}
        self.fake_home = self.root / "fake-home"
        self.fake_home.mkdir(exist_ok=True)
        with self.lane_patches(binary=str(self.binary)), self.real_fence(seen), \
                mock.patch.object(agent_lane.kernel_sandbox, "protected_home",
                                  return_value=str(self.fake_home)):
            return asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))

    def test_non_api_fenced_spawn_sweeps_old_leftovers_where_a_base_exists(self):
        state_base = self.root / "state" / "api-settings"
        home_base = self.root / "fake-home" / ".cache" / "omni-llm" / "api-settings"
        for base in (state_base, home_base):
            base.mkdir(parents=True, mode=0o700)
            os.chmod(base, 0o700)
        old, fresh = {}, {}
        for base in (state_base, home_base):
            old[base] = base / (agent_lane.API_SETTINGS_PREFIX + "old")
            fresh[base] = base / (agent_lane.API_SETTINGS_PREFIX + "fresh")
            for directory in (old[base], fresh[base]):
                directory.mkdir(mode=0o700)
                (directory / "api-settings.json").write_text("{}")
            stale = time.time() - 25 * 3600
            os.utime(old[base], (stale, stale))
        result = self.subscription_spawn()
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.rows()[0]["home"], self.homes["a"])
        for base in (state_base, home_base):
            self.assertFalse(old[base].exists(), base)
            self.assertTrue(fresh[base].exists(), base)

    def test_non_api_fenced_spawn_never_creates_an_absent_base(self):
        result = self.subscription_spawn()
        self.assertTrue(result["ok"], result)
        self.assertFalse((self.root / "state" / "api-settings").exists())
        self.assertFalse((self.root / "fake-home" / ".cache").exists())

    def test_sweep_skips_a_base_that_does_not_validate(self):
        victim = self.root / "victim"
        victim.mkdir(mode=0o700)
        old = victim / (agent_lane.API_SETTINGS_PREFIX + "old")
        old.mkdir(mode=0o700)
        stale = time.time() - 48 * 3600
        os.utime(old, (stale, stale))
        (self.root / "state").mkdir(exist_ok=True)
        os.symlink(victim, self.root / "state" / "api-settings")
        result = self.subscription_spawn()
        self.assertTrue(result["ok"], result)
        self.assertTrue(old.exists())

    def test_settings_inside_a_composed_write_root_are_refused_before_spawn(self):
        seen = {}
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), \
                self.real_fence(seen, roots=(str(self.private_base),)):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))
        self.assertEqual(result["reason"], "fence_path_rejected", result)
        self.assertFalse(self.capture.exists())
        self.assertEqual(list(self.private_base.iterdir()), [])

    def test_no_private_area_outside_the_write_roots_refuses_before_spawn(self):
        health.flag(self.model, "a")
        with mock.patch.object(agent_lane, "_api_settings_base", return_value=None):
            result = self.agent()
        self.assertEqual(result["reason"], "fence_path_rejected", result)
        self.assertFalse(self.capture.exists())

    def test_base_moves_out_of_a_state_dir_under_the_write_roots(self):
        state = self.root / "state"
        home = self.root / "home"
        home.mkdir()
        with mock.patch.object(agent_lane, "_api_settings_base", self.real_base), \
                mock.patch.object(agent_lane.kernel_sandbox, "_write_allow_paths",
                                  return_value=(str(state),)), \
                mock.patch.object(agent_lane.kernel_sandbox, "protected_home", return_value=str(home)):
            base = agent_lane._api_settings_base()
        self.assertEqual(base, str(home / ".cache" / "omni-llm" / "api-settings"))
        self.assertEqual(os.stat(base).st_mode & 0o777, 0o700)
        self.assertFalse(state.joinpath("api-settings").exists())

    def test_base_prefers_the_state_dir_when_it_is_outside_the_write_roots(self):
        home = self.root / "home"
        home.mkdir()
        with mock.patch.object(agent_lane.kernel_sandbox, "_write_allow_paths",
                               return_value=(str(self.root / "elsewhere"),)), \
                mock.patch.object(agent_lane.kernel_sandbox, "protected_home", return_value=str(home)):
            base = self.real_base()
        self.assertEqual(base, os.path.realpath(self.root / "state" / "api-settings"))
        self.assertEqual(os.stat(base).st_mode & 0o777, 0o700)

    def spawn_with_real_base(self, roots, home):
        seen = {}
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), self.real_fence(seen, roots=roots), \
                mock.patch.object(agent_lane, "_api_settings_base", self.real_base), \
                mock.patch.object(agent_lane.kernel_sandbox, "protected_home", return_value=str(home)):
            return asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))

    def test_state_dir_inside_a_composed_only_write_root_falls_back_to_home(self):
        home = self.root / "home"
        home.mkdir()
        result = self.spawn_with_real_base((str(self.root / "state"),), home)
        self.assertTrue(result["ok"], result)
        path = self.rows()[0]["settings"]["path"]
        want = os.path.realpath(home / ".cache" / "omni-llm" / "api-settings")
        self.assertEqual(os.path.dirname(os.path.dirname(path)), want)
        self.assertFalse((self.root / "state" / "api-settings").exists())

    def test_both_candidates_inside_composed_write_roots_are_refused(self):
        home = self.root / "home"
        home.mkdir()
        result = self.spawn_with_real_base((str(self.root / "state"), str(home)), home)
        self.assertEqual(result["reason"], "fence_path_rejected", result)
        self.assertFalse(self.capture.exists())

    def test_state_dir_outside_every_composed_root_is_still_used(self):
        home = self.root / "home"
        home.mkdir()
        result = self.spawn_with_real_base((str(self.root / "elsewhere"),), home)
        self.assertTrue(result["ok"], result)
        path = self.rows()[0]["settings"]["path"]
        want = os.path.realpath(self.root / "state" / "api-settings")
        self.assertEqual(os.path.dirname(os.path.dirname(path)), want)

    def test_base_is_unavailable_when_every_candidate_is_inside_a_write_root(self):
        home = self.root / "home"
        home.mkdir()
        with mock.patch.object(agent_lane.kernel_sandbox, "_write_allow_paths",
                               return_value=(str(self.root),)), \
                mock.patch.object(agent_lane.kernel_sandbox, "protected_home", return_value=str(home)):
            self.assertIsNone(self.real_base())


    def final_check_stack(self, repo_root, home):
        """Real fence preparation (final composition) with only the OS layer faked."""
        stack = contextlib.ExitStack()
        host_fence = agent_lane.host_fence_mod
        stack.enter_context(mock.patch.object(agent_lane, "_apply_kernel_fence", REAL_APPLY_KERNEL_FENCE))
        stack.enter_context(mock.patch.object(agent_lane, "_api_settings_base", self.real_base))
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "available", return_value=True))
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "protected_home", return_value=str(home)))
        stack.enter_context(mock.patch.object(host_fence.platform, "system", return_value="Darwin"))
        stack.enter_context(mock.patch.object(host_fence.kernel_sandbox, "verify_fence", return_value=None))
        stack.enter_context(mock.patch.object(
            host_fence.kernel_sandbox, "wrap_argv",
            side_effect=lambda argv, *a, **k: list(argv)))
        stack.enter_context(mock.patch.object(host_fence.kernel_sandbox, "is_wrapped", return_value=True))
        stack.enter_context(mock.patch.object(host_fence.kernel_sandbox, "agent_fence_policy",
                                              side_effect=lambda *a, **k: self.fence_policy(config_dir=k.get("config_dir"))))
        stack.enter_context(mock.patch.object(agent_lane.kernel_sandbox, "agent_fence_policy",
                                              side_effect=lambda *a, **k: self.fence_policy(config_dir=k.get("config_dir"))))

        def write_roots(workdir, read_only, extra_write_paths=None):
            roots = [] if read_only or workdir is None else [os.path.realpath(str(workdir))]
            return tuple(roots) + tuple(getattr(extra, "path", extra) for extra in extra_write_paths or ())
        stack.enter_context(mock.patch.object(host_fence.kernel_sandbox, "_write_allow_paths",
                                              side_effect=write_roots))
        repo = ((host_fence.kernel_sandbox.admit_path(str(repo_root), "subpath", label="repo"),)
                if repo_root else ())
        stack.enter_context(mock.patch.object(host_fence, "_repo_write_roots",
                                              return_value=(repo, (), (), (), ())))
        stack.enter_context(mock.patch.object(host_fence, "repo_root_candidates",
                                              return_value=tuple(r.path for r in repo)))
        return stack

    def run_final_check(self, repo_root, home):
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), self.final_check_stack(repo_root, home):
            return asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, fence_requested=True))

    def outside_home(self):
        home = Path(tempfile.mkdtemp(prefix="omni-apiset-home-")).resolve()
        self.addCleanup(shutil.rmtree, home, True)
        return home

    def test_repository_root_added_by_the_host_fence_is_refused_at_the_final_check(self):
        # Z1: the workdir alone does not reach the state dir; the repository
        # root the host fence adds does. The selection already moves to the
        # home area, yet the other base is reachable, so the spawn is refused
        # before any child exists.
        home = self.outside_home()
        result = self.run_final_check(self.root, home)
        self.assertEqual(result["reason"], "fence_path_rejected", result)
        self.assertIn("API settings area", result["detail"])
        self.assertFalse(self.capture.exists())
        self.assertEqual(list(home.joinpath(".cache", "omni-llm", "api-settings").glob("omni_agent_api_*")), [])
        self.assertEqual(list(self.root.joinpath("state", "api-settings").glob("omni_agent_api_*")), [])

    def test_repository_root_candidates_steer_the_selection_to_the_next_base(self):
        home = self.outside_home()
        with self.final_check_stack(self.root, home):
            base = agent_lane._api_settings_base(
                [os.path.realpath(self.workdir)] + list(agent_lane.host_fence_mod.repo_root_candidates(self.workdir)))
        self.assertEqual(base, os.path.realpath(home / ".cache" / "omni-llm" / "api-settings"))

    def test_roots_that_miss_both_bases_pass_the_final_check(self):
        home = self.outside_home()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        result = self.run_final_check(elsewhere, home)
        self.assertTrue(result["ok"], result)
        path = self.rows()[0]["settings"]["path"]
        self.assertEqual(os.path.dirname(os.path.dirname(path)),
                         os.path.realpath(self.root / "state" / "api-settings"))


class TestApiSettingsBaseValidation(unittest.TestCase):
    """Z3/Z4: the base and its ancestors are trusted before use; leftovers go."""

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="omni-apiset-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def test_a_clean_chain_is_created_private_and_returned_canonical(self):
        base = agent_lane._validated_private_base(self.path("a", "b", "api-settings"))
        self.assertEqual(base, self.path("a", "b", "api-settings"))
        self.assertEqual(stat.S_IMODE(os.stat(base).st_mode), 0o700)

    def test_group_or_other_writable_ancestor_is_refused(self):
        for mode in (0o770, 0o707, 0o777 | stat.S_ISVTX):
            with self.subTest(mode=oct(mode)):
                shutil.rmtree(self.path("shared"), True)
                os.mkdir(self.path("shared"))
                os.chmod(self.path("shared"), mode)
                self.assertIsNone(agent_lane._validated_private_base(self.path("shared", "x", "api-settings")))

    def test_symlinked_ancestor_below_the_first_user_owned_component_is_refused(self):
        os.makedirs(self.path("real"))
        os.symlink(self.path("real"), self.path("link"))
        self.assertIsNone(agent_lane._validated_private_base(self.path("link", "api-settings")))
        self.assertEqual(os.listdir(self.path("real")), [])

    def test_symlinked_base_is_refused(self):
        os.makedirs(self.path("real"), mode=0o700)
        os.symlink(self.path("real"), self.path("api-settings"))
        self.assertIsNone(agent_lane._validated_private_base(self.path("api-settings")))

    def test_base_that_is_not_exactly_0700_is_refused(self):
        os.makedirs(self.path("api-settings"))
        for mode in (0o750, 0o705, 0o500, 0o755):
            with self.subTest(mode=oct(mode)):
                os.chmod(self.path("api-settings"), mode)
                self.assertIsNone(agent_lane._validated_private_base(self.path("api-settings")))
                self.assertEqual(stat.S_IMODE(os.stat(self.path("api-settings")).st_mode), mode)

    def fake_owner(self, target, uid):
        real = os.lstat

        def lstat(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            if os.path.abspath(os.fspath(path)) == target:
                fields = list(info)
                fields[stat.ST_UID] = uid
                return os.stat_result(fields)
            return info
        return mock.patch.object(agent_lane.os, "lstat", side_effect=lstat)

    def test_base_owned_by_someone_else_is_refused(self):
        os.makedirs(self.path("api-settings"), mode=0o700)
        with self.fake_owner(self.path("api-settings"), os.getuid() + 1):
            self.assertIsNone(agent_lane._validated_private_base(self.path("api-settings")))
        self.assertEqual(agent_lane._validated_private_base(self.path("api-settings")),
                         self.path("api-settings"))

    def test_ancestor_owned_by_someone_else_is_refused(self):
        os.makedirs(self.path("up", "api-settings"), mode=0o700)
        with self.fake_owner(self.path("up"), os.getuid() + 1):
            self.assertIsNone(agent_lane._validated_private_base(self.path("up", "api-settings")))

    def pinned_base(self, *parts):
        real = agent_lane._validated_private_base_ident(self.path(*(parts or ("api-settings",))))
        base = agent_lane._PinnedBase(real[0])
        base.ident = real[1]
        return base

    def test_run_directory_is_made_through_the_pinned_base_and_is_private(self):
        base = self.pinned_base()
        made, fd = agent_lane._new_run_settings_dir(base)
        try:
            self.assertTrue(os.path.basename(made).startswith(agent_lane.API_SETTINGS_PREFIX))
            self.assertEqual(os.path.dirname(made), str(base))
            self.assertEqual(stat.S_IMODE(os.lstat(made).st_mode), 0o700)
            self.assertEqual(os.fstat(fd).st_ino, os.lstat(made).st_ino)
            path, err = agent_lane._write_api_settings_in("api", self.path("nohome"), fd, made)
            self.assertIsNone(path)
            self.assertEqual(err["reason"], "account_unavailable")
        finally:
            os.close(fd)
        self.assertEqual(len({agent_lane._new_run_settings_dir(base)[0] for _ in range(3)}), 3)

    def test_settings_file_is_written_relative_to_the_pinned_directory(self):
        home = self.path("acct")
        os.mkdir(home)
        Path(home, "settings.json").write_text(json.dumps({"apiKeyHelper": "/bin/echo key"}))
        base = self.pinned_base()
        made, fd = agent_lane._new_run_settings_dir(base)
        try:
            with mock.patch.object(agent_lane.provider_accounts, "read_api_key_helper",
                                   return_value="/bin/echo key"):
                path, err = agent_lane._write_api_settings_in("api", home, fd, made)
        finally:
            os.close(fd)
        self.assertIsNone(err)
        self.assertEqual(path, os.path.join(made, "api-settings.json"))
        self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o600)
        self.assertEqual(json.loads(Path(path).read_text()), {"apiKeyHelper": "/bin/echo key"})

    def swap_base(self, name="api-settings"):
        """Replace the validated base with a different private directory."""
        base = self.path(name)
        os.rename(base, base + ".moved")
        os.mkdir(base, 0o700)
        return base

    def test_base_swapped_after_validation_is_refused_and_nothing_is_written_into_it(self):
        base = self.pinned_base()
        swapped = self.swap_base()
        self.assertIsNone(agent_lane._new_run_settings_dir(base))
        self.assertEqual(os.listdir(swapped), [])
        self.assertEqual(os.listdir(str(base) + ".moved"), [])

    def test_base_swapped_for_a_symlink_is_refused(self):
        base = self.pinned_base()
        elsewhere = self.path("elsewhere")
        os.mkdir(elsewhere, 0o700)
        os.rename(str(base), str(base) + ".moved")
        os.symlink(elsewhere, str(base))
        self.assertIsNone(agent_lane._new_run_settings_dir(base))
        self.assertEqual(os.listdir(elsewhere), [])
        self.assertEqual(os.listdir(str(base) + ".moved"), [])

    def test_base_swapped_between_the_open_and_the_mkdir_is_refused_and_cleaned(self):
        base = self.pinned_base()
        real_mkdir = os.mkdir

        def mkdir_then_swap(name, mode=0o777, *, dir_fd=None):
            real_mkdir(name, mode, dir_fd=dir_fd)
            self.swap_base()
        with mock.patch.object(agent_lane.os, "mkdir", side_effect=mkdir_then_swap):
            self.assertIsNone(agent_lane._new_run_settings_dir(base))
        self.assertEqual(os.listdir(str(base)), [])
        self.assertEqual(os.listdir(str(base) + ".moved"), [])

    def test_unpinned_base_that_is_not_private_is_refused(self):
        base = self.path("api-settings")
        os.mkdir(base, 0o700)
        os.chmod(base, 0o755)
        self.assertIsNone(agent_lane._new_run_settings_dir(base))
        self.assertEqual(os.listdir(base), [])

    def aged(self, path, seconds):
        old = time.time() - seconds
        os.utime(path, (old, old), follow_symlinks=False)

    def test_sweep_removes_old_run_directories_and_keeps_fresh_ones(self):
        base = agent_lane._validated_private_base(self.path("api-settings"))
        old = os.path.join(base, agent_lane.API_SETTINGS_PREFIX + "old")
        fresh = os.path.join(base, agent_lane.API_SETTINGS_PREFIX + "fresh")
        other = os.path.join(base, "unrelated-old")
        for directory in (old, fresh, other):
            os.mkdir(directory, 0o700)
            Path(directory, "api-settings.json").write_text("{}")
        for directory in (old, other):
            self.aged(directory, 25 * 3600)
        agent_lane._sweep_stale_api_settings(base)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(os.path.join(fresh, "api-settings.json")))
        self.assertTrue(os.path.exists(os.path.join(other, "api-settings.json")))

    def test_sweep_never_follows_a_symlink_entry(self):
        base = agent_lane._validated_private_base(self.path("api-settings"))
        victim = self.path("victim")
        os.mkdir(victim, 0o700)
        Path(victim, "keep.txt").write_text("keep")
        self.aged(victim, 48 * 3600)
        top = os.path.join(base, agent_lane.API_SETTINGS_PREFIX + "toplink")
        os.symlink(victim, top)
        inner = os.path.join(base, agent_lane.API_SETTINGS_PREFIX + "inner")
        os.mkdir(inner, 0o700)
        os.symlink(victim, os.path.join(inner, "linked-dir"))
        os.symlink(os.path.join(victim, "keep.txt"), os.path.join(inner, "linked-file"))
        self.aged(inner, 48 * 3600)
        agent_lane._sweep_stale_api_settings(base)
        self.assertEqual(Path(victim, "keep.txt").read_text(), "keep")
        self.assertTrue(os.path.islink(top))
        self.assertTrue(os.path.islink(os.path.join(inner, "linked-dir")))
        self.assertTrue(os.path.islink(os.path.join(inner, "linked-file")))

    def test_sweep_on_a_symlinked_base_does_nothing(self):
        victim = self.path("victim")
        os.mkdir(victim, 0o700)
        old = os.path.join(victim, agent_lane.API_SETTINGS_PREFIX + "old")
        os.mkdir(old)
        self.aged(old, 48 * 3600)
        os.symlink(victim, self.path("api-settings"))
        agent_lane._sweep_stale_api_settings(self.path("api-settings"))
        self.assertTrue(os.path.isdir(old))


class TestLateGateRecheck(ApiFixture):
    """S1: the permit is re-read as the last step before the child is created."""
    NOTE = "earlier subscription account recovered"

    def assert_refused_without_spawn(self, result):
        self.assertEqual(result["reason"], "api_not_permitted", result)
        self.assertIn(self.NOTE, result["detail"])
        self.assertFalse(self.capture.exists(), "API child was spawned after the gate closed")
        self.assertEqual(health.seat_kind(self.model, "a"), "healthy")

    def test_consult_refuses_when_subscription_recovers_during_preparation(self):
        from providers import claude
        health.flag(self.model, "a")

        def prepare(argv, **kwargs):
            health.clear(self.model, "a")
            return argv, kwargs["env"], {}
        with mock.patch.object(claude, "resolve_agent_binary", return_value=str(self.binary)), \
                mock.patch.object(claude, "prepare_host_fence", side_effect=prepare):
            result = asyncio.run(claude.run_claude_subprocess("synthetic task", "synthetic-model", 5))
        self.assert_refused_without_spawn(result)
        self.assertEqual(result["tried"][-1]["account_profile"], "api")

    def test_agent_refuses_when_subscription_recovers_during_preparation(self):
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=_recovering_fence(self.model)):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5))
        self.assert_refused_without_spawn(result)

    def test_job_refuses_when_subscription_recovers_during_preparation(self):
        health.flag(self.model, "a")
        origin = self.start_job()
        self.assertNotIn("refused", origin)
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual(data["account_profile"], "api")
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=_recovering_fence(self.model)), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        self.assert_refused_without_spawn(result)

    def test_job_snapshot_phase_recovery_is_caught_too(self):
        health.flag(self.model, "a")
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])

        def snapshot(*args, **kwargs):
            health.clear(self.model, "a")
            return {"state": "ok", "entries": {}, "head": None}
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_workspace_snapshot", side_effect=snapshot), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        self.assert_refused_without_spawn(result)

    def recovering_watch_pane(self):
        """Watch-pane setup (an await inside run_cli) during which "a" recovers."""
        from providers import cli_run
        stack = contextlib.ExitStack()

        async def open_pane(*args, **kwargs):
            health.clear(self.model, "a")
            await asyncio.sleep(0)
            return None
        stack.enter_context(mock.patch.object(cli_run, "watch_enabled", return_value=True))
        stack.enter_context(mock.patch.object(cli_run, "resolve_zellij_session", return_value="synthetic"))
        stack.enter_context(mock.patch.object(cli_run, "resolve_zellij_tab_id", return_value=1))
        stack.enter_context(mock.patch.object(cli_run, "_open_watch_pane", side_effect=open_pane))
        return stack

    def test_consult_refuses_when_subscription_recovers_during_watch_pane_setup(self):
        health.flag(self.model, "a")
        with self.recovering_watch_pane():
            result = self.consult()
        self.assert_refused_without_spawn(result)

    def test_agent_refuses_when_subscription_recovers_during_watch_pane_setup(self):
        health.flag(self.model, "a")
        with self.recovering_watch_pane():
            result = self.agent()
        self.assert_refused_without_spawn(result)

    def test_job_refuses_when_subscription_recovers_during_watch_pane_setup(self):
        health.flag(self.model, "a")
        origin = self.start_job()
        data = jobs._read_job(origin["job_dir"])
        with self.recovering_watch_pane(), self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(job_runner, "read_successor_attest", return_value=None):
            result = asyncio.run(job_runner._execute(origin["job_dir"], data))
        self.assert_refused_without_spawn(result)

    def test_gate_runs_after_every_await_in_run_cli(self):
        from providers import cli_run
        order = []

        async def open_pane(*args, **kwargs):
            order.append("pane")
            return None

        async def spawn(*args, **kwargs):
            order.append("spawn")
            raise AssertionError("spawned")
        with mock.patch.object(cli_run, "watch_enabled", return_value=True), \
                mock.patch.object(cli_run, "resolve_zellij_session", return_value="synthetic"), \
                mock.patch.object(cli_run, "resolve_zellij_tab_id", return_value=1), \
                mock.patch.object(cli_run, "_open_watch_pane", side_effect=open_pane), \
                mock.patch.object(cli_run, "_spawn", side_effect=spawn):
            refused = asyncio.run(cli_run.run_cli(
                ["/bin/true"], pre_spawn=lambda: order.append("gate") or {"ok": False, "reason": "api_not_permitted"}))
            broken = asyncio.run(cli_run.run_cli(["/bin/true"], pre_spawn=lambda: 1 / 0))
        self.assertEqual(order, ["pane", "gate", "pane"])  # the gate follows the pane await
        self.assertEqual(refused["reason"], "api_not_permitted")
        self.assertNotIn("returncode", refused)
        self.assertEqual(broken["reason"], "pre_spawn_error")
        self.assertNotIn("spawn", order)

    def test_other_providers_run_cli_calls_carry_no_gate(self):
        seen = []

        async def runner(argv, **kwargs):
            seen.append(kwargs)
            return {"returncode": 0, "stdout": SUCCESS, "stderr": ""}
        with self.lane_patches(runner):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, account_profile="a"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("pre_spawn", seen[0])

    def test_explicit_pin_is_unaffected_by_a_recovered_subscription(self):
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=_recovering_fence(self.model)):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5, account_profile="api"))
        self.assertTrue(result["ok"], result)
        self.assertEqual([row["home"] for row in self.rows()], [self.homes["api"]])

    def test_still_cooling_subscription_keeps_the_api_path_open(self):
        health.flag(self.model, "a")
        with self.lane_patches(binary=str(self.binary)), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=lambda p, argv, env, *a, **k: (argv, env, "vendor", {}, None)):
            result = asyncio.run(agent_lane.run_agent(
                "synthetic task", self.model, self.workdir, timeout_s=5))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["account_profile"], "api")


class TestApiOptInForPins(ApiFixture):
    """P5: an explicit pin or resume of an api label needs the opt-in too."""
    def test_pin_is_refused_before_spawn_when_opt_in_is_not_exactly_one(self):
        for value in (None, "0", "true", ""):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("OMNI_LLM_API_FALLBACK_CLAUDE", None)
                else:
                    os.environ["OMNI_LLM_API_FALLBACK_CLAUDE"] = value
                health.flag(self.model, "a")
                for lane in (self.consult, self.agent):
                    result = lane(account_profile="api")
                    self.assertEqual(result["reason"], "api_disabled", result)
                    self.assertIn("OMNI_LLM_API_FALLBACK_CLAUDE", result["detail"])
                origin = self.start_job(account_profile="api")
                self.assertEqual(origin["refused"]["reason"], "api_disabled", origin)
        self.assertFalse(self.capture.exists(), "API child was spawned without the opt-in")

    def test_pin_with_no_ring_configured_is_refused_too(self):
        os.environ.pop("OMNI_LLM_API_FALLBACK_CLAUDE")
        os.environ.pop(self.ring)
        for lane in (self.consult, self.agent):
            self.assertEqual(lane(account_profile="api")["reason"], "api_disabled")
        self.assertFalse(self.capture.exists())

    def test_resume_of_an_api_job_is_refused_after_the_opt_in_is_withdrawn(self):
        health.flag(self.model, "a")
        origin, result = self.run_job()
        self.assertTrue(result["ok"], result)
        data = jobs._read_job(origin["job_dir"])
        self.assertEqual(data["account_kind"], "api")
        data.update(state="failed", session_id="synthetic-parent-session")
        jobs._write_job(origin["job_dir"], data)
        os.environ.pop("OMNI_LLM_API_FALLBACK_CLAUDE")
        with self.job_patches():
            refused = jobs.resume(origin["job_dir"], "synthetic continuation", allow_drift=True)
        self.assertEqual(refused.get("refused", {}).get("reason"), "api_disabled", refused)
        self.assertEqual(len(self.rows()), 1)

    def test_subscription_pin_never_needs_the_opt_in(self):
        os.environ.pop("OMNI_LLM_API_FALLBACK_CLAUDE")
        for lane in (self.consult, self.agent):
            self.assertTrue(lane(account_profile="a")["ok"])


class TestHelperRead(ApiFixture):
    """P3: the parent reads apiKeyHelper only from a regular file in the home."""
    def write_settings(self, text, home=None):
        Path(home or self.homes["api"], "settings.json").write_text(text)

    def settings_for(self, helper):
        self.write_settings(json.dumps({"apiKeyHelper": helper}))
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        path, err = provider_accounts.write_api_settings("api", self.homes["api"], str(out))
        return path, err, out

    def test_normal_helpers_are_accepted(self):
        for helper in (HELPER, "security find-generic-password -s omni -w", "~/bin/print-key",
                       "./helper.sh --label x", "pass"):
            with self.subTest(helper=helper):
                path, err, _out = self.settings_for(helper)
                self.assertIsNone(err)
                self.assertEqual(json.loads(Path(path).read_text()), {"apiKeyHelper": helper})

    def test_key_shaped_or_non_command_helpers_are_refused_and_never_copied(self):
        bad = ("printf %s sk-ant-xxxxxxxxxxxxxxxxxxxx", "echo sk-" + "a" * 20,
               "echo xai-" + "b" * 20, "curl -H 'Authorization: Bearer abc.def'",
               "echo eyJhbGciOi.eyJzdWIiOi.c2ln", "-flag first", "'quoted' cmd", "$(cmd)",
               "; ls", " leading", "")
        for helper in bad:
            with self.subTest(helper=helper):
                path, err, out = self.settings_for(helper)
                self.assertIsNone(path)
                self.assertEqual(err["reason"], "account_unavailable")
                self.assertIn("api", err["detail"])
                self.assertNotIn("sk-", err["detail"])
                self.assertEqual(list(out.iterdir()), [])

    def test_symlinked_settings_file_is_refused_and_nothing_is_written(self):
        elsewhere = self.root / "elsewhere.json"
        elsewhere.write_text(json.dumps({"apiKeyHelper": HELPER}))
        target = Path(self.homes["api"], "settings.json")
        target.unlink()
        target.symlink_to(elsewhere)
        out = self.root / "out"
        out.mkdir()
        path, err = provider_accounts.write_api_settings("api", self.homes["api"], str(out))
        self.assertIsNone(path)
        self.assertEqual(err["reason"], "account_unavailable")
        self.assertEqual(list(out.iterdir()), [])
        health.flag(self.model, "a")
        origin, result = self.run_job()
        # The job path may be stopped first by the host fence's protected-name
        # alias refusal; either refusal must spawn nothing and write nothing.
        self.assertIn(result["reason"], ("account_unavailable", "fence_path_rejected"), result)
        self.assertFalse(Path(origin["job_dir"], "api-settings.json").exists())
        self.assertFalse(self.capture.exists())

    def test_settings_reached_through_a_symlinked_home_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.write_settings(json.dumps({"apiKeyHelper": HELPER}), home=outside)
        link = self.root / "linkhome"
        link.symlink_to(outside)
        out = self.root / "out"
        out.mkdir()
        parent = self.root / "parentlink"
        parent.symlink_to(self.root)
        for home in (str(link), str(parent / "outside")):
            with self.subTest(home=home):
                path, err = provider_accounts.write_api_settings("api", home, str(out))
                self.assertIsNone(path)
                self.assertEqual(err["reason"], "account_unavailable")
        self.assertEqual(list(out.iterdir()), [])

    def test_non_regular_settings_file_is_refused_without_blocking(self):
        target = Path(self.homes["api"], "settings.json")
        target.unlink()
        os.mkfifo(target)
        out = self.root / "out"
        out.mkdir()
        path, err = provider_accounts.write_api_settings("api", self.homes["api"], str(out))
        self.assertIsNone(path)
        self.assertEqual(err["reason"], "account_unavailable")

    def test_oversized_settings_file_is_refused(self):
        self.write_settings(json.dumps({"apiKeyHelper": HELPER, "pad": "x" * 70000}))
        self.assertIsNone(provider_accounts.read_api_key_helper(self.homes["api"]))


class ApiEnvDoesNotChangeOtherProviders:
    """T10: the same Codex and Grok contract suites, with Claude API env present."""
    def setUp(self):
        ProviderContract.setUp(self)
        patch = mock.patch.dict(os.environ, {
            "OMNI_LLM_API_PROFILES_CLAUDE": "b", "OMNI_LLM_API_FALLBACK_CLAUDE": "1",
            "OMNI_LLM_API_MAX_BUDGET_USD": "9.99", "OMNI_LLM_CREDIT_COOLDOWN_S": "60"})
        patch.start()
        self.addCleanup(patch.stop)


for _provider in ("codex", "grok"):
    _name = "Test%sAccountsWithApiEnv" % _provider.title()
    globals()[_name] = type(_name, (ApiEnvDoesNotChangeOtherProviders, ProviderContract,
                                    unittest.TestCase), {"provider": _provider})


if __name__ == "__main__":
    unittest.main()
