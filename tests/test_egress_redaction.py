#!/usr/bin/env python3
# no-opencc
"""Issue #155: failure text is masked once at egress, and before the cut.

Vendor output (stderr/stdout, an in-band ``result``, a signal-exit tail) is
cut to 300 characters for an envelope's ``detail``. The cut used to run
before the secret mask, so a key straddling character 300 left a prefix no
pattern matched; several providers never masked at all. These tests drive
each provider through the real tool entry point (server) and the real CLI
with ``run_cli`` mocked, and require that no synthetic key, nor any
8-character fragment of it, reaches the envelope or the CLI output.

Hermetic: no vendor binary is run, nothing leaves the process. Every key is
built at runtime; no key-shaped literal is stored in this file.
"""
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import omni_cli
import server
from providers import (agent_lane, agy, claude, cursor, dispatch, errors,
                       grok, grok_image_cli, health, host_fence, jobs,
                       job_runner, kernel_sandbox, pi, provider_accounts)

# Runtime-built so the file itself never holds a key-shaped string.
BODY = "Qz8Wm3Nv" * 4
KEY = "s" + "k-" + BODY
OAUTH = "1/" + "/" + "0gQz8Wm3NvXy" * 3
FRAGMENT = 8
PADS = (0, 150, 255, 260, 285, 290, 295)


def leaked_fragment(blob, secret, size=FRAGMENT):
    """The first ``size``-char slice of ``secret`` found in ``blob``, else None."""
    for i in range(len(secret) - size + 1):
        if secret[i:i + size] in blob:
            return secret[i:i + size]
    return None


class Env(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="omni-155-")
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_LLM_STATE_DIR": self.root,
               "OMNI_LLM_QUOTA_PROBE": "0", "OMNI_LLM_NOTIFY_QUIET": "1"}
        for patch in (mock.patch.dict(os.environ, env, clear=True),
                      mock.patch.object(health, "_account_identity", return_value=None),
                      mock.patch.object(host_fence.platform, "system", return_value="Linux")):
            patch.start()
            self.addCleanup(patch.stop)


def capture_server(model, args=None, tool="llm_generate"):
    """Run one tool call through server.handle_tool_call; return the sent line."""
    sent = []

    async def fake_send(obj):
        sent.append(obj)

    call = {"prompt": "synthetic", "model": model, "timeout_s": 30}
    call.update(args or {})
    quiet = io.StringIO()
    with mock.patch.object(server, "_send", fake_send), \
            contextlib.redirect_stderr(quiet):
        asyncio.run(server.handle_tool_call(1, tool, call))
    return sent[0]["result"]


def capture_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = omni_cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class ProviderCase(Env):
    """One provider, one failure path, every key position."""

    PROVIDERS = {
        "agy": (agy, "agy/synthetic"),
        "claude": (claude, "claude/synthetic"),
        "pi": (pi, "pi/synthetic"),
        "cursor": (cursor, "cursor/synthetic"),
        "grok": (grok, "grok/synthetic"),
    }

    def patches(self, mod, res):
        async def fake_run_cli(argv, **kwargs):
            return dict(res)

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(mod, "run_cli", fake_run_cli))
        if hasattr(mod, "resolve_bin"):
            stack.enter_context(mock.patch.object(
                mod, "resolve_bin", return_value="/synthetic/bin"))
        if hasattr(mod, "shutil"):
            stack.enter_context(mock.patch.object(
                mod.shutil, "which", return_value="/synthetic/bin"))
        return stack

    def check(self, provider, make_res, label, expect_mask_up_to=200,
              require_mask=True):
        mod, model = self.PROVIDERS[provider]
        for pad in PADS:
            res = make_res("x" * pad + KEY)
            with self.subTest(provider=provider, path=label, pad=pad):
                with self.patches(mod, res):
                    result = capture_server(model)
                    code, cli_out, cli_err = capture_cli(
                        ["gen", "-m", model, "--timeout", "30", "synthetic"])
                    _, plain_out, plain_err = capture_cli(
                        ["gen", "-m", model, "--timeout", "30", "--plain", "synthetic"])
                structured = result["structuredContent"]
                self.assertIs(structured.get("ok"), False, structured)
                self.assertTrue(structured.get("detail"), structured)
                blob = json.dumps(result, ensure_ascii=False)
                for where, text in (("mcp envelope", blob), ("cli stdout", cli_out),
                                    ("cli stderr", cli_err), ("cli --plain stdout", plain_out),
                                    ("cli --plain stderr", plain_err)):
                    self.assertIsNone(leaked_fragment(text, KEY),
                                      "%s leaks %r" % (where, leaked_fragment(text, KEY)))
                if require_mask and pad <= expect_mask_up_to:
                    # Not vacuous: the detail really carries the masked key.
                    self.assertIn("<redacted>", structured["detail"], structured)
                self.assertEqual(code, 1)
                self.assertTrue(json.loads(cli_out)["detail"])


def nonzero(channel):
    def make(text):
        return {"returncode": 1, "stdout": text if channel == "stdout" else "",
                "stderr": text if channel == "stderr" else "", "timed_out": False}
    return make


def signal_exit(channel):
    def make(text):
        return {"returncode": None, "signal": 9,
                "stdout": text if channel == "stdout" else "",
                "stderr": text if channel == "stderr" else "", "timed_out": False}
    return make


def negative_rc(channel):
    def make(text):
        return {"returncode": -9, "stdout": text if channel == "stdout" else "",
                "stderr": text if channel == "stderr" else "", "timed_out": False}
    return make


def empty_reply(text):
    return {"returncode": 0, "stdout": "", "stderr": text, "timed_out": False}


def inband(field, extra=None):
    def make(text):
        doc = {"type": "result", "subtype": "success", "is_error": True, field: text}
        doc.update(extra or {})
        return {"returncode": 0, "stdout": json.dumps(doc), "stderr": "",
                "timed_out": False}
    return make


class TestNonzeroExit(ProviderCase):
    def test_every_provider_stderr(self):
        for provider in self.PROVIDERS:
            self.check(provider, nonzero("stderr"), "nonzero exit, stderr")

    def test_every_provider_stdout(self):
        for provider in ("agy", "claude", "pi", "cursor"):
            self.check(provider, nonzero("stdout"), "nonzero exit, stdout")


class TestSignalExit(ProviderCase):
    def test_signal_field_stderr_and_stdout(self):
        for provider in self.PROVIDERS:
            self.check(provider, signal_exit("stderr"), "signal exit, stderr")
        for provider in ("agy", "claude", "pi", "cursor"):
            self.check(provider, signal_exit("stdout"), "signal exit, stdout")

    def test_negative_returncode(self):
        for provider in self.PROVIDERS:
            self.check(provider, negative_rc("stderr"), "negative rc, stderr")


class TestEmptyReply(ProviderCase):
    def test_every_provider(self):
        for provider in ("agy", "claude", "pi", "cursor", "grok"):
            # grok withholds an unverified stream before any stderr is quoted.
            self.check(provider, empty_reply, "empty reply",
                       require_mask=provider != "grok")


class TestInBandError(ProviderCase):
    def test_claude_and_cursor_result_text(self):
        for provider in ("claude", "cursor"):
            self.check(provider, inband("result"), "in-band is_error, exit 0")

    def test_claude_and_cursor_result_text_with_http_status(self):
        for provider in ("claude", "cursor"):
            self.check(provider, inband("result", {"api_error_status": 401}),
                       "in-band is_error 401")

    def test_claude_and_cursor_nonzero_exit(self):
        def make(text):
            res = inband("result")(text)
            res["returncode"] = 1
            return res
        for provider in ("claude", "cursor"):
            self.check(provider, make, "in-band is_error, nonzero exit")

    def test_agy_status_text(self):
        def make(text):
            return {"returncode": 0, "stdout": json.dumps(
                {"status": text, "response": ""}), "stderr": "", "timed_out": False}
        self.check("agy", make, "agy in-band status")


class TestListModels(Env):
    def run_models(self, rc, out, err):
        class Proc:
            returncode = rc

            async def communicate(self):
                return out.encode(), err.encode()

        async def fake_exec(*argv, **kwargs):
            return Proc()

        with mock.patch.object(dispatch.shutil, "which", return_value="/synthetic/agy"), \
                mock.patch.object(dispatch.asyncio, "create_subprocess_exec", fake_exec):
            return capture_server("m", tool="list_models", args={"timeout_s": 5}), \
                capture_cli(["models"])

    def test_nonzero_exit_and_empty_reply(self):
        for pad in PADS:
            for label, rc, out, err in (
                    ("stderr", 1, "", "x" * pad + KEY),
                    ("stdout", 1, "x" * pad + KEY, ""),
                    ("empty", 0, "", "x" * pad + KEY)):
                with self.subTest(path=label, pad=pad):
                    result, (_code, cli_out, cli_err) = self.run_models(rc, out, err)
                    self.assertIs(result["structuredContent"].get("ok"), False)
                    for text in (json.dumps(result), cli_out, cli_err):
                        self.assertIsNone(leaked_fragment(text, KEY))


class TestGrokImage(Env):
    def run_image(self, rc, out, err):
        proc = types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)
        src = os.path.join(self.root, "src.png")
        with open(src, "wb") as fh:
            fh.write(b"\x89PNG")
        with mock.patch.object(grok_image_cli.shutil, "which", return_value="/synthetic/grok"), \
                mock.patch.object(grok_image_cli, "prepare_host_fence",
                                  side_effect=lambda argv, **kw: (argv, dict(os.environ), {})), \
                mock.patch.object(grok_image_cli.subprocess, "run", return_value=proc):
            return asyncio.run(grok_image_cli.run_grok_image_edit(
                "synthetic", src, 5, out_dir=self.root))

    def test_failure_detail(self):
        for pad in PADS:
            for label, rc, out, err in (
                    ("nonzero stderr", 1, "", "x" * pad + KEY),
                    ("nonzero stdout", 1, "x" * pad + KEY, ""),
                    ("no image", 0, "x" * pad + KEY, "")):
                with self.subTest(path=label, pad=pad):
                    res = self.run_image(rc, out, err)
                    self.assertIs(res.get("ok"), False, res)
                    self.assertIsNone(leaked_fragment(json.dumps(res), KEY), res)


class TestAgentLane(Env):
    """The agent lane's non-codex branch, through the CLI's own entry."""

    def lane(self, provider, res, read_only=True):
        async def fake_run_cli(argv, **kwargs):
            return dict(res)

        model = "%s/synthetic" % provider
        argv = ["agent", "--workdir", self.root, "-m", model, "--timeout", "5"]
        if read_only:
            argv.append("--read-only")
        with mock.patch.object(agent_lane, "run_cli", fake_run_cli), \
                mock.patch.object(agent_lane.shutil, "which", return_value="/synthetic/bin"), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane, "_workspace_snapshot", return_value={}), \
                mock.patch.object(agent_lane, "_workspace_writes", return_value=(0, None)), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value={"ok": True, "vendor_state_writable": True}), \
                mock.patch.object(kernel_sandbox, "wrap_kwargs_from_policy",
                                  return_value={"read_wall": True}), \
                mock.patch.object(kernel_sandbox, "available", return_value=False):
            return capture_cli(argv + ["synthetic task"])

    def assert_clean(self, provider, make_res, label, read_only=True):
        for pad in PADS:
            with self.subTest(provider=provider, path=label, pad=pad):
                code, out, err = self.lane(provider, make_res("x" * pad + KEY), read_only)
                self.assertEqual(code, 1, (out, err))
                self.assertTrue(json.loads(out).get("detail"), out)
                self.assertIsNone(leaked_fragment(out, KEY))
                self.assertIsNone(leaked_fragment(err, KEY))

    def test_nonzero_exit(self):
        for provider in ("claude", "pi", "cursor"):
            self.assert_clean(provider, nonzero("stderr"), "nonzero stderr")
            self.assert_clean(provider, nonzero("stdout"), "nonzero stdout")
        self.assert_clean("grok", nonzero("stderr"), "nonzero stderr", read_only=False)

    def test_signal_exit(self):
        for provider in ("claude", "pi", "cursor"):
            self.assert_clean(provider, signal_exit("stderr"), "signal exit")
        self.assert_clean("grok", signal_exit("stderr"), "signal exit", read_only=False)

    def test_empty_reply(self):
        for provider in ("claude", "pi", "cursor"):
            self.assert_clean(provider, empty_reply, "empty reply")

    def test_in_band_error(self):
        self.assert_clean("claude", inband("result"), "in-band error")
        self.assert_clean("cursor", inband("result"), "in-band error")


class TestSharedHelpers(unittest.TestCase):
    def test_signal_exit_detail_masks_before_the_cut(self):
        for pad in PADS:
            for stream in ("stdout", "stderr"):
                with self.subTest(pad=pad, stream=stream):
                    text = "x" * pad + KEY
                    detail = errors.signal_exit_detail(
                        "claude", -9, text if stream == "stdout" else "",
                        text if stream == "stderr" else "")
                    self.assertLessEqual(len(detail), 300)
                    self.assertIsNone(leaked_fragment(detail, KEY), detail)

    def test_safe_detail_masks_then_cuts(self):
        for pad in PADS:
            with self.subTest(pad=pad):
                detail = errors.safe_detail("x" * pad + KEY)
                self.assertLessEqual(len(detail), 300)
                self.assertIsNone(leaked_fragment(detail, KEY), detail)

    def test_safe_detail_limit_and_non_text(self):
        self.assertEqual(errors.safe_detail(None), "")
        self.assertEqual(errors.safe_detail(12345), "12345")
        self.assertEqual(len(errors.safe_detail("y" * 500, 40)), 40)

    def test_exception_detail_output_is_unchanged(self):
        exc = RuntimeError("boom " + KEY + " reach me at who@example.test")
        self.assertEqual(errors.exception_detail(exc),
                         "RuntimeError: boom <redacted> reach me at who@example.test")

    def test_inband_result_text_is_masked_before_the_cut(self):
        from providers.usage import inband_error
        for pad in PADS:
            with self.subTest(pad=pad):
                doc = json.dumps({"type": "result", "is_error": True,
                                  "result": "x" * pad + KEY})
                reason, detail = inband_error("cursor", doc)
                self.assertLessEqual(len(detail), 300)
                self.assertIsNone(leaked_fragment(detail, KEY), detail)


class TestKeyShapes(unittest.TestCase):
    def assert_masked(self, text, secret, keep=()):
        out = errors.redact_secrets(text)
        self.assertNotIn(secret, out, out)
        self.assertIsNone(leaked_fragment(out, secret), out)
        self.assertIn("<redacted>", out)
        for fragment in keep:
            self.assertIn(fragment, out, out)
        return out

    def test_oauth_refresh_token_bare(self):
        self.assert_masked("refresh failed for token " + OAUTH + " today", OAUTH,
                           keep=("refresh failed for token", "today"))

    def test_oauth_refresh_token_json_value(self):
        value = "opaque" + BODY
        out = self.assert_masked(
            '{"refresh_token": "%s", "scope": "openid"}' % value, value,
            keep=('"refresh_token": "', '"scope": "openid"'))
        self.assertTrue(out.startswith('{"refresh_token": "<redacted>"'), out)

    def test_oauth_refresh_token_json_value_escaped_in_a_string(self):
        value = "opaque" + BODY
        text = json.dumps({"detail": '{"refresh_token": "%s"}' % value})
        self.assert_masked(text, value, keep=("refresh_token",))

    def test_oauth_tokens_in_a_query_string(self):
        value = "opaque" + BODY
        text = "GET /cb?code=1&refresh_token=%s&state=2 failed" % value
        out = self.assert_masked(text, value, keep=("refresh_token=", "&state=2 failed"))
        self.assertIn("refresh_token=<redacted>&state=2", out)
        self.assert_masked("access_token=%s" % value, value)
        self.assert_masked("OPENAI_API_KEY=%s" % value, value)

    def test_unprefixed_api_key_json(self):
        value = "plain" + BODY
        out = self.assert_masked('{"api_key": "%s", "model": "m"}' % value, value,
                                 keep=('"model": "m"',))
        self.assertIn('"api_key": "<redacted>"', out)
        self.assert_masked('{"api_key":"%s"}' % value, value)
        self.assert_masked('{"apiKey": "%s"}' % value, value)

    def test_google_api_key(self):
        secret = "AIza" + "Sy" + BODY + "Ab"
        self.assert_masked("request to ?key=" + secret + " refused", secret)

    def test_google_access_token(self):
        secret = "ya29." + "a0Af" + BODY
        self.assert_masked("Authorization failed for " + secret, secret)

    def test_groq_key(self):
        secret = "gsk_" + BODY
        self.assert_masked("groq said no to " + secret, secret)

    def test_github_tokens(self):
        for prefix in ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"):
            with self.subTest(prefix=prefix):
                secret = prefix + BODY
                self.assert_masked("git push with " + secret + " denied", secret)
        secret = "github_pat_" + "11ABCDE0Y" + BODY
        self.assert_masked("token " + secret, secret)

    def test_huggingface_token(self):
        secret = "hf_" + BODY
        self.assert_masked("download denied for " + secret, secret)

    def test_existing_shapes_still_masked(self):
        for secret in (KEY, "s" + "k-ant-" + BODY, "xai-" + BODY):
            with self.subTest(secret=secret[:6]):
                self.assert_masked("key " + secret + " end", secret)
        self.assert_masked("x" + KEY, KEY)
        out = errors.redact_secrets("Bearer " + BODY + "." + BODY)
        self.assertEqual(out, "Bearer <redacted>")

    def test_hyphenated_words_are_not_keys(self):
        for text in ("desk-notes-archive-2026", "task-runner-config-file",
                     "/Users/someone/desk-notes-archive-2026/readme.md",
                     "risk-assessment-for-q3",
                     "disk-usage-report-2026-10-10", "risk-assessment-2026Q3",
                     "/tmp/task-configuration2026Q3.log", "xai-grok-code-fast-1",
                     "python@3.12", "@scope/pkg@2.0.14",
                     "da39a3ee5e6b4b0d3255bfef95601890afd80709",
                     "123e4567-e89b-12d3-a456-426614174000"):
            with self.subTest(text=text):
                self.assertEqual(errors.mask_text(text), text)

    def test_ordinary_failure_text_is_unchanged(self):
        text = ("error: Request failed with status code 429 key=value "
                "api_key not set; refresh_token expired")
        self.assertEqual(errors.redact_secrets(text), text)


class TestMaskEnvelope(Env):
    def leaky(self):
        text = "x" * 285 + KEY
        return {
            "ok": False, "reason": "runtime_error", "detail": text,
            "error": {"message": text, "code": 7},
            "attempts": [{"model": "a", "detail": text}, {"model": "b", "error": text}],
            "backends": {"agy": {"detail": text}},
            "fallback": [{"message": text}],
            "metrics_error": text, "warning": text,
        }

    def test_masks_every_failure_text_field_and_leaves_the_rest(self):
        env = self.leaky()
        out = errors.mask_envelope(env)
        self.assertIsNone(leaked_fragment(json.dumps(out), KEY), out)
        self.assertEqual(out["error"]["code"], 7)
        self.assertEqual(out["attempts"][0]["model"], "a")
        self.assertIn(KEY, env["detail"], "the input is not mutated")

    def test_successful_text_is_never_masked(self):
        answer = "the config line is " + KEY + " as asked"
        env = {"ok": True, "text": answer, "model": "m",
               "nested": {"text": answer}, "images": [{"path": "p"}]}
        self.assertEqual(errors.mask_envelope(env), env)

    def test_server_exit_masks_whatever_the_dispatcher_returned(self):
        async def leaky_call(name, args):
            return self.leaky()

        with mock.patch.object(server, "_call_tool", leaky_call):
            result = capture_server("m")
        self.assertIsNone(leaked_fragment(json.dumps(result), KEY), result)
        self.assertEqual(result["structuredContent"]["error"]["code"], 7)

    def test_server_exit_leaves_the_answer_alone(self):
        answer = "line " + KEY

        async def ok_call(name, args):
            return {"ok": True, "text": answer, "model": "m"}

        with mock.patch.object(server, "_call_tool", ok_call):
            result = capture_server("m")
        self.assertEqual(result["structuredContent"]["text"], answer)
        self.assertEqual(json.loads(result["content"][0]["text"])["text"], answer)

    def test_cli_exit_masks_json_and_plain(self):
        async def leaky_call(name, args):
            return self.leaky()

        with mock.patch.object(omni_cli, "call_tool", leaky_call):
            code, out, err = capture_cli(["gen", "-m", "m", "synthetic"])
            self.assertEqual(code, 1)
            self.assertIsNone(leaked_fragment(out + err, KEY), out)
            _, out, err = capture_cli(["gen", "-m", "m", "--plain", "synthetic"])
            self.assertIsNone(leaked_fragment(out + err, KEY), err)

    def test_cli_exit_leaves_the_answer_alone(self):
        answer = "line " + KEY

        async def ok_call(name, args):
            return {"ok": True, "text": answer, "model": "m"}

        with mock.patch.object(omni_cli, "call_tool", ok_call):
            _, out, _ = capture_cli(["gen", "-m", "m", "synthetic"])
            self.assertEqual(json.loads(out)["text"], answer)
            _, out, _ = capture_cli(["gen", "-m", "m", "--plain", "synthetic"])
            self.assertEqual(out, answer + "\n")


class TestJobResultFile(Env):
    def test_result_json_is_masked_and_the_winner_is_the_callers_dict(self):
        job_dir = os.path.join(self.root, "job")
        os.makedirs(job_dir)
        text = "x" * 285 + KEY
        result = {"ok": False, "reason": "runtime_error", "detail": text,
                  "attempts": [{"detail": text}], "error": {"message": text}}
        winner = jobs._write_result(job_dir, result)
        self.assertIs(winner, result)
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as fh:
            raw = fh.read()
        self.assertIsNone(leaked_fragment(raw, KEY), raw)
        self.assertIsNone(leaked_fragment(json.dumps(winner), KEY))
        self.assertIsNone(leaked_fragment(json.dumps(jobs._read_result(job_dir)), KEY))

    def test_runner_finalize_goes_through_the_same_door(self):
        job_dir = os.path.join(self.root, "job2")
        os.makedirs(job_dir)
        jobs._write_job(job_dir, {"state": "running"})
        job_runner._finalize(job_dir, {"ok": False, "reason": "runtime_error",
                                       "detail": "x" * 288 + KEY})
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as fh:
            self.assertIsNone(leaked_fragment(fh.read(), KEY))

    def test_a_successful_result_text_is_written_exact(self):
        job_dir = os.path.join(self.root, "job3")
        os.makedirs(job_dir)
        answer = "keep " + KEY
        jobs._write_result(job_dir, {"ok": True, "text": answer})
        self.assertEqual(jobs._read_result(job_dir)["text"], answer)

    def test_stop_reason_is_masked_before_the_cut(self):
        job_dir = os.path.join(self.root, "job4")
        os.makedirs(job_dir)
        jobs._write_job(job_dir, {"state": "running"})
        with mock.patch.object(jobs, "_lock_is_held", return_value=False), \
                mock.patch.object(jobs, "_reap_backend_if_live"):
            view = jobs._repair_failed(job_dir, {"state": "running"},
                                       "x" * 288 + KEY)
        self.assertIsNone(leaked_fragment(json.dumps(view), KEY), view)
        self.assertIsNone(leaked_fragment(json.dumps(jobs._read_result(job_dir)), KEY))


class TestManagedFailureDetail(Env):
    PREFIX = "claude account failed: runtime_error: "

    def selection(self, home):
        return types.SimpleNamespace(provider="claude", homes={}, names=[],
                                     home_key="account_home"), {"account_home": home}

    def detail(self, inner, home="/synthetic/profiles/claude-work-account"):
        selection, account = self.selection(home)
        return provider_accounts.managed_failure_detail(
            selection, account, {"reason": "runtime_error", "detail": inner})

    def test_email_straddling_the_cut_leaves_no_fragment(self):
        email = "jordan.maplewood" + "@" + "example-mail.test"
        for pad in range(240, 262):
            with self.subTest(pad=pad):
                inner = "x" * pad + " " + email + " tail"
                out = self.detail(inner)
                self.assertLessEqual(len(out), 300)
                self.assertIsNone(leaked_fragment(out, email), out)

    def test_home_path_straddling_the_cut_leaves_no_fragment(self):
        home = "/synthetic/profiles/claude-work-account"
        for pad in range(230, 262):
            with self.subTest(pad=pad):
                out = self.detail("x" * pad + " " + home + "/config tail", home)
                self.assertLessEqual(len(out), 300)
                self.assertIsNone(leaked_fragment(out, home), out)

    def test_key_straddling_the_cut_leaves_no_fragment(self):
        for pad in range(240, 262):
            with self.subTest(pad=pad):
                out = self.detail("x" * pad + " " + KEY + " tail")
                self.assertIsNone(leaked_fragment(out, KEY), out)

    def test_a_provider_cut_through_an_email_or_home_leaves_no_fragment(self):
        """The provider cuts its vendor text first (safe_detail); what it hands
        over is already shortened, and a half-written home or address must
        still not survive the managed layer."""
        email = "jordan.maplewood" + "@" + "example-mail.test"
        home = "/synthetic/profiles/claude-work-account"
        for pad in range(200, 300):
            for tail in (email, home + "/config"):
                with self.subTest(pad=pad, tail=tail[:8]):
                    cut = errors.safe_detail("x" * pad + " " + tail + " more")
                    out = self.detail(cut, home)
                    for secret in (email, home):
                        self.assertIsNone(leaked_fragment(out, secret), out)

    def test_plain_text_is_kept(self):
        out = self.detail("exit 3: model unavailable")
        self.assertEqual(out, self.PREFIX + "exit 3: model unavailable")


# ---------------------------------------------------------------------------
# Fix round 1: real key shapes in glued contexts, nested shapes, opaque forms,
# linear time, and the `agent tail` log view.
# ---------------------------------------------------------------------------
def real_key_shapes():
    """Real-format keys, built at runtime. name -> secret."""
    mixed = "Qz8Wm3NvXy2Ab" * 3
    return {
        "sk-ant": "sk" + "-ant-api03-" + mixed + "_" + mixed,
        "sk-proj": "sk" + "-proj-" + mixed + "-" + mixed,
        "sk-svcacct": "sk" + "-svcacct-" + mixed + "_" + mixed,
        "sk-generic": "sk" + "-" + mixed,
        "ghp": "gh" + "p_" + mixed,
        "github_pat": "github" + "_pat_11" + mixed + "_" + mixed,
        "AIza": "AI" + "za" + "Sy" + mixed[:33],
        "ya29": "ya" + "29." + mixed + "-" + mixed,
        "1//": "1" + "//0g" + mixed + "-" + mixed,
        "hf": "hf" + "_" + mixed,
        "gsk": "gs" + "k_" + mixed,
        "xai": "xa" + "i-" + mixed,
        "sk-lowercase-hex": "sk" + "-" + "deadbeef" * 4,
        "sk-uuid": "sk" + "-lf-" + "123e4567-e89b-12d3-a456-426614174000",
    }


GLUE_CONTEXTS = (
    ("json newline escape", "\\n"),
    ("ansi code", "\x1b[1m"),
    ("percent-encoded equals", "%3D"),
    ("letter", "a"),
    ("space", " "),
)


class TestRealKeyShapes(ProviderCase):
    def test_every_shape_in_every_context_through_server_and_cli(self):
        mod, model = self.PROVIDERS["agy"]
        for name, secret in real_key_shapes().items():
            for label, glue in GLUE_CONTEXTS:
                for pad in (0, 255, 290):
                    res = {"returncode": 1, "stdout": "x" * pad + glue + secret,
                           "stderr": "", "timed_out": False}
                    with self.subTest(shape=name, context=label, pad=pad):
                        with self.patches(mod, res):
                            result = capture_server(model)
                            _, cli_out, cli_err = capture_cli(
                                ["gen", "-m", model, "--timeout", "30", "synthetic"])
                        self.assertTrue(result["structuredContent"].get("detail"))
                        for where, text in (("mcp", json.dumps(result)),
                                            ("cli stdout", cli_out), ("cli stderr", cli_err)):
                            self.assertIsNone(leaked_fragment(text, secret),
                                              "%s leaks %r" % (where, leaked_fragment(text, secret)))

    def test_glued_whole_body_is_masked_not_only_a_prefix(self):
        body = "Qz8Wm3NvXy2Ab" * 2
        for text in ("prefixsk-" + body + "_SUFFIXTAIL",
                     "prefixsk-proj-" + body + "-SUFFIXTAIL",
                     "prefixsk-ant-api03-" + body + "_SUFFIXTAIL"):
            with self.subTest(text=text[:20]):
                out = errors.mask_text(text)
                self.assertNotIn("SUFFIXTAIL", out)
                self.assertIsNone(leaked_fragment(out, body))

    def test_a_key_after_a_word_run_is_judged_on_its_own(self):
        secret = "sk" + "-" + "Qz8Wm3NvXy2Ab" * 2
        out = errors.mask_text("task-notes-" + secret)
        self.assertIsNone(leaked_fragment(out, secret), out)


class TestMaskEnvelopeShapes(Env):
    def test_lists_tuples_and_text_inside_failure_fields(self):
        for env in ({"error": [KEY]}, {"detail": (KEY,)}, {"error": {"text": KEY}},
                    {"detail": [{"text": KEY}, [KEY]]},
                    {"attempts": [{"error": [KEY]}]},
                    {"error": {"content": KEY, "preview": KEY}}):
            with self.subTest(env=sorted(env)):
                out = errors.mask_envelope(env)
                self.assertIsNone(leaked_fragment(json.dumps(out), KEY), out)

    def test_nesting_past_the_depth_limit_is_masked_not_returned_raw(self):
        for depth in (20, 40, 80):
            node = [KEY]
            for _ in range(depth):
                node = {"inner": node}
            for env in ({"error": node}, {"outer": node}):
                with self.subTest(depth=depth, key=sorted(env)[0]):
                    out = errors.mask_envelope(
                        env if "error" in env else {"error": env})
                    self.assertIsNone(leaked_fragment(json.dumps(out), KEY))

    def test_successful_top_level_text_is_unchanged(self):
        env = {"ok": True, "text": "answer " + KEY, "model": "m"}
        self.assertEqual(errors.mask_envelope(env), env)

    def test_email_is_masked_in_failure_fields_at_egress(self):
        email = "jordan.maplewood" + "@" + "example-mail.test"
        out = errors.mask_envelope({"ok": False, "detail": "sent to " + email,
                                    "error": ["a " + email]})
        self.assertNotIn("maplewood", json.dumps(out))

    def test_server_and_cli_exits_mask_nested_shapes(self):
        shapes = {"error": [KEY], "detail": (KEY,), "message": {"text": KEY}}

        async def leaky(name, args):
            return dict({"ok": False, "reason": "runtime_error"}, **shapes)

        with mock.patch.object(server, "_call_tool", leaky):
            result = capture_server("m")
        self.assertIsNone(leaked_fragment(json.dumps(result), KEY))
        with mock.patch.object(omni_cli, "call_tool", leaky):
            _, out, err = capture_cli(["gen", "-m", "m", "synthetic"])
        self.assertIsNone(leaked_fragment(out + err, KEY))

    def test_job_result_file_masks_nested_shapes(self):
        job_dir = os.path.join(self.root, "nested")
        os.makedirs(job_dir)
        jobs._write_result(job_dir, {"ok": False, "error": [KEY],
                                     "detail": (KEY,), "message": {"text": KEY}})
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as fh:
            self.assertIsNone(leaked_fragment(fh.read(), KEY))


class TestOpaqueForms(ProviderCase):
    VALUE = "Opq" + BODY + "/zz"

    def assert_value_masked(self, text, keep=()):
        out = errors.mask_text(text)
        self.assertIsNone(leaked_fragment(out, self.VALUE), out)
        self.assertIn("<redacted>", out)
        for fragment in keep:
            self.assertIn(fragment, out, out)

    def test_single_quoted_python_repr(self):
        self.assert_value_masked("{'api_key': '%s', 'n': 1}" % self.VALUE, keep=("'n': 1",))
        self.assert_value_masked("access_token='%s' next" % self.VALUE, keep=(" next",))

    def test_json_values_with_escapes(self):
        for value in (self.VALUE.replace("/", "\\/"), self.VALUE + "\\u002fTAIL" + BODY,
                      self.VALUE + '\\"inner\\"' + BODY):
            with self.subTest(value=value[-12:]):
                out = errors.mask_text('{"refresh_token": "%s", "scope": "openid"}' % value)
                self.assertIsNone(leaked_fragment(out, BODY), out)
                self.assertIn('"scope": "openid"', out)

    def test_camel_case_names(self):
        for name in ("accessToken", "refreshToken", "idToken", "clientSecret", "apiKey"):
            with self.subTest(name=name):
                self.assert_value_masked('{"%s": "%s"}' % (name, self.VALUE), keep=(name,))
                self.assert_value_masked("%s=%s&z=1" % (name, self.VALUE), keep=("&z=1",))

    def test_header_forms(self):
        self.assert_value_masked("x-api-key: " + self.VALUE, keep=("x-api-key:",))
        self.assert_value_masked("Authorization: Basic " + BODY, keep=("Authorization: Basic",))
        out = errors.mask_text("Authorization: Bearer " + BODY)
        self.assertIsNone(leaked_fragment(out, BODY))

    def test_agy_dict_status_end_to_end(self):
        status = {"state": "FAILED", "api_key": self.VALUE}
        res = {"returncode": 0, "stdout": json.dumps({"status": status, "response": ""}),
               "stderr": "", "timed_out": False}
        with self.patches(agy, res):
            result = capture_server("agy/synthetic")
            _, out, err = capture_cli(
                ["gen", "-m", "agy/synthetic", "--timeout", "30", "synthetic"])
        for text in (json.dumps(result), out, err):
            self.assertIsNone(leaked_fragment(text, self.VALUE), text)
        self.assertTrue(result["structuredContent"]["detail"].startswith("agy status="))


class TestLinearTime(unittest.TestCase):
    """Each case runs in a child interpreter with a hard timeout, so a regex
    that backtracks quadratically fails the test instead of hanging the suite.
    The child imports omni's own modules only; no vendor binary is involved."""
    MIB = 1 << 20
    LIMIT = 0.5
    HARD_STOP = 20

    def elapsed(self, setup, op):
        import subprocess
        import time
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = ("import sys, time; sys.path.insert(0, %r)\n"
                "from providers import errors, provider_accounts\n"
                "import types\n%s"
                "t = time.monotonic()\n%s"
                "print(time.monotonic() - t)\n" % (root, setup, op))
        proc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                              text=True, timeout=self.HARD_STOP)
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
        return float(proc.stdout.strip().splitlines()[-1])

    def test_safe_detail_and_mask_envelope_on_adversarial_runs(self):
        for unit in ("a", "eyJ", "a.", "a@", "sk-", "ask-", "key=", "a-"):
            setup = "big = %r * (%d // %d)\n" % (unit, self.MIB, len(unit))
            op = ("errors.safe_detail(big)\n"
                  "errors.mask_envelope({'detail': big, 'error': [big]})\n")
            with self.subTest(unit=unit):
                self.assertLess(self.elapsed(setup, op), self.LIMIT)

    def test_managed_failure_detail_on_a_long_run(self):
        setup = ("sel = types.SimpleNamespace(provider='claude', homes={}, names=[], "
                 "home_key='account_home')\n")
        op = ("provider_accounts.managed_failure_detail(sel, "
              "{'account_home': '/synthetic/h'}, "
              "{'reason': 'runtime_error', 'detail': 'a' * %d})\n" % self.MIB)
        self.assertLess(self.elapsed(setup, op), self.LIMIT)


class TestAgentTail(Env):
    def make_job(self, lines):
        job_dir = os.path.join(self.root, "tailjob")
        os.makedirs(job_dir)
        jobs._write_job(job_dir, {"state": "running"})
        path = os.path.join(job_dir, "stream.log")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(lines)
        return job_dir, path

    def event(self, secret):
        return json.dumps({"type": "result", "is_error": True,
                           "result": "failed with " + secret}) + "\n"

    def test_one_shot_tail_masks_a_failure_event(self):
        job_dir, _ = self.make_job(self.event(KEY))
        code, out, err = capture_cli(["agent", "tail", job_dir])
        self.assertEqual(code, 0)
        self.assertIn("is_error", out)
        self.assertIsNone(leaked_fragment(out + err, KEY), out)

    def test_follow_masks_seed_and_appended_lines_even_when_split(self):
        job_dir, path = self.make_job(self.event(KEY))
        line = self.event(KEY)
        cut = line.index(KEY) + 14
        steps = [line[:cut], line[cut:]]

        def fake_sleep(_seconds):
            if steps:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(steps.pop(0))
                return
            raise KeyboardInterrupt

        with mock.patch.object(omni_cli.time, "sleep", fake_sleep):
            code, out, err = capture_cli(["agent", "tail", job_dir, "--follow"])
        self.assertEqual(code, 0)
        self.assertEqual(out.count("is_error"), 2, out)
        self.assertIsNone(leaked_fragment(out + err, KEY), out)


# ---------------------------------------------------------------------------
# Fix round 2: helper acceptance, depth rule, tail seed and bounds, long and
# multiply-escaped values, JWT, case, short-segment keys, e-mail shapes.
# ---------------------------------------------------------------------------
class TestHelperAcceptance(Env):
    def test_commands_that_merely_name_a_secret_are_accepted(self):
        for helper in ('curl -s -H "x-api-key: $VAULT_TOKEN" https://vault.example/v1/key',
                       "fetch-key --api_key=prod"):
            with self.subTest(helper=helper):
                self.assertTrue(provider_accounts._helper_acceptable(helper), helper)

    def test_commands_that_embed_a_key_are_refused(self):
        body = "Qz8Wm3NvXy2Ab" * 2
        for helper in ("echo " + "sk" + "-ant-api03-" + body, "echo " + "sk" + "-" + body,
                       "echo " + "sk" + "-proj-" + body, "echo gh" + "p_" + body,
                       "echo AI" + "za" + "Sy" + (body * 2)[:33], "echo 1" + "//0g" + body,
                       "echo sk" + "-lf-123e4567-e89b-12d3-a456-426614174000",
                       "echo sk" + "-None-" + body * 2,
                       "echo sk" + "-or-v1-" + "deadbeef" * 8,
                       # 16-23 characters with separators, and a body that holds
                       # an inner "sk-": refused, as on main.
                       "echo sk" + "-Xq7_Lm2Pz9-Rt4Kw8Vn3Bc",
                       "echo sk" + "-0123456789abcdef",
                       "echo sk" + "-AbCdEfGhIjKlMnOpQrsk-TuVwXyZ0123456789",
                       # refused, as on main (the earlier pattern matched them)
                       "pass show clients/sk-ant-production",
                       "/usr/local/bin/sk-keychain-helper-claude",
                       "curl -H 'Authorization: bEaReR abc.def'"):
            with self.subTest(helper=helper[:24]):
                self.assertFalse(provider_accounts._helper_acceptable(helper), helper)

    def test_refusal_detail_says_why(self):
        label, err = "work", None
        with mock.patch.object(provider_accounts, "read_api_key_helper", return_value=None):
            _path, err = provider_accounts.write_api_settings(label, self.root, self.root)
        self.assertIn("key-shaped literal", err["detail"])
        self.assertNotIn("sk-", err["detail"])


class TestDepthRule(unittest.TestCase):
    def test_successful_content_past_the_depth_limit_is_unchanged(self):
        node = {"text": "answer " + KEY}
        for _ in range(33):
            node = {"inner": node}
        env = {"ok": True, "result": node}
        self.assertEqual(errors.mask_envelope(env), env)

    def test_failure_subtree_past_the_depth_limit_is_masked(self):
        node = [KEY]
        for _ in range(40):
            node = {"inner": node}
        out = errors.mask_envelope({"ok": False, "error": node})
        self.assertIsNone(leaked_fragment(json.dumps(out), KEY))


class TestTailSeedAndBounds(Env):
    def job(self, content):
        job_dir = os.path.join(self.root, "tailjob")
        os.makedirs(job_dir, exist_ok=True)
        jobs._write_job(job_dir, {"state": "running"})
        path = os.path.join(job_dir, "stream.log")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return job_dir, path

    def follow(self, job_dir, path, appends):
        steps = list(appends)

        def fake_sleep(_seconds):
            if steps:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(steps.pop(0))
                return
            raise KeyboardInterrupt

        with mock.patch.object(omni_cli.time, "sleep", fake_sleep):
            return capture_cli(["agent", "tail", job_dir, "--follow"])

    def test_a_key_split_across_the_seed_and_the_follow(self):
        shapes = {name: secret for name, secret in real_key_shapes().items()
                  if name in ("sk-generic", "sk-ant", "sk-proj", "ghp")}
        for name, secret in shapes.items():
            for keep in (6, 12, 20):
                line = '{"type": "result", "is_error": true, "result": "x ' + secret + '"}\n'
                cut = line.index(secret) + keep
                with self.subTest(shape=name, keep=keep):
                    job_dir, path = self.job(line[:cut])
                    code, out, err = self.follow(job_dir, path, [line[cut:]])
                    self.assertEqual(code, 0)
                    self.assertIsNone(leaked_fragment(out + err, secret), out)

    def test_one_shot_tail_does_not_print_an_unfinished_key(self):
        secret = "sk" + "-proj-" + "Qz8Wm3NvXy2Ab" * 2
        for keep in (6, 12, 20):
            content = '{"result": "first"}\n{"result": "x ' + secret[:keep]
            with self.subTest(keep=keep):
                job_dir, _path = self.job(content)
                code, out, _err = capture_cli(["agent", "tail", job_dir])
                self.assertEqual(code, 0)
                self.assertIn('"first"', out)
                self.assertIsNone(leaked_fragment(out, secret), out)

    def test_an_unterminated_line_is_held_in_bounded_memory(self):
        captured = io.StringIO()
        pending = ""
        peak = 0
        with contextlib.redirect_stdout(captured):
            for _ in range(8):
                pending = omni_cli._tail_feed("a" * (1 << 20), pending)
                peak = max(peak, len(pending))
        self.assertLessEqual(peak, omni_cli._TAIL_PENDING_MAX)
        self.assertGreater(len(captured.getvalue()), 7 << 20)

    def test_a_long_unterminated_line_through_the_cli_stays_bounded(self):
        job_dir, path = self.job("")
        peaks = []
        real_feed = omni_cli._tail_feed

        def watching_feed(text, pending):
            pending = real_feed(text, pending)
            peaks.append(len(pending))
            return pending

        with mock.patch.object(omni_cli, "_tail_feed", watching_feed):
            code, out, _err = self.follow(job_dir, path, ["b" * (3 << 20)])
        self.assertEqual(code, 0)
        self.assertLessEqual(max(peaks), omni_cli._TAIL_PENDING_MAX)
        self.assertGreaterEqual(out.count("b"), (3 << 20) - omni_cli._TAIL_PENDING_KEEP)

    def test_truncation_resets_what_is_held(self):
        job_dir, path = self.job("first line\n")
        calls = []

        def fake_sleep(_seconds):
            calls.append(1)
            if len(calls) == 1:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("partial-without-newline")
            elif len(calls) == 2:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("fresh\n")
            else:
                raise KeyboardInterrupt

        with mock.patch.object(omni_cli.time, "sleep", fake_sleep):
            code, out, _err = capture_cli(["agent", "tail", job_dir, "--follow"])
        self.assertEqual(code, 0)
        self.assertIn("fresh\n", out)
        self.assertNotIn("partial-without-newlinefresh", out)


class TestLongAndEscapedValues(unittest.TestCase):
    LONG = "Tk9" + "Qz8Wm3NvXy2Ab" * 185          # 2,400-ish characters

    def check(self, text, keep=()):
        out = errors.mask_text(text)
        self.assertIsNone(leaked_fragment(out, self.LONG), out[:200])
        for fragment in keep:
            self.assertIn(fragment, out)

    def test_quoted_unquoted_and_colon_forms(self):
        self.assertGreater(len(self.LONG), 2300)
        self.check('{"refresh_token": "%s", "scope": "openid"}' % self.LONG,
                   keep=('"scope": "openid"',))
        self.check("refresh_token=%s&state=1" % self.LONG, keep=("&state=1",))
        self.check("refresh_token: %s next" % self.LONG, keep=(" next",))
        self.check("refresh_token%%3D%s%%26x" % self.LONG)
        self.check("x-api-key: %s" % self.LONG)

    def test_json_dumped_up_to_three_times(self):
        doc = {"refresh_token": self.LONG, "scope": "openid"}
        text = json.dumps(doc)
        for _ in range(3):
            self.check(text)
            text = json.dumps(text)
        self.check(text)

    def test_a_value_with_escapes_and_no_closing_quote(self):
        self.check('\\"refresh_token\\": \\"%s' % self.LONG)
        self.check('{"refresh_token": "a\\"b\\\\c%s' % self.LONG)

    def test_name_glued_after_a_letter_or_digit(self):
        value = "Opq" + BODY
        for glue in ("\\n", "\x1b[0m", "%26", "\\u0026", "9", "x"):
            with self.subTest(glue=glue):
                out = errors.mask_text(glue + "refresh_token=" + value)
                self.assertIsNone(leaked_fragment(out, value), out)
                out = errors.mask_text(glue + "api_key=" + value)
                self.assertIsNone(leaked_fragment(out, value), out)

    def test_many_backslashes_before_the_quote(self):
        value = "Opq" + BODY
        for slashes in ("\\", "\\\\", "\\\\\\", "\\\\\\\\\\\\\\"):
            with self.subTest(n=len(slashes)):
                text = '{%s"refresh_token%s": %s"%s%s"}' % (slashes, slashes, slashes,
                                                              value, slashes)
                out = errors.mask_text(text)
                self.assertIsNone(leaked_fragment(out, value), out)


class TestJwt(unittest.TestCase):
    def jwt(self, payload_chars=5576):
        header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = "eyJ" + ("Qz8Wm3NvXy2Ab" * (payload_chars // 13 + 1))[:payload_chars]
        signature = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        return header + "." + payload + "." + signature

    def test_glued_and_plain_with_a_long_payload(self):
        token = self.jwt()
        self.assertGreater(len(token), 5576)
        for label, glue in (("start", ""), ("space", " "), ("backslash n", "\\n"),
                            ("after m", "m"), ("ansi", "\x1b[0m"), ("percent", "%3D"),
                            ("underscore", "token_")):
            with self.subTest(context=label):
                out = errors.mask_text("x" + glue + token + " tail" if glue else token)
                self.assertIsNone(leaked_fragment(out, token), out[:200])
                self.assertTrue(out.endswith("tail") or "<redacted>" in out)

    def test_through_server_and_cli(self):
        token = self.jwt()
        res = {"returncode": 1, "stdout": "log\\n" + token, "stderr": "",
               "timed_out": False}

        class Case(ProviderCase):
            def runTest(self):
                pass
        case = Case()
        case.setUp()
        self.addCleanup(case.doCleanups)
        with case.patches(agy, res):
            result = capture_server("agy/synthetic")
            _, out, err = capture_cli(
                ["gen", "-m", "agy/synthetic", "--timeout", "30", "synthetic"])
        for text in (json.dumps(result), out, err):
            self.assertIsNone(leaked_fragment(text, token), text[:200])

    def test_words_that_contain_eyj_are_left_alone(self):
        for text in ("keyJson.data.value", "see eyJ.notajwt", "v1.2.3", "a.b.c"):
            self.assertEqual(errors.mask_text(text), text)


class TestCaseAndShapes(unittest.TestCase):
    def test_bearer_in_any_case_and_glued(self):
        token = "Opq" + BODY
        for scheme in ("Bearer", "bEaReR", "BEARER", "bearer"):
            for glue in ("", "\\n", "x"):
                with self.subTest(scheme=scheme, glue=glue):
                    out = errors.mask_text("Authorization: " + glue + scheme + " " + token)
                    self.assertIsNone(leaked_fragment(out, token), out)

    def test_basic_in_any_case_and_glued(self):
        token = "Opq" + BODY
        out = errors.mask_text("\\nAUTHORIZATION: bAsIc " + token)
        self.assertIsNone(leaked_fragment(out, token), out)

    def test_lowercase_hex_and_uuid_keys_glued_to_a_word(self):
        keys = ("sk" + "-" + "deadbeef" * 4,
                "sk" + "-lf-" + "123e4567-e89b-12d3-a456-426614174000",
                "sk" + "-None-" + "Qz8Wm3NvXy2Ab" * 3)
        for secret in keys:
            for glue in ("a", "\\n", "\x1b[1m", "%3D", "9"):
                with self.subTest(secret=secret[:9], glue=glue):
                    out = errors.mask_text("x" + glue + secret + " end")
                    self.assertIsNone(leaked_fragment(out, secret), out)

    def test_email_with_many_labels_and_punycode(self):
        for address in ("alice" + "@" + "a.b.c.d.e.f.g.h.i.j.k.l.m.example.com",
                        "alice" + "@" + "example.xn--p1ai",
                        "alice" + "@" + "xn--80ak6aa92e.example.org"):
            with self.subTest(address=address[-16:]):
                out = errors.mask_text("sent to " + address + " today")
                self.assertEqual(out, "sent to <email> today")


class TestLinearTimeRound2(unittest.TestCase):
    """Every pattern on 1 MiB adversarial inputs, in a child interpreter."""
    MIB = TestLinearTime.MIB
    LIMIT = TestLinearTime.LIMIT
    HARD_STOP = TestLinearTime.HARD_STOP
    elapsed = TestLinearTime.elapsed

    def test_patterns_on_adversarial_runs(self):
        units = ("\\\\", '"', "\\\"api_key\\\":\\\"", "api_key=", "api_key:", "api_key=\"",
                 "a.b.", "eyJ.", "eyJa.", "sk-a-", "sk-ant-", "Bearer ", "bearer\t",
                 "a@b.", "a@b.c.", "xn--", "1//", "%3D", "token ", "refresh_token=\\\"")
        for unit in units:
            setup = "big = %r * (%d // %d)\n" % (unit, self.MIB, len(unit))
            op = ("errors.safe_detail(big)\n"
                  "errors.mask_envelope({'detail': big, 'error': [big]})\n"
                  "errors.contains_key_literal(big)\n")
            with self.subTest(unit=unit):
                self.assertLess(self.elapsed(setup, op), self.LIMIT)

    def test_long_values_and_one_giant_token(self):
        for setup in ("big = 'refresh_token=' + 'a' * %d\n" % self.MIB,
                      "big = '{\"refresh_token\": \"' + 'a' * %d\n" % self.MIB,
                      "big = 'eyJ' + 'a' * %d + '.' + 'b' * 100 + '.' + 'c' * 100\n" % self.MIB,
                      "big = 'sk-' + 'a-' * %d\n" % (self.MIB // 2)):
            with self.subTest(setup=setup[:30]):
                self.assertLess(
                    self.elapsed(setup, "errors.safe_detail(big)\nerrors.mask_text(big)\n"),
                    self.LIMIT)


if __name__ == "__main__":
    unittest.main()
