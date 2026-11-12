#!/usr/bin/env python3
# no-opencc
"""Acceptance tests — content-free request metrics + attempt tracing (issue #22 P1.1b).

Pinned contract:
  * one JSONL line per consult call under $OMNI_LLM_STATE_DIR/metrics.jsonl;
  * no prompt/output CONTENT — only operational sizes, seats, usage envelopes;
  * envelope usage = winning seat only (losing attempt usage never on envelope);
  * attempts = real provider invocations incl. rate-limit retries, EXCLUDING
    skipped_cooldown; fallback_count = seats called after the first real seat;
  * OMNI_LLM_METRICS=0 disables writes; write failure → metrics_error, result ok;
  * O_APPEND single-write, dir 0700 / file 0600, rotate at >5 MiB → .1.
"""
import asyncio
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-metrics-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
# Ensure metrics are on for this process unless a test opts out.
os.environ.pop("OMNI_LLM_METRICS", None)

from providers import dispatch  # noqa: E402
from providers import health  # noqa: E402
from providers import metrics as metrics_mod  # noqa: E402


def tearDownModule():
    """R8: clean the module-level mkdtemp state dir."""
    import shutil
    try:
        shutil.rmtree(_TMP, ignore_errors=True)
    except Exception:
        pass


def _run(coro):
    return asyncio.run(coro)


def _envelope(ok, reason=None, text="hi", model="m", usage=None):
    res = {"ok": ok, "text": text if ok else "", "model": model}
    if reason:
        res["reason"] = reason
    if usage is not None:
        res["usage"] = usage
    return res


def _usage(input_tokens):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": None,
        "cache_write_tokens": None,
        "output_tokens": 1,
        "reasoning_tokens": None,
        "total_tokens": None,
        "source": "provider",
        "raw": {"input_tokens": input_tokens},
    }


def _read_lines(path=None):
    path = path or metrics_mod.metrics_path()
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(ln) for ln in handle if ln.strip()]


def _clear_metrics_file():
    path = metrics_mod.metrics_path()
    for candidate in (path, path + ".1"):
        try:
            os.unlink(candidate)
        except OSError:
            pass


class MetricsWriteTest(unittest.TestCase):
    def setUp(self):
        health.clear_all()
        _clear_metrics_file()
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        os.environ.pop("OMNI_LLM_METRICS", None)

    def test_success_writes_one_record(self):
        # Long distinctive tokens so any truncated-prefix leak (prompt[:N], N>=4) reds.
        secret_prompt = "SECRETTOK" * 40
        secret_output = "OUTMARKXYZ" * 40

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=secret_output, model=model,
                             usage=_usage(7))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": secret_prompt, "model": "FakeModel",
            }))
        self.assertTrue(res["ok"])
        self.assertNotIn("metrics_error", res)
        lines = _read_lines()
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["tool"], "llm_generate")
        self.assertEqual(rec["entry"], "mcp")
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["request_bytes"], len(secret_prompt.encode("utf-8")))
        self.assertEqual(rec["result_bytes"], len(secret_output.encode("utf-8")))
        self.assertEqual(rec["usage"]["input_tokens"], 7)
        self.assertEqual(rec["seat"]["model"], "FakeModel")
        self.assertEqual(rec["fallback_count"], 0)
        self.assertEqual(len(rec["attempts"]), 1)
        # Content-free: distinctive prompt/output must appear NOWHERE in the file.
        path = metrics_mod.metrics_path()
        whole = open(path, encoding="utf-8").read()
        self.assertNotIn(secret_prompt, whole)
        self.assertNotIn(secret_output, whole)
        # Prefix of length 4+ must also be absent (truncated leak net).
        self.assertNotIn(secret_prompt[:4], whole)
        self.assertNotIn(secret_output[:4], whole)

    def test_failure_writes_record_with_reason(self):
        async def router(prompt, model, *a, **k):
            return _envelope(False, reason="runtime_error", model=model)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "x", "model": "M",
            }))
        self.assertFalse(res["ok"])
        rec = _read_lines()[0]
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["reason"], "runtime_error")
        self.assertEqual(len(rec["attempts"]), 1)
        self.assertEqual(rec["attempts"][0]["reason"], "runtime_error")

    def test_managed_failure_metrics_record_attempted_account(self):
        async def router(prompt, model, *a, **k):
            res = _envelope(False, reason="runtime_error", model=model)
            res["account_profile"] = "api"
            return res

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            _run(dispatch.call_tool("llm_generate", {
                "prompt": "x", "model": "M", "account_profile": "default",
            }))
        self.assertEqual(_read_lines()[-1]["seat"]["account_profile"], "api")

    def test_account_exhausted_metrics_fall_back_to_call_profile(self):
        async def router(prompt, model, *a, **k):
            return _envelope(False, reason="account_exhausted", model=model)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            _run(dispatch.call_tool("llm_generate", {
                "prompt": "x", "model": "M", "account_profile": "work",
            }))
        self.assertEqual(_read_lines()[-1]["seat"]["account_profile"], "work")

    def test_fallback_chain_attempts_and_skipped_cooldown(self):
        health.flag("A", "default", source="test")

        async def router(prompt, model, *a, **k):
            if model == "A":
                return _envelope(False, reason="backend_error", model="A",
                                 usage=_usage(111))
            return _envelope(True, model="B", usage=_usage(3))

        with mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
             mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "A,B", "account_profile": "default",
            }))
        self.assertTrue(res["ok"])
        # Envelope tried should still show skipped_cooldown for A, then maybe not
        # re-call A if B succeeded — cooling A is skipped, B wins.
        tried = res.get("tried") or []
        skipped = [t for t in tried
                   if t.get("model") == "A" and "cooldown" in str(t.get("reason", ""))]
        self.assertTrue(skipped, "expected skipped_cooldown in envelope tried")

        rec = _read_lines()[0]
        # Only real provider calls — skipped_cooldown excluded.
        models = [a["model"] for a in rec["attempts"]]
        self.assertEqual(models, ["B"])
        self.assertEqual(rec["fallback_count"], 0)
        self.assertEqual(rec["usage"]["input_tokens"], 3)

    def test_fallback_count_after_first_real_seat(self):
        async def router(prompt, model, *a, **k):
            if model == "A":
                return _envelope(False, reason="runtime_error", model="A",
                                 usage=_usage(50))
            return _envelope(True, model="B", usage=_usage(9))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "A,B",
            }))
        self.assertTrue(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual([a["model"] for a in rec["attempts"]], ["A", "B"])
        self.assertEqual(rec["fallback_count"], 1)
        # Per-attempt usage only in metrics detail, not mixed into envelope.
        self.assertEqual(rec["attempts"][0]["usage"]["input_tokens"], 50)
        self.assertEqual(rec["usage"]["input_tokens"], 9)

    def test_rate_limit_retries_count_as_attempts_same_seat(self):
        calls = {"n": 0}

        async def router(prompt, model, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _envelope(False, reason="rate_limit", model=model)
            return _envelope(True, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "RATE_LIMIT_BACKOFF_S", 0), \
             mock.patch.object(dispatch, "RATE_LIMIT_RETRIES", 1), \
             mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Solo",
            }))
        self.assertTrue(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual(len(rec["attempts"]), 2)
        self.assertEqual(rec["fallback_count"], 0)
        self.assertEqual(rec["attempts"][0]["reason"], "rate_limit")

    def test_winning_seat_usage_only_on_envelope(self):
        """Pin: losing attempt usage never appears on the envelope."""
        lose_u = _usage(999)
        win_u = _usage(10)

        async def router(prompt, model, *a, **k):
            if model == "Lose":
                return _envelope(False, reason="runtime_error", model="Lose",
                                 usage=lose_u)
            return _envelope(True, model="Win", usage=win_u)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Lose,Win",
            }))
        self.assertTrue(res["ok"])
        self.assertEqual(res["usage"]["input_tokens"], 10)
        self.assertNotEqual(res["usage"]["input_tokens"], 999)
        # Losing usage must not leak into envelope.usage (or raw aliases).
        envelope_blob = json.dumps(res["usage"])
        self.assertNotIn("999", envelope_blob)
        rec = _read_lines()[0]
        self.assertEqual(rec["usage"]["input_tokens"], 10)
        lose_attempts = [a for a in rec["attempts"] if a["model"] == "Lose"]
        self.assertEqual(lose_attempts[0]["usage"]["input_tokens"], 999)

    def test_opt_out_creates_no_file(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        path = metrics_mod.metrics_path()
        self.assertFalse(os.path.exists(path))
        with mock.patch.dict(os.environ, {"OMNI_LLM_METRICS": "0"}), \
             mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "M",
            }))
        self.assertTrue(res["ok"])
        self.assertFalse(os.path.exists(path), "opt-out must not create metrics.jsonl")
        self.assertNotIn("metrics_error", res)

    def test_rotation_at_5_mib(self):
        path = metrics_mod.metrics_path()
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        # Just over the 5 MiB threshold.
        with open(path, "wb") as handle:
            handle.write(b"x" * (5 * 1024 * 1024 + 1))
        metrics_mod.append_record({
            "ts": time.time(), "entry": "mcp", "tool": "llm_generate",
            "seat": {"provider": "agy", "model": "m", "account_profile": "default"},
            "attempts": [], "fallback_count": 0, "request_bytes": 0,
            "result_bytes": 0, "latency_ms": 0, "usage": None, "ok": True,
        })
        self.assertTrue(os.path.exists(path + ".1"))
        self.assertTrue(os.path.exists(path))
        self.assertLess(os.path.getsize(path), 5 * 1024 * 1024)
        lines = _read_lines(path)
        self.assertEqual(len(lines), 1)

    def test_write_failure_sets_metrics_error_result_unchanged(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, text="KEEP", model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router), \
             mock.patch.object(metrics_mod, "append_record",
                               side_effect=OSError("disk full")):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "M",
            }))
        self.assertTrue(res["ok"])
        self.assertEqual(res["text"], "KEEP")
        self.assertEqual(res["usage"]["input_tokens"], 1)
        self.assertIn("metrics_error", res)
        self.assertIn("disk full", res["metrics_error"])

    def test_permission_bits(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, model=model)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            _run(dispatch.call_tool("llm_generate", {"prompt": "p", "model": "M"}))
        path = metrics_mod.metrics_path()
        state_dir = os.path.dirname(path)
        dir_mode = stat.S_IMODE(os.stat(state_dir).st_mode)
        file_mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(dir_mode, 0o700)
        self.assertEqual(file_mode, 0o600)

    def test_concurrent_append_no_tears(self):
        path = metrics_mod.metrics_path()
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        # Fresh file for this test.
        _clear_metrics_file()
        worker = r"""
import json, os, sys, time
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import metrics as m
for i in range(50):
    m.append_record({
        "ts": time.time(), "entry": "mcp", "tool": "llm_generate",
        "seat": {"provider": "agy", "model": "m", "account_profile": "default"},
        "attempts": [], "fallback_count": 0, "request_bytes": 0,
        "result_bytes": 0, "latency_ms": 0, "usage": None, "ok": True,
        "worker": int(sys.argv[1]), "i": i,
    })
""" % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _TMP)
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", worker, str(w)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            for w in (1, 2)
        ]
        for proc in procs:
            out, err = proc.communicate(timeout=30)
            self.assertEqual(proc.returncode, 0, err.decode() or out.decode())
        raw = open(path, "rb").read()
        # Every line must be complete JSON ending with newline; no partial merges.
        lines = raw.splitlines()
        self.assertEqual(len(lines), 100, "expected 100 complete lines, got %d" % len(lines))
        parsed = [json.loads(ln) for ln in lines]
        self.assertEqual(len(parsed), 100)
        workers = sorted((r["worker"], r["i"]) for r in parsed)
        expected = sorted((w, i) for w in (1, 2) for i in range(50))
        self.assertEqual(workers, expected)

    def test_empty_prompt_failure_still_metrics(self):
        res = _run(dispatch.call_tool("llm_generate", {"prompt": "  "}))
        self.assertFalse(res["ok"])
        rec = _read_lines()[0]
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["reason"], "runtime_error")
        self.assertEqual(rec["attempts"], [])

    def test_short_write_never_leaves_torn_line(self):
        """F1: short os.write must complete or truncate — every line still parses."""
        path = metrics_mod.metrics_path()
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        _clear_metrics_file()
        real_write = os.write
        state = {"n": 0}

        def short_then_real(fd, data):
            state["n"] += 1
            if state["n"] == 1 and len(data) > 1:
                # First call: write only half, simulating a short write.
                half = len(data) // 2
                return real_write(fd, data[:half])
            return real_write(fd, data)

        with mock.patch.object(os, "write", side_effect=short_then_real):
            metrics_mod.append_record({
                "ts": time.time(), "entry": "mcp", "tool": "llm_generate",
                "seat": {"provider": "agy", "model": "m", "account_profile": "default"},
                "attempts": [], "fallback_count": 0, "request_bytes": 0,
                "result_bytes": 0, "latency_ms": 0, "usage": None, "ok": True,
                "marker": "short-write-ok",
            })
        raw = open(path, "rb").read()
        self.assertTrue(raw.endswith(b"\n"), "file must end with newline")
        lines = raw.splitlines()
        self.assertGreaterEqual(len(lines), 1)
        for ln in lines:
            parsed = json.loads(ln)
            self.assertIsInstance(parsed, dict)
        self.assertEqual(json.loads(lines[-1])["marker"], "short-write-ok")

    def test_persisted_usage_has_no_raw_envelope_keeps_raw(self):
        """F3: disk usage is allowlist projection; envelope retains usage.raw."""
        full = _usage(42)
        self.assertIn("raw", full)

        async def router(prompt, model, *a, **k):
            return _envelope(True, text="t", model=model, usage=full)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "M",
            }))
        self.assertTrue(res["ok"])
        # Envelope keeps raw untouched.
        self.assertIn("raw", res["usage"])
        self.assertEqual(res["usage"]["raw"], full["raw"])
        # Persisted record has no "raw" key anywhere.
        path = metrics_mod.metrics_path()
        whole = open(path, encoding="utf-8").read()
        self.assertNotIn('"raw"', whole)
        rec = _read_lines()[0]
        self.assertIsNotNone(rec["usage"])
        self.assertNotIn("raw", rec["usage"])
        for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens",
                    "output_tokens", "source"):
            self.assertIn(key, rec["usage"])
        # Extra envelope fields (reasoning/total) must not appear on disk usage.
        self.assertNotIn("reasoning_tokens", rec["usage"])
        self.assertNotIn("total_tokens", rec["usage"])
        self.assertEqual(rec["usage"]["input_tokens"], 42)
        # Per-attempt usage is also projected.
        self.assertNotIn("raw", rec["attempts"][0]["usage"])

    def test_llm_big_context_load_err_request_bytes(self):
        """F6: load_err early-return records instruction-only request_bytes."""
        instruction = "INSTR-ONLY-BYTES"

        def fake_resolve(text, source_path, instruction=""):
            return None, {
                "ok": False,
                "reason": "runtime_error",
                "error_code": "not_found",
                "detail": "missing",
            }

        with mock.patch.object(
                dispatch, "resolve_big_context_source", side_effect=fake_resolve):
            res = _run(dispatch.call_tool("llm_big_context", {
                "instruction": instruction,
                "source_path": "/no/such/path/for/metrics",
                "model": "M",
            }))
        self.assertFalse(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual(rec["tool"], "llm_big_context")
        self.assertEqual(rec["request_bytes"], len(instruction.encode("utf-8")))
        self.assertFalse(rec["ok"])

    def test_llm_big_context_normal_instruction_plus_source(self):
        """F6: normal path request_bytes = instruction + source body sizes."""
        instruction = "DO-THIS"
        source = "SOURCE-BODY-XYZ"

        async def router(prompt, model, *a, **k):
            return _envelope(True, text="big-out", model=model, usage=_usage(2))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_big_context", {
                "instruction": instruction,
                "text": source,
                "model": "BigModel",
            }))
        self.assertTrue(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual(rec["tool"], "llm_big_context")
        expected = len(instruction.encode("utf-8")) + len(source.encode("utf-8"))
        self.assertEqual(rec["request_bytes"], expected)
        self.assertEqual(rec["result_bytes"], len(b"big-out"))
        self.assertEqual(rec["seat"]["model"], "BigModel")
        self.assertEqual(rec["usage"]["input_tokens"], 2)
        whole = open(metrics_mod.metrics_path(), encoding="utf-8").read()
        self.assertNotIn(instruction, whole)
        self.assertNotIn(source, whole)

    def test_llm_image_synthetic_seat(self):
        """F6: llm_image records a synthetic seat + attempt for the image backend."""
        from providers import catalog_v2
        from providers.catalog_record import validate_strict
        from providers.grok_image_cli import GROK_IMAGE_MODEL
        record, _warnings = validate_strict("grok", {
            "provider": "grok",
            "source": "live",
            "observed_at": "2026-11-10T00:00:00Z",
            "availability": "provider_advertised",
            "models": [{
                "id": GROK_IMAGE_MODEL,
                "name": GROK_IMAGE_MODEL,
                "routable_id": "grok/%s" % GROK_IMAGE_MODEL,
                "supported_efforts": ["medium"],
            }],
        })
        catalog_v2.save("grok", "default", record)

        async def fake_cli(prompt, source_image, timeout_s=None, out_dir=None,
                           account_profile=None, apply_aliases=False):
            return {
                "ok": True, "text": "", "model": "grok-image-edit",
                "images": ["fake.png"],
            }

        with mock.patch.object(
                dispatch.grok_image_cli_mod, "run_grok_image_edit",
                side_effect=fake_cli):
            res = _run(dispatch.call_tool("llm_image", {
                "prompt": "draw a cat",
                "backend": "cli",
                "source_image": "/tmp/src.png",
            }))
        self.assertTrue(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual(rec["tool"], "llm_image")
        self.assertEqual(rec["request_bytes"], len(b"draw a cat"))
        self.assertEqual(rec["seat"]["model"], "grok-image-edit")
        self.assertEqual(rec["seat"]["provider"], "grok")
        self.assertEqual(len(rec["attempts"]), 1)
        self.assertEqual(rec["attempts"][0]["model"], "grok-image-edit")
        self.assertEqual(rec["attempts"][0]["provider"], "grok")
        self.assertTrue(rec["attempts"][0]["ok"])
        whole = open(metrics_mod.metrics_path(), encoding="utf-8").read()
        self.assertNotIn("draw a cat", whole)

    def test_overlong_model_name_truncated_and_marked(self):
        """R3: caller-controlled model/profile strings are capped at FIELD_MAX_LEN."""
        long_model = "M" * (metrics_mod.FIELD_MAX_LEN + 50)
        normal_model = "NormalModel"

        # Unit: bound_field itself.
        capped = metrics_mod.bound_field(long_model)
        self.assertEqual(len(capped), metrics_mod.FIELD_MAX_LEN)
        self.assertTrue(
            capped.endswith(metrics_mod.FIELD_TRUNC_MARKER),
            capped[-40:],
        )
        self.assertNotEqual(capped, long_model)
        self.assertEqual(metrics_mod.bound_field(normal_model), normal_model)

        # Disk path: over-long model lands truncated+marked on seat and attempts.
        async def router(prompt, model, *a, **k):
            return _envelope(True, text="ok", model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": long_model,
            }))
        self.assertTrue(res["ok"])
        rec = _read_lines()[0]
        self.assertEqual(rec["seat"]["model"], capped)
        self.assertTrue(rec["seat"]["model"].endswith(metrics_mod.FIELD_TRUNC_MARKER))
        self.assertEqual(rec["attempts"][0]["model"], capped)
        self.assertEqual(len(rec["seat"]["model"]), metrics_mod.FIELD_MAX_LEN)

        # Normal name untouched on a subsequent call.
        _clear_metrics_file()
        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res2 = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": normal_model,
            }))
        self.assertTrue(res2["ok"])
        rec2 = _read_lines()[0]
        self.assertEqual(rec2["seat"]["model"], normal_model)
        self.assertEqual(rec2["attempts"][0]["model"], normal_model)
        self.assertFalse(rec2["seat"]["model"].endswith(metrics_mod.FIELD_TRUNC_MARKER))

        # Profile is capped the same way (build_record seat fields).
        long_profile = "P" * (metrics_mod.FIELD_MAX_LEN + 10)
        profile_capped = metrics_mod.bound_field(long_profile)
        rec3 = metrics_mod.build_record(
            entry="mcp", tool="llm_generate",
            seat={"provider": "agy", "model": normal_model,
                  "account_profile": long_profile},
            attempts=[], fallback_count=0, request_bytes=0, result_bytes=0,
            latency_ms=0, usage=None, ok=True,
        )
        self.assertEqual(rec3["seat"]["account_profile"], profile_capped)
        self.assertTrue(
            rec3["seat"]["account_profile"].endswith(metrics_mod.FIELD_TRUNC_MARKER)
        )

    def test_ensure_state_dir_creates_0700_leaves_existing_alone(self):
        """R4: create with 0700; pre-existing dir mode is not chmod'd."""
        # Branch A: metrics creates the directory → 0700.
        create_root = tempfile.mkdtemp(prefix="omni-metrics-create-")
        try:
            new_dir = os.path.join(create_root, "brand-new-state")
            self.assertFalse(os.path.isdir(new_dir))
            metrics_mod._ensure_state_dir(new_dir)
            self.assertTrue(os.path.isdir(new_dir))
            mode = stat.S_IMODE(os.stat(new_dir).st_mode)
            self.assertEqual(mode, 0o700)
        finally:
            import shutil
            shutil.rmtree(create_root, ignore_errors=True)

        # Branch B: pre-existing looser dir → mode left alone, warn once.
        exist_root = tempfile.mkdtemp(prefix="omni-metrics-exist-")
        try:
            loose = os.path.join(exist_root, "loose-state")
            os.mkdir(loose, 0o755)
            # Force 0755 in case umask tightened mkdir.
            os.chmod(loose, 0o755)
            before = stat.S_IMODE(os.stat(loose).st_mode)
            self.assertEqual(before, 0o755)

            # Reset the per-directory latch so this test observes the warn path.
            metrics_mod._loose_dirs_warned.clear()
            with mock.patch.object(metrics_mod, "log_event") as log_ev:
                metrics_mod._ensure_state_dir(loose)
                after = stat.S_IMODE(os.stat(loose).st_mode)
                self.assertEqual(after, 0o755, "must not chmod a pre-existing dir")
                log_ev.assert_called_once()
                args, kwargs = log_ev.call_args
                self.assertEqual(args[0], "metrics_state_dir_loose_mode")
                self.assertEqual(kwargs.get("path"), loose)

                # Second call: still no chmod; warn only once.
                metrics_mod._ensure_state_dir(loose)
                self.assertEqual(stat.S_IMODE(os.stat(loose).st_mode), 0o755)
                self.assertEqual(log_ev.call_count, 1)
        finally:
            import shutil
            shutil.rmtree(exist_root, ignore_errors=True)
            metrics_mod._loose_dirs_warned.clear()

    def test_each_distinct_loose_dir_warns_once(self):
        """The latch is per-directory: _state_dir() is re-resolved every call, so a
        second root met later in the same process must not be silently skipped."""
        root = tempfile.mkdtemp(prefix="omni-metrics-multi-")
        try:
            first = os.path.join(root, "one")
            second = os.path.join(root, "two")
            for d in (first, second):
                os.mkdir(d, 0o755)
                os.chmod(d, 0o755)
            metrics_mod._loose_dirs_warned.clear()
            with mock.patch.object(metrics_mod, "log_event") as log_ev:
                metrics_mod._ensure_state_dir(first)
                metrics_mod._ensure_state_dir(first)   # repeat: still one warning
                metrics_mod._ensure_state_dir(second)  # distinct dir: its own warning
                paths = [kw.get("path") for _, kw in log_ev.call_args_list]
                self.assertEqual(paths, [first, second])
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
            metrics_mod._loose_dirs_warned.clear()

    def test_append_record_bounds_fields_on_hand_built_record(self):
        """The content-free invariant is a property of what reaches disk, so the cap
        is enforced at the write boundary, not only where build_record happens to
        be used. A hand-built record must not smuggle an unbounded value through."""
        root = tempfile.mkdtemp(prefix="omni-metrics-bound-")
        old = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = root
        try:
            record = {
                "ts": 0, "tool": "llm_generate", "entry": "cli",
                "seat": {"provider": "p", "model": "M" * 400, "account_profile": "P" * 400},
                "attempts": [{"model": "A" * 400}],
                "fallback_count": 0, "request_bytes": 1, "result_bytes": 1,
                "latency_ms": 1, "usage": None, "ok": True,
            }
            metrics_mod.append_record(record)
            written = json.loads(
                open(os.path.join(root, "metrics.jsonl"), encoding="utf-8").read().splitlines()[-1]
            )
            self.assertEqual(len(written["seat"]["model"]), metrics_mod.FIELD_MAX_LEN)
            self.assertEqual(len(written["seat"]["account_profile"]), metrics_mod.FIELD_MAX_LEN)
            self.assertEqual(len(written["attempts"][0]["model"]), metrics_mod.FIELD_MAX_LEN)
            self.assertTrue(written["seat"]["model"].endswith(metrics_mod.FIELD_TRUNC_MARKER))
            # the caller's own dict must not be mutated as a side effect
            self.assertEqual(len(record["seat"]["model"]), 400)
        finally:
            if old is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old
            import shutil
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
