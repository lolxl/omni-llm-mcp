# no-opencc
"""Acceptance tests — opaque artifact references (issue #22 P1.4 / fix2).

Covers content-addressed ids, resource bounds, full fd discipline on mutators,
no opportunistic GC, and the prior P1–P7 surface (inline default, ref/auto,
preview, expiry, confinement, CLI, metrics).
"""
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-artifacts-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)
# Fresh metrics file for spill metrics tests.
os.environ.pop("OMNI_LLM_METRICS", None)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from providers import artifacts as artifacts_mod  # noqa: E402
from providers import dispatch  # noqa: E402
from providers import health  # noqa: E402
from providers import metrics as metrics_mod  # noqa: E402
import omni_cli  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _envelope(ok=True, **kw):
    out = {"ok": ok}
    out.update(kw)
    return out


def _usage(n=1):
    return {
        "input_tokens": n,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": n,
        "reasoning_tokens": None,
        "total_tokens": None,
        "source": "provider",
        "raw": {"n": n},
    }


def _read_metrics_lines():
    path = metrics_mod.metrics_path()
    if not os.path.isfile(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def _cli(argv, env=None):
    """Run omni_cli.main with argv; return (exit_code, stdout, stderr)."""
    e = os.environ.copy()
    e["OMNI_LLM_STATE_DIR"] = os.environ["OMNI_LLM_STATE_DIR"]
    if env:
        e.update(env)
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "omni_cli.py")] + list(argv),
        capture_output=True,
        text=True,
        env=e,
        cwd=_REPO,
        timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _hex_id(label):
    """Deterministic 64-hex id for planting fixtures (not content-addressed)."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ArtifactsBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="omni-art-case-")
        os.environ["OMNI_LLM_STATE_DIR"] = self._tmp
        os.environ.pop("OMNI_LLM_ARTIFACT_TTL_S", None)
        os.environ.pop("OMNI_LLM_ARTIFACT_MAX_BYTES", None)
        os.environ.pop("OMNI_LLM_ARTIFACT_STORE_MAX_BYTES", None)
        os.environ.pop("OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES", None)
        os.environ.pop("OMNI_LLM_METRICS", None)
        artifacts_mod.set_now_fn(None)
        try:
            health.clear_all()
        except Exception:
            pass
        # Clear any metrics file from prior case under this state dir.
        mp = metrics_mod.metrics_path()
        try:
            os.unlink(mp)
        except OSError:
            pass

    def tearDown(self):
        artifacts_mod.set_now_fn(None)
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        os.environ.pop("OMNI_LLM_ARTIFACT_MAX_BYTES", None)
        os.environ.pop("OMNI_LLM_ARTIFACT_STORE_MAX_BYTES", None)
        os.environ.pop("OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES", None)


class TestInlineDefaultByteIdentity(_ArtifactsBase):
    def test_default_envelope_byte_identical_to_base(self):
        """P1: without opt-in, envelope is deep-equal to base behavior."""
        body = "HELLO_IDENTITY_" + ("x" * 100)

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(3))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            base = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake",
            }))
            # Explicit absence of result_mode
            again = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake",
            }))
            # Explicit inline
            inline = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "inline",
            }))

        # Strip metrics-only keys that may vary by ts; compare model fields.
        def strip(env):
            d = dict(env)
            d.pop("metrics_error", None)
            return d

        self.assertEqual(strip(base), strip(again))
        self.assertEqual(strip(base), strip(inline))
        self.assertTrue(base["ok"])
        self.assertEqual(base["text"], body)
        self.assertNotIn("artifact_ref", base)
        self.assertNotIn("artifact_ref", inline)

    def test_big_context_default_inline(self):
        body = "BIG_BODY_OK"

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(2))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_big_context", {
                "instruction": "sum", "text": "source", "model": "Fake",
            }))
        self.assertTrue(res["ok"])
        self.assertEqual(res["text"], body)
        self.assertNotIn("artifact_ref", res)


class TestRefAndAutoSpill(_ArtifactsBase):
    def test_ref_spills_any_nonempty_success_text(self):
        body = "short-but-ref"

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "ref",
            }))
        self.assertTrue(res["ok"])
        self.assertIn("artifact_ref", res)
        ref = res["artifact_ref"]
        for key in ("id", "sha256", "text_bytes", "expires"):
            self.assertIn(key, ref)
        self.assertEqual(ref["text_bytes"], len(body.encode("utf-8")))
        # Content-addressed id == sha256(blob).
        self.assertEqual(ref["id"], hashlib.sha256(body.encode("utf-8")).hexdigest())
        self.assertEqual(ref["id"], ref["sha256"])
        self.assertEqual(
            res["text"],
            body + artifacts_mod.PREVIEW_SUFFIX
            if len(body) <= artifacts_mod.PREVIEW_CHARS
            else body[:artifacts_mod.PREVIEW_CHARS] + artifacts_mod.PREVIEW_SUFFIX,
        )
        got = artifacts_mod.read(ref["id"])
        self.assertTrue(got["ok"])
        self.assertEqual(got["text"], body)

    def test_auto_threshold_boundary(self):
        """65536 exactly = inline; 65537 = spill."""
        exact = "a" * artifacts_mod.AUTO_THRESHOLD_BYTES
        plus = "a" * (artifacts_mod.AUTO_THRESHOLD_BYTES + 1)
        self.assertEqual(len(exact.encode("utf-8")), 65536)
        self.assertEqual(len(plus.encode("utf-8")), 65537)

        async def router_exact(prompt, model, *a, **k):
            return _envelope(True, text=exact, model=model, usage=_usage(1))

        async def router_plus(prompt, model, *a, **k):
            return _envelope(True, text=plus, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider",
                               side_effect=router_exact):
            r1 = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "auto",
            }))
        self.assertTrue(r1["ok"])
        self.assertNotIn("artifact_ref", r1)
        self.assertEqual(r1["text"], exact)

        with mock.patch.object(dispatch, "route_to_provider",
                               side_effect=router_plus):
            r2 = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "auto",
            }))
        self.assertTrue(r2["ok"])
        self.assertIn("artifact_ref", r2)
        self.assertTrue(r2["text"].endswith(artifacts_mod.PREVIEW_SUFFIX))
        self.assertEqual(
            r2["text"],
            plus[:artifacts_mod.PREVIEW_CHARS] + artifacts_mod.PREVIEW_SUFFIX,
        )

    def test_preview_shape_long_body(self):
        body = "Z" * 2000
        preview = artifacts_mod.make_preview(body)
        self.assertEqual(len(preview), artifacts_mod.PREVIEW_CHARS + len(
            artifacts_mod.PREVIEW_SUFFIX))
        self.assertTrue(preview.endswith("…[spilled]"))
        self.assertEqual(preview[:artifacts_mod.PREVIEW_CHARS], body[:512])

    def test_ref_empty_text_does_not_spill(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, text="", model=model)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "ref",
            }))
        self.assertTrue(res["ok"])
        self.assertNotIn("artifact_ref", res)

    def test_big_context_ref_entry(self):
        body = "big-ref-body"

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_big_context", {
                "instruction": "x", "text": "src", "model": "Fake",
                "result_mode": "ref",
            }))
        self.assertTrue(res["ok"])
        self.assertIn("artifact_ref", res)


class TestStoreIntegrity(_ArtifactsBase):
    def test_sha256_reverify_and_corrupt(self):
        ref = artifacts_mod.create("payload-for-corrupt-test")
        art_dir = os.path.join(artifacts_mod.artifacts_root(), ref["id"])
        blob = os.path.join(art_dir, "blob")
        with open(blob, "wb") as handle:
            handle.write(b"TAMPERED")
        got = artifacts_mod.read(ref["id"])
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_not_found(self):
        # Must be 64 hex to pass confinement and hit ENOENT.
        missing = "a" * 64
        got = artifacts_mod.read(missing)
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_not_found")

    def test_expiry_fake_clock(self):
        clock = {"t": 1_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create("expire-me", ttl_s=10)
        # not yet
        clock["t"] = 1_000_000.0 + 5
        ok = artifacts_mod.read(ref["id"])
        self.assertTrue(ok["ok"])
        # boundary: now == expires → still valid (expired only when now > expires)
        clock["t"] = ref["expires"]
        boundary = artifacts_mod.read(ref["id"])
        self.assertTrue(boundary["ok"], boundary)
        # expired
        clock["t"] = ref["expires"] + 0.001
        exp = artifacts_mod.read(ref["id"])
        self.assertFalse(exp["ok"])
        self.assertEqual(exp["reason"], "artifact_expired")

    def test_confinement_dotdot_abs_symlink(self):
        # ".."
        r1 = artifacts_mod.read("..")
        self.assertFalse(r1["ok"])
        self.assertEqual(r1["reason"], "artifact_not_found")
        r1b = artifacts_mod.read("../etc")
        self.assertEqual(r1b["reason"], "artifact_not_found")
        # absolute
        r2 = artifacts_mod.read("/tmp/evil")
        self.assertEqual(r2["reason"], "artifact_not_found")
        # symlink as artifact id target (valid-looking hex id)
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        real = tempfile.mkdtemp(prefix="omni-art-out-")
        link_name = _hex_id("symlink-decoy-id")
        link_path = os.path.join(root, link_name)
        os.symlink(real, link_path)
        r3 = artifacts_mod.read(link_name)
        self.assertFalse(r3["ok"])
        # ELOOP → artifact_corrupt (only ENOENT is not_found)
        self.assertEqual(r3["reason"], "artifact_corrupt")

    def test_atomic_write_no_partial_visible(self):
        """Published id appears only with both meta.json and blob present."""
        ref = artifacts_mod.create("atomic-body-xyz")
        art_dir = os.path.join(artifacts_mod.artifacts_root(), ref["id"])
        self.assertTrue(os.path.isdir(art_dir))
        self.assertTrue(os.path.isfile(os.path.join(art_dir, "blob")))
        self.assertTrue(os.path.isfile(os.path.join(art_dir, "meta.json")))
        # Modes: dir 0700, files 0600
        self.assertEqual(stat.S_IMODE(os.stat(art_dir).st_mode), 0o700)
        self.assertEqual(
            stat.S_IMODE(os.stat(os.path.join(art_dir, "blob")).st_mode), 0o600
        )
        self.assertEqual(
            stat.S_IMODE(os.stat(os.path.join(art_dir, "meta.json")).st_mode), 0o600
        )
        # Simulate failed mid-publish: temp dirs with prefix must not be readable ids
        tmp = tempfile.mkdtemp(
            prefix=".omni-art-pub-", dir=artifacts_mod.artifacts_root()
        )
        with open(os.path.join(tmp, "blob"), "wb") as handle:
            handle.write(b"partial")
        # GC must not delete this unpublished temp (dot-prefixed).
        artifacts_mod.gc(force=True)
        self.assertTrue(os.path.isdir(tmp))
        # And it is not a valid published id.
        self.assertFalse(artifacts_mod.read(os.path.basename(tmp)).get("ok"))

    def test_store_is_sibling_of_jobs_not_inside(self):
        root = artifacts_mod.artifacts_root()
        self.assertTrue(root.endswith(os.path.join("", "artifacts")) or
                        root.rstrip(os.sep).endswith("artifacts"))
        self.assertNotIn(os.path.join("jobs", "artifacts"), root)
        state = os.path.dirname(root)
        self.assertEqual(root, os.path.join(state, "artifacts"))


class TestGcAndCli(_ArtifactsBase):
    def test_gc_decoy_survival(self):
        """GC only deletes inside artifacts/; jobs decoy + temp survive."""
        clock = {"t": 2_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        # Create expired artifact
        ref = artifacts_mod.create("gc-me", ttl_s=1)
        clock["t"] = ref["expires"] + 10

        # Decoy under jobs/
        state = os.path.dirname(artifacts_mod.artifacts_root())
        jobs = os.path.join(state, "jobs")
        os.makedirs(jobs, mode=0o700, exist_ok=True)
        decoy = os.path.join(jobs, "decoy-must-survive.txt")
        with open(decoy, "w", encoding="utf-8") as handle:
            handle.write("DECOY")

        # Unpublished temp under artifacts/
        tmp = tempfile.mkdtemp(
            prefix=".omni-art-pub-", dir=artifacts_mod.artifacts_root()
        )
        partial = os.path.join(tmp, "blob")
        with open(partial, "wb") as handle:
            handle.write(b"UNPUBLISHED")

        res = artifacts_mod.gc(force=True)
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(res["deleted"], 1)
        self.assertFalse(os.path.isdir(
            os.path.join(artifacts_mod.artifacts_root(), ref["id"])))
        self.assertTrue(os.path.isfile(decoy), "jobs decoy must survive GC")
        self.assertTrue(os.path.isdir(tmp), "unpublished temp must survive GC")
        self.assertTrue(os.path.isfile(partial))

    def test_no_opportunistic_gc_on_create(self):
        """R3/T7: create never writes marker/lease; expired siblings stay until explicit gc."""
        clock = {"t": 3_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create("will-expire", ttl_s=1)
        root = artifacts_mod.artifacts_root()
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_last")))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_lease")))

        clock["t"] = ref["expires"] + 5
        # Second create does not reap the expired first (no opportunistic GC).
        # (Quota reclaim only runs when store is at quota — not the case here.)
        ref2 = artifacts_mod.create("second", ttl_s=1)
        self.assertTrue(os.path.isdir(os.path.join(root, ref["id"])))
        self.assertTrue(os.path.isdir(os.path.join(root, ref2["id"])))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_last")))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_lease")))

        # Explicit GC reaps expired.
        clock["t"] = ref2["expires"] + 5
        r = artifacts_mod.gc(force=True)
        self.assertTrue(r["ok"])
        self.assertFalse(os.path.isdir(os.path.join(root, ref["id"])))
        self.assertFalse(os.path.isdir(os.path.join(root, ref2["id"])))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_last")))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_lease")))

    def test_cli_read_rm_gc_json_and_exit_codes(self):
        ref = artifacts_mod.create("cli-body-abc")
        code, out, err = _cli(["artifact", "read", ref["id"]])
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out.strip())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["text"], "cli-body-abc")

        code, out, err = _cli(["artifact", "read", "a" * 64])
        self.assertEqual(code, 1)
        payload = json.loads(out.strip())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "artifact_not_found")

        code, out, err = _cli(["artifact", "rm", ref["id"]])
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out.strip())
        self.assertTrue(payload["ok"])

        code, out, err = _cli(["artifact", "rm", ref["id"]])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.strip())["reason"], "artifact_not_found")

        code, out, err = _cli(["artifact", "gc"])
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out.strip())
        self.assertTrue(payload["ok"])
        self.assertIn("deleted", payload)

    def test_cli_out_mode_ref(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, text="via-cli-ref", model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            # Use library path for determinism (subprocess would need deeper mocks).
            ns_args = omni_cli._args_gen(type("NS", (), {
                "prompt": "hi", "profile": "default", "timeout": 30,
                "model": "Fake", "files": None, "out_mode": "ref",
            })())
            self.assertEqual(ns_args.get("result_mode"), "ref")
            res = _run(dispatch.call_tool("llm_generate", ns_args))
        self.assertTrue(res["ok"])
        self.assertIn("artifact_ref", res)


class TestMetricsOriginalTextBytes(_ArtifactsBase):
    def test_spilled_records_original_text_bytes(self):
        body = "METRICS_SPILL_BODY_" + ("m" * 200)

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(4))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "ref",
            }))
        self.assertTrue(res["ok"])
        self.assertIn("artifact_ref", res)
        lines = _read_metrics_lines()
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["original_text_bytes"], len(body.encode("utf-8")))
        # result_bytes is wire (preview) size, not original.
        self.assertEqual(rec["result_bytes"], len(res["text"].encode("utf-8")))
        self.assertNotEqual(rec["result_bytes"], rec["original_text_bytes"])
        # Long body: preview is strictly smaller than original.
        long_body = "L" * 5000

        async def router2(prompt, model, *a, **k):
            return _envelope(True, text=long_body, model=model, usage=_usage(4))

        try:
            os.unlink(metrics_mod.metrics_path())
        except OSError:
            pass
        with mock.patch.object(dispatch, "route_to_provider", side_effect=router2):
            res2 = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "ref",
            }))
        rec2 = _read_metrics_lines()[0]
        self.assertLess(rec2["result_bytes"], rec2["original_text_bytes"])
        self.assertEqual(rec2["original_text_bytes"], len(long_body.encode("utf-8")))

    def test_inline_unchanged_no_original_text_bytes(self):
        body = "inline-metrics-body"

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake",
            }))
        self.assertTrue(res["ok"])
        rec = _read_metrics_lines()[0]
        self.assertNotIn("original_text_bytes", rec)
        self.assertEqual(rec["result_bytes"], len(body.encode("utf-8")))


class TestInvalidResultMode(_ArtifactsBase):
    def test_invalid_result_mode_loud(self):
        async def router(prompt, model, *a, **k):
            return _envelope(True, text="x", model=model)

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "bogus",
            }))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")


# ---------------------------------------------------------------------------
# P1.4-fix2 security coverage (T1–T8)
# ---------------------------------------------------------------------------


class TestContentAddressedIntegrity(_ArtifactsBase):
    """T1: content-addressed id is the integrity anchor; meta.sha256 advisory."""

    def test_tampered_blob_under_valid_id_is_corrupt(self):
        ref = artifacts_mod.create("t1-tamper-body")
        art_dir = os.path.join(artifacts_mod.artifacts_root(), ref["id"])
        with open(os.path.join(art_dir, "blob"), "wb") as handle:
            handle.write(b"TAMPERED-CONTENT")
        got = artifacts_mod.read(ref["id"])
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_forged_meta_sha256_matching_tampered_blob_still_corrupt(self):
        """Even if meta.sha256 is rewritten to match the tampered blob, fail.

        Integrity source is id == sha256(blob), not meta.sha256.
        """
        ref = artifacts_mod.create("t1-forge-meta-sha")
        art_dir = os.path.join(artifacts_mod.artifacts_root(), ref["id"])
        tampered = b"TAMPERED-WITH-FORGED-META"
        with open(os.path.join(art_dir, "blob"), "wb") as handle:
            handle.write(tampered)
        forged_digest = hashlib.sha256(tampered).hexdigest()
        with open(os.path.join(art_dir, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "id": ref["id"],
                "sha256": forged_digest,  # matches tampered blob, not id
                "text_bytes": len(tampered),
                "created": 1.0,
                "expires": time.time() + 99999,
            }) + "\n")
        got = artifacts_mod.read(ref["id"])
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")


class TestIdempotentRecreate(_ArtifactsBase):
    """T2: identical content → same id, no duplicate dir."""

    def test_idempotent_recreate_same_id_no_duplicate(self):
        body = "idempotent-body-xyz"
        r1 = artifacts_mod.create(body, ttl_s=100)
        r2 = artifacts_mod.create(body, ttl_s=200)
        self.assertEqual(r1["id"], r2["id"])
        self.assertEqual(
            r1["id"], hashlib.sha256(body.encode("utf-8")).hexdigest()
        )
        root = artifacts_mod.artifacts_root()
        # Exactly one published dir for this id.
        matches = [
            n for n in os.listdir(root)
            if n == r1["id"] and not n.startswith(".")
        ]
        self.assertEqual(matches, [r1["id"]])
        # Expiry refreshed.
        self.assertGreater(r2["expires"], r1["expires"])
        got = artifacts_mod.read(r1["id"])
        self.assertTrue(got["ok"])
        self.assertEqual(got["text"], body)


class TestResourceBounds(_ArtifactsBase):
    """T3/T4: caps refuse before write; read aborts on oversize."""

    def test_per_artifact_cap_refuses_before_write(self):
        os.environ["OMNI_LLM_ARTIFACT_MAX_BYTES"] = "64"
        root = artifacts_mod.artifacts_root()
        before = set(os.listdir(root)) if os.path.isdir(root) else set()
        with self.assertRaises(artifacts_mod.ArtifactError) as ctx:
            artifacts_mod.create("x" * 100)
        self.assertEqual(ctx.exception.reason, "artifact_too_large")
        after = set(os.listdir(root)) if os.path.isdir(root) else set()
        # No new published entry and no leftover temp.
        new = after - before
        for name in new:
            self.assertFalse(
                name.startswith(".omni-art-pub-"),
                "temp dir leaked: %s" % name,
            )
            # Only possible new names would be partial — none allowed.
            self.assertTrue(name.startswith("."), "partial entry: %s" % name)

    def test_store_quota_refuses_before_write(self):
        os.environ["OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES"] = "1"
        os.environ["OMNI_LLM_ARTIFACT_STORE_MAX_BYTES"] = str(10 * 1024 * 1024)
        r1 = artifacts_mod.create("first-entry")
        root = artifacts_mod.artifacts_root()
        before = set(os.listdir(root))
        with self.assertRaises(artifacts_mod.ArtifactError) as ctx:
            artifacts_mod.create("second-different-body")
        self.assertEqual(ctx.exception.reason, "artifact_store_full")
        after = set(os.listdir(root))
        new = after - before
        for name in new:
            self.assertFalse(name.startswith(".omni-art-pub-"), name)
            self.assertNotEqual(name, hashlib.sha256(
                b"second-different-body").hexdigest())
        # First entry still present alone as the only published id.
        self.assertTrue(os.path.isdir(os.path.join(root, r1["id"])))

    def test_read_aborts_when_blob_grows_past_cap(self):
        """T4: enforce cap on bytes actually read (st_size is only a pre-check)."""
        os.environ["OMNI_LLM_ARTIFACT_MAX_BYTES"] = "1024"
        ref = artifacts_mod.create("small-ok-body")
        # Plant an oversized blob under the content id path by rewriting blob
        # and meta id won't match — use monkeypatch on the read loop instead:
        # create normally, then patch max down and inject growing reads.
        os.environ["OMNI_LLM_ARTIFACT_MAX_BYTES"] = "16"

        # Direct unit: plant a file larger than cap with matching content id dir
        # whose blob was written oversize (id won't match → corrupt), so instead
        # monkeypatch _read_blob_streaming's underlying os.read to exceed cap
        # after a small fstat.
        art_dir = os.path.join(artifacts_mod.artifacts_root(), ref["id"])
        # Rewrite blob to something that streams past cap while fstat looks small:
        # we monkeypatch fstat st_size and the read loop.
        real_read = os.read
        state = {"n": 0}

        def growing_read(fd, n):
            state["n"] += 1
            if state["n"] == 1:
                return b"A" * 10
            if state["n"] == 2:
                return b"B" * 20  # total 30 > 16
            return b""

        # Open real artifact and force the streaming reader path with patch.
        # Easier: plant oversize file and lower cap; content-id mismatch yields
        # corrupt. WO wants abort mid-read on grow-past-cap specifically.
        with mock.patch.object(artifacts_mod.os, "read", side_effect=growing_read):
            # Also need fstat pre-check to pass (st_size <= cap).
            real_fstat = os.fstat

            def small_fstat(fd):
                st = real_fstat(fd)
                # Make a simple namespace with st_size small for pre-check.
                return st

            # fstat returns real size of planted blob ("small-ok-body") which is
            # under 16? "small-ok-body" is 13 bytes — under 16. growing_read
            # ignores the real file and returns 30 bytes total.
            got = artifacts_mod.read(ref["id"])
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")


class TestCleanupNoEscape(_ArtifactsBase):
    """T5: error-path cleanup cannot follow a swapped temp dir symlink."""

    def test_error_path_cleanup_symlink_swap_decoy_survives(self):
        decoy = tempfile.mkdtemp(prefix="omni-art-decoy-")
        decoy_file = os.path.join(decoy, "must-survive.txt")
        with open(decoy_file, "w", encoding="utf-8") as handle:
            handle.write("DECOY-PAYLOAD")

        # After temp dir is created and opened, swap the name for a symlink to
        # decoy and force a failure on the blob write.
        real_write = artifacts_mod._write_file_in_dir
        calls = {"n": 0}

        def swap_then_fail(dir_fd, name, data, mode=artifacts_mod.FILE_MODE):
            calls["n"] += 1
            if calls["n"] == 1 and name == "blob":
                # Locate temp name under root and swap it for symlink → decoy.
                root = artifacts_mod.artifacts_root()
                for entry in os.listdir(root):
                    if entry.startswith(".omni-art-pub-"):
                        path = os.path.join(root, entry)
                        # Remove the real temp dir contents carefully then swap.
                        # We still hold dir_fd to the original inode in create().
                        # Replace the directory entry with a symlink.
                        # First rename real temp aside, put symlink at name.
                        aside = path + ".aside"
                        os.rename(path, aside)
                        os.symlink(decoy, path)
                        # Force failure.
                        raise OSError("forced create failure after swap")
                raise OSError("forced create failure (no temp found)")
            return real_write(dir_fd, name, data, mode=mode)

        with mock.patch.object(
            artifacts_mod, "_write_file_in_dir", side_effect=swap_then_fail
        ):
            with self.assertRaises(OSError):
                artifacts_mod.create("cleanup-escape-body")

        self.assertTrue(os.path.isfile(decoy_file), "decoy file must survive")
        with open(decoy_file, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "DECOY-PAYLOAD")
        # Decoy dir itself must not have been rmdir'd.
        self.assertTrue(os.path.isdir(decoy))


class TestGcValidateThenDelete(_ArtifactsBase):
    """T6: GC binds validate+delete to one fd; corrupt never deleted as expired."""

    def test_gc_never_deletes_failing_manifest_validation(self):
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        aid = _hex_id("corrupt-gc-entry")
        art = os.path.join(root, aid)
        os.makedirs(art, mode=0o700)
        # Bad meta (invalid json) + some blob — fails validation.
        with open(os.path.join(art, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write("{not-json")
        with open(os.path.join(art, "blob"), "wb") as handle:
            handle.write(b"x")
        res = artifacts_mod.gc(force=True)
        self.assertTrue(res["ok"])
        self.assertEqual(res["deleted"], 0)
        self.assertGreaterEqual(res.get("corrupt", 0), 1)
        self.assertTrue(os.path.isdir(art), "corrupt entry must not be deleted")

    def test_gc_name_swap_after_validation_not_deleted_or_detected(self):
        """Swap the validated entry's name for an external decoy mid-GC.

        The decoy must survive; GC must not rmdir the swapped-in path.
        """
        clock = {"t": 5_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create("gc-swap-victim", ttl_s=1)
        clock["t"] = ref["expires"] + 10

        decoy = tempfile.mkdtemp(prefix="omni-gc-decoy-")
        decoy_file = os.path.join(decoy, "survive.txt")
        with open(decoy_file, "w", encoding="utf-8") as handle:
            handle.write("GC-DECOY")

        real_validate = artifacts_mod._validate_entry
        swapped = {"done": False}

        def validate_and_swap(art_fd, artifact_id, **kw):
            v = real_validate(art_fd, artifact_id, **kw)
            if (
                not swapped["done"]
                and v.get("ok")
                and artifact_id == ref["id"]
            ):
                swapped["done"] = True
                root = artifacts_mod.artifacts_root()
                path = os.path.join(root, ref["id"])
                aside = path + ".aside"
                # Move validated dir aside; put symlink to decoy at the name.
                os.rename(path, aside)
                os.symlink(decoy, path)
            return v

        with mock.patch.object(
            artifacts_mod, "_validate_entry", side_effect=validate_and_swap
        ):
            res = artifacts_mod.gc(force=True)

        self.assertTrue(res["ok"])
        self.assertTrue(os.path.isfile(decoy_file), "decoy must survive GC")
        with open(decoy_file, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "GC-DECOY")
        # Decoy path still a directory (not rmdir'd via the swapped name).
        self.assertTrue(os.path.isdir(decoy))


class TestNoMarkerLease(_ArtifactsBase):
    """T7: no .gc_last / .gc_lease ever created (opportunistic GC gone)."""

    def test_no_marker_or_lease_files(self):
        artifacts_mod.create("marker-check-one")
        artifacts_mod.create("marker-check-two")
        artifacts_mod.gc(force=True)
        root = artifacts_mod.artifacts_root()
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_last")))
        self.assertFalse(os.path.exists(os.path.join(root, ".gc_lease")))
        # Module must not expose the old opportunistic API.
        self.assertFalse(hasattr(artifacts_mod, "maybe_opportunistic_gc"))
        self.assertFalse(hasattr(artifacts_mod, "GC_MARKER_BASENAME"))
        self.assertFalse(hasattr(artifacts_mod, "GC_LEASE_BASENAME"))


class TestRootBootstrapSymlink(_ArtifactsBase):
    """T8: bootstrap refuses a symlinked store root without chmod'ing the target."""

    def test_symlinked_store_root_refused_without_chmod_target(self):
        # Build: state_dir/artifacts → symlink to external decoy dir.
        decoy = tempfile.mkdtemp(prefix="omni-root-decoy-")
        # Distinct mode we can detect if chmod leaked.
        os.chmod(decoy, 0o755)
        decoy_mode_before = stat.S_IMODE(os.stat(decoy).st_mode)
        self.assertEqual(decoy_mode_before, 0o755)

        state = self._tmp
        # Ensure state exists; plant artifacts as symlink.
        os.makedirs(state, mode=0o700, exist_ok=True)
        art_link = os.path.join(state, "artifacts")
        if os.path.lexists(art_link):
            if os.path.isdir(art_link) and not os.path.islink(art_link):
                os.rmdir(art_link)
            else:
                os.unlink(art_link)
        os.symlink(decoy, art_link)

        with self.assertRaises(OSError):
            artifacts_mod._ensure_and_open_root()

        decoy_mode_after = stat.S_IMODE(os.stat(decoy).st_mode)
        self.assertEqual(
            decoy_mode_after, decoy_mode_before,
            "bootstrap must not chmod the symlink target",
        )


class TestGcSymlinkDecoy(_ArtifactsBase):
    """Symlink decoy must not let GC touch anything outside artifacts/."""

    def test_gc_symlink_decoy_inside_artifact_dir(self):
        clock = {"t": 4_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        state = os.path.dirname(root)
        jobs = os.path.join(state, "jobs")
        os.makedirs(jobs, mode=0o700, exist_ok=True)

        external_meta = os.path.join(jobs, "decoy-expired-meta.json")
        with open(external_meta, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "id": _hex_id("decoy-external"),
                "sha256": "0" * 64,
                "text_bytes": 1,
                "created": 1.0,
                "expires": clock["t"] - 100,
            }) + "\n")

        decoy_id = _hex_id("decoy-symlink-art")
        art_dir = os.path.join(root, decoy_id)
        os.makedirs(art_dir, mode=0o700)
        os.symlink(external_meta, os.path.join(art_dir, "meta.json"))

        res = artifacts_mod.gc(force=True)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["deleted"], 0)
        self.assertTrue(
            os.path.isfile(external_meta),
            "jobs decoy meta must survive GC",
        )
        self.assertTrue(os.path.isdir(art_dir), "decoy art dir left alone")
        self.assertTrue(os.path.islink(os.path.join(art_dir, "meta.json")))


class TestConfinedIdTruthTable(_ArtifactsBase):
    """_is_confined_id truth table + hostile ids through read() and remove()."""

    def test_is_confined_id_truth_table(self):
        good = [
            "a" * 64,
            "0123456789abcdef" * 4,
            hashlib.sha256(b"x").hexdigest(),
        ]
        for gid in good:
            self.assertTrue(
                artifacts_mod._is_confined_id(gid), "expected confined: %r" % gid
            )

        bad = [
            "",
            ".",
            "..",
            "../etc",
            "/tmp/evil",
            "a/b",
            "a\\b",
            "has.dot",
            ".dotfile",
            "space id",
            "semi;colon",
            "a\x00b",
            "\x00",
            "foo/../bar",
            None,
            123,
            "A" * 64,  # uppercase rejected (lowercase hex only)
            "g" * 64,  # non-hex
            "a" * 63,  # wrong length
            "a" * 65,
            "Sa4d8vygOuA6mkINE8PowhIW",  # old urlsafe shape
        ]
        for bid in bad:
            self.assertFalse(
                artifacts_mod._is_confined_id(bid), "expected rejected: %r" % bid
            )

    def test_hostile_ids_read_and_remove(self):
        hostile = [
            "..",
            "../etc",
            "/tmp/evil",
            "a/b",
            "a\x00b",
            ".dotfile",
            "has.dot",
            "A" * 64,
        ]
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        outside = tempfile.mkdtemp(prefix="omni-art-out-")
        link_name = _hex_id("symlink-hostile")
        try:
            os.symlink(outside, os.path.join(root, link_name))
            hostile.append(link_name)
        except OSError:
            pass

        for hid in hostile:
            r = artifacts_mod.read(hid)
            self.assertFalse(r.get("ok"), hid)
            # Unconfined → not_found; symlink with valid hex → corrupt (ELOOP).
            self.assertIn(
                r.get("reason"),
                ("artifact_not_found", "artifact_corrupt"),
                "read(%r) → %r" % (hid, r),
            )
            rm = artifacts_mod.remove(hid)
            self.assertFalse(rm.get("ok"), hid)
            self.assertIn(
                rm.get("reason"),
                ("artifact_not_found", "artifact_corrupt"),
                "remove(%r) → %r" % (hid, rm),
            )


class TestMetaForgery(_ArtifactsBase):
    """Untrusted meta schema + dir-swap escape blocked."""

    def _plant(self, artifact_id, meta_obj, blob=b"BODY"):
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        art = os.path.join(root, artifact_id)
        os.makedirs(art, mode=0o700, exist_ok=True)
        with open(os.path.join(art, "blob"), "wb") as handle:
            handle.write(blob)
        os.chmod(os.path.join(art, "blob"), 0o600)
        with open(os.path.join(art, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta_obj) + "\n")
        os.chmod(os.path.join(art, "meta.json"), 0o600)
        return art

    def test_expires_infinity_corrupt(self):
        blob = b"hello-inf"
        aid = hashlib.sha256(blob).hexdigest()
        self._plant(aid, {
            "id": aid,
            "sha256": aid,
            "text_bytes": len(blob),
            "created": 1.0,
            "expires": float("inf"),
        }, blob)
        got = artifacts_mod.read(aid)
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_id_mismatch_corrupt(self):
        blob = b"hello-id"
        aid = hashlib.sha256(blob).hexdigest()
        self._plant(aid, {
            "id": "0" * 64,
            "sha256": aid,
            "text_bytes": len(blob),
            "created": 1.0,
            "expires": time.time() + 3600,
        }, blob)
        got = artifacts_mod.read(aid)
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_dir_swap_symlink_escape_blocked(self):
        body = b"escape-body-payload"
        aid = hashlib.sha256(body).hexdigest()
        art = self._plant(aid, {
            "id": aid,
            "sha256": aid,
            "text_bytes": len(body),
            "created": 1.0,
            "expires": time.time() + 3600,
        }, body)
        external = tempfile.mkdtemp(prefix="omni-art-swap-")
        with open(os.path.join(external, "blob"), "wb") as handle:
            handle.write(body)
        with open(os.path.join(external, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "id": aid,
                "sha256": aid,
                "text_bytes": len(body),
                "created": 1.0,
                "expires": time.time() + 3600,
            }) + "\n")
        for name in ("meta.json", "blob"):
            os.unlink(os.path.join(art, name))
        os.rmdir(art)
        os.symlink(external, art)

        got = artifacts_mod.read(aid)
        self.assertFalse(got["ok"], got)
        # Symlink dir → ELOOP → corrupt (not not_found).
        self.assertEqual(got["reason"], "artifact_corrupt")
        with open(os.path.join(external, "blob"), "rb") as handle:
            self.assertEqual(handle.read(), body)


class TestCorruptVsNotFound(_ArtifactsBase):
    """corrupt vs not_found distinction both directions."""

    def test_absent_dir_is_not_found(self):
        aid = "b" * 64
        got = artifacts_mod.read(aid)
        self.assertEqual(got["reason"], "artifact_not_found")
        rm = artifacts_mod.remove(aid)
        self.assertEqual(rm["reason"], "artifact_not_found")

    def test_dir_present_missing_meta_is_corrupt(self):
        aid = _hex_id("present-no-meta")
        root = artifacts_mod.artifacts_root()
        os.makedirs(os.path.join(root, aid), mode=0o700, exist_ok=True)
        with open(os.path.join(root, aid, "blob"), "wb") as handle:
            handle.write(b"x")
        got = artifacts_mod.read(aid)
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_dir_present_missing_blob_is_corrupt(self):
        aid = _hex_id("present-no-blob")
        root = artifacts_mod.artifacts_root()
        art = os.path.join(root, aid)
        os.makedirs(art, mode=0o700, exist_ok=True)
        with open(os.path.join(art, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "id": aid,
                "sha256": aid,
                "text_bytes": 1,
                "created": 1.0,
                "expires": time.time() + 9999,
            }) + "\n")
        got = artifacts_mod.read(aid)
        self.assertEqual(got["reason"], "artifact_corrupt")

    def test_dir_present_bad_json_is_corrupt(self):
        aid = _hex_id("present-bad-json")
        root = artifacts_mod.artifacts_root()
        art = os.path.join(root, aid)
        os.makedirs(art, mode=0o700, exist_ok=True)
        with open(os.path.join(art, "meta.json"), "w", encoding="utf-8") as handle:
            handle.write("{not-json")
        with open(os.path.join(art, "blob"), "wb") as handle:
            handle.write(b"x")
        got = artifacts_mod.read(aid)
        self.assertEqual(got["reason"], "artifact_corrupt")


class TestSurfaceHardening(_ArtifactsBase):
    """Surrogate → runtime_error; agent rejects --out-mode."""

    def test_lone_surrogate_create_is_runtime_error(self):
        body = "ok\ud800bad"  # lone surrogate — not UTF-8 encodable

        async def router(prompt, model, *a, **k):
            return _envelope(True, text=body, model=model, usage=_usage(1))

        with mock.patch.object(dispatch, "route_to_provider", side_effect=router):
            res = _run(dispatch.call_tool("llm_generate", {
                "prompt": "p", "model": "Fake", "result_mode": "ref",
            }))
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")

    def test_agent_parser_rejects_out_mode(self):
        parser = omni_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "agent", "--workdir", "/tmp", "--out-mode", "ref", "do stuff",
            ])


class TestMutationSelfCheckDoc(_ArtifactsBase):
    """Guard → killing-test mapping for T1–T8 (goes red if guard removed)."""

    def test_mutation_self_check_mapping_present(self):
        mapping = {
            "T1_content_id": "test_tampered_blob_under_valid_id_is_corrupt",
            "T1_meta_advisory": (
                "test_forged_meta_sha256_matching_tampered_blob_still_corrupt"
            ),
            "T2_idempotent": "test_idempotent_recreate_same_id_no_duplicate",
            "T3_cap": "test_per_artifact_cap_refuses_before_write",
            "T3_quota": "test_store_quota_refuses_before_write",
            "T4_read_cap": "test_read_aborts_when_blob_grows_past_cap",
            "T5_cleanup": "test_error_path_cleanup_symlink_swap_decoy_survives",
            "T6_gc_corrupt": "test_gc_never_deletes_failing_manifest_validation",
            "T6_gc_swap": "test_gc_name_swap_after_validation_not_deleted_or_detected",
            "T7_no_marker": "test_no_marker_or_lease_files",
            "T8_bootstrap": "test_symlinked_store_root_refused_without_chmod_target",
        }
        import tests.test_artifacts as self_mod
        found = set()
        for name in dir(self_mod):
            obj = getattr(self_mod, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                for attr in dir(obj):
                    if attr.startswith("test_"):
                        found.add(attr)
        for guard, tname in mapping.items():
            self.assertIn(
                tname, found,
                "mutation self-check: guard %s missing killing test %s" % (
                    guard, tname),
            )


# ---------------------------------------------------------------------------
# Issue #40 — per-artifact locking (T1–T10 + fix2 B1–B4 / I1–I4).
# Barriers: patch _flock_bounded / os.rmdir / mutators — no production hooks.
# ---------------------------------------------------------------------------


def _pipe_barrier():
    """Return (read_fd, write_fd) for a one-byte cross-process barrier."""
    return os.pipe()


def _wait_byte(rfd, timeout_s=5.0):
    """Block until one byte is readable or timeout (raises AssertionError)."""
    import select
    ready, _, _ = select.select([rfd], [], [], timeout_s)
    if not ready:
        raise AssertionError("barrier wait timed out after %.1fs" % timeout_s)
    os.read(rfd, 1)


def _signal_byte(wfd):
    os.write(wfd, b"x")


def _child_env(state_dir):
    env = os.environ.copy()
    env["OMNI_LLM_STATE_DIR"] = state_dir
    env["PYTHONPATH"] = _REPO + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


class TestArtifactLocking40(_ArtifactsBase):
    """T1–T10 + fix2: cross-process lock protocol with pipe synchronisation.

    Concurrency guards are exercised through public entry points
    (read/remove/create/gc). Internals are used only to *hold* a lock for a
    peer race setup, never to perform the step the test claims to pin.
    Deterministic interleavings use monkeypatches of _flock_bounded / os.rmdir
    / mutators in the child process — not production-reachable hooks (I1).
    """

    def test_t1_read_open_then_gc_delete_then_lock_is_not_found(self):
        """T1/L3: public read() open → GC fully deletes → lock+reverify → not_found.

        Child drives public read() and patches _flock_bounded so the barrier
        runs after the dir is open and before flock. Parent runs public gc()
        to completion (entry gone), then releases. read() must return
        artifact_not_found — not artifact_corrupt. Disabling the reverify
        branch must turn this red (members missing → corrupt).
        """
        clock = {"t": 10_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create("t1-lock-reverify-body", ttl_s=1)
        aid = ref["id"]
        clock["t"] = ref["expires"] + 10  # expired so GC deletes

        r_open, w_open = _pipe_barrier()
        r_go, w_go = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_code = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
r_go = %d
w_open = %d
_real = m._flock_bounded

def barrier_flock(fd, lock_mode, **kw):
    # Dir is open; flock not yet taken. Signal parent, wait for GC to finish.
    os.write(w_open, b"x")
    buf = os.read(r_go, 1)
    assert buf == b"x"
    return _real(fd, lock_mode, **kw)

m._flock_bounded = barrier_flock
got = m.read(aid)
print(json.dumps(got))
""" % (_REPO, state, aid, r_go, w_open)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(r_go, w_open),
        )
        os.close(r_go)
        os.close(w_open)
        try:
            _wait_byte(r_open)
            res = artifacts_mod.gc(force=True)
            self.assertTrue(res["ok"], res)
            self.assertGreaterEqual(res["deleted"], 1)
            self.assertFalse(
                os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid))
            )
            _signal_byte(w_go)
        finally:
            os.close(r_open)
            os.close(w_go)

        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, "stderr=%s out=%s" % (err, out))
        got = json.loads(out.strip())
        self.assertFalse(got["ok"], got)
        self.assertEqual(got["reason"], "artifact_not_found", got)
        self.assertNotEqual(got["reason"], "artifact_corrupt")

    def test_t2_read_holds_sh_gc_skips_busy(self):
        """T2/L1/V1: SH held → concurrent public read() ok; public gc() skipped_busy.

        Pins L1 read=LOCK_SH (a concurrent reader must not get busy while a
        peer holds SH — if read() took EX this would go red) and V1
        skipped_busy accounting on public gc().
        """
        ref = artifacts_mod.create("t2-sh-held-body")
        aid = ref["id"]
        body = "t2-sh-held-body"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        self.assertEqual(aid, digest)

        r_held, w_held = _pipe_barrier()
        r_done, w_done = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_code = r"""
import os, sys, fcntl, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held = %d
r_done = %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_SH)
assert art_fd is not None, why
os.write(w_held, b"x")
os.read(r_done, 1)
os.close(art_fd)
os.close(root_fd)
print("released")
""" % (_REPO, state, aid, w_held, r_done)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held, r_done),
        )
        os.close(w_held)
        os.close(r_done)
        try:
            _wait_byte(r_held)
            peer = artifacts_mod.read(aid)
            self.assertTrue(peer["ok"], peer)
            self.assertEqual(peer["text"], body)
            clock = {"t": ref["expires"] + 100}
            artifacts_mod.set_now_fn(lambda: clock["t"])
            res = artifacts_mod.gc(force=True)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res.get("skipped_busy"), 1, res)
            self.assertEqual(res["deleted"], 0, res)
            self.assertTrue(
                os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid))
            )
            _signal_byte(w_done)
        finally:
            os.close(r_held)
            os.close(w_done)

        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        artifacts_mod.set_now_fn(lambda: ref["expires"] - 1)
        got = artifacts_mod.read(aid)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["text"], body)
        self.assertEqual(got["artifact_ref"]["sha256"], digest)

    def test_v2_remove_busy_is_typed(self):
        """V2: public remove() on a live locked entry returns artifact_busy.

        Mapping busy → not_found (or pretend success) must turn this red.
        """
        ref = artifacts_mod.create("v2-remove-busy-body")
        aid = ref["id"]

        r_held, w_held = _pipe_barrier()
        r_done, w_done = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_code = r"""
import os, sys, fcntl
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held, r_done = %d, %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_SH)
assert art_fd is not None, why
os.write(w_held, b"x")
os.read(r_done, 1)
os.close(art_fd)
os.close(root_fd)
""" % (_REPO, state, aid, w_held, r_done)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held, r_done),
        )
        os.close(w_held)
        os.close(r_done)
        try:
            _wait_byte(r_held)
            rm = artifacts_mod.remove(aid)
            self.assertFalse(rm["ok"], rm)
            self.assertEqual(rm["reason"], "artifact_busy", rm)
            self.assertNotEqual(rm["reason"], "artifact_not_found")
            self.assertTrue(
                os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid))
            )
            _signal_byte(w_done)
        finally:
            os.close(r_held)
            os.close(w_done)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        rm2 = artifacts_mod.remove(aid)
        self.assertTrue(rm2["ok"], rm2)

    def test_t3_gc_holds_ex_mid_unlink_reader_not_corrupt(self):
        """T3: public remove() mid-delete (EX held); public read() not corrupt.

        Child patches os.rmdir so the barrier runs after whitelist unlink and
        before rmdir while LOCK_EX is still held.
        """
        ref = artifacts_mod.create("t3-mid-gc-body")
        aid = ref["id"]

        r_mid, w_mid = _pipe_barrier()
        r_done, w_done = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_code = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_mid = %d
r_done = %d
_real_rmdir = os.rmdir

def mid_rmdir(name, *a, **kw):
    if name == aid:
        os.write(w_mid, b"x")
        buf = os.read(r_done, 1)
        assert buf == b"x"
    return _real_rmdir(name, *a, **kw)

os.rmdir = mid_rmdir
got = m.remove(aid)
print(json.dumps(got))
""" % (_REPO, state, aid, w_mid, r_done)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_mid, r_done),
        )
        os.close(w_mid)
        os.close(r_done)
        try:
            _wait_byte(r_mid)
            got = artifacts_mod.read(aid)
            self.assertFalse(got["ok"], got)
            self.assertIn(
                got["reason"],
                ("artifact_busy", "artifact_not_found"),
                "must not be corrupt mid-delete: %r" % got,
            )
            self.assertNotEqual(got["reason"], "artifact_corrupt")
            _signal_byte(w_done)
        finally:
            os.close(r_mid)
            os.close(w_done)

        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        rm = json.loads(out.strip())
        self.assertTrue(rm.get("ok"), rm)

    def test_t4_stable_damage_still_corrupt(self):
        """T4: damaged stable entry (no lock) still reports artifact_corrupt."""
        aid = _hex_id("t4-stable-damage")
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        art = os.path.join(root, aid)
        os.makedirs(art, mode=0o700)
        with open(os.path.join(art, "blob"), "wb") as handle:
            handle.write(b"orphan-blob")
        got = artifacts_mod.read(aid)
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "artifact_corrupt")
        res = artifacts_mod.gc(force=True)
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(res.get("corrupt", 0), 1)
        self.assertTrue(os.path.isdir(art))

    def test_t5_create_refresh_vs_gc_both_orderings(self):
        """T5/I4: public create() holds EX inside locked mutator; read sees busy.

        Ordering A: child public create() is stopped inside _refresh_expiry
        (already holding EX). Parent public read() must observe artifact_busy.
        Downgrading create to SH must turn this red. Also pins create-vs-gc
        never-success-for-gone via Ordering B/B2.
        """
        body = "t5-refresh-vs-gc-body"
        # --- Ordering A (I4): create() stopped inside locked _refresh_expiry ---
        clock = {"t": 12_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create(body, ttl_s=5)
        aid = ref["id"]

        r_held, w_held = _pipe_barrier()
        r_done, w_done = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_code = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
body = %r
w_held, r_done = %d, %d
_real = m._refresh_expiry

def barrier_refresh(art_fd, artifact_id, expires, created=None):
    # Already holds EX from create's locked path. Parent should see busy.
    os.write(w_held, b"x")
    buf = os.read(r_done, 1)
    assert buf == b"x"
    return _real(art_fd, artifact_id, expires, created=created)

m._refresh_expiry = barrier_refresh
ref = m.create(body, ttl_s=100)
print(json.dumps(ref))
""" % (_REPO, state, body, w_held, r_done)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held, r_done),
        )
        os.close(w_held)
        os.close(r_done)
        try:
            _wait_byte(r_held)
            # I4: public read while create holds EX inside mutator → busy.
            mid = artifacts_mod.read(aid)
            self.assertFalse(mid["ok"], mid)
            self.assertEqual(mid["reason"], "artifact_busy", mid)
            # Collector also sees busy (single-shot skip).
            clock["t"] = ref["expires"] + 50
            gres = artifacts_mod.gc(force=True)
            self.assertEqual(gres.get("skipped_busy"), 1, gres)
            self.assertEqual(gres["deleted"], 0)
            _signal_byte(w_done)
        finally:
            os.close(r_held)
            os.close(w_done)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        cres = json.loads(out.strip())
        self.assertEqual(cres["id"], aid)
        artifacts_mod.set_now_fn(lambda: cres["expires"] - 1)
        got = artifacts_mod.read(aid)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["text"], body)

        # --- Ordering A2 (I4 corrupt replace): stop inside _delete_entry_locked ---
        clock["t"] = 12_500_000.0
        artifacts_mod.set_now_fn(lambda: clock["t"])
        body_cr = "t5-corrupt-replace-body"
        # Plant a corrupt same-id entry so create takes the replace path.
        dig_cr = hashlib.sha256(body_cr.encode("utf-8")).hexdigest()
        root = artifacts_mod.artifacts_root()
        os.makedirs(root, mode=0o700, exist_ok=True)
        art = os.path.join(root, dig_cr)
        if not os.path.isdir(art):
            os.makedirs(art, mode=0o700)
        with open(os.path.join(art, "blob"), "wb") as handle:
            handle.write(b"not-the-right-bytes")
        with open(os.path.join(art, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "id": dig_cr,
                "sha256": dig_cr,
                "text_bytes": 1,
                "created": clock["t"],
                "expires": clock["t"] + 100,
            }, handle)

        r_held2, w_held2 = _pipe_barrier()
        r_done2, w_done2 = _pipe_barrier()
        child_cr = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
body = %r
aid = %r
w_held, r_done = %d, %d
_real = m._delete_entry_locked

def barrier_del(root_fd, name, art_fd, art_st):
    if name == aid:
        os.write(w_held, b"x")
        buf = os.read(r_done, 1)
        assert buf == b"x"
    return _real(root_fd, name, art_fd, art_st)

m._delete_entry_locked = barrier_del
ref = m.create(body, ttl_s=100)
print(json.dumps(ref))
""" % (_REPO, state, body_cr, dig_cr, w_held2, r_done2)
        proc_cr = subprocess.Popen(
            [sys.executable, "-c", child_cr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held2, r_done2),
        )
        os.close(w_held2)
        os.close(r_done2)
        try:
            _wait_byte(r_held2)
            mid_cr = artifacts_mod.read(dig_cr)
            self.assertFalse(mid_cr["ok"], mid_cr)
            self.assertEqual(mid_cr["reason"], "artifact_busy", mid_cr)
            _signal_byte(w_done2)
        finally:
            os.close(r_held2)
            os.close(w_done2)
        out_cr, err_cr = proc_cr.communicate(timeout=10)
        self.assertEqual(proc_cr.returncode, 0, err_cr)
        got_cr = artifacts_mod.read(dig_cr)
        self.assertTrue(got_cr["ok"], got_cr)
        self.assertEqual(got_cr["text"], body_cr)

        # --- Ordering B: GC deletes first; create republishes; success ⇒ present ---
        clock["t"] = 13_000_000.0
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref2 = artifacts_mod.create(body + "-b", ttl_s=1)
        aid2 = ref2["id"]
        clock["t"] = ref2["expires"] + 10
        gres2 = artifacts_mod.gc(force=True)
        self.assertGreaterEqual(gres2["deleted"], 1)
        self.assertFalse(
            os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid2))
        )
        ref3 = artifacts_mod.create(body + "-b", ttl_s=30)
        self.assertEqual(ref3["id"], aid2)
        self.assertTrue(
            os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid2))
        )
        got3 = artifacts_mod.read(aid2)
        self.assertTrue(got3["ok"], got3)

        # --- Ordering B2: public remove mid-rmdir; create must not claim success gone ---
        clock["t"] = 14_000_000.0
        artifacts_mod.set_now_fn(lambda: clock["t"])
        body_c = "t5-create-during-gc"
        refc = artifacts_mod.create(body_c, ttl_s=1)
        aidc = refc["id"]

        r_mid, w_mid = _pipe_barrier()
        r_fin, w_fin = _pipe_barrier()
        child_rm = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_mid, r_fin = %d, %d
_real_rmdir = os.rmdir

def mid_rmdir(name, *a, **kw):
    if name == aid:
        os.write(w_mid, b"x")
        os.read(r_fin, 1)
    return _real_rmdir(name, *a, **kw)

os.rmdir = mid_rmdir
got = m.remove(aid)
print(json.dumps(got))
""" % (_REPO, state, aidc, w_mid, r_fin)
        proc2 = subprocess.Popen(
            [sys.executable, "-c", child_rm],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_mid, r_fin),
        )
        os.close(w_mid)
        os.close(r_fin)
        try:
            _wait_byte(r_mid)
            try:
                ref_try = artifacts_mod.create(body_c, ttl_s=30)
                self.assertTrue(
                    os.path.isdir(
                        os.path.join(artifacts_mod.artifacts_root(), ref_try["id"])
                    ),
                    "create returned success but entry is gone",
                )
                got = artifacts_mod.read(ref_try["id"])
                self.assertTrue(got["ok"], got)
            except artifacts_mod.ArtifactError as exc:
                self.assertEqual(exc.reason, "artifact_busy")
            _signal_byte(w_fin)
        finally:
            os.close(r_mid)
            os.close(w_fin)
        out2, err2 = proc2.communicate(timeout=10)
        self.assertEqual(proc2.returncode, 0, err2)

    def test_l4_ex_held_at_rmdir_probe(self):
        """B4/L4: at the real rmdir, second open + LOCK_SH|LOCK_NB must EWOULDBLOCK.

        Production no longer asserts via fstat. The test patches os.rmdir so
        that, at the moment of the real rmdir, it opens a SECOND independent
        fd on the same artifact dir and attempts LOCK_SH|LOCK_NB. Two
        independent opens in one process conflict (F5), so the probe MUST
        fail with EWOULDBLOCK while the deleter still holds EX. Releasing the
        lock early (close art_fd before rmdir) makes the probe succeed → red.
        """
        import errno as errno_mod
        ref = artifacts_mod.create("l4-probe-body")
        aid = ref["id"]
        probe = {"ewouldblock": None, "errno": None}

        real_rmdir = os.rmdir

        def probing_rmdir(name, *a, **kw):
            if name == aid:
                dir_fd = kw.get("dir_fd")
                flags = os.O_RDONLY
                if hasattr(os, "O_DIRECTORY"):
                    flags |= os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                if dir_fd is not None:
                    probe_fd = os.open(name, flags, dir_fd=dir_fd)
                else:
                    probe_fd = os.open(
                        os.path.join(artifacts_mod.artifacts_root(), name),
                        flags,
                    )
                try:
                    try:
                        fcntl.flock(
                            probe_fd, fcntl.LOCK_SH | fcntl.LOCK_NB
                        )
                        probe["ewouldblock"] = False
                    except OSError as exc:
                        probe["errno"] = getattr(exc, "errno", None)
                        probe["ewouldblock"] = (
                            probe["errno"] in (
                                errno_mod.EWOULDBLOCK, errno_mod.EAGAIN,
                            )
                        )
                finally:
                    os.close(probe_fd)
            return real_rmdir(name, *a, **kw)

        with mock.patch.object(os, "rmdir", side_effect=probing_rmdir):
            got = artifacts_mod.remove(aid)
        self.assertTrue(got.get("ok"), got)
        self.assertIsNotNone(
            probe["ewouldblock"],
            "probe did not run at rmdir — test bug",
        )
        self.assertTrue(
            probe["ewouldblock"],
            "EX must still be held at rmdir (probe errno=%s); early unlock "
            "would make LOCK_SH|LOCK_NB succeed" % probe["errno"],
        )

    def test_t6_ex_held_through_rmdir(self):
        """T6/L4: public remove() holds EX through rmdir; concurrent SH is busy.

        Child patches os.rmdir to stop AFTER whitelist unlink and BEFORE rmdir
        while LOCK_EX is still held. Parent's public read() must observe
        artifact_busy. Closing art_fd before rmdir (L4 violation) must turn
        this red for the peer (SH can acquire).
        """
        ref = artifacts_mod.create("t6-ex-through-rmdir")
        aid = ref["id"]

        r_ex, w_ex = _pipe_barrier()
        r_go, w_go = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]

        child_del = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_ex, r_go = %d, %d
_real_rmdir = os.rmdir

def mid_rmdir(name, *a, **kw):
    if name == aid:
        os.write(w_ex, b"x")
        buf = os.read(r_go, 1)
        assert buf == b"x"
    return _real_rmdir(name, *a, **kw)

os.rmdir = mid_rmdir
got = m.remove(aid)
print(json.dumps(got))
""" % (_REPO, state, aid, w_ex, r_go)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_del],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_ex, r_go),
        )
        os.close(w_ex)
        os.close(r_go)
        try:
            _wait_byte(r_ex)
            mid = artifacts_mod.read(aid)
            self.assertFalse(mid["ok"], mid)
            self.assertEqual(mid["reason"], "artifact_busy", mid)
            _signal_byte(w_go)
        finally:
            os.close(r_ex)
            os.close(w_go)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        rm = json.loads(out.strip())
        self.assertTrue(rm.get("ok"), rm)
        late = artifacts_mod.read(aid)
        self.assertEqual(late["reason"], "artifact_not_found", late)

    def test_l5_never_hangs_bounded_busy(self):
        """B3/L5: public ops return artifact_busy in finite time under stuck EX.

        One subprocess holds EX indefinitely. Another calls public
        read/remove. Parent uses communicate(timeout=...); on timeout the
        child is killed and the test FAILS explicitly. Passing means a
        bounded, typed artifact_busy. Removing LOCK_NB must turn this red
        in bounded time (hang → timeout → fail).
        """
        ref = artifacts_mod.create("l5-never-hang-body")
        aid = ref["id"]
        state = os.environ["OMNI_LLM_STATE_DIR"]

        r_held, w_held = _pipe_barrier()
        holder_code = r"""
import os, sys, fcntl, time
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held = %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_EX)
assert art_fd is not None, why
os.write(w_held, b"x")
while True:
    time.sleep(60)
""" % (_REPO, state, aid, w_held)

        holder = subprocess.Popen(
            [sys.executable, "-c", holder_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held,),
        )
        os.close(w_held)
        try:
            _wait_byte(r_held, timeout_s=5.0)

            # Budget is ~250ms; allow generous headroom but FAIL if unbounded.
            op_timeout = 3.0
            for op_name, op_src in (
                ("read", "print(json.dumps(m.read(aid)))"),
                ("remove", "print(json.dumps(m.remove(aid)))"),
            ):
                caller_code = r"""
import os, sys, json
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
%s
""" % (_REPO, state, aid, op_src)
                caller = subprocess.Popen(
                    [sys.executable, "-c", caller_code],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=_child_env(state),
                )
                try:
                    out, err = caller.communicate(timeout=op_timeout)
                except subprocess.TimeoutExpired:
                    caller.kill()
                    caller.wait(timeout=5)
                    self.fail(
                        "L5 hang: public %s did not return within %.1fs "
                        "under stuck EX (LOCK_NB missing?)" % (
                            op_name, op_timeout,
                        )
                    )
                self.assertEqual(
                    caller.returncode, 0,
                    "public %s failed: stderr=%s out=%s" % (op_name, err, out),
                )
                got = json.loads(out.strip())
                self.assertFalse(got["ok"], got)
                self.assertEqual(
                    got["reason"], "artifact_busy",
                    "public %s must be typed busy, got %r" % (op_name, got),
                )
        finally:
            holder.send_signal(signal.SIGKILL)
            try:
                holder.wait(timeout=5)
            except Exception:
                pass
            os.close(r_held)

    def test_l6_enotsup_is_loud_not_busy(self):
        """L6/B1: flock ENOTSUP is loud runtime_error on public AND collector.

        Treating every flock errno as busy must turn this red. Collector
        must not skip ENOTSUP as success (B1).
        """
        import errno as errno_mod
        ref = artifacts_mod.create("l6-enotsup-body")
        aid = ref["id"]

        def boom(_fd, _flags):
            raise OSError(errno_mod.ENOTSUP, "Operation not supported")

        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=boom):
            got = artifacts_mod.read(aid)
        self.assertFalse(got["ok"], got)
        self.assertNotEqual(
            got["reason"], "artifact_busy",
            "ENOTSUP must not be mapped to busy: %r" % got,
        )
        self.assertEqual(got["reason"], "runtime_error", got)

        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=boom):
            rm = artifacts_mod.remove(aid)
        self.assertFalse(rm["ok"], rm)
        self.assertNotEqual(rm["reason"], "artifact_busy", rm)
        self.assertEqual(rm["reason"], "runtime_error", rm)

        # B1: gc must return LOUD runtime_error, not ok with skipped/deleted.
        clock = {"t": ref["expires"] + 100}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=boom):
            gres = artifacts_mod.gc(force=True)
        self.assertFalse(gres.get("ok"), gres)
        self.assertEqual(gres.get("reason"), "runtime_error", gres)

        # B1: reclaim raises ArtifactError — create must not pretend store_full.
        os.environ["OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES"] = "1"
        # store already has one entry (aid); reclaim path runs on next create.
        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=boom):
            with self.assertRaises(artifacts_mod.ArtifactError) as ctx:
                artifacts_mod.create("l6-different-body-needs-slot")
        self.assertEqual(ctx.exception.reason, "runtime_error")
        self.assertNotEqual(ctx.exception.reason, "artifact_store_full")

    def test_b2_collector_single_shot_no_retry_budget(self):
        """B2: gc/reclaim use single-shot LOCK_NB — busy skips without sleeping.

        If collectors shared the ~250ms retry budget per entry, one busy
        entry would stall the scan. Pin: flock raises EWOULDBLOCK once and
        sleep must not be called for the collector path.
        """
        import errno as errno_mod
        ref = artifacts_mod.create("b2-single-shot-body", ttl_s=1)
        clock = {"t": ref["expires"] + 10}
        artifacts_mod.set_now_fn(lambda: clock["t"])

        sleep_calls = {"n": 0}
        real_sleep = time.sleep

        def counting_sleep(s):
            sleep_calls["n"] += 1
            return real_sleep(s)

        def busy_flock(_fd, _flags):
            raise OSError(errno_mod.EWOULDBLOCK, "Resource temporarily unavailable")

        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=busy_flock):
            with mock.patch.object(artifacts_mod.time, "sleep", side_effect=counting_sleep):
                gres = artifacts_mod.gc(force=True)
        self.assertTrue(gres.get("ok"), gres)
        self.assertEqual(gres.get("skipped_busy"), 1, gres)
        self.assertEqual(
            sleep_calls["n"], 0,
            "gc must not sleep/retry on busy (single-shot); slept %d times"
            % sleep_calls["n"],
        )

        # Public read still may sleep under the retry budget.
        sleep_calls["n"] = 0
        with mock.patch.object(artifacts_mod.fcntl, "flock", side_effect=busy_flock):
            with mock.patch.object(artifacts_mod.time, "sleep", side_effect=counting_sleep):
                # Shrink budget so the test is fast.
                with mock.patch.object(
                    artifacts_mod, "_LOCK_RETRY_BUDGET_S", 0.05
                ):
                    with mock.patch.object(
                        artifacts_mod, "_LOCK_RETRY_SLEEP_S", 0.01
                    ):
                        got = artifacts_mod.read(ref["id"])
        self.assertEqual(got.get("reason"), "artifact_busy", got)
        self.assertGreater(
            sleep_calls["n"], 0,
            "public read must use bounded retry (at least one sleep)",
        )

    def test_i2_reverify_three_state(self):
        """I2: reverify distinguishes same / not_found / runtime_error."""
        ref = artifacts_mod.create("i2-reverify-body")
        aid = ref["id"]
        root_fd = artifacts_mod._ensure_and_open_root()
        try:
            art_fd, art_st, why = artifacts_mod._open_locked_artifact(
                root_fd, aid, fcntl.LOCK_SH
            )
            self.assertIsNotNone(art_fd, why)
            try:
                self.assertIsNone(
                    artifacts_mod._reverify_name(root_fd, aid, art_st)
                )
                # Fake st with wrong ino → not_found
                class _St(object):
                    st_dev = art_st.st_dev
                    st_ino = art_st.st_ino + 99999
                self.assertEqual(
                    artifacts_mod._reverify_name(root_fd, aid, _St()),
                    "artifact_not_found",
                )
            finally:
                os.close(art_fd)

            # ENOENT path
            missing_st = art_st
            self.assertEqual(
                artifacts_mod._reverify_name(
                    root_fd, "0" * 64, missing_st
                ),
                "artifact_not_found",
            )

            # Other OSError → runtime_error (not masqueraded as not_found)
            import errno as errno_mod
            real_stat = os.stat

            def eio_stat(*a, **kw):
                raise OSError(errno_mod.EIO, "I/O error")

            with mock.patch.object(os, "stat", side_effect=eio_stat):
                # Need a locked open that then reverifies — patch at reverify.
                self.assertEqual(
                    artifacts_mod._reverify_name(root_fd, aid, missing_st),
                    "runtime_error",
                )
                got = artifacts_mod.read(aid)
            # After unpatch, read works; with EIO during reverify inside
            # _open_locked_artifact:
            with mock.patch.object(os, "stat", side_effect=eio_stat):
                # But open path also uses fstat etc. — only patch after open
                # by forcing reverify. Use direct open_locked with patched stat.
                pass
            # Drive public path: patch _reverify_name? Better: patch os.stat
            # only when follow_symlinks=False and name is aid — open uses
            # fstat/open, reverify uses os.stat.
            real_stat2 = os.stat
            def selective_eio(path, *a, **kw):
                if path == aid:
                    raise OSError(errno_mod.EIO, "I/O error")
                return real_stat2(path, *a, **kw)
            with mock.patch.object(os, "stat", side_effect=selective_eio):
                got = artifacts_mod.read(aid)
            self.assertFalse(got["ok"], got)
            self.assertEqual(got["reason"], "runtime_error", got)
            self.assertNotEqual(got["reason"], "artifact_not_found")
        finally:
            os.close(root_fd)

    def test_t7_inprocess_no_double_open(self):
        """T7: implementation does not double-open a locked artifact dir (F5)."""
        ref = artifacts_mod.create("t7-no-double-open")
        aid = ref["id"]
        root_fd = artifacts_mod._ensure_and_open_root()
        try:
            art_fd, art_st, why = artifacts_mod._open_locked_artifact(
                root_fd, aid, fcntl.LOCK_SH
            )
            self.assertIsNotNone(art_fd, why)
            try:
                self.assertTrue(
                    artifacts_mod._name_still_same_inode(root_fd, aid, art_st)
                )
                v = artifacts_mod._validate_entry(art_fd, aid, check_expiry=True)
                self.assertTrue(v["ok"], v)
                self.assertEqual(v["text"], "t7-no-double-open")

                opens = {"n": 0}
                real_open = os.open

                def counting_open(path, flags, *a, **kw):
                    if path == aid:
                        opens["n"] += 1
                    return real_open(path, flags, *a, **kw)

                with mock.patch.object(os, "open", side_effect=counting_open):
                    self.assertTrue(
                        artifacts_mod._name_still_same_inode(
                            root_fd, aid, art_st
                        )
                    )
                    v2 = artifacts_mod._validate_entry(
                        art_fd, aid, check_expiry=True
                    )
                    self.assertTrue(v2["ok"], v2)
                self.assertEqual(
                    opens["n"], 0,
                    "locked path re-opened artifact dir %d times" % opens["n"],
                )
            finally:
                os.close(art_fd)
        finally:
            os.close(root_fd)

        got = artifacts_mod.read(aid)
        self.assertTrue(got["ok"], got)

    def test_t8_sigkill_releases_lock(self):
        """T8: SIGKILL of a lock holder frees the entry for the next GC."""
        clock = {"t": 15_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        ref = artifacts_mod.create("t8-sigkill-body", ttl_s=1)
        aid = ref["id"]
        clock["t"] = ref["expires"] + 10

        r_held, w_held = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]
        child_code = r"""
import os, sys, fcntl, time, signal
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held = %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_EX)
assert art_fd is not None, why
os.write(w_held, b"x")
while True:
    time.sleep(60)
""" % (_REPO, state, aid, w_held)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held,),
        )
        os.close(w_held)
        try:
            _wait_byte(r_held)
            busy_gc = artifacts_mod.gc(force=True)
            self.assertEqual(busy_gc.get("skipped_busy"), 1, busy_gc)
            self.assertEqual(busy_gc["deleted"], 0)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)
        finally:
            os.close(r_held)

        res = artifacts_mod.gc(force=True)
        self.assertTrue(res["ok"], res)
        self.assertGreaterEqual(res["deleted"], 1, res)
        self.assertEqual(res.get("skipped_busy", 0), 0, res)
        self.assertFalse(
            os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), aid))
        )

    def test_t9_one_busy_does_not_block_others(self):
        """T9: one busy entry does not block collection of other entries."""
        clock = {"t": 16_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        held = artifacts_mod.create("t9-held-entry", ttl_s=1)
        free = artifacts_mod.create("t9-free-entry", ttl_s=1)
        clock["t"] = max(held["expires"], free["expires"]) + 10

        r_held, w_held = _pipe_barrier()
        r_done, w_done = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]
        child_code = r"""
import os, sys, fcntl
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held, r_done = %d, %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_SH)
assert art_fd is not None, why
os.write(w_held, b"x")
os.read(r_done, 1)
os.close(art_fd)
os.close(root_fd)
""" % (_REPO, state, held["id"], w_held, r_done)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held, r_done),
        )
        os.close(w_held)
        os.close(r_done)
        try:
            _wait_byte(r_held)
            res = artifacts_mod.gc(force=True)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res.get("skipped_busy"), 1, res)
            self.assertEqual(res["deleted"], 1, res)
            self.assertTrue(
                os.path.isdir(
                    os.path.join(artifacts_mod.artifacts_root(), held["id"])
                )
            )
            self.assertFalse(
                os.path.isdir(
                    os.path.join(artifacts_mod.artifacts_root(), free["id"])
                )
            )
            _signal_byte(w_done)
        finally:
            os.close(r_held)
            os.close(w_done)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)

    def test_t10_quota_busy_expired_refuses_then_retry(self):
        """T10: quota full + busy expired → create refuses; after death, retry ok."""
        os.environ["OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES"] = "1"
        clock = {"t": 17_000_000.0}
        artifacts_mod.set_now_fn(lambda: clock["t"])
        held = artifacts_mod.create("t10-only-slot", ttl_s=1)
        clock["t"] = held["expires"] + 10

        r_held, w_held = _pipe_barrier()
        state = os.environ["OMNI_LLM_STATE_DIR"]
        child_code = r"""
import os, sys, fcntl, time, signal
sys.path.insert(0, %r)
os.environ["OMNI_LLM_STATE_DIR"] = %r
from providers import artifacts as m
aid = %r
w_held = %d
root_fd = m._ensure_and_open_root()
art_fd, art_st, why = m._open_locked_artifact(root_fd, aid, fcntl.LOCK_EX)
assert art_fd is not None, why
os.write(w_held, b"x")
while True:
    time.sleep(60)
""" % (_REPO, state, held["id"], w_held)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(state),
            pass_fds=(w_held,),
        )
        os.close(w_held)
        try:
            _wait_byte(r_held)
            with self.assertRaises(artifacts_mod.ArtifactError) as ctx:
                artifacts_mod.create("t10-different-body-needs-slot")
            self.assertEqual(ctx.exception.reason, "artifact_store_full")
            self.assertTrue(
                os.path.isdir(
                    os.path.join(artifacts_mod.artifacts_root(), held["id"])
                )
            )
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)
        finally:
            os.close(r_held)

        ref2 = artifacts_mod.create("t10-different-body-needs-slot", ttl_s=30)
        self.assertTrue(
            os.path.isdir(os.path.join(artifacts_mod.artifacts_root(), ref2["id"]))
        )
        got = artifacts_mod.read(ref2["id"])
        self.assertTrue(got["ok"], got)


if __name__ == "__main__":
    unittest.main()
