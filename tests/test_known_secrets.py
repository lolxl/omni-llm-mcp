#!/usr/bin/env python3
# no-opencc
"""Issue #160: credential values omni itself hands to a child are masked verbatim.

The key-shape patterns guess; omni knows the values it puts into a child's
environment and the tokens in the child's login files. These tests drive the
registry (providers.known_secrets), the stream-log writer in run_cli, every
provider failure path through the MCP entry point and the CLI, the job stream
files, ``agent tail --follow``, the log lines and the spill/--out writers.

Hermetic: the "vendor" is a small script of this file that prints the
credential it finds in its own environment; nothing leaves the machine. Every
credential is synthetic and built at runtime; none is a real key.
"""
import asyncio
import base64
import contextlib
import io
import json
import os
import random
import stat
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock
from urllib.parse import quote, quote_plus

import omni_cli
import server
from providers import (agy, artifacts, claude, cli_run, codex, cursor, errors, grok,
                       grok_image_cli, job_runner, jobs, known_secrets, logutil, pi)
from tests.test_egress_redaction import Env, capture_cli, capture_server
from tests import offline_home

# No key-shaped prefix: only the known-value layer can catch these.
PLAIN = "mZ4qT9vL" + "2xW7pR5n" + "K8dJ3hB6"


def _seed_catalog(provider, model_id, efforts=None):
    """A v2 row so the real gate reaches the vendor stub."""
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    item = {
        "id": model_id,
        "name": model_id,
        "routable_id": "%s/%s" % (provider, model_id),
    }
    if efforts is not None:
        item["supported_efforts"] = list(efforts)
    record, _warnings = validate_strict(provider, {
        "provider": provider,
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [item],
    })
    catalog_v2.save(provider, "default", record)


def _seed_grok_image():
    """Image editing admits grok-4.7 before it registers a child environment."""
    from providers.grok_image_cli import GROK_IMAGE_MODEL
    _seed_catalog("grok", GROK_IMAGE_MODEL, ["medium"])


def _seed_synthetic_lanes():
    """Consult failure tests use the id ``synthetic`` on the real gate."""
    medium = ["low", "medium", "high"]
    _seed_catalog("claude", "synthetic", medium)
    _seed_catalog("codex", "synthetic", medium)
    _seed_catalog("grok", "synthetic", medium)
    _seed_catalog("cursor", "synthetic")
    _seed_catalog("agy", "synthetic")
# Characters that change under JSON, URL and base64 encoding, and a multi-byte one.
SPECIAL = 'aB3/xY9+kL2"qR7\\éTz5'
FRAGMENT = 8


def leaked(blob, secret, size=FRAGMENT):
    """First ``size``-char slice of ``secret`` found in ``blob`` (str or bytes)."""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    for i in range(len(secret) - size + 1):
        if secret[i:i + size] in blob:
            return secret[i:i + size]
    return None


def forms(value):
    """The raw, JSON-escaped, URL-encoded and base64 spellings of a value."""
    return {
        "raw": value,
        "json": json.dumps(value)[1:-1],
        "json-ascii-off": json.dumps(value, ensure_ascii=False)[1:-1],
        "url": quote(value, safe=""),
        "url-plus": quote_plus(value),
        "basic-auth": base64.b64encode(("svc:" + value).encode()).decode(),
        "basic-auth-urlsafe": base64.urlsafe_b64encode(("u:" + value).encode()).decode(),
    }


def fragments_of(blob, value):
    """Every spelling whose 8-char slice survives in blob."""
    return [name for name, text in forms(value).items() if leaked(blob, text)]


_HOME = None


def setUpModule():
    # Run directly, the fenced spawns below admit ~/.claude.json and the login
    # keychain by their real names, and a CLI session rewriting them flips a
    # test. Point protected_home() at a synthetic home for this module.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


class Registry(unittest.TestCase):
    def setUp(self):
        known_secrets.clear()
        self.addCleanup(known_secrets.clear)


class RegistryTests(Registry):
    def test_every_encoding_is_masked(self):
        known_secrets.register(SPECIAL, authoritative=True)
        for name, text in forms(SPECIAL).items():
            with self.subTest(form=name):
                out = known_secrets.mask_known("prefix %s suffix" % text)
                self.assertIsNone(leaked(out, text), out)
                self.assertIn("<redacted>", out)

    def test_length_floor_is_eight(self):
        self.assertFalse(known_secrets.register("abcdefg", authoritative=True))
        self.assertTrue(known_secrets.register("abcdefgh", authoritative=True))
        self.assertEqual(known_secrets.mask_known("x abcdefgh y"), "x <redacted> y")

    def test_heuristic_values_are_filtered_authoritative_are_not(self):
        for weak in ("aaaaaaaaaaaa", "changeme", "true"):
            self.assertFalse(known_secrets.register(weak), weak)
        self.assertTrue(known_secrets.register("aaaaaaaaaaaa", authoritative=True))
        self.assertTrue(known_secrets.register("changeme", authoritative=True))

    def test_nothing_known_means_no_change(self):
        text = "plain text with Bearer-free words and 12345678"
        self.assertEqual(known_secrets.mask_known(text), text)
        env = {"ok": False, "detail": text, "n": [1, {"x": text}]}
        self.assertEqual(errors.mask_envelope(env), env)

    def test_text_without_a_known_value_is_unchanged_by_redact_secrets(self):
        samples = [
            "claude exited 1: not logged in",
            "Authorization: Basic Zm9vOmJhcg==",
            "refresh_token=" + "q" * 30,
            "ordinary 429 rate limit text " * 5,
        ]
        before = [errors.redact_secrets(s) for s in samples]
        known_secrets.register(PLAIN, authoritative=True)
        self.assertEqual([errors.redact_secrets(s) for s in samples], before)

    def test_cut_front_and_back_of_a_value_are_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        for cut in (8, 12, len(PLAIN) - 1):
            with self.subTest(cut=cut):
                self.assertIsNone(leaked(
                    known_secrets.mask_known("detail: " + PLAIN[:cut]), PLAIN))
                self.assertIsNone(leaked(
                    known_secrets.mask_known(PLAIN[len(PLAIN) - cut:] + " end"), PLAIN))
        # Shorter than the floor: left alone, nothing to learn from it.
        self.assertEqual(known_secrets.mask_known("x " + PLAIN[:7]), "x " + PLAIN[:7])

    def test_environment_names(self):
        values = {name: "v%02d_" % i + PLAIN for i, name in enumerate((
            "XAI_API_KEY", "CURSOR_API_KEY_ALT", "MY_SERVICE_TOKEN", "DB_PASSWORD"))}
        decoys = {"SERVICE_URL": "https://host.example/" + PLAIN,
                  "TOKEN_FILE": "/tmp/token-file-" + PLAIN,
                  "PATH": "/usr/bin:/bin:" + PLAIN, "HOME": "/home/" + PLAIN}
        known_secrets.register_env({**values, **decoys})
        for name, value in values.items():
            self.assertNotIn(value, known_secrets.mask_known("x " + value), name)
        for name, value in decoys.items():
            self.assertEqual(known_secrets.mask_known("x " + value), "x " + value, name)

    def test_overrides_are_authoritative(self):
        known_secrets.register_overrides({"CURSOR_API_KEY": "cccccccccccc",
                                          "CODEX_HOME": "/tmp/codex-home-aaaa"})
        self.assertEqual(known_secrets.mask_known("k=cccccccccccc"), "k=<redacted>")
        self.assertEqual(known_secrets.mask_known("/tmp/codex-home-aaaa"),
                         "/tmp/codex-home-aaaa")

    def test_mask_envelope_covers_every_string(self):
        known_secrets.register(PLAIN, authoritative=True)
        deep = {"x": PLAIN}
        for _ in range(40):
            deep = {"d": [deep]}
        env = {"ok": True, "text": "answer " + PLAIN, "preview": PLAIN,
               "content": [PLAIN], "meta": {"note": PLAIN}, "deep": deep}
        out = errors.mask_envelope(env)
        self.assertIsNone(leaked(json.dumps(out), PLAIN))
        self.assertEqual(errors.mask_envelope("scalar " + PLAIN), "scalar <redacted>")

    def test_log_lines_are_masked_even_when_the_caller_cut_the_value(self):
        known_secrets.register(PLAIN, authoritative=True)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            logutil.log_event("probe", detail=("x" * 40 + PLAIN)[:50], full=PLAIN,
                              nested={"k": [PLAIN]})
            logutil.log("free text", PLAIN)
        self.assertIsNone(leaked(buf.getvalue(), PLAIN), buf.getvalue())
        self.assertIn("probe", buf.getvalue())

    def test_registry_leaves_nothing_behind_and_shows_no_value(self):
        known_secrets.register(PLAIN, authoritative=True)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": tmp}):
                known_secrets.register_child_env({"XAI_API_KEY": PLAIN})
            self.assertEqual([], list(os.scandir(tmp)))
        masker = known_secrets.StreamMasker()
        masker.feed(PLAIN[:5].encode())
        for obj in (masker, str(masker), known_secrets.StreamMasker.__doc__):
            self.assertNotIn(PLAIN[:5], repr(obj) + str(obj))

    def test_a_failing_registration_raises_nothing_and_shows_no_value(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), \
                mock.patch.object(known_secrets, "_rebuild",
                                  side_effect=ValueError("boom " + PLAIN)):
            self.assertFalse(known_secrets.register(PLAIN, authoritative=True))
            known_secrets.register_child_env({"XAI_API_KEY": PLAIN})
        self.assertEqual(buf.getvalue(), "")
        self.assertFalse(known_secrets.active())      # never half-added
        self.assertEqual(known_secrets.mask_known("x " + PLAIN), "x " + PLAIN)

    def test_lone_surrogates_register_in_every_spelling_that_can_be_derived(self):
        escaped = "abcdefgh\udc80ijklmnop"      # one byte under surrogateescape
        lone = "qrstuvwx\ud800yzABCDEFG"        # only surrogatepass can encode it
        self.assertTrue(known_secrets.register_many([escaped, lone], authoritative=True))
        self.assertEqual(known_secrets.mask_known("x %s y %s z" % (escaped, lone)),
                         "x <redacted> y <redacted> z")
        for raw in (escaped.encode("utf-8", "surrogateescape"),
                    lone.encode("utf-8", "surrogatepass")):
            masker = known_secrets.StreamMasker()
            out = masker.feed(b"x " + raw[:6]) + masker.feed(raw[6:] + b" y") + masker.flush()
            self.assertEqual(out, b"x <redacted> y", raw)

    def test_overlapping_values_are_replaced_as_one_span(self):
        first, second = "abcdefgh12345678X", "12345678Xqrstuvwx"
        known_secrets.register_many([first, second], authoritative=True)
        out = known_secrets.mask_known("<abcdefgh12345678Xqrstuvwx>")
        self.assertEqual(out, "<<redacted>>")
        for piece in (first, second):
            self.assertIsNone(leaked(out, piece))
        masker = known_secrets.StreamMasker()
        raw = b"<abcdefgh12345678Xqrstuvwx>"
        got = b"".join(masker.feed(raw[i:i + 3]) for i in range(0, len(raw), 3))
        self.assertEqual(got + masker.flush(), b"<<redacted>>")

    def test_the_longer_of_nested_values_wins(self):
        short, long_ = "ABCDEFGH", "ABCDEFGHijklmnopqrstuvwxyz"
        known_secrets.register_many([short, long_], authoritative=True)
        self.assertEqual(known_secrets.mask_known("a " + long_ + " b"), "a <redacted> b")
        self.assertEqual(known_secrets.mask_known("a " + short + " b"), "a <redacted> b")
        masker = known_secrets.StreamMasker()
        first = masker.feed(b"xx " + short.encode())
        self.assertEqual(first, b"xx ")           # nothing committed while a longer value may follow
        second = masker.feed(long_[8:].encode() + b" yy")
        self.assertEqual(first + second + masker.flush(), b"xx <redacted> yy")
        masker = known_secrets.StreamMasker()
        out = masker.feed(b"xx " + short.encode()) + masker.feed(b" other") + masker.flush()
        self.assertEqual(out, b"xx <redacted> other")

    def test_dict_keys_are_never_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        env = {"ok": True, PLAIN: "value " + PLAIN, "nested": {PLAIN: [PLAIN]}}
        out = errors.mask_envelope(env)
        self.assertIn(PLAIN, out)
        self.assertEqual(out[PLAIN], "value <redacted>")
        self.assertIn(PLAIN, out["nested"])
        self.assertEqual(out["nested"][PLAIN], ["<redacted>"])
        two = {"a" + PLAIN: 1, "b" + PLAIN: 2}
        self.assertEqual(len(errors.mask_envelope(two)), 2)

    def test_log_arguments_are_masked_whatever_their_type(self):
        known_secrets.register(PLAIN, authoritative=True)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            logutil.log("failed:", RuntimeError("boom " + PLAIN), {"k": PLAIN}, 7)
        self.assertIsNone(leaked(buf.getvalue(), PLAIN), buf.getvalue())
        self.assertIn("boom", buf.getvalue())

    def test_protocol_version_echo_is_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        sent = []

        async def fake_send(obj):
            sent.append(obj)

        meta = {server.META_PROTOCOL_VERSION: "v-" + PLAIN,
                server.META_CLIENT_CAPABILITIES: {}}
        with mock.patch.object(server, "_send", fake_send):
            asyncio.run(server._validate_modern_meta(1, {"_meta": meta}))
        self.assertEqual(sent[0]["error"]["code"], -32022)
        self.assertIsNone(leaked(json.dumps(sent), PLAIN), sent)

    def test_profiles_table_output_is_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        rows = [{"road": "claude", "name": "alt", "source": "env",
                 "identity": {"emailAddress": "who@" + PLAIN + ".example"},
                 "state": "ok", "problems": [PLAIN]}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            omni_cli._emit_profiles({"ok": True, "profiles": rows}, as_json=False)
        self.assertIsNone(leaked(out.getvalue(), PLAIN), out.getvalue())


class AuthFileTests(Registry):
    def write(self, directory, name, payload, mode=0o600):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.chmod(path, mode)
        return path

    def test_only_fixed_keys_with_token_shaped_values(self):
        token = "tok" + "A1b2C3d4" * 4
        jwt = "eyJhbGciOi.eyJzdWIiOiJ4.sig" + "Q" * 20
        planted = "the quick brown fox jumps over"
        with tempfile.TemporaryDirectory() as home:
            self.write(home, "auth.json", {
                "OPENAI_API_KEY": None,
                "tokens": {"access_token": token, "id_token": jwt,
                           "refresh_token": "short", "account_id": "acct-" + PLAIN},
                "note": planted, "other": {"apiKey": planted, "label": token + "x"},
            })
            known_secrets.register_auth_files({"CODEX_HOME": home})
        self.assertEqual(known_secrets.mask_known("a " + token), "a <redacted>")
        self.assertEqual(known_secrets.mask_known("b " + jwt), "b <redacted>")
        self.assertEqual(known_secrets.mask_known("c " + planted), "c " + planted)
        self.assertEqual(known_secrets.mask_known("d acct-" + PLAIN), "d acct-" + PLAIN)

    def test_default_home_and_claude_credentials(self):
        token = "cl" + "Zy8Xw7Vu" * 4
        with tempfile.TemporaryDirectory() as home:
            os.makedirs(os.path.join(home, ".claude"))
            self.write(os.path.join(home, ".claude"), ".credentials.json",
                       {"claudeAiOauth": {"accessToken": token, "expiresAt": 1}})
            known_secrets.register_auth_files({"HOME": home})
        self.assertEqual(known_secrets.mask_known("t " + token), "t <redacted>")

    def test_refused_files(self):
        token = "tk" + "Qw3Er5Ty" * 4
        with tempfile.TemporaryDirectory() as home:
            real = self.write(home, "real.json", {"access_token": token})
            os.symlink(real, os.path.join(home, "auth.json"))
            known_secrets.register_auth_files({"CODEX_HOME": home})
            self.assertEqual(known_secrets.mask_known("s " + token), "s " + token)
            os.unlink(os.path.join(home, "auth.json"))
            self.write(home, "auth.json", {"access_token": token, "pad": "x" * 70000})
            known_secrets.register_auth_files({"CODEX_HOME": home})
            self.assertEqual(known_secrets.mask_known("o " + token), "o " + token)
            with open(os.path.join(home, "auth.json"), "w") as handle:
                handle.write("{not json")
            known_secrets.register_auth_files({"CODEX_HOME": home})
            self.assertEqual(known_secrets.mask_known("j " + token), "j " + token)


    def test_exact_locations_only(self):
        token = "tok" + "A1b2C3d4" * 4
        other = "oth" + "E5f6G7h8" * 4
        with tempfile.TemporaryDirectory() as home:
            self.write(home, "auth.json", {
                "nested": {"tokens": {"access_token": token}},     # wrong depth
                "tokens": {"access_token": other, "account_id": "acct-" + PLAIN},
                "access_token": token,                              # wrong place
            })
            known_secrets.register_auth_files({"CODEX_HOME": home})
        self.assertEqual(known_secrets.mask_known("a " + other), "a <redacted>")
        self.assertEqual(known_secrets.mask_known("b " + token), "b " + token)

    def test_grok_and_pi_entries_and_the_shape_rules(self):
        keep = "kk" + "Ab3Cd4Ef" * 4
        refresh = "rr" + "Gh5Ij6Kl" * 4
        pi_access = "pa" + "Mn7Op8Qr" * 4
        slashed = "sl/" + "Rs9Tu0Vw" * 4          # '/' outside a JWT: rejected
        one_class = "a" * 30                         # a single character class
        with tempfile.TemporaryDirectory() as home:
            self.write(home, "auth.json", {
                "api.x.ai": {"key": keep, "refresh_token": refresh, "other": slashed},
                "scalar": slashed, "plain": {"key": one_class}})
            pi_dir = os.path.join(home, ".pi", "agent")
            os.makedirs(pi_dir)
            self.write(pi_dir, "auth.json", {"anthropic": {"type": "oauth",
                                                          "access": pi_access}})
            known_secrets.register_auth_files({"GROK_HOME": home, "HOME": home})
        for token in (keep, refresh, pi_access):
            self.assertEqual(known_secrets.mask_known("t " + token), "t <redacted>")
        for token in (slashed, one_class):
            self.assertEqual(known_secrets.mask_known("n " + token), "n " + token)

    def test_a_file_registers_a_bounded_number_of_tokens(self):
        tokens = ["t%02d" % i + "Zx9Wv8Ut" * 4 for i in range(40)]   # 35 chars, over the 32 floor
        with tempfile.TemporaryDirectory() as home:
            self.write(home, "auth.json", {"p%02d" % i: {"key": t}
                                           for i, t in enumerate(tokens)})
            known_secrets.register_auth_files({"GROK_HOME": home})
        masked = [t for t in tokens
                  if known_secrets.mask_known("x " + t) == "x <redacted>"]
        self.assertEqual(len(masked), 16)

    def test_a_pathologically_nested_file_is_just_unreadable(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "auth.json"), "w") as handle:
                handle.write("[" * 60000)
            known_secrets.register_auth_files({"CODEX_HOME": home})
            with mock.patch.dict(os.environ, {"CODEX_HOME": home, "HOME": home}, clear=True):
                known_secrets.prime()                     # must not raise
        self.assertFalse(known_secrets.active())

    def test_a_replaced_file_of_the_same_size_and_time_is_read_again(self):
        first = "fi" + "Ab3Cd4Ef" * 4
        second = "se" + "Gh5Ij6Kl" * 4
        self.assertEqual(len(first), len(second))
        with tempfile.TemporaryDirectory() as home:
            path = self.write(home, "auth.json", {"tokens": {"access_token": first}})
            stamp = os.stat(path).st_mtime_ns
            known_secrets.register_auth_files({"CODEX_HOME": home})
            replacement = self.write(home, "new.json", {"tokens": {"access_token": second}})
            os.utime(replacement, ns=(stamp, stamp))
            os.replace(replacement, path)                 # new inode, same size, same mtime
            known_secrets.register_auth_files({"CODEX_HOME": home})
        self.assertEqual(known_secrets.mask_known("1 " + first), "1 <redacted>")
        self.assertEqual(known_secrets.mask_known("2 " + second), "2 <redacted>")


class StreamMaskerTests(Registry):
    def run_masker(self, data, sizes):
        masker = known_secrets.StreamMasker()
        out, pos = b"", 0
        for size in sizes:
            out += masker.feed(data[pos:pos + size])
            pos += size
        out += masker.feed(data[pos:])
        return out + masker.flush()

    def test_value_split_at_every_position(self):
        known_secrets.register(SPECIAL, authoritative=True)
        for name, text in forms(SPECIAL).items():
            data = ("lead " + text + " tail\n").encode("utf-8")
            for cut in range(1, len(data)):
                out = self.run_masker(data, [cut])
                self.assertIsNone(leaked(out, text), (name, cut, out))
                self.assertTrue(out.startswith(b"lead ") and out.endswith(b" tail\n"),
                                (name, cut, out))

    def test_one_byte_chunks_and_multibyte_split(self):
        known_secrets.register(SPECIAL, authoritative=True)
        data = ("x " + SPECIAL + " y").encode("utf-8")
        out = self.run_masker(data, [1] * len(data))
        self.assertEqual(out, b"x <redacted> y")

    def test_unfinished_prefix_at_end_is_replaced_not_emitted(self):
        known_secrets.register(PLAIN, authoritative=True)
        out = self.run_masker(("end " + PLAIN[:14]).encode(), [6])
        self.assertIsNone(leaked(out, PLAIN), out)
        self.assertTrue(out.startswith(b"end "))

    def test_short_look_alike_tail_is_released_at_flush(self):
        known_secrets.register(PLAIN, authoritative=True)
        out = self.run_masker(("see " + PLAIN[:3]).encode(), [5])
        self.assertEqual(out, ("see " + PLAIN[:3]).encode())

    def test_without_values_data_passes_through_untouched(self):
        masker = known_secrets.StreamMasker()
        self.assertEqual(masker.feed(b"abc"), b"abc")
        self.assertEqual(masker.flush(), b"")


FAKE_CHILD = textwrap.dedent('''\
    #!%(python)s
    import json, os, sys, time
    cred = os.environ[os.environ["FAKE_CRED_VAR"]]
    out, err = sys.stdout.buffer, sys.stderr.buffer
    if os.environ.get("FAKE_JSONL"):
        for event in ({"type": "error", "message": "auth failed " + cred},
                      {"type": "turn.failed", "error": {"message": cred}}):
            out.write((json.dumps(event) + "\\n").encode()); out.flush()
    out.write(("stdout: " + cred + "\\n").encode()); out.flush()
    err.write(("auth failed for " + cred + "\\n").encode()); err.flush()
    err.write((json.dumps({"error": {"message": cred}}) + "\\n").encode()); err.flush()
    half = len(cred) // 2
    err.write(b"split-a " + cred[:half].encode("utf-8")); err.flush()
    time.sleep(0.25)
    err.write(cred[half:].encode("utf-8") + b"\\n"); err.flush()
    out.write(b"split-b " + cred[:half].encode("utf-8")); out.flush()
    time.sleep(0.25)
    out.write(cred[half:].encode("utf-8") + b"\\n"); out.flush()
    sys.exit(1)
''')


class FakeVendor(Registry):
    """A script standing in for a vendor binary that echoes its own credential."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory(prefix="omni-160-")
        self.addCleanup(tmp.cleanup)
        self.tmp = os.path.realpath(tmp.name)
        self.binary = os.path.join(self.tmp, "vendor")
        with open(self.binary, "w", encoding="utf-8") as handle:
            handle.write(FAKE_CHILD % {"python": sys.executable})
        os.chmod(self.binary, stat.S_IRWXU)

    def child_env(self, var, value=PLAIN):
        env = {"PATH": os.environ.get("PATH", ""), var: value, "FAKE_CRED_VAR": var}
        return env


class RunCliStreamLogTests(FakeVendor):
    def run_it(self, env, **kwargs):
        return asyncio.run(cli_run.run_cli(
            [self.binary], env=env, timeout_s=30, **kwargs))

    def test_split_and_escaped_credential_never_reaches_a_log_file(self):
        for var, value in (("XAI_API_KEY", PLAIN), ("MY_SERVICE_TOKEN", SPECIAL)):
            with self.subTest(var=var):
                known_secrets.clear()
                raw = os.path.join(self.tmp, "stream.raw.log")
                err = os.path.join(self.tmp, "stream.err.log")
                merged = os.path.join(self.tmp, "merged.log")
                res = self.run_it(self.child_env(var, value), log_path=raw,
                                  err_log_path=err)
                self.run_it(self.child_env(var, value), log_path=merged)
                self.assertEqual(res["returncode"], 1)
                # The caller still gets the raw bytes it must parse.
                self.assertIn(value, res["stdout"])
                for path in (raw, err, merged):
                    with open(path, "rb") as handle:
                        data = handle.read()
                    self.assertTrue(data, path)
                    self.assertEqual(fragments_of(data, value), [], path)
                    self.assertIn(b"<redacted>", data)

    def test_normalizer_and_tail_see_only_masked_files(self):
        job_dir = os.path.join(self.tmp, "job")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "job.json"), "w") as handle:
            json.dump({"provider": "claude", "state": "running"}, handle)
        raw = os.path.join(job_dir, "stream.raw.log")
        err = os.path.join(job_dir, "stream.err.log")
        self.run_it(self.child_env("XAI_API_KEY"), log_path=raw, err_log_path=err)
        normalizer = job_runner._StreamNormalizer(job_dir, "claude")
        normalizer.poll_once()
        normalizer.final_drain()
        # A separate process (the tail command) has an empty registry: the
        # files themselves must already be clean.
        known_secrets.clear()
        for name in ("stream.raw.log", "stream.err.log", "stream.log"):
            with open(os.path.join(job_dir, name), "rb") as handle:
                self.assertEqual(fragments_of(handle.read(), PLAIN), [], name)
        for kwargs in ({}, {"raw": True}, {"err": True}):
            shown = omni_cli._mask_log_text(jobs.tail(job_dir, lines=50, **kwargs))
            self.assertIsNone(leaked(shown, PLAIN), kwargs)

    def test_tail_follow_masks_a_file_written_by_something_else(self):
        job_dir = os.path.join(self.tmp, "job2")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "job.json"), "w") as handle:
            json.dump({"state": "running"}, handle)
        path = os.path.join(job_dir, "stream.raw.log")
        known_secrets.register(PLAIN, authoritative=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("seed line\nold " + PLAIN + "\nhalf " + PLAIN[:11])
        out = io.StringIO()
        polls = []

        def fake_sleep(_seconds):
            polls.append(1)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(PLAIN[11:] + "\n" + json.dumps({"k": PLAIN}) + "\n")
            if len(polls) > 2:
                raise KeyboardInterrupt

        with contextlib.redirect_stdout(out), \
                mock.patch.object(omni_cli.time, "sleep", fake_sleep):
            code = omni_cli._tail_follow(job_dir, raw=True, n=10)
        self.assertEqual(code, 0)
        self.assertIsNone(leaked(out.getvalue(), PLAIN), out.getvalue())
        self.assertIn("seed line", out.getvalue())

    def test_header_with_the_command_line_is_masked(self):
        finished = []
        env = self.child_env("XAI_API_KEY")
        with mock.patch.object(cli_run, "watch_enabled", return_value=True), \
                mock.patch.object(cli_run, "resolve_zellij_session", return_value="s"), \
                mock.patch.object(cli_run, "resolve_zellij_tab_id", return_value=1), \
                mock.patch.object(cli_run, "_open_watch_pane",
                                  new=mock.AsyncMock(return_value="p")), \
                mock.patch.object(cli_run, "_schedule_watch_finish",
                                  side_effect=lambda s, p, work: finished.append(work)):
            asyncio.run(cli_run.run_cli(
                [self.binary, "--token", PLAIN], env=env, timeout_s=30))
        self.assertEqual(len(finished), 1)
        self.addCleanup(__import__("shutil").rmtree, finished[0], True)
        with open(os.path.join(finished[0], "stream.log"), "rb") as handle:
            data = handle.read()
        self.assertIn(b"watch", data)
        self.assertEqual(fragments_of(data, PLAIN), [])


class ProviderFailurePaths(FakeVendor, Env):
    """The vendor prints its own credential on failure; nothing of it escapes."""

    CASES = (
        # provider, module, model, credential variable, binary hook
        ("claude", claude, "claude/synthetic", "ANTHROPIC_API_KEY"),
        ("grok", grok, "grok/synthetic", "XAI_API_KEY"),
        ("cursor", cursor, "cursor/synthetic", "CURSOR_API_KEY"),
        ("pi", pi, "pi/synthetic", "OMNI_TEST_SERVICE_TOKEN"),
        ("agy", agy, "agy/synthetic", "OMNI_TEST_SERVICE_TOKEN"),
        ("codex", codex, "codex/synthetic", "CODEX_API_KEY"),
    )

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        # Env patches os.environ (cleared); FakeVendor makes the script.
        Env.setUp(self)
        FakeVendor.setUp(self)
        _seed_synthetic_lanes()

    def binary_patches(self, mod):
        stack = contextlib.ExitStack()
        if hasattr(mod, "resolve_bin"):
            stack.enter_context(mock.patch.object(
                mod, "resolve_bin", return_value=self.binary))
        if hasattr(mod, "shutil"):
            stack.enter_context(mock.patch.object(
                mod.shutil, "which", return_value=self.binary))
        if hasattr(mod, "vendor_bin"):      # agy looks the binary up through it
            stack.enter_context(mock.patch.object(
                mod.vendor_bin, "which", return_value=self.binary))
        return stack

    def test_mcp_and_cli_output_hold_no_fragment(self):
        for name, mod, model, var in self.CASES:
            for value in (PLAIN, SPECIAL):
                with self.subTest(provider=name, value=value[:4]):
                    known_secrets.clear()
                    with mock.patch.dict(os.environ, dict(
                        {var: value, "FAKE_CRED_VAR": var},
                        **({"FAKE_JSONL": "1"} if name == "codex" else {}))), \
                            self.binary_patches(mod):
                        result = capture_server(model)
                        known_secrets.clear()
                        code, cli_out, cli_err = capture_cli(
                            ["gen", "-m", model, "--timeout", "30", "synthetic"])
                    structured = result["structuredContent"]
                    self.assertIs(structured.get("ok"), False, structured)
                    # Not vacuous: the child really ran and its text was masked.
                    self.assertIn("<redacted>", structured.get("detail", ""), structured)
                    blob = json.dumps(result, ensure_ascii=False)
                    for where, text in (("mcp", blob), ("cli stdout", cli_out),
                                        ("cli stderr", cli_err)):
                        self.assertEqual(fragments_of(text, value), [],
                                         "%s leaks in %s" % (name, where))
                    self.assertEqual(code, 1)

    def test_success_text_is_masked_too(self):
        known_secrets.register(PLAIN, authoritative=True)
        env = {"ok": True, "text": "the key is " + PLAIN, "model": "m"}
        out = errors.mask_envelope(env)
        self.assertEqual(out["text"], "the key is <redacted>")


class WriterTests(Registry):
    def test_out_file_is_masked_before_it_is_written(self):
        known_secrets.register(PLAIN, authoritative=True)
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "answer.txt")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = omni_cli._emit({"ok": True, "text": "answer " + PLAIN},
                                      out=target)
            self.assertEqual(code, 0)
            with open(target, encoding="utf-8") as handle:
                body = handle.read()
            self.assertEqual(body, "answer <redacted>")
            self.assertIsNone(leaked(out.getvalue(), PLAIN))

    def test_spilled_artifact_and_preview_are_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        with tempfile.TemporaryDirectory() as state:
            with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state}):
                env = artifacts.apply_result_mode(
                    {"ok": True, "text": "head " + PLAIN + " " + "z" * 5000}, "ref")
                self.assertIsNone(leaked(json.dumps(env), PLAIN), env)
                stored = artifacts.read(env["artifact_ref"]["id"])
            self.assertIsNone(leaked(json.dumps(stored, default=str), PLAIN))
            for root, _dirs, files in os.walk(state):
                for name in files:
                    with open(os.path.join(root, name), "rb") as handle:
                        self.assertEqual(fragments_of(handle.read(), PLAIN), [], name)


HOLD_CHILD = textwrap.dedent('''\
    #!%(python)s
    import os, sys, time
    cred = os.environ[os.environ["FAKE_CRED_VAR"]]
    sys.stdout.buffer.write(b"start " + cred[:12].encode()); sys.stdout.buffer.flush()
    sys.stderr.buffer.write(b"err " + cred[:11].encode()); sys.stderr.buffer.flush()
    time.sleep(30)
''')


class HeldPrefixAtTheEnd(FakeVendor):
    """A value prefix is in the hold-back buffer when the run ends abnormally."""

    def setUp(self):
        super().setUp()
        self.hold = os.path.join(self.tmp, "hold")
        with open(self.hold, "w", encoding="utf-8") as handle:
            handle.write(HOLD_CHILD % {"python": sys.executable})
        os.chmod(self.hold, stat.S_IRWXU)

    def check_logs(self, raw, err):
        for path in (raw, err):
            with open(path, "rb") as handle:
                data = handle.read()
            self.assertEqual(fragments_of(data, PLAIN), [], (path, data))
        with open(raw, "rb") as handle:
            self.assertIn(b"start ", handle.read())

    def test_timeout(self):
        raw = os.path.join(self.tmp, "raw.log")
        err = os.path.join(self.tmp, "err.log")
        res = asyncio.run(cli_run.run_cli(
            [self.hold], env=self.child_env("XAI_API_KEY"), timeout_s=1,
            log_path=raw, err_log_path=err))
        self.assertTrue(res["timed_out"])
        self.check_logs(raw, err)

    def test_cancel(self):
        raw = os.path.join(self.tmp, "raw.log")
        err = os.path.join(self.tmp, "err.log")

        async def go():
            task = asyncio.ensure_future(cli_run.run_cli(
                [self.hold], env=self.child_env("XAI_API_KEY"), timeout_s=30,
                log_path=raw, err_log_path=err))
            await asyncio.sleep(1.0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(go())
        self.check_logs(raw, err)


class OtherExits(Registry):
    def test_result_json_and_plain_output(self):
        known_secrets.register(PLAIN, authoritative=True)
        with tempfile.TemporaryDirectory() as job_dir:
            for envelope in ({"ok": False, "reason": "runtime_error",
                              "detail": "died: " + PLAIN, "error": {"text": PLAIN}},
                             {"ok": True, "text": "answer " + PLAIN, "model": "m"}):
                path = os.path.join(job_dir, "result.json")
                if os.path.exists(path):
                    os.unlink(path)
                jobs._write_result(job_dir, dict(envelope))
                with open(path, "rb") as handle:
                    self.assertEqual(fragments_of(handle.read(), PLAIN), [])
                for plain in (True, False):
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        omni_cli._emit(dict(envelope), plain=plain)
                    self.assertIsNone(leaked(out.getvalue() + err.getvalue(), PLAIN))

    def test_grok_image_edit_registers_its_own_spawn(self):
        proc = mock.Mock(returncode=1, stdout="", stderr="denied for %s\n" % PLAIN)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {
                    "XAI_API_KEY": PLAIN, "OMNI_LLM_STATE_DIR": tmp}), \
                mock.patch.object(grok_image_cli.shutil, "which", return_value="/bin/true"), \
                mock.patch.object(grok_image_cli, "refuse_payload", return_value=None), \
                mock.patch.object(grok_image_cli, "prepare_host_fence",
                                  side_effect=lambda argv, env, **kw: (argv, env, {})), \
                mock.patch.object(grok_image_cli.subprocess, "run", return_value=proc):
            _seed_grok_image()
            res = asyncio.run(grok_image_cli.run_grok_image_edit(
                "p", os.path.join(tmp, "src.png"), out_dir=tmp))
        self.assertIs(res["ok"], False, res)
        self.assertIsNone(leaked(json.dumps(res), PLAIN), res)
        self.assertIn("<redacted>", res["detail"])


class UnseparatedRuns(Registry):
    """Back-to-back occurrences must neither pile up in the carry nor stall."""

    CASES = (
        ("doubled", "aB3cD4e5" * 2, "aB3cD4e5"),
        ("one-char", "xxxxxxxx", "x"),
        ("periodic-overlap", "abcabcabcabc", "abc"),
    )

    def stream(self, value, unit, megabytes=1):
        known_secrets.register(value, authoritative=True)
        masker = known_secrets.StreamMasker()
        data = (unit.encode() * ((megabytes << 20) // len(unit) + 3))[:(megabytes << 20) + 3]
        bound = 2 * known_secrets._max_blen
        out, first_output_at, worst = [], None, 0
        started = time.perf_counter()
        for n, pos in enumerate(range(0, len(data), 4096)):
            out.append(masker.feed(data[pos:pos + 4096]))
            worst = max(worst, len(masker._carry))
            self.assertLess(len(masker._carry), bound, "carry grew at chunk %d" % n)
            if first_output_at is None and any(out):
                first_output_at = n
        before_flush = sum(len(chunk) for chunk in out)
        out.append(masker.flush())
        return b"".join(out), before_flush, first_output_at, worst, time.perf_counter() - started

    def test_runs_flow_through_with_a_bounded_carry(self):
        for name, value, unit in self.CASES:
            with self.subTest(case=name):
                known_secrets.clear()
                joined, before_flush, first, worst, elapsed = self.stream(value, unit)
                self.assertGreater(before_flush, 0)
                self.assertLessEqual(first, 2)           # output starts at once
                self.assertLess(worst, 2 * known_secrets._max_blen)
                self.assertLess(elapsed, 30.0)          # generous: about 20x this machine
                self.assertIsNone(leaked(joined, value), joined[:80])
                self.assertIn(b"<redacted>", joined)
                # Only markers and (for the periodic case) < 8 loose chars remain.
                rest = joined.replace(b"<redacted>", b"")
                self.assertLess(len(rest), 8, rest[:40])

    def test_the_run_is_still_matched_when_a_chunk_ends_inside_it(self):
        known_secrets.register("aB3cD4e5" * 2, authoritative=True)
        masker = known_secrets.StreamMasker()
        data = b"head " + b"aB3cD4e5" * 5 + b" tail"
        for size in (1, 3, 7, 8, 11, 16, 17):
            masker = known_secrets.StreamMasker()
            out = b"".join(masker.feed(data[i:i + size]) for i in range(0, len(data), size))
            out += masker.flush()
            self.assertRegex(out, rb"\Ahead (<redacted>)+ tail\Z", (size, out))
            self.assertIsNone(leaked(out, "aB3cD4e5" * 2), (size, out))


class NestedValuesInAStream(Registry):
    """A short value inside a longer one must not let the rest of the longer
    one out when the stream is cut at the inner match."""

    CASES = (
        ("inner-in-the-middle", ["abcdefgh" + "SHORTTOK" + "ijklmnopa", "SHORTTOK"],
         "abcdefghSHORTTOKijklmnopa"),
        ("inner-then-a-prefix-tail",
         ["Ab1Cd2Ef" + "Qk3mZ9pL" + "Tx7Wy8VuRr4" + "Qk3m", "Qk3mZ9pL"],
         "Ab1Cd2Ef" + "Qk3mZ9pL" + "Tx7Wy8VuRr4" + "Qk3m"),
        # Ends in its own first character, so the end of the input is held back
        # and the run really crosses the cut.
        ("inner-twice", ["PRE12345" + "Zx8Yw7Vu" + "MID67890" + "Zx8Yw7Vu" + "END5432P",
                         "Zx8Yw7Vu"],
         "PRE12345" + "Zx8Yw7Vu" + "MID67890" + "Zx8Yw7Vu" + "END5432P"),
    )

    def run_split(self, data, cuts):
        masker = known_secrets.StreamMasker()
        out, pos = b"", 0
        for cut in cuts:
            out += masker.feed(data[pos:cut])
            pos = cut
        out += masker.feed(data[pos:])
        return out + masker.flush()

    def test_nothing_of_either_value_survives_any_split(self):
        for name, values, text in self.CASES:
            with self.subTest(case=name):
                known_secrets.clear()
                known_secrets.register_many(values, authoritative=True)
                data = ("<" + text + ">").encode()
                splits = [()] + [(i,) for i in range(1, len(data))]
                splits += [(i, j) for i in range(1, len(data), 3)
                           for j in range(i + 1, len(data), 4)]
                for cuts in splits:
                    out = self.run_split(data, cuts)
                    for value in values:
                        self.assertIsNone(leaked(out, value), (name, cuts, out))
                    self.assertRegex(out, rb"\A<(<redacted>)+>\Z", (name, cuts, out))

    def test_every_case_really_crosses_the_cut(self):
        for name, values, text in self.CASES:
            with self.subTest(case=name):
                known_secrets.clear()
                known_secrets.register_many(values, authoritative=True)
                masker = known_secrets.StreamMasker()
                first = masker.feed(("<" + text).encode())
                self.assertGreater(masker._covered, 0)      # coverage is owed
                # (2) a crossing feed followed directly by flush uses it.
                out = first + masker.flush()
                for value in values:
                    self.assertIsNone(leaked(out, value), (name, out))
                self.assertRegex(out, rb"\A<(<redacted>)+\Z", (name, out))

    def mid_stream(self, later, feeds):
        long_, short = "abcdefghSHORTTOKijklmnopa", "SHORTTOK"
        known_secrets.clear()
        known_secrets.register_many([long_, short], authoritative=True)
        masker = known_secrets.StreamMasker()
        out = masker.feed(("<" + long_).encode())
        owed = [masker._covered]
        known_secrets.register(later, authoritative=True)    # e.g. a concurrent spawn
        for chunk in feeds:
            out += masker.feed(chunk)
            owed.append(masker._covered)
        return out + masker.flush(), owed, long_

    def test_a_value_registered_between_feeds_does_not_drop_the_owed_coverage(self):
        out, owed, long_ = self.mid_stream("SHORTTOKijklmnopaMORE", [b"M", b">"])
        self.assertIsNone(leaked(out, long_), out)
        self.assertNotIn(b"ijklmnop", out)
        self.assertRegex(out, rb"\A<(<redacted>)+M>\Z", out)
        self.assertEqual(known_secrets.mask_known("<" + long_ + "M>"), "<<redacted>M>")
        # (3) the debt survives a feed that emits nothing, and is paid at the end.
        self.assertEqual(owed[1], owed[0])
        self.assertGreater(owed[0], 0)
        self.assertEqual(owed[2], 0)

    def test_the_owed_coverage_survives_more_than_one_held_feed(self):
        out, owed, long_ = self.mid_stream("SHORTTOKijklmnopaMORE12",
                                           [b"M", b"O", b"R", b">"])
        self.assertEqual(owed[0], owed[1])
        self.assertEqual(owed[1], owed[2])
        self.assertEqual(owed[2], owed[3])
        self.assertIsNone(leaked(out, long_), out)
        self.assertNotIn(b"ijklmnop", out)
        self.assertRegex(out, rb"\A<(<redacted>)+MOR>\Z", out)

    def test_the_debt_is_paid_by_flush_when_the_stream_ends_while_held(self):
        out, owed, long_ = self.mid_stream("SHORTTOKijklmnopaMORE", [b"M"])
        self.assertGreater(owed[-1], 0)
        self.assertIsNone(leaked(out, long_), out)
        self.assertNotIn(b"ijklmnop", out)

    def test_the_text_path_agrees(self):
        for name, values, text in self.CASES:
            known_secrets.clear()
            known_secrets.register_many(values, authoritative=True)
            self.assertEqual(known_secrets.mask_known("<" + text + ">"),
                             "<<redacted>>", name)



class CostBounds(Registry):
    """Generous bounds (several times what this machine needs): they catch an
    accidental quadratic step, not a slow runner."""

    def test_many_long_values_register_and_mask_in_bounded_time(self):
        rng = random.Random(160)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        tokens = ["eyJ" + "".join(rng.choice(alphabet) for _ in range(700))
                  for _ in range(64)]
        start = time.perf_counter()
        known_secrets.register_many(tokens, authoritative=False)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertGreaterEqual(len(known_secrets._str_patterns), 64 * 3)
        masker = known_secrets.StreamMasker()
        block = (b"some ordinary vendor output line without any secret\n" * 80)[:4096]
        out = 0
        start = time.perf_counter()
        for _ in range(256):                        # 1 MiB in 4 KiB chunks
            out += len(masker.feed(block))
        out += len(masker.flush())
        self.assertLess(time.perf_counter() - start, 6.0)
        self.assertEqual(out, 256 * len(block))
        start = time.perf_counter()
        known_secrets.mask_known(("x" * 1023 + "\n") * 1024)
        self.assertLess(time.perf_counter() - start, 3.0)

    def test_a_sixteen_token_file_registers_quickly(self):
        tokens = ["t%02d" % i + "Zx9Wv8Ut" * 40 for i in range(16)]
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "auth.json"), "w") as handle:
                json.dump({"p%02d" % i: {"key": t} for i, t in enumerate(tokens)}, handle)
            start = time.perf_counter()
            known_secrets.register_auth_files({"GROK_HOME": home})
            elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.5)
        self.assertEqual(known_secrets.mask_known("x " + tokens[15]), "x <redacted>")


class StreamBackHalfAtTheStart(Registry):
    """Issue #171 A and D: the text path masks a cut-off BACK of a value at the
    start of the text; the stream path must do the same at the start of the
    stream and at the start of a buffer whose front it never saw (a value
    registered after the stream began)."""

    def run_split(self, data, cuts):
        masker = known_secrets.StreamMasker()
        out, pos = b"", 0
        for cut in cuts:
            out += masker.feed(data[pos:cut])
            pos = cut
        out += masker.feed(data[pos:])
        return out + masker.flush()

    def test_a_stream_that_starts_with_the_back_half_is_masked(self):
        known_secrets.register(PLAIN, authoritative=True)
        for size in (len(PLAIN) - 1, 16, 8):
            back = PLAIN[-size:]
            data = (back + " tail\n").encode()
            self.assertEqual(known_secrets.mask_known(back + " tail\n"),
                             "<redacted> tail\n")          # the text path agrees
            splits = [()] + [(i,) for i in range(1, len(data))]
            splits += [(i, i + 1) for i in range(1, len(data) - 1)]
            splits += [tuple(range(1, len(data)))]           # one byte per feed
            for cuts in splits:
                out = self.run_split(data, cuts)
                self.assertEqual(out, b"<redacted> tail\n", (size, cuts, out))

    def test_the_floor_is_eight_characters_as_in_the_text_path(self):
        known_secrets.register(PLAIN, authoritative=True)
        data = (PLAIN[-7:] + " tail").encode()
        self.assertEqual(known_secrets.mask_known(data.decode()), data.decode())
        for cuts in ((), (3,), (1, 2, 5)):
            self.assertEqual(self.run_split(data, cuts), data, cuts)

    def test_a_back_half_with_nothing_after_it_is_masked_at_flush(self):
        known_secrets.register(PLAIN, authoritative=True)
        for cuts in ((), (4,), (9,)):
            self.assertEqual(self.run_split(PLAIN[-12:].encode(), cuts),
                             b"<redacted>", cuts)

    def test_a_back_held_for_a_longer_one_is_masked_when_the_stream_ends(self):
        # "Tx7Wy8Vu" ends the first value and starts the tail of the second, so
        # the stream is held after it and flush has to judge the start.
        values = ["Qk3mZ9pL" + "Tx7Wy8Vu", "zzzzzzzz" + "Tx7Wy8Vu" + "Rr4Ss5Tt"]
        known_secrets.register_many(values, authoritative=True)
        for cuts in ((), (3,), (8,), (9,)):
            out = self.run_split(b"Tx7Wy8VuRr4S", cuts)
            self.assertEqual(out, b"<redacted>Rr4S", cuts)

    def test_output_that_is_the_start_of_a_value_tail_waits_for_one_more_chunk(self):
        # The cost of the check: a buffer that is, so far, the start of some
        # registered value's tail (8+ bytes long) is held back, whatever it
        # says. It is released by the next chunk or by flush, and never grows
        # past the longest value.
        known_secrets.register("abcXYZ" + "hello" + "00000000", authoritative=True)
        masker = known_secrets.StreamMasker()
        self.assertEqual(masker.feed(b"hello"), b"")
        self.assertEqual(masker.feed(b" world\n"), b"hello world\n")
        masker = known_secrets.StreamMasker()
        self.assertEqual(masker.feed(b"hello"), b"")
        self.assertEqual(masker.flush(), b"hello")
        # Once something has gone out, later chunks are not held for this.
        masker = known_secrets.StreamMasker()
        masker.feed(b"first\n")
        self.assertEqual(masker.feed(b"hello"), b"hello")

    def test_the_hold_stays_bounded_and_ordinary_output_is_not_delayed(self):
        known_secrets.register(PLAIN, authoritative=True)
        masker = known_secrets.StreamMasker()
        self.assertEqual(masker.feed(b"hello there, ordinary output\n"),
                         b"hello there, ordinary output\n")
        self.assertEqual(masker._carry, b"")
        masker = known_secrets.StreamMasker()
        bound = 2 * known_secrets._max_blen
        out = b""
        for byte in PLAIN[3:-1].encode():          # a back half still growing
            out += masker.feed(bytes([byte]))
            self.assertLess(len(masker._carry), bound)
        self.assertEqual(out, b"")                 # held, not leaked
        out += masker.feed(b"!")                   # no longer a value's back
        self.assertIn(PLAIN[3:-1].encode(), out)   # released unchanged (a look-alike)

    def test_a_value_registered_after_the_stream_began_masks_its_back(self):
        long_, short = "abcdefghSHORTTOKijklmnopa", "SHORTTOK"
        later = long_ + "MORE12345678"
        for feeds in ([b"MORE12345678", b">"], [b"MORE1234", b"5678>"],
                      [b"M", b"ORE12", b"345678", b">"]):
            known_secrets.clear()
            known_secrets.register_many([long_, short], authoritative=True)
            masker = known_secrets.StreamMasker()
            out = masker.feed(("<" + long_).encode())
            known_secrets.register(later, authoritative=True)   # another child
            for chunk in feeds:
                out += masker.feed(chunk)
            out += masker.flush()
            self.assertNotIn(b"MORE1234", out, (feeds, out))
            self.assertIsNone(leaked(out, later), (feeds, out))
            self.assertRegex(out, rb"\A<(<redacted>)+>\Z", (feeds, out))


NON_UTF8_CHILD = textwrap.dedent('''\
    #!%(python)s
    import os, sys
    if "OMNI_TEST_RAW" in os.environ:       # exactly these bytes, on both streams
        raw = bytes.fromhex(os.environ["OMNI_TEST_RAW"])
        sys.stdout.buffer.write(raw)
        sys.stderr.buffer.write(raw)
        sys.exit(int(os.environ.get("OMNI_TEST_RC", "1")))
    cred = os.fsencode(os.environ["OMNI_TEST_SERVICE_TOKEN"])
    cred += bytes.fromhex(os.environ.get("OMNI_TEST_AFTER", ""))
    sys.stdout.buffer.write(b"out: " + cred + b"\\n")
    sys.stderr.buffer.write(b"auth failed for " + cred + b"\\n")
    sys.exit(1)
''')


class NonUtf8Credential(Registry):
    """Issue #171 B: run_cli decodes child output with errors="replace", so a
    credential holding bytes that are not UTF-8 reaches the failure text in a
    U+FFFD spelling. That spelling is registered too."""

    VALUE = "abcdefgh\udc80ijklmnop\udcffqrstuvwx"

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        super().setUp()
        temp = tempfile.TemporaryDirectory(prefix="omni-171-")
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)

    def test_the_replacement_character_spelling_is_masked(self):
        known_secrets.register(self.VALUE, authoritative=True)
        raw = self.VALUE.encode("utf-8", "surrogateescape")
        decoded = raw.decode("utf-8", errors="replace")
        self.assertIn("�", decoded)
        self.assertEqual(known_secrets.mask_known("x " + decoded + " y"),
                         "x <redacted> y")
        self.assertEqual(errors.redact_secrets("failed: " + decoded),
                         "failed: <redacted>")
        # What the byte stream and the other spellings did before still holds.
        masker = known_secrets.StreamMasker()
        self.assertEqual(masker.feed(b"x " + raw[:9]) + masker.feed(raw[9:] + b" y")
                         + masker.flush(), b"x <redacted> y")
        self.assertEqual(known_secrets.mask_known("x %s y" % self.VALUE), "x <redacted> y")

    def test_a_truncated_multibyte_sequence_decodes_to_the_registered_spelling(self):
        value = "abcdefgh" + "\udce2\udc82" + "ijklmnop"      # e2 82: cut-off euro sign
        known_secrets.register(value, authoritative=True)
        decoded = value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        self.assertEqual(known_secrets.mask_known("x " + decoded), "x <redacted>")

    def test_a_replacement_character_run_is_not_a_secret(self):
        known_secrets.register("\udc80\udc81\udc82\udc83\udc84\udc85\udc86\udc87\udc88",
                               authoritative=True)
        text = "bad bytes: " + "�" * 9 + " in a log"
        self.assertEqual(known_secrets.mask_known(text), text)

    def vendor(self):
        tmp = tempfile.TemporaryDirectory(prefix="omni-171-")
        self.addCleanup(tmp.cleanup)
        binary = os.path.join(os.path.realpath(tmp.name), "vendor")
        with open(binary, "w", encoding="utf-8") as handle:
            handle.write(NON_UTF8_CHILD % {"python": sys.executable})
        os.chmod(binary, stat.S_IRWXU)
        return binary

    def failure_text(self, value, after=b""):
        """The two failure texts run_cli gives for a child echoing *value*."""
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_TEST_SERVICE_TOKEN": value,
               "OMNI_TEST_AFTER": after.hex()}
        res = asyncio.run(cli_run.run_cli([self.vendor()], env=env, timeout_s=30))
        self.assertEqual(res["returncode"], 1)
        return {name: errors.redact_secrets(res[name]) for name in ("stdout", "stderr")}

    def test_run_cli_failure_text_holds_no_fragment(self):
        for stream, shown in self.failure_text(self.VALUE).items():
            self.assertIn("<redacted>", shown, stream)
            for part in ("abcdefgh", "ijklmnop", "qrstuvwx"):
                self.assertNotIn(part, shown, stream)

    def test_a_value_made_of_few_distinct_characters_is_still_masked(self):
        # Three distinct characters once the bad bytes become U+FFFD, but two
        # real 8-character fragments around them.
        value = "ABABABAB\udc80\udc81ABABABAB"
        known_secrets.register(value, authoritative=True)
        decoded = value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        self.assertEqual(known_secrets.mask_known("x " + decoded), "x <redacted>")
        for stream, shown in self.failure_text(value).items():
            self.assertNotIn("ABABABAB", shown, stream)

    def test_the_spelling_does_not_depend_on_the_byte_that_follows(self):
        # The registered spelling ends in a replacement character for the
        # truncated sequence e2 82; the next byte (ac) completes it, so the
        # decoder produces a euro sign instead.
        value = "abcdefghijklmnop\udce2\udc82"
        for after in (b"\xac", b"\xac tail", b"!"):
            for stream, shown in self.failure_text(value, after).items():
                self.assertNotIn("abcdefgh", shown, (stream, after))
                self.assertNotIn("ijklmnop", shown, (stream, after))
                self.assertIn("<redacted>", shown, (stream, after))

    def vendor_raw(self, raw, rc):
        """A binary that writes exactly *raw* on both streams and exits *rc*
        (baked in: a provider hands its child a sanitized environment)."""
        path = os.path.join(self.root, "vendor_raw")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!%s\nimport sys\nraw = bytes.fromhex(%r)\n"
                         "sys.stdout.buffer.write(raw)\nsys.stderr.buffer.write(raw)\n"
                         "sys.exit(%d)\n" % (sys.executable, raw.hex(), rc))
        os.chmod(path, stat.S_IRWXU)
        return path

    def run_raw(self, raw, **extra):
        env = {"PATH": os.environ.get("PATH", ""), "OMNI_TEST_RAW": raw.hex()}
        env.update(extra)
        return asyncio.run(cli_run.run_cli([self.vendor()], env=env, timeout_s=30))

    # A registry holding values of every kind the decoder treats differently,
    # including a UTF-8 value that contains a non-UTF-8 one and a wire line
    # that is a prefix of one.
    INNER = "abcdefghijklmnop\udce2\udc82"
    OUTER = "QRSTUVWX" + "abcdefghijklmnop\u20ac" + "TAILTAIL"
    WIRE = '{"result":"abcdefgh","is_error":true}'

    def mixed_registry(self):
        known_secrets.register_many(
            [self.VALUE, PLAIN, self.INNER, self.OUTER, self.WIRE, self.WIRE + "\udc80",
             "abcdefgh"], authoritative=True)

    def test_what_a_caller_parses_is_the_plain_decode_for_a_mixed_registry(self):
        self.mixed_registry()
        outputs = (
            self.OUTER.encode(), (self.OUTER + "\n").encode() + b"\xff",
            self.INNER.encode("utf-8", "surrogateescape") + b"\xac tail",
            self.VALUE.encode("utf-8", "surrogateescape") + b" " + PLAIN.encode(),
            self.WIRE.encode(), b"head " + self.WIRE.encode() + b"\xe2\x82",
        )
        for raw in outputs:
            res = self.run_raw(raw, OMNI_TEST_RC="0")
            self.assertEqual(res["stdout"], raw.decode(errors="replace"), raw)
            self.assertEqual(res["stderr"], raw.decode(errors="replace"), raw)

    def test_a_longer_value_containing_a_non_utf8_one_is_masked_whole(self):
        self.mixed_registry()
        for raw in (self.OUTER.encode(), b"x " + self.OUTER.encode() + b" y"):
            res = self.run_raw(raw)
            for stream in ("stdout", "stderr"):
                shown = errors.redact_secrets(res[stream])
                for part in ("QRSTUVWX", "TAILTAIL", "abcdefgh", "ijklmnop"):
                    self.assertNotIn(part, shown, (stream, raw))
                self.assertIn("<redacted>", shown, (stream, raw))

    def test_an_in_band_error_wire_line_is_still_a_runtime_error(self):
        # Registering a value that is the wire line plus a bad byte must not
        # change what the provider parses from the plain wire line.
        self.mixed_registry()
        binary = self.vendor_raw(self.WIRE.encode(), 0)
        with mock.patch.dict(os.environ, {
                    "CURSOR_API_KEY": PLAIN,
                    "OMNI_LLM_STATE_DIR": self.root}), \
                mock.patch.object(cursor, "resolve_bin", return_value=binary):
            _seed_catalog("cursor", "synthetic")
            result = capture_server("cursor/synthetic")
        structured = result["structuredContent"]
        self.assertIs(structured.get("ok"), False, structured)
        self.assertEqual(structured.get("reason"), "runtime_error", structured)
        self.assertNotIn("Traceback", structured.get("detail", ""), structured)

    def test_grok_image_failure_text_holds_no_fragment(self):
        known_secrets.register(self.INNER, authoritative=True)
        raw = self.INNER.encode("utf-8", "surrogateescape") + b"\xac tail"
        proc = types.SimpleNamespace(returncode=1, stdout=raw, stderr=raw)
        src = os.path.join(self.root, "src.png")
        with open(src, "wb") as handle:
            handle.write(b"\x89PNG")
        with mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": self.root}), \
                mock.patch.object(grok_image_cli.vendor_bin, "which", return_value="/synthetic/grok"), \
                mock.patch.object(grok_image_cli, "prepare_host_fence",
                                  side_effect=lambda argv, **kw: (argv, dict(os.environ), {})), \
                mock.patch.object(grok_image_cli.subprocess, "run", return_value=proc):
            _seed_grok_image()
            result = asyncio.run(grok_image_cli.run_grok_image_edit(
                "synthetic", src, 5, out_dir=self.root))
        self.assertIs(result.get("ok"), False, result)
        self.assertNotIn("abcdefgh", result["detail"])
        self.assertNotIn("ijklmnop", result["detail"])
        self.assertIn("<redacted>", result["detail"])

    def test_register_decoded_spans_registers_nothing_without_a_non_utf8_value(self):
        known_secrets.register(PLAIN, authoritative=True)
        before = known_secrets._str_patterns
        known_secrets.register_decoded_spans(b"a " + PLAIN.encode() + b"\xff z")
        self.assertIs(known_secrets._str_patterns, before)

    def test_register_decoded_spans_registers_the_spelling_in_context(self):
        known_secrets.register(self.INNER, authoritative=True)
        known_secrets.register_decoded_spans(
            b"a " + self.INNER.encode("utf-8", "surrogateescape") + b"\xac z")
        # The euro sign the follower completes is not part of the span, so a
        # text that spells it that way is masked up to it.
        self.assertEqual(known_secrets.mask_known("a abcdefghijklmnop\u20ac z"),
                         "a <redacted>\u20ac z")


def clean_runs(value):
    """The runs of a value between its undecodable bytes (lone surrogates)."""
    runs, cur = [], ""
    for char in value:
        if "\udc80" <= char <= "\udcff":
            if cur:
                runs.append(cur)
            cur = ""
        else:
            cur += char
    if cur:
        runs.append(cur)
    return runs


def window_leaked(values, text):
    """First 8-character window of a clean run of any value found in text."""
    for value in values:
        for run in clean_runs(value):
            for i in range(len(run) - FRAGMENT + 1):
                if run[i:i + FRAGMENT] in text:
                    return run[i:i + FRAGMENT]
    return None


def failure_text_of(values, raw):
    """Register *values*, decode *raw* as run_cli does, build a failure detail."""
    known_secrets.clear()
    known_secrets.register_many(values, authoritative=True)
    text = cli_run.decode_output(raw)
    assert text == raw.decode("utf-8", errors="replace"), "the parsed text changed"
    return errors.safe_detail(text, limit=len(text) + 64)


def enc(value):
    return value.encode("utf-8", "surrogateescape")


class DecodedSpellingFuzz(Registry):
    """Issue #171 B, closed for text decoded through ``cli_run.decode_output``:
    the failure text holds no 8-character window of any value's clean runs,
    whatever the neighbours, the repeats or the truncations, and what a caller
    parses is the plain decode."""

    NAMED = {
        "adjacent matches (a valid-UTF-8 merge)": (
            ["LEAKLEAK\udce2\udc82", "\udcacSTUVWXYZ"],
            b"err:" + enc("LEAKLEAK\udce2\udc82") + b"\xacSTUVWXYZ end"),
        "touching repeats x500": (
            ["\udc80abcdefghijklmnop\udce2\udc82"],
            b"failed: " + enc("\udc80abcdefghijklmnop\udce2\udc82") * 500 + b" end"),
        "touching repeats x500, starts with a continuation byte": (
            ["\udcacABCDEFGHIJKLMNOP\udce2\udc82"],
            b"x " + enc("\udcacABCDEFGHIJKLMNOP\udce2\udc82") * 500 + b" y"),
        "completed by the next byte": (
            ["abcdefghijklmnop\udce2\udc82"],
            b"auth " + enc("abcdefghijklmnop\udce2\udc82") + b"\xac tail"),
        "truncated at the end of the output": (
            ["abcdefghijklmnop\udce2\udc82"],
            b"auth " + enc("abcdefghijklmnop\udce2\udc82")),
        "few distinct characters": (
            ["ABABABAB\udc80\udc81ABABABAB"],
            b"x " + enc("ABABABAB\udc80\udc81ABABABAB") + b" y"),
    }

    def test_named_cases(self):
        for name, (values, raw) in self.NAMED.items():
            with self.subTest(case=name):
                shown = failure_text_of(values, raw)
                self.assertIsNone(window_leaked(values, shown), (name, shown[:120]))

    def test_touching_repeats_add_a_bounded_number_of_patterns(self):
        value = "\udcacABCDEFGHIJKLMNOP\udce2\udc82"
        known_secrets.register(value, authoritative=True)
        before = len(known_secrets._str_patterns)
        raw = b"x " + enc(value) * 500 + b" y"
        start = time.perf_counter()
        cli_run.decode_output(raw)
        self.assertLess(time.perf_counter() - start, 5.0)
        added = len(known_secrets._str_patterns) - before
        self.assertGreater(added, 0)            # really registered something
        self.assertLess(added, 200, added)      # not one pattern per repeat

    def test_a_value_at_the_length_limit_next_to_buffered_bytes(self):
        # ED A0 is held by the decoder, so the occurrence's piece starts with
        # two U+FFFD and would be MAX_LEN + 1 long: stripped, it must register.
        for pre in (b"\xed\xa0", b"\xed\xbf"):
            with self.subTest(pre=pre):
                value = "K" * (known_secrets.MAX_LEN - 1) + "\udce2"
                self.assertEqual(len(value), known_secrets.MAX_LEN)
                raw = b"x " + pre + enc(value) + b"\x82\xac end"
                shown = failure_text_of([value], raw)
                self.assertIsNone(window_leaked([value], shown), shown[:60])

    def test_a_piece_longer_than_the_limit_is_chunked_not_dropped(self):
        # The limit is lowered only while the output is decoded, so the
        # registered 60-character value yields a piece longer than it.
        run = "".join(chr(65 + (i * 7) % 26) + str(i % 10) for i in range(29))
        value = run + "\udce2"
        known_secrets.clear()
        known_secrets.register(value, authoritative=True)
        before = set(known_secrets._str_patterns)
        with mock.patch.object(known_secrets, "MAX_LEN", 30):
            cli_run.decode_output(b"x " + enc(value) + b"\x82\xac end")
        added = set(known_secrets._str_patterns) - before
        self.assertTrue(added)
        self.assertIn(run[:30], added)           # a chunk of exactly the limit
        for i in range(len(run) - FRAGMENT + 1):
            window = run[i:i + FRAGMENT]
            self.assertTrue(any(window in p for p in added), window)

    def test_seeded_fuzz(self):
        rng = random.Random(1712)
        alpha = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
        lead = [b"\xe2", b"\xe2\x82", b"\xf0\x9f", b"\xf0\x9f\x98", b"\xc3", b"\xed"]
        cont = [b"\x80", b"\x82", b"\xac", b"\xbf", b"\x98\x80"]

        def random_value():
            raw = b""
            if rng.random() < .5:
                raw += rng.choice(cont)
            for i in range(rng.randint(1, 2)):
                if i:
                    raw += rng.choice(lead + cont)
                raw += "".join(rng.choice(alpha) for _ in range(rng.randint(8, 18))).encode()
            if rng.random() < .8:
                raw += rng.choice(lead)
            return raw.decode("utf-8", "surrogateescape")

        cases = 2000
        for case in range(cases):
            values = [random_value() for _ in range(rng.randint(1, 3))]
            raw = b"x "
            for _ in range(rng.randint(1, 8)):
                # touching repeats, including long runs
                raw += enc(rng.choice(values)) * rng.choice([1, 1, 2, 3, 600])
                if rng.random() < .3:
                    raw += rng.choice(cont + lead + [b" "])
            raw += rng.choice([b"", b" end", b"\xac", b"\x80"])
            shown = failure_text_of(values, raw)
            leak = window_leaked(values, shown)
            self.assertIsNone(leak, (case, [repr(v) for v in values], raw[:60]))


class LoginFileTokenFloor(Registry):
    """Issue #171 C: a flag- or model-id-shaped string in a login file is not
    a token. A string that is not a JWT needs 32 characters and two character
    classes."""

    def write(self, home, payload):
        path = os.path.join(home, "auth.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.chmod(path, 0o600)

    def test_short_flag_and_model_id_shapes_are_not_registered(self):
        shaped = ("--dangerously-skip-permissions", "claude-sonnet-4-5-20250929",
                  "gpt-5-codex-high-reasoning", "gemini-2.5-flash-lite-preview",
                  "A" * 19 + "b", "x" * 30 + "7")
        for value in shaped:
            self.assertLess(len(value), 32, value)
        with tempfile.TemporaryDirectory() as home:
            self.write(home, {"api.x.ai": {"key": shaped[0], "refresh_token": shaped[1]},
                              "p2": {"key": shaped[2]}, "p3": {"key": shaped[3]},
                              "p4": {"key": shaped[4]}, "p5": {"key": shaped[5]}})
            known_secrets.register_auth_files({"GROK_HOME": home})
        self.assertFalse(known_secrets.active())
        for value in shaped:
            self.assertEqual(known_secrets.mask_known("reply " + value), "reply " + value)

    def test_real_world_token_shapes_still_register(self):
        # Synthetic strings built from the shapes of real tokens.
        hexkey = "0123456789abcdef" * 2                          # 32 hex digits
        urlsafe = "Zx9Wv8Ut" * 5 + "-_a"                        # base64url, 43
        prefixed = "sk-" + "Ab3Cd4Ef" * 5                        # vendor prefix, 43
        opaque = "rt_" + "Gh5Ij6Kl" * 4                          # refresh style, 35
        dotted = "v1." + "Mn7Op8Qr" * 4                          # with a dot, 35
        short_jwt = "eyJhbGciOi.eyJzdWIiOiJ4.sig"                # a JWT at any length
        tokens = (hexkey, urlsafe, prefixed, opaque, dotted, short_jwt)
        with tempfile.TemporaryDirectory() as home:
            self.write(home, {"p%d" % i: {"key": t} for i, t in enumerate(tokens)})
            known_secrets.register_auth_files({"GROK_HOME": home})
        for token in tokens:
            self.assertEqual(known_secrets.mask_known("t " + token), "t <redacted>", token)

    def test_the_floor_is_thirty_two_exactly(self):
        edge32 = "aB3dE5fG7hJ9" * 2 + "kLmNpQrS"
        edge31 = edge32[:-1]
        self.assertEqual((len(edge32), len(edge31)), (32, 31))
        with tempfile.TemporaryDirectory() as home:
            self.write(home, {"p1": {"key": edge32}})
            known_secrets.register_auth_files({"GROK_HOME": home})
        self.assertEqual(known_secrets.mask_known("t " + edge32), "t <redacted>")
        known_secrets.clear()
        with tempfile.TemporaryDirectory() as home:
            self.write(home, {"p2": {"key": edge31}})
            known_secrets.register_auth_files({"GROK_HOME": home})
        self.assertEqual(known_secrets.mask_known("t " + edge31), "t " + edge31)



if __name__ == "__main__":
    unittest.main()
