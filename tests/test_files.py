#!/usr/bin/env python3
# no-opencc
"""Hermetic tests for `files=` on the consultation lane. No network, no real CLIs.

Written against the FROZEN Phase 3b spec, not against an implementation. Per
provider the spec says:
- agy   (unchanged): path/URL strings appended into the prompt text.
- codex (new): every entry must be an EXISTING local file, handed to `codex exec`
  as repeated `-i <path>` args (native vision). No server-side read, no URLs.
- grok  (new): every entry must be an existing local IMAGE under the
  OMNI_LLM_SOURCE_ROOTS allowlist; the server reads it, base64-encodes it and
  sends prompt+images as ONE inline `--prompt-json` ACP payload (never
  `--prompt-file`, the two flags are mutually exclusive).
- everything else: LOUD unsupported_files.

Everything is faked at the seam the spec names:
- `run_cli` is patched at the MODULE level of the unit under test
  (providers.codex / providers.grok), so no subprocess is ever spawned.
- `shutil.which` reports codex/grok as installed, so argv construction is
  asserted on this machine regardless of what is actually on PATH.
- the allowlist is a realpath'd temp dir injected through
  mock.patch.dict(os.environ, {OMNI_LLM_SOURCE_ROOTS: ...}); OMNI_LLM_DEPTH is
  cleared so the recursion guard never fires first.

Parts of the spec may not be implemented yet. Missing pieces surface as ONE loud
existence failure each (TestSpecUnitsExist) plus granular skips, so a partial
implementation still produces a readable per-spec-point report.
"""
import asyncio
import base64
import binascii
import contextlib
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="omni-files-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP
os.environ["OMNI_LLM_NOTIFY_QUIET"] = "1"
os.environ.pop("OMNI_LLM_NOTIFY_CMD", None)

from providers import claude as claude_mod  # noqa: E402
from providers import pi as pi_mod  # noqa: E402
from providers import cursor as cursor_mod  # noqa: E402
from providers import codex as codex_mod  # noqa: E402
from providers import config as config_mod  # noqa: E402
from providers import dispatch as dispatch_mod  # noqa: E402
from providers import grok as grok_mod  # noqa: E402
from providers import host_fence as host_fence_mod  # noqa: E402
from providers import openai_compat as openai_mod  # noqa: E402
from providers import route as route_mod  # noqa: E402
from providers import source_load as source_load_mod  # noqa: E402

_REAL_WHICH = shutil.which
FILES_CLIS = ("codex", "grok")
ENV_ROOTS = getattr(config_mod, "ENV_SOURCE_ROOTS", "OMNI_LLM_SOURCE_ROOTS")
ENV_DEPTH = getattr(config_mod, "ENV_DEPTH", "OMNI_LLM_DEPTH")
CAP_NAME = "GROK_FILES_MAX_PAYLOAD_BYTES"
MIME_NAME = "FILES_IMAGE_MIME"
TEXT_MIME_NAME = "FILES_TEXT_MIME"
TEXT_CAP_NAME = "GROK_FILES_TEXT_MAX_BYTES"
TEXT_TOTAL_CAP_NAME = "GROK_FILES_TEXT_TOTAL_MAX_BYTES"

# A real 1x1 PNG. Nothing decodes it here; it just keeps the fixtures honest.
PNG_1PX = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    b"hQGAhKmMIQAAAABJRU5ErkJggg=="
)

# error_code taxonomy of the existing source_path reader; the grok binary read
# must reuse it verbatim (spec I5/I6b).
SOURCE_ERROR_CODES = {
    "source_path_disabled",
    "source_path_not_absolute",
    "source_path_not_allowed",
    "source_not_found",
    "source_not_regular",
    "source_too_large",
    "source_io_error",
    "source_invalid_utf8",
    "invalid_arguments",
}


def _accepts(func, name):
    """True when `func` takes a parameter called `name` (spec-lag detector)."""
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


HAS_BINARY_READER = callable(getattr(source_load_mod, "open_binary_under_roots", None))
CODEX_TAKES_FILES = _accepts(codex_mod.run_codex_subprocess, "files")
GROK_TAKES_FILES = _accepts(grok_mod.run_grok_subprocess, "files")


# --------------------------------------------------------------------------- #
# fakes / fixtures
# --------------------------------------------------------------------------- #
class FakeRunCli:
    """Async run_cli stand-in: records (argv, kwargs), never spawns anything.

    Writes the `-o` reply file codex reads back, and snapshots any
    `--prompt-file` body while it still exists (the provider unlinks it in its
    own finally block).
    """

    def __init__(self, returncode=0, stdout="STDOUT REPLY", stderr="",
                 timed_out=False, out_file_text="CODEX REPLY"):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.out_file_text = out_file_text
        self.calls = []
        self.prompt_file_bodies = []

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
        if "--prompt-file" in argv:
            path = argv[argv.index("--prompt-file") + 1]
            try:
                with open(path, encoding="utf-8") as f:
                    self.prompt_file_bodies.append(f.read())
            except OSError:
                self.prompt_file_bodies.append(None)
        return {"returncode": self.returncode, "stdout": self.stdout,
                "stderr": self.stderr, "timed_out": self.timed_out}

    @property
    def argv(self):
        return self.calls[0][0]


class Recorder:
    """Async provider stand-in that records calls and returns a canned envelope."""

    def __init__(self, envelope=None):
        self.envelope = envelope or {"ok": True, "text": "ok", "model": "fake"}
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return dict(self.envelope)


def _fake_which(name, *a, **kw):
    base = os.path.basename(str(name))
    if base in FILES_CLIS:
        return "/fake/bin/" + base
    return _REAL_WHICH(name, *a, **kw)


@contextlib.contextmanager
def sandbox_env(roots=None):
    """OMNI_LLM_SOURCE_ROOTS = `roots` (unset when None); OMNI_LLM_DEPTH cleared."""
    with mock.patch.dict(os.environ, {}):
        os.environ.pop(ENV_DEPTH, None)
        if roots is None:
            os.environ.pop(ENV_ROOTS, None)
        else:
            os.environ[ENV_ROOTS] = roots
        yield


@contextlib.contextmanager
def patched_cli(module, fake):
    """Patch `module.run_cli` and report codex/grok as installed.

    The fake answers in plain/json, not grok's streaming-json; the stream
    verdict it would feed is pinned in test_grok_consult_containment.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(module, "run_cli", fake))
        stack.enter_context(mock.patch.object(shutil, "which", _fake_which))
        if module is grok_mod:
            stack.enter_context(mock.patch.object(
                grok_mod, "_consult_result",
                side_effect=lambda out, sid: (*grok_mod.extract_cli_result("grok", out), None)))
        yield


@contextlib.contextmanager
def shrunk_const(name, n):
    """Shrink a config constant wherever the NAME is bound (config + grok)."""
    targets = [m for m in (config_mod, grok_mod) if hasattr(m, name)]
    if not targets:
        raise unittest.SkipTest("%s not defined yet" % name)
    with contextlib.ExitStack() as stack:
        for m in targets:
            stack.enter_context(mock.patch.object(m, name, n))
        yield


def shrunk_cap(n):
    """Shrink GROK_FILES_MAX_PAYLOAD_BYTES wherever the name is bound."""
    return shrunk_const(CAP_NAME, n)


def flag_values(argv, flag):
    """Every token that directly follows `flag`, in argv order."""
    argv = list(argv)
    out = []
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
    return out


class FilesTestCase(unittest.TestCase):
    """Temp-root helpers + envelope assertions shared by every files= test."""

    def mkdir(self, prefix="omni_files_"):
        """A realpath'd temp dir, so abspath(dir) == dir on macOS (/var symlink)."""
        d = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def write_file(self, directory, name, data=PNG_1PX):
        path = os.path.join(directory, name)
        mode = "wb" if isinstance(data, bytes) else "w"
        with open(path, mode) as f:
            f.write(data)
        return path

    def run_codex(self, **kwargs):
        kwargs.setdefault("prompt", "describe the image")
        kwargs.setdefault("model", "gpt-5.6-sol")
        kwargs.setdefault("timeout_s", 30)
        return asyncio.run(codex_mod.run_codex_subprocess(**kwargs))

    def run_grok(self, **kwargs):
        kwargs.setdefault("prompt", "describe the image")
        kwargs.setdefault("model", "grok-4.5")
        kwargs.setdefault("timeout_s", 30)
        return asyncio.run(grok_mod.run_grok_subprocess(**kwargs))

    def assertRefusal(self, env, reason):
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), reason, env)
        self.assertIsInstance(env.get("detail"), str, env)

    def assertNotCalled(self, fake):
        self.assertEqual(fake.calls, [], "provider must refuse BEFORE spawning a CLI")


# --------------------------------------------------------------------------- #
# 0. existence — one loud failure per missing spec unit (the rest degrade to skips)
# --------------------------------------------------------------------------- #
class TestSpecUnitsExist(unittest.TestCase):
    def test_config_declares_files_image_mime(self):
        mime = getattr(config_mod, MIME_NAME, None)
        self.assertIsInstance(mime, dict, "providers/config.py must define %s" % MIME_NAME)
        self.assertEqual(mime.get(".png"), "image/png")
        self.assertEqual(mime.get(".jpg"), "image/jpeg")
        self.assertEqual(mime.get(".jpeg"), "image/jpeg")
        self.assertEqual(mime.get(".gif"), "image/gif")
        self.assertEqual(mime.get(".webp"), "image/webp")

    def test_config_declares_grok_inline_payload_cap(self):
        cap = getattr(config_mod, CAP_NAME, None)
        self.assertIsInstance(cap, int, "providers/config.py must define %s" % CAP_NAME)
        self.assertGreater(cap, 0)

    def test_config_declares_files_text_mime(self):
        mime = getattr(config_mod, TEXT_MIME_NAME, None)
        self.assertIsInstance(mime, dict,
                              "providers/config.py must define %s" % TEXT_MIME_NAME)
        for ext in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml",
                    ".csv", ".log"):
            with self.subTest(ext=ext):
                self.assertIsInstance(mime.get(ext), str,
                                      "%s must map %s to a MIME type" % (TEXT_MIME_NAME, ext))
        self.assertEqual(mime.get(".txt"), "text/plain")
        self.assertEqual(mime.get(".json"), "application/json")

    def test_image_and_text_tables_do_not_overlap(self):
        image = getattr(config_mod, MIME_NAME, {}) or {}
        text = getattr(config_mod, TEXT_MIME_NAME, {}) or {}
        self.assertEqual(set(image) & set(text), set(),
                         "an extension must classify as exactly one attachment kind")

    def test_config_declares_inline_text_caps_under_the_payload_cap(self):
        payload_cap = getattr(config_mod, CAP_NAME, None)
        per_file = getattr(config_mod, TEXT_CAP_NAME, None)
        total = getattr(config_mod, TEXT_TOTAL_CAP_NAME, None)
        for name, value in ((TEXT_CAP_NAME, per_file), (TEXT_TOTAL_CAP_NAME, total)):
            with self.subTest(const=name):
                self.assertIsInstance(value, int,
                                      "providers/config.py must define %s" % name)
                self.assertGreater(value, 0)
        self.assertLessEqual(per_file, total, "per-file cap must not exceed the total")
        # Both caps must be REACHABLE: if the argv payload cap fired first the
        # files_too_large refusal would be dead code.
        self.assertLessEqual(total, payload_cap,
                             "inline text caps must sit under %s" % CAP_NAME)

    def test_source_load_exposes_binary_reader(self):
        self.assertTrue(
            HAS_BINARY_READER,
            "providers/source_load.py must expose open_binary_under_roots(path, max_bytes)")

    def test_codex_provider_accepts_files(self):
        self.assertTrue(CODEX_TAKES_FILES,
                        "run_codex_subprocess must accept files=None")
        self.assertTrue(hasattr(codex_mod, "run_cli"),
                        "providers.codex must import run_cli at module level (patch seam)")

    def test_grok_provider_accepts_files(self):
        self.assertTrue(GROK_TAKES_FILES,
                        "run_grok_subprocess must accept files=None")
        self.assertTrue(hasattr(grok_mod, "run_cli"),
                        "providers.grok must import run_cli at module level (patch seam)")


# --------------------------------------------------------------------------- #
# g1. route: which providers advertise files support
# --------------------------------------------------------------------------- #
class TestRouteFilesSupport(unittest.TestCase):
    def test_g1_supported_providers(self):
        for provider in ("agy", "codex", "grok"):
            with self.subTest(provider=provider):
                self.assertTrue(route_mod.provider_supports_files(provider),
                                "%s must support files=" % provider)

    def test_g1_unsupported_providers(self):
        for provider in ("openai", "ollama", "lmstudio", "claude"):
            with self.subTest(provider=provider):
                self.assertFalse(route_mod.provider_supports_files(provider),
                                 "%s must NOT support files=" % provider)


# --------------------------------------------------------------------------- #
# g2. dispatch early reject — refusal happens before the provider is touched
# --------------------------------------------------------------------------- #
class TestDispatchEarlyReject(FilesTestCase):
    def route(self, model, files):
        with sandbox_env():
            return asyncio.run(dispatch_mod.route_to_provider(
                "hi", model, 30, files, "default"))

    def test_g2_claude_model_with_files_is_rejected_before_the_provider(self):
        rec = Recorder()
        with mock.patch.object(claude_mod, "run_claude_subprocess", rec):
            env = self.route("claude/opus", ["/tmp/a.png"])
        self.assertRefusal(env, "unsupported_files")
        self.assertEqual(rec.calls, [], "claude provider must never be called")

    def test_g2_pi_model_with_files_is_rejected_before_the_provider(self):
        rec = Recorder()
        with mock.patch.object(pi_mod, "run_pi_subprocess", rec):
            env = self.route("pi/xai/grok-4.6", ["/tmp/a.png"])
        self.assertRefusal(env, "unsupported_files")
        self.assertEqual(rec.calls, [], "pi provider must never be called")

    def test_g2_cursor_model_with_files_is_rejected_before_the_provider(self):
        rec = Recorder()
        with mock.patch.object(cursor_mod, "run_cursor_subprocess", rec):
            env = self.route("cursor/fable", ["/tmp/a.png"])
        self.assertRefusal(env, "unsupported_files")
        self.assertEqual(rec.calls, [], "cursor provider must never be called")

    def test_g2_ollama_model_with_files_is_rejected_before_the_provider(self):
        rec = Recorder()
        with mock.patch.object(openai_mod, "run_openai_compatible", rec):
            env = self.route("ollama/x", ["/tmp/a.png"])
        self.assertRefusal(env, "unsupported_files")
        self.assertEqual(rec.calls, [], "openai-compatible provider must never be called")

    def test_g2_detail_names_all_three_supporting_providers(self):
        env = self.route("claude/opus", ["/tmp/a.png"])
        self.assertRefusal(env, "unsupported_files")
        self.assertIn("agy/codex/grok", env["detail"],
                      "detail must name the three supporting providers")


# --------------------------------------------------------------------------- #
# g3/g4. codex — repeated `-i <path>` pass-through, existence fence
# --------------------------------------------------------------------------- #
@unittest.skipUnless(CODEX_TAKES_FILES, "run_codex_subprocess(files=...) not implemented yet")
class TestCodexFiles(FilesTestCase):
    def call(self, files, prompt="describe the image"):
        fake = FakeRunCli()
        with sandbox_env(), patched_cli(codex_mod, fake):
            env = self.run_codex(prompt=prompt, files=files)
        return env, fake

    def test_g3_single_file_becomes_one_i_pair_after_ephemeral(self):
        png = self.write_file(self.mkdir(), "shot.png")
        env, fake = self.call([png])
        self.assertTrue(env.get("ok"), env)
        argv = fake.argv
        self.assertEqual(flag_values(argv, "-i"), [png], argv)
        self.assertIn("--ephemeral", argv)
        self.assertGreater(argv.index("-i"), argv.index("--ephemeral"),
                           "-i pairs belong after --ephemeral: %r" % argv)
        self.assertEqual(argv[-1], "-", "the stdin marker must stay last: %r" % argv)

    def test_g3_two_files_become_two_i_pairs_in_order(self):
        d = self.mkdir()
        first = self.write_file(d, "a.png")
        second = self.write_file(d, "b.png")
        env, fake = self.call([first, second])
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(flag_values(fake.argv, "-i"), [first, second], fake.argv)

    def test_g3_no_files_keeps_argv_free_of_i(self):
        fake = FakeRunCli()
        with sandbox_env(), patched_cli(codex_mod, fake):
            env = self.run_codex()
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("-i", fake.argv, fake.argv)

    def test_g3_empty_files_list_is_the_same_as_no_files(self):
        env, fake = self.call([])
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("-i", fake.argv, fake.argv)

    def test_g4_missing_file_is_unsupported_files_before_spawn(self):
        missing = os.path.join(self.mkdir(), "no_such.png")
        env, fake = self.call([missing])
        self.assertRefusal(env, "unsupported_files")
        self.assertIn(missing, env["detail"], env)
        self.assertNotCalled(fake)

    def test_g4_directory_and_url_entries_are_refused_too(self):
        d = self.mkdir()
        for bad in (d, "https://example.com/a.png"):
            with self.subTest(entry=bad):
                env, fake = self.call([bad])
                self.assertRefusal(env, "unsupported_files")
                self.assertNotCalled(fake)

    def test_g4_one_bad_entry_poisons_the_whole_call(self):
        d = self.mkdir()
        good = self.write_file(d, "good.png")
        missing = os.path.join(d, "gone.png")
        env, fake = self.call([good, missing])
        self.assertRefusal(env, "unsupported_files")
        self.assertNotCalled(fake)


# --------------------------------------------------------------------------- #
# g5. grok without files — byte-identical to today's --prompt-file path
# --------------------------------------------------------------------------- #
class TestGrokNoFilesRegression(FilesTestCase):
    def test_g5_no_files_still_uses_prompt_file(self):
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(), patched_cli(grok_mod, fake):
            env = self.run_grok(prompt="plain text question")
        self.assertTrue(env.get("ok"), env)
        argv = fake.argv
        self.assertIn("--prompt-file", argv, argv)
        self.assertNotIn("--prompt-json", argv, argv)
        self.assertEqual(fake.prompt_file_bodies, ["plain text question"])

    @unittest.skipUnless(GROK_TAKES_FILES, "run_grok_subprocess(files=...) not implemented yet")
    def test_g5_explicit_empty_files_still_uses_prompt_file(self):
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(), patched_cli(grok_mod, fake):
            env = self.run_grok(prompt="plain text question", files=[])
        self.assertTrue(env.get("ok"), env)
        self.assertIn("--prompt-file", fake.argv, fake.argv)
        self.assertNotIn("--prompt-json", fake.argv, fake.argv)


# --------------------------------------------------------------------------- #
# g6. grok with files — one inline ACP --prompt-json payload
# --------------------------------------------------------------------------- #
@unittest.skipUnless(GROK_TAKES_FILES, "run_grok_subprocess(files=...) not implemented yet")
class TestGrokFilesHappyPath(FilesTestCase):
    def setUp(self):
        self.root = self.mkdir("omni_files_root_")
        self.png = self.write_file(self.root, "red.png")

    def call(self, files=None, prompt="describe the image", roots=None):
        fake = FakeRunCli(stdout="GROK REPLY")
        if files is None:
            files = [self.png]
        with sandbox_env(roots if roots is not None else self.root), \
                patched_cli(grok_mod, fake):
            env = self.run_grok(prompt=prompt, files=files)
        return env, fake

    def payload(self, fake):
        argv = fake.argv
        self.assertIn("--prompt-json", argv, argv)
        raw = argv[argv.index("--prompt-json") + 1]
        self.assertIsInstance(raw, str, "--prompt-json must carry the payload inline")
        try:
            return json.loads(raw)
        except ValueError as e:
            self.fail("--prompt-json value is not JSON (%s): %r" % (e, raw[:200]))

    def test_g6_flags_swap_prompt_file_for_prompt_json(self):
        env, fake = self.call()
        self.assertTrue(env.get("ok"), env)
        self.assertIn("--prompt-json", fake.argv, fake.argv)
        self.assertNotIn("--prompt-file", fake.argv, fake.argv)
        self.assertEqual(fake.prompt_file_bodies, [],
                         "no temp prompt file may be written in files= mode")

    def test_g6_payload_is_an_acp_envelope(self):
        _, fake = self.call()
        payload = self.payload(fake)
        self.assertEqual(payload.get("type"), "acp", payload)
        self.assertIsInstance(payload.get("content"), list, payload)

    def test_g6_first_content_block_is_the_prompt_text(self):
        _, fake = self.call(prompt="what colour is this")
        content = self.payload(fake)["content"]
        self.assertEqual(content[0], {"type": "text", "text": "what colour is this"},
                         content[0])

    def test_g6_image_block_carries_mime_and_base64_bytes(self):
        _, fake = self.call()
        content = self.payload(fake)["content"]
        self.assertEqual(len(content), 2, content)
        block = content[1]
        self.assertEqual(block.get("type"), "image", block)
        self.assertEqual(block.get("mimeType"), "image/png", block)
        try:
            decoded = base64.b64decode(block.get("data") or "", validate=True)
        except (binascii.Error, ValueError) as e:
            self.fail("image block data is not valid base64: %s" % e)
        self.assertEqual(decoded, PNG_1PX, "base64 payload must be the file bytes")

    def test_g6_two_images_produce_two_blocks_in_order(self):
        second = self.write_file(self.root, "blue.jpg", PNG_1PX + b"\x00")
        _, fake = self.call(files=[self.png, second])
        content = self.payload(fake)["content"]
        self.assertEqual([b.get("type") for b in content], ["text", "image", "image"], content)
        self.assertEqual([b.get("mimeType") for b in content[1:]],
                         ["image/png", "image/jpeg"], content)
        self.assertEqual(base64.b64decode(content[2]["data"]), PNG_1PX + b"\x00")

    def test_g6_other_grok_flags_are_unchanged(self):
        _, fake = self.call()
        argv = fake.argv
        for flag in ("--model", "--output-format", "--no-memory",
                     "--disable-web-search", "--sandbox", "--cwd"):
            self.assertIn(flag, argv, argv)
        # Host fence contract (providers/host_fence.py): after a VERIFIED
        # outer kernel wrap (Darwin) the nested grok sandbox is rewritten to
        # "off"; where no host wrap applies the vendor "strict" is kept.
        expected = "off" if platform.system() == "Darwin" else "strict"
        self.assertEqual(flag_values(argv, "--sandbox"), [expected], argv)

    def test_g6_sandbox_stays_strict_when_no_host_wrap_applies(self):
        # Unfenced branch: prepare_host_fence returns argv untouched off
        # Darwin, so grok keeps its own --sandbox strict. The real fence code
        # still runs; only the platform answer it keys on is changed.
        with mock.patch.object(host_fence_mod.platform, "system",
                               return_value="Linux"):
            _, fake = self.call()
        self.assertEqual(flag_values(fake.argv, "--sandbox"), ["strict"], fake.argv)


# --------------------------------------------------------------------------- #
# g7-g10. grok fences — allowlist, extension, inline payload cap
# --------------------------------------------------------------------------- #
@unittest.skipUnless(GROK_TAKES_FILES, "run_grok_subprocess(files=...) not implemented yet")
class TestGrokFilesFences(FilesTestCase):
    def call(self, files, roots, prompt="describe the image"):
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(roots), patched_cli(grok_mod, fake):
            env = self.run_grok(prompt=prompt, files=files)
        return env, fake

    def assertSourceEnvelope(self, env, expected_code=None):
        self.assertFalse(env.get("ok"), env)
        code = env.get("error_code")
        self.assertIn(code, SOURCE_ERROR_CODES,
                      "grok must surface the source_path error taxonomy: %r" % env)
        if expected_code is not None:
            self.assertEqual(code, expected_code, env)

    def test_g7_roots_unset_is_a_loud_source_envelope(self):
        d = self.mkdir()
        png = self.write_file(d, "shot.png")
        env, fake = self.call([png], roots=None)
        self.assertSourceEnvelope(env, "source_path_disabled")
        self.assertNotCalled(fake)

    def test_g8_file_outside_the_roots_is_refused(self):
        root = self.mkdir("omni_files_root_")
        outside = self.write_file(self.mkdir("omni_files_out_"), "shot.png")
        env, fake = self.call([outside], roots=root)
        self.assertSourceEnvelope(env, "source_path_not_allowed")
        self.assertNotCalled(fake)

    def test_g8_symlink_into_the_roots_is_refused(self):
        root = self.mkdir("omni_files_root_")
        real = self.write_file(self.mkdir("omni_files_out_"), "shot.png")
        link = os.path.join(root, "link.png")
        os.symlink(real, link)
        env, fake = self.call([link], roots=root)
        self.assertFalse(env.get("ok"), env)
        self.assertNotCalled(fake)

    def test_g8_missing_file_under_the_roots_is_refused(self):
        root = self.mkdir("omni_files_root_")
        env, fake = self.call([os.path.join(root, "nope.png")], roots=root)
        self.assertSourceEnvelope(env)
        self.assertNotCalled(fake)

    def test_g9_unknown_extension_is_unsupported_files(self):
        # .pdf is neither an image nor a whitelisted text type: no ACP block type
        # has been verified for it, so it stays a loud refusal (issue #53 point 3).
        root = self.mkdir("omni_files_root_")
        pdf = self.write_file(root, "paper.pdf", b"%PDF-1.4 fake")
        env, fake = self.call([pdf], roots=root)
        self.assertRefusal(env, "unsupported_files")
        self.assertNotCalled(fake)

    def test_g9_refusal_detail_names_both_accepted_classes(self):
        root = self.mkdir("omni_files_root_")
        pdf = self.write_file(root, "paper.pdf", b"%PDF-1.4 fake")
        env, _ = self.call([pdf], roots=root)
        detail = env["detail"].lower()
        for token in (".png", ".txt", ".md"):
            with self.subTest(token=token):
                self.assertIn(token, detail,
                              "detail must name both accepted classes: %r" % detail)

    def test_g9_extension_check_precedes_the_allowlist_check(self):
        # A .pdf outside every root must still report the EXTENSION problem, so
        # the caller learns what grok takes, not that the path is fenced.
        outside = self.write_file(self.mkdir("omni_files_out_"), "paper.pdf", b"%PDF")
        env, fake = self.call([outside], roots=self.mkdir("omni_files_root_"))
        self.assertRefusal(env, "unsupported_files")
        self.assertNotCalled(fake)

    def test_g10_inline_payload_cap_is_enforced(self):
        # 1200 raw bytes pass the reader's own size check at a 1500-byte cap, but
        # base64 inflates them to 1600 — only the payload cap can catch this.
        root = self.mkdir("omni_files_root_")
        big = self.write_file(root, "big.png", PNG_1PX + b"\x00" * (1200 - len(PNG_1PX)))
        fake = FakeRunCli(stdout="GROK REPLY")
        with shrunk_cap(1500), sandbox_env(root), patched_cli(grok_mod, fake):
            env = self.run_grok(files=[big])
        self.assertRefusal(env, "unsupported_files")
        self.assertIn("cap", env["detail"].lower(), env)
        self.assertNotCalled(fake)


# --------------------------------------------------------------------------- #
# g12. grok text attachments (issue #53) — `resource` blocks with inline text
# --------------------------------------------------------------------------- #
@unittest.skipUnless(GROK_TAKES_FILES, "run_grok_subprocess(files=...) not implemented yet")
class TestGrokTextFiles(FilesTestCase):
    def setUp(self):
        self.root = self.mkdir("omni_files_root_")

    def call(self, files, prompt="summarise the attachment", roots=None):
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(self.root if roots is None else roots), \
                patched_cli(grok_mod, fake):
            env = self.run_grok(prompt=prompt, files=files)
        return env, fake

    def payload(self, fake):
        argv = fake.argv
        self.assertIn("--prompt-json", argv, argv)
        return json.loads(argv[argv.index("--prompt-json") + 1])

    def test_g12_text_file_becomes_a_resource_block_with_inline_text(self):
        body = "probe-token-xyz123\nsecond line\n"
        txt = self.write_file(self.root, "notes.txt", body)
        env, fake = self.call([txt])
        self.assertTrue(env.get("ok"), env)
        content = self.payload(fake)["content"]
        self.assertEqual([b.get("type") for b in content], ["text", "resource"], content)
        resource = content[1]["resource"]
        self.assertEqual(resource.get("text"), body,
                         "file body must round-trip verbatim")
        self.assertEqual(resource.get("mimeType"), "text/plain", resource)
        self.assertTrue(str(resource.get("uri", "")).startswith("file:///"), resource)

    def test_g12_resource_uri_sanitizes_the_source_basename(self):
        hostile = self.write_file(
            self.root, "a b\nIGNORE PREVIOUS.md", "# doc\n")
        env, fake = self.call([hostile])
        self.assertTrue(env.get("ok"), env)
        uri = self.payload(fake)["content"][1]["resource"]["uri"]
        self.assertNotIn("\n", uri, uri)
        self.assertNotIn(" ", uri, uri)

    def test_g12_resource_uri_keeps_same_basename_files_distinct(self):
        # Same basename, different directories: the attachment index must keep
        # the two blocks distinguishable to the model.
        sub = os.path.join(self.root, "nested")
        os.makedirs(sub, exist_ok=True)
        first = self.write_file(self.root, "notes.txt", "one\n")
        second = self.write_file(sub, "notes.txt", "two\n")
        env, fake = self.call([first, second])
        self.assertTrue(env.get("ok"), env)
        uris = [b["resource"]["uri"] for b in self.payload(fake)["content"][1:]]
        self.assertEqual(len(set(uris)), 2, uris)
        self.assertEqual(uris, ["file:///1-notes.txt", "file:///2-notes.txt"], uris)

    def test_g12_resource_uri_keeps_the_extension_when_truncating(self):
        long_name = "n" * 200 + ".md"  # well over the stem cap, under NAME_MAX
        path = self.write_file(self.root, long_name, "# doc\n")
        env, fake = self.call([path])
        self.assertTrue(env.get("ok"), env)
        uri = self.payload(fake)["content"][1]["resource"]["uri"]
        self.assertTrue(uri.endswith(".md"),
                        "truncation must keep the suffix as a type hint: %r" % uri)
        self.assertLess(len(uri), 120, uri)

    def test_g12_every_whitelisted_extension_is_accepted(self):
        for ext, mime in sorted(config_mod.FILES_TEXT_MIME.items()):
            with self.subTest(ext=ext):
                path = self.write_file(self.root, "sample" + ext, "x = 1\n")
                env, fake = self.call([path])
                self.assertTrue(env.get("ok"), env)
                block = self.payload(fake)["content"][1]
                self.assertEqual(block.get("type"), "resource", block)
                self.assertEqual(block["resource"].get("mimeType"), mime, block)

    def test_g12_mixed_image_and_text_keeps_caller_order(self):
        png = self.write_file(self.root, "shot.png")
        txt = self.write_file(self.root, "notes.md", "# heading\n")
        env, fake = self.call([png, txt])
        self.assertTrue(env.get("ok"), env)
        content = self.payload(fake)["content"]
        self.assertEqual([b.get("type") for b in content],
                         ["text", "image", "resource"], content)
        self.assertEqual(base64.b64decode(content[1]["data"]), PNG_1PX)
        self.assertEqual(content[2]["resource"]["text"], "# heading\n")

    def test_g12_oversized_text_file_is_files_too_large(self):
        big = self.write_file(self.root, "big.txt", "x" * 4096)
        fake = FakeRunCli(stdout="GROK REPLY")
        with shrunk_const(TEXT_CAP_NAME, 1024), sandbox_env(self.root), \
                patched_cli(grok_mod, fake):
            env = self.run_grok(files=[big])
        self.assertRefusal(env, "files_too_large")
        self.assertIn("1024", env["detail"], env)
        self.assertNotCalled(fake)

    def test_g12_total_text_cap_refuses_under_the_per_file_cap(self):
        first = self.write_file(self.root, "a.txt", "a" * 900)
        second = self.write_file(self.root, "b.txt", "b" * 900)
        fake = FakeRunCli(stdout="GROK REPLY")
        with shrunk_const(TEXT_CAP_NAME, 1024), \
                shrunk_const(TEXT_TOTAL_CAP_NAME, 1500), \
                sandbox_env(self.root), patched_cli(grok_mod, fake):
            env = self.run_grok(files=[first, second])
        self.assertRefusal(env, "files_too_large")
        self.assertIn("total", env["detail"].lower(), env)
        self.assertEqual(env.get("error_code"), "source_too_large",
                         "every size-cap refusal carries the same stable code")
        self.assertNotCalled(fake)

    def test_g12_payload_is_not_ascii_escaped(self):
        # ensure_ascii=True would turn every CJK char into a 6-byte \uXXXX escape,
        # doubling the wire size of non-ASCII text for no gain.
        body = "中文測試內容\n" * 50
        path = self.write_file(self.root, "cjk.md", body)
        env, fake = self.call([path])
        self.assertTrue(env.get("ok"), env)
        raw = fake.argv[fake.argv.index("--prompt-json") + 1]
        self.assertNotIn("\\u", raw, "payload must not ASCII-escape non-ASCII text")
        self.assertIn("中文測試內容", raw, "non-ASCII must ride the argv verbatim")
        self.assertEqual(self.payload(fake)["content"][1]["resource"]["text"], body)

    def test_g12_cjk_within_both_text_caps_stays_within_the_payload_cap(self):
        """Regression pin: two CJK files that pass BOTH inline-text caps must not
        be killed by the argv payload cap.

        Under ASCII escaping each 3-byte CJK char became 6 bytes, so ~498 KB of
        zh-TW (inside the 256 KiB per-file and 512 KiB total caps) serialized to
        ~996 KB and blew the 600 KiB argv cap. Real caps on purpose — the whole
        point is that the three real numbers now compose.
        """
        per_file_chars = (249 * 1024) // 3  # ~249 KB of UTF-8 per file
        body = "中" * per_file_chars
        first = self.write_file(self.root, "a.md", body)
        second = self.write_file(self.root, "b.md", body)
        raw_bytes = 2 * len(body.encode("utf-8"))
        self.assertLessEqual(len(body.encode("utf-8")),
                             config_mod.GROK_FILES_TEXT_MAX_BYTES)
        self.assertLessEqual(raw_bytes, config_mod.GROK_FILES_TEXT_TOTAL_MAX_BYTES)
        self.assertGreater(raw_bytes * 2, config_mod.GROK_FILES_MAX_PAYLOAD_BYTES,
                           "fixture must be big enough that ASCII escaping WOULD "
                           "have exceeded the payload cap")

        env, fake = self.call([first, second])
        self.assertTrue(env.get("ok"), env)
        payload = self.payload(fake)
        self.assertEqual([b.get("type") for b in payload["content"]],
                         ["text", "resource", "resource"])

    def test_g12_payload_cap_refusal_with_text_is_files_too_large(self):
        # Same wall, but reached with text: it is a size problem, not a type one,
        # and the advice must not mention images that are not in the payload.
        txt = self.write_file(self.root, "notes.txt", "x" * 4096)
        fake = FakeRunCli(stdout="GROK REPLY")
        with shrunk_cap(1024), sandbox_env(self.root), patched_cli(grok_mod, fake):
            env = self.run_grok(files=[txt])
        self.assertRefusal(env, "files_too_large")
        self.assertEqual(env.get("error_code"), "source_too_large", env)
        self.assertIn("cap", env["detail"].lower(), env)
        self.assertNotIn("image", env["detail"].lower(),
                         "detail must not blame images: %r" % env["detail"])
        self.assertNotCalled(fake)

    def test_g12_non_utf8_text_file_is_loud_never_skipped(self):
        bad = self.write_file(self.root, "latin1.txt", b"caf\xe9\n")
        env, fake = self.call([bad])
        self.assertRefusal(env, "unsupported_files")
        self.assertEqual(env.get("error_code"), "source_invalid_utf8", env)
        self.assertNotCalled(fake)

    def test_g12_text_file_outside_the_roots_still_hits_the_fence(self):
        outside = self.write_file(self.mkdir("omni_files_out_"), "notes.txt", "hi\n")
        env, fake = self.call([outside])
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("error_code"), "source_path_not_allowed", env)
        self.assertNotCalled(fake)

    def test_g12_call_tool_routes_a_text_file_through_grok(self):
        txt = self.write_file(self.root, "notes.txt", "probe-token-xyz123\n")
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(self.root), patched_cli(grok_mod, fake):
            env = asyncio.run(dispatch_mod.call_tool("llm_generate", {
                "prompt": "quote the token",
                "model": "grok-4.5",
                "files": [txt],
                "timeout_s": 30,
            }))
        self.assertTrue(env.get("ok"), env)
        payload = self.payload(fake)
        self.assertEqual(payload["content"][1]["resource"]["text"],
                         "probe-token-xyz123\n")


# --------------------------------------------------------------------------- #
# g11. MCP shape — the same wiring must be reachable through call_tool
# --------------------------------------------------------------------------- #
@unittest.skipUnless(GROK_TAKES_FILES, "run_grok_subprocess(files=...) not implemented yet")
class TestMcpCallToolShape(FilesTestCase):
    def test_g11_call_tool_llm_generate_reaches_the_prompt_json_path(self):
        root = self.mkdir("omni_files_root_")
        png = self.write_file(root, "shot.png")
        fake = FakeRunCli(stdout="GROK REPLY")
        with sandbox_env(root), patched_cli(grok_mod, fake):
            env = asyncio.run(dispatch_mod.call_tool("llm_generate", {
                "prompt": "describe the image",
                "model": "grok-4.5",
                "files": [png],
                "timeout_s": 30,
            }))
        self.assertTrue(env.get("ok"), env)
        self.assertIn("--prompt-json", fake.argv, fake.argv)
        self.assertNotIn("--prompt-file", fake.argv, fake.argv)

    @unittest.skipUnless(CODEX_TAKES_FILES, "run_codex_subprocess(files=...) not implemented yet")
    def test_g11_call_tool_llm_generate_reaches_the_codex_i_path(self):
        png = self.write_file(self.mkdir(), "shot.png")
        fake = FakeRunCli()
        with sandbox_env(), patched_cli(codex_mod, fake):
            env = asyncio.run(dispatch_mod.call_tool("llm_generate", {
                "prompt": "describe the image",
                "model": "codex/gpt-5.6-sol",
                "files": [png],
                "timeout_s": 30,
            }))
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(flag_values(fake.argv, "-i"), [png], fake.argv)


# --------------------------------------------------------------------------- #
# I7. advertised capability — backends_snapshot + tool schema
# --------------------------------------------------------------------------- #
class TestAdvertisedCapability(unittest.TestCase):
    def test_backends_snapshot_marks_codex_and_grok_as_files_capable(self):
        snap = dispatch_mod.backends_snapshot()
        self.assertIs(snap["agy"]["files"], True)
        self.assertIs(snap["codex"]["files"], True)
        self.assertIs(snap["grok"]["files"], True)

    def test_backends_snapshot_keeps_the_rest_files_false(self):
        snap = dispatch_mod.backends_snapshot()
        for name in ("claude", "pi", "cursor", "openai", "ollama", "lmstudio"):
            with self.subTest(backend=name):
                self.assertIs(snap[name]["files"], False)

    def test_llm_generate_schema_documents_the_three_providers(self):
        tool = [t for t in dispatch_mod.TOOLS if t["name"] == "llm_generate"][0]
        desc = tool["inputSchema"]["properties"]["files"]["description"].lower()
        for token in ("agy", "codex", "grok"):
            with self.subTest(token=token):
                self.assertIn(token, desc,
                              "files= schema description must name %s" % token)


if __name__ == "__main__":
    unittest.main()
