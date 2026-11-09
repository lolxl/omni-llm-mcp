# no-opencc
"""Acceptance tests — job-lane images (issue #18 P4, plan R2 + sol T7 ruling).

Pinned contract (Owner-approved S2 mechanism, 2026-08-11):
  * jobs.start(images=[...]) stages each image by reading its BYTES through
    the existing O_NOFOLLOW source-roots reader and writing an immutable copy
    under <workdir>/.omni-attach/<job_id>/ — ALL backends consume the COPY,
    never the original path (kills the TOCTOU re-open);
  * codex argv carries `-i <copy>` per image and never the source path;
    claude/grok get a path-reference line for the copies in their prompt
    payload and NO `-i`;
  * a source outside OMNI_LLM_SOURCE_ROOTS refuses the START with the
    existing typed reason "unsupported_files" (job dir exists, state failed,
    refusal IS result.json, no process spawned) — same for an over-cap file;
  * `.omni-attach/` is appended to the workdir's .git/info/exclude (tracked
    files untouched) so the attach dir never pollutes the diff.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-p4-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP

from providers import jobs  # noqa: E402
from providers.attachments import stage_images  # noqa: E402

FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_ARGV_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
data = sys.stdin.read() if not sys.stdin.isatty() else ""
with open(os.environ["FAKE_STDIN_LOG"], "a") as f:
    f.write(data)
if "--json" in sys.argv:
    print(json.dumps({"type": "thread.started", "thread_id": "t-img"}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "seen"}}), flush=True)
else:
    sys.stdout.write("seen")
'''


def _wait_terminal(job_dir, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(job_dir)["state"]
        if st in ("done", "failed", "stopped"):
            return st
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


class ImageLaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="omni-p4-fix-")
        cls.bindir = os.path.join(cls.root, "bin")
        os.makedirs(cls.bindir)
        for name in ("codex", "claude"):
            p = os.path.join(cls.bindir, name)
            with open(p, "w") as f:
                f.write(FAKE_CLI)
            os.chmod(p, 0o755)
        cls.workdir = os.path.join(cls.root, "wd")
        os.makedirs(cls.workdir)
        subprocess.run(["git", "init", "-q", "-b", "work", cls.workdir],
                       check=True)
        cls.srcroot = os.path.join(cls.root, "allowed")
        os.makedirs(cls.srcroot)
        cls.img = os.path.join(cls.srcroot, "shot.png")
        with open(cls.img, "wb") as f:
            f.write(b"\x89PNG-fake-bytes-" + b"z" * 64)
        cls.outside = os.path.join(cls.root, "outside.png")
        with open(cls.outside, "wb") as f:
            f.write(b"nope")
        cls.argv_log = os.path.join(cls.root, "argv.jsonl")
        cls.stdin_log = os.path.join(cls.root, "stdin.log")
        os.environ["PATH"] = cls.bindir + os.pathsep + os.environ["PATH"]
        os.environ["FAKE_ARGV_LOG"] = cls.argv_log
        os.environ["FAKE_STDIN_LOG"] = cls.stdin_log
        # The fenced fake CLI gets an allowlisted environment (#138): its
        # control variables are passed by name, as an operator would.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = "FAKE_ARGV_LOG,FAKE_STDIN_LOG"
        cls.addClassCleanup(os.environ.pop, "OMNI_LLM_CHILD_ENV_PASS", None)
        os.environ["OMNI_LLM_SOURCE_ROOTS"] = cls.srcroot

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        for p in (self.argv_log, self.stdin_log):
            open(p, "w").close()

    def _argv_flat(self):
        with open(self.argv_log) as f:
            return [a for l in f if l.strip() for a in json.loads(l)]

    def test_codex_gets_copy_via_dash_i_never_the_source(self):
        started = jobs.start(task="look", model="codex/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.img])
        job_dir = started["job_dir"]
        self.assertNotIn("refused", started)
        self.assertEqual(_wait_terminal(job_dir), "done")
        flat = self._argv_flat()
        self.assertIn("-i", flat)
        copy = flat[flat.index("-i") + 1]
        self.assertTrue(copy.startswith(
            os.path.join(os.path.realpath(self.workdir), ".omni-attach")),
            copy)
        with open(copy, "rb") as f_copy, open(self.img, "rb") as f_src:
            self.assertEqual(f_copy.read(), f_src.read())
        self.assertNotIn(self.img, flat, "source path must never reach argv")

    def test_claude_gets_path_reference_in_prompt_no_dash_i(self):
        started = jobs.start(task="look", model="claude/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.img])
        self.assertNotIn("refused", started)
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        self.assertNotIn("-i", self._argv_flat())
        with open(self.stdin_log) as f:
            stdin_text = f.read()
        self.assertIn(".omni-attach", stdin_text)
        self.assertNotIn(self.img, stdin_text)

    def test_source_outside_roots_refuses_start_with_typed_reason(self):
        started = jobs.start(task="look", model="codex/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.outside])
        job_dir = started["job_dir"]
        st = jobs.status(job_dir)
        self.assertEqual(st["state"], "failed")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "unsupported_files")
        self.assertIsNone(st.get("pid"), "refused start must not spawn")
        self.assertEqual(self._argv_flat(), [])

    def test_oversized_image_refuses_start(self):
        big = os.path.join(self.srcroot, "big.png")
        with open(big, "wb") as f:
            f.write(b"x" * (6 * 1024 * 1024))
        try:
            started = jobs.start(task="look", model="codex/fake",
                                 workdir=self.workdir, timeout_s=60,
                                 images=[big])
            with open(os.path.join(started["job_dir"], "result.json")) as f:
                res = json.load(f)
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "unsupported_files")
        finally:
            os.unlink(big)

    def test_attach_dir_is_git_excluded_not_tracked_ignored(self):
        started = jobs.start(task="look", model="codex/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.img])
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        exclude = os.path.join(self.workdir, ".git", "info", "exclude")
        with open(exclude) as f:
            self.assertIn(".omni-attach/", f.read())
        self.assertFalse(
            os.path.exists(os.path.join(self.workdir, ".gitignore")),
            "tracked ignore files must not be touched")
        porcelain = subprocess.run(
            ["git", "-C", self.workdir, "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
        self.assertNotIn(".omni-attach", porcelain,
                         "attach dir must not appear in git status")

    def test_multi_image_staging_both_in_argv_and_prompt(self):
        img2 = os.path.join(self.srcroot, "shot2.png")
        with open(img2, "wb") as f:
            f.write(b"\x89PNG-second-" + b"y" * 32)
        try:
            started = jobs.start(task="look both", model="codex/fake",
                                 workdir=self.workdir, timeout_s=60,
                                 images=[self.img, img2])
            job_dir = started["job_dir"]
            self.assertNotIn("refused", started)
            self.assertEqual(_wait_terminal(job_dir), "done")
            with open(os.path.join(job_dir, "job.json")) as f:
                job = json.load(f)
            staged = job.get("images") or []
            self.assertEqual(len(staged), 2)
            attach_prefix = os.path.join(
                os.path.realpath(self.workdir), ".omni-attach")
            for copy in staged:
                self.assertTrue(copy.startswith(attach_prefix), copy)
                self.assertTrue(os.path.isfile(copy), copy)
            flat = self._argv_flat()
            i_flags = [i for i, a in enumerate(flat) if a == "-i"]
            self.assertEqual(len(i_flags), 2)
            for idx in i_flags:
                self.assertTrue(flat[idx + 1].startswith(attach_prefix))
            self.assertNotIn(self.img, flat)
            self.assertNotIn(img2, flat)

            open(self.argv_log, "w").close()
            open(self.stdin_log, "w").close()
            started2 = jobs.start(task="look both", model="claude/fake",
                                  workdir=self.workdir, timeout_s=60,
                                  images=[self.img, img2])
            self.assertNotIn("refused", started2)
            self.assertEqual(_wait_terminal(started2["job_dir"]), "done")
            self.assertNotIn("-i", self._argv_flat())
            with open(self.stdin_log) as f:
                stdin_text = f.read()
            with open(os.path.join(started2["job_dir"], "job.json")) as jf:
                copies = json.load(jf).get("images") or []
            for copy in copies:
                self.assertIn(copy, stdin_text)
            self.assertNotIn(self.img, stdin_text)
            self.assertNotIn(img2, stdin_text)
        finally:
            try:
                os.unlink(img2)
            except OSError:
                pass

    def test_name_collision_dedup(self):
        sub = os.path.join(self.srcroot, "other")
        os.makedirs(sub, exist_ok=True)
        twin = os.path.join(sub, "shot.png")
        with open(twin, "wb") as f:
            f.write(b"\x89PNG-twin-" + b"w" * 32)
        try:
            started = jobs.start(task="look", model="codex/fake",
                                 workdir=self.workdir, timeout_s=60,
                                 images=[self.img, twin])
            self.assertNotIn("refused", started)
            self.assertEqual(_wait_terminal(started["job_dir"]), "done")
            with open(os.path.join(started["job_dir"], "job.json")) as f:
                staged = json.load(f).get("images") or []
            self.assertEqual(len(staged), 2)
            basenames = [os.path.basename(p) for p in staged]
            self.assertEqual(len(set(basenames)), 2,
                             "collision must yield distinct basenames: %r"
                             % basenames)
            # Safe names are index-based (img-N.ext); source basename unused.
            self.assertEqual(sorted(basenames), ["img-1.png", "img-2.png"])
            self.assertNotIn("shot.png", basenames)
            for copy in staged:
                self.assertTrue(os.path.isfile(copy))
        finally:
            try:
                os.unlink(twin)
            except OSError:
                pass

    def test_total_cap_refusal_under_per_file_cap(self):
        # Each file under 5 MiB per-file cap; sum over 20 MiB total cap.
        chunk = 4 * 1024 * 1024 + 512 * 1024  # 4.5 MiB each
        paths = []
        try:
            for i in range(5):
                p = os.path.join(self.srcroot, "bulk-%d.png" % i)
                with open(p, "wb") as f:
                    f.write(b"b" * chunk)
                paths.append(p)
            started = jobs.start(task="look", model="codex/fake",
                                 workdir=self.workdir, timeout_s=60,
                                 images=paths)
            job_dir = started["job_dir"]
            st = jobs.status(job_dir)
            self.assertEqual(st["state"], "failed")
            with open(os.path.join(job_dir, "result.json")) as f:
                res = json.load(f)
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "unsupported_files")
            self.assertIn("total", (res.get("detail") or "").lower())
            self.assertIsNone(st.get("pid"), "cap refusal must not spawn")
            self.assertEqual(self._argv_flat(), [])
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_non_image_extension_refused_loudly(self):
        """`--image` is image-only on every backend (issue #53 point 4): a
        non-image extension must refuse rather than be copied in unfiltered."""
        for name in ("doc.pdf", "blob.bin", "notes.txt", "noext"):
            with self.subTest(name=name):
                src = os.path.join(self.srcroot, name)
                with open(src, "wb") as f:
                    f.write(b"not-an-image")
                try:
                    staged, refusal = stage_images(
                        self.workdir, "type-fence-%s" % name, [src])
                    self.assertEqual(staged, [])
                    self.assertIsNotNone(refusal, "non-image must be refused")
                    self.assertFalse(refusal["ok"])
                    self.assertEqual(refusal["reason"], "unsupported_files")
                    self.assertIn(".png", refusal["detail"],
                                  "detail must name the accepted extensions")
                finally:
                    os.unlink(src)

    def test_non_image_refusal_leaves_no_staged_copies(self):
        """A good image followed by a bad one stages nothing at all."""
        bad = os.path.join(self.srcroot, "second.pdf")
        with open(bad, "wb") as f:
            f.write(b"%PDF-1.4")
        job_id = "type-fence-partial"
        try:
            staged, refusal = stage_images(
                self.workdir, job_id, [self.img, bad])
            self.assertEqual(staged, [])
            self.assertEqual(refusal["reason"], "unsupported_files")
            leftover = os.path.join(
                os.path.realpath(self.workdir), ".omni-attach", job_id)
            self.assertFalse(os.path.isdir(leftover),
                             "partial staging must be cleaned up")
        finally:
            os.unlink(bad)

    def test_start_refuses_non_image_attachment(self):
        src = os.path.join(self.srcroot, "report.pdf")
        with open(src, "wb") as f:
            f.write(b"%PDF-1.4 fake")
        try:
            started = jobs.start(task="look", model="codex/fake",
                                 workdir=self.workdir, timeout_s=60,
                                 images=[src])
            job_dir = started["job_dir"]
            st = jobs.status(job_dir)
            self.assertEqual(st["state"], "failed")
            with open(os.path.join(job_dir, "result.json")) as f:
                res = json.load(f)
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "unsupported_files")
            self.assertIsNone(st.get("pid"), "type refusal must not spawn")
            self.assertEqual(self._argv_flat(), [])
        finally:
            os.unlink(src)

    def test_exclude_idempotence_second_job(self):
        started1 = jobs.start(task="look1", model="codex/fake",
                              workdir=self.workdir, timeout_s=60,
                              images=[self.img])
        self.assertEqual(_wait_terminal(started1["job_dir"]), "done")
        exclude = os.path.join(self.workdir, ".git", "info", "exclude")
        with open(exclude) as f:
            first = f.read()
        self.assertEqual(first.count(".omni-attach/"), 1)

        started2 = jobs.start(task="look2", model="codex/fake",
                              workdir=self.workdir, timeout_s=60,
                              images=[self.img])
        self.assertEqual(_wait_terminal(started2["job_dir"]), "done")
        with open(exclude) as f:
            second = f.read()
        self.assertEqual(second.count(".omni-attach/"), 1,
                         "second job must not duplicate the exclude line")
        self.assertEqual(first, second)

    def test_anti_toctou_read_once_uses_fenced_bytes_not_later_symlink(self):
        """Staged copy must use bytes from the single fenced read, not a
        later re-open of the source path (kills shutil.copyfile(source))."""
        fenced = b"FENCED-BYTES-FROM-FIRST-READ-v1"
        secret = b"SECRET-OUT-OF-FENCE-MUST-NOT-APPEAR"
        secret_dir = os.path.join(self.root, "toctou-secret")
        os.makedirs(secret_dir, exist_ok=True)
        secret_path = os.path.join(secret_dir, "secret.bin")
        with open(secret_path, "wb") as f:
            f.write(secret)

        victim = os.path.join(self.srcroot, "toctou-victim.png")
        with open(victim, "wb") as f:
            f.write(b"placeholder-before-swap")

        def fake_open_binary(path, max_bytes, roots=None):
            # Simulate a successful fenced read of fixed bytes; never touch
            # whatever currently sits at *path* (may already be a symlink).
            return fenced, None

        try:
            with mock.patch(
                    "providers.attachments.open_binary_under_roots",
                    side_effect=fake_open_binary):
                # After the reader is replaced, point the source path at an
                # out-of-fence secret. A copyfile(source) impl would leak it.
                os.unlink(victim)
                os.symlink(secret_path, victim)
                job_id = "toctou-kill-%d" % os.getpid()
                staged, refusal = stage_images(
                    self.workdir, job_id, [victim])
            self.assertIsNone(refusal, refusal)
            self.assertEqual(len(staged), 1)
            with open(staged[0], "rb") as f:
                got = f.read()
            self.assertEqual(got, fenced)
            self.assertNotEqual(got, secret)
            self.assertNotIn(secret, got)
        finally:
            try:
                if os.path.islink(victim) or os.path.exists(victim):
                    os.unlink(victim)
            except OSError:
                pass
            try:
                os.unlink(secret_path)
            except OSError:
                pass
            try:
                os.rmdir(secret_dir)
            except OSError:
                pass

    def test_staged_copy_mode_is_0o600(self):
        started = jobs.start(task="look", model="codex/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.img])
        self.assertNotIn("refused", started)
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        with open(os.path.join(started["job_dir"], "job.json")) as f:
            staged = json.load(f).get("images") or []
        self.assertEqual(len(staged), 1)
        mode = os.stat(staged[0]).st_mode & 0o777
        self.assertEqual(mode, 0o600, "staged copy mode must be 0o600, got %o"
                         % mode)

    def test_prompt_block_carries_attached_images_header(self):
        started = jobs.start(task="look", model="claude/fake",
                             workdir=self.workdir, timeout_s=60,
                             images=[self.img])
        self.assertNotIn("refused", started)
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        with open(self.stdin_log) as f:
            stdin_text = f.read()
        self.assertIn("[attached images]", stdin_text)
        # Header must be its own line, not merely buried in a path fragment.
        self.assertIn("\n[attached images]\n", stdin_text)
        prompt_path = os.path.join(started["job_dir"], "prompt.txt")
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read()
        self.assertIn("\n[attached images]\n", prompt)


if __name__ == "__main__":
    unittest.main()
