# no-opencc
"""Acceptance tests — P3 runner integration (issue #18, plan R2, WO-5b).

Pinned contract for the JOB lane (sync lane untouched):
  * child stdout goes raw to <job_dir>/stream.raw.log while the job runs (the
    in-run channel to the normalizer; deleted when the job reaches a terminal
    state, #170), child stderr to <job_dir>/stream.err.log, and the omni
    done-footer meta line appears in NEITHER (supervisor.log owns meta);
  * <job_dir>/stream.log holds ONLY contract lines (stream_contract
    .is_contract_line) normalized from the raw stream, fully drained BEFORE
    result.json exists;
  * result.text for a streaming grok/claude job is the assistant's final
    answer (stream_contract.final_text), not event JSONL;
  * `omni agent tail <job>` prints stream.log content; `--raw` prints the
    raw log while the job runs and, after it ended, a typed refusal
    (`raw_log_not_retained`).

The fake backend replays the REAL scrubbed grok fixture line by line.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="omni-p3-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP

from providers import jobs, stream_contract  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE = os.path.join(_REPO, "tests", "fixtures", "grok_stream.jsonl")

FAKE_GROK = r'''#!/usr/bin/env python3
import os, sys
if not sys.stdin.isatty():
    sys.stdin.read()
sys.stderr.write("warn: fixture replay\n")
with open(os.environ["FAKE_FIXTURE"], encoding="utf-8") as f:
    for line in f:
        sys.stdout.write(line)
        sys.stdout.flush()
'''


def _wait_terminal(job_dir, timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(job_dir)["state"]
        if st in ("done", "failed", "stopped"):
            return st
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


class StreamIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Re-pin state dir when co-imported with other job suites.
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        cls.root = tempfile.mkdtemp(prefix="omni-p3-test-")
        cls.bindir = os.path.join(cls.root, "bin")
        os.makedirs(cls.bindir)
        fake = os.path.join(cls.bindir, "grok")
        with open(fake, "w") as f:
            f.write(FAKE_GROK)
        os.chmod(fake, 0o755)
        cls.workdir = os.path.join(cls.root, "wd")
        os.makedirs(cls.workdir)
        subprocess.run(["git", "init", "-q", "-b", "work", cls.workdir],
                       check=True)
        old_env = {k: os.environ.get(k) for k in ("PATH", "FAKE_FIXTURE")}

        def restore_env():
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        # A class cleanup also runs when setUpClass raises past this point.
        cls.addClassCleanup(restore_env)
        os.environ["PATH"] = cls.bindir + os.pathsep + os.environ["PATH"]
        # The host fence's read wall covers the checkout (it lives under
        # home), so the fenced fake CLI cannot open the fixture there. Serve
        # a copy from the temp root, which the fence admits.
        fixture = os.path.join(cls.root, "grok_stream.jsonl")
        shutil.copyfile(_FIXTURE, fixture)
        os.environ["FAKE_FIXTURE"] = fixture
        # The fenced fake CLI gets an allowlisted environment (#138).
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = "FAKE_FIXTURE"
        cls.addClassCleanup(os.environ.pop, "OMNI_LLM_CHILD_ENV_PASS", None)
        started = jobs.start(task="replay", model="grok-4.5",
                             workdir=cls.workdir, timeout_s=60)
        cls.job_dir = started["job_dir"]
        cls.final = _wait_terminal(cls.job_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _read(self, name):
        path = os.path.join(self.job_dir, name)
        self.assertTrue(os.path.isfile(path), "%s missing" % name)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def test_job_completed(self):
        self.assertEqual(self.final, "done")

    def test_raw_and_err_streams_are_separated(self):
        # #170: the raw log is not retained once the job has ended.
        self.assertFalse(os.path.exists(
            os.path.join(self.job_dir, "stream.raw.log")))
        err = self._read("stream.err.log")
        self.assertIn("warn: fixture replay", err)
        self.assertNotIn("warn: fixture replay", self._read("stream.log"))

    def test_meta_footer_stays_out_of_stream_logs(self):
        for name in ("stream.log", "stream.err.log"):
            self.assertNotIn("[omni-llm-mcp] done", self._read(name), name)

    def test_stream_log_is_all_contract_lines_and_complete(self):
        lines = [l for l in self._read("stream.log").splitlines() if l.strip()]
        self.assertTrue(lines)
        kinds = set()
        for line in lines:
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)
            kinds.add(obj["type"])
        self.assertIn("content_block_delta", kinds)
        self.assertIn("message_stop", kinds)

    def test_result_text_is_final_answer_not_jsonl(self):
        with open(os.path.join(self.job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res.get("text", "").strip(), "PONG")

    def test_cli_tail_serves_contract_and_raw(self):
        out = subprocess.run(
            [sys.executable, os.path.join(_REPO, "omni_cli.py"),
             "agent", "tail", self.job_dir, "-n", "500"],
            capture_output=True, text=True,
            env=dict(os.environ, OMNI_LLM_STATE_DIR=_TMP))
        self.assertEqual(out.returncode, 0, out.stderr[-300:])
        self.assertIn('"content_block_delta"', out.stdout)
        raw = subprocess.run(
            [sys.executable, os.path.join(_REPO, "omni_cli.py"),
             "agent", "tail", self.job_dir, "-n", "500", "--raw"],
            capture_output=True, text=True,
            env=dict(os.environ, OMNI_LLM_STATE_DIR=_TMP))
        # #170: after the end the raw log is gone; the CLI says so, typed.
        self.assertEqual(raw.returncode, 1, raw.stderr[-300:])
        self.assertEqual(json.loads(raw.stdout)["reason"], "raw_log_not_retained")

    # --- WO-5c addition: --err pin on the fixture-replay job (assertions above untouchable) ---
    def test_cli_tail_err_serves_stderr(self):
        """CLI tail --err returns the fixture's stderr line."""
        err = subprocess.run(
            [sys.executable, os.path.join(_REPO, "omni_cli.py"),
             "agent", "tail", self.job_dir, "-n", "500", "--err"],
            capture_output=True, text=True,
            env=dict(os.environ, OMNI_LLM_STATE_DIR=_TMP))
        self.assertEqual(err.returncode, 0, err.stderr[-300:])
        self.assertIn("warn: fixture replay", err.stdout)


# --- WO-5b additions (committed assertions above are untouchable) ---

FAKE_CODEX_STREAM = r'''#!/usr/bin/env python3
import json, os, sys
data = sys.stdin.read() if not sys.stdin.isatty() else ""
# Match test_resume FAKE_CLI job-lane shape: thread.started + agent_message.
if "--json" in sys.argv:
    print(json.dumps({"type": "thread.started",
                      "thread_id": os.environ.get("FAKE_THREAD_ID", "t-stream")}),
          flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message",
                               "text": "fake-done"}}),
          flush=True)
else:
    sys.stdout.write("fake-done")
'''

FAKE_PARTIAL_LINES = r'''#!/usr/bin/env python3
import os, sys, time
if not sys.stdin.isatty():
    sys.stdin.read()
sys.stderr.write("partial-fake stderr\n")
# Emit one JSONL line in two chunks with a delay so the normalizer must
# buffer the incomplete first half across poll reads.
line = (
    '{"type":"stream_event","event":{"type":"content_block_delta",'
    '"index":0,"delta":{"type":"text_delta","text":"HI"}}}\n'
)
half = len(line) // 2
sys.stdout.write(line[:half])
sys.stdout.flush()
time.sleep(0.5)
sys.stdout.write(line[half:])
sys.stdout.flush()
# Trailing complete message_stop so stream.log has a known end event.
sys.stdout.write(
    '{"type":"stream_event","event":{"type":"message_stop"}}\n')
sys.stdout.flush()
'''

FAKE_MID_STREAM_STOP = r'''#!/usr/bin/env python3
import os, sys, time
if not sys.stdin.isatty():
    sys.stdin.read()
# Emit a few complete events, then a HALF line (no newline, flush) and hang
# so stop() can interrupt mid-line — pins stop-mid-line parseability.
for piece in (
    '{"type":"stream_event","event":{"type":"message_start"}}\n',
    '{"type":"stream_event","event":{"type":"content_block_delta",'
    '"index":0,"delta":{"type":"text_delta","text":"X"}}}\n',
):
    sys.stdout.write(piece)
    sys.stdout.flush()
    time.sleep(0.05)
# Half-written last line: no trailing newline, flushed to raw log.
sys.stdout.write(
    '{"type":"stream_event","event":{"type":"content_block_delta",'
    '"index":0,"delta":{"type":"text_delta","text":"HALF')
sys.stdout.flush()
time.sleep(30)
'''


def _wait_terminal_extra(job_dir, timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(job_dir)["state"]
        if st in ("done", "failed", "stopped"):
            return st
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


class StreamIntegrationExtrasTest(unittest.TestCase):
    """WO-5b additions: codex contract synthesis, partial lines, stop mid-stream."""

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        self.root = tempfile.mkdtemp(prefix="omni-p3-extra-")
        self.bindir = os.path.join(self.root, "bin")
        os.makedirs(self.bindir)
        self.workdir = os.path.join(self.root, "wd")
        os.makedirs(self.workdir)
        subprocess.run(["git", "init", "-q", "-b", "work", self.workdir],
                       check=True)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self.bindir + os.pathsep + self._old_path

    def tearDown(self):
        os.environ["PATH"] = self._old_path
        shutil.rmtree(self.root, ignore_errors=True)

    def _install(self, name, body):
        path = os.path.join(self.bindir, name)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, 0o755)
        return path

    def test_codex_job_stream_log_has_synthesized_messages_sequence(self):
        """Codex job stream.log contains the synthesized Messages sequence."""
        self._install("codex", FAKE_CODEX_STREAM)
        started = jobs.start(task="replay", model="codex/fake",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        self.assertEqual(_wait_terminal_extra(job_dir), "done")
        with open(os.path.join(job_dir, "stream.log"), encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertTrue(lines)
        kinds = []
        for line in lines:
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)
            kinds.append(obj["type"])
        # item.completed agent_message → message_start … message_stop sequence
        self.assertIn("message_start", kinds)
        self.assertIn("content_block_delta", kinds)
        self.assertIn("message_stop", kinds)
        # thread.started is not a Messages event → x_omni_passthrough
        self.assertTrue(any(k == "x_omni_passthrough" for k in kinds), kinds)

    def test_normalizer_partial_line_safety(self):
        """Line written in two chunks still yields parseable stream.log."""
        self._install("grok", FAKE_PARTIAL_LINES)
        started = jobs.start(task="partial", model="grok-4.5",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        self.assertEqual(_wait_terminal_extra(job_dir), "done")
        with open(os.path.join(job_dir, "stream.log"), encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertTrue(lines)
        kinds = set()
        for line in lines:
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)
            kinds.add(obj["type"])
        self.assertIn("content_block_delta", kinds)
        self.assertIn("message_stop", kinds)

    def _hard_stop(self, job_dir, grace_s=1, reason="mid-stream"):
        """stop() plus SIGKILL fallback when pid identity is unverifiable (no ps)."""
        import signal
        try:
            jobs.stop(job_dir, grace_s=grace_s, reason=reason)
        except (OSError, ValueError):
            pass
        try:
            data = jobs.status(job_dir)
        except (OSError, ValueError):
            return
        if data.get("state") in ("done", "failed", "stopped"):
            return
        for key in ("supervisor_pgid", "pgid", "backend_pgid"):
            pgid = data.get(key)
            if pgid is None:
                continue
            try:
                os.killpg(int(pgid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError, ValueError):
                pass
        deadline = time.time() + 3
        while time.time() < deadline and jobs._lock_is_held(job_dir):
            time.sleep(0.05)
        try:
            jobs.stop(job_dir, grace_s=0, reason=reason)
        except (OSError, ValueError):
            pass

    def test_stop_mid_stream_leaves_stream_log_parseable(self):
        """stop mid-stream: every stream.log line is valid contract JSON."""
        self._install("grok", FAKE_MID_STREAM_STOP)
        started = jobs.start(task="hang", model="grok-4.5",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        # Wait until at least one contract line is visible, then stop.
        deadline = time.time() + 10
        saw = False
        while time.time() < deadline:
            path = os.path.join(job_dir, "stream.log")
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                saw = True
                break
            time.sleep(0.05)
        self.assertTrue(saw, "stream.log never received data before stop")
        try:
            self._hard_stop(job_dir, grace_s=2, reason="mid-stream")
            st = _wait_terminal_extra(job_dir, timeout=10)
            self.assertIn(st, ("stopped", "failed", "done"), st)
            path = os.path.join(job_dir, "stream.log")
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            self.assertTrue(lines, "stream.log empty after mid-stream stop")
            for line in lines:
                obj = json.loads(line)  # must be valid JSON
                self.assertTrue(stream_contract.is_contract_line(obj), obj)
        finally:
            self._hard_stop(job_dir, grace_s=0, reason="cleanup")


# --- WO-5c additions (WO-5b assertions above are untouchable) ---

FAKE_SURROGATE_THEN_ANSWER = r'''#!/usr/bin/env python3
"""Emit a lone-surrogate delta, then a valid assistant ANSWER; exit 0.

Pins: normalizer must not kill a successful job when stream.log would
otherwise raise UnicodeEncodeError on "\ud800".
"""
import json, os, sys
if not sys.stdin.isatty():
    sys.stdin.read()
# Build a stream_event whose delta text is the lone surrogate U+D800.
# ensure_ascii=True puts the \ud800 escape on the wire so the raw log is
# valid UTF-8; the normalizer's json.loads then reintroduces the surrogate.
evt = {
    "type": "stream_event",
    "event": {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "\ud800"},
    },
}
sys.stdout.write(json.dumps(evt, ensure_ascii=True) + "\n")
sys.stdout.flush()
# Valid assistant line with text ANSWER (final_text prefers this).
sys.stdout.write(
    '{"type":"assistant","message":{"role":"assistant","content":'
    '[{"type":"text","text":"ANSWER"}]}}\n'
)
sys.stdout.flush()
'''

FAKE_CJK_SPLIT = r'''#!/usr/bin/env python3
"""Write a CJK JSONL line in two byte-chunks split INSIDE a multibyte char.

Pins: bytes-buffer normalizer keeps intact text, zero U+FFFD.
"""
import os, sys, time
if not sys.stdin.isatty():
    sys.stdin.read()
cjk = "你好世界" * 20
line = (
    '{"type":"stream_event","event":{"type":"content_block_delta",'
    '"index":0,"delta":{"type":"text_delta","text":%s}}}\n'
    % (__import__("json").dumps(cjk, ensure_ascii=False))
)
data = line.encode("utf-8")
# Split inside a multibyte character (continuation byte at split point).
split_at = None
for i in range(1, len(data)):
    if (data[i] & 0xC0) == 0x80:
        split_at = i
        break
if split_at is None:
    split_at = max(1, len(data) // 2)
sys.stdout.buffer.write(data[:split_at])
sys.stdout.buffer.flush()
time.sleep(0.5)
sys.stdout.buffer.write(data[split_at:])
sys.stdout.buffer.flush()
# Trailing message_stop so the job has a clean stream end.
sys.stdout.write(
    '{"type":"stream_event","event":{"type":"message_stop"}}\n')
sys.stdout.flush()
# Assistant whole-message so result.text is non-empty / job stays done.
sys.stdout.write(
    '{"type":"assistant","message":{"role":"assistant","content":'
    '[{"type":"text","text":"CJKOK"}]}}\n'
)
sys.stdout.flush()
'''


class StreamIntegrationFixRoundTest(unittest.TestCase):
    """WO-5c pins: surrogate write safety + UTF-8 mid-char split."""

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        self.root = tempfile.mkdtemp(prefix="omni-p3-fix5c-")
        self.bindir = os.path.join(self.root, "bin")
        os.makedirs(self.bindir)
        self.workdir = os.path.join(self.root, "wd")
        os.makedirs(self.workdir)
        subprocess.run(["git", "init", "-q", "-b", "work", self.workdir],
                       check=True)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self.bindir + os.pathsep + self._old_path

    def tearDown(self):
        os.environ["PATH"] = self._old_path
        shutil.rmtree(self.root, ignore_errors=True)

    def _install(self, name, body):
        path = os.path.join(self.bindir, name)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, 0o755)
        return path

    def test_surrogate_delta_does_not_fail_successful_job(self):
        """Lone \\ud800 in a stream_event must not kill job; ANSWER survives."""
        self._install("grok", FAKE_SURROGATE_THEN_ANSWER)
        started = jobs.start(task="surr", model="grok-4.5",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        self.assertEqual(_wait_terminal_extra(job_dir), "done")
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.assertTrue(res.get("ok"), res)
        self.assertEqual((res.get("text") or "").strip(), "ANSWER")
        path = os.path.join(job_dir, "stream.log")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertTrue(lines, "stream.log empty after surrogate job")
        for line in lines:
            # Every line must still be valid JSON (backslashreplace may
            # leave \\ud800 ASCII escape sequences; that is still JSON).
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)

    def test_utf8_mid_char_split_keeps_cjk_intact(self):
        """CJK line split mid-multibyte char → intact text, zero U+FFFD."""
        self._install("grok", FAKE_CJK_SPLIT)
        started = jobs.start(task="cjk", model="grok-4.5",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        self.assertEqual(_wait_terminal_extra(job_dir), "done")
        with open(os.path.join(job_dir, "stream.log"), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("\ufffd", text)
        self.assertNotIn("\uFFFD", text)
        self.assertIn("你好世界", text)
        lines = [l for l in text.splitlines() if l.strip()]
        self.assertTrue(lines)
        for line in lines:
            obj = json.loads(line)
            self.assertTrue(stream_contract.is_contract_line(obj), obj)


if __name__ == "__main__":
    unittest.main()
