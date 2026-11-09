# no-opencc
"""Acceptance tests — resume lane (issue #18 P2, batch-2 plan R2).

Pinned contract:
  * jobs.start records a drift pin in job.json: repo_root (realpath), branch,
    head sha of the workdir at start time;
  * codex job-lane argv drops --ephemeral, adds --json, and job_runner captures
    session_id from the thread.started event (live-verified shape
    {"type":"thread.started","thread_id":"<uuid>"});
  * claude job-lane argv carries a pre-assigned --session-id whose uuid is in
    job.json BEFORE the backend finishes;
  * jobs.resume(source, prompt): refuses non-terminal source
    (resume_not_terminal), refuses drift (resume_drift; allow_drift=True
    overrides), refuses when the source's resume.lock is held (resume_busy) —
    refusals create NO new job and come back as {"refused": envelope};
  * a successful resume creates a NEW job with resumed_from=<source id> and
    backend argv that resumes the parent session (codex: "resume <parent id>").
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="omni-resume-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP

from providers import jobs  # noqa: E402

FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["FAKE_ARGV_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
data = sys.stdin.read() if not sys.stdin.isatty() else ""
# Job-lane codex: emit thread.started first (flush) so a mid-run stop still
# leaves session_id behind; then honour FAKE_DELAY_S to stay non-terminal.
if "--json" in sys.argv:
    print(json.dumps({"type": "thread.started",
                      "thread_id": os.environ.get("FAKE_THREAD_ID", "t-000")}),
          flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "fake-done"}}),
          flush=True)
delay = float(os.environ.get("FAKE_DELAY_S", "0"))
time.sleep(delay)
if "--json" not in sys.argv:
    sys.stdout.write("fake-done")
'''


def _git(*args, cwd):
    return subprocess.run(["git"] + list(args), cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _wait_terminal(job_dir, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(job_dir)["state"]
        if st in ("done", "failed", "stopped"):
            return st
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


def _hard_stop(job_dir, grace_s=1, reason="cleanup"):
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


class ResumeLaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="omni-resume-fix-")
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
        open(os.path.join(cls.workdir, "seed"), "w").write("x")
        _git("add", "seed", cwd=cls.workdir)
        _git("-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "seed", cwd=cls.workdir)
        cls.argv_log = os.path.join(cls.root, "argv.jsonl")
        os.environ["PATH"] = cls.bindir + os.pathsep + os.environ["PATH"]
        os.environ["FAKE_ARGV_LOG"] = cls.argv_log
        # The fenced fake CLI gets an allowlisted environment (#138): its
        # control variables are passed by name, as an operator would.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = (
            "FAKE_ARGV_LOG,FAKE_THREAD_ID,FAKE_DELAY_S")
        cls.addClassCleanup(os.environ.pop, "OMNI_LLM_CHILD_ENV_PASS", None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        # Re-pin state dir: co-imported suites (test_jobs etc.) also set
        # OMNI_LLM_STATE_DIR at import time; without this, jobs land under
        # another suite's tmp while assertions list _TMP/jobs.
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        open(self.argv_log, "w").close()

    def _argvs(self):
        with open(self.argv_log) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _start(self, model="codex/fake", env=None):
        old = {}
        for k, v in (env or {}).items():
            old[k], os.environ[k] = os.environ.get(k), v
        try:
            return jobs.start(task="hello", model=model, workdir=self.workdir,
                              timeout_s=60)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_start_records_drift_pin(self):
        started = self._start()
        job_dir = started["job_dir"]
        _wait_terminal(job_dir)
        with open(os.path.join(job_dir, "job.json")) as f:
            data = json.load(f)
        self.assertEqual(data.get("repo_root"),
                         os.path.realpath(self.workdir))
        self.assertEqual(data.get("branch"), "work")
        self.assertEqual(data.get("head"),
                         _git("rev-parse", "HEAD", cwd=self.workdir))

    def test_codex_job_lane_json_no_ephemeral_captures_thread_id(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-abc-123"})
        job_dir = started["job_dir"]
        self.assertEqual(_wait_terminal(job_dir), "done")
        flat = [a for argv in self._argvs() for a in argv]
        self.assertIn("--json", flat)
        self.assertNotIn("--ephemeral", flat)
        with open(os.path.join(job_dir, "job.json")) as f:
            self.assertEqual(json.load(f).get("session_id"), "t-abc-123")

    def test_claude_job_lane_preassigns_session_id(self):
        started = self._start(model="claude/fake",
                              env={"FAKE_DELAY_S": "3"})
        job_dir = started["job_dir"]
        with open(os.path.join(job_dir, "job.json")) as f:
            sid = json.load(f).get("session_id")
        self.assertTrue(sid, "session_id must be pre-assigned before finish")
        _wait_terminal(job_dir)
        flat = [a for argv in self._argvs() for a in argv]
        self.assertIn("--session-id", flat)
        self.assertIn(sid, flat)

    def test_resume_refuses_non_terminal_and_creates_no_job(self):
        started = self._start(env={"FAKE_DELAY_S": "10"})
        before = set(os.listdir(os.path.join(_TMP, "jobs")))
        out = jobs.resume(started["job_dir"], "continue please")
        self.assertIn("refused", out)
        self.assertEqual(out["refused"]["reason"], "resume_not_terminal")
        self.assertEqual(before,
                         set(os.listdir(os.path.join(_TMP, "jobs"))))
        jobs.stop(started["job_dir"], grace_s=1, reason="cleanup")

    def test_resume_refuses_drift_and_allow_drift_overrides(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-drift"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        _git("checkout", "-q", "-b", "moved", cwd=self.workdir)
        try:
            out = jobs.resume(started["job_dir"], "go on")
            self.assertEqual(out.get("refused", {}).get("reason"),
                             "resume_drift")
            out2 = jobs.resume(started["job_dir"], "go on",
                               allow_drift=True)
            self.assertNotIn("refused", out2)
            self.assertIn("job_id", out2)
            _wait_terminal(out2["job_dir"])
        finally:
            _git("checkout", "-q", "work", cwd=self.workdir)

    def test_resume_links_child_and_resumes_parent_session(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-parent-9"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        out = jobs.resume(started["job_dir"], "next step")
        self.assertNotIn("refused", out)
        self.assertEqual(_wait_terminal(out["job_dir"]), "done")
        with open(os.path.join(out["job_dir"], "job.json")) as f:
            child = json.load(f)
        with open(os.path.join(started["job_dir"], "job.json")) as f:
            parent = json.load(f)
        self.assertEqual(child.get("resumed_from"), parent["job_id"])
        joined = [" ".join(argv) for argv in self._argvs()]
        self.assertTrue(any("resume t-parent-9" in j for j in joined),
                        "backend argv must resume the parent session: %r"
                        % joined)

    def test_resume_single_flight_via_source_lock(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-lock"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        lock_path = os.path.join(started["job_dir"], "resume.lock")
        with open(lock_path, "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            out = jobs.resume(started["job_dir"], "busy?")
        self.assertEqual(out.get("refused", {}).get("reason"), "resume_busy")

    def test_grok_resume_argv_fork_session(self):
        # Extra backend for this case only (committed suite uses codex/claude).
        grok_path = os.path.join(self.bindir, "grok")
        if not os.path.isfile(grok_path):
            with open(grok_path, "w") as f:
                f.write(FAKE_CLI)
            os.chmod(grok_path, 0o755)
        started = self._start(model="grok-4.5")
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        with open(os.path.join(started["job_dir"], "job.json")) as f:
            parent_sid = json.load(f).get("session_id")
        self.assertTrue(parent_sid, "grok start must pre-assign session_id")
        open(self.argv_log, "w").close()
        out = jobs.resume(started["job_dir"], "fork please")
        self.assertNotIn("refused", out)
        self.assertEqual(_wait_terminal(out["job_dir"]), "done")
        with open(os.path.join(out["job_dir"], "job.json")) as f:
            child = json.load(f)
        child_sid = child.get("session_id")
        self.assertTrue(child_sid)
        self.assertNotEqual(child_sid, parent_sid)
        flat_joined = [" ".join(argv) for argv in self._argvs()]
        self.assertTrue(
            any(("--resume %s --fork-session --session-id %s"
                 % (parent_sid, child_sid)) in j for j in flat_joined),
            "grok resume must fork into a fresh child session: %r" % flat_joined)

    def test_resume_workdir_busy(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-wd-parent"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        other = self._start(env={"FAKE_DELAY_S": "15",
                                 "FAKE_THREAD_ID": "t-wd-other"})
        try:
            before = set(os.listdir(os.path.join(_TMP, "jobs")))
            out = jobs.resume(started["job_dir"], "blocked by peer")
            self.assertEqual(out.get("refused", {}).get("reason"),
                             "resume_workdir_busy")
            self.assertEqual(before,
                             set(os.listdir(os.path.join(_TMP, "jobs"))))
        finally:
            _hard_stop(other["job_dir"], grace_s=1, reason="cleanup")

    def test_resume_unsupported_null_session_id(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-null-me"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        path = os.path.join(started["job_dir"], "job.json")
        with open(path) as f:
            data = json.load(f)
        data["session_id"] = None
        with open(path, "w") as f:
            json.dump(data, f)
            f.write("\n")
        before = set(os.listdir(os.path.join(_TMP, "jobs")))
        out = jobs.resume(started["job_dir"], "no session")
        self.assertEqual(out.get("refused", {}).get("reason"),
                         "resume_unsupported")
        self.assertEqual(before,
                         set(os.listdir(os.path.join(_TMP, "jobs"))))

    def test_codex_crash_mid_run_still_captures_session_id(self):
        started = self._start(env={"FAKE_THREAD_ID": "t-crash-99",
                                   "FAKE_DELAY_S": "20"})
        job_dir = started["job_dir"]
        try:
            deadline = time.time() + 10
            sid = None
            while time.time() < deadline:
                try:
                    with open(os.path.join(job_dir, "job.json")) as f:
                        sid = json.load(f).get("session_id")
                except (OSError, ValueError):
                    sid = None
                if sid:
                    break
                time.sleep(0.1)
            self.assertEqual(sid, "t-crash-99",
                             "session_id must be captured before the run ends")
            _hard_stop(job_dir, grace_s=1, reason="crash-mid-run")
            with open(os.path.join(job_dir, "job.json")) as f:
                self.assertEqual(json.load(f).get("session_id"), "t-crash-99")
        finally:
            _hard_stop(job_dir, grace_s=0, reason="cleanup")

    def test_resume_refuses_head_only_drift_and_allow_drift_overrides(self):
        # Same branch, new commit after source job → resume_drift; allow_drift ok.
        started = self._start(env={"FAKE_THREAD_ID": "t-head-only"})
        self.assertEqual(_wait_terminal(started["job_dir"]), "done")
        with open(os.path.join(self.workdir, "more"), "w") as fh:
            fh.write("y")
        _git("add", "more", cwd=self.workdir)
        _git("-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "head moves", cwd=self.workdir)
        try:
            out = jobs.resume(started["job_dir"], "after new head")
            self.assertEqual(out.get("refused", {}).get("reason"),
                             "resume_drift")
            out2 = jobs.resume(started["job_dir"], "after new head",
                               allow_drift=True)
            self.assertNotIn("refused", out2)
            self.assertIn("job_id", out2)
            _wait_terminal(out2["job_dir"])
        finally:
            # Leave workdir on branch work with whatever HEAD is now; other
            # tests re-read live pin from job.json, not a fixed sha.
            pass

    def test_resume_refusal_order_not_terminal_before_drift(self):
        # Non-terminal AND drifted → resume_not_terminal (not resume_drift).
        started = self._start(env={"FAKE_DELAY_S": "15",
                                   "FAKE_THREAD_ID": "t-order"})
        job_dir = started["job_dir"]
        try:
            _git("checkout", "-q", "-b", "order-drift", cwd=self.workdir)
            before = set(os.listdir(os.path.join(_TMP, "jobs")))
            out = jobs.resume(job_dir, "should hit not_terminal first")
            self.assertIn("refused", out)
            self.assertEqual(out["refused"]["reason"], "resume_not_terminal")
            self.assertNotEqual(out["refused"]["reason"], "resume_drift")
            self.assertEqual(before,
                             set(os.listdir(os.path.join(_TMP, "jobs"))))
        finally:
            try:
                _git("checkout", "-q", "work", cwd=self.workdir)
            except subprocess.CalledProcessError:
                pass
            _hard_stop(job_dir, grace_s=1, reason="cleanup")

    def test_cli_stop_smoke_with_and_without_reason(self):
        # THROUGH omni_cli.main — not jobs.stop — on a live fake job.
        import contextlib
        import io
        import omni_cli

        started = self._start(env={"FAKE_DELAY_S": "20",
                                   "FAKE_THREAD_ID": "t-cli-stop"})
        job_id = started["job_id"]
        try:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                code = omni_cli.main(
                    ["agent", "stop", job_id, "--grace", "1", "--reason", "x"])
            self.assertEqual(code, 0, "stop --reason must exit 0; stderr=%r"
                             % stderr.getvalue())
            self.assertNotIn("Traceback", stdout.getvalue() + stderr.getvalue())
            self.assertNotIn("unrecognized arguments",
                             stdout.getvalue() + stderr.getvalue())

            # Bare stop without --reason also works (may already be terminal).
            stdout2, stderr2 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout2), \
                    contextlib.redirect_stderr(stderr2):
                code2 = omni_cli.main(
                    ["agent", "stop", job_id, "--grace", "1"])
            self.assertEqual(code2, 0, "bare stop must exit 0; stderr=%r"
                             % stderr2.getvalue())
            self.assertNotIn("Traceback",
                             stdout2.getvalue() + stderr2.getvalue())
            self.assertNotIn("AttributeError",
                             stdout2.getvalue() + stderr2.getvalue())
        finally:
            _hard_stop(started["job_dir"], grace_s=0, reason="cleanup")


class ForemanGuardPins(unittest.TestCase):
    """Foreman-added pins for defensive guards the WO rounds left uncovered."""

    def test_capture_never_overwrites_existing_session_id(self):
        import providers.job_runner as job_runner
        job_dir = tempfile.mkdtemp(prefix="pin-d-")
        with open(os.path.join(job_dir, "job.json"), "w") as f:
            json.dump({"job_id": "j", "session_id": "first"}, f)
        with open(os.path.join(job_dir, "stream.log"), "w") as f:
            f.write('{"type": "thread.started", "thread_id": "second"}\n')
        job_runner._try_capture_session_id(job_dir)
        with open(os.path.join(job_dir, "job.json")) as f:
            self.assertEqual(json.load(f)["session_id"], "first")

    def test_session_capture_reads_stream_raw_not_normalized(self):
        """P3: thread.started is captured from stream.raw.log (not stream.log)."""
        import providers.job_runner as job_runner
        job_dir = tempfile.mkdtemp(prefix="pin-raw-")
        with open(os.path.join(job_dir, "job.json"), "w") as f:
            json.dump({"job_id": "j", "session_id": None}, f)
        # Normalized log has only a passthrough — must NOT be the source.
        with open(os.path.join(job_dir, "stream.log"), "w") as f:
            f.write('{"type":"x_omni_passthrough","backend":"codex",'
                    '"event":{"type":"thread.started","thread_id":"wrong"}}\n')
        with open(os.path.join(job_dir, "stream.raw.log"), "w") as f:
            f.write('{"type":"thread.started","thread_id":"from-raw"}\n')
        self.assertTrue(job_runner._try_capture_session_id(job_dir))
        with open(os.path.join(job_dir, "job.json")) as f:
            self.assertEqual(json.load(f)["session_id"], "from-raw")

    def test_grok_resume_without_child_sid_refuses_not_sameid(self):
        import asyncio
        from providers.agent_lane import run_agent
        bindir = tempfile.mkdtemp(prefix="pin-e1-bin-")
        fake = os.path.join(bindir, "grok")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\necho should-not-run\n")
        os.chmod(fake, 0o755)
        wd = tempfile.mkdtemp(prefix="pin-e1-wd-")
        subprocess.run(["git", "init", "-q", "-b", "work", wd], check=True)
        old_path = os.environ["PATH"]
        os.environ["PATH"] = bindir + os.pathsep + old_path
        try:
            res = asyncio.run(run_agent(
                "go", "grok-4.5", wd, resume_session="parent-sid"))
        finally:
            os.environ["PATH"] = old_path
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "resume_unsupported")
        self.assertNotIn("should-not-run", str(res))

    def test_resume_on_non_git_workdir_refuses_all_null_pin(self):
        from providers import jobs as jobs_mod
        self.assertTrue(jobs_mod._pin_differs(
            {"repo_root": None, "branch": None, "head": None},
            {"repo_root": None, "branch": None, "head": None}),
            "all-null vs all-null must count as drift (no identity to verify)")


if __name__ == "__main__":
    unittest.main()
