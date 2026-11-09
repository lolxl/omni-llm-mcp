# no-opencc
"""Acceptance tests — detached job model (issue #18 P1, plan R1 F1-F4, F6).

Pinned contract:
  * jobs.start() runs the agent-lane fences IN the caller's process and returns
    {job_id, job_dir} fast; the work rides a detached supervisor that survives
    the caller and writes stream.log + result.json (typed envelope, unchanged);
  * a fence refusal never creates a running job: state=failed, the refusal
    envelope IS result.json, and no supervisor is spawned;
  * jobs.stop(): SIGTERM the recorded groups, grace, SIGKILL; state=stopped and
    result.json carries reason "stopped" — stopped is DISTINCT from failed;
  * stop on a finished job does not overwrite its terminal state or result;
  * a crashed supervisor (dead liveness, state still "running") is repaired to
    failed with a synthesized result.json on status();
  * identity safety: status/stop never signal a pid whose start-identity token
    does not match job.json.

The backend CLI is faked with a real executable on PATH (the supervisor is a
separate process — in-process mocks cannot reach it).
"""
import fcntl
import json
import io
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="omni-jobs-")
os.environ["OMNI_LLM_STATE_DIR"] = _TMP

from providers import jobs  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_CODEX = r'''#!/usr/bin/env python3
import os, signal, sys, time
if os.environ.get("FAKE_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
data = sys.stdin.read() if not sys.stdin.isatty() else ""
delay = float(os.environ.get("FAKE_DELAY_S", "0"))
deadline = time.time() + delay
while time.time() < deadline:
    time.sleep(0.05)
sys.stdout.write("fake-codex-output:%d" % len(data))
'''


def _make_workdir(root):
    """Non-main git repo dir: passes the agent-lane branch fence."""
    wd = os.path.join(root, "wd")
    os.makedirs(wd)
    subprocess.run(["git", "init", "-q", "-b", "work", wd], check=True)
    return wd


def _wait_state(job_dir, want, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.status(job_dir)["state"]
        if st == want:
            return st
        time.sleep(0.1)
    return jobs.status(job_dir)["state"]


class JobModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Re-pin state dir when co-imported with other job suites.
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        cls.root = tempfile.mkdtemp(prefix="omni-jobs-fix-")
        cls.bindir = os.path.join(cls.root, "bin")
        os.makedirs(cls.bindir)
        fake = os.path.join(cls.bindir, "codex")
        with open(fake, "w") as f:
            f.write(FAKE_CODEX)
        os.chmod(fake, 0o755)
        cls.workdir = _make_workdir(cls.root)
        os.environ["PATH"] = cls.bindir + os.pathsep + os.environ["PATH"]
        # The fenced fake CLI gets an allowlisted environment (#138): its
        # control variables are passed by name, as an operator would.
        os.environ["OMNI_LLM_CHILD_ENV_PASS"] = "FAKE_DELAY_S,FAKE_IGNORE_TERM"
        cls.addClassCleanup(os.environ.pop, "OMNI_LLM_CHILD_ENV_PASS", None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP

    def _start(self, prompt="hello", env=None, timeout_s=60):
        old = {}
        for k, v in (env or {}).items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
        try:
            return jobs.start(task=prompt, model="codex/fake",
                              workdir=self.workdir, timeout_s=timeout_s)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_start_returns_fast_and_job_completes(self):
        t0 = time.monotonic()
        started = self._start(env={"FAKE_DELAY_S": "3"})
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 2.0, "start() blocked on the job itself")
        self.assertIn("job_id", started)
        job_dir = started["job_dir"]
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "job.json")))
        self.assertEqual(jobs.status(job_dir)["state"], "running")

        self.assertEqual(_wait_state(job_dir, "done"), "done")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertTrue(res["ok"], res)
        self.assertIn("fake-codex-output", str(res))
        log = os.path.join(job_dir, "stream.log")
        self.assertTrue(os.path.isfile(log))
        self.assertIn("fake-codex-output", open(log, errors="replace").read())

    def test_fence_refusal_creates_no_running_job(self):
        started = self._start(env={"OMNI_LLM_DEPTH": "1"})
        job_dir = started["job_dir"]
        st = jobs.status(job_dir)
        self.assertEqual(st["state"], "failed")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "recursion_guard")
        self.assertIsNone(st.get("pid"), "refused job must not have a process")

    def test_cursor_job_preassigns_null_session_id_and_seatbelt_fence(self):
        # An unsupported_effort refusal happens AFTER route resolution but
        # BEFORE any spawn (no supervisor process, no real cursor-agent
        # call), so job.json's pre-spawn fields are observable directly:
        # cursor must NOT pre-assign a session_id (codex-style capture from
        # the event stream instead — see job_runner._try_capture_session_id)
        # and must plan the seatbelt-fs kernel fence class like claude/pi.
        started = jobs.start(task="hello", model="cursor/composer-2.5",
                             workdir=self.workdir, timeout_s=30,
                             effort="not-a-real-effort-token")
        job_dir = started["job_dir"]
        self.assertEqual(jobs.status(job_dir)["state"], "failed")
        with open(os.path.join(job_dir, "job.json")) as f:
            job = json.load(f)
        self.assertEqual(job.get("provider"), "cursor")
        self.assertIsNone(job.get("session_id"))
        self.assertEqual(job.get("fence"), "seatbelt-fs")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "unsupported_effort")

    @mock.patch("platform.system", new=lambda: "Darwin")
    def test_codex_job_plans_seatbelt_on_both_mac_lanes(self):
        # Same trick as the cursor case above: an unsupported_effort refusal
        # lands AFTER route resolution and BEFORE any spawn, so job.json's
        # planned fence is observable directly. It has to track run_agent's
        # mandatory host fence on both ordinary macOS lanes.
        for read_only, want in ((False, "seatbelt-fs"), (True, "seatbelt-fs")):
            with self.subTest(read_only=read_only):
                started = jobs.start(task="hello", model="codex/fake",
                                     workdir=self.workdir, timeout_s=30,
                                     read_only=read_only,
                                     effort="not-a-real-effort-token")
                job_dir = started["job_dir"]
                self.assertEqual(jobs.status(job_dir)["state"], "failed")
                with open(os.path.join(job_dir, "job.json")) as f:
                    job = json.load(f)
                self.assertEqual(job.get("provider"), "codex")
                self.assertEqual(job.get("read_only"), read_only)
                self.assertEqual(job.get("fence"), want)

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_codex_write_job_no_fence_plans_vendor(self):
        # --no-fence on the write lane falls back to the vendor sandbox, so
        # the planned fence is `vendor`, not `none`: that spawn really would
        # rest on the vendor sandbox and the argv still carries its flag.
        started = jobs.start(task="hello", model="codex/fake",
                             workdir=self.workdir, timeout_s=30,
                             fence_requested=False,
                             effort="not-a-real-effort-token")
        with open(os.path.join(started["job_dir"], "job.json")) as f:
            job = json.load(f)
        self.assertFalse(job.get("fence_requested"))
        self.assertEqual(job.get("fence"), "vendor")

    def test_stop_yields_stopped_not_failed(self):
        started = self._start(env={"FAKE_DELAY_S": "30"})
        job_dir = started["job_dir"]
        self.assertEqual(jobs.status(job_dir)["state"], "running")
        out = jobs.stop(job_dir, grace_s=2, reason="test says stop")
        self.assertEqual(out["state"], "stopped")
        st = jobs.status(job_dir)
        self.assertEqual(st["state"], "stopped")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "stopped", "stopped must be its own "
                         "reason, distinct from every failure class")
        self.assertEqual(st.get("stop_reason"), "test says stop")

    def test_stop_kills_term_ignoring_child(self):
        started = self._start(env={"FAKE_DELAY_S": "30",
                                   "FAKE_IGNORE_TERM": "1"})
        job_dir = started["job_dir"]
        t0 = time.monotonic()
        out = jobs.stop(job_dir, grace_s=1, reason="kill path")
        self.assertEqual(out["state"], "stopped")
        self.assertLess(time.monotonic() - t0, 15)

    def test_stop_on_finished_job_preserves_terminal_state(self):
        started = self._start(env={"FAKE_DELAY_S": "0"})
        job_dir = started["job_dir"]
        self.assertEqual(_wait_state(job_dir, "done"), "done")
        before = open(os.path.join(job_dir, "result.json")).read()
        out = jobs.stop(job_dir, grace_s=1, reason="too late")
        self.assertEqual(out["state"], "done")
        self.assertEqual(open(os.path.join(job_dir, "result.json")).read(),
                         before, "stop must not rewrite a finished result")

    def test_backend_pid_is_recorded_and_reaped_end_to_end(self):
        """Pins the RECORDING step: on_spawn -> _record_backend -> job.json.

        The synthetic reap pin plants backend_pid by hand; this one drives the
        real runner, so a runner that stops recording the backend fails here.
        """
        started = self._start(env={"FAKE_DELAY_S": "30",
                                   "FAKE_IGNORE_TERM": "1"})
        job_dir = started["job_dir"]
        job_path = os.path.join(job_dir, "job.json")
        backend_pid = None
        deadline = time.time() + 10
        while time.time() < deadline:
            with open(job_path) as f:
                data = json.load(f)
            backend_pid = data.get("backend_pid")
            if backend_pid:
                break
            time.sleep(0.1)
        try:
            self.assertIsNotNone(
                backend_pid, "runner must record the backend pid in job.json")
            os.killpg(int(data["supervisor_pgid"]), signal.SIGKILL)
            deadline = time.time() + 15
            repaired = None
            while time.time() < deadline:
                repaired = jobs.status(job_dir)["state"]
                if repaired == "failed":
                    break
                time.sleep(0.2)
            self.assertEqual(repaired, "failed")
            deadline = time.time() + 10
            alive = True
            while time.time() < deadline:
                try:
                    os.kill(int(backend_pid), 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.2)
            self.assertFalse(alive, "repair must reap the recorded backend")
        finally:
            for key in ("supervisor_pgid", "backend_pgid"):
                try:
                    os.killpg(int(data.get(key)), signal.SIGKILL)
                except (OSError, TypeError, ValueError):
                    pass

    def test_crashed_supervisor_is_repaired_on_status(self):
        started = self._start(env={"FAKE_DELAY_S": "30"})
        job_dir = started["job_dir"]
        job_path = os.path.join(job_dir, "job.json")
        jobs.stop(job_dir, grace_s=1, reason="setup teardown")
        with open(job_path) as f:
            data = json.load(f)
        data["state"] = "running"  # simulate: supervisor died mid-flight
        data["pid"] = 1  # launchd: alive, but identity token cannot match
        with open(job_path, "w") as f:
            json.dump(data, f)
        os.unlink(os.path.join(job_dir, "result.json"))
        st = jobs.status(job_dir)
        self.assertEqual(st["state"], "failed",
                         "dead/foreign supervisor must repair to failed")
        with open(os.path.join(job_dir, "result.json")) as f:
            res = json.load(f)
        self.assertFalse(res["ok"])


class JobFixRoundTest(unittest.TestCase):
    """Regression pins for the WO-3b lifecycle mechanisms."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="omni-jobs-wo3b-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _job(self, state="running", created_at=None):
        job_dir = tempfile.mkdtemp(prefix="job-", dir=self.root)
        for name in ("job.lock", "result.lock"):
            open(os.path.join(job_dir, name), "ab").close()
        data = {
            "job_id": os.path.basename(job_dir),
            "state": state,
            "created_at": jobs._now() if created_at is None else created_at,
            "pid": None,
            "pgid": None,
            "pid_identity": None,
            "supervisor_pid": None,
            "supervisor_pgid": None,
            "supervisor_identity": None,
            "supervisor_start_identity": None,
            "backend_pid": None,
            "backend_pgid": None,
            "backend_pid_identity": None,
            "backend_identity": None,
            "backend_start_identity": None,
            "stop_reason": None,
            "stopped_by": None,
        }
        jobs._write_job(job_dir, data)
        return job_dir, data

    def test_stop_rechecks_identity_before_sigkill(self):
        job_dir, data = self._job()
        data.update({"pid": 111, "pgid": 222, "pid_identity": "old"})
        data.update({"supervisor_pid": 111, "supervisor_pgid": 222,
                     "supervisor_identity": "old"})
        jobs._write_job(job_dir, data)
        identities = iter(("match", "match", "mismatch"))
        with mock.patch.object(jobs, "_lock_is_held", return_value=True), \
                mock.patch.object(jobs, "_identity_status_record",
                                  side_effect=lambda *_args: next(identities)), \
                mock.patch.object(jobs, "_send_group") as send_group:
            out = jobs.stop(job_dir, grace_s=0, reason="identity drift")
        signals = [call.args[2] for call in send_group.call_args_list]
        self.assertIn(signal.SIGTERM, signals)
        self.assertNotIn(signal.SIGKILL, signals,
                         "a changed token must block the KILL escalation")
        self.assertIn("changed before kill", out.get("detail", ""))

    def test_backend_term_and_kill_are_not_deduped(self):
        job_dir, data = self._job()
        data.update({
            "pid": 111, "pgid": 222, "pid_identity": "supervisor",
            "supervisor_pid": 111, "supervisor_pgid": 222,
            "supervisor_identity": "supervisor",
            "backend_pid": 333, "backend_pgid": 444,
            "backend_pid_identity": "backend", "backend_identity": "backend",
        })
        jobs._write_job(job_dir, data)
        with mock.patch.object(jobs, "_lock_is_held", return_value=True), \
                mock.patch.object(jobs, "_identity_status_record",
                                  return_value="match"), \
                mock.patch.object(jobs, "_wait_for_unlock", return_value=True), \
                mock.patch.object(jobs, "_send_group") as send_group:
            out = jobs.stop(job_dir, grace_s=0.02, reason="dedupe pin")
        sent = {(call.args[1], call.args[2])
                for call in send_group.call_args_list}
        self.assertIn(("backend", signal.SIGTERM), sent)
        self.assertIn(("backend", signal.SIGKILL), sent)
        self.assertEqual(out["state"], "stopped")

    def test_result_write_once_returns_first_payload_and_preserves_bytes(self):
        job_dir, _data = self._job()
        first = {"ok": True, "reason": "done", "value": 1}
        second = {"ok": False, "reason": "stopped"}
        self.assertEqual(jobs._write_result(job_dir, first), first)
        path = os.path.join(job_dir, "result.json")
        with open(path, "rb") as handle:
            before = handle.read()
        self.assertEqual(jobs._write_result(job_dir, second), first)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_free_lock_wins_over_live_unrelated_pid(self):
        job_dir, data = self._job()
        token = jobs._pid_identity(os.getpid())
        self.assertIsNotNone(token)
        data.update({"pid": os.getpid(), "pgid": os.getpgrp(),
                     "pid_identity": token, "supervisor_pid": os.getpid(),
                     "supervisor_pgid": os.getpgrp(),
                     "supervisor_identity": token})
        jobs._write_job(job_dir, data)
        with mock.patch.object(
                jobs, "_identity_matches",
                side_effect=AssertionError("status must use flock, not pid identity")), \
                mock.patch.object(
                    jobs, "_identity_status_record",
                    side_effect=AssertionError("status must not probe identity")):
            view = jobs.status(job_dir)
        self.assertEqual(view["state"], "failed")
        self.assertFalse(jobs._lock_is_held(job_dir))

    def test_status_repairs_and_reaps_term_ignoring_backend(self):
        job_dir, data = self._job()
        backend = subprocess.Popen(
            [sys.executable, "-c",
             "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
            start_new_session=True,
        )
        supervisor = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        try:
            backend_token = jobs._pid_identity(backend.pid)
            supervisor_token = jobs._pid_identity(supervisor.pid)
            self.assertIsNotNone(backend_token)
            self.assertIsNotNone(supervisor_token)
            data.update({
                "pid": supervisor.pid,
                "pgid": os.getpgid(supervisor.pid),
                "pid_identity": supervisor_token,
                "supervisor_pid": supervisor.pid,
                "supervisor_pgid": os.getpgid(supervisor.pid),
                "supervisor_identity": supervisor_token,
                "backend_pid": backend.pid,
                "backend_pgid": os.getpgid(backend.pid),
                "backend_pid_identity": backend_token,
                "backend_identity": backend_token,
            })
            jobs._write_job(job_dir, data)
            os.kill(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=5)
            view = jobs.status(job_dir)
            self.assertEqual(view["state"], "failed")
            deadline = time.time() + 5
            while backend.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(backend.poll(),
                                 "repair must reap the orphan backend group")
        finally:
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.wait()
            if backend.poll() is None:
                try:
                    os.killpg(backend.pid, signal.SIGKILL)
                except OSError:
                    backend.kill()
                backend.wait()

    def test_runner_writes_result_before_terminal_job_state(self):
        job_dir, _data = self._job()
        from providers import job_runner
        seen = []
        original = jobs._write_job

        def observe_write(path, value):
            seen.append(os.path.isfile(os.path.join(path, "result.json")))
            return original(path, value)

        with mock.patch.object(jobs, "_write_job", side_effect=observe_write):
            winner = job_runner._finalize(job_dir, {"ok": True, "reason": "done"})
        self.assertEqual(winner["reason"], "done")
        self.assertEqual(seen, [True])

    def test_startup_grace_preserves_starting_then_repairs_after_grace(self):
        job_dir, data = self._job(state="starting", created_at=jobs._now())
        view = jobs.status(job_dir)
        self.assertEqual(view["state"], "starting")
        self.assertFalse(os.path.exists(os.path.join(job_dir, "result.json")))
        data["created_at"] = jobs._now() - 60
        jobs._write_job(job_dir, data)
        view = jobs.status(job_dir)
        self.assertEqual(view["state"], "failed")
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "result.json")))

    def test_result_wins_when_stop_observes_before_state_publish(self):
        job_dir, data = self._job()
        jobs._write_result(job_dir, {"ok": True, "reason": "done"})
        data["state"] = "running"
        jobs._write_job(job_dir, data)
        out = jobs.stop(job_dir, grace_s=0, reason="racing stop")
        self.assertEqual(out["state"], "done")
        self.assertTrue(jobs._read_result(job_dir)["ok"])

    def test_refusal_start_is_returned_and_cli_exits_nonzero(self):
        job_dir, _data = self._job(state="failed")
        refusal = {"ok": False, "reason": "recursion_guard", "detail": "nested"}
        import omni_cli
        output = io.StringIO()
        with mock.patch.object(omni_cli.jobs_mod, "start",
                               return_value={"job_id": "j", "job_dir": job_dir,
                                             "refused": refusal}), \
                redirect_stdout(output):
            code = omni_cli._main_agent_job(
                ["start", "task", "--workdir", self.root])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["refused"], refusal)


class JobStreamPipelineExtras(unittest.TestCase):
    """WO-5b additions: raw/err split + provider pin on completed jobs."""

    @classmethod
    def setUpClass(cls):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP
        cls.root = tempfile.mkdtemp(prefix="omni-jobs-p3-")
        cls.bindir = os.path.join(cls.root, "bin")
        os.makedirs(cls.bindir)
        fake = os.path.join(cls.bindir, "codex")
        with open(fake, "w") as f:
            f.write(FAKE_CODEX)
        os.chmod(fake, 0o755)
        cls.workdir = _make_workdir(cls.root)
        cls._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = cls.bindir + os.pathsep + cls._old_path

    @classmethod
    def tearDownClass(cls):
        os.environ["PATH"] = cls._old_path
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        os.environ["OMNI_LLM_STATE_DIR"] = _TMP

    def test_completed_job_has_raw_err_and_provider(self):
        started = jobs.start(task="hello", model="codex/fake",
                             workdir=self.workdir, timeout_s=60)
        job_dir = started["job_dir"]
        self.assertEqual(_wait_state(job_dir, "done"), "done")
        for name in ("stream.log", "stream.err.log"):
            self.assertTrue(os.path.isfile(os.path.join(job_dir, name)), name)
        # #170: the raw log is the in-run channel only; not retained after the end.
        self.assertFalse(os.path.exists(os.path.join(job_dir, "stream.raw.log")))
        with open(os.path.join(job_dir, "job.json")) as f:
            data = json.load(f)
        self.assertEqual(data.get("provider"), "codex")
        # Default tail reads stream.log; --raw path is covered in integration.
        self.assertIn("fake-codex-output", jobs.tail(job_dir, lines=50))


class SweepOrphansTest(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp(prefix="omni-sweep-")
        self.addCleanup(shutil.rmtree, self.state, ignore_errors=True)
        self._old = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = self.state

    def tearDown(self):
        if self._old is None:
            os.environ.pop("OMNI_LLM_STATE_DIR", None)
        else:
            os.environ["OMNI_LLM_STATE_DIR"] = self._old

    def _job(self, job_id, state="running", created_at=None):
        job_dir = os.path.join(self.state, "jobs", job_id)
        os.makedirs(job_dir)
        for name in ("job.lock", "result.lock"):
            open(os.path.join(job_dir, name), "ab").close()
        data = {
            "job_id": job_id,
            "state": state,
            "created_at": jobs._now() if created_at is None else created_at,
            "pid": None,
            "pgid": None,
            "pid_identity": None,
            "supervisor_pid": None,
            "supervisor_pgid": None,
            "supervisor_identity": None,
            "supervisor_start_identity": None,
            "backend_pid": None,
            "backend_pgid": None,
            "backend_pid_identity": None,
            "backend_identity": None,
            "backend_start_identity": None,
            "stop_reason": None,
            "stopped_by": None,
        }
        jobs._write_job(job_dir, data)
        return job_dir, data

    def test_free_lock_running_job_is_repaired(self):
        job_dir, _ = self._job("orphan1", state="running")
        repaired = jobs.sweep_orphans()
        self.assertEqual(repaired, ["orphan1"])
        self.assertEqual(jobs.status(job_dir)["state"], "failed")
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "runtime_error")
        self.assertIn("supervisor lock is free", res["detail"])
        self.assertEqual(jobs.sweep_orphans(), [])

    def test_held_lock_is_untouched(self):
        job_dir, _ = self._job("held1", state="running")
        lock = open(os.path.join(job_dir, "job.lock"), "a+b")
        self.addCleanup(lock.close)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            self.assertEqual(jobs.sweep_orphans(), [])
            self.assertEqual(jobs.status(job_dir)["state"], "running")
            self.assertFalse(os.path.isfile(
                os.path.join(job_dir, "result.json")))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def test_startup_grace_is_untouched(self):
        self._job("starting1", state="starting", created_at=jobs._now())
        self.assertEqual(jobs.sweep_orphans(), [])

    def test_list_jobs_sweeps_first(self):
        self._job("orphan2", state="running")
        payload = jobs.list_jobs()
        listed = payload["jobs"]
        self.assertEqual(payload["repaired"], ["orphan2"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["state"], "failed")
        self.assertEqual(listed[0]["job_id"], "orphan2")

    def test_existing_result_repairs_to_recorded_state(self):
        job_dir, _ = self._job("orphan-done", state="running")
        jobs._write_result(job_dir, {"ok": True, "text": "hi"})
        repaired = jobs.sweep_orphans()
        self.assertEqual(repaired, ["orphan-done"])
        self.assertEqual(jobs.status(job_dir)["state"], "done")

    def test_corrupt_job_dir_is_skipped_and_logged(self):
        job_dir, _ = self._job("bad1", state="running")
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with mock.patch.object(jobs, "log_event") as logged:
            repaired = jobs.sweep_orphans()
        self.assertEqual(repaired, [])
        events = [c.args[0] for c in logged.call_args_list]
        self.assertIn("job_sweep_skipped", events)
        kwargs = logged.call_args_list[events.index("job_sweep_skipped")].kwargs
        self.assertEqual(kwargs.get("job_dir"), job_dir)


class CursorJobLaneFenceWiringTest(unittest.TestCase):
    def test_job_lane_passes_jobs_root_as_launcher_roots(self):
        import asyncio
        from providers import agent_lane
        from providers import kernel_sandbox
        from tests.test_agent import FakeRunCli, depth, patched_lane

        fake = FakeRunCli(stdout='{"type":"result","subtype":"success",'
                          '"is_error":false,"result":"HI"}')
        wd = tempfile.mkdtemp(prefix="omni_jobs_cursor_wd_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        job_dir = tempfile.mkdtemp(prefix="omni_jobs_cursor_job_")
        self.addCleanup(shutil.rmtree, job_dir, ignore_errors=True)
        log_path = os.path.join(job_dir, "stream.log")
        policy = {
            "wrap_kw": dict(
                read_wall=True, extra_read_paths=(), extra_write_paths=(),
            ),
            "tmpdir": None,
            "vendor_state_writable": True,
            "binary": "/fake/bin/cursor-agent",
        }
        seen = {}
        real_apply = agent_lane._apply_kernel_fence

        def capture_apply(*args, **kwargs):
            seen["kwargs"] = dict(kwargs)
            return real_apply(*args, **kwargs)

        cursor_env_keys = (
            "OMNI_LLM_CURSOR_CONFIG_DIR", "CURSOR_CONFIG_DIR",
        )
        saved_cursor = {k: os.environ[k] for k in cursor_env_keys
                        if k in os.environ}
        with mock.patch.dict(os.environ, saved_cursor, clear=False), \
                depth(None), patched_lane(fake), \
                mock.patch.object(agent_lane, "_workdir_fence",
                                  return_value=None), \
                mock.patch.object(agent_lane, "resolve_agent_binary",
                                  return_value="/fake/bin/cursor-agent"), \
                mock.patch.object(kernel_sandbox, "available",
                                  return_value=True), \
                mock.patch.object(kernel_sandbox, "agent_fence_policy",
                                  return_value=policy), \
                mock.patch.object(kernel_sandbox, "verify_fence",
                                  return_value=None), \
                mock.patch.object(
                    kernel_sandbox, "wrap_argv",
                    lambda argv, *a, **k: (
                        ["/usr/bin/sandbox-exec", "-p", "P"] + list(argv)
                    ),
                ), \
                mock.patch.object(kernel_sandbox, "is_wrapped",
                                  lambda argv: True), \
                mock.patch.object(agent_lane, "_apply_kernel_fence",
                                  side_effect=capture_apply):
            for k in cursor_env_keys:
                os.environ.pop(k, None)
            res = asyncio.run(agent_lane.run_agent(
                task="t", model="cursor/composer-2.5", workdir=wd,
                timeout_s=30, job_lane=True, log_path=log_path))
        self.assertTrue(res.get("ok") or res.get("reason") in (
            "empty_response", "runtime_error",
        ), res)
        self.assertIn("launcher_roots", seen.get("kwargs") or {})
        self.assertEqual(
            seen["kwargs"]["launcher_roots"],
            (os.path.realpath(jobs._jobs_root()),),
        )


if __name__ == "__main__":
    unittest.main()

