#!/usr/bin/env python3
# no-opencc
"""Egress residual seam (2/4): agent lane + job lane.

When OMNI_EGRESS_GATE=1, agent lane scans the caller-authored task string
before run_cli (stdin / grok --prompt-file). Job lane scans the same string
before prompt.txt write and supervisor spawn. Gate off leaves the bare-run
gap (still delivers). No network, no real CLIs, no live git: fences and
spawn are intercepted. Synthetic fixture tokens only.
"""
import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from providers import agent_lane
from providers import jobs
from tests import offline_home

_HOME = None


def setUpModule():
    # Direct runs admit real home paths; keep protected_home() on a synthetic home.
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


TW_MOBILE = "0912345678"


def _env(gate="1"):
    env = os.environ.copy()
    env.pop("OMNI_EGRESS_GATE", None)
    env.pop("OMNI_EGRESS_DENYLIST", None)
    env["OMNI_LLM_METRICS"] = "0"
    env.pop("OMNI_LLM_DEPTH", None)
    if gate is not None:
        env["OMNI_EGRESS_GATE"] = gate
    return mock.patch.dict(os.environ, env, clear=True)


def _detail(envelope):
    return str((envelope or {}).get("detail") or "")


def _fake_which(wanted):
    real = shutil.which

    def which(cmd, *args, **kwargs):
        if cmd == wanted:
            return "/usr/bin/%s" % cmd
        return real(cmd, *args, **kwargs)

    return which


class _FakeProc:
    """Supervisor stand-in: already exited, so start() does not wait on a lock."""

    def __init__(self):
        self.pid = os.getpid()

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def _prompt_txt_paths(root):
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "prompt.txt" in filenames:
            found.append(os.path.join(dirpath, "prompt.txt"))
    return found


class TestAgentLaneRefuse(unittest.TestCase):
    """Entry: providers/agent_lane.py run_agent → refuse_payload(task)
    before run_cli. Covers grok --prompt-file and claude/codex/pi stdin.
    """

    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def _run_agent(self, model, which_bin, task):
        sent = []
        wd = tempfile.mkdtemp(prefix="omni_seam_agent_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)

        def boom(*_a, **_k):
            sent.append("agent")
            raise RuntimeError("stubbed send reached")

        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            with mock.patch.object(agent_lane, "run_cli", boom):
                with mock.patch("providers.agent_lane.shutil.which",
                                _fake_which(which_bin)):
                    res = asyncio.run(agent_lane.run_agent(
                        task, model, wd, timeout_s=5))
        return res, sent

    def test_run_agent_grok_prompt_file_refuses_with_gate_on(self):
        # Gate on: grok --prompt-file path must refuse before run_cli.
        task = "call me at %s please" % TW_MOBILE
        with _env(gate="1"):
            res, sent = self._run_agent("grok-4.6", "grok", task)
        self.assertFalse((res or {}).get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "agent grok --prompt-file must not reach send")

    def test_run_agent_stdin_refuses_with_gate_on(self):
        # Gate on: claude stdin_data=task path must refuse before run_cli.
        task = "call me at %s please" % TW_MOBILE
        with _env(gate="1"):
            res, sent = self._run_agent("claude/sonnet", "claude", task)
        self.assertFalse((res or {}).get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "agent stdin path must not reach send")

    def test_run_agent_reaches_send_with_gate_off(self):
        # Gate off: bare-run gap — task still reaches run_cli.
        task = "call me at %s please" % TW_MOBILE
        with _env(gate=None):
            res, sent = self._run_agent("grok-4.6", "grok", task)
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(sent, ["agent"], "agent lane must still reach send")
        self.assertFalse(
            (res or {}).get("ok") and "OMNI-EGRESS-DENY" in _detail(res)
        )


class TestJobLaneRefuse(unittest.TestCase):
    """Entry: providers/jobs.py start → refuse_payload(task) before
    _new_job (prompt.txt) and subprocess.Popen (supervisor).
    """

    def _start_job(self, task, popen_side):
        wd = tempfile.mkdtemp(prefix="omni_seam_job_wd_")
        state = tempfile.mkdtemp(prefix="omni_seam_job_state_")
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)

        def fake_preflight(model, workdir, read_only, allow_main, effort=None,
                           account_profile="default"):
            return (
                None,
                os.path.realpath(str(workdir)),
                {"provider": "codex", "model": "gpt-5.6-sol"},
            )

        real_popen = jobs.subprocess.Popen

        def fake_popen(argv, *a, **k):
            # Supervisor spawn is the egress attempt. git pin / ps identity
            # also go through Popen via subprocess.run — those stay real.
            if argv and "providers.job_runner" in str(argv):
                popen_side.append("job")
                return _FakeProc()
            return real_popen(argv, *a, **k)

        os.environ["OMNI_LLM_STATE_DIR"] = state
        with mock.patch.object(jobs, "_preflight", fake_preflight):
            with mock.patch.object(jobs, "_drift_pin",
                                   return_value={
                                       "repo_root": None,
                                       "branch": None,
                                       "head": None,
                                   }):
                with mock.patch.object(jobs.subprocess, "Popen", fake_popen):
                    started = jobs.start(
                        task=task, model="codex/gpt-5.6-sol",
                        workdir=wd, timeout_s=5)
        return started, state

    def test_jobs_start_refuses_before_spawn_with_gate_on(self):
        # Gate on: hit task is refused before prompt.txt and supervisor spawn.
        sent = []
        task = "call me at %s please" % TW_MOBILE
        with _env(gate="1"):
            started, state = self._start_job(task, sent)
        refused = (started or {}).get("refused")
        self.assertIsNotNone(refused, started)
        self.assertFalse(refused.get("ok"), refused)
        self.assertTrue(
            _detail(refused).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(refused),
        )
        self.assertNotIn(TW_MOBILE, _detail(refused))
        self.assertNotIn(TW_MOBILE, str(started))
        self.assertEqual(sent, [], "job lane must not spawn supervisor")
        self.assertNotIn("job_dir", started, started)
        self.assertEqual(_prompt_txt_paths(state), [],
                         "no prompt.txt residual on egress refuse")

    def test_jobs_start_reaches_spawn_with_gate_off(self):
        # Gate off: bare-run gap — task is written and supervisor spawns.
        sent = []
        task = "call me at %s please" % TW_MOBILE
        with _env(gate=None):
            started, _state = self._start_job(task, sent)
        self.assertNotIn("refused", started, started)
        self.assertEqual(sent, ["job"], "job lane must still spawn supervisor")
        prompt_path = os.path.join(started["job_dir"], "prompt.txt")
        with open(prompt_path, encoding="utf-8") as handle:
            body = handle.read()
        self.assertIn(TW_MOBILE, body)
        self.assertNotIn("OMNI-EGRESS-DENY", str(started))


if __name__ == "__main__":
    unittest.main()
