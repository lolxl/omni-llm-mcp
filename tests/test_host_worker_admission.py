#!/usr/bin/env python3
"""Admission guards for ordinary macOS sync and detached-job workers."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from providers import agent_lane, jobs, kernel_sandbox


PROVIDER_MODELS = (
    ("codex", "codex/fixture"),
    ("claude", "claude/fixture"),
    ("grok", "grok/fixture"),
    ("pi", "pi/fixture"),
    ("cursor", "cursor/fixture"),
)


def _run(coro):
    return asyncio.run(coro)


class HostWorkerAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="host-admission-")
        self.addCleanup(self.tmp.cleanup)
        self.workdir = os.path.realpath(self.tmp.name)

    def test_shared_host_operation_admits_every_ordinary_provider(self):
        metadata = {
            "fence": "seatbelt-fs",
            "fence_detail": {
                "read_wall": True,
                "workdir_writable": True,
                "vendor_state_writable": False,
            },
        }

        def prepare(argv, **kwargs):
            return ["/usr/bin/sandbox-exec", "-p", "fixture", *argv], {
                **kwargs["env"],
                "HOST_FENCED": "1",
            }, metadata

        with mock.patch.object(agent_lane.platform, "system", return_value="Darwin"), \
             mock.patch.object(kernel_sandbox, "available", return_value=True), \
             mock.patch.object(agent_lane, "prepare_host_fence",
                               side_effect=prepare) as common:
            for provider, _model in PROVIDER_MODELS:
                with self.subTest(provider=provider):
                    argv, env, fence, detail, refusal = (
                        agent_lane._apply_kernel_fence(
                            provider,
                            ["/fixture/%s" % provider, "task"],
                            {"PATH": "/fixture"},
                            self.workdir,
                            provider == "grok",
                            True,
                        )
                    )
                    self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
                    self.assertEqual(env["HOST_FENCED"], "1")
                    self.assertEqual(fence, "seatbelt-fs")
                    self.assertTrue(detail["read_wall"])
                    self.assertIsNone(refusal)

        self.assertEqual(
            [call.kwargs["provider"] for call in common.call_args_list],
            [provider for provider, _model in PROVIDER_MODELS],
        )

    def test_mac_host_failure_precedes_every_vendor_spawn_and_old_canary(self):
        host_refusal = {
            "ok": False,
            "reason": "fixture_host_refusal",
            "detail": "synthetic host admission failure",
            "fence": "none",
            "fence_detail": dict(kernel_sandbox.FENCE_NONE_DETAIL),
        }
        admitted = []

        def refuse(provider, argv, env, workdir, read_only, requested, **kwargs):
            admitted.append((provider, read_only, requested))
            return argv, env, "none", dict(kernel_sandbox.FENCE_NONE_DETAIL), host_refusal

        vendor_spawn = mock.AsyncMock()
        with mock.patch.object(agent_lane.platform, "system", return_value="Darwin"), \
             mock.patch.object(agent_lane, "env_depth", return_value=0), \
             mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
             mock.patch.object(agent_lane, "resolve_profile", return_value=({}, None)), \
             mock.patch.object(agent_lane, "effort_for_provider",
                               return_value=(None, None)), \
             mock.patch.object(agent_lane, "resolve_agent_binary",
                               side_effect=lambda provider: "/fixture/" + provider), \
             mock.patch.object(agent_lane.shutil, "which",
                               side_effect=lambda name: "/fixture/" + name), \
             mock.patch.object(agent_lane.cursor_mod, "resolve_model",
                               return_value=("fixture", None)), \
             mock.patch.object(agent_lane, "pi_child_env",
                               return_value={"PATH": "/fixture"}), \
             mock.patch.object(agent_lane.sandbox_canary, "gate") as old_canary, \
             mock.patch.object(agent_lane, "_apply_kernel_fence",
                               side_effect=refuse), \
             mock.patch.object(agent_lane, "run_cli", vendor_spawn):
            for provider, model in PROVIDER_MODELS:
                with self.subTest(provider=provider):
                    result = _run(agent_lane.run_agent(
                        task="fixture task",
                        model=model,
                        workdir=self.workdir,
                        read_only=(provider == "grok"),
                        timeout_s=30,
                        job_lane=True,
                    ))
                    self.assertEqual(result["reason"], "fixture_host_refusal")

        self.assertEqual(
            admitted,
            [(provider, provider == "grok", True)
             for provider, _model in PROVIDER_MODELS],
        )
        old_canary.assert_not_called()
        vendor_spawn.assert_not_awaited()

    def test_darwin_job_preflight_admits_grok_read_only_without_old_canary(self):
        with mock.patch.object(jobs.platform, "system", return_value="Darwin"), \
             mock.patch.object(agent_lane, "env_depth", return_value=0), \
             mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
             mock.patch.object(jobs, "effort_for_provider",
                               return_value=(None, None)), \
             mock.patch.object(jobs, "resolve_profile", return_value=({}, None)), \
             mock.patch.object(jobs.shutil, "which", return_value="/fixture/grok"), \
             mock.patch.object(jobs.sandbox_canary, "gate") as old_canary:
            refusal, normalized, route = jobs._preflight(
                "grok/fixture", self.workdir, True, False
            )

        self.assertIsNone(refusal)
        self.assertEqual(normalized, self.workdir)
        self.assertEqual(route["provider"], "grok")
        old_canary.assert_not_called()

    def test_job_plan_matches_forced_mac_host_admission_for_all_providers(self):
        jobs_root = os.path.join(self.workdir, "jobs")
        refusal = {
            "ok": False,
            "reason": "fixture_after_route",
            "detail": "stop after route so no supervisor is needed",
        }

        with mock.patch.object(jobs.platform, "system", return_value="Darwin"), \
             mock.patch.object(jobs, "_jobs_root", return_value=jobs_root), \
             mock.patch.object(jobs, "_drift_pin", return_value={
                 "repo_root": None, "branch": None, "head": None,
             }), \
             mock.patch.object(kernel_sandbox, "agent_fence_switch_on",
                               return_value=False), \
             mock.patch.object(jobs.subprocess, "Popen") as supervisor:
            for provider, model in PROVIDER_MODELS:
                with self.subTest(provider=provider):
                    route = {"provider": provider, "model": "fixture"}
                    with mock.patch.object(
                        jobs, "_preflight",
                        return_value=(refusal, self.workdir, route),
                    ):
                        started = jobs.start(
                            task="fixture task",
                            model=model,
                            workdir=self.workdir,
                            read_only=(provider == "grok"),
                            fence_requested=False,
                        )
                    with open(os.path.join(started["job_dir"], "job.json"),
                              encoding="utf-8") as handle:
                        planned = json.load(handle)
                    self.assertTrue(planned["fence_requested"])
                    self.assertEqual(planned["fence"], "seatbelt-fs")
                    self.assertEqual(planned["provider"], provider)
                    self.assertEqual(
                        started["refused"]["reason"], "fixture_after_route"
                    )

        supervisor.assert_not_called()

    def test_foreman_and_non_mac_keep_legacy_admission(self):
        legacy = {
            "ok": False,
            "reason": "fixture_legacy_canary",
            "detail": "legacy admission sentinel",
        }
        for system, foreman in (("Darwin", True), ("Linux", False)):
            with self.subTest(system=system, foreman=foreman), \
                 mock.patch.object(agent_lane.platform, "system",
                                   return_value=system), \
                 mock.patch.object(agent_lane, "env_depth", return_value=0), \
                 mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                 mock.patch.object(agent_lane, "resolve_profile",
                                   return_value=({}, None)), \
                 mock.patch.object(agent_lane, "effort_for_provider",
                                   return_value=(None, None)), \
                 mock.patch.object(agent_lane.sandbox_canary, "gate",
                                   return_value=legacy) as old_canary, \
                 mock.patch.object(agent_lane, "_apply_kernel_fence") as common:
                result = _run(agent_lane.run_agent(
                    task="fixture task",
                    model="codex/fixture",
                    workdir=self.workdir,
                    timeout_s=30,
                    foreman=foreman,
                ))
            self.assertEqual(result["reason"], "fixture_legacy_canary")
            old_canary.assert_called_once_with(False, "codex")
            common.assert_not_called()


if __name__ == "__main__":
    unittest.main()
