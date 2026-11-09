#!/usr/bin/env python3
# no-opencc
"""run_agent refuses a gated task before account selection or any git query."""
import asyncio
import contextlib
import os
import tempfile
import unittest
from unittest import mock

from providers.agent_lane import run_agent


# Fixture value used as a payload. Assertions must never see it in error text.
TW_MOBILE = "0912345678"

_MODELS = ("grok-4.6", "claude/sonnet", "codex/gpt-5.5")

# Record a call and raise. None of these may run when the gate denies.
_SIDE_EFFECTS = (
    "subprocess.run",
    "subprocess.Popen",
    "asyncio.create_subprocess_exec",
    "providers.agent_lane._workdir_fence",
    "providers.provider_accounts.AccountSelection",
    "providers.codex_accounts.AccountSelection",
)

# A configured account ring can return before the workdir fence. Drop it so
# the unset-gate case still reaches that fence, and skip any quota probe.
_RING_VARS = (
    "OMNI_LLM_PROFILES_GROK",
    "OMNI_LLM_PROFILES_CLAUDE",
    "OMNI_LLM_PROFILES_CODEX",
    "OMNI_LLM_API_PROFILES_CLAUDE",
)


def _env(gate=None, denylist=None):
    """Patch OMNI_EGRESS_* only. None = unset that var."""
    env = os.environ.copy()
    env.pop("OMNI_EGRESS_GATE", None)
    env.pop("OMNI_EGRESS_DENYLIST", None)
    if gate is not None:
        env["OMNI_EGRESS_GATE"] = gate
    if denylist is not None:
        env["OMNI_EGRESS_DENYLIST"] = denylist
    return mock.patch.dict(os.environ, env, clear=True)


def _detail(envelope):
    return str((envelope or {}).get("detail") or "")


class _Called(Exception):
    """Sentinel raised by a patched seam."""


class TestRunAgentRefuseEarly(unittest.TestCase):
    def test_gate_on_refuses_before_selection_or_git(self):
        task = "call me at %s please" % TW_MOBILE
        for model in _MODELS:
            with self.subTest(model=model):
                called = []

                with _env(gate="1"):
                    os.environ.pop("OMNI_LLM_DEPTH", None)
                    with tempfile.TemporaryDirectory() as workdir:
                        with contextlib.ExitStack() as stack:
                            for target in _SIDE_EFFECTS:
                                def boom(*_a, _name=target, **_k):
                                    called.append(_name)
                                    raise _Called(_name)
                                stack.enter_context(mock.patch(target, boom))
                            try:
                                result = asyncio.run(run_agent(
                                    task, model, workdir, timeout_s=5))
                            except _Called as exc:
                                self.fail(
                                    "seam called before refuse: %s" % exc)
                self.assertEqual(called, [])
                self.assertFalse(result.get("ok"), result)
                self.assertTrue(
                    _detail(result).startswith(
                        "OMNI-EGRESS-DENY [tw_mobile]"),
                    _detail(result),
                )
                self.assertNotIn(TW_MOBILE, _detail(result))

    def test_gate_unset_reaches_workdir_fence(self):
        task = "call me at %s please" % TW_MOBILE
        for model in _MODELS:
            with self.subTest(model=model):
                called = []

                def fence(*_a, **_k):
                    called.append(True)
                    raise _Called("workdir_fence")

                with _env():
                    os.environ.pop("OMNI_LLM_DEPTH", None)
                    for name in _RING_VARS:
                        os.environ.pop(name, None)
                    os.environ["OMNI_LLM_QUOTA_PROBE"] = "0"
                    with tempfile.TemporaryDirectory() as workdir:
                        with mock.patch(
                                "providers.agent_lane._workdir_fence", fence):
                            propagated = False
                            try:
                                asyncio.run(run_agent(
                                    task, model, workdir, timeout_s=5))
                            except _Called:
                                propagated = True
                self.assertTrue(
                    propagated or called,
                    "workdir fence was not reached",
                )
