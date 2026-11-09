#!/usr/bin/env python3
# no-opencc
"""Egress residual seam (1/4): bare-run — refuse frontend default-off.

CURRENT-BEHAVIOR: when OMNI_EGRESS_GATE is unset (the documented default),
consult-lane sends are not scanned. This file pins that gap honestly.
No network: send is intercepted. Synthetic fixture tokens only.
"""
import asyncio
import os
import shutil
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from providers import dispatch
from providers import grok as grok_mod
from providers import openai_compat
from providers.egress_gate import refuse_payload


# Synthetic only. Same dummy Taiwan mobile the consult-gate tests already use.
TW_MOBILE = "0912345678"


def _env(gate=None):
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


class TestBareRunDefaultOff(unittest.TestCase):
    """Entry: providers/egress_gate.py:118 refuse_payload (enabled() is False).

    Consult hooks exist (openai_compat.py:50, grok.py:261, dispatch.py:642)
    but they no-op when the env switch is not exactly '1'.
    """

    def test_unset_gate_refuse_payload_is_none(self):
        # CURRENT-BEHAVIOR: default-off consult switch does not scan.
        with _env(gate=None):
            self.assertIsNone(refuse_payload("call me at %s" % TW_MOBILE))

    def test_openai_consult_reaches_send_when_gate_unset(self):
        # CURRENT-BEHAVIOR: bare-run consult HTTP send is not gated.
        sent = []
        prompt = "call me at %s please" % TW_MOBILE

        def boom(*_a, **_k):
            sent.append("openai")
            raise RuntimeError("stubbed send reached")

        with _env(gate=None):
            with mock.patch("urllib.request.urlopen", boom):
                res = asyncio.run(openai_compat.run_openai_compatible(
                    prompt, "gpt-4o", 5, "default"))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(sent, ["openai"], "bare-run must still reach send")

    def test_grok_consult_reaches_send_when_gate_unset(self):
        # CURRENT-BEHAVIOR: bare-run consult CLI send is not gated.
        sent = []
        prompt = "call me at %s please" % TW_MOBILE

        def boom(*_a, **_k):
            sent.append("grok")
            raise RuntimeError("stubbed send reached")

        with _env(gate=None):
            with mock.patch("providers.grok.shutil.which", _fake_which("grok")):
                with mock.patch("providers.grok.run_cli", boom):
                    res = asyncio.run(grok_mod.run_grok_subprocess(
                        prompt, "grok-4.6", 5, "default"))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(sent, ["grok"], "bare-run must still reach send")

    def test_dispatch_llm_generate_reaches_provider_when_gate_unset(self):
        # CURRENT-BEHAVIOR: MCP/CLI consult entry (dispatch) does not scan
        # when the refuse frontend is default-off.
        sent = []
        prompt = "call me at %s please" % TW_MOBILE

        async def boom(*_a, **_k):
            sent.append("dispatch")
            return {"ok": True, "text": "stub", "model": "gpt-4o"}

        with _env(gate=None):
            with mock.patch(
                "providers.dispatch.openai_mod.run_openai_compatible", boom
            ):
                res = asyncio.run(dispatch.call_tool("llm_generate", {
                    "prompt": prompt,
                    "model": "gpt-4o",
                }))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(sent, ["dispatch"], "bare-run dispatch must reach provider")


if __name__ == "__main__":
    unittest.main()
