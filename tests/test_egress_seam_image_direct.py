#!/usr/bin/env python3
# no-opencc
"""Egress residual seam (3/4): image-module direct call.

providers.grok_image_cli / providers.xai_image call refuse_payload at the
module entry so an in-process import cannot bypass dispatch.call_tool(
'llm_image'). Gate on + synthetic PII in the prompt refuses before
subprocess.run / urlopen. Gate off leaves the bare-run gap. No network:
subprocess.run / urlopen are intercepted. Synthetic tokens only. XAI_API_BASE
is pinned to a loopback URL so a missed patch cannot hit api.x.ai.
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from providers import grok_image_cli
from providers import kernel_sandbox
from providers import xai_image


TW_MOBILE = "0912345678"


def _env(gate="1"):
    env = os.environ.copy()
    env.pop("OMNI_EGRESS_GATE", None)
    env.pop("OMNI_EGRESS_DENYLIST", None)
    env["OMNI_LLM_METRICS"] = "0"
    env.pop("OMNI_LLM_DEPTH", None)
    env["XAI_API_BASE"] = "http://127.0.0.1:1"
    env["XAI_API_KEY"] = "fixture-local-key"
    if gate is not None:
        env["OMNI_EGRESS_GATE"] = gate
    return mock.patch.dict(os.environ, env, clear=True)


def _detail(envelope):
    return str((envelope or {}).get("detail") or "")


class TestImageModuleDirectRefuse(unittest.TestCase):
    """Entries:
    - providers/grok_image_cli.py run_grok_image_edit → refuse_payload
      before subprocess.run
    - providers/xai_image.py run_xai_image → refuse_payload before urlopen
    """

    def test_grok_image_cli_direct_refuses_with_gate_on(self):
        sent = []
        prompt = "call me at %s please" % TW_MOBILE
        source = "/tmp/fixture-ref.png"
        out = tempfile.mkdtemp(prefix="omni_seam_img_")

        def boom(*_a, **_k):
            sent.append("grok_image_cli")
            raise RuntimeError("stubbed send reached")

        with _env(gate="1"):
            with mock.patch("providers.grok_image_cli.shutil.which",
                            return_value="/usr/bin/grok"):
                with mock.patch("providers.grok_image_cli.subprocess.run", boom):
                    res = asyncio.run(grok_image_cli.run_grok_image_edit(
                        prompt, source, timeout_s=5, out_dir=out))
        self.assertFalse((res or {}).get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "direct grok_image_cli must not reach subprocess.run")

    def test_xai_image_direct_refuses_with_gate_on(self):
        sent = []
        prompt = "call me at %s please" % TW_MOBILE

        def boom(*_a, **_k):
            sent.append("xai_image")
            raise RuntimeError("stubbed send reached")

        with _env(gate="1"):
            with mock.patch("providers.xai_image.urllib.request.urlopen", boom):
                res = asyncio.run(xai_image.run_xai_image(
                    prompt, timeout_s=5))
        self.assertFalse((res or {}).get("ok"), res)
        self.assertTrue(
            _detail(res).startswith("OMNI-EGRESS-DENY [tw_mobile]"),
            _detail(res),
        )
        self.assertNotIn(TW_MOBILE, _detail(res))
        self.assertEqual(sent, [], "direct xai_image must not reach urlopen")

    def test_grok_image_cli_direct_reaches_send_with_gate_off(self):
        sent = []
        prompt = "call me at %s please" % TW_MOBILE
        out = tempfile.mkdtemp(prefix="omni_seam_img_")
        source = os.path.join(out, "fixture-ref.png")
        with open(source, "wb") as fh:
            fh.write(b"fixture image")

        # Keep the host-canary subprocess seam separate from the provider
        # runner. Patching subprocess.run alone makes both names point at the
        # same module function and can count the canary as a provider send.
        tokens = iter((
            "WROTE", "ERRNO=1", "ERRNO=1",
            "READ_OK", "ERRNO=1", "ERRNO=1",
        ))

        def fake_probe(argv, _env):
            return subprocess.CompletedProcess(argv, 0, next(tokens) + "\n", "")

        # prepare_host_fence runs a parent `git rev-parse` through the same
        # subprocess.run before the send. Let that (and any other non-vendor
        # call) through to the real function; only the vendor argv, which
        # carries the stubbed grok binary, counts as the send.
        real_run = subprocess.run

        def boom(argv, *a, **k):
            if "/usr/bin/grok" not in list(argv):
                return real_run(argv, *a, **k)
            sent.append("grok_image_cli")
            raise RuntimeError("stubbed send reached")

        fd = os.open(os.devnull, os.O_RDONLY)
        canary = os.path.join(
            kernel_sandbox.protected_home(), "omni_grok_image_probe_fixture",
        )
        with _env(gate=None):
            with mock.patch.object(kernel_sandbox, "_run_probe", fake_probe), \
                    mock.patch.object(
                        kernel_sandbox.tempfile, "mkstemp",
                        return_value=(fd, canary),
                    ), mock.patch.object(kernel_sandbox.os, "remove"), \
                    mock.patch("providers.grok_image_cli.shutil.which",
                               return_value="/usr/bin/grok"), \
                    mock.patch("providers.grok_image_cli.subprocess.run", boom):
                res = asyncio.run(grok_image_cli.run_grok_image_edit(
                    prompt, source, timeout_s=5, out_dir=out))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(
            sent, ["grok_image_cli"],
            "direct grok_image_cli must still reach subprocess.run",
        )

    def test_xai_image_direct_reaches_send_with_gate_off(self):
        sent = []
        prompt = "call me at %s please" % TW_MOBILE

        def boom(*_a, **_k):
            sent.append("xai_image")
            raise RuntimeError("stubbed send reached")

        with _env(gate=None):
            with mock.patch("providers.xai_image.urllib.request.urlopen", boom):
                res = asyncio.run(xai_image.run_xai_image(
                    prompt, timeout_s=5))
        self.assertNotIn("OMNI-EGRESS-DENY", _detail(res))
        self.assertEqual(
            sent, ["xai_image"],
            "direct xai_image must still reach urlopen",
        )


if __name__ == "__main__":
    unittest.main()
