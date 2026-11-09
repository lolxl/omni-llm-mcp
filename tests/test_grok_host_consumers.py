"""Hermetic tests for the direct Grok CLI host-fence boundaries."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from providers import grok, grok_image_cli
from providers.host_fence import HostFenceError


class GrokTextHostFenceTests(unittest.TestCase):
    def test_text_cli_receives_prepared_argv_and_env(self):
        captured = {}
        grok_home = tempfile.mkdtemp(prefix="grok-host-test-home-")
        self.addCleanup(__import__("shutil").rmtree, grok_home, True)
        child_env = {"CHILD": "sanitized", "GROK_HOME": grok_home}

        def prepare(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            captured["prompt_file_exists_during_prepare"] = os.path.isfile(
                kwargs["extra_read_paths"][0])
            prepared = list(argv)
            prepared[prepared.index("--sandbox") + 1] = "off"
            return ["host-exec", "--", *prepared], dict(child_env), {
                "fence": "seatbelt-fs",
            }

        async def fake_run_cli(argv, **kwargs):
            captured["watch_stdout"] = kwargs.get("watch_stdout")
            sid = argv[argv.index("--session-id") + 1]
            stream = [{"type": "available_commands", "tools": []},
                      {"type": "text", "data": "fixture-output"},
                      {"type": "end", "stopReason": "end_turn",
                       "sessionId": sid, "usage": {}}]
            return {"returncode": 0, "stderr": "",
                    "stdout": "".join(json.dumps(e) + "\n" for e in stream)}

        run_cli = mock.AsyncMock(side_effect=fake_run_cli)
        with mock.patch.object(grok.shutil, "which", return_value="/fixture/grok"), \
                mock.patch.object(grok, "prepare_host_fence", side_effect=prepare), \
                mock.patch.object(grok, "run_cli", run_cli), \
                mock.patch.object(grok, "extract_cli_result", return_value=("fixture text", {})):
            result = asyncio.run(grok.run_grok_subprocess("fixture prompt", "fixture-model", 5))

        self.assertTrue(result["ok"])
        # Released text is the stream's own text events, exact (not re-extracted).
        self.assertEqual(result["text"], "fixture-output")
        self.assertEqual(captured["kwargs"]["provider"], "grok")
        self.assertTrue(captured["kwargs"]["read_only"])
        self.assertTrue(captured["prompt_file_exists_during_prepare"])
        self.assertEqual(captured["argv"][0], "/fixture/grok")
        self.assertEqual(captured["argv"][captured["argv"].index("--sandbox") + 1], "strict")
        run_cli.assert_awaited_once()
        self.assertEqual(run_cli.await_args.args[0][0:2], ["host-exec", "--"])
        forwarded = run_cli.await_args.args[0]
        self.assertEqual(forwarded[forwarded.index("--sandbox") + 1], "off")
        self.assertEqual(run_cli.await_args.kwargs["env"], child_env)
        self.assertIs(captured["watch_stdout"], False)

    def test_text_fence_refusal_returns_before_run_cli(self):
        refusal = HostFenceError({
            "ok": False,
            "reason": "fence_unavailable",
            "detail": "fixture host fence refusal",
        })
        run_cli = mock.AsyncMock()
        with mock.patch.object(grok.shutil, "which", return_value="/fixture/grok"), \
                mock.patch.object(grok, "prepare_host_fence", side_effect=refusal), \
                mock.patch.object(grok, "run_cli", run_cli):
            result = asyncio.run(grok.run_grok_subprocess("fixture prompt", "fixture-model", 5))

        self.assertEqual(result, refusal.envelope)
        run_cli.assert_not_awaited()


class GrokImageHostFenceTests(unittest.TestCase):
    def test_image_cli_limits_declared_paths_and_receives_prepared_command(self):
        with tempfile.TemporaryDirectory(prefix="grok-host-test-") as root:
            out_dir = os.path.join(root, "out")
            os.mkdir(out_dir)
            source = os.path.join(root, "reference.png")
            with open(source, "wb") as image_file:
                image_file.write(b"local fixture")

            captured = {}

            def prepare(argv, **kwargs):
                captured["argv"] = list(argv)
                captured["kwargs"] = kwargs
                return ["host-exec", "--", *argv], {"CHILD": "sanitized"}, {
                    "fence": "seatbelt-fs",
                }

            def fake_run(argv, *, cwd, env, **kwargs):
                captured["run"] = (list(argv), cwd, dict(env))
                with open(os.path.join(cwd, "image_edit.png"), "wb") as image_file:
                    image_file.write(b"generated fixture")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            async def fake_to_thread(func):
                return func()

            with mock.patch.object(grok_image_cli.shutil, "which", return_value="/fixture/grok"), \
                    mock.patch.object(grok_image_cli, "prepare_host_fence", side_effect=prepare), \
                    mock.patch.object(grok_image_cli.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(grok_image_cli.asyncio, "to_thread", side_effect=fake_to_thread):
                result = asyncio.run(grok_image_cli.run_grok_image_edit(
                    "fixture edit", source, out_dir=out_dir))

        self.assertTrue(result["ok"])
        self.assertEqual(result["images"][0]["path"], os.path.join(out_dir, "image_edit.png"))
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["provider"], "grok")
        self.assertFalse(kwargs["read_only"])
        self.assertEqual(kwargs["workdir"], out_dir)
        self.assertEqual(kwargs["extra_read_paths"], (source,))
        self.assertEqual(kwargs.get("extra_write_paths", ()), ())
        self.assertEqual(captured["run"][0][0:2], ["host-exec", "--"])
        self.assertEqual(captured["run"][2], {"CHILD": "sanitized"})

    def test_image_fence_refusal_returns_before_spawn(self):
        refusal = HostFenceError({
            "ok": False,
            "reason": "fence_unavailable",
            "detail": "fixture host fence refusal",
        })
        with tempfile.TemporaryDirectory(prefix="grok-host-test-") as root:
            out_dir = os.path.join(root, "out")
            os.mkdir(out_dir)
            source = os.path.join(root, "reference.png")
            with open(source, "wb") as image_file:
                image_file.write(b"local fixture")
            to_thread = mock.AsyncMock()
            run = mock.Mock()
            with mock.patch.object(grok_image_cli.shutil, "which", return_value="/fixture/grok"), \
                    mock.patch.object(grok_image_cli, "prepare_host_fence", side_effect=refusal), \
                    mock.patch.object(grok_image_cli.asyncio, "to_thread", to_thread), \
                    mock.patch.object(grok_image_cli.subprocess, "run", run):
                result = asyncio.run(grok_image_cli.run_grok_image_edit(
                    "fixture edit", source, out_dir=out_dir))

        self.assertEqual(result, refusal.envelope)
        to_thread.assert_not_awaited()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
