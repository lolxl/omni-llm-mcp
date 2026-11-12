"""Hermetic tests for the direct Grok CLI host-fence boundaries."""
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from providers import grok, grok_image_cli
from providers.host_fence import HostFenceError


def _seed_grok(rows):
    """Write a grok v2 record. ``rows`` is ``(id, supported_efforts)``."""
    from providers import catalog_v2
    from providers.catalog_record import validate_strict
    record, _warnings = validate_strict("grok", {
        "provider": "grok",
        "source": "live",
        "observed_at": "2026-11-10T00:00:00Z",
        "availability": "provider_advertised",
        "models": [{
            "id": model_id,
            "name": model_id,
            "routable_id": "grok/%s" % model_id,
            "supported_efforts": list(efforts),
        } for model_id, efforts in rows],
    })
    catalog_v2.save("grok", "default", record)


def _seed_grok_image():
    from providers.grok_image_cli import GROK_IMAGE_MODEL
    _seed_grok([(GROK_IMAGE_MODEL, ["medium"])])


def _pin_state(case):
    state = tempfile.mkdtemp(prefix="grok-admit-")
    case.addCleanup(shutil.rmtree, state, True)
    env = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": state})
    env.start()
    case.addCleanup(env.stop)


class GrokTextHostFenceTests(unittest.TestCase):
    def setUp(self):
        # Plain unittest uses the real gate. The text lane admits fixture-model
        # before the fence; today's gate ignores this row.
        _pin_state(self)
        _seed_grok([("fixture-model", ["low", "medium", "high"])])

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

        expected = dict(refusal.envelope)
        # Consult now runs through the account ring, which marks a final
        # unmanaged failure account_retry_safe. Today's gate keeps its note;
        # a catalog admission discloses the folded effort.
        expected["account_retry_safe"] = False
        if result.get("effort_source") == "today_gate":
            expected["effort"] = None
            expected["effort_source"] = "today_gate"
            expected["effort_note"] = "effort: (today's gate)"
        else:
            expected["effort"] = "medium"
            expected["effort_source"] = "omni_default"
            expected["effort_note"] = (
                "effort: medium (chosen by omni; no effort was given)")
        self.assertEqual(result, expected)
        run_cli.assert_not_awaited()


class GrokImageHostFenceTests(unittest.TestCase):
    def setUp(self):
        _pin_state(self)
        _seed_grok_image()

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
