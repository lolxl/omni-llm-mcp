"""Exercise the image consumer through real Seatbelt with a local CLI stub."""
import asyncio
import errno
import json
import os
import platform
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from providers import grok_image_cli, kernel_sandbox


_STUB = r'''#!/usr/bin/python3
import errno, json, sys
from pathlib import Path

args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
source_path = Path(prompt.split(
    "Source reference image (filesystem path): ", 1)[1].splitlines()[0])
out_path = Path(prompt.rsplit(
    "Save the resulting image as a PNG to ", 1)[1].split(
        " then print the absolute saved path.", 1)[0])
read_canary = Path(__READ_CANARY__)
write_canary = Path(__WRITE_CANARY__)
result = {"source_errno": 0, "source": ""}
try:
    result["source"] = source_path.read_text(encoding="utf-8")
except OSError as exc:
    result["source_errno"] = exc.errno
try:
    read_canary.read_text(encoding="utf-8")
    result["sibling_read_errno"] = 0
except OSError as exc:
    result["sibling_read_errno"] = exc.errno
try:
    write_canary.write_text("unexpected", encoding="utf-8")
    result["sibling_write_errno"] = 0
except OSError as exc:
    result["sibling_write_errno"] = exc.errno
result["inner_sandbox"] = args[args.index("--sandbox") + 1]
out_path.write_bytes(b"PNG fixture")
(out_path.parent / "host-observation.json").write_text(
    json.dumps(result), encoding="utf-8")
print("synthetic image fixture complete")
'''


@unittest.skipUnless(
    platform.system() == "Darwin"
    and kernel_sandbox.available()
    and os.path.isfile("/usr/bin/python3"),
    "requires macOS Seatbelt and /usr/bin/python3",
)
class GrokImageHostFenceLiveTests(unittest.TestCase):
    def test_production_image_consumer_gets_scoped_home_access(self):
        protected_home = Path(kernel_sandbox.protected_home()).resolve()
        with tempfile.TemporaryDirectory(
                prefix="host-grok-image-", dir=protected_home) as raw_root, \
                tempfile.TemporaryDirectory(
                    prefix=".grok-host-test-", dir=protected_home) as raw_grok_home, \
                tempfile.TemporaryDirectory(prefix="grok-cli-stub-") as raw_bin:
            root = Path(raw_root)
            out_dir = root / "out"
            out_dir.mkdir()
            source = root / "reference.png"
            source.write_text("admitted synthetic image", encoding="utf-8")
            read_canary = root / "sibling-secret.txt"
            read_canary.write_text("synthetic denied read", encoding="utf-8")
            write_canary = root / "sibling-write.txt"

            stub = Path(raw_bin) / "grok"
            stub.write_text(
                _STUB.replace("__READ_CANARY__", repr(str(read_canary)))
                .replace("__WRITE_CANARY__", repr(str(write_canary))),
                encoding="utf-8",
            )
            stub.chmod(0o755)

            env = {
                "PATH": str(stub.parent) + os.pathsep + os.environ.get("PATH", ""),
                "GROK_HOME": raw_grok_home,
            }
            with mock.patch.dict(os.environ, env):
                result = asyncio.run(grok_image_cli.run_grok_image_edit(
                    "synthetic image edit", str(source), out_dir=str(out_dir),
                ))

            self.assertTrue(result.get("ok"), result)
            self.assertEqual(
                Path(result["images"][0]["path"]).read_bytes(), b"PNG fixture")
            observation = json.loads(
                (out_dir / "host-observation.json").read_text(encoding="utf-8"))
            self.assertEqual(observation["source_errno"], 0)
            self.assertEqual(observation["source"], "admitted synthetic image")
            self.assertEqual(observation["sibling_read_errno"], errno.EPERM)
            self.assertEqual(observation["sibling_write_errno"], errno.EPERM)
            self.assertEqual(observation["inner_sandbox"], "off")
            self.assertFalse(write_canary.exists())


if __name__ == "__main__":
    unittest.main()
