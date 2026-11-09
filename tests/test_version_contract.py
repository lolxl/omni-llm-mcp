#!/usr/bin/env python3
# no-opencc
"""`omni --version` and `omni contract`: offline, no vendor binary, no state-dir write."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import omni_cli  # noqa: E402
import server  # noqa: E402
from providers import dispatch_exec, dispatch_result  # noqa: E402
from providers.dispatch import TOOLS  # noqa: E402
from providers.version import CONTRACT_VERSION, __version__  # noqa: E402

CLI = os.path.join(HERE, "omni_cli.py")
KEYS = {"name", "version", "contract_version", "cli", "mcp", "dispatch_exec", "state_dir"}


def _run(*args, env_extra=None):
    env = dict(os.environ)
    env["OMNI_LLM_NOTIFY_QUIET"] = "1"
    env.pop("OMNI_LLM_STATE_DIR", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True,
                          env=env, timeout=60)


class VersionTests(unittest.TestCase):
    def test_version_flag(self):
        r = _run("--version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "omni %s\n" % __version__)
        self.assertEqual(r.stdout.strip(), "omni 0.10.0")

    def test_server_version_is_the_same_source(self):
        self.assertEqual(server.SERVER_VERSION, __version__)


class ContractTests(unittest.TestCase):
    def test_contract_shape(self):
        r = _run("contract")
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(set(doc), KEYS)
        self.assertEqual(doc["name"], "omni-llm-mcp")
        self.assertEqual(doc["version"], server.SERVER_VERSION)
        self.assertEqual(doc["contract_version"], CONTRACT_VERSION)
        self.assertEqual(set(doc["cli"]), {"subcommands"})
        for name in ("host-exec", "dispatch-exec", "agent", "gen", "contract"):
            self.assertIn(name, doc["cli"]["subcommands"])
        self.assertEqual(doc["cli"]["subcommands"], sorted(doc["cli"]["subcommands"]))
        self.assertEqual(doc["mcp"]["server_name"], server.SERVER_NAME)
        self.assertEqual(doc["mcp"]["tools"], sorted(t["name"] for t in TOOLS))
        de = doc["dispatch_exec"]
        self.assertEqual(set(de), {"result_record_version", "providers", "exit_codes"})
        self.assertEqual(de["result_record_version"], dispatch_result.VERSION)
        self.assertEqual(de["providers"], ["grok"])
        self.assertEqual(de["exit_codes"], dispatch_exec.EXIT_CODES)
        self.assertTrue(all(isinstance(k, str) and k.isdigit() for k in de["exit_codes"]))

    def test_exit_code_table_keys_and_shared_start_failure_text(self):
        self.assertEqual(set(dispatch_exec.EXIT_CODES),
                         {"2", "4", "6", "8", "9", "10", "125", "126", "127",
                          "129", "130", "143"})
        self.assertEqual(dispatch_exec.EXIT_CODES["126"],
                         dispatch_exec.EXIT_CODES["127"])

    def test_state_dir_reported_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "x")
            r = _run("contract", env_extra={"OMNI_LLM_STATE_DIR": target})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["state_dir"], os.path.abspath(target))
            self.assertFalse(os.path.exists(target))
            self.assertEqual(os.listdir(tmp), [])

    def test_subcommands_derived_from_parser(self):
        from providers.contract import build_contract
        parser = omni_cli.build_parser()
        sub_action = next(a for a in parser._subparsers._group_actions
                          if hasattr(a, "choices") and a.choices)
        self.assertEqual(build_contract(parser)["cli"]["subcommands"],
                         sorted(sub_action.choices))
        # Drift guard: a subparser added to build_parser() appears without any
        # other edit.
        sub_action.add_parser("zz-drift-probe")
        self.assertIn("zz-drift-probe", build_contract(parser)["cli"]["subcommands"])
        self.assertNotIn("zz-drift-probe",
                         build_contract(omni_cli.build_parser())["cli"]["subcommands"])


if __name__ == "__main__":
    unittest.main()
