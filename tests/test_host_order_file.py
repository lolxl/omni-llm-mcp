"""Narrow Grok workorder admission and real host read/write controls."""
import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

from providers import host_fence as h, kernel_sandbox as k
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


class OrderFileAdmissionTests(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.inbox = os.path.join(self.root, "inbox")
        os.mkdir(self.inbox)
        self.order = os.path.join(self.inbox, "order.txt")
        with open(self.order, "w") as handle:
            handle.write("synthetic order")

    def args(self, path=None, provider="grok"):
        path = self.order if path is None else path
        return ["--provider", provider, "--workdir", self.root,
                "--grok-order-file", path, "--", "/fake/grok", "--prompt-file", path]

    def test_invalid_declaration_refuses_before_vendor_exec(self):
        escape = os.path.join(self.root, "outside.txt")
        with open(escape, "w") as handle:
            handle.write("outside")
        link = os.path.join(self.inbox, "link")
        os.symlink(self.order, link)
        link_inbox = os.path.join(self.root, "inbox-link")
        os.symlink(self.inbox, link_inbox)
        cases = [
            ("non-grok", self.args(provider="pi"), self.inbox),
            ("mismatch", self.args()[:-1] + [escape], self.inbox),
            ("symlink", self.args(link), self.inbox),
            ("directory", self.args(self.inbox), self.inbox),
            ("missing", self.args(os.path.join(self.inbox, "absent")), self.inbox),
            ("inbox escape", self.args(escape), self.inbox),
            ("relative", self.args("order.txt"), self.inbox),
            ("inbox symlink", self.args(), link_inbox),
            ("inbox missing", self.args(), ""),
            ("inbox file", self.args(), self.order),
            ("duplicate", self.args()[:6] + ["--grok-order-file", self.order]
             + self.args()[6:], self.inbox),
            ("delimiter", self.args()[:-2] + ["--", "--prompt-file", self.order], self.inbox),
            ("operand", self.args()[:-2] + ["-p", "--prompt-file", self.order], self.inbox),
        ]
        for label, args, inbox in cases:
            with self.subTest(case=label), mock.patch.dict(os.environ, {"TEAM_INBOX": inbox}), \
                    mock.patch.object(h.os, "execvpe") as execute, \
                    mock.patch.object(h.sys, "stderr", io.StringIO()) as err:
                self.assertEqual(h.main(args), 126)
                self.assertEqual(json.loads(err.getvalue())["reason"], "fence_path_rejected")
                execute.assert_not_called()

    def test_literal_read_and_write_deny_are_shared_by_verify_and_wrap(self):
        policy = {"wrap_kw": {"read_wall": True,
                              "extra_deny_read": tuple(h._grok_instruction_rows(None))}}
        with mock.patch.object(h.platform, "system", return_value="Darwin"), \
                mock.patch.object(k, "available", return_value=True), \
                mock.patch.object(k, "agent_fence_policy", return_value=policy), \
                mock.patch.object(k, "verify_fence", return_value=None) as verify, \
                mock.patch.object(k, "wrap_argv", side_effect=lambda argv, *a, **kw:
                                  [k.SANDBOX_EXEC, "-p", "fixture", *argv]) as wrap, \
                mock.patch.dict(os.environ, {"TEAM_INBOX": self.inbox}), \
                mock.patch.object(h.os, "chdir"), mock.patch.object(h.os, "execvpe") as execute:
            h.main(self.args())
        reads = verify.call_args.kwargs["extra_read_paths"]
        self.assertEqual([(entry.kind, entry.path) for entry in reads],
                         [("literal", self.order)])
        self.assertEqual(verify.call_args.kwargs["extra_deny_write"],
                         (("literal", self.order),))
        self.assertEqual(wrap.call_args.kwargs["extra_read_paths"], reads)
        self.assertEqual(wrap.call_args.kwargs["extra_deny_write"],
                         verify.call_args.kwargs["extra_deny_write"])
        self.assertEqual(execute.call_args.args[1][-2:], ["--prompt-file", self.order])

    def test_repository_postallow_over_the_order_refuses_before_verification(self):
        # Repository T* rows are appended after the policy's own; they come
        # after the order's W* deny too, so they must be checked as well.
        repo = ((), (), (), (("subpath", self.inbox),), ())
        with mock.patch.object(h.platform, "system", return_value="Darwin"), \
                mock.patch.object(k, "available", return_value=True), \
                mock.patch.object(h, "_repo_write_roots", return_value=repo), \
                mock.patch.object(k, "verify_fence") as verify:
            with self.assertRaises(h.HostFenceError):
                h.prepare_host_fence(
                    ["/fake/grok", "--prompt-file", self.order], provider="grok",
                    workdir=self.root, read_only=False, env={"TEAM_INBOX": self.inbox},
                    grok_order_file=self.order, policy={"wrap_kw": {}})
            verify.assert_not_called()

    def test_runtime_postallow_overlap_refuses_before_verification(self):
        def other_case(path):
            # Same directory on a case-insensitive volume; Seatbelt matches it too.
            return os.path.join(os.path.dirname(path), os.path.basename(path).swapcase())
        for entry in (("literal", self.order), ("subpath", self.inbox), ("regex", ".*"),
                      ("literal", other_case(self.order)), ("subpath", other_case(self.inbox))):
            with self.subTest(entry=entry), \
                    mock.patch.object(h.platform, "system", return_value="Darwin"), \
                    mock.patch.object(k, "available", return_value=True), \
                    mock.patch.object(k, "verify_fence") as verify:
                with self.assertRaises(h.HostFenceError):
                    h.prepare_host_fence(
                        ["/fake/grok", "--prompt-file", self.order], provider="grok",
                        workdir=self.root, env={"TEAM_INBOX": self.inbox},
                        grok_order_file=self.order,
                        policy={"wrap_kw": {"extra_post_write": (entry,)}})
                verify.assert_not_called()


_TOOL = r'''#!/usr/bin/python3
import json, os, subprocess, sys
order = sys.argv[-1]
def check(operation):
    try:
        return {"errno": 0, "value": operation()}
    except OSError as exc:
        return {"errno": exc.errno}
def read(path):
    with open(path) as handle:
        return handle.read()
def write(path):
    with open(path, "w") as handle:
        return handle.write("changed")
direct = {
    "order_read": check(lambda: read(order)),
    "order_write": check(lambda: write(order)),
    "parent_list": check(lambda: os.listdir(os.path.dirname(order))),
    "sibling_read": check(lambda: read(os.environ["ORDER_SIBLING"])),
    "sibling_write": check(lambda: write(os.environ["ORDER_SIBLING"])),
    "other_read": check(lambda: read(os.environ["ORDER_OTHER"])),
}
if os.environ.get("ORDER_CHILD") == "1":
    print(json.dumps(direct))
else:
    child = subprocess.run([sys.executable, __file__, *sys.argv[1:]],
        env={**os.environ, "ORDER_CHILD": "1"}, capture_output=True, text=True, check=True)
    print(json.dumps({"direct": direct, "child": json.loads(child.stdout)}))
'''


@unittest.skipUnless(k.available(), "macOS Seatbelt required")
class OrderFileHostTests(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)

    def test_cli_named_home_order_only_and_tmp_order_remains_unwritable(self):
        with ExitStack() as stack:
            home_fixture = stack.enter_context(tempfile.TemporaryDirectory(
                prefix=".omni_order_fixture_", dir=k.protected_home()))
            temp_fixture = stack.enter_context(tempfile.TemporaryDirectory())
            workdir = stack.enter_context(tempfile.TemporaryDirectory())
            tool = os.path.join(workdir, "fixture-tool")
            with open(tool, "w") as handle:
                handle.write(_TOOL)
            os.chmod(tool, 0o700)
            other = os.path.join(home_fixture, "private-fixture.txt")
            with open(other, "w") as handle:
                handle.write("other synthetic fixture")
            for fixture in (home_fixture, temp_fixture):
                with self.subTest(home=fixture == home_fixture):
                    inbox = os.path.join(os.path.realpath(fixture), "repo", "inbox")
                    os.makedirs(inbox)
                    order = os.path.join(inbox, "approved-order.txt")
                    sibling = os.path.join(inbox, "sibling.txt")
                    for path in (order, sibling):
                        with open(path, "w") as handle:
                            handle.write("synthetic order")
                    # Controls establish readable/listable/writable fixtures.
                    self.assertIn("approved-order.txt", os.listdir(inbox))
                    for path in (order, sibling, other):
                        with open(path) as handle:
                            self.assertTrue(handle.read())
                        with open(path, "a") as handle:
                            handle.write("")
                    env = {**os.environ, "TEAM_INBOX": inbox,
                           "ORDER_SIBLING": sibling, "ORDER_OTHER": other,
                           # The fenced tool gets an allowlisted environment
                           # (#138): its fixture variables are passed by name.
                           "OMNI_LLM_CHILD_ENV_PASS":
                               "TEAM_INBOX,ORDER_SIBLING,ORDER_OTHER"}
                    with mock.patch.dict(os.environ, env), mock.patch.object(h.os, "chdir"), \
                            mock.patch.object(h.os, "execvpe") as execute:
                        h.main(["--provider", "grok", "--workdir", workdir,
                                "--grok-order-file", order, "--", tool,
                                "--prompt-file", order])
                    execute.assert_called_once()
                    _binary, argv, child_env = execute.call_args.args
                    result = subprocess.run(argv, cwd=workdir, env=child_env,
                                            capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    observed = json.loads(result.stdout)
                    for generation in ("direct", "child"):
                        row = observed[generation]
                        self.assertEqual(row["order_read"],
                                         {"errno": 0, "value": "synthetic order"})
                        self.assertEqual(row["order_write"]["errno"], errno.EPERM)
                        self.assertEqual(row["other_read"]["errno"], errno.EPERM)
                        if fixture == home_fixture:
                            for operation in ("parent_list", "sibling_read", "sibling_write"):
                                self.assertEqual(row[operation]["errno"], errno.EPERM)
                    with open(order) as handle:
                        self.assertEqual(handle.read(), "synthetic order")


if __name__ == "__main__":
    unittest.main()
