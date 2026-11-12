#!/usr/bin/env python3
# no-opencc
"""Offline canary for providers.egress_canary. No network, no vendor CLIs.

Each seam runs in a fresh child. The deny payload is built at runtime from
pieces so this source never holds a contiguous fixture value.
"""
import datetime
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

from tests import offline_home

# Seeds v2 records; tests/offline_home.py gives this module the real gate.
OMNI_TEST_REAL_ADMISSION = True


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hard-coded. The module's own list is what --list must equal.
SEAM_IDS = (
    "agent_lane",
    "jobs_start",
    "grok_image",
    "xai_image",
    "cli_grok",
    "cli_codex",
    "cli_claude",
    "cli_pi",
    "cli_agy",
    "cli_cursor",
    "agy_sdk",
    "openai_compat",
    "dispatch_route",
    "dispatch_tool",
)

SIGNATURE = "OMNI-EGRESS-DENY ["
FAILOPEN = "FAIL-OPEN-REACHED-TRANSPORT"
SDK_FALLBACK = "SDK-UNAVAILABLE-FALLBACK"

_HOME = None


def setUpModule():
    global _HOME
    _HOME = offline_home.use_synthetic_home()


def tearDownModule():
    try:
        _HOME.assert_module_hermetic()
    finally:
        _HOME.close()


def _mobile():
    """TW mobile fixture, assembled the same way the shell canary does."""
    mob = "09"
    mob = mob + "1234"
    mob = mob + "5678"
    return mob


def _git_ancestor(path):
    cur = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _mkdtemp_outside_git():
    """A temp dir no git work tree contains. agent_lane fences one."""
    tried = []
    for base in (tempfile.gettempdir(), "/tmp", "/private/tmp", "/var/tmp"):
        if not base or not os.path.isdir(base) or not os.access(base, os.W_OK):
            continue
        base = os.path.realpath(base)
        if _git_ancestor(base):
            tried.append(base)
            continue
        made = tempfile.mkdtemp(prefix="omni_ecanary_", dir=base)
        if _git_ancestor(made):
            shutil.rmtree(made, ignore_errors=True)
            tried.append(made)
            continue
        return made
    raise RuntimeError(
        "no temp directory outside a git work tree (rejected: %s)" % (tried,))


def _redact(text, payload):
    if not text:
        return ""
    return text.replace(payload, "<fixture>")


class EgressCanaryTests(unittest.TestCase):
    def setUp(self):
        _HOME.mark()
        self.addCleanup(_HOME.assert_hermetic, self)
        self.root = _mkdtemp_outside_git()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.workdir = os.path.join(self.root, "wd")
        self.state = os.path.join(self.root, "state")
        self.tmpdir = os.path.join(self.root, "tmp")
        self.bindir = os.path.join(self.root, "bin")
        self.image = os.path.join(self.root, "src.png")
        for path in (self.workdir, self.state, self.tmpdir, self.bindir):
            os.makedirs(path)
        with open(self.image, "wb") as fh:
            fh.write(b"")
        # jobs.start resolves the default model through shutil.which before
        # Popen. The seam stub patches Popen only, so the child needs a
        # non-executed stand-in on PATH or preflight returns cli_missing.
        for name in ("grok", "codex", "claude", "pi", "agy", "cursor-agent"):
            path = os.path.join(self.bindir, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\necho FAKE-BIN-EXECUTED >&2\nexit 0\n")
            os.chmod(path, stat.S_IRWXU)
        self.payload = _mobile()

    def _child_env(self, seam):
        env = os.environ.copy()
        env.pop("OMNI_EGRESS_GATE", None)
        env.pop("OMNI_EGRESS_DENYLIST", None)
        env["TMPDIR"] = self.tmpdir
        env["PYTHONIOENCODING"] = "utf-8"
        env["PATH"] = self.bindir + os.pathsep + env.get("PATH", "")
        if seam == "xai_image":
            for key in list(env):
                if key.startswith("XAI_API_KEY_"):
                    env.pop(key, None)
            env["XAI_API_BASE"] = "http://127.0.0.1:1/v1"
            env["XAI_API_KEY"] = "o38-fixture-not-a-key"
        elif seam == "openai_compat":
            for key in list(env):
                if key.startswith("OPENAI_API_KEY_"):
                    env.pop(key, None)
            env["OPENAI_API_BASE"] = "http://127.0.0.1:1/v1"
            env["OPENAI_API_KEY"] = "o42-fixture-not-a-key"
        return env

    def _invoke(self, args, seam=None, timeout=90, cwd=None, env=None):
        proc = subprocess.Popen(
            [sys.executable, "-m", "providers.egress_canary", *args],
            cwd=cwd or ROOT,
            env=env if env is not None else self._child_env(seam),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("canary pid %s timed out: %s" % (proc.pid, args[:4]))
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    def _seam_args(self, seam, gate):
        args = [
            "--seam", seam,
            "--gate", gate,
            "--text", "my number is %s" % self.payload,
            "--state", self.state,
        ]
        if seam in ("agent_lane", "jobs_start"):
            args.extend(["--workdir", self.workdir])
        if seam in ("grok_image", "dispatch_tool"):
            args.extend(["--image", self.image])
        return args

    def _show(self, code, out, err):
        return "exit %s\nstdout:\n%s\nstderr:\n%s" % (
            code, _redact(out, self.payload)[:2000], _redact(err, self.payload)[:2000])

    def _assert_no_fixture(self, out, err):
        if self.payload in out or self.payload in err:
            self.fail("fixture value appeared in seam stdout or stderr")

    def test_list_is_the_fixed_seam_order(self):
        code, out, err = self._invoke(["--list"])
        self.assertEqual(code, 0, self._show(code, out, err))
        self.assertEqual(tuple(out.splitlines()), SEAM_IDS)
        self.assertEqual(err, "")

    def test_where_points_at_this_checkout(self):
        code, out, err = self._invoke(["--where"])
        self.assertEqual(code, 0, self._show(code, out, err))
        lines = out.splitlines()
        self.assertEqual(len(lines), 2, lines)
        self.assertEqual(
            lines[0],
            os.path.realpath(os.path.join(ROOT, "providers", "egress_canary.py")),
        )
        self.assertEqual(lines[1], os.path.realpath(os.path.join(ROOT, "providers")))

    def test_unknown_seam_exits_2_with_usage(self):
        code, out, err = self._invoke([
            "--seam", "not-a-seam",
            "--gate", "1",
            "--text", "x",
            "--state", self.state,
        ])
        self.assertEqual(code, 2, self._show(code, out, err))
        self.assertIn("usage", (out + err).lower())

    def test_product_modules_do_not_import_canary(self):
        offenders = []
        providers = os.path.join(ROOT, "providers")
        for name in os.listdir(providers):
            if not name.endswith(".py") or name == "egress_canary.py":
                continue
            path = os.path.join(providers, name)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                if "egress_canary" in fh.read():
                    offenders.append(os.path.join("providers", name))
        for name in ("server.py", "omni_cli.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
                if "egress_canary" in fh.read():
                    offenders.append(name)
        self.assertEqual(offenders, [])

    def _assert_gate_on(self, seam):
        code, out, err = self._invoke(self._seam_args(seam, "1"), seam=seam)
        shown = self._show(code, out, err)
        self.assertEqual(code, 1, shown)
        lines = out.splitlines()
        self.assertTrue(lines, shown)
        self.assertTrue(lines[0].startswith(SIGNATURE), shown)
        self.assertIn("[tw_mobile]", lines[0], shown)
        if seam == "jobs_start":
            second = lines[1] if len(lines) > 1 else ""
            self.assertEqual(second, "jobs-refuse-no-prompt.txt", shown)
        self._assert_no_fixture(out, err)

    def _assert_gate_off(self, seam):
        code, out, err = self._invoke(self._seam_args(seam, "0"), seam=seam)
        shown = self._show(code, out, err)
        self._assert_no_fixture(out, err)
        if seam == "agy_sdk":
            fallback = (
                code == 4
                and SDK_FALLBACK in out
                and FAILOPEN not in out
            )
            reached = (
                code == 42
                and FAILOPEN in out
                and SDK_FALLBACK not in out
            )
            self.assertTrue(fallback or reached, shown)
            return
        self.assertEqual(code, 42, shown)
        self.assertIn(FAILOPEN, out, shown)

    def _listing(self, root):
        """Relative file paths under *root*, or None when *root* is absent."""
        if not root or not os.path.exists(root):
            return None
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in (".git", "__pycache__"))
            for name in sorted(filenames):
                if name == ".offline-hits.jsonl":
                    continue
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
        return tuple(found)

    def _creation_snapshot(self, cwd):
        """Cwd and the process home a rejected canary must leave untouched.

        Under the suite the process home is the synthetic home. An empty
        ``--state`` falls back through ``expanduser`` to that home, at
        ``~/.local/state/omni-llm``.
        """
        home = os.path.realpath(os.path.expanduser("~"))
        return (
            self._listing(cwd),
            self._listing(home),
            self._listing(self.state),
        )

    def test_empty_text_and_state_exit_2_without_writes(self):
        cwd = os.path.join(self.root, "cwd")
        os.makedirs(cwd)
        env = self._child_env("jobs_start")
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        cases = (
            ["--seam", "jobs_start", "--gate", "1", "--text", "x",
             "--state", "", "--workdir", self.workdir],
            ["--seam", "jobs_start", "--gate", "1", "--text", "",
             "--state", self.state, "--workdir", self.workdir],
            ["--seam", "jobs_start", "--gate", "1", "--text", "x",
             "--state", self.state, "--workdir", ""],
            ["--seam", "grok_image", "--gate", "1", "--text", "x",
             "--state", self.state, "--image", ""],
        )
        for args in cases:
            before = self._creation_snapshot(cwd)
            code, out, err = self._invoke(args, cwd=cwd, env=env)
            shown = self._show(code, out, err)
            self.assertEqual(code, 2, shown)
            self.assertIn("usage", (out + err).lower(), shown)
            self.assertEqual(self._creation_snapshot(cwd), before, shown)

    def test_agent_lane_missing_stub_target_exits_43(self):
        argv = [
            "providers.egress_canary",
            "--seam", "agent_lane",
            "--gate", "1",
            "--text", "my number is %s" % self.payload,
            "--state", self.state,
            "--workdir", self.workdir,
        ]
        code_text = (
            "import runpy, sys\n"
            "import providers.agent_lane as agent_lane\n"
            "del agent_lane.run_cli\n"
            "sys.argv = %r\n"
            "runpy.run_module('providers.egress_canary', run_name='__main__')\n"
        ) % (argv,)
        proc = subprocess.Popen(
            [sys.executable, "-c", code_text],
            cwd=ROOT,
            env=self._child_env("agent_lane"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out_b, err_b = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("missing-stub child timed out")
        out = out_b.decode("utf-8", "replace")
        err = err_b.decode("utf-8", "replace")
        shown = self._show(proc.returncode, out, err)
        self.assertEqual(proc.returncode, 43, shown)
        self.assertIn("STUB-TARGET-MISSING:run_cli", out.splitlines(), shown)
        self._assert_no_fixture(out, err)

    def test_cli_agy_missing_vendor_which_exits_43(self):
        # find_agy resolves the binary through vendor_bin.which, so the
        # canary stubs that name. If it is renamed the seam must stay loud
        # (exit 43) rather than reach a real agy lookup.
        argv = [
            "providers.egress_canary",
            "--seam", "cli_agy",
            "--gate", "1",
            "--text", "my number is %s" % self.payload,
            "--state", self.state,
            "--workdir", self.workdir,
        ]
        code_text = (
            "import runpy, sys\n"
            "import providers.vendor_bin as vendor_bin\n"
            "del vendor_bin.which\n"
            "sys.argv = %r\n"
            "runpy.run_module('providers.egress_canary', run_name='__main__')\n"
        ) % (argv,)
        proc = subprocess.Popen(
            [sys.executable, "-c", code_text],
            cwd=ROOT,
            env=self._child_env("cli_agy"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out_b, err_b = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("missing-which child timed out")
        out = out_b.decode("utf-8", "replace")
        err = err_b.decode("utf-8", "replace")
        shown = self._show(proc.returncode, out, err)
        self.assertEqual(proc.returncode, 43, shown)
        self.assertIn("STUB-TARGET-MISSING:vendor_bin.which", out.splitlines(), shown)
        self._assert_no_fixture(out, err)

    def test_agent_lane_missing_catalog_v2_exits_43(self):
        argv = [
            "providers.egress_canary",
            "--seam", "agent_lane",
            "--gate", "1",
            "--text", "my number is %s" % self.payload,
            "--state", self.state,
            "--workdir", self.workdir,
        ]
        code_text = (
            "import runpy, sys\n"
            "sys.modules['providers.catalog_v2'] = None\n"
            "sys.argv = %r\n"
            "runpy.run_module('providers.egress_canary', run_name='__main__')\n"
        ) % (argv,)
        proc = subprocess.Popen(
            [sys.executable, "-c", code_text],
            cwd=ROOT,
            env=self._child_env("agent_lane"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out_b, err_b = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("missing-catalog child timed out")
        out = out_b.decode("utf-8", "replace")
        err = err_b.decode("utf-8", "replace")
        shown = self._show(proc.returncode, out, err)
        self.assertEqual(proc.returncode, 43, shown)
        self.assertIn("STUB-TARGET-MISSING:catalog_v2", out.splitlines(), shown)
        self._assert_no_fixture(out, err)

    def test_agent_lane_catalog_seed_error_exits_43(self):
        # Detail is the patched message. code/reason stay a short token.
        marker = "seed-boom-do-not-print"
        argv = [
            "providers.egress_canary",
            "--seam", "agent_lane",
            "--gate", "1",
            "--text", "my number is %s" % self.payload,
            "--state", self.state,
            "--workdir", self.workdir,
        ]
        code_text = (
            "import runpy, sys\n"
            "import providers.catalog_v2 as catalog_v2\n"
            "from providers.catalog_record import StrictRecordError\n"
            "def _boom(*_a, **_k):\n"
            "    exc = StrictRecordError('catalog_unreadable', %r)\n"
            "    exc.code = 'catalog_unreadable'\n"
            "    raise exc\n"
            "catalog_v2.save = _boom\n"
            "sys.argv = %r\n"
            "runpy.run_module('providers.egress_canary', run_name='__main__')\n"
        ) % (marker, argv)
        proc = subprocess.Popen(
            [sys.executable, "-c", code_text],
            cwd=ROOT,
            env=self._child_env("agent_lane"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out_b, err_b = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("catalog-seed child timed out")
        out = out_b.decode("utf-8", "replace")
        err = err_b.decode("utf-8", "replace")
        shown = self._show(proc.returncode, out, err)
        self.assertEqual(proc.returncode, 43, shown)
        seed_lines = [
            line for line in out.splitlines()
            if line.startswith("STUB-TARGET-MISSING:catalog_seed:")
        ]
        self.assertEqual(seed_lines, [
            "STUB-TARGET-MISSING:catalog_seed:StrictRecordError:catalog_unreadable",
        ], shown)
        self.assertNotIn("OMNI-EGRESS-DENY", out, shown)
        self.assertNotIn(marker, out, shown)
        self.assertNotIn(marker, err, shown)
        self.assertNotIn("Traceback", err, shown)
        self._assert_no_fixture(out, err)

    def test_jobs_start_nonempty_jobs_dir_exits_2_keeps_file(self):
        job_dir = os.path.join(self.state, "jobs", "preexisting")
        os.makedirs(job_dir)
        prompt_path = os.path.join(job_dir, "prompt.txt")
        marker = "preexisting marker\n"
        with open(prompt_path, "w", encoding="utf-8") as fh:
            fh.write(marker)
        before = os.stat(prompt_path)
        code, out, err = self._invoke(
            self._seam_args("jobs_start", "1"), seam="jobs_start")
        shown = self._show(code, out, err)
        self.assertEqual(code, 2, shown)
        self.assertTrue(out.splitlines(), shown)
        self.assertTrue(out.splitlines()[0].startswith("STATE-NOT-CLEAN "), shown)
        self.assertNotIn("jobs-refuse-leaked-prompt.txt", out + err, shown)
        self.assertNotIn(SIGNATURE, out + err, shown)
        self._assert_no_fixture(out, err)
        after = os.stat(prompt_path)
        self.assertEqual(after.st_ino, before.st_ino, shown)
        with open(prompt_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), marker, shown)

    def _files_containing(self, roots, needle, started, skip_prefix):
        """Regular files newer than *started* whose bytes contain *needle*."""
        hits = []
        skip = os.path.realpath(skip_prefix) if skip_prefix else None
        needle_b = needle.encode("utf-8")
        seen = set()
        for root in roots:
            if not root or not os.path.exists(root):
                continue
            root = os.path.realpath(root)
            if root in seen:
                continue
            seen.add(root)
            for dirpath, dirnames, filenames in os.walk(root):
                real_dir = os.path.realpath(dirpath)
                if skip and (real_dir == skip or real_dir.startswith(skip + os.sep)):
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    try:
                        st = os.lstat(path)
                    except OSError:
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if st.st_mtime < started - 1:
                        continue
                    if st.st_size > 1024 * 1024:
                        continue
                    try:
                        with open(path, "rb") as fh:
                            blob = fh.read()
                    except OSError:
                        continue
                    if needle_b in blob:
                        hits.append(path)
        return hits

    def test_cli_grok_gate_0_without_tmpdir_keeps_payload_in_state(self):
        env = self._child_env("cli_grok")
        env.pop("TMPDIR", None)
        started = time.time()
        code, out, err = self._invoke(
            self._seam_args("cli_grok", "0"), seam="cli_grok", env=env)
        shown = self._show(code, out, err)
        self.assertEqual(code, 42, shown)
        self.assertIn(FAILOPEN, out, shown)
        self._assert_no_fixture(out, err)
        self.assertTrue(os.path.isdir(os.path.join(self.state, "tmp")), shown)
        needle = "my number is %s" % self.payload
        outside_roots = [
            self.root,
            ROOT,
            os.path.realpath(os.path.expanduser("~")),
            "/tmp",
            "/var/tmp",
            "/usr/tmp",
        ]
        outside = self._files_containing(outside_roots, needle, started, self.state)
        self.assertEqual(outside, [], shown)
        inside = self._files_containing([self.state], needle, started, None)
        self.assertTrue(inside, shown)

    def _catalog_load(self, seam, provider, profile):
        """Load with the state dir and the env the canary child was given.

        agy seams set ``AGY_CONFIG_DIR`` to the state dir before seeding, and
        the fingerprint includes that path. Other seams leave it alone.
        """
        from providers import catalog_v2

        previous = {}
        for key in ("OMNI_LLM_STATE_DIR", "AGY_CONFIG_DIR"):
            previous[key] = os.environ.get(key)
        # The canary realpaths --state before arming. The fingerprint and
        # the catalog directory both follow that spelling.
        state = os.path.realpath(self.state)
        os.environ["OMNI_LLM_STATE_DIR"] = state
        if seam in ("cli_agy", "agy_sdk"):
            os.environ["AGY_CONFIG_DIR"] = state
        try:
            return catalog_v2.load(provider, profile)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _admit(self, provider, model_id, effort, record, now):
        from providers.admission import decide

        return decide(
            {"provider": provider, "model": model_id, "effort": effort},
            record, now=now)

    def test_gate_on_seeds_admit_catalog_requests(self):
        from providers.egress_canary import _catalog_seeds

        seeds = _catalog_seeds()
        unseeded = tuple(seam for seam in SEAM_IDS if seam not in seeds)
        self.assertEqual(unseeded, ("xai_image", "cli_pi", "openai_compat"))
        now = datetime.datetime.now(datetime.timezone.utc)
        base_state = self.state
        try:
            for seam, specs in seeds.items():
                self.state = os.path.join(self.root, "seed-%s" % seam)
                os.makedirs(self.state)
                self._assert_gate_on(seam)
                for provider, _profile, models in specs:
                    record, reason = self._catalog_load(seam, provider, _profile)
                    shown = "seam %s %s/%s reason %r" % (
                        seam, provider, _profile, reason)
                    self.assertIsNotNone(record, shown)
                    self.assertIsNone(reason, shown)
                    self.assertEqual(
                        [row["id"] for row in record["models"]],
                        [model_id for model_id, _efforts in models],
                        shown)
                    for model_id, efforts in models:
                        row = next(
                            item for item in record["models"]
                            if item["id"] == model_id)
                        if efforts:
                            self.assertEqual(
                                row["supported_efforts"], list(efforts), shown)
                        else:
                            self.assertNotIn("supported_efforts", row, shown)
                        for effort in (None,) + tuple(efforts):
                            result = self._admit(
                                provider, model_id, effort, record, now)
                            self.assertIsNone(
                                result["refusal"],
                                "%s model %s effort %r -> %s" % (
                                    shown, model_id, effort, result["refusal"]))
        finally:
            self.state = base_state

    def test_removing_one_seed_id_fails_admission(self):
        from providers.egress_canary import _catalog_seeds, _synthetic_record

        provider, _profile, models = _catalog_seeds()["dispatch_tool"][0]
        dropped_id, dropped_efforts = models[0]
        kept = tuple(row for row in models if row[0] != dropped_id)
        self.assertTrue(kept)
        observed_at = datetime.datetime.now(
            datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        now = datetime.datetime.now(datetime.timezone.utc)
        full = _synthetic_record(provider, models, observed_at)
        reduced = _synthetic_record(provider, kept, observed_at)
        for effort in (None,) + tuple(dropped_efforts):
            admitted = self._admit(provider, dropped_id, effort, full, now)
            self.assertIsNone(admitted["refusal"], admitted)
            refused = self._admit(provider, dropped_id, effort, reduced, now)
            self.assertEqual(refused["refusal"], "unsupported_model", refused)

    def _child_main_after_config(self, seam, config_name, value):
        """Run the seam on an already-imported canary after one config change.

        The module import happens before the assignment, so a table frozen
        at import still lists the old id. The child uses this test's state
        dir and env, including the synthetic home.
        """
        argv = ["providers.egress_canary", *self._seam_args(seam, "1")]
        code_text = (
            "import sys\n"
            "import providers.egress_canary as canary\n"
            "import providers.config as config\n"
            "config.%s = %r\n"
            "sys.argv = %r\n"
            "canary.main()\n"
        ) % (config_name, value, argv)
        proc = subprocess.Popen(
            [sys.executable, "-c", code_text],
            cwd=ROOT,
            env=self._child_env(seam),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out_b, err_b = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("%s config child timed out" % seam)
        out = out_b.decode("utf-8", "replace")
        err = err_b.decode("utf-8", "replace")
        shown = self._show(proc.returncode, out, err)
        self.assertEqual(proc.returncode, 1, shown)
        lines = out.splitlines()
        self.assertTrue(lines, shown)
        self.assertTrue(lines[0].startswith(SIGNATURE), shown)
        if seam == "jobs_start":
            second = lines[1] if len(lines) > 1 else ""
            self.assertEqual(second, "jobs-refuse-no-prompt.txt", shown)
        self._assert_no_fixture(out, err)
        return shown

    def _assert_live_model_seed(self, seam, config_name, model_id):
        shown = self._child_main_after_config(seam, config_name, model_id)
        record, reason = self._catalog_load(seam, "grok", "default")
        self.assertIsNotNone(record, "%s reason %r" % (shown, reason))
        self.assertIsNone(reason, shown)
        self.assertEqual(
            [row["id"] for row in record["models"]], [model_id], shown)
        now = datetime.datetime.now(datetime.timezone.utc)
        result = self._admit("grok", model_id, None, record, now)
        self.assertIsNone(
            result["refusal"], "%s %s" % (shown, result["refusal"]))

    def test_jobs_start_seed_follows_live_agent_model(self):
        self._assert_live_model_seed(
            "jobs_start", "DEFAULT_AGENT_MODEL", "grok-9.9")

    def test_cli_grok_seed_follows_live_grok_model(self):
        self._assert_live_model_seed(
            "cli_grok", "DEFAULT_GROK_MODEL", "grok-9.9")

    def test_codex_default_reasoning_effort_is_admitted(self):
        from providers import config
        from providers.egress_canary import _catalog_seeds

        provider, profile, models = _catalog_seeds()["cli_codex"][0]
        model_id, efforts = models[0]
        self.assertIn(config.DEFAULT_CODEX_REASONING_EFFORT, efforts)
        self.assertIn("medium", efforts)
        self._assert_gate_on("cli_codex")
        record, reason = self._catalog_load("cli_codex", provider, profile)
        self.assertIsNotNone(record, reason)
        self.assertIsNone(reason, reason)
        row = next(item for item in record["models"] if item["id"] == model_id)
        self.assertIn(
            config.DEFAULT_CODEX_REASONING_EFFORT, row["supported_efforts"])
        now = datetime.datetime.now(datetime.timezone.utc)
        result = self._admit(
            provider, model_id, config.DEFAULT_CODEX_REASONING_EFFORT,
            record, now)
        self.assertIsNone(result["refusal"], result)


def _bind_seam_tests():
    def _make(seam, gate):
        def _test(self):
            if gate == "1":
                self._assert_gate_on(seam)
            else:
                self._assert_gate_off(seam)
        _test.__name__ = "test_%s_gate_%s" % (seam, gate)
        _test.__doc__ = "%s gate %s" % (seam, gate)
        return _test

    for seam in SEAM_IDS:
        setattr(EgressCanaryTests, "test_%s_gate_1" % seam, _make(seam, "1"))
        setattr(EgressCanaryTests, "test_%s_gate_0" % seam, _make(seam, "0"))


_bind_seam_tests()


if __name__ == "__main__":
    unittest.main()
