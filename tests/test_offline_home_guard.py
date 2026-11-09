"""The offline suite must not touch the real account home (issue #185).

The recorder and the synthetic home live in tests/offline_home.py. These
cases check the mechanism in throwaway interpreters whose "real home" is a
synthetic directory, so nothing here reads the operator's actual home.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests import offline_home

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs in a fresh interpreter: arm the guard against a synthetic "real" home,
# then do the things the guard must see (and some it must not).
_PARENT = textwrap.dedent("""
    import json, os, subprocess, sys, tempfile
    ROOT = @ROOT@
    sys.path.insert(0, ROOT)
    spec = json.loads(sys.argv[1])
    real = spec["real"]
    # Variables an operator may have exported: they must not survive install().
    os.environ["CODEX_HOME"] = os.path.join(real, ".codex")
    os.environ["GROK_HOME"] = os.path.join(real, ".grok")
    os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(real, ".claude")
    os.environ["CLAUDE_CONFIG_DIR_WORK"] = os.path.join(real, ".claude-work")
    os.environ["OMNI_LLM_STATE_DIR"] = os.path.join(real, ".local", "state")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(real, ".config")
    seeded = []
    for prefix in spec["prefixes"]:
        os.environ[prefix + "WORK"] = os.path.join(real, ".codex-x")
        seeded.append(prefix + "WORK")
    for name in spec["names"]:
        os.environ[name] = os.path.join(real, ".codex-x")
        seeded.append(name)
    from tests import offline_home
    from providers import kernel_sandbox

    fake = offline_home.install(real=real)
    # Straight after install, before any provider call.
    seeded_left = [k for k in seeded if k in os.environ]
    offline_home.set_current("synthetic.case")
    scrubbed = ["CODEX_HOME", "GROK_HOME", "CLAUDE_CONFIG_DIR",
                "CLAUDE_CONFIG_DIR_WORK", "OMNI_LLM_STATE_DIR"]
    out = {"fake": fake, "protected_home": kernel_sandbox.protected_home(),
           "home_env": os.environ["HOME"],
           "original_home": offline_home.original_protected_home(),
           "env": {k: os.environ.get(k) for k in scrubbed},
           "xdg": os.environ.get("XDG_CONFIG_HOME"),
           "seeded": seeded, "seeded_left": seeded_left}
    # 1. an admission under the stand-in real home
    kernel_sandbox.admit_path(os.path.join(real, ".claude.json"), "literal",
                              label="agent fence read")
    # 2. an admission under the synthetic home (must NOT be recorded)
    kernel_sandbox.admit_path(os.path.join(fake, ".claude.json"), "literal",
                              label="agent fence read")
    # 3. a direct open of a credential file under the stand-in real home
    open(os.path.join(real, ".codex", "auth.json")).close()
    # 3b. a directory listing of a provider state dir (the agy layout guard)
    os.listdir(os.path.join(real, ".gemini"))
    # 3c. a sibling account home named by prefix, and a look-alike that is not
    open(os.path.join(real, ".codex-x", "auth.json")).close()
    os.listdir(os.path.join(real, ".claude-work"))
    os.listdir(os.path.join(real, "unrelated"))
    # 3d. other spellings of the same directory
    for tag in ("case", "data"):
        base = spec.get(tag)
        if base:
            open(os.path.join(base, ".codex", "auth.json")).close()
    # 3e. links outside the home that lead to credentials inside it
    outside = tempfile.mkdtemp(prefix="omni_guard_out_")
    os.symlink(os.path.join(real, ".codex", "auth.json"),
               os.path.join(outside, "file_link"))
    os.symlink(os.path.join(real, ".gemini"), os.path.join(outside, "dir_link"))
    offline_home.set_current("symlink.file")
    open(os.path.join(outside, "file_link")).close()
    offline_home.set_current("symlink.dir")
    os.listdir(os.path.join(outside, "dir_link"))
    # a directory link that is harmless first and retargeted at the home later
    harmless = tempfile.mkdtemp(prefix="omni_guard_ok_")
    os.makedirs(os.path.join(harmless, ".codex"))
    open(os.path.join(harmless, ".codex", "auth.json"), "w").close()
    moving = os.path.join(outside, "moving")
    os.symlink(harmless, moving)
    offline_home.set_current("symlink.before")
    open(os.path.join(moving, ".codex", "auth.json")).close()
    os.remove(moving)
    os.symlink(real, moving)
    offline_home.set_current("symlink.after")
    open(os.path.join(moving, ".codex", "auth.json")).close()
    # 3f. link/../x: the link is expanded BEFORE the "..", so this is
    #     <home>/.codex/../.ssh/id_rsa, not <outside>/.ssh/id_rsa
    dotlink = os.path.join(outside, "dotlink")
    os.symlink(os.path.join(real, ".codex"), dotlink)
    tricky = os.path.join(dotlink, "..", ".ssh", "id_rsa")
    offline_home.set_current("dotdot.audit")
    open(tricky).close()
    offline_home.set_current("dotdot.record")
    out["dotdot_record"] = offline_home.record("probe", tricky)
    offline_home.set_current("dotdot.scrub")
    scrub_env = {"CODEX_HOME": tricky, "GROK_HOME": os.path.join(outside, "x")}
    offline_home._scrub_env(scrub_env, fake, only_real=True)
    out["dotdot_scrub_left"] = sorted(k for k in scrub_env if k != "XDG_CONFIG_HOME"
                                      and not k.startswith("XDG_"))
    # 3f2. the classification follows the RESOLVED path, not the first literal
    #      ancestor that happens to be the home. <home>/notes is a link to a key.
    os.symlink(os.path.join(".ssh", "id_rsa"), os.path.join(real, "notes"))
    tohome = os.path.join(outside, "tohome")
    os.symlink(real, tohome)
    shapes = {"resolved.dotdot": os.path.join(dotlink, "..", "notes"),
              "resolved.direct": os.path.join(real, "notes"),
              "resolved.viahome": os.path.join(tohome, "notes")}
    out["resolved_record"] = {}
    for tag, spelled in shapes.items():
        offline_home.set_current(tag)
        open(spelled).close()
        out["resolved_record"][tag] = offline_home.record("probe", spelled)
    # a link that IS the home on the way in, then ".." out of it: the resolved
    # path is outside the home, so it is not a real-home access
    with open(os.path.join(os.path.dirname(real), "outside_file"), "w") as fh:
        fh.write("x\\n")
    offline_home.set_current("resolved.leaves")
    leaves = os.path.join(tohome, "..", "outside_file")
    out["leaves_record"] = offline_home.record("probe_leaves", leaves)
    kernel_sandbox.admit_path(leaves, "literal", label="leaves the home")
    os.remove(os.path.join(os.path.dirname(real), "outside_file"))
    # resolution fails (a link loop inside the home): a literal ancestor that
    # is the home still counts as under the home
    os.symlink(".codex-loop", os.path.join(real, ".codex-loop"))
    offline_home.set_current("resolved.loop")
    out["loop_record"] = offline_home.record(
        "probe_loop", os.path.join(tohome, ".codex-loop", "x"))
    # 3g. nothing is cached: an alias seen as the home, then retargeted
    alias = os.path.join(outside, "alias")
    os.symlink(real, alias)
    offline_home.set_current("cache.before")
    open(os.path.join(alias, "unrelated", "note.txt")).close()  # not sensitive
    os.remove(alias)
    os.symlink(os.path.join(real, ".ssh"), alias)
    offline_home.set_current("cache.after")
    open(os.path.join(alias, "id_rsa")).close()
    offline_home.set_current("synthetic.case")
    # 4. a Python child sees the synthetic home too, and its environment
    keys = ["CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_DATA_HOME"]
    code = ("import os, json\\n"
            "from providers import kernel_sandbox as k\\n"
            "print(json.dumps([k.protected_home(), "
            "[os.environ.get(x) for x in %r]]))\\n"
            "k.admit_path(os.path.join(os.environ['OMNI_OFFLINE_REAL_HOME'],"
            " '.claude.json'), 'literal', label='c')\\n" % (keys,))
    keep = tempfile.mkdtemp(prefix="omni_guard_keep_")
    env = dict(os.environ)
    env["CODEX_HOME"] = os.path.join(real, ".codex")   # into the real home
    env["CLAUDE_CONFIG_DIR"] = keep                    # a test's own temp dir
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    first = child.stdout.strip().splitlines()[0] if child.stdout.strip() else "[]"
    out["child_home"], out["child_env"] = json.loads(first)
    out["child_keep"] = keep
    out["child_rc"] = child.returncode
    # 5. a child that cannot arm the guard (no way to import the helper)
    env2 = dict(os.environ)
    env2["PYTHONPATH"] = @SITE@
    broken = subprocess.run([sys.executable, "-c", "pass"],
                            cwd=tempfile.gettempdir(), env=env2,
                            capture_output=True, text=True)
    out["broken_rc"] = broken.returncode
    out["violations"] = offline_home.violations()
    offline_home.uninstall()
    print(json.dumps(out))
""")

# Runs the real runner entry point with a stub test run and a removal that
# silently fails, to prove a left-behind synthetic home fails the runner.
_CLEANUP = textwrap.dedent("""
    import json, os, shutil, sys
    ROOT = @ROOT@
    sys.path.insert(0, ROOT)
    from tests import run_offline_tests as r, offline_home
    kept = {}
    real_rmtree = shutil.rmtree
    def refusing(path, *a, **k):
        kept["home"] = path   # leave it in place
    shutil.rmtree = refusing
    rc = r.main(run=lambda: 0)
    shutil.rmtree = real_rmtree
    leftover = kept.get("home")
    exists = bool(leftover and os.path.exists(leftover))
    if leftover:
        real_rmtree(leftover, ignore_errors=True)
    print(json.dumps({"rc": rc, "home_was_left": exists}))
""")

# The same entry point with a normal removal: must return the run's status.
_CLEAN = _CLEANUP.replace("shutil.rmtree = refusing\n", "")


def _clean_env():
    env = dict(os.environ)
    for key in (offline_home.ENV_HOME, offline_home.ENV_REAL,
                offline_home.ENV_LOG, offline_home.ENV_CURRENT,
                "PYTHONPATH", "OMNI_OFFLINE_RUNNER"):
        env.pop(key, None)
    return env


def _script(template):
    return (template.replace("@ROOT@", repr(ROOT))
            .replace("@SITE@", repr(offline_home.SITE_DIR)))


class TestRecorderMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._real = tempfile.TemporaryDirectory(prefix="omni_guard_real_")
        real = os.path.realpath(cls._real.name)
        for sub in (".codex", ".gemini", ".codex-x", ".claude-work", "unrelated",
                    ".ssh"):
            os.makedirs(os.path.join(real, sub))
        for rel in (".claude.json", os.path.join(".codex", "auth.json"),
                    os.path.join(".codex-x", "auth.json"),
                    os.path.join(".ssh", "id_rsa"),
                    os.path.join("unrelated", "note.txt")):
            with open(os.path.join(real, rel), "w") as fh:
                fh.write("{}\n")
        cls.real = real
        spec = {"real": real}
        # Other spellings of the same directory; only where the host has them.
        case = os.path.join(os.path.dirname(real), os.path.basename(real).upper())
        if case != real and os.path.exists(case) and os.path.samefile(case, real):
            spec["case"] = case
        data = "/System/Volumes/Data" + real
        if os.path.exists(data) and os.path.samefile(data, real):
            spec["data"] = data
        from providers import kernel_sandbox, profile_check, profiles
        prefixes = {"CODEX_HOME_", "GROK_HOME_", "AGY_CONFIG_DIR_"}
        prefixes.update(p for _prov, p in kernel_sandbox._ACCOUNT_HOME_PREFIXES)
        for group in profiles._ROAD_SIBLING_PREFIXES.values():
            prefixes.update(group)
        for group in profile_check._ENV_PREFIXES.values():
            prefixes.update(group)
        keep = lambda n: not n.rstrip("_").endswith(("_KEY", "_BASE"))
        spec["prefixes"] = sorted(p for p in prefixes if keep(p))
        spec["names"] = sorted(n for n in profiles._DROP_EXACT if keep(n))
        cls.spec = spec
        proc = subprocess.run(
            [sys.executable, "-c", _script(_PARENT), json.dumps(spec)],
            cwd=ROOT, env=_clean_env(), capture_output=True, text=True,
            timeout=120)
        if proc.returncode != 0:
            raise AssertionError("probe interpreter failed: " + proc.stderr[-800:])
        cls.out = json.loads(proc.stdout.strip().splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        cls._real.cleanup()

    def hits(self):
        return [list(h) for h in self.out["violations"]]

    def test_protected_home_and_home_point_at_the_synthetic_home(self):
        out = self.out
        self.assertNotEqual(out["protected_home"], self.real)
        self.assertEqual(out["protected_home"], out["fake"])
        self.assertEqual(out["home_env"], out["fake"])

    def test_the_original_protected_home_stays_reachable(self):
        # Tests of the product function itself need it after install().
        self.assertEqual(self.out["original_home"], offline_home.real_home())
        self.assertNotEqual(self.out["original_home"], self.out["protected_home"])

    def test_admission_under_the_real_home_is_recorded(self):
        self.assertIn(["synthetic.case", "admit:agent fence read", ".claude.json"],
                      self.hits())

    def test_synthetic_home_admission_is_not_recorded(self):
        # Exactly one admit hit: the real-home one; the synthetic-home
        # admission of the same file name is not a violation.
        admits = [h for h in self.out["violations"]
                  if h[1] == "admit:agent fence read"]
        self.assertEqual(len(admits), 1, self.out["violations"])

    def test_credential_open_under_the_real_home_is_recorded(self):
        self.assertIn(["synthetic.case", "open", ".codex/auth.json"], self.hits())

    def test_provider_state_dir_listing_under_the_real_home_is_recorded(self):
        self.assertIn(["synthetic.case", "list", ".gemini"], self.hits())

    def test_sibling_account_homes_match_by_prefix(self):
        self.assertIn(["synthetic.case", "open", ".codex-x/auth.json"], self.hits())
        self.assertIn(["synthetic.case", "list", ".claude-work"], self.hits())

    def test_unrelated_entries_are_not_recorded(self):
        self.assertNotIn(["synthetic.case", "list", "unrelated"], self.hits())

    def test_case_variant_spelling_is_recorded(self):
        if "case" not in self.spec:
            self.skipTest("volume is case-sensitive or has no variant")
        rows = [h for h in self.out["violations"]
                if h[0] == "synthetic.case" and h[1] == "open"
                and h[2] == ".codex/auth.json"]
        self.assertEqual(len(rows), 2 + (1 if "data" in self.spec else 0),
                         self.out["violations"])

    def test_data_volume_spelling_is_recorded(self):
        if "data" not in self.spec:
            self.skipTest("no /System/Volumes/Data spelling on this host")
        rows = [h for h in self.out["violations"]
                if h[0] == "synthetic.case" and h[1] == "open"
                and h[2] == ".codex/auth.json"]
        self.assertEqual(len(rows), 2 + (1 if "case" in self.spec else 0),
                         self.out["violations"])

    def test_account_variables_do_not_leak_into_the_run(self):
        self.assertEqual(self.out["env"], {k: None for k in self.out["env"]})
        self.assertTrue(self.out["xdg"].startswith(self.out["fake"]),
                        self.out["xdg"])

    def test_sibling_account_variables_are_gone_before_any_provider_call(self):
        seeded, left = self.out["seeded"], self.out["seeded_left"]
        for needed in ("CODEX_HOME_WORK", "GROK_HOME_WORK",
                       "AGY_CONFIG_DIR_WORK", "CLAUDE_CONFIG_DIR_WORK",
                       "OMNI_LLM_CURSOR_CONFIG_DIR_WORK"):
            self.assertIn(needed, seeded)
        self.assertEqual(left, [])

    def test_the_scrub_list_covers_the_products_own_registries(self):
        names, prefixes = offline_home.product_scrub_registry()
        self.assertLessEqual(names, set(offline_home.SCRUB_ENV_NAMES))
        self.assertLessEqual(prefixes, set(offline_home.SCRUB_ENV_PREFIXES))
        for literal in ("CODEX_HOME_", "GROK_HOME_", "AGY_CONFIG_DIR_"):
            self.assertIn(literal, offline_home.SCRUB_ENV_PREFIXES)

    def test_a_link_outside_the_home_to_a_credential_is_recorded(self):
        self.assertIn(["symlink.file", "open", ".codex/auth.json"], self.hits())
        self.assertIn(["symlink.dir", "list", ".gemini"], self.hits())

    def test_a_directory_link_retargeted_at_the_home_is_caught_later(self):
        self.assertNotIn(["symlink.before", "open", ".codex/auth.json"],
                         self.hits())
        self.assertIn(["symlink.after", "open", ".codex/auth.json"], self.hits())

    def test_a_dotdot_after_a_link_is_resolved_the_way_the_kernel_does(self):
        # <outside>/dotlink/../.ssh/id_rsa names <home>/.ssh/id_rsa; a
        # normalizing check would see <outside>/.ssh/id_rsa and miss it.
        self.assertIn(["dotdot.audit", "open", ".ssh/id_rsa"], self.hits())
        self.assertIn(["dotdot.record", "probe", ".ssh/id_rsa"], self.hits())
        self.assertEqual(self.out["dotdot_record"], ".ssh/id_rsa")
        # the child scrub drops the variable that points into the real home
        # and keeps the one that does not
        self.assertEqual(self.out["dotdot_scrub_left"], ["GROK_HOME"])

    def test_the_resolved_path_decides_not_the_first_literal_home_ancestor(self):
        # <home>/notes -> .ssh/id_rsa, reached three ways; the literal remainder
        # is "notes" (not sensitive) in each, the resolved path is the key.
        for tag in ("resolved.dotdot", "resolved.direct", "resolved.viahome"):
            self.assertIn([tag, "open", ".ssh/id_rsa"], self.hits(), tag)
            self.assertEqual(self.out["resolved_record"][tag], ".ssh/id_rsa", tag)
            self.assertIn([tag, "probe", ".ssh/id_rsa"], self.hits(), tag)

    def test_leaving_the_home_through_a_link_is_not_a_home_access(self):
        # <outside>/tohome/../outside_file: tohome is the home, but ".." leaves it.
        self.assertIsNone(self.out["leaves_record"])
        self.assertEqual([h for h in self.hits() if h[0] == "resolved.leaves"], [])

    def test_a_failed_resolution_counts_as_under_the_home(self):
        self.assertEqual(self.out["loop_record"], ".codex-loop/x")
        self.assertIn(["resolved.loop", "probe_loop", ".codex-loop/x"], self.hits())

    def test_nothing_is_cached_so_a_retargeted_alias_is_caught(self):
        self.assertNotIn(["cache.before", "open", "unrelated/note.txt"],
                         self.hits())
        self.assertIn(["cache.after", "open", ".ssh/id_rsa"], self.hits())

    def test_children_drop_variables_that_point_into_the_real_home(self):
        code_home, kept = self.out["child_env"][0], self.out["child_env"][1]
        self.assertIsNone(code_home)                       # CODEX_HOME -> real
        self.assertEqual(kept, self.out["child_keep"])     # a test's own dir
        self.assertTrue(self.out["child_env"][2].startswith(self.out["fake"]))

    def test_python_children_get_the_synthetic_home_and_report_back(self):
        out = self.out
        self.assertEqual(out["child_rc"], 0)
        self.assertEqual(out["child_home"], out["fake"])
        child_rows = [h for h in out["violations"] if h[1].startswith("child:")]
        self.assertTrue(
            any(h[1] == "child:admit:c" and h[2] == ".claude.json"
                for h in child_rows), out["violations"])

    def test_a_child_that_cannot_arm_fails_the_run(self):
        self.assertEqual(self.out["broken_rc"], 0)  # it still ran
        rows = [h for h in self.out["violations"] if h[1] == "child:arm_failed"]
        self.assertEqual(len(rows), 1, self.out["violations"])
        self.assertEqual(rows[0][0], "synthetic.case")
        self.assertIn("No module named", rows[0][2])
        self.assertIn("could not arm", offline_home.format_violations(
            [tuple(r) for r in rows]))


class TestRunnerCleanup(unittest.TestCase):
    def run_probe(self, template):
        proc = subprocess.run(
            [sys.executable, "-c", _script(template)], cwd=ROOT,
            env=_clean_env(), capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr

    def test_a_synthetic_home_left_behind_makes_the_runner_fail(self):
        out, err = self.run_probe(_CLEANUP)
        self.assertTrue(out["home_was_left"])
        self.assertEqual(out["rc"], 1)
        self.assertIn("left behind", err)

    def test_clean_removal_keeps_the_run_status(self):
        out, _err = self.run_probe(_CLEAN)
        self.assertEqual(out["rc"], 0)
        self.assertFalse(out["home_was_left"])


class TestRunnerArmsTheGuard(unittest.TestCase):
    def test_runner_run_uses_the_synthetic_home(self):
        if not os.environ.get("OMNI_OFFLINE_RUNNER"):
            self.skipTest("only meaningful under tests/run_offline_tests.py")
        from providers import kernel_sandbox
        self.assertTrue(offline_home.active(),
                        "the runner started without installing the guard")
        self.assertNotEqual(kernel_sandbox.protected_home(),
                            offline_home.real_home())
        self.assertEqual(kernel_sandbox.protected_home(), offline_home.home())
        self.assertEqual(os.environ["HOME"], offline_home.home())
        self.assertFalse(
            os.path.realpath(offline_home.home()).startswith(
                offline_home.real_home() + os.sep))
        # A test may set its own temp path (OMNI_LLM_STATE_DIR in a module
        # fixture); what must be gone is anything inherited from the operator.
        for name in offline_home.SCRUB_ENV_NAMES:
            value = os.environ.get(name)
            if value:
                self.assertIsNone(offline_home._classify(value), name)


if __name__ == "__main__":
    unittest.main()
