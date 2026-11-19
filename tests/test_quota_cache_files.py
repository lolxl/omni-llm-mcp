# no-opencc
"""The Codex quota cache is one file per profile (#254).

Every case runs against a throwaway state dir. The Codex RPC and the vendor
binary are replaced; nothing here spawns a real vendor CLI or touches the network.
"""
import contextlib
import copy
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from providers import codex_quota, health, quota

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HASH_NAME = re.compile(r"^[0-9a-f]{64}\.json$")

# A second process that holds one lock file until its stdin closes.
HOLD_SCRIPT = (
    "import fcntl, sys\n"
    "handle = open(sys.argv[1], 'a+b')\n"
    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    "print('ready', flush=True)\n"
    "sys.stdin.read()\n")

# A second process that runs one forced read with the vendor replaced. A spawn
# would be a bug: the lock timeout has to come first.
READ_SCRIPT = r'''
import json, sys
from providers import codex_quota, vendor_bin
home, label = sys.argv[1], sys.argv[2]
vendor_bin.which = lambda name: "/usr/bin/true"
codex_quota.resolve_profile = lambda kind, name: ({"CODEX_HOME": home}, None)
def boom(*args, **kwargs):
    raise AssertionError("spawned")
codex_quota._spawn_rate_limits = boom
info, err = codex_quota.read_rate_limits(label, timeout_s=0.3, force=True)
print(json.dumps({"info": info, "err": err}))
'''


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


class CacheCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="omni-quota-files-")
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(os.path.realpath(self.tmp.name), "state")
        os.makedirs(self.state)
        self.home = os.path.join(os.path.realpath(self.tmp.name), "codex-home")
        os.makedirs(self.home)
        self.home_id = hashlib.sha256(self.home.encode("utf-8")).hexdigest()
        env = mock.patch.dict(os.environ, {
            "OMNI_LLM_STATE_DIR": self.state, "OMNI_LLM_QUOTA_PROBE": "1",
            "OMNI_LLM_NOTIFY_QUIET": "1", "CODEX_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        # An unreadable or malformed file is logged once per path per process.
        health._malformed_logged.clear()
        self.addCleanup(health._malformed_logged.clear)
        health._old_cache_logged.clear()
        self.addCleanup(health._old_cache_logged.clear)
        self.events = []
        log = mock.patch.object(
            health, "log_event",
            side_effect=lambda event, **fields: self.events.append((event, fields)))
        log.start()
        self.addCleanup(log.stop)

    # -- fixtures ---------------------------------------------------------
    @property
    def qdir(self):
        return os.path.join(self.state, "quota")

    def path_of(self, label):
        return os.path.join(self.qdir, digest(label) + ".json")

    def row(self, **over):
        row = {"used_percent": 12.0, "resets_at": None, "window_mins": 300,
               "reached": False, "primary_reached": False, "plan": "pro",
               "checked_at": time.time() - 5, "home_id": self.home_id}
        row.update(over)
        return row

    def put(self, label, **over):
        self.assertTrue(health.put_quota_row(label, self.row(**over)), label)

    def write_raw(self, label, text):
        os.makedirs(self.qdir, exist_ok=True)
        with open(self.path_of(label), "w", encoding="utf-8") as handle:
            handle.write(text)

    def snapshot_of(self, label):
        path = self.path_of(label)
        with open(path, "rb") as handle:
            return handle.read(), os.stat(path).st_mtime_ns

    def logged(self, operation):
        return [fields for event, fields in self.events
                if event == "health_error" and fields.get("operation") == operation]

    @contextlib.contextmanager
    def hold(self, name):
        """Another process holds ``<state>/<name>`` exclusively."""
        proc = subprocess.Popen(
            [sys.executable, "-c", HOLD_SCRIPT, os.path.join(self.state, name)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            yield
        finally:
            proc.stdin.close()
            proc.wait(timeout=10)
            proc.stdout.close()

    def forced_read_elsewhere(self, label):
        """A second process runs one forced read; return its parsed answer."""
        out = subprocess.run(
            [sys.executable, "-c", READ_SCRIPT, self.home, label], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
            env=dict(os.environ))
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        return json.loads(out.stdout.strip().splitlines()[-1])

    def probe_with(self, reply):
        """Replace the vendor; ``reply`` is called as the RPC and returns ``(info, err)``."""
        for target, value in (
                (codex_quota.vendor_bin, ("which", lambda name: "/usr/bin/true")),
                (codex_quota, ("resolve_profile",
                               lambda kind, name: ({"CODEX_HOME": self.home}, None))),
                (codex_quota, ("_spawn_rate_limits",
                               lambda profile, timeout_s, overrides=None: reply(profile)))):
            patcher = mock.patch.object(target, value[0], value[1])
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def no_rpc(profile):
        raise AssertionError("rpc")

    @staticmethod
    def info(used=20.0, **over):
        info = {"used_percent": used, "resets_at": None, "window_mins": 300,
                "reached": False, "primary_reached": False, "plan": "pro"}
        info.update(over)
        return info

    def decide(self, profile="default"):
        policy, revision = quota.load_policy()
        policy = copy.deepcopy(policy)
        policy["pools"]["openai-sub"]["profile"] = profile
        doc, code = quota.decide("openai", "default", "codex:gpt-6.2-sol",
                                 "gpt-6.2-sol", "long", "primary",
                                 policy=policy, revision=revision)
        return code, doc


class PerProfileFileTests(CacheCase):
    def test_a_put_writes_the_row_and_its_label_into_the_hashed_file(self):
        self.put("default")
        self.assertEqual(os.listdir(self.qdir), [digest("default") + ".json"])
        with open(self.path_of("default"), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored, dict(self.row(checked_at=stored["checked_at"]),
                                      profile="default"))
        self.assertEqual(stat.S_IMODE(os.stat(self.qdir).st_mode), 0o700)
        self.assertEqual(health.get_quota_row("default"), stored)

    def test_1_a_put_holds_no_ledger_lock_and_leaves_a_sibling_byte_for_byte(self):
        self.put("work")
        before = self.snapshot_of("work")
        with self.hold("health.lock"), \
                mock.patch.object(health, "HEALTH_LOCK_WAIT_S", 0.1):
            self.assertTrue(health.put_quota_row("default", self.row()))
            self.assertIsNotNone(health.get_quota_row("default"))
            self.assertIsNotNone(health.strict_quota_cache_row("default"))
            self.assertTrue(health.drop_quota_row("default"))
            self.assertTrue(health.put_quota_row("default", self.row()))
            info = codex_quota._cache_put("default", self.info(), self.home_id)
            self.assertEqual(info["used_percent"], 20.0)
        self.assertTrue(os.path.exists(self.path_of("default")))
        self.assertEqual(self.snapshot_of("work"), before)
        self.assertEqual(
            [event for event, _ in self.events if event == "health_skipped"], [])

    def test_no_cache_path_takes_the_ledger_lock(self):
        taken = []
        real = health.try_exclusive_lock

        def spy(path, timeout_s):
            taken.append(os.path.basename(path))
            return real(path, timeout_s)
        self.probe_with(lambda profile: (self.info(), None))
        with mock.patch.object(health, "try_exclusive_lock", spy):
            self.put("default")
            health.get_quota_row("default")
            health.strict_quota_cache_row("default")
            health.drop_quota_row("default")
            info, err = codex_quota.read_rate_limits("default", force=True)
            self.assertIsNone(err)
        self.assertNotIn("health.lock", taken)

    def test_a_drop_unlinks_only_its_own_file_and_a_missing_file_is_fine(self):
        self.put("default")
        self.put("work")
        before = self.snapshot_of("work")
        self.assertTrue(health.drop_quota_row("default"))
        self.assertFalse(os.path.exists(self.path_of("default")))
        self.assertTrue(health.drop_quota_row("default"))
        self.assertEqual(self.snapshot_of("work"), before)
        os.remove(self.path_of("work"))
        os.rmdir(self.qdir)
        self.assertTrue(health.drop_quota_row("default"))   # no directory at all

    def test_a_put_replaces_only_its_own_file(self):
        self.put("default")
        self.put("work")
        before = self.snapshot_of("work")
        self.put("default", used_percent=77.0)
        self.assertEqual(health.get_quota_row("default")["used_percent"], 77.0)
        self.assertEqual(self.snapshot_of("work"), before)

    def test_a_drop_logs_a_malformed_regular_file_once_before_it_unlinks_it(self):
        self.assertTrue(health.drop_quota_row("default"))      # nothing there, nothing to log
        self.assertEqual(self.logged("malformed_file"), [])
        for text in ("{", "[]", '{"profile": "other", "checked_at": 1}'):
            health._malformed_logged.clear()
            self.events.clear()
            self.write_raw("default", text)
            self.assertTrue(health.drop_quota_row("default"), text)
            self.assertFalse(os.path.lexists(self.path_of("default")), text)
            self.assertEqual(len(self.logged("malformed_file")), 1, text)
            self.assertTrue(health.drop_quota_row("default"))    # gone: no second line
            self.assertEqual(len(self.logged("malformed_file")), 1, text)
        self.events.clear()
        self.put("default")
        self.assertTrue(health.drop_quota_row("default"))        # a good file is not logged
        self.assertEqual(self.logged("malformed_file"), [])


class SiblingIsolationTests(CacheCase):
    def test_2_a_malformed_sibling_refuses_only_its_own_decision(self):
        self.put("default")
        self.write_raw("work", '{"profile": "work", "checked_at": "x"}')
        before = self.snapshot_of("work")
        seen = []
        real = health._strict_read_maybe

        def spy(path, reason, **kwargs):
            seen.append(os.path.basename(path))
            return real(path, reason, **kwargs)
        with mock.patch.object(codex_quota, "read_rate_limits",
                               side_effect=AssertionError("rpc")), \
                mock.patch.object(health, "_strict_read_maybe", spy):
            code, doc = self.decide("default")
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (0, "eligible", ["ok"]))
        self.assertIn(digest("default") + ".json", seen)     # the spy saw the read
        self.assertNotIn(digest("work") + ".json", seen)
        with mock.patch.object(codex_quota, "read_rate_limits",
                               side_effect=AssertionError("rpc before the read")):
            code, doc = self.decide("work")
        self.assertEqual((code, doc["decision"], doc["reasons"]),
                         (10, "ineligible", ["telemetry_malformed"]))
        self.assertEqual(self.snapshot_of("work"), before)

    def test_3_a_malformed_own_file_refuses_before_any_rpc_and_a_put_replaces_it_once(self):
        self.write_raw("default", "{")
        with mock.patch.object(codex_quota, "read_rate_limits",
                               side_effect=AssertionError("rpc")):
            code, doc = self.decide("default")
        self.assertEqual((code, doc["reasons"]), (10, ["telemetry_malformed"]))
        self.assertEqual(self.logged("malformed_file"), [])
        # a lenient read is just a miss, never an exception
        self.assertIsNone(health.get_quota_row("default"))
        self.assertIsNone(codex_quota._cache_get("default", self.home_id))
        self.probe_with(lambda profile: (self.info(used=31.0), None))
        info, err = codex_quota.read_rate_limits("default")
        self.assertIsNone(err)
        self.assertEqual(info["used_percent"], 31.0)
        self.assertEqual(len(self.logged("malformed_file")), 1)
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 31.0)
        self.put("default", used_percent=32.0)
        self.assertEqual(len(self.logged("malformed_file")), 1)

    def test_a_decision_on_the_replaced_file_is_back_to_normal(self):
        self.write_raw("default", "[]")
        self.assertEqual(self.decide("default")[1]["reasons"], ["telemetry_malformed"])
        self.put("default")
        self.assertEqual(self.decide("default")[0], 0)


class QuotaLockTests(CacheCase):
    def test_4_a_forced_read_drops_and_puts_under_one_quota_lock_hold(self):
        self.put("default", used_percent=1.0)
        self.put("work", used_percent=2.0)
        work_before = self.snapshot_of("work")
        during = {}

        def reply(profile):
            # The drop has happened, the put has not: the lock is held across both.
            during["own_file_present"] = os.path.exists(self.path_of("default"))
            during["second"] = self.forced_read_elsewhere("work")
            during["work_present"] = os.path.exists(self.path_of("work"))
            during["work_same"] = self.snapshot_of("work") == work_before
            during["own_after_second"] = os.path.exists(self.path_of("default"))
            return self.info(used=44.0), None
        self.probe_with(reply)
        real_put = health.put_quota_row

        def put_after_a_contender(profile, row):
            # The RPC has returned, the put has not: the lock must still be held.
            during["before_put"] = self.forced_read_elsewhere("work")
            during["before_put_own_file"] = os.path.exists(self.path_of("default"))
            return real_put(profile, row)
        with mock.patch.object(health, "put_quota_row", put_after_a_contender):
            info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(err)
        self.assertEqual(info["used_percent"], 44.0)
        self.assertFalse(during["own_file_present"])
        self.assertIsNone(during["second"]["info"])
        self.assertEqual(during["second"]["err"]["detail"], "quota_lock_timeout")
        self.assertIsNone(during["before_put"]["info"])
        self.assertEqual(during["before_put"]["err"]["detail"], "quota_lock_timeout")
        self.assertFalse(during["before_put_own_file"])
        self.assertTrue(during["work_present"])
        self.assertTrue(during["work_same"])
        self.assertFalse(during["own_after_second"])
        self.assertEqual(health.get_quota_row("default")["used_percent"], 44.0)
        self.assertEqual(self.snapshot_of("work"), work_before)

    def test_the_lock_is_free_again_after_the_read(self):
        self.probe_with(lambda profile: (self.info(), None))
        codex_quota.read_rate_limits("default", force=True)
        handle = health.try_exclusive_lock(
            os.path.join(self.state, health.QUOTA_LOCK_NAME), 0.5)
        self.assertIsNotNone(handle)
        health.release_lock(handle)


class LabelTests(CacheCase):
    def test_5_labels_map_to_distinct_files_inside_the_directory(self):
        labels = ("Work", "work", "../x", "a/b", "")
        for label in labels:
            self.put(label or "default")
        names = sorted(os.listdir(self.qdir))
        self.assertEqual(len(names), 5)
        self.assertTrue(all(HASH_NAME.match(name) for name in names), names)
        self.assertEqual(names, sorted(digest(label or "default") + ".json"
                                       for label in labels))
        # nothing landed outside quota/ (only the directory itself is new)
        self.assertEqual(sorted(os.listdir(self.state)), ["quota"])
        for label in labels:
            self.assertEqual(health.get_quota_row(label or "default")["profile"],
                             label or "default")

    def test_5_a_stored_label_that_differs_is_malformed(self):
        self.put("work")
        with open(self.path_of("work"), encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["profile"] = "Work"
        self.write_raw("work", json.dumps(stored))
        with self.assertRaises(health.StrictReadError) as cm:
            health.strict_quota_cache_row("work")
        self.assertEqual(cm.exception.reason, "telemetry_malformed")
        self.assertIsNone(health.get_quota_row("work"))
        self.put("work")
        self.assertEqual(health.strict_quota_cache_row("work")["profile"], "work")
        self.assertEqual(len(self.logged("malformed_file")), 1)

    def test_a_label_that_cannot_be_encoded_writes_nothing_and_never_raises(self):
        label = "bad\udc80"
        self.assertFalse(health.put_quota_row(label, self.row()))
        self.assertIsNone(health.get_quota_row(label))
        self.assertFalse(health.drop_quota_row(label))
        with self.assertRaises(health.StrictReadError):
            health.strict_quota_cache_row(label)
        self.assertEqual(os.listdir(self.state), [])


class LinkTests(CacheCase):
    def target_row(self, label="default"):
        path = os.path.join(self.tmp.name, "elsewhere.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dict(self.row(), profile=label), handle)
        with open(path, "rb") as handle:
            return path, handle.read()

    def test_6_a_symlinked_directory_refuses_the_read_and_takes_no_write(self):
        real = os.path.join(self.tmp.name, "real-quota")
        os.makedirs(real)
        planted = os.path.join(real, digest("default") + ".json")
        with open(planted, "w", encoding="utf-8") as handle:
            json.dump(dict(self.row(), profile="default"), handle)
        with open(planted, "rb") as handle:
            planted_bytes = handle.read()
        os.symlink(real, self.qdir)
        with self.assertRaises(health.StrictReadError) as cm:
            health.strict_quota_cache_row("default")
        self.assertEqual(cm.exception.reason, "telemetry_malformed")
        self.assertIsNone(health.get_quota_row("default"))
        self.assertFalse(health.put_quota_row("work", self.row()))
        self.assertFalse(health.put_quota_row("default", self.row(used_percent=99.0)))
        self.assertFalse(health.drop_quota_row("default"))
        self.assertEqual(os.listdir(real), [digest("default") + ".json"])
        with open(planted, "rb") as handle:
            self.assertEqual(handle.read(), planted_bytes)
        self.assertTrue(os.path.islink(self.qdir))
        self.assertGreaterEqual(len(self.logged("quota_cache")), 2)
        self.assertEqual(self.decide("default")[1]["reasons"], ["telemetry_malformed"])

    def test_6_a_quota_path_that_is_a_file_refuses_and_stays_a_file(self):
        with open(self.qdir, "w", encoding="utf-8") as handle:
            handle.write("x")
        with self.assertRaises(health.StrictReadError):
            health.strict_quota_cache_row("default")
        self.assertFalse(health.put_quota_row("default", self.row()))
        self.assertTrue(os.path.isfile(self.qdir))

    def test_6_a_symlinked_profile_file_refuses_the_read_and_takes_no_write(self):
        target, target_bytes = self.target_row("default")
        self.put("work")
        work_before = self.snapshot_of("work")
        os.symlink(target, self.path_of("default"))
        with self.assertRaises(health.StrictReadError) as cm:
            health.strict_quota_cache_row("default")
        self.assertEqual(cm.exception.reason, "telemetry_malformed")
        self.assertIsNone(health.get_quota_row("default"))
        self.assertFalse(health.put_quota_row("default", self.row(used_percent=99.0)))
        self.assertFalse(health.drop_quota_row("default"))
        self.assertTrue(os.path.islink(self.path_of("default")))
        with open(target, "rb") as handle:
            self.assertEqual(handle.read(), target_bytes)
        self.assertEqual(self.snapshot_of("work"), work_before)
        self.assertEqual(self.logged("malformed_file"), [])
        self.assertEqual(len(self.logged("quota_cache")), 2)
        self.assertEqual(sorted(os.listdir(self.qdir)),
                         sorted([digest("default") + ".json", digest("work") + ".json"]))

    def test_a_dangling_profile_symlink_is_refused_too(self):
        os.makedirs(self.qdir)
        os.symlink(os.path.join(self.tmp.name, "nowhere"), self.path_of("default"))
        with self.assertRaises(health.StrictReadError):
            health.strict_quota_cache_row("default")
        self.assertFalse(health.put_quota_row("default", self.row()))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "nowhere")))


class DirectorySwapTests(CacheCase):
    """``quota/`` is opened once; a swap for a link afterwards changes nothing."""

    def setUp(self):
        super().setUp()
        self.elsewhere = os.path.join(self.tmp.name, "elsewhere")
        os.makedirs(self.elsewhere)
        self.moved = os.path.join(self.state, "quota-moved")
        self.swapped = []

    def plant(self, label, **over):
        """A decoy row of this label in the directory the link will point at."""
        path = os.path.join(self.elsewhere, digest(label) + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dict(self.row(**over), profile=label), handle)
        return path, read_bytes(path)

    def swap(self):
        """Patch the seam so the first verified open is followed by a swap."""
        def hook():
            if self.swapped:
                return
            self.swapped.append(True)
            os.rename(self.qdir, self.moved)
            os.symlink(self.elsewhere, self.qdir)
        patcher = mock.patch.object(health, "_quota_dir_checked", hook)
        patcher.start()
        self.addCleanup(patcher.stop)

    def moved_files(self):
        return sorted(os.listdir(self.moved))

    def test_a_read_after_the_swap_still_reads_the_directory_it_opened(self):
        self.put("default", used_percent=5.0)
        _decoy, decoy_bytes = self.plant("default", used_percent=99.0)
        self.swap()
        row = health.strict_quota_cache_row("default")
        self.assertEqual(self.swapped, [True])
        self.assertTrue(os.path.islink(self.qdir))
        self.assertEqual(row["used_percent"], 5.0)
        self.assertEqual(read_bytes(os.path.join(self.elsewhere, digest("default") + ".json")),
                         decoy_bytes)

    def test_a_put_after_the_swap_lands_in_the_directory_it_opened(self):
        self.put("work")
        self.swap()
        self.assertTrue(health.put_quota_row("default", self.row(used_percent=7.0)))
        self.assertEqual(self.swapped, [True])
        self.assertTrue(os.path.islink(self.qdir))
        self.assertEqual(os.listdir(self.elsewhere), [])
        self.assertEqual(self.moved_files(),
                         sorted([digest("default") + ".json", digest("work") + ".json"]))
        with open(os.path.join(self.moved, digest("default") + ".json"),
                  encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["used_percent"], 7.0)

    def test_a_put_that_creates_the_directory_is_pinned_to_it_too(self):
        self.swap()
        self.assertTrue(health.put_quota_row("default", self.row()))
        self.assertEqual(os.listdir(self.elsewhere), [])
        self.assertEqual(self.moved_files(), [digest("default") + ".json"])

    def test_a_drop_after_the_swap_unlinks_in_the_directory_it_opened(self):
        self.put("default")
        decoy, decoy_bytes = self.plant("default")
        self.swap()
        self.assertTrue(health.drop_quota_row("default"))
        self.assertEqual(self.moved_files(), [])
        self.assertEqual(read_bytes(decoy), decoy_bytes)

    def test_a_directory_owned_by_another_user_is_refused(self):
        self.put("default")
        before = self.snapshot_of("default")
        with mock.patch.object(health.os, "geteuid", return_value=os.geteuid() + 1):
            with self.assertRaises(health.StrictReadError):
                health.strict_quota_cache_row("default")
            self.assertFalse(health.put_quota_row("default", self.row(used_percent=1.0)))
            self.assertFalse(health.drop_quota_row("default"))
        self.assertEqual(self.snapshot_of("default"), before)

    def test_no_temp_file_is_left_in_either_directory(self):
        self.put("work")
        self.swap()
        self.assertTrue(health.put_quota_row("default", self.row()))
        names = os.listdir(self.moved) + os.listdir(self.elsewhere)
        self.assertEqual([n for n in names if n.endswith(".tmp")], [])

    def test_a_row_is_written_through_the_descriptor_not_through_the_path(self):
        calls = []
        real_open, real_replace = os.open, os.replace

        def spy_open(path, flags, *args, **kwargs):
            calls.append(("open", os.path.basename(str(path)), kwargs.get("dir_fd")))
            return real_open(path, flags, *args, **kwargs)

        def spy_replace(src, dst, **kwargs):
            calls.append(("replace", dst, kwargs.get("dst_dir_fd")))
            return real_replace(src, dst, **kwargs)
        with mock.patch.object(health.os, "open", spy_open), \
                mock.patch.object(health.os, "replace", spy_replace):
            self.assertTrue(health.put_quota_row("default", self.row()))
        replaces = [call for call in calls if call[0] == "replace"]
        self.assertEqual(len(replaces), 1)
        self.assertIsNotNone(replaces[0][2])
        temps = [call for call in calls
                 if call[0] == "open" and call[1].startswith(".omni-")]
        self.assertEqual(len(temps), 1)
        self.assertIsNotNone(temps[0][2])



class OldFileTests(CacheCase):
    """The one-file ``quota.json`` is never read; the first put deletes it."""

    @property
    def legacy(self):
        return os.path.join(self.state, "quota.json")

    def write_legacy(self, value, raw=False):
        with open(self.legacy, "w", encoding="utf-8") as handle:
            handle.write(value if raw else json.dumps(value))

    def opened_names(self):
        """Patch the strict reader to record the file names it is asked for."""
        opened = []
        real = health._strict_read_maybe

        def spy(path, reason, **kwargs):
            opened.append(os.path.basename(path))
            return real(path, reason, **kwargs)
        patcher = mock.patch.object(health, "_strict_read_maybe", spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return opened

    def test_a_valid_fresh_old_row_is_never_used_by_the_cache_or_the_decision(self):
        self.write_legacy({"default": self.row(used_percent=9.0)})
        before = read_bytes(self.legacy)
        opened = self.opened_names()
        self.assertIsNone(health.get_quota_row("default"))
        self.assertIsNone(health.strict_quota_cache_row("default"))
        self.assertIsNone(codex_quota._cache_get("default", self.home_id))
        self.assertEqual(read_bytes(self.legacy), before)     # reads leave it alone
        calls = []

        def reply(profile):
            calls.append(profile)
            return self.info(used=77.0), None
        self.probe_with(reply)
        code, doc = self.decide("default")
        self.assertEqual(calls, ["default"])                  # the probe ran
        self.assertEqual((code, doc["decision"], doc["used_bp"]), (0, "eligible", 7700))
        self.assertNotIn("quota.json", opened)

    def test_a_malformed_old_file_refuses_nothing(self):
        for text in ("{", "[]", "5", "", '{"default": {"checked_at": 1, "checked_at": 2}}',
                     '{"default": {"checked_at": NaN}}', '{"default": {"checked_at": "x"}}'):
            self.write_legacy(text, raw=True)
            self.assertIsNone(health.strict_quota_cache_row("default"), text)
            self.assertIsNone(health.get_quota_row("default"), text)
            self.write_raw("default", json.dumps(dict(self.row(), profile="default")))
            with mock.patch.object(codex_quota, "read_rate_limits",
                                   side_effect=AssertionError("rpc")):
                code, doc = self.decide("default")
            self.assertEqual((code, doc["reasons"]), (0, ["ok"]), text)
            os.remove(self.path_of("default"))
        self.assertEqual(self.logged("malformed_file"), [])

    def test_the_old_file_is_gone_after_the_first_successful_put_only(self):
        self.write_legacy({"default": self.row()})
        self.assertFalse(health.put_quota_row("default", self.row(used_percent=float("nan"))))
        self.assertTrue(os.path.lexists(self.legacy))         # nothing was published
        self.assertTrue(health.drop_quota_row("default"))
        self.assertTrue(os.path.lexists(self.legacy))         # a drop does not touch it
        unlinks = []
        real = os.unlink

        def spy(path, *args, **kwargs):
            unlinks.append((path, kwargs.get("dir_fd")))
            return real(path, *args, **kwargs)
        with mock.patch.object(health.os, "unlink", spy):
            self.assertTrue(health.put_quota_row("work", self.row()))
        self.assertFalse(os.path.lexists(self.legacy))
        self.assertEqual([call for call in unlinks if call[0] == "quota.json"],
                         [("quota.json", mock.ANY)])
        self.assertTrue(all(fd is not None for path, fd in unlinks if path == "quota.json"))
        self.assertTrue(health.put_quota_row("default", self.row()))   # already gone: fine
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_write_failed"}])

    def test_a_symlinked_old_file_is_removed_and_not_followed(self):
        target = os.path.join(self.tmp.name, "target.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"default": self.row(used_percent=9.0)}, handle)
        before = read_bytes(target)
        os.symlink(target, self.legacy)
        opened = self.opened_names()
        self.assertIsNone(health.get_quota_row("default"))
        self.assertIsNone(health.strict_quota_cache_row("default"))
        self.assertTrue(health.put_quota_row("default", self.row(used_percent=1.0)))
        self.assertFalse(os.path.lexists(self.legacy))
        self.assertEqual(read_bytes(target), before)
        self.assertNotIn("quota.json", opened)
        self.assertEqual(health.get_quota_row("default")["used_percent"], 1.0)

    def test_a_dangling_or_directory_old_name_never_fails_a_put(self):
        os.symlink(os.path.join(self.tmp.name, "nowhere"), self.legacy)
        self.assertTrue(health.put_quota_row("default", self.row()))
        self.assertFalse(os.path.lexists(self.legacy))
        os.makedirs(self.legacy)
        self.assertTrue(health.put_quota_row("default", self.row()))
        self.assertTrue(os.path.isdir(self.legacy))
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_old_file_not_removed"}])

    def test_a_failed_removal_is_logged_once_and_ignored(self):
        self.write_legacy({"default": self.row(used_percent=9.0)})
        real = os.unlink

        def unlink(path, *args, **kwargs):
            if path == "quota.json":
                raise PermissionError(errno.EACCES, "denied")
            return real(path, *args, **kwargs)
        with mock.patch.object(health.os, "unlink", unlink):
            self.assertTrue(health.put_quota_row("default", self.row(used_percent=1.0)))
            self.assertTrue(health.put_quota_row("work", self.row()))
        self.assertTrue(os.path.lexists(self.legacy))
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_old_file_not_removed"}])
        # still never read, even though it survives: a refresh drops the row and
        # a failed probe leaves nothing to use
        opened = self.opened_names()
        failure = {"ok": False, "reason": "timeout", "detail": "quota_rpc_failed"}
        self.probe_with(lambda profile: (None, failure))
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertEqual((info, err), (None, failure))
        self.assertIsNone(health.strict_quota_cache_row("default"))
        self.assertIsNone(codex_quota._cache_get("default", self.home_id))
        self.assertNotIn("quota.json", opened)


class DropFailureTests(CacheCase):
    """A forced refresh whose invalidation fails does not run the RPC."""

    def deny_unlink(self, label="default"):
        name = digest(label) + ".json"
        real = os.unlink

        def unlink(path, *args, **kwargs):
            if path == name:
                raise PermissionError(errno.EACCES, "denied")
            return real(path, *args, **kwargs)
        patcher = mock.patch.object(health.os, "unlink", unlink)
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def count_rpc(self, reply=None):
        calls = []

        def rpc(profile):
            calls.append(profile)
            return reply(profile) if reply else (self.info(used=44.0), None)
        self.probe_with(rpc)
        return calls

    DROP_FAILED = {"ok": False, "reason": "runtime_error",
                   "detail": "quota_cache_drop_failed"}

    def test_a_denied_unlink_refuses_the_forced_refresh_without_the_rpc(self):
        self.put("default", used_percent=5.0)
        before = self.snapshot_of("default")
        calls = self.count_rpc()
        self.deny_unlink()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(info)
        self.assertEqual(err, self.DROP_FAILED)
        self.assertEqual(calls, [])
        self.assertEqual(self.snapshot_of("default"), before)    # nothing was put either
        handle = health.try_exclusive_lock(
            os.path.join(self.state, health.QUOTA_LOCK_NAME), 0.5)
        self.assertIsNotNone(handle)                              # the lock was released
        health.release_lock(handle)

    def test_a_decision_does_not_admit_from_the_old_row_while_the_drop_keeps_failing(self):
        # A row too old for the decision to use goes to a forced refresh.
        self.put("default", used_percent=5.0, checked_at=time.time() - 100000)
        calls = self.count_rpc()
        denial = self.deny_unlink()
        for _ in range(2):
            code, doc = self.decide("default")
            self.assertNotEqual(code, 0)
            self.assertNotEqual(doc["decision"], "eligible")
            self.assertIsNone(doc.get("used_bp"))
        self.assertEqual(calls, [])
        # the drop works again: the refresh runs and the decision is made on it
        denial.stop()
        calls = self.count_rpc()
        code, doc = self.decide("default")
        self.assertEqual(calls, ["default"])
        self.assertEqual((code, doc["used_bp"]), (0, 4400))

    def test_a_missing_file_or_directory_is_a_successful_drop(self):
        calls = self.count_rpc()
        info, err = codex_quota.read_rate_limits("default", force=True)       # no directory
        self.assertIsNone(err)
        self.assertEqual(calls, ["default"])
        os.remove(self.path_of("default"))
        info, err = codex_quota.read_rate_limits("default", force=True)       # no file
        self.assertIsNone(err)
        self.assertEqual(calls, ["default", "default"])

    def test_an_unsafe_directory_refuses_the_forced_refresh(self):
        real = os.path.join(self.tmp.name, "real-quota")
        os.makedirs(real)
        os.symlink(real, self.qdir)
        calls = self.count_rpc()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(info)
        self.assertEqual(err, self.DROP_FAILED)
        self.assertEqual(calls, [])
        self.assertEqual(os.listdir(real), [])

    def test_a_link_at_the_profile_file_refuses_the_forced_refresh(self):
        os.makedirs(self.qdir)
        os.symlink(os.path.join(self.tmp.name, "nowhere"), self.path_of("default"))
        calls = self.count_rpc()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertEqual(err, self.DROP_FAILED)
        self.assertEqual(calls, [])

    def test_a_forced_refresh_that_drops_cleanly_still_runs(self):
        self.put("default", used_percent=5.0)
        calls = self.count_rpc()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(err)
        self.assertEqual((calls, info["used_percent"]), (["default"], 44.0))


class DirectoryDurabilityTests(CacheCase):
    """The rename and the unlink count only once the directory is synced."""

    def fail_directory_fsync(self, error=None):
        real = os.fsync
        synced = []

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                synced.append(fd)
                raise error or OSError(errno.EIO, "injected")
            return real(fd)
        patcher = mock.patch.object(health.os, "fsync", fsync)
        patcher.start()
        self.addCleanup(patcher.stop)
        return synced

    def spy_directory_fsync(self):
        real = os.fsync
        synced = []

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                synced.append(fd)
            return real(fd)
        patcher = mock.patch.object(health.os, "fsync", fsync)
        patcher.start()
        self.addCleanup(patcher.stop)
        return synced

    def test_a_put_syncs_the_directory_after_the_rename(self):
        order = []
        real_replace = os.replace
        synced = self.spy_directory_fsync()

        def replace(src, dst, **kwargs):
            order.append(("replace", len(synced)))
            return real_replace(src, dst, **kwargs)
        with mock.patch.object(health.os, "replace", replace):
            self.assertTrue(health.put_quota_row("default", self.row()))
        self.assertEqual(order, [("replace", 0)])
        self.assertEqual(len(synced), 1)

    def test_a_failed_directory_sync_is_a_failed_put(self):
        old = os.path.join(self.state, "quota.json")
        with open(old, "w", encoding="utf-8") as handle:
            handle.write("{}")
        synced = self.fail_directory_fsync()
        self.assertFalse(health.put_quota_row("default", self.row()))
        self.assertEqual(len(synced), 1)
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_write_failed"}])
        self.assertTrue(os.path.lexists(old))       # nothing counted as published
        self.assertEqual([n for n in os.listdir(self.qdir) if n.endswith(".tmp")], [])

    def test_a_failed_directory_sync_of_the_probe_put_is_logged_and_the_reading_returned(self):
        self.probe_with(lambda profile: (self.info(used=12.0), None))
        self.fail_directory_fsync()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(err)                       # the observation itself is good
        self.assertEqual(len(self.logged("quota_cache")), 1)

    def test_a_drop_syncs_the_directory_after_the_unlink(self):
        self.put("default")
        order = []
        real_unlink = os.unlink
        synced = self.spy_directory_fsync()

        def unlink(path, *args, **kwargs):
            order.append(("unlink", len(synced)))
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(health.os, "unlink", unlink):
            self.assertTrue(health.drop_quota_row("default"))
        self.assertEqual(order, [("unlink", 0)])
        self.assertEqual(len(synced), 1)

    def test_a_failed_directory_sync_is_a_failed_drop_and_blocks_the_refresh(self):
        self.put("default")
        calls = []
        self.probe_with(lambda profile: (calls.append(profile), (self.info(), None))[1])
        synced = self.fail_directory_fsync()
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(info)
        self.assertEqual(err, DropFailureTests.DROP_FAILED)
        self.assertEqual(calls, [])
        self.assertEqual(len(synced), 1)
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_write_failed"}])
        self.write_raw("default", json.dumps(dict(self.row(), profile="default")))
        self.assertFalse(health.drop_quota_row("default"))
        self.assertEqual(len(self.logged("quota_cache")), 2)

    def test_a_missing_file_error_from_the_directory_sync_is_a_failed_drop(self):
        # ENOENT raised after a successful unlink is a sync error, not an
        # absent file: the drop is not durable and the refresh must not run.
        self.put("default")
        calls = []
        self.probe_with(lambda profile: (calls.append(profile), (self.info(), None))[1])
        synced = self.fail_directory_fsync(FileNotFoundError(errno.ENOENT, "injected"))
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(info)
        self.assertEqual(err, DropFailureTests.DROP_FAILED)
        self.assertEqual(calls, [])
        self.assertEqual(len(synced), 1)
        self.assertEqual(self.logged("quota_cache"), [
            {"operation": "quota_cache", "detail": "quota_cache_write_failed"}])


class LoggingFailureTests(CacheCase):
    """A diagnostic that cannot be written never changes a cache result."""

    def break_log(self):
        patcher = mock.patch.object(health, "log_event", side_effect=BrokenPipeError())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_drop_of_a_malformed_file_still_unlinks_it_and_succeeds(self):
        self.write_raw("default", "{")
        self.break_log()
        self.assertTrue(health.drop_quota_row("default"))
        self.assertFalse(os.path.lexists(self.path_of("default")))

    def test_a_put_over_a_malformed_file_still_publishes_its_row(self):
        self.write_raw("default", "{")
        self.break_log()
        self.assertTrue(health.put_quota_row("default", self.row(used_percent=7.0)))
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 7.0)

    def test_a_failed_old_file_cleanup_does_not_escape_a_put(self):
        os.makedirs(os.path.join(self.state, "quota.json"))   # an unlink of it fails
        self.break_log()
        self.assertTrue(health.put_quota_row("default", self.row(used_percent=8.0)))
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 8.0)

    def test_a_failed_write_still_returns_false_when_its_log_line_fails(self):
        self.break_log()
        self.assertFalse(health.put_quota_row("default", self.row(used_percent=float("nan"))))
        self.assertFalse(os.path.lexists(self.path_of("default")))


class OversizedFileTests(CacheCase):
    """A drop never needs to read the file; every profile-file read is bounded."""

    def big(self, label="default", size=None):
        size = health.QUOTA_ROW_MAX_BYTES + 1 if size is None else size
        body = json.dumps(dict(self.row(), profile=label))
        self.write_raw(label, body + " " * (size - len(body)))
        return size

    def test_a_file_larger_than_the_cap_is_unlinked_by_a_drop_and_logged(self):
        self.big()
        with mock.patch.object(health, "strict_json_loads",
                               side_effect=AssertionError("decoded")):
            self.assertTrue(health.drop_quota_row("default"))
        self.assertFalse(os.path.lexists(self.path_of("default")))
        self.assertEqual(len(self.logged("malformed_file")), 1)
        self.assertEqual(self.logged("quota_cache"), [])
        self.assertTrue(health.drop_quota_row("default"))
        self.assertEqual(len(self.logged("malformed_file")), 1)

    def test_a_file_of_exactly_the_cap_is_still_a_row(self):
        self.big(size=health.QUOTA_ROW_MAX_BYTES)
        self.assertEqual(health.strict_quota_cache_row("default")["profile"], "default")

    def test_a_forced_refresh_replaces_an_oversized_file(self):
        self.big()
        self.probe_with(lambda profile: (self.info(used=3.0), None))
        info, err = codex_quota.read_rate_limits("default", force=True)
        self.assertIsNone(err)
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 3.0)

    def test_an_unreadable_file_is_unlinked_by_a_drop_and_logged(self):
        self.write_raw("default", "{")
        os.chmod(self.path_of("default"), 0)
        self.addCleanup(lambda: os.path.lexists(self.path_of("default"))
                        and os.chmod(self.path_of("default"), 0o600))
        self.assertTrue(health.drop_quota_row("default"))
        self.assertFalse(os.path.lexists(self.path_of("default")))
        self.assertEqual(len(self.logged("malformed_file")), 1)

    def test_a_read_that_fails_in_any_way_does_not_stop_the_unlink(self):
        self.put("default")
        with mock.patch.object(health, "_strict_read_maybe",
                               side_effect=OSError(errno.EIO, "boom")):
            self.assertTrue(health.drop_quota_row("default"))
        self.assertFalse(os.path.lexists(self.path_of("default")))
        self.assertEqual(len(self.logged("malformed_file")), 1)

    def test_the_read_itself_is_bounded_not_only_the_size_check(self):
        # The file is far over the cap but reports a small size: only a
        # bounded read stops it from being pulled in whole.
        self.big(size=health.QUOTA_ROW_MAX_BYTES * 4)
        real_fstat, real_fdopen = os.fstat, os.fdopen
        requested = []

        class Small:
            def __init__(self, info):
                self.st_mode, self.st_size = info.st_mode, 10

        def fstat(fd):
            info = real_fstat(fd)
            return Small(info) if stat.S_ISREG(info.st_mode) \
                and info.st_size > health.QUOTA_ROW_MAX_BYTES else info

        class Recorder:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *exc):
                return self.handle.__exit__(*exc)

            def read(self, *args):
                requested.append(args)
                return self.handle.read(*args)

        def fdopen(fd, mode="r", *args, **kwargs):
            handle = real_fdopen(fd, mode, *args, **kwargs)
            return Recorder(handle) if "b" in mode else handle
        with mock.patch.object(health.os, "fstat", fstat), \
                mock.patch.object(health.os, "fdopen", fdopen), \
                mock.patch.object(health, "strict_json_loads",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(health.StrictReadError):
                health.strict_quota_cache_row("default")
        self.assertEqual(requested, [(health.QUOTA_ROW_MAX_BYTES + 1,)])


class LeftoverTests(CacheCase):
    def test_8_a_leftover_temp_file_is_ignored_by_every_reader_and_writer(self):
        os.makedirs(self.qdir)
        leftovers = {}
        for name in (".omni-abcd1234.tmp", digest("default") + ".json.tmp",
                     digest("default") + ".tmp", "notes.json"):
            path = os.path.join(self.qdir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(self.row(used_percent=99.0),
                                             profile="default")))
            leftovers[path] = read_bytes(path)
        self.assertIsNone(health.strict_quota_cache_row("default"))
        self.assertIsNone(health.get_quota_row("default"))
        self.assertTrue(health.drop_quota_row("default"))
        self.put("default", used_percent=5.0)
        self.assertEqual(health.strict_quota_cache_row("default")["used_percent"], 5.0)
        self.assertEqual(self.decide("default")[0], 0)
        for path, content in leftovers.items():
            self.assertEqual(read_bytes(path), content, path)

    def test_a_failed_write_leaves_no_temp_file_behind(self):
        self.assertFalse(health.put_quota_row(
            "default", self.row(used_percent=float("nan"))))
        self.assertEqual(os.listdir(self.qdir), [])


class NanTests(CacheCase):
    def test_9_a_nan_in_a_reply_writes_nothing(self):
        self.put("work")
        work_before = self.snapshot_of("work")
        self.put("default", checked_at=time.time() - 3600, used_percent=1.0)
        default_before = self.snapshot_of("default")
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.events.clear()
            self.probe_with(lambda profile, bad=bad: (self.info(used=bad), None))
            info, err = codex_quota.read_rate_limits("default")
            self.assertIsNone(err)
            self.assertEqual(self.snapshot_of("default"), default_before)
            self.assertEqual(self.snapshot_of("work"), work_before)
            self.assertEqual(sorted(os.listdir(self.qdir)),
                             sorted([digest("default") + ".json",
                                     digest("work") + ".json"]))
            self.assertEqual(len(self.logged("quota_cache")), 1)
            self.assertNotIn("nan", json.dumps(self.events).lower())

    def test_9_a_nan_in_a_nested_window_writes_nothing(self):
        info = self.info(secondary={"used_percent": float("nan"), "resets_at": None,
                                    "window_mins": 10080, "reached": False})
        self.probe_with(lambda profile: (info, None))
        codex_quota.read_rate_limits("default")
        self.assertFalse(os.path.exists(self.path_of("default")))


if __name__ == "__main__":
    unittest.main()
