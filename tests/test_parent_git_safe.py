# no-opencc
"""Issue #157: git run by the omni PARENT must not execute repo-chosen commands.

Every vector below is set up with a command that appends to a marker file.
A control run of plain ``git status`` first proves the vector really executes
with an unprotected git (otherwise a passing test would prove nothing), then
the real parent entry point runs and the marker must stay absent.
"""
import contextlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from providers import agent_lane, git_safe, host_fence, jobs, kernel_sandbox

REAL_GIT = shutil.which("git")


_STATE_PATCH = None
_STATE_DIR = None


def setUpModule():
    # #168: a write-mode run records a repository's protected branches under the
    # state dir. Keep these fixtures' repositories out of the real one.
    global _STATE_PATCH, _STATE_DIR
    _STATE_DIR = tempfile.mkdtemp(prefix="omni_state_")
    _STATE_PATCH = mock.patch.dict(os.environ, {"OMNI_LLM_STATE_DIR": _STATE_DIR})
    _STATE_PATCH.start()


def tearDownModule():
    import shutil
    _STATE_PATCH.stop()
    shutil.rmtree(_STATE_DIR, ignore_errors=True)



@unittest.skipUnless(REAL_GIT, "git not on PATH")
class ParentGitCase(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="omni157_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = mock.patch.dict(os.environ, {
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in list(os.environ):
            if key.startswith("GIT_") and key not in (
                    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
                os.environ.pop(key)
        self._tick = 0

    # -- helpers ------------------------------------------------------- #
    def git(self, cwd, *args, check=True):
        p = subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=cwd, capture_output=True, text=True)
        if check and p.returncode:
            raise AssertionError((args, p.stderr))
        return p

    def repo(self, name="repo"):
        wd = os.path.join(self.root, name)
        os.makedirs(wd)
        self.git(wd, "init", "-q", "-b", "work")
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("hello\n")
        self.git(wd, "add", "a.txt")
        self.git(wd, "commit", "-qm", "seed")
        return wd

    def marker(self, name):
        return os.path.join(self.root, name + ".marker")

    def script(self, name, marker, body=""):
        path = os.path.join(self.root, name)
        with open(path, "w") as handle:
            handle.write('#!/bin/bash\necho ran >> "%s"\n%s\n' % (marker, body))
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def dirty(self, wd, rel="a.txt"):
        """Make a tracked file stat-dirty (content unchanged) so git rehashes it."""
        self._tick += 1
        stamp = time.mktime((2020, 1, 1, 0, 0, self._tick, 0, 0, -1))
        os.utime(os.path.join(wd, rel), (stamp, stamp))

    def attributes(self, wd, driver, where="info"):
        if where == "info":
            path = os.path.join(wd, ".git", "info", "attributes")
        else:
            path = os.path.join(wd, ".gitattributes")
        with open(path, "w") as handle:
            handle.write("a.txt filter=%s\n" % driver)
        if where != "info":
            self.git(wd, "add", ".gitattributes")
            self.git(wd, "commit", "-qm", "attr")

    def control(self, wd, marker, sub=None):
        """Plain git must execute the vector, else the test is vacuous."""
        self.dirty(sub or wd, "a.txt")
        self.git(wd, "status", "--porcelain", check=False)
        self.assertTrue(os.path.exists(marker), "control: vector does not execute")
        os.unlink(marker)
        self.dirty(sub or wd, "a.txt")

    def layer_b_off(self):
        """Disable every layer-B mitigation, leaving only the sandbox."""
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(
            git_safe, "safe_prefix",
            lambda workdir=None: ["-C", os.fspath(workdir)] if workdir is not None else []))
        stack.enter_context(mock.patch.object(
            git_safe, "_blank_filters", lambda w, g, env, t: env))
        stack.enter_context(mock.patch.object(git_safe, "_content_options", lambda: []))
        stack.enter_context(mock.patch.object(
            git_safe, "_base_env", lambda env: dict(os.environ if env is None else env)))

    def assert_no_exec(self, marker, what):
        self.assertFalse(os.path.exists(marker), "%s executed the vector" % what)

    def parent_entry_points(self, wd):
        """Run every parent git entry point against wd."""
        snap = agent_lane._workspace_snapshot(wd)
        agent_lane._workdir_fence(wd, True)
        jobs._drift_pin(wd)
        host_fence.repo_root_candidates(wd)
        try:
            host_fence._repo_write_roots(wd, required=False, allow_main=True)
        except host_fence.HostFenceError:
            pass
        return snap


class TestStatusVectors(ParentGitCase):
    def test_core_fsmonitor(self):
        wd, m = self.repo(), self.marker("fsm")
        self.git(wd, "config", "core.fsmonitor", self.script("fsm.sh", m))
        self.control(wd, m)
        snap = self.parent_entry_points(wd)
        self.assert_no_exec(m, "core.fsmonitor")
        self.assertEqual(snap["state"], "ok")

    def test_builtin_fsmonitor_daemon_not_started(self):
        wd = self.repo()
        self.git(wd, "config", "core.fsmonitor", "true")
        agent_lane._workspace_snapshot(wd)
        p = self.git(wd, "fsmonitor--daemon", "status", check=False)
        self.assertNotEqual(p.returncode, 0, "daemon is running: " + p.stdout)

    def test_clean_filter_via_info_attributes(self):
        wd, m = self.repo(), self.marker("clean")
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("clean.sh", m, "cat"))
        self.control(wd, m)
        snap = self.parent_entry_points(wd)
        self.assert_no_exec(m, "clean filter")
        self.assertEqual(snap["state"], "ok")

    def test_clean_filter_via_committed_gitattributes(self):
        wd, m = self.repo(), self.marker("clean2")
        self.attributes(wd, "evil", where="tree")
        self.git(wd, "config", "filter.evil.clean", self.script("clean2.sh", m, "cat"))
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "clean filter (.gitattributes)")

    def test_required_process_filter(self):
        wd, m = self.repo(), self.marker("proc")
        self.attributes(wd, "evilp")
        self.git(wd, "config", "filter.evilp.process", self.script("proc.sh", m, "exit 1"))
        self.git(wd, "config", "filter.evilp.required", "true")
        self.control(wd, m)
        snap = self.parent_entry_points(wd)
        self.assert_no_exec(m, "process filter")
        self.assertEqual(snap["state"], "ok", snap)

    def test_filter_driver_name_with_equals_and_dot(self):
        wd, m = self.repo(), self.marker("weird")
        self.attributes(wd, "we.ird=n")
        with open(os.path.join(wd, ".git", "config"), "a") as handle:
            handle.write('[filter "we.ird=n"]\n\tclean = %s\n\trequired = true\n'
                         % self.script("weird.sh", m, "cat"))
        self.control(wd, m)
        snap = self.parent_entry_points(wd)
        self.assert_no_exec(m, "filter named with '=' and '.'")
        self.assertEqual(snap["state"], "ok", snap)

    def test_include_path_pulls_fsmonitor_and_filter(self):
        wd, m = self.repo(), self.marker("inc")
        inc = os.path.join(self.root, "inc.cfg")
        with open(inc, "w") as handle:
            handle.write('[core]\n\tfsmonitor = %s\n[filter "inc"]\n\tclean = %s\n'
                         % (self.script("fsm_inc.sh", m),
                            self.script("clean_inc.sh", m, "cat")))
        self.attributes(wd, "inc")
        self.git(wd, "config", "include.path", inc)
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "include.path")

    def test_include_if_gitdir(self):
        wd, m = self.repo(), self.marker("incif")
        inc = os.path.join(self.root, "incif.cfg")
        with open(inc, "w") as handle:
            handle.write("[core]\n\tfsmonitor = %s\n" % self.script("fsm_if.sh", m))
        with open(os.path.join(wd, ".git", "config"), "a") as handle:
            handle.write('[includeIf "gitdir:%s/.git"]\n\tpath = %s\n' % (wd, inc))
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "includeIf")

    def test_worktree_config_file(self):
        wd, m = self.repo(), self.marker("wtcfg")
        self.git(wd, "config", "extensions.worktreeConfig", "true")
        with open(os.path.join(wd, ".git", "config.worktree"), "w") as handle:
            handle.write("[core]\n\tfsmonitor = %s\n" % self.script("fsm_wt.sh", m))
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "config.worktree")

    def test_gitdir_pointer_file_to_hostile_directory(self):
        src, m = self.repo("elsewhere"), self.marker("ptr")
        self.git(src, "config", "core.fsmonitor", self.script("fsm_ptr.sh", m))
        wd = os.path.join(self.root, "pointed")
        os.makedirs(wd)
        with open(os.path.join(wd, ".git"), "w") as handle:
            handle.write("gitdir: %s\n" % os.path.join(src, ".git"))
        self.git(wd, "status", "--porcelain", check=False)
        self.assertTrue(os.path.exists(m), "control: pointer vector does not execute")
        os.unlink(m)
        agent_lane._workspace_snapshot(wd)
        agent_lane._workdir_fence(wd, True)
        jobs._drift_pin(wd)
        host_fence.repo_root_candidates(wd)
        self.assert_no_exec(m, "gitdir pointer file")

    def test_post_index_change_hook_on_index_refresh(self):
        wd, m = self.repo(), self.marker("hook")
        hook = os.path.join(wd, ".git", "hooks", "post-index-change")
        with open(hook, "w") as handle:
            handle.write('#!/bin/bash\necho ran >> "%s"\n' % m)
        os.chmod(hook, 0o755)
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "post-index-change hook")

    def test_hooks_path_pointing_at_hostile_directory(self):
        wd, m = self.repo(), self.marker("hookspath")
        hooks = os.path.join(self.root, "hp")
        os.makedirs(hooks)
        with open(os.path.join(hooks, "post-index-change"), "w") as handle:
            handle.write('#!/bin/bash\necho ran >> "%s"\n' % m)
        os.chmod(os.path.join(hooks, "post-index-change"), 0o755)
        self.git(wd, "config", "core.hooksPath", hooks)
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "core.hooksPath")

    def test_status_does_not_rewrite_the_index(self):
        wd = self.repo()
        self.dirty(wd)
        index = os.path.join(wd, ".git", "index")
        before = os.stat(index).st_mtime_ns, open(index, "rb").read()
        time.sleep(0.01)
        agent_lane._workspace_snapshot(wd)
        after = os.stat(index).st_mtime_ns, open(index, "rb").read()
        self.assertEqual(before, after)


class TestSubmoduleVectors(ParentGitCase):
    def make(self):
        sub_src = self.repo("subsrc")
        main = self.repo("main")
        self.git(main, "-c", "protocol.file.allow=always", "submodule", "add",
                 "-q", sub_src, "sub")
        self.git(main, "commit", "-qm", "addsub")
        return main, os.path.join(main, "sub")

    def test_submodule_fsmonitor(self):
        main, sub = self.make()
        m = self.marker("subfsm")
        self.git(sub, "config", "core.fsmonitor", self.script("fsm_sub.sh", m))
        self.control(main, m)
        self.parent_entry_points(main)
        self.assert_no_exec(m, "submodule core.fsmonitor")

    def test_submodule_filter_driver(self):
        main, sub = self.make()
        m = self.marker("subclean")
        with open(os.path.join(sub, ".git"), "r") as handle:
            pointer = handle.read().strip()[len("gitdir: "):]
        gitdir = os.path.normpath(os.path.join(sub, pointer))
        with open(os.path.join(gitdir, "info", "attributes"), "w") as handle:
            handle.write("a.txt filter=ev\n")
        self.git(sub, "config", "filter.ev.clean", self.script("clean_sub.sh", m, "cat"))
        self.control(main, m, sub=sub)
        self.parent_entry_points(main)
        self.assert_no_exec(m, "submodule filter driver")

    def test_command_line_beats_gitmodules_ignore_none(self):
        main, sub = self.make()
        m = self.marker("subignore")
        self.git(main, "config", "-f", ".gitmodules", "submodule.sub.ignore", "none")
        self.git(sub, "config", "core.fsmonitor", self.script("fsm_sub2.sh", m))
        self.git(main, "config", "submodule.sub.ignore", "none")
        self.control(main, m)
        agent_lane._workspace_snapshot(main)
        self.assert_no_exec(m, "submodule.<name>.ignore=none")

    def test_moved_submodule_head_is_still_reported(self):
        main, sub = self.make()
        open(os.path.join(sub, "b.txt"), "w").write("b\n")
        self.git(sub, "add", "b.txt")
        self.git(sub, "commit", "-qm", "advance")
        snap = agent_lane._workspace_snapshot(main)
        self.assertEqual(snap["state"], "ok")
        self.assertEqual(len(snap["entries"]), 1, snap)


class TestIssueShape(ParentGitCase):
    """A child-created .git with fsmonitor in a workdir that was not a repo."""

    def test_post_run_snapshot_after_child_creates_repo(self):
        wd = os.path.join(self.root, "plain")
        os.makedirs(wd)
        m = self.marker("issue")
        before = agent_lane._workspace_snapshot(wd)
        self.assertEqual(before["state"], "not_git")
        # what the fenced child does
        self.git(wd, "init", "-q", "-b", "work")
        self.git(wd, "config", "core.fsmonitor", self.script("fsm_issue.sh", m))
        control = os.path.join(self.root, "control_copy")
        shutil.copytree(wd, control)
        self.git(control, "status", "--porcelain", check=False)
        self.assertTrue(os.path.exists(m), "control: vector does not execute")
        os.unlink(m)
        after = agent_lane._workspace_snapshot(wd)
        self.assert_no_exec(m, "post-run snapshot")
        self.assertEqual(after["state"], "ok")

    def test_read_only_lane_with_preplanted_hostile_config(self):
        wd, m = self.repo(), self.marker("planted")
        self.git(wd, "config", "core.fsmonitor", self.script("fsm_pl.sh", m))
        self.control(wd, m)
        agent_lane._workspace_snapshot(wd)
        self.assert_no_exec(m, "pre-run snapshot")

    def test_run_agent_end_to_end(self):
        import asyncio
        wd = os.path.join(self.root, "e2e")
        os.makedirs(wd)
        m = self.marker("e2e")
        script = self.script("fsm_e2e.sh", m)
        outer = self

        async def fake_run_cli(argv, **kwargs):
            outer.git(wd, "init", "-q", "-b", "work")
            outer.git(wd, "config", "core.fsmonitor", script)
            return {"returncode": 0, "stdout": "done", "stderr": "", "timed_out": False}

        old = os.environ.pop(agent_lane.ENV_DEPTH, None)
        self.addCleanup(lambda: old is None or os.environ.__setitem__(agent_lane.ENV_DEPTH, old))
        with mock.patch.dict(os.environ, {"OMNI_LLM_AGENT_FENCE": "0"}), \
                mock.patch.object(agent_lane, "run_cli", fake_run_cli), \
                mock.patch.object(agent_lane.shutil, "which",
                                  lambda n, *a, **k: REAL_GIT if n == "git" else "/fake/bin/" + str(n)), \
                mock.patch.object(agent_lane.kernel_sandbox, "resolve_agent_binary",
                                  lambda p: "/fake/bin/" + p):
            env = asyncio.run(agent_lane.run_agent(
                task="do it", model="claude/sonnet", workdir=wd, timeout_s=30))
        self.assertTrue(env.get("ok"), env)
        self.assert_no_exec(m, "run_agent post-run snapshot")


class TestHostFenceEntryPoints(ParentGitCase):
    def test_exec_path_query_in_hostile_cwd(self):
        wd, m = self.repo(), self.marker("execpath")
        self.git(wd, "config", "core.fsmonitor", self.script("fsm_ep.sh", m))
        old = os.getcwd()
        os.chdir(wd)
        self.addCleanup(os.chdir, old)
        host_fence._git_remote_exec_denies()
        self.assert_no_exec(m, "git --exec-path")

    def test_rev_parse_and_symbolic_ref_run_nothing_even_unprotected(self):
        """Pins why those subcommands skip filter enumeration."""
        wd, m = self.repo(), self.marker("revparse")
        self.git(wd, "config", "core.fsmonitor", self.script("fsm_rp.sh", m))
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("clean_rp.sh", m, "cat"))
        self.dirty(wd)
        for args in (("rev-parse", "--show-toplevel"), ("rev-parse", "--is-inside-work-tree"),
                     ("rev-parse", "HEAD"), ("symbolic-ref", "--short", "-q", "HEAD")):
            self.git(wd, *args)
        self.assert_no_exec(m, "rev-parse / symbolic-ref")

    def test_diff_name_only_runs_no_textconv_or_external_diff(self):
        wd, m = self.repo(), self.marker("diff")
        with open(os.path.join(wd, ".gitattributes"), "w") as handle:
            handle.write("a.txt diff=ev\n")
        self.git(wd, "add", ".gitattributes")
        self.git(wd, "commit", "-qm", "attr")
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("two\n")
        self.git(wd, "commit", "-qam", "two")
        self.git(wd, "config", "diff.ev.textconv", self.script("tc.sh", m, 'cat "$1"'))
        self.git(wd, "config", "diff.external", self.script("ext.sh", m))
        before = agent_lane._workspace_snapshot(wd)
        head = before["head"]
        prev = self.git(wd, "rev-parse", "HEAD~1").stdout.strip()
        old = {"state": "ok", "entries": {}, "head": prev}
        new = {"state": "ok", "entries": {}, "head": head}
        count, unknown = agent_lane._workspace_writes(wd, old, new)
        self.assertEqual((count, unknown), (1, None))
        self.assert_no_exec(m, "diff --name-only")


class TestSnapshotMeaning(ParentGitCase):
    """Entries keep the meaning plain ``git status --porcelain`` gives."""

    def test_entries_match_plain_status(self):
        wd = self.repo()
        for name in ("b.txt", "c.txt", "d.txt"):
            with open(os.path.join(wd, name), "w") as handle:
                handle.write(name)
        self.git(wd, "add", "b.txt", "c.txt", "d.txt")
        self.git(wd, "commit", "-qm", "more")
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("modified\n")                      # ' M'
        self.git(wd, "mv", "b.txt", "b2.txt")                # 'R '
        os.remove(os.path.join(wd, "c.txt"))                 # ' D'
        with open(os.path.join(wd, "new.txt"), "w") as handle:
            handle.write("n")                                # '??'
        os.makedirs(os.path.join(wd, "dir"))
        with open(os.path.join(wd, "dir", "deep.txt"), "w") as handle:
            handle.write("n")                                # '?? dir/deep.txt'
        self.git(wd, "add", "d.txt")
        with open(os.path.join(wd, "d.txt"), "w") as handle:
            handle.write("staged then edited")               # 'MM'
        # .strip() mirrors agent_lane._git, which strips the whole stream
        plain = self.git(wd, "status", "--porcelain", "--untracked-files=all").stdout.strip()
        expected = agent_lane._parse_porcelain(plain)
        snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(snap["state"], "ok")
        self.assertEqual(snap["entries"], expected)
        self.assertGreaterEqual(len(expected), 5)
        self.assertEqual(snap["head"], self.git(wd, "rev-parse", "HEAD").stdout.strip())

    def test_changes_between_snapshots_are_counted(self):
        wd = self.repo()
        before = agent_lane._workspace_snapshot(wd)
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("changed\n")
        with open(os.path.join(wd, "added.txt"), "w") as handle:
            handle.write("x")
        after = agent_lane._workspace_snapshot(wd)
        self.assertEqual(agent_lane._workspace_writes(wd, before, after), (2, None))

    def test_driver_defined_filter_does_not_break_content_status(self):
        wd = self.repo()
        self.attributes(wd, "ok")
        self.git(wd, "config", "filter.ok.clean", "cat")
        self.git(wd, "config", "filter.ok.required", "true")
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("changed\n")
        snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(len(snap["entries"]), 1, snap)


class TestArgvPins(ParentGitCase):
    def test_every_parent_git_call_carries_the_safe_flags(self):
        calls = []
        real_run = subprocess.run

        def spy(argv, *a, **kw):
            calls.append((list(argv), dict(kw.get("env") or {})))
            return real_run(argv, *a, **kw)

        wd = self.repo()
        self.attributes(wd, "pin")
        self.git(wd, "config", "filter.pin.clean", "cat")
        with mock.patch.object(subprocess, "run", spy):
            agent_lane._workspace_snapshot(wd)
            agent_lane._workdir_fence(wd, True)
            jobs._drift_pin(wd)
            host_fence.repo_root_candidates(wd)
            try:
                host_fence._repo_write_roots(wd, required=False, allow_main=True)
            except host_fence.HostFenceError:
                pass
            host_fence._git_remote_exec_denies()
            agent_lane._workspace_writes(
                wd, {"state": "ok", "entries": {}, "head": "a" * 40},
                {"state": "ok", "entries": {}, "head": "b" * 40})
        git_calls = []
        wrappers = []
        for argv, env in calls:
            if os.path.basename(argv[0]) == "sandbox-exec":
                at = next(i for i, word in enumerate(argv)
                          if i and os.path.basename(word) == "git")
                wrappers.append(argv[:at])
                git_calls.append((argv[at:], env))
            elif os.path.basename(argv[0]) == "git":
                git_calls.append((argv, env))
        self.assertGreaterEqual(len(git_calls), 12)
        for argv, env in git_calls:
            sub = argv[:argv.index("-C")] if "-C" in argv else argv
            self.assertIn("--no-optional-locks", sub, argv)
            self.assertIn("core.fsmonitor=false", sub, argv)
            self.assertIn("core.hooksPath=/dev/null", sub, argv)
            self.assertEqual(env.get("GIT_OPTIONAL_LOCKS"), "0", argv)
            self.assertEqual(env.get("GIT_NO_LAZY_FETCH"), "1", argv)
        statuses = [argv for argv, _ in git_calls if "status" in argv]
        self.assertTrue(statuses)
        for argv in statuses:
            self.assertIn("--ignore-submodules=dirty", argv)
            self.assertIn("--no-lazy-fetch", argv)
        diffs = [argv for argv, _ in git_calls if "diff" in argv]
        self.assertTrue(diffs)
        for argv in diffs:
            self.assertIn("--no-ext-diff", argv)
            self.assertIn("--no-textconv", argv)
            self.assertIn("--no-lazy-fetch", argv)
        # --no-lazy-fetch stays off rev-parse / symbolic-ref (old git exits 129,
        # and the host fence reads any non-zero rev-parse as "not a repo")
        for argv, _ in git_calls:
            if "rev-parse" in argv or "symbolic-ref" in argv or "--exec-path" in argv:
                self.assertNotIn("--no-lazy-fetch", argv)
        # the filter driver in this repo is blanked for the content-reading call
        status_env = [env for argv, env in git_calls if "status" in argv][0]
        self.assertGreaterEqual(int(status_env["GIT_CONFIG_COUNT"]), 4)
        keys = {status_env["GIT_CONFIG_KEY_%d" % i]: status_env["GIT_CONFIG_VALUE_%d" % i]
                for i in range(int(status_env["GIT_CONFIG_COUNT"]))}
        self.assertEqual(keys["filter.pin.clean"], "")
        self.assertEqual(keys["filter.pin.process"], "")
        self.assertEqual(keys["filter.pin.required"], "false")
        self.assertTrue(git_safe.git_argv("/w", ["status"]).count("-C") == 1)
        if sys.platform == "darwin" and kernel_sandbox.available():
            # every parent git call, config reads and rev-parse included, is wrapped
            self.assertEqual(len(wrappers), len(git_calls))
            for wrapper in wrappers:
                self.assertEqual(wrapper[0], kernel_sandbox.SANDBOX_EXEC)
                profile = wrapper[2]
                for clause in ("(deny network*)", "(deny process-exec*)",
                               "(deny file-write*)"):
                    self.assertIn(clause, profile)

    def test_inherited_environment_cannot_blind_or_redirect_git(self):
        wd = self.repo()
        seen = []
        real_run = subprocess.run

        def spy(argv, *a, **kw):
            seen.append(dict(kw.get("env") or {}))
            return real_run(argv, *a, **kw)

        hostile = {"GIT_CONFIG": os.devnull, "GIT_DIR": "/nonexistent",
                   "GIT_WORK_TREE": "/nonexistent", "GIT_INDEX_FILE": "/nonexistent",
                   "GIT_OBJECT_DIRECTORY": "/nonexistent",
                   "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/nonexistent",
                   "GIT_COMMON_DIR": "/nonexistent", "GIT_CONFIG_PARAMETERS": "'a.b=c'"}
        with mock.patch.dict(os.environ, hostile), \
                mock.patch.object(subprocess, "run", spy):
            snap = agent_lane._workspace_snapshot(wd)
            pin = jobs._drift_pin(wd)
        self.assertEqual(snap["state"], "ok", snap)
        self.assertEqual(pin["repo_root"], wd)
        self.assertTrue(seen)
        for env in seen:
            for name in hostile:
                self.assertNotIn(name, env)


class TestFailClosed(ParentGitCase):
    def old_git_dir(self):
        """A `git` that ignores GIT_CONFIG_COUNT, like a git older than 2.31."""
        bindir = os.path.join(self.root, "oldbin")
        os.makedirs(bindir)
        path = os.path.join(bindir, "git")
        with open(path, "w") as handle:
            handle.write('#!/bin/bash\nunset GIT_CONFIG_COUNT\nexec "%s" "$@"\n' % REAL_GIT)
        os.chmod(path, 0o755)
        return bindir

    def test_git_that_cannot_blank_filters_makes_the_snapshot_an_error(self):
        wd, m = self.repo(), self.marker("old")
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("clean_old.sh", m, "cat"))
        self.dirty(wd)
        path = self.old_git_dir() + os.pathsep + os.environ["PATH"]
        # layer A off: the wrapper script itself could not be exec'd under it
        with mock.patch.dict(os.environ, {"PATH": path}), \
                mock.patch.object(git_safe, "_sandboxed", return_value=False):
            snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(snap["state"], "error", snap)
        self.assert_no_exec(m, "status under a git without GIT_CONFIG_COUNT")

    def test_same_git_is_fine_when_no_filter_driver_exists(self):
        wd = self.repo()
        path = self.old_git_dir() + os.pathsep + os.environ["PATH"]
        # layer A off: the wrapper script itself could not be exec'd under it
        with mock.patch.dict(os.environ, {"PATH": path}), \
                mock.patch.object(git_safe, "_sandboxed", return_value=False):
            snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(snap["state"], "ok", snap)

    def test_git_safe_error_is_an_indeterminate_state_for_jobs_and_fence(self):
        wd = self.repo()
        boom = git_safe.GitSafeError("x")
        with mock.patch.object(git_safe, "_blank_filters", side_effect=boom):
            rc, _out, err = agent_lane._git(wd, "status", "--porcelain")
        self.assertIsNone(rc)
        self.assertIn("x", err)

    def test_preexisting_git_config_count_is_extended_not_replaced(self):
        wd, m = self.repo(), self.marker("count")
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("clean_cnt.sh", m, "cat"))
        self.dirty(wd)
        with mock.patch.dict(os.environ, {
                "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "status.branch",
                "GIT_CONFIG_VALUE_0": "true"}):
            snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(snap["state"], "ok", snap)
        self.assert_no_exec(m, "status with pre-set GIT_CONFIG_COUNT")


def _lazy_fetch_repo(case, name):
    """A promisor-remote repo whose HEAD tree object is missing (ssh = marker)."""
    wd, m = case.repo(name), case.marker(name)
    tree = case.git(wd, "rev-parse", "HEAD^{tree}").stdout.strip()
    case.git(wd, "config", "remote.x.url", "ssh://host/repo")
    case.git(wd, "config", "remote.x.promisor", "true")
    case.git(wd, "config", "extensions.partialClone", "x")
    case.git(wd, "config", "core.repositoryformatversion", "1")
    case.git(wd, "config", "core.sshCommand", case.script(name + "_ssh.sh", m))
    obj = os.path.join(wd, ".git", "objects", tree[:2], tree[2:])
    os.chmod(obj, 0o644)
    os.remove(obj)
    return wd, m


def _empty_name_repo(case, name):
    wd, m = case.repo(name), case.marker(name)
    case.attributes(wd, "")
    with open(os.path.join(wd, ".git", "config"), "a") as handle:
        handle.write('[filter ""]\n\tclean = %s\n' % case.script(name + ".sh", m, "cat"))
    return wd, m


class TestEmptyNameLazyFetchAndConfigEnv(ParentGitCase):
    """Fix round 1: vectors found by review, all layers on."""

    def test_empty_driver_name_is_blanked(self):
        wd, m = _empty_name_repo(self, "empty")
        self.control(wd, m)
        snap = agent_lane._workspace_snapshot(wd)
        self.assert_no_exec(m, "filter with an empty name")
        self.assertEqual(snap["state"], "ok", snap)

    def test_enumeration_sees_the_empty_name(self):
        wd, _m = _empty_name_repo(self, "empty2")
        env = git_safe._base_env(None)
        self.assertEqual(git_safe._filter_drivers(wd, "git", env, 10), [""])

    def test_verification_runs_even_when_no_driver_was_found(self):
        wd = self.repo()
        calls = []
        real = git_safe._verify_blanked

        def spy(*args):
            calls.append(args)
            return real(*args)

        with mock.patch.object(git_safe, "_verify_blanked", spy):
            git_safe.run_git(wd, ["status", "--porcelain"])
            git_safe.run_git(wd, ["rev-parse", "HEAD"])
        self.assertEqual(len(calls), 1, "content subcommand verifies, rev-parse does not")

    def test_driver_appearing_after_enumeration_is_caught_by_verify(self):
        wd, m = self.repo(), self.marker("race")
        self.attributes(wd, "late")
        self.dirty(wd)
        real = git_safe._filter_drivers

        def enumerate_then_plant(*args):
            names = real(*args)  # harmless config seen here ...
            self.git(wd, "config", "filter.late.clean", self.script("late.sh", m, "cat"))
            return names         # ... hostile one appears before the real call

        with mock.patch.object(git_safe, "_filter_drivers", enumerate_then_plant):
            snap = agent_lane._workspace_snapshot(wd)
        self.assertEqual(snap["state"], "error", snap)
        self.assert_no_exec(m, "driver planted after enumeration")

    def test_promisor_lazy_fetch_runs_no_ssh_command(self):
        wd, m = _lazy_fetch_repo(self, "lazy")
        self.git(wd, "status", "--porcelain", check=False)
        self.assertTrue(os.path.exists(m), "control: lazy fetch does not run ssh")
        os.unlink(m)
        snap = agent_lane._workspace_snapshot(wd)
        head = self.git(wd, "rev-parse", "HEAD").stdout.strip()
        agent_lane._workspace_writes(
            wd, {"state": "ok", "entries": {}, "head": "a" * 40},
            {"state": "ok", "entries": {}, "head": head})
        self.assert_no_exec(m, "promisor lazy fetch")
        self.assertEqual(snap["state"], "error")

    def test_inherited_git_config_does_not_blind_the_enumeration(self):
        wd, m = self.repo(), self.marker("gitconfigenv")
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("gc.sh", m, "cat"))
        with mock.patch.dict(os.environ, {"GIT_CONFIG": os.devnull}):
            self.control(wd, m)
            snap = agent_lane._workspace_snapshot(wd)
        self.assert_no_exec(m, "status with inherited GIT_CONFIG")
        self.assertEqual(snap["state"], "ok", snap)


@unittest.skipUnless(sys.platform == "darwin" and kernel_sandbox.available(),
                     "sandbox-exec layer is Darwin-only")
class TestSandboxLayerAlone(ParentGitCase):
    """Layer A on its own: every layer-B mitigation is switched off."""

    def setUp(self):
        super().setUp()
        self.layer_b_off()

    def test_layer_b_really_is_off(self):
        self.assertEqual(git_safe.git_argv("/w", ["status"]),
                         ["git", "-C", "/w", "status"])
        self.assertNotIn("GIT_OPTIONAL_LOCKS", git_safe._base_env({}))

    def test_fsmonitor(self):
        wd, m = self.repo(), self.marker("a_fsm")
        self.git(wd, "config", "core.fsmonitor", self.script("a_fsm.sh", m))
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "fsmonitor under the sandbox alone")

    def test_clean_filter(self):
        wd, m = self.repo(), self.marker("a_clean")
        self.attributes(wd, "evil")
        self.git(wd, "config", "filter.evil.clean", self.script("a_clean.sh", m, "cat"))
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "clean filter under the sandbox alone")

    def test_empty_name_filter(self):
        wd, m = _empty_name_repo(self, "a_empty")
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "empty-name filter under the sandbox alone")

    def test_post_index_change_hook(self):
        wd, m = self.repo(), self.marker("a_hook")
        hook = os.path.join(wd, ".git", "hooks", "post-index-change")
        with open(hook, "w") as handle:
            handle.write('#!/bin/bash\necho ran >> "%s"\n' % m)
        os.chmod(hook, 0o755)
        self.control(wd, m)
        self.parent_entry_points(wd)
        self.assert_no_exec(m, "post-index-change hook under the sandbox alone")

    def test_promisor_lazy_fetch(self):
        wd, m = _lazy_fetch_repo(self, "a_lazy")
        self.git(wd, "status", "--porcelain", check=False)
        self.assertTrue(os.path.exists(m), "control: lazy fetch does not run ssh")
        os.unlink(m)
        agent_lane._workspace_snapshot(wd)
        self.assert_no_exec(m, "promisor lazy fetch under the sandbox alone")

    def test_submodule_status_child_cannot_exec(self):
        sub_src, main = self.repo("a_subsrc"), self.repo("a_main")
        self.git(main, "-c", "protocol.file.allow=always", "submodule", "add",
                 "-q", sub_src, "sub")
        self.git(main, "commit", "-qm", "addsub")
        m = self.marker("a_sub")
        self.git(os.path.join(main, "sub"), "config", "core.fsmonitor",
                 self.script("a_sub.sh", m))
        self.control(main, m)
        # no --ignore-submodules here: git itself recurses into the submodule
        git_safe.run_git(main, ["status", "--porcelain"])
        self.assert_no_exec(m, "submodule fsmonitor under the sandbox alone")

    def test_exec_allowlist_is_the_one_trusted_git(self):
        trusted = git_safe._trusted_git()
        argv = git_safe._sandbox_argv(trusted, ["git", "status"])
        params = [w for w in argv if w.startswith(("X", "W")) and "=" in w]
        self.assertEqual(params, ["X0=" + trusted, "W0=/dev/null"])
        self.assertEqual(os.path.basename(trusted), "git")
        self.assertEqual(git_safe._WRITE_ALLOW, ("/dev/null",))

    def test_profile_denies_network_exec_and_writes(self):
        profile = git_safe.sandbox_profile(("/x/git",))
        for clause in ("(deny network*)", "(deny process-exec*)", "(deny file-write*)"):
            self.assertIn(clause, profile)
        self.assertNotIn("/x/git", profile)  # paths travel as -D parameters

    def test_wrapped_process_cannot_exec_or_write(self):
        trusted = git_safe._trusted_git()
        marker = self.marker("a_direct")
        argv = git_safe._sandbox_argv(trusted, [trusted])[:-1]
        p = subprocess.run(argv + ["/bin/sh", "-c", 'echo hi > "%s"' % marker],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(os.path.exists(marker))
        p = subprocess.run(argv + [trusted, "--version"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_unavailable_sandbox_fails_closed(self):
        wd = self.repo()
        with mock.patch.object(kernel_sandbox, "available", return_value=False):
            rc, _out, err = agent_lane._git(wd, "rev-parse", "HEAD")
            self.assertIsNone(rc)
            self.assertIn("sandbox", err)
            snap = agent_lane._workspace_snapshot(wd)
            self.assertEqual(snap["state"], "error")
            self.assertIsNotNone(agent_lane._workdir_fence(wd, True))
            self.assertEqual(jobs._drift_pin(wd),
                             {"repo_root": None, "branch": None, "head": None})


class TestWriteAccountingEdges(ParentGitCase):
    def test_commit_range_diff_failure_is_indeterminate_not_zero(self):
        wd = self.repo()
        old = {"state": "ok", "entries": {}, "head": "a" * 40}
        new = {"state": "ok", "entries": {}, "head": "b" * 40}
        count, detail = agent_lane._workspace_writes(wd, old, new)
        self.assertIsNone(count)
        self.assertIn("commit range", detail)
        with mock.patch.object(agent_lane, "_git", return_value=(None, "", "refused")):
            count, detail = agent_lane._workspace_writes(wd, old, new)
        self.assertIsNone(count)

    def test_transforming_filter_may_undercount_but_expect_writes_stays_loud(self):
        """Pins the documented trade-off of blanking repository filters."""
        import asyncio
        wd = self.repo("lfsish")
        self.attributes(wd, "up")
        self.git(wd, "config", "filter.up.clean", "tr a-z A-Z")
        with open(os.path.join(wd, "a.txt"), "w") as handle:
            handle.write("hello\n")
        self.git(wd, "add", "a.txt")
        self.git(wd, "commit", "-qm", "upper in index")
        self.dirty(wd)  # stat-dirty: git must rehash, and the clean filter is blanked
        before = agent_lane._workspace_snapshot(wd)
        self.assertEqual(len(before["entries"]), 1, before)  # shows as modified at once

        async def fake_run_cli(argv, **kwargs):
            with open(os.path.join(wd, "a.txt"), "w") as handle:
                handle.write("world\n")
            return {"returncode": 0, "stdout": "done", "stderr": "", "timed_out": False}

        old_depth = os.environ.pop(agent_lane.ENV_DEPTH, None)
        self.addCleanup(lambda: old_depth is None
                        or os.environ.__setitem__(agent_lane.ENV_DEPTH, old_depth))
        with mock.patch.dict(os.environ, {"OMNI_LLM_AGENT_FENCE": "0"}), \
                mock.patch.object(agent_lane, "run_cli", fake_run_cli), \
                mock.patch.object(agent_lane, "_workdir_fence", return_value=None), \
                mock.patch.object(agent_lane.shutil, "which",
                                  lambda n, *a, **k: REAL_GIT if n == "git" else "/fake/bin/" + str(n)), \
                mock.patch.object(agent_lane.kernel_sandbox, "resolve_agent_binary",
                                  lambda p: "/fake/bin/" + p):
            env = asyncio.run(agent_lane.run_agent(
                task="do it", model="claude/sonnet", workdir=wd, timeout_s=30,
                expect_writes=True))
        self.assertEqual(env.get("workspace_writes"), 0, env)
        self.assertEqual(env.get("reason"), "no_workspace_writes", env)
        self.assertFalse(env.get("ok"))



@unittest.skipUnless(sys.platform == "darwin" and kernel_sandbox.available(),
                     "trusted-git discovery is Darwin-only")
class TestTrustedGitDiscovery(ParentGitCase):
    """Fix round 2: which git binary the parent runs, and when."""

    def planted_git(self, name, marker):
        bindir = os.path.join(self.root, name)
        os.makedirs(bindir)
        path = os.path.join(bindir, "git")
        with open(path, "w") as handle:
            handle.write('#!/bin/bash\necho planted >> "%s"\nexec "%s" "$@"\n'
                         % (marker, REAL_GIT))
        os.chmod(path, 0o755)
        return bindir, path

    def spy_run(self):
        calls = []
        real_run = subprocess.run

        def spy(argv, *a, **kw):
            calls.append((list(argv), dict(kw.get("env") or {})))
            return real_run(argv, *a, **kw)

        return calls, mock.patch.object(subprocess, "run", spy)

    def test_planted_path_git_is_never_run(self):
        wd, m = self.repo(), self.marker("pathgit")
        bindir, planted = self.planted_git("plantedbin", m)
        old = os.getcwd()
        os.chdir(wd)
        self.addCleanup(os.chdir, old)
        calls, patch = self.spy_run()
        with mock.patch.dict(os.environ, {"PATH": bindir + ":/usr/bin:/bin"}), patch:
            snap = agent_lane._workspace_snapshot(wd)
            fence = agent_lane._workdir_fence(wd, True)
            pin = jobs._drift_pin(wd)
            host_fence._git_remote_exec_denies()
        self.assert_no_exec(m, "a git planted on PATH")
        self.assertEqual(snap["state"], "ok", snap)
        self.assertIsNone(fence)
        self.assertEqual(pin["repo_root"], wd)
        self.assertTrue(calls)
        for argv, _env in calls:
            self.assertNotIn(planted, argv)

    def test_planted_git_via_git_exec_path_env_is_not_allowlisted(self):
        wd, m = self.repo(), self.marker("execpathenv")
        libexec = os.path.join(self.root, "evil", "libexec", "git-core")
        os.makedirs(libexec)
        evil_bin = os.path.join(self.root, "evil", "bin")
        os.makedirs(evil_bin)
        planted = os.path.join(evil_bin, "git")
        with open(planted, "w") as handle:
            handle.write('#!/bin/bash\necho planted >> "%s"\n' % m)
        os.chmod(planted, 0o755)
        calls, patch = self.spy_run()
        with mock.patch.dict(os.environ, {"GIT_EXEC_PATH": libexec}), patch:
            snap = agent_lane._workspace_snapshot(wd)
            argv = git_safe._sandbox_argv(git_safe._trusted_git(), ["git", "status"])
        self.assert_no_exec(m, "GIT_EXEC_PATH")
        self.assertEqual(snap["state"], "ok", snap)
        self.assertNotIn(planted, " ".join(argv))
        self.assertNotIn("GIT_EXEC_PATH", git_safe._base_env({"GIT_EXEC_PATH": libexec}))
        for _argv, env in calls:
            self.assertNotIn("GIT_EXEC_PATH", env)

    def test_the_usr_bin_git_shim_is_never_executed(self):
        wd = self.repo()
        calls, patch = self.spy_run()
        with patch:
            agent_lane._workspace_snapshot(wd)
            agent_lane._workdir_fence(wd, True)
            jobs._drift_pin(wd)
            host_fence.repo_root_candidates(wd)
            try:
                host_fence._repo_write_roots(wd, required=False, allow_main=True)
            except host_fence.HostFenceError:
                pass
            host_fence._git_remote_exec_denies()
        git_calls = [(a, e) for a, e in calls if any(os.path.basename(w) == "git" for w in a)]
        self.assertGreaterEqual(len(git_calls), 12)
        trusted = git_safe._trusted_git()
        for argv, env in git_calls:
            self.assertNotIn("/usr/bin/git", argv)
            self.assertIn(trusted, argv)
            self.assertEqual(env.get("xcrun_nocache"), "1")
        # nothing runs outside sandbox-exec: no bare git, no unsandboxed probe
        for argv, _env in calls:
            if any(os.path.basename(w) == "git" for w in argv):
                self.assertEqual(argv[0], kernel_sandbox.SANDBOX_EXEC, argv)

    def fake_developer_dir(self, name="dev"):
        dev = os.path.join(self.root, name)
        os.makedirs(os.path.join(dev, "usr", "bin"))
        binary = os.path.join(dev, "usr", "bin", "git")
        shutil.copy(REAL_GIT, binary)
        os.chmod(binary, 0o755)
        link = os.path.join(self.root, name + "_link")
        os.symlink(dev, link)
        return dev, link, os.path.realpath(binary)

    def trust(self, link, uids=None):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(git_safe, "_XCODE_SELECT_LINK", link))
        stack.enter_context(mock.patch.object(
            git_safe, "_TRUSTED_UIDS", uids or (0, os.getuid())))

    def test_real_host_trusted_git(self):
        trusted = git_safe._trusted_git()
        self.assertEqual(trusted, os.path.join(
            os.path.realpath("/var/db/xcode_select_link"), "usr", "bin", "git"))

    def test_trust_check_accepts_a_clean_chain(self):
        _dev, link, binary = self.fake_developer_dir()
        self.trust(link)
        self.assertEqual(git_safe._trusted_git(), binary)

    def test_group_writable_component_is_refused(self):
        dev, link, binary = self.fake_developer_dir()
        self.trust(link)
        os.chmod(os.path.join(dev, "usr"), 0o775)
        with self.assertRaises(git_safe.GitSafeError) as ctx:
            git_safe._trusted_git()
        self.assertIn("group- or world-writable", str(ctx.exception))
        self.assertIn(binary, str(ctx.exception))
        wd = self.repo()
        rc, _out, err = agent_lane._git(wd, "rev-parse", "HEAD")
        self.assertIsNone(rc)
        self.assertIn("no trusted git", err)

    def test_world_writable_binary_is_refused(self):
        _dev, link, binary = self.fake_developer_dir()
        self.trust(link)
        os.chmod(binary, 0o757)
        with self.assertRaises(git_safe.GitSafeError):
            git_safe._trusted_git()

    def test_binary_not_owned_by_a_trusted_uid_is_refused(self):
        _dev, link, _binary = self.fake_developer_dir()
        self.trust(link, uids=(0,))  # the temp tree belongs to the test user
        with self.assertRaises(git_safe.GitSafeError) as ctx:
            git_safe._trusted_git()
        self.assertIn("not root-owned", str(ctx.exception))

    def test_symlinked_binary_is_refused(self):
        dev, link, binary = self.fake_developer_dir()
        self.trust(link)
        os.remove(binary)
        os.symlink(REAL_GIT, binary)
        with self.assertRaises(git_safe.GitSafeError):
            git_safe._trusted_git()

    def test_missing_binary_names_the_expected_path(self):
        dev, link, binary = self.fake_developer_dir()
        self.trust(link)
        os.remove(binary)
        with self.assertRaises(git_safe.GitSafeError) as ctx:
            git_safe._trusted_git()
        self.assertIn(binary, str(ctx.exception))

    def test_no_lazy_fetch_follows_the_trusted_git_version(self):
        wd = self.repo()
        for version, expect in (((2, 39), False), ((2, 44), False),
                                ((2, 45), True), ((2, 54), True)):
            calls, patch = self.spy_run()
            with mock.patch.object(git_safe, "_git_version", return_value=version), patch:
                snap = agent_lane._workspace_snapshot(wd)
            self.assertEqual(snap["state"], "ok", (version, snap))
            status = [(a, e) for a, e in calls if "status" in a]
            self.assertTrue(status)
            for argv, env in status:
                self.assertEqual("--no-lazy-fetch" in argv, expect, (version, argv))
                self.assertEqual(env.get("GIT_NO_LAZY_FETCH"), "1")
            for argv, _env in calls:
                if "rev-parse" in argv or "symbolic-ref" in argv:
                    self.assertNotIn("--no-lazy-fetch", argv)

    def test_real_trusted_git_version_is_asked_under_the_sandbox(self):
        trusted = git_safe._trusted_git()
        calls, patch = self.spy_run()
        git_safe._version_cache.clear()
        with patch:
            version = git_safe._git_version(trusted, git_safe._base_env(None))
        self.assertGreaterEqual(version, (2, 0))
        self.assertEqual(calls[0][0][0], kernel_sandbox.SANDBOX_EXEC)

    def test_remote_exec_denies_refuses_when_git_cannot_answer(self):
        for effect in (git_safe.GitSafeError("boom"),
                       subprocess.CompletedProcess([], 1, "", "err"),
                       subprocess.CompletedProcess([], 0, "", "")):
            kwargs = ({"side_effect": effect} if isinstance(effect, Exception)
                      else {"return_value": effect})
            with mock.patch.object(git_safe, "run_git", **kwargs):
                with self.assertRaises(host_fence.HostFenceError) as ctx:
                    host_fence._git_remote_exec_denies()
            self.assertEqual(ctx.exception.envelope["reason"], "fence_path_rejected")



if __name__ == "__main__":
    unittest.main()
