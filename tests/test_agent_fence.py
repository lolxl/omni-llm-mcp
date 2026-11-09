#!/usr/bin/env python3
# no-opencc
"""Agent-lane kernel fence for claude/pi (issue #54).

Enforcement cases that need a live Seatbelt are Darwin-only
(``skipIf(not kernel_sandbox.available())``). The module still imports on
Linux. Fake vendor binaries are injected through ``OMNI_LLM_CLAUDE_BIN`` /
``OMNI_LLM_PI_BIN``, never through PATH.
"""
from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
import tempfile
import textwrap
import unittest
import uuid
from unittest import mock

from providers import agent_lane, dispatch, jobs, kernel_sandbox
from providers.config import ENV_DEPTH
from providers.logutil import log_event as real_log_event
from providers.profiles import child_env_for_profile, resolve_profile
from tests import offline_home

_DARWIN_FENCE = kernel_sandbox.available()
_SKIP_NO_FENCE = "sandbox-exec not available at /usr/bin/sandbox-exec"


def _run(coro):
    return asyncio.run(coro)


def _write_exec(path, body):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def _three_step_script():
    return textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import os, sys
        pid = os.getpid()
        rcs = []
        try:
            open("./inside.txt", "w", encoding="utf-8").write("INSIDE\\n")
            rcs.append(0)
        except OSError as exc:
            rcs.append(exc.errno)
        escape = os.environ.get("OMNI54_ESCAPE", "/tmp/omni54-%d.txt" % pid)
        try:
            open(escape, "w", encoding="utf-8").write("ESCAPE\\n")
            rcs.append(0)
        except OSError as exc:
            rcs.append(exc.errno)
        canary = os.environ.get("OMNI54_CANARY", "")
        try:
            data = open(canary, "r", encoding="utf-8").read()
            sys.stdout.write(data)
            rcs.append(0)
        except OSError as exc:
            rcs.append(exc.errno)
        sys.stdout.write("RCS %d %d %d\\n" % tuple(rcs))
        """
    )


class _FakeRunCli:
    def __init__(self, returncode=0, stdout="AGENT OK", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    async def __call__(self, argv=None, **kwargs):
        self.calls.append((list(argv or []), dict(kwargs)))
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": False,
        }

    @property
    def argv(self):
        return self.calls[0][0]


def _plant_home_canary():
    home = kernel_sandbox.protected_home()
    path = os.path.join(home, "omni54-canary-%d" % os.getpid())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("OMNI54_SECRET_CANARY\n")
    return path


class TestAgentFenceImports(unittest.TestCase):
    def test_module_imports_agent_fence_paths(self):
        self.assertTrue(callable(kernel_sandbox.agent_fence_paths))
        self.assertTrue(callable(kernel_sandbox.agent_fence_policy))
        self.assertTrue(callable(agent_lane._apply_kernel_fence))

    def test_agent_child_env_silences_operator_notify_sink(self):
        env = kernel_sandbox.agent_child_env({
            "PATH": "/bin",
            "OMNI_LLM_NOTIFY_CMD": "/tmp/operator-sink",
        })
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", env)
        self.assertEqual(env.get("OMNI_LLM_NOTIFY_QUIET"), "1")


class TestAgentFenceHermetic(unittest.TestCase):
    def setUp(self):
        self.workdir = os.path.realpath(tempfile.mkdtemp(prefix="omni54_wd_"))
        self.bindir = os.path.realpath(tempfile.mkdtemp(prefix="omni54_bin_"))
        self.addCleanup(_rmtree, self.workdir)
        self.addCleanup(_rmtree, self.bindir)
        self.claude = os.path.join(self.bindir, "claude")
        self.pi = os.path.join(self.bindir, "pi")
        _write_exec(self.claude, "#!/bin/sh\necho CLAUDE_FAKE\n")
        _write_exec(self.pi, "#!/bin/sh\necho PI_FAKE\n")
        self.old_env = {
            kernel_sandbox.ENV_CLAUDE_BIN: os.environ.get(
                kernel_sandbox.ENV_CLAUDE_BIN
            ),
            kernel_sandbox.ENV_PI_BIN: os.environ.get(kernel_sandbox.ENV_PI_BIN),
            kernel_sandbox.ENV_AGENT_FENCE: os.environ.get(
                kernel_sandbox.ENV_AGENT_FENCE
            ),
            kernel_sandbox.ENV_AGENT_UNFENCED: os.environ.get(
                kernel_sandbox.ENV_AGENT_UNFENCED
            ),
        }
        os.environ[kernel_sandbox.ENV_CLAUDE_BIN] = self.claude
        os.environ[kernel_sandbox.ENV_PI_BIN] = self.pi
        self.old_env[ENV_DEPTH] = os.environ.pop(ENV_DEPTH, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _run_claude(self, fake, **kwargs):
        with mock.patch.object(agent_lane, "run_cli", fake), \
             mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            return _run(agent_lane.run_agent(
                task="do it",
                model="claude/sonnet",
                workdir=self.workdir,
                timeout_s=30,
                **kwargs,
            ))

    def test_run_agent_child_env_silences_operator_notify_sink(self):
        fake = _FakeRunCli()
        with mock.patch.dict(os.environ, {
            "OMNI_LLM_NOTIFY_CMD": "/tmp/operator-sink",
        }, clear=False):
            env = self._run_claude(fake, fence_requested=False)
        self.assertTrue(env.get("ok"), env)
        child = fake.calls[0][1].get("env") or {}
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", child)
        self.assertEqual(child.get("OMNI_LLM_NOTIFY_QUIET"), "1")

    def test_c_setting_sources_project_and_not_bare(self):
        fake = _FakeRunCli()
        env = self._run_claude(fake, fence_requested=False)
        self.assertTrue(env.get("ok"), env)
        argv = fake.argv
        self.assertIn("--setting-sources", argv)
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project")
        self.assertNotIn("--bare", argv)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_j_switch_off_unwrapped_fence_none_logs(self):
        fake = _FakeRunCli()
        events = []

        def _log(event, **fields):
            events.append((event, fields))
            return real_log_event(event, **fields)

        with mock.patch.object(agent_lane, "log_event", _log):
            env = self._run_claude(fake, fence_requested=False)
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("fence"), "none")
        detail = env.get("fence_detail") or {}
        self.assertFalse(detail.get("read_wall"))
        self.assertFalse(detail.get("workdir_writable"))
        self.assertFalse(detail.get("vendor_state_writable"))
        self.assertNotEqual(fake.argv[0], kernel_sandbox.SANDBOX_EXEC)
        self.assertTrue(
            any(evt == "agent_unfenced" and fields.get("provider") == "claude"
                for evt, fields in events),
            events,
        )

    def test_k_switch_on_available_false_loud_no_spawn(self):
        fake = _FakeRunCli()
        with mock.patch.object(kernel_sandbox, "available", return_value=False):
            env = self._run_claude(fake, fence_requested=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "fence_unavailable")
        self.assertEqual(env.get("fence"), "none")
        self.assertEqual(fake.calls, [])

    def test_e_verify_fence_refusal_no_spawn(self):
        fake = _FakeRunCli()
        refusal = {
            "ok": False,
            "reason": "kernel_fence_unverified",
            "detail": "probe failed",
        }
        policy = {
            "wrap_kw": dict(
                read_wall=True, extra_read_paths=(), extra_write_paths=(),
            ),
            "tmpdir": None,
            "vendor_state_writable": False,
            "binary": self.claude,
        }
        with mock.patch.object(kernel_sandbox, "available", return_value=True), \
             mock.patch.object(
                 kernel_sandbox, "agent_fence_policy", return_value=policy,
             ), \
             mock.patch.object(kernel_sandbox, "verify_fence", return_value=refusal):
            env = self._run_claude(fake, fence_requested=True)
        self.assertFalse(env.get("ok"), env)
        self.assertEqual(env.get("reason"), "kernel_fence_unverified")
        self.assertEqual(env.get("fence"), "none")
        self.assertEqual(fake.calls, [])

    def test_g_backends_snapshot_agent_fence_values(self):
        snap = dispatch.backends_snapshot()
        seatbelt = "seatbelt-fs" if kernel_sandbox.available() else "none"
        self.assertEqual(snap["claude"]["agent_fence"], seatbelt)
        self.assertEqual(snap["pi"]["agent_fence"], seatbelt)
        # codex advertises its WRITE lane (issue #94): seatbelt-fs where the
        # mechanism exists, vendor where it does not -- never "none", because
        # its degraded argv still carries --sandbox workspace-write. grok is
        # vendor on every host.
        self.assertEqual(snap["codex"]["agent_fence"],
                         "seatbelt-fs" if seatbelt == "seatbelt-fs" else "vendor")
        self.assertEqual(snap["grok"]["agent_fence"], "vendor")
        for name in ("agy", "cursor", "openai", "ollama", "lmstudio"):
            self.assertEqual(snap[name]["agent_fence"], "none", name)
        self.assertIn(snap["claude"]["agent_fence_switch"], ("on", "off"))

    def test_h_user_settings_skipped_on_both_lanes(self):
        fake = _FakeRunCli()
        self._run_claude(fake, fence_requested=False)
        argv = fake.argv
        self.assertIn("--setting-sources", argv)
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project")
        self.assertNotIn("--bare", argv)

    def test_a7_verify_and_wrap_same_profile(self):
        wrap_kw = dict(
            read_wall=True,
            extra_read_paths=[],
            extra_write_paths=[],
        )
        built = kernel_sandbox.build_profile(
            self.workdir, read_only=False, **wrap_kw,
        )
        wrapped = kernel_sandbox.wrap_argv(
            ["/bin/true"], self.workdir, False, **wrap_kw,
        )
        if not kernel_sandbox.available():
            self.assertEqual(wrapped, ["/bin/true"])
            return
        self.assertIn("-p", wrapped)
        self.assertEqual(wrapped[wrapped.index("-p") + 1], built)

    def test_grok_policy_denies_global_instruction_reads(self):
        from providers import host_fence
        grok_home = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_home_"))
        outside = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_outside_"))
        self.addCleanup(_rmtree, grok_home)
        self.addCleanup(_rmtree, outside)
        target = os.path.join(outside, "target.md")
        for path in (os.path.join(grok_home, "aGeNtS.md"), target):
            with open(path, "w") as handle:
                handle.write("synthetic fixture")
        os.symlink(target, os.path.join(grok_home, "CLAUDE.md"))
        os.mkdir(os.path.join(grok_home, "rules"))
        policy = kernel_sandbox.agent_fence_policy(
            "grok", workdir=self.workdir, read_only=True, config_dir=grok_home,
        )
        if isinstance(policy, dict) and policy.get("ok") is False:
            self.skipTest(policy.get("detail") or "agent_fence_policy refused")
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        denied = set(wrap_kw["extra_deny_read"])
        home = kernel_sandbox.protected_home()
        for row in (
                ("literal", os.path.join(grok_home, "aGeNtS.md")),   # on-disk case
                ("literal", os.path.join(grok_home, "AGENTS.md")),   # probed name
                ("literal", os.path.join(grok_home, "CLAUDE.md")),   # symlink name
                ("literal", target),                                 # its target
                ("subpath", os.path.join(grok_home, "rules")),
                ("literal", os.path.join(home, ".claude", "CLAUDE.md")),
                ("subpath", os.path.join(home, ".cursor", "rules"))):
            self.assertIn(row, denied)
        self.assertNotIn(("subpath", grok_home), denied)   # config/auth stay readable
        self.assertTrue(wrap_kw["read_wall"])
        self.assertTrue(host_fence._grok_instruction_wall_held(wrap_kw, grok_home))

    def test_grok_tmp_allow_is_pid_bound_and_bounded(self):
        from providers import host_fence
        import re
        # Regex metacharacters in the root must be quoted, not interpreted.
        grok_home = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok.h+(x)[w]{1}|^$*?_"))
        self.addCleanup(_rmtree, grok_home)
        policy = kernel_sandbox.agent_fence_policy(
            "grok", workdir=self.workdir, read_only=True, config_dir=grok_home,
        )
        # A refusal here would hide every assertion below: fail, never skip.
        self.assertFalse(isinstance(policy, dict) and policy.get("ok") is False,
                         policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        kind = kernel_sandbox.GROK_TMP_KIND
        rows = [e for e in wrap_kw["extra_post_write"] if e[0] == kind]
        self.assertEqual(rows, [(kind, grok_home)])
        profile = kernel_sandbox.build_profile(self.workdir, True, **{
            k: v for k, v in wrap_kw.items() if k.startswith("extra_")})
        self.assertIn('(param "%s")' % kernel_sandbox.GROK_PID_PARAM, profile)
        # Rebuild the regex the profile composes, with T<i> as emitted.
        params = kernel_sandbox.profile_params(self.workdir, True, **{
            k: v for k, v in wrap_kw.items() if k.startswith("extra_")})
        index = wrap_kw["extra_post_write"].index(rows[0])
        self.assertIn('(regex (string-append "^" (param "T%d") "/\\\\." '
                      '(param "%s") "\\\\.[0-9]+\\\\.tmp$"))'
                      % (index, kernel_sandbox.GROK_PID_PARAM), profile)
        rx = re.compile("^" + params["T%d" % index] + r"/\.123\.[0-9]+\.tmp$")
        self.assertTrue(rx.search(os.path.join(grok_home, ".123.0.tmp")))
        twin = grok_home.replace(".h+(x)", "Xhh(x)").replace("(x)", "x")
        for path in (os.path.join(grok_home, ".124.0.tmp"),
                     os.path.join(grok_home, ".123.0.tmp.x"),
                     os.path.join(grok_home, ".123.0.tmp", "child"),
                     os.path.join(grok_home, "sub", ".123.0.tmp"),
                     os.path.join(twin, ".123.0.tmp")):
            self.assertFalse(rx.search(path), path)
        argv = kernel_sandbox.wrap_argv(["/bin/true"], self.workdir, True, **wrap_kw)
        if kernel_sandbox.available():
            self.assertEqual(argv[3:5], ["-D", "%s=%s" % (
                kernel_sandbox.GROK_PID_PARAM, kernel_sandbox.GROK_PID_PLACEHOLDER)])
        self.assertEqual(kernel_sandbox._write_entry_bound(*rows[0]),
                         ("subpath", grok_home))
        self.assertIsNone(kernel_sandbox._write_entry_bound("regex", "^/.*$"))
        # Overlap checks judge the row by its bound instead of refusing it.
        self.assertIsNone(host_fence.settings_base_exposed(
            os.path.join(self.workdir, "elsewhere"), rows))
        self.assertIsNotNone(host_fence.settings_base_exposed(
            os.path.join(grok_home, ".1.0.tmp", "base"), rows))
        self.assertIsNotNone(host_fence.settings_base_exposed(
            "/", [("regex", "^/.*$")]))
        host_fence._post_write_overlaps_repo(
            rows, [os.path.join(self.workdir, ".git")])
        with self.assertRaises(host_fence.HostFenceError):
            host_fence._post_write_overlaps_repo(rows, [grok_home])
        _r, writes = kernel_sandbox.agent_fence_paths(
            "grok", workdir=self.workdir, read_only=True, config_dir=grok_home,
        )
        self.assertNotIn(grok_home, writes)

    def test_r2_2_unenumerable_instruction_roots_refuse(self):
        outside = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_outside_"))
        self.addCleanup(_rmtree, outside)
        target = os.path.join(outside, "target.md")
        with open(target, "w") as handle:
            handle.write("synthetic fixture")

        def fresh_root():
            root = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_home_"))
            self.addCleanup(_rmtree, root)
            return root

        # Unlistable root (search-only) hiding a symlink to an outside file.
        root = fresh_root()
        os.symlink(target, os.path.join(root, "AGENTS.md"))
        os.chmod(root, 0o111)
        self.addCleanup(os.chmod, root, 0o700)
        _rows, why = kernel_sandbox._grok_instruction_deny_read(root)
        self.assertIsNotNone(why)
        # Unreadable directory inside rules/.
        root = fresh_root()
        hidden = os.path.join(root, "rules", "hidden")
        os.makedirs(hidden)
        os.symlink(target, os.path.join(hidden, "topic.md"))
        os.chmod(hidden, 0o111)
        self.addCleanup(os.chmod, hidden, 0o700)
        _rows, why = kernel_sandbox._grok_instruction_deny_read(root)
        self.assertIsNotNone(why)
        # Any listdir failure other than a missing root refuses.
        root = fresh_root()
        with mock.patch.object(kernel_sandbox.os, "listdir",
                               side_effect=PermissionError(13, "denied")):
            _rows, why = kernel_sandbox._grok_instruction_deny_read(root)
        self.assertIsNotNone(why)
        # A missing root is simply empty.
        rows, why = kernel_sandbox._grok_instruction_deny_read(
            os.path.join(outside, "absent"))
        self.assertIsNone(why)
        self.assertIn(("literal", os.path.join(outside, "absent", "AGENTS.md")), rows)

    def test_f_read_only_write_allow_excludes_workdir(self):
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=self.workdir, read_only=True,
        )
        if isinstance(policy, dict) and policy.get("ok") is False:
            self.skipTest(policy.get("detail") or "agent_fence_policy refused")
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        params = kernel_sandbox.profile_params(
            self.workdir, True, **wrap_kw,
        )
        self.assertNotIn(os.path.realpath(self.workdir),
                         [v for k, v in params.items() if k.startswith("P")])
        self.assertEqual(params["Q0"], os.path.realpath(self.workdir))

    def test_claude_session_dir_encoding_pinned(self):
        encoded = kernel_sandbox.claude_encoded_workdir(self.workdir)
        self.assertTrue(encoded.startswith("-"))
        self.assertNotIn("/", encoded)
        session = kernel_sandbox.claude_session_dir(self.workdir)
        self.assertEqual(
            session,
            os.path.join(
                kernel_sandbox.protected_home(),
                ".claude", "projects", encoded,
            ),
        )

    def test_claude_session_dir_uses_profile_root(self):
        profile = tempfile.mkdtemp(prefix="omni72_sess_")
        self.addCleanup(_rmtree, profile)
        encoded = kernel_sandbox.claude_encoded_workdir(self.workdir)
        session = kernel_sandbox.claude_session_dir(
            self.workdir, config_dir=profile)
        self.assertEqual(
            session,
            os.path.join(os.path.realpath(profile), "projects", encoded),
        )

    def test_claude_profile_fence_admits_root_not_default_login(self):
        profile = tempfile.mkdtemp(prefix="omni72_prof_")
        self.addCleanup(_rmtree, profile)
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=self.workdir, read_only=False, config_dir=profile,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        rp = os.path.realpath(profile)
        self.assertIn(rp, wrap_kw["extra_write_paths"])
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        self.assertNotIn(default_root, wrap_kw["extra_read_paths"])
        session = kernel_sandbox.claude_session_dir(
            self.workdir, config_dir=profile)
        self.assertTrue(any(p == session for _k, p in wrap_kw["extra_post_write"]),
                        wrap_kw["extra_post_write"])
        self.assertTrue(any(p == os.path.join(rp, "projects")
                            for _k, p in wrap_kw["extra_deny_read"]),
                        wrap_kw["extra_deny_read"])
        self.assertTrue(policy.get("vendor_state_writable"))

    def test_codex_policy_allows_vendor_root_and_denies_instruction_surface(self):
        """#94 piece 3: ~/.codex whole, minus what steers the NEXT run."""
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.workdir, read_only=False,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = policy["wrap_kw"]
        self.assertTrue(wrap_kw["read_wall"])
        home = kernel_sandbox.protected_home()
        codex_root = os.path.realpath(os.path.join(home, ".codex"))
        if not os.path.isdir(codex_root):
            self.skipTest("no ~/.codex on this machine")
        self.assertIn(codex_root, wrap_kw["extra_read_paths"])
        self.assertTrue(policy["vendor_state_writable"])
        # The tree is denied WHOLE and only state leaves are re-allowed:
        # enumerating the dangerous half does not work (review found four
        # missing members in one pass, AGENTS.override.md among them, which
        # takes PRECEDENCE over AGENTS.md). Anything new fails closed.
        denied = {path for _kind, path in wrap_kw["extra_deny_write"]}
        self.assertIn(codex_root, denied,
                      "the codex home must be denied whole, not name by name")
        allowed = {path for _kind, path in wrap_kw["extra_post_write"]}
        for leaf in ("sessions", "shell_snapshots", "thread-writer-locks"):
            self.assertIn(os.path.join(codex_root, leaf), allowed,
                          "%s is state codex must keep writing" % leaf)
        # auth.json stays writable: codex rewrites its own credential store
        # on token refresh, the same accepted exposure as pi's.
        self.assertIn(os.path.join(codex_root, "auth.json"), allowed)
        # Nothing that steers a LATER run is re-allowed.
        for leaf in ("config.toml", "hooks.json", "AGENTS.md",
                     "AGENTS.override.md", "agents", "skills", "plugins",
                     "rules", ".tmp"):
            self.assertNotIn(os.path.join(codex_root, leaf), allowed,
                             "%s must not be re-allowed" % leaf)
        # TMPDIR is not re-pinned for codex -- the child keeps the one the
        # write allow-set was built from.
        self.assertIsNone(policy["tmpdir"])

    def test_codex_read_only_policy_drops_workdir_from_write_set(self):
        # RO excludes cwd from allows and explicitly denies it even under tmp.
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.workdir, read_only=True,
        )
        if isinstance(policy, dict) and policy.get("ok") is False:
            self.skipTest(policy.get("detail") or "agent_fence_policy refused")
        params = kernel_sandbox.profile_params(
            self.workdir, True, **kernel_sandbox.wrap_kwargs_from_policy(policy),
        )
        self.assertNotIn(os.path.realpath(self.workdir),
                         [v for k, v in params.items() if k.startswith("P")])
        self.assertEqual(params["Q0"], os.path.realpath(self.workdir))

    def test_codex_profile_root_admitted_and_carries_the_deny_rows(self):
        root = os.path.realpath(tempfile.mkdtemp(prefix="omni94_codexhome_"))
        self.addCleanup(_rmtree, root)
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.workdir, read_only=False, config_dir=root,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = policy["wrap_kw"]
        self.assertIn(root, wrap_kw["extra_write_paths"])
        self.assertIn(root, wrap_kw["extra_read_paths"])
        home = kernel_sandbox.protected_home()
        default_root = os.path.realpath(os.path.join(home, ".codex"))
        # Assert on the EFFECTIVE profile, not on collected extras:
        # _write_allow_paths adds ~/.codex unconditionally (the consult
        # lane's contract, shared through build_profile), so checking the
        # extras list would certify a property that does not hold. What
        # makes "a profile replaces ~/.codex" true is the deny clause.
        params = kernel_sandbox.profile_params(self.workdir, False, **wrap_kw)
        denied = {v for kk, v in params.items() if kk.startswith("W")}
        self.assertIn(default_root, denied,
                      "a profile must deny the DEFAULT account's ~/.codex, "
                      "or a `work` run reads and writes the default auth.json")
        denied_read = {v for kk, v in params.items() if kk.startswith("X")}
        self.assertIn(default_root, denied_read)
        denied = {path for _kind, path in wrap_kw["extra_deny_write"]}
        self.assertIn(os.path.join(root, "config.toml"), denied,
                      "a profile must not lose the carve-outs")
        self.assertIn(os.path.join(root, "hooks.json"), denied)

    def test_codex_profile_inside_home_is_refused_unless_dotdir(self):
        """Adversarial CODEX_HOME admission (issue #102's rule, re-applied).

        The fence resolves the caller's value a SECOND time, so a directory
        admitted by the producer and swapped afterwards still has to pass.
        """
        home = kernel_sandbox.protected_home()
        for bad in (home,
                    os.path.join(home, ".ssh"),
                    os.path.join(home, "Library", "Keychains"),
                    "/"):
            with self.subTest(bad=bad):
                policy = kernel_sandbox.agent_fence_policy(
                    "codex", workdir=self.workdir, read_only=False,
                    config_dir=bad,
                )
                self.assertEqual(policy.get("ok"), False, policy)
                self.assertEqual(policy.get("reason"), "fence_path_rejected")

    def test_codex_profile_dedicated_dotdir_inside_home_is_admitted(self):
        home = kernel_sandbox.protected_home()
        root = os.path.join(home, ".codex-omni94test")
        if os.path.lexists(root):
            self.skipTest("%s already exists" % root)
        os.mkdir(root, 0o700)
        self.addCleanup(_rmtree, root)
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.workdir, read_only=False, config_dir=root,
        )
        self.assertNotIn("ok", policy, policy)
        self.assertIn(os.path.realpath(root),
                      policy["wrap_kw"]["extra_write_paths"])

    @mock.patch("platform.system", new=lambda: "Linux")
    def test_pi_setting_sources_not_required(self):
        fake = _FakeRunCli()
        with mock.patch.object(agent_lane, "run_cli", fake), \
             mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            env = _run(agent_lane.run_agent(
                task="do it",
                model="pi/xai/grok-4.6",
                workdir=self.workdir,
                timeout_s=30,
                fence_requested=False,
            ))
        self.assertTrue(env.get("ok"), env)
        self.assertNotIn("--setting-sources", fake.argv)
        self.assertEqual(env.get("fence"), "none")

    def test_profile_ring_child_env_and_fence_params_isolate_winner(self):
        root = tempfile.mkdtemp(prefix="omni72_ring_h_")
        self.addCleanup(_rmtree, root)
        dirs = {}
        for name in ("a", "b", "c"):
            path = os.path.join(root, name)
            os.makedirs(path)
            with open(os.path.join(path, "canary"), "w", encoding="utf-8") as handle:
                handle.write("CANARY-%s\n" % name.upper())
            dirs[name] = os.path.realpath(path)
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        parent = {
            "HOME": "/x",
            "PATH": "/bin",
            "CLAUDE_CONFIG_DIR_A": dirs["a"],
            "CLAUDE_CONFIG_DIR_B": dirs["b"],
            "CLAUDE_CONFIG_DIR_C": dirs["c"],
        }
        with mock.patch.dict(os.environ, {
            "CLAUDE_CONFIG_DIR_A": dirs["a"],
            "CLAUDE_CONFIG_DIR_B": dirs["b"],
            "CLAUDE_CONFIG_DIR_C": dirs["c"],
        }, clear=False):
            for winner in ("a", "b", "c"):
                overrides, err = resolve_profile("claude", winner)
                self.assertIsNone(err, err)
                child = child_env_for_profile(parent, overrides)
                self.assertEqual(child["CLAUDE_CONFIG_DIR"], dirs[winner])
                for name in ("A", "B", "C"):
                    self.assertNotIn("CLAUDE_CONFIG_DIR_%s" % name, child)
                policy = kernel_sandbox.agent_fence_policy(
                    "claude", workdir=self.workdir, read_only=False,
                    config_dir=dirs[winner],
                )
                self.assertNotIn("ok", policy, policy)
                wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
                params = kernel_sandbox.profile_params(
                    self.workdir, False, **wrap_kw)
                blob = " ".join(str(v) for v in params.values())
                self.assertIn(dirs[winner], blob)
                self.assertIn(dirs[winner], wrap_kw["extra_write_paths"])
                self.assertNotIn(default_root, wrap_kw["extra_read_paths"])
                self.assertNotIn(default_root, wrap_kw["extra_write_paths"])
                self.assertNotIn("~/.claude", blob)
            scrubbed = child_env_for_profile(
                {**parent, "CLAUDE_CONFIG_DIR": "/login",
                 "OMNI_LLM_PROFILES_CLAUDE": "default,a,b,c"},
                {},
                scrub_siblings=True,
            )
            self.assertEqual(scrubbed.get("CLAUDE_CONFIG_DIR"), "/login")
            for name in ("A", "B", "C"):
                self.assertNotIn("CLAUDE_CONFIG_DIR_%s" % name, scrubbed)

    def _synthetic_codex_home(self):
        fake_home = os.path.realpath(tempfile.mkdtemp(prefix="omni108_"))
        self.addCleanup(_rmtree, fake_home)
        codex_root = os.path.join(fake_home, ".codex")
        os.makedirs(codex_root)
        return fake_home, codex_root

    def test_codex_post_allow_refuses_a_state_dir_symlink_to_the_home(self):
        """A post-allowed leaf that is a symlink to the codex home refuses.

        ``log`` is in the state-dir list and is created on demand, so a
        fenced run may plant it as a symlink. Realpath'd T would then
        name the home itself (last-match-wins). Lexical T would not, but
        the spawn still refuses: a post-allow candidate must not be a
        symlink. The islink check is the unique failing predicate here
        (realpath of the link is the codex home, which is inside itself
        and is not a widening root).
        """
        fake_home, codex_root = self._synthetic_codex_home()
        log_path = os.path.join(codex_root, "log")
        os.symlink(codex_root, log_path)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertEqual(policy.get("ok"), False, policy)
            self.assertEqual(policy.get("reason"), "fence_path_rejected")
            detail = policy.get("detail") or ""
            self.assertIn(log_path, detail)
            self.assertNotIn("wrap_kw", policy)
            collected = kernel_sandbox._collect_agent_fence(
                "codex", workdir=self.workdir, read_only=False,
            )
            for kind, path in collected["extra_post_write"]:
                self.assertNotEqual(path, fake_home, (kind, path))
                self.assertNotEqual(path, codex_root, (kind, path))
                self.assertFalse(
                    kind == "subpath" and path in (fake_home, codex_root),
                    (kind, path),
                )

    def test_codex_post_allow_refuses_a_sqlite_symlink_onto_a_denied_file(self):
        """A ``*.sqlite`` name that is a symlink onto config.toml refuses."""
        fake_home, codex_root = self._synthetic_codex_home()
        target = os.path.join(codex_root, "config.toml")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")
        link = os.path.join(codex_root, "x.sqlite")
        os.symlink(target, link)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertEqual(policy.get("ok"), False, policy)
            self.assertEqual(policy.get("reason"), "fence_path_rejected")
            self.assertIn(link, policy.get("detail") or "")
            self.assertNotIn("wrap_kw", policy)

    def test_codex_post_allow_derives_wal_shm_and_does_not_name_the_home(self):
        """Existing ``.sqlite`` yields literal ``-wal``/``-shm`` T entries.

        Also the policy-level live-acceptance stand-in: no T path equals
        the synthetic home, and ``.tmp`` is absent from T.
        """
        fake_home, codex_root = self._synthetic_codex_home()
        sqlite_name = "memories_1.sqlite"
        with open(os.path.join(codex_root, sqlite_name), "w",
                  encoding="utf-8") as fh:
            fh.write("x")
        self.assertFalse(
            os.path.lexists(os.path.join(codex_root, sqlite_name + "-wal")))
        self.assertFalse(
            os.path.lexists(os.path.join(codex_root, sqlite_name + "-shm")))
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            t_entries = list(policy["wrap_kw"]["extra_post_write"])
            t_paths = [path for _kind, path in t_entries]
            literals = {path for kind, path in t_entries if kind == "literal"}
            sqlite_path = os.path.abspath(
                os.path.join(codex_root, sqlite_name))
            self.assertIn(sqlite_path, literals)
            self.assertIn(sqlite_path + "-wal", literals)
            self.assertIn(sqlite_path + "-shm", literals)
            self.assertNotIn(fake_home, t_paths)
            self.assertNotIn(codex_root, t_paths)
            tmp_dir = os.path.abspath(os.path.join(codex_root, ".tmp"))
            for path in t_paths:
                self.assertNotEqual(path, tmp_dir)
                self.assertFalse(path.startswith(tmp_dir + os.sep), path)
                self.assertNotIn(".tmp", path.split(os.sep))

    def test_codex_dot_tmp_plugins_is_not_covered_by_post_allow(self):
        """``.tmp/plugins/<uuid>`` is not a T path and no T subpath covers it."""
        fake_home, codex_root = self._synthetic_codex_home()
        probe = os.path.join(
            codex_root, ".tmp", "plugins", uuid.uuid4().hex,
        )
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            t_entries = list(policy["wrap_kw"]["extra_post_write"])
            probe_abs = os.path.abspath(probe)
            tmp_dir = os.path.abspath(os.path.join(codex_root, ".tmp"))
            for kind, path in t_entries:
                path_abs = os.path.abspath(path)
                self.assertNotEqual(path_abs, probe_abs)
                self.assertNotEqual(path_abs, tmp_dir)
                if kind == "subpath":
                    self.assertFalse(
                        probe_abs == path_abs
                        or probe_abs.startswith(path_abs + os.sep),
                        (kind, path),
                    )

    def test_codex_named_config_toml_is_not_post_allowed_for_write(self):
        """``<uuid>.config.toml`` is not state, so it is not in T.

        Deny-READ of that file is not the codex-lane contract: this lane
        re-allows the whole home for read (the vendor must load config).
        #109's deny-read of ``~/.codex`` is the other-lanes property.
        """
        fake_home, codex_root = self._synthetic_codex_home()
        name = "%s.config.toml" % uuid.uuid4().hex
        steering = os.path.join(codex_root, name)
        with open(steering, "w", encoding="utf-8") as fh:
            fh.write("x")
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            wrap_kw = policy["wrap_kw"]
            t_paths = {os.path.abspath(p)
                       for _k, p in wrap_kw["extra_post_write"]}
            self.assertNotIn(os.path.abspath(steering), t_paths)
            x_paths = {os.path.abspath(p)
                       for _k, p in wrap_kw["extra_deny_read"]}
            self.assertNotIn(os.path.abspath(steering), x_paths)
            self.assertIn(
                os.path.abspath(codex_root),
                {os.path.abspath(p) for p in wrap_kw["extra_read_paths"]},
            )

    def test_codex_post_allow_refuses_config_dir_through_a_symlinked_parent(self):
        """CODEX_HOME through a symlinked ancestor is a LOUD refusal.

        abspath keeps the symlink spelling; realpath does not. T clauses
        would then name a path Seatbelt never sees.
        """
        fake_home, _codex_root = self._synthetic_codex_home()
        base = os.path.realpath(tempfile.mkdtemp(prefix="omni108_h2_"))
        self.addCleanup(_rmtree, base)
        real = os.path.join(base, "real")
        os.makedirs(real)
        via = os.path.join(base, "via")
        os.symlink(base, via)
        config_dir = os.path.join(via, "real")
        self.assertFalse(os.path.islink(config_dir))
        lexical = os.path.abspath(config_dir)
        resolved = os.path.realpath(config_dir)
        self.assertNotEqual(lexical, resolved)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
                config_dir=config_dir,
            )
        self.assertEqual(policy.get("ok"), False, policy)
        self.assertEqual(policy.get("reason"), "fence_path_rejected")
        detail = policy.get("detail") or ""
        self.assertIn(lexical, detail)
        self.assertIn(resolved, detail)
        self.assertNotIn("wrap_kw", policy)

    def test_codex_policy_emits_file_link_deny_on_the_home(self):
        """Policy-level pin: the profile denies file-link on the codex home."""
        fake_home, codex_root = self._synthetic_codex_home()
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = policy["wrap_kw"]
        self.assertIn(
            os.path.realpath(codex_root),
            {path for _kind, path in wrap_kw["extra_deny_link"]},
        )
        profile = kernel_sandbox.build_profile(
            self.workdir, False, **wrap_kw,
        )
        self.assertIn("(deny file-link", profile)
        self.assertGreater(
            profile.index("(deny file-link"),
            profile.index("(allow default)"),
        )

    def test_apply_kernel_fence_returns_refusal_for_planted_codex_symlink(self):
        """Spawn decision: planted post-allow symlink -> refusal, argv unwrapped."""
        fake_home, codex_root = self._synthetic_codex_home()
        log_path = os.path.join(codex_root, "log")
        os.symlink(codex_root, log_path)
        argv = ["/fake/codex", "exec", "-"]
        env = {"PATH": "/usr/bin"}
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            out_argv, _out_env, fence, _detail, refusal = (
                agent_lane._apply_kernel_fence(
                    "codex", argv, env, self.workdir, False, True,
                )
            )
        self.assertIs(out_argv, argv)
        self.assertFalse(kernel_sandbox.is_wrapped(out_argv))
        self.assertEqual(fence, "none")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.get("ok"), False)
        self.assertEqual(refusal.get("reason"), "fence_path_rejected")
        self.assertIn(log_path, refusal.get("detail") or "")


@unittest.skipUnless(_DARWIN_FENCE, _SKIP_NO_FENCE)
class TestAgentFenceLive(unittest.TestCase):
    def setUp(self):
        self.workdir = os.path.realpath(tempfile.mkdtemp(prefix="omni54_live_"))
        self.bindir = os.path.realpath(tempfile.mkdtemp(prefix="omni54_lbin_"))
        self.addCleanup(_rmtree, self.workdir)
        self.addCleanup(_rmtree, self.bindir)
        self.claude = os.path.join(self.bindir, "claude")
        self.pi = os.path.join(self.bindir, "pi")
        self.cursor = os.path.join(self.bindir, "cursor-agent")
        _write_exec(self.claude, _three_step_script())
        _write_exec(self.pi, _three_step_script())
        _write_exec(self.cursor, _three_step_script())
        self.canary = None
        self.old_env = {
            kernel_sandbox.ENV_CLAUDE_BIN: os.environ.get(
                kernel_sandbox.ENV_CLAUDE_BIN
            ),
            kernel_sandbox.ENV_PI_BIN: os.environ.get(kernel_sandbox.ENV_PI_BIN),
            kernel_sandbox.ENV_CURSOR_BIN: os.environ.get(
                kernel_sandbox.ENV_CURSOR_BIN
            ),
            "OMNI54_CANARY": os.environ.get("OMNI54_CANARY"),
        }
        os.environ[kernel_sandbox.ENV_CLAUDE_BIN] = self.claude
        os.environ[kernel_sandbox.ENV_PI_BIN] = self.pi
        os.environ[kernel_sandbox.ENV_CURSOR_BIN] = self.cursor
        if self.canary:
            os.environ["OMNI54_CANARY"] = self.canary
        self.escape = "/tmp/omni54-%d.txt" % os.getpid()
        os.environ["OMNI54_ESCAPE"] = self.escape
        self.old_env["OMNI54_ESCAPE"] = None
        self.old_env[ENV_DEPTH] = os.environ.pop(ENV_DEPTH, None)
        self.addCleanup(self._restore_env)
        self.addCleanup(self._cleanup_escape)
        self.addCleanup(self._cleanup_canary)

    def _restore_env(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _cleanup_canary(self):
        if self.canary:
            try:
                os.remove(self.canary)
            except OSError:
                pass

    def _cleanup_escape(self):
        path = getattr(self, "escape", None)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def _run(self, model, **kwargs):
        with mock.patch.object(agent_lane, "_workdir_fence", return_value=None):
            return _run(agent_lane.run_agent(
                task="three steps",
                model=model,
                workdir=self.workdir,
                timeout_s=30,
                **kwargs,
            ))

    def _require_canary(self):
        if self.canary:
            return
        try:
            self.canary = _plant_home_canary()
        except OSError:
            self.skipTest("cannot plant home canary under protected_home()")
        os.environ["OMNI54_CANARY"] = self.canary

    def _assert_enforced(self, env, provider):
        self.assertTrue(env.get("ok"), env)
        self.assertEqual(env.get("fence"), "seatbelt-fs")
        inside = os.path.join(self.workdir, "inside.txt")
        self.assertTrue(os.path.isfile(inside), "inside.txt missing")
        with open(inside, encoding="utf-8") as handle:
            self.assertIn("INSIDE", handle.read())
        self.assertFalse(
            os.path.exists(self.escape),
            "escape file exists: %s" % self.escape,
        )
        text = env.get("text") or ""
        self.assertNotIn("OMNI54_SECRET_CANARY", text)
        self.assertIn("RCS", text)
        parts = text.strip().split()
        # RCS <inside> <escape> <home>
        self.assertGreaterEqual(len(parts), 4, text)
        self.assertEqual(parts[-2], "1", "escape write must be EPERM: %s" % text)
        self.assertNotEqual(parts[-1], "0", "home read must not succeed: %s" % text)
        detail = env.get("fence_detail") or {}
        self.assertTrue(detail.get("read_wall"))
        self.assertTrue(detail.get("workdir_writable"))

    def test_a_claude_lane_enforced(self):
        self._require_canary()
        env = self._run("claude/sonnet")
        self._assert_enforced(env, "claude")

    def test_b_pi_lane_enforced(self):
        self._require_canary()
        env = self._run("pi/xai/grok-4.6")
        self._assert_enforced(env, "pi")

    def test_c_cursor_lane_enforced(self):
        self._require_canary()
        env = self._run("cursor/composer-2.5")
        self._assert_enforced(env, "cursor")
        self.assertTrue(
            (env.get("fence_detail") or {}).get("vendor_state_writable"),
            env,
        )

    def test_codex_deny_write_carve_outs_hold_under_case_variants(self):
        """#94: the carve-outs must survive a case-variant path.

        The macOS boot volume is case-insensitive by default, so
        ~/.codex/AGENTS and ~/.codex/agents are one directory. A deny clause
        that matched case-sensitively would be walked straight past -- the
        exact shape issue #102 found in the profile-admission guards. Runs
        the real sandbox-exec under the policy agent_lane builds.

        Writes are attempted with mode 'a' and ZERO bytes: an existing file
        is never truncated and its content never changes.
        """
        home = kernel_sandbox.protected_home()
        codex_root = os.path.join(home, ".codex")
        if not os.path.isdir(codex_root):
            self.skipTest("no ~/.codex on this host")
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.workdir, read_only=False,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        probe = (
            "import sys\n"
            "try:\n"
            "    open(sys.argv[1], 'a').close()\n"
            "    print('WROTE')\n"
            "except OSError:\n"
            "    print('DENIED')\n"
        )

        def attempt(path):
            argv = kernel_sandbox.wrap_argv(
                [sys.executable, "-c", probe, path], self.workdir, False,
                **wrap_kw,
            )
            proc = __import__("subprocess").run(
                argv, capture_output=True, text=True, timeout=20,
                cwd=self.workdir,
                env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
            )
            return (proc.stdout or proc.stderr or "").strip().splitlines()[-1:]

        denied = [
            os.path.join(codex_root, "agents", "omni94_probe.txt"),
            os.path.join(codex_root, "AGENTS", "omni94_probe.txt"),
            os.path.join(codex_root, "rules", "omni94_probe.txt"),
            os.path.join(codex_root, "RULES", "omni94_probe.txt"),
        ]
        for path in denied:
            with self.subTest(path=path):
                existed = os.path.lexists(path)
                if not existed:
                    # Register cleanup BEFORE the attempt: when a carve-out
                    # is lost the assertion below fails and the probe file
                    # would otherwise be left behind in the user's ~/.codex.
                    self.addCleanup(
                        lambda p=path: os.path.exists(p) and os.remove(p))
                self.assertEqual(attempt(path), ["DENIED"],
                                 "carve-out did not hold for %s" % path)
                self.assertEqual(os.path.lexists(path), existed)

        # Control: state IS writable, so the denials above are the carve-outs
        # and not a blanket deny that would prove nothing. Must be a real
        # state path: a bare top-level file is denied BY DESIGN now that the
        # tree is deny-by-default.
        control = os.path.join(codex_root, "shell_snapshots",
                               "%s.txt" % uuid.uuid4().hex)
        if os.path.lexists(control):
            self.skipTest("%s already exists" % control)
        self.addCleanup(lambda: os.path.exists(control) and os.remove(control))
        self.assertEqual(attempt(control), ["WROTE"])
        self.assertTrue(os.path.isfile(control))

    def test_codex_deny_survives_a_symlinked_instruction_file(self):
        """#94 / Reviewer-A C1: unlink does not follow the last component.

        ``_entry`` realpaths, which is right for an allow clause and wrong
        for a deny: when ~/.codex/AGENTS.md is a SYMLINK out of the tree,
        a deny naming only the target left the LINK deletable, and ~/.codex
        is write-allowed -- so an agent could drop a regular file in its
        place and steer the next codex run. Measured on the real machine
        before the fix (UNLINKED) and after (DENIED).

        Runs against a synthetic home so nothing outside a temp dir is ever
        at risk, and so a regression cannot delete the user's real file.
        The synthetic home sits under /private/var/folders, which is itself
        write-allowed -- so this also proves the deny beats a broad allow.
        """
        fake_home = os.path.realpath(tempfile.mkdtemp(prefix="omni94_home_"))
        self.addCleanup(_rmtree, fake_home)
        codex_root = os.path.join(fake_home, ".codex")
        os.makedirs(codex_root)
        # The link target must sit OUTSIDE every writable root (the temp
        # root is one): a protected name that resolves into a root the child
        # can write is refused outright, which has its own tests.
        outside = os.path.realpath(tempfile.mkdtemp(
            prefix="omni94_elsewhere_", dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.addCleanup(_rmtree, outside)
        target = os.path.join(outside, "real-agents.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("REAL INSTRUCTIONS\n")
        link = os.path.join(codex_root, "AGENTS.md")
        os.symlink(target, link)
        # The control below writes into a state dir; create it, since the
        # probe opens a file inside it (a missing dir gives ENOENT, which
        # would read as DENIED and make the control prove nothing).
        os.makedirs(os.path.join(codex_root, "shell_snapshots"))

        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
            probe = (
                "import os, sys\n"
                "op, path = sys.argv[1], sys.argv[2]\n"
                "try:\n"
                "    if op == 'unlink':\n"
                "        os.unlink(path)\n"
                "    else:\n"
                "        open(path, 'a').close()\n"
                "    print('ALLOWED')\n"
                "except OSError:\n"
                "    print('DENIED')\n"
            )

            def attempt(op, path):
                argv = kernel_sandbox.wrap_argv(
                    [sys.executable, "-c", probe, op, path], self.workdir,
                    False, **wrap_kw,
                )
                proc = __import__("subprocess").run(
                    argv, capture_output=True, text=True, timeout=20,
                    cwd=self.workdir,
                    env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
                )
                return ((proc.stdout or "") + (proc.stderr or "")
                        ).strip().splitlines()[-1:]

            self.assertEqual(attempt("write", link), ["DENIED"],
                             "writing through the link must stay denied")
            self.assertEqual(attempt("unlink", link), ["DENIED"],
                             "the LINK itself must not be removable")
            self.assertTrue(os.path.islink(link), "link was removed")
            # Control: state IS writable, so the two denials above are the
            # carve-out and not a blanket deny that would prove nothing.
            # It must be a real state path -- a bare top-level file is now
            # denied BY DESIGN (the tree is deny-by-default).
            self.assertEqual(
                attempt("write",
                        os.path.join(codex_root, "shell_snapshots",
                                     uuid.uuid4().hex)),
                ["ALLOWED"])

    def test_other_lanes_cannot_touch_the_codex_home(self):
        """#107: claude, pi and cursor have no business in ~/.codex.

        They got write access to it by accident: _write_allow_paths appends
        codex_config_dir() unconditionally for the CONSULT lane's sake, and
        build_profile is shared. While it was writable, a fenced run of any
        of the three could rewrite the script hooks.json invokes, which the
        next codex session executes outside any fence. Measured on main
        before this change: WRITABLE for claude and pi.

        Runs the real sandbox-exec. The write is attempted with mode 'a' and
        ZERO bytes, so an existing file is never truncated.
        """
        home = kernel_sandbox.protected_home()
        codex_root = os.path.join(home, ".codex")
        if not os.path.isdir(codex_root):
            self.skipTest("no ~/.codex on this host")
        # Both halves of the change get probed. Write-only probing left the
        # deny_read line naked: removing it kept this test green, while
        # ~/.codex/auth.json (codex's credential store, mode 0600) went back
        # to being readable by a fenced claude/pi/cursor run. deny-write
        # does not stop a read.
        probe = ("import os, sys\n"
                 "op, path = sys.argv[1], sys.argv[2]\n"
                 "try:\n"
                 "    if op == 'write':\n"
                 "        open(path, 'a').close()\n"
                 "    elif op == 'read':\n"
                 "        open(path, 'rb').read(1)\n"
                 "    else:\n"
                 "        os.listdir(path)\n"
                 "    print('OK')\n"
                 "except OSError:\n    print('D')\n")

        for provider in ("claude", "pi", "cursor"):
            with self.subTest(provider=provider):
                config_dir = None
                if provider == "cursor":
                    config_dir = tempfile.mkdtemp(prefix="omni107_cfg_")
                    self.addCleanup(_rmtree, config_dir)
                policy = kernel_sandbox.agent_fence_policy(
                    provider, workdir=self.workdir, read_only=False,
                    config_dir=config_dir,
                )
                if isinstance(policy, dict) and policy.get("ok") is False:
                    self.skipTest(policy.get("detail") or "policy refused")
                wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
                def attempt(op, path):
                    argv = kernel_sandbox.wrap_argv(
                        [sys.executable, "-c", probe, op, path], self.workdir,
                        False, **wrap_kw,
                    )
                    proc = __import__("subprocess").run(
                        argv, capture_output=True, text=True, timeout=20,
                        cwd=self.workdir,
                        env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
                    )
                    return ((proc.stdout or "") + (proc.stderr or "")
                            ).strip().splitlines()[-1:]

                # The last name is generated per run: a mutant that swaps
                # the whole-tree deny for literal denies of the names this
                # test happens to probe cannot enumerate it, so the test
                # pins "any name under the root" rather than three names.
                for leaf in ("herdr-agent-state.sh", "config.toml",
                             "omni107-%s.toml" % uuid.uuid4().hex[:12]):
                    path = os.path.join(codex_root, leaf)
                    existed = os.path.lexists(path)
                    if not existed:
                        self.addCleanup(
                            lambda p=path: os.path.exists(p) and os.remove(p))
                    self.assertEqual(attempt("write", path), ["D"],
                                     "%s must not write ~/.codex/%s"
                                     % (provider, leaf))
                    self.assertEqual(os.path.lexists(path), existed)
                # Read side. auth.json is the one that matters: a credential
                # store deny-write does nothing for.
                for leaf in ("config.toml", "auth.json"):
                    path = os.path.join(codex_root, leaf)
                    if not os.path.isfile(path):
                        continue
                    self.assertEqual(attempt("read", path), ["D"],
                                     "%s must not read ~/.codex/%s"
                                     % (provider, leaf))
                self.assertEqual(attempt("list", codex_root), ["D"],
                                 "%s must not list ~/.codex" % provider)

    def test_codex_root_is_in_both_deny_sets_for_the_other_lanes(self):
        """Cheap structural pin: the root is denied for write AND for read."""
        home = kernel_sandbox.protected_home()
        codex_root = os.path.realpath(os.path.join(home, ".codex"))
        for provider in ("claude", "pi", "cursor"):
            with self.subTest(provider=provider):
                config_dir = None
                if provider == "cursor":
                    config_dir = tempfile.mkdtemp(prefix="omni107_cfg_")
                    self.addCleanup(_rmtree, config_dir)
                policy = kernel_sandbox.agent_fence_policy(
                    provider, workdir=self.workdir, read_only=False,
                    config_dir=config_dir,
                )
                if isinstance(policy, dict) and policy.get("ok") is False:
                    self.skipTest(policy.get("detail") or "policy refused")
                wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
                self.assertIn(
                    codex_root,
                    {path for _kind, path in wrap_kw["extra_deny_write"]})
                self.assertIn(
                    codex_root,
                    {path for _kind, path in wrap_kw["extra_deny_read"]})

    def test_a_planted_symlink_at_a_vendor_leaf_cannot_reopen_codex(self):
        """The deny is in W/X; T/S come after it and win on last match.

        Every lane post-allows vendor-state leaves with NO existence check,
        so a fenced run may create the node -- as a symlink. Pointed at
        ~/.codex, the next policy build realpaths it into T and the deny
        this change adds stops holding. Measured open before the filter
        (T contained ~/.codex for all three lanes) and closed after.

        Synthetic home; nothing real is touched.
        """
        cases = (
            ("claude", os.path.join(".claude", "debug")),
            ("pi", os.path.join(".pi", "agent", "sessions")),
            ("cursor", os.path.join(".cursor", "projects")),
        )
        for provider, leaf_rel in cases:
            with self.subTest(provider=provider):
                fake_home = os.path.realpath(
                    tempfile.mkdtemp(prefix="omni107_tramp_"))
                self.addCleanup(_rmtree, fake_home)
                codex_root = os.path.join(fake_home, ".codex")
                os.makedirs(codex_root)
                trampoline = os.path.join(fake_home, leaf_rel)
                os.makedirs(os.path.dirname(trampoline), exist_ok=True)
                os.symlink(codex_root, trampoline)

                with mock.patch.object(kernel_sandbox, "protected_home",
                                       lambda h=fake_home: h):
                    config_dir = None
                    if provider == "cursor":
                        config_dir = tempfile.mkdtemp(prefix="omni107_tcfg_")
                        self.addCleanup(_rmtree, config_dir)
                    policy = kernel_sandbox.agent_fence_policy(
                        provider, workdir=self.workdir, read_only=False,
                        config_dir=config_dir,
                    )
                    if isinstance(policy, dict) and policy.get("ok") is False:
                        if provider == "claude":
                            # The claude per-run leaves refuse a symlinked
                            # component outright: nothing can re-open codex.
                            self.assertEqual(policy.get("reason"),
                                             "fence_path_rejected", policy)
                            self.assertIn("symlink", policy.get("detail", ""))
                            continue
                        self.skipTest(policy.get("detail") or "refused")
                    wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
                    for key in ("extra_post_write", "extra_post_read"):
                        for _kind, path in wrap_kw[key]:
                            self.assertFalse(
                                path == codex_root
                                or kernel_sandbox._path_inside(path, codex_root),
                                "%s %s re-opens the codex tree via %s"
                                % (key, path, leaf_rel),
                            )

    def test_a_config_dir_under_the_codex_root_cannot_reopen_it(self):
        """#107: the deny is not the last clause, so post-allows must be filtered.

        build_profile emits (deny W*) then (allow T*), and (deny X*) then
        (allow S*). A CLAUDE_CONFIG_DIR pointed at a directory under
        ~/.codex is a home descendant that is not ~/.claude-<name>, so
        _caller_config_dir_rejection now refuses it before any post-allow
        can land inside the denied tree.

        Synthetic home, so nothing real is touched.
        """
        fake_home = os.path.realpath(tempfile.mkdtemp(prefix="omni107_cfg_"))
        self.addCleanup(_rmtree, fake_home)
        codex_root = os.path.join(fake_home, ".codex")
        profile_root = os.path.join(codex_root, "claude-prof")
        os.makedirs(os.path.join(profile_root, "projects"))

        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "claude", workdir=self.workdir, read_only=False,
                config_dir=profile_root,
            )
            # New rule: a home descendant that is not ~/.claude-<name> is
            # refused, so a profile root under ~/.codex cannot enter the
            # allow-set (and therefore cannot reopen it via post-allows).
            self.assertEqual(policy.get("ok"), False, policy)
            self.assertEqual(policy.get("reason"), "fence_path_rejected")
            self.assertIn("inside the home tree", policy.get("detail", ""))

    def test_codex_home_is_deny_by_default_for_anything_not_state(self):
        """#94: the codex home is an allow-list, not a deny-list.

        Two reviewers from different families each found instruction
        surfaces missing from a hand-built deny list -- AGENTS.override.md
        (which takes PRECEDENCE over AGENTS.md in codex-cli 0.154.0),
        $CODEX_HOME/<name>.config.toml profile layers, a script hooks.json
        invokes, and the staged plugin tree under .tmp. The point of this
        test is that NONE of those names appears in the fence code: a new
        one that nobody has thought of is denied because it is not state.

        Synthetic home, so nothing real is ever at risk.
        """
        fake_home = os.path.realpath(tempfile.mkdtemp(prefix="omni94_inv_"))
        self.addCleanup(_rmtree, fake_home)
        codex_root = os.path.join(fake_home, ".codex")
        os.makedirs(os.path.join(codex_root, "shell_snapshots"))
        for name in ("config.toml", "auth.json", "history.jsonl",
                     "logs_9.sqlite", "herdr-agent-state.sh"):
            with open(os.path.join(codex_root, name), "w") as fh:
                fh.write("x")

        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
            # makedirs is part of the probe on purpose: sessions/ grows a
            # fresh <year>/<month>/<day> tree every day, so "can codex create
            # directories under a post-allowed subpath" is the real question,
            # not just "can it write an existing file".
            probe = ("import os, sys\n"
                     "try:\n"
                     "    os.makedirs(os.path.dirname(sys.argv[1]),"
                     " exist_ok=True)\n"
                     "    open(sys.argv[1],'a').close(); print('W')\n"
                     "except OSError:\n    print('D')\n")

            def attempt(rel):
                path = os.path.join(codex_root, rel)
                argv = kernel_sandbox.wrap_argv(
                    [sys.executable, "-c", probe, path], self.workdir, False,
                    **wrap_kw,
                )
                proc = __import__("subprocess").run(
                    argv, capture_output=True, text=True, timeout=20,
                    cwd=self.workdir,
                    env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
                )
                return ((proc.stdout or "") + (proc.stderr or "")
                        ).strip().splitlines()[-1:]

            # Steering surfaces -- none of these names is in the fence code
            # except config.toml; they are denied for being non-state.
            for rel in ("AGENTS.override.md", "work.config.toml",
                        "herdr-agent-state.sh", "config.toml",
                        "a-name-nobody-has-thought-of.toml"):
                with self.subTest(denied=rel):
                    self.assertEqual(attempt(rel), ["D"],
                                     "%s must not be writable" % rel)
            # State codex genuinely needs.
            snap_probe = "shell_snapshots/%s" % uuid.uuid4().hex
            for rel in (snap_probe, "sessions/2026/09/14/r.jsonl",
                        "thread-writer-locks/l", "history.jsonl", "auth.json",
                        "logs_9.sqlite"):
                with self.subTest(allowed=rel):
                    self.assertEqual(attempt(rel), ["W"],
                                     "%s is state and must stay writable" % rel)


    def test_realpath_on_cursor_binary_succeeds_under_the_built_fence(self):
        """Fix-round-1 pin: cursor-agent's launcher does `realpath "$0"`
        before anything else; the read wall's ancestor-metadata gap (fixed
        in kernel_sandbox.build_profile) must not break that. Runs the REAL
        /bin/realpath (not the fake three-step script) against the REAL
        resolved cursor binary, under the exact policy agent_lane builds."""
        os.environ.pop(kernel_sandbox.ENV_CURSOR_BIN, None)  # use the real one
        real_cursor = kernel_sandbox.resolve_agent_binary("cursor")
        if not real_cursor or not os.path.isfile(real_cursor):
            self.skipTest("no real cursor-agent installed on this host")
        policy = kernel_sandbox.agent_fence_policy(
            "cursor", workdir=self.workdir, read_only=False,
            config_dir=tempfile.mkdtemp(prefix="omni54_cursor_cfg_"),
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        argv = kernel_sandbox.wrap_argv(
            ["/bin/realpath", real_cursor], self.workdir, False, **wrap_kw,
        )
        proc = __import__("subprocess").run(
            argv, capture_output=True, text=True, timeout=10, cwd=self.workdir,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), os.path.realpath(real_cursor))

    def test_cursor_path_symlink_cannot_execvp_under_fence(self):
        """The usual PATH hit is ~/.local/bin/cursor-agent, a home-tree
        symlink. execvp of that path is denied; the versioned realpath is
        what the fence re-allows. Consult must spawn the realpath.

        Built in a synthetic home with a stand-in binary: the probe needs a
        symlink under the protected home, not the operator's installed one.
        """
        home = offline_home.make_home()
        self.addCleanup(_rmtree, home)
        versions = os.path.join(home, ".local", "share", "cursor-agent", "v1")
        os.makedirs(versions)
        real_bin = os.path.join(versions, "cursor-agent")
        with open(real_bin, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(real_bin, 0o755)
        bin_dir = os.path.join(home, ".local", "bin")
        os.makedirs(bin_dir, exist_ok=True)
        which_bin = os.path.join(bin_dir, "cursor-agent")
        os.symlink(real_bin, which_bin)
        real_bin = os.path.realpath(real_bin)
        env = {"PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")}
        env_patch = mock.patch.dict(os.environ, env)
        home_patch = mock.patch.object(
            kernel_sandbox, "protected_home", lambda: home)
        env_patch.start()
        home_patch.start()
        self.addCleanup(env_patch.stop)
        self.addCleanup(home_patch.stop)
        os.environ.pop(kernel_sandbox.ENV_CURSOR_BIN, None)
        policy = kernel_sandbox.agent_fence_policy(
            "cursor", workdir=self.workdir, read_only=True,
            config_dir=tempfile.mkdtemp(prefix="omni54_cursor_cfg_"),
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        denied = kernel_sandbox.wrap_argv(
            [which_bin, "--help"], self.workdir, True, **wrap_kw,
        )
        allowed = kernel_sandbox.wrap_argv(
            [real_bin, "--help"], self.workdir, True, **wrap_kw,
        )
        subprocess = __import__("subprocess")
        blocked = subprocess.run(
            denied, capture_output=True, text=True, timeout=10, cwd=self.workdir,
        )
        self.assertEqual(blocked.returncode, 71, blocked.stderr)
        self.assertIn("Operation not permitted", blocked.stderr)
        ok = subprocess.run(
            allowed, capture_output=True, text=True, timeout=10, cwd=self.workdir,
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)

    def test_read_only_workdir_write_eperm(self):
        # A workdir under a temp root is always writable (temp roots are in
        # the allow-set), so this test needs a workdir under the protected
        # home, where only the read re-allow applies.
        home_wd = tempfile.mkdtemp(
            prefix="omni54_ro_", dir=kernel_sandbox.protected_home())
        self.addCleanup(_rmtree, home_wd)
        self.workdir = os.path.realpath(home_wd)
        if kernel_sandbox.verify_fence(
            self.workdir, read_only=True, read_wall=True,
        ) is not None:
            self.skipTest("verify_fence cannot plant a home probe in this environment")
        env = self._run("claude/sonnet", read_only=True)
        self.assertTrue(env.get("ok") or env.get("reason") in (
            "empty_response", "runtime_error",
        ), env)
        self.assertFalse(
            os.path.isfile(os.path.join(self.workdir, "inside.txt")),
            "read_only lane must not write workdir",
        )
        if env.get("ok"):
            self.assertEqual(env.get("fence"), "seatbelt-fs")
            self.assertFalse((env.get("fence_detail") or {}).get("workdir_writable"))

    def test_a3_deny_list_writes_eperm(self):
        probe = kernel_sandbox.wrap_argv(
            ["/usr/bin/true"], self.workdir, True, read_wall=True,
        )
        nested = __import__("subprocess").run(
            probe, capture_output=True, text=True, timeout=5,
        )
        if nested.returncode == 71:
            self.skipTest("nested sandbox_apply refused (rc=71)")
        home = kernel_sandbox.protected_home()
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=self.workdir, read_only=False,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        targets = [
            "%s:%s" % (kind, os.path.join(home, rel))
            for rel, kind in kernel_sandbox._AGENT_DENY_WRITE
        ]
        script = textwrap.dedent(
            """\
            import os, sys
            for spec in sys.argv[1:]:
                kind, t = spec.split(":", 1)
                try:
                    if kind == "subpath":
                        os.makedirs(t, exist_ok=True)
                        open(os.path.join(t, "omni54-x"), "w").write("x")
                    else:
                        open(t, "a").write("x")
                    print("WROTE", t)
                except OSError as exc:
                    print("ERRNO=%d" % exc.errno, t)
            """
        )
        argv = [sys.executable, "-c", script] + targets
        wrapped = kernel_sandbox.wrap_argv(
            argv, self.workdir, False, **wrap_kw,
        )
        env = kernel_sandbox.agent_child_env()
        proc = __import__("subprocess").run(
            wrapped, capture_output=True, text=True, timeout=10, cwd=self.workdir,
            env=env,
        )
        self.assertNotIn("WROTE", proc.stdout, proc.stdout)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("ERRNO="):
                self.assertTrue(
                    line.startswith("ERRNO=1") or "ERRNO=1 " in line,
                    line,
                )

    def test_claude_profile_root_admitted_session_writable_default_denied(self):
        profile = tempfile.mkdtemp(prefix="omni72_live_prof_")
        self.addCleanup(_rmtree, profile)
        policy = kernel_sandbox.agent_fence_policy(
            "claude", workdir=self.workdir, read_only=False, config_dir=profile,
        )
        self.assertNotIn("ok", policy, policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        rp = os.path.realpath(profile)
        self.assertIn(rp, wrap_kw["extra_write_paths"])
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        self.assertNotIn(default_root, wrap_kw["extra_read_paths"])
        session = kernel_sandbox.claude_session_dir(
            self.workdir, config_dir=profile)
        self.assertTrue(session.startswith(rp + os.sep), session)
        home_projects = os.path.join(
            kernel_sandbox.protected_home(), ".claude", "projects")
        probe = kernel_sandbox.wrap_argv(
            ["/usr/bin/true"], self.workdir, True, read_wall=True,
        )
        nested = __import__("subprocess").run(
            probe, capture_output=True, text=True, timeout=5,
        )
        if nested.returncode == 71:
            self.skipTest("nested sandbox_apply refused (rc=71)")
        other = os.path.join(rp, "projects", "-Users-x-other")
        script = textwrap.dedent(
            """\
            import os, sys
            session, projects, other = sys.argv[1], sys.argv[2], sys.argv[3]
            try:
                os.makedirs(session, exist_ok=True)
                open(os.path.join(session, "omni72-session"), "w").write("ok")
                print("SESSION_WROTE")
            except OSError as exc:
                print("SESSION_ERRNO=%d" % exc.errno)
            try:
                os.listdir(projects)
                print("PROJECTS_READ")
            except OSError as exc:
                print("PROJECTS_ERRNO=%d" % exc.errno)
            try:
                os.makedirs(other, exist_ok=True)
                open(os.path.join(other, "omni72-other"), "w").write("x")
                print("OTHER_WROTE")
            except OSError as exc:
                print("OTHER_ERRNO=%d" % exc.errno)
            """
        )
        argv = [sys.executable, "-c", script, session, home_projects, other]
        wrapped = kernel_sandbox.wrap_argv(
            argv, self.workdir, False, **wrap_kw,
        )
        env = kernel_sandbox.agent_child_env(
            {**os.environ, "CLAUDE_CONFIG_DIR": rp},
            tmpdir=policy.get("tmpdir"),
            keep_config_dir=True,
        )
        proc = __import__("subprocess").run(
            wrapped, capture_output=True, text=True, timeout=10,
            cwd=self.workdir, env=env,
        )
        self.assertIn("SESSION_WROTE", proc.stdout, proc.stdout + proc.stderr)
        self.assertNotIn("PROJECTS_READ", proc.stdout, proc.stdout)
        self.assertIn("PROJECTS_ERRNO=", proc.stdout, proc.stdout)
        self.assertNotIn("OTHER_WROTE", proc.stdout, proc.stdout)
        self.assertIn("OTHER_ERRNO=1", proc.stdout, proc.stdout)

    def test_profile_ring_fence_isolates_sibling_canaries(self):
        root = tempfile.mkdtemp(prefix="omni72_ring_live_")
        self.addCleanup(_rmtree, root)
        dirs = {}
        canaries = {}
        for name in ("a", "b", "c"):
            path = os.path.join(root, name)
            os.makedirs(path)
            canary = os.path.join(path, "canary")
            with open(canary, "w", encoding="utf-8") as handle:
                handle.write("CANARY-%s\n" % name.upper())
            dirs[name] = os.path.realpath(path)
            canaries[name] = canary
        default_root = os.path.realpath(
            os.path.join(kernel_sandbox.protected_home(), ".claude"))
        probe = kernel_sandbox.wrap_argv(
            ["/usr/bin/true"], self.workdir, True, read_wall=True,
        )
        nested = __import__("subprocess").run(
            probe, capture_output=True, text=True, timeout=5,
        )
        live = nested.returncode != 71
        env_patch = {
            "CLAUDE_CONFIG_DIR_A": dirs["a"],
            "CLAUDE_CONFIG_DIR_B": dirs["b"],
            "CLAUDE_CONFIG_DIR_C": dirs["c"],
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            for winner in ("a", "b", "c"):
                overrides, err = resolve_profile("claude", winner)
                self.assertIsNone(err, err)
                parent = {**os.environ}
                child = child_env_for_profile(parent, overrides)
                self.assertEqual(child.get("CLAUDE_CONFIG_DIR"), dirs[winner])
                for name in ("A", "B", "C"):
                    self.assertNotIn("CLAUDE_CONFIG_DIR_%s" % name, child)
                policy = kernel_sandbox.agent_fence_policy(
                    "claude", workdir=self.workdir, read_only=False,
                    config_dir=dirs[winner],
                )
                self.assertNotIn("ok", policy, policy)
                wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
                params = kernel_sandbox.profile_params(
                    self.workdir, False, **wrap_kw)
                blob = " ".join(str(v) for v in params.values())
                self.assertIn(dirs[winner], blob)
                self.assertIn(dirs[winner], wrap_kw["extra_write_paths"])
                self.assertNotIn(default_root, wrap_kw["extra_read_paths"])
                self.assertNotIn(default_root, wrap_kw["extra_write_paths"])
                self.assertNotIn("~/.claude", blob)
                if not live:
                    continue
                siblings = [n for n in ("a", "b", "c") if n != winner]
                script = textwrap.dedent(
                    """\
                    import os, sys
                    own, sib1, sib2 = sys.argv[1], sys.argv[2], sys.argv[3]
                    print("OWN=" + open(own, encoding="utf-8").read().strip())
                    for path, label in ((sib1, "SIB1"), (sib2, "SIB2")):
                        try:
                            open(path, encoding="utf-8").read()
                            print(label + "_READ")
                        except OSError as exc:
                            print(label + "_ERRNO=%d" % exc.errno)
                    print("CLAUDE_CONFIG_DIR=" + os.environ.get(
                        "CLAUDE_CONFIG_DIR", ""))
                    sibs = sorted(
                        k for k in os.environ if k.startswith("CLAUDE_CONFIG_DIR_")
                    )
                    print("SIBLINGS=" + ",".join(sibs))
                    """
                )
                argv = [
                    sys.executable, "-c", script,
                    canaries[winner],
                    canaries[siblings[0]],
                    canaries[siblings[1]],
                ]
                wrapped = kernel_sandbox.wrap_argv(
                    argv, self.workdir, False, **wrap_kw,
                )
                env = kernel_sandbox.agent_child_env(
                    child,
                    tmpdir=policy.get("tmpdir"),
                    keep_config_dir=True,
                )
                proc = __import__("subprocess").run(
                    wrapped, capture_output=True, text=True, timeout=10,
                    cwd=self.workdir, env=env,
                )
                self.assertIn(
                    "OWN=CANARY-%s" % winner.upper(),
                    proc.stdout, proc.stdout + proc.stderr)
                self.assertNotIn("SIB1_READ", proc.stdout, proc.stdout)
                self.assertNotIn("SIB2_READ", proc.stdout, proc.stdout)
                self.assertIn("SIB1_ERRNO=1", proc.stdout, proc.stdout)
                self.assertIn("SIB2_ERRNO=1", proc.stdout, proc.stdout)
                self.assertIn(
                    "CLAUDE_CONFIG_DIR=" + dirs[winner],
                    proc.stdout, proc.stdout)
                self.assertIn("SIBLINGS=", proc.stdout, proc.stdout)
                self.assertNotIn("CLAUDE_CONFIG_DIR_A", proc.stdout.split(
                    "SIBLINGS=")[-1].splitlines()[0])
        if not live:
            self.skipTest("nested sandbox_apply refused (rc=71)")

    def _skip_nested_sandbox(self):
        probe = kernel_sandbox.wrap_argv(
            ["/usr/bin/true"], self.workdir, True, read_wall=True,
        )
        nested = __import__("subprocess").run(
            probe, capture_output=True, text=True, timeout=5,
        )
        if nested.returncode == 71:
            self.skipTest("nested sandbox_apply refused (rc=71)")

    def _private_tmp_codex_home(self):
        """Synthetic home under /private/tmp, outside the P allow-set."""
        folders = os.path.realpath("/private/var/folders")
        root = tempfile.mkdtemp(prefix="omni108_live_", dir="/private/tmp")
        self.addCleanup(_rmtree, root)
        rp = os.path.realpath(root)
        self.assertFalse(
            rp == folders or rp.startswith(folders + os.sep),
            "synthetic home landed under /private/var/folders: %s" % rp,
        )
        fake_home = os.path.join(root, "home")
        codex_root = os.path.join(fake_home, ".codex")
        os.makedirs(codex_root)
        return fake_home, codex_root, root

    def test_codex_hard_link_alias_of_config_toml_is_denied(self):
        """link(history.jsonl, workdir alias) is L-denied; without L it succeeds.

        Both ends are writable (T-allowed source, P-allowed dest), so W
        does not refuse this pair. config.toml -> log/ is a W regression.
        """
        self._skip_nested_sandbox()
        fake_home, codex_root, _root = self._private_tmp_codex_home()
        os.makedirs(os.path.join(codex_root, "log"))
        history = os.path.join(codex_root, "history.jsonl")
        with open(history, "w", encoding="utf-8") as fh:
            fh.write("HIST\n")
        config = os.path.join(codex_root, "config.toml")
        with open(config, "w", encoding="utf-8") as fh:
            fh.write("SECRET\n")
        alias_l = os.path.join(self.workdir, uuid.uuid4().hex)
        alias_no_l = os.path.join(self.workdir, uuid.uuid4().hex)
        alias_w = os.path.join(codex_root, "log", uuid.uuid4().hex)
        probe = (
            "import os, sys\n"
            "src, dst = sys.argv[1], sys.argv[2]\n"
            "try:\n"
            "    os.link(src, dst)\n"
            "    print('LINK_OK')\n"
            "except OSError:\n"
            "    print('LINK_DENIED')\n"
        )
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
            wrap_kw_no_l = dict(wrap_kw)
            wrap_kw_no_l["extra_deny_link"] = ()

            def _run_link(src, dst, kw):
                argv = kernel_sandbox.wrap_argv(
                    [sys.executable, "-c", probe, src, dst],
                    self.workdir, False, **kw,
                )
                proc = __import__("subprocess").run(
                    argv, capture_output=True, text=True, timeout=20,
                    cwd=self.workdir,
                    env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
                )
                if proc.returncode == 71:
                    self.skipTest("nested sandbox_apply refused (rc=71)")
                return ((proc.stdout or "") + (proc.stderr or "")).splitlines()

            out_l = _run_link(history, alias_l, wrap_kw)
            self.assertIn("LINK_DENIED", out_l, out_l)
            self.assertNotIn("LINK_OK", out_l)
            self.assertFalse(os.path.lexists(alias_l))
            out_no_l = _run_link(history, alias_no_l, wrap_kw_no_l)
            self.assertIn("LINK_OK", out_no_l, out_no_l)
            self.assertNotIn("LINK_DENIED", out_no_l)
            self.assertTrue(os.path.lexists(alias_no_l))
            # Covered by W: config.toml is not T-allowed.
            self.assertIn("LINK_DENIED", _run_link(config, alias_w, wrap_kw))

    def test_lexical_t_is_inert_on_a_planted_state_dir_symlink(self):
        """Plant log -> outside the home, patch the refusal out, write is denied.

        Lexical T names the symlink path; Seatbelt evaluates file-write on
        the kernel-resolved path, so the target stays outside T.
        """
        self._skip_nested_sandbox()
        fake_home, codex_root, root = self._private_tmp_codex_home()
        outside = os.path.join(root, "outside")
        os.makedirs(outside)
        log_path = os.path.join(codex_root, "log")
        os.symlink(outside, log_path)
        os.makedirs(os.path.join(codex_root, "shell_snapshots"))
        target = os.path.join(log_path, uuid.uuid4().hex)
        with mock.patch.object(kernel_sandbox, "protected_home",
                               lambda: fake_home), \
             mock.patch.object(kernel_sandbox, "_codex_post_allow_refusal",
                               return_value=None):
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=self.workdir, read_only=False,
            )
            self.assertNotIn("ok", policy, policy)
            wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
            t_paths = {os.path.abspath(p)
                       for _k, p in wrap_kw["extra_post_write"]}
            self.assertIn(os.path.abspath(log_path), t_paths)
            self.assertNotIn(os.path.abspath(outside), t_paths)
            probe = (
                "import sys\n"
                "try:\n"
                "    open(sys.argv[1], 'a').close()\n"
                "    print('WROTE')\n"
                "except OSError:\n"
                "    print('DENIED')\n"
            )
            argv = kernel_sandbox.wrap_argv(
                [sys.executable, "-c", probe, target],
                self.workdir, False, **wrap_kw,
            )
            proc = __import__("subprocess").run(
                argv, capture_output=True, text=True, timeout=20,
                cwd=self.workdir,
                env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
            )
            if proc.returncode == 71:
                self.skipTest("nested sandbox_apply refused (rc=71)")
            out = ((proc.stdout or "") + (proc.stderr or "")
                   ).strip().splitlines()[-1:]
            self.assertEqual(out, ["DENIED"], proc.stdout + proc.stderr)
            self.assertFalse(os.path.lexists(target))
            self.assertFalse(
                os.path.lexists(os.path.join(outside, os.path.basename(target)))
            )

    def test_grok_auth_atomic_replace_succeeds_config_stays_denied(self):
        """grok saves auth.json as .<pid>.<seq>.tmp + rename (1.0.46).

        A refreshed token that cannot be saved leaves the rotated-away
        refresh token on disk and the next run loses the login. The child
        may only use its own pid's siblings: another process's sibling (an
        unfenced grok saving config.toml) stays untouchable, and no sibling
        reaches config.toml.
        """
        self._skip_nested_sandbox()
        # Regex metacharacters in the root: over- or under-quoting fails AUTH.
        grok_home = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok.h+(x)[w]{1}|^$*?_"))
        self.addCleanup(_rmtree, grok_home)
        for name in ("auth.json", "config.toml", ".1.0.tmp"):
            with open(os.path.join(grok_home, name), "w") as handle:
                handle.write("OLD\n")
        policy = kernel_sandbox.agent_fence_policy(
            "grok", workdir=self.workdir, read_only=True, config_dir=grok_home,
        )
        self.assertFalse(isinstance(policy, dict) and policy.get("ok") is False,
                         policy)
        wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
        probe = (
            "import os, sys\n"
            "home = sys.argv[1]\n"
            "def step(label, fn):\n"
            "    try:\n"
            "        fn()\n"
            "        print(label + '=OK')\n"
            "    except OSError:\n"
            "        print(label + '=DENIED')\n"
            "def save(target, tmp):\n"
            "    def run():\n"
            "        with open(os.path.join(home, tmp), 'w') as fh:\n"
            "            fh.write('NEW\\n')\n"
            "        os.rename(os.path.join(home, tmp), os.path.join(home, target))\n"
            "    return run\n"
            "pid = os.getpid()\n"
            "step('AUTH', save('auth.json', '.%d.0.tmp' % pid))\n"
            "step('CONFIG', save('config.toml', '.%d.1.tmp' % pid))\n"
            "step('OTHERPID', save('auth.json', '.%d.0.tmp' % (pid + 1)))\n"
            "step('FOREIGN', lambda: os.unlink(os.path.join(home, '.1.0.tmp')))\n"
            "step('SUFFIX', lambda: open(os.path.join(home, '.%d.2.tmp.x' % pid), 'w'))\n"
            "def nested():\n"
            "    os.mkdir(os.path.join(home, '.%d.3.tmp' % pid))\n"
            "    open(os.path.join(home, '.%d.3.tmp' % pid, 'child'), 'w')\n"
            "step('NESTED', nested)\n"
        )

        def run(argv):
            proc = __import__("subprocess").run(
                argv, capture_output=True, text=True, timeout=20,
                cwd=self.workdir,
                env=kernel_sandbox.sanitized_child_env(dict(os.environ)),
            )
            # Nesting already proved possible above; rc 71 now is a regression.
            self.assertNotEqual(proc.returncode, 71, proc.stderr)
            return proc

        wrapped = kernel_sandbox.wrap_argv(
            [sys.executable, "-I", "-c", probe, grok_home],
            self.workdir, True, **wrap_kw,
        )
        # Without the launcher's pid the placeholder grants nothing.
        proc = run(wrapped)
        self.assertEqual(proc.stdout.split(), [
            "AUTH=DENIED", "CONFIG=DENIED", "OTHERPID=DENIED", "FOREIGN=DENIED",
            "SUFFIX=DENIED", "NESTED=DENIED",
        ], proc.stdout + proc.stderr)
        proc = run(["/bin/sh", "-c", kernel_sandbox.GROK_PID_SHIM] + wrapped)
        self.assertEqual(proc.stdout.split(), [
            "AUTH=OK", "CONFIG=DENIED", "OTHERPID=DENIED", "FOREIGN=DENIED",
            "SUFFIX=DENIED", "NESTED=DENIED",
        ], proc.stdout + proc.stderr)
        for name, body in (("auth.json", "NEW\n"), ("config.toml", "OLD\n"),
                           (".1.0.tmp", "OLD\n")):
            with open(os.path.join(grok_home, name)) as handle:
                self.assertEqual(handle.read(), body, name)
        # The shim refuses an argv without the placeholder after the profile.
        tampered = wrapped[:3] + ["-D", "%s=1" % kernel_sandbox.GROK_PID_PARAM] + wrapped[5:]
        self.assertEqual(run(["/bin/sh", "-c", kernel_sandbox.GROK_PID_SHIM]
                             + tampered).returncode, 126)
        # A GROK_HOME spelled in another case than on disk still saves
        # (Seatbelt regex matching is case-insensitive here, measured).
        parent, leaf = os.path.split(grok_home)
        variant = os.path.join(parent, leaf.swapcase())
        self.assertNotEqual(variant, grok_home)
        policy = kernel_sandbox.agent_fence_policy(
            "grok", workdir=self.workdir, read_only=True, config_dir=variant,
        )
        self.assertFalse(isinstance(policy, dict) and policy.get("ok") is False,
                         policy)
        wrapped = kernel_sandbox.wrap_argv(
            [sys.executable, "-I", "-c", probe, grok_home],
            self.workdir, True, **kernel_sandbox.wrap_kwargs_from_policy(policy),
        )
        proc = run(["/bin/sh", "-c", kernel_sandbox.GROK_PID_SHIM] + wrapped)
        self.assertEqual(proc.stdout.split()[:2], ["AUTH=OK", "CONFIG=DENIED"],
                         proc.stdout + proc.stderr)


@unittest.skipUnless(_DARWIN_FENCE, _SKIP_NO_FENCE)
class TestAgentFenceJobLane(unittest.TestCase):
    def test_i_job_json_planned_and_result_observed(self):
        root = tempfile.mkdtemp(prefix="omni54_job_")
        self.addCleanup(_rmtree, root)
        bindir = os.path.join(root, "bin")
        os.makedirs(bindir)
        fake = os.path.join(bindir, "claude")
        _write_exec(fake, _three_step_script() + "\n")
        wd = os.path.join(root, "wd")
        os.makedirs(wd)
        import subprocess
        subprocess.run(
            ["git", "init", "-q", "-b", "work", wd], check=True,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                 "GIT_CONFIG_SYSTEM": os.devnull},
        )
        old_bin = os.environ.get(kernel_sandbox.ENV_CLAUDE_BIN)
        old_depth = os.environ.pop(ENV_DEPTH, None)
        os.environ[kernel_sandbox.ENV_CLAUDE_BIN] = fake
        canary = None
        try:
            canary = _plant_home_canary()
            os.environ["OMNI54_CANARY"] = canary
        except OSError:
            self.skipTest("cannot plant home canary")
        state = tempfile.mkdtemp(prefix="omni54_state_")
        self.addCleanup(_rmtree, state)
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = state

        def restore():
            if old_bin is None:
                os.environ.pop(kernel_sandbox.ENV_CLAUDE_BIN, None)
            else:
                os.environ[kernel_sandbox.ENV_CLAUDE_BIN] = old_bin
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            if canary:
                try:
                    os.remove(canary)
                except OSError:
                    pass
            os.environ.pop("OMNI54_CANARY", None)
            if old_depth is None:
                os.environ.pop(ENV_DEPTH, None)
            else:
                os.environ[ENV_DEPTH] = old_depth

        self.addCleanup(restore)
        started = jobs.start(
            task="three steps",
            model="claude/sonnet",
            workdir=wd,
            timeout_s=30,
        )
        self.assertNotIn("refused", started, started)
        job_dir = started["job_dir"]
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            planned = __import__("json").load(handle)
        self.assertEqual(planned.get("fence"), "seatbelt-fs")
        self.assertTrue(planned.get("fence_requested"))
        deadline = __import__("time").time() + 20
        result = None
        while __import__("time").time() < deadline:
            result = jobs.result(job_dir, wait_s=0)
            if result.get("reason") != "not_finished":
                break
            __import__("time").sleep(0.1)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.get("reason"), "not_finished", result)
        self.assertEqual(result.get("fence"), "seatbelt-fs", result)

    def test_j_cursor_job_lane_durable_config_dir_and_fence(self):
        root = tempfile.mkdtemp(prefix="omni54_cursor_job_")
        self.addCleanup(_rmtree, root)
        bindir = os.path.join(root, "bin")
        os.makedirs(bindir)
        fake = os.path.join(bindir, "cursor-agent")
        _write_exec(fake, _three_step_script() + "\n")
        wd = os.path.join(root, "wd")
        os.makedirs(wd)
        import subprocess
        subprocess.run(
            ["git", "init", "-q", "-b", "work", wd], check=True,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                 "GIT_CONFIG_SYSTEM": os.devnull},
        )
        old_bin = os.environ.get(kernel_sandbox.ENV_CURSOR_BIN)
        old_depth = os.environ.pop(ENV_DEPTH, None)
        os.environ[kernel_sandbox.ENV_CURSOR_BIN] = fake
        state = tempfile.mkdtemp(prefix="omni54_cursor_state_")
        self.addCleanup(_rmtree, state)
        old_state = os.environ.get("OMNI_LLM_STATE_DIR")
        os.environ["OMNI_LLM_STATE_DIR"] = state

        def restore():
            if old_bin is None:
                os.environ.pop(kernel_sandbox.ENV_CURSOR_BIN, None)
            else:
                os.environ[kernel_sandbox.ENV_CURSOR_BIN] = old_bin
            if old_state is None:
                os.environ.pop("OMNI_LLM_STATE_DIR", None)
            else:
                os.environ["OMNI_LLM_STATE_DIR"] = old_state
            if old_depth is None:
                os.environ.pop(ENV_DEPTH, None)
            else:
                os.environ[ENV_DEPTH] = old_depth

        self.addCleanup(restore)
        started = jobs.start(
            task="three steps",
            model="cursor/composer-2.5",
            workdir=wd,
            timeout_s=30,
        )
        self.assertNotIn("refused", started, started)
        job_dir = started["job_dir"]
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            planned = __import__("json").load(handle)
        self.assertEqual(planned.get("provider"), "cursor")
        self.assertEqual(planned.get("fence"), "seatbelt-fs")
        self.assertIsNone(planned.get("session_id"))
        deadline = __import__("time").time() + 20
        result = None
        while __import__("time").time() < deadline:
            result = jobs.result(job_dir, wait_s=0)
            if result.get("reason") != "not_finished":
                break
            __import__("time").sleep(0.1)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.get("reason"), "not_finished", result)
        self.assertEqual(result.get("fence"), "seatbelt-fs", result)
        # The fake binary emits no JSON events, so session_id capture never
        # fires — it must stay null rather than be fabricated.
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            final_job = __import__("json").load(handle)
        self.assertIsNone(final_job.get("session_id"))
        self.assertTrue(
            os.path.isdir(os.path.join(job_dir, "cursor-config")),
            "job lane must pin a durable <job_dir>/cursor-config",
        )


class TestGrokInstructionScanFaults(unittest.TestCase):
    """Fault injection at every syscall the grok instruction scanner makes.

    Scanner call sites (providers/kernel_sandbox.py): os.listdir(root),
    os.lstat on every component of every resolved path, os.readlink on every
    link, os.listdir on every rules directory, and the confirming os.lstat
    behind the two ENOENT exemptions. Nothing else touches the filesystem.
    """

    def setUp(self):
        from providers import host_fence
        self.h = host_fence
        base = os.path.realpath(tempfile.mkdtemp(prefix="omni_grok_scan_"))
        self.addCleanup(_rmtree, base)
        self.root = os.path.join(base, "grokhome")
        self.out = os.path.join(base, "outside")
        os.makedirs(os.path.join(self.root, "rules", "sub"))
        os.makedirs(self.out)
        p = {}
        p["target"] = os.path.join(self.out, "target.md")
        p["mid"] = os.path.join(self.out, "mid")
        p["agents"] = os.path.join(self.root, "AGENTS.md")
        p["claude"] = os.path.join(self.root, "CLAUDE.md")
        p["rules"] = os.path.join(self.root, "rules")
        p["topic_link"] = os.path.join(p["rules"], "topic.md")
        p["topic"] = os.path.join(self.out, "topic.md")
        p["sub"] = os.path.join(p["rules"], "sub")
        p["x"] = os.path.join(p["sub"], "x.md")
        for path in (p["target"], p["claude"], p["topic"], p["x"]):
            with open(path, "w") as handle:
                handle.write("synthetic fixture")
        os.symlink(p["target"], p["mid"])            # two hops: AGENTS.md -> mid -> target
        os.symlink(p["mid"], p["agents"])
        os.symlink(p["topic"], p["topic_link"])
        p["root"] = self.root
        p["outside_dir"] = self.out
        self.p = p

    def inject(self, func, path, err, once=False):
        real = getattr(os, func)
        state = {"fired": 0}

        def fake(arg, *a, **k):
            if os.fspath(arg) == path and not (once and state["fired"]):
                state["fired"] += 1
                raise OSError(err, os.strerror(err), os.fspath(arg))
            return real(arg, *a, **k)
        return mock.patch.object(kernel_sandbox.os, func, side_effect=fake), state

    def scan(self):
        return kernel_sandbox._grok_instruction_deny_read(self.root)

    def assertRefused(self, label, state=None):
        rows, why = self.scan()
        self.assertIsNotNone(why, "%s: scan accepted (rows=%d)" % (label, len(rows)))
        if state is not None:
            state["fired"] = 0      # re-arm a one-shot fault for the second scan
        with self.assertRaises(self.h.HostFenceError) as caught:
            self.h._grok_instruction_rows(self.root)
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected", label)

    # (function, fixture key) for every call site the fixture exercises.
    SITES = (
        ("listdir", "root"), ("listdir", "rules"), ("listdir", "sub"),
        ("lstat", "root"), ("lstat", "outside_dir"),
        ("lstat", "agents"), ("readlink", "agents"),
        ("lstat", "mid"), ("readlink", "mid"), ("lstat", "target"),
        ("lstat", "claude"), ("lstat", "rules"),
        ("lstat", "topic_link"), ("readlink", "topic_link"), ("lstat", "topic"),
        ("lstat", "sub"), ("lstat", "x"),
    )
    # Documented exemption: persistent ENOENT on the last component of a
    # followed link's target is a dangling link (its path is still denied).
    # `mid` is AGENTS.md's own target, so a missing `mid` is dangling too.
    DANGLING = {("lstat", "target"), ("lstat", "topic"), ("lstat", "mid")}

    def test_every_call_site_refuses_every_error(self):
        red = []
        for func, key in self.SITES:
            for err in (errno.ENOENT, errno.EACCES, errno.ELOOP):
                label = "%s(%s) %s" % (func, key, errno.errorcode[err])
                patcher, state = self.inject(func, self.p[key], err)
                with self.subTest(site=label), patcher:
                    if (func, key) in self.DANGLING and err == errno.ENOENT:
                        rows, why = self.scan()
                        self.assertIsNone(why, label)
                        self.assertIn(("literal", self.p[key]), rows, label)
                    else:
                        self.assertRefused(label)
                    self.assertGreater(state["fired"], 0, "%s never reached" % label)

    def test_unconfirmed_enoent_at_a_dangling_site_refuses(self):
        for key in ("target", "topic", "mid"):
            patcher, state = self.inject("lstat", self.p[key], errno.ENOENT, once=True)
            with self.subTest(site=key), patcher:
                self.assertRefused("lstat(%s) ENOENT once" % key, state)

    def test_r4_1_r4_2_enoent_on_an_existing_entry_refuses(self):
        # Probe from the review: lstat ENOENT on an existing AGENTS.md, stat
        # untouched. The scan must refuse; a deny row alone is not enough.
        patcher, _state = self.inject("lstat", self.p["agents"], errno.ENOENT)
        with patcher:
            self.assertRefused("lstat(AGENTS.md) ENOENT")

    def test_absent_root_is_the_only_root_exemption(self):
        missing = os.path.join(self.out, "absent-root")
        rows, why = kernel_sandbox._grok_instruction_deny_read(missing)
        self.assertIsNone(why)
        self.assertIn(("literal", os.path.join(missing, "AGENTS.md")), rows)
        # listdir says ENOENT but the root exists: not absent, refuse.
        patcher, _state = self.inject("listdir", self.root, errno.ENOENT)
        with patcher:
            self.assertRefused("listdir(root) ENOENT, root exists")

    def test_clean_fixture_denies_every_final_target(self):
        rows, why = self.scan()
        self.assertIsNone(why)
        for key in ("target", "claude", "topic", "x"):
            self.assertIn(("literal", self.p[key]), rows, key)
        self.assertIn(("subpath", self.p["sub"]), rows)
        self.assertNotIn(("literal", self.p["mid"]), rows)   # a hop, not a target

    def test_scanner_uses_no_error_swallowing_call(self):
        def forbidden(name):
            def boom(*a, **k):
                raise AssertionError("scanner called %s" % name)
            return boom
        with mock.patch.object(kernel_sandbox.os, "stat", forbidden("os.stat")), \
                mock.patch.object(kernel_sandbox.os, "scandir", forbidden("os.scandir")), \
                mock.patch.object(kernel_sandbox.os.path, "realpath", forbidden("realpath")), \
                mock.patch.object(kernel_sandbox.os.path, "exists", forbidden("exists")), \
                mock.patch.object(kernel_sandbox.os.path, "isdir", forbidden("isdir")), \
                mock.patch.object(kernel_sandbox.os.path, "islink", forbidden("islink")), \
                mock.patch.object(kernel_sandbox.os.path, "lexists", forbidden("lexists")):
            rows, why = self.scan()
        self.assertIsNone(why)
        self.assertIn(("literal", self.p["target"]), rows)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
