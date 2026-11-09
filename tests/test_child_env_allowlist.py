"""Child environment contract of the host fence (issue #138).

Every vendor child launched through ``prepare_host_fence`` receives an
allowlisted environment, and cannot connect to the operator's ssh-agent
socket. Values here are synthetic.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from providers import host_fence as h
from providers import kernel_sandbox, provider_accounts

DARWIN_FENCE = kernel_sandbox.available()
PASS_VAR = "OMNI_LLM_CHILD_ENV_PASS"
PROVIDERS = ("claude", "codex", "grok", "cursor", "pi", "agy")

# Connect to argv[1]; print CONNECTED or FAIL <errno>.
CONNECT_PROBE = (
    "import socket, sys\n"
    "s = socket.socket(socket.AF_UNIX)\n"
    "try:\n"
    "    s.connect(sys.argv[1]); print('CONNECTED')\n"
    "except OSError as e:\n"
    "    print('FAIL', e.errno)\n"
)


def _short_tmp():
    # AF_UNIX paths are limited to ~104 bytes: stay out of long $TMPDIR trees.
    return tempfile.TemporaryDirectory(prefix="cea", dir="/tmp")


class ListeningSocket:
    """A bound, listening unix socket in a short temp dir."""

    def __init__(self):
        self.dir = _short_tmp()
        self.path = os.path.join(self.dir.name, "agent.sock")
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(self.path)
        self.server.listen(8)

    def close(self):
        self.server.close()
        self.dir.cleanup()


def _parent_env(**extra):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SYNTH_SECRET_TOKEN": "synthetic-not-a-secret",
        "AWS_SECRET_ACCESS_KEY": "synthetic-not-a-secret",
        "GH_TOKEN": "synthetic-not-a-secret",
        "GIT_DIR": "/nonexistent-synthetic",
        "SSH_AUTH_SOCK": "/nonexistent-synthetic/agent",
    }
    env.update(extra)
    return env


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class AllowlistedChildEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # os.environ must not leak a pass list or a socket into the cases.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(PASS_VAR, None)
        os.environ.pop("SSH_AUTH_SOCK", None)

    def prepare(self, provider, env, **kw):
        _argv, child_env, _meta = h.prepare_host_fence(
            ["/usr/bin/true"], provider=provider, workdir=self.tmp.name,
            read_only=True, env=env, **kw)
        return child_env

    def test_arbitrary_variable_is_absent_for_every_fenced_provider(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                child = self.prepare(provider, _parent_env())
                for name in ("SYNTH_SECRET_TOKEN", "AWS_SECRET_ACCESS_KEY",
                             "GH_TOKEN", "GIT_DIR", "SSH_AUTH_SOCK"):
                    self.assertFalse(name in child, name)

    def test_codex_consult_custom_policy_path_is_allowlisted_too(self):
        policy = kernel_sandbox.agent_fence_policy(
            "codex", workdir=self.tmp.name, read_only=True)
        child = self.prepare("codex", _parent_env(), policy=policy)
        self.assertNotIn("SYNTH_SECRET_TOKEN", child)
        self.assertNotIn("SSH_AUTH_SOCK", child)
        self.assertEqual(child["HOME"], kernel_sandbox.protected_home())

    def test_keep_set_survives(self):
        keep = {
            "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "LC_CTYPE": "UTF-8",
            "TZ": "UTC", "TERM": "xterm", "COLORTERM": "truecolor",
            "NO_COLOR": "1", "USER": "synth", "LOGNAME": "synth",
            "SHELL": "/bin/zsh",
            "http_proxy": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
            "no_proxy": "localhost", "ALL_PROXY": "socks5://127.0.0.1:1",
            "SSL_CERT_FILE": "/synthetic/ca.pem", "SSL_CERT_DIR": "/synthetic/ca",
            "NODE_EXTRA_CA_CERTS": "/synthetic/ca.pem",
            "REQUESTS_CA_BUNDLE": "/synthetic/ca.pem",
            "OMNI_LLM_DEPTH": "1", "OMNI_LLM_NOTIFY_QUIET": "1",
        }
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                child = self.prepare(provider, _parent_env(**keep))
                for name, value in keep.items():
                    self.assertEqual(child.get(name), value, name)
                self.assertTrue(child["PATH"])
                self.assertEqual(child["HOME"], kernel_sandbox.protected_home())
                self.assertEqual(child["OMNI_LLM_NOTIFY_QUIET"], "1")

    def test_omni_and_pi_prefixes_are_not_passed_wholesale(self):
        env = _parent_env(OMNI_LLM_WEBHOOK_TOKEN="x", PI_PRIVATE_TOKEN="x",
                          OMNI_LLM_STATE_DIR="/synthetic", PI_MODEL="m")
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                child = self.prepare(provider, env)
                for name in ("OMNI_LLM_WEBHOOK_TOKEN", "PI_PRIVATE_TOKEN",
                             "OMNI_LLM_STATE_DIR", "PI_MODEL"):
                    self.assertFalse(name in child, name)
                self.assertEqual(child["OMNI_LLM_NOTIFY_QUIET"], "1")

    def test_notify_sink_is_not_resurrected_by_the_omni_prefix(self):
        child = self.prepare("claude", _parent_env(OMNI_LLM_NOTIFY_CMD="synthetic"))
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", child)

    def test_omni_set_keys_survive(self):
        child = self.prepare("grok", _parent_env())
        self.assertEqual(child["GROK_DISABLE_AUTOUPDATER"], "1")

    def test_pass_variable_lists_extra_names_exactly(self):
        os.environ[PASS_VAR] = " SYNTH_PASSED_A ,SYNTH_PASSED_B,, SYNTH_*"
        env = _parent_env(SYNTH_PASSED_A="a", SYNTH_PASSED_B="b",
                          SYNTH_PASSED_C="c", SYNTH_STAR="d")
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                child = self.prepare(provider, env)
                self.assertEqual(child["SYNTH_PASSED_A"], "a")
                self.assertEqual(child["SYNTH_PASSED_B"], "b")
                self.assertNotIn("SYNTH_PASSED_C", child)    # no patterns
                self.assertNotIn("SYNTH_STAR", child)
                self.assertNotIn("SYNTH_SECRET_TOKEN", child)

    def test_pass_variable_never_resurrects_a_stripped_sink(self):
        os.environ[PASS_VAR] = "OMNI_LLM_NOTIFY_CMD"
        child = self.prepare("claude", _parent_env(OMNI_LLM_NOTIFY_CMD="synthetic"))
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", child)

    def test_non_darwin_environment_is_returned_unchanged(self):
        env = _parent_env()
        with mock.patch.object(h.platform, "system", return_value="Linux"):
            _argv, child, meta = h.prepare_host_fence(
                ["/usr/bin/true"], provider="claude", workdir=self.tmp.name, env=env)
        self.assertIs(child, env)
        self.assertEqual(meta["fence"], "none")


class AllowlistFunctionTests(unittest.TestCase):
    def test_home_config_and_cursor_keys_are_kept(self):
        base = {
            "CLAUDE_CONFIG_DIR": "/c", "CODEX_HOME": "/x", "GROK_HOME": "/g",
            "CURSOR_CONFIG_DIR": "/u", "OMNI_LLM_CURSOR_CONFIG_DIR_ALT": "/a",
            "OMNI_LLM_DEPTH": "1", "OMNI_LLM_WEBHOOK_TOKEN": "x",
            "SYNTH_SECRET_TOKEN": "x",
        }
        out = h.allowlisted_child_env(base, "claude")
        for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "GROK_HOME",
                     "CURSOR_CONFIG_DIR", "OMNI_LLM_DEPTH"):
            self.assertEqual(out[name], base[name])
        # omni's profile-ring names are read by the server, not by a child.
        for name in ("OMNI_LLM_CURSOR_CONFIG_DIR_ALT", "OMNI_LLM_WEBHOOK_TOKEN",
                     "SYNTH_SECRET_TOKEN"):
            self.assertNotIn(name, out)

    def test_vendor_auth_goes_to_its_own_vendor_only(self):
        base = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "XAI_API_KEY": "x",
                "CURSOR_API_KEY": "c", "CODEX_API_KEY": "k"}
        self.assertEqual(
            set(h.allowlisted_child_env(base, "claude")), {"ANTHROPIC_API_KEY"})
        self.assertEqual(
            set(h.allowlisted_child_env(base, "codex")),
            {"OPENAI_API_KEY", "CODEX_API_KEY"})
        self.assertEqual(
            set(h.allowlisted_child_env(base, "grok")), {"XAI_API_KEY"})
        self.assertEqual(
            set(h.allowlisted_child_env(base, "cursor")), {"CURSOR_API_KEY"})
        self.assertEqual(set(h.allowlisted_child_env(base, "pi")), set())

    def test_vendor_auth_table_matches_the_account_layer_denylist(self):
        # The names the account layer strips from an isolated child are the
        # vendor auth variables; the allowlist must name the same ones.
        for provider, names in h._VENDOR_AUTH_ENV.items():
            denied = provider_accounts.account_env(
                {name: "x" for name in names}, provider)
            # cursor auth rides profile overrides; agy has no account ring
            # (its two names are the profile variable and the login selector).
            if provider in ("cursor", "agy"):
                continue
            self.assertEqual(denied, {}, provider)

    def test_launch_context_markers_are_not_vendor_auth(self):
        base = {"CLAUDE_CODE_MESSAGING_TOKEN": "t", "CLAUDECODE": "1",
                "CLAUDE_CODE_SESSION_ID": "s"}
        self.assertEqual(h.allowlisted_child_env(base, "claude"), {})


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class SocketDenyProfileTests(unittest.TestCase):
    def test_empty_deny_keeps_the_profile_byte_identical(self):
        a = kernel_sandbox.build_profile("/tmp", True)
        b = kernel_sandbox.build_profile("/tmp", True, extra_deny_connect=())
        self.assertEqual(a, b)
        self.assertNotIn("network", a)

    def test_entries_become_remote_unix_socket_denies(self):
        profile = kernel_sandbox.build_profile(
            "/tmp", True, extra_deny_connect=(("literal", "/tmp/x"),))
        self.assertIn(
            '(deny network-outbound (remote unix-socket (path-literal (param "N0"))))',
            profile)
        params = kernel_sandbox.profile_params(
            "/tmp", True, extra_deny_connect=(("literal", "/tmp/x"),))
        self.assertEqual(params["N0"], "/tmp/x")

    def test_denies_cover_a_socket_in_either_spelling_and_nothing_else(self):
        sock = ListeningSocket()
        self.addCleanup(sock.close)
        link_dir = _short_tmp()
        self.addCleanup(link_dir.cleanup)
        link = os.path.join(link_dir.name, "link.sock")
        os.symlink(sock.path, link)
        real = os.path.realpath(sock.path)
        self.assertEqual(
            kernel_sandbox.ssh_agent_socket_denies({"SSH_AUTH_SOCK": sock.path}),
            tuple(dict.fromkeys((("literal", real), ("literal", sock.path)))))
        spelled = kernel_sandbox.ssh_agent_socket_denies({"SSH_AUTH_SOCK": link})
        self.assertIn(("literal", real), spelled)
        self.assertIn(("literal", link), spelled)
        # Unset, empty, missing and non-socket values add nothing.
        self.assertEqual(kernel_sandbox.ssh_agent_socket_denies({}), ())
        self.assertEqual(
            kernel_sandbox.ssh_agent_socket_denies({"SSH_AUTH_SOCK": ""}), ())
        self.assertEqual(kernel_sandbox.ssh_agent_socket_denies(
            {"SSH_AUTH_SOCK": os.path.join(link_dir.name, "none")}), ())
        self.assertEqual(kernel_sandbox.ssh_agent_socket_denies(
            {"SSH_AUTH_SOCK": link_dir.name}), ())
        self.assertEqual(kernel_sandbox.ssh_agent_socket_denies(
            {"SSH_AUTH_SOCK": os.devnull}), ())


def _firm(path):
    """The data-volume spelling of *path*, or None where there is none."""
    real = os.path.realpath(path)
    spelled = "/System/Volumes/Data" + real
    try:
        if os.path.samestat(os.stat(spelled), os.stat(real)):
            return spelled
    except OSError:
        pass
    return None


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class FirmlinkSpellingTests(unittest.TestCase):
    """A rule spelled with the data-volume prefix never matches (measured in
    test_same_location), so the deny must carry the plain spelling too."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)
        self.agent = ListeningSocket()
        self.addCleanup(self.agent.close)
        self.firm = _firm(self.agent.path)
        if self.firm is None:
            self.skipTest("no firmlink spelling on this host")

    def test_denies_carry_the_plain_spelling_of_a_firmlink_value(self):
        denies = kernel_sandbox.ssh_agent_socket_denies({"SSH_AUTH_SOCK": self.firm})
        self.assertIn(("literal", os.path.realpath(self.agent.path)), denies)

    def probe(self, path, env):
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", CONNECT_PROBE, path], provider="claude",
            workdir=self.tmp.name, read_only=True, env=env)
        out = subprocess.run(argv, env=child_env, cwd=self.tmp.name,
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip(), out.stderr.strip()

    def test_firmlink_value_denies_the_socket_under_every_spelling(self):
        # Positive control: unfenced, the firmlink spelling connects.
        out = subprocess.run([sys.executable, "-c", CONNECT_PROBE, self.firm],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "CONNECTED")
        env = _parent_env(SSH_AUTH_SOCK=self.firm)
        for label, path in (("plain", self.agent.path),
                            ("real", os.path.realpath(self.agent.path)),
                            ("firm", self.firm)):
            with self.subTest(spelling=label):
                out, err = self.probe(path, env)
                self.assertEqual(out, "FAIL 1", err)


RELOCATE_CHILD = """
import os, socket, sys, json
sock = sys.argv[1]
parent = os.path.dirname(sock)
def conn(p):
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(p)
        return 'CONNECTED'
    except OSError as e:
        return 'FAIL %s' % e.errno
out = {}
new = sock + '.moved'
try:
    os.rename(sock, new)
    out['rename_sock'] = conn(new)
    os.rename(new, sock)
except OSError as e:
    out['rename_sock'] = 'RENAME-FAIL %s' % e.errno
newdir = parent + '.moved'
try:
    os.rename(parent, newdir)
    out['rename_parent'] = conn(os.path.join(newdir, os.path.basename(sock)))
    os.rename(newdir, parent)
except OSError as e:
    out['rename_parent'] = 'RENAME-FAIL %s' % e.errno
try:
    with open(os.path.join(parent, 'sibling.txt'), 'w') as f:
        f.write('x')
    out['sibling_write'] = 'OK'
except OSError as e:
    out['sibling_write'] = 'FAIL %s' % e.errno
print(json.dumps(out))
"""


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class SocketRelocationTests(unittest.TestCase):
    """A writable root must not let the child move the socket (or a parent
    directory) and connect through the new pathname."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)

    def run_child(self, base, name_socket=True):
        parent = tempfile.mkdtemp(prefix="cex", dir=base)
        self.addCleanup(shutil.rmtree, parent + ".moved", True)
        self.addCleanup(shutil.rmtree, parent, True)
        path = os.path.join(parent, "agent.sock")
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        server.listen(4)
        self.addCleanup(server.close)
        env = _parent_env()
        env.pop("SSH_AUTH_SOCK")
        if name_socket:
            env["SSH_AUTH_SOCK"] = path
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", RELOCATE_CHILD, path], provider="claude",
            workdir=self.work.name, read_only=False, env=env)
        out = subprocess.run(argv, env=child_env, cwd=self.work.name,
                             capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout.strip() or "{}"), out.stderr

    def assertNotReachable(self, row, err, writable=True):
        for key in ("rename_sock", "rename_parent"):
            self.assertTrue(row.get(key, "").startswith(("RENAME-FAIL", "FAIL")),
                            (key, row, err))
        if writable:
            # The deny must not cost the child its ordinary writes there.
            self.assertEqual(row.get("sibling_write"), "OK", (row, err))

    def test_relocation_works_without_the_deny_positive_control(self):
        # The layout is writable to the fence: only the deny stops the move.
        row, err = self.run_child(self.work.name, name_socket=False)
        self.assertEqual(row.get("rename_sock"), "CONNECTED", (row, err))
        self.assertEqual(row.get("rename_parent"), "CONNECTED", (row, err))

    def test_socket_under_the_workdir_cannot_be_relocated(self):
        row, err = self.run_child(self.work.name)
        self.assertNotReachable(row, err)

    def test_socket_under_the_temp_root_cannot_be_relocated(self):
        row, err = self.run_child(tempfile.gettempdir())
        self.assertNotReachable(row, err)

    def test_socket_under_the_protected_home_cannot_be_relocated(self):
        row, err = self.run_child(kernel_sandbox.protected_home())
        self.assertNotReachable(row, err, writable=False)


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class FencedConnectProbeTests(unittest.TestCase):
    """Real sandbox-exec: the fenced child cannot reach the agent socket."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)
        self.agent = ListeningSocket()
        self.addCleanup(self.agent.close)
        self.other = ListeningSocket()
        self.addCleanup(self.other.close)

    def probe(self, path, env, provider="claude"):
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", CONNECT_PROBE, path], provider=provider,
            workdir=self.tmp.name, read_only=True, env=env)
        out = subprocess.run(argv, env=child_env, cwd=self.tmp.name,
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip(), out.stderr.strip()

    def test_unfenced_connect_succeeds_positive_control(self):
        out = subprocess.run([sys.executable, "-c", CONNECT_PROBE, self.agent.path],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "CONNECTED")

    def test_fenced_connect_to_the_agent_socket_is_refused(self):
        env = _parent_env(SSH_AUTH_SOCK=self.agent.path)
        for provider in ("claude", "codex", "pi"):
            with self.subTest(provider=provider):
                out, err = self.probe(self.agent.path, env, provider)
                self.assertEqual(out, "FAIL 1", err)   # EPERM

    def test_other_sockets_stay_reachable_so_the_deny_is_what_blocks(self):
        env = _parent_env(SSH_AUTH_SOCK=self.agent.path)
        out, err = self.probe(self.other.path, env)
        self.assertEqual(out, "CONNECTED", err)

    def test_a_symlink_spelling_of_the_socket_is_refused_too(self):
        link_dir = _short_tmp()
        self.addCleanup(link_dir.cleanup)
        link = os.path.join(link_dir.name, "link.sock")
        os.symlink(self.agent.path, link)
        env = _parent_env(SSH_AUTH_SOCK=self.agent.path)
        out, err = self.probe(link, env)
        self.assertEqual(out, "FAIL 1", err)

    def test_pass_variable_does_not_reopen_the_socket(self):
        os.environ[PASS_VAR] = "SSH_AUTH_SOCK"
        env = _parent_env(SSH_AUTH_SOCK=self.agent.path)
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", CONNECT_PROBE, self.agent.path],
            provider="claude", workdir=self.tmp.name, read_only=True, env=env)
        self.assertEqual(child_env["SSH_AUTH_SOCK"], self.agent.path)
        out = subprocess.run(argv, env=child_env, cwd=self.tmp.name,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "FAIL 1", out.stderr)

    def test_socket_named_only_in_the_server_environment_is_denied(self):
        os.environ["SSH_AUTH_SOCK"] = self.agent.path
        out, err = self.probe(self.agent.path, _parent_env(SSH_AUTH_SOCK="/nonexistent"))
        self.assertEqual(out, "FAIL 1", err)

    def test_unset_or_non_socket_agent_variable_adds_no_clause(self):
        for value in (None, "", "/nonexistent-synthetic/agent", self.tmp.name):
            with self.subTest(value=value):
                env = _parent_env()
                env.pop("SSH_AUTH_SOCK")
                if value is not None:
                    env["SSH_AUTH_SOCK"] = value
                argv, _env, _meta = h.prepare_host_fence(
                    ["/usr/bin/true"], provider="claude", workdir=self.tmp.name,
                    read_only=True, env=env)
                self.assertNotIn("network-outbound", argv[2])


LINK_CHILD = """
import os, socket, sys, json
sock, dest = sys.argv[1], sys.argv[2]
out = {}
try:
    os.link(sock, dest)
    out['link'] = 'LINK_OK'
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(dest)
        out['connect'] = 'CONNECTED'
    except OSError as e:
        out['connect'] = 'FAIL %s' % e.errno
except OSError as e:
    out['link'] = 'LINK-FAIL %s' % e.errno
print(json.dumps(out))
"""


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class SocketAliasTests(unittest.TestCase):
    """A hard link to the socket is a second name the connect deny does not
    cover; the profile must refuse to create one."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)

    def prepare(self, argv, path):
        return h.prepare_host_fence(
            argv, provider="claude", workdir=self.work.name, read_only=False,
            env=_parent_env(SSH_AUTH_SOCK=path))

    def test_profile_denies_file_link_on_every_spelling_of_the_socket(self):
        agent = ListeningSocket()
        self.addCleanup(agent.close)
        argv, _env, _meta = self.prepare(["/usr/bin/true"], agent.path)
        profile = argv[2]
        params = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-D"]
        linked = [p.split("=", 1)[1] for p in params if p.startswith("L")]
        self.assertIn("(deny file-link", profile)
        for _kind, path in kernel_sandbox.ssh_agent_socket_denies(
                {"SSH_AUTH_SOCK": agent.path}):
            self.assertIn(path, linked)

    def link_in(self, base):
        parent = tempfile.mkdtemp(prefix="cex", dir=base)
        self.addCleanup(shutil.rmtree, parent, True)
        dest_dir = tempfile.mkdtemp(prefix="dst", dir=self.work.name)
        path = os.path.join(parent, "agent.sock")
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        server.listen(4)
        self.addCleanup(server.close)
        dest = os.path.join(dest_dir, "alias.sock")
        argv, child_env, _meta = self.prepare(
            [sys.executable, "-c", LINK_CHILD, path, dest], path)
        out = subprocess.run(argv, env=child_env, cwd=self.work.name,
                             capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout.strip() or "{}"), out.stderr

    def test_a_hard_link_to_the_socket_cannot_be_made_or_used(self):
        for label, base in (("workdir", self.work.name),
                            ("temp root", tempfile.gettempdir())):
            with self.subTest(layout=label):
                row, err = self.link_in(base)
                self.assertTrue(row.get("link", "").startswith("LINK-FAIL"), (row, err))
                self.assertNotEqual(row.get("connect"), "CONNECTED")

    def test_the_link_control_a_socket_not_named_can_be_linked(self):
        # Positive control: the refusal is the deny's, not the host's.
        parent = tempfile.mkdtemp(prefix="cex", dir=self.work.name)
        path = os.path.join(parent, "agent.sock")
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        server.listen(4)
        self.addCleanup(server.close)
        dest = os.path.join(self.work.name, "alias.sock")
        env = _parent_env()
        env.pop("SSH_AUTH_SOCK")
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", LINK_CHILD, path, dest], provider="claude",
            workdir=self.work.name, read_only=False, env=env)
        out = subprocess.run(argv, env=child_env, cwd=self.work.name,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(out.stdout.strip()).get("link"), "LINK_OK", out.stderr)


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class SocketSpellingCaseTests(unittest.TestCase):
    """Seatbelt rules and APFS both ignore case: every case spelling of the
    data-volume prefix and of the path itself must stay denied."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)
        self.agent = ListeningSocket()
        self.addCleanup(self.agent.close)
        self.real = os.path.realpath(self.agent.path)

    def probe(self, path, env):
        argv, child_env, _meta = h.prepare_host_fence(
            [sys.executable, "-c", CONNECT_PROBE, path], provider="claude",
            workdir=self.tmp.name, read_only=True, env=env)
        out = subprocess.run(argv, env=child_env, cwd=self.tmp.name,
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip(), out.stderr.strip()

    def test_lowercase_data_volume_prefix_still_yields_the_plain_deny(self):
        value = "/system/volumes/data" + self.real
        try:
            if not os.path.samestat(os.stat(value), os.stat(self.real)):
                self.skipTest("no case-insensitive firmlink spelling here")
        except OSError:
            self.skipTest("no case-insensitive firmlink spelling here")
        denies = kernel_sandbox.ssh_agent_socket_denies({"SSH_AUTH_SOCK": value})
        self.assertIn(("literal", self.real), denies)
        # Positive control, then the fenced probe by the plain spelling.
        out = subprocess.run([sys.executable, "-c", CONNECT_PROBE, value],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "CONNECTED")
        out, err = self.probe(self.real, _parent_env(SSH_AUTH_SOCK=value))
        self.assertEqual(out, "FAIL 1", err)

    def test_swapped_case_spelling_of_the_path_is_denied(self):
        swapped = self.real.swapcase()
        control = subprocess.run([sys.executable, "-c", CONNECT_PROBE, swapped],
                                 capture_output=True, text=True, timeout=60)
        if control.stdout.strip() != "CONNECTED":
            self.skipTest("case-sensitive volume")
        out, err = self.probe(swapped, _parent_env(SSH_AUTH_SOCK=self.agent.path))
        self.assertEqual(out, "FAIL 1", err)


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class SocketInspectionFailureTests(unittest.TestCase):
    """Only a confirmed absent path or a confirmed non-socket may yield no
    deny; a path that cannot be inspected must refuse the spawn."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)

    def locked_parent(self):
        parent = tempfile.mkdtemp(prefix="cex", dir=tempfile.gettempdir())
        path = os.path.join(parent, "agent.sock")
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        server.listen(2)
        os.chmod(parent, 0)
        self.addCleanup(shutil.rmtree, parent, True)
        self.addCleanup(os.chmod, parent, 0o700)
        self.addCleanup(server.close)
        return path

    def test_unreadable_parent_refuses_the_spawn_loudly(self):
        path = self.locked_parent()
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(
                ["/usr/bin/true"], provider="claude", workdir=self.tmp.name,
                read_only=True, env=_parent_env(SSH_AUTH_SOCK=path))
        envelope = caught.exception.envelope
        self.assertEqual(envelope["reason"], "fence_path_rejected")
        self.assertIn("SSH_AUTH_SOCK", envelope["detail"])
        self.assertNotIn(path, envelope["detail"])

    def test_symlink_loop_refuses_the_spawn_loudly(self):
        loop = os.path.join(self.tmp.name, "loop")
        os.symlink(loop, loop)
        with self.assertRaises(h.HostFenceError) as caught:
            h.prepare_host_fence(
                ["/usr/bin/true"], provider="claude", workdir=self.tmp.name,
                read_only=True, env=_parent_env(SSH_AUTH_SOCK=loop))
        self.assertEqual(caught.exception.envelope["reason"], "fence_path_rejected")

    def test_confirmed_absent_and_non_socket_do_not_refuse(self):
        for value in (os.path.join(self.tmp.name, "gone"),
                      os.path.join(self.tmp.name, "gone", "deeper"),
                      self.tmp.name):
            with self.subTest(value=os.path.basename(value)):
                h.prepare_host_fence(
                    ["/usr/bin/true"], provider="claude", workdir=self.tmp.name,
                    read_only=True, env=_parent_env(SSH_AUTH_SOCK=value))


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class GitIdentityTests(unittest.TestCase):
    """The fence disables global and system git config, so the launcher's
    explicit identity is the only one a repo-write child has."""

    def test_commit_inside_the_fence_carries_the_launcher_identity(self):
        with tempfile.TemporaryDirectory(prefix="cex_git_", dir="/private/tmp") as raw:
            root = os.path.realpath(raw)
            main = os.path.join(root, "main")
            linked = os.path.join(root, "linked")
            os.mkdir(main)
            identity = {**os.environ, "GIT_AUTHOR_NAME": "Synth Author",
                        "GIT_AUTHOR_EMAIL": "author@example.invalid",
                        "GIT_COMMITTER_NAME": "Synth Committer",
                        "GIT_COMMITTER_EMAIL": "committer@example.invalid"}
            for args in (["init", "-q"], ["commit", "--allow-empty", "-qm", "seed"],
                         ["worktree", "add", "-qb", "feature", linked]):
                subprocess.run(["git", "-C", main, *args], check=True, env=identity,
                               capture_output=True)
            argv, child_env, _meta = h.prepare_host_fence(
                ["/usr/bin/git", "-C", linked, "commit", "--allow-empty", "-qm", "fenced"],
                provider="claude", workdir=linked, env=identity, allow_repo_write=True)
            out = subprocess.run(argv, cwd=linked, env=child_env,
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            shown = subprocess.run(
                ["git", "-C", linked, "log", "-1", "--format=%an|%ae|%cn|%ce"],
                check=True, capture_output=True, text=True, env=identity).stdout.strip()
            self.assertEqual(shown, "Synth Author|author@example.invalid|"
                                    "Synth Committer|committer@example.invalid")


class NotifySinkAlwaysDroppedTests(unittest.TestCase):
    """README: OMNI_LLM_NOTIFY_CMD is stripped even when the operator lists
    it, on every path that builds a child environment."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {
            "OMNI_LLM_NOTIFY_CMD": "synthetic-sink", PASS_VAR: "OMNI_LLM_NOTIFY_CMD"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_allowlist_function_drops_it(self):
        out = h.allowlisted_child_env(dict(os.environ), "codex")
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", out)

    @unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
    def test_codex_custom_policy_path_drops_it(self):
        with tempfile.TemporaryDirectory() as work:
            policy = kernel_sandbox.agent_fence_policy(
                "codex", workdir=work, read_only=True)
            _argv, child, _meta = h.prepare_host_fence(
                ["/usr/bin/true"], provider="codex", workdir=work, read_only=True,
                env=dict(os.environ), policy=policy)
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", child)

    def test_write_canary_probe_drops_it(self):
        import asyncio
        from providers import sandbox_canary
        seen = {}

        async def fake_run_cli(argv, **kwargs):
            seen.update(kwargs)
            return {"timed_out": False, "stdout": "", "stderr": "", "returncode": 1}

        with mock.patch.object(sandbox_canary.vendor_bin, "which", return_value="/bin/echo"), \
                mock.patch.object(sandbox_canary, "run_cli", fake_run_cli):
            asyncio.run(sandbox_canary._run_vendor_readonly_probe("p", 5))
        self.assertNotIn("OMNI_LLM_NOTIFY_CMD", seen["env"])


class UnfencedChildEnvTests(unittest.TestCase):
    """Children launched outside the fence use the same allowlist."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, _parent_env(
            OMNI_LLM_WEBHOOK_TOKEN="x", SYNTH_PASSED="p", OPENAI_API_KEY="k",
            ANTHROPIC_API_KEY="a"))
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ[PASS_VAR] = "SYNTH_PASSED"

    def check(self, env, codex_key=True):
        for name in ("SYNTH_SECRET_TOKEN", "AWS_SECRET_ACCESS_KEY", "GH_TOKEN",
                     "SSH_AUTH_SOCK", "OMNI_LLM_WEBHOOK_TOKEN", "ANTHROPIC_API_KEY"):
            self.assertFalse(name in env, name)
        self.assertEqual(env.get("SYNTH_PASSED"), "p")
        if codex_key:     # codex's own auth (the quota probe strips it already)
            self.assertEqual(env.get("OPENAI_API_KEY"), "k")
        self.assertTrue(env.get("PATH"))

    def test_codex_quota_probe_child_gets_the_allowlist(self):
        from providers import codex_quota
        with mock.patch.object(codex_quota.vendor_bin, "which", return_value="/bin/echo"), \
                mock.patch.object(codex_quota.subprocess, "Popen") as popen, \
                mock.patch.object(codex_quota, "_rpc_session", return_value=({}, None)), \
                mock.patch.object(codex_quota, "_kill_group"):
            codex_quota._spawn_rate_limits("default", 1, overrides={})
        self.check(popen.call_args.kwargs["env"], codex_key=False)

    def test_write_canary_vendor_probe_child_gets_the_allowlist(self):
        import asyncio
        from providers import sandbox_canary
        seen = {}

        async def fake_run_cli(argv, **kwargs):
            seen.update(kwargs)
            return {"timed_out": False, "stdout": "", "stderr": "", "returncode": 1}

        with mock.patch.object(sandbox_canary.vendor_bin, "which", return_value="/bin/echo"), \
                mock.patch.object(sandbox_canary, "run_cli", fake_run_cli):
            asyncio.run(sandbox_canary._run_vendor_readonly_probe("p", 5))
        self.check(seen["env"])


if __name__ == "__main__":
    unittest.main()
