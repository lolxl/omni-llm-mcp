#!/usr/bin/env python3
# no-opencc
"""agy subprocess lane inside the host fence (issue #161).

The agy lane runs ``agy -p`` through ``prepare_host_fence`` like the other
vendor lanes. These cases pin the agy policy (instruction-file deny rows, the
home read wall, the minimal vendor-state writes, the child environment) and the
lane wiring. The kernel cases build a synthetic home in a temp dir and run a
probe under the real Seatbelt wrap (Darwin only); no vendor CLI, no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from providers import agy as agy_mod
from providers import host_fence as h
from providers import kernel_sandbox as ks

DARWIN_FENCE = ks.available()

# Verified instruction loads of agy 1.3.3 (see the comment above
# ``_AGY_STATE_ROOT`` in kernel_sandbox): both dirs, both names.
FAMILY = (
    (".gemini", "GEMINI.md"), (".gemini", "AGENTS.md"),
    (".gemini", "config", "GEMINI.md"), (".gemini", "config", "AGENTS.md"),
)
CONFIG_NAMES = (
    (".gemini", "settings.json"),
    (".gemini", "config", "hooks.json"),
    (".gemini", "config", "mcp_config.json"),
    (".gemini", "antigravity-cli", "settings.json"),
)
# State a previous run left that a fenced run must neither read nor write.
SENSITIVE = (
    (".gemini", "antigravity-cli", "brain"),
    (".gemini", "antigravity-cli", "conversations"),
    (".gemini", "antigravity-cli", "annotations"),
    (".gemini", "antigravity-cli", "history.jsonl"),
    (".gemini", "antigravity-cli", "conversation_summaries.db"),
    (".gemini", "antigravity-cli", "cache", "last_conversations.json"),
    (".gemini", "config", "hooks.json"),
    (".gemini", "config", "mcp_config.json"),
    (".gemini", "settings.json"),
    (".gemini", "antigravity-cli", "settings.json"),
    (".gemini", "antigravity-cli", "bin"),
)

PROBE = (
    "import os, sys\n"
    "out = {}\n"
    "for spec in sys.argv[1:]:\n"
    "    mode, path = spec.split(':', 1)\n"
    "    try:\n"
    "        if mode == 'r':\n"
    "            open(path, 'rb').read(1)\n"
    "        elif mode == 'l':\n"
    "            os.link(*path.split('>'))\n"
    "        else:\n"
    "            open(path, 'wb').write(b'x')\n"
    "        out[spec] = 0\n"
    "    except OSError as e:\n"
    "        out[spec] = e.errno\n"
    "import json; print(json.dumps(out))\n"
)


def _firm(path):
    """The data-volume (firmlink) spelling of *path*, or None where none exists."""
    real = os.path.realpath(path)
    spelled = "/System/Volumes/Data" + real
    try:
        if os.path.samestat(os.stat(spelled), os.stat(real)):
            return spelled
    except OSError:
        pass
    return None


def _touch(path, text="synthetic\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class AgyBase(unittest.TestCase):
    """A synthetic home shaped like a logged-in agy 1.3.3 install."""

    def setUp(self):
        # Under /tmp, not $TMPDIR: the fence always allows writes below
        # /private/var/folders, which would make the synthetic home writable.
        self.home = os.path.realpath(
            tempfile.mkdtemp(prefix="omni_agy_home_", dir="/tmp"))
        self.wd = os.path.realpath(tempfile.mkdtemp(prefix="omni_agy_wd_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.addCleanup(shutil.rmtree, self.wd, True)
        patcher = mock.patch.object(ks, "protected_home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SSH_AUTH_SOCK", None)
        for parts in FAMILY + CONFIG_NAMES + SENSITIVE:
            if parts[-1] in ("brain", "conversations", "annotations", "bin"):
                os.makedirs(os.path.join(self.home, *parts), exist_ok=True)
            else:
                _touch(os.path.join(self.home, *parts))
        self.agy_dir = os.path.join(self.home, ".gemini", "antigravity-cli")
        for parts in (("installation_id",), ("cache", "default_project_id.txt")):
            _touch(os.path.join(self.agy_dir, *parts))
        for name in ("log", "crashes"):
            os.makedirs(os.path.join(self.agy_dir, name), exist_ok=True)
        _touch(os.path.join(self.home, ".gemini", "config", "projects",
                            "default-cli-project.json"))
        _touch(os.path.join(self.home, "Library", "Keychains",
                            "login.keychain-db"))

    def p(self, *parts):
        return os.path.join(self.home, *parts)

    def policy(self):
        return ks.agent_fence_policy("agy", workdir=self.wd, read_only=True)

    def wrap_kw(self):
        policy = self.policy()
        self.assertNotIn("ok", policy, policy)
        return ks.wrap_kwargs_from_policy(policy)

class AgyInstructionDeny(AgyBase):
    def test_family_is_read_denied_as_plain_and_final_rows(self):
        kw = self.wrap_kw()
        for parts in FAMILY:
            row = ("literal", self.p(*parts))
            with self.subTest(path=parts):
                self.assertIn(row, kw["extra_deny_read"])
                self.assertIn(row, kw["extra_deny_read_final"])

    def test_rules_directories_are_read_denied(self):
        kw = self.wrap_kw()
        for rel in ((".gemini", "rules"), (".gemini", "config", "rules")):
            self.assertIn(("subpath", self.p(*rel)), kw["extra_deny_read_final"])

    def test_configuration_names_agy_loads_are_read_and_write_denied(self):
        kw = self.wrap_kw()
        for parts in CONFIG_NAMES:
            row = ("literal", self.p(*parts))
            with self.subTest(path=parts):
                self.assertIn(row, kw["extra_deny_read_final"])
                self.assertIn(row, kw["extra_deny_write_final"])
                self.assertIn(row, kw["extra_deny_write"])
        for rel in ((".gemini", "antigravity-cli", "bin"),
                    (".gemini", "antigravity-cli", "builtin"),
                    (".gemini", "config", "hooks")):
            self.assertIn(("subpath", self.p(*rel)),
                          kw["extra_deny_write_final"])

    def test_family_and_configuration_rows_are_final_write_denies_too(self):
        kw = self.wrap_kw()
        rows = [("literal", self.p(*parts)) for parts in FAMILY + CONFIG_NAMES]
        rows += [("subpath", self.p(".gemini", "rules")),
                 ("subpath", self.p(".gemini", "config", "rules"))]
        for row in rows:
            with self.subTest(row=row):
                self.assertIn(row, kw["extra_deny_write_final"])
                self.assertIn(row, kw["extra_deny_write"])

    def test_family_names_under_projects_are_denied(self):
        kw = self.wrap_kw()
        projects = self.p(".gemini", "config", "projects")
        for name in ("GEMINI.md", "AGENTS.md"):
            row = ("literal", os.path.join(projects, name))
            self.assertIn(row, kw["extra_deny_read_final"])
            self.assertIn(row, kw["extra_deny_write_final"])
        self.assertIn(("subpath", os.path.join(projects, "rules")),
                      kw["extra_deny_read_final"])

    def test_hard_linked_family_file_refuses_the_spawn(self):
        os.link(self.p(".gemini", "AGENTS.md"), self.p("alias.md"))
        policy = self.policy()
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("hard links", policy["detail"])


class AgyReadWall(AgyBase):
    EXPECTED_READS = {
        ("literal", ("installation_id",)),
        ("literal", ("cache", "default_project_id.txt")),
        ("subpath", ("log",)),
        ("subpath", ("crashes",)),
    }

    def test_read_wall_is_on_and_reallows_only_the_login_leaves(self):
        kw = self.wrap_kw()
        self.assertTrue(kw["read_wall"])
        got = {(kind, path) for kind, path in kw["extra_post_read"]}
        want = {(kind, os.path.join(self.agy_dir, *parts))
                for kind, parts in self.EXPECTED_READS}
        want.add(("subpath", self.p(".gemini", "config", "projects")))
        self.assertEqual(got, want)
        admitted = {ap.path for ap in kw["extra_read_paths"]}
        self.assertEqual(admitted, {
            self.p("Library", "Keychains", "login.keychain-db"),
            self.p("Library", "Keychains")})

    def test_no_read_allow_reaches_prior_conversations_or_configuration(self):
        kw = self.wrap_kw()
        allows = list(kw["extra_post_read"]) + [
            (ap.kind, ap.path) for ap in kw["extra_read_paths"]]
        for parts in SENSITIVE + FAMILY:
            rows = [("subpath" if os.path.isdir(self.p(*parts)) else "literal",
                     self.p(*parts))]
            for entry in allows:
                with self.subTest(path=parts, allow=entry):
                    self.assertFalse(h._reopens(entry, rows))

    def test_state_leaf_that_is_a_symlink_refuses(self):
        outside = tempfile.mkdtemp(prefix="omni_agy_out_")
        self.addCleanup(shutil.rmtree, outside, True)
        shutil.rmtree(os.path.join(self.agy_dir, "log"))
        os.symlink(outside, os.path.join(self.agy_dir, "log"))
        policy = self.policy()
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertIn("is a symlink", policy["detail"])


# (label, relative path parts, is a directory) for every class of path the layout
# rule covers. ``rules/`` content and the steering directories are created by
# ``AgyLinkRefusal.setUp``.
LAYOUT = (
    ("dot dir", (".gemini",), True),
    ("config dir", (".gemini", "config"), True),
    ("projects dir", (".gemini", "config", "projects"), True),
    ("projects content", (".gemini", "config", "projects",
                          "default-cli-project.json"), False),
    ("state root", (".gemini", "antigravity-cli"), True),
    ("state leaf installation_id", (".gemini", "antigravity-cli",
                                    "installation_id"), False),
    ("state leaf cache dir", (".gemini", "antigravity-cli", "cache"), True),
    ("state leaf project id", (".gemini", "antigravity-cli", "cache",
                               "default_project_id.txt"), False),
    ("state leaf log", (".gemini", "antigravity-cli", "log"), True),
    ("state leaf crashes", (".gemini", "antigravity-cli", "crashes"), True),
    ("family GEMINI.md", (".gemini", "GEMINI.md"), False),
    ("family AGENTS.md", (".gemini", "AGENTS.md"), False),
    ("family config GEMINI.md", (".gemini", "config", "GEMINI.md"), False),
    ("family config AGENTS.md", (".gemini", "config", "AGENTS.md"), False),
    ("family rules dir", (".gemini", "rules"), True),
    ("family rules content", (".gemini", "rules", "r.md"), False),
    ("family config rules content", (".gemini", "config", "rules", "r.md"), False),
    ("family projects GEMINI.md", (".gemini", "config", "projects",
                                   "GEMINI.md"), False),
    ("steering settings.json", (".gemini", "settings.json"), False),
    ("steering hooks.json", (".gemini", "config", "hooks.json"), False),
    ("steering mcp_config.json", (".gemini", "config", "mcp_config.json"), False),
    ("steering state settings.json", (".gemini", "antigravity-cli",
                                      "settings.json"), False),
    ("steering bin dir", (".gemini", "antigravity-cli", "bin"), True),
    ("steering builtin dir", (".gemini", "antigravity-cli", "builtin"), True),
    ("steering hooks dir", (".gemini", "config", "hooks"), True),
)


class AgyLinkRefusal(AgyBase):
    """Fail closed on a linked or aliased agy dot tree (a class rule)."""

    def setUp(self):
        super().setUp()
        for rel in ((".gemini", "rules", "r.md"), (".gemini", "config", "rules", "r.md"),
                    (".gemini", "config", "projects", "GEMINI.md")):
            _touch(self.p(*rel))
        for rel in ((".gemini", "antigravity-cli", "builtin"),
                    (".gemini", "config", "hooks")):
            os.makedirs(self.p(*rel), exist_ok=True)
        # a baseline that must NOT be refused: the plain layout.
        self.assertNotIn("ok", self.policy())

    def assertRefusal(self, policy, word, label):
        self.assertEqual(policy.get("reason"), "fence_path_rejected",
                         (label, policy))
        detail = policy["detail"]
        self.assertIn(word, detail, (label, detail))
        self.assertIn("agy fence rule", detail, (label, detail))
        self.assertIn("~/.gemini", detail, (label, detail))

    def elsewhere(self, name):
        path = os.path.join(self.home, "elsewhere", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def relink(self, parts, is_dir, target_spelling=None):
        path = self.p(*parts)
        target = self.elsewhere("-".join(parts))
        shutil.move(path, target)
        os.symlink(target_spelling(target) if target_spelling else target, path)

    def test_every_class_of_path_that_is_a_symlink_refuses(self):
        for label, parts, is_dir in LAYOUT:
            with self.subTest(path=label):
                self.setUp()
                self.relink(parts, is_dir)
                self.assertRefusal(self.policy(), "is a symlink", label)

    def test_every_regular_file_with_a_second_name_refuses(self):
        for label, parts, is_dir in LAYOUT:
            if is_dir:
                continue          # a directory cannot be hard-linked
            with self.subTest(path=label):
                self.setUp()
                os.link(self.p(*parts), self.elsewhere("alias-" + label))
                self.assertRefusal(self.policy(), "hard links", label)

    def test_a_symlink_to_a_firmlink_spelled_target_refuses_on_every_class(self):
        if _firm(self.home) is None:
            self.skipTest("no firmlink spelling on this host")
        for label, parts, is_dir in LAYOUT:
            with self.subTest(path=label):
                self.setUp()
                self.relink(parts, is_dir, target_spelling=_firm)
                self.assertRefusal(self.policy(), "is a symlink", label)

    def test_a_home_spelled_through_the_data_volume_refuses(self):
        firm = _firm(self.home)
        if firm is None:
            self.skipTest("no firmlink spelling on this host")
        with mock.patch.object(ks, "protected_home", return_value=firm):
            policy = self.policy()
        self.assertRefusal(policy, "spelled through /System/Volumes/Data", "home")

    def test_a_home_reached_through_a_symlink_refuses(self):
        via = os.path.join(os.path.dirname(self.home), "via-" + os.path.basename(self.home))
        os.symlink(self.home, via)
        self.addCleanup(os.remove, via)
        with mock.patch.object(ks, "protected_home", return_value=via):
            policy = self.policy()
        self.assertRefusal(policy, "resolves elsewhere", "home via link")

    def test_two_hop_dangling_and_relocated_target_layouts_all_refuse(self):
        # two hops: the middle one is replaceable
        shutil.move(self.p(".gemini", "GEMINI.md"), self.elsewhere("real.md"))
        os.symlink(self.elsewhere("real.md"), self.elsewhere("hop.md"))
        os.symlink(self.elsewhere("hop.md"), self.p(".gemini", "GEMINI.md"))
        self.assertRefusal(self.policy(), "is a symlink", "two hops")
        # dangling firmlink-spelled target
        os.remove(self.p(".gemini", "GEMINI.md"))
        os.symlink("/System/Volumes/Data" + self.elsewhere("gone.md"),
                   self.p(".gemini", "GEMINI.md"))
        self.assertRefusal(self.policy(), "is a symlink", "dangling")

    STEERING_DIRS = (
        ("bin", (".gemini", "antigravity-cli", "bin")),
        ("builtin", (".gemini", "antigravity-cli", "builtin")),
        ("hooks", (".gemini", "config", "hooks")),
        ("plugins", (".gemini", "config", "plugins")),
        ("skills", (".gemini", "config", "skills")),
        ("sidecars", (".gemini", "config", "sidecars")),
    )

    def test_a_hard_linked_file_inside_every_steering_dir_refuses(self):
        for label, parts in self.STEERING_DIRS:
            with self.subTest(dir=label):
                self.setUp()
                inner = self.p(*parts, "tool")
                _touch(inner)
                os.link(inner, self.elsewhere("alias-" + label))
                self.assertRefusal(self.policy(), "hard links", label)

    def test_a_symlink_inside_every_steering_dir_refuses(self):
        for label, parts in self.STEERING_DIRS:
            with self.subTest(dir=label):
                self.setUp()
                os.makedirs(self.p(*parts), exist_ok=True)
                os.symlink(self.elsewhere("t-" + label), self.p(*parts, "tool"))
                self.assertRefusal(self.policy(), "is a symlink", label)

    def test_a_nested_entry_inside_a_steering_dir_is_walked(self):
        self.setUp()
        nested = self.p(".gemini", "antigravity-cli", "builtin", "a", "b", "tool")
        _touch(nested)
        os.link(nested, self.elsewhere("alias-nested"))
        self.assertRefusal(self.policy(), "hard links", "nested")

    def test_a_hard_linked_history_file_refuses(self):
        os.link(self.p(".gemini", "antigravity-cli", "history.jsonl"),
                self.elsewhere("alias-history"))
        self.assertRefusal(self.policy(), "hard links", "history.jsonl")

    def test_a_symlinked_history_file_refuses(self):
        self.relink((".gemini", "antigravity-cli", "history.jsonl"), False)
        self.assertRefusal(self.policy(), "is a symlink", "history.jsonl")

    def test_scope_decision_vendor_links_in_state_dirs_are_accepted(self):
        # The vendor itself makes ``cli.log -> log/...`` (relative); walking the
        # state dirs would refuse every real spawn. A hard-linked file under
        # brain/ (a pre-planted alias by an unfenced actor) is a documented
        # residual, like the other lanes' state dirs.
        os.symlink(os.path.join("log", "x"), self.p(".gemini", "antigravity-cli", "cli.log"))
        brain = self.p(".gemini", "antigravity-cli", "brain", "c1", "transcript.jsonl")
        _touch(brain)
        os.link(brain, self.elsewhere("alias-brain"))
        _touch(self.p(".gemini", "antigravity-cli", "conversations", "c.db"))
        os.link(self.p(".gemini", "antigravity-cli", "conversations", "c.db"),
                self.elsewhere("alias-conv"))
        self.assertIsNone(ks.agy_layout_refusal(self.home))
        self.assertNotIn("ok", self.policy())

    def test_walk_cap_refuses_over_and_accepts_exactly_at(self):
        with mock.patch.object(ks, "_AGY_WALK_CAP", 12):
            plugins = self.p(".gemini", "config", "plugins")
            for i in range(12):
                _touch(os.path.join(plugins, "f%d" % i))
            self.assertIsNone(ks.agy_layout_refusal(self.home))
            _touch(os.path.join(plugins, "f12"))
            self.assertRefusal(self.policy(), "more than 12 entries", "wide dir")

    def test_a_steering_dir_whose_stat_fails_is_refused_not_skipped(self):
        # ``os.path.isdir`` swallows an error into False; the walker must use the
        # lstat it already took. A hard-linked file inside hooks/ proves it
        # would otherwise be accepted unseen.
        hooks = self.p(".gemini", "config", "hooks")
        inner = os.path.join(hooks, "tool")
        _touch(inner)
        os.link(inner, self.elsewhere("alias-hooks-tool"))
        real_stat = os.stat

        def flaky(path, *a, **kw):
            if os.fspath(path) == hooks:
                raise PermissionError(13, "Permission denied", hooks)
            return real_stat(path, *a, **kw)

        with mock.patch.object(ks.os, "stat", flaky):
            detail = ks.agy_layout_refusal(self.home)
        self.assertIsNotNone(detail)
        self.assertIn("hard links", detail)

    def test_a_steering_dir_whose_lstat_fails_is_refused(self):
        hooks = self.p(".gemini", "config", "hooks")
        real = os.lstat

        def flaky(path, *a, **kw):
            if os.fspath(path) == hooks:
                raise PermissionError(13, "Permission denied", hooks)
            return real(path, *a, **kw)

        with mock.patch.object(ks.os, "lstat", flaky):
            self.assertRefusal(self.policy(), "cannot be inspected", "hooks lstat")

    def test_inspection_failure_refuses(self):
        real = os.lstat
        hooks = self.p(".gemini", "config", "hooks.json")

        def flaky(path, *a, **kw):
            if os.fspath(path) == hooks:
                raise PermissionError(13, "Permission denied", hooks)
            return real(path, *a, **kw)

        with mock.patch.object(ks.os, "lstat", flaky):
            policy = self.policy()
        self.assertRefusal(policy, "cannot be inspected", "lstat failure")

    def test_refusal_names_the_path_and_the_rule(self):
        self.relink((".gemini", "config", "hooks.json"), False)
        detail = self.policy()["detail"]
        self.assertIn("~/.gemini/config/hooks.json is a symlink", detail)
        self.assertIn("stow-managed ~/.gemini is not supported", detail)

    def test_missing_dot_tree_is_not_a_layout_refusal(self):
        shutil.rmtree(self.p(".gemini"))
        self.assertIsNone(ks.agy_layout_refusal(self.home))


class AgyVendorWrites(AgyBase):
    def test_only_diagnostics_are_writable(self):
        policy = self.policy()
        kw = ks.wrap_kwargs_from_policy(policy)
        self.assertEqual(
            {entry for entry in kw["extra_post_write"]},
            {("subpath", os.path.join(self.agy_dir, "log")),
             ("subpath", os.path.join(self.agy_dir, "crashes"))})
        self.assertEqual(tuple(kw["extra_write_paths"]), ())
        self.assertTrue(policy["vendor_state_writable"])
        self.assertEqual(policy["vendor_root"], self.agy_dir)

    def test_the_whole_dot_tree_takes_no_new_hard_links_in_both_spellings(self):
        kw = self.wrap_kw()
        gemini = self.p(".gemini")
        self.assertIn(("subpath", gemini), kw["extra_deny_link"])
        firm = _firm(gemini)
        if firm is not None:
            self.assertIn(("subpath", firm), kw["extra_deny_link"])

    def test_missing_state_root_leaves_nothing_writable(self):
        shutil.rmtree(self.agy_dir)
        policy = self.policy()
        kw = ks.wrap_kwargs_from_policy(policy)
        self.assertEqual(tuple(kw["extra_post_write"]), ())
        self.assertIsNone(policy["vendor_root"])


class AgyChildEnvironment(unittest.TestCase):
    def test_agy_names_go_to_agy_only(self):
        base = {"AGY_CONFIG_DIR": "/c", "AGY_ADC_AUTH": "1",
                "OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a",
                "SYNTH_SECRET_TOKEN": "s"}
        self.assertEqual(set(h.allowlisted_child_env(base, "agy")),
                         {"AGY_CONFIG_DIR", "AGY_ADC_AUTH"})
        for provider in ("claude", "codex", "grok", "cursor", "pi"):
            out = h.allowlisted_child_env(base, provider)
            self.assertNotIn("AGY_CONFIG_DIR", out, provider)
            self.assertNotIn("AGY_ADC_AUTH", out, provider)

    @unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
    def test_fenced_agy_child_gets_the_allowlist_not_the_server_environment(self):
        home = os.path.realpath(
            tempfile.mkdtemp(prefix="omni_agy_envhome_", dir="/tmp"))
        wd = os.path.realpath(tempfile.mkdtemp(prefix="omni_agy_envwd_"))
        self.addCleanup(shutil.rmtree, home, True)
        self.addCleanup(shutil.rmtree, wd, True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "SYNTH_SECRET_TOKEN": "synthetic-not-a-secret",
               "AWS_SECRET_ACCESS_KEY": "synthetic-not-a-secret",
               "OPENAI_API_KEY": "synthetic-not-a-secret",
               "AGY_CONFIG_DIR": "/synthetic/cfg", "LANG": "en_US.UTF-8"}
        with mock.patch.object(ks, "protected_home", return_value=home), \
                mock.patch.dict(os.environ):
            os.environ.pop("SSH_AUTH_SOCK", None)
            _argv, child, meta = h.prepare_host_fence(
                ["/usr/bin/true"], provider="agy", workdir=wd,
                read_only=True, env=env)
        for name in ("SYNTH_SECRET_TOKEN", "AWS_SECRET_ACCESS_KEY",
                     "OPENAI_API_KEY"):
            self.assertNotIn(name, child)
        self.assertEqual(child["AGY_CONFIG_DIR"], "/synthetic/cfg")
        self.assertEqual(child["LANG"], "en_US.UTF-8")
        self.assertEqual(child["HOME"], home)
        self.assertEqual(meta["fence"], "seatbelt-fs")
        self.assertTrue(meta["fence_detail"]["read_wall"])


@unittest.skipUnless(DARWIN_FENCE, "needs the macOS kernel fence")
class AgyHostFence(AgyBase):
    def prepare(self, **kw):
        return h.prepare_host_fence(
            [sys.executable, "-c", "pass"], provider="agy", workdir=self.wd,
            read_only=True, env={"PATH": os.environ.get("PATH", "/usr/bin")}, **kw)

    def probe(self, argv, env, specs):
        out = subprocess.run(
            argv[:-3] + [sys.executable, "-c", PROBE] + specs, env=env,
            cwd=self.wd, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_a_hard_link_out_of_any_part_of_the_dot_tree_is_refused(self):
        where = os.path.realpath(tempfile.mkdtemp(prefix="omni_agy_ln_"))   # writable
        self.addCleanup(shutil.rmtree, where, True)
        src = self.p(".gemini", "config", "projects", "default-cli-project.json")
        state = self.p(".gemini", "antigravity-cli", "installation_id")
        argv, env, _meta = self.prepare()
        got = self.probe(argv, env, [
            "l:%s>%s" % (src, os.path.join(where, "a")),
            "l:%s>%s" % (state, os.path.join(where, "b"))])
        self.assertEqual(list(got.values()), [1, 1], got)   # EPERM
        # control: the same probe unfenced links both.
        plain = json.loads(subprocess.run(
            [sys.executable, "-c", PROBE, "l:%s>%s" % (src, os.path.join(where, "c")),
             "l:%s>%s" % (state, os.path.join(where, "d"))],
            capture_output=True, text=True, timeout=60).stdout)
        self.assertEqual(list(plain.values()), [0, 0], plain)

    def test_wrap_is_a_verified_read_walled_seatbelt_profile(self):
        argv, _env, meta = self.prepare()
        self.assertTrue(ks.is_wrapped(argv))
        self.assertEqual(meta["fence"], "seatbelt-fs")
        self.assertTrue(meta["fence_detail"]["read_wall"])
        self.assertTrue(meta["fence_detail"]["vendor_state_writable"])
        self.assertFalse(meta["fence_detail"]["workdir_writable"])

    def test_a_read_allow_over_the_instruction_directory_cannot_reopen_it(self):
        policy = self.policy()
        policy["wrap_kw"]["extra_post_read"] = tuple(
            policy["wrap_kw"]["extra_post_read"]) + (
            ("subpath", self.p(".gemini", "config")),)
        argv, env, _meta = self.prepare(policy=policy)
        got = self.probe(argv, env, [
            "r:" + self.p(".gemini", "config", "GEMINI.md"),
            "r:" + self.p(".gemini", "config", "hooks.json"),
            "r:" + self.p(".gemini", "config", "projects",
                          "default-cli-project.json")])
        self.assertEqual(sorted(got.values()), [0, 1, 1], got)
        self.assertEqual(
            got["r:" + self.p(".gemini", "config", "projects",
                              "default-cli-project.json")], 0)

    def test_missing_final_deny_row_fails_the_postcondition(self):
        policy = self.policy()
        kw = policy["wrap_kw"]
        kw["extra_deny_read_final"] = tuple(
            row for row in kw["extra_deny_read_final"]
            if row != ("literal", self.p(".gemini", "GEMINI.md")))
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(policy=policy)
        self.assertIn("agy global-instruction", ctx.exception.envelope["detail"])

    def test_missing_final_write_deny_row_fails_the_postcondition(self):
        policy = self.policy()
        kw = policy["wrap_kw"]
        kw["extra_deny_write_final"] = tuple(
            row for row in kw["extra_deny_write_final"]
            if row != ("literal", self.p(".gemini", "AGENTS.md")))
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(policy=policy)
        self.assertIn("agy global-instruction", ctx.exception.envelope["detail"])

    def test_a_symlink_planted_after_the_policy_is_built_is_refused(self):
        policy = self.policy()
        self.assertNotIn("ok", policy, policy)
        hooks = self.p(".gemini", "config", "hooks.json")
        other = os.path.join(self.home, "elsewhere.json")
        shutil.move(hooks, other)
        os.symlink(other, hooks)
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(policy=policy)
        self.assertEqual(ctx.exception.envelope["reason"], "fence_path_rejected")
        self.assertIn("is a symlink", ctx.exception.envelope["detail"])

    def test_a_hard_link_made_after_the_policy_is_built_is_refused(self):
        policy = self.policy()
        os.link(self.p(".gemini", "AGENTS.md"), os.path.join(self.home, "alias.md"))
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(policy=policy)
        self.assertEqual(ctx.exception.envelope["reason"], "fence_path_rejected")
        self.assertIn("hard links", ctx.exception.envelope["detail"])

    def test_an_alias_planted_in_a_steering_dir_after_the_policy_is_refused(self):
        tool = self.p(".gemini", "antigravity-cli", "bin", "agentapi")
        _touch(tool)
        policy = self.policy()
        self.assertNotIn("ok", policy, policy)
        os.link(tool, os.path.join(self.home, "alias-tool"))
        with self.assertRaises(h.HostFenceError) as ctx:
            self.prepare(policy=policy)
        self.assertEqual(ctx.exception.envelope["reason"], "fence_path_rejected")
        self.assertIn("hard links", ctx.exception.envelope["detail"])

    def test_a_steering_dir_that_cannot_be_inspected_fails_the_postcondition(self):
        kw = self.wrap_kw()
        hooks = self.p(".gemini", "config", "hooks")
        real = os.lstat

        def flaky(path, *a, **k):
            if os.fspath(path) == hooks:
                raise PermissionError(13, "Permission denied", hooks)
            return real(path, *a, **k)

        with mock.patch.object(ks.os, "lstat", flaky), \
                self.assertRaises(h.HostFenceError) as ctx:
            h._agy_instruction_wall_held(kw)
        self.assertIn("cannot be inspected", ctx.exception.envelope["detail"])

    def test_a_steering_dir_whose_stat_fails_fails_the_postcondition(self):
        kw = self.wrap_kw()
        hooks = self.p(".gemini", "config", "hooks")
        inner = os.path.join(hooks, "tool")
        _touch(inner)
        os.link(inner, os.path.join(self.home, "alias-hooks-tool"))
        real = os.stat

        def flaky(path, *a, **k):
            if os.fspath(path) == hooks:
                raise PermissionError(13, "Permission denied", hooks)
            return real(path, *a, **k)

        with mock.patch.object(ks.os, "stat", flaky), \
                self.assertRaises(h.HostFenceError) as ctx:
            h._agy_instruction_wall_held(kw)
        self.assertIn("hard links", ctx.exception.envelope["detail"])

    def test_a_data_volume_spelling_after_the_policy_is_built_is_refused(self):
        firm = _firm(self.home)
        if firm is None:
            self.skipTest("no firmlink spelling on this host")
        kw = self.wrap_kw()
        with mock.patch.object(ks, "protected_home", return_value=firm), \
                self.assertRaises(h.HostFenceError) as ctx:
            h._agy_instruction_wall_held(kw)
        self.assertIn("spelled through /System/Volumes/Data", ctx.exception.envelope["detail"])

    def test_read_wall_off_fails_the_postcondition(self):
        self.assertFalse(h._agy_instruction_wall_held({"read_wall": False}))

    def test_kernel_denies_the_family_and_allows_only_login_state(self):
        probe_argv = [sys.executable, "-c", PROBE]
        specs = (
            ["r:" + self.p(*parts) for parts in FAMILY]
            + ["r:" + self.p(*parts) for parts in SENSITIVE]
            + ["r:" + self.p(".gemini", "antigravity-cli", "installation_id"),
               "r:" + self.p(".gemini", "antigravity-cli", "cache",
                             "default_project_id.txt"),
               "r:" + self.p(".gemini", "config", "projects",
                             "default-cli-project.json"),
               "r:" + self.p("unrelated-home-file")]
            + ["w:" + self.p(".gemini", "antigravity-cli", "log", "run.log"),
               "w:" + self.p(".gemini", "antigravity-cli", "crashes", "c.log"),
               "w:" + self.p(".gemini", "antigravity-cli", "conversations", "x.db"),
               "w:" + self.p(".gemini", "antigravity-cli", "history.jsonl"),
               "w:" + self.p(".gemini", "config", "hooks.json"),
               "w:" + self.p(".gemini", "config", "plant.json"),
               "w:" + self.p(".gemini", "GEMINI.md"),
               "w:" + self.p(".gemini", "antigravity-cli", "bin", "agentapi"),
               "w:" + os.path.join(self.wd, "in-workdir.txt")])
        _touch(self.p("unrelated-home-file"))
        # Positive control: the same probe unfenced reads everything it can.
        plain = json.loads(subprocess.run(
            probe_argv + specs, capture_output=True, text=True, timeout=60,
            cwd=self.wd).stdout)
        for parts in FAMILY:
            self.assertEqual(plain["r:" + self.p(*parts)], 0, parts)
        argv, env, _meta = self.prepare()
        fenced = json.loads(subprocess.run(
            argv[:-3] + probe_argv + specs, env=env, cwd=self.wd,
            capture_output=True, text=True, timeout=60).stdout)
        for parts in FAMILY + SENSITIVE:
            spec = "r:" + self.p(*parts)
            if os.path.isdir(self.p(*parts)):
                continue
            self.assertEqual(fenced[spec], 1, spec)        # EPERM
        for rel in (("installation_id",), ("cache", "default_project_id.txt")):
            self.assertEqual(
                fenced["r:" + os.path.join(self.agy_dir, *rel)], 0, rel)
        self.assertEqual(fenced["r:" + self.p(
            ".gemini", "config", "projects", "default-cli-project.json")], 0)
        self.assertEqual(fenced["r:" + self.p("unrelated-home-file")], 1)
        for rel in (("log", "run.log"), ("crashes", "c.log")):
            self.assertEqual(
                fenced["w:" + os.path.join(self.agy_dir, *rel)], 0, rel)
        for spec in specs:
            if spec.startswith("w:") and "/log/" not in spec \
                    and "/crashes/" not in spec and "in-workdir" not in spec:
                self.assertEqual(fenced[spec], 1, spec)
        # read-only workdir: the lane starts agy in an empty directory.
        self.assertEqual(fenced["w:" + os.path.join(self.wd, "in-workdir.txt")], 1)


class AgyLane(unittest.TestCase):
    """providers/agy.py hands the spawn to the host fence."""

    def run_lane(self, profile="default", prepare=None, env_extra=None):
        seen = {}

        def fake_prepare(argv, **kw):
            seen["prepare"] = (list(argv), kw)
            seen["dir_existed"] = os.path.isdir(kw["workdir"])
            seen["dir_entries"] = os.listdir(kw["workdir"])
            return ["WRAPPED"] + list(argv), {"CHILD": "1"}, {}

        async def fake_run_cli(argv, **kwargs):
            seen["run"] = (list(argv), kwargs)
            return {"returncode": 0, "stdout": json.dumps(
                {"status": "SUCCESS", "response": "ok", "usage": {}}),
                "stderr": "", "timed_out": False, "watched": False}

        env = {"AGY_CONFIG_DIR_WORK": "/synthetic/acct"}
        env.update(env_extra or {})
        with mock.patch.object(agy_mod.vendor_bin, "which",
                               return_value="/usr/bin/agy"), \
                mock.patch.object(agy_mod, "run_cli", fake_run_cli), \
                mock.patch.object(agy_mod, "prepare_host_fence",
                                  prepare or fake_prepare), \
                mock.patch.dict(os.environ, env):
            os.environ.pop("OMNI_LLM_DEPTH", None)
            res = asyncio.run(agy_mod.run_agy_subprocess(
                "hi", "m", 60, profile))
        return res, seen

    def test_spawn_goes_through_prepare_host_fence_for_agy(self):
        res, seen = self.run_lane()
        self.assertTrue(res["ok"], res)
        argv, kw = seen["prepare"]
        self.assertEqual(kw["provider"], "agy")
        self.assertTrue(kw["read_only"])
        self.assertEqual(os.path.basename(argv[0]), "agy")
        # run_cli receives the WRAPPED argv and the fence's environment.
        run_argv, run_kw = seen["run"]
        self.assertEqual(run_argv[0], "WRAPPED")
        self.assertEqual(run_kw["env"], {"CHILD": "1"})

    def test_agent_starts_in_an_empty_throwaway_directory_removed_after(self):
        res, seen = self.run_lane()
        self.assertTrue(res["ok"], res)
        workdir = seen["prepare"][1]["workdir"]
        self.assertEqual(seen["run"][1]["cwd"], workdir)
        self.assertTrue(seen["dir_existed"])
        self.assertEqual(seen["dir_entries"], [])
        self.assertFalse(os.path.exists(workdir))

    def test_profile_variable_reaches_the_fence_input_env(self):
        _res, seen = self.run_lane(profile="work")
        self.assertEqual(
            seen["prepare"][1]["env"]["AGY_CONFIG_DIR"], "/synthetic/acct")

    def test_fence_refusal_is_returned_and_nothing_spawns(self):
        envelope = {"ok": False, "reason": "fence_path_rejected",
                    "detail": "synthetic", "fence": "none"}

        def refuse(argv, **kw):
            raise h.HostFenceError(envelope)

        res, seen = self.run_lane(prepare=refuse)
        self.assertEqual(res["reason"], "fence_path_rejected")
        self.assertNotIn("run", seen)

    def test_refusal_leaves_no_directory_behind(self):
        made = []
        real = tempfile.mkdtemp

        def tracking(*a, **kw):
            path = real(*a, **kw)
            made.append(path)
            return path

        def refuse(argv, **kw):
            raise h.HostFenceError({"ok": False, "reason": "fence_unavailable",
                                    "detail": "synthetic"})

        with mock.patch.object(agy_mod.tempfile, "mkdtemp", tracking):
            self.run_lane(prepare=refuse)
        self.assertEqual(len(made), 1)
        self.assertFalse(os.path.exists(made[0]))


class AgyBinary(unittest.TestCase):
    """The lane execs the file the fence admits (``vendor_bin``), never a shim."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = os.path.realpath(self.tmp.name)
        self.shims = os.path.join(root, "shims")
        self.real = os.path.join(root, "real")
        os.mkdir(self.shims)
        os.mkdir(self.real)
        for where, body in ((self.shims, "#!/bin/sh\n# mofuu-harness-shim\nexit 1\n"),
                            (self.real, "#!/bin/sh\nexit 0\n")):
            path = os.path.join(where, "agy")
            with open(path, "w") as handle:
                handle.write(body)
            os.chmod(path, 0o755)
        self.path = os.pathsep.join([self.shims, self.real])

    def test_find_agy_skips_a_shim_ahead_on_path(self):
        with mock.patch.dict(os.environ, {"PATH": self.path}):
            self.assertEqual(agy_mod.find_agy(), os.path.join(self.real, "agy"))

    def test_the_lane_execs_the_real_binary_not_the_shim(self):
        seen = {}

        def fake_prepare(argv, **kw):
            seen["argv"] = list(argv)
            return list(argv), kw["env"], {}

        async def fake_run_cli(argv, **kwargs):
            return {"returncode": 0, "stdout": json.dumps(
                {"status": "SUCCESS", "response": "ok", "usage": {}}),
                "stderr": "", "timed_out": False, "watched": False}

        with mock.patch.dict(os.environ, {"PATH": self.path}), \
                mock.patch.object(agy_mod, "run_cli", fake_run_cli), \
                mock.patch.object(agy_mod, "prepare_host_fence", fake_prepare):
            os.environ.pop("OMNI_LLM_DEPTH", None)
            res = asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen["argv"][0], os.path.join(self.real, "agy"))

    def test_a_shim_only_path_is_cli_missing(self):
        with mock.patch.dict(os.environ, {"PATH": self.shims}):
            res = asyncio.run(agy_mod.run_agy_subprocess("hi", "m", 60, "default"))
        self.assertEqual(res["reason"], "cli_missing")


class AgyListModels(unittest.TestCase):
    """``list_models`` spawns ``agy models`` through the same fenced helper."""

    def run_models(self, prepare=None):
        from providers import dispatch, model_catalog
        seen = {}
        catalog_summary = {"ok": True, "catalogs": [{"provider": "cursor"}]}

        def fake_prepare(argv, **kw):
            seen["prepare"] = (list(argv), kw)
            seen["cwd"] = kw["workdir"]
            seen["entries"] = os.listdir(kw["workdir"])
            return ["WRAPPED"] + list(argv), {"CHILD": "1"}, {}

        class Proc:
            returncode = 0

            async def communicate(self):
                return b"gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n", b""

        async def fake_exec(*argv, **kwargs):
            seen["exec"] = (list(argv), kwargs)
            return Proc()

        with mock.patch.object(agy_mod.vendor_bin, "which",
                               return_value="/usr/bin/agy"), \
                mock.patch.object(agy_mod, "prepare_host_fence",
                                  prepare or fake_prepare), \
                mock.patch.object(dispatch.asyncio, "create_subprocess_exec",
                                  fake_exec), \
                mock.patch.object(model_catalog, "status",
                                  return_value=catalog_summary):
            res = asyncio.run(dispatch.list_models(5))
        return res, seen

    def test_models_spawn_goes_through_the_agy_fence(self):
        res, seen = self.run_models()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["routable_models"], ["agy/gemini-3.8-flash-high"])
        self.assertEqual(res["catalogs"],
                         {"ok": True, "catalogs": [{"provider": "cursor"}]})
        argv, kw = seen["prepare"]
        self.assertEqual(kw["provider"], "agy")
        self.assertTrue(kw["read_only"])
        self.assertEqual(argv[1:], ["models"])
        exec_argv, exec_kw = seen["exec"]
        self.assertEqual(exec_argv[0], "WRAPPED")
        self.assertEqual(exec_kw["env"], {"CHILD": "1"})
        self.assertEqual(exec_kw["cwd"], seen["cwd"])
        self.assertEqual(seen["entries"], [])
        self.assertFalse(os.path.exists(seen["cwd"]))

    def test_fence_refusal_is_a_loud_envelope_and_nothing_spawns(self):
        def refuse(argv, **kw):
            raise h.HostFenceError({"ok": False, "reason": "fence_unavailable",
                                    "detail": "synthetic"})

        res, seen = self.run_models(prepare=refuse)
        self.assertIs(res["ok"], False)
        self.assertEqual(res["reason"], "fence_unavailable")
        self.assertIn("backends", res)
        self.assertNotIn("exec", seen)


class AgyRouting(unittest.TestCase):
    """On macOS ``run_agy`` never calls the SDK; off macOS the SDK stays preferred."""

    def run_agy(self, system, sub_result=None):
        calls = {"sdk": 0, "sub": []}

        async def sdk(*a, **k):
            calls["sdk"] += 1
            raise AssertionError("the SDK must not be called here")

        async def sub(prompt, model, timeout_s, account_profile):
            calls["sub"].append((prompt, model, timeout_s, account_profile))
            return sub_result or {"ok": True, "text": "via-sub", "model": model}

        with mock.patch.object(agy_mod, "HAS_SDK", True), \
                mock.patch.object(agy_mod.platform, "system", return_value=system), \
                mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}), \
                mock.patch.object(agy_mod, "run_agy_sdk", sdk), \
                mock.patch.object(agy_mod, "run_agy_subprocess", sub):
            res = asyncio.run(agy_mod.run_agy(
                "hello", "My Model", 42, ["/a/b.txt", "/c/d.txt"], "work"))
        return res, calls

    def test_darwin_never_calls_the_sdk_and_passes_everything_to_the_subprocess(self):
        res, calls = self.run_agy("Darwin")
        self.assertEqual(res["text"], "via-sub")
        self.assertEqual(calls["sdk"], 0)
        self.assertEqual(len(calls["sub"]), 1)
        prompt, model, timeout_s, profile = calls["sub"][0]
        self.assertEqual((model, timeout_s, profile), ("My Model", 42, "work"))
        self.assertEqual(prompt, "hello\n\nFiles to analyze:\n/a/b.txt\n/c/d.txt")
        self.assertEqual(prompt.count("Files to analyze:"), 1)

    def test_a_fence_refusal_is_returned_and_not_retried_on_the_sdk(self):
        refusal = {"ok": False, "reason": "fence_unavailable",
                   "detail": "kernel fence unavailable"}
        res, calls = self.run_agy("Darwin", sub_result=refusal)
        self.assertEqual(res["reason"], "fence_unavailable")
        self.assertEqual(calls["sdk"], 0)
        self.assertEqual(len(calls["sub"]), 1)

    def test_effort_and_files_reach_subprocess_once_without_sdk(self):
        calls = {"sdk": 0, "sub": []}

        async def sdk(*_args, **_kwargs):
            calls["sdk"] += 1
            raise AssertionError("explicit effort cannot use the SDK")

        async def sub(prompt, model, timeout_s, account_profile, effort=None):
            calls["sub"].append((prompt, model, timeout_s, account_profile, effort))
            return {"ok": True, "text": "via-sub", "model": model}

        with mock.patch.object(agy_mod, "HAS_SDK", True), \
                mock.patch.object(agy_mod.platform, "system", return_value="Linux"), \
                mock.patch.object(agy_mod, "run_agy_sdk", sdk), \
                mock.patch.object(agy_mod, "run_agy_subprocess", sub), \
                mock.patch("providers.model_catalog.agy_effort_tokens",
                           return_value=frozenset(("high",))):
            result = asyncio.run(agy_mod.run_agy(
                "inspect", "m", 30, ["/a.txt", "/b.txt"], "default", effort="high"))
        self.assertTrue(result["ok"])
        self.assertEqual(calls["sdk"], 0)
        self.assertEqual(calls["sub"], [(
            "inspect\n\nFiles to analyze:\n/a.txt\n/b.txt", "m", 30, "default", "high")])

    def test_missing_named_profile_refuses_without_default_fallback(self):
        with mock.patch.dict(os.environ, {"AGY_CONFIG_DIR_WORK": ""}), \
                mock.patch.object(agy_mod, "HAS_SDK", True), \
                mock.patch.object(agy_mod, "run_agy_sdk") as sdk, \
                mock.patch.object(agy_mod, "run_agy_subprocess") as subprocess_call:
            result = asyncio.run(agy_mod.run_agy(
                "hello", "m", 30, account_profile="work"))
        self.assertEqual(result["reason"], "unsupported_profile")
        self.assertIn("AGY_CONFIG_DIR_WORK", result["detail"])
        sdk.assert_not_called()
        subprocess_call.assert_not_called()

    def test_a_missing_sandbox_exec_is_a_loud_refusal_on_darwin(self):
        wd = tempfile.mkdtemp(prefix="omni_agy_nofence_")
        self.addCleanup(shutil.rmtree, wd, True)
        with mock.patch.object(h.platform, "system", return_value="Darwin"), \
                mock.patch.object(ks, "available", return_value=False), \
                self.assertRaises(h.HostFenceError) as ctx:
            h.prepare_host_fence(["/usr/bin/true"], provider="agy", workdir=wd,
                                 read_only=True, env={})
        self.assertEqual(ctx.exception.envelope["reason"], "fence_unavailable")

    def test_off_darwin_the_sdk_is_still_preferred(self):
        async def sdk(prompt, model, timeout_s, files=None, account_profile="default"):
            return {"ok": True, "text": "via-sdk", "model": model}

        async def sub(*a, **k):
            raise AssertionError("subprocess must not run when the SDK answers")

        with mock.patch.object(agy_mod, "HAS_SDK", True), \
                mock.patch.object(agy_mod.platform, "system", return_value="Linux"), \
                mock.patch.object(agy_mod, "run_agy_sdk", sdk), \
                mock.patch.object(agy_mod, "run_agy_subprocess", sub):
            res = asyncio.run(agy_mod.run_agy("hi", "m", 30, None, "default"))
        self.assertEqual(res["text"], "via-sdk")


class AgyHostExec(unittest.TestCase):
    def test_host_exec_accepts_agy_as_a_provider(self):
        wd = tempfile.mkdtemp(prefix="omni_agy_hx_")
        self.addCleanup(shutil.rmtree, wd, True)
        seen = {}

        def fake_prepare(command, **kw):
            seen.update(kw)
            return command, {}, {}

        with mock.patch.object(h, "prepare_host_fence", fake_prepare), \
                mock.patch.object(h.os, "execvpe") as execvpe, \
                mock.patch.object(h.os, "chdir"):
            h.main(["--provider", "agy", "--workdir", wd, "--read-only",
                    "--", "/usr/bin/true"])
        self.assertEqual(seen["provider"], "agy")
        self.assertTrue(execvpe.called)

    def test_agy_is_not_a_repo_write_provider(self):
        with mock.patch.object(h.platform, "system", return_value="Darwin"), \
                self.assertRaises(h.HostFenceError):
            h.prepare_host_fence(["/usr/bin/true"], provider="agy",
                                 workdir=tempfile.gettempdir(),
                                 allow_repo_write=True, env={})


if __name__ == "__main__":
    unittest.main()
