#!/usr/bin/env python3
"""One "same location" judgement for host-fence refusals (issue #147).

Case, NFC/NFD, a firmlink spelling, a symlinked root and a path that does not
exist yet are one question: is this the same place as that one. The refusal
direction asks it with ``location_inside`` / ``same_location`` (folds the
names below the nearest existing ancestor, an unreadable identity counts as
overlap); the exemption direction asks it with ``exempt_inside`` /
``exempt_same`` / ``_same_dir_exact`` (exact names, an unreadable identity is
"no").

Each test names the issue bullet it reproduces and goes through the real
policy builder or check function. Hermetic: a synthetic home in a temp dir,
local git only, no vendor CLI, no network. The Seatbelt fact checks run the
real ``/usr/bin/sandbox-exec`` on temp files and skip off Darwin.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import unicodedata
import unittest
from unittest import mock

from providers import agent_lane, host_fence, kernel_sandbox, profiles

FIRMLINK = "/System/Volumes/Data"
NFC = unicodedata.normalize("NFC", "café")
NFD = unicodedata.normalize("NFD", "café")


def _firm(path):
    """The data-volume firmlink spelling of *path*, or None off Darwin."""
    if platform.system() != "Darwin" or not os.path.isdir(FIRMLINK):
        return None
    spelled = FIRMLINK + path
    try:
        if os.stat(os.path.dirname(spelled)).st_ino != os.stat(
                os.path.dirname(path)).st_ino:
            return None
    except OSError:
        return None
    return spelled


def _isolated_write_roots(workdir, read_only, extra_write_paths=None):
    """Write allow-set without the host's temp roots.

    The synthetic home lives in a temp dir, which the real allow-set always
    covers; keep only the workdir and the caller's extras so "where the child
    can write" is exactly what each test builds.
    """
    out = []
    if workdir is not None and not read_only:
        out.append(os.path.realpath(str(workdir)))
    out.extend(os.fspath(p) for p in (extra_write_paths or ()))
    return tuple(out)


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



class _Base(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="omni_sloc_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.home = os.path.join(self.root, "home")
        self.wd = os.path.join(self.root, "wd")
        self.ext = os.path.join(self.root, "ext")
        for path in (self.home, self.wd, self.ext):
            os.makedirs(path)

    def firm(self, path):
        spelled = _firm(path)
        if spelled is None:
            self.skipTest("no firmlink spelling on this host")
        return spelled

    def policy(self, provider, **kw):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            return kernel_sandbox.agent_fence_policy(
                provider, workdir=self.wd, read_only=False, **kw)

    def assert_refused(self, policy):
        self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
        self.assertNotIn("wrap_kw", policy)

    def assert_spawns(self, policy):
        self.assertNotIn("ok", policy, policy)
        self.assertIn("wrap_kw", policy)


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------
class LocationHelper(_Base):
    def test_firmlink_spelling_of_an_existing_path_is_the_same_location(self):
        firm = self.firm(self.home)
        self.assertTrue(kernel_sandbox.same_location(self.home, firm))
        self.assertTrue(kernel_sandbox.location_inside(
            os.path.join(firm, "x"), self.home))
        self.assertTrue(kernel_sandbox.exempt_same(self.home, firm))

    def test_firmlink_spelling_of_a_missing_path_is_the_same_location(self):
        missing = os.path.join(self.home, "a", "b")
        firm = self.firm(self.home) + "/a/b"
        self.assertTrue(kernel_sandbox.same_location(missing, firm))
        self.assertTrue(kernel_sandbox.location_inside(
            firm + "/deeper", os.path.join(self.home, "a")))
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(self.home, "a"), firm))
        # Exemption: the existing ancestor is compared by inode, the names
        # below it must match exactly.
        self.assertTrue(kernel_sandbox.exempt_same(missing, firm))

    def test_case_and_normalization_variants(self):
        os.makedirs(os.path.join(self.root, "Acct", NFC))
        self.assertTrue(kernel_sandbox.same_location(
            os.path.join(self.root, "acct"), os.path.join(self.root, "ACCT")))
        self.assertTrue(kernel_sandbox.location_inside(
            os.path.join(self.root, "acct", NFD, "x"),
            os.path.join(self.root, "ACCT", NFC)))
        # Names that do not exist yet fold the same way.
        self.assertTrue(kernel_sandbox.same_location(
            os.path.join(self.root, "NewDir", NFC),
            os.path.join(self.root, "newdir", NFD)))

    def test_symlinked_root_is_the_target(self):
        real = os.path.join(self.root, "real")
        os.makedirs(real)
        os.symlink(real, os.path.join(self.root, "link"))
        self.assertTrue(kernel_sandbox.same_location(
            os.path.join(self.root, "link"), real))
        self.assertTrue(kernel_sandbox.location_inside(
            os.path.join(self.root, "link", "new", "x"), real))
        # A link judged by its own location keeps its name.
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(self.root, "link"), real, resolve_leaf=False))

    def test_exemption_never_folds_names_below_the_existing_ancestor(self):
        a = os.path.join(self.root, "Account")
        b = os.path.join(self.root, "account")
        self.assertTrue(kernel_sandbox.same_location(a, b))
        self.assertFalse(kernel_sandbox.exempt_same(a, b))
        self.assertFalse(kernel_sandbox._same_dir_exact(a, b))
        n_c = os.path.join(self.root, NFC)
        n_d = os.path.join(self.root, NFD)
        self.assertTrue(kernel_sandbox.same_location(n_c, n_d))
        self.assertFalse(kernel_sandbox._same_dir_exact(n_c, n_d))
        self.assertTrue(kernel_sandbox._same_dir_exact(a, a))

    def test_unknown_identity_is_overlap_for_refusal_and_no_for_exemption(self):
        boom = OSError(13, "denied")
        with mock.patch.object(kernel_sandbox, "_stat_dev_ino", side_effect=boom):
            self.assertTrue(kernel_sandbox.location_inside("/a/b", "/c"))
            self.assertTrue(kernel_sandbox.same_location("/a/b", "/c"))
            self.assertFalse(kernel_sandbox.exempt_inside("/a/b", "/a"))
            self.assertFalse(kernel_sandbox._same_dir_exact("/a/b", "/a/c"))
            with self.assertRaises(kernel_sandbox._IdentityUnavailable):
                kernel_sandbox._ci_same_or_inside("/a/b", "/a")

    def test_missing_path_errors_are_not_unknown(self):
        # ENOENT / ENOTDIR / ENAMETOOLONG mean "does not exist yet".
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(self.root, "x" * 300, "y"), os.path.join(self.root, "z")))
        file_path = os.path.join(self.root, "file")
        open(file_path, "w").close()
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(file_path, "sub"), os.path.join(self.root, "other")))

    def test_distinct_siblings_with_different_folded_names_are_not_one_place(self):
        os.makedirs(os.path.join(self.root, "acct"))
        os.makedirs(os.path.join(self.root, "acct2"))
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(self.root, "acct2", "x"), os.path.join(self.root, "acct")))
        self.assertFalse(kernel_sandbox.same_location(
            os.path.join(self.root, "acct"), os.path.join(self.root, "acct2")))
        # Neither exists: still two places.
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join(self.root, "n1", "x"), os.path.join(self.root, "n")))
        self.assertFalse(kernel_sandbox.same_location(
            os.path.join(self.root, "n1"), os.path.join(self.root, "n2")))
        self.assertFalse(kernel_sandbox.location_inside(self.root, self.home))


# --------------------------------------------------------------------------
# Issue bullet 1: a missing path fell back to string comparison
# --------------------------------------------------------------------------
class MissingPathFirmlink(_Base):
    def test_api_settings_exposure_sees_a_firmlink_root_over_a_missing_base(self):
        base = os.path.join(self.home, "x", "api-settings")      # x is missing
        for root in (self.firm(self.home),
                     self.firm(self.home) + "/x"):
            with self.subTest(root=root):
                self.assertTrue(host_fence.settings_base_exposed(
                    base, [("subpath", root)]))
        self.assertTrue(host_fence.settings_base_exposed(
            base, [("literal", self.firm(self.home) + "/x")]))

    def test_api_settings_exposure_negative_controls(self):
        base = os.path.join(self.home, "x", "api-settings")
        for root in (os.path.join(self.root, "elsewhere"),
                     self.firm(self.home) + "/y",
                     os.path.join(self.home, "x", "api-settings", "run")):
            with self.subTest(root=root):
                self.assertIsNone(host_fence.settings_base_exposed(
                    base, [("subpath", root)]))

    def test_codex_filter_sees_a_firmlink_spelling_of_a_missing_subdir(self):
        os.makedirs(os.path.join(self.home, ".codex"))
        codex = os.path.join(self.home, ".codex")
        missing = self.firm(self.home) + "/.codex/newdir"
        self.assertTrue(kernel_sandbox._path_inside_any_case(missing, codex))
        kept, why = kernel_sandbox._drop_inside_codex_home(
            [("subpath", missing), ("subpath", os.path.join(self.ext, "ok"))],
            codex, "read", lambda entry: entry)
        self.assertEqual(kept, [("subpath", os.path.join(self.ext, "ok"))])
        self.assertIsNone(why)


# --------------------------------------------------------------------------
# Issue bullet 2: a symlinked codex home, the reverse containment
# --------------------------------------------------------------------------
class CodexHomeContainment(_Base):
    def _link_codex_into_ext(self):
        os.makedirs(os.path.join(self.ext, "codex"))
        os.symlink(os.path.join(self.ext, "codex"),
                   os.path.join(self.home, ".codex"))

    def test_read_allow_containing_a_symlinked_codex_home_is_refused(self):
        self._link_codex_into_ext()
        # The cursor config root sits outside home and so contains the real
        # codex home; its late read allow would re-open the denied tree.
        self.assert_refused(self.policy("cursor", config_dir=self.ext))

    def test_a_config_root_beside_the_symlinked_codex_home_still_spawns(self):
        self._link_codex_into_ext()
        other = os.path.join(self.root, "other-cfg")
        os.makedirs(other)
        self.assert_spawns(self.policy("cursor", config_dir=other))

    def test_filter_drops_inside_refuses_containing_keeps_unrelated(self):
        os.makedirs(os.path.join(self.home, ".codex", "sub"))
        codex = os.path.join(self.home, ".codex")
        drop = kernel_sandbox._drop_inside_codex_home
        kept, why = drop([("subpath", codex), ("subpath", codex + "/sub")],
                         codex, "write", lambda e: kernel_sandbox._write_entry_bound(*e))
        self.assertEqual((kept, why), ([], None))
        kept, why = drop([("subpath", self.home)], codex, "write",
                         lambda e: kernel_sandbox._write_entry_bound(*e))
        self.assertEqual(kept, [])
        self.assertIn("contains the codex home", why)
        keep = [("subpath", os.path.join(self.home, ".pi")),
                ("literal", os.path.join(self.home, ".codexx"))]
        self.assertEqual(drop(keep, codex, "read", lambda e: e), (keep, None))

    def test_filter_refuses_a_firmlink_spelling_of_a_containing_root(self):
        os.makedirs(os.path.join(self.ext, "codex"))
        codex = os.path.join(self.ext, "codex")
        kept, why = kernel_sandbox._drop_inside_codex_home(
            [("subpath", self.firm(self.ext))], codex, "read", lambda e: e)
        self.assertEqual(kept, [])
        self.assertIsNotNone(why)


# --------------------------------------------------------------------------
# Issue bullet 3: profiles.py default login directory
# --------------------------------------------------------------------------
class DefaultLoginDirectory(_Base):
    CASES = (("cursor", ".cursor", ".cursor-", "CURSOR_CONFIG_DIR"),
             ("claude", ".claude", ".claude-", "CLAUDE_CONFIG_DIR"),
             ("codex", ".codex", ".codex-", "CODEX_HOME"))

    def _check(self, spelled, leaf, prefix, var):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home):
            return profiles._config_dir_unsupported(var, spelled, prefix, leaf)

    def test_other_case_of_the_default_login_dir_names_it(self):
        for _p, leaf, prefix, var in self.CASES:
            with self.subTest(leaf=leaf):
                got = self._check(os.path.join(self.home, leaf.upper()),
                                  leaf, prefix, var)
                self.assertIsNotNone(got)
                self.assertIn("default login directory", got["detail"])

    def test_firmlink_spelling_of_the_default_login_dir_is_refused(self):
        for _p, leaf, prefix, var in self.CASES:
            for exists in (False, True):
                with self.subTest(leaf=leaf, exists=exists):
                    path = os.path.join(self.home, leaf)
                    if exists:
                        os.makedirs(path, exist_ok=True)
                    got = self._check(self.firm(self.home) + "/" + leaf,
                                      leaf, prefix, var)
                    self.assertIsNotNone(got)
                    self.assertIn("default login directory", got["detail"])

    def test_dedicated_dir_outside_home_is_accepted(self):
        for _p, leaf, prefix, var in self.CASES:
            with self.subTest(leaf=leaf):
                self.assertIsNone(self._check(
                    os.path.join(self.ext, "acct"), leaf, prefix, var))
                self.assertIsNone(self._check(
                    os.path.join(self.home, prefix + "work"), leaf, prefix, var))


# --------------------------------------------------------------------------
# Issue bullet 4: _same_dir_exact fell through to folding
# --------------------------------------------------------------------------
class ExemptionIsExact(_Base):
    def test_missing_folded_equal_names_are_two_places(self):
        a, b = (os.path.join(self.root, "Account"),
                os.path.join(self.root, "account"))
        self.assertFalse(kernel_sandbox._same_dir_exact(a, b))

    def test_firmlink_spelling_is_still_the_same_place(self):
        firm = self.firm(self.root)
        self.assertTrue(kernel_sandbox._same_dir_exact(self.root, firm))
        self.assertTrue(kernel_sandbox._same_dir_exact(
            os.path.join(self.root, "new"), firm + "/new"))
        self.assertFalse(kernel_sandbox._same_dir_exact(
            os.path.join(self.root, "new"), firm + "/NEW"))

    def test_launcher_root_membership_never_folds(self):
        # The launcher root and the config dir both do not exist yet and
        # differ by case: an exemption decided by folding would admit a
        # directory the root does not name on a case-sensitive volume.
        home = os.path.join(self.root, "h")
        os.makedirs(os.path.join(home, "x"))
        why = kernel_sandbox._caller_config_dir_rejection(
            os.path.join(home, "x", "NEW", "cfg"), label="CURSOR_CONFIG_DIR",
            home=home, allow_dotdir_prefix=".cursor-",
            launcher_roots=(os.path.join(home, "x", "new"),))
        self.assertIsNotNone(why)
        self.assertIn("inside the home tree", why)
        # The exact spelling is admitted.
        self.assertIsNone(kernel_sandbox._caller_config_dir_rejection(
            os.path.join(home, "x", "new", "cfg"), label="CURSOR_CONFIG_DIR",
            home=home, allow_dotdir_prefix=".cursor-",
            launcher_roots=(os.path.join(home, "x", "new"),)))

    def test_dotdir_exemption_needs_the_parent_to_be_home_exactly_or_by_inode(self):
        os.makedirs(os.path.join(self.home, ".cursor-work"))
        self.assertIsNone(kernel_sandbox.home_tree_rejection(
            os.path.join(self.home, ".cursor-work"), self.home, ".cursor-"))
        self.assertIsNone(kernel_sandbox.home_tree_rejection(
            self.firm(self.home) + "/.cursor-work", self.home, ".cursor-"))
        self.assertEqual(kernel_sandbox.home_tree_rejection(
            os.path.join(self.home, "deeper", ".cursor-work"), self.home,
            ".cursor-"), "inside_home")


# --------------------------------------------------------------------------
# Issue bullet 5: _reopens
# --------------------------------------------------------------------------
class ReopensRows(_Base):
    def setUp(self):
        super().setUp()
        self.grok = os.path.join(self.home, ".grok")
        os.makedirs(self.grok)
        self.row = os.path.join(self.grok, "AGENTS.md")

    def test_firmlink_literal_reopens_a_missing_deny_row(self):
        firm = self.firm(self.home) + "/.grok/AGENTS.md"
        self.assertFalse(os.path.lexists(self.row))
        self.assertTrue(host_fence._reopens(("literal", firm), [("literal", self.row)]))
        self.assertTrue(host_fence._reopens(("subpath", firm), [("literal", self.row)]))

    def test_firmlink_literal_reopens_an_existing_literal_row(self):
        open(self.row, "w").close()
        firm = self.firm(self.home) + "/.grok/AGENTS.md"
        self.assertTrue(host_fence._reopens(("literal", firm), [("literal", self.row)]))
        self.assertTrue(host_fence._reopens(("literal", self.row), [("literal", self.row)]))

    def test_case_and_normalization_variants_of_missing_rows(self):
        for variant in (os.path.join(self.grok, "agents.md"),
                        os.path.join(self.home, ".GROK", "AGENTS.md")):
            with self.subTest(variant=variant):
                self.assertTrue(host_fence._reopens(
                    ("literal", variant), [("literal", self.row)]))
        nfc = os.path.join(self.grok, NFC, "AGENTS.md")
        self.assertTrue(host_fence._reopens(
            ("subpath", os.path.join(self.grok, NFD)), [("literal", nfc)]))

    def test_subpath_row_is_reopened_from_below_by_a_firmlink_allow(self):
        rows = [("subpath", os.path.join(self.grok, "rules"))]
        self.assertTrue(host_fence._reopens(
            ("subpath", self.firm(self.home) + "/.grok/rules/deep"), rows))

    def test_unrelated_allows_do_not_reopen(self):
        rows = [("literal", self.row), ("subpath", os.path.join(self.grok, "rules"))]
        for entry in (("literal", os.path.join(self.grok, "AGENTS.mdx")),
                      ("subpath", os.path.join(self.grok, "rulesets")),
                      ("subpath", os.path.join(self.ext, "AGENTS.md")),
                      ("literal", self.firm(self.home) + "/.grok/other")):
            with self.subTest(entry=entry):
                self.assertFalse(host_fence._reopens(entry, rows))

    def test_wall_held_check_sees_the_firmlink_allow(self):
        with mock.patch.object(kernel_sandbox, "protected_home",
                               return_value=self.home):
            rows = host_fence._grok_instruction_rows(self.grok)
            wrap_kw = {"read_wall": True, "extra_deny_read": tuple(rows),
                       "extra_post_read": (("literal", self.firm(self.home)
                                            + "/.grok/AGENTS.md"),)}
            self.assertFalse(host_fence._grok_instruction_wall_held(wrap_kw, self.grok))
            wrap_kw["extra_post_read"] = (("literal", os.path.join(self.ext, "x")),)
            self.assertTrue(host_fence._grok_instruction_wall_held(wrap_kw, self.grok))


# --------------------------------------------------------------------------
# Issue bullet 6: the protected-branch check
# --------------------------------------------------------------------------
class ProtectedBranch(_Base):
    def _repo(self):
        repo = os.path.join(self.root, "repo")
        os.makedirs(repo)

        def git(*args):
            return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                                  text=True, check=True)
        git("init", "-q")
        # Keep names as stored: git would otherwise print both as NFC.
        git("config", "core.precomposeunicode", "false")
        git("-c", "user.email=t@example.invalid", "-c", "user.name=t",
            "commit", "-q", "--allow-empty", "-m", "init")
        git("update-ref", "refs/remotes/origin/" + unicodedata.normalize("NFC", "Rélease"), "HEAD")
        git("symbolic-ref", "refs/remotes/origin/HEAD",
            "refs/remotes/origin/" + unicodedata.normalize("NFC", "Rélease"))
        git("symbolic-ref", "HEAD",
            "refs/heads/" + unicodedata.normalize("NFD", "Rélease"))
        return repo

    def test_agent_lane_refuses_a_default_branch_in_the_other_normalization(self):
        env = agent_lane._workdir_fence(self._repo(), False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")

    def test_host_fence_refuses_repo_write_on_that_branch(self):
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(self._repo(), required=True, allow_main=False)
        self.assertIn("protected branch", caught.exception.envelope["detail"])

    def test_host_fence_allow_main_and_work_branches_are_unchanged(self):
        repo = self._repo()
        roots = host_fence._repo_write_roots(repo, required=True, allow_main=True)
        self.assertTrue(roots[0])
        subprocess.run(["git", "-C", repo, "symbolic-ref", "HEAD", "refs/heads/work"],
                       check=True)
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        self.assertTrue(host_fence._repo_write_roots(
            repo, required=True, allow_main=False)[0])

    def test_branch_name_in_folds_case_and_normalization_only(self):
        names = {"main", "rélease"}
        self.assertTrue(kernel_sandbox.branch_name_in("MAIN", names))
        self.assertTrue(kernel_sandbox.branch_name_in(
            unicodedata.normalize("NFD", "Rélease"), names))
        self.assertFalse(kernel_sandbox.branch_name_in("main2", names))
        self.assertFalse(kernel_sandbox.branch_name_in("feature/main", names))


class ProtectedBranchShortening(_Base):
    """``symbolic-ref --short`` prints the shortest UNAMBIGUOUS name.

    A tag ``main`` turns HEAD on ``refs/heads/main`` into ``heads/main``, and
    a branch ``origin/develop`` turns the remote HEAD into
    ``remotes/origin/develop``; a fenced child can create either, so the
    protected-branch checks read the full ref and strip the prefix.
    """

    def _git(self, repo, *args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    def _repo(self):
        repo = os.path.join(self.root, "repo")
        os.makedirs(repo)
        self._git(repo, "init", "-q")
        self._git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
                  "commit", "-q", "--allow-empty", "-m", "init")
        return repo

    def _refused_by_both(self, repo):
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertIn("protected branch", caught.exception.envelope["detail"])

    def test_tag_named_like_the_branch_does_not_hide_main(self):
        repo = self._repo()
        self._git(repo, "update-ref", "refs/heads/main", "HEAD")
        self._git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        self._git(repo, "tag", "main")
        self.assertEqual(self._git(repo, "symbolic-ref", "--short", "-q", "HEAD"),
                         "heads/main")         # what --short would have returned
        self._refused_by_both(repo)

    def test_branch_named_origin_slash_x_does_not_hide_the_default_branch(self):
        repo = self._repo()
        self._git(repo, "branch", "develop")
        self._git(repo, "branch", "origin/develop")
        self._git(repo, "update-ref", "refs/remotes/origin/develop", "HEAD")
        self._git(repo, "symbolic-ref", "refs/remotes/origin/HEAD",
                  "refs/remotes/origin/develop")
        self.assertEqual(
            self._git(repo, "symbolic-ref", "--short", "-q", "refs/remotes/origin/HEAD"),
            "remotes/origin/develop")           # what --short would have returned
        self._git(repo, "symbolic-ref", "HEAD", "refs/heads/develop")
        self._refused_by_both(repo)

    def test_feature_branch_is_still_admitted_with_such_a_tag_around(self):
        repo = self._repo()
        self._git(repo, "tag", "main")
        self._git(repo, "symbolic-ref", "HEAD", "refs/heads/feature/x")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        roots = host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertTrue(roots[0])

    def test_allow_main_still_overrides_and_detached_head_is_unchanged(self):
        repo = self._repo()
        self._git(repo, "update-ref", "refs/heads/main", "HEAD")
        self._git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
        self._git(repo, "tag", "main")
        self.assertIsNone(agent_lane._workdir_fence(repo, True))
        self.assertTrue(host_fence._repo_write_roots(
            repo, required=True, allow_main=True)[0])
        self._git(repo, "checkout", "-q", "--detach")
        self.assertIsNone(agent_lane._workdir_fence(repo, False))


class OriginDefaultBranch(_Base):
    """``refs/remotes/origin/HEAD`` names the default branch to protect.

    It is read the first time omni sees a repository (the names are recorded
    then, #168), and a state that is not "genuinely absent" or "a symbolic ref
    into refs/remotes/origin/" refuses instead of dropping the default branch
    from the protected set.
    """
    NBSP = " "

    def _git(self, repo, *args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    def _repo(self):
        repo = os.path.join(self.root, "repo")
        os.makedirs(repo)
        self._git(repo, "init", "-q")
        self._git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
                  "commit", "-q", "--allow-empty", "-m", "init")
        return repo

    def _on(self, repo, branch):
        self._git(repo, "update-ref", "refs/heads/" + branch, "HEAD")
        self._git(repo, "symbolic-ref", "HEAD", "refs/heads/" + branch)

    def _origin_file(self, repo, content):
        path = os.path.join(repo, ".git", "refs", "remotes", "origin", "HEAD")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)

    def _origin_default(self, repo, name):
        self._git(repo, "update-ref", "refs/remotes/origin/" + name, "HEAD")
        self._git(repo, "symbolic-ref", "refs/remotes/origin/HEAD",
                  "refs/remotes/origin/" + name)

    def _refused_by_both(self, repo):
        env = agent_lane._workdir_fence(repo, False)
        self.assertIsNotNone(env)
        self.assertEqual(env["reason"], "workdir_fence")
        with self.assertRaises(host_fence.HostFenceError) as caught:
            host_fence._repo_write_roots(repo, required=True, allow_main=False)
        return env, caught.exception.envelope["detail"]

    def _admitted_by_both(self, repo):
        self.assertIsNone(agent_lane._workdir_fence(repo, False))
        roots = host_fence._repo_write_roots(repo, required=True, allow_main=False)
        self.assertTrue(roots[0])
        return roots

    def test_self_looping_origin_head_refuses(self):
        repo = self._repo()
        self._on(repo, "develop")
        self._origin_file(repo, b"ref: refs/remotes/origin/HEAD\n")
        proc = subprocess.run(
            ["git", "-C", repo, "symbolic-ref", "-q", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 128)
        self._refused_by_both(repo)

    def test_origin_head_pointing_outside_the_remote_namespace_refuses(self):
        repo = self._repo()
        self._git(repo, "branch", "develop")
        self._on(repo, "feature/x")
        self._origin_file(repo, b"ref: refs/heads/develop\n")
        self._refused_by_both(repo)

    def test_origin_head_that_is_not_text_refuses(self):
        repo = self._repo()
        self._on(repo, "feature/x")
        self._origin_file(repo, b"ref: refs/remotes/origin/\xff\xfe\n")
        self._refused_by_both(repo)

    def test_a_failing_query_for_origin_head_refuses(self):
        repo = self._repo()
        self._on(repo, "feature/x")
        real = host_fence.git_safe.run_git

        def failing(workdir, args, **kw):
            if "refs/remotes/origin/HEAD" in list(args):
                raise host_fence.git_safe.GitSafeError("cannot build a safe env")
            return real(workdir, args, **kw)
        with mock.patch.object(host_fence.git_safe, "run_git", failing):
            self._refused_by_both(repo)

    def test_absent_origin_head_is_admitted_as_before(self):
        repo = self._repo()
        self._on(repo, "feature/x")
        self.assertFalse(os.path.lexists(
            os.path.join(repo, ".git", "refs", "remotes", "origin", "HEAD")))
        self._admitted_by_both(repo)
        # ... and main / master stay protected without one.
        self._on(repo, "master")
        self._refused_by_both(repo)

    def test_a_valid_origin_head_still_protects_its_branch(self):
        repo = self._repo()
        self._origin_default(repo, "develop")
        self._on(repo, "feature/x")
        self._admitted_by_both(repo)
        self._on(repo, "develop")
        self._refused_by_both(repo)

    def test_origin_head_ref_is_not_denied_for_writes(self):
        # #168: the protected names are recorded, not read from origin/HEAD,
        # so the deny is gone (and ``git remote set-head`` works in the fence).
        repo = self._repo()
        self._on(repo, "feature/x")
        roots = host_fence._repo_write_roots(repo, required=True, allow_main=False)
        common = os.path.join(os.path.realpath(repo), ".git")
        want = ("literal", os.path.join(common, "refs", "remotes", "origin", "HEAD"))
        self.assertNotIn(want, roots[1])
        self.assertNotIn(want, host_fence._repo_protect_denies(
            common, common, {"main"}, os.path.realpath(repo)))

    def test_ref_names_are_not_stripped(self):
        name = "develop" + self.NBSP
        repo = self._repo()
        self._origin_default(repo, name)
        self._on(repo, "feature/x")
        roots = self._admitted_by_both(repo)
        common = os.path.join(os.path.realpath(repo), ".git")
        # The deny names the real ref, not a stripped neighbour of it.
        self.assertIn(("literal", os.path.join(common, "refs", "heads", name)), roots[1])
        self._on(repo, name)
        self._refused_by_both(repo)

    def test_a_branch_that_only_ends_like_a_protected_one_is_not_refused(self):
        repo = self._repo()
        self._on(repo, "main" + self.NBSP)
        self._admitted_by_both(repo)


class RefDirectoryDenies(_Base):
    """Renaming a directory above a protected ref moves the ref out of its deny.

    ``refs/remotes/origin`` renamed away makes origin/HEAD look absent, and
    ``refs/heads`` renamed away moves every protected branch ref. A literal
    deny on the directory blocks the rename and nothing else (measured, see
    SeatbeltFacts); only directories that exist are denied, so creating the
    first ``refs/remotes/origin/x`` of a fetch keeps working.
    """

    def _git(self, repo, *args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    def _repo(self, origin_dir=True):
        repo = os.path.join(self.root, "repo")
        os.makedirs(repo)
        self._git(repo, "init", "-q")
        self._git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
                  "commit", "-q", "--allow-empty", "-m", "init")
        self._git(repo, "checkout", "-q", "-b", "feature/x")
        if origin_dir:
            os.makedirs(os.path.join(repo, ".git", "refs", "remotes", "origin"))
        return os.path.realpath(repo)

    def _rows(self, repo):
        roots = host_fence._repo_write_roots(repo, required=True, allow_main=False)
        return roots, [row for row in roots[1] if row[0] == "literal"]

    def test_ancestor_directories_of_protected_refs_are_denied(self):
        repo = self._repo()
        _roots, rows = self._rows(repo)
        common = os.path.join(repo, ".git")
        for parts in (("refs",), ("refs", "heads"), ("refs", "remotes"),
                      ("refs", "remotes", "origin")):
            self.assertIn(("literal", os.path.join(common, *parts)), rows, parts)

    def test_intermediate_directories_of_a_slashed_protected_name_are_denied(self):
        repo = self._repo()
        common = os.path.join(repo, ".git")
        os.makedirs(os.path.join(common, "refs", "heads", "release"))
        rows = host_fence._repo_protect_denies(
            common, common, {"main", "release/v1"}, repo)
        self.assertIn(("literal", os.path.join(common, "refs", "heads", "release")), rows)
        self.assertIn(("literal", os.path.join(
            common, "refs", "heads", "release", "v1")), rows)

    def test_directories_that_do_not_exist_are_not_denied(self):
        repo = self._repo(origin_dir=False)
        common = os.path.join(repo, ".git")
        _roots, rows = self._rows(repo)
        self.assertNotIn(("literal", os.path.join(common, "refs", "remotes", "origin")), rows)
        self.assertIn(("literal", os.path.join(common, "refs", "heads")), rows)

    def _fenced(self, repo):
        roots, denies, _meta, posts, finals = host_fence._repo_write_roots(
            repo, required=True, allow_main=False)

        def run(argv):
            wrapped = kernel_sandbox.wrap_argv(
                argv, repo, False, extra_write_paths=roots, extra_deny_write=denies,
                extra_deny_write_final=finals, extra_post_write=posts)
            proc = subprocess.run(wrapped, capture_output=True, text=True, timeout=60)
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        return run

    @unittest.skipUnless(platform.system() == "Darwin"
                         and os.access("/usr/bin/sandbox-exec", os.X_OK),
                         "needs the real sandbox-exec")
    def test_ordinary_git_work_still_succeeds_and_renames_are_refused(self):
        repo = self._repo()
        run = self._fenced(repo)
        if run(["/usr/bin/true"])[0] != 0:
            self.skipTest("sandbox-exec cannot run here")
        git = ["git", "-C", repo, "-c", "user.email=t@example.invalid",
               "-c", "user.name=t"]
        common = os.path.join(repo, ".git")
        # What a fetch and a day of feature-branch work do.
        for argv in (git + ["update-ref", "refs/remotes/origin/newref", "HEAD"],
                     git + ["commit", "-q", "--allow-empty", "-m", "work"],
                     git + ["branch", "topic/y"],
                     git + ["update-ref", "-d", "refs/remotes/origin/newref"]):
            with self.subTest(argv=argv[-4:]):
                code, out = run(argv)
                self.assertEqual(code, 0, out)
        # What hides a protected ref does not.
        for src, dst in (("refs/remotes/origin", "refs/remotes/o2"),
                         ("refs/heads", "refs/h2"), ("refs/remotes", "refs/r2")):
            with self.subTest(rename=src):
                code, _out = run(["/bin/mv", os.path.join(common, src),
                                  os.path.join(common, dst)])
                self.assertNotEqual(code, 0)
                self.assertTrue(os.path.isdir(os.path.join(common, src)))
        self.assertNotEqual(run(git + ["update-ref", "refs/heads/main", "HEAD"])[0], 0)

    @unittest.skipUnless(platform.system() == "Darwin"
                         and os.access("/usr/bin/sandbox-exec", os.X_OK),
                         "needs the real sandbox-exec")
    def test_a_fetch_may_create_the_first_remote_ref_directory(self):
        repo = self._repo(origin_dir=False)
        run = self._fenced(repo)
        if run(["/usr/bin/true"])[0] != 0:
            self.skipTest("sandbox-exec cannot run here")
        code, out = run(["git", "-C", repo, "update-ref",
                         "refs/remotes/origin/first", "HEAD"])
        self.assertEqual(code, 0, out)


# --------------------------------------------------------------------------
# Negative controls: ordinary layouts are not refused
# --------------------------------------------------------------------------
class OrdinaryLayouts(_Base):
    def test_plain_home_spawns_for_every_provider(self):
        for dot in (".pi", ".cursor", ".claude", ".codex", ".grok"):
            os.makedirs(os.path.join(self.home, dot, "agent", "sessions"), exist_ok=True)
        for provider in ("pi", "cursor", "claude", "codex", "grok"):
            with self.subTest(provider=provider):
                self.assert_spawns(self.policy(provider))

    def test_dotfiles_symlink_outside_the_write_roots_spawns(self):
        for provider, dot, leaves in (("pi", ".pi", ("agent", "sessions")),
                                      ("cursor", ".cursor", ("projects",))):
            with self.subTest(provider=provider):
                real = os.path.join(self.ext, provider, dot)
                os.makedirs(os.path.join(real, *leaves))
                os.symlink(real, os.path.join(self.home, dot))
                os.makedirs(os.path.join(self.home, ".codex"), exist_ok=True)
                self.assert_spawns(self.policy(provider))

    def test_codex_home_symlinked_outside_write_roots_spawns_for_other_lanes(self):
        os.makedirs(os.path.join(self.ext, "vendor", "codex"))
        os.symlink(os.path.join(self.ext, "vendor", "codex"),
                   os.path.join(self.home, ".codex"))
        os.makedirs(os.path.join(self.home, ".cursor", "projects"))
        os.makedirs(os.path.join(self.home, ".pi", "agent", "sessions"))
        for provider in ("pi", "cursor"):
            with self.subTest(provider=provider):
                self.assert_spawns(self.policy(provider))

    def test_ordinary_config_dirs_spawn(self):
        os.makedirs(os.path.join(self.home, ".cursor-work"))
        self.assert_spawns(self.policy("cursor", config_dir=os.path.join(
            self.home, ".cursor-work")))
        outside = os.path.join(self.root, "cfg")
        os.makedirs(outside)
        self.assert_spawns(self.policy("cursor", config_dir=outside))

    def test_distinct_account_homes_are_not_nested_in_each_other(self):
        os.makedirs(os.path.join(self.root, "acct"))
        os.makedirs(os.path.join(self.root, "acct2"))
        self.assertFalse(profiles._nested(os.path.join(self.root, "acct2"),
                                          os.path.join(self.root, "acct")))
        self.assertTrue(profiles._nested(os.path.join(self.root, "ACCT", "x"),
                                         os.path.join(self.root, "acct")))

    def test_settings_base_beside_its_siblings_is_not_exposed(self):
        base = os.path.join(self.home, ".cache", "omni-llm", "api-settings")
        self.assertIsNone(host_fence.settings_base_exposed(
            base, [("subpath", os.path.join(self.home, ".cache", "omni-llm2")),
                   ("subpath", self.wd), ("literal", os.path.join(self.home, ".cache"
                                                                 + "x"))]))


# --------------------------------------------------------------------------
# Fix round 1: case-sensitive volumes, "..", rows at a link, shared walks
# --------------------------------------------------------------------------
def _case_sensitive_stat(real_stat):
    """``_stat_dev_ino`` as a case-sensitive volume would answer it.

    On the (case-insensitive) boot volume ``HOME`` is the same inode as
    ``home``; a case-sensitive one would hold two directories. A path with
    any component not spelled as it is on disk gets a second identity.
    """
    def exact(path):
        cur = os.sep
        for part in [p for p in path.split(os.sep) if p]:
            try:
                if part not in os.listdir(cur):
                    return False
            except OSError:
                return True
            cur = os.path.join(cur, part)
        return True

    def stat(path):
        ident = real_stat(path)
        if ident is None or exact(os.path.abspath(path)):
            return ident
        return (ident[0], ident[1] + 10 ** 9)
    return stat


class CaseSensitiveVolume(_Base):
    def setUp(self):
        super().setUp()
        self.alias = os.path.join(self.root, "HOME")
        patcher = mock.patch.object(
            kernel_sandbox, "_stat_dev_ino",
            _case_sensitive_stat(kernel_sandbox._stat_dev_ino))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_distinct_dirs_with_folded_equal_names_are_one_location_to_a_rule(self):
        self.assertNotEqual(kernel_sandbox._stat_dev_ino(self.home),
                            kernel_sandbox._stat_dev_ino(self.alias))
        self.assertTrue(kernel_sandbox.same_location(self.alias, self.home))
        self.assertTrue(kernel_sandbox.location_inside(
            os.path.join(self.alias, "x"), self.home))

    def test_exemptions_keep_exact_spelling_and_inode(self):
        self.assertFalse(kernel_sandbox.exempt_same(self.alias, self.home))
        self.assertFalse(kernel_sandbox.exempt_inside(
            os.path.join(self.alias, "x"), self.home))
        self.assertFalse(kernel_sandbox._same_dir_exact(self.alias, self.home))
        self.assertTrue(kernel_sandbox.exempt_same(self.home, self.home))

    def test_cursor_config_dir_in_other_case_of_the_protected_home_is_refused(self):
        os.makedirs(os.path.join(self.home, ".codex"))
        self.assert_refused(self.policy("cursor", config_dir=self.alias))

    def test_cursor_config_dir_beside_the_protected_home_still_spawns(self):
        other = os.path.join(self.root, "homes")
        os.makedirs(other)
        self.assert_spawns(self.policy("cursor", config_dir=other))

    def test_reopens_and_settings_checks_fold_across_two_existing_dirs(self):
        row = os.path.join(self.home, ".grok", "AGENTS.md")
        os.makedirs(os.path.dirname(row))
        self.assertTrue(host_fence._reopens(
            ("subpath", os.path.join(self.alias, ".grok")), [("literal", row)]))
        self.assertTrue(host_fence.settings_base_exposed(
            os.path.join(self.home, "api"), [("subpath", self.alias)]))


@unittest.skipUnless(platform.system() == "Darwin"
                     and shutil.which("hdiutil") is not None,
                     "needs hdiutil to build a case-sensitive APFS volume")
class RealCaseSensitiveVolume(unittest.TestCase):
    """The same, on a real case-sensitive APFS image (skips when none mounts)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="omni_sloc_cs_"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        image = os.path.join(cls.tmp, "cs.dmg")
        cls.mnt = os.path.join(cls.tmp, "mnt")
        os.makedirs(cls.mnt)
        try:
            made = subprocess.run(
                ["hdiutil", "create", "-size", "20m", "-fs", "Case-sensitive APFS",
                 "-volname", "omnisloc", image],
                capture_output=True, timeout=120)
            attached = made.returncode == 0 and subprocess.run(
                ["hdiutil", "attach", "-mountpoint", cls.mnt, "-nobrowse",
                 "-noverify", image], capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            attached = None
        if not attached or attached.returncode != 0:
            raise unittest.SkipTest("cannot attach a case-sensitive APFS image")
        cls.addClassCleanup(subprocess.run, ["hdiutil", "detach", "-force", cls.mnt],
                            capture_output=True, timeout=120)
        try:
            for name in ("home", "HOME", "wd", "ext"):
                os.makedirs(os.path.join(cls.mnt, name))
        except OSError:
            raise unittest.SkipTest("case-sensitive volume is not writable")
        if os.stat(os.path.join(cls.mnt, "home")).st_ino == os.stat(
                os.path.join(cls.mnt, "HOME")).st_ino:
            raise unittest.SkipTest("image is not case-sensitive")

    def test_two_directories_are_one_location_for_a_rule_but_not_an_exemption(self):
        home, alias = (os.path.join(self.mnt, "home"), os.path.join(self.mnt, "HOME"))
        self.assertTrue(kernel_sandbox.same_location(alias, home))
        self.assertTrue(kernel_sandbox.location_inside(os.path.join(alias, "x"), home))
        self.assertFalse(kernel_sandbox.exempt_same(alias, home))
        self.assertFalse(kernel_sandbox.exempt_inside(os.path.join(alias, "x"), home))
        self.assertFalse(kernel_sandbox.same_location(
            os.path.join(self.mnt, "ext"), home))

    def test_seatbelt_matches_a_rule_across_case_on_this_volume(self):
        if not os.access("/usr/bin/sandbox-exec", os.X_OK):
            self.skipTest("no sandbox-exec")
        rule = os.path.join(self.mnt, "HOME")
        for mode, profile in (
                ("deny", '(version 1)(allow default)(deny file-write* (subpath "%s"))' % rule),
                ("allow", '(version 1)(allow default)(deny file-write*)'
                          '(allow file-write* (subpath "%s"))' % rule)):
            for name in ("HOME", "home"):
                target = os.path.join(self.mnt, name, "f")
                proc = subprocess.run(
                    ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c",
                     ': > "$1"', "sh", target], capture_output=True, timeout=20)
                if os.path.exists(target):
                    os.unlink(target)
                # A rule for HOME governs the distinct directory home too.
                self.assertEqual(proc.returncode == 0, mode == "allow",
                                 (mode, name))

    def test_policy_builder_refuses_the_other_case_config_dir(self):
        home, alias = (os.path.join(self.mnt, "home"), os.path.join(self.mnt, "HOME"))
        with mock.patch.object(kernel_sandbox, "protected_home", return_value=home), \
             mock.patch.object(kernel_sandbox, "_write_allow_paths",
                               _isolated_write_roots):
            policy = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=os.path.join(self.mnt, "wd"), read_only=False,
                config_dir=alias)
            self.assertEqual(policy.get("reason"), "fence_path_rejected", policy)
            ok = kernel_sandbox.agent_fence_policy(
                "cursor", workdir=os.path.join(self.mnt, "wd"), read_only=False,
                config_dir=os.path.join(self.mnt, "ext"))
            self.assertNotIn("ok", ok, ok)


class DotDotBeforeSymlink(_Base):
    """``link/..`` is the parent of the link's TARGET, not of the link."""

    def setUp(self):
        super().setUp()
        os.makedirs(os.path.join(self.home, "sub"))
        os.makedirs(os.path.join(self.ext, "dir"))
        os.symlink(os.path.join(self.home, "sub"), os.path.join(self.ext, "link"))
        self.raw = os.path.join(self.ext, "link", "..", "secret")

    def test_refusal_side_sees_the_target_parent(self):
        self.assertTrue(kernel_sandbox.location_inside(self.raw, self.home))
        self.assertTrue(kernel_sandbox.same_location(
            self.raw, os.path.join(self.home, "secret")))

    def test_exemption_does_not_take_the_link_parent(self):
        self.assertFalse(kernel_sandbox.exempt_inside(self.raw, self.ext))
        self.assertTrue(kernel_sandbox.exempt_inside(self.raw, self.home))

    def test_relative_input_resolves_the_same_way(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.ext)
        rel = os.path.join("link", "..", "secret")
        self.assertTrue(kernel_sandbox.location_inside(rel, self.home))
        self.assertFalse(kernel_sandbox.exempt_inside(rel, self.ext))
        # A plain "..": no link involved, ordinary normalization.
        self.assertTrue(kernel_sandbox.exempt_inside(
            os.path.join("dir", "..", "x"), self.ext))
        self.assertFalse(kernel_sandbox.location_inside(
            os.path.join("dir", "..", "x"), self.home))

    def test_a_link_judged_by_its_own_name_still_resolves_dots_in_its_parent(self):
        spelled = os.path.join(self.ext, "dir", "..", "link")
        self.assertFalse(kernel_sandbox.location_inside(
            spelled, os.path.join(self.home, "sub"), resolve_leaf=False))
        self.assertTrue(kernel_sandbox.location_inside(
            spelled, self.ext, resolve_leaf=False))


class VendorLinkRowAtTheLink(_Base):
    def setUp(self):
        super().setUp()
        self.real = os.path.join(self.ext, "pi")
        os.makedirs(self.real)
        self.link = os.path.join(self.home, ".pi")
        os.symlink(self.real, self.link)

    def _refusal(self, rows):
        return kernel_sandbox._vendor_root_link_refusal(
            "pi", self.link, self.real, rows, [])

    def test_row_spelled_at_the_link_refuses(self):
        for row in (self.link, os.path.join(self.home, ".PI"), self.home):
            with self.subTest(row=row):
                why = self._refusal([row])
                self.assertIsNotNone(why)
                self.assertIn("symlink the run can write", why)

    def test_unrelated_rows_do_not_refuse(self):
        self.assertIsNone(self._refusal([self.wd, os.path.join(self.home, ".pi2")]))


class SharedWalks(_Base):
    def _count(self):
        calls = []
        real = kernel_sandbox._stat_dev_ino

        def counting(path):
            calls.append(path)
            return real(path)
        return calls, mock.patch.object(kernel_sandbox, "_stat_dev_ino", counting)

    def test_a_cache_block_walks_each_path_once(self):
        a, b = os.path.join(self.home, "a", "x"), os.path.join(self.home, "b")
        calls, patch = self._count()
        with patch:
            kernel_sandbox.location_inside(a, b)
            cold = len(calls)
            del calls[:]
            with kernel_sandbox._location_cache():
                kernel_sandbox.location_inside(a, b)
                first = len(calls)
                del calls[:]
                for _ in range(5):
                    kernel_sandbox.location_inside(a, b)
                    kernel_sandbox.location_inside(b, a)
                again = len(calls)
        self.assertGreater(cold, 0)
        self.assertLessEqual(first, cold)
        self.assertEqual(again, 0)

    def test_nothing_is_cached_outside_a_block(self):
        newdir = os.path.join(self.home, "later")
        self.assertFalse(kernel_sandbox.location_inside(newdir, self.wd))
        os.makedirs(newdir)
        with kernel_sandbox._location_cache():
            self.assertTrue(kernel_sandbox.same_location(newdir, newdir))
        os.rmdir(newdir)
        os.symlink(self.wd, newdir)
        self.assertTrue(kernel_sandbox.location_inside(newdir, self.wd))

    def test_grok_policy_build_stays_within_a_small_identity_budget(self):
        for dot in (".grok", ".claude", ".codex", ".cursor", ".pi"):
            os.makedirs(os.path.join(self.home, dot), exist_ok=True)
        calls, patch = self._count()
        with patch:
            self.assert_spawns(self.policy("grok"))
        self.assertLess(len(calls), 120, len(calls))


# --------------------------------------------------------------------------
# Seatbelt facts, measured with the real sandbox-exec
# --------------------------------------------------------------------------
@unittest.skipUnless(platform.system() == "Darwin"
                     and os.access("/usr/bin/sandbox-exec", os.X_OK),
                     "Seatbelt facts are measured on Darwin")
class SeatbeltFacts(_Base):
    """Which spelling of a path rule applies to which spelling of an access.

    These are the facts the refusal/exemption split rests on. Seatbelt judges
    the KERNEL-resolved path of an access: rules match every case and
    normalization spelling and a rule on a symlink never reaches its target,
    but an access spelled through a symlink or the data-volume firmlink is
    judged as the target. A rule spelled with the firmlink prefix therefore
    never matches (a deny spelled that way protects nothing, an allow spelled
    that way grants nothing), which is why spelling is not what the fence
    compares: identity of the existing ancestor plus the names below it is.
    """

    def setUp(self):
        super().setUp()
        self.base = os.path.join(self.root, "sb")
        for sub in ("a", "real", "new", NFC):
            os.makedirs(os.path.join(self.base, sub))
        os.symlink(os.path.join(self.base, "real"), os.path.join(self.base, "link"))
        control = self._run("(version 1)(allow default)",
                            os.path.join(self.base, "a", "ctl"))
        if control is None or control is False:
            self.skipTest("sandbox-exec cannot run here")

    @staticmethod
    def _run_profile(profile, target):
        try:
            proc = subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c",
                 ': > "$1"', "sh", target],
                capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.returncode == 0

    def _run(self, profile, target):
        ok = self._run_profile(profile, target)
        if ok:
            os.unlink(target)
        return ok

    def _write_allowed(self, mode, kind, rule, target):
        if mode == "deny":
            profile = '(version 1)(allow default)(deny file-write* (%s "%s"))' % (kind, rule)
        else:
            profile = ('(version 1)(allow default)(deny file-write*)'
                       '(allow file-write* (%s "%s"))' % (kind, rule))
        got = self._run(profile, target)
        self.assertIsNotNone(got, "sandbox-exec did not run")
        return got

    def test_firmlink_prefixed_rule_never_matches_a_plain_access(self):
        firm = self.firm(self.base)
        for kind, rule, target in (
                ("subpath", firm + "/a", os.path.join(self.base, "a", "f")),
                ("literal", firm + "/a/f", os.path.join(self.base, "a", "f")),
                ("subpath", firm + "/new", os.path.join(self.base, "new", "f"))):
            with self.subTest(kind=kind, rule=rule):
                self.assertTrue(self._write_allowed("deny", kind, rule, target))
                self.assertFalse(self._write_allowed("allow", kind, rule, target))

    def test_plain_rule_matches_a_firmlink_spelled_access(self):
        firm = self.firm(self.base)
        for kind, rule, target in (
                ("subpath", os.path.join(self.base, "a"), firm + "/a/f"),
                ("literal", os.path.join(self.base, "a", "f"), firm + "/a/f")):
            with self.subTest(kind=kind):
                self.assertFalse(self._write_allowed("deny", kind, rule, target))
                self.assertTrue(self._write_allowed("allow", kind, rule, target))

    def test_rule_on_a_symlink_never_reaches_the_target_but_the_target_rule_does(self):
        link, real = (os.path.join(self.base, "link"), os.path.join(self.base, "real"))
        self.assertTrue(self._write_allowed("deny", "subpath", link,
                                            os.path.join(real, "f")))
        self.assertFalse(self._write_allowed("allow", "subpath", link,
                                             os.path.join(real, "f")))
        self.assertFalse(self._write_allowed("deny", "subpath", real,
                                             os.path.join(link, "f")))
        self.assertTrue(self._write_allowed("allow", "subpath", real,
                                            os.path.join(link, "f")))

    def test_a_rule_on_a_path_that_does_not_exist_yet_covers_creating_it(self):
        gone = os.path.join(self.base, "gone")
        for profile, expect in (
                ('(version 1)(allow default)(deny file-write* (subpath "%s"))' % gone, False),
                ('(version 1)(allow default)(deny file-write* (literal "%s"))' % gone, False),
                ('(version 1)(allow default)', True)):
            with self.subTest(profile=profile):
                proc = subprocess.run(
                    ["/usr/bin/sandbox-exec", "-p", profile, "/bin/mkdir", gone],
                    capture_output=True, timeout=20)
                self.assertEqual(proc.returncode == 0, expect)
                shutil.rmtree(gone, True)

    def test_a_literal_deny_on_a_directory_blocks_renaming_it_not_filling_it(self):
        directory = os.path.join(self.base, "a")
        profile = ('(version 1)(allow default)(deny file-write* (literal "%s"))'
                   % directory)

        def run(*argv):
            proc = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, *argv],
                                  capture_output=True, timeout=20)
            return proc.returncode == 0
        moved = os.path.join(self.base, "moved")
        self.assertFalse(run("/bin/mv", directory, moved))
        self.assertTrue(os.path.isdir(directory))
        self.assertTrue(run("/bin/sh", "-c", ': > "$1"', "sh",
                            os.path.join(directory, "newref")))
        self.assertTrue(run("/bin/mkdir", os.path.join(directory, "sub")))
        self.assertTrue(run("/bin/mv", os.path.join(directory, "newref"),
                            os.path.join(directory, "renamed")))

    def test_rules_match_other_case_and_normalization_spellings(self):
        upper = os.path.join(self.base, "A")
        target = os.path.join(self.base, "a", "f")
        self.assertFalse(self._write_allowed("deny", "subpath", upper, target))
        self.assertTrue(self._write_allowed("allow", "subpath", upper, target))
        nfd = os.path.join(self.base, NFD)
        nfc_target = os.path.join(self.base, NFC, "f")
        self.assertFalse(self._write_allowed("deny", "subpath", nfd, nfc_target))
        self.assertTrue(self._write_allowed("allow", "subpath", nfd, nfc_target))
        nfc_rule = os.path.join(self.base, NFC)
        nfd_target = os.path.join(self.base, NFD, "f")
        self.assertFalse(self._write_allowed("deny", "subpath", nfc_rule, nfd_target))
        self.assertTrue(self._write_allowed("allow", "subpath", nfc_rule, nfd_target))


if __name__ == "__main__":
    unittest.main()
