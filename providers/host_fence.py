"""One host-owned verify-and-wrap operation for macOS task subprocesses.

Python callers retain their own subprocess/stream handling. ``host-exec``
replaces itself with the same prepared command; it is not a supervisor.
Non-macOS callers retain their existing containment and environment policy.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unicodedata

from providers import git_safe, kernel_sandbox, vendor_bin
from providers.errors import redact_secrets


_GROK_VALUE_OPTIONS = {
    "-p", "--prompt", "--prompt-file", "-m", "--model",
    "--reasoning-effort", "--output-format", "--cwd",
    "-s", "--session-id", "-r", "--resume", "--session",
    "--allowed-tools", "--disallowed-tools", "--tools",
    "--max-turns", "--max-tokens", "--system-prompt",
    "--append-system-prompt", "--permission-mode",
    "--system-prompt-override", "--prompt-json",
}

# cursor-agent value-taking flags in the supported headless invocations.
# --sandbox is handled separately so a value of "--sandbox" on --model
# (or --workspace, --mode, --resume) is not treated as the vendor flag.
_CURSOR_VALUE_OPTIONS = {
    "-H", "--header", "-e", "--endpoint", "--api-key",
    "--output-format", "--mode", "--resume", "--model",
    "--workspace", "--add-dir", "--plugin-dir", "--worktree-base",
}


class HostFenceError(RuntimeError):
    """A refusal that must be returned before spawning the vendor command."""

    def __init__(self, envelope):
        self.envelope = dict(envelope)
        self.envelope.setdefault("fence", "none")
        self.envelope.setdefault(
            "fence_detail", dict(kernel_sandbox.FENCE_NONE_DETAIL))
        super().__init__(self.envelope.get("detail", "host fence refused"))


def _refuse(reason, detail):
    # Masked where the refusal is made: a detail can quote a branch name.
    raise HostFenceError({"ok": False, "reason": reason, "detail": redact_secrets(detail)})


# The two fixed homes of generated API settings (agent lane). Every fenced
# child of every provider gets a final deny-write on both (they only cover
# omni's own private areas). When API-kind accounts are configured, a child
# is also refused when its write roots could rename either one or an
# ancestor of it; without API accounts that refusal does not apply.
ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"


def api_accounts_configured():
    """True when API-kind accounts are configured (``OMNI_LLM_API_PROFILES_CLAUDE``).

    Reuses the profile parser: a non-empty parsed label set counts. A set but
    malformed value also counts (fail closed: the API lane would refuse it
    loudly, and the containment guard must not be skipped meanwhile). Unset or
    an empty value counts as not configured.
    """
    from providers import profiles
    if not (os.environ.get(profiles.API_PROFILES_VAR) or "").strip():
        return False
    labels, err = profiles.claude_api_labels()
    return bool(labels) or bool(err)


def api_settings_bases():
    """Both API settings base spellings: state-dir area, per-user home area.

    Nothing is created or resolved here. Raises ``RuntimeError`` when the
    account home cannot be resolved (callers fail closed).
    """
    state = os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR))
    home = kernel_sandbox.protected_home()
    return (os.path.join(state, "api-settings"),
            os.path.join(home, ".cache", "omni-llm", "api-settings"))


def settings_base_exposed(base, entries):
    """Reason string when a write entry lets a child rename *base* or an ancestor.

    *entries* is ``[(kind, path)]`` of the composed write set. A subpath root
    containing the base contains every ancestor-or-self it could rename; a
    literal root equal to an ancestor-or-self can be renamed directly. Any
    other kind is judged by its bound (``_write_entry_bound``); one without a
    bound cannot be reasoned about and counts as exposure (fail closed).
    """
    spellings = {os.path.abspath(base), os.path.realpath(base)}
    with kernel_sandbox._location_cache():
        for entry in entries:
            bound = kernel_sandbox._write_entry_bound(*entry)
            if bound is None:
                return "write root kind %r cannot be checked against %s" % (entry[0], base)
            kind, root = bound
            # Either kind reaches the base when the root is the base or one of
            # its ancestors, in any spelling (case, NFC/NFD, inode identity).
            if any(kernel_sandbox._path_inside_any_case(spelling, root)
                   for spelling in spellings):
                if kind == "literal":
                    return "write root %s is %s or one of its ancestors" % (root, base)
                return "write root %s contains %s" % (root, base)
    return None


def repo_root_candidates(workdir):
    """Best-effort Git roots the host fence may admit for *workdir*.

    A superset used only by selection pre-filters (no branch, identity or
    hard-link judgement): the authoritative composition stays in
    ``prepare_host_fence``. Any Git failure yields no roots.
    """
    cwd = os.fspath(workdir) if workdir is not None else None
    git_bin = shutil.which("git", path=os.defpath)
    if not cwd or not git_bin or not os.path.isdir(cwd):
        return ()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    out = []
    for option in ("--show-toplevel", "--absolute-git-dir", "--git-common-dir"):
        try:
            proc = git_safe.run_git(
                cwd, ("rev-parse", option), git_bin=os.path.realpath(git_bin),
                env=env)
        except (OSError, subprocess.SubprocessError, git_safe.GitSafeError):
            return ()
        value = proc.stdout.strip()
        if proc.returncode or not value:
            return ()
        if not os.path.isabs(value):
            value = os.path.join(cwd, value)
        resolved = os.path.realpath(value)
        if resolved not in out:
            out.append(resolved)
    return tuple(out)


def _rewrite_inner_sandbox(wrapped, original_argv, value, value_options):
    """Replace vendor ``--sandbox`` after a verified outer wrap.

    Isolation is the outer kernel wrap. The vendor flag stays as a CLI
    compatibility switch so the child does not apply a second Seatbelt.
    Value-taking flags keep their operands, even when an operand equals
    ``--sandbox``. A bare trailing ``--sandbox`` is dropped, not kept.
    The rewritten vendor argv replaces the original tail.
    """
    vendor_argv = list(original_argv[:1])
    i = 1
    n = len(original_argv)
    while i < n:
        token = original_argv[i]
        if token == "--":
            vendor_argv.extend(original_argv[i:])
            break
        if token == "--sandbox":
            i += 2 if i + 1 < n else 1
            continue
        if isinstance(token, str) and token.startswith("--sandbox="):
            i += 1
            continue
        if token in value_options and i + 1 < n:
            vendor_argv.extend(original_argv[i:i + 2])
            i += 2
            continue
        vendor_argv.append(token)
        i += 1
    vendor_argv[1:1] = ["--sandbox", value]
    return wrapped[:-len(original_argv)] + vendor_argv, vendor_argv


def _sandbox_flag_values(argv, value_options=()):
    """Vendor ``--sandbox`` flag operands in *argv*, in order of appearance.

    When *value_options* is set, *argv* is a vendor argv whose first token
    is the binary: value-taking flags keep their operands, so a model or
    workspace equal to ``--sandbox=enabled`` is not a sandbox flag.
    """
    values = []
    i = 1 if value_options else 0
    n = len(argv)
    while i < n:
        token = argv[i]
        if token == "--":
            break
        if token == "--sandbox" and i + 1 < n:
            values.append(argv[i + 1])
            i += 2
            continue
        if token == "--sandbox":
            values.append("")
            i += 1
            continue
        if isinstance(token, str) and token.startswith("--sandbox="):
            values.append(token.split("=", 1)[1])
            i += 1
            continue
        if token in value_options and i + 1 < n:
            i += 2
            continue
        i += 1
    return tuple(values)


def _ref_storage_format(git_run):
    """The ref storage format Git reports, or "" when it cannot be determined.

    The protected-branch denies below name loose ref files (``refs/heads/*``,
    ``HEAD``, ``packed-refs``). Any other ref backend (reftable keeps refs in
    ``<common dir>/reftable/`` tables) stores them where no deny reaches. The
    format is read from outside the child; only exactly ``files`` is covered.

    ``rev-parse --show-ref-format`` answers when the Git knows the option. A
    Git that does not know it either fails the call or (before 2.45) echoes the
    unknown flag on stdout with exit 0; the declared format is then read from
    ``extensions.refStorage`` (unset means ``files``). This is sound because a
    Git that does not know reftable refuses a repository that declares the
    extension before this point (the earlier ``rev-parse`` calls fail). Every
    other failure, empty or multi-line output counts as undetermined.
    """
    proc = git_run("rev-parse", "--show-ref-format")
    value = (proc.stdout or "").strip() if proc.returncode == 0 else None
    if value is None or value.startswith("-"):
        proc = git_run("config", "--get", "extensions.refStorage")
        out = (proc.stdout or "").strip()
        if proc.returncode == 1 and not out:
            return "files"
        value = out if proc.returncode == 0 else ""
    return value if "\n" not in value else ""


def _ref_storage_refusal(value, subject):
    shown = value if re.fullmatch(r"[a-z]{1,16}", value) else "undetermined"
    return ("%s requires files ref storage (found %s); "
            "branch protection does not cover other ref backends" % (subject, shown))


# Entries the nested-repository scan may visit under a non-git write workdir.
# Past it the scan stops and write mode refuses: a tree that big cannot be
# shown free of a Git repository, and an unverified tree is not admitted.
NESTED_REPO_SCAN_CAP = 20000


def _fold(name):
    return unicodedata.normalize("NFC", name).casefold()


def _is_git_name(name):
    return _fold(name) == ".git"


def _nested_repo_hit(root, exempt=None):
    """``(kind, relative_path)`` of the first Git repository under *root*.

    ``kind`` is ``"repo"`` (an entry named ``.git`` of any type: directory,
    ``gitdir:`` pointer file, or symlink) or ``"bare"`` (a directory with
    ``HEAD``, ``objects`` and ``refs``; each name matched in any spelling,
    NFC and case folded like ``.git``, since a case-insensitive volume lets
    Git open ``head`` or ``Objects``). ``(None, None)`` when the walk
    finishes clean, ``("incomplete", reason)`` when it could not finish: the
    entry cap was reached or a directory could not be read. Symlinked
    directories are never descended (they resolve outside the fence, where the
    kernel already denies writes); a symlink *named* ``.git`` still counts.
    Breadth first, so the hit named is the shallowest.

    *exempt*, when given, is called as ``exempt(relative_path, entry)`` for an
    entry named ``.git`` and returns true to skip it: the walk goes on (an
    exempt ``.git`` is not descended into). Bare-repository detection and every
    other rule are unchanged.
    """
    visited = 0
    pending = [""]
    while pending:
        rel_dir = pending.pop(0)
        here = os.path.join(root, rel_dir) if rel_dir else root
        try:
            entries = []
            with os.scandir(here) as scan:
                for entry in scan:
                    visited += 1
                    if visited > NESTED_REPO_SCAN_CAP:
                        return "incomplete", (
                            "more than %d entries" % NESTED_REPO_SCAN_CAP)
                    entries.append(entry)
        except OSError:
            return "incomplete", "a directory could not be read"
        entries.sort(key=lambda e: e.name)
        has_head = has_objects = has_refs = False
        subdirs = []
        for entry in entries:
            rel = os.path.join(rel_dir, entry.name) if rel_dir else entry.name
            if _is_git_name(entry.name):
                if exempt is not None and exempt(rel, entry):
                    continue
                return "repo", rel
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_link = entry.is_symlink()
            except OSError:
                return "incomplete", "a directory could not be read"
            folded = _fold(entry.name)
            if folded == "head" and not is_dir:
                has_head = True
            elif folded == "objects" and (is_dir or is_link):
                has_objects = True
            elif folded == "refs" and (is_dir or is_link):
                has_refs = True
            if is_dir:
                subdirs.append(rel)
        if has_head and has_objects and has_refs:
            return "bare", rel_dir or "."
        pending.extend(subdirs)
    return None, None


def _refuse_unverified_scan(reason):
    _refuse("fence_path_rejected",
            "write mode could not verify that no Git repository is nested "
            "under the workdir (%s; scan cap %d entries) - fence fails closed"
            % (reason, NESTED_REPO_SCAN_CAP))


def _refuse_nested_repo(cwd):
    """Refuse write mode when a Git repository sits under a non-git *cwd*.

    A workdir that is not a Git tree is writable as a whole, so the ``.git``
    of a repository nested anywhere below it is writable too, and the
    protected-branch denies (which name the workdir repository only) do not
    reach it. The scan runs before any child launches.
    """
    kind, found = _nested_repo_hit(cwd)
    if kind is None:
        return
    if kind == "incomplete":
        _refuse_unverified_scan(found)
    # Mask the whole path, then escape and cut: a credential-shaped name
    # straddling the cut must not leave its prefix behind.
    shown = ascii(redact_secrets(found))[:200]
    _refuse("fence_path_rejected",
            "write mode refuses a workdir that is not a Git tree but holds a "
            "nested Git repository (%s at %s): its branches are writable"
            % (".git entry" if kind == "repo" else "bare repository", shown))


def _linked_pointer(dotgit):
    """``(target, problem)`` of the ``gitdir:`` pointer file *dotgit*.

    *target* is the pointed path made absolute against the file's directory
    (not resolved); *problem* is the refusal detail, or None.
    """
    try:
        with open(dotgit, "r", encoding="utf-8") as stream:
            pointer = stream.read(4097)
    except OSError:
        return None, "worktree .git pointer cannot be read"
    if len(pointer) > 4096 or not pointer.startswith("gitdir: ") or "\n" in pointer.rstrip("\n"):
        return None, "worktree .git pointer is malformed"
    pointed = pointer[len("gitdir: "):].strip()
    if not os.path.isabs(pointed):
        pointed = os.path.join(os.path.dirname(dotgit), pointed)
    return pointed, None


def _linked_metadata_problem(dotgit, git_dir, common_dir, resolve=False):
    """Refusal detail when *git_dir* is not the linked-worktree directory of
    the pointer file *dotgit* under *common_dir*, or None when it is.

    Checks the ``gitdir`` backlink (must name *dotgit*), the ``commondir``
    file (must resolve to *common_dir*) and that *git_dir* sits under
    ``<common_dir>/worktrees/``. With *resolve* the backlink is compared by
    real path (a relative link, as ``worktree.useRelativePaths`` writes, keeps
    ``..`` in its joined spelling); the default compares it lexically.
    """
    gitdir_file = os.path.join(git_dir, "gitdir")
    common_file = os.path.join(git_dir, "commondir")
    try:
        for path in (gitdir_file, common_file):
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                return "linked Git metadata identity is not regular"
        with open(gitdir_file, "r", encoding="utf-8") as stream:
            backlink = stream.read(4097).strip()
        with open(common_file, "r", encoding="utf-8") as stream:
            commondir = stream.read(4097).strip()
    except OSError:
        return "linked Git metadata identity is incomplete"
    if len(backlink) > 4096 or len(commondir) > 4096:
        return "linked Git metadata pointer is oversized"
    if not os.path.isabs(backlink):
        backlink = os.path.join(git_dir, backlink)
    if not os.path.isabs(commondir):
        commondir = os.path.join(git_dir, commondir)
    if resolve:
        if os.path.realpath(backlink) != os.path.realpath(dotgit):
            return "linked Git backlink does not match .git pointer"
    elif backlink != dotgit or os.path.realpath(backlink) != dotgit:
        return "linked Git backlink does not match .git pointer"
    if os.path.realpath(commondir) != common_dir:
        return "linked Git common directory changed"
    if not git_dir.startswith(os.path.join(common_dir, "worktrees") + os.sep):
        return "linked Git directory is outside common worktrees"
    return None


def _is_own_linked_worktree(dotgit, common_dir):
    """True when the regular file *dotgit* is the pointer of a linked worktree
    of the repository whose common directory is *common_dir*.

    The pointer target and the ``gitdir`` backlink may be relative (git's
    ``worktree.useRelativePaths``): both are resolved before they are
    compared. The resolved target must be a real directory that is a direct
    child of ``<common_dir>/worktrees/``, and must name *dotgit* back and
    *common_dir* as its common directory. Any failed check is False.
    """
    try:
        pointed, problem = _linked_pointer(dotgit)
        if problem:
            return False
        resolved = os.path.realpath(pointed)
        common = os.path.realpath(common_dir)
        if os.path.dirname(resolved) != os.path.join(common, "worktrees"):
            return False
        if not stat.S_ISDIR(os.lstat(resolved).st_mode):
            return False
        return _linked_metadata_problem(
            dotgit, resolved, common, resolve=True) is None
    except (OSError, ValueError):
        # ValueError: a pointer that is not text. Not a worktree: refused.
        return False


def _refuse_nested_in_git_workdir(repo_root, common_dir, dotgit_st, git_dir):
    """Refuse write mode when another Git repository lives under a Git workdir.

    The protected-branch denies name the workdir's own repository only. Every
    other repository under the work tree (an untracked or ignored clone, a
    bare repository) keeps its refs where the child can write them, and
    submodule git dirs live under ``<common dir>/modules`` (or, for a submodule
    added inside a linked worktree, ``<git dir>/modules``), which is writable.
    Exempt: the repository's own top-level ``.git`` entry, and a ``.git``
    pointer file that is a linked worktree of this same repository (the walk
    goes on inside it). Anything else, or a scan that cannot finish, refuses.
    """
    def exempt(rel, entry):
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            return False
        if rel == entry.name and os.path.samestat(st, dotgit_st):
            return True
        return stat.S_ISREG(st.st_mode) and _is_own_linked_worktree(
            os.path.join(repo_root, rel), common_dir)

    kind, found = _nested_repo_hit(repo_root, exempt)
    if kind == "incomplete":
        _refuse_unverified_scan(found)
    if kind is not None:
        shown = ascii(redact_secrets(found))[:200]
        _refuse("fence_path_rejected",
                "write mode refuses a Git workdir that holds another Git "
                "repository (%s at %s): its branches are not protected"
                % (".git entry" if kind == "repo" else "bare repository", shown))
    # The workdir's own Git directory differs from the common one when the
    # workdir is a linked worktree; ``git submodule add`` run there keeps the
    # submodule git dir under ``<git dir>/modules``.
    scanned = [(os.path.join(common_dir, "modules"), "modules/")]
    if git_dir != common_dir:
        scanned.append((os.path.join(git_dir, "modules"),
                        "modules/ of the worktree Git directory"))
    for modules, label in scanned:
        try:
            populated = False
            try:
                modules_st = os.lstat(modules)
            except (FileNotFoundError, NotADirectoryError):
                modules_st = None   # absent; any other failed lookup fails closed
            if modules_st is not None:
                if stat.S_ISLNK(modules_st.st_mode) or not stat.S_ISDIR(modules_st.st_mode):
                    populated = True
                else:
                    with os.scandir(modules) as scan:
                        populated = any(True for _ in scan)
        except OSError:
            _refuse("fence_path_rejected",
                    "write mode could not verify that the repository keeps no "
                    "submodule Git directories - fence fails closed")
        if populated:
            _refuse("fence_path_rejected",
                    "write mode refuses a Git workdir whose repository keeps "
                    "submodule Git directories (%s): their branches are not "
                    "protected" % label)


def _discovery_failed(cwd, required):
    """Git could not name the repository around *cwd*: the empty admission.

    Ordinary write mode keeps the workdir writable, so a workdir that is, or
    sits inside, a Git directory must not pass as "not a repository" just
    because discovery failed (a Git too old to open a repository that declares
    ``extensions.refStorage``; a workdir inside ``.git`` itself). That is
    judged by name: ``<workdir>/.git`` exists, or a component of the canonical
    workdir is ``.git``. Anything else is a genuine non-git directory.
    """
    if not required:
        named = any(unicodedata.normalize("NFC", part).casefold() == ".git"
                    for part in cwd.split(os.sep))
        if named or os.path.lexists(os.path.join(cwd, ".git")):
            _refuse("fence_path_rejected",
                    "write mode cannot resolve the Git repository around the workdir")
        _refuse_nested_repo(cwd)
    return (), (), (), (), ()


def _repo_write_roots(workdir, *, required=False, allow_main=False):
    """Resolve the selected repository and its actual Git metadata roots.

    This is a host capability, not a caller-supplied path list. Git
    environment overrides are removed before resolving roots. When
    *required* is false, a non-git workdir yields no extra roots instead
    of refusing. Protected branches (main / master / origin default) are
    not admitted unless *allow_main*.
    """
    cwd = os.fspath(workdir)
    if not os.path.isabs(cwd) or os.path.realpath(cwd) != cwd:
        if required:
            _refuse("fence_path_rejected", "--allow-repo-write requires a canonical workdir")
        return (), (), (), (), ()
    git_env = {key: value for key, value in os.environ.items()
               if not key.startswith("GIT_")}
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    git_bin = shutil.which("git", path=os.defpath)
    if not git_bin:
        if required:
            _refuse("fence_path_rejected", "--allow-repo-write requires system Git")
        return (), (), (), (), ()
    git_bin = os.path.realpath(git_bin)

    def git_run(*args):
        try:
            return git_safe.run_git(cwd, args, git_bin=git_bin, env=git_env,
                                    timeout=None)
        except (OSError, ValueError, git_safe.GitSafeError) as exc:
            # ValueError covers UnicodeDecodeError: a ref name a child wrote
            # that is not text must refuse, not escape as a raw exception.
            _refuse("fence_path_rejected", "cannot resolve Git repository: %s" % redact_secrets(str(exc)))

    def git_path(option):
        proc = git_run("rev-parse", option)
        if proc.returncode:
            if required:
                _refuse("fence_path_rejected", "--allow-repo-write requires a Git worktree")
            return None
        value = proc.stdout.strip()
        if not value:
            if required:
                _refuse("fence_path_rejected", "Git returned an empty repository path")
            return None
        if not os.path.isabs(value):
            value = os.path.join(cwd, value)
        resolved = os.path.realpath(value)
        if not os.path.isdir(resolved) or resolved != value:
            _refuse("fence_path_rejected", "Git repository root is not canonical")
        return resolved

    repo_root = git_path("--show-toplevel")
    if repo_root is None:
        if not required and git_run("rev-parse", "--git-dir").returncode == 0:
            # A bare repository, or a Git directory kept apart from its tree
            # (``--separate-git-dir``), used as the workdir: Git knows the
            # directory but names no work tree. The workdir itself is
            # writable in ordinary write mode, so its refs would be too.
            _refuse("fence_path_rejected",
                    "write mode cannot use a Git directory as the workdir")
        return _discovery_failed(cwd, required)
    git_dir = git_path("--absolute-git-dir")
    common_dir = git_path("--git-common-dir")
    if git_dir is None or common_dir is None:
        return _discovery_failed(cwd, required)
    ref_format = _ref_storage_format(git_run)
    if ref_format != "files":
        _refuse("fence_path_rejected", _ref_storage_refusal(
            ref_format, "--allow-repo-write" if required else "write mode"))
    # Full ref names, never ``--short``: git prints the shortest UNAMBIGUOUS
    # name, so a tag ``refs/tags/main`` turns HEAD on ``refs/heads/main`` into
    # ``heads/main``, and a fenced child can create one.
    branch_proc = git_run("symbolic-ref", "-q", "HEAD")
    head_ref = ref_line(branch_proc.stdout)
    # The branch keeps its original spelling (it ends up in refusal text, where
    # the mask is case-sensitive); every comparison below folds it itself.
    branch = head_ref[len("refs/heads/"):] if head_ref.startswith(
        "refs/heads/") else ""
    if not (branch_proc.returncode == 0 and branch) and not (
            branch_proc.returncode == 1 and not head_ref):
        _refuse("fence_path_rejected", "current branch could not be determined")
    try:
        root_st = os.lstat(repo_root)
        git_st = os.lstat(git_dir)
        common_st = os.lstat(common_dir)
        dotgit_st = os.lstat(os.path.join(repo_root, ".git"))
    except OSError:
        _refuse("fence_path_rejected", "Git repository identity is incomplete")
    if any(stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode)
           for st in (root_st, git_st, common_st)):
        _refuse("fence_path_rejected", "Git repository roots must be real directories")
    dotgit = os.path.join(repo_root, ".git")
    if stat.S_ISLNK(dotgit_st.st_mode) or not (
            stat.S_ISDIR(dotgit_st.st_mode) or stat.S_ISREG(dotgit_st.st_mode)):
        _refuse("fence_path_rejected", "worktree .git entry must be a directory or regular pointer")
    if stat.S_ISREG(dotgit_st.st_mode):
        pointed, problem = _linked_pointer(dotgit)
        if problem:
            _refuse("fence_path_rejected", problem)
        if pointed != git_dir or os.path.realpath(pointed) != git_dir:
            _refuse("fence_path_rejected", "worktree .git pointer changed")
        problem = _linked_metadata_problem(dotgit, git_dir, common_dir)
        if problem:
            _refuse("fence_path_rejected", problem)
    elif git_dir != common_dir or os.path.realpath(dotgit) != common_dir:
        _refuse("fence_path_rejected", "main Git metadata directory changed")
    _refuse_nested_in_git_workdir(repo_root, common_dir, dotgit_st, git_dir)

    def origin_probe():
        proc = git_run("symbolic-ref", "-q", "refs/remotes/origin/HEAD")
        return proc.returncode, proc.stdout

    # Aliases the path denies cannot see through: a symlink under refs/, and a
    # symlink in place of any git metadata path the denies below name.
    alias = (ref_alias_refusal(common_dir)
             or metadata_alias_refusal(git_dir, common_dir, repo_root))
    if alias:
        _refuse("fence_path_rejected", alias)

    # Judged only now, after the identity checks above: a first sight records
    # the names for good, so it must not happen for a layout refused here.
    try:
        names, why = protected_branches(common_dir, origin_probe)
    except ProtectedListError as exc:
        _refuse("fence_path_rejected", exc.detail)
    if why:
        _refuse("fence_path_rejected", why)
    protected = {name.lower() for name in names}
    symref = protected_symref_refusal(git_run, names) or loose_symref_refusal(
        common_dir, names)
    if symref:
        _refuse("fence_path_rejected", symref)
    on_protected = bool(branch) and (
        kernel_sandbox.branch_name_in(branch, protected) and not allow_main)
    denies = _repo_protect_denies(git_dir, common_dir, protected, repo_root)
    # Neither the repository nor anything above it can be renamed away: the
    # worktree root, the main checkout's root and the git directories.
    root_paths = [repo_root, git_dir, common_dir]
    if os.path.basename(common_dir) == ".git":
        root_paths.append(os.path.dirname(common_dir))
    root_denies = ancestor_literal_denies(root_paths)
    # The alias scan of the repository tree is not a privilege of the branch
    # the workdir happens to be on: every write-mode return passes it.
    _reject_hardlinked_files([types.SimpleNamespace(path=path)
                              for path in dict.fromkeys((repo_root, git_dir, common_dir))])
    if on_protected:
        if required:
            _refuse("fence_path_rejected",
                    "--allow-repo-write refuses protected branch %s" % branch)
        return (), denies, (git_dir, common_dir), (), root_denies

    roots = []
    for path in (repo_root, git_dir, common_dir):
        if path not in roots:
            try:
                roots.append(kernel_sandbox.admit_path(
                    path, "subpath", label="repository write root"))
            except kernel_sandbox.FencePathRejected as exc:
                _refuse("fence_path_rejected", exc.detail)
    post_writes = _linked_worktree_post_writes(git_dir, common_dir)
    final_denies = _linked_worktree_final_denies(git_dir, common_dir, protected)
    return (tuple(roots), denies, (git_dir, common_dir), post_writes,
            tuple(final_denies) + root_denies)


_ORIGIN_HEAD = ("refs", "remotes", "origin", "HEAD")
_ORIGIN_PREFIX = "refs/remotes/origin/"


def ref_line(out):
    """The one line ``git symbolic-ref`` printed, minus Git's own newline.

    Nothing else is stripped: a ref name may end in other whitespace (a
    no-break space is a valid branch name), and a stripped name would name a
    different ref than the one to protect.
    """
    out = out or ""
    return out[:-1] if out.endswith("\n") else out


def origin_default_branch(rc, out, common_dir):
    """``(name, None)``, ``(None, None)`` or ``(None, reason)`` for origin/HEAD.

    "No default branch" means ``symbolic-ref -q`` exited 1 AND
    ``refs/remotes/origin/HEAD`` is genuinely absent (no loose file; a
    symbolic ref cannot be packed). Any other exit status (128: a regular
    file, a loop, a broken ref) and anything that is not a symbolic ref into
    ``refs/remotes/origin/`` is indeterminate and refuses: a missing
    protected name would leave the real default branch unprotected.

    Only the first sight of a repository asks (see ``protected_branches``);
    after that the recorded names answer.
    """
    if rc == 0:
        line = ref_line(out)
        if line.startswith(_ORIGIN_PREFIX) and len(line) > len(_ORIGIN_PREFIX):
            return line[len(_ORIGIN_PREFIX):], None
        return None, "origin default branch ref has an unexpected target"
    if rc == 1 and not os.path.lexists(os.path.join(common_dir, *_ORIGIN_HEAD)):
        return None, None
    return None, "origin default branch could not be determined"


ENV_PROTECTED_BRANCHES = "OMNI_LLM_PROTECTED_BRANCHES"
_RECORD_SUBDIR = "protected-refs"
_RECORD_VERSION = 1
_RECORD_MAX_BYTES = 65536
_LEDGER_NAME = "ledger"
_LEDGER_MAX_BYTES = 1 << 20
_KEY_RE = re.compile(r"[0-9a-f]{32}")
_FLOOR_BRANCHES = ("main", "master")
_BAD_BRANCH_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")


class ProtectedListError(RuntimeError):
    """The protected-branch list cannot be established; callers refuse the run."""

    def __init__(self, detail):
        # host-exec prints this verbatim, and it can quote an operator value.
        self.detail = redact_secrets(detail)
        super().__init__(self.detail)


def protected_record_dir():
    """Where the protected-branch records live: under omni's state dir.

    Nothing is created here. The fence denies the child every write below it
    and every rename of an ancestor (``record_dir_denies``), in every fenced
    profile, including when the state dir sits in a writable root.
    """
    state = os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR))
    return os.path.join(state, _RECORD_SUBDIR)


def record_dir_denies():
    """Final deny-write rows for the record dir and every ancestor of it.

    The record dir is a ``subpath`` deny. Each ancestor is a ``literal`` deny:
    that blocks renaming or removing the directory (measured, see
    ``_repo_protect_denies``) while creating files in it stays possible, so a
    writable root above the state dir cannot be used to move the whole tree
    away and let omni enroll the repository afresh. Both the lexical and the
    resolved spelling are denied. No fenced profile is exempt.
    """
    base = protected_record_dir()
    rows = list(_with_plain_spelling(kernel_sandbox._deny_entries("subpath", base)))
    rows.extend(ancestor_literal_denies(_path_hops(base)))
    return tuple(dict.fromkeys(rows))


def _path_hops(path, limit=40):
    """Every path visited when *path* is resolved one component at a time.

    A manual walk, so each symlink on the way (and the path it leads to) is
    seen, not just the end of the chain that ``realpath`` returns: a rename of
    any hop re-points the state dir. Raises ``ProtectedListError`` when a link
    cannot be read or the chain is too long (a loop): refused, not guessed.
    """
    visited = []
    queue = [part for part in os.path.abspath(path).split(os.sep) if part]
    current = os.sep
    links = 0
    while queue:
        part = queue.pop(0)
        if part == ".":
            continue
        if part == "..":
            current = os.path.dirname(current)
            continue
        candidate = os.path.join(current, part)
        visited.append(candidate)
        try:
            is_link = stat.S_ISLNK(os.lstat(candidate).st_mode)
        except FileNotFoundError:
            current = candidate
            continue
        except OSError as exc:
            raise ProtectedListError(
                "the state dir path cannot be inspected at %s (%s)" % (
                    candidate, exc.strerror or type(exc).__name__))
        if not is_link:
            current = candidate
            continue
        links += 1
        if links > limit:
            raise ProtectedListError(
                "the state dir path %s passes through too many symlinks" % path)
        try:
            target = os.readlink(candidate)
        except OSError as exc:
            raise ProtectedListError(
                "the state dir symlink %s cannot be read (%s)" % (
                    candidate, exc.strerror or type(exc).__name__))
        if os.path.isabs(target):
            current = os.sep
        queue = [seg for seg in target.split(os.sep) if seg] + queue
    visited.append(current)
    return tuple(dict.fromkeys(visited))


def ancestor_literal_denies(paths, *, include_self=True):
    """Literal deny rows for each of *paths* and every directory above it.

    A literal deny blocks renaming or removing a directory and nothing else
    (measured), so naming a tree and all of its ancestors makes the tree
    impossible to move away. Both the lexical and the resolved spelling, each
    with its ``/System/Volumes/Data`` plain spelling.
    """
    rows = []
    seen = set()
    for base in paths:
        for spelling in dict.fromkeys((base, os.path.realpath(base))):
            current = spelling if include_self else os.path.dirname(spelling)
            while current and current != os.sep and current not in seen:
                seen.add(current)
                rows.extend(_with_plain_spelling(
                    kernel_sandbox._deny_entries("literal", current)))
                current = os.path.dirname(current)
    return tuple(dict.fromkeys(rows))


_FIRMLINK_PREFIX = "/System/Volumes/Data"


def _plain_spelling(path):
    """*path* without a leading ``/System/Volumes/Data``, or ``None``.

    Seatbelt compares the kernel-resolved path of an access, which is the plain
    spelling (``/Users/...``, ``/private/...``): a rule spelled through the
    firmlink never matches (test_same_location). The prefix is matched case
    insensitively (the volume is) and the two spellings must be the same
    object. A path that does not exist yet cannot be compared: it still gets
    the plain row, because the extra deny can only name the same location.
    """
    head = path[:len(_FIRMLINK_PREFIX)]
    if head.lower() != _FIRMLINK_PREFIX.lower() or path[len(head):len(head) + 1] not in ("/", ""):
        return None
    plain = path[len(head):] or os.sep
    if plain == os.sep:
        return None
    try:
        if not os.path.samestat(os.stat(path), os.stat(plain)):
            return None
    except OSError:
        pass
    return plain


def _with_plain_spelling(rows):
    out = []
    for kind, path in rows:
        out.append((kind, path))
        plain = _plain_spelling(path)
        if plain is not None:
            out.append((kind, plain))
    return out


def _valid_branch_name(name):
    return bool(name) and not _BAD_BRANCH_CHARS.search(name) and not (
        name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock"))
        or ".." in name or "//" in name or "@{" in name)


def _with_floor(names):
    """main / master first, then *names* as spelled (an exact duplicate only drops)."""
    out = list(_FLOOR_BRANCHES)
    for name in names:
        if name not in out:
            out.append(name)
    return tuple(out)


def _override_names():
    """``None`` when the operator did not name a list, else the validated names."""
    raw = os.environ.get(ENV_PROTECTED_BRANCHES)
    if raw is None or not raw.strip():
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        return None     # a list that strips to nothing is "unset"
    for name in names:
        if not _valid_branch_name(name):
            # Masked BEFORE the cut: the value is operator input, and the cut
            # could otherwise leave a masked-pattern prefix readable.
            raise ProtectedListError(
                "%s names an invalid branch %r" % (
                    ENV_PROTECTED_BRANCHES, redact_secrets(name)[:80]))
    return names


def _record_path(identity):
    """The record's file: keyed by the common dir's device and inode.

    Not by its path: two spellings of one directory (``/System/Volumes/Data``
    and plain) are one repository, a moved repository is the same repository
    in another place (refused, see ``_read_record``), and a new repository at
    an old path is a new one.
    """
    digest = hashlib.sha256(
        ("%d:%d" % (identity.st_dev, identity.st_ino)).encode("ascii")).hexdigest()[:32]
    return os.path.join(protected_record_dir(), digest + ".json")


def _legacy_record_paths(common_dir):
    """Where an earlier, path-keyed version of this code kept the record."""
    return [os.path.join(
        protected_record_dir(),
        hashlib.sha256(os.fsencode(spelling)).hexdigest()[:32] + ".json")
        for spelling in dict.fromkeys((common_dir, _plain_spelling(common_dir) or common_dir))]


def _record_hint(path):
    return ("check the repository, then reset it explicitly: delete the record "
            "%s AND its line in %s (or the whole directory)" % (path, _ledger_path()))


def _ledger_path():
    return os.path.join(protected_record_dir(), _LEDGER_NAME)


def _record_key(path):
    return os.path.basename(path)[:-len(".json")]


def _read_ledger():
    """The set of enrolled record keys, ``None`` when there is no ledger file.

    The ledger lists every repository omni ever enrolled, so a record that
    went missing is told from a repository seen for the first time. One JSON
    object per line, ``{"key": ..., "common_dir": ...}``: JSON escapes every
    CR / LF / U+2028 in a path, and lines are split on ``\\n`` only, so no
    path can break the file that every repository shares.
    """
    path = _ledger_path()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProtectedListError(
            "enrollment ledger %s cannot be opened (%s); %s" % (
                path, exc.strerror or type(exc).__name__, _reset_hint()))
    try:
        with os.fdopen(fd, "rb") as stream:
            st = os.fstat(stream.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > _LEDGER_MAX_BYTES:
                raise ValueError("not a regular file of sane size")
            text = stream.read(_LEDGER_MAX_BYTES + 1).decode("utf-8")
    except (OSError, ValueError) as exc:
        raise ProtectedListError(
            "enrollment ledger %s is unreadable (%s); %s" % (
                path, type(exc).__name__, _reset_hint()))
    keys = set()
    for line in text.split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            key = entry["key"]
            ok = isinstance(entry, dict) and isinstance(key, str) and _KEY_RE.fullmatch(key)
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            raise ProtectedListError(
                "enrollment ledger %s is malformed; %s" % (path, _reset_hint()))
        keys.add(key)
    return keys


def _reset_hint():
    return ("check the repository, then reset explicitly: delete the record and "
            "its line in the ledger (or the whole directory %s)" % protected_record_dir())


def _enroll(key, common_dir):
    path = _ledger_path()
    line = (json.dumps({"key": key, "common_dir": common_dir}, sort_keys=True)
            + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _record_lock():
    """Serialise first sights; best effort (a lock that cannot be opened is skipped)."""
    fd = None
    try:
        os.makedirs(protected_record_dir(), mode=0o700, exist_ok=True)
        fd = os.open(os.path.join(protected_record_dir(), ".lock"),
                     os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        if fd is not None:
            os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)


def _read_record(path, common_dir, identity):
    """The recorded names, ``None`` when there is no record yet.

    A record that exists must be readable, well formed and bound to THIS
    common dir (path, device, inode); anything else raises. It is never
    rebuilt from the refs: they are what a child could have rewritten.
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProtectedListError(
            "protected-branch record %s cannot be opened (%s); %s" % (
                path, exc.strerror or type(exc).__name__, _record_hint(path)))
    try:
        with os.fdopen(fd, "rb") as stream:
            st = os.fstat(stream.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > _RECORD_MAX_BYTES:
                raise ValueError("not a regular file of sane size")
            data = json.loads(stream.read(_RECORD_MAX_BYTES + 1).decode("utf-8"))
        names = data["protected"]
        if (not isinstance(data, dict) or data["version"] != _RECORD_VERSION
                or not isinstance(data["common_dir"], str)
                or type(data["dev"]) is not int or type(data["ino"]) is not int
                or not isinstance(names, list) or not names
                or not all(isinstance(n, str) and _valid_branch_name(n) for n in names)):
            raise ValueError("unexpected shape")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProtectedListError(
            "protected-branch record %s is unreadable (%s); %s" % (
                path, type(exc).__name__, _record_hint(path)))
    if (data["dev"], data["ino"]) != (identity.st_dev, identity.st_ino):
        raise ProtectedListError(
            "protected-branch record %s belongs to another repository identity "
            "(device or inode of %s differ); %s" % (path, common_dir, _record_hint(path)))
    # The stored path must still be this very directory (device and inode, not
    # the spelling: a re-cased or firmlink-spelled path is the same directory);
    # a repository that was moved lands here, and so does a stored path that
    # cannot be inspected.
    try:
        stored = os.stat(data["common_dir"])
    except OSError as exc:
        raise ProtectedListError(
            "protected-branch record %s names %s, which cannot be inspected (%s); "
            "the repository may have been moved; %s" % (
                path, data["common_dir"], exc.strerror or type(exc).__name__,
                _record_hint(path)))
    if (stored.st_dev, stored.st_ino) != (data["dev"], data["ino"]):
        raise ProtectedListError(
            "protected-branch record %s names %s, which is another directory now "
            "(the repository was moved); %s" % (path, data["common_dir"], _record_hint(path)))
    return tuple(names)


def _write_record(path, common_dir, identity, names):
    """Create the record; ``False`` when someone recorded this repository first."""
    directory = os.path.dirname(path)
    body = json.dumps({
        "version": _RECORD_VERSION, "common_dir": _plain_spelling(common_dir) or common_dir,
        "dev": identity.st_dev, "ino": identity.st_ino,
        "protected": sorted(names),
    }, sort_keys=True).encode("utf-8")
    tmp = None
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".record-", dir=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)       # never replaces an existing record
        except FileExistsError:
            return False
        return True
    except OSError as exc:
        raise ProtectedListError(
            "cannot record the protected branches of %s under %s (%s); "
            "fix the state dir (%s); %s" % (
                common_dir, directory, exc.strerror or type(exc).__name__,
                ENV_STATE_DIR, _reset_hint()))
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def protected_branches(common_dir, origin_probe, *, record=True):
    """``(names, None)`` or ``(None, reason)``: the local branches to protect.

    The list is NOT derived from refs a fenced child can rewrite. In order:

    * the operator's ``OMNI_LLM_PROTECTED_BRANCHES`` (comma separated; main and
      master always stay), when set: it is the answer and no refs are read for
      it. A write run still enrolls an unseen repository (record + ledger line:
      the derived names plus the override, or main / master plus the override
      when origin/HEAD is indeterminate), so the record answers once the
      override is gone. A record or ledger that cannot be read or written
      refuses even then;
    * the record under omni's state dir for this common dir, keyed by its
      device/inode (never its path) and bound to its plain-spelled path.
      Made the first time omni sees the repository,
      from ``main``, ``master`` and the origin default branch as
      ``origin_probe() -> (rc, stdout)`` resolves it right then (indeterminate
      origin/HEAD -> ``(None, reason)``, nothing recorded). A record that
      exists but is unreadable or belongs to another identity raises
      ``ProtectedListError`` and is never rebuilt;
    * ``record=False`` (a read-only run, which cannot change a ref): the old
      derivation from the refs, nothing written.

    *common_dir* must be the canonical (realpath) common directory.
    """
    override = _override_names()

    def derive():
        name, why = origin_default_branch(*origin_probe(), common_dir)
        if why:
            return None, why
        if name and not _valid_branch_name(name):
            return None, "origin default branch has an unsupported name"
        return _with_floor([name] if name else []), None

    def recorded(extra=()):
        try:
            identity = os.stat(common_dir)
        except OSError as exc:
            raise ProtectedListError(
                "cannot stat the Git common directory (%s)" % (
                    exc.strerror or type(exc).__name__))
        path = _record_path(identity)
        key = _record_key(path)
        with _record_lock():
            names = _read_record(path, common_dir, identity)
            enrolled = _read_ledger()
            if names is not None:
                if enrolled is None or key not in enrolled:
                    raise ProtectedListError(
                        "protected-branch record %s has no enrollment ledger entry "
                        "(ledger %s: %s); %s" % (
                            path, _ledger_path(),
                            "missing" if enrolled is None else "the repository is not listed",
                            _reset_hint()))
                return _with_floor(names), None
            for legacy in _legacy_record_paths(common_dir):
                if os.path.lexists(legacy):
                    raise ProtectedListError(
                        "protected-branch record %s is a legacy path-keyed record; it "
                        "is neither read nor replaced from the refs; %s" % (
                            legacy, _record_hint(legacy)))
            if enrolled is not None and key in enrolled:
                raise ProtectedListError(
                    "protected-branch record %s is missing although the repository is "
                    "enrolled in the ledger; it is not rebuilt from the refs; %s" % (
                        path, _record_hint(path)))
            if enrolled is None and _any_record():
                raise ProtectedListError(
                    "enrollment ledger %s is missing while protected-branch records "
                    "exist; %s" % (_ledger_path(), _reset_hint()))
            # True first sight: nothing was ever recorded for this repository.
            names, why = derive()
            if why:
                if not extra:
                    return None, why
                # An operator override names the list; what origin/HEAD could not
                # say is replaced by it, and the record is made all the same.
                names = _with_floor(list(extra))
            elif extra:
                names = _with_floor(list(names) + list(extra))
            try:
                _enroll(key, common_dir)
            except OSError as exc:
                raise ProtectedListError(
                    "cannot enroll %s in the ledger %s (%s); fix the state dir (%s); %s" % (
                        common_dir, _ledger_path(), exc.strerror or type(exc).__name__,
                        ENV_STATE_DIR, _reset_hint()))
            if not _write_record(path, common_dir, identity, names):
                raise ProtectedListError(
                    "protected-branch record %s appeared while it was being made; %s" % (
                        path, _record_hint(path)))
            return names, None

    if override is not None:
        if record:
            # The override changes the ANSWER, not the record: every writable
            # first contact leaves one (derived names plus the override, or
            # main / master plus the override when origin/HEAD cannot be read),
            # so that once the override is gone the record answers and the refs
            # do not. A record or ledger that cannot be read or written REFUSES
            # (ProtectedListError propagates): the documented reset is the way
            # out, the override is not. An existing record is never replaced.
            recorded(override)
        return _with_floor(override), None
    if not record:
        return derive()
    return recorded()


def _any_record():
    """Whether any record exists. A directory that cannot be listed REFUSES:
    reading it as "no records" would enroll a repository that may have one."""
    try:
        return any(name.endswith(".json") for name in os.listdir(protected_record_dir()))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProtectedListError(
            "the record directory %s cannot be listed (%s); %s" % (
                protected_record_dir(), exc.strerror or type(exc).__name__, _reset_hint()))


def ref_is_watched(rest, names):
    """*rest* (a name under refs/heads) is a protected name, or lies below one.

    Compared the way admission compares (case, NFC/NFD folded). A ref BELOW a
    protected name is a directory/file clash that can shadow it.
    """
    parts = rest.split("/")
    return any(kernel_sandbox.branch_name_in("/".join(parts[:end]), names)
               for end in range(1, len(parts) + 1))


def protected_symref_refusal(git_run, names):
    """A reason when a protected LOCAL name is a symbolic ref, else ``None``.

    ``refs/heads/<name>`` as a symbolic ref (a chain, or a dangling target
    too) lets the real target be reached under another name, so what a
    protected name stands for is no longer the file the denies name. Every
    ref under refs/heads is listed and the protected names are matched the way
    admission matches them (case, NFC/NFD), so ``MaIn`` counts as ``main``;
    the spelling as recorded and its lowercase are also asked directly, which
    catches a dangling target. ``refs/remotes/origin/HEAD`` is a symbolic ref
    by design and is not looked at. The whole text is masked: a branch name
    can look like a credential. *git_run* is ``git_run(*args) ->
    CompletedProcess``.
    """
    def refusal(text):
        return redact_secrets(text)

    proc = git_run("for-each-ref", "--format=%(refname)%00%(symref)")
    if proc.returncode != 0:
        return refusal("protected branches cannot be listed (exit %s)" % proc.returncode)
    for line in (proc.stdout or "").split("\n"):
        if not line:
            continue
        fields = line.split("\0")
        if len(fields) != 2:
            return refusal("protected branches cannot be listed (unexpected line)")
        ref, symref = fields
        if (ref.startswith("refs/heads/") and symref
                and ref_is_watched(ref[len("refs/heads/"):], names)):
            return refusal("protected branch %s is a symbolic ref (%s)" % (ref, symref))
    for name in names:
        for spelling in dict.fromkeys((name, name.lower())):
            ref = "refs/heads/" + spelling
            shown = "refs/heads/" + name        # as recorded, never the folded spelling
            proc = git_run("symbolic-ref", "-q", ref)
            if proc.returncode == 0:
                return refusal("protected branch %s is a symbolic ref (%s)" % (
                    shown, ref_line(proc.stdout)))
            if proc.returncode != 1:
                return refusal("protected branch %s cannot be inspected (exit %s)" % (
                    shown, proc.returncode))
    return None


def metadata_alias_refusal(git_dir, common_dir, repo_root=None):
    """A reason when a git metadata path the protect denies name is a symlink.

    Seatbelt judges the resolved target of an access, so a literal deny on a
    link binds nothing: ``packed-refs``, ``HEAD``, ``config``, ``hooks``, the
    ``refs`` directory, ``worktrees`` and the identity files (``commondir``,
    ``gitdir``) of each git directory, and the repository's ``.git``. A path
    that cannot be inspected counts as refused.
    """
    leaves = ("hooks", "config", "packed-refs", "HEAD", "commondir", "gitdir",
              "refs", "worktrees")
    paths = []
    for meta in dict.fromkeys((git_dir, common_dir)):
        paths.append(meta)
        paths.extend(os.path.join(meta, leaf) for leaf in leaves)
    if repo_root:
        paths.append(os.path.join(repo_root, ".git"))
    for path in dict.fromkeys(paths):
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                return "git metadata path %s is a symlink" % path
        except FileNotFoundError:
            continue
        except OSError as exc:
            return "git metadata path %s cannot be inspected (%s)" % (
                path, type(exc).__name__)
    return None


def loose_symref_refusal(common_dir, names):
    """A reason when a loose ref file under a watched name is a symbolic ref.

    ``for-each-ref`` leaves out a dangling symbolic ref and the exact-name
    probe asks for one spelling, so the loose files under ``refs/heads`` whose
    name folds to a protected name (case, NFC/NFD, or lying below one) are
    read: ``ref:`` as the first bytes is a symbolic ref whatever it points
    at. A tree that cannot be read counts as refused.
    """
    heads = os.path.join(common_dir, "refs", "heads")
    failed = []
    for current, _dirs, files in (os.walk(heads, followlinks=False, onerror=failed.append)
                                  if os.path.isdir(heads) else ()):
        for name in files:
            path = os.path.join(current, name)
            rest = os.path.relpath(path, heads).replace(os.sep, "/")
            if name.endswith(".lock") or not ref_is_watched(rest, names):
                continue
            try:
                with open(path, "rb") as stream:
                    head = stream.read(4)
            except OSError as exc:
                return redact_secrets("loose ref %s cannot be read (%s)" % (
                    rest, type(exc).__name__))
            if head == b"ref:":
                return redact_secrets(
                    "protected branch refs/heads/%s is a symbolic ref (loose ref file)" % rest)
    if failed:
        return "refs/heads cannot be walked (%s)" % type(failed[0]).__name__
    return None


def ref_alias_refusal(common_dir):
    """A reason when anything under ``<common dir>/refs`` is a symlink, else ``None``.

    A symlinked ref file or directory is a filesystem alias the path denies do
    not see through. Nothing is followed; a tree that cannot be inspected
    counts as refused.
    """
    refs = os.path.join(common_dir, "refs")
    failed = []
    try:
        if stat.S_ISLNK(os.lstat(refs).st_mode):
            return "%s is a symlink" % refs
    except FileNotFoundError:
        return None
    except OSError as exc:
        return "%s cannot be inspected (%s)" % (refs, type(exc).__name__)
    for current, dirs, files in os.walk(refs, followlinks=False, onerror=failed.append):
        for name in dirs + files:
            try:
                linked = stat.S_ISLNK(os.lstat(os.path.join(current, name)).st_mode)
            except OSError as exc:
                return "%s cannot be inspected (%s)" % (
                    os.path.join(current, name), type(exc).__name__)
            if linked:
                return "refs entry %s is a symlink" % os.path.join(current, name)
    if failed:
        return "%s cannot be walked (%s)" % (refs, type(failed[0]).__name__)
    return None


def _ref_name_cases(name):
    n = str(name).strip("/")
    if not n:
        return ()
    title = n[0].upper() + n[1:].lower() if len(n) > 1 else n.upper()
    return tuple(dict.fromkeys((n, n.lower(), n.upper(), title)))


def _repo_protect_denies(git_dir, common_dir, protected, repo_root):
    """W* denies that must win last-match inside admitted Git metadata."""
    denies = []
    for path in dict.fromkeys((os.path.join(repo_root, ".git"), git_dir, common_dir)):
        denies.append(("literal", path))
    names = []
    for name in protected:
        names.extend(_ref_name_cases(name))
    seen = []
    for meta in (git_dir, common_dir):
        if meta in seen:
            continue
        seen.append(meta)
        denies.append(("subpath", os.path.join(meta, "hooks")))
        # ``commondir`` / ``gitdir`` say which repository a verification reads:
        # a child that could rewrite them (or plant ``commondir`` in the main
        # git dir) would point every later read at a decoy.
        for leaf in ("config", "packed-refs", "HEAD", "commondir", "gitdir"):
            denies.append(("literal", os.path.join(meta, leaf)))
        # ``refs/remotes/origin/HEAD`` is deliberately NOT denied: the protected
        # names come from the record, not from it, so retargeting it protects
        # nothing less, and ``git remote set-head`` keeps working in the fence.
        # A literal deny on a directory blocks renaming or removing it while
        # creating files and subdirectories inside still works (measured with
        # sandbox-exec). Renaming ``refs/heads`` would move every protected
        # branch ref out from under its deny, so every ancestor directory of a
        # protected ref path is denied too (``refs/remotes`` stays here only
        # as depth). The generic directories are denied only while they exist:
        # the same deny on a missing one would also stop git from creating it
        # (the first ``refs/remotes/origin/x`` of a fetch).
        ref_dirs = [("refs",), ("refs", "heads"), ("refs", "remotes"),
                    ("refs", "remotes", "origin")]
        # The directories above a SLASH-named protected ref are denied even
        # when they do not exist (``release`` for ``release/v1`` kept only in
        # packed-refs): otherwise a directory holding a loose ``v1`` can be
        # created, or moved in whole, and shadows the packed value. A literal
        # deny on a missing path stops both mkdir and rename-into-place
        # (measured, test_protected_refs.DirectoryMoveHole). The price: no
        # other branch can be made below such a directory inside the fence.
        # ``refs`` and ``refs/heads`` of the repository are in that set too: a
        # directory staged outside and moved in as ``refs/heads`` would replace
        # every packed protected ref with its own loose ones. (``refs/remotes*``
        # stay conditional: a fetch has to be able to create them.)
        always = [("refs",), ("refs", "heads")]
        for name in names:
            parts = tuple(name.split("/"))[:-1]
            for end in range(1, len(parts) + 1):
                always.append(("refs", "heads") + parts[:end])
        for parts in dict.fromkeys(ref_dirs + always):
            path = os.path.join(meta, *parts)
            if parts in always or os.path.lexists(path):
                denies.append(("literal", path))
        for name in names:
            denies.append(("literal", os.path.join(meta, "refs", "heads", name)))
    denies.append(("subpath", os.path.join(common_dir, "worktrees")))
    return tuple(denies)


def _linked_worktree_post_writes(git_dir, common_dir):
    marker = os.path.join(common_dir, "worktrees") + os.sep
    if not git_dir.startswith(marker):
        return ()
    return (("subpath", git_dir),)


def _linked_worktree_final_denies(git_dir, common_dir, protected):
    marker = os.path.join(common_dir, "worktrees") + os.sep
    if not git_dir.startswith(marker):
        return ()
    denies = [("literal", git_dir), ("subpath", os.path.join(git_dir, "hooks"))]
    for leaf in ("config", "packed-refs", "HEAD", "commondir", "gitdir"):
        denies.append(("literal", os.path.join(git_dir, leaf)))
    for name in protected:
        for case in _ref_name_cases(name):
            denies.append(("literal", os.path.join(git_dir, "refs", "heads", case)))
    return tuple(denies)


def _git_remote_exec_denies():
    git_bin = shutil.which("git", path=os.defpath)
    if not git_bin:
        return ()
    try:
        proc = git_safe.run_git(
            None, ("--exec-path",), git_bin=os.path.realpath(git_bin),
            env={key: value for key, value in os.environ.items()
                 if not key.startswith("GIT_")}, timeout=None)
    except (OSError, git_safe.GitSafeError) as exc:
        # Not "no helpers to deny": an unanswered query must not loosen the fence.
        _refuse("fence_path_rejected",
                "cannot resolve Git exec path: %s" % redact_secrets(str(exc)))
    exec_path = proc.stdout.strip()
    if proc.returncode != 0 or not exec_path:
        _refuse("fence_path_rejected", "cannot resolve Git exec path")
    denies = []
    for name in ("git-remote-http", "git-remote-https", "git-remote-ftp",
                 "git-remote-ftps", "git-remote-ssh", "git-remote-ext"):
        path = os.path.join(exec_path, name)
        if os.path.lexists(path):
            # Helpers are often symlinks/hardlinks to `git`. Deny the helper
            # path only; realpath(git-remote-https)==git would block commit.
            denies.append(("literal", os.path.abspath(path)))
    return tuple(denies)


def _post_write_overlaps_repo(extra_post_write, meta_dirs):
    """Refuse vendor T* re-allows that would undo Git metadata W* denies."""
    with kernel_sandbox._location_cache():
        for entry in extra_post_write or ():
            bound = kernel_sandbox._write_entry_bound(*entry)
            if bound is None:
                _refuse("fence_path_rejected", "vendor runtime write overlap cannot be verified")
            kind, path = bound
            for meta in meta_dirs:
                if (kernel_sandbox._path_inside_any_case(path, meta)
                        or kernel_sandbox._path_inside_any_case(meta, path)):
                    _refuse("fence_path_rejected", "vendor runtime write overlaps Git metadata")


def _reject_hardlinked_files(roots):
    """Fail closed if an allowed tree already aliases a regular file."""
    scan_roots = []
    for admitted in roots:
        path = admitted.path
        if any(os.path.commonpath((parent, path)) == parent for parent in scan_roots):
            continue
        scan_roots.append(path)
    for root in scan_roots:
        def onerror(_exc):
            _refuse("fence_path_rejected", "repository changed during hardlink scan")

        for current, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            dirs[:] = sorted(dirs)
            for name in list(dirs):
                path = os.path.join(current, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    _refuse("fence_path_rejected", "repository changed during hardlink scan")
                if stat.S_ISLNK(st.st_mode):
                    dirs.remove(name)
                elif not stat.S_ISDIR(st.st_mode):
                    _refuse("fence_path_rejected", "repository contains an unsupported directory entry")
            for name in files:
                path = os.path.join(current, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    _refuse("fence_path_rejected", "repository changed during hardlink scan")
                if stat.S_ISREG(st.st_mode) and st.st_nlink != 1:
                    _refuse("fence_path_rejected", "repo-write refuses pre-existing hard-linked files")


def _admit_grok_order_file(argv, provider, order_file, env):
    """Bind one placed workorder to Grok's actual trailing prompt-file option.

    The trusted team caller independently validates TEAM_INBOX against its
    repo/run. This boundary accepts no directory or arbitrary read-path option.
    """
    if provider != "grok" or list(argv[-2:]) != ["--prompt-file", order_file]:
        _refuse("fence_path_rejected", "Grok order must match trailing --prompt-file FILE")
    i = 1
    while i < len(argv) - 2:
        token = argv[i]
        if token == "--" or token == "--prompt-file" or token.startswith("--prompt-file="):
            _refuse("fence_path_rejected", "Grok order requires one active prompt-file option")
        i += 2 if token in _GROK_VALUE_OPTIONS or token == "--sandbox" else 1
    if i != len(argv) - 2:
        _refuse("fence_path_rejected", "Grok prompt-file cannot be another option's operand")
    inbox = env.get("TEAM_INBOX")
    for path, label in ((order_file, "Grok order file"), (inbox, "TEAM_INBOX")):
        if not path or not os.path.isabs(path) or os.path.realpath(path) != path:
            _refuse("fence_path_rejected", "%s must be an absolute canonical path" % label)
    try:
        root = kernel_sandbox.admit_path(inbox, "subpath", label="TEAM_INBOX")
        order = kernel_sandbox.admit_path(order_file, "literal", label="Grok order file")
    except kernel_sandbox.FencePathRejected as exc:
        _refuse("fence_path_rejected", exc.detail)
    if os.path.commonpath((order.path, root.path)) != root.path:
        _refuse("fence_path_rejected", "Grok order file must be inside TEAM_INBOX")
    return order


def _grok_instruction_rows(grok_home):
    """Every deny-read row the grok fence must carry for global instructions.

    Grok appends ``$GROK_HOME/AGENTS.md``, ``~/.claude/CLAUDE.md`` and their
    ``rules/`` directories to every session; headless lanes must not carry
    the user's interactive instructions. ``--rules`` only appends, and the
    compat env switches leave ``~/.claude/Claude.md`` loaded, so the kernel
    deny is the mechanism. Recomputed here from the same helper the policy
    used, so on-disk variants and symlink targets are part of the check.
    ``grok_home`` None means the default ``~/.grok``.
    """
    home = kernel_sandbox.protected_home()
    grok_root = os.path.abspath(grok_home or os.path.join(home, ".grok"))
    rows = []
    for root in (grok_root, os.path.join(home, ".claude"),
                 os.path.join(home, ".cursor")):
        found, why = kernel_sandbox._grok_instruction_deny_read(root)
        if why:
            _refuse("fence_path_rejected", why)
        rows += found
    return rows


def _canonical_spelling(path):
    """*path* with symlinks resolved by the grok scanner's own resolver.

    The longest resolvable prefix is resolved (``/tmp`` -> ``/private/tmp``)
    and any missing tail re-appended, so an allow for a not-yet-existing
    path is still compared on the same spelling as the deny rows.
    """
    head, tail = os.path.abspath(path), []
    while True:
        try:
            canonical, _st = kernel_sandbox._grok_scan_resolve(head)
            return os.path.join(canonical, *reversed(tail)) if tail else canonical
        except kernel_sandbox._GrokScanRefusal:
            parent = os.path.dirname(head)
            if parent == head:
                return None
            tail.append(os.path.basename(head))
            head = parent


def _reopens(entry, rows):
    """True when a later read allow *entry* reaches any denied row.

    Every comparison is the one refusal-side "same location" judgement
    (``kernel_sandbox.location_inside``): case and Unicode normalization
    (Seatbelt path rules match every spelling, live probe; APFS treats
    NFC/NFD as one name), a firmlink spelling, a symlinked root, and rows or
    allows that do not exist yet all reduce to the identity of the nearest
    existing ancestor plus the folded names below it. It is applied on both
    the lexical and the symlink-resolved spelling of the allow (``/tmp``
    re-opens ``/private/tmp``) and on a row both as a name and as resolved.
    An allow that cannot be resolved at all counts as re-opening.
    """
    kind, path = entry
    if kind not in ("literal", "subpath"):
        return True   # a regex allow cannot be bounded; refuse it
    canonical = _canonical_spelling(path)
    if canonical is None:
        return True
    inside = kernel_sandbox.location_inside
    with kernel_sandbox._location_cache():
        for spelled in {os.path.abspath(path), canonical}:
            for row_kind, row in rows:
                if inside(row, spelled) or inside(row, spelled, resolve_leaf=False):
                    return True
                if row_kind == "subpath" and inside(spelled, row):
                    return True
    return False


def _grok_instruction_wall_held(wrap_kw, grok_home):
    """True when the read wall denies every instruction row and no later
    read allow (``extra_post_read``, emitted after the deny) re-grants one."""
    if not wrap_kw.get("read_wall"):
        return False
    rows = _grok_instruction_rows(grok_home)
    denied = set(wrap_kw.get("extra_deny_read") or ())
    if any(row not in denied for row in rows):
        return False
    with kernel_sandbox._location_cache():
        return not any(_reopens(entry, rows)
                       for entry in (wrap_kw.get("extra_post_read") or ()))


def _agy_instruction_wall_held(wrap_kw):
    """True when every agy instruction / configuration row is present as a
    read deny, a FINAL read deny and a FINAL write deny, and the read wall is on;
    refuses (``fence_path_rejected``) when the agy dot tree breaks the layout
    rule (``kernel_sandbox.agy_layout_refusal``).

    Final rows are the proof instead of judging each allow: the final read and
    write denies follow every allow in the profile (R*, S*, M*, P*, T*,
    repository roots, the temp roots, ``/private/var/folders``), so a later
    allow cannot reopen a row, and a symlink at a family name that points into
    a writable root cannot be written through. A row missing from any of the
    three sets (recomputed here from the disk, so a link planted after the
    policy was built shows up) fails the postcondition.
    """
    if not wrap_kw.get("read_wall"):
        return False
    # The layout rule again, now that the wrap is composed: a link planted
    # between the policy build and here makes a second name for a row's file.
    why = kernel_sandbox.agy_layout_refusal()
    if why:
        _refuse("fence_path_rejected", why)
    rows, why = kernel_sandbox.agy_instruction_deny_rows()
    if why:
        _refuse("fence_path_rejected", why)
    required = (
        set(wrap_kw.get("extra_deny_read") or ()),
        set(wrap_kw.get("extra_deny_read_final") or ()),
        set(wrap_kw.get("extra_deny_write_final") or ()),
    )
    return all(row in held for row in rows for held in required)


# Child environment contract (issue #138). A fenced vendor child gets only
# the names below, not the server's whole environment, so an unrelated
# credential in the server environment (a cloud key, a token, a forwarded
# agent socket) never reaches a model-driven process. The fence applies it on
# macOS only; the two vendor children that run outside the fence (the codex
# quota probe and the write-canary probe) call ``allowlisted_child_env`` too.
CHILD_ENV_PASS_VAR = "OMNI_LLM_CHILD_ENV_PASS"
# The operator notify sink never reaches a child, even when listed.
_NOTIFY_SINK = "OMNI_LLM_NOTIFY_CMD"
_CHILD_ENV_EXACT = frozenset((
    # Process basics the vendor CLI and the tools it runs need to start.
    "PATH", "HOME", "TMPDIR", "USER", "LOGNAME", "SHELL",
    # Locale, timezone and terminal rendering.
    "LANG", "TZ", "TERM", "COLORTERM", "NO_COLOR",
    # Network egress: proxies and the CA bundles behind a TLS-inspecting one.
    "http_proxy", "https_proxy", "no_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE",
    # Per-provider home/config roots (profile and account selection).
    "CLAUDE_CONFIG_DIR", "CODEX_HOME", "GROK_HOME", "CURSOR_CONFIG_DIR",
    # Set by this module for the grok child (see prepare_host_fence).
    "GROK_DISABLE_AUTOUPDATER",
    # omni's own channel into a nested omni (see the note below).
    "OMNI_LLM_DEPTH", "OMNI_LLM_NOTIFY_QUIET",
    # The launcher's explicit git identity: the fence turns the global and
    # system git config off (see prepare_host_fence), so a repo-write child
    # commits with these or not at all.
    "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
))
# LC_* locale categories are the only prefix kept: every other name is
# enumerated. OMNI_LLM_DEPTH is the recursion guard a nested omni in the child
# honors, and OMNI_LLM_NOTIFY_QUIET is the flag omni sets so that nested omni
# stays quiet (the other OMNI_LLM_* names are read by the server process, not
# by a child, or are not meant for one). Nothing in providers/ passes the pi
# child a variable of its own (its PI_* session markers are dropped by the
# lane), so pi relies on OMNI_LLM_CHILD_ENV_PASS for anything beyond this set.
_CHILD_ENV_PREFIXES = ("LC_",)
# The vendor's own auth/endpoint variables, passed to that vendor's child only
# (a login by environment keeps working; another vendor's key is never lent).
# Mirrors the denylist of ``provider_accounts.account_env``, which strips the
# same names from an isolated child; a test pins the correspondence.
_VENDOR_AUTH_ENV = {
    "codex": frozenset((
        "CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL")),
    "claude": frozenset((
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")),
    "grok": frozenset((
        "XAI_API_KEY", "GROK_API_KEY", "XAI_BASE_URL", "XAI_API_BASE",
        "GROK_BASE_URL", "GROK_API_BASE", "GROK_CLI_CHAT_PROXY_BASE_URL",
        "GROK_WS_URL", "GROK_AUTH_PROVIDER_COMMAND", "GROK_MODELS_BASE_URL",
        "GROK_MODELS_LIST_URL", "GROK_XAI_API_BASE_URL", "GROK_OIDC_ISSUER",
        "GROK_OIDC_CLIENT_ID", "GROK_OIDC_AUDIENCE", "GROK_OIDC_SCOPES",
        "GROK_OAUTH2_ISSUER", "GROK_OAUTH2_CLIENT_ID", "GROK_OAUTH2_PRINCIPAL_ID",
        "GROK_OAUTH2_PRINCIPAL_TYPE", "GROK_OAUTH2_REFERRER", "GROK_OAUTH2_SCOPES",
        "GROK_WS_ORIGIN")),
    # cursor-agent's key is supplied by the account profile override.
    "cursor": frozenset(("CURSOR_API_KEY",)),
    # agy: AGY_CONFIG_DIR is the profile variable providers/agy.py sets (agy
    # 1.3.3 does not read it: measured, a run with it pointed at an empty
    # directory still used the default login), and AGY_ADC_AUTH is the
    # login-by-environment selector the CLI documents ("unset AGY_ADC_AUTH to
    # use other sign in methods"). A normal run (keychain login) needs neither.
    # agy has no account-ring layer, so provider_accounts.account_env has no
    # matching denylist for it.
    "agy": frozenset(("AGY_CONFIG_DIR", "AGY_ADC_AUTH")),
}


def _child_env_pass_names(environ=None):
    """Exact names the operator listed in ``OMNI_LLM_CHILD_ENV_PASS``."""
    raw = (os.environ if environ is None else environ).get(CHILD_ENV_PASS_VAR, "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def allowlisted_child_env(env, provider):
    """Copy of *env* holding only the names the child contract allows.

    Kept: the fixed names and prefixes above, the *provider*'s own auth
    variables, and the names listed in ``OMNI_LLM_CHILD_ENV_PASS`` (exact
    names, no patterns). Listing a name passes the variable only: it does not
    reopen anything the fence denies (``SSH_AUTH_SOCK`` stays unreachable);
    ``OMNI_LLM_NOTIFY_CMD`` is dropped even when listed.
    """
    keep = _CHILD_ENV_EXACT | _VENDOR_AUTH_ENV.get(provider, frozenset())
    keep |= _child_env_pass_names()
    return {name: value for name, value in env.items()
            if name != _NOTIFY_SINK
            and (name in keep or name.startswith(_CHILD_ENV_PREFIXES))}


def _same_claude_home(built_for, config_dir):
    """The policy's recorded claude home is the run's *config_dir* (or both default)."""
    if not built_for or not config_dir:
        return not built_for and not config_dir
    want = os.path.realpath(os.fspath(config_dir))
    if built_for == want:
        return True
    try:
        return os.path.samestat(os.stat(built_for), os.stat(want))
    except OSError:
        return False


def _is_codex_app_server_argv(argv, read_only):
    """Match only the resolved, read-only Codex app-server command shape."""
    if not read_only or not isinstance(argv, (list, tuple)) or len(argv) != 2:
        return False
    try:
        head = os.fsdecode(os.fspath(argv[0]))
        subcommand = os.fsdecode(os.fspath(argv[1]))
    except TypeError:
        return False
    if subcommand != "app-server" or not os.path.isabs(head):
        return False
    if head != os.path.realpath(head):
        return False
    found = vendor_bin.which("codex")
    return bool(found and head == os.path.realpath(found))


def prepare_host_fence(argv, *, provider, workdir, read_only=False, env=None,
                       config_dir=None, launcher_roots=(), policy=None,
                       extra_read_paths=(), extra_write_paths=(), grok_order_file=None,
                       allow_repo_write=False, allow_main=False):
    """Return ``(argv, env, metadata)`` after one verified host wrap.

    Darwin has no opt-out. Environment markers never establish containment.
    ``policy`` lets an existing consumer retain its already-declared allow-set
    while sharing this operation; ordinary workers use ``agent_fence_policy``.
    Declared extra paths use the kernel layer's existing admission checks.
    On other platforms argv and env are returned unchanged.

    Write-mode git worktrees admit repository metadata (linked git dir /
    common dir) so a worker can commit. ``allow_repo_write`` makes that
    admission mandatory (non-git refuses). Protected branches are skipped
    unless ``allow_main``.
    """
    none = {"fence": "none",
            "fence_detail": dict(kernel_sandbox.FENCE_NONE_DETAIL)}
    if allow_repo_write:
        if read_only:
            _refuse("fence_path_rejected", "--allow-repo-write cannot be combined with --read-only")
        if platform.system() != "Darwin":
            _refuse("fence_unavailable", "--allow-repo-write requires the macOS host fence")
        if provider not in ("codex", "claude", "grok", "pi", "cursor"):
            _refuse("fence_path_rejected", "--allow-repo-write requires a task worker provider")
    repo_roots = ()
    repo_denies = ()
    repo_meta = ()
    repo_post_writes = ()
    repo_final_denies = ()
    order = None
    if grok_order_file is not None:
        order = _admit_grok_order_file(
            argv, provider, grok_order_file, os.environ if env is None else env)
    if platform.system() != "Darwin":
        return argv, env, none
    if not kernel_sandbox.available():
        _refuse("fence_unavailable", "kernel fence unavailable "
                "(/usr/bin/sandbox-exec missing or not executable)")
    if not argv:
        _refuse("fence_unavailable", "host fence requires a command")
    try:
        # Judge the head as exec will see it: bytes and path objects too.
        head = os.fsdecode(os.fspath(argv[0]))
    except TypeError:
        _refuse("fence_path_rejected", "host fence command head is not a path")
    if head.startswith("-"):
        # sandbox-exec would parse it as its own option (-D, -p, -f) and the
        # caller could override any profile param, the grok pid included.
        _refuse("fence_path_rejected", "host fence command must not start with '-'")
    workdir = os.path.realpath(os.fspath(workdir)) if workdir is not None else None
    if workdir is not None and not os.path.isdir(workdir):
        _refuse("fence_path_rejected", "host workdir is not an existing directory")
    if not read_only and workdir is not None:
        repo_roots, repo_denies, repo_meta, repo_post_writes, repo_final_denies = (
            _repo_write_roots(
                workdir, required=bool(allow_repo_write), allow_main=allow_main)
        )
    child_env = dict(os.environ if env is None else env)
    if provider == "grok":
        # A relative root means something else in the child's cwd than here.
        for label, value in (("GROK_HOME", child_env.get("GROK_HOME")),
                             ("grok config dir", config_dir)):
            if value is not None and not os.path.isabs(os.fspath(value)):
                _refuse("fence_path_rejected", "%s must be an absolute path" % label)
    elif provider == "claude":
        # Same reason: host-exec chdirs into the workdir, so a relative home
        # would name another directory in the child than the one judged here.
        for label, value in (("CLAUDE_CONFIG_DIR", child_env.get("CLAUDE_CONFIG_DIR")),
                             ("claude config dir", config_dir)):
            if value and not os.path.isabs(os.fspath(value)):
                _refuse("fence_path_rejected", "%s must be an absolute path" % label)
    if config_dir is None:
        config_key = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME",
                      "cursor": "CURSOR_CONFIG_DIR", "grok": "GROK_HOME"}.get(provider)
        config_dir = child_env.get(config_key) if config_key else None
    elif provider == "grok":
        # The policy is built for config_dir; the child must run on it.
        env_home = child_env.get("GROK_HOME")
        if env_home and os.path.abspath(env_home) != os.path.abspath(config_dir):
            _refuse("fence_path_rejected",
                    "GROK_HOME differs from the fenced grok config dir")
        child_env["GROK_HOME"] = config_dir
    elif provider == "claude" and config_dir:
        # Same rule as grok: the policy is built for config_dir, so the child
        # must run on it. A keyword never reaches the child by itself, and a
        # child left on the default home would sit outside the judged rows.
        env_home = child_env.get("CLAUDE_CONFIG_DIR")
        if env_home and os.path.abspath(env_home) != os.path.abspath(config_dir):
            _refuse("fence_path_rejected",
                    "CLAUDE_CONFIG_DIR differs from the fenced claude config dir")
        child_env["CLAUDE_CONFIG_DIR"] = config_dir
    custom_policy = policy is not None
    if policy is None:
        policy = kernel_sandbox.agent_fence_policy(
            provider, workdir=workdir, read_only=read_only,
            config_dir=config_dir, launcher_roots=launcher_roots)
    if policy.get("ok") is False:
        raise HostFenceError(policy)
    if provider == "claude":
        # A claude policy is built for one home (a named profile root, or the
        # default home). Its rows and vendor-root guards judge only that
        # home, so a run on another one cannot reuse it.
        built_for = policy.get("claude_config_root")
        if "claude_config_root" not in policy:
            # A policy that never recorded its home (assembled by hand): the
            # default-home guards below still judge a default run, but a run
            # on a named home cannot be matched to it.
            if config_dir:
                _refuse("fence_path_rejected",
                        "claude policy does not record the config home it "
                        "was built for")
        elif not _same_claude_home(built_for, config_dir):
            _refuse("fence_path_rejected",
                    "claude policy was built for a different config home "
                    "than this run's")
    wrap_kw = kernel_sandbox.wrap_kwargs_from_policy(policy)
    if order is not None:
        # W* deny clauses precede runtime T* re-allows. Refuse any overlap,
        # so the named order cannot become writable through a later clause,
        # including the repository T* rows appended below.
        for entry in tuple(wrap_kw.get("extra_post_write") or ()) + tuple(repo_post_writes or ()):
            bound = kernel_sandbox._write_entry_bound(*entry)
            if bound is None:
                _refuse("fence_path_rejected", "Grok order runtime overlap cannot be verified")
            kind, path = bound
            if ((kind == "literal"
                 and kernel_sandbox._same_path_any_case(path, order.path))
                    or (kind == "subpath"
                        and kernel_sandbox._path_inside_any_case(order.path, path))):
                _refuse("fence_path_rejected", "Grok order overlaps writable runtime state")
        extra_read_paths = tuple(extra_read_paths) + (order,)
        wrap_kw["extra_deny_write"] = tuple(wrap_kw.get("extra_deny_write") or ()) + (
            ("literal", order.path),)
    if repo_meta:
        _post_write_overlaps_repo(wrap_kw.get("extra_post_write"), repo_meta)
    if repo_roots:
        extra_read_paths = tuple(extra_read_paths) + repo_roots
        extra_write_paths = tuple(extra_write_paths) + repo_roots
    for key, extra in (("extra_read_paths", extra_read_paths),
                       ("extra_write_paths", extra_write_paths)):
        if extra:
            wrap_kw[key] = tuple(wrap_kw.get(key) or ()) + tuple(extra)
    if repo_roots:
        # Existing hard links are rejected above; this prevents the child from
        # creating a new alias between an allowed repo path and an outside file.
        wrap_kw["extra_deny_link"] = tuple(wrap_kw.get("extra_deny_link") or ()) + tuple(
            ("subpath", root.path) for root in repo_roots)
    if repo_denies:
        wrap_kw["extra_deny_write"] = tuple(wrap_kw.get("extra_deny_write") or ()) + repo_denies
    if repo_post_writes:
        wrap_kw["extra_post_write"] = tuple(wrap_kw.get("extra_post_write") or ()) + repo_post_writes
    if repo_final_denies:
        wrap_kw["extra_deny_write_final"] = (
            tuple(wrap_kw.get("extra_deny_write_final") or ()) + repo_final_denies
        )
    # The record of the protected names (and the ledger beside it) is out of the
    # child's reach in EVERY fenced profile: git workdir or not, read-only or not.
    try:
        record_rows = record_dir_denies()
    except ProtectedListError as exc:
        _refuse("fence_path_rejected", exc.detail)
    wrap_kw["extra_deny_write_final"] = (
        tuple(wrap_kw.get("extra_deny_write_final") or ()) + record_rows
    )
    if repo_meta:
        wrap_kw["extra_deny_exec"] = (
            tuple(wrap_kw.get("extra_deny_exec") or ()) + _git_remote_exec_denies()
        )
    if (provider in ("pi", "cursor", "agy")
            or (provider == "claude" and not config_dir)) \
            and not policy.get("vendor_root"):
        # Vendor state rows without the root they were spelled for cannot be
        # checked against the composed write set: refuse, never skip. A claude
        # run on a NAMED home (CLAUDE_CONFIG_DIR) carries no vendor root by
        # design and its state rows sit under the profile root, which the
        # caller-config-dir rule admits: only the default home is judged here.
        vendor_real = os.path.realpath(os.path.join(
            kernel_sandbox.protected_home(), kernel_sandbox._VENDOR_DOT[provider]))
        for entry in wrap_kw.get("extra_post_write") or ():
            bound = kernel_sandbox._write_entry_bound(*entry)
            if bound is None or kernel_sandbox._path_inside_any_case(
                    bound[1], vendor_real):
                _refuse("fence_path_rejected",
                        "%s policy carries vendor state write rows without "
                        "its vendor root" % provider)
    if provider in ("pi", "cursor", "agy", "claude") and policy.get("vendor_root"):
        # The policy judged the vendor-root symlink chain against its own
        # write entries; the repository roots and declared extras added
        # above can contain a link of that chain or the resolved root.
        try:
            why = kernel_sandbox.pi_cursor_root_link_refusal(
                provider, workdir, read_only,
                wrap_kw.get("extra_write_paths"),
                wrap_kw.get("extra_post_write"), policy["vendor_root"])
        except kernel_sandbox.FencePathRejected as exc:
            _refuse("fence_path_rejected", exc.detail)
        if why:
            _refuse("fence_path_rejected", why)
    # Final containment of generated API settings, for every provider. The
    # deny-write rows on both settings bases are always emitted: they only
    # cover omni's own private areas, so ordinary runs cannot be broken by
    # them. The exposure REFUSAL (a write root containing a base or an
    # ancestor of it, which would let the child substitute its own tree)
    # applies only when API-kind accounts are configured, judged against the
    # FULLY composed write set (repository roots included).
    configured = api_accounts_configured()
    try:
        settings_bases = api_settings_bases()
    except RuntimeError as exc:
        if configured:
            _refuse("fence_path_rejected", "API settings area cannot be resolved: %s" % redact_secrets(str(exc)))
        settings_bases = ()  # nothing to protect without a resolvable home
    if configured:
        try:
            final_entries = [
                ("subpath", path) for path in kernel_sandbox._write_allow_paths(
                    workdir, read_only, extra_write_paths=wrap_kw.get("extra_write_paths"))
            ] + list(wrap_kw.get("extra_post_write") or ())
        except kernel_sandbox.FencePathRejected as exc:
            _refuse("fence_path_rejected", exc.detail)
        for base in settings_bases:
            exposed = settings_base_exposed(base, final_entries)
            if exposed:
                _refuse("fence_path_rejected",
                        "child write roots reach the API settings area: %s" % exposed)
    if settings_bases:
        wrap_kw["extra_deny_write_final"] = tuple(
            wrap_kw.get("extra_deny_write_final") or ()) + tuple(
            row for base in settings_bases
            for row in kernel_sandbox._deny_entries("subpath", base))
    try:
        # The operator's ssh-agent socket: no connect, and no second name for
        # it (a hard link is another pathname the connect deny does not name).
        socket_denies = kernel_sandbox.ssh_agent_socket_denies(
            os.environ, env, cwd=workdir)
    except kernel_sandbox.FencePathRejected as exc:
        _refuse("fence_path_rejected", exc.detail)
    if socket_denies:
        wrap_kw["extra_deny_link"] = (
            tuple(wrap_kw.get("extra_deny_link") or ()) + socket_denies)
    try:
        refusal = kernel_sandbox.verify_fence(workdir, read_only=read_only, **wrap_kw)
        if refusal:
            raise HostFenceError(refusal)
        wrapped = kernel_sandbox.wrap_argv(
            argv, workdir, read_only, **wrap_kw,
            extra_deny_connect=socket_denies)
    except kernel_sandbox.FencePathRejected as exc:
        _refuse("fence_path_rejected", exc.detail)
    if not kernel_sandbox.is_wrapped(wrapped):
        _refuse("fence_unavailable", "kernel fence wrap postcondition failed "
                "(argv not prefixed with sandbox-exec)")
    if provider == "codex":
        if _is_codex_app_server_argv(argv, read_only):
            found_codex = vendor_bin.which("codex")
            resolved_codex = os.path.realpath(found_codex) if found_codex else None
            if (resolved_codex is None
                    or list(argv) != [resolved_codex, "app-server"]
                    or not kernel_sandbox.is_wrapped(wrapped)
                    or wrapped[-len(argv):] != list(argv)
                    or list(argv[1:]) != ["app-server"]
                    or kernel_sandbox.INNER_SANDBOX_BYPASS in wrapped):
                _refuse("fence_unavailable",
                        "Codex app-server fence postcondition failed")
        else:
            wrapped = kernel_sandbox.disable_inner_codex_sandbox(wrapped)
            if kernel_sandbox.inner_sandbox_disable_failed(wrapped):
                _refuse("fence_unavailable", "inner vendor sandbox disable postcondition failed")
    elif provider == "grok":
        # Disable the inner sandbox only after the outer one is verified.
        # Preserve every unrelated argument, including prompt/session flags.
        wrapped, _vendor = _rewrite_inner_sandbox(
            wrapped, argv, "off", _GROK_VALUE_OPTIONS)
        child_env["GROK_DISABLE_AUTOUPDATER"] = "1"
    elif provider == "cursor":
        # cursor-agent's --sandbox enabled launches cursorsandbox, a second
        # sandbox-exec. Nested sandbox_apply is rc=71 under this fence.
        wrapped, vendor = _rewrite_inner_sandbox(
            wrapped, argv, "disabled", _CURSOR_VALUE_OPTIONS)
        if _sandbox_flag_values(vendor, _CURSOR_VALUE_OPTIONS) != ("disabled",):
            _refuse("fence_unavailable",
                    "inner vendor sandbox disable postcondition failed")
    if custom_policy and provider == "codex":
        # Direct Codex consult historically pins HOME only. Keep its notify
        # and config environment contract when reusing its existing policy.
        child_env = kernel_sandbox.sanitized_child_env(child_env)
    else:
        child_env = kernel_sandbox.agent_child_env(
            child_env, tmpdir=policy.get("tmpdir"),
            keep_config_dir=bool(provider == "claude" and config_dir))
    # The last env rewrite before the fence's own GIT_* keys below: nothing
    # else the server environment carried reaches the child.
    child_env = allowlisted_child_env(child_env, provider)
    if provider == "claude":
        # Judge the CLAUDE_CONFIG_DIR the child will actually get, after
        # every env rewrite: empty or absent means the default home.
        child_home = child_env.get("CLAUDE_CONFIG_DIR") or ""
        if bool(child_home) != bool(config_dir) or (
                config_dir and os.path.abspath(child_home) != os.path.abspath(config_dir)):
            _refuse("fence_unavailable",
                    "claude child config home does not match the fenced "
                    "config home")
    if provider == "grok":
        # Judge the GROK_HOME the child will actually get, after every env
        # rewrite; the policy's root is irrelevant if the child uses another.
        child_home = child_env.get("GROK_HOME") or os.path.join(
            child_env.get("HOME") or kernel_sandbox.protected_home(), ".grok")
        if not _grok_instruction_wall_held(wrap_kw, child_home):
            _refuse("fence_unavailable",
                    "grok global-instruction read deny postcondition failed")
    if provider == "agy" and not _agy_instruction_wall_held(wrap_kw):
        _refuse("fence_unavailable",
                "agy global-instruction read deny postcondition failed")
    if repo_meta:
        # Git's global config lives under the protected home wall. The task
        # launcher must pass the resolved author/committer identity explicitly.
        child_env["GIT_CONFIG_GLOBAL"] = os.devnull
        child_env["GIT_CONFIG_SYSTEM"] = os.devnull
        child_env["GIT_CONFIG_NOSYSTEM"] = "1"
        # Local commit only: block git's network transports. HTTPS for the
        # vendor CLI is a different process image and does not use this env.
        child_env["GIT_ALLOW_PROTOCOL"] = ""
    placeholder = ["-D", "%s=%s" % (kernel_sandbox.GROK_PID_PARAM,
                                    kernel_sandbox.GROK_PID_PLACEHOLDER)]
    if provider == "grok" and wrapped[1] == "-p" and wrapped[3:5] == placeholder:
        # Bind grok's temp-file allow to grok's own pid (see GROK_PID_SHIM).
        wrapped = ["/bin/sh", "-c", kernel_sandbox.GROK_PID_SHIM] + wrapped
    return wrapped, child_env, {
        "fence": "seatbelt-fs",
        "fence_detail": {
            "read_wall": bool(wrap_kw.get("read_wall")),
            "workdir_writable": not read_only,
            "vendor_state_writable": bool(policy.get("vendor_state_writable")),
        },
    }


def _vendor_exec_path(command, provider, workdir):
    """Exec the provider's own CLI by realpath, as the Python lanes do.

    A bare name resolves against the caller's PATH, which the sanitized child
    env may not carry (ENOENT), and Seatbelt refuses an execvp through the
    ``~/.local/bin`` symlink even when the leaf is admitted (EPERM). Only a
    command that resolves to ``resolve_agent_binary(provider)`` is rewritten;
    every other command is exec'd as given. A relative path is resolved
    against ``workdir``, where the exec happens after ``chdir``.
    """
    vendor = kernel_sandbox.resolve_agent_binary(provider)
    head = command[0]
    if os.sep in head:
        found = os.path.join(workdir, head)
    else:
        found = vendor_bin.which(head)
        vendor_name = "cursor-agent" if provider == "cursor" else provider
        if head == vendor_name and (found is None or vendor is None):
            _refuse("cli_missing",
                    "%s not found on PATH outside harness launcher shims" % head)
        # A relative PATH entry resolves against the invoker cwd here but
        # against the workdir at exec time: leave such a lookup to execvpe.
        if found and not os.path.isabs(found):
            found = None
    if vendor and found and os.path.realpath(found) == vendor:
        return [vendor, *command[1:]]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(prog="omni host-exec")
    parser.add_argument("--provider", required=True,
                        choices=("claude", "codex", "grok", "pi", "cursor", "agy"))
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--allow-repo-write", action="store_true",
                        help="allow writes to the selected Git worktree and its Git metadata")
    parser.add_argument("--config-dir")
    parser.add_argument("--launcher-root", action="append", default=[])
    parser.add_argument("--grok-order-file", action="append", default=[],
                        help="single placed TEAM_INBOX workorder used by Grok --prompt-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    try:
        command = _vendor_exec_path(command, args.provider, args.workdir)
        if len(args.grok_order_file) > 1:
            _refuse("fence_path_rejected", "only one Grok order file may be declared")
        command, env, _metadata = prepare_host_fence(
            command, provider=args.provider, workdir=args.workdir,
            read_only=args.read_only, env=dict(os.environ),
            config_dir=args.config_dir, launcher_roots=args.launcher_root,
            grok_order_file=args.grok_order_file[0] if args.grok_order_file else None,
            allow_repo_write=args.allow_repo_write)
        os.chdir(args.workdir)
        os.execvpe(command[0], command, env)
    except HostFenceError as exc:
        # Every string that leaves is masked, whoever built the envelope.
        printable = {key: redact_secrets(value) if isinstance(value, str) else value
                     for key, value in exc.envelope.items()}
        print(json.dumps(printable), file=sys.stderr)
        return 126
    except OSError as exc:
        print(json.dumps({"ok": False, "reason": "host_exec_failed",
                          "detail": redact_secrets(str(exc))[:300]}), file=sys.stderr)
        return 126


if __name__ == "__main__":
    # Entered directly (``python -m providers.host_fence``): ``omni_cli.main``
    # did not run, so register the credentials the diagnostics could quote.
    from providers import known_secrets
    known_secrets.prime()
    sys.exit(main())
