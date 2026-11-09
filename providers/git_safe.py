"""Git queries made by the omni PARENT process that cannot run repo-chosen code.

The parent asks git about a workdir before and after every spawn (branch
fence, drift pin, host-fence roots, workspace snapshot), outside any child
fence. Git executes commands named by repository-controlled configuration
and attributes while answering some of those questions, so a repository an
agent (or an archive) shaped could otherwise run code in the parent:

  * ``core.fsmonitor`` (a path, also reachable through ``include.path``,
    ``includeIf``, ``config.worktree`` or a ``.git`` pointer file) is run by
    ``git status``;
  * ``filter.<driver>.clean`` / ``smudge`` / ``process`` named by a
    ``.gitattributes`` or ``info/attributes`` line run when ``git status``
    has to hash a stat-dirty file (the driver name may be empty);
  * an index refresh by ``git status`` fires the ``post-index-change`` hook;
  * ``git status`` recursing into a submodule runs a child ``git status``
    that reads the submodule's own configuration and attributes;
  * reading an object missing from a promisor-remote repository starts a
    lazy ``git fetch``, which runs ``core.sshCommand``, ``uploadpack``,
    ``ext::`` helpers and so on.

Enumerating such vectors does not converge, so every parent git call goes
through :func:`run_git`, which stacks two kinds of layer.

Layer A (structural, Darwin): the git process runs under ``sandbox-exec``
with network denied, ``process-exec`` allowed only for ONE binary and
``file-write*`` allowed only for ``/dev/null``. Whatever the repository's
configuration names, a filter, hook, fsmonitor, ssh or helper cannot be
exec'd, cannot reach a network and cannot write, and a config race between
two reads cannot change that. That one binary is the Command Line Tools /
Xcode git found at ``<realpath(/var/db/xcode_select_link)>/usr/bin/git`` and
accepted only when it and every directory above it is root-owned and neither
group- nor world-writable. PATH, ``GIT_EXEC_PATH``, ``DEVELOPER_DIR`` and the
``/usr/bin/git`` shim (which resolves the real binary through a cache in the
per-user temp dir that fenced children can write) take no part, and nothing
is run before the sandbox is applied. A Darwin host without a usable
``sandbox-exec`` or without that trusted git fails closed (GitSafeError).

Layer B (all platforms, defense in depth):

  * ``--no-optional-locks`` (+ ``GIT_OPTIONAL_LOCKS=0``), ``-c
    core.fsmonitor=false`` and ``-c core.hooksPath=/dev/null``;
  * for content-reading subcommands, ``--no-lazy-fetch`` (+
    ``GIT_NO_LAZY_FETCH=1``) and every filter driver git can see blanked,
    then verified. The verification always runs, so a git too old to honour
    the blanking fails closed instead of running unneutralised;
  * the inherited environment loses the variables that make the config
    readers disagree or point git at another repository (``GIT_CONFIG``,
    ``GIT_DIR`` ...).

Filter drivers are blanked through ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_n``
/ ``GIT_CONFIG_VALUE_n`` rather than ``-c name=value`` because a driver name
may contain ``=``, which ``-c`` would split at the wrong place (verified).
Callers pass ``--ignore-submodules=dirty`` to ``status`` so no child process
is started for a submodule.
"""
from __future__ import annotations

import os
import platform
import re
import stat
import subprocess

from providers import kernel_sandbox

# Subcommands the parent runs that never read work-tree content or objects, so
# no filter driver can fire and no lazy fetch can start. Anything else is
# treated as content-reading. ``for-each-ref`` qualifies only for the ref name,
# object id and symref fields the protected-ref check asks for (no %(contents)).
_NO_CONTENT_SUBCOMMANDS = frozenset(
    {"rev-parse", "symbolic-ref", "for-each-ref", "--exec-path"})
_FILTER_BLANKED = ("clean", "smudge", "process")
# Variables that make `git config` and the real call read different
# configuration, or run against a repository other than the workdir's.
_STRIPPED_ENV = (
    "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_DIR", "GIT_WORK_TREE",
    "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_PREFIX", "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_QUARANTINE_PATH",
    # relocates git's helper tree (and, through it, which binary is "git's own")
    "GIT_EXEC_PATH",
)
# What a read-only status/config/rev-parse writes: found empirically to be
# /dev/null only (git re-opens it to sanitise stdio).
_WRITE_ALLOW = ("/dev/null",)
# Darwin: the only git the parent will run. The developer dir comes from this
# root-owned symlink, never from PATH, DEVELOPER_DIR or the /usr/bin/git shim
# (the shim resolves the real binary through an xcrun cache in the per-user
# temp dir, which fenced children can write).
_XCODE_SELECT_LINK = "/var/db/xcode_select_link"
_TRUSTED_UIDS = (0,)
# `--no-lazy-fetch` first appeared in git 2.45.
_LAZY_FETCH_MIN = (2, 45)
_version_cache = {}


class GitSafeError(RuntimeError):
    """The safe git environment could not be established (callers fail closed)."""


def safe_prefix(workdir=None):
    """The argv words that sit between ``git`` and the subcommand."""
    prefix = ["--no-optional-locks",
              "-c", "core.fsmonitor=false",
              "-c", "core.hooksPath=/dev/null"]
    if workdir is not None:
        prefix += ["-C", os.fspath(workdir)]
    return prefix


def _content_options():
    """Global options for subcommands that read objects or file content."""
    return ["--no-lazy-fetch"]


def git_argv(workdir, args, git_bin="git", content=False, lazy_flag=True,
             git_dir=None):
    """Full (unsandboxed) argv for one parent git call.

    *git_dir* names the repository explicitly (``--git-dir``): nothing is
    discovered from the working directory, so a ``.git`` pointer, a
    ``commondir`` file or a ``.git`` entry planted next to it decides nothing.
    """
    return ([git_bin] + safe_prefix(workdir)
            + (["--git-dir", os.fspath(git_dir)] if git_dir is not None else [])
            + (_content_options() if content and lazy_flag else []) + list(args))


def _is_content(args):
    return not args or args[0] not in _NO_CONTENT_SUBCOMMANDS


def _base_env(env):
    out = {k: v for k, v in (os.environ if env is None else env).items()
           if k not in _STRIPPED_ENV}
    out["GIT_OPTIONAL_LOCKS"] = "0"
    out["GIT_NO_LAZY_FETCH"] = "1"
    out["xcrun_nocache"] = "1"  # defensive: never consult the xcrun lookup cache
    return out


# --------------------------------------------------------------------------- #
# Layer A: sandbox-exec
# --------------------------------------------------------------------------- #
def _sandboxed():
    if platform.system() != "Darwin":
        return False
    if not kernel_sandbox.available():
        raise GitSafeError("sandbox-exec is not available - parent git fails closed")
    return True


def _resolve_git(git_bin, env):
    """Non-Darwin only: absolute path of the git to run (PATH walk, like execvp)."""
    if os.sep in git_bin:
        candidates = [git_bin]
    else:
        candidates = [os.path.join(d or ".", git_bin)
                      for d in (env.get("PATH") or os.defpath).split(os.pathsep)]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    raise FileNotFoundError("git not found: %r" % (git_bin,))


def _check_chain(path, expected):
    """lstat every component of *path* (already a realpath): no symlink, owned
    by a trusted uid, not group- or world-writable."""
    parts = [part for part in path.split("/") if part]
    for i in range(len(parts) + 1):
        here = "/" + "/".join(parts[:i])
        try:
            st = os.lstat(here)
        except OSError as exc:
            raise GitSafeError("no trusted git at %s (%s: %s)" % (expected, here, exc)) from None
        if stat.S_ISLNK(st.st_mode):
            raise GitSafeError("no trusted git at %s (%s is a symlink)" % (expected, here))
        if st.st_uid not in _TRUSTED_UIDS:
            raise GitSafeError("no trusted git at %s (%s is not root-owned)" % (expected, here))
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise GitSafeError("no trusted git at %s (%s is group- or world-writable)"
                               % (expected, here))


def _trusted_git():
    """Darwin: the Command Line Tools / Xcode git at the xcode-select dir.

    ``<realpath(/var/db/xcode_select_link)>/usr/bin/git``; the link, the
    binary and every directory above them must be root-owned, not symlinks
    and not group- or world-writable. Anything else fails closed.
    """
    developer_dir = os.path.realpath(_XCODE_SELECT_LINK)
    expected = os.path.join(developer_dir, "usr", "bin", "git")
    try:
        link = os.lstat(_XCODE_SELECT_LINK)
    except OSError as exc:
        raise GitSafeError("no trusted git at %s (%s: %s)"
                           % (expected, _XCODE_SELECT_LINK, exc)) from None
    if link.st_uid not in _TRUSTED_UIDS:
        raise GitSafeError("no trusted git at %s (%s is not root-owned)"
                           % (expected, _XCODE_SELECT_LINK))
    _check_chain(os.path.realpath(os.path.dirname(_XCODE_SELECT_LINK)), expected)
    if os.path.realpath(expected) != expected:
        raise GitSafeError("no trusted git at %s (path is or crosses a symlink)" % expected)
    _check_chain(expected, expected)
    st = os.lstat(expected)
    if not stat.S_ISREG(st.st_mode) or not os.access(expected, os.X_OK):
        raise GitSafeError("no trusted git at %s (not an executable file)" % expected)
    return expected


def sandbox_profile(exec_paths):
    """The SBPL profile (paths come in through -D parameters, not the text)."""
    allow = "".join('(literal (param "X%d"))' % i for i in range(len(exec_paths)))
    writes = "".join('(literal (param "W%d"))' % i for i in range(len(_WRITE_ALLOW)))
    return ("(version 1)(allow default)(deny network*)"
            "(deny process-exec*)(allow process-exec %s)"
            "(deny file-write*)(allow file-write* %s)" % (allow, writes))


def _sandbox_argv(trusted, argv):
    """Wrap ``argv`` (argv[0] replaced by the trusted binary) in sandbox-exec.

    The exec allowlist is that single binary: it needs no other exec (the
    foreman's check and this module's tests run status, diff, config,
    rev-parse and symbolic-ref with exactly this profile).
    """
    out = [kernel_sandbox.SANDBOX_EXEC, "-p", sandbox_profile((trusted,)),
           "-D", "X0=%s" % trusted]
    for i, path in enumerate(_WRITE_ALLOW):
        out += ["-D", "W%d=%s" % (i, path)]
    return out + [trusted] + list(argv[1:])


def _git_version(trusted, env):
    """(major, minor) of the trusted git, asked once under the sandbox."""
    st = os.stat(trusted)
    key = (trusted, st.st_dev, st.st_ino, st.st_mtime_ns)
    if key not in _version_cache:
        p = subprocess.run(_sandbox_argv(trusted, [trusted, "--version"]),
                           capture_output=True, text=True, env=env, cwd="/",
                           timeout=10, check=False)
        m = re.search(r"git version (\d+)\.(\d+)", p.stdout or "")
        if p.returncode != 0 or not m:
            raise GitSafeError("git --version failed (rc=%s)" % p.returncode)
        _version_cache[key] = (int(m.group(1)), int(m.group(2)))
    return _version_cache[key]


def _lazy_flag_supported(trusted, env):
    return _git_version(trusted, env) >= _LAZY_FETCH_MIN


def _launch(argv, env, timeout):
    """Run one git argv (argv[0] = git), sandboxed on Darwin."""
    if _sandboxed():
        full = _sandbox_argv(_trusted_git(), argv)
    else:
        full = [_resolve_git(argv[0], env)] + list(argv[1:])
    p = subprocess.run(full, capture_output=True, text=True, env=env,
                       timeout=timeout, check=False)
    if full[0] == kernel_sandbox.SANDBOX_EXEC and p.returncode == 71 and (
            "sandbox_apply" in (p.stderr or "") or "can't exec" in (p.stderr or "")
            or "execvp" in (p.stderr or "")):
        raise GitSafeError("sandboxed git could not start: %s"
                           % (p.stderr or "").strip()[:120])
    return p


# --------------------------------------------------------------------------- #
# Layer B: filter drivers
# --------------------------------------------------------------------------- #
def _filter_drivers(workdir, git_bin, env, timeout):
    """Every ``filter.<driver>`` name visible to git (includes honoured).

    A config read: nothing is executed. rc 1 means "no match". Any other
    failure is indeterminate and raises, so the caller fails closed. The
    driver name may be empty (``[filter ""]`` gives the key ``filter..clean``).
    """
    argv = git_argv(workdir, ["config", "--includes", "--name-only", "--null",
                              "--get-regexp", r"^filter\."], git_bin)
    p = _launch(argv, env, timeout)
    if p.returncode == 1:
        return []
    if p.returncode != 0:
        raise GitSafeError("git config read failed (rc=%s): %s"
                           % (p.returncode, (p.stderr or "").strip()[:120]))
    names = []
    for key in (p.stdout or "").split("\0"):
        if not key.startswith("filter."):
            continue
        body = key[len("filter."):]
        if "." not in body:
            continue  # filter.<var> without a driver name: not a driver
        name = body.rsplit(".", 1)[0]
        if name not in names:
            names.append(name)
    return names


def _blank_filters(workdir, git_bin, env, timeout):
    """Return ``env`` extended so every filter driver is a no-op, or raise.

    The verification runs even when no driver was found: a config that
    appears between the enumeration and here, or a git that ignores the
    overrides, must not slip through.
    """
    names = _filter_drivers(workdir, git_bin, env, timeout)
    out = dict(env)
    if names:
        try:
            count = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
            if count < 0:
                raise ValueError(count)
        except ValueError:
            raise GitSafeError("GIT_CONFIG_COUNT is not a count") from None
        pairs = []
        for name in names:
            for var in _FILTER_BLANKED:
                pairs.append(("filter.%s.%s" % (name, var), ""))
            pairs.append(("filter.%s.required" % name, "false"))
        for key, value in pairs:
            out["GIT_CONFIG_KEY_%d" % count] = key
            out["GIT_CONFIG_VALUE_%d" % count] = value
            count += 1
        out["GIT_CONFIG_COUNT"] = str(count)
    _verify_blanked(workdir, git_bin, out, timeout)
    return out


def _verify_blanked(workdir, git_bin, env, timeout):
    """Fail closed unless the last value of every clean/smudge/process is empty."""
    argv = git_argv(workdir, ["config", "--includes", "--null",
                              "--get-regexp", r"^filter\."], git_bin)
    p = _launch(argv, env, timeout)
    if p.returncode not in (0, 1):
        raise GitSafeError("git config verify failed (rc=%s)" % p.returncode)
    last = {}
    for entry in (p.stdout or "").split("\0"):
        if not entry:
            continue
        key, _sep, value = entry.partition("\n")
        last[key] = value  # later (command-scope) entries win
    for key, value in last.items():
        if key.rsplit(".", 1)[-1] in _FILTER_BLANKED and value != "":
            raise GitSafeError("filter driver %r could not be neutralised" % key)


def run_git(workdir, args, *, git_bin="git", env=None, timeout=10, git_dir=None):
    """Run one parent git query; returns the ``CompletedProcess``.

    ``git_dir`` (explicit repository, see ``git_argv``) is for the subcommands
    that read no content only: a content-reading call blanks the filter drivers
    of the repository found from *workdir*, which would not be the one named.

    Raises :class:`GitSafeError` when the safe environment cannot be built
    (callers treat that exactly like an indeterminate git state), and lets
    ``OSError`` / ``subprocess.SubprocessError`` through like ``subprocess.run``.
    """
    args = list(args)
    content = _is_content(args)
    if content and git_dir is not None:
        raise GitSafeError("an explicit git dir is only for subcommands that read no content")
    run_env = _base_env(env)
    if content:
        run_env = _blank_filters(workdir, git_bin, run_env, timeout)
    lazy_flag = True
    if content and _sandboxed():
        # Under the sandbox a lazy fetch cannot exec or connect anyway, so an
        # older git without the flag simply runs without it (GIT_NO_LAZY_FETCH
        # stays set). Elsewhere the flag stays and an old git fails closed.
        lazy_flag = _lazy_flag_supported(_trusted_git(), run_env)
    return _launch(git_argv(workdir, args, git_bin, content, lazy_flag, git_dir),
                   run_env, timeout)
