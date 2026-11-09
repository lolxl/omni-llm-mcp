# no-opencc
"""Kernel Seatbelt fence for the codex consult lane and the agent lanes.

Ordinary macOS agent lanes fenced here: claude, pi, cursor, codex and grok,
including read-only workers. Foreman admission remains separately owned. The
agy consult lane (``providers/agy.py``) is fenced with its own policy.

Consult (``run_codex_subprocess``): on Darwin with sandbox-exec, every consult
spawn is wrapped in a macOS Seatbelt profile that denies file writes outside a
small allow-set. After verify+wrap the inner Codex ``--sandbox`` is disabled
so the process applies one Seatbelt (issue #51 nested ``sandbox_apply``).
Unwrapped consult never bypasses.

Agent lanes use the same wrapper, with provider-specific re-allows from
``agent_fence_paths`` / ``agent_fence_policy``.
Dual wrap of Codex is not defence-in-depth: macOS refuses the inner apply
(rc=71).

Read wall (consult lane, ``read_wall=True``): additionally denies
``file-read*`` under the **protected home** (passwd ``pw_dir``, never
inherited ``$HOME``), then re-allows ``~/.codex`` (auth/config), ``workdir``
when it sits under that home, and any declared ``extra_read_paths`` (consult
``files=`` / ``-i`` inputs). Paths outside home stay readable via
``(allow default)`` — the red line this fence pins is home-tree exfiltration
(private data, other repos, ``~/.agents``), matching the live canary design.
A full "deny all reads then allow cwd only" profile aborts process startup
on this host (dyld / shared cache); deny-home is the workable Seatbelt shape.

Doctrine (PR #29): the allow/deny-set must not be redefinable by inherited
environment; verify+apply is one fail-closed operation; flags never overstate.

**every caller-admitted extra that enters an allow clause is the same filesystem object `(st_dev, st_ino)` that admission validated, and no component of its canonical path is a symlink that the agent's uid could have planted.** Workdir and launcher-owned defaults keep their own handling.

Accepted residuals: the checks are not atomic with the kernel's own path walk; an unfenced process running as the same uid can plant anything and does not need this window; a fenced job can write only inside its own allow-set, and the shared vendor state names are leaf-checked on every spawn; inode reuse after delete+recreate, held descriptors and ``st_birthtime`` comparison are deliberately not implemented.
"""
from __future__ import annotations

import contextlib
import errno
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass

from providers.errors import redact_secrets
from providers.logutil import log_event
from providers import vendor_bin

ALLOW_WRITE_LITERALS = ("/dev/null",)
ALLOW_WRITE_REGEX = (r"^/dev/tty", r"^/dev/fd/", r"^/dev/std")

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

ENV_CLAUDE_BIN = "OMNI_LLM_CLAUDE_BIN"
ENV_PI_BIN = "OMNI_LLM_PI_BIN"
ENV_CURSOR_BIN = "OMNI_LLM_CURSOR_BIN"
ENV_AGENT_FENCE = "OMNI_LLM_AGENT_FENCE"
ENV_AGENT_UNFENCED = "OMNI_LLM_AGENT_UNFENCED"

_PROBE_TIMEOUT_S = 5

# Probe script: numeric errno on stdout (locale-independent).
_PROBE_SCRIPT = (
    "import os, sys\n"
    "t = os.environ['OMNI_KSB_TARGET']\n"
    "try:\n"
    "    open(t, 'w').write('x'); print('WROTE')\n"
    "except OSError as e:\n"
    "    print('ERRNO=%d' % e.errno)\n"
)

# Read probe: print OK if the target is readable, else ERRNO.
_READ_PROBE_SCRIPT = (
    "import os, sys\n"
    "t = os.environ['OMNI_KSB_TARGET']\n"
    "try:\n"
    "    os.listdir(t); print('READ_OK')\n"
    "except OSError as e:\n"
    "    print('ERRNO=%d' % e.errno)\n"
)


def available() -> bool:
    """True only on Darwin with ``/usr/bin/sandbox-exec`` present and executable.

    Absolute path, never a PATH lookup: a shadowed ``sandbox-exec`` on PATH
    must not count as the fence being available.
    """
    return (
        platform.system() == "Darwin"
        and os.path.isfile(SANDBOX_EXEC)
        and os.access(SANDBOX_EXEC, os.X_OK)
    )


def protected_home() -> str:
    """Account home from the passwd database — independent of inherited $HOME.

    Seatbelt deny-set and verification probes MUST use this path. Resolving
    home via ``os.path.expanduser("~")`` / ``$HOME`` lets a caller point the
    wall at a decoy directory while the real home stays readable.
    """
    try:
        return os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError, AttributeError) as exc:
        # Fail closed for callers that need a path: raise a clear error rather
        # than silently falling back to $HOME (which is the bug we are fixing).
        raise RuntimeError(
            "cannot resolve protected home from passwd: %s" % redact_secrets(str(exc))
        ) from exc


def codex_config_dir() -> str:
    """Realpath of ``<protected_home>/.codex`` (auth/config), env-independent."""
    return os.path.realpath(os.path.join(protected_home(), ".codex"))


def sanitized_child_env(base=None) -> dict:
    """Copy of *base* (default: ``os.environ``) with HOME set to protected_home.

    Codex (and fence probes) may need a HOME for config lookup; never inherit
    a caller-spoofed HOME that would redirect ``~`` away from the deny target.
    """
    env = dict(os.environ if base is None else base)
    env["HOME"] = protected_home()
    return env


def _path_inside(path: str, root: str) -> bool:
    """True when realpath(path) is root or a descendant of realpath(root)."""
    path_r = os.path.realpath(path)
    root_r = os.path.realpath(root)
    try:
        common = os.path.commonpath([path_r, root_r])
    except ValueError:
        return False
    return common == root_r


def _fold_path(text: str) -> str:
    """A path spelling folded the way APFS and Seatbelt compare it.

    Any case (Seatbelt rules match every case spelling, measured) and any
    Unicode normalization (APFS treats NFC and NFD spellings as one name).
    """
    return unicodedata.normalize(
        "NFD", unicodedata.normalize("NFD", text).casefold())


def branch_name_in(branch, names) -> bool:
    """*branch* is one of *names*, compared folded (case, NFC/NFD).

    A branch is a file name under ``refs/heads`` and APFS treats its
    spellings as one name, so the protected-branch check must too.
    """
    folded = _fold_path(str(branch))
    return any(folded == _fold_path(str(name)) for name in names)


# One "same location" judgement (issue #147). A location is
#   (identity of its nearest EXISTING ancestor-or-self, the components below)
# where identity is (st_dev, st_ino). Case, NFC/NFD, a firmlink spelling
# (/System/Volumes/Data/...), a symlinked root and a path that does not exist
# yet all reduce to that pair, so one judgement serves every caller. Measured
# with the real sandbox-exec (test_same_location.py pins the facts): Seatbelt
# rules match every case and normalization spelling, and compare the
# KERNEL-resolved path of an access, so a rule spelled through a symlink or
# through the firmlink never matches the target while an access spelled that
# way is judged by the target's own rule.
#
# Two directions, never mixed:
#   * REFUSAL ("does this overlap something protected?"): the components below
#     the existing ancestor are folded, and an identity that cannot be
#     established counts as overlap. ``location_inside`` / ``same_location``.
#   * EXEMPTION ("is this the one allowed location?"): the components below the
#     existing ancestor must be spelled exactly, and an identity that cannot be
#     established is "not the same". ``exempt_inside`` / ``exempt_same``.


# Judgements are cheap but not free: each one walks both paths to "/". A
# caller that makes many of them over a fixed filesystem (a policy build, a
# read-allow check against the grok rows) opens ``_location_cache()`` so one
# resolved path is walked once. Outside such a block nothing is cached, so a
# directory created between two calls is always seen.
_LOCATION_STATE = threading.local()


@contextlib.contextmanager
def _location_cache():
    if getattr(_LOCATION_STATE, "cache", None) is not None:
        yield
        return
    _LOCATION_STATE.cache = {}
    try:
        yield
    finally:
        _LOCATION_STATE.cache = None


def _resolve_location(path, resolve_leaf=True):
    """``(resolved, chain)``: the path as the kernel resolves it, plus its walk.

    *chain* is ``[(identity, rest)]`` for each existing ancestor-or-self,
    deepest first; *rest* is the tuple of names below that ancestor. The path
    is NOT normalized before it is resolved: ``link/..`` is the parent of the
    link's TARGET, and ``os.path.realpath`` applies each ``..`` after
    resolving the components before it. Components below the deepest existing
    ancestor therefore cannot be links. ``resolve_leaf=False`` keeps the last
    component as a name (judging a link by its own location). Only ENOENT,
    ENOTDIR and ENAMETOOLONG mean "does not exist"; any other error raises
    ``_IdentityUnavailable``.
    """
    given = os.fspath(path)
    cache = getattr(_LOCATION_STATE, "cache", None)
    try:
        if not os.path.isabs(given):
            given = os.path.join(os.getcwd(), given)
        key = (given, resolve_leaf)
        if cache is not None and key in cache:
            return cache[key]
        below = []
        base = given
        if not resolve_leaf:
            parent, leaf = os.path.split(given)
            if leaf not in ("", ".", ".."):
                base = parent
                below.append(leaf)
        cur = os.path.realpath(base)
    except OSError as exc:
        raise _IdentityUnavailable(given) from exc
    resolved = os.path.join(cur, *below)
    chain = []
    while True:
        try:
            if cache is not None and ("id", cur) in cache:
                ident = cache[("id", cur)]
            else:
                ident = _stat_dev_ino(cur)
                if cache is not None:
                    cache[("id", cur)] = ident
        except OSError as exc:
            raise _IdentityUnavailable(cur) from exc
        if ident is not None:
            chain.append((ident, tuple(below)))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        below.insert(0, os.path.basename(cur))
        cur = parent
    if not chain:
        raise _IdentityUnavailable(given)
    result = (resolved, chain)
    if cache is not None:
        cache[key] = result
    return result


def _location_chain(path, resolve_leaf=True):
    return _resolve_location(path, resolve_leaf)[1]


def _location_inside_core(path, root, *, fold, resolve_leaf=True):
    """``path`` is ``root`` or below it, as one (identity, components) judgement.

    Exemptions (``fold=False``) accept the same inode with exactly spelled
    names below it. Refusals (``fold=True``) accept that too, folded, and
    also the folded spelling of the two resolved paths: Seatbelt matches
    rules case-insensitively even where the volume does not (measured on
    case-sensitive APFS: a rule for ``HOME`` applies to the distinct
    directory ``home``), so two existing directories whose names fold equal
    are one location to a rule. Raises ``_IdentityUnavailable``; the public
    wrappers choose the side.
    """
    root_resolved, root_chain = _resolve_location(root)
    path_resolved, path_chain = _resolve_location(path, resolve_leaf)
    anchor, tail = root_chain[0]
    if fold:
        tail = tuple(_fold_path(part) for part in tail)
    for ident, rest in path_chain:
        if ident != anchor or len(rest) < len(tail):
            continue
        if fold:
            rest = tuple(_fold_path(part) for part in rest)
        if rest[:len(tail)] == tail:
            return True
    return bool(fold and _lexical_ci_same_or_inside(path_resolved, root_resolved))


def location_inside(path, root, *, resolve_leaf=True) -> bool:
    """Refusal side: *path* is *root* or under it in any spelling.

    Folds case and NFC/NFD below the nearest existing ancestor. An identity
    that cannot be established counts as overlap, and the result must never
    be used to admit a path.
    """
    try:
        return _location_inside_core(path, root, fold=True,
                                     resolve_leaf=resolve_leaf)
    except _IdentityUnavailable:
        return True


def same_location(a, b) -> bool:
    """Refusal side: *a* and *b* are one location in any spelling."""
    return location_inside(a, b) and location_inside(b, a)


def exempt_inside(path, root) -> bool:
    """Exemption side: *path* is *root* or under it, spelled exactly.

    The same inode counts (an existing ancestor is compared by identity),
    names below it must match character for character: two folded-equal
    names are two directories on a case-sensitive volume, and an exemption
    decided by folding would widen access. An identity error is "no".
    """
    try:
        return _location_inside_core(path, root, fold=False)
    except _IdentityUnavailable:
        return False


def exempt_same(a, b) -> bool:
    """Exemption side: *a* and *b* are one location, spelled exactly."""
    return exempt_inside(a, b) and exempt_inside(b, a)


def _path_inside_any_case(path: str, root: str) -> bool:
    """*path* is *root* or under it, in any spelling the filesystem accepts.

    Refusal-side alias of ``location_inside``; kept for the callers and
    tests that already use this name.
    """
    return location_inside(path, root)


def _same_path_any_case(a: str, b: str) -> bool:
    """Refusal-side equality: either spelling nests in the other."""
    return same_location(a, b)


def _same_dir_exact(a, b) -> bool:
    """Exemption-side equality: the same spelling or the same inode only.

    A path that does not exist yet is compared by its nearest existing
    ancestor plus the exact remaining names; it is never folded, and an
    identity error is "not the same".
    """
    return a == b or exempt_same(a, b)


def _system_tempdir_default() -> str:
    """Platform temp dir ignoring inherited TMPDIR/TEMP/TMP (and cache)."""
    keys = ("TMPDIR", "TEMP", "TMP")
    saved = {k: os.environ.pop(k, None) for k in keys}
    old_cache = tempfile.tempdir
    try:
        tempfile.tempdir = None
        return os.path.realpath(tempfile.gettempdir())
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        tempfile.tempdir = old_cache


def _approved_temp_roots() -> tuple[str, ...]:
    """Realpaths of approved temp roots (existing only)."""
    roots: list[str] = []
    seen: set[str] = set()
    for raw in (
        "/private/var/folders",
        _system_tempdir_default(),
        "/private/tmp",
    ):
        rp = os.path.realpath(raw)
        if rp in seen:
            continue
        seen.add(rp)
        if os.path.exists(rp):
            roots.append(rp)
    return tuple(roots)


def _tmpdir_rejected() -> str | None:
    """If TMPDIR is set and outside approved roots, return its realpath; else None."""
    raw = os.environ.get("TMPDIR")
    if not raw:
        return None
    rp = os.path.realpath(raw)
    if any(_path_inside(rp, root) for root in _approved_temp_roots()):
        return None
    return rp


def _codex_home_rejected() -> str | None:
    """If ``<protected_home>/.codex`` is outside protected home, return it."""
    home = protected_home()
    codex = codex_config_dir()
    if _path_inside(codex, home):
        return None
    return codex


class FencePathRejected(Exception):
    """Last-resolver refusal: an extra is not the admitted filesystem object."""

    def __init__(self, detail: str):
        self.detail = redact_secrets(str(detail))[:300]
        super().__init__(self.detail)


@dataclass(frozen=True)
class AdmittedPath:
    """A path admitted into the fence with its filesystem identity.

    ``path`` is canonical (the result of the one realpath admission).
    ``kind`` is ``subpath`` (directory) or ``literal`` (regular file).
    ``dev`` / ``ino`` are ``st_dev`` / ``st_ino`` of that object at admission.
    """

    path: str
    dev: int
    ino: int
    kind: str

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def __eq__(self, other):
        if isinstance(other, AdmittedPath):
            return (self.path, self.dev, self.ino, self.kind) == (
                other.path, other.dev, other.ino, other.kind,
            )
        if isinstance(other, str):
            return self.path == other
        return NotImplemented

    def __hash__(self):
        return hash((self.path, self.dev, self.ino, self.kind))


def _plantable_link_ancestor(canonical) -> str | None:
    """Return a plantable symlink prefix of *canonical*, else None.

    Walks every proper prefix from ``/`` down. A prefix that is a symlink
    is tolerated ONLY if its parent directory is owned by uid 0
    (``os.stat(parent).st_uid == 0``) AND ``os.access(parent, os.W_OK)``
    is False. Ownership is the primary test: a planter can ``chmod 0555``
    its own directory afterwards, and ``os.access`` alone can lie on
    network/FUSE volumes. Any ``OSError`` in that decision returns the
    prefix (fail closed). Root-owned ``/var``, ``/tmp``, ``/etc`` pass;
    anything whose parent is not uid 0 fails even if it is currently
    not writable.
    """
    path = os.fspath(canonical)
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    prefixes: list[str] = []
    cur = path
    while True:
        parent = os.path.dirname(cur.rstrip(os.sep) or cur)
        if parent == cur:
            break
        prefixes.append(parent)
        cur = parent
    for prefix in reversed(prefixes):
        if prefix in ("", "/"):
            continue
        try:
            st_prefix = os.lstat(prefix)
        except OSError:
            return prefix
        if not stat.S_ISLNK(st_prefix.st_mode):
            continue
        parent = os.path.dirname(prefix)
        try:
            owned_by_root = os.stat(parent).st_uid == 0
            writable = os.access(parent, os.W_OK)
        except OSError:
            return prefix
        if owned_by_root and not writable:
            continue
        return prefix
    return None


def _true_ancestor_of(prefix: str, canonical: str) -> bool:
    """True when realpath(*prefix*) is *canonical* or an ancestor of it."""
    try:
        prefix_real = os.path.realpath(prefix)
    except OSError:
        return False
    return _path_inside(canonical, prefix_real)


def admit_path(raw, kind, *, label) -> AdmittedPath:
    """Single evaluation that admits *raw* as *kind* (``subpath`` or ``literal``).

    ``given = os.fspath(raw)``; ``os.lstat(given)`` FIRST — a symlink at
    the admitted spelling is refused before any ``realpath``/``stat``.
    Then ``now = os.path.realpath(given)``; ``st = os.stat(now)``
    (follow). *kind* ``None`` infers ``subpath`` vs ``literal`` from
    *st* (one evaluation for a bare-string extra). Raises
    ``FencePathRejected`` if the admitted spelling itself is a symlink
    (the caller must hand over the target it validated, never the link),
    if the object kind does not match *kind* (directory for
    ``subpath``, regular file for ``literal``), if a plantable symlink
    ancestor sits on *now* or on the prefixes of *given* (both kinds;
    when that spelling still names a true ancestor of *now*), or if the
    path is missing. Launcher-owned defaults keep their own existence
    handling; a caller extra that does not exist is not admitted.
    """
    given = os.fspath(raw)
    try:
        lst = os.lstat(given)
    except OSError as exc:
        raise FencePathRejected(
            "fence path rejected (%s does not exist): %s"
            % (label, given)
        ) from exc
    if stat.S_ISLNK(lst.st_mode):
        raise FencePathRejected(
            "fence path rejected (%s is a symlink; pass the validated "
            "target, not the link): %s"
            % (label, given)
        )
    try:
        now = os.path.realpath(given)
    except OSError as exc:
        raise FencePathRejected(
            "fence path rejected (%s could not be resolved): %s (%s)"
            % (label, given, exc)
        ) from exc
    try:
        st = os.stat(now)
    except OSError as exc:
        raise FencePathRejected(
            "fence path rejected (%s does not exist): %s -> %s"
            % (label, given, now)
        ) from exc
    if kind is None:
        if stat.S_ISDIR(st.st_mode):
            kind = "subpath"
        elif stat.S_ISREG(st.st_mode):
            kind = "literal"
        else:
            raise FencePathRejected(
                "fence path rejected (%s is not a file or directory): %s"
                % (label, now)
            )
    elif kind == "subpath":
        if not stat.S_ISDIR(st.st_mode):
            raise FencePathRejected(
                "fence path rejected (%s is not a directory): %s" % (label, now)
            )
    elif kind == "literal":
        if not stat.S_ISREG(st.st_mode):
            raise FencePathRejected(
                "fence path rejected (%s is not a regular file): %s"
                % (label, now)
            )
    else:
        raise FencePathRejected(
            "fence path rejected (%s has unknown kind %r): %s"
            % (label, kind, now)
        )
    planted = _plantable_link_ancestor(now)
    # A path reached through a plantable symlink parent is refused even
    # though realpath erased that parent from *now*. Walk the given
    # spelling for both kinds (a subpath CLAUDE_CONFIG_DIR through
    # /tmp/job/link/profile would otherwise lose the plantable *link*).
    if planted is None and given != now:
        planted_given = _plantable_link_ancestor(given)
        if planted_given is not None and _true_ancestor_of(planted_given, now):
            planted = planted_given
    if planted is not None:
        raise FencePathRejected(
            "fence path rejected (%s has a plantable symlink ancestor "
            "%s): %s -> %s" % (label, os.path.basename(planted), given, now)
        )
    return AdmittedPath(path=now, dev=st.st_dev, ino=st.st_ino, kind=kind)


def _confirm_admitted(ap: AdmittedPath, *, label: str) -> None:
    """Refuse unless *ap* is still the same object, with no plantable ancestor.

    ``st = os.stat(ap.path)`` (follow). Raises ``FencePathRejected`` if
    the identity differs, if ``ap.path`` is now a symlink, if ``ap.kind``
    no longer matches ``S_ISDIR`` / ``S_ISREG`` of the confirmed object,
    or if a plantable symlink ancestor appeared on the canonical spelling.
    """
    now = os.path.realpath(ap.path)
    if os.path.islink(ap.path):
        raise FencePathRejected(
            "fence path rejected (%s changed between admission and profile "
            "build): %s -> %s" % (label, ap.path, now)
        )
    try:
        st = os.stat(ap.path)
    except OSError as exc:
        raise FencePathRejected(
            "fence path rejected (%s changed between admission and profile "
            "build): %s -> %s" % (label, ap.path, now)
        ) from exc
    if (st.st_dev, st.st_ino) != (ap.dev, ap.ino):
        raise FencePathRejected(
            "fence path rejected (%s changed between admission and profile "
            "build): %s -> %s" % (label, ap.path, now)
        )
    if ap.kind == "subpath":
        kind_ok = stat.S_ISDIR(st.st_mode)
    elif ap.kind == "literal":
        kind_ok = stat.S_ISREG(st.st_mode)
    else:
        kind_ok = False
    if not kind_ok:
        raise FencePathRejected(
            "fence path rejected (%s changed between admission and profile "
            "build): %s -> %s" % (label, ap.path, now)
        )
    planted = _plantable_link_ancestor(ap.path)
    if planted is not None:
        raise FencePathRejected(
            "fence path rejected (%s has a plantable symlink ancestor "
            "%s): %s" % (label, planted, ap.path)
        )


def _coerce_admitted(item, *, label: str, write: bool = False) -> AdmittedPath:
    """Return an ``AdmittedPath``. A plain ``str`` means admit now.

    A production caller that validated earlier MUST pass an
    ``AdmittedPath``. A ``str`` extra is admitted at this moment (one
    ``admit_path`` call, so no separate-validation race exists for it).
    Write extras are ``subpath``. Read extras infer kind from the object
    inside that single admission. An ``AdmittedPath`` instance is
    returned as-is; the last resolver confirms identity and kind.
    """
    if isinstance(item, AdmittedPath):
        return item
    given = os.fspath(item)
    if write:
        return admit_path(given, "subpath", label=label)
    return admit_path(given, None, label=label)


def _write_allow_paths(workdir, read_only, extra_write_paths=None) -> tuple[str, ...]:
    """Realpath-resolved subpaths that may be written.

    Always: ``<protected_home>/.codex`` (only if inside protected home),
    approved temp roots / a validated TMPDIR, and ``/private/var/folders``
    when present. Plus ``workdir`` when ``read_only`` is False and
    ``workdir`` is not None. ``workdir`` and those launcher-owned entries
    keep today's handling (realpath here, drop if missing): ``workdir`` is
    already realpath-normalized by the agent lane at admission, and the
    consult ``tmp_cwd`` is launcher-owned. ``extra_write_paths`` items are
    ``AdmittedPath`` objects (or a plain ``str``, which means admit now);
    the last resolver checks identity, not spelling, and refuses LOUD if
    the object changed or a plantable symlink ancestor appeared. The allow
    entry is ``ap.path`` with ``ap.kind`` (never re-derived from
    ``isdir``). Caller-controlled TMPDIR outside the approved temp roots
    is never added.
    """
    candidates: list[str] = []

    if _codex_home_rejected() is None:
        candidates.append(codex_config_dir())

    # Temp allow-set: validated TMPDIR when set; otherwise system default.
    # Always include /private/var/folders (historical allow-set member).
    # /private/tmp is an approved *root for TMPDIR validation*, not an
    # unconditional allow entry.
    rejected = _tmpdir_rejected()
    if rejected is None:
        tmp_raw = os.environ.get("TMPDIR")
        if tmp_raw:
            candidates.append(os.path.realpath(tmp_raw))
        else:
            candidates.append(_system_tempdir_default())

    candidates.append(os.path.realpath("/private/var/folders"))

    if workdir is not None and not read_only:
        candidates.append(os.path.realpath(str(workdir)))

    extras: list[str] = []
    for raw in extra_write_paths or ():
        ap = _coerce_admitted(raw, label="extra_write_paths", write=True)
        _confirm_admitted(ap, label="extra_write_paths")
        extras.append(ap.path)

    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            out.append(path)
    for path in extras:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return tuple(out)


def _read_reallow_entries(workdir, extra_read_paths=None) -> tuple[tuple[str, str], ...]:
    """``(filter_kind, realpath)`` pairs re-allowed under the home read deny.

    ``filter_kind`` is ``subpath`` (directories) or ``literal`` (files).
    Only paths whose realpath sits under the protected home need re-allow —
    everything outside home remains readable via ``(allow default)``.
    ``workdir`` and the launcher-owned ``<protected_home>/.codex`` keep
    today's handling (realpath here, drop if missing). ``extra_read_paths``
    items are ``AdmittedPath`` objects (or a plain ``str``, which means
    admit now); the last resolver checks identity, not spelling, and
    refuses LOUD if the object changed or a plantable symlink ancestor
    appeared. The allow entry is ``ap.path`` with ``ap.kind`` (never
    re-derived from ``isdir``). Caller TMPDIR is not auto-added here
    (temp trees are outside home on macOS).
    """
    home = protected_home()
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, path: str) -> None:
        rp = os.path.realpath(path)
        key = (kind, rp)
        if key in seen:
            return
        if not _path_inside(rp, home):
            return
        if kind == "subpath":
            if not os.path.isdir(rp):
                return
        else:
            if not os.path.isfile(rp):
                return
        seen.add(key)
        entries.append((kind, rp))

    def _add_admitted(ap: AdmittedPath) -> None:
        key = (ap.kind, ap.path)
        if key in seen:
            return
        if not _path_inside(ap.path, home):
            return
        seen.add(key)
        entries.append((ap.kind, ap.path))

    if _codex_home_rejected() is None:
        codex = codex_config_dir()
        if os.path.isdir(codex):
            _add("subpath", codex)

    if workdir is not None:
        _add("subpath", str(workdir))

    for raw in extra_read_paths or ():
        ap = _coerce_admitted(raw, label="extra_read_paths", write=False)
        _confirm_admitted(ap, label="extra_read_paths")
        _add_admitted(ap)

    return tuple(entries)


def _metadata_ancestor_paths(home: str, path: str) -> list[str]:
    """Directories strictly between *home* (inclusive) and *path* (exclusive).

    realpath(3) lstats every path component on the way down; the read wall
    denies file-read* (metadata included) on home and everything under it,
    and the R*/S* re-allow clauses only cover the LEAF path, never its
    ancestors — so a re-allowed path several directories deep stays
    unresolvable by realpath even though it is itself fully readable
    (open/stat/readdir all succeed; only realpath's own ancestor-walk
    fails). This returns every ancestor that needs a metadata-only re-allow,
    home itself included, the leaf itself excluded (the leaf already has
    full read access from its own R*/S* entry).
    """
    home_rp = os.path.realpath(home)
    path_rp = os.path.realpath(path)
    if not _path_inside(path_rp, home_rp) or path_rp == home_rp:
        return []
    rel = os.path.relpath(path_rp, home_rp)
    parts = rel.split(os.sep)
    out = [home_rp]
    cur = home_rp
    for part in parts[:-1]:
        cur = os.path.join(cur, part)
        out.append(cur)
    return out


def _metadata_reallow_entries(workdir, extra_read_paths=None,
                              extra_post_read=None) -> tuple[str, ...]:
    """Deduplicated, sorted ancestor literals needed for realpath(3) to work
    on every R* (``_read_reallow_entries``) and S* (``extra_post_read``)
    re-allowed read path. Metadata only, literal only, never a directory
    outside home — see ``_metadata_ancestor_paths``.
    """
    home = protected_home()
    seen: set[str] = set()
    out: list[str] = []
    targets = [path for _kind, path in _read_reallow_entries(workdir, extra_read_paths)]
    targets += [path for _kind, path in (extra_post_read or ())]
    for target in targets:
        for anc in _metadata_ancestor_paths(home, target):
            if anc not in seen:
                seen.add(anc)
                out.append(anc)
    out.sort()
    return tuple(out)


def profile_params(workdir, read_only, read_wall=False, extra_read_paths=None,
                   extra_write_paths=None, extra_deny_write=None,
                   extra_post_write=None, extra_deny_read=None,
                   extra_post_read=None, extra_deny_link=None,
                   extra_deny_write_final=None, extra_deny_exec=None,
                   extra_deny_read_final=None,
                   extra_deny_connect=None) -> dict[str, str]:
    """Map profile ``-D`` params: write ``P*``, optional home ``H0`` + read ``R*``.

    Order matches ``build_profile`` / ``wrap_argv``. Literals and regexes for
    write allows are fixed constants in the profile text (not params). When
    ``read_wall`` is False, only ``P0..Pn`` are returned (byte-stable with the
    pre-#45 agent-lane shape). ``H0`` is always ``protected_home()``, never
    the inherited ``$HOME``. Extra deny / post-reallow params (``W*`` ``T*``
    ``X*`` ``S*``) are omitted when those lists are empty, so consult-lane
    callers stay byte-identical. ``L*`` (``file-link`` deny) is omitted the
    same way. ``M*`` (metadata-only ancestor re-allow for
    realpath(3), see ``_metadata_reallow_entries``) is omitted only when
    there are no R*/S* re-allow entries at all — a bare read wall with
    nothing re-allowed needs no ancestor metadata either.
    ``Q0`` binds the canonical read-only workdir for the final write deny,
    regardless of whether a containing temp/runtime root is writable.
    """
    paths = _write_allow_paths(workdir, read_only, extra_write_paths=extra_write_paths)
    params: dict[str, str] = {"P%d" % i: path for i, path in enumerate(paths)}
    if read_wall:
        params["H0"] = protected_home()
        for i, (_kind, path) in enumerate(
            _read_reallow_entries(workdir, extra_read_paths)
        ):
            params["R%d" % i] = path
        for i, path in enumerate(
            _metadata_reallow_entries(workdir, extra_read_paths, extra_post_read)
        ):
            params["M%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_write or ()):
        params["W%d" % i] = path
    for i, (kind, path) in enumerate(extra_post_write or ()):
        params["T%d" % i] = (
            _sbpl_regex_quote(path) if kind == GROK_TMP_KIND else path)
    for i, (_kind, path) in enumerate(extra_deny_read or ()):
        params["X%d" % i] = path
    for i, (_kind, path) in enumerate(extra_post_read or ()):
        params["S%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_link or ()):
        params["L%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_write_final or ()):
        params["F%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_exec or ()):
        params["E%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_read_final or ()):
        params["Y%d" % i] = path
    for i, (_kind, path) in enumerate(extra_deny_connect or ()):
        params["N%d" % i] = path
    for i, (_kind, path) in enumerate(connect_deny_write_entries(extra_deny_connect)):
        params["V%d" % i] = path
    if read_only and workdir is not None:
        params["Q0"] = os.path.realpath(str(workdir))
    return params


def _clause_for_entries(op, action, prefix, entries) -> str:
    """One ``(op action (kind (param PREFIX*)))`` clause, or empty.

    A ``GROK_TMP_KIND`` entry's param is its regex-quoted root; the filter
    adds ``/.<pid>.<seq>.tmp`` with the pid from ``GROK_PID_PARAM``.
    """
    if not entries:
        return ""
    filters = [
        '(regex (string-append "^" (param "%s%d") "/\\\\." (param "%s") '
        '"\\\\.[0-9]+\\\\.tmp$"))' % (prefix, i, GROK_PID_PARAM)
        if kind == GROK_TMP_KIND else
        '(%s (param "%s%d"))' % (kind, prefix, i)
        for i, (kind, _path) in enumerate(entries)
    ]
    return "(%s %s %s)" % (op, action, " ".join(filters))


def build_profile(workdir, read_only, read_wall=False, extra_read_paths=None,
                  extra_write_paths=None, extra_deny_write=None,
                  extra_post_write=None, extra_deny_read=None,
                  extra_post_read=None, extra_deny_link=None,
                  extra_deny_write_final=None, extra_deny_exec=None,
                  extra_deny_read_final=None, extra_deny_connect=None) -> str:
    """Inline Seatbelt profile text.

    Write wall order (always): ``(allow default)`` then ``(deny file-write*)``
    then one ``(allow file-write* ...)`` with every subpath (as params),
    literal, and regex. Paths are never interpolated into the profile text.

    Read wall (when ``read_wall``): after the write clauses,
    ``(deny file-read* (subpath H0))`` then ``(allow file-read* ...)`` for
    re-allow entries (``R*`` params as subpath or literal). Seatbelt is
    last-match-wins.

    Optional later clauses (agent fence only; omitted when empty so consult
    profiles stay byte-identical): deny-write ``W*``, post-deny write
    re-allow ``T*``, deny-read ``X*``, post-deny read re-allow ``S*``. Last,
    when there is at least one R*/S* re-allow entry: one metadata-only,
    literal-only re-allow ``M*`` for every ancestor directory between home
    and each re-allowed leaf, so realpath(3)'s own per-component lstat walk
    succeeds (see ``_metadata_reallow_entries``). This clause is NOT agent-
    fence-only — a bare consult profile with any home-relative re-allow
    (e.g. ``~/.codex``) gets it too, since realpath is a general problem,
    not a claude/pi/cursor-specific one. ``L*`` is a ``file-link`` deny
    (not ``file-write*``); it is emitted after all allows. A read-only
    workdir adds the final ``Q0`` write deny, which also overrides any
    containing temp/runtime allow-root. ``N*`` is an outbound-connect deny
    for one unix socket (the ssh-agent socket, see
    ``ssh_agent_socket_denies``) and ``V*`` a write deny on that socket and
    each of its ancestor directories (``connect_deny_write_entries``), so it
    cannot be renamed into a name the connect deny does not cover; both are
    omitted when empty.
    """
    write_paths = _write_allow_paths(
        workdir, read_only, extra_write_paths=extra_write_paths,
    )
    filters: list[str] = []
    for i in range(len(write_paths)):
        filters.append('(subpath (param "P%d"))' % i)
    for lit in ALLOW_WRITE_LITERALS:
        filters.append('(literal "%s")' % lit)
    for rx in ALLOW_WRITE_REGEX:
        filters.append('(regex "%s")' % rx)

    allow_clause = "(allow file-write* %s)" % " ".join(filters)
    profile = (
        "(version 1)"
        "(allow default)"
        "(deny file-write*)"
        + allow_clause
    )
    profile += _clause_for_entries(
        "deny", "file-write*", "W", extra_deny_write or (),
    )
    profile += _clause_for_entries(
        "allow", "file-write*", "T", extra_post_write or (),
    )
    if read_wall:
        profile += '(deny file-read* (subpath (param "H0")))'
        reallows = _read_reallow_entries(workdir, extra_read_paths)
        if reallows:
            r_filters: list[str] = []
            for i, (kind, _path) in enumerate(reallows):
                r_filters.append('(%s (param "R%d"))' % (kind, i))
            profile += "(allow file-read* %s)" % " ".join(r_filters)
        profile += _clause_for_entries(
            "deny", "file-read*", "X", extra_deny_read or (),
        )
        profile += _clause_for_entries(
            "allow", "file-read*", "S", extra_post_read or (),
        )
        # realpath(3) lstats every ancestor directory on the way down to a
        # re-allowed leaf; the H0 deny blocks metadata on every one of them
        # and the R*/S* clauses above only cover the leaf itself. One
        # metadata-only, literal-only re-allow (placed AFTER S* so a leaf
        # explicitly denied by X* stays denied even though an ancestor it
        # shares with an allowed leaf becomes metadata-readable) fixes
        # realpath without granting content access to anything new.
        meta_entries = _metadata_reallow_entries(
            workdir, extra_read_paths, extra_post_read,
        )
        if meta_entries:
            m_filters = [
                '(literal (param "M%d"))' % i for i in range(len(meta_entries))
            ]
            profile += "(allow file-read-metadata %s)" % " ".join(m_filters)
    # Final write deny (``F*``): emitted after the read wall as well, so it
    # follows every allow row of the profile, not just the write allows.
    profile += _clause_for_entries(
        "deny", "file-write*", "F", extra_deny_write_final or (),
    )
    # Final read deny (``Y*``): after EVERY read allow (R*, S*, M*), so a
    # sibling account home that sits inside an allowed config root stays
    # unreadable. Seatbelt is last-match-wins; the earlier X* deny is
    # followed by the S* re-allow of the containing root and loses to it.
    profile += _clause_for_entries(
        "deny", "file-read*", "Y", extra_deny_read_final or (),
    )
    # After every allow, including (allow default). link(2) is gated by
    # file-write* AND file-link on both endpoints; this clause is an
    # independent second gate that does not rely on that coupling; it is
    # emitted last so an earlier (allow default) cannot undo it.
    profile += _clause_for_entries(
        "deny", "file-link", "L", extra_deny_link or (),
    )
    profile += _clause_for_entries(
        "deny", "process-exec*", "E", extra_deny_exec or (),
    )
    # Outbound connect to a unix socket (``N*``): one clause per entry, the
    # socket compared by the path the kernel resolved the connect to.
    for i in range(len(extra_deny_connect or ())):
        profile += ('(deny network-outbound (remote unix-socket '
                    '(path-literal (param "N%d"))))' % i)
    profile += _clause_for_entries(
        "deny", "file-write*", "V", connect_deny_write_entries(extra_deny_connect),
    )
    if read_only and workdir is not None:
        # Omitting cwd from P* is insufficient when an allowed temp/vendor
        # root contains it. The explicit RO contract wins after every allow,
        # including runtime T* entries; state outside cwd is unaffected.
        profile += '(deny file-write* (subpath (param "Q0")))'
    return profile


def _param_sort_key(key: str) -> tuple:
    """Sort -D params: P*, then H*, then R* by numeric suffix."""
    prefix = key[0]
    try:
        n = int(key[1:])
    except ValueError:
        n = 0
    order = {"P": 0, "H": 1, "R": 2, "W": 3, "T": 4, "F": 5, "X": 6, "S": 7,
             "M": 8, "L": 9, "E": 10, "Q": 11, "Y": 12, "N": 13, "V": 14}.get(prefix, 15)
    return (order, n)


def wrap_argv(argv, workdir, read_only, read_wall=False, extra_read_paths=None,
              extra_write_paths=None, extra_deny_write=None,
              extra_post_write=None, extra_deny_read=None,
              extra_post_read=None, extra_deny_link=None,
              extra_deny_write_final=None, extra_deny_exec=None,
              extra_deny_read_final=None,
              extra_deny_connect=None) -> list[str]:
    """Prefix *argv* with sandbox-exec -p <profile> -D ... when available.

    If ``available()`` is False, return *argv* unchanged (same object).
    The executable is ``/usr/bin/sandbox-exec`` when that file exists;
    ``shutil.which`` is only a fallback for hermetic tests that patch
    ``available()`` True on a host without the binary.

    Extra read/write paths pass through unchanged to ``build_profile``.
    ``FencePathRejected`` from the last resolver propagates (callers map
    it); this function does not catch it.
    """
    if not available():
        return argv
    if os.path.isfile(SANDBOX_EXEC) and os.access(SANDBOX_EXEC, os.X_OK):
        exe = SANDBOX_EXEC
    else:
        exe = shutil.which("sandbox-exec")
    if not exe:
        return argv
    extra = dict(
        extra_read_paths=extra_read_paths,
        extra_write_paths=extra_write_paths,
        extra_deny_write=extra_deny_write,
        extra_post_write=extra_post_write,
        extra_deny_read=extra_deny_read,
        extra_post_read=extra_post_read,
        extra_deny_link=extra_deny_link,
        extra_deny_write_final=extra_deny_write_final,
        extra_deny_exec=extra_deny_exec,
        extra_deny_read_final=extra_deny_read_final,
        extra_deny_connect=extra_deny_connect,
    )
    profile = build_profile(
        workdir, read_only, read_wall=read_wall, **extra,
    )
    params = profile_params(
        workdir, read_only, read_wall=read_wall, **extra,
    )
    out = [exe, "-p", profile]
    if any(kind == GROK_TMP_KIND for kind, _path in extra_post_write or ()):
        # Right after the profile, where GROK_PID_SHIM expects it.
        out.extend(["-D", "%s=%s" % (GROK_PID_PARAM, GROK_PID_PLACEHOLDER)])
    for key in sorted(params, key=_param_sort_key):
        out.extend(["-D", "%s=%s" % (key, params[key])])
    out.extend(list(argv))
    return out


def is_wrapped(argv) -> bool:
    """True iff *argv* is non-empty and argv[0] is sandbox-exec.

    Compares against ``shutil.which("sandbox-exec")`` and also accepts any
    absolute path whose basename is ``sandbox-exec`` (PATH-churn safe).
    """
    if not argv:
        return False
    first = argv[0]
    if not isinstance(first, str) or not first:
        return False
    which = shutil.which("sandbox-exec")
    if which is not None and first == which:
        return True
    return os.path.isabs(first) and os.path.basename(first) == "sandbox-exec"


INNER_SANDBOX_BYPASS = "--dangerously-bypass-approvals-and-sandbox"


def _is_inner_sandbox_eq_flag(tok) -> bool:
    """True for ``--sandbox=...`` or ``-s=...`` (not ``--skip-*``)."""
    if not isinstance(tok, str):
        return False
    if tok.startswith("--sandbox="):
        return True
    return tok.startswith("-s=") and not tok.startswith("--")


def inner_vendor_sandbox_present(argv) -> bool:
    """True if a vendor ``--sandbox`` / ``-s`` flag remains in *argv*."""
    for tok in argv:
        if tok in ("--sandbox", "-s"):
            return True
        if _is_inner_sandbox_eq_flag(tok):
            return True
    return False


def disable_inner_codex_sandbox(argv):
    """Drop inner ``--sandbox`` / ``-s`` and add the vendor bypass flag.

    Isolation is the outer kernel wrap. Codex's own Seatbelt (``--sandbox
    read-only`` / ``workspace-write``, or the default when the flag is
    omitted) is a second ``sandbox_apply`` and is refused on Darwin with
    rc=71 (issue #51). Both codex lanes call this: consult
    (``providers/codex.py``) and the agent write lane
    (``providers/agent_lane.py``).

    Returns None when *argv* is not already wrapped -- the bypass token is
    never emitted on an unwrapped argv, whichever lane asks.

    The bypass is inserted after the ``exec`` subcommand, which keeps it a
    flag of ``exec`` on both spellings: ``codex exec ...`` and the job
    lane's ``codex exec resume <id>``.
    """
    if not is_wrapped(argv):
        return None
    out = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok in ("--sandbox", "-s") and i + 1 < n:
            i += 2
            continue
        if _is_inner_sandbox_eq_flag(tok):
            i += 1
            continue
        if tok == INNER_SANDBOX_BYPASS:
            i += 1
            continue
        out.append(tok)
        i += 1
    try:
        insert_at = out.index("exec") + 1
    except ValueError:
        insert_at = len(out)
    out.insert(insert_at, INNER_SANDBOX_BYPASS)
    return out


def inner_sandbox_disable_failed(argv) -> bool:
    """True when *argv* is not a proven wrapped-and-disabled codex argv.

    The three conjuncts the review asked for as CODE, not a comment: the
    inner vendor sandbox is gone, the bypass token is present, and the argv
    is still wrapped. Callers refuse the spawn when this is True.
    """
    return (
        argv is None
        or inner_vendor_sandbox_present(argv)
        or INNER_SANDBOX_BYPASS not in argv
        or not is_wrapped(argv)
    )


def _refusal(detail: str) -> dict:
    return {
        "ok": False,
        "reason": "kernel_fence_unverified",
        "detail": detail,
    }


def _run_probe(argv, env) -> subprocess.CompletedProcess | dict:
    """Run probe argv; return CompletedProcess or a refusal dict on launch error."""
    # Probe code must not be affected by caller supplied Python hooks/modules.
    # ``-I`` ignores environment and user site; ``-S`` disables site imports.
    isolated_env = dict(env)
    for key in tuple(isolated_env):
        if key.startswith("PYTHON"):
            isolated_env.pop(key, None)
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env=isolated_env,
        )
    except subprocess.TimeoutExpired:
        return _refusal(
            "kernel fence probe timed out after %ds" % _PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _refusal(
            "kernel fence probe failed to launch: %s" % redact_secrets(str(exc))[:200]
        )


def _parse_probe_stdout(stdout: str) -> str | None:
    """Return 'WROTE', 'ERRNO=<n>', or None if neither token is present."""
    text = (stdout or "").strip()
    if text == "WROTE" or (text.startswith("ERRNO=") and text[6:].isdigit()):
        return text
    return None


def _parse_read_probe_stdout(stdout: str) -> str | None:
    """Return 'READ_OK', 'ERRNO=<n>', or None."""
    text = (stdout or "").strip()
    if text == "READ_OK" or (text.startswith("ERRNO=") and text[6:].isdigit()):
        return text
    return None


def verify_fence(workdir=None, read_only=True, read_wall=False,
                 extra_read_paths=None, extra_write_paths=None,
                 extra_deny_write=None, extra_post_write=None,
                 extra_deny_read=None, extra_post_read=None,
                 extra_deny_link=None,
                 extra_deny_write_final=None, extra_deny_exec=None,
                 extra_deny_read_final=None) -> dict | None:
    """Deterministic pre-spawn probe. None only if required postconditions hold.

    Write proof (always):
      1. Positive control (unwrapped): the probe location must accept a write.
      2. Baseline fenced run: the same write must fail with errno EPERM (1)
         under the shared defaults, without a probe-specific deny.
      3. Composed fenced run: caller-specific policy additions must still deny
         the protected target.

    Read proof (when ``read_wall``):
      4. Positive control (unwrapped): the protected home directory can be
         listed.
      5. Baseline and 6. composed fenced runs must both fail with EPERM (1).

    ``workdir=None`` means the read_only allow-set only (no workdir write) —
    the strictest consult-shaped probe.

    Never raises. ``FencePathRejected`` from the last resolver is mapped to
    the refusal envelope ``{"ok": False, "reason": "fence_path_rejected",
    "detail": exc.detail}``. Caller paths are validated first. The probe
    checks shared defaults in isolation and the composed caller profile so
    provider-specific re-allows cannot expose a protected target.
    """
    target = None
    read_target = None
    try:
        if not available():
            return _refusal("sandbox-exec is not available on this host")

        # Fail closed on untrusted allow-roots before any probe runs.
        rejected_tmp = _tmpdir_rejected()
        if rejected_tmp is not None:
            return _refusal(
                "rejected write allow-root TMPDIR=%s (outside approved temp roots)"
                % rejected_tmp
            )
        rejected_codex = _codex_home_rejected()
        if rejected_codex is not None:
            return _refusal(
                "rejected write allow-root ~/.codex=%s (outside home)"
                % rejected_codex
            )

        home = protected_home()
        ro = read_only if workdir is not None else True
        # Preserve validation of caller-supplied policy paths without using
        # those additions in the canary profile itself.
        profile_params(
            workdir, ro, read_wall=read_wall,
            extra_read_paths=extra_read_paths,
            extra_write_paths=extra_write_paths,
            extra_deny_write=extra_deny_write,
            extra_post_write=extra_post_write,
            extra_deny_read=extra_deny_read,
            extra_post_read=extra_post_read,
            extra_deny_link=extra_deny_link,
            extra_deny_write_final=extra_deny_write_final,
            extra_deny_exec=extra_deny_exec,
            extra_deny_read_final=extra_deny_read_final,
        )
        # Probe a path denied by the actual profile: Q0 for read-only
        # workdirs, otherwise the protected home (outside the write allow-set).
        # The unwrapped control must succeed first, so an outer sandbox that
        # also blocks this write makes verification fail closed. Do not add a
        # probe-specific deny: that would prove only the extra rule works.
        probe_dir = (
            os.path.realpath(str(workdir))
            if workdir is not None and read_only else home
        )
        fd, target = tempfile.mkstemp(
            prefix="omni_kernel_fence_probe_", dir=probe_dir,
        )
        os.close(fd)
        target = os.path.realpath(target)

        probe_cmd = [sys.executable, "-I", "-S", "-c", _PROBE_SCRIPT]
        env = sanitized_child_env()
        env["OMNI_KSB_TARGET"] = target
        # --- 1. Positive control, UNWRAPPED ---
        control = _run_probe(probe_cmd, env)
        if isinstance(control, dict):
            return control
        if control.returncode != 0:
            return _refusal("kernel fence control probe exited %s" % control.returncode)
        control_token = _parse_probe_stdout(control.stdout)
        if control_token != "WROTE":
            return _refusal(
                "kernel fence control write failed (probe location unusable); "
                "token=%r exit=%s"
                % (control_token, control.returncode)
            )

        # --- 2. Fenced run, WRAPPED ---
        # Verify the common profile defaults, independent of caller-supplied
        # extra denies or allows which could mask a missing default rule.
        fenced_argv = wrap_argv(
            probe_cmd, workdir, ro, read_wall=read_wall,
        )
        fenced = _run_probe(fenced_argv, env)
        if isinstance(fenced, dict):
            return fenced
        if fenced.returncode != 0:
            return _refusal("kernel fence write probe exited %s" % fenced.returncode)
        fenced_token = _parse_probe_stdout(fenced.stdout)

        if fenced_token is None:
            return _refusal(
                "kernel fence probe produced neither WROTE nor ERRNO "
                "(probe did not run); exit=%s"
                % fenced.returncode
            )
        if fenced_token == "WROTE":
            return _refusal(
                "kernel fence probe write landed at %s (fence did not deny)"
                % target
            )
        if fenced_token != "ERRNO=1":
            return _refusal(
                "kernel fence probe denied write with unexpected errno token %r "
                "(need ERRNO=1/EPERM)"
                % fenced_token
            )

        # The isolated run above proves the shared default deny. Also exercise
        # the composed provider profile, where a provider-specific allow could
        # otherwise reopen this protected target.
        composed_argv = wrap_argv(
            probe_cmd, workdir, ro, read_wall=read_wall,
            extra_read_paths=extra_read_paths,
            extra_write_paths=extra_write_paths,
            extra_deny_write=extra_deny_write,
            extra_post_write=extra_post_write,
            extra_deny_read=extra_deny_read,
            extra_post_read=extra_post_read,
            extra_deny_link=extra_deny_link,
            extra_deny_write_final=extra_deny_write_final,
            extra_deny_exec=extra_deny_exec,
            extra_deny_read_final=extra_deny_read_final,
        )
        composed = _run_probe(composed_argv, env)
        if isinstance(composed, dict):
            return composed
        if composed.returncode != 0:
            return _refusal(
                "kernel fence composed write probe exited %s" % composed.returncode
            )
        composed_token = _parse_probe_stdout(composed.stdout)
        if composed_token != "ERRNO=1":
            return _refusal(
                "kernel fence composed-profile write probe expected ERRNO=1, "
                "got %r" % composed_token
            )

        if not read_wall:
            return None

        # --- 4/5/6. Read wall proof: list the protected home directory. ---
        # This is a guaranteed existing target outside narrower config and
        # workdir re-allows. It probes read access without creating a file
        # under home, where an outer sandbox may deny host canary writes.
        read_target = home

        read_cmd = [sys.executable, "-I", "-S", "-c", _READ_PROBE_SCRIPT]
        read_env = sanitized_child_env()
        read_env["OMNI_KSB_TARGET"] = read_target

        read_control = _run_probe(read_cmd, read_env)
        if isinstance(read_control, dict):
            return read_control
        if read_control.returncode != 0:
            return _refusal(
                "kernel fence read control probe exited %s" % read_control.returncode
            )
        read_control_token = _parse_read_probe_stdout(read_control.stdout)
        if read_control_token != "READ_OK":
            return _refusal(
                "kernel fence read-probe control failed (canary unreadable); "
                "token=%r exit=%s"
                % (read_control_token, read_control.returncode)
            )

        read_fenced_argv = wrap_argv(
            read_cmd, workdir, ro, read_wall=True,
        )
        read_fenced = _run_probe(read_fenced_argv, read_env)
        if isinstance(read_fenced, dict):
            return read_fenced
        if read_fenced.returncode != 0:
            return _refusal(
                "kernel fence read probe exited %s" % read_fenced.returncode
            )
        read_fenced_token = _parse_read_probe_stdout(read_fenced.stdout)

        if read_fenced_token is None:
            return _refusal(
                "kernel fence read-probe produced neither READ_OK nor ERRNO "
                "(probe did not run); exit=%s"
                % read_fenced.returncode
            )
        if read_fenced_token == "READ_OK":
            return _refusal(
                "kernel fence read-probe read succeeded at %s "
                "(read wall did not deny)"
                % read_target
            )
        if read_fenced_token == "ERRNO=1":
            read_composed_argv = wrap_argv(
                read_cmd, workdir, ro, read_wall=True,
                extra_read_paths=extra_read_paths,
                extra_write_paths=extra_write_paths,
                extra_deny_write=extra_deny_write,
                extra_post_write=extra_post_write,
                extra_deny_read=extra_deny_read,
                extra_post_read=extra_post_read,
                extra_deny_link=extra_deny_link,
                extra_deny_write_final=extra_deny_write_final,
                extra_deny_exec=extra_deny_exec,
                extra_deny_read_final=extra_deny_read_final,
            )
            read_composed = _run_probe(read_composed_argv, read_env)
            if isinstance(read_composed, dict):
                return read_composed
            if read_composed.returncode != 0:
                return _refusal(
                    "kernel fence composed read probe exited %s"
                    % read_composed.returncode
                )
            read_composed_token = _parse_read_probe_stdout(read_composed.stdout)
            if read_composed_token == "ERRNO=1":
                return None
            if read_composed_token == "READ_OK":
                return _refusal(
                    "kernel fence composed-profile read-probe succeeded at %s "
                    "(read wall did not deny)" % read_target
                )
            return _refusal(
                "kernel fence composed-profile read-probe expected ERRNO=1, "
                "got %r" % read_composed_token
            )
        return _refusal(
            "kernel fence read-probe denied read with unexpected errno token %r "
            "(need ERRNO=1/EPERM)"
            % read_fenced_token
        )
    except FencePathRejected as exc:
        return {
            "ok": False,
            "reason": "fence_path_rejected",
            "detail": exc.detail,
        }
    except Exception as exc:  # noqa: BLE001 — never raise from verify_fence
        return _refusal("kernel fence probe error: %s" % redact_secrets(str(exc))[:200])
    finally:
        if target:
            try:
                os.remove(target)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Agent-lane fence (claude / pi). Consult callers never hit this block.
# ---------------------------------------------------------------------------

FENCE_NONE_DETAIL = {
    "read_wall": False,
    "workdir_writable": False,
    "vendor_state_writable": False,
}

# Home-relative leaves that may be re-allowed. A candidate whose realpath is
# not exactly protected_home()/<one of these> is refused (a vendor binary
# inside home is a separate literal-file entry, never its directory)
# LOUD, never widened to a parent.
_AGENT_HOME_LEAVES = (
    ".claude",
    ".claude.json",
    os.path.join("Library", "Keychains"),
    os.path.join("Library", "Keychains", "login.keychain-db"),
    ".pi",
    os.path.join(".claude", "todos"),
    os.path.join(".claude", "shell-snapshots"),
    os.path.join(".claude", "debug"),
    os.path.join(".pi", "agent", "sessions"),
    os.path.join(".pi", "agent", "settings.json.lock"),
    os.path.join(".pi", "agent", "auth.json.lock"),
    os.path.join(".pi", "agent", "auth.json"),
    ".codex",
    ".cursor",
    os.path.join(".cursor", "ai-tracking"),
    os.path.join(".cursor", "skills-cursor"),
    os.path.join(".cursor", "projects"),
    os.path.join(".cursor", "plugins"),
    os.path.join(".cursor", "plugins", "cache"),
)

# codex vendor state that a fenced agent run legitimately writes. Everything
# else under the codex home is denied (see _collect_agent_fence): the tree
# mixes machine state with the files that steer the NEXT codex run, and
# enumerating the dangerous half does not work -- review found four members
# missing from a hand-built deny list in one pass (AGENTS.override.md, which
# takes PRECEDENCE over AGENTS.md; $CODEX_HOME/<name>.config.toml profile
# layers; a script hooks.json invokes; the staged plugin tree under .tmp).
# The allow-list closes the FILENAME face: a steering name that is not
# state is denied. models_cache.json is NOT state here: it feeds cached
# model metadata to later sessions on the same account, so it is read-only
# (final write deny via _AGENT_DENY_WRITE); codex refetches when it cannot
# use or write it. Four CONTENT faces stay post-allowed and are tracked
# residual risk: sessions/ (rollout JSONL that exec resume replays as prior
# turns), cache/ (cached tool descriptions, read back as tool docs next
# run), tmp/ (arg0 exec helpers), and the sqlite state
# (thread_dynamic_tools, read back by exec resume).
#
# Directories cover their own new children, which is what keeps this stable:
# sessions/ grows a fresh <year>/<month>/<day> tree every day. Top-level
# state FILES: existing .sqlite/.db names are the anchor; -wal/-shm
# siblings are derived from those names whether or not they exist at
# spawn, because sqlite deletes them on a clean close. Schema ordinals
# (logs_2, state_5, thread_history_1) still move on a codex release.
_CODEX_STATE_DIRS = (
    "sessions",
    "shell_snapshots",
    "thread-writer-locks",
    "cache",
    "tmp",
    "log",
)
_CODEX_STATE_DB_SUFFIXES = (".sqlite", ".db")
_CODEX_STATE_FILES = (
    "auth.json",            # rewritten on token refresh (accepted exposure)
    "history.jsonl",
    "session_index.jsonl",
    "version.json",
    "installation_id",
    ".personality_migration",
    ".sandbox_migration",
)

# Grok runtime leaves; config, hooks, skills and installed binaries remain
# deny-write. The launcher never creates or touches these vendor-owned paths.
_GROK_STATE_DIRS = ("sessions", "logs", "tmp", "crash", "trace-exports")
_GROK_STATE_FILES = (
    "auth.json", "auth.json.lock", "agent_id", "active_sessions.json",
    "active_sessions.lock", "version.json", "models_cache.json",
    "settings_cache.json", "sandbox-events.jsonl",
)
# grok saves those files by writing "<GROK_HOME>/.<pid>.<seq>.tmp" and
# renaming it over the target (grok 1.0.46, live 2026-10-10). Without the
# sibling a refresh the IdP already rotated cannot be saved: the next run
# presents the old refresh token, the IdP rejects it and grok clears the
# login. The rename still needs write on its target, so only the literal
# names above can be replaced this way; config, hooks, skills stay denied.
# The allow is bound to the fenced grok's own pid: an unfenced grok sharing
# the home writes config.toml through the same kind of sibling, and a rename
# needs unlink on its source, so an unbound pattern would let the child
# replace that sibling before it is published (review 2026-10-10).
GROK_TMP_KIND = "grok-tmp"
GROK_PID_PARAM = "G0"
# Placeholder until the launcher supplies the real pid; no process is pid 0.
GROK_PID_PLACEHOLDER = "0"
# Run in front of a wrapped argv that starts "-p PROFILE -D G0=0": drop the
# placeholder and pass this shell's pid, which exec keeps through
# sandbox-exec to grok. Anything else exits 126 without running the command.
GROK_PID_SHIM = (
    '[ "$1" = -p ] && [ "$3" = -D ] && [ "$4" = %s=%s ] || exit 126; '
    'p=$2; shift 4; exec "$0" -p "$p" -D "%s=$$" "$@"'
    % (GROK_PID_PARAM, GROK_PID_PLACEHOLDER, GROK_PID_PARAM)
)

# Global instruction files grok appends to every session's prompt (its own
# AGENTS.md family plus the Claude-compat names it probes in GROK_HOME,
# ~/.claude and ~/.cursor) and each root's always-scanned rules directory.
# GROK_CLAUDE_*_ENABLED=0 does not stop ~/.claude/Claude.md (grok 1.0.41,
# `grok inspect`), so the read deny is the mechanism. Headless lanes run an
# order; the user's interactive instructions must not ride along.
# Seatbelt (live probe): a literal read deny matches every case spelling of
# the name and also denies reads through a symlink at that name, whatever
# the chain points to later, and a subpath deny covers symlinks inside it.
# Final symlink targets are denied too, so the content has no second name;
# a hard-link alias cannot be denied by name, so it is refused.
_GROK_INSTRUCTION_FILES = (
    "Agents.md", "AGENTS.md", "Agent.md", "AGENT.md",
    "Claude.md", "CLAUDE.md", "CLAUDE.local.md",
)
_GROK_INSTRUCTION_DIRS = ("rules",)
_GROK_RULES_MAX_ENTRIES = 512

# Antigravity CLI (agy 1.3.3) fence inputs, live-bisected 2026-10-13 under the
# real read/write wall (a SIGKILL-on-access deny rule per path, plus removing
# one allow at a time):
#   login check, an ordinary reply and `agy models` need, below the home read
#   wall, only: the login keychain (Library/Keychains, as for claude/cursor),
#   ~/.gemini/antigravity-cli/installation_id, .../cache/default_project_id.txt,
#   ~/.gemini/config/projects (the default project record) and metadata of the
#   ancestors (the M* clause). Writes: none are needed; .../log and .../crashes
#   are allowed so a run does not print "operation not permitted" diagnostics.
#   Conversations, brain/ transcripts, annotations, history.jsonl and the
#   summaries database stay unreadable and unwritable (an earlier run's
#   prompts live there).
# Instruction files agy 1.3.3 really loads (canary, unfenced agy run with HOME
# pointed at a scratch directory holding one synthetic file each; a reply of
# the marker word means the file was loaded): <home>/.gemini/GEMINI.md,
# <home>/.gemini/AGENTS.md, <home>/.gemini/config/GEMINI.md and
# <home>/.gemini/config/AGENTS.md. Not loaded: the same names directly under
# <home>, <home>/.agents, <home>/.gemini/antigravity-cli, a rules/ directory in
# any of those, ~/.gemini/CLAUDE.md, and a GEMINI.md in an ancestor of the
# working directory. (A kill-on-access probe cannot see a file that does not
# exist: the kernel answers ENOENT before the sandbox rule is consulted.)
# agy also opens ~/.gemini/config/hooks.json and mcp_config.json (user hooks and
# MCP servers it would execute), antigravity-cli/settings.json, bin/ and
# builtin/ (sidecar programs it launches); a run with all of them denied for
# read still succeeds without a diagnostic, so they are denied outright.
_AGY_STATE_ROOT = os.path.join(".gemini", "antigravity-cli")
_AGY_READ_LEAVES = (
    ("literal", "installation_id"),
    ("literal", os.path.join("cache", "default_project_id.txt")),
    ("subpath", "log"),
    ("subpath", "crashes"),
)
_AGY_WRITE_LEAVES = (("subpath", "log"), ("subpath", "crashes"))
_AGY_PROJECTS_LEAF = os.path.join(".gemini", "config", "projects")
_AGY_INSTRUCTION_FILES = ("GEMINI.md", "Gemini.md", "AGENTS.md", "Agents.md",
                          "AGENT.md")
_AGY_INSTRUCTION_DIRS = ("rules",)
# Home-relative roots agy loads those names from: its dot dir and its
# customization root. Rules directories under them are covered as well. The
# projects directory is not a place agy loads them from (probed: not loaded),
# but it sits inside a read re-allow, so the names are denied there too: a
# future agy must not turn a re-allowed directory into an instruction source.
_AGY_INSTRUCTION_ROOTS = (
    ".gemini", os.path.join(".gemini", "config"), _AGY_PROJECTS_LEAF)
# Names that carry executable or steering configuration (hooks, MCP servers,
# settings, sidecar programs): read-denied and write-denied, never state.
_AGY_CONFIG_NAMES = (
    ("literal", os.path.join(".gemini", "settings.json")),
    ("literal", os.path.join(".gemini", "config", "hooks.json")),
    ("literal", os.path.join(".gemini", "config", "mcp_config.json")),
    ("subpath", os.path.join(".gemini", "config", "hooks")),
    ("subpath", os.path.join(".gemini", "config", "plugins")),
    ("subpath", os.path.join(".gemini", "config", "skills")),
    ("subpath", os.path.join(".gemini", "config", "sidecars")),
    ("literal", os.path.join(_AGY_STATE_ROOT, "settings.json")),
    ("subpath", os.path.join(_AGY_STATE_ROOT, "bin")),
    ("subpath", os.path.join(_AGY_STATE_ROOT, "builtin")),
)


def agy_instruction_deny_rows(home=None):
    """``(rows, rejection)``: every read-deny row of the agy fence.

    The GEMINI.md / AGENTS.md / rules family under each agy root (with the
    same on-disk scan the grok rows get: symlink targets, a hard-linked file
    refuses) plus the configuration names agy loads and executes. The rows
    are emitted as deny-read AND final deny-write. ``agy_layout_refusal``
    guarantees first that none of these names is a link, so every row is the
    one plain path. The host fence recomputes this to verify the wall held.
    """
    home = home or protected_home()
    rows: list[tuple[str, str]] = []
    for rel in _AGY_INSTRUCTION_ROOTS:
        found, why = _grok_instruction_deny_read(
            os.path.join(home, rel), files=_AGY_INSTRUCTION_FILES,
            dirs=_AGY_INSTRUCTION_DIRS)
        if why:
            return rows, "agy instruction scan: " + why
        rows.extend(row for row in found if row not in rows)
    for kind, rel in _AGY_CONFIG_NAMES:
        for row in _deny_entries(kind, os.path.join(home, rel)):
            if row not in rows:
                rows.append(row)
    return rows, None


_AGY_WALK_CAP = 5000
_AGY_LAYOUT_RULE = (
    "agy fence rule: nothing under ~/.gemini may be a link, a file with a "
    "second name or a data-volume spelling; replace it with the plain "
    "directory or file (a stow-managed ~/.gemini is not supported)")


def _on_data_volume(path):
    return path[:len(_DATA_VOLUME) + 1].casefold() == (_DATA_VOLUME + os.sep).casefold() \
        or path.casefold() == _DATA_VOLUME.casefold()


def _agy_spellings(path):
    """*path* as the kernel compares it, plus the data-volume spelling.

    A rule spelled with the ``/System/Volumes/Data`` prefix never matches
    (measured); the prefixed row is emitted anyway so a reader sees both.
    """
    out = [path]
    firm = _DATA_VOLUME + path
    try:
        if os.path.samestat(os.stat(firm), os.stat(path)):
            out.append(firm)
    except OSError:
        pass
    return out


def _agy_node_refusal(home, path):
    """Refusal detail, or None, for one path under the agy dot tree.

    See ``_agy_inspect``; this is its refusal half."""
    return _agy_inspect(home, path)[0]


def _agy_inspect(home, path):
    """``(refusal, lstat result)`` for one path under the agy dot tree.

    The lstat result (None when the path is absent or refused) is what callers
    use to learn the file type: they must not ask ``os.path.isdir`` and the
    like, which swallow an inspection error into "no" and would let a tree
    that could not be looked at pass. Refuses a symlink, a regular file with more than one link, a path whose
    resolved spelling differs from the one the rules are written with (a link
    above it, an alternate case or normalization), anything spelled through
    the data volume, and any inspection failure other than the path being
    absent.
    """
    shown = "~" + path[len(home):] if path.startswith(home + os.sep) or path == home else path
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return ("fence path rejected (agy path cannot be inspected: %s): %s; %s"
                % (exc.strerror or exc, shown, _AGY_LAYOUT_RULE)), None
    if stat.S_ISLNK(st.st_mode):
        return ("fence path rejected (%s is a symlink); %s"
                % (shown, _AGY_LAYOUT_RULE)), None
    if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
        return ("fence path rejected (%s has %d hard links); %s"
                % (shown, st.st_nlink, _AGY_LAYOUT_RULE)), None
    real = os.path.realpath(path)
    if _on_data_volume(path) or _on_data_volume(real):
        return ("fence path rejected (%s is spelled through /System/Volumes/Data); %s"
                % (shown, _AGY_LAYOUT_RULE)), None
    if real != path:
        return ("fence path rejected (%s resolves elsewhere through a link or an "
                "alternate spelling); %s" % (shown, _AGY_LAYOUT_RULE)), None
    return None, st


def _agy_tree_refusal(home, root):
    """``_agy_node_refusal`` for *root* and everything below it.

    Enumerates with ``os.scandir`` and counts while listing, so a directory
    wider than ``_AGY_WALK_CAP`` entries in total refuses before it has been
    read in full (not after sorting a huge listing). A directory that cannot
    be listed refuses. Links are never followed.
    """
    why, st = _agy_inspect(home, root)
    # The type comes from the lstat just taken: absent -> nothing to walk,
    # not a directory -> the node check was all there is.
    if why or st is None or not stat.S_ISDIR(st.st_mode):
        return why
    shown = "~" + root[len(home):]
    pending = [root]
    seen = 0
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    seen += 1
                    if seen > _AGY_WALK_CAP:
                        return ("fence path rejected (agy tree has more than %d "
                                "entries to verify): %s; %s"
                                % (_AGY_WALK_CAP, shown, _AGY_LAYOUT_RULE))
                    why = _agy_node_refusal(home, entry.path)
                    if why:
                        return why
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError as exc:
                        return ("fence path rejected (agy path cannot be "
                                "inspected: %s): %s; %s"
                                % (exc.strerror or exc, shown, _AGY_LAYOUT_RULE))
                    if is_dir:
                        pending.append(entry.path)
        except OSError as exc:
            return ("fence path rejected (agy tree cannot be read: %s): %s; %s"
                    % (exc.strerror or exc, shown, _AGY_LAYOUT_RULE))
    return None


def agy_layout_refusal(home=None):
    """Refusal detail (LOUD, names the path and the rule), or None.

    The agy fence supports only a plain ``~/.gemini`` tree: under it, none of
    these may be a symlink, a regular file with ``st_nlink > 1`` or spelled
    through ``/System/Volumes/Data``, and none may fail inspection:
    ``~/.gemini``, ``config``, ``config/projects`` and everything in it,
    ``antigravity-cli`` with the leaves the policy re-allows, every
    instruction-family name (case variants found on disk, ``rules/`` and its
    contents) and every steering configuration name. Class rule instead of
    chasing each layout (two-hop links, dangling firmlink targets, a target's
    parent renamed and re-linked, a whole dot dir stowed): any of them would
    make a second name for a file the rows protect by one name.
    The policy calls this when it is built and the host fence calls it again
    once the wrap is composed.
    """
    home = home or protected_home()
    gemini = os.path.join(home, ".gemini")
    state = os.path.join(home, _AGY_STATE_ROOT)
    singles = [gemini, os.path.join(gemini, "config"), state,
               os.path.join(state, "cache")]
    singles += [os.path.join(state, leaf) for _kind, leaf in _AGY_READ_LEAVES]
    # The one state file that is checked although it is not re-allowed: the
    # prior-conversation history, whose alias would be readable/writable.
    singles.append(os.path.join(state, "history.jsonl"))
    singles += [os.path.join(home, rel) for kind, rel in _AGY_CONFIG_NAMES
                if kind == "literal"]
    for path in singles:
        why = _agy_node_refusal(home, path)
        if why:
            return why
    # The steering directories (bin/, builtin/, hooks/, plugins/, skills/,
    # sidecars/) hold the programs and definitions agy loads or executes: their
    # CONTENTS are walked, a file with a second name there is a writable alias
    # of something agy runs. The state directories (conversations/, brain/,
    # annotations/, log/ ...) are deliberately NOT walked: the vendor itself
    # creates links in them (``cli.log`` is a relative symlink into ``log/``),
    # and a pre-planted alias of those files is a documented residual.
    for kind, rel in _AGY_CONFIG_NAMES:
        if kind == "subpath":
            why = _agy_tree_refusal(home, os.path.join(home, rel))
            if why:
                return why
    why = _agy_tree_refusal(home, os.path.join(home, _AGY_PROJECTS_LEAF))
    if why:
        return why
    wanted = {name.casefold() for name in
              _AGY_INSTRUCTION_FILES + _AGY_INSTRUCTION_DIRS}
    for rel in _AGY_INSTRUCTION_ROOTS:
        root = os.path.join(home, rel)
        names = []
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if len(names) >= _AGY_WALK_CAP:
                        return ("fence path rejected (agy directory has more than "
                                "%d entries to verify): %s; %s"
                                % (_AGY_WALK_CAP, "~" + root[len(home):],
                                   _AGY_LAYOUT_RULE))
                    names.append(entry.name)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            return ("fence path rejected (agy directory cannot be listed: %s): "
                    "%s; %s" % (exc.strerror or exc, "~" + root[len(home):],
                                _AGY_LAYOUT_RULE))
        for name in sorted(names):
            if name.casefold() in wanted:
                why = _agy_tree_refusal(home, os.path.join(root, name))
                if why:
                    return why
    return None


_GROK_SCAN_MAX_HOPS = 40


class _GrokScanRefusal(Exception):
    """A LOUD scan refusal; ``detail`` becomes ``fence_path_rejected``."""

    def __init__(self, detail):
        self.detail = redact_secrets(str(detail))[:300]
        super().__init__(self.detail)


def _grok_scan_lstat(path):
    """``os.lstat`` or refuse. Every error, ENOENT included, refuses here;
    the two ENOENT exemptions are decided by the callers, not by this."""
    try:
        return os.lstat(path)
    except OSError as exc:
        raise _GrokScanRefusal("grok instruction path cannot be inspected (%s): %s"
                               % (exc.strerror or exc, path))


def _grok_scan_enoent_confirmed(path):
    """True only when a fresh ``os.lstat`` again says ENOENT."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise _GrokScanRefusal("grok instruction path cannot be inspected (%s): %s"
                               % (exc.strerror or exc, path))
    return False


def _grok_scan_resolve(path):
    """``(canonical, lstat)`` of *path* with every symlink followed by hand.

    No ``realpath`` / ``stat`` / ``os.path`` predicate is used: each of them
    can treat a failed syscall as "absent" or "not a link". Every component is
    ``os.lstat``'ed and every link ``os.readlink``'ed with errors refused.
    The single exemption: once at least one link was followed, ENOENT on the
    LAST component of the target (confirmed by a fresh lstat) is a dangling
    link; it returns ``(target_path, None)`` so the caller still denies it.
    """
    if not os.path.isabs(path):
        raise _GrokScanRefusal("grok instruction path is not absolute: %s" % path)
    pending = [part for part in path.split(os.sep) if part]
    resolved = os.sep
    hops = 0
    st = None
    while pending:
        part = pending.pop(0)
        if part == ".":
            continue
        if part == "..":
            resolved = os.path.dirname(resolved) or os.sep
            st = None
            continue
        candidate = os.path.join(resolved, part)
        try:
            st = os.lstat(candidate)
        except FileNotFoundError:
            if hops and not pending and _grok_scan_enoent_confirmed(candidate):
                return candidate, None
            raise _GrokScanRefusal("grok instruction path vanished or does not "
                                   "resolve during scan: %s" % path)
        except OSError as exc:
            raise _GrokScanRefusal("grok instruction path cannot be inspected (%s): %s"
                                   % (exc.strerror or exc, candidate))
        if stat.S_ISLNK(st.st_mode):
            hops += 1
            if hops > _GROK_SCAN_MAX_HOPS:
                raise _GrokScanRefusal("grok instruction path has too many symlinks: %s"
                                       % path)
            try:
                link = os.readlink(candidate)
            except OSError as exc:
                raise _GrokScanRefusal("grok instruction link cannot be read (%s): %s"
                                       % (exc.strerror or exc, candidate))
            pending = [p for p in link.split(os.sep) if p] + pending
            if link.startswith(os.sep):
                resolved = os.sep
            st = None
            continue
        resolved = candidate
    if st is None:   # path was "/" or ended on ".."
        st = _grok_scan_lstat(resolved)
    return resolved, st


def _grok_scan_listdir(path):
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        raise _GrokScanRefusal("grok instruction directory cannot be listed (%s): %s"
                               % (exc.strerror or exc, path))


def _grok_instruction_deny_read(root, files=None, dirs=None):
    """``(rows, rejection)`` for the instruction names under *root*.

    *files* / *dirs* default to the grok name sets; the agy policy passes its
    own (``_AGY_INSTRUCTION_FILES`` / ``_AGY_INSTRUCTION_DIRS``).

    ``rows`` are ``(kind, path)`` deny-read rows: every probed name, each
    ``rules`` directory (under the lexical root and, if it differs, the
    resolved one), and the canonical final target of every entry found at
    those names or anywhere below ``rules`` (links followed, each directory
    once). ``rejection`` is a LOUD detail for a hard-linked file (an alias no
    name rule covers), an over-large ``rules`` tree, or ANY syscall error,
    with exactly two exemptions, each confirmed by a fresh ``os.lstat``:
    ENOENT on the root itself (nothing to deny beyond the fixed names), and
    ENOENT on the last component of a dangling link's target (its path is
    still denied). Syscalls used: ``os.lstat``, ``os.readlink``,
    ``os.listdir`` only.
    """
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    name_files = _GROK_INSTRUCTION_FILES if files is None else tuple(files)
    name_dirs = _GROK_INSTRUCTION_DIRS if dirs is None else tuple(dirs)

    def add(kind, path):
        row = (kind, path)
        if row not in seen:
            seen.add(row)
            rows.append(row)

    def add_names(base):
        for name in name_files:
            add("literal", os.path.join(base, name))
        for name in name_dirs:
            add("subpath", os.path.join(base, name))

    def deny_target(path):
        """Deny the canonical final target of *path*; return its lstat."""
        canonical, st = _grok_scan_resolve(path)
        if st is None:
            add("literal", canonical)            # dangling: deny the name anyway
            return None
        if stat.S_ISDIR(st.st_mode):
            add("subpath", canonical)
            return st
        add("literal", canonical)
        if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
            raise _GrokScanRefusal("grok instruction file has %d hard links (an alias "
                                   "outside the read deny): %s" % (st.st_nlink, canonical))
        return st

    add_names(root)
    try:
        try:
            on_disk = sorted(os.listdir(root))
        except FileNotFoundError:
            if _grok_scan_enoent_confirmed(root):
                return rows, None                # exemption: absent root
            raise _GrokScanRefusal("grok instruction root changed during scan: %s" % root)
        except OSError as exc:
            raise _GrokScanRefusal("grok instruction root cannot be listed (%s): %s"
                                   % (exc.strerror or exc, root))
        canonical_root, _root_st = _grok_scan_resolve(root)
        if canonical_root != root:
            add_names(canonical_root)
        wanted_files = {name.casefold() for name in name_files}
        wanted_dirs = {name.casefold() for name in name_dirs}
        count = 0
        for name in on_disk:
            folded = name.casefold()
            if folded not in wanted_files and folded not in wanted_dirs:
                continue
            st = deny_target(os.path.join(root, name))
            if folded not in wanted_dirs or st is None or not stat.S_ISDIR(st.st_mode):
                continue
            first, _st = _grok_scan_resolve(os.path.join(root, name))
            pending = [first]
            visited = {(st.st_dev, st.st_ino)}
            while pending:
                current = pending.pop()
                for entry in _grok_scan_listdir(current):
                    count += 1
                    if count > _GROK_RULES_MAX_ENTRIES:
                        raise _GrokScanRefusal("grok rules directory has more than %d "
                                               "entries: %s" % (_GROK_RULES_MAX_ENTRIES,
                                                                root))
                    child = os.path.join(current, entry)
                    child_st = deny_target(child)
                    if child_st is not None and stat.S_ISDIR(child_st.st_mode):
                        key = (child_st.st_dev, child_st.st_ino)
                        if key not in visited:
                            visited.add(key)
                            pending.append(_grok_scan_resolve(child)[0])
    except _GrokScanRefusal as refusal:
        return rows, refusal.detail
    return rows, None


# A3 explicit deny-write list (last-match-wins, after the write allow-set).
# Kinds are independent of whether the path exists today.
_AGENT_DENY_WRITE = (
    (".claude/settings.json", "literal"),
    (".claude/settings.local.json", "literal"),
    (".claude/hooks", "subpath"),
    (".claude/plugins", "subpath"),
    (".claude/commands", "subpath"),
    (".claude/agents", "subpath"),
    (".claude/skills", "subpath"),
    (".claude/CLAUDE.md", "literal"),
    (".claude.json", "literal"),
    (".pi/agent/settings.json", "literal"),
    (".pi/agent/AGENTS.md", "literal"),
    (".pi/agent/extensions", "subpath"),
    (".pi/agent/bin", "subpath"),
    (".cursor/cli-config.json", "literal"),
    (".cursor/mcp.json", "literal"),
    (".cursor/plugins", "subpath"),
    (".cursor/chats", "subpath"),
    (".cursor/agent-cli-state.json", "literal"),
    # codex: the home is denied whole; T re-allows only state (see
    # _collect_agent_fence). These rows name the instruction surface that
    # is not state -- config, the global instruction file, hooks (which
    # execute commands), and the agent/skill/plugin/rule definitions --
    # so they stay in W and must not appear in T.
    (".codex/config.toml", "literal"),
    (".codex/AGENTS.md", "literal"),
    # Precedence order per codex-cli 0.154.0 (core/src/agents_md.rs):
    # AGENTS.override.md, then AGENTS.md. Denying only the latter left the
    # file that WINS writable (review finding).
    (".codex/AGENTS.override.md", "literal"),
    (".codex/hooks.json", "literal"),
    (".codex/agents", "subpath"),
    (".codex/skills", "subpath"),
    (".codex/plugins", "subpath"),
    (".codex/rules", "subpath"),
    # Cached model metadata later sessions load; a child must not seed it.
    (".codex/models_cache.json", "literal"),
)

# Accepted exposure (A2): the login keychain FILE is inside the readable set
# so Claude can log in. Secrets stay behind the login password and securityd
# ACLs. --bare / a separate CLAUDE_CONFIG_DIR would avoid the file and lose
# the login; that path is not taken here.


def agent_fence_switch_on() -> bool:
    """True unless ``OMNI_LLM_AGENT_FENCE`` is an explicit off token."""
    raw = os.environ.get(ENV_AGENT_FENCE)
    if raw is None or str(raw).strip() == "":
        return True
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def unfenced_fallback_allowed() -> bool:
    """Off-Darwin unfenced spawn is opt-in via ``OMNI_LLM_AGENT_UNFENCED=1``."""
    return os.environ.get(ENV_AGENT_UNFENCED, "").strip() == "1"


def resolve_agent_binary(provider) -> str | None:
    """Absolute realpath of the vendor CLI, or None if unresolved.

    ``OMNI_LLM_CLAUDE_BIN`` / ``OMNI_LLM_PI_BIN`` / ``OMNI_LLM_CURSOR_BIN``
    override ``shutil.which``.
    """
    env_key = {"claude": ENV_CLAUDE_BIN, "pi": ENV_PI_BIN,
               "cursor": ENV_CURSOR_BIN}.get(provider)
    raw = None
    if env_key:
        override = os.environ.get(env_key)
        if override and str(override).strip():
            raw = str(override).strip()
    if not raw:
        # Cursor's binary is `cursor-agent`, NEVER `agent` (Grok Build CLI
        # collision) and never the bare provider key `cursor`.
        which_name = "cursor-agent" if provider == "cursor" else provider
        raw = vendor_bin.which(which_name)
    if not raw:
        return None
    return os.path.realpath(raw)


def admit_vendor_binary(vendor_bin) -> "AdmittedPath | None":
    """Read admission for a vendor CLI binary under the host fence.

    A binary OUTSIDE the protected home needs no entry (reads there stay
    allowed by default) -> None. One INSIDE home is re-allowed as a LITERAL on
    the resolved file only, never its directory: a directory entry would make
    every sibling under ~/.local/bin or an nvm prefix readable (review
    finding). Pass the realpath and exec that same path: Seatbelt refuses an
    execvp through a home-tree symlink even when the target is admitted.
    Raises FencePathRejected for a widening root or a non-regular file.
    """
    if not vendor_bin or not _path_inside(vendor_bin, protected_home()):
        return None
    if vendor_bin in _widening_roots() or not os.path.isfile(vendor_bin):
        raise FencePathRejected(
            "fence path rejected (vendor binary is not a regular file "
            "under home): %s" % vendor_bin
        )
    return admit_path(vendor_bin, "literal", label="vendor binary")


def claude_encoded_workdir(workdir) -> str:
    """Claude's on-disk project key: realpath with ``/`` replaced by ``-``."""
    return os.path.realpath(str(workdir)).replace("/", "-")


def claude_session_dir(workdir, config_dir=None) -> str:
    """``<config_dir or protected_home()/.claude>/projects/<encoded workdir>``."""
    root = (
        os.path.realpath(config_dir)
        if config_dir else os.path.join(protected_home(), ".claude")
    )
    return os.path.join(root, "projects", claude_encoded_workdir(workdir))


# cursor-agent's `~/.cursor/projects/<slug>` naming is NOT fully known: short
# workdirs slug as realpath-minus-leading-slash with non-alnum runs collapsed
# to "-" (live-verified incl. dots/underscores/spaces), but a longer workdir
# produced a truncated+hashed form ("...rmch0-7c91d62") no checked hash of
# the path explains — see REPORT-cursor-agent-lane.md's bisect table before
# attempting to narrow the `projects` write allow-set again.


def _widening_roots() -> set[str]:
    home = protected_home()
    roots = {home, "/", os.path.realpath("/")}
    parent = os.path.dirname(home)
    if parent:
        roots.add(parent)
    lib = os.path.join(home, "Library")
    roots.add(os.path.realpath(lib))
    return roots


def _admit_home_leaf(path, extra_leaves=()) -> str | None:
    """Return realpath if it equals a listed home leaf, else None (rejected)."""
    rp = os.path.realpath(path)
    if rp in _widening_roots():
        return None
    home = protected_home()
    allowed = [
        os.path.realpath(os.path.join(home, leaf))
        for leaf in _AGENT_HOME_LEAVES + tuple(extra_leaves)
    ]
    if rp in allowed:
        return rp
    return None




def claude_scratch_root() -> tuple[str | None, str | None]:
    """Validated ``/private/tmp/claude-<uid>``. Created 0700 if missing.

    Refused when the path is a symlink, not a directory, not owned by this
    uid, or group/world writable.
    """
    path = "/private/tmp/claude-%d" % os.getuid()
    if os.path.lexists(path) and os.path.islink(path):
        return None, "claude scratch root %s is a symlink" % path
    if not os.path.exists(path):
        try:
            os.mkdir(path, 0o700)
        except OSError as exc:
            return None, "could not create claude scratch root %s: %s" % (
                path, redact_secrets(str(exc))[:120],
            )
    if not os.path.isdir(path):
        return None, "claude scratch root %s is not a directory" % path
    try:
        st = os.stat(path)
    except OSError as exc:
        return None, "could not stat claude scratch root %s: %s" % (
            path, redact_secrets(str(exc))[:120],
        )
    if st.st_uid != os.getuid():
        return None, "claude scratch root %s is not owned by uid %d" % (
            path, os.getuid(),
        )
    if stat.S_IMODE(st.st_mode) & 0o077:
        return None, "claude scratch root %s is group/world writable" % path
    return os.path.realpath(path), None


# Account-home registry: one env prefix per provider that supports named
# accounts, and each provider's default home leaf under the protected home.
_ACCOUNT_HOME_PREFIXES = (
    ("claude", "CLAUDE_CONFIG_DIR_"),
    ("codex", "CODEX_HOME_"),
    ("grok", "GROK_HOME_"),
)
_DEFAULT_HOME_LEAF = {"claude": ".claude", "codex": ".codex", "grok": ".grok"}


def registered_account_homes(environ=None):
    """``[(label, provider, realpath)]`` for every registered account home.

    The label is the env var name (never a path), so refusals can name
    which registrations collide without printing where they live.
    """
    env = os.environ if environ is None else environ
    out = []
    for provider, prefix in _ACCOUNT_HOME_PREFIXES:
        for key in sorted(env):
            val = env[key]
            if not key.startswith(prefix) or not val or not str(val).strip():
                continue
            out.append((key, provider, os.path.realpath(str(val).strip())))
    return out


_CURSOR_CONFIG_ENV = "OMNI_LLM_CURSOR_CONFIG_DIR"


def registered_cursor_config_dirs(environ=None):
    """``[(label, "cursor", realpath)]`` for every operator-set cursor config dir.

    ``OMNI_LLM_CURSOR_CONFIG_DIR`` and ``OMNI_LLM_CURSOR_CONFIG_DIR_<NAME>``.
    Cursor's dir is NOT an account home (it is not read-denied to other
    roads); it is listed so the nested/equal-home conflict check can refuse
    a Claude/Codex/Grok home that is equal to or nested with it. Labels are
    env var names, never paths.
    """
    env = os.environ if environ is None else environ
    out = []
    for key in sorted(env):
        val = env[key]
        if not (key == _CURSOR_CONFIG_ENV
                or key.startswith(_CURSOR_CONFIG_ENV + "_")):
            continue
        if not val or not str(val).strip():
            continue
        raw = os.path.expanduser(str(val).strip())
        out.append((key, "cursor", os.path.realpath(raw)))
    return out


# Default config roots, for the "allowed root contains a registered sibling
# home" check only. Unlike ``_DEFAULT_HOME_LEAF`` this is not a deny list.
_CONTAIN_HOME_LEAF = {
    "claude": ".claude", "codex": ".codex", "grok": ".grok",
    "cursor": ".cursor", "pi": ".pi",
}


def _sibling_home_denies(provider, winner_root, home):
    """Deny-read rows for every registered account home of EVERY provider.

    Exempt: this run's own home (*winner_root*) and the RUNNING provider's
    default home (a ``<PREFIX>MAIN`` that points at it). Other providers'
    default homes are not exempt; a default that lives outside the protected
    home would otherwise stay readable through (allow default), so it is
    denied too. A registered home inside the winner is denied (the winner
    cannot read another account's tree). One that contains the winner cannot
    be denied without punching out the allowed root, and leaving it open
    would expose the other account, so the build fails closed with the
    labels only (``FencePathRejected``). Home and widening roots are skipped
    so a deny cannot wall off the allowed root.

    An allowed config root (this run's home, or the running provider's
    default root) that CONTAINS a registered sibling home also fails closed,
    and so does a cursor/pi config root EQUAL to one: a read deny on the
    inner home would be the only thing keeping it unreadable, and the
    caller emits these rows as FINAL denies precisely so that no later
    allow can reopen them; refusing the layout removes the dependency.
    Callers send the returned rows to both ``extra_deny_read`` and
    ``extra_deny_read_final``.
    """
    running_default = None
    leaf = _DEFAULT_HOME_LEAF.get(provider)
    if leaf:
        running_default = os.path.realpath(os.path.join(home, leaf))
    contain_root = winner_root
    if not contain_root and _CONTAIN_HOME_LEAF.get(provider):
        contain_root = os.path.realpath(
            os.path.join(home, _CONTAIN_HOME_LEAF[provider])
        )
    own_home_is_sibling = provider in ("claude", "codex", "grok")
    widening = _widening_roots()
    rows = []
    seen = set()

    def _add(path):
        if path not in seen:
            seen.add(path)
            rows.append(_entry("subpath", path))

    for label, prov, sib in registered_account_homes():
        if not sib or sib in widening or sib == home:
            continue
        if winner_root and _same_dir_exact(sib, winner_root):
            if own_home_is_sibling:
                continue
            raise FencePathRejected(
                "fence path rejected (registered account home %s is the "
                "same directory as this run's config root; equal homes "
                "are refused)" % label
            )
        if prov == provider and sib == running_default:
            continue
        if winner_root and _path_inside_any_case(winner_root, sib):
            raise FencePathRejected(
                "fence path rejected (registered account home %s contains "
                "this run's home; nested account homes are refused)" % label
            )
        if contain_root and not _same_dir_exact(sib, contain_root) \
                and _path_inside_any_case(sib, contain_root):
            raise FencePathRejected(
                "fence path rejected (registered account home %s sits "
                "inside this run's allowed config root; nested account "
                "homes are refused)" % label
            )
        _add(sib)
    for prov, dot in _DEFAULT_HOME_LEAF.items():
        if prov == provider:
            continue
        default = os.path.realpath(os.path.join(home, dot))
        if default in widening or _path_inside(default, home):
            continue
        if winner_root and _path_inside(winner_root, default):
            continue
        _add(default)
    return rows


def _claude_run_leaf(root, *parts, vendor="claude", resolved=False):
    """``(path, None)`` or ``(None, detail)`` for a per-run vendor state path.

    *path* is ``realpath(root)`` joined with *parts*, lexical. Any component
    below the root that is a symlink refuses (``detail``), never resolves:
    the previous child could have planted it. ``resolved=True`` takes *root*
    as the realpath the caller already recorded, so the recorded root and
    the spelled rows come from one resolution.
    """
    cur = root if resolved else os.path.realpath(root)
    for part in parts:
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            return None, (
                "fence path rejected (%s per-run state path under the "
                "config root is a symlink): %s" % (vendor, "/".join(parts))
            )
    return cur, None


_MAX_LINK_HOPS = 40


def _symlink_chain(path):
    """``(links, resolved)``: every symlink met resolving *path*, hop by hop.

    ``links`` holds each link's OWN location (its resolved parent plus its
    name, never realpath'd), in the order met. ``resolved`` is where the walk
    ends. Raises ``OSError`` on a loop or a hop budget overrun.
    """
    parts = [p for p in os.path.abspath(path).split(os.sep) if p]
    parts.reverse()
    cur = os.sep
    links = []
    while parts:
        part = parts.pop()
        if part == ".":
            continue
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        nxt = os.path.join(cur, part)
        if not os.path.islink(nxt):
            cur = nxt
            continue
        links.append(nxt)
        if len(links) > _MAX_LINK_HOPS:
            raise OSError(errno.ELOOP, "too many symlink hops", path)
        target = os.readlink(nxt)
        if os.path.isabs(target):
            cur = os.sep
        parts.extend(reversed([p for p in target.split(os.sep) if p]))
    return links, cur


def _location_inside_any_case(location, root):
    """*location* (a link's own spelling) is *root* or under it, refusal side.

    The last component stays a NAME: resolving it would resolve the very link
    being judged. Its parent is judged by identity, so a firmlink spelling or
    a symlinked ancestor of the link is still seen. *root* is a realpath'd
    write allow.
    """
    return location_inside(location, root, resolve_leaf=False)


def _vendor_root_link_refusal(vendor, root, expected, location_roots,
                              overlap_roots):
    """Refusal detail, or None, for a vendor root reached through symlinks.

    A vendor root that is a plain directory is not judged here. When the
    path to it crosses symlinks (a stow-managed ``~/.pi``), the late write
    allows are spelled at the RESOLVED root, so a link the child can
    rewrite between runs steers them anywhere. Same rule as the default
    claude/codex home's protected names (``_protected_config_refusal``): the
    chain is accepted only when no link sits in any of *location_roots* (the
    complete write entries of the spawn, the vendor's own state leaves
    included) and the resolved root neither contains nor sits inside any of
    *overlap_roots* (the same set WITHOUT those leaves, which live under the
    root by construction). *expected* is the realpath the state leaves were
    spelled against: the chain must still end there, and a root that has
    vanished since refuses. Anything that cannot be walked refuses.
    """
    try:
        links, resolved = _symlink_chain(root)
    except OSError:
        return ("fence path rejected (%s config root symlink chain cannot "
                "be resolved)" % vendor)
    if expected is not None and os.path.realpath(root) != expected:
        return ("fence path rejected (%s config root changed while the "
                "fence was built)" % vendor)
    if not os.path.isdir(root) or (expected is not None and resolved != expected):
        return ("fence path rejected (%s config root is gone or no longer "
                "the root the state paths were built for)" % vendor)
    if not links:
        return None
    for link in links:
        # The link's parent is already resolved, so the identity-aware
        # containment is safe on it (APFS firmlink spellings such as
        # /System/Volumes/Data/...). A row spelled AT the link resolves to the
        # link's target in that judgement, so the folded spelling test covers
        # a row that is the link itself or a spelled ancestor of it.
        if any(_location_inside_any_case(link, w)
               or _lexical_ci_same_or_inside(link, w)
               or _path_inside_any_case(os.path.dirname(link), w)
               for w in location_roots):
            return ("fence path rejected (%s config root is reached through "
                    "a symlink the run can write): %s" % (vendor, link))
    if any(_path_inside_any_case(resolved, w) or _path_inside_any_case(w, resolved)
           for w in overlap_roots):
        return ("fence path rejected (%s config root resolves into a write "
                "root of this run): %s -> %s" % (vendor, root, resolved))
    return None


_VENDOR_DOT = {"pi": ".pi", "cursor": ".cursor", "agy": _AGY_STATE_ROOT,
               "claude": ".claude"}


def pi_cursor_root_link_refusal(provider, workdir, read_only,
                                extra_write_paths, extra_post_write,
                                expected_root):
    """Chain check of the pi / cursor / agy / claude vendor root against a
    composed write set.

    For claude only the DEFAULT home (no ``CLAUDE_CONFIG_DIR``) has a vendor
    root here; a named profile home is admitted by
    ``_caller_config_dir_rejection``.

    *extra_write_paths* and *extra_post_write* are the spawn's write entries
    as they will be rendered (the host fence calls this again after adding
    repository roots). Returns a refusal detail or None; raises
    ``FencePathRejected`` when the write set itself cannot be resolved.
    """
    vendor_dir = os.path.join(protected_home(), _VENDOR_DOT[provider])
    write_roots = list(_write_allow_paths(
        workdir, read_only, extra_write_paths=extra_write_paths,
    ))
    location_roots = list(write_roots)
    for entry in extra_post_write or ():
        bound = _write_entry_bound(*entry)
        if bound is None:
            return ("fence path rejected (%s write entry cannot be bounded)"
                    % provider)
        location_roots.append(bound[1])
    with _location_cache():
        return _vendor_root_link_refusal(
            provider, vendor_dir, expected_root, location_roots, write_roots,
        )


_STATE_LINK_SCAN_CAP = 200000


def _hard_linked_state_refusal(vendor, path):
    """Refusal detail, or None, for a regular file at/below *path* with links.

    *path* is a writable state leaf (already refused if a symlink). A second
    name for the same inode is a writable alias that outlives the run, so a
    file with ``st_nlink > 1`` refuses. The walk never follows links and
    fails closed: an unreadable directory, a stat error other than a file
    vanishing mid-walk, or a tree past ``_STATE_LINK_SCAN_CAP`` entries
    refuses, because a hidden alias is what the scan exists to find.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return ("fence path rejected (%s state path cannot be checked for "
                "hard links: %s): %s" % (vendor, exc.strerror or exc, path))
    detail = ("fence path rejected (%s state file has %%d hard links, an "
              "alias outside the fence): %%s" % vendor)
    if stat.S_ISREG(st.st_mode):
        return detail % (st.st_nlink, path) if st.st_nlink > 1 else None
    if not stat.S_ISDIR(st.st_mode):
        return None
    failure = []

    def onerror(exc):
        if not isinstance(exc, FileNotFoundError):
            failure.append(exc)

    seen = 0
    for current, dirs, files in os.walk(path, followlinks=False, onerror=onerror):
        if failure:
            break
        dirs.sort()
        seen += len(dirs)
        if seen > _STATE_LINK_SCAN_CAP:
            return ("fence path rejected (%s state tree is too large to "
                    "verify for hard links, more than %d entries): %s"
                    % (vendor, _STATE_LINK_SCAN_CAP, path))
        for name in files:
            seen += 1
            if seen > _STATE_LINK_SCAN_CAP:
                return ("fence path rejected (%s state tree is too large to "
                        "verify for hard links, more than %d entries): %s"
                        % (vendor, _STATE_LINK_SCAN_CAP, path))
            full = os.path.join(current, name)
            try:
                st = os.lstat(full)
            except FileNotFoundError:
                continue
            except OSError as exc:
                failure.append(exc)
                break
            if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                return detail % (st.st_nlink, full)
        if failure:
            break
    if failure:
        exc = failure[0]
        return ("fence path rejected (%s state tree cannot be read to verify "
                "hard links: %s): %s"
                % (vendor, getattr(exc, "strerror", None) or exc,
                   getattr(exc, "filename", None) or path))
    return None


def _entry(kind, path):
    return (kind, os.path.realpath(path))


def _lexical_entry(kind, path):
    """``(kind, path)`` as written, never realpath'd.

    Used only for the codex-home post-allow block. ``_entry`` realpaths,
    which is right for every other caller and wrong here: a symlink at a
    post-allowed name would put the TARGET in T (last-match-wins).
    Seatbelt evaluates file-write on the kernel-resolved path, so a
    lexical allow of a symlink does not reach the target; refusing the
    symlink keeps lexical == resolved for the entries we emit.
    """
    return (kind, os.path.abspath(path))


_SBPL_REGEX_SPECIAL = frozenset("\\.^$*+?()[]{}|")


def _sbpl_regex_quote(text):
    return "".join("\\" + c if c in _SBPL_REGEX_SPECIAL else c for c in text)


def _write_entry_bound(kind, path):
    """``(kind, path)`` that bounds a write entry for overlap checks, or None.

    literal and subpath bound themselves. A ``GROK_TMP_KIND`` entry matches
    only direct children of its root, so ``("subpath", root)`` is a superset
    of it. Anything else cannot be bounded; callers fail closed.
    """
    if kind in ("literal", "subpath"):
        return (kind, path)
    if kind == GROK_TMP_KIND:
        return ("subpath", path)
    return None


def _codex_post_allow_refusal(path, root) -> str | None:
    """None if *path* may be a lexical T entry under *root*; else a detail.

    (i) lexically inside *root* (no ``..``, ``normpath`` equals itself);
    (ii) not currently a symlink (the root itself included);
    (iii) if it exists, realpath is not in ``_widening_roots()`` and is
    inside the realpath of *root*. Any violation is a LOUD spawn refusal
    -- never a silent drop.
    """
    raw = str(path)
    if ".." in raw.split(os.sep) or os.path.normpath(raw) != raw:
        return (
            "fence path rejected (codex post-allow is not a lexical path): %s"
            % path
        )
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    if abs_path != abs_root and not abs_path.startswith(abs_root + os.sep):
        return (
            "fence path rejected (codex post-allow escapes the codex home): %s"
            % path
        )
    if os.path.islink(path) or os.path.islink(abs_path):
        return (
            "fence path rejected (codex post-allow is a symlink): %s" % path
        )
    if os.path.lexists(path) or os.path.lexists(abs_path):
        rp = os.path.realpath(path)
        if rp in _widening_roots():
            return (
                "fence path rejected (codex post-allow resolves to a "
                "widening root): %s" % path
            )
        rp_root = os.path.realpath(root)
        if not _path_inside(rp, rp_root):
            return (
                "fence path rejected (codex post-allow resolves outside "
                "the codex home): %s" % path
            )
    return None


def _deny_entries(kind, path):
    """Deny clauses covering *path* BOTH as written and as it resolves.

    ``_entry`` realpaths, which is right for an allow clause but loses the
    deny when the named path is a SYMLINK out of the guarded tree: unlink
    and rename do not follow the last component, so the kernel checks the
    link's own path. With ``~/.codex`` re-allowed for write, a deny that
    only named ``~/.agents/harness/codex-AGENTS.md`` left
    ``~/.codex/AGENTS.md`` deletable and replaceable by a regular file —
    measured 2026-09-14, and exactly what "an agent may churn codex's state,
    never rewrite what the NEXT run is told to do" promises not to allow.
    """
    resolved = os.path.realpath(path)
    lexical = os.path.abspath(path)
    out = [(kind, resolved)]
    if lexical != resolved:
        out.append((kind, lexical))
    return out


def _protected_config_refusal(label, entries, *, named, writable_roots,
                              home_root):
    """Refusal detail, or None, for protected configuration names that alias.

    *entries* is ``[(name, lexical_path)]`` for the protected configuration
    names of ONE vendor home (the ``_AGENT_DENY_WRITE`` rows rooted there).
    Seatbelt is last-match-wins and judges the kernel-resolved path, so a
    protected name that is a symlink into a writable state leaf
    (``settings.json -> debug/payload``) lets a child holding that leaf
    rewrite what the next run executes: the deny on the resolved target
    sits before the state allowance that covers it. Refuse instead of
    reasoning about row order:

    * a NAMED (operator-registered) home refuses any protected name that is
      a symlink, or that resolves anywhere but its own lexical location;
    * the DEFAULT home refuses the same unless the symlink resolves OUTSIDE
      the vendor home and outside every writable root of this run (the
      operator's dotfile-managed ``CLAUDE.md`` / ``AGENTS.md`` pointing at a
      directory the child cannot write stays usable).

    Only the label and the protected name are reported, never a path.
    """
    for name, lexical in entries:
        if not os.path.lexists(lexical):
            continue
        link = os.path.islink(lexical)
        resolved = os.path.realpath(lexical)
        if not link and resolved == lexical:
            continue
        if link and not named:
            if not _path_inside_any_case(resolved, home_root) \
                    and not _path_inside_any_case(home_root, resolved) \
                    and not any(_path_inside_any_case(resolved, w)
                                or _path_inside_any_case(w, resolved)
                                for w in writable_roots):
                continue
        return (
            "fence path rejected (%s protected configuration name %s is a "
            "symlink or resolves outside its location)" % (label, name)
        )
    return None


def _stat_dev_ino(path):
    """``(st_dev, st_ino)`` for *path*, or None when it does not exist.

    Only ENOENT, ENOTDIR, and ENAMETOOLONG mean "does not exist". Any other
    OSError is re-raised so the caller can fail closed.
    """
    try:
        st = os.stat(path)
        return (st.st_dev, st.st_ino)
    except OSError as exc:
        if getattr(exc, "errno", None) in (
            errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG,
        ):
            return None
        raise


class _IdentityUnavailable(Exception):
    """Non-ENOENT stat failure: filesystem identity cannot be established."""

    def __init__(self, path=""):
        self.path = path
        super().__init__("identity could not be established for %s" % path)


def _lexical_ci_same_or_inside(path, root) -> bool:
    """Folded (case, NFC/NFD) lexical ``path == root or path is under root``.

    The pure-string half of the judgement: no filesystem access, so it cannot
    see an alias. Production code uses ``_location_inside_core``; this stays
    for callers that hold only spellings. A match means REFUSE, so folding
    can only refuse more, never admit more.
    """
    p = _fold_path(os.path.normcase(str(path))).rstrip(os.sep)
    r = _fold_path(os.path.normcase(str(root))).rstrip(os.sep)
    return p == r or p.startswith(r + os.sep)


def _ci_same_or_inside(path, root) -> bool:
    """``path == root or path is under root``, by filesystem identity.

    Refusal-side containment that RAISES instead of choosing a side: the
    nearest existing ancestor is compared by ``st_dev``/``st_ino`` and the
    names below it folded (``_location_inside_core``), so a path that does
    not exist yet is judged as precisely as one that does. A stat failure
    other than a missing path raises ``_IdentityUnavailable`` so the caller
    picks the closed side (containment-in-home -> refuse; launcher-root
    membership -> not admitted; dotdir exemption -> not exempt).
    """
    return _location_inside_core(path, root, fold=True)


def home_tree_rejection(rp_cfg, home, allow_dotdir_prefix=None) -> str | None:
    """Why *rp_cfg* may not be a vendor config dir, as a short code, else None.

    ``"is_home"``: it is the home tree, or contains it.
    ``"inside_home"``: it resolves anywhere inside home without being a direct
    child named ``<allow_dotdir_prefix><name>``.

    The second is the one that is easy to get wrong. Checking only the direct
    child leaves every deeper path admitted — ``~/Library/Keychains`` has a
    parent of ``~/Library``, so a "no direct children" rule waves it through,
    and a vendor config dir is not a reason to hand a fenced run write access
    to an arbitrary home subtree (issue #102).

    For CALLER-SUPPLIED values. Launcher-owned per-run config dirs
    (``<job_dir>/cursor-config``, a per-run mkdtemp) are admitted by
    ``_caller_config_dir_rejection`` via an explicit ``launcher_roots``
    allowlist, not by depth.
    """
    try:
        contains_home = _ci_same_or_inside(home, rp_cfg)
    except _IdentityUnavailable:
        contains_home = True
    if contains_home:
        return "is_home"
    try:
        inside_home = _ci_same_or_inside(rp_cfg, home)
    except _IdentityUnavailable:
        inside_home = True
    if not inside_home:
        return None
    parent = os.path.dirname(str(rp_cfg).rstrip(os.sep))
    base = os.path.basename(str(rp_cfg).rstrip(os.sep))
    prefix = str(allow_dotdir_prefix or "")
    # Exemption side: the dotdir carve-out applies only to a parent that is
    # home by exact spelling or by inode, never by folding.
    parent_is_home = _same_dir_exact(parent, home)
    if (prefix
            and parent_is_home
            and base.casefold().startswith(prefix.casefold())
            and len(base) > len(prefix)):
        return None
    return "inside_home"


def _same_dir(path, root) -> bool:
    """Refusal-side: *path* and *root* are the same directory (identity).

    Raises ``_IdentityUnavailable`` when a non-ENOENT stat error prevents
    the comparison; callers pick the closed side.
    """
    return _ci_same_or_inside(path, root) and _ci_same_or_inside(root, path)


def _launcher_root_usable(rp_root) -> bool:
    """False when *rp_root* would admit a widening tree.

    A launcher root equal to home, ``~/Library``, or any other
    ``_widening_roots()`` member is refused (filesystem identity, so a
    firmlink spelling of home is still home). If ``OMNI_LLM_STATE_DIR``
    itself resolves to a widening root, a jobs root under it must not
    become an any-descendant allowlist. An unreadable state dir or an
    unreadable widening-root identity refuses.
    """
    try:
        for w in _widening_roots():
            if _same_dir(rp_root, w):
                return False
    except _IdentityUnavailable:
        return False
    from providers import jobs
    try:
        state = os.path.realpath(jobs._state_dir())
    except OSError as e:
        log_event("launcher_root_refused", reason="state_dir_unreadable", error=str(e))
        return False
    state_is_widening = False
    for w in _widening_roots():
        try:
            if _same_dir(state, w):
                state_is_widening = True
                break
        except _IdentityUnavailable:
            log_event("launcher_root_refused", reason="identity_unavailable", root=str(w))
            return False
    if not state_is_widening:
        return True
    try:
        if _ci_same_or_inside(rp_root, state):
            return False
    except _IdentityUnavailable:
        return False
    return True


def _is_widening_or_home_ancestor(rp_cfg, home) -> bool:
    """True when *rp_cfg* is a widening root or an ancestor of *home*.

    Widening-root membership is filesystem identity, not string equality
    (a firmlink spelling of ``/Users`` is still ``/Users``). Ancestors of
    home are walked by inode; on Darwin the data-volume firmlink spelling
    of home is walked too, because ``dirname(/Users/...)`` never visits
    ``/System/Volumes/Data`` (distinct inode from ``/``).
    """
    for w in _widening_roots():
        if _same_dir(rp_cfg, w):
            return True
    try:
        cfg_id = _stat_dev_ino(rp_cfg)
    except OSError as exc:
        raise _IdentityUnavailable(rp_cfg) from exc
    if cfg_id is None:
        return False
    starts = [str(home)]
    data = "/System/Volumes/Data"
    home_s = str(home)
    if os.path.isdir(data) and home_s != data and not home_s.startswith(
            data + os.sep):
        alt = data + home_s
        if os.path.isdir(alt):
            starts.append(alt)
    seen = set()
    for start in starts:
        current = start
        while current not in seen:
            seen.add(current)
            try:
                cur_id = _stat_dev_ino(current)
            except OSError as exc:
                raise _IdentityUnavailable(current) from exc
            if cur_id is not None and cur_id == cfg_id:
                return True
            parent = os.path.dirname(current.rstrip(os.sep) or current)
            if parent == current:
                break
            current = parent
    return False


def _caller_config_dir_rejection(
    config_dir,
    *,
    label,
    home=None,
    allow_dotdir_prefix=None,
    refuse_contains_home=False,
    must_exist=False,
    refuse_default_login=None,
    launcher_roots=(),
):
    """Return a LOUD detail if *config_dir* must not enter the fence, else None.

    Shared admission for a caller-supplied vendor root (cursor's
    ``CURSOR_CONFIG_DIR``, claude's profile ``CLAUDE_CONFIG_DIR``). Refused
    when the realpath is a widening root or anywhere inside home, except
    (a) a direct child of home whose basename is *allow_dotdir_prefix*
    plus a non-empty remainder (claude: ``.claude-<name>``, cursor:
    ``.cursor-<name>``), and (b) a path inside an explicit
    *launcher_roots* allowlist (the jobs root, the system temp root).
    Launcher roots that resolve to a widening root are ignored.
    ``refuse_default_login`` (claude: ``.claude``) refuses that exact
    home-child even via symlink: the default login directory cannot be a
    profile. Claude refuses a missing path. Cursor and claude refuse a
    root that contains home.
    """
    if isinstance(config_dir, AdmittedPath):
        # Caller already admitted this object. Do not realpath a second
        # time: a swap between admission and this check must not replace
        # the validated identity with a later target.
        rp_cfg = config_dir.path
        given = config_dir.path
    else:
        given = config_dir
        rp_cfg = os.path.realpath(config_dir)
    if must_exist and not (given and os.path.isdir(os.fspath(given))):
        return (
            "fence path rejected (%s is unset or not an existing directory): %s"
            % (label, config_dir)
        )
    if home is None:
        home = protected_home()
    identity_detail = (
        "fence path rejected (%s identity could not be established): %s -> %s"
        % (label, config_dir, rp_cfg)
    )
    try:
        widening = _is_widening_or_home_ancestor(rp_cfg, home)
    except _IdentityUnavailable:
        return identity_detail
    if widening:
        return (
            "fence path rejected (%s resolves to a widening root): %s -> %s"
            % (label, config_dir, rp_cfg)
        )
    if refuse_contains_home:
        try:
            contains_home = _ci_same_or_inside(home, rp_cfg)
        except _IdentityUnavailable:
            contains_home = True
        if contains_home:
            return (
                "fence path rejected (%s must be a directory inside or beside "
                "the home tree, not the home tree itself): %s -> %s"
                % (label, config_dir, rp_cfg)
            )
    if refuse_default_login:
        default_root = os.path.realpath(os.path.join(home, refuse_default_login))
        if rp_cfg == default_root or _same_path_any_case(rp_cfg, default_root):
            return (
                "fence path rejected (%s is the default login directory and "
                "cannot be a profile): %s -> %s"
                % (label, config_dir, rp_cfg)
            )
    for raw_root in launcher_roots or ():
        rp_root = os.path.realpath(raw_root)
        try:
            _stat_dev_ino(rp_root)
        except OSError:
            return identity_detail
        try:
            usable = _launcher_root_usable(rp_root)
        except _IdentityUnavailable:
            return identity_detail
        if not usable:
            continue
        # Exemption side: membership of an allowed launcher root is exact
        # spelling or inode, never folding (exempt_inside).
        if exempt_inside(rp_cfg, rp_root):
            return None
    parent = os.path.dirname(rp_cfg.rstrip(os.sep) or rp_cfg)
    base = os.path.basename(rp_cfg.rstrip(os.sep) or rp_cfg)
    # Exemption side: the dotdir carve-out applies only to a parent that is
    # home by exact spelling or by inode, never by folding.
    parent_is_home = _same_dir_exact(parent, home)
    if (allow_dotdir_prefix
            and parent_is_home
            and base.startswith(allow_dotdir_prefix)
            and len(base) > len(allow_dotdir_prefix)):
        return None
    if parent_is_home:
        return (
            "fence path rejected (%s is a direct child of home; use a "
            "dedicated ~/.<vendor>-<name>, or a directory outside home): "
            "%s -> %s"
            % (label, config_dir, rp_cfg)
        )
    try:
        inside_home = _ci_same_or_inside(rp_cfg, home)
    except _IdentityUnavailable:
        inside_home = True
    if inside_home:
        if allow_dotdir_prefix:
            hint = (
                "use a dedicated ~/%s<name> directory, or one outside home"
                % allow_dotdir_prefix
            )
        else:
            hint = "use a dedicated vendor dotdir, or one outside home"
        return (
            "fence path rejected (%s is inside the home tree; %s): %s -> %s"
            % (label, hint, config_dir, rp_cfg)
        )
    return None


def _drop_inside_codex_home(entries, codex_root, label, bound_of):
    """``(kept, refusal)``: late allows against a denied codex home.

    An entry inside the home (or the home itself, or one that cannot be
    bounded) is dropped, as before. An entry that CONTAINS the home is
    refused: the deny precedes the allow, so it would re-open the tree.
    Both judgements go through ``location_inside``, so a firmlink spelling,
    another case or normalization, a symlinked home, or a home that does
    not exist yet is the same location.
    """
    kept = []
    refusal = None
    for entry in entries:
        bound = bound_of(entry)
        if bound is None:
            continue
        path = bound[1]
        if location_inside(path, codex_root):
            continue
        if location_inside(codex_root, path):
            refusal = refusal or (
                "fence path rejected (a %s allow contains the codex home, "
                "which this provider may not touch): %s" % (label, path)
            )
            continue
        kept.append(entry)
    return kept, refusal


def _collect_agent_fence(provider, workdir=None, read_only=False, create=False,
                         config_dir=None, launcher_roots=()):
    """Build path sets (see ``_collect_agent_fence_uncached``).

    One policy build compares the same few roots many times; its location
    walks are shared. Not when ``create`` is set: that mode makes
    directories and a stale walk of a path it just made would mislead.
    """
    with (contextlib.nullcontext() if create else _location_cache()):
        return _collect_agent_fence_uncached(
            provider, workdir=workdir, read_only=read_only, create=create,
            config_dir=config_dir, launcher_roots=launcher_roots)


def _collect_agent_fence_uncached(provider, workdir=None, read_only=False,
                                  create=False, config_dir=None,
                                  launcher_roots=()):
    """Build path sets. ``rejected`` is a LOUD detail or None.

    ``create`` mkdir/touch's vendor session/lock paths so they can enter the
    write allow-set; ``agent_fence_paths`` leaves it False so a query does
    not mutate the user's vendor state.

    ``config_dir`` is the caller-supplied vendor config root for this run:
    cursor's ``CURSOR_CONFIG_DIR`` (a fresh temp dir for the sync lane,
    ``<job_dir>/cursor-config`` for the job lane, or an operator-set
    account-profile override), claude's profile ``CLAUDE_CONFIG_DIR``, or
    codex's profile ``CODEX_HOME`` (re-checked here with
    ``home_tree_rejection``, the same rule its producer applied).
    A home descendant is refused unless it is a dedicated
    ``.<vendor>-<name>`` direct child or it sits inside *launcher_roots*
    (jobs root / temp root). Resolving to home itself or another widening
    root is a LOUD rejection. The default ``~/.claude`` cannot be a
    profile. Claude then uses that root in place of ``~/.claude`` — the
    default login is not re-allowed.
    """
    if provider not in ("claude", "pi", "cursor", "codex", "grok", "agy"):
        raise ValueError(
            "agent_fence_paths provider must be 'claude', 'pi', 'cursor' "
            "'codex', 'grok' or 'agy'"
        )

    rejected = None
    extra_leaves = []
    if provider == "claude" and workdir is not None and not config_dir:
        extra_leaves.append(
            os.path.join(".claude", "projects", claude_encoded_workdir(workdir))
        )

    def admit(path):
        nonlocal rejected
        rp = os.path.realpath(path)
        admitted = _admit_home_leaf(rp, extra_leaves=extra_leaves)
        if admitted is None:
            if os.path.lexists(path) or os.path.exists(path):
                rejected = (
                    "fence path rejected (not an exact listed leaf after "
                    "realpath): %s -> %s" % (path, rp)
                )
            return None
        return admitted

    reads: list[AdmittedPath] = []
    writes: list[AdmittedPath] = []
    seen_r: set[str] = set()
    seen_w: set[str] = set()

    def _try_admit(path, kind, *, label) -> AdmittedPath | None:
        nonlocal rejected
        try:
            return admit_path(path, kind, label=label)
        except FencePathRejected as exc:
            if not rejected:
                rejected = exc.detail
            return None

    def add_read(path, must_exist=True):
        if must_exist and not os.path.exists(path):
            return
        admitted = admit(path)
        if not admitted:
            return
        kind = "subpath" if os.path.isdir(admitted) else "literal"
        ap = _try_admit(admitted, kind, label="agent fence read")
        if ap and ap.path not in seen_r:
            seen_r.add(ap.path)
            reads.append(ap)

    def add_write(path):
        if not os.path.exists(path):
            return
        admitted = admit(path)
        if not admitted:
            return
        kind = "subpath" if os.path.isdir(admitted) else "literal"
        ap = _try_admit(admitted, kind, label="agent fence write")
        if ap and ap.path not in seen_w:
            seen_w.add(ap.path)
            writes.append(ap)

    # Binaries: vendor binary admission is admit_vendor_binary (shared with
    # consult lanes that pass their own policy); node below keeps the same
    # outside-home / literal-inside-home rule.
    vendor_bin = resolve_agent_binary(provider)
    home = protected_home()
    try:
        ap_bin = admit_vendor_binary(vendor_bin)
    except FencePathRejected as exc:
        ap_bin = None
        rejected = exc.detail
    if ap_bin and ap_bin.path not in seen_r:
        seen_r.add(ap_bin.path)
        reads.append(ap_bin)

    node_bin = shutil.which("node")
    if node_bin:
        node_real = os.path.realpath(node_bin)
        if _path_inside(node_real, home) and os.path.isfile(node_real) \
                and node_real not in seen_r:
            ap_node = _try_admit(node_real, "literal", label="node binary")
            if ap_node and ap_node.path not in seen_r:
                seen_r.add(ap_node.path)
                reads.append(ap_node)

    tmpdir = None
    profile_root = None
    codex_profile_root = None
    codex_deny: list[tuple[str, str]] = []
    grok_deny: list[tuple[str, str]] = []
    agy_deny: list[tuple[str, str]] = []
    grok_root = None
    deny_link: list[tuple[str, str]] = []
    post_write: list[tuple[str, str]] = []
    deny_read: list[tuple[str, str]] = []
    post_read: list[tuple[str, str]] = []
    deny_write_final: list[tuple[str, str]] = []
    deny_read_final: list[tuple[str, str]] = []
    cursor_cfg_root = None
    vendor_root = None      # realpath the pi / cursor state paths were spelled at
    claude_config_root = None   # named claude home (realpath) this policy was built for

    if provider == "cursor" and vendor_bin and _path_inside(vendor_bin, home) \
            and not rejected:
        # cursor-agent's launcher is a shell wrapper that `realpath`s itself
        # and execs a NODE BINARY + index.js sitting next to it in the same
        # versioned install dir (live-bisected 2026-09-10: the literal-file
        # read alone was not enough — "node: No such file or directory").
        # That whole install dir is re-allowed for READ as a post-deny
        # clause, never write: it is the immutable program bundle, not
        # vendor state, and the leaf is derived from the resolved binary
        # rather than hardcoded so a version bump does not need a new leaf.
        # Fix round 3 (C): an OMNI_LLM_CURSOR_BIN override pointing at
        # e.g. ~/cursor-agent must never turn this into a home-tree
        # re-allow — refuse LOUD instead of re-allowing home itself or a
        # direct child of it (the normal install dir is several levels
        # deep: ~/.local/share/cursor-agent/versions/<ver>/cursor-agent).
        bin_dir = os.path.dirname(vendor_bin)
        if bin_dir in _widening_roots() or bin_dir == home \
                or os.path.dirname(bin_dir) == home:
            rejected = (
                "fence path rejected (cursor binary directory is home "
                "itself or a direct child of home, refusing to re-allow "
                "it for read): %s" % bin_dir
            )
        else:
            post_read.append(_entry("subpath", bin_dir))

    if provider == "claude":
        profile_root = None
        if config_dir and not rejected:
            ap_cfg = _try_admit(
                config_dir, "subpath", label="CLAUDE_CONFIG_DIR",
            )
            if ap_cfg is None:
                pass
            else:
                cfg_err = _caller_config_dir_rejection(
                    ap_cfg,
                    label="CLAUDE_CONFIG_DIR",
                    home=home,
                    allow_dotdir_prefix=".claude-",
                    refuse_contains_home=True,
                    must_exist=True,
                    refuse_default_login=".claude",
                    launcher_roots=launcher_roots,
                )
                if cfg_err:
                    rejected = cfg_err
                else:
                    profile_root = ap_cfg.path
                    claude_config_root = profile_root
                    if profile_root not in seen_r:
                        seen_r.add(profile_root)
                        reads.append(ap_cfg)
                    claude_json = os.path.join(profile_root, ".claude.json")
                    if os.path.exists(claude_json) and claude_json not in seen_r:
                        ap_json = _try_admit(
                            claude_json, "literal", label=".claude.json",
                        )
                        if ap_json and ap_json.path not in seen_r:
                            seen_r.add(ap_json.path)
                            reads.append(ap_json)
                    if profile_root not in seen_w:
                        seen_w.add(profile_root)
                        writes.append(ap_cfg)

        if not profile_root and not config_dir:
            add_read(os.path.join(home, ".claude"))
            add_read(os.path.join(home, ".claude.json"))
        # A2: keychain FILE first; directory also listed so a login that
        # needs sibling keychain files still works. Never writable.
        add_read(os.path.join(home, "Library", "Keychains", "login.keychain-db"))
        add_read(os.path.join(home, "Library", "Keychains"))

        scratch, scratch_err = claude_scratch_root()
        if scratch_err:
            rejected = scratch_err
        elif scratch:
            # Scratch is outside home; A5 validates it separately, not as a leaf.
            ap_scratch = _try_admit(scratch, "subpath", label="claude scratch")
            if ap_scratch and ap_scratch.path not in seen_w:
                seen_w.add(ap_scratch.path)
                writes.append(ap_scratch)
            tmpdir = scratch

        if profile_root:
            claude_root = profile_root
        elif not config_dir:
            claude_root = os.path.join(home, ".claude")
        else:
            claude_root = None
        if claude_root and os.path.isdir(claude_root):
            if not profile_root and not config_dir:
                # Default home: the T* rows below are spelled at the RESOLVED
                # root, so a ~/.claude reached through a link the run can
                # write must be judged (pi_cursor_root_link_refusal, after
                # the provider branches), exactly like ~/.pi / ~/.cursor.
                vendor_root = os.path.realpath(claude_root)
            # Vendor state is re-allowed as post-deny write clauses (T*), which
            # need no existing path: this code never creates or touches
            # anything under ~/.claude (A3: a query must not mutate vendor state).
            #
            # Every per-run allowance below is a LEXICAL path under the
            # realpath of the claude root, and the build refuses when any
            # component under that root is a symlink. These leaves are
            # writable by the previous child, which could plant
            # `todos -> <another account's home>`; resolving it here would
            # turn the next run's allowance into a write (or, for the
            # session dir, read) grant on that other home.
            def _leaf(*parts):
                nonlocal rejected
                # Default home: spell from the recorded vendor_root, so the
                # recorded root and every row are one realpath snapshot.
                path, why = _claude_run_leaf(
                    vendor_root or claude_root, *parts,
                    resolved=bool(vendor_root))
                if why and not rejected:
                    rejected = why
                return path

            if workdir is not None:
                session = _leaf("projects", claude_encoded_workdir(workdir))
                if session:
                    post_write.append(_lexical_entry("subpath", session))
                    post_read.append(_lexical_entry("subpath", session))
            if not read_only:
                for leaf in ("todos", "shell-snapshots", "debug"):
                    path = _leaf(leaf)
                    if path:
                        post_write.append(_lexical_entry("subpath", path))
            projects = _leaf("projects")
            if projects:
                deny_read.append(_entry("subpath", projects))
                if profile_root:
                    # A fresh profile root has no projects/ yet, and the W* deny
                    # on that subtree also covers creating the directory node
                    # itself, so claude's first session mkdir failed EPERM
                    # (bisected live 2026-09-10). Re-allow the NODE as a literal:
                    # the subtree stays deny-write except this run's session dir.
                    # ~/.claude/projects always exists, so the default root is
                    # unchanged.
                    post_write.append(_lexical_entry("literal", projects))
    elif provider == "codex":
        # ~/.codex (or this run's CODEX_HOME) is denied whole for write, then
        # T re-allows only the state leaves. Read still re-allows the tree
        # (the vendor must load config); _write_allow_paths /
        # _read_reallow_entries add the root for the consult lane too, and
        # the W deny is what closes write. The state set is the directories
        # in _CODEX_STATE_DIRS, existing .sqlite/.db plus derived -wal/-shm,
        # and the named state files. .tmp/ is not in that set. An agent may
        # churn those state leaves, never rewrite what the NEXT codex run is
        # told to do.
        #
        # Accepted exposure, stated rather than advertised away: auth.json
        # stays writable, because codex rewrites its own credential store on
        # token refresh (same call as pi's auth.json, issue #77), and the job
        # lane drops --ephemeral so a long-running agent persists sessions
        # here for --resume.
        if config_dir and not rejected:
            # CODEX_HOME from an account profile. admit_path evaluates
            # once; home_tree_rejection / widening checks run on ap.path,
            # and the AdmittedPath is what enters extras. Same re-check
            # the consult lane does (issue #102 / #120).
            ap_cfg = _try_admit(config_dir, "subpath", label="CODEX_HOME")
            if ap_cfg is None:
                pass
            else:
                rp_cfg = ap_cfg.path
                why = home_tree_rejection(rp_cfg, home, ".codex-")
                try:
                    widening = _is_widening_or_home_ancestor(rp_cfg, home)
                except _IdentityUnavailable:
                    widening = True
                if why is not None or widening:
                    rejected = (
                        "fence path rejected (CODEX_HOME resolves %s): %s -> %s"
                        % ("to a widening root" if why is None else why,
                           config_dir, rp_cfg)
                    )
                elif not os.path.isdir(rp_cfg):
                    rejected = (
                        "fence path rejected (CODEX_HOME is not an existing "
                        "directory): %s -> %s" % (config_dir, rp_cfg)
                    )
                else:
                    codex_profile_root = rp_cfg
                    if rp_cfg not in seen_r:
                        seen_r.add(rp_cfg)
                        reads.append(ap_cfg)
                    if rp_cfg not in seen_w:
                        seen_w.add(rp_cfg)
                        writes.append(ap_cfg)
                    default_root = os.path.realpath(os.path.join(home, ".codex"))
                    if not _path_inside(rp_cfg, default_root) \
                            and not _path_inside(default_root, rp_cfg):
                        # Sibling-profile isolation, same shape as claude's ring
                        # (#72): the default account's transcripts and auth.json
                        # stay unreadable from a profile run. Skipped when one
                        # root contains the other, so a deny cannot punch a hole
                        # in the allowed root.
                        deny_read.append(_entry("subpath", default_root))
                        deny_read_final.append(_entry("subpath", default_root))
        elif not config_dir:
            # No profile: the default root. When a profile IS set it is this
            # run's whole vendor state and ~/.codex is NOT re-allowed
            # alongside it -- the same way a claude profile replaces
            # ~/.claude rather than joining it.
            add_read(os.path.join(home, ".codex"))
            codex_profile_root = None
        # Deny the codex home WHOLE, then re-allow only the state leaves.
        # _write_allow_paths adds ~/.codex unconditionally (the consult lane's
        # contract, shared through build_profile) and a profile root is in
        # `writes`, so without this the tree is open and the instruction
        # surface is decided by a deny list that review showed cannot be
        # completed by enumeration.
        # T entries are LEXICAL (never realpath): a symlink at a
        # post-allowed name must not put its target in T. Any candidate
        # that is a symlink, escapes the root, or resolves to a widening
        # root refuses the spawn -- never a silent drop.
        if config_dir and codex_profile_root:
            lexical_root = os.path.abspath(config_dir)
            if codex_profile_root != lexical_root:
                rejected = (
                    "fence path rejected (CODEX_HOME realpath differs from "
                    "lexical): %s -> %s"
                    % (lexical_root, codex_profile_root)
                )
        elif not config_dir and os.path.isdir(os.path.join(home, ".codex")):
            lexical_root = os.path.abspath(os.path.join(home, ".codex"))
        else:
            lexical_root = None
        if lexical_root and not rejected:
            why = _codex_post_allow_refusal(lexical_root, lexical_root)
            if why:
                rejected = why
            else:
                def _emit_codex_post(kind, path):
                    nonlocal rejected
                    if rejected:
                        return
                    why_c = _codex_post_allow_refusal(path, lexical_root)
                    if why_c:
                        rejected = why_c
                        return
                    post_write.append(_lexical_entry(kind, path))

                # Deny still realpaths: the W clause names the resolved
                # tree. T entries below are lexical. link(2) is gated by
                # file-write* AND file-link on both endpoints; this clause
                # is an independent second gate that does not rely on that
                # coupling; it is emitted last so an earlier (allow
                # default) cannot undo it.
                codex_deny.append(_entry("subpath", lexical_root))
                deny_link.append(_entry("subpath", lexical_root))
                for leaf in _CODEX_STATE_DIRS:
                    # No existence check: a subpath clause covers creating
                    # the directory node itself, and codex makes these on
                    # demand. islink is still checked; a planted symlink
                    # at the name refuses the next spawn.
                    _emit_codex_post(
                        "subpath", os.path.join(lexical_root, leaf),
                    )
                if not rejected:
                    present_state = set()
                    for name in sorted(os.listdir(lexical_root)):
                        path = os.path.join(lexical_root, name)
                        if name in _CODEX_STATE_FILES:
                            _emit_codex_post("literal", path)
                            present_state.add(name)
                        elif name.endswith(_CODEX_STATE_DB_SUFFIXES):
                            # Base name is the anchor; -wal/-shm are
                            # derived whether or not they exist now.
                            _emit_codex_post("literal", path)
                            _emit_codex_post("literal", path + "-wal")
                            _emit_codex_post("literal", path + "-shm")
                    for name in _CODEX_STATE_FILES:
                        if name in present_state:
                            continue
                        # Re-allow the NAME so codex can create it on a
                        # fresh home; the subtree stays denied otherwise.
                        _emit_codex_post(
                            "literal", os.path.join(lexical_root, name),
                        )
        # tmpdir stays None on purpose: the child keeps the inherited TMPDIR,
        # which is exactly the path _write_allow_paths validated and allowed.
        # Pinning a different one (pi/cursor pin _system_tempdir_default())
        # would hand codex a TMPDIR the profile does not allow. The consult
        # lane leaves TMPDIR alone for the same reason.
    elif provider == "grok":
        # Read login/config, but post-allow writes only to named state.
        # Use the existing lexical-state validator: a symlink at a writable
        # leaf must never turn its target into a last-match allow clause.
        default_root = os.path.join(home, ".grok")
        lexical_root = os.path.abspath(config_dir or default_root)
        ap_root = _try_admit(lexical_root, "subpath", label="GROK_HOME")
        if ap_root and config_dir and lexical_root != default_root:
            rejected = rejected or _caller_config_dir_rejection(
                ap_root, label="GROK_HOME", home=home,
                allow_dotdir_prefix=".grok-", refuse_contains_home=True,
                launcher_roots=launcher_roots,
            )
        if ap_root and not rejected:
            grok_root = ap_root.path
            if grok_root != lexical_root:
                rejected = "fence path rejected (GROK_HOME realpath differs from lexical)"
            else:
                reads.append(ap_root)
                grok_deny.append(_entry("subpath", grok_root))
                deny_link.append(_entry("subpath", grok_root))
                for kind, names in (("subpath", _GROK_STATE_DIRS),
                                    ("literal", _GROK_STATE_FILES)):
                    for name in names:
                        path = os.path.join(lexical_root, name)
                        why = _codex_post_allow_refusal(path, lexical_root)
                        if why:
                            rejected = why.replace("codex", "grok")
                            break
                        post_write.append(_lexical_entry(kind, path))
                    if rejected:
                        break
                if not rejected:
                    post_write.append(
                        _lexical_entry(GROK_TMP_KIND, lexical_root)
                    )
                # GROK_HOME plus the home-level compat roots grok scans for
                # the same names (the read wall already covers the latter;
                # these rows keep that true if a future re-allow lands).
                for root in (grok_root, os.path.join(home, ".claude"),
                             os.path.join(home, ".cursor")):
                    rows, why = _grok_instruction_deny_read(root)
                    deny_read.extend(rows)
                    if why:
                        rejected = rejected or why
                        break
    elif provider == "agy":
        # Login rides the keychain, as for claude and cursor.
        add_read(os.path.join(home, "Library", "Keychains", "login.keychain-db"))
        add_read(os.path.join(home, "Library", "Keychains"))
        tmpdir = _system_tempdir_default()
        # FAIL CLOSED on any linked or aliased agy dot tree (see
        # ``agy_layout_refusal``): every allow and deny below is spelled once
        # at the plain path, which is only sound when no link, hard link or
        # alternate spelling can make another name for the same file.
        layout_why = agy_layout_refusal(home)
        if layout_why:
            rejected = rejected or layout_why
        else:
            agy_dir = os.path.join(home, _AGY_STATE_ROOT)
            if os.path.isdir(agy_dir):
                for kind, leaf in _AGY_READ_LEAVES:
                    post_read.append((kind, os.path.join(agy_dir, leaf)))
                for kind, leaf in _AGY_WRITE_LEAVES:
                    path = os.path.join(agy_dir, leaf)
                    why = _hard_linked_state_refusal("agy", path)
                    if why:
                        rejected = rejected or why
                        break
                    post_write.append((kind, path))
                # The default project record.
                post_read.append(
                    ("subpath", os.path.join(home, *_AGY_PROJECTS_LEAF.split(os.sep))))
                vendor_root = agy_dir
            # No NEW hard link to or from anything under ~/.gemini, in both
            # spellings (a rule spelled with the data-volume prefix does not
            # match, measured; the plain one is what the kernel compares).
            gemini = os.path.join(home, ".gemini")
            for spelled in _agy_spellings(gemini):
                deny_link.append(("subpath", spelled))
            # Everything agy would load or execute from its dot dir, and the
            # instruction family, is denied for read AND write as FINAL rows
            # (after every re-allow).
            agy_rows, why = agy_instruction_deny_rows(home)
            if why:
                rejected = rejected or why
            deny_read.extend(agy_rows)
            deny_read_final.extend(agy_rows)
            agy_deny.extend(agy_rows)
            deny_write_final.extend(agy_rows)
    elif provider == "pi":
        add_read(os.path.join(home, ".pi"))
        pi_root = os.path.join(home, ".pi")
        if os.path.isdir(pi_root):
            # The root is resolved ONCE: every leaf, the file-link deny and
            # vendor_root below are spelled from this one value, so a root
            # swapped while the fence is built cannot yield allowances under
            # one tree and a recorded root of another.
            pi_real = os.path.realpath(pi_root)
            # pi's settings lock is a DIRECTORY it mkdir/rmdir's itself; a
            # subpath clause covers create and remove of the path and its
            # children. Never pre-create it: a regular file at that path
            # breaks pi startup (ENOTDIR on rmdir).
            # pi rewrites its own credential store on token refresh; without
            # write access to auth.json every fenced run ends "No API key
            # found" (bisected live 2026-09-09). Accepted exposure: the agent
            # can alter pi's stored credentials, nothing else under ~/.pi.
            # Each leaf is writable, so the previous child could have
            # replaced it with a symlink: refuse one, never resolve it into
            # a late write allow at the link's target (issue #149).
            for kind, leaf in (("subpath", ("agent", "sessions")),
                               ("subpath", ("agent", "settings.json.lock")),
                               ("subpath", ("agent", "auth.json.lock")),
                               ("literal", ("agent", "auth.json"))):
                path, why = _claude_run_leaf(pi_real, *leaf, vendor="pi")
                why = why or _hard_linked_state_refusal("pi", path)
                if why:
                    rejected = rejected or why
                    break
                post_write.append((kind, path))
            # Like the codex home: the file-link deny blocks creating a hard
            # link to or from the vendor root DURING a run, so a read-only
            # file in it cannot gain a writable alias in the workdir and a
            # state file cannot gain a second name. It says nothing about
            # links that already exist: those are only scanned for the
            # writable leaves above, never for read-only vendor files.
            deny_link.append(("subpath", pi_real))
            vendor_root = pi_real
        tmpdir = _system_tempdir_default()
    else:  # cursor
        add_read(os.path.join(home, ".cursor"))
        # Login is Keychain, same as claude (2026-09-10 live capture).
        add_read(os.path.join(home, "Library", "Keychains", "login.keychain-db"))
        add_read(os.path.join(home, "Library", "Keychains"))

        cursor_root = os.path.join(home, ".cursor")
        if os.path.isdir(cursor_root):
            # Resolved once, like pi's root: leaf, deny and vendor_root agree.
            cursor_real = os.path.realpath(cursor_root)
            # Fix round 3 (A): add_read(~/.cursor) above grants the whole
            # tree; carve the human's own secrets/transcripts back out.
            # Live-bisected 2026-09-10: acceptance C (write lane) still
            # ok:true with all three denied.
            deny_read.append(_entry("literal", os.path.join(cursor_root, "mcp.json")))
            deny_read.append(_entry("subpath", os.path.join(cursor_root, "chats")))
            deny_read.append(
                _entry("literal", os.path.join(cursor_root, "cli-config.json"))
            )
            # Fix round 3 (B2): STOPPED, not narrowed — reported, not
            # silently left broken. The literal-encoding slug form is
            # live-verified (dots/underscores/spaces), but a longer workdir
            # path produced a slug that form cannot predict: cursor-agent
            # TRUNCATES the encoded path past some length and appends what
            # looks like a short hash suffix, e.g. observed
            # "private-var-folders-sz-67vzb7p16j58fcswyls0rmch0-7c91d62" for
            # workdir ".../67vzb7p16j58fcswyls0rmch0000gn/T/tmp.dvSyEouufD"
            # — truncated mid-word, "-7c91d62" is not a prefix of that
            # path's md5/sha1/sha256/crc32. Per the work order's Step 5,
            # guessing further at an unverified hash was not attempted;
            # falls back to the known-working full-subtree write access
            # (same accepted exposure as skills-cursor/ai-tracking below —
            # see REPORT-cursor-agent-lane.md for the bisect table).
            path, why = _claude_run_leaf(cursor_real, "projects", vendor="cursor")
            why = why or _hard_linked_state_refusal("cursor", path)
            if why:
                rejected = rejected or why
            else:
                post_write.append(("subpath", path))
            # No NEW hard link to or from the cursor root (see pi above).
            deny_link.append(("subpath", cursor_real))
            vendor_root = cursor_real
            # Fix round 3 (B1/B3): skills-cursor (Cursor's own skill
            # definitions — a task could plant one), ai-tracking, and
            # plugins/cache are NOT re-allowed for write. Live-bisected
            # 2026-09-10: acceptance C still ok:true, hello.txt still
            # created, with all three denied — none is required for an
            # ordinary agent run. Accepted exposure if a future cursor-agent
            # version needs one of these: re-add it here with the observed
            # failure line, never blanket-widen.
        tmpdir = _system_tempdir_default()

        # This run's CURSOR_CONFIG_DIR: a temp dir, <job_dir>/cursor-config
        # (job_dir defaults under ~/.local/state/omni-llm — INSIDE home), or
        # an operator-set account-profile override (OMNI_LLM_CURSOR_CONFIG_DIR
        # / _<PROFILE>, never caller/task-controlled). Home descendants are
        # refused unless they are a dedicated ~/.cursor-<name> dotdir or sit
        # inside launcher_roots (the jobs root / the temp root). A typo'd
        # override must not grant all of home, ~/Library, or an arbitrary
        # home subtree.
        if config_dir and not rejected:
            ap_cfg = _try_admit(
                config_dir, "subpath", label="CURSOR_CONFIG_DIR",
            )
            if ap_cfg is None:
                pass
            else:
                cfg_err = _caller_config_dir_rejection(
                    ap_cfg, label="CURSOR_CONFIG_DIR", home=home,
                    allow_dotdir_prefix=".cursor-",
                    refuse_contains_home=True,
                    launcher_roots=launcher_roots,
                )
                if cfg_err:
                    rejected = cfg_err
                else:
                    rp_cfg = ap_cfg.path
                    cursor_cfg_root = rp_cfg
                    if rp_cfg not in seen_w:
                        seen_w.add(rp_cfg)
                        writes.append(ap_cfg)
                        # cursor-agent also READS its own config dir (cli-config.json,
                        # statsig-cache.json, chats/) — write-only left every read
                        # there ENOENT (Seatbelt's read-deny signature: it hides
                        # existence rather than reporting EACCES, live-bisected
                        # 2026-09-10). subpath, post-deny, same as the vendor-state
                        # re-allows above; also feeds _metadata_reallow_entries so
                        # realpath(3) on paths under here works too.
                        post_read.append(_entry("subpath", rp_cfg))

    if not rejected:
        winner_root = {
            "claude": profile_root,
            "codex": codex_profile_root,
            "grok": grok_root,
            "cursor": cursor_cfg_root,
        }.get(provider)
        try:
            sibling_rows = _sibling_home_denies(provider, winner_root, home)
            deny_read.extend(sibling_rows)
            # FINAL as well: the X* rows above are followed by the S* re-allow
            # of this run's config root, so a sibling nested in it would be
            # reopened by last-match.
            deny_read_final.extend(sibling_rows)
        except FencePathRejected as exc:
            rejected = exc.detail

    deny_write = []
    # Protected-configuration rows (this loop and the two re-rooted loops
    # below) are ALSO emitted as FINAL denies: a later state allowance (T*)
    # that covers a symlink target of a protected name would otherwise win.
    for rel, kind in _AGENT_DENY_WRITE:
        rows = _deny_entries(kind, os.path.join(home, rel))
        deny_write.extend(rows)
        deny_write_final.extend(rows)
    # The codex-home deny goes in BEFORE the named rows so that ordering
    # inside the deny clause is irrelevant (they all deny); the post-write
    # re-allows land in a later clause and win by last-match.
    deny_write.extend(codex_deny)
    deny_write.extend(grok_deny)
    deny_write.extend(agy_deny)
    if provider != "codex":
        # ~/.codex reaches EVERY agent spawn's write allow-set because
        # _write_allow_paths appends codex_config_dir() unconditionally and
        # build_profile is shared with the consult lane, which does need it
        # (issue #107). claude, pi and cursor have no business in codex's
        # vendor root at all, and while it was writable a fenced run of any
        # of them could rewrite the script hooks.json invokes -- executed by
        # the next codex session, outside any fence. Denied outright here:
        # for these three the answer is "not yours", not an allow-list.
        # The consult lane is unaffected; it does not build an agent policy.
        codex_root = os.path.join(home, ".codex")
        deny_write.extend(_deny_entries("subpath", codex_root))
        deny_read.append(_entry("subpath", codex_root))
        # The deny is not the last word: build_profile emits (deny W*) then
        # (allow T*), and (deny X*) then (allow S*), so a post-allow landing
        # inside the tree would punch straight back through. Nothing does
        # today, but that is today's configuration and not an invariant --
        # an operator-set CLAUDE_CONFIG_DIR or CURSOR_CONFIG_DIR pointed at
        # a directory under ~/.codex is admitted by
        # _caller_config_dir_rejection and would re-open part of it. Drop
        # those entries rather than rely on nobody doing it. The reverse is
        # refused outright: an allow that CONTAINS the codex home (a config
        # root beside a codex home that is itself a symlink into it) would
        # re-open the whole tree, and dropping it would silently break the
        # vendor's own state.
        post_write, why = _drop_inside_codex_home(
            post_write, codex_root, "write",
            lambda entry: _write_entry_bound(*entry))
        rejected = rejected or why
        post_read, why = _drop_inside_codex_home(
            post_read, codex_root, "read", lambda entry: entry)
        rejected = rejected or why
    if codex_profile_root:
        # The rows above are spelled against ~/.codex; re-point them at this
        # run's CODEX_HOME so a profile does not quietly lose the carve-outs.
        for rel, kind in _AGENT_DENY_WRITE:
            if rel.startswith(".codex/"):
                rows = _deny_entries(
                    kind, os.path.join(codex_profile_root, rel.split("/", 1)[1])
                )
                deny_write.extend(rows)
                deny_write_final.extend(rows)
        # I3: a profile is this run's WHOLE vendor state, so the default
        # ~/.codex must not ride along. _write_allow_paths and
        # _read_reallow_entries add it unconditionally (the consult lane's
        # contract, shared through build_profile), so denying it here is what
        # actually makes "a profile replaces ~/.codex" true for this lane —
        # otherwise a run on profile `work` reads and writes the DEFAULT
        # account's auth.json.
        deny_write.append(_entry("subpath", os.path.join(home, ".codex")))
    if provider == "claude" and profile_root:
        # The default home's settings / hooks / plugins / instruction rows
        # are spelled against ~/.claude; a named home needs the same
        # carve-outs, otherwise its whole-root write allowance lets a child
        # plant a hook or settings.json that the NEXT run (or the operator's
        # own claude on that account) executes outside any fence. The
        # profile's .claude.json (user-scope MCP servers) is in the list too.
        for rel, kind in _AGENT_DENY_WRITE:
            if rel.startswith(".claude/"):
                rows = _deny_entries(
                    kind, os.path.join(profile_root, rel.split("/", 1)[1])
                )
                deny_write.extend(rows)
                deny_write_final.extend(rows)
            elif rel == ".claude.json":
                rows = _deny_entries(
                    kind, os.path.join(profile_root, ".claude.json")
                )
                deny_write.extend(rows)
                deny_write_final.extend(rows)
    if provider == "claude":
        deny_write.append(
            _entry("subpath", os.path.join(home, ".claude", "projects"))
        )
        if profile_root:
            deny_write.append(
                _entry("subpath", os.path.join(profile_root, "projects"))
            )

    if not rejected and provider in ("claude", "codex"):
        # Protected configuration names of the SELECTED home must not alias:
        # see _protected_config_refusal. Default and named homes both.
        prefix = ".%s/" % provider
        if provider == "claude" and profile_root:
            cfg_root, named = profile_root, True
        elif provider == "codex" and codex_profile_root:
            cfg_root, named = codex_profile_root, True
        else:
            cfg_root, named = os.path.realpath(
                os.path.join(home, ".%s" % provider)), False
        names = [
            (rel[len(prefix):], os.path.join(cfg_root, rel[len(prefix):]))
            for rel, _kind in _AGENT_DENY_WRITE if rel.startswith(prefix)
        ]
        if provider == "claude":
            names.append((
                ".claude.json",
                os.path.join(cfg_root if named else home, ".claude.json"),
            ))
        try:
            writable_roots = list(_write_allow_paths(
                workdir, read_only, extra_write_paths=writes,
            )) + [entry_path for _kind, entry_path in post_write]
        except FencePathRejected as exc:
            rejected = exc.detail
        else:
            rejected = _protected_config_refusal(
                "%s %s home" % ("named" if named else "default", provider),
                names, named=named, writable_roots=writable_roots,
                home_root=cfg_root,
            )

    if not rejected and vendor_root:
        # The leaves above are checked below the vendor root; the root itself
        # may be reached through a symlink chain (see
        # _vendor_root_link_refusal). Needs every write entry of this spawn,
        # config dir and the vendor's own state leaves included, hence after
        # the provider branches. prepare_host_fence runs it again once the
        # repository roots are added.
        try:
            rejected = pi_cursor_root_link_refusal(
                provider, workdir, read_only, writes, post_write, vendor_root,
            )
        except FencePathRejected as exc:
            rejected = exc.detail

    vendor_state_writable = False
    claude_root = os.path.join(home, ".claude")
    pi_root = os.path.join(home, ".pi")
    cursor_root = os.path.join(home, ".cursor")
    codex_root = os.path.join(home, ".codex")
    vendor_roots = [claude_root, pi_root, cursor_root, codex_root]
    if grok_root:
        vendor_roots.append(grok_root)
    if provider == "claude" and profile_root:
        vendor_roots.append(profile_root)
    if provider == "agy":
        vendor_roots.append(os.path.join(home, _AGY_STATE_ROOT))
    if codex_profile_root:
        vendor_roots.append(codex_profile_root)
    post_paths = [(_write_entry_bound(*entry) or entry)[1] for entry in post_write]
    for path in list(writes) + post_paths:
        if any(_path_inside(path, root) for root in vendor_roots):
            vendor_state_writable = True
            break

    return {
        "extra_read_paths": tuple(reads),
        "extra_write_paths": tuple(writes),
        "extra_deny_write": tuple(deny_write),
        "extra_post_write": tuple(post_write),
        "extra_deny_read": tuple(deny_read),
        "extra_post_read": tuple(post_read),
        "extra_deny_link": tuple(deny_link),
        "extra_deny_write_final": tuple(deny_write_final),
        "extra_deny_read_final": tuple(deny_read_final),
        "tmpdir": tmpdir,
        "vendor_state_writable": vendor_state_writable,
        "rejected": rejected,
        "binary": vendor_bin,
        "vendor_root": vendor_root,
        "claude_config_root": claude_config_root,
    }


def agent_fence_paths(provider, workdir=None, read_only=False, config_dir=None,
                      launcher_roots=()):
    """``(extra_read, extra_write)`` for claude / pi / cursor / codex / grok.

    Extra read paths and extra write-allow roots are realpath'd; missing
    paths among those are dropped. ``extra_post_write`` entries for the
    codex lane are lexical (``abspath``, never realpath): a realpath'd T
    clause for a symlink at a post-allowed name would put the TARGET in T
    and last-match would allow it. Home is ``protected_home()``. Bin dirs
    are ``dirname(realpath(binary))``.
    ``config_dir`` is this run's vendor config root (cursor
    ``CURSOR_CONFIG_DIR``, a claude profile ``CLAUDE_CONFIG_DIR``, or a codex
    profile ``CODEX_HOME``). ``launcher_roots`` is the explicit allowlist
    for launcher-owned home descendants (jobs root / temp root).
    """
    collected = _collect_agent_fence(
        provider, workdir=workdir, read_only=read_only, create=False,
        config_dir=config_dir, launcher_roots=launcher_roots,
    )
    writes = list(collected["extra_write_paths"])
    for kind, path in collected["extra_post_write"]:
        if kind == GROK_TMP_KIND:
            continue    # a name pattern under the root, not a write root
        if path not in writes:
            writes.append(path)
    return collected["extra_read_paths"], tuple(writes)


def codex_consult_account_fence(config_dir, workdir):
    """Extra fence rows for a codex consult on a codex account home.

    The consult lane builds its own policy over the shared defaults, which
    admit the default ``~/.codex`` for read and write; neither a named
    account nor the default one may be writable whole. This reuses the
    codex AGENT fence's rows: the selected home is write-denied whole with
    only the vendor state leaves re-allowed (config, hooks, AGENTS files,
    the agent/skill/plugin/rule trees, custom ``*.config.toml`` layers and
    ``.tmp/`` stay denied), the protected configuration names are also
    FINAL denies, and every other registered account home is read-denied.

    ``config_dir`` is a NAMED account's ``CODEX_HOME``; the default
    ``~/.codex`` is also write- and read-denied then. ``config_dir=None``
    selects the DEFAULT ``~/.codex`` itself (see
    ``codex_consult_default_fence``). Raises ``FencePathRejected`` on any
    refusal.
    """
    collected = _collect_agent_fence(
        "codex", workdir=workdir, read_only=True, create=False,
        config_dir=config_dir,
    )
    if collected.get("rejected"):
        raise FencePathRejected(collected["rejected"])
    return {
        key: collected[key]
        for key in ("extra_deny_write", "extra_post_write",
                    "extra_deny_read", "extra_deny_link",
                    "extra_deny_write_final", "extra_deny_read_final")
    }


def codex_consult_default_fence(workdir):
    """Extra fence rows for a codex consult on the DEFAULT account home.

    The consult lane builds its own policy over the shared defaults, whose
    write allowance covers the whole ``~/.codex``; denying only the
    protected names left every other path there writable (a script
    ``hooks.json`` invokes, a ``*.config.toml`` layer, the staged plugin
    tree under ``.tmp/``), so one consult could poison later runs. This is
    the named-account structure applied to the default home
    (``codex_consult_account_fence`` with no ``config_dir``): ``~/.codex``
    is write-denied whole, only the agent fence's validated state set is
    re-allowed (state directories, ``.sqlite`` / ``.db`` stores with their
    ``-wal`` / ``-shm`` siblings, and the named state files including
    ``auth.json``), and the protected configuration names (config, hooks,
    both AGENTS files, the agents/skills/plugins/rules trees) get FINAL
    (``F*``) denies on each name as written and as it resolves, so no
    state allowance can reopen them and an unlink or replacement of the
    name is refused too. Raises ``FencePathRejected`` when a protected name
    links to the vendor home or a run-writable root, or a state leaf is a
    symlink; a link into an unwritable shared instructions tree stays
    allowed. Reads are unchanged.
    """
    return codex_consult_account_fence(None, workdir)


def agent_fence_policy(provider, workdir=None, read_only=False, config_dir=None,
                       launcher_roots=()):
    """Wrap/verify kwargs plus metadata, or a LOUD refusal dict.

    Never raises. ``FencePathRejected`` maps to
    ``{"ok": False, "reason": "fence_path_rejected", "detail": exc.detail}``.
    ``config_dir`` is the vendor config root (see ``_collect_agent_fence``).
    ``launcher_roots`` is the explicit allowlist for launcher-owned home
    descendants.
    """
    wrap_kw = None
    collected = None
    try:
        collected = _collect_agent_fence(
            provider, workdir=workdir, read_only=read_only, create=False,
            config_dir=config_dir, launcher_roots=launcher_roots,
        )
        if collected.get("rejected"):
            return {
                "ok": False,
                "reason": "fence_path_rejected",
                "detail": redact_secrets(str(collected["rejected"]))[:300],
            }
        wrap_kw = dict(
            read_wall=True,
            extra_read_paths=collected["extra_read_paths"],
            extra_write_paths=collected["extra_write_paths"],
            extra_deny_write=collected["extra_deny_write"],
            extra_post_write=collected["extra_post_write"],
            extra_deny_read=collected["extra_deny_read"],
            extra_post_read=collected["extra_post_read"],
            extra_deny_link=collected["extra_deny_link"],
            extra_deny_write_final=collected["extra_deny_write_final"],
            extra_deny_read_final=collected["extra_deny_read_final"],
        )
        # Last resolver: a swap after admission (including during
        # ``_caller_config_dir_rejection``) must refuse here, before wrap.
        _write_allow_paths(
            workdir, read_only,
            extra_write_paths=wrap_kw["extra_write_paths"],
        )
        _read_reallow_entries(
            workdir, extra_read_paths=wrap_kw["extra_read_paths"],
        )
    except FencePathRejected as exc:
        return {
            "ok": False,
            "reason": "fence_path_rejected",
            "detail": exc.detail,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": "fence_path_rejected",
            "detail": "agent fence path collection failed: %s" % redact_secrets(str(exc))[:200],
        }
    return {
        "wrap_kw": wrap_kw,
        "tmpdir": collected["tmpdir"],
        "vendor_state_writable": collected["vendor_state_writable"],
        "binary": collected["binary"],
        "vendor_root": collected.get("vendor_root"),
        "claude_config_root": collected.get("claude_config_root"),
    }


def agent_child_env(base=None, tmpdir=None, keep_config_dir=False) -> dict:
    """``sanitized_child_env`` plus ``CLAUDE_CONFIG_DIR`` stripped and TMPDIR pin.

    ``keep_config_dir=True`` leaves a profile-provided ``CLAUDE_CONFIG_DIR``
    in place. The default profile still strips it. The operator notify
    sink is never inherited: ``OMNI_LLM_NOTIFY_CMD`` is dropped and
    ``OMNI_LLM_NOTIFY_QUIET=1`` is set.
    """
    env = sanitized_child_env(base)
    env.pop("OMNI_LLM_NOTIFY_CMD", None)
    env["OMNI_LLM_NOTIFY_QUIET"] = "1"
    if not keep_config_dir:
        env.pop("CLAUDE_CONFIG_DIR", None)
    if tmpdir:
        env["TMPDIR"] = tmpdir
    return env


_DATA_VOLUME = "/System/Volumes/Data"


def _plain_spelling(path):
    """*path* without the data-volume firmlink prefix, or None.

    Seatbelt judges the kernel-resolved path of an access, and a rule spelled
    with ``/System/Volumes/Data`` never matches it (measured in
    test_same_location), so a rule must carry the plain spelling. APFS ignores
    case, so the prefix is recognised in any case; the identity check keeps a
    look-alike path from being rewritten.
    """
    if path[:len(_DATA_VOLUME) + 1].casefold() != (_DATA_VOLUME + os.sep).casefold():
        return None
    plain = path[len(_DATA_VOLUME):]
    try:
        if os.path.samestat(os.stat(plain), os.stat(path)):
            return os.path.realpath(plain)
    except OSError:
        pass
    return None


# errno values that confirm the named path does not exist.
_ABSENT = (errno.ENOENT, errno.ENOTDIR)


def ssh_agent_socket_denies(*environs, cwd=None) -> tuple:
    """Connect-deny entries for the operator's ssh-agent socket.

    ``SSH_AUTH_SOCK`` is read from every mapping in *environs*; a value that
    names an existing socket yields a literal entry for its realpath (the
    kernel matches the resolved path, so a symlink spelling is covered), the
    spelling as written when that differs, and the plain spelling of a
    data-volume firmlink value (in any case). A relative value is judged
    against the server's cwd and against *cwd* (where the child will run).
    Unset, empty, confirmed-absent and non-socket values add nothing. A value
    that cannot be inspected (permission, loop, I/O error) raises
    ``FencePathRejected``: the child could otherwise make the path
    inspectable and connect.
    """
    out = []
    for environ in environs:
        value = (environ or {}).get("SSH_AUTH_SOCK")
        if not isinstance(value, str) or not value or "\0" in value:
            continue
        bases = [None] if os.path.isabs(value) else [None, cwd]
        for base in bases:
            if base is not None and not os.path.isabs(value):
                candidate = os.path.join(base, value)
            else:
                candidate = value
            try:
                mode = os.stat(candidate).st_mode
            except OSError as exc:
                if exc.errno in _ABSENT:
                    continue
                raise FencePathRejected(
                    "SSH_AUTH_SOCK cannot be inspected (%s); the ssh-agent "
                    "socket would not be denied" % errno.errorcode.get(
                        exc.errno, exc.errno)) from None
            if not stat.S_ISSOCK(mode):
                continue
            real = os.path.realpath(candidate)
            spellings = [real, os.path.abspath(candidate)]
            for spelled in tuple(spellings):
                plain = _plain_spelling(spelled)
                if plain:
                    spellings.append(plain)
            for path in spellings:
                if ("literal", path) not in out:
                    out.append(("literal", path))
    return tuple(out)


def connect_deny_write_entries(extra_deny_connect) -> tuple:
    """Write-deny entries that pin each connect-denied socket in place.

    The connect deny names a pathname, so a child with a writable root over
    the socket could rename the socket, or any directory above it, and
    connect through the new name. A literal ``file-write*`` deny on the
    socket and on every ancestor directory stops the rename (the parents'
    own parents are not writable, or the ancestor deny reaches them too);
    it costs the child nothing else, since creating or writing a file inside
    a directory is judged on the file's path.
    """
    out = []
    for _kind, path in extra_deny_connect or ():
        current = path
        while True:
            if ("literal", current) not in out:
                out.append(("literal", current))
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return tuple(out)


def wrap_kwargs_from_policy(policy) -> dict:
    """kwargs shared by ``verify_fence`` and ``wrap_argv`` (A7: one policy)."""
    return dict(policy["wrap_kw"])
