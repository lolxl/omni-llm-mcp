"""Keep the offline suite away from the real account home.

``kernel_sandbox.protected_home()`` resolves the account home from the passwd
database, so a test that goes through the fence (a fenced spawn, an agent
command, a profile build) admits ``~/.claude``, ``~/.claude.json``,
``~/.codex`` and friends by their real names. The real ``~/.claude.json`` is
rewritten atomically by every running CLI session; admission records its
inode and ``_confirm_admitted`` refuses the profile build when the inode moved
in between (``fence_path_rejected: extra_read_paths changed between admission
and profile build``). An unrelated session therefore failed unrelated tests.

``install()`` (called once by ``tests/run_offline_tests.py``, before test
discovery) does three things for the whole run:

* points ``protected_home()`` and ``$HOME`` at a synthetic home that holds
  same-shaped stand-ins (``.claude``, ``.claude.json``, ``.codex``, ...), so
  fence code finds the files it expects without ever naming a real one;
* wraps ``admit_path`` and records every admission that resolves under the
  REAL home;
* hooks ``open`` and records every open of a credential / configuration entry
  under the REAL home (an audit hook, so it also sees reads that never go
  through the fence).

The same three apply to Python child processes started by the tests (the job
runner is one): ``install()`` exports ``PYTHONPATH`` so that
``tests/offline_home_site/sitecustomize.py`` runs ``install_in_child()`` in
them, and their hits come back through an append-only log file.

Coverage limits: the recorder lives in Python processes. A fenced child that
is not Python (a shell, a vendor binary, sandbox-exec itself) and the
``python -I -S`` probe interpreters run without it, so what they open is not
seen; they still get the synthetic ``$HOME`` and scrubbed account variables.
A fenced Python child cannot report either: the hit log lives under the
synthetic home, which is outside its write roots, so its append fails (a
message goes to its stderr, nothing reaches the runner). Known residual.
A hard link to a credential, created outside the home, is invisible to a
path-based detector: it would need an inventory of the real home's
credential inodes, which is the very read this guard forbids, and
~/.claude.json gets a new inode on every rewrite anyway. Known limit.

What a clean run guarantees: zero DETECTED accesses within the coverage
described above. It does not prove that no real-home read happened.

``violations()`` is empty for a clean run; the runner turns a non-empty list
into a failing exit status and names the tests that touched the real home.
"""
import json
import os
import pwd
import re
import shutil
import stat
import sys
import tempfile
from unittest import mock

from providers import kernel_sandbox, profile_check, profiles

# The product function, kept so a test can still exercise it after install()
# replaced the module attribute (see original_protected_home()).
_ORIGINAL_PROTECTED_HOME = kernel_sandbox.protected_home

# Bound at import, before any test patches the os module: the audit hook runs
# inside calls that tests replace (os.stat, os.path.realpath, ...) and must
# neither trip those replacements nor be fooled by them.
_stat = os.stat
_realpath = os.path.realpath
_getcwd = os.getcwd
_lstat = os.lstat
_readlink = os.readlink

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DIR = os.path.join(ROOT, "tests", "offline_home_site")

ENV_HOME = "OMNI_OFFLINE_HOME"        # synthetic home, set for children
ENV_REAL = "OMNI_OFFLINE_REAL_HOME"   # real home, set for children
ENV_LOG = "OMNI_OFFLINE_HITLOG"       # append-only hit log shared with children
ENV_CURRENT = "OMNI_OFFLINE_CURRENT"  # id of the running test

# Entries of the real home whose contents are credentials or live CLI state.
# An ``open`` or listing of one of these is a violation even when it bypasses
# the fence. A top-level name matches by PREFIX (".codex-x", ".claude-work",
# ".grok-host" are account homes too) and case-insensitively.
# Vendor install trees (the binaries themselves) are not account state.
SENSITIVE_TOP_PREFIXES = (
    ".claude", ".codex", ".grok", ".cursor", ".pi", ".gemini",
)
SENSITIVE_TOPS = (
    ".ssh", ".agents", ".config", ".gnupg", ".aws", ".netrc",
)
SENSITIVE_PREFIXES = (
    os.path.join("Library", "Keychains"),
    os.path.join(".local", "state", "omni-llm"),
)
VENDOR_INSTALL_PREFIXES = (
    os.path.join(".codex", "packages"),
    os.path.join(".grok", "downloads"),
    os.path.join(".agents", "bin"),
    os.path.join(".grok", "bin"),
)

# Variables that can point a provider, a child or the state store at real
# account state. Cleared in this process by install(); a Python child drops
# one only when its value resolves into the real home (a test may pass its own
# temp path on purpose). A hand-written floor plus whatever the product itself
# registers: the account-home prefixes of kernel_sandbox and profile_check,
# the sibling prefixes and ring prefix of profiles, and the names profiles
# drops from a child environment. API keys are not paths and stay.
_FLOOR_NAMES = (
    "CODEX_HOME", "GROK_HOME", "CLAUDE_CONFIG_DIR", "AGY_CONFIG_DIR",
    "CURSOR_CONFIG_DIR", "OMNI_LLM_CURSOR_CONFIG_DIR", "OMNI_LLM_STATE_DIR",
    "OMNI_LLM_CODEX_SESSIONS_DIR", "OMNI_LLM_SOURCE_ROOTS",
)
_FLOOR_PREFIXES = (
    "CLAUDE_CONFIG_DIR_", "OMNI_LLM_CURSOR_CONFIG_DIR_", "OMNI_LLM_PROFILES_",
    "XDG_",
)


def _pathlike(name):
    return not name.rstrip("_").endswith(("_KEY", "_BASE"))


def product_scrub_registry():
    """(names, prefixes) the product itself treats as account-home variables."""
    names = {n for n in profiles._DROP_EXACT if _pathlike(n)}
    prefixes = {profiles._RING_PREFIX}
    for group in profiles._ROAD_SIBLING_PREFIXES.values():
        prefixes.update(group)
    for group in profile_check._ENV_PREFIXES.values():
        prefixes.update(group)
    prefixes.update(prefix for _provider, prefix in
                    kernel_sandbox._ACCOUNT_HOME_PREFIXES)
    alternation = re.match(r"\^\((.*?)\)_", profiles._DROP_SUFFIX_RE.pattern)
    for name in (alternation.group(1).split("|") if alternation else ()):
        if _pathlike(name):
            prefixes.add(name + "_")
    return names, {p for p in prefixes if _pathlike(p)}


_product_names, _product_prefixes = product_scrub_registry()
SCRUB_ENV_NAMES = tuple(sorted(set(_FLOOR_NAMES) | _product_names))
SCRUB_ENV_PREFIXES = tuple(sorted(set(_FLOOR_PREFIXES) | _product_prefixes))

_state = {
    "home": None,        # synthetic home (realpath)
    "real": None,        # real home (realpath), fixed before any patching
    "ident": None,       # (st_dev, st_ino) of the real home
    "current": None,     # id of the running test
    "hits": [],          # (test id, how, relative path)
    "patches": [],
    "hook": False,       # audit hook is process-wide and cannot be removed
    "armed": False,
    "log": None,
    "owner": False,      # True in the process that called install()
    "made": [],          # every synthetic home created by this process
}


def real_home():
    """The passwd-database home, resolved without ``protected_home()``."""
    return os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)


def active():
    return _state["armed"]


def home():
    return _state["home"]


def original_protected_home():
    """The product's own ``protected_home`` (passwd lookup), unpatched."""
    return _ORIGINAL_PROTECTED_HOME()


def _under(path, root):
    return path == root or path.startswith(root + os.sep)


def _identity(path):
    try:
        st = _stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


def _resolve(path):
    """Symlink-resolved form of an absolute *path*, from bound originals.

    Does not call os.path.realpath: a test may patch that, or the os calls it
    makes, and the audit hook runs inside such calls.
    """
    comps = [c for c in path.split("/") if c]
    out = []
    links = 0
    while comps:
        c = comps.pop(0)
        if c == ".":
            continue
        if c == "..":
            if out:
                out.pop()
            continue
        cand = "/" + "/".join(out + [c])
        try:
            is_link = stat.S_ISLNK(_lstat(cand).st_mode)
        except (OSError, ValueError):
            out.append(c)
            continue
        if not is_link:
            out.append(c)
            continue
        links += 1
        if links > 40:
            return None
        try:
            target = _readlink(cand)
        except OSError:
            return None
        if target.startswith("/"):
            out = []
        comps = [t for t in target.split("/") if t] + comps
    return "/" + "/".join(out)


def _absolute(raw):
    """Absolute spelling of *raw* WITHOUT normalizing it.

    ``normpath`` would fold ``link/../x`` into ``x`` and the link would never
    be looked at; the kernel walks the components left to right, so do we.
    """
    given = os.fsdecode(os.fspath(raw))
    if not given.startswith("/"):
        given = _getcwd() + "/" + given
    return given


def _rel_by_identity(raw):
    """Path of *raw* below the real home, or None when it is elsewhere.

    Compared by identity, not spelling: some ancestor of the path is the real
    home's directory object. A case variant on a case-insensitive volume and a
    ``/System/Volumes/Data`` spelling both reach the same object and are
    caught; a look-alike name is not. Nothing is cached: a directory symlink
    can be retargeted at the home (or away from it) at any time.
    """
    ident = _state["ident"]
    if ident is None:
        return None
    try:
        given = _absolute(raw)
    except (TypeError, ValueError, OSError):
        return None
    cur = given
    while True:
        if _identity(cur) == ident:
            return given[len(cur):].lstrip(os.sep) or "."
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _classify(raw):
    """Classify the fully RESOLVED path of *raw*; its place below the real home.

    The spelling is expanded left to right (every symlink, then ``..`` against
    the expanded directory) and the identity walk runs over that result. The
    literal spelling decides nothing on its own: a link that is the home on the
    way in may lead somewhere else (``tohome/../x`` is outside the home) and a
    link outside the home may lead to a credential (``link/../x``, or a file
    inside the home that is itself a link to ``.ssh/id_rsa``).

    Only when resolution fails (a loop, an unreadable link) is the literal
    spelling used, and then conservatively: a literal ancestor that is the
    home counts as under the home.
    """
    try:
        given = _absolute(raw)
    except (TypeError, ValueError, OSError):
        return None
    resolved = _resolve(given)
    if resolved is None:
        return _rel_by_identity(given)
    return _rel_by_identity(resolved)


def _sensitive(rel):
    low = rel.casefold()
    if any(low == pre or low.startswith(pre + os.sep)
           for pre in (v.casefold() for v in VENDOR_INSTALL_PREFIXES)):
        return False
    top = low.split(os.sep, 1)[0]
    if top in SENSITIVE_TOPS or top.startswith(SENSITIVE_TOP_PREFIXES):
        return True
    return any(low == pre or low.startswith(pre + os.sep)
               for pre in (v.casefold() for v in SENSITIVE_PREFIXES))


def _add_hit(how, rel):
    row = (_state["current"] or os.environ.get(ENV_CURRENT), how, rel)
    if _state["owner"]:
        _state["hits"].append(row)
    elif _state["log"]:
        # Child process: one short append per hit (atomic on O_APPEND).
        try:
            with open(_state["log"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(list(row) + [os.getpid()]) + "\n")
        except OSError as exc:
            # Cannot be reported through the log: say so where a person can see
            # it. A fenced child cannot write there at all (documented).
            print("offline_home: cannot append to the hit log (%s): %s -> ~/%s"
                  % (type(exc).__name__, how, rel), file=sys.stderr)


def record(how, raw, real=None):
    """Record *raw* when it names something under the real home.

    Classifies the fully resolved path (see ``_classify``).
    """
    rel = _classify(raw)
    if rel is not None:
        _add_hit(how, rel)
    return rel


def set_current(test_id):
    _state["current"] = test_id
    if test_id is None:
        os.environ.pop(ENV_CURRENT, None)
    else:
        os.environ[ENV_CURRENT] = test_id


def violations():
    """Hits from this process and from the children that logged to the file."""
    rows = list(_state["hits"])
    log = _state["log"]
    if _state["owner"] and log and os.path.exists(log):
        with open(log, encoding="utf-8") as fh:
            for line in fh:
                try:
                    t, how, rel, _pid = json.loads(line)
                except ValueError:
                    continue
                rows.append((t, "child:" + how, rel))
    return rows


def _audit(event, args):
    # Directory listings count too: the agy layout guard walks ~/.gemini
    # with listdir / scandir and never opens a file there.
    if not _state["armed"] or event not in ("open", "os.listdir", "os.scandir"):
        return
    target = args[0] if args else None
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    if isinstance(target, bytes):
        target = os.fsdecode(target)
    rel = _classify(target)
    if rel is not None and _sensitive(rel):
        _add_hit("list" if event != "open" else "open", rel)


def _make_home(real):
    """A fresh synthetic home that no fence profile treats as a temp area.

    Fence code gives temp areas special treatment (write-allowed, approved
    TMPDIR roots), so a home created there would turn the "home is denied"
    probes into non-tests. Fence tests also build "/System/Volumes/Data/Users"
    style spellings from the home, so it should live under /Users. Try the
    candidates in order; refuse to run when none qualifies.
    """
    temp_like = tuple(kernel_sandbox._write_allow_paths(None, True)) \
        + tuple(kernel_sandbox._approved_temp_roots())
    tried = []
    for base in ("/Users/Shared", "/private/var/tmp"):
        if not os.path.isdir(base) or not os.access(base, os.W_OK):
            continue
        base = os.path.realpath(base)
        if _under(base, real) or any(_under(base, t) for t in temp_like):
            tried.append(base)
            continue
        made = os.path.realpath(
            tempfile.mkdtemp(prefix="omni_offline_home_", dir=base))
        _state["made"].append(made)
        return made
    raise RuntimeError(
        "no location for the synthetic home outside the real home and the "
        "fence's temp areas (rejected: %s)" % (tried,))


def _populate(root):
    """The shape of an account home the fence tests expect, with no content."""
    dirs = [".claude", ".codex", ".cursor", ".pi", ".grok", ".local", ".ssh",
            ".agents", os.path.join(".gemini", "config"),
            os.path.join(".gemini", "antigravity-cli"),
            os.path.join(".codex", "shell_snapshots"),
            os.path.join(".codex", "sessions"),
            os.path.join("Library", "Keychains")]
    # Live deny probes write to every protected name and expect EPERM; a
    # missing parent would answer ENOENT instead.
    dirs += [os.path.dirname(rel) for rel, _kind in
             getattr(kernel_sandbox, "_AGENT_DENY_WRITE", ()) if os.path.dirname(rel)]
    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    with open(os.path.join(root, ".claude.json"), "w") as fh:
        fh.write("{}\n")
    with open(os.path.join(root, "Library", "Keychains", "login.keychain-db"),
              "w") as fh:
        fh.write("")


def _patch_kernel_sandbox(fake, real):
    real_admit = kernel_sandbox.admit_path

    def admit_path(raw, kind, *, label):
        record("admit:" + str(label), raw, real)
        return real_admit(raw, kind, label=label)

    patches = [
        mock.patch.object(kernel_sandbox, "protected_home", lambda: fake),
        mock.patch.object(kernel_sandbox, "admit_path", admit_path),
    ]
    for p in patches:
        p.start()
    return patches


def make_home(real=None):
    """A populated synthetic home for one test (the caller removes it)."""
    root = _make_home(real or real_home())
    _populate(root)
    return root


def _scrub_env(environ, fake, only_real=False):
    """Clear or redirect the variables that can name real account state.

    In the parent every listed variable is cleared and XDG_* is pointed into
    the synthetic home. With *only_real* (a Python child) a variable is
    dropped only when its value resolves into the real home.
    """
    for key in list(environ):
        if key not in SCRUB_ENV_NAMES and not key.startswith(SCRUB_ENV_PREFIXES):
            continue
        if only_real:
            val = environ[key]
            hit = _classify(val) is not None or any(
                _classify(part) is not None
                for part in val.split(os.pathsep) if part)
            if not hit:
                continue
        environ.pop(key, None)
    if not only_real:
        environ["XDG_CONFIG_HOME"] = os.path.join(fake, ".config")
        environ["XDG_DATA_HOME"] = os.path.join(fake, ".local", "share")
        environ["XDG_STATE_HOME"] = os.path.join(fake, ".local", "state")
        environ["XDG_CACHE_HOME"] = os.path.join(fake, ".cache")


def install(real=None):
    """Arm the synthetic home and the real-home recorder for this process.

    *real* names the home to keep away from; it defaults to the passwd home
    and is overridden only by the guard's own tests.
    """
    if _state["armed"]:
        return _state["home"]
    real = os.path.realpath(real) if real else real_home()
    fake = _make_home(real)
    _populate(fake)
    log = os.path.join(fake, ".offline-hits.jsonl")
    _state.update(real=real, ident=_identity(real), home=fake,
                  hits=[], current=None, armed=True, owner=True, log=log)
    patches = _patch_kernel_sandbox(fake, real)
    env = mock.patch.dict(os.environ, {
        "HOME": fake,
        ENV_HOME: fake,
        ENV_REAL: real,
        ENV_LOG: log,
        "PYTHONPATH": os.pathsep.join(
            [SITE_DIR, ROOT] + [p for p in os.environ.get(
                "PYTHONPATH", "").split(os.pathsep) if p]),
    })
    env.start()
    patches.append(env)
    _scrub_env(os.environ, fake)
    _state["patches"] = patches
    if not _state["hook"]:
        sys.addaudithook(_audit)
        _state["hook"] = True
    return fake


def install_in_child():
    """Same recorder, synthetic home and scrubbed environment in a child."""
    fake, real, log = (os.environ.get(ENV_HOME), os.environ.get(ENV_REAL),
                       os.environ.get(ENV_LOG))
    if _state["armed"]:
        return
    if not (fake and real and log):
        raise RuntimeError("incomplete OMNI_OFFLINE_* environment")
    _state.update(real=real, ident=_identity(real), home=fake,
                  armed=True, owner=False, log=log)
    if _state["ident"] is None:
        raise RuntimeError("real home cannot be identified")
    _patch_kernel_sandbox(fake, real)
    sys.addaudithook(_audit)
    _state["hook"] = True
    _scrub_env(os.environ, fake, only_real=True)
    if kernel_sandbox.protected_home() != fake:
        raise RuntimeError("protected_home was not redirected")


def report_arm_failure(reason, log=None, test_id=None):
    """Tell the runner that a child could not arm the guard."""
    log = log or os.environ.get(ENV_LOG)
    if not log:
        return
    row = [test_id or os.environ.get(ENV_CURRENT), "arm_failed",
           str(reason)[:300], os.getpid()]
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        print("offline_home: child could not arm and cannot report it (%s)"
              % type(exc).__name__, file=sys.stderr)


def uninstall():
    """Disarm and remove every synthetic home; return what went wrong.

    The recorded hits stay. A non-empty result means a synthetic home could
    not be removed, or a per-test home was left behind by its test.
    """
    problems = []
    for p in reversed(_state["patches"]):
        try:
            p.stop()
        except RuntimeError as exc:
            problems.append("could not undo a patch: %s" % exc)
    _state["patches"] = []
    _state["armed"] = False
    main_home = _state["home"]
    for made in list(_state["made"]):
        if made != main_home and os.path.lexists(made):
            problems.append("test home not removed by its test: %s" % made)

        def _onerror(func, path, exc_info):
            problems.append("cannot remove %s: %s" % (path, exc_info[0].__name__))

        if not os.path.lexists(made):
            continue
        shutil.rmtree(made, onerror=_onerror)
        if os.path.lexists(made):
            problems.append("synthetic home left behind: %s" % made)
    _state["made"] = []
    _state["home"] = None
    return problems


def format_violations(hits=None):
    hits = violations() if hits is None else hits
    lines = []
    for test_id, how, rel in hits:
        if how.endswith("arm_failed"):
            lines.append("  %s: a child could not arm the guard: %s"
                         % (test_id, rel))
        else:
            lines.append("  %s: %s -> ~/%s" % (test_id, how, rel))
    return "\n".join(sorted(set(lines)))
