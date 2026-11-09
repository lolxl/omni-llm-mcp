"""Find a vendor CLI on PATH, never a harness launcher shim.

A machine that has switched to the team rule release puts launcher shims
(executables that carry the line ``# mofuu-harness-shim``) ahead of the vendor
CLIs on PATH. A shim routes interactive runs through a launcher that lives
under the protected home, which the sandbox refuses to exec; headless runs
need the vendor binary itself, so the search skips every shim.
"""

import os
import shutil

SHIM_MARKER = b"# mofuu-harness-shim"
_SCAN_LIMIT = 65536


def is_harness_shim(path):
    """True when the file at ``path`` carries the launcher shim marker."""
    try:
        with open(path, "rb") as handle:
            return SHIM_MARKER in handle.read(_SCAN_LIMIT)
    except OSError:
        return False


def which(name, path=None):
    """``shutil.which`` for a bare command name, skipping harness shims.

    Only absolute PATH entries are searched, so a file in the working
    directory is never a candidate. The lookup itself is ``shutil.which``,
    repeated with each shim's directory removed from the search path.
    """
    if path is None:
        # The plain lookup first, exactly as before; only a shim answer needs
        # the narrowed search below.
        found = shutil.which(name)
        if not found or not is_harness_shim(found):
            return found
        search = os.environ.get("PATH", "")
    else:
        search = path
    remaining = [d for d in search.split(os.pathsep) if d and os.path.isabs(d)]
    while remaining:
        found = shutil.which(name, path=os.pathsep.join(remaining))
        if not found:
            return None
        if not is_harness_shim(found):
            return found
        shim_dir = os.path.normpath(os.path.dirname(found))
        kept = [d for d in remaining if os.path.normpath(d) != shim_dir]
        if len(kept) == len(remaining):
            return None
        remaining = kept
    return None
