# no-opencc
"""Allowlisted server-side file reads (llm_big_context source_path; grok `files=`).

Threat model: the MCP server reads the file BEFORE any provider runs. Grok/codex
sandboxes do NOT protect this read. Path mode requires OMNI_LLM_SOURCE_ROOTS;
the path argument only selects within that grant — it is not a capability grant.
Two readers share one fence: read_source_file (UTF-8 text, llm_big_context) and
open_binary_under_roots (raw bytes, grok vision `files=`).

v1 rules (Owner + Codex):
- text XOR source_path (mutual exclusive when path non-empty)
- without source_path: preserve legacy text validation
- absolute path only; never strip path (whitespace may be legal)
- open via openat walk from allowlist root with O_NOFOLLOW (TOCTOU-resistant)
- UTF-8 strict
- size: file + assembled prompt hard caps; read max+1 to detect oversize
- error_code stable; detail avoids full path by default
"""
import errno
import os
import stat

from providers.config import (
    SOURCE_PATH_MAX_BYTES,
    SOURCE_PROMPT_MAX_BYTES,
    ENV_SOURCE_ROOTS,
)


def _err(error_code, detail):
    return {
        "ok": False,
        "reason": "runtime_error",
        "error_code": error_code,
        "detail": detail,
    }


def parse_source_roots(env_value=None):
    """Return list of absolute root dirs from env (colon or comma separated)."""
    if env_value is None:
        env_value = os.environ.get(ENV_SOURCE_ROOTS, "")
    raw = env_value or ""
    roots = []
    for part in raw.replace(",", ":").split(":"):
        p = part.strip()
        if not p:
            continue
        roots.append(p)
    return roots


def _under_root(real_path, real_root):
    if real_path == real_root:
        return True
    prefix = real_root.rstrip(os.sep) + os.sep
    return real_path.startswith(prefix)


def _flags_base():
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _nofollow():
    return getattr(os, "O_NOFOLLOW", 0)


def _odirectory():
    return getattr(os, "O_DIRECTORY", 0)


def _resolve_root_and_parts(abs_path, roots):
    """Map abs_path to (real_root, relative_parts) or (None, error).

    Membership uses realpath. Opening later uses openat from real_root with
    O_NOFOLLOW on every relative component so parent swaps cannot escape.
    """
    if not os.path.isabs(abs_path):
        return None, None, _err(
            "source_path_not_absolute", "source_path must be an absolute path")
    if not roots:
        return None, None, _err(
            "source_path_disabled",
            "source_path requires %s (absolute allowlist roots)" % ENV_SOURCE_ROOTS,
        )

    real_roots = []
    for r in roots:
        if not os.path.isabs(r):
            continue
        try:
            real_roots.append(os.path.realpath(r))
        except OSError:
            continue
    if not real_roots:
        return None, None, _err(
            "source_path_disabled",
            "source_path requires %s with at least one absolute root" % ENV_SOURCE_ROOTS,
        )

    try:
        real_path = os.path.realpath(abs_path)
    except OSError:
        return None, None, _err(
            "source_path_not_allowed", "source_path not under allowlist roots")

    matched = None
    for rr in real_roots:
        if _under_root(real_path, rr):
            matched = rr
            break
    if matched is None:
        return None, None, _err(
            "source_path_not_allowed", "source_path not under allowlist roots")

    if real_path == matched:
        # Path is the root directory itself — not a regular file.
        return matched, [], None

    rel = os.path.relpath(real_path, matched)
    if rel.startswith("..") or os.path.isabs(rel):
        return None, None, _err(
            "source_path_not_allowed", "source_path not under allowlist roots")
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    if any(p == ".." for p in parts):
        return None, None, _err(
            "source_path_not_allowed", "source_path not under allowlist roots")
    return matched, parts, None


def _openat_walk(real_root, parts):
    """Open file at real_root/parts via openat + O_NOFOLLOW each step.

    Returns (fd, None) or (None, error_envelope). Caller owns fd.
    """
    if not parts:
        return None, _err("source_not_regular", "source_path is not a regular file")

    base = _flags_base()
    nofollow = _nofollow()
    o_dir = _odirectory()

    dir_fd = None
    try:
        # Capability root: open the allowlisted directory (absolute).
        root_flags = base
        if o_dir:
            root_flags |= o_dir
        dir_fd = os.open(real_root, root_flags)

        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                flags = base | nofollow
                try:
                    fd = os.open(part, flags, dir_fd=dir_fd)
                except OSError as e:
                    return None, _map_open_oserror(e)
                os.close(dir_fd)
                dir_fd = None
                return fd, None

            # Intermediate directories: must be real dirs, no symlink follow.
            flags = base | nofollow
            if o_dir:
                flags |= o_dir
            try:
                next_fd = os.open(part, flags, dir_fd=dir_fd)
            except OSError as e:
                return None, _map_open_oserror(e)
            os.close(dir_fd)
            dir_fd = next_fd

        return None, _err("source_not_regular", "source_path is not a regular file")
    except OSError as e:
        return None, _map_open_oserror(e)
    finally:
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError:
                pass


def _map_open_oserror(e):
    """Map OSError from open/openat to stable error_code (no path in detail)."""
    en = getattr(e, "errno", None)
    # Symlink / loop / wrong type for O_NOFOLLOW or O_DIRECTORY
    if en in (
        getattr(errno, "ELOOP", -1),
        getattr(errno, "EMLINK", -1),  # some platforms
        getattr(errno, "EFTYPE", -1),
        getattr(errno, "ENOTDIR", -1),
    ):
        # Treat symlink / wrong component as not allowed (or not regular for ENOTDIR last)
        if en == getattr(errno, "ENOTDIR", -1):
            return _err("source_not_regular", "source_path is not a regular file")
        return _err("source_path_not_allowed", "source_path not under allowlist roots")
    if en in (getattr(errno, "ENOENT", -1),):
        return _err("source_not_found", "source file not found")
    if en in (getattr(errno, "EACCES", -1), getattr(errno, "EPERM", -1)):
        return _err("source_io_error", "source file unreadable")
    return _err("source_io_error", "source file unreadable")


def _read_upto(fd, limit):
    """Read at most `limit` bytes, looping until EOF or the limit is reached.

    A single os.read() may legally return short; looping keeps an image payload
    from being silently truncated (the LOUD contract has no room for half a file).
    """
    chunks = []
    got = 0
    while got < limit:
        chunk = os.read(fd, limit - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _read_bytes_under_roots(path, max_bytes, roots=None):
    """Shared fence: allowlist membership + O_NOFOLLOW openat walk + size cap.

    Returns (bytes, None) or (None, error_envelope). Text and binary callers share
    this so there is ONE fence and ONE error taxonomy.
    """
    if roots is None:
        roots = parse_source_roots()
    if path is None or path == "":
        return None, _err("invalid_arguments", "source_path is empty")

    # Do not strip — path characters may include trailing spaces on some systems.
    abs_path = path
    real_root, parts, err = _resolve_root_and_parts(abs_path, roots)
    if err:
        return None, err

    fd = None
    try:
        fd, open_err = _openat_walk(real_root, parts)
        if open_err:
            return None, open_err

        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, _err("source_not_regular", "source_path is not a regular file")
        if st.st_size > max_bytes:
            return None, _err(
                "source_too_large",
                "source file exceeds %d bytes" % max_bytes,
            )
        # Read max+1 to catch growth after fstat.
        data = _read_upto(fd, max_bytes + 1)
        if len(data) > max_bytes:
            return None, _err(
                "source_too_large",
                "source file exceeds %d bytes" % max_bytes,
            )
        return data, None
    except OSError:
        return None, _err("source_io_error", "source file unreadable")
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def open_binary_under_roots(path, max_bytes, roots=None):
    """Read an allowlisted file as RAW BYTES. Returns (bytes, None) or (None, err).

    Same fence, same error_code taxonomy as source_path (llm_generate `files=` on
    grok needs the bytes server-side to base64 them into the inline ACP payload).
    No UTF-8 decode — the payload is an image.
    """
    return _read_bytes_under_roots(path, max_bytes, roots=roots)


def read_source_file(source_path, max_bytes=SOURCE_PATH_MAX_BYTES, roots=None):
    """Read allowlisted file as UTF-8 strict. Returns (text, None) or (None, err)."""
    data, err = _read_bytes_under_roots(source_path, max_bytes, roots=roots)
    if err:
        return None, err
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, _err("source_invalid_utf8", "source file is not valid UTF-8")
    return text, None


def resolve_big_context_source(text, source_path, instruction=""):
    """Resolve source body for llm_big_context.

    Returns (source_body, None) or (None, error_envelope).

    - No source_path (None or ""): legacy — body is text as provided by caller
      (caller still applies empty check). This function returns text unchanged.
    - Non-empty source_path: path mode; text.strip() must be empty (XOR).
    """
    path = source_path
    path_mode = path is not None and path != ""

    if not path_mode:
        return text, None

    if str(text or "").strip():
        return None, _err(
            "invalid_arguments",
            "source_path and text are mutually exclusive; provide only one",
        )

    body, err = read_source_file(path)
    if err:
        return None, err

    # Assembled prompt size (instruction + framing + body).
    prompt = "%s\n\n--- SOURCE (public) ---\n%s" % (instruction, body)
    if len(prompt.encode("utf-8")) > SOURCE_PROMPT_MAX_BYTES:
        return None, _err(
            "source_too_large",
            "assembled prompt exceeds %d bytes" % SOURCE_PROMPT_MAX_BYTES,
        )
    return body, None
