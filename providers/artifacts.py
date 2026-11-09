# no-opencc
"""Opaque artifact store for large tool results (issue #22 P1.4).

Spilled bodies live under $OMNI_LLM_STATE_DIR/artifacts/<id>/{meta.json,blob},
a sibling of jobs/ — never inside it. Default tool behavior stays inline;
callers opt in via result_mode / --out-mode.

Trust model: the integrity anchor is the content-addressed id
(id == sha256(blob).hexdigest()), which travels in the envelope/conversation
OUTSIDE the store. Rehydrated content is untrusted prior LLM output — never a
trusted earlier result. Same-UID writers can poison the store on disk; callers
detect tampering because the id they already hold will not match a mutated blob.
Application code cannot bound a full same-UID attacker with read+write access.

Path ops use dir_fd discipline (openat + O_NOFOLLOW + fstat + fchmod on fds) so
client-supplied ids and store races cannot escape the pinned store anchor.
meta.json is untrusted; its sha256 field is advisory only.
"""
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time

from providers import known_secrets
from providers.errors import redact_secrets

ENV_STATE_DIR = "OMNI_LLM_STATE_DIR"
ENV_ARTIFACT_TTL_S = "OMNI_LLM_ARTIFACT_TTL_S"
ENV_ARTIFACT_MAX_BYTES = "OMNI_LLM_ARTIFACT_MAX_BYTES"
ENV_ARTIFACT_STORE_MAX_BYTES = "OMNI_LLM_ARTIFACT_STORE_MAX_BYTES"
ENV_ARTIFACT_STORE_MAX_ENTRIES = "OMNI_LLM_ARTIFACT_STORE_MAX_ENTRIES"
DEFAULT_STATE_DIR = "~/.local/state/omni-llm"
DEFAULT_TTL_S = 7 * 24 * 3600  # 7 days
DEFAULT_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB per artifact
DEFAULT_STORE_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB store
DEFAULT_STORE_MAX_ENTRIES = 10000
META_MAX_BYTES = 64 * 1024  # 64 KiB independent meta cap
AUTO_THRESHOLD_BYTES = 65536
PREVIEW_CHARS = 512
PREVIEW_SUFFIX = "…[spilled]"
DIR_MODE = 0o700
FILE_MODE = 0o600
RESULT_MODES = frozenset(("inline", "ref", "auto"))
# Content-addressed id: lowercase hex sha256 (fixed length; no case-fold collisions).
_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_META_NAME = "meta.json"
_BLOB_NAME = "blob"
_WHITELIST_NAMES = frozenset((_META_NAME, _BLOB_NAME))
_PUB_PREFIX = ".omni-art-pub-"
# L5: bounded LOCK_NB retry — never hang a tool call on a stuck holder.
# Retry budget is TOTAL per public read/create/remove op, not per entry.
# Collectors (gc / _reclaim_expired) use single-shot LOCK_NB only (B2).
_LOCK_RETRY_BUDGET_S = 0.25  # ~250 ms total
_LOCK_RETRY_SLEEP_S = 0.01  # 10 ms between attempts
# Busy only for EWOULDBLOCK/EAGAIN (L6). EAGAIN is deliberately accepted as
# equivalent to EWOULDBLOCK (identical on Linux; portable busy signal elsewhere).
_LOCK_BUSY_ERRNOS = frozenset(
    e for e in (errno.EWOULDBLOCK, errno.EAGAIN) if e is not None
)

# Tests inject a fake clock via set_now_fn.
_now_fn = None


class ArtifactError(Exception):
    """Typed create/policy failure (quota, size, runtime)."""

    def __init__(self, reason, detail=None):
        self.reason = reason
        self.detail = None if detail is None else redact_secrets(str(detail))[:300]
        msg = reason if self.detail is None else "%s: %s" % (reason, self.detail)
        super().__init__(msg)


def set_now_fn(fn):
    """Install a zero-arg clock (or None to restore time.time)."""
    global _now_fn
    _now_fn = fn


def now():
    if _now_fn is not None:
        return float(_now_fn())
    return time.time()


def _state_dir():
    return os.path.abspath(os.path.expanduser(
        os.environ.get(ENV_STATE_DIR) or DEFAULT_STATE_DIR
    ))


def artifacts_root():
    return os.path.join(_state_dir(), "artifacts")


def default_ttl_s():
    raw = os.environ.get(ENV_ARTIFACT_TTL_S)
    if raw is None or raw == "":
        return DEFAULT_TTL_S
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TTL_S
    if val <= 0:
        return DEFAULT_TTL_S
    return val


def max_artifact_bytes():
    raw = os.environ.get(ENV_ARTIFACT_MAX_BYTES)
    if raw is None or raw == "":
        return DEFAULT_MAX_BYTES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES
    if val <= 0:
        return DEFAULT_MAX_BYTES
    return val


def max_store_bytes():
    raw = os.environ.get(ENV_ARTIFACT_STORE_MAX_BYTES)
    if raw is None or raw == "":
        return DEFAULT_STORE_MAX_BYTES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_STORE_MAX_BYTES
    if val <= 0:
        return DEFAULT_STORE_MAX_BYTES
    return val


def max_store_entries():
    raw = os.environ.get(ENV_ARTIFACT_STORE_MAX_ENTRIES)
    if raw is None or raw == "":
        return DEFAULT_STORE_MAX_ENTRIES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_STORE_MAX_ENTRIES
    if val <= 0:
        return DEFAULT_STORE_MAX_ENTRIES
    return val


def _flags_rdonly():
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _flags_wronly_creat_excl():
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _nofollow():
    return getattr(os, "O_NOFOLLOW", 0)


def _odirectory():
    return getattr(os, "O_DIRECTORY", 0)


def _fstat_owned_dir(fd):
    """fstat fd; True if real directory owned by us."""
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode):
        return False, st
    if st.st_uid != os.getuid():
        return False, st
    return True, st


def _fstat_regular_file(fd):
    """fstat fd; True only for a regular file (reject FIFO/symlink/dir)."""
    st = os.fstat(fd)
    return stat.S_ISREG(st.st_mode), st


def _open_owned_dir_at(path, *, dir_fd=None):
    """Open path as O_DIRECTORY|O_NOFOLLOW; require owned dir; fchmod 0700.

    Never chmod by path string (symlink-safe). Failures are LOUD OSError.
    """
    flags = _flags_rdonly() | _odirectory() | _nofollow()
    if dir_fd is None:
        fd = os.open(path, flags)
    else:
        fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        ok, st = _fstat_owned_dir(fd)
        if not ok:
            if not stat.S_ISDIR(st.st_mode):
                raise OSError(errno.ENOTDIR, "store path is not a directory", path)
            raise OSError(errno.EPERM, "store path not owned by us", path)
        os.fchmod(fd, DIR_MODE)
        st2 = os.fstat(fd)
        mode = stat.S_IMODE(st2.st_mode)
        if mode & 0o077:
            raise OSError(errno.EPERM, "store path mode looser than 0700", path)
        return fd
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _ensure_and_open_root():
    """Ensure state+artifacts exist via fd discipline; return root dir_fd.

    Never chmod a path before it is proven non-symlink: open each component
    with O_DIRECTORY|O_NOFOLLOW, fstat owned directory, then fchmod on the fd.
    """
    state = _state_dir()
    # Create the state path if missing (mkdir is fine; we never chmod-by-path).
    if not os.path.lexists(state):
        os.makedirs(state, mode=DIR_MODE, exist_ok=True)

    state_fd = _open_owned_dir_at(state)
    try:
        try:
            os.mkdir("artifacts", DIR_MODE, dir_fd=state_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            # Concurrent create or pre-existing non-dir — open will diagnose.
            if getattr(exc, "errno", None) not in (errno.EEXIST,):
                # Still try open; open is authoritative.
                pass
        root_fd = _open_owned_dir_at("artifacts", dir_fd=state_fd)
        return root_fd
    finally:
        try:
            os.close(state_fd)
        except OSError:
            pass


def _is_confined_id(artifact_id):
    """True only for a single path component safe under the store root.

    Content-addressed: exactly 64 lowercase hex chars. Structural dirname
    checks remain as defense in depth.
    """
    if not isinstance(artifact_id, str) or not artifact_id:
        return False
    if not _ID_RE.fullmatch(artifact_id):
        return False
    if artifact_id in (".", ".."):
        return False
    if ".." in artifact_id:
        return False
    if os.path.isabs(artifact_id):
        return False
    if os.sep in artifact_id:
        return False
    if os.altsep is not None and os.altsep in artifact_id:
        return False
    if "\x00" in artifact_id:
        return False
    if os.path.dirname(artifact_id) not in ("", "."):
        return False
    return True


def _read_capped_fd(fd, max_bytes):
    """Read fd up to max_bytes; accumulate and abort if more arrives.

    st_size is only a fast pre-check (file can grow after fstat). Returns bytes.
    Raises OSError(EFBIG) if the cap is exceeded.
    """
    st = os.fstat(fd)
    if st.st_size > max_bytes:
        raise OSError(errno.EFBIG, "artifact exceeds size cap")
    chunks = []
    total = 0
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise OSError(errno.EFBIG, "artifact exceeds size cap")
        chunks.append(block)
    return b"".join(chunks)


def _read_blob_streaming(fd, max_bytes):
    """Read blob with streaming sha256; enforce cap on bytes actually read.

    Returns (blob_bytes, hex_digest). Raises OSError(EFBIG) past cap.
    """
    st = os.fstat(fd)
    if st.st_size > max_bytes:
        raise OSError(errno.EFBIG, "artifact exceeds size cap")
    h = hashlib.sha256()
    chunks = []
    total = 0
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise OSError(errno.EFBIG, "artifact exceeds size cap")
        h.update(block)
        chunks.append(block)
    return b"".join(chunks), h.hexdigest()


def _open_artifact_dir(root_fd, artifact_id):
    """Openat artifact id under root; fstat owned directory.

    Returns (art_fd, None) or (None, reason).
    Only ENOENT → artifact_not_found; ELOOP/ENOTDIR/other → artifact_corrupt
    (or runtime via caller for unexpected).
    Does not take a flock — callers that need cooperation use
    `_open_locked_artifact` (L1–L3).
    """
    flags = _flags_rdonly() | _odirectory() | _nofollow()
    try:
        art_fd = os.open(artifact_id, flags, dir_fd=root_fd)
    except OSError as exc:
        en = getattr(exc, "errno", None)
        if en == errno.ENOENT:
            return None, "artifact_not_found"
        if en in (errno.ELOOP, errno.ENOTDIR, errno.EINVAL, errno.EPERM):
            return None, "artifact_corrupt"
        return None, "artifact_corrupt"
    try:
        ok, _st = _fstat_owned_dir(art_fd)
        if not ok:
            os.close(art_fd)
            return None, "artifact_corrupt"
        return art_fd, None
    except Exception:
        try:
            os.close(art_fd)
        except OSError:
            pass
        raise


def _flock_bounded(fd, lock_mode, *, single_shot=False):
    """Acquire flock with LOCK_NB (L5).

    lock_mode is fcntl.LOCK_SH or fcntl.LOCK_EX (LOCK_NB is always OR'd in).
    Returns (None, None) on success, or (reason, detail) on failure.

    single_shot=True: one non-blocking attempt (collectors — B2). Busy returns
    immediately as artifact_busy; never sleeps or retries across entries.

    single_shot=False: retry until total deadline _LOCK_RETRY_BUDGET_S from
    this call (public read/create/remove). Budget is per operation, not per
    busy entry in a scan.

    L6: ONLY EWOULDBLOCK/EAGAIN → artifact_busy. ENOTSUP/EBADF/ENOLCK/other
    are LOUD typed failures — never treated as busy, never silent proceed.
    """
    flags = lock_mode | fcntl.LOCK_NB
    deadline = None if single_shot else (time.monotonic() + _LOCK_RETRY_BUDGET_S)
    while True:
        try:
            fcntl.flock(fd, flags)
            return None, None
        except OSError as exc:
            en = getattr(exc, "errno", None)
            if en in _LOCK_BUSY_ERRNOS:
                if single_shot or time.monotonic() >= deadline:
                    return "artifact_busy", "lock not acquired within bound"
                time.sleep(_LOCK_RETRY_SLEEP_S)
                continue
            # Loud: filesystem cannot lock, bad fd, etc.
            return (
                "runtime_error",
                "flock failed errno=%s %s" % (en, exc),
            )


def _reverify_name(root_fd, name, art_st):
    """Three-state lock-then-reverify (L3 / I2).

    Uses os.stat (not open) so reverify does not open a second descriptor on
    the artifact directory while the caller holds its flock (F5).

    Returns:
      None                 — name still refers to art_st's (dev, ino) directory
      "artifact_not_found" — ENOENT, non-dir, or inode mismatch
      "runtime_error"      — any other OSError (EIO, EACCES, …); never masquerade
                             as not_found
    """
    try:
        st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as exc:
        en = getattr(exc, "errno", None)
        if en == errno.ENOENT:
            return "artifact_not_found"
        return "runtime_error"
    if not stat.S_ISDIR(st.st_mode):
        return "artifact_not_found"
    if st.st_dev == art_st.st_dev and st.st_ino == art_st.st_ino:
        return None
    return "artifact_not_found"


def _name_still_same_inode(root_fd, name, art_st):
    """True iff name still refers to art_st's inode (boolean view of reverify).

    Prefer _reverify_name when the three-state distinction matters.
    """
    return _reverify_name(root_fd, name, art_st) is None


def _open_locked_artifact(root_fd, artifact_id, lock_mode, *, single_shot=False):
    """Open artifact dir, acquire lock once, then lock-then-reverify (L1–L3).

    lock_mode: fcntl.LOCK_SH (read) or fcntl.LOCK_EX (mutators of existing
    entries). Never upgrades SH→EX (L2).

    single_shot: passed to _flock_bounded — True for collectors (B2).

    Returns (art_fd, art_st, None) on success. Caller owns art_fd and must
    close it (closing releases the lock per F3). On failure returns
    (None, None, reason) and no open fd is left behind.

    Any failure after art_fd is opened (including non-OSError from the lock
    helper) closes the fd before returning or re-raising (I1).

    L3: after the lock, name under root_fd must still be the same
    (st_dev, st_ino). Name gone or different inode → artifact_not_found
    (collected/replaced — not corrupt). Stat failures other than ENOENT →
    runtime_error (I2). Member damage is a separate check via validation
    after this returns.
    """
    art_fd, why = _open_artifact_dir(root_fd, artifact_id)
    if art_fd is None:
        return None, None, why
    try:
        try:
            art_st = os.fstat(art_fd)
        except OSError:
            return None, None, "runtime_error"

        lock_why, _detail = _flock_bounded(
            art_fd, lock_mode, single_shot=single_shot
        )
        if lock_why is not None:
            return None, None, lock_why

        # L3 LOCK-THEN-REVERIFY (three-state)
        rev = _reverify_name(root_fd, artifact_id, art_st)
        if rev is not None:
            return None, None, rev

        owned = art_fd
        art_fd = None  # caller owns it; finally must not close
        return owned, art_st, None
    except Exception:
        raise
    finally:
        if art_fd is not None:
            try:
                os.close(art_fd)
            except OSError:
                pass


def _delete_entry_locked(root_fd, name, art_fd, art_st):
    """Unlink whitelist + rmdir while still holding LOCK_EX on art_fd (L4).

    Do NOT close art_fd before rmdir (F4). Caller closes art_fd after this
    returns (success or fail). Returns None on success, or reason string.

    L4 ownership is an ordering property: art_fd must still hold EX at the
    moment of rmdir. That is pinned by tests (second open + LOCK_SH|LOCK_NB
    probe at the real rmdir boundary — F5), not by a production fstat check.
    """
    try:
        _unlink_whitelist(art_fd)
    except ArtifactError as exc:
        return exc.reason
    except OSError:
        return "runtime_error"

    rev = _reverify_name(root_fd, name, art_st)
    if rev is not None:
        # Name swapped/collected — do not rmdir a different inode's name.
        # runtime_error from stat also refuses the rmdir.
        return rev

    try:
        os.rmdir(name, dir_fd=root_fd)
    except OSError as exc:
        en = getattr(exc, "errno", None)
        if en == errno.ENOENT:
            return "artifact_not_found"
        return "runtime_error"
    return None


def _open_reg_in_dir(art_fd, name):
    """Open name under art_fd with O_NOFOLLOW|O_NONBLOCK; require regular file.

    Returns (fd, None) or (None, reason). Only member ENOENT → corrupt
    (dir present, member missing). ELOOP/etc → corrupt.
    """
    flags = _flags_rdonly() | _nofollow()
    try:
        fd = os.open(name, flags, dir_fd=art_fd)
    except OSError as exc:
        en = getattr(exc, "errno", None)
        if en == errno.ENOENT:
            return None, "artifact_corrupt"
        if en in (errno.ELOOP, errno.EISDIR, errno.ENOTDIR, errno.EINVAL):
            return None, "artifact_corrupt"
        return None, "artifact_corrupt"
    try:
        is_reg, _st = _fstat_regular_file(fd)
        if not is_reg:
            os.close(fd)
            return None, "artifact_corrupt"
        return fd, None
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _write_file_in_dir(dir_fd, name, data, mode=FILE_MODE):
    """Create exclusive regular file under dir_fd; write, fchmod, fsync.

    Operates only through the pinned dir fd (O_NOFOLLOW). Returns nothing.
    """
    flags = _flags_wronly_creat_excl() | _nofollow()
    fd = os.open(name, flags, mode, dir_fd=dir_fd)
    try:
        try:
            os.fchmod(fd, mode)
        except OSError:
            pass
        is_reg, _st = _fstat_regular_file(fd)
        if not is_reg:
            raise OSError(errno.EINVAL, "created member is not a regular file", name)
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short artifact write")
            offset += written
        os.fsync(fd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _cleanup_pub_dir(root_fd, tmp_fd, tmp_name):
    """Error-path cleanup via pinned fds only — never path-walk the temp name.

    Unlink whitelisted members through tmp_fd, then rmdir via root_fd. If the
    name was swapped for a symlink to an external decoy, the open tmp_fd still
    addresses the original inode; path-based cleanup would follow the swap.
    """
    if tmp_fd is not None:
        for name in _WHITELIST_NAMES:
            try:
                os.unlink(name, dir_fd=tmp_fd)
            except OSError:
                pass
        try:
            os.close(tmp_fd)
        except OSError:
            pass
    if tmp_name is not None and root_fd is not None:
        try:
            os.rmdir(tmp_name, dir_fd=root_fd)
        except OSError:
            pass


def _unlink_whitelist(art_fd):
    """Unlink only meta.json and blob via art_fd."""
    for name in _WHITELIST_NAMES:
        try:
            os.unlink(name, dir_fd=art_fd)
        except FileNotFoundError:
            continue
        except OSError as exc:
            en = getattr(exc, "errno", None)
            if en == errno.ENOENT:
                continue
            if en == errno.EISDIR:
                raise ArtifactError(
                    "artifact_corrupt", "unexpected subdirectory"
                )
            raise


def _parse_meta_bytes(meta_raw, artifact_id):
    """Closed schema parse. Returns (meta_dict, expires_f) or raises ValueError."""
    try:
        meta = json.loads(meta_raw.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("meta json: %s" % exc) from exc
    if not isinstance(meta, dict):
        raise ValueError("meta is not an object")
    meta_id = meta.get("id")
    if meta_id != artifact_id:
        raise ValueError("id mismatch")
    expires = meta.get("expires")
    try:
        expires_f = float(expires)
    except (TypeError, ValueError) as exc:
        raise ValueError("bad expires") from exc
    if not math.isfinite(expires_f):
        raise ValueError("non-finite expires")
    return meta, expires_f


def make_preview(text):
    body = text if isinstance(text, str) else str(text)
    if len(body) <= PREVIEW_CHARS:
        head = body
    else:
        head = body[:PREVIEW_CHARS]
    return head + PREVIEW_SUFFIX


def utf8_len(text):
    if text is None:
        return 0
    if isinstance(text, bytes):
        return len(text)
    return len(str(text).encode("utf-8"))


def should_spill(result_mode, text):
    """Return True when opt-in mode requires storing full text off-envelope."""
    mode = "inline" if result_mode is None or result_mode == "" else str(result_mode)
    if mode == "inline":
        return False
    if not isinstance(text, str) or text == "":
        return False
    if mode == "ref":
        return True
    if mode == "auto":
        return utf8_len(text) > AUTO_THRESHOLD_BYTES
    return False


def _store_usage(root_fd):
    """Sum blob sizes and count published entries under root_fd."""
    total_bytes = 0
    entries = 0
    try:
        names = os.listdir(root_fd)
    except OSError:
        return 0, 0
    for name in names:
        if name.startswith("."):
            continue
        if not _is_confined_id(name):
            continue
        art_fd, why = _open_artifact_dir(root_fd, name)
        if art_fd is None:
            continue
        try:
            entries += 1
            blob_fd, bwhy = _open_reg_in_dir(art_fd, _BLOB_NAME)
            if blob_fd is None:
                continue
            try:
                st = os.fstat(blob_fd)
                total_bytes += int(st.st_size)
            finally:
                os.close(blob_fd)
        finally:
            os.close(art_fd)
    return total_bytes, entries


def _validate_entry(art_fd, artifact_id, *, check_expiry=True, max_blob=None):
    """Strict manifest+blob validation bound to art_fd (same rules as read).

    Returns dict {ok, reason?, text?, digest?, text_bytes?, expires?} without
    closing art_fd. Does not re-resolve the entry by name.
    """
    if max_blob is None:
        max_blob = max_artifact_bytes()

    meta_fd, why = _open_reg_in_dir(art_fd, _META_NAME)
    if meta_fd is None:
        return {"ok": False, "reason": why or "artifact_corrupt", "detail": "meta.json"}
    try:
        try:
            meta_raw = _read_capped_fd(meta_fd, META_MAX_BYTES)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EFBIG:
                return {"ok": False, "reason": "artifact_corrupt", "detail": "meta too large"}
            return {"ok": False, "reason": "artifact_corrupt", "detail": redact_secrets(str(exc))}
    finally:
        try:
            os.close(meta_fd)
        except OSError:
            pass

    try:
        _meta, expires_f = _parse_meta_bytes(meta_raw, artifact_id)
    except ValueError as exc:
        return {"ok": False, "reason": "artifact_corrupt", "detail": redact_secrets(str(exc))}

    blob_fd, why = _open_reg_in_dir(art_fd, _BLOB_NAME)
    if blob_fd is None:
        return {"ok": False, "reason": why or "artifact_corrupt", "detail": "blob"}
    try:
        if check_expiry and now() > expires_f:
            return {
                "ok": False,
                "reason": "artifact_expired",
                "detail": "expires=%s" % expires_f,
                "expires": expires_f,
            }
        try:
            blob, digest = _read_blob_streaming(blob_fd, max_blob)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EFBIG:
                return {
                    "ok": False,
                    "reason": "artifact_corrupt",
                    "detail": "blob exceeds size cap",
                }
            return {"ok": False, "reason": "artifact_corrupt", "detail": redact_secrets(str(exc))}
    finally:
        try:
            os.close(blob_fd)
        except OSError:
            pass

    # Integrity anchor is the content-addressed id (outside the store).
    # meta.sha256 is advisory only — never the integrity source.
    if digest != artifact_id:
        return {"ok": False, "reason": "artifact_corrupt", "detail": "content id mismatch"}

    if check_expiry and now() > expires_f:
        return {
            "ok": False,
            "reason": "artifact_expired",
            "detail": "expires=%s" % expires_f,
            "expires": expires_f,
        }

    try:
        text = blob.decode("utf-8")
    except UnicodeError as exc:
        return {"ok": False, "reason": "artifact_corrupt", "detail": redact_secrets(str(exc))}

    return {
        "ok": True,
        "text": text,
        "digest": digest,
        "text_bytes": len(blob),
        "expires": expires_f,
    }


def _refresh_expiry(art_fd, artifact_id, expires, created=None):
    """Rewrite meta.json under art_fd with a new expires (idempotent publish)."""
    if created is None:
        created = now()
    meta = {
        "id": artifact_id,
        "sha256": artifact_id,  # advisory mirror of content id
        "text_bytes": None,  # filled below if blob readable
        "created": created,
        "expires": expires,
    }
    # Prefer actual blob size for meta bookkeeping.
    blob_fd, why = _open_reg_in_dir(art_fd, _BLOB_NAME)
    if blob_fd is not None:
        try:
            st = os.fstat(blob_fd)
            meta["text_bytes"] = int(st.st_size)
        finally:
            os.close(blob_fd)
    else:
        meta["text_bytes"] = 0

    # Replace meta: unlink then exclusive create via pinned fd.
    try:
        os.unlink(_META_NAME, dir_fd=art_fd)
    except OSError:
        pass
    meta_bytes = (
        json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_file_in_dir(art_fd, _META_NAME, meta_bytes, mode=FILE_MODE)
    try:
        os.fsync(art_fd)
    except OSError:
        pass


def _reclaim_expired(root_fd):
    """Delete expired (strictly validated) entries to free quota. No markers.

    Mutator of existing entries: single-shot LOCK_EX + lock-then-reverify
    (L1, L3, B2). Busy and competitive not_found are skipped (V4); any
    runtime_error (including non-busy lock failures) raises ArtifactError
    so create never reports store_full off the back of a silent unlock (B1).
    """
    try:
        names = list(os.listdir(root_fd))
    except OSError:
        return 0
    deleted = 0
    now_ts = now()
    for name in names:
        if name.startswith("."):
            continue
        if not _is_confined_id(name):
            continue
        art_fd, art_st, why = _open_locked_artifact(
            root_fd, name, fcntl.LOCK_EX, single_shot=True
        )
        if art_fd is None:
            if why in ("artifact_busy", "artifact_not_found"):
                continue
            if why == "runtime_error":
                raise ArtifactError(
                    "runtime_error",
                    "lock or reverify failed during reclaim",
                )
            # corrupt open — skip; do not pretend reclaimed
            continue
        try:
            # Validate without treating expiry as success-path failure for GC:
            # read meta+blob strictly; content id must match; only then check expires.
            v = _validate_entry(art_fd, name, check_expiry=False)
            if not v.get("ok"):
                # Corrupt → never silently delete as expired.
                continue
            if now_ts <= v["expires"]:
                continue
            # L4: hold EX through rmdir on the same pinned fd.
            del_why = _delete_entry_locked(root_fd, name, art_fd, art_st)
            if del_why is None:
                deleted += 1
            elif del_why == "artifact_not_found":
                continue
            elif del_why == "runtime_error":
                raise ArtifactError(
                    "runtime_error",
                    "delete failed during reclaim",
                )
        finally:
            try:
                os.close(art_fd)
            except OSError:
                pass
    return deleted


def _handle_existing_locked(root_fd, artifact_id, raw, digest, text_bytes,
                            created, expires):
    """Idempotent refresh or corrupt-entry replacement; acquires LOCK_EX itself.

    Callers must NOT hold a lock on the entry. This function opens and locks
    via _open_locked_artifact (bounded retry — public create path), then
    refreshes a valid same-id entry or replaces a corrupt one under EX.

    Returns a ref dict on successful refresh, or None when the name is clear
    so the caller can publish fresh. Raises ArtifactError on hard failures
    (busy, runtime_error, failed corrupt clear).
    """
    art_fd, art_st, why = _open_locked_artifact(
        root_fd, artifact_id, fcntl.LOCK_EX
    )
    if art_fd is None:
        if why == "artifact_busy":
            raise ArtifactError(
                "artifact_busy", "existing entry is locked"
            )
        if why == "artifact_not_found":
            return None  # caller publishes new
        if why == "runtime_error":
            raise ArtifactError("runtime_error", "lock or open failed")
        # corrupt open (symlink etc.) — let publish path attempt / race
        return None

    try:
        v = _validate_entry(art_fd, artifact_id, check_expiry=False)
        if v.get("ok") and v.get("digest") == artifact_id:
            _refresh_expiry(art_fd, artifact_id, expires, created=created)
            return {
                "id": artifact_id,
                "sha256": digest,
                "text_bytes": text_bytes,
                "expires": expires,
            }
        # Corrupt same-id entry: replace via pinned fd while holding EX (L4).
        del_why = _delete_entry_locked(root_fd, artifact_id, art_fd, art_st)
        if del_why not in (None, "artifact_not_found"):
            raise ArtifactError(
                del_why if del_why != "runtime_error" else "runtime_error",
                "failed to clear corrupt entry",
            )
        return None  # proceed to publish
    finally:
        try:
            os.close(art_fd)
        except OSError:
            pass


def create(text, *, ttl_s=None):
    """Persist text; return {id, sha256, text_bytes, expires}.

    id = sha256(blob).hexdigest() (content-addressed). Identical content is
    idempotent (refresh expiry, no duplicate dir). Resource bounds are enforced
    BEFORE any bytes are written. All mutators use pinned dir fds.

    Raises UnicodeEncodeError if text cannot be encoded as UTF-8.
    Raises ArtifactError for size/quota/runtime policy failures.
    """
    if not isinstance(text, str):
        text = str(text)
    raw = text.encode("utf-8")
    text_bytes = len(raw)
    cap = max_artifact_bytes()
    # Refuse over-cap BEFORE writing anything.
    if text_bytes > cap:
        raise ArtifactError(
            "artifact_too_large",
            "text_bytes=%s max=%s" % (text_bytes, cap),
        )

    digest = hashlib.sha256(raw).hexdigest()
    artifact_id = digest  # content-addressed

    if ttl_s is None:
        ttl_s = default_ttl_s()
    try:
        ttl_s = int(ttl_s)
    except (TypeError, ValueError):
        ttl_s = default_ttl_s()
    if ttl_s <= 0:
        ttl_s = default_ttl_s()

    created = now()
    expires = created + float(ttl_s)

    root_fd = _ensure_and_open_root()
    tmp_fd = None
    tmp_name = None
    try:
        # Idempotent re-create / corrupt replace: LOCK_EX on existing (L1).
        existing = _handle_existing_locked(
            root_fd, artifact_id, raw, digest, text_bytes, created, expires
        )
        if existing is not None:
            return existing

        # Store quota: only when the new blob would exceed, attempt to reclaim
        # expired entries, then refuse with a typed reason if still over.
        # Explicit `omni artifact gc` remains the only user-facing collector;
        # create does not opportunistically reap on every publish.
        # V4: reclaim skips busy entries; if still full, report store-full.
        used_bytes, used_entries = _store_usage(root_fd)
        store_cap = max_store_bytes()
        entry_cap = max_store_entries()
        would_exceed = (
            used_entries >= entry_cap
            or (used_bytes + text_bytes) > store_cap
        )
        if would_exceed:
            _reclaim_expired(root_fd)
            used_bytes, used_entries = _store_usage(root_fd)
            if (
                used_entries >= entry_cap
                or (used_bytes + text_bytes) > store_cap
            ):
                raise ArtifactError(
                    "artifact_store_full",
                    "bytes=%s+%s cap=%s entries=%s cap=%s"
                    % (used_bytes, text_bytes, store_cap, used_entries, entry_cap),
                )

        # Publish into a temp dir under the root, all via dir fds.
        # Brand-new entry: no lock before publish (L1). If rename loses to a
        # concurrent winner, fall back into the existing-entry locked path.
        tmp_name = _PUB_PREFIX + digest[:16] + "-%s" % os.getpid()
        # Unique-ify on collision.
        n = 0
        while True:
            candidate = tmp_name if n == 0 else "%s-%d" % (tmp_name, n)
            try:
                os.mkdir(candidate, DIR_MODE, dir_fd=root_fd)
                tmp_name = candidate
                break
            except FileExistsError:
                n += 1
                if n > 100:
                    raise ArtifactError("runtime_error", "temp dir collision")

        # Open temp immediately; all permission work on this fd BEFORE publish.
        tmp_fd = _open_owned_dir_at(tmp_name, dir_fd=root_fd)
        try:
            _write_file_in_dir(tmp_fd, _BLOB_NAME, raw, mode=FILE_MODE)
            meta = {
                "id": artifact_id,
                "sha256": digest,  # advisory mirror; integrity is the id
                "text_bytes": text_bytes,
                "created": created,
                "expires": expires,
            }
            meta_bytes = (
                json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            if len(meta_bytes) > META_MAX_BYTES:
                raise ArtifactError("runtime_error", "meta exceeds cap")
            _write_file_in_dir(tmp_fd, _META_NAME, meta_bytes, mode=FILE_MODE)
            # Durability: fsync blob+meta already done in _write_file_in_dir;
            # fsync temp dir, then rename, then fsync root.
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            tmp_fd = None
            try:
                os.rename(
                    tmp_name, artifact_id,
                    src_dir_fd=root_fd, dst_dir_fd=root_fd,
                )
                tmp_name = None  # published; no cleanup
            except OSError as exc:
                en = getattr(exc, "errno", None)
                # Concurrent winner published the same content-id first (L1).
                if en in (
                    errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR, errno.ENOTDIR
                ):
                    _cleanup_pub_dir(root_fd, None, tmp_name)
                    tmp_name = None
                    existing = _handle_existing_locked(
                        root_fd, artifact_id, raw, digest, text_bytes,
                        created, expires,
                    )
                    if existing is not None:
                        return existing
                    raise ArtifactError(
                        "runtime_error",
                        "publish lost race and existing entry unusable",
                    )
                raise
            try:
                os.fsync(root_fd)
            except OSError:
                pass
        except Exception:
            _cleanup_pub_dir(root_fd, tmp_fd, tmp_name)
            tmp_fd = None
            tmp_name = None
            raise
    finally:
        if tmp_fd is not None or tmp_name is not None:
            _cleanup_pub_dir(root_fd, tmp_fd, tmp_name)
        try:
            os.close(root_fd)
        except OSError:
            pass

    return {
        "id": artifact_id,
        "sha256": digest,
        "text_bytes": text_bytes,
        "expires": expires,
    }


def _loud(reason, detail=None):
    out = {"ok": False, "reason": reason}
    if detail is not None:
        out["detail"] = str(detail)[:300]
    return out


def read(artifact_id):
    """Return {ok:true, text, artifact_ref{...}} or LOUD typed failure.

    Integrity: digest(blob) must equal the requested content-addressed id.
    meta.sha256 is advisory only. not_found ONLY for unconfined id, ENOENT,
    or lock-then-reverify miss (collected underfoot). Dir present but bad →
    artifact_corrupt. Busy lock → artifact_busy (V3). Rehydrated text is
    untrusted prior LLM output.

    Takes LOCK_SH once (L1); never upgrades (L2). Bounded acquisition (L5).
    """
    if not _is_confined_id(artifact_id):
        return _loud("artifact_not_found", "unconfined or invalid id")

    try:
        root_fd = _ensure_and_open_root()
    except OSError as exc:
        return _loud("runtime_error", redact_secrets(str(exc)))

    art_fd = None
    try:
        art_fd, _art_st, why = _open_locked_artifact(
            root_fd, artifact_id, fcntl.LOCK_SH
        )
        if art_fd is None:
            return _loud(why)

        v = _validate_entry(art_fd, artifact_id, check_expiry=True)
        if not v.get("ok"):
            return _loud(v.get("reason") or "artifact_corrupt", v.get("detail"))

        # Expiry re-checked inside _validate_entry immediately before success.
        ref = {
            "id": artifact_id,
            "sha256": v["digest"],
            "text_bytes": v["text_bytes"],
            "expires": v["expires"],
        }
        return {"ok": True, "text": v["text"], "artifact_ref": ref}
    finally:
        if art_fd is not None:
            try:
                os.close(art_fd)
            except OSError:
                pass
        try:
            os.close(root_fd)
        except OSError:
            pass


def remove(artifact_id):
    """Delete one artifact directory. Unknown id → artifact_not_found.

    Unlink ONLY whitelisted names via dir_fd=art_fd; rmdir via dir_fd=root_fd.
    LOCK_EX held through rmdir (L1, L4). Busy → artifact_busy (V2), never a
    pretend success and never not_found for a live locked entry.
    """
    if not _is_confined_id(artifact_id):
        return _loud("artifact_not_found", "unconfined or invalid id")

    try:
        root_fd = _ensure_and_open_root()
    except OSError as exc:
        return _loud("runtime_error", redact_secrets(str(exc)))

    art_fd = None
    try:
        art_fd, art_st, why = _open_locked_artifact(
            root_fd, artifact_id, fcntl.LOCK_EX
        )
        if art_fd is None:
            return _loud(why)

        del_why = _delete_entry_locked(root_fd, artifact_id, art_fd, art_st)
        if del_why is not None:
            return _loud(del_why)
        return {"ok": True, "id": artifact_id}
    finally:
        if art_fd is not None:
            try:
                os.close(art_fd)
            except OSError:
                pass
        try:
            os.close(root_fd)
        except OSError:
            pass


def gc(*, force=False):
    """Delete expired artifacts under artifacts/ only.

    The only collector in v1 (no opportunistic create-path GC). Never touches
    jobs/ or paths outside the store. Unpublished temp dirs (prefix
    .omni-art-pub-) are left alone.

    Validate-then-delete binds one pinned single-shot LOCK_EX fd (L1, L4, B2):
    the entry validated is the entry deleted. Busy holders are skipped and
    counted in skipped_busy (V1); competitive not_found is skipped. Non-busy
    lock/stat failures return LOUD runtime_error — never skipped as success
    (B1). Entries failing strict manifest validation are counted as corrupt
    and never silently treated as expired and deleted.
    """
    try:
        root_fd = _ensure_and_open_root()
    except OSError as exc:
        return _loud("runtime_error", redact_secrets(str(exc)))

    deleted = 0
    examined = 0
    corrupt = 0
    skipped_busy = 0
    now_ts = now()
    try:
        try:
            names = os.listdir(root_fd)
        except OSError as exc:
            return _loud("runtime_error", redact_secrets(str(exc)))

        for name in names:
            if name.startswith("."):
                continue
            if not _is_confined_id(name):
                continue

            art_fd, art_st, why = _open_locked_artifact(
                root_fd, name, fcntl.LOCK_EX, single_shot=True
            )
            if art_fd is None:
                if why == "artifact_busy":
                    skipped_busy += 1
                    continue
                if why == "artifact_not_found":
                    continue
                if why == "artifact_corrupt":
                    corrupt += 1
                    continue
                # runtime_error and any other unexpected reason: LOUD (B1)
                return _loud(
                    why if why else "runtime_error",
                    "lock or reverify failed during gc",
                )

            try:
                v = _validate_entry(art_fd, name, check_expiry=False)
                if not v.get("ok"):
                    # Failing validation → corrupt, never delete as expired.
                    # Missing whitelist member on a stable inode is real damage.
                    corrupt += 1
                    continue
                examined += 1
                if now_ts <= v["expires"]:
                    continue

                # L4: hold EX through rmdir on the same pinned fd.
                del_why = _delete_entry_locked(root_fd, name, art_fd, art_st)
                if del_why is None:
                    deleted += 1
                elif del_why == "artifact_not_found":
                    continue
                elif del_why == "runtime_error":
                    return _loud(
                        "runtime_error",
                        "delete failed during gc",
                    )
            finally:
                try:
                    os.close(art_fd)
                except OSError:
                    pass

        return {
            "ok": True,
            "deleted": deleted,
            "examined": examined,
            "corrupt": corrupt,
            "skipped_busy": skipped_busy,
            "force": bool(force),
        }
    finally:
        try:
            os.close(root_fd)
        except OSError:
            pass


def apply_result_mode(envelope, result_mode):
    """Maybe replace success text with preview + artifact_ref.

    Invalid result_mode → LOUD runtime_error. Default/None/inline → unchanged
    envelope object (byte-identity when caller passes no opt-in).

    Rehydrated artifact bodies are untrusted prior LLM output.
    """
    if result_mode is None or result_mode == "":
        return envelope
    mode = str(result_mode)
    if mode not in RESULT_MODES:
        return {
            "ok": False,
            "reason": "runtime_error",
            "detail": "invalid result_mode %r (use inline|ref|auto)" % mode,
        }
    if mode == "inline":
        return envelope
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        return envelope
    text = envelope.get("text")
    if not should_spill(mode, text):
        return envelope
    # Masked before the body is stored or cut into a preview.
    text = known_secrets.mask_known(text)

    try:
        ref = create(text)
    except UnicodeEncodeError as exc:
        return {
            "ok": False,
            "reason": "runtime_error",
            "detail": redact_secrets("text is not encodable as utf-8: %s" % exc),
        }
    except ArtifactError as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "detail": exc.detail or exc.reason,
        }
    out = dict(envelope)
    out["text"] = make_preview(text)
    out["artifact_ref"] = {
        "id": ref["id"],
        "sha256": ref["sha256"],
        "text_bytes": ref["text_bytes"],
        "expires": ref["expires"],
    }
    return out
