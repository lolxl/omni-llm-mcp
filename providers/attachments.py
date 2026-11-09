# no-opencc
"""Stage job-lane image attachments as immutable workdir-local copies.

Owner-approved S2 (validated-copy): every backend consumes a copy under
``<workdir>/.omni-attach/<job_id>/`` written from bytes read through the
shared O_NOFOLLOW source-roots fence — never re-open the caller's path.

Staged copies use synthesized safe names (``img-<index><ext>``), never the
source basename, so hostile filenames cannot inject prompt lines.
"""
import os
import re
import tempfile

from providers.config import (
    AGENT_IMAGE_MAX_BYTES,
    AGENT_IMAGES_TOTAL_MAX_BYTES,
    FILES_IMAGE_MIME,
)
from providers.errors import redact_secrets
from providers.source_load import open_binary_under_roots

_EXCLUDE_LINE = ".omni-attach/"
_ATTACH_DIRNAME = ".omni-attach"
# Source extension kept only when it is a short alphanumeric suffix.
_SAFE_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def _attach_block(paths):
    """Prompt trailer listing staged copy paths (one per line)."""
    return "\n\n[attached images]\n" + "\n".join(paths)


def _refusal(detail):
    return {
        "ok": False,
        "reason": "unsupported_files",
        "detail": str(detail)[:300],
    }


def _ensure_git_exclude(workdir):
    """Append ``.omni-attach/`` to workdir's .git/info/exclude if needed.

    Never touches .gitignore or any tracked file. Non-git workdir: skip quietly.
    """
    git_meta = os.path.join(workdir, ".git")
    info_dir = None
    if os.path.isdir(git_meta):
        info_dir = os.path.join(git_meta, "info")
    elif os.path.isfile(git_meta):
        # Linked worktree: .git is a file pointing at the real gitdir.
        try:
            with open(git_meta, encoding="utf-8") as handle:
                first = (handle.readline() or "").strip()
        except OSError:
            return
        if first.lower().startswith("gitdir:"):
            gitdir = first.split(":", 1)[1].strip()
            if not os.path.isabs(gitdir):
                gitdir = os.path.normpath(os.path.join(workdir, gitdir))
            info_dir = os.path.join(gitdir, "info")
    if info_dir is None:
        return
    try:
        os.makedirs(info_dir, mode=0o700, exist_ok=True)
        exclude_path = os.path.join(info_dir, "exclude")
        existing = ""
        if os.path.isfile(exclude_path):
            with open(exclude_path, encoding="utf-8") as handle:
                existing = handle.read()
        lines = existing.splitlines()
        if _EXCLUDE_LINE in lines:
            return
        with open(exclude_path, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(_EXCLUDE_LINE + "\n")
    except OSError:
        # Exclude is best-effort hygiene; staging still succeeds without it.
        return


def _safe_copy_name(index, source_path):
    """Synthesize a collision-free staged basename; never use source basename.

    ``img-<index><ext>`` where index is 1-based staging position and <ext> is
    the source extension only if it matches ``^\\.[A-Za-z0-9]{1,8}$``.
    """
    ext = os.path.splitext(source_path)[1]
    if not _SAFE_EXT_RE.match(ext):
        ext = ""
    return "img-%d%s" % (index, ext)


def _atomic_write(dest_path, data):
    """Write *data* to *dest_path* via temp + os.replace; mode 0o600."""
    directory = os.path.dirname(dest_path)
    fd, temp_path = tempfile.mkstemp(
        prefix=".omni-attach-", suffix=".tmp", dir=directory)
    open_fd = fd
    try:
        with os.fdopen(fd, "wb") as handle:
            open_fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, dest_path)
        os.chmod(dest_path, 0o600)
    finally:
        if open_fd is not None:
            try:
                os.close(open_fd)
            except OSError:
                pass
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _cleanup_partial(dest_dir, staged):
    """Best-effort: unlink staged copies and remove the job attach dir."""
    for path in staged:
        try:
            os.unlink(path)
        except OSError:
            pass
    try:
        # Clear any leftover temps so rmdir can succeed.
        if os.path.isdir(dest_dir):
            for name in os.listdir(dest_dir):
                try:
                    os.unlink(os.path.join(dest_dir, name))
                except OSError:
                    pass
        os.rmdir(dest_dir)
    except OSError:
        pass


def stage_images(workdir, job_id, paths):
    """Copy allowlisted image paths into the job attach dir.

    Returns ``(staged_paths, None)`` on success or ``([], refusal_envelope)``
    on fence/cap refusal. Pure w.r.t. job.json — caller persists the list.

    Each source is read once via ``open_binary_under_roots`` (never re-opened).
    Copies land at ``<realpath(workdir)>/.omni-attach/<job_id>/img-<n><ext>``
    with synthesized safe names (source basename never used).

    Extensions are restricted to ``FILES_IMAGE_MIME`` — the same table the
    consult lane fences on — so ``--image`` cannot smuggle arbitrary binaries
    into the workdir. A mismatch is a LOUD ``unsupported_files`` refusal.
    """
    paths = list(paths or [])
    if not paths:
        return [], None

    workdir_real = os.path.realpath(str(workdir))
    job_id = str(job_id)
    attach_root = os.path.join(workdir_real, _ATTACH_DIRNAME)
    dest_dir = os.path.join(attach_root, job_id)

    staged = []
    total_bytes = 0

    try:
        os.makedirs(dest_dir, mode=0o700, exist_ok=True)
        # Tighten modes even if parents pre-existed with looser umask.
        try:
            os.chmod(attach_root, 0o700)
        except OSError:
            pass
        try:
            os.chmod(dest_dir, 0o700)
        except OSError:
            pass
    except OSError as exc:
        return [], _refusal("attach dir unwritable: %s" % redact_secrets(str(exc))[:200])

    # Exclude before any file is staged so a first-file refusal still hides
    # the attach dir from git status (and leaves no untracked pollution).
    _ensure_git_exclude(workdir_real)

    for index, path in enumerate(paths, start=1):
        path = os.fspath(path)
        # Type fence FIRST, so a non-image is reported as a type problem rather
        # than as a path/allowlist one (same ordering as the consult lane's
        # grok filter). `--image` is documented as an image-only flag on every
        # backend; without this check any binary was copied in unfiltered.
        if os.path.splitext(path)[1].lower() not in FILES_IMAGE_MIME:
            _cleanup_partial(dest_dir, staged)
            return [], _refusal(
                "--image accepts %s; got %r"
                % ("/".join(sorted(FILES_IMAGE_MIME)), os.path.basename(path)))

        data, err = open_binary_under_roots(path, AGENT_IMAGE_MAX_BYTES)
        if err is not None:
            _cleanup_partial(dest_dir, staged)
            filename = os.path.basename(path) or str(path)
            err_detail = err.get("detail") or err.get("error_code") or "read failed"
            return [], _refusal("%s (%s)" % (err_detail, filename))

        total_bytes += len(data)
        if total_bytes > AGENT_IMAGES_TOTAL_MAX_BYTES:
            _cleanup_partial(dest_dir, staged)
            return [], _refusal(
                "attached images exceed total cap of %d bytes"
                % AGENT_IMAGES_TOTAL_MAX_BYTES)

        safe_name = _safe_copy_name(index, path)
        dest_path = os.path.join(dest_dir, safe_name)
        try:
            _atomic_write(dest_path, data)
        except OSError as exc:
            _cleanup_partial(dest_dir, staged)
            return [], _refusal(
                "failed to stage image %d: %s" % (index, redact_secrets(str(exc))[:200]))
        staged.append(dest_path)

    return staged, None
