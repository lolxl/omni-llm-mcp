# no-opencc
"""Terminal result record of one ``omni dispatch-exec`` invocation.

The caller names the file (``--result-file``) and the invocation (``--invocation-id``).
omni writes exactly one record, once, after the run has ended and been torn down: a
temp file in the same directory (O_EXCL, 0600) holds the full JSON, is flushed and
fsynced, then replaces the final name; the directory is fsynced where the platform
allows. A reader therefore sees either no file or the complete terminal record, never a
provisional or partial one. A missing or malformed record means omni itself did not
finish: the caller must not retry on its own.

Fields (exactly these keys):
  version         1
  invocation_id   the 32 lowercase hex characters the caller passed
  outcome         vendor_exit | stall | wall_timeout | signal | refused_before_spawn
                  | supervisor_failure
  detail          none | arguments | containment | credential_unproven | prompt_mismatch
                  | launch_unavailable | parent_died | start_unknown | internal
  spawned         false only when omni never created a vendor-capable process (a canary
                  probe is not one); true only when omni itself spawned the vendor
                  directly (never today: the child is the host-exec wrapper); otherwise
                  null, because the wrapper may or may not have started the vendor
  vendor_rc       0-255 for vendor_exit when the vendor exited by itself, else null
  vendor_signal   the signal that ended the vendor for vendor_exit, else null
  signal          the HUP/INT/TERM omni received, else null
  exit_code       omni's own exit status
  elapsed_ms      monotonic, from entry through teardown
  marker          a ``type: "result"`` line was seen in the vendor output
"""
import json
import os
import re
import secrets

VERSION = 1
OUTCOMES = ("vendor_exit", "stall", "wall_timeout", "signal", "refused_before_spawn",
            "supervisor_failure")
DETAILS = ("none", "arguments", "containment", "credential_unproven", "prompt_mismatch",
           "launch_unavailable", "parent_died", "start_unknown", "internal")
_ID = re.compile(r"[0-9a-f]{32}")


class PublishError(Exception):
    """The record could not be published; nothing readable was left at the final name."""


def valid_invocation_id(text):
    return isinstance(text, str) and _ID.fullmatch(text) is not None


def result_file_problem(path):
    """Why ``path`` cannot receive the record, or None: absolute, parent exists, not yet there."""
    if not isinstance(path, str) or not path or "\0" in path:
        return "must be an absolute path"
    if not os.path.isabs(path) or not os.path.basename(path):
        return "must be an absolute file path"
    if not os.path.isdir(os.path.dirname(path)):
        return "parent is not a directory"
    if os.path.lexists(path):
        return "already exists"
    return None


def _int_or_none(value):
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def build(invocation_id, outcome, detail, spawned, vendor_rc, vendor_signal, signum,
          exit_code, elapsed_ms, marker):
    """The record dict, key order as documented. Raises ValueError on an unusable value."""
    if not valid_invocation_id(invocation_id):
        raise ValueError("invocation_id")
    if outcome not in OUTCOMES or detail not in DETAILS:
        raise ValueError("outcome/detail")
    if spawned not in (True, False, None):
        raise ValueError("spawned")
    if vendor_rc is not None and not (isinstance(vendor_rc, int) and 0 <= vendor_rc <= 255):
        raise ValueError("vendor_rc")
    if not (_int_or_none(vendor_signal) and _int_or_none(signum)):
        raise ValueError("signal")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("exit_code")
    if not isinstance(elapsed_ms, int) or isinstance(elapsed_ms, bool) or elapsed_ms < 0:
        raise ValueError("elapsed_ms")
    return {
        "version": VERSION,
        "invocation_id": invocation_id,
        "outcome": outcome,
        "detail": detail,
        "spawned": spawned,
        "vendor_rc": vendor_rc,
        "vendor_signal": vendor_signal,
        "signal": signum,
        "exit_code": exit_code,
        "elapsed_ms": elapsed_ms,
        "marker": bool(marker),
    }


def _fsync_dir(directory):
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # not supported here: the rename itself already happened
    finally:
        os.close(fd)


def publish(path, record):
    """Write ``record`` to ``path`` atomically (temp + fsync + replace); PublishError on failure."""
    directory = os.path.dirname(path)
    tmp = os.path.join(directory, ".dispatch-result.%s.tmp" % secrets.token_hex(8))
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    made = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                     0o600)
        made = True
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise FileExistsError(path)  # one terminal record: never replace another file
        os.replace(tmp, path)
        made = False
    except OSError as exc:
        raise PublishError(str(exc)) from exc
    finally:
        if made:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    _fsync_dir(directory)
