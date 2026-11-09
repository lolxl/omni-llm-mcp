# no-opencc
"""Minimal log tailer for zellij watch panes.

Exits as soon as the done-sentinel file appears (zero hold). Hard-caps on max age
so a stuck MCP cannot leave a pane forever. No third-party deps.
"""
import os
import sys
import time


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        sys.stderr.write("usage: watch_viewer.py <log> <done> <max_s>\n")
        return 2
    log_path, done_path, max_s_raw = argv[0], argv[1], argv[2]
    try:
        max_s = float(max_s_raw)
    except (TypeError, ValueError):
        max_s = 300.0
    max_s = max(1.0, max_s)
    deadline = time.time() + max_s
    pos = 0
    while time.time() < deadline:
        try:
            if os.path.exists(log_path):
                with open(log_path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    if chunk:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        pos += len(chunk)
        except OSError:
            pass
        if os.path.exists(done_path):
            try:
                if os.path.exists(log_path):
                    with open(log_path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        if chunk:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
            except OSError:
                pass
            return 0
        time.sleep(0.05)
    return 1


if __name__ == "__main__":
    sys.exit(main())
