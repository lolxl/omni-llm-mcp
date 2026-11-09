#!/usr/bin/env python3
# no-opencc
"""Deterministic fixture scrubber — run after re-capturing any *_stream*.jsonl.

Captured CLI streams leak the capturing machine's environment (home paths,
plugin/tool inventories, socket paths). This keeps envelope STRUCTURE (the
contract) and scrubs inventory VALUES. Idempotent.
"""
import glob
import json
import os
import sys

HOME = os.path.expanduser("~")

SCRUB_LIST_KEYS = ("tools", "slash_commands", "skills", "agents", "plugins",
                   "capabilities")
SCRUB_STR_KEYS = ("messaging_socket_path", "output_style", "apiKeySource")


def scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in SCRUB_LIST_KEYS and isinstance(v, list):
                out[k] = ["scrubbed"]
            elif k == "mcp_servers" and isinstance(v, list):
                out[k] = [{"name": "scrubbed", "status": "connected"}]
            elif k == "memory_paths" and isinstance(v, dict):
                out[k] = {"auto": "/Users/user/.claude/memory/"}
            elif k == "hook_name":
                out[k] = "scrubbed-hook"
            elif k in SCRUB_STR_KEYS and isinstance(v, str):
                out[k] = "scrubbed"
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, str):
        return obj.replace(HOME, "/Users/user")
    return obj


def main():
    paths = sys.argv[1:] or glob.glob("*_stream*.jsonl")
    for path in paths:
        lines = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.dumps(scrub(json.loads(line)),
                                        ensure_ascii=False, sort_keys=True))
            except ValueError:
                lines.append(line.replace(HOME, "/Users/user"))
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(path, len(lines), "lines scrubbed")


if __name__ == "__main__":
    main()
