#!/usr/bin/env python3
# no-opencc
"""omni-llm-mcp MCP stdio server entry point (plugins and clones launch this file).

The server itself lives in ``providers/mcp_server.py`` so the pipx-installed
``omni mcp`` can run it too. This file is only a shim:

* run as a script (``python3 server.py``) it calls ``providers.mcp_server.main()``;
* imported (``import server``) it IS ``providers.mcp_server`` -- the same module
  object, so tests that patch ``server._send`` / ``server._call_tool`` / ... patch
  the code that actually runs.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import providers.mcp_server as _impl  # noqa: E402

if __name__ == "__main__":
    sys.exit(_impl.main())
else:
    sys.modules[__name__] = _impl
