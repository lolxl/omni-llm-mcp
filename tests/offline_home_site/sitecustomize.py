"""Runs in every Python child of the offline run (see tests/offline_home.py)."""
import os

if os.environ.get("OMNI_OFFLINE_HOME"):
    try:
        from tests import offline_home
        offline_home.install_in_child()
    except Exception as exc:  # the child still starts, but the run must fail
        _log = os.environ.get("OMNI_OFFLINE_HITLOG")
        if _log:
            import json
            try:
                with open(_log, "a", encoding="utf-8") as _fh:
                    _fh.write(json.dumps([
                        os.environ.get("OMNI_OFFLINE_CURRENT"), "arm_failed",
                        "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                        os.getpid()]) + "\n")
            except OSError as _exc:
                import sys
                print("offline_home: child could not arm and cannot report it "
                      "(%s)" % type(_exc).__name__, file=sys.stderr)
