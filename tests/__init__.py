# Package marker so `python3 -m unittest discover -s . -p 'test*.py'` collects
# every tests/test_*.py. The suite imports `server`, `omni_cli` and
# `providers` from the repo root, so the root goes on sys.path here; tests
# import each other as `tests.test_x`.
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
