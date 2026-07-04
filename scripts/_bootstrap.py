"""Shared bootstrap: put ``src/`` on the path and force a headless MPL backend."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ROOT = _ROOT
