"""Top-level package so the project can be launched with `python -m src.main`."""

from __future__ import annotations

import importlib
import sys


sys.modules.setdefault("bci_ecog", importlib.import_module(".bci_ecog", __name__))
