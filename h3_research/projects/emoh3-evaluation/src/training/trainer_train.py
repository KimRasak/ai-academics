#!/usr/bin/env python3
"""Compatibility entry point for the shared H3 trainer."""
from pathlib import Path
import runpy


SHARED = Path(__file__).resolve().parents[4] / "shared" / "h3-training" / "train.py"
runpy.run_path(str(SHARED), run_name="__main__")
