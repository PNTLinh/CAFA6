"""Wrapper — run scripts/kaggle_fix_dgl.py logic."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def ensure_dgl_importable(verbose: bool = True) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_fix_dgl.py"
    runpy.run_path(str(script), run_name="__main__")


def main() -> None:
    ensure_dgl_importable()


if __name__ == "__main__":
    main()
