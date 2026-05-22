"""Symlink CAFA6 data from /kaggle/input (handles nested zip layout)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_cafa6_data_root(input_root: Path) -> Path:
    for ppi in input_root.rglob("ppi_graph_global"):
        if not ppi.is_file():
            continue
        root = ppi.parent.parent
        if (root / "divided_data" / "mf_train_dataset").exists():
            return root
    raise FileNotFoundError(
        "No proceed_data/ppi_graph_global + divided_data/mf_train_dataset under "
        f"{input_root}. Upload kaggle_data.zip from pack_for_kaggle.py."
    )


def main() -> None:
    work = Path(os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    data_root = find_cafa6_data_root(Path("/kaggle/input"))
    print("Data root:", data_root)

    def clear_path(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
            return
        shutil.rmtree(path)

    for name in ("divided_data", "proceed_data"):
        src = data_root / name
        dst = work / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        clear_path(dst)
        dst.symlink_to(src, target_is_directory=True)
        print(f"  {dst} -> {src}")

    assert (work / "proceed_data/ppi_graph_global").exists()
    assert (work / "divided_data/mf_train_dataset").exists()
    print("Data OK")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
