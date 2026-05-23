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


def _clear_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _copy_or_link(path: Path, dst: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(path, dst)
    else:
        dst.symlink_to(path, target_is_directory=path.is_dir())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Link or repair CAFA6 Kaggle data")
    parser.add_argument(
        "--copy-splits",
        nargs="*",
        default=[],
        choices=["mf", "cc", "bp", "all"],
        help="Make selected divided_data branches writable local copies instead of symlinks",
    )
    args = parser.parse_args()

    work = Path(os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    data_root = find_cafa6_data_root(Path("/kaggle/input"))
    print("Data root:", data_root)

    copy_branch_names = set(args.copy_splits)
    copy_all_splits = "all" in copy_branch_names

    # proceed_data stays linked: it is read-only for train/eval.
    src = data_root / "proceed_data"
    dst = work / "proceed_data"
    dst.parent.mkdir(parents=True, exist_ok=True)
    _clear_path(dst)
    dst.symlink_to(src, target_is_directory=True)
    print(f"  {dst} -> {src}")

    src = data_root / "divided_data"
    dst = work / "divided_data"
    dst.parent.mkdir(parents=True, exist_ok=True)
    _clear_path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if copy_all_splits:
        copy_branch_names.update({"mf", "cc", "bp"})

    for child in src.iterdir():
        target = dst / child.name
        if any(child.name.startswith(f"{branch}_") for branch in copy_branch_names):
            _copy_or_link(child, target, copy=True)
        else:
            _copy_or_link(child, target, copy=False)
    print(f"  {dst} <- {src} (writable branches: {sorted(copy_branch_names) or 'none'})")

    assert (work / "proceed_data/ppi_graph_global").exists()
    assert (work / "divided_data/mf_train_dataset").exists()
    print("Data OK")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
