"""Symlink CAFA6 data from /kaggle/input (handles nested zip layout)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_cafa6_data_root(input_root: Path) -> Path:
    def has_raw_split_sources(root: Path) -> bool:
        proc = root / "proceed_data"
        required = (
            proc / "emb_graph_mf",
            proc / "emb_seq_feature_mf",
            proc / "emb_label_mf",
            proc / "ppi_graph_global",
        )
        return all(path.exists() for path in required)

    local_candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for candidate in local_candidates:
        if has_raw_split_sources(candidate):
            return candidate

    for ppi in input_root.rglob("ppi_graph_global"):
        if not ppi.is_file():
            continue
        root = ppi.parent.parent
        if (root / "divided_data" / "mf_train_dataset").exists() and has_raw_split_sources(root):
            return root

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


def _validate_pickle(path: Path, min_bytes: int = 1024) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"{path.name} too small ({size} B) — truncated upload or bad copy")
    import pickle

    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet
    with open(path, "rb") as handle:
        pickle.load(handle)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Link or repair CAFA6 Kaggle data")
    parser.add_argument(
        "--copy-splits",
        nargs="*",
        default=[],
        choices=["mf", "cc", "bp", "all"],
        help="Copy divided_data vào /kaggle/working (tốn ~30GB). Mặc định: symlink từ input.",
    )
    args = parser.parse_args()

    work = Path(os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    data_root = find_cafa6_data_root(Path("/kaggle/input"))
    print("Data root:", data_root)

    proc = data_root / "proceed_data"
    emb_mf = proc / "emb_graph_mf"
    if not emb_mf.exists():
        print(
            "[INFO] proceed_data không có emb_graph_* — KHÔNG chạy divide_data.py trên Kaggle. "
            "Dùng divided_data/*_train_dataset có sẵn trong dataset."
        )

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
    for branch in ("mf", "cc", "bp"):
        for split in ("train", "valid"):
            p = work / "divided_data" / f"{branch}_{split}_dataset"
            if not p.exists():
                print(f"[WARN] missing {p.name}")
                continue
            try:
                _validate_pickle(p)
                print(f"  OK {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
            except Exception as exc:
                raise RuntimeError(
                    f"Pickle invalid: {p}. Re-upload dataset or run with --copy-splits all "
                    f"after fixing source under {data_root / 'divided_data'}"
                ) from exc
    print("Data OK")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
