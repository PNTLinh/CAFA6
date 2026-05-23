"""
pack_for_kaggle.py
==================
Đóng gói các file cần thiết để upload lên Kaggle thành 1 zip.

Output: kaggle_data.zip với cấu trúc:
  divided_data/
      mf_train_dataset
      mf_valid_dataset
      bp_train_dataset (nếu có)
      bp_valid_dataset (nếu có)
      cc_train_dataset (nếu có)
      cc_valid_dataset (nếu có)
  proceed_data/
      label_mf_network
      label_bp_network (nếu có)
      label_cc_network (nếu có)
      ppi_graph_global
      ppi_protein_index

Sau khi tạo xong:
  1. Vào https://www.kaggle.com/datasets → New Dataset
  2. Upload kaggle_data.zip
  3. Đặt tên dataset, ví dụ "cafa6-data"
"""
import os
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIVIDED_DIR = BASE_DIR / "divided_data"
PROC_DIR    = BASE_DIR / "proceed_data"
OUT_ZIP     = BASE_DIR / "kaggle_data.zip"

# Files cần thiết để train
REQUIRED_FILES = []
OPTIONAL_FILES = []

for branch in ("mf", "bp", "cc"):
    train_path = DIVIDED_DIR / f"{branch}_train_dataset"
    valid_path = DIVIDED_DIR / f"{branch}_valid_dataset"
    test_path = DIVIDED_DIR / f"{branch}_test_dataset"
    label_path = PROC_DIR / f"label_{branch}_network"
    vocab_path = PROC_DIR / f"label_vocab_{branch}.json"
    if train_path.exists() and valid_path.exists():
        REQUIRED_FILES.extend([train_path, valid_path])
        if test_path.exists():
            REQUIRED_FILES.append(test_path)
        if label_path.exists():
            REQUIRED_FILES.append(label_path)
        else:
            print(f"[WARN] Thiếu {label_path} — training {branch} sẽ fail")
        if vocab_path.exists():
            REQUIRED_FILES.append(vocab_path)
        acs_path = PROC_DIR / f"human_{branch.upper()}_ACS.json"
        if acs_path.exists():
            REQUIRED_FILES.append(acs_path)

# Cần cho cả 3 branch
ppi_graph = PROC_DIR / "ppi_graph_global"
ppi_index = PROC_DIR / "ppi_protein_index"
for f in (ppi_graph, ppi_index):
    if f.exists():
        REQUIRED_FILES.append(f)
    else:
        print(f"[WARN] Thiếu {f} — PPI branch sẽ không hoạt động")


def total_size_mb(paths):
    return sum(p.stat().st_size for p in paths) / (1024 * 1024)


def main():
    if not REQUIRED_FILES:
        print("ERROR: Không tìm thấy file dataset nào. Đã chạy data_processing/divide_data.py chưa?")
        sys.exit(1)

    total_mb = total_size_mb(REQUIRED_FILES)
    print(f"\nTotal size: {total_mb:,.1f} MB ({total_mb/1024:.2f} GB)")
    print(f"File count: {len(REQUIRED_FILES)}\n")

    print("Files to pack:")
    for f in REQUIRED_FILES:
        size_mb = f.stat().st_size / (1024 * 1024)
        rel = f.relative_to(BASE_DIR)
        print(f"  {size_mb:>8.1f} MB  {rel}")
    print()

    if total_mb > 20 * 1024:
        print("[WARN] Tổng > 20 GB — Kaggle giới hạn 20GB/dataset. Cân nhắc tách thành nhiều dataset.")

    # Tạo zip với compression nhẹ (pickle đã khá compact)
    print(f"Creating {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in REQUIRED_FILES:
            arcname = f.relative_to(BASE_DIR)
            print(f"  + {arcname}")
            zf.write(f, arcname=str(arcname))

    out_size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nDone: {OUT_ZIP} ({out_size_mb:,.1f} MB)")
    print("\nNext steps:")
    print("  1. https://www.kaggle.com/datasets -> New Dataset")
    print("  2. Upload kaggle_data.zip")
    print("  3. Name dataset (e.g. cafa6-data)")
    print("  4. New Notebook -> Add Data -> select dataset")
    print("  5. Copy cells from kaggle_notebook.md")


if __name__ == "__main__":
    main()
