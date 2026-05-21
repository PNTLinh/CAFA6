# Kaggle Notebook — CAFA6 (GPU T4)

**Settings:** GPU T4 x1 · Internet ON · Add Data → `cafa6-data`

> Lỗi `torchdata.datapipes` / `libcudart.so.11.0` xảy ra vì:
> 1. Cài nhầm `dgl+cu118` (cần CUDA 12)
> 2. Shim trong notebook **không** áp dụng cho `python train_Struct2GO2.py` (process riêng) — code repo đã fix trong `model/dgl_compat.py`

---

## Cell 1 — Clone repo (chạy đầu tiên)

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
!git pull
!ls scripts/
```

---

## Cell 2 — Cài DGL + kiểm tra GPU

```python
!pip install -q torchdata==0.7.1 packaging fair-esm transformers biopython tqdm
!python /kaggle/working/CAFA6/scripts/dgl_kaggle_bootstrap.py
```

Phải in ra dòng `OK  DGL ... device=cuda:0`. Nếu lỗi, **Restart session** và chạy lại Cell 1 → Cell 2.

---

## Cell 3 — Symlink dataset

```python
from pathlib import Path

input_dir = next(Path("/kaggle/input").iterdir())
print("Dataset:", input_dir)

!ln -sf {input_dir}/divided_data /kaggle/working/CAFA6/divided_data
!ln -sf {input_dir}/proceed_data /kaggle/working/CAFA6/proceed_data

assert Path("/kaggle/working/CAFA6/proceed_data/ppi_graph_global").exists()
assert Path("/kaggle/working/CAFA6/divided_data/mf_train_dataset").exists()
print("Data OK")
```

---

## Cell 4 — Train MF (`--kaggle`)

```python
import os
os.environ["DGL_CUDA"] = "1"
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"

!cd /kaggle/working/CAFA6 && python train_Struct2GO2.py -branch mf --kaggle
```

Train script tự gọi `model/dgl_compat.py` — **không cần** import dgl trong notebook trước đó.

---

## Cell 5 — Train CC / BP (tùy chọn)

```python
!cd /kaggle/working/CAFA6 && DGL_CUDA=1 DATA_DIR=/kaggle/working/CAFA6 \
    python train_Struct2GO2.py -branch cc --kaggle

!cd /kaggle/working/CAFA6 && DGL_CUDA=1 DATA_DIR=/kaggle/working/CAFA6 \
    python train_Struct2GO2.py -branch bp --kaggle -dropout 0.1
```

---

## Cell 6 — Lưu output

```python
!cp -r /kaggle/working/CAFA6/save_models /kaggle/working/
!cp -r /kaggle/working/CAFA6/log /kaggle/working/
!ls /kaggle/working/save_models/
```

---

## OOM trên T4

```python
!cd /kaggle/working/CAFA6 && python train_Struct2GO2.py -branch mf --kaggle -batch_size 64
```

---

## Checklist nhanh

| Bước | Đúng? |
|---|---|
| Clone repo **trước** khi chạy bootstrap | Cell 1 |
| `dgl_kaggle_bootstrap.py` in `OK ... cuda:0` | Cell 2 |
| Symlink thấy `ppi_graph_global` | Cell 3 |
| `git pull` có `model/dgl_compat.py` | Cell 1 |
