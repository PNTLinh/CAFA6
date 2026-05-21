# Kaggle Notebook Template — CAFA6 Training (GPU T4)

Trước khi chạy:

1. **Settings** → Accelerator → **GPU T4 x1**
2. **Add Data** → dataset `cafa6-data` (tạo bằng `python pack_for_kaggle.py`)
3. **Internet** → ON

---

## Cell 1 — Cài đặt dependencies

> **Không dùng** `dgl==2.1.0+cu118` trên Kaggle — sẽ lỗi `libcudart.so.11.0` vì Kaggle dùng **CUDA 12**.

```python
!pip install -q packaging fair-esm transformers biopython tqdm torchdata

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}")

# Gỡ bản DGL cũ / sai CUDA
!pip uninstall -y dgl

# DGL 2.x + CUDA 12 (Kaggle T4 2025)
!pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.5/cu124/repo.html
```

Nếu cell trên lỗi 404, chạy script tự dò wheel (sau khi clone repo ở Cell 2):

```python
!python /kaggle/working/CAFA6/scripts/install_dgl_kaggle.py
```

Hoặc thử lần lượt (chọn URL khớp output `torch.version.cuda`):

```python
!pip install -q torchdata
import torch
cu = "cu" + torch.version.cuda.replace(".", "")  # ví dụ cu124
!pip uninstall -y dgl
!pip install -q dgl -f https://data.dgl.ai/wheels/torch-{torch.__version__.split('.')[0]}.{torch.__version__.split('.')[1]}/{cu}/repo.html
```

**Kiểm tra:**
```python
import dgl
g = dgl.graph(([0, 1], [1, 2])).to("cuda")
print("DGL", dgl.__version__, "device", g.device)
```

> Sau khi đổi DGL: **Restart session** rồi chạy lại từ Cell 1.

---

## Cell 2 — Clone repo

```python
%cd /kaggle/working
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
!ls
```

---

## Cell 3 — Symlink data từ Kaggle dataset

```python
from pathlib import Path

input_dir = next(Path("/kaggle/input").iterdir())
print(f"Dataset path: {input_dir}")

!ln -sf {input_dir}/divided_data /kaggle/working/CAFA6/divided_data
!ln -sf {input_dir}/proceed_data /kaggle/working/CAFA6/proceed_data

!ls -la /kaggle/working/CAFA6/proceed_data/ppi_graph_global
!ls /kaggle/working/CAFA6/divided_data/
```

---

## Cell 4 — Train với preset T4 (`--kaggle`)

```python
import os
os.environ["DGL_CUDA"] = "1"
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"

!cd /kaggle/working/CAFA6 && python train_Struct2GO2.py -branch mf --kaggle
```

Preset `--kaggle` tự bật: `batch_size=96`, `hid_dim=384`, `num_convs=4`, `epochs=20`, `--amp`, cache PPI.

---

## Cell 5 — Train branch khác

```python
!cd /kaggle/working/CAFA6 && DGL_CUDA=1 DATA_DIR=/kaggle/working/CAFA6 \
    python train_Struct2GO2.py -branch bp --kaggle -dropout 0.1

!cd /kaggle/working/CAFA6 && DGL_CUDA=1 DATA_DIR=/kaggle/working/CAFA6 \
    python train_Struct2GO2.py -branch cc --kaggle
```

---

## Cell 6 — Lưu model + log

```python
!cp -r /kaggle/working/CAFA6/save_models /kaggle/working/
!cp -r /kaggle/working/CAFA6/log /kaggle/working/
!ls -la /kaggle/working/save_models/
```

**Save Version → Save & Run All (Commit)** → tab **Output** để download checkpoint.

---

## Xử lý OOM trên T4

```python
# Giảm batch
!python train_Struct2GO2.py -branch mf --kaggle -batch_size 64

# Model nhỏ hơn
!python train_Struct2GO2.py -branch mf -batch_size 64 --amp \
    -hid_dim 256 -num_convs 3 -epochs 20
```

---

## Lưu ý

- Session Kaggle tối đa **9 giờ** — đủ train cả 3 branch với `--kaggle`.
- Checkpoint: `save_models/bestmodel_{branch}_{batch}_{lr}_{dropout}.pkl`
- Chi tiết pipeline data local: xem `README.md` mục 3 và 8.
