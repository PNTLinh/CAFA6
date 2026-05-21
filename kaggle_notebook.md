# Kaggle Notebook — CAFA6 (GPU T4)

**Settings:** GPU T4 x1 · Internet ON · Add Data → dataset từ `pack_for_kaggle.py`

---

## Bước 0 — Restart session

Nếu từng chạy `torchdata==0.7.1` hoặc cell shim cũ: **Session → Restart session** (bắt buộc).

---

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

Nếu GitHub chưa có bản mới, copy thư mục `model/dgl_patch.py` từ máy local vào `/kaggle/working/CAFA6/model/`.

---

## Cell 2 — Cài + sửa DGL (quan trọng)

```python
!pip install -q packaging fair-esm transformers biopython tqdm
!pip uninstall -y dgl torchdata
!pip install -q dgl==2.1.0
!python /kaggle/working/CAFA6/model/dgl_patch.py
```

Phải in: `[dgl_patch] ... graphbolt/base.py` và `OK DGL 2.1.0`.

**Không** cài `torchdata==0.7.1`. **Không** chạy cell shim cũ trong notebook.

---

## Cell 3 — Link data

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
```

Nếu lỗi: `!find /kaggle/input -name ppi_graph_global`

---

## Cell 4 — Train

```python
import os
os.environ["DGL_CUDA"] = "1"
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
!cd /kaggle/working/CAFA6 && python train_Struct2GO2.py -branch mf --kaggle
```

`train_Struct2GO2.py` tự gọi `ensure_dgl_importable()` — train chạy subprocess riêng, không dùng shim trong notebook.

---

## Cell dự phòng (nếu Cell 2 vẫn lỗi)

Chạy **trong notebook** (cùng kernel):

```python
import sys, site, shutil
from pathlib import Path
sys.path.insert(0, "/kaggle/working/CAFA6")
from model.dgl_patch import patch_dgl_on_disk, ensure_dgl_importable
print("patched files:", patch_dgl_on_disk())
ensure_dgl_importable()
```
