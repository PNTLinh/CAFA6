# Kaggle notebook — copy từng cell

Bật **GPU T4** + **Internet**. Nếu lỗi DGL/numpy → **Restart session** → chạy lại từ Cell 1.

Gắn dataset `cafa6-data` (từ `pack_for_kaggle.py`) vào notebook.

---

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

*(Đã clone rồi, chỉ cập nhật code: `%cd /kaggle/working/CAFA6` rồi `!git pull`)*

---

## Cell 2 — Cài thư viện + DGL CUDA

```python
!pip install -q "numpy>=1.26,<2.4" "scipy>=1.11,<1.16"
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas networkx requests psutil
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

---

## Cell 3 — Kiểm tra GPU

```python
import torch
import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

---

## Cell 4 — Nối dữ liệu

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
!ls /kaggle/working/CAFA6/proceed_data/ppi_graph_global
!ls /kaggle/working/CAFA6/divided_data/*_train_dataset
```

---

## Cell 5 — Train CC → MF → BP, lưu từng nhánh, cập nhật zip

~1–2 giờ tổng (BP chậm nhất). Sau mỗi nhánh zip được ghi lại tại `/kaggle/working/cafa6_output.zip`.

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py
```

**MF đã train xong** (chỉ chạy cc + bp, vẫn copy mf cũ vào zip nếu có):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --branches cc mf bp --skip mf
```

**OOM** → thêm `-batch_size 64`:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py -batch_size 64
```

**Chỉ train, không eval** (nhanh hơn):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --no-eval
```

---

## Cell 6 — Kiểm tra Output

```python
!ls -lh /kaggle/working/cafa6_output.zip
!ls -lh /kaggle/working/log/
!ls -lh /kaggle/working/save_models/
!ls -lh /kaggle/working/test_result/ 2>/dev/null || echo "(chưa eval)"
```

---

## Cell 7 — Tải zip về máy local

```python
from IPython.display import FileLink, display

display(FileLink("/kaggle/working/cafa6_output.zip"))
```

Bấm link **cafa6_output.zip** trong output cell, hoặc **Save Version → Save & Run All** rồi tải từ tab **Output** bên phải.

Nội dung zip:

```
log/mf.log  log/cc.log  log/bp.log
save_models/bestmodel_*.pkl  final_*.pkl
test_result/   (nếu Cell 5 có chạy eval)
```
