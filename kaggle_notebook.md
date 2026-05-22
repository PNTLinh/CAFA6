# Kaggle — CAFA6

Notebook này dành cho **Kaggle T4** và ưu tiên dùng script có sẵn trong repo để tránh lỗi thư viện `torchdata`/`dgl`.

## Trước khi chạy

- Bật **GPU T4** trong Kaggle Notebook.
- Bật **Internet**.
- Nếu trước đó bạn đã thử cài `torchdata` hoặc `dgl` thủ công, hãy **Restart session** trước khi chạy lại notebook này.

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

## Cell 2 — Cài dependencies và DGL

```python
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas scipy networkx requests psutil
!python /kaggle/working/CAFA6/scripts/install_dgl_kaggle.py
```

Nếu cell này báo lỗi hoặc torch/dgl bị đổi phiên bản ngoài ý muốn, restart session rồi chạy lại từ đầu.

## Cell 3 — Kiểm tra CUDA + DGL

```python
import torch
from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

Phải thấy thiết bị dạng `cuda:0`.

## Cell 4 — Nối dữ liệu Kaggle

Upload dataset đã pack bằng `pack_for_kaggle.py`, sau đó chạy:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
```

Script này sẽ tự tìm `proceed_data/ppi_graph_global` và `divided_data/*` trong `/kaggle/input` rồi symlink vào `/kaggle/working/CAFA6`.

## Cell 5 — Train

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1
!python /kaggle/working/CAFA6/train_Struct2GO2.py -branch mf --kaggle
```

Nếu T4 hết VRAM, giảm batch size:

```python
!python /kaggle/working/CAFA6/train_Struct2GO2.py -branch mf --kaggle -batch_size 64
```

## Ghi chú nhanh

- Preset `--kaggle` đã bật `amp`, cache PPI và cấu hình phù hợp T4.
- Không cần sửa path trong code; chỉ cần đặt `DATA_DIR=/kaggle/working/CAFA6`.
- Nếu bạn muốn chạy từ đầu bằng notebook Kaggle khác, thứ tự an toàn là: clone -> install_dgl_kaggle.py -> verify -> kaggle_link_data.py -> train.
