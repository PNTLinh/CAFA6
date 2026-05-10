"""
seq2vec.py — ESM-2 sequence embedding
======================================
Thay thế SeqVec (allennlp) bằng ESM-2 của Meta AI.
  - Model: esm2_t33_650M_UR50D  →  embedding 1280-dim
  - Cài đặt: pip install fair-esm

Input : file FASTA (UniProt format hoặc bất kỳ FASTA nào)
Output: dict pickle {UniProtID → ndarray(1280,)}

Cách lấy UniProtID từ FASTA header:
  >sp|P12345|PROT_HUMAN ... → P12345
  >P12345 ...               → P12345
  >P12345|...               → P12345
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ── Đọc FASTA ────────────────────────────────────────────────────────────────
def read_fasta(fasta_path: str) -> dict[str, str]:
    """Trả về {protein_id: sequence}."""
    sequences: dict[str, str] = {}
    current_id = None
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                # UniProt format: sp|P12345|NAME hoặc tr|P12345|NAME
                parts = header.split("|")
                if len(parts) >= 2 and parts[0] in ("sp", "tr"):
                    current_id = parts[1]
                else:
                    current_id = parts[0].split()[0]
                sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += line
    return sequences


# ── Tạo embedding bằng ESM-2 ─────────────────────────────────────────────────
def embed_with_esm2(sequences: dict[str, str], batch_size: int = 8,
                    device: str = "cuda") -> dict[str, np.ndarray]:
    """
    Encode từng protein thành vector 1280-dim (mean pooling qua chiều dài).
    Protein dài hơn 1022 residue sẽ bị cắt bớt (giới hạn ESM-2).
    """
    try:
        import esm
    except ImportError:
        raise ImportError(
            "Chưa cài fair-esm. Chạy: pip install fair-esm"
        )

    print("Đang tải model ESM-2 (esm2_t33_650M_UR50D, ~1.3GB)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    MAX_LEN = 1022  # giới hạn context của ESM-2
    EMB_DIM = 1280

    items = list(sequences.items())
    emb_dict: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(items), batch_size), desc="ESM-2 embedding"):
        batch = items[i: i + batch_size]
        # Cắt sequence quá dài
        batch_data = [(pid, seq[:MAX_LEN]) for pid, seq in batch]

        try:
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[33], return_contacts=False)

            representations = results["representations"][33]  # (B, L+2, 1280)

            for j, (pid, seq) in enumerate(batch_data):
                seq_len = min(len(seq), MAX_LEN)
                # Mean pool qua các token của sequence (bỏ BOS và EOS)
                emb = representations[j, 1: seq_len + 1].mean(0).cpu().numpy()
                emb_dict[pid] = emb

        except RuntimeError as e:
            # OOM: xử lý từng sequence một
            print(f"\n[WARN] Batch OOM, chuyển single mode: {e}")
            for pid, seq in batch_data:
                try:
                    single_data = [(pid, seq)]
                    _, _, tokens = batch_converter(single_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        results = model(tokens, repr_layers=[33])
                    representations = results["representations"][33]
                    emb = representations[0, 1: len(seq) + 1].mean(0).cpu().numpy()
                    emb_dict[pid] = emb
                except Exception as e2:
                    print(f"[ERROR] Bỏ qua {pid}: {e2}")

    return emb_dict


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ESM-2 sequence embedding")
    parser.add_argument("-i", "--input",  required=True, help="Path to FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Path to output pickle")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (chậm hơn)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    print(f"Đọc FASTA: {args.input}")
    sequences = read_fasta(args.input)
    print(f"  {len(sequences):,} sequence")

    emb_dict = embed_with_esm2(sequences, batch_size=args.batch_size, device=device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"\nĐã lưu {len(emb_dict):,} embedding (dim=1280) → {out_path}")


if __name__ == "__main__":
    main()
