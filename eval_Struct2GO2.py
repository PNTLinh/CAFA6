import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import warnings

from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re


def _argv_has(*names: str) -> bool:
    return any(n in sys.argv for n in names)


_BASELINE_EVAL_THRESH = {"mf": 0.71, "cc": 0.5, "bp": 0.4}


def _load_model_checkpoint(model_path, device):
    """Load trusted full-model checkpoints across torch versions."""
    try:
        # PyTorch>=2.6 defaults to weights_only=True, but this project saves full model objects.
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only argument.
        return torch.load(model_path, map_location=device)

def create_logger(branch_name, data_dir: str):
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"test_{branch_name}")
    if logger.handlers:
        return logger
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(log_dir, "test_" + branch_name + ".log"))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

warnings.filterwarnings('ignore')


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MyDataSet' and module in {'__main__', 'data_processing.divide_data'}:
            return MyDataSet
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, 'rb') as handle:
        return _CompatUnpickler(handle).load()

# TODO 个人认为，测试集不用再枚举thresh了，直接使用验证集得出的最优thresh即可
Thresholds = [x / 100 for x in range(1, 100)]

_ACS_FILES = {
    "mf": "human_MF_ACS.json",
    "cc": "human_CC_ACS.json",
    "bp": "human_BP_ACS.json",
}


def _resolve_data_dir() -> str:
    """Pick the first usable CAFA6 data root for local or Kaggle runs."""
    candidates = []
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    candidates.append(Path("D:/CAFA6"))

    for candidate in candidates:
        if (candidate / "divided_data").exists() and (candidate / "proceed_data").exists():
            return str(candidate)

    return env_data_dir or str(Path(__file__).resolve().parent)


def _resolve_eval_dataset(data_dir: str, branch: str, split: str) -> tuple[str, str]:
    """Return (path, split_name). Kaggle pack often has valid but not test."""
    divided = os.path.join(data_dir, "divided_data")
    order: list[str] = []
    if split in ("test", "auto"):
        order.append(f"{branch}_test_dataset")
    if split in ("valid", "auto"):
        order.append(f"{branch}_valid_dataset")
    if split == "train":
        order.append(f"{branch}_train_dataset")
    for name in order:
        path = os.path.join(divided, name)
        if os.path.isfile(path):
            split_name = name.replace(f"{branch}_", "").replace("_dataset", "")
            return path, split_name
    raise FileNotFoundError(
        f"No eval dataset under {divided} for branch={branch}, split={split}. "
        f"Tried: {order}. Re-run kaggle_link_data.py or pack test_dataset in kaggle_data.zip."
    )


def _resolve_latest_model_path(data_dir: str, branch: str, baseline_parity: bool) -> str:
    """Pick the newest bestmodel checkpoint for a branch when no path is provided."""
    save_models = Path(data_dir) / "save_models"
    if baseline_parity:
        preferred = save_models / f"bestmodel_{branch}_64_0.0001_0.3.pkl"
        if preferred.is_file():
            return str(preferred)

    candidates: list[tuple[float, Path]] = []
    for path in save_models.glob(f"bestmodel_{branch}_*.pkl"):
        match = re.search(r"_(\d+(?:\.\d+)?)\.pkl$", path.name)
        if not match:
            continue
        candidates.append((float(match.group(1)), path))

    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])

    fallback = save_models / f"bestmodel_{branch}_96_0.0001_0.2.pkl"
    return str(fallback)


def _load_vocab(data_dir: str, branch: str) -> list:
    proc = os.path.join(data_dir, "proceed_data")
    vocab_path = os.path.join(proc, f"label_vocab_{branch}.json")
    if os.path.isfile(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    acs_path = os.path.join(proc, _ACS_FILES[branch])
    if os.path.isfile(acs_path):
        with open(acs_path, "r", encoding="utf-8") as f:
            protein_labels = json.load(f)
        terms = sorted({t for terms in protein_labels.values() for t in terms})
        print(f"[INFO] Built vocab from {acs_path} ({len(terms)} terms)")
        return terms
    warnings.warn(
        f"Missing label_vocab_{branch}.json and {_ACS_FILES[branch]} under {proc}; "
        "falling back to placeholder GO term names based on label graph size. "
        "This lets evaluation run, but result JSON will not contain real GO IDs."
    )
    return []


def _align_vocab_to_output(idx2term: list, output_dim: int) -> list:
    """Make vocabulary length match model output dimension."""
    terms = list(idx2term)
    if len(terms) == output_dim:
        return terms
    if len(terms) > output_dim:
        print(f"[WARN] Vocab has {len(terms)} terms but model outputs {output_dim}; truncating vocab")
        return terms[:output_dim]
    print(f"[WARN] Vocab has {len(terms)} terms but model outputs {output_dim}; padding placeholders")
    start = len(terms)
    terms.extend([f"GO_TERM_{i:05d}" for i in range(start, output_dim)])
    return terms


def _as_flat_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def _as_scalar_long_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long).reshape(())


def _fix_dim_1d(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = int(x.shape[0])
    if cur == target_dim:
        return x
    if cur > target_dim:
        return x[:target_dim]
    out = torch.zeros(target_dim, dtype=x.dtype)
    out[:cur] = x
    return out


def _build_eval_collate_fn(seq_dim: int, label_dim: int):
    warned = {"seq": False, "label": False}

    def _collate(batch):
        pids, graphs, labels, seq_feats, ppi_node_ids = zip(*batch)

        fixed_seq_feats = []
        for s in seq_feats:
            t = _as_flat_float_tensor(s)
            if (not warned["seq"]) and int(t.shape[0]) != seq_dim:
                print(
                    f"[WARN] Sequence feature dim mismatch in batch "
                    f"(got {int(t.shape[0])}, expected {seq_dim}); applying pad/truncate"
                )
                warned["seq"] = True
            fixed_seq_feats.append(_fix_dim_1d(t, seq_dim))

        fixed_labels = []
        for y in labels:
            t = _as_flat_float_tensor(y)
            if (not warned["label"]) and int(t.shape[0]) != label_dim:
                print(
                    f"[WARN] Label dim mismatch in batch "
                    f"(got {int(t.shape[0])}, expected {label_dim}); applying pad/truncate"
                )
                warned["label"] = True
            fixed_labels.append(_fix_dim_1d(t, label_dim))

        batched_graph = dgl.batch(list(graphs))
        return (
            list(pids),
            batched_graph,
            torch.stack(fixed_labels, dim=0),
            torch.stack(fixed_seq_feats, dim=0),
            torch.stack([_as_scalar_long_tensor(pid) for pid in ppi_node_ids], dim=0),
        )

    return _collate


if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-thresh', '--thresh',type=float,default=0.71)
    parser.add_argument('-batch', '--batch', type = str, default = '1')
    parser.add_argument("-model_path", "--model_path", type=str, default="")
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "test", "valid", "train"],
        help="auto: test nếu có, không thì valid (phù hợp Kaggle pack)",
    )
    parser.set_defaults(use_ppi=True, baseline_parity=True)
    parser.add_argument(
        "--no-ppi",
        dest="use_ppi",
        action="store_false",
        help="Không load PPI graph khi eval (model train không PPI)",
    )
    parser.add_argument(
        "--no-baseline-parity",
        dest="baseline_parity",
        action="store_false",
        help="Eval nhanh: split auto, model path theo run Kaggle",
    )
    parser.add_argument(
        "--baseline-parity",
        dest="baseline_parity",
        action="store_true",
        help="Eval khớp baseline (mặc định bật)",
    )
    args = parser.parse_args()

    if args.baseline_parity:
        args.split = "test"
        if not _argv_has("-thresh", "--thresh"):
            args.thresh = _BASELINE_EVAL_THRESH.get(args.branch, 0.71)

    input_thresh = args.thresh
    data_dir = _resolve_data_dir()
    test_data_path, eval_split = _resolve_eval_dataset(data_dir, args.branch, args.split)
    label_network_path = f"{data_dir}/proceed_data/label_{args.branch}_network"
    ppi_graph_path = f"{data_dir}/proceed_data/ppi_graph_global"
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = _resolve_latest_model_path(data_dir, args.branch, args.baseline_parity)
    if not os.path.isabs(model_path) and not os.path.isfile(model_path):
        alt = os.path.join(data_dir, model_path)
        if os.path.isfile(alt):
            model_path = alt

    result_dir = os.path.join(data_dir, "test_result")
    os.makedirs(result_dir, exist_ok=True)

    logger = create_logger(args.branch, data_dir)
    logger.info(
        f"eval split={eval_split}, dataset={test_data_path}, "
        f"baseline_parity={args.baseline_parity}, use_ppi={args.use_ppi}, "
        f"thresh={input_thresh}, model={model_path}"
    )

    import __main__

    __main__.MyDataSet = MyDataSet

    test_dataset = _load_pickle(test_data_path)
    label_network = _load_pickle(label_network_path)
    label_network = label_network.to(device)
    ppi_graph = None
    if args.use_ppi:
        ppi_graph = _load_pickle(ppi_graph_path)
        ppi_graph = ppi_graph.to(device)
    idx2term = _load_vocab(data_dir, args.branch)
    if not idx2term:
        idx2term = [f"GO_TERM_{i:05d}" for i in range(label_network.num_nodes())]
        print(f"[WARN] Using placeholder vocabulary with {len(idx2term)} terms")
    model = _load_model_checkpoint(model_path, device)
    model = model.to(device)
    use_ppi = getattr(model, "use_ppi", args.use_ppi)
    if args.use_ppi and not use_ppi:
        print("[WARN] --use_ppi set but checkpoint has use_ppi=False; following model")
    if not use_ppi:
        args.use_ppi = False
    ppi_node_emb = None
    if use_ppi and ppi_graph is not None and hasattr(model, "encode_ppi_nodes"):
        with torch.no_grad():
            ppi_node_emb = model.encode_ppi_nodes(ppi_graph)

    if hasattr(model, "fusion_attn") and hasattr(model.fusion_attn, "seq_proj"):
        seq_dim = int(model.fusion_attn.seq_proj.in_features)
    else:
        seq_dim = int(np.asarray(test_dataset[0][3]).reshape(-1).shape[0])
    if hasattr(model, "lin3"):
        label_dim = int(model.lin3.out_features)
    else:
        label_dim = int(label_network.num_nodes())

    collate_fn = _build_eval_collate_fn(seq_dim=seq_dim, label_dim=label_dim)

    batch_size = 32
    test_dataloader = GraphDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )
    criterion = nn.CrossEntropyLoss()
    logger.info('#########'+args.branch+'###########')
    logger.info('########start testing###########') 

    t_loss = 0
    test_batch_num = 0
    pred = []
    actual = []
    protein_list = []
    model.eval()
    print("testing")
    with torch.no_grad():
        for i, (pids, graphs, labels, seq_feats, ppi_node_ids) in tqdm(enumerate(test_dataloader)):
            graphs = graphs.to(device)
            seq_feats = seq_feats.to(device)
            labels = labels.to(device)
            ppi_node_ids = ppi_node_ids.to(device)
            labels = torch.squeeze(labels)
            if len(labels.shape)==1:
                labels = labels.unsqueeze(0)
            
            logits = model(
                graphs,
                seq_feats,
                label_network,
                ppi_graph=ppi_graph if args.use_ppi else None,
                ppi_node_ids=ppi_node_ids if args.use_ppi else None,
                ppi_node_emb=ppi_node_emb,
            )
            logits = F.sigmoid(logits)
            
            loss = criterion(logits, labels.float())
            
            protein_list += pids
            t_loss += loss.item()
            pred += logits.tolist()
            actual += labels.tolist()
    
    # 为了保持可控，这里使用传入的thresh来确定最终分类结果
    assert len(pred) == len(actual)
    assert len(pred) == len(protein_list)
    if not pred:
        raise RuntimeError("No predictions were generated. Check eval dataset and dataloader.")

    output_dim = len(pred[0])
    idx2term = _align_vocab_to_output(idx2term, output_dim)

    result = {}
    
    temp = {}
    temp["pred"] = pred
    temp["actual"] = actual
    with open(os.path.join(result_dir, args.branch + args.batch + "_pred_actual.pkl"), "wb") as f:
        pickle.dump(temp, f)

    for i in range(len(pred)):
        protein = protein_list[i]
        result[protein] = []
        y_ = pred[i]
        y = actual[i]
        for j in range(output_dim):
            x = idx2term[j]
            # 寻找新预测出来的标签
            if y[j] < 1.0 and y_[j] > input_thresh:
                result[protein].append(x + f" {y_[j]:.5f}")
    with open(os.path.join(result_dir, args.branch + "_result.json"), "w") as f:
        json.dump(result, f, indent=4)
        
    t_loss /= len(test_dataloader)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())

    each_best_fcore = 0
    each_best_scores = []
    for thresh in tqdm(Thresholds):
        f_score,precision, recall  = calculate_performance(actual, pred, label_network,threshold=thresh)
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [thresh, f_score, recall, precision]
    t, f_score, recall, precision = each_best_scores[0], each_best_scores[1], each_best_scores[2], each_best_scores[3]
    logger.info('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    logger.info('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))
    print('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    print('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))
    

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # 参考线（随机分类器）
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({args.branch})')
    plt.legend(loc="lower right")

    # save ROC pic
    roc_curve_path = os.path.join(result_dir, f"{args.branch}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.show()
    
    logger.info(f"ROC curve saved to {roc_curve_path}")

