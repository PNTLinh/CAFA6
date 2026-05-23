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
    args = parser.parse_args()

    input_thresh = args.thresh
    data_dir = os.environ.get("DATA_DIR", "D:/CAFA6")
    test_data_path, eval_split = _resolve_eval_dataset(data_dir, args.branch, args.split)
    label_network_path = f"{data_dir}/proceed_data/label_{args.branch}_network"
    ppi_graph_path = f"{data_dir}/proceed_data/ppi_graph_global"
    model_path = args.model_path or f"{data_dir}/save_models/bestmodel_{args.branch}_96_0.0001_0.2.pkl"
    if not os.path.isabs(model_path) and not os.path.isfile(model_path):
        alt = os.path.join(data_dir, model_path)
        if os.path.isfile(alt):
            model_path = alt

    result_dir = os.path.join(data_dir, "test_result")
    os.makedirs(result_dir, exist_ok=True)

    logger = create_logger(args.branch, data_dir)
    logger.info(f"eval split={eval_split}, dataset={test_data_path}")

    import __main__

    __main__.MyDataSet = MyDataSet

    test_dataset = _load_pickle(test_data_path)
    label_network = _load_pickle(label_network_path)
    ppi_graph = _load_pickle(ppi_graph_path)
    label_network = label_network.to(device)
    idx2term = _load_vocab(data_dir, args.branch)
    if not idx2term:
        idx2term = [f"GO_TERM_{i:05d}" for i in range(label_network.num_nodes())]
        print(f"[WARN] Using placeholder vocabulary with {len(idx2term)} terms")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    ppi_graph = ppi_graph.to(device)
    ppi_node_emb = None
    if hasattr(model, 'encode_ppi_nodes'):
        with torch.no_grad():
            ppi_node_emb = model.encode_ppi_nodes(ppi_graph)

    batch_size = 32
    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size = batch_size, drop_last = False, shuffle = False)
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
                graphs, seq_feats, label_network,
                ppi_graph=ppi_graph, ppi_node_ids=ppi_node_ids, ppi_node_emb=ppi_node_emb,
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
    assert len(pred[0]) == len(idx2term)
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
        for j, x in enumerate(idx2term):
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

