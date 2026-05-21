import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np

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

def create_logger(branch_name):
    logger = logging.getLogger(branch_name)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join('log','test_'+branch_name+'.log'))
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

# TODO 个人认为，测试集不用再枚举thresh了，直接使用验证集得出的最优thresh即可
Thresholds = [x/100 for x in range(1,100)]

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-thresh', '--thresh',type=float,default=0.71)
    parser.add_argument('-batch', '--batch', type = str, default = '1')
    parser.add_argument('-model_path', '--model_path', type=str, default='')
    args = parser.parse_args()
    
    input_thresh = args.thresh
    data_dir = os.environ.get("DATA_DIR", "D:/CAFA6")
    test_data_path = f'{data_dir}/divided_data/{args.branch}_test_dataset'
    label_network_path = f'{data_dir}/proceed_data/label_{args.branch}_network'
    term2idx_path = f'{data_dir}/proceed_data/label_vocab_{args.branch}.json'
    ppi_graph_path = f'{data_dir}/proceed_data/ppi_graph_global'
    model_path = args.model_path or f'save_models/bestmodel_{args.branch}_96_0.0001_0.2.pkl'
    
    logger = create_logger(args.branch)
    
    with open(test_data_path,'rb') as f:
        test_dataset = pickle.load(f)
    with open(label_network_path,'rb') as f:
        label_network = pickle.load(f)
    with open(term2idx_path,'r') as f:
        idx2term = json.load(f)
    with open(ppi_graph_path, 'rb') as f:
        ppi_graph = pickle.load(f)
    label_network = label_network.to(device)
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
    with open('test_result/'+args.branch+args.batch+"_pred_actual.pkl",'wb') as f:
        pickle.dump(temp, f)
        
    for i in range(len(pred)):
        protein = protein_list[i]
        result[protein] = []
        y_ = pred[i]
        y = actual[i]
        for j,x in enumerate(idx2term):
            # 寻找新预测出来的标签
            if y[j] < 1. and y_[j] > input_thresh:
                result[protein].append(x+f" {y_[j]:.5f}")
    with open('test_result/'+args.branch+'_result.json','w') as f:
        json.dump(result,f,indent=4)
        
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
    roc_curve_path = os.path.join("test_result", f"{args.branch}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.show()
    
    logger.info(f"ROC curve saved to {roc_curve_path}")

