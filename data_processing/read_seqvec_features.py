'''
根据论文, 这里是读取SeqVec特征, 输出到dict_sequence_feature文件中
'''
import pickle
import pandas as pd
import numpy as np

# 序列mean特征，从HNetGO上找到，原seq2vec的模型在原服务器上无法找到，使用预训练好的
with open('/mnt/workspace/replicate/9606-avg-emb.pkl','rb')as f:
    sequence_feature = pickle.load(f) 

df=pd.read_csv("/mnt/workspace/replicate/Struct2GO/data/protein_list.csv",sep=" ")
list0=df.values.tolist()
protein_list = np.array(list0)

dict_sequence_feature={}
list1 = []
for i in list0:
    list1.append(i[0])

for name in list1:
    dict_sequence_feature[name] = [0.0]*1024

cnt=0
for protein in sequence_feature.keys():
    if protein in protein_list:
        dict_sequence_feature[protein] = sequence_feature[protein]
        cnt=cnt+1

with open('/mnt/workspace/replicate/Struct2GO/dict_sequence_feature','wb')as f:
    pickle.dump(dict_sequence_feature,f)
