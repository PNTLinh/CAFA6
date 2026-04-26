import os
import numpy as np
import pandas as pd 

# TODO: 统一文件夹路径
protein_list = []
cnt = 0
for path,dir_list,file_list in os.walk("/mnt/workspace/Struct2GO/struct_feature"):
    for file in file_list:
        protein_name = file.split('.')[0]
        protein_list.append(protein_name)
        cnt=cnt+1
data = pd.DataFrame(protein_list)
data.to_csv("/mnt/workspace/replicate/Struct2GO/data/protein_list.csv",sep=" ",index=False,header=False)
print(cnt)