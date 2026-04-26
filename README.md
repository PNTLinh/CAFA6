# ESM-AlphaFold-GO

## enviroment and platform
    Some works on features used Google Colab.
    traing platform using AliCloud, with Nvidia A10 GPU
    enviroment is the same as ./enviroment.yml

## key steps
    1. Process the struct data(pdb files) using ./data_processing/predicted_protein_struct2map.py
    2. Get node embeddiings, refer to ./data_processing/node2vec.ipynb and /read_node2vec_feature.py
    3. Download sequence data from UniProt, and then find out ESM-2, downlod their open-sorce pretrained model (esm2_t33_650M_UR50D), refer to ./data_processing/ESeq2Vec.ipynb;
    4. Run labels_load.py in the path model/. remember the tags number of each GO catgory, we need that in the training part
    5. run trian_Struct2GO2.py, fill the parameters.
