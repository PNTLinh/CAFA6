import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv, AvgPooling, MaxPooling, SAGEConv
from model.layer import ConvPoolBlock, SAGPool


class PPIEncoder(torch.nn.Module):
    """
    2-layer GraphSAGE encoder trên PPI global graph.
    Nhận toàn bộ PPI graph và list node indices của batch,
    trả về embedding [batch_size, out_dim] cho từng protein.

    Protein không có trong PPI graph (node_id == -1) → zero vector.
    """
    def __init__(self, in_dim: int = 1024, hid_dim: int = 512, out_dim: int = 256,
                 dropout: float = 0.3):
        super(PPIEncoder, self).__init__()
        self.sage1 = SAGEConv(in_dim, hid_dim, aggregator_type="mean")
        self.sage2 = SAGEConv(hid_dim, out_dim, aggregator_type="mean")
        self.dropout = dropout

    def forward(self, ppi_graph: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        """
        ppi_graph : DGL graph toàn cục (đã ở đúng device), ndata["feat"] (N, in_dim)
        node_ids  : LongTensor [batch_size] — index node của từng protein (-1 = absent)
        returns   : FloatTensor [batch_size, out_dim]
        """
        h = ppi_graph.ndata["feat"]                       # (N, in_dim)
        h = F.relu(self.sage1(ppi_graph, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.sage2(ppi_graph, h)                      # (N, out_dim)

        # Lấy embedding cho từng protein trong batch; xử lý trường hợp node_id == -1
        valid_mask = node_ids >= 0
        out = torch.zeros(node_ids.shape[0], h.shape[1], device=h.device, dtype=h.dtype)
        out[valid_mask] = h[node_ids[valid_mask]]
        return out


class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_convs: int = 3,
                 pool_ratio: float = 0.5, dropout: float = 0.5,
                 ppi_in_dim: int = 1024, ppi_hid_dim: int = 512, ppi_out_dim: int = 256):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)

        # PPI encoder: GraphSAGE trên PPI global graph
        self.ppi_encoder = PPIEncoder(
            in_dim=ppi_in_dim,
            hid_dim=ppi_hid_dim,
            out_dim=ppi_out_dim,
            dropout=dropout,
        )

        # Fusion: struct(hid*2) + seq(1024) + ppi(ppi_out_dim)
        fusion_dim = hid_dim * 2 + 1024 + ppi_out_dim
        self.lin1 = torch.nn.Linear(fusion_dim, hid_dim * 2)
        self.lin2 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)

        self.line_new = torch.nn.Linear(fusion_dim, out_dim)



    def update_parent_features(self,label_network:dgl.DGLGraph, labels):
        # 获取图中的所有边
        edges = label_network.edges()

        second_dim_elements = labels[0,:]
        # 对于图中的每条边
        for child_idx, parent_idx in zip(edges[0], edges[1]):
            # 如果child节点的特征值大于parent节点的特征值
            if second_dim_elements[child_idx] > second_dim_elements[parent_idx]:
                # 更新parent节点的特征值为child节点的特征值
                second_dim_elements[parent_idx] = second_dim_elements[child_idx]
         # 更新labels的第二列为second_dim_elements
        labels[0, :] = second_dim_elements
        return labels
    

    def forward(self, graph: dgl.DGLGraph, sequence_feature: torch.Tensor,
                label_network: dgl.DGLGraph,
                ppi_graph: dgl.DGLGraph, ppi_node_ids: torch.Tensor) -> torch.Tensor:
        """
        graph            : batched contact-map DGL graph
        sequence_feature : [batch, 1024]
        label_network    : GO label co-occurrence graph (giữ nguyên, không dùng hiện tại)
        ppi_graph        : PPI global DGL graph (đã ở đúng device)
        ppi_node_ids     : [batch] — index node của từng protein trong PPI graph
        """
        # --- Protein structure branch ---
        feat = graph.ndata["feature"]
        final_readout = None
        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            final_readout = readout if final_readout is None else final_readout + readout

        # --- PPI branch ---
        ppi_emb = self.ppi_encoder(ppi_graph, ppi_node_ids)  # [batch, ppi_out_dim]

        # --- Feature fusion: struct + seq + ppi ---
        final_readout = torch.cat((final_readout, sequence_feature, ppi_emb), dim=-1)

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = self.lin3(feat)
        # feat: [batch_size, label_num]
        return feat

class SAGNetworkHierarchical2(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs:int=3,
                 pool_ratio:float=0.5, dropout:float=0.5):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs
       #self.classify = torch.nn.Linear(hid_dim, out_dim)
        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hid_dim * 2 + 1024, nhead=8), num_layers=3)    
        self.lin1 = torch.nn.Linear(hid_dim*2 + 1024, hid_dim*2)
        self.lin2 = torch.nn.Linear(hid_dim*2, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)
        # self.label_network1 = GATConv(1,1,num_heads=8,allow_zero_in_degree=True)

        self.line_new = torch.nn.Linear(hid_dim * 2 + 1024, out_dim)
        


    def update_parent_features(self,label_network:dgl.DGLGraph, feat):
        # feat: [batch_size, label_num]
        feat = feat.t()
        # feat: [label_num, batch_size]
        
        return feat
    

    def forward(self, graph:dgl.DGLGraph, sequence_feature,label_network:dgl.DGLGraph, ):
        feat = graph.ndata["feature"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        final_readout = torch.cat((final_readout,sequence_feature), -1)
        final_readout = self.transformer_encoder(final_readout)
        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)
        # feat = feat.t()
        # max_value,_ = torch.max(self.label_network1(label_network,feat),dim=1)
        # feat = F.relu(max_value)
        # feat = self.update_parent_features(label_network, feat)
        # feat = self.line_new(final_readout)
        # feat = torch.sigmoid(feat)
        
        # feat: [batch_size, label_num]
        return feat



class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs=3,
                 pool_ratio:float=0.5, dropout:float=0.0):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2 + 1024, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)
    
    def forward(self, graph:dgl.DGLGraph, sequence_feature):
        feat = graph.ndata["feature"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
        
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat([self.avg_readout(graph, feat), self.max_readout(graph, feat)], dim=-1)
        feat = torch.cat((feat,sequence_feature),-1)
        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)

        return feat



def get_sag_network(net_type:str="hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError("SAGNetwork type {} is not supported.".format(net_type))
