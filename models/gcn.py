"""GCN（图卷积网络）模型。

复现论文：Semi-Supervised Classification with Graph Convolutional Networks
（Kipf & Welling，ICLR 2017）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_conv import GraphConvolution


class GCN(nn.Module):
    """两层 GCN 网络。

    参数：
        input_dim：输入特征维度。
        hidden_dim：隐藏层维度，论文中为 16。
        output_dim：分类类别数。
        dropout：Dropout 概率，论文中为 0.5。
    """

    graph_type = "adj"

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        """输入节点特征和归一化邻接矩阵，返回分类对数概率。"""
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_embeddings(self, x, adj):
        """提取第一层隐藏表示，供 t-SNE/UMAP 等可视化使用。"""
        self.eval()
        return F.relu(self.gc1(x, adj))
