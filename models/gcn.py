"""
GCN（Graph Convolutional Network）模型。

复现论文：Semi-Supervised Classification with Graph Convolutional Networks
            (Kipf & Welling, ICLR 2017)

两层 GCN：
    Z = softmax(Â ReLU(Â X W₀) W₁)

其中 Â = D^(-1/2)(A+I)D^(-1/2) 是归一化邻接矩阵。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_conv import GraphConvolution


class GCN(nn.Module):
    """两层 GCN 网络。
    参数：
        input_dim:   输入特征维度
        hidden_dim:  隐藏层维度 (论文中 = 16)
        output_dim:  分类类别数
        dropout:     Dropout 率 (论文中 = 0.5)
    """

    graph_type = "adj"              # runner 据此选择 graph 输入

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout
        # 第一层 GCN: X → H
        # H = ReLU(Â X W₀)
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        # 第二层 GCN: H → Z
        # Z = Â H W₁
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
    def forward(self, x, adj):
        """
        参数：
            x:   节点特征 [N, F]
            adj: 归一化邻接矩阵 Â [N, N]

        返回：
            log_softmax 分类概率 [N, C]
        """
        # 第一层：图卷积 + ReLU + Dropout
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 第二层：图卷积 → log_softmax
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_embeddings(self, x, adj):
        """提取隐藏层嵌入（用于 t-SNE / UMAP 可视化）。

        返回论文 Figure 3a 所需的 2D 投影前的隐藏表示。
        """
        self.eval()
        x = self.gc1(x, adj)
        x = F.relu(x)
        # 注意：不经过 dropout（eval 模式下已自动关闭）
        return x