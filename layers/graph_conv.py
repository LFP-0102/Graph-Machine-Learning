import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """手写 Graph Convolution Layer.

    对应论文: Semi-Supervised Classification with Graph Convolutional Networks
    GCN 核心公式: H^(l+1) = σ(Â H^(l) W^(l))
    Â: 归一化邻接矩阵, H: 节点特征矩阵, W: 可学习参数矩阵, σ: 激活函数 (ReLU)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier 初始化权重, 偏置初始化为 0."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """前向传播: Â X W + b.

        x:   节点特征矩阵 [N, F]
        adj: 归一化邻接矩阵 Â [N, N]
        """
        support = torch.mm(x, self.weight)   # XW
        output = torch.mm(adj, support)       # Â XW
        return output + self.bias
