import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    手写 Graph Convolution Layer
    对应论文：
    Semi-Supervised Classification with Graph Convolutional Networks
    GCN核心公式：
    H^(l+1)=σ(A_hat H^(l) W^(l))
    其中：
    A_hat:
        归一化后的邻接矩阵
    H:
        节点特征矩阵
    W:
        可学习参数矩阵
    σ:
        激活函数(ReLU)
    """
    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()
        # 定义可训练权重矩阵 W
        # 输入:
        #     节点特征维度 in_features
        # 输出:
        #     新的节点表示维度 out_features
        # 对应公式：
        # XW
        self.weight = nn.Parameter(
            torch.FloatTensor(
                in_features,
                out_features
            )
        )
        # 偏置参数 b
        #
        # 最终计算：
        #
        # AXW+b
        self.bias = nn.Parameter(
            torch.FloatTensor(
                out_features
            )
        )
        # 初始化参数
        self.reset_parameters()
    def reset_parameters(self):
        """
        参数初始化
        使用 Xavier 初始化权重：
        可以让训练初期梯度更加稳定
        """
        nn.init.xavier_uniform_(
            self.weight
        )
        # 偏置初始化为0
        nn.init.zeros_(
            self.bias
        )
    def forward(
        self,
        x,
        adj
    ):
        """
        前向传播
        输入：
        x:
            节点特征矩阵
            shape:
            [节点数量, 输入特征维度]
        adj:
            归一化邻接矩阵 A_hat
            shape:
            [节点数量, 节点数量]
        输出:
            更新后的节点表示
        """
        # 第一步：
        # 特征变换
        # XW
        # 对应论文：
        # H^(l)W^(l)
        support=torch.mm(
            x,
            self.weight
        )
        # 第二步：
        # 图结构信息传播
        # A_hat XW
        # 每个节点聚合邻居信息
        output=torch.mm(
            adj,
            support
        )
        # 加偏置
        # 得到：
        # A_hat XW+b
        return output+self.bias