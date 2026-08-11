"""
测试 GraphSAGE 模型
功能: 1. 加载 Cora 数据  2. 创建 GraphSAGE  3. 测试 forward
"""
import torch
from datasets.cora import load_cora
from models.graphsage import GraphSAGE

def test_graphsage():
    # 1. 加载数据
    data = load_cora()
    print("数据:", data)

    in_channels = data.x.shape[1]
    out_channels = int(data.y.max()) + 1
    print("输入特征维度:", in_channels)
    print("类别数量:", out_channels)

    # 2. 创建模型
    model = GraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
    print(model)

    # 3. forward 测试
    out = model(data.x, data.edge_index)
    print("输出 shape:", out.shape)

    # 4. 检查结果
    assert out.shape == (data.num_nodes, out_channels)
    print("GraphSAGE 测试通过!")

if __name__ == "__main__":
    test_graphsage()
