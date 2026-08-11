# Graph Machine Learning

复现三篇经典图神经网络论文的对比实验框架。

| 模型 | 论文 | 年份 |
|------|------|------|
| GCN | Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling) | ICLR 2017 |
| GraphSAGE | Inductive Representation Learning on Large Graphs (Hamilton et al.) | NeurIPS 2017 |
| GAT | Graph Attention Networks (Veličković et al.) | ICLR 2018 |

---

## 环境

```bash
# Python 3.10+, PyTorch 2.x
pip install torch numpy scipy pandas matplotlib networkx scikit-learn
# 可选：UMAP 嵌入降维
pip install umap-learn
```

## 项目结构

```
Graph-Machine-Learning/
│
├── models/                     # 模型定义
│   ├── gcn.py                  #   GCN（两层图卷积）
│   ├── graphsage.py            #   GraphSAGE（Mean 聚合器）
│   └── gat.py                  #   GAT（8 头注意力 + 注意力权重提取）
│
├── layers/                     # 底层算子
│   ├── graph_conv.py           #   图卷积层 GCN
│   ├── graph_attention.py      #   图注意力层 GAT（单头 / 多头, 支持 return_attention）
│   ├── graphsage_layer.py      #   GraphSAGE 层
│   ├── sampler.py              #   邻居采样器
│   └── aggregators/            #   聚合器
│       ├── mean.py             #     Mean 聚合
│       ├── pool.py             #     Pool 聚合
│       └── lstm.py             #     LSTM 聚合
│
├── datasets/                   # 数据集加载
│   ├── base.py                 #   统一 DataDict 容器 + 加载入口
│   ├── cora.py / citeseer.py / pubmed.py
│
├── utils/                      # 工具函数
│   ├── training.py             #   train_epoch / evaluate / predict
│   ├── graph_utils.py          #   normalize_adj / add_self_loops
│   ├── paths.py                #   目录常量
│   ├── seed.py                 #   随机种子
│   ├── checkpoint.py           #   模型保存 / 加载
│   └── metrics.py              #   accuracy
│
├── experiments/                # 实验脚本
│   ├── train/                  #   训练脚本
│   │   ├── train_gcn.py        #     GCN → best_gcn.pt
│   │   ├── train_graphsage.py  #     GraphSAGE → best_graphsage.pt
│   │   └── train_gat.py        #     GAT → best_gat.pt
│   ├── visualize/              #   可视化脚本（调用 runner）
│   │   ├── visualize_gcn.py
│   │   ├── visualize_graphsage.py
│   │   └── visualize_gat.py
│   ├── comparison/               #   模型对比
│   │   ├── compare_train.py      #     从头训练对比（慢）
│   │   ├── compare_checkpoint.py #     checkpoint 横向对比（快）
│   │   ├── multi_seed.py         #     多 seed 统计（Mean ± Std）
│   │   └── run_all_datasets.py   #     多数据集实验
│   └── runner.py               #   通用实验运行器（训练 + 记录 + 画图）
│
├── vis_tool/                   # 可视化系统
│   ├── config.py               #   样式 / 配色 / 工具
│   ├── embedding/              #   嵌入降维
│   │   ├── tsne.py             #     t-SNE
│   │   └── umap.py             #     UMAP
│   ├── graph/                  #   图拓扑
│   │   └── topology.py
│   ├── plots/                  #   训练曲线 / 评估曲线
│   │   ├── train_curve.py      #     Loss / Accuracy
│   │   ├── roc.py              #     ROC 曲线
│   │   └── pr.py               #     Precision-Recall 曲线
│   ├── analysis/               #   注意力 / 分类分析
│   │   ├── attention.py        #     注意力权重可视化
│   │   └── classification.py   #     混淆矩阵
│   └── statistics/             #   结果表
│       └── result_table.py
│
├── data/                       # 原始数据（Planetoid 格式 pickle 文件）
└── outputs/                    # 实验产物
    ├── checkpoints/            #   模型权重
    ├── model_comparison.csv    #   模型横向对比表
    ├── results/                #   benchmark.csv（集中记录）
    ├── visualizations/         #   图表（按模型分子目录）
    └── runs/                   #   训练历史 CSV
```

---

## 快速开始

```bash
# 全部命令在项目根目录执行
cd Graph-Machine-Learning

# 设置 PYTHONPATH（Windows PowerShell / Git Bash）
export PYTHONPATH="."       # Git Bash
$env:PYTHONPATH="."         # PowerShell
```

### 1. 训练三个模型

```bash
python experiments/train/train_gcn.py
python experiments/train/train_graphsage.py
python experiments/train/train_gat.py
```

训练完成后 `outputs/checkpoints/` 下生成：

```
best_gcn.pt        best_graphsage.pt        best_gat.pt
```

### 2. 可视化（单个模型）

```bash
python experiments/visualize/visualize_gcn.py
python experiments/visualize/visualize_graphsage.py
python experiments/visualize/visualize_gat.py
```

每个模型生成到 `outputs/visualizations/<model>/`：

| 图表 | 含义 | 说明 |
|------|------|------|
| `training_curves.png` | Loss / Accuracy 曲线 | 所有模型 |
| `embeddings_tsne.png` | 隐藏层 t-SNE 嵌入 | 所有模型 |
| `embeddings_umap.png` | 隐藏层 UMAP 嵌入 | 可选 |
| `confusion_matrix.png` | 归一化混淆矩阵 | 所有模型 |
| `roc_curves.png` | ROC 曲线（多类） | 所有模型 |
| `pr_curves.png` | Precision-Recall 曲线 | 所有模型 |
| `per_class_accuracy.png` | 每类准确率 | 所有模型 |
| `graph_topology.png` | Cora 引文图拓扑 | 共享 |
| `attention_summary.png` | 多节点注意力分布对比 | **GAT 专属** |
| `attention_hub_node.png` | Hub 节点邻居注意力排名 | **GAT 专属** |
| `attention_graph.png` | 注意力权重边着色 | **GAT 专属** |

### 3. 模型横向对比

```bash
# 加载 checkpoint 对比（快，不重新训练）
python experiments/comparison/compare_checkpoint.py

# 从头训练并对比（慢，完整训练流程）
python experiments/comparison/compare_train.py

# 多 seed 重复实验（论文风格 Mean ± Std）
python experiments/comparison/multi_seed.py
```

**单次对比**（`compare_checkpoint.py` / `compare_train.py`）输出到 `outputs/model_comparison.csv`：

```
Model       Accuracy    Parameters   Inference
GCN          78.60%       23.06K      0.003
GraphSAGE    79.10%       46.10K      0.447
GAT          80.30%       92.30K      0.127
```

**多 seed 统计**（`multi_seed.py`）输出到 `outputs/results/multi_seed_summary.csv`：

| Model | Mean | Std |
|-------|------|-----|
| GAT | 80.76% | ±1.11% |
| GCN | 80.30% | ±0.97% |
| GraphSAGE | 76.00% | ±1.33% |

> 单次为 seed=42 结果；多 seed 为 10/20/30/40/50 共 5 次平均。

### 4. 多数据集实验

```bash
python experiments/comparison/run_all_datasets.py
```

自动跑完 3 模型 × 3 数据集共 9 组实验，结果汇总到 `outputs/results/benchmark.csv`。

---

## 数据集

| 数据集 | 节点数 | 特征维 | 类别数 | 训练/验证/测试 |
|--------|--------|--------|--------|----------------|
| Cora | 2,708 | 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 500 | 3 | 60 / 500 / 1,000 |

切换数据集：修改脚本中的 `dataset_name` 参数（`"cora"` / `"citeseer"` / `"pubmed"`）

---

## 模型超参数

### GCN

```python
GCN(input_dim=data.num_features, hidden_dim=16,
    output_dim=data.num_classes, dropout=0.5)

lr=0.01, weight_decay=5e-4
```

### GraphSAGE

当前实现 NeurIPS 2017 GraphSAGE **Mean Aggregator**，其他 Aggregator（Pool / LSTM）作为扩展接口保留在 `layers/aggregators/`。

```python
GraphSAGE(input_dim=data.num_features, hidden_dim=16,
          output_dim=data.num_classes, dropout=0.5)

lr=0.01, weight_decay=5e-4
```

### GAT

```python
GAT(input_dim=data.num_features, hidden_dim=8,
    output_dim=data.num_classes, n_heads=8, dropout=0.6)

lr=0.005, weight_decay=5e-4
```

> 以上参数均为论文原值。GCN 使用 normalized adjacency 矩阵，GraphSAGE / GAT 使用 edge_index。
> Cora 典型结果（seed=42, CPU）：GCN 78.6%，GraphSAGE 79.1%，GAT 80.3%。不同 seed 波动约 ±2%。

---

## 使用 runner 自定义实验

```python
from experiments.runner import run_experiment
from models.gcn import GCN

result = run_experiment(
    model_class=GCN,
    model_name="GCN",
    dataset_name="cora",

    # 模型参数
    model_kwargs={"hidden_dim": 16, "dropout": 0.5},

    # 训练参数
    lr=0.01,
    weight_decay=5e-4,
    epochs=200,
    patience=100,
    seed=42,

    # 可视化开关
    skip_tsne=False,    # True = 跳过 t-SNE（省时间）
    skip_graph=False,   # 图拓扑各模型相同，可跳过一次
)

print(f"Test Acc: {result['test_acc']:.4f}")
print(f"Parameters: {result['parameters']}")
print(f"Train Time: {result['train_time']:.2f}s")
```

runner 自动完成：
- 训练 + early stopping
- 保存 checkpoint → `outputs/checkpoints/best_<model>.pt`
- 保存训练历史 CSV → `outputs/runs/<model>/history.csv`
- 生成全套可视化图表（含 GAT 专属注意力可视化）
- 追加 benchmark → `outputs/results/benchmark.csv`

---

## GAT 注意力可视化

GAT 最大创新是**学习邻居重要性权重**。runner 会为 GAT 自动生成三张注意力图：

| 图表 | 内容 |
|------|------|
| `attention_summary.png` | 6 个高度节点的 Top-8 邻居注意力柱状图网格 |
| `attention_hub_node.png` | 最高度节点对每个邻居的注意力权重排名 |
| `attention_graph.png` | 图拓扑 + 边颜色/粗细按注意力权重编码 |

也可以独立调用：

```python
from models.gat import GAT
from vis_tool.analysis.attention import plot_attention_summary

avg_attn, all_attn = model.get_attention_weights(features, edge_index)
plot_attention_summary(avg_attn.numpy(), edge_index.numpy(),
                       num_nodes=6, top_k=8,
                       save_path="attention_summary.png")
```

---

## 添加新模型

1. 在 `models/` 下创建模型文件，继承 `nn.Module`：

```python
class MyGNN(nn.Module):
    graph_type = "edge_index"   # "adj" 或 "edge_index"

    def forward(self, x, graph):
        ...
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, graph):
        """返回隐藏层嵌入用于可视化（可选）"""
        ...
```

2. 创建 `experiments/train/train_mygnn.py` / `experiments/visualize/visualize_mygnn.py`

3. 在 `experiments/comparison/` 中加入新模型对比

---

## 预期结果（CPU, seed=42）

| Dataset | GCN | GraphSAGE | GAT |
|---------|-----|-----------|-----|
| Cora | 79.30% | 78.30% | **80.30%** |
| CiteSeer | 69.00% | 66.70% | **69.10%** |
| PubMed | 78.20% | 74.70% | **77.80%** |

### 多 seed 统计（Cora, 5 seeds）

| 指标 | GCN | GraphSAGE | GAT |
|------|-----|-----------|-----|
| Test Accuracy | 80.30 ± 0.97% | 76.00 ± 1.33% | **80.76 ± 1.11%** |
| Parameters | 23,063 | 46,103 | 92,302 |
| Train Time | ~1.7s | ~131s | ~242s |

> PyTorch 2.x, Intel i7。GAT 使用稀疏边注意力（O(E) 内存）支持大图。GPU 下训练更快。
