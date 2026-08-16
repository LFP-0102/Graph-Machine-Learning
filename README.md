# 图机器学习论文复现

使用 PyTorch 复现三篇经典图神经网络论文，并提供多随机种子实验、结果汇总与组会汇报图表。

| 模型 | 论文 | 主任务 |
| --- | --- | --- |
| GCN | Semi-Supervised Classification with Graph Convolutional Networks | Cora / CiteSeer / PubMed 节点分类 |
| GAT | Graph Attention Networks | Cora / CiteSeer / PubMed 节点分类 |
| GraphSAGE | Inductive Representation Learning on Large Graphs | PPI 归纳式多标签分类 |

## 实验协议

- GCN 和 GAT 使用 Planetoid 的 Cora、CiteSeer、PubMed 固定划分，指标为 test accuracy。
- GCN 使用两层结构、16 个隐藏维度、dropout 0.5、Adam `lr=0.01`、200 epochs。
- GAT 使用 8 个隐藏注意力头、每头 8 维、dropout 0.6、Adam `lr=0.005`、最多 1000 epochs 与验证集早停。
- GraphSAGE 的论文主实验使用 PPI JSON 数据与 Mean Aggregator、`25/10` 邻居采样、`128/128` 隐藏维度，指标为 micro-F1 和 macro-F1。
- GraphSAGE 在 Cora、CiteSeer、PubMed 上的 accuracy 仅是统一框架下的 citation 消融实验，不能代替论文 PPI 主结果。
- 批量实验只用验证集选择模型，最终测试集只在每次训练结束后评估一次。

## 环境

```powershell
conda activate pyg
pip install -r requirements.txt
```

项目保留 `torch-geometric` 依赖和 PyG 数据缓存，便于后续继续开展 PyG 相关研究；本项目的三种论文复现模型本身不调用 PyG 卷积层。

## 项目结构

```text
Graph-Machine-Learning/
├── data/                         # Cora、CiteSeer、PubMed、PPI 原始数据
├── datasets/                     # Planetoid 与 GraphSAGE JSON 数据加载
├── layers/                       # GCN、GAT、GraphSAGE 算子、聚合器与采样器
├── models/                       # 三种模型定义
├── experiments/
│   ├── train/                    # 单次论文协议训练入口
│   ├── comparison/               # 多种子实验与结果对比
│   ├── visualize/                # 组会图表入口
│   └── runner.py                 # 图表生成运行器
├── vis_tool/                     # 嵌入、注意力、分类和训练曲线可视化
├── notebooks/                    # PyG 数据集检查 Notebook
├── outputs/
│   ├── results/                  # 正式多种子结果 CSV
│   ├── runs/                     # 单次训练历史与汇总
│   └── visualizations/           # 组会图表
└── logs/                         # 每次实验的终端输出与组会实验记录
```

## 单次训练

在项目根目录运行：

```powershell
python experiments/train/train_gcn.py --dataset cora --seed 10
python experiments/train/train_gat.py --dataset cora --seed 10 --epochs 1000
python experiments/train/train_graphsage.py --train-prefix data/PPI/raw/ppi/ppi --sigmoid --seed 10
```

GCN、GAT 与 GraphSAGE citation 消融可选数据集为：`cora`、`citeseer`、`pubmed`。

## 五种子实验

Citation 网络的三种模型与三个数据集。每个“模型 × 数据集 × 随机种子”训练完成后，会立即保存独立的训练曲线、t-SNE、混淆矩阵、ROC、PR 和分类别准确率图像：

```powershell
python experiments/comparison/run_all_datasets.py --seeds 10 20 30 40 50
```

若要忽略已有 CSV 记录并完整重跑：

```powershell
python experiments/comparison/run_all_datasets.py --seeds 10 20 30 40 50 --force
```

GraphSAGE PPI 主实验：

```powershell
python experiments/comparison/multi_seed.py --train-prefix data/PPI/raw/ppi/ppi --seeds 10 20 30 40 50 123
```

结果保存于：

- `outputs/results/citation_multi_seed_raw.csv`
- `outputs/results/citation_multi_seed_summary.csv`
- `outputs/results/graphsage_ppi_multi_seed_raw.csv`
- `outputs/results/graphsage_ppi_multi_seed_summary.csv`
- `outputs/visualizations/<model>_<dataset>/seed_<seed>/`
- `outputs/visualizations/benchmark_summary/`：多模型多数据集汇总图与 GraphSAGE PPI F1 图

查看 citation 多种子汇总：

```powershell
python experiments/comparison/compare_checkpoint.py
```

## 组会图表

以下命令会按当前复现参数训练一次 Cora，并生成训练曲线、t-SNE 节点嵌入、混淆矩阵、ROC、PR、分类别准确率和 citation 子图；GAT 额外生成注意力图。它们不保存模型 checkpoint，不影响正式多种子结果。

```powershell
python experiments/visualize/visualize_gcn.py
python experiments/visualize/visualize_gat.py
python experiments/visualize/visualize_graphsage.py
```

输出目录：

- `outputs/visualizations/<model>_<dataset>/seed_<seed>/`：正式多种子运行的逐次图像
- `outputs/visualizations/benchmark_summary/`：跨模型/数据集和 PPI 的汇总图

对应训练过程和单次结果保存在 `outputs/runs/`，终端运行记录保存在 `logs/`。

## 对外展示操作

建议按下面的顺序完成一次 5--8 分钟的展示。展示时优先读取仓库已提交的多种子结果；现场只运行一个 GCN 示例来证明训练、评估和绘图流程可以端到端执行。不要在现场执行全量五种子实验，它会重新训练多个模型，耗时较长。

### 1. 演示前准备

在项目根目录打开 PowerShell，并确认环境和结果文件均可用：

```powershell
conda activate pyg
pip install -r requirements.txt
python experiments/comparison/compare_checkpoint.py
```

最后一条命令会打印 3 个模型在 Cora、CiteSeer、PubMed 上各 5 个随机种子的 `mean +/- std` test accuracy。说明比较结论使用多随机种子统计，而不是挑选单次最优结果。

### 2. 展示已有的基准结果

依次打开下面的目录和文件：

```powershell
Invoke-Item outputs\visualizations\benchmark_summary
Invoke-Item outputs\results\citation_multi_seed_summary.csv
Invoke-Item outputs\results\graphsage_ppi_multi_seed_summary.csv
```

推荐按以下顺序讲解图表：

1. `accuracy_mean_std.png`：比较三种模型在三个 citation 数据集上的平均准确率和标准差。
2. `seed_stability.png`：说明不同随机种子下的波动，配合 CSV 中的 `runs=5` 解释统计口径。
3. `accuracy_vs_time.png`：说明精度与训练时间的关系，而不只比较最高精度。
4. `ppi_f1_mean_std.png`：单独说明 GraphSAGE 的论文主实验使用 PPI 多标签分类，并以 micro-F1 / macro-F1 评价；它不能与 citation accuracy 放在同一张排名表中比较。

### 3. 现场复现一个端到端样例

运行 GCN 的 Cora 单次展示脚本：

```powershell
python experiments/visualize/visualize_gcn.py
```

命令结束后会在终端打印测试准确率、最佳轮次和图表输出目录。随后打开生成结果：

```powershell
Invoke-Item outputs\runs\gcn_cora\summary.csv
Invoke-Item outputs\visualizations\gcn_cora
```

按“`summary.csv` 中的最佳验证轮次与最终测试准确率 -> `training_curves.png` -> `embeddings_tsne.png` -> `confusion_matrix.png`”的顺序展示即可。这样可以清楚说明：模型选择只依据验证集，测试集只在训练结束后评估一次；图表则提供收敛过程、节点表示和分类错误的可检查证据。

若需要展示注意力机制，可额外运行并打开 GAT 的注意力图：

```powershell
python experiments/visualize/visualize_gat.py
Invoke-Item outputs\visualizations\gat_cora
```

其中 `attention_summary.png`、`attention_hub_node.png` 和 `attention_graph.png` 分别展示多节点汇总、中心节点的邻居权重与局部注意力子图。GraphSAGE 的 PPI 正式实验应使用本 README 中的 `multi_seed.py` 命令；`visualize_graphsage.py` 仅用于 citation 网络上的统一框架消融展示。

### 4. 可能被问到的问题

- **为什么结果不是单个数字？** 每个 citation 组合使用 5 个随机种子，报告均值和标准差，以反映随机初始化与训练波动。
- **为什么 GraphSAGE 的数值较低？** citation 上的 GraphSAGE 是消融实验；其论文主任务是 PPI 归纳式多标签分类，应查看独立的 F1 结果。
- **如何完整重跑？** 使用上文“五种子实验”中的命令；如确实需要覆盖已有 CSV，再加 `--force`。完整重跑会修改 `outputs/` 下的结果和图表，展示现场通常不需要执行。

## 当前实验结果

本表由 2026-08-13 使用种子 `10 20 30 40 50` 完成 3 个模型 × 3 个 citation 数据集的 45 次正式运行生成。数值为 test accuracy 的均值 ± 标准差。

| 模型 | Cora | CiteSeer | PubMed |
| --- | --- | --- | --- |
| GCN | 81.28% +/- 0.40% | 71.20% +/- 0.91% | 79.10% +/- 0.41% |
| GAT | 83.24% +/- 0.58% | 72.26% +/- 0.68% | 77.46% +/- 0.78% |
| GraphSAGE citation 消融 | 77.38% +/- 0.72% | 67.92% +/- 1.65% | 77.22% +/- 1.08% |

GraphSAGE PPI 主实验使用种子 `10 20 30 40 50 123`、10 个 epoch 和 batch size 512，共 6 次运行：micro-F1 为 `0.5854 +/- 0.0100`，macro-F1 为 `0.4337 +/- 0.0146`。其中官方 seed 123 的 micro-F1 为 `0.5995`，macro-F1 为 `0.4579`。

完整逐种子结果、耗时和实验协议见 `outputs/results/`；本轮运行记录见 `logs/2026-08-13.md`。
