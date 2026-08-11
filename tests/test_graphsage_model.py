from datasets.cora_graphsage import load_cora_graphsage
from layers.sampler import NeighborSampler
from layers.aggregators.mean import MeanAggregator
from models.graphsage import GraphSAGE
from utils.paths import DATA_DIR

# 加载数据
data = load_cora_graphsage(str(DATA_DIR))
features = data["features"]
adj_lists = data["adj_lists"]

# 创建 sampler & aggregator
sampler1 = NeighborSampler(adj_lists, num_samples=5)
sampler2 = NeighborSampler(adj_lists, num_samples=5)
agg1 = MeanAggregator(input_dim=1433)
agg2 = MeanAggregator(input_dim=64)

# 创建模型
model = GraphSAGE(
    input_dim=1433, hidden_dim=64, output_dim=7,
    aggregator1=agg1, aggregator2=agg2,
    sampler1=sampler1, sampler2=sampler2,
)

# 测试 forward
nodes = [0, 1, 2, 3]
output = model(nodes, features, adj_lists)
print("输出:")
print(output.shape)
