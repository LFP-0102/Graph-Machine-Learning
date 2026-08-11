from datasets.cora_graphsage import load_cora_graphsage
from utils.paths import DATA_DIR

data = load_cora_graphsage(DATA_DIR)

print(data["features"].shape)
print(data["labels"].shape)
print(data["adj_lists"][0])
print(len(data["train_nodes"]))
