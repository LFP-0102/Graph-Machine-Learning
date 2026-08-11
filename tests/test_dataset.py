from datasets.loader import load_dataset

dataset = load_dataset("Cora")
print(dataset)

data = dataset[0]
print(data)
print("----------------")

print("节点数量:")
print(data.num_nodes)

print("特征维度:")
print(data.num_features)

print("类别数量:")
print(dataset.num_classes)
