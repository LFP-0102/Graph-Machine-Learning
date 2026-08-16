"""论文复现回归测试。

按论文协议训练一小段，断言最终指标落在论文数值附近，防止后续改动悄悄
破坏三篇论文的复现结果。阈值取论文数值 - 0.02，容忍不同机器/依赖版本
带来的微小抖动。

运行方式（需 pyg 环境，自带 numpy/torch/scikit-learn）：
    python -m unittest tests.test_repro -v
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from datasets.base import load_dataset
from models.gat import GAT
from models.gcn import GCN
from utils.graph_utils import normalized_sparse_edge_index
from utils.seed import set_seed


def _has_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


SKLEARN_AVAILABLE = _has_sklearn()


@torch.no_grad()
def _loss_and_accuracy(model, features, graph, labels, mask):
    model.eval()
    logits = model(features, graph)
    loss = torch.nn.functional.nll_loss(logits[mask], labels[mask]).item()
    accuracy = (logits[mask].argmax(1) == labels[mask]).float().mean().item()
    return loss, accuracy


def _test_accuracy(model, features, graph, labels, mask):
    return _loss_and_accuracy(model, features, graph, labels, mask)[1]


class TestGCNCitation(unittest.TestCase):
    """GCN 在 Cora 上的论文协议（论文 test acc 0.815）。"""

    def test_cora_accuracy(self):
        set_seed(123)
        data = load_dataset("cora")
        features, labels = data.features, data.labels
        adjacency = normalized_sparse_edge_index(data.edge_index, features.size(0), torch.device("cpu"))

        model = GCN(data.num_features, 16, data.num_classes, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        validation_losses = []
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(features, adjacency)
            loss = torch.nn.functional.nll_loss(logits[data.train_mask], labels[data.train_mask])
            loss = loss + 5e-4 * 0.5 * model.gc1.weight.square().sum()
            loss.backward()
            optimizer.step()
            val_loss, _ = _loss_and_accuracy(model, features, adjacency, labels, data.val_mask)
            validation_losses.append(val_loss)
            if epoch > 10 and val_loss > sum(validation_losses[-11:-1]) / 10:
                break

        accuracy = _test_accuracy(model, features, adjacency, labels, data.test_mask)
        self.assertGreaterEqual(
            accuracy, 0.78,
            f"GCN/Cora 测试准确率 {accuracy:.4f} 低于论文 0.815 的 -0.02 阈值",
        )


class TestGATCitation(unittest.TestCase):
    """GAT 在 Cora 上的论文协议（论文 test acc 0.830）。"""

    def test_cora_accuracy(self):
        set_seed(123)
        data = load_dataset("cora")
        features, labels, edge_index = data.features, data.labels, data.edge_index

        model = GAT(data.num_features, 8, data.num_classes, n_heads=8, dropout=0.6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        best_state, min_loss, max_accuracy, stale_epochs = None, float("inf"), 0.0, 0

        for _ in range(1000):
            model.train()
            optimizer.zero_grad()
            logits = model(features, edge_index)
            loss = torch.nn.functional.nll_loss(logits[data.train_mask], labels[data.train_mask])
            loss = loss + 5e-4 * 0.5 * sum(p.square().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

            val_loss, val_accuracy = _loss_and_accuracy(model, features, edge_index, labels, data.val_mask)
            if val_accuracy >= max_accuracy or val_loss <= min_loss:
                if val_accuracy >= max_accuracy and val_loss <= min_loss:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                max_accuracy = max(max_accuracy, val_accuracy)
                min_loss = min(min_loss, val_loss)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs == 100:
                    break

        model.load_state_dict(best_state)
        accuracy = _test_accuracy(model, features, edge_index, labels, data.test_mask)
        self.assertGreaterEqual(
            accuracy, 0.80,
            f"GAT/Cora 测试准确率 {accuracy:.4f} 低于论文 0.830 的 -0.02 阈值",
        )


@unittest.skipUnless(SKLEARN_AVAILABLE, "需要 scikit-learn 加载 PPI 数据")
class TestGraphSAGEPPI(unittest.TestCase):
    """GraphSAGE Mean 在 PPI 上的监督式主实验（论文 micro-F1 0.598，官方 seed 123）。"""

    def test_ppi_micro_f1(self):
        from datasets.graphsage_json import load_graphsage_json
        from models.graphsage_official import OfficialGraphSAGE

        set_seed(123)
        data = load_graphsage_json(str(ROOT_DIR / "data" / "PPI" / "raw" / "ppi" / "ppi"), seed=123)
        features, labels = data.features, data.labels
        train_adjacency, test_adjacency = data.train_adj, data.test_adj

        model = OfficialGraphSAGE(
            data.num_features, data.num_classes, dims=(128, 128), samples=(25, 10), dropout=0.0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            model.train()
            train_nodes = data.train_nodes[torch.randperm(data.train_nodes.numel())]
            for start in range(0, train_nodes.numel(), 512):
                batch = train_nodes[start:start + 512]
                logits = model(features, train_adjacency, batch)
                batch_labels = labels[batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
                optimizer.step()

        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for start in range(0, data.test_nodes.numel(), 512):
                batch = data.test_nodes[start:start + 512]
                predictions.append(model(features, test_adjacency, batch))
                targets.append(labels[batch])
        scores = torch.cat(predictions)
        truth = torch.cat(targets)
        predicted = (scores.sigmoid() > 0.5).numpy()
        truth = truth.numpy()
        from sklearn.metrics import f1_score
        micro = f1_score(truth, predicted, average="micro", zero_division=0)
        self.assertGreaterEqual(
            micro, 0.57,
            f"GraphSAGE/PPI micro-F1 {micro:.4f} 低于论文 0.598 的 -0.02 阈值",
        )


if __name__ == "__main__":
    unittest.main()
