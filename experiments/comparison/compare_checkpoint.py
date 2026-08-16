"""不加载模型权重，直接打印已完成的引文网络多种子汇总结果。

项目刻意不保留单次训练的模型 checkpoint，因此本脚本读取已记录的五种子均值和
标准差，而不依赖已清理的 ``best_*.pt`` 文件。
"""
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT_DIR / "outputs" / "results" / "citation_multi_seed_summary.csv"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"未找到基准结果：{SUMMARY_PATH}。"
            "请先运行 experiments/comparison/run_all_datasets.py。"
        )
    summary = pd.read_csv(SUMMARY_PATH)
    display = summary[["model", "dataset", "runs", "accuracy_mean", "accuracy_std"]].copy()
    display["accuracy"] = display.apply(
        lambda row: f"{row.accuracy_mean:.2%} +/- {row.accuracy_std:.2%}", axis=1
    )
    print(display[["model", "dataset", "runs", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
