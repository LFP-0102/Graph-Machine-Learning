"""使用多个随机种子重复运行官方风格的 GraphSAGE PPI 实验。"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
TRAIN_SCRIPT = ROOT_DIR / "experiments" / "train" / "train_graphsage.py"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
OUTPUT_DIR = ROOT_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prefix", required=True)
    # 官方 GraphSAGE 固定使用 seed=123，纳入默认列表以便汇总均值贴近论文值。
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 40, 50, 123])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--force",
        action="store_true",
        help="保留以便与其他批量脚本使用方式一致；该脚本本来就会覆盖本轮结果。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train-prefix", args.train_prefix,
            "--sigmoid",
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]
        print(f"\n{'=' * 56}\nGraphSAGE PPI | 随机种子={seed}\n{'=' * 56}")
        completed = subprocess.run(
            command, cwd=ROOT_DIR, check=True, capture_output=True, text=True
        )
        print(completed.stdout, end="")
        match = re.search(
            r"测试 micro-F1=(?P<micro>\d+\.\d+) \| macro-F1=(?P<macro>\d+\.\d+) \| 用时=(?P<time>\d+\.\d+)s",
            completed.stdout,
        )
        if match is None:
            raise RuntimeError("无法从 GraphSAGE PPI 训练输出中解析评估指标")
        rows.append({
            "seed": seed,
            "test_micro_f1": float(match["micro"]),
            "test_macro_f1": float(match["macro"]),
            "train_time": float(match["time"]),
        })

    raw = pd.DataFrame(rows)
    raw.insert(1, "epochs", args.epochs)
    raw_path = RESULTS_DIR / "graphsage_ppi_multi_seed_raw.csv"
    raw.to_csv(raw_path, index=False)
    summary = pd.DataFrame([{
        "model": "GraphSAGE Mean",
        "runs": len(raw),
        "epochs": args.epochs,
        "micro_f1_mean": raw["test_micro_f1"].mean(),
        "micro_f1_std": raw["test_micro_f1"].std(ddof=1) if len(raw) > 1 else 0.0,
        "macro_f1_mean": raw["test_macro_f1"].mean(),
        "macro_f1_std": raw["test_macro_f1"].std(ddof=1) if len(raw) > 1 else 0.0,
    }])
    summary_path = RESULTS_DIR / "graphsage_ppi_multi_seed_summary.csv"
    summary.to_csv(summary_path, index=False)
    from vis_tool.statistics import plot_ppi_benchmark
    plot_ppi_benchmark(raw, OUTPUT_DIR / "visualizations" / "benchmark_summary")
    record = summary.iloc[0]
    print(
        f"\nMicro-F1：{record['micro_f1_mean']:.4f} +/- {record['micro_f1_std']:.4f}\n"
        f"Macro-F1：{record['macro_f1_mean']:.4f} +/- {record['macro_f1_std']:.4f}\n"
        f"原始结果：{raw_path}\n汇总结果：{summary_path}"
    )


if __name__ == "__main__":
    main()
