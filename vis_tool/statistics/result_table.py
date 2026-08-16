"""实验结果表格工具。"""

import pandas as pd


def create_result_table(results):
    """将实验结果字典列表转换为 Pandas 表格。"""
    return pd.DataFrame(results)


def save_result_table(results, path="./outputs/results.csv"):
    """将实验结果表保存为 CSV 文件。"""
    create_result_table(results).to_csv(path, index=False)
    print(f"结果已保存到：{path}")
