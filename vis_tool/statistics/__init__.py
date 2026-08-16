"""实验结果统计子包。"""

from vis_tool.statistics.result_table import create_result_table, save_result_table

__all__ = ["create_result_table", "save_result_table"]
from vis_tool.statistics.benchmark_plots import plot_citation_benchmark, plot_ppi_benchmark

__all__ = ["plot_citation_benchmark", "plot_ppi_benchmark"]
