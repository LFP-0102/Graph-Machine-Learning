"""可视化模块的全局样式与通用工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.paths import OUTPUT_DIR

DEFAULT_OUTPUT_DIR: Path = OUTPUT_DIR
DEFAULT_FIGSIZE: tuple[float, float] = (8, 6)
DEFAULT_DPI = 150
DEFAULT_CMAP_CATEGORICAL = "tab10"
DEFAULT_CMAP_SEQUENTIAL = "YlOrRd"
DEFAULT_ALPHA = 0.85
CATEGORICAL_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#E8A0A0", "#6BBF6F",
    "#C97140", "#688BCA", "#AE75A7", "#B7A58D", "#C5849D", "#82AAA0",
    "#CFB248", "#7C9ECE",
]


def set_style(style: str = "default") -> None:
    """应用全局 matplotlib 绘图样式。"""
    presets = {
        "default": {"figure.figsize": DEFAULT_FIGSIZE, "figure.dpi": DEFAULT_DPI, "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 10, "savefig.bbox": "tight", "savefig.dpi": DEFAULT_DPI},
        "seaborn": {"axes.facecolor": "#f0f0f0", "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.4, "axes.spines.top": False, "axes.spines.right": False},
        "dark": {"figure.facecolor": "#2e2e2e", "axes.facecolor": "#2e2e2e", "axes.edgecolor": "#cccccc", "axes.labelcolor": "#cccccc", "text.color": "#cccccc", "xtick.color": "#cccccc", "ytick.color": "#cccccc", "grid.color": "#555555", "legend.facecolor": "#3a3a3a", "legend.edgecolor": "#555555"},
    }
    if style in presets:
        matplotlib.rcParams.update(presets[style])
    else:
        plt.style.use(style)


def get_cmap(n_colors: int, categorical: bool = True, cmap_name: Optional[str] = None) -> list:
    """返回指定数量的绘图颜色。"""
    if categorical:
        return [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(n_colors)]
    cmap = plt.get_cmap(cmap_name or DEFAULT_CMAP_SEQUENTIAL)
    return [matplotlib.colors.rgb2hex(cmap(i / max(1, n_colors - 1))) for i in range(n_colors)]


def save_or_show(fig: plt.Figure, save_path: Optional[str] = None, dpi: int = DEFAULT_DPI, close: bool = True) -> plt.Figure:
    """保存图像；未给定路径时直接显示图像。"""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def to_numpy(tensor) -> np.ndarray:
    """将 PyTorch 张量或类数组对象安全转换为 NumPy 数组。"""
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(tensor)


set_style("default")
