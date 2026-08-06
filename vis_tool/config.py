#!/usr/bin/env python
# @File       : config.py
# @Path       : vis_tool/config.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   vis_tool 全局配置：默认样式、配色方案、通用工具函数。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Global configuration and utilities for the vis_tool module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 默认输出目录
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"

# ---------------------------------------------------------------------------
# 全局默认参数
# ---------------------------------------------------------------------------
DEFAULT_FIGSIZE: tuple[float, float] = (8, 6)
DEFAULT_DPI: int = 150
DEFAULT_CMAP_CATEGORICAL: str = "tab10"
DEFAULT_CMAP_SEQUENTIAL: str = "YlOrRd"
DEFAULT_ALPHA: float = 0.85

# 分类配色（最多 20 类）
CATEGORICAL_PALETTE: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E8A0A0", "#6BBF6F", "#C97140", "#688BCA", "#AE75A7",
    "#B7A58D", "#C5849D", "#82AAA0", "#CFB248", "#7C9ECE",
]


# ---------------------------------------------------------------------------
# 样式设置
# ---------------------------------------------------------------------------
def set_style(style: str = "default") -> None:
    """Apply a global matplotlib style.

    Parameters
    ----------
    style : str
        One of 'default', 'seaborn', 'dark', or any valid
        ``matplotlib.style`` name.
    """
    presets = {
        "default": {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": DEFAULT_DPI,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": DEFAULT_DPI,
        },
        "seaborn": {
            "axes.facecolor": "#f0f0f0",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
        "dark": {
            "figure.facecolor": "#2e2e2e",
            "axes.facecolor": "#2e2e2e",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#cccccc",
            "text.color": "#cccccc",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "grid.color": "#555555",
            "legend.facecolor": "#3a3a3a",
            "legend.edgecolor": "#555555",
        },
    }

    if style in presets:
        matplotlib.rcParams.update(presets[style])
    else:
        plt.style.use(style)

    # 尝试注册中文字体 fallback（无中文字体时不报错）
    try:
        from matplotlib.font_manager import FontProperties  # noqa: F811
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 颜色工具
# ---------------------------------------------------------------------------
def get_cmap(
    n_colors: int,
    categorical: bool = True,
    cmap_name: Optional[str] = None,
) -> list:
    """Return a list of *n_colors* hex codes.

    Parameters
    ----------
    n_colors : int
        Number of colours needed.
    categorical : bool
        If True, cycle through a categorical palette; otherwise use a
        sequential colormap.
    cmap_name : str, optional
        Override the default colormap name.
    """
    if categorical:
        # 循环使用预定义调色板
        palette = CATEGORICAL_PALETTE
        return [palette[i % len(palette)] for i in range(n_colors)]

    name = cmap_name or DEFAULT_CMAP_SEQUENTIAL
    cmap = plt.get_cmap(name)
    return [matplotlib.colors.rgb2hex(cmap(i / max(1, n_colors - 1)))
            for i in range(n_colors)]


# ---------------------------------------------------------------------------
# 通用保存 / 显示
# ---------------------------------------------------------------------------
def save_or_show(
    fig: plt.Figure,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> plt.Figure:
    """Save *fig* to *save_path* or display it, then optionally close.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save / show.
    save_path : str, optional
        File path.  Directories are created automatically.  If ``None``,
        ``plt.show()`` is called instead.
    dpi : int
        Resolution used when saving.
    close : bool
        If True, close the figure after saving/showing.
    """
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    if close:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 通用张量 → numpy 转换
# ---------------------------------------------------------------------------
def to_numpy(tensor) -> np.ndarray:
    """Safely convert a torch tensor or array-like to a numpy array."""
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(tensor)


# 应用默认样式
set_style("default")
