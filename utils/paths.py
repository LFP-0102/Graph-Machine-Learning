from pathlib import Path

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "CONFIG_DIR",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "OUTPUT_DIR",
    "RUN_DIR",
    "VIS_DIR",
    "ensure_dirs",
]

# ── 项目根目录 ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── 输入 / 静态资源 ─────────────────────────────────────────
DATA_DIR   = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "configs"

# ── 持久化产物 ──────────────────────────────────────────────
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs"

# ── 输出 ────────────────────────────────────────────────────
OUTPUT_DIR = ROOT_DIR / "outputs"
RUN_DIR    = OUTPUT_DIR / "runs"              # 每次训练的运行结果
VIS_DIR    = OUTPUT_DIR / "visualizations"    # 图表 / 嵌入可视化


def ensure_dirs(*dirs: Path) -> None:
    """按需创建目录（惰性调用，不在 import 时自动执行）。"""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
