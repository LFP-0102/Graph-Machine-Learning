from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"


def ensure_dirs(*dirs: Path) -> None:
    """在实验需要时创建输出目录。"""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
