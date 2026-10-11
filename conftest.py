"""pytest 根配置：把仓库根加入 sys.path，让 core 包可导入。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
