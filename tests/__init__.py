"""测试辅助：把某个案例目录的 main.py 以独立模块名加载，避免不同案例的 `main` 模块相互污染。"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # 仓库根


def load_main(case_dir: str):
    """加载 cases/<case_dir>/main.py，返回模块对象。case_dir 如 '01_tool_call_parse'。"""
    path = ROOT / "cases" / case_dir / "main.py"
    if not path.exists():
        raise FileNotFoundError(f"未找到案例: {path}")
    mod_name = f"case_{case_dir.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
