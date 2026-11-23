"""llm_lab 自检：验证实验骨架和笔记一样都没坏。

这个模块解决什么问题：教程仓库最容易烂在"笔记写了、实验跑不起来"或者
"实验改了、笔记还对不上"。自检负责把三类漂移挡住：

1. **结构漂移**——8 章笔记与 8 个实验模块必须一一对应，且都存在、都可导入。
2. **行为漂移**——每章的确定性路径必须跑通断言。
3. **立教漂移**——这套教程的立身之本是"贴着协议写"，所以要检查实验代码里
   **没有**偷�� import langchain / langgraph。真要用框架，第 8 章会明说。

不依赖 pytest、不联网、不调模型：

    cd Demos/examples
    uv run --project .. python -m llm_lab.selftest
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from .lessons import LESSONS

# 笔记文件与章节编号一一对应；名字改了这里也要改
NOTE_FILES: tuple[str, ...] = (
    "01-模型调用基础与消息协议.md",
    "02-工具调用协议.md",
    "03-结构化输出.md",
    "04-流式输出.md",
    "05-推理模型与思考预算.md",
    "06-多模态输入输出.md",
    "07-上下文窗口、定价与缓存.md",
    "08-模型选型与迁移纪律.md",
)

NOTES_DIR = Path(__file__).resolve().parents[3] / "AI" / "LLM"

# 这套教程禁止出现的框架
FORBIDDEN_IMPORTS: tuple[str, ...] = ("langchain", "langgraph", "deepagents")


def check_notes_exist() -> list[str]:
    """8 篇笔记必须都在，且课程地图里都有入口链接。"""
    failures: list[str] = []
    for name in NOTE_FILES:
        path = NOTES_DIR / name
        if not path.is_file():
            failures.append(f"缺笔记：AI/LLM/{name}")
            continue
        if path.stat().st_size < 2000:
            failures.append(f"笔记过短，疑似没写完：AI/LLM/{name}")

    readme = NOTES_DIR / "README.md"
    if not readme.is_file():
        failures.append("缺课程地图：AI/LLM/README.md")
    else:
        text = readme.read_text(encoding="utf-8")
        for name in NOTE_FILES:
            if name not in text:
                failures.append(f"课程地图没有链接到：{name}")
    return failures


def check_lessons_importable() -> list[str]:
    """每个实验模块都要能导入，且导出 ``run()``。"""
    failures: list[str] = []
    for lesson in LESSONS:
        try:
            module = importlib.import_module(f"llm_lab.lessons.{lesson.module}")
        except Exception as exc:  # noqa: BLE001 - 自检要报出任何导入问题
            failures.append(f"{lesson.key} 导入失败：{type(exc).__name__}: {exc}")
            continue
        if not callable(getattr(module, "run", None)):
            failures.append(f"{lesson.key} 缺少 run()")
        if not (module.__doc__ or "").strip():
            failures.append(f"{lesson.key} 模块没有中文说明（docstring）")
    return failures


def check_framework_free() -> list[str]:
    """检查实验代码里没有偷用 Agent 框架。

    只做静态语法检查（不 import 执行），所以跑得快，也不会触发副作用。
    """
    failures: list[str] = []
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append(f"{path.name} 语法错误：{exc}")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    rel = path.relative_to(package_dir)
                    failures.append(
                        f"{rel} 第 {node.lineno} 行 import 了 {name}——"
                        "这套教程必须贴着协议写，不许偷用框架"
                    )
    return failures


def run_lessons() -> list[str]:
    """逐章跑确定性路径，把断言失败收集起来。"""
    failures: list[str] = []
    for lesson in LESSONS:
        try:
            module = importlib.import_module(f"llm_lab.lessons.{lesson.module}")
        except Exception as exc:  # noqa: BLE001 - 缺章节也要报出来而不是崩掉
            failures.append(f"{lesson.key}（{lesson.title}）：导入失败 {type(exc).__name__}: {exc}")
            continue
        try:
            module.run()
        except AssertionError as exc:
            failures.append(f"{lesson.key}（{lesson.title}）：{exc}")
        except Exception as exc:  # noqa: BLE001 - 自检要暴露运行期错误
            failures.append(f"{lesson.key}（{lesson.title}）：{type(exc).__name__}: {exc}")
    return failures


def run() -> int:
    """跑完整自检。

    返回进程退出码：0 全部通过，1 有失败项。
    """
    failures: list[str] = []

    print("检查笔记与实验的对应关系……")
    failures.extend(check_notes_exist())

    print("检查实验模块可导入……")
    failures.extend(check_lessons_importable())

    print("检查实验是否保持框架无关……")
    failures.extend(check_framework_free())

    print("逐章跑确定性断言……")
    failures.extend(run_lessons())

    print()
    print("=" * 72)
    if failures:
        print(f"自检未通过：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"自检通过：{len(LESSONS)} 章实验 + {len(NOTE_FILES)} 篇笔记")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run())
