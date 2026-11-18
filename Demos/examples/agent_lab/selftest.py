"""agent_lab 自检：验证实验骨架和笔记一样都没坏。

这个模块解决什么问题：教程仓库最容易烂在"笔记写了、实验跑不起来"或者
"实验改了、笔记还对不上"。自检负责把三类漂移挡住：

1. **结构漂移**——17 章笔记与 17 个实验模块必须一一对应，且都存在、都可导入。
2. **行为漂移**——每章的确定性路径必须跑通断言。
3. **立教漂移**——这套教程的立身之本是"框架无关、从零手写"，所以要检查
   实验代码里**没有**偷偷 import langchain / langgraph。一旦有人图省事
   引入框架，读者就再也看不见"框架到底替我做了什么"。

不依赖 pytest、不联网、不调模型：

    cd Demos/examples
    uv run --project .. python -m agent_lab.selftest
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from .lessons import LESSONS

# 笔记文件与章节编号一一对应；名字改了这里也要改
NOTE_FILES: tuple[str, ...] = (
    "01-什么是 Agent：定义、边界与最小内核.md",
    "02-Agent 循环：ReAct 的手写实现.md",
    "03-工具设计：schema、错误面与粒度.md",
    "04-系统提示与角色设计.md",
    "05-上下文工程：窗口预算与压缩.md",
    "06-记忆系统：短期、长期与检索.md",
    "07-任务分解：规划模式与自适应分解.md",
    "08-反思与自纠错.md",
    "09-多智能体编排模式.md",
    "10-可靠性：重试、超时、幂等与降级.md",
    "11-可观测性：事件模型、追踪与回放.md",
    "12-评测体系：黄金集、轨迹评测与回归.md",
    "13-安全：提示注入、沙箱与最小权限.md",
    "14-人在环路：审批、接管与恢复.md",
    "15-成本与延迟工程.md",
    "16-生产化：部署形态与上线检查单.md",
    "17-框架地图：从手写运行时到 LangGraph 与 Deep Agents.md",
)

NOTES_DIR = Path(__file__).resolve().parents[3] / "AI" / "Agent"

# 这套教程禁止出现的框架——第 17 章才允许在**文字**里讨论它们
FORBIDDEN_IMPORTS: tuple[str, ...] = ("langchain", "langgraph", "deepagents")


def check_notes_exist() -> list[str]:
    """17 篇笔记必须都在，且互相之间有入口链接。"""
    failures: list[str] = []
    for name in NOTE_FILES:
        path = NOTES_DIR / name
        if not path.is_file():
            failures.append(f"缺笔记：AI/Agent/{name}")
            continue
        if path.stat().st_size < 2000:
            failures.append(f"笔记过短，疑似没写完：AI/Agent/{name}")

    readme = NOTES_DIR / "README.md"
    if not readme.is_file():
        failures.append("缺课程地图：AI/Agent/README.md")
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
            module = importlib.import_module(f"agent_lab.lessons.{lesson.module}")
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
    第 17 章要讲框架，但那一章也只允许在**文字**里对比，不允许 import。
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
                        "这套教程必须框架无关，框架留到第 17 章"
                    )
    return failures


def run_lessons() -> list[str]:
    """逐章跑确定性路径，把断言失败收集起来。"""
    failures: list[str] = []
    for lesson in LESSONS:
        try:
            module = importlib.import_module(f"agent_lab.lessons.{lesson.module}")
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
