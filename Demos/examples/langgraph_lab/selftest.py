"""langgraph_lab 自检：验证实验骨架和笔记一样都没坏。

这个模块解决什么问题：教程仓库最容易烂在"笔记写了、实验跑不起来"或者
"实验改了、笔记还对不上"。自检负责把两类漂移挡住：

1. **结构漂移**——17 章笔记与 17 个实验模块必须一一对应，且都存在、都可导入。
2. **行为漂移**——每章的确定性路径必须跑通断言。

不依赖 pytest、不联网、不调模型：

    cd Demos/examples
    uv run --project .. python -m langgraph_lab.selftest
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .lessons import LESSONS

# 笔记文件与章节编号一一对应；名字改了这里也要改
NOTE_FILES: tuple[str, ...] = (
    "01-生态全景与选型.md",
    "02-图模型：节点、边与共享状态.md",
    "03-状态定义与归约器.md",
    "04-控制流：分支、循环、Send 与 Command.md",
    "05-节点韧性：重试、超时与错误处理.md",
    "06-工具调用与工具节点.md",
    "07-持久化：检查点与时间旅行.md",
    "08-人在环路：中断、审批与恢复.md",
    "09-流式与事件投影.md",
    "10-子图与多智能体编排.md",
    "11-函数式 API 与 Pregel 运行时.md",
    "12-LangChain 1.x：create_agent 与中间件.md",
    "13-MCP 接入与工具生态.md",
    "14-Deep Agents：harness 视角.md",
    "15-可观测性与评测.md",
    "16-部署：本地服务器、Studio 与生产.md",
    "17-迁移与工程实践.md",
)

NOTES_DIR = Path(__file__).resolve().parents[3] / "AI" / "LangGraph"


def check_notes_exist() -> list[str]:
    """17 篇笔记必须都在，且互相之间有入口链接。"""
    failures: list[str] = []
    for name in NOTE_FILES:
        path = NOTES_DIR / name
        if not path.is_file():
            failures.append(f"缺笔记：AI/LangGraph/{name}")
            continue
        if path.stat().st_size < 2000:
            failures.append(f"笔记过短，疑似没写完：AI/LangGraph/{name}")

    readme = NOTES_DIR / "README.md"
    if not readme.is_file():
        failures.append("缺课程地图：AI/LangGraph/README.md")
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
            module = importlib.import_module(f"langgraph_lab.lessons.{lesson.module}")
        except Exception as exc:  # noqa: BLE001 - 自检要报出任何导入问题
            failures.append(f"{lesson.key} 导入失败：{type(exc).__name__}: {exc}")
            continue
        if not callable(getattr(module, "run", None)):
            failures.append(f"{lesson.key} 缺少 run()")
        if not (module.__doc__ or "").strip():
            failures.append(f"{lesson.key} 模块没有中文说明（docstring）")
    return failures


def run_lessons() -> list[str]:
    """逐章跑确定性路径，把断言失败收集起来。"""
    failures: list[str] = []
    for lesson in LESSONS:
        module = importlib.import_module(f"langgraph_lab.lessons.{lesson.module}")
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
