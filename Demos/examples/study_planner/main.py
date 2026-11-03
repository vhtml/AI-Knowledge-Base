"""命令行入口：交互对话、单句执行。

必须在 Demos/examples 目录下运行（包在 examples 下）：

    cd Demos/examples
    uv run --project .. python -m study_planner                # 交互模式
    uv run --project .. python -m study_planner --task "我想学 Python"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import StepRecord, StudyPlanner
from .storage import Storage

BANNER = """┌──────────────────────────────────────────────┐
│  学习规划师 · 目标 / 计划 / 复盘               │
│  重点：短期记忆 + 长期记忆                    │
└──────────────────────────────────────────────┘
试试说：
  我想学 Python 数据分析        → 记录目标
  生成计划                       → 排计划
  今天完成了一项                → 标记完成
  复盘                           → 每日复盘
  我的进度                      → 查询
  /memory                       → 看记忆统计
  /rollover                     → 模拟跨天
"""

HELP_TEXT = """可用命令：
  /help      显示本帮助
  /memory    查看两级记忆统计
  /recall 关键词   查看召回结果
  /today     查看今天的待办
  /rollover  模拟跨天：清空短期记忆，沉淀到长期
  /exit      退出

也可以直接说话：
  我想学 XX           记录目标
  生成计划            按目标排计划
  完成 / 做完了        标记今天的下一件事完成
  复盘                生成今日复盘并沉淀结论
  我的进度            查看进度和相关记忆
"""


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(
        prog="study_planner",
        description="学习规划师：记录目标、生成计划、每日复盘。重点在短期与长期记忆。",
    )

    parser.add_argument("--task", metavar="TEXT", help="只跑一句话，打印结果后退出")
    parser.add_argument("--data-dir", metavar="DIR", help="数据目录，默认 ~/.study-planner")
    parser.add_argument("--quiet", action="store_true", help="不打印决策过程，只打印回答")

    return parser


def make_reporter(quiet: bool):
    """把每一步的留痕打印出来——记忆动了哪里，一目了然。"""

    if quiet:
        return lambda step: None

    def report(step: StepRecord) -> None:
        mark = "✓" if step.ok else "✗"
        detail = f" · {step.detail}" if step.detail else ""
        print(f"  · [{step.phase}] {step.title} {mark}{detail}")

    return report


def build_planner(args: argparse.Namespace, root: Path) -> StudyPlanner:
    """装配规划师：存储 → 记忆 → 数据 → 编排。"""

    storage = Storage(root)
    memory = storage.build_memory()

    return StudyPlanner(
        memory=memory,
        goals=storage.load_goals(),
        items=storage.load_items(),
        reviews=storage.load_reviews(),
        reporter=make_reporter(args.quiet),
    )


def persist(planner: StudyPlanner, root: Path) -> None:
    """把这一轮的变化落盘。"""

    Storage(root).save_all(planner.memory, planner.goals, planner.items, planner.reviews)


def run_task(planner: StudyPlanner, args: argparse.Namespace, root: Path) -> int:
    """跑一句话，打印结果。"""

    turn = planner.handle(args.task)
    print(turn.reply)
    persist(planner, root)

    return 0


def run_interactive(planner: StudyPlanner, args: argparse.Namespace, root: Path) -> int:
    """交互模式。"""

    print(BANNER)

    while True:
        try:
            line = input("你 › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not line:
            continue

        if line.lower() in {"exit", "quit", "退出"}:
            print("再见。")
            break

        if line.startswith("/"):
            if not handle_command(planner, line, root):
                break
            continue

        turn = planner.handle(line)
        print(f"\n{turn.reply}\n")
        persist(planner, root)

    return 0


def handle_command(planner: StudyPlanner, command: str, root: Path) -> bool:
    """处理斜杠命令。返回 False 表示退出主循环。"""

    parts = command.split()
    name = parts[0].lower()

    if name == "/help":
        print(HELP_TEXT)

    elif name == "/memory":
        stats = planner.memory.stats()
        print("记忆统计：")
        print(f"  短期记忆：{stats['short']} 条（折叠 {stats['short_folded']} 次）")
        print(f"  长期记忆：{stats['long']} 条"
              f"（有效 {stats['long_active']}，被取代 {stats['long_superseded']}）")
        print(f"  晋升次数：{stats['promotions']}")

        for item in planner.memory.long.items:
            mark = "→" if item.superseded_by else " "
            print(f"   {mark} [{item.category}] {item.content}（提及 {item.mentions}）")

    elif name == "/recall" and len(parts) > 1:
        keyword = " ".join(parts[1:])
        bundle = planner.memory.recall(keyword)

        if not bundle.hits:
            print(f"没有召回和「{keyword}」相关的记忆。")
        else:
            print(f"召回「{keyword}」：")

            for hit in bundle.hits:
                print(f"  - {hit.render()}")

    elif name == "/today":
        today = planner.today_items()

        if not today:
            print("今天没有排任务。")
        else:
            print("今天的待办：")

            for item in today:
                mark = "✓" if item.done else "○"
                print(f"  {mark} {item.title}（{item.minutes} 分钟）")

    elif name == "/rollover":
        moved = planner.rollover()
        print(f"跨天完成：短期工作区已清空，{len(moved)} 条结论沉淀到长期记忆。")

        for content in moved:
            print(f"  - {content}")

        persist(planner, root)

    elif name in {"/exit", "/quit"}:
        print("再见。")
        return False

    else:
        print(f"未知命令 {name}，输入 /help 查看用法。")

    return True


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""

    args = build_parser().parse_args(argv)

    if args.data_dir:
        root = Path(args.data_dir).expanduser()
    else:
        from .storage import DEFAULT_ROOT

        root = DEFAULT_ROOT

    planner = build_planner(args, root)

    if args.task:
        return run_task(planner, args, root)

    return run_interactive(planner, args, root)


if __name__ == "__main__":
    raise SystemExit(main())
