"""命令行入口：跑一轮对话、看决策过程、或者直接演一遍内置剧本。

必须在 Demos/examples 目录下运行（包在 examples 下）：

    cd Demos/examples

    uv run --project .. python -m life_assistant              # 交互模式（需要模型接口）
    uv run --project .. python -m life_assistant --offline    # 离线模式：不联网、不消耗 token
    uv run --project .. python -m life_assistant --task "北京今天天气怎么样"
    uv run --project .. python -m life_assistant --demo       # 内置演示剧本

从别的目录运行时用 PYTHONPATH 指向本目录：

    PYTHONPATH=Demos/examples python3 -m life_assistant --offline
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # 没装 python-dotenv 也能跑离线模式
    load_dotenv = None  # type: ignore[assignment]

from .agent import LifeAssistant, Reporter
from .models import StepRecord
from .planners import FallbackPlanner, LLMPlanner, RouterPlanner
from .providers import FakeWeatherProvider, OpenMeteoWeatherProvider
from .tools import build_registry

BANNER = """┌──────────────────────────────────────────────┐
│  生活助手 Agent · 天气 / 计算 / 时间            │
│  重点：工具选择 + 参数校验                      │
└──────────────────────────────────────────────┘
命令：/help 帮助  /tools 工具清单  /exit 退出
"""

HELP_TEXT = """可以这样问：
  北京今天天气怎么样          → 选 get_weather
  明天上海天气如何            → 选 get_weather（day=tomorrow）
  算一下 (12+8)*3             → 选 calculate
  北京时间现在是几点          → 选 get_time
  天气怎么样                  → 参数缺失，追问城市
"""

# 内置剧本：每个场景对应一类"工具选择 / 参数校验"问题
DEMO_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("北京今天天气怎么样", "选工具：城市 + 今天"),
    ("明天上海天气如何", "选工具：明天的预报"),
    ("算一下 (12+8)*3 等于多少", "选工具：抽出表达式再求值"),
    ("现在几点", "选工具：本机时区"),
    ("北京时间现在是几点", "选工具：时区别名"),
    ("三天后是哪天", "选工具：偏移几天"),
    ("北京天气怎么样，顺便算一下 3*4", "一句话两件事 → 两个工具"),
    ("天气怎么样", "参数缺失 → 追问城市"),
    ("算一下", "参数缺失 → 追问表达式"),
    ("算一下 100/0", "工具执行失败 → 如实报错"),
    ("查一下股票行情", "没有对应工具 → 说明能力边界"),
)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(
        prog="life_assistant",
        description="生活助手 Agent：天气 / 计算 / 时间三个工具，重点在工具选择和参数校验。",
    )

    parser.add_argument("--task", metavar="TEXT", help="只跑一句话，打印结果后退出")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：不联网、不消耗 token，用规则路由 + 假天气数据",
    )
    parser.add_argument("--demo", action="store_true", help="运行内置演示剧本（离线）")
    parser.add_argument("--model", metavar="NAME", help="模型名，覆盖 LLM_MODEL")
    parser.add_argument("--base-url", metavar="URL", help="接口地址，覆盖 LLM_BASE_URL")
    parser.add_argument("--api-key", metavar="KEY", help="API Key，覆盖 LLM_API_KEY")
    parser.add_argument("--quiet", action="store_true", help="不打印决策过程，只打印回答")

    return parser


def load_env_config() -> dict[str, str]:
    """从环境变量读取模型配置（.env 由调用方加载）。"""

    return {
        "base_url": os.getenv("LLM_BASE_URL") or "",
        "api_key": os.getenv("LLM_API_KEY") or "",
        "model": os.getenv("LLM_MODEL") or "gpt-4o-mini",
    }


def make_reporter(quiet: bool) -> Reporter:
    """把决策过程打印出来——这是本示例最值得看的部分。"""

    if quiet:
        return lambda step: None

    def report(step: StepRecord) -> None:
        mark = "✓" if step.ok else "✗"
        detail = f" · {step.detail}" if step.detail else ""
        print(f"  · [{step.phase}] {step.title} {mark}{detail}")

    return report


def build_planner(args: argparse.Namespace, registry, offline: bool):
    """选择规划器：离线用规则路由，在线用模型，模型不可用自动退回规则路由。"""

    router_planner = RouterPlanner(registry)

    if offline:
        return router_planner

    config = load_env_config()
    model = args.model or config["model"]
    base_url = args.base_url or config["base_url"] or None
    api_key = args.api_key or config["api_key"] or None

    if not api_key:
        print(
            "[提示] 没有找到 API Key，已切到离线规则路由。\n"
            "        想用模型的话，用 `uv run --project ..` 运行（会自动加载 Demos/.env）。",
            file=sys.stderr,
        )
        return router_planner

    try:
        primary = LLMPlanner(
            model=model,
            base_url=base_url,
            api_key=api_key,
            tool_schemas=registry.to_openai(),
        )
    except ImportError as exc:
        print(
            f"[提示] 缺少依赖 {exc.name}，已切到离线规则路由。\n"
            "        想用模型的话，先 `cd Demos && uv sync`。",
            file=sys.stderr,
        )
        return router_planner

    return FallbackPlanner(primary, router_planner)


def build_assistant(args: argparse.Namespace, offline: bool) -> LifeAssistant:
    """装配 Agent：数据源 → 工具 → 规划器 → 留痕输出。"""

    provider = FakeWeatherProvider() if offline else OpenMeteoWeatherProvider()
    registry = build_registry(provider)
    planner = build_planner(args, registry, offline)

    return LifeAssistant(planner, registry, reporter=make_reporter(args.quiet))


def run_task(assistant: LifeAssistant, text: str) -> int:
    """跑一句话，打印结果。"""

    turn = assistant.handle(text)
    print(turn.reply)

    return 0


def run_interactive(assistant: LifeAssistant) -> int:
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
            if not handle_command(line):
                break
            continue

        turn = assistant.handle(line)
        print(f"\n{turn.reply}\n")

    return 0


def handle_command(command: str) -> bool:
    """处理斜杠命令。返回 False 表示退出主循环。"""

    parts = command.split()
    name = parts[0].lower()

    if name == "/help":
        print(HELP_TEXT)

    elif name == "/tools":
        print("可用工具：get_weather（天气）、calculate（计算）、get_time（时间）")

    elif name in {"/exit", "/quit"}:
        print("再见。")
        return False

    else:
        print(f"未知命令 {name}，输入 /help 查看用法。")

    return True


def run_demo(assistant: LifeAssistant) -> int:
    """跑内置剧本，把两类重点问题都演一遍。"""

    print(BANNER)
    print("离线演示剧本（规则路由 + 假天气数据）：\n")

    for index, (text, expectation) in enumerate(DEMO_SCENARIOS, start=1):
        print("─" * 56)
        print(f"[{index}/{len(DEMO_SCENARIOS)}] {text}")
        print(f"      预期：{expectation}")

        turn = assistant.handle(text)
        print(f"  答：{turn.reply}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""

    args = build_parser().parse_args(argv)

    # .env 放在 Demos/ 下，load_dotenv 会向上查找
    if load_dotenv is not None:
        load_dotenv()

    offline = bool(args.offline or args.demo)
    assistant = build_assistant(args, offline)

    if args.demo:
        return run_demo(assistant)

    if args.task:
        return run_task(assistant, args.task)

    return run_interactive(assistant)


if __name__ == "__main__":
    raise SystemExit(main())
