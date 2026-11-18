"""实验入口：按周调度实验。

用法：
    cd Demos/examples
    uv run --project .. python -m ai_pm_lab week1        跑某一周
    uv run --project .. python -m ai_pm_lab all           跑全部八周
    uv run --project .. python -m ai_pm_lab list          列出八周主题
    uv run --project .. python -m ai_pm_lab selftest      跑自检
    uv run --project .. python -m ai_pm_lab notes        校验教程笔记与实验的对应关系

`all` 逐周跑，某周断言失败不中断整轮，最后汇总 N/8 通过，有失败退出码为 1。
"""
from __future__ import annotations

import sys

from .weeks import WEEKS, load_week


def _run_single(key: str) -> int:
    """跑指定的一周，返回进程退出码。

    断言失败也要返回非 0：单周命令被用在脚本里时，退出码是唯一的信号。
    """
    module = load_week(key)
    if module is None:
        print(f"未知参数：{key}")
        print("可用：" + " / ".join(week_key for week_key, _, _ in WEEKS))
        return 2
    try:
        module.run()
    except AssertionError as exc:
        print(f"\n!!! {key} 断言失败：{exc}")
        return 1
    return 0


def main(argv: list[str]) -> int:
    """命令行入口，返回进程退出码。"""
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2

    command = args[0]

    if command == "list":
        print("AI 产品经理 · 八周实验：")
        for key, title, _ in WEEKS:
            print(f"  {key}  {title}")
        return 0

    if command == "selftest":
        from .selftest import run as run_selftest

        return run_selftest()

    if command == "notes":
        from .notes_check import run as run_notes_check

        return run_notes_check()

    if command == "all":
        failures: list[str] = []
        for key, title, _ in WEEKS:
            print()
            try:
                module = load_week(key)
                if module is None:
                    raise AssertionError("模块缺失")
                module.run()
            except AssertionError as exc:
                failures.append(f"{key}（{title}）：{exc}")
                print(f"\n!!! {key} 断言失败：{exc}")
            except Exception as exc:  # noqa: BLE001 - 单周出错不该中断整轮
                failures.append(f"{key}（{title}）：{type(exc).__name__}: {exc}")
                print(f"\n!!! {key} 运行失败：{type(exc).__name__}: {exc}")
        print()
        print("=" * 78)
        if failures:
            print(f"全部实验结束：{len(WEEKS) - len(failures)}/{len(WEEKS)} 通过")
            for item in failures:
                print(f"  - {item}")
            return 1
        print(f"全部实验结束：{len(WEEKS)}/{len(WEEKS)} 通过")
        return 0

    return _run_single(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
