"""实验入口：按章调度 lesson。

用法：
    cd Demos/examples
    uv run --project .. python -m agent_lab list        列出各章主题
    uv run --project .. python -m agent_lab ch02        跑第 2 章
    uv run --project .. python -m agent_lab all         跑全部 17 章
    uv run --project .. python -m agent_lab selftest    跑自检
    uv run --project .. python -m agent_lab live ch02   用真实模型跑第 2 章

``--live`` 会打开真实模型路径（读 ``Demos/.env`` 里的 OpenAI 兼容端点）；
不加该开关时走确定性脚本路径，不联网、结果可复现。
"""

from __future__ import annotations

import os
import sys

from .lessons import LESSONS


def _enable_live() -> None:
    """打开真实模型开关。"""
    os.environ["AGENT_LAB_LIVE"] = "1"


def _run_chapter(key: str, live: bool) -> int:
    """跑指定章节。

    返回进程退出码：0 成功，非 0 表示断言失败或章节不存在。
    """
    for lesson in LESSONS:
        if lesson.key != key:
            continue
        module = lesson.load()
        if live:
            _enable_live()
            runner = getattr(module, "run_live", None)
            if runner is None:
                print(f"{lesson.key} 没有真实模型路径，跳过。")
                return 0
            runner()
        else:
            module.run()
        return 0
    print(f"未知章节：{key}")
    print("可用：" + " / ".join(lesson.key for lesson in LESSONS))
    return 2


def main(argv: list[str]) -> int:
    """命令行入口。

    返回进程退出码。
    """
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2

    command = args[0]
    live = "--live" in args

    if command == "list":
        print("Agent 开发教程实验（框架无关）：")
        for lesson in LESSONS:
            print(f"  {lesson.key}  {lesson.title}")
        return 0

    if command == "selftest":
        from .selftest import run as run_selftest

        return run_selftest()

    if command == "all":
        failures: list[str] = []
        for lesson in LESSONS:
            print()
            try:
                lesson.load().run()
            except AssertionError as exc:
                failures.append(f"{lesson.key}（{lesson.title}）：{exc}")
                print(f"\n!!! {lesson.key} 断言失败：{exc}")
            except Exception as exc:  # noqa: BLE001 - 单章出错不该中断整轮
                failures.append(f"{lesson.key}（{lesson.title}）：{type(exc).__name__}: {exc}")
                print(f"\n!!! {lesson.key} 运行失败：{type(exc).__name__}: {exc}")
        print()
        print("=" * 72)
        if failures:
            print(f"全部实验结束：{len(failures)}/{len(LESSONS)} 章未通过")
            for item in failures:
                print(f"  - {item}")
            return 1
        print(f"全部实验结束：{len(LESSONS)}/{len(LESSONS)} 通过")
        return 0

    if command == "live":
        rest = [arg for arg in args[1:] if not arg.startswith("--")]
        if not rest:
            failures = []
            for lesson in LESSONS:
                print()
                try:
                    module = lesson.load()
                except Exception as exc:  # noqa: BLE001 - 缺章节也要报出来
                    failures.append(f"{lesson.key}：导入失败 {type(exc).__name__}: {exc}")
                    print(f"\n!!! {lesson.key} 导入失败：{exc}")
                    continue
                runner = getattr(module, "run_live", None)
                if runner is None:
                    print(f"{lesson.key} 没有真实模型路径，跳过。")
                    continue
                try:
                    _enable_live()
                    runner()
                except AssertionError as exc:
                    failures.append(f"{lesson.key}：{exc}")
                    print(f"\n!!! {lesson.key} 断言失败：{exc}")
                except Exception as exc:  # noqa: BLE001 - 单章出错不该中断整轮
                    failures.append(f"{lesson.key}：{type(exc).__name__}: {exc}")
                    print(f"\n!!! {lesson.key} 运行失败：{type(exc).__name__}: {exc}")
            print()
            print("=" * 72)
            if failures:
                print(f"真实模型路径：{len(failures)} 章未通过")
                for item in failures:
                    print(f"  - {item}")
                return 1
            print("真实模型路径：全部通过")
            return 0
        return _run_chapter(rest[0], live=True)

    return _run_chapter(command, live=live)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
