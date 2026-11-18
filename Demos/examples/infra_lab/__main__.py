"""实验入口：按章调度 lesson。

用法：
    cd Demos/examples
    uv run --project .. python -m infra_lab list         列出各章主题
    uv run --project .. python -m infra_lab ch07          跑第 7 章
    uv run --project .. python -m infra_lab all           跑全部 18 章
    uv run --project .. python -m infra_lab selftest      跑自检
    uv run --project .. python -m infra_lab capstone       跑毕业作品
    uv run --project .. python -m infra_lab capstone.selftest

全部实验离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import sys

from .lessons import LESSONS


def _run_chapter(key: str) -> int:
    """跑指定章节。

    返回进程退出码：0 成功，非 0 表示断言失败或章节不存在。
    """
    for lesson in LESSONS:
        if lesson.key != key:
            continue
        lesson.load().run()
        return 0
    print(f"未知章节：{key}")
    print("可用：" + " / ".join(lesson.key for lesson in LESSONS))
    return 2


def _run_all() -> int:
    """跑全部章节，收集失败项而不是中断。"""
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


def main(argv: list[str]) -> int:
    """命令行入口。

    返回进程退出码。
    """
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2

    command = args[0]

    if command == "list":
        print("AI Infra 推理服务教程实验（18 章 · 离线可跑）：")
        for lesson in LESSONS:
            print(f"  {lesson.key}  {lesson.title}")
        print("\n  毕业作品  capstone  迷你推理服务（分页 KV + 连续批处理 + 前缀缓存）")
        return 0

    if command == "selftest":
        from .selftest import run as run_selftest

        return run_selftest()

    if command == "capstone":
        from .capstone import run as run_capstone

        return run_capstone()

    if command == "all":
        return _run_all()

    if command == "capstone.selftest":
        from .capstone.selftest import run as run_capstone_selftest

        return run_capstone_selftest()

    return _run_chapter(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
