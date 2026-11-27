"""实验入口：按章调度 lesson。

用法：
    cd Demos/examples
    uv run --project .. python -m vibe_film_lab list         列出各章主题
    uv run --project .. python -m vibe_film_lab ch07          跑第 7 章
    uv run --project .. python -m vibe_film_lab all           跑全部 6 章实验
    uv run --project .. python -m vibe_film_lab selftest      跑自检

需要本机装了 ego lite 与 ffmpeg 才能跑渲染相关的章节；
没装的话那几章会打印"跳过"并继续，不会中断整轮。
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
        print(f"Vibe 知识动画教程实验（{len(LESSONS)} 章）：")
        for lesson in LESSONS:
            print(f"  {lesson.key}  {lesson.title}")
        return 0

    if command == "selftest":
        from .selftest import run as run_selftest

        return run_selftest()

    if command == "all":
        return _run_all()

    return _run_chapter(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
