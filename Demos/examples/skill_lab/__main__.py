"""实验入口：按天调度 lesson。

用法：
    python -m skill_lab day1        跑某一天
    python -m skill_lab all         跑全部七天
    python -m skill_lab list        列出每天的主题
    python -m skill_lab selftest    跑自检
"""
from __future__ import annotations

import sys

from .lessons import DAYS


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2

    command = args[0]

    if command == "list":
        print("七天实验：")
        for key, title, _ in DAYS:
            print(f"  {key}  {title}")
        return 0

    if command == "selftest":
        from .selftest import run as run_selftest
        return run_selftest()

    if command == "all":
        failures: list[str] = []
        for key, title, module in DAYS:
            print()
            try:
                module.run()  # type: ignore[attr-defined]
            except AssertionError as exc:
                failures.append(f"{key}（{title}）：{exc}")
                print(f"\n!!! {key} 断言失败：{exc}")
        print()
        print("=" * 78)
        if failures:
            print(f"全部实验结束：{len(failures)} 天未通过")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("全部实验结束：7/7 通过")
        return 0

    for key, title, module in DAYS:
        if key == command:
            module.run()  # type: ignore[attr-defined]
            return 0

    print(f"未知参数：{command}")
    print("可用：day1..day7 / all / list / selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
