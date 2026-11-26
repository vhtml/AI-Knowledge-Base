"""命令行入口：跑一遍笔记结构校验。

用法：
    cd Demos/examples
    uv run --project .. python -m notes_check

退出码 0 表示全部通过，1 表示有检查项失败。
"""
from __future__ import annotations

import sys

from .notes_check import run


def main(argv: list[str]) -> int:
    """命令行入口，返回进程退出码。"""
    if len(argv) > 1 and argv[1] in {"-h", "--help"}:
        print(__doc__)
        return 0
    print("AI 产品经理教程 · 笔记结构校验")
    return run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
