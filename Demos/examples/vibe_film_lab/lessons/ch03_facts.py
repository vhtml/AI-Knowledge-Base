"""第 3 章实验：事实表与出处。

这个模块解决什么问题：让"每个事实都带出处"变成一道**双向**的检查。

只查一个方向是不够的。正向查（字幕里的数有没有出处）能防住编造；
反向查（事实表里的数有没有上屏）能防住另一种更隐蔽的翻车——
事实表里备了 5 个关键数据，字幕只用了 2 个，片子看着完整，其实少讲了一半。

还有一个容易被忽略的点：**示意**。有些画面上的数字是为了讲清道理而编的
（比如示意用的 220 Hz）。这类数字不是错，但必须标成"示意"，
否则观众会把它当成测量结果。

被检查的对象是**毕业作品自己**——分镜表里的事实表 ↔ 成片 ``film.js`` 里的字幕。
不是拿一份样例数据自证，而是拿真片当被检查对象：
数字进了字幕却查不到出处，或者事实表写了却没上屏，两个方向都会报出来。

对应笔记：AI/Knowledge-Film/03-事实表与出处.md
"""

from __future__ import annotations

from pathlib import Path

from ..labkit import (
    Fact,
    ensure,
    fact_audit,
    fmt_table,
    load_markdown_storyboard,
    parse_film_subs,
)

PROJECT = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C"
FILM_JS = PROJECT / "film.js"
STORYBOARD_MD = PROJECT / "storyboard.md"


def _where_on_screen(value: str, subs: list[dict]) -> str:
    """这个值出现在第几句字幕里。"""
    for index, sub in enumerate(subs, start=1):
        if value in sub["text"]:
            return f"第 {index} 句"
    return "没上屏"


def run() -> None:
    print("=" * 78)
    print("第 3 章 · 事实表：每个数字都要有出处")
    print("=" * 78)

    storyboard = load_markdown_storyboard(STORYBOARD_MD)
    facts = storyboard.facts
    subs = parse_film_subs(FILM_JS)

    print()
    print(f"【1】选题：{storyboard.topic}")
    print(f"    论点：{storyboard.thesis}")
    print()
    print("【2】成片的事实表（来自 storyboard.md，和代码同源）")
    print(fmt_table(
        ["事实", "上屏的值", "类型", "出处"],
        [
            [
                fact.key,
                fact.value,
                "示意" if fact.illustrative else ("不上屏" if not fact.on_screen else "实测/史实"),
                fact.source,
            ]
            for fact in facts
        ],
    ))

    print()
    print("【3】双向核对：成片字幕 ↔ 事实表")
    print(fmt_table(
        ["事实", "上屏的值", "出现在"],
        [
            [fact.key, fact.value, _where_on_screen(fact.value, subs)]
            for fact in facts
        ],
    ))

    problems = fact_audit(subs, facts)
    for problem in problems:
        print(f"  ✗ {problem}")
    ensure(not problems, f"成片没过事实双向核对：{problems}")
    print(f"  ✓ {len(subs)} 句字幕里的每个数字都能查到出处，事实表里的数字也都上了屏。")

    # ---- 故意做坏 ----
    print()
    print("【4】故意做坏，看核对抓不抓得住")
    only_year = [Fact(key="年份", value="1822", source="出处", illustrative=False)]

    missing = fact_audit([{"text": "这里没有任何数字"}], only_year)
    ensure(missing, "事实表里的数字没上屏时应该报出来")
    print(f"  ✓ 事实表写了 1822 但字幕里没有 → 报出来：{missing[0]}")

    unknown = fact_audit([{"text": "这个数字是 99"}], only_year)
    ensure(unknown, "字幕里的数字查不到出处时应该报出来")
    print(f"  ✓ 字幕里冒出 99 但事实表没有 → 报出来：{unknown[0]}")

    print()
    print("要点：反向漏最危险。事实表写了 5 个数据、字幕只用 2 个，")
    print("      片子里就少了一个关键数据，却没人发现——所以要双向查。")


if __name__ == "__main__":
    run()
