"""第 3 章实验：事实表与出处。

这个模块解决什么问题：让"每个事实都带出处"变成一道**双向**的检查。

只查一个方向是不够的。正向查（字幕里的数有没有出处）能防住编造；
反向查（事实表里的数有没有上屏）能防住另一种更隐蔽的翻车——
事实表里备了 5 个关键数据，字幕只用了 2 个，片子看着完整，其实少讲了一半。

还有一个容易被忽略的点：**示意**。有些画面上的数字是为了讲清道理而编的
（比如示意用的 220 Hz）。这类数字不是错，但必须标成"示意"，
否则观众会把它当成测量结果。

对应笔记：AI/Knowledge-Film/03-事实表与出处.md
"""

from __future__ import annotations

from ..labkit import Fact, ensure, fact_audit, fixtures_dir, fmt_table, load_json


def load_facts() -> list[Fact]:
    """从事实表文件读出 :class:`Fact` 列表。"""
    raw = load_json(fixtures_dir() / "facts.json")
    return [
        Fact(
            key=item["key"],
            value=item["value"],
            source=item["source"],
            illustrative=item.get("illustrative", False),
            on_screen=item.get("on_screen", True),
        )
        for item in raw["facts"]
    ]


def run() -> None:
    print("=" * 78)
    print("第 3 章 · 事实表：每个数字都要有出处")
    print("=" * 78)

    raw = load_json(fixtures_dir() / "facts.json")
    script = load_json(fixtures_dir() / "script.json")
    facts = load_facts()

    print()
    print("【1】论点与事实表")
    print(f"  题目：{raw['title']}（{raw['topic']}）")
    print(f"  论点：{raw['thesis']}")
    print()
    print(fmt_table(
        ["事实", "上屏的值", "类型", "出处"],
        [
            [
                fact.key,
                fact.value,
                "示意" if fact.illustrative else "实测/史实",
                fact.source,
            ]
            for fact in facts
        ],
    ))

    print()
    print("【2】双向核对：字幕 ↔ 事实表")
    problems = fact_audit(script["cues"], facts)
    for problem in problems:
        print(f"  ✗ {problem}")
    if not problems:
        print("  ✓ 字幕里的每个数字都能在事实表里查到出处；")
        print("    事实表里要上屏的数字也都出现在字幕里。")

    ensure(not problems, f"事实核对没通过：{problems}")

    # ---- 故意漏一条，看看检查器抓不抓得住 ----
    print()
    print("【3】把 1822 从字幕里删掉，检查器应该报警")
    broken = [dict(cue) for cue in script["cues"]]
    for cue in broken:
        cue["text"] = cue["text"].replace("1822", "很早以前")
    broken_problems = fact_audit(broken, facts)
    ensure(
        any("1822" in problem for problem in broken_problems),
        "删掉 1822 之后应该报出'查不到出处'或'没有出现在字幕里'",
    )
    for problem in broken_problems:
        print(f"  ✗ {problem}")

    # ---- 示意项必须有标记 ----
    print()
    print("【4】示意值必须标出来")
    illustrative = [fact for fact in facts if fact.illustrative]
    for fact in illustrative:
        print(f"  · {fact.value}：{fact.source}")
    ensure(
        all(fact.source.strip() for fact in illustrative),
        "示意值也必须写清楚为什么是示意",
    )

    # ---- 片尾要印文献 ----
    print()
    print("【5】片尾文献")
    for reference in raw["references"]:
        print(f"  · {reference}")
    ensure(len(raw["references"]) >= 1, "片尾至少要有一条文献")

    print()
    print("要点：知识类内容的信任是靠事实堆出来的。")
    print("      一个没有出处的数字，观众分不清是'真的'还是'你编的'——")
    print("      而这两者对一条知识短片的杀伤力是相反的。")


if __name__ == "__main__":
    run()
