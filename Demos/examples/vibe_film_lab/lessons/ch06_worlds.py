"""第 6 章实验：两套外观。

这个模块解决什么问题：把"选外观"变成一个**有判据的决定**，而不是"我觉得好看"。

这类片子目前公认只有两套外观：**夜曲**（抽象概念、看不见的机制）
和**寓言**（心理、行为类逻辑）。两套的色、字、版式、常驻元素、
片名卡、声音都定好了，选定一套全片不许破例。

最重要的一条不是配色，是**颜色语义锁定**：金色永远表示关键词，
玫红永远表示反直觉的点，青色永远表示数值。一部片子一旦换了颜色含义，
观众就会在同一支片子里遇到两套语法。

对应笔记：AI/Knowledge-Film/06-两套外观.md
"""

from __future__ import annotations

from ..labkit import (
    ensure,
    fixtures_dir,
    fmt_table,
    load_json,
    parse_film_config,
)

# 选题关键词 → 外观。抽象机制走夜曲，行为心理走寓言。
ABSTRACT_TOPIC_WORDS = ("傅里叶", "熵", "卷积", "矩阵", "概率", "算法", "定理", "信号", "机制")
BEHAVIOR_TOPIC_WORDS = ("迷信", "偏差", "焦虑", "紧张", "沉没", "习惯", "情绪", "心理")


def choose_world(topic: str) -> str:
    """按题目选外观。

    判据只有一条：**这个概念能不能当场演出来**。
    能演（抽象机制）→ 夜曲；要从生活场景讲到"你该怎么办" → 寓言。
    """
    if any(word in topic for word in BEHAVIOR_TOPIC_WORDS):
        return "B"
    if any(word in topic for word in ABSTRACT_TOPIC_WORDS):
        return "C"
    return "C"  # 拿不准选 C


def run() -> None:
    print("=" * 78)
    print("第 6 章 · 两套外观：夜曲与寓言")
    print("=" * 78)

    worlds = load_json(fixtures_dir() / "worlds.json")

    print()
    print("【1】两套外观")
    for key in ("C", "B"):
        world = worlds["worlds"][key]
        print(f"  {key} · {world['name']}")
        print(f"      适合：{world['for']}")
        print(f"      声音：{world['sound']}")
        print(f"      费工：{world['effort']}")
    print(f"  规则：{worlds['rule']}")

    # ---- 选型决策 ----
    print()
    print("【2】按题目选外观")
    topics = [
        "傅里叶变换", "鸽子的迷信", "信息熵", "为什么越紧张越做不好",
        "卷积是什么", "幸存者偏差",
    ]
    print(fmt_table(
        ["题目", "选", "为什么"],
        [
            [
                topic,
                worlds["worlds"][choose_world(topic)]["name"],
                "抽象机制，当场能演" if choose_world(topic) == "C" else "行为逻辑，要落到生活",
            ]
            for topic in topics
        ],
    ))

    facts = load_json(fixtures_dir() / "facts.json")
    topic = facts["topic"]
    chosen = choose_world(topic)
    ensure(chosen == "C", f"样片《{facts['title']}》应该选夜曲，实际选了 {chosen}")

    # ---- 一部片子只用一套外观 ----
    config = parse_film_config(
        fixtures_dir().parent / "capstone" / "fourier_C" / "film.js"
    )
    ensure(
        config.get("WORLD") == chosen,
        f"film.js 的 WORLD（{config.get('WORLD')}）和选题判出来的（{chosen}）不一致",
    )
    print(f"  样片《{facts['title']}》→ 夜曲；film.js 的 CONFIG.WORLD = {config.get('WORLD')} ✓")

    # ---- 颜色语义锁定 ----
    print()
    print("【3】颜色语义：全片锁定，不破例")
    for key in ("C", "B"):
        world = worlds["worlds"][key]
        meanings = world["color_meaning"]
        print(f"  {key} · {world['name']}：" + "，".join(f"{k}={v}" for k, v in meanings.items()))
        ensure(
            len(set(meanings.values())) == len(meanings),
            f"{key} 的颜色含义有重复：一个意思占了两个颜色",
        )

    c_mean = worlds["worlds"]["C"]["color_meaning"]
    b_mean = worlds["worlds"]["B"]["color_meaning"]
    ensure(
        c_mean == b_mean,
        "两套外观的颜色语义应该一致：金色=关键词，玫红=反直觉，青色=数值",
    )
    print("  两套外观共用同一套颜色语义——换皮不换语法。")

    print()
    print("【4】锁死的规矩")
    for lock in worlds["locks"]:
        print(f"  · {lock}")

    print()
    print("要点：外观不是审美选择，是**语法选择**。")
    print("      观众看三秒就学会了'金色=关键词'，之后你必须一直遵守；")
    print("      破坏这条，观众会把注意力从内容转回'这片子到底想说什么'。")


if __name__ == "__main__":
    run()
