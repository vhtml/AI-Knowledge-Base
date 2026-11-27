"""第 5 章实验：分镜表。

这个模块解决什么问题：给分镜表配一个**体检器**，让"分镜表过关"不再靠感觉。

分镜表是这条流水线上唯一真正的施工图：文案是给观众的，分镜表是给
画面的。写完之后、开工之前，必须有人（哪怕是自己）拿着分镜表问三遍：
这是什么？它在干什么？观众明白了什么？三问答不上来，这镜就不能画。

体检器查的是那些**可以机械判定**的问题：三问齐不齐、画法在不在六种之内、
"不画"的比例、一镜一条小字、连续三镜同一画法。

被检查的对象是**毕业作品自己的分镜表**（``capstone/fourier_C/storyboard.md``），
并且和成片代码逐字核对——分镜表说第 3 句是什么，``film.js`` 里就必须是那一句。
分镜表是唯一真相，不是参考稿。

对应笔记：AI/Knowledge-Film/05-分镜表.md
"""

from __future__ import annotations

from pathlib import Path

from ..labkit import (
    Shot,
    band_ratio,
    ensure,
    fmt_table,
    load_markdown_storyboard,
    parse_film_subs,
    three_questions,
    validate_storyboard,
)

PROJECT = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C"
FILM_JS = PROJECT / "film.js"
STORYBOARD_MD = PROJECT / "storyboard.md"


def run() -> None:
    print("=" * 78)
    print("第 5 章 · 分镜表：唯一的施工图")
    print("=" * 78)

    storyboard = load_markdown_storyboard(STORYBOARD_MD)
    shots = storyboard.shots

    print()
    print("【1】成片的分镜表（storyboard.md，唯一真相）")
    print(fmt_table(
        ["镜", "字幕", "画法", "屏上是什么", "它在干什么", "观众明白了什么"],
        [
            [
                str(shot.index),
                shot.subtitle,
                shot.method,
                shot.on_screen,
                shot.action,
                shot.takeaway,
            ]
            for shot in shots
        ],
    ))

    # ---- 真片必须一次过 ----
    problems = validate_storyboard(shots)
    for problem in problems:
        print(f"  ✗ {problem}")
    ensure(not problems, f"成片的分镜表没过体检：{problems}")
    print(f"  ✓ 成片分镜表一次通过（{len(shots)} 镜）")

    print()
    print("【2】遮住字幕列，只读屏上是什么和它在干什么")
    for shot in shots:
        missing = three_questions(shot)
        ensure(not missing, f"第 {shot.index} 镜三问不齐：{missing}")
        print(f"  第 {shot.index} 镜 → {shot.on_screen}；{shot.action} → {shot.takeaway}")

    print()
    print("【3】画与不画")
    ratio = band_ratio(shots)
    print(fmt_table(
        ["镜", "画法", "画不画"],
        [[str(shot.index), shot.method, "画" if shot.draw else "不画"] for shot in shots],
    ))
    print(f"  「不画」占比 {ratio:.0%}（参考片的经验值：约五分之一）")
    ensure(0.1 <= ratio <= 0.4, f"「不画」占比 {ratio:.0%} 偏离经验值太远")

    # ---- 分镜表和代码必须逐字一致 ----
    print()
    print("【4】分镜表 ↔ film.js：字幕必须逐字一致")
    subs = parse_film_subs(FILM_JS)
    ensure(
        len(subs) == len(shots),
        f"字幕条数不一致：分镜表 {len(shots)} 镜，film.js {len(subs)} 句",
    )
    for shot, sub in zip(shots, subs):
        ensure(
            shot.subtitle == sub["text"],
            f"第 {shot.index} 镜两边不一致：分镜表 {shot.subtitle!r} vs film.js {sub['text']!r}",
        )
    print(f"  ✓ {len(subs)} 句字幕与分镜表逐字一致（改了文案，两边都要改）。")

    # ---- 故意做坏，检查器要抓得住 ----
    print()
    print("【5】故意做坏一个分镜表，看体检器抓不抓得住")
    broken = [
        Shot(index=1, subtitle="第一句", method="位置排布", on_screen="一个球",
             action="球在动", takeaway="球会动"),
        Shot(index=2, subtitle="第二句", method="位置排布", on_screen="一个球",
             action="球在动", takeaway="球还会动"),
        Shot(index=3, subtitle="第三句", method="位置排布", on_screen="",
             action="", takeaway=""),
        Shot(index=4, subtitle="第四句", method="位置排布", on_screen="一个球",
             action="球在动", takeaway="球老在动", label="小字一；小字二"),
    ]
    broken_problems = validate_storyboard(broken)
    for problem in broken_problems:
        print(f"  ✗ {problem}")
    ensure(len(broken_problems) >= 3, "坏分镜表应该至少被抓出 3 个问题")

    print()
    print("要点：分镜表没有分镜表就别建工程。")
    print("      边画边想的结果是：写到第 8 镜才发现第 3 镜根本画不出来。")


if __name__ == "__main__":
    run()
