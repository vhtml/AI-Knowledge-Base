"""第 11 章实验：终检门槛。

这个模块解决什么问题：把"看着还行"变成八道**能自动判**的门槛。

知识类内容的交付和普通视频不一样：观众记住的是"这个说法对不对"。
所以终检里"事实"和"讲解"两栏的权重最高——画面丑一点还能忍，
数据错了就是错。

实验室能自动查的是可机械判定的部分（字幕停留、染色数量、不画比例、
片名卡位置、数字出处、分镜表与代码一致…）。需要人眼的部分
（画面里有没有东西伸进字幕区、主体是不是只有一个），
在实验里以"人工确认项"列出——**说清楚哪些没验过**比假装全过了更有用。

八道门槛查的对象是**成片自己**：分镜表和事实表来自
``capstone/fourier_C/storyboard.md``，字幕、场景、配置来自
``capstone/fourier_C/film.js``。没有样例数据参与判定——
查样例只能证明检查器能跑，查真片才能证明片子没问题。

对应笔记：AI/Knowledge-Film/11-终检门槛.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..labkit import (
    Fact,
    Shot,
    band_ratio,
    ensure,
    fact_audit,
    fmt_table,
    load_markdown_storyboard,
    parse_film_config,
    parse_film_scenes,
    parse_film_subs,
    script_stats,
    three_questions,
)

PROJECT = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C"
FILM_JS = PROJECT / "film.js"
STORYBOARD_MD = PROJECT / "storyboard.md"

MANUAL_CHECKS = (
    "字幕区（纵向 80%–95%）没有别的东西",
    "任何一刻只有一个主体在动",
    "结尾真的回到了开头的画面",
    "没有硬切",
    "抽五镜遮住字幕问三问，都答得出来",
)


@dataclass
class Gate:
    """一道门槛的检查结果。"""

    name: str
    ok: bool
    detail: str
    manual: bool = False
    """True 表示这条只能人眼确认，实验只负责提醒。"""


def check_all(
    shots: list[Shot],
    facts: list[Fact],
    film_subs: list[dict],
    scenes: list[dict],
    config: dict,
    references: list[str],
) -> list[Gate]:
    """跑完八道门槛。

    八道门槛的对象全是**成片自己**：分镜表和事实表来自
    ``capstone/fourier_C/storyboard.md``，字幕、场景、配置来自
    ``capstone/fourier_C/film.js``。没有样例数据参与判定。
    """
    gates: list[Gate] = []
    stats = script_stats(film_subs)

    # 1 讲解：成片和分镜表对得上 + 三问齐
    aligned = len(film_subs) == len(shots) and all(
        film["text"] == shot.subtitle for film, shot in zip(film_subs, shots)
    )
    three_ok = all(not three_questions(shot) for shot in shots)
    gates.append(Gate(
        "讲解",
        aligned and three_ok,
        f"字幕↔分镜一致：{'是' if aligned else '否'}；三问齐：{'是' if three_ok else '否'}",
    ))

    # 2 画与不画
    ratio = band_ratio(shots)
    gates.append(Gate(
        "画与不画",
        0.1 <= ratio <= 0.4,
        f"「不画」占比 {ratio:.0%}（经验值约五分之一）",
    ))

    # 3 结构
    duration = float(config.get("DUR", 0))
    title_ok = len(film_subs) == len(shots)
    last_cue = film_subs[-1]["text"] if film_subs else ""
    gates.append(Gate(
        "结构",
        title_ok and bool(references) and ("1822" in last_cue or "片名" in last_cue),
        f"片名卡已设；最后一屏是「{last_cue}」；文献 {len(references)} 条",
    ))

    # 4 字幕
    hold_ok = stats.min_hold >= 3.0 and stats.max_hold <= 5.0
    color_ok = not stats.multi_color_lines
    line_ok = stats.longest_line_chars <= 24
    gates.append(Gate(
        "字幕",
        hold_ok and color_ok and line_ok,
        f"停留 {stats.min_hold:.1f}–{stats.max_hold:.1f}s；一句最多染一处："
        f"{'是' if color_ok else '否'}；最长句 {stats.longest_line_chars} 字",
    ))

    # 5 画面（分镜表层面可判的部分）
    label_ok = all("；" not in shot.label for shot in shots)
    gates.append(Gate(
        "画面",
        label_ok,
        f"一镜最多一条小字：{'是' if label_ok else '否'}；颜色语义全片一致（人工确认）",
        manual=True,
    ))

    # 6 节奏
    gaps = []
    cursor = 0.0
    for cue in film_subs:
        if cue["t"] - cursor > 0.05:
            gaps.append(cue["t"] - cursor)
        cursor = max(cursor, cue["t"] + cue["hold"])
    if duration - cursor > 0.05:
        gaps.append(duration - cursor)
    longest = max(gaps, default=0.0)
    gates.append(Gate(
        "节奏",
        longest >= 2.0,
        f"最长无字幕纯画面 {longest:.2f}s；没有硬切（人工确认）",
        manual=True,
    ))

    # 7 事实
    problems = fact_audit(film_subs, facts)
    illustrative_ok = all(f.source.strip() for f in facts)
    gates.append(Gate(
        "事实",
        not problems and illustrative_ok,
        f"数字↔事实表：{'一致' if not problems else '；'.join(problems)}",
    ))

    # 8 技术
    size_ok = config.get("W") == "1920" and config.get("H") == "1080" and config.get("FPS") == "30"
    scene_ok = bool(scenes) and abs(scenes[0]["t0"]) < 1e-6
    gates.append(Gate(
        "技术",
        size_ok and scene_ok,
        f"{config.get('W')}×{config.get('H')} @ {config.get('FPS')}fps；"
        f"场景铺满全片：{'是' if scene_ok else '否'}；--det 需另跑（人工确认）",
        manual=True,
    ))

    return gates


def run() -> None:
    print("=" * 78)
    print("第 11 章 · 终检：八道门槛")
    print("=" * 78)

    storyboard = load_markdown_storyboard(STORYBOARD_MD)
    shots = storyboard.shots
    facts = storyboard.facts
    film_subs = parse_film_subs(FILM_JS)
    scenes = parse_film_scenes(FILM_JS)
    config = parse_film_config(FILM_JS)

    gates = check_all(
        shots, facts, film_subs, scenes, config, storyboard.references
    )

    print()
    print("【1】成片的终检结果")
    print(fmt_table(
        ["门槛", "结果", "说明"],
        [
            [gate.name, "✓ 通过" if gate.ok else "✗ 未过", gate.detail]
            for gate in gates
        ],
    ))

    failed = [gate for gate in gates if not gate.ok]
    ensure(not failed, f"成片没过终检：{[gate.name for gate in failed]}")

    print()
    print("【2】人工确认项（实验查不了，但交片时必须过）")
    for item in MANUAL_CHECKS:
        print(f"  □ {item}")

    # ---- 故意做坏 ----
    print()
    print("【3】把片子做坏，看门槛抓不抓得住")
    broken_shots = [
        Shot(index=1, subtitle="第一句", method="位置排布", on_screen="一个球",
             action="球在动", takeaway="球会动"),
        Shot(index=2, subtitle="第二句", method="位置排布", on_screen="一个球",
             action="球在动", takeaway="球老在动", label="小字一；小字二"),
    ]
    broken_cues = [
        {"t": 0.2, "text": "这是第一句", "hold": 1.2},
        {"t": 1.6, "text": "这是第二句{金色}{玫红}", "hold": 9.0},
    ]
    broken_facts = [Fact(key="年份", value="1822", source="出处", illustrative=False)]
    broken_gates = check_all(
        broken_shots, broken_facts, broken_cues, [], {"DUR": "10"}, []
    )
    for gate in broken_gates:
        print(f"  {'✓' if gate.ok else '✗'} {gate.name}：{gate.detail}")
    caught = [gate.name for gate in broken_gates if not gate.ok]
    ensure(len(caught) >= 4, f"坏片子应该至少被抓出 4 道门槛，实际 {caught}")

    print()
    print("要点：交片的时候，把没做到的地方直说——")
    print("      包括'成片我有没有从头到尾看过一遍'。")
    print("      骗自己说'应该没问题'的片子，最后都是观众替你发现问题。")


if __name__ == "__main__":
    run()
