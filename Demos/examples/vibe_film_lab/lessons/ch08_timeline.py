"""第 8 章实验：时间轴与卡点。

这个模块解决什么问题：把"卡点"从手感变成**可计算的偏差**。

无人声的知识片子，节奏全靠两样东西撑着：字幕什么时候换、画面什么时候动。
这两样东西最好落在音乐的拍子上——观众说不出为什么，但他会觉得"顺"。

这一章做三件事：
1. 排出字幕时间轴，检查它自洽（单调、不重叠）；
2. 量化每句字幕和拍点的偏差，再演示怎么把它对齐（snap）；
3. 检查片名卡的位置和无字幕的纯画面段落。

对应笔记：AI/Knowledge-Film/08-时间轴与卡点.md
"""

from __future__ import annotations

import re
from pathlib import Path

from ..labkit import (
    ensure,
    fmt_table,
    parse_film_config,
    parse_film_subs,
)

FILM_JS = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C" / "film.js"
BPM = 120.0
"""样片的配乐速度。120 BPM 意味着每 0.5 秒一个拍子。"""


def beat_grid(bpm: float, duration: float, offset: float = 0.0) -> list[float]:
    """生成拍点表。"""
    step = 60.0 / bpm
    count = int(duration / step) + 1
    return [offset + index * step for index in range(count)]


def snap_to_beat(time: float, bpm: float) -> float:
    """把一个时刻吸附到最近的拍点。"""
    step = 60.0 / bpm
    return round(round(time / step) * step, 3)


def run() -> None:
    print("=" * 78)
    print("第 8 章 · 时间轴与卡点")
    print("=" * 78)

    config = parse_film_config(FILM_JS)
    duration = float(config.get("DUR", 0))
    subs = parse_film_subs(FILM_JS)

    # ---- 1. 时间轴自洽 ----
    print()
    print("【1】字幕时间轴")
    rows = []
    for index, sub in enumerate(subs):
        rows.append([
            str(index + 1),
            sub["text"][:18] + ("…" if len(sub["text"]) > 18 else ""),
            f"{sub['t']:.2f}s",
            f"{sub['t'] + sub['hold']:.2f}s",
            f"{sub['hold']:.1f}s",
        ])
    print(fmt_table(["#", "字幕", "起", "止", "停留"], rows, align_right={2, 3, 4}))

    for previous, current in zip(subs, subs[1:]):
        ensure(
            previous["t"] + previous["hold"] <= current["t"] + 1e-6,
            f"第 {subs.index(previous) + 1} 句和下一句时间重叠",
        )
    ensure(subs[0]["t"] >= 0.3, "第一句不该压在第 0 秒——片头要留一点白")
    ensure(
        subs[-1]["t"] + subs[-1]["hold"] <= duration + 1e-6,
        "最后一句超出了片长",
    )
    print(f"  ✓ 时间轴单调、不重叠，落在 0–{duration:.1f}s 内。")

    # ---- 2. 卡点 ----
    print()
    print(f"【2】和 {BPM:.0f} BPM 的拍点对齐")
    beats = beat_grid(BPM, duration)
    rows = []
    for index, sub in enumerate(subs):
        snapped = snap_to_beat(sub["t"], BPM)
        offset = abs(snapped - sub["t"])
        rows.append([
            str(index + 1),
            f"{sub['t']:.2f}s",
            f"{snapped:.2f}s",
            f"{offset * 1000:.0f} ms",
            "✓" if offset <= 0.25 else "✗",
        ])
    print(fmt_table(["#", "原起点", "吸附后", "偏差", "命中"], rows, align_right={1, 2, 3}))

    hits = sum(1 for sub in subs if abs(snap_to_beat(sub["t"], BPM) - sub["t"]) <= 0.25)
    ratio = hits / len(subs)
    print(f"  命中率 {hits}/{len(subs)} = {ratio:.0%}（偏差 ≤ 250ms 算命中）")
    ensure(ratio >= 0.75, f"卡点命中率 {ratio:.0%} 偏低")

    # ---- 3. 片名卡 ----
    print()
    print("【3】片名卡位置")
    source = FILM_JS.read_text(encoding="utf-8")
    match = re.search(r"const TITLE_CARD = \{\s*t0:\s*([\d.]+)\s*,\s*t1:\s*([\d.]+)\s*\}", source)
    ensure(match, "film.js 里应该声明 TITLE_CARD")
    t0, t1 = float(match.group(1)), float(match.group(2))
    print(f"  片名卡：{t0:.1f}s – {t1:.1f}s，占片长 {t0 / duration:.0%} – {t1 / duration:.0%}")
    ensure(t0 > 1.0, "片名不该在第 1 秒出现")
    ensure(
        0.05 <= t0 / duration <= 0.5,
        f"片名卡应在片长 5%–50% 之间出现，实际 {t0 / duration:.0%}",
    )
    ensure(t1 > t0, "片名卡区间反了")

    # ---- 4. 无字幕的纯画面 ----
    print()
    print("【4】无字幕的纯画面段落")
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for sub in subs:
        if sub["t"] - cursor > 0.05:
            gaps.append((cursor, sub["t"]))
        cursor = max(cursor, sub["t"] + sub["hold"])
    if duration - cursor > 0.05:
        gaps.append((cursor, duration))

    print(fmt_table(
        ["起", "止", "时长"],
        [[f"{start:.2f}s", f"{end:.2f}s", f"{end - start:.2f}s"]
         for start, end in gaps],
        align_right={2},
    ))
    longest = max((end - start for start, end in gaps), default=0.0)
    print(f"  最长一段纯画面：{longest:.2f}s")
    ensure(
        longest >= 2.0,
        "全片应该至少有一处 2 秒以上的纯画面，让观众喘口气",
    )

    print()
    print("要点：卡点不是把每句都挪到拍点上——那会显得机械。")
    print("      实测下来 3/4 命中就够了，剩下那一句留作呼吸。")


if __name__ == "__main__":
    run()
