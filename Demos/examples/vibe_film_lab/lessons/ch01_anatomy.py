"""第 1 章实验：片型解剖。

这个模块解决什么问题：在动手做之前，先把"Vibe 知识"这类片子**量**一遍。
不量就开做，出来的只是"我以为是的那种视频"；量完你会发现，
它的形态约束比想象中硬——16:9、30fps、无人声、字幕停留 3–5 秒、
片名卡不在第一秒出现。这些约束不是审美偏好，是这个片型能成立的物理条件。

对应笔记：AI/Knowledge-Film/01-片型解剖.md
"""

from __future__ import annotations

import statistics

from ..labkit import ensure, fixtures_dir, fmt_table, load_json


def run() -> None:
    print("=" * 78)
    print("第 1 章 · 片型解剖：先把形态量出来")
    print("=" * 78)

    data = load_json(fixtures_dir() / "reference.json")
    films = data["films"]
    common = data["common"]

    print()
    print("【1】四条参考片的共同规格")
    print(fmt_table(
        ["项目", "值"],
        [
            ["画幅", common["aspect"]],
            ["分辨率", common["resolution"]],
            ["帧率", f"{common['fps']} fps"],
            ["有人声旁白？", "有" if common["voiceover"] else "没有（字幕是唯一信息通道）"],
            ["一句字幕对应一个动作？", "是" if common["one_line_one_action"] else "否"],
            ["片名不在第一秒出现？", "是" if common["title_not_in_first_second"] else "否"],
        ],
    ))

    print()
    print("【2】逐片测量")
    print(fmt_table(
        ["片名", "时长", "字幕条数", "停留中位", "语速(字/秒)", "片名卡(首)", "无字幕间隙"],
        [
            [
                film["name"],
                f"{film['duration']:.1f}s",
                str(film["subtitle_lines"]),
                f"{film['subtitle_hold_median']:.1f}s",
                f"{film['chars_per_sec']:.1f}",
                f"{film['title_card_first_at']:.1f}s",
                f"{film['max_silent_gap']:.1f}s",
            ]
            for film in films
        ],
        align_right={1, 2, 3, 4, 5, 6},
    ))

    holds = [film["subtitle_hold_median"] for film in films]
    speeds = [film["chars_per_sec"] for film in films]
    title_ratios = [
        film["title_card_first_at"] / film["duration"] for film in films
    ]

    print()
    print("【3】归纳出来的硬指标")
    print(fmt_table(
        ["指标", "实测范围", "当成规矩"],
        [
            [
                "字幕停留中位数",
                f"{min(holds):.1f}–{max(holds):.1f} 秒",
                "3–5 秒；短于 3 秒读不完，长于 5 秒画面会僵",
            ],
            [
                "语速",
                f"{min(speeds):.1f}–{max(speeds):.1f} 字/秒",
                "4.5–7 字/秒；再快就是播报，不再是知识动画",
            ],
            [
                "片名卡首次出现位置",
                f"{min(title_ratios):.0%}–{max(title_ratios):.0%} 片长",
                "片头 5%–30% 处出现一次，片尾再出现一次；第 1 秒绝不出现",
            ],
            [
                "无字幕纯画面",
                f"{min(f['max_silent_gap'] for f in films):.0f}–"
                f"{max(f['max_silent_gap'] for f in films):.0f} 秒",
                "每 20–30 秒留一段让人喘口气的纯画面",
            ],
        ],
    ))

    # ---- 断言：形态约束是硬的 ----
    ensure(
        all(f["width"] == 1920 and f["height"] == 1080 for f in films),
        "参考片应该全是 1920×1080",
    )
    ensure(
        all(f["fps"] == 30 for f in films),
        "参考片应该全是 30fps",
    )
    ensure(
        not any(f["voiceover"] for f in films),
        "这个片型没有旁白——有旁白就不是它了",
    )
    ensure(
        all(3.0 <= hold <= 4.2 for hold in holds),
        f"字幕停留中位数应落在 3–4.2 秒，实际 {holds}",
    )
    ensure(
        all(4.5 <= speed <= 7.0 for speed in speeds),
        f"语速应落在 4.5–7 字/秒，实际 {speeds}",
    )
    ensure(
        all(0.05 <= ratio <= 0.30 for ratio in title_ratios),
        f"片名卡首次出现应在片长 5%–30% 处，实际 {title_ratios}",
    )
    ensure(
        all(f["title_card_first_at"] > 1.0 for f in films),
        "片名不该在第 1 秒出现",
    )

    print()
    print("要点：这些数字不是审美，是约束。")
    print("      先照着量，再谈创新——不然你做的只是'一段加了动画的 PPT'。")
    print()
    print(f"      片名卡占比中位数 {statistics.median(title_ratios):.0%}，"
          f"无字幕间隙中位数 "
          f"{statistics.median(f['max_silent_gap'] for f in films):.0f} 秒。")


if __name__ == "__main__":
    run()
