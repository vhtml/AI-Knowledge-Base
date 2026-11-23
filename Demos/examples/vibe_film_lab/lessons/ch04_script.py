"""第 4 章实验：文案即字幕。

这个模块解决什么问题：把"字幕不是稿子的附赠品，而是稿子本身"这件事
变成**可量化的规矩**。

这类片子没有旁白，字幕是观众唯一的信息通道。所以字幕要同时满足两件
互相拉扯的要求：它得是**一句完整的话**（不然观众看不懂），
又得在**3–5 秒**内读完（不然画面会僵在那儿等你）。
绝大多数翻车都出在这条张力上：写成散文就没法读，写成电报就没法懂。

这一章量的是：句长、停留时长、语速、染色数量、开头那句够不够"和观众有关"。

对应笔记：AI/Knowledge-Film/04-文案即字幕.md
"""

from __future__ import annotations

import re

from ..labkit import ensure, fixtures_dir, fmt_table, load_json, script_stats

MAX_LINE_CHARS = 24
"""一行最多几个字。中文全角在 1080p 字幕字号下，一行 24 字是上限。"""

ABSTRACT_WORDS = (
    "概念", "重要", "话题", "内容", "知识", "意义", "价值", "理论", "本质",
    "思想", "精神", "方面", "层面", "维度", "关系", "影响",
)
"""抽象词。字幕里出现它们，而且没有具体物件，观众就不知道屏上该画什么。"""

CONCRETE_WORDS = (
    "声音", "线", "波", "球", "人", "钱", "骰子", "蛋", "鸟", "手机", "门",
    "门", "台阶", "绳子", "水", "火", "星", "钟", "尺", "纸", "卡片", "灯",
    "眼睛", "手", "视频", "画面",
)

CONTENT_TOPIC_WORDS = (
    "巧合", "为什么", "怎么", "是否", "原因", "代价", "秘密", "答案",
)
"""内容指向词。含这些词的问句是在问**内容**，不算"空问句"。

「你准备好了吗」问的是观众；「这是巧合吗」问的是内容。
前者该删，后者该留。"""


def run() -> None:
    print("=" * 78)
    print("第 4 章 · 文案即字幕：写稿就是排时间轴")
    print("=" * 78)

    script = load_json(fixtures_dir() / "script.json")
    cues = script["cues"]
    stats = script_stats(cues)
    # 最后一条是片名卡：它是结尾卡，读者要看清片名和年份，
    # 所以它的停留要求是"不短于 3 秒"而不是"落在 3–5 秒"；
    # 语速也按正文句算，否则片名卡会稀释语速。
    body_stats = script_stats(cues[:-1])

    print()
    print("【1】样片的字幕稿体检")
    print(fmt_table(
        ["指标", "值", "规矩"],
        [
            ["句数", str(len(stats.cues)), "2–5 分钟的片子大约 30–60 句"],
            ["总字数", str(stats.total_chars), "—"],
            ["停留中位/均值", f"{stats.min_hold:.1f}–{stats.max_hold:.1f}s / {stats.avg_hold:.1f}s", "3–5 秒"],
            ["语速", f"{body_stats.avg_chars_per_sec:.1f} 字/秒（正文）", "4.5–7 字/秒"],
            ["最长句", f"{stats.longest_line_chars} 字", f"不超过 {MAX_LINE_CHARS} 字"],
            ["一句话染多处颜色", str(len(stats.multi_color_lines)), "0（一句最多染一处）"],
        ],
        align_right={1},
    ))

    ensure(body_stats.min_hold >= 3.0, f"最短停留 {body_stats.min_hold}s，不到 3 秒")
    ensure(
        stats.cues[-1].hold >= 3.0,
        f"片名卡停留 {stats.cues[-1].hold}s，太短了——观众来不及看清",
    )
    ensure(stats.max_hold <= 5.0, f"最长停留 {stats.max_hold}s，超过 5 秒")
    # 语速按正文句算：片名卡是结尾卡，字少停留长，算进去会稀释语速
    ensure(
        4.5 <= body_stats.avg_chars_per_sec <= 7.0,
        f"语速 {body_stats.avg_chars_per_sec} 字/秒，超出 4.5–7 的区间",
    )
    ensure(
        stats.longest_line_chars <= MAX_LINE_CHARS,
        f"最长句 {stats.longest_line_chars} 字，超过一行 {MAX_LINE_CHARS} 字",
    )
    ensure(not stats.multi_color_lines, "有句子染了多处颜色")

    print()
    print("【2】逐句时间轴")
    print(fmt_table(
        ["#", "字幕", "起", "止", "停留", "字数", "染色"],
        [
            [
                str(cue.index + 1),
                cue.text,
                f"{cue.start:.2f}s",
                f"{cue.end:.2f}s",
                f"{cue.hold:.1f}s",
                str(cue.chars),
                ",".join(cue.colored) or "—",
            ]
            for cue in stats.cues
        ],
        align_right={2, 3, 4, 5},
    ))

    # ---- 时间轴必须自洽 ----
    for previous, current in zip(stats.cues, stats.cues[1:]):
        ensure(
            current.start >= previous.end,
            f"第 {current.index + 1} 句和上一句时间重叠了",
        )

    # ---- 坏稿体检 ----
    print()
    print("【3】四句写坏的文案，逐条打回")
    bad = script["bad_cues"]
    verdicts: list[tuple[str, list[str]]] = []
    for cue in bad:
        problems: list[str] = []
        plain = re.sub(r"\{[^}]*\}|\[[^\]]*\]", "", cue["text"])
        if len(plain) > MAX_LINE_CHARS:
            problems.append(f"{len(plain)} 字，超出一行 {MAX_LINE_CHARS} 字")
        if plain.count("{") or plain.count("["):
            if len(re.findall(r"\{[^}]*\}|\[[^\]]*\]", cue["text"])) >= 2:
                problems.append("一句话染了多处颜色")
        if re.match(r"^在\s*\d{2,4}\s*世纪", plain) or plain.startswith("我们今天"):
            problems.append("开头是历史/自指，不是和观众有关的第一句")
        if any(word in plain for word in ("重要", "认真", "务必", "请大家")):
            problems.append("自说自话，观众不需要被叮嘱")
        if (
            any(word in plain for word in ABSTRACT_WORDS)
            and not any(word in plain for word in CONCRETE_WORDS)
        ):
            problems.append("抽象词堆砌，观众读不出画面")
        if (
            plain.endswith(("吗", "？", "?"))
            and not any(word in plain for word in CONCRETE_WORDS)
            and not any(word in plain for word in CONTENT_TOPIC_WORDS)
        ):
            problems.append("空问句：问的是观众，不是内容")
        ensure(problems, f"这句本该判为不合格：{cue['text']}")
        verdicts.append((cue["text"], problems))

    print(fmt_table(
        ["原句", "问题"],
        [[text[:20] + ("…" if len(text) > 20 else ""), "；".join(problems)]
         for text, problems in verdicts],
    ))

    # ---- 开头那句 ----
    print()
    print("【4】第一句要跟观众有关")
    first = cues[0]["text"]
    print(f"  第一句：{first}")
    ensure(
        not re.match(r"^(在\s*)?\d{2,4}\s*世纪", first),
        "第一句不该从历史讲起",
    )
    ensure("我们" not in first, "第一句不该以'我们'自指开头")

    print()
    print("要点：写稿的时候，心里要同时有两个观众——")
    print("      一个是三秒后要划走的人（所以第一句必须和他有关），")
    print("      一个是五秒后还在读的人（所以每句必须能在五秒内读完）。")


if __name__ == "__main__":
    run()
