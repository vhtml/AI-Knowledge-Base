"""第 2 章实验：形象优先。

这个模块解决什么问题：把"这句话能不能画出来"变成一道**可判定的题**。

Vibe 知识的灵魂是"字幕说到什么，屏上就在发生什么"。做不到这一点，
画面就退化成配图——观众的眼睛和耳朵各看各的，片子立刻变成 PPT。
所以流程必须是反过来的：**先找形象，再写文案**；
发现某句画不出来时，改的是句子，不是加一张图。

这一章用一个词表做启发式判定。它不是语言学判断，是给写稿人当草稿纸用的：
凡是"没有具体东西、没有可演动作"的句子，一律打回重写。

对应笔记：AI/Knowledge-Film/02-形象优先.md
"""

from __future__ import annotations

from ..labkit import (
    DRAW_METHODS,
    ensure,
    fixtures_dir,
    fmt_table,
    load_json,
    load_storyboard,
    parse_markup,
)

# 具体物件：能在屏上画出来的东西
OBJECT_WORDS = (
    "波形", "正弦", "波", "线", "柱子", "球", "蛋", "鸟", "鸽子", "猫", "狗", "人", "孩子",
    "手机", "屏幕", "门", "窗", "钱", "骰子", "台阶", "绳子", "弹簧", "水",
    "火", "星", "钟", "尺", "纸", "卡片", "队列", "人群", "车", "房子", "树",
    "灯", "手", "眼", "耳朵", "硬币", "沙漏", "气球", "镜子",
)

# 抽象名词：画不出来的"东西"
ABSTRACT_WORDS = (
    "概念", "机制", "原理", "定理", "系统", "效应", "规律", "趋势", "结构",
    "框架", "意义", "价值", "关系", "影响", "因素", "维度", "层面", "逻辑",
    "思路", "策略", "优化", "提升", "重要", "广泛", "应用", "领域", "本质",
    "意义", "精神", "思想", "问题", "话题", "内容", "知识",
)

# 可演的动作：屏上能"发生"的事
ACTION_WORDS = (
    "数", "数出", "裂", "叠", "合并", "长", "长高", "摆", "抖", "推", "拉",
    "亮", "亮起", "沉", "浮", "浮出", "落下", "排开", "排", "倒", "盖", "砍",
    "折", "折断", "转", "弹", "追", "挤", "冒", "消失", "出现", "对齐",
    "收拢", "散开", "落下", "起来", "开始", "停止", "停下", "离开", "回来",
    "上升", "下降", "靠近", "远离", "分开", "围住", "穿过", "跳过", "超过",
    "放大", "拉近", "浮现", "压上", "变粗", "变暗", "数着",
)


def drawable(sentence: str) -> tuple[bool, list[str]]:
    """判断一句话画不画得出来。

    返回 ``(能不能画, 原因列表)``。三条硬条件：
    有具体物件、有可演动作、长度放得下一行。
    """
    plain, _ = parse_markup(sentence)
    reasons: list[str] = []

    has_object = any(word in plain for word in OBJECT_WORDS)
    has_action = any(word in plain for word in ACTION_WORDS)
    too_long = len(plain) > 24
    abstract = [word for word in ABSTRACT_WORDS if word in plain]

    if not has_object:
        reasons.append("没有具体物件——屏上画什么？")
    if not has_action:
        reasons.append("没有可演的动作——画面不会动")
    if too_long:
        reasons.append(f"{len(plain)} 字，超过一行 24 字")
    if abstract and not has_object:
        reasons.append(f"只有抽象词（{'、'.join(abstract[:3])}）")

    return (not reasons), reasons


def drawable_shot(shot) -> tuple[bool, list[str]]:
    """判断一镜画不画得出来。

    注意判的是**镜**，不是字幕句：字幕可以是定场的陈述句
    （"这是一段乱糟糟的声音"），只要屏上有个具体的东西在动，这镜就成立。
    真正要求"动词"的是画面，不是字幕。
    """
    if not shot.draw:
        # "不画"的镜（提问、命名、结论）本来就不该有画面动作，
        # 只检查它有没有把"屏上是什么"说清楚。
        return (bool(shot.on_screen.strip()), [] if shot.on_screen.strip() else ["屏上是什么没写"])

    reasons: list[str] = []
    if not any(word in shot.on_screen for word in OBJECT_WORDS):
        reasons.append("屏上没有具体物件")
    if not any(word in shot.action for word in ACTION_WORDS):
        reasons.append("画面不会动")
    return (not reasons), reasons


def rewrite(sentence: str) -> str:
    """给一句画不出来的话一个可用的改写方向。

    实验室不替人写稿——它只指出"这句得改"，并给出改写的**结构**：
    谁 + 做什么 + 变成什么。
    """
    if "重要" in sentence or "认真看完" in sentence:
        return "（结构）[观众熟悉的东西] + （它做了什么）→ （变成什么）"
    if "定理" in sentence or "领域" in sentence:
        return "（结构）[年份/人名] + [一个具体的现象] + [它被拆成了什么]"
    if "抽象" in sentence:
        return "（结构）把抽象名词换成一个能摸到的东西"
    return "（结构）[观众熟悉的东西] + （它做了什么）→ （变成什么）"


def run() -> None:
    print("=" * 78)
    print("第 2 章 · 形象优先：先找形象，再写文案")
    print("=" * 78)

    print()
    print("【1】六种画法")
    print(fmt_table(
        ["画法", "什么时候用"],
        [
            ["位置排布", "几个东西摆在一张台面上，讲它们的关系"],
            ["时间推进", "一件事往前走一步，讲它怎么变的"],
            ["比例分割", "把整体切一块出来，讲它在整体里的位置"],
            ["对照比较", "两个东西并排放，讲差别"],
            ["代入体验", "让观众自己经历一遍"],
            ["道具实验", "拿一个物件做实验，结果看得见"],
        ],
    ))

    # ---- 分镜表里的形象登记 ----
    shots = load_storyboard(fixtures_dir() / "storyboard.json")
    print()
    print("【2】每个概念配了一个形象")
    print(fmt_table(
        ["镜", "观众明白了什么", "屏上是什么", "画法"],
        [
            [str(shot.index), shot.takeaway, shot.on_screen, shot.method]
            for shot in shots
        ],
    ))

    abstract_screens = [
        shot for shot in shots
        if any(word in shot.on_screen for word in ("抽象", "关系", "概念", "本质"))
    ]
    ensure(
        not abstract_screens,
        f"屏上不能出现抽象名词：{[s.index for s in abstract_screens]}",
    )

    # ---- 坏文案体检 ----
    script = load_json(fixtures_dir() / "script.json")
    bad = script["bad_cues"]
    print()
    print("【3】四句写坏的文案，逐句打回")
    rows = []
    for cue in bad:
        ok, reasons = drawable(cue["text"])
        ensure(not ok, f"这句本该判为画不出来：{cue['text']}")
        rows.append([cue["text"][:22] + ("…" if len(cue["text"]) > 22 else ""),
                     "画不出来", "；".join(reasons)])
    print(fmt_table(["原句", "判定", "原因"], rows))

    print()
    print("【4】改写方向（结构，不是成品）")
    for cue in bad:
        print(f"  · {cue['text'][:20]}… → {rewrite(cue['text'])}")

    # ---- 好文案必须都画得出来 ----
    print()
    print("【5】样片的每一镜都要画得出来")
    for shot in shots:
        ok, reasons = drawable_shot(shot)
        ensure(ok, f"样片第 {shot.index} 镜画不出来：{'；'.join(reasons)}")
        ensure(
            len(shot.subtitle) <= 24,
            f"样片第 {shot.index} 镜字幕超长：{len(shot.subtitle)} 字",
        )
    print(f"  {len(shots)} 镜全部通过：屏上都有具体物件、画面都有可演的动作、字幕都不超一行。")
    print("  （注意：判的是镜，不是字幕句——字幕可以是定场的陈述句。）")

    assert len(DRAW_METHODS) == 6
    used = {shot.method for shot in shots} - {"不画"}
    ensure(
        used <= set(DRAW_METHODS),
        f"分镜表用到了六种画法之外的画法：{used - set(DRAW_METHODS)}",
    )

    print()
    print("要点：画不出来的句子，改句子。")
    print("      给一句抽象话配一张抽象图，是这类片子最快的一种死法。")


if __name__ == "__main__":
    run()
