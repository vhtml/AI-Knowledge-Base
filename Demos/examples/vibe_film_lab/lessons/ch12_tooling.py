"""第 12 章实验：工具对照与选型。

这个模块解决什么问题：回答"我到底该用哪条路做"。

工具会一直变，但**路线**比工具稳定。至少现在能跑通的就这六条：
代码渲染、即梦动效工作室、VibeKnow、Manim、Remotion、ComfyUI + 视频模型。
它们不是替代关系——选题类型、素材形态、是否系列化、团队规模，
决定了你该走哪条。

这一章把选型写成**可复现的规则**：给定场景特征打分，
而不是"看哪个工具的广告写得好"。

对应笔记：AI/Knowledge-Film/12-工具对照与选型.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..labkit import ensure, fmt_table


@dataclass(frozen=True)
class Route:
    """一条制作路线。"""

    key: str
    name: str
    controllable: int
    """可控性：画面能不能说到就到。1–5。"""

    consistent: int
    """一致性：多期之间风格能不能稳住。1–5。"""

    cheap: int
    """性价比：钱和时间省不省。1–5。"""

    easy: int
    """上手快不快。1–5。"""

    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]


ROUTES: tuple[Route, ...] = (
    Route(
        key="code",
        name="代码渲染（Canvas/JS + ffmpeg）",
        controllable=5, consistent=5, cheap=5, easy=2,
        strengths=("抽象概念", "系列化", "无人声字幕科普", "完全可控"),
        weaknesses=("费工", "要会一点前端"),
    ),
    Route(
        key="motion_studio",
        name="即梦动效工作室",
        controllable=3, consistent=3, cheap=4, easy=4,
        strengths=("快速出片", "中文提示词", "风格探索"),
        weaknesses=("细节不可控", "闭源", "没有工程文件"),
    ),
    Route(
        key="vibeknow",
        name="VibeKnow（文档/PPT 转讲解视频）",
        controllable=2, consistent=4, cheap=5, easy=5,
        strengths=("已有素材", "课程批量", "七种创作模式"),
        weaknesses=("不能逐帧改", "风格选择有限"),
    ),
    Route(
        key="manim",
        name="Manim",
        controllable=5, consistent=5, cheap=4, easy=2,
        strengths=("数学公式", "推导过程", "精确排版"),
        weaknesses=("只适合公式和图形", "写起来比画起来慢"),
    ),
    Route(
        key="remotion",
        name="Remotion（React 写视频）",
        controllable=4, consistent=4, cheap=3, easy=3,
        strengths=("模板化", "团队协作", "数据驱动"),
        weaknesses=("要装 React 工具链", "复杂动效成本高"),
    ),
    Route(
        key="comfy",
        name="ComfyUI + 视频模型",
        controllable=2, consistent=2, cheap=2, easy=2,
        strengths=("人物质感", "实拍感", "素材生成"),
        weaknesses=("一致性最差", "吃显卡", "版权要核"),
    ),
)


@dataclass(frozen=True)
class Scenario:
    """一个待选型的场景。"""

    name: str
    needs: tuple[str, ...]
    weights: dict[str, int] = field(default_factory=dict)
    """关键词 → 权重。缺省权重按下面 NEED_WEIGHT。"""

    prefer: str = ""
    """硬性偏好：这条路线必须命中它，否则再高分也不选。"""


NEED_WEIGHT: dict[str, int] = {
    "抽象概念": 3,
    "数学公式": 3,
    "系列化": 2,
    "完全可控": 3,
    "无人声字幕科普": 2,
    "已有素材": 3,
    "课程批量": 2,
    "快速出片": 2,
    "人物质感": 4,
    "实拍感": 2,
    "模板化": 2,
    "团队协作": 1,
    "数据驱动": 2,
    "精确排版": 3,
    "风格探索": 1,
}


def recommend(scenario: Scenario) -> tuple[Route, dict[str, int]]:
    """按场景打分选路线。

    打分口径：每命中一个需求关键词，加上对应权重；
    再叠加路线自身的可控性/一致性/性价比/上手快慢的权重分。
    """
    scores: dict[str, int] = {}
    for route in ROUTES:
        score = 0
        for need in scenario.needs:
            weight = scenario.weights.get(need, NEED_WEIGHT.get(need, 1))
            if any(need in strength for strength in route.strengths):
                score += weight
        # 路线自身的四项属性只给"半权"：需求匹配才是决定性的，
        # 否则四条属性漂亮的路线会赢下所有场景。
        score += (route.controllable + route.consistent + route.cheap + route.easy) / 2
        scores[route.key] = score

    if scenario.prefer:
        for route in ROUTES:
            if route.key == scenario.prefer:
                scores[route.key] += 100
    best_key = max(scores, key=lambda key: scores[key])
    return next(route for route in ROUTES if route.key == best_key), scores


def run() -> None:
    print("=" * 78)
    print("第 12 章 · 工具对照与选型")
    print("=" * 78)

    print()
    print("【1】六条路线")
    print(fmt_table(
        ["路线", "可控", "一致", "性价比", "上手", "最适合", "短板"],
        [
            [
                route.name,
                str(route.controllable),
                str(route.consistent),
                str(route.cheap),
                str(route.easy),
                "、".join(route.strengths),
                "、".join(route.weaknesses),
            ]
            for route in ROUTES
        ],
        align_right={1, 2, 3, 4},
    ))

    scenarios = [
        Scenario(
            name="讲清楚傅里叶变换，2 分钟，无人声字幕科普，要连续做 10 期",
            needs=("抽象概念", "系列化", "完全可控", "无人声字幕科普"),
        ),
        Scenario(
            name="手上有 40 页 PPT，要快速变成 20 节课程讲解视频",
            needs=("已有素材", "课程批量", "快速出片"),
        ),
        Scenario(
            name="要讲一道数学题的完整推导，公式必须精确",
            needs=("数学公式", "精确排版", "推导过程"),
        ),
        Scenario(
            name="要做一条有人物出镜、质感接近实拍的品牌故事片",
            needs=("人物质感", "实拍感"),
        ),
        Scenario(
            name="每周固定更新一条数据播报，数字每月自动换",
            needs=("模板化", "数据驱动", "系列化"),
            # 这个场景要的是"换数字不换工程"，模板化和数据驱动是硬需求，
            # "系列化"反而谁都能做，所以权重压低。
            weights={"模板化": 3, "数据驱动": 3, "系列化": 1},
        ),
    ]

    print()
    print("【2】按场景选型")
    rows = []
    for scenario in scenarios:
        route, scores = recommend(scenario)
        rows.append([
            scenario.name,
            route.name,
            "、".join(
                f"{other.name.split('（')[0]} {scores[other.key]}"
                for other in ROUTES
                if other.key != route.key
            ),
        ])
    print(fmt_table(["场景", "推荐路线", "其他路线得分"], rows))

    # ---- 断言：每个场景都要有确定且合理的推荐 ----
    picks = {scenario.name: recommend(scenario)[0].key for scenario in scenarios}
    expected = {
        "讲清楚傅里叶变换，2 分钟，无人声字幕科普，要连续做 10 期": "code",
        "手上有 40 页 PPT，要快速变成 20 节课程讲解视频": "vibeknow",
        "要讲一道数学题的完整推导，公式必须精确": "manim",
        "要做一条有人物出镜、质感接近实拍的品牌故事片": "comfy",
        "每周固定更新一条数据播报，数字每月自动换": "remotion",
    }
    for name, want in expected.items():
        got = picks[name]
        ensure(
            got == want,
            f"场景「{name}」应该选 {want}，实际选了 {got}",
        )

    print()
    print("【3】路线的适用边界")
    print(fmt_table(
        ["路线", "什么时候别用"],
        [
            ["代码渲染", "只做一条、赶时间——成本换不回价值"],
            ["即梦动效工作室", "要求逐帧精确复刻既有风格"],
            ["VibeKnow", "需要改每一帧、每个字的画面"],
            ["Manim", "内容里没有公式和几何图形"],
            ["Remotion", "团队没人写过 React"],
            ["ComfyUI + 视频模型", "需要多期之间人物完全一致"],
        ],
    ))

    print()
    print("要点：工具选错，片子就卡在那个工具的上限上。")
    print("      先确定这条片子要什么（可控？批量？质感？），再选工具；")
    print("      反过来——先喜欢某个工具，再找题材——是这条赛道最常见的弯路。")


if __name__ == "__main__":
    run()
