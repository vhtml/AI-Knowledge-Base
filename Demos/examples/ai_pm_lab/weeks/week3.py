"""第 3 周实验：竞品地图与定价锚点。

这个模块解决什么问题：把竞品从"一堆名字"变成一张能站上去的二维图（工作流深度 × 自动化等级），
把"这个产品该定多少钱"从拍脑袋变成一条从用户省时价值推出来的锚点区间。

竞品表里刻意只有定性字段：地图是用来定位自己的，不是用来拼市场份额报表的；
一旦掺进核实不了的市场份额数字，整张图的可信度就没了。

对应笔记：AI/AI-Product/ 第 05 章《竞品地图》与第 06 章《定价锚点》。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..labkit import FixtureError, fmt_table, load_fixture

# 象限切分点：工作流深度与自动化等级都以 0.5 为界。
# 定在 0.5 而不是中位数，是因为"端到端"和"自动"本身就是质变，
# 而"环节 / 半自动"这类中间态在不同产品里的差距没那么大。
QUADRANT_SPLIT = 0.5

QUADRANTS: dict[str, str] = {
    "low_low": "助手层",
    "low_high": "自动单点工具",
    "high_low": "流程编排层",
    "high_high": "端到端自动平台",
}

# 定价锚点：用户省下的钱里，愿意分给产品多少。
# 10% 是"只当省事的工具"的定价，30% 是"能替掉半个岗位"的定价。
VALUE_SHARE_LOW = 0.10
VALUE_SHARE_HIGH = 0.30


@dataclass
class Position:
    """一个竞品在地图上的坐标。"""

    name: str
    category: str
    depth: float
    automation: float
    quadrant: str


def quadrant_name(depth: float, automation: float) -> str:
    """由两个坐标算出象限名。"""
    high_depth = depth >= QUADRANT_SPLIT
    high_automation = automation >= QUADRANT_SPLIT
    if high_depth and high_automation:
        return QUADRANTS["high_high"]
    if high_depth:
        return QUADRANTS["high_low"]
    if high_automation:
        return QUADRANTS["low_high"]
    return QUADRANTS["low_low"]


def position_map() -> list[Position]:
    """把竞品投到「工作流深度 × 自动化等级」象限图上。"""
    data = load_fixture("competitors")
    depth_axis = data["axes"]["workflow_depth"]
    automation_axis = data["axes"]["automation_level"]

    positions: list[Position] = []
    for item in data["competitors"]:
        for field_name, axis in (
            ("workflow_depth", depth_axis),
            ("automation_level", automation_axis),
        ):
            if item[field_name] not in axis:
                raise FixtureError(
                    f"竞品 {item['name']} 的 {field_name} 取值 {item[field_name]!r} 不在坐标轴定义内"
                )
        depth = float(depth_axis[item["workflow_depth"]])
        automation = float(automation_axis[item["automation_level"]])
        positions.append(
            Position(
                name=item["name"],
                category=item["category"],
                depth=depth,
                automation=automation,
                quadrant=quadrant_name(depth, automation),
            )
        )
    return positions


def price_anchor(
    minutes_saved_per_month: float,
    hourly_cost: float,
    target_margin: float,
) -> dict[str, float | str]:
    """算出定价锚点区间与每用户月成本上限。

    公式链：
        用户每月省下的钱 = 每月节省分钟 / 60 × 人力时薪
        建议定价区间 = 该金额的 10% ~ 30%
        每用户月成本上限 = 区间下沿 × (1 - 目标毛利率)

    用下沿算成本上限是刻意的：定价先往低了报，能不能扛住成本才是真问题。
    """
    if minutes_saved_per_month < 0:
        raise FixtureError("每月节省分钟不能为负")
    if hourly_cost < 0:
        raise FixtureError("人力时薪不能为负")
    if not 0 < target_margin < 1:
        raise FixtureError("目标毛利率必须落在 (0, 1) 之间")

    value = minutes_saved_per_month / 60 * hourly_cost
    price_low = value * VALUE_SHARE_LOW
    price_high = value * VALUE_SHARE_HIGH
    cost_cap = price_low * (1 - target_margin)
    return {
        "value_per_month": value,
        "price_low": price_low,
        "price_high": price_high,
        "target_margin": target_margin,
        "cost_cap_per_user_month": cost_cap,
        "note": "示意测算：以「用户每月省下的时间价值」为锚，实际定价还要看替代方案与预算归属",
    }


def run() -> None:
    """跑第 3 周实验：竞品象限图 → 定位解读 → 定价锚点。"""
    print("=" * 78)
    print("第 3 周 · 竞品地图与定价锚点")
    print("=" * 78)

    positions = position_map()

    # ---- 1. 坐标表 ----
    print()
    print("【1】竞品在「工作流深度 × 自动化等级」上的位置")
    print(
        fmt_table(
            ["竞品", "类别", "工作流深度", "自动化等级", "象限"],
            [
                [
                    item.name,
                    item.category,
                    f"{item.depth:.2f}",
                    f"{item.automation:.2f}",
                    item.quadrant,
                ]
                for item in positions
            ],
            align_right={2, 3},
        )
    )

    # ---- 2. 象限分布 ----
    print()
    print("【2】象限分布")
    buckets: dict[str, list[str]] = {}
    for item in positions:
        buckets.setdefault(item.quadrant, []).append(item.name)
    print(
        fmt_table(
            ["象限", "坐标含义", "数量", "成员"],
            [
                [
                    name,
                    {
                        "助手层": "深度浅、自动化低：只出建议，人来干活",
                        "自动单点工具": "深度浅、自动化高：单点做得很自动",
                        "流程编排层": "深度高、自动化低：串起流程但不自己动手",
                        "端到端自动平台": "深度高、自动化高：整条链路自己跑完",
                    }[name],
                    str(len(members)),
                    "、".join(members),
                ]
                for name, members in sorted(buckets.items())
            ],
            align_right={2},
            max_widths=[16, 34, 4, 44],
        )
    )

    # ---- 3. 定价锚点 ----
    print()
    print("【3】定价锚点（示意测算）")
    assumptions = [
        ("每月节省分钟", 180, "示例：每工作日 9 分钟 × 20 天 + 每月 3 次批量整理"),
        ("人力时薪（元）", 120, "示例：客服岗综合时薪，按月均薪折算"),
        ("目标毛利率", 0.6, "示例：模型与检索成本之外的毛利要求"),
    ]
    minutes = float(assumptions[0][1])
    hourly = float(assumptions[1][1])
    margin = float(assumptions[2][1])
    anchor = price_anchor(minutes, hourly, margin)
    print(
        fmt_table(
            ["项", "取值", "说明"],
            [[key, str(value), note] for key, value, note in assumptions],
            max_widths=[16, 8, 48],
        )
    )
    print()
    print(
        fmt_table(
            ["测算项", "结果", "说明"],
            [
                [
                    "用户每月省下的钱",
                    f"{anchor['value_per_month']:.2f} 元",
                    f"{minutes:.0f} 分钟 ÷ 60 × {hourly:.0f} 元/小时",
                ],
                [
                    "建议定价区间",
                    f"{anchor['price_low']:.2f} ~ {anchor['price_high']:.2f} 元 / 用户 / 月",
                    "取用户省下金额的 10% ~ 30%",
                ],
                [
                    "每用户月成本上限",
                    f"{anchor['cost_cap_per_user_month']:.2f} 元",
                    f"区间下沿 × (1 - {margin:.0%} 毛利率)",
                ],
            ],
            max_widths=[18, 26, 44],
        )
    )

    # ---- 断言 ----
    assert len(positions) == 6, f"竞品记录应为 6 条，实际 {len(positions)}"
    for item in positions:
        assert 0 <= item.depth <= 1 and 0 <= item.automation <= 1, "坐标必须落在 0–1"
        assert item.quadrant == quadrant_name(item.depth, item.automation), (
            f"{item.name} 的象限名与坐标不符"
        )
    assert quadrant_name(0.2, 0.15) == "助手层"
    assert quadrant_name(0.2, 0.65) == "自动单点工具"
    assert quadrant_name(0.5, 0.4) == "流程编排层"
    assert quadrant_name(0.85, 0.9) == "端到端自动平台"

    assert float(anchor["price_low"]) < float(anchor["price_high"]), "定价区间上下界必须有序"
    assert 0 < float(anchor["target_margin"]) < 1
    assert float(anchor["cost_cap_per_user_month"]) < float(anchor["price_low"]), (
        "成本上限必须低于区间下沿，否则毛利撑不住"
    )
    for bad_args in ((-1, 100, 0.6), (180, -100, 0.6), (180, 100, 1.5), (180, 100, 0)):
        try:
            price_anchor(*bad_args)
        except FixtureError:
            pass
        else:
            raise AssertionError(f"非法输入应该报错：{bad_args}")

    print()
    print("要点：竞品地图只回答「我们在哪个象限」，不回答「我们赢不赢」——")
    print("      赢不赢取决于对方的护城河落在数据、集成还是分发，不在坐标上。")
    print("      定价锚点是「从用户省下的钱往回推」，不是「成本加价」——")
    print("      后者只能算出成本线，算不出用户愿意付多少。")
