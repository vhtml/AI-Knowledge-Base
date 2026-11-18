"""第 2 周实验：场景筛选与需求优先级。

这个模块解决什么问题：把"要不要做这个 AI 需求"从拍脑袋变成一次可复算的打分，
把"那先做哪个"从谁嗓门大谁先做变成一条可解释的排序公式。
打分只看任务属性，排序只看价值与成本——两者分开，才知道改哪一项能改变结果。

对应笔记：AI/AI-Product/ 第 03 章《场景筛选》与第 04 章《需求优先级》。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..labkit import FixtureError, Score, fmt_table, load_fixture, pct

# AI Fit 四判据的权重，总和 = 100。
# 权重不是拍脑袋的：它表达的是「这条判据不成立时，项目失败的概率」。
CRITERIA_WEIGHTS: dict[str, float] = {
    "determinism": 30.0,
    "tolerates_error": 25.0,
    "corpus": 20.0,
    "interaction_leverage": 25.0,
}

CRITERIA_LABELS: dict[str, str] = {
    "determinism": "确定性",
    "tolerates_error": "容错",
    "corpus": "语料",
    "interaction_leverage": "交互杠杆",
}

# 判据 → 场景里的字段名。分开写是因为场景字段带 has_ 前缀（表意是"有没有"），
# 而判据名是抽象概念（"确定性"），两者不是一回事，混用会让读代码的人分不清。
CRITERIA_FIELDS: dict[str, str] = {
    "determinism": "has_determinism",
    "tolerates_error": "tolerates_error",
    "corpus": "has_corpus",
    "interaction_leverage": "has_interaction_leverage",
}

CRITERIA_WHY: dict[str, str] = {
    "determinism": "没有客观判据，错了没人会发现——正确性只能靠人读，成本压不下去",
    "tolerates_error": "错一次的代价决定用户会不会用第二次，用一次的产品没有留存",
    "corpus": "有没有语料决定要不要先花钱建检索，但它可以后补，所以权重最低",
    "interaction_leverage": "能借用户的一次确认补上模型短板，等于把一部分判断力外包给用户",
}

# 档位阈值。为什么是这两个数：
# 40 分——低于它意味着四条判据里至少两条不成立（最常见是「不确定 + 不容错」），
#        这类场景人工做的成本远低于模型做错的成本，先别做；
# 70 分——高于它意味着场景「闭」得比较整齐：判据清楚、错了代价可控、语料或交互能兜住，
#        值得真的立项做，而不是先做个试点看看；
# 40–70 之间是试点区：做个小范围验证，拿真实数据再回来打分。
REJECT_BELOW = 40.0
BUILD_AT = 70.0

# 排序时档位的加成：立项级的场景优先级更高，否决级的明显靠后。
# 注意它只是「倾斜」，压不过成本项——这是刻意的，避免一个低价值的 build 场景挤掉高价值的 pilot。
FIT_MULTIPLIER: dict[str, float] = {
    "build": 1.2,
    "pilot": 1.0,
    "reject": 0.5,
}

# 不确定性预算：上线前允许浪费掉的最大比例。
# 越不确定的场景越要小步验证——所以越不确定，能承受的浪费比例越低。
UNCERTAINTY_BUDGET: dict[str, float] = {
    "build": 0.30,
    "pilot": 0.15,
    "reject": 0.0,
}

REQUIRED_FIELDS = (
    "id",
    "name",
    "task",
    "has_determinism",
    "tolerates_error",
    "has_corpus",
    "has_interaction_leverage",
    "frequency",
    "minutes_saved",
    "cost_monthly",
)


@dataclass
class RankedTask:
    """排序后的一条需求。"""

    id: str
    name: str
    verdict: str
    fit_score: float
    monthly_minutes: float
    monthly_hours: float
    cost_monthly: float
    roi: float
    rank_score: float
    budget_ratio: float
    budget_minutes: float
    rank: int


def load_scenarios() -> list[dict]:
    """读取候选场景列表。"""
    data = load_fixture("scenarios")
    scenarios = data["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise FixtureError("scenarios.json 的 scenarios 必须是非空列表")
    return scenarios


def ai_fit_score(scenario: dict) -> Score:
    """给一个场景打 AI Fit 分（0–100），并判定档位。

    分数只由四条判据的布尔值加权得到——不加任何主观调节项，
    这样"两个人打出的分不一样"就一定是判据理解不同，而不是手感不同。
    """
    missing = [name for name in REQUIRED_FIELDS if name not in scenario]
    if missing:
        raise FixtureError(f"场景 {scenario.get('id', '?')} 缺少字段：{'、'.join(missing)}")

    breakdown: dict[str, float] = {}
    notes: list[str] = []
    for criterion, weight in CRITERIA_WEIGHTS.items():
        field_name = CRITERIA_FIELDS[criterion]
        value = scenario[field_name]
        if not isinstance(value, bool):
            raise FixtureError(
                f"场景 {scenario['id']} 的 {field_name} 必须是布尔值，实际 {value!r}"
            )
        if value:
            breakdown[criterion] = weight
        else:
            breakdown[criterion] = 0.0
            notes.append(f"缺{CRITERIA_LABELS[criterion]}：{CRITERIA_WHY[criterion]}")

    total = sum(breakdown.values())
    if total >= BUILD_AT:
        verdict = "build"
    elif total >= REJECT_BELOW:
        verdict = "pilot"
    else:
        verdict = "reject"
    return Score(total=total, verdict=verdict, breakdown=breakdown, notes=notes)


def priority(tasks: list[dict]) -> list[RankedTask]:
    """按「价值 × 收益 / 成本」给需求排序，并标注不确定性预算。

    排序公式：rank_score = (频次 × 单次节省分钟 / 60) / 月成本 × 档位系数
    —— 分子是每月省下的小时数，分母是每月花的钱，比值是"每块钱换回多少小时"。
    稳定性：同分时保持输入顺序，所以两次运行的名次可直接比较。
    """
    ranked: list[RankedTask] = []
    for scenario in tasks:
        fit = ai_fit_score(scenario)
        frequency = float(scenario["frequency"])
        minutes_saved = float(scenario["minutes_saved"])
        cost_monthly = float(scenario["cost_monthly"])
        if frequency < 0 or minutes_saved < 0 or cost_monthly < 0:
            raise FixtureError(f"场景 {scenario['id']} 的频次/节省时间/成本不能为负")

        monthly_minutes = frequency * minutes_saved
        monthly_hours = monthly_minutes / 60
        roi = monthly_hours / cost_monthly if cost_monthly else float("inf")
        rank_score = roi * FIT_MULTIPLIER[fit.verdict]
        budget_ratio = UNCERTAINTY_BUDGET[fit.verdict]
        ranked.append(
            RankedTask(
                id=scenario["id"],
                name=scenario["name"],
                verdict=fit.verdict,
                fit_score=fit.total,
                monthly_minutes=monthly_minutes,
                monthly_hours=monthly_hours,
                cost_monthly=cost_monthly,
                roi=roi,
                rank_score=rank_score,
                budget_ratio=budget_ratio,
                budget_minutes=monthly_minutes * budget_ratio,
                rank=0,
            )
        )

    # 稳定排序：只按分数降序，同分保持输入顺序（Python 的 sort 是稳定的）。
    ranked.sort(key=lambda item: -item.rank_score)
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def run() -> None:
    """跑第 2 周实验：AI Fit 打分 → 档位分布 → 优先级排序。"""
    print("=" * 78)
    print("第 2 周 · 场景筛选与需求优先级")
    print("=" * 78)

    scenarios = load_scenarios()

    # ---- 1. 逐个打分 ----
    print()
    print("【1】AI Fit 打分（四判据加权，满分 100）")
    rows: list[list[str]] = []
    scored: list[tuple[dict, Score]] = []
    for scenario in scenarios:
        fit = ai_fit_score(scenario)
        scored.append((scenario, fit))
        rows.append(
            [
                scenario["id"],
                scenario["name"],
                f"{fit.total:.0f}",
                fit.verdict,
                "、".join(
                    CRITERIA_LABELS[key] for key in CRITERIA_WEIGHTS if fit.breakdown.get(key)
                )
                or "—",
            ]
        )
    print(fmt_table(["编号", "场景", "得分", "档位", "成立的判据"], rows, align_right={2}))

    # ---- 2. 档位分布 ----
    verdicts = [fit.verdict for _, fit in scored]
    print()
    print("【2】档位分布")
    print(
        fmt_table(
            ["档位", "数量", "占比", "含义"],
            [
                ["build", str(verdicts.count("build")), pct(verdicts.count("build") / len(verdicts)),
                 "可以立项做"],
                ["pilot", str(verdicts.count("pilot")), pct(verdicts.count("pilot") / len(verdicts)),
                 "先小范围验证"],
                ["reject", str(verdicts.count("reject")), pct(verdicts.count("reject") / len(verdicts)),
                 "别做，人工更便宜"],
            ],
            align_right={1, 2},
        )
    )

    # ---- 3. 优先级排序 ----
    original_ids = [scenario["id"] for scenario in scenarios]
    ranked = priority(scenarios)
    print()
    print("【3】需求优先级（价值 = 频次 × 节省时间）")
    print(
        fmt_table(
            ["名次", "编号", "场景", "档位", "月省小时", "月成本", "比值", "排序分", "不确定性预算"],
            [
                [
                    str(item.rank),
                    item.id,
                    item.name,
                    item.verdict,
                    f"{item.monthly_hours:.1f}",
                    f"{item.cost_monthly:.0f}",
                    f"{item.roi:.4f}",
                    f"{item.rank_score:.4f}",
                    (
                        f"{pct(item.budget_ratio, 0)}（{item.budget_minutes:.0f} 分钟）"
                        if item.verdict != "reject"
                        else "不立项"
                    ),
                ]
                for item in ranked
            ],
            align_right={0, 4, 5, 6, 7},
            max_widths=[4, 6, 22, 6, 8, 8, 8, 8, 20],
        )
    )
    print()
    print("  比值 = 每 1 元月成本换回的小时数；排序分 = 比值 × 档位系数（build 1.2 / pilot 1.0 / reject 0.5）。")

    # ---- 断言 ----
    for scenario, fit in scored:
        assert 0 <= fit.total <= 100, f"{scenario['id']} 的得分必须在 0–100"
        assert fit.verdict in {"reject", "pilot", "build"}
        assert abs(sum(fit.breakdown.values()) - fit.total) < 1e-9, "分项之和必须等于总分"
    assert abs(sum(CRITERIA_WEIGHTS.values()) - 100) < 1e-9, "四条判据权重之和必须是 100"
    assert verdicts.count("reject") >= 1, "样例里至少要有一个被否决的场景"
    assert verdicts.count("build") >= 1, "样例里至少要有一个可以立项的场景"
    assert [scenario["id"] for scenario in scenarios] == original_ids, "排序不能改动输入顺序"
    # 稳定性检查：两个属性完全相同的场景，比值相同，名次必须保持输入里的先后顺序。
    twin_a = dict(scenarios[0])
    twin_b = dict(scenarios[0], id="SC-TWIN", name="同分对照项")
    twin_order = [item.id for item in priority([twin_a, twin_b])]
    assert twin_order == ["SC-01", "SC-TWIN"], "同分时必须保持输入顺序，否则名次不可比"
    scores = [item.rank_score for item in ranked]
    assert scores == sorted(scores, reverse=True), "排序结果必须按比值降序"
    try:
        ai_fit_score({"id": "X"})
    except FixtureError:
        pass
    else:
        raise AssertionError("缺字段的场景必须报错")

    print()
    print("要点：得分只由四条判据决定，所以分数可以拿来跟别人吵；")
    print("      优先级还要看成本，所以「高频」不等于「值得做」。")
    top = ranked[0]
    print(f"      当前第一名是 {top.name}（{top.verdict}），每 1 元月成本换回 {top.roi:.4f} 小时。")
    print("      不确定性预算 = 试点允许浪费掉的省时比例：越不确定，越要小步验证。")
