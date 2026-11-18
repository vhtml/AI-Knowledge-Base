"""第 6 周实验：评测门禁与单位经济。

这个模块解决什么问题：把"这次改动到底是变好还是变差"变成一个门禁——
关键错误一票否决、软错误扣分，改动后总分低于改动前就不许上线；
再把"这个产品赚不赚钱"变成一条能算的公式链：单会话成本 → 每用户月成本 → 毛利率 → 回本周期。

评分规则里的关键错误一票否决，不是为了让分数好看，而是因为关键错误的性质是"整体失效"：
一条编造了金额的答案，不管其余部分写得多好，用户的决策依据已经被污染了——
平均分会把这种污染摊薄掉，然后让它混进上线版本。

对应笔记：AI/AI-Product/ 第 11 章《评测门禁》与第 12 章《单位经济》。
"""
from __future__ import annotations

from ..labkit import FixtureError, fmt_table, load_fixture, pct

# 软错误每条扣多少分。15 分是这么定的：一条软错误扣到 85 分，
# 单条就足以把"合格"拖到"刚好过线"，逼着人把软错误当硬错误处理。
SOFT_ERROR_PENALTY = 15.0

# 一条用例要算通过，总分得达到这条线，且没有关键错误。
PASS_SCORE = 80.0

# 单位经济需要的变量。
REQUIRED_PARAMS: tuple[str, ...] = (
    "p_in",
    "p_out",
    "cache_hit_ratio",
    "cache_discount",
    "input_tokens",
    "output_tokens",
    "retrieval_cost",
    "retrieval_calls",
    "tool_cost",
    "tool_calls",
    "sessions_per_user",
    "price_per_user_month",
    "active_users",
    "build_cost",
    "support_cost_per_user_month",
)


def score(cases: list[dict]) -> dict:
    """给一批评测用例打分。

    规则：
    - 用例里只要有 critical（关键错误），这条直接判 0 分、不通过——
      关键错误意味着整个输出不可用，平均分会把这种失效摊平；
    - 否则按软错误条数扣分：每条扣 15 分，扣到 0 为止；
    - 一条用例通过 = 没有关键错误 且 分数 ≥ 80。
    """
    if not cases:
        raise FixtureError("评测用例不能为空")

    per_case: list[dict] = []
    total_score = 0.0
    passed_count = 0

    for case in cases:
        critical = case.get("critical", [])
        errors = case.get("errors", [])
        if not isinstance(critical, list) or not isinstance(errors, list):
            raise FixtureError(f"用例 {case.get('id', '?')} 的 critical / errors 必须是列表")

        if critical:
            case_score = 0.0
            reason = f"关键错误 {len(critical)} 条：一票否决"
        else:
            case_score = max(0.0, 100.0 - SOFT_ERROR_PENALTY * len(errors))
            reason = f"软错误 {len(errors)} 条：每条扣 {SOFT_ERROR_PENALTY:.0f} 分"

        case_passed = not critical and case_score >= PASS_SCORE
        passed_count += 1 if case_passed else 0
        total_score += case_score
        per_case.append(
            {
                "id": case["id"],
                "task": case.get("task", ""),
                "score": case_score,
                "passed": case_passed,
                "critical": len(critical),
                "errors": len(errors),
                "reason": reason,
            }
        )

    count = len(cases)
    return {
        "total": total_score / count,
        "pass_rate": passed_count / count,
        "passed": passed_count,
        "total_cases": count,
        "critical_cases": sum(1 for item in per_case if item["critical"]),
        "per_case": per_case,
        "rules": {
            "soft_error_penalty": SOFT_ERROR_PENALTY,
            "pass_score": PASS_SCORE,
            "veto": "关键错误一票否决：该条判 0 分且不通过",
        },
    }


def gate(before: dict, after: dict) -> str:
    """比较改动前后的总分，给出能不能上线的结论。

    规则只有一条：改动后的总分不得低于改动前。允许持平，不允许变差。
    """
    before_total = float(before["total"])
    after_total = float(after["total"])
    delta = after_total - before_total

    if after_total < before_total:
        return (
            f"不通过：{before_total:.2f} → {after_total:.2f}（{delta:+.2f}）——不许变差，"
            f"这次改动不许上线"
        )
    if delta == 0:
        return f"通过：{before_total:.2f} → {after_total:.2f}（持平）——没有变差，但也没有证据说变好"
    return f"通过：{before_total:.2f} → {after_total:.2f}（{delta:+.2f}）——分数没有下降，可以进入下一轮"


def unit_economics(params: dict[str, float]) -> dict:
    """算单位经济：单会话成本 → 每用户月成本 → 毛利率 → 回本周期。

    公式（全部是变量表达，代入真实单价即可）：
        计费输入 token = 输入量 × (1 - 缓存命中率 + 缓存命中率 × 缓存折扣)
        模型成本 = 计费输入 token / 1000 × p_in + 输出量 / 1000 × p_out
        单会话成本 = 模型成本 + 检索次数 × 检索单价 + 工具次数 × 工具单价
        每用户月成本 = 单会话成本 × 每用户月会话数
        每用户月毛利 = 每用户月价 - 每用户月成本 - 每用户月支持成本
        毛利率 = 每用户月毛利 / 每用户月价
        月度毛利总额 = 每用户月毛利 × 付费用户数
        回本周期（月） = 一次性投入 / 月度毛利总额

    回本周期按整个用户基数算，不是按单个用户——有用户基数时，单用户视角下的
    回本周期是没有意义的数字（会把一个 10 个月的回本显示成 1000 个月）。
    """
    for name in REQUIRED_PARAMS:
        if name not in params:
            raise FixtureError(f"单位经济缺少参数：{name}")
    for name, value in params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise FixtureError(f"参数 {name} 必须是数值，实际 {value!r}")
        if value < 0:
            raise FixtureError(f"参数 {name} 不能为负，实际 {value}")
    for name in ("cache_hit_ratio", "cache_discount"):
        if params[name] > 1:
            raise FixtureError(f"参数 {name} 必须在 0–1 之间，实际 {params[name]}")

    billed_input = params["input_tokens"] * (
        (1 - params["cache_hit_ratio"]) + params["cache_hit_ratio"] * params["cache_discount"]
    )
    model_cost = (
        billed_input / 1000 * params["p_in"] + params["output_tokens"] / 1000 * params["p_out"]
    )
    session_cost = (
        model_cost
        + params["retrieval_calls"] * params["retrieval_cost"]
        + params["tool_calls"] * params["tool_cost"]
    )
    monthly_cost = session_cost * params["sessions_per_user"]
    price = params["price_per_user_month"]
    monthly_profit = price - monthly_cost - params["support_cost_per_user_month"]
    gross_margin = monthly_profit / price if price else 0.0
    total_monthly_profit = monthly_profit * params["active_users"]
    payback = params["build_cost"] / total_monthly_profit if total_monthly_profit > 0 else float("inf")

    return {
        "billed_input_tokens": billed_input,
        "model_cost_per_session": model_cost,
        "session_cost": session_cost,
        "monthly_cost_per_user": monthly_cost,
        "price_per_user_month": price,
        "monthly_profit_per_user": monthly_profit,
        "gross_margin": gross_margin,
        "total_monthly_profit": total_monthly_profit,
        "payback_months": payback,
        "formulas": {
            "计费输入 token": "输入量 × (1 - 缓存命中率 + 缓存命中率 × 缓存折扣)",
            "单会话成本": "模型成本 + 检索次数 × 检索单价 + 工具次数 × 工具单价",
            "每用户月成本": "单会话成本 × 每用户月会话数",
            "每用户月毛利": "每用户月价 - 每用户月成本 - 每用户月支持成本",
            "毛利率": "每用户月毛利 ÷ 每用户月价",
            "回本周期": "一次性投入 ÷ (每用户月毛利 × 付费用户数)",
        },
    }


def flat_params(fixture: dict[str, dict]) -> dict[str, float]:
    """把 cost_model.json 的变量表压成扁平参数字典。"""
    return {name: float(spec["value"]) for name, spec in fixture["variables"].items()}


def run() -> None:
    """跑第 6 周实验：评测打分 → 门禁 → 单位经济。"""
    print("=" * 78)
    print("第 6 周 · 评测门禁与单位经济")
    print("=" * 78)

    data = load_fixture("eval_cases")
    cases = data["cases"]

    # ---- 1. 评测打分 ----
    result = score(cases)
    print()
    print("【1】评测打分（关键错误一票否决 + 软错误扣分）")
    print(
        fmt_table(
            ["编号", "任务", "得分", "是否通过", "判定依据"],
            [
                [
                    item["id"],
                    item["task"],
                    f"{item['score']:.0f}",
                    "通过" if item["passed"] else "不通过",
                    item["reason"],
                ]
                for item in result["per_case"]
            ],
            align_right={2},
            max_widths=[6, 30, 4, 8, 24],
        )
    )
    print()
    print(
        fmt_table(
            ["项", "值"],
            [
                ["用例总数", str(result["total_cases"])],
                ["平均总分", f"{result['total']:.2f}"],
                ["通过率", f"{pct(result['pass_rate'])}（{result['passed']}/{result['total_cases']}）"],
                ["含关键错误的用例", f"{result['critical_cases']} 条"],
                ["软错误扣分", f"每条 {SOFT_ERROR_PENALTY:.0f} 分"],
                ["通过线", f"总分 ≥ {PASS_SCORE:.0f} 且无关键错误"],
            ],
            align_right={1},
        )
    )

    # ---- 2. 门禁 ----
    print()
    print("【2】评测门禁：改动后不许变差")
    fixed_cases = []
    for case in cases:
        clone = dict(case)
        if clone["id"] == "EC-07":
            clone["critical"] = []
            clone["errors"] = ["没有说明越权检查在哪一步"]
        fixed_cases.append(clone)
    after = score(fixed_cases)

    regressed_cases = fixed_cases + [
        {
            "id": "EC-REGRESS",
            "task": "回归用例：把已修好的关键错误改回来",
            "answer": "生成的实现直接访问了未授权的表",
            "critical": ["越权访问"],
            "errors": [],
            "notes": "故意做坏，用来演示门禁会拦住",
        }
    ]
    regressed = score(regressed_cases)

    print(f"  基线：总分 {result['total']:.2f}，通过率 {pct(result['pass_rate'])}")
    print(f"  修好 EC-07 后：{gate(result, after)}")
    print(f"  故意做坏后：{gate(after, regressed)}")

    # ---- 3. 单位经济 ----
    print()
    print("【3】单位经济（示意测算，单价为变量）")
    cost_fixture = load_fixture("cost_model")
    params = flat_params(cost_fixture)
    econ = unit_economics(params)
    print(
        fmt_table(
            ["变量", "取值", "单位"],
            [
                [name, f"{spec['value']:g}", spec["unit"]]
                for name, spec in cost_fixture["variables"].items()
            ],
            align_right={1},
            max_widths=[24, 10, 18],
        )
    )
    print()
    print(
        fmt_table(
            ["测算项", "结果", "公式"],
            [
                [
                    "计费输入 token",
                    f"{econ['billed_input_tokens']:.0f}",
                    econ["formulas"]["计费输入 token"],
                ],
                [
                    "单会话成本",
                    f"{econ['session_cost']:.4f} 元",
                    econ["formulas"]["单会话成本"],
                ],
                [
                    "每用户月成本",
                    f"{econ['monthly_cost_per_user']:.2f} 元",
                    econ["formulas"]["每用户月成本"],
                ],
                [
                    "每用户月毛利",
                    f"{econ['monthly_profit_per_user']:.2f} 元",
                    econ["formulas"]["每用户月毛利"],
                ],
                [
                    "毛利率",
                    pct(econ["gross_margin"]),
                    econ["formulas"]["毛利率"],
                ],
                [
                    "月度毛利总额",
                    f"{econ['total_monthly_profit']:.2f} 元",
                    "每用户月毛利 × 付费用户数",
                ],
                [
                    "回本周期",
                    f"{econ['payback_months']:.1f} 个月",
                    econ["formulas"]["回本周期"],
                ],
            ],
            max_widths=[16, 18, 56],
        )
    )

    # ---- 断言 ----
    assert 0 <= result["total"] <= 100
    assert 0 <= result["pass_rate"] <= 1
    for item in result["per_case"]:
        if item["critical"]:
            assert item["score"] == 0 and not item["passed"], "有关键错误的用例必须 0 分且不通过"
        else:
            assert item["score"] == max(0.0, 100.0 - SOFT_ERROR_PENALTY * item["errors"])

    assert "不许变差" in gate(after, regressed), "门禁必须检出变差"
    assert "不许变差" not in gate(result, after), "变好的改动不该被判失败"
    assert after["total"] > result["total"], "修掉关键错误后总分必须上升"

    base = flat_params(cost_fixture)
    no_cache = unit_economics({**base, "cache_hit_ratio": 0.0})
    full_cache = unit_economics({**base, "cache_hit_ratio": 1.0})
    assert abs(no_cache["billed_input_tokens"] - base["input_tokens"]) < 1e-9, (
        "缓存命中率为 0 时应按全部输入计费"
    )
    assert abs(
        full_cache["billed_input_tokens"] - base["input_tokens"] * base["cache_discount"]
    ) < 1e-9, "缓存命中率为 1 时应按折扣后的输入计费"
    assert econ["gross_margin"] > 0, "样例参数的毛利率应为正"
    assert econ["payback_months"] > 0
    for bad in ({**base, "price_per_user_month": -1}, {**base, "cache_hit_ratio": 1.5}):
        try:
            unit_economics(bad)
        except FixtureError:
            pass
        else:
            raise AssertionError(f"非法参数应该报错：{bad}")

    print()
    print("要点：关键错误一票否决不是为了让分数好看，而是因为它属于「整体失效」——")
    print(f"      这一批 {result['critical_cases']} 条关键错误直接把通过率按掉了 "
          f"{pct(result['critical_cases'] / result['total_cases'])}。")
    print("      门禁只看总分会不会掉，不看这次改动「好像更聪明」——")
    print("      没证据的改进不算改进，总分持平就只能说明没变差。")
    print("      单位经济的所有单价都是变量：接入真实模型时只替换数字，公式不用动。")
