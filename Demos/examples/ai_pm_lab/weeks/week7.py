"""第 7 周实验：风险登记册与指标树。

这个模块解决什么问题：把"我担心会出问题"变成一张有触发条件、有责任人、有复检周期的登记表；
把"我们要提升什么"变成一棵三层指标树——北极星、护栏、诊断，外加一条能算出转化率的漏斗。

两份文档的共同要求是"可判定"：风险要能被触发，指标要能被证伪。
写不出触发条件或判定规则的风险，写的是焦虑不是风险；
写不出判定方法的北极星，年底只能靠感觉汇报。

对应笔记：AI/AI-Product/ 第 13 章《风险登记册》与第 14 章《指标树》。
"""
from __future__ import annotations

from ..labkit import (
    CAPSTONE_DIR,
    FixtureError,
    Report,
    fmt_table,
    load_fixture,
    parse_md_table,
    pct,
    read_text,
)

# 风险登记册的必填字段，一个都不能少。
RISK_FIELDS: tuple[str, ...] = (
    "风险描述",
    "触发条件",
    "影响",
    "等级",
    "缓解措施",
    "责任人",
    "复检周期",
)

# 指标树的三层。缺任何一层，指标体系就会退化成"只有北极星"。
METRIC_LAYERS: tuple[str, ...] = ("北极星", "护栏", "诊断")


def check_risk_register(risks: list[dict]) -> Report:
    """检查风险登记册：必填字段齐不齐，缺哪个点名哪个。"""
    report = Report(subject="风险登记册", passed=True)
    if not risks:
        report.passed = False
        report.problems.append("风险登记册为空：没有登记任何风险")
        return report

    for index, risk in enumerate(risks, start=1):
        name = risk.get("编号") or f"第 {index} 条"
        missing = [field_name for field_name in RISK_FIELDS if not str(risk.get(field_name, "")).strip()]
        if missing:
            report.problems.append(f"{name}：缺 {'、'.join(missing)}")
        else:
            report.ok.append(f"{name}（{risk['等级']}）")
    report.passed = not report.problems
    return report


def metric_tree(metrics: dict) -> dict:
    """校验指标树的三层结构，并算漏斗各环节转化率。

    北极星必须存在且"可判定"（falsifiable 为真）——
    不可判定的北极星在评审会上没法被证伪，也就等于没有。
    """
    north_star = metrics.get("north_star")
    if not isinstance(north_star, dict) or not north_star.get("name"):
        raise FixtureError("指标树缺少北极星指标")
    if not north_star.get("falsifiable"):
        raise FixtureError(
            f"北极星指标「{north_star.get('name', '?')}」不可判定：falsifiable 必须为真，"
            "否则年底没法被证伪"
        )

    issues: list[str] = []
    layers: dict[str, bool] = {"北极星": True}

    guardrails = metrics.get("guardrails", [])
    if not guardrails:
        issues.append("护栏层缺失：没有任何护栏指标")
    layers["护栏"] = bool(guardrails)
    for guardrail in guardrails:
        missing = [key for key in ("name", "threshold", "owner") if not guardrail.get(key)]
        if missing:
            issues.append(f"护栏「{guardrail.get('name', '?')}」缺字段：{'、'.join(missing)}")

    diagnostics = metrics.get("diagnostics", [])
    if not diagnostics:
        issues.append("诊断层缺失：没有任何诊断指标")
    layers["诊断"] = bool(diagnostics)
    for diagnostic in diagnostics:
        missing = [key for key in ("name", "question") if not diagnostic.get(key)]
        if missing:
            issues.append(f"诊断「{diagnostic.get('name', '?')}」缺字段：{'、'.join(missing)}")

    funnel: list[dict] = []
    stages = metrics.get("funnel", [])
    previous: int | None = None
    for stage in stages:
        users = int(stage["users"])
        conversion = 1.0 if previous is None else (users / previous if previous else 0.0)
        funnel.append(
            {
                "stage": stage["stage"],
                "users": users,
                "conversion": conversion,
            }
        )
        previous = users
    overall = (funnel[-1]["users"] / funnel[0]["users"]) if funnel else 0.0

    return {
        "north_star": north_star,
        "layers": layers,
        "issues": issues,
        "funnel": funnel,
        "overall_conversion": overall,
        "passed": not issues,
    }


def run() -> None:
    """跑第 7 周实验：风险登记册 → 指标树 → 漏斗转化。"""
    print("=" * 78)
    print("第 7 周 · 风险登记册与指标树")
    print("=" * 78)

    # ---- 1. 风险登记册 ----
    print()
    print("【1】风险登记册体检")
    register_text = read_text(CAPSTONE_DIR / "risk_register.md")
    _, rows = parse_md_table(register_text)
    risks = [
        {
            "编号": row.get("编号", ""),
            "风险描述": row.get("风险描述", ""),
            "触发条件": row.get("触发条件", ""),
            "影响": row.get("影响", ""),
            "等级": row.get("等级", ""),
            "缓解措施": row.get("缓解措施", ""),
            "责任人": row.get("责任人", ""),
            "复检周期": row.get("复检周期", ""),
        }
        for row in rows
    ]
    report = check_risk_register(risks)
    print(
        fmt_table(
            ["编号", "风险描述", "等级", "责任人", "复检周期"],
            [
                [
                    risk["编号"],
                    risk["风险描述"],
                    risk["等级"],
                    risk["责任人"],
                    risk["复检周期"],
                ]
                for risk in risks
            ],
            max_widths=[6, 30, 6, 10, 12],
        )
    )
    print()
    print(f"  登记册结论：{'通过' if report.passed else '不通过'}（{len(report.ok)}/{len(risks)} 条完整）")
    for problem in report.problems:
        print(f"      ✗ {problem}")

    # 故意漏填一条，看看能不能被点名。
    broken = [dict(risk) for risk in risks]
    broken[0]["复检周期"] = ""
    broken[1]["责任人"] = ""
    broken_report = check_risk_register(broken)
    print()
    print("  故意漏填后的体检结果：")
    for problem in broken_report.problems:
        print(f"      ✗ {problem}")

    # ---- 2. 指标树 ----
    metrics = load_fixture("metrics")
    tree = metric_tree(metrics)
    print()
    print("【2】指标树三层结构")
    print(
        fmt_table(
            ["层", "指标", "判定方式 / 阈值", "责任人"],
            [
                [
                    "北极星",
                    tree["north_star"]["name"],
                    tree["north_star"].get("definition", ""),
                    tree["north_star"].get("owner", ""),
                ],
                *[
                    ["护栏", item["name"], item.get("threshold", ""), item.get("owner", "")]
                    for item in metrics["guardrails"]
                ],
                *[
                    ["诊断", item["name"], item.get("question", ""), ""]
                    for item in metrics["diagnostics"]
                ],
            ],
            max_widths=[6, 22, 46, 12],
        )
    )
    for issue in tree["issues"]:
        print(f"      ✗ {issue}")

    # ---- 3. 漏斗 ----
    print()
    print("【3】漏斗转化")
    print(
        fmt_table(
            ["环节", "人数", "较上一环节"],
            [
                [
                    item["stage"],
                    str(item["users"]),
                    "—" if item["conversion"] == 1.0 else pct(item["conversion"]),
                ]
                for item in tree["funnel"]
            ],
            align_right={1, 2},
        )
    )
    print()
    print(f"  全链路转化：{pct(tree['overall_conversion'])}")

    # ---- 断言 ----
    assert len(RISK_FIELDS) == 7, "风险登记册必填字段应为 7 个"
    assert report.passed, "毕业作品的风险登记册必须字段齐全"
    assert len(risks) >= 3, "风险登记册至少登记 3 条"
    assert not broken_report.passed, "漏填字段时必须判不合格"
    assert any("复检周期" in problem for problem in broken_report.problems), "缺复检周期要点名"
    assert any("责任人" in problem for problem in broken_report.problems), "缺责任人要点名"

    assert tree["layers"]["北极星"] and tree["layers"]["护栏"] and tree["layers"]["诊断"], (
        "三层结构必须齐全"
    )
    assert tree["north_star"]["falsifiable"], "北极星必须可判定"
    assert abs(tree["overall_conversion"] - 1150 / 10000) < 1e-9, "全链路转化率算错"
    assert abs(tree["funnel"][1]["conversion"] - 4200 / 10000) < 1e-9, "第二个环节转化率算错"

    try:
        metric_tree({"north_star": {"name": "X", "falsifiable": False}})
    except FixtureError:
        pass
    else:
        raise AssertionError("北极星不可判定时必须报错")
    try:
        metric_tree({"guardrails": []})
    except FixtureError:
        pass
    else:
        raise AssertionError("缺北极星时必须报错")
    missing_guardrail = metric_tree(
        {
            "north_star": {"name": "X", "falsifiable": True},
            "guardrails": [],
            "diagnostics": [{"name": "d", "question": "q"}],
        }
    )
    assert not missing_guardrail["passed"], "护栏缺失必须被点名"
    assert any("护栏" in issue for issue in missing_guardrail["issues"])

    print()
    print("要点：风险登记册里最没用的字段是「风险描述」——最好用的字段是「触发条件」。")
    print("      因为触发条件是可监控的：它一旦被观察到，风险就已经在发生，而不是「可能发生」。")
    print("      指标树同理：北极星不能被证伪，护栏就会被悄悄牺牲。")
    print("      漏斗里最该盯的不是总量，而是掉得最狠的那一环。")
