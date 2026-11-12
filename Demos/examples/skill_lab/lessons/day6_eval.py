"""第 6 天实验：评测与迭代。

这个模块解决什么问题：把"这个技能写得好不好"从主观感觉变成一个可复现的数字。
它加载第 2 天写下的三版 description，用同一份触发评测集打分，
看改动是不是收敛的，并指出下一版该改什么。

⚠️ 这里用的是**关键词命中代理指标**，不是真实模型判断。
真实评测要把 query 和 description 一起喂给模型（或人工）问"该不该触发"。
代理指标的价值在于：免费、可复现、能在写评测脚本时快速定位问题。

对应笔记：AI/Skill/06-评测与迭代.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .day2_description import DRAFT_V1, DRAFT_V2, DRAFT_V3

CASES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "trigger_cases.json"


@dataclass
class Case:
    """一条评测用例。"""

    query: str
    should_trigger: bool
    kind: str


@dataclass
class Outcome:
    """一条用例的评测结果。"""

    case: Case
    predicted: bool
    signals: list[str]
    guards: list[str]

    @property
    def correct(self) -> bool:
        return self.predicted == self.case.should_trigger

    @property
    def reason(self) -> str:
        """判定依据——区分'靠边界排除'和'靠没命中信号'，后者是侥幸通过。"""
        if self.predicted:
            return f"命中信号 {'、'.join(self.signals)}"
        if self.guards:
            return f"靠边界排除（{'、'.join(self.guards)}）"
        return "靠未命中信号排除（侥幸）"


@dataclass
class Report:
    """一轮评测的汇总。"""

    version: str
    positive_hit: int
    positive_total: int
    negative_ok: int
    negative_total: int
    near_miss_ok: int
    near_miss_total: int
    outcomes: list[Outcome]

    @property
    def total_score(self) -> float:
        correct = sum(1 for o in self.outcomes if o.correct)
        return correct / len(self.outcomes) * 100 if self.outcomes else 0.0


def load_cases() -> tuple[list[Case], list[str], list[str]]:
    """读取评测集，返回 (用例, 触发词表, 守卫词表)。"""
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = [Case(q["query"], q["should_trigger"], q["kind"]) for q in data["cases"]]
    return cases, data["trigger_terms"], data["guard_terms"]


def evaluate(description: str, cases: list[Case], trigger_terms: list[str], guard_terms: list[str]) -> Report:
    """对一份 description 跑一遍评测。

    判定模型（代理指标）：
      signals = description 覆盖了、且 query 里出现的触发词
      guards  = description 覆盖了、且 query 里出现的守卫词
      预测触发 = 有 signals 且没有 guards
    """
    outcomes: list[Outcome] = []
    for case in cases:
        signals = [t for t in trigger_terms if t in description and t in case.query]
        guards = [g for g in guard_terms if g in description and g in case.query]
        predicted = bool(signals) and not guards
        outcomes.append(Outcome(case=case, predicted=predicted, signals=signals, guards=guards))

    positives = [o for o in outcomes if o.case.should_trigger]
    negatives = [o for o in outcomes if not o.case.should_trigger]
    near = [o for o in negatives if o.case.kind == "near-miss"]

    return Report(
        version="",
        positive_hit=sum(1 for o in positives if o.correct),
        positive_total=len(positives),
        negative_ok=sum(1 for o in negatives if o.correct),
        negative_total=len(negatives),
        near_miss_ok=sum(1 for o in near if o.correct),
        near_miss_total=len(near),
        outcomes=outcomes,
    )


def render(report: Report, version: str) -> str:
    """把一轮评测渲染成报告文本。"""
    lines = [f"  {version}"]
    lines.append(f"    正例召回   {report.positive_hit}/{report.positive_total}")
    lines.append(f"    负例排除   {report.negative_ok}/{report.negative_total}")
    lines.append(f"    近邻排除   {report.near_miss_ok}/{report.near_miss_total}")
    lines.append(f"    总分       {report.total_score:.0f}%")
    return "\n".join(lines)


def run() -> None:
    print("=" * 78)
    print("第 6 天 · 评测与迭代")
    print("=" * 78)

    cases, trigger_terms, guard_terms = load_cases()
    print()
    print(f"【1】评测集：{len(cases)} 条用例（{sum(1 for c in cases if c.should_trigger)} 正 / "
          f"{sum(1 for c in cases if not c.should_trigger)} 负，含 "
          f"{sum(1 for c in cases if c.kind == 'near-miss')} 条近邻负例）")
    print(f"  触发词表（{len(trigger_terms)} 个）：{'、'.join(trigger_terms)}")
    print(f"  守卫词表（{len(guard_terms)} 个）：{'、'.join(guard_terms)}")

    # ---- 2. 三轮迭代 ----
    print()
    print("【2】三轮迭代：第 2 天写下的三版 description")
    versions = [
        ("V1 朴素版", DRAFT_V1),
        ("V2 官方范式版", DRAFT_V2),
        ("V3 四段式版", DRAFT_V3),
    ]
    reports: list[Report] = []
    for label, desc in versions:
        report = evaluate(desc, cases, trigger_terms, guard_terms)
        report.version = label
        reports.append(report)
        print(render(report, label))

    assert reports[0].total_score < reports[1].total_score < reports[2].total_score, \
        "迭代应当是收敛的：每轮改写都应当让分数上升"

    # ---- 3. 未命中的用例 ----
    print()
    print("【3】最终版本仍未处理好的用例")
    final = reports[-1]
    remaining = [o for o in final.outcomes if not o.correct]
    if not remaining:
        print("  （无）")
    for o in remaining:
        print(f"  ✗ 「{o.case.query}」 应触发={o.case.should_trigger} 实测={o.predicted}（{o.reason}）")

    # ---- 4. 侥幸通过的提醒 ----
    print()
    print("【4】各版本的负例是怎么被排除的")
    for report in reports:
        lucky = [o for o in report.outcomes if not o.case.should_trigger and not o.correct]
        guarded = [o for o in report.outcomes if not o.case.should_trigger and o.correct and o.guards]
        print(f"  {report.version}: 靠边界排除 {len(guarded)} 条，靠没命中信号排除 {report.negative_total - len(guarded) - len(lucky)} 条，误触发 {len(lucky)} 条")

    print()
    print("要点：V1 的负例全对，但那是'什么都没匹配上'的侥幸——")
    print("      真正健壮的排除靠的是 description 里写清楚的边界词。")


if __name__ == "__main__":
    run()
