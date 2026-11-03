"""每日复盘：把当天的执行结果变成下一步的依据。

复盘做三件事：

1. **对账**：今天计划了几条、做完几条、花了多少分钟。
2. **找规律**：连续几天完成率低 → 计划量太大；某类任务反复卡住 → 那是弱点。
3. **沉淀**：把当天值得长期记住的结论晋升进长期档案。

第 3 步是本示例的核心链路：复盘产生的"经验"如果只留在当天的短期记忆里，
明天就忘了；晋升到长期档案后，下次规划时能被召回，
规划就会自动避开上次的坑。

规律识别走规则而不是模型，因为规则的行为可预测、可测试，
复盘结论的来源也更容易追溯。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .models import (
    CATEGORY_PREFERENCE,
    CATEGORY_REVIEW,
    CATEGORY_STRENGTH,
    CATEGORY_SUMMARY,
    CATEGORY_WEAKNESS,
    Goal,
    PlanItem,
    Review,
    today_str,
)
from .planner import items_for_day, next_undone

# 完成率低于这个值，连续两天就判定"计划量过大"
LOW_COMPLETION = 0.5
LOW_COMPLETION_STREAK = 2


@dataclass
class ReviewOutcome:
    """一次复盘的结果：记录 + 沉淀的结论 + 下一步建议。"""

    review: Review
    insights: list[str]
    promotions: list[str]
    next_step: str

    def render(self) -> str:
        lines = [
            f"【{self.review.day} 复盘】",
            f"  计划 {self.review.planned} 条，完成 {self.review.done} 条，"
            f"投入 {self.review.minutes} 分钟",
        ]

        if self.review.highlights:
            lines.append(f"  亮点：{'；'.join(self.review.highlights)}")

        if self.review.blockers:
            lines.append(f"  卡点：{'；'.join(self.review.blockers)}")

        if self.insights:
            lines.append("  规律：")

            for insight in self.insights:
                lines.append(f"    - {insight}")

        if self.promotions:
            lines.append("  已沉淀到长期记忆：")

            for promotion in self.promotions:
                lines.append(f"    - {promotion}")

        if self.next_step:
            lines.append(f"  下一步：{self.next_step}")

        return "\n".join(lines)


def collect_day(
    items: list[PlanItem], day: str | None = None
) -> tuple[list[PlanItem], int, int]:
    """对账：取出当天的待办、已完成条数、投入分钟。

    返回 (当天待办, 完成条数, 投入分钟)。
    """

    today = day or today_str()
    scoped = items_for_day(items, today)
    done = [item for item in scoped if item.done]
    minutes = sum(item.minutes for item in done)

    return scoped, len(done), minutes


def find_patterns(
    reviews: list[Review], items: list[PlanItem], day: str | None = None
) -> list[str]:
    """从历史复盘里找规律。

    目前识别两类：
    - 连续低完成率 → 计划量过大，应该减量
    - 同一天反复未完成的任务 → 难点
    """

    insights: list[str] = []
    target = day or today_str()

    # 兼容已有的同日重复记录：每个自然日以最后一次复盘为准。
    by_day = {review.day: review for review in reviews}
    expected_days = [
        (date.fromisoformat(target) - timedelta(days=offset)).isoformat()
        for offset in reversed(range(LOW_COMPLETION_STREAK))
    ]
    recent = [by_day[day] for day in expected_days if day in by_day]

    if len(recent) == LOW_COMPLETION_STREAK:
        low = all(
            review.planned > 0 and review.completion < LOW_COMPLETION
            for review in recent
        )

        if low:
            days = "、".join(review.day for review in recent)
            insights.append(
                f"连续 {len(recent)} 天（{days}）完成率低于 {int(LOW_COMPLETION * 100)}%，"
                "建议把每日任务量调低一档"
            )

    # 同一条任务被顺延多次 → 那是个难点
    today = target
    yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    titles_today = {item.title for item in items_for_day(items, today) if not item.done}
    titles_yesterday = {item.title for item in items_for_day(items, yesterday) if not item.done}
    repeated = titles_today & titles_yesterday

    for title in sorted(repeated):
        insights.append(f"「{title}」连续两天未完成，可能是难点")

    return insights


def derive_promotions(
    review: Review, insights: list[str], goals: list[Goal]
) -> list[tuple[str, str, int]]:
    """从复盘里推导"值得长期记住"的结论。

    返回 [(内容, 类别, 重要度)]。只挑真正会改变未来行为的：
    - 长期没完成的目标 → 弱点/需要调整
    - 连续完成 → 优势
    - 计划量需要调整 → 偏好（长期的学习节奏）
    """

    promotions: list[tuple[str, str, int]] = []

    for insight in insights:
        if "完成率低于" in insight:
            promotions.append((insight, CATEGORY_PREFERENCE, 4))
        elif "连续两天未完成" in insight:
            title = insight.split("「")[1].split("」")[0] if "「" in insight else insight
            promotions.append((f"反复卡在「{title}」这件事上", CATEGORY_WEAKNESS, 4))

    if review.planned > 0 and review.completion >= 0.9 and review.planned >= 2:
        promotions.append(
            (f"{review.day} 完成了 {review.done}/{review.planned} 条任务，状态很好", CATEGORY_STRENGTH, 3)
        )

    for goal in goals:
        if goal.status == "active" and goal.deadline:
            try:
                deadline = date.fromisoformat(goal.deadline)
            except ValueError:
                continue

            if (deadline - date.fromisoformat(review.day)).days < 0:
                promotions.append(
                    (f"目标「{goal.title}」已经超过截止日期仍未完成", CATEGORY_WEAKNESS, 5)
                )

    return promotions


def summarize_day(review: Review, scoped: list[PlanItem]) -> tuple[list[str], list[str]]:
    """把当天的任务分成亮点和卡点。"""

    highlights = [item.title for item in scoped if item.done]
    blockers = [item.title for item in scoped if not item.done]

    return highlights, blockers


def run_review(
    items: list[PlanItem],
    reviews: list[Review],
    goals: list[Goal],
    day: str | None = None,
) -> ReviewOutcome:
    """跑一次复盘。

    参数：
        items   所有计划待办
        reviews 历史复盘（用来找规律）
        goals   当前目标（用来检查是否逾期）
        day     复盘哪一天，默认今天
    返回：
        ReviewOutcome，含复盘记录、规律、待沉淀的结论、下一步建议
    """

    target = day or today_str()
    scoped, done, minutes = collect_day(items, target)
    planned = len(scoped)
    highlights, blockers = summarize_day(Review(day=target, planned=planned), scoped)

    review = Review(
        day=target,
        planned=planned,
        done=done,
        minutes=minutes,
        highlights=highlights,
        blockers=blockers,
    )

    insights = find_patterns(reviews + [review], items, target)
    promotions = derive_promotions(review, insights, goals)

    pending = next_undone(items, target)
    next_step = ""
    if pending is not None:
        next_step = f"接着做「{pending.title}」（{pending.day}，{pending.minutes} 分钟）"

    return ReviewOutcome(
        review=review,
        insights=insights,
        promotions=[content for content, _, _ in promotions],
        next_step=next_step,
    )


def apply_review_findings(memory, outcome: ReviewOutcome) -> tuple[list[str], list[str]]:
    """把复盘结果落到两级记忆里。

    两级各有分工：
    - **短期**：复盘摘要（今天完成了几条、花了多少时间）属于当天的工作上下文，
      明天回头看还该记得，写进工作区。
    - **长期**：只有会改变未来行为的强信号才晋升——
      连续低完成率、反复卡在同一件事、目标逾期。

    返回 (晋升到长期的内容, 写入短期工作区的内容)。
    """

    short_written: list[str] = []
    applied: list[str] = []

    # 1. 复盘摘要进短期工作区
    summary = (
        f"{outcome.review.day} 复盘：完成 {outcome.review.done}/{outcome.review.planned} 条，"
        f"投入 {outcome.review.minutes} 分钟"
    )
    memory.remember(summary, CATEGORY_REVIEW, importance=3)
    short_written.append(summary)

    # 2. 强信号晋升长期
    for insight in outcome.insights:
        category = CATEGORY_PREFERENCE

        if "连续两天未完成" in insight:
            category = CATEGORY_WEAKNESS
        elif "状态很好" in insight:
            category = CATEGORY_STRENGTH

        memory.promote(insight, category, "每日复盘沉淀", importance=4)
        applied.append(insight)

    for content in outcome.promotions:
        if content in applied:
            continue

        category = CATEGORY_SUMMARY

        if "弱点" in content or "未完成" in content:
            category = CATEGORY_WEAKNESS
        elif "完成" in content and "条任务" in content:
            category = CATEGORY_STRENGTH

        memory.promote(content, category, "每日复盘沉淀", importance=4)
        applied.append(content)

    return applied, short_written


def rollover_day(memory, day: str | None = None) -> list[str]:
    """跨天：先晋升再清空短期工作区。

    参数：
        memory MemoryManager 实例
        day    新的日期
    返回：
        跨天时沉淀进长期档案的内容列表
    """

    decisions = memory.rollover(day)

    return list(decisions)
