"""计划生成：把一个目标拆成可执行的天级待办。

拆解逻辑走的是**规则**而不是模型，原因是规则版本更容易看清"计划是怎么来的"，
而且"每天排几条"这种约束本来就该由代码说了算。要点有三个：

1. 里程碑先行：先把目标切成 2~4 个阶段，再把阶段铺到具体日期。
2. 每天限量：按剩余天数和每日可用时间决定每天排几条，避免排出一堆做不完的事。
3. 留白：最后一天不排任务，留给补做和缓冲。

模型可以介入"生成阶段名称"，但排期和条数始终由规则说了算——
这和"用模型做判断、用代码做约束"是同一个原则。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from .models import Goal, PlanItem, stable_id, today_str

# 常见学习目标的阶段模板。命中不了就走通用模板。
DOMAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "编程": ("环境与基础语法", "核心概念与练习", "小项目实战", "复盘与查漏补缺"),
    "语言": ("发音与基础词汇", "语法与听力", "口语与场景表达", "模拟对话与复盘"),
    "考试": ("考纲拆解与摸底", "专题突破", "真题与限时训练", "冲刺与查漏"),
    "读书": ("通读建立框架", "精读重点章节", "笔记与输出", "回顾与迁移"),
    "健身": ("基础适应", "动作规范与力量", "强度提升", "巩固与习惯固化"),
}

GENERIC_TEMPLATE = ("入门与建立框架", "核心练习", "综合应用", "复盘与巩固")

# 领域关键词 → 领域模板。键里的词只要出现在标题里就命中。
# 比只匹配领域名本身实用：用户通常说"学 Python"而不是"学编程"。
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "编程": ("编程", "代码", "python", "java", "javascript", "go", "rust", "c++", "算法", "前端", "后端", "数据分析"),
    "语言": ("英语", "日语", "法语", "德语", "韩语", "口语", "雅思", "托福", "四级", "六级"),
    "考试": ("考研", "考试", "真题", "中考", "高考", "公考", "考证", "证书"),
    "读书": ("读书", "阅读", "看书", "论文", "书单"),
    "健身": ("健身", "跑步", "增肌", "减脂", "锻炼", "瑜伽", "游泳"),
}

DEFAULT_DAILY_MINUTES = 60
MAX_TASKS_PER_DAY = 3

# 目标没有写截止日期、或截止日期已过时，用这个周期重排
FALLBACK_DAYS = 14


@dataclass
class PlanDraft:
    """一份计划草稿：里程碑 + 每日待办。"""

    milestones: list[str]
    items: list[PlanItem]
    note: str = ""

    def render(self) -> str:
        lines = [f"目标拆成 {len(self.milestones)} 个阶段，{len(self.items)} 条日任务："]

        for index, milestone in enumerate(self.milestones, start=1):
            lines.append(f"  阶段{index}：{milestone}")

        by_day: dict[str, list[PlanItem]] = {}

        for item in self.items:
            by_day.setdefault(item.day, []).append(item)

        for day in sorted(by_day):
            titles = "、".join(item.title for item in by_day[day])
            lines.append(f"  {day}：{titles}")

        if self.note:
            lines.append(f"（{self.note}）")

        return "\n".join(lines)


def detect_domain(title: str, detail: str = "") -> str:
    """猜目标属于哪个领域，决定用哪套阶段模板。

    先看领域关键词（"python" → 编程），再看领域名本身（"编程" → 编程）。
    """

    haystack = f"{title} {detail}".lower()

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return domain

    for domain in DOMAIN_TEMPLATES:
        if domain in haystack:
            return domain

    return "通用"


def parse_deadline(goal: Goal, fallback_days: int = FALLBACK_DAYS) -> date:
    """取目标的截止日期；没写就默认若干天后。"""

    if goal.deadline:
        try:
            return datetime.strptime(goal.deadline, "%Y-%m-%d").date()
        except ValueError:
            pass

    return date.today() + timedelta(days=fallback_days)


def build_milestones(goal: Goal, count: int = 4) -> list[str]:
    """按领域模板生成阶段名。

    模板不够用时，最后一个阶段用"巩固与复盘"兜底。
    """

    template = DOMAIN_TEMPLATES.get(detect_domain(goal.title, goal.detail))

    if template is None:
        template = GENERIC_TEMPLATE

    if count <= len(template):
        return list(template[:count])

    stages = list(template)
    extra = count - len(template)

    for index in range(extra):
        stages.insert(-1 if len(stages) > 1 else 0, f"专项强化 {index + 1}")

    return stages


def distribute(
    milestones: list[str],
    start: date,
    end: date,
    daily_minutes: int,
    goal_id: str = "",
    budget_for_day: Callable[[date], int] | None = None,
) -> list[PlanItem]:
    """把阶段铺到具体日期上。

    每天排几条由**剩余天数**和**每日时间**共同决定：
    时间少就少排，排不完的计划比没有计划更伤人。
    """

    total_days = (end - start).days + 1

    if total_days <= 0:
        return []

    # 最后一天留白，用来补做
    work_days = max(1, total_days - 1)
    per_day = max(1, min(MAX_TASKS_PER_DAY, daily_minutes // 30))

    items: list[PlanItem] = []
    days_total = work_days * per_day
    stage_span = max(1, days_total // max(1, len(milestones)))

    counter = 0

    for offset in range(work_days):
        day = start + timedelta(days=offset)
        available = budget_for_day(day) if budget_for_day else daily_minutes
        per_day = max(1, min(MAX_TASKS_PER_DAY, available // 30))

        for slot in range(per_day):
            counter += 1
            stage_index = min(len(milestones) - 1, counter // stage_span)
            stage = milestones[stage_index]

            # 每个阶段的最后一条待办标记成里程碑，便于回顾时对齐阶段
            is_last_of_stage = (counter % stage_span == 0) or counter == days_total

            title = f"{stage} · 第 {slot + 1} 讲"

            items.append(
                PlanItem(
                    id=stable_id("p", goal_id, day.isoformat(), title),
                    goal_id=goal_id,
                    title=title,
                    day=day.isoformat(),
                    minutes=max(15, available // per_day),
                    milestone=is_last_of_stage,
                )
            )

    return items


def generate_plan(
    goal: Goal,
    start: date | None = None,
    daily_minutes: int = DEFAULT_DAILY_MINUTES,
    milestone_count: int = 4,
    budget_for_day: Callable[[date], int] | None = None,
) -> PlanDraft:
    """为单个目标生成计划草稿。

    参数：
        goal            目标本身
        start           起始日期，默认今天
        daily_minutes   每天能投入多少分钟
        milestone_count 切成几个阶段
    返回：
        PlanDraft，装着阶段名和每日待办
    """

    begin = start or date.today()
    end = parse_deadline(goal) if goal.deadline else begin + timedelta(days=FALLBACK_DAYS - 1)
    milestones = build_milestones(goal, count=milestone_count)

    note = ""
    if end < begin:
        # 截止日期已过：仍然给出一份可执行的短期计划，
        # 空计划对用户没有任何帮助，不如按默认周期重排一次。
        note = f"截止日期 {goal.deadline} 已过，先按默认 {FALLBACK_DAYS} 天重排，建议重设目标"
        end = begin + timedelta(days=FALLBACK_DAYS - 1)

    items = distribute(milestones, begin, end, daily_minutes, goal_id=goal.id, budget_for_day=budget_for_day)

    return PlanDraft(milestones=milestones, items=items, note=note)


def plan_for_goals(
    goals: list[Goal],
    start: date | None = None,
    daily_minutes: int = DEFAULT_DAILY_MINUTES,
) -> list[PlanDraft]:
    """给多个目标各生成一份计划。"""

    return [generate_plan(goal, start=start, daily_minutes=daily_minutes) for goal in goals]


def items_for_day(items: list[PlanItem], day: str | None = None) -> list[PlanItem]:
    """取出某天的待办。"""

    target = day or today_str()

    return [item for item in items if item.day == target]


def progress_of(items: list[PlanItem], goal_id: str = "") -> tuple[int, int]:
    """返回 (已完成条数, 总条数)。

    传 goal_id 就只算这个目标的进度。
    """

    scoped = [item for item in items if not goal_id or item.goal_id == goal_id]
    done = sum(1 for item in scoped if item.done)

    return done, len(scoped)


def next_undone(items: list[PlanItem], day: str | None = None) -> PlanItem | None:
    """找下一条该做的事：优先今天没做完的，其次未来最早的。

    这是每日复盘和"下一步建议"的共同基础。
    """

    today = day or today_str()
    today_pending = [
        item for item in items if item.day == today and not item.done
    ]

    if today_pending:
        return today_pending[0]

    future = sorted(
        (item for item in items if item.day > today and not item.done),
        key=lambda item: item.day,
    )

    return future[0] if future else None


def replace_pending_plan(
    items: list[PlanItem], goal_id: str, draft: list[PlanItem], daily_minutes: int | Callable[[date], int],
    start: date | None = None,
) -> list[PlanItem]:
    """重排未来未完成任务，同时把保留任务计入每日预算。"""

    today = (start or date.today()).isoformat()
    retained = [
        item for item in items
        if item.goal_id != goal_id or item.done or item.day < today
    ]
    used: dict[str, int] = {}
    ids = {item.id for item in retained}
    for item in retained:
        used[item.day] = used.get(item.day, 0) + item.minutes

    fresh: list[PlanItem] = []
    for item in draft:
        if item.id in ids or item.day < today:
            continue
        budget = daily_minutes(date.fromisoformat(item.day)) if callable(daily_minutes) else daily_minutes
        room = budget - used.get(item.day, 0)
        if room < 15:
            continue
        item.minutes = min(item.minutes, room)
        fresh.append(item)
        ids.add(item.id)
        used[item.day] = used.get(item.day, 0) + item.minutes

    items[:] = retained + fresh
    return fresh
