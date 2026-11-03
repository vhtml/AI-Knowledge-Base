"""学习规划师主循环：识别意图 → 读写记忆 → 生成计划或复盘。

一轮处理的顺序（顺序本身是有讲究的）：

1. **路由**：判断用户想干什么（记目标 / 生成计划 / 复盘 / 查询 / 闲聊）。
2. **召回**：把相关的长期记忆和短期记忆取出来，作为这一步的上下文。
   先召回再动手——规划时知道"上次卡在哪"，才不会再排同样的坑。
3. **执行**：生成计划、跑复盘、写记忆。
4. **回写**：把新学到的写进短期记忆，值得长期保留的晋升到长期档案。

每一步都留痕（StepRecord），终端上能直接看到"这轮动了哪块记忆"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .memory import MemoryBundle, MemoryManager
from .models import (
    CATEGORY_GOAL,
    CATEGORY_NOTE,
    CATEGORY_PREFERENCE,
    CATEGORY_SUMMARY,
    Goal,
    PlanItem,
    Review,
    now_iso,
    stable_id,
    today_str,
)
from .planner import (
    DEFAULT_DAILY_MINUTES,
    generate_plan,
    items_for_day,
    next_undone,
    progress_of,
)
from .review import apply_review_findings, run_review

Reporter = Callable[["StepRecord"], None]

# 意图关键字。命中哪个就走哪条路。
INTENT_GOAL = "goal"
INTENT_PLAN = "plan"
INTENT_REVIEW = "review"
INTENT_QUERY = "query"
INTENT_DONE = "done"
INTENT_CHAT = "chat"

GOAL_KEYWORDS = ("目标", "想学", "要学", "学习", "计划学", "打算学")
PLAN_KEYWORDS = ("计划", "排一下", "安排", "规划", "日程")
REVIEW_KEYWORDS = ("复盘", "回顾", "今天怎么样", "总结", "日报")
QUERY_KEYWORDS = ("我的", "查", "看看", "还记得", "我学", "进度")
DONE_KEYWORDS = ("完成", "做完了", "搞定", "done")


@dataclass
class StepRecord:
    """一次决策或执行的留痕。"""

    phase: str
    title: str
    ok: bool = True
    detail: str = ""


@dataclass
class Turn:
    """一轮完整交互的结果。"""

    reply: str
    steps: list[StepRecord] = field(default_factory=list)
    intent: str = INTENT_CHAT
    memories_used: list[str] = field(default_factory=list)
    memories_written: list[str] = field(default_factory=list)


def detect_intent(text: str) -> str:
    """判断用户意图。

    顺序有讲究：先查更具体的关键词，再查宽泛的。
    "帮我生成今天的计划" 同时命中"计划"和"今天"，
    靠"计划"在前才不会误判成复盘。
    """

    lowered = text.lower()

    if any(keyword in lowered for keyword in PLAN_KEYWORDS):
        return INTENT_PLAN

    if any(keyword in lowered for keyword in REVIEW_KEYWORDS):
        return INTENT_REVIEW

    if any(keyword in lowered for keyword in DONE_KEYWORDS):
        return INTENT_DONE

    if any(keyword in lowered for keyword in QUERY_KEYWORDS):
        return INTENT_QUERY

    if any(keyword in lowered for keyword in GOAL_KEYWORDS):
        return INTENT_GOAL

    return INTENT_CHAT


class StudyPlanner:
    """学习规划师。

    持有三样东西：记忆（两级）、目标、计划待办。
    存储由外部注入，本类只管逻辑，方便单独测试。
    """

    def __init__(
        self,
        memory: MemoryManager,
        goals: list[Goal] | None = None,
        items: list[PlanItem] | None = None,
        reviews: list[Review] | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.memory = memory
        self.goals: list[Goal] = list(goals or [])
        self.items: list[PlanItem] = list(items or [])
        self.reviews: list[Review] = list(reviews or [])
        self.reporter = reporter

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------

    def handle(self, user_text: str) -> Turn:
        """处理一轮用户输入。"""

        intent = detect_intent(user_text)
        steps: list[StepRecord] = []
        turn = Turn(reply="", intent=intent)

        # 1. 先召回——后面的每一步都要用到
        bundle = self.memory.recall(user_text)
        self._record(steps, StepRecord("召回", "记忆检索", ok=True, detail=self._describe(bundle)))
        turn.memories_used = [hit.item.content for hit in bundle.hits]

        # 记录写入前后的记忆条数，用来如实报告"这一轮写了什么"
        short_before = len(self.memory.short)
        long_before = len(self.memory.long)

        if intent == INTENT_GOAL:
            reply = self._handle_goal(user_text, bundle, steps)
        elif intent == INTENT_PLAN:
            reply = self._handle_plan(user_text, bundle, steps)
        elif intent == INTENT_REVIEW:
            reply = self._handle_review(bundle, steps)
        elif intent == INTENT_DONE:
            reply = self._handle_done(user_text, steps)
        elif intent == INTENT_QUERY:
            reply = self._handle_query(user_text, bundle, steps)
        else:
            reply = self._handle_chat(user_text, bundle, steps)

        turn.reply = reply
        turn.steps = steps

        # 1.5 时间预算这类陈述是长期约束，要能影响之后的每一次排期
        self._capture_budget(user_text, steps)

        # 2. 回写留痕：这一轮实际动了哪块记忆
        short_added = len(self.memory.short) - short_before
        long_added = len(self.memory.long) - long_before

        turn.memories_written = [
            item.content for item in self.memory.short.items[-short_added:]
        ] if short_added > 0 else []

        detail_parts = []

        if short_added:
            detail_parts.append(f"短期 +{short_added} 条")

        if long_added:
            detail_parts.append(f"长期 +{long_added} 条")

        self._record(
            steps,
            StepRecord(
                "回写",
                "记忆写入",
                ok=True,
                detail="、".join(detail_parts) if detail_parts else "本轮无新记忆写入",
            ),
        )

        return turn

    # ------------------------------------------------------------
    # 各意图的处理
    # ------------------------------------------------------------

    def _handle_goal(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """记录一个新目标。

        目标本身直接进长期记忆并钉住——它不该随当天工作区一起被清掉。
        """

        title = self._clean_title(text)
        goal = Goal(
            id=stable_id("g", title, today_str()),
            title=title,
            deadline=self._guess_deadline(text),
            tags=self._extract_tags(text),
        )
        self.goals.append(goal)

        self.memory.promote(
            f"学习目标：{title}",
            CATEGORY_GOAL,
            "用户明确设定目标",
            importance=5,
            tags=goal.tags,
        )

        self._record(steps, StepRecord("记录目标", goal.title, ok=True, detail=goal.id))
        self._record(steps, StepRecord("长期记忆", "目标已入库", ok=True, detail="目标钉住，不随工作区清空"))

        lines = [f"记下了：{goal.title}"]

        if goal.deadline:
            lines.append(f"截止日期：{goal.deadline}")

        lines.append("要我现在生成一份学习计划吗？")

        return "\n".join(lines)

    def _handle_plan(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """生成计划。

        生成前先看长期记忆：如果之前记录过"只有 30 分钟/天"，
        计划会按这个来，而不是按默认值 60 分钟。
        """

        goal = self._pick_goal(text)

        if goal is None:
            self._record(steps, StepRecord("生成计划", "没有目标", ok=False, detail="请先记录一个目标"))
            return "现在还没有学习目标。先告诉我你想学什么，比如「我想学 Python 数据分析」。"

        minutes, source = self._pick_daily_minutes(bundle)
        draft = generate_plan(goal, daily_minutes=minutes)

        existing = {item.id for item in self.items}
        fresh = [item for item in draft.items if item.id not in existing]
        self.items.extend(fresh)

        # 计划本身进短期记忆：它是"今天要做什么"的依据
        self.memory.remember(
            f"已为「{goal.title}」生成计划，{len(draft.milestones)} 个阶段，"
            f"{len(fresh)} 条日任务，每日 {minutes} 分钟",
            CATEGORY_NOTE,
            importance=4,
            tags=goal.tags,
        )

        detail = f"每日 {minutes} 分钟（{source}）"
        self._record(steps, StepRecord("生成计划", goal.title, ok=True, detail=detail))
        self._record(
            steps,
            StepRecord("排期依据", source, ok=True, detail=f"每日 {minutes} 分钟"),
        )

        return draft.render()

    def _handle_review(self, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """跑每日复盘，并把结论沉淀进长期记忆。"""

        outcome = run_review(self.items, self.reviews, self.goals)
        self.reviews.append(outcome.review)

        applied, written = apply_review_findings(self.memory, outcome)
        self._record(
            steps,
            StepRecord("复盘", outcome.review.day, ok=True, detail=f"完成 {outcome.review.done}/{outcome.review.planned}"),
        )
        self._record(
            steps,
            StepRecord("短期记忆", "复盘摘要", ok=True, detail=f"{len(written)} 条"),
        )
        self._record(
            steps,
            StepRecord("长期记忆", "沉淀结论", ok=True, detail=f"{len(applied)} 条"),
        )

        return outcome.render()

    def _handle_done(self, text: str, steps: list[StepRecord]) -> str:
        """标记一条任务完成。

        完成的是"今天的下一件事"——不要求用户报 ID，
        因为大部分时候人只记得"我刚做完那件事"。
        """

        target = next_undone(self.items)

        if target is None:
            self._record(steps, StepRecord("标记完成", "没有待办", ok=False, detail="所有任务都已完成"))
            return "没有待办任务了，可以去复盘或者生成新计划。"

        target.done = True
        self.memory.remember(f"完成了「{target.title}」", CATEGORY_NOTE, importance=3)

        done, total = progress_of(self.items)
        self._record(steps, StepRecord("标记完成", target.title, ok=True, detail=f"进度 {done}/{total}"))

        return f"完成「{target.title}」。当前进度 {done}/{total}。"

    def _handle_query(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """查询：进度 + 相关记忆。"""

        lines: list[str] = []
        done, total = progress_of(self.items)

        if total:
            lines.append(f"计划进度：{done}/{total} 条已完成")

        if self.goals:
            lines.append("当前目标：")

            for goal in self.goals:
                if goal.status == "active":
                    deadline = f"，截止 {goal.deadline}" if goal.deadline else ""
                    lines.append(f"  - {goal.title}{deadline}")

        if bundle.long:
            lines.append("相关的长期记忆：")

            for hit in bundle.long:
                lines.append(f"  - {hit.item.content}")

        if bundle.short:
            lines.append("相关的短期记忆：")

            for hit in bundle.short:
                lines.append(f"  - {hit.item.content}")

        if not lines:
            lines.append("还没有记录。试试「我想学 Python」或者「生成计划」。")

        self._record(steps, StepRecord("查询", "进度与记忆", ok=True, detail=f"{len(bundle.hits)} 条命中"))

        return "\n".join(lines)

    def _handle_chat(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """闲聊：也能召回记忆，让回答带上上下文。"""

        self.memory.remember(text, CATEGORY_NOTE)
        self._record(steps, StepRecord("闲聊", "已记录", ok=True, detail="写入短期记忆"))

        if bundle.long:
            return "记下了。顺便一提，我记得这些：\n" + bundle.render()

        return "记下了。还有什么想安排的？"

    # ------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------

    def _pick_goal(self, text: str) -> Goal | None:
        """挑目标：文本里提到了就用提到的，否则用最近一个活跃目标。"""

        for goal in self.goals:
            if goal.title in text:
                return goal

        active = [goal for goal in self.goals if goal.status == "active"]

        return active[-1] if active else None

    def _pick_daily_minutes(self, bundle: MemoryBundle) -> tuple[int, str]:
        """从长期档案里读"每天能学多久"。

        这里**不能只靠召回**：召回是按词片相关度匹配的，
        而"生成计划"这四个字跟"每天只能学 30 分钟"没有任何重合，
        相关性检索根本召不回来。

        时间预算是**状态约束**而不是话题——应该直接扫档案取。
        这也是长期记忆区别于"关键词搜索"的地方：
        它保存的是持续生效的前提条件。

        返回 (每日分钟数, 来源说明)。来源要如实区分
        "从记忆里读到的"和"没找到，用的默认值"——
        留痕里说错来源，就等于给自己制造了一个假记忆。
        """

        for hit in bundle.long:
            found = self._parse_duration(hit.item.content)

            if found is not None:
                return found, f"长期记忆：{hit.item.content}"

        # 召回没命中，直接扫档案里的偏好类条目
        for item in self.memory.long.items:
            if item.superseded_by or item.category != CATEGORY_PREFERENCE:
                continue

            found = self._parse_duration(item.content)

            if found is not None:
                return found, f"长期记忆：{item.content}"

        return DEFAULT_DAILY_MINUTES, "默认值（记忆里没有时间预算）"

    def _capture_budget(self, text: str, steps: list[StepRecord]) -> None:
        """识别"我每天只有 30 分钟"这类时间预算陈述，晋升到长期记忆。

        为什么单独拎出来：预算不是话题，是**持续生效的前提**。
        它一旦说出口就应该长期有效，而不是明天就忘了。
        """

        if "分钟" not in text and "小时" not in text:
            return

        minutes = StudyPlanner._parse_duration(text)

        if minutes is None:
            return

        self.memory.promote(
            f"每天可学习时间：{minutes} 分钟",
            CATEGORY_PREFERENCE,
            "用户说明了时间预算",
            importance=5,
        )
        self._record(
            steps,
            StepRecord("长期记忆", "时间预算已记下", ok=True, detail=f"每天 {minutes} 分钟"),
        )

    @staticmethod
    def _parse_duration(content: str) -> int | None:
        """从一句话里解析出分钟数。

        认"30 分钟""2 小时"这类说法；数值超出合理范围就当没解析出来，
        免得一句"学了 300 小时"把排期彻底带偏。

        注意数字和单位之间通常隔着空格（"30 分钟"），
        往回扫的时候必须跳过空白，否则永远扫不到数字。
        """

        for marker in ("分钟", "小时"):
            index = content.find(marker)

            if index < 0:
                continue

            digits = ""
            cursor = index - 1

            # 跳过数字与单位之间的空格
            while cursor >= 0 and content[cursor].isspace():
                cursor -= 1

            while cursor >= 0 and content[cursor].isdigit():
                digits = content[cursor] + digits
                cursor -= 1

            if not digits:
                continue

            value = int(digits)

            if marker == "小时":
                value *= 60

            if 15 <= value <= 480:
                return value

        return None

    @staticmethod
    def _clean_title(text: str) -> str:
        """从一句话里提取目标标题。"""

        cleaned = text.strip()

        for prefix in ("我的目标是", "目标是", "我想学", "我要学", "打算学", "计划学", "我想", "我要"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" ，,。.")

        return cleaned[:60] or text.strip()[:60]

    @staticmethod
    def _guess_deadline(text: str) -> str:
        """猜截止日期：识别"X天后""X月X日"这类说法。"""

        import re

        match = re.search(r"(\d+)\s*天后", text)

        if match:
            from datetime import timedelta

            return (date.today() + timedelta(days=int(match.group(1)))).isoformat()

        match = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)

        if match:
            year = date.today().year

            try:
                candidate = date(year, int(match.group(1)), int(match.group(2)))
            except ValueError:
                return ""

            if candidate < date.today():
                candidate = date(year + 1, int(match.group(1)), int(match.group(2)))

            return candidate.isoformat()

        return ""

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """从标题里抽出领域标签，方便后续召回。"""

        from .planner import detect_domain

        domain = detect_domain(text)

        return [domain] if domain != "通用" else []

    @staticmethod
    def _describe(bundle: MemoryBundle) -> str:
        """把召回结果描述成一行，用于留痕。"""

        parts = []

        if bundle.long:
            parts.append(f"长期 {len(bundle.long)} 条")

        if bundle.short:
            parts.append(f"短期 {len(bundle.short)} 条")

        return "、".join(parts) if parts else "无命中"

    def _record(self, steps: list[StepRecord], step: StepRecord) -> None:
        steps.append(step)

        if self.reporter is not None:
            self.reporter(step)

    # ------------------------------------------------------------
    # 跨天与持久化辅助
    # ------------------------------------------------------------

    def today_items(self) -> list[PlanItem]:
        """今天的待办。"""

        return items_for_day(self.items, today_str())

    def rollover(self, day: str | None = None) -> list[str]:
        """跨天：短期工作区清空，值得沉淀的进长期档案。"""

        from .review import rollover_day

        return rollover_day(self.memory, day)
