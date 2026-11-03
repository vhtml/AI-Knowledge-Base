"""学习规划师主循环：识别意图 → 读写记忆 → 生成计划或复盘。

一轮处理的顺序（顺序本身是有讲究的）：

1. **路由**：判断用户想干什么（记目标 / 生成计划 / 复盘 / 查询 / 闲聊）。
2. **召回**：把相关的长期记忆和短期记忆取出来，作为这一步的上下文。
   先召回再动手——规划时知道"上次卡在哪"，才不会再排同样的坑。
3. **执行**：生成计划、跑复盘、写记忆。
4. **回写**：把新学到的写进短期记忆，值得长期保留的晋升到长期档案。

每一步都留痕（StepRecord），终端上能直接看到"这轮动了哪块记忆"。

两种运行模式：

- **规则模式**（没有模型客户端时，命令行入口走这条）：意图靠关键字判定，
  计划靠模板拆解，回复是写死的字符串。离线可跑，行为可预测。
- **模型模式**（注入了模型客户端时，Web 工作台走这条）：模型读状态和记忆，
  决定这轮做什么并给出要落库的内容；代码负责把决策落到状态上，
  再让模型用自然语言把结果讲出来。

两种模式共用同一套状态与记忆内核，所以记忆的行为约定完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from .llm import (
    ACT_ALL,
    ACT_ANSWER,
    ACT_DONE,
    ACT_GOAL,
    ACT_PLAN,
    ACT_QUERY,
    ACT_REVIEW,
    DECIDE_SYSTEM_PROMPT,
    REPLY_SYSTEM_PROMPT,
    LLMClient,
)
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
DeltaHandler = Callable[[str], None]

# 模型排的计划最多铺到多少天之后——防止模型给出离谱的远期日期
MAX_PLAN_DAYS = 60

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
        llm: LLMClient | None = None,
    ) -> None:
        self.memory = memory
        self.goals: list[Goal] = list(goals or [])
        self.items: list[PlanItem] = list(items or [])
        self.reviews: list[Review] = list(reviews or [])
        self.reporter = reporter
        # 有模型客户端就走模型模式，没有就走规则模式
        self.llm = llm
        # 本次请求临时生效的 reporter，见 _use_reporter
        self._scoped_reporter: Reporter | None = None

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------

    def handle(
        self,
        user_text: str,
        reporter: Reporter | None = None,
        on_delta: DeltaHandler | None = None,
    ) -> Turn:
        """处理一轮用户输入。

        参数：
            user_text 用户这轮说的话
            reporter  可选的每请求回调，收到每一条留痕。
                      传了就覆盖实例级 reporter——Web 端做 SSE 时
                      每个请求需要独立的回调，不能共用实例上的那个。
            on_delta  可选的逐段回调，收到模型回复的文本增量。
                      模型模式下用来做流式推送。
        """

        intent = detect_intent(user_text)
        steps: list[StepRecord] = []
        turn = Turn(reply="", intent=intent)

        # 1. 先召回——后面的每一步都要用到
        bundle = self.memory.recall(user_text)

        # 记录写入前后的记忆条数，用来如实报告"这一轮写了什么"
        short_before = len(self.memory.short)
        long_before = len(self.memory.long)

        # 本次请求的 reporter 临时接管留痕上报
        previous_reporter = self._use_reporter(reporter)

        try:
            if self.llm is not None:
                decision = self._model_decide(user_text, bundle, steps)

                if decision is None:
                    # 模型调不通，退回规则模式：服务不能因为模型挂了就不可用
                    reply = self._run_intent(intent, user_text, bundle, steps, turn)
                    self._capture_budget(user_text, steps)
                else:
                    facts = self._apply_model_decision(
                        decision, user_text, bundle, steps, turn
                    )
                    reply = self._model_reply(
                        user_text, bundle, facts, steps, on_delta
                    )
            else:
                reply = self._run_intent(intent, user_text, bundle, steps, turn)

                # 时间预算这类陈述是长期约束，要能影响之后的每一次排期。
                # 必须在 reporter 作用域内执行——放在外面的话，
                # "时间预算已记下"这条留痕不会推给正在等待的客户端。
                self._capture_budget(user_text, steps)
        finally:
            self._scoped_reporter = previous_reporter

        turn.reply = reply
        turn.steps = steps

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

    def _run_intent(
        self,
        intent: str,
        user_text: str,
        bundle: MemoryBundle,
        steps: list[StepRecord],
        turn: Turn,
    ) -> str:
        """按意图分发，返回本轮的回答文本。

        召回和计数在 handle() 里做，这里只负责分发——
        拆出来是为了让 handle() 的 try/finally 只包住业务执行。
        """

        self._record(
            steps, StepRecord("召回", "记忆检索", ok=True, detail=self._describe(bundle))
        )
        turn.memories_used = [hit.item.content for hit in bundle.hits]

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

        return reply

    # ------------------------------------------------------------
    # 各意图的处理
    # ------------------------------------------------------------

    def _handle_goal(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """记录一个新目标。

        目标本身直接进长期记忆并钉住——它不该随当天工作区一起被清掉。
        """

        title = self._clean_title(text)
        deadline = self._guess_deadline(text)
        tags = self._extract_tags(text)
        goal = self._record_goal(title, deadline, tags, steps)

        lines = [f"记下了：{goal.title}"]

        if goal.deadline:
            lines.append(f"截止日期：{goal.deadline}")

        lines.append("要我现在生成一份学习计划吗？")

        return "\n".join(lines)

    def _record_goal(
        self,
        title: str,
        deadline: str,
        tags: list[str],
        steps: list[StepRecord],
    ) -> Goal:
        """写入或更新一个目标，并把它钉进长期记忆。

        规则模式和模型模式共用这一段：同一个目标重复说
        （标题和日期相同 → id 相同）应该**更新**而不是追加，
        否则列表里会出现两条一模一样的目标。
        """

        goal_id = stable_id("g", title, today_str())

        existing = next((item for item in self.goals if item.id == goal_id), None)

        if existing is not None:
            # 更新已有目标：补上这次给出的信息，没给的不覆盖
            existing.title = title

            if deadline:
                existing.deadline = deadline

            if tags:
                existing.tags = tags

            goal = existing
            action = "目标已更新"

        else:
            goal = Goal(id=goal_id, title=title, deadline=deadline, tags=tags)
            self.goals.append(goal)
            action = "目标已记录"

        self.memory.promote(
            f"学习目标：{title}",
            CATEGORY_GOAL,
            "用户明确设定目标",
            importance=5,
            tags=goal.tags,
        )

        self._record(steps, StepRecord(action, goal.title, ok=True, detail=goal.id))
        self._record(steps, StepRecord("长期记忆", "目标已入库", ok=True, detail="目标钉住，不随工作区清空"))

        return goal

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

        outcome = self._run_review(steps)

        return outcome.render()

    def _run_review(self, steps: list[StepRecord]):
        """复盘的可执行部分：对账、算完成率、沉淀结论。

        模型模式也走这一段：完成率是**数据**，不该让模型自己编；
        模型只负责给出定性的结论（亮点、卡点、下一步）。
        """

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

        return outcome

    def _handle_done(self, text: str, steps: list[StepRecord]) -> str:
        """标记一条任务完成。

        完成的是"今天的下一件事"——不要求用户报 ID，
        因为大部分时候人只记得"我刚做完那件事"。
        """

        target = self._mark_next_done(steps)

        if target is None:
            return "没有待办任务了，可以去复盘或者生成新计划。"

        done, total = progress_of(self.items)

        return f"完成「{target.title}」。当前进度 {done}/{total}。"

    def _mark_next_done(self, steps: list[StepRecord]) -> PlanItem | None:
        """把今天的下一件事标记完成，返回被标记的那条。"""

        target = next_undone(self.items)

        if target is None:
            self._record(steps, StepRecord("标记完成", "没有待办", ok=False, detail="所有任务都已完成"))
            return None

        target.done = True
        self.memory.remember(f"完成了「{target.title}」", CATEGORY_NOTE, importance=3)

        done, total = progress_of(self.items)
        self._record(steps, StepRecord("标记完成", target.title, ok=True, detail=f"进度 {done}/{total}"))

        return target

    def _handle_query(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """查询：进度 + 相关记忆。"""

        self._record(steps, StepRecord("查询", "进度与记忆", ok=True, detail=f"{len(bundle.hits)} 条命中"))

        return self._render_query(bundle)

    def _render_query(self, bundle: MemoryBundle) -> str:
        """把当前进度和召回的记忆拼成可读文本。"""

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

        return "\n".join(lines)

    def _handle_chat(self, text: str, bundle: MemoryBundle, steps: list[StepRecord]) -> str:
        """闲聊：也能召回记忆，让回答带上上下文。"""

        self.memory.remember(text, CATEGORY_NOTE)
        self._record(steps, StepRecord("闲聊", "已记录", ok=True, detail="写入短期记忆"))

        if bundle.long:
            return "记下了。顺便一提，我记得这些：\n" + bundle.render()

        return "记下了。还有什么想安排的？"

    # ------------------------------------------------------------
    # 模型模式
    #
    # 两步走：先让模型给出**结构化决策**，代码把决策落到状态上，
    # 再让模型用自然语言把结果讲出来。
    #
    # 分两步而不是一步到位，是为了让状态永远是真的：
    # 模型只负责"想"（决定做什么、说什么），"做"（改状态、落记忆）
    # 仍然由代码执行。留痕来自代码，也就一定是真的。
    # ------------------------------------------------------------

    def _model_decide(
        self, user_text: str, bundle: MemoryBundle, steps: list[StepRecord]
    ) -> dict[str, Any] | None:
        """问模型这轮要做什么。

        返回 None 表示模型调不通，调用方据此退回规则模式——
        服务不能因为模型挂了就整个不可用。
        """

        prompt = (
            f"{self._state_snapshot()}\n\n"
            f"【本轮召回】\n{bundle.render()}\n\n"
            f"【用户这轮说的话】\n{user_text}\n\n"
            "请给出这一轮的决策 JSON。"
        )

        messages = [
            {"role": "system", "content": DECIDE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            decision = self.llm.complete_json(messages)
        except Exception as exc:
            self._record(
                steps,
                StepRecord("模型", "决策调用失败", ok=False, detail=type(exc).__name__),
            )
            return None

        if not decision:
            self._record(steps, StepRecord("模型", "决策无效", ok=False, detail="模型没给出可解析的 JSON"))
            return None

        action = str(decision.get("action") or "").strip()
        label = action if action in ACT_ALL else "无法识别的动作"

        self._record(steps, StepRecord("决策", "模型决策", ok=True, detail=label))

        return decision

    def _apply_model_decision(
        self,
        decision: dict[str, Any],
        user_text: str,
        bundle: MemoryBundle,
        steps: list[StepRecord],
        turn: Turn,
    ) -> str:
        """把模型的决策落到状态上，返回一段「本轮已发生」的描述。"""

        action = str(decision.get("action") or "").strip()

        if action not in ACT_ALL:
            action = ACT_ANSWER

        turn.intent = action
        facts: list[str] = []

        if action == ACT_GOAL:
            facts.append(self._apply_model_goal(decision.get("goal"), user_text, steps))
        elif action == ACT_PLAN:
            facts.append(
                self._apply_model_plan(
                    decision.get("plan"), decision.get("goal"), user_text, bundle, steps
                )
            )
        elif action == ACT_DONE:
            facts.append(self._apply_model_done(decision.get("done"), steps))
        elif action == ACT_REVIEW:
            facts.append(self._apply_model_review(decision.get("review"), steps))
        elif action == ACT_QUERY:
            self._record(
                steps,
                StepRecord("查询", "进度与记忆", ok=True, detail=f"{len(bundle.hits)} 条命中"),
            )
            facts.append(self._render_query(bundle))
        else:
            facts.append("这轮只是对话，没有改动任何状态。")

        facts.extend(self._apply_model_memory(decision.get("memory"), steps))

        return "\n".join(item for item in facts if item)

    def _apply_model_goal(
        self, goal_raw: Any, user_text: str, steps: list[StepRecord]
    ) -> str:
        """落目标：模型给标题/截止/标签，代码负责去重与钉进长期记忆。"""

        raw = goal_raw if isinstance(goal_raw, dict) else {}
        title = str(raw.get("title") or "").strip() or self._clean_title(user_text)
        deadline = str(raw.get("deadline") or "").strip()
        tags = [str(tag).strip() for tag in (raw.get("tags") or []) if str(tag).strip()]

        # 模型有时会把"我每天只有 30 分钟"当成目标记下来。
        # 那不是目标，是**持续生效的约束**——记错地方的话，
        # 它就变成一条要在排期时被"扫档案"才能捡起来的普通目标。
        minutes = StudyPlanner._parse_duration(title)

        if minutes is not None:
            self._record(
                steps,
                StepRecord("目标已记录", "改记为时间预算", ok=True, detail=f"每天 {minutes} 分钟"),
            )
            self.memory.promote(
                f"每天可学习时间：{minutes} 分钟",
                CATEGORY_PREFERENCE,
                "模型误记为目标，按时间预算归位",
                importance=5,
            )
            return f"记下了你的时间预算：每天 {minutes} 分钟。"

        goal = self._record_goal(title, deadline, tags, steps)

        return f"已记录目标「{goal.title}」。"

    def _apply_model_plan(
        self,
        plan_raw: Any,
        goal_raw: Any,
        user_text: str,
        bundle: MemoryBundle,
        steps: list[StepRecord],
    ) -> str:
        """落计划：模型给任务清单，代码按时间预算兜底裁剪。"""

        goal = self._pick_goal(user_text)

        raw_goal = goal_raw if isinstance(goal_raw, dict) else {}
        wanted = str(raw_goal.get("title") or "").strip()

        if wanted:
            matched = next(
                (item for item in self.goals if item.title == wanted or wanted in item.title),
                None,
            )
            if matched is not None:
                goal = matched

        if goal is None:
            self._record(steps, StepRecord("生成计划", "没有目标", ok=False, detail="请先记录一个目标"))
            return "现在还没有学习目标，先告诉我想学什么。"

        minutes, source = self._pick_daily_minutes(bundle)
        entries = plan_raw if isinstance(plan_raw, list) else []
        draft, dropped = self._build_model_items(entries, goal, minutes)

        if not draft:
            # 模型给的计划一条都用不上（日期不对、标题为空……）。
            # 这时候退回模板重排一份——空计划对用户没有任何帮助，
            # 留痕里会写明这次是模板兜底，不会假装是模型排的。
            self._record(
                steps,
                StepRecord("生成计划", "模型计划不可用", ok=False, detail="按模板重排"),
            )
            fallback = generate_plan(goal, daily_minutes=minutes)
            draft = fallback.items

        existing = {item.id for item in self.items}
        fresh = [item for item in draft if item.id not in existing]
        self.items.extend(fresh)

        # 计划本身进短期记忆：它是"今天要做什么"的依据
        self.memory.remember(
            f"已为「{goal.title}」生成计划，{len(fresh)} 条日任务，每日 {minutes} 分钟",
            CATEGORY_NOTE,
            importance=4,
            tags=goal.tags,
        )

        self._record(
            steps,
            StepRecord("生成计划", goal.title, ok=True, detail=f"每日 {minutes} 分钟（{source}）"),
        )
        self._record(
            steps,
            StepRecord("排期依据", source, ok=True, detail=f"每日 {minutes} 分钟"),
        )

        if dropped:
            self._record(
                steps,
                StepRecord("排期校验", "超出预算已裁剪", ok=True, detail=f"{dropped} 条"),
            )

        return f"已为「{goal.title}」排出 {len(fresh)} 条日任务，每日 {minutes} 分钟（{source}）。"

    def _build_model_items(
        self, entries: list[Any], goal: Goal, daily_minutes: int
    ) -> tuple[list[PlanItem], int]:
        """把模型给的清单落成待办，并把超出预算的部分裁掉。

        模型不知道也不该管"每天最多多少分钟"——那是长期记忆里的约束，
        由代码兜底裁剪，不然模型一发挥就能把排期撑爆。
        """

        items: list[PlanItem] = []
        dropped = 0
        used: dict[str, int] = {}
        today = date.today()
        last_day = today + timedelta(days=MAX_PLAN_DAYS)

        for entry in entries:
            if not isinstance(entry, dict):
                dropped += 1
                continue

            title = str(entry.get("title") or "").strip()

            if not title:
                dropped += 1
                continue

            try:
                day = self._parse_plan_day(entry.get("date"))
            except ValueError:
                dropped += 1
                continue

            if day < today or day > last_day:
                dropped += 1
                continue

            spent = used.get(day.isoformat(), 0)
            room = daily_minutes - spent

            if room < 15:
                dropped += 1
                continue

            try:
                minutes = int(entry.get("minutes") or 0)
            except (TypeError, ValueError):
                minutes = 0

            minutes = max(15, min(minutes or daily_minutes, room))
            used[day.isoformat()] = spent + minutes

            items.append(
                PlanItem(
                    id=stable_id("p", goal.id, day.isoformat(), title),
                    goal_id=goal.id,
                    title=title,
                    day=day.isoformat(),
                    minutes=minutes,
                    milestone=bool(entry.get("milestone")),
                )
            )

        return items, dropped

    @staticmethod
    def _parse_plan_day(raw: Any) -> date:
        """解析模型给的日期。

        模型大概率会照着 YYYY-MM-DD 写，但偶尔会写成 "2026/11/05"
        甚至 "11月5日"。这里都认——日期解析失败就等于把整份计划丢掉，
        代价太大，不值得为格式较真。
        """

        text = str(raw or "").strip().replace("/", "-").replace(".", "-")

        if not text:
            raise ValueError("空日期")

        try:
            return date.fromisoformat(text)
        except ValueError:
            pass

        # "11-05" 这种缺年份的，按今年算
        parts = text.split("-")

        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            today = date.today()
            candidate = date(today.year, int(parts[0]), int(parts[1]))

            if candidate < today:
                candidate = date(today.year + 1, int(parts[0]), int(parts[1]))

            return candidate

        raise ValueError(f"无法解析日期：{raw}")

    def _apply_model_done(self, done_raw: Any, steps: list[StepRecord]) -> str:
        """标记完成：模型指名的就按名字找，没指名就按"今天的下一件事"。"""

        raw = done_raw if isinstance(done_raw, dict) else {}
        wanted = str(raw.get("title") or "").strip()

        if wanted:
            matched = next(
                (item for item in self.items if not item.done and wanted in item.title), None
            )

            if matched is not None:
                matched.done = True
                self.memory.remember(f"完成了「{matched.title}」", CATEGORY_NOTE, importance=3)

                done, total = progress_of(self.items)
                self._record(
                    steps,
                    StepRecord("标记完成", matched.title, ok=True, detail=f"进度 {done}/{total}"),
                )

                return f"已标记「{matched.title}」完成。"

        target = self._mark_next_done(steps)

        if target is None:
            return "没有待办任务了。"

        return f"已标记「{target.title}」完成。"

    def _apply_model_review(self, review_raw: Any, steps: list[StepRecord]) -> str:
        """复盘：完成率由代码算，定性结论由模型给。"""

        outcome = self._run_review(steps)
        raw = review_raw if isinstance(review_raw, dict) else {}

        highlights = [str(x).strip() for x in (raw.get("highlights") or []) if str(x).strip()]
        blockers = [str(x).strip() for x in (raw.get("blockers") or []) if str(x).strip()]
        next_step = str(raw.get("next_step") or "").strip()

        if highlights:
            outcome.review.highlights = highlights

        if blockers:
            outcome.review.blockers = blockers

        if next_step:
            outcome.review.next_step = next_step

        self._record(
            steps,
            StepRecord(
                "复盘结论",
                "模型归纳",
                ok=True,
                detail=f"{len(highlights)} 条亮点 / {len(blockers)} 条卡点",
            ),
        )

        return f"复盘完成：{outcome.review.done}/{outcome.review.planned} 条完成。"

    def _apply_model_memory(self, memory_raw: Any, steps: list[StepRecord]) -> list[str]:
        """把模型判定"值得记住"的内容写进记忆。

        长期记忆仍然只能经 `promote()` 写入——模型只负责判断值不值得记，
        写入通道不变。这样"什么才算值得记住"这条约束不会被绕过。
        """

        entries = memory_raw if isinstance(memory_raw, list) else []
        written: list[str] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            text = str(entry.get("text") or "").strip()

            if not text:
                continue

            category = str(entry.get("category") or CATEGORY_NOTE).strip() or CATEGORY_NOTE

            try:
                importance = int(entry.get("importance") or 3)
            except (TypeError, ValueError):
                importance = 3

            importance = max(1, min(5, importance))
            promote = bool(entry.get("promote"))

            if promote and category == CATEGORY_PREFERENCE:
                # 预算这类偏好必须归一化成统一说法：模型换个措辞就会落到
                # 另一个槽位，档案里会堆出两条互相矛盾的预算。
                minutes = StudyPlanner._parse_duration(text)

                if minutes is not None:
                    text = f"每天可学习时间：{minutes} 分钟"

            if promote:
                self.memory.promote(text, category, "模型判定值得长期保留", importance=importance)
                written.append(f"长期：{text}")
            else:
                self.memory.remember(text, category, importance=importance)
                written.append(f"短期：{text}")

        if written:
            self._record(
                steps,
                StepRecord("记忆判定", "模型给出的记忆条目", ok=True, detail=f"{len(written)} 条"),
            )

        return written

    def _model_reply(
        self,
        user_text: str,
        bundle: MemoryBundle,
        facts: str,
        steps: list[StepRecord],
        on_delta: DeltaHandler | None,
    ) -> str:
        """让模型把这一轮讲给用户听，逐段吐出来。"""

        prompt = (
            f"{self._state_snapshot()}\n\n"
            f"【本轮已发生】\n{facts}\n\n"
            f"【本轮召回】\n{bundle.render()}\n\n"
            f"【用户这轮说的话】\n{user_text}"
        )

        messages = [
            {"role": "system", "content": REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        chunks: list[str] = []

        try:
            for delta in self.llm.stream(messages):
                if not delta.text:
                    continue

                chunks.append(delta.text)

                if on_delta is not None:
                    on_delta(delta.text)
        except Exception as exc:
            self._record(
                steps,
                StepRecord("模型", "回复生成中断", ok=False, detail=type(exc).__name__),
            )

            if not chunks:
                # 一段都没吐出来：把事实直接交给用户，好过一片空白
                return facts

            return "".join(chunks)

        text = "".join(chunks).strip()

        return text or facts

    def _state_snapshot(self) -> str:
        """把当前状态拍平成一段文本，喂给模型。

        长期档案是**全量**给出去的：预算这类约束必须让模型看得见，
        否则它只能靠相关性召回——而召回恰恰召不回时间预算。
        """

        lines = [f"【今天】{date.today().isoformat()}（排计划从这个日期往后排）", "", "【当前目标】"]

        active = [goal for goal in self.goals if goal.status == "active"]

        if active:
            for goal in active:
                deadline = f"（截止 {goal.deadline}）" if goal.deadline else ""
                lines.append(f"- {goal.title}{deadline}")
        else:
            lines.append("（无）")

        lines.extend(["", "【今日待办】"])
        today = self.today_items()

        if today:
            for item in today:
                mark = "✓" if item.done else "○"
                lines.append(f"- {mark} {item.title}（{item.minutes} 分钟）")
        else:
            lines.append("（无）")

        lines.extend(["", "【长期档案】"])
        archived = [item for item in self.memory.long.items if not item.superseded_by]

        if archived:
            for item in archived:
                lines.append(f"- {item.content}")
        else:
            lines.append("（无）")

        lines.extend(["", "【短期工作区】"])

        if self.memory.short.items:
            for item in self.memory.short.items:
                lines.append(f"- {item.content}")
        else:
            lines.append("（无）")

        return "\n".join(lines)

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

        reporter = self._scoped_reporter or self.reporter

        if reporter is not None:
            reporter(step)

    def _use_reporter(self, reporter: Reporter | None) -> "Reporter | None":
        """临时把 reporter 换成本次请求的，返回原值供还原。

        用作用域切换而不是给每个处理函数加参数：处理函数有五个，
        加参数会一路传染到签名里，而它们内部并不关心谁在听。
        """

        previous = self._scoped_reporter
        self._scoped_reporter = reporter

        return previous

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
