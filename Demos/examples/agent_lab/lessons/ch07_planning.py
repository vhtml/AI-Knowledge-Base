"""第 7 章实验：任务分解——规划模式与自适应分解。

这个模块解决什么问题：第 6 章解决"记住什么"，第 7 章解决"先做什么后做什么"。
任务一旦超过一步，"下一步做什么"就有两个来源：模型看着上下文随手决定
（第 2 章的 ReAct 循环），或者先规划出一份计划再照着做。

两种做法都要付钱，而且两种失败都很难靠"看着办"躲过去：

1. **计划什么时候划算**——规划本身就是一次模型调用，也是钱和延迟；
   简单任务直接干比先规划更快。
2. **计划错了怎么办**——计划引用了不存在的工具、步骤之间的依赖断裂、
   计划漏掉了关键步骤，这些都应该在执行前被拦住，而不是跑到一半才炸。

所以本章把三种分解策略（一次性规划 / 计划-执行-重规划 / 自适应分解）
做成同一个执行器上的三种模式，用**调用计数**把差别钉死：
模型被调了几次、哪几步被重做过、计划最后变成了什么样。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..labkit import (
    ChatModel,
    ModelReply,
    Tool,
    ToolCall,
    approx_tokens,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    system,
    tool,
)

# ---------------------------------------------------------------- 常量

TASK = "查清订单 A-1001 的状态和库存，并生成一份汇总报告"

PLANNER_PROMPT = (
    "你是任务规划器。只输出一个 JSON 对象，不要输出解释性文字。"
    "每一步必须写清 tool（只能用给定工具）、done_when（可验证的完成判据）、"
    "depends_on（依赖的步骤 id，没有就写空数组）。"
)

EXECUTOR_PROMPT = (
    "你是执行器。一次只做一件事：按当前步骤调用工具，"
    "拿到结果就返回，不要提前做后面的步骤。"
)

MAX_GOAL_CHARS = 20
"""步骤目标的长度上限：超了通常说明这一步裹了太多事。"""

VAGUE_MARKERS = ("整个", "全部", "所有", "完整", "搞定", "处理完")
"""空步骤的典型词：出现它们说明这一步没有可验证的边界。"""


# ---------------------------------------------------------------- 工具


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return {"A-1001": "已发货", "A-1002": "待付款", "A-1003": "已签收"}.get(order_id, "未找到")


@tool
def 查询库存(sku: str) -> str:
    """按 SKU 查询库存余量。

    Args:
        sku: 商品 SKU，形如 SKU-1001
    """
    if sku == "SKU-404":
        # 这个 SKU 在库存系统里不存在——第 7 章用它制造一次"明确的失败点"
        raise ValueError(f"SKU {sku} 在库存系统中不存在")
    return {"SKU-1001": "有货 42 件", "SKU-1002": "有货 7 件"}.get(sku, "未找到")


@tool
def 生成报告(summary: str) -> str:
    """根据要点生成一段汇总报告。

    Args:
        summary: 报告要包含的要点
    """
    return f"报告：{summary}"


TOOLS: tuple[Tool, ...] = (查询订单, 查询库存, 生成报告)
TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in TOOLS)


# ---------------------------------------------------------------- 数据模型


@dataclass(frozen=True)
class PlanStep:
    """计划里的一步。

    五个字段各自对应"计划质量"的一条标准：
    ``tool``/``args`` 决定它能不能执行，``done_when`` 决定它有没有判据，
    ``goal`` 决定粒度合不合适，``depends_on`` 决定它依赖谁的产出。
    """

    id: str
    goal: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    done_when: str = ""
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """一份计划。

    ``version`` 是计划的血缘：重规划产生的���一份计划，**步骤 id 必须沿用**
    老计划的语义，否则执行器认不出"这一步已经做过了"，就会白做一遍。
    """

    steps: tuple[PlanStep, ...]
    strategy: str = "plan_once"
    version: int = 1
    note: str = ""


@dataclass(frozen=True)
class StepOutcome:
    """一步的执行结果。"""

    step: PlanStep
    ok: bool
    output: str = ""
    error: str = ""


@dataclass(frozen=True)
class StepExecution:
    """一次执行动作的记录。

    存在的唯一理由：证明"哪些步骤被重做过"。
    计划里出现同一个 id 两次，就是重复劳动的铁证。
    """

    step_id: str
    status: Literal["ok", "failed", "skipped"]
    detail: str = ""


@dataclass
class PlanRunResult:
    """一次计划执行的结果。"""

    task: str
    strategy: str
    plan_versions: list[Plan] = field(default_factory=list)
    completed: list[StepOutcome] = field(default_factory=list)
    executions: list[StepExecution] = field(default_factory=list)
    failed: StepOutcome | None = None
    stop_reason: str = "completed"
    error: str = ""
    planner_calls: int = 0
    executor_calls: int = 0
    prompt_tokens: int = 0
    tool_invocations: dict[str, int] = field(default_factory=dict)

    @property
    def plan(self) -> Plan:
        """当前生效的计划——也就是最后一次规划产生的版本。"""
        return self.plan_versions[-1]


@dataclass(frozen=True)
class ExecutionSpec:
    """执行策略的配置。

    ``mode`` 决定第一步失败之后怎么办：
    ``plan_once`` 直接停，``replan`` 整份换掉，
    ``adaptive`` 只换掉剩下的部分。
    """

    mode: Literal["plan_once", "replan", "adaptive"] = "plan_once"
    max_steps: int = 5
    max_replans: int = 1
    skip_step_ids: frozenset[str] = frozenset()
    """被跳过的步骤——人工跳过，或上一个计划版本里被砍掉的步骤。"""


class PlanRejected(Exception):
    """计划没通过校验，不能执行。

    为什么要有这道闸：计划和执行之间必须隔一层。
    没有它，模型写出一个不存在的工具名，就会一路跑到执行阶段才炸给用户看。
    """

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("；".join(self.reasons))


# ---------------------------------------------------------------- 计划解析与校验


def _strip_code_fence(text: str) -> str:
    """剥掉 Markdown 代码围栏，取出里面的 JSON。

    真模型经常把 JSON 包在 ```json 里，甚至前后加一句"好的，这是计划："。
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def parse_plan(text: str) -> Plan:
    """把模型返回的文本解析成一份计划。

    解析失败是计划失败的第一种形态（第 3 章讲的"模型输出不可信"），
    所以这里给的是一句人话，而不是裸的 ``JSONDecodeError``。
    """
    payload = _strip_code_fence(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanRejected([f"模型返回的计划不是合法 JSON：{exc}"]) from exc
    if not isinstance(data, dict):
        raise PlanRejected(["模型返回的计划不是一个 JSON 对象"])

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise PlanRejected(["模型返回的计划里缺少 steps 数组"])

    steps: list[PlanStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            raise PlanRejected([f"计划第 {index} 步不是一个对象"])
        depends = item.get("depends_on") or []
        steps.append(
            PlanStep(
                id=str(item.get("id") or f"s{index}"),
                goal=str(item.get("goal") or ""),
                tool=str(item.get("tool") or ""),
                args=item.get("args") if isinstance(item.get("args"), dict) else {},
                done_when=str(item.get("done_when") or ""),
                depends_on=tuple(str(x) for x in depends),
            )
        )
    return Plan(steps=tuple(steps), strategy=str(data.get("strategy") or "plan_once"))


def validate_plan(plan: Plan, known_tools: set[str]) -> list[str]:
    """检查一份计划能不能执行，返回问题列表（空列表 = 合格）。

    三条标准对应三个真实故障面：

    - **可执行**：每一步都要有对应的工具，否则执行时才会发现模型幻觉了工具名；
    - **无歧义**：每一步都要有完成判据，否则你不知道它做完了没有，
      下游步骤只能靠猜；
    - **粒度合适**：目标太长、或者含"整个/所有"这类空词，
      说明这一步裹了太多事，做完也验证不了。

    另外还要挡住 id 重复和依赖不存在的步骤——它们会让"这一步已经做过了"
    这个判断失效。
    """
    problems: list[str] = []
    if not plan.steps:
        return ["计划里没有任何步骤"]

    known_ids = {step.id for step in plan.steps}
    for index, step in enumerate(plan.steps, start=1):
        if step.tool not in known_tools:
            problems.append(
                f"第 {index} 步（{step.id}）引用了不存在的工具「{step.tool}」"
            )
        if not step.done_when.strip():
            problems.append(f"第 {index} 步（{step.id}）缺少完成判据")
        if len(step.goal) > MAX_GOAL_CHARS:
            problems.append(
                f"第 {index} 步（{step.id}）粒度过粗：目标 {len(step.goal)} 字，"
                f"超过 {MAX_GOAL_CHARS} 字上限"
            )
        elif any(marker in step.goal for marker in VAGUE_MARKERS):
            problems.append(
                f"第 {index} 步（{step.id}）粒度过粗：目标里出现空词"
                f"（{step.goal}）"
            )
        for dep in step.depends_on:
            if dep == step.id:
                problems.append(f"第 {index} 步（{step.id}）依赖了自己")
            elif dep not in known_ids:
                problems.append(
                    f"第 {index} 步（{step.id}）依赖了计划里不存在的步骤 {dep}"
                )
    if len(known_ids) != len(plan.steps):
        problems.append("计划里出现了重复的步骤 id——执行器会认不出哪些步骤已经做过")
    return problems


def detect_broken_dependencies(
    step: PlanStep,
    plan: Plan,
    completed_ids: set[str],
    produced: dict[str, str],
) -> list[str]:
    """检查这一步的依赖是不是还成立。

    依赖成立需要**同时**满足两个条件：被依赖的步骤已经完成，
    并且真的留下了产出。少任何一条，这一步就是悬空的——
    比如上一步被跳过、或者失败了。

    为什么只查当前这一步：计划里靠后的步骤依赖还没执行的步骤是正常的
    （顺序执行时它自然会满足）；只有"轮到你时依赖仍然没满足"才是断裂。
    """
    problems: list[str] = []
    known_ids = {item.id for item in plan.steps}
    for dep in step.depends_on:
        if dep not in known_ids:
            problems.append(f"步骤 {step.id} 依赖 {dep}，但计划里没有这一步")
        elif dep not in completed_ids:
            problems.append(
                f"步骤 {step.id} 依赖 {dep} 的产出，但 {dep} 尚未完成"
                "（被跳过或执行失败），依赖断裂"
            )
        elif dep not in produced:
            problems.append(f"步骤 {step.id} 依赖 {dep} 的产出，但产出缺失")
    return problems


def render_plan_context(plan: Plan, done_ids: set[str]) -> str:
    """把"还剩什么"渲染成给模型看的文本。"""
    return "\n".join(
        f"{'✓' if step.id in done_ids else '·'} {step.id} {step.goal}（工具：{step.tool}）"
        for step in plan.steps
    )


# ---------------------------------------------------------------- 规划器与执行器


@dataclass
class Planner:
    """把任务变成一份可校验的计划。

    为什么规划要单独一个角色：规划的输出必须能被机器检查
    （工具有没有、判据有没有、粒度合不合适），不能是"一段话"。
    所以它只输出 JSON。
    """

    model: ChatModel
    tools: tuple[Tool, ...] = ()
    calls: int = 0
    """规划调用次数——用来量"为了这份计划花了多少钱"。"""

    def plan(self, task: str, context: str = "", remaining_only: bool = False) -> Plan:
        """向模型要一份计划。

        ``remaining_only=True`` 时只要剩下的步骤——自适应分解用这个口径。
        为什么不重写已完成的部分：模型看不到当时的现场，
        重写只会让它有机会把已经做对的事改错。
        """
        requirement = (
            "只输出还需要执行的剩余步骤" if remaining_only else "输出完整的新计划"
        )
        tool_names = "、".join(t.name for t in self.tools) or "（无）"
        prompt = (
            f"任务：{task}\n"
            f"要求：{requirement}。\n"
            f"可用工具：{tool_names}。\n"
            + (f"上下文：\n{context}\n" if context else "")
            + "只输出 JSON，格式："
            '{"strategy":"...","steps":[{"id":"s1","goal":"...","tool":"...",'
            '"args":{},"done_when":"...","depends_on":[]}]}'
        )
        reply = self.model.complete([system(PLANNER_PROMPT), human(prompt)])
        self.calls += 1
        return parse_plan(reply.content)


class PlanExecutor:
    """按计划执行，必要时重规划。

    执行器和规划器分成两个角色，是因为它们的失败面不一样：
    执行器只管"这一步能不能跑"，规划器只管"这份计划合不合格"。
    中间那道闸就是 :func:`validate_plan`。
    """

    def __init__(
        self,
        planner: Planner,
        model: ChatModel,
        tools: Sequence[Tool],
        spec: ExecutionSpec = ExecutionSpec(),
    ) -> None:
        self.planner = planner
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.spec = spec
        self.calls = 0
        """执行阶段的模型调用次数——每执行一步就多一次。"""
        self.prompt_tokens = 0
        self.tool_invocations: dict[str, int] = {}
        """每个工具被调了几次——用来证明已完成的工作没有被重做。"""

    def run(self, task: str, plan: Plan) -> PlanRunResult:
        """执行一份计划，返回完整结果。"""
        result = PlanRunResult(task=task, strategy=self.spec.mode, plan_versions=[plan])
        self._run_inner(task, plan, result)
        result.planner_calls = self.planner.calls
        result.executor_calls = self.calls
        result.prompt_tokens = self.prompt_tokens
        result.tool_invocations = dict(self.tool_invocations)
        return result

    def _run_inner(self, task: str, plan: Plan, result: PlanRunResult) -> None:
        """真正的执行循环——所有出口都写进 ``result``。"""
        problems = validate_plan(plan, set(self.tools))
        if len(plan.steps) > self.spec.max_steps:
            problems.insert(
                0,
                f"计划有 {len(plan.steps)} 步，超过上限 {self.spec.max_steps} 步",
            )
        if problems:
            result.stop_reason = "rejected"
            result.error = "；".join(problems)
            return

        current = plan
        done_ids: set[str] = set()
        produced: dict[str, str] = {}
        replans = 0

        while True:
            restart = False
            for step in current.steps:
                if step.id in done_ids:
                    # 老计划里已经做过的步骤不重做——这是"不浪费"的底线
                    continue
                if step.id in self.spec.skip_step_ids:
                    result.executions.append(
                        StepExecution(step.id, "skipped", "被跳过，没有产出")
                    )
                    continue

                broken = detect_broken_dependencies(step, current, done_ids, produced)
                if broken:
                    # 依赖断裂要在花掉一次模型调用之前发现
                    result.stop_reason = "broken_dependency"
                    result.error = "；".join(broken)
                    return

                outcome = self._execute_step(step, task, current, done_ids)
                result.executions.append(
                    StepExecution(
                        step.id,
                        "ok" if outcome.ok else "failed",
                        outcome.output or outcome.error,
                    )
                )
                if outcome.ok:
                    result.completed.append(outcome)
                    done_ids.add(step.id)
                    produced[step.id] = outcome.output
                    continue

                result.failed = outcome
                if self.spec.mode == "plan_once" or replans >= self.spec.max_replans:
                    result.stop_reason = "failed"
                    result.error = outcome.error
                    return

                new_plan = self._replan(task, current, result.completed, outcome)
                replans += 1
                current = replace(new_plan, version=len(result.plan_versions) + 1)
                result.plan_versions.append(current)
                restart = True
                break

            if not restart:
                result.stop_reason = "completed"
                return

    def _execute_step(
        self,
        step: PlanStep,
        task: str,
        plan: Plan,
        done_ids: set[str],
    ) -> StepOutcome:
        """执行一步：问模型要工具调用，跑工具，判定成败。"""
        prompt = self._build_prompt(task, step, plan, done_ids)
        self.prompt_tokens += approx_tokens(prompt)
        reply = self.model.complete(
            [system(EXECUTOR_PROMPT), human(prompt)],
            tools=list(self.tools.values()),
        )
        self.calls += 1

        if not reply.tool_calls:
            return StepOutcome(step=step, ok=False, error="模型没有给出工具调用")

        call = reply.tool_calls[0]
        target = self.tools.get(call.name)
        if target is None:
            return StepOutcome(step=step, ok=False, error=f"没有名为「{call.name}」的工具")

        self.tool_invocations[call.name] = self.tool_invocations.get(call.name, 0) + 1
        try:
            output = target.invoke(call.arguments)
        except Exception as exc:  # noqa: BLE001 - 工具失败是正常业务分支
            return StepOutcome(step=step, ok=False, error=f"{type(exc).__name__}: {exc}")
        return StepOutcome(step=step, ok=True, output=output)

    def _build_prompt(
        self,
        task: str,
        step: PlanStep,
        plan: Plan,
        done_ids: set[str],
    ) -> str:
        """拼出执行一步时给模型的 prompt。

        为什么每一步都要带"还剩什么"：模型可能需要偏离计划
        （计划漏了一步时它能自己补），看不见全局它只能盲目执行。
        但代价是步骤越多、prompt 越大——过度分解的成本就是这么来的。
        """
        return (
            f"任务：{task}\n"
            f"当前步骤：{step.id} {step.goal}\n"
            f"建议工具：{step.tool}，"
            f"建议参数：{json.dumps(step.args, ensure_ascii=False)}\n"
            f"完成判据：{step.done_when}\n"
            f"剩余计划：\n{render_plan_context(plan, done_ids)}"
        )

    def _replan(
        self,
        task: str,
        plan: Plan,
        completed: list[StepOutcome],
        failed: StepOutcome,
    ) -> Plan:
        """失败后重新要一份计划。

        ``adaptive`` 只要剩下的步骤，``replan`` 要整份新计划。
        两种口径的差别见笔记第 5 节。
        """
        done = "\n".join(
            f"- {item.step.id} {item.step.goal} → {item.output}" for item in completed
        )
        context = (
            f"已完成的步骤：\n{done or '（无）'}\n"
            f"失败步骤：{failed.step.id} {failed.step.goal}\n"
            f"失败原因：{failed.error}\n"
        )
        return self.planner.plan(
            task,
            context=context,
            remaining_only=self.spec.mode == "adaptive",
        )


# ---------------------------------------------------------------- 测试脚手架


def step_json(
    step_id: str,
    goal: str,
    tool: str,
    args: dict[str, Any] | None = None,
    done_when: str = "拿到结果字符串",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    """把一步拼成模型该返回的 JSON 片段。"""
    return {
        "id": step_id,
        "goal": goal,
        "tool": tool,
        "args": args if args is not None else {},
        "done_when": done_when,
        "depends_on": depends_on if depends_on is not None else [],
    }


def plan_json(steps: list[dict[str, Any]], strategy: str = "plan_once") -> str:
    """把一份计划序列化成模型该返回的 JSON 文本。"""
    return json.dumps({"strategy": strategy, "steps": steps}, ensure_ascii=False)


def _make_executor(
    planner_script: list[str],
    executor_script: list[ModelReply],
    spec: ExecutionSpec,
) -> tuple[Planner, PlanExecutor, Any]:
    """搭一套"规划器 + 执行器"，两个模型各走各的脚本。

    为什么要两个模型而不是一个：分开之后才能分别数
    "规划了几次"和"执行了几次"——这正是三种策略的差别所在。
    """
    planner_model = scripted_model([ModelReply(content=text) for text in planner_script])
    executor_model = scripted_model(executor_script)
    planner = Planner(planner_model, TOOLS)
    executor = PlanExecutor(planner, executor_model, TOOLS, spec)
    return planner, executor, planner_model


# ---------------------------------------------------------------- 1. 一次性规划


def check_plan_once() -> None:
    """一次性规划：一次规划调用产出计划，再按计划执行。"""
    section("一次性规划：1 次规划调用 → N 次执行调用")

    steps = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        step_json("s2", "查询库存 SKU-1001 的余量", "查询库存", {"sku": "SKU-1001"}),
        step_json(
            "s3",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单 A-1001 与库存 SKU-1001"},
        ),
    ]
    executor_script = [
        ModelReply(tool_calls=[ToolCall("c1", "查询订单", {"order_id": "A-1001"})]),
        ModelReply(tool_calls=[ToolCall("c2", "查询库存", {"sku": "SKU-1001"})]),
        ModelReply(
            tool_calls=[
                ToolCall("c3", "生成报告", {"summary": "订单 A-1001 与库存 SKU-1001"})
            ]
        ),
    ]
    planner, executor, _ = _make_executor(
        [plan_json(steps)], executor_script, ExecutionSpec(mode="plan_once")
    )

    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)

    ensure(planner.calls == 1, f"期望只规划 1 次，实际规划了 {planner.calls} 次")
    ensure(len(plan.steps) == 3, f"期望计划有 3 步，实际 {len(plan.steps)} 步")
    ensure(
        executor.calls == len(plan.steps),
        f"期望执行调用次数等于计划步数 {len(plan.steps)}，实际 {executor.calls} 次",
    )
    ensure(
        result.stop_reason == "completed",
        f"期望正常跑完，实际 stop_reason={result.stop_reason!r}，error={result.error!r}",
    )
    ensure(
        len(result.completed) == len(plan.steps),
        f"期望完成 {len(plan.steps)} 步，实际完成 {len(result.completed)} 步",
    )
    print(
        f"  规划调用 {planner.calls} 次，执行调用 {executor.calls} 次，"
        f"计划步数 {len(plan.steps)} 步 → 三者对得上"
    )


# ---------------------------------------------------------------- 2. 重规划


def check_replan() -> None:
    """第 2 步失败 → 触发一次重规划 → 计划被替换。"""
    section("重规划：第 2 步失败 → 一次重规划 → 换掉整份计划")

    first = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        step_json("s2", "查询库存 SKU-404 的余量", "查询库存", {"sku": "SKU-404"}),
        step_json(
            "s3",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单 A-1001 与库存"},
        ),
    ]
    second = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        # 新计划的第 2 步换了 SKU——这就是"重规划"的实际产物
        step_json("s2", "查询库存 SKU-1001 的余量", "查询库存", {"sku": "SKU-1001"}),
        step_json(
            "s3",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单 A-1001 与库存"},
        ),
    ]
    executor_script = [
        ModelReply(tool_calls=[ToolCall("c1", "查询订单", {"order_id": "A-1001"})]),
        ModelReply(tool_calls=[ToolCall("c2", "查询库存", {"sku": "SKU-404"})]),
        ModelReply(tool_calls=[ToolCall("c3", "查询库存", {"sku": "SKU-1001"})]),
        ModelReply(
            tool_calls=[ToolCall("c4", "生成报告", {"summary": "订单 A-1001 与库存"})]
        ),
    ]
    planner, executor, _ = _make_executor(
        [plan_json(first), plan_json(second, "replan")],
        executor_script,
        ExecutionSpec(mode="replan"),
    )

    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)

    ensure(
        planner.calls == 2,
        f"期望规划 2 次（1 次计划 + 1 次重规划），实际 {planner.calls} 次",
    )
    ensure(
        len(result.plan_versions) == 2,
        f"期望留下 2 个计划版本，实际 {len(result.plan_versions)} 个",
    )
    ensure(
        result.plan_versions[1].steps != result.plan_versions[0].steps,
        "期望重规划后计划内容发生变化，实际两份计划完全一样",
    )
    ensure(
        result.plan_versions[1].version == 2,
        f"期望第二个计划版本号是 2，实际 {result.plan_versions[1].version}",
    )
    ensure(
        result.plan is result.plan_versions[-1],
        "期望当前生效的计划就是最后一个版本",
    )
    ensure(
        result.stop_reason == "completed",
        f"期望重规划后跑完，实际 stop_reason={result.stop_reason!r}，error={result.error!r}",
    )
    ensure(
        result.failed is not None and "SKU-404" in result.failed.error,
        f"期望记录下第一次失败，实际 {result.failed}",
    )
    ensure(
        executor.tool_invocations.get("查询订单") == 1,
        "期望已完成的第一步没有被重做，实际查询订单被调用了 "
        f"{executor.tool_invocations.get('查询订单')} 次",
    )
    print(
        f"  规划调用 {planner.calls} 次（+1 次重规划），"
        f"执行调用 {executor.calls} 次，查询订单只查了 1 次（已完成的步骤没重做）"
    )


# ---------------------------------------------------------------- 3. 步数上限


def check_step_limit() -> None:
    """计划产出 10 步而上限是 5 → 拒绝执行并说明原因。"""
    section("步数上限：计划 10 步、上限 5 步 → 拒绝执行")

    steps = [
        step_json(
            f"s{index}",
            f"查询订单 A-100{index} 的状态",
            "查询订单",
            {"order_id": f"A-100{index}"},
        )
        for index in range(1, 11)
    ]
    planner, executor, _ = _make_executor(
        [plan_json(steps)],
        [ModelReply(content="这一步不该被执行到")],
        ExecutionSpec(mode="plan_once", max_steps=5),
    )

    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)

    ensure(
        result.stop_reason == "rejected",
        f"期望拒绝执行，实际 stop_reason={result.stop_reason!r}",
    )
    ensure(
        "10" in result.error and "5" in result.error,
        f"期望拒绝原因里写清计划步数 10 和上限 5，实际 {result.error!r}",
    )
    ensure(
        executor.calls == 0,
        f"期望一步都没执行，实际执行了 {executor.calls} 步",
    )
    print(f"  拒绝原因：{result.error}")


# ---------------------------------------------------------------- 4. 不存在的工具


def check_unknown_tool() -> None:
    """计划里出现注册表里没有的工具 → 在执行前就被点名拒绝。"""
    section("幻觉工具：计划里写了不存在的工具 → 拒绝执行")

    steps = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        step_json("s2", "把结果发给客户", "发送邮件", {"to": "a@b.com"}),
    ]
    planner, executor, _ = _make_executor(
        [plan_json(steps)],
        [ModelReply(content="这一步不该被执行到")],
        ExecutionSpec(mode="plan_once"),
    )

    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)

    ensure(
        result.stop_reason == "rejected",
        f"期望拒绝执行，实际 stop_reason={result.stop_reason!r}",
    )
    ensure(
        "发送邮件" in result.error,
        f"期望错误信息点名不存在的工具「发送邮件」，实际 {result.error!r}",
    )
    ensure(
        executor.calls == 0,
        f"期望一步都没执行，实际执行了 {executor.calls} 步",
    )
    print(f"  拒绝原因：{result.error}")


# ---------------------------------------------------------------- 5. 自适应分解


def check_adaptive_replan() -> None:
    """第 3 步失败时只重规划剩余部分，已完成的步骤不重做。"""
    section("自适应分解：只重规划剩余部分，已完成的工作不重做")

    first = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        step_json("s2", "查询库存 SKU-1002 的余量", "查询库存", {"sku": "SKU-1002"}),
        step_json("s3", "查询库存 SKU-404 的余量", "查询库存", {"sku": "SKU-404"}),
        step_json("s4", "查询订单 A-1003 的状态", "查询订单", {"order_id": "A-1003"}),
        step_json(
            "s5",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单与库存汇总"},
        ),
    ]
    remaining = [
        step_json("s3r", "查询库存 SKU-1001 的余量", "查询库存", {"sku": "SKU-1001"}),
        step_json("s4", "查询订单 A-1003 的状态", "查询订单", {"order_id": "A-1003"}),
        step_json(
            "s5",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单与库存汇总"},
        ),
    ]
    executor_script = [
        ModelReply(tool_calls=[ToolCall("c1", "查询订单", {"order_id": "A-1001"})]),
        ModelReply(tool_calls=[ToolCall("c2", "查询库存", {"sku": "SKU-1002"})]),
        ModelReply(tool_calls=[ToolCall("c3", "查询库存", {"sku": "SKU-404"})]),
        ModelReply(tool_calls=[ToolCall("c4", "查询库存", {"sku": "SKU-1001"})]),
        ModelReply(tool_calls=[ToolCall("c5", "查询订单", {"order_id": "A-1003"})]),
        ModelReply(
            tool_calls=[ToolCall("c6", "生成报告", {"summary": "订单与库存汇总"})]
        ),
    ]
    planner, executor, planner_model = _make_executor(
        [plan_json(first), plan_json(remaining, "adaptive")],
        executor_script,
        ExecutionSpec(mode="adaptive"),
    )

    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)

    ensure(
        planner.calls == 2,
        f"期望规划 2 次（1 次计划 + 1 次重规划），实际 {planner.calls} 次",
    )
    ensure(
        executor.calls == 6,
        f"期望执行调用 6 次（失败前 3 次 + 重规划后 3 次），实际 {executor.calls} 次",
    )

    executed_ids = [item.step_id for item in result.executions]
    ensure(
        executed_ids.count("s1") == 1,
        f"期望 s1 只执行 1 次，实际 {executed_ids.count('s1')} 次",
    )
    ensure(
        executed_ids.count("s2") == 1,
        f"期望 s2 只执行 1 次，实际 {executed_ids.count('s2')} 次",
    )
    ensure(
        len(executed_ids) == len(set(executed_ids)),
        f"期望没有步骤被重复执行，实际执行序列 {executed_ids}",
    )
    ensure(
        executor.tool_invocations.get("查询订单") == 2,
        "期望查询订单只被调用 2 次（s1 与 s4 各一次）；"
        f"实际 {executor.tool_invocations.get('查询订单')} 次——s1 被重做了",
    )

    final_ids = {step.id for step in result.plan.steps}
    ensure(
        "s1" not in final_ids and "s2" not in final_ids,
        f"期望新计划只包含剩余步骤，实际包含 {sorted(final_ids)}",
    )
    ensure(
        "s3r" in final_ids,
        f"期望新计划里出现修复后的步骤 s3r，实际 {sorted(final_ids)}",
    )
    ensure(
        result.stop_reason == "completed",
        f"期望重规划后跑完，实际 stop_reason={result.stop_reason!r}，error={result.error!r}",
    )

    replan_prompt = planner_model.calls[1][-1].content
    ensure(
        "剩余" in replan_prompt,
        f"期望重规划时明确要求只输出剩余步骤，实际提示词里没有：{replan_prompt[:120]!r}",
    )
    print(
        f"  执行序列 {' → '.join(executed_ids)}：每一步各做一次，没有重做"
    )


# ---------------------------------------------------------------- 6. 计划质量校验


def check_plan_quality() -> None:
    """可执行、无歧义、粒度合适——三条标准都要能判不合格。"""
    section("计划质量：可执行 / 无歧义 / 粒度合适")

    good = Plan(
        steps=(
            PlanStep(
                "s1",
                "查询订单 A-1001 的状态",
                "查询订单",
                {"order_id": "A-1001"},
                "拿到订单状态文本",
            ),
        )
    )
    ensure(
        validate_plan(good, set(TOOL_NAMES)) == [],
        f"期望这份计划合格，实际问题 {validate_plan(good, set(TOOL_NAMES))}",
    )

    # 缺完成判据：步骤做完了你也说不清
    no_criterion = Plan(
        steps=(
            PlanStep("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}, ""),
        )
    )
    problems = validate_plan(no_criterion, set(TOOL_NAMES))
    ensure(
        any("完成判据" in item for item in problems),
        f"期望检出缺少完成判据，实际 {problems}",
    )

    # 粒度过粗：目标里全是空词，做完也验证不了
    coarse = Plan(
        steps=(
            PlanStep(
                "s1",
                "完成整个项目的所有工作",
                "查询订单",
                {"order_id": "A-1001"},
                "有结果",
            ),
        )
    )
    problems = validate_plan(coarse, set(TOOL_NAMES))
    ensure(
        any("粒度过粗" in item for item in problems),
        f"期望检出粒度过粗，实际 {problems}",
    )

    # 端到端：坏计划必须在执行前被拒
    planner, executor, _ = _make_executor(
        [plan_json([step_json("s1", "完成整个项目的所有工作", "查询订单", {}, "")])],
        [ModelReply(content="这一步不该被执行到")],
        ExecutionSpec(mode="plan_once"),
    )
    plan = planner.plan(TASK)
    result = executor.run(TASK, plan)
    ensure(
        result.stop_reason == "rejected",
        f"期望坏计划被拒绝执行，实际 stop_reason={result.stop_reason!r}",
    )
    ensure(
        executor.calls == 0,
        f"期望坏计划一步都没执行，实际执行了 {executor.calls} 步",
    )
    print(f"  拒绝原因：{result.error}")


# ---------------------------------------------------------------- 7. 依赖断裂


def check_broken_dependency() -> None:
    """第 1 步被跳过 → 第 2 步依赖它的产出 → 检出依赖断裂。"""
    section("依赖断裂：被跳过的步骤提供不了产出")

    steps = [
        step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"}),
        step_json(
            "s2",
            "生成订单与库存汇总报告",
            "生成报告",
            {"summary": "订单 A-1001"},
            depends_on=["s1"],
        ),
    ]
    executor_script = [
        ModelReply(tool_calls=[ToolCall("c1", "查询订单", {"order_id": "A-1001"})]),
        ModelReply(
            tool_calls=[ToolCall("c2", "生成报告", {"summary": "订单 A-1001"})]
        ),
    ]

    # 对照组：不跳过 → 正常跑完，说明这条依赖本身是成立的
    planner_ok, executor_ok, _ = _make_executor(
        [plan_json(steps)], executor_script, ExecutionSpec(mode="plan_once")
    )
    ok_result = executor_ok.run(TASK, planner_ok.plan(TASK))
    ensure(
        ok_result.stop_reason == "completed",
        f"期望对照组正常跑完，实际 stop_reason={ok_result.stop_reason!r}",
    )
    ensure(
        executor_ok.calls == 2,
        f"期望对照组执行 2 步，实际 {executor_ok.calls} 步",
    )

    # 实验组：跳过第 1 步 → 第 2 步悬空
    planner_skip, executor_skip, _ = _make_executor(
        [plan_json(steps)],
        [ModelReply(content="这一步不该被执行到")],
        ExecutionSpec(mode="plan_once", skip_step_ids=frozenset({"s1"})),
    )
    skip_result = executor_skip.run(TASK, planner_skip.plan(TASK))
    ensure(
        skip_result.stop_reason == "broken_dependency",
        f"期望检出依赖断裂，实际 stop_reason={skip_result.stop_reason!r}",
    )
    ensure(
        "s1" in skip_result.error,
        f"期望错误信息点名依赖的 s1，实际 {skip_result.error!r}",
    )
    ensure(
        executor_skip.calls == 0,
        f"期望在花掉一次模型调用之前就检出断裂，实际执行了 {executor_skip.calls} 步",
    )
    print(f"  对照组：{ok_result.stop_reason}；实验组：{skip_result.error}")


# ---------------------------------------------------------------- 8. 分解的代价


def check_decomposition_cost() -> None:
    """分解太细的代价：每一步都要重新调一次模型，prompt 还更长。"""
    section("过度分解的代价：模型调用次数与 prompt 长度一起涨")

    one_step = [step_json("s1", "查询订单 A-1001 的状态", "查询订单", {"order_id": "A-1001"})]
    eight_steps = [
        step_json(
            f"s{index}",
            f"查询订单 A-100{index} 的状态",
            "查询订单",
            {"order_id": f"A-100{index}"},
        )
        for index in range(1, 9)
    ]

    def _run_with(script_steps: list[dict[str, Any]]) -> tuple[PlanExecutor, PlanRunResult]:
        executor_script = [
            ModelReply(
                tool_calls=[
                    ToolCall(f"c{index}", "查询订单", {"order_id": f"A-100{index}"})
                ]
            )
            for index in range(1, len(script_steps) + 1)
        ]
        planner, executor, _ = _make_executor(
            [plan_json(script_steps)],
            executor_script,
            # 这里放开步数上限：这一节要量的就是"分解得细要花多少",
            # 让计划先被步数校验拦掉就量不到了
            ExecutionSpec(mode="plan_once", max_steps=10),
        )
        plan = planner.plan(TASK)
        return executor, executor.run(TASK, plan)

    executor_one, result_one = _run_with(one_step)
    executor_eight, result_eight = _run_with(eight_steps)

    ensure(
        executor_one.calls == 1 and executor_eight.calls == 8,
        f"期望执行调用分别是 1 次和 8 次，实际 {executor_one.calls} 次和 {executor_eight.calls} 次",
    )
    ensure(
        result_one.stop_reason == "completed" and result_eight.stop_reason == "completed",
        "期望两种分解都能跑完",
    )
    ensure(
        executor_eight.prompt_tokens > executor_one.prompt_tokens * 4,
        f"期望 8 步的 prompt 总量远大于 1 步："
        f"实际 {executor_eight.prompt_tokens} vs {executor_one.prompt_tokens} token",
    )
    print(
        f"  1 步：{1 + executor_one.calls} 次模型调用，{executor_one.prompt_tokens} token prompt；"
        f"8 步：{1 + executor_eight.calls} 次模型调用，{executor_eight.prompt_tokens} token prompt"
    )


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型出一份计划，只校验形状不校验内容。"""
    banner("第 7 章 · 真实模型路径")
    from ..labkit import real_model

    planner = Planner(real_model(), TOOLS)
    plan = planner.plan(TASK)

    print(f"  计划步数：{len(plan.steps)}")
    for step in plan.steps:
        print(f"    {step.id} {step.goal}（工具：{step.tool}）")

    problems = validate_plan(plan, set(TOOL_NAMES))
    ensure(len(plan.steps) > 0, "真实模型应当给出至少一个步骤")
    ensure(
        all(step.id for step in plan.steps),
        "真实模型的每一步都应当有 id",
    )
    print(f"  质量校验问题：{problems or '无'}")


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 7 章实验。"""
    banner("第 7 章 · 任务分解：规划模式与自适应分解")
    check_plan_once()
    check_replan()
    check_step_limit()
    check_unknown_tool()
    check_adaptive_replan()
    check_plan_quality()
    check_broken_dependency()
    check_decomposition_cost()
    rule("失败点能预知就重规划，路径未知才自适应；计划必须能被机器校验，否则只是把错误往后拖")
    print("\n✓ 第 7 章通过")
