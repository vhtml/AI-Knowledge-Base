"""第 10 章实验：可靠性——重试、超时、幂等与降级。

这个模块解决什么问题：前九章搭出了一个能跑的 Agent，但它默认"模型和工具
总会成功"。真实世界不成立：模型会调不存在的工具、会传错参数、会幻觉；
工具会超时、会被限流、会碰到资源不存在和权限不足。

失败一旦出现，问题就变成两个：

1. **该重试还是该放弃**——重试错了会烧钱、会把后端打垮、会把副作用做两遍；
2. **失败之后说什么**——错误回填的措辞直接决定模型能自我纠正，还是拿同一组
   参数原地重试到把轮次上限跑满。

所以本章把六件事做成可执行的断言：失败分类、指数退避与抖动、超时的三个
层次、幂等键、降级链、熔断器；再加一组"措辞决定行为"的对照实验。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from ..labkit import (
    ChatModel,
    Message,
    ModelReply,
    Tool,
    ToolCall,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    tool,
    tool_result,
)
from ..runtime import AgentResult, Step

# ---------------------------------------------------------------- 失败分类


class TransientError(Exception):
    """瞬时故障：网络抖动、限流、后端刚重启——重试有可能成功。"""


class DeterministicError(Exception):
    """确定性故障：参数非法、schema 不匹配——同样的输入永远同样的结果。"""


class BusinessError(Exception):
    """业务故障：资源不存在、权限不足——重试和改参数都没用。"""


class ToolTimeout(Exception):
    """工具调用超时：没拿到结果，但也不代表对方没做。"""


FailureKind = Literal[
    "transient",
    "deterministic",
    "business",
    "model",
    "timeout",
    "circuit",
    "unknown",
]

RETRYABLE: frozenset[FailureKind] = frozenset({"transient", "timeout"})
"""只有这两类值得重试；其余几类重试只是把同一件事做第二遍。"""


def classify(exc: BaseException) -> FailureKind:
    """把异常映射到失败类别——决定"该重试还是该放弃"的第一步。

    分类错了，后面全错：把业务故障当瞬时故障，重试就是白白打后端；
    把参数错误当瞬时故障，重试就是让模型用同样的错参数再撞一次。
    """
    if isinstance(exc, ToolTimeout):
        return "timeout"
    if isinstance(exc, TransientError):
        return "transient"
    if isinstance(exc, DeterministicError):
        return "deterministic"
    if isinstance(exc, BusinessError):
        return "business"
    if isinstance(exc, TypeError):
        # 参数名或类型不对——第 2 章运行时就是这么分的
        return "deterministic"
    if isinstance(exc, ValueError):
        # 值不合法：资源不存在、取值越界——业务语义上的否定
        return "business"
    return "unknown"


def feedback_for(kind: FailureKind, exc: BaseException) -> str:
    """把失败翻译成给模型看的文字。

    措辞不是装饰，是控制手段：

    - 确定性故障**回原文**——点名参数、说清期望，模型才改得对；
    - 业务故障**必须改写**——明确告诉模型"重试没用"，否则它会拿同一组
      参数反复重试；
    - 瞬时故障说明系统已经自己重试过，免得模型以为工具没被调用。
    """
    raw = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    if kind == "deterministic":
        return f"参数错误：{raw}。请修正参数后重新调用（系统不会替你重试这次调用）。"
    if kind == "business":
        return (
            f"业务故障：{raw}。重试无法解决该问题，"
            "请改用其他方案或直接告知用户，不要重复调用同一参数。"
        )
    if kind == "model":
        return f"错误：{raw}。请改用工具列表中的其它工具。"
    if kind == "transient":
        return f"临时故障：{raw}。本次调用未成功，请稍后再试或改用其他工具。"
    if kind == "timeout":
        return f"调用超时：{raw}。请缩小请求范围或改用其他工具。"
    if kind == "circuit":
        return f"服务熔断：{raw}。该依赖当前不可用，请直接告知用户或改用其它工具。"
    return f"未知错误：{raw}。无法判断是否可重试，默认不重试。"


# ---------------------------------------------------------------- 重试策略


@dataclass
class RetryPolicy:
    """重试策略：次数上限 + 指数退避 + 有界抖动。"""

    max_attempts: int = 3
    """总尝试次数（含第一次）。3 次的意思是"最多再试两次"。"""

    base_delay: float = 0.1
    """第一次重试前等多久（秒）。"""

    max_delay: float = 2.0
    """单次等待上限，防止指数增长到无法接受。"""

    jitter_ratio: float = 0.5
    """抖动幅度占基线的比例。"""

    def base_delay_for(self, attempt: int) -> float:
        """第 attempt 次重试前的基线等待（attempt 从 1 开始）。

        等比退避而不是等差：等差的等 10 次还是那个间隔，等比的第 10 次
        已经等了 512 倍——后端越是被压垮，越需要退得久一点。
        """
        if attempt < 1:
            raise ValueError("attempt 从 1 开始")
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """带抖动的实际等待。

        抖动幅度不超过基线的一半，这样等待序列在没触到上限前**严格递增**：
        既打散同步，又不会让某一次重试比纯指数退避还慢。

        抖动的作用不是"随机一点好看"，是防止所有客户端在同一时刻醒来——
        服务刚恢复的那一秒最脆弱，所有重试一起打过去等于把它再打挂一次。
        """
        base = self.base_delay_for(attempt)
        return base * (1.0 + self.jitter_ratio * rng.random())


# ---------------------------------------------------------------- 熔断器

CircuitState = Literal["closed", "open", "half_open"]


@dataclass
class CircuitBreaker:
    """熔断器：连续失败到阈值就停止打后端。

    统计的是"后端不健康"，不是"业务没成功"——业务上的资源不存在说明
    后端活得好好的，不该因此熔断。
    """

    threshold: int = 3
    """连续失败多少次就打开熔断。"""

    reset_after: float = 30.0
    """打开后过多久放一个探测请求出去。"""

    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    state: CircuitState = "closed"
    opened_at: float | None = None
    probe_in_flight: bool = False

    def allow(self) -> bool:
        """当前是否允许调用后端。"""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.opened_at is not None and self.clock() - self.opened_at >= self.reset_after:
                # 冷却结束：进半开态，放一个探测请求探探路
                self.state = "half_open"
                self.probe_in_flight = False
            else:
                return False
        if self.state == "half_open":
            if self.probe_in_flight:
                return False
            self.probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        """一次成功就恢复：失败计数清零，熔断关闭。"""
        self.failures = 0
        self.state = "closed"
        self.opened_at = None
        self.probe_in_flight = False

    def record_failure(self) -> None:
        """一次失败就累加计数，达到阈值就打开熔断。"""
        self.failures += 1
        self.probe_in_flight = False
        if self.state == "half_open" or self.failures >= self.threshold:
            self.state = "open"
            self.opened_at = self.clock()


# ---------------------------------------------------------------- 幂等


@dataclass
class IdempotencyStore:
    """幂等键仓库：同一个键只允许产生一次副作用。

    这是"服务端去重"语义，所以放在工具那一侧——只有真正执行副作用的
    系统才知道副作用到底发生没有。客户端拿着同一个键重试，服务端把上次
    的结果原样返回，就不会扣第二次款。
    """

    results: dict[str, str] = field(default_factory=dict)
    side_effects: int = 0
    """真正发生过的副作用次数——用来证明"没重复扣"。"""

    def seen(self, key: str) -> bool:
        """这个幂等键是否已经执行过。"""
        return key in self.results

    def record(self, key: str, result: str) -> None:
        """记住某个幂等键的结果。"""
        self.results[key] = result

    def count_side_effect(self) -> None:
        """副作用计数加一。"""
        self.side_effects += 1


# ---------------------------------------------------------------- 执行器


SKIP = "__skip__"
"""降级链里的哨兵：表示"这一步跳过，别再找别的工具了"。"""


@dataclass
class ToolOutcome:
    """一次工具调用的结果：成功内容，或失败时回填给模型的文字。"""

    content: str
    ok: bool
    kind: FailureKind | None = None
    attempts: int = 1
    degraded: bool = False
    cached: bool = False


@dataclass
class ReliableRunner:
    """带可靠性护栏的工具执行器。

    一条工具调用要依次穿过：**熔断器 → 幂等键 → 超时 → 重试 → 降级**。
    顺序不是随意的：熔断在最前面（不该打后端就别打），幂等在执行之前
    （重复请求先看有没有做过），重试只包住真正的执行，降级在最后兜底。
    """

    tools: dict[str, Tool]
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    idempotency: IdempotencyStore = field(default_factory=IdempotencyStore)
    fallbacks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """工具名 → 备用工具名序列；里面可以放 :data:`SKIP` 哨兵。"""

    stale_cache: dict[str, str] = field(default_factory=dict)
    """工具名 → 上次的成功结果，用于"用旧缓存"降级。"""

    idempotency_param: str = "idempotency_key"
    """哪个参数是幂等键。"""

    call_timeout: float | None = None
    """单次工具调用的超时（秒）；None 表示不限。"""

    sleep: Callable[[float], None] = time.sleep
    """等待函数做成可注入的——测试里换成一个记录列表，就不用真的睡。"""

    rng: random.Random = field(default_factory=lambda: random.Random(20261118))
    """抖动用的随机源。固定种子 → 同一章每次跑出来的退避序列都一样，便于对比。"""

    feedback: Callable[[FailureKind, BaseException], str] = feedback_for
    """错误回填的措辞函数；换成"原样返回异常"就能做对照实验。"""

    def invoke(self, call: ToolCall) -> ToolOutcome:
        """执行一次工具调用，把所有失败都翻译成模型能读懂的话。"""
        target = self.tools.get(call.name)
        if target is None:
            known = ", ".join(sorted(self.tools)) or "（无）"
            exc = DeterministicError(f"没有名为 {call.name} 的工具。可用工具：{known}")
            return ToolOutcome(
                content=self.feedback("model", exc),
                ok=False,
                kind="model",
                attempts=0,
            )

        if not self.breaker.allow():
            exc = BusinessError("熔断器已打开，依赖服务暂时不可用")
            return ToolOutcome(
                content=self.feedback("circuit", exc),
                ok=False,
                kind="circuit",
                attempts=0,
            )

        key = str(call.arguments.get(self.idempotency_param) or "")
        if key and self.idempotency.seen(key):
            # 同一个幂等键已经执行过：直接复用结果，副作用不会再发生一次
            return ToolOutcome(content=self.idempotency.results[key], ok=True, attempts=0, cached=True)

        attempt = 0
        last_exc: BaseException | None = None
        last_kind: FailureKind = "unknown"
        while attempt < self.policy.max_attempts:
            attempt += 1
            try:
                result = self._invoke_with_timeout(target, call.arguments)
            except BaseException as exc:  # noqa: BLE001 - 所有失败都要转成模型能读懂的话
                last_exc = exc
                last_kind = classify(exc)
                if last_kind in {"transient", "timeout", "unknown"}:
                    # 只有"后端不健康"才计入熔断统计
                    self.breaker.record_failure()
                if last_kind not in RETRYABLE or attempt >= self.policy.max_attempts:
                    break
                self.sleep(self.policy.delay_for(attempt, self.rng))
                continue
            self.breaker.record_success()
            if key:
                self.idempotency.record(key, result)
            return ToolOutcome(content=result, ok=True, attempts=attempt)

        exc = last_exc or RuntimeError("未知失败")
        outcome = ToolOutcome(
            content=self.feedback(last_kind, exc),
            ok=False,
            kind=last_kind,
            attempts=attempt,
        )
        return self._degrade(call, outcome)

    def _degrade(self, call: ToolCall, failure: ToolOutcome) -> ToolOutcome:
        """降级阶梯：备用工具 → 旧缓存 → 跳过 → 明确告知用户。

        每一级对用户的可见度不同，从"用户完全无感"到"最诚实但最难看"。
        """
        for name in self.fallbacks.get(call.name, ()):
            if name == SKIP:
                return ToolOutcome(
                    content=f"[已跳过：{call.name}] 主工具不可用，本步骤被跳过，请据此作答。",
                    ok=False,
                    kind=failure.kind,
                    attempts=failure.attempts,
                    degraded=True,
                )
            backup = self.tools.get(name)
            if backup is None:
                continue
            try:
                result = self._invoke_with_timeout(backup, call.arguments)
            except BaseException:  # noqa: BLE001 - 备用工具也失败就继续往下走
                continue
            return ToolOutcome(
                content=f"[降级：{call.name} → {name}] {result}",
                ok=True,
                attempts=failure.attempts,
                degraded=True,
            )

        cached = self.stale_cache.get(call.name)
        if cached is not None:
            return ToolOutcome(
                content=f"[旧缓存：{call.name}] {cached}",
                ok=True,
                attempts=failure.attempts,
                degraded=True,
            )

        return failure

    def _invoke_with_timeout(self, target: Tool, arguments: dict[str, Any]) -> str:
        """在超时约束下执行工具。

        超时必须由运行时兜住，不能指望工具自己记得超时——工具底下是一次
        HTTP 调用，它只会一直挂着。
        """
        if self.call_timeout is None:
            return target.invoke(arguments)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(target.func, **arguments)
            try:
                result = future.result(timeout=self.call_timeout)
            except TimeoutError as exc:
                raise ToolTimeout(
                    f"工具 {target.name} 在 {self.call_timeout}s 内没有返回"
                ) from exc
        finally:
            # 不 wait：超时的那个线程还在跑，等它做完只会白白拖慢退出
            pool.shutdown(wait=False, cancel_futures=True)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------- Agent


@dataclass
class ReliabilityConfig:
    """可靠性配置。

    三个超时分别管三件事：工具超时管"单次调用别挂死"，轮次超时管"这一轮
    别拖太久"，任务超时管"整个任务别不收敛"。
    """

    max_iterations: int = 8
    task_timeout: float | None = None
    round_timeout: float | None = None
    partial_policy: Literal["continue", "fail_fast"] = "continue"
    clock: Callable[[], float] = time.monotonic


class ReliableAgent:
    """在第 2 章运行时之上加可靠性护栏的 Agent。"""

    def __init__(
        self,
        model: ChatModel,
        runner: ReliableRunner,
        config: ReliabilityConfig | None = None,
    ) -> None:
        self.model = model
        self.runner = runner
        self.config = config or ReliabilityConfig()
        self.tools = runner.tools

    def run(self, user_input: str, history: Sequence[Message] = ()) -> AgentResult:
        """跑一次带可靠性护栏的 Agent 循环。

        与第 2 章运行时相比只多了三件事：工具调用走可靠执行器；一轮里多个
        工具的部分失败会被显式表达；整个任务受任务级超时约束。
        """
        config = self.config
        messages: list[Message] = list(history)
        messages.append(human(user_input))
        steps: list[Step] = []
        tool_list = list(self.tools.values())

        deadline = None if config.task_timeout is None else config.clock() + config.task_timeout

        for iteration in range(config.max_iterations):
            reply = self.model.complete(messages, tools=tool_list or None)
            messages.append(reply.to_message())
            steps.append(
                Step(index=iteration, kind="model", label="模型", detail=reply.content[:200])
            )
            if not reply.tool_calls:
                return AgentResult(
                    messages=messages,
                    final_answer=reply.content,
                    steps=steps,
                    stop_reason="completed",
                )

            round_deadline = (
                None if config.round_timeout is None else config.clock() + config.round_timeout
            )
            outcomes: list[tuple[ToolCall, ToolOutcome]] = []
            for call in reply.tool_calls:
                if deadline is not None and config.clock() > deadline:
                    return AgentResult(
                        messages=messages,
                        final_answer="",
                        steps=steps,
                        stop_reason="task_timeout",
                    )
                if round_deadline is not None and config.clock() > round_deadline:
                    outcome = ToolOutcome(
                        content=f"本轮超时（{config.round_timeout}s），工具 {call.name} 未执行：",
                        ok=False,
                        kind="timeout",
                        attempts=0,
                    )
                else:
                    outcome = self.runner.invoke(call)
                outcomes.append((call, outcome))

            for call, outcome in outcomes:
                steps.append(
                    Step(
                        index=iteration,
                        kind="tool",
                        label=call.name,
                        detail=("✓ " if outcome.ok else "✗ ") + outcome.content[:200],
                    )
                )
                messages.append(tool_result(outcome.content, tool_call_id=call.id, name=call.name))

            failed = [item for item in outcomes if not item[1].ok]
            if failed and config.partial_policy == "fail_fast":
                return AgentResult(
                    messages=messages,
                    final_answer="",
                    steps=steps,
                    stop_reason="tool_failed",
                )
            if failed:
                summary = self._partial_summary(outcomes)
                messages.append(human(summary))
                steps.append(
                    Step(index=iteration, kind="model", label="部分失败", detail=summary)
                )

        return AgentResult(
            messages=messages,
            final_answer="",
            steps=steps,
            stop_reason="max_iterations",
        )

    @staticmethod
    def _partial_summary(outcomes: Sequence[tuple[ToolCall, ToolOutcome]]) -> str:
        """把一轮里的部分失败汇总成一句话交给模型。

        为什么需要汇总：模型看到一串工具结果时，得知道哪些是真拿到了数据、
        哪些是错的——否则它可能把失败当成"没有数据"，接着往下编。

        汇总里必须**带上失败原因原文**：汇总消息排在工具消息之后，如果只写
        "失败了"，就把具体的错误埋在了上一条消息里，模型多半不会回头翻。
        """
        ok = sum(1 for _, outcome in outcomes if outcome.ok)
        bad = [
            f"{call.name}（{outcome.kind}）：{outcome.content[:120]}"
            for call, outcome in outcomes
            if not outcome.ok
        ]
        return (
            f"[执行状态] 本轮 {len(outcomes)} 个工具调用：{ok} 个成功、"
            f"{len(outcomes) - ok} 个失败。失败项：{'；'.join(bad)}。"
            "请只基于成功的结果作答，失败项请改用其它方案或告知用户。"
        )


# ---------------------------------------------------------------- 假时钟


@dataclass
class FakeClock:
    """可控时钟：每次读数自动前进，用来测超时而不真的等。"""

    now: float = 0.0
    step: float = 1.0

    def __call__(self) -> float:
        """返回当前时间并前进一格。"""
        value = self.now
        self.now += self.step
        return value


@dataclass
class SwitchingModel:
    """按错误措辞改变行为的假模型。

    真实模型的行为我们控制不了，但"措辞是否影响行为"这件事必须能测——
    所以这里用一个规则明确的假模型：看到"不要重复调用"就换工具，看到别的
    就原样重试。换过一次之后拿到结果就收尾。
    """

    primary: ToolCall
    secondary: ToolCall
    switched: bool = False
    calls: list[list[Message]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """按上一条工具结果的措辞决定下一步。

        注意它是从后往前找**最近一条工具消息**，而不是只看最后一条消息——
        真实模型读的是整段上下文，失败原因可能被排在后面的汇总消息盖住。
        """
        self.calls.append(list(messages))
        last_tool = next((m for m in reversed(messages) if m.role == "tool"), None)
        if self.switched:
            return ModelReply(content="已改用其它工具，结果如上。")
        if last_tool is not None and "不要重复调用" in last_tool.content:
            self.switched = True
            return ModelReply(tool_calls=[self.secondary])
        return ModelReply(tool_calls=[self.primary])


# ---------------------------------------------------------------- 工具

订单调用计数: dict[str, int] = {"ok": 0, "fail": 0}
"""工具调用计数器：让"到底打了几次后端"可观测，本章多处断言靠它。"""


@tool
def 查订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    if order_id == "A-9999":
        # 订单不存在属于业务故障：重试和改参数都没用，只能换一条路
        订单调用计数["fail"] += 1
        raise BusinessError(f"订单 {order_id} 不存在")
    订单调用计数["ok"] += 1
    return {"A-1001": "已发货", "A-1002": "待付款"}.get(order_id, "未找到")


@tool
def 查物流(shipment_id: str) -> str:
    """按运单号查询物流状态。

    Args:
        shipment_id: 运单号，形如 S-1001
    """
    return f"运单 {shipment_id} 运输中"


# ---------------------------------------------------------------- 1. 失败分类


def check_failure_classification() -> None:
    """四类失败，只有瞬时值得重试。"""
    section("失败分类：只有瞬时故障值得重试")

    cases: list[tuple[str, BaseException, str, bool]] = [
        ("网络抖动", TransientError("connection reset"), "transient", True),
        ("参数非法", DeterministicError("quantity 必须是正整数"), "deterministic", False),
        ("资源不存在", BusinessError("订单 A-9999 不存在"), "business", False),
        ("调用超时", ToolTimeout("工具 3s 内没有返回"), "timeout", True),
    ]

    for label, exc, expected_kind, retryable in cases:
        kind = classify(exc)
        ensure(kind == expected_kind, f"{label} 应当归类为 {expected_kind}，实际 {kind}")
        ensure(
            (kind in RETRYABLE) == retryable,
            f"{label} 的可重试性应当是 {retryable}，实际 {kind in RETRYABLE}",
        )
        print(f"  {label:<14} → {kind:<13} 可重试：{'是' if kind in RETRYABLE else '否'}")

    # TypeError 与 ValueError 的归类约定：前者是调用方式错，后者是业务否定
    ensure(classify(TypeError("unexpected keyword argument")) == "deterministic", "TypeError 应当是确定性失败")
    ensure(classify(ValueError("订单不存在")) == "business", "ValueError 应当是业务失败")
    ensure(classify(RuntimeError("说不清是什么")) == "unknown", "无法归类的异常默认按未知处理")
    ensure("unknown" not in RETRYABLE, "归类不出来的异常默认不重试——宁可少试一次，不要打错后端")

    # 第四类失败"模型幻觉了工具名"不走异常通道——它是"工具压根不存在"
    runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )
    outcome = runner.invoke(ToolCall("c1", "查价", {"item": "X"}))
    ensure(not outcome.ok, "调用不存在的工具应当判为失败")
    ensure(
        outcome.kind == "model",
        f"幻觉工具名应当归类为 model，实际 {outcome.kind}",
    )
    ensure(
        outcome.attempts == 0,
        f"幻觉工具名不该真的去调用后端，实际尝试 {outcome.attempts} 次",
    )
    ensure(
        "可用工具" in outcome.content,
        f"幻觉工具名的回填必须列出可用工具让模型改，实际 {outcome.content!r}",
    )
    print(
        f"  {'模型幻觉':<14} → {'model':<13} 可重试：否"
        f"（回填：{outcome.content[:40]}…）"
    )


# ---------------------------------------------------------------- 2. 重试


def check_retry_on_transient() -> None:
    """瞬时失败重试到成功——调用次数必须是 3 次。"""
    section("瞬时失败：重试到成功")

    state = {"calls": 0}
    delays: list[float] = []

    def 查库存(sku: str) -> str:
        """按 SKU 查询库存余量。

        Args:
            sku: 商品 SKU，形如 SKU-1001
        """
        state["calls"] += 1
        if state["calls"] <= 2:
            raise TransientError(f"上游连接被重置（第 {state['calls']} 次）")
        return f"{sku} 有货 42 件"

    runner = ReliableRunner(
        tools={"查库存": tool(查库存)},
        policy=RetryPolicy(max_attempts=3),
        sleep=delays.append,
    )
    outcome = runner.invoke(ToolCall("c1", "查库存", {"sku": "SKU-1001"}))

    ensure(outcome.ok, f"瞬时失败重试后应当成功，实际 {outcome.kind}：{outcome.content}")
    ensure(
        state["calls"] == 3,
        f"期望工具实际执行 3 次（前两次瞬时失败），实际 {state['calls']} 次",
    )
    ensure(outcome.attempts == 3, f"期望尝试次数为 3，实际 {outcome.attempts}")
    ensure(
        len(delays) == 2,
        f"期望两次重试各等一次，实际等待 {len(delays)} 次",
    )
    ensure(
        delays[1] > delays[0],
        f"期望退避间隔递增，实际 {delays}",
    )
    ensure(
        delays[0] != delays[1],
        f"期望退避间隔带抖动（两次不相等），实际 {delays}",
    )
    print(f"  调用 {state['calls']} 次成功；退避序列 {[round(d, 4) for d in delays]}")


def check_deterministic_not_retried() -> None:
    """确定性失败只调用一次，并把错误回填给模型。"""
    section("确定性失败：不重试，回原文让模型改参数")

    state = {"calls": 0}

    def 扣库存(sku: str, quantity: int) -> str:
        """按 SKU 扣减库存。

        Args:
            sku: 商品 SKU
            quantity: 扣减数量，必须是正整数
        """
        state["calls"] += 1
        if not isinstance(quantity, int):
            raise TypeError(f"参数 quantity 必须是整数，实际收到 {type(quantity).__name__}")
        return "已扣减"

    runner = ReliableRunner(
        tools={"扣库存": tool(扣库存)},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )

    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "扣库存", {"sku": "SKU-1", "quantity": "3"})]),
            ModelReply(content="参数类型不对，应当传整数，已更正。"),
        ]
    )
    agent = ReliableAgent(model, runner)
    result = agent.run("给 SKU-1 扣 3 件库存")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 1, f"应当只回填一条工具消息，实际 {len(tool_messages)} 条")
    ensure(
        "参数错误" in tool_messages[0].content,
        f"回填内容应当是参数错误，实际 {tool_messages[0].content!r}",
    )
    ensure(
        "quantity" in tool_messages[0].content,
        f"回填内容必须点名参数，模型才知道改哪个，实际 {tool_messages[0].content!r}",
    )
    ensure(
        state["calls"] == 1,
        f"确定性失败不应重试，实际调用 {state['calls']} 次",
    )
    ensure(
        len(model.calls) == 2,
        f"错误回填后模型应当被叫起来改参数一次，实际模型调用 {len(model.calls)} 次",
    )
    ensure(
        result.stop_reason == "completed",
        f"运行应当正常收尾，实际 {result.stop_reason}",
    )
    print(f"  工具只被调用 {state['calls']} 次，错误已回填：{tool_messages[0].content[:60]}…")


def check_business_not_retried() -> None:
    """业务故障不重试，而且措辞必须和参数错明显不同。"""
    section("业务故障：不重试，且措辞必须改写")

    订单调用计数["fail"] = 0
    runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )
    outcome = runner.invoke(ToolCall("c1", "查订单", {"order_id": "A-9999"}))

    ensure(not outcome.ok, "业务故障应当判为失败")
    ensure(outcome.kind == "business", f"应当归类为 business，实际 {outcome.kind}")
    ensure(
        订单调用计数["fail"] == 1,
        f"业务故障不应重试，实际调用 {订单调用计数['fail']} 次",
    )

    business_text = outcome.content
    parameter_text = feedback_for("deterministic", TypeError("参数 quantity 必须是整数"))

    ensure(
        business_text != parameter_text,
        f"业务故障与参数错的回填措辞必须不同：{business_text!r} vs {parameter_text!r}",
    )
    ensure(
        "不要重复调用" in business_text,
        f"业务故障的措辞必须明确劝阻重试，实际 {business_text!r}",
    )
    ensure(
        "参数错误" not in business_text,
        f"业务故障不能写成参数错误，否则模型会一直改参数，实际 {business_text!r}",
    )
    ensure(
        "修正参数" in parameter_text,
        f"参数错的措辞必须让模型去改参数，实际 {parameter_text!r}",
    )
    print(f"  业务故障措辞：{business_text}")
    print(f"  参数错措辞：{parameter_text}")


# ---------------------------------------------------------------- 3. 超时


def check_timeout_layers() -> None:
    """超时的三个层次：单次调用、单轮、整个任务。"""
    section("超时的三个层次")

    # 层次一：单次工具调用超时——运行时兜住，整个运行不崩
    def 慢查询(q: str) -> str:
        """执行一个很慢的查询。

        Args:
            q: 查询条件
        """
        time.sleep(0.5)
        return "查询结果"

    runner = ReliableRunner(
        tools={"慢查询": tool(慢查询)},
        policy=RetryPolicy(max_attempts=1),
        call_timeout=0.05,
        sleep=lambda _: None,
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "慢查询", {"q": "全部订单"})]),
            ModelReply(content="慢查询超时了，我先给出目前能确认的部分。"),
        ]
    )
    agent = ReliableAgent(model, runner)
    result = agent.run("查一下全部订单")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 1, f"应当回填一条工具消息，实际 {len(tool_messages)} 条")
    ensure(
        "超时" in tool_messages[0].content,
        f"回填内容应当写明超时，实际 {tool_messages[0].content!r}",
    )
    ensure(
        result.stop_reason == "completed",
        f"工具超时不该让整个运行崩掉，实际 stop_reason={result.stop_reason}",
    )
    print(f"  单次调用超时 → 回填：{tool_messages[0].content[:60]}…")

    # 层次二：单轮超时——只放弃这一轮，循环继续
    round_clock = FakeClock(step=1.0)
    round_model = scripted_model([ModelReply(tool_calls=[ToolCall("c1", "查订单", {"order_id": "A-1"})])])
    round_runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    round_agent = ReliableAgent(
        round_model,
        round_runner,
        ReliabilityConfig(max_iterations=3, round_timeout=0.5, clock=round_clock),
    )
    round_result = round_agent.run("查订单")
    ensure(
        round_result.stop_reason == "max_iterations",
        f"单轮超时只放弃本轮，兜不住不收敛的循环，实际 {round_result.stop_reason}",
    )
    ensure(
        sum(1 for m in round_result.messages if m.role == "tool") == 3,
        "每一轮都应当留下一条超时回填",
    )
    print(f"  单轮超时 → 跑了 3 轮仍未收敛（stop_reason={round_result.stop_reason}）")

    # 层次三：整个任务超时——只有它能兜住"循环不收敛"
    task_clock = FakeClock(step=1.0)
    task_model = scripted_model([ModelReply(tool_calls=[ToolCall("c1", "查订单", {"order_id": "A-1"})])])
    task_runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    task_agent = ReliableAgent(
        task_model,
        task_runner,
        ReliabilityConfig(max_iterations=50, task_timeout=2.5, clock=task_clock),
    )
    task_result = task_agent.run("查订单")
    ensure(
        task_result.stop_reason == "task_timeout",
        f"任务级超时应兜住不收敛的循环，实际 {task_result.stop_reason}",
    )
    ensure(
        len(task_model.calls) < 50,
        f"任务级超时应远早于轮次上限生效，实际跑了 {len(task_model.calls)} 轮",
    )
    print(f"  任务超时 → {len(task_model.calls)} 轮后停机（上限是 50 轮）")


# ---------------------------------------------------------------- 4. 幂等


def check_idempotency() -> None:
    """同一个幂等键只产生一次副作用。"""
    section("幂等：重试意味着可能重复执行")

    store = IdempotencyStore()
    state = {"fail_next": True, "calls": 0}

    def 扣款(amount: int, idempotency_key: str) -> str:
        """按订单扣款，同一个幂等键只扣一次。

        Args:
            amount: 扣款金额，单位元
            idempotency_key: 幂等键，同一个键只允许扣款一次
        """
        state["calls"] += 1
        if store.seen(idempotency_key):
            # 服务端去重：副作用已经发生过，直接把上次结果返回
            return store.results[idempotency_key]
        store.count_side_effect()
        result = f"已扣款 {amount} 元"
        store.record(idempotency_key, result)
        if state["fail_next"]:
            state["fail_next"] = False
            # 最危险的情形：扣款成功了，但响应在回程丢了
            raise TransientError("扣款响应丢失，结果未知")
        return result

    runner = ReliableRunner(
        tools={"扣款": tool(扣款)},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )
    call = ToolCall("c1", "扣款", {"amount": 99, "idempotency_key": "pay-2026-0018"})

    # 第一次调用：副作用发生，但响应丢了 → 触发重试
    outcome = runner.invoke(call)
    ensure(outcome.ok, f"重试后应当拿到结果，实际 {outcome.kind}：{outcome.content}")
    ensure(
        outcome.attempts == 2,
        f"期望第一次丢响应、重试一次，共 2 次尝试，实际 {outcome.attempts} 次",
    )
    ensure(
        store.side_effects == 1,
        f"幂等键应当只产生 1 次副作用，实际产生 {store.side_effects} 次",
    )
    ensure(
        "已扣款" in outcome.content,
        f"最终应当返回扣款成功的结果，实际 {outcome.content!r}",
    )
    print(f"  尝试 {outcome.attempts} 次，副作用只发生 {store.side_effects} 次")

    # 再显式调一次同一个幂等键：不该再产生副作用
    again = runner.invoke(call)
    ensure(again.ok, "重复调用应当直接返回上次结果")
    ensure(
        again.cached,
        "重复调用应当命中幂等缓存",
    )
    ensure(
        store.side_effects == 1,
        f"重复调用后副作用仍应是 1 次，实际 {store.side_effects} 次",
    )
    ensure(
        again.content == outcome.content,
        f"重复调用应当返回同一结果，实际 {again.content!r} vs {outcome.content!r}",
    )
    print(f"  重复调用命中缓存，副作用累计仍为 {store.side_effects} 次")


# ---------------------------------------------------------------- 5. 部分失败


def check_partial_failure() -> None:
    """一轮里多个工具，有的成功有的失败——怎么表达、怎么处理。"""
    section("并行工具的部分失败")

    订单调用计数["ok"] = 0
    订单调用计数["fail"] = 0

    runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=2),
        sleep=lambda _: None,
    )
    model = scripted_model(
        [
            ModelReply(
                tool_calls=[
                    ToolCall("c1", "查订单", {"order_id": "A-1001"}),
                    ToolCall("c2", "查订单", {"order_id": "A-9999"}),
                ]
            ),
            ModelReply(content="A-1001 已发货；A-9999 查不到，已在结果里标注。"),
        ]
    )
    agent = ReliableAgent(model, runner, ReliabilityConfig(partial_policy="continue"))
    result = agent.run("查 A-1001 和 A-9999")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 2, f"每个 tool_call_id 都要有回填，实际 {len(tool_messages)} 条")
    ensure(
        any("已发货" in m.content for m in tool_messages),
        f"成功的工具结果必须保留，实际 {[m.content for m in tool_messages]}",
    )
    ensure(
        any("业务故障" in m.content for m in tool_messages),
        f"失败的工具必须有错误回填，实际 {[m.content for m in tool_messages]}",
    )
    summaries = [m for m in result.messages if m.role == "user" and "执行状态" in m.content]
    ensure(len(summaries) == 1, f"应当有一条部分失败汇总，实际 {len(summaries)} 条")
    ensure(
        "1 个成功" in summaries[0].content and "1 个失败" in summaries[0].content,
        f"汇总必须说清成功与失败各几个，实际 {summaries[0].content!r}",
    )
    ensure(
        result.stop_reason == "completed",
        f"部分失败不该让整个任务失败，实际 {result.stop_reason}",
    )
    ensure(
        订单调用计数["ok"] == 1 and 订单调用计数["fail"] == 1,
        "每个工具都只应当被调用一次，"
        f"实际成功 {订单调用计数['ok']} 次、失败 {订单调用计数['fail']} 次",
    )
    print(f"  汇总：{summaries[0].content[:80]}…")

    # 另一种处理：fail_fast——只要有一个失败就放弃整轮
    fail_fast_model = scripted_model(
        [
            ModelReply(
                tool_calls=[
                    ToolCall("c1", "查订单", {"order_id": "A-1001"}),
                    ToolCall("c2", "查订单", {"order_id": "A-9999"}),
                ]
            ),
        ]
    )
    fail_fast_agent = ReliableAgent(
        fail_fast_model,
        runner,
        ReliabilityConfig(partial_policy="fail_fast"),
    )
    fail_fast_result = fail_fast_agent.run("查 A-1001 和 A-9999")
    ensure(
        fail_fast_result.stop_reason == "tool_failed",
        f"fail_fast 策略下应当整轮失败，实际 {fail_fast_result.stop_reason}",
    )
    print(f"  fail_fast 策略 → stop_reason={fail_fast_result.stop_reason}")


# ---------------------------------------------------------------- 6. 降级


def check_degradation_ladder() -> None:
    """降级：备用工具 → 旧缓存 → 跳过 → 明确告知用户。"""
    section("降级阶梯：每级的用户可见度不同")

    state = {"primary": 0, "backup": 0}

    def 主工具(key: str) -> str:
        """主数据源。

        Args:
            key: 数据键
        """
        state["primary"] += 1
        raise BusinessError("主数据源不可用")

    def 备用工具(key: str) -> str:
        """备用数据源。

        Args:
            key: 数据键
        """
        state["backup"] += 1
        return f"备用源返回 {key} 的数据"

    # 第一级：备用工具顶上
    runner = ReliableRunner(
        tools={"主工具": tool(主工具), "备用工具": tool(备用工具)},
        policy=RetryPolicy(max_attempts=1),
        fallbacks={"主工具": ("备用工具",)},
        sleep=lambda _: None,
    )
    outcome = runner.invoke(ToolCall("c1", "主工具", {"key": "K-1"}))
    ensure(outcome.ok, f"主工具失败后应当自动改用备用工具，实际 {outcome.kind}")
    ensure(outcome.degraded, "降级结果必须被标记")
    ensure(
        state["backup"] == 1,
        f"备用工具应当被调用 1 次，实际 {state['backup']} 次",
    )
    ensure(
        "[降级：" in outcome.content,
        f"降级必须对模型可见，实际 {outcome.content!r}",
    )
    print(f"  备用工具：{outcome.content}")

    # 第二级：备用也失败 → 用旧缓存
    def 坏备用(key: str) -> str:
        """也会失败的备用数据源。

        Args:
            key: 数据键
        """
        raise TransientError("备用源也挂了")

    cache_runner = ReliableRunner(
        tools={"主工具": tool(主工具), "备用工具": tool(坏备用)},
        policy=RetryPolicy(max_attempts=1),
        fallbacks={"主工具": ("备用工具",)},
        stale_cache={"主工具": "上次成功拿到的数据"},
        sleep=lambda _: None,
    )
    cached = cache_runner.invoke(ToolCall("c1", "主工具", {"key": "K-1"}))
    ensure(cached.ok, "旧缓存降级应当算成功（有数据可用）")
    ensure("[旧缓存" in cached.content, f"旧缓存降级必须被标记，实际 {cached.content!r}")
    print(f"  旧缓存：{cached.content}")

    # 第三级：跳过该步骤
    skip_runner = ReliableRunner(
        tools={"主工具": tool(主工具)},
        policy=RetryPolicy(max_attempts=1),
        fallbacks={"主工具": (SKIP,)},
        sleep=lambda _: None,
    )
    skipped = skip_runner.invoke(ToolCall("c1", "主工具", {"key": "K-1"}))
    ensure(not skipped.ok, "跳过后这一步没有结果")
    ensure("[已跳过" in skipped.content, f"跳过必须被标记，实际 {skipped.content!r}")
    print(f"  跳过：{skipped.content}")

    # 第四级：明确告知用户
    tell_runner = ReliableRunner(
        tools={"主工具": tool(主工具)},
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    told = tell_runner.invoke(ToolCall("c1", "主工具", {"key": "K-1"}))
    ensure(not told.ok, "没有降级方案时应当判为失败")
    ensure(
        "告知用户" in told.content,
        f"最后一级必须明确要求告知用户，实际 {told.content!r}",
    )
    print(f"  告知用户：{told.content}")


# ---------------------------------------------------------------- 7. 熔断


def check_circuit_breaker() -> None:
    """熔断：连续失败到阈值后，不再打后端。"""
    section("熔断：连续失败后直接拒绝")

    state = {"calls": 0}

    def 查库存(sku: str) -> str:
        """按 SKU 查询库存余量。

        Args:
            sku: 商品 SKU
        """
        state["calls"] += 1
        raise TransientError("上游 503")

    runner = ReliableRunner(
        tools={"查库存": tool(查库存)},
        policy=RetryPolicy(max_attempts=1),
        breaker=CircuitBreaker(threshold=3, reset_after=60.0),
        sleep=lambda _: None,
    )
    outcomes = [
        runner.invoke(ToolCall(f"c{i}", "查库存", {"sku": "SKU-1"})) for i in range(5)
    ]

    ensure(
        state["calls"] == 3,
        f"熔断打开后不该再打后端，期望后端收到 3 次请求，实际 {state['calls']} 次",
    )
    ensure(
        runner.breaker.state == "open",
        f"连续失败到阈值应当打开熔断，实际 {runner.breaker.state}",
    )
    ensure(
        outcomes[2].kind == "transient",
        f"第 3 次还应当真的打后端并失败，实际 {outcomes[2].kind}",
    )
    ensure(
        outcomes[3].kind == "circuit" and outcomes[4].kind == "circuit",
        f"第 4、5 次应当被熔断直接拒绝，实际 {outcomes[3].kind}、{outcomes[4].kind}",
    )
    ensure(
        "熔断" in outcomes[3].content,
        f"熔断拒绝必须对模型说明白，实际 {outcomes[3].content!r}",
    )
    print(f"  后端被打 {state['calls']} 次，之后 2 次被熔断直接拒绝")

    # 业务失败不该触发熔断——后端是好的
    business_breaker = CircuitBreaker(threshold=2)
    business_runner = ReliableRunner(
        tools={"查订单": 查订单},
        policy=RetryPolicy(max_attempts=1),
        breaker=business_breaker,
        sleep=lambda _: None,
    )
    for index in range(3):
        business_runner.invoke(ToolCall(f"c{index}", "查订单", {"order_id": "A-9999"}))
    ensure(
        business_breaker.state == "closed",
        f"业务失败说明后端是健康的，不该熔断，实际 {business_breaker.state}",
    )
    print("  业务失败 3 次 → 熔断仍为 closed（后端本身没问题）")


# ---------------------------------------------------------------- 8. 退避与抖动


def check_backoff_jitter() -> None:
    """指数退避 + 有界抖动：间隔递增，但每次都不一样。"""
    section("退避：递增 + 抖动")

    policy = RetryPolicy(max_attempts=4, base_delay=0.1, max_delay=10.0, jitter_ratio=0.5)
    rng = random.Random(20261118)
    bases = [policy.base_delay_for(index) for index in range(1, 5)]
    delays = [policy.delay_for(index, rng) for index in range(1, 5)]

    ensure(
        all(bases[index + 1] > bases[index] for index in range(len(bases) - 1)),
        f"基线应当等比递增，实际 {bases}",
    )
    ensure(
        all(delays[index + 1] > delays[index] for index in range(len(delays) - 1)),
        f"实际等待应当严格递增，实际 {delays}",
    )
    ensure(
        len({round(value, 9) for value in delays}) == len(delays),
        f"每次等待都应当带上不同的抖动，实际 {delays}",
    )
    ensure(
        all(
            base <= value <= base * (1 + policy.jitter_ratio) + 1e-9
            for base, value in zip(bases, delays)
        ),
        f"抖动幅度应当受限于基线的一半，实际 {list(zip(bases, delays))}",
    )
    ensure(
        policy.base_delay_for(20) == policy.max_delay,
        f"等待应当被上限截断，实际 {policy.base_delay_for(20)}",
    )
    print(f"  基线：{[round(v, 4) for v in bases]}")
    print(f"  实际：{[round(v, 4) for v in delays]}（含抖动）")

    # 为什么固定间隔会在服务恢复时把后端打垮
    fixed = _simulate_recovery_herd(jittered=False)
    spread = _simulate_recovery_herd(jittered=True)
    ensure(
        fixed["max_bucket"] == 20,
        f"固定间隔时所有客户端应当在同一时刻醒来，期望峰值 20，实际 {fixed['max_bucket']}",
    )
    ensure(
        spread["max_bucket"] < fixed["max_bucket"],
        f"带抖动应当把重试摊开，期望峰值小于 {fixed['max_bucket']}，实际 {spread['max_bucket']}",
    )
    print(
        f"  服务恢复瞬间的峰值并发：固定间隔 {fixed['max_bucket']}，带抖动 {spread['max_bucket']}"
    )


def _simulate_recovery_herd(
    *,
    clients: int = 20,
    attempts: int = 5,
    base: float = 0.2,
    jitter_ratio: float = 0.5,
    jittered: bool,
    seed: int = 20261118,
) -> dict[str, int]:
    """模拟服务恢复瞬间的重试洪峰。

    返回同一时刻（0.02 秒一个桶）打向后端的请求数峰值——固定间隔时所有
    客户端会在同一秒醒来，抖动时请求被摊开到一小段时间里。
    """
    rng = random.Random(seed)
    times: list[float] = []
    for _ in range(clients):
        moment = 0.0
        for _ in range(attempts):
            moment += base * (1 + jitter_ratio * rng.random()) if jittered else base
            times.append(moment)
    buckets: dict[int, int] = {}
    for moment in times:
        key = int(round(moment / 0.02))
        buckets[key] = buckets.get(key, 0) + 1
    return {"max_bucket": max(buckets.values()), "buckets": len(buckets)}


# ---------------------------------------------------------------- 9. 措辞


def check_feedback_wording() -> None:
    """坑：错误回填的措辞，决定模型能不能自我纠正。"""
    section("坑：措辞决定模型能不能自我纠正")

    tools = {"查订单": 查订单, "查物流": 查物流}
    primary = ToolCall("c1", "查订单", {"order_id": "A-9999"})
    secondary = ToolCall("c2", "查物流", {"shipment_id": "S-1001"})

    # 对照组：原样回填异常（很多框架的默认做法）
    raw_runner = ReliableRunner(
        tools=tools,
        policy=RetryPolicy(max_attempts=1),
        feedback=lambda kind, exc: f"{type(exc).__name__}: {exc}",
        sleep=lambda _: None,
    )
    raw_model = SwitchingModel(primary=primary, secondary=secondary)
    raw_agent = ReliableAgent(raw_model, raw_runner, ReliabilityConfig(max_iterations=4))
    raw_result = raw_agent.run("查 A-9999")
    raw_primary_calls = sum(
        1
        for message in raw_result.messages
        if message.role == "assistant"
        for call in message.tool_calls
        if call.name == "查订单"
    )
    ensure(
        raw_result.stop_reason == "max_iterations",
        f"原样回填时模型应当反复重试到跑满轮次，实际 {raw_result.stop_reason}",
    )
    ensure(
        raw_primary_calls >= 3,
        f"原样回填时模型应当拿同一组参数反复重试，实际调用了 {raw_primary_calls} 次",
    )

    # 实验组：改写措辞，明确告诉模型"重试没用"
    good_runner = ReliableRunner(
        tools=tools,
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    good_model = SwitchingModel(primary=primary, secondary=secondary)
    good_agent = ReliableAgent(good_model, good_runner, ReliabilityConfig(max_iterations=4))
    good_result = good_agent.run("查 A-9999")
    good_primary_calls = sum(
        1
        for message in good_result.messages
        if message.role == "assistant"
        for call in message.tool_calls
        if call.name == "查订单"
    )
    ensure(
        good_result.stop_reason == "completed",
        f"改写措辞后模型应当换条路并收尾，实际 {good_result.stop_reason}",
    )
    ensure(
        good_primary_calls == 1,
        f"改写措辞后主工具只应当被调用 1 次，实际 {good_primary_calls} 次",
    )
    print(f"  原样回填 → 同一工具被调 {raw_primary_calls} 次，跑满轮次")
    print(f"  改写措辞 → 同一工具被调 {good_primary_calls} 次，模型换了工具并收尾")


# ---------------------------------------------------------------- 10. 成本


def check_retry_cost() -> None:
    """重试的成本：工具级重试不惊动模型，模型级重试每次都多花一次模型调用。"""
    section("重试的成本：钱和上下文都按次涨")

    state = {"calls": 0}

    def 查库存(sku: str) -> str:
        """按 SKU 查询库存余量。

        Args:
            sku: 商品 SKU
        """
        state["calls"] += 1
        if state["calls"] <= 2:
            raise TransientError("连接被重置")
        return "有货"

    # 情况一：运行时自己重试（透明重试）——模型只被叫了一次要工具
    runner = ReliableRunner(
        tools={"查库存": tool(查库存)},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "查库存", {"sku": "SKU-1"})]),
            ModelReply(content="库存有货"),
        ]
    )
    agent = ReliableAgent(model, runner)
    result = agent.run("查库存")

    ensure(
        len(model.calls) == 2,
        f"透明重试只应当花 2 次模型调用（要工具 + 收尾），实际 {len(model.calls)} 次",
    )
    ensure(
        state["calls"] == 3,
        f"工具实际执行 3 次，实际 {state['calls']} 次",
    )
    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(
        len(tool_messages) == 1,
        f"透明重试只应当回填一条工具消息，实际 {len(tool_messages)} 条",
    )
    print(
        f"  透明重试：模型调用 {len(model.calls)} 次，工具执行 {state['calls']} 次，"
        f"上下文只增长 1 条工具消息"
    )

    # 情况二：不透明重试，错误回填给模型让它自己再试——每次都是一次模型调用
    always_fail_state = {"calls": 0}

    def 查库存_总失败(sku: str) -> str:
        """总是失败的库存查询。

        Args:
            sku: 商品 SKU
        """
        always_fail_state["calls"] += 1
        raise TransientError("连接被重置")

    retry_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "查库存_总失败", {"sku": "SKU-1"})]),
            ModelReply(tool_calls=[ToolCall("c2", "查库存_总失败", {"sku": "SKU-1"})]),
            ModelReply(content="库存服务暂时不可用。"),
        ]
    )
    retry_runner = ReliableRunner(
        tools={"查库存_总失败": tool(查库存_总失败)},
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    retry_agent = ReliableAgent(retry_model, retry_runner)
    retry_result = retry_agent.run("查库存")

    ensure(
        len(retry_model.calls) == 3,
        f"模型级重试应当多花模型调用，实际 {len(retry_model.calls)} 次",
    )
    ensure(
        always_fail_state["calls"] == 2,
        f"模型级重试里工具被执行 2 次，实际 {always_fail_state['calls']} 次",
    )
    ensure(
        len(retry_result.messages) > len(result.messages),
        f"模型级重试会撑大上下文，实际 {len(retry_result.messages)} 条 vs {len(result.messages)} 条",
    )
    print(
        f"  模型级重试：模型调用 {len(retry_model.calls)} 次，上下文 {len(retry_result.messages)} 条消息"
        f"（透明重试只要 {len(result.messages)} 条）"
    )


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型驱动一次带重试的工具调用，只校验形状。"""
    banner("第 10 章 · 真实模型路径")
    from ..labkit import real_model

    state = {"calls": 0}

    def 查天气(city: str) -> str:
        """查询城市天气。

        Args:
            city: 城市名
        """
        state["calls"] += 1
        if state["calls"] <= 2:
            raise TransientError("上游限流（429）")
        return f"{city} 晴，25℃"

    runner = ReliableRunner(
        tools={"查天气": tool(查天气)},
        policy=RetryPolicy(max_attempts=3),
        sleep=lambda _: None,
    )
    agent = ReliableAgent(real_model(), runner)
    result = agent.run("北京天气怎么样？")

    print(f"  结束原因：{result.stop_reason}")
    print(f"  最终回答：{result.final_answer[:200]!r}")
    ensure(
        result.stop_reason in {"completed", "max_iterations"},
        f"真实模型路径应当正常收尾，实际 {result.stop_reason}",
    )
    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "工具消息必须带 tool_call_id",
    )


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 10 章实验。"""
    banner("第 10 章 · 可靠性：重试、超时、幂等与降级")
    check_failure_classification()
    check_retry_on_transient()
    check_deterministic_not_retried()
    check_business_not_retried()
    check_timeout_layers()
    check_idempotency()
    check_partial_failure()
    check_degradation_ladder()
    check_circuit_breaker()
    check_backoff_jitter()
    check_feedback_wording()
    check_retry_cost()
    rule("先分类再动手：只有瞬时故障才重试；重试必须带幂等键；超时兜住循环；降级要看得见")
    print("\n✓ 第 10 章通过")
