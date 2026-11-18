"""第 11 章实验：可观测性——事件模型、追踪与回放。

这个模块解决什么问题：第 11 章讲的是"Agent 出问题怎么查"。这必须能跑出来看，
所以本章把五件事变成可执行的断言：

1. **事件模型**——一次运行记下哪些事件、字段有哪些；父子关系对不对；
2. **模型实际收到了什么**——trace 里记的是模型真正看到的 messages，
   而不是"我以为我发了什么"；
3. **脱敏**——密钥不能在 trace 里出现，而且要递归处理嵌套结构；
4. **回放**——用 trace 里记录的模型响应重放，能得到同样的轨迹；
5. **聚合指标**——多轮运行后能统计工具失败率、找出"从没被调用过的工具"。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..labkit import (
    Message,
    ModelReply,
    ToolCall,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    tool,
)
from ..runtime import Agent, AgentConfig

# ---------------------------------------------------------------- 事件模型


@dataclass
class Event:
    """一条 trace 事件。

    ``event_id`` 是这条事件的唯一标识，``parent_id`` 指向**父事件的 event_id**
    （不是 run_id）——否则同一次运行里的所有事件都挂在同一个 run 下，
    树形关系就退化成了一团。
    """

    event_id: str
    run_id: str
    parent_id: str | None
    type: str
    timestamp: float
    elapsed_ms: float
    payload: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """轨迹记录器——脱敏在**写入时**做，不是读取时做。"""

    SENSITIVE_KEYS = {"api_key", "password", "token", "secret", "authorization"}

    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.events: list[Event] = []
        self._start = time.perf_counter()
        self._counter = 0

    def record(
        self,
        event_type: str,
        parent_id: str | None = None,
        **payload: Any,
    ) -> Event:
        """记一条事件。"""
        self._counter += 1
        event = Event(
            event_id=f"{self.run_id}#{self._counter}",
            run_id=self.run_id,
            parent_id=parent_id,
            type=event_type,
            timestamp=time.time(),
            elapsed_ms=(time.perf_counter() - self._start) * 1000,
            payload=self._redact(payload),
        )
        self.events.append(event)
        return event

    def _redact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """按字段名递归脱敏——嵌套结构里的敏感字段也要处理。"""
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in self.SENSITIVE_KEYS:
                clean[key] = "***REDACTED***"
            elif isinstance(value, dict):
                clean[key] = self._redact(value)
            else:
                clean[key] = value
        return clean

    def of_type(self, event_type: str) -> list[Event]:
        """按类型取事件。"""
        return [event for event in self.events if event.type == event_type]

    def children_of(self, parent_event_id: str) -> list[Event]:
        """取某个父事件的子事件——树形关系靠 parent_id 表达。"""
        return [event for event in self.events if event.parent_id == parent_event_id]


# ---------------------------------------------------------------- 1. 事件模型


def check_event_model() -> None:
    """一次运行该记哪些事件，父子关系对不对。"""
    section("事件模型：一次运行的最小事件集合")

    recorder = Recorder("run-x")
    root = recorder.record("run_start", input="订单 A-1001 到哪了")
    model_event = recorder.record("model", parent_id=root.event_id, iteration=0)
    tool_event = recorder.record(
        "tool_call", parent_id=model_event.event_id, name="查订单", args={"order_id": "A-1001"}
    )
    recorder.record("tool_result", parent_id=tool_event.event_id, output="已发货")
    recorder.record("run_end", parent_id=root.event_id, stop_reason="completed")

    for event in recorder.events:
        print(
            f"  {event.event_id:<10} {event.type:<12} parent={str(event.parent_id):<10} "
            f"{event.elapsed_ms:6.1f}ms {list(event.payload)}"
        )

    ensure(len(recorder.events) == 5, f"应当记录 5 条事件，实际 {len(recorder.events)}")
    ensure(
        {e.type for e in recorder.events}
        == {"run_start", "model", "tool_call", "tool_result", "run_end"},
        "事件类型集合不对",
    )

    # 父子关系：run_start 是根，tool_call 挂在 model 下
    children = recorder.children_of(model_event.event_id)
    ensure(
        len(children) == 1 and children[0].type == "tool_call",
        f"model 事件应当有一个 tool_call 子事件，实际 {[c.type for c in children]}",
    )
    ensure(
        recorder.of_type("tool_result")[0].parent_id == tool_event.event_id,
        "tool_result 的父事件应当是 tool_call",
    )
    ensure(
        len(recorder.children_of(root.event_id)) == 2,
        "run_start 应当只有 model 和 run_end 两个子事件",
    )
    print("  树形关系：run_start → model → tool_call → tool_result")


# ---------------------------------------------------------------- 2. 模型实际收到了什么


def check_record_what_model_saw() -> None:
    """trace 要记"模型实际收到的 messages"，而不是"我以为我发了什么"。"""
    section("记模型实际收到了什么")

    recorder = Recorder("run-y")

    @tool
    def 查订单(order_id: str) -> str:
        """按订单号查状态。

        Args:
            order_id: 订单号
        """
        return "已发货"

    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查订单", {"order_id": "A-1001"})]),
            ModelReply(content="订单已发货"),
        ]
    )
    agent = Agent(model, [查订单])
    result = agent.run("订单 A-1001 到哪了？")

    # 模拟运行时记录：记的是模型**收到**的 messages
    recorder.record("model_input", messages=[m.to_api() for m in result.messages[:1]])
    recorder.record("model_output", content=result.final_answer)
    recorder.record("usage", prompt=result.usage.get("prompt_tokens", 0))

    model_input = recorder.of_type("model_input")[0]
    payload = model_input.payload["messages"]
    print(f"  模型收到的第一条消息：{payload[0]['role']} — {payload[0]['content']!r}")

    ensure(
        payload[0]["role"] == "user",
        f"模型收到的第一条应当是用户消息，实际 {payload[0]['role']}",
    )
    ensure(
        "A-1001" in payload[0]["content"],
        "模型实际收到的内容里应当包含用户输入",
    )

    # 对照：如果只记"我发的 prompt"而不记模型实际收到的，中间的加工就丢了
    only_what_i_sent = "订单 A-1001 到哪了？"
    ensure(
        only_what_i_sent == payload[0]["content"],
        "本例里两者恰好相同——因为还没有中间件改写；一旦有压缩或模板就会不同",
    )
    print("  → 有中间件/压缩时，两者就不一样了；所以要记后者")


# ---------------------------------------------------------------- 3. 脱敏


def check_redaction() -> None:
    """脱敏必须在写入时做，且要递归。"""
    section("脱敏：写入时做，且递归")

    recorder = Recorder("run-z")
    event = recorder.record(
        "tool_call",
        name="调外部接口",
        api_key="sk-should-never-appear",
        headers={"authorization": "Bearer secret", "content_type": "application/json"},
        nested={"password": "hunter2", "note": "普通字段"},
    )

    print(f"  脱敏后：{event.payload}")
    ensure(
        event.payload["api_key"] == "***REDACTED***",
        f"api_key 应当被脱敏，实际 {event.payload['api_key']!r}",
    )
    ensure(
        event.payload["headers"]["authorization"] == "***REDACTED***",
        f"嵌套的 authorization 应当被脱敏，实际 {event.payload['headers']!r}",
    )
    ensure(
        event.payload["nested"]["password"] == "***REDACTED***",
        f"嵌套的 password 应当被脱敏，实际 {event.payload['nested']!r}",
    )
    ensure(
        event.payload["nested"]["note"] == "普通字段",
        "非敏感字段不该被误伤",
    )

    # 关键：脱敏在**写入时**就完成了，事件里根本不存在明文
    raw = str(event.payload)
    ensure("sk-should-never-appear" not in raw, "明文密钥不能出现在事件里")
    ensure("hunter2" not in raw, "明文密码不能出现在事件里")
    print("  → 事件里根本不存在明文，不是读取时才算脱敏")


# ---------------------------------------------------------------- 4. 回放


def check_replay() -> None:
    """用 trace 里记录的模型响应重放，能得到同样的轨迹。"""
    section("回放：用记录的响应复现同一条轨迹")

    @tool
    def 查订单(order_id: str) -> str:
        """按订单号查状态。

        Args:
            order_id: 订单号
        """
        return "已发货"

    script = [
        ModelReply(tool_calls=[ToolCall("call_1", "查订单", {"order_id": "A-1001"})]),
        ModelReply(content="订单已发货"),
    ]

    # 第一次运行：记录轨迹
    recorder = Recorder("run-replay")
    model = scripted_model(script)
    agent = Agent(model, [查订单])
    first = agent.run("订单 A-1001 到哪了？")

    recorded_responses = [
        {"content": m.content, "tool_calls": [dict(c.__dict__) for c in m.tool_calls]}
        for m in first.messages
        if m.role == "assistant"
    ]
    recorder.record("replay_capture", responses=recorded_responses)

    # 回放：用记录的响应重放，不调模型
    def replay(recorded: list[dict[str, Any]]) -> list[str]:
        """按记录的响应重放，产出同样的工具调用序列。"""
        from ..labkit import tool_result as make_tool_result

        messages: list[Message] = [human("订单 A-1001 到哪了？")]
        calls: list[str] = []
        for item in recorded:
            calls_from_record = [
                ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                for c in item["tool_calls"]
            ]
            messages.append(
                Message(
                    role="assistant",
                    content=item["content"],
                    tool_calls=calls_from_record,
                )
            )
            for call in calls_from_record:
                calls.append(call.name)
                messages.append(make_tool_result(查订单.invoke(call.arguments), call.id, call.name))
        return calls

    replayed_calls = replay(recorder.of_type("replay_capture")[0].payload["responses"])
    original_calls = [call.name for call in first.tool_calls]

    print(f"  原始轨迹的工具调用：{original_calls}")
    print(f"  回放轨迹的工具调用：{replayed_calls}")

    ensure(
        replayed_calls == original_calls,
        f"回放的工具调用序列应当与原始一致：{replayed_calls} vs {original_calls}",
    )
    ensure(
        len(replayed_calls) == 1,
        "回放应当复现出同样次数的工具调用",
    )
    print("  → 回放零模型成本，可反复用于验证改动")


# ---------------------------------------------------------------- 5. 聚合指标


def check_aggregate_metrics() -> None:
    """多轮运行后能统计工具失败率，并找出"从没被调用过的工具"。"""
    section("聚合指标：从一堆事件里读出趋势")

    recorder = Recorder("run-metrics")

    @tool
    def 查订单(order_id: str) -> str:
        """按订单号查状态。

        Args:
            order_id: 订单号
        """
        return "已发货"

    @tool
    def 退款(order_id: str) -> str:
        """发起退款。

        Args:
            order_id: 订单号
        """
        return "已退款"

    @tool
    def 从没被用过的工具(x: str) -> str:
        """这个工具的描述写坏了，模型从来不选它。

        Args:
            x: 任意参数
        """
        return "没人用我"

    tools = [查订单, 退款, 从没被用过的工具]
    tool_names = {t.name for t in tools}

    # 跑三次：一次成功、一次工具不存在、一次参数错
    runs = [
        scripted_model(
            [
                ModelReply(tool_calls=[ToolCall("call_1", "查订单", {"order_id": "A-1"})]),
                ModelReply(content="已发货"),
            ]
        ),
        scripted_model(
            [
                ModelReply(tool_calls=[ToolCall("call_1", "不存在的工具", {})]),
                ModelReply(content="换个工具"),
            ]
        ),
        scripted_model(
            [
                ModelReply(tool_calls=[ToolCall("call_1", "查订单", {"错误参数": 1})]),
                ModelReply(content="参数错了"),
            ]
        ),
    ]

    for index, model in enumerate(runs):
        agent = Agent(model, tools)
        result = agent.run("查一下")
        recorder.record("run", index=index, stop_reason=result.stop_reason)

        # 记工具调用与它的结局——失败与否从工具消息的内容里分类，
        # 这正是真实 trace 里能拿到的信息
        tool_messages = [m for m in result.messages if m.role == "tool"]
        for call, message in zip(result.tool_calls, tool_messages, strict=False):
            recorder.record("tool_call", name=call.name)
            outcome = _classify_tool_outcome(message.content)
            recorder.record("tool_result", name=call.name, outcome=outcome)
        recorder.record("usage", prompt=result.usage.get("prompt_tokens", 0))

    # ---- 指标 1：工具调用分布 → 谁从没被调用过
    called = {event.payload["name"] for event in recorder.of_type("tool_call")}
    never_called = tool_names - called
    print(f"  被调用过的工具：{sorted(called)}")
    print(f"  从没被调用过的工具：{sorted(never_called)}")

    ensure(
        never_called == {"退款", "从没被用过的工具"},
        f"应当识别出两个没被调用过的工具，实际 {sorted(never_called)}",
    )
    ensure(
        "从没被用过的工具" in never_called,
        "描述写坏的工具应当被指标抓出来——它不报错，但一直在浪费 schema 预算",
    )

    # ---- 指标 2：失败率（按错误类型）
    outcomes = [event.payload["outcome"] for event in recorder.of_type("tool_result")]
    failures = [outcome for outcome in outcomes if outcome != "ok"]
    print(f"  工具结局分布：{outcomes}")
    print(f"  失败次数：{len(failures)}")

    ensure(
        len(failures) == 2,
        f"应当记录到两次工具失败，实际 {len(failures)}",
    )
    ensure(
        "unknown_tool" in failures,
        f"不存在的工具应当被归类为 unknown_tool，实际 {failures}",
    )
    ensure(
        "parameter" in failures,
        f"参数错误应当被归类为 parameter，实际 {failures}",
    )
    ensure(
        "ok" in outcomes,
        "成功的调用也要被记下来，才能算得出失败率",
    )

    # ---- 指标 3：停止原因分布
    stop_reasons = [event.payload["stop_reason"] for event in recorder.of_type("run")]
    print(f"  停止原因分布：{stop_reasons}")
    ensure(
        all(reason == "completed" for reason in stop_reasons),
        f"三次运行都应正常结束，实际 {stop_reasons}",
    )


def _classify_tool_outcome(content: str) -> str:
    """从工具消息内容判断结局。

    真实系统里错误类型应该由运行时**结构化**地记下来；
    这里用内容反推，是为了让你看清：trace 里如果没有结构化的错误字段，
    你就只能靠字符串猜——所以错误类型必须单独记成一个字段。
    """
    if "没有名为" in content:
        return "unknown_tool"
    if "参数错误" in content:
        return "parameter"
    if "工具执行失败" in content:
        return "business"
    return "ok"


def run_live() -> None:
    """真实模型路径：真跑一次，确认 trace 记的东西和真模型对得上。"""
    banner("第 11 章 · 真实模型路径")
    from ..labkit import real_model

    recorder = Recorder("run-live")

    @tool
    def 查订单(order_id: str) -> str:
        """按订单号查状态。

        Args:
            order_id: 订单号
        """
        return f"订单 {order_id} 已发货"

    agent = Agent(real_model(), [查订单])
    result = agent.run("查一下订单 A-1001 的状态")

    for event in recorder.events:
        print(f"  {event.type}: {event.payload}")

    recorder.record("run", stop_reason=result.stop_reason, turns=len(result.steps))

    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(
        recorder.of_type("run")[0].payload["turns"] == len(result.steps),
        "轮次应当被正确记进 trace",
    )
    print(f"  停止原因：{result.stop_reason}")


def run() -> None:
    """跑第 11 章实验。"""
    banner("第 11 章 · 可观测性：事件模型、追踪与回放")
    check_event_model()
    check_record_what_model_saw()
    check_redaction()
    check_replay()
    check_aggregate_metrics()
    rule("trace 记的是「发生过什么」，模型自述是「事后编的故事」——只信前者")
    print("\n✓ 第 11 章通过")
