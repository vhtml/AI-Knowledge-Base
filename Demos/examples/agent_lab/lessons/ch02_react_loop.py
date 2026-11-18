"""第 2 章实验：ReAct 循环的手写实现。

这个模块解决什么问题：第 1 章给出了 Agent 的"最小内核"那 12 行伪代码，
但伪代码不会告诉你循环在真实运行时会在哪儿崩。本章把 ReAct 循环亲手写一遍，
并把最容易出的四类故障变成可执行的断言：

1. **正常路径**——模型要工具 → 执行 → 回填 → 模型给出答案，
   消息序列必须是 ``user → assistant(tool_calls) → tool → assistant``。
2. **回填断裂**——工具结果少了 ``tool_call_id``，模型就不知道这段观察
   是哪次调用的结果；这里验证这种断裂**能被检出**，而不是带病继续跑。
3. **并行调用**——一轮返回多个 tool_calls 时，每条结果要各自带对的 id。
4. **不收敛**——模型原地打转、或者一直要工具，都要有明确的退出理由。

最后把手写循环和参考运行时 ``runtime.Agent`` 在同一条脚本下对照一次，
确认"自己写的"和"给好的"行为等价——这是后面所有章节的基线。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

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
from ..runtime import Agent, AgentConfig

# ---------------------------------------------------------------- 工具


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    orders = {"A-1001": "已发货", "A-1002": "待付款", "A-1003": "已取消"}
    return orders.get(order_id, "未找到")


@tool
def 查询天气(city: str) -> str:
    """查询城市当前天气。

    Args:
        city: 城市名
    """
    weather = {"北京": "晴 26℃", "上海": "多云 29℃"}
    return weather.get(city, "暂无数据")


@tool
def 查询库存(sku: str) -> str:
    """查询商品库存。

    Args:
        sku: 商品编码
    """
    if sku == "BROKEN":
        # 故意制造业务故障：这类错误模型自己修不好，只能换条路
        raise RuntimeError("库存服务连接超时")
    return {"SKU-1": "有货 12 件"}.get(sku, "无货")


# ---------------------------------------------------------------- 手写循环


@dataclass
class StopSignal:
    """外部停止信号。

    循环每轮开头检查一次：一旦被触发就停下，并如实报告原因。
    第 14 章讲"人在环路"时，人工中止走的就是这条通道。
    """

    reason: str = ""
    triggered: bool = False

    def trip(self, reason: str) -> None:
        """触发停止信号。"""
        self.reason = reason
        self.triggered = True


@dataclass
class LoopConfig:
    """手写循环的配置。

    每个参数都对应一个真实故障面：``max_iterations`` 兜住"烧钱"，
    ``repeat_limit`` 兜住"原地打转"，``system_prompt`` 只是开场白。
    """

    max_iterations: int = 8
    repeat_limit: int = 2
    system_prompt: str = ""


@dataclass
class LoopResult:
    """手写循环的一次运行结果。"""

    messages: list[Message] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "completed"
    stop_detail: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0


def _call_key(call: ToolCall) -> str:
    """调用指纹：工具名 + 排序后的参数。

    同名同参的调用重复出现，基本可以认定模型没从上一次结果里学到东西。
    """
    return f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"


def _execute_tool(tool_map: dict[str, Tool], call: ToolCall) -> str:
    """执行一次工具调用，返回要回填给模型的文本。

    错误也变成文本回填——这是 Agent 和普通函数最本质的差别之一：
    模型必须能看见自己犯的错，才有机会纠正。
    但回填的措辞要带上"换条路"的暗示，否则模型只会原样重试。
    """
    target = tool_map.get(call.name)
    if target is None:
        known = "、".join(sorted(tool_map)) or "（无）"
        return f"错误：没有名为 {call.name} 的工具。可用工具：{known}。请换一个工具重试。"
    try:
        return target.invoke(call.arguments)
    except Exception as exc:  # noqa: BLE001 - 工具异常要转成模型能读懂的话
        return (
            f"工具执行失败：{type(exc).__name__}: {exc}。"
            "这个错误通常不是参数问题，请调整参数或换用其他工具，不要重复同样的调用。"
        )


def react_loop(
    model: ChatModel,
    tools: Sequence[Tool],
    user_input: str,
    config: LoopConfig | None = None,
    history: Sequence[Message] = (),
    signal: StopSignal | None = None,
    backfill_call_id: bool = True,
) -> LoopResult:
    """手写一个 ReAct 循环。

    这一版刻意写得比参考运行时更直白，好让每一步都看得见：
    问模型 → 要工具就执行并回填 → 不要工具就收工 → 没到上限就再来一轮。

    停止理由有四种：``completed``（模型自己收工）、``max_iterations``（轮次上限）、
    ``repeated_tool_call``（原地打转）、``stopped``（外部停止信号）。

    ``backfill_call_id`` 留成开关，是为了把"漏掉 tool_call_id"这个真实存在的
    低级 bug 演示出来——它不是一个假想的错误。
    """
    cfg = config or LoopConfig()
    stop = signal or StopSignal()
    tool_map = {t.name: t for t in tools}
    tool_list = list(tools)

    messages: list[Message] = []
    if cfg.system_prompt:
        messages.append(Message(role="system", content=cfg.system_prompt))
    messages.extend(history)
    messages.append(human(user_input))

    calls: list[ToolCall] = []
    seen: dict[str, int] = {}

    for iteration in range(cfg.max_iterations):
        # 停止信号要在"问模型之前"检查：已经决定停了就别再花钱问一次
        if stop.triggered:
            return LoopResult(
                messages=messages,
                stop_reason="stopped",
                stop_detail=stop.reason,
                tool_calls=calls,
                iterations=iteration,
            )

        reply = model.complete(messages, tools=tool_list or None)
        messages.append(reply.to_message())

        if not reply.tool_calls:
            return LoopResult(
                messages=messages,
                final_answer=reply.content,
                stop_reason="completed",
                tool_calls=calls,
                iterations=iteration + 1,
            )

        for call in reply.tool_calls:
            calls.append(call)
            key = _call_key(call)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= cfg.repeat_limit:
                # 原地打转：同一个调用反复出现，继续下去只是烧钱
                return LoopResult(
                    messages=messages,
                    stop_reason="repeated_tool_call",
                    stop_detail=key,
                    tool_calls=calls,
                    iterations=iteration + 1,
                )

            content = _execute_tool(tool_map, call)
            if backfill_call_id:
                messages.append(tool_result(content, tool_call_id=call.id, name=call.name))
            else:
                # 故意造一个坏消息：少了 tool_call_id，模型对不上这是谁的结果
                messages.append(Message(role="tool", content=content, name=call.name))

    return LoopResult(
        messages=messages,
        stop_reason="max_iterations",
        tool_calls=calls,
        iterations=cfg.max_iterations,
    )


# ---------------------------------------------------------------- 诊断工具


def pair_observations(messages: Sequence[Message]) -> tuple[dict[str, str], list[str]]:
    """把工具结果配对到它对应的调用上。

    返回 ``(配对成功的 {调用 id: 结果内容}, 无法配对的结果内容列表)``。

    模型只能靠 ``tool_call_id`` 把"这段观察"和"那次调用意图"对上。
    配不上的观察对它来说就是凭空多出来的一段文字——既不知道是谁的结果，
    也没法拿它去修正下一次的参数。
    """
    issued = [call.id for message in messages for call in message.tool_calls]
    paired: dict[str, str] = {}
    orphans: list[str] = []
    for message in messages:
        if message.role != "tool":
            continue
        if message.tool_call_id and message.tool_call_id in issued:
            paired[message.tool_call_id] = message.content
        else:
            orphans.append(message.content)
    return paired, orphans


def audit_tool_results(messages: Sequence[Message]) -> list[str]:
    """检查工具结果能不能和调用对上，返回问题列表（空列表表示没问题）。"""
    issued = {call.id for message in messages for call in message.tool_calls}
    problems: list[str] = []
    for message in messages:
        if message.role != "tool":
            continue
        if not message.tool_call_id:
            problems.append("工具结果缺少 tool_call_id：模型无法判断这段观察是哪次调用的结果")
        elif message.tool_call_id not in issued:
            problems.append(
                f"工具结果的 tool_call_id={message.tool_call_id!r} 不对应任何一次调用"
            )
    return problems


# ---------------------------------------------------------------- 1. 正常路径


def check_react_roundtrip() -> None:
    """一次完整的"调工具 → 拿结果 → 回答"，并核对消息序列。"""
    section("1. 正常路径：user → assistant(tool_calls) → tool → assistant")

    model = scripted_model(
        [
            ModelReply(
                content="用户问订单状态，先查订单 A-1001。",
                tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})],
            ),
            ModelReply(content="订单 A-1001 已发货。"),
        ]
    )
    result = react_loop(model, [查询订单], "A-1001 什么状态？")

    roles = [m.role for m in result.messages]
    ensure(
        roles == ["user", "assistant", "tool", "assistant"],
        f"消息序列应当是 user→assistant(tool_calls)→tool→assistant，实际 {roles}",
    )

    call = result.messages[1].tool_calls
    ensure(len(call) == 1, f"第二条消息应当带 1 个 tool_call，实际 {len(call)}")
    ensure(call[0].name == "查询订单", f"工具名应当是 查询订单，实际 {call[0].name!r}")
    ensure(
        result.messages[2].tool_call_id == call[0].id,
        f"工具结果的 tool_call_id 应当等于调用 id {call[0].id!r}，"
        f"实际 {result.messages[2].tool_call_id!r}",
    )
    ensure(
        result.messages[2].content == "已发货",
        f"工具结果应当是 '已发货'，实际 {result.messages[2].content!r}",
    )
    ensure(
        result.final_answer == "订单 A-1001 已发货。",
        f"最终回答不对：{result.final_answer!r}",
    )
    ensure(
        result.stop_reason == "completed",
        f"应当正常收工，实际 stop_reason={result.stop_reason!r}",
    )

    # 关键：第 2 轮提问时，模型必须已经看见了工具结果——否则它无从作答
    second_turn = model.calls[1]
    tool_seen = [m for m in second_turn if m.role == "tool"]
    ensure(
        len(tool_seen) == 1 and tool_seen[0].content == "已发货",
        f"模型第 2 轮应当看见一条内容为 '已发货' 的工具消息，实际 {[m.content for m in tool_seen]}",
    )

    print("  第 1 轮：Thought=先查订单 A-1001 → Action=查询订单")
    print("  回填   ：tool('已发货', tool_call_id='call_1')")
    print(f"  第 2 轮：最终回答 {result.final_answer!r}")
    print(f"  消息序列：{' → '.join(roles)}")


# ---------------------------------------------------------------- 2. 回填断裂


def check_tool_call_id_audit() -> None:
    """工具结果少了 tool_call_id，模型就配不上号——而且必须能被检出。"""
    section("2. 回填断裂：少了 tool_call_id 会发生什么")

    script = [
        ModelReply(
            content="先查订单 A-1001。",
            tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})],
        ),
        ModelReply(content="订单 A-1001 已发货。"),
    ]

    good = react_loop(scripted_model(script), [查询订单], "A-1001 什么状态？")
    problems = audit_tool_results(good.messages)
    paired, orphans = pair_observations(good.messages)
    ensure(not problems, f"正确回填不该有诊断信息，实际 {problems}")
    ensure(
        paired.get("call_1") == "已发货",
        f"结果应当配对到 call_1，实际配对结果 {paired}",
    )
    ensure(not orphans, f"正确回填不该有孤儿结果，实际 {orphans}")

    # 同一个脚本，只把回填里的 tool_call_id 去掉
    bad = react_loop(
        scripted_model(script),
        [查询订单],
        "A-1001 什么状态？",
        backfill_call_id=False,
    )
    bad_problems = audit_tool_results(bad.messages)
    ensure(
        len(bad_problems) == 1 and "tool_call_id" in bad_problems[0],
        f"漏掉 tool_call_id 应当被检出一次，实际检出 {bad_problems}",
    )

    bad_paired, bad_orphans = pair_observations(bad.messages)
    ensure(
        not bad_paired,
        f"缺 id 时不该有任何结果能配上号，实际配对到了 {bad_paired}",
    )
    ensure(
        len(bad_orphans) == 1,
        f"缺 id 时应当留下一条无法配对的观察，实际 {len(bad_orphans)} 条",
    )

    # 而且模型确实收到了这条"无主"的观察
    seen_by_model = [m for m in bad.messages if m.role == "tool"]
    ensure(
        len(seen_by_model) == 1 and seen_by_model[0].tool_call_id is None,
        "坏消息应当以缺失 tool_call_id 的形式回填给模型",
    )

    print("  正确回填 → 配对成功 call_1='已发货'，无孤儿结果")
    print(f"  漏掉 id  → 检出问题：{bad_problems[0]}")
    print("  模型视角：这段观察没有来源，既不知道是谁的结果，也无法据此修正参数")
    print("  （真实 API 会直接报错：role=tool 的消息必须带 tool_call_id）")


# ---------------------------------------------------------------- 3. 并行调用


def check_parallel_tool_calls() -> None:
    """一轮两个 tool_calls，要回填两条 tool 消息，id 各归各位。"""
    section("3. 并行调用：一轮两个 tool_calls，两条结果各自回填")

    model = scripted_model(
        [
            ModelReply(
                content="订单和天气要一起看。",
                tool_calls=[
                    ToolCall("call_a", "查询订单", {"order_id": "A-1001"}),
                    ToolCall("call_b", "查询天气", {"city": "北京"}),
                ],
            ),
            ModelReply(content="A-1001 已发货，北京晴 26℃。"),
        ]
    )
    result = react_loop(model, [查询订单, 查询天气], "A-1001 状态和北京天气")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(
        len(tool_messages) == 2,
        f"两个 tool_calls 应当回填两条工具消息，实际 {len(tool_messages)} 条",
    )
    ensure(
        [m.tool_call_id for m in tool_messages] == ["call_a", "call_b"],
        f"两条结果的 tool_call_id 应当分别是 call_a / call_b，实际 {[m.tool_call_id for m in tool_messages]}",
    )
    ensure(
        [m.content for m in tool_messages] == ["已发货", "晴 26℃"],
        f"结果内容应当各自对应自己的工具，实际 {[m.content for m in tool_messages]}",
    )
    ensure(
        len(result.messages[1].tool_calls) == 2,
        f"第一条助手消息应当带 2 个 tool_calls，实际 {len(result.messages[1].tool_calls)}",
    )

    # 回填必须发生在下一次提问之前，且两条都要在
    second_turn_tools = [m for m in model.calls[1] if m.role == "tool"]
    ensure(
        len(second_turn_tools) == 2,
        f"模型第 2 轮应当同时看见两条工具结果，实际 {len(second_turn_tools)} 条",
    )

    paired, orphans = pair_observations(result.messages)
    ensure(
        paired == {"call_a": "已发货", "call_b": "晴 26℃"},
        f"配对结果应当一一对应，实际 {paired}",
    )
    ensure(not orphans, f"不该有孤儿结果，实际 {orphans}")

    print("  一轮请求：call_a=查询订单(A-1001)，call_b=查询天气(北京)")
    print("  回填两条：tool('已发货', call_a)、tool('晴 26℃', call_b)")
    print("  → id 对不上就等于把两份结果张冠李戴，模型会拿天气去改订单的参数")


# ---------------------------------------------------------------- 4. 终止条件


def check_stop_conditions() -> None:
    """循环的三个终止条件：模型收工、轮次上限、外部停止信号。"""
    section("4. 终止条件：模型收工 / 轮次上限 / 外部停止信号")

    # 条件 1：模型不再调工具
    done_model = scripted_model(
        [
            ModelReply(content="我直接回答，不用工具。"),
        ]
    )
    done = react_loop(done_model, [查询订单], "你好")
    ensure(
        done.stop_reason == "completed",
        f"模型不再调工具时应当 completed，实际 {done.stop_reason!r}",
    )
    ensure(
        len(done.messages) == 2,
        f"一轮就结束的消息数应当是 2（user + assistant），实际 {len(done.messages)}",
    )
    print(f"  条件 1 模型收工 → stop_reason={done.stop_reason!r}")

    # 条件 2：达到轮次上限（每轮都换一个工具参数，避免被重复检测提前拦下）
    always_tooling = scripted_model(
        [
            ModelReply(
                content=f"再查一次 {index}。",
                tool_calls=[ToolCall(f"call_{index}", "查询订单", {"order_id": f"A-100{index}"})],
            )
            for index in range(1, 7)
        ]
    )
    capped = react_loop(
        always_tooling, [查询订单], "查很多订单", LoopConfig(max_iterations=4)
    )
    ensure(
        capped.stop_reason == "max_iterations",
        f"模型一直要工具时应当 max_iterations，实际 {capped.stop_reason!r}",
    )
    ensure(
        capped.iterations == 4,
        f"应当正好跑满 4 轮，实际 {capped.iterations} 轮",
    )
    ensure(
        len(capped.tool_calls) == 4,
        f"应当发起 4 次工具调用，实际 {len(capped.tool_calls)} 次",
    )
    print(f"  条件 2 轮次上限 → stop_reason={capped.stop_reason!r}（跑了 {capped.iterations} 轮）")

    # 条件 3：外部停止信号。工具内部触发，循环在下一轮开头收手
    signal = StopSignal()

    @tool
    def 请求人工确认(问题: str) -> str:
        """需要人工确认时调用。

        Args:
            问题: 要问人的问题
        """
        signal.trip("等待人工确认")
        return "已提交人工确认请求"

    ask_model = scripted_model(
        [
            ModelReply(
                content="这笔操作要人工确认。",
                tool_calls=[ToolCall("call_1", "请求人工确认", {"问题": "确认退款吗"})],
            ),
            ModelReply(content="（不该走到这里，信号已经触发）"),
        ]
    )
    halted = react_loop(
        ask_model, [请求人工确认], "退掉这笔订单", signal=signal
    )
    ensure(
        halted.stop_reason == "stopped",
        f"外部停止信号触发时应当 stopped，实际 {halted.stop_reason!r}",
    )
    ensure(
        halted.stop_detail == "等待人工确认",
        f"停止理由应当记录成 '等待人工确认'，实际 {halted.stop_detail!r}",
    )
    ensure(
        len(ask_model.calls) == 1,
        f"信号触发后不该再问模型，实际问了 {len(ask_model.calls)} 次",
    )
    print(f"  条件 3 外部信号 → stop_reason={halted.stop_reason!r}（{halted.stop_detail}）")


# ---------------------------------------------------------------- 5. 不收敛


def check_non_convergence() -> None:
    """循环不收敛的三个原因，以及对应的防线。"""
    section("5. 不收敛：三个原因与三道防线")

    # 原因 1：工具报错被原样回填 → 模型只会原样重试
    broken_model = scripted_model(
        [
            ModelReply(
                content="查库存。",
                tool_calls=[ToolCall("call_1", "查询库存", {"sku": "BROKEN"})],
            ),
            ModelReply(content="库存服务暂时不可用，先不查了。"),
        ]
    )
    broken = react_loop(broken_model, [查询库存], "SKU-BROKEN 还有货吗")
    feedback = broken.messages[2].content
    ensure(
        "连接超时" in feedback,
        f"回填内容应当包含真实原因，实际 {feedback!r}",
    )
    ensure(
        "不要重复同样的调用" in feedback,
        f"回填内容应当带'换条路'的引导，否则模型会原样重试，实际 {feedback!r}",
    )
    print("  原因 1 报错原样回填 → 回填里带原因 + 换路引导，而不是甩一句堆栈")

    # 原因 2：目标不可达但模型不知道 → 同名同参反复要，靠重复检测拦下
    unreachable_model = scripted_model(
        [
            ModelReply(
                content="再查一次订单 A-1001。",
                tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})],
            )
        ]
    )
    unreachable = react_loop(
        unreachable_model, [查询订单], "A-1001 状态", LoopConfig(max_iterations=8)
    )
    ensure(
        unreachable.stop_reason == "repeated_tool_call",
        f"同一个调用反复出现应当被判为原地打转，实际 {unreachable.stop_reason!r}",
    )
    ensure(
        "查询订单" in unreachable.stop_detail,
        f"停止详情应当指出重复的是哪个调用，实际 {unreachable.stop_detail!r}",
    )
    ensure(
        len(unreachable.tool_calls) == 2,
        f"应当在第 2 次重复调用时收手，实际记录了 {len(unreachable.tool_calls)} 次",
    )
    print(f"  原因 2 目标不可达 → 重复检测拦下（{unreachable.stop_detail}）")

    # 原因 3：没有轮次上限 → 无限烧钱
    no_cap_model = scripted_model(
        [
            ModelReply(
                content="再查一次。",
                tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})],
            )
        ]
    )
    capped = react_loop(
        no_cap_model, [查询订单], "A-1001 状态", LoopConfig(max_iterations=3)
    )
    ensure(
        capped.stop_reason in {"repeated_tool_call", "max_iterations"},
        f"重复调用必须被拦下，实际 {capped.stop_reason!r}",
    )
    ensure(
        capped.iterations <= 3,
        f"轮次上限应当生效，实际跑了 {capped.iterations} 轮",
    )
    print("  原因 3 没有轮次上限 → max_iterations 是最后一道兜底，绝不能省")

    # 上下文每轮都在长：这是第 5 章的伏笔
    growing_model = scripted_model(
        [
            ModelReply(
                content=f"第 {index} 步，继续查。",
                tool_calls=[ToolCall(f"call_{index}", "查询订单", {"order_id": f"A-100{index}"})],
            )
            for index in range(1, 4)
        ]
        + [ModelReply(content="三个订单都查完了。")]
    )
    growing = react_loop(growing_model, [查询订单], "批量查订单")
    tokens = [
        sum(len(m.content) for m in snapshot) for snapshot in growing_model.calls
    ]
    ensure(
        all(a < b for a, b in zip(tokens, tokens[1:])),
        f"每轮上下文都应当变长，实际长度序列 {tokens}",
    )
    print(f"  顺带一提：每轮上下文长度 {tokens} → 轮次越多越贵（第 5 章）")


# ---------------------------------------------------------------- 6. 与参考运行时对照


def check_against_reference_runtime() -> None:
    """同一段脚本下，手写循环和 runtime.Agent 应当行为等价。"""
    section("6. 对照：手写循环 vs runtime.Agent")

    script = [
        ModelReply(
            content="订单和天气一起查。",
            tool_calls=[
                ToolCall("call_a", "查询订单", {"order_id": "A-1001"}),
                ToolCall("call_b", "查询天气", {"city": "北京"}),
            ],
        ),
        ModelReply(
            content="再确认一个订单。",
            tool_calls=[ToolCall("call_c", "查询订单", {"order_id": "A-1002"})],
        ),
        ModelReply(content="A-1001 已发货，A-1002 待付款，北京晴 26℃。"),
    ]

    mine = react_loop(scripted_model(script), [查询订单, 查询天气], "查订单和天气")
    reference = Agent(
        scripted_model(script), [查询订单, 查询天气], AgentConfig(max_iterations=8)
    ).run("查订单和天气")

    mine_calls = [(c.name, c.arguments) for c in mine.tool_calls]
    reference_calls = [(c.name, c.arguments) for c in reference.tool_calls]
    ensure(
        mine_calls == reference_calls,
        f"两者的工具调用序列应当一致，手写={mine_calls} 参考={reference_calls}",
    )
    ensure(
        mine.final_answer == reference.final_answer,
        f"两者的最终回答应当一致，手写={mine.final_answer!r} 参考={reference.final_answer!r}",
    )
    ensure(
        mine.stop_reason == reference.stop_reason,
        f"两者的 stop_reason 应当一致，手写={mine.stop_reason!r} 参考={reference.stop_reason!r}",
    )
    ensure(
        len(mine_calls) == 3,
        f"这段脚本应当产生 3 次工具调用，实际 {len(mine_calls)} 次",
    )

    print(f"  工具调用序列（两者一致）：{[(n, a) for n, a in mine_calls]}")
    print(f"  结束原因（两者一致）：{mine.stop_reason}")
    print("  → 手写循环与参考运行时的差别只剩护栏：重复检测、错误改写、停止信号")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型驱动手写循环，只校验形状不校验内容。"""
    banner("第 2 章 · 真实模型路径")
    from ..labkit import real_model

    model = real_model()
    result = react_loop(model, [查询订单, 查询天气], "帮我看看订单 A-1001 到哪了，顺便说下北京天气")

    for message in result.messages:
        calls = [c.name for c in message.tool_calls]
        suffix = f" tool_calls={calls}" if calls else ""
        print(f"  {message.role}: {message.content[:120]!r}{suffix}")

    ensure(result.messages[0].role == "user", "第一条应当是用户消息")
    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "真实模型路径下工具消息也必须带 tool_call_id",
    )
    ensure(
        result.stop_reason in {"completed", "max_iterations", "stopped", "repeated_tool_call"},
        f"结束理由应当是已知取值，实际 {result.stop_reason!r}",
    )
    if result.stop_reason == "completed":
        ensure(bool(result.final_answer), "正常收工时应当给出最终回答")
    print(f"  结束原因：{result.stop_reason}")


def run() -> None:
    """跑第 2 章实验。"""
    banner("第 2 章 · Agent 循环：ReAct 的手写实现")
    check_react_roundtrip()
    check_tool_call_id_audit()
    check_parallel_tool_calls()
    check_stop_conditions()
    check_non_convergence()
    check_against_reference_runtime()
    rule("ReAct 循环 = 问模型 → 要工具就执行并回填 → 不要工具就收工；结果必须带 tool_call_id")
    rule("停止靠三个条件：模型收工 / 轮次上限 / 外部信号；不收敛靠重复检测兜底")
    print("\n✓ 第 2 章通过")
