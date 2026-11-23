"""第 1 章实验：什么是 Agent、边界在哪、最小内核长什么样。

这个模块解决什么问题：第 1 章讲的是判断题——"这事该不该用 Agent"。
判断题不能靠背结论，必须能**跑出来看**。所以本章把四件事变成可执行的断言：

1. **四个必要条件**——模型、工具、循环、自主性，缺一个就不再是 Agent；
   逐个抽掉跑一遍，看行为怎么变。
2. **Workflow vs Agent**——同一个任务两种实现，把"路由由谁决定"这个
   区别跑成可观测的差异：工作流在语义路由上会错，Agent 不会。
3. **最小内核**——亲手写那 12 行循环，让它真的把工具跑起来。
4. **能力梯度 L0–L4**——用同一个任务演示 L2 和 L4 的差别在哪。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..labkit import (
    ModelReply,
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
    orders = {"A-1001": "已发货", "A-1002": "待付款"}
    return orders.get(order_id, "未找到")


# ---------------------------------------------------------------- 1. 四个必要条件


def check_four_necessary_conditions() -> None:
    """模型、工具、循环、自主性——缺一个就不再是 Agent。"""
    section("四个必要条件：逐个抽掉看差别")

    # 条件 1：有模型 + 有工具 + 有循环 → 是 Agent
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})]),
            ModelReply(content="订单 A-1001 已发货"),
        ]
    )
    agent = Agent(model, [查询订单])
    result = agent.run("A-1001 什么状态？")

    ensure(result.final_answer == "订单 A-1001 已发货", f"完整 Agent 应当给出最终回答，实际 {result.final_answer!r}")
    ensure(len(result.tool_calls) == 1, f"应当调用了一次工具，实际 {len(result.tool_calls)}")
    ensure(result.stop_reason == "completed", f"应当正常结束，实际 {result.stop_reason}")
    print(f"  条件齐全 → {result.final_answer!r}（工具调用 {len(result.tool_calls)} 次）")

    # 条件 2：抽掉工具 → 只能说，Agent 退化成聊天机器人
    no_tool_model = scripted_model([ModelReply(content="我没法查订单，你只能自己看后台")])
    no_tool_agent = Agent(no_tool_model, [])
    no_tool_result = no_tool_agent.run("A-1001 什么状态？")
    ensure(
        not no_tool_result.tool_calls,
        "没有工具时不可能有工具调用",
    )
    ensure(
        "没法查" in no_tool_result.final_answer,
        f"没有工具时模型只能口头解释，实际 {no_tool_result.final_answer!r}",
    )
    print(f"  抽掉工具 → {no_tool_result.final_answer!r}（说不了话，只能问）")

    # 条件 3：抽掉循环（只问一次）→ 退化成单次调用
    one_shot_model = scripted_model(
        [ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})])]
    )
    one_shot = _one_shot_loop(one_shot_model, [查询订单], "A-1001 什么状态？")
    ensure(
        one_shot["tool_calls"] == 1,
        f"单次循环只能调一次工具，实际 {one_shot['tool_calls']}",
    )
    ensure(
        one_shot["final_answer"] == "",
        "单次循环拿到工具调用后没有第二次机会，模型看不到结果",
    )
    print("  抽掉循环 → 模型看不到工具结果，回答为空（退化成分类器）")

    # 条件 4：抽掉自主性（路由写死）→ 退化成工作流
    workflow_result = _workflow_route("我想看看我的包裹到哪了")
    ensure(
        workflow_result["route"] == "fallback",
        f"写死路由时语义表述匹配不上关键字，应落到兜底分支，实际 {workflow_result['route']}",
    )
    print(f"  抽掉自主性 → 关键词没匹配上，落到 {workflow_result['route']!r} 分支")


def _one_shot_loop(
    model: Any,
    tools: Sequence[Any],
    user_input: str,
) -> dict[str, Any]:
    """只跑一轮的"循环"——用来演示没有循环意味着模型看不到工具结果。"""
    messages = [human(user_input)]
    reply = model.complete(messages, tools=list(tools))
    messages.append(reply.to_message())
    tool_call_count = 0
    for call in reply.tool_calls:
        target = next(t for t in tools if t.name == call.name)
        messages.append(tool_result(target.invoke(call.arguments), call.id, call.name))
        tool_call_count += 1
    return {
        "tool_calls": tool_call_count,
        "final_answer": reply.content if not reply.tool_calls else "",
    }


def _workflow_route(user_input: str) -> dict[str, Any]:
    """写死路由的工作流——用一个**具体**语义 bug 来体现自主性的价值。"""
    rules = {
        "订单": "query_order",
        "物流": "track_shipping",
        "退款": "refund",
    }
    for keyword, route in rules.items():
        if keyword in user_input:
            return {"route": route}
    return {"route": "fallback"}


# ---------------------------------------------------------------- 2. Workflow vs Agent


def check_workflow_vs_agent() -> None:
    """同一个请求，工作流和 Agent 的表现差在哪。"""
    section("Workflow vs Agent：路由由谁决定")

    request = "我的包裹到哪了"
    print(f"  同一个请求：{request!r}")

    # 工作流：关键词匹配。中文里"包裹"不等于"物流"，匹配不上。
    workflow = _workflow_route(request)
    print(f"  工作流 → {workflow['route']}（关键词匹配 '物流'，没命中）")
    ensure(workflow["route"] == "fallback", "写死路由不该命中 '包裹'")

    # Agent：模型自己判断该调哪个工具。
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})]),
            ModelReply(content="帮你查到了：订单 A-1001 已发货"),
        ]
    )
    agent = Agent(model, [查询订单])
    result = agent.run(request)
    print(f"  Agent   → {result.final_answer!r}（模型自己选了工具）")
    ensure(
        result.final_answer == "帮你查到了：订单 A-1001 已发货",
        f"Agent 应当给出最终回答，实际 {result.final_answer!r}",
    )

    # 关键差别：Agent 的路由选择也是**可以被评测的**——它留在了轨迹里。
    ensure(
        len(result.steps) == 3,
        f"轨迹应当有 3 步（模型→工具→模型），实际 {len(result.steps)}",
    )
    print(f"  轨迹：{' → '.join(s.label for s in result.steps)}")


# ---------------------------------------------------------------- 3. 最小内核


def check_minimal_kernel() -> None:
    """把第 6 节那 12 行循环亲手写一遍。"""
    section("最小内核：12 行循环")

    lines = [
        "def run(user_input):",
        "    messages = [system_prompt, user_input]",
        "    for _ in range(MAX_ITERATIONS):",
        "        reply = model.complete(messages, tools=TOOLS)",
        "        messages.append(reply)",
        "        if not reply.tool_calls:",
        "            return reply.content",
        "        for call in reply.tool_calls:",
        "            result = run_tool(call)",
        "            messages.append(tool_result(result, call.id))",
        "    return '达到最大轮次，没能给出答案'",
    ]
    for line in lines:
        print(f"    {line}")

    # 真正跑一遍这个循环
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1002"})]),
            ModelReply(content="订单 A-1002 待付款"),
        ]
    )
    agent = Agent(model, [查询订单], AgentConfig(max_iterations=8))
    result = agent.run("A-1002 什么状态？")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 1, f"应当回填一条工具消息，实际 {len(tool_messages)}")
    ensure(tool_messages[0].content == "待付款", f"工具结果应是 '待付款'，实际 {tool_messages[0].content!r}")
    ensure(result.final_answer == "订单 A-1002 待付款", f"最终回答不对：{result.final_answer!r}")
    print(f"  跑通：{result.final_answer!r}")


# ---------------------------------------------------------------- 4. 能力梯度


def check_level_ladder() -> None:
    """L0–L4 的差别，用"任务需要几步"来度量。"""
    section("能力梯度：L0 → L4")

    # L0：单次调用
    l0_model = scripted_model([ModelReply(content="已发货")])
    l0 = l0_model.complete([human("A-1001 状态")])
    ensure(l0.content == "已发货", "L0 应当一次给出答案")

    # L2：单 Agent + 工具，需要 2 轮
    l2_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})]),
            ModelReply(content="已发货"),
        ]
    )
    l2 = Agent(l2_model, [查询订单]).run("A-1001 状态")
    ensure(len(l2.steps) == 3, f"L2 应当 3 步，实际 {len(l2.steps)}")

    # L4：长任务，模型反复用工具，每一步都基于上一步的结果
    l4_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})]),
            ModelReply(tool_calls=[ToolCall("call_2", "查询订单", {"order_id": "A-1002"})]),
            ModelReply(tool_calls=[ToolCall("call_3", "查询订单", {"order_id": "A-1003"})]),
            ModelReply(content="三个订单都查完了"),
        ]
    )
    l4 = Agent(l4_model, [查询订单]).run("批量查三个订单")
    ensure(len(l4.steps) == 7, f"L4 应当 7 步（4 模型 + 3 工具），实际 {len(l4.steps)}")

    print(f"  L0 单次调用：1 次模型调用")
    print(f"  L2 单 Agent：{len(l2.steps)} 步")
    print(f"  L4 长任务  ：{len(l4.steps)} 步，轮次上限是 L4 的关键约束")
    print("  → 级别越高，轮次上限和上下文预算越重要（第 5、10、15 章）")


# ---------------------------------------------------------------- 5. 决策表


def check_decision_table() -> None:
    """把"该不该用 Agent"变成一张可执行的表。"""
    section("决策表：该不该用 Agent")

    cases: list[tuple[str, str, str]] = [
        ("步骤完全已知", "workflow", "上 Agent 只增加不确定性"),
        ("一次调用能答完", "direct", "套 Agent 是纯浪费"),
        ("看情况选工具", "agent", "Agent 的主场"),
        ("任务很长要续跑", "stateful_agent", "需要持久化"),
    ]

    for situation, choice, why in cases:
        print(f"  {situation:<12} → {choice:<16} （{why}）")

    ensure(len(cases) == 4, "决策表应有 4 条")
    ensure(
        cases[2][1] == "agent",
        "语义路由类需求应当选 Agent",
    )


def run_live() -> None:
    """真实模型路径：让真模型自己判断该不该用工具，只校验形状。"""
    banner("第 1 章 · 真实模型路径")
    from ..labkit import real_model

    model = real_model()
    agent = Agent(model, [查询订单])

    result = agent.run("帮我看看订单 A-1001 到哪了")
    for message in result.messages:
        print(f"  {message.role}: {message.content[:120]!r}")

    ensure(result.messages[0].role == "user", "第一条应当是用户消息")
    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "工具消息必须带 tool_call_id",
    )
    print(f"  结束原因：{result.stop_reason}")


def run() -> None:
    """跑第 1 章实验。"""
    banner("第 1 章 · 什么是 Agent：定义、边界与最小内核")
    check_four_necessary_conditions()
    check_workflow_vs_agent()
    check_minimal_kernel()
    check_level_ladder()
    check_decision_table()
    rule("Agent = 模型 + 工具 + 循环 + 自主性；少一个就不是；用工作流能解决的别用 Agent")
    print("\n✓ 第 1 章通过")
