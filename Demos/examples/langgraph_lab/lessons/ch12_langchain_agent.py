"""第 12 章实验：LangChain 1.x——create_agent 与中间件。

这个模块解决什么问题：第 12 章的主张要变成能跑的东西：

1. **create_agent 返回的就是一张编译好的 LangGraph 图**；
2. **工具循环**能跑通（模型要工具 → 工具执行 → 模型收尾）；
3. **钩子顺序**：before 正序、after 逆序、wrap 嵌套；
4. **中间件能改状态**（自定义字段计数）和**改模型请求**（注入系统提示）；
5. **HumanInTheLoopMiddleware 能让工具调用停下来等审批**；
6. **带中间件的 Agent 可以整体塞进自己的 StateGraph 当节点**。

全部离线，用 labkit 的脚本化假模型驱动，不联网、不调模型。
"""

from __future__ import annotations

from typing import Any, NotRequired

from ..labkit import ai, banner, ensure, human, rule, scripted_model, tool_call


def build_weather_tool() -> Any:
    """一个最小工具。"""
    from langchain.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """查询某地的天气。"""
        return f"{city} 晴，25 度"

    return get_weather


# ------------------------------------------------------------ 检查项


def check_create_agent_is_a_graph() -> None:
    """create_agent 返回的是编译后的 LangGraph 图。"""
    print("--- create_agent 是藏起来的图 ---")
    from langchain.agents import create_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_agent(scripted_model([ai("你好")]), tools=[])
    print(f"  返回类型：{type(agent).__module__}.{type(agent).__name__}")
    ensure(
        isinstance(agent, CompiledStateGraph),
        f"create_agent 应当返回 CompiledStateGraph，实际 {type(agent)}",
    )
    print("  ✓ 它和手写的 StateGraph 是同一类东西")


def check_tool_loop_works() -> None:
    """模型要工具 → 工具执行 → 模型收尾，这条循环要能跑通。"""
    print("--- 工具循环 ---")
    from langchain.agents import create_agent

    weather = build_weather_tool()
    model = scripted_model(
        [
            tool_call("get_weather", {"city": "杭州"}),
            ai("杭州现在晴，25 度。"),
        ]
    )
    agent = create_agent(model, tools=[weather])
    result = agent.invoke({"messages": [human("杭州天气怎么样？")]})
    messages = result["messages"]
    print(f"  消息条数：{len(messages)}，类型：{[m.type for m in messages]}")

    ensure(len(messages) == 4, f"应当有 4 条消息（用户/要工具/工具结果/回答），实际 {len(messages)}")
    ensure(
        [m.type for m in messages] == ["human", "ai", "tool", "ai"],
        f"消息顺序应当是 human→ai→tool→ai，实际 {[m.type for m in messages]}",
    )
    ensure(messages[1].tool_calls, "第二条应当是带工具调用的助手消息")
    ensure(messages[2].type == "tool", f"第三条应当是工具消息，实际 {messages[2].type}")
    ensure(
        "杭州" in str(messages[2].content),
        f"工具结果里应当有城市名，实际 {messages[2].content}",
    )
    print("  ✓ 模型 → 工具 → 模型，整条循环跑通")


def check_hook_order() -> None:
    """before 正序、after 逆序。"""
    print("--- 钩子执行顺序 ---")
    from langchain.agents import create_agent
    from langchain.agents.middleware import after_model, before_model

    order: list[str] = []

    @before_model
    def first_before(state: Any, runtime: Any) -> None:
        order.append("before-1")
        return None

    @before_model
    def second_before(state: Any, runtime: Any) -> None:
        order.append("before-2")
        return None

    @after_model
    def first_after(state: Any, runtime: Any) -> None:
        order.append("after-1")
        return None

    @after_model
    def second_after(state: Any, runtime: Any) -> None:
        order.append("after-2")
        return None

    agent = create_agent(
        scripted_model([ai("回答")]),
        tools=[],
        middleware=[first_before, second_before, first_after, second_after],
    )
    agent.invoke({"messages": [human("hi")]})
    print(f"  执行顺序：{order}")

    ensure(
        order.index("before-1") < order.index("before-2"),
        f"before 钩子应当正序，实际 {order}",
    )
    ensure(
        order.index("after-2") < order.index("after-1"),
        f"after 钩子应当逆序，实际 {order}",
    )
    print("  ✓ before 正着进，after 倒着出")


def check_middleware_changes_state() -> None:
    """中间件可以往自定义状态里写字段。"""
    print("--- 中间件改状态 ---")
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentState, after_model
    from langchain.messages import AIMessage

    class CountingState(AgentState):
        model_calls: NotRequired[int]

    @after_model(state_schema=CountingState)
    def count_calls(state: CountingState, runtime: Any) -> dict[str, Any]:
        return {"model_calls": state.get("model_calls", 0) + 1}

    agent = create_agent(
        scripted_model([ai("一"), ai("二")]),
        tools=[],
        middleware=[count_calls],
    )
    result = agent.invoke({"messages": [human("hi")]})
    print(f"  自定义状态：{result.get('model_calls')}")
    ensure(
        result.get("model_calls") == 1,
        f"after_model 应当把计数加到 1，实际 {result.get('model_calls')}",
    )
    print("  ✓ 中间件能扩展 Agent 状态")


def check_wrap_model_injects_prompt() -> None:
    """wrap_model_call 能改写发出去的模型请求。"""
    print("--- 包裹式钩子改写请求 ---")
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
    from langchain.messages import SystemMessage

    captured: list[str] = []

    @wrap_model_call
    def inject_context(
        request: ModelRequest, handler: Any
    ) -> ModelResponse:
        # system_message 在设了 system_prompt 时是 SystemMessage 对象，
        # 读它要用 content_blocks；一个 system_prompt 都没设时它是 None
        blocks = list(request.system_message.content_blocks or [])
        blocks.append({"type": "text", "text": "补充上下文：今天是周五。"})
        captured.append(str(blocks))
        new_message = SystemMessage(content=blocks)
        return handler(request.override(system_message=new_message))

    agent = create_agent(
        scripted_model([ai("好的")]),
        tools=[],
        system_prompt="你是一个助手",
        middleware=[inject_context],
    )
    agent.invoke({"messages": [human("今天周几？")]})
    print(f"  发出的系统提示片段数：{captured[0].count('text') if captured else 0}")
    ensure(bool(captured), "wrap_model_call 应当被调用")
    ensure(
        "补充上下文" in captured[0],
        f"改写后的系统提示应当包含补充上下文，实际 {captured[0][:200]}",
    )
    print("  ✓ 包裹式钩子能在模型调用前改写请求")


def check_hitl_middleware_pauses() -> None:
    """HumanInTheLoopMiddleware 能让工具调用停下来等审批。"""
    print("--- 人工审批中间件 ---")
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    weather = build_weather_tool()
    model = scripted_model(
        [
            tool_call("get_weather", {"city": "杭州"}),
            ai("杭州现在晴。"),
        ]
    )
    agent = create_agent(
        model,
        tools=[weather],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
    )
    result = agent.invoke({"messages": [human("杭州天气怎么样？")]})
    print(f"  结果键：{sorted(result.keys())}")
    ensure(
        "__interrupt__" in result,
        f"带审批的 Agent 应当在工具执行前停下，实际 {sorted(result.keys())}",
    )
    print("  ✓ 工具调用被拦下来，等人工决策")


def check_agent_as_graph_node() -> None:
    """带中间件的 Agent 可以整体塞进自己的图。"""
    print("--- Agent 塞进 StateGraph ---")
    from langchain.agents import AgentState, create_agent
    from langchain.agents.middleware import before_model
    from langgraph.graph import END, START, StateGraph

    hits: list[str] = []

    @before_model
    def note(state: Any, runtime: Any) -> None:
        hits.append("before_model")
        return None

    inner = create_agent(scripted_model([ai("子 Agent 的回答")]), tools=[], middleware=[note])

    class OuterState(AgentState):
        extra: str

    def wrap(state: OuterState) -> dict[str, Any]:
        return {"extra": "已处理"}

    builder = StateGraph(OuterState)
    builder.add_node("inner_agent", inner)
    builder.add_node("wrap", wrap)
    builder.add_edge(START, "inner_agent")
    builder.add_edge("inner_agent", "wrap")
    builder.add_edge("wrap", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [human("hi")]})
    print(f"  外层图结果：{result.get('extra')}，中间件触发 {len(hits)} 次")
    ensure(
        result.get("extra") == "已处理",
        f"外层图应当跑完两个节点，实际 {result.get('extra')}",
    )
    ensure(hits, "塞进图里的 Agent，中间件钩子仍然应当运行")
    print("  ✓ 中间件不是独立运行时，Agent 当节点用，钩子照跑")


def run() -> None:
    """跑第 12 章实验。"""
    banner("第 12 章 · LangChain 1.x：create_agent 与中间件")
    check_create_agent_is_a_graph()
    check_tool_loop_works()
    check_hook_order()
    check_middleware_changes_state()
    check_wrap_model_injects_prompt()
    check_hitl_middleware_pauses()
    check_agent_as_graph_node()
    rule("钩子顺序记三遍：before 正、after 逆、wrap 嵌套")
    print("\n✓ 第 12 章通过")


def run_live() -> None:
    """真实模型路径：让真模型带着工具跑一轮完整循环。

    断言只校验形状（消息链路完整、有最终回答），不校验内容——
    真模型说什么不该由运行路径来管。
    """
    from langchain.agents import create_agent

    from ..labkit import real_chat_model

    banner("第 12 章 · 真实模型路径")
    weather = build_weather_tool()
    agent = create_agent(
        real_chat_model(),
        tools=[weather],
        system_prompt="你是一个天气助手，回答要简短。",
    )
    result = agent.invoke({"messages": [human("杭州今天天气怎么样？")]})
    messages = result["messages"]
    for message in messages:
        print(f"  [{message.type}] {str(message.content)[:90]}")

    ensure(len(messages) >= 2, f"至少应当有用户与助手两条消息，实际 {len(messages)}")
    ensure(
        messages[-1].type == "ai",
        f"最后一条应当是助手消息，实际 {messages[-1].type}",
    )
    ensure(bool(str(messages[-1].content).strip()), "最终回答不应当为空")
    used_tool = any(m.type == "tool" for m in messages)
    print(f"  ✓ 真实模型跑通，工具被调用：{used_tool}")
