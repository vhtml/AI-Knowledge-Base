"""第 6 章实验：工具调用与工具节点。

这个模块解决什么问题：把"模型怎么调工具"这条链路拆成能看见的几段：

1. ``@tool`` 怎么把一个普通函数变成 schema——类型来自注解、说明来自 docstring；
2. ``tool_calls`` 与 ``ToolMessage`` 的消息往返：模型说要调什么，工具回什么；
3. ``langgraph.prebuilt.ToolNode`` 是什么、替手写工具节点省掉了什么；
4. 工具参数不合法时的三条报错路径（默认值 / ``handle_tool_errors=False`` / 工具内部异常）；
5. 同一张图里"调用工具"和"直接回答"两条分支都能走通；
6. 结构化输出和工具的区别：多一次模型往返 vs 一次到位。

全部离线运行：不联网、不调模型。
"""

from __future__ import annotations

import inspect
import operator
from typing import Annotated, Any, Literal

from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from ..labkit import ai, banner, ensure, human, rule, section, tool_call


class AgentState(MessagesState):
    """第 6 章的状态：消息列表 + 一条"走过哪些工具"的记录。

    这里用 ``MessagesState`` 起步而不是自己写 ``TypedDict``：
    ``messages`` 通道自带 ``add_messages`` 归约器，能按 id 追加和更新消息。
    """

    calls: Annotated[list[str], operator.add]


@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    return a + b


@tool
def order_status(order_id: str) -> str:
    """查询订单状态。

    这个工具故意抛异常，用来演示"工具内部出错"和"参数不合法"是两条不同的路。
    """
    raise RuntimeError("订单服务不可用")


class WeatherInput(BaseModel):
    """天气查询入参——用 Pydantic 才能给字段写 description。"""

    city: str = Field(description="要查询的城市名")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="温度单位偏好"
    )


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """查询某个城市的当前天气。"""
    return f"{city} 现在 22 度（{units}）"


def check_tool_schema_generation() -> None:
    """``@tool`` 怎么把函数变成模型看得懂的 schema。"""
    section("schema：注解给类型，docstring 给说明")
    ensure(add.name == "add", f"工具名应取自函数名，实际 {add.name}")
    ensure("两数相加" in add.description, f"工具说明应来自 docstring，实际 {add.description!r}")

    schema = add.tool_call_schema.model_json_schema()
    print(f"  add 的 schema：{schema}")
    ensure(
        schema["properties"]["a"]["type"] == "integer",
        f"参数类型应来自类型注解，实际 {schema['properties']['a']}",
    )
    ensure(
        sorted(schema["required"]) == ["a", "b"],
        f"没有默认值的参数应当是 required，实际 {schema['required']}",
    )

    # 坑：docstring 里的 Args 段落**不会**变成字段级 description，
    # 模型只能看到一整坨 description 文本 + 没有字段说明的 schema。
    ensure(
        "description" not in schema["properties"]["a"],
        "docstring 的 Args 不该出现在字段级 description 里（实测确实没有）",
    )

    fixed = get_weather.tool_call_schema.model_json_schema()
    print(f"  get_weather 的字段说明：{fixed['properties']}")
    ensure(
        fixed["properties"]["city"]["description"] == "要查询的城市名",
        f"用 Pydantic args_schema 才能把字段说明送进 schema，实际 {fixed['properties']['city']}",
    )


def check_tool_call_roundtrip() -> None:
    """模型发出 tool_calls → 工具执行 → ToolMessage 回填 state → 条件边再进模型。"""
    section("消息往返：tool_calls → ToolMessage")
    from ..labkit import scripted_model

    model = scripted_model([tool_call("add", {"a": 1, "b": 2}), ai("答案是 3")])
    graph = _build_graph(model, [add])

    result = graph.invoke({"messages": [human("1+2 等于几？")], "calls": []})
    messages = result["messages"]
    for message in messages:
        print(f"  {type(message).__name__}: {getattr(message, 'content', '')!r}")

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    ensure(len(tool_messages) == 1, f"应当恰好有一条 ToolMessage，实际 {len(tool_messages)}")
    ensure(tool_messages[0].content == "3", f"工具结果应是 3，实际 {tool_messages[0].content}")
    ensure(
        tool_messages[0].tool_call_id == "call_1",
        f"ToolMessage 应当回指 tool_call 的 id，实际 {tool_messages[0].tool_call_id}",
    )
    ensure(
        model.cursor == 2,
        f"条件边应当把图送回模型节点，模型应被调用 2 次，实际 {model.cursor}",
    )
    ensure(
        messages[-1].content == "答案是 3",
        f"最后一条应是模型的最终回答，实际 {messages[-1].content!r}",
    )


def check_toolnode_vs_handwritten() -> None:
    """ToolNode 和手写工具节点做同一件事，但后者要多写十几行。"""
    section("ToolNode vs 手写工具节点")
    from ..labkit import scripted_model
    from langgraph.prebuilt import ToolNode

    tools_by_name = {add.name: add}

    def handwritten_tool_node(state: AgentState) -> dict[str, Any]:
        """手写版：自己遍历 tool_calls、自己拼 ToolMessage。"""
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name[call["name"]]
            results.append(
                ToolMessage(content=str(tool.invoke(call["args"])), tool_call_id=call["id"])
            )
        return {"messages": results}

    def build_with(tool_node: Any) -> Any:
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AgentState)
        builder.add_node(
            "model",
            _model_node(scripted_model([tool_call("add", {"a": 1, "b": 2})]), [add]),
        )
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", _route, {"tools": "tools", END: END})
        builder.add_edge("tools", "model")
        return builder.compile()

    manual = build_with(handwritten_tool_node)
    prebuilt = build_with(ToolNode([add]))

    manual_result = manual.invoke({"messages": [human("1+2?")], "calls": []})
    prebuilt_result = prebuilt.invoke({"messages": [human("1+2?")], "calls": []})

    manual_tool = [m for m in manual_result["messages"] if isinstance(m, ToolMessage)]
    prebuilt_tool = [m for m in prebuilt_result["messages"] if isinstance(m, ToolMessage)]
    print(f"  手写节点结果：{manual_tool[0].content!r}，ToolNode 结果：{prebuilt_tool[0].content!r}")

    ensure(
        manual_tool[0].content == prebuilt_tool[0].content,
        "两种写法的工具结果应当一致",
    )
    ensure(
        isinstance(ToolNode([add]), ToolNode),
        "ToolNode 应当能从 langgraph.prebuilt 直接拿到",
    )


def check_tool_error_paths() -> None:
    """工具参数不合法时的三条报错路径。"""
    section("工具出错：三条路径")
    from ..labkit import scripted_model
    from langgraph.prebuilt import ToolNode

    # 路径 A：默认值——校验错误被翻译成 ToolMessage，模型看得见错误并有机会纠正
    model = scripted_model(
        [tool_call("add", {"a": "x"}), ai("参数不合法，我不调用工具了")]
    )
    graph = _build_graph(model, [add], tool_node=ToolNode([add]))
    result = graph.invoke({"messages": [human("1+2?")], "calls": []})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    ensure(len(tool_messages) == 1, f"应当有一条错误 ToolMessage，实际 {len(tool_messages)}")
    ensure(
        getattr(tool_messages[0], "status", None) == "error",
        f"错误 ToolMessage 的 status 应当是 error，实际 {getattr(tool_messages[0], 'status', None)}",
    )
    ensure(
        "Error invoking tool" in tool_messages[0].content,
        f"错误内容应当告诉模型哪里错了，实际 {tool_messages[0].content!r}",
    )
    ensure(
        model.cursor == 2,
        f"错误应当回到模型让它重试，模型应被调用 2 次，实际 {model.cursor}",
    )
    print(f"  路径 A（默认）：ToolMessage 报错 → 模型看到错误 → {result['messages'][-1].content!r}")

    # 路径 B：handle_tool_errors=False——不吞，直接抛
    strict_model = scripted_model([tool_call("add", {"a": "x"})])
    strict_graph = _build_graph(
        strict_model,
        [add],
        tool_node=ToolNode([add], handle_tool_errors=False),
    )
    try:
        strict_graph.invoke({"messages": [human("1+2?")], "calls": []})
    except Exception as exc:
        print(f"  路径 B（不吞）：抛出 {type(exc).__name__}")
        ensure(
            "Error invoking tool" in str(exc),
            f"应当抛出带校验细节的异常，实际 {exc}",
        )
    else:
        raise AssertionError("handle_tool_errors=False 时应当抛异常")

    # 路径 C：工具内部抛异常——默认行为是冒泡，不是变成 ToolMessage
    boom_model = scripted_model([tool_call("order_status", {"order_id": "A1"})])
    boom_graph = _build_graph(boom_model, [order_status])
    try:
        boom_graph.invoke({"messages": [human("查订单")], "calls": []})
    except RuntimeError as exc:
        print(f"  路径 C（工具内部异常）：冒泡为 {type(exc).__name__}: {exc}")
        ensure("订单服务不可用" in str(exc), f"应当把原始异常冒出来，实际 {exc}")
    else:
        raise AssertionError("工具内部异常默认应当冒泡，而不是被 ToolNode 吞掉")


def check_two_branches() -> None:
    """同一张图里"调用工具"和"直接回答"两条分支都能走通。"""
    section("同一张图：调工具 / 直接回答")
    from ..labkit import scripted_model

    tool_model = scripted_model([tool_call("add", {"a": 2, "b": 3}), ai("和是 5")])
    tool_result = _build_graph(tool_model, [add]).invoke(
        {"messages": [human("2+3 等于几？")], "calls": []}
    )
    ensure(
        any(isinstance(m, ToolMessage) for m in tool_result["messages"]),
        "分支一应当出现 ToolMessage",
    )
    ensure(
        tool_result["messages"][-1].content == "和是 5",
        f"分支一最终回答应是 5，实际 {tool_result['messages'][-1].content!r}",
    )

    direct_model = scripted_model([ai("直接回答：LangGraph 是状态机")])
    direct_result = _build_graph(direct_model, [add]).invoke(
        {"messages": [human("介绍一下 LangGraph")], "calls": []}
    )
    ensure(
        not any(isinstance(m, ToolMessage) for m in direct_result["messages"]),
        "分支二不该出现 ToolMessage",
    )
    ensure(
        direct_result["messages"][-1].content == "直接回答：LangGraph 是状态机",
        f"分支二最终回答应原样保留，实际 {direct_result['messages'][-1].content!r}",
    )
    ensure(direct_model.cursor == 1, f"直接回答时模型只该被调用一次，实际 {direct_model.cursor}")
    print("  分支一：模型→工具→模型；分支二：模型→结束")


def check_structured_output_vs_tool() -> None:
    """结构化输出 vs 工具：差的是一次模型往返。"""
    section("结构化输出 vs 工具")
    from langchain.agents import create_agent
    from langgraph.graph.state import CompiledStateGraph

    from ..labkit import scripted_model

    # 工具路线：human / ai(tool_calls) / tool / ai(最终) —— 四条消息
    model = scripted_model([tool_call("add", {"a": 1, "b": 2}), ai("3")])
    result = _build_graph(model, [add]).invoke({"messages": [human("1+2?")], "calls": []})
    tool_route = len(result["messages"])
    ensure(tool_route == 4, f"工具路线应当产生 4 条消息，实际 {tool_route}")
    print(f"  工具路线：{tool_route} 条消息、2 次模型调用")

    # create_agent 的内部结构，就是上面这张"模型 ↔ 工具"的手写图
    agent = create_agent(scripted_model([ai("hi")]), tools=[add])
    node_names = list(agent.get_graph().nodes.keys())
    print(f"  create_agent 的节点：{node_names}")
    ensure(
        isinstance(agent, CompiledStateGraph),
        f"create_agent 应当返回一张编译图，实际 {type(agent)}",
    )
    ensure(
        {"model", "tools"} <= set(node_names),
        f"create_agent 内部应当就是 model + tools 两个节点，实际 {node_names}",
    )
    ensure(
        "response_format" in inspect.signature(create_agent).parameters,
        "create_agent 应当支持 response_format（结构化输出走的是另一条路）",
    )


def run_live() -> None:
    """真实模型路径：真调一次模型，让它自己决定用不用工具；只校验形状。"""
    banner("第 6 章 · 真实模型路径")
    from ..labkit import real_chat_model

    model = real_chat_model()
    graph = _build_graph(model, [add])

    result = graph.invoke({"messages": [human("用 add 工具算一下 17 加 25")], "calls": []})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    for message in result["messages"]:
        print(f"  {type(message).__name__}: {getattr(message, 'content', '')!r}")

    # 真实模型爱调工具还是爱直接回答都不确定，只校验"消息历史形状合法"
    ensure(len(result["messages"]) >= 2, f"消息条数不对：{len(result['messages'])}")
    ensure(
        all(isinstance(m, ToolMessage) for m in tool_messages),
        "工具结果应当是 ToolMessage",
    )
    if tool_messages:
        ensure(
            all(m.tool_call_id for m in tool_messages),
            "ToolMessage 应当带 tool_call_id",
        )
    ensure(
        isinstance(result["messages"][-1].content, str),
        "最后一条消息的 content 应当是字符串",
    )


def _model_node(model: Any, tools: list[Any]) -> Any:
    """造一个"模型节点"：把消息喂给模型，把回复写回 messages。"""

    def model_node(state: AgentState) -> dict[str, Any]:
        reply = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": [reply]}

    return model_node


def _route(state: AgentState) -> str:
    """条件边：模型这次要不要调工具。"""
    from langgraph.graph import END

    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def _build_graph(model: Any, tools: list[Any], tool_node: Any = None) -> Any:
    """搭一张"模型节点 + 工具节点"的循环图。

    ``tool_node`` 省略时用官方预置的 ``ToolNode``，传入时用调用方给的节点。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    builder = StateGraph(AgentState)
    builder.add_node("model", _model_node(model, tools))
    builder.add_node("tools", tool_node if tool_node is not None else ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", _route, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile()


def run() -> None:
    """跑第 6 章实验。"""
    banner("第 6 章 · 工具调用与工具节点")
    check_tool_schema_generation()
    check_tool_call_roundtrip()
    check_toolnode_vs_handwritten()
    check_tool_error_paths()
    check_two_branches()
    check_structured_output_vs_tool()
    rule("工具 = 模型提出调用意图 + 图决定下一步；ToolNode 是官方预置的那一半")
    print("\n✓ 第 6 章通过")
