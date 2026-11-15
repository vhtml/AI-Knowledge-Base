"""第 14 章实验：Deep Agents 的 harness 视角——harness 到底替你做了什么。

这个模块解决什么问题：第 14 章要回答"用 Deep Agents 省下来的那些代码，到底被谁写掉了"。
只读文档容易变成口号，这里把答案变成可执行的断言：

1. **harness 是一张编译好的图**——``create_deep_agent`` 返回 ``CompiledStateGraph``，
   状态里自带 ``files`` / ``todos`` 这些通道；
2. **预置能力是可枚举的**——工具面里能点出文件系统、子智能体委派、规划三块；
3. **规划真的落到了状态上**——``write_todos`` 调完，``todos`` 通道里有内容；
4. **自定义工具挂进 harness 就能用**——和本地 ``@tool`` 完全平权；
5. **降层用 LangGraph 的对照**——同样的需求，用原生图写清楚每条边。

全程离线：不联网、不调模型。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from ..labkit import ai, banner, ensure, human, rule, scripted_model, tool_call

# ------------------------------------------------------------ harness 能力清单

#: harness 默认挂上的文件系统工具（官方文档的工具表）。
FILESYSTEM_TOOLS: frozenset[str] = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"}
)

#: 委派子智能体的工具名。
DELEGATION_TOOL = "task"

#: 结构化待办的工具名，来自 ``TodoListMiddleware``。
PLANNING_TOOL = "write_todos"


def deep_model(responses: Sequence[BaseMessage | str]) -> Any:
    """造一个能进 Deep Agents 的脚本化模型。

    为什么不能直接用 ``labkit.scripted_model``：LangChain 的摘要中间件按**属性**
    读 ``model._llm_type``（``model._llm_type.startswith(...)``），
    而 ``langchain_core`` 里它是 property、``labkit`` 里却是普通方法，
    直接用会抛 ``AttributeError: 'function' object has no attribute 'startswith'``。

    这里只在实例上盖一个同名属性把它变成字符串，不改 ``labkit``，也不另写模型类。
    """
    model = scripted_model(responses)
    model._llm_type = "scripted"
    return model


def harness_tools(agent: Any) -> list[str]:
    """列出这张图实际暴露给模型的工具名。

    Deep Agents 把所有工具汇到一个 ``ToolNode`` 里，
    而编译后的图把这个节点包成了 ``PregelNode``，真正的工具表在 ``.bound`` 上。
    """
    return sorted(agent.nodes["tools"].bound.tools_by_name)


# ------------------------------------------------------------ 断言项


def check_agent_graph_and_state_channels() -> Any:
    """``create_deep_agent`` 返回编译图，状态里自带 ``files`` / ``todos`` 通道。"""
    from langchain.agents.middleware import TodoListMiddleware
    from langgraph.graph.state import CompiledStateGraph

    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=deep_model([ai("跑起来了")]),
        tools=[],
        middleware=[TodoListMiddleware()],
    )

    ensure(
        isinstance(agent, CompiledStateGraph),
        f"create_deep_agent 应当返回 CompiledStateGraph，实际 {type(agent)}",
    )
    channels = set(agent.channels)
    ensure("messages" in channels, "状态里应当有 messages 通道")
    ensure("files" in channels, f"状态里应当有 files 通道，实际 {sorted(channels)}")
    ensure("todos" in channels, f"状态里应当有 todos 通道，实际 {sorted(channels)}")

    result = agent.invoke({"messages": [human("你好")]})
    ensure(
        result["messages"][-1].content == "跑起来了",
        f"harness 应当把模型回答放进 messages，实际 {result['messages'][-1].content!r}",
    )

    print(f"  编译图类型：{type(agent).__module__}.{type(agent).__name__}")
    print(f"  状态通道：{sorted(c for c in channels if not c.startswith(('branch:', '__')))}")
    print("  ✓ harness = 一张预装好的 LangGraph 图，通道也是预置的")
    return agent


def check_preset_capabilities_are_enumerable() -> None:
    """预置能力在工具面上是看得见的：文件系统、委派、规划。"""
    from langchain.agents.middleware import TodoListMiddleware

    from deepagents import create_deep_agent

    @tool
    def ping(host: str) -> str:
        """检查某台机器是否可达。"""
        return f"{host}: pong"

    agent = create_deep_agent(
        model=deep_model([ai("ok")]),
        tools=[ping],
        middleware=[TodoListMiddleware()],
    )
    names = harness_tools(agent)

    missing = FILESYSTEM_TOOLS - set(names)
    ensure(not missing, f"harness 应当预置文件系统工具，缺 {sorted(missing)}")
    ensure(DELEGATION_TOOL in names, f"harness 应当预置 {DELEGATION_TOOL} 工具，实际 {names}")
    ensure(PLANNING_TOOL in names, f"显式加入 TodoListMiddleware 后应有 {PLANNING_TOOL}，实际 {names}")
    ensure("ping" in names, f"自定义工具应当出现在工具面上，实际 {names}")

    # 规划是 opt-in 的：不传 TodoListMiddleware 就没有 write_todos。
    bare = create_deep_agent(model=deep_model([ai("ok")]), tools=[])
    bare_names = harness_tools(bare)
    ensure(
        PLANNING_TOOL not in bare_names,
        f"不传 TodoListMiddleware 时不该有 {PLANNING_TOOL}，实际 {bare_names}",
    )

    print(f"  工具面（带规划）：{names}")
    print(f"  工具面（默认）：  {bare_names}")
    print("  ✓ 预置能力可枚举；且规划能力要显式 opt-in 才出现")


def check_planning_lands_in_state() -> None:
    """``write_todos`` 调完，待办出现在状态通道里。"""
    from langchain.agents.middleware import TodoListMiddleware

    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=deep_model(
            [
                tool_call(
                    PLANNING_TOOL,
                    {
                        "todos": [
                            {"content": "调研 LangGraph 的持久化", "status": "pending"},
                            {"content": "写实验结论", "status": "in_progress"},
                        ]
                    },
                ),
                ai("待办已建立"),
            ]
        ),
        tools=[],
        middleware=[TodoListMiddleware()],
    )

    result = agent.invoke({"messages": [human("先列个待办")]})
    todos = result.get("todos")

    ensure(todos, f"调完 {PLANNING_TOOL} 后 todos 通道应当有内容，实际 {todos!r}")
    ensure(len(todos) == 2, f"应当有两条待办，实际 {todos}")
    ensure(
        todos[0]["content"] == "调研 LangGraph 的持久化",
        f"待办内容应当原样落库，实际 {todos[0]}",
    )
    ensure(
        {item["status"] for item in todos} == {"pending", "in_progress"},
        f"待办状态应当被保存，实际 {todos}",
    )

    print(f"  todos 通道：{todos}")
    print("  ✓ 规划不是提示词里的字面量，它有独立的、可检查的状态通道")


def check_filesystem_lands_in_state() -> None:
    """``write_file`` 调完，文件出现在 ``files`` 通道里。"""
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=deep_model(
            [
                tool_call(
                    "write_file",
                    {"file_path": "/notes.md", "content": "# 调研笔记\n- LangGraph 很强\n"},
                ),
                ai("笔记写好了"),
            ]
        ),
        tools=[],
    )

    result = agent.invoke({"messages": [human("把笔记写到 /notes.md")]})
    files = result.get("files") or {}

    ensure("/notes.md" in files, f"files 通道应当有 /notes.md，实际 {list(files)}")
    ensure(
        "LangGraph 很强" in files["/notes.md"]["content"],
        f"文件内容应当原样落库，实际 {files['/notes.md']!r}",
    )

    print(f"  files 通道：{list(files)}")
    print(f"  文件内容：{files['/notes.md']['content']!r}")
    print("  ✓ 虚拟文件系统也是状态通道，不是一块隐形的本地磁盘")


def check_custom_tool_is_called() -> None:
    """自定义工具挂进 harness 后和内置工具平权。"""
    from deepagents import create_deep_agent

    @tool
    def currency_convert(amount: float, to: str) -> str:
        """把金额换算成目标货币。"""
        return f"{amount} {to} = {amount * 7.2:.2f} CNY"

    agent = create_deep_agent(
        model=deep_model(
            [
                tool_call("currency_convert", {"amount": 10.0, "to": "USD"}),
                ai("换算完成"),
            ]
        ),
        tools=[currency_convert],
    )

    result = agent.invoke({"messages": [human("10 美元折多少人民币")]})
    tool_messages = [m for m in result["messages"] if m.type == "tool"]

    ensure(tool_messages, "应当产生一条工具消息")
    ensure(
        "72.00 CNY" in str(tool_messages[0].content),
        f"自定义工具应当真的被调用，实际 {tool_messages[0].content!r}",
    )
    print(f"  工具返回：{tool_messages[0].content}")
    print("  ✓ 自定义工具进 harness 后和内置工具没有区别")


def check_subagent_delegation() -> None:
    """``task`` 工具能把活派给子智能体，主智能体只收回一份报告。"""
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=deep_model(
            [
                tool_call(
                    DELEGATION_TOOL,
                    {
                        "description": "调研 LangGraph 的持久化机制",
                        "subagent_type": "general-purpose",
                    },
                ),
                ai("子智能体调研完毕：LangGraph 靠检查点器实现持久化。"),
                ai("父智能体汇总完毕。"),
            ]
        ),
        tools=[],
    )

    result = agent.invoke({"messages": [human("调研一下 LangGraph 的持久化")]})
    tool_messages = [m for m in result["messages"] if m.type == "tool"]

    ensure(tool_messages, "委派应当产生一条工具消息")
    ensure(
        "调研完毕" in str(tool_messages[0].content),
        f"子智能体的报告应当回到主智能体，实际 {tool_messages[0].content!r}",
    )
    ensure(
        result["messages"][-1].content == "父智能体汇总完毕。",
        f"主智能体应当在子智能体之后继续，实际 {result['messages'][-1].content!r}",
    )
    print(f"  子智能体报告：{tool_messages[0].content}")
    print("  ✓ 委派是主智能体的一步，中间过程被隔离在子智能体的上下文里")


class InvoiceState(TypedDict):
    """一张发票的处理状态。

    放在模块层而不是函数里：``add_conditional_edges`` 会用
    ``typing.get_type_hints`` 解析路由函数的返回注解，局部定义的类解析不到。
    """

    amount: float
    trace: Annotated[list[str], operator.add]
    """走过的节点，用来断言流程顺序。"""


def check_downgrade_to_langgraph() -> None:
    """对照：同样的确定性需求，用原生 LangGraph 写清楚每条边。"""
    from langgraph.graph import END, START, StateGraph

    def screen(state: InvoiceState) -> dict[str, Any]:
        return {"trace": ["screen"]}

    def review(state: InvoiceState) -> dict[str, Any]:
        return {"trace": ["review"]}

    def settle(state: InvoiceState) -> dict[str, Any]:
        return {"trace": ["settle"]}

    def route(state: InvoiceState) -> str:
        # 分支条件由业务规则决定，不由模型决定。
        return "review" if state["amount"] > 1000 else "settle"

    builder = StateGraph(InvoiceState)
    builder.add_node("screen", screen)
    builder.add_node("review", review)
    builder.add_node("settle", settle)
    builder.add_edge(START, "screen")
    builder.add_conditional_edges("screen", route, {"review": "review", "settle": "settle"})
    builder.add_edge("review", "settle")
    builder.add_edge("settle", END)
    graph = builder.compile()

    big = graph.invoke({"amount": 5000, "trace": []})
    small = graph.invoke({"amount": 100, "trace": []})

    ensure(big["trace"] == ["screen", "review", "settle"], f"大额要走审批，实际 {big['trace']}")
    ensure(small["trace"] == ["screen", "settle"], f"小额直接放行，实际 {small['trace']}")

    edges = [(edge.source, edge.target) for edge in graph.get_graph().edges]
    print(f"  大额路径：{big['trace']}")
    print(f"  小额路径：{small['trace']}")
    print(f"  图的边：{edges}")
    print("  ✓ 需要精确控制每条边时，原生 LangGraph 才是对的工具")


# ------------------------------------------------------------ 入口


def run() -> None:
    """跑第 14 章实验。"""
    banner("第 14 章 · Deep Agents：harness 视角")
    check_agent_graph_and_state_channels()
    check_preset_capabilities_are_enumerable()
    check_planning_lands_in_state()
    check_filesystem_lands_in_state()
    check_custom_tool_is_called()
    check_subagent_delegation()
    check_downgrade_to_langgraph()
    rule("harness 替你写好了脚手架，替你做不了决策")
    print("\n✓ 第 14 章通过")


def run_live() -> None:
    """真实模型路径：让真模型用 harness 的预置能力完成一个小任务。

    断言只校验形状（有待办、有文件、有回答），不校验内容。
    """
    from langchain.agents.middleware import TodoListMiddleware

    from deepagents import create_deep_agent

    from ..labkit import real_chat_model

    banner("第 14 章 · 真实模型路径")

    agent = create_deep_agent(
        model=real_chat_model(),
        tools=[],
        middleware=[TodoListMiddleware()],
    )

    result = agent.invoke(
        {"messages": [human("把今天要做的事列成一个待办，然后写进 /todo.md。")]}
    )

    print(f"  todos：{result.get('todos')}")
    print(f"  files：{list((result.get('files') or {}).keys())}")
    print(f"  最后一条：{str(result['messages'][-1].content)[:160]}")

    ensure(result.get("messages"), "真实模型路径应当至少有一条消息")
    ensure(
        bool(str(result["messages"][-1].content).strip()),
        "真实模型路径应当给出非空回答",
    )
