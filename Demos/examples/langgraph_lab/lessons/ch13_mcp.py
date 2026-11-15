"""第 13 章实验：MCP 接入与工具生态——把一个真的 MCP 服务器接进 LangGraph 图。

这个模块解决什么问题：第 13 章的核心主张是"MCP 工具进图的方式和本地 ``@tool`` 完全一样"。
只靠读文档很容易停留在"听说过"，这里用可运行的断言把它钉死：

1. **真连一个服务器**——在本进程内用 FastMCP 起一个最小 MCP 服务器，
   不连公网、不起子进程，用 ``MCPAdapter`` 发现它的工具目录；
2. **两条接入路径都跑通**——MCP 工具既能当普通 LangChain 工具在节点里直接调，
   也能当 ``ToolNode`` 的工具集交给图执行；
3. **错误语义不变**——服务器报的错变成 ``status="error"`` 的 ``ToolMessage``，
   而不是把异常抛给模型；
4. **beta 状态可见**——``langchain.mcp`` 导入时确实会抛 ``LangChainBetaWarning``。

全程离线：不联网、不调模型。
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
from typing import Any, TypedDict

from langchain_core.messages import AIMessage

from ..labkit import banner, ensure, human, rule, tool_call

# ------------------------------------------------------------ 本地 MCP 服务器


def build_server() -> Any:
    """在本进程内搭一个最小 MCP 服务器。

    为什么不用公网服务器：教学实验必须能离线复现，而且"进程内服务器"
    是 ``MCPAdapter`` 支持的最短路径——没有子进程、没有端口、没有网络。

    为什么工具都写中文 docstring：docstring 会变成模型看到的工具描述，
    它和函数签名一起构成 MCP 协议里的 tool schema。
    """

    from fastmcp import FastMCP

    server = FastMCP("langgraph-lab")

    @server.tool
    def add(a: int, b: int) -> int:
        """两数相加，返回和。"""
        return a + b

    @server.tool
    def divide(a: float, b: float) -> float:
        """两数相除。除数为 0 时报错，用来演示 MCP 的错误语义。"""
        if b == 0:
            raise ValueError("除数不能为 0")
        return a / b

    return server


async def discover_tools(server: Any) -> list[Any]:
    """在 adapter 上下文里发现服务器的工具。

    工具对象自己持有 client，退出上下文之后仍然可调用（官方文档明确的行为），
    所以"发现"和"调用"可以分处两个 ``async with``。
    """
    from langchain.mcp import MCPAdapter

    async with MCPAdapter(server) as adapter:
        return await adapter.list_tools()


def pick(tools: list[Any], name: str) -> Any:
    """按名字挑一个工具，找不到就把可用名字列出来，便于排查。"""
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"工具 {name} 不存在，实际有：{[t.name for t in tools]}")


def texts(payload: Any) -> list[str]:
    """从 MCP 返回的内容块里抽出所有文本。

    MCP 结果落到 LangChain 侧是标准 content block 列表，文本块在 ``text`` 字段；
    图片、文件等块模型自己看，这里只关心文字。
    """
    if not isinstance(payload, list):
        return [str(payload)]
    return [
        block.get("text", "")
        for block in payload
        if isinstance(block, dict) and block.get("type") == "text"
    ]


# ------------------------------------------------------------ 断言项


def check_namespace_is_beta() -> None:
    """``langchain.mcp`` 处在 beta：导入时会抛 ``LangChainBetaWarning``。

    为什么用 reload 重新触发一次：官方说这个警告"每进程只报一次"，
    而自检会把 17 章跑在同一个进程里，不能指望第一次导入一定发生在这里。
    """
    import langchain.mcp as mcp_module
    from langchain_core._api import LangChainBetaWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(mcp_module)

    hits = [w for w in caught if issubclass(w.category, LangChainBetaWarning)]
    ensure(hits, "重新导入 langchain.mcp 应当抛 LangChainBetaWarning")
    ensure(
        hasattr(mcp_module, "MCPAdapter"),
        "langchain.mcp 应当导出 MCPAdapter",
    )
    print(f"  警告原文：{hits[0].message}")
    print("  ✓ beta 状态是显式的，升级前要盯着这个命名空间")


async def check_tools_are_discovered() -> list[Any]:
    """适配器把服务器的工具变成标准 LangChain 工具。"""
    tools = await discover_tools(build_server())
    names = sorted(tool.name for tool in tools)

    ensure("add" in names, f"应当发现 add 工具，实际 {names}")
    ensure("divide" in names, f"应当发现 divide 工具，实际 {names}")

    add_tool = pick(tools, "add")
    ensure(
        add_tool.description == "两数相加，返回和。",
        f"docstring 应当变成工具描述，实际 {add_tool.description!r}",
    )
    ensure(
        set(add_tool.args) == {"a", "b"},
        f"工具参数 schema 应当来自函数签名，实际 {add_tool.args}",
    )
    print(f"  发现工具：{names}")
    print(f"  add 的参数 schema：{add_tool.args}")
    print("  ✓ MCP 工具在 LangChain 侧就是普通 StructuredTool")
    return tools


async def check_tool_works_as_node_tool() -> None:
    """路径 A：MCP 工具当普通 LangChain 工具，在节点里直接调用。"""
    from langgraph.graph import END, START, StateGraph

    tools = await discover_tools(build_server())
    add_tool = pick(tools, "add")

    class State(TypedDict):
        """最小状态：只要一条消息。"""

        messages: list[Any]

    async def call_mcp(state: State) -> dict[str, Any]:
        # MCP 工具和本地 @tool 一样，签名是 ``ainvoke``/``invoke``。
        payload = await add_tool.ainvoke({"a": 2, "b": 40})
        return {"messages": [AIMessage(content=" | ".join(texts(payload)))]}

    builder = StateGraph(State)
    builder.add_node("call_mcp", call_mcp)
    builder.add_edge(START, "call_mcp")
    builder.add_edge("call_mcp", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [human("2 + 40 等于多少？")]})
    answer = result["messages"][-1].content

    ensure("42" in answer, f"MCP 工具应当算出 42，实际 {answer!r}")
    print(f"  节点直接调用 MCP 工具 → {answer}")
    print("  ✓ 路径 A：MCP 工具在节点里就是一个可 await 的 LangChain 工具")


async def check_tool_works_in_tool_node() -> None:
    """路径 B：MCP 工具当 ``ToolNode`` 的工具集，由图来执行。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    tools = await discover_tools(build_server())
    tool_node = ToolNode(tools)

    class State(TypedDict):
        """最小状态：只要一条消息。"""

        messages: list[Any]

    def planner(state: State) -> dict[str, Any]:
        # 脚本化模型替模型做了决定：调用 add。
        return {"messages": [tool_call("add", {"a": 2, "b": 40})]}

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [human("2 + 40 等于多少？")]})
    tool_message = result["messages"][-1]

    ensure(
        tool_message.status == "success",
        f"ToolMessage 应当是 success，实际 {tool_message.status}",
    )
    ensure(
        "42" in " ".join(texts(tool_message.content)),
        f"工具结果应当包含 42，实际 {tool_message.content!r}",
    )
    artifact = tool_message.artifact or {}
    ensure(
        artifact.get("structured_content") == {"result": 42},
        f"结构化结果应当挂在 artifact 上，实际 {artifact!r}",
    )
    print(f"  ToolMessage 状态：{tool_message.status}")
    print(f"  artifact：{artifact}")
    print("  ✓ 路径 B：MCP 工具直接进 ToolNode，模型可见的工具面完全一致")


async def check_error_becomes_failed_tool_message() -> None:
    """服务器报的错变成 ``status="error"`` 的 ToolMessage，模型能读到并自我纠正。"""
    import logging

    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    # FastMCP 会把服务器端异常用 rich traceback 打到 stderr。
    # 这里的失败是故意制造的，traceback 只会淹没实验输出，所以临时掐掉日志。
    fastmcp_logger = logging.getLogger("fastmcp")
    previous_level = fastmcp_logger.level
    fastmcp_logger.setLevel(logging.CRITICAL)
    try:
        await _run_error_case()
    finally:
        fastmcp_logger.setLevel(previous_level)


async def _run_error_case() -> None:
    """真正跑一次除零，把失败转成 ToolMessage。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    tools = await discover_tools(build_server())
    tool_node = ToolNode(tools)

    class State(TypedDict):
        """最小状态：只要一条消息。"""

        messages: list[Any]

    def planner(state: State) -> dict[str, Any]:
        return {"messages": [tool_call("divide", {"a": 10, "b": 0})]}

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [human("10 除以 0")]})
    tool_message = result["messages"][-1]

    ensure(
        tool_message.status == "error",
        f"失败的 MCP 工具应当产出 status=error，实际 {tool_message.status}",
    )
    ensure(
        "除数不能为 0" in " ".join(texts(tool_message.content)),
        f"服务器自己的错误信息应当透传给模型，实际 {tool_message.content!r}",
    )
    print(f"  错误 ToolMessage：{tool_message.content}")
    print("  ✓ 服务器错误不会炸掉整张图，模型能读到并换参数重试")


async def check_mcp_metadata_is_attached() -> None:
    """工具元数据里带 MCP 溯源信息，可以据此做策略。"""
    tools = await discover_tools(build_server())
    metadata = pick(tools, "add").metadata or {}
    mcp_meta = metadata.get("mcp", {})

    ensure("mcp" in metadata, f"工具元数据应当带 mcp 命名空间，实际 {metadata!r}")
    ensure(
        mcp_meta.get("server", {}).get("name") == "langgraph-lab",
        f"元数据应当标明服务器身份，实际 {mcp_meta!r}",
    )
    print(f"  工具元数据：{metadata}")
    print("  ✓ 服务器自报的注解（read_only_hint 等）就在 tool.metadata 里")


async def check_single_tool_conversion() -> None:
    """``as_langchain_tool``：只把一个 MCP 工具包成 LangChain 工具。"""
    from fastmcp.client import Client
    from langchain.mcp import as_langchain_tool

    client = Client(build_server())
    async with client:
        mcp_tools = await client.list_tools()

    divide_tool = next(t for t in mcp_tools if t.name == "divide")
    langchain_tool = await as_langchain_tool(divide_tool, client)

    ensure(
        langchain_tool.name == "divide",
        f"转换后工具名应当保持不变，实际 {langchain_tool.name}",
    )
    payload = await langchain_tool.ainvoke({"a": 9, "b": 3})
    ensure(
        "3" in " ".join(texts(payload)),
        f"转换后的工具应当能正常调用，实际 {payload!r}",
    )
    print(f"  转换前：{type(divide_tool).__name__} → 转换后：{type(langchain_tool).__name__}")
    print(f"  调用结果：{payload}")
    print("  ✓ 只要一个工具时不用起整个 adapter")


# ------------------------------------------------------------ 入口


async def _run_async_checks() -> None:
    """所有需要事件循环的检查集中在这里。

    为什么合成一次 ``asyncio.run``：FastMCP 的进程内传输绑定在事件循环上，
    分多次调用容易踩到跨循环的问题。
    """
    await check_tools_are_discovered()
    await check_tool_works_as_node_tool()
    await check_tool_works_in_tool_node()
    await check_error_becomes_failed_tool_message()
    await check_mcp_metadata_is_attached()
    await check_single_tool_conversion()


def run() -> None:
    """跑第 13 章实验。"""
    banner("第 13 章 · MCP 接入与工具生态")
    check_namespace_is_beta()
    asyncio.run(_run_async_checks())
    rule("MCP 工具进图 = 普通 LangChain 工具进图，剩下的只是生命周期管理")
    print("\n✓ 第 13 章通过")


def run_live() -> None:
    """真实模型路径：让真模型自己决定调 MCP 工具。

    断言只校验形状（工具被发现、模型给出了回答），不校验内容——
    真模型答得对不对是评测集该管的事，不是运行路径该管的事。
    """
    from langchain.agents import create_agent

    from ..labkit import real_chat_model

    banner("第 13 章 · 真实模型路径")

    model = real_chat_model()

    async def main() -> dict[str, Any]:
        tools = await discover_tools(build_server())
        print(f"  服务器工具：{sorted(t.name for t in tools)}")
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [human("请用 add 工具算 2 + 40，只回答结果数字。")]}
        )

    result = asyncio.run(main())
    messages = result.get("messages", [])
    tool_calls = [
        call
        for message in messages
        for call in getattr(message, "tool_calls", []) or []
    ]

    for message in messages:
        print(f"  {type(message).__name__}: {str(message.content)[:120]}")

    ensure(messages, "真实模型路径应当至少有一条消息")
    ensure(
        bool(str(messages[-1].content).strip()),
        "真实模型路径应当给出非空回答",
    )
    if tool_calls:
        print(f"  模型真的调了 MCP 工具：{[c['name'] for c in tool_calls]}")
    else:
        print("  模型没有调工具（回答里已包含结论也算通过）")
