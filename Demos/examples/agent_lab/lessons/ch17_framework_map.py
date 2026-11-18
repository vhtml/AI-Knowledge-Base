"""第 17 章实验：框架地图——手写运行时每一环对应框架的哪个概念。

这个模块解决什么问题：第 17 章是整套教程的收口，要回答"框架到底替我干了什么"。
这不能靠读文档拍脑袋，必须能跑出来看。所以本章把四件事变成可执行的断言：

1. **映射表**——手写运行时的每一环对应框架的哪个概念，
   以及框架额外给了哪三样我手写时**没做**的东西；
2. **顺序写 vs 图写**——同一段 Agent 逻辑写成 while 循环和写成"节点 + 边"，
   跑出来的工具调用序列完全一致，但**声明成本**不一样；
3. **图式的额外成本**——每个状态键都要显式决定归约器语义，这是手写版没有的；
4. **上框架判据与迁移清单**——可执行的判断函数 + 清单校验。

**本章刻意不 import 任何框架**：框架留到对比说明里，代码只对比"两种写法的成本"。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    tool_result,
)

# ---------------------------------------------------------------- 1. 映射表

#: 手写运行时的一环 → 框架里的对应概念
MAPPING: tuple[tuple[str, str], ...] = (
    ("Message 列表", "MessagesState 的 messages 通道"),
    ("Agent.run 的 while 循环", "StateGraph 的 model/tools 节点 + 条件边"),
    ("AgentConfig.max_iterations", "recursion_limit"),
    ("Agent._run_tool", "ToolNode"),
    ("三类失败的错误措辞", "handle_tool_errors"),
    ("AgentResult.steps", "stream_mode='updates' 的事件流"),
    ("上下文预算与压缩", "中间件（摘要中间件）"),
    ("工具 schema 生成", "@tool / StructuredTool"),
    ("ApprovalGate", "interrupt() + Command(resume=...)"),
    ("MemoryStore", "Store / 外部数据库"),
    ("TraceRecorder", "LangSmith tracing"),
    ("黄金集与回归", "LangSmith dataset + evaluator"),
)

#: 框架额外给的、手写运行时**没有做**的三件事
FRAMEWORK_ONLY: tuple[str, ...] = (
    "Checkpointer（持久化、断点续跑、时间旅行）",
    "interrupt()（中断与人在环路恢复）",
    "Studio / 本地服务器（可视化调试与部署）",
)


def check_mapping_table() -> None:
    """映射表要覆盖手写运行时的每一环，并明确列出框架额外给的三样。"""
    section("映射表：每一环对应框架的哪个概念")

    for mine, theirs in MAPPING:
        print(f"  {mine:<28} → {theirs}")

    ensure(
        len(MAPPING) == 12,
        f"映射表应覆盖 12 个手写概念，实际 {len(MAPPING)}",
    )
    ensure(
        all(mine and theirs for mine, theirs in MAPPING),
        "映射表不允许有空项",
    )

    print()
    print("  框架额外给的、手写运行时没做的三样：")
    for item in FRAMEWORK_ONLY:
        print(f"    + {item}")
    ensure(
        len(FRAMEWORK_ONLY) == 3,
        f"应当正好是三件，实际 {len(FRAMEWORK_ONLY)}",
    )

    # 关键判断：这三样恰好是"最费劲"的三件
    ensure(
        any("持久化" in item for item in FRAMEWORK_ONLY),
        "持久化必须在框架额外清单里",
    )
    ensure(
        any("中断" in item for item in FRAMEWORK_ONLY),
        "中断恢复必须在框架额外清单里",
    )


# ---------------------------------------------------------------- 2. 顺序写 vs 图写


@dataclass
class GraphSpec:
    """一张极简"图"的描述：节点、边、条件边。

    刻意把它做成**声明式数据**，而不是真去跑一个图引擎——
    本章要对比的是"声明成本"，不是引擎性能。
    """

    nodes: dict[str, Callable[[list[Message]], list[Message]]]
    edges: list[tuple[str, str]]
    conditional_edges: list[tuple[str, Callable[[list[Message]], str], dict[str, str]]]
    state_keys: dict[str, str] = field(default_factory=dict)
    """状态键 → 归约器语义（覆盖 / 追加）。**图式必须显式声明，顺序式不用。**"""


def _make_model_node(
    replies: Sequence[ModelReply],
    cursor: list[int],
) -> Callable[[list[Message]], list[Message]]:
    """造一个"模型节点"：追加一条助手消息。"""

    def model_node(messages: list[Message]) -> list[Message]:
        index = min(cursor[0], len(replies) - 1)
        cursor[0] += 1
        return messages + [replies[index].to_message()]

    return model_node


def _make_tool_node(
    tool_map: dict[str, Any],
) -> Callable[[list[Message]], list[Message]]:
    """造一个"工具节点"：执行 tool_calls 并追加工具消息。"""

    def tool_node(messages: list[Message]) -> list[Message]:
        last = messages[-1]
        produced: list[Message] = []
        for call in last.tool_calls:
            entry = tool_map[call.name]
            produced.append(
                tool_result(entry.invoke(call.arguments), call.id, call.name)
            )
        return messages + produced

    return tool_node


def check_sequential_vs_graph() -> None:
    """同一段逻辑，顺序写和图写跑出来的轨迹必须一致。"""
    section("顺序写 vs 图写：轨迹一样，声明成本不同")

    @tool
    def 加(a: int, b: int) -> int:
        """两数相加。

        Args:
            a: 第一个加数
            b: 第二个加数
        """
        return a + b

    script = [
        ModelReply(tool_calls=[ToolCall("call_1", "加", {"a": 2, "b": 3})]),
        ModelReply(content="和是 5"),
    ]

    # ---- 顺序式：while 循环，状态就是一个 list
    cursor = [0]
    model = scripted_model(script)

    def sequential(messages: list[Message]) -> list[Message]:
        """顺序式：while 循环。"""
        history = list(messages)
        for _ in range(8):
            reply = model.complete(history)
            history.append(reply.to_message())
            if not reply.tool_calls:
                return history
            for call in reply.tool_calls:
                history.append(tool_result(加.invoke(call.arguments), call.id, call.name))
        return history

    seq_result = sequential([human("2+3 等于几？")])

    # ---- 图式：节点 + 边 + 条件边
    graph_cursor = [0]
    graph = GraphSpec(
        nodes={
            "model": _make_model_node(script, graph_cursor),
            "tools": _make_tool_node({"加": 加}),
        },
        edges=[("tools", "model")],
        conditional_edges=[
            ("model", _route_tool_or_end, {"tools": "tools", END: END}),
        ],
        state_keys={"messages": "追加", "result": "覆盖"},
    )

    graph_result = _run_graph(graph, [human("2+3 等于几？")])

    seq_tools = [m for m in seq_result if m.role == "tool"]
    graph_tools = [m for m in graph_result if m.role == "tool"]

    print(f"  顺序式：{len(seq_result)} 条消息，工具结果 {len(seq_tools)} 条")
    print(f"  图式  ：{len(graph_result)} 条消息，工具结果 {len(graph_tools)} 条")

    ensure(
        len(seq_result) == len(graph_result),
        f"两种写法的消息条数应一致：顺序式 {len(seq_result)} vs 图式 {len(graph_result)}",
    )
    ensure(
        [m.content for m in seq_tools] == [m.content for m in graph_tools],
        "两种写法的工具结果应当一致",
    )
    ensure(
        seq_result[-1].content == graph_result[-1].content == "和是 5",
        f"两种写法的最终回答都应是 '和是 5'，实际 {seq_result[-1].content!r} / {graph_result[-1].content!r}",
    )
    ensure(
        seq_tools[0].tool_call_id == graph_tools[0].tool_call_id == "call_1",
        "两种写法的 tool_call_id 都应回指同一次调用",
    )

    # 声明成本：图式必须显式声明每个状态键的归约器语义
    print()
    print(f"  图式显式声明的状态键：{graph.state_keys}")
    ensure(
        len(graph.state_keys) >= 1,
        "图式必须显式声明状态键的归约器语义",
    )
    ensure(
        "result" in graph.state_keys,
        "覆盖型状态键也必须在图式里显式声明",
    )


END = "__end__"


def _route_tool_or_end(messages: list[Message]) -> str:
    """条件边：模型这次要不要调工具。"""
    last = messages[-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def _run_graph(graph: GraphSpec, initial: list[Message]) -> list[Message]:
    """跑一张极简"图"。

    只支持"条件边 + 固定边"两种，够本章对比用了。
    真正的图引擎（LangGraph 的 Pregel）要处理 super-step、并行分支、
    检查点、中断——那些正是第 17 章说"框架额外给的"东西。
    """
    messages = list(initial)
    node = "model"
    visited = 0

    while visited < 20:
        visited += 1
        if node == END:
            break
        messages = graph.nodes[node](messages)

        next_node = None
        for source, router, mapping in graph.conditional_edges:
            if source == node:
                choice = router(messages)
                next_node = mapping.get(choice, choice)
                break
        if next_node is None:
            for source, target in graph.edges:
                if source == node:
                    next_node = target
                    break
        if next_node is None:
            break
        node = next_node

    return messages


# ---------------------------------------------------------------- 3. 图式的额外成本


def check_graph_extra_cost() -> None:
    """图式多出来的成本：状态键的归约器决定，顺序式一个都不用做。"""
    section("图式的额外成本：每个状态键都要显式决定归约器")

    # 顺序式：状态就是一个 dict，改它就是赋值
    sequential_state: dict[str, Any] = {"messages": [], "result": None, "steps": []}
    sequential_state["result"] = "和是 5"
    sequential_state["steps"].append("model")
    ensure(
        sequential_state["result"] == "和是 5",
        "顺序式的状态赋值应当直接生效",
    )
    print(f"  顺序式状态：{sorted(sequential_state)}")
    print("  → 归约器语义：隐含在代码里（赋值就是覆盖，append 就是累加）")

    # 图式：每个键都要显式声明
    graph_state_keys = {
        "messages": "add_messages（按 id 追加与更新）",
        "result": "覆盖（默认）",
        "steps": "operator.add（累加）",
    }
    for key, reducer in graph_state_keys.items():
        print(f"  图式状态键 {key:<10} → 归约器：{reducer}")

    ensure(
        len(graph_state_keys) == 3,
        f"三个状态键都应显式声明归约器，实际 {len(graph_state_keys)}",
    )
    ensure(
        graph_state_keys["messages"] != graph_state_keys["steps"],
        "messages 与 steps 的归约器语义不同，必须显式区分——这正是顺序式不用管的事",
    )
    ensure(
        graph_state_keys["result"] == "覆盖（默认）",
        "result 是覆盖语义",
    )

    print("  → 图式多出来的成本 = 状态键数 × 一次语义决策")


# ---------------------------------------------------------------- 4. 上框架判据


def check_when_to_use_framework() -> None:
    """上框架的判据要能执行：说得出框架替你干了哪件事才上。"""
    section("上框架判据：能不能一句话说清框架替你干了什么")

    def should_use_framework(
        needs_resume: bool,
        needs_approval: bool,
        needs_multi_agent: bool,
        needs_visual_debug: bool,
        steps_known: bool,
    ) -> bool:
        """判据：只要有一项"框架的独有能力"是刚需，就该上框架。

        ``steps_known`` 为 True 时说明流程是固定的——那用工作流更合适，
        不该为了框架而框架。
        """
        if steps_known:
            return False
        return needs_resume or needs_approval or needs_multi_agent or needs_visual_debug

    cases = [
        (True, False, False, False, False, True, "要断点续跑 → 上"),
        (False, True, False, False, False, True, "要审批 → 上"),
        (False, False, True, False, False, True, "要多 Agent → 上"),
        (False, False, False, True, False, True, "要可视化调试 → 上"),
        (False, False, False, False, True, False, "步骤已知 → 不上，用工作流"),
        (False, False, False, False, False, False, "只是简单调用 → 不上"),
    ]
    for resume, approval, multi, visual, known, expected, why in cases:
        actual = should_use_framework(resume, approval, multi, visual, known)
        print(f"  {why:<28} → {'上框架' if actual else '不上'}")
        ensure(actual == expected, f"{why} 判定应为 {expected}，实际 {actual}")

    ensure(
        not should_use_framework(False, False, False, False, True),
        "步骤已知时不该上框架",
    )


# ---------------------------------------------------------------- 5. 迁移清单


def check_migration_checklist() -> None:
    """迁移清单里"零改动"的那几条，现在就必须成立。"""
    section("迁移清单：能原样带走的，现在就该成立")

    @tool
    def 查订单(order_id: str) -> str:
        """按订单号查询订单状态。

        Args:
            order_id: 订单号
        """
        return f"{order_id} 已发货"

    # 工具函数与 schema 零改动 → 序列化两次应当完全一致
    schema_once = 查订单.to_api()
    schema_twice = 查订单.to_api()
    ensure(
        schema_once == schema_twice,
        "工具声明的序列化必须稳定——框架包装前后要能逐字比对",
    )
    print(f"  工具 schema 稳定：{schema_once['function']['name']}")

    # 系统提示零改动 → 放进 messages 后原样保留
    system_prompt = "你是一个订单助手。"
    messages = [Message(role="system", content=system_prompt)]
    ensure(
        messages[0].content == system_prompt,
        "系统提示放进消息数组后必须原样保留",
    )
    ensure(
        messages[0].role == "system",
        "系统提示必须在第一位",
    )
    print("  系统提示零改动")

    # 错误分类：三类仍在，且措辞不同
    from ..runtime import Agent, AgentConfig

    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "不存在的工具", {})]),
            ModelReply(content="我换用别的工具。"),
        ]
    )
    agent = Agent(model, [], AgentConfig())
    result = agent.run("做点什么")
    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(
        len(tool_messages) == 1,
        f"应当回填一条工具消息，实际 {len(tool_messages)}",
    )
    ensure(
        "没有名为" in tool_messages[0].content,
        f"工具不存在的措辞应当保留，实际 {tool_messages[0].content!r}",
    )
    print(f"  错误分类保留：{tool_messages[0].content[:40]}…")

    # 循环终止条件：语义要对齐 recursion_limit
    ensure(
        agent.config.max_iterations == 8,
        f"默认轮次上限应当是 8，实际 {agent.config.max_iterations}",
    )
    print(f"  循环上限 {agent.config.max_iterations}（对应框架的 recursion_limit）")


def run_live() -> None:
    """真实模型路径：真模型跑一轮，确认手写运行时在真模型下也能工作。"""
    banner("第 17 章 · 真实模型路径")
    from ..labkit import real_model
    from ..runtime import Agent, AgentConfig

    @tool
    def 加(a: int, b: int) -> int:
        """两数相加。

        Args:
            a: 第一个加数
            b: 第二个加数
        """
        return a + b

    agent = Agent(real_model(), [加], AgentConfig(max_iterations=4))
    result = agent.run("用加法工具算一下 17 加 25")

    for message in result.messages:
        print(f"  {message.role}: {getattr(message, 'content', '')!r}")

    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "工具消息必须带 tool_call_id",
    )
    print(f"  结束原因：{result.stop_reason}")


def run() -> None:
    """跑第 17 章实验。"""
    banner("第 17 章 · 框架地图：从手写运行时到 LangGraph 与 Deep Agents")
    check_mapping_table()
    check_sequential_vs_graph()
    check_graph_extra_cost()
    check_when_to_use_framework()
    check_migration_checklist()
    rule("框架替你做决定；手写 16 章换来的是「我知道每一环为什么这样」")
    print("\n✓ 第 17 章通过")
