"""第 2 章实验：图模型——节点、边、共享状态与 super-step。

这个模块解决什么问题：把"状态机不是流程图"这句话变成能看见的东西：

1. 节点只返回**部分更新**，没被写的键保持原值；
2. 直接改 state 入参不生效（这是最常见的低级错误）；
3. 一次 ``invoke`` 内部被切成若干 **super-step**，并行节点互相看不到对方的更新；
4. 条件边只选路、不改状态；
5. 图能导出 Mermaid 拓扑。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..labkit import banner, ensure, rule


class State(TypedDict):
    """第 2 章用的状态。

    ``text`` 是普通键（覆盖语义），``steps`` 用累加归约器（追加语义）。
    归约器是第 3 章的主题，这里先用上，才能演示并行分支的合并。
    """

    text: str
    steps: Annotated[list[str], operator.add]


def append_step(state: State) -> dict[str, Any]:
    """节点示例：只返回自己改的那个键。"""
    return {"text": state["text"].strip()}


def shout(state: State) -> dict[str, Any]:
    """节点示例：基于当前状态产出新值。"""
    return {"text": state["text"].upper(), "steps": ["shout"]}


def mutate_in_place(state: State) -> dict[str, Any]:
    """反面示例：试图直接改 state 入参。

    LangGraph 不会把入参的改动写回状态——节点必须返回更新。
    """
    state["text"] = "被偷偷改了"
    return {}


def check_partial_update() -> None:
    """节点只写自己改的键，其余保持原值。"""
    print("--- 部分更新语义 ---")
    builder = _build_simple_graph()
    graph = builder.compile()

    result = graph.invoke({"text": "  hello  ", "steps": []})
    print(f"  invoke 结果：{result}")
    ensure(result["text"] == "HELLO", f"text 应当被 shout 改写，实际 {result['text']}")
    ensure(result["steps"] == ["shout"], f"steps 应当追加，实际 {result['steps']}")


def check_in_place_mutation_ignored() -> None:
    """直接改 state 入参不生效。"""
    print("--- 入参改写无效 ---")
    builder = _build_simple_graph()
    builder.add_node("mutate", mutate_in_place)
    builder.add_edge("shout", "mutate")
    graph = builder.compile()

    result = graph.invoke({"text": "hello", "steps": []})
    print(f"  偷偷改写后的结果：{result}")
    ensure(
        result["text"] == "HELLO",
        "节点直接改 state 不该生效——必须返回更新字典",
    )


def check_super_steps() -> None:
    """一次 invoke 被切成若干 super-step，并行分支互相看不见对方。"""
    print("--- super-step 与并行分支 ---")
    from langgraph.graph import END, START, StateGraph

    seen: dict[str, list[str]] = {}

    def left(state: State) -> dict[str, Any]:
        seen.setdefault("left", []).append(str(state["steps"]))
        return {"steps": ["left"]}

    def right(state: State) -> dict[str, Any]:
        seen.setdefault("right", []).append(str(state["steps"]))
        return {"steps": ["right"]}

    def merge(state: State) -> dict[str, Any]:
        seen.setdefault("merge", []).append(str(state["steps"]))
        return {}

    builder = StateGraph(State)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("merge", merge)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", "merge")
    builder.add_edge("right", "merge")
    builder.add_edge("merge", END)
    graph = builder.compile()

    result = graph.invoke({"text": "x", "steps": []})
    print(f"  各节点看到的 steps：{seen}")
    print(f"  最终 steps：{result['steps']}")

    ensure(
        seen["left"] == ["[]"] and seen["right"] == ["[]"],
        f"left/right 同属一个 super-step，应都只看到初始状态，实际 {seen}",
    )
    merged = eval(seen["merge"][0]) if seen["merge"] else []
    ensure(
        sorted(merged) == ["left", "right"],
        f"merge 在下一个 super-step，应看到两个分支合并后的状态，实际 {seen['merge']}",
    )
    ensure(
        sorted(result["steps"]) == ["left", "right"],
        f"归约器应把两个分支的结果合并起来，实际 {result['steps']}",
    )


def check_conditional_edge_routes_only() -> None:
    """条件边只负责选路，不改状态。"""
    print("--- 条件边只选路 ---")
    from langgraph.graph import END, START, StateGraph

    def router_node(state: State) -> dict[str, Any]:
        """节点本身必须返回字典——路由判断放在另一个函数里。"""
        return {}

    def route(state: State) -> str:
        """路由函数：只决定去哪个节点，不改状态。"""
        return "long" if len(state["text"]) > 3 else "short"

    def long_node(state: State) -> dict[str, Any]:
        return {"text": state["text"] + "-长"}

    def short_node(state: State) -> dict[str, Any]:
        return {"text": state["text"] + "-短"}

    builder = StateGraph(State)
    builder.add_node("router", router_node)
    builder.add_node("long", long_node)
    builder.add_node("short", short_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route, ["long", "short"])
    builder.add_edge("long", END)
    builder.add_edge("short", END)
    graph = builder.compile()

    result = graph.invoke({"text": "abcd", "steps": []})
    print(f"  长文本结果：{result['text']}")
    ensure(result["text"] == "abcd-长", f"应走 long 分支，实际 {result['text']}")

    result = graph.invoke({"text": "ab", "steps": []})
    print(f"  短文本结果：{result['text']}")
    ensure(result["text"] == "ab-短", f"应走 short 分支，实际 {result['text']}")


def check_mermaid_topology() -> None:
    """图能导出 Mermaid 拓扑。"""
    print("--- Mermaid 拓扑 ---")
    graph = _build_simple_graph().compile()
    mermaid = graph.get_graph().draw_mermaid()
    print("  " + mermaid.replace("\n", "\n  "))
    # 入口/出口节点在 Mermaid 里渲染成 __start__ / __end__
    ensure("shout" in mermaid, "Mermaid 里应该出现节点名 shout")
    ensure("clean" in mermaid, "Mermaid 里应该出现节点名 clean")
    ensure("__start__" in mermaid, f"Mermaid 里应该出现入口节点，实际 {mermaid[:200]}")
    ensure("-->" in mermaid, "Mermaid 里应该出现边的连线")


def _build_simple_graph() -> Any:
    """搭一张"清洗 → 大写"的两节点图。"""
    from langgraph.graph import START, StateGraph

    builder = StateGraph(State)
    builder.add_node("clean", append_step)
    builder.add_node("shout", shout)
    builder.add_edge(START, "clean")
    builder.add_edge("clean", "shout")
    return builder


def run() -> None:
    """跑第 2 章实验。"""
    banner("第 2 章 · 图模型：节点、边与共享状态")
    check_partial_update()
    check_in_place_mutation_ignored()
    check_super_steps()
    check_conditional_edge_routes_only()
    check_mermaid_topology()
    rule("节点返回更新、边只选路、super-step 是并行可见性的边界")
    print("\n✓ 第 2 章通过")
