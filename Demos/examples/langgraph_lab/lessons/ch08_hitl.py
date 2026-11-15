"""第 8 章实验：人在环路——中断、审批与恢复。

这个模块解决什么问题：把"图能暂停、能等外部输入、能在同一个线程上走出不同分支"变成可执行的断言：

1. 图在 ``interrupt()`` 处真的停下，``invoke`` 的返回值里带着 ``__interrupt__``；
2. 暂停那一刻状态已经落进检查点，从检查点用 ``Command(resume=...)`` 能接着跑完；
3. ``interrupt()`` 的返回值在恢复之前**根本不存在**——节点从开头重跑，中断前的代码会执行两次；
4. approve / edit / reject 三种决策能走出三种不同的结果；
5. 同一个 thread_id 上，"批准"和"拒绝"能走出两条互不相同的分支。

全部离线运行，不联网、不调模型。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..labkit import banner, ensure, rule


class ApprovalState(TypedDict):
    """第 8 章用的状态。

    ``decision`` 与 ``request`` 是覆盖语义，``log`` 用累加归约器——
    同一个线程上跑两轮审批时，``log`` 能把两条分支的痕迹并排放着看。
    """

    request: str
    decision: str
    log: Annotated[list[str], operator.add]


def _build_approval_graph(checkpointer: Any, trace: list[str]) -> Any:
    """搭一张"草拟 → 审批 → 执行/驳回"的图。

    ``trace`` 用来记录节点代码的执行顺序：中断前的代码会在恢复时**再跑一遍**，
    这是第 8 章最反直觉的一条规则，必须让实验自己说话。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def approval(state: ApprovalState) -> dict[str, Any]:
        trace.append("before-interrupt")
        answer = interrupt(
            {
                "question": "是否批准这个动作？",
                "request": state["request"],
                "options": ["approve", "edit", "reject"],
            }
        )
        # 这一行在暂停时根本没执行过——返回值此刻才存在
        trace.append("after-interrupt")

        kind = answer.get("type")
        if kind == "edit":
            return {"decision": "edit", "request": answer["request"], "log": ["edited"]}
        if kind == "reject":
            return {"decision": "reject", "log": ["rejected"]}
        return {"decision": "approve", "log": ["approved"]}

    def execute(state: ApprovalState) -> dict[str, Any]:
        # 副作用放在审批之后的独立节点里：暂停时这个节点还没跑
        return {"log": [f"executed:{state['request']}"]}

    def cancel(state: ApprovalState) -> dict[str, Any]:
        return {"log": [f"cancelled:{state['request']}"]}

    def route(state: ApprovalState) -> str:
        return "cancel" if state["decision"] == "reject" else "execute"

    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval)
    builder.add_node("execute", execute)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "approval")
    builder.add_conditional_edges("approval", route, ["execute", "cancel"])
    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer)


def check_interrupt_pauses_graph() -> None:
    """图在 interrupt 处停下，返回值里带着 __interrupt__，副作用还没发生。"""
    print("--- 中断暂停 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    trace: list[str] = []
    graph = _build_approval_graph(InMemorySaver(), trace)
    config = {"configurable": {"thread_id": "pause-thread"}}

    result = graph.invoke({"request": "转账 500 元", "decision": "", "log": []}, config)
    print(f"  返回值的键：{sorted(result.keys())}")

    ensure("__interrupt__" in result, f"invoke 的结果里应当带 __interrupt__，实际 {sorted(result.keys())}")
    pending = result["__interrupt__"]
    ensure(len(pending) == 1, f"应当只有一个待处理的中断，实际 {len(pending)}")
    print(f"  中断载荷：{pending[0].value}")
    ensure(
        pending[0].value.get("request") == "转账 500 元",
        f"中断载荷里应当带上待审批的内容，实际 {pending[0].value}",
    )

    # 暂停时状态已经落进检查点：下一步该跑谁、谁挂起，都查得到
    snapshot = graph.get_state(config)
    print(f"  快照 next={snapshot.next} values={snapshot.values}")
    ensure(snapshot.next == ("approval",), f"下一步应当还是 approval，实际 {snapshot.next}")
    ensure(
        snapshot.values.get("request") == "转账 500 元",
        f"暂停时待审批内容应当已经落盘，实际 {snapshot.values.get('request')}",
    )
    ensure(
        snapshot.tasks and snapshot.tasks[0].interrupts,
        "检查点里应当记录着这个待处理的中断",
    )
    ensure(
        not any("executed" in item for item in snapshot.values.get("log", [])),
        f"暂停时副作用节点还不该跑，实际 log={snapshot.values.get('log')}",
    )
    ensure(trace == ["before-interrupt"], f"中断之后的代码不该执行，实际 trace={trace}")
    print("  ✓ 停在 super-step 边界，副作用节点还没动")


def check_resume_continues_from_checkpoint() -> None:
    """用 Command(resume=...) 从检查点恢复，一路跑完。"""
    print("--- 恢复执行 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    trace: list[str] = []
    graph = _build_approval_graph(InMemorySaver(), trace)
    config = {"configurable": {"thread_id": "resume-thread"}}

    graph.invoke({"request": "转账 500 元", "decision": "", "log": []}, config)
    final = graph.invoke(Command(resume={"type": "approve"}), config)

    print(f"  跑完的状态：{final}")
    ensure(final["decision"] == "approve", f"决策应当是 approve，实际 {final['decision']}")
    ensure(
        final["log"] == ["approved", "executed:转账 500 元"],
        f"应当先记录审批再执行，实际 {final['log']}",
    )
    ensure(
        graph.get_state(config).next == (),
        "跑完之后不该还有下一步节点",
    )
    print("  ✓ 同一个 thread_id 恢复，跑到终点")


def check_value_only_exists_after_resume() -> None:
    """interrupt 的返回值在恢复之前不可用：节点从开头重跑。"""
    print("--- 返回值的时机 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    trace: list[str] = []
    graph = _build_approval_graph(InMemorySaver(), trace)
    config = {"configurable": {"thread_id": "value-thread"}}

    graph.invoke({"request": "转账 500 元", "decision": "", "log": []}, config)
    print(f"  暂停后 trace={trace}")
    ensure(
        trace == ["before-interrupt"],
        f"暂停时只该跑完中断前的代码，实际 trace={trace}",
    )

    graph.invoke(Command(resume={"type": "approve"}), config)
    print(f"  恢复后 trace={trace}")
    ensure(
        trace == ["before-interrupt", "before-interrupt", "after-interrupt"],
        f"恢复时节点应当从头重跑（中断前代码执行两次），实际 trace={trace}",
    )
    print("  ✓ 恢复前拿不到返回值；恢复时整个节点从头再跑一遍")


def check_graph_interrupt_exception() -> None:
    """interrupt() 靠抛 GraphInterrupt 实现暂停。"""
    print("--- 暂停的机制 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.errors import GraphInterrupt
    from langgraph.graph import START, StateGraph
    from langgraph.types import interrupt

    caught: list[str] = []

    def node(state: ApprovalState) -> dict[str, Any]:
        try:
            interrupt("要批准吗？")
        except GraphInterrupt as exc:
            # 抓住再原样抛出去，等于给中断机制打了个日志
            caught.append(type(exc).__name__)
            raise
        return {"decision": "没中断"}

    builder = StateGraph(ApprovalState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(checkpointer=InMemorySaver())

    graph.invoke({"request": "x", "decision": "", "log": []}, {"configurable": {"thread_id": "exc-thread"}})
    print(f"  捕获到：{caught}")
    ensure(caught == ["GraphInterrupt"], f"中断应当抛 GraphInterrupt，实际 {caught}")


def check_three_decisions() -> None:
    """approve / edit / reject 三种决策走出三种结果。"""
    print("--- 三种决策 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    cases: list[tuple[str, dict[str, Any], str]] = [
        ("批准", {"type": "approve"}, "executed:转账 500 元"),
        (
            "改后批准",
            {"type": "edit", "request": "转账 50 元"},
            "executed:转账 50 元",
        ),
        ("拒绝", {"type": "reject", "reason": "金额超限"}, "cancelled:转账 500 元"),
    ]

    for label, answer, expected in cases:
        trace: list[str] = []
        graph = _build_approval_graph(InMemorySaver(), trace)
        config = {"configurable": {"thread_id": f"decision-{label}"}}
        graph.invoke({"request": "转账 500 元", "decision": "", "log": []}, config)
        final = graph.invoke(Command(resume=answer), config)
        print(f"  {label:6} → {final['log']}")
        ensure(
            expected in final["log"],
            f"{label} 之后应当出现 {expected}，实际 {final['log']}",
        )


def check_same_thread_two_branches() -> None:
    """同一个 thread_id：批准一次、拒绝一次，走出两条分支。"""
    print("--- 同线程的两条分支 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    trace: list[str] = []
    graph = _build_approval_graph(InMemorySaver(), trace)
    config = {"configurable": {"thread_id": "branch-thread"}}

    # 第一轮：批准
    graph.invoke({"request": "转账 500 元", "decision": "", "log": []}, config)
    approved = graph.invoke(Command(resume={"type": "approve"}), config)
    print(f"  批准后：{approved['log']}")
    ensure(
        approved["log"] == ["approved", "executed:转账 500 元"],
        f"第一轮应当走执行分支，实际 {approved['log']}",
    )

    # 第二轮：同一个线程，新输入重新进图，这次拒绝
    graph.invoke({"request": "转账 9000 元", "decision": "", "log": []}, config)
    rejected = graph.invoke(Command(resume={"type": "reject", "reason": "金额超限"}), config)
    print(f"  拒绝后：{rejected['log']}")
    ensure(
        "cancelled:转账 9000 元" in rejected["log"],
        f"第二轮应当走驳回分支，实际 {rejected['log']}",
    )
    ensure(
        "executed:转账 500 元" in rejected["log"],
        "同一条线程上，前一轮的痕迹应当还在（归约器是追加语义）",
    )

    # 两条分支互不干扰：状态只有一份，但走向由每次的决策决定
    ensure(
        graph.get_state(config).next == (),
        "两轮跑完后图应当是结束状态",
    )
    print("  ✓ 同一条线程，两种决策，两条分支")


def check_hitl_middleware_shape() -> None:
    """LangChain 的 HumanInTheLoopMiddleware 与本章原语的关系。"""
    print("--- 与 LangChain 中间件的关系 ---")
    import inspect

    from langchain.agents.middleware import HumanInTheLoopMiddleware

    signature = inspect.signature(HumanInTheLoopMiddleware.__init__)
    mode = signature.parameters["interrupt_mode"].default
    print(f"  中间件：{HumanInTheLoopMiddleware.__module__}.HumanInTheLoopMiddleware")
    print(f"  interrupt_mode 默认值：{mode!r}")
    ensure(
        mode == "batched",
        f"批量模式应当是默认档，实际 {mode!r}",
    )
    ensure(
        "interrupt_on" in signature.parameters,
        "中间件必须要求配置 interrupt_on 策略",
    )
    print("  ✓ 中间件就是建在 interrupt() 之上的通用封装")


def run() -> None:
    """跑第 8 章实验。"""
    banner("第 8 章 · 人在环路：中断、审批与恢复")
    check_interrupt_pauses_graph()
    check_resume_continues_from_checkpoint()
    check_value_only_exists_after_resume()
    check_graph_interrupt_exception()
    check_three_decisions()
    check_same_thread_two_branches()
    check_hitl_middleware_shape()
    rule("中断把控制权交出去，恢复把控制权接回来")
    print("\n✓ 第 8 章通过")


def run_live() -> None:
    """真实模型路径：真模型草拟一个动作，再走完审批流。

    断言只校验形状（草拟非空、审批结论正确、副作用只跑一次），不校验内容。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from ..labkit import live_enabled, real_chat_model

    banner("第 8 章 · 真实模型路径")
    if not live_enabled():
        print("  未开启 --live，跳过真实模型路径。")
        return

    model = real_chat_model()
    trace: list[str] = []
    graph = _build_approval_graph(InMemorySaver(), trace)

    # 先让真模型拟一个待审批动作
    reply = model.invoke(
        [{"role": "user", "content": "用一句话写一件需要人工审批才能执行的事，不要解释。"}]
    )
    request = str(reply.content).strip()
    print(f"  模型草拟：{request[:60]}")
    ensure(bool(request), "真实模型应当给出一个非空的动作描述")

    config = {"configurable": {"thread_id": "live-hitl"}}
    paused = graph.invoke({"request": request, "decision": "", "log": []}, config)
    ensure("__interrupt__" in paused, "真实路径同样应当在中断处停下")
    ensure(
        paused["__interrupt__"][0].value.get("request") == request,
        "中断载荷里应当带上待审批的动作",
    )

    final = graph.invoke(Command(resume={"type": "approve"}), config)
    print(f"  跑完：{final['log']}")
    ensure(final["decision"] == "approve", f"审批结论应当是 approve，实际 {final['decision']}")
    ensure(
        sum(1 for item in final["log"] if item.startswith("executed:")) == 1,
        f"副作用应当且只应当发生一次，实际 {final['log']}",
    )
