"""第 4 章实验：控制流——分支、循环、Send 与 Command。

这个模块解决什么问题：图一旦超过三个节点，"下一步去哪"就从连线细节变成了设计决策。
这一章把四种控制流形态都跑一遍，并用断言回答"该用边，还是该用 Send / Command"：

1. 顺序、分支、并行、循环四种形态各自长什么样；
2. 条件分支按状态选路，路由函数只选路不改状态；
3. 循环靠终止条件收敛，``recursion_limit`` 兜底并抛 ``GraphRecursionError``；
4. ``Send`` 按运行时数量动态扇出，每个 worker 只拿到自己那份输入，结果靠归约器汇总；
5. ``Command`` 一步同时改状态并指定下一节点；``Literal`` 里列的目标节点必须真实存在。

全部离线运行，不联网、不调模型。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from ..labkit import banner, ensure, rule, section, sqlite_checkpointer


# ------------------------------------------------------------ 状态定义


class BranchState(TypedDict):
    """分支用：路由函数读它，节点写它。"""

    number: int
    is_even: bool
    message: str


class LoopState(TypedDict):
    """循环用：``rounds`` 累积，``budget`` 决定何时停。"""

    rounds: Annotated[list[str], operator.add]
    budget: int


class DocState(TypedDict):
    """map-reduce 的整体状态：worker 写共享的 ``chunk_summaries``，靠归约器合并。"""

    chunks: list[str]
    chunk_summaries: Annotated[list[dict[str, Any]], operator.add]
    report: str


class ChunkState(TypedDict):
    """worker 的私有输入：只有 ``Send`` 传进来的那两个键。"""

    chunk: str
    index: int


class ReviewState(TypedDict):
    """Command 用：一个节点同时改状态并指定下一节点。"""

    score: int
    verdict: str
    note: str


# ------------------------------------------------------------ 形态一：顺序


def check_sequence() -> None:
    """顺序：固定步骤依次执行。"""
    section("形态一：顺序")

    class SeqState(TypedDict):
        text: str

    def step_1(state: SeqState) -> dict[str, Any]:
        return {"text": state["text"] + "a"}

    def step_2(state: SeqState) -> dict[str, Any]:
        return {"text": state["text"] + "b"}

    builder = StateGraph(SeqState)
    builder.add_sequence([step_1, step_2])
    builder.add_edge(START, "step_1")
    builder.add_edge("step_2", END)
    result = builder.compile().invoke({"text": ""})

    print(f"  顺序结果：{result}")
    ensure(result["text"] == "ab", f"顺序执行应当依次拼出 ab，实际 {result['text']}")


# ------------------------------------------------------------ 形态二：分支


def check_branch() -> None:
    """分支：路由函数按状态选路，选中的那条分支才会执行。"""
    section("形态二：分支")

    def check_even(state: BranchState) -> dict[str, Any]:
        return {"is_even": state["number"] % 2 == 0}

    def even_branch(state: BranchState) -> dict[str, Any]:
        return {"message": "偶数"}

    def odd_branch(state: BranchState) -> dict[str, Any]:
        return {"message": "奇数"}

    def route(state: BranchState) -> Literal["even_branch", "odd_branch"]:
        """路由函数只决定去哪，不改状态。"""
        return "even_branch" if state["is_even"] else "odd_branch"

    builder = StateGraph(BranchState)
    builder.add_node("check_even", check_even)
    builder.add_node("even_branch", even_branch)
    builder.add_node("odd_branch", odd_branch)
    builder.add_edge(START, "check_even")
    builder.add_conditional_edges("check_even", route)
    builder.add_edge("even_branch", END)
    builder.add_edge("odd_branch", END)
    graph = builder.compile()

    even_result = graph.invoke({"number": 8, "is_even": False, "message": ""})
    odd_result = graph.invoke({"number": 7, "is_even": False, "message": ""})

    print(f"  8 的结果：{even_result['message']}")
    print(f"  7 的结果：{odd_result['message']}")
    ensure(even_result["message"] == "偶数", f"8 应当走偶数分支，实际 {even_result['message']}")
    ensure(odd_result["message"] == "奇数", f"7 应当走奇数分支，实际 {odd_result['message']}")
    ensure(
        even_result["is_even"] is True,
        "分支前的节点写入的状态应当保留下来",
    )


# ------------------------------------------------------------ 形态三：并行


def check_parallel() -> None:
    """并行：同一 super-step 里扇出，靠归约器扇入。"""
    section("形态三：并行")

    class ParState(TypedDict):
        aggregate: Annotated[list[str], operator.add]

    def a(state: ParState) -> dict[str, Any]:
        return {"aggregate": ["A"]}

    def b(state: ParState) -> dict[str, Any]:
        return {"aggregate": ["B"]}

    def c(state: ParState) -> dict[str, Any]:
        return {"aggregate": ["C"]}

    def d(state: ParState) -> dict[str, Any]:
        return {"aggregate": ["D"]}

    builder = StateGraph(ParState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_node("c", c)
    builder.add_node("d", d)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)
    graph = builder.compile()

    result = graph.invoke({"aggregate": []})
    print(f"  扇入后的结果：{result['aggregate']}")
    ensure(
        sorted(result["aggregate"]) == ["A", "B", "C", "D"],
        f"四条分支应当都被归约器合并进来，实际 {result['aggregate']}",
    )


# ------------------------------------------------------------ 形态四：循环


def check_loop() -> None:
    """循环：边指回前面的节点，靠终止条件收敛。"""
    section("形态四：循环")

    def work(state: LoopState) -> dict[str, Any]:
        return {"rounds": ["round"]}

    def route(state: LoopState) -> Literal["work", END]:
        """达到预算就停，否则继续转。"""
        return "work" if len(state["rounds"]) < state["budget"] else END

    builder = StateGraph(LoopState)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    # 同一个节点别再挂静态边——路由要么全交给条件边，要么全交给 Command
    builder.add_conditional_edges("work", route)
    graph = builder.compile()

    result = graph.invoke({"rounds": [], "budget": 3})
    print(f"  跑了 {len(result['rounds'])} 轮")
    ensure(
        len(result["rounds"]) == 3,
        f"预算 3 轮应当在第 3 轮停住，实际 {len(result['rounds'])} 轮",
    )


def check_recursion_limit() -> None:
    """``recursion_limit`` 是循环的最后一道防线，超了抛 ``GraphRecursionError``。"""
    section("循环兜底：recursion_limit")

    def work(state: LoopState) -> dict[str, Any]:
        return {"rounds": ["round"]}

    def route(state: LoopState) -> Literal["work"]:
        """这个路由永远不结束——只有 recursion_limit 能拦住它。"""
        return "work"

    builder = StateGraph(LoopState)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_conditional_edges("work", route)
    graph = builder.compile()

    failed = False
    try:
        graph.invoke({"rounds": [], "budget": 99}, {"recursion_limit": 4})
    except GraphRecursionError as exc:
        failed = True
        print(f"  超限被拦下：{str(exc).splitlines()[0]}")
    ensure(failed, "死循环应当在 recursion_limit 处抛 GraphRecursionError")


# ------------------------------------------------------------ Send：map-reduce


def check_send_map_reduce() -> None:
    """``Send`` 按运行时数量扇出，worker 各拿各的输入，结果靠归约器汇总。"""
    section("Send：动态扇出的 map-reduce")

    seen_inputs: list[dict[str, Any]] = []

    def plan(state: DocState) -> DocState:
        """按句子切块——块数只有运行时才知道。"""
        chunks = [part for part in state["chunks"] if part]
        return {"chunks": chunks}

    def worker(state: ChunkState) -> DocState:
        """worker 只拿到 ``Send`` 传进来的那两个键，不是整张状态。"""
        seen_inputs.append(dict(state))
        return {
            "chunk_summaries": [
                {"index": state["index"], "summary": f"[{state['index']}] {state['chunk'][:6]}"}
            ]
        }

    def route_workers(state: DocState) -> list[Send]:
        """运行时决定扇出多少个 worker。"""
        return [
            Send("worker", {"chunk": chunk, "index": index})
            for index, chunk in enumerate(state["chunks"])
        ]

    def reduce_node(state: DocState) -> DocState:
        """并行写入的顺序不保证，汇总前必须按 index 排。"""
        ordered = sorted(state["chunk_summaries"], key=lambda item: item["index"])
        return {"report": " | ".join(item["summary"] for item in ordered)}

    builder = StateGraph(DocState)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker)
    builder.add_node("reduce", reduce_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_workers, ["worker"])
    builder.add_edge("worker", "reduce")
    builder.add_edge("reduce", END)
    graph = builder.compile()

    chunks = ["第一句话讲状态", "第二句话讲归约器", "第三句话讲扇出"]
    result = graph.invoke({"chunks": chunks, "chunk_summaries": [], "report": ""})

    print(f"  扇出 {len(seen_inputs)} 个 worker")
    print(f"  汇总报告：{result['report']}")

    ensure(
        len(seen_inputs) == len(chunks),
        f"worker 数量应当等于块数 {len(chunks)}，实际 {len(seen_inputs)}",
    )
    ensure(
        all(set(item) == {"chunk", "index"} for item in seen_inputs),
        f"worker 只该拿到 Send 传的两个键，实际 {seen_inputs}",
    )
    ensure(
        sorted(item["index"] for item in seen_inputs) == [0, 1, 2],
        f"每个 worker 应当拿到不同的下标，实际 {seen_inputs}",
    )
    ensure(
        len(result["chunk_summaries"]) == len(chunks),
        f"归约器应当把所有 worker 的产出合并起来，实际 {result['chunk_summaries']}",
    )
    ensure(
        result["report"] == "[0] 第一句话讲状 | [1] 第二句话讲归 | [2] 第三句话讲扇",
        f"汇总结果应当按 index 排序，实际 {result['report']}",
    )


# ------------------------------------------------------------ Command


def decide(state: ReviewState) -> Command[Literal["accept", "reject"]]:
    """一步同时改状态并指定下一节点。"""
    goto = "accept" if state["score"] >= 60 else "reject"
    return Command(update={"verdict": goto}, goto=goto)


def check_command_update_and_goto() -> None:
    """``Command``：update + goto 一步到位，图里不需要条件边。"""
    section("Command：改状态 + 指定下一节点")

    def accept(state: ReviewState) -> dict[str, Any]:
        return {"note": "通过"}

    def reject(state: ReviewState) -> dict[str, Any]:
        return {"note": "驳回"}

    builder = StateGraph(ReviewState)
    builder.add_edge(START, "decide")
    builder.add_node("decide", decide)
    builder.add_node("accept", accept)
    builder.add_node("reject", reject)
    builder.add_edge("accept", END)
    builder.add_edge("reject", END)
    graph = builder.compile()

    good = graph.invoke({"score": 90, "verdict": "", "note": ""})
    bad = graph.invoke({"score": 10, "verdict": "", "note": ""})
    print(f"  90 分：{good}")
    print(f"  10 分：{bad}")

    ensure(good["verdict"] == "accept", f"90 分应当判通过，实际 {good['verdict']}")
    ensure(good["note"] == "通过", f"accept 分支应当写 note，实际 {good['note']}")
    ensure(bad["verdict"] == "reject", f"10 分应当判驳回，实际 {bad['verdict']}")
    ensure(bad["note"] == "驳回", f"reject 分支应当写 note，实际 {bad['note']}")

    # ``Command[Literal[...]]`` 会把动态边画进拓扑图里
    edges = [(edge.source, edge.target) for edge in graph.get_graph().edges]
    print(f"  拓扑里的边：{edges}")
    ensure(
        ("decide", "accept") in edges and ("decide", "reject") in edges,
        f"Command 的 goto 应当出现在拓扑图里，实际 {edges}",
    )


def check_command_literal_target_must_exist() -> None:
    """坑：``Literal`` 里列的目标节点必须真实存在，否则编译期就炸。"""
    section("坑：Command[Literal[...]] 的目标节点必须存在")

    def bad_decide(state: ReviewState) -> Command[Literal["nonexistent"]]:
        return Command(update={"verdict": "x"}, goto="nonexistent")

    builder = StateGraph(ReviewState)
    builder.add_edge(START, "bad_decide")
    builder.add_node("bad_decide", bad_decide)

    failed = False
    try:
        builder.compile()
    except ValueError as exc:
        failed = True
        print(f"  编译期就被拦下：{exc}")
    ensure(failed, "Literal 里列了不存在的节点，编译期就应当报错")


def check_route_stays_in_edges() -> None:
    """同一段路由逻辑：放在边里 vs 放进 Command，结果一样，归属不同。"""
    section("路由该放哪：边 vs Command")

    def accept(state: ReviewState) -> dict[str, Any]:
        return {"note": "通过"}

    def reject(state: ReviewState) -> dict[str, Any]:
        return {"note": "驳回"}

    def route(state: ReviewState) -> Literal["accept", "reject"]:
        # 路由函数的注解必须能在模块作用域解析出来，否则 compile 时会 NameError
        return "accept" if state["score"] >= 60 else "reject"

    edge_builder = StateGraph(ReviewState)
    edge_builder.add_node("accept", accept)
    edge_builder.add_node("reject", reject)
    edge_builder.add_conditional_edges(START, route)
    edge_builder.add_edge("accept", END)
    edge_builder.add_edge("reject", END)
    edge_result = edge_builder.compile().invoke({"score": 90, "verdict": "", "note": ""})

    cmd_builder = StateGraph(ReviewState)
    cmd_builder.add_edge(START, "decide")
    cmd_builder.add_node("decide", decide)
    cmd_builder.add_node("accept", accept)
    cmd_builder.add_node("reject", reject)
    cmd_builder.add_edge("accept", END)
    cmd_builder.add_edge("reject", END)
    cmd_result = cmd_builder.compile().invoke({"score": 90, "verdict": "", "note": ""})

    print(f"  放在边里：{edge_result}")
    print(f"  放进 Command：{cmd_result}")

    ensure(
        edge_result["note"] == cmd_result["note"],
        "两种写法应当选到同一个分支",
    )
    ensure(
        cmd_result["verdict"] == "accept",
        f"Command 应当顺手写下 verdict，实际 {cmd_result['verdict']}",
    )
    ensure(
        edge_result["verdict"] == "",
        "放在边里时路由函数不改状态，verdict 应当保持空",
    )


def check_command_as_input_resumes_checkpoint() -> None:
    """坑：把 ``Command`` 当 invoke 入参，是"从最新检查点恢复"，不是"从头开始"。"""
    section("坑：Command 当入参 = 从最新检查点恢复")

    class ResumeState(TypedDict):
        text: str

    calls: list[str] = []

    def start(state: ResumeState) -> dict[str, Any]:
        calls.append("start")
        return {"text": "set"}

    def after(state: ResumeState) -> dict[str, Any]:
        calls.append("after")
        return {}

    builder = StateGraph(ResumeState)
    builder.add_node("start", start)
    builder.add_node("after", after)
    builder.add_edge(START, "start")
    builder.add_edge("start", "after")
    builder.add_edge("after", END)

    with sqlite_checkpointer() as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "ch04-resume"}}

        graph.invoke({"text": ""}, config)
        ensure(calls == ["start", "after"], f"首轮应当跑完两个节点，实际 {calls}")

        calls.clear()
        result = graph.invoke(Command(update={"text": "via-command"}), config)
        print(f"  Command 入参的结果：{result}")
        print(f"  Command 入参时又跑的节点：{calls}")

        ensure(
            calls == [],
            f"Command 入参会从最新检查点恢复，节点不该重跑，实际 {calls}",
        )
        ensure(
            result["text"] == "via-command",
            f"Command 的 update 应当写进状态，实际 {result}",
        )


# ------------------------------------------------------------ 入口


def run() -> None:
    """跑第 4 章实验。"""
    banner("第 4 章 · 控制流：分支、循环、Send 与 Command")
    check_sequence()
    check_branch()
    check_parallel()
    check_loop()
    check_recursion_limit()
    check_send_map_reduce()
    check_command_update_and_goto()
    check_command_literal_target_must_exist()
    check_route_stays_in_edges()
    check_command_as_input_resumes_checkpoint()
    rule("顺序 / 分支 / 并行 / 循环四种形态；Send 扇出；Command 一步改状态并选路")
    print("\n✓ 第 4 章通过")
