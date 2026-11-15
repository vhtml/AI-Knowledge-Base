"""第 11 章实验：函数式 API 与 Pregel 运行时——同一个流程，两种写法。

这个模块解决什么问题：第 11 章的两条主张要变成能跑的东西：

1. **图 API 和函数式 API 是同一套运行时的两种写法**——同一个"取数→加工→汇总"
   流程，两种写法跑出来的结果必须一致；
2. **函数式 API 的恢复是"重放"不是"续跑"**——副作用写在 entrypoint 里会在恢复时
   再执行一遍，写在 task 里则只执行一次。

另外把 task 的并发、`previous` 短期记忆、中断恢复都跑一遍。
全部离线，不联网、不调模型。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, TypedDict

from ..labkit import banner, ensure, rule

#: 假数据源：取数节点读它，避免联网
FAKE_ROWS: tuple[dict[str, int], ...] = (
    {"name": "甲", "score": 80},
    {"name": "乙", "score": 60},
    {"name": "丙", "score": 100},
)


# ------------------------------------------------------------ 同一流程的两种实现


class PipelineState(TypedDict):
    """图 API 用的状态。"""

    rows: list[dict[str, int]]
    doubled: list[dict[str, int]]
    summary: str


def build_graph_pipeline() -> Any:
    """图 API 版本：取数 → 加工 → 汇总。"""
    from langgraph.graph import END, START, StateGraph

    def fetch(state: PipelineState) -> dict[str, Any]:
        return {"rows": [dict(row) for row in FAKE_ROWS]}

    def transform(state: PipelineState) -> dict[str, Any]:
        return {"doubled": [{**row, "score": row["score"] * 2} for row in state["rows"]]}

    def summarize(state: PipelineState) -> dict[str, Any]:
        total = sum(row["score"] for row in state["doubled"])
        return {"summary": f"共 {len(state['doubled'])} 人，合计 {total} 分"}

    builder = StateGraph(PipelineState)
    builder.add_node("fetch", fetch)
    builder.add_node("transform", transform)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "transform")
    builder.add_edge("transform", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def build_functional_pipeline() -> Any:
    """函数式 API 版本：同样的三步，但控制流是普通 Python。"""
    from langgraph.func import entrypoint, task

    @task
    def fetch() -> list[dict[str, int]]:
        return [dict(row) for row in FAKE_ROWS]

    @task
    def transform(rows: list[dict[str, int]]) -> list[dict[str, int]]:
        return [{**row, "score": row["score"] * 2} for row in rows]

    @entrypoint()
    def pipeline(payload: dict) -> dict[str, Any]:
        # entrypoint 必须至少有一个位置参数；这里没有输入也要留一个占位
        rows = fetch().result()
        doubled = transform(rows).result()
        total = sum(row["score"] for row in doubled)
        return {
            "rows": rows,
            "doubled": doubled,
            "summary": f"共 {len(doubled)} 人，合计 {total} 分",
        }

    return pipeline


def check_two_apis_same_result() -> None:
    """图 API 与函数式 API 跑同一流程，结果必须一致。"""
    print("--- 同一流程，两种写法 ---")
    graph_result = build_graph_pipeline().invoke({"rows": [], "doubled": [], "summary": ""})
    func_result = build_functional_pipeline().invoke({})

    print(f"  图 API   ：{graph_result['summary']}")
    print(f"  函数式 API：{func_result['summary']}")

    ensure(
        graph_result["summary"] == func_result["summary"],
        f"两种写法的汇总结果应当一致：{graph_result['summary']} vs {func_result['summary']}",
    )
    ensure(
        graph_result["doubled"] == func_result["doubled"],
        "两种写法的加工结果应当一致",
    )
    print("  ✓ 同一套 Pregel 运行时，两种写法结果一致")


def check_tasks_overlap() -> None:
    """task 调用即开始执行，互不依赖的 task 会重叠。"""
    print("--- task 并发 ---")
    import time

    from langgraph.func import entrypoint, task

    order: list[str] = []

    @task
    def slow_a(value: int) -> int:
        order.append("a-start")
        time.sleep(0.05)
        order.append("a-end")
        return value

    @task
    def slow_b(value: int) -> int:
        order.append("b-start")
        time.sleep(0.05)
        order.append("b-end")
        return value

    @entrypoint()
    def pipeline(value: int) -> dict[str, int]:
        future_a = slow_a(value)  # 立刻开始，不阻塞
        future_b = slow_b(value)  # 也立刻开始
        return {"a": future_a.result(), "b": future_b.result()}

    result = pipeline.invoke(7)
    print(f"  执行顺序：{order}")
    print(f"  结果：{result}")

    ensure(result == {"a": 7, "b": 7}, f"结果应当是两个 7，实际 {result}")
    ensure(
        order.index("b-start") < order.index("a-end"),
        f"两个 task 应当重叠执行，实际顺序 {order}",
    )
    print("  ✓ 调用 task 即启动，互不依赖的 task 真正并发")


def check_previous_is_short_term_memory() -> None:
    """previous 注入参数就是短期记忆。"""
    print("--- previous 是短期记忆 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint

    seen: list[Any] = []

    @entrypoint(checkpointer=InMemorySaver())
    def counter(value: int, *, previous: Any = None) -> dict[str, Any]:
        seen.append(previous)
        return {"count": len(seen), "previous": previous}

    config = {"configurable": {"thread_id": "t-previous"}}
    first = counter.invoke(1, config)
    second = counter.invoke(2, config)
    print(f"  第一次 previous = {first['previous']}")
    print(f"  第二次 previous = {second['previous']}")

    ensure(first["previous"] is None, "第一次调用时 previous 应当是 None")
    ensure(
        second["previous"] == {"count": 1, "previous": None},
        f"第二次调用应当看到上一次的返回值，实际 {second['previous']}",
    )
    print("  ✓ 同一线程的第二次调用能看到上一次的返回值")


def check_side_effect_rules() -> None:
    """副作用写在 entrypoint 里会被重放，写在 task 里不会。"""
    print("--- 恢复时副作用会不会重跑 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint, task
    from langgraph.types import Command, interrupt

    tmp = Path(tempfile.mkdtemp())
    entry_file = tmp / "entry.txt"
    task_file = tmp / "task.txt"

    @task
    def write_in_task() -> str:
        task_file.write_text("x", encoding="utf-8")
        return "written"

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(payload: dict) -> dict[str, Any]:
        # 写在 entrypoint 里的副作用：恢复时会再跑一遍
        with entry_file.open("a", encoding="utf-8") as handle:
            handle.write("x")
        # 写在 task 里的副作用：恢复时从检查点还原
        write_in_task().result()
        approved = interrupt("批准吗？")
        return {"approved": approved}

    config = {"configurable": {"thread_id": "t-side-effect"}}
    workflow.invoke({}, config)
    workflow.invoke(Command(resume=True), config)

    entry_writes = entry_file.read_text("utf-8")
    task_writes = task_file.read_text("utf-8")
    print(f"  entrypoint 里的副作用写了 {len(entry_writes)} 次")
    print(f"  task 里的副作用写了 {len(task_writes)} 次")

    ensure(
        len(entry_writes) == 2,
        f"entrypoint 里的副作用在恢复时应当再执行一次，实际写了 {len(entry_writes)} 次",
    )
    ensure(
        len(task_writes) == 1,
        f"task 里的副作用在恢复时应当被还原而不是重跑，实际写了 {len(task_writes)} 次",
    )
    print("  ✓ 这就是'副作用必须放进 task'的原因")


def check_interrupt_and_resume() -> None:
    """函数式 API 的中断与恢复。"""
    print("--- 函数式 API 的中断与恢复 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint, task
    from langgraph.types import Command, interrupt

    @task
    def prepare(topic: str) -> str:
        return f"草稿：{topic}"

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(topic: str) -> dict[str, Any]:
        draft = prepare(topic).result()
        approved = interrupt({"draft": draft, "action": "批准/驳回"})
        return {"draft": draft, "approved": approved}

    config = {"configurable": {"thread_id": "t-interrupt"}}
    first = workflow.invoke("周报", config)
    print(f"  第一次结果键：{sorted(first.keys())}")
    ensure("__interrupt__" in first, f"第一次调用应当中断，实际 {first}")
    ensure(
        first["__interrupt__"][0].value["draft"] == "草稿：周报",
        f"中断载荷里应当带草稿，实际 {first['__interrupt__'][0].value}",
    )

    second = workflow.invoke(Command(resume=True), config)
    print(f"  恢复结果：{second}")
    ensure(second["approved"] is True, f"恢复值应当被采纳，实际 {second}")
    ensure(second["draft"] == "草稿：周报", f"task 结果应当被还原，实际 {second}")
    print("  ✓ 中断 → 恢复链路通，task 结果是还原的不是重算的")


def run() -> None:
    """跑第 11 章实验。"""
    banner("第 11 章 · 函数式 API 与 Pregel 运行时")
    check_two_apis_same_result()
    check_tasks_overlap()
    check_previous_is_short_term_memory()
    check_side_effect_rules()
    check_interrupt_and_resume()
    rule("图 API 是设计用的，函数式 API 是包装用的")
    print("\n✓ 第 11 章通过")
