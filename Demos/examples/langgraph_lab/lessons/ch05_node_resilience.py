"""第 5 章实验：节点韧性——重试、超时与错误处理。

这个模块解决什么问题：把"节点失败了该怎么办"从口头原则变成可执行的断言：

1. ``RetryPolicy`` 会把失败节点重跑 N 次；N 次用完仍然失败，异常继续往上抛；
2. **默认**只重试"瞬时错误"——``RuntimeError``、``ValueError`` 这类确定性错误默认不重试，
   而 ``ConnectionError`` 默认会被重试（实测 langgraph 1.2.20）；
3. ``error_handler`` 在重试耗尽后接管，把异常翻译成状态里的降级结果，并决定下一步走哪；
4. ``timeout``（``TimeoutPolicy``）只对**异步**节点生效，超时抛 ``NodeTimeoutError``；
5. ``set_node_defaults`` 一次配好全图默认、节点级配置优先；重试发生在 super-step **内部**，
   失败尝试不会多落检查点。

全部离线运行：不联网、不调模型。
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from ..labkit import banner, ensure, human, real_chat_model, rule, section


class State(TypedDict):
    """第 5 章的状态。

    ``status`` 记结论，``detail`` 装错误上下文——错误处理器写回的降级结果就落在这两个键上。
    """

    status: str
    detail: str


class DefaultsState(TypedDict):
    """图级默认那一节专用的状态。

    ``handled`` 用累加归约器：两个并行节点失败时，两个处理器都要写同一个键，
    覆盖语义会直接抛 ``InvalidUpdateError``。
    """

    status: str
    handled: Annotated[list[str], operator.add]


class TransientError(Exception):
    """代表"下游抖动"这类瞬时错误：重试有意义。"""


def _fast_retry(max_attempts: int) -> Any:
    """造一个"不睡长觉"的重试策略。

    官方默认是 ``initial_interval=0.5``、``backoff_factor=2.0``，实验里等不起。
    这里把间隔压到 10 毫秒——被测的是"重试几次"，不是"隔多久"。
    """
    from langgraph.types import RetryPolicy

    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=0.01,
        backoff_factor=1.0,
        max_interval=0.01,
        jitter=False,
    )


def check_retry_exhausts_then_raises() -> None:
    """必然失败的节点在 RetryPolicy 下被重试 N 次，最后仍然抛出。"""
    section("重试：N 次用完仍然失败")
    from langgraph.graph import END, START, StateGraph

    attempts: list[int] = []

    def always_fail(state: State) -> dict[str, Any]:
        attempts.append(1)
        raise TransientError("下游返回 503")

    builder = StateGraph(State)
    builder.add_node("always_fail", always_fail, retry_policy=_fast_retry(3))
    builder.add_edge(START, "always_fail")
    builder.add_edge("always_fail", END)
    graph = builder.compile()

    try:
        graph.invoke({"status": "start", "detail": ""})
    except TransientError as exc:
        print(f"  重试 {len(attempts)} 次后抛出：{exc}")
    else:
        raise AssertionError("重试耗尽后异常应该继续往上抛，而不是被吞掉")

    ensure(
        len(attempts) == 3,
        f"max_attempts=3 时节点体应当执行 3 次（含首次），实际 {len(attempts)}",
    )


def check_default_retry_on_is_not_obvious() -> None:
    """默认 ``retry_on`` 到底重试哪些异常——这是最容易想反的地方。"""
    section("默认 retry_on：谁会被重试，谁不会")
    from langgraph.graph import START, StateGraph
    from langgraph.types import default_retry_on

    ensure(default_retry_on(TransientError("x")) is True, "自定义异常默认应当会被重试")
    ensure(default_retry_on(RuntimeError("x")) is False, "RuntimeError 默认不该被重试")
    ensure(default_retry_on(ValueError("x")) is False, "ValueError 默认不该被重试")
    ensure(
        default_retry_on(ConnectionError("x")) is True,
        "ConnectionError 默认应当会被重试",
    )
    ensure(
        default_retry_on(TimeoutError("x")) is False,
        "TimeoutError 同属 OSError 家族，默认不该被重试",
    )

    runs: list[int] = []

    def boom(state: State) -> dict[str, Any]:
        runs.append(1)
        raise RuntimeError("确定性错误")

    builder = StateGraph(State)
    builder.add_node("boom", boom, retry_policy=_fast_retry(3))
    builder.add_edge(START, "boom")
    graph = builder.compile()

    try:
        graph.invoke({"status": "start", "detail": ""})
    except RuntimeError:
        pass

    ensure(
        len(runs) == 1,
        f"RuntimeError 默认不重试，节点体应当只执行 1 次，实际 {len(runs)} 次",
    )


def check_error_handler_degrades_and_routes() -> None:
    """错误处理器把异常翻译成状态里的降级结果，并决定下一步走哪。"""
    section("error_handler：降级而不是崩掉")
    from langgraph.errors import NodeError
    from langgraph.graph import START, StateGraph
    from langgraph.types import Command

    seen: list[tuple[str, str]] = []

    def charge(state: State) -> dict[str, Any]:
        raise TransientError("支付网关 503")

    def handler(state: State, error: NodeError) -> Command:
        # 处理器拿到的失败上下文是"结构化"的：哪个节点、什么异常
        seen.append((error.node, type(error.error).__name__))
        return Command(
            update={"status": "degraded", "detail": f"已补偿: {error.error}"},
            goto="finalize",
        )

    def finalize(state: State) -> dict[str, Any]:
        return {"detail": state["detail"] + " / 走完了补偿流程"}

    builder = StateGraph(State)
    builder.add_node(
        "charge",
        charge,
        retry_policy=_fast_retry(2),
        error_handler=handler,
    )
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "charge")
    graph = builder.compile()

    result = graph.invoke({"status": "start", "detail": ""})
    print(f"  最终状态：{result}")

    ensure(
        seen == [("charge", "TransientError")],
        f"处理器应当收到 charge 节点的 NodeError，实际 {seen}",
    )
    ensure(result["status"] == "degraded", f"状态应被降级结果覆盖，实际 {result['status']}")
    ensure(
        result["detail"].endswith("走完了补偿流程"),
        f"处理器 goto 的 finalize 应当继续执行，实际 {result['detail']}",
    )


def check_plain_handler_stops_static_edges() -> None:
    """反直觉：处理器返回普通 dict 时，失败节点的静态边**不会**继续走。"""
    section("坑：处理器返回 dict，图就停在那儿")
    from langgraph.errors import NodeError
    from langgraph.graph import START, StateGraph

    after_ran: list[str] = []

    def boom(state: State) -> dict[str, Any]:
        raise TransientError("必然失败")

    def after(state: State) -> dict[str, Any]:
        after_ran.append("after")
        return {"detail": state["detail"] + " / after"}

    def handler(state: State, error: NodeError) -> dict[str, Any]:
        return {"status": "degraded", "detail": "只更新了状态"}

    builder = StateGraph(State)
    builder.add_node("boom", boom, error_handler=handler)
    builder.add_node("after", after)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", "after")
    graph = builder.compile()

    result = graph.invoke({"status": "start", "detail": ""})
    print(f"  最终状态：{result}")

    ensure(
        after_ran == [],
        f"处理器返回普通 dict 时，失败节点的静态边不应继续执行，实际执行了 {after_ran}",
    )
    ensure(
        result["status"] == "degraded",
        f"处理器返回的 dict 应当写回状态，实际 {result['status']}",
    )


def check_timeout_interrupts_node() -> None:
    """超时会中断节点；而且超时只对异步节点生效。"""
    section("timeout：中断挂住的节点")
    from langgraph.errors import NodeTimeoutError
    from langgraph.graph import START, StateGraph
    from langgraph.types import TimeoutPolicy

    async def slow(state: State) -> dict[str, Any]:
        await asyncio.sleep(0.5)
        return {"status": "done"}

    builder = StateGraph(State)
    builder.add_node(
        "slow",
        slow,
        timeout=TimeoutPolicy(run_timeout=0.1),
        retry_policy=_fast_retry(2),
    )
    builder.add_edge(START, "slow")
    graph = builder.compile()

    try:
        asyncio.run(graph.ainvoke({"status": "start", "detail": ""}))
    except NodeTimeoutError as exc:
        print(f"  超时中断：node={exc.node} kind={exc.kind} elapsed={exc.elapsed:.2f}s")
        ensure(exc.kind == "run", f"应当是 run 超时，实际 {exc.kind}")
        ensure(exc.node == "slow", f"超时信息应指出节点名，实际 {exc.node}")
    else:
        raise AssertionError("超时应抛 NodeTimeoutError")

    def sync_node(state: State) -> dict[str, Any]:
        return {"status": "done"}

    bad_builder = StateGraph(State)
    bad_builder.add_node("sync_node", sync_node, timeout=1.0)
    bad_builder.add_edge(START, "sync_node")
    try:
        bad_builder.compile()
    except ValueError as exc:
        ensure("async" in str(exc), f"报错应当说明只支持异步节点，实际 {exc}")
        print(f"  同步节点带 timeout：编译期就被拒 → {exc}")
    else:
        raise AssertionError("同步节点加 timeout 应当在编译期被拒绝")

    # 反过来：异步节点带 timeout 编译通过
    async def async_node(state: State) -> dict[str, Any]:
        return {"status": "done"}

    ok_builder = StateGraph(State)
    ok_builder.add_node("async_node", async_node, timeout=TimeoutPolicy(run_timeout=5))
    ok_builder.add_edge(START, "async_node")
    ok_graph = ok_builder.compile()
    ensure(
        ok_graph.nodes["async_node"].timeout is not None,
        "异步节点的 timeout 应当在编译后的图上读得到",
    )
    print("  异步节点带 timeout：编译通过")


def check_compiled_graph_exposes_policies() -> None:
    """编译后的图上能读回每个节点的重试/超时配置——工程自检就靠它。"""
    section("编译后的图：配置可读")
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import TimeoutPolicy

    async def resilient(state: State) -> dict[str, Any]:
        return {"status": "done"}

    def plain(state: State) -> dict[str, Any]:
        return {"status": "done"}

    builder = StateGraph(State)
    builder.add_node(
        "resilient",
        resilient,
        timeout=TimeoutPolicy(run_timeout=30),
        retry_policy=_fast_retry(5),
    )
    builder.add_node("plain", plain)
    builder.add_edge(START, "resilient")
    builder.add_edge("resilient", "plain")
    builder.add_edge("plain", END)
    graph = builder.compile()

    ensure(isinstance(graph.nodes, dict), f"graph.nodes 应当是 dict，实际 {type(graph.nodes)}")
    resilient_node = graph.nodes["resilient"]
    plain_node = graph.nodes["plain"]

    # retry_policy 是**元组**：add_node 接受"一个或多个"策略，落盘时统一成元组
    ensure(
        isinstance(resilient_node.retry_policy, tuple),
        f"retry_policy 应当是策略元组，实际 {type(resilient_node.retry_policy)}",
    )
    ensure(
        resilient_node.retry_policy[0].max_attempts == 5,
        f"应当能读回 max_attempts=5，实际 {resilient_node.retry_policy[0].max_attempts}",
    )
    ensure(
        isinstance(resilient_node.timeout, TimeoutPolicy),
        f"timeout 应当读回 TimeoutPolicy，实际 {type(resilient_node.timeout)}",
    )
    ensure(
        resilient_node.timeout.run_timeout == 30,
        f"应当能读回 run_timeout=30，实际 {resilient_node.timeout.run_timeout}",
    )
    ensure(
        plain_node.retry_policy is None and plain_node.timeout is None,
        f"没配的节点应当是 None，实际 {plain_node.retry_policy} / {plain_node.timeout}",
    )

    # get_graph().nodes 也是 dict，不要按"节点对象列表"去遍历
    topology = graph.get_graph().nodes
    ensure(isinstance(topology, dict), f"get_graph().nodes 应当是 dict，实际 {type(topology)}")
    ensure(
        "resilient" in topology and "__start__" in topology,
        f"拓扑里应当能看到节点名，实际 {list(topology.keys())}",
    )
    print(f"  节点配置可读：resilient 重试 {resilient_node.retry_policy[0].max_attempts} 次、"
          f"超时 {resilient_node.timeout.run_timeout}s；拓扑是 dict，键有 {list(topology.keys())}")


def check_graph_defaults_apply_to_all_nodes() -> None:
    """``set_node_defaults`` 对全图生效；节点级配置优先。"""
    section("set_node_defaults：图级默认与优先级")
    from langgraph.errors import NodeError
    from langgraph.graph import START, StateGraph

    runs_a: list[int] = []
    runs_b: list[int] = []

    def fail_a(state: DefaultsState) -> dict[str, Any]:
        runs_a.append(1)
        raise TransientError("a 失败")

    def fail_b(state: DefaultsState) -> dict[str, Any]:
        runs_b.append(1)
        raise TransientError("b 失败")

    def default_handler(state: DefaultsState, error: NodeError) -> dict[str, Any]:
        return {"handled": [f"degraded@{error.node}"]}

    builder = StateGraph(DefaultsState)
    builder.set_node_defaults(
        retry_policy=_fast_retry(2),
        error_handler=default_handler,
    )
    builder.add_node("a", fail_a)
    # b 用节点级重试策略覆盖图级默认
    builder.add_node("b", fail_b, retry_policy=_fast_retry(3))
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    graph = builder.compile()

    result = graph.invoke({"status": "start", "handled": []})
    print(f"  a 尝试 {len(runs_a)} 次，b 尝试 {len(runs_b)} 次，处理器记录 {result['handled']}")

    ensure(len(runs_a) == 2, f"a 应当应用图级默认 max_attempts=2，实际 {len(runs_a)}")
    ensure(
        len(runs_b) == 3,
        f"b 的节点级配置应当覆盖图级默认，实际 {len(runs_b)}",
    )
    ensure(
        sorted(result["handled"]) == ["degraded@a", "degraded@b"],
        f"图级默认处理器应当接管两个节点，实际 {result['handled']}",
    )


def check_retry_stays_inside_one_superstep() -> None:
    """重试发生在 super-step 内部，不会多落检查点。"""
    section("重试与检查点：失败尝试不落盘")
    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime

    from ..labkit import sqlite_checkpointer

    def build(fail_times: int) -> Any:
        def work(state: State, runtime: Runtime) -> dict[str, Any]:
            attempt = runtime.execution_info.node_attempt
            if attempt <= fail_times:
                raise TransientError(f"第 {attempt} 次尝试失败")
            return {"status": "ok", "detail": f"第 {attempt} 次成功"}

        builder = StateGraph(State)
        builder.add_node("work", work, retry_policy=_fast_retry(3))
        builder.add_edge(START, "work")
        builder.add_edge("work", END)
        return builder

    with sqlite_checkpointer() as saver:
        counts: dict[int, int] = {}
        for fail_times in (0, 2):
            graph = build(fail_times).compile(checkpointer=saver)
            config = {"configurable": {"thread_id": f"ch05-retry-{fail_times}"}}
            result = graph.invoke({"status": "start", "detail": ""}, config)
            counts[fail_times] = len(list(graph.get_state_history(config)))
            print(f"  重试 {fail_times} 次 → 检查点 {counts[fail_times]} 个，结果 {result['status']}")

        ensure(
            counts[0] == counts[2],
            f"重试不应额外落检查点：一次成功 {counts[0]} 个，重试两次 {counts[2]} 个",
        )


def decide_strategy(exc: Exception, side_effect_free: bool) -> str:
    """按"是不是瞬时错误 + 有没有外部副作用"决定重试还是降级。

    这是笔记里那张决策表的代码版：确定性错误重试一万次还是同样的错；
    已经产生外部副作用的失败重试可能重复扣款——两者都该降级。
    """
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "降级"
    if not side_effect_free:
        return "降级"
    return "重试"


def check_retry_or_degrade_decision() -> None:
    """重试还是降级：决策表可执行。"""
    section("重试还是降级：决策表")

    cases = [
        ("瞬时错误 + 无副作用", TransientError("503"), True, "重试"),
        ("瞬时错误 + 有副作用", TransientError("503"), False, "降级"),
        ("参数错误 + 无副作用", ValueError("字段缺失"), True, "降级"),
        ("参数错误 + 有副作用", ValueError("字段缺失"), False, "降级"),
    ]
    for name, exc, side_effect_free, expected in cases:
        actual = decide_strategy(exc, side_effect_free)
        print(f"  {name:22} → {actual}")
        ensure(actual == expected, f"{name} 应当是 {expected}，实际 {actual}")


def run() -> None:
    """跑第 5 章实验。"""
    banner("第 5 章 · 节点韧性：重试、超时与错误处理")
    check_retry_exhausts_then_raises()
    check_default_retry_on_is_not_obvious()
    check_error_handler_degrades_and_routes()
    check_plain_handler_stops_static_edges()
    check_timeout_interrupts_node()
    check_compiled_graph_exposes_policies()
    check_graph_defaults_apply_to_all_nodes()
    check_retry_stays_inside_one_superstep()
    check_retry_or_degrade_decision()
    rule("重试只对瞬时错误有意义；重试发生在 super-step 内部，不会多落检查点")
    print("\n✓ 第 5 章通过")


def run_live() -> None:
    """真实模型路径：模型节点带重试与错误处理，只校验状态形状。"""
    banner("第 5 章 · 真实模型路径")
    from langgraph.errors import NodeError
    from langgraph.graph import END, START, StateGraph

    model = real_chat_model()

    def ask(state: State) -> dict[str, Any]:
        reply = model.invoke([human(state["detail"] or "用一句话介绍 LangGraph")])
        return {"status": "ok", "detail": str(reply.content)}

    def on_error(state: State, error: NodeError) -> dict[str, Any]:
        return {"status": "degraded", "detail": f"模型调用失败：{error.error}"}

    builder = StateGraph(State)
    builder.add_node(
        "ask",
        ask,
        retry_policy=_fast_retry(2),
        error_handler=on_error,
    )
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile()

    result = graph.invoke({"status": "start", "detail": "用一句话介绍 LangGraph"})
    print(f"  结果：{result}")

    # 真实模型的输出内容不可断言，只校验状态形状
    ensure(result["status"] in {"ok", "degraded"}, f"状态形状不对：{result['status']}")
    ensure(isinstance(result["detail"], str) and result["detail"], "detail 应当是非空字符串")
