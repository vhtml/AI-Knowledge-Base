"""第 7 章实验：持久化——检查点、三个 id 与时间旅行。

这个模块解决什么问题：把"状态存在哪、什么时候落盘、落盘到底能干什么"变成可执行的断言：

1. 同一个 ``thread_id`` 第二次 ``invoke`` 能接上上次的 state；
2. SQLite 文件关掉重开（用 with 块模拟进程重启）后历史仍在，而 ``InMemorySaver`` 重开即丢；
3. ``get_state_history`` 能列出多个检查点，且最新的一条排在最前；
4. ``update_state`` 能改历史状态，并让后续节点读到改后的值；
5. 换 ``thread_id`` 就是一条新线程；checkpointer 管线程内短期状态，store 管跨线程长期记忆。

全部离线运行，不联网、不调模型。
"""

from __future__ import annotations

import operator
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict

from ..labkit import banner, ensure, rule, sqlite_checkpointer


class State(TypedDict):
    """第 7 章用的状态。

    ``rounds`` 是普通键（覆盖语义），``notes`` 用累加归约器（追加语义）——
    有了归约器，两次 invoke 的结果才能一眼看出拼在同一条时间线上。
    """

    topic: str
    rounds: int
    notes: Annotated[list[str], operator.add]


def _build_graph(checkpointer: Any) -> Any:
    """搭一张三节点图：挑主题 → 计轮次 → 写笔记。

    ``count_round`` 会读上一轮的 ``rounds``，所以它就是"状态有没有接上"的探针。
    """
    from langgraph.graph import END, START, StateGraph

    def pick_topic(state: State) -> dict[str, Any]:
        return {"topic": "socks"}

    def count_round(state: State) -> dict[str, Any]:
        return {"rounds": state.get("rounds", 0) + 1}

    def write_note(state: State) -> dict[str, Any]:
        return {"notes": [f"round-{state['rounds']}:{state['topic']}"]}

    builder = StateGraph(State)
    builder.add_node("pick_topic", pick_topic)
    builder.add_node("count_round", count_round)
    builder.add_node("write_note", write_note)
    builder.add_edge(START, "pick_topic")
    builder.add_edge("pick_topic", "count_round")
    builder.add_edge("count_round", "write_note")
    builder.add_edge("write_note", END)
    return builder.compile(checkpointer=checkpointer)


def _build_live_graph(model: Any, checkpointer: Any) -> Any:
    """真实模型路径用的图：第一个节点让真模型挑一个主题。

    检查点器必须编译时就传进去——真实路径要靠它证明"重启后还在"。
    """
    from langgraph.graph import END, START, StateGraph

    def pick_topic(state: State) -> dict[str, Any]:
        reply = model.invoke(
            [{"role": "user", "content": "用一句话说一个适合写段子的日常物品，不要解释。"}]
        )
        return {"topic": str(reply.content).strip()}

    def count_round(state: State) -> dict[str, Any]:
        return {"rounds": state.get("rounds", 0) + 1}

    def write_note(state: State) -> dict[str, Any]:
        return {"notes": [f"round-{state['rounds']}"]}

    builder = StateGraph(State)
    builder.add_node("pick_topic", pick_topic)
    builder.add_node("count_round", count_round)
    builder.add_node("write_note", write_note)
    builder.add_edge(START, "pick_topic")
    builder.add_edge("pick_topic", "count_round")
    builder.add_edge("count_round", "write_note")
    builder.add_edge("write_note", END)
    return builder.compile(checkpointer=checkpointer)


def check_id_layers() -> None:
    """三个 id 的层级：thread_id 定线程，checkpoint_ns 定图层级，checkpoint_id 定时刻。"""
    print("--- 三个 id 的层级 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "id-thread"}}
    graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)

    head = graph.get_state(config)
    configurable = head.config["configurable"]
    print(f"  thread_id     = {configurable['thread_id']}   ← 哪条线程")
    print(f"  checkpoint_ns = {configurable.get('checkpoint_ns')!r}   ← 哪张图（根图为空串）")
    print(f"  checkpoint_id = {configurable['checkpoint_id']}   ← 哪个时刻")

    ensure(configurable["thread_id"] == "id-thread", "thread_id 应当原样保留")
    ensure(
        configurable.get("checkpoint_ns") == "",
        f"根图的 checkpoint_ns 应当是空字符串，实际 {configurable.get('checkpoint_ns')!r}",
    )
    ensure(bool(configurable["checkpoint_id"]), "应当拿到一个 checkpoint_id")

    # checkpoint_id 是被 thread_id 圈住的：换个线程，同一个 id 定位不到任何东西
    elsewhere = graph.get_state(
        {"configurable": {"thread_id": "other-thread", "checkpoint_id": configurable["checkpoint_id"]}}
    )
    ensure(
        elsewhere.values == {},
        f"换个 thread_id 后同一个 checkpoint_id 应当定位不到状态，实际 {elsewhere.values}",
    )
    print("  ✓ checkpoint_id 脱离 thread_id 就失去意义")


def check_thread_continues_state() -> None:
    """同一个 thread_id 第二次 invoke 能接上上次的 state。"""
    print("--- 同线程续跑 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "ch07-thread"}}

    first = graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)
    print(f"  第一次：rounds={first['rounds']} notes={first['notes']}")
    ensure(first["rounds"] == 1, f"第一轮 rounds 应当是 1，实际 {first['rounds']}")
    ensure(
        first["notes"] == ["round-1:socks"],
        f"第一轮 notes 应当只有一条，实际 {first['notes']}",
    )

    # 第二次不带 rounds：节点能读到 1，说明状态是从检查点接上的
    second = graph.invoke({"topic": "", "notes": []}, config)
    print(f"  第二次：rounds={second['rounds']} notes={second['notes']}")
    ensure(
        second["rounds"] == 2,
        f"第二轮 rounds 应当是 2（说明读到了上轮的 1），实际 {second['rounds']}",
    )
    ensure(
        second["notes"] == ["round-1:socks", "round-2:socks"],
        f"notes 应当把两轮拼起来，实际 {second['notes']}",
    )


def check_restart_persistence() -> None:
    """SQLite 文件关掉重开后历史仍在；InMemorySaver 重开即丢。"""
    print("--- 进程重启 ---")
    workdir = Path(tempfile.mkdtemp(prefix="langgraph_ch07_"))
    db_path = workdir / "checkpoints.sqlite"
    try:
        with sqlite_checkpointer(db_path) as saver:
            graph = _build_graph(saver)
            config = {"configurable": {"thread_id": "restart-thread"}}
            graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)
            before_count = len(list(graph.get_state_history(config)))
            print(f"  重启前：values={graph.get_state(config).values} 历史条数={before_count}")

        # 第二个 with 块 = 一个全新的进程，连接的还是同一个 SQLite 文件
        with sqlite_checkpointer(db_path) as saver:
            graph = _build_graph(saver)
            config = {"configurable": {"thread_id": "restart-thread"}}
            after = graph.get_state(config)
            after_count = len(list(graph.get_state_history(config)))
            print(f"  重启后：values={after.values} 历史条数={after_count}")

            ensure(
                after.values.get("rounds") == 1,
                f"重启后 rounds 应当还是 1，实际 {after.values.get('rounds')}",
            )
            ensure(
                after.values.get("notes") == ["round-1:socks"],
                f"重启后 notes 应当还在，实际 {after.values.get('notes')}",
            )
            ensure(
                after_count == before_count,
                f"重启后历史条数应当不变，期望 {before_count}，实际 {after_count}",
            )
            ensure(after_count >= 4, f"历史里至少要有 4 条检查点，实际 {after_count}")

        # 同样的事换成 InMemorySaver：换个进程就什么都没了
        from langgraph.checkpoint.memory import InMemorySaver

        memory_graph = _build_graph(InMemorySaver())
        memory_state = memory_graph.get_state({"configurable": {"thread_id": "restart-thread"}})
        memory_history = list(memory_graph.get_state_history({"configurable": {"thread_id": "restart-thread"}}))
        print(f"  InMemorySaver 重开后：values={memory_state.values} 历史条数={len(memory_history)}")
        ensure(
            memory_state.values == {},
            f"InMemorySaver 重开后应当查不到任何状态，实际 {memory_state.values}",
        )
        ensure(memory_history == [], "InMemorySaver 重开后不该留下任何历史")
        print("  ✓ 落盘与否，差别只在进程重启那一刻")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def check_state_history_listing() -> None:
    """get_state_history 能列出多个检查点，最新的排在最前。"""
    print("--- 检查点历史 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "history-thread"}}
    graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)

    history = list(graph.get_state_history(config))
    print(f"  共 {len(history)} 条检查点：")
    for snap in history:
        configurable = snap.config["configurable"]
        print(
            f"    step={str(snap.metadata.get('step')):>3} "
            f"source={str(snap.metadata.get('source')):<6} "
            f"next={str(snap.next):<18} "
            f"ns={configurable.get('checkpoint_ns')!r} "
            f"cid={configurable['checkpoint_id'][:8]}"
        )

    ensure(len(history) >= 4, f"历史里至少要有 4 条检查点，实际 {len(history)}")
    ensure(history[0].next == (), "最新的一条检查点应当没有后继节点（图已经跑完）")
    steps = [snap.metadata.get("step") for snap in history]
    ensure(steps == sorted(steps, reverse=True), f"历史应当按 step 倒序排列，实际 {steps}")
    ensure(
        history[-1].metadata.get("source") == "input",
        f"最早的一条应当来自 input，实际 {history[-1].metadata.get('source')}",
    )


def check_update_state_fork() -> None:
    """update_state 改历史状态，后续节点能读到改后的值。"""
    print("--- update_state 开新分支 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fork-thread"}}
    graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)

    history = list(graph.get_state_history(config))
    # 找到"还没跑 write_note"的那一刻，从这里分叉
    before_note = next(snap for snap in history if snap.next == ("write_note",))
    print(f"  分叉点：next={before_note.next} topic={before_note.values.get('topic')!r}")

    fork_config = graph.update_state(before_note.config, values={"topic": "chickens"})
    print(f"  update_state 产生的新检查点：{fork_config['configurable']['checkpoint_id'][:8]}")
    ensure(
        fork_config["configurable"]["checkpoint_id"]
        != before_note.config["configurable"]["checkpoint_id"],
        "update_state 应当产生一个新检查点，而不是原地改写",
    )

    fork_result = graph.invoke(None, fork_config)
    print(f"  分支跑完：notes={fork_result['notes']}")
    ensure(
        fork_result["notes"] == ["round-1:chickens"],
        f"后续节点应当读到改后的 topic=chickens，实际 {fork_result['notes']}",
    )

    # 原分支不受影响——update_state 不是回滚，是分叉
    original = graph.get_state(before_note.config)
    ensure(
        original.values.get("topic") == "socks",
        f"原检查点不该被改写，实际 {original.values.get('topic')}",
    )
    print("  ✓ 原历史完好无损，新分支另起一条")


def check_update_state_respects_reducers() -> None:
    """坑：update_state 走归约器，列表通道是追加而不是覆盖。"""
    print("--- update_state 与归约器 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "reducer-thread"}}
    graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)

    head = graph.get_state(config)
    ensure(
        head.values.get("notes") == ["round-1:socks"],
        f"跑完的线程 notes 应当有一条，实际 {head.values.get('notes')}",
    )

    fork_config = graph.update_state(head.config, values={"notes": ["manual"]})
    after = graph.get_state(fork_config)
    print(f"  追加后：notes={after.values.get('notes')}")
    ensure(
        after.values.get("notes") == ["round-1:socks", "manual"],
        f"归约器通道应当追加而不是覆盖，实际 {after.values.get('notes')}",
    )


def check_thread_isolation() -> None:
    """换 thread_id 就是一条新线程，状态互不可见。"""
    print("--- 线程隔离 ---")
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = _build_graph(saver)
    first = {"configurable": {"thread_id": "thread-a"}}
    second = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke({"topic": "", "rounds": 0, "notes": []}, first)
    state_a = graph.get_state(first)
    state_b = graph.get_state(second)
    print(f"  thread-a：{state_a.values}")
    print(f"  thread-b：{state_b.values}")

    ensure(state_a.values.get("rounds") == 1, "线程 A 应当保留自己的状态")
    ensure(state_b.values == {}, f"线程 B 应当是空的，实际 {state_b.values}")
    ensure(
        list(graph.get_state_history(second)) == [],
        "线程 B 不该有历史记录",
    )


def check_store_vs_checkpointer() -> None:
    """store 管跨线程长期记忆，checkpointer 管线程内短期状态。"""
    print("--- store 与 checkpointer 的分工 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    store.put(("user-1", "memories"), "pref-1", {"food": "pizza"})

    saver = InMemorySaver()
    graph = _build_graph(saver)
    first = {"configurable": {"thread_id": "store-thread-1"}}
    second = {"configurable": {"thread_id": "store-thread-2"}}
    graph.invoke({"topic": "", "rounds": 0, "notes": []}, first)

    memory = store.get(("user-1", "memories"), "pref-1")
    ensure(
        memory is not None and memory.value.get("food") == "pizza",
        f"store 里应当能读回记忆，实际 {memory}",
    )

    state_second = graph.get_state(second)
    ensure(state_second.values == {}, "线程 2 的状态应当是空的")
    print(f"  线程 2 状态：{state_second.values}")
    print(f"  线程 2 仍读得到同一份记忆：{memory.value}")
    print("  ✓ 状态按 thread_id 隔离，记忆按 namespace 共享")


def run() -> None:
    """跑第 7 章实验。"""
    banner("第 7 章 · 持久化：检查点与时间旅行")
    check_id_layers()
    check_thread_continues_state()
    check_restart_persistence()
    check_state_history_listing()
    check_update_state_fork()
    check_update_state_respects_reducers()
    check_thread_isolation()
    check_store_vs_checkpointer()
    rule("检查点是线程内的状态快照，线程之间互不可见")
    print("\n✓ 第 7 章通过")


def run_live() -> None:
    """真实模型路径：真模型挑主题，检查点落盘后重启再续跑。

    断言只校验形状（轮次、条数、重启后还在），不校验模型说了什么——
    模型答得好不好是评测的事，不是运行路径的事。
    """
    from ..labkit import live_enabled, real_chat_model

    banner("第 7 章 · 真实模型路径")
    if not live_enabled():
        print("  未开启 --live，跳过真实模型路径。")
        return

    model = real_chat_model()
    workdir = Path(tempfile.mkdtemp(prefix="langgraph_ch07_live_"))
    db_path = workdir / "live.sqlite"
    try:
        with sqlite_checkpointer(db_path) as saver:
            graph = _build_live_graph(model, saver)
            config = {"configurable": {"thread_id": "live-thread"}}
            first = graph.invoke({"topic": "", "rounds": 0, "notes": []}, config)
            print(f"  第一轮 topic={first['topic']!r} rounds={first['rounds']}")
            ensure(first["rounds"] == 1, "真实路径第一轮轮次应当是 1")
            ensure(bool(first["topic"]), "真实模型应当给出一个非空主题")

        with sqlite_checkpointer(db_path) as saver:
            graph = _build_live_graph(model, saver)
            config = {"configurable": {"thread_id": "live-thread"}}
            state = graph.get_state(config)
            ensure(
                state.values.get("rounds") == 1,
                f"重启后轮次应当还是 1，实际 {state.values.get('rounds')}",
            )
            second = graph.invoke({"topic": "", "notes": []}, config)
            print(f"  第二轮 topic={second['topic']!r} rounds={second['rounds']}")
            ensure(second["rounds"] == 2, "第二轮轮次应当是 2，说明状态接上了")
            ensure(len(second["notes"]) == 2, f"notes 应当累积到两条，实际 {second['notes']}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
