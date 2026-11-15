"""第 9 章实验：流式与事件投影——五种 stream_mode 的形状差别。

这个模块解决什么问题：把"五种 stream_mode 到底是什么"变成能跑、能断言的事实：

1. 同一张图用 values / updates / messages / custom 四种模式流式跑，
   拿到的**不是同一份数据的四种叫法**——结构完全不同；
2. updates 只带这一步跑过的节点和它写回的键，values 带完整状态快照；
3. custom 通道能传业务自定义数据（进度、引用），而且**不写进状态**；
4. 子图内部的事件必须开 ``subgraphs=True`` 才透出，且带 ns 前缀；
5. v3 的事件投影把"原始协议事件"和"类型化投影"分开，而投影是单消费者的。

确定性路径不联网、不调模型；真实路径（``--live``）才调本机 OpenAI 兼容端点。
"""

from __future__ import annotations

import warnings
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from ..labkit import ai, banner, ensure, human, rule, scripted_model


class State(TypedDict):
    """第 9 章用的状态。

    三个键分属三个节点，这样 ``updates`` 模式里"哪个节点动了哪个键"
    一眼可辨；再加一个什么都不写的节点，专门暴露"跑过 ≠ 变了"。
    """

    topic: str
    draft: str
    score: int


def _build_graph(model: Any) -> Any:
    """搭一张"规划 → 起草 → 评审 → 发布"的四节点图。

    ``draft`` 节点会往 custom 通道推业务进度，用来看自定义通道怎么用。
    """
    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph

    def plan(state: State) -> dict[str, Any]:
        """只改 topic——用来演示 updates 里"只含变动键"。"""
        return {"topic": state["topic"].strip()}

    def draft(state: State) -> dict[str, Any]:
        """调模型写正文，同时往 custom 通道推两条业务进度。"""
        writer = get_stream_writer()
        writer({"stage": "draft", "progress": 50})
        reply = model.invoke([human(f"围绕「{state['topic']}」写一段介绍")])
        writer({"stage": "draft", "progress": 100})
        return {"draft": str(reply.content)}

    def review(state: State) -> dict[str, Any]:
        """按正文长度打个分——用来看 values 快照随步骤增长。"""
        return {"score": len(state["draft"])}

    def publish(state: State) -> dict[str, Any]:
        """什么都不写的节点：跑过了，但状态没变。"""
        return {}

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_node("publish", publish)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "publish")
    builder.add_edge("publish", END)
    return builder.compile()


def _build_nested_graph() -> Any:
    """搭一张"父节点 + 子图"的图，用来观察子图事件怎么透出。"""
    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph

    class InnerState(TypedDict):
        value: str

    def inner(state: InnerState) -> dict[str, Any]:
        """子图内部节点：往 custom 通道写一条带业务含义的数据。"""
        writer = get_stream_writer()
        writer({"from": "inner", "payload": "子图内部进度"})
        return {"value": state["value"] + "-inner"}

    inner_builder = StateGraph(InnerState)
    inner_builder.add_node("inner", inner)
    inner_builder.add_edge(START, "inner")
    inner_builder.add_edge("inner", END)
    inner_graph = inner_builder.compile()

    class OuterState(TypedDict):
        value: str

    def outer(state: OuterState) -> dict[str, Any]:
        """父图节点：也往 custom 通道写一条，用来和子图的数据做区分。"""
        writer = get_stream_writer()
        writer({"from": "outer", "payload": "父图进度"})
        return {"value": state["value"] + "-outer"}

    builder = StateGraph(OuterState)
    builder.add_node("outer", outer)
    builder.add_node("inner", inner_graph)
    builder.add_edge(START, "outer")
    builder.add_edge("outer", "inner")
    builder.add_edge("inner", END)
    return builder.compile()


# ------------------------------------------------------------ 五种模式的形状


def check_stream_mode_shapes() -> None:
    """同一张图、四种模式，拿到的结构完全不同。"""
    print("--- 四种 stream_mode 的形状 ---")
    graph = _build_graph(scripted_model([ai("流式实验用的正文内容。")]))
    initial: State = {"topic": "  流式输出  ", "draft": "", "score": 0}

    values_chunks = list(graph.stream(initial, stream_mode="values", version="v2"))
    updates_chunks = list(graph.stream(initial, stream_mode="updates", version="v2"))
    messages_chunks = list(graph.stream(initial, stream_mode="messages", version="v2"))
    custom_chunks = list(graph.stream(initial, stream_mode="custom", version="v2"))

    print(f"  values   → {len(values_chunks)} 块，末块 {values_chunks[-1]['data']}")
    print(f"  updates  → {len(updates_chunks)} 块，{updates_chunks}")
    print(f"  messages → {len(messages_chunks)} 块，节点 {messages_chunks[0]['data'][1].get('langgraph_node')}")
    print(f"  custom   → {len(custom_chunks)} 块，{[c['data'] for c in custom_chunks]}")

    # values：每步一个完整快照，键始终是状态 schema 的全部键
    ensure(
        len(values_chunks) == 4,
        f"values 应当给 4 个快照（初始 + 3 次真实变更），实际 {len(values_chunks)}",
    )
    ensure(
        all(set(chunk["data"]) == {"topic", "draft", "score"} for chunk in values_chunks),
        "values 的每一块都该是完整状态快照，键集固定",
    )
    ensure(
        values_chunks[-1]["data"]["topic"] == "流式输出",
        f"values 末块应当是最终状态，实际 {values_chunks[-1]['data']}",
    )

    # updates：每块只有一个节点，值是该节点写回的键
    ensure(
        all(len(chunk["data"]) == 1 for chunk in updates_chunks),
        "updates 的每一块只该含一个节点名",
    )
    ensure(
        {name for chunk in updates_chunks for name in chunk["data"]}
        == {"plan", "draft", "review", "publish"},
        f"updates 应当列出跑过的四个节点，实际 {updates_chunks}",
    )
    plan_update = next(chunk["data"]["plan"] for chunk in updates_chunks if "plan" in chunk["data"])
    ensure(
        set(plan_update) == {"topic"},
        f"plan 只改了 topic，updates 里不该出现别的键，实际 {plan_update}",
    )
    ensure(
        "score" not in plan_update,
        "没被写的键不该出现在 updates 里——这是 updates 的核心语义",
    )

    # 坑：跑过但没改状态的节点，updates 里是 None，而且不产生新的 values 快照
    publish_update = next(
        chunk["data"]["publish"] for chunk in updates_chunks if "publish" in chunk["data"]
    )
    ensure(
        publish_update is None,
        f"空更新节点在 updates 里应当是 None，实际 {publish_update}",
    )
    ensure(
        len(values_chunks) == 4,
        "空更新节点不该多出一个 values 快照（状态没变就没有新快照）",
    )

    # messages：二元组（消息块, 元数据）
    ensure(
        len(messages_chunks) == 1,
        f"脚本化模型只回一条，messages 应当只出一块，实际 {len(messages_chunks)}",
    )
    message_chunk, metadata = messages_chunks[0]["data"]
    ensure(
        isinstance(message_chunk, BaseMessage),
        f"messages 的 data[0] 应当是消息块，实际 {type(message_chunk)}",
    )
    ensure(
        metadata.get("langgraph_node") == "draft",
        f"messages 的元数据应当指出是哪个节点发的，实际 {metadata.get('langgraph_node')}",
    )
    ensure(
        isinstance(metadata, dict) and "langgraph_step" in metadata,
        "messages 的元数据应当是字典且带 step 信息",
    )

    # custom：业务自定义数据，与状态无关
    ensure(
        len(custom_chunks) == 2,
        f"draft 节点推了两条进度，custom 应当出两块，实际 {len(custom_chunks)}",
    )
    ensure(
        all(chunk["data"].get("stage") == "draft" for chunk in custom_chunks),
        f"custom 应当是业务自定义的字典，实际 {[c['data'] for c in custom_chunks]}",
    )
    ensure(
        "stage" not in values_chunks[-1]["data"],
        "custom 数据不该写进状态——它只是旁路输出",
    )


def check_debug_mode_shows_execution_detail() -> None:
    """debug 模式给的是执行细节：任务开始与任务结果。"""
    print("--- debug 模式 ---")
    graph = _build_graph(scripted_model([ai("正文")]))
    chunks = list(graph.stream({"topic": "x", "draft": "", "score": 0}, stream_mode="debug", version="v2"))
    kinds = [chunk["data"]["type"] for chunk in chunks]
    print(f"  debug 事件类型：{kinds}")
    print(f"  首条详情：{chunks[0]['data']}")

    ensure(
        "task" in kinds,
        f"debug 应当包含任务开始事件，实际 {kinds}",
    )
    ensure(
        "task_result" in kinds,
        f"debug 应当包含任务结果事件，实际 {kinds}",
    )
    ensure(
        all(chunk["type"] == "debug" for chunk in chunks),
        "debug 模式下每一块的 type 都该是 debug",
    )


# ------------------------------------------------------------ 子图事件


def check_subgraph_events_need_flag() -> None:
    """子图内部的事件：不开 subgraphs=True 就完全看不见。"""
    print("--- 子图事件要不要 subgraphs=True ---")
    graph = _build_nested_graph()

    flat = list(graph.stream({"value": "v"}, stream_mode="custom", version="v2"))
    nested = list(
        graph.stream({"value": "v"}, stream_mode="custom", subgraphs=True, version="v2")
    )

    print(f"  不开 subgraphs：{[c['data'] for c in flat]}")
    print(f"  开了 subgraphs：{[(c['ns'], c['data']) for c in nested]}")

    ensure(
        [c["data"]["from"] for c in flat] == ["outer"],
        f"不开 subgraphs 只能看到父图自己的 custom 数据，实际 {[c['data'] for c in flat]}",
    )
    ensure(
        len(nested) == 2,
        f"开了 subgraphs 才能看到子图那一条，实际 {len(nested)}",
    )
    root_chunks = [c for c in nested if not c["ns"]]
    subgraph_chunks = [c for c in nested if c["ns"]]
    ensure(
        len(root_chunks) == 1 and root_chunks[0]["data"]["from"] == "outer",
        "父图事件的 ns 应当是空元组",
    )
    ensure(
        len(subgraph_chunks) == 1 and subgraph_chunks[0]["data"]["from"] == "inner",
        f"子图事件应当带 ns 前缀，实际 {subgraph_chunks}",
    )
    ensure(
        subgraph_chunks[0]["ns"][0].startswith("inner:"),
        f"ns 应当是「节点名:运行 id」的形状，实际 {subgraph_chunks[0]['ns']}",
    )


# ------------------------------------------------------------ v3 事件投影


def check_v3_protocol_events_and_projections() -> None:
    """v3：原始协议事件是信封，类型化投影是另一层。"""
    print("--- v3 事件投影 ---")
    graph = _build_graph(scripted_model([ai("v3 正文")]))
    initial: State = {"topic": "x", "draft": "", "score": 0}

    # v3 还是实验特性，跑的时候官方会打 beta 警告；这里静音，免得盖住实验输出
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 先消费投影：v3 是"拉驱动"的，读取投影的过程同时把图跑完
        stream = graph.stream_events(initial, version="v3")
        snapshots = list(stream.values)
        final_output = stream.output

        # 原始协议事件要另开一次流
        raw_stream = graph.stream_events(initial, version="v3")
        raw = list(raw_stream)

    print(f"  原始事件方法：{sorted({event['method'] for event in raw})}")
    print(f"  投影快照数：{len(snapshots)}，最终输出：{final_output}")

    ensure(
        all("method" in event and "params" in event for event in raw),
        "v3 的原始事件应当是带 method/params 的协议信封",
    )
    ensure(
        all(
            "namespace" in event["params"] and "data" in event["params"]
            for event in raw
        ),
        "协议事件的 params 应当含 namespace 与 data",
    )
    ensure(
        {event["method"] for event in raw} <= {
            "values",
            "updates",
            "messages",
            "custom",
            "checkpoints",
            "tasks",
            "debug",
            "tools",
            "lifecycle",
            "input",
        },
        "v3 的 method 应当是文档里列出的通道名",
    )
    ensure(
        len(snapshots) == 4,
        f"values 投影应当给出 4 个快照，实际 {len(snapshots)}",
    )
    ensure(
        final_output == snapshots[-1],
        f"stream.output 应当等于最后一个快照，实际 {final_output}",
    )


def check_v3_projection_must_be_consumed_while_running() -> None:
    """坑：v3 的投影是"拉驱动"的——没订阅就没有数据可读。"""
    print("--- v3 投影的拉驱动语义 ---")
    graph = _build_graph(scripted_model([ai("v3 正文")]))
    initial: State = {"topic": "x", "draft": "", "score": 0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 先把原始事件流读完（等价于把图跑完），再回头读投影
        stream = graph.stream_events(initial, version="v3")
        raw = list(stream)
        late_snapshots = list(stream.values)

    print(f"  先跑完再读投影，拿到 {len(late_snapshots)} 个快照")
    ensure(
        len(raw) > 0,
        "原始事件流应当有内容",
    )
    ensure(
        late_snapshots == [],
        f"没订阅过的投影读不到历史数据，实际 {late_snapshots}",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 边跑边订阅，才有数据
        live_stream = graph.stream_events(initial, version="v3")
        eager_snapshots = list(live_stream.values)

    print(f"  边跑边订阅，拿到 {len(eager_snapshots)} 个快照")
    ensure(
        len(eager_snapshots) == 4,
        f"边跑边订阅应当拿到 4 个快照，实际 {len(eager_snapshots)}",
    )


def check_v3_projection_is_single_consumer() -> None:
    """坑：v3 的投影是单消费者的，第二个订阅者会报错。"""
    print("--- v3 投影的单消费者限制 ---")
    graph = _build_graph(scripted_model([ai("v3 正文")]))
    initial = {"topic": "x", "draft": "", "score": 0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = graph.stream_events(initial, version="v3")
        snapshots = list(first.values)
        error: RuntimeError | None = None
        try:
            list(first.interleave("values"))
        except RuntimeError as exc:
            error = exc

    ensure(
        error is not None,
        "同一个投影被订阅两次应当抛 RuntimeError",
    )
    ensure(
        "tee" in str(error),
        f"报错信息应当提示用 tee 做扇出，实际 {error}",
    )
    print(f"  重复订阅的报错：{error}")

    # 想同时看两份，就用 tee 复制出独立的游标
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        second = graph.stream_events(initial, version="v3")
        left, right = second.values.tee(2)
        left_values = list(left)
        right_values = list(right)

    print(f"  tee 两个分支拿到同样的 {len(left_values)} 个快照")
    ensure(
        left_values == right_values == snapshots,
        "tee 出来的两个分支应当看到同一份快照",
    )


# ------------------------------------------------------------ 入口


def run() -> None:
    """跑第 9 章实验。"""
    banner("第 9 章 · 流式与事件投影")
    check_stream_mode_shapes()
    check_debug_mode_shows_execution_detail()
    check_subgraph_events_need_flag()
    check_v3_protocol_events_and_projections()
    check_v3_projection_must_be_consumed_while_running()
    check_v3_projection_is_single_consumer()
    rule("五种模式是五种形状：状态、增量、消息、自定义数据、执行细节")
    print("\n✓ 第 9 章通过")


def run_live() -> None:
    """真实模型路径：真模型的 token 怎么进 messages 通道。

    断言只校验形状（有没有 chunk、元数据对不对），不校验模型说了什么。
    """
    from ..labkit import live_enabled, real_chat_model

    banner("第 9 章 · 真实模型路径")
    if not live_enabled():
        print("  未开启 --live，跳过真实模型路径。")
        return

    model = real_chat_model()
    graph = _build_graph(model)

    tokens: list[str] = []
    nodes: set[str] = set()
    for chunk in graph.stream(
        {"topic": "流式输出", "draft": "", "score": 0},
        stream_mode="messages",
        version="v2",
    ):
        message_chunk, metadata = chunk["data"]
        text = str(message_chunk.content)
        if text:
            tokens.append(text)
        nodes.add(str(metadata.get("langgraph_node")))

    print(f"  收到 {len(tokens)} 个 token 块，节点：{sorted(nodes)}")
    ensure(len(tokens) > 0, "真实模型路径应当至少收到一个 token 块")
    ensure("draft" in nodes, "token 应当来自 draft 节点")

    progress = [
        chunk["data"]
        for chunk in graph.stream(
            {"topic": "流式输出", "draft": "", "score": 0},
            stream_mode="custom",
            version="v2",
        )
    ]
    print(f"  custom 进度：{progress}")
    ensure(len(progress) == 2, "draft 节点应当推两条业务进度")
    ensure(
        all(item.get("stage") == "draft" for item in progress),
        f"custom 数据应当是业务字典，实际 {progress}",
    )
