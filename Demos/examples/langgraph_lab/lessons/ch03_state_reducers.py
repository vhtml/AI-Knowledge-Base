"""第 3 章实验：状态定义与归约器。

这个模块解决什么问题：第 2 章说"节点只写它改的键"，但没回答"两个节点同时写同一个键会怎样"。
状态的真正契约是"并发时怎么合并"，这一章把三个决定做成可执行的断言：

1. 配了归约器的键可以并行写；没配归约器的键并行写会抛 ``InvalidUpdateError``；
2. 归约器是"合并"不是"覆盖"——返回空列表清不掉已累积的值，只能用 ``Overwrite``；
3. ``Overwrite`` 能绕过归约器，但同一个 super-step 里同一个键只能有一个 ``Overwrite``；
4. input / output / private 三套 schema 各管一段：``invoke`` 只返回 output 字段，
   私有通道却仍然会出现在 ``values`` 流里；
5. ``add_messages`` 除了追加，还能按 id 覆盖旧消息、把 ``{"role": ...}`` 简写转成消息对象。

全部离线运行，不联网、不调模型。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain.messages import AIMessage, AnyMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite
from pydantic import BaseModel, ValidationError

from ..labkit import banner, ensure, human, live_enabled, real_chat_model, rule, section


# ------------------------------------------------------------ 状态 schema 形态


class TypedState(TypedDict):
    """TypedDict schema：节点拿到的是字典，用下标取值。"""

    text: str


@dataclass
class DataState:
    """dataclass schema：节点拿到的是实例，用属性取值；默认值写在这里。"""

    text: str = "初始"


class PydanticState(BaseModel):
    """Pydantic schema：只在**输入**侧做运行时校验。"""

    text: str


class ReviewState(TypedDict):
    """贯穿本章的状态：一条合并键、一条覆盖键、一条需要清空的错误缓冲。"""

    findings: Annotated[list[str], operator.add]
    """并行审查分支各写各的，靠归约器合并。"""

    summary: str
    """普通键，最后一次写入生效。"""

    errors: Annotated[list[str], operator.add]
    """错误缓冲：累积语义，但每轮重试前需要清空。"""


# ------------------------------------------------------------ schema 三选一


def check_schema_forms() -> None:
    """TypedDict / dataclass / Pydantic：取值方式、校验时机、输出类型各不同。"""
    section("三种 schema 形态")

    def typed_node(state: TypedState) -> dict[str, Any]:
        return {"text": state["text"] + "-typed"}

    def data_node(state: DataState) -> dict[str, Any]:
        return {"text": state.text + "-data"}

    def pydantic_node(state: PydanticState) -> dict[str, Any]:
        return {"text": state.text + "-pydantic"}

    typed_graph = (
        StateGraph(TypedState)
        .add_node("n", typed_node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )
    data_graph = (
        StateGraph(DataState)
        .add_node("n", data_node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )
    pydantic_graph = (
        StateGraph(PydanticState)
        .add_node("n", pydantic_node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    typed_out = typed_graph.invoke({"text": "a"})
    data_out = data_graph.invoke({"text": "a"})
    pydantic_out = pydantic_graph.invoke({"text": "a"})

    print(f"  TypedDict 输出：{typed_out}")
    print(f"  dataclass 输出：{data_out}")
    print(f"  Pydantic  输出：{pydantic_out}")

    ensure(typed_out == {"text": "a-typed"}, f"TypedDict 应写入后覆盖，实际 {typed_out}")
    ensure(data_out == {"text": "a-data"}, f"dataclass 应写入后覆盖，实际 {data_out}")
    ensure(
        pydantic_out == {"text": "a-pydantic"},
        f"Pydantic 应写入后覆盖，实际 {pydantic_out}",
    )
    ensure(
        not isinstance(pydantic_out, PydanticState),
        "Pydantic schema 的图输出**不是**模型实例，而是普通字典",
    )

    # dataclass 的价值在于默认值：不传输入也能起跑
    default_out = data_graph.invoke({})
    ensure(
        default_out == {"text": "初始-data"},
        f"dataclass 的默认值应当生效，实际 {default_out}",
    )

    # Pydantic 的价值在于输入校验：非法输入在进图时就被拦下
    rejected = False
    try:
        pydantic_graph.invoke({"text": 123})  # type: ignore[dict-item]
    except ValidationError:
        rejected = True
    ensure(rejected, "Pydantic schema 应当在输入非法时抛 ValidationError")
    print("  ✓ 校验只发生在输入侧；输出侧一律是普通字典")


# ------------------------------------------------------------ 归约器语义


def check_reducer_merge_semantics() -> None:
    """并行写：有归约器的键能合并，没归约器的键直接抛错。"""
    section("归约器：合并 vs 覆盖")

    class MergeState(TypedDict):
        merged: Annotated[list[str], operator.add]
        plain: str

    def left(state: MergeState) -> dict[str, Any]:
        return {"merged": ["left"], "plain": "left"}

    def right(state: MergeState) -> dict[str, Any]:
        return {"merged": ["right"], "plain": "right"}

    builder = StateGraph(MergeState)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)
    graph = builder.compile()

    class OnlyMergeState(TypedDict):
        merged: Annotated[list[str], operator.add]

    def only_left(state: OnlyMergeState) -> dict[str, Any]:
        return {"merged": ["left"]}

    def only_right(state: OnlyMergeState) -> dict[str, Any]:
        return {"merged": ["right"]}

    ok_builder = StateGraph(OnlyMergeState)
    ok_builder.add_node("left", only_left)
    ok_builder.add_node("right", only_right)
    ok_builder.add_edge(START, "left")
    ok_builder.add_edge(START, "right")
    ok_builder.add_edge("left", END)
    ok_builder.add_edge("right", END)
    ok_result = ok_builder.compile().invoke({"merged": []})

    print(f"  有归约器的键并行写：{ok_result['merged']}")
    ensure(
        sorted(ok_result["merged"]) == ["left", "right"],
        f"归约器应当把两个分支合并起来，实际 {ok_result['merged']}",
    )

    failed = False
    try:
        graph.invoke({"merged": [], "plain": ""})
    except InvalidUpdateError as exc:
        failed = True
        print(f"  无归约器的键并行写被拦下：{str(exc).splitlines()[0]}")
    ensure(
        failed,
        "没有归约器的键被两个分支同时写，应当抛 InvalidUpdateError",
    )


# ------------------------------------------------------------ Overwrite


def _two_step_buffer_graph(second_node: Any) -> Any:
    """搭一张"产出错误 → 处理错误"的两节点图，返回编译好的图。"""

    class BufferState(TypedDict):
        errors: Annotated[list[str], operator.add]

    def produce(state: BufferState) -> dict[str, Any]:
        return {"errors": ["bad sql"]}

    builder = StateGraph(BufferState)
    builder.add_node("produce", produce)
    builder.add_node("handle", second_node)
    builder.add_edge(START, "produce")
    builder.add_edge("produce", "handle")
    builder.add_edge("handle", END)
    return builder.compile()


def check_overwrite_bypasses_reducer() -> None:
    """Overwrite 绕过归约器；空列表清不掉合并字段；同一步只能有一个 Overwrite。"""
    section("Overwrite：绕过归约器")

    def clear_with_empty(state: dict[str, Any]) -> dict[str, Any]:
        """返回空列表并不会清空——空更新照样被合并进去。"""
        return {"errors": []}

    def clear_with_overwrite(state: dict[str, Any]) -> dict[str, Any]:
        """显式绕过归约器，直接把通道置空。"""
        return {"errors": Overwrite([])}

    def replace_with_overwrite(state: dict[str, Any]) -> dict[str, Any]:
        """Overwrite 也能整体替换，不只是清空。"""
        return {"errors": Overwrite(["replacement"])}

    empty_result = _two_step_buffer_graph(clear_with_empty).invoke({"errors": []})
    overwrite_result = _two_step_buffer_graph(clear_with_overwrite).invoke({"errors": []})
    replace_result = _two_step_buffer_graph(replace_with_overwrite).invoke({"errors": []})

    print(f"  返回空列表后：{empty_result['errors']}")
    print(f"  Overwrite([]) 后：{overwrite_result['errors']}")
    print(f"  Overwrite([...]) 后：{replace_result['errors']}")

    ensure(
        empty_result["errors"] == ["bad sql"],
        f"合并型归约器下返回空列表不该清空字段，实际 {empty_result['errors']}",
    )
    ensure(
        overwrite_result["errors"] == [],
        f"Overwrite([]) 应当清空字段，实际 {overwrite_result['errors']}",
    )
    ensure(
        replace_result["errors"] == ["replacement"],
        f"Overwrite 应当整体替换，实际 {replace_result['errors']}",
    )

    # 同一个 super-step 里，同一个键只能有一个 Overwrite
    class BufferState(TypedDict):
        errors: Annotated[list[str], operator.add]

    def overwrite_left(state: BufferState) -> dict[str, Any]:
        return {"errors": Overwrite(["left"])}

    def overwrite_right(state: BufferState) -> dict[str, Any]:
        return {"errors": Overwrite(["right"])}

    builder = StateGraph(BufferState)
    builder.add_node("left", overwrite_left)
    builder.add_node("right", overwrite_right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)

    failed = False
    try:
        builder.compile().invoke({"errors": []})
    except InvalidUpdateError as exc:
        failed = True
        print(f"  一步里两个 Overwrite 被拦下：{str(exc).splitlines()[0]}")
    ensure(failed, "同一个 super-step 里同一个键只能有一个 Overwrite，应当抛 InvalidUpdateError")

    # Overwrite 会顺手丢掉同一步其它节点对同一键的普通写入
    def plain_writer(state: BufferState) -> dict[str, Any]:
        return {"errors": ["plain"]}

    mixed_builder = StateGraph(BufferState)
    mixed_builder.add_node("overwrite", overwrite_left)
    mixed_builder.add_node("plain", plain_writer)
    mixed_builder.add_edge(START, "overwrite")
    mixed_builder.add_edge(START, "plain")
    mixed_builder.add_edge("overwrite", END)
    mixed_builder.add_edge("plain", END)
    mixed_result = mixed_builder.compile().invoke({"errors": []})

    print(f"  Overwrite 与普通写入同一步：{mixed_result['errors']}")
    ensure(
        mixed_result["errors"] == ["left"],
        f"Overwrite 应当独占该键，实际 {mixed_result['errors']}",
    )


# ------------------------------------------------------------ 三套 schema


def check_unknown_key_dropped() -> None:
    """坑：写一个 schema 里没声明的键，不报错、不生效，被静默丢掉。"""
    section("坑：未声明的键被静默丢弃")

    def node(state: TypedState) -> dict[str, Any]:
        """``typo`` 不在 TypedState 里，写了也白写。"""
        return {"text": "写好了", "typo": "这个键没人声明"}

    builder = StateGraph(TypedState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    result = builder.compile().invoke({"text": ""})

    print(f"  返回值：{result}")
    ensure(
        "typo" not in result,
        f"未声明的键不该出现在状态里，实际 {result}",
    )
    ensure(
        result == {"text": "写好了"},
        f"只有声明过的键会进状态，实际 {result}",
    )


class InputState(TypedDict):
    """调用方唯一允许传的键。"""

    user_input: str


class OutputState(TypedDict):
    """调用方唯一能拿到的键。"""

    graph_output: str


class OverallState(TypedDict):
    """图内部真正的状态：输入键、中间键、输出键都在这儿。"""

    foo: str
    user_input: str
    graph_output: str


class PrivateState(TypedDict):
    """私有通道：只给内部节点用，不进图的主 schema。"""

    bar: str


def check_input_output_schema() -> None:
    """distinct input/output schema 下，invoke 只返回输出字段。"""
    section("input / output schema")

    def node_1(state: InputState) -> OverallState:
        return {"foo": state["user_input"] + " name"}

    def node_2(state: OverallState) -> PrivateState:
        return {"bar": state["foo"] + " is"}

    def node_3(state: PrivateState) -> OutputState:
        return {"graph_output": state["bar"] + " Lance"}

    builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", "node_3")
    builder.add_edge("node_3", END)
    graph = builder.compile()

    result = graph.invoke({"user_input": "My"})
    print(f"  invoke 返回：{result}")

    ensure(
        set(result) == {"graph_output"},
        f"invoke 应当只返回 output schema 的字段，实际 {sorted(result)}",
    )
    ensure(
        result["graph_output"] == "My name is Lance",
        f"输出内容不对，实际 {result['graph_output']}",
    )


def check_private_channel_still_streams() -> None:
    """私有通道不出现在 invoke 结果里，却照样出现在 values 流里。"""
    section("私有通道与 values 流")

    def node_1(state: InputState) -> OverallState:
        return {"foo": state["user_input"] + " name"}

    def node_2(state: OverallState) -> PrivateState:
        return {"bar": state["foo"] + " is"}

    def node_3(state: PrivateState) -> OutputState:
        return {"graph_output": state["bar"] + " Lance"}

    builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", "node_3")
    builder.add_edge("node_3", END)
    graph = builder.compile()

    snapshots = list(graph.stream({"user_input": "My"}, stream_mode="values"))
    for snapshot in snapshots:
        print(f"  values 快照：{snapshot}")

    leaked = [s for s in snapshots if "bar" in s]
    ensure(leaked, "私有通道 bar 应当出现在 values 流里")
    ensure(
        all("bar" not in s for s in snapshots if s == {"graph_output": "My name is Lance"}),
        "invoke 口径的输出里不该有私有通道",
    )

    filtered = list(
        graph.stream(
            {"user_input": "My"},
            stream_mode="values",
            output_keys=["graph_output"],
        )
    )
    print(f"  output_keys 过滤后：{filtered[-1]}")
    ensure(
        all(set(s) <= {"graph_output"} for s in filtered),
        f"output_keys 过滤后不该出现别的通道，实际 {filtered}",
    )


# ------------------------------------------------------------ add_messages


def check_add_messages() -> None:
    """add_messages：追加 + 按 id 覆盖 + 简写格式反序列化。"""
    section("add_messages")

    class ChatState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def reply(state: ChatState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Hello!", id="m1")]}

    builder = StateGraph(ChatState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "Hi"}]})
    print(f"  简写输入后的消息：{result['messages']}")
    ensure(len(result["messages"]) == 2, f"应当有两条消息，实际 {len(result['messages'])}")
    ensure(
        result["messages"][0].content == "Hi",
        f"简写格式应当被还原成消息对象，实际 {result['messages'][0]}",
    )
    ensure(
        result["messages"][1].content == "Hello!",
        f"第二条应当是模型回复，实际 {result['messages'][1]}",
    )

    def edit(state: ChatState) -> dict[str, Any]:
        """同 id 的消息会被**替换**，而不是再追加一条。"""
        return {"messages": [AIMessage(content="Hello, edited!", id="m1")]}

    edit_builder = StateGraph(ChatState)
    edit_builder.add_node("reply", reply)
    edit_builder.add_node("edit", edit)
    edit_builder.add_edge(START, "reply")
    edit_builder.add_edge("reply", "edit")
    edit_builder.add_edge("edit", END)
    edit_result = edit_builder.compile().invoke({"messages": [{"role": "user", "content": "Hi"}]})

    print(f"  按 id 覆盖后的消息数：{len(edit_result['messages'])}")
    ensure(
        len(edit_result["messages"]) == 2,
        f"同 id 的消息应当被替换而不是追加，实际 {len(edit_result['messages'])} 条",
    )
    ensure(
        edit_result["messages"][1].content == "Hello, edited!",
        f"同 id 消息的内容应当被覆盖，实际 {edit_result['messages'][1].content}",
    )


# ------------------------------------------------------------ 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型产一条消息，验证 add_messages 的形状。

    只校验形状不校验内容——模型说什么不可控，但消息对象的类型和通道结构是稳定的。
    """
    banner("第 3 章 · 状态定义与归约器（真实模型路径）")
    if not live_enabled():
        print("未开启真实模型（LANGGRAPH_LAB_LIVE != 1），跳过。")
        return

    class LiveState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def ask(state: LiveState) -> dict[str, Any]:
        model = real_chat_model()
        reply = model.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(LiveState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [human("用一句话介绍 LangGraph 的状态机。")]})
    print(f"  真实回复：{result['messages'][-1].content}")

    ensure(
        isinstance(result["messages"], list),
        f"messages 通道应当是列表，实际 {type(result['messages'])}",
    )
    ensure(
        len(result["messages"]) == 2,
        f"应当有用户消息和模型回复两条，实际 {len(result['messages'])}",
    )
    ensure(
        hasattr(result["messages"][-1], "content"),
        "模型回复应当是带 content 属性的消息对象",
    )
    print("\n✓ 第 3 章真实路径通过")


# ------------------------------------------------------------ 入口


def run() -> None:
    """跑第 3 章实验。"""
    banner("第 3 章 · 状态定义与归约器")
    check_schema_forms()
    check_reducer_merge_semantics()
    check_overwrite_bypasses_reducer()
    check_unknown_key_dropped()
    check_input_output_schema()
    check_private_channel_still_streams()
    check_add_messages()
    rule("归约器决定并发时怎么合并；Overwrite 绕过它；schema 三件套决定谁能看见什么")
    print("\n✓ 第 3 章通过")
