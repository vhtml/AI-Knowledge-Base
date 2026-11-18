"""第 4 章实验：流式输出。

这个模块解决什么问题：流式输出看着只是"把结果分几次给"，实际上它是**一套
新的数据形状**。同一个模型回答，非流式下是一个字符串，流式下是一串增量分片；
拼错一个运算符，答案就没了。本章把九件事变成可执行的断言：

1. **delta 是增量不是全量**——``+=`` 还原全文，``=`` 只剩最后一个分片；
2. **流式与非流式结果一致**——同一段脚本，两条路径拼出同一个答案；
3. **分片有五类**——文本增量、工具名分片、工具参数分片、finish_reason、usage；
4. **finish_reason 在末尾**——不是每个分片都有；
5. **usage 可能单独占一个分片**——解析器不认它，成本统计永远是 0；
6. **工具调用的流式是另一套形状**——参数是分片的 JSON 字符串，名字也会被切碎；
7. **TTFT ≠ 总延迟**——流式只改善首字延迟，不改善总延迟；
8. **取消**——客户端断开后不再拉取分片，剩下的 token 不用付钱；
9. **半截流**——中途断开时已收到的部分可用，但必须能标记"没正常结束"。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, field

from ..labkit import (
    ChatMessage,
    ChatResponse,
    StreamChunk,
    banner,
    ensure,
    rule,
    scripted_client,
    section,
)

# ---------------------------------------------------------------- 脚本常量


ANSWER_PARTS: tuple[str, ...] = ("流式", "输出", "的", "每个分片", "只带", "新增内容")
"""一段普通文本的流式脚本——每个元素是一个分片。"""

ANSWER = "".join(ANSWER_PARTS)
"""这些分片拼起来的完整答案。"""

USAGE: dict[str, int] = {
    "prompt_tokens": 120,
    "completion_tokens": 14,
    "total_tokens": 134,
}
"""一段脚本对应的 token 统计。"""

TOOL_ARGS: dict[str, str] = {"order_id": "A-1001"}
"""工具调用的参数——流式下它是被切碎的 JSON 片段，拼回来才能解析。"""

TOOL_ARGS_JSON = '{"order_id": "A-1001"}'
"""上面那组参数的 JSON 文本形态。"""


# ---------------------------------------------------------------- 流式数据形状


@dataclass
class ToolCallDelta:
    """流式工具调用里的一个分片——对应厂商协议 ``delta.tool_calls`` 的元素。

    三个 delta 都是**增量**：名字可能被切碎，参数是 JSON 片段而不是完整 JSON。
    """

    index: int
    """第几个工具调用——并行调用靠它区分，串成一团就分不清了。"""
    call_id: str = ""
    """工具调用 id，一般整块到达。"""
    name_delta: str = ""
    """工具名的增量片段。"""
    arguments_delta: str = ""
    """参数 JSON 的增量片段。"""


@dataclass
class StreamEvent:
    """一个流式分片——``labkit.StreamChunk`` 的超集。

    labkit 的 ``StreamChunk`` 只建模了"纯文本流"；真实协议里同一个分片还可能
    带 ``tool_calls``。这里补上这一维，两个类型的 ``delta`` / ``finish_reason``
    / ``usage`` 字段名完全一致，所以拼接器对两者一视同仁。
    """

    delta: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolCallDraft:
    """累积中的工具调用——名字和参数都是分片拼起来的。"""

    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamResult:
    """一次流式会话累积下来的完整结果。"""

    text: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    chunk_count: int = 0
    finished: bool = False
    """收到过 finish_reason 才算正常结束——这是半截流的判据。"""
    _drafts: dict[int, ToolCallDraft] = field(default_factory=dict, repr=False)

    @property
    def tool_calls(self) -> list[ToolCallDraft]:
        """按 index 排序后的工具调用草稿。"""
        return [self._drafts[index] for index in sorted(self._drafts)]

    @property
    def truncated(self) -> bool:
        """撞到输出上限（``finish_reason == "length"``）时为真。"""
        return self.finish_reason == "length"

    @property
    def incomplete(self) -> bool:
        """流没有正常结束——中途断开时为真。"""
        return not self.finished

    def draft(self, index: int) -> ToolCallDraft:
        """按 index 取（必要时新建）工具调用草稿。"""
        draft = self._drafts.get(index)
        if draft is None:
            draft = ToolCallDraft(index=index)
            self._drafts[index] = draft
        return draft


class StreamAssembler:
    """把流式分片累积成一个完整结果。

    这是本章的核心交付物。唯一需要记住的规则：**delta 是增量，一律 ``+=``**。
    """

    def __init__(self) -> None:
        self.result = StreamResult()

    def feed(self, chunk: StreamChunk | StreamEvent) -> None:
        """吃进一个分片，更新累积结果。"""
        result = self.result
        result.chunk_count += 1

        # 1) 文本增量：delta 只带新增字符，必须累加
        result.text += chunk.delta

        # 2) 工具调用：名字和参数都是分片，各自累加
        for piece in getattr(chunk, "tool_calls", None) or []:
            draft = result.draft(piece.index)
            draft.call_id += piece.call_id
            draft.name += piece.name_delta
            draft.arguments += piece.arguments_delta

        # 3) finish_reason：只有末尾那一两个分片带
        if chunk.finish_reason:
            result.finish_reason = chunk.finish_reason
            result.finished = True

        # 4) usage：可能单独占一个分片，不能只盯着某一个分片找
        if chunk.usage:
            result.usage.update(chunk.usage)


# ---------------------------------------------------------------- 脚本构造


def text_script(with_usage: bool = False) -> list[StreamChunk]:
    """一段普通文本的流式脚本。

    ``with_usage=True`` 时在末尾追加一个**只带 usage 的分片**——
    真实协议打开 ``include_usage`` 时就是这个形状。
    """
    chunks = [StreamChunk(delta=part) for part in ANSWER_PARTS]
    chunks.append(StreamChunk(delta="", finish_reason="stop"))
    if with_usage:
        chunks.append(StreamChunk(usage=dict(USAGE)))
    return chunks


def tool_stream_script() -> list[StreamEvent]:
    """一段"流式工具调用"的分片脚本。

    刻意把工具名和参数都切碎——真实协议里就是这样：名字可能只到一半，
    参数是分片到达的 JSON 片段。
    """
    return [
        StreamEvent(
            tool_calls=[ToolCallDelta(index=0, call_id="call_1", name_delta="查订")]
        ),
        StreamEvent(
            tool_calls=[
                ToolCallDelta(index=0, name_delta="单", arguments_delta='{"order_id"')
            ]
        ),
        StreamEvent(
            tool_calls=[ToolCallDelta(index=0, arguments_delta=': "A-1001"}')]
        ),
        StreamEvent(finish_reason="tool_calls"),
    ]


def parallel_tool_script() -> list[StreamEvent]:
    """两次并行工具调用的分片脚本——分片是**交错**到达的。

    靠 index 区分两次调用；不按 index 归位就会把参数混在一起。
    """
    return [
        StreamEvent(
            tool_calls=[ToolCallDelta(index=0, call_id="call_a", name_delta="查订")]
        ),
        StreamEvent(
            tool_calls=[ToolCallDelta(index=1, call_id="call_b", name_delta="查物")]
        ),
        StreamEvent(
            tool_calls=[
                ToolCallDelta(index=0, name_delta="单", arguments_delta='{"order_id"')
            ]
        ),
        StreamEvent(
            tool_calls=[
                ToolCallDelta(index=1, name_delta="流", arguments_delta='{"tracking"')
            ]
        ),
        StreamEvent(
            tool_calls=[ToolCallDelta(index=0, arguments_delta=': "A-1001"}')]
        ),
        StreamEvent(
            tool_calls=[ToolCallDelta(index=1, arguments_delta=': "SF123"}')]
        ),
        StreamEvent(finish_reason="tool_calls"),
    ]


# ---------------------------------------------------------------- 1. delta 是增量


def check_delta_is_increment() -> None:
    """delta 是增量不是全量——``+=`` 还原全文，``=`` 只剩最后一个分片。"""
    section("delta 是增量：用 = 拼只会剩最后一个分片")

    # 只取文本分片来看：末尾那两个元数据分片不参与这个对照
    chunks = [StreamChunk(delta=part) for part in ANSWER_PARTS]

    # 正确写法：累加
    correct = ""
    for chunk in chunks:
        correct += chunk.delta
    print(f"  正确拼接 += → {correct!r}")

    # 错误写法：覆盖——每次把前面的内容冲掉
    wrong = ""
    for chunk in chunks:
        wrong = chunk.delta
    print(f"  错误拼接  = → {wrong!r}")

    ensure(
        correct == ANSWER,
        f"增量拼接应当还原完整文本，期望 {ANSWER!r}，实际 {correct!r}",
    )
    ensure(
        wrong == chunks[-1].delta,
        f"覆盖式拼接应当只剩最后一个分片，期望 {chunks[-1].delta!r}，实际 {wrong!r}",
    )
    ensure(
        wrong != correct,
        "覆盖式拼接的结果必须和正确结果不同——否则说明这个脚本只有一个分片，测不出坑",
    )
    ensure(
        len(wrong) < len(correct),
        f"覆盖式拼接丢掉的内容应当远多于保留的，"
        f"实际只留了 {len(wrong)} / {len(correct)} 个字符",
    )
    ensure(
        correct.endswith(wrong),
        f"覆盖式拼接剩下的应当只是完整文本的尾巴，实际 {wrong!r}",
    )
    print("  → 这是流式最常见的 bug：delta 是新增内容，不是全量")


# ---------------------------------------------------------------- 2. 两条路径一致


def check_stream_matches_complete() -> None:
    """同一段脚本，流式拼出来的结果必须等于非流式的 content。"""
    section("流式与非流式：最终内容必须一致")

    # ScriptedClient 的 complete 和 stream 共用一个 cursor，
    # 所以要比两条路径必须开两个客户端——流式和非流式本来就是两套协议。
    streaming_client = scripted_client(streams=[text_script(with_usage=True)])
    plain_client = scripted_client(
        responses=[
            ChatResponse(content=ANSWER, finish_reason="stop", usage=dict(USAGE))
        ]
    )

    assembler = StreamAssembler()
    for chunk in streaming_client.stream([ChatMessage(role="user", content="讲讲流式")]):
        assembler.feed(chunk)

    plain = plain_client.complete([ChatMessage(role="user", content="讲讲流式")])

    print(f"  流式拼接 → {assembler.result.text!r}")
    print(f"  非流式   → {plain.content!r}")

    ensure(
        assembler.result.text == plain.content,
        f"流式拼出来的内容必须等于非流式的 content，"
        f"期望 {plain.content!r}，实际 {assembler.result.text!r}",
    )
    ensure(
        assembler.result.finish_reason == plain.finish_reason,
        f"finish_reason 必须一致，期望 {plain.finish_reason!r}，"
        f"实际 {assembler.result.finish_reason!r}",
    )
    ensure(
        assembler.result.usage == plain.usage,
        f"usage 必须一致，期望 {plain.usage!r}，实际 {assembler.result.usage!r}",
    )
    print("  → 流式只是换个送达方式，不是换个答案")


# ---------------------------------------------------------------- 3. 分片类型


def check_chunk_kinds() -> None:
    """一个流里有五类分片，各自带不同的信息。"""
    section("分片有五类：文本 / 工具名 / 工具参数 / finish_reason / usage")

    chunks = text_script(with_usage=True)
    for index, chunk in enumerate(chunks):
        kinds: list[str] = []
        if chunk.delta:
            kinds.append("文本增量")
        if chunk.finish_reason:
            kinds.append("finish_reason")
        if chunk.usage:
            kinds.append("usage")
        label = "+".join(kinds) if kinds else "空分片"
        print(f"  #{index:<2} delta={chunk.delta!r:<14} 类型={label}")

    # 不带 usage 的流：最后一个分片带 finish_reason
    plain_chunks = text_script()
    ensure(
        plain_chunks[-1].finish_reason == "stop",
        f"不带 usage 的流，最后一个分片应当带 finish_reason='stop'，"
        f"实际 {plain_chunks[-1].finish_reason!r}",
    )

    # finish_reason 只有末尾那一两个分片带，不是每个都有
    carriers = [i for i, c in enumerate(chunks) if c.finish_reason]
    ensure(
        len(carriers) == 1,
        f"整个流里应当只有一个分片带 finish_reason，实际有 {len(carriers)} 个：{carriers}",
    )
    ensure(
        carriers[0] >= len(chunks) - 2,
        f"finish_reason 应当落在末尾两三个分片内，实际在第 {carriers[0]} 个（共 {len(chunks)} 个）",
    )
    ensure(
        chunks[-1].finish_reason is None,
        "末尾单独发 usage 的那个分片不带 finish_reason",
    )
    print("  → finish_reason 是末尾的一次性信号，不该在每个分片上找")


# ---------------------------------------------------------------- 4. usage


def check_usage_tail() -> None:
    """usage 可能单独占末尾一个分片——不认它，成本统计永远是 0。"""
    section("usage：它可能单独占一个分片")

    chunks = text_script(with_usage=True)
    usage_chunks = [c for c in chunks if c.usage]
    ensure(
        len(usage_chunks) == 1,
        f"应当恰好有一个分片带 usage，实际 {len(usage_chunks)} 个",
    )
    ensure(
        usage_chunks[0].delta == "",
        f"usage 分片不带文本增量，实际 delta={usage_chunks[0].delta!r}",
    )

    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    usage = assembler.result.usage
    print(f"  扫全部分片拼出来的 usage：{usage}")

    ensure(
        usage.get("total_tokens", 0) > 0,
        f"usage 应当有 token 统计，实际 {usage!r}",
    )
    ensure(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        f"total_tokens 应当等于输入加输出，实际 {usage!r}",
    )

    # 反例：只在"带 finish_reason 的那个分片"上读 usage——这是真实代码里常见的写法
    naive: dict[str, int] = {}
    for chunk in chunks:
        if chunk.finish_reason:
            naive = chunk.usage
    ensure(
        naive == {},
        f"只认 finish_reason 分片就读不到 usage，实际拿到了 {naive!r}",
    )
    ensure(
        assembler.result.usage != naive,
        "扫全部分片才能拿到 usage——只盯着一个分片读会永远读到 0",
    )
    print("  → usage 要扫全流；只在一个分片上读，成本统计永远是 0")


# ---------------------------------------------------------------- 5. 流式工具调用


def check_tool_call_stream() -> None:
    """工具调用的流式是另一套形状：参数是分片的 JSON，名字也会被切碎。"""
    section("流式工具调用：名字和参数都是分片")

    client = scripted_client(streams=[tool_stream_script()])
    assembler = StreamAssembler()
    for chunk in client.stream([ChatMessage(role="user", content="查订单 A-1001")]):
        assembler.feed(chunk)

    calls = assembler.result.tool_calls
    ensure(
        len(calls) == 1,
        f"应当只有一次工具调用，实际 {len(calls)} 次",
    )
    call = calls[0]
    print(f"  拼回来的名字：{call.name!r}")
    print(f"  拼回来的参数：{call.arguments!r}")

    ensure(
        call.name == "查订单",
        f"工具名应当被拼成 '查订单'，实际 {call.name!r}",
    )
    ensure(
        call.call_id == "call_1",
        f"call_id 应当保留，实际 {call.call_id!r}",
    )
    ensure(
        call.arguments == TOOL_ARGS_JSON,
        f"参数应当被拼成完整 JSON 文本，期望 {TOOL_ARGS_JSON!r}，实际 {call.arguments!r}",
    )

    # 单个分片的参数片段不是合法 JSON——这正是必须拼完整才能解析的原因
    fragment = '{"order_id"'
    try:
        json.loads(fragment)
    except json.JSONDecodeError:
        parsed_fragment = None
    else:
        parsed_fragment = {"ok": True}
    ensure(
        parsed_fragment is None,
        f"单个参数分片本来就不该能被解析成对象，实际解析成了 {parsed_fragment!r}",
    )

    parsed = json.loads(call.arguments)
    print(f"  json.loads 结果：{parsed}")
    ensure(
        parsed == TOOL_ARGS,
        f"拼完整后应当能解析成 {TOOL_ARGS!r}，实际 {parsed!r}",
    )
    ensure(
        assembler.result.finish_reason == "tool_calls",
        f"工具调用的流应当以 finish_reason='tool_calls' 收尾，"
        f"实际 {assembler.result.finish_reason!r}",
    )
    print("  → 流式下的工具调用是另一套形状：参数拼完整才能 json.loads")


def check_parallel_tool_calls() -> None:
    """并行调用靠 index 区分——不归位就会把两个调用的参数混在一起。"""
    section("并行工具调用：靠 index 归位，别把参数混了")

    client = scripted_client(streams=[parallel_tool_script()])
    assembler = StreamAssembler()
    for chunk in client.stream([ChatMessage(role="user", content="查订单并查物流")]):
        assembler.feed(chunk)

    calls = assembler.result.tool_calls
    ensure(
        len(calls) == 2,
        f"应当累积出两次工具调用，实际 {len(calls)} 次",
    )
    ensure(
        [call.name for call in calls] == ["查订单", "查物流"],
        f"按 index 排序后工具名应当是 ['查订单', '查物流']，"
        f"实际 {[call.name for call in calls]}",
    )
    ensure(
        [call.call_id for call in calls] == ["call_a", "call_b"],
        f"call_id 应当各归各位，实际 {[call.call_id for call in calls]}",
    )

    parsed = [json.loads(call.arguments) for call in calls]
    print(f"  两次调用的参数：{parsed}")
    ensure(
        parsed == [{"order_id": "A-1001"}, {"tracking": "SF123"}],
        f"两次调用的参数不该被混在一起，期望 "
        f"[{{'order_id': 'A-1001'}}, {{'tracking': 'SF123'}}]，实际 {parsed!r}",
    )
    print("  → 并行调用靠 index 归位；参数混起来是最难查的一类 bug")


# ---------------------------------------------------------------- 6. TTFT vs 总延迟


def token_timeline(token_count: int, prefill_ms: float, per_token_ms: float) -> list[float]:
    """每个 token 的到达时刻。

    模型先做 prefill（把上下文算成 KV cache），再逐 token 解码。
    """
    return [prefill_ms + index * per_token_ms for index in range(token_count)]


@dataclass(frozen=True)
class Latency:
    """一次调用的两个延迟。"""

    ttft_ms: float
    """首字延迟——用户看到第一个字符要等多久。"""
    total_ms: float
    """总延迟——整个回答拿到要等多久。"""


def simulate_streaming(timeline: Sequence[float]) -> Latency:
    """流式的送达策略：第一个 token 到达就发出去，最后一个到达时整条流结束。"""
    if not timeline:
        return Latency(ttft_ms=0.0, total_ms=0.0)
    return Latency(ttft_ms=timeline[0], total_ms=timeline[-1])


def simulate_non_streaming(timeline: Sequence[float]) -> float:
    """非流式的送达策略：全部 token 生成完才一次性返回。"""
    if not timeline:
        return 0.0
    return timeline[-1]


def check_ttft_vs_total() -> None:
    """流式只改善首字延迟，不改善总延迟。"""
    section("TTFT vs 总延迟：流式改善的是前者")

    prefill_ms = 800.0
    per_token_ms = 120.0
    timeline = token_timeline(len(ANSWER_PARTS), prefill_ms, per_token_ms)

    # 真的把脚本分片喂一遍，按到达时刻记录
    client = scripted_client(streams=[text_script()])
    assembler = StreamAssembler()
    arrivals: list[float] = []
    for index, chunk in enumerate(client.stream([ChatMessage(role="user", content="讲讲流式")])):
        arrived_at = timeline[index] if index < len(timeline) else timeline[-1]
        arrivals.append(arrived_at)
        assembler.feed(chunk)

    streaming = simulate_streaming(arrivals)
    non_streaming = simulate_non_streaming(timeline)

    print(f"  TTFT           = {streaming.ttft_ms:.0f} ms")
    print(f"  流式总延迟     = {streaming.total_ms:.0f} ms")
    print(f"  非流式总延迟   = {non_streaming:.0f} ms")

    ensure(
        streaming.ttft_ms < streaming.total_ms,
        f"TTFT 应当远小于总延迟，实际 TTFT={streaming.ttft_ms} / total={streaming.total_ms}",
    )
    ensure(
        streaming.total_ms == non_streaming,
        f"流式的总延迟应当和非流式完全一样，实际 {streaming.total_ms} vs {non_streaming}",
    )
    ensure(
        streaming.ttft_ms <= non_streaming,
        f"流式的首字延迟不可能比总延迟还大，实际 {streaming.ttft_ms} vs {non_streaming}",
    )
    ensure(
        assembler.result.text == ANSWER,
        f"流式拼出来的文本仍然要完整，期望 {ANSWER!r}，实际 {assembler.result.text!r}",
    )
    print(
        f"  → 用户感知的等待从 {non_streaming:.0f} ms 降到 {streaming.ttft_ms:.0f} ms，"
        f"但总工作量一点没少"
    )


# ---------------------------------------------------------------- 7. 取消


def check_cancellation() -> None:
    """客户端断开后就不再拉分片——剩下的 token 不用继续付钱。"""
    section("取消：break 之后就不再消费分片")

    total_chunks = len(text_script())
    client = scripted_client(streams=[text_script()])
    produced: list[StreamChunk] = []

    def tracked() -> Iterator[StreamChunk]:
        """包一层计数器——用来观察到底从上游拉了多少个分片出来。"""
        for chunk in client.stream([ChatMessage(role="user", content="讲讲流式")]):
            produced.append(chunk)
            yield chunk

    assembler = StreamAssembler()
    limit = 3
    # closing 保证 break 之后生成器被关闭——这正是"客户端断开"该做的事
    with closing(tracked()) as stream:
        for index, chunk in enumerate(stream):
            assembler.feed(chunk)
            if index + 1 >= limit:
                break

    consumed = len(produced)
    print(f"  消费了 {consumed} 个分片，整条流共 {total_chunks} 个")

    ensure(
        consumed == limit,
        f"应当只消费 {limit} 个分片，实际 {consumed} 个",
    )
    ensure(
        consumed < total_chunks,
        f"取消后消费的分片数应当少于总分片数，实际 {consumed} / {total_chunks}",
    )
    ensure(
        assembler.result.text == "".join(ANSWER_PARTS[:limit]),
        f"已收到的部分应当仍是有效文本，期望 {''.join(ANSWER_PARTS[:limit])!r}，"
        f"实际 {assembler.result.text!r}",
    )
    ensure(
        assembler.result.incomplete,
        "被取消的流没有正常结束，应当被标记成 incomplete",
    )
    print("  → 取消 = 关闭生成；上游据此停止解码（已生成部分是否计费看厂商）")


# ---------------------------------------------------------------- 8. 半截流


def check_partial_stream() -> None:
    """中途断开：已收到的部分能用，但必须能标记"没正常结束"。"""
    section("半截流：收到的部分能用，但不算完成")

    partial = [StreamChunk(delta=part) for part in ANSWER_PARTS[:3]]
    client = scripted_client(streams=[partial])
    assembler = StreamAssembler()
    for chunk in client.stream([ChatMessage(role="user", content="讲讲流式")]):
        assembler.feed(chunk)

    result = assembler.result
    print(f"  收到的部分：{result.text!r}")
    print(f"  finish_reason：{result.finish_reason!r}  finished={result.finished}")

    ensure(
        result.text == "".join(ANSWER_PARTS[:3]),
        f"半截流里收到的文本应当仍然可用，期望 {''.join(ANSWER_PARTS[:3])!r}，"
        f"实际 {result.text!r}",
    )
    ensure(
        result.finish_reason is None,
        f"半截流没有 finish_reason，实际 {result.finish_reason!r}",
    )
    ensure(
        result.incomplete,
        "半截流必须能被标记成未正常结束",
    )
    ensure(
        result.text.startswith(ANSWER[:2]),
        f"半截流的文本是完整回答的前缀，实际 {result.text!r}",
    )

    # 对照：finish_reason == "length" 是"被输出上限截断"，不是"连接断了"
    length_script = [StreamChunk(delta=part) for part in ANSWER_PARTS] + [
        StreamChunk(delta="", finish_reason="length")
    ]
    length_client = scripted_client(streams=[length_script])
    length_assembler = StreamAssembler()
    for chunk in length_client.stream([ChatMessage(role="user", content="讲讲流式")]):
        length_assembler.feed(chunk)

    ensure(
        length_assembler.result.finished,
        "带 finish_reason 的流算正常结束",
    )
    ensure(
        length_assembler.result.truncated,
        "finish_reason == 'length' 表示被输出上限截断",
    )
    ensure(
        not length_assembler.result.incomplete,
        "被 max_tokens 截断不等于连接断了——两者要能区分",
    )
    print("  → 半截流和被截断长得像，但一个没 finish_reason、一个是 length")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：真调一次流式，只校验形状。"""
    banner("第 4 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client(temperature=0.0)
    assembler = StreamAssembler()
    for chunk in client.stream(
        [ChatMessage(role="user", content="用一句话说明什么是 API。")]
    ):
        assembler.feed(chunk)

    result = assembler.result
    print(f"  text: {result.text!r}")
    print(f"  finish_reason: {result.finish_reason}")
    print(f"  usage: {result.usage}")
    print(f"  分片数: {result.chunk_count}")

    ensure(bool(result.text), "真实模型应当流式返回内容")
    ensure(
        result.finish_reason in {"stop", "length"},
        f"finish_reason 异常：{result.finish_reason!r}",
    )
    ensure(
        result.usage.get("total_tokens", 0) > 0,
        f"流式也应当能拿到 usage，实际 {result.usage!r}",
    )
    ensure(result.finished, "真实模型的流应当正常结束")


def run() -> None:
    """跑第 4 章实验。"""
    banner("第 4 章 · 流式输出")
    check_delta_is_increment()
    check_stream_matches_complete()
    check_chunk_kinds()
    check_usage_tail()
    check_tool_call_stream()
    check_parallel_tool_calls()
    check_ttft_vs_total()
    check_cancellation()
    check_partial_stream()
    rule(
        "delta 是增量，一律 +=；工具调用是另一套形状；"
        "流式只改善首字延迟，不改善总延迟；末尾的 finish_reason 和 usage 不能丢"
    )
    print("\n✓ 第 4 章通过")
