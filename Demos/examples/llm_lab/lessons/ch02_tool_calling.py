"""第 2 章实验：工具调用协议。

这个模块解决什么问题：第 2 章讲的是"模型怎么表达'我要调这个工具'"。
这件事在协议上是**三段**——请求侧声明 ``tools``、响应侧交回 ``tool_calls``、
请求侧把结果作为 ``role="tool"`` 的消息回填。

模型不会执行你的函数。它只交回一份**调用意图**，剩下的事全是你的代码干的。
中间那一段最容易写错，有四个真实的故障面：

1. ``arguments`` 是 **JSON 字符串**不是 dict，而且解析可能失败；
2. ``tool_choice`` 的四种模式会让"模型到底会不会调"完全变样；
3. 并行调用时结果要按 ``tool_call_id`` 对齐，按顺序回填必错位；
4. 模型可能调用一个你压根没声明的工具。

所以本章把九件事变成可执行的断言：tools 载荷的结构、``tool_choice`` 四模式、
并行对齐、arguments 解析、``finish_reason`` 信号、幻觉工具、工具数量的上下文
成本、strict schema、一次完整往返。

本章的 :func:`apply_tool_choice` 在本地复刻厂商侧对 ``tool_choice`` 的处理规则
（真实端点是在服务端做这件事的），好处是四种模式的差异能被离线断言。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..labkit import (
    ChatMessage,
    ChatResponse,
    ToolSpec,
    banner,
    ensure,
    estimate_tokens,
    rule,
    scripted_client,
    section,
    tool_spec,
)

# ---------------------------------------------------------------- 协议辅助类型


ToolChoice = str | dict[str, Any]
"""``tool_choice`` 的取值：三个字符串模式，或"指定某个具体函数"的字典。"""

TOOL_CHOICE_MODES: tuple[str, ...] = ("auto", "none", "required")
"""三个字符串模式的取值顺序——第四种是指定具体函数。"""


@dataclass(frozen=True)
class ParsedArguments:
    """一次 ``arguments`` 解析的结果。

    成功和失败都要留痕：失败时 ``raw`` 保留原文、``error`` 写明原因，
    这样排查时能看见模型到底吐了什么，而不是只看到一个空 dict。
    """

    ok: bool
    data: dict[str, Any]
    raw: str
    error: str = ""


def make_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """造一条 ``tool_calls`` 元素。

    ``arguments`` 传的是 **JSON 字符串**——线上协议里它就是字符串，
    不是 dict。传 dict 进去会让序列化结果与真实请求体对不上。
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def order_tool() -> ToolSpec:
    """查订单工具——本章反复用到的那个。"""
    return tool_spec(
        "查订单",
        "按订单号查询订单状态与物流轨迹",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，形如 A-1001"}
            },
            "required": ["order_id"],
        },
    )


def weather_tool() -> ToolSpec:
    """查天气工具——和查订单长得不一样，用来对比工具描述的影响。"""
    return tool_spec(
        "查天气",
        "按城市查询未来两天的天气",
        {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    )


def build_request(
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSpec] | None = None,
    tool_choice: ToolChoice = "auto",
) -> dict[str, Any]:
    """构造一次请求体。

    结构与厂商 SDK 的参数一一对应，方便把这里的字段直接对到线上 payload。
    """
    payload: dict[str, Any] = {"messages": [m.to_api() for m in messages]}
    if tools:
        payload["tools"] = [t.to_api() for t in tools]
        payload["tool_choice"] = tool_choice
    return payload


def apply_tool_choice(
    candidate: ChatResponse,
    tools: Sequence[ToolSpec],
    tool_choice: ToolChoice = "auto",
) -> ChatResponse:
    """模拟厂商侧对 ``tool_choice`` 的处理。

    真实端点在服务端做这件事：把"模型本来想干什么"收敛成"协议允许干什么"。
    这里在本地复刻同一套规则，好让四种模式的差异可以被断言而不用联网。

    注意 ``required`` 只保证"发生了调用"，不保证参数对——参数对不对是模型的事。
    """
    declared = [spec.name for spec in tools]

    if tool_choice == "none":
        # 禁止调用：调用意图被剥掉，模型只能自己回答。
        return ChatResponse(
            content=candidate.content,
            tool_calls=[],
            finish_reason="stop",
            usage=candidate.usage,
            model=candidate.model,
        )

    if tool_choice == "required":
        # 必须调用：模型只想说话也不行，给它一个已声明的工具。
        calls = candidate.tool_calls or [
            make_tool_call("call_forced_1", declared[0] if declared else "", "{}")
        ]
        return ChatResponse(
            content=candidate.content,
            tool_calls=calls,
            finish_reason="tool_calls",
            usage=candidate.usage,
            model=candidate.model,
        )

    if isinstance(tool_choice, dict):
        # 指定函数：强制调这一个，哪怕模型本来想调别的。
        name = tool_choice["function"]["name"]
        return ChatResponse(
            content=candidate.content,
            tool_calls=[make_tool_call("call_forced_1", name, "{}")],
            finish_reason="tool_calls",
            usage=candidate.usage,
            model=candidate.model,
        )

    # auto：模型自己决定，原样透传。
    return candidate


def parse_arguments(raw: str) -> ParsedArguments:
    """把 ``arguments`` 的 JSON 字符串解析成 dict。

    解析失败**不能**静默变成空 dict——空 dict 会被当成"用户没填参数"，
    工具于是带着错误参数跑完全程，报错离根因十万八千里。
    """
    text = raw.strip()
    if not text:
        # 无参工具经常返回空字符串，这是合法的"空对象"，不是失败。
        return ParsedArguments(ok=True, data={}, raw=raw)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedArguments(ok=False, data={}, raw=raw, error=exc.msg)
    if not isinstance(value, dict):
        return ParsedArguments(
            ok=False, data={}, raw=raw, error=f"顶层是 {type(value).__name__}，不是对象"
        )
    return ParsedArguments(ok=True, data=value, raw=raw)


def backfill_tool_messages(
    calls: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
) -> list[ChatMessage]:
    """把工具执行结果回填成 ``role="tool"`` 的消息。

    **按 id 配对而不是按顺序**：并行调用时执行器返回结果的顺序不固定，
    按顺序回填会把结果挂到别的工具名下。
    """
    by_id = {item["tool_call_id"]: item for item in results}
    messages: list[ChatMessage] = []
    for call in calls:
        call_id = call["id"]
        name = call["function"]["name"]
        matched = by_id.get(call_id)
        content = (
            matched["content"]
            if matched
            else f"工具 {name} 没有返回结果（tool_call_id={call_id}）"
        )
        messages.append(
            ChatMessage(role="tool", content=content, tool_call_id=call_id, name=name)
        )
    return messages


def resolve_tool_call(
    call: dict[str, Any], tools: Sequence[ToolSpec]
) -> ToolSpec | None:
    """找到这次调用对应的工具声明；没声明就返回 ``None``。"""
    name = call["function"]["name"]
    for spec in tools:
        if spec.name == name:
            return spec
    return None


def unknown_tool_names(
    calls: Sequence[dict[str, Any]], tools: Sequence[ToolSpec]
) -> list[str]:
    """列出模型调用了、但请求里没声明的工具名。"""
    declared = {spec.name for spec in tools}
    names: list[str] = []
    for call in calls:
        name = call["function"]["name"]
        if name not in declared and name not in names:
            names.append(name)
    return names


def describe_unknown_tool(call: dict[str, Any], tools: Sequence[ToolSpec]) -> str:
    """生成一条能直接进日志的报错信息。

    必须同时说清"调错了什么"和"有哪些可用工具"——只说调错了，
    排查的人还得回头翻代码才知道该改成啥。
    """
    name = call["function"]["name"]
    available = "、".join(spec.name for spec in tools) or "（无）"
    return f"模型调用了未声明的工具 {name}；当前可用工具：{available}"


def needs_tool_execution(response: ChatResponse) -> bool:
    """判断这次响应要不要执行工具。

    以 ``tool_calls`` 是否非空为准：``finish_reason`` 是信号，
    但有些网关会把它写成 ``stop``，只信它就会漏执行。
    """
    return bool(response.tool_calls)


def tools_payload_tokens(tools: Sequence[ToolSpec]) -> int:
    """估算 ``tools`` 声明本身占多少 token。

    这部分**每轮都要重发**，是工具用多了之后上下文膨胀的主要来源。
    """
    payload = json.dumps([spec.to_api() for spec in tools], ensure_ascii=False)
    return estimate_tokens(payload)


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """本地假工具执行器——返回固定结果，重点是协议而不是业务。"""
    if name == "查订单":
        return f"订单 {arguments.get('order_id', '?')} 已发货"
    if name == "查天气":
        return f"{arguments.get('city', '?')} 晴，26℃"
    return f"未实现工具 {name}"


# ---------------------------------------------------------------- 1. tools 载荷


def check_tools_payload() -> None:
    """请求侧的 tools 声明结构必须和厂商协议一致。"""
    section("请求侧：tools 声明长什么样")

    tools = [order_tool(), weather_tool()]
    payload = build_request([ChatMessage(role="user", content="订单 A-1001 到哪了")], tools=tools)

    serialized = json.loads(json.dumps(payload, ensure_ascii=False))
    element = serialized["tools"][0]

    print(f"  tools[0] = {json.dumps(element, ensure_ascii=False)[:120]}…")

    ensure(
        element["type"] == "function",
        f"tools 元素的 type 必须是 function，实际 {element['type']!r}",
    )
    ensure(
        set(element["function"]) == {"name", "description", "parameters"},
        f"function 下应当只有 name/description/parameters，实际 {sorted(element['function'])}",
    )
    ensure(
        isinstance(element["function"]["parameters"], dict),
        "parameters 必须是一个 JSON Schema 对象",
    )
    ensure(
        element["function"]["parameters"].get("type") == "object",
        "parameters.type 必须是 object",
    )
    ensure(
        element["function"]["parameters"].get("required") == ["order_id"],
        f"必填字段应当写进 required，实际 {element['function']['parameters'].get('required')}",
    )
    ensure(
        bool(element["function"]["description"]),
        "description 不能为空——模型只看得懂 description 和 parameters",
    )
    ensure(
        " " not in element["function"]["name"],
        f"工具名不能带空格，实际 {element['function']['name']!r}",
    )
    ensure(
        serialized["tool_choice"] == "auto",
        f"带上 tools 时默认 tool_choice 应当是 auto，实际 {serialized['tool_choice']!r}",
    )

    # 不带工具时，请求里不该出现 tools / tool_choice
    bare = build_request([ChatMessage(role="user", content="你好")])
    ensure("tools" not in bare, "没声明工具时请求体不该有 tools 字段")
    ensure("tool_choice" not in bare, "没声明工具时请求体不该有 tool_choice 字段")
    print("  → 模型看不见你的函数体，只看得懂 name / description / parameters")


# ---------------------------------------------------------------- 2. tool_choice


def check_tool_choice_modes() -> None:
    """四种模式各自把"要不要调"变成什么样。"""
    section("tool_choice：四种模式，四种行为")

    tools = [order_tool(), weather_tool()]

    # 候选一：模型想调工具；候选二：模型想说句话
    wants_tool = ChatResponse(
        content="我来查一下",
        tool_calls=[make_tool_call("call_1", "查订单", '{"order_id": "A-1001"}')],
        finish_reason="tool_calls",
    )
    wants_text = ChatResponse(content="订单 A-1001 已发货", finish_reason="stop")

    # auto：模型自己决定，两种候选都原样透传
    auto_tool = apply_tool_choice(wants_tool, tools, "auto")
    auto_text = apply_tool_choice(wants_text, tools, "auto")
    ensure(
        len(auto_tool.tool_calls) == 1
        and auto_tool.tool_calls[0]["function"]["name"] == "查订单",
        f"auto 模式下模型想调就应当调，实际 {auto_tool.tool_calls}",
    )
    ensure(
        auto_text.tool_calls == [] and auto_text.content == "订单 A-1001 已发货",
        "auto 模式下模型想说话就应当说话，auto 不强推任何一边",
    )

    # none：禁止调用，哪怕模型想调也剥掉
    none_result = apply_tool_choice(wants_tool, tools, "none")
    ensure(
        none_result.tool_calls == [],
        f"none 模式下不该留下任何 tool_calls，实际 {none_result.tool_calls}",
    )
    ensure(
        none_result.finish_reason == "stop",
        f"none 模式下不该以 tool_calls 收尾，实际 {none_result.finish_reason!r}",
    )
    ensure(
        none_result.content == "我来查一下",
        "none 模式下模型只能用自己的文本回答",
    )

    # required：必须调用，哪怕模型只想说话
    required_result = apply_tool_choice(wants_text, tools, "required")
    ensure(
        len(required_result.tool_calls) == 1,
        f"required 模式下必须发生一次调用，实际 {required_result.tool_calls}",
    )
    ensure(
        required_result.tool_calls[0]["function"]["name"] in {t.name for t in tools},
        "required 模式下调用的必须是已声明的工具",
    )
    ensure(
        required_result.finish_reason == "tool_calls",
        f"required 模式下 finish_reason 应当是 tool_calls，实际 {required_result.finish_reason!r}",
    )

    # 指定函数：强制调这一个，哪怕模型想调别的
    specific: ToolChoice = {"type": "function", "function": {"name": "查天气"}}
    forced = apply_tool_choice(wants_tool, tools, specific)
    ensure(
        len(forced.tool_calls) == 1,
        f"指定函数模式下必须且只有一次调用，实际 {forced.tool_calls}",
    )
    ensure(
        forced.tool_calls[0]["function"]["name"] == "查天气",
        f"指定函数模式必须强制调用指定的工具，实际 {forced.tool_calls[0]['function']['name']!r}",
    )

    # 请求体里 tool_choice 的取值必须原样带出去
    for mode in TOOL_CHOICE_MODES:
        request = build_request(
            [ChatMessage(role="user", content="查订单")], tools=tools, tool_choice=mode
        )
        ensure(
            request["tool_choice"] == mode,
            f"tool_choice={mode} 应当原样透传，实际 {request['tool_choice']!r}",
        )
    specific_request = build_request(
        [ChatMessage(role="user", content="查天气")], tools=tools, tool_choice=specific
    )
    ensure(
        specific_request["tool_choice"] == specific,
        f"指定函数的 tool_choice 应当原样透传，实际 {specific_request['tool_choice']!r}",
    )

    print("  auto=模型自己决定 / none=禁止 / required=必须调 / 指定函数=强制调那一个")


# ---------------------------------------------------------------- 3. 并行调用


def check_parallel_calls() -> None:
    """一条 assistant 消息两个 tool_calls，结果要各自对齐。"""
    section("并行调用：结果必须按 tool_call_id 对齐")

    calls = [
        make_tool_call("call_a", "查订单", '{"order_id": "A-1001"}'),
        make_tool_call("call_b", "查天气", '{"city": "杭州"}'),
    ]
    assistant = ChatMessage(role="assistant", content="", tool_calls=calls)
    ensure(
        len(assistant.to_api()["tool_calls"]) == 2,
        "一条 assistant 消息可以携带多个 tool_calls",
    )

    # 执行器返回结果的顺序不固定——故意倒着给
    results = [
        {"tool_call_id": "call_b", "content": "杭州 晴，26℃"},
        {"tool_call_id": "call_a", "content": "订单 A-1001 已发货"},
    ]
    messages = backfill_tool_messages(calls, results)

    ensure(
        len(messages) == 2,
        f"两个 tool_calls 应当回填两条 tool 消息，实际 {len(messages)} 条",
    )
    ensure(
        [m.tool_call_id for m in messages] == ["call_a", "call_b"],
        f"tool_call_id 必须逐条对上，实际 {[m.tool_call_id for m in messages]}",
    )
    ensure(
        messages[0].content == "订单 A-1001 已发货",
        f"call_a 应当拿到订单结果，实际 {messages[0].content!r}",
    )
    ensure(
        messages[1].content == "杭州 晴，26℃",
        f"call_b 应当拿到天气结果，实际 {messages[1].content!r}",
    )

    # 反证：按顺序回填就是错位的形状
    naive = [
        ChatMessage(
            role="tool",
            content=item["content"],
            tool_call_id=calls[index]["id"],
        )
        for index, item in enumerate(results)
    ]
    ensure(
        naive[0].tool_call_id == "call_a" and naive[0].content == "杭州 晴，26℃",
        "按顺序回填会把天气结果挂到订单调用上——这就是错位的形状",
    )

    # 少带 tool_call_id 的 tool 消息：序列化后根本认不出是谁的结果
    broken = ChatMessage(role="tool", content="订单 A-1001 已发货")
    ensure(
        "tool_call_id" not in broken.to_api(),
        "缺 tool_call_id 的 tool 消息序列化后确实丢了对齐依据",
    )
    print("  → 配对靠 id 不靠顺序；少一条、错一位，模型就分不清是谁的结果")


# ---------------------------------------------------------------- 4. arguments


def check_arguments_parsing() -> None:
    """arguments 是 JSON 字符串；解析失败要留痕。"""
    section("arguments：是 JSON 字符串，失败必须留痕")

    call = make_tool_call("call_1", "查订单", '{"order_id": "A-1001", "retry": 2}')
    ensure(
        isinstance(call["function"]["arguments"], str),
        "线上协议里 arguments 是字符串，不是 dict",
    )

    parsed = parse_arguments(call["function"]["arguments"])
    ensure(parsed.ok, f"合法 JSON 应当解析成功，实际错误：{parsed.error}")
    ensure(
        parsed.data == {"order_id": "A-1001", "retry": 2},
        f"解析结果应当是等价的 dict，实际 {parsed.data}",
    )
    ensure(
        isinstance(parsed.data["retry"], int),
        "JSON 里的数字应当解析成 int 而不是字符串",
    )

    # 非法 JSON：保留原文 + 标记，不能静默变成空 dict
    broken_raw = '{"order_id": "A-1001",}'
    broken = parse_arguments(broken_raw)
    ensure(
        not broken.ok,
        "非法 JSON 必须被标记为解析失败，不能当成功",
    )
    ensure(
        broken.raw == broken_raw,
        f"解析失败必须保留原文，实际 {broken.raw!r}",
    )
    ensure(
        bool(broken.error),
        "解析失败必须给出原因，不能只留一个空 dict",
    )
    ensure(
        broken.data == {},
        "解析失败时参数应当是空，且靠 ok/error 标记区分于真的空参数",
    )

    # 空字符串是合法的"无参调用"，不是失败
    empty = parse_arguments("")
    ensure(
        empty.ok and empty.data == {},
        "空 arguments 应当解析成空对象，不是失败",
    )

    # 顶层不是对象也是失败
    not_object = parse_arguments('"A-1001"')
    ensure(
        not not_object.ok,
        f"顶层不是对象应当算失败，实际 ok={not_object.ok}",
    )
    print("  → 解析失败保留原文 + 标记；静默变空 dict 会把故障推到十万八千里外")


# ---------------------------------------------------------------- 5. finish_reason


def check_finish_reason_signal() -> None:
    """finish_reason == "tool_calls" 是"模型还要调工具"的信号。"""
    section("finish_reason == 'tool_calls'：模型还没说完")

    response = ChatResponse(
        content="",
        tool_calls=[make_tool_call("call_1", "查订单", '{"order_id": "A-1001"}')],
        finish_reason="tool_calls",
    )
    ensure(
        response.finish_reason == "tool_calls",
        f"调工具的响应 finish_reason 应当是 tool_calls，实际 {response.finish_reason!r}",
    )
    ensure(
        needs_tool_execution(response),
        "finish_reason 是 tool_calls 时必须进入工具执行分支",
    )

    plain = ChatResponse(content="订单 A-1001 已发货", finish_reason="stop")
    ensure(
        not needs_tool_execution(plain),
        "没有 tool_calls 的正常回答不该进入工具执行分支",
    )

    # 有些网关会把 finish_reason 写成 stop，但 tool_calls 非空——照样要执行
    mislabeled = ChatResponse(
        content="",
        tool_calls=[make_tool_call("call_2", "查订单", "{}")],
        finish_reason="stop",
    )
    ensure(
        needs_tool_execution(mislabeled),
        "finish_reason 不可信，tool_calls 非空就必须执行，否则调用被静默丢弃",
    )
    print("  → 判据是 tool_calls 非空，finish_reason 只是信号")


# ---------------------------------------------------------------- 6. 幻觉工具


def check_unknown_tool() -> None:
    """模型调用了没声明的工具，要能识别并列出可用清单。"""
    section("幻觉工具：调用了没声明的函数")

    tools = [order_tool(), weather_tool()]
    calls = [
        make_tool_call("call_1", "查订单", '{"order_id": "A-1001"}'),
        make_tool_call("call_2", "查快递轨迹", '{"tracking_id": "SF123"}'),
    ]

    unknown = unknown_tool_names(calls, tools)
    ensure(
        unknown == ["查快递轨迹"],
        f"应当只报出未声明的那个工具，实际 {unknown}",
    )
    ensure(
        resolve_tool_call(calls[0], tools) is not None,
        "已声明的工具应当能解析到声明",
    )
    ensure(
        resolve_tool_call(calls[1], tools) is None,
        "未声明的工具不应当被解析成某个已声明的工具",
    )

    message = describe_unknown_tool(calls[1], tools)
    print(f"  {message}")
    ensure(
        "查快递轨迹" in message,
        "报错信息必须说清调错了哪个工具",
    )
    ensure(
        "查订单" in message and "查天气" in message,
        "报错信息必须列出当前可用的工具清单",
    )
    print("  → 命中未声明工具时，错误信息里要带可用工具清单")


# ---------------------------------------------------------------- 7. 工具数量成本


def check_tool_count_cost() -> None:
    """工具越多，tools 载荷越大，且可能比整段对话还长。"""
    section("工具数量：schema 每轮都要重发")

    counts = [1, 2, 4, 8]
    tokens: list[int] = []
    for count in counts:
        many = [
            tool_spec(
                f"工具{index}",
                f"第 {index} 个工具的用途说明，用于演示工具数量对上下文的影响",
                {
                    "type": "object",
                    "properties": {"参数": {"type": "string", "description": "参数说明"}},
                    "required": ["参数"],
                },
            )
            for index in range(count)
        ]
        tokens.append(tools_payload_tokens(many))

    print(f"  工具数 {counts} → tools 载荷 token 估算 {tokens}")
    ensure(
        all(later > earlier for earlier, later in zip(tokens, tokens[1:])),
        f"工具数量增长时 tools 载荷 token 应当单调上升，实际 {tokens}",
    )

    # 8 个工具的声明，往往比整段对话还长
    conversation = [ChatMessage(role="user", content="查订单 A-1001")]
    conversation_tokens = estimate_tokens(
        json.dumps([m.to_api() for m in conversation], ensure_ascii=False)
    )
    ensure(
        tokens[-1] > conversation_tokens,
        f"8 个工具的声明（{tokens[-1]} token）应当比整段对话（{conversation_tokens} token）还长",
    )
    print("  → 工具不是越多越好：每轮都重发，且工具之间会互相干扰")


# ---------------------------------------------------------------- 8. strict schema


def check_strict_schema() -> None:
    """strict schema 的两条硬约束。"""
    section("strict schema：约束解码比提示词请求可靠")

    loose = tool_spec(
        "查订单",
        "按订单号查询订单状态",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["order_id"],
        },
    )
    strict = tool_spec(
        "查订单",
        "按订单号查询订单状态",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "note": {"type": "string", "nullable": True},
            },
            "required": ["order_id", "note"],
            "additionalProperties": False,
        },
    )

    ensure(
        "additionalProperties" not in loose.parameters,
        "普通 schema 默认允许模型多吐字段",
    )
    ensure(
        loose.parameters["required"] == ["order_id"],
        "普通 schema 里可选字段可以不填",
    )
    ensure(
        strict.parameters.get("additionalProperties") is False,
        "strict schema 必须显式禁止额外字段",
    )
    ensure(
        set(strict.parameters["required"]) == set(strict.parameters["properties"]),
        "strict schema 要求所有字段都必填，否则字段会被模型省略",
    )
    ensure(
        strict.parameters["properties"]["note"].get("nullable") is True,
        "strict schema 下的可选字段要写成 nullable，而不是从 required 里省掉",
    )
    print("  → strict 是协议层的硬约束，普通 schema 只是给模型的建议")


# ---------------------------------------------------------------- 9. 完整往返


def check_full_roundtrip() -> None:
    """声明 → 调用意图 → 结果回填 → 二次请求带完整历史。"""
    section("完整往返：四步走完一轮工具调用")

    tools = [order_tool()]
    client = scripted_client(
        responses=[
            ChatResponse(
                content="",
                tool_calls=[
                    make_tool_call("call_1", "查订单", '{"order_id": "A-1001"}')
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="订单 A-1001 已发货", finish_reason="stop"),
        ]
    )

    history: list[ChatMessage] = [
        ChatMessage(role="user", content="订单 A-1001 到哪了")
    ]

    # 第一步：带工具声明发起请求 → 拿到调用意图
    first = client.complete(history, tools=tools)
    ensure(
        first.finish_reason == "tool_calls" and needs_tool_execution(first),
        f"第一次请求应当拿到调用意图，实际 finish_reason={first.finish_reason!r}",
    )
    ensure(
        len(client.seen_calls[0]["tools"]) == 1,
        "第一次请求必须带工具声明",
    )

    # 第二步：把调用意图记进历史
    history.append(
        ChatMessage(role="assistant", content=first.content, tool_calls=first.tool_calls)
    )

    # 第三步：执行工具，把结果回填成 tool 消息
    for call in first.tool_calls:
        parsed = parse_arguments(call["function"]["arguments"])
        ensure(
            parsed.ok,
            f"回填前必须先解析 arguments，实际失败：{parsed.error}（原文 {parsed.raw!r}）",
        )
        history.append(
            ChatMessage(
                role="tool",
                content=run_tool(call["function"]["name"], parsed.data),
                tool_call_id=call["id"],
                name=call["function"]["name"],
            )
        )

    # 第四步：带完整历史二次请求
    second = client.complete(history, tools=tools)
    ensure(
        second.content == "订单 A-1001 已发货",
        f"二次请求应当拿到最终回答，实际 {second.content!r}",
    )
    ensure(
        len(client.seen_calls) == 2,
        f"一轮工具调用应当是两次模型请求，实际 {len(client.seen_calls)} 次",
    )

    sent = client.seen_calls[1]["messages"]
    ensure(
        len(sent) == 3,
        f"二次请求应当带 3 条消息（user / assistant / tool），实际 {len(sent)} 条",
    )
    ensure(
        [m.role for m in sent] == ["user", "assistant", "tool"],
        f"二次请求的消息角色顺序不对，实际 {[m.role for m in sent]}",
    )
    ensure(
        sent[1].tool_calls and sent[1].tool_calls[0]["id"] == "call_1",
        "二次请求里必须带上原来的调用意图（含 id）",
    )
    ensure(
        sent[2].tool_call_id == "call_1",
        f"回填的 tool 消息必须对上 tool_call_id，实际 {sent[2].tool_call_id!r}",
    )
    ensure(
        sent[2].content == "订单 A-1001 已发货",
        f"回填的工具结果内容不对，实际 {sent[2].content!r}",
    )
    ensure(
        len(client.seen_calls[1]["tools"]) == 1,
        "二次请求仍然要带上工具声明——schema 每轮都要重发",
    )
    print("  → 四步走完：声明 → 调用意图 → 结果回填 → 带历史二次请求")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：真调一次带工具的请求，只校验协议形状。"""
    banner("第 2 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client(temperature=0.0)
    response = client.complete(
        [ChatMessage(role="user", content="订单 A-1001 到哪了？请调用查订单工具。")],
        tools=[order_tool()],
    )

    print(f"  content: {response.content!r}")
    print(f"  finish_reason: {response.finish_reason}")
    print(f"  tool_calls: {len(response.tool_calls)}")

    ensure(
        response.finish_reason in {"stop", "tool_calls", "length"},
        f"finish_reason 异常：{response.finish_reason}",
    )
    if response.tool_calls:
        for call in response.tool_calls:
            ensure(
                call.get("type") == "function",
                f"tool_call 的 type 应当是 function，实际 {call.get('type')!r}",
            )
            ensure(
                isinstance(call.get("id"), str) and bool(call["id"]),
                "tool_call 必须带非空 id，否则结果没法回填",
            )
            ensure(
                isinstance(call["function"]["arguments"], str),
                "arguments 必须是 JSON 字符串",
            )
            parsed = parse_arguments(call["function"]["arguments"])
            ensure(
                parsed.ok,
                f"真实模型返回了非法 arguments：{parsed.raw!r} → {parsed.error}",
            )
    else:
        ensure(bool(response.content), "没有工具调用时必须有文本回答")


def run() -> None:
    """跑第 2 章实验。"""
    banner("第 2 章 · 工具调用协议")
    check_tools_payload()
    check_tool_choice_modes()
    check_parallel_calls()
    check_arguments_parsing()
    check_finish_reason_signal()
    check_unknown_tool()
    check_tool_count_cost()
    check_strict_schema()
    check_full_roundtrip()
    rule("工具调用是三段协议：声明 → 调用意图 → 结果回填；tool_call_id 是唯一的对齐依据")
    print("\n✓ 第 2 章通过")
