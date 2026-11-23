"""第 1 章实验：模型调用基础与消息协议。

这个模块解决什么问题：第 1 章讲的是"模型调用看着是一个函数，其实是八个旋钮
加一份消息协议"。这些结论必须能跑出来看，所以本章把五件事变成可执行的断言：

1. **消息协议**——四种 role 的形状、序列化结果、system 的位置约束、
   tool 消息必须带 tool_call_id；
2. **多轮对话的形状**——历史在协议上就是越来越长的数组，每轮都要重发；
3. **换模型零改动**——脚本化客户端和真实客户端都满足同一套协议，
   上层代码一行不用改；
4. **参数透传**——temperature / tools / response_format 确实到了模型手里；
5. **finish_reason 与错误分类**——哪些结果不能当正常结果用，
   哪些错误能重试。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

from typing import Any

from ..labkit import (
    ChatMessage,
    ChatResponse,
    StreamChunk,
    banner,
    ensure,
    rule,
    satisfies_client_protocol,
    scripted_client,
    section,
    tool_spec,
)


# ---------------------------------------------------------------- 1. 消息协议


def check_message_protocol() -> None:
    """四种 role 的形状、序列化，以及两条硬约束。"""
    section("消息协议：四种 role")

    system = ChatMessage(role="system", content="你是订单助手")
    user = ChatMessage(role="user", content="订单 A-1001 到哪了")
    assistant = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "查订单", "arguments": '{"order_id": "A-1001"}'},
            }
        ],
    )
    tool = ChatMessage(role="tool", content="已发货", tool_call_id="call_1")

    print(f"  system    → {system.to_api()}")
    print(f"  assistant → {assistant.to_api()}")
    print(f"  tool      → {tool.to_api()}")

    # 约束一：assistant 要调工具时，content 是 null 而不是空字符串
    ensure(
        assistant.content is None,
        f"带 tool_calls 的 assistant 消息 content 应为 None，实际 {assistant.content!r}",
    )
    ensure(
        assistant.to_api()["tool_calls"][0]["function"]["name"] == "查订单",
        "序列化后 tool_calls 结构应当保持原样",
    )

    # 约束二：tool 消息必须带 tool_call_id
    ensure(
        tool.tool_call_id == "call_1",
        f"tool 消息必须带 tool_call_id，实际 {tool.tool_call_id!r}",
    )
    ensure(
        "tool_call_id" in tool.to_api(),
        "序列化后 tool_call_id 不能丢——丢了模型就对不上是哪次调用的结果",
    )

    # 约束三：system 必须在最前
    messages = [system, user, assistant, tool]
    ensure(
        messages[0].role == "system",
        f"system 消息必须在最前，实际第一位是 {messages[0].role}",
    )
    ensure(
        [m.role for m in messages] == ["system", "user", "assistant", "tool"],
        f"消息顺序不对：{[m.role for m in messages]}",
    )
    print("  三条硬约束：assistant.content=None / tool 带 tool_call_id / system 在最前")


# ---------------------------------------------------------------- 2. 多轮形状


def check_multi_turn_shape() -> None:
    """多轮对话在协议上就是越来越长的数组，每轮都要重发。"""
    section("多轮对话：历史是越来越长的数组")

    client = scripted_client(
        responses=[
            ChatResponse(content="第一轮回答", usage={"prompt_tokens": 100, "completion_tokens": 10}),
            ChatResponse(content="第二轮回答", usage={"prompt_tokens": 180, "completion_tokens": 8}),
            ChatResponse(content="第三轮回答", usage={"prompt_tokens": 260, "completion_tokens": 6}),
        ]
    )

    history: list[ChatMessage] = []
    for turn in ("问题一", "问题二", "问题三"):
        history.append(ChatMessage(role="user", content=turn))
        response = client.complete(history)
        history.append(ChatMessage(role="assistant", content=response.content))

    lengths = [len(call["messages"]) for call in client.seen_calls]
    print(f"  每轮发送的消息条数：{lengths}")

    ensure(lengths == [1, 3, 5], f"每轮应带上全部历史，长度应递增，实际 {lengths}")
    ensure(
        all(call["messages"][-1].role == "user" for call in client.seen_calls),
        "每轮最后一条应当是用户消息",
    )

    # 输入 token 随历史增长——这是成本的主要来源
    usage = [call["messages"] for call in client.seen_calls]
    ensure(
        len(usage[0]) < len(usage[2]),
        "第三轮携带的历史应当比第一轮多",
    )
    print("  → 输入 token 随历史增长：优化成本的第一杠杆是别塞没用的东西")


# ---------------------------------------------------------------- 3. 换模型零改动


def check_client_swap() -> None:
    """脚本化客户端和真实客户端满足同一套协议，上层零改动。"""
    section("换模型：上层代码一行不用改")

    fake = scripted_client(responses=[ChatResponse(content="来自假模型")])
    ensure(
        satisfies_client_protocol(fake),
        "脚本化客户端应当满足客户端协议",
    )

    # 真实客户端只做构造检查，不发请求（不联网）
    from ..labkit import OpenAICompatibleClient

    real = OpenAICompatibleClient(model="some-model", base_url="http://localhost:9", api_key="x")
    ensure(
        satisfies_client_protocol(real),
        "真实客户端应当满足客户端协议",
    )
    ensure(
        type(fake) is not type(real),
        "两个实现应当是不同的类型，否则说明没有真正解耦",
    )

    # 同一段上层代码，两个实现都能跑
    def ask(client: Any, question: str) -> str:
        """一段不关心底层是谁的上层代码。"""
        response = client.complete([ChatMessage(role="user", content=question)])
        return response.content

    ensure(ask(fake, "你好") == "来自假模型", "上层代码应当能直接用假客户端")
    print("  上层代码只认协议：换实现不用改调用方")


# ---------------------------------------------------------------- 4. 参数透传


def check_parameters() -> None:
    """temperature / tools / response_format 确实到了模型手里。"""
    section("参数透传：旋钮真的转了吗")

    client = scripted_client(responses=[ChatResponse(content="ok")])
    spec = tool_spec("查订单", "按订单号查状态", {"type": "object", "properties": {}})

    client.complete(
        [ChatMessage(role="user", content="查 A-1001")],
        tools=[spec],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    call = client.seen_calls[0]
    ensure(call["temperature"] == 0.0, f"temperature 应当透传，实际 {call['temperature']}")
    ensure(len(call["tools"]) == 1, f"tools 应当透传，实际 {len(call['tools'])}")
    ensure(
        call["tools"][0].to_api()["function"]["name"] == "查订单",
        "工具声明的结构应当保持正确",
    )
    ensure(
        call["response_format"] == {"type": "json_object"},
        f"response_format 应当透传，实际 {call['response_format']}",
    )
    print(f"  temperature={call['temperature']} / tools={len(call['tools'])} / response_format 到位")


# ---------------------------------------------------------------- 5. finish_reason


def check_finish_reason() -> None:
    """哪些 finish_reason 不能当正常结果用。"""
    section("finish_reason：最容易被丢掉却最该保留的字段")

    cases = [
        ("stop", "正常说完", True),
        ("length", "撞到输出上限", False),
        ("tool_calls", "要调工具", True),
        ("content_filter", "被内容过滤", False),
    ]

    for reason, meaning, usable in cases:
        print(f"  {reason:<16} {meaning:<12} → {'可直接用' if usable else '不能当正常结果'}")

    ensure(cases[0][2], "stop 的结果可以直接用")
    ensure(not cases[1][2], "length 的结果不能当正常结果用")
    ensure(not cases[3][2], "content_filter 的结果不能当正常结果用")

    # length 意味着模型没说完整——这是最难查的故障之一
    truncated = ChatResponse(content="被截断的内容", finish_reason="length")
    ensure(
        truncated.finish_reason != "stop",
        "被截断的响应 finish_reason 必须是 length，不能伪装成 stop",
    )
    print("  → finish_reason == 'length' 时先怀疑输入，再怀疑模型")


# ---------------------------------------------------------------- 6. usage


def check_usage() -> None:
    """usage 告诉你花了多少；输入通常远多于输出。"""
    section("usage：输入 token 通常远多于输出")

    response = ChatResponse(
        content="回答",
        usage={"prompt_tokens": 1240, "completion_tokens": 86, "total_tokens": 1326},
    )
    ensure(response.usage["prompt_tokens"] == 1240, "prompt_tokens 应当被读出来")
    ensure(
        response.usage["total_tokens"]
        == response.usage["prompt_tokens"] + response.usage["completion_tokens"],
        "total_tokens 应当等于输入加输出",
    )
    ratio = response.usage["prompt_tokens"] / max(response.usage["completion_tokens"], 1)
    print(f"  输入/输出比 = {ratio:.1f} 倍")
    ensure(ratio > 3, "典型场景下输入应当远多于输出")
    print("  → 成本优化的第一杠杆是上下文，不是让模型少说话")


# ---------------------------------------------------------------- 7. 错误分类


def check_error_classification() -> None:
    """哪些错误能重试，哪些重试一万次也是错的。"""
    section("错误分类：重试还是改请求")

    def classify(status: int) -> str:
        """把 HTTP 状态码映射成处理策略。"""
        if status in (401, 403):
            return "fatal"          # 认证失败，重试无意义
        if status == 429:
            return "backoff"        # 限流，退避后重试
        if status >= 500:
            return "retry"          # 服务端故障
        return "fix_request"        # 400：请求本身有问题

    expectations = {
        401: "fatal",
        403: "fatal",
        429: "backoff",
        500: "retry",
        503: "retry",
        400: "fix_request",
        422: "fix_request",
    }
    for status, expected in expectations.items():
        actual = classify(status)
        print(f"  {status} → {actual}")
        ensure(actual == expected, f"{status} 应当判为 {expected}，实际 {actual}")

    ensure(classify(400) != classify(429), "400 和 429 的处理方式必须不同")
    print("  → 把 400 当 429 重试 = 在制造噪声，真原因一次都没被修")


def run_live() -> None:
    """真实模型路径：真调一次模型，校验协议形状。"""
    banner("第 1 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client()
    response = client.complete(
        [
            ChatMessage(role="system", content="你是订单助手，回答不超过两句话。"),
            ChatMessage(role="user", content="用一句话说明什么是 API。"),
        ],
        temperature=0.0,
    )

    print(f"  content: {response.content!r}")
    print(f"  finish_reason: {response.finish_reason}")
    print(f"  usage: {response.usage}")

    ensure(bool(response.content), "真实模型应当返回内容")
    ensure(response.finish_reason in {"stop", "length"}, f"finish_reason 异常：{response.finish_reason}")
    ensure(response.usage.get("total_tokens", 0) > 0, "usage 应当有 token 统计")
    ensure(bool(response.model), "响应应当带上模型名")


def run() -> None:
    """跑第 1 章实验。"""
    banner("第 1 章 · 模型调用基础与消息协议")
    check_message_protocol()
    check_multi_turn_shape()
    check_client_swap()
    check_parameters()
    check_finish_reason()
    check_usage()
    check_error_classification()
    rule("messages 是骨架，finish_reason 是最该保留的字段，错误分类决定你是在修问题还是在制造噪声")
    print("\n✓ 第 1 章通过")
