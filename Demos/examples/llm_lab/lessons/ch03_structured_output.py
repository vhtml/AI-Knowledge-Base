"""第 3 章实验：结构化输出。

这个模块解决什么问题：第 3 章讲的是"怎么让模型稳定吐出可用的 JSON"。
这件事在工程上不是一次调用，而是**三段管道**——

- **解析层**：从混杂文本里把 JSON 抠出来（模型爱包 ```` ```json ```` 围栏、
  爱加一句"好的，结果如下："）；
- **校验层**：对着 JSON Schema 挑毛病，并且能指出**是哪个字段**错了；
- **修复层**：把校验错误回给模型让它自己改，比我们替它猜靠谱。

只靠"提示词里写一句只输出 JSON"是不够的：它靠模型听话，而协议级的
``response_format`` 靠解码器保证语法合法。所以本章把十件事变成可执行的断言：

1. **三条路**——提示词请求 / ``response_format`` 的 json_object / json_schema，
   请求体的结构必须和厂商协议对得上；
2. **提示词请求的翻车形态**——围栏、前言、截断；
3. **解析层**——三种形态都能抠出 JSON，抠不出来要抛**可读**异常；
4. **截断识别**——半截 JSON 的报错要让人一眼看出"像是被截断了"；
5. **校验层**——六类错误都要能指出具体字段；
6. **解析成功 ≠ 校验通过**——不通过时返回错误列表，不返回数据；
7. **修复重试**——错误回给模型，第二次改对，且确实发生了两次模型调用；
8. **重试上限**——连续 N 次不合格就停，调用次数必须等于上限；
9. **工具 vs 结构化输出**——工具路径产生 tool_calls，结构化输出路径产生纯 content；
10. **schema 复杂度预算**——太复杂就该拆成多次调用。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..labkit import (
    ChatMessage,
    ChatResponse,
    banner,
    ensure,
    estimate_cost,
    extract_json,
    rule,
    scripted_client,
    section,
    tool_spec,
    validate_against_schema,
)

# ---------------------------------------------------------------- 协议常量


JSON_OBJECT_FORMAT: dict[str, Any] = {"type": "json_object"}
"""``response_format`` 的 json_object 形态——只保证"是个 JSON 对象"。"""

MAX_SCHEMA_FIELDS: int = 12
"""一次抽取里允许出现的字段数上限（教学用的经验值）。"""

MAX_SCHEMA_DEPTH: int = 2
"""一次抽取里允许出现的对象嵌套层数上限（教学用的经验值）。"""


def json_schema_format(
    name: str,
    schema: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    """构造 ``response_format`` 的 json_schema 形态。

    结构必须和厂商协议一字不差：
    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}``。
    ``strict=True`` 是关键——它表示字段集合是封闭的，多出来的字段会被拒。
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "description": "订单号，形如 A-1001"},
        "status": {"type": "string", "enum": ["待付款", "在途", "已发货"]},
        "amount": {"type": "number", "minimum": 0, "maximum": 100000},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qty": {"type": "integer", "minimum": 1},
                },
                "required": ["name", "qty"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["order_id", "status", "amount", "items"],
    "additionalProperties": False,
}
"""本章反复用到的订单抽取 schema——覆盖 enum / minimum / maximum / 数组元素 / 封闭字段。"""


# ---------------------------------------------------------------- 异常与结果类型


class StructuredOutputError(ValueError):
    """结构化输出在重试上限内没能通过。

    继承 ``ValueError`` 是为了让上层用 ``except ValueError`` 也能兜住，
    但异常信息必须自带定位线索：最后几次的错误原文 + 尝试次数。
    """


@dataclass(frozen=True)
class ParseOutcome:
    """解析层的结果。

    成功才有 ``data``；失败只有 ``error``——**不返回半成品**。
    """

    ok: bool
    data: dict[str, Any] | list[Any] | None
    error: str = ""
    truncated: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    """一次抽取（含修复重试）的结果。"""

    ok: bool
    data: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------- 解析层


def looks_truncated(text: str) -> bool:
    """判断一段输出是不是"说到一半被掐断了"。

    判据是**结构字符有没有闭合**：括号没配平、字符串没闭合，
    都是截断的典型特征。逐字符扫描而不是数个数，
    因为 JSON 字符串内部可能出现括号（``{"note": "a)b"}``）。
    """
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return depth > 0 or in_string


def parse_model_json(text: str, finish_reason: str = "stop") -> ParseOutcome:
    """解析层：从混杂文本里抠 JSON，抠不出来就报清楚。

    抠不出来时**抛异常而不是返回空对象**——空的 dict 会被下游当成
    "用户什么都没填"，真正的错误被吞掉，报错离根因十万八千里。
    """
    try:
        value = extract_json(text)
    except ValueError as exc:
        truncated = looks_truncated(text)
        if truncated:
            hint = (
                f"输出看起来被截断了（finish_reason={finish_reason}）："
                "结构字符没有闭合，先查输出上限，再查 schema 是不是太复杂"
            )
        else:
            hint = f"输出里找不到可解析的 JSON（finish_reason={finish_reason}）"
        raise StructuredOutputError(f"{hint}；原始输出：{text[:200]!r}") from exc
    return ParseOutcome(ok=True, data=value)


# ---------------------------------------------------------------- 校验层与修复层


def repair_instruction(errors: Sequence[str]) -> str:
    """把校验错误回给模型。

    只说"哪里错了"，不说"你应该怎么改"——后者是我们替模型猜，
    猜错了模型会照着错答案改。只有确定的事实（哪个字段、缺了还是类型不对）
    才能回灌。
    """
    lines = "\n".join(f"- {item}" for item in errors)
    return (
        "上一次输出没有通过校验，请只输出修正后的 JSON，不要加任何解释。\n"
        f"需要修正的问题：\n{lines}"
    )


@dataclass
class StructuredExtractor:
    """带解析、校验和修复重试的抽取器。

    三层各司其职：``parse_model_json`` 负责抠，``validate_against_schema``
    负责挑毛病，这里负责在两者之间搭一条"改完再试"的回路。
    """

    client: Any
    schema: dict[str, Any]
    name: str = "抽取结果"
    max_attempts: int = 3

    def _system_prompt(self) -> str:
        """系统提示只说清输出形态，不重复 schema——schema 已经进了 response_format。"""
        return (
            f"你是{self.name}抽取器。只输出一个 JSON 对象，"
            "不要加解释、不要加 markdown 围栏。"
        )

    def extract(self, source: str) -> ExtractionResult:
        """抽取一次，失败就把错误回给模型重试。

        重试次数有上限；用尽仍不合格就抛 :class:`StructuredOutputError`，
        绝不无限重试——每一次重试都是真金白银。
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self._system_prompt()),
            ChatMessage(role="user", content=source),
        ]
        errors: list[str] = []
        attempts = 0
        prompt_tokens = 0
        completion_tokens = 0

        for _ in range(self.max_attempts):
            attempts += 1
            response = self.client.complete(messages)
            usage = response.usage or {}
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)

            # 解析层：抠不出来就把原始输出留给模型看，同时把失败原因记下来
            try:
                parsed = parse_model_json(response.content, response.finish_reason)
            except StructuredOutputError as exc:
                errors = [str(exc)]
                messages.append(ChatMessage(role="assistant", content=response.content))
                messages.append(ChatMessage(role="user", content=repair_instruction(errors)))
                continue

            # 校验层：能解析 ≠ 符合 schema
            schema_errors = validate_against_schema(parsed.data, self.schema)
            if not schema_errors:
                return ExtractionResult(
                    ok=True,
                    data=parsed.data,
                    errors=[],
                    attempts=attempts,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            errors = schema_errors
            messages.append(ChatMessage(role="assistant", content=response.content))
            messages.append(ChatMessage(role="user", content=repair_instruction(errors)))

        raise StructuredOutputError(
            f"{self.name}在 {self.max_attempts} 次尝试内都没通过校验；"
            f"最后一次的错误：{'；'.join(errors)}"
        )


# ---------------------------------------------------------------- schema 复杂度


def schema_complexity(schema: dict[str, Any]) -> tuple[int, int]:
    """数出一份 schema 的（字段总数, 最大嵌套层数）。

    层数只按 ``properties`` 递归——数组的 ``items`` 不额外占一层，
    因为数组本身不是一层新的对象结构。
    """

    def walk(node: dict[str, Any], depth: int) -> tuple[int, int]:
        total = 0
        deepest = depth
        properties = node.get("properties")
        if isinstance(properties, dict):
            for sub in properties.values():
                if not isinstance(sub, dict):
                    continue
                total += 1
                sub_total, sub_depth = walk(sub, depth + 1)
                total += sub_total
                deepest = max(deepest, sub_depth)
        items = node.get("items")
        if isinstance(items, dict):
            item_total, item_depth = walk(items, depth)
            total += item_total
            deepest = max(deepest, item_depth)
        return total, deepest

    return walk(schema, 0)


def deep_order_schema() -> dict[str, Any]:
    """一份"太复杂"的 schema——嵌套三层以上，用来演示什么时候该拆调用。"""
    return {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "price": {"type": "number"},
                                },
                                "required": ["key", "price"],
                            },
                        },
                    },
                    "required": ["name", "options"],
                },
            },
        },
        "required": ["order_id", "items"],
    }


# ---------------------------------------------------------------- 1. 三条路


def check_three_routes() -> None:
    """提示词请求 / json_object / json_schema：请求体的三种形状。"""
    section("三条路：提示词请求 / json_object / json_schema")

    messages = [ChatMessage(role="user", content="订单 A-1001 已发货，金额 128.5 元")]

    # 路一：提示词请求——请求体里没有 response_format，全靠 system 说话
    prompt_only = scripted_client(responses=[ChatResponse(content='{"order_id": "A-1001"}')])
    prompt_only.complete(
        [ChatMessage(role="system", content="只输出 JSON"), *messages],
    )
    ensure(
        prompt_only.seen_calls[0]["response_format"] is None,
        "提示词请求不应当带 response_format，实际 "
        f"{prompt_only.seen_calls[0]['response_format']!r}",
    )

    # 路二：json_object——请求体里只有一个 type 字段，没有 schema
    object_client = scripted_client(responses=[ChatResponse(content='{"order_id": "A-1001"}')])
    object_client.complete(messages, response_format=JSON_OBJECT_FORMAT)
    object_format = object_client.seen_calls[0]["response_format"]
    ensure(
        object_format == {"type": "json_object"},
        f"json_object 形态应当只有 type 一个键，实际 {sorted(object_format)}",
    )
    ensure(
        "json_schema" not in object_format,
        "json_object 形态里不该出现 json_schema——它不约束字段",
    )

    # 路三：json_schema——schema 跟着请求走，字段和类型有协议级约束
    schema_client = scripted_client(responses=[ChatResponse(content='{"order_id": "A-1001"}')])
    schema_client.complete(messages, response_format=json_schema_format("order", ORDER_SCHEMA))
    schema_format = schema_client.seen_calls[0]["response_format"]

    ensure(
        schema_format["type"] == "json_schema",
        f"json_schema 形态的 type 应当是 json_schema，实际 {schema_format['type']!r}",
    )
    ensure(
        set(schema_format["json_schema"]) == {"name", "schema", "strict"},
        "json_schema 下应当恰好是 name/schema/strict 三个键，实际 "
        f"{sorted(schema_format['json_schema'])}",
    )
    ensure(
        schema_format["json_schema"]["name"] == "order",
        f"json_schema.name 应当是 order，实际 {schema_format['json_schema']['name']!r}",
    )
    ensure(
        schema_format["json_schema"]["strict"] is True,
        "strict 必须是 True——否则多出来的字段照样能混进来",
    )
    ensure(
        schema_format["json_schema"]["schema"]["required"] == ORDER_SCHEMA["required"],
        "json_schema 里的 schema 必须和校验用的那份是同一份",
    )

    # 序列化一次，确认请求体本身是合法 JSON（线上就是这么发出去的）
    wire = json.loads(json.dumps(schema_format, ensure_ascii=False))
    ensure(
        wire["json_schema"]["schema"]["properties"]["status"]["enum"]
        == ["待付款", "在途", "已发货"],
        "序列化后 schema 里的 enum 不该丢",
    )
    print("  路一：无 response_format｜路二：type=json_object｜路三：type=json_schema")
    print("  → 只有 json_schema 把字段名和类型写进了协议里")


# ---------------------------------------------------------------- 2. 提示词翻车


def check_prompt_failure_modes() -> None:
    """提示词请求的三种翻车形态：围栏、前言、截断。"""
    section("提示词请求：它不是语法保证，只是'希望模型听话'")

    fenced = "```json\n{\"order_id\": \"A-1001\"}\n```"
    preamble = "好的，结果如下：\n{\"order_id\": \"A-1001\"}"
    truncated = '{"order_id": "A-1001", "status": "在途", "amount": 12'

    print(f"  围栏形态：{fenced.splitlines()[0]} …")
    print(f"  前言形态：{preamble.splitlines()[0]} …")
    print(f"  截断形态：{truncated!r}")

    ensure(
        fenced.startswith("```"),
        "围栏形态的样本应当以 ``` 开头",
    )
    ensure(
        preamble.splitlines()[0].endswith("："),
        "前言形态的样本应当先有一句废话",
    )
    ensure(
        looks_truncated(truncated),
        "截断形态的样本应当被识别成结构未闭合",
    )
    ensure(
        not looks_truncated(fenced) and not looks_truncated(preamble),
        "完整的两种形态不该被误判成截断",
    )

    # 截断的真正信号在 finish_reason 上——第 1 章说过它最容易被丢掉
    response = ChatResponse(content=truncated, finish_reason="length")
    ensure(
        response.finish_reason == "length",
        "输出被截断时 finish_reason 应当是 length",
    )
    print("  → 截断先看 finish_reason，再看 JSON 自己的结构")


# ---------------------------------------------------------------- 3. 解析层


def check_parse_layer() -> None:
    """从三种形态里都能抠出 JSON；抠不出来要抛可读异常。"""
    section("解析层：抠得出来才算数，抠不出来必须报清楚")

    cases = [
        ("裸 JSON", '{"order_id": "A-1001"}'),
        ("围栏", '```json\n{"order_id": "A-1001"}\n```'),
        ("前言废话", '好的，结果如下：\n{"order_id": "A-1001"}'),
    ]
    for label, text in cases:
        parsed = parse_model_json(text)
        ensure(parsed.ok, f"{label} 应当解析成功，实际失败：{parsed.error}")
        ensure(
            parsed.data == {"order_id": "A-1001"},
            f"{label} 解析出来的数据不对，实际 {parsed.data!r}",
        )
        print(f"  {label:<8} → {parsed.data!r}")

    # 完全无法解析：异常必须可读，且带着原始输出片段
    garbage = "对不起，我无法完成这个请求。"
    raised = False
    try:
        extract_json(garbage)
    except ValueError as exc:
        raised = True
        message = str(exc)
    ensure(raised, f"无法解析时应当抛异常，实际正常返回了：{garbage!r}")
    ensure(
        "无法完成这个请求" in message,
        f"异常信息里应当带上原始输出片段，实际 {message!r}",
    )
    ensure(
        "JSON" in message,
        f"异常信息应当说清是 JSON 解析失败，实际 {message!r}",
    )
    print(f"  不可解析 → ValueError：{message[:60]}…")

    # 本章自己的解析层同样要抛，且异常类型要能被上层统一捕获
    try:
        parse_model_json(garbage)
    except StructuredOutputError as exc:
        message = str(exc)
    else:
        raise AssertionError("完全无法解析时应当抛 StructuredOutputError")
    ensure(
        "无法完成这个请求" in message,
        f"StructuredOutputError 里应当带上原始输出片段，实际 {message!r}",
    )
    ensure(
        isinstance(StructuredOutputError("x"), ValueError),
        "StructuredOutputError 应当能被 except ValueError 兜住",
    )
    print("  → 解析失败要抛异常，不能静默返回空 dict")


# ---------------------------------------------------------------- 4. 截断识别


def check_truncation_detection() -> None:
    """半截 JSON 的报错要让人看出'像是被截断了'。"""
    section("截断识别：报错信息里要能看出'没说完'")

    truncated = '{"order_id": "A-1001", "items": [{"name": "键盘", "qty": 2'
    try:
        parse_model_json(truncated, finish_reason="length")
    except StructuredOutputError as exc:
        message = str(exc)
    else:
        raise AssertionError("半截 JSON 应当解析失败")

    ensure(
        "截断" in message,
        f"异常信息应当点明'像是被截断了'，实际 {message!r}",
    )
    ensure(
        "length" in message,
        f"异常信息应当带上 finish_reason，实际 {message!r}",
    )
    ensure(
        truncated[:40] in message,
        f"异常信息应当带上原始输出片段，实际 {message!r}",
    )
    print(f"  → {message[:90]}…")

    # 完整的 JSON 不该被当成截断
    complete = '{"order_id": "A-1001", "items": [{"name": "键盘", "qty": 2}]}'
    ensure(
        not looks_truncated(complete),
        "完整的 JSON 不该被误判成截断",
    )
    # 字符串里带括号不算截断——所以要逐字符扫，不能数括号个数
    tricky = '{"note": "a)b", "order_id": "A-1001"}'
    ensure(
        not looks_truncated(tricky),
        "字符串内部的括号不该被当成未闭合",
    )


# ---------------------------------------------------------------- 5. 校验层


def check_validation_layer() -> None:
    """六类错误都要能指出具体字段。"""
    section("校验层：解析成功 ≠ 符合 schema")

    cases = [
        (
            "缺必填字段",
            {"status": "在途", "amount": 12.5, "items": []},
            "缺少必填字段",
            "$.order_id",
        ),
        (
            "类型错",
            {
                "order_id": "A-1001",
                "status": "在途",
                "amount": "12.5",
                "items": [],
            },
            "期望 number",
            "$.amount",
        ),
        (
            "额外字段",
            {
                "order_id": "A-1001",
                "status": "在途",
                "amount": 12.5,
                "items": [],
                "coupon": "X",
            },
            "不允许的额外字段",
            "$.coupon",
        ),
        (
            "枚举非法",
            {
                "order_id": "A-1001",
                "status": "已签收",
                "amount": 12.5,
                "items": [],
            },
            "不在允许的取值",
            "$.status",
        ),
        (
            "数值越界",
            {
                "order_id": "A-1001",
                "status": "在途",
                "amount": 999999,
                "items": [],
            },
            "大于最大值",
            "$.amount",
        ),
        (
            "数组元素类型错",
            {
                "order_id": "A-1001",
                "status": "在途",
                "amount": 12.5,
                "items": [{"name": "键盘", "qty": "两"}],
            },
            "期望 integer",
            "$.items[0].qty",
        ),
    ]

    for label, payload, keyword, path in cases:
        errors = validate_against_schema(payload, ORDER_SCHEMA)
        ensure(
            bool(errors),
            f"{label} 应当被检出错误，实际校验通过：{payload!r}",
        )
        joined = "；".join(errors)
        ensure(
            keyword in joined,
            f"{label} 的报错应当包含「{keyword}」，实际 {joined!r}",
        )
        ensure(
            path in joined,
            f"{label} 的报错应当指出字段路径 {path}，实际 {joined!r}",
        )
        print(f"  {label:<12} → {errors[0]}")

    # 完全合法的一份必须零错误——不然校验器就是个坏掉的报警器
    good = {
        "order_id": "A-1001",
        "status": "在途",
        "amount": 12.5,
        "items": [{"name": "键盘", "qty": 2}],
    }
    ensure(
        validate_against_schema(good, ORDER_SCHEMA) == [],
        "合法数据不该报错："
        f"{validate_against_schema(good, ORDER_SCHEMA)}",
    )
    print("  合法数据 → 零错误")


# ---------------------------------------------------------------- 6. 通过与不通过


def check_pass_fail_difference() -> None:
    """不通过时不返回数据，只返回错误列表。"""
    section("通过与不通过：返回的东西必须不一样")

    client = scripted_client(
        responses=[ChatResponse(content='{"order_id": "A-1001", "status": "在途"}')]
    )
    extractor = StructuredExtractor(client=client, schema=ORDER_SCHEMA, max_attempts=1)

    # 第一次输出不合法：字段不全
    try:
        extractor.extract("订单 A-1001 在途")
    except StructuredOutputError as exc:
        message = str(exc)
    else:
        raise AssertionError("校验不通过且不重试时应当抛异常")

    ensure(
        "缺少必填字段" in message,
        f"异常里应当带上校验错误原文，实际 {message!r}",
    )
    ensure(
        "1 次尝试" in message,
        f"异常里应当说清试了几次，实际 {message!r}",
    )

    # 一次就对：返回数据，错误列表为空
    ok_client = scripted_client(
        responses=[
            ChatResponse(
                content=json.dumps(
                    {
                        "order_id": "A-1001",
                        "status": "在途",
                        "amount": 12.5,
                        "items": [],
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )
    ok_extractor = StructuredExtractor(client=ok_client, schema=ORDER_SCHEMA, max_attempts=1)
    result = ok_extractor.extract("订单 A-1001 在途，12.5 元")

    ensure(result.ok, f"合法输出应当成功，实际错误：{result.errors}")
    ensure(result.errors == [], f"成功时错误列表应当为空，实际 {result.errors}")
    ensure(result.data is not None, "成功时必须返回数据")
    ensure(
        result.data is not None and result.data["order_id"] == "A-1001",
        f"成功时返回的数据不对，实际 {result.data!r}",
    )
    ensure(result.attempts == 1, f"一次就对时尝试次数应当是 1，实际 {result.attempts}")
    print("  通过 → 返回 data + 空 errors；不通过 → 只有 errors，没有 data")


# ---------------------------------------------------------------- 7. 修复重试


def check_repair_retry() -> None:
    """把校验错误回给模型，第二次改对。"""
    section("修复重试：把错误回给模型，比自己猜靠谱")

    client = scripted_client(
        responses=[
            # 第一次：少了 amount 和 items
            ChatResponse(content='{"order_id": "A-1001", "status": "在途"}'),
            # 第二次：按错误提示改对了
            ChatResponse(
                content=json.dumps(
                    {
                        "order_id": "A-1001",
                        "status": "在途",
                        "amount": 128.5,
                        "items": [{"name": "键盘", "qty": 1}],
                    },
                    ensure_ascii=False,
                ),
                usage={"prompt_tokens": 120, "completion_tokens": 40},
            ),
        ]
    )
    extractor = StructuredExtractor(client=client, schema=ORDER_SCHEMA, max_attempts=3)
    result = extractor.extract("订单 A-1001 在途，128.5 元，键盘 1 个")

    ensure(result.ok, f"重试后应当成功，实际错误：{result.errors}")
    ensure(
        result.data is not None and result.data["amount"] == 128.5,
        f"重试后拿到的数据不对，实际 {result.data!r}",
    )
    ensure(
        result.attempts == 2,
        f"应当发生两次模型调用才成功，实际 {result.attempts} 次",
    )
    ensure(
        len(client.seen_calls) == 2,
        f"确实应当发生两次模型请求，实际 {len(client.seen_calls)} 次",
    )

    # 关键：第二次请求里必须真的带着校验错误回去，否则"重试"是假的
    second_messages = client.seen_calls[1]["messages"]
    repair_text = "\n".join(str(m.content) for m in second_messages)
    ensure(
        "缺少必填字段" in repair_text,
        f"第二次请求应当把校验错误回给模型，实际内容：{repair_text[-160:]!r}",
    )
    ensure(
        "$.amount" in repair_text,
        f"回灌的错误应当指出具体字段，实际内容：{repair_text[-160:]!r}",
    )
    ensure(
        any(m.role == "assistant" for m in second_messages),
        "第二次请求应当带上模型上一次的原话，才能对照着改",
    )
    print("  第 1 次缺字段 → 回灌错误 → 第 2 次改对（共 2 次模型调用）")


# ---------------------------------------------------------------- 8. 重试上限


def check_retry_limit() -> None:
    """连续 N 次不合格就停，不能无限重试。"""
    section("重试上限：每一次重试都是真金白银")

    bad = '{"order_id": "A-1001"}'
    client = scripted_client(responses=[ChatResponse(content=bad) for _ in range(5)])
    extractor = StructuredExtractor(client=client, schema=ORDER_SCHEMA, max_attempts=3)

    try:
        extractor.extract("订单 A-1001")
    except StructuredOutputError as exc:
        message = str(exc)
    else:
        raise AssertionError("用尽重试上限应当抛异常")

    ensure(
        len(client.seen_calls) == 3,
        f"应当正好调用 3 次模型，实际 {len(client.seen_calls)} 次",
    )
    ensure(
        "3 次尝试" in message,
        f"异常应当说清试了几次，实际 {message!r}",
    )
    ensure(
        "缺少必填字段" in message,
        f"异常应当带上最后一次的校验错误，实际 {message!r}",
    )
    print(f"  连续 3 次不合格 → 停止（实际模型调用 {len(client.seen_calls)} 次）")
    print("  → 没有上限的重试循环 = 一张无限额的发票")


# ---------------------------------------------------------------- 9. 工具 vs 结构化


def check_tool_vs_structured() -> None:
    """工具调用产生 tool_calls；结构化输出产生纯 content。"""
    section("工具调用 vs 结构化输出：给选项，还是让它填空")

    # 工具路径：模型交回调用意图，content 为空
    tool_client = scripted_client(
        responses=[
            ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "查订单",
                            "arguments": '{"order_id": "A-1001"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    tool_response = tool_client.complete(
        [ChatMessage(role="user", content="订单 A-1001 到哪了")],
        tools=[
            tool_spec(
                "查订单",
                "按订单号查询订单状态",
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            )
        ],
    )

    # 结构化输出路径：模型直接交回 JSON，content 是纯文本，没有 tool_calls
    structured_client = scripted_client(
        responses=[ChatResponse(content='{"order_id": "A-1001", "status": "在途"}')]
    )
    structured_response = structured_client.complete(
        [ChatMessage(role="user", content="订单 A-1001 到哪了")],
        response_format=json_schema_format("order", ORDER_SCHEMA),
    )

    ensure(
        len(tool_response.tool_calls) == 1,
        f"工具路径应当产生 tool_calls，实际 {tool_response.tool_calls}",
    )
    ensure(
        not tool_response.content,
        f"工具路径的 content 应当为空，实际 {tool_response.content!r}",
    )
    ensure(
        tool_response.finish_reason == "tool_calls",
        f"工具路径的 finish_reason 应当是 tool_calls，实际 {tool_response.finish_reason!r}",
    )

    ensure(
        structured_response.tool_calls == [],
        f"结构化输出路径不应当产生 tool_calls，实际 {structured_response.tool_calls}",
    )
    ensure(
        bool(structured_response.content),
        "结构化输出路径应当有 content",
    )
    parsed = parse_model_json(structured_response.content)
    ensure(
        parsed.ok and isinstance(parsed.data, dict),
        f"结构化输出的 content 应当直接是 JSON，实际 {structured_response.content!r}",
    )
    ensure(
        structured_client.seen_calls[0]["tools"] == [],
        "结构化输出路径不声明 tools——两条路走的是不同的通道",
    )
    print("  工具路径 → tool_calls + 空 content｜结构化路径 → 纯 content")


# ---------------------------------------------------------------- 10. 复杂度与成本


def check_schema_complexity() -> None:
    """schema 太复杂就该拆成多次调用；重试会让成本翻倍。"""
    section("schema 复杂度：一次生成对不了，就拆成两次")

    flat_fields, flat_depth = schema_complexity(ORDER_SCHEMA)
    deep_fields, deep_depth = schema_complexity(deep_order_schema())

    print(f"  扁平 schema：{flat_fields} 个字段 / {flat_depth} 层")
    print(f"  深层 schema：{deep_fields} 个字段 / {deep_depth} 层")

    ensure(
        flat_depth <= MAX_SCHEMA_DEPTH,
        f"扁平 schema 应当在小预算内，实际 {flat_depth} 层",
    )
    ensure(
        deep_depth > MAX_SCHEMA_DEPTH,
        f"深层 schema 应当超出层数预算（{MAX_SCHEMA_DEPTH}），实际 {deep_depth} 层",
    )

    # 拆调用：第一步只抽订单骨架，第二步再抽每个商品的选项
    step_one = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        "required": ["order_id", "items"],
    }
    step_two = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "price": {"type": "number"},
        },
        "required": ["key", "price"],
    }
    for label, schema in (("第一步", step_one), ("第二步", step_two)):
        fields, depth = schema_complexity(schema)
        ensure(
            fields <= MAX_SCHEMA_FIELDS,
            f"{label} 的字段数应当在小预算内（{MAX_SCHEMA_FIELDS}），实际 {fields}",
        )
        ensure(
            depth <= MAX_SCHEMA_DEPTH,
            f"{label} 的层数应当在小预算内（{MAX_SCHEMA_DEPTH}），实际 {depth}",
        )
        print(f"  {label}：{fields} 个字段 / {depth} 层 → 在预算内")

    # 成本：修复重试意味着多付一次钱
    client = scripted_client(
        responses=[
            ChatResponse(
                content='{"order_id": "A-1001"}',
                usage={"prompt_tokens": 200, "completion_tokens": 20},
            ),
            ChatResponse(
                content=json.dumps(
                    {
                        "order_id": "A-1001",
                        "status": "在途",
                        "amount": 1.0,
                        "items": [],
                    },
                    ensure_ascii=False,
                ),
                usage={"prompt_tokens": 260, "completion_tokens": 40},
            ),
        ]
    )
    extractor = StructuredExtractor(client=client, schema=ORDER_SCHEMA, max_attempts=3)
    result = extractor.extract("订单 A-1001")
    ensure(result.ok, f"应当重试成功，实际 {result.errors}")

    first_cost = estimate_cost("cheap", 200, 20)
    total_cost = estimate_cost("cheap", result.prompt_tokens, result.completion_tokens)
    ensure(
        total_cost > first_cost,
        f"带重试的总成本应当高于单次调用，实际 {total_cost} vs {first_cost}",
    )
    print(
        f"  成本：单次 ≈ ${first_cost:.6f} → 带一次重试 ≈ ${total_cost:.6f}"
        f"（{total_cost / first_cost:.1f} 倍）"
    )
    print("  → 所以第一遍就把 schema 写简单，是最便宜的优化")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：真调一次结构化输出，只校验形状。"""
    banner("第 3 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client(temperature=0.0)
    messages = [
        ChatMessage(role="system", content="你是订单抽取器，只输出 JSON。"),
        ChatMessage(role="user", content="订单 A-1001 已发货，金额 128.5 元，键盘 1 个。"),
    ]

    try:
        response = client.complete(
            messages,
            response_format=json_schema_format("order", ORDER_SCHEMA),
        )
    except Exception as exc:  # noqa: BLE001 - 端点不支持 json_schema 时退回 json_object
        print(f"  端点不支持 json_schema（{type(exc).__name__}: {exc}），退回 json_object")
        response = client.complete(messages, response_format=JSON_OBJECT_FORMAT)

    print(f"  content: {response.content!r}")
    print(f"  finish_reason: {response.finish_reason}")

    parsed = parse_model_json(response.content, response.finish_reason)
    ensure(parsed.ok, f"真实模型应当返回可解析的 JSON：{parsed.error}")
    errors = validate_against_schema(parsed.data, ORDER_SCHEMA)
    ensure(not errors, f"真实模型输出未通过 schema 校验：{errors}")
    print("  ✓ 真实模型输出可解析且符合 schema")


def run() -> None:
    """跑第 3 章实验。"""
    banner("第 3 章 · 结构化输出")
    check_three_routes()
    check_prompt_failure_modes()
    check_parse_layer()
    check_truncation_detection()
    check_validation_layer()
    check_pass_fail_difference()
    check_repair_retry()
    check_retry_limit()
    check_tool_vs_structured()
    check_schema_complexity()
    rule(
        "结构化输出是三段管道：解析层容错、校验层指出字段、"
        "修复层把错误回给模型；约束解码和提示词请求差一个量级"
    )
    print("\n✓ 第 3 章通过")
