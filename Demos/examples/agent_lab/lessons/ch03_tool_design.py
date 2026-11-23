"""第 3 章实验：工具设计——schema、错误面与粒度。

这个模块解决什么问题：第 2 章把 ReAct 循环跑通了，但**循环跑得通不等于模型调得对**。
模型看不到你的 Python 函数，它只看到 name / description / parameters 三样东西；
这一章就把"模型调错工具、传错参数"拆成能看见、能断言的几段：

1. **schema 从哪来**——docstring 的 ``Args:`` 段落怎么变成字段级 description；
   以及一个真坑：开了 PEP 563 的模块里注解是字符串，``int`` 参数会被声明成
   ``string``，schema 当场对模型说谎。
2. **三类参数失败**——缺参数、类型错、取值域非法，看模型能不能看见错误并自我纠正。
3. **错误面的三种设计**——抛异常 / 返回结构化错误 / 直接吞掉，各自的后果。
4. **工具粒度**——粗粒度单一工具 vs 细粒度多工具，同一脚本下"选错之后能不能恢复"。
5. **规模与命名**——工具越多 schema 越长；同义工具名是可检测的坑。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..labkit import (
    Message,
    ModelReply,
    Tool,
    ToolCall,
    _parse_arg_descriptions,
    approx_tokens,
    banner,
    ensure,
    rule,
    scripted_model,
    section,
    tool,
)
from ..runtime import Agent, AgentConfig

# ---------------------------------------------------------------- 工具定义


@tool
def 查订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    orders = {"A-1001": "已发货", "A-1002": "待付款"}
    if order_id not in orders:
        raise ValueError(f"订单号 {order_id!r} 不存在，可用的有 A-1001、A-1002")
    return orders[order_id]


@tool
def 查订单无字段说明(order_id: str) -> str:
    """按订单号查询订单状态。（对照组：故意不写 Args 段落）"""
    return "已发货"


@tool
def 查订单详情(order_id: str) -> str:
    """按订单号查询订单详情（含物流与退款信息）。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return "已发货（含物流）"


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return "已发货"


def typed_tool(func: Callable[..., Any]) -> Tool:
    """把 PEP 563 压成字符串的注解换回真实类型对象，再交给 ``tool()``。

    为什么值得单独写一遍：``labkit.tool()`` 现在已经用 ``get_type_hints``
    解析过注解，不会踩坑；但**你自己写 schema 生成器时一定会踩**——
    开了 ``from __future__ import annotations`` 的模块里注解是字符串，
    按类型对象查表会把 ``quantity: int`` 声明成 ``string``。

    这里保留手工版本，一是让本章的工具显式声明"我处理过这个坑"，
    二是给读者一份可以照抄的修复模板。
    """
    mapping: dict[str, Any] = {"str": str, "int": int, "float": float, "bool": bool}
    func.__annotations__ = {
        name: mapping.get(value, value) if isinstance(value, str) else value
        for name, value in func.__annotations__.items()
    }
    return tool(func)


@typed_tool
def 查库存(item_id: str, quantity: int) -> str:
    """查库存余量。

    Args:
        item_id: 库存条目编号，形如 S-200
        quantity: 需要扣减的数量，必须是正整数
    """
    if not isinstance(quantity, int):
        # 点名参数：模型只有看到"哪个参数错了"才可能改对
        raise TypeError(f"参数 quantity 需要整数，实际收到 {type(quantity).__name__}")
    if quantity <= 0:
        raise ValueError(f"参数 quantity 必须是正整数，实际收到 {quantity}")
    return f"{item_id} 现有库存 {quantity * 3} 件"


@typed_tool
def 查库存无参数名(item_id: str, quantity: int) -> str:
    """查库存余量（反面教材：报错文案不点名参数）。

    Args:
        item_id: 库存条目编号，形如 S-200
        quantity: 需要扣减的数量，必须是正整数
    """
    if not isinstance(quantity, int):
        # 少了参数名，模型无从知道该改哪一个
        raise TypeError("参数不合法")
    return f"{item_id} 现有库存 {quantity * 3} 件"


@typed_tool
def 扣库存(item_id: str, quantity: int) -> str:
    """扣减库存余量（取值不合法时返回结构化错误，而不是抛异常）。

    Args:
        item_id: 库存条目编号，形如 S-200
        quantity: 需要扣减的数量，必须是正整数
    """
    if quantity <= 0:
        # 取值非法属于"业务上的否定结果"，调用本身是成功的，
        # 所以返回结构化错误，让模型读 JSON 而不是去猜异常文案
        return json.dumps(
            {
                "ok": False,
                "error": {
                    "param": "quantity",
                    "expected": "正整数",
                    "received": quantity,
                    "hint": "把 quantity 改成 1 以上的整数",
                },
            },
            ensure_ascii=False,
        )
    return json.dumps({"ok": True, "remaining": quantity * 3}, ensure_ascii=False)


@tool
def 查库存吞错误(item_id: str) -> str:
    """查库存余量（反面教材：把异常吞掉，模型永远不知道自己错了）。"""
    try:
        if item_id != "S-200":
            raise RuntimeError("库存服务不可用")
        return "S-200 现有库存 30 件"
    except Exception:  # noqa: BLE001 - 这里就是要演示"吞掉"的后果
        return "查询完成"


@tool
def 查物流(运单号: str) -> str:
    """查询物流轨迹（只处理物流运单，不处理订单和退款）。

    Args:
        运单号: 物流运单号，形如 S-2001
    """
    if not 运单号.startswith("S-"):
        raise ValueError("本工具只处理 S- 开头的物流运单号，不支持订单号或退款单号")
    return f"运单 {运单号} 在途中"


@tool
def 查退款(退款单号: str) -> str:
    """查询退款到账进度。

    Args:
        退款单号: 退款单号，形如 R-3001
    """
    return f"退款 {退款单号} 已到账"


@tool
def 业务查询(业务类型: str, 编号: str) -> str:
    """按业务类型查询（订单、物流、退款共用这一个工具）。

    Args:
        业务类型: 业务类型代码，T1=订单，T2=物流，T3=退款
        编号: 业务编号
    """
    table = {"T1": "订单已发货", "T2": "物流在途中", "T3": "退款已到账"}
    if 业务类型 not in table:
        raise ValueError(f"业务类型 {业务类型!r} 不支持，可用代码：T1 / T2 / T3")
    return f"{编号}：{table[业务类型]}"


@tool
def 模糊业务查询(业务类型: str, 编号: str) -> str:
    """按业务类型查询（订单、物流、退款共用这一个工具）。

    Args:
        业务类型: 业务类型代码
        编号: 业务编号
    """
    table = {"T1": "订单已发货", "T2": "物流在途中", "T3": "退款已到账"}
    if 业务类型 not in table:
        raise ValueError(f"业务类型 {业务类型!r} 不支持，可用代码：T1 / T2 / T3")
    return f"{编号}：{table[业务类型]}"


# ---------------------------------------------------------------- 辅助工具


_ERROR_PREFIXES = ("错误：", "参数错误：", "参数取值非法：", "工具执行失败：")


def _is_error_text(text: str) -> bool:
    """判断一条工具消息是不是"错误回填"。

    运行时把三类失败都翻译成一条以固定前缀开头的消息，
    模型就是靠看见它才知道自己错了——所以这里用同一套前缀判断。
    """
    return text.startswith(_ERROR_PREFIXES)


def _schema_tokens(tools: Sequence[Tool]) -> int:
    """量一下这批工具塞进上下文要花多少 token。"""
    payload = json.dumps([t.to_api() for t in tools], ensure_ascii=False)
    return approx_tokens(payload)


def _has_prefix_collision(names: Sequence[str]) -> bool:
    """一个工具名是另一个的前缀——模型很容易选错。

    ``查订单`` / ``查订单详情`` 就是典型：模型只看到名字，
    分不清用户问的是哪个，只能靠描述硬猜。
    """
    for name in names:
        for other in names:
            if name != other and other.startswith(name):
                return True
    return False


def _duplicate_descriptions(tools: Sequence[Tool]) -> list[str]:
    """说明一字不差重复的工具名——同义工具，纯浪费上下文。"""
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for item in tools:
        if item.description in seen:
            duplicates.append(item.name)
        else:
            seen[item.description] = item.name
    return duplicates


def _make_indexed_tool(index: int) -> Tool:
    """造第 index 个同构工具——规模实验需要一把"一样的工具"来量体积。"""

    def 查询数据(record_id: str, limit: int = 10) -> str:
        """查询数据。"""
        return record_id

    查询数据.__name__ = f"查询数据_{index}"
    查询数据.__doc__ = (
        f"查询第 {index} 类数据。\n\n"
        "Args:\n"
        "    record_id: 记录标识，形如 R-100\n"
        "    limit: 最多返回多少条记录，默认 10\n"
    )
    return typed_tool(查询数据)


@dataclass
class RepairModel:
    """按错误文本来修参数的模型——用来证明"文案质量决定纠正能力"。

    它模拟一个称职但信息受限的模型：只修**错误文案点名**的参数；
    文案没点名，它就无从下手，只能原样重试。真实模型也是这样——
    能修的前提是错误信息里真的有线索。
    """

    tool_name: str
    initial: dict[str, Any]
    repairs: dict[str, Any]
    final_answer: str

    current: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0

    def __post_init__(self) -> None:
        if not self.current:
            self.current = dict(self.initial)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """按上一条工具消息决定"修参数"还是"收尾"。"""
        self.rounds += 1
        last = messages[-1] if messages else None
        if last is None or last.role != "tool":
            return ModelReply(
                tool_calls=[ToolCall(f"call_{self.rounds}", self.tool_name, dict(self.current))]
            )

        if _is_error_text(last.content):
            fixed = dict(self.current)
            for name, value in self.repairs.items():
                if name in last.content and fixed.get(name) != value:
                    fixed[name] = value
                    self.current = fixed
                    return ModelReply(
                        tool_calls=[ToolCall(f"call_{self.rounds}", self.tool_name, dict(fixed))]
                    )
            # 文案没点名参数 → 无从纠正，原样重试
            return ModelReply(
                tool_calls=[ToolCall(f"call_{self.rounds}", self.tool_name, dict(self.current))]
            )

        return ModelReply(content=self.final_answer)


# ---------------------------------------------------------------- 1. schema


def check_schema_from_docstring() -> None:
    """参数说明必须落到字段级 description，否则模型只能从一整坨文本里猜。"""
    section("schema：模型看到的只有 name / description / parameters")

    parsed = _parse_arg_descriptions(查订单.func.__doc__ or "")
    ensure(
        parsed == {"order_id": "订单号，形如 A-1001"},
        f"docstring 的 Args 段落应当被解析成参数字典，实际 {parsed!r}",
    )

    field = 查订单.parameters["properties"]["order_id"]
    ensure(
        field.get("description") == "订单号，形如 A-1001",
        f"字段级 description 应当来自 docstring 的 Args 段落，实际 {field!r}",
    )
    ensure(
        field.get("type") == "string",
        f"字段类型应当来自类型注解，实际 {field!r}",
    )
    ensure(
        查订单.parameters["required"] == ["order_id"],
        f"没有默认值的参数应当是 required，实际 {查订单.parameters['required']!r}",
    )

    # 对照组：没有 Args 段落的工具——函数级说明还在，字段级说明没了
    bare = 查订单无字段说明.parameters["properties"]["order_id"]
    ensure(
        "description" not in bare,
        f"没有 Args 段落时字段级不应有说明，实际 {bare!r}",
    )
    ensure(
        bool(查订单无字段说明.description.strip()),
        "对照组的函数级说明应当非空",
    )
    print(f"  带 Args 段落的字段说明：{field}")
    print(f"  没写 Args 段落的字段说明：{bare}")
    print(f"  两种写法的 schema 体积：{approx_tokens(json.dumps(查订单.parameters, ensure_ascii=False))}"
          f" vs {approx_tokens(json.dumps(查订单无字段说明.parameters, ensure_ascii=False))} tokens")

    # 坑：PEP 563 把注解压成字符串，naive 生成器按类型对象查表 → int 变 string
    def _探针(item_id: str, quantity: int) -> str:
        """探针：只用来观察 schema 生成，不注册成工具。"""
        return item_id

    # 本仓库的 labkit 已经用 get_type_hints 解析过注解，所以类型是对的
    fixed_schema = tool(_探针).parameters["properties"]["quantity"]
    ensure(
        fixed_schema["type"] == "integer",
        f"labkit 应当把 int 注解解析成 integer，实际 {fixed_schema!r}",
    )

    # 但如果你自己按 inspect 的原始注解查类型表，就会踩这个坑
    naive_annotation = inspect.signature(_探针).parameters["quantity"].annotation
    ensure(
        naive_annotation == "int",
        f"开了 PEP 563 之后原始注解应当是字符串 'int'，实际 {naive_annotation!r}",
    )
    naive_type = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
    naive_result = naive_type.get(naive_annotation, "string")
    ensure(
        naive_result == "integer",
        f"按字符串查表也应当查到 integer，实际 {naive_result!r}",
    )

    resolved = inspect.signature(_探针, eval_str=True).parameters["quantity"].annotation
    ensure(
        resolved is int,
        f"eval_str=True 应当把注解还原成真实类型，实际 {resolved!r}",
    )
    print(f"  坑：PEP 563 下原始注解是 {naive_annotation!r}（字符串），不是类型对象")
    print(f"      → naive 生成器会把它当成 string；labkit 用 get_type_hints 解析成了 {fixed_schema['type']!r}")


# ---------------------------------------------------------------- 2. 幻觉工具名


def check_hallucinated_tool() -> None:
    """模型幻觉了工具名——运行时得回一句"没有这个工具"并列出可用清单。"""
    section("幻觉工具名：回一句'没有这个工具' + 可用清单")

    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查询订单状态", {"order_id": "A-1001"})]),
            ModelReply(tool_calls=[ToolCall("call_2", "查订单", {"order_id": "A-1001"})]),
            ModelReply(content="订单 A-1001 已发货"),
        ]
    )
    agent = Agent(model, [查订单], AgentConfig(max_iterations=6))
    result = agent.run("A-1001 什么状态？")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 2, f"应当有一次失败 + 一次成功，实际 {len(tool_messages)} 次工具调用")
    error_text = tool_messages[0].content
    ensure(
        "没有名为" in error_text,
        f"工具不存在时应当回一句'没有名为…的工具'，实际 {error_text!r}",
    )
    ensure(
        "查询订单状态" in error_text,
        f"错误应当点名幻觉出来的工具名，实际 {error_text!r}",
    )
    ensure(
        "查订单" in error_text,
        f"错误应当列出可用工具，实际 {error_text!r}",
    )
    ensure(
        result.tool_calls[1].name == "查订单",
        f"模型应当改用正确的工具，实际 {result.tool_calls[1].name!r}",
    )
    ensure(
        result.final_answer == "订单 A-1001 已发货",
        f"最终回答不对：{result.final_answer!r}",
    )
    print(f"  第一次：{error_text}")
    print(f"  第二次：{tool_messages[1].content}")


# ---------------------------------------------------------------- 3. 缺参数


def check_missing_parameter() -> None:
    """缺参数是最容易纠正的一类失败——只要错误文案点名了参数。"""
    section("缺参数：错误可见 → 模型改对")

    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查订单", {})]),
            ModelReply(tool_calls=[ToolCall("call_2", "查订单", {"order_id": "A-1001"})]),
            ModelReply(content="订单 A-1001 已发货"),
        ]
    )
    agent = Agent(model, [查订单], AgentConfig(max_iterations=6))
    result = agent.run("A-1001 什么状态？")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 2, f"应当有一次失败 + 一次成功，实际 {len(tool_messages)} 次工具调用")
    error_text = tool_messages[0].content
    ensure(
        error_text.startswith("参数错误："),
        f"缺参数应当回一条参数错误，实际 {error_text!r}",
    )
    ensure(
        "order_id" in error_text,
        f"错误文案应当点名参数 order_id，实际 {error_text!r}",
    )
    ensure(
        result.tool_calls[1].arguments == {"order_id": "A-1001"},
        f"模型应当补上缺的参数，实际 {result.tool_calls[1].arguments!r}",
    )
    ensure(
        result.final_answer == "订单 A-1001 已发货",
        f"最终回答不对：{result.final_answer!r}",
    )
    print(f"  第一次：{error_text}")
    print(f"  第二次：{tool_messages[1].content}")


# ---------------------------------------------------------------- 4. 类型错


def check_type_error() -> None:
    """类型错——错误信息里必须带参数名，模型才知道该改哪个。"""
    section("类型错：错误带参数名 → 模型纠正")

    model = scripted_model(
        [
            ModelReply(
                tool_calls=[ToolCall("call_1", "查库存", {"item_id": "S-200", "quantity": "3"})]
            ),
            ModelReply(
                tool_calls=[ToolCall("call_2", "查库存", {"item_id": "S-200", "quantity": 3})]
            ),
            ModelReply(content="S-200 现有库存 9 件"),
        ]
    )
    agent = Agent(model, [查库存], AgentConfig(max_iterations=6))
    result = agent.run("S-200 要 3 件")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 2, f"应当有一次失败 + 一次成功，实际 {len(tool_messages)} 次工具调用")
    error_text = tool_messages[0].content
    ensure(
        "参数错误" in error_text,
        f"类型错应当回一条参数错误，实际 {error_text!r}",
    )
    ensure(
        "quantity" in error_text,
        f"错误信息里必须带参数名 quantity，实际 {error_text!r}",
    )
    ensure(
        isinstance(result.tool_calls[1].arguments["quantity"], int),
        f"模型应当把 quantity 改成整数，实际 {result.tool_calls[1].arguments!r}",
    )
    ensure(
        result.final_answer == "S-200 现有库存 9 件",
        f"最终回答不对：{result.final_answer!r}",
    )
    print(f"  第一次：{error_text}")
    print(f"  第二次：{tool_messages[1].content}")


# ---------------------------------------------------------------- 5. 取值域非法


def check_value_domain() -> None:
    """取值域非法——最难纠正的一类，除非错误里写清了合法范围。"""
    section("取值域非法：结构化错误 → 模型纠正")

    model = scripted_model(
        [
            ModelReply(
                tool_calls=[ToolCall("call_1", "扣库存", {"item_id": "S-200", "quantity": -3})]
            ),
            ModelReply(
                tool_calls=[ToolCall("call_2", "扣库存", {"item_id": "S-200", "quantity": 3})]
            ),
            ModelReply(content="已扣减 3 件"),
        ]
    )
    agent = Agent(model, [扣库存], AgentConfig(max_iterations=6))
    result = agent.run("S-200 扣 3 件")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 2, f"应当有一次失败 + 一次成功，实际 {len(tool_messages)} 次工具调用")
    error_text = tool_messages[0].content
    payload = json.loads(error_text)
    ensure(
        payload["ok"] is False,
        f"结构化错误应当把 ok 标成 False，实际 {payload!r}",
    )
    ensure(
        payload["error"]["param"] == "quantity",
        f"结构化错误应当点名参数，实际 {payload!r}",
    )
    ensure(
        payload["error"]["expected"] == "正整数",
        f"结构化错误应当写清期望值，实际 {payload!r}",
    )
    ensure(
        payload["error"]["received"] == -3,
        f"结构化错误应当回显实际收到的值，实际 {payload!r}",
    )
    ensure(
        result.tool_calls[1].arguments["quantity"] == 3,
        f"模型应当改成合法取值，实际 {result.tool_calls[1].arguments!r}",
    )
    ensure(
        result.final_answer == "已扣减 3 件",
        f"最终回答不对：{result.final_answer!r}",
    )
    print(f"  第一次：{error_text}")

    # 坑：用 ValueError 做取值校验，会落进运行时的"业务故障"那一档，
    # 文案变成"或换用其他工具"——对一个能改参数的错来说，纯误导
    value_error_model = scripted_model(
        [ModelReply(tool_calls=[ToolCall("call_1", "查库存", {"item_id": "S-200", "quantity": 0})])]
    )
    value_error_result = Agent(
        value_error_model, [查库存], AgentConfig(max_iterations=1)
    ).run("S-200 扣 0 件")
    value_error_text = [m for m in value_error_result.messages if m.role == "tool"][0].content
    ensure(
        "工具执行失败" in value_error_text and "或换用其他工具" in value_error_text,
        f"ValueError 应当落进业务故障那一档的文案，实际 {value_error_text!r}",
    )
    print(f"  对照：ValueError 的文案 → {value_error_text}")


# ---------------------------------------------------------------- 6. 三种错误面


def check_error_faces() -> None:
    """抛异常 / 返回结构化错误 / 直接吞掉——第三种最糟。"""
    section("错误面的三种设计")

    a_model = scripted_model(
        [ModelReply(tool_calls=[ToolCall("call_1", "查库存", {"item_id": "S-200", "quantity": "3"})])]
    )
    a_result = Agent(a_model, [查库存], AgentConfig(max_iterations=1)).run("查库存")
    a_text = [m for m in a_result.messages if m.role == "tool"][0].content

    b_model = scripted_model(
        [ModelReply(tool_calls=[ToolCall("call_1", "扣库存", {"item_id": "S-200", "quantity": -3})])]
    )
    b_result = Agent(b_model, [扣库存], AgentConfig(max_iterations=1)).run("扣库存")
    b_text = [m for m in b_result.messages if m.role == "tool"][0].content

    c_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查库存吞错误", {"item_id": "S-999"})]),
            ModelReply(content="库存查询完成，现有 30 件"),
        ]
    )
    c_result = Agent(c_model, [查库存吞错误], AgentConfig(max_iterations=3)).run("S-999 还有多少库存？")
    c_text = [m for m in c_result.messages if m.role == "tool"][0].content

    for label, text in (
        ("A 抛异常", a_text),
        ("B 结构化错误", b_text),
        ("C 直接吞掉", c_text),
    ):
        print(f"  {label:<14} → {text[:96]}")

    ensure("quantity" in a_text, f"设计 A 的错误应当点名参数，实际 {a_text!r}")
    ensure(
        "\"param\"" in b_text and "quantity" in b_text,
        f"设计 B 的结构化错误应当点名参数，实际 {b_text!r}",
    )
    ensure(
        not _is_error_text(c_text),
        f"设计 C 应当让模型看不出这是错误，实际 {c_text!r}",
    )
    ensure(
        "30" not in c_text,
        f"设计 C 里的工具从没返回过 30 件，实际 {c_text!r}",
    )
    ensure(
        "30" in c_result.final_answer,
        f"模型会拿着'查询完成'编一个数字出来，实际 {c_result.final_answer!r}",
    )
    print(f"  设计 C 的最终回答：{c_result.final_answer!r} ← 数字是编的")


# ---------------------------------------------------------------- 7. 文案决定纠正能力


def check_error_text_drives_repair() -> None:
    """同样的 schema、同样的输入，一个能纠正一个卡死——差别只在错误文案。"""
    section("错误文案质量决定纠正能力")

    good = RepairModel(
        tool_name="查库存",
        initial={"item_id": "S-200", "quantity": "3"},
        repairs={"quantity": 3},
        final_answer="S-200 现有库存 9 件",
    )
    good_result = Agent(good, [查库存], AgentConfig(max_iterations=5)).run("S-200 要 3 件")
    ensure(
        good_result.final_answer == "S-200 现有库存 9 件",
        f"错误文案点名了参数，模型应当能纠正并收尾，实际 {good_result.final_answer!r}",
    )
    ensure(
        good_result.tool_calls[-1].arguments["quantity"] == 3,
        f"最后一次调用应当把 quantity 改对，实际 {good_result.tool_calls[-1].arguments!r}",
    )

    bad = RepairModel(
        tool_name="查库存无参数名",
        initial={"item_id": "S-200", "quantity": "3"},
        repairs={"quantity": 3},
        final_answer="查好了",
    )
    bad_result = Agent(bad, [查库存无参数名], AgentConfig(max_iterations=5)).run("S-200 要 3 件")
    ensure(
        bad_result.stop_reason == "max_iterations",
        f"错误文案不点名参数时模型应当卡死，实际 stop_reason={bad_result.stop_reason!r}",
    )
    ensure(
        bad_result.final_answer == "",
        f"卡死时不该有最终回答，实际 {bad_result.final_answer!r}",
    )
    ensure(
        all(_is_error_text(m.content) for m in bad_result.messages if m.role == "tool"),
        "每一次工具调用都应当是错误回填",
    )

    # 两个工具的 schema 一模一样，唯一差别是错误文案
    ensure(
        查库存.parameters == 查库存无参数名.parameters,
        "两个工具的 schema 应当完全一致，差别只在错误文案",
    )
    print(f"  点名参数：{good_result.stop_reason}，{good_result.final_answer!r}")
    print(f"  不点名  ：{bad_result.stop_reason}，{len(bad_result.tool_calls)} 次重试后卡死")


# ---------------------------------------------------------------- 8. 工具粒度


def check_granularity() -> None:
    """一个工具 = 一个用户意图。粒度不对，选错之后的恢复方式完全不同。"""
    section("工具粒度：粗粒度 vs 细粒度")

    request = "A-2001 的退款什么时候到账？"

    # 细粒度：选错工具 → 那个工具自己说清边界 → 模型换工具 → 答对
    fine_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "查物流", {"运单号": "A-2001"})]),
            ModelReply(tool_calls=[ToolCall("call_2", "查退款", {"退款单号": "A-2001"})]),
            ModelReply(content="退款 A-2001 已到账"),
        ]
    )
    fine = Agent(
        fine_model, [查订单, 查物流, 查退款], AgentConfig(max_iterations=6)
    ).run(request)
    fine_error = [m for m in fine.messages if m.role == "tool"][0].content
    ensure(
        "退款" in fine_error and "运单号" in fine_error,
        f"细粒度选错时错误应当说清工具边界，实际 {fine_error!r}",
    )
    ensure(
        fine.tool_calls[0].name != fine.tool_calls[1].name,
        f"细粒度的恢复方式是换一个工具，实际 {[c.name for c in fine.tool_calls]!r}",
    )
    ensure(
        fine.tool_calls[1].name == "查退款",
        f"应当换到查退款，实际 {fine.tool_calls[1].name!r}",
    )
    ensure(
        fine.final_answer == "退款 A-2001 已到账",
        f"细粒度应当拿到退款状态，实际 {fine.final_answer!r}",
    )

    # 粗粒度 + 字段说明写清映射：一次就对
    coarse_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "业务查询", {"业务类型": "T3", "编号": "A-2001"})]),
            ModelReply(content="退款 A-2001 已到账"),
        ]
    )
    coarse = Agent(coarse_model, [业务查询], AgentConfig(max_iterations=6)).run(request)
    mapping = 业务查询.parameters["properties"]["业务类型"]["description"]
    ensure(
        "T3=退款" in mapping,
        f"字段说明应当写清代码到意图的映射，实际 {mapping!r}",
    )
    ensure(
        coarse.final_answer == "退款 A-2001 已到账",
        f"写了映射的粗粒度工具应当一次答对，实际 {coarse.final_answer!r}",
    )

    # 粗粒度 + 字段说明没写映射：模型只能瞎猜 → 技术上成功，语义上跑偏
    vague_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("call_1", "模糊业务查询", {"业务类型": "退款", "编号": "A-2001"})]),
            ModelReply(tool_calls=[ToolCall("call_2", "模糊业务查询", {"业务类型": "T1", "编号": "A-2001"})]),
            ModelReply(content="A-2001 订单已发货"),
        ]
    )
    vague = Agent(vague_model, [模糊业务查询], AgentConfig(max_iterations=6)).run(request)
    vague_error = [m for m in vague.messages if m.role == "tool"][0].content
    ensure(
        "T1 / T2 / T3" in vague_error,
        f"取值非法时应当列出可用代码，实际 {vague_error!r}",
    )
    ensure(
        "T3=退款" not in vague_error,
        f"错误文案里没有意图到代码的映射，模型无从纠正，实际 {vague_error!r}",
    )
    ensure(
        "退款" not in vague.final_answer,
        f"映射缺失时模型只能猜，答案会跑偏，实际 {vague.final_answer!r}",
    )
    ensure(
        vague.final_answer == "A-2001 订单已发货",
        f"本场景里模型猜了 T1（订单），实际 {vague.final_answer!r}",
    )

    fine_tokens = _schema_tokens([查订单, 查物流, 查退款])
    coarse_tokens = _schema_tokens([业务查询])
    ensure(
        fine_tokens > coarse_tokens,
        f"细粒度的 schema 更大，期望细 > 粗，实际 {fine_tokens} ≤ {coarse_tokens}",
    )
    print(f"  细粒度（3 个工具）：选错 → 换工具 → {fine.final_answer!r}（schema ≈ {fine_tokens} tokens）")
    print(f"  粗粒度（写了映射）：一次就对 → {coarse.final_answer!r}（schema ≈ {coarse_tokens} tokens）")
    print(f"  粗粒度（没写映射）：猜了一个 → {vague.final_answer!r} ← 静默的错答")


# ---------------------------------------------------------------- 9. 工具命名


def check_naming_collisions() -> None:
    """同义工具名、前缀相近的工具名——上线前就能查出来。"""
    section("工具命名：同义工具名是可检测的坑")

    bad_names = [查订单.name, 查订单详情.name, 查询订单.name]
    good_names = [查订单.name, 查物流.name, 查退款.name]

    ensure(
        _has_prefix_collision(bad_names),
        f"应当检测到前缀冲突（查订单 / 查订单详情），实际 {bad_names!r}",
    )
    ensure(
        not _has_prefix_collision(good_names),
        f"正常的一组工具不该有前缀冲突，实际 {good_names!r}",
    )

    duplicates = _duplicate_descriptions([查订单, 查订单详情, 查询订单])
    ensure(
        "查询订单" in duplicates,
        f"应当检测到说明一字不差的同义工具，实际 {duplicates!r}",
    )
    ensure(
        not _duplicate_descriptions([查订单, 查物流, 查退款]),
        "正常的一组工具不该有说明重复",
    )

    ensure(
        查订单.description == 查询订单.description,
        "这两个同义工具的说明应当一字不差",
    )
    ensure(
        _schema_tokens([查订单, 查询订单]) > _schema_tokens([查订单]),
        "同义工具让 schema 变长",
    )
    print(f"  问题组合：{bad_names!r} → 前缀冲突 + 说明重复")
    print(f"  正常组合：{good_names!r} → 无冲突")


# ---------------------------------------------------------------- 10. 工具数量


def check_tool_count_scaling() -> None:
    """工具越多，schema 越长；这段开销每次模型调用都要付。"""
    section("工具数量 → schema 体积")

    counts = [1, 4, 8, 16, 32]
    rows: list[tuple[int, int]] = []
    for count in counts:
        tools = [_make_indexed_tool(index) for index in range(count)]
        rows.append((count, _schema_tokens(tools)))

    for count, tokens in rows:
        print(f"  {count:>3} 个工具 → tools payload ≈ {tokens} tokens")

    for (prev_count, prev_tokens), (count, tokens) in zip(rows, rows[1:]):
        ensure(
            tokens > prev_tokens,
            f"{count} 个工具的 schema 应当比 {prev_count} 个大，实际 {tokens} ≤ {prev_tokens}",
        )

    base_count, base_tokens = rows[0]
    top_count, top_tokens = rows[-1]
    ensure(
        top_tokens >= 10 * base_tokens,
        f"{top_count} 个工具的 schema 应当至少是 {base_count} 个的 10 倍，实际 {top_tokens} vs {base_tokens}",
    )
    print(f"  {top_count} 个工具的固定开销 ≈ {top_tokens} tokens，"
          f"而每次模型调用都会把它整段再付一遍")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型面对一个会拒绝错误参数的工具，只校验形状。"""
    banner("第 3 章 · 真实模型路径")
    from ..labkit import real_model

    model = real_model()
    agent = Agent(model, [查库存, 查订单], AgentConfig(max_iterations=6))
    result = agent.run("S-200 还有多少库存？我要 3 件")

    for message in result.messages:
        print(f"  {message.role}: {message.content[:120]!r}")

    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "工具消息必须带 tool_call_id",
    )
    print(f"  结束原因：{result.stop_reason}")


def run() -> None:
    """跑第 3 章实验。"""
    banner("第 3 章 · 工具设计：schema、错误面与粒度")
    check_schema_from_docstring()
    check_hallucinated_tool()
    check_missing_parameter()
    check_type_error()
    check_value_domain()
    check_error_faces()
    check_error_text_drives_repair()
    check_granularity()
    check_naming_collisions()
    check_tool_count_scaling()
    rule("模型只看得见 schema 和错误文案：schema 说谎它就传错，错误文案含糊它就卡死")
    print("\n✓ 第 3 章通过")
