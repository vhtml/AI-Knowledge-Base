"""LangChain 工具（@tool）内部机制的可运行演示。

把 @tool 拆成三个阶段各看一遍，再跑一次零网络的 Graph 闭环：

① 装饰期：Python 函数 → StructuredTool + Pydantic 入参 schema
② 绑定期：StructuredTool → 发给模型的 function 定义（真正上线的 payload）
③ 调用期：tool_calls → 参数校验 → 执行 → ToolMessage

本模块不联网：用按脚本吐消息的假模型驱动 Graph，所以不受模型端点限流影响，
任何时候都能跑出同样的结果。

运行：cd Demos && uv run python langgraph-demo/tool_basics.py
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


# ---------- 工具定义 ----------

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool(parse_docstring=True)
def divide(a: int, b: int) -> float:
    """Divide `a` by `b`.

    Args:
        a: Dividend.
        b: Divisor.
    """
    return a / b


def _safe_add(a: int, b: int) -> int:
    """Adds `a` and `b`, turning a validation failure into readable output."""
    return a + b


def _guarded_divide(a: float, b: float) -> float:
    """Divide `a` by `b`, reporting a zero divisor as readable output."""
    if b == 0:
        raise ToolException("除数不能为 0")
    return a / b


def _detailed_add(a: int, b: int) -> int:
    """Adds `a` and `b`, keeping the full validation error as readable output."""
    return a + b


# 错误处理开关是 BaseTool 的字段，@tool() 装饰器不暴露这两个参数，
# 所以需要自定义错误行为时直接构造 StructuredTool
safe_add = StructuredTool.from_function(
    _safe_add,
    name="safe_add",
    description="Adds `a` and `b`, turning a validation failure into readable output.",
    handle_validation_error=True,
)
# flag=True 只会给一句笼统的 "Tool input validation error"，
# 换成可调用形式才能保留 pydantic 的具体报错
detailed_add = StructuredTool.from_function(
    _detailed_add,
    name="detailed_add",
    description="Adds `a` and `b`, keeping the full validation error as readable output.",
    handle_validation_error=lambda exc: f"参数不合法：{exc}",
)
guarded_divide = StructuredTool.from_function(
    _guarded_divide,
    name="guarded_divide",
    description="Divide `a` by `b`, reporting a zero divisor as readable output.",
    handle_tool_error=True,
)


TOOLS = [add, divide, safe_add, detailed_add, guarded_divide]
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in TOOLS}


# ---------- 断言收集 ----------

@dataclass
class Checks:
    """收集演示过程中的断言结果，最后统一汇报。"""

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def expect(self, label: str, condition: bool) -> None:
        """记录一条检查；condition 为 False 时记为失败。"""
        (self.passed if condition else self.failed).append(label)


def show(title: str) -> None:
    """打印分节标题，让演示输出可读。"""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def dump(payload: Any) -> None:
    """以缩进 JSON 打印结构化内容。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


# ---------- ① 装饰期 ----------

def demo_decoration_phase(checks: Checks) -> None:
    """展示 @tool 如何把函数变成带 schema 的 StructuredTool。"""
    show("① 装饰期：函数 → StructuredTool + Pydantic schema")

    print(f"add 的类型链: StructuredTool ← BaseTool ← RunnableSerializable")
    print(f"实际类型: {type(add).__name__}")
    print(f"name: {add.name}")
    print(f"description: {add.description!r}")
    print(f"原函数仍在 .func 上: {add.func.__name__}")
    print(f"args_schema: {add.args_schema}（Pydantic 模型，字段 {list(add.args_schema.model_fields)}）")

    checks.expect("@tool 后 add 是 StructuredTool", type(add).__name__ == "StructuredTool")
    checks.expect("工具名来自函数名", add.name == "add")
    checks.expect("description 来自 docstring", add.description.startswith("Adds `a` and `b`."))
    checks.expect("args_schema 自动生成 int 字段", set(add.args_schema.model_fields) == {"a", "b"})

    print("\n--- add.args（注意：这是 properties 层，不含 required/type 包装）---")
    dump(add.args)
    print("\n--- add.tool_call_schema 的完整 object schema（发给模型的 parameters）---")
    dump(add.tool_call_schema.model_json_schema())
    print("\n--- divide.args（parse_docstring=True，参数描述被解析出来）---")
    dump(divide.args)

    add_has_desc = any("description" in prop for prop in add.args.values())
    divide_has_desc = all("description" in prop for prop in divide.args.values())
    checks.expect("默认 parse_docstring=False，参数无 description", not add_has_desc)
    checks.expect("parse_docstring=True 时参数带 description", divide_has_desc)


# ---------- ② 绑定期 ----------

def demo_binding_phase(checks: Checks) -> None:
    """展示工具如何被转成发给模型的 function 定义。"""
    show("② 绑定期：StructuredTool → 发给 API 的 tools payload")

    payload = convert_to_openai_tool(add)
    print("convert_to_openai_tool(add) 的结果：")
    dump(payload)

    func = payload["function"]
    checks.expect("payload 是 OpenAI function 格式", payload.get("type") == "function")
    checks.expect("payload 含 name/description/parameters", {"name", "description", "parameters"} <= set(func))
    checks.expect("required 由 schema 生成", func["parameters"].get("required") == ["a", "b"])
    checks.expect("转换后剥掉了 title", "title" not in func["parameters"]["properties"]["a"])


# ---------- ③ 调用期 ----------

def demo_invocation_phase(checks: Checks) -> None:
    """展示 tool_calls 如何被路由、校验、执行。"""
    show("③ 调用期：派发 → 校验 → 执行 → ToolMessage")

    tool_call = {"name": "add", "args": {"a": 3, "b": 4}, "id": "call_1", "type": "tool_call"}

    raw = add.invoke({"a": 3, "b": 4})
    print(f"add.invoke(纯 args)      -> {raw!r}（{type(raw).__name__}）")

    as_message = add.invoke(tool_call)
    print(f"add.invoke(完整 tool_call) -> {as_message!r}")

    checks.expect("传纯 args 返回原始 Python 值", raw == 7 and isinstance(raw, int))
    checks.expect("传完整 tool_call 返回 ToolMessage", isinstance(as_message, ToolMessage))
    checks.expect("ToolMessage 带 tool_call_id", getattr(as_message, "tool_call_id", None) == "call_1")
    checks.expect("ToolMessage 内容是工具结果", as_message.content == "7")

    print("\n--- 参数校验（Pydantic 兜底）---")
    for bad, label in (
        ({"a": "不是数字", "b": 4}, "类型错误"),
        ({"a": 3}, "缺少必填参数 b"),
    ):
        try:
            add.invoke(bad)
            checks.expect(f"{label} 应该抛 ValidationError", False)
        except Exception as exc:  # noqa: BLE001 - 演示脚本：任何异常都要抓住并展示
            print(f"{label}: {type(exc).__name__}")
            checks.expect(f"{label} 抛 ValidationError", type(exc).__name__ == "ValidationError")

    print("\n--- 校验失败转成可读输出（handle_validation_error=True）---")
    readable = safe_add.invoke({"a": "不是数字", "b": 4})
    print(f"safe_add.invoke(坏参数) -> {readable!r}")
    checks.expect("handle_validation_error=True 时不再抛异常", isinstance(readable, str))

    print("\n--- 可调用 handler 保留具体报错 ---")
    detailed = detailed_add.invoke({"a": "不是数字", "b": 4})
    print(f"detailed_add.invoke(坏参数) -> {detailed!r}")
    checks.expect("可调用 handler 带出具体报错", "参数不合法" in detailed)
    checks.expect(
        "flag=True 只给笼统提示", readable == "Tool input validation error"
    )

    print("\n--- 业务异常转成可读输出（ToolException + handle_tool_error=True）---")
    guarded = guarded_divide.invoke({"a": 1, "b": 0})
    print(f"guarded_divide.invoke(1, 0) -> {guarded!r}")
    checks.expect("ToolException 被转成可读 observation", guarded == "除数不能为 0")

    print("\n--- 普通异常不会被 handle_tool_error 兜住 ---")
    try:
        divide.invoke({"a": 1, "b": 0})
        checks.expect("ZeroDivisionError 应当继续抛出", False)
    except ZeroDivisionError as exc:
        print(f"divide.invoke(1, 0) -> {type(exc).__name__}: {exc}")
        checks.expect("非 ToolException 直接抛出", True)


# ---------- 假模型 + Graph 闭环 ----------

class ScriptedChatModel(BaseChatModel):
    """按脚本依次吐出预设回复的假模型：用来离线演示 tool_calls 链路。"""

    replies: list[AIMessage]
    cursor: int = 0
    received_tools: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        """LangChain 要求实现的模型类型标识。"""
        return "scripted"

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """把工具转成 function payload 绑定进来；这里顺便记录模型实际收到什么。"""
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """按顺序返回预设回复，最后一条会重复。"""
        self.received_tools = list(kwargs.get("tools") or [])
        reply = self.replies[min(self.cursor, len(self.replies) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


class MessagesState(TypedDict):
    """演示用的最小状态：只要消息和 LLM 调用次数。"""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def build_graph(model: BaseChatModel) -> Any:
    """按 quickstart 的结构搭出 llm_call → tool_node → END 的图。"""
    model_with_tools = model.bind_tools(TOOLS)

    def llm_call(state: MessagesState) -> dict:
        """调用模型；模型是否要调工具完全由它返回的 tool_calls 决定。"""
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: MessagesState) -> dict:
        """按名字路由到对应工具并执行，把结果包成 ToolMessage。"""
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            selected = TOOLS_BY_NAME[tool_call["name"]]
            observation = selected.invoke(tool_call["args"])
            results.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": results}

    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """模型带 tool_calls 就去执行工具，否则收尾。"""
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile()


def demo_graph_loop(checks: Checks) -> None:
    """用假模型跑一遍完整闭环，展示消息轨迹。"""
    show("④ Graph 闭环（假模型，零网络）")

    scripted = ScriptedChatModel(
        replies=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 3, "b": 4}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="The sum of 3 and 4 is 7."),
        ]
    )

    agent = build_graph(scripted)
    result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]})

    print("--- 模型实际收到的 tools payload（第一个工具）---")
    dump(scripted.received_tools[0])

    print("\n--- 消息轨迹 ---")
    for msg in result["messages"]:
        kind = type(msg).__name__
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"AIMessage(tool_calls) -> {msg.tool_calls}")
        else:
            print(f"{kind} -> {msg.content!r}")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    checks.expect("轨迹里有 1 条 ToolMessage", len(tool_messages) == 1)
    checks.expect("ToolMessage 内容是 7", tool_messages and tool_messages[0].content == "7")
    checks.expect("模型共被调用 2 次（一次要工具，一次作答）", result["llm_calls"] == 2)
    checks.expect("模型收到了 function payload", bool(scripted.received_tools))
    checks.expect(
        "payload 里参数是 integer", scripted.received_tools[0]["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    )


def main() -> None:
    """跑完整套演示并汇报检查结果。"""
    checks = Checks()
    demo_decoration_phase(checks)
    demo_binding_phase(checks)
    demo_invocation_phase(checks)
    demo_graph_loop(checks)

    show("检查汇总")
    for label in checks.passed:
        print(f"  ✓ {label}")
    for label in checks.failed:
        print(f"  ✗ {label}")
    print(f"\n通过 {len(checks.passed)} 项，失败 {len(checks.failed)} 项")
    if checks.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
