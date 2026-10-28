"""跨模块流转的数据结构。

这里只放"事实记录"：谁产生、谁消费一眼能看到，不做任何业务判断。
工具参数相关的校验错误类型也放在这里，避免 tools / providers / validation
互相 import 形成环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """工具执行阶段的可预期失败：参数没问题，但活干不成。

    和"参数校验失败"是两回事——校验失败发生在执行之前，
    ToolError 发生在执行之中，消息最终会回到模型或用户那里。
    """


@dataclass
class FieldError:
    """一条参数错误：字段名 + 错误码 + 中文说明 + 修复建议。"""

    field: str
    code: str
    message: str
    hint: str = ""

    def render(self) -> str:
        hint = f"（建议：{self.hint}）" if self.hint else ""
        return f"{self.field}: {self.message}{hint}"


@dataclass
class ToolCall:
    """模型或路由规则提出的一次工具调用。

    raw_arguments 保留模型给的原始 JSON 字符串：解析失败时，
    它是排查"模型到底吐了什么"的唯一线索。
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_arguments: str = ""
    parse_error: str = ""

    def render(self) -> str:
        if not self.arguments:
            return f"{self.name}()"

        args = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.name}({args})"


@dataclass
class ToolResult:
    """一次工具执行的结果，成功的和失败的都走同一个结构。"""

    name: str
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[FieldError] = field(default_factory=list)
    call: ToolCall | None = None

    def render(self) -> str:
        mark = "✓" if self.ok else "✗"
        return f"{mark} {self.name}: {self.content}"


@dataclass
class ToolPayload:
    """工具 handler 的返回值：给人看的正文 + 给程序用的结构化数据。"""

    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerTurn:
    """一轮规划的结果：要么有工具调用，要么有最终回答。"""

    calls: list[ToolCall] = field(default_factory=list)
    reply: str = ""
    clarification: str = ""


@dataclass
class StepRecord:
    """一次决策或执行的留痕，用于终端展示和事后排查。"""

    phase: str
    title: str
    ok: bool = True
    detail: str = ""


@dataclass
class Turn:
    """一轮完整交互的结果。"""

    reply: str
    steps: list[StepRecord] = field(default_factory=list)
    clarification: str = ""
