"""生活助手主循环：选择工具 → 校验参数 → 执行工具 → 组织回答。

这一层不碰模型接口，也不认识具体工具，只管流程和顺序。
每一步都留痕（StepRecord），终端上能直接看到两件事：
"为什么选了这个工具""参数为什么被拦下来"。

参数的流转顺序是刻意安排的：
1. 先校验——不合法就不执行，工具不会拿到脏数据；
2. 能安全修复的先修一次（数字字符串、枚举大小写）；
3. 修不好的把错误原样回给规划器，由它决定是让模型重调还是回问用户。
"""

from __future__ import annotations

from typing import Callable

from .models import FieldError, StepRecord, ToolCall, ToolError, ToolResult, Turn
from .tools import ToolRegistry
from .validation import (
    apply_defaults,
    coerce_arguments,
    format_errors,
    validate_arguments,
)

Reporter = Callable[[StepRecord], None]

MAX_ROUNDS = 6
DEFAULT_REPLY = "我没找到能处理这个问题的工具。可以帮你查天气、算表达式、查时间。"


class LifeAssistant:
    """生活助手 Agent。"""

    def __init__(
        self,
        planner,
        registry: ToolRegistry,
        reporter: Reporter | None = None,
    ) -> None:
        self.planner = planner
        self.registry = registry
        self.reporter = reporter

    def handle(self, user_text: str) -> Turn:
        """处理一轮用户输入，返回回答和全过程留痕。"""

        steps: list[StepRecord] = []
        turn = self.planner.plan(user_text)

        for _ in range(MAX_ROUNDS):
            if turn.calls:
                results = self._run_calls(turn.calls, steps)
                turn = self.planner.continue_plan(results)
                continue

            if turn.clarification:
                self._record(
                    steps,
                    StepRecord("追问", "信息不足", ok=True, detail=turn.clarification),
                )
                return Turn(
                    reply=turn.clarification,
                    steps=steps,
                    clarification=turn.clarification,
                )

            reply = turn.reply or DEFAULT_REPLY
            return Turn(reply=reply, steps=steps)

        reply = turn.reply or DEFAULT_REPLY
        self._record(steps, StepRecord("收尾", "达到最大轮数", ok=False, detail=reply))
        return Turn(reply=reply, steps=steps)

    # ------------------------------------------------------------
    # 工具调用：校验 → 修复 → 执行
    # ------------------------------------------------------------

    def _run_calls(self, calls: list[ToolCall], steps: list[StepRecord]) -> list[ToolResult]:
        """依次处理一组工具调用，返回对应的执行结果。"""

        results: list[ToolResult] = []

        for call in calls:
            tool = self.registry.get(call.name)

            if tool is None:
                # 工具选择错误：模型调了一个不存在的工具
                available = self.registry.describe()
                self._record(
                    steps,
                    StepRecord(
                        "选工具",
                        call.name,
                        ok=False,
                        detail=f"工具不存在，可用工具：{available}",
                    ),
                )
                results.append(
                    ToolResult(
                        name=call.name,
                        ok=False,
                        content=f"工具不存在：{call.name}。可用工具：{available}。",
                        call=call,
                    )
                )
                continue

            self._record(
                steps,
                StepRecord("选工具", call.render(), ok=True, detail=tool.description),
            )

            if call.parse_error:
                detail = call.parse_error
                self._record(
                    steps,
                    StepRecord("校验参数", call.name, ok=False, detail=detail),
                )
                results.append(
                    ToolResult(
                        name=call.name,
                        ok=False,
                        content=(
                            f"参数解析失败：{detail}。"
                            f"请重新调用 {call.name}，arguments 必须是一个 JSON 对象。"
                        ),
                        call=call,
                    )
                )
                continue

            arguments, errors = self._validated_arguments(tool, call, steps)

            if arguments is None:
                results.append(
                    ToolResult(
                        name=call.name,
                        ok=False,
                        content=(
                            f"参数校验失败：\n{format_errors(errors)}\n"
                            f"请修正后重新调用 {call.name}。"
                        ),
                        errors=errors,
                        call=call,
                    )
                )
                continue

            payload, error_text = self._execute(tool, call, arguments, steps)

            if payload is None:
                results.append(
                    ToolResult(
                        name=call.name,
                        ok=False,
                        content=f"工具执行失败：{error_text}",
                        call=call,
                    )
                )
                continue

            results.append(
                ToolResult(
                    name=call.name,
                    ok=True,
                    content=payload.text,
                    data=payload.data,
                    call=call,
                )
            )

        return results

    def _validated_arguments(
        self,
        tool,
        call: ToolCall,
        steps: list[StepRecord],
    ) -> tuple[dict | None, list[FieldError]]:
        """校验参数，能修就修一次。

        返回 (可用的参数, 错误列表)；参数为 None 表示校验没通过。
        """

        errors = validate_arguments(tool.parameters, call.arguments)

        if not errors:
            self._record(
                steps,
                StepRecord("校验参数", call.name, ok=True, detail="通过"),
            )
            return apply_defaults(tool.parameters, call.arguments), []

        fixed, remaining = coerce_arguments(tool.parameters, call.arguments)

        if not remaining:
            detail = f"自动修复：{format_errors(errors)} → {call.render()}"
            call.arguments = fixed
            self._record(
                steps,
                StepRecord("校验参数", call.name, ok=True, detail=detail),
            )
            return apply_defaults(tool.parameters, call.arguments), []

        self._record(
            steps,
            StepRecord("校验参数", call.name, ok=False, detail=format_errors(remaining)),
        )
        call.arguments = fixed
        return None, remaining

    def _execute(self, tool, call: ToolCall, arguments: dict, steps: list[StepRecord]):
        """执行工具，失败翻译成结果而不是异常。"""

        try:
            payload = tool.handler(arguments)

        except ToolError as exc:
            self._record(
                steps,
                StepRecord("执行工具", call.name, ok=False, detail=str(exc)),
            )
            return None, str(exc)

        except Exception as exc:  # 兜底：工具内部的意外错误不该打断整轮对话
            detail = f"{type(exc).__name__}: {exc}"
            self._record(
                steps,
                StepRecord("执行工具", call.name, ok=False, detail=detail),
            )
            return None, detail

        self._record(
            steps,
            StepRecord("执行工具", call.name, ok=True, detail=payload.text),
        )
        return payload, ""

    # ------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------

    def _record(self, steps: list[StepRecord], step: StepRecord) -> None:
        """登记一条留痕，并通知监听者。"""

        steps.append(step)

        if self.reporter is not None:
            self.reporter(step)
