"""框架无关的最小 Agent 运行时。

这个模块解决什么问题：第 2 章要手写一个 ReAct 循环，但后面 15 章还要在这条
循环上继续加东西——上下文压缩（第 5 章）、记忆（第 6 章）、规划（第 7 章）、
重试与幂等（第 10 章）、追踪（第 11 章）、审批（第 14 章）。

如果每章各写各的，这些加法就没有共同的落点。所以这里给出一个**参考实现**：
一个不依赖任何框架、带完整护栏的最小运行时。各章有两种用法：

1. 直接继承或复制 :class:`Agent`，在它基础上加本章的能力；
2. 需要做对照实验时，另写一版更笨的实现，跑出来比一比。

设计上刻意保留的"笨"：

- 不做异步——同步循环最容易看清控制流，异步是第 16 章的话题；
- 不做并行工具调用——虽然模型一次能返回多个 tool_calls，
  但并行意味着要处理"部分失败"，那是第 10 章的内容；
- 错误不抛给调用方，而是**回填给模型**，让模型有机会纠正。
  这是 Agent 和普通函数最本质的差别之一：模型必须能看见自己犯的错。

后面每一章都可以把这三处"笨"换掉，换的过程就是那一章的内容。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .labkit import ChatModel, Message, Tool, ToolCall, tool_result

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Step",
]


@dataclass
class AgentConfig:
    """运行时配置。

    每个字段都对应一个真实的失败模式，不是可有可无的旋钮：

    - ``max_iterations``：模型可能在两个工具之间反复横跳，没有上限就是烧钱；
    - ``max_tool_output_chars``：工具返回一整页网页会挤爆上下文窗口，
      而且模型真正需要的往往只是前几行；
    - ``tool_error_feedback``：工具出错时回给模型的措辞，
      决定了模型是能自我纠正还是彻底卡死。
    """

    system_prompt: str = ""
    max_iterations: int = 8
    max_tool_output_chars: int = 4000
    tool_error_feedback: str = "工具执行失败：{error}。请根据错误信息调整参数后重试，或换用其他工具。"


@dataclass
class Step:
    """轨迹里的一步。

    追踪的第一课：Agent 的过程不是"一个输入一个输出"，
    而是"一串可回放的步骤"。第 11 章整章都在讨论怎么把这串步骤记好。
    """

    index: int
    kind: Literal["model", "tool"]
    label: str
    detail: str = ""


@dataclass
class AgentResult:
    """一次运行的完整结果。"""

    messages: list[Message] = field(default_factory=list)
    final_answer: str = ""
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = "completed"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """整条轨迹里出现过的所有工具调用。"""
        return [c for m in self.messages for c in m.tool_calls]


class Agent:
    """最小 Agent：模型 + 工具 + 一个循环。

    这个类刻意很短，因为它暴露的是 Agent 的**本质结构**，
    不是生产系统的全部细节：

        while 没结束:
            问模型要下一步
            如果模型要调工具:
                执行工具，把结果回填
            否则:
                结束

    后面 15 章做的所有事，都是在这个骨架上加东西或者换掉其中一环。
    """

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[Tool] = (),
        config: AgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.config = config or AgentConfig()

    # ------------------------------------------------------------ 工具执行

    def _run_tool(self, call: ToolCall) -> tuple[str, bool]:
        """执行一次工具调用，返回 ``(回填给模型的内容, 是否成功)``。

        三类失败要分开处理，它们的后果完全不同：

        1. **工具不存在**——通常是模型幻觉了工具名。回一句"没有这个工具"，
           模型会自己改用别的工具。
        2. **参数不合法**——模型传错了参数。回一句具体错在哪，
           模型多半能改对。
        3. **工具内部出错**——业务系统挂了。这种错误模型无法修复，
           回给它只会让它反复重试，所以要带上"换别的路"的暗示。

        关键区别：第 2 类回原文，第 3 类必须改写措辞。
        把业务故障原样回给模型，是最常见的"Agent 卡死"原因。
        """
        target = self.tools.get(call.name)
        if target is None:
            known = ", ".join(sorted(self.tools)) or "（无）"
            return (
                f"错误：没有名为 {call.name} 的工具。可用工具：{known}。",
                False,
            )

        try:
            result = target.invoke(call.arguments)
        except TypeError as exc:
            # 参数名或类型不对——这是模型能自我修复的错误
            return (f"参数错误：{exc}。请检查参数名与类型。", False)
        except Exception as exc:  # noqa: BLE001 - 业务故障要转成模型能读懂的话
            message = self.config.tool_error_feedback.format(error=f"{type(exc).__name__}: {exc}")
            return (message, False)

        if len(result) > self.config.max_tool_output_chars:
            result = (
                result[: self.config.max_tool_output_chars]
                + f"\n…（已截断，原始长度 {len(result)} 字符）"
            )
        return (result, True)

    # ------------------------------------------------------------ 主循环

    def run(
        self,
        user_input: str,
        history: Sequence[Message] = (),
    ) -> AgentResult:
        """跑一次完整的 Agent 循环。

        ``history`` 是之前几轮的用户/助手消息，用于多轮对话；
        第 5 章会说明为什么"把全部历史原样塞进去"是错的。
        """
        messages: list[Message] = []
        if self.config.system_prompt:
            messages.append(Message(role="system", content=self.config.system_prompt))
        messages.extend(history)
        messages.append(Message(role="user", content=user_input))

        steps: list[Step] = []
        usage: dict[str, int] = {}
        tool_list = list(self.tools.values())

        for iteration in range(self.config.max_iterations):
            reply = self.model.complete(messages, tools=tool_list or None)

            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + value

            steps.append(
                Step(
                    index=iteration,
                    kind="model",
                    label="模型",
                    detail=reply.content[:200],
                )
            )
            messages.append(reply.to_message())

            if not reply.tool_calls:
                return AgentResult(
                    messages=messages,
                    final_answer=reply.content,
                    steps=steps,
                    stop_reason="completed",
                    usage=usage,
                )

            for call in reply.tool_calls:
                content, ok = self._run_tool(call)
                steps.append(
                    Step(
                        index=iteration,
                        kind="tool",
                        label=call.name,
                        detail=("✓ " if ok else "✗ ") + content[:200],
                    )
                )
                messages.append(tool_result(content, tool_call_id=call.id, name=call.name))

        # 走到这里说明模型一直在要工具，没给出最终答案
        return AgentResult(
            messages=messages,
            final_answer="",
            steps=steps,
            stop_reason="max_iterations",
            usage=usage,
        )

    # ------------------------------------------------------------ 辅助

    def tool_schemas(self) -> list[dict[str, Any]]:
        """导出工具的 JSON Schema——第 3 章讲工具设计时用得上。"""
        return [t.to_api() for t in self.tools.values()]
