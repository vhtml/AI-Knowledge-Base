"""上下文管理：把一段历史压进模型能吃下的 token 预算里。

上下文窗口的层次（从固定到可丢弃）：

    1. 系统提示    —— 常驻，永远保留
    2. 历史摘要    —— 压缩态：更早的对话被折叠成一段摘要
    3. 近期原文    —— 近期态：保留最近 N 轮的原始消息
    4. 当前输入    —— 本轮用户消息，永远保留

本模块只做"决定送哪些消息"，不碰网络，方便单独阅读和测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .messages import (
    MESSAGE_OVERHEAD_TOKENS,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    Message,
    truncate_to_tokens,
)

# 压缩器：把一批旧消息 + 上一版摘要，压成一段新摘要
Summarizer = Callable[[list[Message], Message | None], str]

SUMMARY_HEADER = "【历史对话摘要】以下是被折叠的更早对话，仅供参考：\n"

DEFAULT_BUDGET_TOKENS = 4000
DEFAULT_KEEP_RECENT_TURNS = 2


def naive_summary(
    messages: list[Message], previous_summary: Message | None = None
) -> str:
    """不调用模型的降级摘要：每条消息保留开头一小段。

    用于模型不可用时兜底，至少能保住"聊过什么"的线索。
    """

    lines: list[str] = []

    if previous_summary is not None and previous_summary.content.strip():
        lines.append(previous_summary.content.strip())

    for message in messages:
        text = message.content.replace("\n", " ").strip()

        if not text:
            continue

        if len(text) > 60:
            text = text[:60] + "…"

        lines.append(f"{message.label}：{text}")

    if not lines:
        return "（暂无历史摘要）"

    return "\n".join(lines)


@dataclass
class ContextPlan:
    """一次上下文装配的结果，方便观察和调试。"""

    api_messages: list[dict[str, str]]
    used_tokens: int
    budget: int
    kept: list[Message] = field(default_factory=list)
    summary: Message | None = None
    dropped_count: int = 0
    total_tokens: int = 0

    @property
    def usage_ratio(self) -> float:
        return self.used_tokens / self.budget if self.budget else 0.0

    def render(self) -> str:
        """把装配结果渲染成可读文本，供 /context 命令使用。"""

        lines = [
            "上下文装配结果",
            f"  预算      : {self.budget} token",
            f"  已用      : {self.used_tokens} token "
            f"({self.usage_ratio * 100:.1f}%)",
            f"  工作记忆  : {self.total_tokens} token（未裁剪）",
            "",
        ]

        for index, item in enumerate(self.api_messages, start=1):
            preview = item["content"].replace("\n", " ").strip()

            if len(preview) > 40:
                preview = preview[:40] + "…"

            role = {"system": "系统", "user": "用户", "assistant": "助手"}.get(
                item["role"], item["role"]
            )
            lines.append(f"  {index}. [{role}] {preview}")

        if self.dropped_count:
            lines.append("")
            lines.append(
                f"  因预算被丢弃：{self.dropped_count} 条"
                f"（完整历史仍保留在会话文件中）"
            )

        return "\n".join(lines)


class ContextWindow:
    """工作记忆：维护"系统提示 + 摘要 + 近期消息"。

    - add()：追加一条消息（进入近期区）
    - build()：按预算装配出送给模型的消息列表
    - compact()：把旧的近期消息折叠进摘要
    """

    def __init__(
        self,
        system_prompt: str,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    ) -> None:
        self.system_prompt = system_prompt
        self.budget_tokens = budget_tokens
        self.keep_recent_turns = keep_recent_turns

        self.summary: Message | None = None
        self.messages: list[Message] = []

    # ------------------------------------------------------------
    # 1. 状态查询
    # ------------------------------------------------------------

    @property
    def system_tokens(self) -> int:
        return Message(ROLE_SYSTEM, self.system_prompt).tokens

    @property
    def summary_tokens(self) -> int:
        return self.summary.tokens if self.summary is not None else 0

    @property
    def recent_tokens(self) -> int:
        return sum(message.tokens for message in self.messages)

    @property
    def total_tokens(self) -> int:
        """系统提示 + 摘要 + 近期消息的 token 总量。"""

        return self.system_tokens + self.summary_tokens + self.recent_tokens

    def message_count(self) -> int:
        return len(self.messages)

    # ------------------------------------------------------------
    # 2. 写入
    # ------------------------------------------------------------

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def restore(self, summary: Message | None, messages: list[Message]) -> None:
        """从历史文件回放，恢复工作记忆。"""

        self.summary = summary
        self.messages = list(messages)

    # ------------------------------------------------------------
    # 3. 装配：决定这一轮送哪些消息进模型
    # ------------------------------------------------------------

    def build(self) -> ContextPlan:
        """按预算装配出送给模型的 messages。

        保留优先级（高 → 低）：
        1. 系统提示：常驻，超预算时截断。
        2. 最新一条消息（当前输入）：永远保留。
        3. 历史摘要：占满剩余预算，放不下就截断正文而不是整条丢弃。
        4. 其余近期消息：从新到旧保留，放不下的丢弃。
        """

        api_messages: list[dict[str, str]] = []
        kept: list[Message] = []
        dropped = 0

        # 3.1 系统提示
        system = Message(ROLE_SYSTEM, self.system_prompt)

        if system.tokens > self.budget_tokens:
            room = max(0, self.budget_tokens - MESSAGE_OVERHEAD_TOKENS)
            system = Message(
                ROLE_SYSTEM,
                truncate_to_tokens(self.system_prompt, room),
                meta={"truncated": True},
            )

        api_messages.append(system.to_api())
        used = system.tokens

        # 3.2 预留最新一条消息的位置
        newest = self.messages[-1] if self.messages else None

        if newest is not None:
            used += newest.tokens

        # 3.3 历史摘要：用掉剩余预算，放不下就截断正文
        summary_included: Message | None = None

        if self.summary is not None:
            room = self.budget_tokens - used
            wrapped = self._wrap_summary(self.summary, room)

            if wrapped is not None:
                api_messages.append(wrapped.to_api())
                used += wrapped.tokens
                summary_included = self.summary
            else:
                dropped += 1

        # 3.4 其余近期消息：从新到旧装
        ordered: list[Message] = []

        for message in reversed(self.messages):
            if message is newest:
                ordered.append(message)
                continue

            if used + message.tokens <= self.budget_tokens:
                ordered.append(message)
                used += message.tokens
            else:
                dropped += 1

        ordered.reverse()
        kept.extend(ordered)
        api_messages.extend(message.to_api() for message in ordered)

        return ContextPlan(
            api_messages=api_messages,
            used_tokens=used,
            budget=self.budget_tokens,
            kept=kept,
            summary=summary_included,
            dropped_count=dropped,
            total_tokens=self.total_tokens,
        )

    @staticmethod
    def _wrap_summary(summary: Message, room: int) -> Message | None:
        """把摘要包成一条 system 消息，并裁剪到可用预算内。

        摘要放不下时截断正文（保留"这是摘要"的提示头），
        因为半截摘要也比完全没有摘要有用。
        """

        if room <= 0:
            return None

        header_tokens = Message(ROLE_SYSTEM, SUMMARY_HEADER).tokens
        body_room = room - header_tokens

        if body_room <= 0:
            return None

        body = summary.content

        if Message(ROLE_SYSTEM, body).tokens > body_room:
            body = truncate_to_tokens(body, body_room) + "…"

        return Message(
            ROLE_SYSTEM,
            SUMMARY_HEADER + body,
            meta={"synthetic": True},
        )

    # ------------------------------------------------------------
    # 4. 压缩：把旧的近期消息折叠进摘要
    # ------------------------------------------------------------

    def split_recent(
        self, messages: list[Message] | None = None
    ) -> tuple[list[Message], list[Message]]:
        """按"轮"切分消息。

        一轮 = 一条用户消息 + 后续的助手消息。
        返回 (保留的近期消息, 需要压缩的旧消息)。
        """

        source = list(self.messages if messages is None else messages)

        if self.keep_recent_turns <= 0:
            return [], source

        user_positions = [
            index
            for index, message in enumerate(source)
            if message.role == ROLE_USER
        ]

        if len(user_positions) <= self.keep_recent_turns:
            return source, []

        start = user_positions[-self.keep_recent_turns]

        return source[start:], source[:start]

    def compact(self, summarizer: Summarizer) -> tuple[int, Message | None]:
        """把超出保留窗口的旧消息折叠进摘要。

        返回 (被压缩的消息条数, 新的摘要消息)。
        """

        recent, old = self.split_recent()

        if not old:
            return 0, None

        summary_text = summarizer(old, self.summary)

        if not summary_text or not summary_text.strip():
            return 0, None

        new_summary = Message(
            ROLE_SYSTEM,
            summary_text.strip(),
            meta={
                "synthetic": True,
                "kind": "summary",
                "compacted_count": len(old),
                "compacted_ids": [message.id for message in old],
            },
        )

        self.summary = new_summary
        self.messages = recent

        return len(old), new_summary

    def reset(self) -> None:
        """清空工作记忆（磁盘上的历史不受影响）。"""

        self.summary = None
        self.messages = []
