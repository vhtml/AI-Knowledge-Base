"""交互层：把"上下文装配 + 流式调用 + 历史落盘"串成一个多轮对话。

两个类：
- Conversation  负责一轮对话的处理流程（不含终端 I/O）
- ChatSession   负责终端交互（输入、命令、流式打印）
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterator

from .context import ContextPlan, ContextWindow
from .llm import Delta, make_summarizer
from .messages import ROLE_ASSISTANT, ROLE_USER, Message
from .storage import SessionStore

BANNER = """┌──────────────────────────────────────────────┐
│  命令行聊天助手  ·  多轮对话 / 流式输出 / 历史存档 │
└──────────────────────────────────────────────┘
命令：/help 帮助  /history 历史  /context 上下文  /compact 压缩
      /sessions 会话列表  /clear 清空上下文  /exit 退出
"""

HELP_TEXT = """可用命令：
  /help              显示这份帮助
  /history [n]       查看本会话最近 n 条消息（默认 10）
  /context           查看本轮送给模型的上下文构成与 token 用量
  /compact           手动触发一次上下文压缩（把旧消息折叠成摘要）
  /sessions          列出本机保存的所有会话
  /clear             清空当前上下文（磁盘上的历史保留）
  /exit              退出（也可直接输入 exit / quit）
"""


class Style:
    """极简 ANSI 着色，非终端环境自动降级为纯文本。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text

        return f"\033[{code}m{text}\033[0m"

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)


class Conversation:
    """多轮对话的编排：接收输入 → 装配上下文 → 流式调用 → 落盘。"""

    def __init__(
        self,
        window: ContextWindow,
        client,
        store: SessionStore,
        session_id: str,
        use_stream: bool = True,
        compact_ratio: float = 0.75,
    ) -> None:
        self.window = window
        self.client = client
        self.store = store
        self.session_id = session_id
        self.use_stream = use_stream
        self.compact_ratio = compact_ratio

        self.summarizer = make_summarizer(client)
        self.last_plan: ContextPlan | None = None
        self.last_reply = ""
        self.last_compaction: tuple[int, Message | None] | None = None
        meta = self.store.load_meta(session_id)
        self._title_set = bool(meta and meta.title)

    # ------------------------------------------------------------
    # 1. 一轮对话
    # ------------------------------------------------------------

    def ask(self, user_text: str) -> Iterator[Delta]:
        """处理一轮对话，逐段产出模型输出。

        顺序很重要：
        1. 先把用户消息落盘——即使请求失败，这句话也已经进入历史
        2. 装配上下文（此时历史里已经有本轮输入）
        3. 流式请求，边收边打印
        4. 助手消息落盘
        5. 必要时压缩上下文
        """

        user_message = Message(ROLE_USER, user_text)

        self.last_compaction = None
        self.window.add(user_message)
        self.store.append_message(self.session_id, user_message)
        self.store.update_meta(
            self.session_id,
            add_messages=1,
            add_tokens=user_message.tokens,
        )

        if not self._title_set:
            self._title_set = True
            self.store.update_meta(
                self.session_id,
                title=user_text.strip().replace("\n", " ")[:24] or "未命名会话",
            )

        plan = self.window.build()
        self.last_plan = plan

        chunks: list[str] = []
        finish_reason = None

        if self.use_stream:
            for delta in self.client.stream(plan.api_messages):
                if delta.finish_reason:
                    finish_reason = delta.finish_reason

                if delta.text:
                    chunks.append(delta.text)
                    yield delta
        else:
            text = self.client.complete(plan.api_messages)
            chunks.append(text)
            finish_reason = "stop"
            yield Delta(text=text, finish_reason=finish_reason)

        reply = "".join(chunks)

        assistant_message = Message(
            ROLE_ASSISTANT,
            reply,
            meta={
                "model": getattr(self.client, "model", ""),
                "finish_reason": finish_reason,
            },
        )

        self.window.add(assistant_message)
        self.store.append_message(self.session_id, assistant_message)
        self.store.update_meta(
            self.session_id,
            add_messages=1,
            add_tokens=assistant_message.tokens,
        )

        self.last_reply = reply
        self._maybe_compact()

    # ------------------------------------------------------------
    # 2. 上下文压缩
    # ------------------------------------------------------------

    def _maybe_compact(self) -> None:
        """上下文用量超过阈值时，自动把旧消息折叠成摘要。"""

        threshold = self.window.budget_tokens * self.compact_ratio

        if self.window.total_tokens <= threshold:
            return

        self.compact()

    def compact(self) -> tuple[int, Message | None]:
        """执行一次压缩：旧消息 → 摘要。

        返回 (被压缩的消息条数, 新的摘要消息)。
        """

        count, summary = self.window.compact(self.summarizer)

        if not count or summary is None:
            return 0, None

        compacted_ids = summary.meta.get("compacted_ids") or []
        self.store.append_compaction(self.session_id, summary, compacted_ids)
        self.store.update_meta(self.session_id, add_compactions=1)

        self.last_compaction = (count, summary)

        return count, summary


class ChatSession:
    """命令行交互层。"""

    def __init__(
        self,
        conversation: Conversation,
        store: SessionStore,
        session_id: str,
        model: str = "",
        mock: bool = False,
    ) -> None:
        self.conversation = conversation
        self.store = store
        self.session_id = session_id
        self.model = model
        self.mock = mock
        self.style = Style(sys.stdout.isatty() and os.getenv("NO_COLOR") is None)

    # ------------------------------------------------------------
    # 1. 主循环
    # ------------------------------------------------------------

    def run(self) -> None:
        print(BANNER)

        meta = self.store.load_meta(self.session_id)

        if meta is not None:
            print(
                self.style.dim(
                    f"会话 {self.session_id} · 模型 {self.model}"
                    f" · 已存 {meta.message_count} 条消息\n"
                )
            )
        else:
            print(self.style.dim(f"会话 {self.session_id} · 模型 {self.model}\n"))

        if self.mock:
            print(self.style.yellow("离线演示模式：不联网，用假客户端跑通全流程。\n"))

        while True:
            try:
                line = input(self.style.cyan("你 › "))
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break

            line = line.strip()

            if not line:
                continue

            if line.lower() in {"exit", "quit"}:
                print("再见。")
                break

            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue

            self.handle_turn(line)

    # ------------------------------------------------------------
    # 2. 一轮对话的终端呈现
    # ------------------------------------------------------------

    def handle_turn(self, user_text: str) -> None:
        started = time.perf_counter()
        first_delta_at: float | None = None
        received = 0

        print(self.style.green("助手 › "), end="", flush=True)

        try:
            for delta in self.conversation.ask(user_text):
                if not delta.text:
                    continue

                if first_delta_at is None:
                    first_delta_at = time.perf_counter()

                received += len(delta.text)
                sys.stdout.write(delta.text)
                sys.stdout.flush()

        except Exception as exc:
            print(f"\n{self.style.yellow('[错误]')} {type(exc).__name__}: {exc}")
            return

        total_elapsed = time.perf_counter() - started
        ttft = (first_delta_at - started) if first_delta_at else total_elapsed

        print()

        plan = self.conversation.last_plan

        stats = [
            f"首字 {ttft:.2f}s",
            f"总耗时 {total_elapsed:.2f}s",
            f"输出 {received} 字",
        ]

        print(self.style.dim("  ↳ " + " · ".join(stats)))

        if plan is not None:
            # 两个数字含义不同：
            #   本轮发送 = 这次请求实际带进模型的内容
            #   下轮携带 = 按预算重新装配后，下一轮会带上的内容
            next_plan = self.conversation.window.build()
            budget = self.conversation.window.budget_tokens
            ratio = next_plan.used_tokens / budget * 100 if budget else 0.0

            detail = [
                f"本轮发送 {plan.used_tokens}t",
                f"下轮携带 ~{next_plan.used_tokens}/{budget}t ({ratio:.0f}%)",
            ]

            compaction = self.conversation.last_compaction

            if compaction:
                count, summary = compaction
                detail.append(f"已压缩 {count} 条 → 摘要 {summary.tokens}t")

            print(self.style.dim("    " + " · ".join(detail)))

    # ------------------------------------------------------------
    # 3. 斜杠命令
    # ------------------------------------------------------------

    def handle_command(self, command: str) -> bool:
        """处理斜杠命令。返回 False 表示要退出主循环。"""

        parts = command.split()
        name = parts[0].lower()

        if name in {"/exit", "/quit"}:
            print("再见。")
            return False

        if name == "/help":
            print(HELP_TEXT)

        elif name == "/history":
            limit = 10

            if len(parts) > 1 and parts[1].isdigit():
                limit = int(parts[1])

            self.show_history(limit)

        elif name == "/context":
            plan = self.conversation.window.build()
            print(plan.render())

        elif name == "/compact":
            count, summary = self.conversation.compact()

            if count:
                print(
                    self.style.dim(
                        f"已压缩 {count} 条旧消息，摘要 {summary.tokens} token。"
                    )
                )
            else:
                print(self.style.dim("没有需要压缩的旧消息。"))

        elif name == "/sessions":
            self.show_sessions()

        elif name == "/clear":
            self.store.append_clear(self.session_id)
            self.conversation.window.reset()
            print(self.style.dim("当前上下文已清空（磁盘历史保留）。"))

        else:
            print(self.style.yellow(f"未知命令 {name}，输入 /help 查看用法。"))

        return True

    # ------------------------------------------------------------
    # 4. 展示
    # ------------------------------------------------------------

    def show_history(self, limit: int) -> None:
        """展示本会话的完整历史（从磁盘读，不受上下文压缩影响）。"""

        messages = self.store.load_messages(self.session_id)

        if not messages:
            print(self.style.dim("（本会话还没有消息）"))
            return

        print(self.style.bold(f"本会话共 {len(messages)} 条消息，显示最近 {limit} 条："))

        for message in messages[-limit:]:
            print("  " + message.summary_line())

    def show_sessions(self) -> None:
        """列出本机保存的所有会话。"""

        sessions = self.store.list_sessions()

        if not sessions:
            print(self.style.dim("（还没有保存任何会话）"))
            return

        print(self.style.bold(f"共 {len(sessions)} 个会话："))

        for meta in sessions:
            marker = "*" if meta.id == self.session_id else " "
            print(
                f" {marker} {meta.id}  {meta.created_at}  "
                f"{meta.message_count:>3} 条  {meta.total_tokens:>6}t  "
                f"{meta.title or '未命名'}"
            )
