"""历史持久化：append-only 的 JSONL 事件流 + 一个会话元数据文件。

为什么用 JSONL 而不是单个 JSON：
- 追加写：进程被 kill 也不会丢掉已经写入的历史
- 逐行可读：可以直接 cat / grep 查看
- 便于流式处理

事件类型：
- session_start  会话创建
- message       一条消息
- compaction    一次上下文压缩（摘要 + 被折叠的消息 id）
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .messages import Message, now_iso

SESSION_SUFFIX = ".jsonl"
META_SUFFIX = ".meta.json"

DEFAULT_SESSION_DIR = Path.home() / ".chat-cli" / "sessions"


@dataclass
class SessionMeta:
    """会话元信息：会话级，不随每条消息变化。"""

    id: str
    title: str = ""
    model: str = ""
    system_prompt: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    message_count: int = 0
    total_tokens: int = 0
    compactions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "compactions": self.compactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt", ""),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            message_count=int(data.get("message_count") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            compactions=int(data.get("compactions") or 0),
        )


def new_session_id() -> str:
    """时间戳 + 随机后缀，天然按时间排序。"""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return f"{stamp}-{uuid.uuid4().hex[:4]}"


class SessionStore:
    """会话文件的读写。"""

    def __init__(self, root: Path | str = DEFAULT_SESSION_DIR) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. 路径
    # ------------------------------------------------------------

    def session_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}{SESSION_SUFFIX}"

    def meta_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}{META_SUFFIX}"

    def exists(self, session_id: str) -> bool:
        return self.session_path(session_id).exists()

    # ------------------------------------------------------------
    # 2. 追加写事件
    # ------------------------------------------------------------

    def _append_event(self, session_id: str, event: dict[str, Any]) -> None:
        path = self.session_path(session_id)

        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def append_message(self, session_id: str, message: Message) -> None:
        self._append_event(
            session_id,
            {"type": "message", "at": now_iso(), "message": message.to_dict()},
        )

    def append_compaction(
        self, session_id: str, summary: Message, compacted_ids: list[str]
    ) -> None:
        self._append_event(
            session_id,
            {
                "type": "compaction",
                "at": now_iso(),
                "summary": summary.to_dict(),
                "compacted_ids": compacted_ids,
                "compacted_count": len(compacted_ids),
            },
        )

    def append_session_start(
        self, session_id: str, model: str, system_prompt: str
    ) -> None:
        self._append_event(
            session_id,
            {
                "type": "session_start",
                "at": now_iso(),
                "model": model,
                "system_prompt": system_prompt,
            },
        )

    # ------------------------------------------------------------
    # 3. 读事件
    # ------------------------------------------------------------

    def read_events(self, session_id: str) -> Iterator[dict[str, Any]]:
        """逐行读取事件流，跳过写坏的行（崩溃恢复时可能留下半行）。"""

        path = self.session_path(session_id)

        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def load_messages(self, session_id: str) -> list[Message]:
        """读出全部原始消息（不含摘要）。"""

        messages: list[Message] = []

        for event in self.read_events(session_id):
            if event.get("type") != "message":
                continue

            payload = event.get("message") or {}

            try:
                messages.append(Message.from_dict(payload))
            except (KeyError, ValueError):
                continue

        return messages

    def load_context_state(
        self, session_id: str
    ) -> tuple[list[Message], Message | None]:
        """回放事件流，还原"工作记忆"。

        返回 (近期消息, 摘要消息)。

        规则：每遇到一次 compaction 事件，就把摘要替换成最新的，
        并把当时被折叠掉的消息从近期区移除。
        """

        recent: list[Message] = []
        summary: Message | None = None

        for event in self.read_events(session_id):
            kind = event.get("type")

            if kind == "message":
                payload = event.get("message") or {}

                try:
                    recent.append(Message.from_dict(payload))
                except (KeyError, ValueError):
                    continue

            elif kind == "compaction":
                payload = event.get("summary") or {}

                try:
                    summary = Message.from_dict(payload)
                except (KeyError, ValueError):
                    summary = None

                compacted = set(event.get("compacted_ids") or [])
                recent = [
                    message for message in recent if message.id not in compacted
                ]

        return recent, summary

    # ------------------------------------------------------------
    # 4. 会话元信息
    # ------------------------------------------------------------

    def save_meta(self, meta: SessionMeta) -> None:
        path = self.meta_path(meta.id)
        path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_meta(self, session_id: str) -> SessionMeta | None:
        path = self.meta_path(session_id)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        return SessionMeta.from_dict(data)

    def list_sessions(self) -> list[SessionMeta]:
        """列出所有会话，按更新时间倒序。"""

        sessions: list[SessionMeta] = []

        for path in sorted(self.root.glob(f"*{META_SUFFIX}")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            sessions.append(SessionMeta.from_dict(data))

        sessions.sort(key=lambda item: item.updated_at, reverse=True)

        return sessions

    def update_meta(
        self,
        session_id: str,
        *,
        model: str | None = None,
        title: str | None = None,
        system_prompt: str | None = None,
        add_messages: int = 0,
        add_tokens: int = 0,
        add_compactions: int = 0,
    ) -> SessionMeta:
        """增量更新会话元信息并落盘。"""

        meta = self.load_meta(session_id) or SessionMeta(id=session_id)

        if model is not None:
            meta.model = model

        if title:
            meta.title = title

        if system_prompt is not None:
            meta.system_prompt = system_prompt

        meta.message_count += add_messages
        meta.total_tokens += add_tokens
        meta.compactions += add_compactions
        meta.updated_at = now_iso()

        self.save_meta(meta)

        return meta
