"""消息结构：一条消息到底由什么组成。

一条消息 = role（角色） + content（内容） + 元数据（本地信息）

关键约定：
1. 只有 role 和 content 会进入模型上下文；元数据留在本地，用于展示、统计、断点续聊。
2. 消息是"事实记录"，落盘后可以被完整还原（to_dict / from_dict），不做二次加工。
3. token 估算是本地启发式，只服务于"上下文预算"控制，不等于服务端计费。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

VALID_ROLES = frozenset({ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT})

# 协议层的固定开销（role 标记、分隔符等），每条消息统一加上
MESSAGE_OVERHEAD_TOKENS = 4

ROLE_LABELS = {
    ROLE_SYSTEM: "系统",
    ROLE_USER: "用户",
    ROLE_ASSISTANT: "助手",
}


def now_iso() -> str:
    """本地时间，秒级精度，够用且可读。"""

    return time.strftime("%Y-%m-%d %H:%M:%S")


def _is_cjk(ch: str) -> bool:
    """判断是否为需要按 1 字 ≈ 1 token 计费的字符。"""

    code = ord(ch)

    return (
        0x4E00 <= code <= 0x9FFF  # 基本汉字
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0xF900 <= code <= 0xFAFF  # 兼容汉字
        or 0x3040 <= code <= 0x30FF  # 日文假名
        or 0xAC00 <= code <= 0xD7AF  # 韩文
    )


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数。

    没有 tokenizer 时的启发式估算：
    - CJK 字符按 1 字 ≈ 1 token（偏保守，宁可高估也不要撑破预算）
    - 其余字符按 4 字符 ≈ 1 token
    真实用量以服务端返回的 usage 为准。
    """

    if not text:
        return 0

    cjk = 0
    other = 0

    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1

    return cjk + (other + 3) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按估算 token 数截断文本，尽量在语义完整处收尾。"""

    if max_tokens <= 0:
        return ""

    kept: list[str] = []
    used = 0.0

    for ch in text:
        cost = 1.0 if _is_cjk(ch) else 0.25

        if used + cost > max_tokens:
            break

        used += cost
        kept.append(ch)

    return "".join(kept)


@dataclass
class Message:
    """一条消息。

    role/content 是"给模型看"的；
    id/created_at/tokens/meta 是"给自己看"的，不进入上下文。
    """

    role: str
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=now_iso)
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"未知角色 {self.role!r}，合法角色：{sorted(VALID_ROLES)}"
            )

        self.content = self.content or ""

        if self.tokens <= 0:
            self.tokens = estimate_tokens(self.content) + MESSAGE_OVERHEAD_TOKENS

    @property
    def is_synthetic(self) -> bool:
        """是否为本地生成的派生消息（摘要等），不是原始对话内容。"""

        return bool(self.meta.get("synthetic"))

    @property
    def label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    def to_api(self) -> dict[str, str]:
        """转成模型接口需要的最小结构：只有 role 和 content。

        这是"上下文"和"历史"的分界线：历史可以有任意元数据，
        送进模型的只有这两个字段。
        """

        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典，用于落盘。"""

        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "tokens": self.tokens,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """从字典还原消息，用于历史回放。"""

        return cls(
            role=data["role"],
            content=data.get("content", ""),
            id=data.get("id") or uuid.uuid4().hex[:12],
            created_at=data.get("created_at") or now_iso(),
            tokens=int(data.get("tokens") or 0),
            meta=dict(data.get("meta") or {}),
        )

    def summary_line(self) -> str:
        """给 /history 用的紧凑展示行。"""

        flag = "·摘要" if self.is_synthetic else ""
        preview = self.content.replace("\n", " ").strip()

        if len(preview) > 50:
            preview = preview[:50] + "…"

        return (
            f"{self.created_at}  {self.label}{flag}  "
            f"{self.tokens:>5}t  {preview}"
        )
