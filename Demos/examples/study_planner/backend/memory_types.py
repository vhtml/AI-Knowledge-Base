"""固定操作、记忆候选和上下文边界，避免接口间传递任意状态。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.code, self.status = code, status


@dataclass
class MemoryCandidate:
    content: str
    category: str = "note"
    status: str = "pending"
    kind: str = "experience"
    source_id: str = ""
    excerpt: str = ""
    scope: str = "default"
    value: int | None = None


@dataclass
class OperationResult:
    turn_id: str
    status: str
    state_revision: int = 0
    facts: str = ""
    reply: str = ""
    error_code: str = ""
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextBundle:
    messages: list[dict[str, str]]
    token_count: int
    source_ids: list[str] = field(default_factory=list)
    degraded: bool = False
    clipping_reasons: list[str] = field(default_factory=list)
