"""学习规划师的数据模型。

这个模块只回答"事实长什么样"，不做任何业务判断。
业务规则分散在 memory.py（怎么记）、planner.py（怎么排）、review.py（怎么复盘）里。

四类实体：
- Goal        学习目标，带截止日期，是所有计划的源头
- PlanItem    计划里的一条待办，挂在某个目标下，按天排布
- Review      某一天的复盘记录，是短期经验变成长期经验的入口
- MemoryItem  记忆条目，短期和长期共用同一个结构，用 scope 区分

统一用 dataclass + to_dict/from_dict，方便直接落成 JSON，
也不需要引入 pydantic 依赖。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCOPE_SHORT = "short"
SCOPE_LONG = "long"

# 记忆条目的类别，决定它在召回时的默认权重
CATEGORY_GOAL = "goal"
CATEGORY_PLAN = "plan"
CATEGORY_REVIEW = "review"
CATEGORY_WEAKNESS = "weakness"
CATEGORY_STRENGTH = "strength"
CATEGORY_PREFERENCE = "preference"
CATEGORY_MILESTONE = "milestone"
CATEGORY_SUMMARY = "summary"
CATEGORY_NOTE = "note"

CATEGORIES = (
    CATEGORY_GOAL,
    CATEGORY_PLAN,
    CATEGORY_REVIEW,
    CATEGORY_WEAKNESS,
    CATEGORY_STRENGTH,
    CATEGORY_PREFERENCE,
    CATEGORY_MILESTONE,
    CATEGORY_SUMMARY,
    CATEGORY_NOTE,
)


def now_iso() -> str:
    """当前时间的秒级 ISO 字符串。"""

    return datetime.now().isoformat(timespec="seconds")


def today_str() -> str:
    """今天日期，YYYY-MM-DD。"""

    return datetime.now().strftime("%Y-%m-%d")


def stable_id(prefix: str, *parts: str) -> str:
    """由内容派生的稳定 id。

    同样的内容永远得到同样的 id，这样"重复提到同一件事"能被识别成
    强化而不是新条目——长期记忆的强化机制依赖这一点。
    """

    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]

    return f"{prefix}-{digest}"


@dataclass
class Goal:
    """一个学习目标。"""

    id: str
    title: str
    detail: str = ""
    deadline: str = ""
    created_at: str = field(default_factory=now_iso)
    status: str = "active"  # active / done / dropped
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "deadline": self.deadline,
            "created_at": self.created_at,
            "status": self.status,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            detail=data.get("detail", ""),
            deadline=data.get("deadline", ""),
            created_at=data.get("created_at") or now_iso(),
            status=data.get("status", "active"),
            tags=list(data.get("tags") or []),
        )


@dataclass
class PlanItem:
    """计划里的一条待办。"""

    id: str
    goal_id: str
    title: str
    day: str  # YYYY-MM-DD
    minutes: int = 45
    done: bool = False
    note: str = ""
    milestone: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "title": self.title,
            "day": self.day,
            "minutes": self.minutes,
            "done": self.done,
            "note": self.note,
            "milestone": self.milestone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanItem:
        return cls(
            id=data["id"],
            goal_id=data.get("goal_id", ""),
            title=data.get("title", ""),
            day=data.get("day", ""),
            minutes=int(data.get("minutes") or 45),
            done=bool(data.get("done")),
            note=data.get("note", ""),
            milestone=bool(data.get("milestone")),
        )


@dataclass
class Review:
    """一天的复盘记录。"""

    day: str
    planned: int = 0
    done: int = 0
    minutes: int = 0
    highlights: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_step: str = ""
    created_at: str = field(default_factory=now_iso)

    @property
    def completion(self) -> float:
        """完成率，没有计划时按 0 处理（避免除零）。"""

        if self.planned <= 0:
            return 0.0

        return self.done / self.planned

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "planned": self.planned,
            "done": self.done,
            "minutes": self.minutes,
            "highlights": list(self.highlights),
            "blockers": list(self.blockers),
            "next_step": self.next_step,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Review:
        return cls(
            day=data["day"],
            planned=int(data.get("planned") or 0),
            done=int(data.get("done") or 0),
            minutes=int(data.get("minutes") or 0),
            highlights=list(data.get("highlights") or []),
            blockers=list(data.get("blockers") or []),
            next_step=data.get("next_step", ""),
            created_at=data.get("created_at") or now_iso(),
        )


@dataclass
class MemoryItem:
    """一条记忆。

    短期和长期共用这个结构，区别只在 scope 和生命周期：
    - scope=short：当天的工作区，满了会被压缩成摘要，跨天清空
    - scope=long ：长期档案，永久保留，靠召回打分被使用

    mentions 记录"这条内容被提到过几次"，是强化的依据；
    superseded_by 记录"这条内容被哪条新记忆取代"，旧内容仍在档案里可查。
    """

    id: str
    scope: str
    category: str
    content: str
    tags: list[str] = field(default_factory=list)
    importance: int = 3  # 1..5
    created_at: str = field(default_factory=now_iso)
    last_used_at: str = field(default_factory=now_iso)
    use_count: int = 0
    mentions: int = 1
    superseded_by: str = ""
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "category": self.category,
            "content": self.content,
            "tags": list(self.tags),
            "importance": self.importance,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "mentions": self.mentions,
            "superseded_by": self.superseded_by,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        return cls(
            id=data["id"],
            scope=data.get("scope", SCOPE_SHORT),
            category=data.get("category", CATEGORY_NOTE),
            content=data.get("content", ""),
            tags=list(data.get("tags") or []),
            importance=int(data.get("importance") or 3),
            created_at=data.get("created_at") or now_iso(),
            last_used_at=data.get("last_used_at") or now_iso(),
            use_count=int(data.get("use_count") or 0),
            mentions=int(data.get("mentions") or 1),
            superseded_by=data.get("superseded_by", ""),
            pinned=bool(data.get("pinned")),
        )
