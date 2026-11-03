"""持久化：目标、计划待办、复盘记录、记忆，全部落成 JSON。

为什么用 JSON 而不是数据库：
- 这个示例要能被直接读懂，JSON 文件比 SQL 直观得多
- 数据量是"一个人几个月的学习记录"，文件完全够用
- 记忆的读写路径才是重点，不该被存储层盖住

目录结构（默认 ~/.study-planner/）：
    goals.json      目标
    plans.json      计划待办
    reviews.json    复盘记录
    memory.json     两级记忆

每个文件独立读写，一个坏了不影响其他。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .memory import MemoryManager, MemoryStore
from .models import Goal, PlanItem, Review

DEFAULT_ROOT = Path.home() / ".study-planner"


class Storage:
    """学习规划师的数据仓库。"""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    # ------------------------------------------------------------
    # 通用读写
    # ------------------------------------------------------------

    def _read(self, name: str) -> Any:
        """读一个 JSON 文件；文件缺失或损坏时返回空，不抛异常。

        损坏时返回空而不是崩溃，是因为学习数据丢一次不该让整个工具不能用。
        """

        path = self._path(name)

        if not path.exists():
            return []

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, name: str, payload: Any) -> None:
        path = self._path(name)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------
    # 目标
    # ------------------------------------------------------------

    def load_goals(self) -> list[Goal]:
        return [Goal.from_dict(item) for item in self._read("goals")]

    def save_goals(self, goals: list[Goal]) -> None:
        self._write("goals", [goal.to_dict() for goal in goals])

    # ------------------------------------------------------------
    # 计划待办
    # ------------------------------------------------------------

    def load_items(self) -> list[PlanItem]:
        return [PlanItem.from_dict(item) for item in self._read("plans")]

    def save_items(self, items: list[PlanItem]) -> None:
        self._write("plans", [item.to_dict() for item in items])

    # ------------------------------------------------------------
    # 复盘
    # ------------------------------------------------------------

    def load_reviews(self) -> list[Review]:
        return [Review.from_dict(item) for item in self._read("reviews")]

    def save_reviews(self, reviews: list[Review]) -> None:
        self._write("reviews", [review.to_dict() for review in reviews])

    # ------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------

    def build_memory(self) -> MemoryManager:
        """装配一个带持久化的记忆管理器。"""

        return MemoryManager(MemoryStore(self._path("memory")))

    # ------------------------------------------------------------
    # 整体保存
    # ------------------------------------------------------------

    def save_all(
        self,
        memory: MemoryManager,
        goals: list[Goal],
        items: list[PlanItem],
        reviews: list[Review],
    ) -> None:
        memory.save()
        self.save_goals(goals)
        self.save_items(items)
        self.save_reviews(reviews)
