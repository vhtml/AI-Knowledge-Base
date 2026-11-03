"""两级记忆：短期工作区 + 长期档案。

这是本示例的重点。两级记忆的差别不在容量，而在**生命周期和写入路径**：

短期记忆（short）
    生命周期：当天。工作区满了会把最旧的条目折叠成一条摘要；
    跨天时留下值得沉淀的部分，其余清空。
    写入路径：随手写。每句话、每次复盘都可以往里丢。

长期记忆（long）
    生命周期：跨天、跨会话。永远留在档案里，靠召回打分决定"这次要不要用"。
    写入路径：只能由**晋升**进来，不能直接写。
    这样做的意义是：长期记忆里的每一条都经过了"值得记住"的判断。

两个方向的数据流：

    短期 --晋升(见 promotion.py)--> 长期
    长期 --召回(recall)--> 短期（拼进本轮上下文）

长期记忆还有一个关键性质：**同一件事只保留最新的说法**。
用户说"每天能学 2 小时"，又说"现在改成 3 小时"——后者占用同一个槽位，
前者标记为 superseded 但仍可追溯。这样档案不会随着时间自相矛盾。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from .models import (
    CATEGORY_GOAL,
    CATEGORY_NOTE,
    CATEGORY_PREFERENCE,
    CATEGORY_STRENGTH,
    CATEGORY_SUMMARY,
    CATEGORY_WEAKNESS,
    SCOPE_LONG,
    SCOPE_SHORT,
    MemoryItem,
    now_iso,
    today_str,
)

# 长期记忆里可以互相取代的类别：同一槽位只保留最新说法。
# 目标和里程碑不在此列——它们是历史事实，不该被"最新说法"覆盖。
SLOT_CATEGORIES = (CATEGORY_PREFERENCE, CATEGORY_WEAKNESS, CATEGORY_STRENGTH)

DEFAULT_SHORT_CAPACITY = 24
DEFAULT_LONG_CAPACITY = 200
DEFAULT_RECALL_K = 6

# 中文按字切，英文数字按词切，再补二元组。
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> set[str]:
    """把一段文本切成用于匹配的小片段。

    中文不分词，直接切单字会让召回过宽，所以中文按**二元组**切
    （"注意力" → "注意"、"意力"），英文和数字按整词切。
    """

    lowered = text.lower()
    tokens: set[str] = set(_TOKEN_PATTERN.findall(lowered))

    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(run) == 1:
            tokens.add(run)
            continue

        for index in range(len(run) - 1):
            tokens.add(run[index : index + 2])

    return tokens


def overlap_score(query_tokens: set[str], text: str) -> float:
    """查询与文本的词片重合度，0..1。"""

    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)

    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def recency_weight(item: MemoryItem, now: datetime) -> float:
    """新鲜度权重：越近越高，范围 0.3..1.0。

    但过期衰减有下限——三个月前的事实不该彻底消失，
    真正决定去留的是重要度和是否被取代。
    """

    try:
        last = datetime.fromisoformat(item.last_used_at)
    except ValueError:
        return 1.0

    days = max(0.0, (now - last).total_seconds() / 86400.0)

    return max(0.3, 1.0 - days / 180.0)


@dataclass
class RecallHit:
    """一次召回的结果，带上分数和来源，方便解释"为什么想到这条"。"""

    item: MemoryItem
    score: float
    reason: str

    def render(self) -> str:
        return f"[{self.item.scope}/{self.item.category}] {self.item.content}（{self.reason} {self.score:.2f}）"


class ShortTermMemory:
    """短期工作区：容量受限，跨天清空。

    满了之后不是简单丢弃最旧的，而是把最旧的一批**折叠成一条摘要**——
    摘要本身是一条记忆条目（category=summary），可以被后续召回命中。
    这和 chat_cli 的上下文压缩是同一个思路：丢原文，留语义。
    """

    def __init__(self, capacity: int = DEFAULT_SHORT_CAPACITY) -> None:
        self.capacity = capacity
        self.items: list[MemoryItem] = []
        self.fold_count = 0
        self.day = today_str()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, content: str, category: str = CATEGORY_NOTE, **kwargs) -> MemoryItem:
        """写入一条短期记忆，超容量时自动折叠最旧的一批。

        内容和工作区里已有的某条**完全一致**时，不新增条目，
        而是把那条的 mentions 加一——同一件事反复被提起，
        说明它值得被记住，这是跨天晋升的依据之一。
        """

        for existing in self.items:
            if existing.content == content and existing.category == category:
                existing.mentions += 1
                existing.last_used_at = now_iso()
                return existing

        item = MemoryItem(
            id=f"m-{uuid.uuid4().hex[:8]}",
            scope=SCOPE_SHORT,
            category=category,
            content=content,
            **kwargs,
        )
        self.items.append(item)
        self._fold_if_needed()

        return item

    def _fold_if_needed(self) -> None:
        """超出容量时，把最旧的一半折叠成一条摘要。"""

        if len(self.items) <= self.capacity:
            return

        keep = max(1, self.capacity // 2)
        old = self.items[:-keep]
        self.items = self.items[-keep:]

        if not old:
            return

        folded = MemoryItem(
            id=f"m-{uuid.uuid4().hex[:8]}",
            scope=SCOPE_SHORT,
            category=CATEGORY_SUMMARY,
            content="；".join(item.content for item in old),
            importance=min(5, max((item.importance for item in old), default=3)),
            tags=sorted({tag for item in old for tag in item.tags}),
        )
        self.items.insert(0, folded)
        self.fold_count += 1

    def clear(self) -> None:
        """清空工作区（跨天时调用）。"""

        self.items.clear()

    def snapshot(self) -> list[MemoryItem]:
        return list(self.items)

    def restore(self, items: Iterable[MemoryItem]) -> None:
        self.items = list(items)


class LongTermMemory:
    """长期档案：持久化，槽位去重，靠打分召回。

    "同一件事只保留最新说法"靠**槽位**实现：槽位名由
    (类别, 归一化主题) 决定。写入新条目时如果槽位已被占用，
    旧的标记 superseded_by，但不删除——追溯时还能看到它曾经是什么。
    """

    def __init__(self, capacity: int = DEFAULT_LONG_CAPACITY) -> None:
        self.capacity = capacity
        self.items: list[MemoryItem] = []

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def slot_of(item: MemoryItem) -> str:
        """算条目的槽位名。

        归一化主题：去掉标点、空白和**数字**，再取前 12 个字符。

        去掉数字是关键——"每天能学 2 小时"和"每天能学 3 小时"是同一件事的
        两种说法，应该落在同一个槽位里（后者取代前者）；
        如果把数字留在槽位里，每次改时间预算都会变成一条新记忆，
        档案就会被碎片撑爆。
        """

        if item.category not in SLOT_CATEGORIES:
            return f"{item.category}:{item.id}"

        theme = re.sub(r"[\s，。,.、:：;；!！?？\"'“”‘’()（）0-9]", "", item.content)
        theme = theme[:12].lower()

        return f"{item.category}:{theme}"

    def add(self, item: MemoryItem) -> MemoryItem:
        """写入长期记忆，处理槽位冲突。"""

        item.scope = SCOPE_LONG
        slot = self.slot_of(item)

        for existing in self.items:
            if existing.superseded_by:
                continue

            if self.slot_of(existing) != slot:
                continue

            existing.superseded_by = item.id
            existing.last_used_at = now_iso()
            break

        self.items.append(item)
        self._trim()

        return item

    def _trim(self) -> None:
        """超出容量时淘汰最不重要、且没被反复提及的条目。

        pinned（目标、里程碑等）和有 superseded 标记的条目不参与淘汰，
        后者要保留做追溯。
        """

        if len(self.items) <= self.capacity:
            return

        candidates = [
            item
            for item in self.items
            if not item.pinned and not item.superseded_by and item.category != CATEGORY_GOAL
        ]

        def rank(item: MemoryItem) -> tuple[int, int, str]:
            return (item.importance, item.mentions, item.last_used_at)

        candidates.sort(key=rank)

        overflow = len(self.items) - self.capacity
        drop_ids = {item.id for item in candidates[:overflow]}
        self.items = [item for item in self.items if item.id not in drop_ids]

    def recall(
        self, query: str, limit: int = DEFAULT_RECALL_K, now: datetime | None = None
    ) -> list[RecallHit]:
        """按相关性召回长期记忆。

        打分 = 词片重合 + 重要度 + 新鲜度 + 使用次数，
        被取代的条目不参与召回（但仍在档案里）。
        """

        moment = now or datetime.now()
        query_tokens = tokenize(query)
        hits: list[RecallHit] = []

        for item in self.items:
            if item.superseded_by:
                continue

            relevance = overlap_score(query_tokens, item.content)

            if relevance <= 0 and not item.pinned:
                continue

            score = (
                relevance * 3.0
                + item.importance * 0.4
                + recency_weight(item, moment) * 0.6
                + min(item.use_count, 10) * 0.1
            )

            reason = "匹配" if relevance > 0 else "重要"

            hits.append(RecallHit(item=item, score=score, reason=reason))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        selected = hits[:limit]

        # 召回即使用：被选中的条目使用次数 +1，避免重要但久未召回的内容被淘汰
        for hit in selected:
            self.touch(hit.item)

        return selected

    def touch(self, item: MemoryItem) -> None:
        """标记某条记忆刚被用过——用得越多越不容易被淘汰。"""

        item.use_count += 1
        item.last_used_at = now_iso()

    def find_by_content(self, content: str) -> MemoryItem | None:
        """找内容完全一致的条目（强化机制要用）。"""

        for item in self.items:
            if item.content == content and not item.superseded_by:
                return item

        return None

    def reinforce(self, content: str) -> MemoryItem | None:
        """同一件事被再次提到 → 提升重要度并计数。

        返回被强化的条目；没有匹配则返回 None。
        """

        item = self.find_by_content(content)

        if item is None:
            return None

        item.mentions += 1
        item.use_count += 1
        item.last_used_at = now_iso()
        item.importance = min(5, item.importance + (1 if item.mentions >= 3 else 0))

        return item

    def snapshot(self) -> list[MemoryItem]:
        return list(self.items)

    def restore(self, items: Iterable[MemoryItem]) -> None:
        self.items = list(items)


@dataclass
class MemoryBundle:
    """一次召回的完整结果：短期的 + 长期的。"""

    short: list[RecallHit] = field(default_factory=list)
    long: list[RecallHit] = field(default_factory=list)

    @property
    def hits(self) -> list[RecallHit]:
        return self.short + self.long

    def render(self) -> str:
        """渲染成可以拼进模型上下文的文本。"""

        lines: list[str] = []

        if self.long:
            lines.append("【长期记忆】")

            for hit in self.long:
                lines.append(f"- {hit.item.content}")

        if self.short:
            lines.append("【短期记忆】")

            for hit in self.short:
                lines.append(f"- {hit.item.content}")

        return "\n".join(lines)


class MemoryStore:
    """记忆的持久化：一个 JSON 文件装下两级。

    短期记忆也要存——进程重启后今天的工作区不该丢。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()

    def save(self, short: ShortTermMemory, long: LongTermMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "short": [item.to_dict() for item in short.items],
            "long": [item.to_dict() for item in long.items],
            "fold_count": short.fold_count,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> tuple[ShortTermMemory, LongTermMemory]:
        """读取记忆；文件不存在或损坏时返回空记忆，不让示例直接崩。"""

        if not self.path.exists():
            return ShortTermMemory(), LongTermMemory()

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ShortTermMemory(), LongTermMemory()

        short = ShortTermMemory()
        short.restore(MemoryItem.from_dict(item) for item in payload.get("short") or [])
        short.fold_count = int(payload.get("fold_count") or 0)

        long = LongTermMemory()
        long.restore(MemoryItem.from_dict(item) for item in payload.get("long") or [])

        return short, long


@dataclass
class PromotionDecision:
    """一次晋升判断的留痕。"""

    content: str
    category: str
    reason: str
    accepted: bool

    def render(self) -> str:
        mark = "✓" if self.accepted else "✗"
        return f"{mark} [{self.category}] {self.content} —— {self.reason}"


class MemoryManager:
    """两级记忆的门面：Agent 只跟它打交道。

    对外只有四件事：
    - remember()   写一条短期记忆
    - recall()     召回相关的短期 + 长期记忆
    - promote()    把短期条目晋升到长期
    - rollover()   跨天：清空工作区，但先做一次晋升
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        short_capacity: int = DEFAULT_SHORT_CAPACITY,
        long_capacity: int = DEFAULT_LONG_CAPACITY,
    ) -> None:
        self.store = store
        self.short = ShortTermMemory(capacity=short_capacity)
        self.long = LongTermMemory(capacity=long_capacity)
        self.decisions: list[PromotionDecision] = []

        if store is not None:
            self.short, self.long = store.load()

    def remember(
        self, content: str, category: str = CATEGORY_NOTE, **kwargs
    ) -> MemoryItem:
        """写一条短期记忆。

        内容与长期档案里某条**完全一致**时，视为强化而不是新增：
        这条"我说过很多遍"的事实本来就重要，反复说说明更重要。
        """

        reinforced = self.long.reinforce(content)

        if reinforced is not None:
            self.short.add(content, category, **{**kwargs, "importance": reinforced.importance})
            self._record(
                PromotionDecision(
                    content=content,
                    category=category,
                    reason=f"与长期记忆重复提及，强化至重要度 {reinforced.importance}",
                    accepted=False,
                )
            )
            return reinforced

        return self.short.add(content, category, **kwargs)

    def recall(self, query: str, limit: int = DEFAULT_RECALL_K) -> MemoryBundle:
        """召回：长期优先（更可信），短期补充（更贴近当下）。"""

        long_hits = self.long.recall(query, limit=limit)
        short_hits: list[RecallHit] = []
        query_tokens = tokenize(query)

        for item in self.short.items:
            relevance = overlap_score(query_tokens, item.content)

            if relevance <= 0:
                continue

            score = relevance * 3.0 + item.importance * 0.3
            short_hits.append(RecallHit(item=item, score=score, reason="匹配"))

        short_hits.sort(key=lambda hit: hit.score, reverse=True)

        # 长期记忆的召回已在 LongTermMemory.recall 内计数，这里不重复累加
        return MemoryBundle(short=short_hits[:limit], long=long_hits)

    def promote(
        self,
        content: str,
        category: str,
        reason: str,
        importance: int = 4,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        """把一条内容写进长期档案。

        只允许通过这条路径写长期记忆——它就是"值得记住"的判断本身。
        """

        existing = self.long.find_by_content(content)

        if existing is not None:
            existing.mentions += 1
            existing.last_used_at = now_iso()
            self._record(
                PromotionDecision(content, category, "已在长期档案中，累计强化", True)
            )
            return existing

        item = MemoryItem(
            id=f"m-{uuid.uuid4().hex[:8]}",
            scope=SCOPE_LONG,
            category=category,
            content=content,
            tags=list(tags or []),
            importance=importance,
            pinned=category in (CATEGORY_GOAL, "milestone"),
        )
        self.long.add(item)
        self._record(PromotionDecision(content, category, reason, True))

        return item

    def rollover(self, day: str | None = None) -> list[str]:
        """跨天。

        顺序很重要：**先晋升，再清空**。反过来的话，
        值得沉淀的内容会跟着工作区一起被丢掉。

        返回跨天时晋升进长期档案的内容列表。
        """

        target = day or today_str()
        moved: list[str] = []

        for item in list(self.short.items):
            if item.category == CATEGORY_SUMMARY:
                continue

            if item.mentions >= 2 or item.importance >= 4:
                self.promote(
                    item.content,
                    item.category,
                    f"跨天沉淀（提及 {item.mentions} 次，重要度 {item.importance}）",
                    importance=item.importance,
                    tags=item.tags,
                )
                moved.append(item.content)

        self.short.clear()
        self.short.day = target

        return moved

    def _record(self, decision: PromotionDecision) -> None:
        self.decisions.append(decision)

    def save(self) -> None:
        if self.store is not None:
            self.store.save(self.short, self.long)

    def stats(self) -> dict[str, int]:
        """给 /memory 命令看的统计。"""

        return {
            "short": len(self.short),
            "short_folded": self.short.fold_count,
            "long": len(self.long),
            "long_active": sum(1 for item in self.long.items if not item.superseded_by),
            "long_superseded": sum(1 for item in self.long.items if item.superseded_by),
            "promotions": len(self.decisions),
        }
