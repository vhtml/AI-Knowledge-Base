"""第 6 章实验：记忆系统——短期、长期与检索。

这个模块解决什么问题：第 5 章讲的是"这一轮给模型看什么"，本章讲的是
"跨轮次要留下什么、什么时候把它拿回来"。记忆是最容易做坏的一层：
写少了，Agent 每次都从零开始；写多了，错误信息被固化下来，反复毒害后续会话。

所以本章把七件事变成可执行的断言：

1. **短期记忆 = 消息列表**——``history`` 里的消息真的会进模型输入；
2. **长期记忆跨会话**——同一个存储 + 新会话，Agent 读得到上次的偏好；
3. **检索正确性**——存多条记忆，按关键词召回的是**那一条**，不是一堆；
4. **TTL 遗忘**——过期条目不再被召回；
5. **记忆污染**——写错一次，后续每个会话都按错的做；用户更正后行为变对；
6. **容量淘汰**——超过上限时按策略淘汰，被淘汰的那条不再召回；
7. **记忆不进上下文**——没被召回的记忆，绝不出现在发给模型的消息里。

全部离线运行：不联网、不调真实模型、不花 token。

有一处刻意不用 ``scripted_model``：第 2、5 项要求证明"Agent 是**因为读到了
记忆**才这么回答"。写死的脚本答案证明不了因果——它只是让输出长得像期望。
所以那两处用 :class:`MemoryEchoModel`：一个从上下文推导答案的确定性假模型，
"上下文里有什么"就变成"回答里有什么"，记忆是否真的进了上下文才可被断言。
其余各节一律用 ``scripted_model`` 回放。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Literal

from ..labkit import (
    Message,
    ModelReply,
    ai,
    approx_tokens,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    system,
)
from ..runtime import Agent, AgentConfig

# ---------------------------------------------------------------- 常量

SYSTEM_PROMPT = "你是订单服务助手。回答前先确认用户偏好与已知事实。"

MEMORY_HEADER = "【长期记忆】以下是关于这个用户的已确认事实，回答时必须遵守："
"""注入上下文的记忆块的头。

为什么用 system 角色：记忆是"背景事实"，不是用户这一轮说的话。
放成 system 能压得住模型的临场发挥——但代价是错误记忆的杀伤力也更大，
因为高优先级的东西一旦是错的，模型几乎不会怀疑它。
"""

RetrievalStrategy = Literal["keyword", "semantic", "recency", "importance", "mixed"]

STRATEGY_LABELS: dict[str, str] = {
    "keyword": "关键词",
    "semantic": "语义",
    "recency": "时间",
    "importance": "重要度",
    "mixed": "混合",
}
"""四种检索策略的中文名——笔记里那张表的对应实现。"""

FUZZY_THRESHOLD = 0.5
"""字符相似度低于这个值就不算命中——相似度是"广撒网"，门槛必须高。"""

ALIASES: dict[str, tuple[str, ...]] = {
    "回复": ("回答",),
    "汉语": ("中文",),
    "笔电": ("笔记本",),
}
"""同义词扩展表。

真实系统靠 embedding 模型做这件事；这里用一张手写表把"语义检索"的
效果近似出来，好让读者看清两件事：语义检索能接住同义表述，
但它比关键词检索贵，而且更容易召回噪声。
"""

_LATIN = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[一-鿿]+")

# ---------------------------------------------------------------- 文本工具


def tokenize(text: str) -> set[str]:
    """把文本切成用于匹配的小片段。

    中文按二元组切（"回答" → "回答"），英文和数字按整词切。
    用二元组而不是单字：单字切会让"我用中文"匹配上"我在上海"，
    召回集合立刻变脏——检索正确性是记忆系统的地基。
    """
    lowered = text.lower()
    tokens = set(_LATIN.findall(lowered))

    for run in _CJK.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
            continue
        for index in range(len(run) - 1):
            tokens.add(run[index : index + 2])

    return tokens


def expand_tokens(tokens: set[str]) -> set[str]:
    """用同义词表扩展 token 集合——语义检索的近似实现。"""
    expanded = set(tokens)
    for token in tokens:
        expanded.update(ALIASES.get(token, ()))
    return expanded


def keyword_score(query_tokens: set[str], text: str) -> float:
    """查询与文本的词片重合度，0..1。"""
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def fuzzy_score(query: str, text: str) -> float:
    """字符相似度（difflib），0..1。

    关键词检索漏掉"回复/回答"这种同义改写时，它能兜住一点——
    但它也会把不相干的文本拉进来，所以只能当辅助信号。
    """
    return SequenceMatcher(None, query, text).ratio()


# ---------------------------------------------------------------- 数据模型


@dataclass
class MemoryItem:
    """一条长期记忆。

    字段对应"记忆四问"里的"存什么"：内容、类型、重要度、来源、生命周期。
    """

    id: str
    content: str
    kind: Literal["fact", "preference"] = "fact"
    importance: int = 3
    slot: str = ""
    """冲突覆盖用的槽位。同槽位只保留最新说法（旧的标记为被取代）。"""
    source: Literal["explicit", "implicit"] = "explicit"
    """写入来源：显式（用户明说"记住"）还是隐式（从对话里推断）。"""
    needs_confirmation: bool = False
    """待确认标记——未经确认的推断不许进上下文。"""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    superseded_by: str | None = None


@dataclass
class FakeClock:
    """可推进的假时钟。

    TTL 和淘汰策略只有在**时间可控**时才能断言——真实时钟只能干等。
    """

    now: datetime = datetime(2026, 1, 1, 9, 0, 0)

    def __call__(self) -> datetime:
        """返回当前时间，签名和 ``datetime.now`` 一样。"""
        return self.now

    def advance(self, seconds: int) -> None:
        """把时间往前推。"""
        self.now = self.now + timedelta(seconds=seconds)


def recency_score(item: MemoryItem, now: datetime) -> float:
    """新鲜度权重，0..1，半衰期约 1 小时。"""
    age = max(0.0, (now - item.created_at).total_seconds())
    return 1.0 / (1.0 + age / 3600.0)


# ---------------------------------------------------------------- 记忆存储


class MemoryStore:
    """长期记忆存储：写、取、忘，三件事都在这里。

    刻意做成"进程内的一个列表"而不是数据库：这一章要验证的是**策略**
    （什么时候写、什么时候召回、什么时候忘），不是持久化技术。
    换成文件或数据库时，下面这些策略一行都不用改。
    """

    def __init__(
        self,
        capacity: int = 8,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.capacity = capacity
        self._clock: Callable[[], datetime] = clock or datetime.now
        self.items: list[MemoryItem] = []
        self._seq = 0

    # ------------------------------------------------------------ 写

    def write(
        self,
        content: str,
        kind: Literal["fact", "preference"] = "fact",
        importance: int = 3,
        slot: str = "",
        source: Literal["explicit", "implicit"] = "explicit",
        ttl_seconds: int | None = None,
    ) -> MemoryItem:
        """写入一条记忆，返回写进去的条目。

        隐式写入一律标记为待确认：从对话里推断出来的东西，
        未经用户确认就进上下文，是记忆污染最常见的来源。
        """
        self._seq += 1
        now = self._clock()

        item = MemoryItem(
            id=f"m{self._seq:02d}",
            content=content,
            kind=kind,
            importance=importance,
            slot=slot,
            source=source,
            needs_confirmation=source == "implicit",
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
        )

        self._supersede_conflicts(item)
        self.items.append(item)
        self._evict_if_needed()
        return item

    def confirm(self, memory_id: str) -> MemoryItem | None:
        """确认一条待确认的记忆——用户点头之后它才允许进上下文。"""
        item = self.get(memory_id)
        if item is None:
            return None
        item.needs_confirmation = False
        return item

    def correct(self, memory_id: str, new_content: str) -> MemoryItem | None:
        """用新说法覆盖旧记忆（冲突覆盖），返回新条目。

        旧条目不删，只标记为被取代——"曾经这么以为"必须能追溯，
        这是记忆污染事后复盘的唯一线索。
        """
        old = self.get(memory_id)
        if old is None:
            return None

        return self.write(
            new_content,
            kind=old.kind,
            importance=old.importance,
            slot=old.slot,
            source="explicit",
        )

    def _supersede_conflicts(self, item: MemoryItem) -> None:
        """同槽位冲突：旧的标记为被取代（不删，保留可追溯性）。"""
        if not item.slot:
            return

        for existing in self.items:
            if existing.superseded_by:
                continue
            if existing.slot == item.slot:
                existing.superseded_by = item.id

    # ------------------------------------------------------------ 忘

    def forget(self, memory_id: str) -> bool:
        """用户显式删除一条记忆，返回是否真的删掉了。"""
        before = len(self.items)
        self.items = [item for item in self.items if item.id != memory_id]
        return len(self.items) < before

    def forget_expired(self) -> list[str]:
        """清掉 TTL 已过期的条目，返回被清掉的 id。"""
        now = self._clock()
        expired = {
            item.id for item in self.items if item.expires_at and item.expires_at <= now
        }
        self.items = [item for item in self.items if item.id not in expired]
        return sorted(expired)

    def _evict_if_needed(self) -> None:
        """容量淘汰：超上限时淘汰"又轻又旧"的那一类。

        排序键是 (重要度, 写入时间)：一条被引用过十次的长期偏好，
        比一条今天刚记的临时链接更该留下。
        """
        overflow = len(self.items) - self.capacity
        if overflow <= 0:
            return

        active = [item for item in self.items if not item.superseded_by]
        ranked = sorted(active, key=lambda item: (item.importance, item.created_at))
        drop = {item.id for item in ranked[:overflow]}
        remaining = [item for item in self.items if item.id not in drop]

        if len(remaining) > self.capacity:
            # 活跃条目不够淘汰时，才动被取代的历史条目
            stale = sorted(
                (item for item in remaining if item.superseded_by),
                key=lambda item: item.created_at,
            )
            for item in stale:
                remaining.remove(item)
                if len(remaining) <= self.capacity:
                    break

        self.items = remaining

    # ------------------------------------------------------------ 取

    def get(self, memory_id: str) -> MemoryItem | None:
        """按 id 取一条记忆（含已被取代的）——追溯用。"""
        for item in self.items:
            if item.id == memory_id:
                return item
        return None

    def recall(
        self,
        query: str,
        limit: int = 3,
        strategy: RetrievalStrategy = "mixed",
    ) -> list[MemoryItem]:
        """按策略召回记忆。

        检索策略分两层：**候选**（谁有资格进召回）和**排序**（进来之后谁排前面）。
        关键词/语义决定候选；时间/重要度决定排序。
        混合格式用词片重合度当门槛，靠相关度、重要度、新鲜度一起排序。
        """
        now = self._clock()
        query_tokens = tokenize(query)
        expanded = expand_tokens(query_tokens)
        scored: list[tuple[MemoryItem, float]] = []

        for item in self.items:
            if item.superseded_by:
                continue
            if item.needs_confirmation:
                continue
            if item.expires_at is not None and item.expires_at <= now:
                continue

            keyword = keyword_score(query_tokens, item.content)
            semantic = keyword_score(expanded, item.content)
            fuzzy = fuzzy_score(query, item.content)

            if strategy == "keyword":
                if keyword <= 0:
                    continue
                score = keyword
            elif strategy == "semantic":
                if semantic <= 0 and fuzzy < FUZZY_THRESHOLD:
                    continue
                score = semantic * 0.7 + fuzzy * 0.3
            elif strategy == "recency":
                score = recency_score(item, now)
            elif strategy == "importance":
                score = item.importance / 5.0
            else:
                if keyword <= 0 and semantic <= 0 and fuzzy < FUZZY_THRESHOLD:
                    continue
                score = (
                    keyword * 3.0
                    + semantic * 1.0
                    + item.importance * 0.4
                    + recency_score(item, now) * 0.6
                )

            scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in scored[:limit]]


# ---------------------------------------------------------------- 装配上下文


def extract_memory_lines(messages: Sequence[Message]) -> list[str]:
    """从消息里取出记忆块的条目——回答里出现什么，全看这里。"""
    lines: list[str] = []

    for message in messages:
        if not message.content.startswith(MEMORY_HEADER):
            continue
        for line in message.content.splitlines()[1:]:
            text = line.strip().lstrip("- ").strip()
            if text:
                lines.append(text)

    return lines


def build_memory_context(
    store: MemoryStore,
    query: str,
    limit: int = 3,
    strategy: RetrievalStrategy = "mixed",
) -> list[Message]:
    """把召回的记忆装配成可注入上下文的消息。

    只有**被召回的**记忆才进上下文——没召回的记忆一条都不许进。
    记忆是给模型的输入，不是给模型看的数据库。
    """
    hits = store.recall(query, limit=limit, strategy=strategy)
    if not hits:
        return []

    body = "\n".join(f"- {item.content}" for item in hits)
    return [Message(role="system", content=MEMORY_HEADER + "\n" + body)]


@dataclass
class MemoryEchoModel:
    """一个"只按记忆回答"的确定性假模型。

    为什么自己写一个，而不用 ``scripted_model`` 回放固定答案：
    写死的答案证明不了因果——它只是让输出长得像期望。
    这个模型把"上下文里有什么"变成"回答里有什么"，
    于是"Agent 是不是因为读到了记忆才这么答"可以被断言。
    """

    calls: list[list[Message]] = field(default_factory=list)
    fallback: str = "本轮没有可用的记忆。"

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] | None = None,
    ) -> ModelReply:
        """按上下文里的记忆块作答，并把收到的消息记下来。"""
        self.calls.append(list(messages))

        recalled = extract_memory_lines(messages)
        if not recalled:
            return ModelReply(content=self.fallback)

        return ModelReply(content="已知的信息：" + "；".join(recalled))


# ---------------------------------------------------------------- 1. 短期记忆


def check_short_term_memory() -> None:
    """短期记忆就是消息列表：history 里的消息会真的进模型输入。"""
    section("短期记忆：就是消息列表，随 history 进上下文")

    history = [
        human("我叫小王，常用订单号 A-1001。"),
        ai("收到，小王。需要我查哪一单？"),
    ]

    model = scripted_model(["小王，你的订单 A-1001 已发货。"])
    agent = Agent(model, [], AgentConfig(system_prompt=SYSTEM_PROMPT))
    result = agent.run("我的订单怎么了？", history=history)

    seen = model.calls[0]
    ensure(
        any(m.content == "我叫小王，常用订单号 A-1001。" for m in seen),
        "期望模型收到历史里的用户消息，实际收到的消息里没有它",
    )
    ensure(
        any(m.content == "收到，小王。需要我查哪一单？" for m in seen),
        "期望模型收到历史里的助手消息，实际收到的消息里没有它",
    )
    ensure(
        len(seen) == len(history) + 2,
        f"期望模型收到 {len(history)} 条历史 + 系统提示 + 本轮提问共 {len(history) + 2} 条，"
        f"实际收到 {len(seen)} 条",
    )
    ensure(
        any(m.content == "我叫小王，常用订单号 A-1001。" for m in result.messages),
        "期望历史消息也出现在运行结果的消息列表里，实际没有",
    )

    # 不带 history，模型就不知道你是谁——短期记忆丢了就是丢了
    blind_model = scripted_model(["我不知道你是谁。"])
    blind = Agent(blind_model, [], AgentConfig(system_prompt=SYSTEM_PROMPT)).run(
        "我的订单怎么了？"
    )
    ensure(
        not any("小王" in m.content for m in blind.messages),
        "期望不带 history 时模型看不到上一轮的信息，实际看到了「小王」",
    )

    print(f"  带 history：模型收到 {len(seen)} 条消息（含 {len(history)} 条历史）")
    print(f"  不带 history：模型只收到 {len(blind_model.calls[0])} 条消息")
    print("  → 短期记忆没有跨会话能力：进程外就没有了")


# ---------------------------------------------------------------- 2. 长期记忆


def check_long_term_memory() -> None:
    """长期记忆跨会话：同一个存储 + 新会话，Agent 读得到上次的偏好。"""
    section("长期记忆：换个会话还记得")

    store = MemoryStore()
    store.write(
        "用户偏好用中文回答",
        kind="preference",
        slot="answer_language",
        importance=5,
    )

    model = MemoryEchoModel()
    agent = Agent(model, [], AgentConfig(system_prompt=SYSTEM_PROMPT))
    result = agent.run(
        "以后怎么跟我说话？", history=build_memory_context(store, "回答语言")
    )

    ensure(
        "中文回答" in result.final_answer,
        f"期望 Agent 说出上次的偏好「中文回答」，实际回答 {result.final_answer!r}",
    )
    ensure(
        any(m.content.startswith(MEMORY_HEADER) for m in model.calls[0]),
        "期望记忆块进了模型输入，实际模型输入里没有记忆块",
    )

    # 换一个全新的存储（= 另一个用户），同样的问题就答不上来
    fresh_model = MemoryEchoModel()
    fresh = Agent(fresh_model, [], AgentConfig(system_prompt=SYSTEM_PROMPT)).run(
        "以后怎么跟我说话？", history=build_memory_context(MemoryStore(), "回答语言")
    )
    ensure(
        "中文回答" not in fresh.final_answer,
        f"期望没有记忆的新会话答不出偏好，实际回答 {fresh.final_answer!r}",
    )

    print(f"  会话二（同一存储）：{result.final_answer!r}")
    print(f"  会话二（全新存储）：{fresh.final_answer!r}")
    print("  → 差别不在 Agent，在存储：记忆活在存储里，不在 Agent 里")


# ---------------------------------------------------------------- 3. 检索正确性


def check_retrieval_precision() -> None:
    """召回的是那一条，不是一堆。"""
    section("检索正确性：召回集合必须精确")

    store = MemoryStore()
    pref = store.write(
        "用户偏好用中文回答", kind="preference", slot="answer_language", importance=5
    )
    city = store.write("用户常驻上海", kind="fact", slot="city", importance=3)
    order = store.write("用户的常用订单号是 A-1001", kind="fact", slot="order", importance=2)

    hits = store.recall("回答语言", strategy="keyword")
    ensure(
        [item.id for item in hits] == [pref.id],
        "期望关键词检索只召回「用户偏好用中文回答」一条，"
        f"实际召回 {[item.content for item in hits]}",
    )

    hits_city = store.recall("常驻城市", strategy="keyword")
    ensure(
        [item.id for item in hits_city] == [city.id],
        "期望「常驻城市」只召回常驻城市那一条，"
        f"实际召回 {[item.content for item in hits_city]}",
    )

    hits_order = store.recall("订单号", strategy="keyword")
    ensure(
        [item.id for item in hits_order] == [order.id],
        "期望「订单号」只召回订单号那一条，"
        f"实际召回 {[item.content for item in hits_order]}",
    )

    # 关键词检索的代价：换个说法就抓不到了
    paraphrase = "以后回复我时请用汉语"
    keyword_hits = store.recall(paraphrase, strategy="keyword")
    semantic_hits = store.recall(paraphrase, strategy="semantic")

    ensure(
        keyword_hits == [],
        f"期望关键词检索接不住同义改写「{paraphrase}」，"
        f"实际召回了 {[item.content for item in keyword_hits]}",
    )
    ensure(
        [item.id for item in semantic_hits] == [pref.id],
        "期望语义检索（近义词扩展）召回「用户偏好用中文回答」，"
        f"实际召回 {[item.content for item in semantic_hits]}",
    )

    print(f"  关键词检索「{paraphrase}」→ {len(keyword_hits)} 条（漏了）")
    print(f"  语义检索「{paraphrase}」→ {[item.content for item in semantic_hits]}")
    print("  → 关键词精确但接不住同义表述；语义能接住，但更容易召回噪声")


def check_retrieval_strategies() -> None:
    """四种检索策略，四种取舍。"""
    section("四种检索策略：关键词 / 语义 / 时间 / 重要度")

    clock = FakeClock()
    store = MemoryStore(clock=clock)

    oldest = store.write("用户常驻上海", kind="fact", slot="city", importance=5)
    clock.advance(3600)
    store.write("用户的常用订单号是 A-1001", kind="fact", slot="order", importance=3)
    clock.advance(3600)
    newest = store.write("用户偏好用中文回答", kind="preference", slot="answer_language", importance=2)

    picked: dict[str, list[str]] = {}
    for strategy in ("keyword", "semantic", "recency", "importance"):
        hits = store.recall("常驻", limit=1, strategy=strategy)
        picked[strategy] = [item.content for item in hits]
        print(f"    {STRATEGY_LABELS[strategy]:<4} → {picked[strategy]}")

    ensure(
        picked["keyword"] == [oldest.content],
        f"期望关键词检索命中「{oldest.content}」，实际 {picked['keyword']}",
    )
    ensure(
        picked["semantic"] == [oldest.content],
        f"期望语义检索命中「{oldest.content}」，实际 {picked['semantic']}",
    )
    ensure(
        picked["recency"] == [newest.content],
        f"期望时间检索命中最新写入的「{newest.content}」，实际 {picked['recency']}",
    )
    ensure(
        picked["importance"] == [oldest.content],
        f"期望重要度检索命中最重要的「{oldest.content}」，实际 {picked['importance']}",
    )
    print("  → 候选靠匹配，排序靠时间/重要度；换策略就是换一批事实")


# ---------------------------------------------------------------- 4. 遗忘


def check_ttl_expiry() -> None:
    """TTL 过期后不再被召回。"""
    section("遗忘（一）：TTL 过期")

    clock = FakeClock()
    store = MemoryStore(clock=clock)
    temporary = store.write(
        "用户正在赶一个临时 deadline",
        kind="fact",
        slot="deadline",
        ttl_seconds=3600,
    )
    permanent = store.write(
        "用户偏好用中文回答", kind="preference", slot="answer_language", importance=5
    )

    ensure(
        [item.id for item in store.recall("deadline")] == [temporary.id],
        "期望还没过期时能召回临时记忆，实际召回不到",
    )

    clock.advance(7200)

    # 先不清理：召回这一层本身就该跳过过期条目
    ensure(
        store.recall("deadline") == [],
        "期望过期条目不再被召回，实际仍被召回",
    )

    expired = store.forget_expired()
    ensure(
        expired == [temporary.id],
        f"期望清掉过期条目 {[temporary.id]}，实际清掉 {expired}",
    )
    ensure(
        store.get(temporary.id) is None,
        "期望过期条目已从存储里移除，实际还在",
    )
    ensure(
        [item.id for item in store.recall("回答语言")] == [permanent.id],
        "期望没过期的记忆仍然可以被召回，实际召回不到",
    )

    print(f"  过期清理：{expired}（TTL 3600s，实际过了 7200s）")
    print("  → 过期条目在召回层就被跳过，清理只是回收空间")


def check_capacity_eviction() -> None:
    """超过容量上限时按策略淘汰，被淘汰的那条不再召回。"""
    section("遗忘（二）：容量淘汰")

    clock = FakeClock()
    store = MemoryStore(capacity=3, clock=clock)

    light = store.write("用户常驻上海", kind="fact", slot="city", importance=2)
    clock.advance(600)
    middle = store.write(
        "用户的常用订单号是 A-1001", kind="fact", slot="order", importance=4
    )
    clock.advance(600)
    store.write("用户偏好用中文回答", kind="preference", slot="answer_language", importance=5)
    clock.advance(600)
    store.write("用户在做 Python 迁移项目", kind="fact", slot="project", importance=3)

    ensure(
        len(store.items) == 3,
        f"期望存储容量收敛到 3 条，实际 {len(store.items)} 条",
    )
    ensure(
        store.get(light.id) is None,
        "期望最不重要、写入最早的那条被淘汰，实际它还在",
    )
    ensure(
        store.get(middle.id) is not None,
        "期望重要度更高的条目留下，实际它被淘汰了",
    )
    ensure(
        store.recall("常驻城市", strategy="keyword") == [],
        "期望被淘汰的那条不再被召回，实际还能召回",
    )

    print(f"  写入 4 条 → 留下 {len(store.items)} 条，淘汰「{light.content}」")
    print("  → 淘汰顺序是 (重要度, 写入时间)：又轻又旧的先走")


# ---------------------------------------------------------------- 5. 记忆污染


def check_memory_poisoning() -> None:
    """写错一次，后续每个会话都错；用户更正后行为变对。"""
    section("记忆污染：写错一次，之后每个会话都错")

    store = MemoryStore()

    # 隐式写入：用户随口一句，Agent 推断成了长期偏好
    candidate = store.write(
        "用户主力使用的电脑系统是 macOS",
        kind="preference",
        slot="os",
        source="implicit",
        importance=4,
    )
    ensure(
        candidate.source == "implicit" and candidate.needs_confirmation,
        "期望隐式写入的记忆默认待确认，实际没有",
    )

    # 未确认的隐式推断不进上下文——第一道防线
    ensure(
        build_memory_context(store, "主力系统") == [],
        "期望待确认的隐式记忆不进上下文，实际进了",
    )

    confirmed = store.confirm(candidate.id)
    ensure(
        confirmed is not None and not confirmed.needs_confirmation,
        "期望确认后记忆变为可用状态，实际没有",
    )

    def ask() -> str:
        """开一个全新的会话，带着同一份长期记忆问一次。"""
        model = MemoryEchoModel()
        agent = Agent(model, [], AgentConfig(system_prompt=SYSTEM_PROMPT))
        result = agent.run(
            "给我一个主力系统下的命令",
            history=build_memory_context(store, "主力系统"),
        )
        return result.final_answer

    first = ask()
    ensure(
        "macOS" in first,
        f"期望被污染的记忆让 Agent 按 macOS 回答，实际回答 {first!r}",
    )

    second = ask()
    ensure(
        "macOS" in second,
        f"期望污染跨会话持续存在（新会话仍按 macOS），实际回答 {second!r}",
    )

    # 用户更正：同槽位覆盖，旧的标记为被取代
    fixed = store.correct(candidate.id, "用户主力使用的电脑系统是 Windows")
    ensure(
        fixed is not None and fixed.id != candidate.id,
        "期望更正产生一条新记忆，实际没有",
    )
    ensure(
        candidate.superseded_by == fixed.id,
        "期望旧记忆被标记为被取代（仍要可追溯），实际没有标记",
    )

    after = ask()
    ensure(
        "Windows" in after and "macOS" not in after,
        f"期望更正后行为变对（回答 Windows），实际回答 {after!r}",
    )
    ensure(
        store.get(candidate.id) is not None,
        "期望旧记忆仍留在存储里可追溯，实际被物理删除了",
    )

    print(f"  污染后  ：{first!r}")
    print(f"  下个会话：{second!r}（仍然错）")
    print(f"  更正后  ：{after!r}")
    print("  → 记忆污染比记不住严重：错误会被反复放大，直到有人更正它")


# ---------------------------------------------------------------- 6. 记忆不进上下文


def check_memory_not_in_context() -> None:
    """没被召回的记忆，一条都不许进模型输入。"""
    section("记忆不进上下文：没召回的，模型看不见")

    store = MemoryStore()
    store.write("用户偏好用中文回答", kind="preference", slot="answer_language")
    store.write("用户常驻上海", kind="fact", slot="city")
    store.write("用户的常用订单号是 A-1001", kind="fact", slot="order")

    model = MemoryEchoModel()
    agent = Agent(model, [], AgentConfig(system_prompt=SYSTEM_PROMPT))
    agent.run("我的常用订单号是多少？", history=build_memory_context(store, "订单号"))

    seen = model.calls[0]
    seen_text = "\n".join(m.content for m in seen)

    ensure(
        "A-1001" in seen_text,
        "期望被召回的记忆进了模型输入，实际没进去",
    )
    ensure(
        "常驻上海" not in seen_text,
        "期望未被召回的记忆不出现在模型输入里，实际出现了「常驻上海」",
    )
    ensure(
        "用中文回答" not in seen_text,
        "期望未被召回的记忆不出现在模型输入里，实际出现了「用户偏好用中文回答」",
    )
    ensure(
        sum(1 for m in seen if m.content.startswith(MEMORY_HEADER)) == 1,
        f"期望模型输入里恰好有一个记忆块，实际有 "
        f"{sum(1 for m in seen if m.content.startswith(MEMORY_HEADER))} 个",
    )

    # 成本：召回一条 ≠ 把整个库塞进去
    whole = approx_tokens("；".join(item.content for item in store.items))
    context_message = build_memory_context(store, "订单号")[0]
    body = context_message.content.split("\n", 1)[1]
    only = approx_tokens(body)
    header_cost = approx_tokens(context_message.content) - only

    ensure(
        only < whole,
        f"期望只召回一条的正文比整个记忆库小，实际 {only} vs {whole} token",
    )

    print(
        f"  整个记忆库 {whole} token → 只召回一条 {only} token"
        f"（块头固定开销 {header_cost} token）"
    )
    print(f"  模型输入 {len(seen)} 条消息，其中记忆块 1 条")
    print("  → 记忆是给模型的输入，不是给模型看的数据库")


# ---------------------------------------------------------------- 7. 记忆 vs RAG


def check_memory_vs_rag() -> None:
    """记忆检索的是"关于这个用户的事实"，RAG 检索的是"外部知识"。"""
    section("记忆 vs RAG：检索的是两种东西")

    store = MemoryStore()
    store.write("用户偏好用中文回答", kind="preference", slot="answer_language")

    documents = [
        "《运维手册》第 3 章：连接池打满的典型表现是 P99 延迟上升。",
        "《运维手册》第 5 章：连接池打满时先看活跃连接数是否触顶。",
    ]

    def rag(query: str) -> list[str]:
        """一个最朴素的关键词 RAG——只为对照，不为生产。"""
        tokens = tokenize(query)
        return [doc for doc in documents if keyword_score(tokens, doc) > 0]

    memory_hits = store.recall("回答语言", strategy="keyword")
    ensure(
        [item.content for item in memory_hits] == ["用户偏好用中文回答"],
        f"期望记忆检索命中用户偏好，实际 {[item.content for item in memory_hits]}",
    )
    ensure(
        rag("回答语言") == [],
        f"期望 RAG 在运维手册里找不到「回答语言」，实际找到了 {rag('回答语言')}",
    )

    ensure(
        store.recall("连接池打满", strategy="keyword") == [],
        "期望记忆库里没有「连接池打满」，实际召回了什么",
    )
    ensure(
        len(rag("连接池打满")) == 2,
        f"期望 RAG 召回 2 段运维手册，实际召回 {len(rag('连接池打满'))} 段",
    )

    print("  查询「回答语言」：记忆 → 偏好；RAG → 无命中")
    print("  查询「连接池打满」：记忆 → 无命中；RAG → 2 段手册")
    print("  → 同一个查询串，两套语料两套答案：记忆关于用户，RAG 关于世界")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型基于召回的记忆回答，只校验形状。"""
    banner("第 6 章 · 真实模型路径")
    from ..labkit import real_model

    store = MemoryStore()
    store.write(
        "用户偏好用中文回答", kind="preference", slot="answer_language", importance=5
    )

    context = build_memory_context(store, "回答语言")
    messages = [system(SYSTEM_PROMPT), *context, human("以后怎么跟我说话？")]

    model = real_model()
    reply = model.complete(messages)

    print(f"  注入 {len(context)} 条记忆块后提问")
    print(f"  回答：{reply.content[:200]!r}")

    ensure(bool(reply.content), "真实模型应当给出回答")
    ensure(
        len(context) == 1,
        f"期望注入 1 个记忆块，实际 {len(context)} 个",
    )


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 6 章实验。"""
    banner("第 6 章 · 记忆系统：短期、长期与检索")
    check_short_term_memory()
    check_long_term_memory()
    check_retrieval_precision()
    check_retrieval_strategies()
    check_ttl_expiry()
    check_memory_poisoning()
    check_capacity_eviction()
    check_memory_not_in_context()
    check_memory_vs_rag()
    rule("记忆四问想不清楚，记忆系统必然失控；写错一次的代价远大于记不住")
    print("\n✓ 第 6 章通过")
