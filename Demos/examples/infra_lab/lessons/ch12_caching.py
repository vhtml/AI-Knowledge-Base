"""第 12 章实验：缓存体系——前缀、语义与响应缓存。

这个模块解决什么问题：推理服务的缓存分三级，各有不同的收益和代价，
选错层级比不缓存更危险。本章把三件事变成可验证的断言：

1. **前缀缓存的内部机制**：按 block 粒度哈希、引用计数、LRU 淘汰。
   共享前缀的请求应该只占一份 KV，而不是每个请求各占一份。
2. **三级缓存在同一份流量下的对比**：命中率、内存代价、正确性风险。
   结论是"前缀缓存收益最高且风险最低，语义缓存命中率最高但会答错"。
3. **语义缓存的阈值陷阱**：相似度阈值放松一点，命中率上去了，
   错误答案也跟着上去了。阈值必须靠人工确认，不是靠调参调出来的。
4. **跨租户隔离**：不带租户维度的缓存键就是数据泄漏。

全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..labkit import (
    PagedKVCache,
    banner,
    ensure,
    fmt_gib,
    gpu,
    kv_bytes_per_token,
    model,
    prefix_hash,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
SPEC = model("qwen2.5-7b")

BLOCK_SIZE = 16
"""前缀哈希与分页缓存共用的 block 粒度。"""

PREFIX_TOKENS = 256
"""共享前缀长度（模拟系统提示词）。"""

QUERY_TOKENS = 10
"""单条请求的 query 长度。"""

PREFIX_BLOCKS = -(-PREFIX_TOKENS // BLOCK_SIZE)
"""共享前缀占用的 block 数。"""

EXACT_REPEATS = 20
"""完全相同的重复请求数。"""

PARAPHRASES = 20
"""同义改写的请求数：意图与参数都相同，措辞不同。"""

NEAR_MISSES = 20
"""只差一个关键参数（SKU）的请求数：措辞几乎一样，答案不同。"""

RESPONSE_CACHE_BYTES = 1024
"""响应缓存单条条目的字节数（响应文本本身）。"""

EMBEDDING_DIM = 768
"""语义缓存单条向量的维度，对应 768 维 embedding。"""

EMBEDDING_BYTES = EMBEDDING_DIM * 4
"""语义缓存单条向量的字节数（FP32）。"""


def fmt_bytes(value: int) -> str:
    """把字节数格式化成人类可读的字符串。

    缓存条目的量级差别很大（1 KiB vs 1 GiB），统一用 GiB 会把 1 KiB 显示成 0.00，
    所以小的用 KiB、大的用 GiB。容量换算仍以 labkit 的 BYTES_PER_GIB 口径为准。
    """
    if value >= 1024**3:
        return fmt_bytes(value)
    return f"{value / 1024:.1f}KiB"


@dataclass(frozen=True)
class CacheRequest:
    """一条待缓存的请求。"""

    req_id: int
    prefix_tokens: tuple[int, ...]
    """共享前缀的 token。所有请求共用同一段系统提示词。"""

    query_tokens: tuple[int, ...]
    """query 的 token。措辞相近的请求在这里只差一两个 token。"""

    tenant: str
    intent: str
    """意图标签，比如"查库存"。"""

    params: tuple[str, ...]
    """决定答案的具体参数，比如 SKU 编号。"""

    @property
    def answer_key(self) -> str:
        """这条请求的正确答案标识。"""
        return f"{self.intent}:{'/'.join(self.params)}"

    @property
    def full_tokens(self) -> tuple[int, ...]:
        """整条 prompt。"""
        return self.prefix_tokens + self.query_tokens


@dataclass
class Workload:
    """构造好的请求集合与它们的正确答案。"""

    requests: list[CacheRequest]
    expected: dict[int, str]
    """请求 id → 正确答案标识。"""


QUESTION_COUNT = 5
"""工作集里有 5 个不同的问题（不同 SKU）。"""


def build_workload() -> Workload:
    """造一份流量：重复请求、同义改写、只差参数的近似请求。

    关键设计：**query 指纹与正确答案必须一一对应**。
    否则"完全相同的请求"在不同条目上对应不同答案，响应缓存也会答错，
    实验就测不出缓存层级本身的差别了。

    三类的相似度是刻意设计出来的：
    近似请求（只差 SKU，Jaccard ≈ 0.82）比同义改写（≈ 0.54）**更像**——
    这正是语义缓存最危险的地方，也导致一个无法回避的结论：
    **任何能抓住同义改写的阈值，必然也抓住只差参数的近似请求**。
    """
    prefix = tuple(range(1000, 1000 + PREFIX_TOKENS))
    requests: list[CacheRequest] = []
    expected: dict[int, str] = {}
    req_id = 0

    def base_query(index: int) -> tuple[int, ...]:
        """第 index 个问题的原始 query。

        每个问题用一段**互不重叠**的 token 段，否则不同问题的 token 集合会撞上，
        "完全相同的请求"就会对应不同答案，实验就测不出缓存层级本身的差别。
        """
        start = 5000 + index * 100
        return tuple(range(start, start + 10))

    def add(query: tuple[int, ...], intent: str, params: tuple[str, ...], tenant: str) -> None:
        nonlocal req_id
        requests.append(
            CacheRequest(
                req_id=req_id,
                prefix_tokens=prefix,
                query_tokens=query,
                tenant=tenant,
                intent=intent,
                params=params,
            )
        )
        expected[req_id] = f"{intent}:{'/'.join(params)}"
        req_id += 1

    # 完全相同的重复请求：指纹一样、答案也一样，响应缓存命中必然正确
    for repeat in range(EXACT_REPEATS):
        index = repeat % QUESTION_COUNT
        add(base_query(index), "查库存", (f"SKU-{index:03d}",), "tenant-a")

    # 同义改写：换掉 3 个词，意图与参数都不变
    for variant in range(PARAPHRASES):
        index = variant % QUESTION_COUNT
        query = base_query(index)
        paraphrase = (
            query[:3]
            + (900 + variant,)
            + query[4:5]
            + (910 + variant,)
            + query[6:7]
            + (920 + variant,)
            + query[8:]
        )
        add(paraphrase, "查库存", (f"SKU-{index:03d}",), "tenant-a")

    # 只差 SKU 的近似请求：只换掉最后一个 token，但答案完全不同
    for variant in range(NEAR_MISSES):
        index = variant % QUESTION_COUNT
        query = base_query(index)
        near_miss = query[:9] + (930 + variant,)
        add(near_miss, "查库存", (f"SKU-{100 + index:03d}",), "tenant-a")

    return Workload(requests=requests, expected=expected)


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """两组 token 的 Jaccard 相似度。

    这里用它**代替**真实的 embedding 余弦相似度：离线实验不能调 embedding 模型，
    但阈值的行为是同一个——相似度越高、命中越宽、错答越多。
    """
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class PrefixCacheManager:
    """带引用计数与 LRU 淘汰的前缀缓存管理器。

    块账本完全交给 labkit 的 :class:`PagedKVCache`，
    这里只加两件事：LRU 顺序，以及"同一前缀被多个序列共享时块不能被提前释放"。
    """

    def __init__(self, capacity_blocks: int | None = None, block_size: int = BLOCK_SIZE) -> None:
        self.cache = PagedKVCache(block_size=block_size)
        self.capacity_blocks = capacity_blocks
        self.lru: list[tuple[int, ...]] = []
        """前缀哈希 → 访问顺序，最新的在末尾。"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.peak_blocks = 0

    def lookup(self, key: tuple[int, ...]) -> list[int] | None:
        """查前缀是否在缓存里，命中时续租引用并刷新 LRU。"""
        blocks = self.cache.lookup_prefix(key)
        if blocks is None:
            self.misses += 1
            return None
        self.hits += 1
        for block in blocks:
            self.cache.refcount[block] += 1
        self.lru.remove(key)
        self.lru.append(key)
        return blocks

    def store(self, key: tuple[int, ...], tokens: tuple[int, ...]) -> list[int]:
        """把一段前缀写进缓存，必要时按 LRU 淘汰最久未用的条目。"""
        needed = max(1, -(-len(tokens) // self.cache.block_size))
        blocks: list[int] = []
        for index in range(needed):
            block = self.cache.allocate()
            self.cache.append_tokens(
                block,
                min(self.cache.block_size, len(tokens) - index * self.cache.block_size),
            )
            self.cache.refcount[block] = 1
            blocks.append(block)
        self.cache.cache_prefix(key, blocks)
        self.lru.append(key)
        self.peak_blocks = max(self.peak_blocks, self.cache.used_blocks())
        self._evict_until_fits()
        return blocks

    def release(self, blocks: list[int]) -> None:
        """序列结束，释放它对块的引用。引用计数归零的块才真正回收。"""
        for block in blocks:
            self.cache.free(block)

    def _evict_until_fits(self) -> None:
        """超出容量时按 LRU 淘汰整条前缀。"""
        while (
            self.capacity_blocks is not None
            and self.cache.used_blocks() > self.capacity_blocks
            and len(self.lru) > 1
        ):
            victim = self.lru.pop(0)
            for block in self.cache.lookup_prefix(victim) or []:
                self.cache.free(block)
            del self.cache.prefix_cache[victim]
            self.evictions += 1


class ResponseCache:
    """响应缓存：键是整条请求的精确指纹。"""

    def __init__(self, *, scope_by_tenant: bool = True) -> None:
        self.scope_by_tenant = scope_by_tenant
        """是否把租户放进缓存键。不放就是跨租户泄漏。"""
        self.entries: dict[tuple[int, ...], str] = {}
        self.hits = 0
        self.misses = 0
        self.wrong_hits = 0

    def _key(self, request: CacheRequest) -> tuple[int, ...]:
        """缓存键。带租户维度时把租户哈希混进去。"""
        key = prefix_hash(request.full_tokens, granularity=BLOCK_SIZE)
        if self.scope_by_tenant:
            return (hash(request.tenant) & 0xFFFF, *key)
        return key

    def lookup(self, request: CacheRequest) -> str | None:
        """查响应缓存。命中但答案对不上就是错答。"""
        key = self._key(request)
        cached = self.entries.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        return cached

    def store(self, request: CacheRequest, answer: str) -> None:
        """写响应缓存。"""
        self.entries[self._key(request)] = answer


class SemanticCache:
    """语义缓存：按向量相似度阈值匹配，命中就整段复用旧答案。"""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.entries: list[tuple[tuple[int, ...], str, str]] = []
        """(query token, 租户, 答案)。"""
        self.hits = 0
        self.misses = 0
        self.wrong_hits = 0

    def lookup(self, request: CacheRequest) -> str | None:
        """找最相似的历史条目。相似度超过阈值就整段复用。"""
        best: tuple[float, str] | None = None
        for query, _tenant, answer in self.entries:
            score = jaccard(request.query_tokens, query)
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, answer)
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        return best[1]

    def store(self, request: CacheRequest, answer: str) -> None:
        """写语义缓存。"""
        self.entries.append((request.query_tokens, request.tenant, answer))


def check_prefix_cache_shares_blocks() -> None:
    """共享前缀的请求应该只占一份 KV，引用计数保证它不会提前被释放。"""
    section("1. 前缀缓存：共享前缀只占一份 KV")
    workload = build_workload()
    manager = PrefixCacheManager()
    key = prefix_hash(workload.requests[0].prefix_tokens, granularity=BLOCK_SIZE)

    first = manager.lookup(key)
    ensure(first is None, "首次查询必须未命中")
    manager.store(key, workload.requests[0].prefix_tokens)
    ensure(
        manager.cache.used_blocks() == PREFIX_BLOCKS,
        f"一个 {PREFIX_TOKENS} token 前缀应占 {PREFIX_BLOCKS} 个 block，"
        f"实际 {manager.cache.used_blocks()}",
    )

    # 第二个请求命中同一前缀：块必须被复用而不是重新分配
    second = manager.lookup(key)
    ensure(second is not None, "第二个请求必须命中已存前缀")
    ensure(
        manager.cache.used_blocks() == PREFIX_BLOCKS,
        f"命中时不应新增 block，实际 {manager.cache.used_blocks()}",
    )
    ensure(
        manager.cache.refcount[second[0]] == 2,
        f"共享前缀的引用计数应为 2，实际 {manager.cache.refcount[second[0]]}",
    )

    # 一个序列结束，另一个还在用：块不能被回收
    manager.release(second)
    ensure(
        manager.cache.used_blocks() == PREFIX_BLOCKS,
        "还有一个序列引用时块不能被回收",
    )
    manager.release(second)
    ensure(
        manager.cache.used_blocks() == 0,
        f"所有引用释放后块应被回收，实际仍占 {manager.cache.used_blocks()}",
    )

    # 收益换算成显存
    naive_bytes = len(workload.requests) * PREFIX_BLOCKS * BLOCK_SIZE * kv_bytes_per_token(SPEC, 2)
    shared_bytes = PREFIX_BLOCKS * BLOCK_SIZE * kv_bytes_per_token(SPEC, 2)
    rule(
        f"{len(workload.requests)} 条请求共享 {PREFIX_TOKENS} token 前缀："
        f"各自一份要 {fmt_bytes(naive_bytes)}，共享一份只要 {fmt_bytes(shared_bytes)}"
    )
    ensure(
        shared_bytes * 10 < naive_bytes,
        "共享前缀的显存代价应远小于各自一份",
    )


def check_lru_eviction() -> None:
    """容量不够时按 LRU 淘汰：老前缀被踢掉，新前缀留下。"""
    section("2. LRU 淘汰：容量是硬约束")
    workload = build_workload()
    manager = PrefixCacheManager(capacity_blocks=PREFIX_BLOCKS * 2)
    keys = []
    for index in range(4):
        # 每条前缀都按 PREFIX_TOKENS 长度写入，容量才真的会被撑破
        tokens = tuple(range(3000 + index * PREFIX_TOKENS, 3000 + (index + 1) * PREFIX_TOKENS))
        key = prefix_hash(tokens, granularity=BLOCK_SIZE)
        keys.append(key)
        manager.store(key, tokens)

    ensure(
        manager.cache.used_blocks() <= PREFIX_BLOCKS * 2,
        f"缓存占用不得超过容量，实际 {manager.cache.used_blocks()} blocks",
    )
    ensure(
        manager.evictions > 0,
        "超出容量必须发生淘汰",
    )
    ensure(
        manager.cache.lookup_prefix(keys[0]) is None,
        "最久未用的前缀应被淘汰",
    )
    ensure(
        manager.cache.lookup_prefix(keys[-1]) is not None,
        "最新写入的前缀必须还在",
    )
    rule(
        f"容量 {PREFIX_BLOCKS * 2} blocks 写入 4 个前缀："
        f"淘汰 {manager.evictions} 次，最老的已经不在，新的还在"
    )


def check_semantic_cache_threshold() -> None:
    """语义缓存的阈值是个陷阱：放松一点，错误答案跟着涨。"""
    section("3. 语义缓存：阈值与错答")
    workload = build_workload()
    rows: list[list[str]] = []
    for threshold in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95):
        cache = SemanticCache(threshold=threshold)
        wrong = 0
        for request in workload.requests:
            cached = cache.lookup(request)
            if cached is None:
                cache.store(request, workload.expected[request.req_id])
            elif cached != workload.expected[request.req_id]:
                wrong += 1
        hit_rate = cache.hits / len(workload.requests)
        rows.append(
            [
                f"{threshold:.2f}",
                f"{hit_rate * 100:.1f}%",
                str(cache.hits),
                str(wrong),
                f"{wrong / cache.hits * 100:.1f}%" if cache.hits else "0%",
            ]
        )
    print_table(["相似度阈值", "命中率", "命中数", "答错数", "错答占命中"], rows)

    loose = SemanticCache(threshold=0.30)
    strict = SemanticCache(threshold=0.95)
    for request in workload.requests:
        for cache in (loose, strict):
            cached = cache.lookup(request)
            if cached is None:
                cache.store(request, workload.expected[request.req_id])
            elif cached != workload.expected[request.req_id]:
                cache.wrong_hits += 1

    ensure(
        loose.wrong_hits > 0,
        "宽松阈值下必须出现错答，否则这个实验没意义",
    )
    ensure(
        strict.wrong_hits == 0,
        f"严格阈值下不应有错答，实际 {strict.wrong_hits}",
    )
    ensure(
        loose.hits > strict.hits,
        f"宽松阈值命中率应更高，实际 {loose.hits} vs {strict.hits}",
    )
    rule(
        f"阈值 0.30：命中 {loose.hits} 次、答错 {loose.wrong_hits} 次；"
        f"阈值 0.95：命中 {strict.hits} 次、答错 {strict.wrong_hits} 次"
    )


def check_three_tier_comparison() -> None:
    """三级缓存在同一份流量下对比：收益、内存代价、正确性风险。

    三级缓存省的东西**不一样**，所以必须分开记两个指标，否则表格没法比：

    - ``prefill 节省率``：省掉了多少 prefill 计算。前缀缓存只省这一段。
    - ``请求级命中率``：整条请求免于推理。响应缓存和语义缓存能做到，
      前缀缓存做不到——decode 还是要跑。
    """
    section("4. 三级缓存横向对比")
    workload = build_workload()
    total = len(workload.requests)
    total_prefill = sum(len(r.full_tokens) for r in workload.requests)

    # 响应缓存：精确指纹，只有完全相同的请求才命中
    response = ResponseCache()
    response_saved = 0
    for request in workload.requests:
        cached = response.lookup(request)
        if cached is None:
            response.store(request, workload.expected[request.req_id])
        else:
            response_saved += len(request.full_tokens)
            if cached != workload.expected[request.req_id]:
                response.wrong_hits += 1
    response_hit = response.hits / total

    # 前缀缓存：共享前缀全部命中，答案仍然由模型生成
    manager = PrefixCacheManager()
    prefix_hit = 0
    prefix_saved = 0
    for request in workload.requests:
        key = prefix_hash(request.prefix_tokens, granularity=BLOCK_SIZE)
        blocks = manager.lookup(key)
        if blocks is None:
            manager.store(key, request.prefix_tokens)
        else:
            prefix_hit += 1
            prefix_saved += len(request.prefix_tokens)
    prefix_hit_rate = prefix_hit / total

    # 语义缓存：宽松阈值看收益，严格阈值看能不能上生产
    semantic_loose = SemanticCache(threshold=0.30)
    semantic_strict = SemanticCache(threshold=0.95)
    loose_saved = 0
    strict_saved = 0
    for request in workload.requests:
        for cache, is_loose in ((semantic_loose, True), (semantic_strict, False)):
            cached = cache.lookup(request)
            if cached is None:
                cache.store(request, workload.expected[request.req_id])
                continue
            if is_loose:
                loose_saved += len(request.full_tokens)
            else:
                strict_saved += len(request.full_tokens)
            if cached != workload.expected[request.req_id]:
                cache.wrong_hits += 1
    loose_hit = semantic_loose.hits / total
    strict_hit = semantic_strict.hits / total

    # 内存代价：每条请求摊到的字节数
    response_bytes = RESPONSE_CACHE_BYTES
    prefix_bytes = kv_bytes_per_token(SPEC, 2) * PREFIX_TOKENS
    semantic_bytes = EMBEDDING_BYTES + RESPONSE_CACHE_BYTES

    print_table(
        [
            "缓存层级",
            "请求级命中",
            "prefill 节省",
            "答错数",
            "单条内存代价",
            "失效难度",
            "正确性风险",
        ],
        [
            [
                "响应缓存",
                f"{response_hit * 100:.1f}%",
                f"{response_saved / total_prefill * 100:.1f}%",
                str(response.wrong_hits),
                fmt_bytes(response_bytes),
                "易（TTL/参数版本）",
                "低",
            ],
            [
                "前缀缓存",
                f"{prefix_hit_rate * 100:.1f}%（仅省 prefill）",
                f"{prefix_saved / total_prefill * 100:.1f}%",
                "0",
                fmt_bytes(prefix_bytes),
                "中（block 哈希+引用计数）",
                "低",
            ],
            [
                "语义缓存（宽松 0.30）",
                f"{loose_hit * 100:.1f}%",
                f"{loose_saved / total_prefill * 100:.1f}%",
                str(semantic_loose.wrong_hits),
                fmt_bytes(semantic_bytes),
                "难（意图漂移无法自动判断）",
                "高",
            ],
            [
                "语义缓存（严格 0.95）",
                f"{strict_hit * 100:.1f}%",
                f"{strict_saved / total_prefill * 100:.1f}%",
                str(semantic_strict.wrong_hits),
                fmt_bytes(semantic_bytes),
                "难（意图漂移无法自动判断）",
                "中",
            ],
        ],
    )
    ensure(
        response.wrong_hits == 0,
        f"响应缓存不应答错，实际 {response.wrong_hits}",
    )
    ensure(
        prefix_saved / total_prefill > response_saved / total_prefill,
        "前缀缓存的 prefill 节省率应高于响应缓存",
    )
    ensure(
        semantic_loose.wrong_hits > 0,
        f"宽松阈值的语义缓存必须出现错答，实际 {semantic_loose.wrong_hits}",
    )
    ensure(
        semantic_strict.wrong_hits == 0,
        f"严格阈值的语义缓存不应有错答，实际 {semantic_strict.wrong_hits}",
    )
    ensure(
        loose_hit > strict_hit,
        f"宽松阈值请求级命中率应高于严格阈值，实际 {loose_hit:.3f} vs {strict_hit:.3f}",
    )
    ensure(
        loose_hit > response_hit,
        f"宽松阈值语义缓存的请求级命中率应高于响应缓存，"
        f"实际 {loose_hit:.3f} vs {response_hit:.3f}",
    )
    rule(
        f"前缀缓存 prefill 节省 {prefix_saved / total_prefill * 100:.1f}% 且零错答；"
        f"语义缓存请求级命中 {loose_hit * 100:.1f}% 最高，但错了 {semantic_loose.wrong_hits} 次"
    )


def check_cross_tenant_leak() -> None:
    """缓存键不带租户维度，就是把 A 的答案卖给了 B。"""
    section("5. 跨租户隔离")
    workload = build_workload()
    leaky = ResponseCache(scope_by_tenant=False)
    scoped = ResponseCache(scope_by_tenant=True)

    a_request = workload.requests[0]
    b_request = CacheRequest(
        req_id=9999,
        prefix_tokens=a_request.prefix_tokens,
        query_tokens=a_request.query_tokens,
        tenant="tenant-b",
        intent=a_request.intent,
        params=a_request.params,
    )
    answer = workload.expected[a_request.req_id]

    leaky.store(a_request, answer)
    scoped.store(a_request, answer)
    leak_hit = leaky.lookup(b_request)
    scoped_hit = scoped.lookup(b_request)

    ensure(
        leak_hit is not None,
        "不带租户维度的缓存键必然让 B 命中 A 的答案",
    )
    ensure(
        scoped_hit is None,
        "带租户维度的缓存键不应让 B 命中 A 的答案",
    )
    rule(
        f"同一份请求：共享缓存让 tenant-b 读到了 tenant-a 的答案"
        f"（{leak_hit}），按租户隔离后读不到（{scoped_hit}）"
    )


def check_cache_routing_coupling() -> None:
    """缓存条目长在某个副本上，路由不认它，这个条目就白占显存。"""
    section("6. 缓存与调度耦合")
    # 这一节要用**多个不同前缀**，单前缀看不出"副本上缓存放不下"的矛盾
    replicas = 4
    prefix_count = 8
    prefixes = [
        tuple(range(4000 + index * PREFIX_TOKENS, 4000 + (index + 1) * PREFIX_TOKENS))
        for index in range(prefix_count)
    ]
    keys = [prefix_hash(p, granularity=BLOCK_SIZE) for p in prefixes]
    # 每个副本只能放下 3 个前缀——缓存是硬约束
    per_replica_capacity = 3

    # 情形一：前缀亲和。请求按前缀哈希落到固定的副本，同一前缀永远在一起
    affinity_resident: dict[int, list[tuple[int, ...]]] = {i: [] for i in range(replicas)}
    affinity_hits = 0
    for round_index in range(20):
        for index, key in enumerate(keys):
            replica = index % replicas
            if key in affinity_resident[replica]:
                affinity_hits += 1
            else:
                affinity_resident[replica].append(key)
                if len(affinity_resident[replica]) > per_replica_capacity:
                    affinity_resident[replica].pop(0)
    affinity_total = 20 * prefix_count
    affinity_hit_rate = affinity_hits / affinity_total
    affinity_copies = sum(len(v) for v in affinity_resident.values())

    # 情形二：随机路由。同一前缀散落到多个副本，每个副本都要为它单独占一份
    import random

    rng = random.Random(11)
    random_resident: dict[int, list[tuple[int, ...]]] = {i: [] for i in range(replicas)}
    random_hits = 0
    for _ in range(20):
        for key in keys:
            replica = rng.randrange(replicas)
            if key in random_resident[replica]:
                random_hits += 1
            else:
                random_resident[replica].append(key)
                if len(random_resident[replica]) > per_replica_capacity:
                    random_resident[replica].pop(0)
    random_hit_rate = random_hits / affinity_total
    random_copies = sum(len(v) for v in random_resident.values())

    print_table(
        ["路由策略", "前缀命中率", "常驻前缀总份数", "重复份数"],
        [
            [
                "前缀亲和",
                f"{affinity_hit_rate * 100:.1f}%",
                str(affinity_copies),
                str(affinity_copies - prefix_count),
            ],
            [
                "随机",
                f"{random_hit_rate * 100:.1f}%",
                str(random_copies),
                str(random_copies - prefix_count),
            ],
        ],
    )
    ensure(
        affinity_hit_rate > random_hit_rate,
        f"亲和路由命中率应高于随机，实际 {affinity_hit_rate:.3f} vs {random_hit_rate:.3f}",
    )
    ensure(
        affinity_copies <= prefix_count,
        f"亲和路由下每个前缀只存一份，总份数应 ≤{prefix_count}，实际 {affinity_copies}",
    )
    ensure(
        random_copies > prefix_count,
        f"随机路由下同一前缀散落多副本，总份数应 >{prefix_count}，实际 {random_copies}",
    )
    rule(
        "前缀缓存的收益依赖路由：条目在副本 A 上，请求就得回到 A 才吃得到"
    )


def run() -> None:
    """跑完第 12 章的全部实验。"""
    banner("第 12 章 · 缓存体系：前缀、语义与响应缓存")
    rule(
        f"口径：{SPEC.name} / block={BLOCK_SIZE} / 共享前缀 {PREFIX_TOKENS} tokens / "
        f"请求 {EXACT_REPEATS + PARAPHRASES + NEAR_MISSES} 条"
    )
    check_prefix_cache_shares_blocks()
    check_lru_eviction()
    check_semantic_cache_threshold()
    check_three_tier_comparison()
    check_cross_tenant_leak()
    check_cache_routing_coupling()
    print()
    rule(
        "结论：前缀缓存命中率最高且零错答，是唯一能默认开启的一级；"
        "响应缓存命中率低但绝对安全；语义缓存要靠人工确认阈值，"
        "否则就用错误答案换命中率"
    )
