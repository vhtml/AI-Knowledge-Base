"""迷你推理服务：把 18 章的机制装进一个能跑的服务里。

这个模块解决什么问题：第 5、6、7、11、12 章各自都实现过一遍机制，
但工程上真正的问题是**它们合在一起是什么形状**。这份实现把关键机制
拼成一个最小可运行的服务：

- **分页 KV 缓存**（第 5、6 章）：块表 + 引用计数，前缀块可共享
- **连续批处理**（第 7 章）：iteration 级调度，每轮补位，长短请求不互相拖死
- **前缀缓存**（第 12 章）：block 粒度哈希，命中则跳过已算过的 prefill
- **前缀亲和路由**（第 11 章）：相同前缀的请求落到同一个副本，缓存才命中
- **令牌桶限流**（第 11 章）：网关入口削峰

时间全部用解析模型算（:mod:`infra_lab.labkit` 里的 prefill/decode 公式），
所以没有 GPU 也能跑出可复现的指标。

**为什么不用 labkit 的 EngineSimulator**：那个模拟器是单引擎单队列的，
用来讲清调度策略；这个毕业作品要的是**多副本 + 跨副本缓存共享 + 网关**
的形状，所以引擎循环在这里重写一遍——这正是第 7 章教的做法。
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from ..labkit import (
    GPUSpec,
    ModelSpec,
    PagedKVCache,
    block_chain,
    decode_step_time,
    gpu,
    kv_bytes_per_token,
    model,
    percentile,
    prefill_time,
)

__all__ = [
    "Completion",
    "MiniServe",
    "RateLimiter",
    "Replica",
    "ServeConfig",
    "ServeReport",
    "ServeRequest",
    "Sequence",
    "synthetic_tokens",
]


# ---------------------------------------------------------------- 配置与请求


@dataclass(frozen=True)
class ServeConfig:
    """服务配置。

    默认值对应一张 H100 跑 7B 模型的服务形态：
    FP16 权重占 14.2 GiB，留 60 GiB 给 KV cache，
    批上限 16，单副本 KV 预算 12 万 token。
    """

    device_name: str = "h100_sxm"
    model_name: str = "qwen2.5-7b"
    replicas: int = 2
    max_batch: int = 16
    kv_token_budget: int = 120_000
    """单副本 KV cache 的 token 预算。超了就不许再接新请求。"""

    block_size: int = 16
    weight_bits: int = 16
    mfu: float = 0.5

    prefix_cache_blocks: int = 2048
    """前缀缓存最多占多少块。超了按 LRU 淘汰。"""

    admission_rate_per_s: float = 0.0
    """令牌桶速率，0 表示不限流。"""

    admission_burst: float = 0.0
    """令牌桶桶容量。"""

    slo_ttft_ms: float = 2000.0
    slo_tpot_ms: float = 100.0
    horizon_s: float = 600.0


@dataclass(frozen=True)
class ServeRequest:
    """一条进入服务的请求。

    ``segments`` 是"片段 key + token 数"的序列，模拟**共享系统提示词**的
    常见形态：第一个片段是所有请求共用的系统提示，后面才是各自的用户问题。

    这样构造请求，前缀缓存才有意义——真实服务里最常见的复用来源
    就是同一份 system prompt 加上不同的用户输入。
    """

    req_id: int
    arrival_s: float
    segments: tuple[tuple[str, int], ...]
    max_output_tokens: int
    tenant: str = "default"
    priority: int = 0
    deadline_s: float | None = None

    @property
    def prompt_tokens(self) -> int:
        """prompt 总 token 数。"""
        return sum(count for _, count in self.segments)

    def token_ids(self) -> list[int]:
        """把片段展开成确定的 token 序列。

        同一个片段 key 永远派生出同一串 token，所以"共享片段"
        在 token 层面真的共享，前缀哈希才命中。
        """
        tokens: list[int] = []
        for key, count in self.segments:
            tokens.extend(synthetic_tokens(key, count))
        return tokens

    def chain(self, block_size: int) -> list[tuple[tuple[int, int], int]]:
        """本请求的块级链式哈希。

        前 N 个块相同的两个请求，前 N 个节点键必然相同——
        这正是前缀缓存能共享 KV 块的原因（第 12 章）。
        """
        return block_chain(self.token_ids(), block_size)


def synthetic_tokens(prompt_key: str, count: int) -> list[int]:
    """从一个片段 key 派生出确定性的 token 序列。

    为什么不引入真实分词器：教程关心的是"前缀相同 → 缓存命中"这条因果链，
    而不是分词细节。只要同一个 key 派生同一串 token，缓存语义就是对的，
    而且完全离线可复现。

    用 SHA-256 做种子 + 线性同余递推：既确定又不依赖 Python 的 hash 随机化。
    """
    seed = int.from_bytes(hashlib.sha256(prompt_key.encode("utf-8")).digest()[:8], "big")
    tokens: list[int] = []
    for _ in range(count):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**64)
        tokens.append((seed >> 33) % 50_000)
    return tokens


# ---------------------------------------------------------------- 限流


@dataclass
class RateLimiter:
    """令牌桶限流器。

    为什么用令牌桶而不是漏桶：交互式服务的突发是真实存在的——
    用户点一下刷新可能瞬间来三个请求。令牌桶允许桶里有存量，
    所以突发能过，但长期平均速率仍然受控。

    漏桶会把突发削成匀速，对交互式服务反而更差。

    实现上有个容易踩的坑：用浮点累加信用时，
    ``0.1 × 10`` 这种乘积会得到 ``0.9999999999999998``，
    于是"信用满 1"的条件永远差一点点，表现为**放行率莫名其妙偏低**。
    所以比较时留一个极小的容差（:data:`_EPS`）。
    """

    rate_per_s: float
    burst: float
    tokens: float = 0.0
    last_refill_s: float = 0.0
    _initialized: bool = False

    _EPS: float = 1e-9
    """浮点比较容差。只影响"信用是否够 1 个 token"的判断，不影响速率本身。"""

    def allow(self, now: float) -> bool:
        """判断此刻能否放行一个请求。"""
        if not self._initialized:
            self._initialized = True
            self.last_refill_s = now
            self.tokens = self.burst

        if now > self.last_refill_s:
            self.tokens += (now - self.last_refill_s) * self.rate_per_s
            self.last_refill_s = now
        # 桶容量限制：超出部分直接丢弃，这就是 burst 的意义
        if self.tokens > self.burst:
            self.tokens = self.burst

        if self.tokens + self._EPS >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def available(self) -> float:
        """当前桶里剩多少令牌。"""
        return self.tokens


# ---------------------------------------------------------------- 副本


@dataclass
class Sequence:
    """一条正在服务（或刚服务完）的序列。"""

    request: ServeRequest
    prompt_tokens: int
    cached_tokens: int
    """本次从前缀缓存复用的 token 数。"""

    shared_blocks: list[int] = field(default_factory=list)
    own_blocks: list[int] = field(default_factory=list)
    produced: int = 0
    enqueue_s: float = 0.0
    first_token_s: float | None = None
    finish_s: float | None = None
    cache_hit: bool = False

    @property
    def context(self) -> int:
        """当前上下文长度。"""
        return self.prompt_tokens + self.produced


@dataclass
class Completion:
    """一条请求的完成情况。"""

    request: ServeRequest
    replica: str
    enqueue_s: float
    prefill_end_s: float
    finish_s: float
    ttft_s: float
    tpot_s: float
    cache_hit: bool

    @property
    def total_time_s(self) -> float:
        """端到端时延。"""
        return self.finish_s - self.request.arrival_s

    @property
    def output_tokens(self) -> int:
        """产出 token 数。"""
        return self.request.max_output_tokens

    def missed_ttft_slo(self, slo_ms: float) -> bool:
        """是否违反 TTFT 约定。"""
        return self.ttft_s * 1000 > slo_ms


class Replica:
    """一个推理副本：分页 KV + 前缀缓存 + 连续批处理调度。

    这是整个服务里最像真实推理引擎的部分。三件事在这里耦合：

    1. **连续批处理**——每轮 decode 之后立刻尝试补位，新请求不必等整批结束；
    2. **分页 KV**——块按需分配，前缀块共享（引用计数），最后一块的碎片是唯一浪费；
    3. **前缀缓存**——block 粒度哈希，命中则这部分 prefill 直接跳过。
    """

    def __init__(self, name: str, config: ServeConfig, device: GPUSpec, spec: ModelSpec) -> None:
        self.name = name
        self.config = config
        self.device = device
        self.spec = spec

        self.kv = PagedKVCache(block_size=config.block_size)
        """分页 KV 缓存的元数据。只记录块结构，不存真实向量。"""

        self.cache_hold: dict[int, int] = {}
        """块被前缀缓存持有的次数。引用计数减到 0 且不被缓存持有时才真正回收。"""

        self.block_index: dict[tuple[int, int], int] = {}
        """块级前缀树：节点键 → 块号。

        节点键是 ``(父链哈希, 本块哈希)``，所以**共享前缀的请求会命中同一批块**。
        这比"整段序列一个哈希"强得多——后者只能命中完全相同的请求。
        """

        self.node_lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        """前缀树的 LRU 顺序，用于容量淘汰。"""

        self.queue: deque[ServeRequest] = deque()
        self.active: list[Sequence] = []
        self.now = 0.0
        self.completions: list[Completion] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.reused_tokens = 0
        self.rejected_for_kv = 0

    # -- 队列 -------------------------------------------------------------

    def submit(self, request: ServeRequest) -> None:
        """接收一个请求进队列。"""
        self.queue.append(request)

    def is_idle(self) -> bool:
        """是否既没在服务也没排队。"""
        return not self.queue and not self.active

    def load(self) -> int:
        """负载（队列 + 运行中）。路由用它做负载均衡。"""
        return len(self.queue) + len(self.active)

    # -- 块生命周期 -------------------------------------------------------

    def _retain_block(self, block_id: int) -> None:
        """序列持有一个块。"""
        self.kv.refcount[block_id] = self.kv.refcount.get(block_id, 0) + 1

    def _release_block(self, block_id: int) -> None:
        """序列释放一个块；没人要也没有缓存持有时才回收。"""
        current = max(0, self.kv.refcount.get(block_id, 0) - 1)
        self.kv.refcount[block_id] = current
        if current == 0 and self.cache_hold.get(block_id, 0) == 0:
            self.kv.blocks[block_id] = 0
            self.kv.free_blocks.append(block_id)

    def _evict_prefix_if_needed(self) -> None:
        """前缀缓存超容量时按 LRU 淘汰**块**。

        淘汰到块粒度而不是整条链：被淘汰的块如果还有活跃序列在用，
        引用计数不为 0，就先不回收，只是不再作为新请求的缓存命中。
        这样既不会破坏正在跑的请求，又能控制缓存占用。
        """
        while len(self.block_index) > self.config.prefix_cache_blocks:
            if not self.node_lru:
                break
            oldest, _ = self.node_lru.popitem(last=False)
            block_id = self.block_index.pop(oldest, None)
            if block_id is None:
                continue
            hold = max(0, self.cache_hold.get(block_id, 0) - 1)
            self.cache_hold[block_id] = hold
            if hold == 0 and self.kv.refcount.get(block_id, 0) == 0:
                self.kv.blocks[block_id] = 0
                self.kv.free_blocks.append(block_id)

    # -- 前缀缓存 ---------------------------------------------------------

    def match_prefix(self, request: ServeRequest) -> int:
        """本副本能复用这个请求的前多少个块。

        从链头往后走，遇到第一个没命中的块就停——
        因为前缀缓存只能复用**连续的前缀**，中间断了一截后面就不算共享。
        """
        matched = 0
        for node_key, _ in request.chain(self.config.block_size):
            if node_key not in self.block_index:
                break
            matched += 1
        return matched

    def _lookup_prefix(self, request: ServeRequest) -> tuple[list[int], int]:
        """查前缀缓存，返回（命中的块列表, 复用的 token 数）。"""
        matched = self.match_prefix(request)
        if matched == 0:
            return [], 0
        chain = request.chain(self.config.block_size)
        blocks = [self.block_index[node_key] for node_key, _ in chain[:matched]]
        for node_key, _ in chain[:matched]:
            self.node_lru.move_to_end(node_key)
        cached_tokens = min(matched * self.config.block_size, request.prompt_tokens)
        return blocks, cached_tokens

    # -- 准入与调度 -------------------------------------------------------

    def _kv_reserved_tokens(self) -> int:
        """运行中序列预留的 KV token 数（prompt + 预留输出）。"""
        return sum(
            seq.prompt_tokens + seq.request.max_output_tokens for seq in self.active
        )

    def _has_kv_room(self) -> bool:
        """KV 预算是否还够。"""
        return self._kv_reserved_tokens() < self.config.kv_token_budget

    def _admit(self, request: ServeRequest) -> float:
        """把请求接进来做 prefill，返回本轮占用的时间。"""
        self.queue.remove(request)
        tokens = request.token_ids()
        chain = request.chain(self.config.block_size)

        cached_blocks, cached_tokens = self._lookup_prefix(request)
        shared_blocks: list[int] = []
        if cached_blocks:
            self.cache_hits += 1
            self.reused_tokens += cached_tokens
            for block_id in cached_blocks:
                self._retain_block(block_id)
            shared_blocks = cached_blocks
        else:
            self.cache_misses += 1

        # 复用之后的部分：新分配块，并登记进前缀树
        own_blocks: list[int] = []
        remaining_chain = chain[len(cached_blocks) :]
        for offset, (node_key, _) in enumerate(remaining_chain):
            block_id = self.kv.allocate()
            # 这一块实际装多少 token：除最后一个块外都是满的
            chunk = len(tokens) - (len(cached_blocks) + offset) * self.config.block_size
            self.kv.append_tokens(block_id, max(0, min(self.config.block_size, chunk)))
            self._retain_block(block_id)
            self.cache_hold[block_id] = self.cache_hold.get(block_id, 0) + 1
            self.block_index[node_key] = block_id
            self.node_lru[node_key] = None
            own_blocks.append(block_id)
        self._evict_prefix_if_needed()

        # 只对没命中缓存的那部分做 prefill —— 这就是前缀缓存省下的时间
        prefill_tokens = len(tokens) - cached_tokens
        prefill_s = prefill_time(
            self.device,
            self.spec,
            prefill_tokens,
            mfu=self.config.mfu,
            weight_bits=self.config.weight_bits,
        )

        self.active.append(
            Sequence(
                request=request,
                prompt_tokens=len(tokens),
                cached_tokens=cached_tokens,
                shared_blocks=shared_blocks,
                own_blocks=own_blocks,
                enqueue_s=self.now,
                cache_hit=bool(cached_blocks),
            )
        )
        return prefill_s

    def _decode_time(self, batch: list[Sequence]) -> float:
        """算本轮 decode 的耗时。

        batch 内并行，所以取**最慢**那条序列的时间：上下文最长的那条
        要读最多的 KV。这个"取 max"是第 7 章讲的连续批处理的代价来源。
        """
        if not batch:
            return 0.0
        max_context = max(seq.context for seq in batch)
        return decode_step_time(
            self.device,
            self.spec,
            len(batch),
            mfu=self.config.mfu,
            weight_bits=self.config.weight_bits,
            context_tokens=max_context,
        )

    def _iteration(self) -> None:
        """跑一轮：先补位，再 decode。

        顺序说明：真实引擎也是先处理 prefill 再 decode，
        一个 iteration 的时长取两者的**较大值**（它们共享同一套算力）。

        补位时只接**已经到达**的请求——没到的请求还在"路上"，
        提前接进来会让它的 TTFT 算成负数。
        """
        prefill_s = 0.0
        eligible = next((r for r in self.queue if r.arrival_s <= self.now), None)
        if (
            eligible is not None
            and len(self.active) < self.config.max_batch
            and self._has_kv_room()
        ):
            prefill_s = self._admit(eligible)

        batch = list(self.active)
        decode_s = self._decode_time(batch)
        self.now += max(prefill_s, decode_s)

        finished: list[Sequence] = []
        for seq in batch:
            seq.produced += 1
            if seq.produced == 1:
                seq.first_token_s = self.now
            if seq.produced >= seq.request.max_output_tokens:
                seq.finish_s = self.now
                finished.append(seq)

        for seq in finished:
            self._finish(seq)

    def _finish(self, seq: Sequence) -> None:
        """收尾一条序列：记完成情况、释放块、从活跃列表移除。"""
        first = seq.first_token_s if seq.first_token_s is not None else self.now
        produced = max(1, seq.produced)
        self.completions.append(
            Completion(
                request=seq.request,
                replica=self.name,
                enqueue_s=seq.enqueue_s,
                prefill_end_s=first,
                finish_s=self.now,
                ttft_s=first - seq.request.arrival_s,
                tpot_s=(self.now - first) / max(1, produced - 1),
                cache_hit=seq.cache_hit,
            )
        )
        for block_id in seq.shared_blocks + seq.own_blocks:
            self._release_block(block_id)
        self.active.remove(seq)

    def run_until_idle(self, horizon_s: float) -> None:
        """一直跑到队列和活跃列表都空。"""
        while (self.queue or self.active) and self.now < horizon_s:
            if not self.active and self.queue:
                next_arrival = self.queue[0].arrival_s
                if next_arrival > self.now:
                    self.now = next_arrival
            if not self.active and not self.queue:
                break
            self._iteration()

    # -- 指标 -------------------------------------------------------------

    def metrics(self) -> dict[str, float]:
        """这个副本的运行指标。"""
        per_token = kv_bytes_per_token(self.spec, 2)
        allocated_tokens = self.kv.used_blocks() * self.config.block_size
        budget_bytes = self.config.kv_token_budget * per_token
        return {
            "completed": float(len(self.completions)),
            "cache_hits": float(self.cache_hits),
            "cache_misses": float(self.cache_misses),
            "reused_tokens": float(self.reused_tokens),
            "blocks_allocated": float(self.kv.total_blocks()),
            "blocks_used": float(self.kv.used_blocks()),
            "fragmentation": self.kv.fragmentation(),
            "kv_utilization": (allocated_tokens * per_token / budget_bytes) if budget_bytes > 0 else 0.0,
            "busy_s": self.now,
        }


# ---------------------------------------------------------------- 网关与服务


class MiniServe:
    """推理服务门面：限流 + 路由 + 副本池。

    这三件事的真实分工：**限流在入口**（保护后面的卡），
    **路由决定请求落到哪个副本**（决定前缀缓存能不能命中），
    **副本池干活**。
    """

    def __init__(self, config: ServeConfig) -> None:
        self.config = config
        self.device = gpu(config.device_name)
        self.spec = model(config.model_name)
        self.replicas = [
            Replica(f"replica-{i}", config, self.device, self.spec)
            for i in range(config.replicas)
        ]
        self.limiter = (
            RateLimiter(config.admission_rate_per_s, config.admission_burst)
            if config.admission_rate_per_s > 0
            else None
        )
        self.accepted = 0
        self.rejected = 0

    # -- 路由 -------------------------------------------------------------

    def route(self, request: ServeRequest) -> Replica:
        """前缀亲和路由。

        第一步找**能复用这个前缀**的副本——这是前缀缓存能命中的前提。
        多个副本都能复用时，选负载最低的；没人能复用就按负载挑最闲的。

        注意亲和不等于"永远钉住"：持有前缀的副本全忙时，
        还是应该把请求放出去，而不是干等着缓存。
        """
        matches = [(replica.match_prefix(request), replica) for replica in self.replicas]
        best_match = max(count for count, _ in matches)
        if best_match > 0:
            candidates = [replica for count, replica in matches if count == best_match]
            return min(candidates, key=lambda r: r.load())
        return min(self.replicas, key=lambda r: r.load())

    # -- 主流程 -----------------------------------------------------------

    def run(self, requests: list[ServeRequest]) -> ServeReport:
        """服务一批请求，返回报告。"""
        for request in sorted(requests, key=lambda r: r.arrival_s):
            if self.limiter is not None and not self.limiter.allow(request.arrival_s):
                self.rejected += 1
                continue
            self.accepted += 1
            self.route(request).submit(request)

        for replica in self.replicas:
            replica.run_until_idle(self.config.horizon_s)

        return self.report(requests)

    def report(self, requests: list[ServeRequest]) -> ServeReport:
        """汇总所有副本的指标。"""
        completions = [c for replica in self.replicas for c in replica.completions]
        return ServeReport(
            config=self.config,
            device=self.device,
            spec=self.spec,
            completions=completions,
            replica_metrics=[replica.metrics() for replica in self.replicas],
            replica_names=[replica.name for replica in self.replicas],
            accepted=self.accepted,
            rejected=self.rejected,
            submitted=len(requests),
        )


@dataclass
class ServeReport:
    """服务报告：把指标算清楚并能打印成表。"""

    config: ServeConfig
    device: GPUSpec
    spec: ModelSpec
    completions: list[Completion]
    replica_metrics: list[dict[str, float]]
    replica_names: list[str]
    accepted: int
    rejected: int
    submitted: int

    # -- 指标 -------------------------------------------------------------

    def ttft_p(self, q: float) -> float:
        """TTFT 分位数（毫秒）。"""
        values = [c.ttft_s * 1000 for c in self.completions]
        return percentile(values, q)

    def tpot_p(self, q: float) -> float:
        """TPOT 分位数（毫秒）。"""
        values = [c.tpot_s * 1000 for c in self.completions]
        return percentile(values, q)

    def cache_hit_rate(self) -> float:
        """前缀缓存命中率。"""
        hits = sum(m["cache_hits"] for m in self.replica_metrics)
        misses = sum(m["cache_misses"] for m in self.replica_metrics)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def output_tokens_per_s(self) -> float:
        """整体输出吞吐（token/s）。

        分母用**墙钟跨度**：并行副本的墙钟是最大完成时刻，
        不是各副本时间之和。这一点第 15 章讲扩缩容时还要再用。
        """
        if not self.completions:
            return 0.0
        total_tokens = sum(c.output_tokens for c in self.completions)
        span = max(c.finish_s for c in self.completions) - min(
            c.request.arrival_s for c in self.completions
        )
        return total_tokens / max(span, 1e-9)

    def slo_violation_rate(self) -> float:
        """TTFT 违约率。"""
        if not self.completions:
            return 0.0
        violations = sum(
            1 for c in self.completions if c.missed_ttft_slo(self.config.slo_ttft_ms)
        )
        return violations / len(self.completions)

    def cost_per_million_tokens_usd(self) -> float:
        """每百万输出 token 的成本（美元）。

        口径：所有副本的**墙钟**成本 ÷ 产出 token 数。
        副本并行，所以墙钟取最大而不是求和。
        """
        if not self.completions:
            return 0.0
        total_tokens = sum(c.output_tokens for c in self.completions)
        span = max(c.finish_s for c in self.completions) - min(
            c.request.arrival_s for c in self.completions
        )
        hours = span / 3600.0
        cost = hours * self.config.replicas * self.device.price_per_hour_usd
        if total_tokens == 0:
            return 0.0
        return cost / (total_tokens / 1e6)

    # -- 打印 -------------------------------------------------------------

    def render(self) -> None:
        """把报告打印成可读表格。"""
        from ..labkit import print_table

        print()
        print("=" * 72)
        print(f"迷你推理服务报告 · {self.spec.name} on {self.device.name}")
        print("=" * 72)

        print()
        print("--- 请求概览 ---")
        print_table(
            ["指标", "值"],
            [
                ["提交请求", self.submitted],
                ["放行", self.accepted],
                ["限流拒绝", self.rejected],
                ["完成", len(self.completions)],
            ],
        )

        print()
        print("--- 时延与吞吐 ---")
        print_table(
            ["指标", "值"],
            [
                ["TTFT p50", f"{self.ttft_p(0.50):.1f}ms"],
                ["TTFT p99", f"{self.ttft_p(0.99):.1f}ms"],
                ["TPOT p50", f"{self.tpot_p(0.50):.2f}ms"],
                ["输出吞吐", f"{self.output_tokens_per_s():.1f} token/s"],
                ["前缀缓存命中率", f"{self.cache_hit_rate() * 100:.1f}%"],
                ["TTFT SLO 违约率", f"{self.slo_violation_rate() * 100:.1f}%"],
                ["每百万 token 成本", f"${self.cost_per_million_tokens_usd():.3f}"],
            ],
        )

        print()
        print("--- 副本状态 ---")
        rows = []
        for name, metrics in zip(self.replica_names, self.replica_metrics):
            rows.append(
                [
                    name,
                    int(metrics["completed"]),
                    int(metrics["cache_hits"]),
                    int(metrics["cache_misses"]),
                    f"{metrics['fragmentation'] * 100:.1f}%",
                    f"{metrics['kv_utilization'] * 100:.1f}%",
                ]
            )
        print_table(
            ["副本", "完成", "缓存命中", "缓存未命中", "KV 碎片率", "KV 利用率"],
            rows,
        )
