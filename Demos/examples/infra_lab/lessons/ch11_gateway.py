"""第 11 章实验：网关——路由、限流与租户隔离。

这个模块解决什么问题：网关是推理服务对外的唯一入口，它要在"请求进来"到"请求进引擎"之间
做三件会直接决定成本和 SLA 的决策：路由到哪个副本、按什么规则限流、租户之间怎么隔离。
本章把这三件事变成可验证的断言：

1. **前缀亲和 vs 随机路由**：同一份 KV 预算在两种路由下的前缀缓存命中率。
   随机路由让每个副本都缓存全部前缀，等于把缓存浪费了 N 倍；
   前缀亲和路由让每个前缀只存一份，命中率不随副本数塌。
2. **限流算法对比**：令牌桶、漏桶、滑动窗口对同一份突发流量的整形结果。
   令牌桶允许桶容量内的突发，漏桶只按固定速率吐，滑动窗口做硬窗口封顶。
3. **租户隔离**：一个租户吃掉自己的配额，不能影响另一个租户的放行数。
4. **副本健康与负载均衡**：副本挂掉后流量必须转移到健康副本，
   剩余副本的利用率随之上升——这个数用 labkit 的 ``load_balancing_delta`` 算。

全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..labkit import (
    PagedKVCache,
    banner,
    ensure,
    fmt_bytes,
    gpu,
    load_balancing_delta,
    model,
    prefix_hash,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
SPEC = model("qwen2.5-7b")

BLOCK_SIZE = 16
"""前缀哈希与分页缓存共用的 block 粒度。真实引擎常见 16/32/64。"""

PREFIX_COUNT = 16
"""工作集里有多少个不同的 prompt 前缀。可以理解成 16 个不同的系统提示/少样本模板。"""

PREFIX_TOKENS = 128
"""每个前缀的 token 数。"""

REQUEST_COUNT = 4000
"""压测请求数。"""

FLEET_BUDGET_BLOCKS = PREFIX_COUNT * (PREFIX_TOKENS // BLOCK_SIZE)
"""整个机队的 KV 前缀预算（block 数）。

刻意设成"刚好装下一份完整工作集"：这样才能看出**副本越多、每份副本能存的副本数越少**。
机队总预算固定时，副本数 N 上升 → 每个副本的预算 1/N 下降 → 随机路由的命中率必然塌。
"""

BURST_SIZE = 100
"""突发流量规模。"""

BUCKET_CAPACITY = 20
"""令牌桶容量：允许一次放行的突发量。"""

REFILL_RATE = 10.0
"""令牌桶补充速率（个/秒），也是稳态准入速率。"""

STEADY_SECONDS = 10
"""稳态流量持续时长。"""

STEADY_RATE = 10.0
"""稳态到达速率（个/秒）。"""


@dataclass(frozen=True)
class GatewayRequest:
    """一条穿过网关的请求。"""

    req_id: int
    prefix_id: int
    """属于工作集里的哪个前缀。"""

    prefix_tokens: tuple[int, ...]
    """共享前缀的 token。同一 ``prefix_id`` 的请求前缀完全一样。"""

    tenant: str = "default"
    """租户标识。"""


def build_workload(
    count: int = REQUEST_COUNT,
    prefixes: int = PREFIX_COUNT,
    tokens: int = PREFIX_TOKENS,
    seed: int = 20251118,
) -> list[GatewayRequest]:
    """造一份请求流：前缀均匀分散，尾部是各不相同的私有内容。

    用固定种子保证实验可复现。
    """
    rng = random.Random(seed)
    requests: list[GatewayRequest] = []
    for req_id in range(count):
        prefix_id = rng.randrange(prefixes)
        prefix = tuple(prefix_id * 1000 + i for i in range(tokens))
        requests.append(
            GatewayRequest(
                req_id=req_id,
                prefix_id=prefix_id,
                prefix_tokens=prefix,
                tenant=f"tenant-{prefix_id % 4}",
            )
        )
    return requests


class ReplicaPrefixCache:
    """单个副本上的前缀缓存：按 block 预算 + LRU 淘汰。

    块分配与前缀登记直接用 labkit 的 :class:`PagedKVCache`，
    网关侧要额外管的两件事——容量上限与淘汰顺序——放在这一层。
    """

    def __init__(self, capacity_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        self.cache = PagedKVCache(block_size=block_size)
        self.capacity_blocks = capacity_blocks
        self.lru: list[tuple[int, ...]] = []
        """哈希链 → 访问顺序，最新的在末尾。"""
        self.block_cost: dict[tuple[int, ...], int] = {}
        """哈希链 → 占用的 block 数。"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def used_blocks(self) -> int:
        """当前缓存占用的 block 数。"""
        return sum(self.block_cost.values())

    def lookup(self, key: tuple[int, ...]) -> bool:
        """查前缀是否还在缓存里，命中时顺便更新 LRU 顺序。"""
        if self.cache.lookup_prefix(key) is None:
            self.misses += 1
            return False
        self.hits += 1
        self.lru.remove(key)
        self.lru.append(key)
        return True

    def store(self, key: tuple[int, ...], tokens: tuple[int, ...]) -> None:
        """把一段前缀写进缓存，必要时按 LRU 淘汰旧条目。

        淘汰时会真的把 block 还给 labkit 的分页缓存，所以显存账是连着的。
        """
        needed = max(1, -(-len(tokens) // self.cache.block_size))
        block_ids: list[int] = []
        for index in range(needed):
            block = self.cache.allocate()
            # 最后一个块可能没填满，剩下的 token 数如实写进去
            self.cache.append_tokens(block, min(self.cache.block_size, len(tokens) - index * self.cache.block_size))
            block_ids.append(block)
        self.cache.cache_prefix(key, block_ids)
        self.block_cost[key] = needed
        self.lru.append(key)
        self._evict_until_fits()

    def _evict_until_fits(self) -> None:
        """按 LRU 把最久没用的前缀踢出去，直到回到容量上限内。"""
        while self.used_blocks > self.capacity_blocks and len(self.lru) > 1:
            victim = self.lru.pop(0)
            for block in self.cache.lookup_prefix(victim) or []:
                self.cache.free(block)
            del self.cache.prefix_cache[victim]
            self.block_cost.pop(victim, None)
            self.evictions += 1


def simulate_routing(
    replicas: int,
    requests: list[GatewayRequest],
    *,
    strategy: str,
    fleet_budget_blocks: int = FLEET_BUDGET_BLOCKS,
    seed: int = 7,
) -> dict[str, float]:
    """跑一遍路由策略，统计前缀缓存命中率与缓存浪费。

    ``strategy`` 取 ``"random"`` 或 ``"affinity"``。
    机队 KV 预算固定，副本数越多，每个副本分到的预算越少——
    这正是"扩副本反而让缓存命中率塌掉"的机制。
    """
    per_replica_budget = max(1, fleet_budget_blocks // replicas)
    caches = [ReplicaPrefixCache(per_replica_budget) for _ in range(replicas)]
    rng = random.Random(seed)
    for request in requests:
        if strategy == "affinity":
            # 前缀亲和：同一个前缀永远去同一个副本（生产里用前缀哈希做一致性哈希）
            index = request.prefix_id % replicas
        elif strategy == "random":
            index = rng.randrange(replicas)
        else:
            raise ValueError(f"未知路由策略：{strategy}")
        key = prefix_hash(request.prefix_tokens, granularity=BLOCK_SIZE)
        cache = caches[index]
        if not cache.lookup(key):
            cache.store(key, request.prefix_tokens)

    hits = sum(c.hits for c in caches)
    misses = sum(c.misses for c in caches)
    total = hits + misses
    # 统计"同一前缀在几个副本上各存了一份"——这才是随机路由真正的浪费
    copies: dict[tuple[int, ...], int] = {}
    for cache in caches:
        for key in cache.block_cost:
            copies[key] = copies.get(key, 0) + 1
    duplicated_copies = sum(count - 1 for count in copies.values())
    stored_copies = sum(copies.values())
    return {
        "hit_rate": hits / total if total else 0.0,
        "hits": float(hits),
        "misses": float(misses),
        "used_blocks": float(sum(c.used_blocks for c in caches)),
        "stored_copies": float(stored_copies),
        "resident_prefixes": float(len(copies)),
        "duplicate_copies": float(duplicated_copies),
        "evictions": float(sum(c.evictions for c in caches)),
        "per_replica_budget": float(per_replica_budget),
    }


def check_prefix_affinity_beats_random() -> None:
    """同一份 KV 预算下，前缀亲和路由的命中率不随副本数塌陷。"""
    section("1. 前缀亲和 vs 随机路由：前缀缓存命中率")
    requests = build_workload()
    rows: list[list[str]] = []
    results: dict[int, dict[str, float]] = {}
    for replicas in (1, 2, 4, 8, 16):
        affinity = simulate_routing(replicas, requests, strategy="affinity")
        random_route = simulate_routing(replicas, requests, strategy="random")
        results[replicas] = {"affinity": affinity, "random": random_route}
        rows.append(
            [
                str(replicas),
                f"{affinity['per_replica_budget']:.0f}",
                f"{affinity['hit_rate'] * 100:.1f}%",
                f"{random_route['hit_rate'] * 100:.1f}%",
                f"{affinity['resident_prefixes']:.0f}",
                f"{random_route['resident_prefixes']:.0f}",
                f"{random_route['duplicate_copies']:.0f}",
            ]
        )
    print_table(
        [
            "副本数",
            "每副本 block 预算",
            "亲和命中率",
            "随机命中率",
            "亲和常驻前缀",
            "随机常驻前缀",
            "随机重复份数",
        ],
        rows,
    )

    for replicas, item in results.items():
        ensure(
            item["affinity"]["hit_rate"] >= 0.95,
            f"{replicas} 副本时亲和路由命中率应 ≥95%，实际 {item['affinity']['hit_rate']:.3f}",
        )
        ensure(
            item["affinity"]["duplicate_copies"] == 0,
            f"{replicas} 副本时亲和路由不应产生重复前缀副本",
        )
        ensure(
            item["affinity"]["resident_prefixes"] == PREFIX_COUNT,
            f"{replicas} 副本时亲和路由应常驻全部 {PREFIX_COUNT} 个前缀，"
            f"实际 {item['affinity']['resident_prefixes']:.0f}",
        )
    # 随机路由：副本越多，每份副本能存的前缀越少 → 命中率塌
    ensure(
        results[16]["random"]["hit_rate"] < 0.2,
        f"16 副本时随机路由命中率应 <20%，实际 {results[16]['random']['hit_rate']:.3f}",
    )
    ensure(
        results[16]["random"]["hit_rate"] < results[2]["random"]["hit_rate"] * 0.5,
        "随机路由的命中率应随副本数上升而明显下滑",
    )
    ensure(
        results[16]["affinity"]["hit_rate"] > results[16]["random"]["hit_rate"] * 3,
        "16 副本时亲和路由应显著优于随机路由",
    )
    rule(
        f"随机路由在 16 副本下命中率只剩 {results[16]['random']['hit_rate'] * 100:.1f}%，"
        f"亲和路由稳在 {results[16]['affinity']['hit_rate'] * 100:.1f}%"
    )


def check_cache_memory_cost() -> None:
    """命中率要换算成显存账：省下的 prefill 对应多少 GiB 的 KV。"""
    section("2. 前缀缓存省下的显存账")
    requests = build_workload()
    affinity = simulate_routing(4, requests, strategy="affinity")
    total_tokens = sum(len(r.prefix_tokens) for r in requests)
    # 亲和路由下每个前缀只在属主副本上冷启动一次，其余请求全部命中
    cold_tokens = PREFIX_COUNT * PREFIX_TOKENS
    saved_tokens = total_tokens - cold_tokens
    saved_bytes = saved_tokens * SPEC.kv_bytes_per_token(2)
    rule(
        f"{len(requests)} 条请求、{PREFIX_COUNT} 个共享前缀："
        f"亲和路由命中 {affinity['hits']:.0f} 次，省下 {saved_tokens} 个 token 的 prefill，"
        f"相当于 {fmt_bytes(saved_bytes)} 的 KV 不用重算"
    )
    ensure(
        saved_tokens > 0,
        "共享前缀的请求流必须能省下 prefill，否则实验没意义",
    )
    ensure(
        saved_tokens < total_tokens,
        "省下的 token 必须小于总 token 数，否则说明口径算错了",
    )
    per_token_kib = SPEC.kv_bytes_per_token(2) / 1024
    rule(
        f"Qwen2.5-7B 每 token KV = {SPEC.kv_bytes_per_token(2)} 字节"
        f"（{per_token_kib:.0f} KiB），GQA {SPEC.num_kv_heads} 头把 KV 摊薄了"
    )


@dataclass
class TokenBucket:
    """令牌桶：以固定速率补充令牌，桶满即停。

    允许桶容量内的突发，这是它和漏桶的根本区别。
    """

    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill_s = 0.0

    def _refill(self, now_s: float) -> None:
        """按流逝时间补充令牌。"""
        elapsed = max(0.0, now_s - self.last_refill_s)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill_s = now_s

    def allow(self, now_s: float, cost: float = 1.0) -> bool:
        """尝试消耗令牌放行一个请求。"""
        self._refill(now_s)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


@dataclass
class LeakyBucket:
    """漏桶：出口严格按固定速率吐，请求先进队列。

    它不丢请求，只把突发摊平成匀速；队列无限长时延迟会线性增长。
    """

    rate: float
    next_free_s: float = 0.0

    def admit_at(self, now_s: float) -> float:
        """返回该请求真正被放行的时刻。"""
        slot = max(now_s, self.next_free_s)
        self.next_free_s = slot + 1.0 / self.rate
        return slot


@dataclass
class SlidingWindowCounter:
    """滑动窗口：任意连续 ``window_s`` 秒内最多放行 ``limit`` 个请求。

    窗口是严格滑动的，所以既没有令牌桶那种"补充后突然再放一批"的锯齿，
    也不会像漏桶那样把请求无限延后。
    """

    limit: int
    window_s: float
    stamps: list[float] = field(default_factory=list)

    def allow(self, now_s: float) -> bool:
        """尝试放行一个请求。"""
        cutoff = now_s - self.window_s
        self.stamps = [t for t in self.stamps if t > cutoff]
        if len(self.stamps) >= self.limit:
            return False
        self.stamps.append(now_s)
        return True


def burst_trace() -> list[float]:
    """造一条流量曲线：开头一次突发，之后是恒定速率。"""
    trace = [0.0] * BURST_SIZE
    step = 1.0 / STEADY_RATE
    for i in range(STEADY_SECONDS * int(STEADY_RATE)):
        trace.append(1.0 + i * step)
    return trace


def check_rate_limit_shaping() -> None:
    """令牌桶把突发削平，稳态速率与配置一致。"""
    section("3. 限流：令牌桶 / 漏桶 / 滑动窗口")
    trace = burst_trace()

    bucket = TokenBucket(capacity=BUCKET_CAPACITY, refill_rate=REFILL_RATE)
    admitted_at: list[float] = []
    rejected = 0
    for t in trace:
        if bucket.allow(t):
            admitted_at.append(t)
        else:
            rejected += 1

    immediate = sum(1 for t in admitted_at if t <= 0.0)
    ensure(
        immediate <= BUCKET_CAPACITY,
        f"桶容量内的突发上限是 {BUCKET_CAPACITY}，实际瞬间放行 {immediate}",
    )
    ensure(
        immediate == BUCKET_CAPACITY,
        f"桶初始是满的，应该瞬间放行 {BUCKET_CAPACITY} 个，实际 {immediate}",
    )
    ensure(
        rejected > 0,
        "突发流量必须有一部分被限流削掉，否则说明限流没生效",
    )

    # 稳态速率：只统计 t ≥ 1s 之后放行的请求
    steady = [t for t in admitted_at if t >= 1.0]
    span = trace[-1] - 1.0
    steady_rate = len(steady) / span if span > 0 else 0.0
    ensure(
        abs(steady_rate - REFILL_RATE) < 0.5,
        f"稳态放行速率应接近 {REFILL_RATE}/s，实际 {steady_rate:.2f}/s",
    )
    rule(
        f"令牌桶：瞬间放行 {immediate} 个（桶容量 {BUCKET_CAPACITY}），"
        f"限流 {rejected} 个，稳态速率 {steady_rate:.2f}/s"
    )

    # 三种算法对同一份流量的对比
    leaky = LeakyBucket(rate=REFILL_RATE)
    leaky_slots = [leaky.admit_at(t) for t in trace]
    leaky_gap = min(
        (b - a for a, b in zip(leaky_slots, leaky_slots[1:]) if b > a),
        default=1.0,
    )
    ensure(
        leaky_gap >= 1.0 / REFILL_RATE - 1e-9,
        f"漏桶的出流间隔不能小于 1/{REFILL_RATE}s，实际 {leaky_gap:.4f}s",
    )

    window = SlidingWindowCounter(limit=int(REFILL_RATE * 1.0), window_s=1.0)
    window_admitted = sum(1 for t in trace if window.allow(t))
    # 窗口是严格滑动的：任意连续 1 秒内最多 limit 个。
    # 整段流量的放行数上界是 limit × (时长 / 窗口 + 1)，多出来的那一份是窗口边界的重叠。
    overlap = int(REFILL_RATE * 1.0)
    ensure(
        window_admitted <= overlap * (STEADY_SECONDS + 1),
        f"滑动窗口放行数 {window_admitted} 超过 {overlap}/s 的硬上限",
    )
    ensure(
        window_admitted >= overlap * STEADY_SECONDS,
        f"滑动窗口放行数 {window_admitted} 不应低于稳态速率下的 {overlap * STEADY_SECONDS}",
    )

    # 滑动窗口的瞬时放行量：直接数 t=0 时刻被放行的个数
    window_probe = SlidingWindowCounter(limit=overlap, window_s=1.0)
    window_burst = sum(1 for t in trace[:BURST_SIZE] if window_probe.allow(t))
    ensure(
        window_burst == overlap,
        f"滑动窗口在突发瞬间最多放行 {overlap} 个，实际 {window_burst}",
    )

    print_table(
        ["算法", "总放行", "被拒", "瞬间突发", "出流间隔"],
        [
            [
                "令牌桶",
                str(len(admitted_at)),
                str(rejected),
                str(immediate),
                f"≈{1.0 / REFILL_RATE:.3f}s",
            ],
            [
                "漏桶",
                str(len(leaky_slots)),
                "0（不丢，延后）",
                "0",
                f"{leaky_gap:.3f}s",
            ],
            [
                "滑动窗口",
                str(window_admitted),
                str(len(trace) - window_admitted),
                str(window_burst),
                "锯齿",
            ],
        ],
    )
    rule("令牌桶允许桶容量内的突发；漏桶把突发摊平但不丢；滑动窗口做硬窗口封顶")


def check_tenant_isolation() -> None:
    """租户之间各用各的令牌桶，一个租户爆量不影响另一个。"""
    section("4. 租户隔离：配额不能互相偷")
    shared = TokenBucket(capacity=BUCKET_CAPACITY, refill_rate=REFILL_RATE)
    tenant_buckets = {
        "tenant-a": TokenBucket(capacity=BUCKET_CAPACITY, refill_rate=REFILL_RATE),
        "tenant-b": TokenBucket(capacity=BUCKET_CAPACITY, refill_rate=REFILL_RATE),
    }
    burst = [0.0] * 50
    admitted: dict[str, int] = {"tenant-a": 0, "tenant-b": 0, "shared": 0}
    for t in burst:
        if tenant_buckets["tenant-a"].allow(t):
            admitted["tenant-a"] += 1
    for t in burst:
        if tenant_buckets["tenant-b"].allow(t):
            admitted["tenant-b"] += 1
    for t in burst:
        if shared.allow(t):
            admitted["shared"] += 1

    ensure(
        admitted["tenant-a"] == admitted["tenant-b"],
        f"两个租户遭遇同样的突发，放行数应相同，实际 {admitted}",
    )
    ensure(
        admitted["tenant-a"] == BUCKET_CAPACITY,
        f"租户独立配额下应放行满桶容量 {BUCKET_CAPACITY}，实际 {admitted['tenant-a']}",
    )
    ensure(
        admitted["shared"] == BUCKET_CAPACITY,
        f"共享配额下 50 个请求只能放行 {BUCKET_CAPACITY} 个，实际 {admitted['shared']}",
    )
    rule(
        f"独立配额：tenant-a/b 各放行 {admitted['tenant-a']} 个；"
        f"共享配额：50 个请求只放行 {admitted['shared']} 个"
    )


def check_replica_health_and_balance() -> None:
    """副本挂掉后流量要转移到健康副本，剩余副本的利用率随之上升。"""
    section("5. 副本健康与负载均衡")
    qps = 0.8
    healthy = load_balancing_delta(
        DEVICE, SPEC, replicas=2, qps=qps, avg_prompt_tokens=500, avg_output_tokens=200
    )
    degraded = load_balancing_delta(
        DEVICE, SPEC, replicas=1, qps=qps, avg_prompt_tokens=500, avg_output_tokens=200
    )
    enough = load_balancing_delta(
        DEVICE, SPEC, replicas=4, qps=qps, avg_prompt_tokens=500, avg_output_tokens=200
    )
    print_table(
        ["副本状态", "副本数", "单机容量(单流)", "集群容量", "利用率"],
        [
            [
                "全健康",
                "2",
                f"{healthy['capacity_per_replica']:.4f}/s",
                f"{healthy['total_capacity']:.4f}/s",
                f"{healthy['utilization'] * 100:.1f}%",
            ],
            [
                "挂掉一个",
                "1",
                f"{degraded['capacity_per_replica']:.4f}/s",
                f"{degraded['total_capacity']:.4f}/s",
                f"{degraded['utilization'] * 100:.1f}%",
            ],
            [
                "扩到 4 个",
                "4",
                f"{enough['capacity_per_replica']:.4f}/s",
                f"{enough['total_capacity']:.4f}/s",
                f"{enough['utilization'] * 100:.1f}%",
            ],
        ],
    )
    ensure(
        degraded["utilization"] > healthy["utilization"],
        "副本挂掉后剩余副本的利用率必须上升",
    )
    ensure(
        healthy["utilization"] < 1.0,
        f"2 个健康副本扛 {qps} QPS 不应过载，实际利用率 {healthy['utilization']:.3f}",
    )
    ensure(
        degraded["utilization"] > 1.0,
        f"挂掉一个后单副本应被打爆（需要立刻扩容），实际利用率 {degraded['utilization']:.3f}",
    )
    ensure(
        enough["utilization"] < healthy["utilization"],
        "扩副本必须降低单机利用率",
    )
    ensure(
        healthy["replicas_needed"] <= 2,
        f"按负载均衡口径，{qps} QPS 用 2 个副本应够，实际需要 {healthy['replicas_needed']:.2f}",
    )
    rule(
        f"单请求占用 {healthy['per_request_s'] * 1000:.1f}ms；"
        f"2 副本扛 {qps} QPS 时利用率 {healthy['utilization'] * 100:.1f}%，"
        f"挂一个后升到 {degraded['utilization'] * 100:.1f}%"
    )


def run() -> None:
    """跑完第 11 章的全部实验。"""
    banner("第 11 章 · 网关：路由、限流与租户隔离")
    rule(
        f"口径：{DEVICE.name} / {SPEC.name} / block={BLOCK_SIZE} / "
        f"工作集 {PREFIX_COUNT} 个前缀 × {PREFIX_TOKENS} tokens / "
        f"机队 KV 预算 {FLEET_BUDGET_BLOCKS} blocks"
    )
    check_prefix_affinity_beats_random()
    check_cache_memory_cost()
    check_rate_limit_shaping()
    check_tenant_isolation()
    check_replica_health_and_balance()
    print()
    rule(
        "结论：随机路由让每个副本都缓存一份同样的前缀，副本越多缓存越碎、命中率越塌；"
        "前缀亲和路由让前缀只存一份，命中率稳住。限流用令牌桶（允许桶内突发），"
        "租户配额必须各算各的"
    )
