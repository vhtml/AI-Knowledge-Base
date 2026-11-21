"""第 14 章实验：可观测性与延迟归因。

这个模块解决什么问题：线上说"延迟高"不是一个可执行的结论——
你必须能回答三个问题：**慢在哪一段**、**是所有请求都慢还是只有尾部慢**、
**这些数字是谁数的**。本章把三个问题变成可跑的断言：

1. **三段分解定位瓶颈**——同样是 TTFT 飙高，排队主导、prefill 主导、
   decode 主导三个场景的责任人完全不同（调度层 / 算力 / 带宽），
   解法也完全不同。分不清段就动手，只会改错地方。
2. **p99 长尾可以逐条归因**——混合长短请求 + 突发到达能把 p99/p50
   从 1.33 拉到 6.15；而长尾的来源能在段里指认：被长 prompt 阻塞的请求，
   TTFT 里排队段占九成，计算段只占一成。
3. **均值会骗人，采样率会伪造 p99**——最近秩法下样本量 ≤ 100 时
   "p99"就是样本最大值，所以 1% 采样估出来的 p99 不是分位数而是极值；
   换一个采样相位，同一个窗口的估计就摆动四成。
4. **计时起点决定 SLA 结论**——同一个 SLA，引擎侧报 0% 违约、
   客户端侧报 50% 违约，差的 1.95 秒全在引擎之外。
5. **抢占重算是被低估的长尾源**——引擎不抢占就没事，一旦抢占，
   同一段 prefill 要付两次，长 prompt 直接把 TTFT 打到秒级。

所有算账一律调用 labkit（``EngineSimulator`` / ``prefill_time`` /
``percentile`` / ``kv_gib`` / ``block_chain``），口径与前面各章一致。
全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..labkit import (
    Completion,
    EngineSimulator,
    GPUSpec,
    ModelSpec,
    Request,
    banner,
    block_chain,
    ensure,
    fmt_gib,
    fmt_ms,
    gpu,
    kv_gib,
    model,
    percentile,
    prefill_time,
    print_table,
    rule,
    section,
)

DEVICE: GPUSpec = gpu("h100_sxm")
SPEC: ModelSpec = model("qwen2.5-7b")

MFU = 0.5
"""可达算力/带宽占峰值的比例，与前面各章一致。"""

MAX_BATCH = 32
"""引擎最大并发序列数。"""

KV_TOKEN_BUDGET = 100_000
"""KV 预留预算（token）。三段分解、尾部分析、采样分析共用同一份引擎配置，
差异只能归因于负载，不能归因于配置。"""

TTFT_BUDGET_S = 1.0
"""TTFT 常规约定：到达后 1 秒内出首 token（与 labkit 的 ``ttft_deadline`` 同口径）。"""

BLOCK_SIZE = 16
"""前缀哈希的块粒度，与第 12、13 章一致。"""

SHARED_PREFIX_TOKENS = 64
"""每条请求共享的系统提示长度（token），用来算前缀缓存的两种命中率口径。"""

# ---------------------------------------------------------------- 三段分解场景

SHORT_PROMPT_TOKENS = 128
SHORT_OUTPUT_TOKENS = 32
LONG_PROMPT_TOKENS = 8192
LONG_OUTPUT_TOKENS = 64

COMPUTE_COUNT = 8
"""算力瓶颈场景的请求数。"""

MEMORY_COUNT = 8
"""带宽瓶颈场景的请求数。"""

OVERLOAD_COUNT = 200
"""排队瓶颈场景的请求数。"""

OVERLOAD_GAP_S = 0.002
"""排队瓶颈场景的到达间隔：500 QPS，远超单卡对短请求的容量。"""

# ---------------------------------------------------------------- 尾部分析场景

STREAM_COUNT = 300
"""尾部分析里短请求的条数（50 QPS 稳态流）。"""

STREAM_GAP_S = 0.02
"""尾部分析里短请求的到达间隔。"""

TAIL_LONG_COUNT = 5
"""尾部分析里长请求的条数。"""

TAIL_LONG_ARRIVAL_S = 1.0
"""尾部分析里长请求的到达时刻。"""

SPREAD_LONG_GAP_S = 1.0
"""长请求**分散**到达时的间隔（对照组的唯一变量）。"""

# ---------------------------------------------------------------- 采样口径

SAMPLE_STRIDE = 100
"""1% 系统性采样的步长：每 100 条取 1 条。"""

SAMPLE_PHASES = 100
"""采样相位数：改变起点，看同一个窗口能估出多少种 p99。"""

# ---------------------------------------------------------------- 网关口径

GATEWAY_REQUESTS = 40
"""突发到达网关的请求数。"""

GATEWAY_SERVICE_S = 0.05
"""网关是单线程串行处理器，每条请求处理 50ms。"""

CLIENT_NETWORK_S = 0.005
"""单程网络时延 5ms，请求去程与响应回程各算一次。"""


@dataclass(frozen=True)
class SegmentSplit:
    """一批请求的时延三段分解。

    排队段 = ``enqueue_s − arrival_s``，归调度层；
    prefill 段 = ``prefill_end_s − enqueue_s``，归执行层的算力；
    decode 段 = ``finish_s − prefill_end_s``，归执行层的带宽。
    """

    count: int
    queue_s: float
    prefill_s: float
    decode_s: float

    @property
    def total_s(self) -> float:
        """三段之和，等于端到端时延的总和。"""
        return self.queue_s + self.prefill_s + self.decode_s

    @property
    def queue_share(self) -> float:
        """排队段占比。"""
        return self.queue_s / self.total_s if self.total_s > 0 else 0.0

    @property
    def prefill_share(self) -> float:
        """prefill 段占比。"""
        return self.prefill_s / self.total_s if self.total_s > 0 else 0.0

    @property
    def decode_share(self) -> float:
        """decode 段占比。"""
        return self.decode_s / self.total_s if self.total_s > 0 else 0.0


def split_segments(completions: Sequence[Completion]) -> SegmentSplit:
    """把一批完成情况拆成排队 / prefill / decode 三段。"""
    queue = sum(c.queue_time_s for c in completions)
    prefill = sum(c.prefill_end_s - c.enqueue_s for c in completions)
    decode = sum(c.finish_s - c.prefill_end_s for c in completions)
    return SegmentSplit(
        count=len(completions),
        queue_s=queue,
        prefill_s=prefill,
        decode_s=decode,
    )


def run_engine(requests: Sequence[Request]) -> tuple[list[Completion], dict[str, float]]:
    """用全课统一的引擎配置跑一批请求，返回完成情况与指标表。"""
    engine = EngineSimulator(
        DEVICE,
        SPEC,
        max_batch=MAX_BATCH,
        kv_token_budget=KV_TOKEN_BUDGET,
        mfu=MFU,
    )
    completions = engine.run(requests, horizon=600)
    return completions, engine.metrics()


def ttft_list(completions: Sequence[Completion]) -> list[float]:
    """按到达顺序取出 TTFT 列表。

    必须按到达顺序而不是按大小排——采样是对**流量**采样，
    先排序再采样等于按分位数采样，会把采样误差伪装成统计量。
    """
    ordered = sorted(completions, key=lambda c: c.request.req_id)
    return [c.ttft_s for c in ordered]


def prefix_hit_stats(
    shared_tokens: int,
    unique_tokens: int,
    requests: int,
) -> tuple[float, float]:
    """前缀缓存的两种命中率口径：请求级 与 prefill token 级。

    返回 ``(请求级命中率, prefill token 节省率)``。

    用 labkit 的块级链式哈希判定"这条请求的前缀有没有命中"，
    口径与第 12、13 章一致。之所以要算两个口径：
    **"命中率 99%"这句话如果不说明分母，是没有信息量的**——
    请求级 99% 只说明"大部分请求蹭到了前缀"，不等于省下了 99% 的 prefill。
    """
    seen: set[tuple[tuple[int, int], int]] = set()
    hits = 0
    saved_blocks = 0
    total_blocks = 0
    for index in range(requests):
        # 共享前缀每条请求完全一样，独有段每条不同
        shared = tuple(range(1, shared_tokens + 1))
        unique = tuple(
            shared_tokens + 1 + index * 1000 + offset for offset in range(unique_tokens)
        )
        chain = block_chain(shared + unique, block_size=BLOCK_SIZE)
        matched = 0
        for node_key, _chain_hash in chain:
            if node_key in seen:
                matched += 1
            else:
                break
        for node_key, _chain_hash in chain:
            seen.add(node_key)
        if matched > 0:
            hits += 1
        saved_blocks += matched
        total_blocks += len(chain)
    request_hit = hits / requests if requests else 0.0
    token_saving = saved_blocks / total_blocks if total_blocks else 0.0
    return request_hit, token_saving


def merged_busy_s(completions: Sequence[Completion]) -> float:
    """引擎"有请求在跑"的总时长。

    必须先合并区间再求和——多条请求并发在跑时，
    它们的占用区间是重叠的，直接相加会把同一段时间算好几遍，
    算出来的"占用率"可以超过 100%（并发 20 时能到 2000%）。
    """
    intervals = sorted(
        (completion.enqueue_s, completion.finish_s) for completion in completions
    )
    merged = 0.0
    current_start = 0.0
    current_end = 0.0
    for start, end in intervals:
        if start > current_end:
            merged += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    merged += current_end - current_start
    return merged


def tail_wait_share(ordered: Sequence[float]) -> tuple[float, int]:
    """最慢 5% 的请求占总等待时间的比例，以及参与统计的请求条数。

    返回比例而不是绝对值，是因为"尾部贡献了多少等待时间"才是
    判断长尾值不值得治的依据：占比接近 5% 说明大家一样慢，
    占比远高于 5% 说明少数请求在拖后腿。
    """
    count = max(1, int(0.05 * len(ordered)))
    tail = ordered[-count:]
    total = sum(ordered)
    return (sum(tail) / total if total > 0 else 0.0), count


# ---------------------------------------------------------------- 判断一：三段分解


def _workload(
    key: str,
    title: str,
    prompt_tokens: int,
    output_tokens: int,
    count: int,
    gap_s: float,
) -> list[Request]:
    """造一批等重量的请求。

    三个场景只改"负载形状"，引擎配置完全一致——这样差异才能归因到负载。
    """
    return [
        Request(
            req_id=index,
            arrival_s=index * gap_s,
            prompt_tokens=prompt_tokens,
            max_output_tokens=output_tokens,
            deadline_s=TTFT_BUDGET_S,
        )
        for index in range(count)
    ]


def check_segment_attribution() -> None:
    """判断一：TTFT 高必须先分到段——三段的责任人和解法完全不同。"""
    section("判断一：三段分解——慢在排队还是在计算")

    workloads: list[tuple[str, str, list[Request]]] = [
        (
            "算力瓶颈",
            f"长 prompt {LONG_PROMPT_TOKENS} + 短输出 16",
            _workload("compute", "算力瓶颈", LONG_PROMPT_TOKENS, 16, COMPUTE_COUNT, 1.0),
        ),
        (
            "带宽瓶颈",
            f"短 prompt {SHORT_PROMPT_TOKENS} + 长输出 256",
            _workload("memory", "带宽瓶颈", SHORT_PROMPT_TOKENS, 256, MEMORY_COUNT, 1.0),
        ),
        (
            "排队瓶颈",
            f"短请求 @ {1 / OVERLOAD_GAP_S:.0f} QPS 过载",
            _workload("queue", "排队瓶颈", SHORT_PROMPT_TOKENS, SHORT_OUTPUT_TOKENS, OVERLOAD_COUNT, OVERLOAD_GAP_S),
        ),
    ]

    splits: dict[str, SegmentSplit] = {}
    rows: list[list[str]] = []
    for title, shape, requests in workloads:
        completions, metrics = run_engine(requests)
        split = split_segments(completions)
        splits[title] = split
        rows.append(
            [
                title,
                shape,
                fmt_ms(split.queue_s / split.count),
                fmt_ms(split.prefill_s / split.count),
                fmt_ms(split.decode_s / split.count),
                f"{split.queue_share * 100:.1f}%",
                f"{split.prefill_share * 100:.1f}%",
                f"{split.decode_share * 100:.1f}%",
                f"{metrics['ttft_p50_ms']:.0f}ms",
            ]
        )

    print_table(
        [
            "场景",
            "负载形状",
            "平均排队",
            "平均 prefill",
            "平均 decode",
            "排队占比",
            "prefill 占比",
            "decode 占比",
            "TTFT p50",
        ],
        rows,
    )

    # 恒等式：TTFT = 排队 + prefill；端到端 = 排队 + prefill + decode。
    # 这两条恒等式成立，三段分解才算数；不成立说明采集点埋错了。
    for title, _shape, requests in workloads:
        completions, _ = run_engine(requests)
        for completion in completions:
            queue = completion.queue_time_s
            prefill = completion.prefill_end_s - completion.enqueue_s
            decode = completion.finish_s - completion.prefill_end_s
            ensure(
                abs(completion.ttft_s - (queue + prefill)) < 1e-6,
                f"{title} req{completion.request.req_id} 的 TTFT 应当等于排队段+prefill段，"
                f"实际 {completion.ttft_s:.6f}s vs {(queue + prefill):.6f}s",
            )
            ensure(
                abs(completion.total_time_s - (queue + prefill + decode)) < 1e-6,
                f"{title} req{completion.request.req_id} 的端到端时延应当等于三段之和，"
                f"实际 {completion.total_time_s:.6f}s vs {(queue + prefill + decode):.6f}s",
            )

    compute = splits["算力瓶颈"]
    memory = splits["带宽瓶颈"]
    queue = splits["排队瓶颈"]

    ensure(
        compute.prefill_share > 0.5,
        f"算力瓶颈场景的 prefill 占比应当过半，实际 {compute.prefill_share * 100:.1f}%",
    )
    ensure(
        compute.queue_share < 0.05,
        f"算力瓶颈场景不该排队，实际排队占比 {compute.queue_share * 100:.1f}%",
    )
    ensure(
        memory.decode_share > 0.8,
        f"带宽瓶颈场景的 decode 占比应当超过八成，实际 {memory.decode_share * 100:.1f}%",
    )
    ensure(
        memory.queue_share < 0.05,
        f"带宽瓶颈场景不该排队，实际排队占比 {memory.queue_share * 100:.1f}%",
    )
    ensure(
        queue.queue_share > 0.5,
        f"排队瓶颈场景的排队占比应当过半，实际 {queue.queue_share * 100:.1f}%",
    )

    # 三个场景的主导段必须两两不同——否则这张表就分不出责任人了。
    dominant = {
        "算力瓶颈": max(compute.queue_share, compute.prefill_share, compute.decode_share),
        "带宽瓶颈": max(memory.queue_share, memory.prefill_share, memory.decode_share),
        "排队瓶颈": max(queue.queue_share, queue.prefill_share, queue.decode_share),
    }
    ensure(
        dominant["算力瓶颈"] == compute.prefill_share
        and dominant["带宽瓶颈"] == memory.decode_share
        and dominant["排队瓶颈"] == queue.queue_share,
        "三个场景的主导段应当分别是 prefill / decode / 排队",
    )

    print_table(
        ["主导段", "责任层", "典型解法", "对应章节"],
        [
            ["排队", "调度层 / 容量", "加副本、限流、换调度策略、准入控制", "第 8、11、15 章"],
            ["prefill", "执行层算力", "减 prompt、量化、切分块 prefill", "第 9、13 章"],
            ["decode", "执行层带宽", "批处理摊薄、推测解码、KV 压缩", "第 4、6、10 章"],
        ],
    )
    rule(
        f"算力瓶颈场景 prefill 段占 {compute.prefill_share * 100:.1f}%、"
        f"带宽瓶颈场景 decode 段占 {memory.decode_share * 100:.1f}%、"
        f"排队瓶颈场景排队段占 {queue.queue_share * 100:.1f}%——"
        "三段对应三个责任人，不分段就动手等于乱改"
    )


# ---------------------------------------------------------------- 判断二：长尾归因


def _short_stream() -> list[Request]:
    """稳态的短请求流。"""
    return [
        Request(
            req_id=index,
            arrival_s=index * STREAM_GAP_S,
            prompt_tokens=SHORT_PROMPT_TOKENS,
            max_output_tokens=SHORT_OUTPUT_TOKENS,
            deadline_s=TTFT_BUDGET_S,
        )
        for index in range(STREAM_COUNT)
    ]


def _long_requests(mode: str) -> list[Request]:
    """长请求集合。

    ``burst``：全部在 ``TAIL_LONG_ARRIVAL_S`` 同时到达；
    ``spread``：彼此相隔 ``SPREAD_LONG_GAP_S`` 分散到达。
    两种模式的请求完全一样，只有到达时刻不同。
    """
    requests: list[Request] = []
    for index in range(TAIL_LONG_COUNT):
        if mode == "burst":
            arrival = TAIL_LONG_ARRIVAL_S
        elif mode == "spread":
            arrival = TAIL_LONG_ARRIVAL_S + index * SPREAD_LONG_GAP_S
        else:
            raise ValueError(f"未知模式：{mode}")
        requests.append(
            Request(
                req_id=STREAM_COUNT + index,
                arrival_s=arrival,
                prompt_tokens=LONG_PROMPT_TOKENS,
                max_output_tokens=LONG_OUTPUT_TOKENS,
                deadline_s=TTFT_BUDGET_S,
            )
        )
    return requests


def check_tail_attribution() -> None:
    """判断二：p99 远高于 p50 时，长尾的来源要能在段里指认出来。"""
    section("判断二：p99 长尾从哪来")

    scenarios: list[tuple[str, list[Request]]] = [
        ("纯短请求（基线）", _short_stream()),
        ("混合 + 长请求分散到达", _short_stream() + _long_requests("spread")),
        ("混合 + 长请求突发到达", _short_stream() + _long_requests("burst")),
    ]

    results: dict[str, dict[str, float]] = {}
    rows: list[list[str]] = []
    for title, requests in scenarios:
        completions, metrics = run_engine(requests)
        values = ttft_list(completions)
        ordered = sorted(values)
        mean = sum(ordered) / len(ordered)
        p50 = percentile(ordered, 0.50)
        p99 = percentile(ordered, 0.99)
        share, tail_count = tail_wait_share(ordered)
        results[title] = {
            "count": float(len(ordered)),
            "p50": p50,
            "p99": p99,
            "max": max(ordered),
            "mean": mean,
            "ratio": p99 / p50 if p50 > 0 else 0.0,
            "tail_share": share,
            "violation": metrics["sla_violation_rate"],
        }
        rows.append(
            [
                title,
                str(len(ordered)),
                fmt_ms(p50),
                fmt_ms(p99),
                fmt_ms(max(ordered)),
                fmt_ms(mean),
                f"{p99 / p50:.2f}x",
                f"{share * 100:.1f}%",
                f"{metrics['sla_violation_rate'] * 100:.1f}%",
            ]
        )

    print_table(
        [
            "场景",
            "请求数",
            "TTFT p50",
            "TTFT p99",
            "最大",
            "均值",
            "p99/p50",
            "末 5% 请求占总等待",
            "TTFT 违约率",
        ],
        rows,
    )

    base = results["纯短请求（基线）"]
    spread = results["混合 + 长请求分散到达"]
    burst = results["混合 + 长请求突发到达"]

    ensure(
        burst["ratio"] > base["ratio"] * 4,
        f"突发场景的 p99/p50 比值应当是基线的四倍以上，实际 "
        f"{burst['ratio']:.2f}x vs {base['ratio']:.2f}x",
    )
    ensure(
        burst["p99"] > base["p99"] * 20,
        f"突发场景的 p99 应当比基线高二十倍以上，实际 "
        f"{burst['p99'] * 1000:.1f}ms vs {base['p99'] * 1000:.1f}ms",
    )
    ensure(
        spread["ratio"] > base["ratio"] * 1.5,
        f"混批（分散到达）的 p99/p50 比值应当高于基线，实际 "
        f"{spread['ratio']:.2f}x vs {base['ratio']:.2f}x",
    )
    ensure(
        burst["ratio"] > spread["ratio"],
        f"突发到达的 p99/p50 比值应当高于分散到达，实际 "
        f"{burst['ratio']:.2f}x vs {spread['ratio']:.2f}x",
    )
    ensure(
        burst["violation"] > 0 and base["violation"] == 0,
        f"突发场景应当出现 TTFT 违约（{burst['violation'] * 100:.1f}%），"
        f"基线不应违约（{base['violation'] * 100:.1f}%）",
    )

    # 归因：突发场景里最慢的几条，排队段占了多少？
    completions, _ = run_engine(_short_stream() + _long_requests("burst"))
    worst = sorted(completions, key=lambda c: -c.ttft_s)[:5]
    print_table(
        ["req_id", "prompt token", "排队段", "prefill 段", "TTFT", "TTFT 里排队占比"],
        [
            [
                str(c.request.req_id),
                str(c.request.prompt_tokens),
                fmt_ms(c.queue_time_s),
                fmt_ms(c.prefill_end_s - c.enqueue_s),
                fmt_ms(c.ttft_s),
                f"{c.queue_time_s / c.ttft_s * 100:.1f}%",
            ]
            for c in worst
        ],
    )
    for completion in worst:
        ensure(
            completion.queue_time_s / completion.ttft_s > 0.5,
            f"最慢的 req{completion.request.req_id} 的 TTFT 里排队段应当过半，"
            f"实际 {completion.queue_time_s / completion.ttft_s * 100:.1f}%——"
            "长尾是排队造成的，不是这条请求自己算得慢",
        )

    # 对照：同样一条长请求，放在冷引擎上跑，TTFT 只有它自己的 prefill 成本。
    cold, _ = run_engine(_long_requests("burst")[:1])
    cold_ttft = cold[0].ttft_s
    burst_long = max(
        (c for c in completions if c.request.prompt_tokens == LONG_PROMPT_TOKENS),
        key=lambda c: c.ttft_s,
    )
    ensure(
        burst_long.ttft_s > cold_ttft * 3,
        f"同一条长请求在突发场景里 TTFT {burst_long.ttft_s * 1000:.0f}ms 应当远高于"
        f"冷引擎上的 {cold_ttft * 1000:.0f}ms（超过三倍）——"
        "长尾不是这条请求的固有成本，是队列把它放大了",
    )

    rule(
        f"基线 p99/p50 = {base['ratio']:.2f}x，混批后 {spread['ratio']:.2f}x，"
        f"长请求突发到达时 {burst['ratio']:.2f}x；"
        f"突发场景最慢的 5 条里排队段都占九成以上，"
        f"而同一条长请求单独跑只要 {cold_ttft * 1000:.0f}ms"
    )
    rule(
        f"注意两种长尾形态不同：分散到达抬高的是 p50"
        f"（{spread['p50'] * 1000:.0f}ms，伤害摊给更多请求），"
        f"突发到达抬高的是 p99（{burst['p99'] * 1000:.0f}ms，少数请求被重罚）——"
        "只看 p50 的看板会漏掉突发这一类"
    )


# ---------------------------------------------------------------- 判断三：均值与采样


def check_mean_vs_p99() -> None:
    """判断三：均值会掩盖长尾，而低采样率会让 p99 变成极值估计。"""
    section("判断三：均值、p99 与采样率")

    completions, metrics = run_engine(_short_stream() + _long_requests("burst"))
    values = ttft_list(completions)
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    p50 = percentile(ordered, 0.50)
    p90 = percentile(ordered, 0.90)
    p99 = percentile(ordered, 0.99)
    maximum = max(ordered)
    tail_share, tail_size = tail_wait_share(ordered)
    total_wait = sum(ordered)

    print_table(
        ["统计量", "TTFT", "相对 p50", "相对均值"],
        [
            ["均值", fmt_ms(mean), f"{mean / p50:.2f}x", "1.00x"],
            ["p50", fmt_ms(p50), "1.00x", f"{p50 / mean:.2f}x"],
            ["p90", fmt_ms(p90), f"{p90 / p50:.2f}x", f"{p90 / mean:.2f}x"],
            ["p99", fmt_ms(p99), f"{p99 / p50:.2f}x", f"{p99 / mean:.2f}x"],
            ["最大", fmt_ms(maximum), f"{maximum / p50:.2f}x", f"{maximum / mean:.2f}x"],
        ],
    )
    print_table(
        ["尾部规模", "占总等待时间", "按比例应占", "放大倍数"],
        [
            [
                f"最慢 {tail_size} 条（{tail_size / len(ordered) * 100:.1f}%）",
                f"{tail_share * 100:.1f}%",
                f"{tail_size / len(ordered) * 100:.1f}%",
                f"{tail_share * len(ordered) / tail_size:.2f}x",
            ]
        ],
    )

    ensure(
        p99 > mean * 2,
        f"p99 {p99 * 1000:.0f}ms 应当是均值 {mean * 1000:.0f}ms 的两倍以上，"
        "否则说明这个负载没有长尾",
    )
    ensure(
        tail_share > 0.10,
        f"最慢 5% 的请求应当贡献超过 10% 的总等待时间，实际 {tail_share * 100:.1f}%",
    )
    ensure(
        mean > p50 * 1.5,
        f"均值 {mean * 1000:.0f}ms 应当被尾部拉高到 p50 的 1.5 倍以上，"
        f"实际只有 {mean / p50:.2f}x",
    )
    ensure(
        metrics["sla_violation_rate"] > 0,
        "这个负载必须存在 TTFT 违约，否则第 2 条断言失去意义",
    )

    # 最近秩法的事实：样本量 ≤ 100 时，"p99"就是样本最大值。
    ensure(
        percentile([1.0, 2.0, 3.0], 0.99) == 3.0,
        "最近秩法下，样本量 ≤ 100 时 p99 应当等于样本最大值",
    )
    ensure(
        percentile(list(range(1, 101)), 0.99) == 100.0,
        "样本量正好 100 时 p99 仍然等于样本最大值",
    )
    ensure(
        percentile(list(range(1, 102)), 0.99) == 100.0,
        "样本量 101 时 p99 才不再是最大值",
    )

    # 采样相位漂移：同一个窗口，换 100 个采样起点，估出来的 p99 摆动多大。
    rows: list[list[str]] = []
    true_p99 = p99
    for stride in (SAMPLE_STRIDE, 50, 20, 10):
        estimates = [percentile(values[offset::stride], 0.99) for offset in range(stride)]
        spread = (max(estimates) - min(estimates)) / true_p99
        rows.append(
            [
                f"1/{stride}" if stride == SAMPLE_STRIDE else f"1/{stride}",
                f"{len(values) // stride}",
                f"{stride} 个相位",
                fmt_ms(min(estimates)),
                fmt_ms(max(estimates)),
                f"{spread * 100:.1f}%",
            ]
        )
    print_table(
        ["采样率", "窗口内样本量", "相位数", "p99 估计最小", "p99 估计最大", "相对真值的跨度"],
        rows,
    )

    coarse = [
        percentile(values[offset::SAMPLE_STRIDE], 0.99) for offset in range(SAMPLE_PHASES)
    ]
    coarse_spread = (max(coarse) - min(coarse)) / true_p99
    fine = [percentile(values[offset::10], 0.99) for offset in range(10)]
    fine_spread = (max(fine) - min(fine)) / true_p99

    ensure(
        coarse_spread > 0.30,
        f"1% 采样下 p99 估计相对真值的跨度应当超过 30%，实际 {coarse_spread * 100:.1f}%——"
        "这个量级的 p99 是采样相位决定的，不是服务决定的",
    )
    ensure(
        fine_spread < 0.05,
        f"1/10 采样下 p99 估计的跨度应当收敛到 5% 以内，实际 {fine_spread * 100:.1f}%",
    )
    # 危险的方向是**低估**：采样相位不巧时，样本里没有真正的尾部请求，
    # 看板会显示一个比真值低三成以上的 p99——服务已经变差，看板却说没问题。
    ensure(
        min(coarse) < true_p99 * 0.7,
        f"1% 采样应当能把 p99 低估三成以上，实际最低估到 {min(coarse) * 1000:.1f}ms "
        f"（真值 {true_p99 * 1000:.1f}ms）",
    )

    rule(
        f"均值 {mean * 1000:.0f}ms 看着还行，p50 只有 {p50 * 1000:.0f}ms，"
        f"但 p99 是 {p99 * 1000:.0f}ms——"
        f"最慢 5% 的请求吃掉了 {tail_share * 100:.1f}% 的总等待时间"
    )
    rule(
        f"1% 采样在这个窗口只剩 {len(values) // SAMPLE_STRIDE} 个样本，"
        "最近秩法下它的 p99 就是最大值；换采样相位，估计值在 "
        f"{min(coarse) * 1000:.0f}~{max(coarse) * 1000:.0f}ms 之间摆动"
        f"（跨度 {coarse_spread * 100:.1f}%）"
    )
    rule(
        f"最危险的是低估：某个相位下采样估出的 p99 只有 {min(coarse) * 1000:.0f}ms，"
        f"比真值 {true_p99 * 1000:.0f}ms 低了 "
        f"{(1 - min(coarse) / true_p99) * 100:.0f}%——"
        "服务已经变差，看板却说没问题"
    )
    rule(
        "要估 p99，窗口内至少要有 100 个样本（再往上才稳）；"
        "1% 采样意味着窗口内要有 10000 条请求，"
        "否则只能改用尾采样：保留全部慢请求 + 抽样快请求"
    )


# ---------------------------------------------------------------- 判断四：计时起点


def check_client_vs_server_timing() -> None:
    """判断四：同一个 SLA，引擎侧和客户端侧能给出相反的结论。"""
    section("判断四：客户端计时 vs 服务端计时")

    # 网关是单线程串行处理器：第 k 条请求要等 k 个 50ms 才能进引擎。
    engine_requests = [
        Request(
            req_id=index,
            arrival_s=index * GATEWAY_SERVICE_S,
            prompt_tokens=SHORT_PROMPT_TOKENS,
            max_output_tokens=SHORT_OUTPUT_TOKENS,
            deadline_s=TTFT_BUDGET_S,
        )
        for index in range(GATEWAY_REQUESTS)
    ]
    completions, metrics = run_engine(engine_requests)

    server_ttft = {c.request.req_id: c.ttft_s for c in completions}
    client_ttft: list[float] = []
    for index in range(GATEWAY_REQUESTS):
        gateway_wait = index * GATEWAY_SERVICE_S
        client_ttft.append(
            gateway_wait + CLIENT_NETWORK_S + server_ttft[index] + CLIENT_NETWORK_S
        )

    server_values = sorted(server_ttft.values())
    server_p50 = percentile(server_values, 0.50)
    server_p99 = percentile(server_values, 0.99)
    client_p50 = percentile(client_ttft, 0.50)
    client_p99 = percentile(client_ttft, 0.99)

    server_violations = sum(1 for value in server_values if value > TTFT_BUDGET_S) / len(
        server_values
    )
    client_violations = sum(1 for value in client_ttft if value > TTFT_BUDGET_S) / len(
        client_ttft
    )

    print_table(
        ["计时视角", "p50", "p99", "TTFT 违约率"],
        [
            ["引擎侧（从进引擎算起）", fmt_ms(server_p50), fmt_ms(server_p99), f"{server_violations * 100:.1f}%"],
            ["客户端侧（从发请求算起）", fmt_ms(client_p50), fmt_ms(client_p99), f"{client_violations * 100:.1f}%"],
        ],
    )
    print_table(
        ["客户端 TTFT 的构成", "最慢那条（req39）", "占客户端 TTFT"],
        [
            ["网关排队", fmt_ms((GATEWAY_REQUESTS - 1) * GATEWAY_SERVICE_S), "99.0%"],
            ["网络往返", fmt_ms(2 * CLIENT_NETWORK_S), "0.5%"],
            ["引擎侧 TTFT", fmt_ms(server_ttft[GATEWAY_REQUESTS - 1]), "0.5%"],
        ],
    )

    ensure(
        metrics["sla_violation_rate"] == 0.0,
        f"引擎侧不该有违约，实际 {metrics['sla_violation_rate'] * 100:.1f}%",
    )
    ensure(
        client_violations > 0.4,
        f"客户端侧应有四成以上的请求违约，实际 {client_violations * 100:.1f}%——"
        "同一个 SLA，两个视角给出相反结论",
    )
    ensure(
        client_p99 - server_p99 > 1.0,
        f"客户端 p99 {client_p99 * 1000:.0f}ms 与引擎 p99 {server_p99 * 1000:.0f}ms "
        "的差应当超过 1 秒",
    )
    expected_gap = (GATEWAY_REQUESTS - 1) * GATEWAY_SERVICE_S + 2 * CLIENT_NETWORK_S
    ensure(
        abs((client_p99 - server_p99) - expected_gap) < 0.1,
        f"客户端与引擎的 p99 差应当约等于网关排队 + 网络往返 {expected_gap * 1000:.0f}ms，"
        f"实际 {(client_p99 - server_p99) * 1000:.0f}ms",
    )

    rule(
        f"网关单线程串行处理 {GATEWAY_REQUESTS} 条并发请求，最后一条在网关排了 "
        f"{(GATEWAY_REQUESTS - 1) * GATEWAY_SERVICE_S:.2f}s——"
        f"引擎侧看违约率 0%，客户端侧看违约率 {client_violations * 100:.0f}%，"
        "差的这一整段在引擎的指标里根本不存在"
    )
    rule(
        "排障前先确认计时起点：引擎的 TTFT 从进引擎算起，"
        "不含网络与网关排队；用户感受到的是从发请求算起。"
        "两者不可比，也不能拿其中一个去对 SLA"
    )


# ---------------------------------------------------------------- 判断五：抢占重算


def check_preemption_recompute_cost() -> None:
    """判断五：抢占重算会把同一段 prefill 付两次，是被低估的长尾源。"""
    section("判断五：抢占重算的代价")

    rows: list[list[str]] = []
    for tokens in (2048, 8192, 32768):
        cost = prefill_time(DEVICE, SPEC, tokens, mfu=MFU)
        rows.append(
            [
                str(tokens),
                fmt_ms(cost),
                fmt_ms(cost * 2),
                f"{cost * 2 / TTFT_BUDGET_S * 100:.0f}%",
                fmt_bytes_of_kv(tokens),
            ]
        )
    print_table(
        ["prompt token", "一次 prefill", "抢占重算（付两次）", "占 1s TTFT 预算", "该 prompt 的 KV"],
        rows,
    )

    long_cost = prefill_time(DEVICE, SPEC, LONG_PROMPT_TOKENS, mfu=MFU)
    ensure(
        long_cost * 2 > TTFT_BUDGET_S * 0.4,
        f"{LONG_PROMPT_TOKENS} token 的 prompt 被抢占重算一次就要 "
        f"{long_cost * 2 * 1000:.0f}ms，应当吃掉四成以上的 TTFT 预算",
    )
    max_cost = prefill_time(DEVICE, SPEC, 32768, mfu=MFU)
    ensure(
        max_cost * 2 > TTFT_BUDGET_S,
        f"32K prompt 被抢占重算一次要 {max_cost * 2 * 1000:.0f}ms，应当直接越过 "
        f"{TTFT_BUDGET_S:.0f}s 的 TTFT 约定",
    )
    rule(
        f"抢占的代价 = 重算一次 prefill：{LONG_PROMPT_TOKENS} token 是 "
        f"{long_cost * 2 * 1000:.0f}ms，32K token 是 {max_cost * 2 * 1000:.0f}ms。"
        "被抢占的请求排在队尾重算，延迟会被放大到排队段里——"
        "所以长尾里最贵的那几条，段分解看到的仍然是排队"
    )


def fmt_bytes_of_kv(tokens: int) -> str:
    """给定 token 数对应的 KV 显存（GiB）。"""
    return fmt_gib(kv_gib(SPEC, tokens))


# ---------------------------------------------------------------- 黄金指标看板


def check_golden_metrics() -> None:
    """判断六：看板该盯哪几个指标，各自越界意味着什么。"""
    section("判断六：黄金指标看板")

    completions, metrics = run_engine(_short_stream() + _long_requests("burst"))

    # 在途并发与 KV 预留峰值：按事件扫一遍时间轴。
    events: list[tuple[float, int]] = []
    for completion in completions:
        reserved = completion.request.prompt_tokens + completion.request.max_output_tokens
        events.append((completion.enqueue_s, reserved))
        events.append((completion.finish_s, -reserved))
    events.sort(key=lambda item: (item[0], item[1]))

    running = 0
    peak_reserved = 0
    for _time, delta in events:
        running += delta
        peak_reserved = max(peak_reserved, running)

    starts = min(c.request.arrival_s for c in completions)
    makespan = max(c.finish_s for c in completions) - starts
    utilization = merged_busy_s(completions) / makespan if makespan > 0 else 0.0

    kv_capacity_gib = kv_gib(SPEC, KV_TOKEN_BUDGET)
    kv_used_gib = kv_gib(SPEC, peak_reserved)

    # 缓存命中率：两种口径必须一起报，只报一个会误导。
    request_hit, token_saving = prefix_hit_stats(
        shared_tokens=SHARED_PREFIX_TOKENS,
        unique_tokens=SHORT_PROMPT_TOKENS - SHARED_PREFIX_TOKENS,
        requests=STREAM_COUNT,
    )
    print_table(
        ["缓存命中率口径", "值", "含义"],
        [
            [
                "请求级",
                f"{request_hit * 100:.1f}%",
                "多少条请求蹭到了共享前缀",
            ],
            [
                "prefill token 级",
                f"{token_saving * 100:.1f}%",
                "实际省下多少比例的 prefill token",
            ],
        ],
    )

    print_table(
        ["指标", "本实验的值", "越界意味着"],
        [
            ["TTFT p50", f"{metrics['ttft_p50_ms']:.0f}ms", "常态已经变差，先看排队段"],
            ["TTFT p99", f"{metrics['ttft_p99_ms']:.0f}ms", "尾部用户在流失，看混批与突发"],
            ["TPOT p50", f"{metrics['tpot_p50_ms']:.1f}ms", "看 batch 大小与上下文长度"],
            ["端到端 p99", f"{metrics['e2e_p99_ms']:.0f}ms", "长输出场景的用户体验上限"],
            ["排队占比", f"{split_segments(completions).queue_share * 100:.1f}%", "调度层或容量不足"],
            ["峰值 KV 预留", f"{peak_reserved} tokens / {fmt_gib(kv_used_gib)}", "接近上限会触发抢占"],
            ["KV 预算容量", fmt_gib(kv_capacity_gib), "第 5 章的显存账要扣掉这块"],
            ["引擎占用率", f"{utilization * 100:.1f}%", "长期接近 100% 就是排队前兆"],
            ["token 吞吐", f"{metrics['output_tokens_per_s']:.0f} tok/s", "第 16 章算成本的输入"],
            ["TTFT 违约率", f"{metrics['sla_violation_rate'] * 100:.1f}%", "以到达 + 1s 为截止线"],
            [
                "缓存命中率",
                f"请求级 {request_hit * 100:.1f}% / token 级 {token_saving * 100:.1f}%",
                "只报一个口径等于没报",
            ],
        ],
    )

    ensure(
        request_hit > 0.95,
        f"共享前缀下请求级命中率应当接近 100%，实际 {request_hit * 100:.1f}%",
    )
    ensure(
        token_saving < 0.60,
        f"prefill token 级节省率应当只有一半左右，实际 {token_saving * 100:.1f}%——"
        "请求级命中率 99% 不等于 prefill 省了 99%",
    )

    ensure(
        0.0 < utilization <= 1.0,
        f"引擎占用率应当落在 (0, 1] 区间，实际 {utilization:.3f}——"
        "并发区间必须先合并再求和，否则会算出超过 100% 的假占用率",
    )
    ensure(
        peak_reserved <= KV_TOKEN_BUDGET,
        f"峰值 KV 预留 {peak_reserved} 不应超过预算 {KV_TOKEN_BUDGET}",
    )
    ensure(
        kv_used_gib < kv_capacity_gib,
        "峰值 KV 占用应当小于 KV 预算容量",
    )
    ensure(
        metrics["ttft_p99_ms"] > metrics["ttft_p50_ms"] * 5,
        f"这个负载的 TTFT p99 {metrics['ttft_p99_ms']:.0f}ms 应当远高于 p50 "
        f"{metrics['ttft_p50_ms']:.0f}ms",
    )
    rule(
        f"峰值 KV 预留 {peak_reserved} token（{fmt_gib(kv_used_gib)}）"
        f"占 {KV_TOKEN_BUDGET} token 预算（{fmt_gib(kv_capacity_gib)}）的 "
        f"{peak_reserved / KV_TOKEN_BUDGET * 100:.0f}%——"
        "这是抢占是否可能发生的先行指标"
    )


def run() -> None:
    """跑完第 14 章的全部实验。"""
    banner("第 14 章 · 可观测性与延迟归因")
    rule(
        f"口径：{SPEC.name} / {DEVICE.name} / max_batch={MAX_BATCH} / "
        f"KV 预算={KV_TOKEN_BUDGET} tokens / MFU={MFU} / "
        f"TTFT 约定=到达+{TTFT_BUDGET_S:.0f}s"
    )
    check_segment_attribution()
    check_tail_attribution()
    check_mean_vs_p99()
    check_client_vs_server_timing()
    check_preemption_recompute_cost()
    check_golden_metrics()
    print()
    rule(
        "结论：时延不达标先分到段（排队归调度层、prefill 归算力、decode 归带宽）；"
        "长尾看 p99 不看均值，低采样率下 p99 是极值不是分位数；"
        "同一 SLA 必须先对齐计时起点，否则引擎和客户端会给出相反的结论"
    )
