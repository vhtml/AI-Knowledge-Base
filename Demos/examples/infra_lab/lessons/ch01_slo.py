"""第 1 章实验：推理服务全景与 SLO。

这个模块解决什么问题：同一张 GPU、同一个模型，在"短问答""长文档摘要"
"流式长输出"三种负载画像下，服务等级（SLO）表现完全不同——TTFT、TPOT、
吞吐三项指标互相拉扯，谁也做不到同时最优。本章要把这几件事变成可执行的断言：

1. **一次请求的时延可以拆成三段**——排队、prefill、decode，
   每段归属不同的层：接入层、调度层、执行层各背各的锅，
   所以"时延不达标"必须先定位到段，再定位到层；
2. **TTFT / TPOT / 吞吐是三角**——把负载压上去，吞吐涨、TPOT 涨、
   TTFT 涨得更凶，三者不能同时最优；
3. **容量必须按负载画像算**——同一张卡在三种画像下的 SLO 表现差着量级，
   用平均值算容量必然翻车；
4. **单位经济要看每 token 成本**——同样的 QPS，不同画像的每请求成本
   可以差一个数量级以上，拿"每卡 QPS"当比较指标会误判。

本章所有算账都走 :mod:`infra_lab.labkit`：显存账、FLOPs、时延一律调用公共层，
不另立公式。全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labkit import (
    Completion,
    EngineSimulator,
    GPUSpec,
    Request,
    approx,
    banner,
    ensure,
    fmt_ms,
    gpu,
    load_balancing_delta,
    model,
    print_table,
    rule,
    section,
)

DEVICE = "h100_sxm"
"""算账用的卡：H100 SXM 80GB，3350 GB/s、989 TFLOPS FP16、$4.5/卡时。"""

MODEL = "qwen2.5-7b"
"""算账用的模型：7.62B 参数、28 层、28 查询头 / 4 KV 头。"""

MFU = 0.5
"""可达算力/带宽占峰值的比例（推理侧工程常见 40%~70%）。"""

MAX_BATCH = 32
"""引擎最大并发序列数。三种画像共用同一份引擎配置，才能比出差异。"""

KV_TOKEN_BUDGET = 100_000
"""KV 预留预算（token）。按「prompt + max_output」预留，三种画像同一口径。"""

BASE_QPS = 8.0
"""基准到达率：三种画像用同一个 QPS，才能把差异归到"画像"而不是"压力"。"""

REQUESTS = 32
"""每种画像的请求条数。32 条足够让 p50/p99 稳定，又不会跑太久。"""

SATURATION_REQUESTS = 200
"""饱和压测的请求条数：压到队列再也排不下，才量得出可持续容量。"""

SATURATION_GAP = 0.01
"""饱和压测的到达间隔：100 QPS，远超任何一种画像在单卡上的容量。"""


@dataclass(frozen=True)
class LoadProfile:
    """一种负载画像：请求的"重量"分布。

    画像是本章的主角：同样是 8 QPS，短问答和长文档摘要消耗的卡时差很多倍。
    """

    key: str
    title: str
    prompt_tokens: int
    output_tokens: int
    requests: int = REQUESTS
    gap_s: float = 1.0 / BASE_QPS
    """到达间隔。1/gap_s 就是这一画像的到达率。"""


PROFILES: tuple[LoadProfile, ...] = (
    LoadProfile(
        key="short_qa",
        title="短问答",
        prompt_tokens=128,
        output_tokens=64,
    ),
    LoadProfile(
        key="long_doc",
        title="长文档摘要",
        prompt_tokens=4096,
        output_tokens=256,
    ),
    LoadProfile(
        key="streaming",
        title="流式长输出",
        prompt_tokens=256,
        output_tokens=1024,
    ),
)
"""三种负载画像。第 1 章的核心结论就建立在这三行上。"""


@dataclass(frozen=True)
class ProfileResult:
    """一种画像跑完一遍之后的结果汇总。"""

    profile: LoadProfile
    metrics: dict[str, float]
    completions: list[Completion]

    span_s: float
    """墙钟跨度：最后一个请求结束 − 第一个请求到达。"""

    total_cost_usd: float
    """这段墙钟内这张卡的租金。"""

    cost_per_request_usd: float
    """每请求成本（卡时摊薄到单条请求）。"""

    cost_per_mtok_output_usd: float
    """每百万**输出** token 的成本。"""

    cost_per_mtok_processed_usd: float
    """每百万**处理** token（输入 + 输出）的成本。"""

    input_tokens: int
    output_tokens: int
    avg_e2e_s: float
    """平均端到端时延。"""

    queue_share: float
    """排队时延占端到端时延的平均比例。"""


def build_requests(
    profile: LoadProfile,
    gap_s: float | None = None,
    requests: int | None = None,
) -> list[Request]:
    """按画像造一批请求。

    到达间隔默认取画像自己的 ``gap_s``；传 ``gap_s`` 可以造出"同样的请求数、
    不同的到达率"这组对照；传 ``requests`` 可以造出饱和压测的长队列。
    """
    gap = profile.gap_s if gap_s is None else gap_s
    count = profile.requests if requests is None else requests
    return [
        Request(
            req_id=index,
            arrival_s=index * gap,
            prompt_tokens=profile.prompt_tokens,
            max_output_tokens=profile.output_tokens,
            tenant="default",
            deadline_s=1.0,
        )
        for index in range(count)
    ]


def run_profile(
    device: GPUSpec,
    profile: LoadProfile,
    gap_s: float | None = None,
    requests: int | None = None,
) -> ProfileResult:
    """用同一份引擎配置跑一种画像，返回时延、吞吐与成本。

    引擎参数（max_batch、KV 预算、MFU）三种画像完全一致——这样跑出来的差异
    只能归因于画像本身，不能归因于配置。
    """
    engine = EngineSimulator(
        device,
        model(MODEL),
        max_batch=MAX_BATCH,
        kv_token_budget=KV_TOKEN_BUDGET,
        mfu=MFU,
    )
    completions = engine.run(build_requests(profile, gap_s, requests))
    metrics = engine.metrics()

    span = max(
        max(c.finish_s for c in completions) - min(c.request.arrival_s for c in completions),
        1e-9,
    )
    total_cost = device.price_per_hour_usd * span / 3600.0
    input_tokens = sum(c.request.prompt_tokens for c in completions)
    output_tokens = sum(c.output_tokens for c in completions)
    processed = input_tokens + output_tokens
    count = max(1, len(completions))
    avg_e2e = sum(c.total_time_s for c in completions) / count
    queue_share = (
        sum(c.queue_time_s for c in completions) / count / avg_e2e if avg_e2e > 0 else 0.0
    )

    return ProfileResult(
        profile=profile,
        metrics=metrics,
        completions=completions,
        span_s=span,
        total_cost_usd=total_cost,
        cost_per_request_usd=total_cost / count,
        cost_per_mtok_output_usd=(
            total_cost / output_tokens * 1e6 if output_tokens > 0 else 0.0
        ),
        cost_per_mtok_processed_usd=(
            total_cost / processed * 1e6 if processed > 0 else 0.0
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        avg_e2e_s=avg_e2e,
        queue_share=queue_share,
    )


def check_lifecycle_decomposition() -> None:
    """判断：一次请求的时延能拆成"排队 + prefill + decode"三段，各归各的层。"""
    section("判断一：请求生命周期三段分解")
    device = gpu(DEVICE)
    results = {p.key: run_profile(device, p) for p in PROFILES}

    rows: list[list[object]] = []
    for profile in PROFILES:
        result = results[profile.key]
        queue = sum(c.queue_time_s for c in result.completions)
        prefill = sum(c.prefill_end_s - c.enqueue_s for c in result.completions)
        decode = sum(c.finish_s - c.prefill_end_s for c in result.completions)
        total = queue + prefill + decode
        rows.append(
            [
                profile.title,
                f"{queue:.3f}s",
                f"{prefill:.3f}s",
                f"{decode:.3f}s",
                f"{queue / total * 100:.1f}%",
                f"{prefill / total * 100:.1f}%",
                f"{decode / total * 100:.1f}%",
            ]
        )
    print_table(
        ["画像", "排队段", "prefill 段", "decode 段", "排队占比", "prefill 占比", "decode 占比"],
        rows,
    )

    for profile in PROFILES:
        for completion in results[profile.key].completions:
            ensure(
                approx(
                    completion.ttft_s,
                    completion.queue_time_s + (completion.prefill_end_s - completion.enqueue_s),
                    1e-6,
                ),
                f"{profile.title} req{completion.request.req_id} 的 TTFT 应当等于"
                f"排队段 + prefill 段，实际 {completion.ttft_s:.4f}s",
            )
            ensure(
                approx(
                    completion.total_time_s,
                    completion.ttft_s
                    + completion.tpot_s * max(1, completion.output_tokens - 1),
                    1e-6,
                ),
                f"{profile.title} req{completion.request.req_id} 的端到端时延应当等于"
                f"TTFT + TPOT×(输出token数−1)，实际 {completion.total_time_s:.4f}s",
            )

    short = results["short_qa"]
    long_doc = results["long_doc"]
    ensure(
        long_doc.queue_share > short.queue_share,
        f"长文档画像的排队占比 {long_doc.queue_share * 100:.1f}% 应当高于短问答的 "
        f"{short.queue_share * 100:.1f}%——长 prompt 的 prefill 会占住引擎，"
        "后面的请求只能等",
    )
    ensure(
        long_doc.metrics["ttft_p50_ms"] > short.metrics["ttft_p50_ms"] * 3,
        f"长文档画像的 TTFT p50 {long_doc.metrics['ttft_p50_ms']:.0f}ms 应当是短问答的 "
        f"{short.metrics['ttft_p50_ms']:.0f}ms 的三倍以上",
    )
    rule(
        "排队段归调度层，prefill 段归执行层的算力，decode 段归执行层的带宽——"
        "定位时延必须先分到段，再分到层"
    )


def check_slo_triangle() -> None:
    """判断：TTFT / TPOT / 吞吐不能同时最优，加负载必然牺牲前两者。"""
    section("判断二：TTFT / TPOT / 吞吐的三角")
    device = gpu(DEVICE)

    rows: list[list[object]] = []
    for profile in PROFILES:
        light = run_profile(device, profile, gap_s=0.5)  # 2 QPS
        heavy = run_profile(device, profile, gap_s=1.0 / BASE_QPS)  # 8 QPS
        rows.append(
            [
                profile.title,
                f"{light.metrics['output_tokens_per_s']:.0f}",
                f"{heavy.metrics['output_tokens_per_s']:.0f}",
                f"{light.metrics['ttft_p50_ms']:.0f} → {heavy.metrics['ttft_p50_ms']:.0f}",
                f"{light.metrics['tpot_p50_ms']:.1f} → {heavy.metrics['tpot_p50_ms']:.1f}",
                f"{light.metrics['sla_violation_rate'] * 100:.0f}% → "
                f"{heavy.metrics['sla_violation_rate'] * 100:.0f}%",
            ]
        )
    print_table(
        [
            "画像",
            "吞吐(轻) tok/s",
            "吞吐(重) tok/s",
            "TTFT p50 轻→重",
            "TPOT p50 轻→重",
            "TTFT 违约率 轻→重",
        ],
        rows,
    )

    for profile in PROFILES:
        light = run_profile(device, profile, gap_s=0.5)
        heavy = run_profile(device, profile, gap_s=1.0 / BASE_QPS)

        ensure(
            heavy.metrics["output_tokens_per_s"] > light.metrics["output_tokens_per_s"],
            f"{profile.title} 画像加重负载后吞吐应当上升，实际 "
            f"{heavy.metrics['output_tokens_per_s']:.0f} vs "
            f"{light.metrics['output_tokens_per_s']:.0f}",
        )
        # TPOT 允许 2% 的数值抖动：batch 变大时单步时延只微微变差，
        # 真正把 TPOT 顶上去的是"上下文长 + 并发多"的组合
        ensure(
            heavy.metrics["tpot_p50_ms"] >= light.metrics["tpot_p50_ms"] * 0.98,
            f"{profile.title} 画像加重负载后 TPOT 不该明显变好，实际 "
            f"{heavy.metrics['tpot_p50_ms']:.2f}ms vs {light.metrics['tpot_p50_ms']:.2f}ms",
        )
        ensure(
            heavy.metrics["ttft_p50_ms"] >= light.metrics["ttft_p50_ms"] * 0.95,
            f"{profile.title} 画像加重负载后 TTFT 不该明显变好，实际 "
            f"{heavy.metrics['ttft_p50_ms']:.0f}ms vs {light.metrics['ttft_p50_ms']:.0f}ms",
        )

    heavy_long = run_profile(device, PROFILES[1], gap_s=1.0 / BASE_QPS)
    light_long = run_profile(device, PROFILES[1], gap_s=0.5)
    ensure(
        heavy_long.metrics["ttft_p50_ms"] > light_long.metrics["ttft_p50_ms"],
        f"长文档画像加重负载后 TTFT 应当变差，实际 "
        f"{heavy_long.metrics['ttft_p50_ms']:.0f}ms vs {light_long.metrics['ttft_p50_ms']:.0f}ms",
    )
    ensure(
        heavy_long.metrics["sla_violation_rate"] > light_long.metrics["sla_violation_rate"],
        f"长文档画像加重负载后 TTFT 违约率应当上升，实际 "
        f"{heavy_long.metrics['sla_violation_rate'] * 100:.0f}% vs "
        f"{light_long.metrics['sla_violation_rate'] * 100:.0f}%",
    )
    ensure(
        heavy_long.metrics["sla_violation_rate"] > 0,
        "长文档画像在 8 QPS 下应当出现 TTFT 违约（截止线 = 到达 + 1s）",
    )
    rule(
        "吞吐要涨只能靠把更多请求塞进同一个 batch——"
        "代价是每个请求的排队时间和出字间隔一起变差，三角没有第四个解"
    )


def check_profiles_differ() -> None:
    """判断：同一张卡、同一个 QPS，不同画像的 SLO 表现差着量级。"""
    section("判断三：容量必须按负载画像算，不能按平均")
    device = gpu(DEVICE)
    results = {p.key: run_profile(device, p) for p in PROFILES}

    print_table(
        [
            "画像",
            "TTFT p50",
            "TTFT p99",
            "TPOT p50",
            "端到端 p99",
            "token 吞吐",
            "请求吞吐",
            "TTFT 违约率",
        ],
        [
            [
                profile.title,
                f"{results[profile.key].metrics['ttft_p50_ms']:.0f}ms",
                f"{results[profile.key].metrics['ttft_p99_ms']:.0f}ms",
                f"{results[profile.key].metrics['tpot_p50_ms']:.1f}ms",
                f"{results[profile.key].metrics['e2e_p99_ms']:.0f}ms",
                f"{results[profile.key].metrics['output_tokens_per_s']:.0f} tok/s",
                f"{results[profile.key].metrics['requests_per_s']:.1f} req/s",
                f"{results[profile.key].metrics['sla_violation_rate'] * 100:.0f}%",
            ]
            for profile in PROFILES
        ],
    )

    short = results["short_qa"]
    long_doc = results["long_doc"]
    streaming = results["streaming"]

    ensure(
        long_doc.metrics["ttft_p50_ms"] > short.metrics["ttft_p50_ms"] * 3,
        f"同样是 {BASE_QPS:.0f} QPS，长文档画像的 TTFT p50 "
        f"{long_doc.metrics['ttft_p50_ms']:.0f}ms 应当远高于短问答的 "
        f"{short.metrics['ttft_p50_ms']:.0f}ms",
    )
    ensure(
        long_doc.metrics["sla_violation_rate"] > 0
        and short.metrics["sla_violation_rate"] == 0,
        f"同样是 {BASE_QPS:.0f} QPS，长文档画像应当违约（"
        f"{long_doc.metrics['sla_violation_rate'] * 100:.0f}%），"
        f"短问答不应违约（{short.metrics['sla_violation_rate'] * 100:.0f}%）",
    )
    ensure(
        streaming.metrics["tpot_p50_ms"] < short.metrics["tpot_p50_ms"] * 1.5,
        f"流式长输出的 TPOT {streaming.metrics['tpot_p50_ms']:.1f}ms 不该比短问答的 "
        f"{short.metrics['tpot_p50_ms']:.1f}ms 高太多——"
        "TPOT 由 batch 与上下文决定，与输出长度无关",
    )
    ensure(
        streaming.metrics["requests_per_s"] < short.metrics["requests_per_s"],
        f"流式长输出的请求吞吐 {streaming.metrics['requests_per_s']:.1f} req/s 应当低于短问答的 "
        f"{short.metrics['requests_per_s']:.1f} req/s——长输出占着 batch 槽位，"
        "单位时间放行的请求数反而更少",
    )
    ensure(
        streaming.metrics["e2e_p99_ms"] > short.metrics["e2e_p99_ms"] * 5,
        f"流式长输出的端到端 p99 {streaming.metrics['e2e_p99_ms']:.0f}ms 应当远高于短问答的 "
        f"{short.metrics['e2e_p99_ms']:.0f}ms",
    )
    rule(
        "同一张卡、同一个 QPS，三种画像的 TTFT 差出一个数量级、违约率从 0 到有——"
        "容量必须按画像分别算"
    )


def check_unit_economics() -> None:
    """判断：跨方案比较要看每 token 成本，每卡 QPS 会骗人。"""
    section("判断四：单位经济——每 token 成本 vs 每卡 QPS")
    device = gpu(DEVICE)
    spec = model(MODEL)
    results = {p.key: run_profile(device, p) for p in PROFILES}

    print_table(
        ["画像", "墙钟跨度", "卡时成本", "每请求成本", "每百万输出token", "每百万处理token", "平均端到端"],
        [
            [
                profile.title,
                f"{results[profile.key].span_s:.2f}s",
                f"${results[profile.key].total_cost_usd:.4f}",
                f"${results[profile.key].cost_per_request_usd * 1e6:.1f}μ",
                f"${results[profile.key].cost_per_mtok_output_usd:.2f}",
                f"${results[profile.key].cost_per_mtok_processed_usd:.2f}",
                f"{results[profile.key].avg_e2e_s:.2f}s",
            ]
            for profile in PROFILES
        ],
    )

    for profile in PROFILES:
        result = results[profile.key]
        expected = device.price_per_hour_usd * result.span_s / 3600.0
        ensure(
            approx(result.total_cost_usd, expected, 1e-9),
            f"{profile.title} 的卡时成本应当等于 单价×跨度/3600，实际 "
            f"{result.total_cost_usd:.6f} vs {expected:.6f}",
        )
        ensure(
            approx(
                result.cost_per_mtok_output_usd,
                result.total_cost_usd / result.output_tokens * 1e6,
                1e-9,
            ),
            f"{profile.title} 的每百万输出 token 成本应当等于 总成本÷输出token×1e6",
        )
        ensure(
            approx(
                result.cost_per_mtok_processed_usd,
                result.total_cost_usd / (result.input_tokens + result.output_tokens) * 1e6,
                1e-9,
            ),
            f"{profile.title} 的每百万处理 token 成本应当等于 总成本÷(输入+输出)token×1e6",
        )

    short = results["short_qa"]
    long_doc = results["long_doc"]
    streaming = results["streaming"]
    ensure(
        streaming.cost_per_request_usd > short.cost_per_request_usd * 2,
        f"流式长输出的每请求卡时成本 ${streaming.cost_per_request_usd * 1e6:.1f}μ 应当是短问答的 "
        f"${short.cost_per_request_usd * 1e6:.1f}μ 的两倍以上",
    )
    ensure(
        short.cost_per_mtok_output_usd > streaming.cost_per_mtok_output_usd * 3,
        f"短问答的每百万输出 token 成本 ${short.cost_per_mtok_output_usd:.2f} 应当高于流式长输出的 "
        f"${streaming.cost_per_mtok_output_usd:.2f}——"
        "短问答的输出 token 少，固定开销摊到每个 token 上就贵",
    )
    ensure(
        long_doc.cost_per_mtok_processed_usd < short.cost_per_mtok_processed_usd / 3,
        f"长文档画像的每百万处理 token 成本 ${long_doc.cost_per_mtok_processed_usd:.2f} 应当远低于短问答的 "
        f"${short.cost_per_mtok_processed_usd:.2f}——它的 token 大头是输入，不是输出",
    )

    # 饱和压测：把到达率拉到 100 QPS，量出每种画像在这张卡上的可持续容量。
    # 这是"容量必须按画像算"的直接证据——同样一张卡，三种画像的可持续 req/s 差着量级。
    rows: list[list[object]] = []
    saturated: dict[str, ProfileResult] = {}
    for profile in PROFILES:
        result = run_profile(
            device,
            profile,
            gap_s=SATURATION_GAP,
            requests=SATURATION_REQUESTS,
        )
        saturated[profile.key] = result
        rows.append(
            [
                profile.title,
                f"{result.metrics['requests_per_s']:.1f}",
                f"{result.metrics['output_tokens_per_s']:.0f}",
                f"{result.span_s:.1f}s",
                f"{result.completions[0].queue_time_s * 1000:.0f}ms → "
                f"{result.completions[-1].queue_time_s * 1000:.0f}ms",
            ]
        )
    print_table(
        ["画像", "可持续 req/s", "可持续 tok/s", "压测墙钟", "首条→末条排队时延"],
        rows,
    )

    short_cap = saturated["short_qa"].metrics["requests_per_s"]
    long_cap = saturated["long_doc"].metrics["requests_per_s"]
    stream_cap = saturated["streaming"].metrics["requests_per_s"]
    ensure(
        short_cap > long_cap * 5,
        f"饱和时短问答的可持续吞吐 {short_cap:.1f} req/s 应当是长文档的 "
        f"{long_cap:.1f} req/s 的五倍以上",
    )
    ensure(
        short_cap > stream_cap * 5,
        f"饱和时短问答的可持续吞吐 {short_cap:.1f} req/s 应当是流式长输出的 "
        f"{stream_cap:.1f} req/s 的五倍以上",
    )
    ensure(
        short_cap > BASE_QPS,
        f"短问答画像在单卡上应当撑得住 {BASE_QPS:.0f} QPS，实际可持续 {short_cap:.1f} req/s",
    )
    ensure(
        stream_cap < BASE_QPS,
        f"流式长输出画像在单卡上撑不住 {BASE_QPS:.0f} QPS，实际可持续只有 {stream_cap:.1f} req/s——"
        "同样的 QPS，换个画像就得加副本",
    )
    ensure(
        saturated["streaming"].completions[-1].queue_time_s
        > saturated["short_qa"].completions[-1].queue_time_s * 10,
        "饱和时流式长输出的排队时延应当远高于短问答——队列在长输出画像下会持续变长",
    )

    # 单流口径下，一条请求要占卡多久：这就是 Little's law 的 W
    rows: list[list[object]] = []
    for profile in PROFILES:
        delta = load_balancing_delta(
            device,
            spec,
            replicas=1,
            qps=BASE_QPS,
            avg_prompt_tokens=profile.prompt_tokens,
            avg_output_tokens=profile.output_tokens,
            mfu=MFU,
        )
        concurrency = BASE_QPS * delta["per_request_s"]
        rows.append(
            [
                profile.title,
                f"{delta['per_request_s']:.3f}s",
                f"{delta['capacity_per_replica']:.2f} req/s",
                f"{concurrency:.1f} 条",
                f"{delta['utilization'] * 100:.0f}%",
            ]
        )
    print_table(
        ["画像", "单流占用时间", "单卡理论容量", "8QPS 所需并发", "利用率"],
        rows,
    )

    short_concurrency = BASE_QPS * load_balancing_delta(
        device,
        spec,
        replicas=1,
        qps=BASE_QPS,
        avg_prompt_tokens=PROFILES[0].prompt_tokens,
        avg_output_tokens=PROFILES[0].output_tokens,
        mfu=MFU,
    )["per_request_s"]
    stream_concurrency = BASE_QPS * load_balancing_delta(
        device,
        spec,
        replicas=1,
        qps=BASE_QPS,
        avg_prompt_tokens=PROFILES[2].prompt_tokens,
        avg_output_tokens=PROFILES[2].output_tokens,
        mfu=MFU,
    )["per_request_s"]

    ensure(
        stream_concurrency > short_concurrency * 10,
        f"流式长输出在 {BASE_QPS:.0f} QPS 下需要 {stream_concurrency:.1f} 条并发，"
        f"应当远高于短问答的 {short_concurrency:.1f} 条——"
        "请求占用时间越长，同时服务的请求就越少",
    )
    ensure(
        stream_concurrency > MAX_BATCH,
        f"流式长输出需要 {stream_concurrency:.1f} 条并发，已经超过 max_batch={MAX_BATCH}，"
        "说明它必须靠多副本或排队才能守住到达率",
    )
    rule(
        f"同样的 {BASE_QPS:.0f} QPS：单卡可持续吞吐 {short_cap:.1f} vs {stream_cap:.1f} req/s（差 "
        f"{short_cap / stream_cap:.0f} 倍），所需并发 {short_concurrency:.1f} vs "
        f"{stream_concurrency:.1f} 条——容量和成本都必须按画像算，每卡 QPS 是个会骗人的指标"
    )


def run() -> None:
    """跑第 1 章实验。"""
    banner("第 1 章 · 推理服务全景与 SLO")
    check_lifecycle_decomposition()
    check_slo_triangle()
    check_profiles_differ()
    check_unit_economics()
    rule(
        "时延先分三段再分到层；TTFT/TPOT/吞吐三角不可兼得；"
        "容量按负载画像算，单位经济按每 token 成本算"
    )
    print("\n✓ 第 1 章通过")
