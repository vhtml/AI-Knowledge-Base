"""第 16 章实验：容量规划与上线检查单。

这个模块解决什么问题：上线前最容易被拍脑袋的两个数是"要几台机器"和"每百万 token 多少钱"。
本章把这两个数变成一条可验证的推导链：

1. **容量从负载画像长出来**——先给输入/输出长度画像，再用 labkit 的
   ``load_balancing_delta`` 反推副本数，而不是先定机器数再倒推能扛多少。
2. **冗余要算给谁看**——N+1 扛不扛得住"坏一台"，跨可用区扛不扛得住"坏一个 AZ"。
3. **成本按卡时折算**——并行副本的墙钟取**最大**而不是求和，
   否则会把并行算成串行，每百万 token 成本会凭空翻倍。
4. **平均值会骗人**——均值相同、token 总量相同的两种负载画像，
   真实吞吐能差近两倍；而单流模型对长尾几乎不敏感，
   所以"按平均值算容量"会系统性低估所需副本数。

所有算账都走 :mod:`infra_lab.labkit`，本章只写容量规划本身的推导。
全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..labkit import (
    EngineSimulator,
    GPUSpec,
    ModelSpec,
    Request,
    banner,
    ensure,
    fmt_gib,
    fmt_ms,
    gpu,
    load_balancing_delta,
    model,
    plan_memory,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
"""容量表用的参考机型：H100 SXM 80GB。"""

SPEC = model("qwen2.5-7b")
"""容量表用的参考模型：Qwen2.5-7B-Instruct。"""

MFU = 0.5
"""工程折损系数，与 labkit 全部时延模型同一口径。"""

TARGET_UTILIZATION = 0.6
"""目标利用率。留 40% 余量给突发、故障与长请求——不是拍脑袋，是下面
``check_redundancy`` 里"坏一台还扛不扛得住"这条约束反推出来的。"""

PEAK_QPS = 1.0
"""峰值 QPS。容量表按这个流量算，改一个数整张表跟着变。"""

AVAILABILITY_ZONES = 3
"""可用区数量。跨可用区部署时按"坏一个区"算冗余。"""

EVIDENCE_REQUESTS = 200
"""长尾对比实验里的请求数。取 200 是为了让 85%/13%/2% 的比例落成整数。"""

EVIDENCE_QPS = 12.0
"""长尾对比实验对应的流量。取 12 QPS 是为了让副本数落在整数级，
否则会出现"0.09 个副本"这种没有工程意义的数。"""


@dataclass(frozen=True)
class LoadProfile:
    """一份负载画像。

    只用"平均输入长度 + 平均输出长度"两个数是**不够**的，
    画像里必须能表达长度分布的形状，所以这里保留扩展字段。
    """

    name: str
    prompt_tokens: int
    output_tokens: int
    note: str = ""


PROFILES: tuple[LoadProfile, ...] = (
    LoadProfile("短问答", 200, 80, "客服/检索命中后的一句话"),
    LoadProfile("常规对话", 500, 200, "最常见的多轮补全"),
    LoadProfile("RAG 长文档", 2000, 300, "带一篇文档的问答"),
    LoadProfile("Agent 工具链", 1200, 600, "多轮工具调用，输出很长"),
)
"""四种典型画像。容量表按这四种算。"""


@dataclass(frozen=True)
class HardwareOption:
    """一个部署方案（硬件 + 模型 + 精度 + 切分方式）。"""

    name: str
    device: GPUSpec
    spec: ModelSpec
    cards_per_replica: int
    """一个副本占几张卡。大于 1 表示张量并行切分，成本要按卡数算。"""
    weight_bits: int
    description: str


def tp_shard(spec: ModelSpec, tp_degree: int) -> ModelSpec:
    """把模型规格切成分到每张卡上的**单卡视角**规格。

    为什么这样切：张量并行下每张卡只存 1/N 的权重、只算 1/N 的注意力头，
    但所有卡是并行干活的，所以"单卡耗时"就是"整个 TP 组的耗时"。
    切完交给 labkit 的 ``decode_step_time`` / ``prefill_time``，
    显存与带宽口径就自动和前面各章对齐了，本章不用另写一套公式。

    ``hidden_size`` 也要除以 TP 度数：注意力 FLOPs 按头切分，
    只切 KV 头会把注意力算力算重，让切分看起来比实际更慢。
    """
    if tp_degree <= 1:
        return spec
    return replace(
        spec,
        name=f"{spec.name}（TP-{tp_degree} 单卡视角）",
        params_b=spec.params_b / tp_degree,
        embedding_params_b=spec.embedding_params_b / tp_degree,
        hidden_size=spec.hidden_size // tp_degree,
        num_heads=spec.num_heads // tp_degree,
        num_kv_heads=max(1, spec.num_kv_heads // tp_degree),
    )


def hardware_options() -> tuple[HardwareOption, ...]:
    """三种候选硬件方案：7B FP16 / 7B INT8 / 70B 张量并行切分。"""
    small = model("qwen2.5-7b")
    big = model("llama3.1-70b")
    return (
        HardwareOption(
            name="7B FP16 · 1×L40S",
            device=gpu("l40s"),
            spec=small,
            cards_per_replica=1,
            weight_bits=16,
            description="不量化，权重 14.19GiB，显存宽裕、精度无损",
        ),
        HardwareOption(
            name="7B INT8 · 1×L40S",
            device=gpu("l40s"),
            spec=replace(small, weight_bits=8),
            cards_per_replica=1,
            weight_bits=8,
            description="权重砍半，decode 访存减半（第 9 章）",
        ),
        HardwareOption(
            name="70B FP16 · TP-4 × H100",
            device=gpu("h100_sxm"),
            spec=tp_shard(big, 4),
            cards_per_replica=4,
            weight_bits=16,
            description="单卡装不下 131.5GiB 权重，必须 4 卡切分",
        ),
    )


def capacity_plan(
    profile: LoadProfile,
    *,
    device: GPUSpec = DEVICE,
    spec: ModelSpec = SPEC,
    qps: float = PEAK_QPS,
    mfu: float = MFU,
) -> dict[str, float]:
    """按一份负载画像算出容量账。

    口径全部来自 labkit 的 ``load_balancing_delta``：
    单请求占用时间 = prefill + decode，副本数 = QPS ÷ 单副本容量 ÷ 目标利用率。
    """
    delta = load_balancing_delta(
        device,
        spec,
        replicas=1,
        qps=qps,
        avg_prompt_tokens=profile.prompt_tokens,
        avg_output_tokens=profile.output_tokens,
        mfu=mfu,
    )
    per_request = delta["per_request_s"]
    capacity = delta["capacity_per_replica"]
    # delta["replicas_needed"] 是 100% 利用率下的副本数，除以目标利用率才是规划值
    replicas = delta["replicas_needed"] / TARGET_UTILIZATION
    return {
        "per_request_s": per_request,
        "capacity_per_replica": capacity,
        "replicas_raw": delta["replicas_needed"],
        "replicas_planned": replicas,
        "replicas_deployed": int(replicas) + 1,
    }


def redundancy_plan(
    profile: LoadProfile,
    *,
    device: GPUSpec = DEVICE,
    spec: ModelSpec = SPEC,
    qps: float = PEAK_QPS,
    mfu: float = MFU,
) -> dict[str, float]:
    """算冗余：坏一台、坏一个可用区之后还扛不扛得住峰值。

    两种冗余口径必须分开报，因为它们的代价差得很远：
    坏一台只需要多 1 个副本，坏一个可用区要按 1/3 的容量损失来补。
    """
    plan = capacity_plan(profile, device=device, spec=spec, qps=qps, mfu=mfu)
    capacity = plan["capacity_per_replica"]
    needed = plan["replicas_planned"]
    """按目标利用率算出的副本数（可能是小数）。"""
    needed_at_full = plan["replicas_raw"]
    """按 100% 利用率算出的副本数——这就是"不留任何余量"的配法。"""

    # 不加冗余：按目标利用率配够，坏一台就少一个，余量被吃光
    without = math.ceil(needed)
    util_without_failure = qps / (capacity * (without - 1)) if without > 1 else float("inf")

    # N+1：多部署一个，坏一台后还剩 without 个
    with_redundancy = without + 1
    util_with_failure = qps / (capacity * without) if without > 0 else float("inf")

    # 按 100% 利用率配、坏一台：必然过载
    util_full_then_failure = (
        qps / (capacity * (math.ceil(needed_at_full) - 1))
        if math.ceil(needed_at_full) > 1
        else float("inf")
    )

    # 跨可用区：找一个最小的副本数 n，使得 n - ceil(n/3) 仍能扛住峰值
    az_total = math.ceil(needed)
    while az_total - -(-az_total // AVAILABILITY_ZONES) < needed:
        az_total += 1
    az_surviving = az_total - -(-az_total // AVAILABILITY_ZONES)
    util_after_az_loss = qps / (capacity * az_surviving) if az_surviving > 0 else float("inf")

    return {
        "replicas_needed": needed,
        "replicas_no_redundancy": without,
        "replicas_with_n_plus_one": with_redundancy,
        "util_after_one_failure_without_redundancy": util_without_failure,
        "util_after_one_failure_with_n_plus_one": util_with_failure,
        "util_full_sizing_after_failure": util_full_then_failure,
        "replicas_for_zone_failover": az_total,
        "util_after_zone_loss": util_after_az_loss,
    }


def cost_of_request(
    option: HardwareOption,
    profile: LoadProfile,
    *,
    mfu: float = MFU,
) -> dict[str, float]:
    """算一个副本跑完一条请求要花多少钱，以及每百万 token 的成本。

    关键口径：**并行副本的墙钟取最大而不是求和**。
    一个副本由 N 张卡组成时，这 N 张卡是并行干活的，
    所以一条请求占用的是"单卡耗时 × N 张卡的租金"，而不是"墙钟 × N"。
    把并行算成串行，每百万 token 成本会直接翻 N 倍。
    """
    delta = load_balancing_delta(
        option.device,
        option.spec,
        replicas=1,
        qps=1.0,
        avg_prompt_tokens=profile.prompt_tokens,
        avg_output_tokens=profile.output_tokens,
        mfu=mfu,
    )
    per_request = delta["per_request_s"]
    hourly = option.device.price_per_hour_usd * option.cards_per_replica
    cost_per_request = per_request / 3600.0 * hourly
    total_tokens = profile.prompt_tokens + profile.output_tokens
    return {
        "per_request_s": per_request,
        "hourly_usd": hourly,
        "cost_per_request_usd": cost_per_request,
        "usd_per_million_total": cost_per_request / total_tokens * 1e6,
        "usd_per_million_prompt": cost_per_request / profile.prompt_tokens * 1e6,
        "usd_per_million_output": cost_per_request / profile.output_tokens * 1e6,
    }


def _build_requests(lengths: list[tuple[int, int]]) -> list[Request]:
    """把 (输入长度, 输出长度) 列表变成同时到达的一批请求。

    全部在 t=0 到达，模拟"压满"的状态：引擎从第一刻就是满负荷，
    跑完的总墙钟就是这台副本的真实容量。
    """
    return [
        Request(
            req_id=index,
            arrival_s=0.0,
            prompt_tokens=prompt,
            max_output_tokens=output,
            tenant="default",
        )
        for index, (prompt, output) in enumerate(lengths)
    ]


def _uniform_lengths(count: int, prompt: int, output: int) -> list[tuple[int, int]]:
    """均匀画像：每条请求长度一样。"""
    return [(prompt, output) for _ in range(count)]


def _long_tail_lengths(count: int) -> list[tuple[int, int]]:
    """长尾画像：均值与均匀画像**完全相同**，但形状完全不同。

    85% 短请求 / 13% 中等 / 2% 超长。
    均值对齐是刻意的——只有这样，两边的 token 总量才相同，
    吞吐对比才是"同工不同速"，而不是"总量不同"。
    """
    lengths: list[tuple[int, int]] = []
    short_count = int(count * 0.85)
    medium_count = int(count * 0.13)
    tail_count = count - short_count - medium_count
    lengths.extend([(400, 80)] * short_count)
    lengths.extend([(2615, 708)] * medium_count)
    lengths.extend([(16000, 2000)] * tail_count)
    return lengths


def _measure_capacity(lengths: list[tuple[int, int]]) -> dict[str, float]:
    """用引擎模拟器压满一台副本，量出它的真实吞吐。

    墙钟跨度由**最慢的那条请求**决定——这正是长尾拖垮容量的机制：
    均值一样，尾巴不一样，墙钟就不一样。
    """
    engine = EngineSimulator(
        DEVICE,
        SPEC,
        max_batch=32,
        kv_token_budget=400_000,
        mfu=MFU,
    )
    engine.run(_build_requests(lengths), horizon=1200.0)
    metrics = engine.metrics()
    return {
        "requests_per_s": metrics["requests_per_s"],
        "output_tokens_per_s": metrics["output_tokens_per_s"],
        "ttft_p99_ms": metrics["ttft_p99_ms"],
        "tpot_p50_ms": metrics["tpot_p50_ms"],
        "e2e_p99_ms": metrics["e2e_p99_ms"],
    }


def check_capacity_table() -> None:
    """容量表：副本数是负载画像的函数，不是拍出来的。"""
    section("1. 容量表：不同负载画像要几台机器")
    rows: list[list[str]] = []
    plans: dict[str, dict[str, float]] = {}
    for profile in PROFILES:
        plan = capacity_plan(profile)
        plans[profile.name] = plan
        rows.append(
            [
                profile.name,
                f"{profile.prompt_tokens}/{profile.output_tokens}",
                fmt_ms(plan["per_request_s"]),
                f"{plan['capacity_per_replica']:.3f}",
                f"{plan['replicas_planned']:.2f}",
                str(plan["replicas_deployed"]),
            ]
        )
    print_table(
        ["画像", "输入/输出", "单请求占用", "单副本容量/s", "规划副本数", "含冗余部署"],
        rows,
    )

    short = plans["短问答"]
    agent = plans["Agent 工具链"]
    ensure(
        agent["per_request_s"] > short["per_request_s"] * 3,
        f"长输出画像的单请求占用应显著更高，实际 {fmt_ms(agent['per_request_s'])} vs "
        f"{fmt_ms(short['per_request_s'])}",
    )
    ensure(
        agent["replicas_planned"] > short["replicas_planned"],
        "长输出画像需要的副本数必须更多",
    )

    # 副本数必须随 QPS 线性变化——这是"算出来的"而不是"拍的"的直接证据
    double = capacity_plan(PROFILES[1], qps=PEAK_QPS * 2)
    base = plans["常规对话"]
    ensure(
        abs(double["replicas_planned"] - base["replicas_planned"] * 2) < 1e-6,
        "QPS 翻倍时规划副本数必须精确翻倍",
    )
    rule(
        f"画像从'短问答'换成'Agent 工具链'，单请求占用从 "
        f"{fmt_ms(short['per_request_s'])} 涨到 {fmt_ms(agent['per_request_s'])}，"
        f"副本数从 {short['replicas_planned']:.2f} 涨到 {agent['replicas_planned']:.2f}"
    )


def check_redundancy() -> None:
    """冗余：N+1 扛得住坏一台，跨可用区扛得住坏一个区。"""
    section("2. 冗余：坏一台 / 坏一个可用区之后还扛不扛得住")
    rows: list[list[str]] = []
    for profile in PROFILES:
        plan = redundancy_plan(profile)
        rows.append(
            [
                profile.name,
                f"{plan['replicas_needed']:.2f}",
                str(plan["replicas_no_redundancy"]),
                str(plan["replicas_with_n_plus_one"]),
                f"{plan['util_after_one_failure_without_redundancy']:.2f}",
                f"{plan['util_after_one_failure_with_n_plus_one']:.2f}",
                str(plan["replicas_for_zone_failover"]),
                f"{plan['util_after_zone_loss']:.2f}",
            ]
        )
    print_table(
        [
            "画像",
            "需求副本",
            "无冗余",
            "N+1",
            "无冗余坏一台",
            "N+1 坏一台",
            "跨AZ副本",
            "坏一个AZ",
        ],
        rows,
    )

    for profile in PROFILES:
        plan = redundancy_plan(profile)
        ensure(
            plan["util_after_one_failure_with_n_plus_one"] <= TARGET_UTILIZATION + 1e-9,
            f"{profile.name}：N+1 布局坏一台后仍应在目标利用率 "
            f"{TARGET_UTILIZATION} 以内，实际 {plan['util_after_one_failure_with_n_plus_one']:.3f}",
        )
        ensure(
            plan["util_after_one_failure_without_redundancy"] > TARGET_UTILIZATION,
            f"{profile.name}：不加冗余时坏一台会把余量吃光，"
            f"实际利用率 {plan['util_after_one_failure_without_redundancy']:.3f}",
        )
        ensure(
            plan["util_full_sizing_after_failure"] > 1.0,
            f"{profile.name}：按 100% 利用率配、坏一台必然过载，"
            f"实际利用率 {plan['util_full_sizing_after_failure']:.3f}",
        )
        ensure(
            plan["replicas_for_zone_failover"] >= plan["replicas_with_n_plus_one"],
            f"{profile.name}：跨可用区的副本数不应少于 N+1",
        )
    rule(
        "余量必须来自目标利用率，不是来自'多买一台'："
        "按 100% 利用率配，坏一台必然过载；跨可用区还要按 1/3 的容量损失补副本"
    )


def check_cost_per_million() -> None:
    """成本：每百万 token 多少钱，以及并行副本不能算成串行。"""
    section("3. 每百万 token 成本：三种硬件方案")
    profile = PROFILES[1]
    rows: list[list[str]] = []
    results: dict[str, dict[str, float]] = {}
    for option in hardware_options():
        cost = cost_of_request(option, profile)
        results[option.name] = cost
        rows.append(
            [
                option.name,
                f"{option.cards_per_replica}",
                f"${option.device.price_per_hour_usd:.2f}",
                f"${cost['hourly_usd']:.2f}",
                fmt_ms(cost["per_request_s"]),
                f"${cost['cost_per_request_usd']:.6f}",
                f"${cost['usd_per_million_total']:.2f}",
            ]
        )
    print_table(
        [
            "方案",
            "卡数",
            "单卡时薪",
            "副本时薪",
            "单请求占用",
            "每请求成本",
            "每百万token",
        ],
        rows,
    )

    fp16 = results["7B FP16 · 1×L40S"]
    int8 = results["7B INT8 · 1×L40S"]
    tp4 = results["70B FP16 · TP-4 × H100"]

    ensure(
        int8["usd_per_million_total"] < fp16["usd_per_million_total"],
        "INT8 的每百万 token 成本必须低于 FP16——权重访存减半，decode 更快",
    )
    ensure(
        tp4["usd_per_million_total"] > fp16["usd_per_million_total"],
        "70B TP-4 的每百万 token 成本必须高于 7B——大模型更贵，这是事实不是选择",
    )

    # 并行不能算成串行：TP-4 的墙钟是 4 张卡并行的结果
    single = model("llama3.1-70b")
    budget = plan_memory(single, gpu("h100_sxm"), kv_tokens=100_000)
    ensure(
        not budget.fits(),
        f"70B FP16 权重 {budget.weights_gib:.1f}GiB 单卡装不下，必须切分",
    )
    sharded = plan_memory(
        tp_shard(single, 4), gpu("h100_sxm"), kv_tokens=100_000
    )
    ensure(
        sharded.fits(),
        f"TP-4 切分后每卡权重 {sharded.weights_gib:.1f}GiB，应装得下",
    )
    rule(
        f"70B 单卡放不下（权重 {budget.weights_gib:.1f}GiB > 80GiB），"
        f"TP-4 后每卡 {sharded.weights_gib:.1f}GiB；"
        f"但 TP-4 的每百万 token 成本仍是 7B 的 "
        f"{tp4['usd_per_million_total'] / fp16['usd_per_million_total']:.1f} 倍"
    )


def check_long_tail_underestimation() -> None:
    """平均值会骗人：均值相同，长尾画像的真实容量差出一大截。"""
    section("4. 用平均值算容量，会低估多少")

    uniform = _uniform_lengths(EVIDENCE_REQUESTS, 1000, 200)
    tail = _long_tail_lengths(EVIDENCE_REQUESTS)

    uniform_prompt = sum(p for p, _ in uniform) / len(uniform)
    tail_prompt = sum(p for p, _ in tail) / len(tail)
    uniform_output = sum(o for _, o in uniform) / len(uniform)
    tail_output = sum(o for _, o in tail) / len(tail)
    ensure(
        abs(uniform_prompt - tail_prompt) < 5 and abs(uniform_output - tail_output) < 5,
        "两种画像的输入输出均值必须对齐，否则吞吐对比不成立",
    )

    # 单流模型对长尾几乎不敏感——这正是"按平均值算容量"会骗人的原因
    mean_plan = capacity_plan(LoadProfile("均值口径", 1000, 200))
    tail_profile = LoadProfile("长尾均值口径", int(tail_prompt), int(tail_output))
    tail_plan = capacity_plan(tail_profile)
    gap = abs(tail_plan["per_request_s"] - mean_plan["per_request_s"]) / mean_plan["per_request_s"]
    ensure(
        gap < 0.05,
        f"单流模型对长尾的敏感度应低于 5%（实测 {gap * 100:.2f}%），"
        "超过说明模型本身有问题，不是长尾问题",
    )

    uniform_metrics = _measure_capacity(uniform)
    tail_metrics = _measure_capacity(tail)
    print_table(
        ["画像", "输入均值", "输出均值", "总输出token", "吞吐token/s", "p99 TTFT", "p99 端到端"],
        [
            [
                "均匀",
                f"{uniform_prompt:.0f}",
                f"{uniform_output:.0f}",
                f"{sum(o for _, o in uniform)}",
                f"{uniform_metrics['output_tokens_per_s']:.0f}",
                f"{uniform_metrics['ttft_p99_ms']:.0f}ms",
                f"{uniform_metrics['e2e_p99_ms']:.0f}ms",
            ],
            [
                "长尾",
                f"{tail_prompt:.0f}",
                f"{tail_output:.0f}",
                f"{sum(o for _, o in tail)}",
                f"{tail_metrics['output_tokens_per_s']:.0f}",
                f"{tail_metrics['ttft_p99_ms']:.0f}ms",
                f"{tail_metrics['e2e_p99_ms']:.0f}ms",
            ],
        ],
    )

    ensure(
        tail_metrics["output_tokens_per_s"] < uniform_metrics["output_tokens_per_s"],
        "长尾画像的吞吐必须低于均值相同的均匀画像",
    )
    ratio = uniform_metrics["output_tokens_per_s"] / tail_metrics["output_tokens_per_s"]
    ensure(
        ratio > 1.2,
        f"长尾导致的吞吐差距应显著（实测 {ratio:.2f} 倍），"
        "差距太小说明模拟器没把长请求的代价体现出来",
    )

    # 两个副本数用**同一套引擎模型**量，才能比：按"均值画像"规划 vs 按长尾实测
    replicas_planned = EVIDENCE_QPS / uniform_metrics["requests_per_s"]
    replicas_true = EVIDENCE_QPS / tail_metrics["requests_per_s"]
    ensure(
        replicas_true > replicas_planned,
        f"按均值规划的副本数 {replicas_planned:.2f} 必须小于按长尾实测的 "
        f"{replicas_true:.2f}",
    )
    underestimate = (replicas_true - replicas_planned) / replicas_true
    rule(
        f"{EVIDENCE_QPS:.0f} QPS 下：按均值画像规划 {replicas_planned:.2f} 个副本，"
        f"长尾实测需要 {replicas_true:.2f} 个——按平均值算容量低估了 "
        f"{underestimate * 100:.0f}% 的副本"
    )

    # 顺带披露口径差距：单流模型是保守上界，批量摊薄（第 4 章）会把真实副本数压下来
    stream_replicas = mean_plan["replicas_planned"] * EVIDENCE_QPS / PEAK_QPS
    ensure(
        stream_replicas > replicas_planned,
        "单流模型给出的副本数应大于批量引擎实测的副本数（单流是保守上界）",
    )
    rule(
        f"同一份流量：单流模型要 {stream_replicas:.1f} 个副本，"
        f"批量引擎实测只要 {replicas_planned:.2f} 个——"
        f"容量公式用来定档位，最终数字必须靠压测校准"
    )


def run() -> None:
    """跑完第 16 章的全部检查。"""
    banner("第 16 章 · 容量规划与上线检查单")
    check_capacity_table()
    check_redundancy()
    check_cost_per_million()
    check_long_tail_underestimation()
    print()
    print("=" * 72)
    print("第 16 章结论：副本数 = QPS × 单请求占用时间 ÷ 目标利用率；")
    print("冗余按'坏一个可用区'算；成本按卡时折算（并行不求和）；")
    print("容量必须用负载画像算，用平均值会低估副本数。")
