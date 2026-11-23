"""第 17 章实验：训练侧 Infra 地基。

这个模块解决什么问题：推理工程师不一定要训模型，但**推理侧几乎所有"更聪明的模型"
都来自训练侧**——LoRA 微调、蒸馏出的小模型、量化校准数据、投机采样的 draft 模型。
不懂训练侧，就说不清这些产物是怎么来的、代价在哪、约束是什么。

本章把训练侧的四件事变成可验证的断言：

1. **四种并行策略的通信量**：数据并行每步 all-reduce 梯度，张量并行每层 2 次
   all-reduce 激活，流水线并行只传激活，专家并行 all-to-all。
   核心结论：**张量并行的通信量随层数线性增长**，且落在每层的关键路径上。
2. **集合通信的带宽模型**：ring all-reduce ≈ 2×(N-1)/N× 数据量 ÷ 带宽。
   用它就能算出"跨节点做 TP 会把多少训练时间花在通信上"。
3. **ZeRO 1/2/3 的显存拆分**：切优化器状态 → 再切梯度 → 连参数也切。
   70B 在 64 张 80GB 卡上只有 ZeRO-3 装得下。
4. **混合精度与激活重计算**：用 33% 的算力换 12 倍的激活显存。

模型维���全部取自 labkit 的 ``ModelSpec``（参数量、层数、hidden），
推理侧的算账口径（GiB、权重字节）也全部走 labkit。
本章只新定义训练侧自己的常量（优化器状态字节数、互联带宽），并逐条注明出处。

全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..labkit import (
    BYTES_PER_GIB,
    approx,
    ModelSpec,
    banner,
    ensure,
    fmt_gib,
    gpu,
    model,
    plan_memory,
    prefill_flops,
    print_table,
    quantize_values,
    rule,
    section,
    weight_gib,
)
from .ch09_quantization import build_activation_samples
from .ch10_speculative import DRAFT_VARIANTS, break_even_acceptance, draft_round_cost_s, target_step_s

DEVICE = gpu("h100_sxm")
"""参考机型：H100 SXM 80GB。"""

TARGET = model("llama3.1-70b")
"""参考模型：Llama-3.1-70B。层多、hidden 大，通信账最难看。"""

SMALL = model("qwen2.5-7b")
"""小模型：数据并行可行的那个。"""

MOE = model("mixtral-8x7b")
"""专家并行用的 MoE 模型。"""

PARAM_BYTES = 2
"""bf16 参数：2 字节/参数。与 labkit 里 KV 用 2 字节是同一精度口径。"""

GRAD_BYTES = 2
"""bf16 梯度：2 字节/参数。"""

ADAM_STATE_BYTES = 12
"""Adam 的优化器状态：fp32 主权重 4 + 一阶矩 4 + 二阶矩 4 = 12 字节/参数。
这是本课**首次定义**的训练侧口径，ZeRO 的显存账全部基于它。"""

ACTIVATION_BYTES = 2
"""bf16 激活：2 字节/元素。"""

NVLINK_GB_S = 900.0
"""H100 NVLink 4：单卡双向约 900 GB/s。节点内张量并行靠它。"""

PCIE_GB_S = 64.0
"""PCIe Gen5 x16：约 64 GB/s。同一台机器里两张卡的通用互联。"""

IB_GB_S = 50.0
"""400G InfiniBand：约 50 GB/s。跨节点互联，比 NVLink 慢一个数量级。"""

SAVED_TENSORS_PER_LAYER = 12
"""不做激活重计算时，每层大约要保存的 hidden 尺寸张量个数。
粗略口径：注意力输出 + MLP 中间结果。"""

RECOMPUTE_TENSORS_PER_LAYER = 1
"""做整层重计算时，每层只需要保存层输入（1 个张量）。"""

TRAIN_BATCH = 4
"""每张卡每次前向的 batch 数。"""

TRAIN_SEQ = 8192
"""每张卡每次前向的序列长度。"""

TRAIN_GPUS = 8
"""一个并行组的卡数。一个 8 卡节点。"""

MOE_TOP_K = 2
"""Mixtral 每 token 激活的专家数。"""

MOE_EXPERTS = 8
"""Mixtral 的专家总数。"""


@dataclass(frozen=True)
class CommPlan:
    """一种并行策略每步的通信账。"""

    name: str
    rounds: int
    """每步的集合通信次数。"""
    payload_bytes: float
    """单次集合通信的数据量。"""
    total_bytes: float
    """每步的通信总量。"""
    node_time_s: float
    """节点内（NVLink）的通信时间。"""
    cross_node_time_s: float
    """跨节点（IB）的通信时间。"""
    bottleneck: str
    """这个策略真正卡在哪——通信量不一定是大头。"""


def ring_all_reduce_time(payload_bytes: float, gpus: int, bandwidth_gbs: float) -> float:
    """ring all-reduce 的时间。

    口径：``2×(N-1)/N× 数据量 ÷ 带宽``。
    系数 2 来自两趟（reduce-scatter + all-gather），
    (N-1)/N 是环上每张卡实际负责的比例。
    """
    if gpus <= 1 or bandwidth_gbs <= 0:
        return 0.0
    return 2.0 * (gpus - 1) / gpus * payload_bytes / (bandwidth_gbs * 1e9)


def collective_times(payload_bytes: float, gpus: int, bandwidth_gbs: float) -> dict[str, float]:
    """四种集合通信原语的时间。

    只有 all-reduce 要跑两趟（reduce-scatter + all-gather），
    其余三个都是一趟，所以它们的通信量都只有 all-reduce 的一半。
    """
    if gpus <= 1:
        return {"all-reduce": 0.0, "all-gather": 0.0, "reduce-scatter": 0.0, "all-to-all": 0.0}
    one_pass = (gpus - 1) / gpus * payload_bytes / (bandwidth_gbs * 1e9)
    return {
        "all-reduce": 2.0 * one_pass,
        "all-gather": one_pass,
        "reduce-scatter": one_pass,
        "all-to-all": one_pass,
    }


def activation_payload_bytes(spec: ModelSpec, batch: int, seq: int) -> float:
    """张量并行/流水线并行每层要交换的激活字节数。

    口径：batch × seq × hidden × 2 字节。
    注意这个量和**模型参数量无关**，只和 hidden、序列长度有关——
    这正是 TP 和 DP 的通信量差一个数量级的原因。
    """
    return batch * seq * spec.hidden_size * ACTIVATION_BYTES


def training_step_flops(spec: ModelSpec, tokens: int) -> float:
    """一步训练的浮点运算量。

    口径：前向 1 份 + 反向 2 份 = 3 份前向。
    前向的量取 labkit 的 ``prefill_flops``（同样的 GEMM + 注意力结构），
    所以训练侧和推理侧的模型口径不会漂。
    """
    return 3.0 * prefill_flops(spec, tokens)


def parallel_comm_plans(
    spec: ModelSpec,
    *,
    gpus: int = TRAIN_GPUS,
    batch: int = TRAIN_BATCH,
    seq: int = TRAIN_SEQ,
    stages: int | None = None,
) -> list[CommPlan]:
    """算四种并行策略每步的通信账。"""
    stages = stages or gpus
    payload = activation_payload_bytes(spec, batch, seq)
    grad_payload = spec.params_b * 1e9 * GRAD_BYTES

    dp_rounds = 1
    dp_total = grad_payload
    dp_node = ring_all_reduce_time(grad_payload, gpus, NVLINK_GB_S)
    dp_cross = ring_all_reduce_time(grad_payload, gpus, IB_GB_S)

    tp_rounds = 2 * spec.num_layers
    tp_total = tp_rounds * payload
    tp_node = tp_rounds * ring_all_reduce_time(payload, gpus, NVLINK_GB_S)
    tp_cross = tp_rounds * ring_all_reduce_time(payload, gpus, IB_GB_S)

    pp_rounds = 2 * (stages - 1)
    pp_total = pp_rounds * payload
    pp_node = pp_rounds * payload / (NVLINK_GB_S * 1e9)
    pp_cross = pp_rounds * payload / (IB_GB_S * 1e9)

    # 专家并行：每个 token 的 hidden 要发给 top_k 个专家
    tokens = batch * seq
    ep_payload = tokens * MOE_TOP_K * spec.hidden_size * ACTIVATION_BYTES
    ep_rounds = spec.num_layers
    ep_total = ep_rounds * ep_payload
    ep_node = ep_rounds * collective_times(ep_payload, gpus, NVLINK_GB_S)["all-to-all"]
    ep_cross = ep_rounds * collective_times(ep_payload, gpus, IB_GB_S)["all-to-all"]

    return [
        CommPlan(
            "数据并行 DP",
            dp_rounds,
            dp_total,
            dp_total,
            dp_node,
            dp_cross,
            "每步一次 all-reduce，可与反向计算重叠",
        ),
        CommPlan(
            "张量并行 TP",
            tp_rounds,
            payload,
            tp_total,
            tp_node,
            tp_cross,
            "每层 2 次 all-reduce，落在层间关键路径上，不能重叠",
        ),
        CommPlan(
            "流水线并行 PP",
            pp_rounds,
            payload,
            pp_total,
            pp_node,
            pp_cross,
            "通信最省，真正的代价是流水线气泡",
        ),
        CommPlan(
            "专家并行 EP",
            ep_rounds,
            ep_payload,
            ep_total,
            ep_node,
            ep_cross,
            "MoE 专属：all-to-all 的量随 token 数线性增长",
        ),
    ]


def zero_stage_memory(
    spec: ModelSpec,
    gpus: int,
    *,
    batch: int = 1,
    seq: int = TRAIN_SEQ,
    recompute: bool = True,
) -> list[dict[str, float]]:
    """算 ZeRO 各阶段下每张卡的显存账。

    拆分顺序：ZeRO-1 切优化器状态 → ZeRO-2 再切梯度 → ZeRO-3 连参数也切。
    激活显存不参与分片（它由 batch/seq 决定），单独加上去。
    """
    params_bytes = spec.params_b * 1e9 * PARAM_BYTES
    grad_bytes = spec.params_b * 1e9 * GRAD_BYTES
    optim_bytes = spec.params_b * 1e9 * ADAM_STATE_BYTES

    tensors = RECOMPUTE_TENSORS_PER_LAYER if recompute else SAVED_TENSORS_PER_LAYER
    activation_bytes = spec.num_layers * tensors * batch * seq * spec.hidden_size * ACTIVATION_BYTES

    stages = {
        "无 ZeRO": (params_bytes, grad_bytes, optim_bytes),
        "ZeRO-1": (params_bytes, grad_bytes, optim_bytes / gpus),
        "ZeRO-2": (params_bytes, grad_bytes / gpus, optim_bytes / gpus),
        "ZeRO-3": (params_bytes / gpus, grad_bytes / gpus, optim_bytes / gpus),
    }
    rows: list[dict[str, float]] = []
    for name, (params, grads, optim) in stages.items():
        state = params + grads + optim
        total = state + activation_bytes
        rows.append(
            {
                "stage": name,
                "params_gib": params / BYTES_PER_GIB,
                "grads_gib": grads / BYTES_PER_GIB,
                "optim_gib": optim / BYTES_PER_GIB,
                "state_gib": state / BYTES_PER_GIB,
                "activation_gib": activation_bytes / BYTES_PER_GIB,
                "total_gib": total / BYTES_PER_GIB,
                "fits": total / BYTES_PER_GIB <= DEVICE.memory_gib,
            }
        )
    return rows


def check_collective_primitives() -> None:
    """集合通信原语：同样的数据量，all-reduce 要花两趟。"""
    section("1. 集合通信原语的带宽模型")
    payload = 1.0 * BYTES_PER_GIB
    gpus = 64
    rows: list[list[str]] = []
    for name, seconds in collective_times(payload, gpus, NVLINK_GB_S).items():
        rows.append(
            [
                name,
                f"{payload / BYTES_PER_GIB:.0f}GiB",
                str(gpus),
                f"{seconds * 1000:.2f}ms",
                f"{payload / (seconds * 1e9):.0f}GB/s",
            ]
        )
    print_table(["原语", "数据量", "卡数", "耗时", "有效带宽"], rows)

    times = collective_times(payload, gpus, NVLINK_GB_S)
    ensure(
        approx(times["all-reduce"], 2.0 * times["all-gather"], 1e-9),
        f"all-reduce 必须是 all-gather 的两倍（两趟），实际 {times['all-reduce']:.6f} vs "
        f"{times['all-gather']:.6f}",
    )
    ensure(
        times["all-reduce"] < times["all-gather"] * 2.001,
        "all-reduce 的系数必须略小于 2（环上每卡只负责 (N-1)/N）",
    )
    ring = ring_all_reduce_time(payload, gpus, NVLINK_GB_S)
    ensure(
        approx(ring, 2.0 * (gpus - 1) / gpus * payload / (NVLINK_GB_S * 1e9), 1e-9),
        "ring all-reduce 必须等于 2×(N-1)/N× 数据量 ÷ 带宽",
    )
    rule(
        f"64 卡、1GiB 数据、NVLink：all-reduce {times['all-reduce'] * 1000:.2f}ms，"
        f"all-gather {times['all-gather'] * 1000:.2f}ms——"
        "同样的字节数，all-reduce 贵在要多跑一趟"
    )


def check_parallel_comm() -> None:
    """四种并行策略的通信账：TP 的通信量随层数线性增长。"""
    section("2. 四种并行策略的通信量与时间")
    plans = parallel_comm_plans(TARGET)
    rows: list[list[str]] = []
    for plan in plans:
        rows.append(
            [
                plan.name,
                str(plan.rounds),
                f"{plan.payload_bytes / BYTES_PER_GIB:.3f}GiB",
                f"{plan.total_bytes / BYTES_PER_GIB:.1f}GiB",
                f"{plan.node_time_s * 1000:.1f}ms",
                f"{plan.cross_node_time_s * 1000:.1f}ms",
                plan.bottleneck,
            ]
        )
    print_table(
        ["策略", "通信次数/步", "单次数据量", "每步总量", "节点内(NVLink)", "跨节点(IB)", "真正的瓶颈"],
        rows,
    )

    tp = next(p for p in plans if p.name.startswith("张量并行"))
    dp = next(p for p in plans if p.name.startswith("数据并行"))

    # 核心断言一：TP 通信量随层数线性增长
    doubled = replace(TARGET, num_layers=TARGET.num_layers * 2)
    tp_double = next(
        p for p in parallel_comm_plans(doubled) if p.name.startswith("张量并行")
    )
    ensure(
        approx(tp_double.total_bytes, tp.total_bytes * 2, 1e-9),
        f"层数翻倍时 TP 通信量必须精确翻倍，实际 {tp_double.total_bytes / BYTES_PER_GIB:.1f}GiB "
        f"vs {tp.total_bytes / BYTES_PER_GIB:.1f}GiB",
    )
    dp_double = next(
        p for p in parallel_comm_plans(doubled) if p.name.startswith("数据并行")
    )
    ensure(
        approx(dp_double.total_bytes, dp.total_bytes, 1e-9),
        "DP 的通信量与层数无关（梯度大小不随深度变）",
    )

    # 核心断言二：跨节点做 TP，通信时间会吃掉大部分训练步
    per_layer_compute = training_step_flops(TARGET, TRAIN_BATCH * TRAIN_SEQ) / TARGET.num_layers
    per_layer_compute /= TRAIN_GPUS
    compute_s = per_layer_compute / (DEVICE.fp16_tflops * 1e12 * 0.5)
    node_ratio = (tp.node_time_s / TARGET.num_layers) / compute_s
    cross_ratio = (tp.cross_node_time_s / TARGET.num_layers) / compute_s
    ensure(
        node_ratio < 0.05,
        f"节点内 TP 每层通信应只占每层计算的一小部分，实际 {node_ratio * 100:.2f}%",
    )
    ensure(
        cross_ratio > node_ratio * 5,
        f"跨节点 TP 的通信占比应比节点内高一个数量级，实际 {cross_ratio * 100:.1f}% vs "
        f"{node_ratio * 100:.2f}%",
    )
    ensure(
        cross_ratio > 0.2,
        f"跨节点做 TP 时，每层通信必须明显吃紧，实际占比 {cross_ratio * 100:.1f}%",
    )
    rule(
        f"TP 每层 2 次 all-reduce，通信量 ∝ 层数：{TARGET.num_layers} 层时每步 "
        f"{tp.total_bytes / BYTES_PER_GIB:.1f}GiB。节点内占每层计算 {node_ratio * 100:.2f}%，"
        f"跨节点变成 {cross_ratio * 100:.1f}%——这就是 TP 只能留在 NVLink 域内的原因"
    )

    # 流水线的气泡才是真瓶颈
    for microbatches in (8, 16, 32, 64):
        bubble = (TRAIN_GPUS - 1) / (microbatches + TRAIN_GPUS - 1)
        rule(f"PP 用 {microbatches} 个 microbatch 时气泡率 {bubble * 100:.1f}%")
        if microbatches == 32:
            ensure(
                bubble > 0.1 and bubble < 0.3,
                f"32 个 microbatch 的气泡率应在 10%~30% 之间，实际 {bubble * 100:.1f}%",
            )


def check_zero_stages() -> None:
    """ZeRO 1/2/3：把 70B 塞进 64 张 80GB 卡。"""
    section("3. ZeRO 1/2/3 的显存拆分")
    gpus = 64
    rows = zero_stage_memory(TARGET, gpus)
    print_table(
        ["方案", "每卡参数", "每卡梯度", "每卡优化器", "状态合计", "激活", "总计", "装得下"],
        [
            [
                f"{row['stage']}",
                fmt_gib(row["params_gib"]),
                fmt_gib(row["grads_gib"]),
                fmt_gib(row["optim_gib"]),
                fmt_gib(row["state_gib"]),
                fmt_gib(row["activation_gib"]),
                fmt_gib(row["total_gib"]),
                "是" if row["fits"] else "**否**",
            ]
            for row in rows
        ],
    )
    rule(
        f"（口径：{TARGET.name} / {gpus} 张 {DEVICE.name} / "
        f"bf16 参数与梯度 + Adam 状态 12 字节/参数 + 激活重计算后的 {TRAIN_SEQ} 序列）"
    )

    by_stage = {row["stage"]: row for row in rows}
    ensure(
        by_stage["ZeRO-3"]["total_gib"] < by_stage["ZeRO-2"]["total_gib"]
        < by_stage["ZeRO-1"]["total_gib"]
        < by_stage["无 ZeRO"]["total_gib"],
        "ZeRO 越深，每卡显存必须越少",
    )
    ensure(
        by_stage["ZeRO-3"]["fits"],
        f"70B 在 {gpus} 张 80GB 卡上必须能靠 ZeRO-3 装下，"
        f"实际 {by_stage['ZeRO-3']['total_gib']:.1f}GiB",
    )
    ensure(
        not by_stage["ZeRO-2"]["fits"],
        "ZeRO-2 只切梯度和优化器状态，权重仍是一整份，70B 装不下（这是必须用 ZeRO-3 的原因）",
    )
    ensure(
        not by_stage["ZeRO-1"]["fits"],
        "ZeRO-1 连梯度都不切，更装不下",
    )
    ensure(
        by_stage["ZeRO-1"]["optim_gib"] < by_stage["无 ZeRO"]["optim_gib"] / 10,
        "ZeRO-1 必须把优化器状态切到 1/64",
    )
    rule(
        f"参数 {fmt_gib(by_stage['无 ZeRO']['params_gib'])} 就已经超过一张 "
        f"{DEVICE.memory_gib:.0f}GiB 卡——ZeRO-3 之所以必要，不是因为省优化器状态，"
        "而是因为**参数本身也必须切**"
    )


def check_mixed_precision_recompute() -> None:
    """混合精度与激活重计算：用算力换显存。"""
    section("4. 混合精度与激活重计算")
    tokens = TRAIN_BATCH * TRAIN_SEQ
    forward = prefill_flops(TARGET, tokens)
    train = training_step_flops(TARGET, tokens)
    with_recompute = 4.0 * forward
    overhead = (with_recompute - train) / train

    rows = zero_stage_memory(TARGET, 64, recompute=False)
    row_with = next(
        r for r in zero_stage_memory(TARGET, 64, recompute=True) if r["stage"] == "ZeRO-3"
    )
    row_without = next(r for r in rows if r["stage"] == "ZeRO-3")
    ratio = row_without["activation_gib"] / row_with["activation_gib"]

    print_table(
        ["项", "不做重计算", "做重计算"],
        [
            [
                "每卡激活显存（ZeRO-3）",
                fmt_gib(row_without["activation_gib"]),
                fmt_gib(row_with["activation_gib"]),
            ],
            ["每卡总计", fmt_gib(row_without["total_gib"]), fmt_gib(row_with["total_gib"])],
        ],
    )
    ensure(
        approx(ratio, SAVED_TENSORS_PER_LAYER, 1e-9),
        f"激活显存应降到 1/{SAVED_TENSORS_PER_LAYER}，实际 1/{ratio:.1f}",
    )
    ensure(
        approx(overhead, 1.0 / 3.0, 1e-9),
        f"整层重计算应多花 1/3 的训练算力，实际 {overhead * 100:.2f}%",
    )
    ensure(
        not row_without["fits"],
        f"不做激活重计算时，ZeRO-3 每卡仍要 {row_without['total_gib']:.1f}GiB，"
        f"已经超过 {DEVICE.memory_gib:.0f}GiB 的卡容量——说明重计算在这套配置下是必需的",
    )
    ensure(
        row_with["fits"],
        "70B 训练必须靠激活重计算才装得下 64 张 80GB 卡",
    )
    rule(
        f"激活显存 {fmt_gib(row_without['activation_gib'])} → {fmt_gib(row_with['activation_gib'])}"
        f"（{SAVED_TENSORS_PER_LAYER}×），代价是训练算力 +{overhead * 100:.0f}%——"
        "重计算多跑一次前向，而反向本来就是前向的 2 倍，所以多出来的是 1/3"
    )


def check_training_to_inference() -> None:
    """训练侧的产物怎么变成推理侧的资产。"""
    section("5. 从训练到推理：checkpoint、量化校准、draft 模型")

    # 5.1 checkpoint 格式决定推理侧的切分方式
    budget = plan_memory(TARGET, DEVICE, kv_tokens=0)
    ensure(
        not budget.fits(),
        f"70B 的 bf16 权重 {budget.weights_gib:.1f}GiB 装不进一张 {DEVICE.memory_gib:.0f}GiB 卡，"
        "推理侧必然要分片部署",
    )
    rule(
        f"checkpoint 的字节数直接决定推理侧的拓扑：70B bf16 权重 {budget.weights_gib:.1f}GiB，"
        "单卡放不下，推理侧必须用分片 safetensors 加载到多卡上"
    )

    # 5.2 量化校准：校准集的形状决定量化误差
    uniform = [math.sin(index * 0.37) * 0.5 for index in range(4096)]
    realistic = build_activation_samples(4096)
    rows: list[list[str]] = []
    for label, samples in (("均匀样本", uniform), ("真实激活（带离群点）", realistic)):
        for group_size in (64, 128, 256):
            result = quantize_values(samples, 4, group_size)
            rows.append(
                [
                    label,
                    str(group_size),
                    f"{result.snr_db:.1f}dB",
                    f"{result.relative_error * 100:.3f}%",
                ]
            )
    print_table(["校准集", "分组大小", "SNR", "相对误差"], rows)

    uniform_snr = quantize_values(uniform, 4, 128).snr_db
    realistic_snr = quantize_values(realistic, 4, 128).snr_db
    ensure(
        realistic_snr < uniform_snr,
        f"真实激活上的量化 SNR 必须低于均匀样本，否则说明校准集没有代表性"
        f"（实际 {realistic_snr:.1f}dB vs {uniform_snr:.1f}dB）",
    )

    # 5.3 draft 模型：蒸馏质量直接决定线上收益
    target_cost = target_step_s()
    draft_rows: list[list[str]] = []
    for label, draft_spec in DRAFT_VARIANTS:
        cost = draft_round_cost_s(draft_spec)
        threshold = break_even_acceptance(
            draft_cost=cost, target_cost=target_cost, draft_length=4
        )
        draft_rows.append(
            [
                label,
                f"{weight_gib(draft_spec):.2f}GiB",
                f"{cost / target_cost * 100:.1f}%",
                f"{threshold * 100:.2f}%",
            ]
        )
    print_table(["draft 规模", "权重", "每轮代价占 target", "盈亏平衡接受率"], draft_rows)

    thresholds = [
        break_even_acceptance(
            draft_cost=draft_round_cost_s(spec), target_cost=target_cost, draft_length=4
        )
        for _, spec in DRAFT_VARIANTS
    ]
    ensure(
        thresholds[0] < thresholds[1],
        f"draft 越大盈亏平衡接受率越高，实测 {thresholds}",
    )
    rule(
        "draft 模型是训练侧蒸馏出来的：它的参数量决定每轮代价，"
        "蒸馏质量决定接受率，接受率决定线上能拿到多少加速比——"
        "两个变量都出自训练侧"
    )


def run() -> None:
    """跑完第 17 章的全部检查。"""
    banner("第 17 章 · 训练侧 Infra 地基")
    check_collective_primitives()
    check_parallel_comm()
    check_zero_stages()
    check_mixed_precision_recompute()
    check_training_to_inference()
    print()
    print("=" * 72)
    print("第 17 章结论：张量并行的通信量随层数线性增长且落在层间关键路径上，")
    print("跨节点做 TP 会把每层计算的很大一部分变成通信——TP 只能留在 NVLink 域内；")
    print("70B 在 64 张 80GB 卡上只有 ZeRO-3 装得下；")
    print("推理侧几乎所有更聪明的模型（LoRA、蒸馏、量化校准、draft）都出自训练侧。")
