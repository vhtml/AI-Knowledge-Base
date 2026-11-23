"""第 2 章实验：GPU 规格与显存账本。

这个模块解决什么问题："这张卡能不能跑这个模型"——这个问题问得太粗，
真正要回答的是"**四块显存分别占多少、哪一块是瓶颈、并发能开多少**"。
本章把显存账本变成可执行的断言：

1. **显存是四块，不是一块**——权重、KV cache、激活、运行时开销；
   低并发时"激活 + 运行时"这类固定块占比反而更大，高并发时 KV 变成大头；
2. **容量 ≠ 显存总量**——同一张卡上，权重是死重，KV 是弹性块；
   能不能多开并发只看"扣掉权重后还剩多少给 KV"；
3. **量化后权重装得下，不等于服务起得来**——KV 才是长上下文场景的真正瓶颈；
4. **买错卡的代价是可量化的**——同一模型换卡，并发序列数的差别
   全部来自 KV 空间，换算成"每个并发序列每小时多少钱"就能直接比价。

本章所有显存数字都走 :mod:`infra_lab.labkit` 的 :func:`~infra_lab.labkit.plan_memory`
与 :func:`~infra_lab.labkit.kv_capacity_tokens`，权重/KV 字节数一律来自
:mod:`infra_lab.labkit`，不另立公式。全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labkit import (
    BYTES_PER_GIB,
    GPUSpec,
    ModelSpec,
    approx,
    banner,
    ensure,
    fmt_gib,
    gpu,
    kv_capacity_tokens,
    model,
    plan_memory,
    print_table,
    rule,
    section,
)

RUNTIME_GIB = 2.0
"""运行时开销（GiB/进程）：CUDA context、框架初始化、通信缓冲。

这是**固定块**，与并发数无关、同型号的卡上取值一样——所以卡越小，它占比越高。
真实值要在自己的镜像上实测（`nvidia-smi` 看进程占用）。
"""

ACTIVATION_TENSORS = 4
"""同时活着的中间张量个数：输入 hidden、注意力输出、MLP 上投影、MLP 下投影。

推理侧的激活峰值由 **prefill** 决定（decode 一次只算 1 个 token，激活可忽略）。
用 FlashAttention 时注意力的 O(ctx²) 中间矩阵不会被物化，所以不计入。
"""

ACTIVATION_TOKENS = 2048
"""同时在算的 token 数（prefill 分块上限，第 7 章的 chunked prefill）。

激活占用按这个数预留，**与序列能有多长无关**——序列长度变长只会撑大 KV 块。
"""

SEQ_TOKENS = 2048
"""折算并发序列数时，假设每条序列占用 2048 token 的 KV（prompt + 预留输出）。"""

LEDGER_CONTEXT = 4096
"""四块账演示用的单序列上下文。长上下文服务里 KV 才会显形，2048 短上下文看不出来。"""

PLAN_BATCH = 32
"""规划用的最大并发 batch。激活块按这个上限预留——这是引擎的配置值，不是实时用量。"""

GRID_DEVICES = ("h100_sxm", "a100_80g", "l40s", "l4", "mi300x")
"""网格里的卡：80GB（SXM）、80GB（PCIe 代）、48GB、24GB、192GB。"""

GRID_MODELS = ("qwen2.5-7b", "llama2-13b", "qwen2.5-32b", "llama3.1-70b")
"""网格里的模型：7B / 13B / 32B / 70B，覆盖 MHA 到 GQA 的各种 KV 规格。"""


def activation_gib(spec: ModelSpec, batch: int, tokens_in_flight: int = ACTIVATION_TOKENS) -> float:
    """推理侧的激活占用（GiB）。

    口径（本章的**输入假设**，不是 labkit 的口径，真实值必须实测）：
    激活 ≈ batch × 在飞 token 数 × hidden_size × 2 字节 × 同时活着的张量个数。

    之所以按"batch × 在飞 token 数"算：prefill 阶段这些 token 的中间结果同时活着，
    这是推理侧激活峰值的来源。decode 阶段每步只算 1 个 token，激活可以忽略。

    **这一项必须显式预留**：引擎按配置的最大 batch 预留激活池，
    实际 batch 再小，这块显存也占着——低负载时段它是纯占位。
    """
    total_bytes = batch * tokens_in_flight * spec.hidden_size * 2 * ACTIVATION_TENSORS
    return total_bytes / BYTES_PER_GIB


@dataclass(frozen=True)
class GridRow:
    """一张卡 + 一个模型的显存账本结果。"""

    device: GPUSpec
    spec: ModelSpec
    weight_bits: int
    weights_gib: float
    activation_gib: float
    runtime_gib: float
    kv_headroom_gib: float
    """扣掉权重、激活、运行时之后，还能给 KV 的空间。"""
    max_kv_tokens: int
    """这块 KV 空间能装多少 token。"""
    max_sequences: int
    """按每序列 2048 token 折算的并发序列数。"""
    fits: bool
    """权重 + 固定开销是否装得下。"""


def ledger(
    device: GPUSpec,
    spec: ModelSpec,
    *,
    batch: int = PLAN_BATCH,
    context_tokens: int = SEQ_TOKENS,
    weight_bits: int = 16,
    runtime_gib: float = RUNTIME_GIB,
) -> GridRow:
    """算一张卡上的显存账本（权重 / 激活 / 运行时 / KV 余量）。

    KV 余量由 :func:`infra_lab.labkit.plan_memory` 的 ``free_gib`` 给出，
    再用 :func:`infra_lab.labkit.kv_capacity_tokens` 折算成 token 数，
    全程不自己算字节。
    """
    act = activation_gib(spec, batch)
    base = plan_memory(
        spec,
        device,
        kv_tokens=0,
        weight_bits=weight_bits,
        activation_gib=act,
        runtime_gib=runtime_gib,
    )
    if not base.fits():
        return GridRow(
            device=device,
            spec=spec,
            weight_bits=weight_bits,
            weights_gib=base.weights_gib,
            activation_gib=act,
            runtime_gib=runtime_gib,
            kv_headroom_gib=0.0,
            max_kv_tokens=0,
            max_sequences=0,
            fits=False,
        )
    headroom_bytes = base.free_gib * BYTES_PER_GIB
    max_tokens = kv_capacity_tokens(spec, headroom_bytes)
    return GridRow(
        device=device,
        spec=spec,
        weight_bits=weight_bits,
        weights_gib=base.weights_gib,
        activation_gib=act,
        runtime_gib=runtime_gib,
        kv_headroom_gib=base.free_gib,
        max_kv_tokens=max_tokens,
        max_sequences=max_tokens // max(1, context_tokens),
        fits=True,
    )


def check_four_blocks() -> None:
    """判断：显存是四块，低并发时固定块占比反而更大。"""
    section("判断一：显存四块账——权重 / KV / 激活 / 运行时")
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    rows: list[list[object]] = []
    fixed_shares: list[float] = []
    kv_shares: list[float] = []
    for batch in (1, 8, 32):
        act = activation_gib(spec, PLAN_BATCH, LEDGER_CONTEXT)
        budget = plan_memory(
            spec,
            device,
            kv_tokens=batch * LEDGER_CONTEXT,
            weight_bits=16,
            activation_gib=act,
            runtime_gib=RUNTIME_GIB,
        )
        fixed = (budget.activations_gib + budget.runtime_gib) / budget.used_gib
        fixed_shares.append(fixed)
        kv_shares.append(budget.kv_share)
        rows.append(
            [
                f"batch={batch}",
                fmt_gib(budget.weights_gib),
                fmt_gib(budget.kv_cache_gib),
                fmt_gib(budget.activations_gib),
                fmt_gib(budget.runtime_gib),
                fmt_gib(budget.used_gib),
                f"{budget.weights_gib / budget.used_gib * 100:.1f}%",
                f"{budget.kv_share * 100:.1f}%",
                f"{fixed * 100:.1f}%",
            ]
        )
    print_table(
        [
            "并发",
            "权重",
            "KV",
            "激活",
            "运行时",
            "合计",
            "权重占比",
            "KV 占比",
            "固定块占比",
        ],
        rows,
    )
    rule(
        f"激活按配置上限（batch={PLAN_BATCH}、{LEDGER_CONTEXT} token）预留 = "
        f"{fmt_gib(activation_gib(spec, PLAN_BATCH, LEDGER_CONTEXT))}，"
        "与实际并发无关——实际 batch 小时它是纯占位"
    )

    ensure(
        fixed_shares[0] > kv_shares[0],
        f"batch=1 时固定块占比 {fixed_shares[0] * 100:.1f}% 应当高于 KV 占比 "
        f"{kv_shares[0] * 100:.1f}%——并发低时 KV 几乎不占地方，固定块才是主角",
    )
    ensure(
        kv_shares[-1] > fixed_shares[-1],
        f"batch={PLAN_BATCH} 时 KV 占比 {kv_shares[-1] * 100:.1f}% 应当高于固定块占比 "
        f"{fixed_shares[-1] * 100:.1f}%——并发上去后 KV 变成大头",
    )
    ensure(
        fixed_shares[0] > fixed_shares[-1],
        "固定块占比应当随并发下降——并发的边际成本全部落在 KV 上",
    )

    # 同一份账放到 24GB 卡上，固定块占用的**容量比例**会更刺眼
    small = gpu("l4")
    act_gib = activation_gib(spec, PLAN_BATCH, LEDGER_CONTEXT)
    small_budget = plan_memory(
        spec,
        small,
        kv_tokens=LEDGER_CONTEXT,
        weight_bits=16,
        activation_gib=act_gib,
        runtime_gib=RUNTIME_GIB,
    )
    big_budget = plan_memory(
        spec,
        device,
        kv_tokens=LEDGER_CONTEXT,
        weight_bits=16,
        activation_gib=act_gib,
        runtime_gib=RUNTIME_GIB,
    )
    small_fixed = (
        small_budget.activations_gib + small_budget.runtime_gib
    ) / small.memory_gib
    big_fixed = (
        big_budget.activations_gib + big_budget.runtime_gib
    ) / device.memory_gib
    ensure(
        small_fixed > big_fixed * 2,
        f"24GB 卡上固定块占了容量的 {small_fixed * 100:.1f}%，应当是 80GB 卡 "
        f"{big_fixed * 100:.1f}% 的两倍以上——固定开销在越小的卡上越刺眼",
    )
    ensure(
        small_budget.used_gib <= small.memory_gib,
        "7B + 4096 上下文的单序列账在 24GB 卡上应当装得下",
    )
    rule("权重是死重、KV 是弹性块、固定块是税：容量规划必须扣掉后两者再谈并发")


def check_grid() -> None:
    """判断：跨「卡 × 模型」网格算账，找出每种组合的瓶颈。"""
    section("判断二：GPU × 模型网格——谁装得下、谁跑得动")
    rows: list[list[object]] = []
    for device_key in GRID_DEVICES:
        device = gpu(device_key)
        for model_key in GRID_MODELS:
            spec = model(model_key)
            row = ledger(device, spec)
            rows.append(
                [
                    device.name,
                    spec.name,
                    fmt_gib(row.weights_gib),
                    fmt_gib(row.kv_headroom_gib),
                    f"{row.max_kv_tokens:,}",
                    row.max_sequences if row.fits else "装不下",
                    "权重" if not row.fits else ("权重" if row.weights_gib / device.memory_gib > 0.6 else "KV"),
                ]
            )
    print_table(
        ["卡", "模型", "权重", "KV 余量", "可装 KV token", "并发序列(2K/条)", "瓶颈"],
        rows,
    )

    h100 = gpu("h100_sxm")
    l4 = gpu("l4")
    seven = model("qwen2.5-7b")
    seventy = model("llama3.1-70b")

    small = ledger(l4, seven)
    big = ledger(h100, seven)
    ensure(
        small.fits and big.fits,
        "7B 在 24GB 和 80GB 卡上都应当装得下",
    )
    ensure(
        big.max_sequences > small.max_sequences * 5,
        f"7B 在 80GB 卡上的并发序列数 {big.max_sequences} 应当是 24GB 卡 "
        f"{small.max_sequences} 的五倍以上",
    )
    ensure(
        approx(
            big.max_sequences / small.max_sequences,
            big.kv_headroom_gib / small.kv_headroom_gib,
            0.05,
        ),
        "同一模型换卡，并发序列数的比值应当等于 KV 空间的比值——"
        "因为权重和固定开销没变，差别全部来自 KV 空间",
    )

    heavy = ledger(h100, seventy)
    ensure(
        not heavy.fits,
        f"70B fp16 的权重是 {heavy.weights_gib:.1f}GiB，应当装不进 80GB 卡",
    )
    heavy_int8 = ledger(h100, seventy, weight_bits=8)
    ensure(
        heavy_int8.fits,
        f"70B int8 的权重是 {heavy_int8.weights_gib:.1f}GiB，应当装得进 80GB 卡",
    )
    ensure(
        heavy_int8.weights_gib / h100.memory_gib > 0.8,
        f"70B int8 的权重占了 80GB 卡的 "
        f"{heavy_int8.weights_gib / h100.memory_gib * 100:.0f}%，"
        "说明这张卡的瓶颈是权重，不是 KV",
    )
    ensure(
        heavy_int8.kv_headroom_gib < heavy_int8.weights_gib / 2,
        f"70B int8 只剩 {heavy_int8.kv_headroom_gib:.1f}GiB 给 KV，"
        "不到权重的一半——权重是瓶颈的直接证据",
    )
    rule(
        "24GB 卡跑 7B，瓶颈是 KV 空间；80GB 卡跑 70B，瓶颈是权重——"
        "同一张卡、同一套账，瓶颈位置完全不同"
    )


def check_quantization_trap() -> None:
    """判断：量化后权重装得下，KV 未必装得下。"""
    section("判断三：常见陷阱——权重装得下 ≠ 服务起得来")
    spec = model("llama3.1-70b")
    l40s = gpu("l40s")

    rows: list[list[object]] = []
    for context in (2048, 8192, 32768, 65536):
        row = ledger(l40s, spec, context_tokens=context, weight_bits=4)
        rows.append(
            [
                f"{context // 1024}K",
                fmt_gib(row.weights_gib),
                fmt_gib(row.kv_headroom_gib),
                f"{row.max_kv_tokens:,}",
                row.max_sequences,
                "起得来" if row.max_sequences >= 1 else "起不来",
            ]
        )
    print_table(
        ["单序列上下文", "权重(int4)", "KV 余量", "可装 KV token", "并发序列", "结论"],
        rows,
    )

    base = ledger(l40s, spec, context_tokens=65536, weight_bits=4)
    ensure(
        base.fits,
        f"70B int4 的权重 {base.weights_gib:.1f}GiB 应当装得进 48GB 卡",
    )
    ensure(
        base.max_sequences == 0,
        f"但 64K 上下文下 KV 余量 {base.kv_headroom_gib:.1f}GiB 连一条序列都装不下"
        f"（需要 {65536 * 327680 / BYTES_PER_GIB:.1f}GiB），服务根本起不来",
    )

    # 把 KV 也量化到 1 字节，容量确实翻倍——**但翻一倍仍然不够**
    kv_budget = base.kv_headroom_gib * BYTES_PER_GIB
    tokens_fp16 = kv_capacity_tokens(spec, kv_budget, dtype_bytes=2)
    tokens_fp8 = kv_capacity_tokens(spec, kv_budget, dtype_bytes=1)
    ensure(
        tokens_fp8 > tokens_fp16 * 1.9,
        f"KV 量化到 FP8 后可装 token 数 {tokens_fp8:,} 应当是 FP16 的 "
        f"{tokens_fp16:,} 的两倍左右",
    )
    ensure(
        tokens_fp8 < 65536,
        f"KV 量化到 FP8 后可装 {tokens_fp8:,} token，仍然装不下一条 64K 序列"
        "（需要 65,536）——量化能把容量翻倍，但翻倍不等于够用",
    )
    rule(
        "量化不是万能钥匙：70B int4 跑在 48GB 卡上，KV 量化到 FP8 把容量翻了一倍，"
        "依然差 9% 装不下 64K 上下文。真正的解法是换更大的卡、切分模型，"
        "或者限制 max context——不是继续量化"
    )

    # 单序列 KV 需求 vs 卡上余量：这一步的判据就是"装不装得下一条最长的序列"
    one_sequence_gib = 65536 * 327680 / BYTES_PER_GIB
    ensure(
        base.kv_headroom_gib < one_sequence_gib,
        f"单条 64K 序列要 {one_sequence_gib:.1f}GiB，卡上只剩 "
        f"{base.kv_headroom_gib:.1f}GiB——差 {one_sequence_gib - base.kv_headroom_gib:.1f}GiB",
    )
    rule(
        "容量规划要先问一句：最长的那一条请求装得下吗？"
        "权重装得下只是入场券，KV 才是长上下文的真正门槛"
    )


def check_buying_cost() -> None:
    """判断：买错卡的代价——把并发数换算成"每个并发序列每小时多少钱"。"""
    section("判断四：买错卡的代价")
    spec = model("qwen2.5-7b")

    rows: list[list[object]] = []
    for device_key in ("l4", "a10g", "l40s", "a100_80g", "h100_sxm"):
        device = gpu(device_key)
        row = ledger(device, spec)
        cost_per_seq_hour = (
            device.price_per_hour_usd / row.max_sequences if row.max_sequences > 0 else 0.0
        )
        rows.append(
            [
                device.name,
                f"{device.memory_gib:.0f}GiB",
                fmt_gib(row.weights_gib),
                fmt_gib(row.kv_headroom_gib),
                row.max_sequences,
                f"${device.price_per_hour_usd:.2f}",
                f"${cost_per_seq_hour:.4f}",
            ]
        )
    print_table(
        ["卡", "容量", "权重", "KV 余量", "并发序列", "卡价/小时", "每个并发序列每小时"],
        rows,
    )

    cheap = next(row for row in rows if row[0] == gpu("l4").name)
    rich = next(row for row in rows if row[0] == gpu("h100_sxm").name)
    ensure(
        rich[4] > cheap[4] * 5,
        f"80GB 卡的并发序列数 {rich[4]} 应当是 24GB 卡 {cheap[4]} 的五倍以上",
    )
    ensure(
        cheap[6] > rich[6],
        f"24GB 卡的每个并发序列每小时成本 {cheap[6]} 应当高于 80GB 卡的 {rich[6]}——"
        "卡便宜，但能同时服务的序列太少，单序列成本反而更高",
    )
    rule(
        "比价不能只看卡价：便宜的小卡在长上下文场景里，"
        "每个并发序列的单价反而更贵——这是买错卡最常见的亏法"
    )


def run() -> None:
    """跑第 2 章实验。"""
    banner("第 2 章 · GPU 规格与显存账本")
    check_four_blocks()
    check_grid()
    check_quantization_trap()
    check_buying_cost()
    rule(
        "显存是四块：权重是死重、KV 是弹性块、固定块是税；"
        "容量看 KV 余量，瓶颈在权重还是在 KV 要逐组合算；"
        "量化后权重装得下不等于服务起得来"
    )
    print("\n✓ 第 2 章通过")
