"""第 4 章实验：访存墙与批量摊薄。

这个模块解决什么问题："推理慢是因为模型太大、算力不够"——这个直觉在 decode 阶段
是错的。decode（逐 token 生成）每走一步都要把**整套权重**从 HBM 读一遍，
而此时算术强度低到离谱，GPU 的算力单元大部分时间在空转，瓶颈是**显存带宽**。

这一章把四个判断变成可执行的断言：

1. **batch=1 时算力大量闲置**——算术强度比 Roofline 拐点低两个数量级，
   工作点稳稳落在访存受限区；
2. **批量能摊薄权重读取**——权重读一次服务多条序列，每序列分摊到的权重成本
   随 batch 下降，吞吐近乎线性地涨；
3. **延迟在 batch 上单调上升**——这是访存瓶颈的必然结果，不是意外；
   真正的判据是**吞吐增益 ÷ 时延代价 < 1**；
4. **Roofline 的算力拐点对 decode 很远**——每 token 的计算量比每 token 要读的
   KV 字节小两个数量级，所以**计算项永远追不上访存项**，
   decode 的 batch 上限来自带宽，不来自算力。

本模块的"拐点"指的是**访存项内部成本构成的平衡点**（权重读取量 = KV 读取量），
不是延迟曲线的极值点——原因见 :func:`check_optimal_batch_has_a_knee`。

全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labkit import (
    GPUSpec,
    ModelSpec,
    approx,
    banner,
    decode_flops,
    decode_step_time,
    ensure,
    fmt_ms,
    gpu,
    model,
    print_table,
    roofline,
    rule,
    section,
)

DEVICE = "h100_sxm"
"""算账用的卡。H100 SXM 80GB：3350 GB/s 带宽、989 TFLOPS FP16。"""

MODEL = "qwen2.5-7b"
"""算账用的模型。7.6B 参数、28 层、28 个查询头 / 4 个 KV 头（GQA-7）。"""

CONTEXT = 4096
"""上下文长度（token）。解码阶段每一步都要读这么长的历史。"""

MFU = 0.5
"""可达带宽/算力占峰值的比例。工程上单卡批量推理常见 40%~70%。"""

BATCH_SCAN = tuple(range(1, 129))
"""batch 扫描范围：1→128。"""

WIDE_SCAN = tuple(range(1, 513))
"""加宽扫描：1→512，用来验证"访存项始终主导"在更大 batch 上依然成立。"""


@dataclass(frozen=True)
class SweepRow:
    """一次 batch 扫描的结果行。"""

    batch: int
    step_ms: float
    """一个 decode 步的耗时（毫秒）。批内所有序列并行，所以这也是每序列的出字间隔。"""

    memory_ms: float
    """访存项耗时：搬权重 + KV 要多久。"""

    compute_ms: float
    """计算项耗时：跑完这一步的 FLOPs 要多久。"""

    throughput: float
    """整机吞吐（token/s）。"""

    arithmetic_intensity: float
    """算术强度：这一步的 FLOPs ÷ 访存项搬运字节数。"""

    weight_share: float
    """权重读取占这一步访存总量的比例。摊薄程度看这一列。"""

    per_seq_weight_ms: float
    """每序列分摊到的权重读取耗时（毫秒）。这是"摊薄"最直接的刻度。"""

    bound: str
    """Roofline 判定：受限于算力还是带宽。"""


def decode_bytes(spec: ModelSpec, batch_size: int, context_tokens: int) -> float:
    """一个 decode 步的访存项字节数：权重（整批共享）+ KV（每序列一份）。

    这里直接用模型规格算字节，不再从时延反解——``decode_step_time`` 取的是
    访存项与计算项的**较大者**，反解会把计算时间误当成访存时间，
    在计算项占优的场景下整本账都会错。
    """
    return (
        spec.weights_bytes(spec.weight_bits)
        + batch_size * spec.kv_bytes_per_token(2) * context_tokens
    )


def decode_memory_time(
    device: GPUSpec,
    spec: ModelSpec,
    batch_size: int,
    context_tokens: int,
) -> float:
    """访存项耗时（秒）：把这些字节搬完要多久。"""
    effective_bw = device.hbm_bandwidth_gbs * 1e9 * MFU
    if effective_bw <= 0:
        return 0.0
    return decode_bytes(spec, batch_size, context_tokens) / effective_bw


def decode_compute_time(
    device: GPUSpec,
    spec: ModelSpec,
    batch_size: int,
    context_tokens: int,
) -> float:
    """计算项耗时（秒）：跑完这一步的 FLOPs 要多久。"""
    peak = device.peak_tflops("fp16") * 1e12 * MFU
    if peak <= 0:
        return 0.0
    return batch_size * decode_flops(spec, context_tokens) / peak


def arithmetic_intensity(
    spec: ModelSpec,
    batch_size: int,
    context_tokens: int,
) -> float:
    """一个 decode 步的算术强度（FLOPs / 访存字节）。"""
    return batch_size * decode_flops(spec, context_tokens) / decode_bytes(
        spec, batch_size, context_tokens
    )


def sweep(
    device: GPUSpec,
    spec: ModelSpec,
    context_tokens: int,
    batches: tuple[int, ...] = BATCH_SCAN,
) -> list[SweepRow]:
    """扫描 batch，返回每个 batch 的时延、吞吐、算术强度。"""
    rows: list[SweepRow] = []
    for batch in batches:
        step = decode_step_time(
            device,
            spec,
            batch,
            mfu=MFU,
            context_tokens=context_tokens,
        )
        memory = decode_memory_time(device, spec, batch, context_tokens)
        compute = decode_compute_time(device, spec, batch, context_tokens)
        # 本模块把时延拆成访存项与计算项，必须和 labkit 的口径逐位一致
        ensure(
            approx(step, max(memory, compute), 1e-9),
            f"batch={batch} 的时延应当等于 max(访存 {memory * 1000:.4f}ms, "
            f"计算 {compute * 1000:.4f}ms)，实际 {step * 1000:.4f}ms",
        )
        weight_bytes = spec.weights_bytes(spec.weight_bits)
        moved = decode_bytes(spec, batch, context_tokens)
        intensity = arithmetic_intensity(spec, batch, context_tokens)
        rows.append(
            SweepRow(
                batch=batch,
                step_ms=step * 1000,
                memory_ms=memory * 1000,
                compute_ms=compute * 1000,
                throughput=batch / step,
                arithmetic_intensity=intensity,
                weight_share=weight_bytes / moved,
                per_seq_weight_ms=(weight_bytes / moved) * step * 1000 / batch,
                bound=roofline(device, intensity).bound,
            )
        )
    return rows


def knee_batch(spec: ModelSpec, context_tokens: int) -> float:
    """拐点 batch：权重读取量恰好等于 KV 读取量的那一刻。

    **为什么用这个定义，而不是延迟曲线的极值点**：

    延迟曲线在访存瓶颈下是**单调上升**的，根本没有极值点可找。
    真正有意义的是**访存项内部成本构成的平衡点**——权重读取和 KV 读取
    谁更贵。小于它，新增一条序列几乎只花 KV 的钱，权重被无限摊薄；
    大于它，每条新序列的成本完全由 KV 决定，延迟跟着 batch 线性变差。

    所以"拐点"在这里的含义是：**权重摊薄的收益在哪里耗尽**。
    """
    weight_bytes = spec.weights_bytes(spec.weight_bits)
    kv_bytes_per_seq = spec.kv_bytes_per_token(2) * context_tokens
    return weight_bytes / kv_bytes_per_seq


def check_decode_is_memory_bound() -> None:
    """判断：decode 阶段缺的不是算力，是带宽。"""
    section("判断一：decode 是访存瓶颈，batch=1 时算力大量闲置")
    device = gpu(DEVICE)
    spec = model(MODEL)

    ridge = roofline(device, 1.0).ridge_intensity
    ai1 = arithmetic_intensity(spec, 1, CONTEXT)
    point1 = roofline(device, ai1)
    ratio = ai1 / ridge

    rule(
        f"Roofline 拐点算术强度 = {ridge:.1f} FLOP/byte"
        f"（{device.name}：{device.fp16_tflops:.0f} TFLOPS ÷ {device.hbm_bandwidth_gbs:.0f} GB/s）"
    )
    rule(
        f"batch=1 时算术强度只有 {ai1:.2f} FLOP/byte = 拐点的 {ratio * 100:.2f}%"
        f"，工作点判定：{point1.bound} bound"
    )

    ensure(
        point1.bound == "memory",
        f"batch=1 的 decode 步应当受限于带宽，实际判定为 {point1.bound} bound",
    )
    ensure(
        ratio < 0.01,
        f"batch=1 时算术强度应当低于拐点的 1%（算力严重闲置），"
        f"实际是拐点的 {ratio * 100:.2f}%",
    )

    rows = sweep(device, spec, CONTEXT, batches=(1, 2, 4, 8, 16, 32, 64, 128))
    print_table(
        ["batch", "单步时延", "访存项", "计算项", "整机吞吐 tok/s", "算术强度", "Roofline"],
        [
            [
                row.batch,
                fmt_ms(row.step_ms / 1000),
                fmt_ms(row.memory_ms / 1000),
                fmt_ms(row.compute_ms / 1000),
                f"{row.throughput:.0f}",
                f"{row.arithmetic_intensity:.2f}",
                row.bound,
            ]
            for row in rows
        ],
    )

    ensure(
        all(row.bound == "memory" for row in rows),
        "本场景下 decode 步在所有 batch 下都应受限于带宽——算力瓶颈是 prefill 的事",
    )
    ensure(
        rows[-1].arithmetic_intensity < ridge,
        f"即使 batch=128，算术强度 {rows[-1].arithmetic_intensity:.1f} 仍应低于拐点 {ridge:.1f}",
    )
    rule("批量能把算术强度往上推，但推不到拐点——decode 的 batch 上限来自带宽，不是算力")


def check_batch_amortizes_weights() -> None:
    """判断：权重读一次服务多条序列，批量收益来自权重摊薄。"""
    section("判断二：批量摊薄权重读取")
    device = gpu(DEVICE)
    spec = model(MODEL)

    rows = sweep(device, spec, CONTEXT, batches=(1, 8, 16, 32, 64, 128))
    print_table(
        ["batch", "单步时延", "整机吞吐 tok/s", "权重占访存量", "每序列分摊的权重读取"],
        [
            [
                row.batch,
                fmt_ms(row.step_ms / 1000),
                f"{row.throughput:.0f}",
                f"{row.weight_share * 100:.1f}%",
                f"{row.per_seq_weight_ms:.3f}ms",
            ]
            for row in rows
        ],
    )

    first, last = rows[0], rows[-1]
    ensure(
        last.weight_share < first.weight_share * 0.5,
        f"batch=128 的权重占比 {last.weight_share * 100:.1f}% 应当显著低于 "
        f"batch=1 的 {first.weight_share * 100:.1f}%，否则说明权重没有被摊薄",
    )

    mid = next(row for row in rows if row.batch == 16)
    ensure(
        mid.throughput > first.throughput * 10,
        f"batch=16 的吞吐 {mid.throughput:.0f} tok/s 应当比 batch=1 的 "
        f"{first.throughput:.0f} tok/s 高一个数量级（权重摊薄的直接收益）",
    )
    ensure(
        mid.step_ms < first.step_ms * 1.5,
        f"batch=16 的单步时延 {mid.step_ms:.2f}ms 不该超过 batch=1 的 "
        f"{first.step_ms * 1.5:.2f}ms——低 batch 区间时延几乎不涨",
    )
    rule(
        f"batch 从 1 提到 16：吞吐 ×{mid.throughput / first.throughput:.1f}，"
        f"单步时延只从 {first.step_ms:.2f}ms 涨到 {mid.step_ms:.2f}ms"
    )


def check_optimal_batch_has_a_knee() -> None:
    """判断：存在最优 batch，越过拐点后延迟的代价超过吞吐的收益。"""
    section("判断三：存在最优 batch，不是越大越好")
    device = gpu(DEVICE)
    spec = model(MODEL)

    rows = sweep(device, spec, CONTEXT)
    knee = knee_batch(spec, CONTEXT)
    weight_bytes = spec.weights_bytes(spec.weight_bits)

    # 延迟必须严格单调上升：访存项随 batch 增长、计算项随 batch 增长，
    # 两者都单调，所以 max() 也单调——这是访存瓶颈的必然结果，不是意外
    for prev, cur in zip(rows, rows[1:]):
        ensure(
            cur.step_ms > prev.step_ms,
            f"单步时延应当随 batch 单调上升，"
            f"但 batch={cur.batch} 的 {cur.step_ms:.4f}ms 不大于 "
            f"batch={prev.batch} 的 {prev.step_ms:.4f}ms",
        )

    at_knee = min(rows, key=lambda row: abs(row.batch - knee))
    kv_at_knee = weight_bytes * (at_knee.batch / knee)
    ensure(
        abs(kv_at_knee - weight_bytes) / weight_bytes < 0.02,
        f"batch={at_knee.batch}（拐点 {knee:.1f}）处 KV 读取量应当与权重读取量相等，"
        f"实际 KV={kv_at_knee / 1e9:.2f}GB vs 权重={weight_bytes / 1e9:.2f}GB",
    )

    below = at_knee
    above = next(row for row in rows if row.batch == 128)
    gain_ratio = above.throughput / below.throughput
    cost_ratio = above.step_ms / below.step_ms
    efficiency = gain_ratio / cost_ratio

    print_table(
        ["batch", "单步时延", "整机吞吐 tok/s", "算术强度", "权重占访存量"],
        [
            [
                row.batch,
                fmt_ms(row.step_ms / 1000),
                f"{row.throughput:.0f}",
                f"{row.arithmetic_intensity:.1f}",
                f"{row.weight_share * 100:.1f}%",
            ]
            for row in rows
            if row.batch in (1, 2, 4, 8, 16, 32, 48, 64, 65, 80, 96, 112, 128)
        ],
    )

    rule(
        f"拐点 batch = {knee:.1f}（权重读取量 = KV 读取量，访存项成本构成的平衡点），"
        f"最优 batch 建议取 {at_knee.batch}"
    )
    rule(
        f"batch {below.batch}→{above.batch}：吞吐 ×{gain_ratio:.3f}，"
        f"时延 ×{cost_ratio:.3f}，性价比（吞吐增益÷时延代价）= {efficiency:.3f}"
    )

    ensure(
        efficiency < 1.0,
        f"过了拐点后，batch {below.batch}→{above.batch} 的性价比应当跌破 1"
        f"（延迟代价大于吞吐收益），实际是 {efficiency:.3f}",
    )
    ensure(
        above.step_ms > at_knee.step_ms,
        f"batch={above.batch} 的时延 {above.step_ms:.2f}ms 应当高于拐点处的 "
        f"{at_knee.step_ms:.2f}ms——过了拐点延迟不降反升",
    )
    ensure(
        at_knee.batch < len(BATCH_SCAN),
        f"拐点 {at_knee.batch} 应当落在扫描范围内，否则这段实验说明不了问题",
    )
    rule(
        "延迟单调上升是访存瓶颈的必然结果；"
        "'batch 不是越大越好'的真正判据是吞吐增益 ÷ 时延代价 < 1"
    )


def check_compute_never_catches_memory() -> None:
    """判断：Roofline 的算力拐点对 decode 很远，瓶颈始终在带宽侧。"""
    section("判断四：计算项永远追不上访存项")
    device = gpu(DEVICE)
    spec = model(MODEL)

    ridge = roofline(device, 1.0).ridge_intensity
    flops_per_token = decode_flops(spec, CONTEXT)
    kv_bytes_per_token_step = spec.kv_bytes_per_token(2) * CONTEXT
    # 每 token 的算术强度上限：batch 再大也超不过这个数
    saturation = flops_per_token / kv_bytes_per_token_step

    rule(
        f"每 token 的 FLOPs = {flops_per_token / 1e9:.2f} GFLOP，"
        f"每 token 每步要读的 KV = {kv_bytes_per_token_step / 1e9:.4f} GB"
    )
    rule(
        f"算术强度上限 = {saturation:.1f} FLOP/byte，而 Roofline 拐点 = {ridge:.1f}"
        f"——上限比拐点低 {ridge / saturation:.1f} 倍"
    )

    ensure(
        saturation < ridge,
        f"decode 的算术强度上限 {saturation:.1f} 应当低于拐点 {ridge:.1f}，"
        "否则计算项会追上访存项",
    )

    # 交叉点解出来是负数 ⇒ 两项永不相交
    peak = device.peak_tflops("fp16") * 1e12 * MFU
    effective_bw = device.hbm_bandwidth_gbs * 1e9 * MFU
    denominator = flops_per_token * effective_bw - peak * kv_bytes_per_token_step
    ensure(
        denominator < 0,
        f"访存项与计算项的交叉方程分母应当为负（永不相交），实际 {denominator:.4e}",
    )

    rows = sweep(device, spec, CONTEXT, batches=WIDE_SCAN)
    worst = max(rows, key=lambda row: row.compute_ms / row.memory_ms)
    ensure(
        all(row.compute_ms < row.memory_ms for row in rows),
        f"batch 1→{len(WIDE_SCAN)} 的每一步都应当是访存项主导",
    )
    rule(
        f"batch=1→{len(WIDE_SCAN)} 全程访存项主导；"
        f"即使 batch={worst.batch}，计算项 {worst.compute_ms:.2f}ms 也只有访存项 "
        f"{worst.memory_ms:.2f}ms 的 {worst.compute_ms / worst.memory_ms * 100:.0f}%"
    )
    rule(
        "所以 decode 的 batch 上限来自带宽而非算力：加 batch 只会让访存项更长，"
        "永远等不到计算项来当瓶颈"
    )


def check_slo_bounds_batch() -> None:
    """判断：时延预算是 batch 的硬上限，往往比最优 batch 更早生效。"""
    section("判断五：时延预算给 batch 划了硬上限")
    device = gpu(DEVICE)
    spec = model(MODEL)

    budget_ms = 15.0
    rows = sweep(device, spec, CONTEXT)
    feasible = [row.batch for row in rows if row.step_ms <= budget_ms]
    max_batch = max(feasible)
    knee = knee_batch(spec, CONTEXT)
    at_max = next(row for row in rows if row.batch == max_batch)

    rule(
        f"给定 TPOT 预算 {budget_ms:.0f}ms，单步时延允许的最大 batch = {max_batch}"
        f"（拐点在 {knee:.1f}）"
    )
    rule(
        f"被时延预算卡住时，整机吞吐只有 {at_max.throughput:.0f} tok/s，"
        f"而拐点处能跑到 {at_max.throughput * (knee / max_batch):.0f} tok/s 量级"
    )

    ensure(
        max_batch < knee,
        f"时延预算卡住的 batch {max_batch} 应当小于拐点 {knee:.1f}——"
        "说明 SLO 往往比吞吐最优更早成为约束",
    )
    ensure(
        at_max.step_ms <= budget_ms,
        f"batch={max_batch} 的单步时延 {at_max.step_ms:.2f}ms 应当仍在预算内",
    )
    rule("先问 SLO 要多少 batch，再问显存装不装得下，最后才谈吞吐最优")


def run() -> None:
    """跑第 4 章实验。"""
    banner("第 4 章 · 访存墙与批量摊薄")
    check_decode_is_memory_bound()
    check_batch_amortizes_weights()
    check_optimal_batch_has_a_knee()
    check_compute_never_catches_memory()
    check_slo_bounds_batch()
    rule(
        "decode 的瓶颈是带宽不是算力；批量摊薄权重读取；"
        "延迟随 batch 单调上升，判据是吞吐增益÷时延代价<1；"
        "Roofline 的算力拐点对 decode 很远"
    )
    print("\n✓ 第 4 章通过")
