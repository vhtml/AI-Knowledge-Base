"""第 3 章实验：Roofline 与算术强度。

这个模块解决什么问题：判断"这张卡跑得快不快"之前，得先知道它**被什么卡住**。
买卡的直觉是看算力，但推理服务的 decode 阶段每生成一个 token 都要把整套权重
从 HBM 读一遍，算术强度低到两个数量级以下——GPU 的算力单元大部分时间在空转。

这一章把 Roofline 模型变成四个可执行的判断：

1. **拐点算术强度**决定一张卡在哪种工作点上算力瓶颈——H100 是
   989 TFLOPS ÷ 3350 GB/s ≈ 295 FLOP/byte，不同卡的拐点差近 3 倍；
2. **prefill 与 decode 站在 Roofline 图的两侧**——prefill 算术强度上千，
   受限于算力；decode 算术强度个位数，受限于带宽；
3. **decode 的每 token 成本下界只由带宽决定**——峰值算力再高也压不下去；
4. **选卡先看带宽**——RTX 4090 的算力只有 H100 的 17%，每百万 token 成本却更低。

所有算账走 :mod:`infra_lab.labkit`，不另立公式常量表。
全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..labkit import (
    GPUSpec,
    ModelSpec,
    approx,
    attention_flops,
    banner,
    decode_flops,
    decode_step_time,
    ensure,
    fmt_bytes,
    fmt_ms,
    gpu,
    kv_bytes_per_token,
    model,
    prefill_flops,
    prefill_time,
    print_table,
    roofline,
    rule,
    section,
)

DEVICE = "h100_sxm"
"""算账用的卡。H100 SXM 80GB：3350 GB/s 带宽、989 TFLOPS FP16。"""

MODEL = "qwen2.5-7b"
"""算账用的模型。7.62B 参数、28 层、28 个查询头 / 4 个 KV 头（GQA-7）。"""

CONTEXT = 4096
"""上下文长度（token）。decode 每一步都要读这么长的历史。"""

MFU = 0.5
"""可达比例：实际带宽/算力能到峰值的多少。工程上单卡批量推理常见 40%~70%。"""

GRID = (
    "h100_sxm",
    "h100_pcie",
    "a100_80g",
    "a100_40g",
    "l40s",
    "l4",
    "a10g",
    "rtx4090",
    "mi300x",
)
"""参与对比的卡：覆盖 24GB 消费级到 192GB 数据中心级。"""

PREFILL_TOKENS = 2048
"""prefill 演示用的 prompt 长度。"""

PREFILL_CROSSOVER_LIMIT = 4096
"""找 prefill 交叉点时的扫描上限。"""

DECODE_BATCHES = (1, 8, 32, 128)
"""decode 演示用的 batch 档位。"""

BIG_BATCH = 4096
"""求 KV 地板用的超大 batch：权重被摊薄到可以忽略，剩下的就是 KV 读取时间。"""


# ---------------------------------------------------------------- 算术强度


def fmt_flops(value: float) -> str:
    """把 FLOPs 数量格式化成可读字符串。

    labkit 的 :func:`~infra_lab.labkit.fmt_bytes` 只管字节数，
    FLOPs 属于"算力账"，量纲不同，不能套用。
    """
    if value >= 1e12:
        return f"{value / 1e12:.2f}TFLOP"
    if value >= 1e9:
        return f"{value / 1e9:.2f}GFLOP"
    return f"{value:.0f}FLOP"


def prefill_bytes(spec: ModelSpec) -> float:
    """prefill 阶段搬运的字节数：把整套权重读一遍。

    口径与 labkit 的 :func:`~infra_lab.labkit.prefill_time` 访存项一致——
    权重在所有 token 之间复用，所以只算一遍。注意力那部分 O(s²) 的中间矩阵
    在 FlashAttention 下不物化，不进这一项。
    """
    return float(spec.weights_bytes(spec.weight_bits))


def prefill_intensity(spec: ModelSpec, tokens: int) -> float:
    """prefill 的算术强度（FLOPs ÷ 搬运字节）。"""
    return prefill_flops(spec, tokens) / prefill_bytes(spec)


def decode_bytes(spec: ModelSpec, batch_size: int, context_tokens: int) -> float:
    """一个 decode 步搬运的字节数：权重（整批共享）+ KV（每序列一份）。"""
    return (
        float(spec.weights_bytes(spec.weight_bits))
        + batch_size * kv_bytes_per_token(spec, 2) * context_tokens
    )


def decode_intensity(spec: ModelSpec, batch_size: int, context_tokens: int) -> float:
    """一个 decode 步的算术强度（FLOPs ÷ 搬运字节）。"""
    return batch_size * decode_flops(spec, context_tokens) / decode_bytes(
        spec, batch_size, context_tokens
    )


def decode_attention_flops(spec: ModelSpec, context_tokens: int) -> float:
    """一个 decode 步里花在注意力上的 FLOPs。

    用 labkit 的 :func:`~infra_lab.labkit.attention_flops` 差分得到：
    ``attention_flops(spec, ctx, 1) - attention_flops(spec, ctx, 0)``
    正好是"在 ctx 长度的上下文上多走一步 decode"的注意力代价。
    这样不用自己另写注意力公式。
    """
    return attention_flops(spec, context_tokens, 1) - attention_flops(
        spec, context_tokens, 0
    )


def prefill_crossover_tokens(
    device: GPUSpec, spec: ModelSpec, limit: int = PREFILL_CROSSOVER_LIMIT
) -> int:
    """prefill 从访存受限翻到算力受限所需的最小 token 数。

    算术强度随 token 数单调上升，所以从 1 往上扫，
    第一个落在拐点之上的 token 数就是交叉点。
    """
    for tokens in range(1, limit + 1):
        if roofline(device, prefill_intensity(spec, tokens)).bound == "compute":
            return tokens
    return limit + 1


# ---------------------------------------------------------------- 结果行


@dataclass(frozen=True)
class RidgeRow:
    """一张卡的 Roofline 拐点。"""

    device: GPUSpec
    ridge: float
    """FP16 拐点算术强度（FLOP/byte）。"""
    fp8_ridge: float
    """FP8 拐点算术强度——峰值算力翻倍，拐点也跟着翻倍。"""
    bandwidth_per_dollar: float
    """每美元每小时能买到多少 GB/s 带宽：选卡先看带宽的排序依据。"""


@dataclass(frozen=True)
class CostRow:
    """一张卡在「每 token 成本」这条轴上的位置。"""

    device: GPUSpec
    weight_floor_s: float
    """权重地板：batch=1、满带宽时每 token 的最短耗时（秒）。"""
    kv_floor_s: float
    """KV 地板：batch 拉到极大时每 token 的最短耗时（秒），MFU 取现实值。"""
    tokens_per_s: float
    """权重地板对应的吞吐上界（token/s）。"""
    cost_per_million: float
    """每百万 token 的成本下界（美元，按小时租金折算）。"""
    bandwidth_per_dollar: float
    """每美元每小时能买到多少 GB/s 带宽。"""
    key: str
    """GPU 规格表里的键名，用来在排序结果里做短标签。"""


def ridge_rows(devices: tuple[str, ...] = GRID) -> list[RidgeRow]:
    """扫出每张卡的拐点算术强度。"""
    rows: list[RidgeRow] = []
    for name in devices:
        device = gpu(name)
        rows.append(
            RidgeRow(
                device=device,
                ridge=roofline(device, 1.0).ridge_intensity,
                fp8_ridge=roofline(device, 1.0, "fp8").ridge_intensity,
                bandwidth_per_dollar=device.hbm_bandwidth_gbs / device.price_per_hour_usd,
            )
        )
    return rows


def cost_rows(
    spec: ModelSpec, devices: tuple[str, ...] = GRID, mfu: float = MFU
) -> list[CostRow]:
    """扫出每张卡的每 token 成本下界与每百万 token 成本。"""
    rows: list[CostRow] = []
    for name in devices:
        device = gpu(name)
        # 权重地板：batch=1、满带宽、上下文取 1 token（只搬权重那一项的极限）
        weight_floor = decode_step_time(
            device, spec, 1, mfu=1.0, context_tokens=1
        )
        # KV 地板：batch 拉到极大，权重被摊薄到可忽略，剩下的就是 KV 读取
        kv_floor = decode_step_time(
            device, spec, BIG_BATCH, mfu=mfu, context_tokens=CONTEXT
        ) / BIG_BATCH
        tokens_per_s = 1.0 / weight_floor if weight_floor > 0 else 0.0
        hours = 1e6 / tokens_per_s / 3600.0 if tokens_per_s > 0 else 0.0
        rows.append(
            CostRow(
                device=device,
                weight_floor_s=weight_floor,
                kv_floor_s=kv_floor,
                tokens_per_s=tokens_per_s,
                cost_per_million=device.price_per_hour_usd * hours,
                bandwidth_per_dollar=device.hbm_bandwidth_gbs / device.price_per_hour_usd,
                key=name,
            )
        )
    return rows


# ---------------------------------------------------------------- 判断一


def check_ridge_point_across_gpus() -> None:
    """判断：拐点算术强度 = 峰值算力 ÷ 峰值带宽，不同卡差近 3 倍。"""
    section("判断一：拐点算术强度——每张卡的「分水岭」位置不同")
    rows = ridge_rows()

    print_table(
        ["卡", "带宽 GB/s", "FP16 峰值 TFLOPS", "FP16 拐点", "FP8 拐点", "带宽 GB/s per $/小时"],
        [
            [
                row.device.name,
                f"{row.device.hbm_bandwidth_gbs:.0f}",
                f"{row.device.fp16_tflops:.0f}",
                f"{row.ridge:.1f}",
                f"{row.fp8_ridge:.1f}",
                f"{row.bandwidth_per_dollar:.0f}",
            ]
            for row in rows
        ],
    )

    h100 = next(row for row in rows if row.device.name == "H100 SXM 80GB")
    rule(
        f"H100 SXM：{h100.device.fp16_tflops:.0f} TFLOPS ÷ "
        f"{h100.device.hbm_bandwidth_gbs:.0f} GB/s = {h100.ridge:.1f} FLOP/byte"
    )
    ensure(
        approx(h100.ridge, 295.2, 1e-3),
        f"H100 的拐点应当是 295.2 FLOP/byte（989 TFLOPS ÷ 3350 GB/s），"
        f"实际是 {h100.ridge:.4f}",
    )

    lowest = min(rows, key=lambda row: row.ridge)
    highest = max(rows, key=lambda row: row.ridge)
    rule(
        f"拐点最低的是 {lowest.device.name}（{lowest.ridge:.1f}），"
        f"最高的是 {highest.device.name}（{highest.ridge:.1f}），"
        f"差 {highest.ridge / lowest.ridge:.2f} 倍"
    )
    ensure(
        highest.ridge / lowest.ridge > 2.0,
        f"不同卡的拐点应当差出 2 倍以上，实际只差 "
        f"{highest.ridge / lowest.ridge:.2f} 倍",
    )

    # 拐点越高 = 越难走到算力受限区 = 越依赖带宽而不是算力
    l40s = next(row for row in rows if row.device.name == "L40S 48GB")
    ensure(
        l40s.ridge > h100.ridge,
        f"L40S 的拐点 {l40s.ridge:.1f} 应当高于 H100 的 {h100.ridge:.1f}——"
        "它的算力相对带宽偏强，同一个工作点在 L40S 上更容易访存受限",
    )

    # 量化把峰值算力抬高 → 拐点同步抬高 → 算力瓶颈被推得更远
    ensure(
        all(row.fp8_ridge >= row.ridge for row in rows),
        "FP8 拐点不应当低于 FP16 拐点——峰值算力只涨不跌，拐点只会更靠右",
    )
    ensure(
        approx(h100.fp8_ridge, h100.ridge * 2.0, 1e-3),
        f"H100 的 FP8 拐点应当约为 FP16 拐点的 2 倍（{h100.fp8_ridge:.1f} vs {h100.ridge:.1f}）",
    )
    consumer = next(row for row in rows if row.device.name == "RTX 4090 24GB")
    ensure(
        approx(consumer.fp8_ridge, consumer.ridge * 4.0, 1e-3),
        f"RTX 4090 的 FP8 峰值是 FP16 的 4 倍，拐点也应当是 4 倍"
        f"（{consumer.fp8_ridge:.1f} vs {consumer.ridge:.1f}）",
    )
    rule(
        "量化把峰值算力翻倍，拐点也翻倍——等于把算力瓶颈推得更远；"
        "decode 本来就在拐点之下，算力再多也用不上"
    )

    ensure(
        all(100.0 <= row.ridge <= 500.0 for row in rows),
        "本课涉及的所有卡的拐点都应当落在 100~500 FLOP/byte 这个量级",
    )
    rule(
        "拐点越高，同一个工作点越靠左（越访存受限）；"
        "所有卡的拐点都在三位数，而 decode 的算术强度只有个位数——"
        "差两个数量级，不是差一点"
    )


# ---------------------------------------------------------------- 判断二


def check_prefill_and_decode_live_on_different_sides() -> None:
    """判断：prefill 在拐点之上（算力瓶颈），decode 在拐点之下（带宽瓶颈）。"""
    section("判断二：prefill 和 decode 站在 Roofline 图的两侧")
    device = gpu(DEVICE)
    spec = model(MODEL)
    ridge = roofline(device, 1.0).ridge_intensity

    rows: list[list[str]] = []
    prefill_point = roofline(device, prefill_intensity(spec, PREFILL_TOKENS))
    rows.append(
        [
            f"prefill {PREFILL_TOKENS} tok",
            fmt_flops(prefill_flops(spec, PREFILL_TOKENS)),
            fmt_bytes(prefill_bytes(spec)),
            f"{prefill_point.arithmetic_intensity:.1f}",
            f"{ridge:.1f}",
            prefill_point.bound,
            f"{prefill_point.attainable_tflops / 1e12:.0f}",
            f"{prefill_point.attainable_tflops / prefill_point.peak_flops_per_s * 100:.1f}%",
        ]
    )
    for batch in DECODE_BATCHES:
        point = roofline(device, decode_intensity(spec, batch, CONTEXT))
        rows.append(
            [
                f"decode batch={batch}",
                fmt_flops(batch * decode_flops(spec, CONTEXT)),
                fmt_bytes(decode_bytes(spec, batch, CONTEXT)),
                f"{point.arithmetic_intensity:.1f}",
                f"{ridge:.1f}",
                point.bound,
                f"{point.attainable_tflops / 1e12:.1f}",
                f"{point.attainable_tflops / point.peak_flops_per_s * 100:.2f}%",
            ]
        )

    print_table(
        [
            "工作点",
            "FLOPs",
            "访存字节",
            "算术强度",
            "拐点",
            "判定",
            "可达算力 TFLOPS",
            "占峰值算力",
        ],
        rows,
    )

    ensure(
        prefill_point.bound == "compute",
        f"prefill {PREFILL_TOKENS} token 应当受限于算力，实际判定为 {prefill_point.bound}",
    )
    ensure(
        prefill_point.arithmetic_intensity > ridge,
        f"prefill 的算术强度 {prefill_point.arithmetic_intensity:.1f} 应当高于拐点 {ridge:.1f}",
    )

    decode_points = [
        roofline(device, decode_intensity(spec, batch, CONTEXT))
        for batch in DECODE_BATCHES
    ]
    ensure(
        all(point.bound == "memory" for point in decode_points),
        "所有 batch 下的 decode 都应当受限于带宽——算力瓶颈是 prefill 的事",
    )
    ensure(
        max(point.arithmetic_intensity for point in decode_points) < ridge,
        f"即使 batch={DECODE_BATCHES[-1]}，decode 的算术强度也应当低于拐点 {ridge:.1f}",
    )

    # 拐点之下的代价：decode 能用上的算力只有峰值的百分之零点几
    widest = max(decode_points, key=lambda point: point.arithmetic_intensity)
    rule(
        f"batch={DECODE_BATCHES[-1]} 时 decode 能用上峰值算力的 "
        f"{widest.attainable_tflops / widest.peak_flops_per_s * 100:.2f}%"
        f"（{widest.attainable_tflops / 1e12:.1f} / "
        f"{widest.peak_flops_per_s / 1e12:.0f} TFLOPS）——"
        "批量能把算力利用率抬起来，但抬不过拐点"
    )

    solo = decode_points[0]
    utilization = solo.attainable_tflops / solo.peak_flops_per_s
    rule(
        f"batch=1 时 decode 能用上峰值算力的 {utilization * 100:.2f}%"
        f"（{solo.attainable_tflops / 1e12:.2f} / "
        f"{solo.peak_flops_per_s / 1e12:.0f} TFLOPS）"
    )
    ensure(
        utilization < 0.01,
        f"batch=1 的 decode 算力利用率应当低于 1%，实际是 {utilization * 100:.2f}%",
    )

    # prefill 也有访存受限的小 prompt 区间
    crossover = prefill_crossover_tokens(device, spec)
    rule(
        f"H100 + {spec.name}：prefill 要到 {crossover} token 才越过拐点，"
        f"比拐点低的短 prompt 同样访存受限"
    )
    ensure(
        roofline(device, prefill_intensity(spec, crossover - 1)).bound == "memory",
        f"{crossover - 1} token 的 prefill 应当还在访存受限侧",
    )
    ensure(
        roofline(device, prefill_intensity(spec, crossover)).bound == "compute",
        f"{crossover} token 的 prefill 应当已经进入算力受限侧",
    )
    ensure(
        300 <= crossover <= 400,
        f"prefill 交叉点应当在 300~400 token 这个量级，实际是 {crossover}",
    )

    # 长上下文会把 decode 的算术强度抬起来，但离拐点还差两个数量级
    for ctx in (1024, 4096, 32768):
        ai = decode_intensity(spec, 1, ctx)
        attn = decode_attention_flops(spec, ctx)
        total = decode_flops(spec, ctx)
        rule(
            f"上下文 {ctx:>5}：decode 算术强度 {ai:6.2f}（拐点的 {ai / ridge * 100:.2f}%），"
            f"注意力占这一步算力 {attn / total * 100:5.1f}%"
        )
    ensure(
        decode_intensity(spec, 1, 32768) < ridge,
        "上下文拉到 32K，decode 的算术强度依然应当在拐点之下",
    )


# ---------------------------------------------------------------- 判断三


def check_decode_cost_floor_is_bandwidth() -> None:
    """判断：decode 的每 token 成本下界只由带宽决定，峰值算力无关。"""
    section("判断三：decode 的每 token 成本下界 = 字节数 ÷ 带宽")
    device = gpu(DEVICE)
    spec = model(MODEL)
    rows = cost_rows(spec)

    print_table(
        ["卡", "权重地板 ms/token", "KV 地板 ms/token", "吞吐上界 tok/s", "每小时 $", "每百万 token $"],
        [
            [
                row.device.name,
                fmt_ms(row.weight_floor_s),
                fmt_ms(row.kv_floor_s),
                f"{row.tokens_per_s:.1f}",
                f"{row.device.price_per_hour_usd:.2f}",
                f"{row.cost_per_million:.2f}",
            ]
            for row in rows
        ],
    )

    h100 = next(row for row in rows if row.device.name == "H100 SXM 80GB")
    weight_bytes = float(spec.weights_bytes(spec.weight_bits))
    # 上下文取 1 token 时这一步还要读一份 KV，但只有 1 个 token 的量，可以忽略
    solo_kv_bytes = kv_bytes_per_token(spec, 2)
    rule(
        f"权重地板 = {fmt_bytes(weight_bytes)} ÷ "
        f"{h100.device.hbm_bandwidth_gbs:.0f} GB/s = {fmt_ms(h100.weight_floor_s)}/token"
    )
    ensure(
        approx(
            h100.weight_floor_s,
            (weight_bytes + solo_kv_bytes) / (h100.device.hbm_bandwidth_gbs * 1e9),
            1e-9,
        ),
        "权重地板应当精确等于（权重字节数 + 1 token 的 KV）÷ 带宽",
    )
    ensure(
        solo_kv_bytes / weight_bytes < 0.01,
        "上下文取 1 token 时 KV 项应当可以忽略——这个地板就是纯权重地板",
    )
    ensure(
        approx(h100.weight_floor_s, 4.549e-3, 1e-2),
        f"H100 的权重地板应当是 4.549 ms/token，实际是 {h100.weight_floor_s * 1000:.4f} ms",
    )

    # 同一模型换卡：每 token 差距完全由带宽比解释
    pcie = next(row for row in rows if row.device.name == "H100 PCIe 80GB")
    bandwidth_ratio = (
        h100.device.hbm_bandwidth_gbs / pcie.device.hbm_bandwidth_gbs
    )
    flops_ratio = h100.device.fp16_tflops / pcie.device.fp16_tflops
    time_ratio = pcie.weight_floor_s / h100.weight_floor_s
    rule(
        f"H100 SXM vs PCIe：带宽比 {bandwidth_ratio:.3f}，算力比 {flops_ratio:.3f}，"
        f"每 token 时间比 {time_ratio:.3f}"
    )
    ensure(
        approx(time_ratio, bandwidth_ratio, 1e-3),
        f"每 token 时间比应当等于带宽比 {bandwidth_ratio:.3f}，实际是 {time_ratio:.3f}",
    )
    ensure(
        time_ratio > flops_ratio,
        f"每 token 的差距（{time_ratio:.3f}）应当比算力差距（{flops_ratio:.3f}）更大——"
        "说明差距由带宽而不是算力决定",
    )

    # 算力再高也压不下去：峰值算力估算 decode 吞吐会乐观三个数量级
    flops_per_token = decode_flops(spec, CONTEXT)
    naive_tokens_per_s = device.fp16_tflops * 1e12 / flops_per_token
    rule(
        f"若误按峰值算力估 decode 吞吐：{device.fp16_tflops:.0f} TFLOPS ÷ "
        f"{fmt_flops(flops_per_token)} = {naive_tokens_per_s:.0f} tok/s，"
        f"而带宽上界只有 {h100.tokens_per_s:.1f} tok/s"
    )
    ensure(
        naive_tokens_per_s > h100.tokens_per_s * 100,
        f"用峰值算力估 decode 吞吐应当乐观 100 倍以上，"
        f"实际乐观 {naive_tokens_per_s / h100.tokens_per_s:.0f} 倍",
    )

    # 第二个地板：batch 拉到极大时，权重摊薄到 0，只剩 KV 读取——照样是带宽
    big = decode_step_time(
        device, spec, BIG_BATCH, mfu=MFU, context_tokens=CONTEXT
    ) / BIG_BATCH
    bigger = decode_step_time(
        device, spec, BIG_BATCH * 2, mfu=MFU, context_tokens=CONTEXT
    ) / (BIG_BATCH * 2)
    rule(
        f"KV 地板（batch→∞，MFU {MFU}）：{big * 1000:.4f}ms/token，"
        f"batch 加倍后收敛到 {bigger * 1000:.4f}ms/token"
    )
    ensure(
        abs(bigger - big) / big < 0.01,
        f"batch 加倍后每 token 时间应当收敛（变化 < 1%），"
        f"实际从 {big * 1000:.4f}ms 变到 {bigger * 1000:.4f}ms",
    )
    ensure(
        bigger < h100.weight_floor_s,
        "KV 地板应当明显低于权重地板——批量摊薄能把每 token 成本压下去，"
        "但两个地板都由带宽决定",
    )
    rule(
        "两个地板（权重地板、KV 地板）都由带宽决定："
        "batch 摊薄能突破第一个地板，但突破不了「每 token 必须读自己的 KV」这条硬约束"
    )


# ---------------------------------------------------------------- 判断四


def check_peak_flops_overestimates_capacity() -> None:
    """判断：拿峰值当可达值，容量估算乐观一倍；decode 上更夸张。"""
    section("判断四：峰值 ≠ 可达，用峰值算容量会乐观")
    device = gpu(DEVICE)
    spec = model(MODEL)

    rows: list[list[str]] = []
    for mfu in (1.0, 0.7, 0.5, 0.4):
        step = decode_step_time(device, spec, 1, mfu=mfu, context_tokens=CONTEXT)
        prefill = prefill_time(device, spec, PREFILL_TOKENS, mfu=mfu)
        tokens_per_s = 1.0 / step
        cost = device.price_per_hour_usd * (1e6 / tokens_per_s) / 3600.0
        rows.append(
            [
                f"{mfu:.1f}",
                fmt_ms(step),
                f"{tokens_per_s:.1f}",
                fmt_ms(prefill),
                f"{cost:.2f}",
            ]
        )
    print_table(
        ["可达比例 MFU", "每 token 时延", "吞吐 tok/s", "prefill 2048 tok", "每百万 token $"],
        rows,
    )

    ideal = decode_step_time(device, spec, 1, mfu=1.0, context_tokens=CONTEXT)
    realistic = decode_step_time(device, spec, 1, mfu=0.5, context_tokens=CONTEXT)
    rule(
        f"MFU 1.0→0.5：每 token 时延从 {fmt_ms(ideal)} 涨到 {fmt_ms(realistic)}，"
        f"正好是 2 倍——把峰值当可达值，容量估算乐观一倍"
    )
    ensure(
        approx(realistic / ideal, 2.0, 1e-6),
        f"MFU 从 1.0 降到 0.5，每 token 时延应当正好翻倍，实际是 {realistic / ideal:.4f} 倍",
    )

    # prefill 是真算力瓶颈：它的时延几乎全部来自计算项
    prefill_flops_total = prefill_flops(spec, PREFILL_TOKENS)
    compute_only = prefill_flops_total / (device.fp16_tflops * 1e12 * MFU)
    memory_only = prefill_bytes(spec) / (device.hbm_bandwidth_gbs * 1e9 * MFU)
    modeled = prefill_time(device, spec, PREFILL_TOKENS, mfu=MFU)
    rule(
        f"prefill {PREFILL_TOKENS} token：计算项 {fmt_ms(compute_only)} vs "
        f"访存项 {fmt_ms(memory_only)}，模型给 {fmt_ms(modeled)}——受限于算力"
    )
    ensure(
        approx(modeled, compute_only, 1e-6),
        f"prefill 的时延应当等于计算项 {compute_only:.6f}s，实际是 {modeled:.6f}s",
    )
    ensure(
        compute_only > memory_only,
        "prefill 的计算项应当大于访存项——这正是它算力受限的证据",
    )

    # decode 侧：可达算力离峰值差两个数量级
    point = roofline(device, decode_intensity(spec, 1, CONTEXT))
    rule(
        f"decode batch=1 的可达算力 {point.attainable_tflops / 1e12:.2f} TFLOPS，"
        f"只有峰值 {point.peak_flops_per_s / 1e12:.0f} TFLOPS 的 {point.attainable_tflops / point.peak_flops_per_s * 100:.2f}%"
    )
    ensure(
        point.attainable_tflops / point.peak_flops_per_s < 0.01,
        "decode 的可达算力应当低于峰值的 1%",
    )
    rule(
        "算力侧的「可达比例」对 decode 几乎失效："
        "真正该调的是带宽与算法，不是换更会算的卡"
    )


# ---------------------------------------------------------------- 判断五


def check_pick_bandwidth_before_flops() -> None:
    """判断：选卡先看带宽——算力低 6 倍的卡每 token 成本反而更低。"""
    section("判断五：选卡先看带宽，不是算力")
    spec = model(MODEL)
    rows = cost_rows(spec)

    print_table(
        [
            "卡",
            "带宽 GB/s",
            "FP16 峰值 TFLOPS",
            "拐点",
            "每 token 下界 ms",
            "吞吐上界 tok/s",
            "每小时 $",
            "每百万 token $",
        ],
        [
            [
                row.device.name,
                f"{row.device.hbm_bandwidth_gbs:.0f}",
                f"{row.device.fp16_tflops:.0f}",
                f"{roofline(row.device, 1.0).ridge_intensity:.1f}",
                f"{row.weight_floor_s * 1000:.3f}",
                f"{row.tokens_per_s:.1f}",
                f"{row.device.price_per_hour_usd:.2f}",
                f"{row.cost_per_million:.2f}",
            ]
            for row in rows
        ],
    )

    by_cost = min(rows, key=lambda row: row.cost_per_million)
    by_flops = max(rows, key=lambda row: row.device.fp16_tflops)
    h100 = next(row for row in rows if row.device.name == "H100 SXM 80GB")
    consumer = next(row for row in rows if row.device.name == "RTX 4090 24GB")

    rule(
        f"每百万 token 成本最低：{by_cost.device.name}（{by_cost.cost_per_million:.2f} $）；"
        f"峰值算力最高：{by_flops.device.name}（{by_flops.device.fp16_tflops:.0f} TFLOPS）"
    )
    ensure(
        by_cost.device.name != by_flops.device.name,
        "如果成本最低的卡就是算力最高的卡，说明这张表没讲出「选卡先看带宽」这件事",
    )
    ensure(
        consumer.cost_per_million < h100.cost_per_million,
        f"RTX 4090 的每百万 token 成本（{consumer.cost_per_million:.2f} $）"
        f"应当低于 H100 SXM（{h100.cost_per_million:.2f} $）",
    )
    ensure(
        consumer.device.fp16_tflops < h100.device.fp16_tflops * 0.25,
        f"RTX 4090 的算力（{consumer.device.fp16_tflops:.0f} TFLOPS）"
        f"应当远低于 H100（{h100.device.fp16_tflops:.0f} TFLOPS）——"
        "算力低这么多，成本却更低",
    )
    rule(
        f"RTX 4090：算力只有 H100 的 "
        f"{consumer.device.fp16_tflops / h100.device.fp16_tflops * 100:.0f}%，"
        f"每百万 token 成本却便宜 "
        f"{(1 - consumer.cost_per_million / h100.cost_per_million) * 100:.0f}%"
    )

    # 排序依据：每美元能买多少带宽
    by_bw_dollar = max(rows, key=lambda row: row.bandwidth_per_dollar)
    ensure(
        by_bw_dollar.device.name == by_cost.device.name,
        f"带宽价格比最高的卡（{by_bw_dollar.device.name}，"
        f"{by_bw_dollar.bandwidth_per_dollar:.0f} GB/s per $/小时）"
        f"应当就是每百万 token 成本最低的卡，实际成本最低的是 {by_cost.device.name}",
    )

    # 两个地板给出的跨卡排序必须一致：说明结论不依赖选哪个口径
    def rank_by(metric: Callable[[CostRow], float]) -> list[str]:
        ordered = sorted(rows, key=metric)
        return [row.key for row in ordered]

    weight_rank = rank_by(lambda row: row.cost_per_million)
    kv_rank = rank_by(
        lambda row: row.device.price_per_hour_usd * (1e6 / (1.0 / row.kv_floor_s)) / 3600.0
    )
    ensure(
        weight_rank == kv_rank,
        "权重地板与 KV 地板给出的跨卡排序应当一致——"
        "否则「选卡先看带宽」的结论就只在一个口径下成立",
    )
    rule(
        "跨卡排序看的是「每美元能买多少 GB/s 带宽」，与峰值算力无关；"
        "两个口径给出同一个顺序，说明结论稳"
    )
    rule(
        "每百万 token 成本从低到高：" + " < ".join(weight_rank)
    )


def run() -> None:
    """跑第 3 章实验。"""
    banner("第 3 章 · Roofline 与算术强度")
    check_ridge_point_across_gpus()
    check_prefill_and_decode_live_on_different_sides()
    check_decode_cost_floor_is_bandwidth()
    check_peak_flops_overestimates_capacity()
    check_pick_bandwidth_before_flops()
    rule(
        "拐点算术强度 = 峰值算力 ÷ 峰值带宽；prefill 在拐点之上（算力瓶颈），"
        "decode 永远在拐点之下（带宽瓶颈）；decode 的每 token 成本下界由带宽决定；"
        "选卡先看带宽，不是算力"
    )
    print("\n✓ 第 3 章通过")
