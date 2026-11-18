"""第 9 章实验：量化——精度边界与成本收益。

这个模块解决什么问题：把"要不要量化"这个判断题变成可量化的三张表。

1. **精度表**：位宽 × 分组大小 → 信噪比（SNR）。要证明"group size 越小精度越好，
   但元数据开销越大"——这是量化里唯一真正需要权衡的地方。
2. **收益表**：同一张卡上，按 FP16 / INT8 / INT4 算 decode 步耗时与 prefill 耗时。
   要证明**decode 变快的幅度 ≈ 权重变小的幅度，而 prefill 几乎不变**——
   量化省的是带宽，不是算力。
3. **摊薄表**：batch 变大、上下文变长之后，量化的收益被 KV 流量摊薄多少。

所有耗时、显存口径一律调用 :mod:`infra_lab.labkit`（``decode_step_time`` /
``prefill_time`` / ``weight_gib`` / ``quantize_values``），
本文件不另写任何带宽或 FLOPs 公式。
"""

from __future__ import annotations

import math

from ..labkit import (
    banner,
    decode_step_time,
    ensure,
    fmt_bytes,
    gpu,
    model,
    plan_memory,
    prefill_time,
    print_table,
    quantize_values,
    rule,
    section,
    weight_gib,
)

__all__ = [
    "build_activation_samples",
    "check_decode_scales_with_weights",
    "check_group_size_tradeoff",
    "check_prefill_is_not_affected",
    "check_quantization_pays_off",
]

GROUP_SIZES: tuple[int, ...] = (32, 64, 128, 256)
"""常见的分组大小。分组越小，每组一个缩放因子，元数据越多、精度越高。"""

BIT_WIDTHS: tuple[int, ...] = (4, 8, 16)
"""量化位宽：INT4 / INT8 / FP16（16 位在这里当作"不量化"的参照）。"""


def build_activation_samples(count: int = 8192) -> list[float]:
    """造一份"像激活值"的样本：多数接近零，偶尔有离群点。

    为什么必须带离群点：均匀分布的数据上，分组大小几乎不影响精度
    （每组的最大值都差不多），那样就测不出分组的作用了。
    真实激活是**重尾**的——少数通道的值大出一个数量级，
    而分组量化用组内最大值定标点，一个离群点会污染整组的精度。
    """
    values: list[float] = []
    state = 7
    for index in range(count):
        state = (1103515245 * state + 12345) % 2147483648
        uniform = (state / 2147483648.0) * 2 - 1
        value = math.tanh(uniform * 2.2)
        if index % 97 == 0:
            value *= 12.0
        values.append(value)
    return values


def _metadata_bytes(params_b: float, group_size: int, dtype_bytes: int = 2) -> int:
    """分组量化的元数据开销：每组一个缩放因子（用 FP16 存）。

    这是量化唯一"多出来"的东西：位宽越小、省下的权重越少，
    元数据占比就越显眼。
    """
    groups = params_b * 1e9 / group_size
    return int(groups * dtype_bytes)


def check_group_size_tradeoff() -> list[dict[str, float]]:
    """位宽 × 分组大小的 SNR 表：小分组更准，但元数据更贵。

    要证明的是两件事同时成立：
    - 位宽每降一档，SNR 掉一个数量级（4 bit 的误差是 8 bit 的 3~4 倍）；
    - 分组每缩小一档，SNR 回升，但元数据开销按比例涨上去。
    """
    samples = build_activation_samples()
    rows: list[dict[str, float]] = []

    for bits in BIT_WIDTHS:
        for group_size in GROUP_SIZES:
            result = quantize_values(samples, bits, group_size)
            rows.append(
                {
                    "bits": float(bits),
                    "group_size": float(group_size),
                    "snr_db": result.snr_db,
                    "relative_error": result.relative_error,
                    "max_abs_error": result.max_abs_error,
                }
            )

    by_key = {(int(row["bits"]), int(row["group_size"])): row for row in rows}

    # 1) 位宽是主变量：INT4 的 SNR 应该明显低于 INT8，INT8 明显低于 FP16
    ensure(
        by_key[(4, 128)]["snr_db"] < by_key[(8, 128)]["snr_db"] - 10,
        f"INT4 的 SNR（{by_key[(4, 128)]['snr_db']:.1f}dB）应该明显低于 INT8"
        f"（{by_key[(8, 128)]['snr_db']:.1f}dB）",
    )
    ensure(
        by_key[(8, 128)]["snr_db"] < by_key[(16, 128)]["snr_db"] - 20,
        f"INT8 的 SNR（{by_key[(8, 128)]['snr_db']:.1f}dB）应该明显低于 FP16"
        f"（{by_key[(16, 128)]['snr_db']:.1f}dB）",
    )

    # 2) 分组大小是次变量：分组越小，同一位宽下 SNR 越高
    for bits in (4, 8):
        small = by_key[(bits, 32)]["snr_db"]
        large = by_key[(bits, 256)]["snr_db"]
        ensure(
            small > large,
            f"{bits} bit 下 group 32 的 SNR（{small:.1f}dB）应该高于 group 256"
            f"（{large:.1f}dB）——离群点污染的组更少了",
        )

    # 3) 精度换来的代价：元数据开销
    spec = model("qwen2.5-7b")
    payload_int4 = spec.weights_bytes(4)
    meta_small = _metadata_bytes(spec.params_b, 32)
    meta_large = _metadata_bytes(spec.params_b, 256)
    ensure(
        meta_small > meta_large * 3,
        f"group 32 的元数据（{fmt_bytes(meta_small)}）应该是 group 256"
        f"（{fmt_bytes(meta_large)}）的好几倍",
    )
    overhead_small = meta_small / payload_int4
    ensure(
        overhead_small > 0.05,
        f"INT4 + group 32 的元数据开销（{overhead_small * 100:.1f}%）已经不可忽略",
    )
    return rows


def check_decode_scales_with_weights() -> dict[str, dict[str, float]]:
    """decode 的加速幅度 ≈ 权重变小的幅度。

    权重从 16 bit 降到 4 bit，体积变成 1/4；
    如果 decode 真的是被权重访存卡住的，那步耗时就该掉到 1/4 附近。
    """
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    results: dict[str, dict[str, float]] = {}
    for bits in (16, 8, 4):
        step = decode_step_time(
            device, spec, 8, mfu=0.5, weight_bits=bits, context_tokens=512
        )
        results[str(bits)] = {
            "step_ms": step * 1000,
            "weight_gib": weight_gib(spec, bits),
        }

    fp16 = results["16"]
    int8 = results["8"]
    int4 = results["4"]

    ensure(
        fp16["weight_gib"] / int4["weight_gib"] > 3.5,
        "权重体积应该从 16 bit 到 4 bit 缩小约 4 倍",
    )
    speedup = fp16["step_ms"] / int4["step_ms"]
    ensure(
        3.0 < speedup < 4.2,
        f"decode 步耗时应该跟着权重缩小差不多（实测加速 {speedup:.2f} 倍，"
        f"权重缩小 {fp16['weight_gib'] / int4['weight_gib']:.2f} 倍）",
    )
    speedup_8 = fp16["step_ms"] / int8["step_ms"]
    ensure(
        1.6 < speedup_8 < 2.1,
        f"INT8 的 decode 加速（{speedup_8:.2f} 倍）应该约等于权重缩小 2 倍",
    )
    return results


def check_prefill_is_not_affected() -> dict[str, float]:
    """prefill 几乎不受量化影响：它是算力瓶颈，不是带宽瓶颈。

    prefill 阶段 token 多、权重被反复复用，算术强度高；
    权重变小省下的那点带宽，早就被算力上限吃掉了。
    """
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    times = {
        str(bits): prefill_time(device, spec, 2048, mfu=0.5, weight_bits=bits) * 1000
        for bits in (16, 8, 4)
    }
    ensure(
        times["16"] > 0 and abs(times["4"] - times["16"]) / times["16"] < 0.02,
        f"prefill 耗时应��随位宽变化（FP16 {times['16']:.1f}ms vs INT4 {times['4']:.1f}ms）",
    )
    return times


def check_quantization_pays_off() -> dict[str, float]:
    """什么时候量化不划算：显存不紧张、或者 KV 流量占主导的时候。

    量化省的是**访存**，所以它只在"权重访存占比高"的场景里值钱：
    - batch 越小、上下文越短，权重占比越高，量化收益越足；
    - batch 越大、上下文越长，KV 流量占比越高，权重的份额被摊薄。
    """
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    small_batch = decode_step_time(
        device, spec, 1, mfu=0.5, weight_bits=16, context_tokens=512
    )
    small_batch_int4 = decode_step_time(
        device, spec, 1, mfu=0.5, weight_bits=4, context_tokens=512
    )
    big_batch = decode_step_time(
        device, spec, 64, mfu=0.5, weight_bits=16, context_tokens=8192
    )
    big_batch_int4 = decode_step_time(
        device, spec, 64, mfu=0.5, weight_bits=4, context_tokens=8192
    )

    speedup_small = small_batch / small_batch_int4
    speedup_big = big_batch / big_batch_int4

    ensure(
        speedup_small > speedup_big,
        f"小 batch 时量化的收益（{speedup_small:.2f} 倍）应该大于大 batch + 长上下文"
        f"（{speedup_big:.2f} 倍）——后者 KV 流量把权重份额摊薄了",
    )

    # 显存本来就不紧张时，量化的收益只剩速度，没有容量收益
    budget = plan_memory(spec, device, kv_tokens=0, weight_bits=16)
    ensure(
        budget.fits(),
        "FP16 权重 + 激活 + 运行时开销本来就装得下，"
        "说明这张卡上量化的主要收益是速度而不是容量",
    )
    return {
        "speedup_small_batch": speedup_small,
        "speedup_big_batch": speedup_big,
        "fp16_weight_gib": weight_gib(spec, 16),
        "int4_weight_gib": weight_gib(spec, 4),
    }


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 9 章的全部实验与断言。"""
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    banner("第 9 章 · 量化：精度边界与成本收益")
    rule(
        f"设备：{device.name}　模型：{spec.name}　"
        f"权重 FP16 {weight_gib(spec, 16):.2f}GiB / INT8 {weight_gib(spec, 8):.2f}GiB / "
        f"INT4 {weight_gib(spec, 4):.2f}GiB"
    )

    section("一、精度表：位宽 × 分组大小 → 信噪比")
    rows = check_group_size_tradeoff()
    print_table(
        ["位宽", "group size", "SNR(dB)", "相对误差", "最大绝对误差", "权重元数据开销"],
        [
            [
                f"{int(row['bits'])} bit",
                int(row["group_size"]),
                f"{row['snr_db']:.2f}",
                f"{row['relative_error']:.5f}",
                f"{row['max_abs_error']:.5f}",
                f"{_metadata_bytes(spec.params_b, int(row['group_size'])) / spec.weights_bytes(int(row['bits'])) * 100:.2f}%",
            ]
            for row in rows
        ],
    )
    rule(
        f"INT4 下 group 32 的 SNR 是 {_snr(rows, 4, 32):.1f}dB，group 256 只有 "
        f"{_snr(rows, 4, 256):.1f}dB；代价是元数据从 "
        f"{_metadata_bytes(spec.params_b, 32) / spec.weights_bytes(4) * 100:.1f}% 涨到 "
        f"{_metadata_bytes(spec.params_b, 256) / spec.weights_bytes(4) * 100:.1f}%"
    )

    section("二、收益表：量化到底让什么变快")
    decode_results = check_decode_scales_with_weights()
    prefill_times = check_prefill_is_not_affected()
    print_table(
        ["方案", "权重体积(GiB)", "decode 步(ms)", "相对 FP16", "prefill 2048(ms)", "相对 FP16"],
        [
            [
                f"FP16",
                f"{decode_results['16']['weight_gib']:.2f}",
                f"{decode_results['16']['step_ms']:.2f}",
                "1.00×",
                f"{prefill_times['16']:.1f}",
                "1.00×",
            ],
            [
                f"INT8",
                f"{decode_results['8']['weight_gib']:.2f}",
                f"{decode_results['8']['step_ms']:.2f}",
                f"{decode_results['16']['step_ms'] / decode_results['8']['step_ms']:.2f}×",
                f"{prefill_times['8']:.1f}",
                f"{prefill_times['16'] / prefill_times['8']:.2f}×",
            ],
            [
                f"INT4",
                f"{decode_results['4']['weight_gib']:.2f}",
                f"{decode_results['4']['step_ms']:.2f}",
                f"{decode_results['16']['step_ms'] / decode_results['4']['step_ms']:.2f}×",
                f"{prefill_times['4']:.1f}",
                f"{prefill_times['16'] / prefill_times['4']:.2f}×",
            ],
        ],
    )
    rule(
        "decode 跟着权重一起缩小，prefill 几乎不动——"
        "量化省的是访存带宽，prefill 卡在算力上限上"
    )

    section("三、什么时候量化不划算")
    payoff = check_quantization_pays_off()
    print_table(
        ["场景", "FP16 步耗时(ms)", "INT4 步耗时(ms)", "加速"],
        [
            [
                "batch=1, ctx=512",
                f"{decode_step_time(device, spec, 1, mfu=0.5, weight_bits=16, context_tokens=512) * 1000:.2f}",
                f"{decode_step_time(device, spec, 1, mfu=0.5, weight_bits=4, context_tokens=512) * 1000:.2f}",
                f"{payoff['speedup_small_batch']:.2f}×",
            ],
            [
                "batch=64, ctx=8192",
                f"{decode_step_time(device, spec, 64, mfu=0.5, weight_bits=16, context_tokens=8192) * 1000:.2f}",
                f"{decode_step_time(device, spec, 64, mfu=0.5, weight_bits=4, context_tokens=8192) * 1000:.2f}",
                f"{payoff['speedup_big_batch']:.2f}×",
            ],
        ],
    )
    rule(
        f"大 batch + 长上下文时，KV 流量把权重的份额摊薄，量化加速从 "
        f"{payoff['speedup_small_batch']:.2f}× 掉到 {payoff['speedup_big_batch']:.2f}×"
    )
    rule(
        f"而且这张卡上 FP16 权重（{payoff['fp16_weight_gib']:.2f}GiB）本来就装得下，"
        "量化的主要收益是速度而不是容量"
    )


def _snr(rows: list[dict[str, float]], bits: int, group_size: int) -> float:
    """从扫描结果里取一行 SNR。"""
    for row in rows:
        if int(row["bits"]) == bits and int(row["group_size"]) == group_size:
            return row["snr_db"]
    return 0.0
