"""第 10 章实验：推测解码——原理、盈亏平衡点与失效边界。

这个模块解决什么问题：推测解码常被宣传成"免费加速"，但它只在三个条件同时成立时才有收益——
target 模型没有被算力填满、draft 接受率够高、draft 本身足够便宜。
本章把这三个条件变成可计算的门槛：

1. **接受率扫描**：用 labkit 的 ``simulate_speculative`` 扫 0.1~1.0 画出加速比曲线，
   再用二分法求出**盈亏平衡接受率**（加速比刚好越过 1 的临界点）。
2. **批次衰减**：用 Roofline 工作点算出"多验证 K 个位置"的真实代价。
   访存瓶颈下它几乎免费，算力瓶颈下它按 K 倍收费——加速比因此随 batch 崩塌。
3. **draft 代价敏感度**：draft 从 0.5B 换到 7B，盈亏平衡点如何上移。

全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from ..labkit import (
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
    simulate_speculative,
)

DEVICE = gpu("h100_sxm")
TARGET = model("qwen2.5-7b")

DRAFT_VARIANTS: tuple[tuple[str, ModelSpec], ...] = (
    (
        "0.5B",
        ModelSpec(
            name="Qwen2.5-0.5B-Instruct（draft 角色）",
            params_b=0.494,
            num_layers=24,
            hidden_size=896,
            num_heads=14,
            num_kv_heads=2,
            head_dim=64,
            vocab_size=151936,
            max_context=32768,
            weight_bits=16,
            embedding_params_b=0.27,
        ),
    ),
    (
        "1.5B",
        ModelSpec(
            name="Qwen2.5-1.5B-Instruct（draft 角色）",
            params_b=1.543,
            num_layers=28,
            hidden_size=1536,
            num_heads=12,
            num_kv_heads=2,
            head_dim=128,
            vocab_size=151936,
            max_context=32768,
            weight_bits=16,
            embedding_params_b=0.47,
        ),
    ),
)
"""候选 draft 模型的规格。数值取自各模型公开 config.json，参数只当输入用，算账仍走 labkit。"""

CONTEXT_TOKENS = 512
"""decode 阶段已生成的上下文长度。它同时决定 KV 读取量与注意力的 FLOPs。"""

DRAFT_LENGTH = 4
"""一轮里 draft 猜几个 token。"""

MFU = 0.5
"""工程折损系数，与 labkit 的 decode_step_time 同一口径。"""

STEPS = 1000
"""模拟产出的 token 数。它只影响绝对时间，不影响加速比。"""

ACCEPTANCE = 0.6
"""默认接受率。取 0.6 是因为结构化/低熵任务的实测值大致在这个量级。"""


def round_time_s(
    spec: ModelSpec,
    *,
    batch: int,
    positions: int,
    context_tokens: int = CONTEXT_TOKENS,
    mfu: float = MFU,
) -> float:
    """一轮 forward 的耗时（秒）。

    flops 与访存字节都取自 labkit，可达吞吐由 labkit 的 ``roofline`` 给出，
    所以本函数只负责把"一次批量 forward"翻译成时间，不另立公式。

    关键在于 positions 只影响 flops、不影响访存：新位置的 KV 是**写**不是**读**。
    于是访存瓶颈区间里多验证几个位置几乎免费；一旦工作点越过 Roofline 拐点，
    flops 随 positions 线性增长，代价立刻显现——这正是 batch 变大后加速比崩塌的原因。
    """
    flops = batch * positions * decode_flops(spec, context_tokens)
    bytes_moved = spec.weights_bytes() + batch * context_tokens * spec.kv_bytes_per_token(2)
    point = roofline(DEVICE, flops / bytes_moved)
    # roofline 的 attainable 字段在两个分支下都是 FLOP/s 量纲（峰值算力 / 带宽上限）
    return flops / (point.attainable_tflops * mfu)


def target_step_s(batch: int = 1, context_tokens: int = CONTEXT_TOKENS) -> float:
    """target 模型一个 decode 步的耗时（秒），直接取 labkit 的带宽模型。"""
    return decode_step_time(
        DEVICE, TARGET, batch, mfu=MFU, context_tokens=context_tokens
    )


def draft_round_cost_s(
    draft_spec: ModelSpec,
    *,
    batch: int = 1,
    draft_length: int = DRAFT_LENGTH,
    context_tokens: int = CONTEXT_TOKENS,
) -> float:
    """一轮里 draft 的耗时（秒）。

    draft 要**连续**产 K 个 token，代价是 K 个串行 decode 步；
    它们可以被 target 的验证步掩盖，但物理时间确实要花。
    """
    return draft_length * round_time_s(
        draft_spec, batch=batch, positions=1, context_tokens=context_tokens
    )


def speedup_at_batch(
    batch: int,
    acceptance_rate: float,
    *,
    draft_spec: ModelSpec | None = None,
    draft_length: int = DRAFT_LENGTH,
    context_tokens: int = CONTEXT_TOKENS,
) -> float:
    """给定 batch 下的加速比。

    baseline 一轮每个序列产 1 个 token；推测一轮每个序列产
    ``1 + draft_length × 接受率`` 个 token，代价是 draft 的 K 个 decode 步
    加上 target 一次多位置验证。因为一批里所有序列并行，这个比值与 batch 无关地成立。
    """
    spec = draft_spec or DRAFT_VARIANTS[0][1]
    base = round_time_s(
        TARGET, batch=batch, positions=1, context_tokens=context_tokens
    )
    verify = round_time_s(
        TARGET, batch=batch, positions=draft_length + 1, context_tokens=context_tokens
    )
    draft = draft_round_cost_s(
        spec,
        batch=batch,
        draft_length=draft_length,
        context_tokens=context_tokens,
    )
    tokens_per_round = 1.0 + draft_length * acceptance_rate
    return tokens_per_round * base / (draft + verify)


def break_even_acceptance(
    *,
    draft_cost: float,
    target_cost: float,
    draft_length: int = DRAFT_LENGTH,
) -> float:
    """用二分法求"加速比刚好越过 1"的接受率。

    判定全部交给 labkit 的 ``simulate_speculative``，本函数只做搜索，
    不自己写加速比公式。
    """

    def speedup(rate: float) -> float:
        return simulate_speculative(
            rate,
            draft_cost=draft_cost,
            target_cost=target_cost,
            draft_length=draft_length,
        )["speedup"]

    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2
        if speedup(mid) > 1.0:
            high = mid
        else:
            low = mid
    return high


def check_token_conservation() -> None:
    """推测解码不改变输出 token 数，只改变每个 token 的平均耗时。"""
    section("1. 质量不变式：token 数守恒")
    result = simulate_speculative(
        ACCEPTANCE,
        draft_cost=draft_round_cost_s(DRAFT_VARIANTS[0][1]),
        target_cost=target_step_s(),
        draft_length=DRAFT_LENGTH,
    )
    produced = result["rounds"] * result["accepted_per_round"]
    ensure(
        approx(produced, STEPS, 1e-9),
        f"token 数应该守恒为 {STEPS}，实际 {produced:.4f}",
    )
    ensure(
        result["speedup"] <= DRAFT_LENGTH + 1 + 1e-9,
        f"并行验证的加速上界是 1+K={DRAFT_LENGTH + 1}，实际 {result['speedup']:.3f}",
    )
    rule(
        f"产出 {STEPS:.0f} 个 token 不变：baseline {fmt_ms(result['baseline_time'])} "
        f"→ 推测 {fmt_ms(result['speculative_time'])}，"
        f"加速比 {result['speedup']:.2f}×，每轮有效 token {result['accepted_per_round']:.2f}"
    )


def check_break_even_acceptance() -> None:
    """盈亏平衡接受率：低于它就是亏的，而且门槛随 draft 变贵而抬高。"""
    section("2. 接受率扫描与盈亏平衡点")
    target_cost = target_step_s()
    rows: list[list[str]] = []
    for rate in [i / 10 for i in range(1, 11)]:
        result = simulate_speculative(
            rate,
            draft_cost=draft_round_cost_s(DRAFT_VARIANTS[0][1]),
            target_cost=target_cost,
            draft_length=DRAFT_LENGTH,
        )
        rows.append(
            [
                f"{rate:.1f}",
                f"{result['accepted_per_round']:.2f}",
                f"{result['speedup']:.2f}",
            ]
        )
    print_table(["接受率", "每轮有效 token", "加速比"], rows)

    # 二分求临界点：所有判定都走 labkit 的 simulate_speculative
    draft_cost = draft_round_cost_s(DRAFT_VARIANTS[0][1])
    threshold = break_even_acceptance(
        draft_cost=draft_cost, target_cost=target_cost, draft_length=DRAFT_LENGTH
    )
    below = simulate_speculative(
        threshold - 1e-4,
        draft_cost=draft_cost,
        target_cost=target_cost,
        draft_length=DRAFT_LENGTH,
    )["speedup"]
    above = simulate_speculative(
        threshold + 1e-4,
        draft_cost=draft_cost,
        target_cost=target_cost,
        draft_length=DRAFT_LENGTH,
    )["speedup"]
    ensure(below <= 1.0, f"盈亏平衡点下方的加速比应该 ≤1，实际 {below:.4f}")
    ensure(above > 1.0, f"盈亏平衡点上方的加速比应该 >1，实际 {above:.4f}")
    rule(
        f"0.5B draft：盈亏平衡接受率 {threshold * 100:.2f}%（0.5B draft / 7B target，K={DRAFT_LENGTH}）"
    )

    # 闭式解与二分结果必须一致——这是模型自洽性检查，不是空断言
    closed_form = draft_cost / (DRAFT_LENGTH * target_cost)
    ensure(
        approx(threshold, closed_form, 1e-3),
        f"盈亏平衡点的闭式解 {closed_form:.6f} 与二分结果 {threshold:.6f} 不一致",
    )


def check_draft_cost_sensitivity() -> None:
    """draft 越贵，盈亏平衡点越高、可拿到的加速比越低。"""
    section("3. draft 代价敏感度")
    target_cost = target_step_s()
    rows: list[list[str]] = []
    thresholds: list[float] = []
    for label, spec in (*DRAFT_VARIANTS, ("7B（与 target 同规模）", TARGET)):
        draft_cost = draft_round_cost_s(spec)
        threshold = break_even_acceptance(
            draft_cost=draft_cost, target_cost=target_cost, draft_length=DRAFT_LENGTH
        )
        thresholds.append(threshold)
        best = simulate_speculative(
            1.0, draft_cost=draft_cost, target_cost=target_cost, draft_length=DRAFT_LENGTH
        )["speedup"]
        rows.append(
            [
                label,
                fmt_ms(draft_cost),
                f"{draft_cost / target_cost * 100:.1f}%",
                f"{threshold * 100:.2f}%",
                f"{best:.2f}",
            ]
        )
    print_table(
        ["draft 规模", "每轮 draft 代价", "占 target 比", "盈亏平衡接受率", "接受率=1 时加速比"],
        rows,
    )
    ensure(
        thresholds[0] < thresholds[1] < thresholds[2],
        f"draft 越大盈亏平衡点越高，实测 {thresholds}",
    )
    rule("draft 参数量每上一个数量级，盈亏平衡接受率就跟着抬高一截")


def check_batch_decay() -> None:
    """batch 变大后 target 算力饱和，加速比崩塌。"""
    section("4. 批次放大：target 算力饱和后加速比崩塌")
    rows: list[list[str]] = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        base = round_time_s(TARGET, batch=batch, positions=1)
        verify = round_time_s(TARGET, batch=batch, positions=DRAFT_LENGTH + 1)
        draft = draft_round_cost_s(DRAFT_VARIANTS[0][1], batch=batch)
        speedup = speedup_at_batch(batch, ACCEPTANCE)
        bound = roofline(
            DEVICE,
            batch * decode_flops(TARGET, CONTEXT_TOKENS)
            / (TARGET.weights_bytes() + batch * CONTEXT_TOKENS * TARGET.kv_bytes_per_token(2)),
        ).bound
        rows.append(
            [
                str(batch),
                fmt_ms(base),
                fmt_ms(verify),
                fmt_ms(draft),
                f"{verify / base:.2f}",
                f"{speedup:.2f}",
                "访存" if bound == "memory" else "算力",
            ]
        )
    print_table(
        ["batch", "target 一步", "target 验证 5 位", "draft 一轮", "验证/一步", "加速比", "瓶颈"],
        rows,
    )

    small = speedup_at_batch(1, ACCEPTANCE)
    ensure(
        small > 2.0,
        f"小 batch + 高接受率应该拿到 2× 以上，实际 {small:.3f}",
    )
    # 批量越大越差：抽查若干单调点
    for lo, hi in ((1, 32), (32, 128), (128, 256)):
        ensure(
            speedup_at_batch(hi, ACCEPTANCE) < speedup_at_batch(lo, ACCEPTANCE),
            f"batch 从 {lo} 到 {hi} 加速比应该下降",
        )
    crowded = speedup_at_batch(1024, ACCEPTANCE)
    ensure(
        crowded < 1.0,
        f"batch=1024 时 target 已算力饱和，加速比应该跌破 1，实际 {crowded:.3f}",
    )

    # 找出加速比跌破 1 的临界 batch
    cross_over = max(
        batch
        for batch in range(1, 2049)
        if speedup_at_batch(batch, ACCEPTANCE) > 1.0
    )
    rule(
        f"接受率 {ACCEPTANCE:.0%}、K={DRAFT_LENGTH} 时，加速比在 batch≈{cross_over} 处跌破 1"
    )


def check_low_acceptance_fails() -> None:
    """低接受率是另一个独立的失效原因，与 batch 无关。"""
    section("5. 接受率太低：再小的 batch 也救不回来")
    target_cost = target_step_s()
    draft_cost = draft_round_cost_s(DRAFT_VARIANTS[0][1])
    rows: list[list[str]] = []
    for rate in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        result = simulate_speculative(
            rate,
            draft_cost=draft_cost,
            target_cost=target_cost,
            draft_length=DRAFT_LENGTH,
        )
        rows.append(
            [f"{rate:.2f}", f"{result['accepted_per_round']:.2f}", f"{result['speedup']:.3f}"]
        )
    print_table(["接受率", "每轮有效 token", "加速比（batch=1 口径）"], rows)
    weak = simulate_speculative(
        0.1, draft_cost=draft_cost, target_cost=target_cost, draft_length=DRAFT_LENGTH
    )["speedup"]
    ensure(
        weak < 1.2,
        f"接受率 0.1 时收益应该微乎其微，实际加速比 {weak:.3f}",
    )


def check_consistency_with_labkit() -> None:
    """本模块的 Roofline 口径必须与 labkit 的 decode_step_time 在访存瓶颈区完全一致。"""
    section("6. 口径自检：与 labkit 的 decode_step_time 对齐")
    for batch in (1, 8, 32):
        mine = round_time_s(TARGET, batch=batch, positions=1)
        theirs = decode_step_time(
            DEVICE, TARGET, batch, mfu=MFU, context_tokens=CONTEXT_TOKENS
        )
        ensure(
            approx(mine, theirs, 1e-9),
            f"batch={batch} 时本模块 {mine * 1000:.4f}ms 与 labkit {theirs * 1000:.4f}ms 不一致",
        )
    # 访存瓶颈下"多验证几个位置"确实几乎免费
    base = round_time_s(TARGET, batch=1, positions=1)
    wide = round_time_s(TARGET, batch=1, positions=DRAFT_LENGTH + 1)
    ensure(
        approx(base, wide, 1e-9),
        f"batch=1 时验证 5 个位置应与 1 个位置同价，实测 {wide * 1000:.4f}ms vs {base * 1000:.4f}ms",
    )
    rule("访存瓶颈区：验证 5 个位置与验证 1 个位置同价——这就是小 batch 下加速比能翻倍的原因")


def run() -> None:
    """跑完第 10 章的全部实验。"""
    banner("第 10 章 · 推测解码与时延收益")
    rule(
        f"口径：{DEVICE.name} 单卡 / {TARGET.name}（fp16）/ "
        f"context={CONTEXT_TOKENS} tokens / mfu={MFU} / K={DRAFT_LENGTH}"
    )
    check_token_conservation()
    check_break_even_acceptance()
    check_draft_cost_sensitivity()
    check_batch_decay()
    check_low_acceptance_fails()
    check_consistency_with_labkit()
    print()
    rule(
        "结论：推测解码只在 target 未算力饱和、接受率过盈亏平衡线、draft 足够便宜时成立；"
        "输出 token 数一个不少，省下来的只是每个 token 的平均耗时"
    )
