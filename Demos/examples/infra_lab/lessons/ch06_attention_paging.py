"""第 6 章实验：Attention 变体与 PagedAttention 原理。

这个模块解决什么问题：两个"看起来不相关"的问题其实是同一个问题的两面——
**显存被谁吃掉了**。

- **变体这一面**：MHA / MQA / GQA 改的是 KV 头数，直接决定 KV 显存；
- **PagedAttention 这一面**：改的是 KV 的**分配方式**，决定同样大小的显存能装多少序列。

这一章把三个判断变成可执行的断言：

1. **GQA 让 KV 显存按 group_size 成倍下降**，且下降倍数严格等于分组数；
2. **FlashAttention 不改数学只改访存**：FLOPs 一个不少，省下的是峰值显存和 HBM 往返；
   而且它救的是 prefill，不是 decode——decode 的瓶颈始终是 KV 读取；
3. **分页把浪费从"按最大长度预留"压缩到"最后一页"**：连续预留浪费 63%，分页浪费 0.2%。

本章只讲原理，不写 kernel。全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..labkit import (
    GPUSpec,
    ModelSpec,
    PagedKVCache,
    banner,
    decode_step_time,
    ensure,
    gpu,
    kv_bytes_per_token,
    kv_capacity_tokens,
    kv_gib,
    model,
    prefill_flops,
    print_table,
    roofline,
    rule,
    section,
)

BLOCK_SIZE = 16
"""KV 分页的块大小。真实引擎常见 16/32/64。"""

TILE_ROWS = 128
"""FlashAttention 的行块大小 M。

真实引擎在 head_dim=128 上常用 128×64 的块配置。这个参数很关键：
M 越大，K/V 被重复搬运的次数越少，带宽代价越接近"只读一遍"。
"""

MFU = 0.5
"""可达带宽/算力占峰值的比例，和第 4 章保持同一口径。"""

SEQ_LENGTHS = (500, 1000, 1500, 2000, 3000, 4000, 5000, 7000)
"""并发场景：8 条序列的真实长度。长度各不相同，碎片才暴露得出来。"""

MAX_CONTEXT = 8192
"""上下文上限。传统做法按它预留空间，这正是浪费的来源。"""

DEVICE = "h100_sxm"
"""算账用的卡。"""


@dataclass(frozen=True)
class AttentionCost:
    """一次 prefill 的访存账。"""

    context: int
    """序列长度 S。"""

    flops: float
    """整个 prefill 的浮点运算量。"""

    naive_matrix_gb: float
    """朴素实现单层 S×S 注意力矩阵的体积（GB）。"""

    flash_buffer_mb: float
    """FlashAttention 单层片上缓冲的体积（MB）。"""

    naive_extra_gb: float
    """朴素实现多出的 HBM 往返（矩阵写出→读回→softmax 写出→读回），全模型合计。"""

    flash_extra_gb: float
    """FlashAttention 的 K/V 重复搬运量，全模型合计。"""


def attention_cost(
    device: GPUSpec,
    spec: ModelSpec,
    context: int,
) -> AttentionCost:
    """算一次 prefill 的访存账。

    这里补的是 labkit 没覆盖的一块：**注意力中间量的访存**。
    labkit 负责权重、KV、FLOPs、Roofline 这些通用口径；
    注意力矩阵是 FlashAttention 独有的账，只能在这里按论文模型展开。
    """
    flops = prefill_flops(spec, context)
    naive_matrix = context * context * 2 * spec.num_heads
    flash_buffer = context * spec.head_dim * 2 * spec.num_heads
    # 朴素实现：每层每头 4 趟 S×S 往返（写矩阵、读回、softmax 写出、再读回）
    naive_extra = 4 * context * context * 2 * spec.num_heads * spec.num_layers
    # FlashAttention：每行块把 K、V 各读一遍，S/M 个行块
    flash_extra = (
        2 * spec.num_heads * 2 * context * spec.head_dim * (context / TILE_ROWS) * spec.num_layers
    )
    return AttentionCost(
        context=context,
        flops=flops,
        naive_matrix_gb=naive_matrix / 1e9,
        flash_buffer_mb=flash_buffer / 1e6,
        naive_extra_gb=naive_extra / 1e9,
        flash_extra_gb=flash_extra / 1e9,
    )


def check_attention_variants_shrink_kv() -> None:
    """判断：GQA 让 KV 显存按 group_size 成倍下降。"""
    section("判断一：MHA / GQA / MQA 是同一个旋钮的三档")
    base = model("llama3.1-8b")

    variants = [
        ("MHA（KV 头 = 查询头）", replace(base, num_kv_heads=base.num_heads)),
        ("GQA-8（32 个查询头分 8 组）", replace(base, num_kv_heads=8)),
        ("GQA-4", replace(base, num_kv_heads=4)),
        ("MQA（所有查询头共享 1 组 KV）", replace(base, num_kv_heads=1)),
    ]
    per_head_bytes = kv_bytes_per_token(base) / base.num_kv_heads

    rows: list[list[str]] = []
    for label, variant in variants:
        per_token = kv_bytes_per_token(variant)
        ensure(
            per_token == per_head_bytes * variant.num_kv_heads,
            f"{label} 的每 token KV 字节数应当与 KV 头数严格成正比",
        )
        ensure(
            variant.num_kv_heads <= variant.num_heads,
            f"{label} 的 KV 头数不能超过查询头数",
        )
        rows.append(
            [
                label,
                str(variant.num_kv_heads),
                f"{per_token / 1024:.0f} KB",
                f"{base.num_heads / variant.num_kv_heads:.0f}×",
                f"{kv_gib(variant, 32768):.2f}",
            ]
        )

    print_table(
        ["注意力变体", "KV 头数", "每 token 字节", "相对 MHA 省", "32K 单条 KV (GiB)"],
        rows,
    )

    mha, gqa8, mqa = variants[0][1], variants[1][1], variants[3][1]
    ensure(
        kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(gqa8),
        "从 MHA 换成 GQA-8，KV 字节数应当正好降到 1/4",
    )
    ensure(
        kv_bytes_per_token(gqa8) == 8 * kv_bytes_per_token(mqa),
        "从 GQA-8 换成 MQA，KV 字节数应当再降到 1/8",
    )
    ensure(
        kv_gib(mha, 32768) == 32 * kv_gib(mqa, 32768),
        "32K 上下文下，MQA 的 KV 应当只有 MHA 的 1/32",
    )
    rule(
        "KV 显存 = 2 × 层数 × KV头数 × head_dim × 元素字节；"
        "GQA 把 KV 头数砍到 1/group_size，显存就砍到 1/group_size"
    )
    rule(
        "代价不在显存而在质量：MQA 把所有查询头压到一组 KV 上，"
        "多任务混合训练时会掉点；GQA 折中，group_size=8 时质量基本回到 MHA 水平"
    )


def check_flash_attention_saves_bandwidth_not_flops() -> None:
    """判断：FlashAttention 省的是访存和显存，不是 FLOPs。"""
    section("判断二：FlashAttention 改的是访存，不是数学")
    device = gpu(DEVICE)
    spec = model("qwen2.5-7b")
    weight_bytes = spec.weights_bytes(spec.weight_bits)

    costs = [
        attention_cost(device, spec, ctx) for ctx in (2048, 4096, 8192, 32768)
    ]
    rows: list[list[str]] = []
    for cost in costs:
        rows.append(
            [
                f"{cost.context:,}",
                f"{cost.flops / 1e12:.1f} T",
                f"{cost.naive_matrix_gb:.2f}",
                f"{cost.flash_buffer_mb:.1f}",
                f"{cost.naive_matrix_gb * 1e9 / (cost.flash_buffer_mb * 1e6):.0f}×",
            ]
        )
    print_table(
        ["上下文 S", "prefill FLOPs", "朴素单层矩阵 (GB)", "Flash 单层片上 (MB)", "峰值显存之比"],
        rows,
    )

    # FLOPs 由数学唯一确定：注意力项是 2 × 层数 × hidden × S²，
    # 里面没有任何"实现参数"——FlashAttention 换实现，这一项纹丝不动
    biggest = costs[-1]
    attention_term = 2 * spec.num_layers * spec.hidden_size * biggest.context**2
    gemm_term = 2 * spec.non_embedding_params_b * 1e9 * biggest.context
    ensure(
        abs((prefill_flops(spec, biggest.context) - gemm_term) - attention_term)
        < 1e-6 * attention_term,
        "prefill 的注意力 FLOPs 项应当严格等于 2 × 层数 × hidden × S²",
    )
    ensure(
        biggest.naive_matrix_gb * 1e9 / (biggest.flash_buffer_mb * 1e6)
        == biggest.context / spec.head_dim,
        "峰值显存之比应当正好等于 S ÷ head_dim",
    )
    rule(
        f"FLOPs 是 {biggest.flops / 1e12:.1f} TFLOP，FlashAttention 一个都没少——"
        f"它省的是峰值显存：32K 上下文下单层从 {biggest.naive_matrix_gb:.1f} GB "
        f"降到 {biggest.flash_buffer_mb:.0f} MB，正好省 S÷head_dim = "
        f"{biggest.context // spec.head_dim} 倍"
    )

    # 带宽账：朴素实现把 S×S 矩阵在 HBM 里来回搬 4 趟，
    # FlashAttention 换成"每个行块重读一遍 K/V"
    naive_ai = biggest.flops / (weight_bytes + biggest.naive_extra_gb * 1e9)
    flash_ai = biggest.flops / (weight_bytes + biggest.flash_extra_gb * 1e9)
    naive_point = roofline(device, naive_ai)
    flash_point = roofline(device, flash_ai)
    effective_bw = device.hbm_bandwidth_gbs * 1e9 * MFU
    effective_tflops = device.fp16_tflops * 1e12 * MFU
    traffic_ms = biggest.naive_extra_gb * 1e9 / effective_bw * 1000
    compute_ms = biggest.flops / effective_tflops * 1000

    rule(
        f"32K prefill 的算术强度：朴素 {naive_ai:.0f} FLOP/byte（{naive_point.bound} bound）"
        f" → Flash {flash_ai:.0f} FLOP/byte（{flash_point.bound} bound）"
    )
    rule(
        f"朴素实现多搬 {biggest.naive_extra_gb:,.0f} GB，"
        f"按有效带宽 {device.hbm_bandwidth_gbs:.0f} GB/s × MFU {MFU} 算是 {traffic_ms:.0f} ms；"
        f"而全部计算只要 {compute_ms:.0f} ms"
    )

    ensure(
        biggest.flash_extra_gb < biggest.naive_extra_gb,
        f"FlashAttention 的 HBM 搬运量应当少于朴素实现，"
        f"实际 {biggest.flash_extra_gb:.0f} GB vs {biggest.naive_extra_gb:.0f} GB",
    )
    ensure(
        abs(biggest.flash_extra_gb * 2 - biggest.naive_extra_gb) < 1e-6,
        "在 head_dim = M = 128 时，FlashAttention 的搬运量应当正好是朴素实现的一半",
    )
    ensure(
        flash_ai > naive_ai,
        f"FlashAttention 的算术强度应当高于朴素实现，"
        f"实际 {flash_ai:.0f} vs {naive_ai:.0f}",
    )
    ensure(
        naive_point.bound == "memory" and flash_point.bound == "memory",
        f"32K 上下文下 prefill 仍然访存受限（朴素 {naive_point.bound} / "
        f"Flash {flash_point.bound}）——FlashAttention 省的是往返，不是定性",
    )
    ensure(
        traffic_ms > compute_ms,
        f"朴素实现的额外搬运时间 {traffic_ms:.0f}ms 应当超过全部计算时间 "
        f"{compute_ms:.0f}ms——这正是 FlashAttention 要消灭的那笔开销",
    )

    # 但 decode 不受这件事影响：decode 的字节数只由权重和 KV 决定
    step = decode_step_time(device, spec, 1, context_tokens=32768)
    ensure(
        step > 0,
        "decode 步时延应当为正——它的字节数只由权重与 KV 决定，与注意力实现无关",
    )
    rule(
        "FlashAttention 救的是 prefill（长序列计算），不是 decode——"
        "decode 的瓶颈从头到尾是读 KV，换什么 kernel 都一样（第 4、5 章的结论）"
    )


def check_paging_beats_contiguous_allocation() -> None:
    """判断：分页把浪费从"按最大长度预留"压缩到"最后一页"。"""
    section("判断三：连续预留 vs 分页按需分配")
    lengths = list(SEQ_LENGTHS)
    max_context = MAX_CONTEXT
    useful_tokens = sum(lengths)

    # 连续分配：每条序列按 max_context 预留一整块连续空间
    contiguous_blocks = sum(math.ceil(max_context / BLOCK_SIZE) for _ in lengths)
    contiguous_capacity = contiguous_blocks * BLOCK_SIZE
    contiguous_waste = contiguous_capacity - useful_tokens
    contiguous_frag = contiguous_waste / contiguous_capacity

    # 分页分配：按需分配块，只浪费最后一页没填满的部分
    cache = PagedKVCache(block_size=BLOCK_SIZE)
    for length in lengths:
        remaining = length
        while remaining > 0:
            block_id = cache.allocate()
            cache.append_tokens(block_id, min(BLOCK_SIZE, remaining))
            remaining -= BLOCK_SIZE
    paged_blocks = cache.total_blocks()
    paged_capacity = paged_blocks * BLOCK_SIZE
    paged_waste = paged_capacity - useful_tokens
    paged_frag = cache.fragmentation()

    print_table(
        ["分配方式", "占用块数", "容量 (token)", "有效 token", "浪费 token", "浪费率"],
        [
            [
                "连续预留（按 max_context）",
                str(contiguous_blocks),
                f"{contiguous_capacity:,}",
                f"{useful_tokens:,}",
                f"{contiguous_waste:,}",
                f"{contiguous_frag * 100:.2f}%",
            ],
            [
                "分页按需分配",
                str(paged_blocks),
                f"{paged_capacity:,}",
                f"{useful_tokens:,}",
                f"{paged_waste:,}",
                f"{paged_frag * 100:.2f}%",
            ],
        ],
    )

    ensure(
        abs(paged_frag - paged_waste / paged_capacity) < 1e-9,
        f"分页的浪费率应当等于 (容量 - 有效) ÷ 容量，实际 {paged_frag}",
    )
    ensure(
        contiguous_frag > 0.5,
        f"连续预留的浪费率应当过半，实际 {contiguous_frag * 100:.2f}%",
    )
    ensure(
        paged_waste <= len(lengths) * (BLOCK_SIZE - 1),
        f"分页的浪费应当只来自每条序列的最后一页（≤ {len(lengths)} × {BLOCK_SIZE - 1} "
        f"= {len(lengths) * (BLOCK_SIZE - 1)} token），实际 {paged_waste}",
    )
    ensure(
        paged_frag < contiguous_frag / 50,
        f"分页的浪费率 {paged_frag * 100:.3f}% 应当比连续预留的 "
        f"{contiguous_frag * 100:.2f}% 低两个数量级",
    )
    ensure(
        paged_blocks == sum(math.ceil(length / BLOCK_SIZE) for length in lengths),
        "分页分配的块数应当正好等于各序列实际需要向上取整后的块数之和",
    )
    rule(
        f"同一批请求：连续预留浪费 {contiguous_waste:,} token（{contiguous_frag * 100:.1f}%，"
        f"相当于要了 {contiguous_capacity / useful_tokens:.2f} 倍的显存），"
        f"分页只浪费 {paged_waste} token（{paged_frag * 100:.2f}%）"
    )
    rule(
        f"差距 {contiguous_frag / paged_frag:.0f} 倍——"
        "浪费从'按最大长度预留'压缩到了'最后一页没填满'"
    )


def check_paging_enables_more_concurrency() -> None:
    """判断：同样的 KV 预算，分页能装下多得多的并发序列。"""
    section("判断四：省下来的碎片就是并发量")
    spec = model("llama3.1-8b")
    budget_gib = 20.0
    budget_bytes = budget_gib * 1024**3
    max_context = MAX_CONTEXT

    # 平均真实长度直接取自判断三的同一批请求，不另设口径
    avg_actual = sum(SEQ_LENGTHS) / len(SEQ_LENGTHS)
    contiguous_seqs = int(budget_bytes // (kv_bytes_per_token(spec) * max_context))
    paged_seqs = int(budget_bytes // (kv_bytes_per_token(spec) * avg_actual))
    # 连续预留下这批序列真正用掉的 KV
    contiguous_real = contiguous_seqs * kv_bytes_per_token(spec) * avg_actual

    print_table(
        ["分配方式", "KV 预算", "可放序列数", "实际有效 KV", "预算利用率"],
        [
            [
                "连续预留",
                f"{budget_gib:.0f} GiB",
                str(contiguous_seqs),
                f"{contiguous_real / 1024**3:.2f} GiB",
                f"{contiguous_real / budget_bytes * 100:.1f}%",
            ],
            [
                "分页按需",
                f"{budget_gib:.0f} GiB",
                str(paged_seqs),
                f"{budget_gib:.0f} GiB",
                "100.0%",
            ],
        ],
    )

    ensure(
        paged_seqs > contiguous_seqs * 2,
        f"同样的 KV 预算，分页应当能装下 {contiguous_seqs * 2} 条以上的序列，"
        f"实际连续 {contiguous_seqs} 条 vs 分页 {paged_seqs} 条",
    )
    ensure(
        contiguous_real < budget_bytes * 0.6,
        f"连续预留下预算利用率应当明显偏低，实际 {contiguous_real / budget_bytes * 100:.1f}%",
    )
    ensure(
        kv_capacity_tokens(spec, budget_bytes) >= paged_seqs,
        "分页并发数不应超过按每 token 字节数算出的理论上限",
    )
    rule(
        f"同样的 {budget_gib:.0f} GiB KV 预算：连续预留只能放 {contiguous_seqs} 条，"
        f"分页能放 {paged_seqs} 条——多出来的 {paged_seqs - contiguous_seqs} 条并发"
        "就是碎片换来的"
    )
    rule(
        "分页的另一个副产品是块表：每条序列记一串块号，注意力核按块表取 KV，"
        "KV 因此不必连续——这正是 PagedAttention 名字的来源"
    )


def run() -> None:
    """跑第 6 章实验。"""
    banner("第 6 章 · Attention 变体与 PagedAttention 原理")
    check_attention_variants_shrink_kv()
    check_flash_attention_saves_bandwidth_not_flops()
    check_paging_beats_contiguous_allocation()
    check_paging_enables_more_concurrency()
    rule(
        "GQA 按 group_size 省 KV；FlashAttention 省访存不省 FLOPs，"
        "且只救 prefill；分页把浪费从'预留最大长度'压到'最后一页'"
    )
    print("\n✓ 第 6 章通过")
