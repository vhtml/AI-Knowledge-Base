"""第 5 章实验：KV Cache 的显存账与复用。

这个模块解决什么问题：长上下文服务最常见的 OOM 原因不是模型太大，而是**KV cache
把显存吃光了**。但 KV 该占多少、为什么会占这么多、哪些手段能省，
往往只靠"感觉"。这一章把 KV 的账算到字节，并把四条判断变成可执行的断言：

1. **每 token 的 KV 字节数是精确公式**，不是估算——2 × 层数 × KV 头数 × head_dim × 元素字节；
2. **GQA 让 KV 显存按 group_size 成倍下降**，几乎不损质量；
3. **长上下文服务的瓶颈是 KV 而不是权重**：给出"并发数 × 上下文长度"的判定阈值；
4. **KV 量化**用一半显存换有限精度，**前缀复用**直接把重复的 prefill 省掉。

全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..labkit import (
    ModelSpec,
    PagedKVCache,
    banner,
    ensure,
    fmt_bytes,
    fmt_gib,
    fmt_ms,
    gpu,
    kv_bytes_per_token,
    kv_gib,
    model,
    plan_memory,
    prefill_time,
    prefix_hash,
    print_table,
    quantize_values,
    rule,
    section,
    weight_gib,
)

DEVICE = "h100_sxm"
"""算账用的卡。80GB 显存，够放下大部分 7B~13B 模型的 fp16 权重。"""

ACTIVATION_GIB = 1.0
"""推理时的临时激活显存。"""

RUNTIME_GIB = 2.0
"""框架、CUDA context、通信缓冲这类固定开销。"""

BLOCK_SIZE = 16
"""KV 分页的块大小。真实引擎常见 16/32/64。"""


@dataclass(frozen=True)
class KVRow:
    """一个模型在某个上下文长度下的 KV 账本行。"""

    name: str
    kv_bytes_per_token: int
    """每 token 的 KV 字节数。"""

    group_size: int
    """GQA 分组大小：查询头数 ÷ KV 头数。等于 1 就是 MHA。"""

    context_tokens: int
    kv_gib: float
    """单条序列在给定上下文下的 KV 占用。"""

    concurrent: int
    """剩余显存能装下的并发序列数。"""

    kv_share: float
    """KV 占已用显存的比例。"""


def group_size(spec: ModelSpec) -> int:
    """GQA 分组大小：多少个查询头共享一组 KV 头。"""
    return spec.num_heads // spec.num_kv_heads


def kv_weight_threshold(spec: ModelSpec) -> float:
    """KV 总量超过权重总量所需的"并发数 × 上下文长度"阈值（token 数）。

    推导：KV 总量 = N × ctx × kv_bytes_per_token，权重 = weights_bytes。
    两者相等时解出 N × ctx = weights_bytes ÷ kv_bytes_per_token。
    """
    return spec.weights_bytes(spec.weight_bits) / kv_bytes_per_token(spec)


def check_kv_bytes_formula() -> None:
    """判断：每 token 的 KV 字节数是精确展开，不是拍脑袋估算。"""
    section("判断一：KV 显存账是精确公式")
    names = ("llama2-13b", "qwen2.5-7b", "llama3.1-8b", "qwen2.5-32b", "llama3.1-70b")
    rows: list[list[str]] = []
    for name in names:
        spec = model(name)
        per_token = kv_bytes_per_token(spec)
        # 公式的每一项都来自模型 config，不允许另起一套口径
        expanded = (
            2
            * spec.num_layers
            * spec.num_kv_heads
            * spec.head_dim
            * 2
        )
        ensure(
            per_token == expanded,
            f"{name} 的每 token KV 字节数应当等于 2×层×KV头×head_dim×2，"
            f"实际 {per_token} ≠ {expanded}",
        )
        rows.append(
            [
                name,
                str(spec.num_layers),
                str(spec.num_kv_heads),
                str(spec.head_dim),
                f"{per_token:,}",
                fmt_bytes(per_token),
            ]
        )

    print_table(
        ["模型", "层数", "KV 头数", "head_dim", "每 token 字节", "折合"],
        rows,
    )

    spec = model("llama3.1-8b")
    per_token = kv_bytes_per_token(spec)
    ensure(
        kv_gib(spec, 131072) * (1024**3) == per_token * 131072,
        "KV 总量应当等于每 token 字节数 × token 数",
    )
    rule(
        f"llama3.1-8b：2 × 32 层 × 8 KV 头 × 128 head_dim × 2 字节 = "
        f"{per_token:,} 字节/token = {fmt_bytes(per_token)}/token"
    )
    rule(
        f"上下文 128K 时，单条序列的 KV = {fmt_gib(kv_gib(spec, 131072))}"
    )


def check_gqa_shrinks_kv() -> None:
    """判断：GQA 让 KV 显存按 group_size 成倍下降。"""
    section("判断二：GQA 是 KV 显存的主杠杆")
    base = model("llama3.1-8b")

    # 同一个模型，只改 KV 头数，把"变体的功劳"和"模型本身的差别"分开
    variants = [
        ("MHA（32 个 KV 头）", replace(base, num_kv_heads=base.num_heads)),
        ("GQA-8（8 个 KV 头）", replace(base, num_kv_heads=8)),
        ("GQA-4（4 个 KV 头）", replace(base, num_kv_heads=4)),
        ("MQA（1 个 KV 头）", replace(base, num_kv_heads=1)),
    ]
    rows: list[list[str]] = []
    base_bytes = kv_bytes_per_token(base) / base.num_kv_heads
    for label, variant in variants:
        per_token = kv_bytes_per_token(variant)
        ensure(
            per_token == base_bytes * variant.num_kv_heads,
            f"{label} 的每 token KV 字节数应当与 KV 头数严格成正比",
        )
        rows.append(
            [
                label,
                str(variant.num_kv_heads),
                fmt_bytes(per_token),
                f"{base.num_heads / variant.num_kv_heads:.0f}×",
                fmt_gib(kv_gib(variant, 32768)),
            ]
        )

    print_table(
        ["注意力变体", "KV 头数", "每 token 字节", "相对 MHA 省", "32K 上下文单条 KV"],
        rows,
    )

    mha = variants[0][1]
    gqa8 = variants[1][1]
    ensure(
        kv_bytes_per_token(mha) == kv_bytes_per_token(gqa8) * 4,
        "从 MHA 换成 GQA-8，KV 字节数应当正好降到 1/4",
    )
    ensure(
        kv_gib(mha, 32768) == kv_gib(gqa8, 32768) * 4,
        "KV 总量也应当同步降到 1/4",
    )

    # 真实模型的对照：同样带 GQA 的模型，KV 阈值高得多
    mha_model = model("llama2-13b")
    ensure(
        kv_weight_threshold(mha_model) < kv_weight_threshold(base) / 3,
        f"MHA 模型 llama2-13b 的 KV 阈值 {kv_weight_threshold(mha_model):.0f} "
        f"应当明显低于 GQA-4 模型 llama3.1-8b 的 {kv_weight_threshold(base):.0f}",
    )
    rule(
        f"llama2-13b（MHA，group=1）：{fmt_bytes(kv_bytes_per_token(mha_model))}/token；"
        f"llama3.1-8b（GQA-4）：{fmt_bytes(kv_bytes_per_token(base))}/token"
    )
    rule(
        "同样的注意力头数下，KV 头数每减半，KV 显存就减半——质量代价却远小于这个幅度"
    )


def check_kv_beats_weights_at_long_context() -> None:
    """判断：长上下文服务的瓶颈是 KV，不是权重。"""
    section("判断三：长上下文服务的瓶颈是 KV 而不是权重")
    device = gpu(DEVICE)
    spec = model("llama3.1-8b")

    rows: list[KVRow] = []
    for ctx in (2048, 8192, 32768, 131072):
        per_seq = kv_gib(spec, ctx)
        free = (
            device.memory_gib
            - weight_gib(spec)
            - ACTIVATION_GIB
            - RUNTIME_GIB
        )
        concurrent = int(free / per_seq)
        budget = plan_memory(
            spec,
            device,
            kv_tokens=concurrent * ctx,
            activation_gib=ACTIVATION_GIB,
            runtime_gib=RUNTIME_GIB,
        )
        ensure(
            budget.fits(),
            f"llama3.1-8b 在 {ctx} 上下文下装 {concurrent} 条序列应当装得下",
        )
        rows.append(
            KVRow(
                name=spec.name,
                kv_bytes_per_token=kv_bytes_per_token(spec),
                group_size=group_size(spec),
                context_tokens=ctx,
                kv_gib=budget.kv_cache_gib,
                concurrent=concurrent,
                kv_share=budget.kv_share,
            )
        )

    print_table(
        ["上下文", "单条 KV", "可并发序列", "KV 总量", "权重", "KV 占已用显存"],
        [
            [
                f"{row.context_tokens:,}",
                fmt_gib(kv_gib(spec, row.context_tokens)),
                str(row.concurrent),
                fmt_gib(row.kv_gib),
                fmt_gib(weight_gib(spec)),
                f"{row.kv_share * 100:.1f}%",
            ]
            for row in rows
        ],
    )

    ensure(
        all(row.kv_gib > weight_gib(spec) for row in rows),
        "每个上下文档位下 KV 总量都应当超过权重——这正是'长上下文的瓶颈是 KV'的含义",
    )
    ensure(
        rows[-1].concurrent < rows[0].concurrent / 8,
        f"上下文从 2048 涨到 131072，并发数应当从 {rows[0].concurrent} "
        f"掉到 {rows[-1].concurrent}，掉一个数量级",
    )

    threshold = kv_weight_threshold(spec)
    rule(
        f"KV 超过权重的判定式：并发数 × 上下文长度 > {threshold:,.0f} token"
        f"（llama3.1-8b fp16 在 80GB 卡上）"
    )
    rule(
        f"32K 上下文时，只要并发超过 {threshold / 32768:.1f} 条，"
        f"KV 总量就压过权重总量——这是容量规划的第一条红线"
    )
    rule(
        "权重是固定的，KV 随上下文和并发一起长——长上下文服务先撞上的永远是 KV 这堵墙"
    )


def check_kv_quantization_tradeoff() -> None:
    """判断：KV 量化用一半显存换有限精度，INT4 就开始伤筋动骨。"""
    section("判断四：KV 量化的收益与代价")
    spec = model("qwen2.5-7b")
    per_token_fp16 = kv_bytes_per_token(spec, 2)
    per_token_int8 = kv_bytes_per_token(spec, 1)

    ensure(
        per_token_int8 * 2 == per_token_fp16,
        "KV 从 FP16 量化到 8 bit，字节数应当正好减半",
    )

    # 用确定性的伪 KV 值序列衡量误差量级（KV 向量本身接近零均值小信号）
    values = [
        math.sin(i / 9.0) * 0.8 + math.cos(i / 31.0) * 0.2 + math.sin(i * 1.7) * 0.05
        for i in range(1024)
    ]
    rows: list[list[str]] = []
    for bits in (16, 8, 4):
        result = quantize_values(values, bits, group_size=spec.head_dim)
        # 位宽直接换算成元素字节数交给 labkit 算账，不自己另写一套缩放公式
        rows.append(
            [
                f"{bits} bit",
                f"{kv_bytes_per_token(spec, bits / 8):,.0f}",
                f"{kv_gib(spec, 32768, bits / 8):.2f}",
                f"{result.max_abs_error:.5f}",
                f"{result.snr_db:.1f} dB",
            ]
        )

    print_table(
        ["KV 精度", "每 token 字节", "32K 单条 KV", "最大绝对误差", "信噪比"],
        rows,
    )

    fp16 = quantize_values(values, 16, group_size=spec.head_dim)
    int8 = quantize_values(values, 8, group_size=spec.head_dim)
    int4 = quantize_values(values, 4, group_size=spec.head_dim)

    ensure(
        int8.snr_db > fp16.snr_db - 60,
        f"8 bit 量化的信噪比 {int8.snr_db:.1f}dB 不该比 FP16 差 60dB 以上",
    )
    ensure(
        int8.snr_db > int4.snr_db,
        f"8 bit 的信噪比 {int8.snr_db:.1f}dB 应当高于 4 bit 的 {int4.snr_db:.1f}dB",
    )
    ensure(
        kv_gib(spec, 32768, 1) * 2 == kv_gib(spec, 32768, 2),
        "8 bit KV 的总量应当是 FP16 的一半",
    )
    ensure(
        kv_bytes_per_token(spec, 4 / 8) * 2 == kv_bytes_per_token(spec, 8 / 8),
        "4 bit KV 的每 token 字节数应当是 8 bit 的两倍——位宽减半，字节数减半",
    )
    rule(
        f"FP16→FP8/INT8：KV 显存直接减半（32K 上下文 {kv_gib(spec, 32768):.2f} → "
        f"{kv_gib(spec, 32768, 1):.2f} GiB），信噪比从 {fp16.snr_db:.0f}dB 掉到 "
        f"{int8.snr_db:.0f}dB"
    )
    rule(
        f"再压到 INT4 信噪比只剩 {int4.snr_db:.0f}dB——省显存到这一步，"
        "质量风险要靠实测兜底，不能只看显存账"
    )
    rule(
        f"分组大小取 {spec.head_dim}（= head_dim）时，每组正好是一个头的向量，"
        "scale 的统计范围与 KV 的真实分布对齐"
    )


def check_prefix_reuse_saves_prefill() -> None:
    """判断：相同前缀可以共享已算好的 KV，连 prefill 一起省掉。"""
    section("判断五：前缀复用把重复的 prefill 直接砍掉")
    device = gpu(DEVICE)
    spec = model("qwen2.5-7b")

    prefix_tokens = list(range(1024))
    digest = prefix_hash(prefix_tokens, BLOCK_SIZE)
    same = prefix_hash(list(range(1024)), BLOCK_SIZE)
    other = prefix_hash(list(range(1000)) + [7, 7, 7, 7], BLOCK_SIZE)

    ensure(
        digest == same,
        "同一段前缀必须算出同一个哈希，否则前缀缓存永远命不中",
    )
    ensure(
        digest != other,
        "不同前缀必须算出不同的哈希，否则会命中错误的 KV",
    )

    cache = PagedKVCache(block_size=BLOCK_SIZE)
    prefix_blocks: list[int] = []
    remaining = 1024
    while remaining > 0:
        block_id = cache.allocate()
        cache.append_tokens(block_id, min(BLOCK_SIZE, remaining))
        remaining -= BLOCK_SIZE
        prefix_blocks.append(block_id)
    cache.cache_prefix(digest, prefix_blocks)

    hit = cache.lookup_prefix(digest)
    ensure(
        hit is not None and list(hit) == prefix_blocks,
        "前缀缓存应当能查回之前登记的块号",
    )
    ensure(
        cache.lookup_prefix(other) is None,
        "没有登记过的前缀不应命中",
    )

    # 第二条序列挂上同一个前缀：块被共享，引用计数 +1，而不是重新分配
    for block_id in prefix_blocks:
        cache.refcount[block_id] += 1
    blocks_after_share = cache.total_blocks()
    ensure(
        blocks_after_share == len(prefix_blocks),
        f"共享前缀后总块数应当仍是 {len(prefix_blocks)}，实际 {blocks_after_share}",
    )

    full = prefill_time(device, spec, 2048)
    tail = prefill_time(device, spec, 1024)
    saved_ms = (full - tail) * 1000
    saved_ratio = (full - tail) / full

    print_table(
        ["场景", "prefill 耗时", "节省"],
        [
            ["2048 token 全量 prefill", fmt_ms(full), "—"],
            ["命中 1024 前缀，只算后 1024", fmt_ms(tail), f"{saved_ms:.2f}ms（{saved_ratio * 100:.1f}%）"],
        ],
    )

    ensure(
        tail < full,
        "命中前缀后 prefill 应当比全量 prefill 更快",
    )
    ensure(
        0.4 < saved_ratio < 0.6,
        f"共享一半前缀应当省下约一半 prefill 时间，实际 {saved_ratio * 100:.1f}%",
    )
    ensure(
        cache.total_blocks() == len(prefix_blocks),
        "第二条序列挂上共享前缀后不该新增块",
    )
    rule(
        f"共享 1024 token 前缀：prefill 从 {fmt_ms(full)} 降到 {fmt_ms(tail)}"
        f"（省 {saved_ratio * 100:.1f}%），"
        f"每多一条序列还省下 {fmt_gib(kv_gib(spec, 1024))} 的 KV 显存"
    )
    rule(
        "前缀复用的真正价值不是省显存，是省掉那段重复的 prefill 计算——"
        "共享系统提示词、多轮对话历史时，这一项直接决定首字时延"
    )


def run() -> None:
    """跑第 5 章实验。"""
    banner("第 5 章 · KV Cache：显存账与复用")
    check_kv_bytes_formula()
    check_gqa_shrinks_kv()
    check_kv_beats_weights_at_long_context()
    check_kv_quantization_tradeoff()
    check_prefix_reuse_saves_prefill()
    rule(
        "KV 的账是精确公式；GQA 按 group_size 省显存；长上下文先撞 KV 这堵墙；"
        "量化换显存、前缀复用省 prefill"
    )
    print("\n✓ 第 5 章通过")
