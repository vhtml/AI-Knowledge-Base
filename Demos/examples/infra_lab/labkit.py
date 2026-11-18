"""AI Infra 推理服务教程的公共工具层——纯 Python，不依赖任何推理框架。

这个模块解决什么问题：推理服务这套课反复要用到同一批"底层设施"——GPU 规格、
模型规格、显存账本、FLOPs 估算、Roofline 上界、分页 KV 缓存、量化误差、
推测解码收益、连续批处理引擎模拟。这些东西每章都要用，但每章的重心不同，
如果各章各写一遍，不仅会算出互相矛盾的数，读者也会分不清"哪个是本课口径"。

所以约定：**所有数字只在这里算一次**。各章实验只写当天的教学重点，
需要算账就调用本模块，绝不自己另起一套公式。

三条贯穿全课的口径：

1. **算力口径**——所有 TFLOPS 取**稠密**值（不含结构化稀疏），
   所有 GPU 都按**单卡**计算；跨卡能力在第 17 章单独讲。
2. **显存口径**——1 GiB = 2^30 字节。权重按量化位宽算，
   KV cache 按"每 token 每层"精确展开，不做粗略近似。
3. **性能口径**——所有时延都用**解析模型**估算（Roofline + MFU 系数），
   不跑真实 GPU。所以所有实验离线、确定性、可复现。

刻意不引入 vLLM / SGLang / TensorRT-LLM：第 18 章才在**文字**里做机制对照，
实验代码必须自己实现一遍，否则读者永远看不见"框架替我做了什么"。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "BYTES_PER_GIB",
    "DPCTokens",
    "EngineSimulator",
    "GPUSpec",
    "MemoryBudget",
    "ModelSpec",
    "PagedKVCache",
    "QuantResult",
    "Request",
    "Completion",
    "RooflinePoint",
    "SchedulePolicy",
    "approx",
    "banner",
    "decode_step_time",
    "ensure",
    "fake_quantize",
    "fmt_gib",
    "fmt_ms",
    "gpu",
    "model",
    "plan_memory",
    "prefill_flops",
    "decode_flops",
    "attention_flops",
    "print_table",
    "quantize_values",
    "roofline",
    "rule",
    "section",
    "simulate_speculative",
    "table",
    "weight_gib",
    "kv_bytes_per_token",
    "kv_gib",
    "kv_capacity_tokens",
]

BYTES_PER_GIB = 1024**3
"""1 GiB 的字节数。全课统一用 GiB，不用 GB（十进制），避免口径漂移。"""

DPCTokens = float
"""显存计量简写：DPCTokens 其实是 token 数，只是为了类型可读性。"""


# ---------------------------------------------------------------- 输出与断言


def banner(title: str) -> None:
    """打印章节标题。"""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def section(text: str) -> None:
    """打印小节标题。"""
    print()
    print(f"--- {text} ---")


def rule(text: str = "") -> None:
    """打印一条提示行，可带一句说明。"""
    print(f"\n· {text}" if text else "")


def ensure(condition: bool, message: str) -> None:
    """断言辅助：条件不成立就抛 :class:`AssertionError`。

    报错信息用中文写清"期望什么、实际什么"，比裸 ``assert`` 好读。
    """
    if not condition:
        raise AssertionError(message)


def approx(actual: float, expected: float, tol: float = 1e-6) -> bool:
    """浮点近似比较。

    单独抽出来是因为推理服务的性能模型全是浮点估算，
    直接用 ``==`` 比较会让实验变成"比机器精度还严"的脆测试。
    """
    return abs(actual - expected) <= tol * max(1.0, abs(expected))


def fmt_ms(seconds: float) -> str:
    """把秒格式化成毫秒字符串，保留两位小数。"""
    return f"{seconds * 1000:.2f}ms"


def fmt_gib(value: float) -> str:
    """把字节数格式化成 GiB 字符串。"""
    return f"{value / BYTES_PER_GIB:.2f}GiB"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """把表头和行渲染成对齐的等宽表格文本。"""
    headers = [str(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in body:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))
    lines = [
        "  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  " + "-+-".join("-" * w for w in widths),
    ]
    for row in body:
        lines.append("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def print_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """打印表格。"""
    print(table(headers, rows))


# ---------------------------------------------------------------- 硬件规格


@dataclass(frozen=True)
class GPUSpec:
    """一张 GPU 的规格。

    所有带宽单位是 **GB/s**（十进制，硬件厂商标称口径），
    所有算力单位是 **TFLOPS**（稠密，不含稀疏）。

    为什么混用 GB/s 和 GiB：显存的容量厂商按 GiB 标，带宽按 GB/s 标，
    混着记会错。容量一律走 :data:`BYTES_PER_GIB` 换算。
    """

    name: str
    memory_gib: float
    """显存容量（GiB）。"""

    hbm_bandwidth_gbs: float
    """显存带宽（GB/s）。推理阶段几乎总是被这个卡住。"""

    fp16_tflops: float
    """FP16 稠密算力（TFLOPS）。"""

    fp8_tflops: float
    """FP8 稠密算力（TFLOPS）。"""

    int8_tops: float
    """INT8 算力（TOPS）。"""

    price_per_hour_usd: float
    """整机每卡每小时租金（美元）。做单位经济用。"""

    @property
    def memory_bytes(self) -> int:
        """显存字节数。"""
        return int(self.memory_gib * BYTES_PER_GIB)

    def peak_tflops(self, dtype: str = "fp16") -> float:
        """按精度取峰值算力。"""
        table = {
            "fp16": self.fp16_tflops,
            "fp8": self.fp8_tflops,
            "int8": self.int8_tops,
        }
        if dtype not in table:
            raise ValueError(f"未知精度：{dtype}，可选 {sorted(table)}")
        return table[dtype]


GPUS: dict[str, GPUSpec] = {
    "h100_sxm": GPUSpec(
        name="H100 SXM 80GB",
        memory_gib=80,
        hbm_bandwidth_gbs=3350,
        fp16_tflops=989,
        fp8_tflops=1979,
        int8_tops=1979,
        price_per_hour_usd=4.5,
    ),
    "h100_pcie": GPUSpec(
        name="H100 PCIe 80GB",
        memory_gib=80,
        hbm_bandwidth_gbs=2000,
        fp16_tflops=756,
        fp8_tflops=1513,
        int8_tops=1513,
        price_per_hour_usd=3.2,
    ),
    "a100_80g": GPUSpec(
        name="A100 80GB SXM",
        memory_gib=80,
        hbm_bandwidth_gbs=2039,
        fp16_tflops=312,
        fp8_tflops=624,
        int8_tops=624,
        price_per_hour_usd=2.8,
    ),
    "a100_40g": GPUSpec(
        name="A100 40GB SXM",
        memory_gib=40,
        hbm_bandwidth_gbs=1555,
        fp16_tflops=312,
        fp8_tflops=624,
        int8_tops=624,
        price_per_hour_usd=2.2,
    ),
    "l40s": GPUSpec(
        name="L40S 48GB",
        memory_gib=48,
        hbm_bandwidth_gbs=864,
        fp16_tflops=362,
        fp8_tflops=733,
        int8_tops=733,
        price_per_hour_usd=1.6,
    ),
    "l4": GPUSpec(
        name="L4 24GB",
        memory_gib=24,
        hbm_bandwidth_gbs=300,
        fp16_tflops=121,
        fp8_tflops=242,
        int8_tops=242,
        price_per_hour_usd=0.8,
    ),
    "a10g": GPUSpec(
        name="A10G 24GB",
        memory_gib=24,
        hbm_bandwidth_gbs=600,
        fp16_tflops=125,
        fp8_tflops=250,
        int8_tops=250,
        price_per_hour_usd=1.2,
    ),
    "rtx4090": GPUSpec(
        name="RTX 4090 24GB",
        memory_gib=24,
        hbm_bandwidth_gbs=1008,
        fp16_tflops=165,
        fp8_tflops=660,
        int8_tops=660,
        price_per_hour_usd=0.9,
    ),
    "mi300x": GPUSpec(
        name="MI300X 192GB",
        memory_gib=192,
        hbm_bandwidth_gbs=5300,
        fp16_tflops=1307,
        fp8_tflops=2614,
        int8_tops=2614,
        price_per_hour_usd=6.0,
    ),
}
"""GPU 规格表。数值取自厂商公开规格的公开档位，做量级估算用。"""


def gpu(name: str = "h100_sxm") -> GPUSpec:
    """按名字取 GPU 规格。"""
    if name not in GPUS:
        raise KeyError(f"未知 GPU：{name}，可选 {sorted(GPUS)}")
    return GPUS[name]


# ---------------------------------------------------------------- 模型规格


@dataclass(frozen=True)
class ModelSpec:
    """一个开源权重的模型规格。

    ``params_b`` 计入 embedding；FLOPs 计算里 embedding 会被单独扣掉，
    因为推理时 embedding 只是一次查表，不参与每 token 的矩阵乘。
    """

    name: str
    params_b: float
    """总参数量（十亿）。"""

    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    """KV 头数（GQA）。等于 ``num_heads`` 就是 MHA，小于就是 GQA。"""

    head_dim: int
    vocab_size: int
    max_context: int
    weight_bits: int = 16
    """权重存储位宽。FP16=16，INT8=8，INT4=4。"""

    embedding_params_b: float = 0.0
    """embedding 与 lm_head 的参数量（十亿）。0 表示按经验比例估算。"""

    @property
    def non_embedding_params_b(self) -> float:
        """参与每 token 矩阵乘的参数量（十亿）。"""
        if self.embedding_params_b:
            return max(0.0, self.params_b - self.embedding_params_b)
        # 经验比例：词表 × hidden × 2（embedding + lm_head），换算成十亿
        embed = 2 * self.vocab_size * self.hidden_size / 1e9
        return max(0.0, self.params_b - embed)

    @property
    def is_gqa(self) -> bool:
        """是否是 GQA/MQA（KV 头数少于注意力头数）。"""
        return self.num_kv_heads < self.num_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """每个 token 的 KV cache 字节数。

        公式：2（K 和 V） × 层数 × KV 头数 × head_dim × 元素字节数。

        这是 KV cache 显存账的唯一口径，第 5 章整章围绕它展开。
        """
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * dtype_bytes
        )

    def weights_bytes(self, bits: int | None = None) -> int:
        """权重占用字节数。"""
        bits = self.weight_bits if bits is None else bits
        return int(self.params_b * 1e9 * bits / 8)


MODELS: dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec(
        name="Qwen2.5-7B-Instruct",
        params_b=7.62,
        num_layers=28,
        hidden_size=3584,
        num_heads=28,
        num_kv_heads=4,
        head_dim=128,
        vocab_size=151936,
        max_context=32768,
        weight_bits=16,
        embedding_params_b=1.09,
    ),
    "llama3.1-8b": ModelSpec(
        name="Llama-3.1-8B-Instruct",
        params_b=8.03,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_context=131072,
        weight_bits=16,
        embedding_params_b=1.05,
    ),
    "llama2-13b": ModelSpec(
        name="Llama-2-13B-Chat",
        params_b=13.0,
        num_layers=40,
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=40,
        head_dim=128,
        vocab_size=32000,
        max_context=4096,
        weight_bits=16,
        embedding_params_b=0.33,
    ),
    "qwen2.5-32b": ModelSpec(
        name="Qwen2.5-32B-Instruct",
        params_b=32.8,
        num_layers=64,
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=152064,
        max_context=32768,
        weight_bits=16,
        embedding_params_b=1.56,
    ),
    "llama3.1-70b": ModelSpec(
        name="Llama-3.1-70B-Instruct",
        params_b=70.6,
        num_layers=80,
        hidden_size=8192,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_context=131072,
        weight_bits=16,
        embedding_params_b=2.10,
    ),
    "mixtral-8x7b": ModelSpec(
        name="Mixtral-8x7B-Instruct",
        params_b=46.7,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=32000,
        max_context=32768,
        weight_bits=16,
        embedding_params_b=0.26,
    ),
    "deepseek-v3": ModelSpec(
        name="DeepSeek-V3",
        params_b=671,
        num_layers=61,
        hidden_size=7168,
        num_heads=128,
        num_kv_heads=128,
        head_dim=128,
        vocab_size=129280,
        max_context=131072,
        weight_bits=8,
        embedding_params_b=1.85,
    ),
}
"""模型规格表。数值取自各模型公开的 config.json。"""


def model(name: str = "qwen2.5-7b") -> ModelSpec:
    """按名字取模型规格。"""
    if name not in MODELS:
        raise KeyError(f"未知模型：{name}，可选 {sorted(MODELS)}")
    return MODELS[name]


# ---------------------------------------------------------------- 显存账本


def weight_gib(spec: ModelSpec, bits: int | None = None) -> float:
    """权重占用的 GiB。"""
    return spec.weights_bytes(bits) / BYTES_PER_GIB


def kv_bytes_per_token(spec: ModelSpec, dtype_bytes: int = 2) -> int:
    """每个 token 的 KV cache 字节数。"""
    return spec.kv_bytes_per_token(dtype_bytes)


def kv_gib(spec: ModelSpec, tokens: int, dtype_bytes: int = 2) -> float:
    """给定 token 数的 KV cache 占用 GiB。"""
    return tokens * spec.kv_bytes_per_token(dtype_bytes) / BYTES_PER_GIB


def kv_capacity_tokens(
    spec: ModelSpec,
    budget_bytes: float,
    dtype_bytes: int = 2,
) -> int:
    """给定显存预算能装多少 token 的 KV cache。"""
    per_token = spec.kv_bytes_per_token(dtype_bytes)
    if per_token <= 0:
        return 0
    return int(budget_bytes // per_token)


@dataclass(frozen=True)
class MemoryBudget:
    """一张卡上的显存分配账本。

    这四项之和必须小于显存容量，否则就是 OOM。
    第 2 章和第 16 章的容量规划全靠这张表。
    """

    weights_gib: float
    kv_cache_gib: float
    activations_gib: float
    runtime_gib: float
    capacity_gib: float
    """这张卡的显存容量。"""

    @property
    def used_gib(self) -> float:
        """已占用显存。"""
        return self.weights_gib + self.kv_cache_gib + self.activations_gib + self.runtime_gib

    @property
    def free_gib(self) -> float:
        """剩余显存。"""
        return self.capacity_gib - self.used_gib

    @property
    def kv_share(self) -> float:
        """KV cache 占已用显存的比例。"""
        used = self.used_gib
        return self.kv_cache_gib / used if used > 0 else 0.0

    def fits(self) -> bool:
        """是否装得下。"""
        return self.used_gib <= self.capacity_gib


def plan_memory(
    spec: ModelSpec,
    device: GPUSpec,
    *,
    kv_tokens: int = 0,
    weight_bits: int | None = None,
    activation_gib: float = 1.0,
    runtime_gib: float = 2.0,
    kv_dtype_bytes: int = 2,
) -> MemoryBudget:
    """算一张卡上的显存账本。

    ``activation_gib`` 是推理时的临时激活（中间的 hidden state、中间结果），
    ``runtime_gib`` 是框架/CUDA context/通信缓冲这类固定开销。
    """
    return MemoryBudget(
        weights_gib=weight_gib(spec, weight_bits),
        kv_cache_gib=kv_gib(spec, kv_tokens, kv_dtype_bytes),
        activations_gib=activation_gib,
        runtime_gib=runtime_gib,
        capacity_gib=device.memory_gib,
    )


# ---------------------------------------------------------------- FLOPs 估算


def prefill_flops(spec: ModelSpec, prompt_tokens: int) -> float:
    """prefill 阶段的浮点运算量。

    拆成两部分：每个 token 过一次权重的 GEMM，加上注意力矩阵本身。

    - GEMM：2 × 非 embedding 参数量 × token 数
    - 注意力：因果注意力下约为 2 × 层数 × hidden_size × s²
      （每层每个 token 对历史做 QK^T 和 AV，各 2×d 的代价）

    返回 FLOPs（不是 FLOPS）。
    """
    if prompt_tokens <= 0:
        return 0.0
    gemm = 2.0 * spec.non_embedding_params_b * 1e9 * prompt_tokens
    attn = 2.0 * spec.num_layers * spec.hidden_size * prompt_tokens**2
    return gemm + attn


def decode_flops(spec: ModelSpec, context_tokens: int) -> float:
    """decode 一个 token 时的浮点运算量（上下文长度为 ``context_tokens``）。"""
    if context_tokens <= 0:
        return 0.0
    gemm = 2.0 * spec.non_embedding_params_b * 1e9
    attn = 2.0 * 2.0 * spec.num_layers * spec.hidden_size * context_tokens
    return gemm + attn


def attention_flops(spec: ModelSpec, prompt_len: int, output_len: int) -> float:
    """整段生成的注意力运算量：prefill 一次 + 每步 decode 对全部历史的注意力。

    返回 FLOPs。用于分离"注意力开销占比"，这是第 3、4 章的关键分解。
    """
    if prompt_len <= 0:
        return 0.0
    total = 2.0 * spec.num_layers * spec.hidden_size * prompt_len**2
    for step in range(output_len):
        ctx = prompt_len + step
        total += 2.0 * 2.0 * spec.num_layers * spec.hidden_size * ctx
    return total


# ---------------------------------------------------------------- Roofline


@dataclass(frozen=True)
class RooflinePoint:
    """Roofline 图上的一个工作点。"""

    arithmetic_intensity: float
    """算术强度：FLOPs / 访存字节。"""

    bound: Literal["compute", "memory"]
    """受限于算力还是带宽。"""

    peak_tflops: float
    bandwidth_gbs: float
    dtype: str

    @property
    def peak_flops_per_s(self) -> float:
        """峰值浮点吞吐（FLOPs/s）。"""
        return self.peak_tflops * 1e12

    @property
    def attainable_tflops(self) -> float:
        """该工作点上的可达算力（FLOPs/s）。"""
        if self.bound == "compute":
            return self.peak_flops_per_s
        return self.bandwidth_gbs * 1e9 * self.arithmetic_intensity

    @property
    def ridge_intensity(self) -> float:
        """拐点算术强度：峰值算力 ÷ 峰值带宽。"""
        return self.peak_flops_per_s / (self.bandwidth_gbs * 1e9)


def roofline(
    device: GPUSpec,
    arithmetic_intensity: float,
    dtype: str = "fp16",
) -> RooflinePoint:
    """算一个工作点在 Roofline 图上的位置。"""
    peak = device.peak_tflops(dtype)
    bound: Literal["compute", "memory"] = (
        "compute" if arithmetic_intensity >= peak * 1e12 / (device.hbm_bandwidth_gbs * 1e9) else "memory"
    )
    return RooflinePoint(
        arithmetic_intensity=arithmetic_intensity,
        bound=bound,
        peak_tflops=peak,
        bandwidth_gbs=device.hbm_bandwidth_gbs,
        dtype=dtype,
    )


# ---------------------------------------------------------------- 时延估算


def decode_step_time(
    device: GPUSpec,
    spec: ModelSpec,
    batch_size: int,
    *,
    dtype: str = "fp16",
    mfu: float = 0.5,
    weight_bits: int | None = None,
    context_tokens: int = 0,
) -> float:
    """一次 decode 步（batch 内所有序列各出一个 token）的耗时（秒）。

    推理的 decode 阶段是**访存瓶颈**：每个 decode 步都要把整套权重从 HBM 读一遍，
    算力用不满。所以这里的模型是"读权重 + 读 KV"两个带宽项。

    - 权重读取：整批共享，batch 越大越摊薄
    - KV 读取：随 batch 和上下文长度线性增长

    ``mfu`` 是"实际可达带宽占峰值带宽的比例"，工程上单卡批量推理
    通常在 40%~70% 之间。
    """
    if batch_size <= 0:
        return 0.0
    weight_bytes = spec.weights_bytes(weight_bits)
    kv_bytes = batch_size * spec.kv_bytes_per_token(2) * max(1, context_tokens)
    total_bytes = weight_bytes + kv_bytes
    effective_bandwidth = device.hbm_bandwidth_gbs * 1e9 * mfu
    if effective_bandwidth <= 0:
        return 0.0
    return total_bytes / effective_bandwidth


def prefill_time(
    device: GPUSpec,
    spec: ModelSpec,
    tokens: int,
    *,
    dtype: str = "fp16",
    mfu: float = 0.5,
    weight_bits: int | None = None,
) -> float:
    """prefill 一段 prompt 的耗时（秒）。

    prefill 是**算力瓶颈**：token 多，权重被反复复用，算术强度高。
    """
    if tokens <= 0:
        return 0.0
    flops = prefill_flops(spec, tokens)
    peak = device.peak_tflops(dtype) * 1e12 * mfu
    if peak <= 0:
        return 0.0
    # prefill 同时要读一次权重（至少一遍），算上这个下限
    weight_bytes = spec.weights_bytes(weight_bits)
    memory_time = weight_bytes / (device.hbm_bandwidth_gbs * 1e9 * mfu)
    return max(flops / peak, memory_time)


# ---------------------------------------------------------------- 分页 KV 缓存


@dataclass
class PagedKVCache:
    """分页 KV 缓存：块表 + 引用计数 + 前缀缓存。

    这是第 5、6、12 章共用的数据结构，对应真实推理引擎里的
    "PagedAttention + prefix caching" 的核心机制。

    为什么要分页：传统做法给每条序列**预留**最大长度的连续显存，
    实际用不到的那部分全浪费。分页把 KV 切成固定大小的块按需分配，
    浪费只剩"最后一块没填满"的部分。

    这里只模拟**元数据与分配**，不存真实向量——存的是每块的 token 数。
    """

    block_size: int = 16
    """每块容纳的 token 数。真实引擎常见 16/32/64。"""

    blocks: dict[int, int] = field(default_factory=dict)
    """块号 → 已填充 token 数。"""

    refcount: dict[int, int] = field(default_factory=dict)
    """块号 → 引用计数。多个序列共享前缀块时靠它计数。"""

    free_blocks: deque[int] = field(default_factory=deque)
    """空闲块队列。"""

    next_block_id: int = 0
    """下一个可分配的块号。"""

    prefix_cache: dict[tuple[int, ...], list[int]] = field(default_factory=dict)
    """前缀哈希 → 块号列表。用于 prefix caching。"""

    def allocate(self) -> int:
        """分配一个新块。"""
        if self.free_blocks:
            block_id = self.free_blocks.popleft()
        else:
            block_id = self.next_block_id
            self.next_block_id += 1
        self.blocks[block_id] = 0
        self.refcount[block_id] = 0
        return block_id

    def free(self, block_id: int) -> None:
        """释放一个块（引用计数减到 0 时真正回收）。"""
        count = self.refcount.get(block_id, 0)
        if count <= 1:
            self.refcount[block_id] = 0
            self.blocks[block_id] = 0
            self.free_blocks.append(block_id)
        else:
            self.refcount[block_id] = count - 1

    def append_tokens(self, block_id: int, count: int) -> None:
        """往块里写入 token。"""
        self.blocks[block_id] = self.blocks.get(block_id, 0) + count

    def total_blocks(self) -> int:
        """已分配块数（含空闲）。"""
        return len(self.blocks)

    def used_blocks(self) -> int:
        """正在使用的块数。"""
        return sum(1 for used in self.blocks.values() if used > 0)

    def fragmentation(self) -> float:
        """碎片率：已分配但没填满的块占已分配块的比例。"""
        allocated = self.total_blocks()
        if allocated == 0:
            return 0.0
        wasted = sum(block_size - used for used in self.blocks.values() if used < block_size)
        return wasted / (allocated * block_size)

    def cache_prefix(self, prefix_hash: tuple[int, ...], block_ids: Sequence[int]) -> None:
        """把一段前缀登记进前缀缓存。"""
        self.prefix_cache[prefix_hash] = list(block_ids)

    def lookup_prefix(self, prefix_hash: tuple[int, ...]) -> list[int] | None:
        """查前缀缓存，命中返回块号列表。"""
        return self.prefix_cache.get(prefix_hash)


def prefix_hash(tokens: Sequence[int], granularity: int = 16) -> tuple[int, ...]:
    """按块粒度给 token 序列算一个可复现的哈希。

    用滚动多项式哈希（模 2^61-1 的简化版），只用于教程模拟，
    不追求抗碰撞——真实引擎用更严谨的哈希。
    """
    blocks = [
        tuple(tokens[i : i + granularity])
        for i in range(0, len(tokens), granularity)
    ]
    acc = 0
    for block in blocks:
        block_acc = 0
        for token in block:
            block_acc = (block_acc * 131 + token + 1) % (2**61 - 1)
        acc = (acc * 1000003 + block_acc) % (2**61 - 1)
    return (acc, len(blocks))


# ---------------------------------------------------------------- 量化模拟


@dataclass(frozen=True)
class QuantResult:
    """一次伪量化实验的结果。"""

    bits: int
    group_size: int
    max_abs_error: float
    mean_abs_error: float
    relative_error: float
    """相对误差（均方根 / 信号均方根），也就是常说的量化信噪比倒数。"""

    snr_db: float
    """信噪比（dB）。越高越好。"""

    values: list[float]
    """反量化后的值，用来和原始值对比。"""


def quantize_values(
    values: Sequence[float],
    bits: int,
    group_size: int = 128,
) -> QuantResult:
    """模拟分组对称量化（round-to-nearest）。

    真实引擎的量化（AWQ/GPTQ/FP8）比这个复杂得多，
    但**误差量级**和"分组大小决定精度"这个结论是成立的，
    足够支撑第 9 章的判断。
    """
    if bits < 2:
        raise ValueError("量化位宽至少为 2")
    levels = 2 ** (bits - 1)
    scale_max = float(levels - 1)
    dequantized: list[float] = []
    errors: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        abs_max = max((abs(v) for v in group), default=0.0)
        if abs_max == 0.0:
            dequantized.extend(group)
            errors.extend(0.0 for _ in group)
            continue
        scale = abs_max / scale_max
        for value in group:
            quantized = int(round(value / scale))
            clamped = max(-levels, min(levels - 1, quantized))
            restored = clamped * scale
            dequantized.append(restored)
            errors.append(abs(restored - value))

    if not errors:
        return QuantResult(bits, group_size, 0.0, 0.0, 0.0, float("inf"), [])

    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    signal_rms = math.sqrt(sum(v * v for v in values) / len(values))
    noise_rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    rel = noise_rms / signal_rms if signal_rms > 0 else 0.0
    snr = 20 * math.log10(1 / rel) if rel > 0 else float("inf")
    return QuantResult(bits, group_size, max_err, mean_err, rel, snr, dequantized)


def fake_quantize(
    values: Sequence[float],
    bits: int,
    group_size: int = 128,
) -> list[float]:
    """只返回反量化后的值，方便在仿真里替换权重。"""
    return quantize_values(values, bits, group_size).values


# ---------------------------------------------------------------- 推测解码


def simulate_speculative(
    acceptance_rate: float,
    *,
    draft_cost: float = 1.0,
    target_cost: float = 5.0,
    draft_length: int = 4,
    steps: int = 1000,
) -> dict[str, float]:
    """估算推测解码的时延收益。

    模型：每一轮先由 draft 模型猜 ``draft_length`` 个 token，
    再由 target 模型**并行**验证这 ``draft_length`` 个位置。
    期望接受数量 = ``draft_length × acceptance_rate``。

    - 原始：``steps`` 个 token，耗时 ``steps × target_cost``
    - 推测：每轮产出的有效 token 数 ≈ 1 + draft_length × acceptance_rate
      （第一个位置总是接受），每轮耗时 ≈ draft_cost + target_cost

    返回时延、加速比、每轮有效 token 数等指标。
    """
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError("接受率必须在 0~1 之间")
    if draft_length < 1:
        raise ValueError("draft 长度至少为 1")

    accepted_per_round = 1.0 + draft_length * acceptance_rate
    rounds = steps / accepted_per_round
    baseline_time = steps * target_cost
    speculative_time = rounds * (draft_cost + target_cost)
    return {
        "baseline_time": baseline_time,
        "speculative_time": speculative_time,
        "speedup": baseline_time / speculative_time if speculative_time > 0 else 0.0,
        "accepted_per_round": accepted_per_round,
        "rounds": rounds,
        "tokens_per_verify": accepted_per_round,
    }


# ---------------------------------------------------------------- 请求与引擎模拟


@dataclass
class Request:
    """一条推理请求。"""

    req_id: int
    arrival_s: float
    """到达时刻（秒）。"""

    prompt_tokens: int
    max_output_tokens: int
    tenant: str = "default"
    priority: int = 0
    """优先级，数字越小越优先。"""

    deadline_s: float | None = None
    """SLA 截止时刻，超过就算违约。"""

    actual_output_tokens: int | None = None
    """实际输出长度。``None`` 表示按 ``max_output_tokens`` 算。"""

    def output_tokens(self) -> int:
        """本条请求实际产出的 token 数。"""
        return self.actual_output_tokens or self.max_output_tokens

    def ttft_deadline(self) -> float | None:
        """TTFT 的截止时刻（到达时刻 + 1 秒，服务等级约定里的常规口径）。"""
        if self.deadline_s is None:
            return None
        return self.arrival_s + 1.0


@dataclass
class Completion:
    """一条请求的完成情况。"""

    request: Request
    enqueue_s: float
    """进入引擎的时刻（含排队）。"""

    prefill_end_s: float
    """prefill 结束、第一个 token 产出的时刻。"""

    finish_s: float
    """全部产出完成的时刻。"""

    ttft_s: float
    """首 token 时延：从到达算起。"""

    tpot_s: float
    """每输出 token 时延：首 token 之后的平均间隔。"""

    @property
    def total_time_s(self) -> float:
        """端到端时延。"""
        return self.finish_s - self.request.arrival_s

    @property
    def queue_time_s(self) -> float:
        """排队时延。"""
        return self.enqueue_s - self.request.arrival_s

    @property
    def output_tokens(self) -> int:
        """产出 token 数。"""
        return self.request.output_tokens()

    def missed_ttft_deadline(self) -> bool:
        """是否违反 TTFT 约定。"""
        deadline = self.request.ttft_deadline()
        if deadline is None:
            return False
        return self.prefill_end_s > deadline


SchedulePolicy = Literal["fcfs", "sjf", "priority"]
"""调度策略：先到先服务 / 短作业优先 / 优先级。"""


class EngineSimulator:
    """连续批处理引擎的解析模拟器。

    **它模拟什么**：一张 GPU 上跑一个 OpenAI 兼容服务的核心调度循环。
    请求到达后进入队列，引擎按策略选请求做 prefill，
    prefill 完的序列进入运行批，每轮每个序列产出一个 token。

    **它怎么算时间**：这是全课最重要的近似，必须说清楚——

    - 一个 batch 内的所有序列**并行**跑，所以一轮的时间 = 这一轮里
      **最慢的那个序列**的时间；
    - 不同序列的 decode 步耗时不同（上下文长度不同 → KV 读取量不同），
      所以取 max；
    - prefill 与 decode **不重叠**（真实引擎有 chunked prefill 做重叠，
      这里默认不重叠，第 7 章会讲重叠的收益）；
    - 每轮结束后，立刻从队列补入新请求（连续批处理的核心）。

    **它不模拟什么**：不模拟 NCCL 通信、不模拟真实 kernel 效率、
    不模拟多卡切分、不模拟抢占重算。这些不影响"排队—批处理—显存"这条主线。
    """

    def __init__(
        self,
        device: GPUSpec,
        spec: ModelSpec,
        *,
        max_batch: int = 32,
        kv_token_budget: int = 100_000,
        mfu: float = 0.5,
        weight_bits: int | None = None,
        policy: SchedulePolicy = "fcfs",
        chunked_prefill: bool = False,
        chunk_size: int = 2048,
    ) -> None:
        self.device = device
        self.spec = spec
        self.max_batch = max_batch
        self.kv_token_budget = kv_token_budget
        self.mfu = mfu
        self.weight_bits = weight_bits
        self.policy: SchedulePolicy = policy
        self.chunked_prefill = chunked_prefill
        self.chunk_size = chunk_size

        self.now = 0.0
        self.running: list[dict[str, Any]] = []
        self.completions: list[Completion] = []
        self._pending: list[Request] = []

    # -- 队列策略 ---------------------------------------------------------

    def _select_next(self) -> Request | None:
        """按当前策略从队列里挑下一个要 prefill 的请求。"""
        if not self._pending:
            return None
        if self.policy == "sjf":
            # 短作业优先：按 prompt + max_output 的估计代价排序
            self._pending.sort(key=lambda r: r.prompt_tokens + r.max_output_tokens)
        elif self.policy == "priority":
            # 优先级：先按 priority，再按到达时间
            self._pending.sort(key=lambda r: (r.priority, r.arrival_s))
        else:
            self._pending.sort(key=lambda r: r.arrival_s)
        return self._pending.pop(0)

    def _kv_used_tokens(self) -> int:
        """当前运行批占用的 KV token 数。"""
        total = 0
        for seq in self.running:
            total += seq["context"] + seq["produced"]
        return total

    def _can_admit(self, request: Request) -> bool:
        """能不能装得下：batch 槽位 + KV 预算。"""
        if len(self.running) >= self.max_batch:
            return False
        need = request.prompt_tokens + request.max_output_tokens
        return self._kv_used_tokens() + need <= self.kv_token_budget

    # -- 主循环 -----------------------------------------------------------

    def run(self, requests: Sequence[Request], *, horizon: float = 600.0) -> list[Completion]:
        """跑完一批请求，返回完成情况（按完成时间排序）。"""
        self.now = 0.0
        self.running = []
        self.completions = []
        self._pending = list(requests)

        arrivals = sorted(self._pending, key=lambda r: r.arrival_s)
        index = 0

        while index < len(arrivals) or self._pending or self.running:
            if self.now > horizon:
                break

            # 1. 把已经到达的请求放进队列
            while index < len(arrivals) and arrivals[index].arrival_s <= self.now:
                self._pending.append(arrivals[index])
                index += 1

            # 2. 引擎空闲且队列有请求 → 做一次 prefill
            if not self.running and self._pending:
                request = self._select_next()
                if request is None:
                    break
                if not self._can_admit(request):
                    # 装不下：放回队列，等有序列跑完腾位置
                    self._pending.insert(0, request)
                    if not self.running:
                        # 没有运行中的序列却装不下——只能强行腾挪，放宽限制
                        self.kv_token_budget = max(
                            self.kv_token_budget,
                            self._kv_used_tokens() + request.prompt_tokens + request.max_output_tokens,
                        )
                        continue
                    break
                self._admit(request)
                continue

            if not self.running:
                # 队列空、运行批空，但还有请求没到达 → 跳到下一个到达时刻
                if index < len(arrivals):
                    self.now = max(self.now, arrivals[index].arrival_s)
                    continue
                break

            # 3. 跑一轮 decode
            self._decode_step()

        self.completions.sort(key=lambda c: c.finish_s)
        return self.completions

    def _admit(self, request: Request) -> None:
        """prefill 一个请求并把它放进运行批。"""
        enqueue_s = self.now
        if self.chunked_prefill and request.prompt_tokens > self.chunk_size:
            # 分块 prefill：每块单独占一轮引擎时间
            chunks = math.ceil(request.prompt_tokens / self.chunk_size)
            per_chunk = prefill_time(
                self.device,
                self.spec,
                self.chunk_size,
                mfu=self.mfu,
                weight_bits=self.weight_bits,
            )
            self.now += per_chunk * chunks
        else:
            self.now += prefill_time(
                self.device,
                self.spec,
                request.prompt_tokens,
                mfu=self.mfu,
                weight_bits=self.weight_bits,
            )

        self.running.append(
            {
                "request": request,
                "enqueue_s": enqueue_s,
                "prefill_end_s": self.now,
                "context": request.prompt_tokens,
                "produced": 0,
            }
        )

    def _decode_step(self) -> None:
        """跑一轮 decode：每个运行中的序列产出一个 token。"""
        step_time = 0.0
        for seq in self.running:
            ctx = seq["context"] + seq["produced"]
            t = decode_step_time(
                self.device,
                self.spec,
                len(self.running),
                mfu=self.mfu,
                weight_bits=self.weight_bits,
                context_tokens=ctx,
            )
            step_time = max(step_time, t)

        self.now += step_time

        finished: list[dict[str, Any]] = []
        for seq in self.running:
            seq["produced"] += 1
            if seq["produced"] >= seq["request"].output_tokens():
                finished.append(seq)

        for seq in finished:
            request: Request = seq["request"]
            produced = seq["produced"]
            self.completions.append(
                Completion(
                    request=request,
                    enqueue_s=seq["enqueue_s"],
                    prefill_end_s=seq["prefill_end_s"],
                    finish_s=self.now,
                    ttft_s=seq["prefill_end_s"] - request.arrival_s,
                    tpot_s=(self.now - seq["prefill_end_s"]) / max(1, produced - 1)
                    if produced > 1
                    else 0.0,
                )
            )
            self.running.remove(seq)

    # -- 指标 -------------------------------------------------------------

    def metrics(self) -> dict[str, float]:
        """把完成情况汇总成一张指标表。"""
        if not self.completions:
            return {
                "count": 0,
                "ttft_p50_ms": 0.0,
                "ttft_p99_ms": 0.0,
                "tpot_p50_ms": 0.0,
                "e2e_p99_ms": 0.0,
                "output_tokens_per_s": 0.0,
                "requests_per_s": 0.0,
                "sla_violation_rate": 0.0,
            }

        ttfts = sorted(c.ttft_s for c in self.completions)
        tpots = sorted(c.tpot_s for c in self.completions)
        e2es = sorted(c.total_time_s for c in self.completions)
        total_output = sum(c.output_tokens for c in self.completions)
        makespan = max(c.finish_s for c in self.completions)
        start = min(c.request.arrival_s for c in self.completions)
        span = max(makespan - start, 1e-9)
        violations = sum(1 for c in self.completions if c.missed_ttft_deadline())

        def percentile(values: Sequence[float], q: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, int(q * len(values)))
            return values[index]

        return {
            "count": float(len(self.completions)),
            "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
            "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
            "tpot_p50_ms": percentile(tpots, 0.50) * 1000,
            "e2e_p99_ms": percentile(e2es, 0.99) * 1000,
            "output_tokens_per_s": total_output / span,
            "requests_per_s": len(self.completions) / span,
            "sla_violation_rate": violations / len(self.completions),
        }


def percentile(values: Sequence[float], q: float) -> float:
    """取分位数（最近秩法）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def load_balancing_delta(
    device: GPUSpec,
    spec: ModelSpec,
    *,
    replicas: int,
    qps: float,
    avg_prompt_tokens: int = 500,
    avg_output_tokens: int = 200,
    mfu: float = 0.5,
) -> dict[str, float]:
    """粗算多副本部署下每卡的负载，用于第 11、15、16 章。

    口径：单请求的端到端"占用时间" = prefill + decode，
    每张卡每秒能处理的请求数 = 1 / 平均占用时间（单流近似）。
    """
    pre = prefill_time(device, spec, avg_prompt_tokens, mfu=mfu)
    step = decode_step_time(device, spec, 1, mfu=mfu, context_tokens=avg_prompt_tokens)
    decode = step * avg_output_tokens
    per_request = pre + decode
    capacity_per_replica = 1.0 / per_request if per_request > 0 else 0.0
    total_capacity = capacity_per_replica * replicas
    return {
        "per_request_s": per_request,
        "capacity_per_replica": capacity_per_replica,
        "total_capacity": total_capacity,
        "offered_qps": qps,
        "utilization": (qps / total_capacity) if total_capacity > 0 else float("inf"),
        "replicas_needed": qps / capacity_per_replica if capacity_per_replica > 0 else float("inf"),
    }
