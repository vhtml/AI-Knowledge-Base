"""第 13 章实验：提示词流水线与长上下文混批。

这个模块解决什么问题：请求从网关进来到引擎吐出第一个 token，中间要过
tokenize、套 chat 模板、截断、注入、prefill 五道工序。这五道工序里藏着
两把"静默的刀"——一把砍掉前缀缓存的收益，一把砍掉 TTFT 的余量。
本章把这两件事变成可验证的断言：

1. **tokenize 是模型相关的**：同一段文本在不同 tokenizer 下 token 序列不同，
   第 12 章那套"按块哈希的前缀缓存"必然全部失效——不是命中率下降，是直接归零。
2. **prompt 的构造顺序决定共享块数**：每请求都变的字段（时间戳、随机 ID）
   放在最前面，共享块数直接归零；放在最后才保得住前缀。
3. **长 prompt 的 prefill 就是 TTFT 尖刺的来源**：32K token 的 prompt
   prefill 要 1.3 秒，单条请求就能把 TTFT 打到 1 秒的常规约定之外，
   并把同期所有请求一起拖高。
4. **分块 prefill 能压住长尾**：把长 prompt 切块后，同期短请求的
   p99 TTFT 从 1.40 秒降到 0.99 秒。

所有算账一律调用 labkit（``prefill_time`` / ``prefill_flops`` / ``kv_gib`` /
``EngineSimulator``），口径与前面各章一致。
全部离线运行：不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from ..labkit import (
    EngineSimulator,
    Request,
    block_chain,
    banner,
    ensure,
    fmt_gib,
    gpu,
    kv_gib,
    model,
    prefill_flops,
    prefill_time,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
SPEC = model("qwen2.5-7b")

BLOCK_SIZE = 16
"""前缀哈希与分页缓存共用的 block 粒度。"""

MFU = 0.5
"""与前面各章一致的可达算力比例。"""

SYSTEM_WORDS_A = 256
"""系统提示的前半段词数（tokenizer A 下即 token 数）。"""

SYSTEM_WORDS_B = 256
"""系统提示的后半段词数。"""

VARIABLE_WORDS = 4
"""每请求都变的字段（时间戳、随机 ID、会话 ID）占的词数。"""

CONTEXT_WORDS = 20
"""每请求都变的正文（检索结果、用户问题）占的词数。"""

SHORT_PROMPT_TOKENS = 128
"""混批实验里短请求的 prompt 长度。"""

SHORT_OUTPUT_TOKENS = 32
"""混批实验里短请求的输出长度。"""

LONG_PROMPT_TOKENS = 32768
"""混批实验里长请求的 prompt 长度（Qwen2.5-7B 的上下文上限）。"""

LONG_OUTPUT_TOKENS = 64
"""混批实验里长请求的输出长度。"""

SHORT_REQUESTS = 12
"""混批实验里短请求的条数。"""

CHUNK_SIZE = 2048
"""分块 prefill 的块大小。"""

PROMPT_LENGTHS = (512, 2048, 8192, 32768)
"""量化 prefill 成本时扫的 prompt 长度。"""

TTFT_BUDGET_S = 1.0
"""TTFT 的常规约定：到达后 1 秒内出首 token（与 labkit 的 ``ttft_deadline`` 一致）。"""


def _stable_hash(text: str) -> int:
    """文本的确定性哈希。

    不能用内置 ``hash()``——它对字符串按进程随机化种子取值，
    同样的输入在不同进程会得到不同的结果，实验就不可复现了。
    """
    return zlib.crc32(text.encode("utf-8"))


@dataclass(frozen=True)
class FakeTokenizer:
    """一个离线可跑的假 tokenizer。

    真实 tokenizer（BPE / WordPiece / SentencePiece）词表和合并规则各不相同，
    这里不还原它们的细节，只还原一条最关键的性质：
    **同一段文本在不同 tokenizer 下会变成不同的 token 序列**。

    ``pieces_per_word`` 模拟"词表更小、单词被切得更碎"的 tokenizer：
    每个词被切成 ``pieces_per_word`` 个 token。
    """

    name: str
    pieces_per_word: int

    def encode(self, text: str) -> tuple[int, ...]:
        """把文本编码成 token 序列。

        token id 用文本的 CRC32 派生，保证同一个 tokenizer 下**同样输入永远同样输出**
        ——这是前缀缓存能用的前提。
        """
        tokens: list[int] = []
        for word in text.split():
            digest = _stable_hash(word)
            for piece in range(self.pieces_per_word):
                tokens.append((digest + piece * 7919) % 100_000)
        return tuple(tokens)


TOKENIZER_A = FakeTokenizer(name="Tokenizer-A（词表大，整词入表）", pieces_per_word=1)
TOKENIZER_B = FakeTokenizer(name="Tokenizer-B（词表小，切成子词）", pieces_per_word=2)


@dataclass(frozen=True)
class PromptTexts:
    """一条请求的 prompt 素材。"""

    system_a: str
    system_b: str
    variable: str
    """每请求都变的字段。"""
    context: str
    """每请求都变的正文。"""


def build_texts(req_id: int) -> PromptTexts:
    """构造第 ``req_id`` 条请求的 prompt 素材。

    ``variable`` 段刻意放进请求 id 和一个"会话时间戳"——
    这类字段每请求都变，是前缀缓存失效的头号原因。
    """
    system_a = " ".join(f"sysa{index}" for index in range(SYSTEM_WORDS_A))
    system_b = " ".join(f"sysb{index}" for index in range(SYSTEM_WORDS_B))
    variable = f"session sess-{req_id} ts-{1730000000 + req_id * 37} trace-{req_id}"
    context = " ".join(f"ctx{req_id}-{index}" for index in range(CONTEXT_WORDS))
    return PromptTexts(system_a=system_a, system_b=system_b, variable=variable, context=context)


def assemble(layout: str, texts: PromptTexts, tokenizer: FakeTokenizer) -> tuple[int, ...]:
    """按指定布局拼装 prompt 的 token 序列。

    三种布局：

    - ``variable_first``：每请求都变的字段放在最前面（最坏情况）
    - ``variable_middle``：放在系统提示中间
    - ``variable_last``：放在最后面（推荐做法）
    """
    if layout == "variable_first":
        pieces = [texts.variable, texts.system_a, texts.system_b, texts.context]
    elif layout == "variable_middle":
        pieces = [texts.system_a, texts.variable, texts.system_b, texts.context]
    elif layout == "variable_last":
        pieces = [texts.system_a, texts.system_b, texts.variable, texts.context]
    else:
        raise ValueError(f"未知布局：{layout}")
    tokens: list[int] = []
    for piece in pieces:
        tokens.extend(tokenizer.encode(piece))
    return tuple(tokens)


def shared_leading_blocks(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """两条 token 序列**开头连续相同**的 block 数。

    这就是前缀缓存能共享的块数：链式哈希下，前 N 个块相同就意味着
    前 N 个块可以复用同一份 KV。
    """
    left_chain = block_chain(left, block_size=BLOCK_SIZE)
    right_chain = block_chain(right, block_size=BLOCK_SIZE)
    shared = 0
    for (left_key, _), (right_key, _) in zip(left_chain, right_chain):
        if left_key != right_key:
            break
        shared += 1
    return shared


def check_tokenizer_mismatch_breaks_prefix_cache() -> None:
    """换 tokenizer = 换 token 序列 = 前缀缓存归零。

    这一节要证的是：第 12 章的前缀缓存键是**token 序列的分块哈希**，
    所以只要 token 序列变了，块哈希就变了，缓存命中不了。
    """
    section("1. tokenize 的坑：换模型就换前缀")
    texts_a = build_texts(0)
    texts_b = build_texts(1)

    tokens_a = TOKENIZER_A.encode(texts_a.system_a)
    tokens_b = TOKENIZER_B.encode(texts_a.system_a)
    ensure(
        len(tokens_a) != len(tokens_b),
        f"两个 tokenizer 的 token 数应不同，实际都是 {len(tokens_a)}",
    )

    # 同一个 tokenizer 下，同样的文本永远得到同样的 token 序列
    ensure(
        TOKENIZER_A.encode(texts_a.system_a) == tokens_a,
        "同一个 tokenizer 的编码必须可复现，否则前缀缓存无法命中",
    )

    # 换 tokenizer：token 序列不同 → 块哈希从第一块起就对不上
    shared = shared_leading_blocks(tokens_a, tokens_b)
    ensure(
        shared == 0,
        f"换 tokenizer 后不应共享任何块，实际共享 {shared} 块",
    )

    # 代价不只是"缓存没命中"，prefill 本身也变贵了
    prefill_a = prefill_time(DEVICE, SPEC, len(tokens_a), mfu=MFU)
    prefill_b = prefill_time(DEVICE, SPEC, len(tokens_b), mfu=MFU)
    kv_a = kv_gib(SPEC, len(tokens_a))
    kv_b = kv_gib(SPEC, len(tokens_b))

    print_table(
        ["tokenizer", "系统提示 token 数", "prefill 时间", "KV 显存"],
        [
            [TOKENIZER_A.name, str(len(tokens_a)), f"{prefill_a * 1000:.2f}ms", fmt_gib(kv_a)],
            [TOKENIZER_B.name, str(len(tokens_b)), f"{prefill_b * 1000:.2f}ms", fmt_gib(kv_b)],
        ],
    )
    ensure(
        prefill_b > prefill_a,
        f"token 数更多的 tokenizer prefill 应更贵，"
        f"实际 {prefill_b * 1000:.2f}ms vs {prefill_a * 1000:.2f}ms",
    )
    ensure(
        kv_b > kv_a,
        f"token 数更多的 tokenizer KV 显存应更大，"
        f"实际 {fmt_gib(kv_b)} vs {fmt_gib(kv_a)}",
    )
    rule(
        f"同一段系统提示：tokenizer A 编出 {len(tokens_a)} token，"
        f"tokenizer B 编出 {len(tokens_b)} token，共享块数 {shared}——"
        "前缀缓存对两个模型各存一份，KV 显存和 prefill 都翻倍"
    )


def check_prompt_layout_decides_prefix_hit() -> None:
    """prompt 的拼装顺序决定共享块数：变化的字段必须放最后。

    共享块数直接等于"每条请求可以省下的 prefill token 数 × block 粒度"。
    """
    section("2. 缓存友好的 prompt 构造：变化的字段放最后")
    rows: list[list[str]] = []
    results: dict[str, tuple[int, int, float]] = {}
    for layout, label in (
        ("variable_first", "变量在最前"),
        ("variable_middle", "变量在中间"),
        ("variable_last", "变量在最后"),
    ):
        first = assemble(layout, build_texts(0), TOKENIZER_A)
        second = assemble(layout, build_texts(1), TOKENIZER_A)
        shared = shared_leading_blocks(first, second)
        shared_tokens = shared * BLOCK_SIZE
        saved_prefill = prefill_time(DEVICE, SPEC, shared_tokens, mfu=MFU)
        saved_kv = kv_gib(SPEC, shared_tokens)
        results[layout] = (shared, shared_tokens, saved_prefill)
        rows.append(
            [
                label,
                str(len(first)),
                str(shared),
                str(shared_tokens),
                f"{saved_prefill * 1000:.2f}ms",
                fmt_gib(saved_kv),
            ]
        )

    print_table(
        ["prompt 布局", "总 token 数", "共享块数", "可省 token", "省下的 prefill", "省下的 KV"],
        rows,
    )

    first_shared = results["variable_first"][0]
    middle_shared = results["variable_middle"][0]
    last_shared = results["variable_last"][0]
    ensure(
        first_shared == 0,
        f"变量放最前时共享块数应为 0，实际 {first_shared}",
    )
    ensure(
        middle_shared < last_shared,
        f"变量在中间的共享块数应少于放最后，实际 {middle_shared} vs {last_shared}",
    )
    ensure(
        last_shared * BLOCK_SIZE >= SYSTEM_WORDS_A + SYSTEM_WORDS_B,
        f"变量放最后时应保住整段系统提示，实际只共享 {last_shared * BLOCK_SIZE} token",
    )
    ensure(
        results["variable_last"][2] > results["variable_first"][2],
        "变量放最后应比放最前省下更多 prefill",
    )
    rule(
        f"同样两条请求：变量在最前共享 0 块（每条请求都重算全部 "
        f"{SYSTEM_WORDS_A + SYSTEM_WORDS_B} token 系统提示），"
        f"变量在最后共享 {last_shared} 块，"
        f"每条省下 {results['variable_last'][2] * 1000:.2f}ms prefill"
    )


def check_long_prompt_ttft_spike() -> None:
    """长 prompt 的 prefill 是算力瓶颈，也是 TTFT 尖刺的来源。"""
    section("3. 长上下文的 prefill 成本与 TTFT 尖刺")
    rows: list[list[str]] = []
    base_prefill = prefill_time(DEVICE, SPEC, PROMPT_LENGTHS[0], mfu=MFU)
    for tokens in PROMPT_LENGTHS:
        flops = prefill_flops(SPEC, tokens)
        gemm = 2.0 * SPEC.non_embedding_params_b * 1e9 * tokens
        attention = flops - gemm
        cost = prefill_time(DEVICE, SPEC, tokens, mfu=MFU)
        rows.append(
            [
                str(tokens),
                f"{gemm / 1e12:.1f}T",
                f"{attention / 1e12:.1f}T",
                f"{attention / flops * 100:.1f}%",
                f"{cost * 1000:.2f}ms",
                f"{cost / base_prefill:.1f}x",
                fmt_gib(kv_gib(SPEC, tokens)),
            ]
        )
    print_table(
        ["prompt token", "GEMM FLOPs", "注意力 FLOPs", "注意力占比", "prefill 时间", "相对 512", "KV 显存"],
        rows,
    )

    long_prefill = prefill_time(DEVICE, SPEC, LONG_PROMPT_TOKENS, mfu=MFU)
    ensure(
        long_prefill > TTFT_BUDGET_S,
        f"32K prompt 的 prefill 应超过 {TTFT_BUDGET_S:.0f}s 的 TTFT 约定，"
        f"实际 {long_prefill:.3f}s",
    )
    ensure(
        long_prefill > base_prefill * 20,
        f"32K prompt 的 prefill 应远高于 512 token，实际只高 "
        f"{long_prefill / base_prefill:.1f} 倍",
    )
    # 超线性：token 数翻 4 倍，prefill 时间远不止 4 倍（注意力是平方项）
    mid = prefill_time(DEVICE, SPEC, 8192, mfu=MFU)
    ensure(
        mid / base_prefill > 4,
        f"prefill 时间对 token 数应超线性，8192/512 = {mid / base_prefill:.1f} 倍",
    )
    rule(
        f"512 token 的 prefill 是 {base_prefill * 1000:.2f}ms，"
        f"32768 token 是 {long_prefill * 1000:.2f}ms（{long_prefill / base_prefill:.0f} 倍），"
        f"单条就能越过 {TTFT_BUDGET_S:.0f}s 的 TTFT 约定"
    )


def _mixed_workload() -> list[Request]:
    """混批负载：12 条短请求 + 1 条 32K 长请求几乎同时到达。

    长请求排在短请求后面但到达更早，所以它先被 prefill，
    短请求全被堵在它后面——这就是"长请求阻塞短请求"的最小复现。
    """
    requests: list[Request] = []
    for index in range(SHORT_REQUESTS):
        requests.append(
            Request(
                req_id=index,
                arrival_s=0.0 + index * 0.01,
                prompt_tokens=SHORT_PROMPT_TOKENS,
                max_output_tokens=SHORT_OUTPUT_TOKENS,
            )
        )
    requests.append(
        Request(
            req_id=SHORT_REQUESTS,
            arrival_s=0.025,
            prompt_tokens=LONG_PROMPT_TOKENS,
            max_output_tokens=LONG_OUTPUT_TOKENS,
        )
    )
    return requests


def check_chunked_prefill_limits_tail() -> None:
    """分块 prefill 压住长尾：同期短请求的 p99 TTFT 明显下降。"""
    section("4. 长请求阻塞短请求：分块 prefill 能压住长尾")
    rows: list[list[str]] = []
    metrics: dict[bool, dict[str, float]] = {}
    for chunked in (False, True):
        engine = EngineSimulator(
            DEVICE,
            SPEC,
            max_batch=32,
            kv_token_budget=200_000,
            mfu=MFU,
            chunked_prefill=chunked,
            chunk_size=CHUNK_SIZE,
        )
        completions = engine.run(_mixed_workload(), horizon=600)
        metrics[chunked] = engine.metrics()

        long_completion = next(
            c for c in completions if c.request.prompt_tokens == LONG_PROMPT_TOKENS
        )
        short_ttfts = sorted(c.ttft_s for c in completions if c.request.prompt_tokens < 1000)
        short_p99 = short_ttfts[min(len(short_ttfts) - 1, int(0.99 * len(short_ttfts)))]
        rows.append(
            [
                "分块 prefill" if chunked else "整块 prefill",
                f"{long_completion.ttft_s * 1000:.1f}ms",
                f"{short_p99 * 1000:.1f}ms",
                f"{metrics[chunked]['ttft_p50_ms']:.1f}ms",
                f"{metrics[chunked]['ttft_p99_ms']:.1f}ms",
            ]
        )

    print_table(
        ["prefill 方式", "长请求 TTFT", "短请求 p99 TTFT", "整体 p50 TTFT", "整体 p99 TTFT"],
        rows,
    )

    monolithic = metrics[False]
    chunked = metrics[True]
    ensure(
        chunked["ttft_p99_ms"] < monolithic["ttft_p99_ms"],
        f"分块 prefill 的 p99 TTFT 应低于整块，"
        f"实际 {chunked['ttft_p99_ms']:.1f}ms vs {monolithic['ttft_p99_ms']:.1f}ms",
    )
    ensure(
        chunked["ttft_p50_ms"] < monolithic["ttft_p50_ms"],
        f"分块 prefill 的 p50 TTFT 也应更低，"
        f"实际 {chunked['ttft_p50_ms']:.1f}ms vs {monolithic['ttft_p50_ms']:.1f}ms",
    )
    drop = (monolithic["ttft_p99_ms"] - chunked["ttft_p99_ms"]) / monolithic["ttft_p99_ms"]
    rule(
        f"同一条 32K prompt：整块 prefill 独占引擎 "
        f"{prefill_time(DEVICE, SPEC, LONG_PROMPT_TOKENS, mfu=MFU) * 1000:.0f}ms，"
        f"切 {CHUNK_SIZE} token 的块后降到 "
        f"{prefill_time(DEVICE, SPEC, CHUNK_SIZE, mfu=MFU) * 16 * 1000:.0f}ms，"
        f"整体 p99 TTFT 下降 {drop * 100:.0f}%"
    )
    rule(
        "口径说明：labkit 的解析模型把注意力项按块结算，"
        "所以分块后长 prompt 的引擎占用时间也变短；"
        "真实引擎里 FLOPs 不会这样下降，分块的主要收益是让 decode 有机会插进来"
    )


def run() -> None:
    """跑完第 13 章的全部实验。"""
    banner("第 13 章 · 提示词流水线与长上下文混批")
    rule(
        f"口径：{SPEC.name} / {DEVICE.name} / block={BLOCK_SIZE} / MFU={MFU} / "
        f"长 prompt {LONG_PROMPT_TOKENS} tokens"
    )
    check_tokenizer_mismatch_breaks_prefix_cache()
    check_prompt_layout_decides_prefix_hit()
    check_long_prompt_ttft_spike()
    check_chunked_prefill_limits_tail()
    print()
    rule(
        "结论：tokenize 是模型相关的（换 tokenizer 前缀缓存直接归零）；"
        "每请求都变的字段必须放在 prompt 最后面；"
        "长 prompt 的 prefill 是 TTFT 尖刺的根源，"
        "分块 prefill 能把 p99 TTFT 压下来"
    )
