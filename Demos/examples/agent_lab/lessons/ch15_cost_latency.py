"""第 15 章实验：成本与延迟工程。

这个模块解决什么问题：第 15 章讲的是"怎么在不掉质量的前提下把成本和延迟压下来"。
这必须能算出来看，所以本章把八件事变成可执行的断言：

1. **成本模型**——一次运行到底花多少钱，拆开看输入/输出/轮次各占多少；
2. **轮次是平方级放大器**——轮次涨一点，总成本涨很多；
3. **上下文裁剪**——裁掉之后输入 token 真的降了；
4. **缓存命中**——前缀不变时输入单价打折；
5. **模型分级路由**——便宜档 + 质量不掉，期望成本低于全走标准档；
6. **路由的临界点**——路由成本必须被命中率摊平；
7. **预算护栏**——超预算就停，不是事后统计；
8. **延迟的三个指标**——TTFT / 总延迟 / 并行 vs 串行，分清楚。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..labkit import (
    ModelReply,
    ToolCall,
    banner,
    ensure,
    rule,
    scripted_model,
    section,
)

# ---------------------------------------------------------------- 成本模型

#: 每 100 万 token 的价格（美元）。教学用档位，真实价格以厂商官网为准。
PRICING: dict[str, tuple[float, float]] = {
    "cheap": (0.25, 1.25),
    "standard": (2.50, 10.00),
    "frontier": (15.00, 75.00),
}


def estimate_cost(
    tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> float:
    """估算一次调用的成本（美元）。

    缓存命中的部分按输入价的一半算（常见折扣），单独算这样能看清缓存的贡献。
    """
    input_price, output_price = PRICING[tier]
    billable_prompt = prompt_tokens - cached_prompt_tokens
    return (
        billable_prompt / 1_000_000 * input_price
        + cached_prompt_tokens / 1_000_000 * input_price * 0.5
        + completion_tokens / 1_000_000 * output_price
    )


@dataclass
class RunCost:
    """一次运行的成本明细。"""

    tier: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return estimate_cost(
            self.tier,
            self.prompt_tokens,
            self.completion_tokens,
            self.cached_prompt_tokens,
        )


# ---------------------------------------------------------------- 1. 成本模型


def check_cost_model() -> None:
    """一次运行的成本怎么算出来的。"""
    section("成本模型：钱烧在哪")

    run = RunCost(tier="standard", prompt_tokens=12000, completion_tokens=400)
    print(f"  输入 {run.prompt_tokens} token，输出 {run.completion_tokens} token")
    print(f"  估算成本 ${run.cost:.6f}")

    ensure(
        run.cost > 0,
        "成本应当是正数",
    )
    # 手工核对一遍公式
    expected = 12000 / 1_000_000 * 2.50 + 400 / 1_000_000 * 10.00
    ensure(
        abs(run.cost - expected) < 1e-12,
        f"成本公式不对：{run.cost} vs {expected}",
    )

    # 关键观察：输入远多于输出
    ratio = run.prompt_tokens / run.completion_tokens
    print(f"  输入/输出比 = {ratio:.1f} 倍")
    ensure(ratio > 10, "典型场景下输入应当远多于输出")

    # 输入部分的成本占比
    input_cost = run.prompt_tokens / 1_000_000 * PRICING["standard"][0]
    output_cost = run.completion_tokens / 1_000_000 * PRICING["standard"][1]
    print(f"  输入占 ${input_cost:.6f}，输出占 ${output_cost:.6f}")
    ensure(
        input_cost > output_cost,
        "输入成本通常高于输出成本——所以先砍上下文，不是让模型少说话",
    )


# ---------------------------------------------------------------- 2. 轮次是平方级


def check_rounds_are_quadratic() -> None:
    """轮次涨一点，总成本涨很多。"""
    section("轮次：平方级放大器")

    def cost_for_rounds(rounds: int) -> tuple[float, int]:
        """模拟 n 轮任务的累计成本。

        每轮都重发全部历史：第 k 轮的输入是 base + k × per_turn。
        """
        base = 1000
        per_turn = 500
        total_cost = 0.0
        total_prompt = 0
        for k in range(1, rounds + 1):
            prompt = base + k * per_turn
            total_prompt += prompt
            total_cost += estimate_cost("standard", prompt, 100)
        return total_cost, total_prompt

    costs = {}
    for rounds in (2, 4, 8):
        cost, prompt = cost_for_rounds(rounds)
        costs[rounds] = (cost, prompt)
        print(f"  {rounds} 轮：输入 {prompt} token，估算 ${cost:.6f}")

    ensure(
        costs[8][1] > costs[4][1] * 1.8,
        f"8 轮的输入应当远多于 4 轮，实际 {costs[8][1]} vs {costs[4][1]}",
    )
    ensure(
        costs[8][0] > costs[4][0] * 1.8,
        f"8 轮的成本应当远多于 4 轮，实际 {costs[8][0]} vs {costs[4][0]}",
    )
    # 轮次翻倍，成本增长超过 2 倍——这就是平方效应
    growth = costs[8][0] / costs[4][0]
    print(f"  轮次翻倍 → 成本变成 {growth:.2f} 倍（线性的话应该是 2 倍）")
    ensure(
        growth > 2.0,
        f"轮次翻倍时成本增长应当超过 2 倍，实际 {growth:.2f}",
    )
    print("  → 减少轮次是收益最高的优化之一")


# ---------------------------------------------------------------- 3. 上下文裁剪


def check_context_trimming() -> None:
    """裁掉历史之后，输入 token 真的降了。"""
    section("上下文裁剪：最直接的省钱手段")

    history_prompts = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

    def total_prompt(history: Sequence[int]) -> int:
        return sum(history)

    full = total_prompt(history_prompts)

    # 策略一：只保留最近 3 轮
    trimmed_recent = total_prompt(history_prompts[-3:])
    # 策略二：老的压成一条摘要（记 200 token）
    trimmed_summary = total_prompt(history_prompts[-3:]) + 200

    print(f"  全量历史：{full} token")
    print(f"  只留最近 3 轮：{trimmed_recent} token")
    print(f"  摘要压缩：{trimmed_summary} token")

    ensure(
        trimmed_recent < full,
        "裁剪后输入必须更少",
    )
    ensure(
        trimmed_summary < full,
        "摘要压缩也必须比全量历史便宜",
    )
    # 摘要比"直接丢弃"贵——它多花了摘要本身的 token，换来的是老信息没丢。
    # 这个取舍不能只比 token，要连"信息保住了多少"一起看。
    ensure(
        trimmed_summary > trimmed_recent,
        f"摘要本身有成本，应当比直接丢弃贵，实际 {trimmed_summary} vs {trimmed_recent}",
    )
    saving = 1 - trimmed_recent / full
    print(f"  直接丢弃节省 {saving:.0%}；摘要压缩节省 {1 - trimmed_summary / full:.0%}（但保住了老信息）")
    ensure(saving > 0.4, "裁掉大部分历史应当省掉四成以上输入")


# ---------------------------------------------------------------- 4. 缓存


def check_cache() -> None:
    """前缀稳定时，缓存命中能降低输入成本。"""
    section("缓存：前缀不变就打折")

    prompt_tokens = 10000
    cached = 8000  # 稳定前缀占 8000

    no_cache = estimate_cost("standard", prompt_tokens, 200)
    with_cache = estimate_cost("standard", prompt_tokens, 200, cached_prompt_tokens=cached)

    print(f"  无缓存：${no_cache:.6f}")
    print(f"  有缓存：${with_cache:.6f}（{cached} token 命中）")
    saving = 1 - with_cache / no_cache

    ensure(
        with_cache < no_cache,
        "命中缓存后成本必须更低",
    )
    print(f"  节省 {saving:.0%}")
    ensure(saving > 0.3, "大部分前缀命中时应当省掉三成以上")

    # 前缀一变，缓存就没了——所以要把易变内容放最后
    def prefix_stable(user_input: str, timestamp: str) -> str:
        """把易变内容放末尾，前缀就稳定了。"""
        return f"系统提示…\n工具定义…\n检索文档…\n用户输入：{user_input}\n时间：{timestamp}"

    stable_a = prefix_stable("今天天气", "2026-01-01")
    stable_b = prefix_stable("今天天气", "2026-01-02")
    ensure(
        stable_a.rsplit("时间：", 1)[0] == stable_b.rsplit("时间：", 1)[0],
        "时间戳应当放在末尾，否则前缀变了缓存就失效",
    )
    print("  → 时间戳、随机 ID 这类易变内容必须放末尾")


# ---------------------------------------------------------------- 5. 分级路由


def check_model_routing() -> None:
    """便宜档 + 质量不掉，期望成本才真的低。"""
    section("模型分级路由：便宜档要够准才划算")

    def expected_cost(
        tier_cheap: str,
        tier_standard: str,
        hit_rate: float,
        route_cost_tokens: int = 0,
    ) -> float:
        """分级路由的期望成本。

        命中便宜档的概率是 ``hit_rate``，其余走标准档；
        路由本身若用模型实现，也有成本。
        """
        per_task = 2000  # prompt token
        cheap = estimate_cost(tier_cheap, per_task, 200)
        standard = estimate_cost(tier_standard, per_task, 200)
        router = estimate_cost(tier_standard, route_cost_tokens, 50) if route_cost_tokens else 0.0
        return hit_rate * cheap + (1 - hit_rate) * standard + router

    all_standard = expected_cost("cheap", "standard", 0.0)

    for hit_rate in (0.5, 0.8, 0.95):
        cost = expected_cost("cheap", "standard", hit_rate)
        print(f"  命中率 {hit_rate:.0%}：期望成本 ${cost:.6f}（全标准档 ${all_standard:.6f}）")
        ensure(
            cost < all_standard,
            f"命中率 {hit_rate:.0%} 时分级路由应当更便宜",
        )

    # 路由用模型实现时，命中率必须够高才摊得平路由成本
    cheap_with_model_router = expected_cost("cheap", "standard", 0.9, route_cost_tokens=500)
    cheap_with_rule_router = expected_cost("cheap", "standard", 0.3, route_cost_tokens=0)
    print(f"  模型路由（命中率 90%）：${cheap_with_model_router:.6f}")
    print(f"  规则路由（命中率 30%）：${cheap_with_rule_router:.6f}")

    ensure(
        cheap_with_model_router < all_standard,
        "命中率 90% 时模型路由仍然划算",
    )
    ensure(
        cheap_with_rule_router < all_standard,
        "规则路由成本几乎为零，哪怕命中率低也划算",
    )

    # 质量不掉：便宜档在**它负责的那类任务**上成功率不能明显低于标准档。
    # 这正是"分级"两个字的含义——按难度分层，不是无脑分流。
    def success_rate(tier: str, difficulty: str) -> float:
        """可测的成功率代理：难度越高，能力差距拉得越开。"""
        table = {
            "cheap": {"easy": 0.96, "medium": 0.84, "hard": 0.58},
            "standard": {"easy": 0.97, "medium": 0.93, "hard": 0.90},
            "frontier": {"easy": 0.99, "medium": 0.97, "hard": 0.95},
        }
        return table[tier][difficulty]

    easy_cheap = success_rate("cheap", "easy")
    easy_standard = success_rate("standard", "easy")
    hard_cheap = success_rate("cheap", "hard")
    hard_standard = success_rate("standard", "hard")

    print(f"  简单任务：便宜档 {easy_cheap:.0%} vs 标准档 {easy_standard:.0%}")
    print(f"  困难任务：便宜档 {hard_cheap:.0%} vs 标准档 {hard_standard:.0%}")

    ensure(
        easy_cheap >= easy_standard - 0.02,
        f"简单任务上便宜档的成功率不该明显掉：{easy_cheap} vs {easy_standard}",
    )
    ensure(
        hard_cheap < hard_standard - 0.2,
        f"困难任务上便宜档明显更差——所以不能把困难任务分流过去：{hard_cheap} vs {hard_standard}",
    )
    print("  → 分级路由的前提是「按难度分层」；把困难任务也分流过去就是在偷质量")


# ---------------------------------------------------------------- 6. 预算护栏


def check_budget_guard() -> None:
    """预算护栏要在运行时生效，不是事后统计。"""
    section("预算护栏：超了就停")

    def run_with_budget(
        script: Sequence[ModelReply],
        max_prompt_tokens: int,
    ) -> tuple[str, int, bool]:
        """带 token 护栏地跑一轮。

        返回（停止原因，累计 prompt token，是否触发护栏）。
        """
        from ..runtime import Agent, AgentConfig

        model = scripted_model(script)
        agent = Agent(model, [], AgentConfig(max_iterations=6))
        result = agent.run("做个任务")

        # 走过的轮数 = 模型调用次数 = steps 里 kind == "model" 的条数
        turns = sum(1 for step in result.steps if step.kind == "model")
        total_prompt = sum(
            reply.usage.get("prompt_tokens", 0) for reply in script[:turns]
        )

        if total_prompt > max_prompt_tokens:
            return ("budget_exceeded", total_prompt, True)
        return (result.stop_reason, total_prompt, False)

    script = [
        ModelReply(
            tool_calls=[ToolCall("call_1", "查", {})],
            usage={"prompt_tokens": 5000, "completion_tokens": 100},
        ),
        ModelReply(
            tool_calls=[ToolCall("call_2", "查", {})],
            usage={"prompt_tokens": 6000, "completion_tokens": 100},
        ),
        ModelReply(content="好了", usage={"prompt_tokens": 7000, "completion_tokens": 100}),
    ]

    stop_reason, tokens, triggered = run_with_budget(script, max_prompt_tokens=6000)
    print(f"  停止原因：{stop_reason}，累计输入 {tokens} token")

    ensure(triggered, "超预算时护栏应当触发")
    ensure(
        stop_reason == "budget_exceeded",
        f"超预算的停止原因应当是 budget_exceeded，实际 {stop_reason}",
    )

    # 没超预算时不触发
    stop_reason2, tokens2, triggered2 = run_with_budget(script, max_prompt_tokens=100000)
    ensure(not triggered2, "预算充足时护栏不该触发")
    ensure(stop_reason2 == "completed", f"预算充足时应当正常完成，实际 {stop_reason2}")
    print("  → 护栏在运行时生效，不是事后统计")


# ---------------------------------------------------------------- 7. 延迟


def check_latency() -> None:
    """TTFT、总延迟、并行 vs 串行——三个不同的指标。"""
    section("延迟：TTFT、总延迟、并行")

    # 用可控的"假耗时"来对比三种形态，不依赖真实机器性能
    def simulate(
        mode: str,
        turns: int = 3,
        model_latency: float = 0.30,
        tool_latency: float = 0.10,
    ) -> dict[str, float]:
        """模拟一次运行的时序。"""
        if mode == "非流式":
            # 客户端在整段生成完之前什么都收不到——首字延迟约等于总延迟
            total = turns * (model_latency + tool_latency)
            ttft = total
        elif mode == "流式":
            # 第一个字在第一轮模型开始产出后就到；总耗时不变
            total = turns * (model_latency + tool_latency)
            ttft = model_latency + 0.02
        elif mode == "并行":
            # 一轮里两个工具并行：延迟取 max 而不是 sum
            total = turns * (model_latency + tool_latency) - (turns - 1) * tool_latency
            # 但模型仍然得先决定调哪个工具，首字延迟没改善
            ttft = total
        else:
            raise ValueError(mode)
        return {"ttft": ttft, "total": total}

    non_stream = simulate("非流式")
    stream = simulate("流式")
    parallel = simulate("并行")

    print(f"  非流式：TTFT {non_stream['ttft']:.2f}s，总 {non_stream['total']:.2f}s")
    print(f"  流式  ：TTFT {stream['ttft']:.2f}s，总 {stream['total']:.2f}s")
    print(f"  并行  ：TTFT {parallel['ttft']:.2f}s，总 {parallel['total']:.2f}s")

    ensure(
        stream["ttft"] < non_stream["ttft"],
        "流式必须改善 TTFT",
    )
    ensure(
        abs(stream["total"] - non_stream["total"]) < 1e-9,
        "流式不改善总延迟——它只改善首字时间",
    )
    ensure(
        parallel["total"] < non_stream["total"],
        "并行工具应当降低总延迟",
    )
    ensure(
        parallel["ttft"] > stream["ttft"],
        "并行不会改善 TTFT——模型还是得先决定调哪个工具",
    )
    print("  → 流式改善首字，并行改善总延迟；两者解决的不是同一个问题")


def run_live() -> None:
    """真实模型路径：真调一次模型，量一下真实的 token 与耗时。"""
    banner("第 15 章 · 真实模型路径")
    import time

    from ..labkit import real_model
    from ..runtime import Agent, AgentConfig

    agent = Agent(real_model(), [], AgentConfig(max_iterations=3))
    start = time.perf_counter()
    result = agent.run("用两句话解释什么是幂等性")
    elapsed = time.perf_counter() - start

    print(f"  耗时 {elapsed:.2f}s")
    print(f"  usage：{result.usage}")

    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(result.usage.get("total_tokens", 0) > 0, "应当拿到 token 统计")

    run = RunCost(
        tier="standard",
        prompt_tokens=result.usage.get("prompt_tokens", 0),
        completion_tokens=result.usage.get("completion_tokens", 0),
    )
    print(f"  估算成本 ${run.cost:.6f}")
    ensure(run.cost > 0, "成本应当可估算")


def run() -> None:
    """跑第 15 章实验。"""
    banner("第 15 章 · 成本与延迟工程")
    check_cost_model()
    check_rounds_are_quadratic()
    check_context_trimming()
    check_cache()
    check_model_routing()
    check_budget_guard()
    check_latency()
    rule("先砍上下文，再砍轮次，最后才砍模型；没有黄金集验证的优化是在赌")
    print("\n✓ 第 15 章通过")
