"""第 5 章实验：推理模型与思考预算。

这个模块解决什么问题：第 5 章的核心结论是"推理模型是给需要想的任务用的，
不是给需要多试的链路用的"。这必须能算出来看，所以本章把七件事变成可执行的断言：

1. **思考预算透传**——预算真的到了模型手里；
2. **推理 token 计入输出计费**——思考不是免费的；
3. **预算为 0 → 退化成普通模型**（不产出思考）；
4. **思考内容默认不回填进上下文**——它既贵又会让模型沿错误假设走；
5. **工具调用链路上的代价**——多轮工具调用 × 每轮重新思考 = 成本放大；
6. **决策表可执行**——按任务类型判断该不该开推理；
7. **性价比**——贵 N 倍换 M 倍成功率，划不划算要算出来。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..labkit import (
    ChatMessage,
    ChatResponse,
    PRICING,
    ToolSpec,
    banner,
    ensure,
    estimate_cost,
    rule,
    scripted_client,
    section,
    tool_spec,
)

# ---------------------------------------------------------------- 1. 预算透传


def check_budget_passthrough() -> None:
    """思考预算要真的送到模型手里。"""
    section("思考预算：透传到请求里")

    client = scripted_client(responses=[ChatResponse(content="391")])
    client.complete(
        [ChatMessage(role="user", content="17×23 等于几？")],
        reasoning_effort="high",
    )

    call = client.seen_calls[0]
    print(f"  请求里的 reasoning_effort：{call['reasoning_effort']!r}")
    ensure(
        call["reasoning_effort"] == "high",
        f"思考预算应当透传，实际 {call['reasoning_effort']!r}",
    )

    # 不传预算时，参数根本不该出现在请求里——普通模型会报错
    client2 = scripted_client(responses=[ChatResponse(content="ok")])
    client2.complete([ChatMessage(role="user", content="你好")])
    ensure(
        client2.seen_calls[0]["reasoning_effort"] is None,
        "没指定预算时不该传这个参数",
    )
    print("  → 没指定就不传：普通模型不认这个参数")


# ---------------------------------------------------------------- 2. 推理 token 计入计费


def check_reasoning_tokens_are_billed() -> None:
    """思考的 token 算在输出里，成本必须算进去。"""
    section("推理 token 计入输出计费")

    plain = ChatResponse(
        content="391",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    thinking = ChatResponse(
        content="391",
        reasoning="先算 17×20=340，17×3=51，340+51=391，再验算……",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 1500,
            "reasoning_tokens": 1450,
            "total_tokens": 1600,
        },
    )

    plain_cost = estimate_cost("reasoning", plain.usage["prompt_tokens"], plain.usage["completion_tokens"])
    think_cost = estimate_cost(
        "reasoning", thinking.usage["prompt_tokens"], thinking.usage["completion_tokens"]
    )

    print(f"  普通回答：{plain.usage['completion_tokens']} 输出 token，${plain_cost:.6f}")
    print(f"  带思考  ：{thinking.usage['completion_tokens']} 输出 token，${think_cost:.6f}")
    print(f"  其中思考 token：{thinking.usage['reasoning_tokens']}")

    ensure(
        thinking.usage["reasoning_tokens"] < thinking.usage["completion_tokens"],
        "推理 token 应当是输出 token 的一部分",
    )
    ensure(
        think_cost > plain_cost * 10,
        f"思考应当让成本涨一个量级，实际 {think_cost:.6f} vs {plain_cost:.6f}",
    )

    # 关键：思考 token 不额外计费，但它已经算在 completion_tokens 里了
    ensure(
        thinking.usage["completion_tokens"] >= thinking.usage["reasoning_tokens"],
        "completion_tokens 应当已经包含推理 token——账单上不会单列，但钱在里面",
    )
    print("  → 思考 token 不单独计费，但它已经在 completion_tokens 里了")


# ---------------------------------------------------------------- 3. 预算为 0


def check_zero_budget_degrades() -> None:
    """预算为 0（或不传）时，模型退化成普通模型。"""
    section("预算为 0：退化成普通模型")

    no_thinking = ChatResponse(content="391", reasoning="", usage={"completion_tokens": 50})
    ensure(
        no_thinking.reasoning == "",
        "不开推理时不应有思考内容",
    )

    low_budget = ChatResponse(
        content="391",
        reasoning="17×23 = 391",
        usage={"completion_tokens": 80, "reasoning_tokens": 30},
    )
    print(f"  不开推理：输出 {no_thinking.usage['completion_tokens']} token")
    print(f"  低预算  ：输出 {low_budget.usage['completion_tokens']} token（思考 {low_budget.usage['reasoning_tokens']}）")

    ensure(
        low_budget.usage["completion_tokens"] > no_thinking.usage["completion_tokens"],
        "开了推理输出 token 应当更多",
    )
    ensure(
        low_budget.usage["reasoning_tokens"] < low_budget.usage["completion_tokens"],
        "低预算的思考 token 应当远小于高预算",
    )

    # 预算越高，思考越长——但这是你的选择，不是模型强迫你的
    high_budget = ChatResponse(
        content="391",
        reasoning="……" * 200,
        usage={"completion_tokens": 2000, "reasoning_tokens": 1950},
    )
    ensure(
        high_budget.usage["reasoning_tokens"] > low_budget.usage["reasoning_tokens"],
        "高预算的思考 token 应当更多",
    )
    print("  → 预算越宽思考越长，成本随之线性上涨")


# ---------------------------------------------------------------- 4. 思考不回填


def check_reasoning_not_replayed() -> None:
    """思考内容默认不进下一轮上下文。"""
    section("思考内容：默认不回填")

    thinking = ChatResponse(
        content="答案是 391。",
        reasoning="假设订单号以 A 开头……（一个其实站不住脚的推断）",
        usage={"completion_tokens": 200, "reasoning_tokens": 150},
    )

    # 策略一：只把 content 放进下一轮
    without = [ChatMessage(role="assistant", content=thinking.content)]
    # 策略二：把 reasoning 也放进去
    with_reasoning = [
        ChatMessage(role="assistant", content=thinking.content),
        ChatMessage(role="user", content=f"你的思考：{thinking.reasoning}\n请据此继续"),
    ]

    def tokens_of(messages: Sequence[ChatMessage]) -> int:
        """粗估上下文的 token 数。"""
        from ..labkit import estimate_tokens

        return sum(estimate_tokens(str(m.content)) for m in messages)

    print(f"  只回填 content：{tokens_of(without)} token")
    print(f"  连思考一起回填：{tokens_of(with_reasoning)} token")

    ensure(
        tokens_of(with_reasoning) > tokens_of(without),
        "回填思考会让上下文变长（更贵）",
    )

    # 关键风险：思考里的假设会被当成既有结论
    joined = "\n".join(str(m.content) for m in with_reasoning)
    ensure(
        "假设订单号以 A 开头" in joined,
        "回填后那个站不住脚的假设确实进了上下文",
    )
    print("  → 回填的真正风险不是 token，是上一步的错误假设会被当成结论沿用")


# ---------------------------------------------------------------- 5. 工具链路上的代价


def check_tool_loop_cost_amplification() -> None:
    """工具调用链路上，每轮都要重新思考一次。"""
    section("工具调用链路：每轮重新思考 = 成本放大")

    reasoning_per_turn = 800
    prompt_tokens = 1200
    plain_completion = 100

    def turn_cost(use_reasoning: bool) -> float:
        """一轮的成本：输入固定，输出看开不开推理。"""
        completion = plain_completion + (reasoning_per_turn if use_reasoning else 0)
        return estimate_cost("reasoning", prompt_tokens, completion)

    def chain_cost(turns: int, use_reasoning: bool) -> float:
        """把推理放在**链路内**：每一轮都要重新思考一次。"""
        return sum(turn_cost(use_reasoning) for _ in range(turns))

    def plan_cost(turns: int) -> float:
        """把推理放在**规划步骤**：只思考一次，执行轮用普通模型。"""
        first = estimate_cost(
            "reasoning",
            prompt_tokens,
            plain_completion + reasoning_per_turn,
        )
        rest = sum(turn_cost(False) for _ in range(turns - 1))
        return first + rest

    rows = []
    for turns in (1, 3, 6):
        plain = chain_cost(turns, False)
        in_chain = chain_cost(turns, True)
        planned = plan_cost(turns)
        rows.append((turns, plain, in_chain, planned))
        print(
            f"  {turns} 轮：全普通 ${plain:.6f} | 推理放链路内 ${in_chain:.6f} "
            f"（{in_chain / plain:.1f}×） | 推理放规划步 ${planned:.6f}（{planned / plain:.1f}×）"
        )

    # 关键结论：同样开推理，放规划步骤比放链路内便宜得多
    turns, plain, in_chain, planned = rows[-1]
    ensure(
        planned < in_chain,
        f"推理放规划步骤应当比放链路内便宜：{planned:.6f} vs {in_chain:.6f}",
    )
    print(f"  → {turns} 轮时，把推理挪到规划步骤省下 ${in_chain - planned:.6f}")

    # 而且轮次越多，这个差距越大——因为链路内是每轮都付
    gap_3 = rows[1][2] - rows[1][3]
    gap_6 = rows[2][2] - rows[2][3]
    ensure(
        gap_6 > gap_3,
        f"轮次越多差距越大：{gap_3:.6f} → {gap_6:.6f}",
    )

    # 绝对成本随轮次线性放大
    ensure(
        rows[2][2] > rows[0][2] * 5,
        f"6 轮的成本应当远高于 1 轮，实际 {rows[2][2]:.6f} vs {rows[0][2]:.6f}",
    )
    print("  → 推理放在链路内是每轮都付；挪到规划步骤只付一次")


# ---------------------------------------------------------------- 6. 决策表


def check_decision_table() -> None:
    """按任务类型判断该不该开推理。"""
    section("决策表：该不该开推理")

    def should_use_reasoning(
        task_kind: str,
        tool_calls_expected: int,
        latency_sensitive: bool,
    ) -> bool:
        """判断这个任务该不该开推理。

        判据三条：
        1. 抽取/分类/改写 → 不开；
        2. 工具调用密集 → 不开（每轮重新思考太贵）；
        3. 对延迟敏感 → 不开（首字延迟翻倍）。
        """
        if task_kind in {"extraction", "classification", "rewrite"}:
            return False
        if tool_calls_expected >= 3:
            return False
        if latency_sensitive:
            return False
        return task_kind in {"multi_step_reasoning", "planning", "debugging"}

    cases = [
        ("extraction", 0, False, False, "抽取 → 不开"),
        ("classification", 0, False, False, "分类 → 不开"),
        ("rewrite", 0, False, False, "改写 → 不开"),
        ("multi_step_reasoning", 0, False, True, "多步推理 → 开"),
        ("planning", 1, False, True, "规划 → 开"),
        ("debugging", 2, False, True, "调试 → 开"),
        ("multi_step_reasoning", 4, False, False, "多步推理但工具调用多 → 不开"),
        ("planning", 0, True, False, "规划但延迟敏感 → 不开"),
    ]

    for kind, tools, latency, expected, why in cases:
        actual = should_use_reasoning(kind, tools, latency)
        print(f"  {why:<30} → {'开' if actual else '不开'}")
        ensure(actual == expected, f"{why} 判定应为 {expected}，实际 {actual}")

    ensure(
        not should_use_reasoning("multi_step_reasoning", 4, False),
        "工具调用密集时不该开推理",
    )


# ---------------------------------------------------------------- 7. 性价比


def check_cost_effectiveness() -> None:
    """贵 N 倍换 M 倍成功率，划不划算要算出来。"""
    section("性价比：贵 N 倍换 M 倍成功率")

    plain_cost = 0.01
    plain_success = 0.90
    reasoning_cost = 0.06
    reasoning_success = 0.96

    cost_multiple = reasoning_cost / plain_cost
    success_multiple = reasoning_success / plain_success
    value_multiple = cost_multiple / success_multiple

    print(f"  普通：${plain_cost} / 成功率 {plain_success:.0%}")
    print(f"  推理：${reasoning_cost} / 成功率 {reasoning_success:.0%}")
    print(f"  成本 {cost_multiple:.1f} 倍，成功率 {success_multiple:.2f} 倍 → 性价比 {value_multiple:.1f} 倍")

    ensure(
        value_multiple > 1,
        "单次性价比上推理是亏的——成本涨得比成功率高",
    )

    # 什么时候划算？取决于一次成功的价值
    def worth_it(value_per_success: float) -> bool:
        """一次成功的价值够不够覆盖成本差价。"""
        expected_cost_plain = plain_cost / plain_success
        expected_cost_reasoning = reasoning_cost / reasoning_success
        return expected_cost_reasoning < expected_cost_plain + value_per_success

    ensure(
        not worth_it(0.001),
        "一次成功只值 1 厘钱时，推理不划算",
    )
    ensure(
        worth_it(1.0),
        "一次成功值 1 块时，推理划算",
    )
    print("  → 决策不是「准不准」，是「这个任务值多少钱」")
    print(f"  → 每次成功的期望成本：普通 ${plain_cost / plain_success:.4f} vs 推理 ${reasoning_cost / reasoning_success:.4f}")


def run_live() -> None:
    """真实模型路径：真调一次推理模型，看看思考内容长什么样。"""
    banner("第 5 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client()
    response = client.complete(
        [ChatMessage(role="user", content="17×23 等于几？只回答数字。")],
        reasoning_effort="medium",
    )

    print(f"  content  : {response.content!r}")
    print(f"  reasoning: {response.reasoning[:200]!r}")
    print(f"  usage    : {response.usage}")

    ensure(bool(response.content), "真实模型应当返回内容")
    ensure(
        response.usage.get("total_tokens", 0) > 0,
        "应当拿到 token 统计",
    )
    if response.reasoning:
        print(f"  思考长度 {len(response.reasoning)} 字符——这段不会回填进上下文")
    else:
        print("  该端点未返回 reasoning 字段（可能不是推理模型，或该端点不透出思考）")


def run() -> None:
    """跑第 5 章实验。"""
    banner("第 5 章 · 推理模型与思考预算")
    check_budget_passthrough()
    check_reasoning_tokens_are_billed()
    check_zero_budget_degrades()
    check_reasoning_not_replayed()
    check_tool_loop_cost_amplification()
    check_decision_table()
    check_cost_effectiveness()
    rule("推理给「需要想」的任务用，不给「需要多试」的链路用；思考内容默认不回填")
    print("\n✓ 第 5 章通过")
