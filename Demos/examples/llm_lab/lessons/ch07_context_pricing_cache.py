"""第 7 章实验：上下文窗口、定价与缓存。

这个模块解决什么问题：第 7 章讲的是三个账单层的硬问题——一次调用到底花多少钱、
缓存怎么命中、上下文塞不下的时候你怎么知道。这些结论必须能算出来，所以本章把
十件事变成可执行的断言：

1. **窗口 vs 有效窗口**——总窗口是输入与输出共享的额度，输出要预留；
2. **定价结构**——输入单价与输出单价分档，换档位成本就变；
3. **输入输出比**——带历史的场景输入远多于输出，输入占账单的大头；
4. **成本曲线**——同档内线性，跨阶梯跳升；
5. **前缀缓存**——前缀相同才命中，前缀一变成本立刻回升；
6. **缓存 TTL**——过期即失效，缓存不认"我刚才发过一样的话"；
7. **写入费**——短会话开缓存反而更贵，这是本章最大的坑；
8. **批处理折扣**——同样的 token，批处理更便宜；
9. **usage 优先**——估算值不能代替 usage 结算；
10. **两道闸**——预算是你自己设的，限流是供应商设的。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..labkit import (
    PRICING,
    ChatMessage,
    ChatResponse,
    banner,
    ensure,
    estimate_cost,
    estimate_tokens,
    rule,
    scripted_client,
    section,
)


# ---------------------------------------------------------------- 窗口模型


@dataclass(frozen=True)
class ContextBudget:
    """一个模型的上下文预算——总窗口与输出预留。"""

    model: str
    context_window: int
    """总窗口：输入和输出**共享**的额度，不是输入独占。"""

    reserved_output_tokens: int
    """为输出预留的额度——回答本身要占的地方。"""

    @property
    def effective_input_window(self) -> int:
        """有效输入窗口 = 总窗口 − 输出预留。

        规划上下文时该按这个数算，而不是按总窗口算——
        否则留给输出的空间会被输入吃光。
        """
        return self.context_window - self.reserved_output_tokens


@dataclass(frozen=True)
class FitReport:
    """一次请求能不能装进窗口的判定结果。"""

    prompt_tokens: int
    desired_output_tokens: int
    effective_input_window: int
    status: str
    """``fits`` / ``tight`` / ``output_truncated`` / ``input_overflow``。"""

    @property
    def fits(self) -> bool:
        """只有 ``fits`` 才算装得下。"""
        return self.status == "fits"


def check_fit(
    prompt_tokens: int,
    desired_output_tokens: int,
    budget: ContextBudget,
) -> FitReport:
    """判断一次请求会不会撑爆窗口——协议层能做的全部判断。

    区分四种状态而不是简单的能/不能，是因为它们对应完全不同的处置：
    ``tight`` 还能发但输出有风险，``output_truncated`` 一定会被截断，
    ``input_overflow`` 根本发不出去。
    """
    effective = budget.effective_input_window
    if prompt_tokens > budget.context_window:
        status = "input_overflow"
    elif prompt_tokens + desired_output_tokens > budget.context_window:
        status = "output_truncated"
    elif prompt_tokens + desired_output_tokens > effective:
        status = "tight"
    else:
        status = "fits"
    return FitReport(
        prompt_tokens=prompt_tokens,
        desired_output_tokens=desired_output_tokens,
        effective_input_window=effective,
        status=status,
    )


def check_context_window() -> None:
    """有效窗口 = 总窗口 − 输出预留；预留不足要有可测的信号。"""
    section("上下文窗口：输入和输出共享同一份额度")

    budget = ContextBudget(
        model="standard",
        context_window=128_000,
        reserved_output_tokens=8_000,
    )

    ensure(
        budget.effective_input_window == 120_000,
        f"有效输入窗口应当是总窗口减输出预留，实际 {budget.effective_input_window}",
    )
    ensure(
        budget.effective_input_window < budget.context_window,
        "有效输入窗口必须小于总窗口——输出也要占额度",
    )

    # 四种状态各有各的处置方式
    ok = check_fit(100_000, 4_000, budget)
    tight = check_fit(118_000, 8_000, budget)
    truncated = check_fit(124_000, 8_000, budget)
    overflow = check_fit(130_000, 8_000, budget)

    for report in (ok, tight, truncated, overflow):
        print(f"  输入 {report.prompt_tokens:>7} + 输出 {report.desired_output_tokens:>6} → {report.status}")

    ensure(ok.status == "fits", f"留足预留的请求应当装得下，实际 {ok.status}")
    ensure(
        tight.status == "tight",
        f"超出有效窗口但没超总窗口应当判为 tight，实际 {tight.status}",
    )
    ensure(
        truncated.status == "output_truncated",
        f"输入加输出超总窗口必然截断，实际 {truncated.status}",
    )
    ensure(
        overflow.status == "input_overflow",
        f"输入本身超总窗口请求发不出去，实际 {overflow.status}",
    )

    # 关键判断：装得下总窗口 ≠ 装得下有效窗口
    ensure(
        tight.prompt_tokens + tight.desired_output_tokens <= budget.context_window,
        "tight 场景应当仍然装得进总窗口，否则就该判成截断",
    )
    ensure(
        tight.prompt_tokens + tight.desired_output_tokens > budget.effective_input_window,
        "tight 场景应当超出有效窗口——这正是预留不足的代价",
    )
    ensure(not tight.fits, "tight 状态不能算作装得下")
    print("  → 协议层能提前告诉你会截断，别等 finish_reason == 'length' 才发现")


# ---------------------------------------------------------------- 定价


@dataclass(frozen=True)
class PriceBand:
    """一个按上下文长度划的价格档。"""

    max_input_tokens: int
    """这一档覆盖到的最大输入 token 数。"""

    input_price: float
    """输入单价，美元 / 每 100 万 token。"""

    output_price: float
    """输出单价，美元 / 每 100 万 token。"""


# 示意价格（美元 / 每 100 万 token）：真实厂商按"上下文长度"分档，
# 数值随调价会变，教学用途请以厂商官网当日价格为准。
PRICE_BANDS: tuple[PriceBand, ...] = (
    PriceBand(128_000, 2.50, 10.00),
    PriceBand(200_000, 5.00, 15.00),
    PriceBand(1 << 30, 10.00, 30.00),
)


def band_for(prompt_tokens: int) -> PriceBand:
    """输入长度落在哪一档。"""
    for band in PRICE_BANDS:
        if prompt_tokens <= band.max_input_tokens:
            return band
    return PRICE_BANDS[-1]


def tiered_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """按输入长度落档计价——整个请求按它落进的那一档计价。"""
    band = band_for(prompt_tokens)
    return (
        prompt_tokens / 1_000_000 * band.input_price
        + completion_tokens / 1_000_000 * band.output_price
    )


def check_pricing_tiers() -> None:
    """换档位成本变；输出比输入贵。"""
    section("定价结构：输入单价和输出单价不是一回事")

    prompt_tokens, completion_tokens = 10_000, 500

    cheap = estimate_cost("cheap", prompt_tokens, completion_tokens)
    standard = estimate_cost("standard", prompt_tokens, completion_tokens)
    frontier = estimate_cost("frontier", prompt_tokens, completion_tokens)

    ensure(
        cheap < standard < frontier,
        f"档位越高成本应当越高，实际 cheap={cheap} standard={standard} frontier={frontier}",
    )

    # 同样多的 token，从输入挪到输出，总价会涨
    all_input = estimate_cost("standard", prompt_tokens, 0)
    all_output = estimate_cost("standard", 0, prompt_tokens)
    ensure(
        all_output > all_input,
        f"输出单价应当高于输入单价，实际全输入 {all_input} 全输出 {all_output}",
    )

    for tier, (input_price, output_price) in PRICING.items():
        ensure(
            output_price >= input_price * 2,
            f"{tier} 档输出单价应当至少是输入的 2 倍，实际 {output_price} / {input_price}",
        )

    print(f"  同样 10000 输入 + 500 输出：cheap={cheap:.6f} standard={standard:.6f} frontier={frontier:.6f}")
    print(f"  同样 10000 token：全输入 {all_input:.6f} vs 全输出 {all_output:.6f} 美元")
    print("  → 输出贵是因为生成比理解更费算力，差额就是模型的'思考税'")


def check_cost_linearity() -> None:
    """同档内线性，跨阶梯跳升。"""
    section("成本曲线：同档内线性，跨阶梯跳升")

    # 同档内：token 翻倍 → 成本翻倍
    base_prompt, base_output = 40_000, 1_000
    cost_1 = tiered_cost(base_prompt, base_output)
    cost_2 = tiered_cost(base_prompt * 2, base_output * 2)
    ensure(
        abs(cost_2 / cost_1 - 2.0) < 1e-9,
        f"同档内 token 翻倍成本应当翻倍，实际比例 {cost_2 / cost_1}",
    )
    ensure(
        band_for(base_prompt * 2).max_input_tokens == band_for(base_prompt).max_input_tokens,
        "线性示例必须落在同一档内，否则测的就不是线性",
    )

    # 跨阶梯：单价更高，曲线出现跳升
    below = tiered_cost(120_000, 500)
    above = tiered_cost(140_000, 500)
    ensure(
        band_for(120_000) is not band_for(140_000),
        "跨越示例必须落在不同档，否则测不出阶梯",
    )
    ensure(
        above > below * (140_000 / 120_000),
        f"跨档后应当比线性外推更贵，实际 {above} vs 线性外推 {below * 140_000 / 120_000}",
    )

    # 两套价格表在第一档上必须对得上，否则估算和账单永远对不上
    ensure(
        abs(tiered_cost(1_000, 100) - estimate_cost("standard", 1_000, 100)) < 1e-12,
        "阶梯表第一档应当与 PRICING 的 standard 档一致",
    )

    print(f"  同档内：{base_prompt}→{base_prompt * 2} token，成本 {cost_1:.6f}→{cost_2:.6f}")
    print(f"  跨阶梯：120k→140k token，成本 {below:.6f}→{above:.6f}（线性外推只要 {below * 140_000 / 120_000:.6f}）")
    print("  → 上下文长度的钱是线性的，但单价会跳档；跳档点就在档位阈值上")


def check_input_output_ratio() -> None:
    """带历史的场景输入远多于输出，输入占账单大头。"""
    section("输入输出比：账单的大头在输入那一侧")

    client = scripted_client(
        responses=[
            ChatResponse(
                content=f"第 {turn} 轮回答",
                usage={"prompt_tokens": 600 + 400 * turn, "completion_tokens": 40},
            )
            for turn in range(1, 9)
        ]
    )

    history: list[ChatMessage] = []
    prompt_total = 0
    completion_total = 0
    per_turn_prompt: list[int] = []

    for _ in range(8):
        history.append(ChatMessage(role="user", content="继续"))
        response = client.complete(history)
        history.append(ChatMessage(role="assistant", content=response.content))
        prompt_total += response.usage["prompt_tokens"]
        completion_total += response.usage["completion_tokens"]
        per_turn_prompt.append(response.usage["prompt_tokens"])

    ratio = prompt_total / completion_total
    input_cost = estimate_cost("standard", prompt_total, 0)
    output_cost = estimate_cost("standard", 0, completion_total)
    input_share = input_cost / (input_cost + output_cost)

    ensure(
        ratio > 10,
        f"带历史的场景输入应当远多于输出，实际比值 {ratio:.1f}",
    )
    ensure(
        input_share > 0.8,
        f"输入成本应当占账单大头，实际占比 {input_share:.1%}",
    )
    ensure(
        per_turn_prompt == sorted(per_turn_prompt) and per_turn_prompt[0] < per_turn_prompt[-1],
        f"每轮输入 token 应当随历史增长，实际 {per_turn_prompt}",
    )

    print(f"  8 轮累计：输入 {prompt_total} token / 输出 {completion_total} token = {ratio:.1f} 倍")
    print(f"  成本构成：输入 {input_cost:.6f} 美元（{input_share:.1%}）/ 输出 {output_cost:.6f} 美元")
    print("  → 先砍输入，不是先砍输出")


# ---------------------------------------------------------------- 缓存


@dataclass
class PrefixCache:
    """前缀缓存的最小可测模型。

    只保留三条真会影响账单的规则：

    1. **命中靠前缀相同**——用前缀内容算 key，key 一样才算命中；
    2. **命中按折扣价，未命中按写入价**——写入价比全价还贵；
    3. **TTL 过期即失效**。
    """

    read_multiplier: float = 0.10
    """命中部分的单价倍率（相对全价）。"""

    write_multiplier: float = 1.25
    """写入缓存的单价倍率（相对全价）——写入是要额外付钱的。"""

    ttl_seconds: float = 300.0
    """缓存有效期。"""

    min_cacheable_tokens: int = 1024
    """低于这个长度的前缀根本不会被缓存。"""

    entries: dict[str, float] = field(default_factory=dict)
    """prefix_key → 过期时刻。"""

    def lookup(self, prefix_key: str, now: float) -> str:
        """查缓存：``hit`` / ``miss`` / ``expired``。"""
        expire_at = self.entries.get(prefix_key)
        if expire_at is None:
            return "miss"
        if now >= expire_at:
            del self.entries[prefix_key]
            return "expired"
        return "hit"

    def write(self, prefix_key: str, now: float) -> None:
        """写入缓存并起算 TTL。"""
        self.entries[prefix_key] = now + self.ttl_seconds


@dataclass(frozen=True)
class Bill:
    """一次调用的账单。"""

    prompt_tokens: int
    completion_tokens: int
    prefix_tokens: int
    input_cost: float
    output_cost: float
    cache_status: str
    """``hit`` / ``miss`` / ``expired`` / ``too_short`` / ``disabled``。"""

    @property
    def total_cost(self) -> float:
        """总成本 = 输入 + 输出。"""
        return self.input_cost + self.output_cost


def prefix_key(
    system_prompt: str,
    tools: list[dict[str, object]] | None = None,
    documents: str = "",
) -> str:
    """用前缀内容算缓存 key——命中与否全看它是否一致。

    系统提示、工具声明、检索文档这些"每次都一样"的部分拼起来就是前缀。
    """
    payload = json.dumps(
        {"system": system_prompt, "tools": tools or [], "documents": documents},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bill_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prefix_tokens: int,
    key: str,
    cache: PrefixCache | None = None,
    now: float = 0.0,
) -> Bill:
    """算一次调用的账——命中与否走完全不同的单价。

    ``prefix_tokens`` 是前缀占的 token 数，剩下的算"尾巴"（本次才有的内容）。
    """
    input_price, output_price = PRICING[model]
    prefix_tokens = min(prefix_tokens, prompt_tokens)
    tail_tokens = prompt_tokens - prefix_tokens
    output_cost = completion_tokens / 1_000_000 * output_price

    if cache is None:
        return Bill(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefix_tokens=prefix_tokens,
            input_cost=prompt_tokens / 1_000_000 * input_price,
            output_cost=output_cost,
            cache_status="disabled",
        )

    status = cache.lookup(key, now)
    if prefix_tokens < cache.min_cacheable_tokens:
        # 前缀太短：厂商根本不给缓存，开了也白开——按全价走
        status = "too_short"
        input_cost = prompt_tokens / 1_000_000 * input_price
    elif status == "hit":
        input_cost = (
            prefix_tokens / 1_000_000 * input_price * cache.read_multiplier
            + tail_tokens / 1_000_000 * input_price
        )
    else:
        # 没命中就要写：前缀部分按写入价付，比全价还贵
        input_cost = (
            prefix_tokens / 1_000_000 * input_price * cache.write_multiplier
            + tail_tokens / 1_000_000 * input_price
        )
        cache.write(key, now)

    return Bill(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prefix_tokens=prefix_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_status=status,
    )


SYSTEM_PROMPT = "你是订单助手，回答不超过两句话。"
TOOL_DECLS: list[dict[str, object]] = [
    {"name": "查订单", "description": "按订单号查物流状态"},
]
DOCUMENTS = "订单 A-1001 已于今日发出，承运商顺丰。"


def check_prefix_cache() -> None:
    """前缀相同才命中；前缀变了成本立刻回升。"""
    section("前缀缓存：命中靠的是前缀一模一样")

    cache = PrefixCache()
    key_same = prefix_key(SYSTEM_PROMPT, TOOL_DECLS, DOCUMENTS)
    key_changed = prefix_key(SYSTEM_PROMPT + "今天优先处理加急件。", TOOL_DECLS, DOCUMENTS)

    ensure(
        key_same == prefix_key(SYSTEM_PROMPT, TOOL_DECLS, DOCUMENTS),
        "前缀内容相同就应当算出同一个 key",
    )
    ensure(
        key_same != key_changed,
        "前缀内容变了就应当算出不同的 key",
    )

    prompt_tokens, completion_tokens = 6_000, 300
    prefix_tokens = 4_000

    first = bill_call(
        "standard", prompt_tokens, completion_tokens, prefix_tokens, key_same, cache, now=0.0
    )
    second = bill_call(
        "standard", prompt_tokens, completion_tokens, prefix_tokens, key_same, cache, now=10.0
    )
    changed = bill_call(
        "standard", prompt_tokens, completion_tokens, prefix_tokens, key_changed, cache, now=20.0
    )

    print(f"  首次（写入）{first.cache_status:<8} 输入 {first.input_cost:.6f}")
    print(f"  再来一次（命中）{second.cache_status:<8} 输入 {second.input_cost:.6f}")
    print(f"  改了前缀（不中）{changed.cache_status:<8} 输入 {changed.input_cost:.6f}")

    ensure(first.cache_status == "miss", f"首次调用应当写入缓存，实际 {first.cache_status}")
    ensure(second.cache_status == "hit", f"前缀相同应当命中，实际 {second.cache_status}")
    ensure(
        second.input_cost < first.input_cost,
        f"命中后输入成本应当下降，实际命中 {second.input_cost} vs 写入 {first.input_cost}",
    )
    ensure(
        changed.cache_status == "miss",
        f"前缀变了就不该命中，实际 {changed.cache_status}",
    )
    ensure(
        changed.input_cost > second.input_cost,
        f"前缀变了成本应当回升，实际 {changed.input_cost} vs 命中 {second.input_cost}",
    )
    ensure(
        changed.input_cost == first.input_cost,
        "不同的前缀应当同样按写入价计费",
    )

    # 命中时前缀部分确实按折扣价算了
    input_price, _ = PRICING["standard"]
    expected_hit = (
        prefix_tokens / 1_000_000 * input_price * cache.read_multiplier
        + (prompt_tokens - prefix_tokens) / 1_000_000 * input_price
    )
    ensure(
        abs(second.input_cost - expected_hit) < 1e-12,
        f"命中时前缀应当按折扣价计费，期望 {expected_hit}，实际 {second.input_cost}",
    )
    print("  → 命中省的是输入单价，输出那部分一分钱都不省")


def check_cache_ttl() -> None:
    """TTL 过期即失效。"""
    section("缓存 TTL：过期就不再命中")

    cache = PrefixCache(ttl_seconds=300.0)
    key = prefix_key(SYSTEM_PROMPT, TOOL_DECLS, DOCUMENTS)

    first = bill_call("standard", 6_000, 300, 4_000, key, cache, now=0.0)
    within = bill_call("standard", 6_000, 300, 4_000, key, cache, now=299.0)
    after = bill_call("standard", 6_000, 300, 4_000, key, cache, now=301.0)

    print(f"  t=0    {first.cache_status}")
    print(f"  t=299  {within.cache_status}")
    print(f"  t=301  {after.cache_status}")

    ensure(within.cache_status == "hit", f"TTL 内应当命中，实际 {within.cache_status}")
    ensure(after.cache_status == "expired", f"TTL 过期应当失效，实际 {after.cache_status}")
    ensure(
        after.input_cost > within.input_cost,
        f"过期后成本应当回到写入价，实际 {after.input_cost} vs 命中 {within.input_cost}",
    )
    ensure(
        after.input_cost == first.input_cost,
        "过期后应当和首次一样按写入价计费",
    )
    print("  → 缓存不认'我刚才发过一样的话'，它只认 TTL 里的前缀")


def check_cache_write_cost() -> None:
    """写入费会让短会话更贵——本章最大的坑。"""
    section("坑：短会话开缓存反而更贵")

    prompt_tokens, completion_tokens = 6_000, 300
    prefix_tokens = 4_000

    # 对照组：完全不用缓存
    no_cache = bill_call("standard", prompt_tokens, completion_tokens, prefix_tokens, "k")

    # 只调一次模型：写入费一分没省
    cache = PrefixCache()
    single = bill_call(
        "standard", prompt_tokens, completion_tokens, prefix_tokens, "k", cache, now=0.0
    )
    ensure(
        single.total_cost > no_cache.total_cost,
        f"只调一次时缓存应当更贵，实际缓存 {single.total_cost} vs 不缓存 {no_cache.total_cost}",
    )
    ensure(
        single.input_cost > no_cache.input_cost,
        "写入费高于全价，单次调用时没有任何节省可摊销",
    )

    # 前缀太短：厂商根本不缓存，开了也白开
    short_cache = PrefixCache(min_cacheable_tokens=1024)
    short = bill_call(
        "standard", 2_000, 300, 512, "k", short_cache, now=0.0
    )
    ensure(
        short.cache_status == "too_short",
        f"前缀过短应当不缓存，实际 {short.cache_status}",
    )
    ensure(
        abs(short.total_cost - estimate_cost("standard", 2_000, 300)) < 1e-12,
        "前缀过短时应当按全价计费，不多不少",
    )

    # 两次调用但中间过期：写了两回，亏得更多
    expiring = PrefixCache(ttl_seconds=60.0)
    call_1 = bill_call("standard", prompt_tokens, completion_tokens, prefix_tokens, "k", expiring, now=0.0)
    call_2 = bill_call("standard", prompt_tokens, completion_tokens, prefix_tokens, "k", expiring, now=120.0)
    baseline = 2 * no_cache.total_cost
    ensure(
        call_1.total_cost + call_2.total_cost > baseline,
        f"两次都没命中时缓存应当比不缓存贵，实际 {call_1.total_cost + call_2.total_cost} vs {baseline}",
    )

    # 前提相反的场景：前缀稳定、多次复用——这才是缓存划算的形态
    stable = PrefixCache()
    cached_total = 0.0
    for turn in range(8):
        cached_total += bill_call(
            "standard", prompt_tokens, completion_tokens, prefix_tokens, "k", stable, now=float(turn)
        ).total_cost
    baseline_8 = 8 * no_cache.total_cost
    ensure(
        cached_total < baseline_8,
        f"前缀稳定时缓存应当明显更便宜，实际缓存 {cached_total} vs 不缓存 {baseline_8}",
    )

    print(f"  不缓存一次      {no_cache.total_cost:.6f}")
    print(f"  开缓存只调一次  {single.total_cost:.6f}（写入费没摊销）")
    print(f"  前缀太短（512）  {short.total_cost:.6f}（压根不缓存）")
    print(f"  两次都过期      {call_1.total_cost + call_2.total_cost:.6f} vs 不缓存 {baseline:.6f}")
    print(f"  前缀稳定 8 次   {cached_total:.6f} vs 不缓存 {baseline_8:.6f}（这才划算）")
    print("  → 缓存不是免费开关：会话太短，写入费就是净亏")


BATCH_DISCOUNT = 0.5
"""批处理折扣：同样的 token，批处理只按在线价格的 50% 收。"""


def check_batch_discount() -> None:
    """可以等的任务，批处理更便宜。"""
    section("批处理折扣：同样的 token 更便宜")

    prompt_tokens, completion_tokens = 20_000, 800
    online_input = estimate_cost("standard", prompt_tokens, 0)
    online_output = estimate_cost("standard", 0, completion_tokens)
    online = online_input + online_output
    batch = online * BATCH_DISCOUNT

    ensure(
        batch < online,
        f"批处理应当比在线便宜，实际批处理 {batch} vs 在线 {online}",
    )
    ensure(
        abs(online / batch - 2.0) < 1e-9,
        f"折扣比例应当是 50%，实际 {online / batch}",
    )

    # 折扣对输入输出一视同仁——不偏袒某一侧，账才算得清
    batch_input = online_input * BATCH_DISCOUNT
    batch_output = online_output * BATCH_DISCOUNT
    ensure(
        abs((batch_input + batch_output) - batch) < 1e-12,
        "批处理总价应当等于输入折扣价加输出折扣价",
    )
    ensure(
        abs(batch_input / batch_output - online_input / online_output) < 1e-9,
        "折扣不改变输入输出的成本结构",
    )

    print(f"  在线 {online:.6f} vs 批处理 {batch:.6f} 美元（同量 token）")
    print("  → 折扣拿延迟换：对用户等着看的请求不划算")


# ---------------------------------------------------------------- usage 与两道闸


def check_usage_vs_estimate() -> None:
    """估算不能代替 usage 结算。"""
    section("别用估算代替 usage")

    text = "上下文窗口决定了一次调用能塞多少东西进去，定价决定这些东西值多少钱。"
    estimated = estimate_tokens(text)

    # 脚本扮演厂商的 tokenizer：它数出来的和你的估算不一样
    vendor_a = scripted_client(
        responses=[
            ChatResponse(
                content="回答",
                usage={"prompt_tokens": estimated - 9, "completion_tokens": 12},
            )
        ]
    )
    vendor_b = scripted_client(
        responses=[
            ChatResponse(
                content="回答",
                usage={"prompt_tokens": estimated + 17, "completion_tokens": 12},
            )
        ]
    )

    response_a = vendor_a.complete([ChatMessage(role="user", content=text)])
    response_b = vendor_b.complete([ChatMessage(role="user", content=text)])

    real_a = response_a.usage["prompt_tokens"]
    real_b = response_b.usage["prompt_tokens"]

    ensure(
        real_a != estimated,
        f"估算值和 usage 不可能总是相等，估算 {estimated} 实际 {real_a}",
    )
    ensure(
        real_b != real_a,
        f"不同厂商对同一段文本的 token 数不同，实际 {real_a} vs {real_b}",
    )

    deviation = abs(real_a - estimated) / real_a
    cost_from_estimate = estimate_cost("standard", estimated, 12)
    cost_from_usage = estimate_cost("standard", real_a, 12)
    cost_error = abs(cost_from_estimate - cost_from_usage) / cost_from_usage

    ensure(
        deviation > 0.05,
        f"估算偏差应当能观察到 5% 以上，实际 {deviation:.1%}",
    )
    ensure(
        cost_error > 0.05,
        f"用估算结算会让账单偏差超过 5%，实际 {cost_error:.1%}",
    )

    print(f"  同一段文本：估算 {estimated} token / 厂商 A {real_a} / 厂商 B {real_b}")
    print(f"  按估算结算 {cost_from_estimate:.6f} vs 按 usage 结算 {cost_from_usage:.6f}")
    print("  → 估算只用来事前估大小，结算一律以 usage 为准")


@dataclass
class BudgetGuard:
    """你自己设的预算闸——超了就停。"""

    limit_usd: float
    spent_usd: float = 0.0

    def charge(self, cost: float) -> bool:
        """记一笔账；返回 False 表示这一笔超预算，该停了。"""
        if self.spent_usd + cost > self.limit_usd:
            return False
        self.spent_usd += cost
        return True


@dataclass
class RateLimiter:
    """供应商设的限流闸——它管的是节奏，不是钱。"""

    requests_per_minute: int
    used: int = 0

    def allow(self) -> bool:
        """这一分钟还放不放行。"""
        if self.used >= self.requests_per_minute:
            return False
        self.used += 1
        return True


def check_budget_and_rate_limit() -> None:
    """预算是自己设的，限流是供应商设的。"""
    section("账单前面有两道闸：预算和限流")

    budget = BudgetGuard(limit_usd=0.10)
    ensure(budget.charge(0.04), "预算之内应当放行")
    ensure(budget.charge(0.05), "预算之内应当放行")
    ensure(
        not budget.charge(0.02),
        "超出预算应当拦住——预算是你自己设的那道闸",
    )
    ensure(
        abs(budget.spent_usd - 0.09) < 1e-12,
        f"被拦下的账不该计入已花额度，实际 {budget.spent_usd}",
    )

    limiter = RateLimiter(requests_per_minute=3)
    allowed = [limiter.allow() for _ in range(5)]
    ensure(
        allowed == [True, True, True, False, False],
        f"限流应当放行前 3 次、拦住后 2 次，实际 {allowed}",
    )

    # 两道闸互不替代：限流拦住的是"贵不贵无关"的请求
    ensure(
        not limiter.allow() and budget.charge(0.0001),
        "限流和预算是两回事：被限流的请求可能很便宜，没超预算的请求也可能超限流",
    )
    print(f"  预算：0.10 美元上限，已花 {budget.spent_usd:.4f}，下一笔被拦")
    print("  限流：每分钟 3 次，第 4 次起被拦")
    print("  → 预算是你对自己设的闸，限流是供应商设的闸，两道都得留")


def run_live() -> None:
    """真实模型路径：真调一次模型，用 usage 结算一次账。"""
    banner("第 7 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client()
    response = client.complete(
        [
            ChatMessage(role="system", content="你是订单助手，回答不超过两句话。"),
            ChatMessage(role="user", content="用一句话说明什么是上下文窗口。"),
        ],
        temperature=0.0,
    )

    prompt_tokens = response.usage.get("prompt_tokens", 0)
    completion_tokens = response.usage.get("completion_tokens", 0)
    # 真实端点的模型名不是价格档位，这里统一按 standard 档折算，
    # 实际单价以厂商账单为准
    cost = estimate_cost("standard", prompt_tokens, completion_tokens)

    print(f"  content: {response.content!r}")
    print(f"  usage: {response.usage}")
    print(f"  按 standard 档折算：{cost:.6f} 美元")

    ensure(bool(response.content), "真实模型应当返回内容")
    ensure(prompt_tokens > 0, "usage 应当报告输入 token")
    ensure(cost > 0, "有 token 用量就应当有成本")
    ensure(
        response.finish_reason in {"stop", "length"},
        f"finish_reason 异常：{response.finish_reason}",
    )


def run() -> None:
    """跑第 7 章实验。"""
    banner("第 7 章 · 上下文窗口、定价与缓存")
    check_context_window()
    check_pricing_tiers()
    check_cost_linearity()
    check_input_output_ratio()
    check_prefix_cache()
    check_cache_ttl()
    check_cache_write_cost()
    check_batch_discount()
    check_usage_vs_estimate()
    check_budget_and_rate_limit()
    rule(
        "窗口是输入输出共享的额度，账单大头在输入那一侧；"
        "缓存只认前缀和 TTL，写入费会让短会话净亏；结算一律以 usage 为准"
    )
    print("\n✓ 第 7 章通过")
