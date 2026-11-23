"""第 8 章实验：模型选型与迁移纪律。

这个模块解决什么问题：第 8 章讲两件容易出事故的事——**给任务选模型**，
以及**换模型/升版本时不把线上搞坏**。这两件事都必须能判定，所以本章把九件事
变成可执行的断言：

1. **三维选型矩阵**——能力、延迟、成本三个维度都要摆出来，档位由任务难度决定；
2. **分级路由的成本**——分流后总成本必须低于全走标准档，且便宜档在它负责的
   那类任务上成功率不降；
3. **版本 pin 死**——请求体里的 ``model`` 是构造时定下的常量，不会随调用漂移；
4. **迁移清单逐项可判定**——七类破坏性变化能被一张清单全部检出；
5. **JSON 严格性差异**——同一份 schema，严格模型直接过，啰嗦模型直接坏；
6. **上下文长度差异**——同一份长历史，新模型接受、旧模型拒绝；
7. **流式事件形状差异**——两个模型的流式脚本形状不同，消费代码必须分别处理；
8. **评测集不许动**——迁移前后用同一份评测集打分才可比，改了评测集分数就虚高；
9. **决策表可执行**——场景 → 该不该换模型 / 该不该升版本。

另外补三件收尾的事：影子流量怎么比对、迁移流程的步骤顺序、以及"选型不是
一次性的"（价格会漂移，公开榜单靠不住）。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..labkit import (
    PRICING,
    ChatMessage,
    ChatResponse,
    OpenAICompatibleClient,
    StreamChunk,
    banner,
    ensure,
    estimate_cost,
    extract_json,
    rule,
    scripted_client,
    section,
    validate_against_schema,
)


# ---------------------------------------------------------------- 1. 三维选型矩阵


@dataclass(frozen=True)
class TierProfile:
    """一个模型档位在三维选型矩阵里的位置。

    三个维度缺一不可：只比价格会选到太弱的，只比能力会选到太贵的，
    只比延迟会选到干不了活的。
    """

    tier: str
    capability: int
    """能力分 1–5：这个档位能稳定完成什么难度的任务。"""

    ttft_ms: int
    """首字延迟（毫秒）——对话类场景的硬指标。"""

    @property
    def input_price(self) -> float:
        """每百万输入 token 的价格（美元）。"""
        return PRICING[self.tier][0]

    @property
    def output_price(self) -> float:
        """每百万输出 token 的价格（美元）。"""
        return PRICING[self.tier][1]

    @property
    def blended_price(self) -> float:
        """按输入:输出 = 4:1 折算的单价——横向比档位时用它。"""
        return self.input_price * 0.8 + self.output_price * 0.2


TIERS: dict[str, TierProfile] = {
    "cheap-small": TierProfile("cheap-small", capability=2, ttft_ms=250),
    "cheap": TierProfile("cheap", capability=3, ttft_ms=400),
    "standard": TierProfile("standard", capability=4, ttft_ms=800),
    "frontier": TierProfile("frontier", capability=5, ttft_ms=1500),
    "reasoning": TierProfile("reasoning", capability=5, ttft_ms=2500),
}

# 能力匹配顺序：从便宜到贵。推理档不在这条线上——
# 它按"要不要思考预算"单独选（第 5 章），不是"更贵的通用档"。
CAPABILITY_ORDER: tuple[str, ...] = ("cheap-small", "cheap", "standard", "frontier")


@dataclass(frozen=True)
class TaskType:
    """一类任务的画像——难度与能力要求决定它该落到哪个档位。"""

    name: str
    required_capability: int
    """完成这类任务至少需要的档位能力分。"""

    needs_thinking_budget: bool = False
    """要不要思考预算——多步推理的题不该用"再试一次"的档位换。"""

    example: str = ""


TASK_TYPES: tuple[TaskType, ...] = (
    TaskType("分类", 2, example="意图识别、请求分流"),
    TaskType("抽取", 3, example="从工单里取订单号和金额"),
    TaskType("改写", 3, example="把口语改写成工单描述"),
    TaskType("常规问答", 4, example="知识问答、解释概念"),
    TaskType("工具调用", 4, example="查订单、调用外部 API"),
    TaskType("多步推理", 5, example="多条件约束下的计算与排错"),
    TaskType("长程规划", 5, needs_thinking_budget=True, example="多步任务分解与取舍"),
)

TASK_BY_NAME: dict[str, TaskType] = {t.name: t for t in TASK_TYPES}


def select_tier(task: TaskType) -> str:
    """按任务画像选档位：能便宜就用便宜的，要思考预算才上推理档。

    规则只有两条：

    1. 需要思考预算 → 推理档；
    2. 否则选**能力够用的最便宜档位**——不是"最好的那个"。
    """
    if task.needs_thinking_budget:
        return "reasoning"
    for tier in CAPABILITY_ORDER:
        if TIERS[tier].capability >= task.required_capability:
            return tier
    return "frontier"


def check_selection_matrix() -> None:
    """三维选型矩阵可执行：每类任务都落在正确档位上。"""
    section("三维选型矩阵：能力 / 延迟 / 成本")

    expected = {
        "分类": "cheap-small",
        "抽取": "cheap",
        "改写": "cheap",
        "常规问答": "standard",
        "工具调用": "standard",
        "多步推理": "frontier",
        "长程规划": "reasoning",
    }

    for task in TASK_TYPES:
        tier = select_tier(task)
        profile = TIERS[tier]
        print(
            f"  {task.name:<6} 需要能力 {task.required_capability} → {tier:<12}"
            f" 单价 {profile.blended_price:>6.2f} / TTFT {profile.ttft_ms:>5}ms"
            f"  （{task.example}）"
        )
        ensure(
            tier == expected[task.name],
            f"{task.name} 应当落在 {expected[task.name]} 档，实际落在 {tier} 档",
        )

    # 选型必须是"最便宜够用"：便宜档够用就不许往上跳
    for task in TASK_TYPES:
        tier = select_tier(task)
        if task.needs_thinking_budget:
            continue
        for cheaper in CAPABILITY_ORDER[: CAPABILITY_ORDER.index(tier)]:
            ensure(
                TIERS[cheaper].capability < task.required_capability,
                f"{task.name} 选了 {tier}，但更便宜的 {cheaper} 能力也够——"
                "选型没有遵守'最便宜够用'的规则",
            )

    # 三个维度必须各自动：能力、单价随档位单调不降，延迟另算
    caps = [TIERS[t].capability for t in CAPABILITY_ORDER]
    prices = [TIERS[t].blended_price for t in CAPABILITY_ORDER]
    ensure(caps == sorted(caps), f"能力分应当随档位单调不降，实际 {caps}")
    ensure(prices == sorted(prices), f"单价应当随档位单调不降，实际 {prices}")

    # 成本轴：frontier 比 cheap 贵一个数量级以上——所以"越贵越好"不成立
    ratio = TIERS["frontier"].blended_price / TIERS["cheap"].blended_price
    ensure(
        ratio > 20,
        f"frontier 应当比 cheap 贵一个数量级以上，实际只贵 {ratio:.1f} 倍",
    )
    print(f"  价格轴：frontier 单价是 cheap 的 {ratio:.1f} 倍——所以不是越贵越好")

    # 延迟轴：推理档要先思考，它不是"更快的档位"
    ensure(
        TIERS["reasoning"].ttft_ms > TIERS["frontier"].ttft_ms,
        f"推理档 TTFT 应当高于前沿档，实际推理 {TIERS['reasoning'].ttft_ms}ms "
        f"vs 前沿 {TIERS['frontier'].ttft_ms}ms",
    )
    ensure(
        TIERS["cheap"].ttft_ms < TIERS["standard"].ttft_ms,
        "便宜档的 TTFT 应当低于标准档",
    )
    print("  延迟轴：推理档 TTFT 最高——贵的那一档不一定更快")


# ---------------------------------------------------------------- 2. 分级路由


@dataclass(frozen=True)
class TrafficItem:
    """一条真实流量的抽象——任务类型 + 规模。"""

    task_type: str
    count: int
    prompt_tokens: int
    completion_tokens: int


TRAFFIC: tuple[TrafficItem, ...] = (
    TrafficItem("分类", 200, 500, 20),
    TrafficItem("抽取", 600, 800, 60),
    TrafficItem("常规问答", 150, 1500, 200),
    TrafficItem("工具调用", 80, 2500, 120),
    # 多步推理的请求占比很低，但它是"贵档"——这一类会吃掉大部分省下来的钱
    TrafficItem("多步推理", 8, 3000, 400),
)

# 各档位在各类任务上的成功率——真实系统里这张表来自评测集，不是拍脑袋
SUCCESS_RATE: dict[tuple[str, str], float] = {
    ("cheap-small", "分类"): 0.98,
    ("standard", "分类"): 0.98,
    ("cheap", "抽取"): 0.99,
    ("standard", "抽取"): 0.99,
    ("standard", "常规问答"): 0.94,
    ("standard", "工具调用"): 0.92,
    ("standard", "多步推理"): 0.70,
    ("frontier", "多步推理"): 0.90,
}


@dataclass(frozen=True)
class RoutePlan:
    """一次分级路由的结果。"""

    tier_by_task: dict[str, str]
    total_cost: float
    """总成本（美元）。"""

    weighted_success: float
    """按请求数加权的成功率。"""


def route_traffic(traffic: Sequence[TrafficItem]) -> RoutePlan:
    """按任务类型分流，算出总成本与加权成功率。"""
    tier_by_task: dict[str, str] = {}
    total_cost = 0.0
    weighted_success = 0.0
    total_requests = 0

    for item in traffic:
        tier = select_tier(TASK_BY_NAME[item.task_type])
        tier_by_task[item.task_type] = tier
        total_cost += item.count * estimate_cost(
            tier, item.prompt_tokens, item.completion_tokens
        )
        weighted_success += SUCCESS_RATE.get((tier, item.task_type), 0.0) * item.count
        total_requests += item.count

    return RoutePlan(
        tier_by_task=tier_by_task,
        total_cost=total_cost,
        weighted_success=weighted_success / total_requests,
    )


def route_all_standard(traffic: Sequence[TrafficItem]) -> RoutePlan:
    """基线：所有请求一律走标准档。"""
    total_cost = 0.0
    weighted_success = 0.0
    total_requests = 0

    for item in traffic:
        total_cost += item.count * estimate_cost(
            "standard", item.prompt_tokens, item.completion_tokens
        )
        weighted_success += SUCCESS_RATE.get(("standard", item.task_type), 0.0) * item.count
        total_requests += item.count

    return RoutePlan(
        tier_by_task={item.task_type: "standard" for item in traffic},
        total_cost=total_cost,
        weighted_success=weighted_success / total_requests,
    )


def check_routing_cost() -> None:
    """分流后成本必须下降，且便宜档在它负责的任务上成功率不降。"""
    section("分级路由：便宜档能不能扛住它负责的活")

    baseline = route_all_standard(TRAFFIC)
    routed = route_traffic(TRAFFIC)

    for item in TRAFFIC:
        tier = routed.tier_by_task[item.task_type]
        print(
            f"  {item.task_type:<6} {item.count:>4} 条 → {tier:<12}"
            f" 单价 {TIERS[tier].blended_price:>6.2f}"
        )

    print(f"  全走标准档：{baseline.total_cost:.4f} 美元，成功率 {baseline.weighted_success:.3f}")
    print(f"  分级路由后：{routed.total_cost:.4f} 美元，成功率 {routed.weighted_success:.3f}")

    ensure(
        routed.total_cost < baseline.total_cost,
        f"分级路由的总成本应当低于全走标准档，实际 {routed.total_cost:.4f} "
        f"vs {baseline.total_cost:.4f}",
    )
    saving = 1 - routed.total_cost / baseline.total_cost
    ensure(saving > 0.3, f"成本下降幅度应当超过 30%，实际只降了 {saving:.1%}")
    print(f"  → 成本降了 {saving:.1%}")

    # 走最贵档位的那类任务是"请求少、钱多"——账单的大头在那儿
    total_requests = sum(item.count for item in TRAFFIC)
    cost_by_task = {
        item.task_type: item.count
        * estimate_cost(
            routed.tier_by_task[item.task_type], item.prompt_tokens, item.completion_tokens
        )
        for item in TRAFFIC
    }
    priciest = max(
        routed.tier_by_task,
        key=lambda name: TIERS[routed.tier_by_task[name]].blended_price,
    )
    request_share = next(i.count for i in TRAFFIC if i.task_type == priciest) / total_requests
    cost_share = cost_by_task[priciest] / routed.total_cost
    print(
        f"  → {priciest}只占 {request_share:.1%} 的请求，却占了 {cost_share:.1%} 的成本"
        "——贵档那一小撮请求才是账单的大头"
    )
    ensure(
        request_share < 0.05,
        f"最贵档位的请求占比应当很低，实际 {request_share:.1%}",
    )
    ensure(
        cost_share > request_share * 5,
        f"最贵档位的成本占比应当远高于请求占比，实际 {cost_share:.1%} vs {request_share:.1%}",
    )

    # 便宜档在它负责的任务上成功率不许比标准档低——这是路由的护栏
    for task_name, tier in routed.tier_by_task.items():
        if tier not in {"cheap", "cheap-small"}:
            continue
        cheap_rate = SUCCESS_RATE[(tier, task_name)]
        standard_rate = SUCCESS_RATE[("standard", task_name)]
        ensure(
            cheap_rate >= standard_rate,
            f"{task_name} 分流到 {tier} 后成功率 {cheap_rate} 低于标准档的 "
            f"{standard_rate}——便宜档在它负责的任务上不能掉质量",
        )
        print(f"  {task_name}: {tier} 成功率 {cheap_rate} ≥ 标准档 {standard_rate} ✓")

    ensure(
        routed.weighted_success >= baseline.weighted_success,
        f"整体成功率不该低于全走标准档，实际 {routed.weighted_success:.3f} "
        f"vs {baseline.weighted_success:.3f}",
    )
    print("  → 分流是省钱，不是赌质量：难的那部分反而换到了更强的档")


# ---------------------------------------------------------------- 3. 版本 pin 死


def validate_pin(model: str) -> str | None:
    """检查模型标识是不是"钉死"的版本。

    返回问题描述；``None`` 表示合格。生产环境不接受浮动别名——
    ``latest`` 意味着今天的请求和昨天的请求可能落在不同的模型上。
    """
    if not model.strip():
        return "模型标识为空"
    lowered = model.lower()
    for floating in ("latest", "preview", "auto", "default"):
        if floating in lowered:
            return f"模型标识 {model!r} 含浮动别名 {floating!r}——生产必须 pin 死版本"
    if not re.search(r"\d{4}-\d{2}(-\d{2})?", model):
        return f"模型标识 {model!r} 里看不到版本号——无法确认线上到底在用哪个版本"
    return None


def check_version_pin() -> None:
    """请求体里的 model 是构造时固定的常量。"""
    section("版本 pin 死：请求里的 model 不会自己漂移")

    pinned = "vendor-x-2025-03-01"
    client = OpenAICompatibleClient(
        model=pinned, base_url="http://localhost:9", api_key="k"
    )

    # 只在本地构造载荷、不发请求——这样才能直接断言请求体里的 model
    models: list[str] = []
    for index in range(3):
        payload = client._payload(
            [ChatMessage(role="user", content=f"第 {index} 个问题")], None, 0.0, None
        )
        models.append(payload["model"])

    print(f"  三次调用请求体里的 model：{models}")
    ensure(
        all(m == pinned for m in models),
        f"请求体里的 model 应当恒等于 {pinned}，实际 {models}",
    )
    ensure(
        client.model == pinned,
        f"客户端持有的 model 应当就是构造时传入的常量，实际 {client.model!r}",
    )

    # 换版本 = 换一个客户端对象，旧客户端不受任何影响
    upgraded = OpenAICompatibleClient(
        model="vendor-x-2025-09-01", base_url="http://localhost:9", api_key="k"
    )
    ensure(
        upgraded._payload([ChatMessage(role="user", content="x")], None, 0.0, None)["model"]
        == "vendor-x-2025-09-01",
        "新客户端应当用新版本号",
    )
    ensure(
        client._payload([ChatMessage(role="user", content="x")], None, 0.0, None)["model"]
        == pinned,
        "新建客户端不应当改动旧客户端——版本切换必须是显式动作",
    )
    print("  → 升版本不会自己发生：换客户端对象才是那个显式动作")

    # 浮动别名必须被判为不合格
    ensure(
        validate_pin(pinned) is None,
        f"带版本号的模型标识应当合格，实际问题是：{validate_pin(pinned)}",
    )
    for floating in ("vendor-x", "vendor-x-latest", "vendor-x-preview"):
        problem = validate_pin(floating)
        ensure(problem is not None, f"{floating!r} 这种标识必须被判为不合格")
    print("  → latest / preview / 无版本号：一律不许进生产")


# ---------------------------------------------------------------- 4. 迁移清单


@dataclass(frozen=True)
class ModelProfile:
    """一个模型的"可观察行为画像"。

    迁移清单比的就是这份画像——不是营销文案，也不是排行榜名次。
    """

    name: str
    version: str
    tool_arguments_type: str
    """``arguments`` 是 JSON 字符串还是已经解析好的对象。"""

    tool_argument_key: str
    """参数字段名：``arguments`` / ``parameters`` / ``input``。"""

    json_strict: bool
    """模型是否严格只吐 JSON。"""

    context_window: int
    stream_granularity: str
    """分片粒度：``char`` / ``word`` / ``block``。"""

    stream_finish_timing: str
    """finish_reason 出现的时机：``last_chunk`` / ``same_chunk``。"""

    has_reasoning_field: bool
    """端点是否返回思考字段。"""

    rate_limit_status: int
    """限流时返回的状态码。"""

    safety_strictness: str
    """内容过滤严格程度：``strict`` / ``lenient``。"""


OLD_PROFILE = ModelProfile(
    name="vendor-x",
    version="2025-03-01",
    tool_arguments_type="json_string",
    tool_argument_key="arguments",
    json_strict=True,
    context_window=8_000,
    stream_granularity="char",
    stream_finish_timing="last_chunk",
    has_reasoning_field=False,
    rate_limit_status=429,
    safety_strictness="strict",
)

NEW_PROFILE = ModelProfile(
    name="vendor-x",
    version="2025-09-01",
    tool_arguments_type="object",
    tool_argument_key="parameters",
    json_strict=False,
    context_window=128_000,
    stream_granularity="block",
    stream_finish_timing="same_chunk",
    has_reasoning_field=True,
    rate_limit_status=503,
    safety_strictness="lenient",
)


@dataclass(frozen=True)
class BreakingChange:
    """一处必须重测的破坏性变化。"""

    category: str
    detail: str
    action: str
    """重测这一项要做什么。"""


def diff_profiles(old: ModelProfile, new: ModelProfile) -> list[BreakingChange]:
    """比对两份画像，列出所有需要重测的破坏性变化。

    每一类变化都对应一次独立的重测——把它们合成"重新跑一遍测试"，
    就等于没有清单。
    """
    changes: list[BreakingChange] = []

    if old.tool_arguments_type != new.tool_arguments_type:
        changes.append(
            BreakingChange(
                "工具调用格式",
                f"arguments 从 {old.tool_arguments_type} 变成 {new.tool_arguments_type}",
                "重跑工具调用用例，确认解析端两种形状都能吃",
            )
        )
    if old.tool_argument_key != new.tool_argument_key:
        changes.append(
            BreakingChange(
                "工具调用格式",
                f"参数字段名从 {old.tool_argument_key} 变成 {new.tool_argument_key}",
                "重跑工具调用用例，确认字段映射没漏",
            )
        )
    if old.json_strict != new.json_strict:
        changes.append(
            BreakingChange(
                "JSON 严格性",
                f"严格遵守 schema 变成 {new.json_strict}",
                "重跑全部结构化输出用例，校验失败率必须统计",
            )
        )
    if old.context_window != new.context_window:
        changes.append(
            BreakingChange(
                "上下文长度",
                f"上下文窗口从 {old.context_window} 变成 {new.context_window}",
                "用最长历史的那条用例重跑，确认不再被拒或被截",
            )
        )
    if (
        old.stream_granularity != new.stream_granularity
        or old.stream_finish_timing != new.stream_finish_timing
    ):
        changes.append(
            BreakingChange(
                "流式事件形状",
                f"分片粒度 {old.stream_granularity}→{new.stream_granularity}，"
                f"finish 时机 {old.stream_finish_timing}→{new.stream_finish_timing}",
                "重跑流式用例，确认拼接器对两种形状都成立",
            )
        )
    if old.has_reasoning_field != new.has_reasoning_field:
        changes.append(
            BreakingChange(
                "思考字段",
                f"是否返回思考字段变成 {new.has_reasoning_field}",
                "重跑思考预算用例，确认解析端能容忍字段缺失",
            )
        )
    if old.rate_limit_status != new.rate_limit_status:
        changes.append(
            BreakingChange(
                "错误码与限流",
                f"限流状态码从 {old.rate_limit_status} 变成 {new.rate_limit_status}",
                "重跑错误分类用例，确认退避策略覆盖新状态码",
            )
        )
    if old.safety_strictness != new.safety_strictness:
        changes.append(
            BreakingChange(
                "安全策略",
                f"内容过滤从 {old.safety_strictness} 变成 {new.safety_strictness}",
                "用被拦过的输入重跑，确认拦截口径没变",
            )
        )

    return changes


def check_migration_checklist() -> None:
    """迁移清单必须能逐项检出破坏性变化。"""
    section("迁移清单：换模型要重测的七件事")

    changes = diff_profiles(OLD_PROFILE, NEW_PROFILE)
    categories = {c.category for c in changes}
    for change in changes:
        print(f"  [{change.category}] {change.detail}")
        print(f"      → {change.action}")

    expected = {
        "工具调用格式",
        "JSON 严格性",
        "上下文长度",
        "流式事件形状",
        "思考字段",
        "错误码与限流",
        "安全策略",
    }
    ensure(
        categories == expected,
        f"清单应当覆盖全部七类破坏性变化，实际检出 {sorted(categories)}",
    )
    ensure(
        len(categories) >= 5,
        f"清单应当至少检出 5 类破坏性变化，实际只检出 {len(categories)} 类",
    )
    print(f"  → 共检出 {len(categories)} 类破坏性变化")

    # 没有变化就不该报警——清单太吵就没人看
    ensure(
        diff_profiles(OLD_PROFILE, OLD_PROFILE) == [],
        "两份相同的画像之间不该检出任何变化",
    )
    print("  → 画像没变时清单保持安静：不误报才有资格当闸门")


# ---------------------------------------------------------------- 5. 工具调用格式


def check_tool_arguments_shape() -> None:
    """arguments 是 JSON 字符串还是对象？换模型最容易在这翻车。"""
    section("工具调用格式：arguments 是字符串还是对象")

    string_client = scripted_client(
        responses=[
            ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "查订单", "arguments": '{"order_id":"A-1001"}'},
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    object_client = scripted_client(
        responses=[
            ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "查订单", "arguments": {"order_id": "A-1001"}},
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )

    messages = [ChatMessage(role="user", content="查 A-1001")]
    string_call = string_client.complete(messages).tool_calls[0]
    object_call = object_client.complete(messages).tool_calls[0]

    def parse_arguments(call: dict) -> dict:
        """很多项目就写了这一行——它只吃字符串。"""
        return json.loads(call["function"]["arguments"])

    ensure(
        parse_arguments(string_call) == {"order_id": "A-1001"},
        "字符串形状的参数应当能直接解析出来",
    )

    broke = False
    try:
        parse_arguments(object_call)
    except TypeError:
        broke = True
    ensure(
        broke,
        "对象形状的参数会让 json.loads 直接抛 TypeError——这就是换模型时的断点",
    )

    def tolerant_parse(call: dict) -> dict:
        """容错解析：两种形状都能吃。"""
        raw = call["function"]["arguments"]
        if isinstance(raw, str):
            return json.loads(raw)
        return dict(raw)

    ensure(
        tolerant_parse(string_call) == tolerant_parse(object_call),
        "容错解析之后，两种形状应当解析出同样的结果",
    )
    print("  → 解析端必须容错：形状差异是厂商的事，不是你的 bug")


# ---------------------------------------------------------------- 6. JSON 严格性


EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}


def check_json_strictness() -> None:
    """同一份 schema，严格模型直接过，啰嗦模型直接坏。"""
    section("JSON 严格性：换模型后结构化输出最容易坏的地方")

    strict_text = '{"city":"杭州","days":3}'
    chatty_text = (
        "好的，结果如下：\n```json\n"
        '{"city":"杭州","days":3}\n'
        "```\n还需要别的城市随时告诉我。"
    )
    extra_field_text = '{"city":"杭州","days":3,"confidence":"高"}'

    strict_obj = json.loads(strict_text)
    strict_errors = validate_against_schema(strict_obj, EXTRACT_SCHEMA)
    chatty_errors = validate_against_schema(chatty_text, EXTRACT_SCHEMA)

    print(f"  严格模型输出：{strict_text!r}")
    print(f"  啰嗦模型输出：{chatty_text!r}")

    ensure(
        strict_errors == [],
        f"严格模型的输出应当通过校验，实际报错：{strict_errors}",
    )
    ensure(
        chatty_errors != [],
        "啰嗦模型的输出应当校验不通过——它不是 JSON 对象",
    )
    ensure(
        (strict_errors == []) != (chatty_errors == []),
        "同一份 schema，两个模型的输出应当出现不同的校验结果",
    )

    # 直接 json.loads 在啰嗦模型上就炸了
    parsed_directly = True
    try:
        json.loads(chatty_text)
    except json.JSONDecodeError:
        parsed_directly = False
    ensure(
        not parsed_directly,
        "啰嗦模型的输出不该能被 json.loads 直接解析——这是换模型后最容易炸的一行",
    )

    # 容错解析能救回来——这才是正确的防线
    recovered = extract_json(chatty_text)
    ensure(
        validate_against_schema(recovered, EXTRACT_SCHEMA) == [],
        "容错解析之后应当能通过校验",
    )
    print("  → 严格模型直接过，啰嗦模型要抠——同一份提示词，换个模型就坏")

    # 合法 JSON ≠ 合法结构：多出来的字段同样要挡
    extra_errors = validate_against_schema(json.loads(extra_field_text), EXTRACT_SCHEMA)
    ensure(
        extra_errors != [],
        "多出来的字段应当被校验挡住——能 parse 不代表能用",
    )
    print("  → 结构化输出必须校验：能 parse 不代表符合 schema")


# ---------------------------------------------------------------- 7. 上下文长度


@dataclass(frozen=True)
class WindowDecision:
    """长历史装不装得下的判定。"""

    accepted: bool
    status: str
    """``fits`` / ``tight`` / ``input_overflow``。"""

    effective_input: int
    """有效输入窗口 = 总窗口 − 输出预留。"""


def judge_window(
    prompt_tokens: int, window: int, reserved_output: int = 2_000
) -> WindowDecision:
    """判断一次请求会不会撑爆窗口。"""
    effective = window - reserved_output
    if prompt_tokens > window:
        return WindowDecision(False, "input_overflow", effective)
    if prompt_tokens > effective:
        return WindowDecision(True, "tight", effective)
    return WindowDecision(True, "fits", effective)


def check_context_length() -> None:
    """同一份长历史，新模型接受、旧模型拒绝。"""
    section("上下文长度：同一份历史，两个模型两个结果")

    long_history_tokens = 40_000
    old_window = 8_000
    new_window = 128_000

    old_decision = judge_window(long_history_tokens, old_window)
    new_decision = judge_window(long_history_tokens, new_window)

    print(f"  历史 {long_history_tokens} token：")
    print(f"    旧模型（窗口 {old_window}）→ {old_decision.status}")
    print(f"    新模型（窗口 {new_window}）→ {new_decision.status}")

    ensure(
        not old_decision.accepted,
        f"旧模型应当拒绝这份超长历史，实际判定 {old_decision.status}",
    )
    ensure(
        old_decision.status == "input_overflow",
        f"旧模型的判定应当是 input_overflow，实际 {old_decision.status}",
    )
    ensure(
        new_decision.accepted,
        f"新模型应当接受这份历史，实际判定 {new_decision.status}",
    )
    ensure(
        old_decision.accepted != new_decision.accepted,
        "同一份长历史，新旧模型的接受与否应当不同",
    )

    # 请求过大是 400 家族——重试一万次也是错的（第 1 章的错误分类）
    def classify(status: int) -> str:
        """把状态码映射成处理策略（第 1 章的四分类）。"""
        if status in (401, 403):
            return "fatal"
        if status == 429:
            return "backoff"
        if status >= 500:
            return "retry"
        return "fix_request"

    ensure(
        classify(400) == "fix_request",
        f"请求过大属于 400 家族，应当判为 fix_request，实际 {classify(400)}",
    )
    ensure(
        classify(400) != classify(429),
        "400 和 429 的处理方式必须不同——超长历史不该退避重试",
    )
    print("  → 请求过大属于 400 家族：改请求，不是退避重试")
    print("  → 更危险的是旧模型静默截断：历史被切掉，答案看着正常但依据不全")


# ---------------------------------------------------------------- 8. 流式形状


@dataclass(frozen=True)
class StreamResult:
    """消费流式分片后的结果——形状归一化之后的产物。"""

    text: str
    finish_reason: str | None
    total_tokens: int
    chunk_count: int


def consume_robust(chunks: Sequence[StreamChunk]) -> StreamResult:
    """形状无关的消费代码。

    文本累加；finish_reason 取第一个出现的；usage 取任意一片里带的。
    """
    text = ""
    finish: str | None = None
    usage: dict = {}
    for chunk in chunks:
        text += chunk.delta
        if finish is None and chunk.finish_reason:
            finish = chunk.finish_reason
        if chunk.usage:
            usage = chunk.usage
    return StreamResult(text, finish, int(usage.get("total_tokens", 0)), len(chunks))


def consume_naive(chunks: Sequence[StreamChunk]) -> StreamResult:
    """只读最后一片的消费代码——很多项目就是这么写的，换模型就翻车。"""
    text = "".join(chunk.delta for chunk in chunks)
    last = chunks[-1]
    return StreamResult(
        text, last.finish_reason, int(last.usage.get("total_tokens", 0)), len(chunks)
    )


def consume_skip_finish_text(chunks: Sequence[StreamChunk]) -> StreamResult:
    """假设"带 finish_reason 的那片没有正文"的消费代码。"""
    text = "".join(
        chunk.delta for chunk in chunks if not chunk.finish_reason
    )
    finish = next((c.finish_reason for c in chunks if c.finish_reason), None)
    usage: dict = {}
    for chunk in chunks:
        if chunk.usage:
            usage = chunk.usage
    return StreamResult(text, finish, int(usage.get("total_tokens", 0)), len(chunks))


def check_stream_shape() -> None:
    """两个模型的流式脚本形状不同，消费代码必须分别处理。"""
    section("流式事件形状：分片粒度和 finish 时机都可能变")

    # 模型 A：逐字分片，finish_reason 与最后一片正文同片，usage 单独一片
    char_chunks = [
        StreamChunk(delta="订单"),
        StreamChunk(delta="已"),
        StreamChunk(delta="发货"),
        StreamChunk(delta=""),
        StreamChunk(delta="。", finish_reason="stop"),
        StreamChunk(usage={"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104}),
    ]
    # 模型 B：整块分片，finish_reason / usage 与正文同片，末尾还有一片空的
    block_chunks = [
        StreamChunk(
            delta="订单已发货。",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104},
        ),
        StreamChunk(delta=""),
    ]

    robust_a = consume_robust(char_chunks)
    robust_b = consume_robust(block_chunks)
    naive_a = consume_naive(char_chunks)
    naive_b = consume_naive(block_chunks)
    skip_b = consume_skip_finish_text(block_chunks)

    print(f"  模型 A：{len(char_chunks)} 片（逐字，usage 单独一片）")
    print(f"  模型 B：{len(block_chunks)} 片（整块，finish/usage 与正文同片）")

    ensure(
        robust_a.chunk_count != robust_b.chunk_count,
        f"两个模型的流式分片数应当不同，实际都是 {robust_a.chunk_count} 片",
    )
    ensure(
        robust_a.text == robust_b.text == "订单已发货。",
        f"归一化之后两个模型的正文应当一致，实际 {robust_a.text!r} / {robust_b.text!r}",
    )
    ensure(
        robust_a.finish_reason == robust_b.finish_reason == "stop",
        f"归一化之后 finish_reason 应当都是 stop，实际 {robust_a.finish_reason} / "
        f"{robust_b.finish_reason}",
    )
    ensure(
        robust_a.total_tokens == robust_b.total_tokens == 104,
        f"归一化之后 token 统计应当一致，实际 {robust_a.total_tokens} / {robust_b.total_tokens}",
    )

    # 只读最后一片的消费代码在两个模型上都会丢东西
    ensure(
        naive_a.finish_reason is None,
        f"只读最后一片会丢掉模型 A 的 finish_reason，实际 {naive_a.finish_reason!r}",
    )
    ensure(
        naive_b.finish_reason is None and naive_b.total_tokens == 0,
        f"只读最后一片会丢掉模型 B 的 finish_reason 和 usage，实际 "
        f"{naive_b.finish_reason!r} / {naive_b.total_tokens}",
    )
    ensure(
        skip_b.text == "",
        f"假设 finish 片没有正文，会把模型 B 的正文全丢掉，实际 {skip_b.text!r}",
    )
    print("  → 形状差异会咬人：同一份拼接器，换个模型就得重写")


# ---------------------------------------------------------------- 9. 评测集不许动


@dataclass(frozen=True)
class EvalCase:
    """评测集里的一条用例。"""

    case_id: str
    task_type: str
    prompt: str
    expected: str
    hard: bool = False
    """旧模型答不对的难例——迁移最容易在这里翻车。"""

    unsolved: bool = False
    """连新模型也答不对的用例——它恰恰是最该留在评测集里的。"""


EVAL_SET: tuple[EvalCase, ...] = (
    EvalCase("e01", "分类", "把这句话归到账单类：我想查上个月的账单", "账单"),
    EvalCase("e02", "分类", "把这句话归到技术类：连不上 VPN 了", "技术"),
    EvalCase("e03", "抽取", "从『订单 A-1001 已付 320 元』里取订单号和金额", "A-1001/320"),
    EvalCase("e04", "抽取", "从『订单 B-2002 金额 88.5 元』里取订单号和金额", "B-2002/88.5"),
    EvalCase("e05", "改写", "把『坏了 快点修』写成工单描述", "设备故障，需尽快维修"),
    EvalCase("e06", "常规问答", "解释什么是幂等", "重复执行结果相同"),
    EvalCase("e07", "常规问答", "解释什么是退避", "失败后逐步延长等待再重试"),
    EvalCase("e08", "工具调用", "调用查订单工具查 A-1001", "查订单(A-1001)"),
    EvalCase("e09", "多步推理", "预算 1000 元买 A 价 320、B 价 400，最多买几件 A", "3"),
    EvalCase("e10", "多步推理", "两台机器各 8 核共 64 核，每台可用比例 0.75 是多少", "48", hard=True),
    EvalCase("e11", "多步推理", "税率随区间变化，先求分段总额再比较方案", "分段求和", hard=True),
    EvalCase("e12", "多步推理", "三个约束互相矛盾时给出取舍方案", "取舍方案", unsolved=True),
    EvalCase("e13", "多步推理", "跨两表对齐后判断是否超预算", "对齐后比较", unsolved=True),
)


def fingerprint(cases: Sequence[EvalCase]) -> str:
    """评测集的指纹——指纹变了，两次分数就不可比。"""
    payload = json.dumps(
        [[c.case_id, c.task_type, c.expected] for c in cases],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def simulate_answers(cases: Sequence[EvalCase], model: str) -> dict[str, str]:
    """模拟某个模型在评测集上的作答——真实系统里这一步是调真模型。

    ``old`` 在难例上答不对；``new`` 只在 unsolved 的用例上还答不对。
    """
    answers: dict[str, str] = {}
    for case in cases:
        if case.unsolved:
            answers[case.case_id] = "答错了"
        elif case.hard and model == "old":
            answers[case.case_id] = "答错了"
        else:
            answers[case.case_id] = case.expected
    return answers


def score(cases: Sequence[EvalCase], answers: dict[str, str]) -> float:
    """给一份评测集打分——正确率。"""
    if not cases:
        return 0.0
    hits = sum(1 for case in cases if answers.get(case.case_id) == case.expected)
    return hits / len(cases)


def check_eval_set_immutable() -> None:
    """迁移前后必须用同一份评测集；改了评测集分数就虚高。"""
    section("评测集不许动：分数要可比，先得数据集不变")

    before_fingerprint = fingerprint(EVAL_SET)

    old_answers = simulate_answers(EVAL_SET, "old")
    new_answers = simulate_answers(EVAL_SET, "new")
    score_old = score(EVAL_SET, old_answers)
    score_new = score(EVAL_SET, new_answers)

    print(f"  评测集指纹：{before_fingerprint}（{len(EVAL_SET)} 条用例）")
    print(f"  旧模型：{score_old:.3f}    新模型：{score_new:.3f}")

    ensure(
        set(old_answers) == set(new_answers),
        "迁移前后应当跑同一批用例",
    )
    ensure(
        set(new_answers) == {case.case_id for case in EVAL_SET},
        "作答集合应当覆盖评测集里的全部用例",
    )
    ensure(
        score_new > score_old,
        f"同一份评测集上，新模型应当比旧模型高，实际 {score_new:.3f} vs {score_old:.3f}",
    )
    ensure(
        fingerprint(EVAL_SET) == before_fingerprint,
        "评测集指纹在迁移前后应当一致",
    )
    print("  → 同一份评测集跑出的两个分数，才能拿来比")

    # 反例：把答不对的用例从评测集里删掉，分数立刻虚高——模型却一点没变
    weakened = tuple(case for case in EVAL_SET if not case.unsolved)
    score_new_on_weakened = score(weakened, new_answers)
    score_old_on_weakened = score(weakened, old_answers)

    print(f"  删掉 2 条难例后，新模型：{score_new_on_weakened:.3f}")
    ensure(
        score_new_on_weakened > score_new,
        f"改评测集之后分数应当虚高，实际 {score_new_on_weakened:.3f} vs {score_new:.3f}",
    )
    ensure(
        fingerprint(weakened) != before_fingerprint,
        "改了评测集，指纹应当变了——两份分数因此不可比",
    )
    print(f"  指纹变成 {fingerprint(weakened)}，与迁移前不同 → 不可比")
    print(f"  → 评测集一改，分数就是新的尺子量新东西：{score_old_on_weakened:.3f} → "
          f"{score_new_on_weakened:.3f} 全是虚的")


# ---------------------------------------------------------------- 10. 决策表


@dataclass(frozen=True)
class Scenario:
    """一个"要不要换模型"的场景。"""

    name: str
    score_before: float
    score_after: float
    cost_before: float
    cost_after: float
    has_eval_evidence: bool
    old_version_deprecated: bool = False
    context_overflow: bool = False


@dataclass(frozen=True)
class Decision:
    """决策结果。"""

    action: str
    """``switch_model`` / ``bump_version`` / ``stay``。"""

    reason: str


QUALITY_GAIN_THRESHOLD = 0.05
"""质量提升要达到这个幅度，才允许用"更贵"换"更好"。"""


def decide(scenario: Scenario) -> Decision:
    """按场景判定该不该换模型 / 该不该升版本。

    规则顺序本身就是纪律：硬约束先看，评测证据第二，成本第三。
    """
    if scenario.context_overflow:
        return Decision("switch_model", "旧模型装不下这份上下文，只能换长上下文档位")
    if scenario.old_version_deprecated:
        return Decision("bump_version", "旧版本被下线，必须升级——但走独立流程，不和换模型混在一起")
    if not scenario.has_eval_evidence:
        return Decision("stay", "没有评测证据就不动模型")
    if scenario.score_after < scenario.score_before:
        return Decision("stay", "质量掉了，不能换")
    quality_up = scenario.score_after > scenario.score_before
    cost_down = scenario.cost_after < scenario.cost_before
    if not quality_up and not cost_down:
        return Decision("stay", "质量和成本都没变，没有理由动模型")
    if cost_down:
        return Decision("switch_model", "质量没降、成本还降了，换")
    if scenario.cost_after == scenario.cost_before:
        return Decision("switch_model", "成本没变、质量更好了，换")
    if scenario.score_after - scenario.score_before >= QUALITY_GAIN_THRESHOLD:
        return Decision("switch_model", "质量提升超过阈值，贵一点也换")
    return Decision("stay", "质量提升不到阈值，成本又涨了，不换")


def check_decision_table() -> None:
    """场景 → 决策，逐条判定。"""
    section("决策表：该不该换模型 / 该不该升版本")

    scenarios = (
        Scenario("更准且更便宜", 0.83, 1.00, 0.50, 0.20, True),
        Scenario("更准但贵 3 倍", 0.83, 0.85, 0.50, 1.50, True),
        Scenario("更便宜但质量掉了", 0.83, 0.70, 0.50, 0.30, True),
        Scenario("看着更好但没评测", 0.83, 0.90, 0.50, 0.60, False),
        Scenario("旧版本被下线", 0.83, 0.83, 0.50, 0.50, True, old_version_deprecated=True),
        Scenario("旧模型装不下", 0.83, 0.83, 0.50, 0.50, True, context_overflow=True),
        Scenario("一切都没变", 0.83, 0.83, 0.50, 0.50, True),
    )
    expected = (
        "switch_model",
        "stay",
        "stay",
        "stay",
        "bump_version",
        "switch_model",
        "stay",
    )

    for scenario, want in zip(scenarios, expected):
        decision = decide(scenario)
        print(f"  {scenario.name:<12} → {decision.action:<14} {decision.reason}")
        ensure(
            decision.action == want,
            f"场景 {scenario.name!r} 应当判为 {want}，实际判为 {decision.action}",
        )
        ensure(bool(decision.reason), f"场景 {scenario.name!r} 必须给出理由")

    ensure(
        decide(scenarios[4]).action != decide(scenarios[0]).action,
        "升版本和换模型必须是两种不同的动作，不能混在一个开关里",
    )
    print("  → 升版本走独立流程：版本号变化不等于行为变化，别顺手一起做")


# ---------------------------------------------------------------- 11. 影子流量与流程


@dataclass(frozen=True)
class ShadowReport:
    """影子流量的比对结果。"""

    total: int
    divergent: int
    divergence_rate: float
    blocked: bool


def shadow_compare(
    old_outputs: Sequence[str], new_outputs: Sequence[str], threshold: float = 0.15
) -> ShadowReport:
    """并跑同一批真实请求，比对新旧输出。

    分歧率超过阈值就不许放量——影子流量是放量的前置条件，不是事后复盘。
    """
    total = len(old_outputs)
    divergent = sum(1 for a, b in zip(old_outputs, new_outputs) if a != b)
    rate = divergent / total if total else 0.0
    return ShadowReport(total, divergent, rate, rate > threshold)


@dataclass(frozen=True)
class MigrationStep:
    """迁移流程里的一步。"""

    name: str
    gate: str
    """这一步要过的闸。"""


MIGRATION_PLAN: tuple[MigrationStep, ...] = (
    MigrationStep("pin_version", "生产模型标识必须是带版本号的常量，不允许 latest/preview"),
    MigrationStep("baseline_eval", "用固定评测集跑出基线分数"),
    MigrationStep("shadow_traffic", "新旧模型并跑同一批真实请求，比对输出"),
    MigrationStep("canary", "按比例放量，同时盯成功率与成本"),
    MigrationStep("rollback", "能一键切回旧版本"),
)


def check_shadow_and_plan() -> None:
    """影子流量怎么判、迁移流程的顺序不能乱。"""
    section("影子流量与迁移流程")

    old_outputs = [
        "订单已发货",
        "预计明天到",
        "已退款",
        "查不到订单",
        "已签收",
        "配送中",
        "已取消",
        "明天送达",
        "库存不足",
        "等待付款",
    ]
    new_outputs = list(old_outputs)
    new_outputs[5] = "配送中（次日达）"  # 10 条里分歧 1 条

    report = shadow_compare(old_outputs, new_outputs)
    print(f"  影子流量：{report.total} 条，分歧 {report.divergent} 条"
          f"（分歧率 {report.divergence_rate:.1%}）")

    ensure(
        report.total == len(old_outputs),
        f"影子流量应当覆盖全部请求，实际 {report.total}",
    )
    ensure(
        abs(report.divergence_rate - 0.1) < 1e-9,
        f"分歧率应当是 10%，实际 {report.divergence_rate:.3f}",
    )
    ensure(
        not report.blocked,
        "分歧率 10% 在 15% 阈值内，应当允许放量",
    )
    tight = shadow_compare(old_outputs, new_outputs, threshold=0.05)
    ensure(
        tight.blocked,
        "分歧率 10% 超过 5% 阈值，应当拦住放量",
    )
    print("  → 阈值是提前定的：分歧超了就停，不是事后解释")

    names = [step.name for step in MIGRATION_PLAN]
    ensure(
        names == ["pin_version", "baseline_eval", "shadow_traffic", "canary", "rollback"],
        f"迁移流程的步骤顺序不对，实际 {names}",
    )
    ensure("rollback" in names, "迁移流程里必须有回滚这一步")
    ensure(
        "版本号" in MIGRATION_PLAN[0].gate,
        "第一步必须是 pin 死版本号",
    )
    print("  → pin 版本 → 跑基线 → 影子流量 → 灰度 → 可回滚，顺序不能跳")


# ---------------------------------------------------------------- 12. 定期复评


def check_price_drift(
    recorded: dict[str, tuple[float, float]], current: dict[str, tuple[float, float]]
) -> list[str]:
    """比对新旧价格表——价格变了就该触发一次复评。

    价格、上下文长度、模型能力都在变，所以选型不是一次性的决策。
    """
    drift: list[str] = []
    for tier, price in current.items():
        old = recorded.get(tier)
        if old is None:
            drift.append(f"{tier}: 新增档位 {price[0]}/{price[1]}")
        elif old != price:
            drift.append(f"{tier}: {old[0]}/{old[1]} → {price[0]}/{price[1]}")
    for tier in recorded:
        if tier not in current:
            drift.append(f"{tier}: 档位消失")
    return drift


@dataclass(frozen=True)
class ScoreCard:
    """一个模型在两套评测上的分数。"""

    model: str
    public_score: float
    our_score: float


PUBLIC_CARDS: tuple[ScoreCard, ...] = (
    ScoreCard("model-A", 0.95, 0.72),
    ScoreCard("model-B", 0.90, 0.88),
    ScoreCard("model-C", 0.88, 0.85),
)


def check_recheck_cycle() -> None:
    """价格会漂移，榜单靠不住——所以要定期复评。"""
    section("定期复评：价格会漂移，榜单不能当依据")

    recorded = {tier: price for tier, price in PRICING.items()}
    recorded["standard"] = (2.00, 8.00)  # 假设上次记的价格更低

    drift = check_price_drift(recorded, PRICING)
    print(f"  价格漂移：{drift}")
    ensure(
        len(drift) == 1 and drift[0].startswith("standard"),
        f"应当只检出 standard 一档的价格变化，实际 {drift}",
    )

    stable = check_price_drift(PRICING, PRICING)
    ensure(stable == [], f"价格没变时不该报漂移，实际 {stable}")
    print("  → 价格表变了就该重跑一遍选型矩阵：档位可能已经不值得了")

    public_best = max(PUBLIC_CARDS, key=lambda card: card.public_score)
    our_best = max(PUBLIC_CARDS, key=lambda card: card.our_score)
    print(f"  公开榜第一：{public_best.model}（{public_best.public_score:.2f}）")
    print(f"  我们评测第一：{our_best.model}（{our_best.our_score:.2f}）")

    ensure(
        public_best.model != our_best.model,
        "本例应当出现公开榜与自有评测结论不一致的情况",
    )
    public_order = [card.model for card in sorted(PUBLIC_CARDS, key=lambda c: -c.public_score)]
    our_order = [card.model for card in sorted(PUBLIC_CARDS, key=lambda c: -c.our_score)]
    ensure(
        public_order != our_order,
        f"两套排序应当不同，实际公开 {public_order} / 自有 {our_order}",
    )
    print("  → 榜单的任务分布不是你的业务分布：选型只认自己的评测集")


# ---------------------------------------------------------------- 运行


def run_live() -> None:
    """真实模型路径：真调一次模型，看它落在哪个档位、返回什么形状。"""
    banner("第 8 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client()
    response = client.complete(
        [
            ChatMessage(role="system", content="你只输出 JSON，不要任何解释。"),
            ChatMessage(role="user", content='抽取城市与天数，输出 {"city":"","days":0}'),
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    print(f"  content       : {response.content!r}")
    print(f"  model         : {response.model}")
    print(f"  finish_reason : {response.finish_reason}")
    print(f"  usage         : {response.usage}")
    print(f"  reasoning     : {bool(response.reasoning)}")

    ensure(bool(response.content), "真实模型应当返回内容")
    ensure(bool(response.model), "响应应当带上模型名——这正是要 pin 住的那个名字")
    ensure(
        response.finish_reason in {"stop", "length"},
        f"finish_reason 异常：{response.finish_reason}",
    )
    ensure(
        response.usage.get("total_tokens", 0) > 0,
        "usage 应当有 token 统计——账单按它结算",
    )

    # 真实模型的 JSON 严不严格，要靠解析端容错兜住
    try:
        parsed = extract_json(response.content)
        strict = validate_against_schema(parsed, EXTRACT_SCHEMA)
        print(f"  抠出来的 JSON ：{parsed}")
        print(f"  schema 校验   ：{'通过' if not strict else strict}")
    except ValueError:
        print("  抠 JSON 失败：真实模型没按 JSON 输出——这正是第 6 节说的坑")


def run() -> None:
    """跑第 8 章实验。"""
    banner("第 8 章 · 模型选型与迁移纪律")
    check_selection_matrix()
    check_routing_cost()
    check_version_pin()
    check_migration_checklist()
    check_tool_arguments_shape()
    check_json_strictness()
    check_context_length()
    check_stream_shape()
    check_eval_set_immutable()
    check_decision_table()
    check_shadow_and_plan()
    check_recheck_cycle()
    rule(
        "选型按任务难度定档位、迁移按清单逐项重测；"
        "版本 pin 死、评测集不动、影子流量并跑、灰度放量、一键回滚"
    )
    print("\n✓ 第 8 章通过")
