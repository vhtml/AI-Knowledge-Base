"""第 8 章实验：多租户调度与 SLA。

这个模块解决什么问题：把"多租户共享一张卡"这件事从口号变成四个可验证的判断：

1. **调度策略的代价**——先到先服务、短作业优先、优先级，各自把谁给害了；
2. **配额保护并发度**——保底槽位能保证低优先级租户"排得上"，代价是谁来付；
3. **老化保护尾延迟**——把"饿死"变成"最多等这么久"，但只在顺序是瓶颈时有效；
4. **SLA 只能看 p99**——均值达标不代表 SLA 达标，违约率是唯一能签字的指标。

第 7 章的调度内核直接复用（同一套时间口径），本章只在其上加租户维度：
``QuotaEngine`` 加保底槽位，``AgingEngine`` 加老化——
这两个扩展点是公共层引擎目前没有的，所以本章的表格用本章内核；
末尾再用**公共层的** :class:`infra_lab.labkit.EngineSimulator` 直接跑一遍做对照，
确认两套实现在同一负载上的策略结果一致（口径一致性）。

所有耗时口径一律调用 labkit（``prefill_time`` / ``decode_step_time`` / ``kv_capacity_tokens``）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labkit import (
    BYTES_PER_GIB,
    Completion,
    EngineSimulator,
    Request,
    banner,
    ensure,
    gpu,
    kv_capacity_tokens,
    model,
    percentile,
    print_table,
    rule,
    section,
)
from .ch07_continuous_batching import ContinuousBatchEngine

__all__ = [
    "AgingEngine",
    "QuotaEngine",
    "Tenant",
    "build_flood_workload",
    "build_multitenant_workload",
    "tenant_report",
]


# ---------------------------------------------------------------- 租户画像


@dataclass(frozen=True)
class Tenant:
    """一个租户画像。

    ``priority`` 数字越小越优先；``sla`` 表示这条租户签了"首 token 1 秒内"的约定
    （口径来自 labkit 的 ``Request.ttft_deadline()``）。
    """

    name: str
    priority: int
    count: int
    prompt_tokens: int
    output_tokens: int
    sla: bool
    arrival_start_s: float
    arrival_step_s: float

    def to_request(self, req_id: int, index: int) -> Request:
        """按索引生成一条请求。"""
        arrival = self.arrival_start_s + index * self.arrival_step_s
        return Request(
            req_id=req_id,
            arrival_s=arrival,
            prompt_tokens=self.prompt_tokens,
            max_output_tokens=self.output_tokens,
            tenant=self.name,
            priority=self.priority,
            deadline_s=arrival if self.sla else None,
        )


MIXED_TENANTS: tuple[Tenant, ...] = (
    Tenant("bulk", 2, 6, 2048, 512, True, 0.000, 0.005),
    Tenant("premium", 0, 12, 256, 192, True, 0.030, 0.010),
    Tenant("standard", 1, 12, 128, 192, True, 0.030, 0.010),
)
"""混合负载的三个租户。

刻意让**长任务先到**：这样"先到先服务"会把长任务排前面（害了后到的交互请求），
而"短作业优先"会把长任务推到队尾（害了长任务自己）。
另外 premium 的请求比 standard 更长，于是"按优先级"和"按长短"的排序会给出不同答案。
"""

FLOOD_TENANTS: tuple[Tenant, ...] = (
    Tenant("bulk", 2, 6, 2048, 512, True, 0.000, 0.005),
    Tenant("premium", 0, 40, 128, 48, True, 0.030, 0.050),
    Tenant("standard", 1, 4, 256, 64, True, 0.030, 0.050),
)
"""洪峰负载：premium 每 50ms 来一条、持续 2 秒，是典型的"吵闹邻居"。

bulk 的 6 条长请求在最前面就到，却要排在洪峰后面——
这是"顺序约束"型饥饿，配额和老化都能治。
"""


def _build(tenants: tuple[Tenant, ...]) -> list[Request]:
    """把租户画像展开成请求列表。"""
    requests: list[Request] = []
    req_id = 0
    for tenant in tenants:
        for index in range(tenant.count):
            requests.append(tenant.to_request(req_id, index))
            req_id += 1
    return requests


def build_multitenant_workload() -> list[Request]:
    """三租户混合负载：长短都有，租户之间体量相当。"""
    return _build(MIXED_TENANTS)


def build_flood_workload() -> list[Request]:
    """洪峰负载：一个租户持续打请求，另两个租户排队等。"""
    return _build(FLOOD_TENANTS)


def tenant_report(completions: list[Completion]) -> dict[str, dict[str, float]]:
    """按租户拆一份指标表：TTFT 分布、违约率、拿到的 token 数。

    违约率用 labkit 的 ``Completion.missed_ttft_deadline()``，
    口径是"首 token 晚于到达后 1 秒"。
    """
    report: dict[str, dict[str, float]] = {}
    tenants = sorted({c.request.tenant for c in completions})
    for tenant in tenants:
        subset = [c for c in completions if c.request.tenant == tenant]
        if not subset:
            continue
        ttfts = [c.ttft_s for c in subset]
        violations = sum(1 for c in subset if c.missed_ttft_deadline())
        report[tenant] = {
            "count": float(len(subset)),
            "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
            "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
            "ttft_mean_ms": (sum(ttfts) / len(ttfts)) * 1000,
            "sla_violation_rate": violations / len(subset),
            "output_tokens": float(sum(c.output_tokens for c in subset)),
        }
    return report


# ---------------------------------------------------------------- 策略变体


class QuotaEngine(ContinuousBatchEngine):
    """带"保底槽位"的引擎：给某个租户预留几个 batch 槽位。

    为什么需要它：纯优先级调度下，低优先级租户会被高优先级租户挤到队尾，
    而队列是有序的——它不是"慢一点"，而是**永远排不上**（饿死）。
    预留槽位就是把"至少分到多少并发"写进契约，而不是留给调度器的善意。

    代价也很实在：被保护的租户照样要占整轮迭代做 prefill，
    共享池小一分，所有人排队就长一分——配额是**交换**，不是免费。
    """

    def __init__(self, *args, reserved_slots: dict[str, int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reserved_slots = dict(reserved_slots or {})
        """租户 → 保底预留几个槽位（这些槽位别人占不了）。"""

    def _slot_budget_exceeded(self, running: list, request: Request) -> bool:
        """共享池的规则：所有租户合起来最多占 ``max_batch - 预留总数`` 个槽位。

        超出自己预留部分的占用叫"溢出"，溢出量不能超过共享池大小——
        这样被预留的槽位就一定是空着的，等着给对应租户。
        """
        shared = max(0, self.max_batch - sum(self.reserved_slots.values()))
        used: dict[str, int] = {}
        for seq in running:
            used[seq.request.tenant] = used.get(seq.request.tenant, 0) + 1
        used[request.tenant] = used.get(request.tenant, 0) + 1
        overflow = sum(
            max(0, count - self.reserved_slots.get(tenant, 0))
            for tenant, count in used.items()
        )
        return overflow > shared


class AgingEngine(ContinuousBatchEngine):
    """带老化（aging）的引擎：等得越久，优先级自动提升。

    老化的作用是把"饿死"变成"最多等这么久"：
    一个请求每等过 ``aging_window_s`` 就自动升一级优先级，
    所以老请求最终会插到新来的高优先级请求前面。

    注意它的适用条件：老化治的是**顺序**型饥饿。
    如果瓶颈是**容量**（根本没有空槽位），那谁先谁后都一样，老化无效。
    """

    def __init__(self, *args, aging_window_s: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aging_window_s = aging_window_s

    def _pending_order_key(self, request: Request) -> tuple:
        waited = max(0.0, self.now - request.arrival_s)
        promotion = int(waited // self.aging_window_s)
        effective_priority = request.priority - promotion
        if self.policy == "sjf":
            return (effective_priority, request.prompt_tokens + request.max_output_tokens)
        return (effective_priority, request.arrival_s)


# ---------------------------------------------------------------- 公共层对照


def check_public_engine_agrees() -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """公共层引擎与本章内核在同一个负载上必须给出同一个结论。

    公共层的 :class:`infra_lab.labkit.EngineSimulator` 已经修好了到达记账、
    去重和补位语义，本章直接用它跑一遍做对照：
    两套实现的指标应该在 10% 以内吻合，否则章节之间的口径就断了。
    """
    workload = build_multitenant_workload()
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    rows: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for policy in ("fcfs", "sjf", "priority"):
        public = EngineSimulator(
            device, spec, max_batch=16, kv_token_budget=100_000, mfu=0.5, policy=policy
        )
        public.run(workload)
        public_report = tenant_report(public.completions)

        kernel = ContinuousBatchEngine(max_batch=16, policy=policy)
        kernel.run(workload)
        kernel_report = tenant_report(kernel.completions)

        for tenant in ("premium", "standard", "bulk"):
            left = public_report[tenant]["ttft_p99_ms"]
            right = kernel_report[tenant]["ttft_p99_ms"]
            ensure(
                abs(left - right) <= max(left, right) * 0.1,
                f"公共层引擎与本章内核在 {policy}/{tenant} 上的 p99 TTFT 应该一致："
                f"{left:.1f}ms vs {right:.1f}ms",
            )
        rows.append((policy, public_report, kernel_report))
    return rows


# ---------------------------------------------------------------- 断言


def check_sjf_helps_short_but_starves_long() -> dict[str, dict[str, dict[str, float]]]:
    """短作业优先：短请求的 TTFT 明显变好，长请求被饿死。

    这条判断不是"短作业优先更好/更差"，而是"它把代价转嫁给了谁"。
    """
    workload = build_multitenant_workload()

    reports: dict[str, dict[str, dict[str, float]]] = {}
    for policy in ("fcfs", "sjf", "priority"):
        engine = ContinuousBatchEngine(max_batch=16, policy=policy)
        engine.run(workload)
        reports[policy] = tenant_report(engine.completions)

    fcfs = reports["fcfs"]
    sjf = reports["sjf"]

    ensure(
        sjf["standard"]["ttft_p99_ms"] < fcfs["standard"]["ttft_p99_ms"],
        f"短作业优先下 standard 的 TTFT p99（{sjf['standard']['ttft_p99_ms']:.1f}ms）应该低于"
        f"先到先服务（{fcfs['standard']['ttft_p99_ms']:.1f}ms）",
    )
    ensure(
        sjf["bulk"]["ttft_p99_ms"] > fcfs["bulk"]["ttft_p99_ms"],
        f"代价是长任务被饿死：bulk 的 TTFT p99 从 {fcfs['bulk']['ttft_p99_ms']:.1f}ms "
        f"涨到 {sjf['bulk']['ttft_p99_ms']:.1f}ms",
    )
    ensure(
        reports["priority"]["premium"]["ttft_p99_ms"]
        <= reports["sjf"]["premium"]["ttft_p99_ms"],
        f"按优先级调度时 premium（优先级最高）应该比按长短调度更快："
        f"{reports['priority']['premium']['ttft_p99_ms']:.1f}ms ≤ "
        f"{reports['sjf']['premium']['ttft_p99_ms']:.1f}ms",
    )
    return reports


def check_aging_bounds_starvation() -> dict[str, dict[str, float]]:
    """老化把"饿死"变成"最多等这么久"。

    用的洪峰负载：bulk 的长请求最先到，却要排在 premium 的洪峰后面。
    没有老化时它被挤到最后；加了老化之后，老请求插到新来的高优先级请求前面。
    """
    workload = build_flood_workload()

    plain = ContinuousBatchEngine(max_batch=16, policy="priority")
    plain.run(workload)
    plain_report = tenant_report(plain.completions)

    aging = AgingEngine(max_batch=16, policy="priority", aging_window_s=0.25)
    aging.run(workload)
    aging_report = tenant_report(aging.completions)

    ensure(
        plain_report["bulk"]["sla_violation_rate"] > 0,
        "没有保护时，被挤到最后的长任务应该出现 SLA 违约",
    )
    ensure(
        aging_report["bulk"]["ttft_p99_ms"] < plain_report["bulk"]["ttft_p99_ms"] * 0.6,
        f"老化后 bulk 的 TTFT p99（{aging_report['bulk']['ttft_p99_ms']:.1f}ms）应该明显低于"
        f"无保护时（{plain_report['bulk']['ttft_p99_ms']:.1f}ms）",
    )
    ensure(
        aging_report["bulk"]["sla_violation_rate"] < plain_report["bulk"]["sla_violation_rate"],
        f"老化的目标是消灭违约：{plain_report['bulk']['sla_violation_rate'] * 100:.0f}% → "
        f"{aging_report['bulk']['sla_violation_rate'] * 100:.0f}%",
    )
    ensure(
        aging_report["premium"]["ttft_p50_ms"] <= plain_report["premium"]["ttft_p50_ms"] * 2,
        f"老化的代价要有限：premium 的 TTFT p50 从 {plain_report['premium']['ttft_p50_ms']:.1f}ms "
        f"变到 {aging_report['premium']['ttft_p50_ms']:.1f}ms",
    )
    return aging_report


def check_quota_guarantees_capacity() -> dict[str, dict[str, float]]:
    """保底槽位保证的是"并发度"：被保护的租户一定排得上。

    它比老化更硬（直接给槽位），但代价也直接写在别人的 TTFT 上。
    """
    workload = build_flood_workload()

    plain = ContinuousBatchEngine(max_batch=16, policy="priority")
    plain.run(workload)
    plain_report = tenant_report(plain.completions)

    quota = QuotaEngine(max_batch=16, policy="priority", reserved_slots={"bulk": 6})
    quota.run(workload)
    quota_report = tenant_report(quota.completions)

    ensure(
        quota_report["bulk"]["ttft_p99_ms"] < plain_report["bulk"]["ttft_p99_ms"],
        f"预留 6 个槽位后 bulk 的 TTFT p99（{quota_report['bulk']['ttft_p99_ms']:.1f}ms）应该低于"
        f"无配额时（{plain_report['bulk']['ttft_p99_ms']:.1f}ms）",
    )
    ensure(
        quota_report["bulk"]["sla_violation_rate"] == 0,
        "被保底保护的租户不该再有 SLA 违约",
    )
    ensure(
        quota_report["premium"]["ttft_p50_ms"] > plain_report["premium"]["ttft_p50_ms"],
        f"配额是交换：共享池小了一分，premium 的 TTFT p50 从 "
        f"{plain_report['premium']['ttft_p50_ms']:.1f}ms 涨到 "
        f"{quota_report['premium']['ttft_p50_ms']:.1f}ms",
    )
    return quota_report


def check_p99_beats_mean_for_sla() -> dict[str, dict[str, float]]:
    """均值达标不代表 SLA 达标：违约率必须单独看。

    这里专门造一个"大部分很快、偶尔很慢"的租户：
    同一个租户平时发 128 token 的小请求，偶尔发一条 32K token 的大文档。
    结果是均值没超 1 秒、p99 超过 1 秒、违约率不为 0——
    只看均值会得出"没问题"的错误结论。
    """
    requests: list[Request] = []
    req_id = 0
    for index in range(9):
        arrival = 0.0 + index * 0.002
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=arrival,
                prompt_tokens=128,
                max_output_tokens=32,
                tenant="burst",
                priority=0,
                deadline_s=arrival,
            )
        )
        req_id += 1
    # 同一条租户线上的大文档：prefill 一次就要 1 秒多
    requests.append(
        Request(
            req_id=req_id,
            arrival_s=0.020,
            prompt_tokens=32768,
            max_output_tokens=32,
            tenant="burst",
            priority=0,
            deadline_s=0.020,
        )
    )
    req_id += 1

    engine = ContinuousBatchEngine(max_batch=16, policy="priority")
    engine.run(requests)
    report = tenant_report(engine.completions)

    metrics = report["burst"]
    ensure(
        metrics["ttft_mean_ms"] < 1000,
        f"burst 租户的 TTFT 均值（{metrics['ttft_mean_ms']:.1f}ms）应该没超 1 秒——"
        "这正是只看均值会误判的地方",
    )
    ensure(
        metrics["sla_violation_rate"] > 0,
        f"但违约率是 {metrics['sla_violation_rate'] * 100:.0f}%——"
        "慢的那一条照样违约，均值看不出来",
    )
    ensure(
        metrics["ttft_p99_ms"] > metrics["ttft_mean_ms"] * 2,
        f"p99（{metrics['ttft_p99_ms']:.1f}ms）应该远高于均值"
        f"（{metrics['ttft_mean_ms']:.1f}ms）",
    )
    return report


def check_memory_quota_math() -> dict[str, float]:
    """显存配额怎么算：KV 预算 ÷ 每 token 字节数 = 能装多少 token。

    这是租户隔离的第一个维度，也是唯一能"算"出来的维度：
    优先级可以调、抢占权可以谈，但显存就那么多，必须分账。
    """
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    budget_gib = 6.0
    total_tokens = kv_capacity_tokens(spec, budget_gib * BYTES_PER_GIB, 2)
    per_tenant = total_tokens // 3

    ensure(
        total_tokens > 0 and per_tenant * 3 <= total_tokens,
        f"三个租户平分 {budget_gib:.0f} GiB 的 KV 预算，每租户 {per_tenant} token",
    )

    # 一条 2048 prompt + 512 输出的请求要占多少 token
    heavy = 2048 + 512
    ensure(
        per_tenant // heavy >= 1,
        f"每租户 {per_tenant} token 的配额至少要能装下一条重请求（{heavy} token）",
    )
    ensure(
        per_tenant // heavy * 3 <= total_tokens // heavy,
        "平分之后，三个租户能同时跑的重请求数量不应该超过总配额能跑的",
    )
    return {
        "total_tokens": float(total_tokens),
        "per_tenant_tokens": float(per_tenant),
        "heavy_requests_per_tenant": float(per_tenant // heavy),
    }


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 8 章的全部实验与断言。"""
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    banner("第 8 章 · 多租户调度与 SLA")
    rule(f"设备：{device.name}　模型：{spec.name}")

    section("一、三租户混合负载")
    workload = build_multitenant_workload()
    print_table(
        ["租户", "优先级", "条数", "prompt", "output", "是否签 SLA", "到达区间(s)"],
        [
            [
                tenant.name,
                tenant.priority,
                tenant.count,
                tenant.prompt_tokens,
                tenant.output_tokens,
                "是" if tenant.sla else "否",
                f"{tenant.arrival_start_s:.3f}~"
                f"{tenant.arrival_start_s + (tenant.count - 1) * tenant.arrival_step_s:.3f}",
            ]
            for tenant in MIXED_TENANTS
        ],
    )

    section("二、三种调度策略的代价")
    reports = check_sjf_helps_short_but_starves_long()
    rows = []
    for policy in ("fcfs", "sjf", "priority"):
        for tenant in ("premium", "standard", "bulk"):
            metrics = reports[policy][tenant]
            rows.append(
                [
                    policy,
                    tenant,
                    f"{metrics['ttft_p50_ms']:.1f}",
                    f"{metrics['ttft_p99_ms']:.1f}",
                    f"{metrics['sla_violation_rate'] * 100:.0f}%",
                    f"{metrics['output_tokens']:.0f}",
                ]
            )
    print_table(["策略", "租户", "TTFT p50(ms)", "TTFT p99(ms)", "SLA 违约率", "产出 token"], rows)
    rule("每种策略都把代价转嫁给了某个租户：先到先服务牺牲后到的，短作业优先牺牲长的")

    section("三、洪峰负载：老化和配额各治一种病")
    aging_report = check_aging_bounds_starvation()
    quota_report = check_quota_guarantees_capacity()
    flood = build_flood_workload()
    plain = ContinuousBatchEngine(max_batch=16, policy="priority")
    plain.run(flood)
    plain_report = tenant_report(plain.completions)
    print_table(
        ["方案", "bulk p99(ms)", "bulk 违约率", "premium p50(ms)", "premium 违约率"],
        [
            [
                "纯优先级",
                f"{plain_report['bulk']['ttft_p99_ms']:.1f}",
                f"{plain_report['bulk']['sla_violation_rate'] * 100:.0f}%",
                f"{plain_report['premium']['ttft_p50_ms']:.1f}",
                f"{plain_report['premium']['sla_violation_rate'] * 100:.0f}%",
            ],
            [
                "+ 老化(0.25s)",
                f"{aging_report['bulk']['ttft_p99_ms']:.1f}",
                f"{aging_report['bulk']['sla_violation_rate'] * 100:.0f}%",
                f"{aging_report['premium']['ttft_p50_ms']:.1f}",
                f"{aging_report['premium']['sla_violation_rate'] * 100:.0f}%",
            ],
            [
                "+ 保底 6 槽位",
                f"{quota_report['bulk']['ttft_p99_ms']:.1f}",
                f"{quota_report['bulk']['sla_violation_rate'] * 100:.0f}%",
                f"{quota_report['premium']['ttft_p50_ms']:.1f}",
                f"{quota_report['premium']['sla_violation_rate'] * 100:.0f}%",
            ],
        ],
    )
    rule("老化治顺序型饥饿（它挤掉的是别人的排队时间），保底槽位治容量型饥饿（它挤掉的是别人的槽位）")

    section("四、为什么必须看 p99 而不是均值")
    report = check_p99_beats_mean_for_sla()
    print_table(
        ["租户", "TTFT 均值(ms)", "TTFT p50(ms)", "TTFT p99(ms)", "SLA 违约率"],
        [
            [
                tenant,
                f"{metrics['ttft_mean_ms']:.1f}",
                f"{metrics['ttft_p50_ms']:.1f}",
                f"{metrics['ttft_p99_ms']:.1f}",
                f"{metrics['sla_violation_rate'] * 100:.0f}%",
            ]
            for tenant, metrics in report.items()
        ],
    )
    rule("SLA 的口径是违约率，不是均值——均值没超标也可能有请求违约")

    section("五、显存配额怎么分")
    quota_math = check_memory_quota_math()
    rule(
        f"{quota_math['total_tokens']:.0f} token 的 KV 预算，三家平分各"
        f"{quota_math['per_tenant_tokens']:.0f} token，"
        f"够同时跑 {quota_math['heavy_requests_per_tenant']:.0f} 条重请求"
    )

    section("六、公共层 EngineSimulator 对照")
    rule(
        "公共层引擎已修正到达记账与补位语义，这里直接用它跑同一负载，"
        "和本章内核对照口径是否一致"
    )
    rows = []
    for policy, public_report, kernel_report in check_public_engine_agrees():
        for tenant in ("premium", "standard", "bulk"):
            rows.append(
                [
                    policy,
                    "公共层",
                    tenant,
                    f"{public_report[tenant]['ttft_p99_ms']:.1f}",
                    f"{public_report[tenant]['sla_violation_rate'] * 100:.0f}%",
                ]
            )
            rows.append(
                [
                    policy,
                    "本章内核",
                    tenant,
                    f"{kernel_report[tenant]['ttft_p99_ms']:.1f}",
                    f"{kernel_report[tenant]['sla_violation_rate'] * 100:.0f}%",
                ]
            )
    print_table(["策略", "引擎", "租户", "TTFT p99(ms)", "违约率"], rows)
