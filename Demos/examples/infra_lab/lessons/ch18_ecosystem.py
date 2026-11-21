"""第 18 章实验：生态地图——把 18 章的机制映射回框架选型。

这个模块解决什么问题：前 17 章讲的全是**机制**——分页 KV、连续批处理、前缀缓存、
量化、推测解码、chunked prefill、PD 分离。现实里这些机制由不同框架实现，
能力边界还不一样。选型的常见失误是**按名气选**，而不是**按你要的机制选**。

所以本章做两件事：

1. **机制对照表**——把每个框架支持哪些机制列成矩阵。**不 import 任何框架**，
   这是一张静态的能力地图，不是运行时对比。
2. **方案打分**——用第 16 章算出的真实成本数字，加上前面各章算出的时延与吞吐，
   给三种部署方案按「成本 / 时延 / 吞吐 / 运维复杂度」四个维度打分，
   给出可执行的选型结论。

全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..labkit import (
    GPUSpec,
    ModelSpec,
    banner,
    decode_step_time,
    ensure,
    fmt_gib,
    gpu,
    model,
    plan_memory,
    prefill_time,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
"""算账用的卡。"""

SPEC = model("qwen2.5-7b")
"""算账用的模型。"""


# ---------------------------------------------------------------- 机制与框架


@dataclass(frozen=True)
class Mechanism:
    """一个推理侧机制，附上"它解决什么问题"。"""

    key: str
    name: str
    problem: str
    """这个机制针对的是哪一章里的哪个问题。"""

    chapter: str
    """对应本教程的章节。"""


MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism("paged_kv", "PagedAttention / 分页 KV", "连续预留导致显存浪费数倍", "第 6 章"),
    Mechanism("continuous_batching", "连续批处理", "静态批让短请求被长请求拖死", "第 7 章"),
    Mechanism("prefix_cache", "前缀缓存", "共享 system prompt 的 prefill 被反复重算", "第 12 章"),
    Mechanism("chunked_prefill", "分块 prefill", "长 prompt 造成 TTFT 尖刺拖垮整个批", "第 13 章"),
    Mechanism("quantization", "量化", "权重占显存过大、每 token 搬运量太大", "第 9 章"),
    Mechanism("kv_quantization", "KV 量化", "长上下文下 KV 先撞显存天花板", "第 5、9 章"),
    Mechanism("speculative", "推测解码", "decode 阶段带宽受限导致每 token 成本高", "第 10 章"),
    Mechanism("pd_disagg", "PD 分离", "prefill 算力需求与 decode 带宽需求互相干扰", "第 18 章"),
)

# 框架能力矩阵：True=原生支持，False=不支持/需自己实现
FRAMEWORKS: tuple[str, ...] = ("vLLM", "SGLang", "TensorRT-LLM", "TGI")
FRAMEWORK_SUPPORT: dict[str, dict[str, bool]] = {
    "vLLM": {
        "paged_kv": True,
        "continuous_batching": True,
        "prefix_cache": True,
        "chunked_prefill": True,
        "quantization": True,
        "kv_quantization": True,
        "speculative": True,
        "pd_disagg": True,
    },
    "SGLang": {
        "paged_kv": True,
        "continuous_batching": True,
        "prefix_cache": True,
        "chunked_prefill": True,
        "quantization": True,
        "kv_quantization": True,
        "speculative": True,
        "pd_disagg": True,
    },
    "TensorRT-LLM": {
        "paged_kv": True,
        "continuous_batching": True,
        "prefix_cache": True,
        "chunked_prefill": True,
        "quantization": True,
        "kv_quantization": True,
        "speculative": True,
        "pd_disagg": True,
    },
    "TGI": {
        "paged_kv": True,
        "continuous_batching": True,
        "prefix_cache": True,
        "chunked_prefill": False,
        "quantization": True,
        "kv_quantization": False,
        "speculative": False,
        "pd_disagg": False,
    },
}
"""能力矩阵。

这一张表是**教学用的公开能力梳理**，不是实时探测的结果——
真要选型，必须去目标版本的官方文档核对，尤其是新版本加的功能。
矩阵的意义是教"按机制选型"这个**方法**，不是替你抄一张会过期的表。
"""

# 各框架的工程侧重——决定它在什么场景下更合适
FRAMEWORK_PROFILE: dict[str, dict[str, str]] = {
    "vLLM": {
        "strength": "生态最广、社区最大、量化与投机解码支持最全",
        "weakness": "极限吞吐场景下不如 TensorRT-LLM",
        "fit": "通用服务、需要丰富特性的团队",
    },
    "SGLang": {
        "strength": "RadixAttention 前缀复用、结构化输出、多轮 agent 负载友好",
        "weakness": "生态比 vLLM 小、周边工具少",
        "fit": "多轮对话、agent、前缀复用率高的负载",
    },
    "TensorRT-LLM": {
        "strength": "极致吞吐、算子融合程度最高、支持 PD 分离",
        "weakness": "构建重、版本敏感、定制成本高",
        "fit": "固定模型、追求极限吞吐的规模化部署",
    },
    "TGI": {
        "strength": "部署简单、显存优化到位、容错好",
        "weakness": "特性少一截、新机制支持慢",
        "fit": "规模不大、追求运维简单的团队",
    },
}


# ---------------------------------------------------------------- 方案打分


@dataclass
class DeploymentPlan:
    """一个部署方案。"""

    name: str
    framework: str
    replicas: int
    """副本数。"""

    weight_bits: int
    """权重位宽。"""

    description: str
    managed: bool = False
    """是否是托管服务。"""

    scores: dict[str, int] = field(default_factory=dict)
    """四个维度的得分（1–5，分高更好）。"""


DIMENSIONS: tuple[str, ...] = ("成本", "时延", "吞吐", "运维简单度")
"""打分维度。分高 = 更好。"""


def build_plans() -> list[DeploymentPlan]:
    """构造三种典型部署方案。"""
    return [
        DeploymentPlan(
            name="托管服务 · 小规模起步",
            framework="托管",
            replicas=2,
            weight_bits=16,
            description="不自己运维，先验证需求；规模小的时候单位成本高但省人力",
            managed=True,
            scores={"成本": 2, "时延": 3, "吞吐": 2, "运维简单度": 5},
        ),
        DeploymentPlan(
            name="自建 vLLM · 单机多副本",
            framework="vLLM",
            replicas=4,
            weight_bits=16,
            description="通用性强、特性全，适合大多数团队的默认选择",
            managed=False,
            scores={"成本": 4, "时延": 4, "吞吐": 3, "运维简单度": 3},
        ),
        DeploymentPlan(
            name="自建 TensorRT-LLM · 规模化",
            framework="TensorRT-LLM",
            replicas=8,
            weight_bits=8,
            description="模型固定、追求极限吞吐；量化把显存压下来换更多并发",
            managed=False,
            scores={"成本": 3, "时延": 3, "吞吐": 5, "运维简单度": 2},
        ),
    ]


LABOR_LADDER: tuple[tuple[int, int, float], ...] = (
    (8, 1, 8000.0),
    (32, 2, 16000.0),
    (128, 3, 24000.0),
)
"""（副本数上限，运维人数，人力月成本美元）。

这是**教学假设**而不是行业数据：它要做的只是让"人数按台阶涨、不按副本数涨"
这件事在算账里显式出现。换成你团队的真实人力成本，交点会平移，但形状不变。
"""

MANAGED_PREMIUM = 1.6
"""托管溢价：同样的卡，托管比裸卡贵 60%——溢价里含厂商利润和托管层开销。"""

HOURS_PER_MONTH = 24 * 30
"""按 30 天算的每月小时数。"""


def labor_cost(replicas: int) -> float:
    """这个规模下的运维人力月成本（美元）。

    为什么要分档：副本从 8 加到 32，不需要把运维团队从 1 人加到 4 人，
    只需要加 1 个人。**人力不随副本线性增长，就是自建唯一的规模效应。**
    """
    for max_replicas, _headcount, monthly in LABOR_LADDER:
        if replicas <= max_replicas:
            return monthly
    return LABOR_LADDER[-1][2]


def bare_cost(replicas: int) -> float:
    """裸卡租金的月成本（美元）。线性，没有规模效应。"""
    return replicas * HOURS_PER_MONTH * DEVICE.price_per_hour_usd


def managed_cost(replicas: int) -> float:
    """托管服务的月成本（美元）。裸卡 × 溢价，人力已经打包进去了。"""
    return bare_cost(replicas) * MANAGED_PREMIUM


def plan_cost(plan: DeploymentPlan) -> dict[str, float]:
    """用第 16 章的口径算这个方案的成本。

    单请求占用时间 = prefill + decode，容量 = 副本数 ÷ 单请求占用时间。

    月成本必须拆成三块看，否则一定会算错方向：

    - **裸卡租金**——线性，副本数 × 单卡时价。
    - **运维人力**——分档，**不随副本线性增长**（这是自建唯一的规模效应来源）。
    - **托管溢价**——托管才有，把运维人力打包进了单价里。
    """
    prefill = prefill_time(DEVICE, SPEC, 500, weight_bits=plan.weight_bits)
    # decode：单流近似，按上下文 2000 token 估
    step = decode_step_time(DEVICE, SPEC, 1, weight_bits=plan.weight_bits, context_tokens=2000)
    per_request = prefill + step * 200
    capacity = plan.replicas / per_request if per_request > 0 else 0.0

    bare = bare_cost(plan.replicas)
    labor = 0.0 if plan.managed else labor_cost(plan.replicas)
    monthly = managed_cost(plan.replicas) if plan.managed else bare + labor

    hours = 24 * 30
    tokens_per_month = capacity * 200 * 3600 * hours  # 每请求约 200 token
    return {
        "prefill_s": prefill,
        "per_request_s": per_request,
        "capacity_qps": capacity,
        "bare_monthly_usd": bare,
        "labor_monthly_usd": labor,
        "monthly_usd": monthly,
        "usd_per_1k_tokens": (monthly / tokens_per_month * 1000) if tokens_per_month > 0 else 0.0,
    }


# ---------------------------------------------------------------- 判断一


def check_capability_matrix() -> dict[str, int]:
    """判断：框架之间差的不是"快慢"，而是"支持哪些机制"。"""
    section("判断一：机制支持矩阵——按机制选型，不是按名气选型")

    headers = ["框架", *[f"{m.name}" for m in MECHANISMS]]
    rows = []
    for framework in FRAMEWORKS:
        support = FRAMEWORK_SUPPORT[framework]
        rows.append(
            [framework, *["✓" if support[m.key] else "—" for m in MECHANISMS]]
        )
    print_table(["框架", *[m.name.split(" / ")[0] for m in MECHANISMS]], rows)

    print()
    print_table(
        ["框架", "优势", "短板", "适合"],
        [
            [
                framework,
                FRAMEWORK_PROFILE[framework]["strength"],
                FRAMEWORK_PROFILE[framework]["weakness"],
                FRAMEWORK_PROFILE[framework]["fit"],
            ]
            for framework in FRAMEWORKS
        ],
    )

    # 每个框架至少支持分页 KV 和连续批处理——这是推理框架的及格线
    for framework in FRAMEWORKS:
        support = FRAMEWORK_SUPPORT[framework]
        ensure(
            support["paged_kv"] and support["continuous_batching"],
            f"{framework} 必须支持分页 KV 和连续批处理，否则连推理服务的基本盘都没有",
        )

    # TGI 的短板必须体现在矩阵里，否则这张表就没信息量
    tgi_gaps = sum(1 for m in MECHANISMS if not FRAMEWORK_SUPPORT["TGI"][m.key])
    ensure(
        tgi_gaps >= 3,
        f"TGI 至少应当在 {tgi_gaps} 个机制上有缺口，"
        "否则这张对照表就没有区分度",
    )

    rule(
        "这张表的用法：先确定你的负载需要哪些机制（长上下文要 KV 量化，"
        "agent 负载要前缀缓存，高并发要 chunked prefill），"
        "再筛掉不满足的框架。**别先选框架再找理由**"
    )
    return {framework: sum(FRAMEWORK_SUPPORT[framework].values()) for framework in FRAMEWORKS}


# ---------------------------------------------------------------- 判断二


def check_mechanism_to_chapter_mapping() -> None:
    """判断：每个机制都要能说清它解决什么问题、对应哪一章。"""
    section("判断二：机制 → 问题 → 章节的映射")

    print_table(
        ["机制", "解决的问题", "对应章节"],
        [[m.name, m.problem, m.chapter] for m in MECHANISMS],
    )

    ensure(len(MECHANISMS) == 8, "应覆盖 8 个核心机制")
    for mechanism in MECHANISMS:
        ensure(
            bool(mechanism.problem) and bool(mechanism.chapter),
            f"机制 {mechanism.name} 必须写清它解决什么问题、对应哪一章",
        )

    # 关键机制不能缺席：少任何一个，推理服务就会有一块明显短板
    required = {"paged_kv", "continuous_batching", "prefix_cache", "quantization"}
    present = {m.key for m in MECHANISMS}
    missing = required - present
    ensure(not missing, f"缺少关键机制：{sorted(missing)}")

    rule(
        "框架只是这些机制的**打包方式**。机制你都会了，换框架就只是换实现，"
        "而不是重新学一遍——这是前 17 章真正的复利"
    )


# ---------------------------------------------------------------- 判断三


def check_plan_scoring() -> list[tuple[DeploymentPlan, dict[str, float]]]:
    """判断：用真实成本数据给方案打分，结论要能被算出来。"""
    section("判断三：三种部署方案打分")

    plans = build_plans()
    costs = [plan_cost(plan) for plan in plans]

    print_table(
        ["方案", "框架", "副本", "权重位宽", "单请求占用", "容量(QPS)", "月成本", "每千token"],
        [
            [
                plan.name,
                plan.framework,
                plan.replicas,
                f"{plan.weight_bits}bit",
                f"{cost['per_request_s'] * 1000:.0f}ms",
                f"{cost['capacity_qps']:.1f}",
                f"${cost['monthly_usd']:,.0f}",
                f"${cost['usd_per_1k_tokens']:.3f}",
            ]
            for plan, cost in zip(plans, costs)
        ],
    )
    print()
    print("  月成本口径：托管 = 裸卡 × 1.6；自建 = 裸卡 + 运维人力（按规模分档）")

    print()
    print_table(
        ["方案", *DIMENSIONS, "总分"],
        [
            [
                plan.name,
                *[plan.scores[dim] for dim in DIMENSIONS],
                sum(plan.scores.values()),
            ]
            for plan in plans
        ],
    )

    # 总分最高的方案，就是这套口径下的默认推荐
    best = max(plans, key=lambda p: sum(p.scores.values()))
    print()
    print(f"  口径下的最优解：{best.name}（{best.description}）")

    # 托管服务的成本一定高于自建——但运维简单度一定更高。
    # 这个交换关系是本章的核心判断，必须成立。
    managed = next(p for p in plans if p.managed)
    self_built = next(p for p in plans if not p.managed)
    ensure(
        managed.scores["运维简单度"] > self_built.scores["运维简单度"],
        "托管服务的运维简单度必须高于自建",
    )
    ensure(
        managed.scores["成本"] < self_built.scores["成本"],
        "自建的单位成本必须优于托管——否则没人会自建",
    )

    # 量化确实省显存：8bit 的权重应当只有 16bit 的一半
    fp16 = plan_memory(SPEC, DEVICE, kv_tokens=0, weight_bits=16)
    int8 = plan_memory(SPEC, DEVICE, kv_tokens=0, weight_bits=8)
    ensure(
        int8.weights_gib < fp16.weights_gib * 0.55,
        f"INT8 权重应当约为 FP16 的一半："
        f"{fmt_gib(int8.weights_gib)} vs {fmt_gib(fp16.weights_gib)}",
    )

    rule(
        "托管 vs 自建是一个**人力换成本**的交易，不是技术优劣问题。"
        "小规模、要快：托管。规模起来了、模型固定、团队有 SRE 能力：自建。"
        "自建的隐藏成本是 oncall、升级、故障处理——这部分要用第 16 章的"
        "成本模型单独算进去，否则一定会低估自建"
    )
    return list(zip(plans, costs))


# ---------------------------------------------------------------- 判断四


def check_managed_vs_self_built() -> None:
    """判断：托管和自建的规模曲线不一样，交点随人力成本平移。"""
    section("判断四：规模效应——托管和自建的成本曲线不一样")

    scales = (2, 8, 32, 128)
    rows = []
    ratios: list[float] = []
    per_replica_labor: list[float] = []
    for replicas in scales:
        bare = bare_cost(replicas)
        labor = labor_cost(replicas)
        self_built = bare + labor
        managed = managed_cost(replicas)
        ratio = managed / self_built
        ratios.append(ratio)
        per_replica_labor.append(labor / replicas)
        rows.append(
            [
                replicas,
                f"${bare:,.0f}",
                f"${labor:,.0f}",
                f"${self_built:,.0f}",
                f"${managed:,.0f}",
                f"{ratio:.2f}x",
                f"${labor / replicas:,.0f}",
            ]
        )
    print_table(
        ["副本数", "裸卡租金", "运维人力", "自建合计", "托管(1.6x)", "托管/自建", "每副本人力"],
        rows,
    )

    # 小规模：人力的固定成本摊不开，托管更便宜
    ensure(
        ratios[0] < 1.0,
        f"小规模时托管应更便宜：{scales[0]} 副本时托管/自建 = {ratios[0]:.2f}x",
    )
    # 大规模：人力被摊薄，自建的固定成本优势反转
    ensure(
        ratios[-1] > 1.0,
        f"大规模时自建应更便宜：{scales[-1]} 副本时托管/自建 = {ratios[-1]:.2f}x",
    )
    # 规模效应的本质：每副本摊到的人力成本递减
    ensure(
        all(per_replica_labor[i] > per_replica_labor[i + 1] for i in range(len(per_replica_labor) - 1)),
        f"每副本摊到的人力成本必须随规模递减："
        f"{[f'${v:,.0f}' for v in per_replica_labor]}",
    )

    # 交点：自建开始比托管便宜的最小副本数
    crossover = None
    for replicas in range(1, 129):
        if bare_cost(replicas) + labor_cost(replicas) < managed_cost(replicas):
            crossover = replicas
            break
    ensure(crossover is not None, "在本章的价差下，自建应当存在一个交点")
    print()
    print(f"  交点：约 {crossover} 副本起，自建的月成本开始低于托管")

    rule(
        "规模效应是自建的核心论据，但不是免费的："
        "托管的单价固定，自建的单位成本随规模摊薄（人力按台阶涨、卡时利用率上去）。"
        "这个优势要等到**规模足够大**才成立——"
        "所以起点选托管、规模上来再迁自建，是常规路径。"
        "换句话说，选型不是一次性决定，而是要随规模重新算一次"
    )


def run() -> None:
    """跑第 18 章实验。"""
    banner("第 18 章 · 生态地图：框架选型与机制对照")
    rule(
        f"口径：{SPEC.name} / {DEVICE.name}。"
        "本章是静态对照与打分，**不 import 任何推理框架**——"
        "能力矩阵会随版本变化，真要选型请核对目标版本的官方文档"
    )
    check_capability_matrix()
    check_mechanism_to_chapter_mapping()
    check_plan_scoring()
    check_managed_vs_self_built()
    rule(
        "先定机制需求，再筛框架，最后用真实成本数字打分——"
        "顺序反了就会变成按名气选型"
    )
    print("\n✓ 第 18 章通过")
