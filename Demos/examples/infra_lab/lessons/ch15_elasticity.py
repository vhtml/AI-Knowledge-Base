"""第 15 章实验：弹性——自动扩缩容与冷启动。

这个模块解决什么问题：扩容这件事最容易犯的错，是以为"触发扩容"就等于"扩容生效"。
实际上中间隔着冷启动（拉镜像、下载权重、加载显存、预热），几十秒到几分钟。
本章把这段延迟显式建模，然后回答三个判断题：

1. **该用什么指标触发扩容**——利用率还是队列深度？模拟突发流量对比两种策略，
   证明利用率触发会滞后，队列深度触发才来得及。
2. **冷启动期间会发生什么**——突发流量打进来、容量还没到位，
   这段时间的 p99 会飙到多少？扩容不是免费的。
3. **副本数多少合适**——扫不同副本数找 p99 与卡时成本的平衡点，
   证明副本数不是越多越好（成本线性涨），也不是越少越好（p99 爆炸）。

全部离线运行，不联网、不调模型、不需要 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..labkit import (
    Completion,
    GPUSpec,
    ModelSpec,
    Request,
    banner,
    decode_step_time,
    ensure,
    fmt_ms,
    gpu,
    model,
    percentile,
    prefill_time,
    print_table,
    rule,
    section,
)

DEVICE = gpu("h100_sxm")
"""算账用的卡。"""

SPEC = model("qwen2.5-7b")
"""算账用的模型。"""

MAX_BATCH = 16
"""单副本的批上限。"""

KV_TOKEN_BUDGET = 120_000
"""单副本的 KV 预算（token）。"""


# ---------------------------------------------------------------- 配置与负载


@dataclass(frozen=True)
class AutoscalerConfig:
    """自动扩缩容配置。

    ``cold_start_s`` 是本章的核心参数：新副本从"创建"到"能接单"的时间。
    它决定了扩容能多快见效。
    """

    min_replicas: int = 1
    max_replicas: int = 8
    scale_up_queue_depth: int = 4
    """队列深度超过这个值触发扩容。"""

    scale_down_idle_s: float = 30.0
    """空闲超过这个时间就缩容。"""

    cold_start_s: float = 45.0
    """冷启动时间：新副本能接单前的准备时间。"""

    target_queue_depth: int = 2
    """目标队列深度：扩容决策想维持的水平。"""


@dataclass
class _Seq:
    """副本里的一条序列。"""

    request: Request
    enqueued_s: float
    produced: int = 0
    """已经产出多少 token。"""

    first_token_s: float | None = None
    finish_s: float | None = None


class _Replica:
    """一个副本：连续批处理 + 冷启动状态机。

    副本有两个状态：
    - ``starting``——正在冷启动，来了请求也接不了
    - ``ready``——可以接单
    """

    def __init__(self, name: str, config: AutoscalerConfig, device: GPUSpec, spec: ModelSpec) -> None:
        self.name = name
        self.config = config
        self.device = device
        self.spec = spec
        self.status: Literal["starting", "ready"] = "starting"
        self.ready_at = 0.0
        """冷启动结束的时刻（墙钟）。"""
        self.now = 0.0
        """本副本的墙钟。"""
        self.queue: list[_Seq] = []
        self.active: list[_Seq] = []
        self.completions: list[Completion] = []
        self.last_busy_s = 0.0
        """最后一次有活干的时刻，用来判断能不能缩容。

        创建时就要初始化成当前墙钟——否则刚出生的副本 ``last_busy_s=0``，
        会被缩容逻辑当成"空闲了很久"当场回收，然后又被重新拉起，
        形成"建了就删、删了再建"的抖动。
        """
        self.created_s = 0.0

    # -- 容量 -------------------------------------------------------------

    def _has_room(self) -> bool:
        """还有空位吗（批槽位 + KV 预算）。"""
        if len(self.active) >= MAX_BATCH:
            return False
        reserved = sum(
            seq.request.prompt_tokens + seq.request.max_output_tokens
            for seq in self.active
        ) + sum(
            seq.request.prompt_tokens + seq.request.max_output_tokens
            for seq in self.queue
        )
        return reserved <= KV_TOKEN_BUDGET

    # -- 调度 -------------------------------------------------------------

    def dispatch(self, pending: list[Request], wall: float) -> bool:
        """把一个待处理请求接进队列，返回是否接到了。

        **队列必须有上界**（``MAX_BATCH``）。否则第一个副本会把积压全揽走，
        后来的新副本一个请求都分不到——扩容变成了空转。
        """
        if self.status != "ready":
            return False
        if len(self.queue) >= MAX_BATCH:
            return False
        if not pending:
            return False
        request = pending[0]
        if request.arrival_s > wall:
            return False
        if not self._has_room():
            return False
        pending.pop(0)
        self.queue.append(_Seq(request=request, enqueued_s=wall))
        return True

    def _admit(self) -> float:
        """把队首请求接进来做 prefill，**只返回耗时，不推进时钟**。

        时钟统一由 :meth:`step` 推进——因为一轮里 prefill 和 decode 共享同一套算力，
        真实引擎取的是两者的较大值。
        """
        seq = self.queue.pop(0)
        request = seq.request
        # 副本的时钟不能早于请求到达时刻，否则 TTFT 会算成负数
        self.now = max(self.now, request.arrival_s)
        prefill_s = prefill_time(self.device, self.spec, request.prompt_tokens)
        self.active.append(seq)
        return prefill_s

    def _decode_step(self) -> float:
        """跑一轮 decode：批内每个序列产出一个 token，返回本轮耗时。"""
        if not self.active:
            return 0.0
        max_context = max(seq.request.prompt_tokens + seq.produced for seq in self.active)
        return decode_step_time(
            self.device,
            self.spec,
            len(self.active),
            context_tokens=max_context,
        )

    def _produce(self) -> None:
        """本轮每个活跃序列产出一个 token，并收尾跑完的序列。"""
        for seq in self.active:
            seq.produced += 1
            if seq.produced == 1 and seq.first_token_s is None:
                seq.first_token_s = self.now
            if seq.produced >= seq.request.max_output_tokens:
                seq.finish_s = self.now
                self.completions.append(
                    Completion(
                        request=seq.request,
                        enqueue_s=seq.enqueued_s,
                        prefill_end_s=seq.first_token_s or self.now,
                        finish_s=self.now,
                        ttft_s=(seq.first_token_s or self.now) - seq.request.arrival_s,
                        tpot_s=0.0,
                    )
                )
                self.active.remove(seq)

    def step(self) -> None:
        """跑一轮：先补位，再 decode。

        **每轮都会补位**——这就是连续批处理。运行批非空时不再接新请求，
        退化成的就是第 7 章讲的"静态批"，长短请求会互相拖死。
        """
        prefill_s = 0.0
        if self.queue and len(self.active) < MAX_BATCH and self._has_room():
            prefill_s = self._admit()

        if not self.active:
            if prefill_s:
                self.now += prefill_s
                self._produce()
            return

        step = self._decode_step()
        # prefill 与 decode 共享算力，一轮时长取较大者
        self.now += max(prefill_s, step)
        self._produce()
        if self.active or self.queue:
            self.last_busy_s = self.now

    def is_idle(self) -> bool:
        """是否完全空闲。"""
        return not self.queue and not self.active


# ---------------------------------------------------------------- 集群


@dataclass
class ClusterReport:
    """集群跑完之后的报告。"""

    completions: list[Completion]
    wall_s: float
    """墙钟总时长（秒）。"""

    replicas_used: int
    """实际用过的副本数（含冷启动失败的）。"""

    peak_replicas: int
    """同时存在过的最大副本数。"""

    scale_up_events: list[float]
    """每次扩容发生的时刻。"""

    scale_down_events: list[float]
    """每次缩容发生的时刻。"""

    cold_start_s: float
    cost_per_hour_usd: float
    """单卡每小时租金。"""

    @property
    def replica_hours(self) -> float:
        """卡时（卡 × 小时）。"""
        return self.peak_replicas * (self.wall_s / 3600.0)

    def ttft(self, q: float) -> float:
        """TTFT 分位数（毫秒）。"""
        return percentile([c.ttft_s for c in self.completions], q) * 1000

    def ttft_since(self, q: float, since_s: float) -> float:
        """只看某个时刻之后到达的请求的 TTFT 分位数（毫秒）。

        分段看是必须的：整体 p99 会被**第一段高压**主导，
        于是"第二次突发缩过容的代价"就被第一次的噪声淹没了。
        """
        values = [c.ttft_s for c in self.completions if c.request.arrival_s >= since_s]
        return percentile(values, q) * 1000

    def p99_violation_rate(self, slo_ms: float = 1000.0) -> float:
        """TTFT 超过 SLO 的比例。"""
        if not self.completions:
            return 0.0
        hits = sum(1 for c in self.completions if c.ttft_s * 1000 > slo_ms)
        return hits / len(self.completions)

    def output_tokens_per_s(self) -> float:
        """输出吞吐（token/s，按墙钟算）。"""
        if not self.completions:
            return 0.0
        total = sum(c.output_tokens for c in self.completions)
        return total / max(self.wall_s, 1e-9)


class ElasticCluster:
    """有扩容延迟的推理集群。

    模型里显式包含三件事：

    1. **扩容有延迟**——新副本要 ``cold_start_s`` 才能接单；
    2. **副本并行**——各副本有各自的时钟，墙钟取最大值；
    3. **缩容会丢预热**——空闲超过阈值的副本被回收，重启又要再付一次冷启动。
    """

    def __init__(self, config: AutoscalerConfig, device: GPUSpec, spec: ModelSpec) -> None:
        self.config = config
        self.device = device
        self.spec = spec
        self.replicas: list[_Replica] = []
        self._retired: list[Completion] = []
        """被缩容回收的副本已经处理完的请求。回收副本时如果不收走，
        这些请求的时延记录就凭空消失了——统计口径会失真。"""
        self._next_id = 0

    def _spawn(self, wall: float) -> _Replica:
        """新建一个副本，进入冷启动。"""
        replica = _Replica(f"replica-{self._next_id}", self.config, self.device, self.spec)
        self._next_id += 1
        replica.created_s = wall
        replica.ready_at = wall + self.config.cold_start_s
        replica.now = wall
        replica.last_busy_s = wall
        self.replicas.append(replica)
        return replica

    def _queue_depth(self, pending: list[Request], wall: float) -> int:
        """当前真正的队列深度：**已到达但还没拿到 token** 的请求数。

        两个坑：

        1. 不能直接用 ``len(pending)``——那里面混着还没到达的请求。
           把未来的请求算进深度，等于在流量还没来的时候就开始扩容。
        2. 也不能只看 ``pending``——请求一旦被分发进副本，就进了副本自己的队列。
           只看 ``pending`` 会永远看到深度为 0，于是永远不扩容。

        所以真实积压 = 集群里还没分发的 + 各副本队列里的。
        """
        undispatched = sum(1 for request in pending if request.arrival_s <= wall)
        in_replica_queues = sum(len(replica.queue) for replica in self.replicas)
        return undispatched + in_replica_queues

    def run(self, requests: list[Request], *, horizon_s: float = 600.0) -> ClusterReport:
        """跑完这批请求，返回报告。"""
        pending = sorted(requests, key=lambda r: r.arrival_s)
        scale_up_events: list[float] = []
        scale_down_events: list[float] = []
        wall = 0.0
        peak = 0

        # 初始副本：直接就绪（假装服务早就起来了）。
        # 数量要按 min_replicas 来——固定副本数的实验全靠这个参数，
        # 只起一个的话"扫不同副本数"扫出来的全是同一组数。
        for _ in range(self.config.min_replicas):
            initial = self._spawn(0.0)
            initial.status = "ready"
            initial.ready_at = 0.0
            initial.last_busy_s = 0.0
            peak = max(peak, len(self.replicas))

        while pending or any(not r.is_idle() for r in self.replicas) or any(
            r.status == "starting" for r in self.replicas
        ):
            if wall > horizon_s:
                break

            ready = [r for r in self.replicas if r.status == "ready"]

            # 1. 扩容判定：队列深度超阈值
            depth = self._queue_depth(pending, wall)
            if depth > self.config.scale_up_queue_depth and len(self.replicas) < self.config.max_replicas:
                self._spawn(wall)
                scale_up_events.append(wall)
                peak = max(peak, len(self.replicas))
                ready = [r for r in self.replicas if r.status == "ready"]

            # 2. 分发：按负载从轻到重派发，避免一个副本把积压全揽走
            for replica in sorted(ready, key=lambda r: len(r.queue) + len(r.active)):
                replica.dispatch(pending, wall)

            # 3. 各就绪副本各跑一轮（并行，墙钟取各自时钟的最大值）
            for replica in ready:
                replica.step()
            # 墙钟必须**单调递增**：空闲副本的时钟是旧的（它没干活），
            # 直接取 max 会让墙钟倒退，缩容的级联就会中途断掉。
            wall = max(wall, max((r.now for r in self.replicas), default=wall))

            # 墙钟只有在**所有就绪副本都空闲**时才能跳到下一个到达时刻。
            # 否则会把还没到的请求当成"已经该处理了"，把积压算成排队。
            #
            # 跳转目标取"下一个到达时刻"和"最早该缩容的时刻"里较早的那个——
            # 否则会一口气跳过整个低谷，把本该发生的缩容事件漏掉。
            if pending and not any(
                r.status == "ready" and not r.is_idle() for r in self.replicas
            ):
                next_arrival = pending[0].arrival_s
                # 只有**还能缩容**时，"该缩容的时刻"才是一个有意义的时间点。
                # 已经到最小副本数了，再等下去也缩不掉——这时只能等下一个请求到达，
                # 否则墙钟卡在原地不动，模拟会陷入死循环。
                idle_deadlines = [
                    r.last_busy_s + self.config.scale_down_idle_s
                    for r in self.replicas
                    if r.status == "ready" and len(self.replicas) > self.config.min_replicas
                ]
                idle_deadline = min(idle_deadlines) if idle_deadlines else next_arrival
                # 还要考虑**冷启动即将完成**的时刻。
                # 漏掉它的话，墙钟会一步跳过冷启动完成点，
                # 新副本被当成"刚就绪且已经不空闲"，永远等不到该缩容的时刻。
                starting_deadlines = [
                    r.ready_at for r in self.replicas if r.status == "starting"
                ]
                if starting_deadlines:
                    idle_deadline = min(idle_deadline, min(starting_deadlines))
                wall = max(wall, min(next_arrival, idle_deadline))

            # 4. 冷启动完成的副本转为就绪
            for replica in self.replicas:
                if replica.status == "starting" and wall >= replica.ready_at:
                    replica.status = "ready"
                    replica.now = max(replica.now, replica.ready_at)
                    # 空闲计时从**就绪**这一刻起算，而不是从创建那一刻。
                    # 按创建算的话，冷启动 45 秒 + 空闲阈值 30 秒 = 75 秒，
                    # 副本一就绪就已经"空闲超时"，会被当场回收再重建，
                    # 形成"建了就删、删了再建"的抖动。
                    replica.last_busy_s = wall

            # 5. 缩容：空闲超时回收。
            #    被回收的副本已经处理过的请求不能丢——它的完成记录要收进集群。
            if len(self.replicas) > self.config.min_replicas:
                for replica in list(self.replicas):
                    # 不能删到低于 min_replicas 为止——
                    # 一次循环删光的话，最后那个初始副本也会被回收，
                    # 等于把服务的基础容量也缩没了。
                    if len(self.replicas) <= self.config.min_replicas:
                        break
                    if (
                        replica.status == "ready"
                        and replica.is_idle()
                        and wall - replica.last_busy_s >= self.config.scale_down_idle_s - 1e-9
                    ):
                        self.replicas.remove(replica)
                        self._retired.extend(replica.completions)
                        scale_down_events.append(wall)

            # 6. 死锁保护：没有就绪副本、还有请求 → 等最快那个冷启动完
            if pending and not any(r.status == "ready" for r in self.replicas):
                starting = [r for r in self.replicas if r.status == "starting"]
                if starting:
                    wall = max(wall, min(r.ready_at for r in starting))
                else:
                    break

            # 7. 全部跑完就收工
            if not pending and all(r.is_idle() for r in self.replicas):
                break

        completions = self._retired + [c for r in self.replicas for c in r.completions]
        return ClusterReport(
            completions=completions,
            wall_s=wall,
            replicas_used=len(self.replicas),
            peak_replicas=peak,
            scale_up_events=scale_up_events,
            scale_down_events=scale_down_events,
            cold_start_s=self.config.cold_start_s,
            cost_per_hour_usd=self.device.price_per_hour_usd,
        )


# ---------------------------------------------------------------- 负载


def build_load(
    *,
    baseline_count: int = 40,
    baseline_interval_s: float = 0.25,
    burst_count: int = 40,
    burst_start_s: float = 3.0,
    burst_window_s: float = 0.4,
    prompt_tokens: int = 400,
    output_tokens: int = 80,
) -> list[Request]:
    """构造"稳态 + 突发"的负载。

    稳态部分给引擎留出喘息，突发部分在 0.4 秒内砸进 40 条请求——
    这正是"队列深度瞬间飙升"的典型形态。
    """
    requests: list[Request] = []
    req_id = 1
    for index in range(baseline_count):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=index * baseline_interval_s,
                prompt_tokens=prompt_tokens,
                max_output_tokens=output_tokens,
            )
        )
        req_id += 1
    for index in range(burst_count):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=burst_start_s + index * (burst_window_s / max(1, burst_count)),
                prompt_tokens=prompt_tokens,
                max_output_tokens=output_tokens,
                tenant="burst",
            )
        )
        req_id += 1
    return requests


def build_sustained_load(
    *,
    rate: float = 30.0,
    duration_s: float = 60.0,
    prompt_tokens: int = 400,
    output_tokens: int = 80,
) -> list[Request]:
    """构造**持续高压**负载：单副本扛不住的稳定流量。

    为什么判断一要用它而不是短突发：冷启动要几十秒，
    而突发只有零点几秒——**新副本还没就绪，突发就结束了**，
    扩容对这一次的 p99 毫无帮助。只有持续高压，
    "扩容早了几十秒"才会兑现成"后面几千个请求的时延被救回来"。

    单副本在这个请求尺寸下的理论容量约 21.5 req/s，所以 30 req/s 会明显把队列顶起来。
    """
    requests: list[Request] = []
    req_id = 1
    total = int(rate * duration_s)
    for index in range(total):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=index / rate,
                prompt_tokens=prompt_tokens,
                max_output_tokens=output_tokens,
            )
        )
        req_id += 1
    return requests


# ---------------------------------------------------------------- 判断一


def check_trigger_metric() -> dict[str, ClusterReport]:
    """判断：该按队列深度扩容，不是按利用率。"""
    section("判断一：扩容触发指标——队列深度 vs 利用率")

    load = build_sustained_load(rate=30.0, duration_s=60.0)

    # 策略 A：队列深度触发（深度超过 4 就扩）
    queue_config = AutoscalerConfig(
        min_replicas=1,
        max_replicas=6,
        scale_up_queue_depth=4,
        cold_start_s=45.0,
    )
    queue_cluster = ElasticCluster(queue_config, DEVICE, SPEC)
    queue_report = queue_cluster.run(load)

    # 策略 B：利用率触发——必须等队列堆到 30 才反应，模拟"利用率信号滞后"
    util_config = AutoscalerConfig(
        min_replicas=1,
        max_replicas=6,
        scale_up_queue_depth=30,
        cold_start_s=45.0,
    )
    util_cluster = ElasticCluster(util_config, DEVICE, SPEC)
    util_report = util_cluster.run(load)

    print_table(
        ["策略", "扩容次数", "扩容时刻(s)", "TTFT p50", "TTFT p99", "违约率(>1s)"],
        [
            [
                "队列深度>4 触发",
                len(queue_report.scale_up_events),
                ", ".join(f"{t:.2f}" for t in queue_report.scale_up_events) or "无",
                f"{queue_report.ttft(0.50):.0f}ms",
                f"{queue_report.ttft(0.99):.0f}ms",
                f"{queue_report.p99_violation_rate() * 100:.1f}%",
            ],
            [
                "利用率触发(深度>30)",
                len(util_report.scale_up_events),
                ", ".join(f"{t:.2f}" for t in util_report.scale_up_events) or "无",
                f"{util_report.ttft(0.50):.0f}ms",
                f"{util_report.ttft(0.99):.0f}ms",
                f"{util_report.p99_violation_rate() * 100:.1f}%",
            ],
        ],
    )

    ensure(
        len(queue_report.scale_up_events) > 0,
        "队列深度触发策略应当在突发到来时触发扩容",
    )
    ensure(
        queue_report.ttft(0.99) < util_report.ttft(0.99),
        f"队列深度触发的 p99（{queue_report.ttft(0.99):.0f}ms）应当低于"
        f"利用率触发的 p99（{util_report.ttft(0.99):.0f}ms）——"
        "利用率信号滞后，等它涨起来时用户已经卡住了",
    )
    rule(
        "利用率是滞后信号：它要先等系统过载才升高，那时队列已经堆起来了。"
        "队列深度是领先信号——它一涨就可以立刻决定扩容"
    )
    return {"queue": queue_report, "util": util_report}


# ---------------------------------------------------------------- 判断二


def check_cold_start_cost() -> dict[str, ClusterReport]:
    """判断：扩容不是瞬时的，冷启动期间用户要自己扛。"""
    section("判断二：冷启动成本——扩容生效之前发生什么")

    load = build_load()

    fast = ElasticCluster(
        AutoscalerConfig(min_replicas=1, max_replicas=6, scale_up_queue_depth=4, cold_start_s=0.0),
        DEVICE,
        SPEC,
    ).run(load)
    slow = ElasticCluster(
        AutoscalerConfig(min_replicas=1, max_replicas=6, scale_up_queue_depth=4, cold_start_s=60.0),
        DEVICE,
        SPEC,
    ).run(load)

    print_table(
        ["冷启动", "扩容次数", "TTFT p50", "TTFT p99", "违约率(>1s)", "墙钟"],
        [
            [
                "0s（理想）",
                len(fast.scale_up_events),
                f"{fast.ttft(0.50):.0f}ms",
                f"{fast.ttft(0.99):.0f}ms",
                f"{fast.p99_violation_rate() * 100:.1f}%",
                f"{fast.wall_s:.2f}s",
            ],
            [
                "60s（真实）",
                len(slow.scale_up_events),
                f"{slow.ttft(0.50):.0f}ms",
                f"{slow.ttft(0.99):.0f}ms",
                f"{slow.p99_violation_rate() * 100:.1f}%",
                f"{slow.wall_s:.2f}s",
            ],
        ],
    )

    ensure(
        slow.ttft(0.99) > fast.ttft(0.99),
        f"冷启动 60s 时的 p99（{slow.ttft(0.99):.0f}ms）应当高于理想情况"
        f"（{fast.ttft(0.99):.0f}ms）——扩容生效前，新副本接不了单",
    )
    ensure(
        slow.scale_up_events and slow.scale_up_events[0] > 0,
        "扩容事件应当发生在负载起来之后，而不是服务启动时",
    )
    rule(
        f"冷启动 {slow.cold_start_s:.0f}s 意味着：扩容决策做出后，"
        f"这 {slow.cold_start_s:.0f}s 内的流量全部压在老副本上。"
        "所以容量规划必须按'扩容生效前的最坏负载'来留余量，而不是按稳态负载"
    )
    return {"fast": fast, "slow": slow}


# ---------------------------------------------------------------- 判断三


def check_optimal_replicas() -> list[ClusterReport]:
    """判断：副本数不是越多越好，要找 p99 与成本的平衡点。"""
    section("判断三：副本数怎么定——p99 与成本的平衡")

    load = build_load()
    reports: list[ClusterReport] = []
    for replicas in (1, 2, 3, 4, 6, 8):
        cluster = ElasticCluster(
            AutoscalerConfig(
                min_replicas=replicas,
                max_replicas=replicas,
                scale_up_queue_depth=100,
                cold_start_s=0.0,
            ),
            DEVICE,
            SPEC,
        )
        reports.append(cluster.run(load))

    rows = []
    for replicas, report in zip((1, 2, 3, 4, 6, 8), reports):
        rows.append(
            [
                replicas,
                f"{report.ttft(0.50):.0f}ms",
                f"{report.ttft(0.99):.0f}ms",
                f"{report.p99_violation_rate() * 100:.1f}%",
                f"{report.output_tokens_per_s():.0f}",
                f"{report.replica_hours:.4f}",
                f"${report.replica_hours * DEVICE.price_per_hour_usd:.5f}",
            ]
        )
    print_table(
        ["副本数", "TTFT p50", "TTFT p99", "违约率", "吞吐 tok/s", "卡时", "成本"],
        rows,
    )

    single = reports[0]
    best_index = min(range(len(reports)), key=lambda i: reports[i].ttft(0.99))
    best = reports[best_index]

    ensure(
        single.p99_violation_rate() > reports[3].p99_violation_rate(),
        f"单副本的违约率（{single.p99_violation_rate() * 100:.1f}%）应当高于 4 副本"
        f"（{reports[3].p99_violation_rate() * 100:.1f}%）",
    )
    ensure(
        reports[3].ttft(0.99) < single.ttft(0.99) * 0.5,
        f"从 1 副本加到 4 副本，p99 应当显著下降："
        f"{single.ttft(0.99):.0f}ms → {reports[3].ttft(0.99):.0f}ms",
    )
    ensure(
        best_index < len(reports) - 1,
        f"p99 最优的副本数是 {reports[best_index].peak_replicas}，"
        "不应该是最大值——加副本总有上限",
    )
    ensure(
        reports[-1].replica_hours > single.replica_hours * 7,
        f"卡时随副本数线性增长：1 副本 {single.replica_hours:.4f} → "
        f"8 副本 {reports[-1].replica_hours:.4f}",
    )
    rule(
        f"副本数的边际收益递减：p99 在 {best.peak_replicas} 副本时最好"
        f"（{best.ttft(0.99):.0f}ms），再加副本 p99 反而略微变差"
        f"（{reports[-1].ttft(0.99):.0f}ms）——因为请求被摊薄到更多副本，"
        "每批的规模变小，权重摊薄的收益也就没了。"
        "而卡时是线性增长的。所以正确做法是「刚好够用 + 一点余量」，不是越多越好"
    )
    return reports


# ---------------------------------------------------------------- 判断四


def check_scale_down_cost() -> None:
    """判断：缩容省了钱，但下次扩容要重付冷启动。"""
    section("判断四：缩容的代价——预热不是免费的")

    idle = AutoscalerConfig(
        min_replicas=1,
        max_replicas=6,
        scale_up_queue_depth=4,
        scale_down_idle_s=20.0,
        cold_start_s=30.0,
    )
    sticky = AutoscalerConfig(
        min_replicas=1,
        max_replicas=6,
        scale_up_queue_depth=4,
        scale_down_idle_s=1000.0,
        cold_start_s=30.0,
    )

    # 三段负载：高压（触发扩容）→ 长时间低谷（触发缩容）→ 再来一次高压
    # 低谷期长度必须超过缩容阈值，第二次高压必须落在低谷之后，
    # 这样"要不要在低谷把副本缩掉"才会变成一个真实的取舍。
    load: list[Request] = []
    req_id = 1
    for index in range(240):
        load.append(
            Request(
                req_id=req_id,
                arrival_s=index / 40.0,
                prompt_tokens=300,
                max_output_tokens=60,
            )
        )
        req_id += 1
    for index in range(120):
        load.append(
            Request(
                req_id=req_id,
                arrival_s=70.0 + index / 40.0,
                prompt_tokens=300,
                max_output_tokens=60,
            )
        )
        req_id += 1

    eager = ElasticCluster(idle, DEVICE, SPEC).run(load)
    sticky_report = ElasticCluster(sticky, DEVICE, SPEC).run(load)

    print_table(
        ["策略", "扩容次数", "缩容次数", "TTFT p99", "违约率(>1s)", "峰值副本"],
        [
            [
                "空闲 20s 就缩",
                len(eager.scale_up_events),
                len(eager.scale_down_events),
                f"{eager.ttft_since(0.99, 70.0):.0f}ms",
                f"{eager.p99_violation_rate() * 100:.1f}%",
                eager.peak_replicas,
            ],
            [
                "不缩容",
                len(sticky_report.scale_up_events),
                len(sticky_report.scale_down_events),
                f"{sticky_report.ttft_since(0.99, 70.0):.0f}ms",
                f"{sticky_report.p99_violation_rate() * 100:.1f}%",
                sticky_report.peak_replicas,
            ],
        ],
    )
    rule("（TTFT p99 一列只统计第二次突发（t≥70s）里的请求）")

    ensure(
        len(eager.scale_down_events) > 0,
        "空闲 20s 的策略应当在低峰期缩过容",
    )
    ensure(
        len(sticky_report.scale_down_events) == 0,
        "不缩容的策略不应该发生缩容事件",
    )
    ensure(
        eager.ttft_since(0.99, 70.0) > sticky_report.ttft_since(0.99, 70.0),
        f"缩过容的策略第二次高压时 p99 应当更高："
        f"{eager.ttft_since(0.99, 70.0):.0f}ms（缩过） vs "
        f"{sticky_report.ttft_since(0.99, 70.0):.0f}ms（没缩）——"
        "新副本要重付一次冷启动",
    )
    rule(
        "缩容省下的是低谷期的卡时，代价是下次突发要重付一次冷启动。"
        "对交互式服务，宁可多留一个副本常驻，也不要让它在低谷被回收"
    )


def run() -> None:
    """跑第 15 章实验。"""
    banner("第 15 章 · 弹性：自动扩缩容与冷启动")
    rule(
        f"口径：{SPEC.name} / {DEVICE.name} / 单副本批上限 {MAX_BATCH} / "
        f"KV 预算 {KV_TOKEN_BUDGET} tokens"
    )
    check_trigger_metric()
    check_cold_start_cost()
    check_optimal_replicas()
    check_scale_down_cost()
    rule(
        "扩容按队列深度触发，按冷启动时间预留余量，按 p99 定容量，"
        "按业务形态决定要不要缩容"
    )
    print("\n✓ 第 15 章通过")
