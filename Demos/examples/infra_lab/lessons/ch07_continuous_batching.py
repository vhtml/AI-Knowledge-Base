"""第 7 章实验：连续批处理与调度内核。

这个模块解决什么问题：把"静态批和连续批到底差在哪"从一句口号变成可复现的数字。
读者最常见的误解是"连续批只是把 batch 调大"，所以本模块把两件事分开做：

1. **静态批**（本文件自己实现简化版）：固定槽位、批次内不补位、整批一起结束。
   槽位空转和"短请求被同批长请求拖死"这两件事必须被看见，所以不能只引用现成框架。
2. **连续批**（iteration 级调度）：每一轮都重新决定谁在跑，谁跑完谁的位置立刻给谁。
3. **分块 prefill**：长 prompt 切成块与 decode 交织，看它到底保护了谁。

所有耗时口径一律调用 :mod:`infra_lab.labkit`（``prefill_time`` / ``decode_step_time``），
本文件不另写任何显存、带宽、FLOPs 公式，避免和别的章节算出两套数。

本模块导出的 :class:`ContinuousBatchEngine` 与 :func:`summarize` 被第 8 章复用，
所以调度口径在两章之间是一致的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ..labkit import (
    BYTES_PER_GIB,
    Completion,
    Request,
    banner,
    decode_step_time,
    ensure,
    fmt_ms,
    gpu,
    kv_capacity_tokens,
    model,
    percentile,
    prefill_time,
    print_table,
    rule,
    section,
)

__all__ = [
    "ContinuousBatchEngine",
    "StaticBatchEngine",
    "Policy",
    "build_mixed_workload",
    "summarize",
]

Policy = Literal["fcfs", "sjf", "priority"]
"""调度策略：先到先服务 / 短作业优先 / 优先级。"""


# ---------------------------------------------------------------- 负载构造


def build_mixed_workload() -> list[Request]:
    """构造一份"长短混合、到达不均匀"的负载。

    负载刻意包含三类请求，才能把调度内核的三种代价都暴露出来：

    - ``interactive``：短 prompt、短输出，代表人机对话，对 TTFT 敏感；
    - ``longform``：长 prompt、长输出，代表文档总结，代表"TTFT 尖刺源"；
    - ``batched``：中等长度、晚到，代表离线批处理，代表"可以等的请求"。

    到达时刻刻意错开：静态批会因为凑批而等待，连续批不会。
    """
    requests: list[Request] = []
    req_id = 0

    # 第一波：4 条交互请求，几乎同时到达
    for i in range(4):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=0.0 + i * 0.002,
                prompt_tokens=128,
                max_output_tokens=32,
                tenant="interactive",
                priority=1,
                deadline_s=0.0 + i * 0.002,
            )
        )
        req_id += 1

    # 第二波：一条长文档请求。它故意晚到一点点，落在交互请求**正在 decode** 的窗口里，
    # 这样才能看出长 prompt 的 prefill 是"卡住正在生成的序列"，而不只是"自己慢"
    requests.append(
        Request(
            req_id=req_id,
            arrival_s=0.060,
            prompt_tokens=4096,
            max_output_tokens=160,
            tenant="longform",
            priority=0,
            deadline_s=0.060,
        )
    )
    req_id += 1

    # 第三波：8 条交互请求，在长 prompt 的 prefill 窗口内陆续到达
    for i in range(8):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=0.070 + i * 0.010,
                prompt_tokens=128,
                max_output_tokens=32,
                tenant="interactive",
                priority=1,
                deadline_s=0.070 + i * 0.010,
            )
        )
        req_id += 1

    # 第四波：3 条批处理请求，最后到达，输出很长
    for i in range(3):
        requests.append(
            Request(
                req_id=req_id,
                arrival_s=0.150 + i * 0.010,
                prompt_tokens=512,
                max_output_tokens=256,
                tenant="batched",
                priority=2,
                deadline_s=0.150 + i * 0.010,
            )
        )
        req_id += 1

    return requests


# ---------------------------------------------------------------- 指标汇总


def summarize(
    completions: list[Completion],
    *,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """把一批完成情况汇总成指标表。

    口径与 :meth:`infra_lab.labkit.EngineSimulator.metrics` 对齐：
    TTFT 从**到达**算起（不是从入队算起），TPOT 是首 token 之后的平均间隔，
    吞吐按"从第一条请求到达到最后一条完成"的时间跨度算。
    """
    if not completions:
        return {"count": 0.0}

    ttfts = [c.ttft_s for c in completions]
    tpots = [c.tpot_s for c in completions]
    e2es = [c.total_time_s for c in completions]
    total_output = sum(c.output_tokens for c in completions)
    makespan = max(c.finish_s for c in completions)
    start = min(c.request.arrival_s for c in completions)
    span = max(makespan - start, 1e-9)

    metrics: dict[str, float] = {
        "count": float(len(completions)),
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
        "tpot_p50_ms": percentile(tpots, 0.50) * 1000,
        "tpot_p99_ms": percentile(tpots, 0.99) * 1000,
        "e2e_p99_ms": percentile(e2es, 0.99) * 1000,
        "output_tokens_per_s": total_output / span,
        "requests_per_s": len(completions) / span,
        "makespan_s": makespan,
    }
    if extra:
        metrics.update(extra)
    return metrics


def _subset_metrics(
    completions: list[Completion],
    predicate,
) -> dict[str, float]:
    """只看满足条件的子集（比如"交互式租户"）的 TTFT 分布。"""
    subset = [c for c in completions if predicate(c)]
    if not subset:
        return {"count": 0.0, "ttft_p50_ms": 0.0, "ttft_p99_ms": 0.0}
    ttfts = [c.ttft_s for c in subset]
    return {
        "count": float(len(subset)),
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
    }


# ---------------------------------------------------------------- 连续批引擎


@dataclass
class _Seq:
    """引擎内部的一条序列状态。"""

    request: Request
    enqueue_s: float
    """被准入（占用槽位）的时刻。"""
    prefilled: int = 0
    """已经算过的 prompt token 数；分块 prefill 时会分多次增长。"""
    context: int = 0
    """当前上下文长度（prompt + 已产出 token）。"""
    produced: int = 0
    """已产出的 token 数。"""
    ready: bool = False
    """prompt 是否已经全部 prefill 完，可以开始 decode。"""
    prefill_end_s: float = 0.0
    """prefill 结束、可以产出第一个 token 的时刻。"""


class ContinuousBatchEngine:
    """连续批处理引擎：iteration 级调度的最小可运行实现。

    每一轮迭代做两件事（时间取两者较大者，模拟"同一次 forward 里既能 decode 又能 prefill"）：

    1. **decode**：所有 prefill 已完成的序列各产出一个 token；
    2. **prefill**：一个 prefill 工作单位。分块模式下是一个 chunk，
       非分块模式下是一条新请求的完整 prefill。

    与静态批的本质差别在准入时机：**每轮迭代结束就重新看队列**，
    谁跑完谁的位置立刻让给等待中的请求，而不是等整批结束。

    调度策略影响两处：准入顺序（选谁进批）和分块 prefill 的处理顺序
    （长请求的剩余 chunk 会不会被短请求插队）。
    """

    def __init__(
        self,
        device=None,
        spec=None,
        *,
        max_batch: int = 16,
        kv_budget_gib: float = 6.0,
        mfu: float = 0.5,
        weight_bits: int | None = None,
        policy: Policy = "fcfs",
        chunked_prefill: bool = False,
        chunk_size: int = 512,
        tenant_slots: dict[str, int] | None = None,
    ) -> None:
        self.device = device if device is not None else gpu("h100_sxm")
        self.spec = spec if spec is not None else model("qwen2.5-7b")
        self.max_batch = max_batch
        self.kv_token_budget = kv_capacity_tokens(
            self.spec, kv_budget_gib * BYTES_PER_GIB, 2
        )
        self.mfu = mfu
        self.weight_bits = weight_bits
        self.policy: Policy = policy
        self.chunked_prefill = chunked_prefill
        self.chunk_size = chunk_size
        self.tenant_slots = dict(tenant_slots or {})
        """租户 → 最多占用几个 batch 槽位。这是第 8 章"显存配额"的简化版。"""

        self.now = 0.0
        self.completions: list[Completion] = []
        self.iterations = 0
        self.batch_sizes: list[int] = []
        self.preempted = 0
        """被抢占并从头上重算的序列数（第 7 章讲抢占时的记账）。"""

    # -- 队列与准入 --------------------------------------------------------

    def _select_pending(self, pending: list[Request]) -> Request | None:
        """按策略从等待队列里挑下一个要准入的请求。"""
        if not pending:
            return None
        if self.policy == "sjf":
            pending.sort(key=lambda r: r.prompt_tokens + r.max_output_tokens)
        elif self.policy == "priority":
            pending.sort(key=lambda r: (r.priority, r.arrival_s))
        else:
            pending.sort(key=lambda r: r.arrival_s)
        return pending.pop(0)

    def _tenant_slot_used(self, running: list[_Seq], tenant: str) -> int:
        """某租户当前占用的槽位数。"""
        return sum(1 for seq in running if seq.request.tenant == tenant)

    def _reserved_tokens(self, running: list[_Seq]) -> int:
        """已准入序列预留的 KV token 数。

        准入时就按"prompt + 最大输出"预留，避免半途发现装不下再回滚。
        """
        return sum(
            seq.request.prompt_tokens + seq.request.max_output_tokens for seq in running
        )

    def _can_admit(self, running: list[_Seq], request: Request) -> bool:
        """能不能装得下：槽位数、租户配额、KV 预算三关都要过。"""
        if len(running) >= self.max_batch:
            return False
        limit = self.tenant_slots.get(request.tenant)
        if limit is not None and self._tenant_slot_used(running, request.tenant) >= limit:
            return False
        need = request.prompt_tokens + request.max_output_tokens
        return self._reserved_tokens(running) + need <= self.kv_token_budget

    # -- 主循环 ------------------------------------------------------------

    def run(self, requests: list[Request], *, horizon: float = 600.0) -> list[Completion]:
        """跑完一批请求，返回完成情况（按完成时间排序）。"""
        self.now = 0.0
        self.completions = []
        self.iterations = 0
        self.batch_sizes = []
        self.preempted = 0

        arrivals = sorted(requests, key=lambda r: r.arrival_s)
        pending: list[Request] = []
        running: list[_Seq] = []
        index = 0

        while index < len(arrivals) or pending or running:
            if self.now > horizon:
                break

            # 1. 把已经到达的请求放进等待队列
            while index < len(arrivals) and arrivals[index].arrival_s <= self.now:
                pending.append(arrivals[index])
                index += 1

            if not running and not pending:
                if index < len(arrivals):
                    self.now = max(self.now, arrivals[index].arrival_s)
                    continue
                break

            self.iterations += 1
            iteration_time = 0.0

            # 2. 本轮开始时的可解码集合（新准入的请求下一轮才 decode）
            decodable = [seq for seq in running if seq.ready]

            # 3. prefill 工作单位
            iteration_time = max(iteration_time, self._do_prefill_unit(running, pending))

            # 4. decode：可解码的序列各产出一个 token
            if decodable:
                step = 0.0
                for seq in decodable:
                    step = max(
                        step,
                        decode_step_time(
                            self.device,
                            self.spec,
                            len(decodable),
                            mfu=self.mfu,
                            weight_bits=self.weight_bits,
                            context_tokens=seq.context + seq.produced,
                        ),
                    )
                iteration_time = max(iteration_time, step)
                for seq in decodable:
                    seq.produced += 1
                    seq.context += 1

            self.now += iteration_time
            self.batch_sizes.append(len([s for s in running if s.ready]))

            # 5. 收尾：产出够了的序列离开运行批
            for seq in list(running):
                if seq.ready and seq.produced >= seq.request.output_tokens():
                    self._finish(seq, running)

        self.completions.sort(key=lambda c: c.finish_s)
        return self.completions

    def _do_prefill_unit(self, running: list[_Seq], pending: list[Request]) -> float:
        """执行一个 prefill 工作单位，返回它占用的时间。

        分块模式：优先推进已经准入但没 prefill 完的序列（按策略排序），
        每次推进 ``chunk_size`` 个 token；没有在途的才准入新请求。
        非分块模式：一次准入一条请求并把它的 prompt 一次算完。
        """
        if self.chunked_prefill:
            inflight = [seq for seq in running if not seq.ready]
            if inflight:
                seq = self._pick_inflight(inflight)
                chunk = min(self.chunk_size, seq.request.prompt_tokens - seq.prefilled)
                unit = prefill_time(
                    self.device,
                    self.spec,
                    chunk,
                    mfu=self.mfu,
                    weight_bits=self.weight_bits,
                )
                seq.prefilled += chunk
                if seq.prefilled >= seq.request.prompt_tokens:
                    seq.ready = True
                    seq.prefill_end_s = self.now + unit
                return unit

        request = self._select_pending(pending)
        if request is None:
            return 0.0
        if not self._can_admit(running, request):
            # 装不下就放回队列，等有序列跑完腾位置
            pending.insert(0, request)
            return 0.0

        seq = _Seq(
            request=request,
            enqueue_s=self.now,
            prefilled=0,
            context=request.prompt_tokens,
        )
        running.append(seq)

        if self.chunked_prefill:
            chunk = min(self.chunk_size, request.prompt_tokens)
            unit = prefill_time(
                self.device, self.spec, chunk, mfu=self.mfu, weight_bits=self.weight_bits
            )
            seq.prefilled = chunk
            if seq.prefilled >= request.prompt_tokens:
                seq.ready = True
                seq.prefill_end_s = self.now + unit
            return unit

        unit = prefill_time(
            self.device,
            self.spec,
            request.prompt_tokens,
            mfu=self.mfu,
            weight_bits=self.weight_bits,
        )
        seq.prefilled = request.prompt_tokens
        seq.ready = True
        seq.prefill_end_s = self.now + unit
        return unit

    def _pick_inflight(self, inflight: list[_Seq]) -> _Seq:
        """挑一个继续 prefill 的在途序列。

        短作业优先策略下，短请求可以插到长请求的剩余分块前面——
        这正是"分块 prefill 的收益取决于调度策略"的原因。
        """
        if self.policy == "sjf":
            inflight.sort(key=lambda s: s.request.prompt_tokens - s.prefilled)
        elif self.policy == "priority":
            inflight.sort(key=lambda s: (s.request.priority, s.enqueue_s))
        else:
            inflight.sort(key=lambda s: s.enqueue_s)
        return inflight[0]

    def _finish(self, seq: _Seq, running: list[_Seq]) -> None:
        """一条序列跑完：记账并把它移出运行批。"""
        produced = seq.produced
        self.completions.append(
            Completion(
                request=seq.request,
                enqueue_s=seq.enqueue_s,
                prefill_end_s=seq.prefill_end_s,
                finish_s=self.now,
                ttft_s=seq.prefill_end_s - seq.request.arrival_s,
                tpot_s=(self.now - seq.prefill_end_s) / max(1, produced - 1)
                if produced > 1
                else 0.0,
            )
        )
        running.remove(seq)

    # -- 指标 -------------------------------------------------------------

    def metrics(self) -> dict[str, float]:
        """本引擎跑完之后的指标表。"""
        batch = self.batch_sizes
        return summarize(
            self.completions,
            extra={
                "iterations": float(self.iterations),
                "avg_batch": (sum(batch) / len(batch)) if batch else 0.0,
            },
        )


# ---------------------------------------------------------------- 静态批引擎


@dataclass
class _StaticSeq:
    """静态批里的一条序列。"""

    request: Request
    produced: int = 0
    context: int = 0
    finished_s: float | None = None
    """跑完的时刻；用来统计空转的槽位时间。"""


class StaticBatchEngine:
    """静态批处理引擎（简化版）：固定槽位、批次内不补位。

    三条规矩，正好定义了"静态批"这个概念：

    1. **凑批**：引擎空闲时等一批请求到齐（``batch_window_s`` 是等待上限）；
    2. **整批并行**：批内所有请求一起 prefill，耗时取最慢的那条；
    3. **不补位**：批内某条请求先跑完，它的槽位就空着，直到整批结束才释放。

    第 3 条是静态批吞吐低的根源，也是第 1 条是静态批 TTFT 尖刺的根源。
    """

    def __init__(
        self,
        device=None,
        spec=None,
        *,
        max_batch: int = 16,
        kv_budget_gib: float = 6.0,
        mfu: float = 0.5,
        weight_bits: int | None = None,
        batch_window_s: float = 0.05,
    ) -> None:
        self.device = device if device is not None else gpu("h100_sxm")
        self.spec = spec if spec is not None else model("qwen2.5-7b")
        self.max_batch = max_batch
        self.kv_budget_gib = kv_budget_gib
        self.mfu = mfu
        self.weight_bits = weight_bits
        self.batch_window_s = batch_window_s

        self.now = 0.0
        self.completions: list[Completion] = []
        self.batches = 0
        self.idle_slot_seconds = 0.0
        """槽位空转的总时长——静态批浪费掉的算力。"""
        self.slot_seconds = 0.0
        """槽位占用的总时长（空转 + 干活），用来算空转率。"""

    def run(self, requests: list[Request], *, horizon: float = 600.0) -> list[Completion]:
        """跑完一批请求，返回完成情况（按完成时间排序）。"""
        self.now = 0.0
        self.completions = []
        self.batches = 0
        self.idle_slot_seconds = 0.0
        self.slot_seconds = 0.0

        arrivals = sorted(requests, key=lambda r: r.arrival_s)
        pending: list[Request] = []
        index = 0

        while index < len(arrivals) or pending:
            if self.now > horizon:
                break

            while index < len(arrivals) and arrivals[index].arrival_s <= self.now:
                pending.append(arrivals[index])
                index += 1

            if not pending:
                self.now = max(self.now, arrivals[index].arrival_s)
                continue

            # 1. 凑批：等到批满或者等待窗口耗尽
            window_deadline = self.now + self.batch_window_s
            while (
                len(pending) < self.max_batch
                and index < len(arrivals)
                and self.now < window_deadline
            ):
                self.now = max(self.now, min(arrivals[index].arrival_s, window_deadline))
                while index < len(arrivals) and arrivals[index].arrival_s <= self.now:
                    pending.append(arrivals[index])
                    index += 1

            batch = pending[: self.max_batch]
            pending = pending[self.max_batch :]
            self.batches += 1

            # 2. 整批并行 prefill：耗时取批内最慢的一条
            prefill = max(
                prefill_time(
                    self.device,
                    self.spec,
                    req.prompt_tokens,
                    mfu=self.mfu,
                    weight_bits=self.weight_bits,
                )
                for req in batch
            )
            self.now += prefill
            prefill_end = self.now

            seqs = [
                _StaticSeq(
                    request=req,
                    context=req.prompt_tokens,
                )
                for req in batch
            ]

            # 3. 整批 decode：谁先跑完谁停下，但槽位不释放
            while True:
                alive = [s for s in seqs if s.finished_s is None]
                if not alive:
                    break
                step = 0.0
                for seq in alive:
                    step = max(
                        step,
                        decode_step_time(
                            self.device,
                            self.spec,
                            len(alive),
                            mfu=self.mfu,
                            weight_bits=self.weight_bits,
                            context_tokens=seq.context + seq.produced,
                        ),
                    )
                self.now += step
                for seq in alive:
                    seq.produced += 1
                    seq.context += 1
                    if seq.produced >= seq.request.output_tokens():
                        seq.finished_s = self.now

            # 4. 记账：完成情况 + 槽位空转
            batch_start = prefill_end - prefill
            for seq in seqs:
                produced = seq.produced
                self.completions.append(
                    Completion(
                        request=seq.request,
                        enqueue_s=batch_start,
                        prefill_end_s=prefill_end,
                        finish_s=seq.finished_s if seq.finished_s is not None else self.now,
                        ttft_s=prefill_end - seq.request.arrival_s,
                        tpot_s=(self.now - prefill_end) / max(1, produced - 1)
                        if produced > 1
                        else 0.0,
                    )
                )
            # 槽位空转：某条请求跑完之后，它的槽位一直空到整批结束
            batch_span = self.now - batch_start
            self.slot_seconds += batch_span * len(batch)
            self.idle_slot_seconds += sum(
                max(0.0, self.now - (seq.finished_s if seq.finished_s is not None else self.now))
                for seq in seqs
            )

        self.completions.sort(key=lambda c: c.finish_s)
        return self.completions

    def metrics(self) -> dict[str, float]:
        """本引擎跑完之后的指标表。"""
        return summarize(
            self.completions,
            extra={
                "batches": float(self.batches),
                "idle_slot_ratio": (self.idle_slot_seconds / self.slot_seconds)
                if self.slot_seconds > 0
                else 0.0,
            },
        )


# ---------------------------------------------------------------- 断言


def check_static_batch_tail_is_driven_by_longest() -> tuple[dict, dict]:
    """静态批的 p99 由同批最长请求决定，不是由平均长度决定。

    要证明的是：批内一条长 prompt 会把整批的 TTFT 一起抬高，
    所以静态批的尾延迟由"最坏的一条"说了算。
    """
    workload = build_mixed_workload()
    static = StaticBatchEngine(max_batch=16, batch_window_s=0.05)
    static.run(workload)

    metrics = static.metrics()
    # 静态批里，交互请求的 TTFT 应该明显被同批的长请求拖高
    interactive = _subset_metrics(static.completions, lambda c: c.request.tenant == "interactive")
    longform = _subset_metrics(static.completions, lambda c: c.request.tenant == "longform")

    ensure(
        metrics["ttft_p99_ms"] > metrics["ttft_p50_ms"] * 2,
        f"静态批的 p99 TTFT（{metrics['ttft_p99_ms']:.1f}ms）应该远大于 p50"
        f"（{metrics['ttft_p50_ms']:.1f}ms）——静态批的尾延迟由最长的那条决定",
    )
    ensure(
        interactive["ttft_p50_ms"] > 0
        and longform["ttft_p99_ms"] >= interactive["ttft_p50_ms"],
        "静态批里长请求的 TTFT 不该低于同批交互请求的中位数——说明整批一起等最慢的",
    )
    return metrics, interactive


def check_continuous_batching_tames_the_tail() -> tuple[dict, dict, dict]:
    """连续批把 p99 压到接近 p50：谁跑完谁的位置立刻让出去。

    静态批必须"等整批结束"，连续批不需要，所以尾延迟差距应该显著。
    """
    workload = build_mixed_workload()

    static = StaticBatchEngine(max_batch=16, batch_window_s=0.05)
    static.run(workload)
    static_metrics = static.metrics()

    continuous = ContinuousBatchEngine(max_batch=16, policy="fcfs")
    continuous.run(workload)
    continuous_metrics = continuous.metrics()

    static_ratio = static_metrics["ttft_p99_ms"] / max(static_metrics["ttft_p50_ms"], 1e-9)
    continuous_ratio = continuous_metrics["ttft_p99_ms"] / max(
        continuous_metrics["ttft_p50_ms"], 1e-9
    )

    ensure(
        continuous_ratio < static_ratio,
        f"连续批的 p99/p50（{continuous_ratio:.2f}）应该小于静态批（{static_ratio:.2f}）",
    )
    ensure(
        continuous_ratio <= 2.0,
        f"连续批应该把 p99 压到接近 p50，实际 p99/p50 = {continuous_ratio:.2f}",
    )
    ensure(
        continuous_metrics["ttft_p99_ms"] < static_metrics["ttft_p99_ms"],
        f"连续批的 p99 TTFT（{continuous_metrics['ttft_p99_ms']:.1f}ms）应该低于静态批"
        f"（{static_metrics['ttft_p99_ms']:.1f}ms）",
    )
    return static_metrics, continuous_metrics, {"static_ratio": static_ratio, "continuous_ratio": continuous_ratio}


def check_chunked_prefill_protects_running_sequences() -> tuple[dict, dict]:
    """分块 prefill 保护的是"正在生成的请求"，不是 TTFT。

    长 prompt 一次性 prefill 会把整卡卡住 100 多毫秒，
    正在 decode 的序列这段时间一个 token 都产不出来；
    分块之后，decode 与 chunk 交织，停顿被摊到每个 chunk 上。
    """
    workload = build_mixed_workload()

    monolithic = ContinuousBatchEngine(max_batch=16, chunked_prefill=False)
    monolithic.run(workload)
    mono_metrics = monolithic.metrics()

    chunked = ContinuousBatchEngine(max_batch=16, chunked_prefill=True, chunk_size=512)
    chunked.run(workload)
    chunked_metrics = chunked.metrics()

    ensure(
        chunked_metrics["tpot_p99_ms"] < monolithic_metrics["tpot_p99_ms"],
        f"分块 prefill 的 TPOT p99（{chunked_metrics['tpot_p99_ms']:.1f}ms）应该低于"
        f"整块 prefill（{monolithic_metrics['tpot_p99_ms']:.1f}ms）——"
        "长 prompt 不再把正在生成的序列卡住",
    )
    ensure(
        chunked_metrics["e2e_p99_ms"] <= monolithic_metrics["e2e_p99_ms"] * 1.05,
        "分块 prefill 不该让端到端时延明显变差（它只是把同一份算力摊开）",
    )
    return monolithic_metrics, chunked_metrics


def check_preemption_recompute_cost() -> float:
    """抢占重算的代价 = 一次完整 prefill，用 labkit 的 prefill_time 记。

    这是第 7 章给第 8 章的伏笔：能抢回来的前提是重算够便宜，
    而重算成本随 prompt 长度线性上升，所以长请求不该随便抢。
    """
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    short_cost = prefill_time(device, spec, 512, mfu=0.5)
    long_cost = prefill_time(device, spec, 4096, mfu=0.5)

    ensure(
        long_cost > short_cost * 3,
        f"重算 4096 token 的代价（{long_cost * 1000:.1f}ms）应该远高于"
        f"512 token（{short_cost * 1000:.1f}ms）——抢占长请求要先想清楚重算成本",
    )
    ensure(
        prefill_time(device, spec, 4096, mfu=0.5) > prefill_time(device, spec, 2048, mfu=0.5),
        "prefill 代价应该随 prompt 变长而上升",
    )
    return long_cost


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 7 章的全部实验与断言。"""
    device = gpu("h100_sxm")
    spec = model("qwen2.5-7b")

    banner("第 7 章 · 连续批处理与调度内核")
    rule(f"设备：{device.name}　模型：{spec.name}　口径：{device.hbm_bandwidth_gbs:.0f} GB/s 带宽")

    section("一、负载：一批长短混合的请求")
    workload = build_mixed_workload()
    print_table(
        ["租户", "条数", "prompt tokens", "output tokens", "到达区间(s)"],
        [
            [
                tenant,
                len(items),
                f"{min(r.prompt_tokens for r in items)}~{max(r.prompt_tokens for r in items)}",
                f"{min(r.max_output_tokens for r in items)}~{max(r.max_output_tokens for r in items)}",
                f"{min(r.arrival_s for r in items):.3f}~{max(r.arrival_s for r in items):.3f}",
            ]
            for tenant, items in sorted(
                {t: [r for r in workload if r.tenant == t] for t in {r.tenant for r in workload}}.items()
            )
        ],
    )
    rule(f"单条 4096 token 的 prefill 成本：{fmt_ms(prefill_time(device, spec, 4096, mfu=0.5))}")

    section("二、静态批 vs 连续批")
    static_metrics, interactive = check_static_batch_tail_is_driven_by_longest()
    static_metrics, continuous_metrics, ratios = check_continuous_batching_tames_the_tail()
    print_table(
        ["方案", "TTFT p50(ms)", "TTFT p99(ms)", "p99/p50", "e2e p99(ms)", "输出 token/s", "完成时间(s)"],
        [
            [
                "静态批",
                f"{static_metrics['ttft_p50_ms']:.1f}",
                f"{static_metrics['ttft_p99_ms']:.1f}",
                f"{ratios['static_ratio']:.2f}",
                f"{static_metrics['e2e_p99_ms']:.1f}",
                f"{static_metrics['output_tokens_per_s']:.1f}",
                f"{static_metrics['makespan_s']:.3f}",
            ],
            [
                "连续批",
                f"{continuous_metrics['ttft_p50_ms']:.1f}",
                f"{continuous_metrics['ttft_p99_ms']:.1f}",
                f"{ratios['continuous_ratio']:.2f}",
                f"{continuous_metrics['e2e_p99_ms']:.1f}",
                f"{continuous_metrics['output_tokens_per_s']:.1f}",
                f"{continuous_metrics['makespan_s']:.3f}",
            ],
        ],
    )
    rule(
        f"静态批交互请求的 TTFT 中位数 {interactive['ttft_p50_ms']:.1f}ms，"
        f"槽位空转率 {static_metrics['idle_slot_ratio'] * 100:.1f}%"
    )

    section("三、分块 prefill 保护的是谁")
    mono_metrics, chunk_metrics = check_chunked_prefill_protects_running_sequences()
    print_table(
        ["方案", "TTFT p50(ms)", "TTFT p99(ms)", "TPOT p50(ms)", "TPOT p99(ms)", "e2e p99(ms)", "输出 token/s"],
        [
            [
                "整块 prefill",
                f"{mono_metrics['ttft_p50_ms']:.1f}",
                f"{mono_metrics['ttft_p99_ms']:.1f}",
                f"{mono_metrics['tpot_p50_ms']:.1f}",
                f"{mono_metrics['tpot_p99_ms']:.1f}",
                f"{mono_metrics['e2e_p99_ms']:.1f}",
                f"{mono_metrics['output_tokens_per_s']:.1f}",
            ],
            [
                "分块 prefill(512)",
                f"{chunk_metrics['ttft_p50_ms']:.1f}",
                f"{chunk_metrics['ttft_p99_ms']:.1f}",
                f"{chunk_metrics['tpot_p50_ms']:.1f}",
                f"{chunk_metrics['tpot_p99_ms']:.1f}",
                f"{chunk_metrics['e2e_p99_ms']:.1f}",
                f"{chunk_metrics['output_tokens_per_s']:.1f}",
            ],
        ],
    )

    section("四、抢占与重算的代价")
    recompute = check_preemption_recompute_cost()
    rule(
        f"被抢占的序列要重算：4096 token 的 prefill 一次要 {fmt_ms(recompute)}，"
        "长请求不是随便能抢的"
    )
