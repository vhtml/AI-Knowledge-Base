"""第 16 章实验：生产化——部署形态与上线检查单。

这个模块解决什么问题：第 16 章讲的是"demo 跑通之后，离上线还差什么"。
前面 15 章补的是 **Agent 内部**的正确性——循环、工具、上下文、记忆、可靠性、
可观测性、评测、安全、审批、成本；这一章补的是**运行时之外**那一层：
任务怎么排队、状态怎么活过进程重启、并发超了怎么办、新旧版本怎么灰度和回滚、
上线前那张检查单怎么变成一道**闸门**。

所以本章把九件事变成可执行的断言：

1. **后台任务形态**——提交 → pending → running → succeeded，状态序列可断言；
2. **状态持久化**——状态写盘，重新构造运行时对象（模拟进程重启）后仍能取回进度；
3. **任务状态机**——非法转移（succeeded → running）被拒绝；
4. **并发与限流**——执行侧超并发就排队等待，提交侧队列满就明确拒绝（429）；
5. **幂等**——同一个幂等键重复提交只产生一个任务；
6. **灰度**——1000 个请求按用户稳定分流到两个版本；
7. **回滚**——一键切回后全部流量走旧版本；
8. **上线检查单门禁**——缺 1–2 项就阻断，补齐就通过；
9. **模型版本 pin**——请求里的 model 字段是常量，不随请求、不随环境变量变化。

另外补三件生产必答、但不那么好断言的事：**多租户隔离**、
**配置与密钥走环境变量**、**运维指标**。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..labkit import (
    ChatModel,
    Message,
    ModelReply,
    Tool,
    ToolCall,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    tool,
)
from ..runtime import Agent

# ---------------------------------------------------------------- 常量与状态机

#: 模型版本必须 pin 死。升级走独立流程（新版本 + 评测 + 灰度），
#: 而不是改一个环境变量重启一下——后者会让同一份代码产生两种行为，无法复现。
PINNED_MODEL = "agent-prod-2024-05-01"

TaskState = Literal[
    "pending",
    "running",
    "waiting_approval",
    "succeeded",
    "failed",
    "cancelled",
]

#: 任务状态机的合法转移表。
#:
#: 终态没有任何出边——"跑完了就别再动它"必须是硬保证。否则一次误操作就能把
#: succeeded 改回 running，任务重跑一遍，写操作类的工具就会重复扣一次款。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "cancelled"}),
    "running": frozenset({"waiting_approval", "succeeded", "failed", "cancelled"}),
    "waiting_approval": frozenset({"running", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

#: 检查单必须覆盖的全部小节——对应 `AI/Agent/附录-B-工程检查单.md` 的十节。
REQUIRED_SECTIONS: tuple[str, ...] = (
    "一、该不该用",
    "二、工具与提示",
    "三、上下文与记忆",
    "四、可靠性与安全",
    "五、人在环路",
    "六、可观测性与评测",
    "七、成本与延迟",
    "八、上线与运维",
    "九、框架迁移",
    "十、终审三问",
)


class IllegalTransition(RuntimeError):
    """非法状态转移。"""


class TenantIsolationError(PermissionError):
    """跨租户访问。"""


# ---------------------------------------------------------------- 工具


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    orders = {"A-1001": "已发货", "A-1002": "待付款"}
    return orders.get(order_id, "未找到")


#: 本章演示用的工具集。真实系统里工具按租户授权，见 :class:`TenantToolRegistry`。
ALL_TOOLS: tuple[Tool, ...] = (查询订单,)


@dataclass
class ExplodingModel:
    """一个必定抛异常的模型。

    用来制造"任务失败"这个真实分支——生产上最常见的失败不是模型不会答，
    而是上游超时、限流、进程被杀。
    """

    message: str = "上游返回 500"

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """直接抛错，模拟上游故障。"""
        raise RuntimeError(self.message)


def _model_factory(answer: str = "订单 A-1001 已发货，报告已生成") -> Callable[[Task], ChatModel]:
    """造一个"每个任务一份独立脚本化模型"的工厂。

    为什么不用一个共享的模型实例：``ScriptedModel`` 的游标是全局的，
    上一个任务会把下一个任务的脚本推进去，导致同一批任务走出不同轨迹——
    排查问题时这是最恼人的噪声。生产上的等价物是**每请求一个会话**。
    """

    def factory(task: Task) -> ChatModel:
        return scripted_model(
            [
                ModelReply(tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})]),
                ModelReply(content=answer),
            ]
        )

    return factory


# ---------------------------------------------------------------- 任务与存储


@dataclass
class Task:
    """一个后台任务。

    ``history`` 记的是**状态流转序列**，不是消息历史。生产排障第一眼看的就是它：
    "这个任务到底走到哪一步停的"。``transcript`` 则是第 11 章那条轨迹的摘要，
    两者一起落盘，进程重启后还能拼出"它当时干了什么"。
    """

    id: str
    tenant_id: str
    idempotency_key: str
    payload: str
    state: TaskState = "pending"
    result: str = ""
    error: str = ""
    stop_reason: str = ""
    attempts: int = 0
    version: str = "v1"
    history: list[str] = field(default_factory=lambda: ["pending"])
    transcript: list[str] = field(default_factory=list)

    def transition(self, to_state: TaskState) -> None:
        """按状态机转移；非法转移直接抛错。

        不做"容错地忽略"——非法转移要么是代码 bug，要么是并发写坏了状态，
        两种都必须当场暴露，而不是悄悄把状态改成另一个样子。
        """
        allowed = ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if to_state not in allowed:
            reachable = "、".join(sorted(allowed)) if allowed else "无（终态）"
            raise IllegalTransition(
                f"任务 {self.id} 不能从 {self.state} 转到 {to_state}；合法去向：{reachable}"
            )
        self.state = to_state
        self.history.append(to_state)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 化的字典。"""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "idempotency_key": self.idempotency_key,
            "payload": self.payload,
            "state": self.state,
            "result": self.result,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "attempts": self.attempts,
            "version": self.version,
            "history": list(self.history),
            "transcript": list(self.transcript),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """从落盘的字典还原任务。"""
        return cls(
            id=data["id"],
            tenant_id=data["tenant_id"],
            idempotency_key=data["idempotency_key"],
            payload=data.get("payload", ""),
            state=data.get("state", "pending"),
            result=data.get("result", ""),
            error=data.get("error", ""),
            stop_reason=data.get("stop_reason", ""),
            attempts=data.get("attempts", 0),
            version=data.get("version", "v1"),
            history=list(data.get("history", ["pending"])),
            transcript=list(data.get("transcript", [])),
        )


class TaskStore:
    """任务状态存储：写 JSON 文件，进程重启后仍能读回来。

    为什么教学用文件、生产用 SQLite/Postgres：**接口一样，位置不同**。
    内存态一重启，队列里所有 pending 的任务就没了，用户看到的是"任务凭空消失"——
    这是生产事故里最常见也最难解释的一类。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tasks: dict[str, Task] = {}
        self._index: dict[tuple[str, str], str] = {}
        self._load()

    def _load(self) -> None:
        """从磁盘恢复全部任务。文件不存在就是全新队列，不是错误。"""
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw.get("tasks", []):
            task = Task.from_dict(item)
            self._tasks[task.id] = task
            self._index[(task.tenant_id, task.idempotency_key)] = task.id

    def _flush(self) -> None:
        """把全部任务写回磁盘。

        先写临时文件再替换：写到一半崩溃不会把状态文件变成半截 JSON，
        那会让下次启动直接读不出来——状态文件坏了，等于所有任务都丢了。
        """
        payload = {"tasks": [t.to_dict() for t in self._tasks.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def put(self, task: Task) -> None:
        """写入或更新一个任务，并立即落盘。"""
        self._tasks[task.id] = task
        self._index[(task.tenant_id, task.idempotency_key)] = task.id
        self._flush()

    def get(self, task_id: str) -> Task | None:
        """按 id 取任务，不存在返回 None。"""
        return self._tasks.get(task_id)

    def find_by_key(self, tenant_id: str, idempotency_key: str) -> Task | None:
        """按（租户, 幂等键）取任务——幂等去重的索引。"""
        task_id = self._index.get((tenant_id, idempotency_key))
        return self._tasks.get(task_id) if task_id else None

    def list_tasks(self, tenant_id: str | None = None) -> list[Task]:
        """列出任务；给了租户就只返回该租户的——隔离在查询层做，不在调用方做。"""
        if tenant_id is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.tenant_id == tenant_id]

    @property
    def count(self) -> int:
        """任务总数。"""
        return len(self._tasks)


# ---------------------------------------------------------------- 限流


class ConcurrencyLimiter:
    """执行侧并发闸门。

    超过上限时**不执行也不丢**，只是这一轮不领新任务——任务留在队列里等。
    这就是"排队"语义；"拒绝"语义在 :class:`QueueDepthLimiter` 和
    :meth:`TaskRuntime.submit` 里。
    """

    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max_concurrent
        self.active = 0
        self.queued = 0

    def try_acquire(self) -> bool:
        """尝试占一个并发槽位；满了返回 False。"""
        if self.active >= self.max_concurrent:
            self.queued += 1
            return False
        self.active += 1
        return True

    def release(self) -> None:
        """归还一个并发槽位。"""
        if self.active > 0:
            self.active -= 1


@dataclass
class QueueDepthLimiter:
    """提交侧闸门：队列深度超过上限就明确拒绝（429）。

    排队不能无限排——队列本身也要占内存。满了就立刻告诉调用方"稍后重试"，
    比默默堆着、最后 OOM 掉强得多。
    """

    max_depth: int
    rejected: int = 0

    def admit(self, depth: int) -> bool:
        """当前深度还能不能接一个新任务。"""
        if depth > self.max_depth:
            self.rejected += 1
            return False
        return True


@dataclass
class UpstreamLimiter:
    """上游模型的并发闸门。

    上游 429 不是你的错，也不是任务失败——该做的是退避后重排（第 10 章）。
    所以这里只做"不让上游被我们打爆"这一件事，失败判定不在这一层。
    """

    max_concurrent: int
    active: int = 0
    throttled: int = 0

    def try_acquire(self) -> bool:
        """尝试占用一个上游槽位。"""
        if self.active >= self.max_concurrent:
            self.throttled += 1
            return False
        self.active += 1
        return True

    def release(self) -> None:
        """归还上游槽位。"""
        if self.active > 0:
            self.active -= 1


# ---------------------------------------------------------------- 配置与指标


@dataclass(frozen=True)
class DeploymentConfig:
    """部署配置。

    注意 ``model`` **不来自环境变量**：版本 pin 死之后，升级就必须是"发一个新版本"，
    而不是"改个环境变量重启一下"。后者会让同一份代码产生两种行为，出事时无法复现。
    """

    env_name: str = "local"
    api_key: str | None = None
    max_concurrent: int = 4
    max_queue_depth: int = 100
    upstream_max_concurrent: int = 4
    model: str = PINNED_MODEL

    @classmethod
    def from_env(cls) -> DeploymentConfig:
        """从环境变量读配置。

        密钥缺失就留空，**不留兜底默认值**——默认密钥意味着生产上
        "忘了配"和"配错了"会静悄悄地连上某个不该连的端点。
        """
        raw_concurrency = os.getenv("AGENT_LAB_MAX_CONCURRENCY", "").strip()
        raw_depth = os.getenv("AGENT_LAB_MAX_QUEUE_DEPTH", "").strip()
        return cls(
            env_name=os.getenv("AGENT_LAB_ENV", "local"),
            api_key=os.getenv("AGENT_LAB_API_KEY") or None,
            max_concurrent=int(raw_concurrency) if raw_concurrency else 4,
            max_queue_depth=int(raw_depth) if raw_depth else 100,
        )


@dataclass
class Metrics:
    """运维指标。

    告警看的就是这几个数：成功率、成本、轮次、停止原因分布。
    这里只实现前两类计数，轮次与停止原因落在任务的 ``attempts`` 和 ``stop_reason`` 上。
    """

    submitted: int = 0
    duplicates: int = 0
    rejected: int = 0
    queued: int = 0
    succeeded: int = 0
    failed: int = 0
    upstream_throttled: int = 0

    @property
    def success_rate(self) -> float:
        """已结束任务里的成功率。还没结束的不进分母。"""
        finished = self.succeeded + self.failed
        return self.succeeded / finished if finished else 0.0

    def stop_reasons(self, tasks: Sequence[Task]) -> dict[str, int]:
        """停止原因分布——"为什么停"比"停没停"更能定位问题。"""
        counts: dict[str, int] = {}
        for task in tasks:
            key = task.stop_reason or task.state
            counts[key] = counts.get(key, 0) + 1
        return counts


# ---------------------------------------------------------------- 运行时


@dataclass
class SubmitResult:
    """提交任务的返回。

    状态是显式枚举而不是异常：客户端要拿到一个能直接转成 HTTP 状态码的结果，
    而不是被一个异常打断。``duplicate`` 也算接受——幂等命中不是错误。
    """

    status: str
    task: Task | None
    reason: str = ""
    http_status: int = 202

    @property
    def accepted(self) -> bool:
        """是否拿到了可用任务。"""
        return self.status in {"accepted", "duplicate"}


class TaskRuntime:
    """后台任务 + 队列形态的 Agent 运行时。

    这一章的主角。和第 2 章那个同步 Agent 的差别只有一处，但这一处改变了
    **全部**的接口设计：这里没有"请求进来直接返回结果"，而是"提交拿一个 task_id，
    之后轮询状态"。
    """

    def __init__(
        self,
        store_path: Path,
        config: DeploymentConfig,
        model_factory: Callable[[Task], ChatModel],
        tools: Sequence[Tool] = (),
    ) -> None:
        self.store = TaskStore(store_path)
        self.config = config
        self.model_factory = model_factory
        self.tools = list(tools)
        self.limiter = ConcurrencyLimiter(config.max_concurrent)
        self.queue_gate = QueueDepthLimiter(config.max_queue_depth)
        # 上游模型商给的是另一个额度：自己通常要自我限流在它下面，
        # 这样两个闸门不会同时踩到——踩到了就分不清是谁在限。
        self.upstream = UpstreamLimiter(config.upstream_max_concurrent)
        self.metrics = Metrics()
        self._counter = self._highest_task_number()

    # ------------------------------------------------------------ 提交

    def submit(self, payload: str, *, tenant_id: str, idempotency_key: str) -> SubmitResult:
        """提交一个任务。

        幂等：同一租户同一个幂等键重复提交，只会得到同一个任务。
        客户端重试、网络抖动、用户手抖，全收敛到这一条上。
        """
        existing = self.store.find_by_key(tenant_id, idempotency_key)
        if existing is not None:
            self.metrics.duplicates += 1
            return SubmitResult("duplicate", existing, "幂等键命中，返回已有任务", 200)

        depth = len(self.store.list_tasks(tenant_id))
        if not self.queue_gate.admit(depth):
            self.metrics.rejected += 1
            return SubmitResult(
                "rejected",
                None,
                f"429 队列已满（深度上限 {self.queue_gate.max_depth}），请稍后重试",
                429,
            )

        self._counter += 1
        task = Task(
            id=f"t-{self._counter:04d}",
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        self.store.put(task)
        self.metrics.submitted += 1
        return SubmitResult("accepted", task, "已入队", 202)

    # ------------------------------------------------------------ 执行

    def claim(self) -> Task | None:
        """从队列里领一个任务：pending → running。

        并发已满就返回 None——**任务不能丢**，它留在 pending 里等下一轮。
        上游被限流时同理：这一轮不领，退避后重排。
        """
        if not self.upstream.try_acquire():
            self.metrics.upstream_throttled += 1
            return None
        if not self.limiter.try_acquire():
            self.upstream.release()
            self.metrics.queued += 1
            return None

        task = next((t for t in self.store.list_tasks() if t.state == "pending"), None)
        if task is None:
            self.limiter.release()
            self.upstream.release()
            return None

        task.transition("running")
        task.attempts += 1
        self.store.put(task)
        return task

    def execute(self, task: Task) -> Task:
        """把一个 running 任务跑完。"""
        try:
            agent = Agent(self.model_factory(task), tools=self.tools)
            result = agent.run(task.payload)
        except Exception as exc:  # noqa: BLE001 - 任务失败是业务结果，不是程序崩溃
            task.transition("failed")
            task.error = f"{type(exc).__name__}: {exc}"
            task.stop_reason = "error"
            self.metrics.failed += 1
        else:
            task.transition("succeeded")
            task.result = result.final_answer
            task.stop_reason = result.stop_reason
            task.transcript = [f"{step.kind}:{step.label}" for step in result.steps]
            self.metrics.succeeded += 1
        finally:
            self.limiter.release()
            self.upstream.release()
            self.store.put(task)
        return task

    def run_next(self) -> Task | None:
        """领一个并跑完——后台 worker 主循环的一轮。"""
        task = self.claim()
        if task is None:
            return None
        return self.execute(task)

    # ------------------------------------------------------------ 查询

    def get(self, task_id: str, *, tenant_id: str | None = None) -> Task:
        """取任务；给了租户就做租户校验。"""
        task = self.store.get(task_id)
        if task is None:
            raise KeyError(f"没有这个任务：{task_id}")
        if tenant_id is not None and task.tenant_id != tenant_id:
            raise TenantIsolationError(
                f"任务 {task_id} 属于租户 {task.tenant_id}，租户 {tenant_id} 无权访问"
            )
        return task

    def _highest_task_number(self) -> int:
        """从已有任务里推出编号起点——重启后不能把 id 分配成已存在的值。"""
        numbers = [
            int(task.id.split("-", 1)[1])
            for task in self.store.list_tasks()
            if task.id.startswith("t-") and task.id.split("-", 1)[1].isdigit()
        ]
        return max(numbers, default=0)


# ---------------------------------------------------------------- 灰度路由


@dataclass
class ReleaseRouter:
    """灰度路由：按用户稳定分桶，新旧版本并行。

    为什么按**用户**而不是按请求：同一个用户两次请求落到不同版本，
    他的会话就被劈开了——上下文、记忆、任务状态全都对不上。
    """

    stable: str = "v1"
    weights: dict[str, int] = field(default_factory=lambda: {"v1": 100})
    total_weight: int = 100
    scenario_overrides: dict[str, str] = field(default_factory=dict)

    def pick(self, user_id: str, scenario: str = "") -> str:
        """决定这个用户走哪个版本。同一用户的结果永远一致。"""
        if scenario and scenario in self.scenario_overrides:
            return self.scenario_overrides[scenario]
        bucket = _bucket(user_id, self.total_weight)
        cumulative = 0
        for version, weight in self.weights.items():
            cumulative += weight
            if bucket < cumulative:
                return version
        return self.stable

    def canary(self, percent: int, version: str = "v2") -> None:
        """开一个百分比的灰度。"""
        if not 0 <= percent <= 100:
            raise ValueError(f"灰度比例必须在 0–100 之间，收到 {percent}")
        self.weights = {self.stable: 100 - percent, version: percent}

    def rollback(self) -> None:
        """一键切回稳定版本。"""
        self.weights = {self.stable: 100}

    def distribution(self, user_ids: Sequence[str]) -> dict[str, int]:
        """统计一批用户的版本分布。"""
        counts: dict[str, int] = {}
        for user_id in user_ids:
            version = self.pick(user_id)
            counts[version] = counts.get(version, 0) + 1
        return counts


def _bucket(user_id: str, total_weight: int) -> int:
    """把用户稳定地映射到 [0, total_weight) 的一个桶。

    用哈希而不是内置 ``hash()``：内置哈希对字符串是随机化的，
    换个进程就变了——灰度必须跨进程稳定，否则同一个用户一会儿新一会儿旧。
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % total_weight


# ---------------------------------------------------------------- 上线门禁


@dataclass(frozen=True)
class ChecklistItem:
    """检查单里的一条。

    ``satisfied`` 是可判定的布尔值——不是"要考虑并发问题"，
    而是"压测时 100 并发下 P99 延迟低于 800ms"。不可判定的项一律算不通过。
    """

    section: str
    title: str
    satisfied: bool
    evidence: str = ""


@dataclass(frozen=True)
class GateResult:
    """门禁判定结果。"""

    passed: bool
    total: int
    missing: tuple[str, ...] = ()
    uncovered_sections: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        """给 CI 用的退出码：通过 0，阻断 1。"""
        return 0 if self.passed else 1

    def report(self) -> str:
        """人看的报告——门禁要能说清"卡在哪一条"。"""
        if self.passed:
            return f"门禁通过：{self.total} 项检查全部满足"
        lines = [f"门禁未通过：{self.total} 项检查，{len(self.missing)} 项未满足"]
        lines.extend(f"  - 未满足：{item}" for item in self.missing)
        lines.extend(f"  - 未覆盖小节：{section}" for section in self.uncovered_sections)
        return "\n".join(lines)


class ReleaseGate:
    """上线闸门：检查单不通过，就不许发布。

    这是附录 B 的可执行版本——把"逐条过一遍"变成一道有退出码的闸门，
    缺项就阻断，而不是靠上线前那天的记性。
    """

    def __init__(self, items: Sequence[ChecklistItem]) -> None:
        self.items = list(items)

    def evaluate(self) -> GateResult:
        """判定：既要有未满足的项，也要有覆盖全十个小节。"""
        missing = tuple(
            f"[{item.section}] {item.title}" for item in self.items if not item.satisfied
        )
        covered = {item.section for item in self.items}
        uncovered = tuple(s for s in REQUIRED_SECTIONS if s not in covered)
        return GateResult(
            passed=not missing and not uncovered,
            total=len(self.items),
            missing=missing,
            uncovered_sections=uncovered,
        )


def _canonical_items() -> tuple[ChecklistItem, ...]:
    """附录 B 十条小节的机器可判定版本。

    每条都写成能回答"是/否"的形式——凡是要靠人判断的，
    在这一层就已经不成立，要么补成可判定的条件，要么先不算通过。
    """
    return (
        ChecklistItem("一、该不该用", "任务的步骤事先不可知，否则改用工作流", True),
        ChecklistItem("一、该不该用", "失败可接受，且有明确的完成判据", True),
        ChecklistItem("一、该不该用", "单次任务的成本上限已经算成数字", True),
        ChecklistItem("二、工具与提示", "每个参数都有字段级 description", True),
        ChecklistItem("二、工具与提示", "工具数量不超过 15，超出则分组暴露", True),
        ChecklistItem("二、工具与提示", "工具报错回填给模型前经过改写", True),
        ChecklistItem("三、上下文与记忆", "有显式的上下文预算表", True),
        ChecklistItem("三、上下文与记忆", "工具结果截断带显式标记", True),
        ChecklistItem("三、上下文与记忆", "记忆有写入/检索/遗忘/用户更正四条路径", True),
        ChecklistItem("四、可靠性与安全", "429 用指数退避加抖动", True),
        ChecklistItem("四、可靠性与安全", "写操作工具幂等或被标记为不可重试", True),
        ChecklistItem("四、可靠性与安全", "高危工具在沙箱里执行", True),
        ChecklistItem("五、人在环路", "审批按动作风险触发，不按工具名触发", True),
        ChecklistItem("五、人在环路", "审批超时的默认策略是拒绝", True),
        ChecklistItem("五、人在环路", "中断后能从暂停点恢复", True),
        ChecklistItem("六、可观测性与评测", "每次运行有 trace，且无明文密钥", True),
        ChecklistItem("六、可观测性与评测", "黄金集样本不少于 30 条", True),
        ChecklistItem("六、可观测性与评测", "回归门禁在 CI 里，分数低于阈值就阻断", True),
        ChecklistItem("七、成本与延迟", "单次任务成本上限是数字而非描述", True),
        ChecklistItem("七、成本与延迟", "提示前缀稳定，能命中缓存", True),
        ChecklistItem("七、成本与延迟", "流式首字延迟达标", True),
        ChecklistItem("八、上线与运维", "选定了部署形态并说明理由", True),
        ChecklistItem("八、上线与运维", "会话与任务状态持久化，重启不丢", True),
        ChecklistItem("八、上线与运维", "有并发上限和限流，超限有明确响应", True),
        ChecklistItem("八、上线与运维", "配置与密钥走环境变量，代码里没有硬编码", True),
        ChecklistItem("八、上线与运维", "模型版本已 pin 死，升级有单独流程", True),
        ChecklistItem("八、上线与运维", "有灰度开关，能一键回滚", True),
        ChecklistItem("九、框架迁移", "知道自己的运行时每一环对应框架的哪个概念", True),
        ChecklistItem("九、框架迁移", "迁移时评测集保持不变", True),
        ChecklistItem("十、终审三问", "坏掉时 5 分钟内能定位到是哪一步", True),
        ChecklistItem("十、终审三问", "烧掉的钱有硬上限", True),
        ChecklistItem("十、终审三问", "做错事时有人能拦住它", True),
    )


# ---------------------------------------------------------------- 工具权限


class TenantToolRegistry:
    """按租户授权工具。

    多租户系统里的"最小权限"（第 13 章）要落到租户维度：A 租户的 Agent
    看不到 B 租户的工具，更不该能调用它。
    """

    def __init__(self, grants: dict[str, tuple[str, ...]]) -> None:
        self._grants = grants

    def tools_for(self, tenant_id: str) -> list[Tool]:
        """这个租户被授权的工具。"""
        names = self._grants.get(tenant_id, ())
        return [t for t in ALL_TOOLS if t.name in names]


# ---------------------------------------------------------------- 请求体


def build_request_payload(payload: str, tools: Sequence[Tool] = ()) -> dict[str, Any]:
    """构造发给上游的请求体。

    ``model`` 取自 pin 死的常量，**不接受调用方传入**——
    否则一个拼错的参数就能让线上悄悄跑到别的模型上，而且没人会发现。
    """
    messages: list[Message] = [human(payload)]
    body: dict[str, Any] = {
        "model": PINNED_MODEL,
        "messages": [m.to_api() for m in messages],
        "temperature": 0.0,
    }
    if tools:
        body["tools"] = [t.to_api() for t in tools]
        body["stream"] = False
    return body


# ---------------------------------------------------------------- 断言场景


def check_background_task_shape(workspace: Path) -> None:
    """后台任务形态：提交 → pending → running → succeeded。"""
    section("后台任务形态：提交拿 id，轮询拿结果")

    runtime = TaskRuntime(
        workspace / "shape.json",
        DeploymentConfig(),
        _model_factory(),
        tools=ALL_TOOLS,
    )

    submitted = runtime.submit("查一下 A-1001 并出报告", tenant_id="t-tenant", idempotency_key="k-1")
    ensure(submitted.accepted, f"任务应当被接受，实际状态 {submitted.status}")
    ensure(submitted.task is not None, "接受的任务必须带 task 对象")
    task_id = submitted.task.id

    ensure(
        submitted.task.state == "pending",
        f"刚提交的任务应当是 pending，实际 {submitted.task.state}",
    )
    ensure(
        submitted.http_status == 202,
        f"提交成功应当返回 202，实际 {submitted.http_status}",
    )
    print(f"  提交 → {task_id}（{submitted.http_status}），状态 {submitted.task.state}")

    claimed = runtime.claim()
    ensure(claimed is not None, "队列里有任务时应当能领到一个")
    ensure(claimed.id == task_id, f"领到的应当是刚提交的任务，实际 {claimed.id}")
    ensure(
        claimed.state == "running",
        f"领走之后应当是 running，实际 {claimed.state}",
    )
    print(f"  领取 → 状态 {claimed.state}，第 {claimed.attempts} 次尝试")

    done = runtime.execute(claimed)
    ensure(
        done.state == "succeeded",
        f"跑完应当是 succeeded，实际 {done.state}（错误：{done.error}）",
    )
    ensure(bool(done.result), "成功任务必须有结果")
    ensure(
        "已发货" in done.result,
        f"结果应当来自工具调用之后的回答，实际 {done.result!r}",
    )
    ensure(
        done.transcript == ["model:模型", "tool:查询订单", "model:模型"],
        f"轨迹应当是 模型→工具→模型 三步，实际 {done.transcript}",
    )

    # 客户端拿到的是 task_id，轮询的是状态，不是结果本身
    polled = runtime.get(task_id)
    ensure(polled.state == "succeeded", f"轮询应当拿到 succeeded，实际 {polled.state}")
    ensure(
        polled.history == ["pending", "running", "succeeded"],
        f"状态流转序列应当是 pending→running→succeeded，实际 {polled.history}",
    )
    print(f"  完成 → 状态流转 {' → '.join(polled.history)}")
    print(f"  结果：{polled.result!r}")


def check_state_persistence(workspace: Path) -> None:
    """进程重启不能丢任务：状态写盘，重启后还能接着跑。"""
    section("状态持久化：进程重启不丢任务")

    store_path = workspace / "persist.json"
    runtime = TaskRuntime(store_path, DeploymentConfig(), _model_factory(), tools=ALL_TOOLS)

    submitted = runtime.submit("跑一个要几分钟的任务", tenant_id="t-tenant", idempotency_key="k-p")
    task_id = submitted.task.id
    claimed = runtime.claim()
    ensure(claimed is not None, "应当能领到任务")

    # 模拟进程崩溃：内存里的一切都消失，只剩磁盘上的状态文件
    del runtime
    ensure(store_path.is_file(), "任务状态必须已经写盘")

    restarted = TaskRuntime(store_path, DeploymentConfig(), _model_factory(), tools=ALL_TOOLS)
    recovered = restarted.get(task_id)
    ensure(
        recovered.state == "running",
        f"重启后未完成的任务应当仍然是 running，实际 {recovered.state}",
    )
    ensure(
        recovered.history == ["pending", "running"],
        f"重启后状态流转历史应当保留，实际 {recovered.history}",
    )
    ensure(
        recovered.payload == "跑一个要几分钟的任务",
        f"重启后任务内容应当保留，实际 {recovered.payload!r}",
    )
    print(f"  重启前 → {claimed.history}")
    print(f"  重启后 → {recovered.history}（任务没丢）")

    # 重启之后还能接着跑完——这才是持久化的意义
    finished = restarted.execute(recovered)
    ensure(
        finished.state == "succeeded",
        f"重启后应当能把任务接着跑完，实际 {finished.state}（错误：{finished.error}）",
    )
    ensure(bool(finished.result), "重启后跑完的任务也要有结果")
    ensure(
        restarted.store.count == 1,
        f"重启后不应产生新任务，总数应当仍是 1，实际 {restarted.store.count}",
    )
    print(f"  重启后接着跑完 → {finished.state}，结果 {finished.result!r}")


def check_state_machine() -> None:
    """任务状态机：非法转移必须被拒绝。"""
    section("任务状态机：非法转移被拒绝")

    task = Task(id="t-0001", tenant_id="t", idempotency_key="k", payload="p")
    ensure(task.state == "pending", "新任务应当是 pending")

    task.transition("running")
    task.transition("succeeded")
    ensure(task.state == "succeeded", f"应当已经 succeeded，实际 {task.state}")

    # 终态没有出边：succeeded → running 必须被拒绝
    try:
        task.transition("running")
    except IllegalTransition as exc:
        message = str(exc)
        ensure("succeeded" in message and "running" in message, f"异常信息要说清从哪到哪，实际 {message}")
        print(f"  拒绝 → {message}")
    else:
        raise AssertionError(
            f"succeeded → running 应当被拒绝，实际状态变成了 {task.state}"
        )

    ensure(task.state == "succeeded", f"被拒绝的转移不能改变状态，实际 {task.state}")
    ensure(
        task.history == ["pending", "running", "succeeded"],
        f"被拒绝的转移不能写进历史，实际 {task.history}",
    )

    # 终态集合里所有状态都没有出边
    for terminal in ("succeeded", "failed", "cancelled"):
        ensure(
            not ALLOWED_TRANSITIONS[terminal],
            f"终态 {terminal} 不该有出边，实际 {sorted(ALLOWED_TRANSITIONS[terminal])}",
        )

    # pending 不能直接跳到 succeeded：中间必须真的跑过
    fresh = Task(id="t-0002", tenant_id="t", idempotency_key="k2", payload="p")
    try:
        fresh.transition("succeeded")
    except IllegalTransition:
        print("  拒绝 → pending → succeeded（必须先 running）")
    else:
        raise AssertionError("pending → succeeded 应当被拒绝")


def check_rate_limit(workspace: Path) -> None:
    """并发与限流：执行侧排队，提交侧拒绝。"""
    section("并发与限流：超了要明确，不能默默干等")

    runtime = TaskRuntime(
        workspace / "rate.json",
        DeploymentConfig(max_concurrent=2),
        _model_factory(),
        tools=ALL_TOOLS,
    )

    ids = [
        runtime.submit(f"任务 {i}", tenant_id="t-tenant", idempotency_key=f"k-{i}").task.id
        for i in range(3)
    ]
    first = runtime.claim()
    second = runtime.claim()
    ensure(first is not None and second is not None, "并发上限是 2，应当能领到两个")
    ensure(runtime.limiter.active == 2, f"在跑的任务应当是 2 个，实际 {runtime.limiter.active}")

    # 第三个领不到：不是失败，是排队等待
    third = runtime.claim()
    ensure(third is None, f"并发已满时不该再领任务，实际领到了 {third}")
    ensure(runtime.limiter.queued == 1, f"应当记一次排队，实际 {runtime.limiter.queued}")
    ensure(
        runtime.get(ids[2]).state == "pending",
        f"没被领走的任务必须留在队列里，实际 {runtime.get(ids[2]).state}",
    )
    print(f"  执行侧：并发上限 2，第 3 个任务排队等待（状态仍是 {runtime.get(ids[2]).state}）")

    # 提交侧队列满了要明确拒绝，不能无限堆
    shallow = TaskRuntime(
        workspace / "rate2.json",
        DeploymentConfig(max_concurrent=1, max_queue_depth=1),
        _model_factory(),
    )
    shallow.submit("a", tenant_id="t-tenant", idempotency_key="k-a")
    shallow.submit("b", tenant_id="t-tenant", idempotency_key="k-b")
    rejected = shallow.submit("c", tenant_id="t-tenant", idempotency_key="k-c")
    ensure(not rejected.accepted, f"队列满时应当拒绝，实际 {rejected.status}")
    ensure(rejected.http_status == 429, f"队列满应当返回 429，实际 {rejected.http_status}")
    ensure("429" in rejected.reason, f"拒绝原因要写清状态码，实际 {rejected.reason!r}")
    ensure(shallow.store.count == 2, f"被拒绝的任务不该入库，实际 {shallow.store.count}")
    print(f"  提交侧：队列满 → {rejected.http_status} {rejected.reason}")

    # 上游模型限流：任务留在队列里等，不算失败
    upstream = TaskRuntime(
        workspace / "rate3.json",
        DeploymentConfig(max_concurrent=1),
        _model_factory(),
    )
    upstream.submit("a", tenant_id="t-tenant", idempotency_key="k-a")
    upstream.upstream.active = upstream.upstream.max_concurrent  # 上游槽位已被占满
    blocked = upstream.claim()
    ensure(blocked is None, "上游没空位时不该领任务")
    ensure(upstream.metrics.upstream_throttled == 1, f"应当记一次上游限流，实际 {upstream.metrics.upstream_throttled}")
    ensure(
        upstream.store.list_tasks()[0].state == "pending",
        "上游限流时任务应当留在队列里等，而不是被判失败",
    )
    print("  上游限流：任务留在队列里等，不记失败（退避重排见第 10 章）")


def check_idempotency(workspace: Path) -> None:
    """幂等：同一个幂等键重复提交只产生一个任务。"""
    section("幂等：重复提交只产生一个任务")

    runtime = TaskRuntime(workspace / "idem.json", DeploymentConfig(), _model_factory())

    first = runtime.submit("查 A-1001", tenant_id="t-tenant", idempotency_key="same-key")
    second = runtime.submit("查 A-1001", tenant_id="t-tenant", idempotency_key="same-key")

    ensure(first.status == "accepted", f"第一次提交应当被接受，实际 {first.status}")
    ensure(second.status == "duplicate", f"重复提交应当命中幂等，实际 {second.status}")
    ensure(first.task.id == second.task.id, "重复提交必须返回同一个任务")
    ensure(runtime.store.count == 1, f"重复提交后任务数应当仍是 1，实际 {runtime.store.count}")
    ensure(runtime.metrics.duplicates == 1, f"应当记一次幂等命中，实际 {runtime.metrics.duplicates}")
    print(f"  第一次 → {first.task.id}（{first.status}）")
    print(f"  第二次 → {second.task.id}（{second.status}），任务总数 {runtime.store.count}")

    # 幂等键的作用域是租户：不同租户用同一个键，是两个不同的任务
    other = runtime.submit("查 A-1001", tenant_id="t-other", idempotency_key="same-key")
    ensure(other.status == "accepted", f"别的租户应当是独立任务，实际 {other.status}")
    ensure(runtime.store.count == 2, f"跨租户不共享幂等键，实际 {runtime.store.count}")
    print(f"  换租户 → {other.task.id}（{other.status}），任务总数 {runtime.store.count}")


def check_canary() -> None:
    """灰度：按用户稳定分流，比例准确。"""
    section("灰度：按比例把流量切到新版本")

    router = ReleaseRouter(stable="v1")
    router.canary(percent=10, version="v2")

    users = [f"u-{i}" for i in range(1000)]
    counts = router.distribution(users)
    v2_ratio = counts.get("v2", 0) / len(users)

    ensure(
        set(counts) == {"v1", "v2"},
        f"应当只有两个版本有流量，实际 {sorted(counts)}",
    )
    ensure(
        abs(v2_ratio - 0.10) < 0.03,
        f"1000 个请求里 v2 应当约占 10%，实际 {v2_ratio:.1%}",
    )
    print(f"  分布：{counts}（v2 占 {v2_ratio:.1%}）")

    # 稳定性：同一个用户永远落同一个版本，否则他的会话会被劈开
    for user in users[:50]:
        ensure(
            router.pick(user) == router.pick(user),
            f"用户 {user} 的版本选择必须稳定",
        )
    ensure(
        router.pick("u-7") == router.pick("u-7"),
        "同一个用户的版本选择必须稳定",
    )
    print("  稳定性：同一用户始终落同一版本（会话不会被劈开）")

    # 按场景分流：某些场景整体切新版本
    router.scenario_overrides["批量报表"] = "v2"
    ensure(
        router.pick("u-1", scenario="批量报表") == "v2",
        "命中场景覆盖时应当走新版本",
    )
    ensure(
        router.pick("u-1", scenario="实时问答") == router.pick("u-1", scenario="实时问答"),
        "未命中场景时按比例分流，结果仍然稳定",
    )
    print("  按场景：命中场景覆盖的直接走新版本，其余按比例")


def check_rollback() -> None:
    """回滚：一键切回，全部流量走旧版本。"""
    section("回滚：一键切回旧版本")

    router = ReleaseRouter(stable="v1")
    router.canary(percent=30, version="v2")
    users = [f"u-{i}" for i in range(1000)]

    before = router.distribution(users)
    ensure(before.get("v2", 0) > 0, "回滚前应当有流量在 v2 上")

    router.rollback()
    after = router.distribution(users)
    ensure(
        after.get("v2", 0) == 0,
        f"回滚后不该还有流量在 v2 上，实际 {after.get('v2', 0)}",
    )
    ensure(after.get("v1", 0) == 1000, f"回滚后全部流量应当走 v1，实际 {after}")
    print(f"  回滚前：{before}")
    print(f"  回滚后：{after}（全部走 v1）")


def _index_of(items: Sequence[ChecklistItem], keyword: str) -> int:
    """按标题关键词找到检查项下标。

    用关键词而不是写死下标：检查单加一条或调个顺序，下标就全错位了，
    而错位会让"故意缺项"的实验变成"缺了另一项"——断言照样通过，门禁却是坏的。
    """
    for index, item in enumerate(items):
        if keyword in item.title:
            return index
    raise LookupError(f"检查单里没有包含 {keyword!r} 的条目")


def check_release_gate() -> None:
    """上线检查单门禁：缺项就阻断，补齐就通过。"""
    section("上线检查单门禁：缺一项就不许发版")

    items = list(_canonical_items())

    # 1) 全部满足 → 通过
    passed = ReleaseGate(items).evaluate()
    ensure(passed.passed, f"检查项全部满足时应当通过，实际未通过：{passed.report()}")
    ensure(passed.missing == (), f"不该有未满足项，实际 {passed.missing}")
    ensure(passed.exit_code == 0, f"通过时退出码应当是 0，实际 {passed.exit_code}")
    print(f"  全部满足 → 通过（{passed.total} 项，退出码 {passed.exit_code}）")

    # 2) 故意缺 1–2 项 → 阻断
    broken = list(items)
    for index, reason in ((_index_of(items, "模型版本"), "模型版本没 pin 死"),
                          (_index_of(items, "工具数量"), "工具数量 27 个")):
        broken[index] = ChecklistItem(
            broken[index].section, broken[index].title, False, reason
        )
    blocked = ReleaseGate(broken).evaluate()
    ensure(not blocked.passed, "缺项时门禁必须阻断")
    ensure(len(blocked.missing) == 2, f"应当报出 2 项未满足，实际 {len(blocked.missing)}")
    ensure(blocked.exit_code == 1, f"阻断时退出码应当是 1，实际 {blocked.exit_code}")
    ensure(
        any("模型版本" in item for item in blocked.missing),
        f"未满足项里应当包含模型版本那条，实际 {blocked.missing}",
    )
    ensure(
        any("工具数量" in item for item in blocked.missing),
        f"未满足项里应当包含工具数量那条，实际 {blocked.missing}",
    )
    print(f"  缺 2 项 → 阻断（退出码 {blocked.exit_code}）")
    for line in blocked.report().splitlines()[1:]:
        print(f"    {line.strip()}")

    # 3) 整节漏掉 → 覆盖不全也要阻断
    partial = [item for item in items if item.section != "十、终审三问"]
    uncovered = ReleaseGate(partial).evaluate()
    ensure(not uncovered.passed, "漏掉一整节时门禁必须阻断")
    ensure(
        uncovered.uncovered_sections == ("十、终审三问",),
        f"应当报出缺失的小节，实际 {uncovered.uncovered_sections}",
    )
    print(f"  漏一节 → 阻断（未覆盖 {uncovered.uncovered_sections}）")

    # 门禁覆盖附录 B 的全部十节
    ensure(
        {item.section for item in items} == set(REQUIRED_SECTIONS),
        "检查单必须覆盖附录 B 的全部十节",
    )
    ensure(len(items) >= 30, f"检查项应当不少于 30 条，实际 {len(items)}")


def check_model_pin() -> None:
    """模型版本 pin：请求里的 model 是常量。"""
    section("模型版本 pin：升级不能靠改环境变量")

    first = build_request_payload("查 A-1001", tools=ALL_TOOLS)
    second = build_request_payload("查 A-2002", tools=ALL_TOOLS)

    ensure(
        first["model"] == second["model"],
        f"不同请求的 model 必须一致，实际 {first['model']} vs {second['model']}",
    )
    ensure(
        first["model"] == PINNED_MODEL,
        f"model 应当是 pin 死的常量 {PINNED_MODEL}，实际 {first['model']}",
    )
    print(f"  两个不同请求 → model 都是 {first['model']!r}")

    # 请求内容里就算写了别的模型名，也不该改掉版本
    spoofed = build_request_payload("请用 gpt-4o-mini 回答")
    ensure(
        spoofed["model"] == PINNED_MODEL,
        f"请求内容不得改写模型版本，实际 {spoofed['model']}",
    )
    print("  请求内容里写别的模型名 → 版本不变")

    # 环境变量也不该改掉版本：升级要走独立流程
    with _env(AGENT_LAB_MODEL="gpt-4o-mini"):
        config = DeploymentConfig.from_env()
    ensure(
        config.model == PINNED_MODEL,
        f"配置里的模型版本必须 pin 死，实际 {config.model}",
    )
    print(f"  环境变量被改 → 版本仍是 {config.model!r}（升级走发版流程）")


def check_multi_tenant(workspace: Path) -> None:
    """多租户隔离：数据、工具权限都按租户隔离。"""
    section("多租户隔离：数据与工具权限")

    runtime = TaskRuntime(workspace / "tenant.json", DeploymentConfig(), _model_factory())
    mine = runtime.submit("我的任务", tenant_id="tenant-a", idempotency_key="k-1").task
    runtime.submit("别人的任务", tenant_id="tenant-b", idempotency_key="k-1")

    # 数据隔离：查询按租户过滤
    visible = runtime.store.list_tasks("tenant-a")
    ensure(len(visible) == 1, f"租户 A 应当只看到 1 个任务，实际 {len(visible)}")
    ensure(visible[0].id == mine.id, "租户 A 看到的应当是自己的任务")

    # 越权访问直接报错——不是返回 None，也不是悄悄返回别人的数据
    try:
        runtime.get(mine.id, tenant_id="tenant-b")
    except TenantIsolationError as exc:
        ensure("tenant-a" in str(exc), f"异常要说清是谁的任务，实际 {exc}")
        print(f"  越权读取 → 被拒绝（{exc}）")
    else:
        raise AssertionError("跨租户读取必须被拒绝")

    # 工具权限隔离：租户拿不到没授权的工具
    registry = TenantToolRegistry({"tenant-a": ("查询订单",), "tenant-b": ()})
    ensure(
        [t.name for t in registry.tools_for("tenant-a")] == ["查询订单"],
        "tenant-a 应当拿到被授权的工具",
    )
    ensure(
        registry.tools_for("tenant-b") == [],
        f"tenant-b 没有授权，不该拿到任何工具，实际 {registry.tools_for('tenant-b')}",
    )
    print("  工具权限：没授权的租户拿不到任何工具")


def check_config_and_secrets(workspace: Path) -> None:
    """配置与密钥：走环境变量，代码里没有硬编码。"""
    section("配置与密钥：全走环境变量")

    with _env():
        bare = DeploymentConfig.from_env()
    ensure(bare.api_key is None, f"没配密钥时应当是 None，实际 {bare.api_key!r}")
    ensure(bare.env_name == "local", f"默认环境名应当是 local，实际 {bare.env_name}")

    with _env(AGENT_LAB_API_KEY="unit-test-key-value", AGENT_LAB_ENV="staging"):
        staging = DeploymentConfig.from_env()
    ensure(
        staging.api_key == "unit-test-key-value",
        f"密钥应当从环境变量读出来，实际 {staging.api_key!r}",
    )
    ensure(staging.env_name == "staging", f"环境名应当来自环境变量，实际 {staging.env_name}")
    ensure(staging.env_name != bare.env_name, "不同环境的配置必须不同")
    print("  环境变量 → 密钥与环境名都读得到，不同环境配置不同")

    # 静态扫描源码：密钥不能硬编码在代码里
    source = Path(__file__)
    hits = scan_source_for_hardcoded_secrets(source)
    ensure(not hits, f"源码里不许出现硬编码密钥，实际命中：{hits}")
    print(f"  静态扫描 {source.name}：没有硬编码密钥")

    # 但扫描器不能是摆设：造一个带密钥的文件，它必须抓得到。
    # 密钥字面量在运行时拼出来——写死在本文件里的话，它自己就会被扫出来。
    planted = workspace / "planted_secret.py"
    planted_key = "sk-" + "abcdefghijklmnopqrstuvwx"
    planted.write_text(f'API_KEY = "{planted_key}"\n', encoding="utf-8")
    caught = scan_source_for_hardcoded_secrets(planted)
    ensure(len(caught) == 1, f"扫描器必须抓到硬编码密钥，实际抓到 {caught}")
    print(f"  反向验证：扫描器确实能抓到（{caught[0]}）")


def scan_source_for_hardcoded_secrets(path: Path) -> list[str]:
    """扫描源码里的硬编码密钥。

    这是检查单"配置与密钥走环境变量"那条的可执行版本——不靠人肉 review，
    靠机器扫。扫的是本仓库自己的实验代码，所以规则故意保守：
    只认"看起来像真密钥"的形态，避免把正常代码误判成泄露。
    """
    text = path.read_text(encoding="utf-8")
    patterns = {
        "疑似密钥字面量": r"sk-[A-Za-z0-9]{16,}",
        "硬编码 api_key": r"api_key\s*=\s*[\"'][^\"']{8,}[\"']",
    }
    hits: list[str] = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            # 只报位置和长度，不回显原文——扫描器的输出会进 CI 日志，
            # 把密钥原样打出来等于把它抄送了一份。
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{label}（第 {line} 行，{len(match.group(0))} 字符）")
    return hits


def check_metrics(workspace: Path) -> None:
    """运维指标：成功率和停止原因分布。"""
    section("运维指标：成功率与停止原因")

    runtime = TaskRuntime(workspace / "metrics.json", DeploymentConfig(), _model_factory())

    def factory(task: Task) -> ChatModel:
        """第三个任务用会抛异常的模型，制造一次真实失败。"""
        if task.idempotency_key == "k-fail":
            return ExplodingModel("上游返回 500")
        return scripted_model([ModelReply(content="完成")])

    runtime.model_factory = factory
    runtime.submit("a", tenant_id="t-tenant", idempotency_key="k-ok")
    runtime.submit("b", tenant_id="t-tenant", idempotency_key="k-fail")
    runtime.submit("c", tenant_id="t-tenant", idempotency_key="k-ok2")

    while runtime.run_next() is not None:
        pass  # worker 主循环：一直跑到队列空

    metrics = runtime.metrics
    ensure(metrics.submitted == 3, f"应当提交了 3 个任务，实际 {metrics.submitted}")
    ensure(metrics.succeeded == 2, f"应当成功 2 个，实际 {metrics.succeeded}")
    ensure(metrics.failed == 1, f"应当失败 1 个，实际 {metrics.failed}")
    ensure(
        abs(metrics.success_rate - 2 / 3) < 1e-9,
        f"成功率应当是 2/3，实际 {metrics.success_rate}",
    )

    tasks = runtime.store.list_tasks()
    finished = [t for t in tasks if t.state in {"succeeded", "failed"}]
    ensure(
        len(finished) == metrics.succeeded + metrics.failed,
        "已结束任务数必须和指标对上，否则告警会骗人",
    )
    ensure(
        all(t.state in {"succeeded", "failed"} for t in tasks),
        f"队列应当被跑空，实际还有 {[t.state for t in tasks]}",
    )

    reasons = metrics.stop_reasons(tasks)
    ensure(
        set(reasons) == {"completed", "error"},
        f"停止原因分布应当同时有 completed 和 error，实际 {reasons}",
    )
    print(f"  提交 {metrics.submitted}，成功 {metrics.succeeded}，失败 {metrics.failed}，成功率 {metrics.success_rate:.0%}")
    print(f"  停止原因分布：{reasons}")


# ---------------------------------------------------------------- 辅助


@contextmanager
def _env(**overrides: str) -> Iterator[None]:
    """临时设置环境变量，退出时恢复原值。"""
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_live() -> None:
    """真实模型路径：真跑一个后台任务，看状态流转和耗时。

    只校验形状，不校验内容——真模型的输出不确定。
    """
    banner("第 16 章 · 真实模型路径")
    import time

    from ..labkit import real_model

    with tempfile.TemporaryDirectory(prefix="agent_lab_ch16_live_") as tmp:
        workspace = Path(tmp)
        runtime = TaskRuntime(
            workspace / "live.json",
            DeploymentConfig.from_env(),
            lambda task: real_model(),
            tools=ALL_TOOLS,
        )

        submitted = runtime.submit("用两句话解释什么是幂等性", tenant_id="live", idempotency_key="live-1")
        ensure(submitted.accepted, f"提交应当被接受，实际 {submitted.status}")

        start = time.perf_counter()
        finished = runtime.run_next()
        elapsed = time.perf_counter() - start

        ensure(finished is not None, "应当把任务跑完")
        ensure(finished.state == "succeeded", f"任务应当成功，实际 {finished.state}（{finished.error}）")
        ensure(bool(finished.result), "成功任务必须有结果")
        ensure(
            finished.history == ["pending", "running", "succeeded"],
            f"状态流转应当完整，实际 {finished.history}",
        )
        print(f"  耗时 {elapsed:.2f}s")
        print(f"  状态流转：{' → '.join(finished.history)}")
        print(f"  结果：{finished.result[:120]!r}")


def run() -> None:
    """跑第 16 章实验。"""
    banner("第 16 章 · 生产化：部署形态与上线检查单")
    with tempfile.TemporaryDirectory(prefix="agent_lab_ch16_") as tmp:
        workspace = Path(tmp)
        check_background_task_shape(workspace)
        check_state_persistence(workspace)
        check_state_machine()
        check_rate_limit(workspace)
        check_idempotency(workspace)
        check_canary()
        check_rollback()
        check_release_gate()
        check_model_pin()
        check_multi_tenant(workspace)
        check_config_and_secrets(workspace)
        check_metrics(workspace)
    rule("形态按任务时长和反馈需求选；状态必须活过重启；上线检查单要变成有退出码的闸门")
    print("\n✓ 第 16 章通过")
