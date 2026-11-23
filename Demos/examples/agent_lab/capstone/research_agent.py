"""毕业作品：一个把 17 章护栏全部装上的研究助手 Agent。

这个模块解决什么问题：前面 17 章每章解决一个"这一环会怎么崩"的问题，
但它们是分开的。这一章（毕业作品）把它们**装进同一个运行时**，让它们互相约束：

- **第 5 章**：上下文预算——每一部分有配额，超了就压缩或拒绝
- **第 6 章**：记忆——跨会话的长期记忆，带 TTL 和用户更正
- **第 3 章**：工具——按风险分级，参数说明落到字段级
- **第 13、14 章**：权限与审批——危险操作在**执行层**拦截，模型绕不过
- **第 10 章**：可靠性——错误分类、重试上限、幂等
- **第 11 章**：轨迹——每一步都记，可回放
- **第 15 章**：成本——每次运行返回 token 与估算成本

这个运行时**不依赖任何 Agent 框架**，只用 ``labkit`` 的消息与模型抽象。
它的价值不在于代码，而在于你会看到：**每一章的护栏落在同一个文件的哪一行**。

默认路径完全离线：用脚本化模型驱动，断言严格、结果可复现。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ..labkit import (
    ChatModel,
    Message,
    ModelReply,
    ToolCall,
    ensure,
    tool_result,
)

__all__ = [
    "ApprovalRequest",
    "ApprovalGate",
    "Budget",
    "MemoryStore",
    "ResearchAgent",
    "ResearchResult",
    "RiskLevel",
    "ToolEntry",
    "TraceRecorder",
]


# ---------------------------------------------------------------- 风险分级（第 13 章）


class RiskLevel(IntEnum):
    """工具的风险等级。

    用整数而不是字符串，是为了让"阈值判断"变成简单的比较——
    审批规则就是"风险 ≥ 某阈值就要人点头"。
    """

    READ_ONLY = 0
    """只读：查、读、搜索。出错不造成外部影响。"""

    WRITE = 1
    """写入：创建、更新内容。可逆或半可逆。"""

    DANGEROUS = 2
    """危险：删除、对外发送、不可逆变更。"""


APPROVAL_THRESHOLD = RiskLevel.WRITE
"""默认审批阈值：风险达到 WRITE 就需要人点头。"""


@dataclass
class ToolEntry:
    """一个注册进 Agent 的工具。"""

    name: str
    description: str
    risk: RiskLevel
    func: Callable[..., str]
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )

    def invoke(self, arguments: dict[str, Any]) -> str:
        """执行工具。异常交给调用方统一处理（第 10 章）。"""
        result = self.func(**arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def to_api(self) -> dict[str, Any]:
        """转成 OpenAI 兼容接口的 tools 数组元素。

        和 :class:`labkit.Tool` 的形状保持一致——换框架时这一层不用改。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------- 上下文预算（第 5 章）


@dataclass
class Budget:
    """上下文预算表。

    六个配额之和必须等于总窗口——这是自检里的一条硬断言。
    预算不是"能塞多少"，是"每一部分分多少"。
    """

    total: int = 8000
    system: int = 500
    tool_schemas: int = 1200
    history: int = 3000
    tool_results: int = 1500
    retrieval: int = 1000
    output_reserved: int = 800

    def allocated(self) -> int:
        """已分配的配额总和。"""
        return (
            self.system
            + self.tool_schemas
            + self.history
            + self.tool_results
            + self.retrieval
            + self.output_reserved
        )

    def remaining(self) -> int:
        """尚未分配的余量——负数说明超配了。"""
        return self.total - self.allocated()


# ---------------------------------------------------------------- 记忆（第 6 章）


@dataclass
class MemoryEntry:
    """一条长期记忆。"""

    key: str
    content: str
    created_at: float
    expires_at: float | None = None
    importance: int = 1

    def is_expired(self, now: float) -> bool:
        """是否已过期。"""
        return self.expires_at is not None and now >= self.expires_at


_LATIN_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


def _terms(text: str) -> set[str]:
    """把文本切成可比较的"词"。

    英文按单词切；中文按 2-gram 切——中文没有空格，
    按 2-gram 切能在不引入分词器的前提下得到稳定的相似度。
    """
    terms = set(_LATIN_TOKEN.findall(text.lower()))
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


class MemoryStore:
    """长期记忆存储：写入 / 检索 / 遗忘 / 更正，四条路径齐全。

    刻意内置"用户更正"这一条：记忆污染比记不住严重得多，
    而没有更正路径的记忆系统一旦被污染就废了（第 6 章）。
    """

    def __init__(self, capacity: int = 50, ttl_seconds: float | None = None) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, MemoryEntry] = {}

    def write(
        self,
        key: str,
        content: str,
        importance: int = 1,
        ttl_seconds: float | None = None,
    ) -> MemoryEntry:
        """写入一条记忆；同 key 视为覆盖（旧信息被替换而不是堆积）。"""
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        entry = MemoryEntry(
            key=key,
            content=content,
            created_at=now,
            expires_at=(now + ttl) if ttl else None,
            importance=importance,
        )
        self._entries[key] = entry
        self._evict()
        return entry

    def correct(self, key: str, content: str) -> MemoryEntry:
        """用户更正一条记忆——污染的修复路径。"""
        return self.write(key, content, importance=2)

    def recall(self, query: str, limit: int = 5, now: float | None = None) -> list[MemoryEntry]:
        """按相关度召回记忆。

        相关度用**词与二元组**的重合度算：英文按词切，中文按 2-gram 切。
        不用向量检索是有意的——教学环境要的是"召回逻辑可以被断言"，
        而 2-gram 重合度足够可解释、也足够稳定。
        """
        moment = now if now is not None else time.time()
        query_terms = _terms(query)
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._entries.values():
            if entry.is_expired(moment):
                continue
            overlap = len(query_terms & _terms(entry.content))
            if overlap:
                scored.append((overlap + entry.importance, entry))
        scored.sort(key=lambda item: (-item[0], item[1].created_at))
        return [entry for _, entry in scored[:limit]]

    def forget(self, key: str) -> bool:
        """显式遗忘一条记忆（用户说"忘掉我说过的 X"）。"""
        return self._entries.pop(key, None) is not None

    def _evict(self) -> None:
        """容量淘汰：超了容量就丢最不重要、最旧的。"""
        while len(self._entries) > self.capacity:
            victim = min(self._entries.values(), key=lambda e: (e.importance, e.created_at))
            del self._entries[victim.key]

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------- 审批（第 14 章）


@dataclass
class ApprovalRequest:
    """一次待审批的操作。"""

    tool_name: str
    arguments: dict[str, Any]
    risk: RiskLevel
    reason: str = ""

    def approve_with(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """批准，并可**修改参数**——用修改后的参数执行。"""
        return dict(arguments if arguments is not None else self.arguments)

    def reject(self) -> str:
        """拒绝时的回填文案——要告诉模型为什么，它才能换条路。"""
        return f"操作被拒绝：{self.tool_name} 未获批准（{self.reason or '需要人工确认'}）。请改用其他方式完成。"


class ApprovalGate:
    """审批关卡：按**动作风险**触发，而不是按工具名。

    这是第 14 章的核心——按工具名分级的话，模型换个工具名就绕过了。
    """

    def __init__(self, threshold: RiskLevel = APPROVAL_THRESHOLD) -> None:
        self.threshold = threshold
        self.pending: list[ApprovalRequest] = []
        self.decisions: list[tuple[str, bool]] = []

    def needs_approval(self, tool: ToolEntry) -> bool:
        """这个工具的这次调用要不要人点头。"""
        return tool.risk >= self.threshold

    def request(self, tool: ToolEntry, arguments: dict[str, Any]) -> ApprovalRequest:
        """登记一条待审批请求。"""
        request = ApprovalRequest(
            tool_name=tool.name,
            arguments=dict(arguments),
            risk=tool.risk,
            reason=f"风险等级 {tool.risk.name} ≥ 阈值 {self.threshold.name}",
        )
        self.pending.append(request)
        return request

    def approve(self, request: ApprovalRequest, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """批准并返回（可能被修改过的）参数。"""
        self.decisions.append((request.tool_name, True))
        return request.approve_with(arguments)

    def reject(self, request: ApprovalRequest) -> str:
        """拒绝并返回给模型的回填文案。"""
        self.decisions.append((request.tool_name, False))
        return request.reject()


# ---------------------------------------------------------------- 轨迹（第 11 章）


@dataclass
class TraceEvent:
    """轨迹里的一个事件。"""

    kind: str
    payload: dict[str, Any]
    elapsed_ms: float = 0.0


class TraceRecorder:
    """轨迹记录器。

    记录一切，但不记录密钥——脱敏在写入时就做，不在读取时做，
    因为 trace 通常会被发到日志系统，事后补救来不及。
    """

    SENSITIVE_KEYS = {"api_key", "password", "token", "secret", "authorization"}

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._start = time.perf_counter()

    def record(self, event_type: str, **payload: Any) -> TraceEvent:
        """记一个事件，敏感字段就地脱敏。

        参数名用 ``event_type`` 而不是 ``kind``——否则 payload 里的
        ``kind="business"`` 会和位置参数撞名。
        """
        elapsed = (time.perf_counter() - self._start) * 1000
        event = TraceEvent(kind=event_type, payload=self._redact(payload), elapsed_ms=elapsed)
        self.events.append(event)
        return event

    def _redact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """把敏感字段的值替换掉。"""
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in self.SENSITIVE_KEYS:
                clean[key] = "***REDACTED***"
            elif isinstance(value, dict):
                clean[key] = self._redact(value)
            else:
                clean[key] = value
        return clean

    def of_kind(self, kind: str) -> list[TraceEvent]:
        """按类型取事件。"""
        return [event for event in self.events if event.kind == kind]


# ---------------------------------------------------------------- 运行结果


@dataclass
class ResearchResult:
    """一次运行的完整结果。"""

    final_answer: str = ""
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: TraceRecorder = field(default_factory=TraceRecorder)
    approvals: list[ApprovalRequest] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost(self, input_price: float = 2.5, output_price: float = 10.0) -> float:
        """按每百万 token 单价估算成本（美元）。"""
        return (
            self.prompt_tokens / 1_000_000 * input_price
            + self.completion_tokens / 1_000_000 * output_price
        )


# ---------------------------------------------------------------- 主运行时


class ResearchAgent:
    """研究助手 Agent——把 17 章的护栏装在一起。

    控制流还是那个 12 行循环（第 2 章）。区别在于每一环外面都套了护栏：

    ```
    while 没结束:
        模型决策                    ← 第 4 章：系统提示
        要调工具?
            风险够高? → 审批        ← 第 13、14 章：审批在执行层
            执行 → 失败分类重试     ← 第 3、10 章：工具与可靠性
            截断 + 记账             ← 第 5 章：预算
        收口                        ← 第 11、15 章：轨迹与成本
    ```
    """

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[ToolEntry],
        memory: MemoryStore | None = None,
        approval: ApprovalGate | None = None,
        budget: Budget | None = None,
        system_prompt: str = "",
        max_iterations: int = 8,
        max_tool_output_chars: int = 1500,
        max_retries: int = 2,
        auto_approve: bool = False,
    ) -> None:
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory or MemoryStore()
        self.approval = approval or ApprovalGate()
        self.budget = budget or Budget()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tool_output_chars = max_tool_output_chars
        self.max_retries = max_retries
        self.auto_approve = auto_approve

    # ------------------------------------------------------------ 工具执行

    def _execute_tool(
        self,
        entry: ToolEntry,
        arguments: dict[str, Any],
        result: ResearchResult,
    ) -> str:
        """执行一次工具调用——审批、错误分类、截断都在这里收口。"""
        # 第 14 章：审批在**执行层**，模型绕不过
        if self.approval.needs_approval(entry):
            request = self.approval.request(entry, arguments)
            result.approvals.append(request)
            result.trace.record("approval_requested", tool=entry.name, risk=entry.risk.name)
            if not self.auto_approve:
                # 未获批准 → 回填给模型，让它换条路，而不是崩掉
                message = self.approval.reject(request)
                result.trace.record("approval_rejected", tool=entry.name)
                return message
            arguments = self.approval.approve(request)
            result.trace.record("approval_granted", tool=entry.name)

        # 第 10 章：错误分类——参数错（模型能修）vs 业务故障（模型修不了）
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                output = entry.invoke(arguments)
            except TypeError as exc:
                last_error = f"参数错误：{exc}。请检查参数名与类型。"
                result.trace.record("tool_error", tool=entry.name, kind="parameter", attempt=attempt)
                break
            except Exception as exc:  # noqa: BLE001 - 业务故障要转成模型能读懂的话
                last_error = (
                    f"工具执行失败：{type(exc).__name__}: {exc}。"
                    "这是服务端故障，重试无用，请改用其他工具或直接告知用户。"
                )
                result.trace.record("tool_error", tool=entry.name, kind="business", attempt=attempt)
                break

            # 第 5 章：截断要带标记，模型才知道这不是全文
            if len(output) > self.max_tool_output_chars:
                output = (
                    output[: self.max_tool_output_chars]
                    + f"\n…（已截断，原始长度 {len(output)} 字符）"
                )
                result.trace.record("tool_result_truncated", tool=entry.name, original_length=len(output))
            result.trace.record("tool_result", tool=entry.name, length=len(output))
            return output

        return last_error or "工具执行失败，未知原因。"

    # ------------------------------------------------------------ 记忆注入

    def _memory_context(self, user_input: str) -> list[str]:
        """召回与当前任务相关的记忆。

        未召回的记忆**不进上下文**——这是第 6 章的一条硬规则。
        """
        recalled = self.memory.recall(user_input, limit=3)
        return [entry.content for entry in recalled]

    # ------------------------------------------------------------ 主循环

    def run(self, user_input: str, history: Sequence[Message] = ()) -> ResearchResult:
        """跑一次完整的研究任务。"""
        result = ResearchResult()
        result.trace.record("run_start", input=user_input)

        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        # 第 6 章：召回的记忆注入上下文
        memories = self._memory_context(user_input)
        if memories:
            messages.append(
                Message(
                    role="system",
                    content="关于这位用户的已知信息：\n" + "\n".join(f"- {m}" for m in memories),
                )
            )
            result.trace.record("memory_recalled", count=len(memories))

        messages.extend(history)
        messages.append(Message(role="user", content=user_input))

        tool_list = list(self.tools.values())

        for iteration in range(self.max_iterations):
            reply = self.model.complete(messages, tools=tool_list)
            result.prompt_tokens += reply.usage.get("prompt_tokens", 0)
            result.completion_tokens += reply.usage.get("completion_tokens", 0)

            result.trace.record(
                "model",
                iteration=iteration,
                content=reply.content[:200],
                tool_call_count=len(reply.tool_calls),
            )
            messages.append(reply.to_message())

            if not reply.tool_calls:
                result.final_answer = reply.content
                result.stop_reason = "completed"
                result.trace.record("run_end", stop_reason="completed")
                break

            for call in reply.tool_calls:
                result.tool_calls.append(call)
                entry = self.tools.get(call.name)
                if entry is None:
                    known = ", ".join(sorted(self.tools))
                    output = f"错误：没有名为 {call.name} 的工具。可用工具：{known}。"
                    result.trace.record("tool_unknown", tool=call.name)
                else:
                    output = self._execute_tool(entry, call.arguments, result)
                result.steps.append(f"{call.name}")
                messages.append(tool_result(output, tool_call_id=call.id, name=call.name))
        else:
            result.stop_reason = "max_iterations"
            result.trace.record("run_end", stop_reason="max_iterations")

        result.messages = messages
        return result
