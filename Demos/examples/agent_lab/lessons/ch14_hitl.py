"""第 14 章实验：人在环路——审批、接管与恢复。

这个模块解决什么问题：第 13 章把危险操作关进了权限层，但"关起来"只是问题的一半。
另一半是**人**：什么时候必须让人点头？人点了之后怎么往下走？人不在的时候怎么办？
人不想点头而想自己动手时，控制权怎么干净地交出去？

所以本章把九件事变成可执行的断言：

1. **按风险触发**——读操作不打断，写操作会打断（第 13 章的延续）；
2. **批准 → 继续执行**——副作用真的发生；
3. **拒绝 → 不执行**——模型收到一条能换路的回填；
4. **改参数后放行**——执行用的是改后的参数；
5. **超时策略**——默认拒绝，副作用不发生，台账里有超时记录；
6. **人工接管**——Agent 停止自动执行，把控制权交出去；
7. **恢复**——从暂停点继续，已完成的动作不重做；
8. **审批粒度**——一批同质操作一次点头；
9. **拒绝后不死循环**——重复请求被识别，第二轮就停。

基础类型全部复用毕业作品里的 :class:`RiskLevel` / :class:`ApprovalGate` /
:class:`ApprovalRequest` / :class:`ToolEntry`——本章只在它们之上补三件事：
**超时策略**、**挂起与恢复**、**批量与接管**。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..capstone.research_agent import (
    ApprovalGate,
    ApprovalRequest,
    RiskLevel,
    ToolEntry,
)
from ..labkit import (
    ChatModel,
    Message,
    ModelReply,
    ScriptedModel,
    ToolCall,
    banner,
    ensure,
    rule,
    scripted_model,
    section,
    tool_result,
)

TimeoutPolicy = Literal["reject", "allow", "hold"]
"""人没响应时的三种处理方式。

默认必须是 ``reject``：``allow`` 等于把审批变成橡皮章——人没看过的操作照样执行，
审批就只剩记录功能了。
"""


# ---------------------------------------------------------------- 审批的数据结构


@dataclass
class ApprovalResponse:
    """审批人对一次请求的回应。

    ``outcome="takeover"`` 表示"我不批，我自己做"——这是**接管**，不是拒绝：
    拒绝是"这件事别做"，接管是"这件事我来做"。
    """

    outcome: Literal["approve", "reject", "takeover"]
    arguments: dict[str, Any] | list[dict[str, Any]] | None = None
    """改参数后放行时填**改后的**参数；单条填 dict，批量填 list。"""
    reason: str = ""
    approver: str = "owner"


@dataclass
class ApprovalRecord:
    """一条审批台账。

    审批必须可审计：**谁批的、批了什么、什么时候**——三样缺一不可。
    没有台账，事后复盘就只能靠人回忆。
    """

    request_id: str
    tool_name: str
    outcome: Literal["approved", "rejected", "timeout", "taken_over"]
    approver: str
    arguments: dict[str, Any]
    """最终实际使用的参数（人改过就是改后的）。"""
    original_arguments: dict[str, Any]
    """模型原本请求的参数——和上一条一比就知道人改了什么。"""
    edited: bool
    reason: str
    at: float


@dataclass
class TrackedApproval(ApprovalRequest):
    """带审批单号与工具调用 id 的请求。

    继承毕业作品的 :class:`ApprovalRequest`，多带两样东西：
    审批单号用来串台账，调用 id 用来在恢复时把结果回填给正确的调用。
    """

    request_id: str = ""
    call_ids: list[str] = field(default_factory=list)


@dataclass
class BatchedApproval(TrackedApproval):
    """一批同质操作合并成的一条审批请求。

    审批粒度的关键手段：人一次点头覆盖 5 个同质动作，而不是点 5 次。
    """

    operations: list[dict[str, Any]] = field(default_factory=list)

    def approve_all(self, arguments: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """批准整批，返回每条操作实际要用的参数。"""
        if arguments is None:
            return [dict(op) for op in self.operations]
        return [dict(item) for item in arguments]


@dataclass
class ApprovalOutcome:
    """一次审批的结果。"""

    status: Literal["approved", "rejected", "timeout", "held", "taken_over"]
    arguments: dict[str, Any] | list[dict[str, Any]] = field(default_factory=dict)
    message: str = ""
    """回填给模型的文案；被拒绝或超时时非空。"""
    request: ApprovalRequest | None = None


# ---------------------------------------------------------------- 审批控制器


class HitlController:
    """审批控制器：把毕业作品的 :class:`ApprovalGate` 扩成完整的 HITL 关卡。

    capstone 的关卡只回答两个问题：要不要人点头、批不批。
    生产里还要回答三个：**人没来怎么办**、**人来了能不能改参数**、**批过留不留证据**。
    前两件里的"批不批"直接复用 capstone 的实现，后三件在这里补上。
    """

    def __init__(
        self,
        threshold: RiskLevel = RiskLevel.WRITE,
        timeout_policy: TimeoutPolicy = "reject",
        approver: str = "owner",
    ) -> None:
        self.gate = ApprovalGate(threshold=threshold)
        self.timeout_policy = timeout_policy
        self.approver = approver
        self.responder: Callable[[ApprovalRequest], ApprovalResponse | None] | None = None
        """审批人。返回 ``None`` 表示"人没响应"——超时策略接手。"""
        self.records: list[ApprovalRecord] = []
        self._counter = 0

    @property
    def pending(self) -> list[ApprovalRequest]:
        """登记过的审批请求。

        注意这是**台账**而不是未决队列——批过的请求也留在里面，
        这样"批过什么"才有据可查（第 11 章的 trace 是同一个道理）。
        """
        return self.gate.pending

    def needs_approval(self, tool: ToolEntry) -> bool:
        """按**动作风险**判断要不要人点头，而不是按工具名。"""
        return self.gate.needs_approval(tool)

    def request(
        self,
        tool: ToolEntry,
        arguments: dict[str, Any],
        call_ids: Sequence[str],
    ) -> TrackedApproval:
        """登记一条待审批请求。"""
        self._counter += 1
        request = TrackedApproval(
            request_id=f"ap-{self._counter:03d}",
            tool_name=tool.name,
            arguments=dict(arguments),
            risk=tool.risk,
            reason=f"风险等级 {tool.risk.name} ≥ 阈值 {self.gate.threshold.name}",
            call_ids=list(call_ids),
        )
        self.gate.pending.append(request)
        return request

    def request_batch(
        self,
        tool: ToolEntry,
        operations: Sequence[dict[str, Any]],
        call_ids: Sequence[str],
    ) -> BatchedApproval:
        """把一批同质操作登记成**一条**审批请求。"""
        self._counter += 1
        request = BatchedApproval(
            request_id=f"ap-{self._counter:03d}",
            tool_name=tool.name,
            arguments=dict(operations[0]) if operations else {},
            risk=tool.risk,
            reason=f"批量操作 {len(operations)} 次 · 风险等级 {tool.risk.name}",
            operations=[dict(op) for op in operations],
            call_ids=list(call_ids),
        )
        self.gate.pending.append(request)
        return request

    def await_decision(self, request: ApprovalRequest) -> ApprovalOutcome:
        """问一次审批人；人没响应就按超时策略处理。

        这里的"问一次"是**单点**：运行时不会反复追问同一个人——
        反复追问就是审批疲劳的温床（第 14 章的"坑"）。
        """
        response = self.responder(request) if self.responder is not None else None

        if response is None:
            return self._handle_timeout(request)

        if response.outcome == "takeover":
            self._record(request, "taken_over", {}, reason=response.reason, approver=response.approver)
            return ApprovalOutcome("taken_over", {}, "控制权已交给人工，Agent 停止自动执行。", request)

        if response.outcome == "reject":
            # 回填文案沿用 capstone 的：它明确写了"未获批准"，模型才知道该换条路
            message = self.gate.reject(request)
            self._record(request, "rejected", {}, reason=response.reason, approver=response.approver)
            return ApprovalOutcome("rejected", {}, message, request)

        if isinstance(request, BatchedApproval):
            overrides = response.arguments if isinstance(response.arguments, list) else None
            final = request.approve_all(overrides)
            self.gate.approve(request)
            self._record(
                request,
                "approved",
                final[0] if final else {},
                edited=overrides is not None,
                reason=response.reason,
                approver=response.approver,
            )
            return ApprovalOutcome("approved", final, "", request)

        final = dict(response.arguments if response.arguments is not None else request.arguments)
        self.gate.approve(request, response.arguments)
        self._record(
            request,
            "approved",
            final,
            edited=response.arguments is not None,
            reason=response.reason,
            approver=response.approver,
        )
        return ApprovalOutcome("approved", final, "", request)

    def _handle_timeout(self, request: ApprovalRequest) -> ApprovalOutcome:
        """人没响应：按策略处理。默认拒绝。"""
        if self.timeout_policy == "allow":
            final = dict(request.arguments)
            self.gate.approve(request)
            self._record(request, "approved", final, reason="超时自动放行——等于橡皮章")
            return ApprovalOutcome("approved", final, "", request)

        if self.timeout_policy == "hold":
            self._record(request, "timeout", {}, reason="超时挂起，等待人工")
            return ApprovalOutcome("held", {}, "", request)

        self.gate.reject(request)
        reason = f"审批超时未响应（策略 {self.timeout_policy}）"
        self._record(request, "timeout", {}, reason=reason)
        message = (
            f"操作未获批准：{request.tool_name} 的审批超时未响应（策略：{self.timeout_policy}）。"
            "请改用其他方式完成。"
        )
        return ApprovalOutcome("timeout", {}, message, request)

    def _record(
        self,
        request: ApprovalRequest,
        outcome: str,
        arguments: dict[str, Any],
        edited: bool = False,
        reason: str = "",
        approver: str | None = None,
    ) -> ApprovalRecord:
        """写一条审批台账。"""
        record = ApprovalRecord(
            request_id=getattr(request, "request_id", "ap-?"),
            tool_name=request.tool_name,
            outcome=outcome,  # type: ignore[arg-type]
            approver=approver or self.approver,
            arguments=dict(arguments),
            original_arguments=dict(request.arguments),
            edited=edited,
            reason=reason,
            at=time.time(),
        )
        self.records.append(record)
        return record


# ---------------------------------------------------------------- 运行时


@dataclass
class HitlCheckpoint:
    """一次可恢复的检查点。

    检查点要存够三样东西才能真的"从暂停点继续"：
    已经产出的消息历史、卡住的那组调用、以及那组调用对应的待审批请求。
    """

    messages: list[Message]
    iteration: int
    remaining: list[list[ToolCall]]
    """还没处理的工具调用组；第一组就是卡住的那一组。"""
    pending_request: ApprovalRequest | None
    executed: int
    """暂停前已经真正执行了多少次工具——恢复时用来证明"没有重做"。"""


@dataclass
class HitlResult:
    """一次运行的结果。"""

    final_answer: str = ""
    messages: list[Message] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    approvals: list[ApprovalRequest] = field(default_factory=list)
    taken_over: bool = False
    checkpoint: HitlCheckpoint | None = None


class HitlRuntime:
    """带人在环路能力的最小运行时。

    和 runtime.py 里的 :class:`Agent` 比，多了三件事：

    1. 工具执行前先过审批关卡（按动作风险，不按工具名）；
    2. 关卡可以**挂起**——运行时落检查点，把控制权交出去，而不是报错退出；
    3. 挂起之后可以**恢复**——从检查点继续，已完成的动作不重做。
    """

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[ToolEntry],
        controller: HitlController,
        max_iterations: int = 8,
        detect_repeat_rejection: bool = True,
        batch_homogeneous: bool = True,
    ) -> None:
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.controller = controller
        self.max_iterations = max_iterations
        self.detect_repeat_rejection = detect_repeat_rejection
        self.batch_homogeneous = batch_homogeneous
        self.messages: list[Message] = []
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.checkpoint: HitlCheckpoint | None = None
        self._iteration = 0
        self._rejected: set[str] = set()
        """被拒过的操作指纹（工具名 + 参数）。模型再提一次时要能认出来。"""

    # ------------------------------------------------------------ 入口

    def run(self, user_input: str) -> HitlResult:
        """跑一次任务。遇到挂起就落检查点返回，不抛异常。"""
        self.messages = [Message(role="user", content=user_input)]
        return self._loop(0)

    def resume(self) -> HitlResult:
        """从检查点恢复。

        关键：**不重跑已经完成的动作**。检查点里存着消息历史和卡住的那组调用，
        恢复时只补做那一组，然后从下一轮继续。
        """
        checkpoint = self.checkpoint
        if checkpoint is None:
            raise RuntimeError("没有可恢复的检查点")
        if checkpoint.pending_request is None:
            raise RuntimeError("检查点里没有待审批的请求")

        self.messages = list(checkpoint.messages)
        result = HitlResult(messages=list(self.messages))

        outcome = self.controller.await_decision(checkpoint.pending_request)

        if outcome.status == "taken_over":
            return self._settle(result, outcome)
        if outcome.status == "held":
            self.checkpoint = HitlCheckpoint(
                messages=list(self.messages),
                iteration=checkpoint.iteration,
                remaining=checkpoint.remaining,
                pending_request=checkpoint.pending_request,
                executed=len(self.executed),
            )
            result.checkpoint = self.checkpoint
            return self._settle(result, outcome)
        if outcome.status in {"rejected", "timeout"}:
            self._backfill(checkpoint.remaining[0], outcome.message)
            self._mark_rejected(checkpoint.remaining[0])
            return self._settle(result, outcome)

        # 批准：先补做卡住的那一组，再处理同轮剩下的组
        self._run_group(checkpoint.remaining[0], outcome.arguments)
        rest = checkpoint.remaining[1:]
        if rest:
            gate = self._process_groups(rest, result)
            if gate.status in {"held", "taken_over", "repeat_rejected"}:
                return self._settle(result, gate)

        return self._loop(checkpoint.iteration + 1, result)

    # ------------------------------------------------------------ 主循环

    def _loop(self, iteration: int, result: HitlResult | None = None) -> HitlResult:
        """主循环。和第 2 章的 12 行同构，多了"执行前过关卡"这一环。"""
        result = result or HitlResult()
        tool_list = list(self.tools.values())

        for index in range(iteration, self.max_iterations):
            self._iteration = index
            reply = self.model.complete(self.messages, tools=tool_list)
            self.messages.append(reply.to_message())

            if not reply.tool_calls:
                result.final_answer = reply.content
                result.stop_reason = "completed"
                result.messages = list(self.messages)
                return result

            groups = self._group_calls(reply.tool_calls)
            outcome = self._process_groups(groups, result)
            if outcome.status in {"held", "taken_over", "repeat_rejected"}:
                return self._settle(result, outcome)

        result.stop_reason = "max_iterations"
        result.messages = list(self.messages)
        return result

    def _group_calls(self, calls: Sequence[ToolCall]) -> list[list[ToolCall]]:
        """把一轮里的工具调用按"同工具同风险"分组。

        同质的批量操作合成一次审批——这是审批粒度的主要手段：
        人一次点头覆盖 5 个同质动作，而不是点 5 次。
        """
        groups: list[list[ToolCall]] = []
        index: dict[tuple[str, str], int] = {}
        for call in calls:
            if not self.batch_homogeneous:
                # 关掉合批：每个调用各自过一次关卡——这正是审批疲劳的来源
                groups.append([call])
                continue
            entry = self.tools.get(call.name)
            key = (call.name, entry.risk.name if entry is not None else "UNKNOWN")
            if key in index:
                groups[index[key]].append(call)
            else:
                index[key] = len(groups)
                groups.append([call])
        return groups

    def _process_groups(
        self,
        groups: Sequence[list[ToolCall]],
        result: HitlResult,
    ) -> ApprovalOutcome:
        """按组处理一轮里的工具调用。"""
        for index, group in enumerate(groups):
            outcome = self._handle_group(group, result)
            if outcome.status == "held":
                self.checkpoint = HitlCheckpoint(
                    messages=list(self.messages),
                    iteration=self._iteration,
                    remaining=[list(item) for item in groups[index:]],
                    pending_request=outcome.request,
                    executed=len(self.executed),
                )
                return outcome
            if outcome.status in {"taken_over", "repeat_rejected"}:
                return outcome
        return ApprovalOutcome("approved")

    def _handle_group(self, group: list[ToolCall], result: HitlResult) -> ApprovalOutcome:
        """处理一组（同工具同风险）的调用。"""
        first = group[0]
        entry = self.tools.get(first.name)

        if entry is None:
            known = ", ".join(sorted(self.tools)) or "（无）"
            for call in group:
                self.messages.append(
                    tool_result(f"错误：没有名为 {call.name} 的工具。可用工具：{known}。", call.id, call.name)
                )
            return ApprovalOutcome("approved")

        if not self.controller.needs_approval(entry):
            # 只读操作：直接执行，不打断人
            for call in group:
                self._execute(entry, call, call.arguments)
            return ApprovalOutcome("approved")

        # 需要审批：先看这个操作是不是刚被拒过
        if self.detect_repeat_rejection:
            blocked = [call for call in group if self._fingerprint(call) in self._rejected]
            if blocked:
                message = (
                    f"操作 {entry.name} 此前已被拒绝，重复请求不会再次执行。"
                    "请改用其他方式完成，或直接告诉用户这项操作无法进行。"
                )
                for call in blocked:
                    self.messages.append(tool_result(message, call.id, entry.name))
                return ApprovalOutcome("repeat_rejected", message=message, request=None)

        if len(group) == 1:
            request = self.controller.request(entry, first.arguments, [first.id])
        else:
            request = self.controller.request_batch(
                entry,
                [call.arguments for call in group],
                [call.id for call in group],
            )
        result.approvals.append(request)

        outcome = self.controller.await_decision(request)
        if outcome.status == "approved":
            self._run_group(group, outcome.arguments)
        elif outcome.status in {"rejected", "timeout"}:
            self._backfill(group, outcome.message)
            self._mark_rejected(group)
        return outcome

    # ------------------------------------------------------------ 执行细节

    def _run_group(self, group: Sequence[ToolCall], arguments: Any) -> None:
        """按（可能被改过的）参数执行一组调用。"""
        if isinstance(arguments, list):
            final = [dict(item) for item in arguments]
            if len(final) != len(group):
                final = [dict(call.arguments) for call in group]
        elif isinstance(arguments, dict) and arguments:
            final = [dict(arguments) for _ in group]
        else:
            final = [dict(call.arguments) for call in group]

        for call, args in zip(group, final):
            entry = self.tools.get(call.name)
            if entry is None:
                continue
            self._execute(entry, call, args)

    def _execute(self, entry: ToolEntry, call: ToolCall, arguments: dict[str, Any]) -> str:
        """真正执行一次工具调用并把结果回填给模型。"""
        try:
            output = entry.invoke(arguments)
        except Exception as exc:  # noqa: BLE001 - 工具故障要转成模型能读懂的话
            output = f"工具执行失败：{type(exc).__name__}: {exc}。请改用其他方式完成。"
        self.executed.append((entry.name, dict(arguments)))
        self.messages.append(tool_result(output, call.id, entry.name))
        return output

    def _backfill(self, group: Sequence[ToolCall], message: str) -> None:
        """把拒绝/超时的文案回填给模型——它得知道下一步该干什么。"""
        for call in group:
            self.messages.append(tool_result(message, call.id, call.name))

    def _mark_rejected(self, group: Sequence[ToolCall]) -> None:
        """记住被拒过的操作——模型再提一次要能认出来。"""
        for call in group:
            self._rejected.add(self._fingerprint(call))

    @staticmethod
    def _fingerprint(call: ToolCall) -> str:
        """操作指纹：工具名 + 参数。用来识别"同一个操作又被提了一遍"。"""
        return call.name + "|" + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)

    def _settle(self, result: HitlResult, outcome: ApprovalOutcome) -> HitlResult:
        """把关卡结果落到运行结果上。"""
        if outcome.status == "taken_over":
            result.taken_over = True
            result.stop_reason = "taken_over"
        elif outcome.status == "repeat_rejected":
            result.stop_reason = "repeat_rejected"
        elif outcome.status == "held":
            result.stop_reason = "awaiting_human"
            result.checkpoint = self.checkpoint
        result.approvals = list(self.controller.pending)
        result.messages = list(self.messages)
        return result


# ---------------------------------------------------------------- 工具


def build_tools(effects: dict[str, list[Any]]) -> list[ToolEntry]:
    """构造一组带风险分级的工具。

    ``effects`` 是副作用台账——所有断言都看它，而不是看"模型有没有发起调用"。
    模型发起调用 ≠ 操作真的发生，这正是第 13、14 章要盯的那条分界线。
    """

    def 读笔记(note_id: str) -> str:
        """按 ID 读取笔记内容。只读，不改变任何东西。"""
        effects.setdefault("read", []).append(note_id)
        return f"笔记 {note_id} 的正文"

    def 写笔记(title: str, content: str) -> str:
        """写入一篇笔记。会改变仓库内容，需要审批。"""
        effects.setdefault("written", []).append({"title": title, "content": content})
        return f"已写入《{title}》"

    def 删笔记(note_id: str) -> str:
        """删除一篇笔记。不可逆操作。"""
        effects.setdefault("deleted", []).append(note_id)
        return f"已删除 {note_id}"

    return [
        ToolEntry(
            name="读笔记",
            description="按 ID 读取笔记内容。",
            risk=RiskLevel.READ_ONLY,
            func=读笔记,
            parameters={
                "type": "object",
                "properties": {"note_id": {"type": "string", "description": "笔记 ID"}},
                "required": ["note_id"],
            },
        ),
        ToolEntry(
            name="写笔记",
            description="写入一篇笔记。会覆盖同名笔记。",
            risk=RiskLevel.WRITE,
            func=写笔记,
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "标题"},
                    "content": {"type": "string", "description": "正文"},
                },
                "required": ["title", "content"],
            },
        ),
        ToolEntry(
            name="删笔记",
            description="删除一篇笔记。不可逆操作。",
            risk=RiskLevel.DANGEROUS,
            func=删笔记,
            parameters={
                "type": "object",
                "properties": {"note_id": {"type": "string", "description": "笔记 ID"}},
                "required": ["note_id"],
            },
        ),
    ]


# ---------------------------------------------------------------- 1. 触发点


def check_trigger_by_risk() -> None:
    """读操作不打断，写操作打断——按动作风险，不按工具名。"""
    section("触发点：按动作风险触发，不按工具名")

    # 情况 A：只读操作 —— 全程不产生一条待审批
    effects_a: dict[str, list[Any]] = {}
    controller_a = HitlController(threshold=RiskLevel.WRITE)
    controller_a.responder = lambda request: ApprovalResponse(outcome="approve")
    model_a = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "读笔记", {"note_id": "n1"})]),
            ModelReply(content="笔记 n1 的正文是……"),
        ]
    )
    result_a = HitlRuntime(model_a, build_tools(effects_a), controller_a).run("读一下 n1")

    ensure(
        len(controller_a.pending) == 0,
        f"只读操作不该产生待审批请求，实际 {len(controller_a.pending)} 条",
    )
    ensure(
        effects_a.get("read") == ["n1"],
        f"只读操作应当直接执行，实际读了 {effects_a.get('read')}",
    )
    print(f"  只读：待审批 {len(controller_a.pending)} 条，副作用 {effects_a.get('read')}")

    # 情况 B：写操作 —— 产生一条待审批，且只此一条
    effects_b: dict[str, list[Any]] = {}
    controller_b = HitlController(threshold=RiskLevel.WRITE)
    controller_b.responder = lambda request: ApprovalResponse(outcome="approve")
    model_b = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "读笔记", {"note_id": "n1"})]),
            ModelReply(tool_calls=[ToolCall("c2", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="周报已写入"),
        ]
    )
    result_b = HitlRuntime(model_b, build_tools(effects_b), controller_b).run("整理周报")

    ensure(
        len(controller_b.pending) == 1,
        f"读 + 写只应当产生 1 条待审批（写那一条），实际 {len(controller_b.pending)} 条",
    )
    ensure(
        controller_b.pending[0].tool_name == "写笔记",
        f"待审批的应当是写操作，实际 {controller_b.pending[0].tool_name}",
    )
    ensure(
        controller_b.pending[0].risk == RiskLevel.WRITE,
        f"待审批项的风险级别应当是 WRITE，实际 {controller_b.pending[0].risk}",
    )
    ensure(
        effects_b.get("read") == ["n1"],
        f"读操作应当照常执行，实际读了 {effects_b.get('read')}",
    )
    print(f"  写操作：待审批 {len(controller_b.pending)} 条 → {controller_b.pending[0].tool_name}")

    # 触发点跟着风险走，不跟着工具名走：换个名字风险一样，照样要审批
    renamed = ToolEntry(
        name="save_note",
        description="写入一篇笔记（换了名字）。",
        risk=RiskLevel.WRITE,
        func=lambda title, content: "已写入",
    )
    ensure(
        controller_b.needs_approval(renamed),
        "换个工具名不该绕过审批——审批看的是动作风险",
    )
    print("  改名成 save_note 之后照样要审批 → 触发点跟的是风险，不是名字")


# ---------------------------------------------------------------- 2. 批准


def check_approve_executes() -> None:
    """批准 → 动作真的发生。"""
    section("批准：点头之后副作用真的发生")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(outcome="approve", approver="张三")
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="周报已写入"),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("写周报")

    ensure(
        effects.get("written") == [{"title": "周报", "content": "初稿"}],
        f"批准后副作用应当真的发生，实际 {effects.get('written')}",
    )
    ensure(
        result.final_answer == "周报已写入",
        f"批准后应当继续走到最终回答，实际 {result.final_answer!r}",
    )
    ensure(
        len(controller.records) == 1 and controller.records[0].outcome == "approved",
        f"台账应当记下一条批准，实际 {[(r.outcome, r.tool_name) for r in controller.records]}",
    )
    print(f"  副作用：{effects.get('written')}")
    print(f"  台账：{controller.records[0].outcome} / {controller.records[0].approver}")


# ---------------------------------------------------------------- 3. 拒绝


def check_reject_blocks() -> None:
    """拒绝 → 不执行，且模型收到一条能换路的回填。"""
    section("拒绝：不执行，但要让模型知道还能换条路")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(
        outcome="reject", reason="这周的数据还没核对完"
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="那我把草稿放到草稿箱，等你核对完再写。"),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("写周报")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(
        effects.get("written", []) == [],
        f"拒绝后副作用不该发生，实际 {effects.get('written')}",
    )
    ensure(
        bool(tool_messages) and ("未获批准" in tool_messages[-1].content or "被拒绝" in tool_messages[-1].content),
        "回填文案应当说清「未获批准/被拒绝」，实际 "
        + (repr(tool_messages[-1].content) if tool_messages else "没有回填消息"),
    )
    ensure(
        "换" in tool_messages[-1].content or "其他方式" in tool_messages[-1].content,
        f"回填文案应当暗示换条路，实际 {tool_messages[-1].content!r}",
    )
    ensure(
        controller.records[0].outcome == "rejected",
        f"台账应当记下一条拒绝，实际 {controller.records[0].outcome}",
    )
    print(f"  回填给模型：{tool_messages[-1].content}")
    print(f"  模型的下一步：{result.final_answer!r}")


# ---------------------------------------------------------------- 4. 改参数后放行


def check_edit_arguments() -> None:
    """人改了参数 → 执行用的是改后的参数，而且改了什么要进台账。"""
    section("改参数后放行：用改后的参数执行")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(
        outcome="approve",
        arguments={"title": "周报", "content": "人工润色后的正文"},
        reason="初稿有事实错误，我改过了",
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="周报已写入"),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("写周报")

    ensure(
        len(effects.get("written", [])) == 1,
        f"应当只写一次，实际 {len(effects.get('written', []))} 次",
    )
    ensure(
        effects["written"][0]["content"] == "人工润色后的正文",
        f"执行应当用改后的参数，实际写入 {effects['written'][0]}",
    )
    ensure(
        effects["written"][0]["content"] != "初稿",
        "原始参数不该被执行",
    )
    record = controller.records[0]
    ensure(record.edited is True, f"台账应当标记参数被改过，实际 edited={record.edited}")
    ensure(
        record.original_arguments["content"] == "初稿",
        f"台账应当保留模型原始参数，实际 {record.original_arguments}",
    )
    ensure(
        record.arguments["content"] == "人工润色后的正文",
        f"台账应当记录实际执行的参数，实际 {record.arguments}",
    )
    print(f"  模型要的：{record.original_arguments}")
    print(f"  实际跑的：{record.arguments}")


# ---------------------------------------------------------------- 5. 超时策略


def _write_scenario(
    policy: TimeoutPolicy,
    responder: Callable[[ApprovalRequest], ApprovalResponse | None],
    final_reply: str = "好的，这一步结束了。",
) -> tuple[dict[str, list[Any]], HitlController, HitlResult]:
    """跑一次"只写一 篇笔记"的场景，用来对比三种超时策略。"""
    effects: dict[str, list[Any]] = {}
    controller = HitlController(timeout_policy=policy)
    controller.responder = responder
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content=final_reply),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("写周报")
    return effects, controller, result


def check_timeout_policy() -> None:
    """人没响应怎么办：三种策略，默认必须是拒绝。"""
    section("超时策略：人不在的时候怎么办")

    no_reply: Callable[[ApprovalRequest], ApprovalResponse | None] = lambda request: None

    # 默认策略：拒绝
    effects, controller, result = _write_scenario("reject", no_reply)
    ensure(
        effects.get("written", []) == [],
        f"超时拒绝后副作用不该发生，实际 {effects.get('written')}",
    )
    ensure(
        bool(controller.records) and controller.records[0].outcome == "timeout",
        f"台账应当记下一条超时，实际 {[r.outcome for r in controller.records]}",
    )
    ensure(
        "超时" in controller.records[0].reason,
        f"超时记录应当写明原因，实际 {controller.records[0].reason!r}",
    )
    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(
        bool(tool_messages) and "未获批准" in tool_messages[-1].content,
        f"超时应按拒绝回填给模型，实际 {tool_messages[-1].content!r}" if tool_messages else "没有回填消息",
    )
    print(f"  默认（拒绝）：副作用 {effects.get('written', [])}，台账 {controller.records[0].outcome}")

    # 对照：自动放行 / 挂起等待
    allow_effects, allow_controller, allow_result = _write_scenario("allow", no_reply)
    hold_effects, hold_controller, hold_result = _write_scenario("hold", no_reply)

    ensure(
        len(allow_effects.get("written", [])) == 1,
        f"自动放行策略下副作用会真的发生，实际 {allow_effects.get('written')}",
    )
    ensure(
        hold_effects.get("written", []) == [],
        f"挂起策略下副作用不该发生，实际 {hold_effects.get('written')}",
    )
    ensure(
        hold_result.stop_reason == "awaiting_human",
        f"挂起策略下运行应当停在等人，实际 {hold_result.stop_reason}",
    )
    ensure(
        hold_result.checkpoint is not None,
        "挂起策略下必须落检查点，否则没法恢复",
    )
    ensure(
        allow_result.stop_reason == "completed",
        f"自动放行后应当正常跑完，实际 {allow_result.stop_reason}",
    )

    print("  策略    副作用        停止原因        台账结论")
    print(f"  reject  {effects.get('written', [])}  {result.stop_reason:<14}  {controller.records[0].outcome}")
    print(f"  allow   {len(allow_effects.get('written', []))} 次写入     {allow_result.stop_reason:<14}  {allow_controller.records[0].outcome}")
    print(f"  hold    {hold_effects.get('written', [])}  {hold_result.stop_reason:<14}  {hold_controller.records[0].outcome}")
    print("  → 默认必须是 reject：allow 等于橡皮章，hold 要求有人真的会回来")


# ---------------------------------------------------------------- 6. 人工接管


def check_takeover() -> None:
    """人接管：Agent 停止自动执行，把控制权交出去。"""
    section("接管：我不批，我自己做")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(
        outcome="takeover", reason="这条我自己删比较快", approver="张三"
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "删笔记", {"note_id": "note-42"})]),
            ModelReply(tool_calls=[ToolCall("c2", "写笔记", {"title": "记录", "content": "已删除"})]),
            ModelReply(content="已经处理完了。"),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("删掉 note-42")

    ensure(result.taken_over is True, f"应当标记为已接管，实际 taken_over={result.taken_over}")
    ensure(
        result.stop_reason == "taken_over",
        f"接管后运行应当停在交接点，实际 {result.stop_reason}",
    )
    ensure(
        effects.get("deleted", []) == [],
        f"接管后 Agent 不该自己执行删除，实际 {effects.get('deleted')}",
    )
    ensure(
        len(model.calls) == 1,
        f"接管后不该再问模型要下一步，实际问 了 {len(model.calls)} 次",
    )
    ensure(
        controller.records[0].outcome == "taken_over",
        f"台账应当记下接管，实际 {controller.records[0].outcome}",
    )
    print(f"  接管标记：{result.taken_over}，停止原因：{result.stop_reason}")
    print(f"  接管后模型被问了 {len(model.calls)} 次，Agent 侧副作用：{effects.get('deleted', [])}")


# ---------------------------------------------------------------- 7. 恢复


def check_resume() -> None:
    """挂起 → 恢复：从暂停点继续，已完成的动作不重做。"""
    section("恢复：从暂停点继续，不从头重跑")

    effects: dict[str, list[Any]] = {}
    controller = HitlController(timeout_policy="hold")
    controller.responder = lambda request: None  # 人一时没响应 → 挂起
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "读笔记", {"note_id": "n1"})]),
            ModelReply(tool_calls=[ToolCall("c2", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="周报已写入"),
        ]
    )
    runtime = HitlRuntime(model, build_tools(effects), controller)
    result = runtime.run("整理周报")

    ensure(
        result.stop_reason == "awaiting_human",
        f"应当停在等人审批的地方，实际 {result.stop_reason}",
    )
    ensure(result.checkpoint is not None, "挂起时必须落检查点")
    ensure(
        effects.get("read") == ["n1"],
        f"暂停前读操作应当已经完成，实际读了 {effects.get('read')}",
    )
    ensure(
        effects.get("written", []) == [],
        f"暂停前写操作不该执行，实际 {effects.get('written')}",
    )
    read_before = len(effects.get("read", []))
    executed_before = len(runtime.executed)
    print(f"  暂停时：读了 {read_before} 次，执行过 {executed_before} 次工具")

    # 人回来了：批准挂起的那条请求
    controller.responder = lambda request: ApprovalResponse(outcome="approve")
    resumed = runtime.resume()

    ensure(
        len(effects.get("read", [])) == read_before,
        f"恢复时不该重做已完成的读操作，实际读了 {len(effects.get('read', []))} 次",
    )
    ensure(
        len(runtime.executed) == executed_before + 1,
        f"恢复后只该补做写这一件事，实际执行了 {len(runtime.executed)} 次工具",
    )
    ensure(
        effects.get("written") == [{"title": "周报", "content": "初稿"}],
        f"恢复后写操作应当按原参数执行，实际 {effects.get('written')}",
    )
    ensure(
        resumed.final_answer == "周报已写入",
        f"恢复后应当接着走到最终回答，实际 {resumed.final_answer!r}",
    )
    ensure(
        resumed.stop_reason == "completed",
        f"恢复后应当正常结束，实际 {resumed.stop_reason}",
    )
    print(f"  恢复后：读了 {len(effects.get('read', []))} 次，执行了 {len(runtime.executed)} 次工具")
    print(f"  最终回答：{resumed.final_answer!r}")


# ---------------------------------------------------------------- 8. 审批粒度


def check_batch_approval() -> None:
    """一批同质操作 → 一次点头，而不是五次。"""
    section("审批粒度：五个同质动作一次批")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(outcome="approve")
    calls = [
        ToolCall(f"c{index}", "写笔记", {"title": f"笔记{index}", "content": "内容"})
        for index in range(5)
    ]
    model = scripted_model(
        [
            ModelReply(tool_calls=calls),
            ModelReply(content="五篇笔记都写好了。"),
        ]
    )
    result = HitlRuntime(model, build_tools(effects), controller).run("批量整理五篇笔记")

    ensure(
        len(controller.pending) == 1,
        f"五个同质操作应当只产生 1 条待审批，实际 {len(controller.pending)} 条",
    )
    request = controller.pending[0]
    ensure(
        len(getattr(request, "operations", [])) == 5,
        f"这一条审批应当覆盖 5 个操作，实际 {len(getattr(request, 'operations', []))} 个",
    )
    ensure(
        len(effects.get("written", [])) == 5,
        f"批准后五个动作都该执行，实际执行了 {len(effects.get('written', []))} 个",
    )
    ensure(
        len(controller.records) == 1,
        f"台账也只应当有一条，实际 {len(controller.records)} 条",
    )
    print(f"  模型发了 {len(calls)} 个调用 → 待审批 {len(controller.pending)} 条")
    print(f"  这一条覆盖了 {len(request.operations)} 个操作，一次点头全部放行")

    # 反例：粒度太细 = 5 次点头 = 审批疲劳
    fine_controller = HitlController()
    fine_controller.responder = lambda request: ApprovalResponse(outcome="approve")
    fine_model = scripted_model(
        [
            ModelReply(
                tool_calls=[
                    ToolCall(f"f{index}", "写笔记", {"title": f"笔记{index}", "content": "内容"})
                    for index in range(5)
                ]
            ),
            ModelReply(content="写完了。"),
        ]
    )
    HitlRuntime(
        fine_model,
        build_tools({}),
        fine_controller,
        batch_homogeneous=False,
    ).run("批量整理五篇笔记")

    ensure(
        len(fine_controller.pending) == 5,
        f"关掉合批后每个调用各问一次，应当是 5 条待审批，实际 {len(fine_controller.pending)} 条",
    )
    ensure(
        len(fine_controller.records) == 5,
        f"关掉合批后要打扰人 5 次，实际 {len(fine_controller.records)} 次",
    )
    print(f"  反例（关掉合批）：5 个动作 → 待审批 {len(fine_controller.pending)} 条，人要点 {len(fine_controller.records)} 次")


# ---------------------------------------------------------------- 9. 拒绝后不死循环


def _repeat_scenario(
    detect: bool,
) -> tuple[HitlController, HitlResult, dict[str, list[Any]], ScriptedModel]:
    """跑一次"被拒后又提同一个操作"的场景。"""
    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(outcome="reject", reason="不允许")
    # 脚本里连着三次提同一个操作——没有重复检测时，它会一直转到轮次上限
    same_call = {"title": "周报", "content": "初稿"}
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", dict(same_call))]),
            ModelReply(tool_calls=[ToolCall("c2", "写笔记", dict(same_call))]),
            ModelReply(tool_calls=[ToolCall("c3", "写笔记", dict(same_call))]),
        ]
    )
    runtime = HitlRuntime(model, build_tools(effects), controller, detect_repeat_rejection=detect)
    result = runtime.run("写周报")
    return controller, result, effects, model


def check_no_retry_loop() -> None:
    """被拒过的操作再提一次，运行时要认出来，不能无限重试。"""
    section("拒绝后不死循环：重复请求要被识别")

    controller, result, effects, model = _repeat_scenario(detect=True)

    ensure(
        result.stop_reason == "repeat_rejected",
        f"第二次请求同一个被拒操作时应当停下，实际 {result.stop_reason}",
    )
    ensure(
        len(model.calls) == 2,
        f"有重复检测时应当第二轮就停，实际问了模型 {len(model.calls)} 次",
    )
    ensure(
        len(effects.get("written", [])) == 0,
        f"被拒的操作不该真的执行，实际 {effects.get('written')}",
    )
    ensure(
        len(controller.records) == 1,
        f"同一个被拒操作不该反复打扰人，实际问了 {len(controller.records)} 次",
    )

    # 反例：没有重复检测时，运行会一直转下去
    no_guard_controller, no_guard_result, no_guard_effects, no_guard_model = _repeat_scenario(detect=False)
    ensure(
        no_guard_result.stop_reason == "max_iterations",
        f"没有重复检测时应当一路转到轮次上限，实际 {no_guard_result.stop_reason}",
    )
    ensure(
        len(no_guard_model.calls) > len(model.calls),
        f"没有重复检测时模型被问的次数应当更多（说明它一直在重试），"
        f"实际 {len(no_guard_model.calls)} vs {len(model.calls)}",
    )
    ensure(
        len(no_guard_controller.records) == len(no_guard_model.calls),
        f"没有重复检测时人要被问 {len(no_guard_model.calls)} 次，实际 {len(no_guard_controller.records)} 次",
    )
    ensure(
        no_guard_effects.get("written", []) == [],
        f"反例里被拒的操作同样不该执行，实际 {no_guard_effects.get('written')}",
    )
    print(f"  有重复检测：{result.stop_reason}，问模型 {len(model.calls)} 次，打扰人 {len(controller.records)} 次")
    print(
        f"  没有检测  ：{no_guard_result.stop_reason}，问模型 {len(no_guard_model.calls)} 次，"
        f"打扰人 {len(no_guard_controller.records)} 次"
    )


# ---------------------------------------------------------------- 10. 审计


def check_audit_trail() -> None:
    """审批记录要可审计：谁批的、批了什么、什么时候。"""
    section("审批记录可审计")

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(
        outcome="approve",
        arguments={"title": "周报", "content": "人工润色后的正文"},
        reason="改了正文",
        approver="张三",
    )
    model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "写笔记", {"title": "周报", "content": "初稿"})]),
            ModelReply(content="周报已写入"),
        ]
    )
    HitlRuntime(model, build_tools(effects), controller).run("写周报")

    ensure(len(controller.records) == 1, f"应当有一条审批记录，实际 {len(controller.records)} 条")
    record = controller.records[0]
    ensure(record.approver == "张三", f"记录应当写明谁批的，实际 {record.approver!r}")
    ensure(record.tool_name == "写笔记", f"记录应当写明批了什么，实际 {record.tool_name!r}")
    ensure(record.at > 0, "记录应当带时间戳")
    ensure(record.request_id.startswith("ap-"), f"记录应当带审批单号，实际 {record.request_id!r}")
    ensure(record.edited is True, "记录应当标出参数被改过")
    print(f"  单号：{record.request_id}")
    print(f"  谁批：{record.approver} | 批了什么：{record.tool_name} | 什么时候：{record.at:.0f}")
    print(f"  原始参数：{record.original_arguments}")
    print(f"  实际参数：{record.arguments}")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：真模型面对写操作会怎么走，只校验形状。"""
    banner("第 14 章 · 真实模型路径")
    from ..labkit import real_model

    effects: dict[str, list[Any]] = {}
    controller = HitlController()
    controller.responder = lambda request: ApprovalResponse(outcome="approve")
    agent = HitlRuntime(real_model(), build_tools(effects), controller, max_iterations=4)
    result = agent.run("读一下 n1，然后写一篇标题为《周报》的笔记")

    for message in result.messages:
        print(f"  {message.role}: {message.content[:120]!r}")

    ensure(
        all(m.role != "tool" or m.tool_call_id for m in result.messages),
        "工具消息必须带 tool_call_id",
    )
    ensure(
        controller.needs_approval(build_tools({})[1]),
        "写操作在真实路径上也必须触发审批",
    )
    print(f"  停止原因：{result.stop_reason}，审批 {len(controller.records)} 条")


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 14 章实验。"""
    banner("第 14 章 · 人在环路：审批、接管与恢复")
    check_trigger_by_risk()
    check_approve_executes()
    check_reject_blocks()
    check_edit_arguments()
    check_timeout_policy()
    check_takeover()
    check_resume()
    check_batch_approval()
    check_no_retry_loop()
    check_audit_trail()
    rule("审批触发看动作风险；默认拒绝而不是自动放行；恢复要接着跑而不是重跑")
    print("\n✓ 第 14 章通过")
