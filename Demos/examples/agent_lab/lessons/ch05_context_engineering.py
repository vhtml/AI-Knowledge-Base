"""第 5 章实验：上下文工程——窗口预算与压缩。

这个模块解决什么问题：第 4 章把系统提示写好了，第 6 章要谈长期记忆，
但中间这一层还悬着——**模型这一轮到底看到了什么、看到了多少**。
窗口塞不下时该砍谁、该保谁，光靠讲是讲不清楚的，必须能跑出来看。

所以本章把上下文工程变成八条可执行的断言：

1. **预算表自洽**——上下文由六部分构成，每部分定配额，配额之和等于总窗口；
   装配后的实测大小逐项不超配额。
2. **工具结果截断**——超长结果被裁剪，并带显式标记，让模型知道"这不是全文"。
3. **摘要压缩**——老消息折叠成一条摘要，条数下降，但结论必须还在摘要里。
4. **错误永不被压**——工具的错误消息、当前任务约束、模型刚做出的决策依据，谁都不许动。
5. **上下文腐化**——长对话里中段信息最先被丢掉（lost in the middle）。
6. **token 下降可度量**——压缩前后用 ``approx_tokens`` 估大小，比例可断言。
7. **短任务不压缩**——对话够短时压缩策略是恒等变换（幂等）。
8. **上下文该放什么**——寒暄与重复被剔除，事实与状态被保留。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..labkit import (
    Message,
    ModelReply,
    Tool,
    ToolCall,
    ai,
    approx_tokens,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    system,
    tool,
    tool_result,
)
from ..runtime import Agent, AgentConfig

# ---------------------------------------------------------------- 常量

SUMMARY_HEADER = "【历史摘要】以下是已被折叠的更早对话，只保留结论与约束："
"""摘要消息的头。带这个头，模型才知道这段文字是"压缩产物"，不是新指令。"""

TRUNCATION_SUFFIX = "…（已截断，原始长度 {n} 字符）"
"""工具结果截断标记。缺了它，模型会把残缺结果当成完整事实。"""

KEY_MARKERS = ("结论", "决定", "约束", "注意", "错误", "失败")
"""抽取式摘要会额外保留含这些词的句子——压缩要丢过程，不是丢结论。"""

ERROR_MARKERS = ("错误", "失败", "Error", "error")
"""工具结果开头出现这些词，就判定为错误消息。"""

ROLE_LABELS = {
    "system": "系统",
    "user": "用户",
    "assistant": "助手",
    "tool": "工具",
}

GREETINGS = {"你好", "您好", "在吗", "hi", "hello", "谢谢", "好的", "收到", "ok", "嗯嗯"}
"""纯寒暄：占预算但不携带状态。"""

SYSTEM_PROMPT = (
    "你是一个订单服务的运维助手。回答前先确认约束："
    "本轮只做只读操作；任何会修改数据的动作都要先说明影响面。"
)

RETRIEVAL = "《运维手册》第 3 章：连接池打满的典型表现是 P99 延迟上升、活跃连接数触顶。"


# ---------------------------------------------------------------- 工具（用来量 schema 体积）


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return {"A-1001": "已发货", "A-1002": "待付款"}.get(order_id, "未找到")


@tool
def 检索运维手册(keyword: str, top_k: int = 3) -> str:
    """在运维手册里检索相关章节。

    Args:
        keyword: 检索关键词
        top_k: 返回的片段条数
    """
    return f"《运维手册》关于 {keyword} 的片段 × {top_k}"


@tool
def 查询订单明细(order_id: str) -> str:
    """返回订单的完整明细行——刻意返回得很长，用来演示截断。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return "明细行：" + "A-1001,已发货,3 件,顺丰,SF123456," * 500


# ---------------------------------------------------------------- 1. 预算表


@dataclass(frozen=True)
class Budget:
    """上下文窗口的预算表。

    每个字段对应上下文中的一部分，配额之和必须等于总窗口——
    这不是形式主义：预算表不闭合，就意味着某一部分在偷偷吃别人的份额，
    超限时你会去砍错东西（通常是去砍历史，而不是去砍那个撑爆的工具结果）。
    """

    total_tokens: int = 128_000
    system_tokens: int = 2_000
    tool_schema_tokens: int = 6_000
    history_tokens: int = 40_000
    tool_result_tokens: int = 30_000
    retrieval_tokens: int = 12_000
    output_tokens: int = 4_000
    reserve_tokens: int = 34_000
    """余量：给 tokenizer 误差、突发输出和重试留的空间。"""

    @property
    def allocated_tokens(self) -> int:
        """所有配额之和（含余量）。"""
        return (
            self.system_tokens
            + self.tool_schema_tokens
            + self.history_tokens
            + self.tool_result_tokens
            + self.retrieval_tokens
            + self.output_tokens
            + self.reserve_tokens
        )

    def validate(self) -> None:
        """检查预算表是否自洽，失败信息写清"期望什么、实际什么"。"""
        ensure(
            self.allocated_tokens == self.total_tokens,
            f"期望预算表闭合：各部分配额之和应等于总窗口 {self.total_tokens}，"
            f"实际等于 {self.allocated_tokens}，差 {self.allocated_tokens - self.total_tokens}",
        )
        ensure(
            self.reserve_tokens >= self.total_tokens // 10,
            f"期望余量至少占总窗口的 10%（{self.total_tokens // 10} token），"
            f"实际余量只有 {self.reserve_tokens} token——突发输出会直接挤掉历史",
        )
        ensure(
            self.output_tokens > 0,
            "期望输出预留大于 0，实际为 0——模型就没有地方写答案",
        )
        ensure(
            min(
                self.system_tokens,
                self.tool_schema_tokens,
                self.history_tokens,
                self.tool_result_tokens,
                self.retrieval_tokens,
            )
            > 0,
            "期望每一部分的配额都大于 0",
        )

    def rows(self) -> list[tuple[str, int]]:
        """按"上下文各部分"的顺序列出配额，方便打印。"""
        return [
            ("系统提示", self.system_tokens),
            ("工具 schema", self.tool_schema_tokens),
            ("历史消息", self.history_tokens),
            ("工具结果", self.tool_result_tokens),
            ("检索内容", self.retrieval_tokens),
            ("输出预留", self.output_tokens),
            ("余量", self.reserve_tokens),
        ]


DEFAULT_BUDGET = Budget()
"""128k 窗口的标准预算表，和笔记里的那张表一一对应。"""

SMALL_BUDGET = Budget(
    total_tokens=4_000,
    system_tokens=200,
    tool_schema_tokens=400,
    history_tokens=1_400,
    tool_result_tokens=800,
    retrieval_tokens=200,
    output_tokens=400,
    reserve_tokens=600,
)
"""教学用的小预算表：把 128k 缩到 4k，"超配额"才会真的发生。"""

QUOTA_KEYS = {
    "系统提示": "system_tokens",
    "工具 schema": "tool_schema_tokens",
    "历史消息": "history_tokens",
    "工具结果": "tool_result_tokens",
    "检索内容": "retrieval_tokens",
}


@dataclass
class ContextParts:
    """一次上下文装配的各部分——预算表的"被填对象"。"""

    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    retrieval: str = ""

    def measure(self) -> dict[str, int]:
        """逐部分估算 token 数。

        工具结果单独算一份配额：它们增长得最快，也最容易被忽略。
        """
        schema_text = " ".join(
            json.dumps(item.to_api(), ensure_ascii=False) for item in self.tools
        )
        return {
            "系统提示": approx_tokens(self.system_prompt),
            "工具 schema": approx_tokens(schema_text),
            "历史消息": sum(
                approx_tokens(m.content) for m in self.history if m.role != "tool"
            ),
            "工具结果": sum(
                approx_tokens(m.content) for m in self.history if m.role == "tool"
            ),
            "检索内容": approx_tokens(self.retrieval),
        }

    def total_tokens(self) -> int:
        """装配后各部分的总 token 估计。"""
        return sum(self.measure().values())


def tool_output_limit(budget: Budget) -> int:
    """工具结果的截断上限——从预算表推导，而不是拍脑袋定一个魔法数字。

    留出 24 token 给截断标记本身，否则"裁完还是超配额"。
    """
    return max(50, budget.tool_result_tokens - 24)


def check_budget_table() -> None:
    """预算表必须闭合，且装配后的实测大小逐项不超配额。"""
    section("预算表：配额之和 == 总窗口，且逐项不超")

    budget = DEFAULT_BUDGET
    budget.validate()
    print(f"  默认预算（{budget.total_tokens} token 窗口）：")
    for label, quota in budget.rows():
        print(f"    {label:<12} {quota:>7} token")
    print(f"    {'合计':<12} {budget.allocated_tokens:>7} token（== 总窗口）")

    ensure(
        budget.allocated_tokens == budget.total_tokens,
        f"期望预算表闭合，实际合计 {budget.allocated_tokens} != {budget.total_tokens}",
    )

    # 用小预算表做真正的"装配"：不截断的话，工具结果一项就能把配额撑爆
    small = SMALL_BUDGET
    small.validate()
    limit = tool_output_limit(small)
    huge = "日志行：连接池 active=200 idle=0 waiters=37\n" * 400

    raw_tokens = approx_tokens(huge)
    ensure(
        raw_tokens > small.tool_result_tokens * 5,
        f"期望未截断的工具结果远超配额 {small.tool_result_tokens} token，"
        f"实际只有 {raw_tokens} token——样例数据不够狠，演示不出问题",
    )

    truncated, was_truncated = truncate_tool_output(huge, limit)
    parts = ContextParts(
        system_prompt=SYSTEM_PROMPT,
        tools=[查询订单, 检索运维手册],
        history=[
            human("任务：定位订单服务的线上延迟。"),
            ai("我先看监控面板。"),
            tool_result(truncated, "call_1", "查询日志"),
        ],
        retrieval=RETRIEVAL,
    )
    measured = parts.measure()

    for label, tokens in measured.items():
        quota = getattr(small, QUOTA_KEYS[label])
        ensure(
            tokens <= quota,
            f"期望「{label}」不超过配额 {quota} token，实际 {tokens} token",
        )
        print(f"    {label:<12} 实测 {tokens:>5} / 配额 {quota:>5} token  ✓")

    ensure(
        parts.total_tokens() <= small.total_tokens - small.reserve_tokens,
        f"期望装配结果不超过「总窗口 − 余量」"
        f"{small.total_tokens - small.reserve_tokens} token，实际 {parts.total_tokens()}",
    )
    print(
        f"  未截断时工具结果要 {raw_tokens} token（配额的 {raw_tokens / small.tool_result_tokens:.0f} 倍）；"
        f"截断后 {measured['工具结果']} token"
    )


# ---------------------------------------------------------------- 2. 工具结果截断


def truncate_tool_output(content: str, max_chars: int) -> tuple[str, bool]:
    """把超长工具结果裁到上限，并显式标记"这不是全文"。

    为什么一定要带标记：模型看不到被裁掉的部分。如果不告诉它，
    它会把残缺的结果当成完整事实去推理——这比直接报错更危险，
    因为模型不会怀疑自己的输入，只会怀疑结论。
    """
    if len(content) <= max_chars:
        return content, False
    kept = content[:max_chars]
    return kept + "\n" + TRUNCATION_SUFFIX.format(n=len(content)), True


def ingest_tool_result(
    content: str,
    tool_call_id: str,
    name: str,
    max_chars: int,
) -> Message:
    """工具结果进上下文的第一步：截断并打标记。

    为什么在"进上下文"这一步就截：等它进了历史再补救，
    那一大坨已经占过预算了，而且它往往是最容易挤掉别人的那一份。
    """
    text, _ = truncate_tool_output(content, max_chars)
    return tool_result(text, tool_call_id=tool_call_id, name=name)


def check_tool_output_truncation() -> None:
    """超长工具结果被裁剪，且带截断标记。"""
    section("工具结果截断：模型必须知道「这不是全文」")

    budget = SMALL_BUDGET
    limit = tool_output_limit(budget)
    raw = "订单明细：" + "A-1001,已发货,3 件," * 500

    message = ingest_tool_result(raw, "call_1", "查询订单", limit)
    marker = TRUNCATION_SUFFIX.format(n=len(raw))

    ensure(
        marker in message.content,
        f"期望截断后的工具结果里带截断标记 {marker!r}，实际内容里没有",
    )
    ensure(
        len(message.content) < len(raw),
        f"期望工具结果被裁短，原始 {len(raw)} 字符，实际 {len(message.content)} 字符",
    )
    ensure(
        approx_tokens(message.content) <= budget.tool_result_tokens,
        f"期望截断后不超过配额 {budget.tool_result_tokens} token，"
        f"实际 {approx_tokens(message.content)} token",
    )
    print(f"  原始 {len(raw)} 字符 → 裁后 {len(message.content)} 字符")
    print(f"  标记：{message.content.splitlines()[-1]!r}")

    # 短结果不该被乱标——否则模型会满屏"已截断"而不知道发生了什么
    short = ingest_tool_result("已发货", "call_2", "查询订单", limit)
    ensure(
        "已截断" not in short.content,
        f"期望短结果保持原样，实际 {short.content!r}",
    )

    # 第 2 章的运行时里已经有一份等价实现，这里验证两者口径一致
    runtime_agent = Agent(
        scripted_model(
            [
                ModelReply(
                    tool_calls=[
                        ToolCall("call_1", "查询订单明细", {"order_id": "A-1001"})
                    ]
                ),
                ModelReply(content="订单明细很长，我只看了前几行。"),
            ]
        ),
        [查询订单明细],
        AgentConfig(max_tool_output_chars=limit),
    )
    runtime_result = runtime_agent.run("查 A-1001 明细")
    runtime_tool_messages = [m for m in runtime_result.messages if m.role == "tool"]
    ensure(
        len(runtime_tool_messages) == 1,
        f"期望运行时回填 1 条工具消息，实际 {len(runtime_tool_messages)}",
    )
    ensure(
        "已截断" in runtime_tool_messages[0].content,
        "期望运行时也打上截断标记（AgentConfig.max_tool_output_chars），"
        f"实际内容：{runtime_tool_messages[0].content[-60:]!r}",
    )
    print("  第 2 章运行时的 max_tool_output_chars 用的是同一套口径")


# ---------------------------------------------------------------- 3. 摘要压缩


@dataclass(frozen=True)
class CompactionPolicy:
    """压缩策略的旋钮。

    每个旋钮都对应一个真实决策：保多少开头、保多少结尾、
    工具结果截多长、历史多大才值得动手。
    """

    keep_head: int = 1
    keep_recent: int = 4
    history_token_budget: int = 4_000
    max_tool_output_chars: int = 2_000
    max_summary_chars: int = 400


@dataclass
class CompactionResult:
    """一次压缩的结果。"""

    messages: list[Message]
    summary: Message | None = None
    compacted_count: int = 0
    protected_count: int = 0


def _history_tokens(messages: Sequence[Message]) -> int:
    """一段历史的 token 估计。"""
    return sum(approx_tokens(m.content) for m in messages)


def _looks_like_error(content: str) -> bool:
    """工具结果是不是一条错误消息。

    只看开头 40 个字符：结果正文里出现"错误"两个字很常见
    （比如某个字段名叫 error_code），不能一律当成失败。
    """
    head = content[:40]
    return any(marker in head for marker in ERROR_MARKERS)


def is_protected(message: Message) -> bool:
    """一条消息是不是"绝对不能压"。

    两类：系统提示（当前任务的关键约束），以及工具的错误消息。
    还有一类要看整段历史才能判定——模型刚做出的决策依据，见
    :func:`_latest_decision`。
    """
    if message.role == "system":
        return True
    if message.role == "tool":
        return _looks_like_error(message.content)
    return False


def _latest_decision(messages: Sequence[Message]) -> Message | None:
    """最近一条带工具调用的助手消息——模型"刚做出的决策依据"。

    它记录了模型为什么选了这条路、传了什么参数。
    压掉它，模型下一轮就不知道自己为什么在这儿了。
    """
    for message in reversed(messages):
        if message.role == "assistant" and message.tool_calls:
            return message
    return None


def extractive_summary(
    messages: Sequence[Message],
    previous_summary: Message | None = None,
) -> str:
    """把一批旧消息压成一段抽取式摘要。

    真实系统里这一步通常调模型写摘要；这里用确定性的抽取式代替，
    保证离线可复现。抽取规则：每条消息保留首句，外加含
    "结论/决定/约束"的句子——压缩要丢的是过程，不是结论。
    """
    lines: list[str] = []
    seen: set[str] = set()

    if previous_summary is not None:
        previous = previous_summary.content.strip()
        if previous:
            lines.append(previous)
            seen.add(previous)

    for message in messages:
        text = message.content.strip()
        if not text:
            continue
        raw_lines = text.splitlines()
        candidates = [raw_lines[0]] + [
            line for line in raw_lines[1:] if any(m in line for m in KEY_MARKERS)
        ]
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            lines.append(f"[{ROLE_LABELS.get(message.role, message.role)}] {candidate}")

    return "\n".join(lines)


def _clamp(text: str, max_chars: int) -> str:
    """把摘要截到上限，并保留"这是摘要"的可识别性。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…（摘要已截断）"


def compact(
    messages: Sequence[Message],
    policy: CompactionPolicy,
    previous_summary: Message | None = None,
) -> CompactionResult:
    """把一段历史压进预算：老消息折叠成摘要，尾部保留原文。

    三步：

    1. 历史本来就装得下 → 什么都不做（短任务不需要压缩）；
    2. 划出"保开头 / 保结尾 / 可折叠中段"三块；
    3. 中段里受保护的消息（错误、系统约束、最近一次决策）单独留下，
       其余折叠进摘要。
    """
    history = list(messages)
    if not history:
        return CompactionResult(messages=[])

    if _history_tokens(history) <= policy.history_token_budget:
        return CompactionResult(messages=history)

    keep_head = max(0, policy.keep_head)
    keep_recent = max(0, policy.keep_recent)
    if len(history) <= keep_head + keep_recent:
        return CompactionResult(messages=history)

    head = history[:keep_head]
    tail = history[len(history) - keep_recent :]
    middle = history[keep_head : len(history) - keep_recent]

    middle_ids = {id(m) for m in middle}
    protected_ids = {id(m) for m in middle if is_protected(m)}
    decision = _latest_decision(history)
    if decision is not None and id(decision) in middle_ids:
        protected_ids.add(id(decision))

    pinned = [m for m in middle if id(m) in protected_ids]
    foldable = [m for m in middle if id(m) not in protected_ids]

    if not foldable:
        return CompactionResult(messages=history, protected_count=len(pinned))

    summary_text = _clamp(
        extractive_summary(foldable, previous_summary), policy.max_summary_chars
    )
    summary_message = Message(role="system", content=SUMMARY_HEADER + summary_text)

    new_messages = list(head) + [summary_message] + pinned + list(tail)
    return CompactionResult(
        messages=new_messages,
        summary=summary_message,
        compacted_count=len(foldable),
        protected_count=len(pinned),
    )


def build_work_history() -> list[Message]:
    """构造一段真实形态的历史：结论在中间，错误在中间，末尾是当前提问。"""
    history: list[Message] = [
        human("任务：排查订单服务的线上延迟。"),
        ai("我先看监控面板。\n结论：延迟来自数据库连接池打满，采用方案 B。"),
    ]
    for index in range(10):
        history.append(human(f"补充：第 {index} 轮排查记录。"))
        history.append(ai(f"排查记录 {index}：日志正常，继续。"))
    history.append(human("把结论发我。"))
    history.append(ai("结论稍后整理给你。"))
    return history


def check_summary_compaction() -> None:
    """老消息被压成一条摘要，条数下降，但结论还在。"""
    section("摘要压缩：条数下降，结论必须留下")

    history = build_work_history()
    policy = CompactionPolicy(keep_head=1, keep_recent=4, history_token_budget=200)
    result = compact(history, policy)

    ensure(
        result.summary is not None,
        "期望压缩后产出一条摘要消息，实际没有摘要",
    )
    ensure(
        len(result.messages) < len(history),
        f"期望压缩后历史条数少于原来的 {len(history)}，实际 {len(result.messages)}",
    )
    ensure(
        "延迟来自数据库连接池打满" in result.summary.content,
        "期望摘要里保留结论「延迟来自数据库连接池打满」，实际摘要内容："
        f"{result.summary.content!r}",
    )
    ensure(
        sum(1 for m in result.messages if m.content.startswith(SUMMARY_HEADER)) == 1,
        "期望被折叠的老消息合成且仅合成一条摘要",
    )
    ensure(
        any(m.content == "把结论发我。" for m in result.messages),
        "期望最近的提问被原样保留",
    )
    print(f"  历史 {len(history)} 条 → {len(result.messages)} 条"
          f"（折叠 {result.compacted_count} 条，保护 {result.protected_count} 条）")
    print(f"  摘要开头：{result.summary.content.splitlines()[1][:60]!r}")


# ---------------------------------------------------------------- 4. 绝对不能压的东西


def check_protected_messages() -> None:
    """工具错误、任务约束、刚做出的决策依据，谁都不许动。"""
    section("什么绝对不能压：错误 / 约束 / 决策依据")

    constraint = system("约束：本轮只做只读操作，不得修改数据库。")
    decision = ai(
        "先查订单 A-1001 确认状态，再看延迟。",
        tool_calls=[ToolCall("call_1", "查询订单", {"order_id": "A-1001"})],
    )
    error = tool_result("参数错误：缺少 order_id 参数。", "call_0", "查询订单")

    history = [
        human("开始排查订单服务延迟。"),
        constraint,
        ai("我看了一下监控。\n结论：延迟来自数据库连接池打满。"),
        decision,
        error,
        human("继续排查。"),
    ]
    policy = CompactionPolicy(keep_head=1, keep_recent=1, history_token_budget=50)
    result = compact(history, policy)

    ensure(
        any(m.role == "tool" and "参数错误" in m.content for m in result.messages),
        "期望工具错误消息原样保留，实际结果里没有它",
    )
    ensure(
        any(m.role == "system" and "只读操作" in m.content for m in result.messages),
        "期望当前任务的关键约束原样保留，实际结果里没有它",
    )
    ensure(
        any(m.tool_calls and m.tool_calls[0].name == "查询订单" for m in result.messages),
        "期望模型刚做出的决策依据（带 tool_calls 的助手消息）原样保留，实际没有",
    )
    ensure(
        result.protected_count == 3,
        f"期望识别出 3 条受保护消息，实际 {result.protected_count}",
    )
    ensure(
        result.summary is not None
        and "延迟来自数据库连接池打满" in result.summary.content,
        "期望被折叠的那条普通消息以结论形式留在摘要里",
    )
    print(f"  受保护消息 {result.protected_count} 条：错误 / 约束 / 决策依据")
    print("  → 压掉它们，模型就不知道自己为什么在干这件事")


# ---------------------------------------------------------------- 5. 上下文腐化


def build_long_conversation(rounds: int) -> list[Message]:
    """构造一段长对话：每条消息带唯一编号，便于事后看谁被丢了。"""
    messages: list[Message] = []
    for index in range(rounds):
        body = f"事实{index:02d}：这是第 {index} 条记录，内容本身无关紧要。"
        messages.append(human(body) if index % 2 == 0 else ai(body))
    return messages


def keep_both_ends(messages: Sequence[Message], head: int, tail: int) -> list[Message]:
    """位置策略：保住两头，牺牲中段。"""
    if len(messages) <= head + tail:
        return list(messages)
    return list(messages[:head]) + list(messages[len(messages) - tail :])


def keep_recent_only(messages: Sequence[Message], tail: int) -> list[Message]:
    """近因策略：只保最近——最省，但开头也会丢。"""
    if len(messages) <= tail:
        return list(messages)
    return list(messages[len(messages) - tail :])


def survival_map(original: Sequence[Message], kept: Sequence[Message]) -> str:
    """一张"存活地图"：每个原始位置标 ✓ 或 ✗。"""
    kept_ids = {id(m) for m in kept}
    return "".join("✓" if id(m) in kept_ids else "✗" for m in original)


def check_context_corruption() -> None:
    """长对话里中段信息最先被丢掉——这就是 lost in the middle。"""
    section("上下文腐化：中段信息最先被忘掉")

    original = build_long_conversation(30)
    both_ends = keep_both_ends(original, head=6, tail=10)
    recent_only = keep_recent_only(original, tail=10)

    ensure(
        id(original[0]) in {id(m) for m in both_ends},
        "期望保两头策略留下开头的消息，实际开头被丢了",
    )
    ensure(
        id(original[15]) not in {id(m) for m in both_ends},
        "期望保两头策略丢掉中段的第 15 条，实际它还在",
    )
    ensure(
        id(original[29]) in {id(m) for m in both_ends},
        "期望保两头策略留下结尾的消息，实际结尾被丢了",
    )
    ensure(
        id(original[0]) not in {id(m) for m in recent_only},
        "期望近因策略把开头也丢掉（这正是它更省、也更危险的代价），实际开头还在",
    )

    print(f"  30 条消息，按位置保留两端（头 6 / 尾 10）：")
    print(f"    {survival_map(original, both_ends)}")
    print(f"  只保最近 10 条：")
    print(f"    {survival_map(original, recent_only)}")
    print("  → 中段最先消失，这就是 lost in the middle")

    # 对策：把关键事实前置成系统消息——它受保护，任何策略都动不了
    key_fact = "关键事实：延迟来自数据库连接池打满。"
    guarded = [system(key_fact), *build_long_conversation(30)]
    policy = CompactionPolicy(keep_head=1, keep_recent=6, history_token_budget=100)
    compacted = compact(guarded, policy)
    ensure(
        any(key_fact in m.content for m in compacted.messages),
        "期望关键事实前置成系统消息后被保留，实际被压掉了",
    )
    print("  对策：关键事实前置成系统消息 → 压缩后仍在")


# ---------------------------------------------------------------- 6. token 下降


def check_token_reduction() -> None:
    """压缩前后用 approx_tokens 估大小，下降比例可断言。"""
    section("token 预算：压缩前后的大小要真的降下来")

    budget = SMALL_BUDGET
    limit = tool_output_limit(budget)
    huge = "日志行：连接池 active=200 idle=0 waiters=37\n" * 400
    raw_tokens = approx_tokens(huge)

    message = ingest_tool_result(huge, "call_1", "查询日志", limit)
    history = [
        human("任务：定位线上延迟。"),
        ai("我先看监控面板。\n结论：延迟来自数据库连接池打满，采用方案 B。"),
        message,
    ]
    history.extend(ai(f"排查记录 {index}：日志正常，继续。") for index in range(10))
    history.append(human("把结论发我。"))

    policy = CompactionPolicy(
        keep_head=1, keep_recent=1, history_token_budget=200, max_summary_chars=300
    )
    result = compact(history, policy)

    before = _history_tokens(history)
    after = _history_tokens(result.messages)
    ratio = after / before if before else 1.0

    ensure(
        after < before,
        f"期望压缩后上下文变小，压缩前 {before} token，压缩后 {after} token",
    )
    ensure(
        ratio <= 0.8,
        f"期望压缩后至少降到原来的 80%，实际比例 {ratio:.2f}"
        f"（{before} → {after} token）",
    )
    ensure(
        result.summary is not None
        and "延迟来自数据库连接池打满" in result.summary.content,
        "期望压缩后结论仍在摘要里",
    )
    ensure(
        approx_tokens(message.content) <= budget.tool_result_tokens,
        f"期望截断后的工具结果不超过配额 {budget.tool_result_tokens} token",
    )

    print(f"  未截断的工具结果：{raw_tokens} token"
          f"（配额的 {raw_tokens / budget.tool_result_tokens:.0f} 倍）")
    print(f"  压缩前 {before} token → 压缩后 {after} token（降到 {ratio:.0%}）")
    print("  → 两道闸：进上下文时截断，跑长了才压缩")


# ---------------------------------------------------------------- 7. 短任务不该压


def check_short_task_no_compression() -> None:
    """对话够短时，压缩策略是恒等变换。"""
    section("短任务不需要压缩：策略是恒等变换")

    short = [
        human("查一下 A-1001。"),
        ai("已发货。"),
        human("再查 A-1002。"),
    ]
    policy = CompactionPolicy(history_token_budget=4_000)

    once = compact(short, policy)
    ensure(
        once.messages == short,
        f"期望短对话原样返回，实际返回了 {len(once.messages)} 条不同的消息",
    )
    ensure(
        once.summary is None,
        "期望短对话不产生摘要，实际产生了一条",
    )
    ensure(
        once.compacted_count == 0,
        f"期望短对话折叠 0 条消息，实际折叠了 {once.compacted_count} 条",
    )

    twice = compact(once.messages, policy)
    ensure(
        twice.messages == once.messages,
        "期望压缩是幂等的：压两次和压一次结果相同",
    )
    ensure(
        twice.summary == once.summary,
        "期望压缩是幂等的：两次产生的摘要相同",
    )
    print(f"  短对话 {len(short)} 条 → 原样返回，压缩 0 条，幂等 ✓")

    # 长对话压完之后再压一次，不应该越压越大
    long_history = build_work_history()
    policy_long = CompactionPolicy(
        keep_head=1, keep_recent=4, history_token_budget=200, max_summary_chars=300
    )
    first = compact(long_history, policy_long)
    second = compact(first.messages, policy_long)
    ensure(
        _history_tokens(second.messages) <= _history_tokens(first.messages),
        "期望重复压缩不会让上下文变大，"
        f"第一次 {_history_tokens(first.messages)} token，"
        f"第二次 {_history_tokens(second.messages)} token",
    )
    print(
        f"  长对话压一次 {len(first.messages)} 条，再压一次 {len(second.messages)} 条（收敛）"
    )


# ---------------------------------------------------------------- 8. 上下文该放什么


def strip_noise(messages: Sequence[Message]) -> list[Message]:
    """去掉寒暄和完全重复的消息。

    寒暄不携带状态，重复内容是纯浪费——两者都在吃窗口预算。
    上下文里应该放事实与状态，不是礼貌用语。
    """
    kept: list[Message] = []
    seen: set[str] = set()
    for message in messages:
        text = message.content.strip().lower()
        if text in GREETINGS:
            continue
        if message.role != "tool" and text in seen:
            continue
        seen.add(text)
        kept.append(message)
    return kept


def check_what_belongs_in_context() -> None:
    """寒暄与重复被剔除，事实与状态被保留。"""
    section("上下文该放什么：事实与状态，不是寒暄与重复")

    noisy = [
        human("你好"),
        human("查一下 A-1001。"),
        ai("已发货。"),
        ai("已发货。"),
        human("谢谢"),
        tool_result("状态：已发货", "call_1", "查询订单"),
        human("再查 A-1002。"),
    ]
    cleaned = strip_noise(noisy)

    ensure(
        not any(m.content.strip().lower() in GREETINGS for m in cleaned),
        "期望寒暄被剔除，实际还在",
    )
    ensure(
        sum(1 for m in cleaned if m.content.strip() == "已发货。") == 1,
        "期望完全重复的消息只保留一条，实际保留多条",
    )
    ensure(
        any(m.role == "tool" and m.content == "状态：已发货" for m in cleaned),
        "期望工具结果（状态）被保留，实际被误删",
    )
    ensure(
        any(m.content == "再查 A-1002。" for m in cleaned),
        "期望任务事实被保留，实际被误删",
    )
    ensure(
        len(cleaned) < len(noisy),
        f"期望清理后条数少于 {len(noisy)}，实际 {len(cleaned)}",
    )
    print(f"  {len(noisy)} 条 → {len(cleaned)} 条（省下的是寒暄和重复，不是事实）")


# ---------------------------------------------------------------- 9. 端到端


def check_end_to_end_with_runtime() -> None:
    """把压缩接到第 2 章的运行时上，看模型这一轮到底收到了什么。"""
    section("端到端：模型这一轮实际看到了什么")

    history = build_work_history()
    policy = CompactionPolicy(keep_head=1, keep_recent=4, history_token_budget=200)
    compacted = compact(history, policy)

    model = scripted_model(["结论：延迟来自数据库连接池打满，采用方案 B。"])
    agent = Agent(model, [查询订单], AgentConfig(system_prompt=SYSTEM_PROMPT))
    agent.run("把结论发我。", history=compacted.messages)

    seen = model.calls[0]
    seen_text = "\n".join(m.content for m in seen)

    ensure(
        "延迟来自数据库连接池打满" in seen_text,
        "期望模型收到压缩后的摘要（结论仍在），实际没收到",
    )
    ensure(
        not any(m.content == "排查记录 5：日志正常，继续。" for m in seen),
        "期望被折叠的中间消息不再以原文出现在模型输入里，实际还在",
    )
    ensure(
        sum(1 for m in seen if m.content.startswith(SUMMARY_HEADER)) == 1,
        "期望模型输入里恰好有一条摘要消息",
    )
    ensure(
        any(m.role == "system" and "运维助手" in m.content for m in seen),
        "期望系统提示原样进入模型输入，实际没有",
    )
    ensure(
        any("把结论发我。" in m.content for m in seen),
        "期望最近的消息仍然出现在模型输入里，实际不在",
    )
    ensure(
        len(seen) < len(history),
        f"期望模型收到的消息少于原始历史 {len(history)} 条，实际 {len(seen)} 条",
    )
    print(f"  原始历史 {len(history)} 条 → 模型实际收到 {len(seen)} 条")
    summary_lines = next(
        (m.content.splitlines() for m in seen if m.content.startswith(SUMMARY_HEADER)),
        [],
    )
    if len(summary_lines) > 1:
        print(f"  模型看到的摘要第一行：{summary_lines[1][:60]!r}")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型基于压缩后的上下文回答，只校验形状。"""
    banner("第 5 章 · 真实模型路径")
    from ..labkit import real_model

    model = real_model()

    history = build_work_history()
    policy = CompactionPolicy(keep_head=1, keep_recent=4, history_token_budget=200)
    compacted = compact(history, policy)

    messages = [system(SYSTEM_PROMPT)]
    if compacted.summary is not None:
        messages.append(compacted.summary)
    messages.extend(compacted.messages)
    messages.append(human("这次延迟的结论是什么？"))

    reply = model.complete(messages)
    print(f"  压缩后 {len(messages)} 条消息")
    print(f"  回答：{reply.content[:200]!r}")

    ensure(bool(reply.content), "真实模型应当给出回答")
    ensure(
        "延迟" in reply.content or "连接池" in reply.content,
        f"期望回答提到延迟的结论，实际 {reply.content!r}",
    )


# ---------------------------------------------------------------- 入口


def run() -> None:
    """跑第 5 章实验。"""
    banner("第 5 章 · 上下文工程：窗口预算与压缩")
    check_budget_table()
    check_tool_output_truncation()
    check_summary_compaction()
    check_protected_messages()
    check_context_corruption()
    check_token_reduction()
    check_short_task_no_compression()
    check_what_belongs_in_context()
    check_end_to_end_with_runtime()
    rule("上下文预算是分配问题：先定配额，再定砍谁；错误、约束、决策依据绝不压缩")
    print("\n✓ 第 5 章通过")
