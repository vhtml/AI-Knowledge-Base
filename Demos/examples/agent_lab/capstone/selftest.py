"""毕业作品自检：把 17 章的毕业标准变成可执行的断言。

这个模块解决什么问题：教程最后总要有个"学完了吗"的答案。这里不靠感觉，
而是把 `AI/Agent/README.md` 里的毕业标准逐条变成断言：

1. **黄金集全过**——回归基线是 8/8，少一条就红；
2. **审批真的拦得住**——危险工具在被批准前不会执行（不只是"有审批记录"）；
3. **审批可以被人工改参数**——改完用的是改后的参数；
4. **记忆跨会话有效**，且**能被用户更正**；
5. **未召回的记忆不进上下文**；
6. **预算自洽**，且超预算的任务被判失败；
7. **轨迹完整可回放**，且**敏感字段已脱敏**；
8. **成本可估算**。

不依赖 pytest、不联网、不调模型。

    cd Demos/examples
    uv run --project .. python -m agent_lab.capstone.selftest
"""

from __future__ import annotations

from ..labkit import ModelReply, ScriptedModel, ToolCall, banner, ensure, rule, section
from .evals import GOLDEN_CASES, build_agent, build_tools, run_golden_set, score
from .research_agent import (
    ApprovalGate,
    Budget,
    MemoryStore,
    ResearchAgent,
    RiskLevel,
    ToolEntry,
)

# 回归基线：黄金集必须全过
BASELINE_PASSED = 8
BASELINE_TOTAL = 8


def check_golden_set() -> list[str]:
    """黄金集必须全过——这就是回归基线。"""
    section("黄金集：回归基线")
    report = run_golden_set()

    print(f"  {report.summary()}")
    print(f"  耗时 {report.elapsed_ms:.0f} ms")
    for failure in report.failures:
        print(f"  ✗ {failure}")

    failures: list[str] = []
    if report.total != BASELINE_TOTAL:
        failures.append(f"黄金集用例数应为 {BASELINE_TOTAL}，实际 {report.total}")
    if report.passed != BASELINE_PASSED:
        failures.append(
            f"回归基线未达成：应过 {BASELINE_PASSED}/{BASELINE_TOTAL}，实际 {report.passed}/{report.total}"
        )
    return failures


def check_approval_blocks_execution() -> list[str]:
    """危险工具在被批准前**不得执行**——不只是"有一条审批记录"。"""
    section("审批：危险操作真的被拦住了")

    tools = build_tools()
    delete_tool = next(t for t in tools if t.name == "delete_note")

    executed: list[str] = []
    original = delete_tool.func

    def spy(note_id: str) -> str:
        executed.append(note_id)
        return original(note_id)

    delete_tool.func = spy

    model = ScriptedModel(
        script=[
            ModelReply(tool_calls=[ToolCall("call_1", "delete_note", {"note_id": "old-1"})]),
            ModelReply(content="删除未获批准，我没有执行。"),
        ]
    )
    agent = ResearchAgent(model, tools=tools, approval=ApprovalGate(), auto_approve=False)
    result = agent.run("删掉那篇旧笔记。")

    ensure(not executed, f"未批准时 delete_note 不该被执行，实际执行了 {executed}")
    ensure(
        len(result.approvals) == 1,
        f"应当登记一次审批，实际 {len(result.approvals)} 次",
    )
    ensure(
        "批准" in result.final_answer or "拒绝" in result.final_answer or "未获" in result.final_answer,
        f"最终回答应当说明被拦下了，实际 {result.final_answer!r}",
    )
    print(f"  未批准 → 未执行（审批记录 {len(result.approvals)} 条）")
    return []


def check_approval_can_modify_arguments() -> list[str]:
    """人可以改参数后放行，执行时用的是**改后**的参数。"""
    section("审批：人工改参数后放行")

    tools = build_tools()
    write_tool = next(t for t in tools if t.name == "write_note")

    captured: list[dict[str, object]] = []
    original = write_tool.func

    def spy(title: str, content: str) -> str:
        captured.append({"title": title, "content": content})
        return original(title, content)

    write_tool.func = spy

    # 路径一：走完整的 Agent 流程，自动批准时写操作真的执行了
    model = ScriptedModel(
        script=[
            ModelReply(
                tool_calls=[
                    ToolCall("call_1", "write_note", {"title": "草稿", "content": "原始内容"})
                ]
            ),
            ModelReply(content="已按你的要求写入。"),
        ]
    )
    agent = ResearchAgent(model, tools=tools, approval=ApprovalGate(), auto_approve=True)
    result = agent.run("写一篇笔记。")

    ensure(
        len(captured) == 1,
        f"批准后写操作应当执行一次，实际 {len(captured)} 次",
    )
    ensure(
        captured[0]["title"] == "草稿",
        f"自动批准时应当用原参数执行，实际 {captured[0]}",
    )

    # 路径二：人工在批准时改参数，关卡返回的是**改后**的参数
    gate = ApprovalGate()
    request = gate.request(write_tool, {"title": "草稿", "content": "原始内容"})
    ensure(gate.needs_approval(write_tool), "write_note 应当需要审批")

    approved = gate.approve(request, {"title": "定稿", "content": "原始内容"})
    ensure(
        approved["title"] == "定稿",
        f"批准时改的参数应当生效，实际 {approved}",
    )
    ensure(
        request.arguments["title"] == "草稿",
        "原始请求参数不应被就地改写（留痕用）",
    )
    print("  自动批准用原参数；人工批准可用改后参数")
    return []


def check_memory_persists_and_corrects() -> list[str]:
    """记忆跨会话有效，且能被用户更正（污染的修复路径）。"""
    section("记忆：跨会话有效 + 可被更正")

    memory = MemoryStore(capacity=10)

    # 记忆写入必须走一条显式的工具调用——这样它可审计
    agent = build_agent(
        [
            ModelReply(
                tool_calls=[
                    ToolCall(
                        "call_1",
                        "remember_preference",
                        {"key": "语言偏好", "content": "只接受中文回答"},
                    )
                ]
            ),
            ModelReply(content="好的，我记住了。"),
        ],
        memory=memory,
        auto_approve=True,
    )
    agent.run("记住：我只接受中文回答")

    ensure(len(memory) == 1, f"应当写入一条记忆，实际 {len(memory)} 条")

    # 新会话（新 Agent 实例，同一个 memory store）应当能召回
    recalled = memory.recall("回答语言")
    ensure(len(recalled) == 1, f"新会话应当能召回，实际召回 {len(recalled)} 条")
    ensure(
        "中文" in recalled[0].content,
        f"召回的内容不对：{recalled[0].content!r}",
    )
    print(f"  跨会话召回：{recalled[0].content!r}")

    # 污染：写入一条错误事实
    memory.write("语言偏好", "用户只接受英文回答", importance=1)
    polluted = memory.recall("回答语言")
    ensure(
        any("英文" in entry.content for entry in polluted),
        "错误事实应当确实污染了召回结果（这正是风险所在）",
    )

    # 更正：用户改回来
    memory.correct("语言偏好", "用户只接受中文回答")
    corrected = memory.recall("回答语言")
    ensure(
        all("英文" not in entry.content for entry in corrected),
        f"更正后不该再召回错误事实，实际 {[e.content for e in corrected]}",
    )
    print("  污染 → 更正 → 修复路径可用")
    return []


def check_unrecalled_memory_not_in_context() -> list[str]:
    """未召回的记忆不得出现在发给模型的消息里。"""
    section("记忆：未召回的不进上下文")

    memory = MemoryStore(capacity=10)
    memory.write("无关偏好", "用户喜欢蓝色", importance=1)

    agent = build_agent(
        [ModelReply(content="收到。")],
        memory=memory,
    )
    result = agent.run("今天天气怎么样")

    injected = [m.content for m in result.messages if m.role == "system"]
    joined = "\n".join(injected)
    ensure(
        "蓝色" not in joined,
        f"未召回的记忆不应进入上下文，实际上下文里出现了：{joined!r}",
    )
    print("  未召回的记忆不在上下文里")
    return []


def check_budget() -> list[str]:
    """预算表必须自洽，且超预算的任务被判失败。"""
    section("预算：配额自洽 + 超预算判失败")

    budget = Budget()
    ensure(
        budget.allocated() <= budget.total,
        f"配额之和 {budget.allocated()} 不应超过总窗口 {budget.total}",
    )
    ensure(
        budget.remaining() >= 0,
        f"预算应当有正余量，实际 {budget.remaining()}",
    )
    print(
        f"  配额合计 {budget.allocated()} / 总窗口 {budget.total}，余量 {budget.remaining()}"
    )

    # 超预算的用例在打分时应当判失败
    from .evals import EvalCase

    tight = EvalCase(
        case_id="budget-check",
        question="测试",
        script=[
            ModelReply(
                content="好",
                usage={"prompt_tokens": 500, "completion_tokens": 100},
            )
        ],
        expect_answer="好",
        max_tokens=1,
    )
    agent = build_agent(tight.script)
    result = agent.run("测试")
    ok, problems = score(result, tight)
    ensure(not ok, "超预算的用例应当判失败")
    ensure(
        any("超预算" in problem for problem in problems),
        f"失败原因应当提到超预算，实际 {problems}",
    )
    print("  超预算 → 判失败")
    return []


def check_trace_is_complete_and_redacted() -> list[str]:
    """轨迹要完整（含工具调用与 token），且敏感字段已脱敏。"""
    section("轨迹：完整 + 脱敏")

    agent = build_agent(
        [
            ModelReply(
                tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索增强生成"})],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
            ModelReply(content="切分、索引、召回。", usage={"prompt_tokens": 400, "completion_tokens": 15}),
        ]
    )
    result = agent.run("RAG 三步是什么？")

    kinds = [event.kind for event in result.trace.events]
    print(f"  轨迹事件：{kinds}")

    ensure("run_start" in kinds, "轨迹应当有 run_start")
    ensure("model" in kinds, "轨迹应当有 model 事件")
    ensure("tool_result" in kinds, "轨迹应当有 tool_result 事件")
    ensure("run_end" in kinds, "轨迹应当有 run_end")

    ensure(result.total_tokens == 735, f"token 应当被累加，实际 {result.total_tokens}")
    ensure(result.estimated_cost() > 0, "应当能估算出成本")

    # 脱敏：直接往 trace 里写敏感字段
    result.trace.record("test", api_key="sk-should-not-appear", nested={"password": "hunter2"})
    payload = result.trace.events[-1].payload
    ensure(
        payload["api_key"] == "***REDACTED***",
        f"api_key 应当被脱敏，实际 {payload['api_key']!r}",
    )
    ensure(
        payload["nested"]["password"] == "***REDACTED***",
        f"嵌套的 password 应当被脱敏，实际 {payload['nested']['password']!r}",
    )
    print(f"  token {result.total_tokens}，成本估算 ${result.estimated_cost():.6f}，脱敏正常")
    return []


def check_unknown_tool_and_business_failure() -> list[str]:
    """幻觉工具名与业务故障都要回填给模型，且措辞不同。"""
    section("工具失败：两类错误两种措辞")

    tools = build_tools()

    # 幻觉工具名
    model = ScriptedModel(
        script=[
            ModelReply(tool_calls=[ToolCall("call_1", "calculator", {"expr": "1+1"})]),
            ModelReply(content="没有 calculator 工具。"),
        ]
    )
    agent = ResearchAgent(model, tools=tools)
    result = agent.run("算个数")
    tool_messages = [m for m in result.messages if m.role == "tool"]
    ensure(len(tool_messages) == 1, f"应当有一条工具消息，实际 {len(tool_messages)}")
    ensure(
        "没有名为 calculator 的工具" in tool_messages[0].content,
        f"幻觉工具名的回填应当点名工具，实际 {tool_messages[0].content!r}",
    )

    # 业务故障（读不存在的文档）
    model2 = ScriptedModel(
        script=[
            ModelReply(tool_calls=[ToolCall("call_1", "read_doc", {"doc_id": "doc-999"})]),
            ModelReply(content="文档不存在。"),
        ]
    )
    agent2 = ResearchAgent(model2, tools=build_tools())
    result2 = agent2.run("读 doc-999")
    tool_messages2 = [m for m in result2.messages if m.role == "tool"]
    ensure(
        "工具执行失败" in tool_messages2[0].content,
        f"业务故障的回填应当说明是工具故障，实际 {tool_messages2[0].content!r}",
    )
    ensure(
        "重试无用" in tool_messages2[0].content,
        f"业务故障的回填应当提示模型换路，实际 {tool_messages2[0].content!r}",
    )
    print("  幻觉工具名 / 业务故障 → 两种措辞")
    return []


def run() -> int:
    """跑完整自检。

    返回进程退出码：0 全部通过，1 有失败项。
    """
    banner("毕业作品自检 · 把 17 章的毕业标准变成断言")

    failures: list[str] = []
    for check in (
        check_golden_set,
        check_approval_blocks_execution,
        check_approval_can_modify_arguments,
        check_memory_persists_and_corrects,
        check_unrecalled_memory_not_in_context,
        check_budget,
        check_trace_is_complete_and_redacted,
        check_unknown_tool_and_business_failure,
    ):
        failures.extend(check())

    print()
    print("=" * 72)
    if failures:
        print(f"毕业自检未通过：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"毕业自检通过：8 组断言 + 黄金集 {BASELINE_PASSED}/{BASELINE_TOTAL}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run())
