"""第 12 章实验：评测体系——黄金集、轨迹评测与回归。

这个模块解决什么问题：第 11 章解决"出问题时怎么查"，这一章解决
"改完之后怎么知道是变好了还是变差了"。两件事缺一不可：查得出原因，
不等于说得清改动有没有用。

具体把四件事变成可执行的断言：

1. **"对"是三个维度同时成立**——结果（任务完成了吗）、轨迹（过程对吗）、
   成本（代价多少），任一维度不合格都算失败；
2. **黄金集跑分是可复现的**——同一份输入永远拿到同一个分数，
   否则分数没法当基线用；
3. **改动前后可比**——同一份黄金集跑两遍，涨了算变好、跌了算变差，
   跌了要被回归门禁挡住；
4. **分数会骗人**——LLM-as-judge 的偏袒，以及"把期望泄漏进提示词"造成的虚高。

本章建立在本仓库已有的毕业作品评测集（``agent_lab.capstone.evals``）之上：
``EvalCase`` / ``GOLDEN_CASES`` / ``score`` / ``run_golden_set`` 一律不重复实现，
这里只做三件事——把它按三个维度拆开、补上"改动前后可比"、补上"防过拟合"。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..capstone.evals import (
    GOLDEN_CASES,
    EvalCase,
    EvalReport,
    build_agent,
    run_golden_set,
    score,
)
from ..capstone.research_agent import ResearchResult
from ..labkit import (
    ChatModel,
    ModelReply,
    ToolCall,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
)

__all__ = ["run", "run_live"]


# ---------------------------------------------------------------- 三维评测


DIMENSIONS: tuple[str, ...] = ("结果", "轨迹", "成本")
"""三个评测维度。任一维度不合格，整条用例就算失败。"""

_PROBLEM_PREFIX: dict[str, str] = {
    "结果不对：": "结果",
    "轨迹不对：": "轨迹",
    "超预算：": "成本",
}
"""毕业作品的 ``score()`` 把三个维度压成"一个布尔值 + 一份问题清单"。

这里只把问题清单按前缀映射回维度名——不重新实现打分逻辑，
但这样"三维缺一不可"才能被断言，而不只是看到一个总的 False。
"""


@dataclass(frozen=True)
class DimensionResult:
    """一条用例的逐维度判定结果。"""

    passed: bool
    problems: dict[str, list[str]]

    def failed_dimensions(self) -> list[str]:
        """哪些维度不合格，按"结果 → 轨迹 → 成本"的顺序返回。"""
        return [name for name in DIMENSIONS if self.problems.get(name)]


def _evaluate(case: EvalCase) -> DimensionResult:
    """跑一条用例，给出逐维度判定。

    复用毕业作品的 :func:`score`——判定标准、工具集、审批关卡全都不重写，
    保证本章的结论和毕业作品的评测口径完全一致。
    """
    agent = build_agent(case.script)
    result = agent.run(case.question)
    ok, problems = score(result, case)

    buckets: dict[str, list[str]] = {name: [] for name in DIMENSIONS}
    buckets["其它"] = []
    for problem in problems:
        for prefix, name in _PROBLEM_PREFIX.items():
            if problem.startswith(prefix):
                buckets[name].append(problem)
                break
        else:
            buckets["其它"].append(problem)

    return DimensionResult(passed=ok, problems=buckets)


def success_rate(report: EvalReport) -> tuple[int, int, float]:
    """任务成功率：分子 = 通过用例数，分母 = 用例总数。

    分母里**包含**失败用例。把失败从分母里剔掉是常见的粉饰手法——
    跑挂了 3 条但只报跑了 1 条且成功，通过率就从 0% 变成 100%。
    """
    passed, total = report.passed, report.total
    return passed, total, (passed / total if total else 0.0)


def step_efficiency(result: ResearchResult, ideal_rounds: int) -> float:
    """步骤效率 = 实际模型轮次 / 理想轮次。

    小于 1 通常意味着模型没真做（跳过了必要步骤）；
    大于 1 说明它绕了路——绕路在结果评测里是查不出来的。
    """
    actual = sum(1 for m in result.messages if m.role == "assistant")
    return actual / max(1, ideal_rounds)


# ---------------------------------------------------------------- 用例集


def _known_failure_case() -> EvalCase:
    """一条从真实失败里抓出来的用例：结果对，但轨迹错了。

    线上事故原话：提示词里加了一句"回答前先把要点存一份笔记"，
    于是只读的问答任务也开始写笔记——多一次写操作、多一次审批打扰，
    而答案本身完全正确。只看结果的评测集，**看不见**这次事故。
    """
    return EvalCase(
        case_id="x01",
        question="知识库里 RAG 的三步是什么？",
        script=[
            ModelReply(
                tool_calls=[
                    ToolCall("call_1", "write_note", {"title": "RAG 三步", "content": "切分"})
                ],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
            ModelReply(
                content="切分、索引、召回。",
                usage={"prompt_tokens": 420, "completion_tokens": 15},
            ),
        ],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
        description="已知失败：结果对，但多调了不该调的写工具",
    )


CH12_CASES: tuple[EvalCase, ...] = GOLDEN_CASES + (_known_failure_case(),)
"""本章的黄金集：毕业作品的 8 条 + 1 条真实失败案例。

特意保留一条**已知失败**：黄金集里没有失败用例，门禁就没有灵敏度。
"""


@dataclass(frozen=True)
class HeldOutCase:
    """留出集用例。

    留出集**不参与任何调优**，只在最后验收——
    它的作用是回答"这个改动是不是只对被反复考过的那几道题有效"。
    """

    case: EvalCase
    guess: str
    """被泄漏策略"猜"出来的答案：不含期望关键词，所以会失败。"""


HELD_OUT: tuple[HeldOutCase, ...] = (
    HeldOutCase(
        case=EvalCase(
            case_id="h01",
            question="知识库里 RAG 的三个基本步骤，第二步叫什么？",
            script=[
                ModelReply(
                    tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索增强生成"})],
                    usage={"prompt_tokens": 300, "completion_tokens": 20},
                ),
                ModelReply(
                    content="检索增强生成的三个基本步骤：切分、索引、召回。",
                    usage={"prompt_tokens": 420, "completion_tokens": 15},
                ),
            ],
            expect_answer="索引",
            forbidden_tools=("write_note", "delete_note"),
            description="留出集：没参与调优的线上新问题",
        ),
        guess="切分和召回。",
    ),
    HeldOutCase(
        case=EvalCase(
            case_id="h02",
            question="知识库里说上下文预算是什么性质的问题？",
            script=[
                ModelReply(
                    tool_calls=[ToolCall("call_1", "search_docs", {"query": "上下文预算"})],
                    usage={"prompt_tokens": 300, "completion_tokens": 20},
                ),
                ModelReply(
                    content="上下文预算是分配问题，不是容量问题。",
                    usage={"prompt_tokens": 420, "completion_tokens": 15},
                ),
            ],
            expect_answer="容量",
            forbidden_tools=("write_note", "delete_note"),
            description="留出集：没参与调优的线上新问题",
        ),
        guess="上下文预算是资源有限的问题。",
    ),
)
"""留出集。

两条用例的期望关键词（"索引"、"容量"）**没有出现在黄金集任何一条的期望里**——
这是刻意的：如果泄漏的提示词里已经含有答案，留出集就测不出差别了。
"""


# ---------------------------------------------------------------- 行为策略


@dataclass(frozen=True)
class Policy:
    """一份"行为策略"——提示词的离线替身。

    真实世界里策略来自提示词与工具配置；这里用脚本代替，
    好处是同一份输入永远跑出同一个分数，改动前后可以直接比。

    局限也要说清楚：**提示词改动的连锁影响离线预测不了**。
    一个脚本只能说明"在我们指定的这条轨迹上分数变了"，
    说不出"线上所有任务都会这样变"——所以线上还要跑抽样评测。
    """

    name: str
    scripts: dict[str, list[ModelReply]] = field(default_factory=dict)

    def script_for(self, case: EvalCase) -> list[ModelReply]:
        """这条用例在该策略下走哪条轨迹。"""
        return self.scripts.get(case.case_id, case.script)


def _fixed_x01_script() -> list[ModelReply]:
    """修好版：把"回答前先存一份笔记"这句提示词去掉，模型回到只检索。"""
    return [
        ModelReply(
            tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索增强生成"})],
            usage={"prompt_tokens": 300, "completion_tokens": 20},
        ),
        ModelReply(
            content="切分、索引、召回。",
            usage={"prompt_tokens": 420, "completion_tokens": 15},
        ),
    ]


def _regressed_g08_script() -> list[ModelReply]:
    """退化版：提示词里加了"回答前一律先检索知识库"，连寒暄都开始检索。

    g08 的期望轨迹是"不调任何工具"，所以这条改动会直接踩中轨迹维度。
    """
    return [
        ModelReply(
            tool_calls=[ToolCall("call_1", "search_docs", {"query": "你好"})],
            usage={"prompt_tokens": 200, "completion_tokens": 12},
        ),
        ModelReply(
            content="你好，有什么可以帮你的？",
            usage={"prompt_tokens": 260, "completion_tokens": 10},
        ),
    ]


def _baseline_policy() -> Policy:
    """基线策略：完全按录制下来的真实轨迹跑。"""
    return Policy(name="基线（录制的真实轨迹）", scripts={})


def _fixed_policy() -> Policy:
    """修好版：只修掉那条已知失败。"""
    return Policy(name="修好版（去掉多余的存档提示词）", scripts={"x01": _fixed_x01_script()})


def _regressed_policy() -> Policy:
    """退化版：多了一句"回答前一律先检索"。"""
    return Policy(name="退化版（回答前一律先检索）", scripts={"g08": _regressed_g08_script()})


def _leaked_policy() -> Policy:
    """泄漏版：把黄金集的期望答案贴进提示词。

    真实的泄漏长这样：为了"提高通过率"，把标准答案写进 few-shot 示例，
    或者在系统提示里直接写"本类问题请回答：切分、索引、召回"。

    模型于是学会了一件事：**不检索，直接背**。
    黄金集里背得到 → 分数虚高；留出集里背不到 → 开始瞎猜。
    """
    scripts: dict[str, list[ModelReply]] = {}
    for case in CH12_CASES:
        scripts[case.case_id] = [
            ModelReply(
                content=f"答案：{case.expect_answer}",
                usage={"prompt_tokens": 120, "completion_tokens": 8},
            )
        ]
    for item in HELD_OUT:
        scripts[item.case.case_id] = [
            ModelReply(content=item.guess, usage={"prompt_tokens": 120, "completion_tokens": 8})
        ]
    return Policy(name="泄漏版（把期望贴进提示词）", scripts=scripts)


# ---------------------------------------------------------------- 跑评测


def _run_cases(cases: Sequence[EvalCase], policy: Policy) -> EvalReport:
    """按某份策略跑一批用例，返回报告。

    口径和毕业作品的 ``run_golden_set`` 一致：三维里任一维度不合格就记 0 分。
    """
    report = EvalReport()
    for case in cases:
        agent = build_agent(policy.script_for(case))
        result = agent.run(case.question)
        ok, problems = score(result, case)
        report.total += 1
        report.total_tokens += result.total_tokens
        if ok:
            report.passed += 1
            report.scores.append(1.0)
        else:
            report.scores.append(0.0)
            report.failures.append(f"{case.case_id}：{'；'.join(problems)}")
    return report


# ---------------------------------------------------------------- 回归门禁


@dataclass(frozen=True)
class GateResult:
    """回归门禁的判定结果。"""

    passed: bool
    exit_code: int
    message: str


def check_gate(report: EvalReport, baseline: float, *, tolerance: float = 0.0) -> GateResult:
    """回归门禁：分数低于基线就阻断。

    ``tolerance`` 给真实模型留一点抖动空间——真模型有随机性，
    同一份用例跑两遍分数会飘。但确定性路径下它应当是 0，
    飘了就说明你的"确定性"其实不成立。
    """
    if report.rate + tolerance < baseline:
        gap = baseline - report.rate
        return GateResult(
            passed=False,
            exit_code=1,
            message=(
                f"回归门禁未通过：本次 {report.rate:.0%} 低于基线 {baseline:.0%}"
                f"（差 {gap:.0%}）"
            ),
        )
    return GateResult(
        passed=True,
        exit_code=0,
        message=f"回归门禁通过：本次 {report.rate:.0%}，基线 {baseline:.0%}",
    )


@dataclass
class Baseline:
    """回归基线。

    基线要随时间更新，但不能随便更新。所以这里给更新加了两道闸：

    1. 必须写明原因——不许无声无息地改基线；
    2. 分数下降必须显式声明（``allow_drop=True``）——
       否则"把基线调低"就成了绕开门禁的后门。
    """

    score: float
    reason: str = ""
    updated_at: str = ""

    def update(self, score: float, reason: str, *, allow_drop: bool = False) -> None:
        """更新基线。

        参数：
            score: 新的基线分数
            reason: 更新原因，必填
            allow_drop: 是否允许分数下降，默认不允许
        """
        if not reason.strip():
            raise ValueError("更新基线必须写明原因")
        if score < self.score and not allow_drop:
            raise ValueError(
                f"新基线 {score:.0%} 低于当前基线 {self.score:.0%}："
                "分数下降必须显式 allow_drop=True 并写清原因"
            )
        self.score = score
        self.reason = reason
        self.updated_at = datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- LLM-as-judge


JUDGE_RUBRIC = """判定标准（逐条检查，全部满足才算通过）：
1. 答案直接回答了问题，没有跑题；
2. 答案包含关键信息「{keyword}」；
3. 答案没有编造与问题无关的事实。

输出格式：只写一行「得分：0」或「得分：1」，后面可以跟一句理由。"""
"""评委的判定标准。

标准必须**可执行**：读完这几条，评委（或人）能明确落成 0 或 1。
"回答得好不好"不是标准；"是否包含关键信息 X"才是标准。
"""


@dataclass(frozen=True)
class Verdict:
    """评委给出的判定。"""

    score: int
    """1 = 通过，0 = 不通过。"""

    reason: str


def parse_verdict(text: str) -> Verdict:
    """解析评委输出。

    只认「得分：N」这一种形式。评委的输出格式必须**固定**——
    否则每次都要写正则去捞，评测本身就成了一团不可靠的东西。
    """
    match = re.search(r"得分[:：]\s*([01])", text)
    if match is None:
        return Verdict(score=0, reason=f"评委输出无法解析：{text[:80]!r}")
    return Verdict(score=int(match.group(1)), reason=text.strip())


def judge_with_model(model: ChatModel, case: EvalCase, answer: str) -> Verdict:
    """用一个模型当评委，按 rubric 给答案打分。

    ``model`` 用 :func:`scripted_model` 驱动，所以这里的"评委"是确定性的。
    """
    prompt = (
        f"问题：{case.question}\n"
        f"待评答案：{answer}\n\n"
        + JUDGE_RUBRIC.format(keyword=case.expect_answer)
    )
    reply = model.complete([human(prompt)])
    return parse_verdict(reply.content)


_JUDGE_CASE = EvalCase(
    case_id="j01",
    question="知识库里 RAG 的三步是什么？",
    script=[],
    expect_answer="切分",
)
"""评委自测用的固定用例。"""


def _judge_samples() -> tuple[tuple[str, str, bool], ...]:
    """评委自测集：(样本名, 答案, 外部断言认为是否合格)。"""
    return (
        ("合格答案", "切分、索引、召回。", True),
        ("缺关键词", "大概就是那么回事。", False),
        ("空答案", "", False),
        ("跑题", "今天天气不错。", False),
    )


def _judge_selfcheck(replies: Sequence[str]) -> tuple[int, int, list[str]]:
    """拿一组脚本化回复当评委，跑一遍自测集。

    返回 ``(与外部断言一致的条数, 总条数, 假阳性列表)``。

    **假阳性比准确率更要命**：评委把不合格的说成合格，门禁就形同虚设；
    把合格的说成不合格只是保守，浪费一点时间而已。
    """
    model = scripted_model(list(replies))
    agree = 0
    false_positives: list[str] = []
    for name, answer, external_ok in _judge_samples():
        verdict = judge_with_model(model, _JUDGE_CASE, answer)
        if (verdict.score == 1) == external_ok:
            agree += 1
        elif verdict.score == 1:
            false_positives.append(name)
    return agree, len(_judge_samples()), false_positives


# ---------------------------------------------------------------- 1. 三维缺一不可


def check_three_dimensions() -> None:
    """结果对轨迹错 / 结果错轨迹对 / 结果轨迹都对但超成本。"""
    section("三个维度：任一维度不合格都算失败")

    case_a = EvalCase(
        case_id="dim-a",
        question="知识库里 RAG 的三步是什么？",
        script=[
            ModelReply(
                tool_calls=[
                    ToolCall("call_1", "write_note", {"title": "存档", "content": "切分"})
                ],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
            ModelReply(
                content="切分、索引、召回。",
                usage={"prompt_tokens": 420, "completion_tokens": 15},
            ),
        ],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
        description="结果对、轨迹错",
    )
    case_b = EvalCase(
        case_id="dim-b",
        question="知识库里怎么定义上下文预算？",
        script=[
            ModelReply(
                tool_calls=[ToolCall("call_1", "search_docs", {"query": "上下文预算"})],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
            ModelReply(
                content="上下文很重要，省着用。",
                usage={"prompt_tokens": 420, "completion_tokens": 15},
            ),
        ],
        expect_answer="分配",
        forbidden_tools=("write_note", "delete_note"),
        description="结果错、轨迹对",
    )
    case_c = EvalCase(
        case_id="dim-c",
        question="知识库里 RAG 的三步是什么？",
        script=[
            ModelReply(
                tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索增强生成"})],
                usage={"prompt_tokens": 5000, "completion_tokens": 800},
            ),
            ModelReply(
                content="切分、索引、召回。",
                usage={"prompt_tokens": 6000, "completion_tokens": 900},
            ),
        ],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
        max_tokens=4000,
        description="结果对、轨迹对，但超成本",
    )

    expected = {
        "dim-a": ["轨迹"],
        "dim-b": ["结果"],
        "dim-c": ["成本"],
    }
    for case in (case_a, case_b, case_c):
        outcome = _evaluate(case)
        failed = outcome.failed_dimensions()
        print(f"  {case.case_id}（{case.description}）→ 失败维度：{failed or '无'}")
        ensure(
            not outcome.passed,
            f"{case.case_id} 应当判为失败（三维里有一个不合格），实际判为通过",
        )
        ensure(
            failed == expected[case.case_id],
            f"{case.case_id} 期望失败维度是 {expected[case.case_id]}，实际 {failed}",
        )
        ensure(
            len(failed) == 1,
            f"{case.case_id} 应当只有一个维度不合格，实际 {len(failed)} 个：{failed}",
        )

    print("  → 三条都判失败，但失败原因各不相同：只有把维度拆开，才知道该改哪里")


# ---------------------------------------------------------------- 2. 黄金集跑分


def check_golden_set_score() -> None:
    """跑一遍黄金集，拿到一个确定的、可复现的分数。"""
    section("黄金集：同一份输入，永远拿到同一个分数")

    report = run_golden_set()
    passed, total, rate = success_rate(report)
    print(f"  第一次：{report.summary()}")

    ensure(report.total == 8, f"黄金集应当有 8 条用例，实际 {report.total}")
    ensure(report.passed == 8, f"黄金集应当 8 条全过，实际 {passed} 条通过")
    ensure(rate == 1.0, f"通过率应当是 1.0，实际 {rate}")
    ensure(not report.failures, f"黄金集不该有失败项，实际 {report.failures}")
    ensure(
        all(score_value == 1.0 for score_value in report.scores),
        f"每条用例的分数都应当是 1.0，实际 {report.scores}",
    )

    again = run_golden_set()
    ensure(
        again.rate == report.rate and again.passed == report.passed,
        f"跑两遍分数必须一样：第一次 {report.rate:.0%}，第二次 {again.rate:.0%}",
    )
    print(f"  第二次：{again.passed}/{again.total} 通过（{again.rate:.0%}）—— 可复现")
    print(f"  指标定义：任务成功率 = 通过数 / 总数 = {passed}/{total} = {rate:.0%}")
    print("  → 分子分母必须钉死。失败用例留在分母里，不能从分母中剔除")


# ---------------------------------------------------------------- 3. 改动前后可比


def check_change_detection() -> None:
    """改一个行为，让分数上升或下降，两个方向都要能检出。"""
    section("改动前后可比：涨了是变好，跌了是变差")

    policies = (_baseline_policy(), _fixed_policy(), _regressed_policy())
    results: dict[str, EvalReport] = {}
    for policy in policies:
        report = _run_cases(CH12_CASES, policy)
        results[policy.name] = report
        passed, total, rate = success_rate(report)
        print(f"  {policy.name:<28} {passed}/{total}（{rate:.0%}）")

    baseline = results[policies[0].name]
    fixed = results[policies[1].name]
    regressed = results[policies[2].name]

    ensure(
        fixed.passed == baseline.passed + 1,
        f"修好版应当比基线多通过 1 条（x01），实际 {baseline.passed} → {fixed.passed}",
    )
    ensure(
        fixed.rate > baseline.rate,
        f"修好版的分数应当高于基线：{baseline.rate:.0%} → {fixed.rate:.0%}",
    )
    ensure(
        regressed.passed == baseline.passed - 1,
        f"退化版应当比基线少通过 1 条（g08），实际 {baseline.passed} → {regressed.passed}",
    )
    ensure(
        regressed.rate < baseline.rate,
        f"退化版的分数应当低于基线：{baseline.rate:.0%} → {regressed.rate:.0%}",
    )
    ensure(
        fixed.rate - baseline.rate > 0 > regressed.rate - baseline.rate,
        "同一个评测器应当能同时检出'变好'和'变差'两个方向",
    )

    print(f"  基线 {baseline.rate:.0%} → 修好 {fixed.rate:.0%}（+{fixed.rate - baseline.rate:.0%}）")
    print(f"  基线 {baseline.rate:.0%} → 退化 {regressed.rate:.0%}（{regressed.rate - baseline.rate:.0%}）")
    for failure in regressed.failures:
        print(f"    退化明细：{failure}")


# ---------------------------------------------------------------- 4. 回归门禁


def check_regression_gate() -> None:
    """分数低于基线 → 判失败并给出非零退出码。"""
    section("回归门禁：分数掉了就阻断")

    baseline_report = _run_cases(CH12_CASES, _baseline_policy())
    baseline_rate = baseline_report.rate
    baseline = Baseline(score=baseline_rate, reason="初始基线")

    fixed_gate = check_gate(_run_cases(CH12_CASES, _fixed_policy()), baseline_rate)
    regressed_gate = check_gate(_run_cases(CH12_CASES, _regressed_policy()), baseline_rate)

    print(f"  修好版：{fixed_gate.message}（exit_code={fixed_gate.exit_code}）")
    print(f"  退化版：{regressed_gate.message}（exit_code={regressed_gate.exit_code}）")

    ensure(fixed_gate.passed, f"分数上升应当放行，实际：{fixed_gate.message}")
    ensure(fixed_gate.exit_code == 0, f"放行时退出码应当是 0，实际 {fixed_gate.exit_code}")
    ensure(not regressed_gate.passed, f"分数下跌应当被拦住，实际：{regressed_gate.message}")
    ensure(
        regressed_gate.exit_code != 0,
        f"被拦住时退出码必须非零（CI 靠它判定失败），实际 {regressed_gate.exit_code}",
    )

    # 基线可以更新，但不能随便更新
    try:
        baseline.update(0.9, "")
    except ValueError as exc:
        print(f"  无原因更新基线 → 被拒：{exc}")
    else:
        raise AssertionError("更新基线不写原因应当被拒绝")

    try:
        baseline.update(0.5, "线上退化了")
    except ValueError as exc:
        print(f"  分数下降就更新基线 → 被拒：{exc}")
    else:
        raise AssertionError("分数下降时更新基线应当默认被拒绝")

    baseline.update(0.5, "线上退化已确认，需要临时放宽", allow_drop=True)
    ensure(
        baseline.score == 0.5 and "临时放宽" in baseline.reason,
        f"显式允许下降后应当更新成功，实际 score={baseline.score} reason={baseline.reason!r}",
    )
    print(f"  显式允许下降 → 基线更新为 {baseline.score:.0%}（原因已记录）")


# ---------------------------------------------------------------- 5. LLM-as-judge


def check_llm_as_judge() -> None:
    """评委模型：判定要和外部断言一致，偏袒要被看见。"""
    section("LLM-as-judge：能用，但不能当判据")

    strict_replies = (
        "得分：1 答案包含关键信息且直接回答了问题",
        "得分：0 缺少关键信息",
        "得分：0 没有回答问题",
        "得分：0 与问题无关",
    )
    biased_replies = (
        "得分：1 答案包含关键信息且直接回答了问题",
        "得分：1 看起来没什么问题",
        "得分：0 没有回答问题",
        "得分：1 整体没问题",
    )

    agree, total, false_positives = _judge_selfcheck(strict_replies)
    print(f"  严格评委：{agree}/{total} 与外部断言一致，假阳性 {len(false_positives)} 条")
    ensure(
        agree == total,
        f"严格评委应当和外部断言完全一致，实际 {agree}/{total}",
    )
    ensure(
        not false_positives,
        f"严格评委不该有假阳性，实际 {false_positives}",
    )

    agree_biased, _, false_positives_biased = _judge_selfcheck(biased_replies)
    print(f"  偏袒评委：{agree_biased}/{total} 一致，假阳性 {len(false_positives_biased)} 条 {false_positives_biased}")
    ensure(
        len(false_positives_biased) == 2,
        f"偏袒评委应当出现 2 条假阳性，实际 {len(false_positives_biased)} 条",
    )

    # 现场演示"judge 自评说没问题，但实际有问题"
    judge_case = EvalCase(
        case_id="j-demo",
        question="知识库里 RAG 的三步是什么？",
        script=[],
        expect_answer="切分",
    )
    weak_answer = "大概就是那么回事。"
    biased_model = scripted_model([biased_replies[1]])
    verdict = judge_with_model(biased_model, judge_case, weak_answer)
    external_ok = judge_case.expect_answer in weak_answer

    print(f"  偏袒评委对不合格答案的判定：得分 {verdict.score}（{verdict.reason}）")
    print(f"  外部断言（必须包含「{judge_case.expect_answer}」）：{'通过' if external_ok else '不通过'}")

    ensure(
        verdict.score == 1,
        f"这个场景要能复现偏袒：评委应当给 1 分，实际给了 {verdict.score}",
    )
    ensure(
        not external_ok,
        "外部断言应当判定为不通过——答案里根本没有期望的关键信息",
    )

    # judge 的一致性：同一个评委跑两遍，判定必须一样
    first = judge_with_model(scripted_model(strict_replies), judge_case, "切分、索引、召回。")
    second = judge_with_model(scripted_model(strict_replies), judge_case, "切分、索引、召回。")
    ensure(
        first == second,
        f"同一份判定标准跑两遍结果应当一致：{first} vs {second}",
    )
    print(f"  judge 一致性自测：同一输入两遍判定相同（{first.score}）")
    print("  → 评委是启发式，不是判据。门禁必须用外部断言兜底")


# ---------------------------------------------------------------- 6. 轨迹评测


def check_trajectory_catches_forbidden_tool() -> None:
    """结果对、但调用了不该调的工具——只看结果看不见。"""
    section("轨迹评测：结果对但过程错")

    case = _known_failure_case()
    outcome = _evaluate(case)
    print(f"  用例：{case.description}")
    print(f"  失败维度：{outcome.failed_dimensions()}")

    ensure(
        not outcome.passed,
        f"调用了不该调的工具应当判为失败，实际判为通过",
    )
    ensure(
        outcome.failed_dimensions() == ["轨迹"],
        f"应当只有轨迹维度不合格，实际 {outcome.failed_dimensions()}",
    )
    ensure(
        not outcome.problems["结果"],
        f"答案本身是对的，结果维度不该报问题，实际 {outcome.problems['结果']}",
    )
    ensure(
        any("write_note" in problem for problem in outcome.problems["轨迹"]),
        f"轨迹问题里应当点名 write_note，实际 {outcome.problems['轨迹']}",
    )

    # 轨迹评测抓的是"调用行为"，不只是"是否真的执行"——
    # 写工具被审批关卡拦下了（没有真的写），但这次调用依然是不该发生的。
    agent = build_agent(case.script)
    result = agent.run(case.question)
    ensure(
        len(result.tool_calls) == 1 and result.tool_calls[0].name == "write_note",
        f"应当记录到一次 write_note 调用，实际 {[c.name for c in result.tool_calls]}",
    )
    print("  → 审批拦住了执行，但轨迹评测抓的是「这次调用不该发生」——副作用和额外成本都是它带来的")


# ---------------------------------------------------------------- 7. 成本维度


def check_cost_budget() -> None:
    """结果对、轨迹对，但超预算 → 判失败。"""
    section("成本维度：结果对、轨迹对，但超预算")

    case = EvalCase(
        case_id="cost-a",
        question="知识库里 RAG 的三步是什么？",
        script=[
            ModelReply(
                tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索增强生成"})],
                usage={"prompt_tokens": 5000, "completion_tokens": 800},
            ),
            ModelReply(
                content="切分、索引、召回。",
                usage={"prompt_tokens": 6000, "completion_tokens": 900},
            ),
        ],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
        max_tokens=4000,
        description="结果对、轨迹对，但花了 12700 token",
    )
    outcome = _evaluate(case)
    print(f"  失败维度：{outcome.failed_dimensions()}")

    ensure(
        not outcome.passed,
        "超预算应当判为失败，实际判为通过",
    )
    ensure(
        outcome.failed_dimensions() == ["成本"],
        f"应当只有成本维度不合格，实际 {outcome.failed_dimensions()}",
    )
    ensure(
        not outcome.problems["结果"] and not outcome.problems["轨迹"],
        "结果和轨迹都该合格，实际 "
        f"结果 {outcome.problems['结果']}、轨迹 {outcome.problems['轨迹']}",
    )
    ensure(
        any("12700" in problem for problem in outcome.problems["成本"]),
        f"成本问题里应当写出实际用量，实际 {outcome.problems['成本']}",
    )

    # 步骤效率：同一批答案，理想 2 轮，实际 3 轮 → 绕了路
    wandering = EvalCase(
        case_id="cost-b",
        question="知识库里 RAG 的三步是什么？",
        script=[
            ModelReply(
                tool_calls=[ToolCall("call_1", "search_docs", {"query": "检索"})],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
            ModelReply(
                tool_calls=[ToolCall("call_2", "search_docs", {"query": "检索增强生成"})],
                usage={"prompt_tokens": 400, "completion_tokens": 20},
            ),
            ModelReply(
                content="切分、索引、召回。",
                usage={"prompt_tokens": 420, "completion_tokens": 15},
            ),
        ],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
        description="结果对、轨迹对、也不超预算，但绕了一圈",
    )
    agent = build_agent(wandering.script)
    result = agent.run(wandering.question)
    efficiency = step_efficiency(result, ideal_rounds=2)
    print(f"  步骤效率：{efficiency:.1f}（实际 3 轮 / 理想 2 轮）")
    ensure(
        efficiency == 1.5,
        f"步骤效率应当是 1.5（3 轮 / 2 轮），实际 {efficiency}",
    )
    print("  → 三个维度全过，但多绕了一圈——步骤效率是三维之外的第四个视角")


# ---------------------------------------------------------------- 8. 防过拟合


def check_overfitting() -> None:
    """把期望泄漏进提示词：分数虚高，线上没变好。"""
    section("防过拟合：分数涨了，但线上没变好")

    golden_cases = [item.case for item in HELD_OUT]
    baseline_golden = _run_cases(CH12_CASES, _baseline_policy())
    fixed_golden = _run_cases(CH12_CASES, _fixed_policy())
    leaked_golden = _run_cases(CH12_CASES, _leaked_policy())

    baseline_held = _run_cases(golden_cases, _baseline_policy())
    fixed_held = _run_cases(golden_cases, _fixed_policy())
    leaked_held = _run_cases(golden_cases, _leaked_policy())

    def _row(label: str, golden: EvalReport, held: EvalReport) -> None:
        print(
            f"  {label:<28} 黄金集 {golden.passed}/{golden.total}（{golden.rate:.0%}）"
            f"　留出集 {held.passed}/{held.total}（{held.rate:.0%}）"
        )

    _row("基线（真实轨迹）", baseline_golden, baseline_held)
    _row("修好版（真改了行为）", fixed_golden, fixed_held)
    _row("泄漏版（贴了答案）", leaked_golden, leaked_held)

    ensure(
        leaked_golden.passed == fixed_golden.passed,
        f"泄漏版和修好版在黄金集上的分数应当一样（都是 {fixed_golden.passed}），"
        f"实际泄漏版 {leaked_golden.passed}、修好版 {fixed_golden.passed}",
    )
    ensure(
        leaked_golden.rate > baseline_golden.rate,
        f"泄漏版的黄金集分数应当虚高：基线 {baseline_golden.rate:.0%} → 泄漏 {leaked_golden.rate:.0%}",
    )
    ensure(
        leaked_held.rate < fixed_held.rate,
        f"留出集上泄漏版应当明显更差：修好版 {fixed_held.rate:.0%} → 泄漏版 {leaked_held.rate:.0%}",
    )
    ensure(
        fixed_held.rate == baseline_held.rate == 1.0,
        f"修好版不该伤到留出集，实际 {fixed_held.rate:.0%}（基线 {baseline_held.rate:.0%}）",
    )

    print(f"  泄漏版在黄金集上和修好版同分（{leaked_golden.rate:.0%}），")
    print(f"  但在留出集上塌了：{leaked_held.rate:.0%} vs 修好版 {fixed_held.rate:.0%}")
    print("  → 这就是防过拟合要防的东西：分数涨了，线上没变好")
    print("  → 扩充机制：线上每出现一次真实失败，就往黄金集里补一条用例，并记录来源")


# ---------------------------------------------------------------- 汇总


def run() -> None:
    """跑第 12 章实验。"""
    banner("第 12 章 · 评测体系：黄金集、轨迹评测与回归")
    check_three_dimensions()
    check_golden_set_score()
    check_change_detection()
    check_regression_gate()
    check_llm_as_judge()
    check_trajectory_catches_forbidden_tool()
    check_cost_budget()
    check_overfitting()
    rule("对 = 结果对 ∧ 轨迹对 ∧ 成本没超；黄金集从失败里长出来；分数掉了就阻断")
    print("\n✓ 第 12 章通过")


def run_live() -> None:
    """真实模型路径：拿真模型当被评对象，只校验"评测器还能跑"，不校验分数。

    真实模型的分数会飘——温度、模型版本、上下文长度都会影响它。
    所以线上抽样评测看的是**趋势**，不是某个绝对数字；
    而 CI 里的确定性路径才用来卡门禁。
    """
    banner("第 12 章 · 真实模型路径")
    from ..labkit import real_model

    case = EvalCase(
        case_id="live-1",
        question="知识库里 RAG 的三步是什么？",
        script=[],
        expect_answer="切分",
        forbidden_tools=("write_note", "delete_note"),
    )

    model = real_model()
    result = model.complete([human(case.question)])

    print(f"  模型回答：{result.content[:120]!r}")
    ensure(bool(result.content), "真实模型应当给出非空回答")
    ensure(
        isinstance(result.usage, dict),
        f"真实模型应当返回 usage 统计，实际 {type(result.usage).__name__}",
    )
    print(f"  usage：{result.usage}")
    print("  → 真实路径只校验形状：分数会飘，门禁交给确定性路径")
