"""第 8 章实验：反思与自纠错。

这个模块解决什么问题：第 7 章让 Agent 会"先想再做"，第 10 章讲"失败了怎么重试"，
中间还缺一块——模型**已经产出一个结果之后**，怎么发现它不对并改对。

"让模型自己检查自己的输出"这句话听起来很美，但它是本章要证伪的对象。所以本章
把"反思"拆成七条可执行的断言，全部离线运行（不联网、不调模型、不花钱）：

1. **反思能改对**——初稿有错 → 批评点名具体问题 → 修订后 checker 判定正确；
   同时证明初稿确实是错的（不是把对的改错）。
2. **消息数会增加**——反思比不反思多出"批评"和"修订"两轮往返。
3. **循环要有上限**——连续 N 轮评分没提升就停，而且返回**最好的一版**，
   不是最后一版（用确定性评分函数证明最后那版更差）。
4. **外部验证优先**——模型自评说"没问题"，checker 说"有问题"，以 checker 为准；
   有 checker 时运行时**根本不该**问模型"你觉得对吗"。
5. **成本**——反思路径的模型调用次数严格多于不反思路径，且上下文逐轮变长。
6. **无效反思**——批评是假的（"这里逻辑不严谨"），好稿被改坏；
   "返回最好的一版"这道机制挡住了这次破坏。
7. **开放任务不适用**——同一套机制跑在开放式写作上，单位 token 买到的
   可测评分更少，而且第二轮反思会把分数改回去。

两个 checker 都是确定性的：一个跑测试用例（代码任务），一个查机械性质
（写作任务）。**没有一处判定交给模型**——这正是本章的结论。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..labkit import (
    ChatModel,
    Message,
    approx_tokens,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    system,
)

# ---------------------------------------------------------------- 常量

SYSTEM_PROMPT = (
    "你是一个严谨的工程师。只在被要求时给出最终答案，"
    "不要输出与代码本身无关的解释。"
)

CRITIQUE_PROMPT = """请审查下面这份初稿。

要求：
- 只列出**具体**问题——哪一行、哪个输入值、哪个用例不对，为什么不对。
- 不要写"不够好""建议优化"这类无法执行的评语。
- 如果确实没有可指出的具体问题，直接回答"没有具体问题"。

任务：
{task}

初稿：
{draft}
"""

REVISE_PROMPT = """请根据下面的审查意见改写初稿。

只针对意见里点名的具体问题修改，其他部分尽量保持原样。
改完后只输出改写后的完整内容，不要输出解释。

审查意见：
{critique}
"""

SELF_EVAL_PROMPT = """判定：这段答案是否正确？请只回答「没问题」或「有问题：……」。

答案：
{answer}
"""

SELF_EVAL_MARK = "判定：这段答案是否正确"
"""出现这个标记，就说明有人让模型给自己的答案打分了——第 8 章要证伪的正是这条路。"""

CODE_TASK = "写一个函数 sum_squares(n)，返回 1²+2²+…+n²。"
"""可验证任务：结果能用测试用例判定真假。"""

WRONG_CODE = """def sum_squares(n):
    return n * (n + 1) // 2
"""
"""初稿：用等差数列求和冒充平方和。"""

PARTIAL_CODE = """def sum_squares(n):
    if n <= 2:
        return sum(i * i for i in range(1, n + 1))
    return n * (n + 1) // 2
"""
"""改了一半：小的 n 对了，大的 n 还错。"""

CORRECT_CODE = """def sum_squares(n):
    return sum(i * i for i in range(1, n + 1))
"""
"""正确版本。"""

OFF_BY_ONE_CODE = """def sum_squares(n):
    return n * n
"""

SQUARE_FORMULA_CODE = """def sum_squares(n):
    return n * (n - 1)
"""

TRUE_CRITIQUE = (
    "第 2 行 return n * (n + 1) // 2 求的是 1+2+…+n 的和，不是平方和。"
    "n=3 时正确答案是 14，这行返回 6，用例 sum_squares(3) 会失败。"
    "请改成对 i*i 求和。"
)
"""合格的批评：点名了行号、输入值、期望值和错误原因。"""

PARTIAL_CRITIQUE = (
    "n=3 时你应该返回 14，但现在返回的是 6；n=4 时应该返回 30，现在返回 10。"
    "分支条件写错了，只覆盖了很小的 n。"
)

FALSE_CRITIQUE = "这里的实现逻辑不严谨，建议直接用等差数列公式一行简化掉。"
"""幻觉批评：听起来像那么回事，但没指出任何具体问题——照着改反而更糟。"""

SUM_SQUARES_CASES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 5),
    (3, 14),
    (4, 30),
    (5, 55),
    (6, 91),
    (7, 140),
)
"""确定性测试用例：(输入 n, 1²+…+n² 的期望值)。"""

COPY_TASK = "写一段 80 字左右的产品介绍，语气亲切，介绍效率工具「星尘」。"
"""开放式任务：没有"正确答案"，只能看机械性质。"""

COPY_TARGET_NAME = "星尘"
COPY_MIN_CHARS = 60
COPY_MAX_CHARS = 120

COPY_DRAFT = "星尘 是一款新的效率工具。"
"""初稿：有产品名，但字数远不够 60。"""

COPY_REVISED = (
    "星尘 是一款新的效率工具，它把重复的整理工作交给我们，"
    "让你把时间留给真正重要的事。今天就把繁琐的流程交出来吧，试试看会怎样。"
)

COPY_OVERWRITTEN = (
    "我们想了很久，试过很多种办法，最后决定把它交到你手上，"
    "希望它能让你的每一天都轻松一点，也让你把时间留给真正更重要的事情。"
)
"""第二轮反思的产物：为了"更有冲击力"把产品名改没了——字数还对，内容却坏了。"""

COPY_CRITIQUE = (
    "这段只有 13 个字，离 80 字的要求差太远，也没有说清星尘能帮用户做什么。"
    "请补上具体用途和一句行动号召，字数控制在 60–120 之间。"
)

COPY_FALSE_CRITIQUE = (
    "第一句太平，建议直接用一个更有冲击力的开场，把重点放到用户收益上。"
)

# 批评"具体性"的判定规则：既要能**定位**到问题，也要能**核对**对错。
_LOCATOR = re.compile(r"n=\d+|第\s*\d+\s*行|sum_squares\(\d+\)|用例")
_EXPECTATION = re.compile(r"正确答案是\s*\d+|应该是\s*\d+|返回\s*\d+|应该是\s*\d+")


# ---------------------------------------------------------------- 确定性 checker


def score_sum_squares(code: str) -> float:
    """执行候选代码，按测试用例通过率打分（外部验证器）。

    这是本章的"外部验证"：一个完全不依赖模型判断的 checker。
    评分范围 [0, 1]，通过率就是分数——这样"哪一版更好"可以被测量，
    而不是靠感觉。

    教学环境里执行模型写的代码正是本意：脚本化模型的输出是固定字符串，
    没有外部输入。真实系统里这一步必须放进沙箱（第 13 章）。
    """
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "<candidate>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - 代码跑不起来本身就是一种失败
        return 0.0
    func = namespace.get("sum_squares")
    if not callable(func):
        return 0.0
    passed = 0
    for n, expected in SUM_SQUARES_CASES:
        try:
            got = func(n)
        except Exception:  # noqa: BLE001 - 单个用例崩了算这个用例没过
            continue
        if got == expected:
            passed += 1
    return passed / len(SUM_SQUARES_CASES)


def score_copy(text: str) -> float:
    """产品介绍的**弱**检查器：只检查两个机械性质。

    刻意做得很弱——开放式写作真正要的东西（说服力、语气）没法用确定性规则判。
    这恰恰是本章的论点：能机械判的才值得用 checker 判，剩下的部分反思也验不了。
    """
    checks = [
        COPY_TARGET_NAME in text,
        COPY_MIN_CHARS <= len(text) <= COPY_MAX_CHARS,
    ]
    return sum(1 for check in checks if check) / len(checks)


def critique_is_specific(text: str) -> bool:
    """判断一条批评是不是"点名了具体问题"。

    规则：批评里必须同时出现一个**可定位**的东西（行号、用例、具体输入值）
    和一个**可核对**的期望值。只说"不够好""逻辑不严谨""建议优化"的一律不合格。
    """
    return bool(_LOCATOR.search(text)) and bool(_EXPECTATION.search(text))


def copy_critique_is_actionable(text: str) -> bool:
    """开放任务的批评是否指向了**可机械检查**的性质（产品名、字数下限）。"""
    return COPY_TARGET_NAME in text or str(COPY_MIN_CHARS) in text


# ---------------------------------------------------------------- 反思循环


@dataclass(frozen=True)
class ReflectionConfig:
    """反思循环的旋钮。

    每个旋钮都对应一个真实的失败模式：

    - ``max_rounds``：没有上限，反思会一直改下去，烧钱而且越改越差；
    - ``min_rounds``：强制至少反思几轮——用来演示"批评本身可能是假的"；
    - ``patience``：连续几轮评分没提升就停，避免在同一个错上反复打转；
    - ``min_score``：达标线，够格就停，不必改到完美。
    """

    max_rounds: int = 2
    min_rounds: int = 0
    patience: int = 1
    min_score: float = 1.0


@dataclass
class ReflectionResult:
    """一次反思运行的结果。"""

    task: str
    best_text: str
    best_score: float
    last_text: str
    last_score: float
    rounds: int
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    scores: list[float] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    prompt_tokens_per_call: list[int] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        """这次运行的 token 估算——反思的真实成本。"""
        return self.prompt_tokens + self.completion_tokens


class Reflector:
    """反思循环：生成 → 批评 → 修订，可选地反复几轮。

    两个关键设计：

    1. **判定权交给 checker**：能不能停、哪一版最好，都由 ``checker`` 说了算。
       模型只负责"提意见"和"改"，不负责"打分"——这是本章最重要的一条。
    2. **永远返回最好的一版**：模型改完可能更差（假批评、空改、越改越偏），
       所以必须记住历史最高分的那一版。
    """

    def __init__(
        self,
        model: ChatModel,
        checker: Callable[[str], float],
        config: ReflectionConfig | None = None,
    ) -> None:
        self.model = model
        self.checker = checker
        self.config = config or ReflectionConfig()

    def run(self, task: str) -> ReflectionResult:
        """跑一次反思，返回最好的一版。"""
        messages: list[Message] = [system(SYSTEM_PROMPT), human(task)]
        prompt_tokens = 0
        completion_tokens = 0
        prompt_tokens_per_call: list[int] = []

        def ask(sent: Sequence[Message]) -> str:
            """调一次模型，并把这一次的上下文成本记下来。"""
            nonlocal prompt_tokens, completion_tokens
            prompt_tokens_per_call.append(sum(approx_tokens(m.content) for m in sent))
            reply = self.model.complete(list(sent))
            completion_tokens += approx_tokens(reply.content)
            return reply.content

        draft = ask(messages)
        messages.append(Message(role="assistant", content=draft))
        score = self.checker(draft)
        best_text, best_score = draft, score
        last_text, last_score = draft, score
        scores = [score]
        rounds = 0
        no_improve = 0
        stop_reason = "max_rounds" if self.config.max_rounds else "no_reflection"

        while rounds < self.config.max_rounds:
            # 达标就停——但 min_rounds 要求至少再走几轮时才允许停。
            if score >= self.config.min_score and rounds >= self.config.min_rounds:
                stop_reason = "passed"
                break

            critique_prompt = human(
                CRITIQUE_PROMPT.format(task=task, draft=last_text)
            )
            critique = ask(messages + [critique_prompt])
            messages.append(critique_prompt)
            messages.append(Message(role="assistant", content=critique))

            revise_prompt = human(REVISE_PROMPT.format(critique=critique))
            revision = ask(messages + [revise_prompt])
            messages.append(revise_prompt)
            messages.append(Message(role="assistant", content=revision))

            rounds += 1
            last_text = revision
            last_score = self.checker(revision)
            scores.append(last_score)

            if last_score > best_score:
                best_text, best_score = revision, last_score
                no_improve = 0
            else:
                no_improve += 1
            score = last_score

            if score >= self.config.min_score and rounds >= self.config.min_rounds:
                stop_reason = "passed"
                break
            if no_improve >= self.config.patience:
                stop_reason = "no_improvement"
                break

        return ReflectionResult(
            task=task,
            best_text=best_text,
            best_score=best_score,
            last_text=last_text,
            last_score=last_score,
            rounds=rounds,
            model_calls=1 + 2 * rounds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            scores=scores,
            messages=messages,
            prompt_tokens_per_call=prompt_tokens_per_call,
            stop_reason=stop_reason,
        )


def self_evaluate(model: ChatModel, answer: str) -> str:
    """让模型给自己的答案打一个"对/错"分——本章要证伪的做法。

    单独抽出来，是因为生产里它太常见了：加一句"再检查一下"，然后把模型的意见
    当成验收标准。它唯一的作用就是把不确定性从"答案对不对"搬到了
    "模型觉得自己对不对"，而后者没有任何外部依据。
    """
    reply = model.complete(
        [system(SYSTEM_PROMPT), human(SELF_EVAL_PROMPT.format(answer=answer))]
    )
    return reply.content


# ---------------------------------------------------------------- 1. 反思能改对


def check_reflection_fixes_draft() -> None:
    """初稿有错 → 批评点名 → 修订正确。"""
    section("反思一次能改对：初稿有错 → 批评点名 → 修订正确")

    draft_score = score_sum_squares(WRONG_CODE)
    ensure(
        draft_score < 1.0,
        f"期望初稿确实是错的（用例通过率 < 1），实际通过率 {draft_score}",
    )

    model = scripted_model([WRONG_CODE, TRUE_CRITIQUE, CORRECT_CODE])
    result = Reflector(
        model, score_sum_squares, ReflectionConfig(max_rounds=1)
    ).run(CODE_TASK)

    ensure(
        result.best_score == 1.0,
        f"期望修订后通过率 1.0，实际 {result.best_score}",
    )
    ensure(
        result.best_text == CORRECT_CODE,
        f"期望最终版本就是正确代码，实际 {result.best_text!r}",
    )
    ensure(result.rounds == 1, f"期望只反思 1 轮，实际 {result.rounds} 轮")

    # 批评必须"点名"——这是反思有没有用的分水岭
    ensure(
        critique_is_specific(TRUE_CRITIQUE),
        "期望这条批评包含可定位的位置和可核对的期望值，实际它只说了'不够好'这类空话",
    )
    ensure(
        not critique_is_specific(FALSE_CRITIQUE),
        "期望'逻辑不严谨'这种没有具体指向的批评被判为不合格，实际被判为合格",
    )

    # 批评必须真的被送进修订那一步
    revise_messages = [m.content for m in result.messages if m.role == "user"]
    ensure(
        any(TRUE_CRITIQUE in text for text in revise_messages),
        "期望批评意见被送进了修订请求，实际修订请求里没有它",
    )
    print(f"  初稿通过率 {draft_score:.2f} → 修订后 {result.best_score:.2f}")
    print(f"  批评：{TRUE_CRITIQUE[:36]}…（点名了行号、输入值、期望值）")


# ---------------------------------------------------------------- 2. 消息数增加


def check_message_growth() -> None:
    """反思比不反思多出批评与修订两轮往返。"""
    section("消息数：反思多出批评与修订两轮往返")

    plain = Reflector(
        scripted_model([WRONG_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=0),
    ).run(CODE_TASK)
    reflect = Reflector(
        scripted_model([WRONG_CODE, TRUE_CRITIQUE, CORRECT_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=1),
    ).run(CODE_TASK)

    plain_len = len(plain.messages)
    reflect_len = len(reflect.messages)
    ensure(
        reflect_len > plain_len,
        f"期望反思路径的消息更多，实际直出 {plain_len} 条、反思 {reflect_len} 条",
    )
    ensure(
        reflect_len == plain_len + 4,
        f"期望多出 4 条消息（批评请求 + 批评 + 修订请求 + 修订），"
        f"实际多出 {reflect_len - plain_len} 条",
    )
    ensure(
        plain.model_calls == 1 and reflect.model_calls == 3,
        f"期望直出 1 次模型调用、反思 3 次，实际 {plain.model_calls} / {reflect.model_calls}",
    )
    print(f"  直出 {plain_len} 条消息 / {plain.model_calls} 次调用")
    print(f"  反思 {reflect_len} 条消息 / {reflect.model_calls} 次调用（+{reflect_len - plain_len} 条）")


# ---------------------------------------------------------------- 3. 循环上限与最好一版


def check_loop_cap_and_best_version() -> None:
    """连续 N 轮无改进就停，并且返回最好的一版，不是最后一版。"""
    section("循环上限：无改进即停，而且返回最好的一版")

    model = scripted_model(
        [
            WRONG_CODE,
            TRUE_CRITIQUE,
            PARTIAL_CODE,
            PARTIAL_CRITIQUE,
            CORRECT_CODE,
            FALSE_CRITIQUE,
            WRONG_CODE,
        ]
    )
    result = Reflector(
        model,
        score_sum_squares,
        ReflectionConfig(max_rounds=3, min_rounds=3, patience=1),
    ).run(CODE_TASK)

    ensure(result.rounds == 3, f"期望反思了 3 轮，实际 {result.rounds} 轮")
    ensure(
        result.model_calls == 7,
        f"期望模型调用 1 + 2×3 = 7 次，实际 {result.model_calls} 次",
    )
    ensure(
        result.stop_reason == "no_improvement",
        f"期望因连续无改进而停止，实际停止原因 {result.stop_reason!r}",
    )
    ensure(
        result.best_score == 1.0,
        f"期望返回最好的一版（通过率 1.0），实际 {result.best_score}",
    )
    ensure(
        result.best_text == CORRECT_CODE,
        f"期望最好的一版是正确代码，实际 {result.best_text!r}",
    )
    ensure(
        result.last_score < result.best_score,
        f"期望最后一版比最好的一版更差（否则这个断言没意义），"
        f"实际最后一版 {result.last_score}、最好的一版 {result.best_score}",
    )
    ensure(
        result.best_text != result.last_text,
        "期望返回的不是最后一版",
    )
    ensure(
        result.scores
        == [
            score_sum_squares(WRONG_CODE),
            score_sum_squares(PARTIAL_CODE),
            score_sum_squares(CORRECT_CODE),
            score_sum_squares(WRONG_CODE),
        ],
        f"期望评分轨迹是[初稿, 改一半, 正确, 被改坏]，实际 {result.scores}",
    )
    print(f"  评分轨迹：{' → '.join(f'{s:.2f}' for s in result.scores)}")
    print(f"  停止原因 {result.stop_reason!r}；返回最好的一版（{result.best_score:.2f}），不是最后一版（{result.last_score:.2f}）")

    # 轮数上限本身：模型一直改不好时，循环必须自己停下来
    stuck = Reflector(
        scripted_model(
            [
                WRONG_CODE,
                TRUE_CRITIQUE,
                OFF_BY_ONE_CODE,
                PARTIAL_CRITIQUE,
                SQUARE_FORMULA_CODE,
            ]
        ),
        score_sum_squares,
        ReflectionConfig(max_rounds=2, patience=99),
    ).run(CODE_TASK)
    ensure(stuck.rounds == 2, f"期望在轮数上限 2 处停下，实际 {stuck.rounds} 轮")
    ensure(
        stuck.model_calls == 5,
        f"期望模型调用 1 + 2×2 = 5 次，实际 {stuck.model_calls} 次",
    )
    ensure(
        stuck.stop_reason == "max_rounds",
        f"期望因触到轮数上限而停止，实际停止原因 {stuck.stop_reason!r}",
    )
    ensure(
        stuck.best_text == WRONG_CODE,
        "期望没有任何一版比初稿更好时返回初稿，实际返回了别的版本",
    )
    print(f"  卡住时：{stuck.rounds} 轮后按上限停止，返回初稿（{stuck.best_score:.2f}）")


# ---------------------------------------------------------------- 4. 外部验证优先


def check_external_verification_wins() -> None:
    """模型自评说"没问题"，checker 说"有问题"——以 checker 为准。"""
    section("外部验证优先：模型自评说「没问题」，checker 说「有问题」")

    # 先看模型自评有多不靠谱
    self_eval_model = scripted_model([WRONG_CODE, "没问题，逻辑严谨，可以直接使用。"])
    draft = self_eval_model.complete(
        [system(SYSTEM_PROMPT), human(CODE_TASK)]
    ).content
    verdict = self_evaluate(self_eval_model, draft)
    checker_score = score_sum_squares(draft)

    ensure(
        "没问题" in verdict,
        f"期望模型自评说「没问题」，实际自评结论 {verdict!r}",
    )
    ensure(
        checker_score < 1.0,
        f"期望 checker 判定初稿有问题（通过率 < 1），实际通过率 {checker_score}",
    )
    print(f"  模型自评：{verdict!r}")
    print(f"  checker ：通过率 {checker_score:.2f} → 有问题")
    print("  → 两者结论冲突时，checker 是外部事实，模型自评只是模型的自我感觉")

    # 有 checker 的反思路径：运行时根本不该问模型"你觉得对吗"
    model = scripted_model([WRONG_CODE, TRUE_CRITIQUE, CORRECT_CODE])
    result = Reflector(
        model, score_sum_squares, ReflectionConfig(max_rounds=1)
    ).run(CODE_TASK)

    ensure(
        result.model_calls == 3,
        f"期望有 checker 时只用 3 次模型调用（生成 + 批评 + 修订），实际 {result.model_calls} 次",
    )
    ensure(
        not any(SELF_EVAL_MARK in m.content for m in result.messages),
        "期望运行时没有让模型给自己的答案打分",
    )
    ensure(
        result.best_score == 1.0,
        f"期望以 checker 为准后修对了，实际通过率 {result.best_score}",
    )
    ensure(
        result.best_score == score_sum_squares(result.best_text),
        "期望最终分数就是 checker 对最终版本的判定值",
    )
    print(f"  以 checker 为准：{result.model_calls} 次调用，无自评调用，修到通过率 {result.best_score:.2f}")


# ---------------------------------------------------------------- 5. 成本


def check_reflection_cost() -> None:
    """每次反思至少多出两次模型调用和更长的上下文。"""
    section("成本：反思至少多出两次模型调用和更长的上下文")

    plain = Reflector(
        scripted_model([WRONG_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=0),
    ).run(CODE_TASK)
    reflect = Reflector(
        scripted_model([WRONG_CODE, TRUE_CRITIQUE, CORRECT_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=1),
    ).run(CODE_TASK)

    ensure(
        reflect.model_calls == plain.model_calls + 2,
        f"期望反思多出 2 次模型调用（批评 + 修订），"
        f"实际直出 {plain.model_calls} 次、反思 {reflect.model_calls} 次",
    )
    ensure(
        reflect.total_tokens > plain.total_tokens,
        f"期望反思的 token 成本更高，实际直出 {plain.total_tokens}、反思 {reflect.total_tokens}",
    )
    ensure(
        len(reflect.prompt_tokens_per_call) == 3,
        f"期望记录了 3 次调用的上下文大小，实际 {len(reflect.prompt_tokens_per_call)} 条",
    )
    ensure(
        reflect.prompt_tokens_per_call[0]
        < reflect.prompt_tokens_per_call[1]
        < reflect.prompt_tokens_per_call[2],
        f"期望上下文逐轮变长，实际 {reflect.prompt_tokens_per_call}",
    )
    print(f"  直出：{plain.model_calls} 次调用 / {plain.total_tokens} token")
    print(f"  反思：{reflect.model_calls} 次调用 / {reflect.total_tokens} token"
          f"（{reflect.total_tokens / plain.total_tokens:.1f}×）")
    print(f"  每次调用的上下文：{reflect.prompt_tokens_per_call} token")


# ---------------------------------------------------------------- 6. 无效反思


def check_false_critique() -> None:
    """假批评把好稿改坏——"返回最好的一版"挡住了这次破坏。"""
    section("无效反思：假批评把好稿改坏")

    good_score = score_sum_squares(CORRECT_CODE)
    ensure(
        good_score == 1.0,
        f"期望初稿本来就是对的（通过率 1.0），实际 {good_score}",
    )
    ensure(
        not critique_is_specific(FALSE_CRITIQUE),
        "期望这条幻觉批评没有点名任何具体问题，实际被判为具体",
    )

    model = scripted_model([CORRECT_CODE, FALSE_CRITIQUE, WRONG_CODE])
    result = Reflector(
        model, score_sum_squares, ReflectionConfig(max_rounds=1, min_rounds=1)
    ).run(CODE_TASK)

    ensure(
        result.last_score < result.best_score,
        f"期望照着假批评改完之后更差，实际最后一版 {result.last_score}、"
        f"最好的一版 {result.best_score}",
    )
    ensure(
        result.best_score == 1.0,
        f"期望运行时返回最好的一版（通过率 1.0），实际 {result.best_score}",
    )
    ensure(
        result.best_text == CORRECT_CODE,
        f"期望返回的是初稿而不是被改坏的版本，实际 {result.best_text!r}",
    )
    print(f"  初稿通过率 {good_score:.2f}；照着假批评改完降到 {result.last_score:.2f}")
    print(f"  运行时返回最好的一版（{result.best_score:.2f}）——机制挡住了这次改坏")


# ---------------------------------------------------------------- 7. 开放任务


def check_open_task_reflection() -> None:
    """同样的机制跑在开放式写作上：成本更高，可测收益更低。"""
    section("开放任务：反思的额外成本大于可测收益")

    plain = Reflector(
        scripted_model([COPY_DRAFT]),
        score_copy,
        ReflectionConfig(max_rounds=0),
    ).run(COPY_TASK)
    one_round = Reflector(
        scripted_model([COPY_DRAFT, COPY_CRITIQUE, COPY_REVISED]),
        score_copy,
        ReflectionConfig(max_rounds=1, min_rounds=1),
    ).run(COPY_TASK)
    two_rounds = Reflector(
        scripted_model(
            [
                COPY_DRAFT,
                COPY_CRITIQUE,
                COPY_REVISED,
                COPY_FALSE_CRITIQUE,
                COPY_OVERWRITTEN,
            ]
        ),
        score_copy,
        ReflectionConfig(max_rounds=2, min_rounds=2, patience=2),
    ).run(COPY_TASK)

    ensure(
        plain.best_score < one_round.best_score,
        f"期望反思一轮后弱检查器给出的分数上升，实际 {plain.best_score} → {one_round.best_score}",
    )
    ensure(
        copy_critique_is_actionable(COPY_CRITIQUE),
        "期望第一轮批评指向了可机械检查的性质（产品名 / 字数），实际没有",
    )
    ensure(
        not copy_critique_is_actionable(COPY_FALSE_CRITIQUE),
        "期望第二轮的幻觉批评没有指向任何可检查的性质，实际它却指向了",
    )

    # 幻觉风险在开放任务上同样可测：第二轮把分数改回去了
    ensure(
        two_rounds.last_score < one_round.best_score,
        f"期望第二轮反思把分数改低，实际 {one_round.best_score} → {two_rounds.last_score}",
    )
    ensure(
        two_rounds.best_score == one_round.best_score,
        f"期望最好的一版机制保住了第一轮的成果，实际 {two_rounds.best_score}",
    )

    # 成本对比：同样的机制，在开放任务上要多花 token
    code_plain = Reflector(
        scripted_model([WRONG_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=0),
    ).run(CODE_TASK)
    code_reflect = Reflector(
        scripted_model([WRONG_CODE, TRUE_CRITIQUE, CORRECT_CODE]),
        score_sum_squares,
        ReflectionConfig(max_rounds=1),
    ).run(CODE_TASK)

    code_cost = code_reflect.total_tokens - code_plain.total_tokens
    copy_cost = one_round.total_tokens - plain.total_tokens
    code_gain = code_reflect.best_score - code_plain.best_score
    copy_gain = one_round.best_score - plain.best_score

    ensure(
        copy_cost > code_cost,
        f"期望开放任务的反思花费更多 token（散文批评更长），实际代码 {code_cost}、写作 {copy_cost}",
    )
    ensure(
        code_gain / code_cost > copy_gain / copy_cost,
        f"期望每 token 买到的可测分数在可验证任务上更高，"
        f"实际代码 {code_gain / code_cost:.5f}、写作 {copy_gain / copy_cost:.5f}",
    )
    print(f"  可验证任务：多花 {code_cost} token 换 +{code_gain:.2f} 分")
    print(f"  开放任务  ：多花 {copy_cost} token 换 +{copy_gain:.2f} 分")
    print(f"  开放任务第二轮：{one_round.best_score:.2f} → {two_rounds.last_score:.2f}（幻觉把分数改回去了）")


# ---------------------------------------------------------------- 入口


def run_live() -> None:
    """真实模型路径：让真模型自己批评自己，只校验形状不校验内容。"""
    banner("第 8 章 · 真实模型路径")
    from ..labkit import real_model

    model = real_model()
    reflector = Reflector(
        model, score_sum_squares, ReflectionConfig(max_rounds=1)
    )
    result = reflector.run(CODE_TASK)

    for index, message in enumerate(result.messages):
        print(f"  [{index}] {message.role}: {message.content[:120]!r}")

    ensure(bool(result.messages), "真实模型路径应当产生至少一条消息")
    ensure(result.model_calls >= 1, "真实模型路径至少要调一次模型")
    ensure(
        result.best_score >= 0.0,
        f"checker 应当给出 [0, 1] 的分数，实际 {result.best_score}",
    )
    print(f"  停止原因：{result.stop_reason}，通过率 {result.best_score:.2f}")


def run() -> None:
    """跑第 8 章实验。"""
    banner("第 8 章 · 反思与自纠错")
    check_reflection_fixes_draft()
    check_message_growth()
    check_loop_cap_and_best_version()
    check_external_verification_wins()
    check_reflection_cost()
    check_false_critique()
    check_open_task_reflection()
    rule(
        "反思 = 生成 → 批评 → 修订；有确定性 checker 就别让模型自评；"
        "永远返回最好的一版，别返回最后一版"
    )
    print("\n✓ 第 8 章通过")
