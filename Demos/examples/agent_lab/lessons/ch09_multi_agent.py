"""第 9 章实验：多智能体编排模式。

这个模块解决什么问题：第 7 章解决"先做什么后做什么"，第 8 章解决"发现错了怎么纠"，
第 9 章解决的是另一个维度——**要不要把一件事拆给多个 Agent，拆成什么形状**。

多 Agent 本质是**编排**问题：谁先谁后、信息怎么流、结果怎么合。
本章把五种最常见的拓扑各做成一个能跑的实现，并把"该不该拆"变成可执行的判据：

1. **Router**——分类器把任务派给合适的专家，断言走的是哪条分支；
2. **Parallel + 聚合**——三个 Agent 并行出结论，断言多数票与分歧处理；
3. **Sequential**——A 的产出进 B 的输入，断言 B 确实看到了；
4. **Supervisor**——主管分派、执行、收口，断言调用顺序；
5. **共享状态 vs 隔离上下文**——断言产出汇入同一状态，也断言污染与搬运成本；
6. **拆分判据**——``should_split`` 给出"拆 / 不拆"的判定；
7. **成本与延迟**——并行比串行快，但也比串行贵（并行让每个 Agent 重复读同一份材料）。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..labkit import (
    ChatModel,
    ModelReply,
    ScriptedModel,
    ToolCall,
    approx_tokens,
    banner,
    ensure,
    human,
    rule,
    scripted_model,
    section,
    system,
    tool,
)
from ..runtime import Agent, AgentConfig, AgentResult

# ---------------------------------------------------------------- 提示词常量

ROUTER_PROMPT = (
    "你是任务分派器。根据用户请求，从给定的专家里挑一个最合适的，"
    "只输出专家名字，不要解释。"
)

SUPERVISOR_PROMPT = (
    "你是主管 Agent。你的职责是：先指派一位专家去干活，"
    "拿到专家产出之后，再给出最终答复。"
)

EXPERI_PROMPT = "你是{role}。只回答与你职责有关的问题。"


# ---------------------------------------------------------------- 工具


@tool
def 查询订单(order_id: str) -> str:
    """按订单号查询订单状态。

    Args:
        order_id: 订单号，形如 A-1001
    """
    return {"A-1001": "已发货", "A-1002": "待付款", "A-1003": "已签收"}.get(order_id, "未找到")


@tool
def 发起退款(order_id: str, 金额: float) -> str:
    """按订单号发起退款。

    Args:
        order_id: 要退款的订单号
        金额: 退款金额，单位元
    """
    return f"已为订单 {order_id} 提交退款 {金额} 元"


# ---------------------------------------------------------------- 轨迹


@dataclass(frozen=True)
class TraceEvent:
    """轨迹上的一个事件。"""

    node: str
    """谁干的（supervisor / 某专家 / router）。"""

    action: str
    """干了什么。"""

    parent: str | None = None
    """是谁把谁叫起来的——多 Agent 的轨迹是树，这一位就是树的边。"""


@dataclass
class Trace:
    """一次编排留下的轨迹。

    为什么不用 ``AgentResult.steps``：那是单个 Agent 的**一条线**，
    多 Agent 的轨迹是**一棵树**。父子关系必须单独记下来，
    否则第 11 章讲回放时会发现"谁把谁叫起来的"已经丢了。
    """

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, node: str, action: str, parent: str | None = None) -> None:
        """记一个事件。"""
        self.events.append(TraceEvent(node=node, action=action, parent=parent))

    @property
    def labels(self) -> list[str]:
        """按发生顺序拼出的 ``节点:动作`` 列表。"""
        return [f"{e.node}:{e.action}" for e in self.events]

    @property
    def branches(self) -> int:
        """分支数：有多少个节点把别人叫了起来。"""
        return len({e.parent for e in self.events if e.parent})


def _match_name(text: str, names: Sequence[str]) -> str | None:
    """在文本里找出最长的那个匹配名。

    为什么取最长匹配：名字之间可能互为子串（"订单专家" 和 "订单专家组长"），
    先匹配到短的那个就会选错人。
    """
    for name in sorted(names, key=len, reverse=True):
        if name in text:
            return name
    return None


# ---------------------------------------------------------------- Router


@dataclass
class Expert:
    """注册在路由表里的专家 Agent。

    刻意只记 ``calls`` 和轨迹位置：多 Agent 里最常被问到的问题是
    "到底是谁做的、做了几次"，而不是"它内部的循环长什么样"。
    """

    name: str
    agent: Agent
    trace: Trace
    calls: int = 0

    def run(self, task: str) -> AgentResult:
        """执行一次任务，返回完整结果（含工具调用，便于断言）。"""
        self.calls += 1
        self.trace.record(self.name, "执行")
        return self.agent.run(task)


@dataclass
class Classifier:
    """Router 里的分类器：看一眼任务，决定派给谁。

    它是整个系统里**唯一一个需要看全部输入**的角色——
    这就是 Router 拓扑的隐含代价：分类器选错了，下游全错，
    而且错误发生在最前面、最不容易被察觉的地方。
    """

    model: ChatModel
    trace: Trace
    calls: int = 0

    def route(
        self,
        task: str,
        registry: dict[str, Expert],
        fallback: str = "",
    ) -> str:
        """返回被选中的专家名。

        ``fallback`` 非空时，分类器没给出可识别的专家就走兜底——
        真实模型路径需要它，脚本路径则让"识别失败"直接炸出来。
        """
        names = list(registry)
        prompt = (
            f"任务：{task}\n"
            f"可选专家：{'、'.join(names)}\n"
            "只输出专家名字。"
        )
        reply = self.model.complete([system(ROUTER_PROMPT), human(prompt)])
        self.calls += 1

        chosen = _match_name(reply.content, names)
        if chosen is None:
            if not fallback:
                raise ValueError(f"分类器没有给出可识别的专家：{reply.content!r}")
            self.trace.record("router", "无法识别→兜底")
            return fallback
        self.trace.record("router", f"分派→{chosen}")
        return chosen


def _build_experts(trace: Trace) -> tuple[dict[str, Expert], ScriptedModel, ScriptedModel]:
    """造两个工具集完全不同的专家。

    工具集不同是多 Agent 最硬的一条理由：订单专家要读订单库，
    退款专家要写退款单——同一套工具干不了这两件事。
    """
    order_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("c1", "查询订单", {"order_id": "A-1001"})]),
            ModelReply(content="订单 A-1001 已发货"),
        ]
    )
    refund_model = scripted_model(
        [
            ModelReply(
                tool_calls=[
                    ToolCall("c2", "发起退款", {"order_id": "A-1002", "金额": 88})
                ]
            ),
            ModelReply(content="已为订单 A-1002 提交退款 88 元"),
        ]
    )
    registry = {
        "订单专家": Expert(
            name="订单专家",
            agent=Agent(
                order_model,
                [查询订单],
                AgentConfig(system_prompt=EXPERI_PROMPT.format(role="订单专家")),
            ),
            trace=trace,
        ),
        "退款专家": Expert(
            name="退款专家",
            agent=Agent(
                refund_model,
                [发起退款],
                AgentConfig(system_prompt=EXPERI_PROMPT.format(role="退款专家")),
            ),
            trace=trace,
        ),
    }
    return registry, order_model, refund_model


def check_router() -> None:
    """Router：一个分类器把任务派给合适的专家。"""
    section("Router：分类器决定谁来做")

    trace = Trace()
    registry, order_model, refund_model = _build_experts(trace)

    # 第一类请求：查订单 → 应当走订单专家
    router = Classifier(model=scripted_model(["订单专家"]), trace=trace)
    task = "帮我查一下订单 A-1001 到哪了"
    chosen = router.route(task, registry)
    ensure(
        chosen == "订单专家",
        f"查订单的请求应当派给订单专家，实际派给了 {chosen!r}",
    )

    result = registry[chosen].run(task)
    ensure(
        result.final_answer == "订单 A-1001 已发货",
        f"订单专家应当给出查询结果，实际 {result.final_answer!r}",
    )
    ensure(
        [c.name for c in result.tool_calls] == ["查询订单"],
        f"订单专家只应调用 查询订单，实际调用了 {[c.name for c in result.tool_calls]}",
    )
    ensure(
        registry["退款专家"].calls == 0,
        f"未被选中的退款专家不该被执行，实际执行了 {registry['退款专家'].calls} 次",
    )
    ensure(
        len(refund_model.calls) == 0,
        f"未被选中的专家的模型一次都不该被调用，实际被调了 {len(refund_model.calls)} 次",
    )
    ensure(
        trace.labels == ["router:分派→订单专家", "订单专家:执行"],
        f"轨迹应当是 分类→执行 两步，实际 {trace.labels}",
    )
    print(f"  「{task}」 → {chosen}（工具：{[c.name for c in result.tool_calls]}）")

    # 第二类请求：退款 → 应当走退款专家，且不会碰订单专家
    router2 = Classifier(model=scripted_model(["退款专家"]), trace=trace)
    refund_task = "订单 A-1002 我不要了，请退款 88 元"
    chosen2 = router2.route(refund_task, registry)
    ensure(
        chosen2 == "退款专家",
        f"退款的请求应当派给退款专家，实际派给了 {chosen2!r}",
    )
    result2 = registry[chosen2].run(refund_task)
    ensure(
        [c.name for c in result2.tool_calls] == ["发起退款"],
        f"退款专家只应调用 发起退款，实际调用了 {[c.name for c in result2.tool_calls]}",
    )
    ensure(
        registry["订单专家"].calls == 1,
        f"第二次路由不该再动订单专家，实际它累计执行了 {registry['订单专家'].calls} 次",
    )
    ensure(
        len(order_model.calls) == 2,
        f"订单模型应当只被第一次请求调用 2 轮（调工具 + 作答），实际 {len(order_model.calls)} 轮",
    )
    print(f"  「{refund_task}」 → {chosen2}（工具：{[c.name for c in result2.tool_calls]}）")
    print("  → 路由的价值在于：每个专家只带自己那套工具，不用互相迁就")


# ---------------------------------------------------------------- Parallel + 聚合


@dataclass(frozen=True)
class VoteResult:
    """一次投票聚合的结果。

    ``agreement`` 把"3/3 一致"和"2/3 一致"分开——这是多 Agent 里
    最容易被糊弄过去的地方：**多数票也是有分歧**，
    反对票必须原样留着，不能当成一致通过。
    """

    decision: str
    """最终结论；没有结论时是空串（弃权）。"""

    votes: tuple[str, ...]
    """原始票，必须保留。"""

    agreement: str
    """unanimous（全体一致）/ majority（有严格多数）/ none（没有多数）。"""

    contested: bool
    """是否存在反对票——有分歧就要让人知道，而不是悄悄通过。"""

    escalation: str
    """没有多数时的处置：tiebreaker（请仲裁者裁决）/ abstain（弃权）/ 空串（不需要）。"""


def aggregate_votes(
    votes: Sequence[str],
    tiebreaker: Callable[[Sequence[str]], str] | None = None,
) -> VoteResult:
    """把多个 Agent 的结论合成一个。

    分三档，标准是"能确定多少"：

    - 全体一致 → 直接采信；
    - 有严格多数 → 采信，但标记成有分歧；
    - 没有多数 → 要么请仲裁者裁决，要么弃权，把决定权交回给人。
    """
    if not votes:
        return VoteResult("", (), "none", False, "abstain")

    counts = Counter(votes)
    top, top_n = counts.most_common(1)[0]

    if top_n == len(votes):
        return VoteResult(top, tuple(votes), "unanimous", False, "")
    if top_n * 2 > len(votes):
        return VoteResult(top, tuple(votes), "majority", True, "")
    if tiebreaker is not None:
        return VoteResult(tiebreaker(votes), tuple(votes), "none", True, "tiebreaker")
    return VoteResult("", tuple(votes), "none", True, "abstain")


def check_parallel_vote() -> None:
    """Parallel + 聚合：多个 Agent 各出结论，取多数。"""
    section("Parallel + 聚合：三个 Agent 并行出结论")

    question = "这份合同条款有没有法律风险？"
    specs = [("审查员甲", "同意"), ("审查员乙", "同意"), ("审查员丙", "反对")]

    agents: list[tuple[str, Agent, ScriptedModel]] = []
    for name, answer in specs:
        model = scripted_model([ModelReply(content=answer)])
        agent = Agent(model, [], AgentConfig(system_prompt=EXPERI_PROMPT.format(role=name)))
        agents.append((name, agent, model))

    # 三个 Agent 互不依赖，可以同时跑
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        futures = [pool.submit(agent.run, question) for _, agent, _ in agents]
        results = [future.result() for future in futures]

    votes = tuple(result.final_answer for result in results)
    ensure(
        votes == ("同意", "同意", "反对"),
        f"三个 Agent 的结论应当是 (同意, 同意, 反对)，实际 {votes}",
    )
    ensure(
        sum(len(model.calls) for _, _, model in agents) == 3,
        "三个 Agent 各调用模型一次，合计应当是 3 次",
    )

    vote = aggregate_votes(votes)
    ensure(vote.decision == "同意", f"多数票应当是「同意」，实际 {vote.decision!r}")
    ensure(
        vote.agreement == "majority",
        f"2/3 一致应当标记为 majority，实际 {vote.agreement!r}",
    )
    ensure(
        vote.contested,
        "2/3 一致属于有分歧，必须标记 contested，不能当成一致通过",
    )
    ensure(
        "反对" in vote.votes,
        f"反对票必须原样保留，实际保留的是 {vote.votes}",
    )
    ensure(
        vote.escalation == "",
        f"有严格多数时不该升级处理，实际 escalation={vote.escalation!r}",
    )
    print(f"  票型 {votes} → 结论「{vote.decision}」（{vote.agreement}，有分歧={vote.contested}）")

    # 全体一致：不该被标成有分歧
    unanimous = aggregate_votes(("同意", "同意", "同意"))
    ensure(
        unanimous.agreement == "unanimous" and not unanimous.contested,
        f"3/3 一致应当是 unanimous 且无分歧，实际 {unanimous.agreement}/{unanimous.contested}",
    )

    # 没有多数：弃权，或者请仲裁者裁决
    split = aggregate_votes(("同意", "反对", "弃权"))
    ensure(
        split.agreement == "none" and split.escalation == "abstain" and split.decision == "",
        f"三方各执一词且没有仲裁者时应当弃权，实际 {split.agreement}/{split.escalation}/{split.decision!r}",
    )
    arbitrated = aggregate_votes(("同意", "反对", "弃权"), tiebreaker=lambda _: "提交人工复核")
    ensure(
        arbitrated.escalation == "tiebreaker" and arbitrated.decision == "提交人工复核",
        f"没有多数时应当交给仲裁者裁决，实际 {arbitrated.escalation}/{arbitrated.decision!r}",
    )
    print(f"  票型 ('同意', '反对', '弃权') → 无多数 → {arbitrated.escalation} → 「{arbitrated.decision}」")


# ---------------------------------------------------------------- Sequential


def check_sequential() -> None:
    """Sequential：A 的产出是 B 的输入。"""
    section("Sequential：A 的产出进 B 的输入")

    trace = Trace()
    raw = "客户来信：我的订单还没收到，我要退款。"
    extractor_model = scripted_model(
        [ModelReply(content="订单号=A-1001 | 诉求=退款 | 客户等级=VIP")]
    )
    writer_model = scripted_model(
        [ModelReply(content="已受理，VIP 客户的退款将在 1 个工作日内到账。")]
    )

    extractor = Agent(extractor_model, [], AgentConfig(system_prompt="你是信息抽取器。"))
    writer = Agent(writer_model, [], AgentConfig(system_prompt="你是客服回信撰写员。"))

    trace.record("抽取员", "执行")
    extracted = extractor.run(raw).final_answer
    ensure(
        "A-1001" in extracted and "VIP" in extracted,
        f"抽取员应当抽出订单号和客户等级，实际 {extracted!r}",
    )

    # 显式搬运：A 的产出作为 B 的历史消息传下去
    trace.record("撰写员", "执行")
    result = writer.run(
        f"客户来信：{raw}\n请写一封回信。",
        history=[human(f"已抽取的信息：{extracted}")],
    )

    # 关键断言：B 的模型确实看到了 A 的产出
    seen = " ".join(message.content for message in writer_model.calls[0])
    ensure(
        "VIP" in seen,
        f"撰写员应当看到抽取员产出的「客户等级=VIP」，实际它看到的上下文是：{seen!r}",
    )
    ensure(
        "VIP" not in raw,
        "原始来信里没有 VIP，这个信息只可能来自抽取员——否则上面那条断言就是假阳性",
    )
    ensure(
        result.final_answer.startswith("已受理"),
        f"撰写员的最终回答不对：{result.final_answer!r}",
    )
    ensure(
        trace.branches == 0,
        f"流水线的轨迹是一条线，不该有分支，实际分支数 {trace.branches}",
    )
    print(f"  抽取员 → {extracted}")
    print(f"  撰写员 → {result.final_answer}")
    print("  → 关键不是「串起来了」，而是 B 真的看到了 A 的产出")


# ---------------------------------------------------------------- Supervisor


@dataclass
class Worker:
    """主管手下的执行者。"""

    name: str
    agent: Agent
    trace: Trace
    calls: int = 0

    def run(self, task: str) -> str:
        """执行任务，返回产出文本。"""
        self.calls += 1
        self.trace.record(self.name, "执行", parent="supervisor")
        return self.agent.run(task).final_answer


@dataclass
class Supervisor:
    """主管：决定派谁、收结果、最后收口。

    和 Router 的区别只有一处，但很关键：**主管不只做一次分类**，
    它在执行完还要再被叫醒一次收口。代价是多一次模型调用，
    收益是全局视角——只有主管看得到所有专家的产出。
    """

    model: ScriptedModel
    workers: dict[str, Worker]
    trace: Trace
    calls: int = 0

    def run(self, task: str) -> str:
        """跑完一轮：分派 → 执行 → 收口。"""
        names = list(self.workers)

        # 1. 分派：主管的"派谁去"记成一次结构化的工具调用
        reply = self.model.complete(
            [
                system(SUPERVISOR_PROMPT),
                human(
                    f"任务：{task}\n"
                    f"可用专家：{'、'.join(names)}\n"
                    "请指派一位专家开始工作。"
                ),
            ]
        )
        self.calls += 1
        chosen = self._parse_choice(reply, names)
        self.trace.record("supervisor", f"分派→{chosen}")

        # 2. 执行：子 Agent 干活
        output = self.workers[chosen].run(task)

        # 3. 收口：主管再看一次所有产出，给最终答复
        final = self.model.complete(
            [
                system(SUPERVISOR_PROMPT),
                human(f"任务：{task}\n专家产出：\n{output}\n请给出最终答复。"),
            ]
        )
        self.calls += 1
        self.trace.record("supervisor", "收口")
        return final.content

    def _parse_choice(self, reply: ModelReply, names: Sequence[str]) -> str:
        """从主管的回复里取出它选中的专家。

        主管的"分派"是一次工具调用——把"我要谁去做"记成结构化事件，
        而不是从一段自然语言里猜。这跟第 3 章"输出要可校验"是同一条纪律。
        """
        for call in reply.tool_calls:
            if call.name == "指派专家":
                candidate = str(call.arguments.get("专家名", ""))
                if candidate in self.workers:
                    return candidate
        matched = _match_name(reply.content, names)
        if matched is None:
            raise ValueError(f"主管没有给出可识别的专家：{reply.content!r}")
        return matched


def check_supervisor() -> None:
    """Supervisor：主管分派 → 子 Agent 执行 → 主管收口。"""
    section("Supervisor：主管分派 → 执行 → 收口")

    trace = Trace()
    supervisor_model = scripted_model(
        [
            ModelReply(tool_calls=[ToolCall("s1", "指派专家", {"专家名": "研究员"})]),
            ModelReply(content="结论：合同条款存在法律风险，建议修改违约责任那一条。"),
        ]
    )
    researcher_model = scripted_model(
        [ModelReply(content="研究员结论：违约责任条款对客户过于苛刻。")]
    )

    worker = Worker(
        name="研究员",
        agent=Agent(
            researcher_model,
            [],
            AgentConfig(system_prompt=EXPERI_PROMPT.format(role="研究员")),
        ),
        trace=trace,
    )
    supervisor = Supervisor(
        model=supervisor_model,
        workers={"研究员": worker},
        trace=trace,
    )

    final = supervisor.run("审一下这份合同有没有法律风险")
    ensure(
        trace.labels == ["supervisor:分派→研究员", "研究员:执行", "supervisor:收口"],
        f"调用顺序应当是 分派→执行→收口，实际 {trace.labels}",
    )
    ensure(
        supervisor.calls == 2,
        f"主管应当被调用两次（分派 + 收口），实际 {supervisor.calls} 次",
    )
    ensure(
        worker.calls == 1,
        f"子 Agent 应当只执行一次，实际 {worker.calls} 次",
    )
    ensure(
        len(supervisor_model.calls) == 2,
        f"主管的模型应当有两轮调用记录，实际 {len(supervisor_model.calls)} 轮",
    )

    # 收口时主管必须真的看到了子 Agent 的产出
    seen = " ".join(message.content for message in supervisor_model.calls[1])
    ensure(
        "过于苛刻" in seen,
        f"收口时主管应当看到研究员的产出，实际它看到的是：{seen!r}",
    )
    ensure(
        final.startswith("结论："),
        f"主管的最终答复不对：{final!r}",
    )
    ensure(
        trace.branches == 1,
        f"主管拓扑的轨迹是一棵树（主管把专家叫了起来），分支数应当是 1，实际 {trace.branches}",
    )
    print(f"  轨迹：{' → '.join(trace.labels)}")
    print(f"  最终答复：{final}")
    print("  → 主管是全局视角的代价：每轮多一次模型调用")


# ---------------------------------------------------------------- 共享状态


@dataclass
class SharedState:
    """多个 Agent 共享的状态。

    ``facts`` 是**要的结果**（谁发现了什么），``scratch`` 是**过程中的中间产物**。
    把两者放在一起是故意的：共享状态最大的坑就是
    下游 Agent 看见了上游的中间过程，而这些过程**本来不该给它看**。
    """

    facts: dict[str, str] = field(default_factory=dict)
    scratch: list[str] = field(default_factory=list)

    def render(self) -> str:
        """把当前状态渲染成给 Agent 看的文本。"""
        lines = [f"[{key}] {value}" for key, value in self.facts.items()]
        lines += [f"(过程) {item}" for item in self.scratch]
        return "\n".join(lines)

    @property
    def implicit_chars(self) -> int:
        """共享状态里"过程"部分的字符数——共享带来的隐式携带量。"""
        return sum(len(item) for item in self.scratch)


@dataclass
class FactCollector:
    """查一条事实的小 Agent。

    它把**结果**写进状态，也会把自己的**中间过程**写进去——
    后者是有意的：共享状态最大的坑就是下游看见了上游的中间过程。
    """

    name: str
    model: ScriptedModel
    trace: Trace
    seen: list[str] = field(default_factory=list)

    def run(self, task: str, context: str, state: SharedState) -> str:
        """在给定上下文下干活，把结果和过程都写进状态。"""
        prompt = f"任务：{task}\n" + (f"当前已知状态：\n{context}\n" if context else "")
        self.seen.append(prompt)
        answer = self.model.complete([human(prompt)]).content
        state.facts[self.name] = answer
        state.scratch.append(f"{self.name} 的中间过程：{answer}")
        self.trace.record(self.name, "执行")
        return answer


def _make_collectors(trace: Trace) -> list[FactCollector]:
    """造三个会往状态里写东西的小 Agent。"""
    specs = [
        ("检索员", "库存还有 42 件"),
        ("核对员", "政策允许 7 天内无理由退款"),
        ("汇总员", "客户可退款且库存充足"),
    ]
    collectors: list[FactCollector] = []
    for name, answer in specs:
        model = scripted_model([ModelReply(content=answer)])
        collectors.append(FactCollector(name=name, model=model, trace=trace))
    return collectors


def check_shared_state() -> None:
    """共享状态 vs 隔离上下文：产出都要拿到，代价各不同。"""
    section("共享状态 vs 隔离上下文")

    task = "核对订单 A-1001 能不能退"
    trace = Trace()

    # ---- 共享模式：所有 Agent 看得见同一份状态
    shared_state = SharedState()
    shared_collectors = _make_collectors(trace)
    for collector in shared_collectors:
        collector.run(task, shared_state.render(), shared_state)

    ensure(
        len(shared_state.facts) == 3,
        f"共享模式下最终状态里应当有 3 个子 Agent 的产出，实际 {len(shared_state.facts)} 个："
        f"{list(shared_state.facts)}",
    )
    ensure(
        shared_state.facts.get("检索员") == "库存还有 42 件",
        f"共享状态里应当能取到检索员的产出，实际 {shared_state.facts.get('检索员')!r}",
    )
    ensure(
        shared_state.facts.get("汇总员") == "客户可退款且库存充足",
        f"共享状态里应当能取到汇总员的产出，实际 {shared_state.facts.get('汇总员')!r}",
    )

    polluted = [c.name for c in shared_collectors if any("中间过程" in s for s in c.seen)]
    ensure(
        "核对员" in polluted and "汇总员" in polluted,
        f"共享状态下，下游 Agent 应当看见上游的中间过程（污染），实际被污染的有 {polluted}",
    )
    ensure(
        shared_state.implicit_chars > 0,
        "共享状态把上游的中间过程一并带了下去，隐式携带量应当大于 0",
    )
    print(f"  共享模式：产出 {len(shared_state.facts)} 份 ✓，隐式携带 {shared_state.implicit_chars} 字符，污染 {len(polluted)} 人")

    # ---- 隔离模式：每个 Agent 有自己的上下文，聚合器只搬运"结果"
    isolated_trace = Trace()
    isolated_collectors = _make_collectors(isolated_trace)
    final_state = SharedState()
    for collector in isolated_collectors:
        private = SharedState()
        collector.run(task, "", private)
        # 显式搬运：只搬结果字段，不搬中间过程
        final_state.facts.update(private.facts)

    ensure(
        len(final_state.facts) == 3,
        f"隔离模式下最终状态里同样应当有 3 个子 Agent 的产出，实际 {len(final_state.facts)} 个",
    )
    ensure(
        final_state.facts.get("汇总员") == "客户可退款且库存充足",
        f"隔离模式下最终状态里应当能取到汇总员的产出，实际 {final_state.facts.get('汇总员')!r}",
    )
    ensure(
        not final_state.scratch,
        f"隔离模式下不该把别人的中间过程搬进最终状态，实际搬进来 {final_state.scratch}",
    )
    isolated_polluted = [
        c.name for c in isolated_collectors if any("中间过程" in s for s in c.seen)
    ]
    ensure(
        not isolated_polluted,
        f"隔离模式下不该有 Agent 看见别人的中间过程，实际 {isolated_polluted}",
    )
    carry_chars = sum(len(value) for value in final_state.facts.values())
    ensure(
        carry_chars > 0,
        "隔离模式虽然干净，但搬运结果本身也要花钱，搬运量应当大于 0",
    )
    print(f"  隔离模式：产出 {len(final_state.facts)} 份 ✓，显式搬运 {carry_chars} 字符，污染 0 人")
    print("  → 共享省搬运但会污染；隔离干净但搬运本身就是成本")


# ---------------------------------------------------------------- 拆分判据

TOOL_DOMAIN: dict[str, str] = {
    "查询订单": "订单域",
    "查询物流": "订单域",
    "发起退款": "退款域",
    "查询库存": "库存域",
    "读取文档": "文档域",
    "撰写回复": "文档域",
}
"""工具 → 它操作的上下文域。

"需要的上下文不同"没法直接问模型，得落到能判断的东西上：
两个 Agent 读写的**数据域**不同，它们的上下文就不可能共享。
同一个域里的工具（查订单 vs 查物流）拆开没有意义。
"""


def _domain_of(tools: Iterable[str]) -> frozenset[str]:
    """一组工具对应的上下文域集合。"""
    return frozenset(TOOL_DOMAIN.get(name, name) for name in tools)


@dataclass(frozen=True)
class SplitVerdict:
    """拆不拆的判定结果。"""

    split: bool
    different_tools: bool
    different_context: bool
    different_criteria: bool
    reasons: tuple[str, ...]


def should_split(
    tools_a: Iterable[str],
    tools_b: Iterable[str],
    criteria_a: Iterable[str],
    criteria_b: Iterable[str],
) -> SplitVerdict:
    """判断两个 Agent 该不该拆开。

    三条判据**同时成立**才值得拆：工具集不同 / 需要的上下文不同 / 评价标准不同。
    少一条就别拆——每多一个 Agent 就多一次模型调用、多一份上下文、
    多一处可能出错的地方，而收益并没有兑现。

    这是一个刻意保守的判据：多 Agent 的收益是真的，
    但协调成本涨得比想象快，所以默认值应该是"不拆"。
    """
    tools_a_set = set(tools_a)
    tools_b_set = set(tools_b)
    criteria_a_set = set(criteria_a)
    criteria_b_set = set(criteria_b)

    different_tools = tools_a_set != tools_b_set
    different_context = _domain_of(tools_a_set) != _domain_of(tools_b_set)
    different_criteria = criteria_a_set != criteria_b_set

    reasons: list[str] = []
    if different_tools:
        reasons.append("工具集不同")
    if different_context:
        reasons.append("上下文域不同")
    if different_criteria:
        reasons.append("评价标准不同")

    split = different_tools and different_context and different_criteria
    if not split:
        missing = [
            label
            for label, ok in (
                ("工具集不同", different_tools),
                ("上下文域不同", different_context),
                ("评价标准不同", different_criteria),
            )
            if not ok
        ]
        reasons.append("缺少：" + "、".join(missing))

    return SplitVerdict(
        split=split,
        different_tools=different_tools,
        different_context=different_context,
        different_criteria=different_criteria,
        reasons=tuple(reasons),
    )


def check_split_criteria() -> None:
    """拆分判据：三条同时成立才拆。"""
    section("拆分判据：该不该拆")

    # 案例 1（反面）：工具集相同 + 评价标准相同 → 不拆
    same = should_split(
        ["查询订单"],
        ["查询订单"],
        ["事实准确"],
        ["事实准确"],
    )
    ensure(
        not same.split,
        f"工具集相同且评价标准相同时应当判定为「不拆」，实际判定为 {same.split}",
    )
    ensure(
        not same.different_tools and not same.different_context and not same.different_criteria,
        f"案例 1 的三条判据都应当不成立，实际 tools={same.different_tools} "
        f"context={same.different_context} criteria={same.different_criteria}",
    )
    print(f"  同样工具 + 同样标准 → 不拆（{'；'.join(same.reasons)}）")

    # 案例 2（正面）：工具集不同 + 上下文域不同 + 评价标准不同 → 拆
    yes = should_split(
        ["查询订单"],
        ["发起退款"],
        ["事实准确"],
        ["事实准确", "合规"],
    )
    ensure(
        yes.split,
        f"三条判据同时成立时应当判定为「拆」，实际判定为 {yes.split}（{'；'.join(yes.reasons)}）",
    )
    ensure(
        yes.different_tools and yes.different_context and yes.different_criteria,
        "案例 2 的三条判据都应当成立",
    )
    print(f"  不同工具 + 不同域 + 不同标准 → 拆（{'；'.join(yes.reasons)}）")

    # 案例 3（容易搞错）：工具不同但同域 → 还是不拆
    same_domain = should_split(
        ["查询订单"],
        ["查询物流"],
        ["事实准确"],
        ["时效性"],
    )
    ensure(
        not same_domain.split,
        f"两个工具在同一个上下文域时应当判定为「不拆」，实际判定为 {same_domain.split}",
    )
    ensure(
        same_domain.different_tools and not same_domain.different_context,
        f"案例 3 应当是「工具不同但上下文同域」，实际 tools={same_domain.different_tools} "
        f"context={same_domain.different_context}",
    )
    print(f"  工具不同但同域 → 不拆（{'；'.join(same_domain.reasons)}）")

    # 案例 4：评价标准相同，工具再多也不拆
    same_criteria = should_split(
        ["查询订单"],
        ["发起退款"],
        ["事实准确"],
        ["事实准确"],
    )
    ensure(
        not same_criteria.split,
        f"评价标准相同时应当判定为「不拆」，实际判定为 {same_criteria.split}",
    )
    print("  → 判据是「与」不是「或」：少一条就退回单 Agent")


# ---------------------------------------------------------------- 成本与延迟

_MEMO_LINE = "客户 A-1001 于上周三下单，商品含退款政策争议条款，客服记录显示已两次沟通。"
SOURCE_DOC = "\n".join(f"{index + 1}. {_MEMO_LINE}" for index in range(30))
"""一份大材料：每个并行 Agent 都要重读一遍它，串行时只读一次。"""


@dataclass
class TimedAgent:
    """带固定延迟的模型包装。

    为什么要真的 sleep：并行"更快"这件事只有用墙钟才量得出来。
    纯逻辑的调用计数只能证明"并行了"，证不了"并行更省时间"。
    """

    name: str
    model: ScriptedModel
    delay: float
    calls: int = 0

    def run(self, task: str, context: str = "") -> tuple[str, int]:
        """跑一次，返回（产出, 本次 prompt 的估算 token 数）。"""
        prompt = f"任务：{task}\n材料：\n{context}" if context else f"任务：{task}"
        time.sleep(self.delay)
        reply = self.model.complete([human(prompt)])
        self.calls += 1
        return reply.content, approx_tokens(prompt)


def check_cost_and_latency() -> None:
    """并行 vs 串行：并行更快，但并行更贵。"""
    section("成本与延迟：并行买到延迟，付出重复读")

    task = "根据材料给出一条处理建议"
    delay = 0.05

    # ---- 串行：每一环只把上一环的产出传下去
    serial_agents = [
        TimedAgent("摘要员", scripted_model([ModelReply(content="摘要：订单存在退款争议。")]), delay),
        TimedAgent("撰写员", scripted_model([ModelReply(content="报告：建议人工复核退款。")]), delay),
        TimedAgent("审校员", scripted_model([ModelReply(content="终稿：建议 24 小时内人工复核。")]), delay),
    ]
    serial_started = time.perf_counter()
    material = SOURCE_DOC
    serial_tokens = 0
    for agent in serial_agents:
        output, tokens = agent.run(task, material)
        serial_tokens += tokens
        # 关键：只传上一环的产出，不把原文再传一遍
        material = output
    serial_elapsed = time.perf_counter() - serial_started

    # ---- 并行：三个 Agent 各自读全量材料
    parallel_agents = [
        TimedAgent("审查员甲", scripted_model([ModelReply(content="建议人工复核")]), delay),
        TimedAgent("审查员乙", scripted_model([ModelReply(content="建议人工复核")]), delay),
        TimedAgent("审查员丙", scripted_model([ModelReply(content="建议人工复核")]), delay),
    ]
    parallel_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(parallel_agents)) as pool:
        futures = [pool.submit(agent.run, task, SOURCE_DOC) for agent in parallel_agents]
        parallel_results = [future.result() for future in futures]
    parallel_elapsed = time.perf_counter() - parallel_started
    parallel_tokens = sum(tokens for _, tokens in parallel_results)

    ensure(
        parallel_elapsed < serial_elapsed,
        f"并行应当比串行快：并行 {parallel_elapsed:.3f}s，串行 {serial_elapsed:.3f}s",
    )
    ensure(
        parallel_tokens > serial_tokens,
        f"并行应当比串行贵（每个 Agent 都要重读全量材料）："
        f"并行 {parallel_tokens} tokens，串行 {serial_tokens} tokens",
    )
    ensure(
        sum(agent.calls for agent in serial_agents)
        == sum(agent.calls for agent in parallel_agents),
        "两种拓扑的模型调用次数应当相同——差别不在调用次数，在每次调用带了多少上下文",
    )
    ensure(
        sum(agent.calls for agent in parallel_agents) == 3,
        "并行这一侧应当正好 3 次模型调用",
    )

    print(f"  串行：{serial_elapsed:.3f}s，{serial_tokens} tokens（原文只读一次）")
    print(f"  并行：{parallel_elapsed:.3f}s，{parallel_tokens} tokens（原文读了三次）")
    print(f"  → 延迟降了 {serial_elapsed / max(parallel_elapsed, 1e-9):.1f} 倍，"
          f"token 涨了 {parallel_tokens / max(serial_tokens, 1):.1f} 倍")


# ---------------------------------------------------------------- 真实模型路径


def run_live() -> None:
    """真实模型路径：让真模型自己分派，断言只校验形状。"""
    banner("第 9 章 · 真实模型路径")
    from ..labkit import real_model

    trace = Trace()
    classifier = Classifier(model=real_model(), trace=trace)

    order_model = real_model()
    refund_model = real_model()
    registry = {
        "订单专家": Expert(
            name="订单专家",
            agent=Agent(
                order_model,
                [查询订单],
                AgentConfig(system_prompt=EXPERI_PROMPT.format(role="订单专家")),
            ),
            trace=trace,
        ),
        "退款专家": Expert(
            name="退款专家",
            agent=Agent(
                refund_model,
                [发起退款],
                AgentConfig(system_prompt=EXPERI_PROMPT.format(role="退款专家")),
            ),
            trace=trace,
        ),
    }

    task = "帮我查一下订单 A-1001 到哪了"
    chosen = classifier.route(task, registry, fallback="订单专家")
    ensure(chosen in registry, f"分类器给出的专家不在注册表里：{chosen!r}")

    result = registry[chosen].run(task)
    ensure(bool(result.final_answer), "真实模型应当给出最终回答")
    ensure(
        all(message.role != "tool" or message.tool_call_id for message in result.messages),
        "工具消息必须带 tool_call_id",
    )

    print(f"  分派到：{chosen}")
    print(f"  回答：{result.final_answer[:200]}")
    print(f"  轨迹：{' → '.join(trace.labels)}")


def run() -> None:
    """跑第 9 章实验。"""
    banner("第 9 章 · 多智能体编排模式")
    check_router()
    check_parallel_vote()
    check_sequential()
    check_supervisor()
    check_shared_state()
    check_split_criteria()
    check_cost_and_latency()
    rule("多 Agent 是分工问题：先证明三条判据同时成立，再挑拓扑")
    print("\n✓ 第 9 章通过")
