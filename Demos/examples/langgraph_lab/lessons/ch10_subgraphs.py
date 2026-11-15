"""第 10 章实验：子图与多智能体编排——状态怎么跨边界传。

这个模块解决什么问题：把"子图是节点"与"子图是独立图"两种边界的差别变成可断言的事实：

1. 子图作为节点直接挂上去：共享父图的 key，子图写什么父图就写回什么，私有键不外泄；
2. 子图独立调用：父子 state 不同名时字段必须手工映射，**映射写错不报错**，只会给你一个看起来合理的错答案；
3. ``Command`` 能把子图的返回值包成"改状态 + 选路"两件事；
4. 编排器把任务路由到不同专家子图，路由结果写进父状态；
5. 子图内部的事件在父图流里带 subgraph 命名空间前缀。

确定性路径不联网、不调模型；真实路径（``--live``）才调本机 OpenAI 兼容端点。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.types import Command

from ..labkit import banner, ensure, human, rule


class OrchestratorState(TypedDict):
    """编排器的父图状态。

    三个专家子图的 state schema 各不相同：``general`` 与父图共享 ``question``/``answer``，
    ``math``/``code`` 用的是另一套键名——这样才能看清两种跨边界方式的差别。
    """

    question: str
    expert: str
    answer: str


# ------------------------------------------------------------ 专家子图


def build_general_expert() -> Any:
    """通用专家：state 与父图**共享键**，所以适合直接当节点挂上去。"""
    from langgraph.graph import END, START, StateGraph

    class GeneralState(TypedDict):
        question: str
        answer: str
        scratch: str
        """私有键：只有子图自己看得见，不该出现在父图状态里。"""

    def answer_it(state: GeneralState) -> dict[str, Any]:
        return {
            "scratch": f"草稿：{state['question']}",
            "answer": f"通用回答：{state['question']}",
        }

    builder = StateGraph(GeneralState)
    builder.add_node("answer_it", answer_it)
    builder.add_edge(START, "answer_it")
    builder.add_edge("answer_it", END)
    return builder.compile()


def build_math_expert() -> Any:
    """数学专家：state 的键名和父图**完全不同**，只能手工映射。"""
    from langgraph.graph import END, START, StateGraph

    class MathState(TypedDict):
        expression: str
        result: str

    def compute(state: MathState) -> dict[str, Any]:
        # 用 get 而不是 state[...]：键没对上时这里不会炸，只会算出一个空结果——
        # 这正是手工映射最危险的地方，错误被静默吞掉了。
        expression = state.get("expression", "")
        if not expression:
            return {"result": "空式子"}
        return {"result": f"{expression} 的答案是 42"}

    builder = StateGraph(MathState)
    builder.add_node("compute", compute)
    builder.add_edge(START, "compute")
    builder.add_edge("compute", END)
    return builder.compile()


def build_code_expert() -> Any:
    """代码专家：同样用另一套键名，演示独立调用的映射。"""
    from langgraph.graph import END, START, StateGraph

    class CodeState(TypedDict):
        snippet: str
        language: str

    def write_code(state: CodeState) -> dict[str, Any]:
        snippet = state.get("snippet", "")
        if not snippet:
            return {"language": "unknown"}
        # 从自然语言里挑出合法的标识符当函数名——这里只关心"映射过来了没有"
        name = "".join(
            ch for ch in snippet if ch.isascii() and (ch.isalnum() or ch == "_")
        ) or "handle"
        return {"language": "python", "snippet": f"def {name}():\n    return 1"}

    builder = StateGraph(CodeState)
    builder.add_node("write_code", write_code)
    builder.add_edge(START, "write_code")
    builder.add_edge("write_code", END)
    return builder.compile()


# ------------------------------------------------------------ 两种跨边界方式


def check_shared_subgraph_writes_back() -> None:
    """子图作为节点：共享键写回父图，私有键不外泄。"""
    print("--- 子图作为节点 ---")
    from langgraph.graph import END, START, StateGraph

    general = build_general_expert()

    builder = StateGraph(OrchestratorState)
    builder.add_node("general", general)
    builder.add_edge(START, "general")
    builder.add_edge("general", END)
    graph = builder.compile()

    result = graph.invoke({"question": "什么是子图", "expert": "general", "answer": ""})
    print(f"  父图最终状态：{result}")

    ensure(
        result["answer"] == "通用回答：什么是子图",
        f"子图共享键的更新应当写回父图，实际 {result['answer']}",
    )
    ensure(
        "scratch" not in result,
        f"子图的私有键不该泄漏到父图状态，实际 {result}",
    )
    ensure(
        result["question"] == "什么是子图",
        "父图里没被改的键应当保持原值",
    )


def check_isolated_subgraph_needs_mapping() -> None:
    """子图独立调用：字段按映射关系传，映射写错只会静默给错答案。"""
    print("--- 子图独立调用与手工映射 ---")
    math_expert = build_math_expert()

    good = math_expert.invoke({"expression": "1+1"})
    print(f"  正确映射的输出：{good}")
    ensure(
        good["result"] == "1+1 的答案是 42",
        f"子图应当算出结果，实际 {good}",
    )

    # 坑：父图叫 question，子图叫 expression，映射漏了键名
    wrong = math_expert.invoke({"question": "1+1"})
    print(f"  漏掉映射的输出：{wrong}")
    ensure(
        wrong["result"] == "空式子",
        f"键名没对上时子图读到空值，实际 {wrong}",
    )
    ensure(
        wrong["result"] != good["result"],
        "这正是要点：映射错了不抛异常，只是结果悄悄变了",
    )


def check_command_wraps_subgraph_result() -> None:
    """Command：把子图的返回值包成"改状态 + 选路"。"""
    print("--- Command 包装子图返回值 ---")
    from langgraph.graph import END, START, StateGraph

    class ReviewState(TypedDict):
        text: str
        verdict: str

    def judge(state: ReviewState) -> dict[str, Any]:
        return {"verdict": "通过" if state.get("text") else "驳回"}

    review_builder = StateGraph(ReviewState)
    review_builder.add_node("judge", judge)
    review_builder.add_edge(START, "judge")
    review_builder.add_edge("judge", END)
    reviewer = review_builder.compile()

    class RouteState(TypedDict):
        text: str
        verdict: str
        stage: str

    def call_reviewer(state: RouteState) -> Command:
        """子图只负责给结论；改状态和选路由 Command 一起做。"""
        outcome = reviewer.invoke({"text": state["text"]})
        return Command(
            update={"verdict": outcome["verdict"], "stage": "reviewed"},
            goto="publish",
        )

    def publish(state: RouteState) -> dict[str, Any]:
        """只有 Command.goto 过来才会跑到这里。"""
        return {"stage": state["stage"] + "-published"}

    builder = StateGraph(RouteState)
    builder.add_node("review", call_reviewer, destinations=("publish",))
    builder.add_node("publish", publish)
    builder.add_edge(START, "review")
    builder.add_edge("publish", END)
    graph = builder.compile()

    result = graph.invoke({"text": "一段待审的文字", "verdict": "", "stage": "new"})
    print(f"  路由结果：{result}")

    ensure(
        result["verdict"] == "通过",
        f"Command.update 应当把子图的结论写进状态，实际 {result['verdict']}",
    )
    ensure(
        result["stage"] == "reviewed-published",
        f"Command.goto 应当把执行转到 publish，实际 {result['stage']}",
    )


# ------------------------------------------------------------ 编排器


def build_orchestrator() -> Any:
    """搭一张编排图：先分流，再交给不同的专家子图。"""
    from langgraph.graph import END, START, StateGraph

    general = build_general_expert()
    math_expert = build_math_expert()
    code_expert = build_code_expert()

    def dispatch(state: OrchestratorState) -> dict[str, Any]:
        """按关键词分流——业务规则决定的路由，不需要模型。"""
        question = state["question"]
        if any(token in question for token in ("算", "计算", "+", "*")):
            expert = "math"
        elif any(token in question for token in ("代码", "函数", "写个")):
            expert = "code"
        else:
            expert = "general"
        return {"expert": expert}

    def call_math(state: OrchestratorState) -> dict[str, Any]:
        """独立调用：父图的 question 要手工映射成子图的 expression。"""
        outcome = math_expert.invoke({"expression": state["question"]})
        return {"answer": outcome["result"]}

    def call_code(state: OrchestratorState) -> dict[str, Any]:
        """独立调用：snippet / language 两个键都要映射。"""
        outcome = code_expert.invoke({"snippet": state["question"], "language": "python"})
        return {"answer": f"{outcome['language']}：{outcome['snippet']}"}

    builder = StateGraph(OrchestratorState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("general", general)
    builder.add_node("math", call_math)
    builder.add_node("code", call_code)
    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", lambda state: state["expert"], ["general", "math", "code"])
    builder.add_edge("general", END)
    builder.add_edge("math", END)
    builder.add_edge("code", END)
    return builder.compile()


def check_orchestrator_routes_to_experts() -> None:
    """编排器把任务分到不同专家子图，结果写进父状态。"""
    print("--- 编排器路由到专家子图 ---")
    graph = build_orchestrator()

    cases = [
        ("帮我算 1+1", "math", "42"),
        ("写个函数 parse_text", "code", "def parse_text()"),
        ("讲个笑话", "general", "通用回答"),
    ]
    for question, expected_expert, expected_fragment in cases:
        result = graph.invoke({"question": question, "expert": "", "answer": ""})
        print(f"  {question!r:16} → 专家 {result['expert']:8} 答案 {result['answer']!r}")
        ensure(
            result["expert"] == expected_expert,
            f"{question!r} 应当路由到 {expected_expert}，实际 {result['expert']}",
        )
        ensure(
            expected_fragment in result["answer"],
            f"{question!r} 的答案应当包含 {expected_fragment!r}，实际 {result['answer']!r}",
        )


def check_subgraph_events_carry_namespace() -> None:
    """子图内部的事件在父图流里带 subgraph 前缀。"""
    print("--- 子图事件的命名空间 ---")
    graph = build_orchestrator()

    chunks = list(
        graph.stream(
            {"question": "讲个笑话", "expert": "", "answer": ""},
            stream_mode="updates",
            subgraphs=True,
            version="v2",
        )
    )
    for chunk in chunks:
        print(f"  ns={chunk['ns']} data={chunk['data']}")

    namespaced = [chunk for chunk in chunks if chunk["ns"]]
    root = [chunk for chunk in chunks if not chunk["ns"]]

    ensure(
        len(namespaced) >= 1,
        "子图内部的事件应当带 ns 前缀透出",
    )
    ensure(
        all(chunk["ns"][0].startswith("general:") for chunk in namespaced),
        f"ns 应当是「子图节点名:运行 id」，实际 {[c['ns'] for c in namespaced]}",
    )
    ensure(
        any("answer_it" in chunk["data"] for chunk in namespaced),
        f"ns 里应当是子图内部节点的结果，实际 {[c['data'] for c in namespaced]}",
    )
    ensure(
        any("general" in chunk["data"] for chunk in root),
        f"根层事件里应当是父图节点的结果，实际 {[c['data'] for c in root]}",
    )


# ------------------------------------------------------------ 入口


def run() -> None:
    """跑第 10 章实验。"""
    banner("第 10 章 · 子图与多智能体编排")
    check_shared_subgraph_writes_back()
    check_isolated_subgraph_needs_mapping()
    check_command_wraps_subgraph_result()
    check_orchestrator_routes_to_experts()
    check_subgraph_events_carry_namespace()
    rule("共享键就挂成节点，不同名就得手工映射，Command 负责改状态 + 选路")
    print("\n✓ 第 10 章通过")


def run_live() -> None:
    """真实模型路径：真模型在编排器里做分流。

    断言只校验形状（路由结果、答案非空），不校验模型说了什么。
    """
    from ..labkit import live_enabled, real_chat_model

    banner("第 10 章 · 真实模型路径")
    if not live_enabled():
        print("  未开启 --live，跳过真实模型路径。")
        return

    model = real_chat_model()
    general = build_general_expert()
    math_expert = build_math_expert()

    from langgraph.graph import END, START, StateGraph

    class LiveState(TypedDict):
        question: str
        expert: str
        answer: str

    def dispatch(state: LiveState) -> dict[str, Any]:
        """让真模型挑专家：一次调用，形状不校验内容。"""
        reply = model.invoke(
            [
                human(
                    "只回答一个词：这个问题属于 math 还是 general？"
                    f"问题：{state['question']}"
                )
            ]
        )
        expert = str(reply.content).strip().lower()
        return {"expert": "math" if "math" in expert else "general"}

    def call_math(state: LiveState) -> dict[str, Any]:
        outcome = math_expert.invoke({"expression": state["question"]})
        return {"answer": outcome["result"]}

    builder = StateGraph(LiveState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("general", general)
    builder.add_node("math", call_math)
    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", lambda s: s["expert"], ["general", "math"])
    builder.add_edge("general", END)
    builder.add_edge("math", END)
    graph = builder.compile()

    result = graph.invoke({"question": "帮我算 1+1", "expert": "", "answer": ""})
    print(f"  路由结果：expert={result['expert']!r} answer={result['answer']!r}")
    ensure(
        result["expert"] in {"math", "general"},
        f"专家名应当是 math 或 general，实际 {result['expert']}",
    )
    ensure(bool(result["answer"]), "真实路径也应当拿到非空答案")
