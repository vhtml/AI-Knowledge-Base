"""第 15 章实验：可观测性与评测——用本地 trace + 黄金集证明"改动能被看见"。

这个模块解决什么问题：第 15 章的两条主张要变成能跑的东西：

1. **没有 trace 只能猜**——用本地回调收集器把每个节点的输入输出抓下来，
   定位"答错的是哪一步"；
2. **没有评测集等于自我感动**——用一份黄金集跑两个实现（好版本 / 坏版本），
   证明分数会随改动变化，并把结果写进日志文件方便前后对比。

全程离线：不联网、不调模型、不上报任何数据到 LangSmith。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.callbacks import BaseCallbackHandler

from ..labkit import ai, banner, ensure, human, rule, scripted_model

# ------------------------------------------------------------ 知识与黄金集


@dataclass(frozen=True)
class Policy:
    """一条政策条目。

    ``keywords`` 用来做关键词匹配，``needle`` 是回答里必须出现的关键片段。
    """

    id: str
    keywords: tuple[str, ...]
    text: str


POLICIES: tuple[Policy, ...] = (
    Policy("P1", ("退款", "退货"), "退款申请需要在订单完成后 7 天内发起。"),
    Policy("P2", ("发票", "开票"), "发票可以在订单完成后随时申请开具。"),
    Policy("P3", ("保修", "维修"), "电子产品的保修期是一年。"),
    Policy("P4", ("运费", "配送", "免运费"), "订单满 99 元免运费，否则收取 10 元运费。"),
)

#: 黄金集：每条给出问题与"答案里必须出现的关键片段"。
GOLD: tuple[dict[str, str], ...] = (
    {"id": "G1", "question": "退款要多久内申请？", "needle": "7 天"},
    {"id": "G2", "question": "怎么开发票？", "needle": "随时"},
    {"id": "G3", "question": "保修期是多久？", "needle": "一年"},
    {"id": "G4", "question": "运费怎么算？", "needle": "99"},
    {"id": "G5", "question": "免运费门槛是多少？", "needle": "99"},
)

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "eval_log.json"


# ------------------------------------------------------------ 本地追踪


@dataclass
class TraceCollector(BaseCallbackHandler):
    """把节点级的输入输出收集到内存里。

    为什么不用 LangSmith：追踪要联网、要密钥、要担心数据外泄。
    本地回调零依赖，单元测试里比云端追踪更好用——断言能写死。
    """

    steps: list[dict[str, Any]] = field(default_factory=list)

    def on_chain_start(self, serialized: dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        """节点开始执行时记一笔。

        LangGraph 的节点链不会填 ``serialized``，节点名要从 ``kwargs["name"]`` 拿。
        """
        name = kwargs.get("name") or (serialized or {}).get("name") or "<unknown>"
        self.steps.append({"node": name, "event": "start", "input": _brief(inputs)})

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        """节点结束时把输出补上。

        注意 langchain-core 1.x 的签名是 ``on_chain_end(self, outputs, *, run_id, ...)``，
        只有 outputs 一个位置参数——写成老版的 ``(self, inputs, outputs)`` 会直接报
        "missing 1 required positional argument"。
        """
        if not self.steps:
            return
        self.steps[-1]["event"] = "end"
        self.steps[-1]["output"] = _brief(outputs)

    def node_names(self) -> list[str]:
        """按执行顺序返回节点名（去掉图级别的 LangGraph 根 run）。"""
        seen: list[str] = []
        for step in self.steps:
            name = step["node"]
            if name.startswith("LangGraph") or name in seen:
                continue
            seen.append(name)
        return seen


def _brief(value: Any) -> Any:
    """把值压成可打印的短摘要。"""
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "..."


# ------------------------------------------------------------ 被评测的图


class State(TypedDict):
    """政策问答图的状态。"""

    question: str
    hits: list[str]
    answer: str


def build_graph(variant: str) -> Any:
    """构建两个实现之一。

    ``variant="good"``：检索按关键词命中数打分，选最好的政策。
    ``variant="bad"``：只认"退款"关键词——模拟一次改坏了的改动。
    """
    from langgraph.graph import END, START, StateGraph

    def retrieve(state: State) -> dict[str, Any]:
        question = state["question"]
        if variant == "bad":
            matched = [p.id for p in POLICIES if "退款" in question and "退款" in p.keywords]
        else:
            scored = [
                (sum(1 for kw in p.keywords if kw in question), p.id)
                for p in POLICIES
            ]
            best = max(scored, key=lambda item: item[0])
            matched = [best[1]] if best[0] > 0 else []
        return {"hits": matched}

    def answer(state: State) -> dict[str, Any]:
        texts = [p.text for p in POLICIES if p.id in state["hits"]]
        return {"answer": texts[0] if texts else "抱歉，我没有找到对应的政策。"}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder


def run_case(graph: Any, question: str) -> dict[str, Any]:
    """跑一个问题，返回结果与 trace。"""
    collector = TraceCollector()
    result = graph.invoke(
        {"question": question, "hits": [], "answer": ""},
        config={"callbacks": [collector]},
    )
    return {
        "question": question,
        "answer": result["answer"],
        "hits": result["hits"],
        "nodes": collector.node_names(),
        "steps": collector.steps,
    }


# ------------------------------------------------------------ 评测器


def evaluate(graph: Any, gold: tuple[dict[str, str], ...]) -> dict[str, Any]:
    """跑一遍黄金集，返回逐条结果与通过率。

    这是**确定性评估器**：判定只看关键片段在不在答案里，不依赖模型，
    所以同一份代码每次跑出来的分数都一样——分数变了，说明实现变了。
    """
    rows: list[dict[str, Any]] = []
    for item in gold:
        case = run_case(graph, item["question"])
        hit = item["needle"] in case["answer"]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "needle": item["needle"],
                "answer": case["answer"],
                "hit": hit,
                "nodes": case["nodes"],
            }
        )
    passed = sum(1 for row in rows if row["hit"])
    return {"passed": passed, "total": len(rows), "rows": rows}


# ------------------------------------------------------------ 检查项


def check_trace_locates_the_failure() -> None:
    """trace 能指出"错在哪一步"。"""
    print("--- trace 定位问题 ---")
    graph = build_graph("bad").compile()
    case = run_case(graph, "保修期是多久？")

    print(f"  问题：{case['question']}")
    print(f"  答案：{case['answer']}")
    print(f"  执行过的节点：{case['nodes']}")
    for step in case["steps"]:
        print(f"    · {step['node']:10} {step.get('input')} → {step.get('output')}")

    ensure(
        "retrieve" in case["nodes"] and "answer" in case["nodes"],
        f"trace 应该记到两个节点，实际 {case['nodes']}",
    )
    retrieve_step = next(s for s in case["steps"] if s["node"] == "retrieve")
    ensure(
        "hits" in str(retrieve_step.get("output")),
        "trace 里应该能看到 retrieve 的输出（含命中的政策 id）",
    )
    ensure(
        case["answer"] == "抱歉，我没有找到对应的政策。",
        f"坏版本应当答不上来，实际 {case['answer']}",
    )
    print("  ✓ 一次运行就能看出是 retrieve 没命中，不是 answer 不会说")


def check_eval_detects_regression() -> None:
    """黄金集能发现改动：好版本全过，坏版本掉分。"""
    print("--- 黄金集发现回归 ---")
    good = evaluate(build_graph("good").compile(), GOLD)
    bad = evaluate(build_graph("bad").compile(), GOLD)

    print(f"  good 版本：{good['passed']}/{good['total']}")
    print(f"  bad  版本：{bad['passed']}/{bad['total']}")
    for row in bad["rows"]:
        flag = "✓" if row["hit"] else "✗"
        print(f"    {flag} {row['id']} {row['question']} → {row['answer']}")

    ensure(
        good["passed"] == good["total"],
        f"好版本应当全过，实际 {good['passed']}/{good['total']}",
    )
    ensure(
        bad["passed"] < good["passed"],
        f"坏版本分数应当更低，实际 bad={bad['passed']} good={good['passed']}",
    )
    ensure(
        bad["passed"] == 1,
        f"坏版本只有'退款'类问题能答上，实际 {bad['passed']}",
    )
    print("  ✓ 同一份黄金集把两个实现的差距量化出来了")


def check_eval_is_reproducible() -> None:
    """评估器是确定性的：同样输入跑两次，分数必须一样。"""
    print("--- 评估可复现 ---")
    graph = build_graph("good").compile()
    first = evaluate(graph, GOLD)
    second = evaluate(graph, GOLD)
    ensure(
        first["passed"] == second["passed"],
        f"两次评测分数应当一致，实际 {first['passed']} vs {second['passed']}",
    )
    print(f"  两次评测都是 {first['passed']}/{first['total']}，分数可复现")


def check_eval_log_written() -> None:
    """评测结果要落盘，方便前后对比。"""
    print("--- 评测日志落盘 ---")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "gold_size": len(GOLD),
        "variants": {
            "good": evaluate(build_graph("good").compile(), GOLD),
            "bad": evaluate(build_graph("bad").compile(), GOLD),
        },
    }
    LOG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"  已写入 {LOG_FILE}")

    ensure(LOG_FILE.is_file(), "评测日志文件应当存在")
    saved = json.loads(LOG_FILE.read_text("utf-8"))
    ensure(
        saved["variants"]["good"]["passed"] == len(GOLD),
        "日志里记录的好版本分数应当是全过",
    )


def check_env_var_names() -> None:
    """环境变量名必须是 LANGSMITH_*，老名字已经失效。"""
    print("--- 环境变量命名 ---")
    import os

    old = "LANGCHAIN_TRACING_V2"
    ensure(
        old not in os.environ,
        f"{old} 已经失效，仓库里不应该再出现这个变量名",
    )
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        print(f"  {name:24} 当前值：{os.environ.get(name, '<未设置>')}")
    print("  ✓ 老变量名 LANGCHAIN_TRACING_V2 写错不报错，只会静默不上报")


def run() -> None:
    """跑第 15 章实验。"""
    banner("第 15 章 · 可观测性与评测")
    check_trace_locates_the_failure()
    check_eval_detects_regression()
    check_eval_is_reproducible()
    check_eval_log_written()
    check_env_var_names()
    rule("先有 trace 才能定位，先有黄金集才能谈优化")
    print("\n✓ 第 15 章通过")


def run_live() -> None:
    """真实模型路径：让真模型回答黄金集的问题，记进同一份日志。

    断言只校验形状（每条都有非空回答），不校验内容——
    真模型答得对不对，那是评测集该管的事，不是运行路径该管的事。
    """
    from ..labkit import real_chat_model

    banner("第 15 章 · 真实模型路径")
    model = real_chat_model()
    rows: list[dict[str, Any]] = []
    for item in GOLD:
        answer = model.invoke(
            [
                {"role": "system", "content": "你是政策助手，请用一句话回答，只给结论。"},
                {"role": "user", "content": item["question"]},
            ]
        ).content
        answer = str(answer)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "needle": item["needle"],
                "answer": answer,
                "hit": item["needle"] in answer,
            }
        )
        print(f"  {item['id']} {item['question']} → {answer[:80]}")

    payload = json.loads(LOG_FILE.read_text("utf-8")) if LOG_FILE.is_file() else {}
    payload["live"] = {"rows": rows, "passed": sum(1 for r in rows if r["hit"])}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    ensure(len(rows) == len(GOLD), f"应当跑完 {len(GOLD)} 条，实际 {len(rows)}")
    ensure(all(row["answer"].strip() for row in rows), "每条都应当有非空回答")
    print(f"\n真实模型通过率：{payload['live']['passed']}/{len(GOLD)}")
