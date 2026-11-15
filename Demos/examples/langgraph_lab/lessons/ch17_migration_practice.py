"""第 17 章实验：迁移与工程实践——把上生产前的检查单写成能跑的代码。

这个模块解决什么问题：第 17 章的毕业作品是一份"十条检查单"。
口头清单会被遗忘，代码清单不会。这里把能自动查的部分真的查起来：

1. 状态里的 list/dict 字段有没有配归约器（并行写会炸）；
2. 有外部副作用的节点有没有重试策略；
3. 慢节点有没有超时；
4. 有副作用的节点是不是幂等的；
5. 高风险操作前有没有审批节点；
6. 检查点是不是内存版（重启就没）；
7. 用到中断却没有检查点（暂停的线程无处安放）；
8. 源码里有没有废弃导入（create_react_agent 等）；
9. 源码里有没有失效的环境变量名（LANGCHAIN_TRACING_V2）；
10. 依赖有没有锁文件。

做法是造两张图：一张"有问题的"，一张"健康的"，
让检查单在前者上逐条报错、在后者上全部放行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict, get_args, get_origin, get_type_hints

from ..labkit import banner, ensure, rule

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
LEGACY_SOURCE = FIXTURE_DIR / "legacy_usage.py.txt"

#: 迁移指南里列出的废弃导入
DEPRECATED_IMPORTS: tuple[str, ...] = (
    "from langgraph.prebuilt import create_react_agent",
    "from langgraph.prebuilt import AgentState",
    "from langgraph.prebuilt import ValidationNode",
    "from langgraph.prebuilt import MessageGraph",
    "from langgraph.prebuilt import HumanInterruptConfig",
)

#: 已经失效的环境变量名（写了不报错，只是静默不生效）
STALE_ENV_VARS: tuple[str, ...] = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY")


# ------------------------------------------------------------ 检查单的数据结构


@dataclass(frozen=True)
class NodeProfile:
    """节点的人工标注——工具查不了、只能人写的那部分。"""

    side_effect: bool = False
    """是否会改外部世界（写库、发消息、扣款）。"""

    idempotent: bool = True
    """重跑一次结果是否一样。"""

    slow: bool = False
    """是不是慢节点（该配超时）。"""

    high_risk: bool = False
    """是不是高风险操作（该有审批）。"""

    uses_interrupt: bool = False
    """节点内部是否用了 interrupt。"""


@dataclass(frozen=True)
class RuleResult:
    """一条检查的结果。"""

    key: str
    title: str
    passed: bool
    detail: str


@dataclass
class AuditInput:
    """审计需要的全部输入。"""

    graph: Any
    state_cls: type
    profiles: dict[str, NodeProfile] = field(default_factory=dict)
    source_files: tuple[Path, ...] = ()


# ------------------------------------------------------------ 规则实现


def _state_reducer_problems(state_cls: type) -> list[str]:
    """找出没有配归约器的 list/dict 状态字段。"""
    problems: list[str] = []
    try:
        hints = get_type_hints(state_cls, include_extras=True)
    except Exception:  # noqa: BLE001 - 取不到类型提示就跳过，不误报
        return problems
    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            base = args[0] if args else None
            has_reducer = len(args) > 1
        else:
            base = hint
            has_reducer = False
        # 注意要取泛型参数的原类型：list[str] 的原类型是 list，不是 list[str]
        origin = get_origin(base) or base
        if origin in (list, dict, set) and not has_reducer:
            problems.append(name)
    return problems


def audit_state_reducers(audit: AuditInput) -> RuleResult:
    """规则 1：会累加的状态字段必须有归约器。"""
    problems = _state_reducer_problems(audit.state_cls)
    return RuleResult(
        "state_reducers",
        "累加型状态字段有归约器",
        not problems,
        f"缺归约器：{problems}" if problems else "全部有归约器",
    )


def audit_retry_policy(audit: AuditInput) -> RuleResult:
    """规则 2：有外部副作用的节点必须配重试。"""
    missing: list[str] = []
    for name, profile in audit.profiles.items():
        if not profile.side_effect:
            continue
        node = audit.graph.nodes.get(name)
        policy = getattr(node, "retry_policy", None) if node is not None else None
        if not policy:
            missing.append(name)
    return RuleResult(
        "retry_policy",
        "副作用节点有重试策略",
        not missing,
        f"缺重试：{missing}" if missing else "全部配了重试",
    )


def audit_timeout(audit: AuditInput) -> RuleResult:
    """规则 3：慢节点必须配超时。"""
    missing: list[str] = []
    for name, profile in audit.profiles.items():
        if not profile.slow:
            continue
        node = audit.graph.nodes.get(name)
        policy = getattr(node, "timeout", None) if node is not None else None
        if not policy:
            missing.append(name)
    return RuleResult(
        "timeout",
        "慢节点有超时",
        not missing,
        f"缺超时：{missing}" if missing else "全部配了超时",
    )


def audit_idempotency(audit: AuditInput) -> RuleResult:
    """规则 4：有副作用的节点必须幂等（恢复时节点会重跑）。"""
    risky = [
        name
        for name, profile in audit.profiles.items()
        if profile.side_effect and not profile.idempotent
    ]
    return RuleResult(
        "idempotency",
        "副作用节点幂等",
        not risky,
        f"不幂等：{risky}" if risky else "全部幂等",
    )


def audit_approval(audit: AuditInput) -> RuleResult:
    """规则 5：高风险操作前要有审批节点。"""
    has_approval_node = any(
        name.lower().startswith("approve") for name in audit.graph.nodes
    )
    risky = [name for name, p in audit.profiles.items() if p.high_risk]
    passed = has_approval_node or not risky
    return RuleResult(
        "approval",
        "高风险操作前有审批",
        passed,
        f"高风险节点 {risky} 但图里没有审批节点" if not passed else "有审批节点或无高风险节点",
    )


def audit_checkpointer(audit: AuditInput) -> RuleResult:
    """规则 6：生产不能用内存检查点。"""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = getattr(audit.graph, "checkpointer", None)
    if checkpointer is None:
        return RuleResult("checkpointer", "检查点不是内存版", False, "图没有配检查点")
    if isinstance(checkpointer, InMemorySaver):
        return RuleResult(
            "checkpointer", "检查点不是内存版", False, "用了 InMemorySaver，重启就没"
        )
    return RuleResult("checkpointer", "检查点不是内存版", True, type(checkpointer).__name__)


def audit_interrupt_has_checkpointer(audit: AuditInput) -> RuleResult:
    """规则 7：用了中断就必须有检查点。"""
    uses_interrupt = any(p.uses_interrupt for p in audit.profiles.values())
    if not uses_interrupt:
        return RuleResult("interrupt", "中断有检查点兜底", True, "没有节点用中断")
    has_cp = getattr(audit.graph, "checkpointer", None) is not None
    return RuleResult(
        "interrupt",
        "中断有检查点兜底",
        has_cp,
        "用了 interrupt 但没有检查点" if not has_cp else "有检查点",
    )


def audit_no_deprecated_imports(audit: AuditInput) -> RuleResult:
    """规则 8：源码里没有废弃导入。"""
    hits: list[str] = []
    for path in audit.source_files:
        text = path.read_text("utf-8")
        hits.extend(item for item in DEPRECATED_IMPORTS if item in text)
    return RuleResult(
        "deprecated_imports",
        "没有废弃导入",
        not hits,
        f"发现废弃导入：{sorted(set(hits))}" if hits else "干净",
    )


def audit_no_stale_env_vars(audit: AuditInput) -> RuleResult:
    """规则 9：源码里没有失效的环境变量名。"""
    hits: list[str] = []
    for path in audit.source_files:
        text = path.read_text("utf-8")
        hits.extend(item for item in STALE_ENV_VARS if item in text)
    return RuleResult(
        "stale_env_vars",
        "没有失效的环境变量名",
        not hits,
        f"发现失效变量：{sorted(set(hits))}" if hits else "干净",
    )


def audit_lockfile(audit: AuditInput) -> RuleResult:
    """规则 10：依赖有锁文件。"""
    from ..labkit import repo_root

    lock = repo_root() / "Demos" / "uv.lock"
    return RuleResult(
        "lockfile",
        "依赖有锁文件",
        lock.is_file(),
        str(lock) if lock.is_file() else "找不到 Demos/uv.lock",
    )


RULES: tuple[Callable[[AuditInput], RuleResult], ...] = (
    audit_state_reducers,
    audit_retry_policy,
    audit_timeout,
    audit_idempotency,
    audit_approval,
    audit_checkpointer,
    audit_interrupt_has_checkpointer,
    audit_no_deprecated_imports,
    audit_no_stale_env_vars,
    audit_lockfile,
)


def run_checklist(audit: AuditInput) -> list[RuleResult]:
    """跑完整份检查单。"""
    return [rule(audit) for rule in RULES]


# ------------------------------------------------------------ 被检查的图


def _append(left: list[str], right: list[str]) -> list[str]:
    """累加归约器。"""
    return [*left, *right]


class BadState(TypedDict):
    """有问题的状态：``hits`` 是 list 却没配归约器。"""

    question: str
    hits: list[str]
    answer: str


class GoodState(TypedDict):
    """健康的状态：累加字段都配了归约器。"""

    question: str
    hits: Annotated[list[str], _append]
    answer: str


def build_bad_graph() -> tuple[Any, dict[str, NodeProfile]]:
    """造一张"有问题的图"，让检查单逐条报错。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def fetch(state: BadState) -> dict[str, Any]:
        return {"hits": ["a"]}

    def decide(state: BadState) -> dict[str, Any]:
        # 用了中断却没有检查点
        interrupt({"question": state["question"]})
        return {"answer": "ok"}

    def charge(state: BadState) -> dict[str, Any]:
        # 高风险、慢、且不幂等，却没有重试和超时
        return {"answer": "已扣款"}

    builder = StateGraph(BadState)
    builder.add_node("fetch", fetch)
    builder.add_node("decide", decide)
    builder.add_node("charge", charge)
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "decide")
    builder.add_edge("decide", "charge")
    builder.add_edge("charge", END)
    graph = builder.compile()  # 没有 checkpointer

    profiles = {
        "fetch": NodeProfile(side_effect=True, idempotent=False),
        "decide": NodeProfile(uses_interrupt=True),
        "charge": NodeProfile(
            side_effect=True, idempotent=False, slow=True, high_risk=True
        ),
    }
    return graph, profiles


def build_good_graph(checkpointer: Any) -> tuple[Any, dict[str, NodeProfile]]:
    """造一张"健康的图"，让检查单全部放行。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import RetryPolicy, TimeoutPolicy

    async def fetch(state: GoodState) -> dict[str, Any]:
        return {"hits": ["a"]}

    async def approve(state: GoodState) -> dict[str, Any]:
        return {}

    async def charge(state: GoodState) -> dict[str, Any]:
        return {"answer": "已扣款"}

    builder = StateGraph(GoodState)
    builder.add_node(
        "fetch",
        fetch,
        retry_policy=RetryPolicy(max_attempts=3),
        timeout=TimeoutPolicy(run_timeout=30),
    )
    builder.add_node("approve", approve)
    builder.add_node(
        "charge",
        charge,
        retry_policy=RetryPolicy(max_attempts=3),
        timeout=TimeoutPolicy(run_timeout=30),
    )
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "approve")
    builder.add_edge("approve", "charge")
    builder.add_edge("charge", END)
    graph = builder.compile(checkpointer=checkpointer)

    profiles = {
        "fetch": NodeProfile(side_effect=True, idempotent=True),
        "approve": NodeProfile(),
        "charge": NodeProfile(
            side_effect=True, idempotent=True, slow=True, high_risk=True
        ),
    }
    return graph, profiles


# ------------------------------------------------------------ 检查项


def check_bad_graph_is_flagged() -> None:
    """有问题的图必须被逐条报出来。"""
    print("--- 有问题的图 ---")
    graph, profiles = build_bad_graph()
    results = run_checklist(
        AuditInput(
            graph=graph,
            state_cls=BadState,
            profiles=profiles,
            source_files=(LEGACY_SOURCE,),
        )
    )
    for result in results:
        flag = "✓" if result.passed else "✗"
        print(f"  {flag} {result.title:22} {result.detail}")

    failed = {r.key for r in results if not r.passed}
    for key in (
        "state_reducers",
        "retry_policy",
        "timeout",
        "idempotency",
        "approval",
        "checkpointer",
        "interrupt",
        "deprecated_imports",
        "stale_env_vars",
    ):
        ensure(key in failed, f"有问题的图应当在 {key} 上被报出，实际报出 {failed}")
    print(f"  ✓ 九条问题全部被抓到")


def check_healthy_graph_passes() -> None:
    """健康的图必须全部放行。"""
    print("--- 健康的图 ---")
    from ..labkit import sqlite_checkpointer

    with sqlite_checkpointer() as checkpointer:
        graph, profiles = build_good_graph(checkpointer)
        results = run_checklist(
            AuditInput(
                graph=graph,
                state_cls=GoodState,
                profiles=profiles,
                source_files=(),
            )
        )
    for result in results:
        flag = "✓" if result.passed else "✗"
        print(f"  {flag} {result.title:22} {result.detail}")

    failed = [r for r in results if not r.passed]
    ensure(not failed, f"健康的图应当全部通过，未通过：{[r.title for r in failed]}")
    print("  ✓ 十条全部放行")


def check_deprecated_usage_fixture() -> None:
    """废弃用法样本必须能被扫出来。"""
    print("--- 废弃用法样本 ---")
    text = LEGACY_SOURCE.read_text("utf-8")
    print("  样本里的关键行：")
    for line in text.splitlines():
        if "import" in line or "LANGCHAIN" in line:
            print(f"    {line.strip()}")
    ensure(
        "create_react_agent" in text and "LANGCHAIN_TRACING_V2" in text,
        "样本里应当同时包含废弃导入和失效变量名",
    )
    print("  ✓ 扫描器认得这两种写法")


def check_checklist_report_written() -> None:
    """检查结果要落盘，方便接进 CI。"""
    print("--- 检查单报告落盘 ---")
    graph, profiles = build_bad_graph()
    results = run_checklist(
        AuditInput(
            graph=graph,
            state_cls=BadState,
            profiles=profiles,
            source_files=(LEGACY_SOURCE,),
        )
    )
    payload = {
        "total": len(results),
        "failed": sum(1 for r in results if not r.passed),
        "items": [
            {
                "key": r.key,
                "title": r.title,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in results
        ],
    }
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "checklist.json"
    log_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"  已写入 {log_file}")
    ensure(log_file.is_file(), "检查单报告应当落盘")


def run() -> None:
    """跑第 17 章实验。"""
    banner("第 17 章 · 迁移与工程实践")
    check_bad_graph_is_flagged()
    check_healthy_graph_passes()
    check_deprecated_usage_fixture()
    check_checklist_report_written()
    rule("清单要能跑，才不会被遗忘")
    print("\n✓ 第 17 章通过")
