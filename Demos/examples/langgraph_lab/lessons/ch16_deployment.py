"""第 16 章实验：部署——配置校验、部署前置检查单，以及"改代码会打断在跑的会话"。

这个模块解决什么问题：第 16 章的两条主张要变成能跑的东西：

1. **配置不能只是"看起来对"**——langgraph.json 的每个字段都要能被校验，
   图路径要真的能 import 出图来；
2. **发版会立刻作用到所有在跑的线程**——把节点改名后，老线程恢复会直接失败；
   add-then-remove 模式能救回来。

不真的起服务器（langgraph-cli 未安装），全部离线完成。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

from ..labkit import banner, ensure, rule

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy"
CONFIG_PATH = DEPLOY_DIR / "langgraph.json"

#: 密钥的典型特征，出现在配置文件里就是事故
SECRET_PATTERNS = ("sk-", "lsv2_", "api_key", "apikey", "password", "secret", "token")


# ------------------------------------------------------------ 配置校验


@dataclass(frozen=True)
class RuleResult:
    """一条检查规则的结果。"""

    key: str
    title: str
    passed: bool
    detail: str


def _load_config(path: Path) -> dict[str, Any]:
    """读取配置文件。"""
    return json.loads(path.read_text("utf-8"))


def _resolve_graph(path: Path, spec: str) -> Any:
    """按 ``文件:变量`` 的写法加载图对象。"""
    file_part, _, var = spec.partition(":")
    module_path = (path.parent / file_part).resolve()
    spec_obj = importlib.util.spec_from_file_location(
        f"_deploy_{module_path.stem}", module_path
    )
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"无法加载模块：{module_path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[spec_obj.name] = module
    spec_obj.loader.exec_module(module)
    return getattr(module, var)


def rule_required_keys(config: dict[str, Any]) -> RuleResult:
    """必需字段齐全。"""
    required = ("dependencies", "graphs", "env", "python_version")
    missing = [key for key in required if key not in config]
    return RuleResult(
        "required_keys",
        "必需字段齐全",
        not missing,
        f"缺少 {missing}" if missing else "dependencies / graphs / env / python_version 都在",
    )


def rule_graphs_resolvable(config: dict[str, Any]) -> RuleResult:
    """每个图路径都要能真的 import 出一个图对象。"""
    graphs = config.get("graphs") or {}
    if not graphs:
        return RuleResult("graphs", "图可解析", False, "graphs 是空的")
    problems: list[str] = []
    for name, spec in graphs.items():
        try:
            graph = _resolve_graph(CONFIG_PATH, spec)
        except Exception as exc:  # noqa: BLE001 - 检查清单要报出任何加载失败
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not hasattr(graph, "invoke"):
            problems.append(f"{name}: {spec} 不是可执行的图")
    return RuleResult(
        "graphs",
        "图可解析",
        not problems,
        "；".join(problems) if problems else f"{len(graphs)} 个图都能加载",
    )


def rule_no_secrets(config: dict[str, Any]) -> RuleResult:
    """配置文件里不能出现密钥。"""
    raw = json.dumps(config, ensure_ascii=False).lower()
    hits = [pattern for pattern in SECRET_PATTERNS if pattern in raw]
    return RuleResult(
        "no_secrets",
        "配置里没有密钥",
        not hits,
        f"发现疑似密钥字段：{hits}" if hits else "干净",
    )


def rule_python_version(config: dict[str, Any]) -> RuleResult:
    """python_version 要和当前解释器一致。"""
    declared = str(config.get("python_version", ""))
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    return RuleResult(
        "python_version",
        "Python 版本匹配",
        declared == current,
        f"声明 {declared}，当前 {current}",
    )


def rule_env_is_file_path(config: dict[str, Any]) -> RuleResult:
    """env 必须指向文件路径，不能是字典。"""
    env = config.get("env")
    if not isinstance(env, str):
        return RuleResult(
            "env", "env 指向文件", False, f"env 应当是字符串路径，实际是 {type(env).__name__}"
        )
    return RuleResult("env", "env 指向文件", True, f"{env}（文件可不存在，部署时注入）")


CHECKS: tuple[Callable[[dict[str, Any]], RuleResult], ...] = (
    rule_required_keys,
    rule_graphs_resolvable,
    rule_no_secrets,
    rule_python_version,
    rule_env_is_file_path,
)


def run_checklist(config: dict[str, Any]) -> list[RuleResult]:
    """跑一遍部署前置检查单。"""
    return [check(config) for check in CHECKS]


# ------------------------------------------------------------ 检查项


def check_real_config_is_valid() -> None:
    """仓库里那份 langgraph.json 必须过全部检查。"""
    print("--- 真实配置校验 ---")
    config = _load_config(CONFIG_PATH)
    results = run_checklist(config)
    for result in results:
        flag = "✓" if result.passed else "✗"
        print(f"  {flag} {result.title:14} {result.detail}")

    failed = [r for r in results if not r.passed]
    ensure(
        not failed,
        f"真实配置应当全部通过，未通过：{[r.title for r in failed]}",
    )
    print(f"  ✓ {CONFIG_PATH.relative_to(CONFIG_PATH.parents[3])} 全部检查通过")


def check_broken_config_is_caught() -> None:
    """坏配置必须被检查单拦下来。"""
    print("--- 坏配置必须被拦下 ---")
    broken = {
        "dependencies": ["langgraph"],
        # 缺 env、缺 python_version
        "graphs": {"agent": "./nonexistent.py:agent"},
        "env": {"LLM_API_KEY": "sk-abcdef123456"},  # 字典 + 密钥，两个错
    }
    results = run_checklist(broken)
    for result in results:
        flag = "✓" if result.passed else "✗"
        print(f"  {flag} {result.title:14} {result.detail}")

    failed_keys = {r.key for r in results if not r.passed}
    for key in ("required_keys", "graphs", "no_secrets", "python_version", "env"):
        ensure(key in failed_keys, f"坏配置应当在 {key} 上被拦下，实际拦下的是 {failed_keys}")
    print("  ✓ 五个错全部被检查单抓到")


# ------------------------------------------------------------ 向后兼容


class CompatState(TypedDict):
    """兼容性实验用的状态。"""

    question: str
    draft: str
    approved: bool


def _build_compat_graph(node_name: str, keep_legacy: bool, checkpointer: Any) -> Any:
    """构建某一版本的图。

    ``node_name`` 是"产出草稿"节点的新名字；
    ``keep_legacy=True`` 时额外保留旧名字节点，让老线程还能恢复。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    def fetch(state: CompatState) -> dict[str, Any]:
        return {"draft": f"草稿：{state['question']}"}

    def make_node(name: str) -> Callable[[CompatState], dict[str, Any]]:
        def node(state: CompatState) -> dict[str, Any]:
            # 在这里中断，线程会停在这个节点上
            decision = interrupt({"draft": state["draft"], "question": "批准吗？"})
            return {"approved": bool(decision)}

        node.__name__ = name
        return node

    builder = StateGraph(CompatState)
    builder.add_node("fetch", fetch)
    builder.add_node(node_name, make_node(node_name))
    if keep_legacy and node_name != "compose":
        builder.add_node("compose", make_node("compose"))
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", node_name)
    if keep_legacy and node_name != "compose":
        builder.add_edge("fetch", "compose")
    builder.add_edge(node_name, END)
    if keep_legacy and node_name != "compose":
        builder.add_edge("compose", END)
    return builder.compile(checkpointer=checkpointer)


def check_rename_breaks_inflight_thread() -> None:
    """改名会让暂停中的老线程丢掉那一步。

    实测（langgraph 1.2.20）：老线程恢复时**不抛异常**，
    找不到旧名字的待执行节点，就把它**静默丢弃**——运行"完成"了，
    但那一步根本没执行，状态里的结果停在旧值。
    这比直接报错更危险：没有告警，只有一个安静的错误结果。
    """
    print("--- 改节点名会打断在跑的线程 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t-rename"}}

    # v1：节点名叫 compose，线程停在这里
    v1 = _build_compat_graph("compose", keep_legacy=False, checkpointer=saver)
    v1.invoke({"question": "写个方案", "draft": "", "approved": False}, config)
    state = v1.get_state(config)
    print(f"  v1 线程停在节点：{state.next}")
    ensure(state.next == ("compose",), f"线程应当停在 compose，实际停在 {state.next}")

    # v2：把 compose 改名成 write，老 checkpoint 还在等 compose
    v2 = _build_compat_graph("write", keep_legacy=False, checkpointer=saver)
    result = v2.invoke(Command(resume=True), config)
    after = v2.get_state(config)
    print(f"  v2 恢复结果：{result}")
    print(f"  v2 恢复后 next：{after.next}（空 = 运行结束）")

    ensure(
        after.next == (),
        f"恢复后运行应当已经结束，实际 next={after.next}",
    )
    ensure(
        result.get("approved") is False,
        f"改名后审批这一步被静默丢弃了，approved 应当仍是 False，实际 {result}",
    )
    print("  ✓ 关键发现：没有报错，但那一步被静默丢了——运行结束了，结果是错的")
    print("  ✓ 所以发版前必须查有没有线程停在要改的节点上")


def check_add_then_remove_saves_inflight_thread() -> None:
    """add-then-remove：保留旧节点名，老线程照样能恢复。"""
    print("--- add-then-remove 能救回老线程 ---")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t-add-remove"}}

    v1 = _build_compat_graph("compose", keep_legacy=False, checkpointer=saver)
    v1.invoke({"question": "写个方案", "draft": "", "approved": False}, config)

    # v2：新节点叫 write，但旧名字 compose 还在
    v2 = _build_compat_graph("write", keep_legacy=True, checkpointer=saver)
    result = v2.invoke(Command(resume=True), config)
    print(f"  v2 恢复结果：{result}")
    ensure(
        result.get("approved") is True,
        f"老线程应当在新代码下正常恢复，实际 {result}",
    )
    print("  ✓ 保留旧节点名，老线程顺利恢复")


def run() -> None:
    """跑第 16 章实验。"""
    banner("第 16 章 · 部署：本地服务器、Studio 与生产")
    check_real_config_is_valid()
    check_broken_config_is_caught()
    check_rename_breaks_inflight_thread()
    check_add_then_remove_saves_inflight_thread()
    rule("配置要能校验，发版要照顾还在跑的线程")
    print("\n✓ 第 16 章通过")
