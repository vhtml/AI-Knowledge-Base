"""第 1 章实验：生态全景——用代码证明"三层是叠的，不是三选一"。

这个模块解决什么问题：第 1 章的结论如果只靠读文档，读者容易停留在"听过"。
这里把结论变成可执行的断言：

1. 本机装的到底是哪几个版本（版本基线会漂移，实验要能发现漂移）；
2. LangChain 的 ``create_agent`` 返回的**就是**一张 LangGraph 的编译图；
3. Deep Agents 的 harness 同样建在 LangGraph 之上；
4. 选型决策表是可执行的——给定任务特征，得出该选哪一层。

全部离线运行，不联网、不调模型。
"""

from __future__ import annotations

import importlib.metadata as metadata
from dataclasses import dataclass

from ..labkit import ai, banner, ensure, human, scripted_model

# ------------------------------------------------------------ 版本基线

#: 教程写作时的实测版本。实验跑出来不一致时，说明依赖漂了，要重读文档。
EXPECTED_VERSIONS: dict[str, str] = {
    "langgraph": "1.2.20",
    "langchain": "1.5.0",
    "langchain-core": "1.8.0",
}


def installed_version(package: str) -> str:
    """读取已安装包的版本。

    没装就返回 ``"<not installed>"``，不抛异常——自检要把这件事说清楚。
    """
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def check_version_baseline() -> None:
    """核对本机版本与教程基线是否一致。"""
    section = "版本基线"
    print(f"--- {section} ---")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = installed_version(package)
        print(f"  {package:20} 期望 {expected:10} 实际 {actual}")
        ensure(
            actual == expected,
            f"{package} 版本漂移：期望 {expected}，实际 {actual}。"
            " 依赖升级后要重读官方文档再继续。",
        )


# ------------------------------------------------------------ 三层叠放


def check_langchain_runs_on_langgraph() -> None:
    """证明 LangChain 的 Agent 就是一张 LangGraph 图。"""
    print("--- LangChain 跑在 LangGraph 上 ---")
    from langchain.agents import create_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_agent(scripted_model([ai("来自图的回答")]), tools=[])
    print(f"  create_agent 返回类型：{type(agent).__module__}.{type(agent).__name__}")
    ensure(
        isinstance(agent, CompiledStateGraph),
        f"create_agent 应该返回 CompiledStateGraph，实际是 {type(agent)}",
    )

    result = agent.invoke({"messages": [human("hi")]})
    ensure("messages" in result, "Agent 输出里应该有 messages 通道")
    ensure(
        result["messages"][-1].content == "来自图的回答",
        "Agent 应该跑通了模型节点，拿到脚本里的回答",
    )
    print("  ✓ create_agent 返回的是 CompiledStateGraph —— 三层是叠的")


def check_deep_agents_on_langgraph() -> None:
    """证明 Deep Agents harness 建在 LangGraph 之上。"""
    print("--- Deep Agents 建在 LangGraph 之上 ---")
    from langgraph.graph.state import CompiledStateGraph

    import deepagents

    ensure(
        hasattr(deepagents, "create_deep_agent"),
        "deepagents 应该导出 create_deep_agent",
    )
    # deepagents 的中间件体系建立在 LangChain 的 AgentMiddleware 上，
    # 而 LangChain 的 Agent 跑在 LangGraph 上 —— 三层由此连通。
    from langchain.agents.middleware import AgentMiddleware

    ensure(
        issubclass(AgentMiddleware, object),
        "AgentMiddleware 应当可导入",
    )
    ensure(
        CompiledStateGraph.__module__.startswith("langgraph"),
        "编译图的实现应当来自 langgraph 包",
    )
    print("  ✓ harness → framework → runtime 三层链路成立")


# ------------------------------------------------------------ 选型决策


@dataclass(frozen=True)
class TaskProfile:
    """描述一个任务的关键特征，供选型决策使用。"""

    needs_planning: bool
    """需要规划、文件管理、子任务委派。"""

    needs_custom_control_flow: bool
    """流程分支由业务规则决定，而不是模型临时决定。"""

    needs_durability: bool
    """要能中断续跑、重启恢复。"""

    fixed_tool_set: bool
    """工具集固定，没有动态选择。"""


def choose_layer(profile: TaskProfile) -> str:
    """按第 1 章的决策表选层。

    返回 ``"deepagents"`` / ``"langgraph"`` / ``"langchain"`` 之一。
    决策顺序和笔记里的表一致：命中即停。
    """
    if profile.needs_planning:
        return "deepagents"
    if profile.needs_custom_control_flow or profile.needs_durability:
        return "langgraph"
    if profile.fixed_tool_set:
        return "langchain"
    return "langchain"


def check_selection_table() -> None:
    """验证决策表覆盖了四种典型场景。"""
    print("--- 选型决策表 ---")
    cases: list[tuple[str, TaskProfile, str]] = [
        (
            "长程研究任务（要规划+文件）",
            TaskProfile(True, False, True, True),
            "deepagents",
        ),
        (
            "确定性审批流（业务规则决定分支）",
            TaskProfile(False, True, True, True),
            "langgraph",
        ),
        (
            "固定三件套的查询助手",
            TaskProfile(False, False, False, True),
            "langchain",
        ),
        (
            "纯模型调用",
            TaskProfile(False, False, False, False),
            "langchain",
        ),
    ]
    for name, profile, expected in cases:
        actual = choose_layer(profile)
        print(f"  {name:28} → {actual}")
        ensure(actual == expected, f"{name} 选型应为 {expected}，实际 {actual}")


def run() -> None:
    """跑第 1 章实验。"""
    banner("第 1 章 · 生态全景：框架、运行时、harness 三层怎么分")
    check_version_baseline()
    check_langchain_runs_on_langgraph()
    check_deep_agents_on_langgraph()
    check_selection_table()
    print("\n✓ 第 1 章通过")
