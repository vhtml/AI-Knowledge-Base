"""部署示例图：给 ``langgraph.json`` 一个真能被加载的目标。

这个模块解决什么问题：部署配置最容易变成"看起来对"——
json 写得挺齐，但图路径指向的变量根本 import 不到。
这里导出一个真实可编译、可运行的图，让配置校验能真的跑一遍。

被 ``langgraph.json`` 引用：

    "graphs": {"agent": "./graph.py:agent"}
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """示例图的状态。"""

    question: str
    answer: str


def answer(state: State) -> dict[str, Any]:
    """把问题原样回显，够用来验证服务链路了。"""
    return {"answer": f"已收到：{state['question']}"}


def build_agent() -> Any:
    """构建并编译示例图。"""
    builder = StateGraph(State)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


agent = build_agent()
