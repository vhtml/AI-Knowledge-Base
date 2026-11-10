"""LangGraph Graph API 计算器 agent。

按官方 quickstart（Python）的顺序组织：定义工具与模型 → 定义状态 → 模型节点 →
工具节点 → 条件边 → 编译成图 → 运行。与指南的唯一差异是模型侧：改用仓库既有的
OpenAI 兼容端点（Demos/.env 里的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL），
这样不必额外引入 ANTHROPIC_API_KEY。

运行：cd Demos && uv run python langgraph-demo/agent.py
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

# .env 固定指向仓库的 Demos/ 目录，脚本从任意工作目录启动都读到同一份配置
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

# 模型名带 "qiniu/" 前缀，init_chat_model 无法自动推断 provider，所以显式指定 openai
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ---------- 1. 定义工具与模型 ----------

model = init_chat_model(
    MODEL_NAME,
    model_provider="openai",
    base_url=os.getenv("LLM_BASE_URL") or None,
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# 给模型挂上工具
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# ---------- 2. 定义状态 ----------

class MessagesState(TypedDict):
    """Agent 运行时共享的状态。

    messages 用 operator.add 做 reducer：新消息追加到列表末尾，而不是覆盖掉历史。
    """

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ---------- 3. 模型节点 ----------

def llm_call(state: MessagesState) -> dict:
    """调用 LLM，由它决定是发起工具调用还是直接作答。"""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------- 4. 工具节点 ----------

def tool_node(state: MessagesState) -> dict:
    """执行 LLM 上一轮请求的工具调用，把结果回灌成 ToolMessage。"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# ---------- 5. 条件边 ----------

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据最后一条消息是否带工具调用，决定继续循环还是收尾。"""
    messages = state["messages"]
    last_message = messages[-1]

    # LLM 发起了工具调用 -> 先执行工具
    if last_message.tool_calls:
        return "tool_node"

    # 否则直接回复用户，结束
    return END


# ---------- 6. 组装并编译图 ----------

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()


# ---------- 7. 运行 ----------

def main() -> None:
    """用指南里的示例算式跑一遍，打印完整消息轨迹。"""
    messages = [HumanMessage(content="Add 3 and 4.")]
    result = agent.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()
    print(f"\nllm_calls: {result.get('llm_calls')}")


if __name__ == "__main__":
    main()
