"""17 章实验模块登记表。

这张表是实验的调度中枢：``__main__`` 按它分发命令，
``selftest`` 按它逐章跑断言。加一章只需要在这里加一行。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

__all__ = ["LESSONS", "Lesson"]


@dataclass(frozen=True)
class Lesson:
    """一章实验的登记信息。"""

    key: str
    """命令行里用的键，例如 ``ch07``。"""

    title: str
    """章节标题。"""

    module: str
    """模块名，位于本包内。"""

    def load(self) -> ModuleType:
        """按需导入模块——不预先 import，避免某个章节的依赖问题拖垮整包。"""
        return importlib.import_module(f"{__package__}.{self.module}")


LESSONS: tuple[Lesson, ...] = (
    Lesson("ch01", "生态全景与选型", "ch01_ecosystem"),
    Lesson("ch02", "图模型：节点、边与共享状态", "ch02_graph_basics"),
    Lesson("ch03", "状态定义与归约器", "ch03_state_reducers"),
    Lesson("ch04", "控制流：分支、循环、Send 与 Command", "ch04_control_flow"),
    Lesson("ch05", "节点韧性：重试、超时与错误处理", "ch05_node_resilience"),
    Lesson("ch06", "工具调用与工具节点", "ch06_tools"),
    Lesson("ch07", "持久化：检查点与时间旅行", "ch07_persistence"),
    Lesson("ch08", "人在环路：中断、审批与恢复", "ch08_hitl"),
    Lesson("ch09", "流式与事件投影", "ch09_streaming"),
    Lesson("ch10", "子图与多智能体编排", "ch10_subgraphs"),
    Lesson("ch11", "函数式 API 与 Pregel 运行时", "ch11_functional_api"),
    Lesson("ch12", "LangChain 1.x：create_agent 与中间件", "ch12_langchain_agent"),
    Lesson("ch13", "MCP 接入与工具生态", "ch13_mcp"),
    Lesson("ch14", "Deep Agents：harness 视角", "ch14_deep_agents"),
    Lesson("ch15", "可观测性与评测", "ch15_observability_eval"),
    Lesson("ch16", "部署：本地服务器、Studio 与生产", "ch16_deployment"),
    Lesson("ch17", "迁移与工程实践", "ch17_migration_practice"),
)
