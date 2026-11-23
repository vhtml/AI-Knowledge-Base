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
    Lesson("ch01", "什么是 Agent：定义、边界与最小内核", "ch01_agent_basics"),
    Lesson("ch02", "Agent 循环：ReAct 的手写实现", "ch02_react_loop"),
    Lesson("ch03", "工具设计：schema、错误面与粒度", "ch03_tool_design"),
    Lesson("ch04", "系统提示与角色设计", "ch04_system_prompt"),
    Lesson("ch05", "上下文工程：窗口预算与压缩", "ch05_context_engineering"),
    Lesson("ch06", "记忆系统：短期、长期与检索", "ch06_memory"),
    Lesson("ch07", "任务分解：规划模式与自适应分解", "ch07_planning"),
    Lesson("ch08", "反思与自纠错", "ch08_reflection"),
    Lesson("ch09", "多智能体编排模式", "ch09_multi_agent"),
    Lesson("ch10", "可靠性：重试、超时、幂等与降级", "ch10_reliability"),
    Lesson("ch11", "可观测性：事件模型、追踪与回放", "ch11_observability"),
    Lesson("ch12", "评测体系：黄金集、轨迹评测与回归", "ch12_evaluation"),
    Lesson("ch13", "安全：提示注入、沙箱与最小权限", "ch13_security"),
    Lesson("ch14", "人在环路：审批、接管与恢复", "ch14_hitl"),
    Lesson("ch15", "成本与延迟工程", "ch15_cost_latency"),
    Lesson("ch16", "生产化：部署形态与上线检查单", "ch16_production"),
    Lesson("ch17", "框架地图：从手写运行时到 LangGraph/Deep Agents", "ch17_framework_map"),
)
