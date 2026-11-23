"""8 章实验模块登记表。

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
    """命令行里用的键，例如 ``ch03``。"""

    title: str
    """章节标题。"""

    module: str
    """模块名，位于本包内。"""

    def load(self) -> ModuleType:
        """按需导入模块——不预先 import，避免某个章节的依赖问题拖垮整包。"""
        return importlib.import_module(f"{__package__}.{self.module}")


LESSONS: tuple[Lesson, ...] = (
    Lesson("ch01", "模型调用基础与消息协议", "ch01_model_basics"),
    Lesson("ch02", "工具调用协议", "ch02_tool_calling"),
    Lesson("ch03", "结构化输出", "ch03_structured_output"),
    Lesson("ch04", "流式输出", "ch04_streaming"),
    Lesson("ch05", "推理模型与思考预算", "ch05_reasoning"),
    Lesson("ch06", "多模态输入输出", "ch06_multimodal"),
    Lesson("ch07", "上下文窗口、定价与缓存", "ch07_context_pricing_cache"),
    Lesson("ch08", "模型选型与迁移纪律", "ch08_model_selection"),
)
