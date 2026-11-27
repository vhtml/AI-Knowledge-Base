"""章节实验登记表。

这张表是实验的调度中枢：``__main__`` 按它分发命令，
``selftest`` 按它逐章跑断言。加一章只需要在这里加一行。

只收录**能对着成片跑出结论**的章节实验——检查器查的是
``capstone/fourier_C/`` 里的真片，不是样例数据。
其余章节（片型解剖、形象优先、文案体检、两套外观、声音、工具选型）
是纯判断标准，留在 ``AI/Knowledge-Film/`` 的笔记里，不占代码区的位置。
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
    Lesson("ch03", "事实表：每个数字都要有出处", "ch03_facts"),
    Lesson("ch05", "分镜表：唯一的施工图", "ch05_storyboard"),
    Lesson("ch07", "渲染引擎：画面必须是时间的纯函数", "ch07_engine"),
    Lesson("ch08", "时间轴与卡点", "ch08_timeline"),
    Lesson("ch10", "工程管线：先渲片段，再渲全片", "ch10_pipeline"),
    Lesson("ch11", "终检：八道门槛", "ch11_qc"),
)
