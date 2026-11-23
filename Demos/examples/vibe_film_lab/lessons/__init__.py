"""12 章实验模块登记表。

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
    Lesson("ch01", "片型解剖：先把形态量出来", "ch01_anatomy"),
    Lesson("ch02", "形象优先：先找形象，再写文案", "ch02_metaphor"),
    Lesson("ch03", "事实表：每个数字都要有出处", "ch03_facts"),
    Lesson("ch04", "文案即字幕：写稿就是排时间轴", "ch04_script"),
    Lesson("ch05", "分镜表：唯一的施工图", "ch05_storyboard"),
    Lesson("ch06", "两套外观：夜曲与寓言", "ch06_worlds"),
    Lesson("ch07", "渲染引擎：画面必须是时间的纯函数", "ch07_engine"),
    Lesson("ch08", "时间轴与卡点", "ch08_timeline"),
    Lesson("ch09", "声音：没有人声，节奏靠声音撑", "ch09_audio"),
    Lesson("ch10", "工程管线：先渲片段，再渲全片", "ch10_pipeline"),
    Lesson("ch11", "终检：八道门槛", "ch11_qc"),
    Lesson("ch12", "工具对照与选型", "ch12_tooling"),
)
