"""18 章实验模块登记表。

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
    Lesson("ch01", "推理服务全景与 SLO", "ch01_slo"),
    Lesson("ch02", "GPU 规格与显存账本", "ch02_gpu_memory"),
    Lesson("ch03", "Roofline 与算术强度", "ch03_roofline"),
    Lesson("ch04", "访存墙与批量摊薄", "ch04_memory_wall"),
    Lesson("ch05", "KV Cache：显存账与复用", "ch05_kv_cache"),
    Lesson("ch06", "Attention 变体与 PagedAttention 原理", "ch06_attention_paging"),
    Lesson("ch07", "连续批处理与调度内核", "ch07_continuous_batching"),
    Lesson("ch08", "多租户调度与 SLA", "ch08_multitenancy"),
    Lesson("ch09", "量化：精度边界与成本收益", "ch09_quantization"),
    Lesson("ch10", "推测解码与时延收益", "ch10_speculative"),
    Lesson("ch11", "网关：路由、限流与租户隔离", "ch11_gateway"),
    Lesson("ch12", "缓存体系：前缀、语义与响应缓存", "ch12_caching"),
    Lesson("ch13", "提示词流水线与长上下文混批", "ch13_prompt_pipeline"),
    Lesson("ch14", "可观测性与延迟归因", "ch14_observability"),
    Lesson("ch15", "弹性：自动扩缩容与冷启动", "ch15_elasticity"),
    Lesson("ch16", "容量规划与上线检查单", "ch16_capacity"),
    Lesson("ch17", "训练侧 Infra 地基", "ch17_training_ground"),
    Lesson("ch18", "生态地图：框架选型与机制对照", "ch18_ecosystem"),
)
