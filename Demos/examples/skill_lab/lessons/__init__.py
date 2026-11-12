"""七天实验的每日 lesson。

每个模块导出 run()，由 __main__.py 统一调度。
"""
from __future__ import annotations

from . import (
    day1_context,
    day2_description,
    day3_freedom,
    day4_resources,
    day5_scripts,
    day6_eval,
    day7_packaging,
)

# 天数 -> lesson 模块，顺序即学习路径
DAYS: list[tuple[str, str, object]] = [
    ("day1", "技能的本质与边界", day1_context),
    ("day2", "触发设计", day2_description),
    ("day3", "正文与自由度分级", day3_freedom),
    ("day4", "资源拆分与渐进式披露", day4_resources),
    ("day5", "脚本化", day5_scripts),
    ("day6", "评测与迭代", day6_eval),
    ("day7", "打包分发与生态", day7_packaging),
]

__all__ = ["DAYS"]
