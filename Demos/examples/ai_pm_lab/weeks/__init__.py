"""八周实验模块登记表。

这张表是实验的调度中枢：``__main__`` 按它分发命令，``selftest`` 按它逐周跑断言。
加一周只需要在这里加一行 + 新建一个模块，不用改别的地方。
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["WEEKS", "load_week"]

# (键, 标题, 模块名)
WEEKS: tuple[tuple[str, str, str], ...] = (
    ("week1", "能力雷达与技术边界速查卡", "week1"),
    ("week2", "场景筛选与需求优先级", "week2"),
    ("week3", "竞品地图与定价锚点", "week3"),
    ("week4", "信任式回答与检索策略评估", "week4"),
    ("week5", "自动化等级与工具权限", "week5"),
    ("week6", "评测门禁与单位经济", "week6"),
    ("week7", "风险登记册与指标树", "week7"),
    ("week8", "毕业作品校验", "week8"),
)


def load_week(key: str) -> ModuleType | None:
    """按键导入某周的模块；找不到返回 None。

    按需导入而不是预先 import 全部模块，是为了让某一周的依赖问题只影响那一周，
    而不是拖垮整包——`all` 要能一周一周地跑完。
    """
    for week_key, _, module_name in WEEKS:
        if week_key == key:
            return importlib.import_module(f"{__package__}.{module_name}")
    return None
