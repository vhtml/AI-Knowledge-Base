"""AI 产品经理 · 八周系统教程的可运行实验包。

这个包解决什么问题：把教程里的八套方法（能力雷达、场景筛选、竞品地图、信任评测、
自动化分级、评测门禁、风险登记、毕业校验）变成能跑、能断言、能回归的实验。

配套笔记在 ``AI/AI-Product/``。全部实验**离线运行**：不联网、不消耗 token、不调用模型，
断言就写在各周模块里——跑挂了就是没学会。

用法：
    cd Demos/examples
    uv run --project .. python -m ai_pm_lab week1   跑某一周
    uv run --project .. python -m ai_pm_lab all      跑全部八周
    uv run --project .. python -m ai_pm_lab list     列出八周主题
    uv run --project .. python -m ai_pm_lab selftest 跑自检
"""

from __future__ import annotations

from .labkit import CAPSTONE_DIR, FIXTURES_DIR, LAB_ROOT

__all__ = ["LAB_ROOT", "FIXTURES_DIR", "CAPSTONE_DIR"]
