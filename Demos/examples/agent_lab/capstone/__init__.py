"""毕业作品：一个把 17 章护栏装在一起的研究助手 Agent。

配套笔记在 [`AI/Agent/`](../../../../AI/Agent/)。

- :mod:`research_agent` — 主运行时：预算、记忆、风险分级、审批、轨迹、成本
- :mod:`evals` — 黄金集与打分
- :mod:`selftest` — 毕业标准自检

用法：
    cd Demos/examples
    uv run --project .. python -m agent_lab.capstone.selftest
"""

from __future__ import annotations

__all__ = ["evals", "research_agent", "selftest"]
