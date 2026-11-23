"""Agent 开发教程实验包。

**框架无关**的 17 章动手实验，与 ``AI/Agent/`` 下的笔记一一对应。

整套实验只用标准库加 ``openai`` SDK，不引入 LangChain / LangGraph——
目的是先把 Agent 运行时从头手写一遍，看清每一层的职责。

用法：
    cd Demos/examples
    uv run --project .. python -m agent_lab list
    uv run --project .. python -m agent_lab ch02
    uv run --project .. python -m agent_lab all
    uv run --project .. python -m agent_lab selftest
"""

from __future__ import annotations

__all__ = ["labkit", "lessons", "runtime"]
