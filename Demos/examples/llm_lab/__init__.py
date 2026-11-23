"""LLM 地基教程实验包。

8 章实验，贴着模型 API 的线上协议写，与 ``AI/LLM/`` 下的笔记一一对应。

整套实验只用标准库加 ``openai`` SDK，不引入 LangChain / LangGraph——
目的是先把模型调用这一层看清，再谈 Agent。

用法：
    cd Demos/examples
    uv run --project .. python -m llm_lab list
    uv run --project .. python -m llm_lab ch03
    uv run --project .. python -m llm_lab all
    uv run --project .. python -m llm_lab selftest
"""

from __future__ import annotations

__all__ = ["labkit", "lessons"]
