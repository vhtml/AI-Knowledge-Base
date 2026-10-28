"""模型调用层：流式输出 + 非流式摘要。

对外暴露两种能力：
- stream()  逐段吐出文本增量，用于终端流式输出
- complete() 一次性拿到完整文本，用于上下文摘要

离线演示模式（--mock）会换成 MockLLMClient，不联网也能跑通整条链路。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

from .context import naive_summary
from .messages import (
    ROLE_SYSTEM,
    ROLE_USER,
    Message,
    truncate_to_tokens,
)

SUMMARY_SYSTEM_PROMPT = """你是一个对话压缩器。
把给定的对话片段压缩成不超过 200 字的要点摘要，保留：
- 用户的目标和已经做出的决定
- 仍然有效的关键事实（名称、数字、路径）
- 尚未解决的问题
不要添加原文中没有的信息，不要写寒暄。直接输出摘要正文。"""


@dataclass
class Delta:
    """流式输出中的一段增量。"""

    text: str
    finish_reason: str | None = None


class LLMClient:
    """基于 OpenAI 兼容接口的客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
        )

    # ------------------------------------------------------------
    # 1. 流式对话
    # ------------------------------------------------------------

    def stream(self, api_messages: list[dict[str, str]]) -> Iterator[Delta]:
        """逐段吐出文本增量。"""

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "stream": True,
        }

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            text = choice.delta.content or ""

            if text:
                yield Delta(text=text)

            if choice.finish_reason:
                yield Delta(text="", finish_reason=choice.finish_reason)

    # ------------------------------------------------------------
    # 2. 非流式对话
    # ------------------------------------------------------------

    def complete(self, api_messages: list[dict[str, str]]) -> str:
        """一次性拿到完整回复，用于摘要等场景。"""

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": 0.2,
        }

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(**kwargs)

        if not response.choices:
            return ""

        return response.choices[0].message.content or ""


class MockLLMClient:
    """离线演示用的假客户端：接口与 LLMClient 一致，但不联网。

    它的回复会"回声"出本次上下文的构成，方便在没有 API Key 时
    观察多轮对话、上下文装配、流式输出和历史存档是否正常工作。
    """

    model = "mock-model"

    def stream(self, api_messages: list[dict[str, str]]) -> Iterator[Delta]:
        last_user = ""

        for item in api_messages:
            if item.get("role") == ROLE_USER:
                last_user = item.get("content", "")

        reply = (
            f"（离线演示）收到 {len(api_messages)} 条上下文消息，"
            f"最近一轮输入是「{last_user[:30]}」。"
            f"流式输出正常，历史已落盘。"
        )

        for ch in reply:
            yield Delta(text=ch)

        yield Delta(text="", finish_reason="stop")

    def complete(self, api_messages: list[dict[str, str]]) -> str:
        return f"（离线摘要）共 {len(api_messages)} 条消息的摘要。"


def make_summarizer(client, max_summary_tokens: int = 200) -> callable:
    """构造一个压缩器：把旧消息折叠成摘要。

    模型不一定会遵守"不超过 200 字"的要求，所以本地再截一次，
    避免摘要膨胀、把上下文预算吃光。

    模型调用失败时降级为 context.naive_summary，保证压缩流程不会中断。
    """

    def summarize(
        old_messages: list[Message], previous_summary: Message | None
    ) -> str:
        lines: list[str] = []

        if previous_summary is not None and previous_summary.content.strip():
            lines.append("已有摘要：")
            lines.append(previous_summary.content.strip())
            lines.append("")

        lines.append("需要压缩的对话片段：")

        for message in old_messages:
            lines.append(f"{message.label}：{message.content}")

        transcript = "\n".join(lines)

        api_messages = [
            {"role": ROLE_SYSTEM, "content": SUMMARY_SYSTEM_PROMPT},
            {"role": ROLE_USER, "content": transcript},
        ]

        try:
            result = client.complete(api_messages)

            if result and result.strip():
                return truncate_to_tokens(result.strip(), max_summary_tokens)

        except Exception as exc:  # 摘要失败不能影响主流程
            print(f"[摘要] 模型调用失败，降级为本地摘要：{type(exc).__name__}")

        return naive_summary(old_messages, previous_summary)

    return summarize


def load_env_config() -> dict[str, str]:
    """从环境变量读取模型配置（.env 由调用方加载）。"""

    return {
        "base_url": os.getenv("LLM_BASE_URL") or "",
        "api_key": os.getenv("LLM_API_KEY") or "",
        "model": os.getenv("LLM_MODEL") or "gpt-4o-mini",
    }
