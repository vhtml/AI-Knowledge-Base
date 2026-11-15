"""LangGraph 教程实验的公共工具层。

这个模块解决什么问题：17 章实验反复要用到同一批"底层设施"——可切换的模型工厂
（真实模型 / 脚本化假模型）、临时的检查点与存储实例、统一的输出与断言工具。
把它们集中在这里，各章实验只写当天的教学重点，不重复造轮子。

两条运行路径，区别只有模型从哪来：

- **确定性路径**（默认）：用 :class:`ScriptedChatModel` 按脚本回放消息，
  不联网、结果可复现，断言可以写得很死。
- **真实路径**（``--live`` / ``LANGGRAPH_LAB_LIVE=1``）：用本机 ``Demos/.env``
  里配置的 OpenAI 兼容端点真调模型，断言只校验"形状"不校验"内容"。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

__all__ = [
    "ScriptedChatModel",
    "ai",
    "banner",
    "ensure",
    "human",
    "live_enabled",
    "real_chat_model",
    "repo_root",
    "rule",
    "scripted_model",
    "section",
    "sqlite_checkpointer",
    "tool_call",
]


# ---------------------------------------------------------------- 路径与环境


def repo_root() -> Path:
    """定位仓库根目录：从本文件向上找带 AGENTS.md 的目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


def _load_dotenv_once() -> None:
    """加载 ``Demos/.env``。

    真实模型走的是 OpenAI 兼容接口，配置项沿用仓库既有约定：
    ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 依赖缺失时退化成"没有真实模型"
        return
    env_path = repo_root() / "Demos" / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def live_enabled() -> bool:
    """当前是否允许调用真实模型。

    默认关闭。命令行 ``--live`` 或环境变量 ``LANGGRAPH_LAB_LIVE=1`` 打开。
    """
    return os.getenv("LANGGRAPH_LAB_LIVE", "").strip().lower() in {"1", "true", "yes"}


def real_chat_model(temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    """构造一个连到本机 OpenAI 兼容端点的对话模型。

    配置缺失时抛 :class:`RuntimeError`，并把该配什么写清楚——比让它在
    HTTP 层报 401 更好排查。
    """
    _load_dotenv_once()
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "真实模型不可用：Demos/.env 缺少 " + ", ".join(missing) + "。"
            " 参考 Demos/.env.example 补齐后重试。"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------- 消息构造


def human(text: str) -> HumanMessage:
    """构造一条用户消息。"""
    return HumanMessage(content=text)


def ai(text: str) -> AIMessage:
    """构造一条不带工具调用的助手消息。"""
    return AIMessage(content=text)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """构造一条"只带工具调用、不带正文"的助手消息。

    脚本化模型用它模拟模型决定调用工具的那一步。
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ---------------------------------------------------------------- 脚本化模型


class ScriptedChatModel(BaseChatModel):
    """按脚本依次回放消息的假模型。

    为什么不用 ``langchain_core`` 自带的 ``GenericFakeChatModel``：它没实现
    ``bind_tools``，Agent 类实验一调工具就抛 ``NotImplementedError``。
    这里补上这一层，脚本里的 ``AIMessage`` 原样返回，包括 ``tool_calls``。
    """

    responses: list[AnyMessage]
    """脚本内容。最后一条会被循环使用，避免脚本耗尽时直接崩掉。"""

    cursor: int = 0
    """当前读到脚本的第几条。"""

    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise RuntimeError("ScriptedChatModel 没有可用脚本")
        index = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        message = self.responses[index]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> ScriptedChatModel:
        """假模型不需要真的知道工具长什么样，直接返回自己。"""
        return self


def scripted_model(responses: Sequence[BaseMessage | str]) -> ScriptedChatModel:
    """用一组消息或字符串造一个脚本化模型。"""
    messages: list[AnyMessage] = []
    for item in responses:
        messages.append(item if isinstance(item, BaseMessage) else ai(str(item)))
    return ScriptedChatModel(responses=messages)


# ---------------------------------------------------------------- 检查点


@contextmanager
def sqlite_checkpointer(path: Path | None = None) -> Iterator[Any]:
    """开一个 SQLite 检查点器，退出时自动关闭。

    持久化章节要用它证明"进程重启后状态还在"。传 ``path=None`` 时用临时文件，
    退出即删。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    if path is None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.close()
        target = Path(tmp.name)
    else:
        target = path
        target.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(target)) as saver:
        yield saver


# ---------------------------------------------------------------- 输出工具


def banner(title: str) -> None:
    """打印章节标题。"""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def section(text: str) -> None:
    """打印小节标题。"""
    print()
    print(f"--- {text} ---")


def rule(text: str = "") -> None:
    """打印一条分隔线，可带一句说明。"""
    print(f"\n· {text}" if text else "")


def ensure(condition: bool, message: str) -> None:
    """断言辅助：条件不成立就抛 :class:`AssertionError`。

    报错信息用中文写清楚"期望什么、实际什么"，比裸 ``assert`` 好读。
    """
    if not condition:
        raise AssertionError(message)
