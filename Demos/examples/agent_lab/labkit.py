"""Agent 开发教程的公共工具层——不依赖任何 Agent 框架。

这个模块解决什么问题：17 章实验反复要用到同一批"底层设施"——消息协议、
工具协议、模型抽象、可切换的模型实现（脚本化假模型 / 真实 OpenAI 兼容端点）、
统一的输出与断言工具。集中在这里，各章实验只写当天的教学重点。

**这套公共层刻意不引入 LangChain 或 LangGraph**：整套教程的目标是先把
Agent 运行时从头手写一遍，看清每一层的职责，再在第 17 章映射回框架。
一旦从第一行就依赖框架，读者就永远看不清"框架替我做了什么"。

两条运行路径，区别只有模型从哪来：

- **确定性路径**（默认）：用 :class:`ScriptedModel` 按脚本回放回复，
  不联网、结果可复现，断言可以写得很死。
- **真实路径**（``--live`` / ``AGENT_LAB_LIVE=1``）：用本机 ``Demos/.env``
  里配置的 OpenAI 兼容端点真调模型，断言只校验"形状"不校验"内容"。
"""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "AgentModel",
    "ChatModel",
    "Message",
    "ModelReply",
    "OpenAICompatibleModel",
    "ScriptedModel",
    "Tool",
    "ToolCall",
    "ai",
    "banner",
    "ensure",
    "human",
    "live_enabled",
    "real_model",
    "repo_root",
    "rule",
    "scripted_model",
    "section",
    "system",
    "tool",
    "tool_result",
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

    默认关闭。命令行 ``--live`` 或环境变量 ``AGENT_LAB_LIVE=1`` 打开。
    """
    return os.getenv("AGENT_LAB_LIVE", "").strip().lower() in {"1", "true", "yes"}


def real_model(temperature: float = 0.0, **kwargs: Any) -> OpenAICompatibleModel:
    """构造一个连到本机 OpenAI 兼容端点的模型。

    配置缺失时抛 :class:`RuntimeError`，并把该配什么写清楚——比让它在
    HTTP 层报 401 更好排查。
    """
    _load_dotenv_once()
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
            ("LLM_MODEL", model_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "真实模型不可用：Demos/.env 缺少 " + ", ".join(missing) + "。"
            " 参考 Demos/.env.example 补齐后重试。"
        )
    return OpenAICompatibleModel(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------- 消息协议


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """模型提出的一次工具调用。

    和多数 SDK 一致，这里把"调用意图"和数据分开：
    ``id`` 用于把工具结果回填给对应的调用，``arguments`` 是已经解析好的字典
    （解析这一步是第 2 章要处理的失败点之一）。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """一条对话消息。

    为什么自己定义而不是直接用某个 SDK 的类型：教程要先把"消息是什么"
    讲清楚，再看各家 SDK 怎么映射。真实工程里你会用 SDK 的类型，
    但结构是一样的——``role`` + ``content`` + 可选的 ``tool_calls``。
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        """转成 OpenAI 兼容接口的请求体格式。"""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


def system(text: str) -> Message:
    """构造一条系统消息。"""
    return Message(role="system", content=text)


def human(text: str) -> Message:
    """构造一条用户消息。"""
    return Message(role="user", content=text)


def ai(text: str, tool_calls: Sequence[ToolCall] | None = None) -> Message:
    """构造一条助手消息，可带工具调用。"""
    return Message(role="assistant", content=text, tool_calls=list(tool_calls or []))


def tool_result(
    content: str,
    tool_call_id: str,
    name: str | None = None,
) -> Message:
    """构造一条工具结果消息。

    ``tool_call_id`` 不可省：模型靠它把结果和之前那次调用意图对上。
    """
    return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


# ---------------------------------------------------------------- 工具协议


@dataclass
class Tool:
    """一个可被模型调用的工具。

    ``parameters`` 是 JSON Schema，模型只看得到它和 ``description``——
    也就是说，工具能不能被正确调用，取决于这两样写得够不够清楚。
    第 3 章整章都在讲这件事。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_api(self) -> dict[str, Any]:
        """转成 OpenAI 兼容接口的 tools 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: dict[str, Any]) -> str:
        """执行工具并把结果转成字符串。

        异常**不在这里**吞掉，而是冒给调用方——第 10 章会讲清楚
        "该在哪一层处理工具异常"这个容易做错决定。
        """
        result = self.func(**arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


_ARG_SECTION = re.compile(r"Args:\s*\n(?P<body>(?:[ \t]+.*\n?)+)", re.MULTILINE)


def _parse_arg_descriptions(doc: str) -> dict[str, str]:
    """从 docstring 的 ``Args:`` 段落里抽出"参数名 → 参数说明"。

    这里手写一个极简解析器，而不是引入框架，是为了让第 3 章能看清：
    **工具的参数说明本来就该写在 docstring 里**，框架只是帮你搬进 schema。
    """
    match = _ARG_SECTION.search(doc or "")
    if not match:
        return {}
    descriptions: dict[str, str] = {}
    for line in match.group("body").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        name, _, desc = stripped.partition(":")
        name = name.split("(")[0].strip()
        if name:
            descriptions[name] = desc.strip()
    return descriptions


def _json_schema_for(func: Callable[..., Any]) -> dict[str, Any]:
    """根据函数签名和类型注解拼一份 JSON Schema。

    只认几种常见类型——教学环境够用，且能让读者看清 schema 是怎么来的。
    """
    type_map: dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    signature = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    arg_docs = _parse_arg_descriptions(doc)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in {"self", "cls"} or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = param.annotation
        json_type = type_map.get(annotation, "string")
        entry: dict[str, Any] = {"type": json_type}
        if name in arg_docs:
            entry["description"] = arg_docs[name]
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            required.append(name)
        properties[name] = entry

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(func: Callable[..., Any]) -> Tool:
    """把一个普通函数变成模型可调用的工具。

    类型来自注解、说明来自 docstring——这两样是模型能看到的全部。
    """
    signature = inspect.signature(func)
    return Tool(
        name=func.__name__,
        description=(inspect.getdoc(func) or "").strip(),
        parameters=_json_schema_for(func),
        func=func,
    )


# ---------------------------------------------------------------- 模型抽象


@dataclass
class ModelReply:
    """模型的一次回复。

    ``content`` 与 ``tool_calls`` 二选一或并存：一轮里模型可以既说话
    又要调工具（并行工具调用就长这样）。
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def to_message(self) -> Message:
        """把回复转成一条助手消息。"""
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=list(self.tool_calls),
        )


@runtime_checkable
class ChatModel(Protocol):
    """模型协议：只看这两个方法，不关心背后是谁。

    这就是第 1 章"最小内核"里说的那层边界——只要模型实现 ``complete``，
    整套运行时就能换模型、换供应商、换成假模型。
    """

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """给定历史消息与可用工具，返回一次模型回复。"""
        ...


@dataclass
class ScriptedModel:
    """按脚本依次回放回复的假模型。

    为什么不用框架自带的 fake model：它们多数没实现工具调用协议，
    而这套教程从头到尾都要能离线跑通"模型要调工具"这条链路。
    这里把 ``tool_calls`` 原样回放，行为和真模型一致。
    """

    script: list[ModelReply]
    cursor: int = 0
    calls: list[list[Message]] = field(default_factory=list)
    """每次调用收到的消息历史快照，用于断言"模型当时看到了什么"。"""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """回放脚本里的下一条回复。

        脚本用尽后重复最后一条——这样"多跑一轮"的场景不会直接崩掉，
        而是可以观察到"模型开始重复自己"这个典型故障。
        """
        if not self.script:
            raise RuntimeError("ScriptedModel 没有可用脚本")
        self.calls.append(list(messages))
        index = min(self.cursor, len(self.script) - 1)
        self.cursor += 1
        return self.script[index]


def scripted_model(script: Sequence[ModelReply | str]) -> ScriptedModel:
    """用一组回复或字符串造一个脚本化模型。"""
    replies: list[ModelReply] = []
    for item in script:
        replies.append(item if isinstance(item, ModelReply) else ModelReply(content=str(item)))
    return ScriptedModel(script=replies)


@dataclass
class OpenAICompatibleModel:
    """连到 OpenAI 兼容端点的真实模型。

    只依赖 ``openai`` SDK，不依赖任何 Agent 框架——这样第 2 章的运行时
    才能宣称自己是"从零手写"的。
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> ModelReply:
        """调一次模型并把回复转成 :class:`ModelReply`。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [t.to_api() for t in tools]
            payload["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        raw = choice.message

        calls: list[ToolCall] = []
        for item in getattr(raw, "tool_calls", None) or []:
            try:
                arguments = json.loads(item.function.arguments or "{}")
            except json.JSONDecodeError:
                # 解析失败是一个真实故障面，第 2、10 章会重点处理
                arguments = {"__raw__": item.function.arguments}
            calls.append(
                ToolCall(
                    id=item.id,
                    name=item.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {"__raw__": arguments},
                )
            )

        usage: dict[str, int] = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return ModelReply(content=raw.content or "", tool_calls=calls, usage=usage)


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


def approx_tokens(text: str) -> int:
    """粗估一段文本的 token 数。

    中英混排的粗估：英文按 4 字符 1 token，中文按 1 字 1 token。
    只用于教学里的"相对大小"比较，不追求精确。
    """
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - chinese
    return chinese + (other + 3) // 4
