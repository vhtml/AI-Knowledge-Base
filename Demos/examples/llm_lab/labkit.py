"""LLM 地基教程的公共工具层——贴着线上协议写，不做 Agent 层的抽象。

这个模块解决什么问题：8 章实验反复要用到同一批"底层设施"——聊天请求与响应
的消息协议、工具声明、脚本化假模型、真实 OpenAI 兼容端点、流式分片、
token 估算与成本计算、以及一个极简的 JSON Schema 校验器。

**和 ``agent_lab.labkit`` 的分工**：那套讲的是"Agent 运行时怎么组织消息"，
抽象在语义层；这套讲的是"模型 API 到底怎么被调用"，贴着协议层。
所以这里的类型刻意和厂商 SDK 的原始结构一一对应（role / content /
tool_calls / usage / finish_reason），方便你把本教程的代码直接对上真实 SDK。

两条运行路径，区别只有模型从哪来：

- **确定性路径**（默认）：用 :class:`ScriptedModel` 回放响应，不联网、可复现，
  断言可以写得很死。
- **真实路径**（``--live`` / ``LLM_LAB_LIVE=1``）：用本机 ``Demos/.env`` 里
  配置的 OpenAI 兼容端点真调模型，断言只校验"形状"。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ModelClient",
    "OpenAICompatibleClient",
    "PRICING",
    "ScriptedClient",
    "StreamChunk",
    "ToolSpec",
    "banner",
    "ensure",
    "estimate_cost",
    "estimate_tokens",
    "extract_json",
    "live_enabled",
    "real_client",
    "repo_root",
    "rule",
    "satisfies_client_protocol",
    "scripted_client",
    "section",
    "tool_spec",
    "validate_against_schema",
]


# ---------------------------------------------------------------- 路径与环境


def repo_root() -> Path:
    """定位仓库根目录：从本文件向上找带 AGENTS.md 的目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


def _load_dotenv_once() -> None:
    """加载 ``Demos/.env``（沿用仓库既有约定）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 依赖缺失时退化成"没有真实模型"
        return
    env_path = repo_root() / "Demos" / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def live_enabled() -> bool:
    """当前是否允许调用真实模型。"""
    return os.getenv("LLM_LAB_LIVE", "").strip().lower() in {"1", "true", "yes"}


def real_client(temperature: float = 0.0, **kwargs: Any) -> OpenAICompatibleClient:
    """构造一个连到本机 OpenAI 兼容端点的客户端。

    配置缺失时抛 :class:`RuntimeError`，并把该配什么写清楚。
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
    return OpenAICompatibleClient(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------- 协议类型


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    """一条聊天消息——和厂商 SDK 的消息结构一一对应。

    ``content`` 可能是字符串，也可能是多模态内容块列表（第 6 章会展开）。
    """

    role: Role
    content: Any = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        """转成请求体格式。"""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ToolSpec:
    """一个工具声明。

    注意 ``parameters`` 是**发给模型的 JSON Schema**——
    模型就是照着它生成调用意图的。
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )

    def to_api(self) -> dict[str, Any]:
        """转成请求体的 tools 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_spec(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> ToolSpec:
    """造一个工具声明。

    ``parameters`` 是发给模型的 JSON Schema；不给就用最简的空对象 schema。
    """
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters
        if parameters is not None
        else {"type": "object", "properties": {}, "required": []},
    )


@dataclass
class ChatResponse:
    """一次非流式响应。

    ``finish_reason`` 是被低估的字段：它决定了"模型是不是还想说但被截断了"
    以及"是不是因为要调工具所以才停"（第 2、4 章会重点用）。

    ``reasoning`` 是推理模型的思考内容——**默认不要回填进上下文**，
    它既贵又会污染后续判断（第 5 章）。
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    reasoning: str = ""
    """推理模型的思考过程。普通模型为空串。"""


@dataclass
class StreamChunk:
    """一个流式分片。

    ``delta`` 是**增量**内容，不是全量——这是流式最容易搞错的地方。
    """

    delta: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- 模型客户端


class ModelClient:
    """模型客户端协议：只看 ``complete`` 和 ``stream`` 两个方法。

    第 1 章的核心结论就建立在这个边界上——**换模型 = 换一个实现**，
    上层代码一行不用改。
    """

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        """发一次非流式请求。

        ``reasoning_effort`` 是思考预算（``low`` / ``medium`` / ``high``），
        只有推理模型认这个参数；普通模型忽略它（第 5 章）。
        """
        raise NotImplementedError

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[StreamChunk]:
        """发一次流式请求，产出增量分片。"""
        raise NotImplementedError


def satisfies_client_protocol(client: Any) -> bool:
    """检查一个对象是否满足客户端协议——只看有没有那两个方法。

    比 ``isinstance`` 更直白：只要 ``complete`` 和 ``stream`` 都在，
    上层代码就能把它当模型用。
    """
    return callable(getattr(client, "complete", None)) and callable(
        getattr(client, "stream", None)
    )


@dataclass
class ScriptedClient:
    """按脚本回放响应的假客户端。

    ``responses`` 是非流式脚本；``streams`` 是流式脚本，每个元素是一串分片。
    两者分开是因为"流式"和"非流式"的行为差异本身就是要教的内容（第 4 章）。
    """

    responses: list[ChatResponse] = field(default_factory=list)
    streams: list[list[StreamChunk]] = field(default_factory=list)
    cursor: int = 0
    seen_messages: list[list[ChatMessage]] = field(default_factory=list)
    seen_calls: list[dict[str, Any]] = field(default_factory=list)
    """每次调用的完整参数快照——用于断言"模型到底收到了什么旋钮"。"""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        """回放脚本里的下一个响应；脚本用尽后重复最后一条。"""
        if not self.responses:
            raise RuntimeError("ScriptedClient 没有可用脚本")
        self.seen_messages.append(list(messages))
        self.seen_calls.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "temperature": temperature,
                "response_format": response_format,
                "reasoning_effort": reasoning_effort,
            }
        )
        index = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        return self.responses[index]

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamChunk]:
        """回放脚本里的下一组流式分片。"""
        if not self.streams:
            raise RuntimeError("ScriptedClient 没有可用流式脚本")
        self.seen_messages.append(list(messages))
        index = min(self.cursor, len(self.streams) - 1)
        self.cursor += 1
        yield from self.streams[index]


def scripted_client(
    responses: Sequence[ChatResponse] | None = None,
    streams: Sequence[Sequence[StreamChunk]] | None = None,
) -> ScriptedClient:
    """造一个脚本化客户端。"""
    return ScriptedClient(
        responses=list(responses or []),
        streams=[list(s) for s in (streams or [])],
    )


@dataclass
class OpenAICompatibleClient:
    """连到 OpenAI 兼容端点的真实客户端。

    只依赖 ``openai`` SDK，不依赖任何 Agent 框架——第 1 章要讲的就是
    "这一层可以有多薄"。
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
        temperature: float | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [t.to_api() for t in tools]
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        """发一次非流式请求。"""
        payload = self._payload(messages, tools, temperature, response_format, reasoning_effort)
        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        raw = choice.message
        usage: dict[str, int] = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
            details = getattr(response.usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens:
                # 推理 token 计入输出计费——账单上不会单独列，但成本在里面
                usage["reasoning_tokens"] = int(reasoning_tokens)

        # 推理模型的思考内容在不同厂商里字段名不一样，都试一遍
        reasoning = (
            getattr(raw, "reasoning_content", None)
            or getattr(raw, "reasoning", None)
            or ""
        )

        return ChatResponse(
            content=raw.content or "",
            tool_calls=[c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in (getattr(raw, "tool_calls", None) or [])],
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=response.model or self.model,
            reasoning=str(reasoning),
        )

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[StreamChunk]:
        """发一次流式请求，产出增量分片。"""
        payload = self._payload(messages, tools, temperature, None, reasoning_effort)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        stream = self._client.chat.completions.create(**payload)
        for event in stream:
            if not getattr(event, "choices", None):
                if getattr(event, "usage", None):
                    yield StreamChunk(
                        usage={
                            "prompt_tokens": event.usage.prompt_tokens or 0,
                            "completion_tokens": event.usage.completion_tokens or 0,
                            "total_tokens": event.usage.total_tokens or 0,
                        }
                    )
                continue
            choice = event.choices[0]
            delta = getattr(choice.delta, "content", None)
            yield StreamChunk(
                delta=delta or "",
                finish_reason=choice.finish_reason,
            )


# ---------------------------------------------------------------- token 与成本


def estimate_tokens(text: str) -> int:
    """粗估一段文本的 token 数。

    中英混排的粗估：英文按 4 字符 1 token，中文按 1 字 1 token。
    只用于教学里的"相对大小"比较与预算估算，不追求精确。
    """
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - chinese
    return chinese + (other + 3) // 4


# 每 1M token 的价格（美元）。数值随厂商调价会变，
# 教学用途下请以厂商官网当日价格为准。
PRICING: dict[str, tuple[float, float]] = {
    "cheap-small": (0.15, 0.60),
    "cheap": (0.25, 1.25),
    "standard": (2.50, 10.00),
    "frontier": (15.00, 75.00),
    "reasoning": (3.00, 15.00),
}
"""价格档位 → (输入单价, 输出单价)，单位是美元 / 每 100 万 token。"""


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> float:
    """按档位估算一次调用的成本（美元）。

    ``model`` 直接用 ``PRICING`` 的键名表示档位。
    """
    if model not in PRICING:
        raise KeyError(f"未知的价格档位：{model}，可选：{sorted(PRICING)}")
    input_price, output_price = PRICING[model]
    return (
        prompt_tokens / 1_000_000 * input_price
        + completion_tokens / 1_000_000 * output_price
    )


# ---------------------------------------------------------------- 结构化输出


_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """从模型输出里抠出 JSON。

    模型即使被要求"只输出 JSON"，也经常包一层 ```` ```json ```` 围栏或加一句
    "好的，结果是："。所以解析端必须容错——这是第 3 章的核心结论之一。
    """
    candidates = [text.strip()]
    match = _FENCE.search(text)
    if match:
        candidates.append(match.group("body").strip())

    brace = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法从模型输出中解析出 JSON：{text[:200]!r}")


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_against_schema(
    data: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """极简 JSON Schema 校验，返回错误列表（空列表 = 通过）。

    只支持教学里最常用的一小撮关键字：
    ``type`` / ``required`` / ``properties`` / ``enum`` / ``items`` /
    ``additionalProperties`` / ``minimum`` / ``maximum``。

    不追求覆盖 JSON Schema 规范全集——够用、能讲清楚"模型输出哪里错了"，
    这才是这一层的目的。真上生产请用 ``jsonschema`` 或 Pydantic。
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected:
        expected_types = _TYPE_MAP.get(expected, ())
        if expected_types:
            # bool 是 int 的子类，但 JSON 里 true 不是 number
            if expected == "number" and isinstance(data, bool):
                errors.append(f"{path}: 期望 number，实际 boolean")
            elif not isinstance(data, expected_types):
                errors.append(f"{path}: 期望 {expected}，实际 {type(data).__name__}")
                return errors

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: 值 {data!r} 不在允许的取值 {schema['enum']} 里")

    if isinstance(data, dict):
        for name in schema.get("required", []):
            if name not in data:
                errors.append(f"{path}.{name}: 缺少必填字段")
        properties = schema.get("properties", {})
        for name, value in data.items():
            if name in properties:
                errors.extend(validate_against_schema(value, properties[name], f"{path}.{name}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{name}: 不允许的额外字段")

    if isinstance(data, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(data):
            errors.extend(validate_against_schema(item, schema["items"], f"{path}[{index}]"))

    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: {data} 小于最小值 {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: {data} 大于最大值 {schema['maximum']}")

    return errors


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
