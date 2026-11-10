"""三个生活工具：天气、计算、时间。

每个工具都自带一份 JSON Schema，这份 schema 同时干三件事：
1. 交给模型，让它知道该传什么参数（工具选择的一部分）；
2. 交给校验器，挡掉不合法参数（参数校验）；
3. 交给修复逻辑，作为"该往哪个方向改"的依据。

handler 只负责"拿到合法参数之后把活干完"——参数已经过校验层，
工具内部不再重复校验，省得两套规则说不一样的话。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import calc, providers
from .models import ToolPayload

WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "要查询的城市名，用常见的中文写法，例如「北京」「上海」。",
            "minLength": 1,
        },
        "day": {
            "type": "string",
            "description": "查哪一天。",
            "enum": ["today", "tomorrow"],
            "default": "today",
        },
        "unit": {
            "type": "string",
            "description": "温度单位。",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius",
        },
    },
    "required": ["city"],
    "additionalProperties": False,
}

CALCULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "要计算的数学表达式，例如「(12+8)*3」「2**10」「sqrt(16)」"
                "「max(3, 7)」。"
            ),
            "minLength": 1,
            "maxLength": 200,
        }
    },
    "required": ["expression"],
    "additionalProperties": False,
}

TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "IANA 时区名，例如「Asia/Shanghai」「America/New_York」。",
        },
        "city": {
            "type": "string",
            "description": "城市名，等价于查这个城市所在时区的时间，例如「东京」。",
        },
        "format": {
            "type": "string",
            "description": "输出格式。",
            "enum": ["readable", "iso", "date", "time", "timestamp"],
            "default": "readable",
        },
        "offset_days": {
            "type": "integer",
            "description": "相对今天偏移的天数，0 表示今天，-1 表示昨天，3 表示三天后。",
            "minimum": -365,
            "maximum": 365,
            "default": 0,
        },
    },
    "required": [],
    "additionalProperties": False,
}


@dataclass
class Tool:
    """一个工具：描述 + 参数 schema + 执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolPayload]

    def to_openai(self) -> dict[str, Any]:
        """转成 chat.completions 的 tools 数组元素。

        校验用的 schema 里有 minLength / minimum 这类约束，
        部分接口不接受这些关键字，所以发给模型之前把它们折进描述文本里——
        模型仍然知道边界在哪，接口也不会因为 schema 不认识而报错。
        """

        properties: dict[str, Any] = {}

        for name, spec in self.parameters.get("properties", {}).items():
            description = str(spec.get("description", "")).strip()
            constraints = describe_constraints(spec)

            if constraints:
                description = f"{description}（{constraints}）" if description else constraints

            properties[name] = {
                "type": spec.get("type", "string"),
                "description": description,
            }

            if spec.get("enum"):
                properties[name]["enum"] = list(spec["enum"])

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(self.parameters.get("required", [])),
                    "additionalProperties": False,
                },
            },
        }


def describe_constraints(spec: dict[str, Any]) -> str:
    """把 schema 里的约束写成一句人话。"""

    parts: list[str] = []

    if spec.get("enum"):
        parts.append("取值：" + " / ".join(str(item) for item in spec["enum"]))

    if spec.get("minimum") is not None or spec.get("maximum") is not None:
        low = spec.get("minimum")
        high = spec.get("maximum")
        parts.append(f"取值范围：{low} ~ {high}")

    if spec.get("minLength") is not None:
        parts.append(f"至少 {spec['minLength']} 个字符")

    if spec.get("maxLength") is not None:
        parts.append(f"最多 {spec['maxLength']} 个字符")

    return "，".join(parts)


def _handle_weather(provider: providers.WeatherProvider, args: dict[str, Any]) -> ToolPayload:
    """天气工具：查某座城市今天或明天的天气。"""

    city = str(args["city"]).strip()
    day = str(args.get("day", "today"))
    unit = str(args.get("unit", "celsius"))

    report = provider.get(city=city, day=day, unit=unit)

    return ToolPayload(
        text=report.render(),
        data={
            "city": report.city,
            "day": report.day,
            "condition": report.condition,
            "temperature": report.temperature,
            "unit": report.unit,
            "source": report.source,
        },
    )


def _handle_calculate(args: dict[str, Any]) -> ToolPayload:
    """计算工具：求一个数学表达式的值。"""

    expression = str(args["expression"]).strip()
    value = calc.evaluate(expression)

    return ToolPayload(
        text=f"{expression} = {calc.format_number(value)}",
        data={"expression": expression, "value": value},
    )


def _handle_get_time(args: dict[str, Any]) -> ToolPayload:
    """时间工具：查某时区的当前时间，或相对今天偏移几天的日期。"""

    timezone = args.get("timezone") or None
    city = args.get("city") or None
    offset_days = int(args.get("offset_days", 0))
    output_format = str(args.get("format", "readable"))

    name, moment = providers.current_moment(timezone, city, offset_days)
    text = providers.format_moment(moment, output_format)

    if city:
        scope = f"{city}（{name}）"
    elif timezone:
        scope = name
    else:
        scope = f"本机时区（{name}）"

    return ToolPayload(
        text=f"{scope} 的时间是 {text}",
        data={
            "timezone": name,
            "moment": moment.isoformat(),
            "format": output_format,
            "offset_days": offset_days,
        },
    )


class ToolRegistry:
    """工具清单：按名字取工具，也能一次性导出成 API 要的格式。"""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        """按名字取一个工具，取不到返回 None。"""

        return self._tools.get(name)

    def names(self) -> list[str]:
        """所有工具名。"""

        return list(self._tools)

    def to_openai(self) -> list[dict[str, Any]]:
        """导出成 chat.completions 的 tools 数组。"""

        return [tool.to_openai() for tool in self._tools.values()]

    def describe(self) -> str:
        """给人看的工具清单。"""

        return "、".join(self._tools)


def build_registry(weather_provider: providers.WeatherProvider) -> ToolRegistry:
    """装配三个工具。

    天气工具的数据源是注入进来的：在线用 Open-Meteo，
    离线用假数据，其它两层完全不用改。
    """

    return ToolRegistry(
        [
            Tool(
                name="get_weather",
                description="查询某个城市今天或明天的天气（温度、体感、湿度、风速）。",
                parameters=WEATHER_SCHEMA,
                handler=lambda args: _handle_weather(weather_provider, args),
            ),
            Tool(
                name="calculate",
                description="计算一个数学表达式的值，例如「(12+8)*3」。",
                parameters=CALCULATE_SCHEMA,
                handler=_handle_calculate,
            ),
            Tool(
                name="get_time",
                description="查询当前时间、某个时区的时间，或相对今天偏移几天的日期。",
                parameters=TIME_SCHEMA,
                handler=_handle_get_time,
            ),
        ]
    )
