"""确定性路由：不联网也能选工具、也能修参数。

两个用途：
1. 离线兜底——没有 API Key，或者接口不支持 function calling 时的退路；
2. 参数修复——模型给的参数不合法时，用规则再抽一次槽位。

关键词分强、弱两档：强词记 2 分，弱词记 1 分，达到 2 分才算选中。
这样"现在北京天气怎么样"里的"现在"（弱词，1 分）不会把时间工具勾出来，
而"算一下"（强词，2 分）会稳稳命中计算工具。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import cities
from .models import ToolCall, ToolResult

SELECT_THRESHOLD = 2

TOOL_LABELS = {
    "get_weather": "天气",
    "calculate": "计算",
    "get_time": "时间",
}

CHINESE_OPERATORS = (
    ("乘以", "*"),
    ("乘", "*"),
    ("加上", "+"),
    ("加", "+"),
    ("减去", "-"),
    ("减", "-"),
    ("除以", "/"),
    ("除", "/"),
)

MATH_PATTERN = re.compile(r"[\(\[]?\s*\d[\d\s.,()+\-*/^×÷]*[\d\)]")
CHINESE_MATH_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:乘以|乘|加上|加|减去|减|除以|除)\s*\d+(?:\.\d+)?"
)
DAYS_OFFSET_PATTERN = re.compile(r"([一二两三四五六七八九十\d]+)\s*天(后|前)")

CN_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass
class Rule:
    """一个工具的关键词规则。"""

    tool: str
    strong: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()


@dataclass
class Route:
    """一次规则命中：选中的工具和抽到的槽位。"""

    tool: str
    score: int
    slots: dict[str, Any] = field(default_factory=dict)


RULES: tuple[Rule, ...] = (
    Rule(
        tool="get_weather",
        strong=(
            "天气",
            "气温",
            "温度",
            "下雨",
            "降雨",
            "多少度",
            "冷不冷",
            "热不热",
            "天气预报",
            "天气如何",
            "天气怎样",
            "天气怎么样",
        ),
        weak=("晴", "阴", "多云", "阵雨", "湿度", "风力", "下雪", "雨伞"),
    ),
    Rule(
        tool="calculate",
        strong=(
            "计算",
            "算一下",
            "算算",
            "等于多少",
            "等于几",
            "加多少",
            "减多少",
            "乘以",
            "除以",
            "求值",
            "是多少",
        ),
        weak=("加", "减", "乘", "除", "求和", "结果是"),
    ),
    Rule(
        tool="get_time",
        strong=(
            "时间",
            "几点",
            "星期几",
            "星期",
            "几号",
            "哪一天",
            "哪天",
            "日期",
            "时钟",
            "钟表",
            "当前时间",
            "现在几点",
        ),
        weak=("现在", "今天", "明天", "昨天", "早上", "中午", "晚上", "时区", "utc"),
    ),
)


def score_tools(text: str) -> list[Route]:
    """给每个工具打分，返回分数从高到低的候选。"""

    routes: list[Route] = []

    for rule in RULES:
        score = 0

        for keyword in rule.strong:
            if keyword in text:
                score += 2
                break

        for keyword in rule.weak:
            if keyword in text:
                score += 1

        if score > 0:
            routes.append(Route(tool=rule.tool, score=score))

    routes.sort(key=lambda route: route.score, reverse=True)
    return routes


def select_tools(text: str) -> list[str]:
    """选出该用哪些工具（可能多个：用户一句话问了两件事）。"""

    return [
        route.tool
        for route in score_tools(text)
        if route.score >= SELECT_THRESHOLD
    ]


# ------------------------------------------------------------
# 槽位抽取
# ------------------------------------------------------------

def extract_city(text: str) -> str:
    """从一句话里抽出城市名，抽不到返回空字符串。"""

    found = cities.find_city(text)

    if found is not None:
        return found.cn

    quoted = re.search(r"[「『\"'“”]([^「』\"'“”]{2,10})[」』\"'“”]", text)

    if quoted:
        return quoted.group(1).strip()

    pattern = re.search(r"(?:在|去|查|看看|看下)\s*([\u4e00-\u9fa5]{2,8}?)(?:今天|明天|后天|现在|的|市|天气|温度|$)", text)

    if pattern:
        return pattern.group(1).strip()

    return ""


def extract_expression(text: str) -> str:
    """从一句话里抽出算式，抽不到返回空字符串。"""

    chinese = CHINESE_MATH_PATTERN.search(text)

    if chinese:
        return _normalize_operators(chinese.group(0))

    match = MATH_PATTERN.search(text)

    if not match:
        return ""

    return _normalize_operators(match.group(0))


def _normalize_operators(text: str) -> str:
    """统一符号写法并去掉多余空格。"""

    for source, target in CHINESE_OPERATORS:
        text = text.replace(source, target)

    replacements = {
        "×": "*",
        "÷": "/",
        "，": ".",
        "．": ".",
        ",": ".",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    text = text.replace(" ", "").replace("\t", "")
    return text.strip("+-*/")


def extract_offset_days(text: str) -> int:
    """抽"几天后 / 几天前"，抽不到返回 0。"""

    if "大后天" in text:
        return 3

    if "后天" in text:
        return 2

    if "明天" in text:
        return 1

    if "前天" in text:
        return -2

    if "昨天" in text:
        return -1

    match = DAYS_OFFSET_PATTERN.search(text)

    if match:
        amount = match.group(1)
        number = CN_NUMBERS.get(amount)

        if number is None and amount.isdigit():
            number = int(amount)

        if number is not None:
            return number if match.group(2) == "后" else -number

    return 0


def extract_timezone(text: str) -> str:
    """抽时区：先看固定说法，再看城市名。"""

    if "北京时" in text:
        return "Asia/Shanghai"

    if "utc" in text.lower():
        return "UTC"

    found = cities.find_city(text)

    if found is not None:
        return found.tz

    return ""


def build_slots(tool: str, text: str) -> dict[str, Any]:
    """按工具抽槽位。"""

    if tool == "get_weather":
        slots: dict[str, Any] = {}

        city = extract_city(text)

        if city:
            slots["city"] = city

        if "明天" in text:
            slots["day"] = "tomorrow"

        if "华氏" in text or "℉" in text:
            slots["unit"] = "fahrenheit"

        return slots

    if tool == "calculate":
        expression = extract_expression(text)

        return {"expression": expression} if expression else {}

    if tool == "get_time":
        slots = {}

        timezone = extract_timezone(text)

        if timezone:
            slots["timezone"] = timezone

        offset = extract_offset_days(text)

        if offset:
            slots["offset_days"] = offset

        return slots

    return {}


def build_calls(text: str) -> list[ToolCall]:
    """把一句话变成一组工具调用。"""

    calls: list[ToolCall] = []

    for tool in select_tools(text):
        calls.append(ToolCall(name=tool, arguments=build_slots(tool, text)))

    return calls


def label_for(tool: str) -> str:
    """工具的中文标签。"""

    return TOOL_LABELS.get(tool, tool)


def compose_reply(results: list[ToolResult]) -> str:
    """把多个工具结果拼成一段回答（成功的、失败的都算进去）。"""

    lines: list[str] = []

    for result in results:
        lines.append(f"【{label_for(result.name)}】{result.content}")

    return "\n".join(lines)
