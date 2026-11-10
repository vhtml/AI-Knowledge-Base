"""两种"选工具"的策略，对 Agent 层是同一套接口。

- LLMPlanner：走 OpenAI 兼容接口的 function calling，让模型自己决定
  调不调工具、调哪个、参数怎么填——这是主路径。
- RouterPlanner：不联网，用关键词规则选工具，兼做参数修复和追问——
  这是兜底路径，也是离线演示用的路径。

两条路径的分工正好说明了"工具选择"这件事的两个层面：
谁来选（模型 / 规则），以及选错了怎么收拾（报错 → 修复 → 追问）。
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import router
from .models import PlannerTurn, ToolCall, ToolResult
from .tools import ToolRegistry

SYSTEM_PROMPT = """你是一个生活助手，可以调用三个工具来回答用户的问题：
- get_weather：查询某个城市今天或明天的天气
- calculate：计算一个数学表达式
- get_time：查询当前时间、某个时区的时间，或相对今天偏移几天的日期

规则：
1. 天气、时间这类外部数据必须调用工具，不要凭记忆回答。
2. 只调用当前问题需要的工具；用户同时问了两件事，就分两轮调用。
3. 参数必须严格符合工具说明：城市用常见中文写法；day 只能是 today/tomorrow；
   unit 只能是 celsius/fahrenheit；offset_days 是 -365 ~ 365 的整数。
4. 工具返回"参数校验失败"时，先读懂错误信息，改掉出错的参数再重新调用一次。
5. 参数缺失、且没法从上下文推断时，直接向用户追问，不要瞎猜。
6. 用中文简短回答，工具结果里的数字要如实转述，不要自己改算。"""


class LLMPlanner:
    """用模型的 function calling 选工具。

    对话历史由它自己维护：每轮把用户消息、工具结果按 OpenAI 的顺序追加进去，
    这样模型才能看到"上一轮我调了什么、返回了什么"。
    """

    def __init__(
        self,
        model: str,
        base_url: str | None,
        api_key: str | None,
        tool_schemas: list[dict[str, Any]],
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.2,
        timeout: float = 90.0,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._pending_call_ids: list[str] = []

    def plan(self, user_text: str) -> PlannerTurn:
        """新的一轮用户输入：请求模型给出工具调用或直接回答。"""

        self._messages.append({"role": "user", "content": user_text})
        return self._request()

    def continue_plan(self, results: list[ToolResult]) -> PlannerTurn:
        """把工具结果回填给模型，让它决定下一步。"""

        for result, call_id in zip(results, self._pending_call_ids):
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"[{result.name}] {result.content}",
                }
            )

        return self._request()

    def _request(self) -> PlannerTurn:
        """调一次模型，把回复翻译成工具调用或最终回答。"""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            tools=self.tool_schemas,
            tool_choice="auto",
            temperature=self.temperature,
        )

        if not response.choices:
            return PlannerTurn(reply="模型没有返回内容，请再问一次。")

        message = response.choices[0].message
        content = (message.content or "").strip()
        raw_calls = list(message.tool_calls or [])

        self._messages.append(self._assistant_message(message))
        self._pending_call_ids = [call.id for call in raw_calls]

        if not raw_calls:
            return PlannerTurn(reply=content)

        return PlannerTurn(calls=[self._to_call(call) for call in raw_calls], reply=content)

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        """把助手消息还原成 API 能接受的格式。"""

        payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        return payload

    @staticmethod
    def _to_call(call: Any) -> ToolCall:
        """把 API 的 tool_call 转成内部结构，JSON 坏了也要留下线索。"""

        raw = call.function.arguments or "{}"
        parse_error = ""

        try:
            arguments = json.loads(raw)

            if not isinstance(arguments, dict):
                arguments = {}
                parse_error = "工具参数必须是 JSON 对象。"

        except json.JSONDecodeError as exc:
            arguments = {}
            parse_error = f"参数不是合法 JSON：{exc.msg}"

        return ToolCall(
            name=call.function.name,
            arguments=arguments,
            call_id=call.id,
            raw_arguments=raw,
            parse_error=parse_error,
        )


class RouterPlanner:
    """规则版规划器：离线兜底 + 参数修复 + 追问。

    它也实现了 Planner 接口，所以 Agent 层不用关心当前跑的是哪条路径。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._results: list[ToolResult] = []
        self._pending: ToolCall | None = None
        self._last_text: str = ""

    def plan(self, user_text: str) -> PlannerTurn:
        """新的一轮用户输入。"""

        self._results = []
        self._last_text = user_text

        # 上一轮缺参数、正在等用户补充时，这一句话要先试着补上那个参数。
        # 但只有当这句话自己不像一个新问题时才这么做——
        # 否则"算一下"这种新问题会被当成对上一轮的补充。
        if self._pending is not None and not router.select_tools(user_text):
            filled = self._fill_pending(user_text)

            if filled is not None:
                self._pending = None
                return PlannerTurn(calls=[filled])

        self._pending = None
        calls = router.build_calls(user_text)

        if not calls:
            return PlannerTurn(reply=self._capability_reply())

        return PlannerTurn(calls=calls)

    def continue_plan(self, results: list[ToolResult]) -> PlannerTurn:
        """工具结果回来了：能修就修，修不好就追问用户。

        参数校验失败和工具执行失败要分开看：
        前者还能再修一次，后者是工具自己干不成的事，再调一遍也是白调。
        """

        self._results.extend(results)
        failures = [result for result in results if not result.ok]
        validation_failures = [result for result in failures if result.errors]

        if not validation_failures:
            return PlannerTurn(reply=router.compose_reply(self._results))

        repaired: list[ToolCall] = []

        for failure in validation_failures:
            call = self._repair(failure)

            if call is not None:
                repaired.append(call)

        if repaired:
            return PlannerTurn(calls=repaired)

        self._pending = validation_failures[0].call
        return PlannerTurn(reply="", clarification=self._clarify(validation_failures))

    # ------------------------------------------------------------
    # 内部：追问与修复
    # ------------------------------------------------------------

    def _capability_reply(self) -> str:
        """没有工具能处理这句话时的回话。"""

        return (
            "这句话我暂时没有对应的工具，可以帮你做三件事："
            "查天气（get_weather）、算表达式（calculate）、查时间（get_time）。"
        )

    def _clarify(self, failures: list[ToolResult]) -> str:
        """根据失败的工具生成一句追问。"""

        prompts = {
            "get_weather": "请问要查哪个城市的天气？例如「北京」。",
            "calculate": "请给出要计算的表达式，例如「(12+8)*3」。",
            "get_time": "请问要查哪个时区的时间？例如「北京时间」或「Asia/Shanghai」。",
        }

        for failure in failures:
            prompt = prompts.get(failure.name)

            if prompt:
                return prompt

        return "这个工具的参数没能补全，能再说得具体一点吗？"

    def _repair(self, failure: ToolResult) -> ToolCall | None:
        """针对一次失败，尝试本地修一次参数。"""

        call = failure.call

        if call is None:
            return None

        arguments = dict(call.arguments)

        if call.name == "get_weather":
            city = str(arguments.get("city", "")).strip()

            if not city:
                city = router.extract_city(self._last_text)

            if not city:
                return None

            arguments["city"] = city

        elif call.name == "calculate":
            expression = str(arguments.get("expression", "")).strip()

            if not expression:
                expression = router.extract_expression(self._last_text)

            if not expression:
                return None

            arguments["expression"] = expression

        elif call.name == "get_time":
            timezone = str(arguments.get("timezone", "")).strip()

            if timezone and "/" not in timezone:
                from . import providers

                try:
                    providers.resolve_timezone(timezone, None)
                except Exception:
                    arguments["timezone"] = router.extract_timezone(self._last_text)

            if not arguments.get("timezone") and not arguments.get("city"):
                return None

        else:
            return None

        return ToolCall(name=call.name, arguments=arguments, call_id=call.call_id)

    def _fill_pending(self, user_text: str) -> ToolCall | None:
        """把用户补充的那一句填进上一轮缺的那个参数。"""

        pending = self._pending

        if pending is None:
            return None

        arguments = dict(pending.arguments)
        text = user_text.strip()

        if pending.name == "get_weather" and not arguments.get("city"):
            city = router.extract_city(text) or (
                text if _looks_like_value(text) else ""
            )

            if not city:
                return None

            arguments["city"] = city

        elif pending.name == "calculate" and not arguments.get("expression"):
            expression = router.extract_expression(text) or (
                text if _looks_like_value(text) else ""
            )

            if not expression:
                return None

            arguments["expression"] = expression

        elif pending.name == "get_time":
            timezone = router.extract_timezone(text)

            if timezone:
                arguments["timezone"] = timezone
            elif _looks_like_value(text):
                arguments["city"] = text
            else:
                return None

        else:
            return None

        return ToolCall(
            name=pending.name,
            arguments=arguments,
            call_id=pending.call_id,
        )


def _looks_like_value(text: str) -> bool:
    """判断一句话像不像"用户补上的一个值"（而不是一个新问题）。"""

    if not text or len(text) > 16:
        return False

    if any(mark in text for mark in "，。？！?!\n"):
        return False

    return True


class FallbackPlanner:
    """先走模型；模型一报错就退到规则路由。

    模型接口挂了不该让整个示例不可用——工具还在，
    规则路由照样能把天气、计算、时间答出来，
    只是工具选择从"模型说了算"变成"规则说了算"。
    """

    def __init__(
        self,
        primary,
        fallback: RouterPlanner,
        notice: str = "模型接口不可用，已切换到离线规则路由",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._notice = notice
        self._switched = False

    @property
    def switched(self) -> bool:
        """是否已经切换到规则路由。"""

        return self._switched

    def plan(self, user_text: str) -> PlannerTurn:
        """新的一轮用户输入。"""

        if self._switched:
            return self._fallback.plan(user_text)

        try:
            return self._primary.plan(user_text)
        except Exception as exc:
            self._switch(exc)
            return self._fallback.plan(user_text)

    def continue_plan(self, results: list[ToolResult]) -> PlannerTurn:
        """把工具结果回填给模型（或规则路由）。"""

        if self._switched:
            return self._fallback.continue_plan(results)

        try:
            return self._primary.continue_plan(results)
        except Exception as exc:
            self._switch(exc)
            return self._fallback.continue_plan(results)

    def _switch(self, exc: Exception) -> None:
        """切换到规则路由，并说明原因。"""

        self._switched = True
        print(
            f"[提示] {self._notice}（{type(exc).__name__}: {exc}）",
            file=sys.stderr,
        )
