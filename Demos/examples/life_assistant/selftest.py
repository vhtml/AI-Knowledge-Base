"""自检脚本：验证工具选择与参数校验的核心不变量。

不依赖 pytest，也不联网，直接运行：

    cd Demos/examples
    python -m life_assistant.selftest
"""

from __future__ import annotations

from .agent import LifeAssistant
from .calc import evaluate, format_number
from .models import ToolError
from .models import PlannerTurn, ToolCall
from .planners import RouterPlanner
from .providers import FakeWeatherProvider, format_moment, current_moment
from .router import (
    build_calls,
    extract_city,
    extract_expression,
    extract_offset_days,
    select_tools,
)
from .tools import (
    CALCULATE_SCHEMA,
    TIME_SCHEMA,
    WEATHER_SCHEMA,
    build_registry,
)
from .validation import coerce_arguments, validate_arguments

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """登记一条检查结果。"""

    global PASSED, FAILED

    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f"  → {detail}" if detail else ""))


def codes(errors) -> list[str]:
    """取出错误码列表。"""

    return [error.code for error in errors]


def test_validation() -> None:
    """参数校验：拦得住，也修得了。"""

    print("\n[1] 参数校验")

    errors = validate_arguments(WEATHER_SCHEMA, {})
    check("缺必填参数被拦下", "missing" in codes(errors), str(codes(errors)))

    errors = validate_arguments(WEATHER_SCHEMA, {"city": "北京", "day": "weekend"})
    check("枚举取值不合法被拦下", "enum" in codes(errors), str(codes(errors)))

    errors = validate_arguments(WEATHER_SCHEMA, {"city": 123})
    check("类型不对被拦下", "type" in codes(errors), str(codes(errors)))

    errors = validate_arguments(WEATHER_SCHEMA, {"city": "北京", "temp": 20})
    check("多余参数被拦下", "unknown" in codes(errors), str(codes(errors)))

    errors = validate_arguments(WEATHER_SCHEMA, {"city": "  "})
    check("空字符串被拦下", "length" in codes(errors), str(codes(errors)))

    errors = validate_arguments(TIME_SCHEMA, {"offset_days": 9999})
    check("数值越界被拦下", "range" in codes(errors), str(codes(errors)))

    errors = validate_arguments(CALCULATE_SCHEMA, {"expression": "1"})
    check("合法参数没有错误", not errors, str(codes(errors)))

    fixed, remaining = coerce_arguments(WEATHER_SCHEMA, {"city": "北京", "day": "Today"})
    check(
        "枚举大小写自动修复",
        not remaining and fixed["day"] == "today",
        str(fixed),
    )

    fixed, remaining = coerce_arguments(TIME_SCHEMA, {"offset_days": "3"})
    check(
        "数字字符串自动转成整数",
        not remaining and fixed["offset_days"] == 3,
        str(fixed),
    )

    fixed, remaining = coerce_arguments(CALCULATE_SCHEMA, {"expression": " 1+1 "})
    check("字符串去空格", not remaining and fixed["expression"] == "1+1", str(fixed))


def test_router() -> None:
    """工具选择：选对工具，抽对槽位。"""

    print("\n[2] 工具选择")

    check("天气问题选中天气工具", select_tools("北京今天天气怎么样") == ["get_weather"])
    check("计算问题选中计算工具", select_tools("算一下 (12+8)*3") == ["calculate"])
    check("时间问题选中时间工具", select_tools("现在几点") == ["get_time"])
    check(
        "一句话两件事选中两个工具",
        sorted(select_tools("北京天气怎么样，顺便算一下 3*4")) == ["calculate", "get_weather"],
    )
    check("无关问题不选工具", select_tools("查一下股票行情") == [], str(select_tools("查一下股票行情")))
    check(
        "弱关键词不误触发",
        select_tools("现在北京天气怎么样") == ["get_weather"],
        str(select_tools("现在北京天气怎么样")),
    )

    check("抽出城市", extract_city("北京今天天气怎么样") == "北京", extract_city("北京今天天气怎么样"))
    check(
        "抽出引号里的城市",
        extract_city("「上海」的天气") == "上海",
        extract_city("「上海」的天气"),
    )
    check("抽出表达式", extract_expression("算一下 (12+8)*3") == "(12+8)*3", extract_expression("算一下 (12+8)*3"))
    check(
        "中文运算符归一化",
        extract_expression("帮我算 12 乘以 8") == "12*8",
        extract_expression("帮我算 12 乘以 8"),
    )
    check("抽出偏移天数", extract_offset_days("三天后是哪天") == 3, str(extract_offset_days("三天后是哪天")))
    check("抽出昨天", extract_offset_days("昨天上海天气如何") == -1, str(extract_offset_days("昨天上海天气如何")))

    calls = build_calls("明天上海天气如何")
    check(
        "天气槽位带上了城市和明天",
        len(calls) == 1
        and calls[0].name == "get_weather"
        and calls[0].arguments.get("city") == "上海"
        and calls[0].arguments.get("day") == "tomorrow",
        str(calls),
    )


def test_calculate_tool() -> None:
    """计算工具：算得对，也拦得住危险表达式。"""

    print("\n[3] 计算工具")

    check("四则运算", evaluate("(12+8)*3") == 60.0)
    check("幂运算", evaluate("2**10") == 1024.0)
    check("开方", evaluate("sqrt(16)") == 4.0)
    check("白名单函数", evaluate("max(3, 7)") == 7.0)
    check("常量", abs(evaluate("pi") - 3.14159) < 0.001)
    check("结果格式化成整数", format_number(60.0) == "60", format_number(60.0))

    check("除以 0 被拦下", _raises(lambda: evaluate("1/0")))
    check("危险代码被拦下", _raises(lambda: evaluate("__import__('os').system('ls')")))
    check("属性访问被拦下", _raises(lambda: evaluate("().__class__")))
    check("未知常量被拦下", _raises(lambda: evaluate("unknown + 1")))
    check("超长表达式被拦下", _raises(lambda: evaluate("1+" * 80 + "1")))


def test_time_tool() -> None:
    """时间工具：时区、偏移、格式。"""

    print("\n[4] 时间工具")

    name, moment = current_moment("Asia/Shanghai", None, 0)
    check("时区名解析正确", name == "Asia/Shanghai", name)
    check("当前时刻可用", moment.year >= 2024, str(moment))

    _, tomorrow = current_moment("Asia/Shanghai", None, 1)
    check("偏移一天有效", (tomorrow - moment).days == 1, str(tomorrow - moment))

    check("日期格式", format_moment(moment, "date") == moment.strftime("%Y-%m-%d"))
    check("时间格式", format_moment(moment, "time") == moment.strftime("%H:%M"))
    check("ISO 格式", format_moment(moment, "iso").startswith(moment.strftime("%Y-%m-%dT")))
    check("时间戳是整数", format_moment(moment, "timestamp").isdigit())

    check("非法时区被拦下", _raises(lambda: current_moment("Mars/Olympus", None, 0)))
    check("未知城市被拦下", _raises(lambda: current_moment(None, "火星镇", 0)))


def test_weather_tool() -> None:
    """天气工具：离线假数据也能跑。"""

    print("\n[5] 天气工具")

    registry = build_registry(FakeWeatherProvider())
    tool = registry.get("get_weather")

    check("工具注册成功", tool is not None)

    payload = tool.handler({"city": "北京", "day": "today", "unit": "celsius"})
    check("假天气有结果", "北京" in payload.text and "℃" in payload.text, payload.text)

    again = tool.handler({"city": "北京", "day": "today", "unit": "celsius"})
    check("假数据是确定的", again.text == payload.text)

    fahrenheit = tool.handler({"city": "北京", "day": "today", "unit": "fahrenheit"})
    check("华氏度换算生效", "℉" in fahrenheit.text, fahrenheit.text)


def test_end_to_end() -> None:
    """端到端：选择 → 校验 → 执行 → 回答。"""

    print("\n[6] 端到端（离线）")

    registry = build_registry(FakeWeatherProvider())
    assistant = LifeAssistant(RouterPlanner(registry), registry)

    turn = assistant.handle("北京今天天气怎么样")
    check("天气问题走完了流程", "北京" in turn.reply, turn.reply)
    check(
        "流程留痕齐全",
        [step.phase for step in turn.steps][:2] == ["选工具", "校验参数"],
        str([step.phase for step in turn.steps]),
    )

    turn = assistant.handle("算一下 (12+8)*3")
    check("计算结果正确", "60" in turn.reply, turn.reply)

    turn = assistant.handle("天气怎么样")
    check("缺参数时追问用户", turn.clarification != "" and "城市" in turn.reply, turn.reply)

    turn = assistant.handle("北京")
    check("用户补充后继续完成", "北京" in turn.reply and "℃" in turn.reply, turn.reply)

    turn = assistant.handle("算一下")
    check("缺表达式时追问", "表达式" in turn.reply, turn.reply)

    # 回归：正在追问参数时来了一句新问题，新问题不能被当成对上一轮的补充
    assistant.handle("天气怎么样")
    turn = assistant.handle("算一下")
    check("新问题不会被当成补充", "表达式" in turn.reply, turn.reply)

    turn = assistant.handle("算一下 100/0")
    check("执行失败如实报错", "除数为 0" in turn.reply, turn.reply)

    turn = assistant.handle("查一下股票行情")
    check("没有工具时说明边界", "工具" in turn.reply, turn.reply)


def _raises(func) -> bool:
    """判断函数有没有抛异常。"""

    try:
        func()
    except Exception:
        return True

    return False


class RepairPlanner:
    """假扮一个"第一次参数填错、看到报错后自己改对"的模型。

    用来验证这条链路：参数不合法 → 工具不执行 → 错误回喂给规划器 →
    规划器改参数再调一次 → 这次通过 → 拿到结果。
    """

    def __init__(self, first_arguments: dict, second_arguments: dict) -> None:
        self.first_arguments = first_arguments
        self.second_arguments = second_arguments
        self.errors: list[str] = []
        self.rounds = 0

    def plan(self, user_text: str):
        """第一轮：按设计给一个不合法参数。"""

        self.rounds += 1
        return PlannerTurn(calls=[ToolCall(name="get_weather", arguments=self.first_arguments)])

    def continue_plan(self, results):
        """第二轮：读到报错后改对参数；第三轮：给出最终回答。"""

        for result in results:
            self.errors.append(result.content)

        self.rounds += 1

        if self.rounds == 2:
            return PlannerTurn(calls=[ToolCall(name="get_weather", arguments=self.second_arguments)])

        return PlannerTurn(reply="北京明天有雨，记得带伞。")


class UnknownToolPlanner:
    """假扮一个调了不存在工具的模型。"""

    def plan(self, user_text: str):
        """第一轮：调一个不存在的工具。"""

        return PlannerTurn(calls=[ToolCall(name="get_stock", arguments={})])

    def continue_plan(self, results):
        """第二轮：把错误如实转述给用户。"""

        return PlannerTurn(reply=results[0].content)


def test_validation_feedback() -> None:
    """校验失败 → 错误回喂 → 改对参数 → 再执行。"""

    print("\n[7] 校验失败的自我修复")

    registry = build_registry(FakeWeatherProvider())
    planner = RepairPlanner(
        first_arguments={"city": "北京", "day": "weekend"},
        second_arguments={"city": "北京", "day": "tomorrow"},
    )
    assistant = LifeAssistant(planner, registry)

    turn = assistant.handle("北京明天天气怎么样")

    check("第一次调用被拦下", planner.rounds >= 2, str(planner.rounds))
    check(
        "错误被回喂给规划器",
        bool(planner.errors) and "参数校验失败" in planner.errors[0],
        str(planner.errors),
    )
    check("错误里点名了具体字段", "day" in planner.errors[0], str(planner.errors))
    check("改对参数后执行成功", "明天" in turn.reply, turn.reply)
    check(
        "最终回答来自模型",
        turn.reply == "北京明天有雨，记得带伞。",
        turn.reply,
    )
    check(
        "只有合法的调用被执行过",
        sum(1 for step in turn.steps if step.phase == "执行工具") == 1,
        str([step.title for step in turn.steps]),
    )


def test_unknown_tool() -> None:
    """模型调了不存在的工具。"""

    print("\n[8] 工具选择出错")

    registry = build_registry(FakeWeatherProvider())
    assistant = LifeAssistant(UnknownToolPlanner(), registry)

    turn = assistant.handle("查一下股票")

    check("不存在的工具被拦下", "工具不存在" in turn.reply, turn.reply)
    check(
        "错误里给出了可用工具",
        "get_weather" in turn.reply and "calculate" in turn.reply,
        turn.reply,
    )


def main() -> int:
    """自检入口。"""

    print("=" * 56)
    print("life_assistant 自检")
    print("=" * 56)

    test_validation()
    test_router()
    test_calculate_tool()
    test_time_tool()
    test_weather_tool()
    test_end_to_end()
    test_validation_feedback()
    test_unknown_tool()

    print("\n" + "=" * 56)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
