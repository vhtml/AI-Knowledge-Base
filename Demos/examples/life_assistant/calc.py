"""安全表达式求值：只支持算术，不执行任意 Python 代码。

用 ast.parse(mode="eval") 解析表达式，然后只遍历白名单里的节点
（数字常量、四则运算、少量白名单函数）。出现白名单之外的节点
——属性访问、导入、函数定义、赋值——一律拒绝。

这是"参数校验"的一部分：光看 schema 只能保证 expression 是个字符串，
字符串里装的是不是表达式、会不会变成代码执行，得在工具内部再挡一道。
"""

from __future__ import annotations

import ast
import math

from .models import ToolError

MAX_EXPRESSION_LENGTH = 200
MAX_DEPTH = 20
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# 中文/全角写法先归一化，省得在解析阶段处理一堆符号
NORMALIZE = {
    "×": "*",
    "÷": "/",
    "＋": "+",
    "－": "-",
    "（": "(",
    "）": ")",
    "，": ",",
    "^": "**",
}


def normalize_expression(expression: str) -> str:
    """把全角符号和中文标点换成 ASCII。"""

    text = expression.strip()

    for source, target in NORMALIZE.items():
        text = text.replace(source, target)

    return text


def _evaluate_node(node: ast.AST, depth: int = 0) -> float:
    """递归求值，只认白名单节点。"""

    if depth > MAX_DEPTH:
        raise ToolError(f"表达式嵌套太深（上限 {MAX_DEPTH} 层）")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("只支持数字常量")
        return float(node.value)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _evaluate_node(node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            return -_evaluate_node(node.operand, depth + 1)
        raise ToolError("不支持的一元运算符")

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, depth + 1)
        right = _evaluate_node(node.right, depth + 1)
        operator = node.op

        if isinstance(operator, ast.Add):
            return left + right
        if isinstance(operator, ast.Sub):
            return left - right
        if isinstance(operator, ast.Mult):
            return left * right
        if isinstance(operator, ast.Div):
            if right == 0:
                raise ToolError("除数为 0，无法计算")
            return left / right
        if isinstance(operator, ast.Mod):
            if right == 0:
                raise ToolError("取模的除数为 0，无法计算")
            return math.fmod(left, right)
        if isinstance(operator, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolError(f"指数过大（上限 {MAX_EXPONENT}）")
            return left**right

        raise ToolError("不支持的二元运算符")

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ToolError(f"未知常量 {node.id}，可用常量：{', '.join(ALLOWED_CONSTANTS)}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ToolError(
                f"只允许使用这些函数：{', '.join(sorted(ALLOWED_FUNCTIONS))}"
            )

        if node.keywords:
            raise ToolError("函数不支持关键字参数")

        arguments = [_evaluate_node(argument, depth + 1) for argument in node.args]
        return float(ALLOWED_FUNCTIONS[node.func.id](*arguments))

    raise ToolError(f"不支持的表达式片段：{type(node).__name__}")


def evaluate(expression: str) -> float:
    """计算一个数学表达式，返回结果。

    失败时抛 ToolError，消息是给用户看的中文提示。
    """

    text = normalize_expression(expression)

    if not text:
        raise ToolError("表达式不能为空")

    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"表达式太长了（上限 {MAX_EXPRESSION_LENGTH} 个字符）")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError(f"表达式语法有误：{expression}") from None

    result = _evaluate_node(tree.body)

    if not math.isfinite(result):
        raise ToolError("结果不是一个有限的数")

    if abs(result) > MAX_MAGNITUDE:
        raise ToolError(f"结果太大了（上限 {MAX_MAGNITUDE:.0e}）")

    return result


def format_number(value: float) -> str:
    """把计算结果格式化成人类友好的字符串。"""

    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"
