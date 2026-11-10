"""轻量 JSON Schema 校验器：把模型给的工具参数挡在真正执行之前。

只实现本示例用得到的子集：type / required / properties / enum /
minimum / maximum / minLength / maxLength / additionalProperties。

校验和修复刻意分成两步：
- validate_arguments() 只判对错，不改数据；
- coerce_arguments() 先做安全修复（数字字符串转数字、枚举忽略大小写、
  字符串去空格），再重新校验一遍，剩下的硬错误才交出去。

这样"能自动修的"不会白白打断模型一次往返，"修不好的"才值得回问用户。
"""

from __future__ import annotations

from typing import Any

from .models import FieldError


def _type_matches(value: Any, expected: str) -> bool:
    """判断值的类型是否符合 schema 声明。

    bool 是 int 的子类，所以数字类型要显式排除 bool，
    否则 True 会被当成 1 放行。
    """

    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)

    return True


def _validate_value(name: str, value: Any, spec: dict) -> list[FieldError]:
    """校验单个字段，type 不对时不再往下查（后面的规则都没意义了）。"""

    errors: list[FieldError] = []
    expected = spec.get("type")

    if expected and not _type_matches(value, expected):
        return [
            FieldError(
                name,
                "type",
                f"类型应为 {expected}，实际是 {type(value).__name__}",
                f"改成 {expected} 类型的值",
            )
        ]

    if expected == "string":
        min_length = spec.get("minLength")
        max_length = spec.get("maxLength")

        if min_length is not None and len(value.strip()) < min_length:
            errors.append(
                FieldError(name, "length", f"不能为空（至少 {min_length} 个字符）", "给出具体内容")
            )

        if max_length is not None and len(value) > max_length:
            errors.append(
                FieldError(
                    name,
                    "length",
                    f"长度不能超过 {max_length} 个字符",
                    "缩短内容后再试",
                )
            )

    if expected in {"number", "integer"}:
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")

        if minimum is not None and value < minimum:
            errors.append(
                FieldError(name, "range", f"不能小于 {minimum}", f"改成一个 >= {minimum} 的数")
            )

        if maximum is not None and value > maximum:
            errors.append(
                FieldError(name, "range", f"不能大于 {maximum}", f"改成一个 <= {maximum} 的数")
            )

    enum = spec.get("enum")

    if enum is not None and value not in enum:
        errors.append(
            FieldError(
                name,
                "enum",
                f"只能是 {'/'.join(str(item) for item in enum)} 之一，实际是 {value!r}",
                f"改用 {'/'.join(str(item) for item in enum)} 中的一个",
            )
        )

    return errors


def validate_arguments(schema: dict, arguments: dict) -> list[FieldError]:
    """按 schema 校验参数，返回所有错误（空列表表示通过）。

    顺序固定为：缺参 → 多余参数 → 逐字段校验，
    这样报错顺序稳定，方便写自检。
    """

    errors: list[FieldError] = []
    properties: dict = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for name in required:
        if name not in arguments or arguments[name] is None:
            errors.append(
                FieldError(name, "missing", f"缺少必填参数 {name}", f"补上 {name}")
            )

    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                allowed = ", ".join(properties) or "（无）"
                errors.append(
                    FieldError(name, "unknown", f"不支持的参数 {name}", f"只允许：{allowed}")
                )

    for name, value in arguments.items():
        spec = properties.get(name)

        if spec is None:
            continue

        errors.extend(_validate_value(name, value, spec))

    return errors


def coerce_arguments(schema: dict, arguments: dict) -> tuple[dict, list[FieldError]]:
    """尝试修复参数，再校验一次。

    返回 (修复后的参数, 剩余错误)。修复只做"改写语义不变"的调整：
    - 数字字符串 → 数字（"12" → 12）
    - 枚举忽略大小写（"Today" → "today"）
    - 字符串去掉首尾空格
    - None 视为没传
    """

    properties: dict = schema.get("properties", {})
    fixed = dict(arguments)

    for name, value in list(arguments.items()):
        spec = properties.get(name)

        if spec is None:
            continue

        expected = spec.get("type")

        if expected == "string" and isinstance(value, str):
            fixed[name] = value.strip()

        elif expected in {"number", "integer"} and isinstance(value, str):
            text = value.strip()

            try:
                number = float(text)
            except ValueError:
                continue

            if expected == "integer":
                if number.is_integer():
                    fixed[name] = int(number)
            else:
                fixed[name] = number

    for name, spec in properties.items():
        enum = spec.get("enum")
        value = fixed.get(name)

        if not enum or not isinstance(value, str):
            continue

        for option in enum:
            if isinstance(option, str) and option.lower() == value.lower():
                fixed[name] = option
                break

    fixed = {key: value for key, value in fixed.items() if value is not None}

    return fixed, validate_arguments(schema, fixed)


def apply_defaults(schema: dict, arguments: dict) -> dict:
    """给缺省的可选参数填上默认值。"""

    properties: dict = schema.get("properties", {})
    filled = dict(arguments)

    for name, spec in properties.items():
        if name not in filled and "default" in spec:
            filled[name] = spec["default"]

    return filled


def format_errors(errors: list[FieldError]) -> str:
    """把错误列表渲染成一段能直接回喂给模型的中文文本。"""

    if not errors:
        return "通过"

    lines = [f"- {error.render()}" for error in errors]
    return "\n".join(lines)
