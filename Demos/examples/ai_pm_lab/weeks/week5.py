"""第 5 周实验：自动化等级与工具权限。

这个模块解决什么问题：把"这个动作要不要让人点确认"从个人习惯变成一条可查的规则——
风险 × 频次 × 确定性决定自动化等级，等级决定必须有哪些兜底项；
同时给每个工具加一道权限体检，挡住"声明只读、实际在写"这类越权。

对应笔记：AI/AI-Product/ 第 09 章《自动化等级》与第 10 章《工具权限》。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..labkit import FixtureError, Report, fmt_table

# 六级自动化，从最保守到最激进。
LEVEL_NAMES: tuple[str, ...] = ("手动", "建议", "草稿", "带审批自动", "自动", "全自动")

# 每一级的必须兜底项。等级越高，兜底越偏向"事后可查、可回滚"，
# 而不是"事前拦得住"——事前拦得住的东西，模型就做不到全自动。
LEVEL_GUARDRAILS: dict[str, tuple[str, ...]] = {
    "手动": ("系统只提供信息，不代替人操作",),
    "建议": ("建议必须给出理由与依据", "人始终是执行者"),
    "草稿": ("内容生效前必须由人确认", "草稿要留修改记录"),
    "带审批自动": ("指定审批人且审批可追溯", "不可逆动作要有回滚路径", "审批超时默认不执行"),
    "自动": ("异常率超阈值自动降级", "按比例抽样人工复核", "失败必须告警而不是静默重试"),
    "全自动": ("全量执行日志与审计追溯", "一键停用开关", "定期人工复检并回收授权"),
}

# 可自动化程度 = 确定性 × 0.55 + 频次 × 0.45。
# 确定性权重更高：判据硬的任务自动化才划算，判据软的任务自动化只会把混乱放大。
BASE_WEIGHTS: dict[str, float] = {"determinism": 0.55, "frequency": 0.45}

# 可自动化程度 → 等级的下界（从小到大找第一个满足的）。
LEVEL_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.85, "全自动"),
    (0.70, "自动"),
    (0.55, "带审批自动"),
    (0.38, "草稿"),
    (0.18, "建议"),
)

# 风险给出的等级上限。风险是硬约束：不可逆的动作不管多值得自动化，
# 也不能超过带审批自动这一格——这是整套规则里唯一不让步的一条。
RISK_CAPS: tuple[tuple[float, str], ...] = (
    (0.66, "带审批自动"),
    (0.33, "自动"),
)

# 工具权限三档。
TOOL_LEVELS: tuple[str, ...] = ("read", "write", "dangerous")

# 工具清单里每个工具必须声明的字段。
REQUIRED_TOOL_FIELDS: tuple[str, ...] = ("level", "failure_mode", "confirmation")


@dataclass
class Level:
    """一个自动化等级判定结果。"""

    name: str
    index: int
    rationale: str
    guardrails: list[str] = field(default_factory=list)


def _level_index_by_base(base: float) -> int:
    """由可自动化程度算出等级索引。"""
    for threshold, name in LEVEL_THRESHOLDS:
        if base >= threshold:
            return LEVEL_NAMES.index(name)
    return 0


def _level_cap_by_risk(risk: float) -> int:
    """由风险算出等级上限索引。"""
    for threshold, name in RISK_CAPS:
        if risk >= threshold:
            return LEVEL_NAMES.index(name)
    return LEVEL_NAMES.index("全自动")


def automation_level(risk: float, frequency: float, determinism: float) -> Level:
    """按风险 × 频次 × 确定性推荐自动化等级。

    三个输入都归一化到 0–1：
    - risk：错误代价的不可逆程度（0 = 错了能改回来，1 = 错了不可逆）；
    - frequency：这件事的发生频率（0 = 一年几次，1 = 天天发生）；
    - determinism：判断标准的客观程度（0 = 全靠人判断，1 = 有明确判据）。

    算成两段：先按「可自动化程度」定一个等级，再按风险压一个上限，取两者更保守的那个。
    写成"取更小值"而不是"分一堆 if"，是为了让等级随风险单调不升成为结构保证，
    而不是靠人记住每条分支——这种规则一旦靠人记，早晚会在某条分支上漏掉约束。
    """
    for name, value in (("风险", risk), ("频次", frequency), ("确定性", determinism)):
        if not 0 <= value <= 1:
            raise FixtureError(f"{name}必须落在 0–1 之间，实际 {value!r}")

    base = (
        BASE_WEIGHTS["determinism"] * determinism
        + BASE_WEIGHTS["frequency"] * frequency
    )
    base_index = _level_index_by_base(base)
    cap_index = _level_cap_by_risk(risk)
    index = min(base_index, cap_index)
    name = LEVEL_NAMES[index]

    if index == cap_index and cap_index < base_index:
        why = f"判据和频次都够（可自动化程度 {base:.2f}），但风险压住了上限——只能到{name}"
    elif index == base_index and base_index > cap_index:
        why = f"风险可接受，判据与频次定级为{name}"
    else:
        why = f"可自动化程度 {base:.2f}，风险 {risk:.2f} → {name}"

    return Level(name=name, index=index, rationale=why, guardrails=list(LEVEL_GUARDRAILS[name]))


def check_tool_manifest(tools: list[dict]) -> Report:
    """体检工具清单：权限级别、失败模式、确认方式都必须声明。

    另外检出"声明只读、实际在写"这类越权——声明和真实效果不一致时，
    审批闸门形同虚设，这是最容易被忽略也最危险的一类问题。
    """
    report = Report(subject="工具权限清单", passed=True)
    for tool in tools:
        name = tool.get("name", "<未命名工具>")
        missing = [field_name for field_name in REQUIRED_TOOL_FIELDS if field_name not in tool]
        if missing:
            report.problems.append(f"{name}：缺少必填字段 {'、'.join(missing)}")
            continue

        level = tool["level"]
        if level not in TOOL_LEVELS:
            report.problems.append(
                f"{name}：level 取值 {level!r} 非法（只能是 {'/'.join(TOOL_LEVELS)}）"
            )
            continue
        if not str(tool["failure_mode"]).strip():
            report.problems.append(f"{name}：failure_mode 不能为空")
            continue
        if not isinstance(tool["confirmation"], bool):
            report.problems.append(f"{name}：confirmation 必须是布尔值")
            continue
        if level == "dangerous" and not tool["confirmation"]:
            report.problems.append(f"{name}：dangerous 级工具必须要求确认")
            continue

        effect = tool.get("effect", "")
        if level == "read" and effect in {"write", "delete"}:
            report.problems.append(
                f"{name}：声明为 read 但实际效果是 {effect}——越权，审批闸门会被绕过"
            )
            continue

        report.ok.append(f"{name}（{level}）")
    report.passed = not report.problems
    return report


# 工具清单样例：最后一个是刻意的越权示例，用来演示体检能把它揪出来。
TOOL_MANIFESTS: list[dict] = [
    {
        "name": "search_kb",
        "level": "read",
        "failure_mode": "检索服务超时或返回空结果",
        "confirmation": False,
        "effect": "read",
    },
    {
        "name": "create_ticket",
        "level": "write",
        "failure_mode": "重复创建同一工单",
        "confirmation": True,
        "effect": "write",
    },
    {
        "name": "assign_ticket",
        "level": "write",
        "failure_mode": "分派到不存在的队列",
        "confirmation": True,
        "effect": "write",
    },
    {
        "name": "refund_order",
        "level": "dangerous",
        "failure_mode": "错误退款不可逆",
        "confirmation": True,
        "effect": "delete",
    },
    {
        "name": "export_records",
        "level": "read",
        "failure_mode": "导出接口超时",
        "confirmation": False,
        "effect": "write",
    },
]


def run() -> None:
    """跑第 5 周实验：自动化等级 → 兜底项 → 工具权限体检。"""
    print("=" * 78)
    print("第 5 周 · 自动化等级与工具权限")
    print("=" * 78)

    # ---- 1. 自动化等级 ----
    print()
    print("【1】按风险 × 频次 × 确定性定级")
    cases = [
        ("工单自动分类", 0.2, 0.9, 0.9),
        ("合同条款抽取", 0.7, 0.8, 0.9),
        ("会议纪要转任务", 0.4, 0.9, 0.8),
        ("退款操作", 0.9, 0.3, 0.6),
        ("舆情周报初稿", 0.5, 0.2, 0.3),
    ]
    print(
        fmt_table(
            ["场景", "风险", "频次", "确定性", "等级", "为什么"],
            [
                [name, f"{risk:.2f}", f"{freq:.2f}", f"{det:.2f}", level.name, level.rationale]
                for name, risk, freq, det in cases
                for level in [automation_level(risk, freq, det)]
            ],
            align_right={1, 2, 3},
            max_widths=[18, 6, 6, 8, 12, 46],
        )
    )

    # ---- 2. 每级的兜底项 ----
    print()
    print("【2】每一级必须有的兜底项")
    for name in LEVEL_NAMES:
        print(f"  ▸ {name}：")
        for item in LEVEL_GUARDRAILS[name]:
            print(f"      - {item}")

    # ---- 3. 工具权限体检 ----
    print()
    print("【3】工具权限体检")
    report = check_tool_manifest(TOOL_MANIFESTS)
    print(f"  通过项：{'、'.join(report.ok) if report.ok else '—'}")
    if report.problems:
        print("  问题项：")
        for problem in report.problems:
            print(f"      ✗ {problem}")

    # ---- 断言 ----
    assert len(LEVEL_NAMES) == 6, "自动化等级必须是六级"
    levels = {name: automation_level(risk, freq, det).name for name, risk, freq, det in cases}
    assert len(set(levels.values())) >= 4, "不同输入应该给出不同等级"
    for name in LEVEL_NAMES:
        assert LEVEL_GUARDRAILS[name], f"{name} 级必须列出兜底项"

    # 单调性：风险越高，等级不得越激进（索引不增）。
    for freq in (0.2, 0.6, 0.9):
        for det in (0.3, 0.6, 0.9):
            indexes = [automation_level(risk, freq, det).index for risk in (0.1, 0.5, 0.9)]
            assert indexes == sorted(indexes, reverse=True), (
                f"频次 {freq} / 确定性 {det} 下，等级随风险应当单调不升：{indexes}"
            )

    assert not report.passed, "样例清单里有一个越权工具，体检必须不通过"
    assert any("越权" in problem for problem in report.problems), "必须点名越权的那一个"
    assert len(report.ok) == 4, "四个合规工具应当通过"

    missing_field = check_tool_manifest([{"name": "bad_tool", "level": "write"}])
    assert not missing_field.passed, "缺字段的工具必须不合格"
    assert any("缺少必填字段" in problem for problem in missing_field.problems)

    no_confirm = check_tool_manifest(
        [{"name": "wipe", "level": "dangerous", "failure_mode": "不可逆", "confirmation": False}]
    )
    assert not no_confirm.passed, "dangerous 级没有确认必须不合格"
    assert any("必须要求确认" in problem for problem in no_confirm.problems)

    print()
    print("要点：自动化等级不是「模型行不行」，是「这件事错了谁负责」。")
    print("      风险高就永远留一道人工闸门，哪怕它很烦人。")
    print("      工具权限体检里最危险的不是缺字段，是「声明只读、实际在写」——")
    print("      审批闸门会被绕过，出事时没人知道发生过什么。")
