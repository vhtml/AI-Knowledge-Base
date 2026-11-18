"""第 8 周实验：毕业作品校验。

这个模块解决什么问题：把"我写了个 AI 产品方案"变成"这份方案过了哪些硬性检查"。
校验器逐项检查毕业作品包里的五份产物——PRD 十节、评测集、指标字典、单位经济、风险登记册，
输出逐项 checklist 与总评；任何一项不通过，总评就是不通过，进程退出码非 0。

这是整套教程的收口：前七周练的是怎么把一件事想清楚、写下来、算明白；
第八周检查的是"想清楚的东西有没有真的落在文档里"。

对应笔记：AI/AI-Product/ 第 15 章《方案写作》与第 16 章《毕业标准》。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..labkit import CAPSTONE_DIR, Checklist, ChecklistItem, fmt_table, load_json, read_text

# PRD 必须有的十节。关键词写全称，匹配时去掉空格再比对——
# 「成本与 SLO」和「成本与SLO」应该是同一节，不能因为排版差一个空格就算缺节。
PRD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("问题", "谁在什么场景下被什么卡住"),
    ("目标与非目标", "这次要解决什么，明确不解决什么"),
    ("用户与场景", "谁在什么时候用它做什么"),
    ("方案", "产品怎么做，含交互与模型分工"),
    ("评测标准", "怎么判断做对了，含关键错误定义"),
    ("失败态", "做不出来、出错、越权时怎么办"),
    ("成本与SLO", "单次成本、延迟目标与降级策略"),
    ("指标", "北极星、护栏、诊断三层"),
    ("风险", "风险登记册摘要与缓解措施"),
    ("上线与灰度", "放量节奏、回滚条件与值班安排"),
)

# 评测集里每条用例的必填字段。
EVAL_REQUIRED_FIELDS: tuple[str, ...] = ("id", "task", "input", "expected")

# 指标字典必须出现的两个小节标题。
METRICS_SECTIONS: tuple[str, ...] = ("北极星", "护栏")

# 单位经济文档必须出现的两样东西。
UNIT_ECONOMICS_KEYWORDS: tuple[str, ...] = ("公式", "变量")

# 风险登记册必填字段，与第 7 周一致。
RISK_FIELDS: tuple[str, ...] = (
    "风险描述",
    "触发条件",
    "影响",
    "等级",
    "缓解措施",
    "责任人",
    "复检周期",
)

# 校验用的方案包目录，可用环境变量指向另一份方案包（自检用它验证残缺副本会被拦下）。
SPEC_DIR_ENV = "AI_PM_LAB_SPEC_DIR"


def spec_dir() -> Path:
    """当前要校验的方案包目录。"""
    override = os.environ.get(SPEC_DIR_ENV)
    return Path(override).expanduser() if override else CAPSTONE_DIR


def _headings(text: str) -> list[str]:
    """取出 Markdown 里的标题行（去掉 # 与空格）。"""
    out: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", line)
        if match:
            out.append(match.group(1))
    return out


def _normalize(text: str) -> str:
    """比较用的归一化：去掉空格与全角括号里的空白。"""
    return re.sub(r"\s+", "", text)


def check_prd(text: str) -> ChecklistItem:
    """检查 PRD 是否包含十节。"""
    headings = [_normalize(item) for item in _headings(text)]
    missing = [
        f"{index}. {name}"
        for index, (name, _) in enumerate(PRD_SECTIONS, start=1)
        if not any(_normalize(name) in heading for heading in headings)
    ]
    if missing:
        return ChecklistItem(
            name="PRD 十节齐全",
            passed=False,
            detail=f"缺 {len(missing)} 节：{'、'.join(missing)}",
        )
    return ChecklistItem(
        name="PRD 十节齐全",
        passed=True,
        detail=f"十节都在：{'、'.join(name for name, _ in PRD_SECTIONS)}",
    )


def check_eval_set(data: object) -> ChecklistItem:
    """检查评测集非空且字段齐全。"""
    if not isinstance(data, dict) or "cases" not in data:
        return ChecklistItem(name="评测集", passed=False, detail="评测集缺少 cases 列表")
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        return ChecklistItem(name="评测集", passed=False, detail="评测集为空")
    problems: list[str] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            problems.append(f"第 {index} 条不是对象")
            continue
        missing = [key for key in EVAL_REQUIRED_FIELDS if not str(case.get(key, "")).strip()]
        if missing:
            problems.append(f"{case.get('id', f'第 {index} 条')} 缺 {'、'.join(missing)}")
    if problems:
        return ChecklistItem(
            name="评测集",
            passed=False,
            detail=f"{len(cases)} 条用例，{len(problems)} 处问题：{'；'.join(problems[:3])}",
        )
    return ChecklistItem(name="评测集", passed=True, detail=f"{len(cases)} 条用例字段齐全")


def check_metrics(text: str) -> ChecklistItem:
    """检查指标字典是否含北极星与护栏。"""
    headings = [_normalize(item) for item in _headings(text)]
    missing = [
        name for name in METRICS_SECTIONS if not any(_normalize(name) in heading for heading in headings)
    ]
    if missing:
        return ChecklistItem(name="指标字典", passed=False, detail=f"缺小节：{'、'.join(missing)}")
    return ChecklistItem(name="指标字典", passed=True, detail="北极星与护栏小节都在")


def check_unit_economics(text: str) -> ChecklistItem:
    """检查单位经济文档是否写了公式与变量定义。"""
    body = _normalize(text)
    missing = [keyword for keyword in UNIT_ECONOMICS_KEYWORDS if _normalize(keyword) not in body]
    if missing:
        return ChecklistItem(
            name="单位经济",
            passed=False,
            detail=f"缺 {'、'.join(missing)}",
        )
    return ChecklistItem(name="单位经济", passed=True, detail="公式与变量定义都在")


def check_risk_register(text: str) -> ChecklistItem:
    """检查风险登记册的必填字段是否齐全。"""
    from ..labkit import parse_md_table

    _, rows = parse_md_table(text)
    if not rows:
        return ChecklistItem(name="风险登记册", passed=False, detail="登记册为空或没有表格")
    problems: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing = [key for key in RISK_FIELDS if not str(row.get(key, "")).strip()]
        if missing:
            problems.append(f"{row.get('编号', f'第 {index} 条')} 缺 {'、'.join(missing)}")
    if problems:
        return ChecklistItem(
            name="风险登记册",
            passed=False,
            detail=f"{len(rows)} 条风险，{len(problems)} 处问题：{'；'.join(problems[:3])}",
        )
    return ChecklistItem(name="风险登记册", passed=True, detail=f"{len(rows)} 条风险字段齐全")


def validate_spec(directory: Path) -> Checklist:
    """校验一份方案包，输出逐项 checklist。"""
    checklist = Checklist()

    prd_path = directory / "PRD.md"
    if not prd_path.is_file():
        checklist.items.append(ChecklistItem(name="PRD 十节齐全", passed=False, detail=f"缺少文件：{prd_path.name}"))
    else:
        checklist.items.append(check_prd(read_text(prd_path)))

    eval_path = directory / "eval_set.json"
    if not eval_path.is_file():
        checklist.items.append(ChecklistItem(name="评测集", passed=False, detail=f"缺少文件：{eval_path.name}"))
    else:
        checklist.items.append(check_eval_set(load_json(eval_path)))

    metrics_path = directory / "metrics.md"
    if not metrics_path.is_file():
        checklist.items.append(ChecklistItem(name="指标字典", passed=False, detail=f"缺少文件：{metrics_path.name}"))
    else:
        checklist.items.append(check_metrics(read_text(metrics_path)))

    econ_path = directory / "unit_economics.md"
    if not econ_path.is_file():
        checklist.items.append(ChecklistItem(name="单位经济", passed=False, detail=f"缺少文件：{econ_path.name}"))
    else:
        checklist.items.append(check_unit_economics(read_text(econ_path)))

    risk_path = directory / "risk_register.md"
    if not risk_path.is_file():
        checklist.items.append(ChecklistItem(name="风险登记册", passed=False, detail=f"缺少文件：{risk_path.name}"))
    else:
        checklist.items.append(check_risk_register(read_text(risk_path)))

    return checklist


def run() -> None:
    """跑第 8 周实验：逐项校验毕业作品并给出总评。"""
    print("=" * 78)
    print("第 8 周 · 毕业作品校验")
    print("=" * 78)

    directory = spec_dir()
    print()
    print(f"【1】校验方案包：{directory}")
    checklist = validate_spec(directory)
    print(
        fmt_table(
            ["检查项", "结果", "说明"],
            [
                [item.name, "通过" if item.passed else "不通过", item.detail]
                for item in checklist.items
            ],
            max_widths=[16, 8, 60],
        )
    )

    passed_count = sum(1 for item in checklist.items if item.passed)
    print()
    print(f"【2】总评：{checklist.summary}（{passed_count}/{len(checklist.items)} 项通过）")
    if not checklist.passed:
        print("      不通过时进程退出码为非 0——方案包不合格就不该上线。")

    # ---- 断言 ----
    assert len(checklist.items) == 5, "必须校验五份产物"
    assert checklist.passed, "毕业作品必须通过全部检查：\n" + "\n".join(
        f"  - {item.name}：{item.detail}" for item in checklist.items if not item.passed
    )

    print()
    print("要点：这份方案包能过检，不代表它会成功——")
    print("      检查项只保证「该想的都写了、该算的都算了」，不保证判断是对的。")
    print("      所以第 6 周的门禁要一直挂在发布流程上，而不是只在交付前跑一次。")
