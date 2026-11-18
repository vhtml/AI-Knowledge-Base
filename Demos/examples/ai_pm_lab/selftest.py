"""ai_pm_lab 自检：验证实验骨架本身没坏。

这个模块解决什么问题：实验代码本身也要有回归保障。
自检不依赖 pytest、不联网、不调模型，验证的是"素材是完整的、骨架是能跑的、
校验器确实会拦住残缺方案包"——而不是"教程内容讲对了"，后者由各周模块里的断言负责。

用法：
    cd Demos/examples
    python -m ai_pm_lab selftest
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = LAB_ROOT.parent

from .labkit import CAPSTONE_DIR, FixtureError, load_fixture  # noqa: E402
from .weeks import WEEKS, load_week  # noqa: E402

# 各素材文件里必须有的条数。
EXPECTED_COUNTS = {
    "capabilities": ("dimensions", 8),
    "scenarios": ("scenarios", 12),
    "competitors": ("competitors", 6),
    "eval_cases": ("cases", 20),
    "retrieval_cases": ("cases", 10),
}


def check_fixtures_parse() -> None:
    """素材 JSON 都能解析，且条数正确。"""
    for name, (key, expected) in EXPECTED_COUNTS.items():
        data = load_fixture(name)
        assert key in data, f"{name}.json 缺少 {key} 字段"
        actual = len(data[key])
        assert actual == expected, f"{name}.json 的 {key} 应有 {expected} 条，实际 {actual}"
    cost = load_fixture("cost_model")
    assert cost["variables"], "cost_model.json 的 variables 不能为空"
    metrics = load_fixture("metrics")
    assert metrics["north_star"]["name"], "metrics.json 缺北极星指标"
    assert metrics["guardrails"], "metrics.json 缺护栏指标"
    assert metrics["diagnostics"], "metrics.json 缺诊断指标"
    assert len(metrics["funnel"]) >= 2, "漏斗至少要两环"


def check_fixture_fields() -> None:
    """素材里的字段与枚举合法。"""
    from .weeks.week2 import CRITERIA_FIELDS, CRITERIA_WEIGHTS, REQUIRED_FIELDS
    from .weeks.week3 import QUADRANTS
    from .weeks.week5 import LEVEL_NAMES, TOOL_LEVELS
    from .weeks.week7 import METRIC_LAYERS, RISK_FIELDS

    assert abs(sum(CRITERIA_WEIGHTS.values()) - 100) < 1e-9, "四判据权重之和必须是 100"
    assert len(REQUIRED_FIELDS) == 10, "场景必填字段应为 10 个"
    assert len(QUADRANTS) == 4 and len(LEVEL_NAMES) == 6 and len(TOOL_LEVELS) == 3
    assert len(METRIC_LAYERS) == 3 and len(RISK_FIELDS) == 7

    for scenario in load_fixture("scenarios")["scenarios"]:
        for field in REQUIRED_FIELDS:
            assert field in scenario, f"场景 {scenario.get('id', '?')} 缺字段 {field}"
        for criterion in CRITERIA_WEIGHTS:
            field_name = CRITERIA_FIELDS[criterion]
            assert isinstance(scenario[field_name], bool), f"{field_name} 必须是布尔值"

    competitors = load_fixture("competitors")
    for item in competitors["competitors"]:
        assert item["workflow_depth"] in competitors["axes"]["workflow_depth"], (
            f"竞品 {item['name']} 的工作流深度取值非法"
        )
        assert item["automation_level"] in competitors["axes"]["automation_level"], (
            f"竞品 {item['name']} 的自动化等级取值非法"
        )
        assert not any(key in item for key in ("market_share", "funding", "revenue")), (
            "竞品表里不允许出现市场份额 / 融资额这类编造数字"
        )

    for case in load_fixture("eval_cases")["cases"]:
        assert isinstance(case["critical"], list) and isinstance(case["errors"], list), (
            f"用例 {case.get('id', '?')} 的 critical / errors 必须是列表"
        )


def check_weeks_registry() -> None:
    """八周模块都在，都导出 run()，且与登记表一致。"""
    assert len(WEEKS) == 8, f"应有 8 周实验，实际 {len(WEEKS)}"
    assert [key for key, _, _ in WEEKS] == [f"week{index}" for index in range(1, 9)], (
        "周的键必须是 week1..week8 且顺序一致"
    )
    for key, title, _ in WEEKS:
        assert title.strip(), f"{key} 缺标题"
        module = load_week(key)
        assert module is not None, f"{key} 模块导入失败"
        assert hasattr(module, "run"), f"{key} 缺少 run()"


def check_labkit_boundaries() -> None:
    """公共工具层的边界行为。"""
    from .labkit import clamp, display_width, fmt_table, parse_md_table, pct, truncate

    assert display_width("中文") == 4, "全角字符应占两列"
    assert display_width("ab") == 2
    assert truncate("中文中文", 4) == "中…", "截断要按显示宽度算，并给省略号留一列"
    assert truncate("abc", 10) == "abc", "够短就不截断"
    assert pct(0.5) == "50.0%"
    assert clamp(2.0, 0.0, 1.0) == 1.0

    table = fmt_table(["名称", "值"], [["甲", "1"], ["乙乙乙", "22"]])
    lines = table.splitlines()
    assert len(lines) == 4, "表头 + 分隔线 + 两行"
    assert all(display_width(line) <= 30 for line in lines), "表格不能超出限宽"

    header, rows = parse_md_table("| a | b |\n| --- | --- |\n| 1 | 2 |\n")
    assert header == ["a", "b"]
    assert rows == [{"a": "1", "b": "2"}]

    try:
        load_fixture("不存在的素材")
    except FixtureError:
        pass
    else:
        raise AssertionError("缺素材必须报错")


def check_week_functions() -> None:
    """各周的关键函数在边界输入下不出错。"""
    from .weeks.week1 import TASK_TYPES, boundary_card, radar
    from .weeks.week2 import ai_fit_score, priority
    from .weeks.week3 import price_anchor, quadrant_name
    from .weeks.week4 import check_answer, retrieval_metrics
    from .weeks.week5 import automation_level, check_tool_manifest
    from .weeks.week6 import gate, score, unit_economics
    from .weeks.week7 import check_risk_register, metric_tree

    capabilities = load_fixture("capabilities")
    scores = {dim["id"]: 3 for dim in capabilities["dimensions"]}
    assert len(radar(scores)) == 8, "全 3 分时八项都是短板"
    assert len(TASK_TYPES) >= 10
    assert "建议做法" in boundary_card("摘要")

    scenarios = load_fixture("scenarios")["scenarios"]
    for scenario in scenarios:
        fit = ai_fit_score(scenario)
        assert 0 <= fit.total <= 100
    assert len(priority(scenarios)) == len(scenarios)

    assert quadrant_name(0.85, 0.9) == "端到端自动平台"
    anchor = price_anchor(180, 120, 0.6)
    assert anchor["price_low"] < anchor["price_high"]

    assert not check_answer("一句话回答。").passed
    retrieval = retrieval_metrics(load_fixture("retrieval_cases")["cases"])
    assert set(retrieval["metrics"]) == {"recall", "citation", "faithfulness"}

    assert automation_level(0.1, 0.9, 0.9).name == "全自动"
    assert not check_tool_manifest([{"name": "x", "level": "dangerous"}]).passed

    result = score(load_fixture("eval_cases")["cases"])
    assert 0 <= result["total"] <= 100
    assert "不许变差" in gate(result, {**result, "total": result["total"] - 1})
    assert unit_economics(
        {name: float(spec["value"]) for name, spec in load_fixture("cost_model")["variables"].items()}
    )["gross_margin"] > 0

    assert metric_tree(load_fixture("metrics"))["passed"]
    assert not check_risk_register([{"风险描述": "只有描述"}]).passed


def check_capstone_artifacts() -> None:
    """毕业作品的五份产物齐全。"""
    expected = ["PRD.md", "eval_set.json", "metrics.md", "unit_economics.md", "risk_register.md"]
    for name in expected:
        path = CAPSTONE_DIR / name
        assert path.is_file(), f"缺少毕业作品产物 {name}"
        assert path.stat().st_size > 0, f"{name} 是空文件"

    data = json.loads((CAPSTONE_DIR / "eval_set.json").read_text(encoding="utf-8"))
    assert data["cases"], "评测集不能为空"


def check_capstone_prd_sections() -> None:
    """PRD 十节齐全。"""
    from .weeks.week8 import PRD_SECTIONS, check_prd
    from .labkit import read_text

    item = check_prd(read_text(CAPSTONE_DIR / "PRD.md"))
    assert item.passed, f"PRD 检查不通过：{item.detail}"
    assert len(PRD_SECTIONS) == 10, "PRD 必须要求十节"


def check_incomplete_spec_rejected() -> None:
    """故意残缺的方案包必须被校验器拦下（且 CLI 退出码非 0）。"""
    from .weeks.week8 import validate_spec

    with tempfile.TemporaryDirectory() as tmp:
        broken_dir = Path(tmp) / "ai-product-spec"
        shutil.copytree(CAPSTONE_DIR, broken_dir)

        prd_path = broken_dir / "PRD.md"
        text = prd_path.read_text(encoding="utf-8")
        # 删掉「上线与灰度」整节，制造一个缺节但文件仍在的方案包。
        kept: list[str] = []
        skipping = False
        for line in text.splitlines():
            if line.startswith("## "):
                skipping = "上线与灰度" in line
            if not skipping:
                kept.append(line)
        prd_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

        checklist = validate_spec(broken_dir)
        assert not checklist.passed, "缺节的方案包必须不通过"
        assert any("PRD" in item.name and not item.passed for item in checklist.items), (
            "必须点名 PRD 缺节"
        )

        env = dict(os.environ)
        env["AI_PM_LAB_SPEC_DIR"] = str(broken_dir)
        proc = subprocess.run(
            [sys.executable, "-m", "ai_pm_lab", "week8"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLES_DIR),
            env=env,
        )
        assert proc.returncode != 0, (
            f"残缺方案包的退出码必须非 0，实际 {proc.returncode}\n{proc.stdout}"
        )


def check_cli_list() -> None:
    """CLI 的 list 子命令能跑通，且列出八周。"""
    proc = subprocess.run(
        [sys.executable, "-m", "ai_pm_lab", "list"],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLES_DIR),
    )
    assert proc.returncode == 0, f"list 子命令失败：\n{proc.stdout}\n{proc.stderr}"
    for index in range(1, 9):
        assert f"week{index}" in proc.stdout, f"list 输出缺少 week{index}"


def check_notes_match_lab() -> None:
    """教程笔记与实验一一对应：章节齐、结构齐、互链可解析、章节引用的周对得上。"""
    from .notes_check import assert_all_passed

    assert_all_passed()


def run() -> int:
    """跑完全部自检项。"""
    checks = [
        ("素材可解析且条数正确", check_fixtures_parse),
        ("素材字段与枚举合法", check_fixture_fields),
        ("八周模块齐全", check_weeks_registry),
        ("公共工具边界行为", check_labkit_boundaries),
        ("各周关键函数", check_week_functions),
        ("毕业作品五份产物", check_capstone_artifacts),
        ("PRD 十节齐全", check_capstone_prd_sections),
        ("残缺方案包被拦下", check_incomplete_spec_rejected),
        ("CLI list 可用", check_cli_list),
        ("笔记与实验一一对应", check_notes_match_lab),
    ]

    failures: list[str] = []
    for name, func in checks:
        try:
            func()
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failures.append(f"{name}：{exc}")
            print(f"  ✗ {name} —— {exc}")
        except Exception as exc:  # noqa: BLE001 - 一项检查崩了不该中断整轮自检
            failures.append(f"{name}：{type(exc).__name__}: {exc}")
            print(f"  ✗ {name} —— {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"自检未通过：{len(failures)}/{len(checks)} 项失败")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"自检通过：{len(checks)}/{len(checks)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(run())
