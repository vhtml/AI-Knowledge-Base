"""skill_lab 自检：验证实验骨架本身没坏。

这个模块解决什么问题：实验代码本身也要有回归保障。
自检不依赖 pytest、不联网、不调模型，验证的是"实验的脚手架和素材是完整的"，
而不是"教程内容讲对了"——后者由每天 lesson 里的断言负责。

用法：
    python -m skill_lab.selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent

from .labkit import estimate_tokens, parse_frontmatter, scan_skills  # noqa: E402


def check_token_estimate() -> None:
    """token 估算的边界行为。"""
    assert estimate_tokens("") == 0
    assert estimate_tokens("中文") == 2, "CJK 应按 1 字 1 token 估"
    assert estimate_tokens("abcd") == 1, "英文应按 4 字符 1 token 估"
    assert estimate_tokens("abcdefgh") == 2


def check_frontmatter_parsing() -> None:
    """frontmatter 解析的三种情况：正常、缺分隔符、空。"""
    meta, body, err = parse_frontmatter("---\nname: x\ndescription: y\n---\n正文\n")
    assert err == "", f"正常 frontmatter 不该报错：{err}"
    assert meta["name"] == "x" and meta["description"] == "y"
    assert "正文" in body

    _, _, err = parse_frontmatter("没有 frontmatter\n")
    assert err, "缺 frontmatter 应该报错"

    _, _, err = parse_frontmatter("---\nname: x\n")
    assert err, "未闭合的 frontmatter 应该报错"


def check_skill_scan() -> None:
    """技能盘点能跑通且数字自洽。"""
    scan = scan_skills()
    assert scan.stats, "本机应该至少能扫到一个技能"
    assert scan.always_on_tokens == sum(s.desc_tokens for s in scan.stats)
    assert scan.body_tokens == sum(s.body_tokens for s in scan.stats)


def check_lessons_importable() -> None:
    """七天的 lesson 都在，且都导出 run()。"""
    from .lessons import DAYS

    assert len(DAYS) == 7, f"应有 7 天实验，实际 {len(DAYS)}"
    for key, title, module in DAYS:
        assert hasattr(module, "run"), f"{key} 缺少 run()"
        assert isinstance(title, str) and title


def check_fixtures() -> None:
    """评测集素材完整且格式正确。"""
    cases_path = LAB_ROOT / "fixtures" / "trigger_cases.json"
    assert cases_path.is_file(), "缺少评测集 fixtures/trigger_cases.json"

    data = json.loads(cases_path.read_text(encoding="utf-8"))
    assert data["trigger_terms"], "触发词表不能为空"
    assert data["guard_terms"], "守卫词表不能为空"

    cases = data["cases"]
    positives = [c for c in cases if c["should_trigger"]]
    negatives = [c for c in cases if not c["should_trigger"]]
    near = [c for c in negatives if c["kind"] == "near-miss"]

    assert len(positives) == 10, f"应有 10 条正例，实际 {len(positives)}"
    assert len(negatives) == 10, f"应有 10 条负例，实际 {len(negatives)}"
    assert len(near) >= 3, f"至少要 3 条近邻负例，实际 {len(near)}"

    for c in cases:
        assert c["query"].strip(), "用例的 query 不能为空"
        assert c["kind"] in {"clear", "near-miss"}, f"未知的 kind：{c['kind']}"


def check_capstone_structure() -> None:
    """毕业作品的目录结构完整。"""
    capstone = LAB_ROOT / "capstone" / "kb-note-writer"
    assert capstone.is_dir(), "缺少毕业作品 capstone/kb-note-writer"
    assert (capstone / "SKILL.md").is_file(), "缺少 SKILL.md"
    assert (capstone / "references").is_dir(), "缺少 references/"
    assert (capstone / "scripts").is_dir(), "缺少 scripts/"
    assert (capstone / "assets").is_dir(), "缺少 assets/"

    meta, body, err = parse_frontmatter((capstone / "SKILL.md").read_text(encoding="utf-8"))
    assert err == "", f"毕业作品的 frontmatter 有问题：{err}"
    assert meta.get("name") == "kb-note-writer"
    assert len(meta.get("description", "")) <= 1024, "description 超长"
    assert "<" not in meta.get("description", "") and ">" not in meta.get("description", "")
    assert len(body.splitlines()) <= 500, "正文超过 500 行"
    assert "何时使用" not in body, "正文不该出现'何时使用'——那是 description 的活"


def check_validator_script() -> None:
    """本校验脚本能跑，且对毕业作品返回 0。"""
    validator = LAB_ROOT / "scripts" / "validate_skill.py"
    assert validator.is_file(), "缺少 scripts/validate_skill.py"

    proc = subprocess.run(
        [sys.executable, str(validator), str(LAB_ROOT / "capstone" / "kb-note-writer")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"毕业作品应通过校验：\n{proc.stdout}\n{proc.stderr}"


def run() -> int:
    """跑完全部自检项。"""
    checks = [
        ("token 估算", check_token_estimate),
        ("frontmatter 解析", check_frontmatter_parsing),
        ("技能盘点", check_skill_scan),
        ("lesson 可导入", check_lessons_importable),
        ("评测集素材", check_fixtures),
        ("毕业作品结构", check_capstone_structure),
        ("校验脚本", check_validator_script),
    ]

    failures: list[str] = []
    for name, fn in checks:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failures.append(f"{name}：{exc}")
            print(f"  ✗ {name} —— {exc}")

    print()
    if failures:
        print(f"自检未通过：{len(failures)}/{len(checks)} 项失败")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"自检通过：{len(checks)}/{len(checks)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(run())
