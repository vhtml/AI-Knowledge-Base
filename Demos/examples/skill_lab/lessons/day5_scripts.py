"""第 5 天实验：脚本化。

这个模块解决什么问题：把"脚本要真的跑过"变成可验证的动作。
它调起同目录下的 scripts/validate_skill.py，检查退出码是否可靠，
再故意制造三类问题（非法 frontmatter、超长正文、多余 README），确认脚本**能检出**，
最后修好并复检。

对应笔记：AI/Skill/05-脚本化.md
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..labkit import repo_root

VALIDATOR = Path(__file__).resolve().parent.parent / "scripts" / "validate_skill.py"


def run_validator(skill_dir: Path, use_json: bool = False) -> tuple[int, str]:
    """调用校验脚本，返回 (退出码, 输出)。"""
    cmd = [sys.executable, str(VALIDATOR), str(skill_dir)]
    if use_json:
        cmd.append("--json")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def make_broken_skill(base: Path) -> Path:
    """造一个有典型问题的技能目录。"""
    skill_dir = base / "broken-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Broken_Skill\n"          # 非法：不是 hyphen-case
        "description: 短\n"             # 警告：过短
        "author: someone\n"             # 非法：不在键白名单里
        "---\n\n"
        + "占位行。\n" * 520,            # 非法：正文超 500 行
        encoding="utf-8",
    )
    (skill_dir / "README.md").write_text("这是给人看的说明，模型不需要。\n", encoding="utf-8")
    nested = skill_dir / "references" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "extra.md").write_text("# 嵌套太深\n", encoding="utf-8")
    return skill_dir


def run() -> None:
    print("=" * 78)
    print("第 5 天 · 脚本化")
    print("=" * 78)

    # ---- 1. --help 是模型能读的说明书 ----
    print()
    print("【1】脚本的说明书")
    proc = subprocess.run([sys.executable, str(VALIDATOR), "--help"], capture_output=True, text=True)
    for line in proc.stdout.strip().splitlines():
        print(f"  {line}")

    # ---- 2. 校验一个健康技能 ----
    print()
    print("【2】校验健康的技能")
    demo = repo_root() / "Demos" / "examples" / "skills" / "demo-assistant"
    if demo.is_dir():
        code, out = run_validator(demo)
        print(out)
        print(f"  退出码 = {code}（0 表示通过）")
        assert code == 0, f"demo-assistant 应该通过校验，实际退出码 {code}"

    # ---- 3. JSON 输出：stdout 要干净 ----
    print()
    print("【3】JSON 输出（stdout 只有 JSON，可以直接重定向）")
    if demo.is_dir():
        code, out = run_validator(demo, use_json=True)
        print("  " + out.replace("\n", "\n  "))
        assert code == 0

    # ---- 4. 故意搞坏，确认能检出 ----
    print()
    print("【4】故意搞坏：确认脚本能检出问题")
    tmp = Path(tempfile.mkdtemp(prefix="skill-lab-"))
    try:
        broken = make_broken_skill(tmp)
        code, out = run_validator(broken)
        print(out)
        print(f"  退出码 = {code}（1 表示不通过）")
        assert code == 1, "有问题的技能必须返回非零退出码"

        # ---- 5. 修好，再校验 ----
        print()
        print("【5】修好之后复检")
        (broken / "SKILL.md").write_text(
            "---\n"
            "name: broken-skill\n"
            "description: 演示如何被校验脚本拦下的技能，用于教学；触发词：校验、门禁、退出码。\n"
            "---\n\n"
            "# broken-skill\n\n"
            "## 何时使用本技能\n\n"
            "这个段落本身就是反面教材。\n",
            encoding="utf-8",
        )
        (broken / "README.md").unlink()
        shutil.rmtree(broken / "references")
        code, out = run_validator(broken)
        print(out)
        print(f"  退出码 = {code}")
        assert code == 0, f"修好后应该通过，实际退出码 {code}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("要点：退出码是脚本与模型之间最可靠的接口——")
    print("      模型不需要解析输出文字，看 exit code 就知道成功还是失败。")


if __name__ == "__main__":
    run()
