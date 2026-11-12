"""第 7 天实验：打包分发与生态。

这个模块解决什么问题：把毕业作品从"一个目录"变成"一个可分发的安装包"，并验证它装得上。
它优先调用官方 quick_validate.py / package_skill.py；本机没有官方脚本时，
退回到本地等价实现，保证实验在任何机器上都能跑完。

对应笔记：AI/Skill/07-打包分发与生态.md
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from ..labkit import find_official_script
from .day5_scripts import run_validator

CAPSTONE = Path(__file__).resolve().parent.parent / "capstone" / "kb-note-writer"
DIST = Path(__file__).resolve().parent.parent / "dist"


def run_official_quick_validate(skill_dir: Path) -> tuple[bool, str]:
    """调用官方校验脚本，返回 (是否通过, 输出)。"""
    script = find_official_script("quick_validate.py")
    if script is None:
        return False, "（本机没有官方 quick_validate.py，跳过）"
    proc = subprocess.run([sys.executable, str(script), str(skill_dir)], capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_official_package(skill_dir: Path, out_dir: Path) -> Path | None:
    """调用官方打包脚本，产出 .skill 文件。"""
    script = find_official_script("package_skill.py")
    if script is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(script), str(skill_dir), str(out_dir)],
        capture_output=True,
        text=True,
    )
    produced = out_dir / f"{skill_dir.name}.skill"
    return produced if produced.is_file() else None


def local_package(skill_dir: Path, out_dir: Path) -> Path:
    """本地等价打包：zip 结构与官方一致（包内自带技能目录名作为根）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{skill_dir.name}.skill"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(skill_dir.parent))
    return target


def publish_checklist(skill_dir: Path) -> list[tuple[str, bool, str]]:
    """逐条跑发布前检查清单。"""
    checks: list[tuple[str, bool, str]] = []

    code, out = run_validator(skill_dir)
    checks.append(("校验脚本通过（退出码 0）", code == 0, out.splitlines()[-1] if out else ""))

    skill_md = skill_dir / "SKILL.md"
    body = skill_md.read_text(encoding="utf-8").split("---", 2)[-1]
    lines = len(body.splitlines())
    checks.append(("正文 ≤ 500 行", lines <= 500, f"当前 {lines} 行"))

    refs = list((skill_dir / "references").rglob("*.md")) if (skill_dir / "references").is_dir() else []
    depth = max((len(p.relative_to(skill_dir / 'references').parts) - 1 for p in refs), default=0)
    checks.append(("references 只一层深", depth <= 1, f"当前最大深度 {depth}"))

    forbidden = [p.name for p in skill_dir.rglob("*") if p.is_file() and p.name in {"README.md", "CHANGELOG.md"}]
    checks.append(("没有 README / CHANGELOG", not forbidden, "、".join(forbidden) or "无"))

    scripts = [p for p in (skill_dir / "scripts").rglob("*.py")] if (skill_dir / "scripts").is_dir() else []
    all_run = True
    for s in scripts:
        proc = subprocess.run([sys.executable, str(s), "--help"], capture_output=True, text=True)
        all_run = all_run and proc.returncode == 0
    checks.append(("脚本都能执行（--help 返回 0）", all_run, f"{len(scripts)} 个脚本"))

    checks.append(("有 references 分流", len(refs) > 0, f"{len(refs)} 份参考文档"))
    checks.append(("有 assets 模板", (skill_dir / "assets").is_dir(), "assets/"))

    return checks


def run() -> None:
    print("=" * 78)
    print("第 7 天 · 打包分发与生态")
    print("=" * 78)

    if not CAPSTONE.is_dir():
        print(f"找不到毕业作品：{CAPSTONE}")
        return

    # ---- 1. 官方校验 ----
    print()
    print("【1】官方校验")
    ok, out = run_official_quick_validate(CAPSTONE)
    print(f"  {out}")
    if ok:
        print("  官方校验通过")

    # ---- 2. 本仓库校验脚本（更严） ----
    print()
    print("【2】本仓库的校验脚本（多查了正文行数、references 深度、多余文件）")
    code, out = run_validator(CAPSTONE)
    print(out)
    print(f"  退出码 = {code}")
    assert code == 0, "毕业作品必须通过本校验脚本"

    # ---- 3. 发布前清单 ----
    print()
    print("【3】发布前检查清单")
    checks = publish_checklist(CAPSTONE)
    for name, passed, detail in checks:
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name:32} {detail}")
    assert all(p for _, p, _ in checks), "发布前清单必须全部通过"

    # ---- 4. 打包 ----
    print()
    print("【4】打包")
    produced = run_official_package(CAPSTONE, DIST)
    if produced is not None:
        print(f"  官方 package_skill.py 产出：{produced}")
    else:
        produced = local_package(CAPSTONE, DIST)
        print(f"  本机没有官方 package_skill.py，使用本地等价实现产出：{produced}")

    assert produced.is_file()

    with zipfile.ZipFile(produced) as zf:
        names = zf.namelist()
    print()
    print("  包内结构：")
    for name in sorted(names):
        print(f"    {name}")
    assert f"{CAPSTONE.name}/SKILL.md" in names, "包内必须自带技能目录名作为根"
    assert not any("__pycache__" in n for n in names), "包里不该有 __pycache__"

    # ---- 5. 安装模拟 ----
    print()
    print("【5】安装模拟：解压到临时目录并校验")
    tmp = Path(tempfile.mkdtemp(prefix="skill-install-"))
    try:
        with zipfile.ZipFile(produced) as zf:
            zf.extractall(tmp)
        installed = tmp / CAPSTONE.name
        assert (installed / "SKILL.md").is_file(), "解压后应能找到 SKILL.md"
        code, out = run_validator(installed)
        print(f"  解压到：{installed}")
        print(f"  安装后复检退出码 = {code}")
        assert code == 0
        print("  安装后的技能能被正确识别（下一个会话生效）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"  产物：{produced}")
    print()
    print("要点：.skill 就是 zip，包内自带技能目录名；")
    print("      MCode 装到 ~/.minimax/skills/<name>/，下个会话生效。")


if __name__ == "__main__":
    run()
