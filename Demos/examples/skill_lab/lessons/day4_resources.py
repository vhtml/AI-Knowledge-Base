"""第 4 天实验：资源拆分与渐进式披露。

这个模块解决什么问题：让"什么该拆、拆到多深、哪些文件不该存在"变成可执行的检查。
它实现一个技能目录体检器，量化拆分前后"触发后必读部分"的体积变化，
并模拟官方 init_skill.py 生成骨架、再把没用的示例文件清干净的全过程。

对应笔记：AI/Skill/04-资源拆分与渐进式披露.md
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..labkit import estimate_tokens, fmt_table, repo_root

REFERENCE_DEPTH_LIMIT = 1
TOC_LINE_THRESHOLD = 100


@dataclass
class AuditResult:
    """技能目录体检结果。"""

    skill: str
    has_skill_md: bool
    body_lines: int
    body_tokens: int
    ref_files: int
    max_ref_depth: int
    forbidden: list[str]
    missing_toc: list[str]
    script_files: int
    asset_files: int

    @property
    def problems(self) -> list[str]:
        out: list[str] = []
        if not self.has_skill_md:
            out.append("缺少 SKILL.md")
        if self.body_lines > 500:
            out.append(f"正文 {self.body_lines} 行，超过 500 行，应该拆 references")
        if self.max_ref_depth > REFERENCE_DEPTH_LIMIT:
            out.append(f"references 嵌套深度 {self.max_ref_depth}，超过 {REFERENCE_DEPTH_LIMIT} 层")
        for f in self.forbidden:
            out.append(f"存在不该有的文件：{f}")
        for f in self.missing_toc:
            out.append(f"reference 超过 {TOC_LINE_THRESHOLD} 行却没有目录：{f}")
        return out


def audit_skill_dir(skill_dir: Path) -> AuditResult:
    """体检一个技能目录。"""
    skill_md = skill_dir / "SKILL.md"
    has_skill_md = skill_md.is_file()

    body_lines = body_tokens = 0
    if has_skill_md:
        text = skill_md.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        body = parts[-1] if len(parts) >= 3 else text
        body_lines = len(body.splitlines())
        body_tokens = estimate_tokens(body)

    ref_dir = skill_dir / "references"
    ref_files = sorted(ref_dir.rglob("*.md")) if ref_dir.is_dir() else []
    max_depth = 0
    missing_toc: list[str] = []
    for f in ref_files:
        depth = len(f.relative_to(ref_dir).parts) - 1
        max_depth = max(max_depth, depth)
        content = f.read_text(encoding="utf-8")
        if len(content.splitlines()) > TOC_LINE_THRESHOLD and "## 目录" not in content and "## Table of Contents" not in content:
            missing_toc.append(f.relative_to(skill_dir).as_posix())

    forbidden = sorted(
        p.name for p in skill_dir.rglob("*")
        if p.is_file() and p.name in {"README.md", "INSTALLATION_GUIDE.md", "QUICK_REFERENCE.md", "CHANGELOG.md"}
    )

    script_dir = skill_dir / "scripts"
    asset_dir = skill_dir / "assets"

    return AuditResult(
        skill=skill_dir.name,
        has_skill_md=has_skill_md,
        body_lines=body_lines,
        body_tokens=body_tokens,
        ref_files=len(ref_files),
        max_ref_depth=max_depth,
        forbidden=forbidden,
        missing_toc=missing_toc,
        script_files=sum(1 for p in script_dir.rglob("*") if p.is_file()) if script_dir.is_dir() else 0,
        asset_files=sum(1 for p in asset_dir.rglob("*") if p.is_file()) if asset_dir.is_dir() else 0,
    )


def scaffold_skill(name: str, target: Path) -> Path:
    """本地复刻官方 init_skill.py 的骨架产出。

    官方脚本在本机不一定存在，所以这里自己生成一份等价的骨架，
    顺便让"脚手架会生成一堆用不上的示例文件"这件事变得可见。
    """
    skill_dir = target / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "assets").mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: " + name + "\n"
        "description: TODO: 用一句话说明这个技能做什么、什么时候用。\n"
        "---\n\n"
        "# " + name + "\n\n"
        "TODO: 说明这个技能解决什么问题。\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "example.py").write_text(
        "# TODO: 删掉或替换成真正有用的脚本\nprint('hello')\n", encoding="utf-8"
    )
    (skill_dir / "references" / "example.md").write_text(
        "# TODO: 删掉或替换成真正有用的参考文档\n", encoding="utf-8"
    )
    (skill_dir / "assets" / "example.txt").write_text("TODO: 删掉或替换\n", encoding="utf-8")
    return skill_dir


# 模拟拆分：把一份长正文拆成 SKILL.md + 两份 reference
SPLIT_SKILL_MD = """# 报告生成

## 快速开始

运行 `scripts/build_report.py --input <数据> --out <报告>`。

## 进阶功能

- **销售维度**：见 [SALES.md](references/sales.md)
- **财务维度**：见 [FINANCE.md](references/finance.md)
"""

SPLIT_REFERENCE = """# {title}

## 目录
- [口径定义](#口径定义)
- [常见问题](#常见问题)

## 口径定义

{body}

## 常见问题

遇到对不上的数据时，先核对口径，再怀疑数据源。
"""


def run() -> None:
    print("=" * 78)
    print("第 4 天 · 资源拆分与渐进式披露")
    print("=" * 78)

    # ---- 1. 体检仓库现有技能 ----
    print()
    print("【1】体检仓库现有的技能目录")
    for candidate in [
        repo_root() / "Demos" / "examples" / "skills" / "demo-assistant",
        repo_root() / ".minimax" / "skills" / "langchain-rag",
    ]:
        if not candidate.is_dir():
            continue
        result = audit_skill_dir(candidate)
        print(fmt_table(
            ["指标", "值"],
            [
                ["技能", result.skill],
                ["正文行数", str(result.body_lines)],
                ["正文 token", str(result.body_tokens)],
                ["references 文件", str(result.ref_files)],
                ["references 最大深度", str(result.max_ref_depth)],
                ["scripts 文件", str(result.script_files)],
                ["assets 文件", str(result.asset_files)],
                ["不该有的文件", "、".join(result.forbidden) or "（无）"],
            ],
            align_right={1},
        ))
        for p in result.problems:
            print(f"  ✗ {p}")
        print()

    # ---- 2. 拆分前后的必读体积 ----
    print("【2】拆分前后的「触发后必读体积」对比")
    long_body = "这是一段很长的正文内容。" * 400  # 模拟一份臃肿技能
    before_tokens = estimate_tokens(long_body)
    after_tokens = estimate_tokens(SPLIT_SKILL_MD)
    print(f"  拆分前（全部正文）: {before_tokens:,} token")
    print(f"  拆分后（必读部分）: {after_tokens:,} token")
    print(f"  触发一个场景时的实际读取: ~{after_tokens:,} token，节省 {(1 - after_tokens / before_tokens) * 100:.1f}%")

    assert after_tokens < before_tokens

    # ---- 3. 脚手架 → 体检 → 清理 ----
    print()
    print("【3】脚手架生成 → 体检 → 清理")
    tmp = Path(tempfile.mkdtemp(prefix="skill-lab-"))
    try:
        scaffold = scaffold_skill("demo-scaffold", tmp)
        before_clean = audit_skill_dir(scaffold)
        print(f"  刚生成时的体检问题：")
        for p in before_clean.problems:
            print(f"    ✗ {p}")
        if not before_clean.problems:
            print("    （无）")

        # 用不上的示例文件必须删掉
        removed = 0
        for junk in [
            scaffold / "scripts" / "example.py",
            scaffold / "references" / "example.md",
            scaffold / "assets" / "example.txt",
        ]:
            if junk.is_file():
                junk.unlink()
                removed += 1
        print(f"  清理掉 {removed} 个用不上的示例文件")

        after_clean = audit_skill_dir(scaffold)
        print(f"  清理后正文 token: {after_clean.body_tokens}（模板里的 TODO 也要改）")
        print("  目录结构：")
        for p in sorted(scaffold.rglob("*")):
            print(f"    {p.relative_to(scaffold).as_posix()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("要点：拆分不是为了显得专业，而是为了让触发后**必读**的部分尽量小；")
    print("      同时不要往 skill 里塞给人看的 README / CHANGELOG。")


if __name__ == "__main__":
    run()
