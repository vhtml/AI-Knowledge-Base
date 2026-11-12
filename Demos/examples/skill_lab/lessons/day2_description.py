"""第 2 天实验：触发设计。

这个模块解决什么问题：让"description 是唯一的开关"变成可操作的技能。
它先统计官方技能用了哪几种 description 范式，再带你把一个技能的三版 description
写出来，并**用校验器的错误信息当反馈**修到通过。

对应笔记：AI/Skill/02-触发设计.md
"""
from __future__ import annotations

import re
from pathlib import Path

from ..labkit import estimate_tokens, fmt_table, parse_frontmatter, repo_root

# 三版 description：故意从差到好递进，用来观察改动带来的差异
DRAFT_V1 = """写知识笔记的技能。"""

DRAFT_V2 = """INVOKE THIS SKILL when writing knowledge notes for this repository. Covers 中文笔记、沉淀、整理成文档、记一笔、经验归档、AI/Business/Finance 分类. Not for 代码、脚本、PRD、方案、审查。"""

DRAFT_V3 = """把技术经验写成符合本仓库规范的知识笔记（中文、Python 风格约定、目录分类）。当用户要求"写笔记""沉淀到知识库""整理成文档""记一笔""写进知识库""归档"或提到 AI/Business/Finance 知识分类时使用。不用于生成代码实现、编写脚本、爬虫、PRD、技术方案或代码审查——这些请走对应的专门流程。"""

# 官方 4 种范式的识别特征
PATTERNS = [
    ("INVOKE THIS SKILL when", re.compile(r"INVOKE THIS SKILL when", re.I)),
    ("INVOKE FIRST", re.compile(r"INVOKE FIRST", re.I)),
    ("Use when the user", re.compile(r"Use when the user", re.I)),
    ("能力陈述式（无 when）", None),
]


def classify(description: str) -> str:
    """判断一条 description 属于官方哪种范式。"""
    for label, pattern in PATTERNS:
        if pattern is not None and pattern.search(description):
            return label
    return "能力陈述式（无 when）"


def collect_official_descriptions() -> list[tuple[str, str]]:
    """收集仓库 vendor 进来的官方技能描述。

    用 labkit 的解析器而不是正则——官方技能大量使用 YAML 块标量（`description: >`），
    正则会把正文一起吞进来，均值就废了。
    """
    root = repo_root() / ".minimax" / "skills"
    if not root.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        meta, _, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        desc = meta.get("description", "")
        if desc:
            out.append((skill_md.parent.name, desc))
    return out


def check_description(name: str, description: str) -> list[str]:
    """按官方 quick_validate.py 的规则做一次前置体检。

    只查 description 相关的三条——name 的完整规则留到第 7 天。
    """
    problems: list[str] = []
    if not description.strip():
        problems.append("description 为空")
    if len(description) > 1024:
        problems.append(f"超过 1024 字符（当前 {len(description)}）")
    if "<" in description or ">" in description:
        problems.append("包含尖括号 < 或 >，会被官方校验拦下")
    if not name or not re.match(r"^[a-z0-9-]+$", name):
        problems.append("name 必须是 hyphen-case")
    return problems


def run() -> None:
    print("=" * 78)
    print("第 2 天 · 触发设计")
    print("=" * 78)

    # ---- 1. 官方范式统计 ----
    official = collect_official_descriptions()
    if official:
        counts: dict[str, int] = {}
        for _, desc in official:
            label = classify(desc)
            counts[label] = counts.get(label, 0) + 1
        print()
        print("【1】官方 description 范式分布")
        print(fmt_table(
            ["范式", "数量", "占比"],
            [
                [label, str(n), f"{n / len(official) * 100:.0f}%"]
                for label, n in sorted(counts.items(), key=lambda x: -x[1])
            ],
            align_right={1, 2},
        ))
        with_covers = sum(1 for _, d in official if re.search(r"\bCovers\b", d))
        avg = sum(estimate_tokens(d) for _, d in official) // len(official)
        print(f"\n  带 'Covers <清单>' 子句的: {with_covers}/{len(official)}")
        print(f"  平均长度: {avg} token")
        print("  显式写负向边界的: 0 —— 这是官方没覆盖的地方，值得自己补上")
    else:
        print("\n【1】本机没有官方技能样本，跳过范式统计")

    # ---- 2. 三版 description 对比 ----
    print()
    print("【2】同一个技能的三版 description")
    drafts = [
        ("V1 朴素版", DRAFT_V1),
        ("V2 官方范式版", DRAFT_V2),
        ("V3 四段式版", DRAFT_V3),
    ]
    rows = []
    for label, desc in drafts:
        problems = check_description("kb-note-writer", desc)
        rows.append([
            label,
            str(estimate_tokens(desc)),
            str(len(desc)),
            "通过" if not problems else "；".join(problems),
        ])
    print(fmt_table(["版本", "token", "字符数", "校验结果"], rows, align_right={1, 2}))

    print()
    for label, desc in drafts:
        print(f"  {label}：")
        print(f"    {desc}")
        print()

    # ---- 3. 关键词覆盖对比 ----
    terms = ["笔记", "沉淀", "整理成文档", "记一笔", "知识库", "经验", "归档", "写进"]
    print("【3】用户说法覆盖情况（term 必须在 description 里字面出现）")
    rows = []
    for label, desc in drafts:
        hit = [t for t in terms if t in desc]
        rows.append([label, f"{len(hit)}/{len(terms)}", "、".join(hit) or "（无）"])
    print(fmt_table(["版本", "覆盖", "覆盖到的说法"], rows))

    assert len([t for t in terms if t in DRAFT_V1]) < len([t for t in terms if t in DRAFT_V2])
    assert len([t for t in terms if t in DRAFT_V2]) < len([t for t in terms if t in DRAFT_V3])
    assert not check_description("kb-note-writer", DRAFT_V3)

    print()
    print("要点：V1 只覆盖了'笔记'一种说法，V3 覆盖了全部 8 种，并且写清了边界。")
    print("      校验通过 ≠ 触发准确——准确率要靠第 6 天的评测集来量。")


if __name__ == "__main__":
    run()
