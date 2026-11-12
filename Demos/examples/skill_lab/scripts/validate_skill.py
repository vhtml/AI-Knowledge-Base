#!/usr/bin/env python3
"""校验一个技能目录是否符合发布要求。

这个脚本解决什么问题：技能打包前需要一道自动化的门禁。
人工检查会漏，脚本不会——它把第 2、4、5 天讲过的规则固化成可重复执行的检查，
并用退出码告诉调用方"通过"还是"有问题"。

用法：
    python validate_skill.py <技能目录>
    python validate_skill.py <技能目录> --json

退出码：
    0  全部检查通过
    1  存在检查失败项
    2  用法错误（目录不存在等）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 与官方 quick_validate.py 对齐的硬约束
ALLOWED_FRONTMATTER_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}
NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
NAME_MAX_LEN = 64
DESCRIPTION_MAX_LEN = 1024

# 本教程补充的质量约束
BODY_LINE_LIMIT = 500
REFERENCE_DEPTH_LIMIT = 1
TOC_LINE_THRESHOLD = 100
FORBIDDEN_FILES = {"README.md", "INSTALLATION_GUIDE.md", "QUICK_REFERENCE.md", "CHANGELOG.md"}


def check_frontmatter(meta: dict[str, str], body: str, dir_name: str) -> list[dict[str, str]]:
    """检查 frontmatter，返回问题列表。"""
    issues: list[dict[str, str]] = []

    def fail(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    unexpected = set(meta) - ALLOWED_FRONTMATTER_KEYS
    if unexpected:
        fail("error", f"frontmatter 出现不允许的键：{', '.join(sorted(unexpected))}")

    name = meta.get("name", "")
    if not name:
        fail("error", "缺少 name")
    else:
        if not NAME_PATTERN.match(name):
            fail("error", f"name '{name}' 必须是 hyphen-case（小写字母、数字、连字符）")
        if len(name) > NAME_MAX_LEN:
            fail("error", f"name 超过 {NAME_MAX_LEN} 字符")
        if name.startswith("-") or name.endswith("-") or "--" in name:
            fail("error", f"name '{name}' 不能以 - 开头/结尾，也不能有连续 --")
        if name != dir_name:
            fail("warn", f"name '{name}' 与目录名不一致")

    description = meta.get("description", "")
    if not description:
        fail("error", "缺少 description")
    else:
        if len(description) > DESCRIPTION_MAX_LEN:
            fail("error", f"description 超过 {DESCRIPTION_MAX_LEN} 字符")
        if "<" in description or ">" in description:
            fail("error", "description 不能包含尖括号 < 或 >")
        if len(description) < 30:
            fail("warn", "description 过短，很可能触发不准")

    # "何时使用"写在正文里是无效的——那是 description 的活
    if re.search(r"何时使用|When to use|使用场景", body):
        fail("warn", "正文里出现了'何时使用'字样：触发信息必须写在 description 里，正文只写怎么用")

    return issues


def check_structure(skill_dir: Path, body: str) -> list[dict[str, str]]:
    """检查目录结构与资源组织。"""
    issues: list[dict[str, str]] = []

    def fail(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    body_lines = len(body.splitlines())
    if body_lines > BODY_LINE_LIMIT:
        fail("error", f"正文 {body_lines} 行，超过 {BODY_LINE_LIMIT} 行上限，应拆到 references/")

    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for path in sorted(ref_dir.rglob("*.md")):
            depth = len(path.relative_to(ref_dir).parts) - 1
            if depth > REFERENCE_DEPTH_LIMIT:
                fail("error", f"{path.relative_to(skill_dir).as_posix()} 嵌套深度 {depth}，超过 {REFERENCE_DEPTH_LIMIT} 层")
            content = path.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            if lines > TOC_LINE_THRESHOLD and "## 目录" not in content and "## Table of Contents" not in content:
                fail("warn", f"{path.relative_to(skill_dir).as_posix()} 有 {lines} 行却没有目录，建议加 ## 目录")

    for path in sorted(skill_dir.rglob("*")):
        if path.is_file() and path.name in FORBIDDEN_FILES:
            fail("warn", f"存在不该有的文件：{path.relative_to(skill_dir).as_posix()}（这是给人看的，不是给模型看的）")

    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        for path in sorted(scripts_dir.rglob("*")):
            if path.is_file() and path.suffix == ".py":
                source = path.read_text(encoding="utf-8")
                if "def main(" not in source and "__main__" not in source:
                    fail("warn", f"{path.relative_to(skill_dir).as_posix()} 看起来没有可直接执行的入口")

    return issues


def _parse_frontmatter(raw: str) -> dict[str, str]:
    """解析 frontmatter 的顶层 key: value。

    这里刻意不依赖 PyYAML，也不复用 labkit——这个脚本要能单独拷进任意技能的
    scripts/ 目录里跑。但必须支持 YAML 块标量（`description: |` / `>-`），
    否则折叠写法的描述会被解析成 ">"，进而被误判成"含尖括号"。
    """
    lines = raw.splitlines()
    meta: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#") or line[:1] in (" ", "\t"):
            i += 1
            continue
        key, sep, value = line.partition(":")
        if not sep:
            i += 1
            continue
        key = key.strip()
        value = value.strip()
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            block: list[str] = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
                if lines[i].strip():
                    block.append(lines[i].strip())
                i += 1
            meta[key] = " ".join(block)
            continue
        meta[key] = value.strip('"').strip("'")
        i += 1
    return meta


def validate_skill(skill_dir: Path) -> tuple[bool, list[dict[str, str]]]:
    """校验一个技能目录，返回 (是否通过, 问题列表)。"""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return False, [{"level": "error", "message": "SKILL.md not found"}]

    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return False, [{"level": "error", "message": "No YAML frontmatter found"}]

    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return False, [{"level": "error", "message": "Invalid frontmatter format"}]

    raw = match.group(1)
    body = text[match.end():]

    meta = _parse_frontmatter(raw)
    body = text[match.end():]

    issues = check_frontmatter(meta, body, skill_dir.name) + check_structure(skill_dir, body)
    passed = not any(i["level"] == "error" for i in issues)
    return passed, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="校验技能目录是否符合发布要求")
    parser.add_argument("skill_dir", help="技能目录（内含 SKILL.md）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args()

    skill_dir = Path(args.skill_dir).expanduser()
    if not skill_dir.is_dir():
        print(f"目录不存在：{skill_dir}", file=sys.stderr)
        return 2

    passed, issues = validate_skill(skill_dir)

    if args.json:
        print(json.dumps({
            "skill": skill_dir.name,
            "passed": passed,
            "issues": issues,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"技能：{skill_dir.name}")
        if passed:
            print("结果：通过")
        else:
            print("结果：不通过")
        for issue in issues:
            mark = "✗" if issue["level"] == "error" else "!"
            print(f"  {mark} [{issue['level']}] {issue['message']}")
        if not issues:
            print("  没有发现问题")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
