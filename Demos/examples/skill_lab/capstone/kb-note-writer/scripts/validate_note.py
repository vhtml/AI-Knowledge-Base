#!/usr/bin/env python3
"""校验一篇知识笔记是否符合本仓库约定。

这个脚本解决什么问题：笔记是"写给自己半年后看的"，但半年后的自己已经不记得当时的约束。
把约定变成可执行的检查，比记在脑子里可靠。

用法：
    python validate_note.py <笔记路径>
    python validate_note.py <笔记路径> --strict

退出码：
    0  通过
    1  存在必须修的问题
    2  用法错误（文件不存在等）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 敏感信息特征——命中即报错，笔记里绝不能出现这些
SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "疑似 OpenAI 风格 API Key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "疑似 GitHub Token"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[A-Za-z0-9/+_\-]{16,}"), "疑似硬编码的密钥"),
    (re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@"), "含账号密码的数据库连接串"),
    (re.compile(r"\b10\.\d+\.\d+\.\d+\b"), "疑似内网地址"),
]

# 硬编码的敏感变量名——笔记里提到它们是允许的，但写成赋值就是问题
SECRET_ASSIGN = re.compile(r"(?i)\b(?:api_key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]")


def check_structure(text: str) -> list[str]:
    """检查标题层级、开头定位、代码块语言标注。"""
    problems: list[str] = []
    lines = text.splitlines()

    if not lines or not lines[0].startswith("# "):
        problems.append("第一行必须是 H1 标题（# 主题名）")

    # 标题层级不能跳级（H1 -> H3）
    levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6}) ", text, re.MULTILINE)]
    for prev, curr in zip(levels, levels[1:]):
        if curr > prev + 1:
            problems.append(f"标题层级跳级：H{prev} 之后直接出现 H{curr}")
            break

    # 开头第一段要说清"解决什么问题"
    body = "\n".join(lines[1:]).strip()
    first_para = body.split("\n\n")[0] if body else ""
    if first_para and not re.search(r"解决|什么时候|翻它|什么时候该", first_para):
        problems.append("开头第一段没有说明这篇解决什么问题、什么时候该翻它")

    # 代码块必须标语言
    for i, line in enumerate(lines):
        if line.strip().startswith("```") and line.strip() != "```":
            lang = line.strip()[3:].strip()
            if not lang:
                problems.append(f"第 {i + 1} 行的代码块没有标语言")

    return problems


def check_secrets(text: str) -> list[str]:
    """检查敏感信息。"""
    problems: list[str] = []
    for pattern, hint in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            problems.append(f"{hint}：{match.group(0)[:12]}…（已截断）")
    if SECRET_ASSIGN.search(text):
        problems.append("疑似把密钥写成了字面量赋值")
    return problems


def check_links(text: str, note_path: Path) -> list[str]:
    """检查相对链接指向的文件是否存在。"""
    problems: list[str] = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (note_path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            problems.append(f"链接指向不存在的文件：{target}")
    return problems


def check_length(text: str, strict: bool) -> list[str]:
    """检查篇幅。"""
    problems: list[str] = []
    lines = len(text.splitlines())
    if lines > 300:
        problems.append(f"篇幅 {lines} 行，超过 300 行，建议拆成两篇")
    elif strict and lines > 200:
        problems.append(f"篇幅 {lines} 行，超过 strict 模式的 200 行上限")
    return problems


def validate_note(note_path: Path, strict: bool = False) -> list[str]:
    """校验一篇笔记，返回问题列表（空列表表示通过）。"""
    text = note_path.read_text(encoding="utf-8")
    problems: list[str] = []
    problems += check_secrets(text)
    problems += check_structure(text)
    problems += check_links(text, note_path)
    problems += check_length(text, strict)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="校验知识笔记是否符合本仓库约定")
    parser.add_argument("note", help="笔记文件路径")
    parser.add_argument("--strict", action="store_true", help="启用更严的篇幅限制")
    args = parser.parse_args()

    note_path = Path(args.note).expanduser()
    if not note_path.is_file():
        print(f"文件不存在：{note_path}", file=sys.stderr)
        return 2

    problems = validate_note(note_path, strict=args.strict)
    print(f"笔记：{note_path}")
    if problems:
        print(f"结果：不通过（{len(problems)} 个问题）")
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("结果：通过")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
