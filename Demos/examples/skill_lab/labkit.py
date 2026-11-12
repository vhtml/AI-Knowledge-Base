"""Skill 教程实验的公共工具层。

这个模块解决什么问题：七天实验都要用到同一批"底层事实"——token 估算、frontmatter 解析、
技能目录发现、仓库路径定位。把它们集中在这里，各天实验只写当天的教学重点。

注意：token 估算是**启发式估算**（与本仓库 chat_cli 的口径一致），不是真实 tokenizer 的结果。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# frontmatter 的键白名单，与官方 quick_validate.py 保持一致
ALLOWED_FRONTMATTER_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}

# name 的命名约束
NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")

# 技能里不该出现的"给人看的"文件
FORBIDDEN_FILES = {
    "README.md",
    "INSTALLATION_GUIDE.md",
    "QUICK_REFERENCE.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
}


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数。

    口径和本仓库 chat_cli 一致：CJK 字符 1 字 ≈ 1 token，其余字符 4 个 ≈ 1 token。
    这是启发式估算，用来比较"同一段内容的不同写法"的相对成本，不能当作计费依据。
    """
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + (len(text) - cjk) // 4


def repo_root() -> Path:
    """定位仓库根目录：从本文件向上找带 AGENTS.md 的目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    # 兜底：按 Demos/examples/skill_lab 的深度往上数
    return Path(__file__).resolve().parents[3]


def _parse_yaml_block(raw: str) -> dict[str, str]:
    """解析 frontmatter 里的顶层 key: value。

    刻意只做最小解析：够本教程用，且不引入 PyYAML 依赖。
    但必须支持 YAML 块标量（`description: |` / `>-`）——真实的技能描述大量使用这种写法，
    漏掉它会把常驻成本算成 0。
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
            # 块标量：收集后续所有缩进行
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


def parse_frontmatter(text: str) -> tuple[dict[str, str], str, str]:
    """拆分 SKILL.md 的 frontmatter 与正文。

    返回 (元数据字典, 正文, 错误信息)。错误信息为空表示格式正确。
    这里刻意**不复刻**官方的全部校验规则，只做解析；
    规则校验交给 scripts/validate_skill.py。
    """
    if not text.startswith("---"):
        return {}, text, "缺少 frontmatter（文件必须以 --- 开头）"
    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return {}, text, "frontmatter 格式错误（需要 --- 开头、--- 结束的成对分隔）"

    meta = _parse_yaml_block(match.group(1))
    body = text[match.end():]
    return meta, body, ""


def read_skill(skill_md: Path) -> tuple[dict[str, str], str, str]:
    """读取一个 SKILL.md，返回 (元数据, 正文, 错误信息)。"""
    return parse_frontmatter(skill_md.read_text(encoding="utf-8"))


@dataclass
class SkillStat:
    """一个已安装技能的统计快照。"""

    name: str
    origin: str
    desc_tokens: int
    body_tokens: int
    resource_files: int

    @property
    def total_tokens(self) -> int:
        return self.desc_tokens + self.body_tokens


@dataclass
class SkillScan:
    """一次技能盘点的结果。"""

    stats: list[SkillStat] = field(default_factory=list)

    @property
    def always_on_tokens(self) -> int:
        """常驻成本：所有技能的 description 之和。"""
        return sum(s.desc_tokens for s in self.stats)

    @property
    def body_tokens(self) -> int:
        """正文总成本：所有技能正文之和——只有触发才会真的进上下文。"""
        return sum(s.body_tokens for s in self.stats)


def skill_roots() -> list[tuple[str, Path]]:
    """列出本机所有可能的技能安装根目录。

    覆盖 MCode 的三个位置、Claude 风格目录、以及本仓库 vendor 进来的第三方技能。
    """
    home = Path.home()
    roots: list[tuple[str, Path]] = [
        ("MCode 用户级", home / ".minimax" / "skills"),
        ("MCode agent 私有", home / ".minimax" / "agents" / "mavis" / "skills"),
        ("MCode 内置", home / ".minimax" / ".builtin-skills"),
        ("Claude 风格", home / ".claude" / "skills"),
        ("仓库 vendor", repo_root() / ".minimax" / "skills"),
        ("仓库示例", repo_root() / "Demos" / "examples" / "skills"),
    ]
    return [(label, path) for label, path in roots if path.is_dir()]


def scan_skills() -> SkillScan:
    """盘点本机所有技能，统计常驻成本与正文成本。"""
    scan = SkillScan()
    seen: set[str] = set()

    for label, root in skill_roots():
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue
            key = f"{label}/{skill_dir.name}"
            if key in seen:
                continue
            seen.add(key)

            meta, body, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            desc = meta.get("description", "")
            resources = sum(1 for p in skill_dir.rglob("*") if p.is_file() and p.name != "SKILL.md")
            scan.stats.append(
                SkillStat(
                    name=skill_dir.name,
                    origin=label,
                    desc_tokens=estimate_tokens(desc),
                    body_tokens=estimate_tokens(body),
                    resource_files=resources,
                )
            )
    return scan


def find_official_script(name: str) -> Path | None:
    """在已知的 skill-creator 安装位置里找官方脚本。

    找不到返回 None——实验会退回到本地实现，不因为缺一个脚本就整个跑不起来。
    """
    home = Path.home()
    candidates = [
        home / ".claude" / "skills" / "skill-creator" / "scripts" / name,
        home / ".minimax" / ".builtin-skills" / "skill-creator" / "scripts" / name,
        home / ".minimax" / "skills" / "skill-creator" / "scripts" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def fmt_table(headers: list[str], rows: list[list[str]], align_right: set[int] | None = None) -> str:
    """把二维数组渲染成等宽表格。"""
    align_right = align_right or set()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(str(cell).rjust(widths[i]) if i in align_right else str(cell).ljust(widths[i]))
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line([str(c) for c in row]) for row in rows)
    return "\n".join(out)
