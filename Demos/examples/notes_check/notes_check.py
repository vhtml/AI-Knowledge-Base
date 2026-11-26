"""校验教程笔记的结构完整性。

这个模块解决什么问题：教程的十六章正文、四个附录和索引是一份**手工维护的长文档**，
最容易坏在四处——链接指向了不存在的文件、表格列数错位、代码围栏没闭合、占位符忘了删。
这些错误肉眼看一遍很难发现，但会让读者在某个链接上白跑一趟。

它是纯只读检查：不联网、不调模型、不写任何文件。
用法：
    cd Demos/examples
    uv run --project .. python -m notes_check
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

# 教程的章节清单：文件名 → 标题。附录与索引单列。
CHAPTER_FILES = (
    "01-AI 产品经理的角色、能力模型与边界.md",
    "02-技术地基：模型能力边界、上下文与推理参数.md",
    "03-场景筛选：什么该用 AI、什么不该.md",
    "04-用户研究：还原真实工作流与需求优先级.md",
    "05-竞品分析与市场地图.md",
    "06-定位、价值主张与商业模式.md",
    "07-对话式体验设计：信任、不确定性与拒答.md",
    "08-知识类产品设计：检索与引用.md",
    "09-Agent 与工作流：从 Copilot 到 Autopilot.md",
    "10-工具、多模态与企业系统集成.md",
    "11-评测体系：AI 产品的质量标准.md",
    "12-成本、延迟与单位经济.md",
    "13-安全、合规与风险治理.md",
    "14-指标体系、数据闭环与 PMF.md",
    "15-从 0 到 1：一个 AI 产品的完整落地流程.md",
    "16-跨职能协作、需求文档与职业发展.md",
)

APPENDIX_FILES = (
    "附录-A-术语表.md",
    "附录-B-模板库.md",
    "附录-C-资源与工具清单.md",
    "附录-D-案例库.md",
)

INDEX_FILE = "README.md"

# 每章必须有的结构件。少了任何一件，读者都会在某处找不到东西。
# 标题允许带序号（"## 9. 本章练习"），所以比对的是"标题里含不含这个词"。
REQUIRED_SECTIONS = (
    "> **这篇要回答**",
    "本章练习",
    "本章要点",
    "一句话结论",
)

HEADING_PATTERN = re.compile(r"^#{2,4}\s+(.+?)\s*$", re.MULTILINE)
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
PLACEHOLDER_MARKERS = ("TODO", "待补", "待填", "FIXME", "XXX")


@dataclass
class Report:
    """一次校验的结论。"""

    name: str
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


def repo_root() -> Path:
    """定位仓库根目录：从本文件向上找带 AGENTS.md 的目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


def notes_dir() -> Path:
    """被校验的教程目录。"""
    return repo_root() / "AI" / "AI-Product"


def check_files_exist() -> Report:
    """十六章 + 四附录 + 索引都在。"""
    report = Report("文件齐全")
    base = notes_dir()
    for name in (*CHAPTER_FILES, *APPENDIX_FILES, INDEX_FILE):
        if not (base / name).is_file():
            report.problems.append(f"缺少文件：{name}")
    return report


def check_chapter_structure() -> Report:
    """每章都有固定的四件结构件。"""
    report = Report("章节结构")
    base = notes_dir()
    for name in CHAPTER_FILES:
        path = base / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        number = name[:2]
        if not text.startswith(f"# {number} · "):
            report.problems.append(f"{name} 的首行标题应是 `# {number} · 标题`")
        headings = [h.lstrip("0123456789. 、") for h in HEADING_PATTERN.findall(text)]
        for section in REQUIRED_SECTIONS:
            if section.startswith(">"):
                if section not in text:
                    report.problems.append(f"{name} 缺少「{section}」")
            elif not any(section in heading for heading in headings):
                report.problems.append(f"{name} 缺少「{section}」小节")
    return report


def check_code_fences() -> Report:
    """代码围栏成对。

    围栏没闭合会让后面的内容整段被当成代码渲染——这是最隐蔽也最致命的排版错误。
    """
    report = Report("代码围栏")
    base = notes_dir()
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fences = [line for line in text.splitlines() if line.strip().startswith("```")]
        if len(fences) % 2 != 0:
            report.problems.append(f"{path.name} 的代码围栏数量为奇数（{len(fences)}）")
    return report


def check_relative_links() -> Report:
    """所有相对链接都指向真实存在的文件。"""
    report = Report("相对链接")
    base = notes_dir()
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for raw in LINK_PATTERN.findall(text):
            if raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(raw.split("#", 1)[0]).strip()
            if not target:
                continue
            if not (path.parent / target).exists():
                report.problems.append(f"{path.name} 指向不存在的目标：{raw}")
    return report


def check_index_covers_all() -> Report:
    """索引把十六章和四份附录都列进去了。"""
    report = Report("索引覆盖")
    index = notes_dir() / INDEX_FILE
    if not index.is_file():
        report.problems.append("缺少 README.md")
        return report
    text = index.read_text(encoding="utf-8")
    # README 里链接目标写成 %20 编码，文件名本身却是空格——两边都要试。
    decoded = unquote(text)
    for name in (*CHAPTER_FILES, *APPENDIX_FILES):
        if name not in text and name not in decoded:
            report.problems.append(f"README.md 没有列出 {name}")
    return report


def check_no_placeholder_left() -> Report:
    """不留 TODO / 待补 这类占位。"""
    report = Report("无占位符")
    base = notes_dir()
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for marker in PLACEHOLDER_MARKERS:
            if marker in text:
                report.problems.append(f"{path.name} 残留占位符：{marker}")
    return report


def split_cells(line: str) -> list[str]:
    """拆一行表格的单元格。

    要同时躲开两种情况：单元格里的转义竖线 ``\\|``，以及行内代码里的竖线
    （`` `|error` `` 这种）。所以不能直接 str.split("|")。
    """
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    escaped = False
    for char in line.strip():
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            current.append(char)
            escaped = True
        elif char == "`":
            in_code = not in_code
            current.append(char)
        elif char == "|" and not in_code:
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
    cells.append("".join(current))
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def _table_problems(name: str, block: list[tuple[int, str]]) -> list[str]:
    """检查一段连续表格行的列数是否自洽。"""
    problems: list[str] = []
    widths = {number: len(split_cells(line)) for number, line in block}
    if len(set(widths.values())) > 1:
        detail = "、".join(f"第 {number} 行 {width} 列" for number, width in widths.items())
        problems.append(f"{name} 的表格列数不一致：{detail}")
    if len(block) >= 2:
        separator = split_cells(block[1][1])
        if not all(set(cell.strip()) <= set("-: ") and "-" in cell for cell in separator):
            problems.append(f"{name} 第 {block[1][0]} 行不是表格分隔行")
    return problems


def check_table_integrity() -> Report:
    """每张表的列数一致，且第二行是分隔行。

    笔记里的表是主要信息载体，列数错位比少写一句话更难读，
    而且渲染出来很难一眼看出问题在哪，所以值得单独查一道。
    """
    report = Report("表格完整性")
    base = notes_dir()
    for path in sorted(base.glob("*.md")):
        in_fence = False
        block: list[tuple[int, str]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                if block:
                    report.problems.extend(_table_problems(path.name, block))
                    block = []
                continue
            if in_fence:
                continue
            if line.strip().startswith("|"):
                block.append((number, line))
            elif block:
                report.problems.extend(_table_problems(path.name, block))
                block = []
        if block:
            report.problems.extend(_table_problems(path.name, block))
    return report


CHECKS = (
    check_files_exist,
    check_chapter_structure,
    check_code_fences,
    check_relative_links,
    check_index_covers_all,
    check_no_placeholder_left,
    check_table_integrity,
)


def check_all() -> list[Report]:
    """跑完全部检查项。"""
    return [check() for check in CHECKS]


def assert_all_passed() -> None:
    """给自检用：任何一项不过就抛断言错误。"""
    for report in check_all():
        assert report.passed, f"{report.name}：" + "；".join(report.problems)


def run() -> int:
    """独立运行时的入口。"""
    reports = check_all()
    failures = 0
    for report in reports:
        if report.passed:
            print(f"  ✓ {report.name}")
        else:
            failures += 1
            print(f"  ✗ {report.name} —— {'；'.join(report.problems)}")

    print()
    if failures:
        print(f"笔记校验未通过：{failures}/{len(reports)} 项失败")
        return 1
    print(f"笔记校验通过：{len(reports)}/{len(reports)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(run())
