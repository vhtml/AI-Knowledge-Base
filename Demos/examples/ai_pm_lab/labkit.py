"""AI 产品经理教程实验的公共工具层。

这个模块解决什么问题：八周实验共用同一批"底层事实"——素材加载、评分口径、数据模型、表格渲染。
把它们集中在这里，各周实验只写当周的教学重点；要改口径只改一处，不会出现"第 3 周扣 10 分、
第 6 周扣 15 分"这种互相打架的规则。

另外提供两个"防自己骗自己"的工具：

- 表格按**显示宽度**对齐（CJK 字符占两列），否则中文表格在终端里全是错位的；
- 素材缺失/损坏时抛 :class:`FixtureError`，让"素材坏了"和"算法算错了"能被分开定位。
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = LAB_ROOT / "fixtures"
CAPSTONE_DIR = LAB_ROOT / "capstone" / "ai-product-spec"


class FixtureError(ValueError):
    """素材缺失、不可解析或字段不合法。

    单独一个异常类型，是为了让自检能把"素材坏了"和"算法算错了"分开报。
    """


# ---------------------------------------------------------------------------
# 素材加载
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    """读取任意 JSON 文件；缺文件或不是合法 JSON 都抛 FixtureError。"""
    if not path.is_file():
        raise FixtureError(f"缺少文件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureError(f"不是合法 JSON：{path}（{exc}）") from exc


def load_fixture(name: str) -> dict[str, Any]:
    """读取 fixtures/<name>.json 并返回顶层字典。"""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.is_file():
        raise FixtureError(f"缺少素材文件：{path}")
    data = load_json(path)
    if not isinstance(data, dict):
        raise FixtureError(f"素材顶层必须是对象：{path}")
    return data


def read_text(path: Path) -> str:
    """读取文本文件；缺文件直接报错。"""
    if not path.is_file():
        raise FixtureError(f"缺少文件：{path}")
    return path.read_text(encoding="utf-8")


def require_fields(obj: dict[str, Any], fields: list[str], subject: str) -> None:
    """检查字典里的必填字段，缺一个报出全部缺项（不逐个报错，免得只修一个又报下一个）。"""
    missing = [name for name in fields if name not in obj or obj[name] in (None, "")]
    if missing:
        raise FixtureError(f"{subject} 缺少必填字段：{'、'.join(missing)}")


def parse_md_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    """从 Markdown 里取出第一张表格，返回 (表头, 每行的字典)。

    只处理第一张表：风险登记册、指标字典这类文档一个文件一张表，够用。
    """
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        stripped = line.replace("|", "").replace(" ", "")
        if stripped and set(stripped) <= set("-:"):
            continue  # 表格的分隔行
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    if not rows:
        return [], []
    header = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        record: dict[str, str] = {}
        for index, key in enumerate(header):
            record[key] = row[index] if index < len(row) else ""
        records.append(record)
    return header, records


# ---------------------------------------------------------------------------
# 表格渲染（按显示宽度对齐）
# ---------------------------------------------------------------------------


def display_width(text: str) -> int:
    """文本在等宽终端里占的列数。

    CJK 全角字符占两列——用 len() 排版会让中文表格整个错位，所以这里自己算。
    """
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def truncate(text: str, limit: int) -> str:
    """按显示宽度截断，超出部分用省略号收尾。"""
    if display_width(text) <= limit:
        return text
    kept: list[str] = []
    width = 0
    for char in text:
        char_width = display_width(char)
        if width + char_width > limit - 1:
            break
        kept.append(char)
        width += char_width
    return "".join(kept) + "…"


def pad(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补齐到指定列数。"""
    gap = width - display_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def fmt_table(
    headers: list[str],
    rows: list[list[Any]],
    align_right: set[int] | None = None,
    max_widths: list[int] | None = None,
) -> str:
    """把二维数组渲染成等宽表格。

    max_widths 给定时按显示宽度截断该列——表格太宽在终端里要横向滚动，
    而"可归档的结论"必须一眼能看完。
    """
    align_right = align_right or set()
    limits = max_widths or []
    cells = [[str(cell) for cell in row] for row in rows]

    widths: list[int] = []
    for index, header in enumerate(headers):
        width = display_width(str(header))
        for row in cells:
            if index < len(row):
                width = max(width, display_width(row[index]))
        if index < len(limits):
            width = min(width, limits[index])
        widths.append(width)

    def render(row: list[str]) -> str:
        parts: list[str] = []
        for index, cell in enumerate(row):
            if index < len(limits):
                cell = truncate(cell, widths[index])
            parts.append(pad(cell, widths[index], "right" if index in align_right else "left"))
        return "  ".join(parts).rstrip()

    lines = [
        render([str(header) for header in headers]),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(render(row) for row in cells)
    return "\n".join(lines)


def fmt_kv(pairs: list[tuple[str, Any]], key_width: int | None = None) -> str:
    """渲染"键：值"块，键列对齐。"""
    if not pairs:
        return ""
    width = key_width or max(display_width(str(key)) for key, _ in pairs)
    return "\n".join(f"{pad(str(key), width)}  {value}" for key, value in pairs)


def pct(value: float, digits: int = 1) -> str:
    """把 0–1 的比例渲染成百分比字符串。"""
    return f"{value * 100:.{digits}f}%"


def clamp(value: float, low: float, high: float) -> float:
    """把数值夹到区间内。"""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 共享数据模型
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """一次打分的结论：总分、档位判定与分项拆解。

    breakdown 必须留着——只给一个 72 分没法让人复盘，分项才是可改进的东西。
    """

    total: float
    verdict: str
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    """一份检查报告：通过项、问题项与补充说明。"""

    subject: str
    passed: bool
    ok: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ChecklistItem:
    """毕业作品校验清单里的一项。"""

    name: str
    passed: bool
    detail: str


@dataclass
class Checklist:
    """毕业作品校验清单的汇总。"""

    items: list[ChecklistItem] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """全部项都通过才算通过。"""
        return all(item.passed for item in self.items)

    @property
    def summary(self) -> str:
        return "通过" if self.passed else "不通过"
