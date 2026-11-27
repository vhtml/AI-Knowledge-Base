"""Vibe 知识动画实验室的公共工具层。

这个模块解决什么问题：把"一条 Vibe 知识短片"拆成**可计算**的几件事——
字幕时间轴、分镜表、事实表、终检门槛——并且让每章实验共用同一套口径。
各章各算各的，读者就会得到互相矛盾的结论（比如第 4 章说字幕停 3 秒、
第 8 章算出来平均 4.6 秒却没人觉得有问题）。

所以约定：**所有数字只在这里算一次**。各章实验只写当天的教学重点，
需要算账就调用本模块，绝不自己另起一套公式。

四条贯穿全课的口径：

1. **字幕就是文案**。没有旁白，字幕是唯一的信息通道：它既要是完整的话，
   又要能在 3–5 秒内读完完。
2. **画面必须解释**。每一句字幕都要在屏上找到对应的东西；找不到就改句子，
   不是加一张配图。
3. **规格统一**。横屏 1920×1080、30fps；成片时长由字幕和分镜算出来，
   不靠拍脑袋。
4. **确定性优先**。画面必须是 ``render(t)`` 这种纯函数：同一个 t 永远画出
   同一帧，否则没法逐帧渲染、没法复核、没法局部重渲。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "COLOR_TOKENS",
    "Fact",
    "Shot",
    "SubCue",
    "ScriptStats",
    "display_width",
    "ensure",
    "ease_in_out",
    "ease_out_cubic",
    "fact_audit",
    "fmt_table",
    "load_json",
    "parse_markup",
    "parse_segments",
    "parse_film_config",
    "parse_film_scenes",
    "parse_film_subs",
    "print_table",
    "script_stats",
    "strip_js_comments",
    "subtitle_timeline",
    "three_questions",
    "validate_storyboard",
    "band_ratio",
    "repo_root",
]

# ---------------------------------------------------------------- 颜色标记

COLOR_TOKENS: dict[str, str] = {
    "金色": "gold",
    "玫红": "rose",
    "红色": "red",
    "青色": "cyan",
    "蓝色": "blue",
    "灰色": "grey",
}
"""字幕里允许的染色标记。键是中文词，值是颜色语义名。

一条字幕最多染一处——两处以上观众读不完，这是终检门槛里的硬指标。
"""

_MARKUP_RE = re.compile(r"\{([^}]*)\}|\[([^\]]*)\]")

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，。、；：！？!?;])")
"""按句读切分。用来确定"一个染色标记到底染哪一段"。"""


# ---------------------------------------------------------------- 断言与输出


def ensure(condition: bool, message: str) -> None:
    """断言辅助：条件不成立就抛 :class:`AssertionError`。

    报错信息用中文写清"期望什么、实际什么"，比裸 ``assert`` 好读。
    """
    if not condition:
        raise AssertionError(message)


def display_width(text: str) -> int:
    """算字符串在终端里的显示宽度。

    中日韩字符占**两列**，英文占一列。按字符数对齐的话，表格里只要有中文
    就会歪——这不是美观问题，歪了的表格会让人看错数字和列的对应关系。
    """
    width = 0
    for char in text:
        code = ord(char)
        if (
            0x1100 <= code <= 0x115F
            or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE6F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
            or 0x20000 <= code <= 0x3FFFD
        ):
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格。"""
    return text + " " * max(0, width - display_width(text))


def fmt_table(
    headers: list[str],
    rows: list[list[Any]],
    align_right: set[int] | None = None,
) -> str:
    """把表头和行渲染成对齐的等宽表格文本。"""
    align_right = align_right or set()
    headers = [str(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    widths = [display_width(h) for h in headers]
    for row in body:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], display_width(cell))

    def line(cells: list[str]) -> str:
        parts = []
        for index, cell in enumerate(cells):
            pad = _pad(cell, widths[index]) if index < len(widths) else cell
            parts.append(pad.rjust(len(pad)) if index in align_right else pad)
        return "  " + " | ".join(parts)

    out = [line(headers), "  " + "-+-".join("-" * w for w in widths)]
    out.extend(line(row) for row in body)
    return "\n".join(out)


def print_table(
    headers: list[str],
    rows: list[list[Any]],
    align_right: set[int] | None = None,
) -> None:
    """打印表格。"""
    print(fmt_table(headers, rows, align_right))


def repo_root() -> Path:
    """仓库根目录。"""
    return Path(__file__).resolve().parents[4]


def load_json(path: Path) -> Any:
    """读 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 缓动


def ease_out_cubic(x: float) -> float:
    """三次缓出：起步快、收尾稳。字幕入场默认用它。"""
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    """两端缓、中间快：场景之间的推拉用它。"""
    x = max(0.0, min(1.0, x))
    return 2 * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 2 / 2


# ---------------------------------------------------------------- 字幕


@dataclass(frozen=True)
class SubCue:
    """一条字幕在时间轴上的位置。"""

    index: int
    start: float
    end: float
    text: str
    """去掉染色标记后的纯文本。"""

    colored: tuple[str, ...] = ()
    """这一条用到的颜色语义名，按出现顺序。"""

    @property
    def hold(self) -> float:
        """停留时长（秒）。"""
        return self.end - self.start

    @property
    def chars(self) -> int:
        """纯文本字数。"""
        return len(self.text)


def parse_segments(text: str) -> list[tuple[str, str | None]]:
    """把带标记的字幕拆成 ``(片段, 颜色)`` 序列。

    约定：**标记染它前面那一段**——从上一个句读起，到标记为止的那段。
    所以 ``"把它拆开{金色}，里面全是简单的波"`` 里，金色只落在
    "把它拆开"上，不会把整句染掉。

    颜色词不在 :data:`COLOR_TOKENS` 里就当普通字符——写错标记时不能
    吞掉，否则写稿的人永远不知道自己打错了。
    """
    segments: list[tuple[str, str | None]] = []
    pending: list[str] = []

    def flush() -> None:
        """把暂存的一段落成片段（颜色先留空）。"""
        if pending:
            segments.append(("".join(pending), None))
            pending.clear()

    position = 0
    for match in _MARKUP_RE.finditer(text):
        pending.append(text[position : match.start()])
        token = match.group(1) or match.group(2) or ""
        if token in COLOR_TOKENS:
            # 标记往前找一段：先按句读切，再取最后一段
            chunk = "".join(pending)
            parts = [p for p in _CLAUSE_SPLIT_RE.split(chunk) if p]
            if parts:
                *head, last = parts
                pending.clear()
                pending.extend(head)
                flush()
                segments.append((last, COLOR_TOKENS[token]))
            else:
                flush()
        else:
            pending.append(match.group(0))
        position = match.end()
    pending.append(text[position:])
    flush()

    return [(seg, color) for seg, color in segments if seg]


def parse_markup(text: str) -> tuple[str, tuple[str, ...]]:
    """拆出纯文本和用到的颜色。

    支持 ``{金色}`` 和 ``[玫红]`` 两种标记。颜色词不在
    :data:`COLOR_TOKENS` 里就当普通字符——写错标记时不能吞掉，
    否则写稿的人永远不知道自己打错了。
    """
    segments = parse_segments(text)
    plain = "".join(seg for seg, _ in segments)
    colors = tuple(color for _, color in segments if color)
    return plain, colors


def subtitle_timeline(
    cues: list[dict[str, Any]],
    *,
    gap: float = 0.15,
    lead_in: float = 0.0,
    respect_explicit: bool = True,
) -> list[SubCue]:
    """把字幕稿排成时间轴。

    每条字幕的停留时间由它的**字数**决定：中文按每字 0.22 秒，
    再加一个下限（太短的句子读起来赶）和上限（太长的句子屏幕塞不下）。
    句与句之间留 ``gap`` 秒呼吸——无人声的片子全靠这个空隙制造节奏。

    ``respect_explicit`` 为真时，字幕稿里显式写了 ``t`` 的条目**用它自己的
    起始时间**（手工排过的片子必须这样，否则算出来的时间轴和成片对不上）。
    没写 ``t`` 的条目接在上一条之后——这是"自动排版"模式，
    写完稿子不用管时间轴，让它自己排。

    返回的 :class:`SubCue` 按开始时间单调不减。
    """
    timeline: list[SubCue] = []
    cursor = lead_in
    for index, cue in enumerate(cues):
        text = cue.get("text", "")
        plain, colors = parse_markup(text)
        hold = cue.get("hold")
        if hold is None:
            hold = max(1.8, min(5.0, len(plain) * 0.22))
        explicit = cue.get("t")
        start = float(explicit) if (respect_explicit and explicit is not None) else cursor
        timeline.append(
            SubCue(
                index=index,
                start=round(start, 3),
                end=round(start + hold, 3),
                text=plain,
                colored=colors,
            )
        )
        cursor = start + hold + gap
    return timeline


@dataclass(frozen=True)
class ScriptStats:
    """字幕稿的统计量。"""

    cues: tuple[SubCue, ...]
    total_chars: int
    avg_hold: float
    min_hold: float
    max_hold: float
    avg_chars_per_sec: float
    longest_line: str
    longest_line_chars: int
    multi_color_lines: list[int]
    """一句话染了两种及以上颜色的行号（从 1 开始）。"""

    @property
    def duration(self) -> float:
        """字幕总时长（含句间呼吸）。"""
        return self.cues[-1].end if self.cues else 0.0

    @property
    def ok_hold_ratio(self) -> float:
        """停留时长落在 3–5 秒区间的比例。"""
        if not self.cues:
            return 0.0
        hits = sum(1 for cue in self.cues if 3.0 <= cue.hold <= 5.0)
        return hits / len(self.cues)


def script_stats(cues: list[dict[str, Any]], *, gap: float = 0.15) -> ScriptStats:
    """统计字幕稿：停留时长、语速、最长句、染色超标。"""
    timeline = subtitle_timeline(cues, gap=gap)
    if not timeline:
        return ScriptStats((), 0, 0.0, 0.0, 0.0, 0.0, "", 0, [])

    holds = [cue.hold for cue in timeline]
    chars = [cue.chars for cue in timeline]
    longest = max(timeline, key=lambda cue: cue.chars)
    multi = [cue.index + 1 for cue in timeline if len(cue.colored) >= 2]
    total_chars = sum(chars)
    total_time = sum(holds)
    return ScriptStats(
        cues=tuple(timeline),
        total_chars=total_chars,
        avg_hold=round(sum(holds) / len(holds), 3),
        min_hold=round(min(holds), 3),
        max_hold=round(max(holds), 3),
        avg_chars_per_sec=round(total_chars / total_time, 2) if total_time else 0.0,
        longest_line=longest.text,
        longest_line_chars=longest.chars,
        multi_color_lines=multi,
    )


# ---------------------------------------------------------------- 分镜表


@dataclass
class Shot:
    """分镜表里的一镜。

    字段和实战里的分镜表一一对应：一句字幕一行，
    "屏上是什么 / 它在干什么 / 观众明白了什么"是必填的三问。
    """

    index: int
    subtitle: str
    method: str
    """画法。六种之一，或者"不画"。"""

    on_screen: str
    """屏上是什么。"""

    action: str
    """它在干什么。"""

    takeaway: str
    """观众明白了什么。"""

    label: str = ""
    """屏上的小字。一镜最多一条。"""

    sfx: str = ""
    """声音事件名。"""

    draw: bool = True
    """False 表示这一镜"不画"——提问、命名、结论不硬配图。"""

    def three_answers(self) -> dict[str, bool]:
        """三问检验：这是什么 / 它在干什么 / 观众明白了什么。"""
        return {
            "on_screen": bool(self.on_screen.strip()),
            "action": bool(self.action.strip()),
            "takeaway": bool(self.takeaway.strip()),
        }


def load_storyboard(path: Path) -> list[Shot]:
    """从 JSON 读分镜表。

    接受两种结构：直接是一个数组，或者 ``{"shots": [...]}``。
    后者多一个位置放说明文字，实际工程里更好维护。
    """
    raw = load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("shots", [])
    shots: list[Shot] = []
    for index, item in enumerate(raw, start=1):
        shots.append(
            Shot(
                index=index,
                subtitle=item.get("subtitle", ""),
                method=item.get("method", ""),
                on_screen=item.get("on_screen", ""),
                action=item.get("action", ""),
                takeaway=item.get("takeaway", ""),
                label=item.get("label", ""),
                sfx=item.get("sfx", ""),
                draw=item.get("draw", True),
            )
        )
    return shots


def three_questions(shot: Shot) -> list[str]:
    """返回这一镜过不了三问的原因；全过则返回空列表。"""
    answers = shot.three_answers()
    labels = {
        "on_screen": "屏上是什么",
        "action": "它在干什么",
        "takeaway": "观众明白了什么",
    }
    return [labels[key] for key, ok in answers.items() if not ok]


def band_ratio(shots: list[Shot]) -> float:
    """"不画"的镜头占比。

    参考片的经验值是**约五分之一**：一个都没有是把结论也画了，
    占了一半是没找到形象。这是个经验判据，不是硬门槛。
    """
    if not shots:
        return 0.0
    return sum(1 for shot in shots if not shot.draw) / len(shots)


def validate_storyboard(shots: list[Shot]) -> list[str]:
    """体检分镜表，返回问题清单（空列表表示通过）。

    检查项对应实战里"分镜表过关"的那几条：每句一句字幕、三问齐、
    不画的比例、一个场景不要连续多镜只改一处（这里简化为不允许连续三镜
    同 method）、小字一镜最多一条。
    """
    problems: list[str] = []

    for shot in shots:
        missing = three_questions(shot)
        if missing:
            problems.append(f"第 {shot.index} 镜三问不齐：缺 {'、'.join(missing)}")
        if shot.method not in DRAW_METHODS and shot.method != "不画":
            problems.append(
                f"第 {shot.index} 镜的画法「{shot.method}」不在六种之内"
            )
        if shot.label and shot.label.count("；") >= 1:
            problems.append(f"第 {shot.index} 镜的小字超过一条")
        if not shot.subtitle.strip():
            problems.append(f"第 {shot.index} 镜没有字幕")

    ratio = band_ratio(shots)
    if shots and ratio == 0:
        problems.append("没有一镜是「不画」：提问、命名、结论不该硬配图")
    if ratio > 0.5:
        problems.append(f"「不画」占比 {ratio:.0%}：多半是没找到形象，不是刻意留白")

    for index in range(len(shots) - 2):
        window = shots[index : index + 3]
        if len({shot.method for shot in window}) == 1 and window[0].method != "不画":
            problems.append(
                f"第 {window[0].index}–{window[-1].index} 镜连续三镜同一画法：画面会像一张静态图"
            )

    return problems


DRAW_METHODS: tuple[str, ...] = (
    "位置排布",
    "时间推进",
    "比例分割",
    "对照比较",
    "代入体验",
    "道具实验",
)
"""六种画法。名字不重要，重要的是"这个概念用哪种空间关系讲得清"。"""


def strip_js_comments(source: str) -> str:
    """去掉 JS 注释，只留代码。

    静态检查（比如"不许出现 Math.random"）必须只看代码——注释里
    写着"不要用 Math.random"是正常的说明文字，不该被当成违规。

    这是个简写实现：不处理字符串里的 ``//``（本实验室的 film.js 里
    没有 URL 字面量）。要更严格可以上真正的 JS 解析器，但那样就得
    多一个依赖，不划算。
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


# ---------------------------------------------------------------- film.js 解析

_SUBS_BLOCK_RE = re.compile(r"const SUBS\s*=\s*\[(.*?)\n\];", re.S)
_CUE_RE = re.compile(
    r"\{\s*t:\s*([\d.]+)\s*,\s*text:\s*'([^']*)'\s*,\s*hold:\s*([\d.]+)\s*\}"
)
_SCENE_RE = re.compile(
    r"\{\s*t0:\s*([\d.]+)\s*,\s*t1:\s*([\d.]+)\s*,\s*name:\s*'([^']*)'\s*,\s*draw:\s*(\w+)\s*\}"
)
_CONFIG_RE = re.compile(r"const CONFIG = \{(.*?)\n\};", re.S)


def parse_film_subs(path: Path) -> list[dict[str, Any]]:
    """从 film.js 里读出字幕表。

    用正则读而不是跑 node：实验室只依赖标准库 + ego lite + ffmpeg，
    多一个运行时依赖就多一处装不上的风险。
    """
    source = path.read_text(encoding="utf-8")
    match = _SUBS_BLOCK_RE.search(source)
    if not match:
        return []
    return [
        {"t": float(t), "text": text, "hold": float(hold)}
        for t, text, hold in _CUE_RE.findall(match.group(1))
    ]


def parse_film_scenes(path: Path) -> list[dict[str, Any]]:
    """从 film.js 里读出场景表。"""
    source = path.read_text(encoding="utf-8")
    return [
        {"t0": float(t0), "t1": float(t1), "name": name, "draw": draw == "drawNoise"}
        for t0, t1, name, draw in _SCENE_RE.findall(source)
    ]


def parse_film_config(path: Path) -> dict[str, Any]:
    """从 film.js 里读出 CONFIG。"""
    source = path.read_text(encoding="utf-8")
    match = _CONFIG_RE.search(source)
    config: dict[str, Any] = {}
    if not match:
        return config
    for key, value in re.findall(r"(\w+)\s*:\s*('[^']*'|[\d.]+)", match.group(1)):
        config[key] = value.strip("'")
    return config


# ---------------------------------------------------------------- 事实表


@dataclass(frozen=True)
class Fact:
    """事实表里的一条。

    ``value`` 是屏上要出现的那个数（或者人名、年份），
    ``source`` 是出处，``illustrative`` 标记这是示意而不是真数据。
    """

    key: str
    value: str
    source: str
    illustrative: bool = False
    on_screen: bool = True
    """这条事实会不会上屏。不上屏的事实不要求出现在字幕里——
    但它必须留着：片尾要印文献，讲解时要能追溯。"""


def fact_audit(cues: list[dict[str, Any]], facts: list[Fact]) -> list[str]:
    """核对事实表和字幕稿：双向查。

    - 正向：字幕里出现的每个数字，都能在事实表里找到出处；
    - 反向：事实表里的数字，要么出现在字幕里，要么明确是"不上屏"的背景。

    实战里最常见的翻车是**反向漏**：事实表写了 5 个数据，字幕只用 2 个，
    于是片子里"少了一个关键数据"却没人发现。
    """
    problems: list[str] = []
    numbers_in_script: set[str] = set()
    for cue in cues:
        plain, _ = parse_markup(cue.get("text", ""))
        for token in re.findall(r"\d+(?:\.\d+)?%?", plain):
            numbers_in_script.add(token)

    for fact in facts:
        if (
            fact.value not in numbers_in_script
            and not fact.illustrative
            and fact.on_screen
        ):
            problems.append(f"事实「{fact.key}」的数值 {fact.value} 没有出现在字幕里")

    known = {fact.value for fact in facts}
    for token in sorted(numbers_in_script):
        if token not in known:
            problems.append(f"字幕里的 {token} 在事实表里查不到出处")

    return problems


# ---------------------------------------------------------------- Markdown 分镜表


@dataclass(frozen=True)
class Storyboard:
    """一份 Markdown 分镜表读出来的全部内容。

    Markdown 分镜表是**唯一真相**：镜头、事实、时间轴、文献都写在一张表里，
    渲染代码必须和它逐字一致——分镜表改了，``film.js`` 必须跟着改，
    改完不跑终检，门槛会把这件事报出来。
    """

    shots: list[Shot]
    facts: list[Fact]
    timeline: list[dict[str, Any]]
    references: list[str]
    topic: str
    thesis: str


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """把 Markdown 里的表格拆成 ``(表头, 数据行)`` 的列表。

    只认表格里第一行是表头、第二行是分隔线的那种写法；
    代码块里的 ``|`` 不算表格。
    """
    tables: list[tuple[list[str], list[list[str]]]] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    in_code = False

    def flush() -> None:
        nonlocal header, rows
        if header is not None and rows:
            tables.append((header, rows))
        header = None
        rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped.startswith("|"):
            flush()
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(cell and set(cell) <= set("-: ") for cell in cells):
            continue  # 表头下面的分隔线
        if header is None:
            header = cells
        else:
            rows.append(cells)
    flush()
    return tables


def _parse_seconds(text: str) -> float:
    """从「0.60s」这样的写法里读出秒数。"""
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def load_markdown_storyboard(path: Path) -> Storyboard:
    """从 Markdown 分镜表读出镜头、事实、时间轴和文献。

    分镜表里有四张内容：事实表、分镜表、时间轴、片尾文献。
    表格的列名就是字段名，缺列就当空——宁可少信息，也不要猜。
    """
    text = path.read_text(encoding="utf-8")

    shots: list[Shot] = []
    facts: list[Fact] = []
    timeline: list[dict[str, Any]] = []

    for header, rows in _markdown_tables(text):
        if header[:3] == ["镜", "字幕", "画法"]:
            for cells in rows:
                shots.append(Shot(
                    index=int(cells[0]),
                    subtitle=cells[1],
                    method=cells[2],
                    on_screen=cells[3],
                    action=cells[4],
                    takeaway=cells[6],
                    label=cells[5],
                    sfx=cells[7],
                    draw=cells[2] != "不画",
                ))
        elif header[:2] == ["事实", "上屏的值"]:
            for cells in rows:
                facts.append(Fact(
                    key=cells[0],
                    value=cells[1],
                    source=cells[3],
                    illustrative=cells[2] == "示意",
                    on_screen=cells[1] != "不上屏",
                ))
        elif header == ["镜", "起", "止", "停留"]:
            for cells in rows:
                if not cells[0].isdigit():
                    continue  # 「—」那一行是无字幕纯画面的留白，不是字幕
                timeline.append({
                    "index": int(cells[0]),
                    "start": _parse_seconds(cells[1]),
                    "end": _parse_seconds(cells[2]),
                    "hold": _parse_seconds(cells[3]),
                })

    references: list[str] = []
    in_references = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_references = "文献" in line
            continue
        if in_references and line.strip().startswith("- "):
            references.append(line.strip()[2:].strip())

    topic = ""
    match = re.search(r"选题：([^｜|]+)", text)
    if match:
        topic = match.group(1).strip()

    thesis = ""
    match = re.search(r"## 论点\s*\n+(.+)", text)
    if match:
        thesis = match.group(1).strip()

    return Storyboard(
        shots=shots,
        facts=facts,
        timeline=timeline,
        references=references,
        topic=topic,
        thesis=thesis,
    )
