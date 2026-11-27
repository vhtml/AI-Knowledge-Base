"""vibe_film_lab 自检：验证实验骨架本身没坏。

这个模块解决什么问题：实验代码本身也要有回归保障。
自检不依赖 pytest、不联网、不调模型，验证的是"实验的脚手架、引擎和素材是完整的"，
而不是"教程内容讲对了"——后者由每章 lesson 里的断言负责。

被检查的素材是**毕业作品自己**（``capstone/fourier_C/``）：
分镜表、事实表、成片代码三者必须对得上。

用法：
    cd Demos/examples
    uv run --project .. python -m vibe_film_lab.selftest
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
PROJECT = LAB_ROOT / "capstone" / "fourier_C"

from .labkit import (  # noqa: E402
    Fact,
    Shot,
    fact_audit,
    load_markdown_storyboard,
    parse_film_config,
    parse_film_scenes,
    parse_film_subs,
    parse_markup,
    parse_segments,
    strip_js_comments,
    subtitle_timeline,
    validate_storyboard,
)


def check_lessons_importable() -> None:
    """六章的 lesson 都在，且都导出 run()。"""
    from .lessons import LESSONS

    assert len(LESSONS) == 6, f"应有 6 章实验，实际 {len(LESSONS)}"
    for lesson in LESSONS:
        module = lesson.load()
        assert hasattr(module, "run"), f"{lesson.key} 缺少 run()"
        assert isinstance(lesson.title, str) and lesson.title


def check_truth_storyboard() -> None:
    """成片的分镜表齐全，且自身能过体检。

    这份 Markdown 分镜表是唯一真相：镜头、事实、时间轴、文献都在里面。
    """
    storyboard = load_markdown_storyboard(PROJECT / "storyboard.md")

    assert len(storyboard.shots) == 8, f"成片应有 8 镜，实际 {len(storyboard.shots)}"
    assert len(storyboard.facts) == 5, f"成片应有 5 条事实，实际 {len(storyboard.facts)}"
    assert len(storyboard.timeline) == 8, f"成片应有 8 行时间轴，实际 {len(storyboard.timeline)}"
    assert storyboard.references, "片尾文献不能为空"
    assert storyboard.topic, "分镜表里要有选题"
    assert storyboard.thesis, "分镜表里要有论点"

    for shot in storyboard.shots:
        for field in ("subtitle", "method", "on_screen", "action", "takeaway"):
            assert getattr(shot, field), f"第 {shot.index} 镜缺 {field}"

    problems = validate_storyboard(storyboard.shots)
    assert not problems, f"成片的分镜表没过体检：{problems}"


def check_markup_parsing() -> None:
    """染色标记的边界行为。"""
    segments = parse_segments("把它{金色}拆开，里面全是简单的波")
    colors = [color for _, color in segments if color]
    assert colors == ["gold"], f"标记应该只染它前面那一段，实际 {colors}"
    plain = "".join(text for text, _ in segments)
    assert plain == "把它拆开，里面全是简单的波", f"纯文本拼错了：{plain}"

    # 写错的标记不能被吞掉——写稿的人要看得见自己打错了
    plain, colors = parse_markup("测试{荧光绿}文本")
    assert "{荧光绿}" in plain, "颜色词写错时不该被吞掉"
    assert not colors, "颜色词写错时不该产生颜色"


def check_timeline() -> None:
    """时间轴口径：单调、不重叠、停留被钳制在合理区间。"""
    timeline = subtitle_timeline([
        {"text": "短"},
        {"text": "这是一句很长的字幕，长到需要被截断，不然一行放不下"},
    ])
    assert timeline[0].hold >= 1.8, "太短的句子也要给一个下限"
    assert timeline[1].hold <= 5.0, "太长的句子要被钳制"
    for previous, current in zip(timeline, timeline[1:]):
        assert current.start >= previous.end, "时间轴不能重叠"


def check_storyboard_validator() -> None:
    """分镜体检器该抓的抓得住。"""
    broken = [
        Shot(index=1, subtitle="第一句", method="位置排布", on_screen="",
             action="", takeaway=""),
        Shot(index=2, subtitle="第二句", method="不存在的画法", on_screen="球",
             action="球在动", takeaway="球会动"),
    ]
    problems = validate_storyboard(broken)
    assert len(problems) >= 2, f"坏分镜表应该被抓出多个问题，实际 {problems}"


def check_fact_audit() -> None:
    """事实核对是双向的。"""
    facts = [Fact(key="年份", value="1822", source="出处", illustrative=False)]
    ok = fact_audit([{"text": "事情发生在 1822 年"}], facts)
    assert not ok, f"事实表和字幕一致时不该报错：{ok}"

    missing = fact_audit([{"text": "没有数字"}], facts)
    assert missing, "事实表里的数字没上屏时应该报出来"

    unknown = fact_audit([{"text": "数字是 99"}], facts)
    assert unknown, "字幕里的数字查不到出处时应该报出来"


def check_engine_files() -> None:
    """引擎文件齐全，且 film.js 里没有不确定性来源。"""
    engine = LAB_ROOT / "engine"
    for name in ("render.py", "template.html"):
        assert (engine / name).is_file(), f"缺少引擎文件 engine/{name}"

    for name in ("index.html", "film.js", "storyboard.md", "README.md"):
        assert (PROJECT / name).is_file(), f"缺少毕业作品文件 capstone/fourier_C/{name}"

    # 工程页必须和模板一致——模板改了，所有工程都要跟着改
    assert (PROJECT / "index.html").read_text(encoding="utf-8") == (
        engine / "template.html"
    ).read_text(encoding="utf-8"), "index.html 和 engine/template.html 不一致"

    source = strip_js_comments((PROJECT / "film.js").read_text(encoding="utf-8"))
    for marker in ("Math.random", "Date.now", "performance.now"):
        assert marker not in source, f"film.js 里出现了不确定性来源：{marker}"
    assert "function render(" in source, "film.js 里应该有 render(t)"
    assert "function seek(" in source, "film.js 里应该有 seek(t)"


def check_film_matches_storyboard() -> None:
    """film.js 的字幕表必须和分镜表逐字一致，时间也要对得上。"""
    storyboard = load_markdown_storyboard(PROJECT / "storyboard.md")
    film_subs = parse_film_subs(PROJECT / "film.js")

    assert len(film_subs) == len(storyboard.shots), "字幕条数和分镜表镜数对不上"
    for shot, sub in zip(storyboard.shots, film_subs):
        assert shot.subtitle == sub["text"], f"第 {shot.index} 镜两边不一致"
        row = next(
            (row for row in storyboard.timeline if row["index"] == shot.index), None
        )
        assert row is not None, f"分镜表的时间轴里没有第 {shot.index} 镜"
        assert abs(sub["t"] - row["start"]) < 1e-6, f"第 {shot.index} 镜起始时间不一致"
        assert abs(sub["hold"] - row["hold"]) < 1e-6, f"第 {shot.index} 镜停留时间不一致"

    scenes = parse_film_scenes(PROJECT / "film.js")
    config = parse_film_config(PROJECT / "film.js")
    assert scenes, "场景表不能为空"
    assert abs(scenes[0]["t0"]) < 1e-6, "第一个场景必须从 0 开始"
    for previous, current in zip(scenes, scenes[1:]):
        assert abs(previous["t1"] - current["t0"]) < 1e-6, "场景之间有缝或有重叠"
    assert abs(scenes[-1]["t1"] - float(config["DUR"])) < 1e-6, "场景没铺满全片"


def check_render_pipeline() -> None:
    """端到端：渲几帧小图 → 编码 → ffprobe 校验。

    这一步需要本机有 ego lite 和 ffmpeg；没有就跳过（不算失败）。
    """
    if not shutil.which("ffmpeg"):
        print("    （本机没有 ffmpeg，跳过端到端渲染）")
        return

    from .engine.render import BrowserNotFound, encode, probe, render_frames

    template = (LAB_ROOT / "engine" / "template.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "index.html").write_text(template, encoding="utf-8")
        (workdir / "film.js").write_text(
            (PROJECT / "film.js").read_text(encoding="utf-8"), encoding="utf-8"
        )
        try:
            frames = render_frames(
                workdir / "index.html", workdir / "frames",
                [index / 30 for index in range(6)], 320, 180, jobs=2,
            )
        except BrowserNotFound:
            print("    （本机没有 ego lite，跳过端到端渲染）")
            return

        assert len(frames) == 6, "应该渲出 6 帧"
        out = workdir / "film.mp4"
        encode(workdir / "frames", out, 30)
        info = probe(out)
        assert int(info["width"]) == 320, f"宽度不对：{info['width']}"
        assert int(info["height"]) == 180, f"高度不对：{info['height']}"
        assert info["r_frame_rate"] == "30/1", f"帧率不对：{info['r_frame_rate']}"


def check_validator_exit_code() -> None:
    """渲染脚本的命令行入口能跑通（--det）。"""
    from .engine.render import BrowserNotFound, find_browser, find_ffmpeg

    try:
        find_browser()
        find_ffmpeg()
    except (BrowserNotFound, RuntimeError):
        print("    （本机缺 ego lite 或 ffmpeg，跳过）")
        return

    template = (LAB_ROOT / "engine" / "template.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "index.html").write_text(template, encoding="utf-8")
        (workdir / "film.js").write_text(
            (PROJECT / "film.js").read_text(encoding="utf-8"), encoding="utf-8"
        )
        proc = subprocess.run(
            [sys.executable, "-m", "vibe_film_lab.engine.render", str(workdir), "--det"],
            capture_output=True, text=True, cwd=str(LAB_ROOT.parent),
        )
        assert proc.returncode == 0, f"--det 应该通过：\n{proc.stdout}\n{proc.stderr}"


def run() -> int:
    """跑完全部自检项。"""
    checks = [
        ("lesson 可导入", check_lessons_importable),
        ("成片分镜表", check_truth_storyboard),
        ("染色标记解析", check_markup_parsing),
        ("时间轴口径", check_timeline),
        ("分镜体检器", check_storyboard_validator),
        ("事实双向核对", check_fact_audit),
        ("引擎文件", check_engine_files),
        ("film.js 与分镜表一致", check_film_matches_storyboard),
        ("端到端渲染", check_render_pipeline),
        ("渲染脚本入口", check_validator_exit_code),
    ]

    failures: list[str] = []
    for name, fn in checks:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failures.append(f"{name}：{exc}")
            print(f"  ✗ {name} —— {exc}")

    print()
    if failures:
        print(f"自检未通过：{len(failures)}/{len(checks)} 项失败")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"自检通过：{len(checks)}/{len(checks)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(run())
