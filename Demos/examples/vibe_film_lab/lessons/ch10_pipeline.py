"""第 10 章实验：工程管线。

这个模块解决什么问题：把"怎么把代码变成成片"这条流水线量出来，
让人知道**一条片子要花多少时间和算力**。

关键结论有三个：
- 帧数是算出来的：``时长 × 帧率``，不是"渲到满意为止"；
- 并行是线性收益：``--jobs 4`` 大约快 3 倍（省的是 Chrome 启动开销）；
- 顺序不能反：先联络表、再静帧、再转场、最后才渲全片。
  全片渲完才发现第 40 秒有一根线穿过了字幕区，是最贵的一种返工。

对应笔记：AI/Knowledge-Film/10-工程管线.md
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from ..labkit import ensure, fmt_table, parse_film_config

FILM_JS = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C" / "film.js"
SAMPLE_FRAMES = 6
"""实测渲染速度时渲多少帧。6 帧足够看出量级，又不至于让实验跑太久。"""


def prepare_workdir() -> Path:
    """搭一个临时的工程目录（index.html + film.js）。"""
    workdir = Path(tempfile.mkdtemp(prefix="vibe_film_lab_"))
    template = (FILM_JS.parents[2] / "engine" / "template.html").read_text(encoding="utf-8")
    (workdir / "index.html").write_text(template, encoding="utf-8")
    (workdir / "film.js").write_text(FILM_JS.read_text(encoding="utf-8"), encoding="utf-8")
    return workdir


def run() -> None:
    print("=" * 78)
    print("第 10 章 · 工程管线：先渲片段，再渲全片")
    print("=" * 78)

    config = parse_film_config(FILM_JS)
    duration = float(config.get("DUR", 0))
    fps = int(config.get("FPS", 30))
    width = int(config.get("W", 1920))
    height = int(config.get("H", 1080))

    frame_count = round(duration * fps)
    print()
    print("【1】帧数是算出来的")
    print(fmt_table(
        ["项目", "值"],
        [
            ["片长", f"{duration:.1f} 秒"],
            ["帧率", f"{fps} fps"],
            ["分辨率", f"{width}×{height}"],
            ["总帧数", f"{frame_count} 帧"],
            ["1080p 单帧 PNG", "约 1–3 MB"],
            ["帧目录体积", f"约 {frame_count * 2 / 1024:.0f} GB"],
        ],
        align_right={1},
    ))
    ensure(frame_count > 0, "帧数应该是正数")
    ensure(width % 2 == 0 and height % 2 == 0, "分辨率必须是偶数，yuv420p 才认")

    # ---- 实测渲染速度 ----
    print()
    print("【2】实测渲染速度")
    if not shutil.which("ffmpeg"):
        print("  ⊘ 跳过：本机没有 ffmpeg")
        return

    from ..engine.render import ChromeNotFound, render_frames

    workdir = prepare_workdir()
    html = workdir / "index.html"
    times = [index / fps for index in range(SAMPLE_FRAMES)]

    results: list[tuple[int, float, float]] = []
    digests: dict[int, list[str]] = {}
    for jobs in (1, 4):
        try:
            start = time.perf_counter()
            with tempfile.TemporaryDirectory() as tmp:
                frames = render_frames(html, Path(tmp), times, width, height, jobs=jobs)
                digests[jobs] = [
                    hashlib.sha256(frame.read_bytes()).hexdigest() for frame in frames
                ]
            elapsed = time.perf_counter() - start
        except ChromeNotFound:
            print("  ⊘ 跳过：本机没有 Chrome/Chromium/Edge")
            return
        results.append((jobs, elapsed, SAMPLE_FRAMES / elapsed))

    print(fmt_table(
        ["并行路数", "耗时", "帧/秒", "换算全片"],
        [
            [
                f"{jobs} 路",
                f"{elapsed:.1f}s",
                f"{rate:.2f}",
                f"{frame_count / rate / 60:.1f} 分钟",
            ]
            for jobs, elapsed, rate in results
        ],
        align_right={1, 2, 3},
    ))

    single_rate = results[0][2]
    parallel_rate = results[1][2]
    speedup = parallel_rate / single_rate
    print(f"  并行加速比 {speedup:.1f}x")
    print("  （加速比是**实测值**，不是保证值：瓶颈在浏览器启动时并行才划算，")
    print("   机器核数紧张时并行反而更慢。要在自己的机器上量一遍再定 --jobs。）")

    # ---- 并行不能改变结果 ----
    ensure(
        digests[1] == digests[4],
        "并行渲染和单路渲染的结果不一致——并行只是调度问题，不该改变像素",
    )
    print(f"  ✓ 并行与单路渲染出完全相同的 {len(digests[1])} 帧（逐字节比对）")
    ensure(single_rate > 0, "渲染速度应该是正数")

    # ---- 工作流的顺序 ----
    print()
    print("【3】检查动作的顺序不能反")
    print(fmt_table(
        ["顺序", "动作", "看什么", "成本"],
        [
            ["1", "--sheet 若干时刻", "风格是否一致、字幕区是否被占", "秒级"],
            ["2", "--stills 关键帧", "字号、描边、小字位置", "秒级"],
            ["3", "--motion 一段时间", "转场怎么接的", "秒级"],
            ["4", "--clip 起 止", "这一段对不对", "分钟级"],
            ["5", "--det", "确定性（必须先过）", "秒级"],
            ["6", "全片 --jobs 4", "成片", "十分钟级"],
        ],
    ))

    print()
    print("要点：把'检查'放到'渲染'之前，是这条流水线里最划算的习惯。")
    print("      全片渲完才发现问题，返工成本是片段的几十倍。")


if __name__ == "__main__":
    run()
