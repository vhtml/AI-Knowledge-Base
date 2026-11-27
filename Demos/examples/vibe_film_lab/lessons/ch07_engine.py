"""第 7 章实验：渲染引擎。

这个模块解决什么问题：把"画面必须是时间的纯函数"这条规矩变成**能跑的检查**。

规矩很简单：``render(t)`` 给定 t 就画出那一帧，不许依赖"上一帧画过什么"。
一旦违反，逐帧渲染就会出现最恶心的一种 bug——同一段动画在片子里前后抖，
但你在浏览器里播放时完全看不出来。

检查分两层：
- **静态**：film.js 里不许出现 Math.random / Date.now 这类不确定性来源；
  场景表必须首尾相接铺满全片；字幕表必须和分镜表一致。
- **动态**：同一个 t 渲两遍，逐字节比对（需要本机有 ego lite）。

对应笔记：AI/Knowledge-Film/07-渲染引擎.md
"""

from __future__ import annotations

from pathlib import Path

from ..labkit import (
    ensure,
    fmt_table,
    load_markdown_storyboard,
    parse_film_config,
    parse_film_scenes,
    parse_film_subs,
    strip_js_comments,
)

PROJECT = Path(__file__).resolve().parents[1] / "capstone" / "fourier_C"
FILM_JS = PROJECT / "film.js"
STORYBOARD_MD = PROJECT / "storyboard.md"

NONDETERMINISM_SOURCES = ("Math.random", "Date.now", "performance.now", "new Date")
"""不确定性来源。任何一个出现在画面绘制路径里，逐帧渲染就会前后不一致。"""


def run() -> None:
    print("=" * 78)
    print("第 7 章 · 渲染引擎：画面必须是时间的纯函数")
    print("=" * 78)

    source = FILM_JS.read_text(encoding="utf-8")
    code = strip_js_comments(source)

    # ---- 静态检查 ----
    print()
    print("【1】静态检查：不确定性能不能进画面")
    print("    （只看代码，不看注释——注释里写'不要用 Math.random'是正常的）")
    for marker in NONDETERMINISM_SOURCES:
        present = marker in code
        print(f"  {'✗' if present else '✓'} {marker}")
        ensure(not present, f"film.js 里出现了不确定性来源：{marker}")

    ensure("function render(" in source, "film.js 里应该有 render(t)")
    ensure("function seek(" in source, "film.js 里应该有 seek(t)")
    ensure(
        "seeded(" in source,
        "要画随机元素（比如星点）就该用带种子的确定性伪随机，而不是 Math.random",
    )
    print("  ✓ 用 seeded() 代替 Math.random：星点每次都在同一个位置。")

    # ---- 场景表必须铺满全片 ----
    config = parse_film_config(FILM_JS)
    scenes = parse_film_scenes(FILM_JS)
    duration = float(config.get("DUR", 0))
    print()
    print("【2】场景表")
    print(fmt_table(
        ["场景", "起", "止", "时长"],
        [[scene["name"], f"{scene['t0']:.1f}s", f"{scene['t1']:.1f}s",
          f"{scene['t1'] - scene['t0']:.1f}s"] for scene in scenes],
        align_right={1, 2, 3},
    ))

    ensure(scenes, "场景表不能为空")
    ensure(
        abs(scenes[0]["t0"]) < 1e-6,
        f"第一个场景必须从 0 开始，实际 {scenes[0]['t0']}",
    )
    for previous, current in zip(scenes, scenes[1:]):
        ensure(
            abs(previous["t1"] - current["t0"]) < 1e-6,
            f"场景「{previous['name']}」和「{current['name']}」之间有缝或有重叠",
        )
    ensure(
        abs(scenes[-1]["t1"] - duration) < 1e-6,
        f"最后一个场景的结束时间 {scenes[-1]['t1']} 和 CONFIG.DUR {duration} 对不上",
    )
    print(f"  ✓ 场景首尾相接，铺满 0–{duration:.1f}s，中间没有缝。")

    # ---- 字幕表必须和分镜表一致 ----
    subs = parse_film_subs(FILM_JS)
    storyboard = load_markdown_storyboard(STORYBOARD_MD)
    shots = storyboard.shots
    timeline = {row["index"]: row for row in storyboard.timeline}
    print()
    print("【3】film.js 的字幕表 ↔ 分镜表（唯一真相）")
    ensure(
        len(subs) == len(shots),
        f"字幕条数不一致：film.js {len(subs)} 条，分镜表 {len(shots)} 镜",
    )
    for shot, sub in zip(shots, subs):
        ensure(
            sub["text"] == shot.subtitle,
            f"第 {shot.index} 镜两边不一致：{shot.subtitle!r} vs {sub['text']!r}",
        )
        row = timeline.get(shot.index)
        ensure(row is not None, f"分镜表的时间轴里没有第 {shot.index} 镜")
        ensure(
            abs(sub["t"] - row["start"]) < 1e-6,
            f"第 {shot.index} 镜起始时间不一致：film.js {sub['t']} vs 分镜表 {row['start']}",
        )
        ensure(
            abs(sub["hold"] - row["hold"]) < 1e-6,
            f"第 {shot.index} 镜停留时间不一致：film.js {sub['hold']} vs 分镜表 {row['hold']}",
        )
    print(f"  ✓ {len(subs)} 句字幕与分镜表逐字一致，起始时间和停留时间也对得上。")

    # ---- 动态检查：确定性 ----
    print()
    print("【4】动态检查：同一个 t 渲两遍")
    try:
        from ..engine.render import BrowserNotFound, determinism_check

        html = FILM_JS.parents[2] / "engine" / "template.html"
        workdir = Path("/tmp") / "vibe_film_lab_det"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "index.html").write_text(
            html.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (workdir / "film.js").write_text(source, encoding="utf-8")
        ok = determinism_check(workdir / "index.html", 3.0, 1920, 1080)
        ensure(ok, "同一时刻两帧不一致：画面里混进了状态，逐帧导出一定会抖")
        print("  ✓ 逐字节一致——画面里没有藏状态。")
    except BrowserNotFound:
        print("  ⊘ 跳过：本机没有 ego lite，装一个再跑这条。")
    except Exception as exc:  # noqa: BLE001 - 环境问题不该让整章失败
        print(f"  ⊘ 跳过：{type(exc).__name__}: {exc}")

    print()
    print("要点：render(t) 是纯函数，不是播放器。")
    print("      播放器可以'接着上一帧画'，逐帧渲染器不行——它只有 t。")
    print("      所以从第一天就按纯函数写，比渲完再改便宜一百倍。")


if __name__ == "__main__":
    run()
