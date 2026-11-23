"""第 9 章实验：声音。

这个模块解决什么问题：回答"没有人声，节奏靠什么撑"。

答案是两样：画面里的**变化**和声音里的**事件**。两者要对齐——
屏上东西裂开的那一帧，正好响一声。

夜曲（C）这条路更硬：声音完全由代码合成，每个事件由画面调用登记，
不依赖任何采样库，所以永远和画面对得上，也不会有版权问题。

这一章做三件事：生成事件表 → 检查"提问那镜不出声" → 真合成一条音轨出来。

对应笔记：AI/Knowledge-Film/09-声音.md
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..labkit import ensure, fixtures_dir, fmt_table, load_json, load_storyboard
from ..engine.audio import SFX_TABLE, event_table, synthesize

WORK = Path("/tmp") / "vibe_film_lab_audio"


def run() -> None:
    print("=" * 78)
    print("第 9 章 · 声音：没有人声，节奏靠声音撑")
    print("=" * 78)

    shots = load_storyboard(fixtures_dir() / "storyboard.json")
    script = load_json(fixtures_dir() / "script.json")
    events = event_table(shots, script["cues"])

    print()
    print("【1】事件音表")
    print(fmt_table(
        ["时刻", "事件", "对应镜", "频率", "时长", "为什么"],
        [
            [
                f"{event['time']:.2f}s",
                event["name"],
                str(event["shot"]),
                f"{SFX_TABLE[event['name']][0]:.0f} Hz",
                f"{SFX_TABLE[event['name']][1]:.2f}s",
                "屏上这一下变化，声音在这里响",
            ]
            for event in events
        ],
        align_right={0},
    ))

    ensure(events, "事件表不该是空的")
    ensure(
        all(event["time"] >= 0 for event in events),
        "事件时刻不能是负数",
    )

    # ---- 提问那镜不出声 ----
    print()
    print("【2】规则：提问那一镜不出声")
    question_shots = [
        shot for shot in shots
        if "问" in shot.action or "提问" in shot.takeaway
    ]
    for shot in question_shots:
        noisy = any(event["shot"] == shot.index for event in events)
        print(f"  第 {shot.index} 镜：{shot.action} → {'有声音 ✗' if noisy else '不出声 ✓'}")
        ensure(not noisy, f"第 {shot.index} 镜是提问，不该配声音")

    # ---- 真合成一条音轨 ----
    print()
    print("【3】合成事件音轨")
    if not shutil.which("ffmpeg"):
        print("  ⊘ 跳过：本机没有 ffmpeg")
        return

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "events.wav"
    synthesize(events, out)
    size = out.stat().st_size
    print(f"  ✓ {out}（{size} 字节）")

    # 用 ffprobe 验证音轨本身是好的
    probe = shutil.which("ffprobe")
    if probe:
        proc = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True,
        )
        duration = float(proc.stdout.strip() or 0)
        print(f"  音轨时长 {duration:.2f}s")
        ensure(duration > 0, "音轨时长为 0，合成有问题")

    # ---- BGM 的规矩 ----
    print()
    print("【4】寓言（B）世界：BGM 怎么对")
    print(fmt_table(
        ["环节", "规矩"],
        [
            ["段落开头", "对到 BGM 的小节线上，不要对到拍点"],
            ["最大的揭晓", "对到音乐重新进入的那一拍"],
            ["音量", "BGM 压到 -8dB 左右，别盖住事件音"],
            ["没有 BGM", "就明说：成片目前只有事件音，等配乐"],
        ],
    ))

    print()
    print("要点：声音不是配上去的，是**登记**上去的。")
    print("      画面调用 fr.sfx(时刻, 名字) 登记事件，混音时按登记合成——")
    print("      所以声音和画面对得上是结构保证的，不是靠人手对。")


if __name__ == "__main__":
    run()
