"""事件音效与配乐。

这个模块解决什么问题：把"无人声片子怎么撑节奏"这件事落到实处。

这类片子没有旁白，节奏全靠画面和声音扛。两条路：

- **夜曲（C）**：声音完全由代码合成。每一下变化配一声，
  推拉时给一个上滑音，落定时给一个低音，**提问那一镜不出声**
  （留白本身就是信号）。
- **寓言（B）**：用户给一首 BGM，把每段开头对到小节线上，
  最大的那次揭晓对到音乐重新进入的那一拍。

事件音用 ffmpeg 的正弦源合成，不依赖任何采样库。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["SFX_TABLE", "event_table", "mix_with_bgm", "synthesize"]

SFX_TABLE: dict[str, tuple[float, float]] = {
    "noise": (180.0, 0.30),
    "split": (520.0, 0.14),
    "rise": (760.0, 0.12),
    "merge": (260.0, 0.34),
}
"""事件名 → (频率 Hz, 时长秒)。

频率是手挑的，不是声学设计：低音收、高音起、撕裂用中频。
目的是让观众感到"这里发生了一件事"，不是听出这是什么音。
"""


def event_table(shots: list[Any], cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从分镜表生成事件音表。

    规则：
    - 有 sfx 的镜头，在该句字幕开始的时刻响一声；
    - "提问"那一镜不出声——留白比音效更能提示"这里要停一下"。
    """
    events: list[dict[str, Any]] = []
    for index, shot in enumerate(shots):
        name = getattr(shot, "sfx", "")
        if not name:
            continue
        if getattr(shot, "method", "") == "不画" and "问" in getattr(shot, "action", ""):
            continue
        if index >= len(cues):
            break
        events.append({
            "time": float(cues[index]["t"]),
            "name": name,
            "shot": index + 1,
        })
    return events


def synthesize(events: list[dict[str, Any]], out_path: Path) -> Path:
    """用 ffmpeg 合成事件音轨。

    每个事件一路正弦源 + ``adelay`` 定位到对应时刻，再用 ``amix`` 混在一起。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")
    if not events:
        raise ValueError("没有事件可以合成")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [ffmpeg, "-y"]
    for event in events:
        frequency, duration = SFX_TABLE.get(event["name"], (440.0, 0.15))
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={duration}"]

    filters = []
    labels = []
    for index, event in enumerate(events):
        delay = int(event["time"] * 1000)
        filters.append(f"[{index}:a]adelay={delay}|{delay},volume=0.35[a{index}]")
        labels.append(f"[a{index}]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[out]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-ac", "1",
        "-ar", "44100",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"合成失败：\n{proc.stderr.strip()}")
    return out_path


def mix_with_bgm(video: Path, bgm: Path, out_path: Path, bgm_gain_db: float = -8.0) -> Path:
    """把 BGM 混进视频，BGM 默认压低 8dB。

    BGM 压得太满会盖住事件音——这是这类片子最常见的"难听"来源。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(bgm),
        "-filter_complex",
        f"[1:a]volume={bgm_gain_db}dB[a1];[0:a][a1]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"混音失败：\n{proc.stderr.strip()}")
    return out_path
