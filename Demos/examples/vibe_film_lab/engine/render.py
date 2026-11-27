"""把一条短片从代码渲成 MP4。

这个模块解决什么问题：把"画面是纯函数"这条规矩变成一条**可执行的流水线**。
ego lite 逐帧截图 → ffmpeg 编码，全程离线，不需要 GPU、不需要录屏。

浏览器用 **ego lite**（`ego-browser` CLI 驱动），不依赖系统里装没装 Chrome：
视口用 CDP 的 ``Emulation.setDeviceMetricsOverride`` 钉死成目标分辨率，
所以截出来的帧就是 1920×1080，和窗口多大无关。

四个检查动作，都是实战里每天要用的：

- ``--sheet``：在几个时刻各截一帧，拼成一张图，用来一眼看出一片里
  哪几镜的风格不一致、字幕区有没有被别的元素占掉。
- ``--stills``：截原尺寸静帧，用来看细节（字号、描边、小字位置）。
- ``--motion``：在一段时间里连取 N 帧拼成一张图，用来看转场——
  转场翻车基本都发生在两个场景交接的那两三帧。
- ``--det``：同一个 t 渲两遍，必须**逐字节一致**。这是整条流水线的
  地基：不一致就说明画面里藏了状态，后面的逐帧导出全是错的。

用法：

    python -m vibe_film_lab.engine.render <工程目录> --sheet 3 8 15 22
    python -m vibe_film_lab.engine.render <工程目录> --stills 5.9 18.1
    python -m vibe_film_lab.engine.render <工程目录> --motion 8.2 10.6
    python -m vibe_film_lab.engine.render <工程目录> --det
    python -m vibe_film_lab.engine.render <工程目录> --clip 0 30
    python -m vibe_film_lab.engine.render <工程目录> --jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

__all__ = [
    "BrowserNotFound",
    "encode",
    "find_browser",
    "probe",
    "render_frames",
    "screenshot",
    "tile_sheet",
]

EGO_BROWSER_CANDIDATES = ("ego-browser",)
"""ego lite 的命令行入口。它通过 ego lite 服务驱动浏览器，不直接拉内核。"""


class BrowserNotFound(RuntimeError):
    """找不到 ego lite 的命令行入口。"""


def find_browser() -> str:
    """找 ego-browser 可执行文件。

    渲染用的是 **ego lite**，不是系统里的 Chrome/Chromium/Edge。
    装好 ego lite 之后 ``ego-browser`` 会在 PATH 上（默认 ``~/.local/bin``）。
    """
    for name in EGO_BROWSER_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise BrowserNotFound(
        "找不到 ego-browser。装一下 ego lite（装完 CLI 在 ~/.local/bin/ego-browser），"
        "或者把它的路径加进 PATH。"
    )


def find_ffmpeg() -> str:
    """找 ffmpeg。"""
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("找不到 ffmpeg。macOS 用 brew install ffmpeg。")
    return found


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """跑一条命令，失败时把 stderr 抛出来。"""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"命令失败（退出码 {proc.returncode}）：{' '.join(cmd[:3])} …\n{proc.stderr.strip()}"
        )
    return proc


# ---------------------------------------------------------------- ego lite 驱动脚本


_NODE_RENDER = """\
const payload = __PAYLOAD__;

const task = await taskSpace(payload.space);
let failure = null;
let pages = [];

try {
  for (let index = 0; index < payload.jobs; index += 1) {
    let page;
    if (index === 0) {
      page = task.page("p1");
    } else {
      try {
        page = await task.newPage();
      } catch (err) {
        break;  // 页面预算用完了，就用手上这几页
      }
    }
    // 把视口钉死成目标分辨率：截出来的帧和浏览器窗口多大无关。
    await page.cdp("Emulation.setDeviceMetricsOverride", {
      width: payload.width,
      height: payload.height,
      deviceScaleFactor: 1,
      mobile: false,
    });
    pages.push(page);
  }

  // 按页分片，每页一条串行队列，页与页之间并行。
  const chunks = pages.map(() => []);
  payload.items.forEach((item, index) => chunks[index % pages.length].push(item));

  await Promise.all(chunks.map(async (chunk, index) => {
    const page = pages[index];
    for (const item of chunk) {
      const url = payload.base + "?t=" + item.t.toFixed(6);
      await page.goto(url);
      await page.waitForLoadState("load");
      try {
        await page.screenshot({ path: item.path });
      } catch (err) {
        // 截图偶尔会因为页面无响应失败，重试一次就好。
        await page.screenshot({ path: item.path });
      }
    }
  }));
} catch (err) {
  failure = String(err && err.stack ? err.stack : err);
}

if (!failure) {
  await task.finish({ keep: [] });
}
console.log(JSON.stringify({ ok: !failure, error: failure, count: payload.items.length }));
"""

_NODE_DET = """\
const payload = __PAYLOAD__;

const task = await taskSpace(payload.space);
let failure = null;

try {
  const page = task.page("p1");
  await page.cdp("Emulation.setDeviceMetricsOverride", {
    width: payload.width,
    height: payload.height,
    deviceScaleFactor: 1,
    mobile: false,
  });
  const url = payload.base + "?t=" + payload.t.toFixed(6);

  await page.goto(url);
  await page.waitForLoadState("load");
  await page.screenshot({ path: payload.first });

  // 重新加载让 JS 从头再跑一遍——这才是第二次独立的渲染。
  // 直接 goto 同一个 URL 可能被当成没有变化，起不到"渲两遍"的作用。
  await page.reload();
  await page.waitForLoadState("load");
  await page.screenshot({ path: payload.second });
} catch (err) {
  failure = String(err && err.stack ? err.stack : err);
}

if (!failure) {
  await task.finish({ keep: [] });
}
console.log(JSON.stringify({ ok: !failure, error: failure }));
"""


def _run_ego(script: str, payload: dict) -> dict:
    """把一段 Node 脚本交给 ego-browser 跑，拿回 JSON 结果。

    脚本从 stdin 进去、结果从 stdout 出来，中间不经过 shell 引号，
    所以路径里有空格、中文也不会被吃掉。
    """
    browser = find_browser()
    # 每次调用开一个**独立命名**的 task space：同名空间是复用的，
    # 两个渲染并发跑时后者会把前者的空间关掉，前者就报"空间不存在"了。
    payload = {**payload, "space": f"vibe_film_{uuid.uuid4().hex[:12]}"}
    source = script.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    proc = subprocess.run(
        [browser, "nodejs"],
        input=source,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ego-browser 退出码 {proc.returncode}：\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    # ego-browser 把脚本结果写进 stderr（stdout 留给浏览器侧的输出），
    # 两个流都扫一遍，哪边有 JSON 用哪边。
    line = ""
    for stream in (proc.stdout, proc.stderr):
        for candidate in reversed(stream.strip().splitlines()):
            if candidate.strip().startswith("{"):
                line = candidate.strip()
                break
        if line:
            break
    if not line:
        raise RuntimeError(
            f"ego-browser 没吐出结果：\n{proc.stdout.strip()[-800:]}\n{proc.stderr.strip()[-800:]}"
        )
    result = json.loads(line)
    if not result.get("ok"):
        raise RuntimeError(f"ego lite 渲染失败：{result.get('error')}")
    return result


def screenshot(
    html: Path,
    t: float,
    out_png: Path,
    width: int,
    height: int,
) -> Path:
    """截一帧。

    单帧也走 ego lite：一次调用渲一张，和批量渲染用同一条路径，
    免得出现"单帧能截、批量渲不出来"这种鬼故事。
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    render_frames(html, out_png.parent, [t], width, height, names=[out_png.name])
    return out_png


def render_frames(
    html: Path,
    out_dir: Path,
    times: list[float],
    width: int,
    height: int,
    *,
    jobs: int = 1,
    start_index: int = 0,
    names: list[str] | None = None,
) -> list[Path]:
    """按给定时刻批量渲帧，文件名按序号连续编号（ffmpeg 要连续的序列）。

    ``start_index`` 是**续渲**用的：渲到第 400 帧断了，
    可以只补后面那几帧，编号接着往下排，不用从头再来。

    ``names`` 给出时按它命名（单帧截图用），否则按序号连续编号。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = [f"{start_index + index:06d}.png" for index in range(len(times))]
    items = [
        {"t": t, "path": str((out_dir / name).resolve())}
        for t, name in zip(times, names)
    ]
    _run_ego(_NODE_RENDER, {
        "base": html.resolve().as_uri(),
        "width": width,
        "height": height,
        "jobs": max(1, jobs),
        "items": items,
    })
    targets = [out_dir / name for name in names]
    missing = [path for path in targets if not path.is_file()]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 帧没生成，第一张是 {missing[0]}")
    return targets


def encode(frames_dir: Path, out_path: Path, fps: int) -> Path:
    """把帧序列编码成 MP4。

    ``yuv420p`` + ``+faststart``：前者保证微信、B 站、各手机都认，
    后者把 moov 原子搬到文件头，别人点开就能直接播。
    """
    ffmpeg = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return out_path


def probe(path: Path) -> dict:
    """读视频信息：分辨率、帧率、帧数、时长。"""
    ffmpeg = shutil.which("ffprobe") or find_ffmpeg().replace("ffmpeg", "ffprobe")
    proc = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or [{}]
    return streams[0]


def tile_sheet(frames: list[Path], out_path: Path, columns: int = 4) -> Path:
    """把若干帧拼成一张联络表（contact sheet）。"""
    ffmpeg = find_ffmpeg()
    rows = max(1, (len(frames) + columns - 1) // columns)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            shutil.copy(frame, tmp_dir / f"{index:03d}.png")
        _run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "1",
                "-i",
                str(tmp_dir / "%03d.png"),
                "-vf",
                f"scale=640:-1,tile={columns}x{rows}:padding=10:margin=10:color=black",
                "-frames:v",
                "1",
                str(out_path),
            ]
        )
    return out_path


def determinism_check(html: Path, t: float, width: int, height: int) -> bool:
    """同一个 t 渲两遍，比对字节是否一致。

    不一致说明画面里藏了状态（累加的计数、没播种的随机数、
    依赖上一帧的插值）。这种 bug 在预览里看不出来，
    只会在逐帧导出后表现为"同一段动画在片子里前后抖"。

    第二次是**重新加载页面**后渲的——必须让 JS 真的从头再跑一遍，
    否则比的是同一张图，比什么都过。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        first = tmp_dir / "a.png"
        second = tmp_dir / "b.png"
        _run_ego(_NODE_DET, {
            "base": html.resolve().as_uri(),
            "width": width,
            "height": height,
            "t": t,
            "first": str(first),
            "second": str(second),
        })
        digest_a = hashlib.sha256(first.read_bytes()).hexdigest()
        digest_b = hashlib.sha256(second.read_bytes()).hexdigest()
    return digest_a == digest_b


def _read_config(html_dir: Path) -> tuple[int, int, int, float]:
    """从 film.js 里读 CONFIG。

    用正则读而不是跑 node：少一个运行时依赖，实验室就能只用
    标准库 + ego lite + ffmpeg 跑起来。
    """
    source = (html_dir / "film.js").read_text(encoding="utf-8")
    import re

    def num(name: str, default: int) -> int:
        match = re.search(rf"{name}\s*:\s*(\d+)", source)
        return int(match.group(1)) if match else default

    dur_match = re.search(r"DUR\s*:\s*([\d.]+)", source)
    return num("W", 1920), num("H", 1080), num("FPS", 30), float(dur_match.group(1)) if dur_match else 0.0


def main(argv: list[str]) -> int:
    """命令行入口。

    返回进程退出码。
    """
    parser = argparse.ArgumentParser(description="把 Vibe 知识短片渲成 MP4")
    parser.add_argument("film", help="工程目录（里面有 index.html 和 film.js）")
    parser.add_argument("--out", default=None, help="输出 MP4 路径，默认 <工程目录>/out/film.mp4")
    parser.add_argument("--fps", type=int, default=None, help="帧率，默认读 film.js 的 CONFIG.FPS")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--clip", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--sheet", nargs="+", type=float, metavar="T")
    parser.add_argument("--stills", nargs="+", type=float, metavar="T")
    parser.add_argument("--motion", nargs=2, type=float, metavar=("T0", "T1"))
    parser.add_argument("--det", action="store_true")
    parser.add_argument(
        "--frames", nargs=2, type=int, metavar=("START", "END"),
        help="只渲这一段帧号（续渲用），不重新编码",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--serve", action="store_true", help="起本地服务器预览")
    args = parser.parse_args(argv[1:])

    film_dir = Path(args.film).resolve()
    html = film_dir / "index.html"
    if not html.is_file():
        print(f"找不到 {html}", file=sys.stderr)
        return 2

    width, height, fps, duration = _read_config(film_dir)
    width = args.width or width
    height = args.height or height
    fps = args.fps or fps

    if args.serve:
        import http.server
        import socketserver

        handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=film_dir, **kw)  # noqa: E731
        with socketserver.TCPServer(("127.0.0.1", 8737), handler) as httpd:
            print(f"预览地址：http://127.0.0.1:8737/  （Ctrl-C 结束）")
            httpd.serve_forever()
        return 0

    if args.det:
        ok = determinism_check(html, 3.0, width, height)
        print("确定性检查：" + ("通过，同一 t 两帧逐字节一致" if ok else "失败！画面里混进了状态"))
        return 0 if ok else 1

    if args.sheet:
        with tempfile.TemporaryDirectory() as tmp:
            frames = render_frames(html, Path(tmp), args.sheet, width, height, jobs=args.jobs)
            out = Path(tmp) / "sheet.jpg"
            tile_sheet(frames, out)
            target = film_dir / "out" / "sheet.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, target)
        print(f"联络表：{target}")
        return 0

    if args.stills:
        out_dir = film_dir / "out" / "stills"
        frames = render_frames(html, out_dir, args.stills, width, height, jobs=args.jobs)
        print(f"静帧：{len(frames)} 张 → {out_dir}")
        return 0

    if args.motion:
        t0, t1 = args.motion
        times = [t0 + (t1 - t0) * i / 11 for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            frames = render_frames(html, Path(tmp), times, width, height, jobs=args.jobs)
            out = Path(tmp) / "motion.jpg"
            tile_sheet(frames, out, columns=6)
            target = film_dir / "out" / "motion.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, target)
        print(f"转场检查图：{target}")
        return 0

    if args.frames:
        start_frame, end_frame = args.frames
        count = end_frame - start_frame
        times = [start_frame / fps + index / fps for index in range(count)]
        frames_dir = film_dir / "out" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"续渲 {count} 帧（帧号 {start_frame}–{end_frame - 1}）…")
        render_frames(
            html, frames_dir, times, width, height, jobs=args.jobs, start_index=start_frame
        )
        print(f"帧目录：{frames_dir}")
        return 0

    start, end = (args.clip or (0.0, duration))
    count = int(round((end - start) * fps))
    times = [start + i / fps for i in range(count)]
    frames_dir = film_dir / "out" / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    print(f"渲 {count} 帧（{start:.2f}s – {end:.2f}s，{fps}fps，{args.jobs} 路并行）…")
    render_frames(html, frames_dir, times, width, height, jobs=args.jobs)
    out_path = Path(args.out) if args.out else film_dir / "out" / "film.mp4"
    encode(frames_dir, out_path, fps)
    print(f"成片：{out_path}")
    print(json.dumps(probe(out_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
