"""把一条短片从代码渲成 MP4。

这个模块解决什么问题：把"画面是纯函数"这条规矩变成一条**可执行的流水线**。
单路 Chrome 逐帧截图 → ffmpeg 编码，全程离线，不需要 GPU、不需要录屏。

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
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = [
    "ChromeNotFound",
    "encode",
    "find_chrome",
    "probe",
    "render_frame",
    "render_frames",
    "screenshot",
    "tile_sheet",
]

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


class ChromeNotFound(RuntimeError):
    """找不到可用的 Chrome/Chromium/Edge。"""


def find_chrome() -> str:
    """找浏览器可执行文件。

    优先用系统里已经装好的 Chrome——不额外下载一份 Chromium，
    省几百 MB，也省掉"下载被墙"这种问题。
    """
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise ChromeNotFound(
        "找不到 Chrome/Chromium/Edge。装一个 Chrome，或者改 CHROME_PATH 环境变量指到可执行文件。"
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


def screenshot(
    chrome: str,
    html: Path,
    t: float,
    out_png: Path,
    width: int,
    height: int,
    timeout: float = 90.0,
) -> Path:
    """截一帧。

    ``--virtual-time-budget`` 让 Chrome 把页面里的 JS 跑完再截图，
    不加这个参数会截到一张白纸——这是最常见的"为什么渲出来是空白"。

    ``timeout`` 是防挂死的兜底：Chrome 偶尔会卡住不退出（尤其是并发时），
    没有超时的话整个渲染会永远停在这一帧上。超时会连同它拉起的
    浏览器子进程一起杀掉（单独开进程组，否则子进程会变孤儿）。
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    url = html.resolve().as_uri() + f"?t={t:.6f}"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--default-background-color=00000000",
        "--disable-extensions",
        "--mute-audio",
        # 关掉一切后台网络行为：Chrome 会尝试连组件更新、变体种子等
        # 服务，在网络不稳的机器上会拖慢甚至卡死整帧渲染。
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-sync",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--password-store=basic",
        "--use-mock-keychain",
        f"--window-size={width},{height}",
        "--virtual-time-budget=3000",
        f"--screenshot={out_png}",
        url,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise RuntimeError(f"Chrome 截图超时（{timeout:.0f}s）：t={t:.3f}")

    if proc.returncode != 0:
        raise RuntimeError(f"Chrome 退出码 {proc.returncode}：t={t:.3f}")
    if not out_png.is_file():
        raise RuntimeError(f"截图没生成：{out_png}")
    return out_png


def _kill_tree(proc: subprocess.Popen) -> None:
    """杀掉浏览器进程组。

    Chrome 会拉起一堆子进程，只杀父进程的话它们会变成孤儿继续占着 CPU。
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def render_frames(
    html: Path,
    out_dir: Path,
    times: list[float],
    width: int,
    height: int,
    *,
    jobs: int = 1,
    chrome: str | None = None,
    start_index: int = 0,
) -> list[Path]:
    """按给定时刻批量渲帧，文件名按序号连续编号（ffmpeg 要连续的序列）。

    ``start_index`` 是**续渲**用的：渲到第 400 帧断了，
    可以只补后面那几帧，编号接着往下排，不用从头再来。
    """
    chrome = chrome or find_chrome()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"{start_index + index:06d}.png" for index in range(len(times))]

    def one(pair: tuple[float, Path]) -> Path:
        t, path = pair
        # 单帧失败重试一次：并发渲染时浏览器偶发退出是常见情况，
        # 一次偶发失败不该让整条片子前功尽弃。
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return screenshot(chrome, html, t, path, width, height)
            except Exception as exc:  # noqa: BLE001 - 换成重试
                last_error = exc
        raise RuntimeError(f"渲染 t={t:.3f} 连续两次失败：{last_error}")

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(one, zip(times, targets)))
    else:
        for pair in zip(times, targets):
            one(pair)
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
    """
    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        first = screenshot(chrome, html, t, tmp_dir / "a.png", width, height)
        second = screenshot(chrome, html, t, tmp_dir / "b.png", width, height)
        digest_a = hashlib.sha256(first.read_bytes()).hexdigest()
        digest_b = hashlib.sha256(second.read_bytes()).hexdigest()
    return digest_a == digest_b


def _read_config(html_dir: Path) -> tuple[int, int, int, float]:
    """从 film.js 里读 CONFIG。

    用正则读而不是跑 node：少一个运行时依赖，实验室就能只用
    标准库 + Chrome + ffmpeg 跑起来。
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
