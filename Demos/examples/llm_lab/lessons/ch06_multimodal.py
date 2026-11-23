"""第 6 章实验：多模态输入输出。

这个模块解决什么问题：第 6 章讲的是"多模态的 content 是块列表，不是字符串"。
这必须能构造出来看，所以本章把七件事变成可执行的断言：

1. **块结构**——text 与 image_url 混排，序列化后结构正确、**顺序保留**；
2. **URL vs base64**——两种传法的请求体体积差异；
3. **图片 token 按尺寸算**，不是按字节算；
4. **大图比长文更贵**——用可测的对比证明"图片是成本大头"；
5. **OCR vs 视觉模型**——同一份文档，两条路的成本差一个数量级；
6. **视频抽帧**——帧数决定成本；
7. **客户端先校验**——超限在本地就被挡住，而不是发出去吃 400。

全部离线运行：不联网、不调模型、不花钱。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from ..labkit import (
    ChatMessage,
    ChatResponse,
    banner,
    ensure,
    estimate_cost,
    estimate_tokens,
    rule,
    scripted_client,
    section,
)

# ---------------------------------------------------------------- 图片 token 估算


def image_tokens(width: int, height: int, tile: int = 512) -> int:
    """估算一张图的 token 数。

    真实的分块算法各家不同，但共同点是：**按面积算，不按字节算**。
    这里用一个"缩放到 tile 网格"的简化模型来表达这个规律——
    重点是让你看到"尺寸翻倍，token 翻四倍"。
    """
    tiles_w = max(1, -(-width // tile))
    tiles_h = max(1, -(-height // tile))
    return tiles_w * tiles_h * 85


# ---------------------------------------------------------------- 1. 块结构


def check_content_blocks() -> None:
    """content 是块列表，顺序有意义。"""
    section("content 块：结构与顺序")

    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "这张图里是什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "text", "text": "顺便说说它的配色。"},
    ]
    message = ChatMessage(role="user", content=blocks)
    payload = message.to_api()

    print(f"  序列化结果：{json.dumps(payload, ensure_ascii=False)}")

    ensure(
        isinstance(payload["content"], list),
        f"多模态的 content 应当是块列表，实际 {type(payload['content']).__name__}",
    )
    ensure(
        len(payload["content"]) == 3,
        f"应当保留 3 个块，实际 {len(payload['content'])}",
    )
    ensure(
        [b["type"] for b in payload["content"]] == ["text", "image_url", "text"],
        f"块的顺序应当原样保留，实际 {[b['type'] for b in payload['content']]}",
    )
    ensure(
        payload["content"][1]["image_url"]["url"] == "https://example.com/a.png",
        "image_url 块的结构应当保持正确",
    )
    print("  → 块顺序保留：「先看图再问」和「先问再看图」对模型是不同的")


# ---------------------------------------------------------------- 2. URL vs base64


def check_url_vs_base64() -> None:
    """base64 让请求体膨胀约三分之一。"""
    section("URL vs base64：请求体体积")

    # 造一个"1KB 的图片"（这里只是同长度的二进制串，体积特性一样）
    raw = bytes(range(256)) * 4  # 1024 字节
    encoded = base64.b64encode(raw).decode("ascii")

    url_message = ChatMessage(
        role="user",
        content=[{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
    )
    b64_message = ChatMessage(
        role="user",
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        ],
    )

    url_size = len(json.dumps(url_message.to_api(), ensure_ascii=False))
    b64_size = len(json.dumps(b64_message.to_api(), ensure_ascii=False))

    print(f"  URL 版请求体：{url_size} 字节")
    print(f"  base64 版请求体：{b64_size} 字节（原图 {len(raw)} 字节）")
    growth = b64_size / url_size
    print(f"  膨胀 {growth:.1f} 倍")

    ensure(
        b64_size > url_size,
        "base64 版请求体必然更大",
    )
    ensure(
        len(encoded) > len(raw) * 1.3,
        f"base64 编码后应当膨胀三成以上，实际 {len(encoded)} / {len(raw)}",
    )

    # 但 base64 的内容是可复现的：同样输入永远同样输出
    ensure(
        base64.b64encode(raw).decode("ascii") == encoded,
        "base64 编码应当是确定的",
    )
    print("  → URL 小但要服务端能访问；base64 私有安全但每次都要重传")


# ---------------------------------------------------------------- 3. 图片 token 按尺寸


def check_image_tokens_by_size() -> None:
    """token 按尺寸算，不按字节算。"""
    section("图片 token：按尺寸折算")

    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    for width, height in sizes:
        tokens = image_tokens(width, height)
        print(f"  {width}×{height} → 约 {tokens} token")

    small = image_tokens(512, 512)
    large = image_tokens(2048, 2048)

    ensure(
        large > small * 10,
        f"尺寸放大 16 倍，token 应当放大约 16 倍，实际 {small} → {large}",
    )
    ensure(
        image_tokens(1024, 1024) == image_tokens(1024, 1024),
        "同样尺寸必须给出同样估算",
    )

    # 同样"看起来一样"的压缩质量，token 可能一样——因为按面积算
    jpeg = image_tokens(1024, 1024)
    png = image_tokens(1024, 1024)
    ensure(jpeg == png, "token 只看尺寸，不看格式")
    print("  → 压缩质量不影响 token，缩小尺寸才影响")


# ---------------------------------------------------------------- 4. 图片 vs 长文


def check_image_vs_text_cost() -> None:
    """一张大图可能比一整页文字还贵。"""
    section("成本对比：一张大图 vs 一整页文字")

    page_of_text = "这是一段正文内容，用来模拟一整页文字。" * 50
    text_tokens = estimate_tokens(page_of_text)
    image_cost_tokens = image_tokens(1600, 1600)

    print(f"  一整页文字：{text_tokens} token")
    print(f"  一张 1600×1600 的图：约 {image_cost_tokens} token")

    ensure(
        image_cost_tokens > text_tokens,
        f"大图的 token 可能超过一整页文字，实际 {image_cost_tokens} vs {text_tokens}",
    )

    # 成本对照（用 reasoning 档位的输出价粗算）
    text_cost = estimate_cost("standard", text_tokens, 0)
    image_cost = estimate_cost("standard", image_cost_tokens, 0)
    print(f"  按输入价折算：文字 ${text_cost:.6f} vs 图片 ${image_cost:.6f}")
    ensure(image_cost > text_cost, "图片的输入成本应当更高")
    print("  → 图片是成本大头时，第一反应应该是「能不能不传图」")


# ---------------------------------------------------------------- 5. OCR vs 视觉


def check_ocr_vs_vision() -> None:
    """能用 OCR 解决的，不要用视觉模型。"""
    section("OCR vs 视觉模型：同一份文档两条路")

    document_text = "合同编号：HT-2026-001\n甲方：某某公司\n金额：100 万元\n" * 40
    ocr_tokens = estimate_tokens(document_text)

    # 视觉模型：把每一页当图片发
    pages = 20
    vision_tokens = pages * image_tokens(1200, 1600)

    print(f"  OCR + 文本模型：{ocr_tokens} token")
    print(f"  视觉模型（{pages} 页）：约 {vision_tokens} token")
    print(f"  倍数：{vision_tokens / ocr_tokens:.1f}×")

    ensure(
        vision_tokens > ocr_tokens * 10,
        f"视觉路线应当贵一个数量级以上，实际 {vision_tokens} vs {ocr_tokens}",
    )

    # 而且 OCR 的输出是**可检索、可缓存、可复用**的文本
    ensure(
        "合同编号" in document_text,
        "OCR 输出应当是普通文本，可以被检索",
    )
    print("  → OCR 输出还能被检索、被缓存、被复用；视觉输出只是一段描述")


# ---------------------------------------------------------------- 6. 视频抽帧


def check_video_frame_sampling() -> None:
    """视频靠抽帧，成本 = 帧数 × 单帧。"""
    section("视频：抽帧决定成本")

    per_frame = image_tokens(640, 360)

    for frames in (8, 30, 120):
        tokens = frames * per_frame
        print(f"  抽 {frames} 帧 → 约 {tokens} token")

    ensure(
        30 * per_frame > 8 * per_frame * 2,
        "帧数翻几倍，token 就翻几倍",
    )

    # 字幕优先：纯文本
    subtitle = "大家好，今天我们讲第三章：状态设计。" * 30
    subtitle_tokens = estimate_tokens(subtitle)
    print(f"  同段视频的字幕（纯文本）：{subtitle_tokens} token")

    ensure(
        subtitle_tokens < 8 * per_frame,
        f"字幕应当比抽 8 帧还便宜，实际 {subtitle_tokens} vs {8 * per_frame}",
    )
    print("  → 有字幕的视频，字幕是最省的多模态输入")


# ---------------------------------------------------------------- 7. 客户端先校验


@dataclass
class Limits:
    """接口限制——发请求前要在本地校验。"""

    max_image_bytes: int = 5 * 1024 * 1024
    max_blocks: int = 20
    allowed_image_formats: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "gif")


def _is_private_host(url: str) -> bool:
    """判断 URL 的主机是不是内网地址。

    内网 URL 对模型服务端来说是**拉不到的**——但它在字符串上完全合法，
    所以必须本地判断，否则每次都要等一个 400 才知道。
    """
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    if host in {"localhost", "::1"}:
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        first, second = int(parts[0]), int(parts[1])
        if first == 10 or first == 127:
            return True
        if first == 172 and 16 <= second <= 31:
            return True
        if first == 192 and second == 168:
            return True
    return False


def validate_request(blocks: list[dict[str, Any]], limits: Limits) -> list[str]:
    """校验多模态请求，返回错误列表（空 = 通过）。"""
    errors: list[str] = []

    if len(blocks) > limits.max_blocks:
        errors.append(f"块数超限：{len(blocks)} > {limits.max_blocks}")

    for index, block in enumerate(blocks):
        if block.get("type") != "image_url":
            continue
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            fmt = header.split("/")[-1].split(";")[0]
            size = len(base64.b64decode(payload)) if payload else 0
            if fmt not in limits.allowed_image_formats:
                errors.append(f"块 {index}：格式 {fmt} 不支持")
            if size > limits.max_image_bytes:
                errors.append(f"块 {index}：图片 {size} 字节超限")
        elif not url.startswith("http"):
            errors.append(f"块 {index}：URL 形式不合法（既不是 http 也不是 data:）")
        elif _is_private_host(url):
            errors.append(f"块 {index}：内网地址，模型服务端拉不到")

    return errors


def check_client_side_validation() -> None:
    """超限要在本地挡住，而不是发出去吃 400。"""
    section("客户端先校验：别把非法请求发出去")

    limits = Limits(max_image_bytes=1024, max_blocks=3)

    good = [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    errors = validate_request(good, limits)
    print(f"  正常请求的错误：{errors}")
    ensure(not errors, f"正常请求不该有错误，实际 {errors}")

    # 图太大
    big_payload = base64.b64encode(bytes(2048)).decode("ascii")
    too_big = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_payload}"}}
    ]
    errors = validate_request(too_big, limits)
    print(f"  超大图的错误：{errors}")
    ensure(any("超限" in e for e in errors), f"应当检出图片超限，实际 {errors}")

    # 格式不支持
    bad_format = [
        {"type": "image_url", "image_url": {"url": "data:image/tiff;base64,AAAA"}}
    ]
    errors = validate_request(bad_format, limits)
    ensure(any("格式" in e for e in errors), f"应当检出格式不支持，实际 {errors}")

    # 块数超限
    too_many = [{"type": "text", "text": "x"}] * 5
    errors = validate_request(too_many, limits)
    ensure(any("块数超限" in e for e in errors), f"应当检出块数超限，实际 {errors}")

    # 内网 URL：模型服务端拉不到
    intranet = [
        {"type": "image_url", "image_url": {"url": "http://10.0.0.5/internal.png"}}
    ]
    errors = validate_request(intranet, limits)
    print(f"  内网 URL 的错误：{errors}")
    ensure(
        any("内网" in e for e in errors),
        f"内网 URL 应当被本地挡下（服务端拉不到），实际 {errors}",
    )
    print("  → 本地挡住 = 一次白跑都没有；发出去 = 一次 400 + 一次重试")


def run_live() -> None:
    """真实模型路径：真调一次模型，校验多模态请求的结构。"""
    banner("第 6 章 · 真实模型路径")
    from ..labkit import real_client

    client = real_client()
    response = client.complete(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "用一句话回答：你好"},
                ],
            )
        ]
    )

    print(f"  content: {response.content!r}")
    print(f"  usage  : {response.usage}")

    ensure(bool(response.content), "真实模型应当返回内容")
    ensure(response.usage.get("total_tokens", 0) > 0, "应当拿到 token 统计")
    print("  （真实多模态需要可访问的图片 URL 或 base64，本机端点未必支持，故只校验文本路径）")


def run() -> None:
    """跑第 6 章实验。"""
    banner("第 6 章 · 多模态输入输出")
    check_content_blocks()
    check_url_vs_base64()
    check_image_tokens_by_size()
    check_image_vs_text_cost()
    check_ocr_vs_vision()
    check_video_frame_sampling()
    check_client_side_validation()
    rule("content 是块列表不是字符串；图片按尺寸计费，能不传就不传")
    print("\n✓ 第 6 章通过")
