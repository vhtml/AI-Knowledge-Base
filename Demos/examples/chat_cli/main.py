"""命令行入口：解析参数 → 装配会话 → 进入交互循环。

必须在 Demos/examples 目录下运行（包在 examples 下）：

    cd Demos/examples

    uv run --project .. python -m chat_cli              # 推荐：自动带上 Demos 的依赖
    uv run --project .. python -m chat_cli --mock       # 离线演示，不联网
    python3 -m chat_cli --mock                          # 裸解释器也能跑离线模式
    python3 -m chat_cli --resume 20261026-101500-a1b2
    python3 -m chat_cli --list
    python3 -m chat_cli --show 20261026-101500-a1b2

从别的目录运行时用 PYTHONPATH 指向本目录：

    PYTHONPATH=Demos/examples python3 -m chat_cli --mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # 没装 python-dotenv 也能跑 --mock
    load_dotenv = None  # type: ignore[assignment]

from .chat import ChatSession, Conversation
from .context import (
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_KEEP_RECENT_TURNS,
    ContextWindow,
)
from .llm import LLMClient, MockLLMClient, load_env_config
from .storage import (
    DEFAULT_SESSION_DIR,
    SessionMeta,
    SessionStore,
    new_session_id,
)

DEFAULT_SYSTEM_PROMPT = """你是一个命令行对话助手。
回答简洁、结构清晰，默认使用中文。
如果引用了之前对话里的结论，会简短说明依据来自哪一轮。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat_cli",
        description="命令行聊天助手：多轮对话、流式输出、历史存档。",
    )

    parser.add_argument("--session", metavar="ID", help="使用指定会话 ID")
    parser.add_argument("--resume", metavar="ID", help="恢复指定会话的历史")
    parser.add_argument("--list", action="store_true", help="列出所有会话")
    parser.add_argument("--show", metavar="ID", help="打印某个会话的完整消息")
    parser.add_argument(
        "--session-dir",
        metavar="DIR",
        default=str(DEFAULT_SESSION_DIR),
        help=f"会话文件存放目录（默认 {DEFAULT_SESSION_DIR}）",
    )

    parser.add_argument("--model", metavar="NAME", help="模型名，覆盖 LLM_MODEL")
    parser.add_argument(
        "--base-url", metavar="URL", help="接口地址，覆盖 LLM_BASE_URL"
    )
    parser.add_argument("--api-key", metavar="KEY", help="API Key，覆盖 LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")

    parser.add_argument(
        "--system", metavar="TEXT", help="系统提示词，覆盖默认值"
    )
    parser.add_argument(
        "--system-file", metavar="PATH", help="从文件读取系统提示词"
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_BUDGET_TOKENS,
        help=f"上下文 token 预算（默认 {DEFAULT_BUDGET_TOKENS}）",
    )
    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=DEFAULT_KEEP_RECENT_TURNS,
        help=f"始终保留最近几轮原文（默认 {DEFAULT_KEEP_RECENT_TURNS}）",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出，一次性返回完整回答",
    )
    parser.add_argument(
        "--mock", action="store_true", help="离线演示模式，使用假客户端不联网"
    )

    return parser


def resolve_system_prompt(args: argparse.Namespace) -> str:
    """系统提示词优先级：--system > --system-file > 默认值。"""

    if args.system:
        return args.system

    if args.system_file:
        path = Path(args.system_file).expanduser()

        if not path.exists():
            print(f"[错误] 系统提示词文件不存在：{path}", file=sys.stderr)
            raise SystemExit(1)

        return path.read_text(encoding="utf-8").strip()

    return DEFAULT_SYSTEM_PROMPT.strip()


def list_sessions(store: SessionStore) -> int:
    sessions = store.list_sessions()

    if not sessions:
        print("（还没有保存任何会话）")
        return 0

    print(f"共 {len(sessions)} 个会话（目录：{store.root}）：\n")

    for meta in sessions:
        print(
            f"  {meta.id}  {meta.created_at}  "
            f"{meta.message_count:>3} 条  {meta.total_tokens:>6}t  "
            f"{meta.title or '未命名'}"
        )

    return 0


def show_session(store: SessionStore, session_id: str) -> int:
    if not store.exists(session_id):
        print(f"[错误] 会话不存在：{session_id}", file=sys.stderr)
        return 1

    meta = store.load_meta(session_id)
    messages = store.load_messages(session_id)

    if meta is not None:
        print(f"会话 {meta.id}")
        print(f"  创建    : {meta.created_at}")
        print(f"  更新    : {meta.updated_at}")
        print(f"  模型    : {meta.model}")
        print(f"  消息数  : {meta.message_count}（实际 {len(messages)}）")
        print(f"  压缩次数: {meta.compactions}\n")

    for message in messages:
        flag = "（摘要）" if message.is_synthetic else ""
        print(f"[{message.created_at}] {message.label}{flag}：")
        print(message.content)
        print("-" * 60)

    return 0


def build_client(args: argparse.Namespace) -> tuple[object, str]:
    """构造模型客户端，返回 (client, model)。"""

    config = load_env_config()

    model = args.model or config["model"]
    base_url = args.base_url or config["base_url"] or None
    api_key = args.api_key or config["api_key"] or None

    if args.mock:
        return MockLLMClient(), "mock-model"

    # 先查依赖：没有 openai 包，后面所有检查都没意义
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        print(
            f"[错误] 缺少依赖：{exc.name}。\n"
            "        请先执行 `cd Demos && uv sync` 安装依赖，\n"
            "        或直接用 `uv run --project .. python -m chat_cli` 运行；\n"
            "        只想看效果可以用 `--mock` 离线模式（不需要任何依赖）。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not api_key:
        print(
            "[错误] 没有找到 API Key。请设置环境变量 LLM_API_KEY，"
            "或使用 --api-key 指定；也可以用 --mock 离线演示。\n"
            "        提示：仓库里的密钥在 Demos/.env，"
            "用 `uv run --project .. python -m chat_cli` 会自动加载。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        client = LLMClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
        )
    except ImportError as exc:
        print(
            f"[错误] 缺少依赖：{exc.name}。\n"
            "        请先执行 `cd Demos && uv sync` 安装依赖，\n"
            "        或直接用 `uv run --project .. python -m chat_cli` 运行；\n"
            "        只想看效果可以用 `--mock` 离线模式（不需要任何依赖）。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return client, model


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # .env 放在 Demos/ 下，load_dotenv 会向上查找
    if load_dotenv is not None:
        load_dotenv()
    elif not args.mock:
        print(
            "[提示] 未安装 python-dotenv，改为直接读取环境变量。",
            file=sys.stderr,
        )

    if args.max_context_tokens < 8 or args.keep_recent_turns < 1:
        print("[错误] 上下文预算至少为 8，保留轮数至少为 1。", file=sys.stderr)
        return 1

    store = SessionStore(args.session_dir)

    if args.list:
        return list_sessions(store)

    if args.show:
        return show_session(store, args.show)

    system_prompt = resolve_system_prompt(args)
    client, model = build_client(args)

    # ------------------------------------------------------------
    # 1. 定位会话
    # ------------------------------------------------------------

    resuming = bool(args.resume)

    if resuming:
        session_id = args.resume

        if not store.exists(session_id):
            print(f"[错误] 会话不存在：{session_id}", file=sys.stderr)
            return 1
    else:
        session_id = args.session or new_session_id()

    if resuming and not args.system and not args.system_file:
        meta = store.load_meta(session_id)
        if meta is not None:
            system_prompt = meta.system_prompt

    if not resuming and store.exists(session_id):
        print(f"[错误] 会话已存在：{session_id}；请使用 --resume。", file=sys.stderr)
        return 1

    # ------------------------------------------------------------
    # 2. 恢复或创建工作记忆
    # ------------------------------------------------------------

    window = ContextWindow(
        system_prompt=system_prompt,
        budget_tokens=args.max_context_tokens,
        keep_recent_turns=args.keep_recent_turns,
    )

    if resuming:
        recent, summary = store.load_context_state(session_id)

        if summary is not None:
            window.restore(summary, recent)
            print(f"[恢复] 已载入摘要 + {len(recent)} 条近期消息")
        else:
            window.restore(None, recent)
            print(f"[恢复] 已载入 {len(recent)} 条历史消息")
    else:
        store.save_meta(
            SessionMeta(
                id=session_id,
                model=model,
                system_prompt=system_prompt,
                title="",
            )
        )
        store.append_session_start(session_id, model, system_prompt)

    # ------------------------------------------------------------
    # 3. 进入交互
    # ------------------------------------------------------------

    conversation = Conversation(
        window=window,
        client=client,
        store=store,
        session_id=session_id,
        use_stream=not args.no_stream,
    )

    session = ChatSession(
        conversation=conversation,
        store=store,
        session_id=session_id,
        model=model,
        mock=args.mock,
    )

    session.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
