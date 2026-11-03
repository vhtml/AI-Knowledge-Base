"""命令行与 Web 共用事务操作，避免各自保存一份过期状态。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .repositories import uid
from .turn_service import TurnService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='学习规划师：本地规则模式')
    parser.add_argument('--task', help='只执行一句话')
    parser.add_argument('--data-dir', type=Path, default=Path.home() / '.study-planner')
    parser.add_argument('--conversation', default='default')
    parser.add_argument('--request-id', help='重试同一操作时复用此标识')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = TurnService(args.data_dir)

    def run(text: str) -> None:
        if text in ('/memory', '/today'):
            state = service.repo.snapshot(args.conversation)
            print(json.dumps(state['memory'] if text == '/memory' else state['items'], ensure_ascii=False, indent=2))
            return
        if text.startswith('/recall '):
            from .retrieval import Retrieval
            print(json.dumps(Retrieval(service.repo).search(text[8:]), ensure_ascii=False, indent=2))
            return
        result = service.submit(args.conversation, args.request_id or uid('cli'), text,
                                command='rollover' if text == '/rollover' else 'chat')
        print(result.reply or f'{result.status}：{result.error_code}（{result.turn_id}）')

    if args.task:
        run(args.task)
        return 0
    print('学习规划师 · /memory 查看记忆，/today 查看待办，/rollover 整理，/exit 退出')
    while True:
        try:
            text = input('你 > ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ('/exit', '/quit'):
            break
        if text:
            try:
                run(text)
            except Exception as exc:
                print(f'操作失败：{type(exc).__name__}: {exc}')
    return 0
