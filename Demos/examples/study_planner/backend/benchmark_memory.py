"""用固定合成规模测量上下文构建，不接触个人数据或网络。"""
from __future__ import annotations

import json
import platform
import sqlite3
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backup import backup_repository, restore_repository
from .db import transaction
from .memory import tokenize
from .repositories import now_iso
from .turn_service import TurnService


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        service = TurnService(tmp)
        tid, _ = service.accept('default', 'query', '课程笔记有什么建议')
        with transaction(service.repo.path) as db:
            stamp = now_iso()
            db.executemany("INSERT INTO turns(id,request_id,input_hash,conversation_id,status,created_at) VALUES (?,?,?,'default','completed',?)",
                           [(f'old-{i}', f'old-{i}', 'fixture', stamp) for i in range(50000)])
            db.executemany("INSERT INTO messages VALUES (?,'default',?,?,?, ?,?,'2026-11-04',0)",
                           [(f'raw-{i}', i + 2, f'old-{i//2}', 'user' if i % 2 == 0 else 'assistant', f'历史课程记录 {i}', stamp) for i in range(100000)])
            db.executemany("INSERT INTO memories(id,kind,category,content,status,created_at,source_type) VALUES (?,'experience','note',?,'active',?,'fixture')",
                           [(f'mem-{i:05}', f'课程笔记 第{i}章，复习知识点', stamp) for i in range(10000)])
            terms = tokenize('课程笔记 复习知识点')
            db.executemany('INSERT INTO memory_terms VALUES (?,?)', [(f'mem-{i:05}', term) for i in range(10000) for term in terms])
        def sample(_: int) -> float:
            start = time.perf_counter()
            hits = service.retrieval.search('课程笔记')
            context = service.context.build(tid, 'decision', recalled=hits)
            assert context.token_count <= service.context.budget
            return (time.perf_counter() - start) * 1000
        cold = sample(0)
        with ThreadPoolExecutor(max_workers=5) as pool:
            timings = list(pool.map(sample, range(50)))
        backup = Path(tmp) / 'benchmark-backup.db'
        backup_repository(service.repo, backup)
        service.close()
        start = time.perf_counter()
        restore_repository(Path(tmp), backup, False)
        recovery_ms = (time.perf_counter() - start) * 1000
        print(json.dumps({'active_memories': 10000, 'messages': 100001, 'concurrency': 5, 'samples': 50,
                          'cold_ms': round(cold, 2), 'p50_ms': round(statistics.median(timings), 2),
                          'p95_ms': round(sorted(timings)[47], 2), 'restore_ms': round(recovery_ms, 2),
                          'machine': platform.machine(), 'platform': platform.system(), 'python': platform.python_version(),
                          'sqlite': sqlite3.sqlite_version, 'journal': 'DELETE', 'index': 'memory_terms(term,memory_id)'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
