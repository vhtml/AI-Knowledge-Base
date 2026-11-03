"""验证重试、取消和模型等待不占有数据库锁。"""
from __future__ import annotations

import tempfile
import threading
import unittest
import json
from pathlib import Path
from .db import transaction
from .memory_types import ServiceError
from .turn_service import TurnService


class TurnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.service = TurnService(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_idempotency(self) -> None:
        first = self.service.submit('default', 'key', '我想学 Python')
        second = self.service.submit('default', 'key', '我想学 Python')
        self.assertEqual(first.turn_id, second.turn_id)
        self.assertEqual(len(self.service.repo.snapshot()['goals']), 1)
        with self.assertRaises(ServiceError):
            self.service.submit('default', 'key', '我想学 Rust')

    def test_cancel(self) -> None:
        tid, _ = self.service.accept('default', 'cancel', '我想学 Python')
        self.service.cancel(tid)
        self.assertEqual(self.service.run(tid, '我想学 Python').status, 'cancelled')
        self.assertFalse(self.service.repo.snapshot()['goals'])
        result = self.service.submit('default', 'committed', '我想学 Python')
        self.assertEqual(self.service.cancel(result.turn_id).status, 'completed')

    def test_model_outside_transaction(self) -> None:
        entered, release = threading.Event(), threading.Event()
        class SlowModel:
            def complete_json(self, messages):
                entered.set(); release.wait(3)
                return {'action': 'answer'}
            def stream(self, messages):
                return iter([])
        self.service.llm = SlowModel()
        results = []
        worker = threading.Thread(target=lambda: results.append(self.service.submit('default', 'slow', '你好')))
        worker.start()
        self.assertTrue(entered.wait(2))
        other = TurnService(self.temp.name)
        other.submit('default', 'other', '我想学 Python')
        release.set(); worker.join(4)
        self.assertEqual(results[0].status, 'completed')
        self.assertEqual(len(other.repo.snapshot()['goals']), 1)

    def test_recovery(self) -> None:
        tid, _ = self.service.accept('default', 'recover', '你好')
        with transaction(self.service.repo.path) as db:
            db.execute("UPDATE turns SET status='committed',facts='已保存',lease_until=0 WHERE id=?", (tid,))
        TurnService(self.temp.name)
        result = self.service.get(tid)
        self.assertEqual(result.reply, '已保存')
        self.assertTrue(result.degraded)

    def test_empty_legacy_files_do_not_block_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('goals', 'plans', 'reviews'):
                (root / f'{name}.json').write_text('[]')
            (root / 'memory.json').write_text(json.dumps({'short': [], 'long': [], 'fold_count': 0}))
            before = {p.name: p.read_bytes() for p in root.glob('*.json')}
            service = TurnService(root)
            self.assertEqual(service.repo.snapshot()['goals'], [])
            with transaction(service.repo.path, write=False) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM imports').fetchone()[0], 1)
            self.assertEqual(before, {name: (root / name).read_bytes() for name in before})
            service.close()
            reopened = TurnService(root)
            self.assertEqual(reopened.repo.snapshot()['memory']['long'], [])
            reopened.close()

    def test_real_legacy_data_still_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'goals.json').write_text(json.dumps([{'id': 'g', 'title': '保留原目标'}]))
            with self.assertRaises(ServiceError) as error:
                TurnService(root)
            self.assertEqual(error.exception.code, 'MIGRATION_REQUIRED')
            self.assertEqual(error.exception.status, 503)
            self.assertEqual(json.loads((root / 'goals.json').read_text())[0]['title'], '保留原目标')

    def test_empty_legacy_does_not_overwrite_existing_sqlite_data(self) -> None:
        self.service.submit('default', 'goal', '我想学 Python')
        root = Path(self.temp.name)
        (root / 'goals.json').write_text('[]')
        (root / 'memory.json').write_text('{"short":[],"long":[]}')
        reopened = TurnService(root)
        self.assertEqual(len(reopened.repo.snapshot()['goals']), 1)
        reopened.close()

    def test_startup_api_returns_actionable_response(self) -> None:
        import os
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from . import main
        for populated in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'goals.json').write_text(json.dumps([{'id':'g','title':'原目标'}] if populated else []))
                session = main.SessionStore()
                with patch.object(main, 'session', session), patch.dict(os.environ, {main.DATA_DIR_ENV: str(root), 'LLM_API_KEY': ''}):
                    response = TestClient(main.app, raise_server_exceptions=False).get('/health')
                self.assertEqual(response.status_code, 503 if populated else 200)
                if populated:
                    self.assertEqual(response.json()['code'], 'MIGRATION_REQUIRED')
                else:
                    self.assertEqual(response.json()['status'], 'ok')
                    session._service.close()


if __name__ == '__main__':
    unittest.main()
