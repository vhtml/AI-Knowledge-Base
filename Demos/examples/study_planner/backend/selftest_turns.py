"""验证重试、取消和模型等待不占有数据库锁。"""
from __future__ import annotations

import tempfile
import threading
import unittest
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


if __name__ == '__main__':
    unittest.main()
