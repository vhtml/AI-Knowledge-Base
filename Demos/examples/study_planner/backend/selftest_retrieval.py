"""验证失效过滤、删除传播及删除意图在崩溃后恢复。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from .db import transaction
from .memory_policy import resolve_daily_minutes, store_candidate
from .memory_types import MemoryCandidate, ServiceError
from .turn_service import TurnService


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.service = TurnService(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_filtering(self) -> None:
        with transaction(self.service.repo.path) as db:
            ids = [store_candidate(db, MemoryCandidate('算法练习' + str(n), status=status, source_id=f's-{n}', excerpt='原始证据')) for n, status in enumerate(('active','pending','superseded','expired'))]
        result = self.service.retrieval.search('算法练习')
        self.assertEqual([r['id'] for r in result], ids[:1])
        self.assertEqual(self.service.retrieval.search('无匹配指纹xyz'), [])

    def test_delete_clears_constraint_and_original_reply(self) -> None:
        result = self.service.submit('default', 'budget', '我每天只有30分钟')
        memory = self.service.repo.snapshot()['memory']['long'][0]
        self.service.submit('default', 'delete', '清除记忆', command='memory_delete', payload={'id': memory['id'], 'expected_revision': memory['revision']})
        self.assertFalse(self.service.retrieval.search('学习时间'))
        self.assertEqual(self.service.get(result.turn_id).reply, '')
        with transaction(self.service.repo.path, write=False) as db:
            self.assertFalse(db.execute("SELECT content FROM messages WHERE deleted=0 AND content LIKE '%30%'").fetchone())
            with self.assertRaises(ServiceError):
                resolve_daily_minutes(db, date.today())
        TurnService(self.temp.name)
        self.assertEqual(self.service.repo.snapshot()['memory']['long'], [])

    def test_intent_survives_database_rollback(self) -> None:
        self.service.submit('default', 'budget', '我每天只有30分钟')
        memory = self.service.repo.snapshot()['memory']['long'][0]
        with patch.object(self.service.deletion, 'apply', side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):
                self.service.deletion.delete(memory['id'], memory['revision'])
        recovered = TurnService(self.temp.name)
        self.assertFalse(recovered.repo.snapshot()['memory']['long'])

    def test_fsync_failure_does_not_delete(self) -> None:
        self.service.submit('default', 'budget', '我每天只有30分钟')
        memory = self.service.repo.snapshot()['memory']['long'][0]
        with patch.object(self.service.deletion, 'append', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.service.deletion.delete(memory['id'], memory['revision'])
        self.assertTrue(self.service.repo.snapshot()['memory']['long'])

    def test_confirmation_revision(self) -> None:
        self.service.submit('default', 'uncertain', '我可能每天只有30分钟')
        memory = self.service.repo.snapshot()['memory']['long'][0]
        self.service.submit('default', 'confirm', '确认学习预算', command='memory_confirm', payload={'id': memory['id'], 'expected_revision': 1, 'value': 30, 'scope': 'default'})
        with transaction(self.service.repo.path, write=False) as db:
            self.assertEqual(resolve_daily_minutes(db, date.today()), 30)
        with self.assertRaises(ServiceError):
            self.service.submit('default', 'old', '再次确认', command='memory_confirm', payload={'id': memory['id'], 'expected_revision': 1})


if __name__ == '__main__':
    unittest.main()
