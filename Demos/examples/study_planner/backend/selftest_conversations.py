"""验证会话来源定位、历史删除范围，以及恢复旧备份后的可见性。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .backup import backup_repository, restore_repository
from .db import transaction
from .main import session
from .selftest_api import build_client
from .turn_service import TurnService


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.client = build_client(self.root)
        self.service = session.service
        self.addCleanup(self.service.close)
        self.path = '/api/conversations/old'
        self.service.submit('old', 'budget', '我每天只有30分钟')

    def test_source_and_delete_survive_restore(self) -> None:
        self.service.submit('old', 'goal', '我想学 Python 数据分析')
        self.service.submit('other', 'plan', '生成计划')
        original = self.service.repo.snapshot('old')
        memory = next(m for m in original['memory']['long'] if m['constraint_id'])
        source = memory['sources'][0]
        self.assertEqual(source['conversation_id'], 'old')
        link = {'source': source['source_id'], 'limit': 1}
        rows = self.client.get(self.path + '/messages', params=link).json()['messages']
        self.assertEqual([m['id'] for m in rows], [source['source_id']])
        self.assertEqual(self.client.get('/api/conversations/other/messages', params=link).status_code, 404)
        backup = self.root / 'before.db'
        backup_repository(self.service.repo, backup)
        preview = self.client.get(self.path + '/deletion-preview').json()
        self.assertIn(memory['id'], [m['id'] for m in preview['memories']])
        self.assertEqual(self.client.request('DELETE', self.path, json=preview).status_code, 200)
        self.assertEqual(self.client.request('DELETE', self.path, json=preview).status_code, 200)
        self.assertEqual(self.client.get(self.path + '/messages', params=link).status_code, 404)
        self.assertEqual([c['id'] for c in self.client.get('/api/conversations').json()], ['other'])
        self.service.close()
        restore_repository(self.root, backup, dry_run=False)
        restored = TurnService(self.root)
        self.addCleanup(restored.close)
        state = restored.repo.snapshot('old')
        self.assertEqual(state['memory']['short'], [])
        self.assertEqual(state['goals'], original['goals'])
        self.assertEqual(state['items'], original['items'])
        self.assertTrue(state['items'])
        kept = next(m for m in state['memory']['long'] if m['id'] == memory['id'])
        self.assertEqual((kept['status'], kept['budget_value']), ('active', 30))
        self.assertEqual((kept['sources'][0]['excerpt'], kept['sources'][0]['deleted']), ('', 1))
        with transaction(restored.repo.path, write=False) as db:
            self.assertFalse(db.execute("SELECT 1 FROM turns WHERE conversation_id='old' AND (facts!='' OR reply!='')").fetchone())
        self.assertTrue(restored.repo.snapshot('other')['memory']['short'])

    def test_stale_preview_and_active_turn(self) -> None:
        preview = self.client.get(self.path + '/deletion-preview').json()
        turn, _ = self.service.accept('old', 'pending', '查询今日任务')
        self.assertEqual(self.client.request('DELETE', self.path, json=preview).json()['code'], 'CONVERSATION_BUSY')
        self.service.cancel(turn)
        self.assertEqual(self.client.request('DELETE', self.path, json=preview).json()['code'], 'REVISION_CONFLICT')
        self.assertTrue(self.service.repo.snapshot('old')['memory']['short'])


if __name__ == '__main__':
    unittest.main()
