"""验证目标删除、请求重试和恢复重放不会影响其他目标及历史记忆。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .db import transaction
from .repositories import Repository
from .main import session
from .selftest_api import build_client


class GoalDeletionTests(unittest.TestCase):
    def test_delete_goal_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = build_client(Path(tmp))
            try:
                first = client.post('/api/goals', json={'title': 'Python', 'daily_minutes': 30}).json()
                goal_id = first['goals'][0]['id']
                before = client.post('/api/goals', json={'title': 'Rust'}).json()
                service = session.service
                with transaction(service.repo.path, write=False) as db:
                    domain = service.repo.load_domain(db)
                    history = [tuple(r) for r in db.execute('SELECT id,content,deleted FROM messages')]
                payload = {'request_id': 'delete-goal'}
                response = client.request('DELETE', f'/api/goals/{goal_id}', json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                after = response.json()
                self.assertEqual([g['title'] for g in after['goals']], ['Rust'])
                self.assertTrue(any(i['goal_id'] == goal_id for i in before['items']))
                self.assertEqual(after['items'], [i for i in before['items'] if i['goal_id'] != goal_id])
                self.assertEqual(after['memory']['long'], before['memory']['long'])
                self.assertEqual(after['memory']['short'], before['memory']['short'])
                with transaction(service.repo.path, write=False) as db:
                    self.assertEqual(history, [tuple(r) for r in db.execute(
                        'SELECT id,content,deleted FROM messages')])
                retry = client.request('DELETE', f'/api/goals/{goal_id}', json=payload)
                self.assertEqual(retry.status_code, 200)
                self.assertEqual(retry.json()['state_revision'], after['state_revision'])
                missing = client.request('DELETE', f'/api/goals/{goal_id}', json={})
                self.assertEqual(missing.status_code, 404)
                # 旧版面板生成的消息在重新打开档案后也应从聊天记录中消失。
                with transaction(service.repo.path) as db:
                    turn = db.execute("SELECT id FROM turns WHERE request_id='delete-goal'").fetchone()[0]
                    db.execute("""INSERT INTO messages VALUES
                        ('old-delete','default',999,?,'user','删除目标及关联待办','','',0)""", (turn,))
                Repository(service.repo.root)
                visible = client.get('/api/conversations/default/messages').json()['messages']
                self.assertEqual([row['id'] for row in visible], [row[0] for row in history])
                # 模拟恢复删除前的业务快照和删除应用记录，外部清单应再次清理目标。
                with transaction(service.repo.path) as db:
                    service.repo.save_domain(db, *domain)
                    db.execute('DELETE FROM deletions')
                service.deletion.replay()
                restored = client.get('/api/state').json()
                self.assertEqual(restored['goals'], after['goals'])
                self.assertEqual(restored['items'], after['items'])
            finally:
                session.service.close()


if __name__ == '__main__':
    unittest.main()
