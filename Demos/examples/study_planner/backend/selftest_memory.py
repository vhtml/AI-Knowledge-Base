"""验证事实范围、来源可信度与模型输入的边界。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from .db import transaction
from .memory_policy import budget_candidate, resolve_daily_minutes
from .turn_service import TurnService
from .context import Clock, ContextBuilder, count_tokens
from .memory_types import ServiceError
from datetime import datetime, timezone


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.service = TurnService(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_budgets(self) -> None:
        self.service.submit('default', 'b1', '我每天只有30分钟')
        self.service.submit('default', 'b2', '周末能学2小时')
        with transaction(self.service.repo.path, write=False) as db:
            self.assertEqual(resolve_daily_minutes(db, date(2026, 11, 4)), 30)
            self.assertEqual(resolve_daily_minutes(db, date(2026, 11, 7)), 120)
        self.service.submit('default', 'b3', '我每天可学习60分钟')
        with transaction(self.service.repo.path, write=False) as db:
            self.assertEqual(resolve_daily_minutes(db, date(2026, 11, 4)), 60)
            self.assertEqual(db.execute("SELECT count(*) FROM constraints WHERE scope='default' AND status='active'").fetchone()[0], 1)

    def test_uncertain_does_not_become_constraint(self) -> None:
        for text in ('朋友每天学30分钟', '我不是每天能学30分钟', '假设每天能学30分钟', '我可能每天只有30分钟', '周一只能学30分钟'):
            self.assertNotEqual(budget_candidate(text, date(2026,11,4), 'source').status, 'active')
        self.service.submit('default', 'uncertain', '我可能每天只有30分钟')
        with transaction(self.service.repo.path, write=False) as db:
            self.assertEqual(resolve_daily_minutes(db, date.today()), 60)
            self.assertEqual(db.execute('SELECT status FROM memories').fetchone()[0], 'pending')

    def test_daily_scope(self) -> None:
        candidate = budget_candidate('今天只有15分钟', date(2026,11,4), 's')
        self.assertEqual(candidate.scope, 'date:2026-11-04')
        self.assertEqual(candidate.value, 15)

    def test_context_budget_and_pending_history(self) -> None:
        self.service.submit('default', 'done', '我想学 Python')
        self.service.accept('default', 'other', '不应进入其他轮次的待处理秘密')
        tid, _ = self.service.accept('default', 'current', '接下来呢')
        context = self.service.context.build(tid, 'decision')
        self.assertLessEqual(count_tokens(context.messages), 6000)
        self.assertIn('Python', str(context.messages))
        self.assertNotIn('待处理秘密', str(context.messages))
        long_id, _ = self.service.accept('default', 'large', '长' * 3000)
        with self.assertRaises(ServiceError):
            self.service.context.build(long_id, 'decision')
        with self.assertRaises(ValueError):
            ContextBuilder(self.service.repo, input_budget=7000)

    def test_summary_preserves_raw_history_and_clock(self) -> None:
        for index in range(8):
            self.service.submit('default', f'history-{index}', '随手记录' + ('内容' * 200))
        with transaction(self.service.repo.path, write=False) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0], 16)
            self.assertGreater(db.execute('SELECT count(*) FROM summaries WHERE valid=1').fetchone()[0], 0)
        clock = Clock(lambda: datetime(2030, 1, 1, 20, tzinfo=timezone.utc))
        self.assertEqual(clock.day('Asia/Shanghai'), date(2030,1,2))


if __name__ == '__main__':
    unittest.main()
