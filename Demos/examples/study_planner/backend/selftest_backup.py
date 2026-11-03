"""实际演练备份、离线恢复和删除清单重放。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .backup import backup_repository, restore_repository
from .turn_service import TurnService


class BackupTests(unittest.TestCase):
    def test_restore_replays_new_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TurnService(root)
            service.submit('default', 'b', '我每天只有30分钟')
            backup = root / 'snapshot.db'
            backup_repository(service.repo, backup)
            memory = service.repo.snapshot()['memory']['long'][0]
            service.deletion.delete(memory['id'], memory['revision'])
            with self.assertRaises(ValueError):
                restore_repository(root, backup)
            service.close()
            before = service.repo.path.read_bytes()
            self.assertEqual(restore_repository(root, backup, True)['integrity'], 'ok')
            self.assertEqual(before, service.repo.path.read_bytes())
            restore_repository(root, backup, False)
            restored = TurnService(root)
            self.assertFalse(restored.repo.snapshot()['memory']['long'])
            self.assertFalse(restored.retrieval.search('学习时间'))
            restored.close()

    def test_corrupt_backup_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TurnService(root)
            backup = root / 'snapshot.db'
            backup_repository(service.repo, backup)
            service.close()
            backup.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                restore_repository(root, backup)

    def test_missing_deletion_log_blocks_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TurnService(root)
            backup = root / 'snapshot.db'
            backup_repository(service.repo, backup)
            service.close()
            (root / 'deletions.jsonl').unlink()
            with self.assertRaises(ValueError):
                restore_repository(root, backup)

    def test_corrupted_current_database_can_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TurnService(root)
            service.submit('default', 'g', '我想学 Python')
            backup = root / 'snapshot.db'
            backup_repository(service.repo, backup)
            service.close()
            (root / 'planner.db').write_bytes(b'corrupt')
            restore_repository(root, backup, False)
            restored = TurnService(root)
            self.assertEqual(len(restored.repo.snapshot()['goals']), 1)
            restored.close()

    def test_truncated_committed_intent_blocks_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TurnService(root)
            service.submit('default', 'b', '我每天只有30分钟')
            backup = root / 'snapshot.db'
            backup_repository(service.repo, backup)
            memory = service.repo.snapshot()['memory']['long'][0]
            service.deletion.delete(memory['id'], memory['revision'])
            service.close()
            log = root / 'deletions.jsonl'
            raw = log.read_bytes()
            log.write_bytes(raw[:-20])
            with self.assertRaises(ValueError):
                restore_repository(root, backup, False)


if __name__ == '__main__':
    unittest.main()
