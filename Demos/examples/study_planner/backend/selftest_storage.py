"""用真实数据库和进程退出验证提交、约束与恢复边界。"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .db import open_database, transaction


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "planner.db"
        open_database(self.path).close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_atomicity_and_foreign_keys(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError), transaction(self.path) as db:
            db.execute("INSERT INTO goals VALUES ('g', '{}')")
            db.execute("INSERT INTO plan_items VALUES ('p', 'missing', '2026-11-04', 0, '{}')")
        with transaction(self.path, write=False) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM goals").fetchone()[0], 0)

    def test_schema_and_unique_constraint(self) -> None:
        open_database(self.path).close()
        with transaction(self.path) as db:
            db.execute("INSERT INTO reviews VALUES ('2026-11-04', '{}')")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO reviews VALUES ('2026-11-04', '{}')")
        with transaction(self.path) as db:
            db.execute("PRAGMA user_version=999")
        with self.assertRaises(ValueError):
            open_database(self.path)

    def test_corruption_visible(self) -> None:
        bad = self.path.with_name("bad.db")
        bad.write_bytes(b"broken database")
        with self.assertRaises(sqlite3.DatabaseError):
            open_database(bad)

    def test_process_crash(self) -> None:
        for commit in (False, True):
            code = """
import os, sqlite3, sys
c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN IMMEDIATE')
c.execute("INSERT OR REPLACE INTO goals VALUES ('crash', '{}')")
if sys.argv[2]=='yes': c.commit()
os._exit(9)
"""
            result = subprocess.run([sys.executable, "-c", code, str(self.path), "yes" if commit else "no"])
            self.assertEqual(result.returncode, 9)
            with transaction(self.path, write=False) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM goals").fetchone()[0], int(commit))

    def test_import_validation_and_idempotency(self) -> None:
        import json
        from .migrate_data import migrate
        source = Path(self.temp.name) / "legacy"
        source.mkdir()
        (source / "goals.json").write_text(json.dumps([{"id": "g", "title": "Python"}]))
        (source / "memory.json").write_text(json.dumps({"long": [{"id": "m", "content": "每天30分钟"}]}))
        target = Path(self.temp.name) / "imported"
        self.assertEqual(migrate(source, target)["memories"], 1)
        self.assertFalse(target.exists())
        before = (source / "memory.json").read_bytes()
        self.assertTrue(migrate(source, target, False)["pending_budget"])
        self.assertTrue(migrate(source, target, False)["already_imported"])
        self.assertEqual(before, (source / "memory.json").read_bytes())
        with transaction(target / "planner.db", write=False) as db:
            self.assertEqual(db.execute("SELECT status FROM memories").fetchone()[0], "pending")
        (source / "plans.json").write_text('[{"id":"p","goal_id":"missing","day":"2026-11-04"}]')
        with self.assertRaises(ValueError):
            migrate(source, target)
        (source / "memory.json").write_text("{broken")
        with self.assertRaises(ValueError):
            migrate(source, target)


if __name__ == "__main__":
    unittest.main()
