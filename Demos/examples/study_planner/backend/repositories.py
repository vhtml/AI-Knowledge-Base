"""把领域状态和记忆统一读写到同一个事务，界面只获得有界视图。"""
from __future__ import annotations

import json
import os
from zoneinfo import ZoneInfo
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import open_database, transaction
from .models import Goal, PlanItem, Review


def uid(prefix: str = "m") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Repository:
    def __init__(self, root: Path | str, initialize: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.path = self.root / "planner.db"
        if not initialize:
            return
        fresh = not self.path.exists()
        timezone_name = os.getenv("STUDY_PLANNER_TIMEZONE", "Asia/Shanghai")
        if fresh:
            ZoneInfo(timezone_name)
        db = open_database(self.path)
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS reply_steps (
                message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                steps TEXT NOT NULL)""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS clear_reply_steps
                AFTER UPDATE OF deleted ON messages WHEN NEW.deleted=1
                BEGIN DELETE FROM reply_steps WHERE message_id=NEW.id; END""")
            # 兼容旧版本：按操作类型隐藏面板生成的消息，保留用户输入的同名文本。
            db.execute("""UPDATE messages SET deleted=1 WHERE deleted=0
                          AND turn_id IN (SELECT id FROM turns WHERE command='goal_delete')""")
            if fresh:
                db.execute("UPDATE profile SET timezone=?", (timezone_name,))
        finally:
            db.close()

    def profile(self, db: sqlite3.Connection) -> dict[str, Any]:
        return dict(db.execute("SELECT * FROM profile WHERE id=1").fetchone())

    def load_domain(self, db: sqlite3.Connection) -> tuple[list[Goal], list[PlanItem], list[Review]]:
        return (
            [Goal.from_dict(json.loads(r[0])) for r in db.execute("SELECT payload FROM goals ORDER BY rowid")],
            [PlanItem.from_dict(json.loads(r[0])) for r in db.execute("SELECT payload FROM plan_items ORDER BY day,rowid")],
            [Review.from_dict(json.loads(r[0])) for r in db.execute("SELECT payload FROM reviews ORDER BY day")],
        )

    def save_domain(self, db: sqlite3.Connection, goals: list[Goal], items: list[PlanItem], reviews: list[Review]) -> None:
        db.execute("DELETE FROM plan_items")
        db.execute("DELETE FROM goals")
        db.execute("DELETE FROM reviews")
        db.executemany("INSERT INTO goals VALUES (?,?)", [(g.id, dumps(g.to_dict())) for g in goals])
        db.executemany("INSERT INTO plan_items VALUES (?,?,?,?,?)", [(i.id, i.goal_id, i.day, int(i.done), dumps(i.to_dict())) for i in items])
        db.executemany("INSERT OR REPLACE INTO reviews VALUES (?,?)", [(r.day, dumps(r.to_dict())) for r in reviews])

    def memories(self, db: sqlite3.Connection, status: str | None = None, offset: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        rows = db.execute("SELECT * FROM memories WHERE status != 'deleted' AND (? IS NULL OR status=?) ORDER BY created_at DESC,id LIMIT ? OFFSET ?", (status, status, min(limit, 100), offset))
        result = []
        for row in rows:
            item = dict(row)
            item.update(scope="long", superseded=item["status"] == "superseded")
            if item["constraint_id"]:
                constraint = db.execute("SELECT scope,value FROM constraints WHERE id=?", (item["constraint_id"],)).fetchone()
                item.update(budget_scope=constraint["scope"], budget_value=constraint["value"])
            item["sources"] = [dict(s) for s in db.execute("""
                SELECT s.source_id,s.excerpt,m.conversation_id,m.deleted
                FROM memory_sources s LEFT JOIN messages m ON m.id=s.source_id
                WHERE s.memory_id=?
            """, (item["id"],))]
            result.append(item)
        return result

    def snapshot(self, conversation_id: str = "default") -> dict[str, Any]:
        with transaction(self.path, write=False) as db:
            goals, items, reviews = self.load_domain(db)
            profile = self.profile(db)
            rows = db.execute("SELECT id,content,role,day FROM messages WHERE conversation_id=? AND deleted=0 ORDER BY seq DESC LIMIT 12", (conversation_id,)).fetchall()
            short = [{**dict(r), "scope": "short", "category": "note"} for r in reversed(rows)]
            counts = {r[0]: r[1] for r in db.execute("SELECT status,count(*) FROM memories WHERE status!='deleted' GROUP BY status")}
            return {"goals": [g.to_dict() for g in goals], "items": [i.to_dict() for i in items], "reviews": [r.to_dict() for r in reviews],
                    "state_revision": profile["state_revision"], "privacy_revision": profile["privacy_revision"], "budget_status": profile["budget_status"], "timezone": profile["timezone"],
                    "memory": {"short": short, "long": self.memories(db), "has_more": sum(counts.values()) > 50,
                               "stats": {"short": len(short), "long": sum(counts.values()), "long_active": counts.get("active", 0), "pending": counts.get("pending", 0), "long_superseded": counts.get("superseded", 0)}}}

    def export(self) -> dict[str, Any]:
        with transaction(self.path, write=False) as db:
            goals, items, reviews = self.load_domain(db)
            memories = []
            offset = 0
            while True:
                batch = self.memories(db, offset=offset, limit=100)
                memories.extend(batch)
                if len(batch) < 100:
                    break
                offset += 100
            return {"format": "study-planner-export-v1", "exported_at": now_iso(), "profile": self.profile(db),
                    "goals": [g.to_dict() for g in goals], "items": [i.to_dict() for i in items], "reviews": [r.to_dict() for r in reviews],
                    "memories": memories, "messages": [dict(r) for r in db.execute("SELECT id,conversation_id,seq,role,content,created_at FROM messages WHERE deleted=0 ORDER BY conversation_id,seq")]}
