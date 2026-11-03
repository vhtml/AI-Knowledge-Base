"""用持久租约整理原始会话，派生失败不阻断用户写入。"""
from __future__ import annotations

import json
import time
from typing import Any

from .context import Clock, count_tokens
from .db import transaction
from .repositories import Repository, dumps, uid


class Maintenance:
    def __init__(self, repo: Repository, clock: Clock | None = None) -> None:
        self.repo, self.clock = repo, clock or Clock()

    def schedule(self, conversation_id: str) -> None:
        with transaction(self.repo.path) as db:
            profile = self.repo.profile(db)
            day = self.clock.day(profile["timezone"]).isoformat()
            db.execute("UPDATE profile SET maintenance_day=?", (day,))
            db.execute("UPDATE memories SET status='expired',revision=revision+1 WHERE status='active' AND effective_to<?", (day,))
            db.execute("UPDATE constraints SET status='expired',revision=revision+1 WHERE status='active' AND effective_to<?", (day,))
            rows = db.execute("SELECT m.seq,m.content,m.turn_id FROM messages m JOIN turns t ON t.id=m.turn_id WHERE m.conversation_id=? AND m.deleted=0 AND t.status='completed' ORDER BY m.seq DESC LIMIT 128", (conversation_id,)).fetchall()
            turn_ids = list(dict.fromkeys(r["turn_id"] for r in rows))
            if len(turn_ids) <= 4 or count_tokens([{"content": r["content"]} for r in rows]) < 4800:
                return
            older = [r for r in rows if r["turn_id"] not in turn_ids[:4]]
            end = max(r["seq"] for r in older)
            payload = {"conversation_id": conversation_id, "end": end, "privacy_revision": profile["privacy_revision"]}
            key = f"summary:{conversation_id}:{end}:{profile['privacy_revision']}"
            db.execute("INSERT OR IGNORE INTO jobs(id,job_key,kind,payload) VALUES (?,?,'summary',?)", (uid("job"), key, dumps(payload)))

    def run_once(self) -> bool:
        stamp, token = time.time(), time.time() + 60
        with transaction(self.repo.path) as db:
            row = db.execute("SELECT * FROM jobs WHERE attempts<3 AND next_run<=? AND (status='queued' OR (status='running' AND lease_until<?)) ORDER BY rowid LIMIT 1", (stamp, stamp)).fetchone()
            if not row:
                return False
            db.execute("UPDATE jobs SET status='running',lease_until=?,attempts=attempts+1 WHERE id=?", (token, row["id"]))
        try:
            payload = json.loads(row["payload"])
            with transaction(self.repo.path, write=False) as db:
                sources = db.execute("SELECT m.* FROM messages m JOIN turns t ON t.id=m.turn_id WHERE m.conversation_id=? AND m.seq<=? AND m.deleted=0 AND t.status='completed' ORDER BY m.seq DESC LIMIT 128", (payload["conversation_id"], payload["end"])).fetchall()
            # 提取式摘要保留原话的首句；不对旧摘要再次改写。
            lines, source_ids = [], []
            for source in sources:
                piece = f"{source['role']}：{source['content'].splitlines()[0][:100]}"
                proposed = "\n".join([piece] + lines)
                if len(proposed.encode()) > 800:
                    continue
                lines.insert(0, piece)
                source_ids.append(source["id"])
            content = "\n".join(lines)
            with transaction(self.repo.path) as db:
                profile = self.repo.profile(db)
                owned = db.execute("SELECT status,lease_until FROM jobs WHERE id=?", (row["id"],)).fetchone()
                if owned["status"] != "running" or owned["lease_until"] != token:
                    return False
                if profile["privacy_revision"] == payload["privacy_revision"] and content and len(content.encode()) < sum(len(s["content"].encode()) for s in sources):
                    summary_id = uid("sum")
                    db.execute("UPDATE summaries SET valid=0 WHERE conversation_id=?", (payload["conversation_id"],))
                    db.execute("INSERT INTO summaries VALUES (?,?,?,?,?,?,?,?,1)", (summary_id, payload["conversation_id"], min(s["seq"] for s in sources), payload["end"], profile["privacy_revision"], content, len(content.encode()), "extractive-v1"))
                    db.executemany("INSERT INTO artifact_sources VALUES ('summary',?,?)", [(summary_id, sid) for sid in source_ids])
                db.execute("UPDATE jobs SET status='succeeded',lease_until=0 WHERE id=?", (row["id"],))
        except Exception as exc:
            with transaction(self.repo.path) as db:
                db.execute("UPDATE jobs SET status=CASE WHEN attempts>=3 THEN 'failed' ELSE 'queued' END,next_run=?,error_code=? WHERE id=? AND lease_until=?", (stamp + 2 ** row["attempts"], type(exc).__name__, row["id"], token))
        return True
