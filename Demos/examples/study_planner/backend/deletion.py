"""删除意图先落盘，再失效来源及派生产物，恢复旧备份也能重放。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .db import transaction
from .memory_types import ServiceError
from .repositories import Repository, dumps, now_iso, uid


class DeletionService:
    def __init__(self, repo: Repository) -> None:
        self.repo, self.path = repo, repo.root / "deletions.jsonl"

    @contextmanager
    def locked_log(self, create: bool = False) -> Iterator[Any]:
        if not self.path.exists() and not create:
            raise ValueError("删除清单缺失，拒绝开放档案")
        with self.path.open("a+", encoding="utf-8") as handle:
            os.chmod(self.path, 0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def initialize(self) -> None:
        if not self.path.exists():
            with transaction(self.repo.path, write=False) as db:
                if db.execute("SELECT 1 FROM turns LIMIT 1").fetchone() or db.execute("SELECT 1 FROM imports LIMIT 1").fetchone():
                    raise ValueError("已有档案的删除清单缺失，拒绝自动创建空清单")
            with self.locked_log(create=True) as handle:
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(dumps({"version": 1}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    directory = os.open(self.repo.root, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
        self.replay()

    def records(self, handle: Any) -> list[dict]:
        handle.seek(0)
        lines = handle.readlines()
        if not lines or json.loads(lines[0]) != {"version": 1}:
            raise ValueError("删除清单头部损坏")
        result, good_length = [], len(lines[0].encode())
        for index, line in enumerate(lines[1:]):
            if not line.endswith("\n") and index == len(lines) - 2:
                # 恢复旧快照时无法证明尾部是否已经提交，宁可阻断也不能复活内容。
                raise ValueError("删除清单尾部截断，需要核验修复后再开放档案")
            entry = json.loads(line)
            if hashlib.sha256(dumps(entry["intent"]).encode()).hexdigest() != entry["sha256"]:
                raise ValueError("删除清单校验失败")
            result.append(entry["intent"])
            good_length += len(line.encode())
        return result

    def append(self, handle: Any, intent: dict) -> None:
        handle.seek(0, 2)
        handle.write(dumps({"intent": intent, "sha256": hashlib.sha256(dumps(intent).encode()).hexdigest()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def preview(self, memory_id: str) -> dict[str, Any]:
        with transaction(self.repo.path, write=False) as db:
            memory = db.execute("SELECT * FROM memories WHERE id=? AND status!='deleted'", (memory_id,)).fetchone()
            if not memory:
                raise ServiceError("NOT_FOUND", "记忆不存在", 404)
            sources = [r[0] for r in db.execute("SELECT source_id FROM memory_sources WHERE memory_id=?", (memory_id,))]
            return {"id": memory_id, "revision": memory["revision"], "source_ids": sources, "constraint_removed": bool(memory["constraint_id"]),
                    "effect": "清除关联来源和引用它的会话回复、摘要；保留已提交的目标与待办。删除预算后可能需要重新确认。"}

    def delete(self, memory_id: str, expected_revision: int, turn_id: str = "") -> None:
        # 所有删除入口使用同样的锁顺序：清单锁 → 数据库写锁。
        # 本地 fsync 在短事务内完成，避免预览检查与意图落盘间被其他写入修改。
        with self.locked_log() as handle, transaction(self.repo.path) as db:
            self.records(handle)
            row = db.execute("SELECT * FROM memories WHERE id=? AND status!='deleted'", (memory_id,)).fetchone()
            if not row:
                raise ServiceError("NOT_FOUND", "记忆不存在", 404)
            if row["revision"] != expected_revision:
                raise ServiceError("REVISION_CONFLICT", "记忆已更新，请刷新影响预览")
            intent = {"id": uid("del"), "memory_ids": [memory_id], "source_ids": [r[0] for r in db.execute("SELECT source_id FROM memory_sources WHERE memory_id=?", (memory_id,))], "turn_id": turn_id, "time": now_iso()}
            self.append(handle, intent)
            self.apply(db, intent)

    def reset(self, turn_id: str) -> None:
        with self.locked_log() as handle, transaction(self.repo.path) as db:
            self.records(handle)
            intent = {"id": uid("del"), "memory_ids": [r[0] for r in db.execute("SELECT id FROM memories")],
                      "source_ids": [r[0] for r in db.execute("SELECT id FROM messages")], "turn_id": turn_id, "time": now_iso(),
                      "goals": [r[0] for r in db.execute("SELECT id FROM goals")], "reviews": [r[0] for r in db.execute("SELECT day FROM reviews")]}
            self.append(handle, intent)
            self.apply(db, intent)

    def conversation_preview(self, db: sqlite3.Connection, conversation_id: str) -> dict[str, Any]:
        if db.execute("""SELECT 1 FROM turns WHERE conversation_id=?
                         AND status IN ('accepted','analyzing','committed') LIMIT 1""", (conversation_id,)).fetchone():
            raise ServiceError("CONVERSATION_BUSY", "会话正在处理回复，请结束后再删除")
        row = db.execute("""SELECT count(*) AS count,max(seq) AS last_seq FROM messages
                            WHERE conversation_id=? AND deleted=0""", (conversation_id,)).fetchone()
        if not row["count"]:
            raise ServiceError("NOT_FOUND", "会话已删除或不存在", 404)
        related = db.execute("""SELECT DISTINCT m.id,m.content FROM memories m
            JOIN memory_sources s ON s.memory_id=m.id JOIN messages msg ON msg.id=s.source_id
            WHERE msg.conversation_id=? AND msg.deleted=0 AND m.status!='deleted' ORDER BY m.id""", (conversation_id,))
        return {**dict(row), "memories": [dict(item) for item in related],
                "revision": self.repo.profile(db)["state_revision"]}

    def delete_conversation(self, conversation_id: str, last_seq: int, revision: int) -> None:
        with self.locked_log() as handle, transaction(self.repo.path) as db:
            for intent in self.records(handle):
                if (intent.get("conversation"), intent.get("last_seq"), intent.get("revision")) == (conversation_id, last_seq, revision):
                    self.apply(db, intent)
                    return
            preview = self.conversation_preview(db, conversation_id)
            if preview["last_seq"] != last_seq or preview["revision"] != revision:
                raise ServiceError("REVISION_CONFLICT", "会话或档案已更新，请重新查看删除范围")
            intent = {"id": uid("del"), "conversation": conversation_id, "last_seq": last_seq, "revision": revision, "time": now_iso()}
            self.append(handle, intent)
            self.apply(db, intent)

    def _erase_conversation(self, db: sqlite3.Connection, intent: dict) -> None:
        # 用消息序号限定删除范围；恢复备份时也只清除当时确认过的会话内容。
        target = (intent["conversation"], intent["last_seq"])
        db.execute("""UPDATE memory_sources SET excerpt='' WHERE source_id IN
            (SELECT id FROM messages WHERE conversation_id=? AND seq<=?)""", target)
        db.execute("""UPDATE turns SET facts='',reply='',error_code='SOURCE_DELETED',lease_until=0,
            status=CASE WHEN status IN ('accepted','analyzing','committed') THEN 'cancelled' ELSE status END
            WHERE id IN (SELECT turn_id FROM messages WHERE conversation_id=? AND seq<=?)""", target)
        db.execute("UPDATE messages SET deleted=1,content='' WHERE conversation_id=? AND seq<=?", target)
        db.execute("UPDATE summaries SET valid=0,content='' WHERE conversation_id=? AND start_seq<=?", target)

    def apply(self, db: sqlite3.Connection, intent: dict) -> None:
        if db.execute("SELECT 1 FROM deletions WHERE id=?", (intent["id"],)).fetchone():
            return
        if "conversation" in intent:
            self._erase_conversation(db, intent)
        else:
            self._erase_sources(db, intent)
        db.execute("UPDATE profile SET state_revision=state_revision+1,privacy_revision=privacy_revision+1")
        db.execute("UPDATE summaries SET valid=0 WHERE privacy_revision!=(SELECT privacy_revision FROM profile)")
        db.execute("INSERT INTO deletions VALUES (?,?)", (intent["id"], now_iso()))
        if intent.get("turn_id"):
            revision = db.execute("SELECT state_revision FROM profile").fetchone()[0]
            db.execute("UPDATE turns SET status='completed',facts='已清除所选内容',reply='已清除所选内容',state_revision=?,lease_until=0 WHERE id=?", (revision, intent["turn_id"]))

    def _erase_sources(self, db: sqlite3.Connection, intent: dict) -> None:
        visited, pending = set(), list(intent["source_ids"] + intent["memory_ids"])
        while pending:
            source_id = pending.pop()
            if source_id in visited:
                continue
            visited.add(source_id)
            pending.extend(r[0] for r in db.execute("SELECT memory_id FROM memory_sources WHERE source_id=?", (source_id,)))
            for artifact in db.execute("SELECT artifact_type,artifact_id FROM artifact_sources WHERE source_id=?", (source_id,)).fetchall():
                aid = artifact["artifact_id"]
                if artifact["artifact_type"] == "summary":
                    db.execute("UPDATE summaries SET valid=0,content='' WHERE id=?", (aid,))
                elif artifact["artifact_type"] == "turn":
                    db.execute("UPDATE turns SET facts='',reply='',error_code='SOURCE_DELETED' WHERE id=?", (aid,))
                pending.append(aid)
            memory = db.execute("SELECT constraint_id FROM memories WHERE id=?", (source_id,)).fetchone()
            if memory and memory[0]:
                db.execute("UPDATE constraints SET status='deleted',revision=revision+1 WHERE id=?", (memory[0],))
                db.execute("UPDATE profile SET budget_status='needs_confirmation'")
            db.execute("UPDATE memories SET status='deleted',content='',revision=revision+1 WHERE id=?", (source_id,))
            db.execute("DELETE FROM memory_terms WHERE memory_id=?", (source_id,))
            db.execute("UPDATE memory_sources SET excerpt='' WHERE memory_id=? OR source_id=?", (source_id, source_id))
            db.execute("UPDATE messages SET deleted=1,content='' WHERE id=?", (source_id,))
            # 原始输入被删除时，该轮结果也不能留作缓存副本。
            original = db.execute("SELECT turn_id FROM messages WHERE id=?", (source_id,)).fetchone()
            if original:
                pending.append(original[0])
            if db.execute("SELECT 1 FROM turns WHERE id=?", (source_id,)).fetchone():
                db.execute("UPDATE turns SET facts='',reply='',error_code='SOURCE_DELETED',status=CASE WHEN status IN ('accepted','analyzing') THEN 'cancelled' ELSE status END WHERE id=?", (source_id,))
                pending.extend(r[0] for r in db.execute("SELECT id FROM messages WHERE turn_id=?", (source_id,)))
        for gid in intent.get("goals", []):
            db.execute("DELETE FROM plan_items WHERE goal_id=?", (gid,))
            db.execute("DELETE FROM goals WHERE id=?", (gid,))
        for day in intent.get("reviews", []):
            db.execute("DELETE FROM reviews WHERE day=?", (day,))

    def replay(self) -> None:
        with self.locked_log() as handle, transaction(self.repo.path) as db:
            intents = self.records(handle)
            applied = {r[0] for r in db.execute("SELECT id FROM deletions")}
            if not applied <= {entry["id"] for entry in intents}:
                raise ValueError("删除清单不完整，拒绝开放档案")
            for intent in intents:
                self.apply(db, intent)
