"""把明确的时间预算与模型推测分开，并保存每条结论的证据。"""
from __future__ import annotations

import re
import sqlite3
from datetime import date
from typing import Any

from .memory import tokenize
from .memory_types import MemoryCandidate, ServiceError
from .repositories import now_iso, uid


def budget_candidate(text: str, day: date, source_id: str) -> MemoryCandidate | None:
    duration = re.search(r"(\d+(?:\.\d+)?)\s*(分钟|小时)", text)
    if not duration:
        return None
    value = round(float(duration[1]) * (60 if duration[2] == "小时" else 1))
    uncertain = re.search(r"不是|不想|不能|没有|朋友|他说|她说|比如|例如|假设|如果|可能|也许|建议|[？?‘’“”\"']|周[一二三四五六日天]", text)
    scope = f"date:{day.isoformat()}" if "今天" in text else "weekend" if "周末" in text else "weekday" if "工作日" in text else "default"
    explicit = re.search(r"每天|每日|一天|今天|周末|工作日", text) and re.search(r"只有|有|能|可|时间|学习|学|投入", text)
    active = bool(explicit and not uncertain and 15 <= value <= 480)
    return MemoryCandidate(f"可学习时间：{value} 分钟", "preference", "active" if active else "pending", "fact", source_id, text, scope, value)


def resolve_daily_minutes(db: sqlite3.Connection, day: date) -> int:
    if db.execute("SELECT budget_status FROM profile").fetchone()[0] == "needs_confirmation":
        raise ServiceError("BUDGET_CONFIRMATION_REQUIRED", "请先确认待迁移或已删除的学习时间预算", 422)
    for scope in (f"date:{day.isoformat()}", "weekend" if day.weekday() >= 5 else "weekday", "default"):
        row = db.execute("SELECT value FROM constraints WHERE key='learning.daily_minutes' AND scope=? AND status='active' AND effective_from<=? AND (effective_to IS NULL OR effective_to>=?)", (scope, day.isoformat(), day.isoformat())).fetchone()
        if row:
            return row[0]
    return 60


def store_candidate(db: sqlite3.Connection, candidate: MemoryCandidate) -> str:
    if not candidate.source_id or not candidate.excerpt:
        raise ValueError("记忆必须有可定位证据")
    old = db.execute("SELECT m.id FROM memories m JOIN memory_sources s ON s.memory_id=m.id WHERE s.source_id=? AND m.content=? AND m.status!='deleted'", (candidate.source_id, candidate.content)).fetchone()
    if old:
        return old[0]
    mid, constraint_id = uid(), None
    if candidate.kind == "fact" and candidate.status == "active":
        if candidate.scope not in ("default", "weekday", "weekend"):
            if not candidate.scope.startswith("date:"):
                raise ValueError("未知预算作用域")
            date.fromisoformat(candidate.scope[5:])
        if candidate.value is None or not 15 <= candidate.value <= 480:
            raise ValueError("预算必须为 15–480 分钟")
        previous = db.execute("SELECT id FROM constraints WHERE key='learning.daily_minutes' AND scope=? AND status='active'", (candidate.scope,)).fetchone()
        if previous:
            db.execute("UPDATE constraints SET status='superseded',revision=revision+1 WHERE id=?", (previous[0],))
            db.execute("UPDATE memories SET status='superseded',superseded_by=?,revision=revision+1 WHERE constraint_id=?", (mid, previous[0]))
        constraint_id = uid("c")
        source = candidate.source_id if db.execute("SELECT 1 FROM messages WHERE id=?", (candidate.source_id,)).fetchone() else None
        effective_from = candidate.scope[5:] if candidate.scope.startswith("date:") else "0001-01-01"
        effective_to = candidate.scope[5:] if candidate.scope.startswith("date:") else None
        db.execute("INSERT INTO constraints(id,key,scope,value,effective_from,effective_to,source_message_id) VALUES (?,'learning.daily_minutes',?,?,?,?,?)", (constraint_id, candidate.scope, candidate.value, effective_from, effective_to, source))
        if candidate.scope == "default":
            db.execute("UPDATE profile SET budget_status='confirmed'")
    db.execute("INSERT INTO memories(id,kind,category,content,status,constraint_id,created_at,confirmed_at,effective_to,source_type) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (mid, candidate.kind, candidate.category, candidate.content, candidate.status, constraint_id, now_iso(), now_iso() if candidate.status == "active" else None,
                candidate.scope[5:] if candidate.scope.startswith("date:") else None, "user" if candidate.status == "active" else "candidate"))
    db.execute("INSERT INTO memory_sources VALUES (?,?,?,?)", (mid, candidate.source_id, candidate.excerpt, "evidence-v1"))
    db.executemany("INSERT OR IGNORE INTO memory_terms VALUES (?,?)", [(mid, term) for term in tokenize(candidate.content)])
    return mid


def confirm_memory(db: sqlite3.Connection, memory_id: str, expected_revision: int, content: str | None = None,
                   scope: str | None = None, value: int | None = None, day: date | None = None) -> str:
    row = db.execute("SELECT * FROM memories WHERE id=? AND status NOT IN ('deleted','superseded')", (memory_id,)).fetchone()
    if row is None:
        raise ServiceError("NOT_FOUND", "记忆不存在或已失效", 404)
    if row["revision"] != expected_revision:
        raise ServiceError("REVISION_CONFLICT", "记忆已被更新")
    source = db.execute("SELECT * FROM memory_sources WHERE memory_id=? LIMIT 1", (memory_id,)).fetchone()
    candidate = budget_candidate(content or row["content"], day or date.today(), uid("confirmation"))
    if row["constraint_id"] and value is None:
        constraint = db.execute("SELECT value,scope FROM constraints WHERE id=?", (row["constraint_id"],)).fetchone()
        value, scope = constraint["value"], scope or constraint["scope"]
    if value is not None:
        candidate = MemoryCandidate(f"可学习时间：{value} 分钟", "preference", "active", "fact", uid("confirmation"), content or row["content"], scope or "default", value)
    elif candidate:
        candidate.status = "active"
        candidate.scope = scope or candidate.scope
    else:
        candidate = MemoryCandidate(content or row["content"], row["category"], "active", "experience", uid("confirmation"), content or row["content"])
    # 确认是新的独立用户证据，保留原记录用于追溯。
    new_id = store_candidate(db, candidate)
    db.execute("UPDATE memories SET status='superseded',superseded_by=?,revision=revision+1 WHERE id=?", (new_id, memory_id))
    db.execute("UPDATE profile SET state_revision=state_revision+1")
    return new_id
