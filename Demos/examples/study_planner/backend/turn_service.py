"""将分析与短事务提交分开，使断线、并发和重试都有可查询结果。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta

from .agent import StepRecord, StudyPlanner, Turn, detect_intent
from .db import transaction
from .llm import ACT_ALL, DECIDE_SYSTEM_PROMPT, REPLY_SYSTEM_PROMPT
from .memory import MemoryManager
from .memory_types import OperationResult, ServiceError
from .models import MemoryItem
from .memory_types import MemoryCandidate
from .memory_policy import budget_candidate, resolve_daily_minutes, store_candidate
from .planner import generate_plan, replace_pending_plan
from .repositories import Repository, dumps, now_iso, uid


class RecordingMemory(MemoryManager):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[dict[str, Any]] = []

    def remember(self, content: str, category: str = "note", **kwargs: Any) -> MemoryItem:
        self.writes.append({"content": content, "category": category, "importance": kwargs.get("importance", 3), "long": False})
        return super().remember(content, category, **kwargs)

    def promote(self, content: str, category: str, reason: str, importance: int = 4, tags: list[str] | None = None) -> MemoryItem:
        self.writes.append({"content": content, "category": category, "importance": importance, "long": True})
        return super().promote(content, category, reason, importance, tags)


class TurnService:
    def __init__(self, root: Path | str, llm: Any = None) -> None:
        self.repo = Repository(root)
        self.llm = llm
        # 旧 JSON 需要显式导入，不能悄悄启动一份空档案。
        with transaction(self.repo.path, write=False) as db:
            imported = db.execute("SELECT 1 FROM imports LIMIT 1").fetchone()
        if not imported and any((self.repo.root / name).exists() for name in ("goals.json", "plans.json", "memory.json", "reviews.json")):
            raise ValueError("发现旧 JSON，请先运行 migrate_data --dry-run 并正式导入")
        self.recover()

    def recover(self) -> None:
        with transaction(self.repo.path) as db:
            db.execute("UPDATE turns SET status='completed',reply=facts,degraded=1 WHERE status='committed' AND lease_until<?", (time.time(),))
            db.execute("UPDATE turns SET status='failed',error_code='INTERRUPTED' WHERE status IN ('accepted','analyzing') AND lease_until<?", (time.time(),))

    def get(self, turn_id: str) -> OperationResult:
        with transaction(self.repo.path, write=False) as db:
            row = db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
            if row is None:
                raise ServiceError("NOT_FOUND", "操作不存在", 404)
            return OperationResult(turn_id, row["status"], row["state_revision"], row["facts"], row["reply"], row["error_code"], bool(row["degraded"]))

    def cancel(self, turn_id: str) -> OperationResult:
        with transaction(self.repo.path) as db:
            db.execute("UPDATE turns SET status='cancelled' WHERE id=? AND status IN ('accepted','analyzing')", (turn_id,))
        return self.get(turn_id)

    def accept(self, conversation_id: str, request_id: str, text: str, command: str = "chat", payload: dict[str, Any] | None = None) -> tuple[str, bool]:
        fingerprint = hashlib.sha256(dumps([conversation_id, command, text, payload]).encode()).hexdigest()
        with transaction(self.repo.path) as db:
            old = db.execute("SELECT id,input_hash FROM turns WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old["input_hash"] != fingerprint:
                    raise ServiceError("IDEMPOTENCY_KEY_REUSED", "请求标识已用于不同输入")
                return old["id"], False
            turn_id = uid("t")
            stamp = now_iso()
            db.execute("INSERT OR IGNORE INTO conversations VALUES (?,?)", (conversation_id, stamp))
            db.execute("INSERT INTO turns(id,request_id,input_hash,conversation_id,command,status,created_at,lease_until) VALUES (?,?,?,?,?,'accepted',?,?)",
                       (turn_id, request_id, fingerprint, conversation_id, command, stamp, time.time() + 900))
            seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
            profile = self.repo.profile(db)
            day = datetime.now(ZoneInfo(profile["timezone"])).date().isoformat()
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,0)", (uid("msg"), conversation_id, seq, turn_id, "user", text, stamp, day))
        return turn_id, True

    def submit(self, conversation_id: str, request_id: str, text: str, *, command: str = "chat", payload: dict[str, Any] | None = None,
               on_event: Callable[[str, dict[str, Any]], None] | None = None) -> OperationResult:
        turn_id, fresh = self.accept(conversation_id, request_id, text, command, payload)
        if on_event:
            on_event("operation", {"turn_id": turn_id, "request_id": request_id})
        return self.run(turn_id, text, command, payload, on_event) if fresh else self.get(turn_id)

    def run(self, turn_id: str, text: str, command: str = "chat", payload: dict[str, Any] | None = None,
            on_event: Callable[[str, dict[str, Any]], None] | None = None) -> OperationResult:
        emit = on_event or (lambda *_: None)
        try:
            for attempt in range(2):
                with transaction(self.repo.path) as db:
                    row = db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
                    if row["status"] not in ("accepted", "analyzing"):
                        break
                    profile = self.repo.profile(db)
                    revision = profile["state_revision"]
                    goals, items, reviews = self.repo.load_domain(db)
                    source = db.execute("SELECT id FROM messages WHERE turn_id=? AND role='user'", (turn_id,)).fetchone()[0]
                    memory = RecordingMemory()
                    for entry in self.repo.memories(db, "active"):
                        memory.long.items.append(MemoryItem.from_dict(entry))
                    db.execute("UPDATE turns SET status='analyzing',base_revision=? WHERE id=?", (revision, turn_id))
                business_day = datetime.now(ZoneInfo(profile["timezone"])).date()
                def budget(day: date) -> int:
                    candidate = budget_candidate(text, business_day, source) if command == "chat" else None
                    with transaction(self.repo.path, write=False) as db:
                        if self.repo.profile(db)["budget_status"] == "needs_confirmation" and not (candidate and candidate.status == "active" and candidate.scope == "default"):
                            return resolve_daily_minutes(db, day)
                        for scope in (f"date:{day.isoformat()}", "weekend" if day.weekday() >= 5 else "weekday", "default"):
                            if candidate and candidate.status == "active" and candidate.scope == scope:
                                return candidate.value
                            active = db.execute("SELECT value FROM constraints WHERE scope=? AND status='active' AND effective_from<=? AND (effective_to IS NULL OR effective_to>=?)", (scope, day.isoformat(), day.isoformat())).fetchone()
                            if active:
                                return active[0]
                    return 60
                planner = StudyPlanner(memory, goals, items, reviews, budget_for_day=budget, business_day=business_day)
                planner.reporter = lambda step: emit("step", vars(step))
                emit("status", {"text": "正在读取学习档案…"})
                facts = self.analyze(planner, text, command, payload or {}, turn_id)
                with transaction(self.repo.path) as db:
                    current = self.repo.profile(db)
                    status = db.execute("SELECT status FROM turns WHERE id=?", (turn_id,)).fetchone()[0]
                    if status == "cancelled":
                        break
                    if current["state_revision"] != revision:
                        if attempt == 0:
                            continue
                        raise ServiceError("REVISION_CONFLICT", "状态已更新，请重新提交")
                    if command == "reset":
                        db.execute("DELETE FROM memory_terms")
                        db.execute("DELETE FROM memory_sources")
                        db.execute("DELETE FROM memories")
                        db.execute("DELETE FROM constraints")
                    self.repo.save_domain(db, planner.goals, planner.items, planner.reviews)
                    self.save_memory(db, memory, source, text, command, payload or {}, business_day)
                    db.execute("UPDATE profile SET state_revision=state_revision+1")
                    db.execute("UPDATE turns SET status='committed',facts=?,state_revision=? WHERE id=?", (facts, revision + 1, turn_id))
                emit("step", {"phase": "提交", "title": "变更已保存", "ok": True, "detail": f"版本 {revision+1}"})
                reply, degraded = facts, False
                if self.llm is not None and command == "chat":
                    try:
                        chunks = []
                        messages = [{"role": "system", "content": REPLY_SYSTEM_PROMPT}, {"role": "user", "content": facts}]
                        for delta in self.llm.stream(messages):
                            if delta.text:
                                chunks.append(delta.text)
                                emit("delta", {"text": delta.text})
                        reply = "".join(chunks).strip() or facts
                    except Exception:
                        reply, degraded = facts, True
                with transaction(self.repo.path) as db:
                    db.execute("UPDATE turns SET status='completed',reply=?,degraded=?,lease_until=0 WHERE id=?", (reply, int(degraded), turn_id))
                    seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM messages WHERE conversation_id=?", (row["conversation_id"],)).fetchone()[0]
                    db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,0)", (uid("msg"), row["conversation_id"], seq, turn_id, "assistant", reply, now_iso(), datetime.now(ZoneInfo(profile["timezone"])).date().isoformat()))
                break
        except ServiceError as exc:
            with transaction(self.repo.path) as db:
                db.execute("UPDATE turns SET status='failed',error_code=? WHERE id=? AND status IN ('accepted','analyzing')", (exc.code, turn_id))
            raise
        except Exception:
            with transaction(self.repo.path) as db:
                db.execute("UPDATE turns SET status='failed',error_code='PROCESSING_FAILED' WHERE id=? AND status IN ('accepted','analyzing')", (turn_id,))
            raise
        return self.get(turn_id)

    def analyze(self, planner: StudyPlanner, text: str, command: str, payload: dict[str, Any], turn_id: str) -> str:
        steps: list[Any] = []
        bundle = planner.memory.recall(text)
        if command == "goal":
            goal = planner._record_goal(payload["title"], payload.get("deadline", ""), [], steps)
            draft = generate_plan(goal, start=planner.business_day, daily_minutes=payload.get("daily_minutes", 60))
            replace_pending_plan(planner.items, goal.id, draft.items, payload.get("daily_minutes", 60))
            return f"已创建目标「{goal.title}」并排期。"
        if command == "toggle":
            target = next((i for i in planner.items if i.id == payload["id"]), None)
            if target is None:
                raise ServiceError("NOT_FOUND", "任务不存在", 404)
            target.done = payload["done"]
            return f"{'完成' if target.done else '取消完成'}「{target.title}」"
        if command == "rollover":
            return "已整理当前业务日，原始会话保留。"
        if command == "reset":
            planner.goals.clear()
            planner.items.clear()
            planner.reviews.clear()
            return "学习档案已清空。"
        if command != "chat":
            raise ServiceError("INVALID_COMMAND", "未知操作", 422)
        decision = None
        if self.llm:
            try:
                decision = self.llm.complete_json([{"role": "system", "content": DECIDE_SYSTEM_PROMPT}, {"role": "user", "content": text}])
            except Exception:
                pass
        if isinstance(decision, dict) and decision.get("action") in ACT_ALL:
            planner._record(steps, StepRecord("决策", "模型决策", True, decision["action"]))
            facts = planner._apply_model_decision(decision, text, bundle, steps, Turn("", ""))
        else:
            facts = planner._run_intent(detect_intent(text), text, bundle, steps, Turn("", ""))
        candidate = budget_candidate(text, planner.business_day, "analysis")
        if candidate and candidate.status == "active":
            planner._capture_budget(text, steps)
        return facts

    def save_memory(self, db: Any, memory: RecordingMemory, source: str, text: str, command: str,
                    payload: dict[str, Any], day: date) -> None:
        candidate = budget_candidate(text, day, source) if command == "chat" else None
        if command == "goal":
            candidate = MemoryCandidate(f"可学习时间：{payload.get('daily_minutes', 60)} 分钟", "preference", "active", "fact", source, text, "default", payload.get("daily_minutes", 60))
        if candidate:
            store_candidate(db, candidate)
        for item in memory.writes:
            if not item["long"] or item["category"] == "preference":
                continue
            active = item["category"] == "goal" and item["content"].removeprefix("学习目标：") in text
            store_candidate(db, MemoryCandidate(item["content"], item["category"], "active" if active else "pending", "experience", source, text))
