"""将分析与短事务提交分开，使断线、并发和重试都有可查询结果。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta

from .context import Clock, ContextBuilder
from .maintenance import Maintenance
from .deletion import DeletionService
from .retrieval import Retrieval
from .agent import StepRecord, StudyPlanner, Turn, detect_intent
from .db import transaction, runtime_lock
from .llm import ACT_ALL, validate_decision
from .memory import MemoryManager
from .memory_types import OperationResult, ServiceError
from .models import MemoryItem
from .memory_types import MemoryCandidate
from .memory_policy import budget_candidate, resolve_daily_minutes, store_candidate, confirm_memory
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
    def __init__(self, root: Path | str, llm: Any = None, clock: Clock | None = None) -> None:
        self._runtime_guard = runtime_lock(Path(root).expanduser())
        self._runtime_guard.__enter__()
        try:
            self.repo = Repository(root)
            self.llm = llm
            self.clock = clock or Clock()
            self.context = ContextBuilder(self.repo, self.clock)
            self.maintenance = Maintenance(self.repo, self.clock)
            # 旧 JSON 需要显式导入，不能悄悄启动一份空档案。
            with transaction(self.repo.path, write=False) as db:
                imported = db.execute("SELECT 1 FROM imports LIMIT 1").fetchone()
            if not imported and any((self.repo.root / name).exists() for name in ("goals.json", "plans.json", "memory.json", "reviews.json")):
                from .migrate_data import inspect_source, migrate
                try:
                    data, _, _ = inspect_source(self.repo.root)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ServiceError("MIGRATION_REQUIRED", "旧档案校验失败，请先运行 migrate_data --dry-run 检查并迁移", 503) from exc
                if any(data[name] for name in ("goals", "plans", "reviews", "memories")):
                    raise ServiceError("MIGRATION_REQUIRED", "发现旧学习记录，请先运行 migrate_data --dry-run 检查并迁移；原文件已保留", 503)
                with transaction(self.repo.path, write=False) as db:
                    empty_database = not any(db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() for table in ("goals", "plan_items", "reviews", "turns", "memories", "constraints", "messages"))
                if empty_database:
                    # 旧版首次启动也会生成空 JSON；只为空档案初始化并留下导入审计。
                    migrate(self.repo.root, self.repo.root, dry_run=False)
            self.deletion = DeletionService(self.repo)
            self.deletion.initialize()
            self.retrieval = Retrieval(self.repo, self.clock)
            self.recover()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        guard = getattr(self, "_runtime_guard", None)
        if guard is not None:
            self._runtime_guard = None
            guard.__exit__(None, None, None)

    def recover(self) -> None:
        with transaction(self.repo.path) as db:
            interrupted = db.execute("SELECT * FROM turns WHERE status='committed' AND lease_until<? LIMIT 100", (time.time(),)).fetchall()
            for turn in interrupted:
                db.execute("UPDATE turns SET status='completed',reply=facts,degraded=1,lease_until=0 WHERE id=?", (turn["id"],))
                if turn["facts"] and not db.execute("SELECT 1 FROM messages WHERE turn_id=? AND role='assistant'", (turn["id"],)).fetchone():
                    seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM messages WHERE conversation_id=?", (turn["conversation_id"],)).fetchone()[0]
                    mid = uid("msg")
                    day = self.clock.day(self.repo.profile(db)["timezone"]).isoformat()
                    db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,0)", (mid, turn["conversation_id"], seq, turn["id"], "assistant", turn["facts"], now_iso(), day))
                    db.execute("INSERT INTO artifact_sources VALUES ('message',?,?)", (mid, turn["id"]))
            db.execute("UPDATE turns SET status='failed',error_code='INTERRUPTED' WHERE status IN ('accepted','analyzing') AND lease_until<?", (time.time(),))

    def get(self, turn_id: str) -> OperationResult:
        self.recover()
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
            day = self.clock.day(profile["timezone"]).isoformat()
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,0)", (uid("msg"), conversation_id, seq, turn_id, "user", text, stamp, day))
        return turn_id, True

    def submit(self, conversation_id: str, request_id: str, text: str, *, command: str = "chat", payload: dict[str, Any] | None = None,
               on_event: Callable[[str, dict[str, Any]], None] | None = None) -> OperationResult:
        turn_id, fresh = self.accept(conversation_id, request_id, text, command, payload)
        if on_event:
            on_event("operation", {"turn_id": turn_id, "request_id": request_id})
        if fresh and command in ("reset", "memory_delete"):
            try:
                if command == "reset":
                    self.deletion.reset(turn_id)
                else:
                    self.deletion.delete(payload["id"], payload["expected_revision"], turn_id)
            except Exception as exc:
                with transaction(self.repo.path) as db:
                    db.execute("UPDATE turns SET status='failed',error_code=? WHERE id=?", (getattr(exc, "code", "DELETE_PENDING_RECOVERY"), turn_id))
                raise
            return self.get(turn_id)
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
                business_day = self.clock.day(profile["timezone"])
                def budget(day: date) -> int:
                    candidate = budget_candidate(text, business_day, source) if command == "chat" else None
                    if command == "goal" and (payload or {}).get("daily_minutes") is not None:
                        candidate = MemoryCandidate("", status="active", kind="fact", value=payload["daily_minutes"])
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
                planner.context_source_ids = []
                facts = self.analyze(planner, text, command, payload or {}, turn_id)
                with transaction(self.repo.path) as db:
                    current = self.repo.profile(db)
                    status = db.execute("SELECT status FROM turns WHERE id=?", (turn_id,)).fetchone()[0]
                    if status != "analyzing":
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
                    if command == "memory_confirm":
                        confirm_memory(db, payload["id"], payload["expected_revision"], payload.get("content"), payload.get("scope"), payload.get("value"), business_day, source)
                    self.save_memory(db, memory, source, text, command, payload or {}, business_day)
                    db.executemany("INSERT OR IGNORE INTO artifact_sources VALUES ('turn',?,?)", [(turn_id, sid) for sid in planner.context_source_ids + [source]])
                    db.execute("UPDATE profile SET state_revision=state_revision+1")
                    db.execute("UPDATE turns SET status='committed',facts=?,state_revision=? WHERE id=?", (facts, revision + 1, turn_id))
                emit("step", {"phase": "提交", "title": "变更已保存", "ok": True, "detail": f"版本 {revision+1}"})
                reply, degraded = facts, False
                if self.llm is not None and command == "chat":
                    try:
                        chunks = []
                        reply_context = self.context.build(turn_id, "reply", facts, self.retrieval.search(text))
                        messages = reply_context.messages
                        with transaction(self.repo.path) as db:
                            db.executemany("INSERT OR IGNORE INTO artifact_sources VALUES ('turn',?,?)", [(turn_id, sid) for sid in reply_context.source_ids])
                        for delta in self.llm.stream(messages):
                            if delta.text:
                                with transaction(self.repo.path, write=False) as db:
                                    if self.repo.profile(db)["privacy_revision"] != profile["privacy_revision"]:
                                        chunks = []
                                        break
                                chunks.append(delta.text)
                                emit("delta", {"text": delta.text})
                        reply = "".join(chunks).strip() or facts
                    except Exception:
                        reply, degraded = facts, True
                with transaction(self.repo.path) as db:
                    if self.repo.profile(db)["privacy_revision"] != profile["privacy_revision"]:
                        reply, degraded = "来源已更新或删除，请重新发起这一轮。", True
                    db.execute("UPDATE turns SET status='completed',reply=?,degraded=?,lease_until=0 WHERE id=?", (reply, int(degraded), turn_id))
                    seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM messages WHERE conversation_id=?", (row["conversation_id"],)).fetchone()[0]
                    assistant_id = uid("msg")
                    db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,0)", (assistant_id, row["conversation_id"], seq, turn_id, "assistant", reply, now_iso(), self.clock.day(profile["timezone"]).isoformat()))
                    db.execute("INSERT INTO artifact_sources VALUES ('message',?,?)", (assistant_id, turn_id))
                try:
                    self.maintenance.schedule(row["conversation_id"])
                    self.maintenance.run_once()
                    from .backup import daily_backup
                    daily_backup(self.repo, business_day.isoformat())
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("maintenance_failed type=%s", type(exc).__name__)
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
        if command == "memory_confirm":
            return "记忆已确认或纠正。"
        if command == "goal":
            goal = planner._record_goal(payload["title"], payload.get("deadline", ""), [], steps)
            minutes = planner.budget_for_day(planner.business_day)
            draft = generate_plan(goal, start=planner.business_day, daily_minutes=minutes, budget_for_day=planner.budget_for_day)
            replace_pending_plan(planner.items, goal.id, draft.items, planner.budget_for_day, start=planner.business_day)
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
            context = self.context.build(turn_id, "decision", recalled=self.retrieval.search(text))
            planner.context_source_ids = context.source_ids
            try:
                decision = validate_decision(self.llm.complete_json(context.messages), text)
            except Exception:
                pass
        if isinstance(decision, dict) and decision.get("action") in ACT_ALL:
            planner._record(steps, StepRecord("决策", "模型决策", True, decision["action"]))
            entries = decision.get("memory", [])
            facts = planner._apply_model_decision({**decision, "memory": []}, text, bundle, steps, Turn("", ""))
            planner._apply_model_memory(entries, steps)
            if any(entry.get("promote") for entry in entries):
                facts += "\n已保存长期记忆候选，需在记忆面板确认后生效。"
        else:
            facts = planner._run_intent(detect_intent(text), text, bundle, steps, Turn("", ""))
        candidate = budget_candidate(text, planner.business_day, "analysis")
        if candidate and candidate.status == "active":
            planner._capture_budget(text, steps)
        return facts

    def save_memory(self, db: Any, memory: RecordingMemory, source: str, text: str, command: str,
                    payload: dict[str, Any], day: date) -> None:
        candidate = budget_candidate(text, day, source) if command == "chat" else None
        if command == "goal" and payload.get("daily_minutes") is not None:
            candidate = MemoryCandidate(f"可学习时间：{payload.get('daily_minutes', 60)} 分钟", "preference", "active", "fact", source, text, "default", payload.get("daily_minutes", 60))
        if candidate:
            store_candidate(db, candidate)
        for item in memory.writes:
            if not item["long"] or (item["category"] == "preference" and candidate is not None):
                continue
            active = item["category"] == "goal" and item["content"].removeprefix("学习目标：") in text
            store_candidate(db, MemoryCandidate(item["content"], item["category"], "active" if active else "pending", "experience", source, text))
