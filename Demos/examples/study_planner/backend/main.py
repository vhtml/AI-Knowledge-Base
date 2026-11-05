"""统一 Web 操作入口，持久化结果比 SSE 传输更可靠。"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sqlite3
import threading
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .llm import build_client_from_env
from .memory_types import ServiceError
from .repositories import uid
from .turn_service import TurnService
from .db import transaction

DEFAULT_DATA_DIR = Path.home() / '.study-planner'
DATA_DIR_ENV = 'STUDY_PLANNER_DATA_DIR'


class SessionStore:
    """只持有配置与服务，不持有可变的用户业务快照。"""
    def __init__(self) -> None:
        self.root = Path(os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR)
        self._service: TurnService | None = None
        self.lock = threading.Lock()

    @property
    def service(self) -> TurnService:
        with self.lock:
            if self._service is None:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[3] / '.env')
                self.root = Path(os.getenv(DATA_DIR_ENV) or self.root)
                try:
                    self._service = TurnService(self.root, build_client_from_env())
                except (ValueError, OSError) as exc:
                    logging.getLogger(__name__).warning('startup_failed type=%s', type(exc).__name__)
                    raise ServiceError('STARTUP_VALIDATION_FAILED', '档案初始化未通过，请检查数据目录、配置和删除清单', 503) from exc
            return self._service

    def rebind(self, root: Path | str) -> None:
        with self.lock:
            if self._service is not None:
                self._service.close()
            self.root = Path(root)
            self._service = TurnService(self.root)


session = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stopped = threading.Event()

    def maintain() -> None:
        while not stopped.wait(5):
            service = session._service
            if service is None:
                continue
            try:
                from .backup import daily_backup
                service.deletion.replay()
                service.recover()
                service.maintenance.run_once()
                with transaction(service.repo.path, write=False) as db:
                    day = service.clock.day(service.repo.profile(db)['timezone']).isoformat()
                daily_backup(service.repo, day)
            except Exception as exc:
                logging.getLogger(__name__).warning('maintenance_failed type=%s', type(exc).__name__)
    worker = threading.Thread(target=maintain, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=10)
        if session._service is not None and not worker.is_alive():
            session._service.close()


app = FastAPI(title='学习规划师 API', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])


@app.exception_handler(ServiceError)
async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={'detail': str(exc), 'code': exc.code})


@app.exception_handler(sqlite3.Error)
async def database_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
    code = 'STORE_BUSY' if 'locked' in str(exc) else 'STORE_UNAVAILABLE'
    return JSONResponse(status_code=503, content={'detail': '档案暂时不可用，请稍后重试', 'code': code})


class WriteRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}
    conversation_id: str = Field(default='default', min_length=1, max_length=120)
    request_id: str | None = Field(default=None, min_length=1, max_length=120)


class ChatRequest(WriteRequest):
    message: str = Field(min_length=1, max_length=4000)


class ToggleRequest(WriteRequest):
    done: bool


class ConversationDeletion(BaseModel):
    last_seq: int = Field(ge=1)
    revision: int = Field(ge=0)


class GoalRequest(WriteRequest):
    title: str = Field(min_length=1, max_length=120)
    deadline: str = ''
    daily_minutes: int | None = Field(default=None, ge=15, le=480)


class MemoryRequest(WriteRequest):
    expected_revision: int = Field(ge=1)
    content: str | None = Field(default=None, min_length=1, max_length=1000)
    scope: str | None = Field(default=None, max_length=30, pattern=r"^(default|weekday|weekend|date:\d{4}-\d{2}-\d{2})$")
    value: int | None = Field(default=None, ge=15, le=480)


def sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


@app.get('/health')
def health() -> dict[str, Any]:
    service = session.service
    with transaction(service.repo.path, write=False) as db:
        failed = db.execute("SELECT count(*) FROM jobs WHERE status='failed'").fetchone()[0]
    backups = sorted((service.repo.root / 'backups').glob('*.db.json'))
    return {'status': 'ok', 'mode': 'model' if service.llm is not None else 'rule', 'storage': 'sqlite',
            'sqlite_version': sqlite3.sqlite_version, 'failed_jobs': failed,
            'last_backup_at': json.loads(backups[-1].read_text())['created_at'] if backups else None}


@app.get('/api/state')
def get_state(conversation_id: str = 'default') -> dict[str, Any]:
    return session.service.repo.snapshot(conversation_id)


@app.get('/api/turns/{turn_id}')
def get_turn(turn_id: str) -> dict[str, Any]:
    return session.service.get(turn_id).to_dict()


@app.get('/api/requests/{request_id}')
def request_result(request_id: str) -> dict[str, Any]:
    with transaction(session.service.repo.path, write=False) as db:
        row = db.execute('SELECT id FROM turns WHERE request_id=?', (request_id,)).fetchone()
        if not row:
            raise ServiceError('NOT_FOUND', '请求尚未被接收，可使用原标识重试', 404)
    return session.service.get(row[0]).to_dict()


@app.post('/api/turns/{turn_id}/cancel')
def cancel_turn(turn_id: str) -> dict[str, Any]:
    return session.service.cancel(turn_id).to_dict()


@app.post('/api/chat')
async def chat(body: ChatRequest) -> StreamingResponse:
    service = session.service
    request_id = body.request_id or uid('request')
    turn_id, fresh = service.accept(body.conversation_id, request_id, body.message)
    with transaction(service.repo.path, write=False) as db:
        stream_privacy = service.repo.profile(db)['privacy_revision']
    events: queue.Queue[Any] = queue.Queue(maxsize=256)

    def emit(event: str, data: dict[str, Any]) -> None:
        try:
            events.put_nowait((event, data))
        except queue.Full:
            if event in ('delta', 'step', 'status'):
                return
            try:
                events.get_nowait()
            except queue.Empty:
                pass
            events.put_nowait((event, data))

    def worker() -> None:
        try:
            emit('operation', {'turn_id': turn_id, 'request_id': request_id})
            result = service.run(turn_id, body.message, on_event=emit) if fresh else service.get(turn_id)
            if result.status == 'completed':
                emit('reply', {'text': result.reply, **result.to_dict()})
            emit('state', service.repo.snapshot(body.conversation_id))
            emit('done', result.to_dict())
        except Exception as exc:
            emit('error', {'message': str(exc) if isinstance(exc, ServiceError) else '处理失败，可通过操作编号查询结果', 'code': getattr(exc, 'code', 'PROCESSING_FAILED'), 'turn_id': turn_id})
        finally:
            emit('end', {})

    threading.Thread(target=worker, daemon=True).start()

    async def stream():
        while True:
            event, data = await asyncio.to_thread(events.get)
            if event == 'end':
                break
            if event in ('step', 'delta', 'reply', 'done'):
                with transaction(service.repo.path, write=False) as db:
                    if service.repo.profile(db)['privacy_revision'] != stream_privacy:
                        continue
            yield sse(event, data)

    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def execute(command: str, text: str, body: WriteRequest, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = session.service.submit(body.conversation_id, body.request_id or uid('request'), text, command=command, payload=payload)
    return {**session.service.repo.snapshot(body.conversation_id), 'operation': result.to_dict()}


@app.post('/api/tasks/{task_id}/toggle')
def toggle_task(task_id: str, body: ToggleRequest) -> dict[str, Any]:
    return execute('toggle', '更新待办完成状态', body, {'id': task_id, 'done': body.done})


@app.post('/api/goals')
def create_goal(body: GoalRequest) -> dict[str, Any]:
    return execute('goal', f'创建目标：{body.title}', body, body.model_dump(exclude={'request_id', 'conversation_id'}))


@app.delete('/api/goals/{goal_id}')
def delete_goal(goal_id: str, body: WriteRequest) -> dict[str, Any]:
    return execute('goal_delete', '删除目标及关联待办', body, {'id': goal_id})


@app.post('/api/rollover')
def rollover(body: WriteRequest | None = None) -> dict[str, Any]:
    return execute('rollover', '整理当前业务日', body or WriteRequest())


@app.post('/api/reset')
def reset(body: WriteRequest | None = None) -> dict[str, Any]:
    return execute('reset', '清空学习档案', body or WriteRequest())


@app.get('/api/conversations')
def conversations(offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=100)) -> list[dict[str, Any]]:
    with transaction(session.service.repo.path, write=False) as db:
        rows = db.execute("""
            SELECT c.id, c.created_at, max(m.created_at) AS last_activity,
                   (SELECT substr(content, 1, 60) FROM messages
                    WHERE conversation_id=c.id AND role='user' AND deleted=0
                    ORDER BY seq LIMIT 1) AS title
            FROM conversations c JOIN messages m ON m.conversation_id=c.id AND m.deleted=0
            GROUP BY c.id ORDER BY last_activity DESC, c.id LIMIT ? OFFSET ?
        """, (limit, offset))
        return [dict(row) for row in rows]


@app.get('/api/export')
def export_archive() -> JSONResponse:
    return JSONResponse(content=session.service.repo.export(), headers={'Content-Disposition': 'attachment; filename="study-planner-export.json"'})


@app.get('/api/conversations/{conversation_id}/deletion-preview')
def conversation_deletion_preview(conversation_id: str) -> dict[str, Any]:
    with transaction(session.service.repo.path, write=False) as db:
        return session.service.deletion.conversation_preview(db, conversation_id)


@app.delete('/api/conversations/{conversation_id}')
def delete_conversation(conversation_id: str, body: ConversationDeletion) -> dict[str, bool]:
    session.service.deletion.delete_conversation(conversation_id, body.last_seq, body.revision)
    return {'deleted': True}


@app.get('/api/conversations/{conversation_id}/messages')
def conversation_messages(conversation_id: str, before_seq: int = 9223372036854775807,
                          limit: int = Query(default=50, ge=1, le=100), source: str | None = None) -> dict[str, Any]:
    with transaction(session.service.repo.path, write=False) as db:
        if source is not None:
            target = db.execute('SELECT seq FROM messages WHERE id=? AND conversation_id=? AND deleted=0', (source, conversation_id)).fetchone()
            if target is None:
                raise ServiceError('SOURCE_UNAVAILABLE', '原始消息已删除或无法访问', 404)
            # 带上紧随来源的回复，即使来源在几百轮之前，也只读取一页。
            before_seq = target['seq'] + min(limit, 3)
        rows = db.execute('SELECT id,seq,role,content,created_at,turn_id FROM messages WHERE conversation_id=? AND seq<? AND deleted=0 ORDER BY seq DESC LIMIT ?', (conversation_id, before_seq, limit)).fetchall()
        return {'messages': [dict(r) for r in reversed(rows)], 'has_more': len(rows) == limit,
                'privacy_revision': session.service.repo.profile(db)['privacy_revision']}


@app.get('/api/memories')
def memories(status: str | None = None, offset: int = Query(default=0, ge=0),
             limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    with transaction(session.service.repo.path, write=False) as db:
        rows = session.service.repo.memories(db, status, offset, limit)
        return {'items': rows, 'has_more': len(rows) == limit, 'offset': offset,
                'state_revision': session.service.repo.profile(db)['state_revision']}


@app.get('/api/memories/search')
def search_memories(q: str = Query(max_length=1000)) -> list[dict]:
    return session.service.retrieval.search(q)


@app.get('/api/memories/{memory_id}/deletion-preview')
def deletion_preview(memory_id: str) -> dict[str, Any]:
    return session.service.deletion.preview(memory_id)


@app.patch('/api/memories/{memory_id}')
def correct_memory(memory_id: str, body: MemoryRequest) -> dict[str, Any]:
    payload = body.model_dump(exclude={'request_id', 'conversation_id'}, exclude_none=True)
    payload['id'] = memory_id
    return execute('memory_confirm', '确认或纠正学习记忆', body, payload)


@app.delete('/api/memories/{memory_id}')
def delete_memory(memory_id: str, body: MemoryRequest) -> dict[str, Any]:
    return execute('memory_delete', '清除所选学习记忆', body, {'id': memory_id, 'expected_revision': body.expected_revision})
