"""统一 Web 操作入口，持久化结果比 SSE 传输更可靠。"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .llm import build_client_from_env
from .memory_types import ServiceError
from .repositories import uid
from .turn_service import TurnService

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
                self._service = TurnService(self.root, build_client_from_env())
            return self._service

    def rebind(self, root: Path | str) -> None:
        with self.lock:
            self.root = Path(root)
            self._service = TurnService(self.root)


session = SessionStore()
app = FastAPI(title='学习规划师 API')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])


@app.exception_handler(ServiceError)
async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={'detail': str(exc), 'code': exc.code})


@app.exception_handler(sqlite3.Error)
async def database_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
    code = 'STORE_BUSY' if 'locked' in str(exc) else 'STORE_UNAVAILABLE'
    return JSONResponse(status_code=503, content={'detail': '档案暂时不可用，请稍后重试', 'code': code})


class WriteRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=120)


class ChatRequest(WriteRequest):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(default='default', min_length=1, max_length=120)


class ToggleRequest(WriteRequest):
    done: bool


class GoalRequest(WriteRequest):
    title: str = Field(min_length=1, max_length=120)
    deadline: str = ''
    daily_minutes: int = Field(default=60, ge=15, le=480)


def sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'mode': 'model' if session.service.llm is not None else 'rule', 'storage': 'sqlite'}


@app.get('/api/state')
def get_state(conversation_id: str = 'default') -> dict[str, Any]:
    return session.service.repo.snapshot(conversation_id)


@app.get('/api/turns/{turn_id}')
def get_turn(turn_id: str) -> dict[str, Any]:
    return session.service.get(turn_id).to_dict()


@app.post('/api/turns/{turn_id}/cancel')
def cancel_turn(turn_id: str) -> dict[str, Any]:
    return session.service.cancel(turn_id).to_dict()


@app.post('/api/chat')
async def chat(body: ChatRequest) -> StreamingResponse:
    service = session.service
    request_id = body.request_id or uid('request')
    turn_id, fresh = service.accept(body.conversation_id, request_id, body.message)
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
            yield sse(event, data)

    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def execute(command: str, text: str, body: WriteRequest, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = session.service.submit('default', body.request_id or uid('request'), text, command=command, payload=payload)
    return {**session.service.repo.snapshot(), 'operation': result.to_dict()}


@app.post('/api/tasks/{task_id}/toggle')
def toggle_task(task_id: str, body: ToggleRequest) -> dict[str, Any]:
    return execute('toggle', '更新待办完成状态', body, {'id': task_id, 'done': body.done})


@app.post('/api/goals')
def create_goal(body: GoalRequest) -> dict[str, Any]:
    return execute('goal', f'创建目标：{body.title}', body, body.model_dump(exclude={'request_id'}))


@app.post('/api/rollover')
def rollover(body: WriteRequest | None = None) -> dict[str, Any]:
    return execute('rollover', '整理当前业务日', body or WriteRequest())


@app.post('/api/reset')
def reset(body: WriteRequest | None = None) -> dict[str, Any]:
    return execute('reset', '清空学习档案', body or WriteRequest())
