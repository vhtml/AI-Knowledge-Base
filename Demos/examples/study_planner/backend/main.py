"""FastAPI 后端：把学习规划师暴露成可流式消费的工作台接口。

接口分两类：

**流式（SSE）** —— `POST /api/chat`
    一边处理一边把中间状态推给前端：先推"正在召回"，
    再推每一步留痕，最后推回答正文和最新状态快照。
    前端因此能实时显示"这轮动了哪块记忆"，而不是等一个干巴巴的结果。

**快照（REST）** —— `/api/state`、`/api/tasks/{id}/toggle` 等
    工作台的各个面板（今日待办、目标、记忆）需要读全量状态。
    用 REST 拉快照比用 SSE 推更合适：面板状态是"要什么拿什么"，
    不是"有什么推什么"。

会话状态由 `SessionStore` 持有，整个进程一份。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent import StepRecord, StudyPlanner
from ..memory import MemoryManager
from ..models import CATEGORY_GOAL, CATEGORY_NOTE, Goal, PlanItem, Review
from ..storage import Storage

DEFAULT_DATA_DIR = Path.home() / ".study-planner"

# 数据目录可以用环境变量覆盖，这样开发时不必动真实数据
DATA_DIR_ENV = "STUDY_PLANNER_DATA_DIR"


# ------------------------------------------------------------
# 会话状态
# ------------------------------------------------------------


@dataclass
class SessionStore:
    """进程内的会话状态。

    整个服务共用一份 StudyPlanner 实例：目标、待办、记忆都是
    全局状态，不是"每个连接一份"。所以所有写操作都要串行化，
    否则两个并发请求会同时改记忆，导致条数对不上。
    """

    planner: StudyPlanner
    storage: Storage
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def build(cls, root: Path | str = DEFAULT_DATA_DIR) -> "SessionStore":
        """从磁盘恢复一份会话状态。"""

        storage = Storage(root)
        memory = storage.build_memory()

        planner = StudyPlanner(
            memory=memory,
            goals=storage.load_goals(),
            items=storage.load_items(),
            reviews=storage.load_reviews(),
        )

        return cls(planner=planner, storage=storage)
    def persist(self) -> None:
        """把当前状态落盘。"""

        self.storage.save_all(
            self.planner.memory,
            self.planner.goals,
            self.planner.items,
            self.planner.reviews,
        )

    def rebind(self, root: Path | str) -> None:
        """换到另一个数据目录并重新载入。

        测试里用它把会话指到临时目录，避免动真实数据。
        """

        self.storage = Storage(root)
        self.planner.memory = self.storage.build_memory()
        self.planner.goals = self.storage.load_goals()
        self.planner.items = self.storage.load_items()
        self.planner.reviews = self.storage.load_reviews()

    def snapshot(self) -> dict[str, Any]:
        """全量状态快照——前端各个面板的数据都从这里来。

        长期记忆带一个 `superseded` 标记而不是直接过滤掉：
        被取代的旧说法也是信息（它说明用户改过主意），
        前端可以折叠显示而不是假装没发生过。
        """

        planner = self.planner
        stats = planner.memory.stats()

        return {
            "goals": [goal.to_dict() for goal in planner.goals],
            "items": [item.to_dict() for item in planner.items],
            "reviews": [review.to_dict() for review in planner.reviews],
            "memory": {
                "stats": stats,
                "short": [item.to_dict() for item in planner.memory.short.items],
                "long": [
                    {
                        **item.to_dict(),
                        "superseded": bool(item.superseded_by),
                    }
                    for item in planner.memory.long.items
                ],
            },
        }


# ------------------------------------------------------------
# 请求 / 响应模型
# ------------------------------------------------------------


class ChatRequest(BaseModel):
    """聊天请求。"""

    message: str = Field(min_length=1, max_length=4000)


class ToggleRequest(BaseModel):
    """切换任务完成状态。"""

    done: bool


class GoalRequest(BaseModel):
    """直接创建目标（表单入口，不必靠自然语言）。"""

    title: str = Field(min_length=1, max_length=120)
    deadline: str = ""
    daily_minutes: int = Field(default=60, ge=15, le=480)


# ------------------------------------------------------------
# SSE 事件
# ------------------------------------------------------------


def sse(event: str, data: dict[str, Any]) -> str:
    """把一个事件序列化成 SSE 帧。"""

    payload = json.dumps(data, ensure_ascii=False)

    return f"event: {event}\ndata: {payload}\n\n"


# ------------------------------------------------------------
# 应用
# ------------------------------------------------------------

app = FastAPI(title="学习规划师 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = SessionStore.build(os.getenv(DATA_DIR_ENV) or DEFAULT_DATA_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查。"""

    return {"status": "ok"}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    """全量状态快照。"""

    with session.lock:
        return session.snapshot()


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """流式对话：边处理边推送。

    推的事件依次是：
    - `status`  阶段提示（"正在召回记忆"这类）
    - `step`    每一条留痕，前端用它渲染"这轮动了哪块记忆"
    - `reply`   回答正文
    - `state`   最新状态快照
    - `error`   出错时的说明

    处理逻辑跑在一个工作线程里：Agent 是同步的，
    而 FastAPI 的事件循环不能被同步代码阻塞住。
    线程和事件循环之间用队列传递，队列满了就丢最旧的
    （宁可少显示一条留痕，也不能让生产端卡死）。
    """

    events: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=256)

    def report(step: StepRecord) -> None:
        """留痕回调——从工作线程调用。"""

        try:
            events.put_nowait(
                {
                    "event": "step",
                    "data": {
                        "phase": step.phase,
                        "title": step.title,
                        "ok": step.ok,
                        "detail": step.detail,
                    },
                }
            )
        except queue.Full:
            pass

    def worker() -> None:
        """在工作线程里跑完一轮对话。"""

        try:
            with session.lock:
                events.put_nowait({"event": "status", "data": {"text": "正在召回记忆…"}})
                turn = session.planner.handle(body.message, reporter=report)
                session.persist()

            events.put_nowait({"event": "reply", "data": {"text": turn.reply}})
            events.put_nowait({"event": "state", "data": session.snapshot()})

        except Exception as exc:  # 兜底：不能因为一轮失败就断掉整个流
            events.put_nowait({"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}})

        finally:
            events.put_nowait(None)

    threading.Thread(target=worker, daemon=True).start()

    async def stream() -> Iterator[str]:
        """把队列里的事件转成 SSE 帧吐出去。"""

        loop = asyncio.get_running_loop()

        while True:
            item = await loop.run_in_executor(None, events.get)

            if item is None:
                break

            yield sse(item["event"], item["data"])

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tasks/{task_id}/toggle")
def toggle_task(task_id: str, body: ToggleRequest) -> dict[str, Any]:
    """切换某条待办的完成状态。

    任务勾选是工作台最高频的交互，单独给一个接口——
    不必为了点一下勾选就走一轮自然语言。
    """

    with session.lock:
        target = next((item for item in session.planner.items if item.id == task_id), None)

        if target is None:
            raise HTTPException(404, f"任务不存在：{task_id}")

        target.done = body.done
        session.planner.memory.remember(
            f"{'完成' if body.done else '取消完成'}「{target.title}」",
            CATEGORY_NOTE,
            importance=3,
        )
        session.persist()

        return session.snapshot()


@app.post("/api/goals")
def create_goal(body: GoalRequest) -> dict[str, Any]:
    """直接创建目标并按它排计划。

    同一个目标重复提交是**更新**而不是新增——
    id 由标题和日期派生，重复提交会撞上同一个 id。
    """

    from ..models import stable_id, today_str

    with session.lock:
        goal_id = stable_id("g", body.title, today_str())
        existing = next((item for item in session.planner.goals if item.id == goal_id), None)

        if existing is not None:
            existing.title = body.title

            if body.deadline:
                existing.deadline = body.deadline

            goal = existing

        else:
            goal = Goal(id=goal_id, title=body.title, deadline=body.deadline)
            session.planner.goals.append(goal)

        session.planner.memory.promote(
            f"学习目标：{body.title}",
            CATEGORY_GOAL,
            "用户在界面上创建目标",
            importance=5,
        )

        from ..planner import generate_plan

        draft = generate_plan(goal, daily_minutes=body.daily_minutes)

        existing_ids = {item.id for item in session.planner.items}
        session.planner.items.extend(
            item for item in draft.items if item.id not in existing_ids
        )

        session.persist()

        return session.snapshot()


@app.post("/api/rollover")
def rollover() -> dict[str, Any]:
    """模拟跨天：短期工作区清空，值得沉淀的进长期档案。"""

    with session.lock:
        session.planner.rollover()
        session.persist()

        return session.snapshot()


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    """清空全部数据，回到初始状态。"""

    with session.lock:
        planner = session.planner
        planner.goals.clear()
        planner.items.clear()
        planner.reviews.clear()
        planner.memory.short.clear()
        planner.memory.long.items.clear()
        session.persist()

        return session.snapshot()
