"""学习规划师的 Python 侧。

分三块：
- 领域内核（`memory` / `planner` / `review` / `agent` / `storage` / `models`）：纯逻辑，不联网
- FastAPI 服务（`main`）：SSE 流式对话 + REST 状态快照
- 命令行入口（`cli`）：`python -m study_planner.backend`

前端在同级 `frontend/`，工作台通过 SSE 订阅这一轮的留痕与状态快照。
"""

from __future__ import annotations
