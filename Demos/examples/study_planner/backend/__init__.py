"""学习规划师的 FastAPI 后端。

暴露两类接口：
- SSE 流式对话（`POST /api/chat`），边处理边推送留痕
- REST 状态快照（`/api/state` 等），供工作台各面板拉取
"""

from __future__ import annotations
