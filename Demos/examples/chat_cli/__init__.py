"""命令行聊天助手：多轮对话 + 流式输出 + 历史存档。

模块划分：
- messages.py  消息结构：一条消息由什么组成
- context.py   上下文管理：哪些消息会被送进模型
- storage.py   历史持久化：JSONL 事件流
- llm.py       模型调用：流式输出与摘要
- chat.py      对话编排与终端交互
- main.py      命令行入口
"""

from .context import ContextPlan, ContextWindow
from .messages import Message
from .storage import SessionStore

__all__ = ["Message", "ContextWindow", "ContextPlan", "SessionStore"]
