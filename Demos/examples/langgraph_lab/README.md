# langgraph_lab · 十七章动手实验

配套笔记在 [`AI/LangGraph/`](../../../AI/LangGraph/)。**笔记讲原理，这里跑代码**——
每章一个可独立运行的实验，跑挂了就是没学会。

---

## 快速开始

```bash
cd Demos/examples

# 列出各章
uv run --project .. python -m langgraph_lab list

# 跑某一章
uv run --project .. python -m langgraph_lab ch07

# 跑全部十七章
uv run --project .. python -m langgraph_lab all

# 自检：验证笔记与实验一一对应，且断言都还成立
uv run --project .. python -m langgraph_lab selftest
```

依赖装在 `Demos/` 这一层（`cd Demos && uv sync`），实验本身不装任何东西。

---

## 两条运行路径

| 路径 | 触发 | 模型来源 | 断言强度 | 什么时候用 |
| --- | --- | --- | --- | --- |
| **确定性路径** | 默认 | `labkit.scripted_model()` 脚本化假模型 | 严格（校验具体值） | 学习、回归、自检 |
| **真实路径** | `--live` | `Demos/.env` 里的 OpenAI 兼容端点 | 宽松（只校验形状） | 看真模型怎么跑 |

```bash
uv run --project .. python -m langgraph_lab ch07 --live   # 单章真实路径
uv run --project .. python -m langgraph_lab live            # 全部章节真实路径
```

真实路径读 `Demos/.env` 的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（仓库既有约定）。
**没配也能跑**——会明确告诉你要补哪个变量，而不是在 HTTP 层报 401。

为什么默认走确定性路径：真实模型的输出不确定，断言写不死，
实验就退化成"看一眼觉得差不多"。确定性路径能把每条结论钉成可复现的断言，
真模型路径用来确认"脚本里的形状和真模型对得上"。

---

## 章节对照

| 章 | 主题 | 关键命令 |
| --- | --- | --- |
| 1 | 生态全景：框架/运行时/harness 三层 | `ch01` |
| 2 | 图模型：节点、边、共享状态、super-step | `ch02` |
| 3 | 状态定义与归约器 | `ch03` |
| 4 | 控制流：分支、循环、Send 与 Command | `ch04` |
| 5 | 节点韧性：重试、超时与错误处理 | `ch05` |
| 6 | 工具调用与工具节点 | `ch06` |
| 7 | 持久化：检查点与时间旅行 | `ch07` |
| 8 | 人在环路：中断、审批与恢复 | `ch08` |
| 9 | 流式与事件投影 | `ch09` |
| 10 | 子图与多智能体编排 | `ch10` |
| 11 | 函数式 API 与 Pregel 运行时 | `ch11` |
| 12 | LangChain 1.x：create_agent 与中间件 | `ch12` |
| 13 | MCP 接入与工具生态 | `ch13` |
| 14 | Deep Agents：harness 视角 | `ch14` |
| 15 | 可观测性与评测 | `ch15` |
| 16 | 部署：本地服务器、Studio 与生产 | `ch16` |
| 17 | 迁移与工程实践 | `ch17` |

---

## 目录结构

```
langgraph_lab/
├── __main__.py     命令行入口（按章调度）
├── labkit.py       公共工具层：模型工厂、检查点、输出与断言
├── selftest.py     自检：笔记↔实验对应关系 + 逐章断言
└── lessons/
    ├── __init__.py 章节登记表
    ├── ch01_ecosystem.py
    └── ...
```

**新增一章**只要三步：在 `lessons/` 下加 `chNN_xxx.py`（导出 `run()`），
在 `lessons/__init__.py` 的 `LESSONS` 里加一行，
在 `selftest.py` 的 `NOTE_FILES` 里登记对应笔记文件名。

---

## 公共层提供了什么（`labkit.py`）

| 工具 | 用途 |
| --- | --- |
| `scripted_model(responses)` | 脚本化假模型，实现�� `bind_tools`，可离线驱动 Agent 循环 |
| `tool_call(name, args)` | 造一条"只带工具调用"的助手消息 |
| `human(text)` / `ai(text)` | 造用户/助手消息 |
| `real_chat_model()` | 连 `Demos/.env` 的真实模型；缺配置时给清楚的提示 |
| `sqlite_checkpointer()` | 临时 SQLite 检查点器上下文管理器，退出自动清理 |
| `ensure(cond, msg)` | 中文断言辅助，失败时直接说明"期望什么、实际什么" |
| `banner/section/rule` | 统一输出格式 |
| `repo_root()` | 定位仓库根目录 |

**不要自己再造模型工厂**——统一走 labkit，将来换模型只改一处。

---

## 依赖版本（本机实测）

```
langgraph 1.2.20 / langchain 1.5.0 / langchain-core 1.8.0
langgraph-checkpoint 4.3.1 / langgraph-checkpoint-sqlite 3.1.3
deepagents 0.7.38 / langsmith 0.14.17 / langgraph-sdk 0.4.14
fastmcp 4.2.0 / mcp 2.3.0
Python 3.13.2
```

版本漂了先读官方 changelog，再改实验——**实验断言是跟着版本走的**。
