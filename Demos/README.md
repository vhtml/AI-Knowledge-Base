# Demos · 唯一的代码区

这一层装下整个仓库的**可运行代码**：依赖、`.env`、`.venv` 都在这里。
笔记在 `AI/` `Business/` `Finance/`，代码只在这里——不散落到别处。

---

## 环境

Python 3.13，用 [`uv`](https://docs.astral.sh/uv/) 管理依赖。

```bash
cd Demos && uv sync
```

密钥只写进 `Demos/.env`（已在 `.gitignore` 中），提交的是 `Demos/.env.example`。
后端统一读 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（OpenAI 兼容接口）。

---

## 示例总览

| 示例 | 一句话 | 看点 | 自检 |
| --- | --- | --- | --- |
| `examples/agent_lab/` | **Agent 开发教程实验（17 章）** | 框架无关：手写 ReAct、工具设计、上下文预算、记忆、审批、评测 | `python -m agent_lab.selftest` |
| `examples/agent_lab/capstone/` | Agent 开发**毕业作品** | 把 17 章的护栏装进一个运行时 | `python -m agent_lab.capstone.selftest` |
| `examples/llm_lab/` | **LLM 地基教程实验（8 章）** | 贴着协议：消息、工具调用协议、结构化输出、流式、定价 | `python -m llm_lab.selftest` |
| `examples/langgraph_lab/` | LangGraph 深度教程实验（17 章） | 状态图、归约器、HITL、持久化、流式、中间件、MCP | `python -m langgraph_lab.selftest` |
| `examples/chat_cli/` | 命令行聊天助手 | 多轮对话、流式输出、历史存档、**上下文压缩** | `python -m chat_cli.selftest` |
| `examples/rag-demo/` | 检索问答（FastAPI + Vue 3） | 分块、检索、**GOLD 黄金集评估** | `python -m backend.rag eval`（需联网） |
| `examples/study_planner/` | 学习规划师（SSE 流式 + 记忆系统） | 对话流、目标与待办、**记忆分层与检索** | `python -m study_planner.backend.selftest` |
| `examples/life_assistant/` | 三工具 Agent | **工具选择**与**参数校验** | `python -m life_assistant.selftest` |
| `examples/skill_lab/` | Skill 制作七天教程 | 7 个可运行实验 + 校验脚本 + 毕业作品 | `python -m skill_lab.selftest` |
| `examples/ai_pm_lab/` | AI 产品经理八周教程实验 | 场景打分、评测门禁、单位经济、风险登记册 | `python -m ai_pm_lab selftest` |
| `examples/skills/` | 最小 skill 样例 | 一个 SKILL.md 长什么样 | — |
| `examples/retry-demo/` | 重试与熔断 | 指数退避、熔断器 | — |
| `examples/log-demo/` | 日志分级落盘 | loguru 分级 + 轮转 | — |
| `langgraph-demo/` | LangGraph 入门（旧草稿，已被 `langgraph_lab/` 取代） | `hello_graph.py` / `tool_basics.py` / `agent.py` | — |

---

## 跑教程实验

**所有教程实验都在 `Demos/examples` 下跑**，依赖从上一层解析：

```bash
cd Demos/examples
```

### Agent 开发教程（17 章，框架无关）

```bash
uv run --project .. python -m agent_lab list       # 列出各章
uv run --project .. python -m agent_lab ch07       # 跑第 7 章
uv run --project .. python -m agent_lab all        # 跑全部十七章
uv run --project .. python -m agent_lab selftest   # 自检
uv run --project .. python -m agent_lab ch07 --live  # 用真实模型跑第 7 章
```

### LLM 地基教程（8 章）

```bash
uv run --project .. python -m llm_lab list
uv run --project .. python -m llm_lab ch03
uv run --project .. python -m llm_lab all
uv run --project .. python -m llm_lab selftest
```

### 毕业作品自检

```bash
uv run --project .. python -m agent_lab.capstone.selftest
```

### 其它示例

```bash
uv run --project .. python -m skill_lab all        # 七天全跑
uv run --project .. python -m ai_pm_lab all         # 八周全跑
uv run --project .. python -m chat_cli.selftest
uv run --project .. python -m life_assistant.selftest
uv run --project .. python -m study_planner.backend.selftest
uv run --project .. python -m skill_lab.selftest
uv run --project .. python -m langgraph_lab.selftest
uv run --project .. python -m ai_pm_lab selftest
```

---

## 两条运行路径

| 路径 | 触发 | 模型来源 | 断言强度 | 什么时候用 |
| --- | --- | --- | --- | --- |
| **确定性路径** | 默认 | 脚本化假模型 | 严格（校验具体值） | 学习、回归、自检 |
| **真实路径** | `--live` | `Demos/.env` 里的 OpenAI 兼容端点 | 宽松（只校验形状） | 看真模型怎么跑 |

真实路径读 `Demos/.env` 的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。
**没配也能跑**——会明确告诉你要补哪个变量，而不是在 HTTP 层报 401。

---

## 起服务

```bash
# 检索问答后端
cd Demos && uv run uvicorn examples.rag-demo.backend.main:app --reload --port 8000

# 学习规划师后端
cd Demos && uv run uvicorn examples.study_planner.backend.main:app --reload --port 8000

# 前端（任选其一）
cd Demos/examples/rag-demo/frontend && npm install && npm run dev
cd Demos/examples/study_planner/frontend && npm install && npm run dev
```

---

## 目录约定

- `examples/` 下每个示例一个目录，自带 README 写清"怎么跑"
- 带前端的服务按 `backend/` 与 `frontend/` 分离
- 前端接口基地址统一读 `import.meta.env.VITE_API_BASE`，默认 `http://localhost:8000`
- 可测逻辑放同目录 `selftest.py` 或 `test_*.py`，保证能 `python -m` 直接运行，**不依赖 pytest**
- 日志与运行产物不要提交（`logs/` 与 `*.log` 已在 `.gitignore` 中）
