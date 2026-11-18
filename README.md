# AI-Knowledge-Base

个人知识库 + 可运行实验场。

用 `AI/` `Business/` `Finance/` 沉淀**判断依据**（知识笔记），
用 `Demos/` 沉淀**能跑起来的代码**（每个示例都能独立运行、独立自检）。

> 这不是教程仓库，也不是框架项目。它是一个人的工作台：
> 想清楚的东西写进笔记，写不清楚的东西做成实验，实验跑通了再回头写成笔记。

---

## 目录结构

| 目录 | 放什么 | 现状 |
| --- | --- | --- |
| `AI/` | AI 领域笔记：Agent / LLM / MCP / Prompt / RAG / Skill / 产品经理 / Infra 等 | `Skill/` `LangGraph/` `AI-Product/` `AI-Infra/` 各有一套完整笔记，其余待填 |
| `Business/` | 商业知识：Growth / Marketing / Pricing / SaaS / Sales | 骨架 |
| `Finance/` | 金融知识：Macro / Equity / Portfolio / Valuation | 骨架 |
| `Projects/` | 独立项目工作区，一个项目一个子目录 | 骨架 |
| `Demos/` | **唯一的代码区**：Python 依赖、`.env`、`.venv` 都在这一层 | `examples/` 10 个示例 + `langgraph-demo/` |
| `docs/` | 方案与计划文档 | `superpowers/plans/` |

`AI/` `Business/` `Finance/` `Projects/` 下每个分类目录都带 `.gitkeep`——
空目录也要能被 git 跟踪，否则 clone 出来就散架了。

---

## 主要示例

| 示例 | 一句话 | 看点 | 自检 |
| --- | --- | --- | --- |
| `Demos/examples/infra_lab/` | **AI Infra 推理服务教程实验（18 章）** | 显存账、Roofline、KV cache、连续批处理、量化、调度、容量规划 | `python -m infra_lab.selftest` |
| `Demos/examples/infra_lab/capstone/` | AI Infra **毕业作品** | 迷你推理服务：分页 KV + 连续批处理 + 前缀缓存 + 亲和路由 | `python -m infra_lab.capstone.selftest` |
| `Demos/examples/langgraph_lab/` | **LangGraph 深度教程实验（17 章）** | 状态图、归约器、HITL、持久化、流式、中间件、MCP | `python -m langgraph_lab.selftest` |
| `Demos/examples/chat_cli/` | 命令行聊天助手 | 多轮对话、流式输出、历史存档、**上下文压缩** | `python -m chat_cli.selftest` |
| `Demos/examples/rag-demo/` | 检索问答（FastAPI + Vue 3） | 分块、检索、**GOLD 黄金集评估** | `python -m backend.rag eval`（需联网） |
| `Demos/examples/study_planner/` | 学习规划师（SSE 流式 + 记忆系统） | 对话流、目标与待办、**记忆分层与检索** | `python -m study_planner.backend.selftest` |
| `Demos/examples/life_assistant/` | 三工具 Agent | **工具选择**与**参数校验** | `python -m life_assistant.selftest` |
| `Demos/examples/skill_lab/` | Skill 制作七天教程 | 7 个可运行实验 + 校验脚本 + 毕业作品 | `python -m skill_lab.selftest` |
| `Demos/examples/ai_pm_lab/` | **AI 产品经理八周教程实验（16 章）** | 场景打分、评测门禁、单位经济、风险登记册 | `python -m ai_pm_lab selftest` |
| `Demos/examples/skills/` | 最小 skill 样例 | 一个 SKILL.md 长什么样 | — |
| `Demos/examples/retry-demo/` | 重试与熔断 | 指数退避、熔断器 | — |
| `Demos/examples/log-demo/` | 日志分级落盘 | loguru 分级 + 轮转 | — |
| `Demos/langgraph-demo/` | LangGraph 入门（旧草稿，已被 `langgraph_lab/` 取代） | `hello_graph.py` / `tool_basics.py` / `agent.py` | — |

`Demos/examples/agent1.py` ~ `agent5.py` 是**学习草稿**：从最小模型调用一路写到工具调用和完整逻辑。
它们不追求结构，但每一步都留着当时踩的坑。

表里的自检命令都在 `Demos/examples` 目录下用 `uv run --project ..` 执行，
完整写法见下方「跑各示例的自检」。

---

## 快速开始

Python 3.13 + [`uv`](https://docs.astral.sh/uv/)，前端用 npm。

```bash
# 1. 装依赖（依赖、.env、.venv 都在 Demos/ 这一层）
cd Demos && uv sync

# 2. 跑自检
cd Demos/examples && uv run --project .. python -m chat_cli.selftest
```

### 起服务

```bash
# 检索问答后端
cd Demos && uv run uvicorn examples.rag-demo.backend.main:app --reload --port 8000

# 学习规划师后端
cd Demos && uv run uvicorn examples.study_planner.backend.main:app --reload --port 8000

# 前端（任选其一）
cd Demos/examples/rag-demo/frontend && npm install && npm run dev
cd Demos/examples/study_planner/frontend && npm install && npm run dev
```

### 跑 skill 教程实验

```bash
cd Demos/examples
uv run --project .. python -m skill_lab all       # 七天全跑
uv run --project .. python -m skill_lab day3      # 只跑第 3 天
uv run --project .. python -m skill_lab.selftest  # 自检
```

全部实验离线运行：不联网、不消耗 token、不调用模型。跑挂了就是没学会。

### 跑 AI 产品经理教程实验

```bash
cd Demos/examples
uv run --project .. python -m ai_pm_lab all        # 八周全跑
uv run --project .. python -m ai_pm_lab week3       # 只跑第 3 周
uv run --project .. python -m ai_pm_lab selftest    # 自检
```

### 跑 AI Infra 推理服务教程实验

```bash
cd Demos/examples
uv run --project .. python -m infra_lab list        # 列出 18 章
uv run --project .. python -m infra_lab ch07        # 跑第 7 章
uv run --project .. python -m infra_lab all         # 跑全部 18 章
uv run --project .. python -m infra_lab capstone     # 跑毕业作品
uv run --project .. python -m infra_lab.selftest    # 全量自检
```

全部离线运行：不联网、不消耗 token、不需要 GPU。跑挂了就是没学会。

### 跑各示例的自检

自检大多不联网，但依赖在 `Demos/` 这一层，用裸 `python3` 会缺包：

```bash
cd Demos/examples
uv run --project .. python -m infra_lab.selftest
uv run --project .. python -m chat_cli.selftest
uv run --project .. python -m life_assistant.selftest
uv run --project .. python -m study_planner.backend.selftest
uv run --project .. python -m skill_lab.selftest
uv run --project .. python -m langgraph_lab.selftest
uv run --project .. python -m ai_pm_lab selftest

# AI Infra 毕业作品自检
uv run --project .. python -m infra_lab.capstone.selftest

# rag-demo 的检索评估需要联网调 embedding，跑在它自己的目录下
cd Demos/examples/rag-demo && uv run --project ../.. python -m backend.rag eval
```

---

## 密钥与配置

- 密钥只写进 `Demos/.env`（已在 `.gitignore` 中），**提交的是 `Demos/.env.example`**
- 后端统一读 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（OpenAI 兼容接口）
- 前端用 `VITE_API_BASE` 指向后端，默认 `http://localhost:8000`

---

## 协作约定

**[`AGENTS.md`](AGENTS.md) 是本仓库的唯一协作规范**，包括目录职责、代码风格、提交规范、密钥规则。

要点：

- `Demos/` 是唯一的代码区，笔记里不放大段可运行代码
- 从 `main` 切分支，不直接推 `main`
- 提交信息用中文，一句话说清目的，必要时带范围前缀（`chat_cli: 超预算时截断摘要而非丢弃`）
- 一个提交只做一件事；每完成一个实现就立刻提交，不要攒到最后
- 提交前至少跑一遍对应的自检

---

## 看哪里

- 想学 **AI Infra 推理服务**（显存账、KV cache、连续批处理、量化、调度、容量规划）→ [`AI/AI-Infra/`](AI/AI-Infra/README.md)，配 `Demos/examples/infra_lab/`
- 想学 **AI 产品经理** → [`AI/AI-Product/`](AI/AI-Product/README.md)，配 `Demos/examples/ai_pm_lab/`
- 想学 **Skill 怎么写** → [`AI/Skill/`](AI/Skill/README.md)，配 `Demos/examples/skill_lab/`
- 想学 **LangGraph 及其生态** → [`AI/LangGraph/`](AI/LangGraph/README.md)，配 `Demos/examples/langgraph_lab/`
- 想看**上下文怎么管** → `Demos/examples/chat_cli/README.md` 的上下文管理章节
- 想看**检索质量怎么评** → `Demos/examples/rag-demo/backend/rag.py` 的 `GOLD` 集
- 想看**记忆怎么分层** → `Demos/examples/study_planner/backend/memory*.py`
- 想看**推理服务怎么调度** → `Demos/examples/infra_lab/capstone/miniserve.py` 的连续批处理循环

### 推荐的学法

AI Infra 教程的推荐顺序是**从地基往上**——每一层都建立在下一层之上：

```
模型协议地基（AI/LLM/，另一分支建设中）
      ↓
应用层运行时（AI/Agent/，另一分支建设中）
      ↓
服务层（AI/AI-Infra/ ← 你在这里）
```

理由：AI Infra 讲的是"怎么把模型高效地服务出去"，
不懂模型协议和 Agent 的请求形态，容量规划就没有输入。

AI Infra 教程本身**不依赖**上面两层——18 章可以独立学完，
只需要知道 Transformer 的基本结构和 KV cache 的来历（第 5 章会重新讲一遍）。
