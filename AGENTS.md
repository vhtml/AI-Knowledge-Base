# AGENTS.md

个人 AI 知识库仓库：`AI/` `Business/` `Finance/` 沉淀知识笔记，`Demos/` 存放所有可运行的实验性代码（Python 后端 + Vue 前端）。

## 目录结构

- `AI/` — AI 领域笔记：Agent / LLM / MCP / Prompt / RAG / AI-Business 等分类目录
- `Business/` — 商业知识：Growth / Marketing / Pricing / SaaS / Sales
- `Finance/` — 金融知识：Macro / Equity / Portfolio / Valuation
- `Projects/` — 独立项目工作区，每个项目一个子目录
- `Demos/` — **唯一的代码区**；Python 依赖、`.env`、`.venv` 都在这一层
- `Demos/examples/` — 可独立运行的示例：`chat_cli/`（命令行聊天）、`rag-demo/`（检索问答）、`study_planner/`（学习规划师）、`skills/`、`retry-demo/` 等
- `Demos/examples/rag-demo/backend/` — FastAPI 服务；`frontend/` — Vue 3 + Vite 前端
- `Demos/examples/study_planner/backend/` — **该示例的全部 Python**（FastAPI 服务 SSE 流式对话 + 领域内核 + 命令行入口）；`frontend/` — Vue 3 + Vite + Pinia 前端

## 环境与命令

Python 用 `uv` 管理依赖（Python 3.13），前端用 `npm`。

- 安装依赖：`cd Demos && uv sync`
- 自检：`cd Demos/examples && uv run --project .. python -m chat_cli.selftest`
- 启动后端：`cd Demos && uv run uvicorn examples.rag-demo.backend.main:app --reload --port 8000`
- 启动学习规划师后端：`cd Demos && uv run uvicorn examples.study_planner.backend.main:app --reload --port 8000`
- 学习规划师命令行：`cd Demos/examples && uv run --project .. python -m study_planner.backend`（固定规则模式，不联网）
- 启动前端：`cd Demos/examples/rag-demo/frontend && npm install && npm run dev`（http://localhost:5173）
- 启动学习规划师前端：`cd Demos/examples/study_planner/frontend && npm install && npm run dev`（http://localhost:5173）
- 前端构建：`npm run build`（产物在 `dist/`，不提交）
- 架构图：`chat_cli/architecture.drawio`、`study_planner/architecture.drawio` 是可编辑源头，`.svg` / `.png` 由它渲染——改图请改 `.drawio`

## 代码风格

- Python：`from __future__ import annotations`；内置泛型写法（`list[dict]`、`dict[str, str]`）；数据模型用 `@dataclass`
- 模块顶部写中文 docstring，说明"这个模块解决什么问题"，不是复述函数名
- 公开函数必须标注参数与返回类型；注释写"为什么"，不写"做了什么"
- 前端：Vue 3 `<script setup>` + Vite；接口基地址统一读 `import.meta.env.VITE_API_BASE`
- 仓库目前没有 lint / formatter 配置，新代码沿用邻近文件的风格；确需引入时用 `uv add --dev` / `npm i -D` 并提交锁文件

## 测试与自检

- `chat_cli` 用 `selftest.py` 自检（不依赖 pytest）：`python -m chat_cli.selftest`。改动 `messages.py` / `context.py` / `storage.py` 后必须跑通
- 其它示例暂无测试；新增可测逻辑时在同目录放 `test_*.py` 或 `selftest.py`，保证能 `python -m` 直接运行
- RAG 检索质量由 `backend/rag.py` 内置的 `GOLD` 黄金集评估：`cd Demos/examples/rag-demo && python -m backend.rag eval`（需联网调 embedding），结果写入 `data/eval_log.json`
- 提交前至少跑一遍 `python -m chat_cli.selftest`，改动前端时再加一次 `npm run build`

## 提交规范

- 从 `main` 切分支，不直接推 `main`
- 提交信息用中文，一句话说清目的，必要时带范围前缀（例：`chat_cli: 超预算时截断摘要而非丢弃`）
- 一个提交只做一件事；无关的重排、格式化不要和功能改动混在一起
- 每完成一个实现或配置就立刻提交一次，不要攒到最后；提交只落在本地，是否 push 另行指示
- 新增示例目录时同步补 README，写清"怎么跑"

## 密钥与环境变量

- 密钥只写进 `Demos/.env`（已在 `.gitignore` 中），提交的是 `.env.example`
- 后端统一读 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（OpenAI 兼容接口），用 `python-dotenv` 的 `load_dotenv()` 加载
- `study_planner` 后端按 `LLM_API_KEY` 是否存在自动切换：有 key 走模型模式（模型出结构化决策 → 代码执行 → 模型流式回复），没 key 或模型调不通退回规则模式；`GET /health` 返回 `mode: model | rule`
- 前端用 `VITE_API_BASE` 指向后端，默认 `http://localhost:8000`
- 不要提交 `.env`、`.venv/`、`node_modules/`、`dist/`、`.DS_Store`、运行日志（`logs/` 与 `*.log` 已在 `.gitignore` 中）
