# AGENTS.md

个人 AI 知识库仓库。`AI/` `Business/` `Finance/` 沉淀知识笔记，`Demos/` 存放与教程配套的可运行实验，`scripts/` 放仓库自己的维护工具。

## 目录职责

| 目录 | 放什么 | 判断标准 | 例子 |
| --- | --- | --- | --- |
| `AI/` `Business/` `Finance/` | 按领域分类的**知识笔记** | 读给人看的判断依据，不是给人跑的程序 | `AI/AI-Product/`、`AI/LangGraph/` |
| `Demos/` | 与教程配套的**可运行实验** | 读者会去跑它，跑挂了等于没学会 | `Demos/examples/langgraph_lab/` |
| `scripts/` | 仓库**维护工具** | 只服务于仓库自己，不产出教学内容 | `scripts/check_ai_product_notes.py` |
| `Projects/` | 独立项目工作区 | 一个项目一个子目录 | |
| `docs/` | 方案与计划文档 | 过程性文档，不是知识沉淀 | `docs/superpowers/plans/` |

### 三条最容易搞混的边界

**笔记 vs 实验**：`AI/` 里的笔记**不放可运行代码**，代码一律在 `Demos/`。
但笔记可以**引用**实验——写清路径与命令即可。

**实验 vs 维护脚本**：两者都是代码，都不在 `AI/` 里，靠一个问题分开——
**这个工具是给读者用的，还是给仓库自己用的？**
给读者用的（能跑、有学习目标、有毕业作品）放 `Demos/examples/`；
给仓库自己用的（校验、检查、迁移）放 `scripts/`。
所以 `Demos/examples/` 下每个目录都该对应一份**可跑的学习材料**，
维护脚本混进去会让读者以为这也是个教程。

**维护脚本 vs 示例**：写法也不同。
维护脚本是**单文件**工具，只用标准库、不引入第三方依赖，在仓库根直接跑：
`python3 scripts/<工具>.py`。
示例是**包**，依赖装在 `Demos/` 这层，在 `Demos/examples` 下跑：
`uv run --project .. python -m <包>`。

### 其它约定

- `Demos/examples/` 下每个示例一个目录，自带 README 写清"怎么跑"；带前端的服务按 `backend/` 与 `frontend/` 分离
- 空目录里放一个 `.gitkeep`——空目录也要能被 git 跟踪，否则 clone 出来就散架了
- 笔记里不放大段可运行代码；代码也不写进笔记
- 示例的自检可以顺带确认**配套笔记在位**（轻量，如 `check_notes_exist()`）；
  **深度结构校验**——链接是否失效、表格列数是否错位、代码围栏是否闭合——
  统一交给 `scripts/` 里的工具，不要每个示例各写一份

## 环境

- Python 3.13，用 `uv` 管理依赖（`Demos/pyproject.toml`）；前端用 npm
- 依赖装在 `Demos/` 这一层：`cd Demos && uv sync`
- 跑示例统一在 `Demos/examples` 下：`uv run --project .. python -m <包>`
- 跑仓库维护脚本统一在仓库根：`python3 scripts/<工具>.py`（只用标准库，不依赖 `Demos/` 的环境）
- 各示例的具体启动命令写在它自己的 README 里，不在这里重复
- 架构图以 `.drawio` 为可编辑源头，`.svg` / `.png` 由它渲染——改图改 `.drawio`

## 代码风格

- Python：`from __future__ import annotations`；内置泛型写法（`list[dict]`、`dict[str, str]`）；数据模型用 `@dataclass`
- 模块顶部写中文 docstring，说明"这个模块解决什么问题"，不是复述函数名
- 公开函数必须标注参数与返回类型；注释写"为什么"，不写"做了什么"
- 前端：Vue 3 `<script setup>` + Vite；接口基地址统一读 `import.meta.env.VITE_API_BASE`
- 仓库没有 lint / formatter 配置，新代码沿用邻近文件的风格；确需引入时用 `uv add --dev` / `npm i -D` 并提交锁文件

## 测试与自检

- 可测逻辑放同目录 `selftest.py` 或 `test_*.py`，保证能 `python -m` 直接运行，不依赖 pytest
- 改了哪个模块就跑对应自检；提交前至少跑一遍相关自检，改动前端再加一次 `npm run build`
- 改了 `AI/AI-Product/` 下的笔记，额外跑一次 `python3 scripts/check_ai_product_notes.py`——它查的是结构（链接失效、表格错位、围栏未闭合），肉眼看一遍很容易漏
- 有确定答案的逻辑（检索质量、参数校验这类）配黄金集评估，结果写进日志文件，方便对比改动前后

## 提交规范

- 从 `main` 切分支，不直接推 `main`
- 提交信息用中文，一句话说清目的，必要时带范围前缀（例：`chat_cli: 超预算时截断摘要而非丢弃`）
- 一个提交只做一件事；无关的重排、格式化不要和功能改动混在一起
- 每完成一个实现或配置就立刻提交一次，不要攒到最后；提交只落在本地，是否 push 另行指示
- 新增示例目录时同步补 README，写清"怎么跑"

## 密钥与环境变量

- 密钥只写进 `Demos/.env`（已在 `.gitignore` 中），提交的是 `.env.example`
- 后端统一读 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`（OpenAI 兼容接口），用 `python-dotenv` 的 `load_dotenv()` 加载
- 前端用 `VITE_API_BASE` 指向后端，默认 `http://localhost:8000`
- 不要提交 `.env`、`.venv/`、`node_modules/`、`dist/`、`.DS_Store`、运行日志（`logs/` 与 `*.log` 已在 `.gitignore` 中）
