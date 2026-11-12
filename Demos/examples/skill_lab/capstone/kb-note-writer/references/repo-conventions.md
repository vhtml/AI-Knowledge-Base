# 本仓库笔记约定

> 这份参考是 `kb-note-writer` 的判定依据。写笔记前对一遍，不确定的地方以仓库根目录的
> `AGENTS.md` 为准（那份是唯一权威来源）。

## 目录结构

```
AI/          AI 领域笔记：Agent / LLM / MCP / Prompt / RAG / AI-Business 等分类目录
Business/    商业知识：Growth / Marketing / Pricing / SaaS / Sales
Finance/     金融知识：Macro / Equity / Portfolio / Valuation
Projects/    独立项目工作区，每个项目一个子目录
Demos/       唯一的代码区；Python 依赖、.env、.venv 都在这一层
```

**笔记放 `AI/` `Business/` `Finance/`，代码放 `Demos/`。** 两者不要混在一起——
一篇笔记里出现大段可运行代码，说明它该被拆到 `Demos/examples/` 下的示例里。

## 分类归属判断

| 经验类型 | 落点 |
| --- | --- |
| 模型能力、Agent 架构、推理技巧 | `AI/LLM/`、`AI/Agent/`、`AI/Prompt/` |
| 检索、向量库、RAG 管线 | `AI/RAG/` |
| 工具协议、上下文协议 | `AI/MCP/` |
| 框架用法（LangGraph 等） | `AI/LangGraph/` |
| 定价、增长、销售漏斗 | `Business/` 下对应分类 |
| 宏观、估值、组合 | `Finance/` 下对应分类 |
| 跟某个具体项目强绑定的过程记录 | `Projects/<项目名>/` |

拿不准就**先看同主题现有笔记在哪**，不要新建分类目录。

## 笔记写作约定

- 中文正文，术语保留英文原词（不要强行翻译 API、库名、协议名）
- 模块顶部有中文 docstring，说明"这个模块解决什么问题"，不是复述函数名
- 公开函数标注参数与返回类型；注释写"为什么"，不写"做了什么"
- Python 用 `from __future__ import annotations`，内置泛型写法（`list[dict]`、`dict[str, str]`）
- 数据模型用 `@dataclass`

## 提交约定

- 从 `main` 切分支，不直接推 `main`
- 提交信息用中文，一句话说清目的，必要时带范围前缀（如 `chat_cli: 超预算时截断摘要而非丢弃`）
- 一个提交只做一件事；无关的重排、格式化不要和功能改动混在一起
- 每完成一个实现或配置就立刻提交一次，不要攒到最后

## 密钥与环境变量

- 密钥只写进 `Demos/.env`（已在 `.gitignore` 中），提交的是 `.env.example`
- 后端统一读 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`
- 前端用 `VITE_API_BASE` 指向后端
- **不要提交**：`.env`、`.venv/`、`node_modules/`、`dist/`、`.DS_Store`、`logs/`、`*.log`

## 敏感信息红线

以下内容一律不进笔记、不进代码、不进提交信息：

- 真实 API Key、访问令牌、密码
- 内网地址、数据库连接串、真实客户数据
- 未公开的营收、用户规模等内部数据

需要说明某个配置项时，写**它叫什么、放在哪、怎么读**，不写具体值。
