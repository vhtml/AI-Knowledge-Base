# 14 · Deep Agents：harness 视角

> **这篇要回答**：Deep Agents 到底是什么东西？我写一行 `create_deep_agent(...)`，它替我做了什么、又没替我做什么？我什么时候该用它，什么时候该退回 LangGraph？

---

## 1. harness 是什么：把"长任务会失败的地方"提前打包

第 1 章说过三层关系。Deep Agents 是上面那层——**harness**。

harness（挽具）这个词选得很准：它不是发动机，是**把马套上、缰绳、嚼子、马车都给你备好**。
你不用自己造车了，但你也不知道车能跑多快、跑哪条路。

```
┌──────────────────────────────────────────────────────────────┐
│  harness      Deep Agents                                     │
│  预置：规划 / 虚拟文件系统 / 子智能体 / 摘要 / 记忆 / skills   │
│  ↓ create_deep_agent(...)  →  预装好中间件的 create_agent     │
├──────────────────────────────────────────────────────────────┤
│  framework    LangChain  create_agent + AgentMiddleware       │
├──────────────────────────────────────────────────────────────┤
│  runtime      LangGraph  StateGraph / 检查点 / 中断 / 流式    │
└──────────────────────────────────────────────────────────────┘
```

`create_deep_agent` 的返回值就是一张 **`CompiledStateGraph`**——
和 `create_agent`、和你手写的 `StateGraph` 是同一个类型。
harness 没有引入新的运行时，它只是**预装了一叠中间件**。

---

## 2. 预置能力清单：逐条对应到底层机制

官方把 Deep Agents 的能力分成四组（[Overview](https://docs.langchain.com/oss/python/deepagents/overview)）。
下表把每一块和"它到底是靠什么实现的"对上：

| 能力 | 怎么开 | 底层机制 | 状态通道 / 工具 |
| --- | --- | --- | --- |
| **虚拟文件系统** | 默认 | `FilesystemMiddleware` + 可插拔 backend | 通道 `files`；工具 `ls` `read_file` `write_file` `edit_file` `delete` `glob` `grep` |
| **shell / 代码执行** | 配 sandbox backend | 沙箱 backend 提供 `execute`；解释器提供 `eval` | 工具 `execute` |
| **子智能体委派** | 默认 | `SubAgentMiddleware` 提供 `task` 工具 | 工具 `task`；默认带一个 `general-purpose` 子智能体 |
| **结构化规划** | **需 opt-in** | `TodoListMiddleware`（来自 `langchain.agents.middleware`） | 通道 `todos`；工具 `write_todos` |
| **上下文摘要 + 大结果卸载** | 默认 | `SummarizationMiddleware`（deepagents 自己的实现）+ 复用文件系统工具转存超限内容 | 内部通道 `_summarization_event`；通道 `files` |
| **持久记忆** | `memory=[...]` | `MemoryMiddleware` + `AGENTS.md` 文件 | 总是注入 system prompt |
| **skills** | `skills=[...]` | `SkillsMiddleware`，渐进披露 | 启动只读 frontmatter |
| **人工审批** | `interrupt_on={...}` | `HumanInTheLoopMiddleware` + LangGraph `interrupt()` | 需要 checkpointer |
| **提示词缓存** | 自动（Anthropic / Bedrock） | system prompt 静态段可缓存 | — |

**这张表就是本章的答案**：harness 替你写的，全是"长任务容易失败"的那几块——
规划、上下文膨胀、委派、文件落地、审批点。它**没替你写**的，是"业务规则决定走哪条路"。

---

## 3. `create_deep_agent` 的参数与返回值（[Customization 文档](https://docs.langchain.com/oss/python/deepagents/customization)）

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-6.1-sol",      # provider:model 字符串，或已初始化的模型实例
    tools=[search, fetch_url],        # 自定义函数 / LangChain 工具 / MCP 工具
    system_prompt="你是一个研究助手",
    memory=["./AGENTS.md"],           # 总是加载的持久记忆
    skills=["./skills/"],             # 按需加载的程序性知识
    backend=None,                     # 文件系统 backend，默认 StateBackend
    subagents=None,                   # 自定义子智能体
    middleware=(),                    # 额外中间件，合并进 Deep Agents 栈
    interrupt_on=None,                # 工具级人工审批
    permissions=None,                 # 文件路径级读写权限规则
    response_format=None,             # 结构化输出 schema
    state_schema=None,                # 自定义图状态
    context_schema=None,              # 每次运行的运行时上下文
    checkpointer=None, store=None,    # 持久化
)
```

返回值：`CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]`。
**它就是一张图**，可以 `invoke` / `stream` / `ainvoke`，可以加 checkpointer、可以放进别的图当子图。

`model` 有两个坑值得单独记：**必须支持工具调用**（官方明确要求，不支持的模型进不了 harness）；
可以传字符串也可以传实例，传字符串省事，但要记得 `init_chat_model` 走的是 provider 默认参数，
超时/重试/温度得自己配。

---

## 4. 状态通道：harness 的预置都落在这里

harness 预置的东西**不是藏在提示词里的**，是图状态上的一等公民通道：

| 通道 | 装什么 |
| --- | --- |
| `messages` | 对话历史（`DeltaChannel` 归约器） |
| `files` | 虚拟文件系统的内容 |
| `todos` | 结构化待办（配了 `TodoListMiddleware` 才有） |
| `structured_response` | 结构化输出（配了 `response_format` 才有） |
| `jump_to` | 内部控制用的跳转标记 |

配套实验里跑出来的一次真实状态，通道有 `messages` / `files` / `todos` /
`structured_response` / `jump_to`，外加三个内部通道
（`_blob_payloads`、`_summarization_event`、`_summarization_session_id`）。

> **为什么这件事重要**：因为待办和文件**可以被检查**。
> 你可以让 CI 断言"这次运行确实建了 3 条待办且都标成 completed"，
> 而不是只能靠读最终回答猜。这是 harness 相对于"一坨提示词"的最大工程价值。

---

## 5. 工具面：harness 到底给了模型哪些工具（[Tools 文档](https://docs.langchain.com/oss/python/deepagents/tools)）

不猜，直接看。实验里把编译后图里的 ToolNode 挖出来：

| 配置 | 模型看到的工具 |
| --- | --- |
| 默认 | `ls` `read_file` `write_file` `edit_file` `delete` `glob` `grep` `execute` `task` |
| 加 `TodoListMiddleware()` | 上面全部 + `write_todos` |
| 再加自定义工具 | 上面全部 + 你的工具 |

> **坑 1（反直觉）**：**结构化规划是 opt-in 的**。
> 从 `deepagents 0.7` 开始，`TodoListMiddleware` **不再默认挂上**——
> 不显式传 `middleware=[TodoListMiddleware()]`，工具面里就没有 `write_todos`，
> `todos` 通道也不存在。老文章里"Deep Agents 自带待办清单"的说法已经过期。

另一个容易忽略的：**未使用的内置工具，每一轮仍然把完整 schema 发给模型。**
工具多了，基线 prompt 就大。用 harness profile 的 `excluded_tools`
把不该出现的工具摘掉（**不要用 `excluded_middleware`**——
官方明确禁止移除 `FilesystemMiddleware` / `SubAgentMiddleware` 这类脚手架，移除会抛 `ValueError`）。

---

## 6. 委派：上下文隔离（[Subagents 文档](https://docs.langchain.com/oss/python/deepagents/subagents)）

`task` 工具让主智能体派活给子智能体，**主智能体只收到一份报告**。
这是 harness 解决"上下文膨胀"的主要手段。

| | `mode: "isolated"`（默认） | `mode: "fork"` |
| --- | --- | --- |
| 子智能体看到什么 | 只有你写的那段任务描述 | 父智能体的完整对话历史 + 任务描述 |
| system prompt | 必须自己写，不继承父的 | 继承父的，可追加补充说明 |
| 能不能再调 `task` | 能（会套娃） | 不能 |
| 适合 / 版本要求 | 边界清楚的专项调研（基础能力） | 父智能体已诊断完、子智能体接着干（需 `deepagents>=0.7.13`，**beta**） |

自定义子智能体就是一个字典：

```python
research_subagent = {
    "name": "researcher",                      # 主智能体调用时用的名字
    "description": "深入调研某个主题并返回结论",  # 主智能体据此决定委不委派
    "system_prompt": "你是研究员，只返回 500 字以内的结论",
    "tools": [internet_search],
}
agent = create_deep_agent(model=..., subagents=[research_subagent])
```
`description` 写得含糊，主智能体就**不会委派**——它会自己硬干。
这是子智能体最常见的失败原因，不是 bug，是接口设计。

---

## 7. 上下文管理：摘要 + 卸载（[Context engineering 文档](https://docs.langchain.com/oss/python/deepagents/context-engineering)）

长任务必然撑爆上下文窗口，harness 用了两个机制：

**卸载（offloading）**：工具调用的入参或返回值超过 **20,000 token** 阈值时，
内容被写进文件系统，对话里只留一个文件路径引用 + 前 10 行预览，
模型想细看再 `read_file` / `grep` 回来。

**摘要（summarization）**：上下文占用超过模型 `max_input_tokens` 的 **85%** 时触发，
保留最近 **10%** 的 token 作为近期上下文，更早的消息由模型压成结构化摘要
（会话意图、已产出物、下一步），同时把原始消息的文本渲染写进文件系统留档。
拿不到模型 profile 时的兜底是 **170,000 token 触发 / 保留 6 条消息**。
摘要**只改"发给模型的消息"**，不改持久化的 `state["messages"]`——回放和评测不受影响。
想提前手动压缩，可以加 `create_summarization_tool_middleware` 给模型一个
`compact_conversation` 工具（不关掉自动摘要，两套共用同一个摘要引擎）。

---

## 8. 记忆与 skills：一个总是加载，一个按需加载（[Memory 文档](https://docs.langchain.com/oss/python/deepagents/memory)）

这两个最容易混：

| | `memory=[...]` | `skills=[...]` |
| --- | --- | --- |
| 载体 | `AGENTS.md` 文件 | 目录下的 `SKILL.md` |
| 加载时机 | **总是**注入 system prompt | 启动只读 frontmatter，**需要时才读全文** |
| 适合 | 编码风格、偏好、项目约定 | 具体工作流、领域知识 |
| 内容多了会怎样 | 直接撑爆上下文 | 上下文保持精简 |

**skills 用的是 [Agent Skills 标准](https://agentskills.io/)**：
渐进披露——启动时只读 `SKILL.md` 的 frontmatter（name + description），
任务对上了才去读正文。一个 skill 目录还可以带脚本、模板、参考文档。

**长期记忆（跨会话）**是第三件事：默认的 `StateBackend` 只在单个 thread 内存活。
要跨线程持久，得用 `CompositeBackend` 把 `/memories/` 这类路径路由到
`StoreBackend`（LangGraph store）：

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model=...,
    store=InMemoryStore(),
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=lambda rt: ("memories",))},
    ),
)
```

**安全提醒**：能写的记忆就是**别的用户能读到的记忆**。
共享的组织级记忆应当用权限规则设成只读，或者在写入前加人工审批——
否则一个用户就能通过记忆往别人的上下文里塞指令。
---

## 9. 人工审批（[Human-in-the-loop 文档](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)）

```python
agent = create_deep_agent(
    model=...,
    tools=[remove_file, fetch_file],
    interrupt_on={
        "remove_file": True,                                  # 默认：批准/编辑/拒绝/回应
        "fetch_file": False,
        "notify_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=MemorySaver(),   # ← 必需
)
```
| 决策 | 含义 |
| --- | --- |
| `approve` | 按原参数执行 |
| `edit` | 改参数后再执行 |
| `reject` | 不执行，把拒绝理由回给模型 |
| `respond` | 把人的回复当作工具结果返回（"问用户"型工具用） |

三个硬要求：**必须有 checkpointer**（中断期间状态要存得住）、
**恢复必须用同一个 `thread_id`**、决策列表的**顺序必须和 `action_requests` 一一对应**。
还能按参数条件中断：`InterruptOnConfig` 加 `when` 谓词，
比如"只有写到 `/workspace/` 之外才中断"。

---

## 10. harness 的代价

harness 不是免费的，四笔账要算清：

### 10.1 绑定生态

你的 Agent 从此依赖 Deep Agents 的中间件栈，官方明确**不许**移除 `FilesystemMiddleware` /
`SubAgentMiddleware` 这类脚手架。好处是升级能拿到新能力；坏处是**你很难只拿走其中一块**。

### 10.2 "信任 LLM"这条边界要自己划

harness 默认给了模型一整套文件写入、编辑、删除（以及 sandbox 下的 shell 执行）能力。
**模型看到工具 ≠ 模型有权限用工具。** 这条边界只有你能画：
`permissions=[...]` 用声明式规则限制路径（`allow` / `deny` / `interrupt`），
`interrupt_on` 在关键工具前设人工闸门，`FilesystemMiddleware(tools=[...])` 只暴露只读工具子集，
sandbox backend 和 `FilesystemBackend` 的目录边界同样要划。

**默认配置是一个"模型想删就删"的 Agent。** 上生产前先把它锁上。

### 10.3 预设偏重时要降层

当"每条边由业务规则决定"时，harness 是负担：它给的是**模型自主的循环**，
不是你要的确定性流程。降层用 LangGraph 的信号：

| 症状 | 降层 |
| --- | --- |
| 分支条件是业务规则，不是模型判断 | 手写 `StateGraph` + `add_conditional_edges` |
| 每一步的顺序必须固定、可审计 | 手写图，节点即步骤 |
| 需要精确的重试/超时/补偿语义 | 手写节点（第 5 章） |
| 工具集固定、任务单一 | `create_agent`（第 12 章），甚至纯函数 |
| 只是要多几步 LLM 调用 | 普通 Python 函数 |

**降层不等于倒退。** 一个 20 行的确定性审批流，用 harness 反而更慢、更难测、更难解释。
配套实验里两者都有：大额发票走 `screen → review → settle`，小额直接 `screen → settle`。

### 10.4 上下文和 token 成本

harness 的能力全都要 token：工具 schema 每轮都发、子智能体要重建一套上下文、
摘要要额外调一次模型。**harness 是用 token 换可靠性**。

---

## 11. 什么时候不该用 Deep Agents

| 场景 | 为什么别用 |
| --- | --- |
| 任务简单，三步以内 | harness 的脚手架比任务本身还复杂 |
| 需要精确控制每条边 | harness 的循环由模型决定，你只能加闸门不能改路 |
| 固定工作流，只是每步要调 LLM | 普通 Python 函数 + 几次 `invoke` 更清楚 |
| 强合规流程，每一步都要留证 | 自己写的节点能控制留什么、留多少 |
| 模型不支持工具调用 / 延迟敏感 | harness 的地基是工具调用，子智能体和摘要都是额外开销 |

---

## 12. 四个必须知道的坑

### 坑 1：规划是 opt-in 的

见第 5 节。`deepagents 0.7` 起 `write_todos` 不再默认挂载。

### 坑 2：自定义模型的 `_llm_type` 必须是**属性**

LangChain 的摘要中间件按属性读它
（`if model._llm_type.startswith("anthropic-chat"):`，
见 `langchain/agents/middleware/summarization.py`），
而 `langchain_core` 里 `_llm_type` 是 `@property`。
如果你的模型把它定义成**普通方法**，`model._llm_type` 拿到的是函数对象，
`.startswith` 直接抛 `AttributeError: 'function' object has no attribute 'startswith'`。
配套实验里用的脚本化模型就踩了这个坑，解决办法是在实例上盖一个同名属性。

**自定义 `BaseChatModel` 时，`_llm_type` 一定写成 property。**

### 坑 3：memory 每次都进 prompt，skills 不是

`memory` 是**总是注入**的，往里塞大段文档 = 每轮都付这个 token。
把"偶尔才用得上的工作流"塞进 memory 是常见误用——那属于 skills。

### 坑 4：恢复运行会重跑工具

审批中断后用 `Command(resume=...)` 恢复，**工具从头重跑**。
和第 13 章 MCP 的 elicitation 是同一个坑：有副作用的工具必须自己保证幂等。

---

## 配套实验

`Demos/examples/langgraph_lab/lessons/ch14_deep_agents.py`

- 断言 `create_deep_agent` 返回 `CompiledStateGraph`，
  且状态通道里有 `files` 和 `todos`；
- 枚举工具面：默认有文件系统全套 + `task`；加 `TodoListMiddleware()` 才多出 `write_todos`；
- 驱动 `write_todos` → `todos` 通道真的出现两条带状态的待办；
- 驱动 `write_file` → `files` 通道出现 `/notes.md` 且内容原样落库；
- 挂一个自定义 `@tool` → 真的被调用并返回结果；
- 驱动 `task` → 子智能体的报告回到主智能体；
- 对照组：同样的确定性需求用原生 LangGraph 写清楚每条边。

跑法：`cd Demos/examples && uv run --project .. python -m langgraph_lab ch14`

---

## 一句话结论

> **Deep Agents 是 harness，不是新运行时**：它把"长任务容易崩的几块"——规划、文件落地、
> 子智能体委派、上下文摘要、持久记忆、人工审批——预装成一叠中间件，
> 返回给你的仍然是一张 `CompiledStateGraph`。
> 它替你写的是**脚手架**，替你做不了的是**决策**：流程该走哪条路、哪些文件能写、
> 哪些工具要人点头，这些边界必须你自己画。
> 任务简单或者要精确控制每条边时，退回 LangGraph 不是倒退，是更便宜、更快、更可测的选择。
