# 12 · LangChain 1.x：create_agent 与中间件

> **这篇要回答**：`create_agent` 到底帮你省掉了什么？中间件在哪个时机动手？怎么把一个 Agent 塞进自己的图里？

---

## 1. `create_agent` 是"藏起来的图"

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=[...], system_prompt="你是一个助手")
```

十行起步。但**它返回的不是什么神秘对象**——第 1 章的结论在这里落地：

```python
from langgraph.graph.state import CompiledStateGraph
assert isinstance(agent, CompiledStateGraph)   # 它就是一张编译好的图
```

**所以 LangChain 的 Agent 和你手写的 LangGraph 图，是同一类东西。**
区别只是：官方替你画好了"模型 → 工具 → 模型"这个循环，还留了挂钩子的位置。

---

## 2. 签名逐项看

```python
create_agent(
    model,                       # 必填
    tools=None,
    system_prompt=None,
    middleware=(),
    response_format=None,        # 结构化输出
    state_schema=None,           # 扩展状态
    context_schema=None,
    checkpointer=None,           # ← 持久化从这里进
    store=None,                  # ← 长期记忆从这里进
    interrupt_before=None,
    interrupt_after=None,
    debug=False,
    name=None,
    cache=None,
    transformers=None,
)
```

**注意 `checkpointer` 和 `store` 是直接参数**——
第 7 章讲的持久化，对 `create_agent` 来说就是加一个参数的事。

---

## 3. 中间件：两种钩子

中间件是**在 Agent 循环的特定位置动手**的机制。

### 节点式钩子（顺序执行）

| 钩子 | 什么时候跑 |
| --- | --- |
| `before_agent` | Agent 开始前（每次调用一次） |
| `before_model` | 每次调模型前 |
| `after_model` | 每次模型返回后 |
| `after_agent` | Agent 结束后（每次调用一次） |

适合：日志、校验、改状态、计数。

### 包裹式钩子（环绕执行）

| 钩子 | 什么时候跑 |
| --- | --- |
| `wrap_model_call` | 环绕每次模型调用 |
| `wrap_tool_call` | 环绕每次工具调用 |

适合：重试、降级、缓存、动态换模型、动态换工具。
**你决定 handler 被调用 0 次（短路）、1 次（正常）还是多次（重试）。**

---

## 4. 执行顺序（最容易记错的地方）

```
before_agent:  m1 → m2 → m3          （正序）
Agent 循环开始
before_model:  m1 → m2 → m3          （正序）
wrap_model:    m1 ┐
                m2 ┤  嵌套（m1 包住 m2 包住 m3）
                m3 ┘  → 模型
after_model:   m3 → m2 → m1          （逆序！）
Agent 循环结束
after_agent:   m3 → m2 → m1          （逆序）
```

> **一句话记法**：before 正着进，after 倒着出，wrap 一层套一层。

---

## 5. 中间件怎么改状态

两种钩子改状态的方式**不一样**：

| 钩子类型 | 返回什么 |
| --- | --- |
| 节点式 | 直接返回 `dict`，按图的归约器合并 |
| 包裹式（模型） | 返回 `ExtendedModelResponse(model_response=..., command=Command(update={...}))` |
| 包裹式（工具） | 直接返回 `Command(update={...})` |

`Command` 会**穿过归约器**——所以 `messages` 是**追加**而不是替换。

多个中间件同时写同一个非归约器字段时：**外层优先**（outer wins）。
想让"后写覆盖"就用 last-wins 归约器；想让消息累加就用加法归约器。

---

## 6. 提前退出：`jump_to`

节点式钩子可以返回 `{"jump_to": "end"}` 提前结束 Agent。

| 目标 | 跳到哪 |
| --- | --- |
| `'end'` | Agent 结束（或第一个 `after_agent` 钩子） |
| `'model'` | 回到模型节点 |
| `'tools'` | 跳到工具节点 |

需要声明能跳：`@hook_config(can_jump_to=["end"])`。

---

## 7. 内置中间件清单

| 类别 | 内置件 |
| --- | --- |
| 上下文 | `SummarizationMiddleware`（摘要）、`ContextEditingMiddleware`（裁剪）、`TodoListMiddleware`（待办） |
| 人工介入 | `HumanInTheLoopMiddleware`（工具审批） |
| 可靠性 | `ModelFallbackMiddleware`（降级）、`ModelRetryMiddleware`、`ToolRetryMiddleware`、`ToolErrorMiddleware` |
| 限额 | `ModelCallLimitMiddleware`、`ToolCallLimitMiddleware` |
| 安全 | `PIIMiddleware`（个人信息检测）、`ModelRefusalMiddleware`（识别拒答） |
| 选型 | `LLMToolSelectorMiddleware`、`ModelRouterMiddleware`、`AutoModeMiddleware` |
| 工具环境 | `ShellToolMiddleware`、`FilesystemFileSearchMiddleware`、`ProviderToolSearchMiddleware` |

**内置件优先**——能省一次自己写钩子的机会。

---

## 8. 中间件不是独立运行时

这条很重要：

> **中间件的钩子跑在 `create_agent` 返回的那张 LangGraph 图内部。**
> 把整个 Agent 当成一个节点塞进你自己的 `StateGraph`，
> 所有钩子照常运行。

```python
graph = (
    StateGraph(AgentState)
    .add_node("classify", classify_node)
    .add_node("email_agent", email_agent)   # 带中间件的 Agent 整体当节点
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route)
    .compile()
)
```

**什么时候该这么干**：外层拓扑不是"循环到结束"那种标准形态时——
先分类再路由到多个 Agent、并行扇出、或者把 Agent 调用和确定性的步骤缝在一起。

---

## 9. 什么时候该从 `create_agent` 下钻到图 API

| 情况 | 选择 |
| --- | --- |
| 固定工具集的常规 Agent | `create_agent` |
| 需要在工具执行**前后**插入自定义节点 | 下钻到图 API |
| 需要动态扇出（每条输入一个 worker） | 下钻（用 `Send`） |
| 需要精确控制每条边和每个 super-step | 下钻 |
| 需要在 Agent 之外还有复杂流程 | 混合：Agent 当子图 |

**下钻的信号**：你开始想"这个钩子能不能改成节点"——
能，就该下钻了。

---

## 一句话结论

> **`create_agent` 是官方替你画好的循环，中间件是留在循环上的挂钩子。**
> 钩子分两型：节点式顺序跑、包裹式环绕跑；
> before 正序、after 逆序、wrap 嵌套——
> 记住这三件事，中间件就不会写反。
