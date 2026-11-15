# 13 · MCP 接入与工具生态：Agent 怎么用上外部工具

> **这篇要回答**：MCP 到底是个什么协议？我怎么把一个 MCP 服务器的工具接进 LangGraph 图？直接写个 `@tool` 不行吗，为什么官方要搞一套新东西？

---

## 1. MCP 解决的是"接入方式"，不是"工具能力"

先把话说死：**MCP 不是"新的工具"，MCP 是"新的接法"。**

你原来写 `@tool` 接数据库、写 `@tool` 接浏览器，每个服务一套 SDK、一套鉴权、一套错误处理。
MCP（[Model Context Protocol](https://modelcontextprotocol.io)）把这套接入统一成一个协议：

```
┌──────────────────┐        MCP 协议（JSON-RPC over HTTP/stdio）
│  你的 Agent      │  ──►  ┌──────────────────────────────┐
│  LangGraph 图    │        │  MCP Server（服务端）        │
└──────────────────┘        │  ├─ tools     可调用的函数    │
        ▲                   │  ├─ resources 可读的数据源    │
        │                   │  └─ prompts   可复用的模板    │
        │                   └──────────────────────────────┘
        │                             │
        │                    ┌────────┴────────┐
        │                    │  一个 server    │
        │                    │  可以是：       │
        │                    │  · 本地进程      │
        │                    │  · 远程 HTTP     │
        │                    │  · 数据库        │
        │                    │  · 浏览器/文件系统│
        │                    └─────────────────┘
```

三类原语（[官方规范](https://modelcontextprotocol.io)）：

| 原语 | 语义 | 谁负责调用 | 类比 |
| --- | --- | --- | --- |
| **tools** | 有副作用的可执行函数，模型主动调 | 客户端发起，服务端执行 | HTTP POST `/api` |
| **resources** | 只读的数据，用 URI 寻址 | 客户端读，服务端给 | HTTP GET |
| **prompts** | 可复用的提示词模板 | 客户端取，服务端给 | 服务端模板 |

**关键事实**：`langchain.mcp` 这个命名空间**只把 tools 适配成了 LangChain 工具**。
resources 和 prompts 仍然要通过原始 MCP 客户端（`fastmcp.Client` 的
`list_resources` / `read_resource` / `list_prompts` / `get_prompt`）去拿。
所以"MCP 生态"这个说法，在 LangChain 这边目前实际落地的是 tools 这一块。

---

## 2. `langchain.mcp`：beta，但已经是官方路线

命名空间在 `langchain 1.4.0` 引入，要求 `langchain[mcp]>=1.4.0`，**目前是 beta**：

```python
from langchain.mcp import MCPAdapter, MCPToolArtifact, as_langchain_tool
```

导入时会抛一次 `LangChainBetaWarning`（每进程一次）。本教程配套实验里有一行断言专门盯这件事。

三个入口：

| 名字 | 干什么 | 什么时候用 |
| --- | --- | --- |
| `MCPAdapter` | 打开连接 → 发现工具 → 返回标准 LangChain 工具 | 要整个服务器的工具目录时 |
| `as_langchain_tool` | 把**单个** MCP 工具包成 LangChain 工具 | 只想借一个工具时 |
| `langchain.mcp.tools` | 工具函数 + 内容块构造器（`create_text_block` 等） | 自己处理多模态结果时 |

> **坑 1（迁移）**：1.4.0 之前的老写法是单独的 `langchain-mcp-adapters` 包，
> API 完全不同（`MultiServerMCPClient` 那一套）。网上大量教程还在用老包，
> 照抄会报 `ImportError`。官方有专门的[迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters)。

---

## 3. 服务器形态：同一个适配器，四种 target

`MCPAdapter(target)` 的 target 决定传输方式，这是 FastMCP 帮它做的传输推断：

| target 写法 | 连到什么 | 什么时候用 |
| --- | --- | --- |
| `Path("server.py")` | 本地子进程，走 stdio | 自己写的 Python 服务 |
| `MCPAdapter(fastmcp_instance)` | **进程内**，无子进程无端口 | 测试、教学、本地嵌入 |
| `"https://host/mcp"` | 远程，走 Streamable HTTP | 托管的服务 |
| `{"mcpServers": {...}}` | 多个服务器合成一个连接 | 工具来自好几个后端 |

进程内这一档是本教程实验用的——不连公网、不起子进程，断言能写死：

```python
from fastmcp import FastMCP
from langchain.mcp import MCPAdapter

server = FastMCP("lab")

@server.tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b

async def main():
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()   # -> list[BaseTool]
```

---

## 4. 把 MCP 工具接进图：两条路径

这是本章的核心问题。答案是：**和普通 LangChain 工具完全一样**，两条路径都行。

```
                       MCP 服务器
                            │  MCPAdapter.list_tools()
                            ▼
                 list[BaseTool]  ← 适配后的工具，长得和 @tool 一模一样
                     │        │
       路径 A         │        │         路径 B
   ┌─────────────────┘        └──────────────────┐
   │  节点里直接 await                        ToolNode(tools)
   │  ┌──────────────┐                      ┌──────────────┐
   │  │ my_node      │                      │  tools       │
   │  │ await tool.  │                      │  ToolNode    │
   │  │  ainvoke()   │                      └──────┬───────┘
   │  └──────────────┘                             │
   └───────────────────────────────────────────────┘
```

路径 A（节点直接调）适合"这一步固定要用某个外部能力"的场景；
路径 B（当工具集）适合"让模型自己决定用不用"：

```python
from langgraph.prebuilt import ToolNode

async def my_node(state):
    payload = await mcp_tool.ainvoke({"a": 2, "b": 40})   # 路径 A
    return {"messages": [AIMessage(content=str(payload))]}

tool_node = ToolNode(tools)          # 路径 B：tools 就是 list_tools() 的结果
builder.add_node("tools", tool_node)
```

| | 路径 A：节点里直接调 | 路径 B：当 ToolNode 的工具集 |
| --- | --- | --- |
| 谁决定调不调 | 你的代码 | 模型 |
| 失败怎么冒泡 | 异常（可在节点里重试） | `status="error"` 的 ToolMessage |
| 适合 | 固定流程里的取数 | 探索式任务、模型自选工具 |
| MCP 特有能力 | 用不上 | 错误语义、artifact、注解全都在 |

> **反直觉点**：路径 A 里 MCP 工具是**异步**的（`ainvoke`）。
> 在同步节点里直接调会拿到一个 coroutine 对象而不是结果——
> 这是接 MCP 时最常见的"看起来没报错但结果是空的"的原因。

---

## 5. 结果长什么样：内容块、artifact、错误

MCP 的返回不是字符串，是**内容块**。适配器把它翻译成 LangChain 的标准 content block：

| MCP 侧 | LangChain 侧 | 在哪 |
| --- | --- | --- |
| 文本结果 | `{"type": "text", "text": ...}` | `ToolMessage.content` |
| 图片结果 | `{"type": "image", ...}` | `ToolMessage.content`（模型能直接看） |
| `structuredContent` | 原样挂在 artifact 上 | `ToolMessage.artifact["structured_content"]` |
| `isError=True` | `ToolMessage(status="error")` | 模型能读到并自己纠正 |

结构化结果单独放 artifact、**不折进给模型看的文本**，这是有意的设计：
程序要精确取数用 artifact，模型要理解用文本。实验里断言过这一条——
`artifact` 的内容是 `{'structured_content': {'result': 42}}`。

还有一层**溯源信息**挂在 `tool.metadata["mcp"]` 下：服务器自己声明的
`read_only_hint` / `destructive_hint` 等注解、`_meta`、服务器名与版本。
这让你可以**按服务器声明**而不是按工具名硬编码来做审批门禁：

```python
def is_destructive(tool) -> bool:
    ann = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return ann.get("destructive_hint", False)
```

---

## 6. 生命周期与多服务器

### 6.1 生命周期：工具活得比上下文久

`MCPAdapter` 是异步上下文管理器。官方推荐的默认写法是"在里面发现工具，然后退出"：

```
async with MCPAdapter(target) as adapter:   # ← 连接在这里建立
    tools = await adapter.list_tools()      # ← 发现
# ← 连接在这里释放
agent = create_agent(model, tools)          # ← 但 tools 仍然可用
```

> **坑 2（反直觉）**：**工具在上下文退出后依然能调用**。
> 适配器返回的工具对象自己持有 client，而 FastMCP 的 client 是**可重入**的：
> 每次调用自己开一个会话、用完释放。
> 好处是长跑 Agent 不会在两次工具调用之间挂着一个空闲连接；
> 代价是**你没法靠"退出上下文"来强制断开**——要断连得自己管 client 生命周期。

如果你希望整个 run 共用一个会话，就把 adapter 的上下文一直开着，
包住 agent 的调用。

### 6.2 多服务器：两种组装方式

| | `MCPConfig` 字典 | `ClientGroup` |
| --- | --- | --- |
| 连接数 | 一个聚合连接 | 每个服务器一条独立连接 |
| 协议纪元 | **整个舰队统一协商一个纪元** | 每个服务器各谈各的 |
| 鉴权 | 每个 server 同一个 key 约定 | 每个 client 可以完全不同的凭证 |
| 工具命名 | 加 config key 前缀（`weather_xxx`） | 同样加前缀（`server_tool`） |
| 什么时候用 | 都是现代服务器、鉴权一致 | 有老服务器、要每服务器不同凭证 |

> **坑 3**：把一台"老协议"服务器塞进 `MCPConfig` 舰队，
> **整个舰队会被降到最老的那个纪元**。要混跑新旧，就得用 `ClientGroup`
> 或给每台服务器单独一个 adapter。

### 6.3 发现结果可以缓存

`list_tools(cache_mode=...)`：`use`（默认，遵守服务器 TTL 提示）/
`refresh`（强制拉新）/ `bypass`（完全跳过缓存）。多副本部署时缓存要**按用户隔离**，
否则一个用户可能看到别人的工具列表。

---

## 7. 鉴权：三种粒度

鉴权全部委托给 FastMCP，凭证挂在**预建的 client** 上，再把 client 交给适配器
（`Client(url, auth=...)` 的 `auth` 接受 bearer 字符串、字面量 `"oauth"` 或任意 `httpx.Auth`）：

```python
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth

client = Client("https://example.com/mcp", auth=BearerAuth(token="..."))
async with MCPAdapter(client) as adapter:
    tools = await adapter.list_tools()
```

| 粒度 | 怎么配 | 场景 |
| --- | --- | --- |
| **静态 bearer** | `auth="<token>"` | 服务端认死一个 token |
| **OAuth 2.1** | `auth="oauth"` | 客户端跑完整的发现→动态注册→浏览器跳转→换 token |
| **任意 httpx.Auth** | `auth=MyAuth()` | 自签、mTLS、公司网关 |
| **每服务器一份** | `ClientGroup` 里每个 client 各自 `auth` | 一个 Agent 连多个后端 |
| **每用户一份** | graph factory 里按 `runtime.user.identity` 换 token | 生产部署 |

两个细节：

- OAuth 的 token **默认只存在内存里**，每次运行都重走一遍浏览器。
  要跨运行复用，得自己传 `OAuth(..., token_storage=...)`。
- **每用户鉴权**要配两半：Agent Server 侧先把请求解析成用户身份，
  graph factory 里再拿这个身份换一张 per-user token 去连 MCP。
  配套的缓存也要按用户分区。

---

## 8. MCP 工具 vs 本地 `@tool`：怎么选

| | 本地 `@tool` | MCP 工具 |
| --- | --- | --- |
| 部署 | 跟着应用走 | 独立进程 / 独立服务 |
| 复用 | 只能在这个应用里用 | 多个客户端共用一个实现 |
| 语言 | Python | 任何实现了 MCP 的语言 |
| 调试 | 直接打断点 | 要跨进程看日志 |
| 延迟 | 函数调用 | 网络 / 进程往返 |
| 版本耦合 | 和应用一起升 | 各自升 |
| 鉴权 | 进程内权限 | 独立的凭证体系 |

**判断标准很简单**：

- 这个能力**只服务于这一个应用** → 写 `@tool`。省掉一整条协议链路。
- 这个能力**要给多个 Agent / 多个团队用** → 写 MCP server。一次实现，到处接入。
- 能力在**远端服务**里（比如一个真正的数据库网关） → MCP 更自然，
  因为你已经要过网络了，多一层协议协议栈不亏。

**别为了"看起来更现代"把本地 `@tool` 改写成 MCP**——
那会让你的单元测试从"调一个函数"变成"起一个服务"。

---

## 9. 四个必须知道的坑

### 坑 1：字符串 target 必须是 http(s) URL

`MCPAdapter("server.py")` 会被**拒绝**并抛 `ValueError`。

原因值得记住：`fastmcp.Client` 解析字符串时**先当文件路径试，再当 URL 试**。
一个碰巧叫 `server.py` 的字符串会被当成脚本启动。
而字符串恰恰是最容易从配置、请求体、甚至模型输出里流进来的形态。
所以官方把"字符串 = 远程 URL"这条边界写死了，本地服务必须显式用 `Path(...)`。

**这是安全设计，不是 usability bug。** 想要本地脚本，老老实实写 `Path("server.py")`。

### 坑 2：工具活得比连接久（见 6.1）

退出 adapter 上下文**不会**让工具失效，因为工具自己持有可重入的 client。

### 坑 3：恢复运行会重跑工具

MCP 的 **elicitation**（服务端在工具执行中途反问客户端要输入）
在 LangChain 侧被翻译成 LangGraph 的 `interrupt()`：

```python
paused = await agent.ainvoke({"messages": [...]}, config)
[interrupt] = paused["__interrupt__"]
[question] = interrupt.value["requests"]
resumed = await agent.ainvoke(
    Command(resume={"responses": {question["key"]: {"action": "accept", "content": {...}}}}),
    config,
)
```
恢复时**工具从头重跑一遍**。服务端在反问之前做过的副作用会**再来一次**。
所以：**凡是有副作用的工具，必须自己保证幂等**（用请求 id 去重，不要靠"框架会帮我处理"）。

回答只有三种：`accept` / `decline` / `cancel`。

### 坑 4：`isError` 和"传输失败"是两回事

服务端报的错（`isError=True`）→ 变成 `status="error"` 的 ToolMessage，模型看得见、能纠正。
但**传输层失败**（连不上、超时）→ 直接**抛异常**，不会变成 ToolMessage。
因为模型对"网络断了"没法采取行动，只能看到一句没头没尾的错误。

### 坑 5：beta

`langchain.mcp` 是 beta，API 可能变。**锁版本**：`langchain[mcp]==<固定版本>`，
升级前重读官方文档。本教程实验里对"导入确实抛 beta 警告"做了断言——
这样 beta 状态哪天悄悄变成 stable，你也会立刻知道。

---

## 10. 什么时候不该用 MCP

| 场景 | 为什么别用 |
| --- | --- |
| 只有一个 Python 应用、一个工具 | 多一层进程/网络往返，收益为零 |
| 需要亚毫秒级延迟 | 工具调用从"函数调用"变成"IO" |
| 需要事务性回滚 | MCP 工具调用是远程副作用，没有跨服务事务 |
| 服务器不可信、工具有破坏性 | 远程工具拿到的是你 Agent 的全部权限 |
| 需要精确控制"这次到底调了什么" | 协议层多了一段，出问题时归因更难 |

---

## 配套实验

`Demos/examples/langgraph_lab/lessons/ch13_mcp.py`

- 在**本进程内**用 FastMCP 起一个最小 MCP 服务器（不连公网、不起子进程）；
- 断言 `MCPAdapter` 能发现工具，且工具的 docstring/签名正确变成了 schema；
- 路径 A：MCP 工具在 LangGraph 节点里直接调用，算出 42；
- 路径 B：MCP 工具进 `ToolNode`，产出 `status="success"` 的 ToolMessage，
  且 `artifact["structured_content"] == {"result": 42}`；
- 错误路径：除零变成 `status="error"` 的 ToolMessage，服务器原话透传给模型；
- `as_langchain_tool` 单工具转换；
- `langchain.mcp` 的 beta 警告确实会抛。

跑法：

```bash
cd Demos/examples && uv run --project .. python -m langgraph_lab ch13
```

---

## 一句话结论

> **MCP 不是"新的工具"，是"新的接法"**：它把客户端/服务端、三类原语、传输与鉴权标准化，
> 于是 LangChain 侧只剩一件事——`MCPAdapter.list_tools()` 拿回来的 `list[BaseTool]`，
> 和你自己写的 `@tool` 完全平权：要么在节点里直接 await，要么丢给 `ToolNode`。
> 真正要花心思的不是"怎么接"，而是**生命周期**（工具活得比连接久）、
> **幂等**（恢复运行会重跑工具）和**信任边界**（一个远程服务器拿到的是你 Agent 的全部权限）。
