# 16 · 部署：本地服务器、Studio 与生产

> **这篇要回答**：怎么把一张图变成一个能被别人调用的服务？`langgraph.json` 到底写什么？改代码时怎么不打断正在跑的会话？

---

## 1. 一个 LangGraph 应用是四件套

```
my-app/
├── my_agent/          业务代码（图、工具、状态）
├── langgraph.json     配置文件：依赖、图、环境变量
├── pyproject.toml     依赖声明（或 requirements.txt）
└── .env               环境变量（本地开发用，密钥不进仓库）
```

**缺任何一样都部署不了。** `langgraph.json` 只是"清单"，它不含代码也不含密钥。

---

## 2. langgraph.json：四个段

```json
{
  "dependencies": ["langgraph>=1.2.20", "langchain>=1.5.0", "./my_agent"],
  "graphs": {
    "my_agent": "./my_agent/agent.py:agent"
  },
  "env": "./.env",
  "python_version": "3.13"
}
```

| 段 | 作用 | 常见错法 |
| --- | --- | --- |
| `dependencies` | 运行时还要装什么（含本地包） | 漏掉本地包，线上 import 失败 |
| `graphs` | 图的名字 → 文件路径:变量名 | 变量名写错，报"找不到图" |
| `env` | 本地环境变量文件路径 | **把密钥写进 json** |
| `python_version` | 运行时 Python 版本 | 与依赖要求不一致，线上装不上 |

图的路径有两种写法：**指向已编译的图对象**，或**指向一个返回图的函数**。

> **坑**：`env` 指的是**文件路径**，不是字典。
> 想在 json 里直接塞环境变量是做不到的——也做不到，这正是好事。

---

## 3. 三条命令，三种场景

| 命令 | 起什么 | 要 Docker 吗 | 用途 |
| --- | --- | --- | --- |
| `langgraph dev` | 内存态 Agent Server | **不要** | 本地开发 + Studio 调试 |
| `langgraph up` | 本机 Docker 里的服务 | 要 | 接近生产的本地验证 |
| `langgraph build` | 构建镜像 | 要 | 交给 CI / 部署平台 |

`langgraph dev` 的输出会直接给你三个地址：

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

> **内存态不能上生产**：进程一重启，持久化的状态就没了。
> 生产要用带持久化后端的 Agent Server（LangSmith Deployment 或自托管）。

---

## 4. Studio：浏览器版，不是桌面应用

**现在没有 LangGraph Studio 桌面 App。** 它是 `smith.langchain.com` 上的网页，
连你本机的 Agent Server 来可视化调试：跑图、看每步状态、打断点、**从任意 checkpoint fork 重跑**。

Google 搜"LangGraph Studio"排前面的老教程还在教下载 Apple Silicon 桌面版——
那是 2024 年的事，别跟着装。

### 连不上 localhost 的两个原因

| 浏览器 | 现象 | 解法 |
| --- | --- | --- |
| Chrome 142+ | Studio 报 Failed to fetch | 地址栏锁图标 → 允许 **Local network access** |
| Safari | 一直连不上 | 启动时加 `--tunnel`：`langgraph dev --tunnel` |

先确认端口对不对：Studio URL 里的端口要和服务打印的一致。

---

## 5. 用 SDK 调服务

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")
for chunk in client.runs.stream(
    None,              # Threadless run
    "agent",           # langgraph.json 里定义的名字
    input={"messages": [{"role": "human", "content": "hi"}]},
    stream_mode="updates",
):
    print(chunk.event, chunk.data)
```

`None` 表示**无线程运行**（一次性请求）。要延续上下文就传 `thread_id`。

---

## 6. 生产部署

| 路径 | 谁负责 | 适合 |
| --- | --- | --- |
| LangSmith Cloud | 平台托管 | 快速上线，不想自己运维 |
| 独立服务器 / 混合部署 | 你 | 有数据合规或成本约束 |

流程都是：代码进 GitHub 仓库 → 在 LangSmith 里建 Agent Server → 关联仓库 → 部署 → 用 Studio 验证。

---

## 7. 最反直觉的一条：**改代码会立刻影响所有在跑的会话**

别的引擎会把一次运行**钉在启动时的代码版本**上。LangGraph 不会：

> **LangGraph 用最新的已部署图，立即作用于每一个线程——包括那些从 checkpoint 恢复的旧线程。**

好处很直接：修 bug 立刻生效，不用等老会话跑完。
代价是：**你每次发版，都是一次针对已有 checkpoint 的 API 变更**。

三类兼容性问题：

1. **技术兼容**（最常见）：新代码还要能读懂老 checkpoint 存下来的状态。
   - 重命名/删除节点 → 暂停在那个节点的线程恢复时**找不到那个节点**；
   - 重命名/删除状态键 → 老 checkpoint 里还有这个键，下游读不到就炸；
   - 收紧状态字段（把可选改成必填、收窄类型）→ 老 checkpoint 校验不过。

   > **实测警告（langgraph 1.2.20）**：节点改名后，老线程恢复**不抛异常**——
   > 找不到旧名字的待执行节点，就把它**静默丢弃**。运行显示"完成"，
   > 但那一步根本没执行，状态停在旧值。
   > **没有报错、没有告警，只有一个安静的错误结果**——这比直接崩掉危险得多。
   > 第 16 章配套实验把这条复现出来了。

   > **边拓扑本身不存进 checkpoint**——增删改边是安全的，
   > 只要节点还在。唯一会打断暂停线程的拓扑改动是**重命名或删除节点**。

2. **业务兼容**：新逻辑对老会话不适用。老线程该走老路，新线程走新路。
   做法是在状态里**盖一个版本戳**，用条件边分流：

   ```python
   class State(TypedDict):
       flow_version: NotRequired[int]   # 老 checkpoint 没这个键也能过
   ```

   > 版本戳必须在**线程开始时**打；打晚了，老线程恢复时用不上。

3. **非确定性**：只影响函数式 API。`@task` / `interrupt` 的调用**位置**不能动——
   恢复时靠位置匹配缓存结果，位置一挪，值就错位了。

### 三条保命规矩

- 新字段一律 `NotRequired` / `Optional[...] = None`，让老 checkpoint 还能校验；
- 删除要做成"弃用"：字段/节点先留一个排空周期，确认没有在跑的线程依赖再删；
- 重命名走 **add-then-remove**：新旧并存、双写一段时间，再摘掉旧的。

**怎么知道有没有线程正挂在要删的节点上？**
Agent Server 的线程搜索按 `status` 过滤（`idle` / `busy` / `interrupted` / `error`），
或者用 LangSmith 追踪看哪个节点还在被进入；单个线程直接 `graph.get_state(config)` 看它停在哪。

---

## 一句话结论

> **部署是把图变成服务，兼容是把服务变成能持续演进的东西。**
> `langgraph.json` 写清依赖、图和环境变量，`langgraph dev` 起内存服务配 Studio 调试；
> 但真正要警惕的是——**你的每次发版都会立刻作用到所有还在跑的会话上**，
> 而且改错节点名**不会报错**，只会安静地丢掉那一步。
> 所以删节点、改状态键之前，先看有没有线程正停在那儿。
