# 11 · 函数式 API 与 Pregel 运行时

> **这篇要回答**：已经有了一套 Agent 代码，不想重写成图怎么办？函数式 API 和图 API 到底差在哪？底下的 Pregel 引擎到底是什么？

---

## 1. 两个 API，一套运行时

LangGraph 提供两套写法，**底层都是同一个 Pregel 引擎**：

| | 图 API（Graph API） | 函数式 API（Functional API） |
| --- | --- | --- |
| 写什么 | `StateGraph` + 节点 + 边 | `@entrypoint` + `@task` |
| 控制流 | 边与条件边 | 普通的 `if` / `for` / 函数调用 |
| 数据传递 | 共享 state schema | 普通变量、返回值 |
| 状态落盘 | 每个 super-step 后落一次 | task 的结果存在 entrypoint 的检查点上 |
| 可视化 | 有静态拓扑图 | **没有** |
| 适合 | 从零设计新 Agent | **把已有代码包起来部署** |

官方态度很明确：**新项目用图 API，函数式 API 主要是为了"把别的框架的 Agent 部署到 LangSmith Agent Server"**
（Claude Agent SDK、Strands、CrewAI、AutoGen 这些）。

**换句话说：图 API 是给"设计流程"用的，函数式 API 是给"包装流程"用的。**

---

## 2. 两个装饰器

```python
from langgraph.func import entrypoint, task

@task
def fetch(url: str) -> str:
    """一个离散的工作单元：一次 API 调用。"""
    return requests.get(url).text

@entrypoint(checkpointer=checkpointer)
def workflow(url: str) -> str:
    content = fetch(url).result()      # 调用 task 立即开始执行
    return summarize(content)
```

- **`@entrypoint`**：工作流的起点。它负责管理执行、中断和检查点。
  装饰完就是一个 Pregel 对象，可以 `invoke` / `ainvoke` / `stream`。
  **它必须只接受一个位置参数**——要传多个值就打包成字典。
- **`@task`**：一个离散的工作单元。调用它**立即开始执行**，返回一个 future；
  你在 `.result()`（同步）或 `await`（异步）时才会阻塞。
  所以互不依赖的 task 可以**并发**跑。

> **task 只能从 entrypoint、另一个 task 或图 API 的节点里调用**，
> 不要在工作流之外的普通代码里调。

---

## 3. 注入参数：不用层层传参

entrypoint 可以声明这些关键字参数，LangGraph 会在运行时注入：

| 参数 | 是什么 |
| --- | --- |
| `previous` | 同一线程上一次调用的返回值（**短期记忆**，需要检查点） |
| `config` | `RunnableConfig`，运行期配置 |
| `store` | `BaseStore`，**跨线程长期记忆** |
| `writer` | `StreamWriter`，自定义流式输出 |
| `runtime` | `Runtime`，当前运行的上下文、store、writer 都在里面 |

> **坑**：注入参数必须**写对名字和类型标注**。
> 名字写错不会报错——它就变成了一个普通参数，然后永远是默认值。

---

## 4. 恢复语义：不是"从断点继续"，是"重放"

这是函数式 API 最反直觉、也最重要的一条：

> **恢复不会从停下的那一行继续。** 执行会退回检查点边界，然后**从头重放**，
> 直到再次撞上暂停点。已完成 task 的结果从检查点**还原**，不重算。

```
第一次运行：task A ✓ → task B ✓ → interrupt ⏸
恢复：       task A（还原，不重算）→ task B（还原，不重算）→ interrupt ✓
```

推论有三条：

1. **副作用要放进 task**。写在 entrypoint 里的文件写入、邮件发送，
   恢复时会**再执行一遍**。
2. **随机和时间要放进 task**。放在 entrypoint 里，重放时值会变，
   可能走进另一个分支。
3. **task 必须幂等**。"开始了但没完成"的 task 恢复时可能重跑——
   写操作用幂等键，或者先查再写。

> **坑**：`interrupt` 的恢复值是**按调用下标**匹配的。
> 中间如果插了一个新的 `interrupt`，恢复值就会**贴到错的那个上**。

---

## 5. 序列化：配了检查点就有硬要求

| 对象 | 要求 |
| --- | --- |
| entrypoint 的输入 / 输出 | 必须 JSON 可序列化 |
| task 的输出 | 必须 JSON 可序列化 |

不可序列化的对象（比如一个连接句柄）会直接抛运行时错误。
这是"能不能断点续跑"的分水岭：**序列化不过，检查点就白配。**

---

## 6. Pregel：底下的引擎

Pregel 来自图计算里的 BSP 模型。一次运行被切成若干 **super-step**：

```
super-step 0:  START ──► node_a ──► node_b      ← 同一步里的节点可以并行
super-step 1:  node_b ──► node_c
super-step 2:  node_c ──► END
```

三条推论（回头看第 2 章）：

- 同一 super-step 里的节点**看不到彼此的更新**，只看到上一步结束时的快照；
- 每个 super-step 结束都可以落一个**检查点**——这是持久化的最小粒度；
- 图 API 的节点在恢复时**从头重跑**（所以节点要幂等）；
  函数式 API 的 task 结果被还原（所以副作用要放进 task）。

**图 API 和函数式 API 的恢复差异，根源就在这条。**

---

## 7. 什么时候用哪个

| 场景 | 选择 |
| --- | --- |
| 从零设计一个有分支、并行、循环的 Agent | 图 API |
| 已有 CrewAI / AutoGen / 自研循环，想部署到 LangSmith | 函数式 API |
| 需要静态拓扑图给人看、需要 Studio 里拖拽调试 | 图 API |
| 只是几个函数的线性流程 | **两个都别用**，普通 Python 就行 |

---

## 一句话结论

> **图 API 是给"设计"用的，函数式 API 是给"包装"用的。**
> 前者把控制流画成图，代价是要重新设计状态；
> 后者把控制流留在普通 Python 里，代价是恢复时从头重放——
> 所以副作用和随机性都得塞进 task，才有幂等和可恢复。
