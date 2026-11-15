# 04 · 控制流：分支、循环、Send 与 Command

> **这篇要回答**：图里"下一步去哪"该怎么表达？四种控制流形态怎么选？什么时候一条边就够了，什么时候必须上 `Send` / `Command`？循环卡死了怎么兜底？

---

## 1. 四种形态，就这四种

官方 checklist 把控制流拆成四类（[Build checklist](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md#2-choose-control-flow)、[Control flow](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow.md)）：

| 形态 | 长什么样 | 用什么表达 |
| --- | --- | --- |
| **顺序** — 固定步骤依次跑 | `a → b → c` | `add_edge` / `add_sequence` |
| **分支** ─ 运行时按状态选一条 | `a ─┬─ b` | `add_conditional_edges` |
| **并行** ─ 同时扇出再扇入 | `a ─┬─ b ─┬─ d` | 多条出边 + 归约器 |
| **循环** ─ 边指回前面 | `a → b → a` | 边 + 终止条件 |

```
   顺序            分支             并行              循环
  ┌───┐         ┌───┐          ┌───┐             ┌───┐
  │ a │──▶ b    │ a │──▶ b     │ a │──▶ b        │ a │──┐
  └───┘         └───┘──▶ c     │   │──▶ c        └───┘  │
                                └───┘──▶ d             │  │
                                                       └──┘
```

**这四种都能用边表达。**`Send` 和 `Command` 不是"更高级的边"，
它们是边**表达不了**时的两个逃生舱：

- `Send`：目标节点的数量和入参**只有运行时才知道**；
- `Command`：这一步要**同时**改状态和选下一个节点。

---

## 2. 分支：条件边只选路，不改状态

```python
def route(state: State) -> Literal["even_branch", "odd_branch"]:
    return "even_branch" if state["is_even"] else "odd_branch"

builder.add_conditional_edges("check_even", route)
```

路由函数的返回值**就是下一个节点名**（或节点名列表，或 `END`）。
它只负责选路；想同时改状态，就得上 `Command`（第 6 节）。

路由函数可以返回**多个**节点名 → 多路扇出（各分支并行）。

### 坑 1：条件边的路由函数和节点函数不是一回事

节点函数返回 `dict`（状态更新），路由函数返回 `str`（节点名）。
把两者写成一个函数，编译期不报错，运行期直接抛
`InvalidUpdateError: Expected dict, got long`。

还有个隐蔽版：路由函数如果**改了状态**（比如 `state["x"] = 1`），
那次修改**不会**进状态——节点入参是不可写的快照。
想改状态又不想加一个中间节点，就用 `Command`。

---

## 3. 并行：扇出、扇入，以及"顺序不保证"

同一个 super-step 里的分支**互相看不到对方的更新**，只能看到上一步结束时的快照。
扇入时靠归约器合并（第 3 章）。

```
   super-step 0        super-step 1
  ┌───┬───┐            ┌───────┐
  │ a │ b │  ────────► │  d    │   d 看到的是 a、b 合并后的状态
  └───┴───┘            └───────┘
```

### 坑 2：并行分支的产出顺序**没有保证**

官方文档原话：*"updates from a parallel superstep may not be ordered consistently"*。
本机实测里三个 worker 的完成顺序是 `a、c、b`——和注册顺序无关。

所以**汇总节点必须自己排序**。要么像配套实验那样让 worker 带上 `index` 再排，
要么把结果写进一个带排序键的结构里。想让"第几个分支的结果"固定在第几位，
就别依赖执行顺序。

顺带一条工程细节：**super-step 是事务性的**——任一分支抛异常，这一整步的更新**都不落盘**。
所以并行分支里的错误处理要么自己 `try/except`，要么给节点挂重试策略（第 5 章）。

分支长度不一致时，扇入节点要等最长的那条。
官方给的办法是 `add_node("d", d, defer=True)`——`defer` 会让 `d` 一直等到全图没有待办任务为止。
（`add_edge(["b2", "c"], "d")` 这种列表形式也能等，但它和"多条独立边"语义不同，
细节见 [Defer node execution](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow.md#defer-node-execution)。）

---

## 4. 循环：终止条件 + `recursion_limit` 双保险

循环没有特殊机制，就是"边指回了前面的节点"。收敛靠两样东西：

1. **终止条件**：一条条件边，条件满足就路由到 `END`。
2. **`recursion_limit`**：兜底，超了抛 `GraphRecursionError`。

```python
def route(state: State) -> Literal["work", END]:
    return "work" if len(state["rounds"]) < state["budget"] else END

builder.add_conditional_edges("work", route)
```

### 坑 3：`recursion_limit` 数的是 super-step，不是"圈"

`recursion_limit` 是**一次运行允许执行多少个 super-step**。
一轮循环如果包含 4 个 super-step，那它就烧掉 4 点额度。

```
  a → b → c → d → a → b → c → d → ...
  └────── 1 圈 ──────┘   = 4 个 super-step
```

版本事实（[Graph API 概览](https://docs.langchain.com/oss/python/langgraph/graph-api.md#recursion-limit)）：
自 1.1.4 起默认 **10007** 个 super-step（1.0.6–1.1.3 是 10000）；
可用环境变量 `LANGGRAPH_DEFAULT_RECURSION_LIMIT` 改默认值，
也可以在调用时传——注意它是**顶层** config 键，不在 `configurable` 里面：

```python
graph.invoke(inputs, {"recursion_limit": 4})
```

### 两种处理超限的姿势

| 方式 | 怎么发现 | 在哪处理 | 结果 |
| --- | --- | --- | --- |
| 预防式 `RemainingSteps` | 撞限**之前** | 图**内部**条件路由 | 图继续走到收尾分支 |
| 兜底式 `GraphRecursionError` | 撞限**之后** | 图**外部** `try/except` | 整个运行中断 |

`RemainingSteps` 是 Python 独有的托管值（`from langgraph.managed.is_last_step import RemainingSteps`），
放在状态里就能读"还剩几步"，快到限了就切到降级分支。

也可以在节点里读 `config["metadata"]["langgraph_step"]` 拿当前步号（第 9 章细讲流式）。

**什么时候不该用循环**：步数本来就能算出来的时候，别用循环。
固定步数用固定节点，推理、调试、断点续跑全都容易一个量级。
循环是给"步数由运行时决定的迭代"用的——ReAct 循环、反思重试。

---

## 5. `Send`：运行时才知道要扇出多少

普通边的拓扑是**建图时**定死的：选中的节点拿到的永远是**同一份共享状态**。
"同一个节点跑 N 次，每次给它不同的输入"这件事，边做不到——[Send 登场](https://docs.langchain.com/oss/python/langgraph/graph-api.md#send)。

```python
def route_workers(state: State) -> list[Send]:
    return [Send("worker", {"chunk": c, "index": i}) for i, c in enumerate(state["chunks"])]

builder.add_conditional_edges("plan", route_workers, ["worker"])
```

`Send("worker", payload)` 的两个参数：**目标节点名** + **这一次调用的输入**。

### 坑 4：worker 拿到的是**它自己的 payload**，不是整张状态

这是 `Send` 最反直觉的一点，也是它最大的价值：

```
   整体状态  DocState{chunks, chunk_summaries, report}
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  Send(chunk=0)   Send(chunk=1)   Send(chunk=2)
      │                │                │
      ▼                ▼                ▼
   worker          worker          worker
  只看到          只看到          只看到
  {chunk:0,index:0} ...
```

配套实验里每个 worker 收到的 state **只有两个键**：`chunk` 和 `index`。

好处是 worker 的输入契约极窄、可单测；代价是 worker 拿不到整体状态——
要的东西得在 `Send` 的 payload 里显式塞进去。

扇入仍然靠归约器：所有 worker 写同一个共享键（`chunk_summaries`），
`operator.add` 把它们合并。配套实验里的 map-reduce 图：

```
   plan ──(Send×N)──▶ worker ──▶ worker ──▶ worker ──▶ reduce
                                                        ▲
                                              归约器合并 + 按 index 排序
```

`reduce` 节点排在所有 worker 之后**跑一次**（不是每个 worker 跑一次）。

---

## 6. `Command`：改状态和选路，一步到位

正常情况下这两件事是分开的：节点返回更新，边决定去哪。
**当"下一步"取决于"这一步刚做完什么"时**，拆开写会很别扭——这时用 `Command`（[文档](https://docs.langchain.com/oss/python/langgraph/graph-api.md#command)）。

`Command` 有四个字段：

| 字段 | 干什么 | 用在哪 |
| --- | --- | --- |
| `update` | 状态更新 | 节点返回 |
| `goto` | 下一个（或几个）节点 | 节点返回 |
| `graph` | 目标在父图里（`Command.PARENT`） | 子图节点返回 |
| `resume` | 中断后恢复的值 | `invoke` / `stream` 的**入参** |

```python
def decide(state: State) -> Command[Literal["accept", "reject"]]:
    goto = "accept" if state["score"] >= 60 else "reject"
    return Command(update={"verdict": goto}, goto=goto)
```

一个 `Command` 同时完成"写状态"和"选路"，图里就不需要条件边了。

### 坑 5：`Command[Literal[...]]` 里列的目标节点必须真实存在

`Literal` 声明不是装饰——**它就是图的边**。
声明了但没建这个节点，**编译期**就报错：

```
ValueError: Found edge ending at unknown node `nonexistent`
```

好处是图能正确渲染出动态边（`get_graph()` 里能看到 `decide → accept`）。
写 `Literal` 时把目标一个个核对一遍。

### 坑 6：别把 `Command` 和静态边混在同一个节点上

官方明确警告：*"`Command` adds dynamic edges. Static edges from the same node still run."*
同一个节点既 `add_edge` 又返回 `Command` → **两条路都会跑**。

规则很简单：**一个节点的出向，要么全是静态边，要么全是动态路由（条件边 / `Command`），二选一。**

### 坑 7：`Command` 当入参 ≠ "开始新一轮"

`Command(resume=...)` 是**中断恢复**的入口。但"任何 `Command` 作为入参"都会
**从最新检查点恢复，而不是从 `START` 重来**——一个已经跑完的 thread 会看起来"卡住了"。

配套实验实测：同一个 thread 跑完后再 `invoke(Command(update={...}))`，
**一个节点都没重跑**，只是把 update 写进了状态。

```python
graph.invoke({"messages": [...]}, config)               # 对：新的一轮，从 START
graph.invoke(Command(resume="yes"), config)             # 对：恢复中断
graph.invoke(Command(update={"messages": [...]}), config)  # 错：变成从检查点恢复
```

想开新一轮，就传**普通字典**。

---

## 7. 该用边，还是 `Send` / `Command`

官方 checklist 的原话是：*"Most graphs only need nodes, edges, and conditional edges.
Keep routing in edges unless you hit one of the cases below."*

| 你的需求 | 用什么 | 为什么 |
| --- | --- | --- |
| 固定顺序 | `add_edge` / `add_sequence` | 最直白，最好推理 |
| 按状态选一个节点 | `add_conditional_edges` | 路由是纯函数，单独可测 |
| 已知数量的并行分支 | 多条边 + 归约器 | `Send` 的动态性是浪费 |
| **同一节点跑 N 次，每次输入不同，N 运行时才知道** | **`Send`** | 边的拓扑建图时就定了 |
| **这一步既要改状态又要选路** | **`Command`** | 拆成"节点 + 条件边"是硬凑 |
| 中断后恢复 | `Command(resume=...)` 作入参 | 唯一的恢复入口 |
| 从工具里写状态并路由 | 工具返回 `Command` | `ToolNode` 会帮你传播 |
| 子图节点跳到父图节点 | `Command(goto=..., graph=Command.PARENT)` | `goto` 只在当前图内解析 |

**判断口诀**：先问"这条路由**需要**改状态吗"（要 → `Command`），
再问"目标节点**有几个**、**入参一样吗**"（运行时才知道、且入参不同 → `Send`）。
两个都是"否"，就老老实实用边。

**什么时候不该用 `Command`**：只要路由就别用。`Command` 的好处是"一步到位"，
代价是把路由逻辑藏进了节点函数——图的可视化会多几条虚边，
而看图的人得先知道 `Literal` 声明才能读懂流向。路由逻辑复杂、需要单独测试时，
条件边更合适。

---

## 8. 配套实验

四种形态 + `Send` map-reduce + `Command` + 三个坑，都有断言：

`Demos/examples/langgraph_lab/lessons/ch04_control_flow.py`

---

## 参考

- [Build sequences, branches, and loops](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow.md)
- [Use Send and Command](https://docs.langchain.com/oss/python/langgraph/graph-api-send-command.md)
- [Graph API 概览（Send / Command / Recursion limit）](https://docs.langchain.com/oss/python/langgraph/graph-api.md#send)
- [Build checklist for the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md)
- [Workflows and agents（官方工作流模式库）](https://docs.langchain.com/oss/python/langgraph/workflows-agents.md)

## 一句话结论

> **边是骨架，`Send` 是动态扇出的手，`Command` 是"边做不了"时的逃生舱**——
> 四种形态（顺序 / 分支 / 并行 / 循环）能靠边表达的都别上高级 API；
> 真要上，就必须清楚它换来了什么、又多付了什么代价。
