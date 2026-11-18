# 附录 A · 术语表（协议层）

这份术语表只收**模型 API 协议层**的词。Agent 运行时层的术语见
[`AI/Agent/附录-A-术语表.md`](../Agent/附录-A-术语表.md)。

每条给出：一句话定义 + 一个判别问题。

---

## 一、请求参数

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Messages** | 聊天历史数组，唯一必填参数 | system 在第几条？ |
| **System Message** | 行为规则，最高优先级 | 它是"规则"还是"对话内容"？ |
| **User Message** | 终端用户输入 | 模型是在回应它，还是在遵守它？ |
| **Assistant Message** | 模型输出；可含 `tool_calls` | 这条是说话还是调用意图？ |
| **Tool Message** | 工具执行结果 | `tool_call_id` 填了吗？ |
| **Temperature** | 采样随机性 | 任务需要复现吗？ |
| **Top-p** | 核采样 | 和 temperature 一起调了吗？ |
| **Max Tokens** | 输出上限 | 撞上限时 `finish_reason` 是什么？ |
| **Stop Sequences** | 停止序列 | 你是不是在用 stop 干结构化输出的活？ |
| **Stream** | 是否流式返回 | 你要的是首字快还是总时快？ |
| **Tools** | 可用工具声明数组 | 每轮都要重发吗？ |
| **Tool Choice** | 工具调用策略 | 该强制时用的是哪个模式？ |
| **Response Format** | 输出格式约束 | 用的是 json_object 还是 json_schema？ |
| **Reasoning Effort / Budget** | 思考预算 | 这个任务真的需要推理吗？ |
| **Seed** | 随机种子 | 同 seed 真的可复现吗？ |
| **User / Metadata** | 透传字段 | 会进账单或日志吗？ |

---

## 二、响应字段

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Content** | 文本内容 | 调工具时它是 `null` 吗？ |
| **Tool Calls** | 调用意图列表 | 一轮里有几个？ |
| **Finish Reason** | 停止原因 | 这个响应能直接用吗？ |
| **`stop`** | 正常说完 | — |
| **`length`** | 撞输出上限 | 先怀疑输入，还是怀疑模型？ |
| **`tool_calls`** | 要调工具 | 执行后要回填 tool 消息 |
| **`content_filter`** | 被内容过滤 | 改写输入，别重试原请求 |
| **Usage** | token 统计 | 输入输出比是多少？ |
| **Prompt Tokens** | 输入 token | 为什么它总是远大于输出？ |
| **Completion Tokens** | 输出 token | 思考 token 算在这里吗？ |
| **Reasoning Tokens** | 推理模型的思考 token | 它算进输出计费吗？ |
| **Cached Tokens** | 命中缓存的输入 token | 缓存折扣生效了吗？ |
| **System Fingerprint** | 后端配置指纹 | 同样的参数结果不稳定时看它 |

---

## 三、工具调用

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Tool Spec** | 工具声明（name/description/parameters） | 模型能看到我的实现吗？ |
| **Tool Call** | 一次调用意图 | 有 `id` 吗？ |
| **`arguments`** | 调用参数，**JSON 字符串** | 解析失败时你怎么办？ |
| **Strict Schema** | 严格约束的 schema | 嵌套多深？ |
| **Parallel Tool Calls** | 一轮多个调用意图 | 部分失败怎么表达？ |
| **`auto`** | 由模型自己决定要不要调 | 默认值 |
| **`none`** | 禁止调用工具 | — |
| **`required`** | 必须调用某个工具 | 抽取任务里比"请返回 JSON"可靠 |
| **Named Tool Choice** | 强制调用指定工具 | 流程完全确定时用 |
| **Tool Result Message** | 工具结果回填 | id 对得上吗？ |

---

## 四、结构化输出

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **JSON Mode** | 只保证输出是 JSON 对象 | 字段你控制吗？ |
| **JSON Schema Mode** | 按 schema 约束输出 | schema 合法吗？ |
| **Constrained Decoding（约束解码）** | 解码时按 schema 约束合法 token | 合法性由谁保证？ |
| **Prompted JSON** | 靠提示词请求 JSON | 校验了吗？ |
| **Schema Validation** | 校验输出是否符合 schema | 校验失败怎么办？ |
| **Repair（修复重试）** | 把校验错误回给模型让它改 | 重试上限是多少？ |
| **Fenced JSON** | 模型输出包在 ``` 围栏里 | 解析端能抠出来吗？ |
| **JSON Extraction** | 从混杂文本里抠 JSON | 抠不出来时抛什么错？ |

---

## 五、流式

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Chunk（分片）** | 流式返回的增量单元 | `delta` 是增量还是全量？ |
| **Delta** | 本次新增的内容 | 你是 `+=` 还是 `=`？ |
| **SSE** | 服务端事件流 | 连接断了会重连吗？ |
| **TTFT** | 首 token 延迟 | 用户等多久看到第一个字？ |
| **Throughput** | 生成速度（token/s） | 吞吐和延迟是一回事吗？ |
| **Streamed Tool Call** | 流式里的工具调用 | `arguments` 是分片到达的吗？ |
| **Usage Chunk** | 末尾单独发 usage 的分片 | 你的解析器认它吗？ |
| **Cancellation** | 客户端主动断开 | 断了会继续计费吗？ |

---

## 六、多模态

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Content Block** | 消息内容块 | 块的顺序有意义吗？ |
| **`text` block** | 文本块 | — |
| **`image_url` block** | 图片块 | 用 URL 还是 base64？ |
| **`input_audio` block** | 音频块 | 格式和采样率支持吗？ |
| **Base64 Inline** | 图片以 base64 内联 | 请求体会膨胀多少？ |
| **Vision（视觉理解）** | 模型看图并理解 | OCR 够用就别用视觉模型 |
| **Image Token** | 图片折算的 token | 按什么尺寸算的？ |
| **Video Frame Sampling** | 视频抽帧 | 抽几帧？关键帧怎么选？ |
| **Audio Transcript** | 音频转写 | 转写后再进文本模型 |

---

## 七、定价与缓存

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Input Price** | 输入单价（每百万 token） | — |
| **Output Price** | 输出单价 | 通常是输入的几倍？ |
| **Prefix Caching** | 复用相同前缀的缓存 | 你的前缀稳定吗？ |
| **Cache Hit** | 命中缓存 | 哪些部分算前缀？ |
| **Cache Write Fee** | 缓存写入费用 | 短会话是否亏了？ |
| **Cache TTL** | 缓存存活时间 | 你的会话间隔多久？ |
| **Batch Discount** | 批量提交折扣 | 任务能等吗？ |
| **Effective Window** | 窗口 − 输出预留 | 你的输出预留够吗？ |
| **Context Rot** | 长上下文中段权重下降 | 关键信息还在中间吗？ |

---

## 八、模型选型

| 术语 | 定义 | 判别问题 |
| --- | --- | --- |
| **Capability Tier** | 能力档位 | 你的任务需要哪一档？ |
| **Model Routing** | 按任务难度选模型 | 路由判据是什么？ |
| **Version Pin** | 固定模型版本 | 升级走什么流程？ |
| **Migration Checklist** | 换模型要重测的清单 | 评测集跑了吗？ |
| **Shadow Traffic** | 影子流量 | 新模型和旧模型并跑，比的是输出吗？ |
| **Canary** | 金丝雀发布 | 放量比例怎么定？ |
| **Regression Gate** | 分数低于阈值阻断 | 门禁在 CI 里吗？ |
| **Eval Set Overfitting** | 评测集被考熟 | 样本来源是否单一？ |

---

## 九、最容易混淆的四组

**1. `json_object` vs `json_schema`**
- `json_object`：只保证"是 JSON"。
- `json_schema`：按你给的 schema 约束字段。
- 判别：你能不能描述出确切的字段名和类型？不能就别用 schema 模式。

**2. `arguments` 字符串 vs 结构化对象**
- 协议层：`arguments` 是 **JSON 字符串**，你要自己解析。
- 判别：你的代码里有没有 `json.loads`？没有就是没解析。

**3. `finish_reason=length` vs 模型不说了**
- 撞上限：被截断了，输出是半截的。
- 正常停止：内容完整。
- 判别：`finish_reason` 是什么？半截 JSON 几乎都是 `length`。

**4. 缓存命中 vs 自己存了历史**
- 厂商缓存：前缀没变就自动便宜。
- 自己存：你自己决定发什么、发多少。
- 判别：你的请求前缀每轮都一样吗？

---

## 十、这份教程的三个术语约定

- **"非流式"**：一次请求拿完整响应；**"流式"**：拿一串增量分片。
- **"协议层"**：`AI/LLM/` 覆盖的范围——消息、参数、响应、工具、流式、定价。
- **"运行时层"**：`AI/Agent/` 覆盖的范围——循环、上下文、记忆、审批、评测。
  两层的分界是：**这一层只跟模型说话，还是也在跟工具和状态说话。**
