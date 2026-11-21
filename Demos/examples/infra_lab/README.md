# infra_lab —— AI Infra 推理服务教程实验包

18 章 + 毕业作品的**可运行实验**。

配套笔记：[`AI/AI-Infra/`](../../../AI/AI-Infra/README.md)

## 特点

- **离线**：不联网、不调模型、不需要 GPU、不花 token
- **确定性**：同样的输入永远得到同样的输出，断言可以写得很死
- **框架无关**：不 import vLLM / SGLang / TensorRT-LLM。第 18 章只在文字里做机制对照
- **自包含**：所有算账都走 `labkit.py`，各章不重复实现公式

跑挂了就是没学会——这是这套实验的设计意图。

## 跑起来

```bash
cd Demos/examples

uv run --project .. python -m infra_lab list        # 列出 18 章
uv run --project .. python -m infra_lab ch07        # 跑第 7 章
uv run --project .. python -m infra_lab all         # 跑全部 18 章
uv run --project .. python -m infra_lab selftest    # 全量自检（笔记 + 实验 + 毕设）
uv run --project .. python -m infra_lab capstone     # 跑毕业作品
uv run --project .. python -m infra_lab capstone.selftest   # 毕业作品自检
```

## 目录结构

```
infra_lab/
├── labkit.py        # 公共层：GPU/模型规格、显存账、FLOPs、Roofline、
│                    #           分页 KV、量化、推测解码、引擎模拟器
├── lessons/         # 18 个章节实验模块，与笔记一一对应
│   ├── __init__.py  #   章节登记表（加一章只需要在这里加一行）
│   ├── ch01_slo.py … ch18_ecosystem.py
├── capstone/        # 毕业作品：迷你推理服务
│   ├── miniserve.py #   分页 KV + 连续批处理 + 前缀缓存 + 亲和路由 + 限流
│   ├── scenario.py  #   演示负载
│   └── selftest.py  #   逐条机制证明其生效
├── selftest.py      # 全量自检：笔记 ↔ 实验对应、逐章断言、毕设自检
└── __main__.py      # 命令入口
```

## 公共层提供的算账口径

**所有章节共用同一套口径**，任何数字都能在 `labkit.py` 里找到出处：

| 领域 | API | 说明 |
| --- | --- | --- |
| 硬件 | `GPUS` / `gpu(name)` | 9 张卡的规格：显存、带宽、FP16/FP8/INT8 算力、卡时单价 |
| 模型 | `MODELS` / `model(name)` | 7 个开放模型的完整 config |
| 显存 | `plan_memory(...)` | 权重 / KV / 激活 / 运行时 四段账本 |
| 算力 | `prefill_flops` / `decode_flops` | prefill 与 decode 分开算 |
| 上界 | `roofline(...)` | 判断受算力还是带宽限制 |
| 时延 | `prefill_time` / `decode_step_time` | 解析模型，不跑真实 kernel |
| KV | `PagedKVCache` / `block_chain` | 分页块表 + 块级链式哈希 |
| 量化 | `quantize_values(...)` | 分组对称量化的误差与 SNR |
| 调度 | `EngineSimulator` | 单引擎连续批处理模拟器，支持三种调度策略 |

**统一口径**：
- 显存用 **GiB**（2³⁰ 字节），带宽用 **GB/s**（硬件厂商标称口径）
- 算力取**稠密**值，不含结构化稀疏
- 所有 GPU 按**单卡**计算，跨卡能力在第 17 章单独讲
- 所有时延用**解析模型**估算，不跑真实 GPU

## 各章实验

| 章 | 主题 | 实验在算什么 |
| --- | --- | --- |
| 01 | 推理服务全景与 SLO | 三种负载画像的 SLO 表现与每请求成本 |
| 02 | GPU 规格与显存账本 | GPU × 模型网格的最大可并发 token 数 |
| 03 | Roofline 与算术强度 | prefill/decode 的算术强度与拐点对比 |
| 04 | 访存墙与批量摊薄 | 扫描 batch size 找 decode 延迟拐点 |
| 05 | KV Cache | 多模型 × 多上下文长度的 KV 账本 |
| 06 | PagedAttention 原理 | 连续预留 vs 分页的碎片率对比 |
| 07 | 连续批处理 | 静态批 vs 连续批处理的 p99 对比 |
| 08 | 多租户调度 | FCFS/SJF/优先级三策略的 SLO 与饿死分析 |
| 09 | 量化 | bits × group_size 的 SNR 扫描与带宽收益 |
| 10 | 推测解码 | 接受率 → 加速比曲线与盈亏平衡点 |
| 11 | 网关 | 前缀亲和路由命中率对比 + 令牌桶削峰 |
| 12 | 缓存体系 | 前缀缓存块复用、LRU 淘汰、三级缓存对比 |
| 13 | 提示词流水线 | 长 prompt 的 TTFT 尖刺与 chunked prefill 改善 |
| 14 | 可观测性 | 延迟三段分解与 p99 长尾归因 |
| 15 | 弹性 | 有扩容延迟的队列模型与最优副本数 |
| 16 | 容量规划 | 反推所需副本数与每百万 token 成本 |
| 17 | 训练侧地基 | 四种并行策略的通信量与 ZeRO 显存拆分 |
| 18 | 生态地图 | 机制 → 框架能力对照表与选型打分 |

## 毕业作品

一个能跑的迷你推理服务，把第 5、6、7、11、12 章的机制装在一起：

```
请求 → 令牌桶限流 → 前缀亲和路由 ─┬→ replica-0
                                  └→ replica-1
                                        │
                                   连续批处理调度器
                                   ├─ 分页 KV 缓存（块表 + 引用计数）
                                   ├─ 前缀缓存（块级链式哈希）
                                   └─ 每轮补位，长短请求不互相拖死
```

自检不满足于"跑通"，而是逐条机制去证明它真的在起作用：
前缀缓存必须真的让 TTFT 下降、亲和路由必须真的把请求钉在同一副本、
限流器必须真的削平突发、分页碎片率必须真的低于连续预留。

## 加一章

1. 在 `lessons/` 下新建 `chNN_xxx.py`，导出 `run() -> None`
2. 在 `lessons/__init__.py` 的 `LESSONS` 里加一行 `Lesson(...)`
3. 在 `AI/AI-Infra/` 写对应笔记
4. 在 `selftest.py` 的 `NOTE_FILES` 里加上笔记文件名
5. 跑 `uv run --project .. python -m infra_lab.selftest`
