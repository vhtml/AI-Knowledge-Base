"""毕业作品的演示场景：一段有代表性的服务负载。

这个模块解决什么问题：自检需要一份**可复现的负载**才能断言，
演示也需要一份**读得懂**的负载才说明问题。这里构造的负载刻意还原
真实推理服务的三个典型特征：

1. **共享系统提示词**——所有请求都带同一份 800 token 的 system prompt，
   只有用户问题不同。这是前缀缓存收益最大的形态（第 12 章）。
2. **长短混合**——既有 100 token 的短问答，也有 4000 token 的长文档问答。
   长短混排正是连续批处理要解决的问题（第 7 章）。
3. **多租户**——两个租户，一个是高优先级的交互式应用，
   一个是低优先级的批处理应用（第 8 章）。
"""

from __future__ import annotations

from .miniserve import MiniServe, ServeConfig, ServeReport, ServeRequest

__all__ = ["build_load", "build_config", "run_demo"]


SYSTEM_PROMPT = "你是一个严谨的技术助手，必须基于给定材料回答，并标注引用来源。"
"""所有请求共享的系统提示词片段 key。"""

SHARED_DOC = "参考资料：检索增强系统设计文档，第 3 章讲索引构建，第 5 章讲重排序。"


def build_config(**overrides: object) -> ServeConfig:
    """构造演示用的服务配置。"""
    defaults: dict[str, object] = {
        "device_name": "h100_sxm",
        "model_name": "qwen2.5-7b",
        "replicas": 2,
        "max_batch": 16,
        "kv_token_budget": 120_000,
        "admission_rate_per_s": 0.0,
        "slo_ttft_ms": 2000.0,
        "slo_tpot_ms": 100.0,
    }
    defaults.update(overrides)
    return ServeConfig(**defaults)  # type: ignore[arg-type]


def build_load() -> list[ServeRequest]:
    """构造演示负载。

    时间轴刻意做成"前松后紧"：先来一波短问答（服务有空闲），
    再来一波长文档请求（把 batch 填满），最后再来几个交互式短请求
    ——这时候长请求还在跑，短请求的 TTFT 会被拖高，
    正好暴露连续批处理的效果。
    """
    requests: list[ServeRequest] = []
    req_id = 1

    # 第一波：共享前缀的短问答，间隔 0.3s
    for i in range(6):
        requests.append(
            ServeRequest(
                req_id=req_id,
                arrival_s=i * 0.3,
                segments=((SYSTEM_PROMPT, 800), (f"短问题-{i}", 100)),
                max_output_tokens=64,
                tenant="interactive",
                priority=0,
            )
        )
        req_id += 1

    # 第二波：共享前缀的长文档问答，紧挨着来
    for i in range(4):
        requests.append(
            ServeRequest(
                req_id=req_id,
                arrival_s=1.8 + i * 0.1,
                segments=((SYSTEM_PROMPT, 800), (SHARED_DOC, 4000), (f"长问题-{i}", 200)),
                max_output_tokens=256,
                tenant="batch",
                priority=5,
            )
        )
        req_id += 1

    # 第三波：交互式请求，此时长请求还在跑
    for i in range(4):
        requests.append(
            ServeRequest(
                req_id=req_id,
                arrival_s=3.0 + i * 0.2,
                segments=((SYSTEM_PROMPT, 800), (f"追问-{i}", 80)),
                max_output_tokens=48,
                tenant="interactive",
                priority=0,
            )
        )
        req_id += 1

    return requests


def run_demo(config: ServeConfig | None = None) -> ServeReport:
    """跑一遍演示负载，返回报告。"""
    config = config or build_config()
    service = MiniServe(config)
    return service.run(build_load())
