"""毕业作品：一个能跑的迷你推理服务。

这个模块解决什么问题：18 章学下来，每个机制都有单独的实验，
但**真正的工作是把它们装到一起**。这个毕业作品把前 12 章的核心机制
装进一个可以运行的服务里，让读者亲眼看到"这些机制合在一起是什么形状"。

装进去的机制：

- **第 5、6 章**：分页 KV 缓存（块表 + 引用计数 + 碎片率）
- **第 7 章**：连续批处理（iteration 级调度，每轮补位）
- **第 11 章**：令牌桶限流 + 前缀亲和路由
- **第 12 章**：前缀缓存（block 粒度哈希 + 复用）
- **第 8、14 章**：多租户优先级 + SLO 判定与指标汇总

**没有装进去的**（毕业作品不是完整推理引擎）：真实 kernel、NCCL 通信、
多卡切分、量化 kernel、推测解码。这些在真实系统里由框架负责，
教程第 18 章会讲清各自的边界。

全程离线：用解析模型算时间，不碰 GPU。
"""

from __future__ import annotations

from .miniserve import (
    MiniServe,
    RateLimiter,
    Replica,
    ServeConfig,
    ServeReport,
    Sequence,
)
from .scenario import build_load, run_demo

__all__ = [
    "MiniServe",
    "RateLimiter",
    "Replica",
    "ServeConfig",
    "ServeReport",
    "Sequence",
    "build_load",
    "run_demo",
    "run",
]


def run() -> int:
    """跑毕业作品的演示场景，打印服务报告。

    返回进程退出码：0 成功。
    """
    report = run_demo()
    report.render()
    return 0
