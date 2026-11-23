"""毕业作品自检：验证迷你推理服务的每条机制都真的在起作用。

这个模块解决什么问题：毕业作品把 12 章的机制装在一起，最容易出的问题是
"代码跑通了，但机制其实没生效"——比如前缀缓存命中率永远是 0、
限流器形同虚设、亲和路由退化成随机。

所以自检不满足于"跑通"，而是**逐条机制去证明它起作用**：

1. 前缀缓存真的命中，而且命中后 TTFT 明显下降
2. 前缀亲和路由真的把共享前缀的请求送到同一个副本
3. 令牌桶真的削平了突发
4. 分页 KV 的碎片率真的低于"按最大长度预留"
5. 连续批处理真的让短请求不被长请求拖死
6. 两次运行结果完全一致（确定性）

不依赖 pytest、不联网、不需要 GPU：

    cd Demos/examples
    uv run --project .. python -m infra_lab.capstone.selftest
"""

from __future__ import annotations

from ..labkit import PagedKVCache, ensure, fmt_ms, print_table
from .miniserve import MiniServe, RateLimiter, ServeRequest
from .scenario import SYSTEM_PROMPT, build_config, build_load, run_demo

__all__ = ["run"]


def _request(
    req_id: int,
    arrival_s: float,
    question: str,
    *,
    prompt_tokens: int = 100,
    output_tokens: int = 64,
    tenant: str = "interactive",
    priority: int = 0,
) -> ServeRequest:
    """构造一条带共享系统提示词的请求。"""
    return ServeRequest(
        req_id=req_id,
        arrival_s=arrival_s,
        segments=((SYSTEM_PROMPT, 800), (question, prompt_tokens)),
        max_output_tokens=output_tokens,
        tenant=tenant,
        priority=priority,
    )


# ---------------------------------------------------------------- 1. 前缀缓存


def check_prefix_cache_works() -> list[str]:
    """前缀缓存要真的命中，并且命中后 TTFT 明显下降。"""
    failures: list[str] = []

    service = MiniServe(build_config(replicas=1))
    # 同样的问题问两次：第一条冷启动，第二条应当命中前缀缓存
    service.run(
        [
            _request(1, 0.0, "同一个问题"),
            _request(2, 5.0, "同一个问题"),
        ]
    )

    completions = sorted(
        (c for replica in service.replicas for c in replica.completions),
        key=lambda c: c.request.req_id,
    )
    if len(completions) != 2:
        failures.append(f"应当完成 2 条请求，实际 {len(completions)}")
        return failures

    cold, hot = completions
    if not hot.cache_hit:
        failures.append("第二条请求应当命中前缀缓存，实际未命中")
    if cold.cache_hit:
        failures.append("第一条请求不该命中前缀缓存（缓存是冷的）")
    if hot.ttft_s >= cold.ttft_s:
        failures.append(
            f"命中缓存后 TTFT 应更短：冷 {fmt_ms(cold.ttft_s)} vs 热 {fmt_ms(hot.ttft_s)}"
        )

    print(
        f"  冷启动 TTFT {fmt_ms(cold.ttft_s)} → 命中缓存 {fmt_ms(hot.ttft_s)}"
        f"（prompt {hot.request.prompt_tokens} token）"
    )
    return failures


# ---------------------------------------------------------------- 2. 亲和路由


def check_affinity_routing() -> list[str]:
    """相同前缀的请求应当落到同一个副本。"""
    failures: list[str] = []

    service = MiniServe(build_config(replicas=3))
    requests = [_request(i, 0.0, f"问题-{i}") for i in range(6)]
    for request in requests:
        service.route(request)

    landing = [service.route(request).name for request in requests]
    # 第一条建立缓存，后面的应当都跟着它走
    if len(set(landing)) != 1:
        failures.append(f"相同前缀的请求应落到同一副本，实际分布 {landing}")

    print(f"  6 条共享前缀请求的落点：{landing}")
    return failures


# ---------------------------------------------------------------- 3. 限流


def check_rate_limiter() -> list[str]:
    """令牌桶要真的削平突发。"""
    failures: list[str] = []

    limiter = RateLimiter(rate_per_s=10.0, burst=5.0)
    allowed_at_zero = sum(1 for _ in range(20) if limiter.allow(0.0))
    if allowed_at_zero != 5:
        failures.append(f"桶容量 5 时应放行 5 个，实际放行 {allowed_at_zero}")

    # 一秒后应补充 10 个令牌，但桶上限仍是 5
    allowed_after = sum(1 for _ in range(20) if limiter.allow(1.0))
    if allowed_after != 5:
        failures.append(f"一秒后桶应补满到上限 5，实际放行 {allowed_after}")

    # 稳态速率：10 秒内应放行约 100 个
    limiter2 = RateLimiter(rate_per_s=10.0, burst=1.0)
    total = 0
    for step in range(100):
        if limiter2.allow(step * 0.1):
            total += 1
    if not 90 <= total <= 110:
        failures.append(f"稳态 10 秒应放行约 100 个，实际 {total}")

    print(f"  突发 20 个 → 立即放行 {allowed_at_zero} 个；稳态 10 秒放行 {total} 个")
    return failures


# ---------------------------------------------------------------- 4. 分页碎片


def check_paged_fragmentation() -> list[str]:
    """分页分配的碎片率应远低于"按最大长度预留"。"""
    failures: list[str] = []

    block_size = 16
    # 连续分配：每条序列都按最大长度预留 4096 token
    reserved_blocks = 4096 // block_size
    reserved_total = reserved_blocks * 4
    reserved_used = (800 + 100) // block_size * 4
    reserved_waste = (reserved_total - reserved_used) / reserved_total

    # 分页分配：按需分配，每条序列 900 token
    paged = PagedKVCache(block_size=block_size)
    for _ in range(4):
        remaining = 900
        while remaining > 0:
            block_id = paged.allocate()
            paged.append_tokens(block_id, min(block_size, remaining))
            remaining -= block_size

    paged_waste = paged.fragmentation()
    paged_used_blocks = paged.used_blocks()

    if paged_waste >= reserved_waste:
        failures.append(
            f"分页碎片率 {paged_waste:.1%} 应低于连续预留 {reserved_waste:.1%}"
        )
    if paged_used_blocks >= reserved_total:
        failures.append(
            f"分页实际占用 {paged_used_blocks} 块，应少于连续预留的 {reserved_total} 块"
        )

    print(
        f"  连续预留浪费 {reserved_waste:.1%}（{reserved_total} 块）"
        f" → 分页浪费 {paged_waste:.1%}（{paged_used_blocks} 块）"
    )
    return failures


# ---------------------------------------------------------------- 5. 连续批处理


def check_continuous_batching() -> list[str]:
    """短请求不该被长请求拖死。"""
    failures: list[str] = []

    service = MiniServe(build_config(replicas=1))
    # 先来一条超长请求（4000 token prompt + 512 输出），会占住 batch 很久
    long_request = ServeRequest(
        req_id=1,
        arrival_s=0.0,
        segments=((SYSTEM_PROMPT, 800), ("超长文档", 4000)),
        max_output_tokens=512,
        tenant="batch",
        priority=5,
    )
    # 紧接着来一条短请求
    short_request = _request(2, 0.05, "短问题")
    service.run([long_request, short_request])

    completions = sorted(
        (c for replica in service.replicas for c in replica.completions),
        key=lambda c: c.request.req_id,
    )
    if len(completions) != 2:
        failures.append(f"应当完成 2 条请求，实际 {len(completions)}")
        return failures

    long_c, short_c = completions
    if short_c.ttft_s >= long_c.ttft_s:
        failures.append(
            f"短请求 TTFT {fmt_ms(short_c.ttft_s)} 不该慢过长请求 {fmt_ms(long_c.ttft_s)}"
        )
    # 短请求在长请求跑完之前就该拿到第一个 token
    if short_c.prefill_end_s >= long_c.finish_s:
        failures.append(
            "短请求应在长请求结束前就拿到首 token（连续批处理的意义所在）"
        )

    print(
        f"  长请求跑 {fmt_ms(long_c.finish_s)}，短请求首 token 在 {fmt_ms(short_c.prefill_end_s)}"
        f"（TTFT {fmt_ms(short_c.ttft_s)}）"
    )
    return failures


# ---------------------------------------------------------------- 6. 确定性


def check_determinism() -> list[str]:
    """两次跑必须得到完全一样的结果。"""
    failures: list[str] = []

    first = run_demo()
    second = run_demo()

    pairs = [
        ("TTFT p50", first.ttft_p(0.50), second.ttft_p(0.50)),
        ("TTFT p99", first.ttft_p(0.99), second.ttft_p(0.99)),
        ("TPOT p50", first.tpot_p(0.50), second.tpot_p(0.50)),
        ("吞吐", first.output_tokens_per_s(), second.output_tokens_per_s()),
        ("缓存命中率", first.cache_hit_rate(), second.cache_hit_rate()),
    ]
    for name, left, right in pairs:
        if abs(left - right) > 1e-9:
            failures.append(f"{name} 两次运行不一致：{left} vs {right}")

    if len(first.completions) != len(second.completions):
        failures.append("两次运行的完成请求数不一致")

    print(f"  两次运行完成 {len(first.completions)} 条请求，指标完全一致")
    return failures


# ---------------------------------------------------------------- 7. 报告可用性


def check_report_metrics() -> list[str]:
    """报告里的指标要自洽。"""
    failures: list[str] = []

    report = run_demo()

    if report.submitted != report.accepted + report.rejected:
        failures.append(
            f"提交数 {report.submitted} 应等于放行 {report.accepted} + 拒绝 {report.rejected}"
        )
    if not 0.0 <= report.cache_hit_rate() <= 1.0:
        failures.append(f"缓存命中率越界：{report.cache_hit_rate()}")
    if report.ttft_p(0.50) > report.ttft_p(0.99):
        failures.append("TTFT p50 不应大于 p99")
    if report.output_tokens_per_s() <= 0:
        failures.append("吞吐应为正数")
    if report.cost_per_million_tokens_usd() <= 0:
        failures.append("成本应为正数")
    if len(report.completions) != report.submitted:
        failures.append(
            f"完成数 {len(report.completions)} 应等于提交数 {report.submitted}"
        )

    print(
        f"  提交 {report.submitted} 条 · 缓存命中 {report.cache_hit_rate():.1%}"
        f" · TTFT p99 {report.ttft_p(0.99):.0f}ms"
        f" · 吞吐 {report.output_tokens_per_s():.0f} token/s"
        f" · ${report.cost_per_million_tokens_usd():.3f}/M token"
    )
    return failures


def run() -> list[str]:
    """跑完毕业作品的所有自检。

    返回失败项列表，空列表表示全部通过。
    """
    failures: list[str] = []

    checks = (
        ("前缀缓存命中并降低 TTFT", check_prefix_cache_works),
        ("前缀亲和路由", check_affinity_routing),
        ("令牌桶限流削峰", check_rate_limiter),
        ("分页 KV 碎片率", check_paged_fragmentation),
        ("连续批处理不拖死短请求", check_continuous_batching),
        ("运行确定性", check_determinism),
        ("报告指标自洽", check_report_metrics),
    )

    for title, check in checks:
        print(f"检查{title}……")
        result = check()
        failures.extend(result)
        if result:
            for item in result:
                print(f"  - {item}")

    print()
    print("=" * 72)
    if failures:
        print(f"毕业作品自检未通过：{len(failures)} 项")
        return failures
    print(f"毕业作品自检通过：{len(checks)} 组机制全部生效")
    return failures


if __name__ == "__main__":
    import sys

    result = run()
    sys.exit(1 if result else 0)
