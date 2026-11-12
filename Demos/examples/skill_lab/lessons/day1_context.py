"""第 1 天实验：技能的本质与边界。

这个模块解决什么问题：把"skill 有三层加载机制"从口口相传变成可测量的数字。
它盘点本机所有已安装技能，分别算出 description（永远常驻）和正文（触发才载入）的成本，
让"渐进式披露省了多少"变成一个具体数字。

对应笔记：AI/Skill/01-技能的本质与边界.md
"""
from __future__ import annotations

from ..labkit import fmt_table, scan_skills

# 正文超过这个 token 数就值得怀疑"是不是该拆了"
BODY_BUDGET = 5000


def run() -> None:
    scan = scan_skills()
    if not scan.stats:
        print("本机没有扫描到任何技能，跳过盘点。")
        return

    print("=" * 78)
    print("第 1 天 · 技能盘点：常驻成本 vs 正文成本")
    print("=" * 78)

    rows = [
        [
            s.name,
            s.origin,
            str(s.desc_tokens),
            str(s.body_tokens),
            str(s.resource_files),
        ]
        for s in sorted(scan.stats, key=lambda x: -x.body_tokens)
    ]
    print(fmt_table(
        ["技能", "来源", "描述token", "正文token", "附属文件"],
        rows,
        align_right={2, 3, 4},
    ))

    always_on = scan.always_on_tokens
    body = scan.body_tokens
    saving = body / always_on if always_on else 0.0

    print()
    print("-" * 78)
    print("账本")
    print("-" * 78)
    print(f"  技能数量            : {len(scan.stats)}")
    print(f"  常驻成本（描述合计）: {always_on:,} token   ← 每个会话都要付")
    print(f"  正文总量            : {body:,} token   ← 触发时才付")
    print(f"  渐进式披露省下      : {saving:.1f} 倍  （{(1 - always_on / body) * 100:.1f}%）")

    over = [s for s in scan.stats if s.body_tokens > BODY_BUDGET]
    if over:
        print()
        print(f"  正文超过 {BODY_BUDGET} token 的技能（该考虑拆 references 了）：")
        for s in over:
            print(f"    - {s.name}  {s.body_tokens:,} token")

    # 分层结论：描述决定"会不会被用"，正文决定"用得对不对"
    assert always_on < body, "常驻成本应当显著低于正文总量，否则说明分层失效"
    avg_desc = always_on // len(scan.stats)
    assert avg_desc < 100, f"平均描述成本 {avg_desc} token 偏高，说明 description 写得啰嗦"

    # 个别技能的描述会明显偏长——通常是因为它需要覆盖大量触发说法
    long_desc = [s for s in scan.stats if s.desc_tokens >= 150]
    if long_desc:
        print()
        print("  描述超过 150 token 的个例（需要覆盖大量触发说法时才合理）：")
        for s in long_desc:
            print(f"    - {s.name}  {s.desc_tokens} token")

    print()
    print(f"  平均描述成本：{avg_desc} token/个")

    print()
    print("要点：描述是唯一的开关，正文是按需展开的载荷。两者成本差得越远，说明分层做得越好。")


if __name__ == "__main__":
    run()
