"""第 3 天实验：正文与自由度分级。

这个模块解决什么问题：把"正文该写多细"变成可检查的结构。
它实现一个正文体检器（有没有决策点、有没有失败处理、篇幅是否超限），
拿仓库里的 demo-assistant 当反例重写一遍，并演示三档自由度该怎么选。

对应笔记：AI/Skill/03-正文与自由度分级.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..labkit import estimate_tokens, fmt_table, repo_root

BODY_LINE_BUDGET = 500

# 正文体检的三个信号词
DECISION_MARKERS = ("？", "?", "如果", "则", "→", "→")
FAILURE_MARKERS = ("失败", "错误", "异常", "降级", "不要", "无法", "禁止")
REDUNDANT_MARKERS = ("本技能用于", "希望这", "请注意", "综上所述")


@dataclass
class Diagnosis:
    """一份正文体检报告。"""

    name: str
    lines: int
    tokens: int
    has_decision: bool
    has_failure: bool
    has_redundant: bool
    problems: list[str]

    @property
    def score(self) -> int:
        return sum([
            self.lines <= BODY_LINE_BUDGET,
            self.has_decision,
            self.has_failure,
            not self.has_redundant,
        ])


def diagnose(name: str, body: str) -> Diagnosis:
    """体检一份 SKILL.md 正文。"""
    lines = body.splitlines()
    problems: list[str] = []

    has_decision = any(m in body for m in DECISION_MARKERS)
    has_failure = any(m in body for m in FAILURE_MARKERS)
    has_redundant = any(m in body for m in REDUNDANT_MARKERS)

    if len(lines) > BODY_LINE_BUDGET:
        problems.append(f"正文 {len(lines)} 行，超过 {BODY_LINE_BUDGET} 行上限")
    if not has_decision:
        problems.append("没有决策点：遇到岔路时模型只能自己发挥")
    if not has_failure:
        problems.append("没有失败处理：模型遇到意外会默认自己想办法")
    if has_redundant:
        problems.append("含有可删的自我说明/客套话")

    return Diagnosis(
        name=name,
        lines=len(lines),
        tokens=estimate_tokens(body),
        has_decision=has_decision,
        has_failure=has_failure,
        has_redundant=has_redundant,
        problems=problems,
    )


# 重写版：按「流程 / 决策点 / 边界与失败处理」三块骨架写
REWRITTEN = """# demo-assistant

演示技能加载。加载后按下面的规则回答。

## 流程

1. 用一句话说明已加载 `demo-assistant`。
2. 给出结论本身，不写铺垫。
3. 结论超过 3 条时用编号列表。

## 决策点

- 结论依赖外部事实（价格、版本号、新闻）？→ 先调用 `web_search` 再作答。
- 需要查看或修改本机文件？→ 先调用 `run_command`。
- 两者都不需要？→ 直接回答，不调工具。

## 边界与失败处理

- `web_search` 无结果时不要编造，回答"没有查到相关信息"并给出建议的检索词。
- 用户要求输出很长时，先给结论摘要，再问是否需要展开。
"""


# 三条真实指令，各判一个自由度档位
FREEDOM_CASES = [
    (
        "优先复用仓库里已有的模块，不要重复造轮子",
        "高",
        "换个实现也能接受，取决于上下文；给原则比给命令更合适",
    ),
    (
        "用 scripts/rotate_pdf.py <路径> <角度> 旋转 PDF",
        "中",
        "有推荐做法，允许换参数；但不该让模型手写 PDF 变换代码",
    ),
    (
        "密钥只写进 Demos/.env，不要写进任何会被提交的文件",
        "低",
        "做错就是安全事故，一致性比灵活性重要，必须没有选择空间",
    ),
]


def run() -> None:
    print("=" * 78)
    print("第 3 天 · 正文与自由度分级")
    print("=" * 78)

    # ---- 1. 体检仓库现有样例 ----
    demo = repo_root() / "Demos" / "examples" / "skills" / "demo-assistant" / "SKILL.md"
    before = None
    if demo.is_file():
        text = demo.read_text(encoding="utf-8")
        body = text.split("---", 2)[-1]
        before = diagnose("demo-assistant（原版）", body)

        print()
        print("【1】体检仓库现有的 demo-assistant")
        print(fmt_table(
            ["指标", "值"],
            [
                ["正文行数", str(before.lines)],
                ["正文 token", str(before.tokens)],
                ["有决策点", "是" if before.has_decision else "否"],
                ["有失败处理", "是" if before.has_failure else "否"],
                ["有冗余自我说明", "是" if before.has_redundant else "否"],
                ["体检得分", f"{before.score}/4"],
            ],
            align_right={1},
        ))
        for p in before.problems:
            print(f"  ✗ {p}")

    # ---- 2. 重写 ----
    after = diagnose("demo-assistant（重写版）", REWRITTEN)
    print()
    print("【2】按骨架重写之后")
    print(fmt_table(
        ["指标", "原版", "重写版"],
        [
            ["正文行数", str(before.lines) if before else "-", str(after.lines)],
            ["正文 token", str(before.tokens) if before else "-", str(after.tokens)],
            ["有决策点", "是" if before and before.has_decision else "否", "是" if after.has_decision else "否"],
            ["有失败处理", "是" if before and before.has_failure else "否", "是" if after.has_failure else "否"],
            ["体检得分", f"{before.score}/4" if before else "-", f"{after.score}/4"],
        ],
        align_right={1, 2},
    ))
    print()
    print("  重写版正文：")
    for line in REWRITTEN.splitlines():
        print(f"    {line}")

    # ---- 3. 自由度分档 ----
    print()
    print("【3】同一件事的三档自由度")
    rows = [[stmt, level, why] for stmt, level, why in FREEDOM_CASES]
    print(fmt_table(["指令", "档位", "理由"], rows))

    assert after.score >= (before.score if before else 0), "重写版不应该比原版差"
    assert after.has_decision and after.has_failure

    print()
    print("要点：重写可能让正文变长，但多出来的每一行都在做决策或兜底——")
    print("      这正是'该写的没写'比'写了太多'更贵的道理。")


if __name__ == "__main__":
    run()
