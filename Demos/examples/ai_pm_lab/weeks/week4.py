"""第 4 周实验：信任式回答与检索策略评估。

这个模块解决什么问题：给"回答得像不像样"一个可检查的定义（四要素），
给"检索有没有用"三个可算的指标（召回命中、引用正确、忠实度），
让"我觉得它答得还行"变成"这三条过线了吗"。

检索指标是启发式离线算法——它算的是接线和接地，不是事实本身对不对；
真要判断事实正确，还得靠人抽查或另一套标注。写清楚这一点，比指标本身更重要。

对应笔记：AI/AI-Product/ 第 07 章《信任式回答》与第 08 章《检索评估》。
"""
from __future__ import annotations

from ..labkit import FixtureError, Report, fmt_table, load_fixture, pct

# 信任四要素：(要素名, 为什么必须有, 识别用的标记)
# 标记是启发式的关键词——用来做"模板体检"，不是用来判定语义正确性。
TRUST_ELEMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "来源引用",
        "用户得能自己核一遍，否则回答再准也没法被信任",
        ("来源", "引用", "参考", "依据", "出处", "文档", "http"),
    ),
    (
        "置信表达",
        "模型必须说清自己有多确定，把不确定甩给用户",
        ("置信", "把握", "大概率", "很可能", "不太确定", "不确定", "约有", "大约", "可能有"),
    ),
    (
        "拒答出口",
        "答不了的时候要有路可走，而不是硬编一个答案",
        ("无法", "不能确定", "没有找到", "建议人工", "转人工", "不知道", "请补充", "需要您提供"),
    ),
    (
        "可撤销路径",
        "用户要能推翻答案——不撤销的回答会变成不可逆的错误",
        ("撤销", "回滚", "撤回", "改写", "重新生成", "编辑", "确认后", "可修改"),
    ),
)

# 三条检索指标的阈值。为什么是这三个数：
# 召回 0.80——十条里捞上来八条，低于这条说明知识库本身或切分策略要重做；
# 引用 0.90——引用错一次，用户就有理由不再相信整段回答，所以比召回更严；
# 忠实 0.80——三成断言悬空就等于编造，宁可少答。
THRESHOLDS: dict[str, float] = {
    "recall": 0.80,
    "citation": 0.90,
    "faithfulness": 0.80,
}

METRIC_LABELS: dict[str, str] = {
    "recall": "召回命中率",
    "citation": "引用正确率",
    "faithfulness": "回答忠实度",
}

METRIC_DEFS: dict[str, str] = {
    "recall": "检索结果里至少命中一条期望来源的查询占比",
    "citation": "被引用的来源中，确实来自期望来源的比例",
    "faithfulness": "支撑来源全部出现在检索结果里的断言占比（接地率）",
}


def check_answer(answer: str) -> Report:
    """检查一份回答模板是否具备信任四要素。

    缺哪一项就点名哪一项——不给笼统的"格式不对"，因为没法改。
    """
    if not isinstance(answer, str) or not answer.strip():
        raise FixtureError("回答不能为空")

    report = Report(subject="回答模板", passed=True)
    for name, why, markers in TRUST_ELEMENTS:
        if any(marker in answer for marker in markers):
            report.ok.append(name)
        else:
            report.problems.append(f"缺{name}")
            report.notes.append(f"{name}：{why}")
    report.passed = not report.problems
    return report


def retrieval_metrics(cases: list[dict]) -> dict:
    """算三项检索指标，并各自给出阈值判定。

    三个定义写在 METRIC_DEFS 里，这里只负责按定义算——定义和算法分开放，
    免得"改阈值"和"改算法"混成一件事。
    """
    if not cases:
        raise FixtureError("检索样本不能为空")

    hits = 0
    cited_total = 0
    cited_correct = 0
    claims_total = 0
    claims_supported = 0
    details: list[list[str]] = []

    for case in cases:
        expected = set(case["expected"])
        retrieved = set(case["retrieved"])
        cited = list(case["cited"])
        claims = case["claims"]

        hit = bool(expected & retrieved)
        hits += 1 if hit else 0

        for source in cited:
            cited_total += 1
            if source in expected:
                cited_correct += 1

        for claim in claims:
            claims_total += 1
            if set(claim["supported_by"]) <= retrieved:
                claims_supported += 1

        details.append(
            [
                case["id"],
                "命中" if hit else "未命中",
                f"{len(expected & retrieved)}/{len(expected)}",
                f"{len(cited)} 条引用",
                f"{len(claims)} 条断言",
            ]
        )

    total = len(cases)
    scores = {
        "recall": hits / total,
        "citation": cited_correct / cited_total if cited_total else 0.0,
        "faithfulness": claims_supported / claims_total if claims_total else 0.0,
    }
    verdicts = {
        key: {
            "score": value,
            "threshold": THRESHOLDS[key],
            "passed": value >= THRESHOLDS[key],
            "label": METRIC_LABELS[key],
            "definition": METRIC_DEFS[key],
        }
        for key, value in scores.items()
    }
    return {
        "cases": total,
        "metrics": verdicts,
        "passed": all(item["passed"] for item in verdicts.values()),
        "details": details,
        "notes": [
            "忠实度算的是「断言的支撑来源有没有被检索到」，不等于事实正确率——",
            "事实正确率需要人工抽查或另一套标注，这是两件事。",
        ],
    }


def run() -> None:
    """跑第 4 周实验：信任四要素 → 检索三指标。"""
    print("=" * 78)
    print("第 4 周 · 信任式回答与检索策略评估")
    print("=" * 78)

    # ---- 1. 信任四要素 ----
    print()
    print("【1】回答模板体检（四要素）")
    samples = [
        (
            "合格样例",
            "根据《退款政策 v3》第 2 节（来源：知识库 S-101），签收后 7 天内可无理由退货。"
            "这条我把握比较大；如果商品已激活则不适用，不能确定请以人工工单为准。"
            "你可以改写这段回答，也可以撤销后让我重新生成。",
        ),
        (
            "缺要素样例",
            "签收后 7 天内可以退货。",
        ),
    ]
    for label, text in samples:
        report = check_answer(text)
        print()
        print(f"  ▸ {label}：{'通过' if report.passed else '不通过'}")
        print(f"    通过项：{'、'.join(report.ok) if report.ok else '—'}")
        if report.problems:
            print(f"    缺项：{'、'.join(report.problems)}")
        for note in report.notes:
            print(f"    · {note}")

    # ---- 2. 检索三指标 ----
    data = load_fixture("retrieval_cases")
    result = retrieval_metrics(data["cases"])
    print()
    print("【2】检索策略评估")
    print(
        fmt_table(
            ["指标", "定义", "得分", "阈值", "判定"],
            [
                [
                    item["label"],
                    item["definition"],
                    pct(item["score"]),
                    pct(item["threshold"], 0),
                    "过线" if item["passed"] else "未过线",
                ]
                for item in result["metrics"].values()
            ],
            align_right={2, 3},
            max_widths=[12, 40, 8, 8, 8],
        )
    )
    print()
    print(f"  逐条明细（{result['cases']} 条样本）")
    print(
        fmt_table(
            ["编号", "召回", "命中来源", "引用", "断言"],
            result["details"],
            align_right={2},
        )
    )
    for note in result["notes"]:
        print(f"  · {note}")

    # ---- 断言 ----
    good = check_answer(samples[0][1])
    assert good.passed, "四要素齐全的样例必须通过"
    assert len(good.ok) == 4, "四要素齐全时四项都要记为通过"
    bad = check_answer(samples[1][1])
    assert not bad.passed, "缺要素的样例必须不通过"
    assert "缺来源引用" in bad.problems and "缺置信表达" in bad.problems
    assert "缺拒答出口" in bad.problems and "缺可撤销路径" in bad.problems
    for report in (good, bad):
        assert len(report.ok) + len(report.problems) == 4, "每份报告都要点名全部四要素"

    for key, item in result["metrics"].items():
        assert 0 <= item["score"] <= 1, f"{key} 的得分必须在 0–1"
        assert item["passed"] == (item["score"] >= item["threshold"]), f"{key} 的阈值判定不稳定"
    assert set(result["metrics"]) == {"recall", "citation", "faithfulness"}
    again = retrieval_metrics(data["cases"])
    assert again["metrics"] == result["metrics"], "同一批样本两次计算必须一致"
    try:
        retrieval_metrics([])
    except FixtureError:
        pass
    else:
        raise AssertionError("空样本必须报错")

    print()
    print("要点：召回过线不等于可信——上面这批评测里召回达标，")
    print("      但引用正确率被检索结果之外的来源拉垮了，这一条就足以拦住上线。")
    print("      忠实度也只是「接地率」，事实对不对仍然要人抽查。")
