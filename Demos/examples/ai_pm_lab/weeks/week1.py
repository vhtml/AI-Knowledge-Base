"""第 1 周实验：能力雷达与技术边界速查卡。

这个模块解决什么问题：把"我是不是个合格的 AI 产品经理"变成一张可排序的雷达，
把"这个任务能不能交给模型"变成一张能查的速查卡——两张表都是先给结论再展开细节，
不用每次开会前重新想一遍。

对应笔记：AI/AI-Product/ 第 01 章《能力雷达》与第 02 章《技术边界速查卡》。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..labkit import FixtureError, fmt_table, load_fixture

# 短板线的含义：3 分及以下，意味着这条能力在真实项目里会反复卡住。
DEFAULT_SHORTFALL_THRESHOLD = 3

# 速查卡：任务类型 → (建议做法, 常见翻车点, 产品兜底)
# 收录标准是"这一类任务真的有人踩过坑"，不是"这个模型能不能做"。
TASK_TYPES: dict[str, tuple[str, str, str]] = {
    "分类/抽取": (
        "先定少而稳的标签集，配 3–5 个示例；先做抽取再做分类，顺序反了会互相污染",
        "标签定义含糊、类别不互斥，模型在边界样本上来回摇摆",
        "低置信样本走人工队列；永远保留「无法判断」这个出口",
    ),
    "摘要": (
        "先定「摘要给谁看、用来做什么决策」，再选长摘要 / 要点式 / 结构化摘要",
        "摘要读起来通顺，却丢掉了关键否定条件和数字",
        "关键数字与时间点强制回引原句，允许点开核对",
    ),
    "写作": (
        "给结构、语气和受众，把事实与表达分开：模型只负责表达，事实由检索提供",
        "模型为了让文风完整而自行补全事实（幻觉）",
        "事实句必须带来源标记，无来源的句子不允许进终稿",
    ),
    "翻译": (
        "固定术语表 + 领域语料优先；长文分句对齐，不要整篇直译",
        "专业术语被直译、占位符与格式丢失",
        "术语表命中不一致就报错回退人工；保留原文对照",
    ),
    "复杂推理": (
        "拆成可验证的小步骤，每一步都要能引用证据；先让模型写检查清单再让它作答",
        "过程看着严谨，结论却没有证据支撑",
        "关键推理节点设置人工确认点",
    ),
    "多步规划": (
        "先出计划再执行；工具调用失败时允许重规划，而不是沿着错的路硬跑",
        "计划一旦生成就不变，遇到失败一路错到底",
        "计划变更需显式审批；每一步留执行日志",
    ),
    "代码生成": (
        "给接口契约和测试，让模型对着测试写实现",
        "生成能跑但边界条件错的代码——没测试时最危险",
        "必须有测试覆盖才能合并；危险命令只在沙箱里跑",
    ),
    "实时对话": (
        "把状态显式存在会话里，检索与工具调用做异步补偿",
        "用户打断或改口后状态没同步，副作用动作被重复执行",
        "所有有副作用的动作走幂等接口，并留确认",
    ),
    "图像理解": (
        "先确认是判别还是生成；判别类必须输出可核对的位置框",
        "对模糊、低分辨率图片给出非常自信的结论",
        "低置信度与低分辨率图片转人工",
    ),
    "长文档问答": (
        "先做结构化索引再问答，答案必须落到具体段落",
        "跨章节问题被单段证据带偏（中间丢失）",
        "跨章节问题必须同时给出多段证据",
    ),
    "表格计算": (
        "让模型写公式而不是心算；用代码执行器算，不让它口算",
        "数字错一位且难以察觉，表越大越容易错",
        "计算走确定性代码；关键数字回显计算式",
    ),
}


@dataclass
class Shortfall:
    """一条能力短板：哪一维、多少分、该补什么。"""

    dimension: str
    name: str
    score: int
    action: str
    rank: int


@dataclass
class Dimension:
    """一个能力维度的定义。"""

    id: str
    name: str
    question: str
    rubric: dict[str, str]
    action: str


def load_capabilities() -> tuple[list[Dimension], int, dict[str, int]]:
    """读取能力维度定义、短线阈值与样例自评。

    返回 (维度列表, 短板阈值, 样例自评分)。
    """
    data = load_fixture("capabilities")
    dimensions = [
        Dimension(
            id=item["id"],
            name=item["name"],
            question=item["question"],
            rubric={str(k): v for k, v in item["rubric"].items()},
            action=item["action"],
        )
        for item in data["dimensions"]
    ]
    threshold = int(data.get("shortfall_threshold", DEFAULT_SHORTFALL_THRESHOLD))
    return dimensions, threshold, {k: int(v) for k, v in data["sample_scores"].items()}


def radar(scores: dict[str, int], threshold: int | None = None) -> list[Shortfall]:
    """把自评分变成按优先级排序的短板清单。

    排序规则：分数低的在前；分数相同时保持维度定义里的先后顺序——
    这条是刻意的：同分的短板没有客观优劣，用固定顺序保证每次运行结果一致，
    这样"上周的清单"和"这周的清单"才可比。
    """
    dimensions, default_threshold, _ = load_capabilities()
    limit = default_threshold if threshold is None else threshold

    missing = [dim.id for dim in dimensions if dim.id not in scores]
    if missing:
        raise FixtureError(f"自评分缺少维度：{'、'.join(missing)}")
    extra = [key for key in scores if key not in {dim.id for dim in dimensions}]
    if extra:
        raise FixtureError(f"自评分出现未知维度：{'、'.join(extra)}")
    for dim in dimensions:
        value = scores[dim.id]
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise FixtureError(f"维度 {dim.name} 的分数必须是 1–5 的整数，实际 {value!r}")

    shortfalls: list[Shortfall] = []
    for dim in dimensions:
        if scores[dim.id] <= limit:
            shortfalls.append(
                Shortfall(
                    dimension=dim.id,
                    name=dim.name,
                    score=scores[dim.id],
                    action=dim.action,
                    rank=0,
                )
            )
    shortfalls.sort(key=lambda item: item.score)
    for index, item in enumerate(shortfalls, start=1):
        item.rank = index
    return shortfalls


def boundary_card(task_type: str) -> str:
    """取一张技术边界速查卡：建议做法、常见翻车点、产品兜底。"""
    if task_type not in TASK_TYPES:
        raise FixtureError(f"速查卡里没有这个任务类型：{task_type}")
    approach, pitfall, fallback = TASK_TYPES[task_type]
    return (
        f"【{task_type}】\n"
        f"  建议做法：{approach}\n"
        f"  常见翻车点：{pitfall}\n"
        f"  产品兜底：{fallback}"
    )


def run() -> None:
    """跑第 1 周实验：雷达 → 短板 → 速查卡。"""
    print("=" * 78)
    print("第 1 周 · 能力雷达与技术边界速查卡")
    print("=" * 78)

    dimensions, threshold, sample_scores = load_capabilities()

    # ---- 1. 能力雷达 ----
    print()
    print("【1】八维能力自评（样例数据）")
    rows: list[list[str]] = []
    for dim in dimensions:
        score = sample_scores[dim.id]
        bar = "■" * score + "□" * (5 - score)
        rows.append(
            [
                dim.name,
                str(score),
                bar,
                dim.rubric.get(str(score), ""),
                "短板" if score <= threshold else "",
            ]
        )
    print(fmt_table(["维度", "分数", "档位", "该分档的含义", "标记"], rows, align_right={1}))

    # ---- 2. 短板清单 ----
    shortfalls = radar(sample_scores)
    print()
    print(f"【2】短板清单（短板线 = {threshold} 分，共 {len(shortfalls)} 项）")
    print(
        fmt_table(
            ["优先级", "维度", "分数", "补齐动作"],
            [[str(item.rank), item.name, str(item.score), item.action] for item in shortfalls],
            align_right={0, 2},
            max_widths=[6, 10, 4, 62],
        )
    )

    # ---- 3. 技术边界速查卡 ----
    print()
    print(f"【3】技术边界速查卡（{len(TASK_TYPES)} 类任务）")
    print(
        fmt_table(
            ["任务类型", "建议做法", "常见翻车点", "产品兜底"],
            [[name, card[0], card[1], card[2]] for name, card in TASK_TYPES.items()],
            max_widths=[10, 30, 30, 30],
        )
    )
    print()
    for task_type in ("分类/抽取", "长文档问答", "表格计算"):
        print(boundary_card(task_type))
        print()

    # ---- 断言 ----
    assert len(dimensions) == 8, f"能力维度应为 8 个，实际 {len(dimensions)}"
    assert {dim.id for dim in dimensions} == set(sample_scores), "样例自评必须覆盖全部维度"
    assert shortfalls, "样例数据里应该至少有一项短板"
    assert [item.score for item in shortfalls] == sorted(item.score for item in shortfalls), (
        "短板必须按分数升序排列"
    )
    assert [item.rank for item in shortfalls] == list(range(1, len(shortfalls) + 1)), (
        "优先级编号必须连续"
    )
    assert len(TASK_TYPES) >= 10, f"速查卡至少 10 行，实际 {len(TASK_TYPES)}"
    for name, card in TASK_TYPES.items():
        assert name.strip(), "速查卡的行名不能为空"
        assert len(card) == 3, f"{name} 的速查卡必须有三列"
        assert all(part.strip() for part in card), f"{name} 的速查卡有空列"
    try:
        radar({key: 3 for key in list(sample_scores)[:7]})
    except FixtureError:
        pass
    else:
        raise AssertionError("缺维度时必须报错")

    print("要点：雷达告诉你「差在哪」，速查卡告诉你「这类活该怎么接」。")
    print(f"      短板 {len(shortfalls)} 项里，补齐动作写的是「下周做什么」，不是「要学什么」。")
    print(f"      当前最短的一块是：{shortfalls[0].name}（{shortfalls[0].score} 分）。")
    print(f"      速查卡收了 {len(TASK_TYPES)} 类任务；表里没有的，先问一句「错一次的代价是什么」。")
