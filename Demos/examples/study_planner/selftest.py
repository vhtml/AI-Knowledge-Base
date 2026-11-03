"""自检脚本：验证两级记忆的核心不变量。

不依赖 pytest，直接运行：

    cd Demos/examples
    python -m study_planner.selftest

重点检查的是记忆的行为约定——这些约定一旦破了，
示例想讲的东西就讲不明白了：
- 短期记忆满了要折叠，不是丢弃
- 跨天时先晋升再清空，顺序不能反
- 长期记忆同一槽位只保留最新说法，旧的仍可追溯
- 重复提到同一件事要强化，不能塞成两条
- 召回要能按相关度排序
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

from .agent import (
    INTENT_CHAT,
    INTENT_DONE,
    INTENT_GOAL,
    INTENT_PLAN,
    INTENT_QUERY,
    INTENT_REVIEW,
    StudyPlanner,
    detect_intent,
)
from .memory import (
    LongTermMemory,
    MemoryManager,
    MemoryStore,
    ShortTermMemory,
    tokenize,
)
from .models import (
    CATEGORY_GOAL,
    CATEGORY_NOTE,
    CATEGORY_PREFERENCE,
    CATEGORY_WEAKNESS,
    Goal,
    PlanItem,
    Review,
)
from .planner import (
    build_milestones,
    detect_domain,
    distribute,
    generate_plan,
    items_for_day,
    next_undone,
    progress_of,
)
from .review import apply_review_findings, find_patterns, run_review

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """登记一条检查结果。"""

    global PASSED, FAILED

    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f"  → {detail}" if detail else ""))


def test_tokenize() -> None:
    """分词：中文按二元组，英文按整词。"""

    print("\n[1] 分词与匹配")

    tokens = tokenize("注意力 集中")
    check("中文切出二元组", "注意" in tokens and "意力" in tokens, str(tokens))
    check("英文按整词切", "focus" in tokenize("Focus Time"), str(tokenize("Focus Time")))

    check(
        "词序不同仍能匹配",
        tokenize("注意力集中") & tokenize("集中注意力") != set(),
        "二元组无序",
    )


def test_short_term_fold() -> None:
    """短期记忆满了要折叠成摘要，不能直接丢。"""

    print("\n[2] 短期记忆折叠")

    memory = ShortTermMemory(capacity=10)

    for index in range(15):
        memory.add(f"笔记 {index}")

    check("超出容量后条数回落", len(memory) <= 10, str(len(memory)))
    check("发生了折叠", memory.fold_count == 1, str(memory.fold_count))

    first = memory.items[0]
    check("最前面是摘要条目", first.category == "summary", first.category)
    check("摘要保留了旧内容", "笔记 0" in first.content, first.content[:40])
    check("最新条目仍在", memory.items[-1].content == "笔记 14", memory.items[-1].content)


def test_long_term_slot() -> None:
    """长期记忆：同一槽位只保留最新说法，旧的仍可追溯。"""

    print("\n[3] 长期记忆槽位")

    long = LongTermMemory()
    first = long.add(
        _item(CATEGORY_PREFERENCE, "每天能学 2 小时")
    )
    second = long.add(
        _item(CATEGORY_PREFERENCE, "每天能学 3 小时")
    )

    check("新条目写入成功", second is not None)
    check("旧条目标记为被取代", first.superseded_by == second.id, first.superseded_by)
    check("旧条目仍在档案里", first in long.items)
    check("新条目没被取代", second.superseded_by == "", second.superseded_by)

    hits = long.recall("每天能学")
    check("召回只给最新说法", all(hit.item.id != first.id for hit in hits), str(len(hits)))


def test_reinforce() -> None:
    """重复提到同一件事要强化，不能塞成两条。"""

    print("\n[4] 强化机制")

    long = LongTermMemory()
    item = long.add(_item(CATEGORY_WEAKNESS, "指针这块总忘"))

    check("初始重要度 3", item.importance == 3, str(item.importance))

    long.reinforce("指针这块总忘")
    long.reinforce("指针这块总忘")

    check("提及次数累加", item.mentions == 3, str(item.mentions))
    check("反复提到后重要度提升", item.importance == 4, str(item.importance))
    check("档案里仍然只有一条", len([i for i in long.items if not i.superseded_by]) == 1)


def test_recall_ranking() -> None:
    """召回要按相关度排序。"""

    print("\n[5] 召回排序")

    long = LongTermMemory()
    long.add(_item(CATEGORY_NOTE, "每天练习英语听力"))
    long.add(_item(CATEGORY_NOTE, "整理房间和书桌"))

    hits = long.recall("英语听力")

    check("命中相关条目", len(hits) == 1, str(len(hits)))
    check("命中的是正确那条", hits[0].item.content == "每天练习英语听力", hits[0].item.content)
    check("召回会更新使用次数", hits[0].item.use_count == 1, str(hits[0].item.use_count))


def test_manager_promote_and_rollover() -> None:
    """晋升与跨天：顺序必须是先晋升再清空。"""

    print("\n[6] 晋升与跨天")

    manager = MemoryManager()

    manager.promote("学习目标：Python 数据分析", CATEGORY_GOAL, "用户设定目标", importance=5)
    check("目标进了长期档案", len(manager.long) == 1, str(len(manager.long)))

    # 短期里放一条被反复提到的高价值内容
    item = manager.short.add("数学基础薄弱，需要补", CATEGORY_WEAKNESS, importance=4)
    item.mentions = 3

    manager.short.add("今天天气不错", CATEGORY_NOTE, importance=1)

    moved = manager.rollover("2026-01-02")

    check("高价值内容被晋升", any("数学基础薄弱" in content for content in moved), str(moved))
    check("低价值内容没被晋升", not any("天气不错" in content for content in moved), str(moved))
    check("晋升后进了长期档案", len(manager.long) == 2, str(len(manager.long)))
    check("短期工作区已清空", len(manager.short) == 0, str(len(manager.short)))
    check("长期记忆跨天保留", len(manager.long) == 2, str(len(manager.long)))


def test_manager_remember_reinforces() -> None:
    """写短期记忆时，内容和长期档案一致的算强化。"""

    print("\n[7] 写入即强化")

    manager = MemoryManager()
    manager.promote("每天最多学 30 分钟", CATEGORY_PREFERENCE, "用户说过", importance=4)

    manager.remember("每天最多学 30 分钟", CATEGORY_PREFERENCE)

    long_item = manager.long.find_by_content("每天最多学 30 分钟")
    check("长期条目的提及次数上升", long_item.mentions == 2, str(long_item.mentions))
    check("长期档案没有多出一条", len(manager.long) == 1, str(len(manager.long)))
    check("短期记忆记下了这条", len(manager.short) == 1, str(len(manager.short)))


def test_memory_persistence() -> None:
    """记忆要能落盘并读回来。"""

    print("\n[8] 记忆持久化")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        manager = MemoryManager(MemoryStore(path))
        manager.promote("长期保留的结论", CATEGORY_PREFERENCE, "测试", importance=4)
        manager.remember("短期的工作笔记", CATEGORY_NOTE)
        manager.save()

        check("文件已生成", path.exists())

        restored = MemoryManager(MemoryStore(path))
        check("长期记忆读回来了", len(restored.long) == 1, str(len(restored.long)))
        check("短期记忆读回来了", len(restored.short) == 1, str(len(restored.short)))
        check(
            "内容一致",
            restored.long.items[0].content == "长期保留的结论",
            restored.long.items[0].content,
        )


def test_planner() -> None:
    """计划生成：阶段、限量、留白。"""

    print("\n[9] 计划生成")

    goal = Goal(
        id="g-1",
        title="学习 Python 数据分析",
        detail="",
        deadline=(date.today() + timedelta(days=30)).isoformat(),
    )

    check("识别出编程领域", detect_domain("学习 Python 数据分析") == "编程", detect_domain("学习 Python 数据分析"))
    check("识别出语言领域", detect_domain("练英语口语") == "语言", detect_domain("练英语口语"))
    check("识别出考试领域", detect_domain("准备考研数学") == "考试", detect_domain("准备考研数学"))

    milestones = build_milestones(goal, count=4)
    check("切出 4 个阶段", len(milestones) == 4, str(len(milestones)))
    check("阶段名带领域特征", "环境" in milestones[0] or "基础" in milestones[0], str(milestones))

    draft = generate_plan(goal, daily_minutes=60)
    days = {item.day for item in draft.items}
    check("计划铺到了多天", len(days) > 1, str(len(days)))
    check("每天不超过 3 条", all(len(items_for_day(draft.items, day)) <= 3 for day in days))
    check("任务都挂在目标下", all(item.goal_id == goal.id for item in draft.items))
    check("计划留了最后一天", max(days) < goal.deadline, str(max(days)))

    tight = generate_plan(goal, daily_minutes=30)
    check("时间少时排得更少", len(tight.items) < len(draft.items), f"{len(tight.items)} vs {len(draft.items)}")

    # 截止日期已过：仍要给得出计划，而不是交白卷
    expired = Goal(id="g-2", title="学游泳", deadline=(date.today() - timedelta(days=5)).isoformat())
    expired_draft = generate_plan(expired)
    check("截止日过期仍能排出计划", len(expired_draft.items) > 0, str(len(expired_draft.items)))
    check("过期时给出提示", "已过" in expired_draft.note, expired_draft.note)


def test_next_undone() -> None:
    """下一件事：先看今天没做完的。"""

    print("\n[10] 下一件事")

    items = [
        PlanItem(id="p-1", goal_id="g-1", title="今天第一讲", day="2026-01-01", done=True),
        PlanItem(id="p-2", goal_id="g-1", title="今天第二讲", day="2026-01-01", done=False),
        PlanItem(id="p-3", goal_id="g-1", title="明天第一讲", day="2026-01-02", done=False),
    ]

    check("优先今天的未完成项", next_undone(items, "2026-01-01").id == "p-2")
    check(
        "今天做完则找未来最早的",
        next_undone([items[0], items[2]], "2026-01-01").id == "p-3",
    )
    check("没有待办时返回 None", next_undone([items[0]], "2026-01-01") is None)

    done, total = progress_of(items)
    check("进度统计正确", (done, total) == (1, 3), f"{done}/{total}")


def test_review() -> None:
    """复盘：对账、找规律、沉淀。"""

    print("\n[11] 每日复盘")

    items = [
        PlanItem(id="p-1", goal_id="g-1", title="基础语法", day="2026-01-01", minutes=30, done=True),
        PlanItem(id="p-2", goal_id="g-1", title="数据清洗", day="2026-01-01", minutes=30, done=True),
        PlanItem(id="p-3", goal_id="g-1", title="可视化", day="2026-01-01", minutes=30, done=False),
    ]

    outcome = run_review(items, [], [], day="2026-01-01")

    check("计划条数对账正确", outcome.review.planned == 3, str(outcome.review.planned))
    check("完成条数对账正确", outcome.review.done == 2, str(outcome.review.done))
    check("投入分钟对账正确", outcome.review.minutes == 60, str(outcome.review.minutes))
    check("完成率正确", abs(outcome.review.completion - 2 / 3) < 0.01, str(outcome.review.completion))
    check("亮点收集正确", len(outcome.review.highlights) == 2, str(outcome.review.highlights))
    check("卡点收集正确", len(outcome.review.blockers) == 1, str(outcome.review.blockers))
    check("给出了下一步建议", outcome.next_step != "", outcome.next_step)

    # 连续两天低完成率 → 识别出计划量过大
    reviews = [
        Review(day="2026-01-01", planned=4, done=1, minutes=30),
        Review(day="2026-01-02", planned=4, done=1, minutes=30),
    ]
    insights = find_patterns(reviews, items, "2026-01-02")
    check("识别出连续低完成率", any("完成率低于" in text for text in insights), str(insights))

    # 沉淀进两级记忆：摘要进短期，强信号进长期
    manager = MemoryManager()
    applied, written = apply_review_findings(manager, outcome)
    check("复盘摘要进了短期工作区", len(written) > 0, str(written))
    check("复盘摘要在短期记忆里", len(manager.short) > 0, str(len(manager.short)))
    check("没有强信号时不硬塞长期记忆", len(manager.long) == 0, str(len(manager.long)))


def test_review_promotes_to_long_term() -> None:
    """复盘的强信号要真的进长期记忆，并能被召回。"""

    print("\n[12] 复盘强信号晋升长期")

    manager = MemoryManager()

    # 连续两天低完成率 → 强信号
    items = [
        PlanItem(id="p-1", goal_id="g-1", title="基础语法", day="2026-01-01", minutes=30, done=False),
        PlanItem(id="p-2", goal_id="g-1", title="数据清洗", day="2026-01-01", minutes=30, done=False),
        PlanItem(id="p-3", goal_id="g-1", title="基础语法", day="2026-01-02", minutes=30, done=False),
        PlanItem(id="p-4", goal_id="g-1", title="数据清洗", day="2026-01-02", minutes=30, done=False),
    ]
    reviews = [Review(day="2026-01-01", planned=2, done=0, minutes=0)]

    outcome = run_review(items, reviews, [], day="2026-01-02")
    applied, _ = apply_review_findings(manager, outcome)

    check("识别出强信号", len(applied) > 0, str(applied))
    check("强信号进了长期档案", len(manager.long) > 0, str(len(manager.long)))
    check("长期档案里是完成率结论",
          any("完成率低于" in item.content for item in manager.long.items),
          str([item.content for item in manager.long.items]))

    bundle = manager.recall("完成率")
    check("能召回复盘沉淀的结论", len(bundle.long) > 0, str(len(bundle.long)))


def test_intent_detection() -> None:
    """意图路由。"""

    print("\n[13] 意图识别")

    check("识别目标", detect_intent("我想学 Python") == INTENT_GOAL, detect_intent("我想学 Python"))
    check("识别计划", detect_intent("生成计划") == INTENT_PLAN)
    check("识别复盘", detect_intent("今天复盘一下") == INTENT_REVIEW)
    check("识别完成", detect_intent("这项做完了") == INTENT_DONE)
    check("识别查询", detect_intent("我的进度怎么样") == INTENT_QUERY)
    check("闲聊兜底", detect_intent("你好啊") == INTENT_CHAT)


def test_end_to_end() -> None:
    """端到端：记目标 → 生成计划 → 完成 → 复盘 → 查询。"""

    print("\n[14] 端到端")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = __import__("study_planner.storage", fromlist=["Storage"]).Storage(root)
        planner = StudyPlanner(memory=storage.build_memory())

        turn = planner.handle("我想学 Python 数据分析")
        check("目标已记录", len(planner.goals) == 1, str(len(planner.goals)))
        check("目标进了长期记忆", len(planner.memory.long) == 1, str(len(planner.memory.long)))
        check("回答里含目标", "Python" in turn.reply, turn.reply)

        turn = planner.handle("生成计划")
        check("计划已生成", len(planner.items) > 0, str(len(planner.items)))
        check("回答里提到阶段", "阶段" in turn.reply, turn.reply[:80])

        turn = planner.handle("今天完成了一项")
        check("任务被标记完成", planner.items[0].done, "第一条未完成")
        check("回答里有进度", "进度" in turn.reply, turn.reply)

        turn = planner.handle("复盘")
        check("复盘有记录", len(planner.reviews) == 1, str(len(planner.reviews)))
        check("复盘回答含结论", "复盘" in turn.reply, turn.reply[:80])

        turn = planner.handle("我的进度")
        check("查询返回了进度", "进度" in turn.reply, turn.reply[:80])
        check("查询带上了记忆", len(turn.memories_used) > 0, str(turn.memories_used))

        # 落盘后重载，数据还在
        storage.save_all(planner.memory, planner.goals, planner.items, planner.reviews)
        fresh_storage = __import__("study_planner.storage", fromlist=["Storage"]).Storage(root)
        check("目标已落盘", len(fresh_storage.load_goals()) == 1)
        check("计划已落盘", len(fresh_storage.load_items()) == len(planner.items))
        check("复盘已落盘", len(fresh_storage.load_reviews()) == 1)


def test_short_term_mentions() -> None:
    """同一件事在短期里反复出现要累加提及次数。"""

    print("\n[2b] 短期记忆提及计数")

    memory = ShortTermMemory(capacity=10)
    first = memory.add("注意力总是飘走", CATEGORY_WEAKNESS)
    second = memory.add("注意力总是飘走", CATEGORY_WEAKNESS)

    check("重复写入不新增条目", len(memory) == 1, str(len(memory)))
    check("提及次数累加", first.mentions == 2, str(first.mentions))
    check("返回的是同一条", first.id == second.id, f"{first.id} vs {second.id}")


def test_long_term_budget_affects_plan() -> None:
    """长期记忆里的时间预算要真的改变新计划。"""

    print("\n[15] 长期记忆影响排期")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = __import__("study_planner.storage", fromlist=["Storage"]).Storage(root)
        planner = StudyPlanner(memory=storage.build_memory())

        planner.handle("我想学 Python 数据分析")
        planner.memory.promote("每天最多学 30 分钟", CATEGORY_PREFERENCE, "用户说过", importance=5)

        planner.handle("生成计划")
        minutes = {item.minutes for item in planner.items}
        check("排期按记忆里的 30 分钟走", minutes == {30}, str(minutes))


def test_budget_capture() -> None:
    """说出时间预算要被记住，并影响后续排期。"""

    print("\n[16] 时间预算捕获")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = __import__("study_planner.storage", fromlist=["Storage"]).Storage(root)
        planner = StudyPlanner(memory=storage.build_memory())

        planner.handle("我每天只有 30 分钟")
        check(
            "预算进了长期记忆",
            any("30 分钟" in item.content for item in planner.memory.long.items),
            str([item.content for item in planner.memory.long.items]),
        )

        planner.handle("我想学 Python 数据分析")
        planner.handle("生成计划")

        minutes = {item.minutes for item in planner.items}
        check("排期按 30 分钟走", minutes == {30}, str(minutes))

        # 预算变了 → 同一个槽位，旧说法被取代
        planner.handle("我每天只有 60 分钟")
        active = [
            item for item in planner.memory.long.items
            if item.category == CATEGORY_PREFERENCE and not item.superseded_by
        ]
        check("新预算占用同一槽位", len(active) == 1, str([i.content for i in active]))
        check("新预算生效", "60 分钟" in active[0].content, active[0].content)

        superseded = [
            item for item in planner.memory.long.items if item.superseded_by
        ]
        check("旧预算被标记取代但保留", len(superseded) == 1, str(len(superseded)))


def _item(category: str, content: str):
    """造一条长期记忆条目。"""

    from .models import MemoryItem, SCOPE_LONG

    return MemoryItem(
        id=f"m-{abs(hash(content)) % 100000}",
        scope=SCOPE_LONG,
        category=category,
        content=content,
    )


def main() -> int:
    """自检入口。"""

    print("=" * 56)
    print("study_planner 自检")
    print("=" * 56)

    test_tokenize()
    test_short_term_fold()
    test_short_term_mentions()
    test_long_term_slot()
    test_reinforce()
    test_recall_ranking()
    test_manager_promote_and_rollover()
    test_manager_remember_reinforces()
    test_memory_persistence()
    test_planner()
    test_next_undone()
    test_review()
    test_review_promotes_to_long_term()
    test_intent_detection()
    test_end_to_end()
    test_long_term_budget_affects_plan()
    test_budget_capture()

    print("\n" + "=" * 56)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
