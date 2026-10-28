"""自检脚本：验证消息结构与上下文管理的核心不变量。

不依赖 pytest，直接运行：
    python -m chat_cli.selftest
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .context import ContextWindow, naive_summary
from .messages import ROLE_ASSISTANT, ROLE_USER, Message
from .storage import SessionStore

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED

    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f"  → {detail}" if detail else ""))


def test_message_roundtrip() -> None:
    print("\n[1] 消息结构：序列化往返")

    message = Message(ROLE_USER, "你好，世界", meta={"source": "test"})
    restored = Message.from_dict(message.to_dict())

    check(
        "字段完整还原",
        restored.id == message.id
        and restored.role == message.role
        and restored.content == message.content
        and restored.tokens == message.tokens
        and restored.meta == message.meta,
    )
    check("只把 role/content 送进模型", set(message.to_api()) == {"role", "content"})
    check("非法角色被拒绝", _raises(lambda: Message("robot", "hi")))


def test_context_budget() -> None:
    print("\n[2] 上下文：预算约束")

    window = ContextWindow(
        system_prompt="你是一个助手。" * 10,
        budget_tokens=300,
        keep_recent_turns=1,
    )

    for index in range(8):
        window.add(Message(ROLE_USER, f"第 {index} 轮用户输入" * 6))
        window.add(Message(ROLE_ASSISTANT, f"第 {index} 轮助手回复" * 6))

    plan = window.build()

    check(
        f"装配后不超预算（{plan.used_tokens} <= {plan.budget}）",
        plan.used_tokens <= plan.budget,
    )
    check(
        "最新一条消息一定在上下文里",
        plan.api_messages[-1]["content"] == window.messages[-1].content,
    )
    check("系统提示在最前面", plan.api_messages[0]["role"] == "system")

    window2 = ContextWindow("短系统提示", budget_tokens=100, keep_recent_turns=1)
    window2.add(Message(ROLE_USER, "很长的问题" * 200))

    plan2 = window2.build()
    check(
        "超大单条消息也不会丢（可能超预算）",
        len(plan2.api_messages) >= 2,
    )


def test_context_compact() -> None:
    print("\n[3] 上下文：压缩")

    window = ContextWindow("系统提示", budget_tokens=4000, keep_recent_turns=1)

    for index in range(4):
        window.add(Message(ROLE_USER, f"用户 {index}"))
        window.add(Message(ROLE_ASSISTANT, f"助手 {index}"))

    count, summary = window.compact(
        lambda old, previous: naive_summary(old, previous)
    )

    check("压缩后只剩保留窗口内的消息", len(window.messages) == 2, str(len(window.messages)))
    check("压缩产生了摘要", summary is not None and bool(summary.content))
    check("摘要被标记为派生消息", summary is not None and summary.is_synthetic)
    check("记录了被折叠的消息 id", len(summary.meta.get("compacted_ids", [])) == count)

    plan = window.build()
    check(
        "摘要出现在上下文里",
        any("历史对话摘要" in item["content"] for item in plan.api_messages),
    )


def test_storage() -> None:
    print("\n[4] 存储：落盘与恢复")

    tmp_dir = Path(tempfile.mkdtemp(prefix="chat-cli-selftest-"))

    try:
        store = SessionStore(tmp_dir)
        session_id = "selftest-session"

        store.save_meta(_new_meta(session_id))

        window = ContextWindow("系统提示", keep_recent_turns=1)
        window.add(Message(ROLE_USER, "旧消息 A"))
        window.add(Message(ROLE_ASSISTANT, "旧消息 B"))
        window.add(Message(ROLE_USER, "新消息 C"))

        for message in window.messages:
            store.append_message(session_id, message)

        count, summary = window.compact(
            lambda old, previous: naive_summary(old, previous)
        )
        store.append_compaction(
            session_id, summary, summary.meta.get("compacted_ids", [])
        )

        messages = store.load_messages(session_id)
        check("全部消息都能读回", len(messages) == 3, str(len(messages)))
        check(
            "完整历史不受压缩影响",
            [message.content for message in messages]
            == ["旧消息 A", "旧消息 B", "新消息 C"],
        )

        recent, restored_summary = store.load_context_state(session_id)
        check("恢复出的摘要一致", restored_summary is not None)
        check(
            "被折叠的消息不在近期区",
            all(message.content not in {"旧消息 A", "旧消息 B"} for message in recent),
        )
        check("近期区保留了最新消息", recent[-1].content == "新消息 C")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _new_meta(session_id: str):
    from .storage import SessionMeta

    return SessionMeta(id=session_id, model="test-model", system_prompt="系统提示")


def _raises(func) -> bool:
    try:
        func()
    except Exception:
        return True

    return False


def main() -> int:
    print("=" * 56)
    print("chat_cli 自检")
    print("=" * 56)

    test_message_roundtrip()
    test_context_budget()
    test_context_compact()
    test_storage()

    print("\n" + "=" * 56)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
