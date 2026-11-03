"""后端自检：验证 SSE 事件流与状态接口的行为。

不依赖 pytest，直接运行：

    cd Demos/examples
    uv run --project .. python -m study_planner.backend.selftest

用 FastAPI 的 TestClient 发请求，不需要真的起服务。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

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


def build_client(root: Path) -> TestClient:
    """装配一个指向临时数据目录的测试客户端。

    每个测试用独立目录，互不干扰，也不动真实数据。
    """

    from .main import app, session

    session.rebind(root)

    return TestClient(app)


def parse_events(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 [(事件名, 数据)]。"""

    events: list[tuple[str, dict]] = []
    current: str | None = None
    data_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("event:"):
            current = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif not line.strip() and current is not None:
            try:
                events.append((current, json.loads("\n".join(data_lines))))
            except json.JSONDecodeError:
                pass
            current = None
            data_lines = []

    return events


def test_health() -> None:
    """健康检查。"""

    print("\n[1] 健康检查")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        res = client.get("/health")
        check("健康检查返回 200", res.status_code == 200, str(res.status_code))
        check("状态是 ok", res.json().get("status") == "ok", str(res.json()))


def test_state_snapshot() -> None:
    """状态快照接口。"""

    print("\n[2] 状态快照")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        res = client.get("/api/state")
        check("快照返回 200", res.status_code == 200, str(res.status_code))

        data = res.json()
        check("包含目标字段", "goals" in data, str(list(data)))
        check("包含待办字段", "items" in data, str(list(data)))
        check("包含记忆字段", "memory" in data, str(list(data)))
        check("记忆里有统计", "stats" in data["memory"], str(data["memory"]))
        check("初始为空", not data["goals"] and not data["items"], str(data["goals"]))


def test_chat_stream() -> None:
    """SSE 流式对话。"""

    print("\n[3] SSE 对话流")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        res = client.post("/api/chat", json={"message": "我想学 Python 数据分析"})
        check("返回 200", res.status_code == 200, str(res.status_code))
        check(
            "Content-Type 是 event-stream",
            "text/event-stream" in res.headers.get("content-type", ""),
            res.headers.get("content-type", ""),
        )

        events = parse_events(res.text)
        names = [name for name, _ in events]

        check("有 status 事件", "status" in names, str(names))
        check("有 step 事件", "step" in names, str(names))
        check("有 reply 事件", "reply" in names, str(names))
        check("有 state 事件", "state" in names, str(names))

        steps = [data for name, data in events if name == "step"]
        check("留痕有多条", len(steps) >= 3, str(len(steps)))
        check(
            "留痕带阶段名",
            all("phase" in step and "title" in step for step in steps),
            str(steps[:1]),
        )

        reply = next(data for name, data in events if name == "reply")
        check("回复里有目标标题", "Python" in reply["text"], reply["text"])

        state = next(data for name, data in events if name == "state")
        check("状态快照带回了目标", len(state["goals"]) == 1, str(len(state["goals"])))
        check("目标进了长期记忆", len(state["memory"]["long"]) == 1, str(len(state["memory"]["long"])))

        # 顺序必须是 status → step… → reply → state
        check(
            "事件顺序正确",
            names.index("reply") > names.index("step")
            and names.index("state") > names.index("reply"),
            str(names),
        )


def test_budget_flow() -> None:
    """预算 → 排期，走完整链路。"""

    print("\n[4] 预算影响排期")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        client.post("/api/chat", json={"message": "我想学 Python 数据分析"})
        res = client.post("/api/chat", json={"message": "我每天只有 30 分钟"})

        events = parse_events(res.text)
        steps = [data for name, data in events if name == "step"]

        check(
            "预算被记进长期记忆",
            any(
                step["phase"] == "长期记忆" and "30 分钟" in step["detail"]
                for step in steps
            ),
            str(steps),
        )

        res = client.post("/api/chat", json={"message": "生成计划"})
        events = parse_events(res.text)

        steps = [data for name, data in events if name == "step"]
        plan_steps = [step for step in steps if step["phase"] == "生成计划"]

        check("有排期留痕", bool(plan_steps), str(steps))
        check(
            "排期用了记忆里的预算",
            bool(plan_steps) and "30 分钟" in plan_steps[0]["detail"],
            plan_steps[0]["detail"] if plan_steps else "",
        )

        state = next(data for name, data in events if name == "state")
        check("待办已生成", len(state["items"]) > 0, str(len(state["items"])))
        check(
            "每条待办都是 30 分钟",
            all(item["minutes"] == 30 for item in state["items"]),
            str({item["minutes"] for item in state["items"]}),
        )


def test_toggle_task() -> None:
    """勾选任务。"""

    print("\n[5] 任务勾选")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        res = client.post(
            "/api/goals",
            json={"title": "Python 数据分析", "daily_minutes": 30},
        )
        state = res.json()
        check("创建目标成功", len(state["goals"]) == 1, str(len(state["goals"])))
        check("生成了待办", len(state["items"]) > 0, str(len(state["items"])))

        task_id = state["items"][0]["id"]

        res = client.post(f"/api/tasks/{task_id}/toggle", json={"done": True})
        check("勾选返回 200", res.status_code == 200, str(res.status_code))

        state = res.json()
        target = next(item for item in state["items"] if item["id"] == task_id)
        check("任务标记为完成", target["done"] is True, str(target))
        check("完成写进了短期记忆", len(state["memory"]["short"]) > 0, str(state["memory"]["short"]))

        # 再点一次应该取消
        res = client.post(f"/api/tasks/{task_id}/toggle", json={"done": False})
        state = res.json()
        target = next(item for item in state["items"] if item["id"] == task_id)
        check("可以取消完成", target["done"] is False, str(target))

        res = client.post("/api/tasks/不存在/toggle", json={"done": True})
        check("不存在的任务返回 404", res.status_code == 404, str(res.status_code))


def test_rollover() -> None:
    """跨天：短期清空，长期保留。"""

    print("\n[6] 跨天滚动")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        client.post("/api/chat", json={"message": "我想学 Python 数据分析"})
        client.post("/api/chat", json={"message": "我每天只有 30 分钟"})

        before = client.get("/api/state").json()
        check("跨天前短期有内容", len(before["memory"]["short"]) > 0, str(len(before["memory"]["short"])))
        check("跨天前长期有内容", len(before["memory"]["long"]) > 0, str(len(before["memory"]["long"])))

        res = client.post("/api/rollover")
        check("跨天返回 200", res.status_code == 200, str(res.status_code))

        after = res.json()
        check("短期工作区已清空", len(after["memory"]["short"]) == 0, str(len(after["memory"]["short"])))
        check(
            "长期档案保留并可能增加",
            len(after["memory"]["long"]) >= len(before["memory"]["long"]),
            f"{len(before['memory']['long'])} → {len(after['memory']['long'])}",
        )
        check("待办不受跨天影响", len(after["items"]) == len(before["items"]), "待办数量变了")


def test_reset() -> None:
    """重置。"""

    print("\n[7] 重置")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        client.post("/api/goals", json={"title": "Python 数据分析", "daily_minutes": 30})

        res = client.post("/api/reset")
        check("重置返回 200", res.status_code == 200, str(res.status_code))

        state = res.json()
        check("目标已清空", not state["goals"], str(state["goals"]))
        check("待办已清空", not state["items"], str(len(state["items"])))
        check("记忆已清空", not state["memory"]["long"], str(len(state["memory"]["long"])))


def test_validation() -> None:
    """请求参数校验。"""

    print("\n[8] 参数校验")

    with tempfile.TemporaryDirectory() as tmp:
        client = build_client(Path(tmp))

        res = client.post("/api/chat", json={"message": ""})
        check("空消息被拒", res.status_code == 422, str(res.status_code))

        res = client.post("/api/chat", json={"message": "x" * 5000})
        check("超长消息被拒", res.status_code == 422, str(res.status_code))

        res = client.post("/api/goals", json={"title": ""})
        check("空标题被拒", res.status_code == 422, str(res.status_code))

        res = client.post("/api/goals", json={"title": "学", "daily_minutes": 5})
        check("过短的每日时长被拒", res.status_code == 422, str(res.status_code))


def main() -> int:
    """自检入口。"""

    print("=" * 56)
    print("study_planner 后端自检")
    print("=" * 56)

    test_health()
    test_state_snapshot()
    test_chat_stream()
    test_budget_flow()
    test_toggle_task()
    test_rollover()
    test_reset()
    test_validation()

    print("\n" + "=" * 56)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
