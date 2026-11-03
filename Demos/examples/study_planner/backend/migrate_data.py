"""只读校验旧 JSON，再将可对账的数据原子导入新数据库。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from .db import transaction
from .models import Goal, PlanItem, Review
from .repositories import Repository, dumps, now_iso


def inspect_source(source: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not source.is_dir():
        raise ValueError("源目录不存在")
    data: dict[str, Any] = {}
    digest = hashlib.sha256()
    for name in ("goals", "plans", "reviews", "memory"):
        path = source / f"{name}.json"
        raw = path.read_bytes() if path.exists() else b""
        digest.update(name.encode() + b"\0" + raw)
        value = json.loads(raw) if raw else ({} if name == "memory" else [])
        expected = dict if name == "memory" else list
        if not isinstance(value, expected):
            raise ValueError(f"{name}.json 格式不正确")
        data[name] = value
    for name in ("goals", "plans"):
        unique: dict[str, Any] = {}
        for item in data[name]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError(f"{name} 存在无效 ID")
            if item["id"] in unique and unique[item["id"]] != item:
                raise ValueError(f"{name} 存在同 ID 不同内容：{item['id']}")
            unique[item["id"]] = item
        data[name] = list(unique.values())
    goal_ids = {g["id"] for g in data["goals"]}
    for item in data["plans"]:
        if item.get("goal_id") not in goal_ids:
            raise ValueError(f"待办 {item['id']} 引用了不存在的目标")
        date.fromisoformat(item["day"])
    reviews = {}
    review_count = len(data["reviews"])
    for item in data["reviews"]:
        date.fromisoformat(item["day"])
        reviews[item["day"]] = item
    data["reviews"] = list(reviews.values())
    memories: dict[str, Any] = {}
    for scope in ("short", "long"):
        entries = data["memory"].get(scope, [])
        if not isinstance(entries, list):
            raise ValueError("记忆列表格式不正确")
        for item in entries:
            if not isinstance(item, dict) or not item.get("id") or not isinstance(item.get("content"), str):
                raise ValueError("记忆条目格式不正确")
            if item["id"] in memories and memories[item["id"]] != item:
                raise ValueError("记忆存在同 ID 不同内容")
            memories[item["id"]] = item
    data["memories"] = list(memories.values())
    report = {"goals": len(data["goals"]), "items": len(data["plans"]), "reviews": len(reviews),
              "duplicate_reviews": review_count - len(reviews), "memories": len(memories), "raw_reviews": review_count}
    return data, digest.hexdigest(), report


def migrate(source: Path, target: Path, dry_run: bool = True) -> dict[str, Any]:
    data, batch, report = inspect_source(source)
    if dry_run:
        return {**report, "batch": batch, "dry_run": True}
    repo = Repository(target)
    # 保留逐字节快照，恢复时不依赖已做容错解析的数据。
    backup = target / f"legacy-{batch[:12]}"
    backup.mkdir(mode=0o700, exist_ok=True)
    for path in source.glob("*.json"):
        if path.name not in {"goals.json", "plans.json", "reviews.json", "memory.json"}:
            continue
        copy = backup / path.name
        if not copy.exists():
            shutil.copyfile(path, copy)
            os.chmod(copy, 0o600)
    with transaction(repo.path) as db:
        if db.execute("SELECT 1 FROM imports WHERE id=?", (batch,)).fetchone():
            return {**report, "batch": batch, "already_imported": True}
        for table in ("goals", "plan_items", "reviews", "memories", "turns", "imports"):
            if db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise ValueError("目标库已有数据，拒绝覆盖")
        repo.save_domain(db, [Goal.from_dict(g) for g in data["goals"]],
                         [PlanItem.from_dict(i) for i in data["plans"]], [Review.from_dict(r) for r in data["reviews"]])
        budget_pending = False
        for item in data["memories"]:
            status = "superseded" if item.get("superseded_by") else "pending"
            db.execute("INSERT INTO memories(id,kind,category,content,status,importance,created_at,superseded_by,source_type) VALUES (?,?,?,?,?,?,?,?,?)",
                       (item["id"], "legacy", item.get("category", "note"), item["content"], status,
                        max(1, min(5, int(item.get("importance", 3)))), item.get("created_at") or now_iso(), item.get("superseded_by", ""), "legacy"))
            db.execute("INSERT INTO memory_sources VALUES (?,?,?,?)", (item["id"], f"import:{batch}:{item['id']}", item["content"], "legacy-v1"))
            if status == "pending" and any(word in item["content"] for word in ("分钟", "小时")):
                budget_pending = True
        db.execute("UPDATE profile SET budget_status=?,state_revision=state_revision+1", ("needs_confirmation" if budget_pending else "unset",))
        report["pending_budget"] = budget_pending
        db.execute("INSERT INTO imports VALUES (?,?)", (batch, dumps({**report, "original_reviews": data["reviews"]})))
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("外键检查失败")
    return {**report, "batch": batch, "dry_run": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并迁移旧学习档案")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(dumps(migrate(args.source, args.target, args.dry_run)))


if __name__ == "__main__":
    main()
