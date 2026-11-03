"""用一致快照备份，停写恢复时先重放删除清单，避免内容复活。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .db import SCHEMA_VERSION, open_database, runtime_lock
from .deletion import DeletionService
from .repositories import Repository, dumps, now_iso


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def backup_repository(repo: Repository, target: Path) -> dict[str, Any]:
    target = target.expanduser().resolve()
    if target == repo.path.resolve():
        raise ValueError("备份不能覆盖运行数据库")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    deletion = DeletionService(repo)
    with runtime_lock(repo.root), deletion.locked_log() as handle:
        records = deletion.records(handle)
        fd, temporary = tempfile.mkstemp(prefix=".backup-", dir=target.parent)
        os.close(fd)
        staging = Path(temporary)
        try:
            source = open_database(repo.path)
            destination = sqlite3.connect(staging)
            try:
                source.backup(destination)
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("备份完整性检查失败")
            finally:
                destination.close()
                source.close()
            sync_file(staging)
            manifest = {"format": 1, "schema": SCHEMA_VERSION, "created_at": now_iso(), "sha256": file_hash(staging), "deletion_ids": [r["id"] for r in records]}
            os.replace(staging, target)
            os.chmod(target, 0o600)
            manifest_path = target.with_suffix(target.suffix + ".json")
            manifest_path.write_text(dumps(manifest))
            os.chmod(manifest_path, 0o600)
            sync_file(manifest_path)
            return manifest
        finally:
            staging.unlink(missing_ok=True)


def restore_repository(root: Path, backup: Path, dry_run: bool = True) -> dict[str, Any]:
    manifest = json.loads(backup.with_suffix(backup.suffix + ".json").read_text())
    if manifest.get("format") != 1 or manifest.get("schema") != SCHEMA_VERSION or file_hash(backup) != manifest.get("sha256"):
        raise ValueError("备份校验或版本不匹配")
    with runtime_lock(root, exclusive=True):
        # 恢复是修复损坏数据库的入口，不能要求先成功打开当前数据库。
        repo = Repository(root, initialize=False)
        deletion = DeletionService(repo)
        with deletion.locked_log() as handle:
            records = deletion.records(handle)
            ids = {r["id"] for r in records}
            if not set(manifest["deletion_ids"]) <= ids:
                raise ValueError("删除清单缺少备份已知的删除记录")
            if repo.path.exists():
                try:
                    current = sqlite3.connect(repo.path.resolve().as_uri() + '?mode=ro', uri=True)
                    try:
                        applied_now = {r[0] for r in current.execute('SELECT id FROM deletions')}
                    finally:
                        current.close()
                except sqlite3.DatabaseError:
                    # 当前库损坏时依赖完整且校验成功的独立清单，不能跳过坏清单。
                    applied_now = set()
                if not applied_now <= ids:
                    raise ValueError("删除清单少于当前库已知记录，拒绝恢复")
            fd, temporary = tempfile.mkstemp(prefix=".restore-", dir=root)
            os.close(fd)
            staging = Path(temporary)
            try:
                source = sqlite3.connect(backup.resolve().as_uri() + "?mode=ro", uri=True)
                dest = sqlite3.connect(staging, isolation_level=None)
                try:
                    source.backup(dest)
                    dest.row_factory = sqlite3.Row
                    if dest.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION or dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("备份内部检查失败")
                    applied = {r[0] for r in dest.execute("SELECT id FROM deletions")}
                    if not applied <= ids:
                        raise ValueError("恢复缺少删除记录")
                    dest.execute("BEGIN IMMEDIATE")
                    for record in records:
                        deletion.apply(dest, record)
                    if dest.execute("PRAGMA foreign_key_check").fetchone():
                        raise ValueError("恢复后外键检查失败")
                    dest.commit()
                finally:
                    dest.close()
                    source.close()
                if not dry_run:
                    sync_file(staging)
                    os.replace(staging, repo.path)
                    os.chmod(repo.path, 0o600)
                return {"dry_run": dry_run, "deletions_replayed": len(records), "backup_time": manifest["created_at"], "integrity": "ok"}
            finally:
                staging.unlink(missing_ok=True)


def daily_backup(repo: Repository, day: str) -> None:
    folder = repo.root / "backups"
    target = folder / f"{day}.db"
    if not target.exists() or not target.with_suffix(".db.json").exists():
        backup_repository(repo, target)
    for stale in sorted(folder.glob("????-??-??.db"))[:-7]:
        stale.unlink()
        stale.with_suffix(".db.json").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="学习档案一致快照备份与离线恢复")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".study-planner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backup", type=Path)
    group.add_argument("--restore", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = backup_repository(Repository(args.data_dir), args.backup) if args.backup else restore_repository(args.data_dir, args.restore, args.dry_run)
    print(dumps(result))


if __name__ == "__main__":
    main()
