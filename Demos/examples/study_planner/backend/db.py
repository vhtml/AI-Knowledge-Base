"""明确数据库版本和短事务边界，故障不得被解释为空数据。"""
from __future__ import annotations

import os
import fcntl
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1


@contextmanager
def runtime_lock(root: Path, exclusive: bool = False) -> Iterator[None]:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / "runtime.lock").open("a") as handle:
        os.chmod(root / "runtime.lock", 0o600)
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("档案正在使用，请先停止 Web/CLI 再恢复") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(path, timeout=3, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=3000")
        if db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal" and sqlite3.sqlite_version_info < (3, 51, 3):
            raise ValueError("当前 SQLite 运行库未达到 WAL 模式最低版本 3.51.3")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise ValueError(f"不支持的数据库版本：{version}")
        if version == 0:
            schema = Path(__file__).with_name("migrations").joinpath("001_initial.sql").read_text()
            try:
                db.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")
            except sqlite3.OperationalError:
                # 两个入口首次启动时，另一个连接可能已经完成初始化。
                if db.in_transaction:
                    db.rollback()
                if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    raise
        os.chmod(path, 0o600)
        return db
    except BaseException:
        db.close()
        raise


@contextmanager
def transaction(path: Path, write: bool = True) -> Iterator[sqlite3.Connection]:
    db = open_database(path)
    try:
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()
