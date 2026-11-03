"""先过滤有效性，再按词片相关度检索，访问次数不改变事实权重。"""
from __future__ import annotations

from .context import Clock
from .db import transaction
from .memory import tokenize
from .repositories import Repository


class Retrieval:
    def __init__(self, repo: Repository, clock: Clock | None = None) -> None:
        self.repo, self.clock = repo, clock or Clock()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = sorted(tokenize(query))[:100]
        if not terms:
            return []
        with transaction(self.repo.path, write=False) as db:
            day = self.clock.day(self.repo.profile(db)["timezone"]).isoformat()
            placeholders = ",".join("?" for _ in terms)
            rows = db.execute(f"""SELECT m.*,count(*) AS matches FROM memory_terms t JOIN memories m ON m.id=t.memory_id
                WHERE t.term IN ({placeholders}) AND m.status='active' AND (m.effective_to IS NULL OR m.effective_to>=?)
                GROUP BY m.id ORDER BY matches DESC,m.confirmed_at DESC,m.id LIMIT ?""", [*terms, day, max(1, min(limit, 20))]).fetchall()
            result, seen = [], set()
            for row in rows:
                if row["content"] in seen:
                    continue
                seen.add(row["content"])
                item = dict(row)
                item["score"] = row["matches"] / len(terms)
                item["sources"] = [dict(r) for r in db.execute("SELECT source_id,excerpt FROM memory_sources WHERE memory_id=?", (row["id"],))]
                result.append(item)
            return result
