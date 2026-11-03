"""固定离线样例给出词法召回基线，语义检索是否值得引入由结果决定。"""
from __future__ import annotations

import json
import tempfile
from .db import transaction
from .memory_policy import store_candidate
from .memory_types import MemoryCandidate
from .turn_service import TurnService

# 每行一个明确事实和三个查询；第三个查询有意包含同义表达。
GOLD = [
    ('喜欢图解学习Python编程', ['Python学习偏好', '编程图解', '脚本语言入门方法']),
    ('英语听力适合每天跟读', ['英语听力', '每天跟读', '外语口头模仿']),
    ('数学函数需要反复练习', ['数学函数', '反复练习', '映射关系怎么掌握']),
    ('阅读历史书籍时喜欢先看目录', ['历史书籍', '先看目录', '史学著作建立框架']),
    ('算法动态规划常常卡在状态转移', ['算法动态规划', '状态转移', '递推求最优解难点']),
    ('日语词汇适合使用卡片记忆', ['日语词汇', '卡片记忆', '单词闪卡复习']),
    ('数据分析练习喜欢真实案例', ['数据分析', '真实案例', '统计实战材料']),
    ('健身力量训练先掌握动作规范', ['健身力量', '动作规范', '抗阻运动入门']),
    ('钢琴练习优先慢速分手弹奏', ['钢琴练习', '慢速分手', '键盘乐器左右手分别训练']),
    ('写作练习先列提纲再改稿', ['写作练习', '提纲改稿', '文章组织方法']),
]
NEGATIVE = ['量子隧穿', '火山灰', '珊瑚礁', '极光预测', '航天推进', '炼钢工艺', '古琴定弦', '沙漠生态', '戏剧表演', '陶瓷釉料']


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        service = TurnService(tmp)
        ids = []
        with transaction(service.repo.path) as db:
            for index, (text, _) in enumerate(GOLD):
                ids.append(store_candidate(db, MemoryCandidate(text, status='active', source_id=f'fixture-{index}', excerpt=text)))
            for status in ('pending', 'expired', 'superseded'):
                store_candidate(db, MemoryCandidate('禁止召回的过期秘密', status=status, source_id=f'source-{status}', excerpt='固定测试证据'))
        hits, reciprocal = 0, 0.0
        for mid, (_, queries) in zip(ids, GOLD):
            for query in queries:
                found = [r['id'] for r in service.retrieval.search(query)]
                if mid in found:
                    hits += 1
                    reciprocal += 1 / (found.index(mid) + 1)
        correct_negatives = sum(not service.retrieval.search(q) for q in NEGATIVE)
        assert not service.retrieval.search('禁止召回的过期秘密')
        print(json.dumps({'samples': 40, 'positive_queries': 30, 'recall_at_5': round(hits/30, 4), 'mrr': round(reciprocal/30, 4),
                          'negative_correct': correct_negatives, 'negative_total': 10, 'invalid_filtered': True, 'embedding_enabled': False}, ensure_ascii=False))
        service.close()


if __name__ == '__main__':
    main()
