# backend/rag.py
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL") or None,
    api_key=os.getenv("LLM_API_KEY"),
)

DATA = ROOT / "data" / "agent_notes.txt"
PAGE_HEAD = re.compile(r"^# 第(\d+)页\s+(.+)$", re.M)

# 已知答案：正确片段 = 对应页码 + 正文中的特征句
GOLD = [
    {"id": 1, "question": "Agent 由哪几部分组成？", "page": 1, "title": "智能体组成", "needle": "模型、工具、记忆和执行循环"},
    {"id": 2, "question": "调用外部工具前必须做哪些限制？", "page": 2, "title": "工具调用规范", "needle": "校验参数类型与取值范围"},
    {"id": 3, "question": "资料库里找不到依据时，RAG 应该怎么回答？", "page": 3, "title": "RAG 回答原则", "needle": "明确说不知道"},
    {"id": 4, "question": "向量检索用什么方式给文档块排序？", "page": 4, "title": "向量检索", "needle": "余弦相似度"},
    {"id": 5, "question": "切块之后每一块要附带什么元数据？", "page": 5, "title": "分块与元数据", "needle": "页码和标题"},
    {"id": 6, "question": "资料不足时为什么必须拒答？", "page": 6, "title": "幻觉与拒答", "needle": "容易产生幻觉"},
    {"id": 7, "question": "短期记忆和长期记忆分别存在哪里？", "page": 7, "title": "记忆分层", "needle": "当前对话上下文"},
    {"id": 8, "question": "智能体的执行循环包含哪些步骤？", "page": 8, "title": "执行循环", "needle": "观察、思考、行动、再观察"},
    {"id": 9, "question": "引用资料时除了编号还要写出什么，好让读者核对原文？", "page": 9, "title": "来源引用格式", "needle": "[来源1 p.3《RAG 回答原则》]"},
    {"id": 10, "question": "怎样判断检索有没有找到正确片段？", "page": 10, "title": "检索评测", "needle": "已知答案的问题"},
]


def split_pages(text: str) -> list[dict]:
    hits = list(PAGE_HEAD.finditer(text))
    pages = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        pages.append({
            "page": int(m.group(1)),
            "title": m.group(2).strip(),
            "text": text[m.end():end].strip(),
        })
    return pages


def chunks(page: dict, size=120) -> list[dict]:
    body = page["text"]
    parts = [body[i:i + size] for i in range(0, len(body), size)] or [""]
    return [
        {
            "source": DATA.name,
            "page": page["page"],
            "title": page["title"],
            "text": part,
        }
        for part in parts
    ]


def load_chunks() -> list[dict]:
    pages = split_pages(DATA.read_text("utf-8"))
    out = []
    for p in pages:
        out.extend(chunks(p))
    return out


def embed(texts):
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([x.embedding for x in r.data])


def search(question, top_k=3):
    parts = load_chunks()
    vectors = embed([p["text"] for p in parts])
    q = embed([question])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    ids = np.argsort(scores)[::-1][:top_k]
    return [{**parts[i], "score": float(scores[i])} for i in ids]


def answer(question):
    hits = search(question)
    context = "\n".join(
        f"[来源{i + 1} p.{h['page']}《{h['title']}》] {h['text']}"
        for i, h in enumerate(hits)
    )
    r = client.responses.create(
        model=os.getenv("LLM_MODEL") or "gpt-5.4-mini",
        instructions="只能根据资料回答；不足就说不知道；引用[来源N]并带页码和标题。",
        input=f"资料：\n{context}\n\n问题：{question}",
    )
    return r.output_text


def evaluate(top_k=3):
    rows = []
    for item in GOLD:
        hits = search(item["question"], top_k=top_k)
        ok = any(h["page"] == item["page"] and item["needle"] in h["text"] for h in hits)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "expect_page": item["page"],
            "expect_title": item["title"],
            "needle": item["needle"],
            "hit": ok,
            "retrieved": [
                {
                    "page": h["page"],
                    "title": h["title"],
                    "score": round(h["score"], 4),
                    "text": h["text"],
                }
                for h in hits
            ],
        })

    hit_n = sum(r["hit"] for r in rows)
    payload = {"hit": hit_n, "total": len(rows), "top_k": top_k, "rows": rows}
    log = ROOT / "data" / "eval_log.json"
    log.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    print(f"命中 {hit_n}/{len(rows)}  （top_k={top_k}）")
    for r in rows:
        mark = "HIT" if r["hit"] else "MISS"
        got = ", ".join(f"p.{x['page']}《{x['title']}》" for x in r["retrieved"])
        print(f"{mark} Q{r['id']} {r['question']}")
        print(f"     期望 p.{r['expect_page']}《{r['expect_title']}》 | 检索: {got}")
    print(f"记录已写入 {log}")
    return payload


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        print(answer(input("问题：")))
