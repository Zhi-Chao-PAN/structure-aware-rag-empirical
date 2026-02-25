"""
Task 05: Text-BM25 Baseline — The Shame Column
=================================================

传统 Text-RAG 基线构建:
  1. PyMuPDF 强力文本剥离 (page.get_text("text"))
  2. BM25 倒排索引构建 (rank_bm25)
  3. Oracle QA 碰撞预演 — 展示 BM25 在复杂表格检索上的惨烈失败

学术意义:
  Vision-RAG 论文 Table 3 的对照组。如果 BM25 也能找对页面,
  我们的创新就没有价值。BM25 必须在这些嵌套表格上翻车。

输出:
  data/vision_rag/bm25_index.pkl      — 序列化 BM25 索引
  data/vision_rag/bm25_baseline.json   — 碰撞预演结果

依赖: pip install rank_bm25
"""

import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pandas as pd


# =====================================================================
# Step 1: 文本强力剥离 (Text Extraction)
# =====================================================================

def extract_all_page_texts(df: pd.DataFrame) -> list[dict]:
    """
    遍历所有页面, 用 PyMuPDF 强行提取纯文本。

    警告: 对于精心设计的金融表格, get_text("text") 会完全丧失
         列对齐信息, 将 "71,628  65,597  9 %" 变成 "71,628 65,597 9 %",
         数字之间的空间语义关系被彻底摧毁。

    这正是我们需要的 — 展示文本扁平化后信息的毁灭性损失。
    """
    results = []
    pdf_cache = {}

    for _, row in df.iterrows():
        pdf_path = row.pdf_path
        page_num = row.page_num

        # PDF 文件缓存 (避免重复打开)
        if pdf_path not in pdf_cache:
            try:
                pdf_cache[pdf_path] = fitz.open(pdf_path)
            except Exception as e:
                print(f"  [WARN] Cannot open {pdf_path}: {e}")
                continue

        doc = pdf_cache[pdf_path]
        if page_num >= len(doc):
            continue

        page = doc[page_num]
        raw_text = page.get_text("text")

        # 基本清理: 合并多余空白, 保留换行
        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        results.append({
            "page_id": row.page_id,
            "ticker": row.ticker,
            "fiscal_year": row.fiscal_year,
            "quarter": row.quarter,
            "report_type": row.report_type,
            "page_num": int(row.page_num),
            "text": cleaned.strip(),
            "text_length": len(cleaned.strip()),
        })

    # 关闭所有 PDF
    for doc in pdf_cache.values():
        doc.close()

    return results


# =====================================================================
# Step 2: BM25 倒排索引构建
# =====================================================================

def tokenize(text: str) -> list[str]:
    """
    极简 BM25 tokenizer。

    特意不做高级 NLP (stemming, lemmatization) —
    让 BM25 在最公平的条件下都输给 Vision-RAG。
    """
    # 保留数字和逗号 (金额), 小写化
    tokens = re.findall(r"[a-zA-Z0-9,.$%]+", text.lower())
    # 去掉单字符 token
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(page_texts: list[dict]):
    """
    使用 rank_bm25 构建 BM25 Okapi 索引。

    BM25 参数: k1=1.5, b=0.75 (经典设定)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[ERROR] rank_bm25 not installed. Run: pip install rank_bm25")
        sys.exit(1)

    corpus = [tokenize(item["text"]) for item in page_texts]
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

    return bm25, corpus


def bm25_query(
    bm25,
    page_texts: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    执行 BM25 查询, 返回 Top-K 页面。
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Top-K
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        item = page_texts[idx]
        results.append({
            "rank": rank + 1,
            "page_id": item["page_id"],
            "ticker": item["ticker"],
            "fiscal_year": item["fiscal_year"],
            "quarter": item["quarter"],
            "bm25_score": round(float(scores[idx]), 4),
            "text_preview": item["text"][:150].replace("\n", " "),
        })

    return results


# =====================================================================
# Step 3: 碰撞预演 (Collision Demo)
# =====================================================================

def run_collision_demo(bm25, page_texts: list[dict]) -> list[dict]:
    """
    用 Oracle QA 中的 query 测试 BM25 检索。
    展示 BM25 在嵌套表格问题上的结构性失败。
    """
    # 从 oracle_qa.json 加载
    oracle_path = Path("data/vision_rag/oracle_qa.json")
    if not oracle_path.exists():
        print("  [WARN] oracle_qa.json not found. Run build_oracle_qa.py first.")
        return []

    with open(oracle_path, "r", encoding="utf-8") as f:
        oracle_qa = json.load(f)

    collision_results = []

    for qa in oracle_qa:
        query = qa["query"]
        correct_page = qa["page_id"]
        answer = qa["answer_text"]

        results = bm25_query(bm25, page_texts, query, top_k=5)

        # 检查正确页面是否在 Top-5
        top5_pages = [r["page_id"] for r in results]
        hit_at_1 = top5_pages[0] == correct_page if top5_pages else False
        hit_at_5 = correct_page in top5_pages

        collision_results.append({
            "query": query,
            "answer": answer,
            "correct_page": correct_page,
            "bm25_hit_at_1": hit_at_1,
            "bm25_hit_at_5": hit_at_5,
            "bm25_top5": results,
        })

    return collision_results


def main():
    print("=" * 70)
    print("Task 05: Text-BM25 Baseline — The Shame Column")
    print("=" * 70)

    # Step 1: 文本剥离
    print("\n[Step 1] Extracting text from all PDF pages...")
    df = pd.read_parquet("data/vision_rag/page_metadata.parquet")
    t0 = time.time()
    page_texts = extract_all_page_texts(df)
    t_extract = time.time() - t0
    print(f"  Extracted: {len(page_texts)} pages in {t_extract:.1f}s")

    # 展示表格文本被破坏的程度
    print(f"\n  --- TEXT DESTRUCTION DEMO ---")
    # 找一个有表格的页面
    for pt in page_texts:
        if pt["page_id"] == "AAPL_FY2022_Q1_10-Q_p017":
            lines = pt["text"].split("\n")
            print(f"  Page: {pt['page_id']}")
            print(f"  Text length: {pt['text_length']} chars")
            print(f"  First 10 lines of extracted text:")
            for line in lines[:10]:
                print(f"    | {line.strip()}")
            print(f"  ...")
            print(f"  [NOTICE] Table alignment completely destroyed!")
            break

    # Step 2: BM25 索引
    print(f"\n[Step 2] Building BM25 Okapi index...")
    t0 = time.time()
    bm25, corpus = build_bm25_index(page_texts)
    t_index = time.time() - t0
    print(f"  Index built: {len(corpus)} documents in {t_index:.1f}s")

    # 序列化
    index_path = Path("data/vision_rag/bm25_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "page_ids": [pt["page_id"] for pt in page_texts],
            "page_texts_meta": [
                {k: v for k, v in pt.items() if k != "text"}
                for pt in page_texts
            ],
        }, f)
    index_size = index_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {index_path} ({index_size:.1f} MB)")

    # Step 3: 碰撞预演
    print(f"\n[Step 3] Running collision demo with Oracle QA...")
    collisions = run_collision_demo(bm25, page_texts)

    if collisions:
        hit1 = sum(1 for c in collisions if c["bm25_hit_at_1"])
        hit5 = sum(1 for c in collisions if c["bm25_hit_at_5"])
        total = len(collisions)

        print(f"\n  {'=' * 50}")
        print(f"  BM25 SHAME COLUMN — Oracle QA Collision Results")
        print(f"  {'=' * 50}")
        print(f"  Total queries:   {total}")
        print(f"  Hit@1:           {hit1}/{total} ({hit1/total*100:.0f}%)")
        print(f"  Hit@5:           {hit5}/{total} ({hit5/total*100:.0f}%)")
        print(f"  Miss@5:          {total - hit5}/{total} ({(total - hit5)/total*100:.0f}%)")

        # 详细展示每个 query
        for c in collisions:
            status = "HIT@1" if c["bm25_hit_at_1"] else ("HIT@5" if c["bm25_hit_at_5"] else "MISS")
            marker = "V" if "HIT" in status else "X"
            print(f"\n  [{marker}] {status}")
            print(f"      Query:   {c['query'][:80]}...")
            print(f"      Answer:  {c['answer']}")
            print(f"      Correct: {c['correct_page']}")
            print(f"      BM25 Top-5:")
            for r in c["bm25_top5"]:
                flag = " <<< CORRECT" if r["page_id"] == c["correct_page"] else ""
                print(f"        #{r['rank']} [{r['bm25_score']:.2f}] {r['page_id']}{flag}")

        # 保存结果
        collision_path = Path("data/vision_rag/bm25_baseline.json")
        with open(collision_path, "w", encoding="utf-8") as f:
            json.dump({
                "hit_at_1": hit1,
                "hit_at_5": hit5,
                "total": total,
                "recall_at_1": round(hit1 / total, 4),
                "recall_at_5": round(hit5 / total, 4),
                "collisions": collisions,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {collision_path}")

    print(f"\n{'=' * 70}")
    print(f"Text-BM25 Baseline Complete")
    print(f"  Text extraction:  {t_extract:.1f}s")
    print(f"  Index build:      {t_index:.1f}s")
    print(f"  Index size:       {index_size:.1f} MB")
    print(f"  BM25 Recall@1:    {hit1 if collisions else 'N/A'}/{total if collisions else 'N/A'}")
    print(f"  BM25 Recall@5:    {hit5 if collisions else 'N/A'}/{total if collisions else 'N/A'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
