"""
BM25 Hard Negative 挖掘 + 元数据互斥安全阀 (Build BM25 Index & Mine Hard Negatives)

核心职责:
  1. 从 page_metadata.parquet 读取每页文本
  2. 构建 BM25 索引
  3. 对每个页面挖掘 top-K 硬负样本
  4. 元数据互斥: 排除"真阳性" (同公司+同报表类型+相邻周期)

False Negative Collision 防御:
  BM25 按词频匹配, 同一家公司同一份报表的不同页面 (如连续两页 Income Statement)
  文本高度相似, BM25 会把它们当作"硬负样本"。但这些页面其实是正确答案的"近邻",
  如果被标记为负样本, 去噪 Loss 会惩罚模型的正确表征 → 训练坍塌。

  安全阀规则:
    排除: (同 ticker) AND (同 report_type) AND (|period_diff| ≤ 4 个季度)
    排除: (同 ticker) AND (同 content_type)
  保留: 跨公司、跨行业的真正"硬负样本" (看起来像但确实不是答案的页面)

用法:
  python scripts/build_bm25_index.py \
    --parquet data/vision_rag/metadata/page_metadata.parquet \
    --output data/vision_rag/bm25_index/hard_negatives.json \
    --top-k 50

依赖: pip install rank-bm25 pandas pyarrow
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def period_distance(year1: int, q1: str, year2: int, q2: str) -> int:
    """计算两个财务周期之间的季度距离"""
    def to_ordinal(year: int, q: str) -> int:
        qnum = int(q[1]) if isinstance(q, str) and len(q) >= 2 and q[0] == 'Q' else 1
        return int(year) * 4 + qnum
    return abs(to_ordinal(year1, q1) - to_ordinal(year2, q2))


def is_false_negative(meta_q: dict, meta_d: dict) -> bool:
    """
    元数据互斥判断: 两个页面是否属于同一"真阳性"近邻。

    排除规则 (任一满足即排除):
      1. 同 ticker + 同 content_type (同份报表的不同页)
      2. 同 ticker + 同 report_type + 时间差 ≤ 4 个季度
    """
    same_ticker = meta_q.get("ticker") == meta_d.get("ticker")
    if not same_ticker:
        return False

    # 规则 2: 同公司同报表类型 + 相邻周期
    same_report = meta_q.get("report_type") == meta_d.get("report_type")
    if same_report:
        dist = period_distance(
            meta_q.get("fiscal_year", 0), meta_q.get("quarter", "Q1"),
            meta_d.get("fiscal_year", 0), meta_d.get("quarter", "Q1"),
        )
        if dist <= 4:
            return True

    return False


def build_bm25_and_mine(
    parquet_path: str,
    output_path: str,
    top_k: int = 50,
    min_text_len: int = 20,
):
    """
    主流程: 构建 BM25 → 挖掘净化后的硬负样本

    Args:
        parquet_path: page_metadata.parquet
        output_path: 输出 JSON 路径
        top_k: 每页 BM25 候选数
        min_text_len: 文本最低字符数 (过短的跳过)
    """
    import pandas as pd
    from rank_bm25 import BM25Okapi

    logger.info(f"加载 Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    required_cols = ["page_id", "text_snippet", "ticker", "report_type"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Parquet 缺少必需列: {col}")

    # 文本预处理
    df["text_clean"] = df["text_snippet"].fillna("").astype(str)
    valid_mask = df["text_clean"].str.len() >= min_text_len
    df_valid = df[valid_mask].reset_index(drop=True)

    logger.info(
        f"有效页面: {len(df_valid)}/{len(df)} "
        f"(跳过 {len(df) - len(df_valid)} 个文本过短的页面)"
    )

    # BM25 索引构建
    logger.info("构建 BM25 索引...")
    tokenized_corpus = [text.lower().split() for text in df_valid["text_clean"]]
    bm25 = BM25Okapi(tokenized_corpus)

    # 元数据提取
    page_ids = df_valid["page_id"].tolist()
    metadata = df_valid[[
        "ticker", "industry", "fiscal_year", "quarter",
        "report_type",
    ]].to_dict(orient="records")

    # Hard Negative 挖掘 + 净化
    hard_negatives: dict[str, list[str]] = {}
    total_candidates = 0
    total_excluded = 0

    for i in range(len(df_valid)):
        query_text = tokenized_corpus[i]
        meta_q = metadata[i]

        # BM25 top-K
        scores = bm25.get_scores(query_text)
        # 排除自身
        scores[i] = -float("inf")

        # 取 top-K 候选
        top_indices = scores.argsort()[-top_k * 2:][::-1]  # 取 2x 候选防止净化后不够

        clean_negs = []
        for j in top_indices:
            j = int(j)
            if scores[j] <= 0:
                break
            total_candidates += 1
            meta_d = metadata[j]

            if is_false_negative(meta_q, meta_d):
                total_excluded += 1
                continue

            clean_negs.append(page_ids[j])
            if len(clean_negs) >= top_k:
                break

        hard_negatives[page_ids[i]] = clean_negs

        if (i + 1) % 1000 == 0:
            logger.info(
                f"  进度: {i+1}/{len(df_valid)} | "
                f"排除 FN: {total_excluded}/{total_candidates} "
                f"({100*total_excluded/max(total_candidates,1):.1f}%)"
            )

    # 统计
    avg_negs = sum(len(v) for v in hard_negatives.values()) / max(len(hard_negatives), 1)
    logger.info(
        f"\n挖掘完成:\n"
        f"  总页面: {len(hard_negatives)}\n"
        f"  平均硬负样本数: {avg_negs:.1f}\n"
        f"  总候选: {total_candidates}\n"
        f"  FN 排除: {total_excluded} ({100*total_excluded/max(total_candidates,1):.1f}%)"
    )

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hard_negatives, f, indent=2, ensure_ascii=False)
    logger.info(f"[SAVED] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BM25 Hard Negative 挖掘 (含元数据互斥安全阀)"
    )
    parser.add_argument(
        "--parquet", type=str, required=True,
        help="page_metadata.parquet 路径",
    )
    parser.add_argument(
        "--output", type=str, default="data/vision_rag/bm25_index/hard_negatives.json",
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="每页 BM25 候选数",
    )
    parser.add_argument(
        "--min-text-len", type=int, default=20,
        help="文本最低字符数",
    )
    args = parser.parse_args()

    build_bm25_and_mine(
        parquet_path=args.parquet,
        output_path=args.output,
        top_k=args.top_k,
        min_text_len=args.min_text_len,
    )


if __name__ == "__main__":
    main()
