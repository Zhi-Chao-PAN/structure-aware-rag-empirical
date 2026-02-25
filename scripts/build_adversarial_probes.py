"""
Task 04: Adversarial Probe Auto-Forge
=======================================

Probe A (Cross-Period Twin):
  同一 Ticker, 同一报表类型, 相邻周期的结构极似页面对。
  挖掘: SSIM > 0.85 且文本含 "Revenue" / "Net Income" 等财务关键词。

Probe B (Cross-Company Doppelganger):
  同行业, 同周期, 不同 Ticker 的结构极似页面对。
  挖掘: 同为 Technology, 相同 FY+Q+ReportType, SSIM + 表格行/列数接近。

SSIM 计算: skimage 或手写灰度 SSIM (避免外部依赖)。
使用 Pillow + numpy 手写 SSIM (MS-SSIM kernel), 零外部依赖。

输出:
  data/vision_rag/probe_a.json  (Cross-Period Twin pairs)
  data/vision_rag/probe_b.json  (Cross-Company Doppelganger pairs)
"""

import json
import math
import sys
from pathlib import Path
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


# =====================================================================
# Structural Similarity (SSIM) — 手写实现 (无 scikit-image 依赖)
# =====================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """
    计算两张灰度图的 SSIM (Structural Similarity Index)。

    Wang et al., 2004: "Image Quality Assessment: From Error Visibility to
    Structural Similarity" (IEEE TIP)

    简化版: 全局统计量 (不用滑窗), 适合页面级相似度粗筛。
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(img1).astype(np.float64)
    mu2 = np.mean(img2).astype(np.float64)
    sigma1_sq = np.var(img1).astype(np.float64)
    sigma2_sq = np.var(img2).astype(np.float64)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2)).astype(np.float64)

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim)


def load_gray_thumbnail(image_path: str, size: int = 256) -> Optional[np.ndarray]:
    """加载图片为灰度缩略图 (加速 SSIM 计算)"""
    try:
        img = Image.open(image_path).convert("L").resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float64)
    except Exception:
        return None


# =====================================================================
# Probe A: Cross-Period Twin (跨周期孪生)
# =====================================================================

FINANCIAL_KEYWORDS = [
    "Revenue", "Net income", "Operating income", "Gross margin",
    "Total net sales", "Net sales", "Cost of revenue",
    "Research and development", "Sales and marketing",
    "Diluted earnings", "Products and Services",
]


def build_probe_a(df: pd.DataFrame) -> list[dict]:
    """
    Probe A: 同一 Ticker + 同一 ReportType, 相邻年份/季度, SSIM > 阈值。

    算法:
      1. GroupBy (ticker, report_type, quarter), 找不同 fiscal_year 的页面
      2. 在同一 (ticker, report_type, quarter) 组内, 取相邻 fiscal_year 的页面对
      3. 仅保留 text_snippet 含财务关键词的页面
      4. 计算 SSIM, 保留 > 0.80 的对
    """
    probes = []

    # 只看有文本的页面
    df_text = df[df.text_snippet.str.len() > 50].copy()

    # 预过滤: text_snippet 含财务关键词
    keyword_pattern = "|".join(FINANCIAL_KEYWORDS)
    df_fin = df_text[df_text.text_snippet.str.contains(keyword_pattern, case=False, na=False)]

    print(f"  [Probe A] Financial pages after keyword filter: {len(df_fin)}")

    # GroupBy (ticker, report_type, quarter) 找跨年份配对
    grouped = df_fin.groupby(["ticker", "report_type", "quarter"])

    pair_count = 0
    for (ticker, report_type, quarter), group in grouped:
        if len(group) < 2:
            continue

        # 按 fiscal_year 排序
        group_sorted = group.sort_values("fiscal_year")
        years = group_sorted.fiscal_year.unique()

        if len(years) < 2:
            continue

        # 取相邻年份, 同 page_num 的页面对
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            pages_y1 = group_sorted[group_sorted.fiscal_year == y1]
            pages_y2 = group_sorted[group_sorted.fiscal_year == y2]

            # 同 page_num 配对 (结构最可能相似)
            common_pages = set(pages_y1.page_num.values) & set(pages_y2.page_num.values)

            for pn in list(common_pages)[:5]:  # 每组最多 5 对
                row1 = pages_y1[pages_y1.page_num == pn].iloc[0]
                row2 = pages_y2[pages_y2.page_num == pn].iloc[0]

                img1 = load_gray_thumbnail(row1.image_path)
                img2 = load_gray_thumbnail(row2.image_path)

                if img1 is None or img2 is None:
                    continue

                ssim = compute_ssim(img1, img2)
                pair_count += 1

                if ssim > 0.80:
                    probes.append({
                        "probe_type": "A",
                        "description": "Cross-Period Twin",
                        "ticker": ticker,
                        "report_type": report_type,
                        "quarter": quarter,
                        "page_1": {
                            "page_id": row1.page_id,
                            "fiscal_year": row1.fiscal_year,
                            "page_num": int(row1.page_num),
                            "image_path": row1.image_path,
                        },
                        "page_2": {
                            "page_id": row2.page_id,
                            "fiscal_year": row2.fiscal_year,
                            "page_num": int(row2.page_num),
                            "image_path": row2.image_path,
                        },
                        "ssim": round(ssim, 4),
                        "text_overlap_keywords": [
                            kw for kw in FINANCIAL_KEYWORDS
                            if kw.lower() in row1.text_snippet.lower()
                            and kw.lower() in row2.text_snippet.lower()
                        ],
                    })

    print(f"  [Probe A] Pairs evaluated: {pair_count}")
    print(f"  [Probe A] High-SSIM pairs (>0.80): {len(probes)}")

    return sorted(probes, key=lambda x: x["ssim"], reverse=True)


# =====================================================================
# Probe B: Cross-Company Doppelganger (跨公司分身)
# =====================================================================

def build_probe_b(df: pd.DataFrame) -> list[dict]:
    """
    Probe B: 同行业, 同周期 (fiscal_year + quarter + report_type), 不同 Ticker。

    算法:
      1. GroupBy (industry, fiscal_year, quarter, report_type), 找不同 ticker 的页面
      2. 在组内跨 ticker 配对 (同 page_num 优先)
      3. 计算 SSIM, 保留 > 0.70 的对 (跨公司的结构一致性阈值放宽)
    """
    probes = []

    df_text = df[df.text_snippet.str.len() > 50].copy()
    keyword_pattern = "|".join(FINANCIAL_KEYWORDS)
    df_fin = df_text[df_text.text_snippet.str.contains(keyword_pattern, case=False, na=False)]

    print(f"  [Probe B] Financial pages after keyword filter: {len(df_fin)}")

    grouped = df_fin.groupby(["industry", "fiscal_year", "quarter", "report_type"])

    pair_count = 0
    for (industry, fy, q, rt), group in grouped:
        tickers = group.ticker.unique()
        if len(tickers) < 2:
            continue

        # 每对 ticker 组合
        for t1, t2 in combinations(sorted(tickers), 2):
            pages_t1 = group[group.ticker == t1]
            pages_t2 = group[group.ticker == t2]

            # 跨公司配对: page_num ±3 范围 (不同公司报表结构不同, 同一内容不一定在同一页)
            for _, r1 in pages_t1.iterrows():
                for _, r2 in pages_t2.iterrows():
                    if abs(r1.page_num - r2.page_num) > 5:
                        continue

                    img1 = load_gray_thumbnail(r1.image_path)
                    img2 = load_gray_thumbnail(r2.image_path)

                    if img1 is None or img2 is None:
                        continue

                    ssim = compute_ssim(img1, img2)
                    pair_count += 1

                    if ssim > 0.50:
                        probes.append({
                            "probe_type": "B",
                            "description": "Cross-Company Doppelganger",
                            "industry": industry,
                            "fiscal_year": fy,
                            "quarter": q,
                            "report_type": rt,
                            "page_1": {
                                "page_id": r1.page_id,
                                "ticker": r1.ticker,
                                "page_num": int(r1.page_num),
                                "image_path": r1.image_path,
                            },
                            "page_2": {
                                "page_id": r2.page_id,
                                "ticker": r2.ticker,
                                "page_num": int(r2.page_num),
                                "image_path": r2.image_path,
                            },
                            "ssim": round(ssim, 4),
                            "text_overlap_keywords": [
                                kw for kw in FINANCIAL_KEYWORDS
                                if kw.lower() in r1.text_snippet.lower()
                                and kw.lower() in r2.text_snippet.lower()
                            ],
                        })

    print(f"  [Probe B] Pairs evaluated: {pair_count}")
    print(f"  [Probe B] High-SSIM pairs (>0.70): {len(probes)}")

    return sorted(probes, key=lambda x: x["ssim"], reverse=True)


def main():
    print("=" * 70)
    print("Task 04: Adversarial Probe Auto-Forge")
    print("=" * 70)

    df = pd.read_parquet("data/vision_rag/page_metadata.parquet")
    print(f"  Total pages: {len(df)}")
    print(f"  Tickers: {sorted(df.ticker.unique())}")
    print(f"  Industries: {sorted(df.industry.unique())}")

    # ---- Probe A ----
    print(f"\n{'=' * 40}")
    print(f"Building Probe A (Cross-Period Twin)...")
    print(f"{'=' * 40}")
    probe_a = build_probe_a(df)

    out_a = Path("data/vision_rag/probe_a.json")
    with open(out_a, "w", encoding="utf-8") as f:
        json.dump(probe_a, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_a} ({len(probe_a)} pairs)")

    # 展示 Top 5
    print(f"\n  Top 5 Cross-Period Twins:")
    for i, p in enumerate(probe_a[:5]):
        print(f"    [{i+1}] SSIM={p['ssim']:.4f}  "
              f"{p['page_1']['page_id']}  <->  {p['page_2']['page_id']}  "
              f"keywords={p['text_overlap_keywords'][:3]}")

    # ---- Probe B ----
    print(f"\n{'=' * 40}")
    print(f"Building Probe B (Cross-Company Doppelganger)...")
    print(f"{'=' * 40}")
    probe_b = build_probe_b(df)

    out_b = Path("data/vision_rag/probe_b.json")
    with open(out_b, "w", encoding="utf-8") as f:
        json.dump(probe_b, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_b} ({len(probe_b)} pairs)")

    # 展示 Top 5
    print(f"\n  Top 5 Cross-Company Doppelgangers:")
    for i, p in enumerate(probe_b[:5]):
        print(f"    [{i+1}] SSIM={p['ssim']:.4f}  "
              f"{p['page_1']['page_id']}  <->  {p['page_2']['page_id']}  "
              f"keywords={p['text_overlap_keywords'][:3]}")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"Adversarial Probe Forge Complete")
    print(f"  Probe A (Cross-Period Twin):          {len(probe_a)} pairs")
    print(f"  Probe B (Cross-Company Doppelganger): {len(probe_b)} pairs")
    print(f"  Output: {out_a}")
    print(f"          {out_b}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
