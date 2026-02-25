"""
Task 03: VisFinQA Oracle QA + Reverse Mapping Pipeline
========================================================

逆映射链路:
  Query → Answer String → PyMuPDF search_for → BBox(PDF坐标)
  → pdf_bbox_to_pixel → IoPA mask → Activated Patch Indices

生成 5 页 × 2 QA = 10 个 Oracle QA 对, 并执行完整逆映射验证。
输出: data/vision_rag/oracle_qa.json

依赖: PyMuPDF, torch, pandas
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import fitz  # PyMuPDF

# 延迟导入 spatial_mask
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vision_rag.data.spatial_mask import (
    compute_fit_resize_info,
    pdf_bbox_to_pixel,
    compute_iopa_mask,
)

import torch


# =====================================================================
# Oracle QA Pairs (Claude 4.6 Opus 神谕注入)
# =====================================================================
# 每张图片 2 个极度刁钻的金融问题, 答案从图片中肉眼可读。
# 关键: answer_text 必须与 PDF 原文中的字符串精确匹配。

ORACLE_QA = [
    # ---- Page 1: AAPL FY2022 Q1 10-Q p017 ----
    # 表格: Products and Services Performance (Three Months Ended Dec 25, 2021 vs Dec 26, 2020)
    {
        "page_id": "AAPL_FY2022_Q1_10-Q_p017",
        "pdf_path": "data/raw_pdfs/technology/AAPL/2022_Q1_10-Q.pdf",
        "page_num": 17,
        "query": "What was Apple's iPhone net sales for the three months ended December 25, 2021?",
        "answer_text": "71,628",
        "answer_context": "iPhone revenue in Q1 FY2022 (Three Months Ended December 25, 2021)",
        "difficulty": "cross-column lookup",
    },
    {
        "page_id": "AAPL_FY2022_Q1_10-Q_p017",
        "pdf_path": "data/raw_pdfs/technology/AAPL/2022_Q1_10-Q.pdf",
        "page_num": 17,
        "query": "What was the percentage change in Services net sales between December 2020 and December 2021?",
        "answer_text": "24",
        "answer_context": "Services Change column: 24 %",
        "difficulty": "cross-row cross-column",
    },

    # ---- Page 2: AAPL FY2022 Q3 10-Q p017 ----
    # 表格: Products and Services Performance (Three & Nine Months Ended Jun 25, 2022 vs Jun 26, 2021)
    {
        "page_id": "AAPL_FY2022_Q3_10-Q_p017",
        "pdf_path": "data/raw_pdfs/technology/AAPL/2022_Q3_10-Q.pdf",
        "page_num": 17,
        "query": "What was Apple's total net sales for the nine months ended June 25, 2022?",
        "answer_text": "304,182",
        "answer_context": "Nine Months Ended Total net sales, June 25, 2022",
        "difficulty": "disambiguate Three-Month vs Nine-Month columns",
    },
    {
        "page_id": "AAPL_FY2022_Q3_10-Q_p017",
        "pdf_path": "data/raw_pdfs/technology/AAPL/2022_Q3_10-Q.pdf",
        "page_num": 17,
        "query": "What was the percentage change in iPad net sales for the three months ended June 2022 vs June 2021?",
        "answer_text": "(2)",
        "answer_context": "iPad Three Months Change column: (2) %",
        "difficulty": "negative percentage in parentheses",
    },

    # ---- Page 3: MSFT FY2023 Q3 10-Q p040 ----
    # SUMMARY RESULTS OF OPERATIONS (Three & Nine Months Ended Mar 31, 2023 vs 2022)
    {
        "page_id": "MSFT_FY2023_Q3_10-Q_p040",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2023_Q3_10-Q.pdf",
        "page_num": 40,
        "query": "What was Microsoft's operating income for Q3 FY2023 (three months ended March 31, 2023)?",
        "answer_text": "22,352",
        "answer_context": "Operating income row, 2023 column",
        "difficulty": "exact row-column intersection",
    },
    {
        "page_id": "MSFT_FY2023_Q3_10-Q_p040",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2023_Q3_10-Q.pdf",
        "page_num": 40,
        "query": "What was Microsoft's diluted EPS (non-GAAP) for the nine months ended March 31, 2023?",
        "answer_text": "7.12",
        "answer_context": "Adjusted diluted EPS (non-GAAP), Nine Months 2023",
        "difficulty": "GAAP vs non-GAAP disambiguation + column selection",
    },

    # ---- Page 4: MSFT FY2022 Q2 10-Q p035 ----
    # SUMMARY RESULTS OF OPERATIONS (Three & Six Months Ended Dec 31, 2021 vs 2020)
    {
        "page_id": "MSFT_FY2022_Q2_10-Q_p035",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2022_Q2_10-Q.pdf",
        "page_num": 35,
        "query": "What was Microsoft's gross margin for the six months ended December 31, 2021?",
        "answer_text": "66,439",
        "answer_context": "Gross margin, Six Months 2021 column",
        "difficulty": "three-month vs six-month column disambiguation",
    },
    {
        "page_id": "MSFT_FY2022_Q2_10-Q_p035",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2022_Q2_10-Q.pdf",
        "page_num": 35,
        "query": "What was the percentage change in Microsoft's net income for the three months ended December 31, 2021 compared to the same period in 2020?",
        "answer_text": "21%",
        "answer_context": "Net income Percentage Change column = 21%",
        "difficulty": "percentage change column lookup",
    },

    # ---- Page 5: MSFT FY2023 annual 10-K p046 ----
    # SEGMENT RESULTS OF OPERATIONS (FY2023 vs FY2022)
    {
        "page_id": "MSFT_FY2023_annual_10-K_p046",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2023_annual_10-K.pdf",
        "page_num": 46,
        "query": "What was Microsoft Intelligent Cloud segment's revenue in fiscal year 2023?",
        "answer_text": "87,907",
        "answer_context": "Intelligent Cloud Revenue, 2023 column",
        "difficulty": "segment-level revenue identification",
    },
    {
        "page_id": "MSFT_FY2023_annual_10-K_p046",
        "pdf_path": "data/raw_pdfs/technology/MSFT/2023_annual_10-K.pdf",
        "page_num": 46,
        "query": "What was the percentage change in More Personal Computing operating income between FY2022 and FY2023?",
        "answer_text": "(20)",
        "answer_context": "More Personal Computing OI, Percentage Change = (20)%",
        "difficulty": "negative percentage in parentheses + segment decomposition",
    },
]


def reverse_map_answer(
    pdf_path: str,
    page_num: int,
    answer_text: str,
    img_size: int = 1024,
    patch_size: int = 14,
    has_cls: bool = True,
) -> dict:
    """
    逆映射核心: Answer String → PDF BBox → Pixel BBox → IoPA Mask → Patch Indices

    算法:
      1. 打开 PDF 指定页, 用 page.search_for(answer_text) 定位 BBox
      2. 通过 compute_fit_resize_info 将 BBox 转为 ViT 输入像素坐标
      3. 通过 compute_iopa_mask 计算该 BBox 的软掩码
      4. 提取 mask > 0 的 patch 索引

    Returns:
        dict with keys: bbox_pdf, bbox_pixel, iopa_mask_shape, activated_patches, num_activated
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_width = page.rect.width
    page_height = page.rect.height

    # Step 1: PyMuPDF 文本搜索
    rects = page.search_for(answer_text)

    if not rects:
        # 尝试去掉逗号搜索
        rects = page.search_for(answer_text.replace(",", ""))
    if not rects:
        # 尝试带括号搜索 (负百分比)
        rects = page.search_for(f"({answer_text})")

    doc.close()

    if not rects:
        return {
            "status": "NOT_FOUND",
            "answer_text": answer_text,
            "search_attempts": [answer_text, answer_text.replace(",", ""), f"({answer_text})"],
        }

    # 取第一个匹配 (通常表格中数字只出现一次)
    rect = rects[0]
    bbox_pdf = (rect.x0, rect.y0, rect.x1, rect.y1)

    # Step 2: PDF → Pixel 坐标
    info = compute_fit_resize_info(page_width, page_height, img_size)
    bbox_pixel = pdf_bbox_to_pixel(bbox_pdf, info)

    # Step 3: IoPA 软掩码
    iopa_mask = compute_iopa_mask(
        bbox_pdf, page_width, page_height,
        img_size, patch_size, has_cls,
    )

    # Step 4: 激活的 Patch 索引 (mask > 0)
    activated_indices = torch.where(iopa_mask > 0)[0].tolist()

    # 转换为 2D 坐标 (row, col) 用于可视化
    N = img_size // patch_size  # 73
    activated_2d = []
    for idx in activated_indices:
        if has_cls and idx == 0:
            activated_2d.append(("CLS", "CLS"))
            continue
        flat_idx = idx - (1 if has_cls else 0)
        row = flat_idx // N
        col = flat_idx % N
        activated_2d.append((int(row), int(col)))

    return {
        "status": "FOUND",
        "answer_text": answer_text,
        "num_search_results": len(rects),
        "page_dims": {"width": float(page_width), "height": float(page_height)},
        "bbox_pdf": [round(x, 2) for x in bbox_pdf],
        "bbox_pixel": [round(x, 2) for x in bbox_pixel],
        "fit_resize": {
            "scale": round(info.scale, 4),
            "pad_x": round(info.pad_x, 2),
            "pad_y": round(info.pad_y, 2),
        },
        "iopa_mask_shape": list(iopa_mask.shape),
        "num_activated_patches": len(activated_indices),
        "activated_patch_indices": activated_indices,
        "activated_patch_2d": activated_2d,
        "iopa_values_at_activated": [round(iopa_mask[i].item(), 4) for i in activated_indices],
    }


def main():
    print("=" * 70)
    print("Task 03: VisFinQA Oracle QA + Reverse Mapping Pipeline")
    print("=" * 70)

    results = []
    success = 0
    fail = 0

    for i, qa in enumerate(ORACLE_QA):
        print(f"\n--- [{i+1}/{len(ORACLE_QA)}] {qa['page_id']} ---")
        print(f"  Query:  {qa['query']}")
        print(f"  Answer: {qa['answer_text']}")

        result = reverse_map_answer(
            pdf_path=qa["pdf_path"],
            page_num=qa["page_num"],
            answer_text=qa["answer_text"],
        )

        qa_result = {**qa, "reverse_map": result}
        results.append(qa_result)

        if result["status"] == "FOUND":
            success += 1
            print(f"  [OK] BBox(PDF): {result['bbox_pdf']}")
            print(f"       BBox(Pixel): {result['bbox_pixel']}")
            print(f"       Fit Resize: scale={result['fit_resize']['scale']}, "
                  f"pad=({result['fit_resize']['pad_x']}, {result['fit_resize']['pad_y']})")
            print(f"       Activated Patches: {result['num_activated_patches']} / 5330")
            print(f"       Patch Indices: {result['activated_patch_indices'][:20]}"
                  f"{'...' if len(result['activated_patch_indices']) > 20 else ''}")
            print(f"       Patch 2D (row,col): {result['activated_patch_2d'][:10]}"
                  f"{'...' if len(result['activated_patch_2d']) > 10 else ''}")
            print(f"       IoPA Values: {result['iopa_values_at_activated'][:10]}"
                  f"{'...' if len(result['iopa_values_at_activated']) > 10 else ''}")
        else:
            fail += 1
            print(f"  [FAIL] Answer not found in PDF text layer!")
            print(f"         Attempts: {result.get('search_attempts', [])}")

    # 保存 Oracle QA JSON
    output_path = Path("data/vision_rag/oracle_qa.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"Oracle QA Reverse Mapping Complete")
    print(f"  Total QA pairs: {len(ORACLE_QA)}")
    print(f"  Mapped:    {success}")
    print(f"  Failed:    {fail}")
    print(f"  Hit rate:  {success / len(ORACLE_QA) * 100:.0f}%")
    print(f"  Output:    {output_path}")
    print(f"{'=' * 70}")

    # ================================================
    # 详细展示 1 个完整逆映射链路
    # ================================================
    print("\n")
    print("=" * 70)
    print("DETAILED REVERSE-MAPPING TRACE (Demo)")
    print("=" * 70)

    # 选取第一个成功的样本
    demo = None
    for r in results:
        if r["reverse_map"]["status"] == "FOUND":
            demo = r
            break

    if demo:
        rm = demo["reverse_map"]
        print(f"  Page:        {demo['page_id']}")
        print(f"  Query:       {demo['query']}")
        print(f"  Answer:      {demo['answer_text']}")
        print(f"  Difficulty:  {demo['difficulty']}")
        print()
        print(f"  [Step 1] PyMuPDF search_for('{demo['answer_text']}')")
        print(f"           Found {rm['num_search_results']} match(es)")
        print(f"           Page dims: {rm['page_dims']['width']:.1f} x {rm['page_dims']['height']:.1f} pt")
        print()
        print(f"  [Step 2] BBox (PDF coordinates):")
        print(f"           x0={rm['bbox_pdf'][0]}, y0={rm['bbox_pdf'][1]}, "
              f"x1={rm['bbox_pdf'][2]}, y1={rm['bbox_pdf'][3]}")
        print()
        print(f"  [Step 3] Fit Resize Transform:")
        print(f"           scale = {rm['fit_resize']['scale']}")
        print(f"           pad_x = {rm['fit_resize']['pad_x']}")
        print(f"           pad_y = {rm['fit_resize']['pad_y']}")
        print()
        print(f"  [Step 4] BBox (Pixel coordinates in 1024x1024 image):")
        print(f"           x0={rm['bbox_pixel'][0]:.1f}, y0={rm['bbox_pixel'][1]:.1f}, "
              f"x1={rm['bbox_pixel'][2]:.1f}, y1={rm['bbox_pixel'][3]:.1f}")
        print()
        print(f"  [Step 5] IoPA Mask: shape={rm['iopa_mask_shape']}")
        print(f"           Activated patches: {rm['num_activated_patches']} / {rm['iopa_mask_shape'][0]}")
        print(f"           Patch indices (1D): {rm['activated_patch_indices']}")
        print(f"           Patch coords (row,col):")
        for idx, coord in zip(rm["activated_patch_indices"], rm["activated_patch_2d"]):
            iopa_val = rm["iopa_values_at_activated"][rm["activated_patch_indices"].index(idx)]
            print(f"             patch[{idx}] = ({coord[0]}, {coord[1]})  IoPA = {iopa_val:.4f}")
        print()
        print(f"  [Step 6] Ground Truth Tensor Verification:")
        print(f"           torch.Tensor[{rm['iopa_mask_shape'][0]}]")
        print(f"           sum(mask > 0) = {rm['num_activated_patches']}")
        print(f"           max(IoPA) = {max(rm['iopa_values_at_activated']):.4f}")
        print(f"           VGS-IoU ready: YES")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
