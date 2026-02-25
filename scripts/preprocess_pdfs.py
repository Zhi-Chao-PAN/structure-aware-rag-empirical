"""
工业级离线 PDF 预处理管线 (Industrial Offline PDF Preprocessing Pipeline)

设计原则:
  1. 多进程并发: ProcessPoolExecutor 满载 CPU 全核心
  2. Parquet 聚合: 所有元数据归入单一 Parquet 文件, 零碎文件雪崩防御
  3. DPI 对齐: 渲染分辨率精确匹配 ColPali 输入 (1024px 长边), 消除训练时插值开销
  4. 1024×1024 Padding: 短边白色填充, Padding Mask 预计算, 与 spatial_mask.py 精确对齐
  5. 断点续传: 跳过已处理文件, 支持增量预处理

存储布局:
  data/vision_rag/
  ├── page_images/           ← JPEG 页面图片 (Fit Resize + Padding, 1024×1024)
  │   ├── AAPL_FY2023_Q1_10-Q_p001.jpg
  │   └── ...
  ├── page_metadata.parquet  ← 全局元数据索引 (一个文件, 所有页面)
  └── spatial_masks/         ← 预计算掩码 (按 ticker 分片的 .pt 文件)
      ├── AAPL.pt            ← {page_id: {"padding_mask": Tensor[L], "iopa_mask": Tensor[L]}}
      └── ...

用法:
  python scripts/preprocess_pdfs.py \\
    --pdf-dir data/raw_pdfs \\
    --filing-index data/vision_rag/filing_index.json \\
    --output-dir data/vision_rag \\
    --workers 0 \\
    --img-size 1024 \\
    --patch-size 14

依赖: pip install PyMuPDF pyarrow pandas Pillow
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# =====================================================================
# Worker 不可 import torch/fitz at top-level (multiprocessing fork 安全)
# =====================================================================


@dataclass
class PageRecord:
    """单页的完整元数据记录 (将写入 Parquet)"""
    page_id: str              # 全局唯一: {ticker}_FY{year}_{quarter}_{type}_p{page_num:03d}
    ticker: str
    industry: str
    fiscal_year: str
    quarter: str
    report_type: str
    pdf_path: str             # 原始 PDF 路径
    page_num: int             # 0-indexed 页码
    page_width: float         # PDF 坐标系宽度 (pt)
    page_height: float        # PDF 坐标系高度 (pt)
    image_path: str           # 渲染后的 JPEG 路径 (1024×1024)
    render_dpi: float         # 实际渲染 DPI
    padded_width: int         # Padding 后的图像宽度 (always 1024)
    padded_height: int        # Padding 后的图像高度 (always 1024)
    num_text_blocks: int      # 文本块数量
    num_tables: int           # 表格数量 (启发式)
    text_snippet: str         # 前 500 字符 (用于 BM25 索引)
    bboxes_json: str          # JSON: list of [x0,y0,x1,y1] (PDF 坐标系)


def compute_render_matrix(page_width: float, page_height: float, target_size: int = 1024):
    """
    计算 PyMuPDF 渲染矩阵, 使输出图片长边 = target_size。

    Fit Resize 策略:
      scale = target_size / max(page_width, page_height)

    渲染后图片尺寸:
      rendered_w = page_width * scale
      rendered_h = page_height * scale
      max(rendered_w, rendered_h) == target_size
      min(rendered_w, rendered_h) < target_size  ← 短边需要 Padding

    Returns:
        fitz.Matrix, 实际 DPI
    """
    import fitz

    # PDF 标准 DPI = 72, PyMuPDF 的 Matrix zoom factor 表示相对于 72dpi 的倍率
    scale = target_size / max(page_width, page_height)
    actual_dpi = 72.0 * scale
    return fitz.Matrix(scale, scale), actual_dpi


def pad_pixmap_to_square(pix, target_size: int = 1024, bg_color=(255, 255, 255)):
    """
    将 PyMuPDF Pixmap 嵌入 target_size × target_size 正方形 (居中, 白色背景)。

    关键对齐约束:
      pad_x, pad_y 的计算逻辑与 spatial_mask.py 中的 compute_fit_resize_info 完全一致:
        pad_x = (target_size - rendered_width) / 2
        pad_y = (target_size - rendered_height) / 2

      这保障了:
        1. 渲染图像中第 (r, c) 个 patch 的像素区域精确对应 compute_padding_mask 的有效/无效判定
        2. 训练时 PyTorch DataLoader 无需额外 resize/pad, 直接读取 JPEG 即可

    Returns:
        PIL.Image (RGB, target_size × target_size)
    """
    from PIL import Image

    # PyMuPDF Pixmap → PIL Image
    img_data = pix.tobytes("ppm")
    rendered_img = Image.open(io.BytesIO(img_data))

    rw, rh = rendered_img.size

    # 创建正方形白色画布
    canvas = Image.new("RGB", (target_size, target_size), bg_color)

    # 居中粘贴
    pad_x = (target_size - rw) // 2
    pad_y = (target_size - rh) // 2
    canvas.paste(rendered_img, (pad_x, pad_y))

    return canvas


def extract_text_and_bboxes(page) -> tuple[str, list[list[float]], int, int]:
    """
    从 PyMuPDF page 提取文本内容和 BBox 坐标。

    Returns:
        text_snippet: 前 500 字符
        bboxes: [[x0,y0,x1,y1], ...] (PDF 坐标系)
        num_blocks: 文本块数量
        num_tables: 表格区域数量 (启发式)
    """
    blocks = page.get_text("dict", sort=True)["blocks"]

    all_text = []
    bboxes = []
    num_tables = 0

    for block in blocks:
        if block["type"] == 0:  # 文本块
            bbox = block["bbox"]  # (x0, y0, x1, y1)
            bboxes.append(list(bbox))

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_text.append(span.get("text", ""))

        elif block["type"] == 1:  # 图片块
            bboxes.append(list(block["bbox"]))

    # 启发式表格检测: 大量短小的等间隔文本块
    if len(blocks) > 20:
        heights = [b["bbox"][3] - b["bbox"][1] for b in blocks if b["type"] == 0]
        if heights:
            avg_h = sum(heights) / len(heights)
            short_blocks = sum(1 for h in heights if h < avg_h * 0.5)
            if short_blocks > len(heights) * 0.4:
                num_tables = 1

    full_text = " ".join(all_text)
    text_snippet = full_text[:500]

    return text_snippet, bboxes, len(blocks), num_tables


def process_single_pdf(args_tuple) -> list[dict]:
    """
    Worker 函数: 处理单个 PDF 的所有页面。

    在 ProcessPoolExecutor 中运行, 每个进程独立:
      1. 打开 PDF
      2. 逐页渲染 → Fit Resize → Padding to 1024×1024 → JPEG
      3. 提取 BBox + 计算 Padding Mask + IoPA Mask
      4. 返回 PageRecord 列表

    Args:
        args_tuple: (pdf_path, metadata, output_dir, img_size, patch_size, has_cls, jpeg_quality)

    Returns:
        list[dict]: PageRecord 字典列表
    """
    # 延迟导入 (multiprocessing fork 安全)
    import fitz
    import torch

    # 延迟导入 spatial_mask
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from vision_rag.data.spatial_mask import compute_multi_bbox_mask, compute_padding_mask

    pdf_path, metadata, output_dir, img_size, patch_size, has_cls, jpeg_quality = args_tuple

    image_dir = Path(output_dir) / "page_images"
    mask_dir = Path(output_dir) / "spatial_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ticker = metadata["ticker"]
    industry = metadata["industry"]
    fiscal_year = metadata["fiscal_year"]
    quarter = metadata["quarter"]
    report_type = metadata["report_type"]

    records = []
    masks_for_ticker = {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [ERROR] 无法打开 {pdf_path}: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        # 页面 ID: {ticker}_FY{year}_{quarter}_{type}_p{page_num:03d}
        page_id = f"{ticker}_FY{fiscal_year}_{quarter}_{report_type}_p{page_num:03d}"

        # 检查是否已处理 (断点续传)
        image_path = image_dir / f"{page_id}.jpg"
        if image_path.exists():
            continue

        # ================================================
        # 1. Fit Resize 渲染 + 1024×1024 Padding
        # ================================================
        mat, actual_dpi = compute_render_matrix(page_width, page_height, img_size)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # 短边 Padding → 1024×1024 正方形 (白色背景, 居中)
        padded_img = pad_pixmap_to_square(pix, img_size)
        padded_img.save(str(image_path), "JPEG", quality=jpeg_quality)

        # ================================================
        # 2. 提取文本和 BBox
        # ================================================
        text_snippet, bboxes, num_blocks, num_tables = extract_text_and_bboxes(page)

        # ================================================
        # 3. 预计算 Padding Mask (有效内容=1, Padding=0)
        # ================================================
        padding_mask = compute_padding_mask(
            page_width, page_height,
            img_size, patch_size, has_cls,
        )

        # ================================================
        # 4. 预计算 IoPA 掩码 (BBox 覆盖率)
        # ================================================
        if bboxes:
            bbox_tuples = [tuple(b) for b in bboxes]
            iopa_mask = compute_multi_bbox_mask(
                bbox_tuples, page_width, page_height,
                img_size, patch_size, has_cls,
            )
        else:
            # 无 BBox → 均匀掩码
            seq_len = (img_size // patch_size) ** 2 + (1 if has_cls else 0)
            iopa_mask = torch.ones(seq_len, dtype=torch.float32) * 0.5

        # 存储: {page_id: {"padding_mask": Tensor, "iopa_mask": Tensor}}
        masks_for_ticker[page_id] = {
            "padding_mask": padding_mask,
            "iopa_mask": iopa_mask,
        }

        # ================================================
        # 5. 构建 PageRecord
        # ================================================
        record = PageRecord(
            page_id=page_id,
            ticker=ticker,
            industry=industry,
            fiscal_year=fiscal_year,
            quarter=quarter,
            report_type=report_type,
            pdf_path=str(pdf_path),
            page_num=page_num,
            page_width=page_width,
            page_height=page_height,
            image_path=str(image_path),
            render_dpi=actual_dpi,
            padded_width=img_size,
            padded_height=img_size,
            num_text_blocks=num_blocks,
            num_tables=num_tables,
            text_snippet=text_snippet,
            bboxes_json=json.dumps(bboxes),
        )
        records.append(asdict(record))

    doc.close()

    # ================================================
    # 6. 保存该 ticker 的 mask 分片 (追加模式)
    # ================================================
    if masks_for_ticker:
        mask_path = mask_dir / f"{ticker}.pt"
        import torch
        if mask_path.exists():
            existing = torch.load(mask_path, weights_only=False)
            existing.update(masks_for_ticker)
            torch.save(existing, mask_path)
        else:
            torch.save(masks_for_ticker, mask_path)

    return records


def collect_pdf_tasks(
    pdf_dir: Path,
    filing_index: Optional[Path] = None,
) -> list[tuple[str, dict]]:
    """
    扫描 PDF 目录, 构建 (pdf_path, metadata) 任务列表。

    优先使用 filing_index.json (download_sec_filings.py 生成的),
    回退到从目录结构解析: {industry}/{ticker}/{year}_{quarter}_{type}.pdf
    """
    tasks = []

    if filing_index and filing_index.exists():
        with open(filing_index, 'r', encoding='utf-8') as f:
            index = json.load(f)
        for entry in index:
            # file_path 字段: "technology\\AAPL\\2022_annual_10-K.pdf"
            rel_path = entry.get("file_path", "")
            abs_path = pdf_dir / rel_path

            if abs_path.exists():
                tasks.append((str(abs_path), {
                    "ticker": entry["ticker"],
                    "industry": entry["industry"],
                    "fiscal_year": entry["fiscal_year"],
                    "quarter": entry["quarter"],
                    "report_type": entry["report_type"],
                }))
            else:
                print(f"  [WARN] filing_index 中的文件不存在: {abs_path}")
        return tasks

    # Fallback: 从目录结构解析
    for industry_dir in pdf_dir.iterdir():
        if not industry_dir.is_dir():
            continue
        industry = industry_dir.name

        for ticker_dir in industry_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name

            for pdf_file in ticker_dir.glob("*.pdf"):
                # 文件名格式: {year}_{quarter}_{type}.pdf
                stem = pdf_file.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    fiscal_year = parts[0]
                    quarter = parts[1]
                    report_type = "_".join(parts[2:])
                else:
                    fiscal_year = "unknown"
                    quarter = "unknown"
                    report_type = "unknown"

                tasks.append((str(pdf_file), {
                    "ticker": ticker,
                    "industry": industry,
                    "fiscal_year": fiscal_year,
                    "quarter": quarter,
                    "report_type": report_type,
                }))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="工业级离线 PDF 预处理管线"
    )
    parser.add_argument(
        "--pdf-dir", type=str, default="data/raw_pdfs",
        help="原始 PDF 目录",
    )
    parser.add_argument(
        "--filing-index", type=str, default=None,
        help="filing_index.json 路径 (优先于目录扫描)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/vision_rag",
        help="输出目录",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="并发进程数 (0=CPU 核心数)",
    )
    parser.add_argument(
        "--img-size", type=int, default=1024,
        help="渲染目标尺寸 (长边像素, 短边 Padding 至正方形)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=14,
        help="ViT Patch 尺寸",
    )
    parser.add_argument(
        "--has-cls", action="store_true", default=True,
        help="是否有 [CLS] token",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG 压缩质量 (1-100)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filing Index: 优先用户指定, 其次 vision_rag 目录下, 最后 pdf_dir 下
    if args.filing_index:
        filing_index = Path(args.filing_index)
    elif (output_dir / "filing_index.json").exists():
        filing_index = output_dir / "filing_index.json"
    else:
        filing_index = None

    # 收集任务
    tasks = collect_pdf_tasks(pdf_dir, filing_index)
    if not tasks:
        print("[ERROR] 未找到 PDF 文件. 请先运行 download_sec_filings.py")
        return

    num_workers = args.workers if args.workers > 0 else os.cpu_count()
    num_workers = min(num_workers, len(tasks))

    print("=" * 70)
    print("工业级 PDF 预处理管线 — Vision Tensor Forge")
    print("=" * 70)
    print(f"  PDF 总数:      {len(tasks)}")
    print(f"  并发进程:      {num_workers}")
    print(f"  渲染尺寸:      {args.img_size}×{args.img_size}px (Fit Resize + Padding)")
    print(f"  Patch 尺寸:    {args.patch_size}px")
    print(f"  JPEG 质量:     {args.jpeg_quality}")
    print(f"  输出目录:      {output_dir}")
    print("=" * 70)

    # 构建 worker 参数
    worker_args = [
        (path, meta, str(output_dir), args.img_size, args.patch_size,
         args.has_cls, args.jpeg_quality)
        for path, meta in tasks
    ]

    # ---- 多进程并发执行 ----
    all_records = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_pdf, wa): wa[1]["ticker"]
            for wa in worker_args
        }

        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1

            try:
                records = future.result()
                all_records.extend(records)
                if records:
                    print(
                        f"  [{completed}/{len(tasks)}] "
                        f"{ticker}: {len(records)} 页 ✓"
                    )
                else:
                    print(
                        f"  [{completed}/{len(tasks)}] "
                        f"{ticker}: 跳过 (已处理或空文件)"
                    )
            except Exception as e:
                print(f"  [{completed}/{len(tasks)}] {ticker}: ERROR - {e}")

    elapsed = time.time() - t_start

    # ---- Parquet 聚合写入 ----
    if all_records:
        parquet_path = output_dir / "page_metadata.parquet"
        df = pd.DataFrame(all_records)

        # 如果已存在, 追加 (去重)
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["page_id"], keep="last")

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"\n[PARQUET] 元数据已写入: {parquet_path}")
        print(f"  总页面数: {len(df)}")
        print(f"  列: {list(df.columns)}")

        # 统计摘要
        print(f"\n--- 行业分布 ---")
        print(df.groupby("industry")["page_id"].count().to_string())
        print(f"\n--- Ticker 分布 ---")
        print(df.groupby("ticker")["page_id"].count().sort_values(ascending=False).to_string())
    else:
        print("\n[INFO] 没有新页面需要处理 (全部已存在)")

    # ---- 汇总 ----
    print(f"\n{'=' * 70}")
    print(f"预处理完成")
    print(f"  新增页面: {len(all_records)}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  吞吐: {len(all_records) / max(elapsed, 0.1):.1f} 页/秒")
    print(f"  并发: {num_workers} 进程")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
