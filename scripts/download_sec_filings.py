"""
SEC EDGAR 高保真数据采集管线 (SEC Filing Download + HTML-to-PDF Rendering)

致命架构陷阱防御:
  SEC EDGAR 提供的 10-K / 10-Q 主文档是 HTML, 不是 PDF。
  Vision-RAG (Document-as-Image) 需要视觉完整的 PDF 页面渲染图。
  如果不转换, PyMuPDF 将读取到 HTML 标签文本而非排版后的金融表格。

管线流程:
  1. sec-edgar-downloader 拉取原始 HTML filing
  2. pdfkit (wkhtmltopdf 后端) 将 HTML 渲染为高保真 PDF
  3. 结构化存储:  data/raw_pdfs/{Industry}/{Ticker}/{FY}_{Q}_{Type}.pdf
  4. 增量构建 filing_index.json (Layer 1 Denoising 锚点)

SEC 合规:
  - User-Agent: "FirstName LastName email@domain.com" (强制格式)
  - 速率限制: 每次请求后 sleep 0.2s (< 10 req/s)

系统依赖:
  Windows: choco install wkhtmltopdf
    或下载: https://wkhtmltopdf.org/downloads.html
    安装后确保 wkhtmltopdf.exe 在 PATH 中
  Linux:   sudo apt-get install wkhtmltopdf
  macOS:   brew install wkhtmltopdf

Python 依赖: pip install sec-edgar-downloader pdfkit

用法:
  # 最小化验证 (仅 AAPL/MSFT, 2022-2023)
  python scripts/download_sec_filings.py \\
    --name "Zhichao Pan" --email "zpan0001@student.edu" \\
    --tickers AAPL MSFT --years 2022 2023

  # 全量下载 (4 行业 × 24 tickers)
  python scripts/download_sec_filings.py \\
    --name "Zhichao Pan" --email "zpan0001@student.edu" \\
    --years 2021 2022 2023
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# 行业-公司映射 (对齐 ConcentricAdversarialSampler 拓扑)
# =====================================================================

INDUSTRY_TICKERS: dict[str, list[str]] = {
    "technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "financial":  ["JPM", "GS", "BAC", "MS", "C", "WFC"],
    "healthcare": ["PFE", "JNJ", "UNH", "MRK", "ABT", "LLY"],
    "energy":     ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"],
}

TICKER_TO_INDUSTRY = {}
for _ind, _tickers in INDUSTRY_TICKERS.items():
    for _t in _tickers:
        TICKER_TO_INDUSTRY[_t] = _ind


# =====================================================================
# 非标准财年映射 (Fiscal Year End Month Hardcode)
#
# 致命架构陷阱防御:
#   SEC 的日历日期 ≠ 公司财务季度。
#   AAPL 财年 9 月结束 → 12 月是 FY Q1 (不是日历 Q4)
#   MSFT 财年 6 月结束 → 9 月是 FY Q1 (不是日历 Q3)
#
#   如果用日历推断, 同一 Batch 中 AAPL 和 MSFT 的 "Q1" 指代的
#   实际时间窗口差了 3-6 个月 → Concentric Adversarial Sampler 的
#   "同周期跨公司对抗样本 (Ring 2)" 在时间维度错开 → 脏梯度。
#
#   物理底线: 元数据里的 (ticker, fiscal_year, quarter) 必须反映
#   该公司真实的财务周期, 不是日历月份。
# =====================================================================

FISCAL_YEAR_END: dict[str, int] = {
    # Technology
    "AAPL": 9,    # Apple: 9 月 (FY starts Oct)
    "MSFT": 6,    # Microsoft: 6 月 (FY starts Jul)
    "NVDA": 1,    # NVIDIA: 1 月 (FY starts Feb)
    # Financial (多数 12 月, 不需要特别处理)
    "GS": 12,
    "JPM": 12,
    # Healthcare
    "JNJ": 12,
    "PFE": 12,
    # Energy
    "XOM": 12,
    "CVX": 12,
    # 标准财年默认 12 月 → 不在此 dict 中的 ticker 自动用 12
}


def infer_fiscal_period(
    ticker: str,
    period_of_report: str,
    filing_type: str,
) -> dict[str, str]:
    """
    基于 CONFORMED PERIOD OF REPORT + 公司财年结束月 推断精确财务周期。

    输入:
      ticker: 公司代号 (查 FISCAL_YEAR_END 表)
      period_of_report: "YYYY-MM-DD" (SEC 报告覆盖期截止日)
      filing_type: "10-K" 或 "10-Q"

    算法:
      1. 查表获取 fiscal_year_end_month (FYE)
      2. 10-K: 该日期所属的财年, quarter="annual"
      3. 10-Q: 计算从 FY 起始月到 period_of_report 的月数 → 季度

    验证:
      AAPL (FYE=9), period=2021-12-25, 10-Q:
        month=12 > FYE=9 → FY=2022
        fy_start=10; months_in=12-10+1=3 → Q1 ✓

      MSFT (FYE=6), period=2022-09-30, 10-Q:
        month=9 > FYE=6 → FY=2023
        fy_start=7; months_in=9-7+1=3 → Q1 ✓
    """
    try:
        dt = datetime.strptime(period_of_report, "%Y-%m-%d")
    except ValueError:
        return {
            "fiscal_year": period_of_report[:4] if len(period_of_report) >= 4 else "unknown",
            "quarter": "unknown",
            "report_type": filing_type,
        }

    month, year = dt.month, dt.year
    fye = FISCAL_YEAR_END.get(ticker, 12)

    # 判断所属财年: period month > FYE → 下一个 FY
    if month > fye:
        fiscal_year = year + 1
    else:
        fiscal_year = year

    if filing_type == "10-K":
        return {
            "fiscal_year": str(fiscal_year),
            "quarter": "annual",
            "report_type": "10-K",
        }

    # 10-Q: 计算季度
    fy_start_month = (fye % 12) + 1  # FY 起始月 (FYE 的下一个月)

    if month >= fy_start_month:
        months_into_fy = month - fy_start_month + 1
    else:
        months_into_fy = (12 - fy_start_month + 1) + month

    quarter = min((months_into_fy - 1) // 3 + 1, 4)

    return {
        "fiscal_year": str(fiscal_year),
        "quarter": f"Q{quarter}",
        "report_type": "10-Q",
    }


# =====================================================================
# HTML-to-PDF 高保真渲染引擎
# =====================================================================

def get_wkhtmltopdf_path() -> Optional[str]:
    """
    自动检测 wkhtmltopdf 可执行文件路径。
    Windows 上常见安装位置:
      C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe
    """
    # 1. PATH 中直接可用
    if shutil.which("wkhtmltopdf"):
        return shutil.which("wkhtmltopdf")

    # 2. Windows 默认安装路径
    win_paths = [
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
    ]
    for p in win_paths:
        if os.path.isfile(p):
            return p

    return None


def convert_html_to_pdf(
    html_path: Path,
    pdf_path: Path,
    wkhtmltopdf_path: Optional[str] = None,
    timeout: int = 120,
) -> bool:
    """
    使用 pdfkit (wkhtmltopdf 后端) 将 HTML 转换为高保真 PDF。

    保真要求:
      - 金融表格的列对齐、边框线必须完整
      - SEC 的内联 CSS 样式必须被渲染
      - 大型 filing (200+ 页) 必须在 timeout 内完成

    Args:
        html_path: 输入 HTML 文件路径
        pdf_path: 输出 PDF 文件路径
        wkhtmltopdf_path: wkhtmltopdf 可执行文件路径
        timeout: 转换超时 (秒)

    Returns:
        True if conversion succeeded, False otherwise.
    """
    try:
        import pdfkit
    except ImportError:
        logger.error("pdfkit 未安装。运行: pip install pdfkit")
        return False

    # wkhtmltopdf 配置
    config = None
    if wkhtmltopdf_path:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    # 渲染选项: 优化金融文档排版
    options = {
        "page-size": "A4",
        "margin-top": "10mm",
        "margin-right": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "10mm",
        "encoding": "UTF-8",
        "no-outline": None,
        "enable-local-file-access": None,  # 允许加载本地 CSS/图片资源
        "disable-smart-shrinking": None,   # 禁用自动缩放, 保持表格原始排版
        "print-media-type": None,          # 使用 print 媒体类型的 CSS
        "quiet": None,                     # 静默模式
    }

    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdfkit.from_file(str(html_path), str(pdf_path), options=options, configuration=config)
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"  PDF 转换失败: {html_path.name} -> {e}")
        return False


# =====================================================================
# SEC EDGAR 下载 + 文件扫描
# =====================================================================

def extract_filing_date(accession_dir: Path) -> tuple[str, str]:
    """
    从 filing 目录提取精确日期。

    数据源优先级:
      1. full-submission.txt 中的 CONFORMED PERIOD OF REPORT (报告覆盖期)
         和 FILED AS OF DATE (提交日期)  ← SEC 官方必有字段
      2. filing-details.json / index.json (sec-edgar-downloader 生成)
      3. accession number 年份推断 (最终兜底)

    Returns:
        (filed_date, period_of_report) 格式均为 "YYYY-MM-DD"
        filed_date: 提交日期 (用于 parse_fiscal_period)
        period_of_report: 报告覆盖期 (更精确的周期信号)
    """
    filed_date = None
    period_of_report = None

    # 策略 1: full-submission.txt (SEC 官方必有)
    submission_file = accession_dir / "full-submission.txt"
    if submission_file.exists():
        try:
            # 头部元数据通常在前 3000 字符
            with open(submission_file, "r", errors="ignore") as f:
                header = f.read(5000)
            # CONFORMED PERIOD OF REPORT:	20211225
            por_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)
            if por_match:
                d = por_match.group(1)
                period_of_report = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            # FILED AS OF DATE:		20220128
            fad_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", header)
            if fad_match:
                d = fad_match.group(1)
                filed_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        except Exception:
            pass

    # 策略 2: JSON 元数据
    if not filed_date:
        for meta_name in ["filing-details.json", "index.json"]:
            meta_file = accession_dir / meta_name
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for key in ("filingDate", "filing_date", "date"):
                        if key in data:
                            filed_date = data[key]
                            break
                except (json.JSONDecodeError, KeyError):
                    pass

    # 策略 3: accession number 兜底
    if not filed_date:
        match = re.search(r"-(\d{2})-", accession_dir.name)
        if match:
            year_2d = int(match.group(1))
            year = 2000 + year_2d if year_2d < 50 else 1900 + year_2d
            filed_date = f"{year}-06-15"
        else:
            filed_date = "2023-01-01"

    if not period_of_report:
        period_of_report = filed_date

    return filed_date, period_of_report


def find_primary_html(accession_dir: Path) -> Optional[Path]:
    """
    在 sec-edgar-downloader 的 accession 目录中定位主文档 HTML。

    优先级:
      1. primary-document.html (sec-edgar-downloader 3.x 标准)
      2. 任何 .htm / .html 文件 (排除 full-submission)
      3. 如果已有 .pdf, 直接返回 (SEC 偶尔提供 PDF)
    """
    # 优先: 官方命名
    for name in ["primary-document.html", "primary-document.htm"]:
        candidate = accession_dir / name
        if candidate.exists():
            return candidate

    # 搜索 .htm / .html
    for ext in [".htm", ".html"]:
        candidates = [
            f for f in accession_dir.glob(f"*{ext}")
            if "full-submission" not in f.name.lower()
            and "filing-details" not in f.name.lower()
        ]
        if candidates:
            # 优先选最大的文件 (主文档通常最大)
            return max(candidates, key=lambda f: f.stat().st_size)

    return None


def find_existing_pdf(accession_dir: Path) -> Optional[Path]:
    """检查 accession 目录中是否已有 PDF 文件。"""
    pdfs = [
        f for f in accession_dir.glob("*.pdf")
        if "full-submission" not in f.name.lower()
    ]
    if pdfs:
        return max(pdfs, key=lambda f: f.stat().st_size)
    return None


def collect_and_convert_filings(
    download_dir: Path,
    ticker: str,
    filing_type: str,
    wkhtmltopdf_path: Optional[str] = None,
) -> list[dict]:
    """
    扫描 sec-edgar-downloader 输出 → 定位 HTML → 转换 PDF。

    Returns:
        list of {"accession": str, "pdf_path": Path, "date": str}
    """
    base = download_dir / "sec-edgar-filings" / ticker / filing_type
    if not base.exists():
        return []

    results = []
    for accession_dir in sorted(base.iterdir()):
        if not accession_dir.is_dir():
            continue

        accession = accession_dir.name
        filed_date, period_of_report = extract_filing_date(accession_dir)

        # 策略 1: 已有 PDF → 直接使用
        existing_pdf = find_existing_pdf(accession_dir)
        if existing_pdf:
            logger.info(f"  [PDF-DIRECT] {accession}: {existing_pdf.name}")
            results.append({
                "accession": accession,
                "pdf_path": existing_pdf,
                "filed_date": filed_date,
                "period_of_report": period_of_report,
            })
            continue

        # 策略 2: HTML → PDF 转换
        html_file = find_primary_html(accession_dir)
        if html_file is None:
            logger.warning(f"  [SKIP] {accession}: 无 HTML 或 PDF 主文档")
            continue

        # 转换目标: 放在同目录下
        converted_pdf = accession_dir / f"{html_file.stem}.pdf"

        if converted_pdf.exists() and converted_pdf.stat().st_size > 0:
            logger.info(f"  [CACHED] {accession}: 已有转换 PDF")
        else:
            logger.info(f"  [CONVERT] {accession}: {html_file.name} -> PDF")
            success = convert_html_to_pdf(
                html_file, converted_pdf, wkhtmltopdf_path,
            )
            if not success:
                logger.warning(f"  [FAIL] {accession}: HTML->PDF 转换失败, 跳过")
                continue

        results.append({
            "accession": accession,
            "pdf_path": converted_pdf,
            "filed_date": filed_date,
            "period_of_report": period_of_report,
        })

    return results


# =====================================================================
# 结构化存储 + 元数据索引
# =====================================================================

def organize_filings(
    download_dir: Path,
    output_dir: Path,
    ticker: str,
    industry: str,
    filing_type: str,
    wkhtmltopdf_path: Optional[str] = None,
) -> list[dict]:
    """
    下载 → 转换 → 重组为 Concentric Adversarial Storage 结构。

    输出: {output_dir}/{industry}/{ticker}/{FY}_{Q}_{Type}.pdf
    """
    target_dir = output_dir / industry / ticker
    target_dir.mkdir(parents=True, exist_ok=True)

    filings = collect_and_convert_filings(
        download_dir, ticker, filing_type, wkhtmltopdf_path,
    )
    organized = []

    for filing in filings:
        period = infer_fiscal_period(ticker, filing["period_of_report"], filing_type)
        fy = period["fiscal_year"]
        q = period["quarter"]
        rt = period["report_type"]
        filename = f"{fy}_{q}_{rt}.pdf"
        dest = target_dir / filename

        if dest.exists():
            logger.info(f"  [SKIP] {dest.name} (already exists)")
            # 仍然记录元数据
            organized.append({
                "ticker": ticker,
                "industry": industry,
                "fiscal_year": fy,
                "quarter": q,
                "report_type": rt,
                "file_path": str(dest.relative_to(output_dir)),
                "accession": filing["accession"],
                "filing_date": filing["filed_date"],
            })
            continue

        # 复制转换后的 PDF
        shutil.copy2(filing["pdf_path"], dest)
        file_size_kb = dest.stat().st_size / 1024
        logger.info(f"  [OK] {dest.name} ({file_size_kb:.0f} KB)")

        organized.append({
            "ticker": ticker,
            "industry": industry,
            "fiscal_year": fy,
            "quarter": q,
            "report_type": rt,
            "file_path": str(dest.relative_to(output_dir)),
            "accession": filing["accession"],
            "filing_date": filing["filed_date"],
        })

    return organized


def save_filing_index(all_metadata: list[dict], output_path: Path):
    """
    增量保存 filing_index.json (Layer 1 Denoising 锚点)。

    格式: [{"ticker", "industry", "fiscal_year", "quarter", "report_type", "file_path"}, ...]
    """
    # 合并已有索引 (增量)
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing = []

        # 去重: 按 (ticker, fiscal_year, quarter, report_type) 为 key
        existing_keys = {
            (m["ticker"], m["fiscal_year"], m["quarter"], m["report_type"])
            for m in existing
        }
        for m in all_metadata:
            key = (m["ticker"], m["fiscal_year"], m["quarter"], m["report_type"])
            if key not in existing_keys:
                existing.append(m)
                existing_keys.add(key)
        all_metadata = existing

    # 排序
    all_metadata.sort(key=lambda m: (
        m.get("industry", ""), m.get("ticker", ""),
        m.get("fiscal_year", ""), m.get("quarter", ""),
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\n[INDEX] {output_path}: {len(all_metadata)} filings")

    # 统计
    by_industry: dict[str, int] = {}
    for m in all_metadata:
        ind = m.get("industry", "unknown")
        by_industry[ind] = by_industry.get(ind, 0) + 1
    for ind, cnt in sorted(by_industry.items()):
        logger.info(f"  {ind}: {cnt} filings")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SEC EDGAR HTML-to-PDF Filing Pipeline"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw_pdfs",
        help="Concentric Adversarial Storage root",
    )
    parser.add_argument(
        "--temp-dir", type=str, default="data/.sec_download_cache",
        help="sec-edgar-downloader cache dir",
    )
    parser.add_argument(
        "--index-path", type=str, default="data/vision_rag/filing_index.json",
        help="filing_index.json output path",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2022, 2023],
        help="Target fiscal years (filing submission window)",
    )
    parser.add_argument(
        "--industries", type=str, nargs="+", default=None,
        help="Industries to download (default: all)",
    )
    parser.add_argument(
        "--tickers", type=str, nargs="+", default=None,
        help="Specific tickers (overrides --industries)",
    )
    parser.add_argument(
        "--limit-per-type", type=int, default=10,
        help="Max filings per ticker per type",
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="SEC User-Agent: Your name (e.g. 'Zhichao Pan')",
    )
    parser.add_argument(
        "--email", type=str, required=True,
        help="SEC User-Agent: Your email",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without downloading",
    )
    args = parser.parse_args()

    # ---- 检查 wkhtmltopdf ----
    wkhtmltopdf_path = get_wkhtmltopdf_path()
    if wkhtmltopdf_path:
        logger.info(f"wkhtmltopdf: {wkhtmltopdf_path}")
    else:
        logger.error(
            "wkhtmltopdf 未找到!\n"
            "  Windows: choco install wkhtmltopdf\n"
            "     或下载: https://wkhtmltopdf.org/downloads.html\n"
            "  Linux:   sudo apt-get install wkhtmltopdf\n"
            "  macOS:   brew install wkhtmltopdf"
        )
        sys.exit(1)

    # ---- 检查 sec-edgar-downloader ----
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        logger.error("sec-edgar-downloader 未安装。运行: pip install sec-edgar-downloader")
        sys.exit(1)

    # ---- 构建 ticker 列表 ----
    if args.tickers:
        tickers_to_download = [
            (t, TICKER_TO_INDUSTRY.get(t, "technology")) for t in args.tickers
        ]
    else:
        industries = args.industries or list(INDUSTRY_TICKERS.keys())
        tickers_to_download = []
        for industry in industries:
            if industry in INDUSTRY_TICKERS:
                for t in INDUSTRY_TICKERS[industry]:
                    tickers_to_download.append((t, industry))

    # ---- 日期范围 ----
    min_year, max_year = min(args.years), max(args.years)
    after_date = f"{min_year}-01-01"
    before_date = f"{max_year + 1}-12-31"

    filing_types = ["10-K", "10-Q"]
    total_tasks = len(tickers_to_download) * len(filing_types)
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plan Summary ----
    print("=" * 60)
    print("SEC EDGAR HTML-to-PDF Pipeline")
    print("=" * 60)
    print(f"  Tickers:  {[t for t, _ in tickers_to_download]}")
    print(f"  Forms:    {filing_types}")
    print(f"  Window:   {after_date} ~ {before_date}")
    print(f"  Tasks:    {total_tasks}")
    print(f"  Output:   {output_dir}")
    print(f"  PDF Engine: {wkhtmltopdf_path}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        return

    # ---- 初始化 SEC 下载器 ----
    dl = Downloader(args.name, args.email, str(temp_dir))

    all_metadata = []
    completed = 0

    for ticker, industry in tickers_to_download:
        print(f"\n--- {ticker} ({industry}) ---")

        for ft in filing_types:
            completed += 1
            print(f"  [{completed}/{total_tasks}] Downloading {ft}...")

            # Step 1: 下载 HTML filing
            try:
                dl.get(
                    ft, ticker,
                    limit=args.limit_per_type,
                    after=after_date,
                    before=before_date,
                    download_details=True,
                )
                time.sleep(0.3)  # SEC 速率限制 (保守)
            except Exception as e:
                logger.error(f"  Download failed: {ticker}/{ft}: {e}")
                continue

            # Step 2: HTML→PDF 转换 + 结构化存储
            organized = organize_filings(
                temp_dir, output_dir, ticker, industry, ft,
                wkhtmltopdf_path=wkhtmltopdf_path,
            )
            all_metadata.extend(organized)

    # ---- 保存元数据索引 ----
    save_filing_index(all_metadata, Path(args.index_path))

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete: {len(all_metadata)} filings")
    print(f"  PDFs:  {output_dir}")
    print(f"  Index: {args.index_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
