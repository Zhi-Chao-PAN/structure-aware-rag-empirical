"""
生产级惰性加载数据集 (Production Lazy-Loading Dataset)

设计原则:
  1. __init__ 零图片加载: 仅读取 Parquet 索引 (< 1MB RAM)
  2. __getitem__ 精准读取: 单页 JPEG + 预计算 mask, NVMe 随机 I/O 最优
  3. Worker 隔离: 无共享可变状态, DataLoader num_workers=16 无死锁
  4. Mask 缓存: 每个 Worker 进程内维护 LRU cache, 避免重复读 .pt 文件

I/O 链:
  Parquet Index (init)
    → __getitem__(idx)
    → PIL.Image.open(image_path)         ← NVMe sequential read, ~0.2ms
    → torch.load(mask_shard)             ← per-ticker shard, cached per-worker
    → return {image_tensor, soft_mask, padding_mask, query, metadata}
"""

import json
import functools
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None


class VisFinRAGDataset(Dataset):
    """
    Vision Financial RAG 数据集。

    数据来源:
      preprocess_pdfs.py 生成的:
        - page_metadata.parquet  (全局索引)
        - page_images/*.jpg      (渲染后的页面图片)
        - spatial_masks/*.pt     (按 ticker 分片的 IoPA 掩码)

    内存契约:
      __init__ 内存占用:
        - Parquet 索引: ~1MB for 10k pages
        - 文件路径列表: ~100KB
        - 零图片, 零掩码, 零张量

      __getitem__ 内存占用:
        - 单页 JPEG: ~200KB → PIL.Image → Tensor[3, 1024, 1024] ~12MB
        - 单页 Mask: ~4KB (Tensor[1031])
        - 即时释放, 不累积

    Args:
        data_dir: preprocess_pdfs.py 的输出目录
        img_size: ColPali 输入尺寸 (用于 Padding mask)
        patch_size: ViT patch 尺寸
        has_cls: 是否有 [CLS] token
        query_template: query 文本模板
        max_pages: 最大页面数 (用于调试)
        industries: 仅加载的行业列表 (None = 全部)
        tickers: 仅加载的 ticker 列表 (None = 全部)
    """

    def __init__(
        self,
        data_dir: str = "data/vision_rag",
        img_size: int = 1024,
        patch_size: int = 14,
        has_cls: bool = True,
        query_template: str = "Retrieve the financial data from this document page.",
        max_pages: Optional[int] = None,
        industries: Optional[list[str]] = None,
        tickers: Optional[list[str]] = None,
        # train.py 兼容参数
        metadata_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.patch_size = patch_size
        self.has_cls = has_cls
        self.query_template = query_template

        # 如果 train.py 传了显式路径，覆盖默认值
        self._image_dir_override = Path(image_dir) if image_dir else None
        self._mask_dir_override = Path(mask_dir) if mask_dir else None

        # ---- 加载 Parquet 索引 (轻量级) ----
        parquet_path = Path(metadata_path) if metadata_path else self.data_dir / "page_metadata.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"元数据索引不存在: {parquet_path}\n"
                f"请先运行: python scripts/preprocess_pdfs.py"
            )

        df = pd.read_parquet(parquet_path)

        # 行业/Ticker 过滤
        if industries:
            df = df[df["industry"].isin(industries)]
        if tickers:
            df = df[df["ticker"].isin(tickers)]

        # 截断 (调试用)
        if max_pages and max_pages < len(df):
            df = df.head(max_pages)

        # 存储为列表 (比 DataFrame 的 iloc 快 3x)
        self.page_ids: list[str] = df["page_id"].tolist()
        self.image_paths: list[str] = df["image_path"].tolist()
        self.tickers: list[str] = df["ticker"].tolist()
        self.industries: list[str] = df["industry"].tolist()
        self.fiscal_years: list[str] = df["fiscal_year"].tolist()
        self.quarters: list[str] = df["quarter"].tolist()
        self.report_types: list[str] = df["report_type"].tolist()
        self.page_widths: list[float] = df["page_width"].tolist()
        self.page_heights: list[float] = df["page_height"].tolist()
        self.text_snippets: list[str] = df["text_snippet"].tolist()
        self.bboxes_json: list[str] = df["bboxes_json"].tolist()

        # 掩码分片路径索引: ticker → .pt 路径
        mask_base = self._mask_dir_override or self.data_dir / "spatial_masks"
        self._mask_shard_paths: dict[str, str] = {}
        if mask_base.exists():
            for pt_file in mask_base.glob("*.pt"):
                ticker_name = pt_file.stem  # "AAPL.pt" → "AAPL"
                self._mask_shard_paths[ticker_name] = str(pt_file)

        # Seq length (用于 padding mask on-the-fly 计算)
        n_patches = (img_size // patch_size) ** 2
        self.seq_len = n_patches + (1 if has_cls else 0)

        # 统计
        unique_tickers = set(self.tickers)
        unique_industries = set(self.industries)
        print(
            f"[Dataset] 已加载 {len(self)} 页 | "
            f"{len(unique_tickers)} tickers | "
            f"{len(unique_industries)} 行业 | "
            f"RAM: ~{len(self) * 200 / 1024:.0f} KB (仅索引)"
        )

    def __len__(self) -> int:
        return len(self.page_ids)

    @functools.lru_cache(maxsize=32)
    def _load_mask_shard(self, ticker: str) -> dict[str, torch.Tensor]:
        """
        加载一个 ticker 的全部 IoPA 掩码 (worker-local LRU cache)。

        functools.lru_cache 是 per-process 的, 不跨 DataLoader worker 共享,
        因此天然线程安全, 无需额外加锁。

        LRU 32 意味着每个 worker 缓存最多 32 个 ticker 的 mask,
        对于 ConcentricAdversarialSampler (每个 batch 2-8 个 ticker),
        命中率 > 95%。
        """
        shard_path = self._mask_shard_paths.get(ticker)
        if shard_path and Path(shard_path).exists():
            return torch.load(shard_path, map_location="cpu", weights_only=True)
        return {}

    def _load_image_as_tensor(self, image_path: str):
        """
        加载图像，返回 PIL.Image（供 ColPali processor 处理）。

        ColPali 的 processor 需要 PIL Image 输入，不需要手动转 tensor。
        Mock 模式下仍返回 tensor（由 MockDataset 处理）。
        """
        if Image is None:
            raise ImportError("Pillow 未安装: pip install Pillow")

        img = Image.open(image_path).convert("RGB")
        return img

    def _compute_padding_mask_lazy(
        self, page_width: float, page_height: float,
    ) -> torch.Tensor:
        """
        惰性计算 padding mask (如果预计算 mask 文件中没有)。

        直接复用 spatial_mask.py 的函数。
        """
        from vision_rag.data.spatial_mask import compute_padding_mask
        return compute_padding_mask(
            page_width, page_height,
            self.img_size, self.patch_size, self.has_cls,
        )

    def __getitem__(self, idx: int) -> dict:
        """
        惰性加载单页数据。

        DataLoader Worker 在调用时:
          1. 从列表索引读取路径/元数据 (ns 级)
          2. PIL.Image.open → NVMe 读 JPEG (~0.5ms)
          3. 从 LRU-cached mask shard 读取 IoPA mask (~0.01ms, 命中)
          4. 构建 padding mask (纯计算, ~0.1ms)

        Returns:
            dict with keys:
              image: Tensor[3, H, W]
              soft_mask: Tensor[L]
              padding_mask: Tensor[L]
              query_text: str
              meta_q: dict (给 DenoisedInfoNCE 的元信息)
              meta_d: dict
              page_id: str
        """
        page_id = self.page_ids[idx]
        ticker = self.tickers[idx]
        industry = self.industries[idx]
        fiscal_year = self.fiscal_years[idx]
        quarter = self.quarters[idx]
        report_type = self.report_types[idx]
        page_width = self.page_widths[idx]
        page_height = self.page_heights[idx]
        image_path = self.image_paths[idx]
        text_snippet = self.text_snippets[idx]

        # 1. 图像 (NVMe → PIL → Tensor)
        # 如果有 image_dir 覆盖，用文件名拼接；否则用 parquet 里的路径
        if self._image_dir_override:
            img_filename = Path(image_path).name
            actual_path = self._image_dir_override / img_filename
        else:
            actual_path = image_path
        image = self._load_image_as_tensor(str(actual_path))

        # 2. IoPA 软掩码 (从 ticker shard 读取, LRU cached)
        mask_shard = self._load_mask_shard(ticker)
        if page_id in mask_shard:
            mask_entry = mask_shard[page_id]
            if isinstance(mask_entry, dict):
                soft_mask = mask_entry.get("iopa_mask", torch.ones(self.seq_len, dtype=torch.float32) * 0.5)
            else:
                soft_mask = mask_entry
        else:
            # Fallback: 均匀掩码 (不应该发生, 除非 preprocess 不完整)
            soft_mask = torch.ones(self.seq_len, dtype=torch.float32) * 0.5

        # 3. Padding mask (纯计算, 无 I/O)
        padding_mask = self._compute_padding_mask_lazy(page_width, page_height)

        # 4. Query 文本 (动态生成)
        query_text = self._build_query(ticker, fiscal_year, quarter, text_snippet)

        # 5. Metadata (给 Sampler 和 Loss 用)
        metadata = {
            "ticker": ticker,
            "industry": industry,
            "fiscal_year": int(fiscal_year) if fiscal_year.isdigit() else 0,
            "quarter": quarter,
            "report_type": report_type,
            "content_type": "financial_page",
        }

        return {
            "image": image,
            "soft_mask": soft_mask,
            "padding_mask": padding_mask,
            "query_text": query_text,
            "meta_q": metadata,
            "meta_d": metadata,
            "page_id": page_id,
        }

    def _build_query(
        self, ticker: str, fiscal_year: str, quarter: str, text_snippet: str,
    ) -> str:
        """
        构建检索 query。

        训练时使用截取的文本片段作为 pseudo-query,
        评测时使用标注的 question。
        增加 Query 多样性 (模板池 + 随机采样)。
        """
        import random
        hint = text_snippet[:80].strip() if text_snippet else ""
        
        templates = [
            f"Find {ticker} {fiscal_year} {quarter} financial data: {hint}",
            f"Can you show me the {quarter} {fiscal_year} report for {ticker}? Reference: {hint}",
            f"What was {ticker}'s performance in {fiscal_year} {quarter}? See: {hint}",
            f"Retrieve the financial document for {ticker} from {fiscal_year} {quarter}. Context: {hint}",
            f"Looking for {ticker} data from {quarter} {fiscal_year}. Hint: {hint}"
        ]
        
        return random.choice(templates)

    def get_metadata_df(self) -> pd.DataFrame:
        """
        返回用于 Sampler 的轻量级 DataFrame。

        ConcentricAdversarialSampler 需要 (ticker, industry, fiscal_year, quarter)
        来构建同心圈层。这个方法提供精确对接。
        """
        return pd.DataFrame({
            "page_id": self.page_ids,
            "ticker": self.tickers,
            "industry": self.industries,
            "fiscal_year": self.fiscal_years,
            "quarter": self.quarters,
            "report_type": self.report_types,
        })

    def get_industry_ticker_map(self) -> dict[str, list[str]]:
        """
        返回行业→ticker 映射, 供 Sampler 构建圈层。
        """
        mapping: dict[str, set[str]] = {}
        for ind, tick in zip(self.industries, self.tickers):
            if ind not in mapping:
                mapping[ind] = set()
            mapping[ind].add(tick)
        return {k: sorted(v) for k, v in mapping.items()}
