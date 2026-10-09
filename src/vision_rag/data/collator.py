"""
生产级批次整理器 (Production Batch Collator)

核心职责:
  1. 将 Dataset 的单样本字典列表组装为批次张量字典
  2. 通过 VLM image_processor 对图像做标准化 + 生成 pixel_attention_mask
  3. pixel_attention_mask 传播到 MaxSim, 屏蔽 padding patch 的注意力

Vision Padding Trap 防御:
  ColPali 接收不同长宽比的 PDF 页图时, image_processor 会做 Fit Resize + Padding。
  Padding 区域是黑色补零像素, 如果不被 mask 掉, 模型会学到"黑边=有信息",
  导致 patch attention 向无意义区域发散, 检索质量崩溃。

  解决方案:
    - 真实模式: ColPaliProcessor.process_images() 自动输出 pixel_attention_mask
    - Mock 模式: 全 1 mask (Mock 图片无 padding)
"""

import torch
from typing import Any, Optional, Callable

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

class VisFinRAGCollator:
    """
    VisFinRAG 训练用 Batch Collator.

    Args:
        image_processor: 可选的 VLM 图像预处理器.
            - 真实模式: ColPaliProcessor (处理 resize/pad/normalize + 生成 attention_mask)
            - Mock 模式: None (直接 stack)
        mock_mode: 是否为 Mock 模式 (跳过 processor, 直接 stack)

    输入: list[dict], 每个 dict 包含:
      - "image":        [C, H, W] 或 PIL.Image
      - "query_text":   str
      - "soft_mask":    [L] IoPA 软掩码
      - "padding_mask": [L] Padding 区域掩码
      - "meta_q":       dict
      - "meta_d":       dict

    输出: dict, 包含:
      - "images":               [B, C, H, W] (已标准化) 或 list[PIL.Image]
      - "query_text":           list[str]
      - "soft_masks":           [B, L]
      - "padding_mask":         [B, L]
      - "meta_q":               list[dict]
      - "meta_d":               list[dict]
    """

    def __init__(
        self,
        image_processor: Optional[Any] = None,
        mock_mode: bool = True,
    ):
        self.image_processor = image_processor
        self.mock_mode = mock_mode

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        query_texts = [f["query_text"] for f in features]
        soft_masks = torch.stack([f["soft_mask"] for f in features])
        padding_masks = torch.stack([f["padding_mask"] for f in features])
        meta_q = [f["meta_q"] for f in features]
        meta_d = [f["meta_d"] for f in features]

        first_image = features[0]["image"]
        
        # Determine if images are PIL -> meaning we are in real mode
        if PILImage is not None and isinstance(first_image, PILImage.Image):
            images = [f["image"] for f in features]
        else:
            # Tensor -> mock mode or pre-processed
            images = torch.stack([f["image"] for f in features])

        batch = {
            "images": images,
            "query_text": query_texts,
            "soft_masks": soft_masks,
            "padding_mask": padding_masks,
            "meta_q": meta_q,
            "meta_d": meta_d,
        }
        return batch
