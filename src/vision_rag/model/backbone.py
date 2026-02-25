"""
视觉骨干网络封装 (Vision Backbone Wrapper)

提供两个实现:
  1. MockColPaliBackbone: 本地测试用, 输出维度与真实 ColPali 一致的随机张量
  2. ColPaliBackbone: AutoDL 部署用, 加载真实 ColPali-v1.2 权重

ColPali (PaliGemma 3B) 输出规格:
  - 视觉特征: [B, 1030, 1152]  (1024 patches + 6 special tokens, hidden_dim=1152)
  - 需经过 D-dim 投影后变为 [B, 1030, 128]

接口契约:
  encode_image(images: Tensor[B,C,H,W]) → Tensor[B, M, D]
  encode_query(query_ids: Tensor[B, N]) → Tensor[B, N, D]
"""

import torch
import torch.nn as nn
from typing import Optional


class MockColPaliBackbone(nn.Module):
    """
    ColPali 骨干的 Mock 实现, 用于本地全流程闭环测试。

    特性:
      - 输出维度严格对齐真实 ColPali: M=1030, hidden_dim=1152
      - 含可训练的 Linear 层 (验证梯度流)
      - 无需下载权重, 纯 CPU 可运行

    训练循环中的使用方式与真实模型 100% 相同:
      E_d = backbone.encode_image(images)
      E_q = backbone.encode_query(query_text)
    """

    # ColPali-v1.2 (PaliGemma 3B) 真实参数
    HIDDEN_DIM = 1152       # PaliGemma hidden size
    NUM_PATCHES = 1030      # 1024 vision patches + 6 special tokens
    MAX_QUERY_TOKENS = 64   # Query 截断长度

    def __init__(
        self,
        projection_dim: int = 128,
        hidden_dim: int = HIDDEN_DIM,
        num_patches: int = NUM_PATCHES,
        max_query_tokens: int = MAX_QUERY_TOKENS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.max_query_tokens = max_query_tokens
        self.projection_dim = projection_dim

        # 模拟 backbone 的可训练参数 (保证梯度流测试有效)
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # ColBERT-style 投影头: hidden_dim → D
        self.output_proj = nn.Linear(hidden_dim, projection_dim)

    def encode_image(
        self,
        images: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:          # [B, M, D]
        """
        模拟图像编码: 生成 [B, num_patches, projection_dim] 的特征。

        真实实现中这里调用 PaliGemma 的 forward。
        """
        B = images.shape[0]
        device = images.device
        dtype = images.dtype

        # 生成伪特征 (纯随机, 保证两个样本即便图像相同特征也绝对不同)
        mock_hidden = torch.randn(
            B, self.num_patches, self.hidden_dim,
            device=device, dtype=dtype,
        )

        # 过可训练层 (验证梯度)
        mock_hidden = self.image_proj(mock_hidden)
        features = self.output_proj(mock_hidden)  # [B, M, D]

        return features

    def encode_query(
        self,
        query_text: list[str],  # B 个 query 字符串
    ) -> torch.Tensor:           # [B, N, D]
        """
        模拟文本编码: 生成 [B, max_query_tokens, projection_dim] 的特征。

        真实实现中这里调用 Tokenizer + PaliGemma text branch。
        """
        B = len(query_text)
        device = self.output_proj.weight.device
        dtype = self.output_proj.weight.dtype

        mock_hidden = torch.randn(
            B, self.max_query_tokens, self.hidden_dim,
            device=device, dtype=dtype,
        )

        mock_hidden = self.query_proj(mock_hidden)
        features = self.output_proj(mock_hidden)  # [B, N, D]

        return features


class ColPaliBackbone(nn.Module):
    """
    真实 ColPali-v1.2 骨干网络封装。

    仅在 AutoDL (有 GPU + 足够显存) 上实例化。
    本地代码中通过条件 import 切换:

        if config.use_mock:
            backbone = MockColPaliBackbone(projection_dim=128)
        else:
            backbone = ColPaliBackbone(model_name="vidore/colpali-v1.2", ...)

    Args:
        model_name: HuggingFace 模型 ID
        projection_dim: ColBERT 投影维度
        device: 目标设备
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        projection_dim: int = 128,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        # 延迟导入: 本地环境可能没有 colpali-engine
        try:
            from colpali_engine.models import ColPali
            from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
        except ImportError:
            raise ImportError(
                "colpali-engine 未安装。请在 AutoDL 环境中运行:\n"
                "pip install colpali-engine>=0.3.0"
            )

        # 加载预训练模型
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.processor = ColPaliProcessor.from_pretrained(model_name)

        # 无论维度是否相等，我们都在最后加一个 projection，
        # 这保证了再 8GB VRAM 上冻结 ColPali 主干后，仍有一层轻量的 Linear 能够进行 overfit verification。
        original_dim = self.model.dim  # ColPali 内部属性 (128)
        self.custom_proj = nn.Linear(original_dim, projection_dim, dtype=self.model.dtype, device=self.model.device)
        
        # 数学关键：如果是 128 -> 128，必须初始化为 Identity，否则随机正交变换会彻底摧毁预训练的 ColBERT Late Interaction 分布！
        if original_dim == projection_dim:
            with torch.no_grad():
                self.custom_proj.weight.copy_(torch.eye(projection_dim, dtype=self.model.dtype, device=self.model.device))
                self.custom_proj.bias.fill_(0.0)

    def encode_image(
        self,
        images: list,  # list of PIL Images
    ) -> torch.Tensor:
        """
        编码文档页面图像。
        冻结时用 no_grad 加速；解冻后允许梯度回传。
        """
        batch_images = self.processor.process_images(images)
        batch_images = {k: v.to(self.model.device) for k, v in batch_images.items()}

        if not any(p.requires_grad for p in self.model.parameters()):
            # backbone 冻结 → no_grad 加速
            with torch.no_grad():
                embeddings = self.model(**batch_images)
        else:
            # backbone 解冻 → 需要梯度流
            embeddings = self.model(**batch_images)

        return self.custom_proj(embeddings)

    def encode_query(
        self,
        query_text: list[str],
    ) -> torch.Tensor:
        """
        编码查询文本。
        冻结时用 no_grad 加速；解冻后允许梯度回传。
        """
        batch_queries = self.processor.process_queries(query_text)
        batch_queries = {k: v.to(self.model.device) for k, v in batch_queries.items()}

        if not any(p.requires_grad for p in self.model.parameters()):
            with torch.no_grad():
                embeddings = self.model(**batch_queries)
        else:
            embeddings = self.model(**batch_queries)

        return self.custom_proj(embeddings)
