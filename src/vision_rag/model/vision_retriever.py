"""
完整检索模型组装 (Vision Retriever)

将所有子模块拼装为统一的 nn.Module:
  Backbone → Query Encoder → NumericAwareMaxSim → Score Matrix

对外暴露:
  forward(images, query_text, soft_masks, padding_mask)
    → scores[B,B], gate[B,N]

  encode_document(images) → E_d[B, M, D]
  encode_query(query_text) → E_q[B, N, D], gate[B, N]
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import MockColPaliBackbone, ColPaliBackbone
from .numeric_maxsim import NumericAwareMaxSim


class VisionRetriever(nn.Module):
    """
    Structure-Aware Vision Late Interaction Retriever。

    子模块组合:
      1. Backbone: ColPali (或 Mock) → 图像/文本特征
      2. NumericAwareMaxSim: 凸组合门控相似度
      3. Query Gate: 内嵌在 MaxSim 中

    Args:
        projection_dim: ColBERT 投影维度
        gamma: 凸组合偏置
        gate_init_bias: Gate 初始偏置
        use_mock: 使用 MockColPali (本地) 还是真实 ColPali (AutoDL)
        backbone_name: HuggingFace 模型 ID (仅 use_mock=False 时有效)
    """

    def __init__(
        self,
        projection_dim: int = 128,
        gamma: float = 0.1,
        gate_init_bias: float = -2.0,
        use_mock: bool = True,
        backbone_name: str = "vidore/colpali-v1.2",
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Backbone
        if use_mock:
            self.backbone = MockColPaliBackbone(
                projection_dim=projection_dim,
            )
        else:
            self.backbone = ColPaliBackbone(
                model_name=backbone_name,
                projection_dim=projection_dim,
            )

        # Numeric-Aware MaxSim 算子
        self.maxsim = NumericAwareMaxSim(
            dim=projection_dim,
            gamma=gamma,
            gate_init_bias=gate_init_bias,
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结 backbone 参数 (仅训练投影头和 Gate)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 backbone (在指定 epoch 后调用)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def encode_document(
        self,
        images: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:          # [B, M, D]
        """编码文档页面图像"""
        return self.backbone.encode_image(images)

    def encode_query(
        self,
        query_text: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码查询文本 + 计算 gate。

        Returns:
            E_q: [B, N, D]
            gate: [B, N]
        """
        E_q = self.backbone.encode_query(query_text)
        gate = self.maxsim.compute_gate(E_q)
        return E_q, gate

    @staticmethod
    def _adapt_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """将 mask 从任意长度自适应到 target_len（截断或插值）。"""
        if mask.shape[-1] == target_len:
            return mask
        if mask.shape[-1] > target_len:
            return mask[..., :target_len]
        # 插值扩展
        return torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(), size=target_len, mode="linear", align_corners=False
        ).squeeze(1).to(mask.dtype)

    def forward(
        self,
        images: torch.Tensor,           # [B, C, H, W]
        query_text: list[str],           # B 个 query 字符串
        soft_masks: torch.Tensor,        # [B, M]
        padding_mask: Optional[torch.Tensor] = None,  # [B, M]
        all_pairs: bool = True,          # 训练模式
    ) -> dict[str, torch.Tensor]:
        """
        完整前向传播。

        Returns:
            dict with:
              "scores": [B, B] (all_pairs=True) or [B] (all_pairs=False)
              "gate": [B, N]
              "E_q": [B, N, D]
              "E_d": [B, M, D]
        """
        # 编码
        E_d = self.encode_document(images)
        E_q = self.backbone.encode_query(query_text)

        # 自适应 mask 到模型实际输出的序列长度
        M = E_d.shape[1]  # ColPali 实际输出的 token 数
        if soft_masks is not None and soft_masks.shape[-1] != M:
            soft_masks = self._adapt_mask(soft_masks, M)
        if padding_mask is not None and padding_mask.shape[-1] != M:
            padding_mask = self._adapt_mask(padding_mask, M)

        # MaxSim (gate 在内部计算)
        scores, gate = self.maxsim(
            E_q, E_d, soft_masks, padding_mask, all_pairs=all_pairs,
        )

        return {
            "scores": scores,
            "gate": gate,
            "E_q": E_q,
            "E_d": E_d,
        }
