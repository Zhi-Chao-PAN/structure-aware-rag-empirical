"""
门控加权 Patch 注意力提取 (Gate-Weighted Patch Attention Extraction)

核心职责:
  从 Late Interaction 的相似度矩阵中反向推导每个 document patch 的"重要性"。

  SOTA 修正: 使用 Query Gate 加权, 仅让语义相关的 query token 贡献注意力权重。
  停用词 token (如 "what", "is", "the") 的 gate 值接近 0, 不会产生噪声。
"""

import torch
import torch.nn.functional as F


def extract_patch_attention(
    E_q: torch.Tensor,        # [1, N, D] 单个 query
    E_d: torch.Tensor,        # [1, M, D] 单个检索到的文档
    gate: torch.Tensor,       # [1, N] query gate values
    padding_mask: torch.Tensor | None = None,  # [1, M]
) -> torch.Tensor:            # [M] 每个 patch 的注意力权重
    """
    从 Late Interaction 中提取 patch-level 注意力分布。

    方法:
      1. 计算 [N, M] 逐 token 相似度矩阵
      2. 每个 query token 找到其 argmax patch
      3. 将 gate-weighted 相似度累加到被选中的 patch 上

    gate 加权的意义:
      - 停用词 ("what", "is") 的 gate ≈ 0 → 不贡献注意力
      - 关键词 ("revenue", "2023") 的 gate > 0 → 精准聚焦
      - 结果: 注意力像激光一样打在数字和表头区域

    Args:
        E_q: [1, N, D] 单个 query 的 token embeddings
        E_d: [1, M, D] 单个文档的 patch embeddings
        gate: [1, N] query gate 值 ∈ [0, 1]
        padding_mask: [1, M] 有效 patch=1, padding=0

    Returns:
        patch_attention: [M] 每个 patch 的累积注意力权重 (已归一化到 [0,1])
    """
    N = E_q.shape[1]
    M = E_d.shape[1]

    # 相似度矩阵
    sim = torch.mm(
        F.normalize(E_q.squeeze(0), dim=-1),
        F.normalize(E_d.squeeze(0), dim=-1).T,
    )  # [N, M]

    # 屏蔽 padding
    if padding_mask is not None:
        pad = padding_mask.squeeze(0)  # [M]
        sim = sim.masked_fill(pad.unsqueeze(0) == 0, float("-inf"))

    # 每个 query token 的最佳匹配 patch
    max_patch_per_query = sim.argmax(dim=-1)    # [N]
    max_sim_per_query = sim.max(dim=-1).values  # [N]

    # Gate 加权: 停用词 token 的贡献被压制
    gate_values = gate.squeeze(0)  # [N]
    weighted_sim = max_sim_per_query * gate_values  # [N]

    # 累加到 patch 维度
    patch_attention = torch.zeros(M, device=sim.device, dtype=sim.dtype)
    patch_attention.scatter_add_(0, max_patch_per_query, weighted_sim)

    # 归一化到 [0, 1]
    att_max = patch_attention.max()
    if att_max > 0:
        patch_attention = patch_attention / att_max

    return patch_attention
