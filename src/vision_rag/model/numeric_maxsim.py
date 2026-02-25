"""
Numeric-Aware MaxSim 算子 (Convex Combination Gated)

核心算子: Structure-Aware Late Interaction 的相似度计算引擎。
采用凸组合门控设计, 保证分数严格有界, 避免 InfoNCE 温度缩放下的数值溢出。

数学公式:
  S_std(q_i, d) = max_j (q_i^T · d_j)
  S_num(q_i, d) = max_j (q_i^T · d_j · M_soft(j))
  S(q_i, d) = (1 - g_i) · S_std + g_i · (S_num + γ)

  其中 g_i = σ(w_g^T · q_i + b_g) 是 query-conditioned gate

分数有界性保证:
  S_std ∈ [-1, 1] (L2 归一化后的余弦相似度)
  S_num ∈ [-1, 1] (M_soft ∈ [0,1] 不扩大值域)
  S ∈ [-1, 1+γ]   (γ = 0.1 → 最大 1.1)

张量维度推演 (B=16, N=64, M=1024, D=128, BF16):
  E_q_exp: [16, 1, 64, 128]  → broadcast to [16, 16, 64, 128]
  E_d_exp: [1, 16, 128, 1024] → broadcast to [16, 16, 128, 1024]
  sim:     [16, 16, 64, 1024] → 16×16×64×1024 × 2B = 32 MB
  峰值显存: ~64 MB (sim + sim*mask 同时存在时)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NumericAwareMaxSim(nn.Module):
    """
    Query-Gated, Numeric-Aware MaxSim (Convex Combination 版本)

    输入:
      E_q:          [B, N, D]  query token embeddings
      E_d:          [B, M, D]  document patch embeddings
      M_soft:       [B, M]    IoPA 软掩码 ∈ [0,1]
      padding_mask: [B, M]    有效 patch = 1, padding = 0

    输出 (单对模式):   [B]       per-sample scores
    输出 (全对模式):   [B, B]    all-pairs score matrix

    Args:
        dim: embedding dimension D
        gamma: 凸组合数值偏置 (默认 0.1)
        gate_init_bias: gate 初始偏置 (默认 -2.0, sigmoid(-2)≈0.12 → 保守启动)
    """

    def __init__(
        self,
        dim: int,
        gamma: float = 0.1,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()

        # Query-side gate: 判断 query token 是否在"询问数字"
        self.gate_proj = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

        # 凸组合数值偏置 γ (固定, 不参与梯度)
        self.register_buffer("gamma", torch.tensor(gamma))

    def compute_gate(self, E_q: torch.Tensor) -> torch.Tensor:
        """
        计算 query gate: g_i = σ(w_g^T · q_i + b_g)

        Args:
            E_q: [B, N, D]

        Returns:
            gate: [B, N] ∈ [0.01, 0.99] (强制裁剪以防 NaN)
        """
        gate_logits = self.gate_proj(E_q).squeeze(-1)
        return torch.sigmoid(gate_logits).clamp(min=0.01, max=0.99)

    def forward_single(
        self,
        E_q: torch.Tensor,           # [B, N, D]
        E_d: torch.Tensor,           # [B, M, D]
        M_soft: torch.Tensor,        # [B, M]
        padding_mask: Optional[torch.Tensor] = None,  # [B, M]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单对 MaxSim: 只计算对角线 (q_i, d_i) 的分数。
        用于推理阶段 (检索 top-K 后的精排)。

        Returns:
            scores: [B]
            gate: [B, N]
        """
        B, N, D = E_q.shape
        M = E_d.shape[1]

        # L2 归一化
        E_q_norm = F.normalize(E_q, p=2, dim=-1)
        E_d_norm = F.normalize(E_d, p=2, dim=-1)

        # Query gate
        gate = self.compute_gate(E_q)  # [B, N]

        # 相似度矩阵: [B, N, M]
        sim = torch.bmm(E_q_norm, E_d_norm.transpose(1, 2))

        # 屏蔽 padding patches
        if padding_mask is not None:
            # padding_mask [B, M] → [B, 1, M]
            pad_mask = padding_mask.unsqueeze(1)
            # EMERGENCY FIX: 不要用 float("-inf"), 因为 -inf * 0 = nan
            # 使用 -100.0 对余弦相似度 [-1, 1] 已经足够作为负无穷
            sim = sim.masked_fill(pad_mask == 0, -100.0)

        # S_std: 标准 MaxSim
        S_std = sim.max(dim=-1).values  # [B, N]

        # S_num: 数值感知 MaxSim (sim * M_soft)
        M_soft_expanded = M_soft.unsqueeze(1)  # [B, 1, M]
        sim_weighted = sim * M_soft_expanded    # [B, N, M]
        S_num = sim_weighted.max(dim=-1).values  # [B, N]

        # 凸组合: S = (1-g) · S_std + g · (S_num + γ)
        S_token = (1.0 - gate) * S_std + gate * (S_num + self.gamma)  # [B, N]

        # 聚合: 所有 query token 的分数求平均以确保有界
        scores = S_token.mean(dim=-1)  # [B]

        return scores, gate

    def forward_all_pairs(
        self,
        E_q: torch.Tensor,           # [B, N, D]
        E_d: torch.Tensor,           # [B, M, D]
        M_soft: torch.Tensor,        # [B, M]
        padding_mask: Optional[torch.Tensor] = None,  # [B, M]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        全对 MaxSim: 计算 Batch 内所有 B×B 的 query-doc 交叉相似度。
        用于训练阶段的 In-Batch Contrastive Learning。

        张量维度推演 (以 B=16, N=64, M=1024, D=128 为例):
        ┌─────────────────────────────────────────────────────────────┐
        │ Step 1: L2 归一化                                          │
        │   E_q_norm: [16, 64, 128]   E_d_norm: [16, 1024, 128]     │
        │                                                             │
        │ Step 2: 扩维准备广播                                       │
        │   E_q_exp: [16, 1, 64, 128]   (第 i 个 query 广播到所有 doc)│
        │   E_d_exp: [1, 16, 1024, 128] (第 j 个 doc 广播到所有 query)│
        │                                                             │
        │ Step 3: matmul 计算全对相似度                               │
        │   sim = matmul(E_q_exp, E_d_exp.T)                         │
        │       = [16,1,64,128] × [1,16,128,1024]                    │
        │       → [16, 16, 64, 1024]                                 │
        │   显存: 16×16×64×1024 × 2B(BF16) = 32 MB                  │
        │                                                             │
        │ Step 4: Gate + Mask 广播注入                                │
        │   gate:   [16, 64] → [16, 1, 64]     (query 维度)          │
        │   M_soft: [16, 1024] → [1, 16, 1024]  (doc 维度)           │
        │   sim_weighted = sim * M_soft_exp                           │
        │       = [16,16,64,1024] * [1,16,1,1024]                    │
        │       → [16, 16, 64, 1024]  (额外 32 MB)                  │
        │                                                             │
        │ Step 5: MaxSim + 凸组合                                    │
        │   S_std:  max(sim, dim=-1)    → [16, 16, 64]               │
        │   S_num:  max(sim_w, dim=-1)  → [16, 16, 64]               │
        │   S_token: convex combo       → [16, 16, 64]               │
        │                                                             │
        │ Step 6: 聚合 query tokens                                  │
        │   scores = S_token.sum(dim=-1) → [16, 16]                  │
        │                                                             │
        │ 峰值显存 (BF16): ~64 MB (sim + sim_weighted 同时存在)       │
        │ 24 GB VRAM 预算: 64 MB 仅占 0.26%, 完全安全                │
        └─────────────────────────────────────────────────────────────┘

        Returns:
            scores: [B, B]
            gate: [B, N]
        """
        B, N, D = E_q.shape
        M = E_d.shape[1]

        # Step 1: L2 归一化
        E_q_norm = F.normalize(E_q, p=2, dim=-1)  # [B, N, D]
        E_d_norm = F.normalize(E_d, p=2, dim=-1)  # [B, M, D]

        # Query gate
        gate = self.compute_gate(E_q)  # [B, N]

        # Step 2: 扩维准备全对广播
        E_q_exp = E_q_norm.unsqueeze(1)  # [B, 1, N, D]
        E_d_exp = E_d_norm.unsqueeze(0)  # [1, B, M, D]

        # Step 3: 全对相似度矩阵
        # matmul: [B, 1, N, D] × [1, B, D, M] → [B, B, N, M]
        sim = torch.matmul(E_q_exp, E_d_exp.transpose(-1, -2))

        # 屏蔽 padding patches
        if padding_mask is not None:
            # padding_mask: [B, M] → [1, B, 1, M]
            pad_mask_exp = padding_mask.unsqueeze(0).unsqueeze(2)
            # EMERGENCY FIX: 不要用 float("-inf"), 因为 -inf * 0 = nan
            sim = sim.masked_fill(pad_mask_exp == 0, -100.0)

        # Step 4: S_std — 标准 MaxSim
        S_std = sim.max(dim=-1).values  # [B, B, N]

        # Step 5: S_num — 数值感知 MaxSim
        # M_soft: [B, M] → [1, B, 1, M]
        M_soft_exp = M_soft.unsqueeze(0).unsqueeze(2)
        sim_weighted = sim * M_soft_exp  # [B, B, N, M]
        S_num = sim_weighted.max(dim=-1).values  # [B, B, N]

        # Step 6: 凸组合
        # gate: [B, N] → [B, 1, N] (query gate 只沿 query 维展开)
        gate_exp = gate.unsqueeze(1)  # [B, 1, N]
        S_token = (1.0 - gate_exp) * S_std + gate_exp * (S_num + self.gamma)

        # Step 7: 聚合 query tokens (使用 mean 保证严格有界在 [-1, 1+gamma])
        scores = S_token.mean(dim=-1)  # [B, B]

        return scores, gate

    def forward(
        self,
        E_q: torch.Tensor,
        E_d: torch.Tensor,
        M_soft: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        all_pairs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        统一入口: 根据 all_pairs 标志选择计算模式。

        Args:
            all_pairs: True → 训练模式 (返回 [B,B]), False → 推理模式 (返回 [B])

        Returns:
            scores: [B, B] or [B]
            gate: [B, N]
        """
        if all_pairs:
            return self.forward_all_pairs(E_q, E_d, M_soft, padding_mask)
        else:
            return self.forward_single(E_q, E_d, M_soft, padding_mask)
