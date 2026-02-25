"""
三层去噪 InfoNCE 损失函数 (Denoised Financial InfoNCE)

层级:
  Layer 1: 元数据感知结构性 FN 掩码 (确定性)
  Layer 2: EMA Teacher 自蒸馏软去噪 (概率性, 含课程调度)
  Layer 3: 最小负样本权重 w_min (防坍塌兜底)

公式:
  L = -1/B Σ_i log [ exp(s_ii/τ) / (exp(s_ii/τ) + Σ_{j≠i} w_ij · exp(s_ij/τ)) ]

数值稳定性:
  手写 weighted-log-sum-exp, 减去 row_max 后再 exp, 避免 FP16 溢出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class LossOutput:
    """结构化的 loss 输出, 便于 logging 和调试"""
    loss: torch.Tensor
    pos_sim_mean: float         # 正样本对平均相似度
    neg_sim_mean: float         # 负样本平均相似度
    fn_ratio: float             # 被 denoised 的负样本比例
    effective_neg_count: float  # 有效负样本数 (去噪后)
    temperature: float          # 当前温度值


class DenoisedFinancialInfoNCE(nn.Module):
    """
    三层去噪 InfoNCE, 专为金融文档检索设计。

    Args:
        tau_init: 初始温度 (默认 0.02)
        tau_self: 自蒸馏温度 (默认 0.1, 需高于 tau)
        learn_tau: 温度是否可学习
        lambda_meta: 元数据去噪强度
        mu: 自蒸馏激活阈值
        w_min: 最小负样本权重
    """

    def __init__(
        self,
        tau_init: float = 0.02,
        tau_self: float = 0.1,
        learn_tau: bool = True,
        lambda_meta: float = 0.7,
        mu: float = 0.5,
        w_min: float = 0.05,
    ):
        super().__init__()

        # 学习 log(1/τ), 保证 τ > 0
        if learn_tau:
            self.log_inv_tau = nn.Parameter(torch.tensor(1.0 / tau_init).log())
        else:
            self.register_buffer("log_inv_tau", torch.tensor(1.0 / tau_init).log())

        self.tau_self = tau_self
        self.lambda_meta = lambda_meta
        self.mu = mu
        self.w_min = w_min

    @property
    def inv_tau(self) -> torch.Tensor:
        """1/τ, clamp 到安全范围"""
        return self.log_inv_tau.exp().clamp(min=1.0, max=200.0)

    @property
    def temperature(self) -> float:
        """当前温度值 (用于 logging)"""
        return (1.0 / self.inv_tau).item()

    def compute_fn_risk_matrix(
        self,
        meta_q: list[dict],
        meta_d: list[dict],
    ) -> torch.Tensor:
        """
        Layer 1: 基于元数据构建 FN 风险矩阵 R ∈ [0,1]^{B×B}

        风险规则:
          1.0: 同公司 + 同报表类型 + 相邻周期 (≤1年)
          0.5: 同公司 + 同报表大类 / 同公司跨周期 >1年
          0.3: 同行业 + 同报表类型
          0.0: 无关

        Args:
            meta_q: 每个 query 的元数据 dict 列表
            meta_d: 每个 doc 的元数据 dict 列表

        Returns:
            R: [B, B] FN 风险矩阵
        """
        B = len(meta_q)
        R = torch.zeros(B, B)

        for i in range(B):
            for j in range(B):
                if i == j:
                    continue

                mq, md = meta_q[i], meta_d[j]
                risk = 0.0

                same_ticker = mq.get("ticker") == md.get("ticker")
                same_industry = mq.get("industry") == md.get("industry")
                same_content = mq.get("content_type") == md.get("content_type")
                same_report = mq.get("report_type") == md.get("report_type")

                if same_ticker and same_content:
                    period_diff = self._period_distance(
                        mq.get("fiscal_year", 0), mq.get("quarter", "Q1"),
                        md.get("fiscal_year", 0), md.get("quarter", "Q1"),
                    )
                    if period_diff <= 4:
                        risk = max(risk, 1.0)
                    elif period_diff <= 8:
                        risk = max(risk, 0.5)

                elif same_ticker and same_report:
                    risk = max(risk, 0.5)

                elif same_industry and same_report:
                    risk = max(risk, 0.3)

                R[i, j] = risk

        return R

    @staticmethod
    def _period_distance(year1: int, q1: str, year2: int, q2: str) -> int:
        """计算两个财务周期之间的季度距离"""
        def to_ordinal(year: int, q: str) -> int:
            if not isinstance(q, str):
                qnum = 1
            elif q.upper().startswith("Q") and len(q) >= 2 and q[1].isdigit():
                qnum = int(q[1])
            elif "ANNUAL" in q.upper():
                qnum = 4
            elif "INTERIM" in q.upper():
                qnum = 2
            else:
                qnum = 1
            return year * 4 + qnum
        return abs(to_ordinal(year1, q1) - to_ordinal(year2, q2))

    def forward(
        self,
        scores: torch.Tensor,                         # [B, B] 全对相似度矩阵
        meta_q: Optional[list[dict]] = None,           # query 元数据
        meta_d: Optional[list[dict]] = None,           # doc 元数据
        teacher_scores: Optional[torch.Tensor] = None, # [B, B] EMA Teacher 的分数
        lambda_self: float = 0.0,                      # 当前课程调度的 λ_self
    ) -> LossOutput:
        """
        Args:
            scores: [B, B] Student 的 all-pairs score matrix
            teacher_scores: [B, B] EMA Teacher 的 all-pairs score matrix (detached)
            lambda_self: 当前课程阶段的自蒸馏强度
        """
        B = scores.shape[0]
        device = scores.device

        # ======== Layer 1: 元数据 FN 掩码 ========
        if meta_q is not None and meta_d is not None:
            R = self.compute_fn_risk_matrix(meta_q, meta_d).to(device)
        else:
            R = torch.zeros(B, B, device=device)

        # ======== Layer 2: EMA Teacher 自蒸馏 ========
        self_fn_signal = torch.zeros(B, B, device=device)
        if teacher_scores is not None and lambda_self > 0:
            with torch.no_grad():
                teacher_dist = F.softmax(teacher_scores / self.tau_self, dim=-1)
                self_fn_signal = torch.clamp(teacher_dist - self.mu, min=0.0)

        # ======== Layer 3: 去噪权重 ========
        w = 1.0 - self.lambda_meta * R - lambda_self * self_fn_signal
        w = w.clamp(min=self.w_min)

        # 正样本位的权重清零 (不参与负样本求和)
        diag_mask = torch.eye(B, device=device, dtype=torch.bool)
        w = w.masked_fill(diag_mask, 0.0)

        # ======== 数值稳定的去噪 InfoNCE ========
        # 1. 计算 Logits
        logits = scores * self.inv_tau  # [B, B]

        # 2. EMERGENCY: 强制数值裁剪，防止 Exp(50+) 导致的数值溢出
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        # 3. Log-Sum-Exp Trick: 减去每行最大值保证 exp(x) <= 1
        # 注意：这里取所有样本的最大值进行平移
        row_max = logits.max(dim=-1, keepdim=True).values.detach()
        logits_stable = logits - row_max  # [B, B]
        exp_matrix = logits_stable.exp()

        # 4. 提取对角线正样本
        pos_logits_stable = logits_stable.diag()  # [B]
        pos_exp = pos_logits_stable.exp()

        # 5. 加权负样本求和
        # w 已经包含了 diag_mask=0 的逻辑
        weighted_neg_sum = (w * exp_matrix).sum(dim=-1)  # [B]

        # 6. 计算 Log-Denominator (Epsilon 保护防止 log(0))
        # Denominator = exp(pos) + sum(w * exp(neg))
        denominator = pos_exp + weighted_neg_sum + 1e-8
        log_denominator = denominator.log()

        # 7. Final Loss: -log(pos/denom) = -(pos_logit_stable) + log_denom
        loss_per_sample = -pos_logits_stable + log_denominator
        loss = loss_per_sample.mean()

        # ======== 诊断指标 (不参与梯度) ========
        with torch.no_grad():
            pos_sim = scores.diag().mean().item()
            neg_mask_float = 1.0 - torch.eye(B, device=device)
            neg_sim = (scores * neg_mask_float).sum() / neg_mask_float.sum()
            neg_sim = neg_sim.item()
            fn_count = ((w < 0.5) & ~diag_mask).float().sum()
            total_neg = B * (B - 1)
            fn_ratio = (fn_count / max(total_neg, 1)).item()
            eff_neg = w[~diag_mask].sum().item() / B

        return LossOutput(
            loss=loss,
            pos_sim_mean=pos_sim,
            neg_sim_mean=neg_sim,
            fn_ratio=fn_ratio,
            effective_neg_count=eff_neg,
            temperature=self.temperature,
        )
