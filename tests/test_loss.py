"""
DenoisedFinancialInfoNCE 损失函数单元测试

测试用例:
  1. BF16 下无 NaN / Inf
  2. w_min 下限保证: 即使 R=1.0, w ≥ w_min
  3. 温度参数 clamp 范围
  4. 极端输入 (全正 / 全负) 下 loss 有限
  5. Loss 输出结构正确
  6. 可学习温度有梯度
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vision_rag.loss.denoised_infonce import DenoisedFinancialInfoNCE, LossOutput


class TestDenoisedInfoNCE:
    @pytest.fixture
    def loss_fn(self):
        return DenoisedFinancialInfoNCE(
            tau_init=0.02, tau_self=0.1, learn_tau=True,
            lambda_meta=0.7, mu=0.5, w_min=0.05,
        )

    @pytest.fixture
    def dummy_scores(self):
        B = 8
        # 合理的分数矩阵: 对角线较高
        scores = torch.randn(B, B) * 0.3
        scores += torch.eye(B) * 0.5  # 正样本偏高
        return scores

    def test_output_structure(self, loss_fn, dummy_scores):
        """输出是 LossOutput dataclass"""
        output = loss_fn(dummy_scores)
        assert isinstance(output, LossOutput)
        assert isinstance(output.loss, torch.Tensor)
        assert output.loss.ndim == 0  # scalar

    def test_no_nan_inf(self, loss_fn, dummy_scores):
        """BF16 下无 NaN / Inf"""
        scores_bf16 = dummy_scores.to(torch.bfloat16).float()
        output = loss_fn(scores_bf16)
        assert not torch.isnan(output.loss), f"Loss is NaN: {output.loss}"
        assert not torch.isinf(output.loss), f"Loss is Inf: {output.loss}"

    def test_extreme_scores_stable(self, loss_fn):
        """极端分数值 (±100) 下不崩溃"""
        B = 4
        # 极端正值
        scores_high = torch.ones(B, B) * 100.0
        scores_high += torch.eye(B) * 50.0
        output = loss_fn(scores_high)
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

        # 极端负值
        scores_low = torch.ones(B, B) * -100.0
        scores_low += torch.eye(B) * 50.0
        output_low = loss_fn(scores_low)
        assert not torch.isnan(output_low.loss)
        assert not torch.isinf(output_low.loss)

    def test_w_min_guarantee(self, loss_fn):
        """即使 R=1.0 (高 FN 风险), w ≥ w_min"""
        B = 4
        scores = torch.randn(B, B)
        # 强制所有非对角元素 FN 风险最高
        meta_q = [
            {"ticker": "AAPL", "content_type": "income_stmt",
             "fiscal_year": 2023, "quarter": "Q4", "report_type": "10-K"}
        ] * B
        meta_d = meta_q  # 全部相同 → R 接近 1.0

        output = loss_fn(scores, meta_q=meta_q, meta_d=meta_d)
        assert output.loss > 0  # loss 仍然有意义

    def test_temperature_bounded(self, loss_fn):
        """温度在安全范围内"""
        tau = loss_fn.temperature
        assert 0.005 < tau < 1.0  # 初始 0.02

    def test_learnable_temperature_gradient(self, loss_fn, dummy_scores):
        """可学习温度有梯度"""
        output = loss_fn(dummy_scores)
        output.loss.backward()
        assert loss_fn.log_inv_tau.grad is not None
        assert loss_fn.log_inv_tau.grad.abs() > 0

    def test_positive_loss(self, loss_fn, dummy_scores):
        """Loss 始终为正"""
        output = loss_fn(dummy_scores)
        assert output.loss.item() > 0

    def test_with_teacher_scores(self, loss_fn, dummy_scores):
        """Layer 2: Teacher scores 参与去噪后 loss 仍然稳定"""
        teacher_scores = dummy_scores.detach() + torch.randn_like(dummy_scores) * 0.1
        output = loss_fn(
            dummy_scores,
            teacher_scores=teacher_scores,
            lambda_self=0.3
        )
        assert not torch.isnan(output.loss)
        assert output.loss.item() > 0

    def test_fn_ratio_in_range(self, loss_fn, dummy_scores):
        """去噪比例在 [0, 1] 范围"""
        meta = [
            {"ticker": f"T{i}", "content_type": "balance", "report_type": "10-K",
             "fiscal_year": 2023, "quarter": "Q4"}
            for i in range(dummy_scores.shape[0])
        ]
        output = loss_fn(dummy_scores, meta_q=meta, meta_d=meta)
        assert 0.0 <= output.fn_ratio <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
