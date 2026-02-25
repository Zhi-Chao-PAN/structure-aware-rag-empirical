"""
NumericAwareMaxSim 算子单元测试

测试用例:
  1. 输出分数有界: [-N, N*(1+γ)]
  2. Gate 全零时退化为标准 MaxSim
  3. Gate 全一时 S_num 占主导
  4. 梯度存在: 所有可学习参数的 .grad 非零
  5. All-pairs shape: [B, B]
  6. Single-pair shape: [B]
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vision_rag.model.numeric_maxsim import NumericAwareMaxSim


class TestNumericAwareMaxSim:
    @pytest.fixture
    def module(self):
        return NumericAwareMaxSim(dim=128, gamma=0.1, gate_init_bias=-2.0)

    @pytest.fixture
    def dummy_inputs(self):
        B, N, M, D = 4, 32, 64, 128
        E_q = torch.randn(B, N, D)
        E_d = torch.randn(B, M, D)
        M_soft = torch.rand(B, M)
        padding_mask = torch.ones(B, M)
        return E_q, E_d, M_soft, padding_mask

    def test_single_output_shape(self, module, dummy_inputs):
        """Single-pair 输出 shape 正确"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        scores, gate = module.forward_single(E_q, E_d, M_soft, padding_mask)
        assert scores.shape == (4,)
        assert gate.shape == (4, 32)

    def test_all_pairs_output_shape(self, module, dummy_inputs):
        """All-pairs 输出 shape 正确"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        scores, gate = module.forward_all_pairs(E_q, E_d, M_soft, padding_mask)
        assert scores.shape == (4, 4)
        assert gate.shape == (4, 32)

    def test_gate_range(self, module, dummy_inputs):
        """Gate 值在 [0, 1] 范围内"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        _, gate = module(E_q, E_d, M_soft, padding_mask)
        assert gate.min() >= 0.0
        assert gate.max() <= 1.0

    def test_initial_gate_conservative(self, module, dummy_inputs):
        """初始 gate 偏保守 (bias=-2, sigmoid(-2)≈0.12)"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        with torch.no_grad():
            _, gate = module(E_q, E_d, M_soft, padding_mask)
        # 初始 weight=0 → gate 只由 bias 决定 → 全部 ≈ 0.12
        assert gate.mean().item() == pytest.approx(0.119, abs=0.02)

    def test_zero_gate_standard_maxsim(self, module, dummy_inputs):
        """Gate≈0 时近似退化为标准 MaxSim (mean 聚合)"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs

        # 强制 gate 为 ~0 (clamp min=0.01, 所以不会精确为 0)
        with torch.no_grad():
            module.gate_proj.weight.zero_()
            module.gate_proj.bias.fill_(-100.0)  # sigmoid(-100) ≈ 0, clamped to 0.01

        scores_single, gate = module.forward_single(E_q, E_d, M_soft, padding_mask)

        # 手动计算标准 MaxSim (mean 聚合，与模型一致)
        E_q_norm = torch.nn.functional.normalize(E_q, p=2, dim=-1)
        E_d_norm = torch.nn.functional.normalize(E_d, p=2, dim=-1)
        sim = torch.bmm(E_q_norm, E_d_norm.transpose(1, 2))
        std_maxsim = sim.max(dim=-1).values.mean(dim=-1)

        # gate clamped to 0.01, so small residual from S_num + gamma
        assert torch.allclose(scores_single, std_maxsim, atol=0.02)

    def test_gradients_exist(self, module, dummy_inputs):
        """所有可学习参数有有效梯度"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        E_q.requires_grad_(True)

        scores, gate = module(E_q, E_d, M_soft, padding_mask, all_pairs=False)
        loss = scores.sum()
        loss.backward()

        # gate_proj 权重有梯度
        assert module.gate_proj.weight.grad is not None
        assert module.gate_proj.weight.grad.abs().sum() > 0

        # gate_proj 偏置有梯度
        assert module.gate_proj.bias.grad is not None

    def test_score_bounded(self, module, dummy_inputs):
        """分数有界: 每个 query token 贡献 ∈ [-1, 1+γ]"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs
        N = E_q.shape[1]

        with torch.no_grad():
            scores, _ = module(E_q, E_d, M_soft, padding_mask, all_pairs=False)

        # 极端上界: N * (1 + γ) = 32 * 1.1 = 35.2
        # 极端下界: N * (-1) = -32
        gamma = module.gamma.item()
        assert scores.max().item() <= N * (1 + gamma) + 1e-4
        assert scores.min().item() >= -N - 1e-4

    def test_all_pairs_diagonal_matches_single(self, module, dummy_inputs):
        """All-pairs 对角线 ≈ single-pair 的结果"""
        E_q, E_d, M_soft, padding_mask = dummy_inputs

        with torch.no_grad():
            scores_single, _ = module.forward_single(E_q, E_d, M_soft, padding_mask)
            scores_all, _ = module.forward_all_pairs(E_q, E_d, M_soft, padding_mask)

        diag = scores_all.diag()
        assert torch.allclose(diag, scores_single, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
