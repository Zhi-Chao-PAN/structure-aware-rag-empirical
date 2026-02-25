"""
EMA 教师网络 (Exponential Moving Average Teacher)

核心职责:
  维护模型权重的指数滑动平均副本, 用于:
  1. Layer 2 自蒸馏: 提供平滑、低方差的软标签
  2. 避免确认偏误: Teacher 不参与梯度更新, 天然抗噪

更新规则:
  θ_teacher = m · θ_teacher + (1 - m) · θ_student
  其中 m 随训练从 momentum_init 线性增长到 momentum_final

依赖: 仅依赖 torch, 无外部框架
"""

import torch
import torch.nn as nn
import copy
from typing import Optional


class EMATeacher:
    """
    指数滑动平均教师网络。

    使用方式:
        teacher = EMATeacher(student_model, momentum_init=0.999)
        # 在每个训练步后:
        teacher.update()
        # 前向推理:
        with torch.no_grad():
            teacher_output = teacher.model(input)

    Args:
        student_model: 学生模型 (参与梯度更新的模型)
        momentum_init: 初始 EMA 动量 (越接近 1 越平滑)
        momentum_final: 最终 EMA 动量
        warmup_steps: 动量从 init 增长到 final 的步数
    """

    def __init__(
        self,
        student_model: nn.Module,
        momentum_init: float = 0.999,
        momentum_final: float = 0.9995,
        warmup_steps: int = 5000,
    ):
        # 深拷贝学生模型作为教师
        self.model = copy.deepcopy(student_model)

        # 冻结教师模型: 不参与梯度计算
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # 保持学生模型引用 (用于 update)
        self._student = student_model

    @property
    def momentum(self) -> float:
        """
        当前 EMA 动量, 随训练步数线性增长。

        训练初期 (小 momentum): Teacher 快速跟踪 Student → 减少延迟
        训练后期 (大 momentum): Teacher 高度平滑 → 稳定的软标签
        """
        if self.step_count >= self.warmup_steps:
            return self.momentum_final

        progress = self.step_count / self.warmup_steps
        return self.momentum_init + (self.momentum_final - self.momentum_init) * progress

    @torch.no_grad()
    def update(self):
        """
        执行一步 EMA 更新: θ_t = m · θ_t + (1-m) · θ_s

        必须在 optimizer.step() 之后调用。
        """
        m = self.momentum
        for t_param, s_param in zip(
            self.model.parameters(), self._student.parameters()
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        # Buffer 也需要同步 (比如 BatchNorm 的 running_mean/var)
        for t_buf, s_buf in zip(
            self.model.buffers(), self._student.buffers()
        ):
            t_buf.data.copy_(s_buf.data)

        self.step_count += 1

    def state_dict(self) -> dict:
        """保存 EMA 状态 (用于 checkpoint)"""
        return {
            "model_state": self.model.state_dict(),
            "step_count": self.step_count,
            "momentum_init": self.momentum_init,
            "momentum_final": self.momentum_final,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict):
        """恢复 EMA 状态"""
        self.model.load_state_dict(state["model_state"])
        self.step_count = state["step_count"]
        self.momentum_init = state.get("momentum_init", self.momentum_init)
        self.momentum_final = state.get("momentum_final", self.momentum_final)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
