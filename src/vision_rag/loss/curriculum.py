"""
课程学习调度器 (Curriculum Scheduler)

控制三个维度的训练难度递进:
  1. λ_self: 自蒸馏去噪强度 (0 → max, warmup 步数后释放)
  2. Batch 对抗烈度: 通过行业数衰减控制 (由 Sampler 接收)
  3. EMA 动量: 从 init 线性增长到 final (由 EMATeacher 内部处理)

时间线:
  Epoch 0~2 (Easy):    λ_self=0, 8 industries/batch
  Epoch 3~6 (Balanced): λ_self warmup, 4 industries/batch
  Epoch 7+  (Hard):    λ_self=max, 2 industries/batch
"""


class CurriculumScheduler:
    """
    课程学习统一调度器。

    在每个训练步/epoch 时调用, 返回当前的课程参数。
    不拥有模型或优化器的引用, 仅提供参数值。

    Args:
        lambda_self_max: λ_self 的最大值
        lambda_self_warmup_steps: λ_self 从 0 增长到 max 的步数
        easy_epochs: Easy Mode 持续的 epoch 数
        hard_epoch_start: Adversarial Mode 开始的 epoch
        n_industries_easy: Easy 阶段每 batch 行业数
        n_industries_balanced: Balanced 阶段
        n_industries_hard: Hard 阶段
    """

    def __init__(
        self,
        lambda_self_max: float = 0.3,
        lambda_self_warmup_steps: int = 1000,
        easy_epochs: int = 3,
        hard_epoch_start: int = 7,
        n_industries_easy: int = 8,
        n_industries_balanced: int = 4,
        n_industries_hard: int = 2,
    ):
        self.lambda_self_max = lambda_self_max
        self.lambda_self_warmup_steps = lambda_self_warmup_steps
        self.easy_epochs = easy_epochs
        self.hard_epoch_start = hard_epoch_start
        self.n_industries_easy = n_industries_easy
        self.n_industries_balanced = n_industries_balanced
        self.n_industries_hard = n_industries_hard

        self._global_step = 0
        self._current_epoch = 0

    def step(self):
        """每个训练步调用 (在 optimizer.step() 之后)"""
        self._global_step += 1

    def set_epoch(self, epoch: int):
        """每个 epoch 开始时调用"""
        self._current_epoch = epoch

    @property
    def lambda_self(self) -> float:
        """
        当前自蒸馏去噪强度。

        训练初期 (epoch < easy_epochs): 强制为 0, 不允许"瞎子"教自己。
        之后线性 warmup 到 lambda_self_max。
        """
        if self._current_epoch < self.easy_epochs:
            return 0.0

        progress = min(1.0, self._global_step / max(1, self.lambda_self_warmup_steps))
        return self.lambda_self_max * progress

    @property
    def n_industries(self) -> int:
        """当前 batch 应使用的行业数 (传给 Sampler)"""
        if self._current_epoch < self.easy_epochs:
            return self.n_industries_easy
        elif self._current_epoch < self.hard_epoch_start:
            return self.n_industries_balanced
        else:
            return self.n_industries_hard

    @property
    def phase_name(self) -> str:
        """当前课程阶段名称 (用于 logging)"""
        if self._current_epoch < self.easy_epochs:
            return "easy"
        elif self._current_epoch < self.hard_epoch_start:
            return "balanced"
        else:
            return "adversarial"

    def state_dict(self) -> dict:
        return {
            "global_step": self._global_step,
            "current_epoch": self._current_epoch,
        }

    def load_state_dict(self, state: dict):
        self._global_step = state["global_step"]
        self._current_epoch = state["current_epoch"]
