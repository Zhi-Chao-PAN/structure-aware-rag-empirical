"""
配置管理 (Configuration)

支持 YAML 配置文件加载, 提供类型安全的配置 dataclass。
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """模型架构配置"""
    backbone: str = "vidore/colpali-v1.2"      # VLM backbone (HuggingFace ID)
    projection_dim: int = 128                   # ColBERT 投影维度 D
    img_size: int = 1024                        # ViT 输入尺寸 (正方形)
    patch_size: int = 14                        # ViT patch 尺寸
    has_cls: bool = True                        # 是否有 [CLS] token
    gamma: float = 0.1                          # 凸组合数值偏置 γ
    gate_init_bias: float = -2.0                # Query gate 初始偏置
    freeze_backbone_epochs: int = 2             # backbone 冻结的 epoch 数


@dataclass
class LossConfig:
    """损失函数配置"""
    tau_init: float = 0.02                      # InfoNCE 初始温度
    tau_self: float = 0.1                       # 自蒸馏温度
    learn_tau: bool = True                      # 温度是否可学习
    lambda_meta: float = 0.7                    # 元数据去噪强度
    lambda_self_max: float = 0.3                # 自蒸馏去噪最大强度
    mu: float = 0.5                             # 自蒸馏激活阈值
    w_min: float = 0.05                         # 最小负样本权重 (防坍塌)


@dataclass
class EMAConfig:
    """EMA 教师网络配置"""
    momentum_init: float = 0.999                # 初始 EMA 动量
    momentum_final: float = 0.9995              # 最终 EMA 动量
    warmup_epochs: int = 5                      # EMA 动量 warmup epoch 数


@dataclass
class CurriculumConfig:
    """课程学习配置"""
    lambda_self_warmup_steps: int = 1000        # λ_self 从 0 增长到 max 的步数
    easy_epochs: int = 3                        # Easy Mode 持续的 epoch 数
    hard_epoch_start: int = 7                   # Adversarial Mode 开始的 epoch
    n_industries_easy: int = 8                  # Easy 阶段每 batch 行业数
    n_industries_balanced: int = 4              # Balanced 阶段
    n_industries_hard: int = 2                  # Hard 阶段


@dataclass
class DataConfig:
    """数据管线配置"""
    batch_size: int = 32                        # Batch 大小
    num_workers: int = 4                        # DataLoader worker 数
    pdf_dir: str = "data/raw_pdfs"              # 原始 PDF 目录
    page_image_dir: str = "data/vision_rag/page_images"
    mask_cache_dir: str = "data/vision_rag/spatial_masks"
    metadata_dir: str = "data/vision_rag/metadata"
    bm25_index_path: str = "data/vision_rag/bm25_index"
    visfinqa_dir: str = "data/vision_rag/visfinqa"
    bm25_top_k: int = 50                        # BM25 Hard Neg 挖掘数


@dataclass
class TrainConfig:
    """训练配置"""
    num_epochs: int = 15
    lr: float = 2e-5                            # 学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1                   # LR warmup 比例
    max_grad_norm: float = 1.0                  # 梯度裁剪
    mixed_precision: str = "bf16"               # 混合精度: "bf16", "fp16", "no"
    log_interval: int = 10                      # 每 N 步 log
    eval_interval_epochs: int = 1               # 每 N epoch 评测
    checkpoint_dir: str = "checkpoints/vision_rag"
    wandb_project: str = "vision-rag-financial"
    seed: int = 42


@dataclass
class EvalConfig:
    """评测配置"""
    top_k_pages: list[int] = field(default_factory=lambda: [1, 5, 10])
    top_k_patches: int = 10                     # VGS-Hit 的 top-K
    vgs_iou_threshold: float = 0.1              # IoPA > 此值视为 GT 活跃 patch
    grs_beta: float = 0.5                       # GRS 平衡因子


@dataclass
class VisionRAGConfig:
    """顶层配置 (聚合所有子配置)"""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(config_path: str | Path) -> VisionRAGConfig:
    """
    从 YAML 文件加载并合并配置。

    YAML 文件中未指定的字段使用 dataclass 默认值。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        VisionRAGConfig 实例
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    config = VisionRAGConfig()

    # 递归合并 YAML 值到 dataclass
    sub_configs = {
        'model': (config.model, ModelConfig),
        'loss': (config.loss, LossConfig),
        'ema': (config.ema, EMAConfig),
        'curriculum': (config.curriculum, CurriculumConfig),
        'data': (config.data, DataConfig),
        'train': (config.train, TrainConfig),
        'eval': (config.eval, EvalConfig),
    }

    for key, (sub_cfg, _cls) in sub_configs.items():
        if key in raw and isinstance(raw[key], dict):
            for field_name, value in raw[key].items():
                if hasattr(sub_cfg, field_name):
                    setattr(sub_cfg, field_name, value)

    # 环境变量 override: VRAG_OVERRIDE_MODEL_GAMMA=0.5 → config.model.gamma = 0.5
    import os
    prefix = "VRAG_OVERRIDE_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field_name = parts[0], parts[1]
        if section in sub_configs:
            sub_cfg, _ = sub_configs[section]
            if hasattr(sub_cfg, field_name):
                current = getattr(sub_cfg, field_name)
                try:
                    if isinstance(current, bool):
                        casted = env_val.lower() in ("true", "1", "yes")
                    elif isinstance(current, int):
                        casted = int(float(env_val))
                    elif isinstance(current, float):
                        casted = float(env_val)
                    else:
                        casted = env_val
                    setattr(sub_cfg, field_name, casted)
                    print(f"[CONFIG] Override: {section}.{field_name} = {casted}")
                except (ValueError, TypeError):
                    pass

    return config
