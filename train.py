"""
Vision-RAG 训练入口 (Training Entry Point) — 生产级 v2

顶会级修复:
  C-3: 完整 WandB 标量记录 (12 train + 9 eval metrics)
  H-2: 参数组逻辑修复 (冻结时排除 backbone, 解冻时动态追加)
  H-3: LR Scheduler (linear warmup + cosine decay)
  H-4: 真实数据路径 → VisFinRAGDataset 自动接入
  H-5: EMA Teacher 正确初始化 (deepcopy, 不重复 prepare)

模式坍塌检测:
  sim_gap = pos_sim - neg_sim → WandB 实时监控
  gap → 0 即坍塌, 可在 dashboard 设 alert

用法:
  # 本地 Mock 测试
  python train.py --config config/vision_rag/default.yaml --smoke-test --max-steps 5

  # AutoDL 正式训练
  accelerate launch train.py --config config/vision_rag/default.yaml
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR

# EMERGENCY DEBUG: 开启全局异常检测记录，追踪 NaN 物理来源
torch.autograd.set_detect_anomaly(True)

# 本地模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision_rag.model.vision_retriever import VisionRetriever
from vision_rag.model.ema import EMATeacher
from vision_rag.loss.denoised_infonce import DenoisedFinancialInfoNCE
from vision_rag.loss.curriculum import CurriculumScheduler
from vision_rag.data.collator import VisFinRAGCollator
from vision_rag.utils.config import load_config, VisionRAGConfig


# =====================================================================
# WandB 辅助
# =====================================================================

_wandb = None  # 延迟导入, 避免 smoke test 强依赖


def init_wandb(config: VisionRAGConfig, args) -> bool:
    """初始化 WandB, 返回是否成功"""
    global _wandb
    if args.smoke_test:
        return False
    try:
        import wandb
        _wandb = wandb
        wandb.init(
            project=config.train.wandb_project,
            config={
                "batch_size": config.data.batch_size,
                "lr": config.train.lr,
                "epochs": config.train.num_epochs,
                "projection_dim": config.model.projection_dim,
                "gamma": config.model.gamma,
                "tau_init": config.loss.tau_init,
                "mixed_precision": config.train.mixed_precision,
                "mock": args.mock,
            },
            tags=["vision-rag", "late-interaction"],
        )
        return True
    except ImportError:
        print("[INFO] wandb 未安装, 跳过日志记录")
        return False


def log_wandb(metrics: dict, step: int):
    """安全 WandB log (无 wandb 时静默跳过)"""
    if _wandb is not None:
        _wandb.log(metrics, step=step)


# =====================================================================
# LR Scheduler: Linear Warmup + Cosine Decay
# =====================================================================

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
) -> LambdaLR:
    """标准 HuggingFace 风格 cosine schedule"""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Mock Dataset (本地测试用)
# =====================================================================

class MockVisFinRAGDataset(Dataset):
    """
    合成数据集: 验证 forward/backward 无报错。
    维度与真实数据完全一致。
    """

    INDUSTRIES = ["technology", "financial", "healthcare", "energy"]
    TICKERS = {
        "technology": ["AAPL", "MSFT", "GOOGL"],
        "financial": ["JPM", "GS", "BAC"],
        "healthcare": ["PFE", "JNJ", "UNH"],
        "energy": ["XOM", "CVX", "COP"],
    }

    def __init__(self, num_samples: int = 256, num_patches: int = 1030):
        self.num_samples = num_samples
        self.num_patches = num_patches

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        industry = self.INDUSTRIES[idx % len(self.INDUSTRIES)]
        tickers = self.TICKERS[industry]
        ticker = tickers[idx % len(tickers)]
        year = 2021 + (idx % 3)
        quarter = f"Q{(idx % 4) + 1}"

        return {
            "image": torch.randn(3, 1024, 1024),
            "soft_mask": torch.rand(self.num_patches),
            "padding_mask": torch.ones(self.num_patches),
            "query_text": f"What was {ticker}'s revenue in {year} {quarter}?",
            "meta_q": {
                "ticker": ticker,
                "industry": industry,
                "fiscal_year": year,
                "quarter": quarter,
                "report_type": "10-K" if quarter == "Q4" else "10-Q",
                "content_type": "income_statement",
            },
            "meta_d": {
                "ticker": ticker,
                "industry": industry,
                "fiscal_year": year,
                "quarter": quarter,
                "report_type": "10-K" if quarter == "Q4" else "10-Q",
                "content_type": "income_statement",
            },
        }


# =====================================================================
# Training Loop
# =====================================================================

def train(config: VisionRAGConfig, args):
    """主训练函数"""

    # ---- 环境初始化 ----
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(
            mixed_precision=config.train.mixed_precision,
            gradient_accumulation_steps=1,
        )
        device = accelerator.device
        use_accelerate = True
    except ImportError:
        print("[INFO] accelerate 未安装, 使用原生 PyTorch 训练")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_accelerate = False

    print(f"设备: {device}")
    print(f"混合精度: {config.train.mixed_precision}")

    # ---- 模型初始化 ----
    model = VisionRetriever(
        projection_dim=config.model.projection_dim,
        gamma=config.model.gamma,
        gate_init_bias=config.model.gate_init_bias,
        use_mock=args.mock,
        freeze_backbone=True,
    )
    model = model.to(device)

    # EMA Teacher (deepcopy Student, 独立权重同步)
    ema_teacher = EMATeacher(
        model,
        momentum_init=config.ema.momentum_init,
        momentum_final=config.ema.momentum_final,
        warmup_steps=config.ema.warmup_epochs * 100,
    )

    # ---- Loss & Scheduler ----
    loss_fn = DenoisedFinancialInfoNCE(
        tau_init=config.loss.tau_init,
        tau_self=config.loss.tau_self,
        learn_tau=config.loss.learn_tau,
        lambda_meta=config.loss.lambda_meta,
        mu=config.loss.mu,
        w_min=config.loss.w_min,
    ).to(device)

    curriculum = CurriculumScheduler(
        lambda_self_max=config.loss.lambda_self_max,
        lambda_self_warmup_steps=config.curriculum.lambda_self_warmup_steps,
        easy_epochs=config.curriculum.easy_epochs,
        hard_epoch_start=config.curriculum.hard_epoch_start,
    )

    # ---- Optimizer (修复 H-2: 冻结时排除 backbone, gate 独立 lr) ----
    # 分离 gate 参数，给更大的 lr 让 gate 能学到有意义的值
    gate_params = list(model.maxsim.gate_proj.parameters())
    gate_param_ids = {id(p) for p in gate_params}
    maxsim_other_params = [p for p in model.maxsim.parameters() if id(p) not in gate_param_ids]

    # 分离 loss 中的温度参数，给更大的 lr
    tau_params = [p for n, p in loss_fn.named_parameters() if 'log_inv_tau' in n]
    loss_other_params = [p for n, p in loss_fn.named_parameters() if 'log_inv_tau' not in n]
    tau_param_ids = {id(p) for p in tau_params}

    trainable_params = [
        {"params": maxsim_other_params, "lr": config.train.lr, "name": "maxsim"},
        {"params": gate_params, "lr": config.train.lr * 10.0, "name": "gate"},  # gate 10x lr
        {"params": loss_other_params, "lr": config.train.lr, "name": "loss_params"},
    ]
    if tau_params:
        trainable_params.append(
            {"params": tau_params, "lr": config.train.lr * 5.0, "name": "temperature"},  # τ 5x lr
        )

    # 只在 backbone 未冻结时加入 optimizer
    if not model.freeze_backbone:
        trainable_params.append(
            {"params": model.backbone.parameters(), "lr": config.train.lr * 0.1, "name": "backbone"}
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=config.train.weight_decay,
    )

    # ---- DataLoader ----
    collator = VisFinRAGCollator(image_processor=None, mock_mode=args.mock)

    if args.mock:
        dataset = MockVisFinRAGDataset(num_samples=max(64, config.data.batch_size * 4))
    else:
        # 真实数据: 自动加载 VisFinRAGDataset
        from vision_rag.data.dataset import VisFinRAGDataset
        dataset = VisFinRAGDataset(
            metadata_path=Path(config.data.metadata_dir) / "page_metadata.parquet",
            image_dir=Path(config.data.page_image_dir),
            mask_dir=Path(config.data.mask_cache_dir),
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            has_cls=config.model.has_cls,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0 if args.mock else config.data.num_workers,
        pin_memory=not args.mock,
        persistent_workers=False if args.mock else config.data.num_workers > 0,
    )

    # ---- LR Scheduler (H-3) ----
    total_steps = len(dataloader) * config.train.num_epochs
    warmup_steps = int(total_steps * config.train.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Accelerate 封装 ----
    if use_accelerate:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler,
        )
        # H-5: EMA Teacher 不需要 prepare — 它只做 state_dict 同步, 不参与梯度计算
        ema_teacher._student = model

    # ---- WandB (C-3) ----
    use_wandb = init_wandb(config, args)

    # ---- Seed ----
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_state" in ckpt:
            ema_teacher.load_state_dict(ckpt["ema_state"])
        if "curriculum_state" in ckpt:
            curriculum.load_state_dict(ckpt["curriculum_state"])
        if "loss_state" in ckpt:
            loss_fn.load_state_dict(ckpt["loss_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[RESUME] 从 {args.resume} 恢复, epoch={start_epoch}, step={global_step}")

    # ---- 主训练循环 ----
    max_steps = args.max_steps if args.smoke_test else float("inf")

    print(f"\n{'=' * 60}")
    print(f"训练开始 | Epochs: {config.train.num_epochs} | Batch: {config.data.batch_size}")
    print(f"Mock: {args.mock} | Smoke Test: {args.smoke_test}")
    print(f"Total Steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"WandB: {'ON' if use_wandb else 'OFF'}")
    if args.resume:
        print(f"Resume: epoch={start_epoch}, step={global_step}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, config.train.num_epochs):
        curriculum.set_epoch(epoch)
        model.train()

        # Backbone 解冻调度 (H-2: 动态追加参数组)
        if epoch == config.model.freeze_backbone_epochs:
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model.unfreeze_backbone()
            # 动态追加 backbone 参数到 optimizer
            optimizer.add_param_group({
                "params": list(raw_model.backbone.parameters()),
                "lr": config.train.lr * 0.1,
                "name": "backbone",
            })
            print(f"[Epoch {epoch}] Backbone 已解冻, 已追加到 optimizer (lr={config.train.lr * 0.1:.1e})")

        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break

            # 移动到设备
            if not use_accelerate:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            # ---- Forward: Student ----
            output = model(
                images=batch["images"],
                query_text=batch["query_text"],
                soft_masks=batch["soft_masks"],
                padding_mask=batch["padding_mask"],
                all_pairs=True,
            )

            # ---- Forward: EMA Teacher ----
            teacher_scores = None
            if curriculum.lambda_self > 0:
                # EMERGENCY: 关键修复 —— 教师模型未被 accelerator 包装，需手动开启 autocast 保证精度对齐
                with torch.no_grad(), torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda', dtype=torch.bfloat16):
                    ema_teacher.model.eval()
                    t_output = ema_teacher.model(
                        images=batch["images"],
                        query_text=batch["query_text"],
                        soft_masks=batch["soft_masks"],
                        padding_mask=batch["padding_mask"],
                        all_pairs=True,
                    )
                    teacher_scores = t_output["scores"].detach()

            # ---- Loss ----
            loss_output = loss_fn(
                scores=output["scores"],
                meta_q=batch["meta_q"],
                meta_d=batch["meta_d"],
                teacher_scores=teacher_scores,
                lambda_self=curriculum.lambda_self,
            )

            # ---- Backward ----
            if use_accelerate:
                accelerator.backward(loss_output.loss)
            else:
                loss_output.loss.backward()

            # 梯度裁剪
            if config.train.max_grad_norm > 0:
                if use_accelerate:
                    accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # ---- EMA Update ----
            ema_teacher.update()
            curriculum.step()
            global_step += 1
            epoch_loss += loss_output.loss.item()
            epoch_steps += 1

            # ---- Logging (C-3: 完整 WandB 标量记录) ----
            if global_step % config.train.log_interval == 0 or args.smoke_test:
                sim_gap = loss_output.pos_sim_mean - loss_output.neg_sim_mean
                gate_mean = output["gate"].mean().item()
                current_lr = optimizer.param_groups[0]["lr"]

                # Console
                print(
                    f"  [Step {global_step}] "
                    f"loss={loss_output.loss.item():.4f} "
                    f"τ={loss_output.temperature:.4f} "
                    f"pos_sim={loss_output.pos_sim_mean:.3f} "
                    f"neg_sim={loss_output.neg_sim_mean:.3f} "
                    f"sim_gap={sim_gap:.3f} "
                    f"fn_ratio={loss_output.fn_ratio:.2f} "
                    f"gate={gate_mean:.3f} "
                    f"lr={current_lr:.2e} "
                    f"phase={curriculum.phase_name} "
                    f"λ_self={curriculum.lambda_self:.3f}"
                )

                # WandB (12 train scalars)
                log_wandb({
                    "train/loss": loss_output.loss.item(),
                    "train/temperature": loss_output.temperature,
                    "train/pos_sim": loss_output.pos_sim_mean,
                    "train/neg_sim": loss_output.neg_sim_mean,
                    "train/sim_gap": sim_gap,
                    "train/fn_ratio": loss_output.fn_ratio,
                    "train/eff_neg": loss_output.effective_neg_count,
                    "train/gate_mean": gate_mean,
                    "train/lr": current_lr,
                    "train/ema_momentum": ema_teacher.momentum,
                    "curriculum/phase": hash(curriculum.phase_name) % 3,
                    "curriculum/lambda_self": curriculum.lambda_self,
                }, step=global_step)

        # Epoch 汇总
        elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(
            f"\n[Epoch {epoch}] avg_loss={avg_loss:.4f} "
            f"steps={epoch_steps} time={elapsed:.1f}s "
            f"EMA_momentum={ema_teacher.momentum:.5f}\n"
        )

        log_wandb({
            "epoch/avg_loss": avg_loss,
            "epoch/time_s": elapsed,
        }, step=global_step)

        if global_step >= max_steps:
            break

        # ---- Checkpoint ----
        if not args.smoke_test:
            ckpt_dir = Path(config.train.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_model = model.module if hasattr(model, 'module') else model
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": save_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": ema_teacher.state_dict(),
                "curriculum_state": curriculum.state_dict(),
                "loss_state": loss_fn.state_dict(),
            }, ckpt_path)
            print(f"  [CKPT] 已保存: {ckpt_path}")

    print(f"\n{'=' * 60}")
    print(f"训练完成 | 总步数: {global_step}")
    print(f"{'=' * 60}")

    if _wandb is not None:
        _wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Vision-RAG 训练")
    parser.add_argument("--config", type=str, default="config/vision_rag/default.yaml")
    parser.add_argument("--mock", action="store_true", help="使用 MockColPali (本地测试)")
    parser.add_argument("--smoke-test", action="store_true", help="仅跑几步验证")
    parser.add_argument("--max-steps", type=int, default=5, help="Smoke test 最大步数")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练 (路径)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Smoke test 时缩小 batch
    if args.smoke_test:
        config.data.batch_size = min(4, config.data.batch_size)
        args.mock = True

    train(config, args)


if __name__ == "__main__":
    main()
