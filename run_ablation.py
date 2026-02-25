"""
消融实验脚本 (Ablation Study Runner)

三组对比实验:
  1. Full Model:     完整 NumericAwareMaxSim + DenoisedInfoNCE + EMA
  2. No Gate:        禁用 NumericAwareMaxSim gate (gate 固定为 0)
  3. No Denoise:     标准 InfoNCE (无 FN 去噪)
  4. No EMA:         无 EMA Teacher 自蒸馏

用法:
  python run_ablation.py --config config/vision_rag/default.yaml --mock
  python run_ablation.py --config config/vision_rag/autodl_train.yaml  # AutoDL 真实训练
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


ABLATION_CONFIGS = {
    "full": {
        "desc": "Full Model (NumericAwareMaxSim + DenoisedInfoNCE + EMA)",
        "config": "config/vision_rag/default.yaml",
        "overrides": {},
    },
    "no_gate": {
        "desc": "Ablation: gate 固定为 0 (禁用 NumericAwareMaxSim)",
        "config": "config/vision_rag/ablation_no_gate.yaml",
        "overrides": {"model.gate_init_bias": -100.0},
    },
    "no_denoise": {
        "desc": "Ablation: 标准 InfoNCE (无 FN 去噪)",
        "config": "config/vision_rag/ablation_no_denoise.yaml",
        "overrides": {"loss.lambda_meta": 0.0, "loss.lambda_self_max": 0.0},
    },
    "no_ema": {
        "desc": "Ablation: 无 EMA Teacher 自蒸馏",
        "config": "config/vision_rag/ablation_no_ema.yaml",
        "overrides": {"loss.lambda_self_max": 0.0},
    },
}


def run_experiment(name: str, config_path: str, overrides: dict, mock: bool, smoke: bool):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"  Ablation: {name}")
    print(f"  Config:   {config_path}")
    print(f"{'='*60}")

    cmd = [sys.executable, "train.py", "--config", config_path]
    if mock:
        cmd.append("--mock")
    if smoke:
        cmd.extend(["--smoke-test", "--max-steps", "10"])

    # 通过环境变量传递 overrides
    env_overrides = {}
    for key, val in overrides.items():
        env_key = f"VRAG_OVERRIDE_{key.upper().replace('.', '_')}"
        env_overrides[env_key] = str(val)

    import os
    env = os.environ.copy()
    env.update(env_overrides)

    # 设置 WandB run name
    env["WANDB_RUN_NAME"] = f"ablation_{name}_{datetime.now().strftime('%m%d_%H%M')}"

    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode == 0


def run_eval(name: str, config_path: str, checkpoint: str, mock: bool):
    """运行评测"""
    cmd = [
        sys.executable, "eval.py",
        "--config", config_path,
        "--output", f"eval_results_{name}.json",
    ]
    if checkpoint and Path(checkpoint).exists():
        cmd.extend(["--checkpoint", checkpoint])
    if mock:
        cmd.append("--mock")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def collect_results():
    """收集所有消融实验结果"""
    results = {}
    for name in ABLATION_CONFIGS:
        result_file = Path(f"eval_results_{name}.json")
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            results[name] = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float))
            }

    if results:
        print(f"\n{'='*60}")
        print("  消融实验结果汇总")
        print(f"{'='*60}")
        print(f"{'Experiment':<15} {'Recall@1':>10} {'Recall@5':>10} {'MRR':>10} {'nDCG@5':>10}")
        print("-" * 60)
        for name, metrics in results.items():
            print(
                f"{name:<15} "
                f"{metrics.get('Recall@1', 0):.4f}     "
                f"{metrics.get('Recall@5', 0):.4f}     "
                f"{metrics.get('MRR', 0):.4f}     "
                f"{metrics.get('nDCG@5', 0):.4f}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Vision-RAG 消融实验")
    parser.add_argument("--config", type=str, default="config/vision_rag/default.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="仅评测，跳过训练")
    parser.add_argument("--experiments", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        help="指定要跑的实验 (默认全部)")
    args = parser.parse_args()

    for name in args.experiments:
        if name not in ABLATION_CONFIGS:
            print(f"未知实验: {name}, 可选: {list(ABLATION_CONFIGS.keys())}")
            continue

        cfg = ABLATION_CONFIGS[name]
        print(f"\n>>> {cfg['desc']}")

        # 优先使用实验专属配置文件，回退到命令行指定的配置
        exp_config = cfg.get("config", args.config)

        if not args.eval_only:
            success = run_experiment(name, exp_config, cfg["overrides"], args.mock, args.smoke_test)
            if not success:
                print(f"[WARN] {name} 训练失败，跳过评测")
                continue

        # 找最新 checkpoint
        ckpt_dir = Path("checkpoints/vision_rag")
        # 消融实验可能有独立 checkpoint 目录
        ablation_ckpt_dir = Path(f"checkpoints/ablation_{name}")
        if ablation_ckpt_dir.exists():
            ckpt_dir = ablation_ckpt_dir

        ckpt = None
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
            if ckpts:
                ckpt = str(ckpts[-1])

        run_eval(name, exp_config, ckpt, args.mock)

    collect_results()


if __name__ == "__main__":
    main()
