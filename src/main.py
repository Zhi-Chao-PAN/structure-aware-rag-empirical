"""
Structure-Aware RAG - Main Entry Point

Provides a unified CLI for the entire project lifecycle:
1. Parse Documents
2. Run Experiments 
3. Score Results
4. Visualize

Usage:
    python -m src.main experiment --safe
    python -m src.main parse
"""
import argparse
import sys
import asyncio
import logging
from pathlib import Path

# Standard import
from src.experiments.run_comparison import run_experiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-cli")

def run_parse_command(args):
    """Wrapper for document parsing."""
    logger.info("📄 Starting Document Parsing Pipeline...")
    # Import here to avoid heavy loading if not needed
    try:
        from src.parsing.generate_datasets import main as parse_main
        parse_main()
    except Exception as e:
        logger.error(f"Parsing Failed: {e}")
        sys.exit(1)

def run_experiment_command(args):
    """Wrapper for running experiments."""
    logger.info("🧪 Starting RAG Comparison Experiment...")
    try:
        asyncio.run(run_experiment(
            safe=args.safe,
            concurrency=args.concurrency,
            no_checkpoint=args.no_checkpoint,
            limit=args.limit
        ))
    except Exception as e:
        logger.error(f"Experiment Failed: {e}")
        sys.exit(1)

def run_score_command(args):
    """Wrapper for auto-scoring."""
    logger.info("✅ Starting Auto-Scoring...")
    try:
        # Assuming auto_score.py is importable or we run via subprocess
        # Since auto_score.py might update files, better to run it via subprocess if it's not well factored
        # But let's try to import
        import subprocess
        subprocess.run([sys.executable, "src/evaluation/auto_score.py"], check=True)
    except Exception as e:
        logger.error(f"Scoring Failed: {e}")
        sys.exit(1)

def run_viz_command(args):
    """Wrapper for visualization."""
    logger.info("📊 Generating Visualizations...")
    try:
        import subprocess
        subprocess.run([sys.executable, "src/evaluation/visualize.py"], check=True)
    except Exception as e:
        logger.error(f"Visualization Failed: {e}")
        sys.exit(1)

def run_verify_command(args):
    """Run hardware verification checks."""
    logger.info("🕵️ Verifying Environment...")
    try:
        # Inline implementation to avoid keeping scripts/verify_hardware.py
        import platform
        import psutil
        print(f"\n{'='*50}\n  System Information\n{'='*50}")
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Python: {sys.version.split()[0]}")
        
        # CPU
        import multiprocessing
        print(f"  CPU Cores: {multiprocessing.cpu_count()}")
        mem = psutil.virtual_memory()
        print(f"  RAM: {mem.available / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB")

        # PyTorch
        import torch
        print(f"\n{'='*50}\n  PyTorch & GPU\n{'='*50}")
        print(f"  Torch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("  ⚠️ CUDA Not Available")
            
    except Exception as e:
        logger.error(f"Verify Failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Structure-Aware RAG: Scientific Toolbelt",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: experiment
    exp_parser = subparsers.add_parser("experiment", help="Run the RAG comparison experiment")
    exp_parser.add_argument("--safe", action="store_true", help="Run in Safe Mode (1 worker, low memory)")
    exp_parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent queries")
    exp_parser.add_argument("--no-checkpoint", action="store_true", help="Disable incremental saving")
    exp_parser.add_argument("--limit", type=int, default=None, help="Limit number of questions for quick testing")

    # Command: parse
    subparsers.add_parser("parse", help="Parse PDF documents into Markdown")

    # Command: score
    subparsers.add_parser("score", help="Score experiment results")

    # Command: viz
    subparsers.add_parser("viz", help="Generate plots and reports")

    # Command: verify
    subparsers.add_parser("verify", help="Check hardware compatibility")

    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment_command(args)
    elif args.command == "parse":
        run_parse_command(args)
    elif args.command == "score":
        run_score_command(args)
    elif args.command == "viz":
        run_viz_command(args)
    elif args.command == "verify":
        run_verify_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
