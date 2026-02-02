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

# Fix relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment_command(args)
    elif args.command == "parse":
        run_parse_command(args)
    elif args.command == "score":
        run_score_command(args)
    elif args.command == "viz":
        run_viz_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
