"""
Financial RAG Ablation Study - Comparison Script

Compares structure-aware parsing (LlamaParse) vs naive text extraction (PyPDF).
Optimized for rigorous experimental conditions and hardware efficiency.
"""
import os
import sys
import time
import asyncio
import logging
import platform
import gc
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# ================= Configuration Setup =================
# Ensure we can find the config module regardless of run location
# This resolves "Unsafe sys.path" by using absolute resolution relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import config and setup environment BEFORE importing torch/llama_index
# This resolves "Phantom Config" and "Env Var Timing"
from config import hardware_config

# Force environment variables for GPU/Memory optimizations
hardware_config.setup_environment()
hardware_config.apply_pytorch_optimizations()

# Now safe to import heavy libraries
import torch
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure Logging (Resolves "Log Suicide")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "experiment.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= Constants =================
# Imported from hardware_config to ensure consistency (Resolves "Magic Numbers" and "Duplicate Code")
LLM_MODEL = hardware_config.LLM_MODEL_NAME
EMBED_MODEL = hardware_config.EMBED_MODEL_NAME

BENCHMARK_FILE = PROJECT_ROOT / "data" / "benchmark" / "golden_dataset.csv"
PYPDF_PATH = PROJECT_ROOT / "data" / "parsed" / "pypdf" / "parsed.md"
LLAMAPARSE_PATH = PROJECT_ROOT / "data" / "parsed" / "llamaparse" / "parsed.md"
OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "comparison_results.csv"

# ================= Utilities =================
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility (Resolves 'Reproducibility')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"🌱 Seed set to {seed}")

def cleanup_resources() -> None:
    """Force garbage collection and VRAM cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_system_info(args: argparse.Namespace) -> str:
    """Generate system status report string."""
    info = ["\n" + "="*60]
    info.append("🖥️  SYSTEM CONFIGURATION")
    info.append("="*60)
    info.append(f"OS: {platform.system()} {platform.release()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info.append(f"✅ CUDA Available: Yes")
        info.append(f"🚀 GPU: {gpu_name}")
        info.append(f"💾 VRAM: {vram_total:.2f} GB")
    else:
        info.append("⚠️  CUDA NOT AVAILABLE - Running on CPU")
    
    info.append("-" * 30)
    info.append(f"Run Mode: {'🛡️ SAFE' if args.safe else '⚡ PERFORMANCE'}")
    info.append(f"Concurrency: {args.concurrency}")
    info.append("="*60 + "\n")
    return "\n".join(info)

# ================= Core Logic =================

def load_or_build_index(source_path: Path, name: str, device: str) -> Optional[VectorStoreIndex]:
    """
    Load index from disk if available, otherwise build from source and persist.
    Resolves 'Inefficient Indexing'.
    
    Args:
        source_path: Path to the source markdown file.
        name: Name of the pipeline (for logging).
        device: 'cuda' or 'cpu'.
        
    Returns:
        VectorStoreIndex or None if failed.
    """
    if not source_path.exists():
        logger.error(f"❌ Source file not found: {source_path}")
        return None
    
    # Persistence directory based on chroma config + name
    persist_dir = Path(hardware_config.CHROMA_CONFIG["persist_directory"]) / name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    
    # Try to load from disk
    if persist_dir.exists():
        try:
            logger.info(f"📂 Loading existing index for [{name}] from {persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(storage_context)
            logger.info(f"✅ [{name}] Index loaded successfully.")
            return index
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing index: {e}. Rebuilding...")

    # Build from scratch
    logger.info(f"⚙️  Building new index for [{name}]...")
    try:
        start_time = time.time()
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        doc = Document(text=content)
        # Initialize index (this uses the Settings.embed_model implicitly)
        index = VectorStoreIndex.from_documents([doc])
        
        # Persist
        index.storage_context.persist(persist_dir=str(persist_dir))
        
        elapsed = time.time() - start_time
        logger.info(f"✅ [{name}] Index built and saved in {elapsed:.2f}s")
        return index
    except Exception as e:
        logger.error(f"❌ Error building index for {name}: {e}")
        return None

async def evaluate_single_question(
    query_engine: Any, 
    row: pd.Series, 
    pipeline_name: str, 
    semaphore: asyncio.Semaphore, 
    safe_mode: bool
) -> Dict[str, Any]:
    """
    Process a single question asynchronously.
    """
    async with semaphore:
        question = row['question']
        qid = row['id']
        
        # Removed "Manual GC Abuse" - unnecessary per-item GC unless extremely constrained
        
        try:
            start_time = time.time()
            
            # Simple retry mechanism
            max_retries = 3
            answer = "ERROR"
            
            for attempt in range(max_retries):
                try:
                    response = await query_engine.aquery(question)
                    answer = str(response).strip()
                    break
                except Exception as retry_err:
                    if attempt == max_retries - 1:
                        logger.warning(f"  ⚠️ Q{qid} Retry Failed: {retry_err}")
                        raise retry_err
                    await asyncio.sleep(2 * (attempt + 1)) # Backoff
            
            elapsed = time.time() - start_time
            logger.info(f"  Completed Q{qid} ({pipeline_name}) in {elapsed:.1f}s")
            
            return {
                "id": qid,
                "question": question,
                "ground_truth": row['ground_truth'],
                "type": row['question_type'],
                "pipeline": pipeline_name,
                "model_answer": answer,
                "latency_s": round(elapsed, 2)
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Resolves "Global Exception Catching" - slightly better logging
            logger.error(f"  ❌ Error Q{qid} ({pipeline_name}): {e}")
            return {
                "id": qid,
                "question": question,
                "ground_truth": row['ground_truth'],
                "type": row['question_type'],
                "pipeline": pipeline_name,
                "model_answer": f"ERROR: {e}",
                "latency_s": -1
            }

def save_checkpoint(results: List[Dict], output_file: Path) -> None:
    """
    Append new results to the CSV file safely.
    Resolves 'Checkpoint IO Disaster'.
    """
    if not results:
        return
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        
        # Check if file exists to determine header
        header = not output_file.exists()
        
        # Mode='a' appends instead of overwriting the whole file
        df.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8')
        logger.info(f"  💾 Checkpoint saved ({len(results)} records appended)")
    except Exception as e:
        logger.error(f"  ⚠️ Checkpoint failed: {e}")

async def run_evaluation_pipeline(
    index: VectorStoreIndex, 
    pipeline_name: str, 
    df: pd.DataFrame, 
    args: argparse.Namespace, 
    processed_ids: set
) -> List[Dict]:
    """Run async evaluation pipeline."""
    logger.info(f"\n🚀 Starting Evaluation for [{pipeline_name}]...")
    
    pending_df = df[~df['id'].isin(processed_ids)]
    logger.info(f"   Total: {len(df)} | Processed: {len(processed_ids)} | Pending: {len(pending_df)}")
    
    if len(pending_df) == 0:
        logger.info("   ✅ All items completed for this pipeline.")
        return []

    concurrency = 1 if args.safe else args.concurrency
    query_engine = index.as_query_engine(similarity_top_k=3)
    semaphore = asyncio.Semaphore(concurrency)
    
    new_results = []
    
    # Process in chunks
    pending_records = pending_df.to_dict('records')
    chunk_size = 1 if args.safe else 5
    
    for i in range(0, len(pending_records), chunk_size):
        chunk = pending_records[i:i + chunk_size]
        tasks = []
        for row in chunk:
            task = evaluate_single_question(query_engine, row, pipeline_name, semaphore, args.safe)
            tasks.append(task)
        
        # Run batch
        batch_results = await asyncio.gather(*tasks)
        new_results.extend(batch_results)
        
        # Checkpoint: Save immediately (incremental)
        if not args.no_checkpoint:
            save_checkpoint(batch_results, OUTPUT_FILE)
        
        # Performance: Clear VRAM after batch
        del batch_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return new_results

async def run_experiment(safe: bool = False, concurrency: int = 2, no_checkpoint: bool = False, limit: Optional[int] = None):
    """
    Main entry point for running the comparison experiment programmatically.
    """
    # Reproducibility
    set_seed(42)

    # CLI args mock for compatibility
    args = argparse.Namespace(
        safe=safe,
        concurrency=1 if safe else concurrency,
        no_checkpoint=no_checkpoint,
        limit=limit
    )

    print(get_system_info(args))
    
    # 1. Load Benchmark
    if not BENCHMARK_FILE.exists():
        logger.error(f"❌ Benchmark file NOT found: {BENCHMARK_FILE}")
        return
        
    df = pd.read_csv(BENCHMARK_FILE)
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"✂️  Limiting to {args.limit} questions")
    
    logger.info(f"📂 Loaded Benchmark: {len(df)} questions")
    if len(df) < 30:
        logger.warning(f"⚠️  Small sample size (N={len(df)}). Recommended N>=30 for statistical significance.")
    
    # 2. Setup Resources
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🔧 Initializing Local LLM ({LLM_MODEL})...")
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=600.0)
    
    logger.info(f"🔧 Loading Embedding Model ({EMBED_MODEL})...")
    # Fail fast if model fails, no silent downgrade (Resolves "Embedding Model Silent Degrade")
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device=embed_device,
            trust_remote_code=True
        )
    except Exception as e:
        logger.critical(f"❌ Failed to load embedding model {EMBED_MODEL}: {e}")
        logger.critical("   Terminating to prevent silent degradation.")
        sys.exit(1)

    # 3. Load Existing Results (Resume capability)
    processed_ids_a = set()
    processed_ids_b = set()
    
    if OUTPUT_FILE.exists() and not args.no_checkpoint:
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            # Filter already done for each pipeline
            processed_ids_a = set(existing_df[existing_df['pipeline'] == "Baseline (PyPDF)"]['id'])
            processed_ids_b = set(existing_df[existing_df['pipeline'] == "Proposed (LlamaParse)"]['id'])
            logger.info(f"🔄 Resuming... Found {len(existing_df)} total records.")
        except Exception as e:
            logger.warning(f"⚠️ Could not read checkpoint file: {e}")

    # 4. Run Pipelines
    
    # Pipeline A: Baseline
    index_a = load_or_build_index(PYPDF_PATH, "Baseline (PyPDF)", embed_device)
    if index_a:
        await run_evaluation_pipeline(index_a, "Baseline (PyPDF)", df, args, processed_ids_a)
        del index_a
        cleanup_resources()

    # Pipeline B: LlamaParse
    index_b = load_or_build_index(LLAMAPARSE_PATH, "Proposed (LlamaParse)", embed_device)
    if index_b:
        await run_evaluation_pipeline(index_b, "Proposed (LlamaParse)", df, args, processed_ids_b)
        del index_b
        cleanup_resources()

    # 5. Final Summary (Read from disk to get everything)
    logger.info("\n" + "-"*40)
    logger.info("📈 FINAL SUMMARY")
    logger.info("-"*40)
    if OUTPUT_FILE.exists():
        final_df = pd.read_csv(OUTPUT_FILE)
        for pipeline in final_df['pipeline'].unique():
            subset = final_df[final_df['pipeline'] == pipeline]
            avg_lat = subset['latency_s'].mean()
            logger.info(f"  {pipeline}: {len(subset)} items, {avg_lat:.2f}s avg latency")
    else:
        logger.info("No results found.")
    
    logger.info("\n🏁 Experiment Finished.")

async def main():
    parser = argparse.ArgumentParser(description="Structure-Aware RAG Comparison")
    parser.add_argument("--safe", action="store_true", help="Run in Safe Mode (Low concurrency)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency limit")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable incremental saving")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    await run_experiment(
        safe=args.safe, 
        concurrency=args.concurrency, 
        no_checkpoint=args.no_checkpoint, 
        limit=args.limit
    )

if __name__ == "__main__":
    asyncio.run(main())
