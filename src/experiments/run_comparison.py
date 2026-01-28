"""
Financial RAG Ablation Study - Comparison Script

This module runs the core experiment comparing two RAG pipelines:
1. Baseline: PyPDF-based unstructured text extraction.
2. Proposed: LlamaParse-based structure-aware Markdown extraction.

It handles:
- Asynchronous query processing with concurrency control.
- Fault tolerance via incremental checkpointing.
- Resource management (VRAM/RAM aggressive cleanup).
- Automatic metric tracking (Latency, Success Rate).

Usage:
    python run_comparison.py --safe --concurrency 2
"""
Financial RAG Ablation Study - Comparison Script
Compares structure-aware parsing vs naive text extraction.

Author: Zhichao Pan
Version: 1.0.0

Features:
- Checkpointing: Saves progress after every batch to prevent data loss.
- Safe Mode: Reduces load to prevent thermal/power crashes.
- Tunable Concurrency: User-defined parallelism.
"""
import pandas as pd
import os
import sys
import time
import asyncio
import torch
import platform
import gc
import argparse
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

# Suppress noisy logs
logging.getLogger().setLevel(logging.CRITICAL)

# ================= Configuration Constants =================
LLM_MODEL = "deepseek-r1:8b"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BENCHMARK_FILE = "data/benchmark/golden_dataset.csv"
PYPDF_PATH = "data/parsed/pypdf/parsed.md"
LLAMAPARSE_PATH = "data/parsed/llamaparse/parsed.md"
OUTPUT_DIR = "experiments"
OUTPUT_FILE = f"{OUTPUT_DIR}/comparison_results.csv"

# ================= System Utilities =================
def cleanup_resources():
    """Force garbage collection and VRAM cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_system_info(args):
    print("\n" + "="*60)
    print("🖥️  SYSTEM CONFIGURATION")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA Available: Yes")
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 VRAM: {vram_total:.2f} GB")
    else:
        print("⚠️  CUDA NOT AVAILABLE - Running on CPU")
    
    print("-" * 30)
    print(f"Run Mode: {'🛡️ SAFE' if args.safe else '⚡ PERFORMANCE'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Checkpointing: {'✅ Enabled' if not args.no_checkpoint else '❌ Disabled'}")
    print("="*60 + "\n")

# ================= Core Logic =================

def load_index(filepath: str, name: str, device: str) -> VectorStoreIndex:
    """Load markdown and build index."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    print(f"⚙️  Building Index for [{name}]...")
    try:
        start_time = time.time()
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        doc = Document(text=content)
        # Re-initialize embedding model to be safe or if context changes (optional optimization)
        index = VectorStoreIndex.from_documents([doc])
        
        elapsed = time.time() - start_time
        print(f"✅ [{name}] Index built in {elapsed:.2f}s")
        return index
    except Exception as e:
        print(f"❌ Error building index for {name}: {e}")
        return None

async def evaluate_single_question(query_engine, row, pipeline_name, semaphore, safe_mode):
    """Process a single question asynchronously with safety checks."""
    async with semaphore:
        question = row['question']
        qid = row['id']
        
        # In Safe Mode, aggressive cleanup before/after extensive work
        if safe_mode:
            cleanup_resources()
            # Cooling pause
            await asyncio.sleep(1.0) 

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
                        print(f"  ⚠️ Q{qid} Retry Failed: {retry_err}")
                        raise retry_err
                    await asyncio.sleep(2 * (attempt + 1)) # Backoff
            
            elapsed = time.time() - start_time
            print(f"  Completed Q{qid} ({pipeline_name}) in {elapsed:.1f}s")
            
            return {
                "id": qid,
                "question": question,
                "ground_truth": row['ground_truth'],
                "type": row['question_type'],
                "pipeline": pipeline_name,
                "model_answer": answer,
                "latency_s": round(elapsed, 2)
            }
        except Exception as e:
            print(f"  ❌ Error Q{qid} ({pipeline_name}): {e}")
            return {
                "id": qid,
                "question": question,
                "ground_truth": row['ground_truth'],
                "type": row['question_type'],
                "pipeline": pipeline_name,
                "model_answer": f"ERROR: {e}",
                "latency_s": -1
            }

async def run_evaluation_pipeline(index, pipeline_name, df, args, existing_results):
    """Run async evaluation with checkpointing."""
    print(f"\n🚀 Starting Evaluation for [{pipeline_name}]...")
    
    # Filter out already processed questions
    processed_ids = set()
    if existing_results:
        processed_ids = set(
            r['id'] for r in existing_results 
            if r['pipeline'] == pipeline_name and r['model_answer'] != "ERROR"
        )
    
    pending_df = df[~df['id'].isin(processed_ids)]
    print(f"   Total: {len(df)} | Processed: {len(processed_ids)} | Pending: {len(pending_df)}")
    
    if len(pending_df) == 0:
        print("   ✅ All items completed for this pipeline.")
        return []

    concurrency = 1 if args.safe else args.concurrency
    query_engine = index.as_query_engine(similarity_top_k=3)
    semaphore = asyncio.Semaphore(concurrency)
    
    new_results = []
    
    # Process in chunks to enable checkpointing
    # Convert dataframe to list of dicts for easier chunking
    pending_records = pending_df.to_dict('records')
    chunk_size = 1 if args.safe else 5  # Save more often in safe mode
    
    for i in range(0, len(pending_records), chunk_size):
        chunk = pending_records[i:i + chunk_size]
        tasks = []
        for row in chunk:
            task = evaluate_single_question(query_engine, row, pipeline_name, semaphore, args.safe)
            tasks.append(task)
        
        # Run batch
        batch_results = await asyncio.gather(*tasks)
        new_results.extend(batch_results)
        
        # Checkpoint: Save immediately
        if not args.no_checkpoint:
            save_checkpoint(existing_results + new_results)
            
    return new_results

def save_checkpoint(results):
    """Save current results to disk."""
    if not results:
        return
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        # print(f"  💾 Checkpoint saved ({len(results)} records)", end='\r')
    except Exception as e:
        print(f"  ⚠️ Checkpoint failed: {e}")

async def main():
    parser = argparse.ArgumentParser(description="RAG Comparison Experiment")
    parser.add_argument("--safe", action="store_true", help="Run in Safe Mode (Low concurrency, aggressive GC)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency limit (default: 2, ignored in Safe Mode)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable incremental saving")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    args = parser.parse_args()

    # Override concurrency in safe mode
    if args.safe:
        args.concurrency = 1

    print_system_info(args)
    
    # 1. Load Benchmark
    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ Benchmark file NOT found: {BENCHMARK_FILE}")
        return
        
    df = pd.read_csv(BENCHMARK_FILE)
    if args.limit:
        df = df.head(args.limit)
    print(f"📂 Loaded Benchmark: {len(df)} questions")
    
    # 2. Setup Resources
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 Initializing Local LLM ({LLM_MODEL})...")
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=600.0)
    
    print(f"🔧 Loading Embedding Model ({EMBED_MODEL})...")
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device=embed_device,
            trust_remote_code=True
        )
    except:
        print("⚠️ Failed to load large model, falling back to small.")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device=embed_device
        )

    # 3. Load Existing Results (Resume capability)
    existing_results = []
    if os.path.exists(OUTPUT_FILE) and not args.no_checkpoint:
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_results = existing_df.to_dict('records')
            print(f"🔄 Resuming from {len(existing_results)} existing records")
        except:
            print("⚠️ Checkpoint file corrupt or empty, starting fresh.")

    # 4. Run Pipelines
    # Pipeline A
    index_a = load_index(PYPDF_PATH, "Baseline (PyPDF)", embed_device)
    if index_a:
        res_a = await run_evaluation_pipeline(index_a, "Baseline (PyPDF)", df, args, existing_results)
        existing_results.extend([r for r in res_a if r not in existing_results])
        # Free memory
        del index_a
        cleanup_resources()

    # Pipeline B
    index_b = load_index(LLAMAPARSE_PATH, "Proposed (LlamaParse)", embed_device)
    if index_b:
        res_b = await run_evaluation_pipeline(index_b, "Proposed (LlamaParse)", df, args, existing_results)
        existing_results.extend([r for r in res_b if r not in existing_results])
        del index_b
        cleanup_resources()

    # 5. Final Summary
    print("\n" + "-"*40)
    print("📈 FINAL SUMMARY")
    print("-"*40)
    if existing_results:
        final_df = pd.DataFrame(existing_results)
        for pipeline in final_df['pipeline'].unique():
            subset = final_df[final_df['pipeline'] == pipeline]
            avg_lat = subset['latency_s'].mean()
            print(f"  {pipeline}: {len(subset)} items, {avg_lat:.2f}s avg latency")
    else:
        print("No results found.")
    
    print("\n🏁 Experiment Finished.")

if __name__ == "__main__":
    asyncio.run(main())

