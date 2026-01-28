"""
Hardware Performance Configuration
Optimized for Intel Core Ultra 9 + 40GB RAM + RTX 4060

Author: Zhichao Pan
Version: 1.0.0
"""
import os
import multiprocessing

# ============== CPU Configuration ==============
CPU_CORES = multiprocessing.cpu_count()
# Number of workers for data loading (reserve 2 cores for system)
NUM_WORKERS = max(1, CPU_CORES - 2)
# Thread pool size
THREAD_POOL_SIZE = CPU_CORES * 2

# ============== Memory Configuration ==============
TOTAL_RAM_GB = 40
# Vector database cache size (allocate 60% of RAM)
VECTOR_DB_CACHE_GB = int(TOTAL_RAM_GB * 0.6)
# Batch sizes (adjusted for memory)
BATCH_SIZE_EMBEDDING = 64  # Embedding generation batch
BATCH_SIZE_INFERENCE = 32  # Inference batch
# Document processing batch size
DOC_BATCH_SIZE = 100

# ============== GPU Configuration ==============
# RTX 4060 (8GB VRAM) optimization
GPU_MEMORY_GB = 8
# Reserve 1GB for system
GPU_MEMORY_FRACTION = 0.875  # Use 7GB
# Mixed precision training
USE_FP16 = True
USE_BF16 = False  # RTX 4060 supports it but FP16 is more stable

# ============== PyTorch Optimization ==============
PYTORCH_CONFIG = {
    "torch.backends.cudnn.benchmark": True,  # Enable cuDNN auto-tuning
    "torch.backends.cudnn.deterministic": False,  # Disable determinism for speed
    "torch.set_float32_matmul_precision": "medium",  # Matrix operation precision
}

# ============== Vector Database Configuration ==============
CHROMA_CONFIG = {
    "anonymized_telemetry": False,
    "persist_directory": "./data/chroma_db",
    "collection_metadata": {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # Index construction quality
        "hnsw:search_ef": 100,  # Search quality
        "hnsw:M": 32,  # Connection count (can increase with sufficient memory)
    }
}

# ============== Embedding Model Configuration ==============
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-large-en-v1.5",
    "device": "cuda",
    "normalize_embeddings": True,
    "batch_size": BATCH_SIZE_EMBEDDING,
}

# ============== LLM Configuration ==============
LLM_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True,
    "use_cache": True,
    "torch_dtype": "float16",  # Use FP16 to save VRAM
}

# ============== Environment Variable Setup ==============
def setup_environment():
    """Set performance-related environment variables."""
    # CUDA optimization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # PyTorch memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Multi-threading optimization
    os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)
    os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS)
    
    # Tokenizers parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # HuggingFace cache directory
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")


def get_torch_device():
    """Get optimal compute device."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ GPU not available, using CPU")
    return device


def apply_pytorch_optimizations():
    """Apply PyTorch performance optimizations."""
    import torch
    
    # Enable cuDNN optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set matrix operation precision
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')
    
    # If GPU available, set memory allocation strategy
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        torch.cuda.empty_cache()
        
    print("✓ PyTorch optimizations applied")


if __name__ == "__main__":
    setup_environment()
    apply_pytorch_optimizations()
    device = get_torch_device()
    
    print(f"\nHardware Configuration Overview:")
    print(f"  CPU Cores: {CPU_CORES}")
    print(f"  Worker Threads: {NUM_WORKERS}")
    print(f"  RAM: {TOTAL_RAM_GB}GB")
    print(f"  Vector DB Cache: {VECTOR_DB_CACHE_GB}GB")
    print(f"  Embedding Batch Size: {BATCH_SIZE_EMBEDDING}")
    print(f"  Inference Batch Size: {BATCH_SIZE_INFERENCE}")
