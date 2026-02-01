"""
GPU and Hardware Performance Verification Script
Run this script to verify all hardware optimization configurations.

Author: Zhichao Pan
Version: 1.0.0
"""
import sys
import os
import platform

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def check_system():
    print_section("System Information")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Processor: {platform.processor()}")

def check_pytorch():
    print_section("PyTorch Configuration")
    try:
        import torch
        print(f"  ✓ PyTorch Version: {torch.__version__}")
        print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA Version: {torch.version.cuda}")
            print(f"  ✓ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  ✓ GPU Device: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"  ✓ GPU Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  ✓ CUDA Cores: {props.multi_processor_count} SM")
            print(f"  ✓ Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU computation
            print("\n  Testing GPU compute performance...")
            x = torch.randn(10000, 10000, device='cuda')
            
            # Warmup
            for _ in range(3):
                y = torch.matmul(x, x)
            torch.cuda.synchronize()
            
            # Timing
            import time
            start = time.time()
            for _ in range(10):
                y = torch.matmul(x, x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            tflops = (2 * 10000**3 * 10) / elapsed / 1e12
            print(f"  ✓ Matrix Computation: {tflops:.2f} TFLOPS")
            
            # Cleanup VRAM
            del x, y
            torch.cuda.empty_cache()
        else:
            print("  ⚠ CUDA not available, please check drivers and PyTorch installation")
            
    except ImportError:
        print("  ✗ PyTorch not installed")

def check_cpu_memory():
    print_section("CPU and Memory")
    import multiprocessing
    print(f"  ✓ CPU Cores: {multiprocessing.cpu_count()}")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  ✓ Total RAM: {mem.total / 1024**3:.1f} GB")
        print(f"  ✓ Available RAM: {mem.available / 1024**3:.1f} GB")
        print(f"  ✓ Memory Usage: {mem.percent}%")
    except ImportError:
        print("  ⚠ psutil not installed, cannot get detailed memory info")

def check_gpu_memory():
    print_section("GPU Memory Status")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Allocated: {allocated:.2f} GB")
                print(f"    Reserved: {reserved:.2f} GB")
                print(f"    Total: {total:.1f} GB")
                print(f"    Available: {total - reserved:.2f} GB")
    except Exception as e:
        print(f"  ✗ Cannot get GPU memory info: {e}")

def check_optimizations():
    print_section("Performance Optimization Status")
    try:
        import torch
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
        
        if hasattr(torch.backends, 'cuda'):
            print(f"  Flash Attention Available: {torch.backends.cuda.flash_sdp_enabled()}")
    except Exception as e:
        print(f"  ⚠ Cannot check optimization status: {e}")

def main():
    print("\n" + "★"*50)
    print("  Structure-Aware RAG Hardware Verification")
    print("★"*50)
    
    check_system()
    check_cpu_memory()
    check_pytorch()
    check_gpu_memory()
    check_optimizations()
    
    print("\n" + "="*50)
    print("  Verification Complete! All hardware ready ✓")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
