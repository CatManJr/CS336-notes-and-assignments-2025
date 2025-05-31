#!/bin/bash
"""
Run all distributed communication benchmarks

This script runs the complete suite of distributed communication benchmarks
across different backend/device combinations and process counts.

Since we're on macOS, we'll focus on CPU-based benchmarks with Gloo backend.
For GPU benchmarks, this would need to be run on a system with CUDA support.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_benchmark(backend, device, num_procs, sizes):
    """Run a single benchmark configuration"""
    print(f"\n{'='*60}")
    print(f"Running: {backend} + {device}, {num_procs} processes")
    print(f"Data sizes: {sizes}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={num_procs}",
        "cs336_systems/distributed_benchmark.py",
        "--backend", backend,
        "--device", device,
        "--sizes"
    ] + [str(s) for s in sizes]
    
    try:
        # Run the benchmark
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✅ Benchmark completed successfully")
            print("STDOUT:")
            print(result.stdout)
        else:
            print("❌ Benchmark failed")
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Benchmark timed out (5 minutes)")
    except Exception as e:
        print(f"💥 Error running benchmark: {e}")

def main():
    """Run complete benchmark suite"""
    print("🚀 Starting Distributed Communication Benchmark Suite")
    print(f"Timestamp: {datetime.now()}")
    
    # Data sizes to test (in MB)
    data_sizes = [1.0, 10.0, 100.0]  # Reduced for macOS testing
    
    # Test configurations
    # Note: On macOS without CUDA, we can only test Gloo + CPU
    configs = [
        ("gloo", "cpu", 2),
        ("gloo", "cpu", 4),
    ]
    
    # Add GPU configs if CUDA is available (commented out for macOS)
    # if torch.cuda.is_available():
    #     configs.extend([
    #         ("nccl", "cuda", 2),
    #         ("nccl", "cuda", 4),
    #         ("nccl", "cuda", 6),
    #     ])
    
    print(f"\nPlanned configurations:")
    for backend, device, num_procs in configs:
        print(f"  - {backend} + {device}, {num_procs} processes")
    
    # Run each configuration
    for i, (backend, device, num_procs) in enumerate(configs):
        print(f"\n🔄 Running configuration {i+1}/{len(configs)}")
        run_benchmark(backend, device, num_procs, data_sizes)
    
    print(f"\n🏁 Benchmark suite completed!")
    print(f"Results are saved in the 'benchmark_results' directory")

if __name__ == "__main__":
    main()