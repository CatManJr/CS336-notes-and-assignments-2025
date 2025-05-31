#!/usr/bin/env python3
"""
Distributed Communication Benchmark Script

This script benchmarks all-reduce operations across different:
- Backends (Gloo + CPU, NCCL + GPU)
- Data sizes (1MB, 10MB, 100MB, 1GB)
- Number of processes (2, 4, 6)

Usage:
    # Single node with 4 processes using NCCL + GPU
    torchrun --nproc_per_node=4 distributed_benchmark.py --backend nccl --device cuda
    
    # Single node with 2 processes using Gloo + CPU  
    torchrun --nproc_per_node=2 distributed_benchmark.py --backend gloo --device cpu
"""

import argparse
import json
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


def setup_distributed(backend: str, device: str):
    """Initialize distributed training setup"""
    # Initialize process group
    dist.init_process_group(backend=backend)
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device_obj = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    else:
        device_obj = torch.device('cpu')
    
    return rank, world_size, device_obj


def create_tensor(size_mb: float, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Create a tensor of specified size in MB"""
    # Calculate number of elements needed
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    num_elements = int((size_mb * 1024 * 1024) / bytes_per_element)
    
    # Create tensor with random data
    tensor = torch.randn(num_elements, device=device, dtype=dtype)
    return tensor


def benchmark_allreduce(tensor: torch.Tensor, num_warmup: int = 5, num_trials: int = 10) -> Dict[str, float]:
    """
    Benchmark all-reduce operation with proper warmup and synchronization
    
    Args:
        tensor: Input tensor for all-reduce
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement iterations
        
    Returns:
        Dictionary with timing statistics
    """
    device = tensor.device
    
    # Warmup iterations
    for _ in range(num_warmup):
        tensor_copy = tensor.clone()
        dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Measurement iterations
    times = []
    for _ in range(num_trials):
        tensor_copy = tensor.clone()
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dist.barrier()  # Ensure all processes start at the same time
        
        start_time = time.time()
        
        # Perform all-reduce
        dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)
        
        # Synchronize after operation
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(elapsed_time)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'all_times': times
    }


def collect_results_from_all_ranks(local_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect results from all ranks using all-gather"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Convert results to tensor for all-gather
    # We'll serialize to JSON string and then to bytes
    json_str = json.dumps(local_results)
    json_bytes = json_str.encode('utf-8')
    
    # Pad to fixed size (we'll use 4KB which should be enough)
    max_size = 4096
    if len(json_bytes) > max_size:
        raise ValueError(f"Results too large: {len(json_bytes)} > {max_size}")
    
    padded_bytes = json_bytes + b'\x00' * (max_size - len(json_bytes))
    tensor = torch.frombuffer(padded_bytes, dtype=torch.uint8)
    
    # All-gather
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    
    # Decode results
    all_results = []
    for gathered_tensor in gathered_tensors:
        gathered_bytes = gathered_tensor.numpy().tobytes()
        # Remove padding
        gathered_bytes = gathered_bytes.rstrip(b'\x00')
        json_str = gathered_bytes.decode('utf-8')
        result = json.loads(json_str)
        all_results.append(result)
    
    return all_results


def run_benchmark_suite(backend: str, device: str, data_sizes_mb: List[float]):
    """Run complete benchmark suite"""
    rank, world_size, device_obj = setup_distributed(backend, device)
    
    # Print system info (only rank 0)
    if rank == 0:
        print(f"=" * 80)
        print(f"DISTRIBUTED COMMUNICATION BENCHMARK")
        print(f"=" * 80)
        print(f"Backend: {backend}")
        print(f"Device: {device} ({device_obj})")
        print(f"World size: {world_size}")
        print(f"Data sizes: {data_sizes_mb} MB")
        
        if device == 'cuda' and torch.cuda.is_available():
            print(f"CUDA devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        print(f"=" * 80)
    
    all_results = []
    
    for size_mb in data_sizes_mb:
        if rank == 0:
            print(f"\nBenchmarking {size_mb} MB tensors...")
        
        try:
            # Create tensor
            tensor = create_tensor(size_mb, device_obj)
            actual_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
            
            # Run benchmark
            timing_results = benchmark_allreduce(tensor, num_warmup=5, num_trials=10)
            
            # Calculate bandwidth
            # All-reduce transfers 2*(n-1)/n * data_size in ring algorithm
            data_transferred_mb = 2 * (world_size - 1) / world_size * actual_size_mb
            bandwidth_mb_per_sec = data_transferred_mb / (timing_results['mean_ms'] / 1000)
            
            local_result = {
                'rank': rank,
                'backend': backend,
                'device': device,
                'world_size': world_size,
                'size_mb': actual_size_mb,
                'timing': timing_results,
                'bandwidth_mb_per_sec': bandwidth_mb_per_sec,
                'timestamp': datetime.now().isoformat()
            }
            
            if rank == 0:
                print(f"  Mean time: {timing_results['mean_ms']:.3f} ± {timing_results['std_ms']:.3f} ms")
                print(f"  Bandwidth: {bandwidth_mb_per_sec:.1f} MB/s")
            
            all_results.append(local_result)
            
        except Exception as e:
            if rank == 0:
                print(f"  Error with {size_mb} MB: {e}")
            continue
    
    # Collect results from all ranks
    if rank == 0:
        print(f"\nCollecting results from all ranks...")
    
    final_results = []
    for local_result in all_results:
        rank_results = collect_results_from_all_ranks(local_result)
        if rank == 0:  # Only rank 0 needs to store all results
            final_results.extend(rank_results)
    
    # Save and display results (only rank 0)
    if rank == 0:
        save_results(final_results, backend, device, world_size)
        display_summary(final_results)
    
    # Clean up
    dist.destroy_process_group()


def save_results(results: List[Dict[str, Any]], backend: str, device: str, world_size: int):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"distributed_benchmark_{backend}_{device}_{world_size}proc_{timestamp}.json"
    
    os.makedirs("benchmark_results", exist_ok=True)
    filepath = os.path.join("benchmark_results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


def display_summary(results: List[Dict[str, Any]]):
    """Display summary table of results"""
    print(f"\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Size (MB)':<12} {'Mean Time (ms)':<15} {'Std (ms)':<10} {'Bandwidth (MB/s)':<16} {'Min Time (ms)':<12}")
    print("-" * 80)
    
    # Group by size and calculate statistics across ranks
    size_groups = {}
    for result in results:
        size = result['size_mb']
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(result)
    
    for size in sorted(size_groups.keys()):
        group = size_groups[size]
        
        # Calculate statistics across ranks
        mean_times = [r['timing']['mean_ms'] for r in group]
        std_times = [r['timing']['std_ms'] for r in group]
        min_times = [r['timing']['min_ms'] for r in group]
        bandwidths = [r['bandwidth_mb_per_sec'] for r in group]
        
        avg_mean_time = np.mean(mean_times)
        avg_std_time = np.mean(std_times)
        avg_min_time = np.mean(min_times)
        avg_bandwidth = np.mean(bandwidths)
        
        print(f"{size:<12.1f} {avg_mean_time:<15.3f} {avg_std_time:<10.3f} {avg_bandwidth:<16.1f} {avg_min_time:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark distributed all-reduce operations')
    parser.add_argument('--backend', type=str, choices=['gloo', 'nccl'], required=True,
                        help='Distributed backend to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], required=True,
                        help='Device type to use')
    parser.add_argument('--sizes', type=float, nargs='+', 
                        default=[1.0, 10.0, 100.0, 1000.0],
                        help='Data sizes in MB to benchmark (default: 1 10 100 1000)')
    
    args = parser.parse_args()
    
    # Validate backend/device combination
    if args.backend == 'nccl' and args.device == 'cpu':
        raise ValueError("NCCL backend requires CUDA device")
    if args.backend == 'gloo' and args.device == 'cuda':
        print("Warning: Gloo with CUDA is supported but not optimal. Consider using NCCL.")
    
    # Run benchmark
    try:
        run_benchmark_suite(args.backend, args.device, args.sizes)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == '__main__':
    main()