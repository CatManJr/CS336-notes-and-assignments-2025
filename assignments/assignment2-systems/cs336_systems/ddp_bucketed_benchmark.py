"""
DDPBucketed Performance Benchmarking Script

This script benchmarks the DDPBucketed implementation with various bucket sizes
and compares against baseline DDP implementations.

Usage:
    # Run with 2 processes
    uv run python -m torch.distributed.run --nproc_per_node=2 cs336_systems/ddp_bucketed_benchmark.py
    
    # Run with 4 processes  
    uv run python -m torch.distributed.run --nproc_per_node=4 cs336_systems/ddp_bucketed_benchmark.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import json
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Import our transformer model from cs336-basics
import sys
sys.path.append('cs336-basics')
from cs336_basics.transformer import TransformerLM

# Import our DDP implementations
from naive_ddp import DDPIndividualParameters, DDPOverlapIndividualParameters, DDPBucketed


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize distributed training"""
    if not dist.is_initialized():
        dist.init_process_group('gloo')  # Use Gloo for CPU on macOS
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def create_xl_model():
    """Create XL model configuration - too large for laptops, use medium instead"""
    # Medium configuration - more suitable for laptops
    model = TransformerLM(
        vocab_size=50000,  
        context_length=512,  
        d_model=1600,      
        num_heads=25,      
        num_layers=48,    
        d_ff=6400         
    )
    return model


def create_medium_model():
    """Create medium model configuration suitable for laptops"""
    model = TransformerLM(
        vocab_size=5000,
        context_length=128,
        d_model=256,
        num_heads=4,
        num_layers=6,
        d_ff=1024
    )
    return model


def create_large_model():
    """Create large model configuration - bigger than medium but smaller than XL"""
    model = TransformerLM(
        vocab_size=25000,
        context_length=512,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072
    )
    return model


def generate_batch_data(batch_size: int, seq_len: int, vocab_size: int):
    """Generate random batch data for training"""
    # Generate random token ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Targets are input_ids shifted by 1 (next token prediction)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return input_ids, targets


def benchmark_ddp_bucketed(model, batch_data, rank, world_size, bucket_size_mb=25.0,
                          num_warmup=3, num_trials=10):
    """
    Benchmark DDPBucketed with a specific bucket size
    
    Args:
        model: Model to wrap with DDPBucketed
        batch_data: List of (input, target) tuples for training
        rank: Current process rank
        world_size: Number of processes
        bucket_size_mb: Bucket size in MB
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement iterations
        
    Returns:
        Dictionary with timing results and bucket info
    """
    # Wrap model with DDPBucketed
    ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    num_buckets = len(ddp_model._buckets)
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Split batch data across ranks
    local_batch_data = []
    for input_ids, targets in batch_data:
        batch_size_per_rank = input_ids.size(0) // world_size
        start_idx = rank * batch_size_per_rank
        end_idx = start_idx + batch_size_per_rank
        local_input = input_ids[start_idx:end_idx]
        local_targets = targets[start_idx:end_idx]
        local_batch_data.append((local_input, local_targets))
    
    # Warmup iterations
    for i in range(num_warmup):
        input_ids, targets = local_batch_data[i % len(local_batch_data)]
        optimizer.zero_grad()
        
        logits = ddp_model(input_ids)
        # Flatten for cross entropy loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    
    # Synchronize before timing
    dist.barrier()
    
    # Measurement iterations
    times = []
    for i in range(num_trials):
        input_ids, targets = local_batch_data[i % len(local_batch_data)]
        
        # Start timing
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = ddp_model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass (communication may overlap here)
        loss.backward()
        
        # Wait for gradient synchronization
        ddp_model.finish_gradient_synchronization()
        
        # Optimizer step
        optimizer.step()
        
        # End timing
        end_time = time.time()
        
        iteration_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(iteration_time)
    
    return {
        'mean_ms': float(torch.tensor(times).mean()),
        'std_ms': float(torch.tensor(times).std()),
        'min_ms': float(torch.tensor(times).min()),
        'max_ms': float(torch.tensor(times).max()),
        'all_times': times,
        'num_buckets': num_buckets,
        'bucket_size_mb': bucket_size_mb
    }


def benchmark_baseline_ddp(model, batch_data, rank, world_size, num_warmup=3, num_trials=10):
    """
    Benchmark baseline DDP (overlap individual parameters) for comparison
    """
    # Wrap model with baseline DDP
    ddp_model = DDPOverlapIndividualParameters(model)
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Split batch data across ranks
    local_batch_data = []
    for input_ids, targets in batch_data:
        batch_size_per_rank = input_ids.size(0) // world_size
        start_idx = rank * batch_size_per_rank
        end_idx = start_idx + batch_size_per_rank
        local_input = input_ids[start_idx:end_idx]
        local_targets = targets[start_idx:end_idx]
        local_batch_data.append((local_input, local_targets))
    
    # Warmup iterations
    for i in range(num_warmup):
        input_ids, targets = local_batch_data[i % len(local_batch_data)]
        optimizer.zero_grad()
        
        logits = ddp_model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    
    # Synchronize before timing
    dist.barrier()
    
    # Measurement iterations
    times = []
    for i in range(num_trials):
        input_ids, targets = local_batch_data[i % len(local_batch_data)]
        
        # Start timing
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = ddp_model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Wait for gradient synchronization
        ddp_model.finish_gradient_synchronization()
        
        # Optimizer step
        optimizer.step()
        
        # End timing
        end_time = time.time()
        
        iteration_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(iteration_time)
    
    return {
        'mean_ms': float(torch.tensor(times).mean()),
        'std_ms': float(torch.tensor(times).std()),
        'min_ms': float(torch.tensor(times).min()),
        'max_ms': float(torch.tensor(times).max()),
        'all_times': times
    }


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='Benchmark DDPBucketed with different bucket sizes')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (will be split across ranks)')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of timing trials')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'xl'], help='Model size to use')
    parser.add_argument('--bucket-sizes', type=str, default='1,10,100,1000',
                        help='Comma-separated list of bucket sizes in MB to test')
    
    args = parser.parse_args()
    
    # Parse bucket sizes
    bucket_sizes = [float(x) for x in args.bucket_sizes.split(',')]
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"=" * 80)
        print(f"DDPBUCKETED PERFORMANCE BENCHMARK")
        print(f"=" * 80)
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size} (total)")
        print(f"Batch size per rank: {args.batch_size // world_size}")
        print(f"Model: {args.model_size}")
        print(f"Trials: {args.num_trials}")
        print(f"Bucket sizes to test: {bucket_sizes} MB")
        print(f"=" * 80)
    
    # Set deterministic seed
    set_seed(42 + rank)
    
    # Create model based on size
    if args.model_size == 'xl':
        model = create_xl_model()
        seq_len = 256  # Reduced from 512
        vocab_size = 10000  # Reduced from 50257
    elif args.model_size == 'large':
        model = create_large_model()
        seq_len = 512
        vocab_size = 25000
    elif args.model_size == 'medium':
        model = create_medium_model()
        seq_len = 128
        vocab_size = 5000
    else:
        # Small model for very quick testing
        model = nn.Sequential(
            nn.Embedding(1000, 128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1000)
        )
        seq_len = 64
        vocab_size = 1000
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"Model parameters: {total_params:,} ({model_size_mb:.1f} MB)")
    
    # Generate batch data
    batch_data = []
    for _ in range(args.num_trials + 5):  # Extra for warmup
        input_ids, targets = generate_batch_data(args.batch_size, seq_len, vocab_size)
        batch_data.append((input_ids, targets))
    
    results = {}
    
    # First benchmark baseline (individual parameters)
    if rank == 0:
        print(f"\nBenchmarking Baseline DDP (Individual Parameters)...")
    
    try:
        baseline_model = create_medium_model() if args.model_size == 'medium' else \
                        create_large_model() if args.model_size == 'large' else \
                        create_xl_model() if args.model_size == 'xl' else \
                        nn.Sequential(
                            nn.Embedding(1000, 128),
                            nn.Linear(128, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 1000)
                        )
        
        baseline_results = benchmark_baseline_ddp(
            baseline_model, batch_data, rank, world_size, 
            num_warmup=3, num_trials=args.num_trials
        )
        
        results['Baseline (Individual Params)'] = baseline_results
        
        if rank == 0:
            print(f"  Mean time: {baseline_results['mean_ms']:.1f} ± {baseline_results['std_ms']:.1f} ms")
            
    except Exception as e:
        if rank == 0:
            print(f"  Error: {e}")
    
    # Benchmark each bucket size
    for bucket_size in bucket_sizes:
        if rank == 0:
            print(f"\nBenchmarking DDPBucketed with {bucket_size}MB buckets...")
        
        try:
            # Create a fresh copy of the model for each bucket size
            fresh_model = create_medium_model() if args.model_size == 'medium' else \
                         create_large_model() if args.model_size == 'large' else \
                         create_xl_model() if args.model_size == 'xl' else \
                         nn.Sequential(
                             nn.Embedding(1000, 128),
                             nn.Linear(128, 256),
                             nn.ReLU(),
                             nn.Linear(256, 128),
                             nn.ReLU(),
                             nn.Linear(128, 1000)
                         )
            
            bucketed_results = benchmark_ddp_bucketed(
                fresh_model, batch_data, rank, world_size, bucket_size_mb=bucket_size,
                num_warmup=3, num_trials=args.num_trials
            )
            
            results[f'Bucketed ({bucket_size}MB)'] = bucketed_results
            
            if rank == 0:
                print(f"  Mean time: {bucketed_results['mean_ms']:.1f} ± {bucketed_results['std_ms']:.1f} ms")
                print(f"  Number of buckets: {bucketed_results['num_buckets']}")
                
        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
            continue
    
    # Save and display results (only rank 0)
    if rank == 0:
        print(f"\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Implementation':<25} {'Mean (ms)':<12} {'Std (ms)':<10} {'Buckets':<10} {'Speedup':<10}")
        print("-" * 80)
        
        baseline_time = None
        for impl_name, timing_results in results.items():
            mean_time = timing_results['mean_ms']
            std_time = timing_results['std_ms']
            num_buckets = timing_results.get('num_buckets', 0)
            
            if baseline_time is None:
                baseline_time = mean_time
                speedup_str = "1.00x"
            else:
                speedup = baseline_time / mean_time
                speedup_str = f"{speedup:.2f}x"
            
            buckets_str = f"{num_buckets}" if num_buckets > 0 else "N/A"
            print(f"{impl_name:<25} {mean_time:<12.1f} {std_time:<10.1f} {buckets_str:<10} {speedup_str:<10}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ddp_bucketed_results_{world_size}proc_{args.model_size}_{timestamp}.json"
        
        os.makedirs("benchmark_results", exist_ok=True)
        filepath = os.path.join("benchmark_results", filename)
        
        detailed_results = {
            'timestamp': timestamp,
            'world_size': world_size,
            'batch_size': args.batch_size,
            'model_size': args.model_size,
            'num_trials': args.num_trials,
            'bucket_sizes_tested': bucket_sizes,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filepath}")
        
        # Analysis
        print(f"\nANALYSIS:")
        if len(results) >= 2:
            print("Bucket Size vs Performance:")
            bucket_results = [(k, v) for k, v in results.items() if 'Bucketed' in k]
            if bucket_results:
                best_bucket = min(bucket_results, key=lambda x: x[1]['mean_ms'])
                print(f"- Best performing bucket size: {best_bucket[0]} ({best_bucket[1]['mean_ms']:.1f}ms)")
                print(f"- Number of buckets: {best_bucket[1]['num_buckets']}")
                
                if baseline_time:
                    improvement = ((baseline_time - best_bucket[1]['mean_ms']) / baseline_time) * 100
                    if improvement > 0:
                        print(f"- Best bucket configuration is {improvement:.1f}% faster than baseline")
                    else:
                        print(f"- Baseline is {-improvement:.1f}% faster than best bucket configuration")
        
        # Part (b) Mathematical Analysis
        print(f"\n" + "=" * 80)
        print("PART (B): MATHEMATICAL ANALYSIS")
        print("=" * 80)
        print("DDP Communication Overhead Model:")
        print("  overhead = (s/w + o) × nb")
        print("  where:")
        print("    s = total model parameter size (bytes)")
        print("    w = all-reduce bandwidth (bytes/second)")
        print("    o = per-communication overhead (seconds)")
        print("    nb = number of buckets")
        print()
        print("Optimal bucket size:")
        print("  optimal_bucket_size = sqrt(2×s×o×w)")
        print("  This minimizes total overhead by balancing communication frequency vs efficiency")
        print()
        print("Trade-offs observed:")
        print("- Smaller buckets: More frequent communication, better overlap, higher overhead")
        print("- Larger buckets: Less frequent communication, worse overlap, lower overhead")
        print("- The optimal bucket size depends on model size, network bandwidth, and compute speed")
        
        print(f"\nNOTE: This benchmark runs on CPU with Gloo backend on macOS.")
        print(f"For GPU benchmarking with NCCL, run this on a CUDA-enabled system.")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()