#!/usr/bin/env python3
"""
DDP Performance Benchmarking Script

This script benchmarks different DDP implementations:
1. Naive DDP (individual parameters, no overlap)
2. Overlapped DDP (individual parameters with compute/communication overlap)  
3. Bucketed DDP (grouped parameters for fewer communication calls)

Usage:
    # Run with 2 processes
    python -m torch.distributed.run --nproc_per_node=2 cs336_systems/ddp_benchmark.py
    
    # Run with 4 processes  
    python -m torch.distributed.run --nproc_per_node=4 cs336_systems/ddp_benchmark.py
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


def benchmark_ddp_implementation(ddp_class, model, batch_data, rank, world_size, 
                               num_warmup=3, num_trials=10):
    """
    Benchmark a specific DDP implementation
    
    Args:
        ddp_class: DDP class to benchmark
        model: Model to wrap with DDP
        batch_data: List of (input, target) tuples for training
        rank: Current process rank
        world_size: Number of processes
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement iterations
        
    Returns:
        Dictionary with timing results
    """
    # Wrap model with DDP
    if ddp_class == DDPBucketed:
        ddp_model = ddp_class(model, bucket_size_mb=25.0)
    else:
        ddp_model = ddp_class(model)
    
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
        'all_times': times
    }


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='Benchmark DDP implementations')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (will be split across ranks)')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of timing trials')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'xl'], help='Model size to use')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"=" * 80)
        print(f"DDP PERFORMANCE BENCHMARK")
        print(f"=" * 80)
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size} (total)")
        print(f"Batch size per rank: {args.batch_size // world_size}")
        print(f"Model: {args.model_size}")
        print(f"Trials: {args.num_trials}")
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
    
    # DDP implementations to benchmark
    ddp_implementations = [
        ('Naive DDP', DDPIndividualParameters),
        ('Overlapped DDP', DDPOverlapIndividualParameters),
    ]
    
    results = {}
    
    # Benchmark each implementation
    for impl_name, ddp_class in ddp_implementations:
        if rank == 0:
            print(f"\nBenchmarking {impl_name}...")
        
        try:
            # Create a fresh copy of the model for each implementation
            if args.model_size == 'xl':
                fresh_model = create_xl_model()
            elif args.model_size == 'large':
                fresh_model = create_large_model()
            elif args.model_size == 'medium':
                fresh_model = create_medium_model()
            else:
                fresh_model = nn.Sequential(
                    nn.Embedding(1000, 128),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1000)
                )
            
            timing_results = benchmark_ddp_implementation(
                ddp_class, fresh_model, batch_data, rank, world_size, 
                num_warmup=3, num_trials=args.num_trials
            )
            
            results[impl_name] = timing_results
            
            if rank == 0:
                print(f"  Mean time: {timing_results['mean_ms']:.1f} ± {timing_results['std_ms']:.1f} ms")
                print(f"  Min time: {timing_results['min_ms']:.1f} ms")
                print(f"  Max time: {timing_results['max_ms']:.1f} ms")
                
        except Exception as e:
            if rank == 0:
                print(f"  Error: {e}")
            continue
    
    # Save and display results (only rank 0)
    if rank == 0:
        print(f"\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Implementation':<20} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Speedup':<10}")
        print("-" * 80)
        
        baseline_time = None
        for impl_name, timing_results in results.items():
            mean_time = timing_results['mean_ms']
            std_time = timing_results['std_ms']
            min_time = timing_results['min_ms']
            
            if baseline_time is None:
                baseline_time = mean_time
                speedup_str = "1.00x"
            else:
                speedup = baseline_time / mean_time
                speedup_str = f"{speedup:.2f}x"
            
            print(f"{impl_name:<20} {mean_time:<12.1f} {std_time:<10.1f} {min_time:<10.1f} {speedup_str:<10}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ddp_benchmark_results_{world_size}proc_{args.model_size}_{timestamp}.json"
        
        os.makedirs("benchmark_results", exist_ok=True)
        filepath = os.path.join("benchmark_results", filename)
        
        detailed_results = {
            'timestamp': timestamp,
            'world_size': world_size,
            'batch_size': args.batch_size,
            'model_size': args.model_size,
            'num_trials': args.num_trials,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filepath}")
        
        # Analysis
        print(f"\nANALYSIS:")
        if len(results) >= 2:
            impl_names = list(results.keys())
            naive_time = results[impl_names[0]]['mean_ms']
            overlap_time = results[impl_names[1]]['mean_ms']
            
            if overlap_time < naive_time:
                improvement = ((naive_time - overlap_time) / naive_time) * 100
                print(f"- Overlapped DDP is {improvement:.1f}% faster than naive DDP")
                print(f"- This demonstrates successful compute/communication overlap")
            else:
                print(f"- Overlapped DDP shows no improvement (overhead may dominate)")
                print(f"- Consider larger models or more communication-intensive workloads")
        
        print(f"\nNOTE: This benchmark runs on CPU with Gloo backend on macOS.")
        print(f"For GPU benchmarking with NCCL, run this on a CUDA-enabled system.")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()