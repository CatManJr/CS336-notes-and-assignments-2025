#!/usr/bin/env python3
"""
Multi-process Sharded Optimizer Benchmark

This script automatically spawns multiple processes to benchmark the sharded optimizer
implementation against regular optimizers in a realistic distributed training scenario.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
import gc
import json
from typing import Dict, Any, List
from datetime import datetime

# Add cs336-basics to path
sys.path.append('cs336-basics')
from cs336_basics.transformer import TransformerLM

# Import our adapters
sys.path.append('tests')
from adapters import get_sharded_optimizer


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_model(size: str = 'medium'):
    """Create model based on size specification"""
    if size == 'small':
        return TransformerLM(
            vocab_size=5000,
            context_length=128,
            d_model=256,
            num_heads=4,
            num_layers=4,
            d_ff=1024
        )
    elif size == 'medium':
        return TransformerLM(
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_heads=8,
            num_layers=8,
            d_ff=2048
        )
    elif size == 'large':
        return TransformerLM(
            vocab_size=25000,
            context_length=512,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072
        )
    else:  # xl
        return TransformerLM(
            vocab_size=32000,
            context_length=512,
            d_model=1024,
            num_heads=16,
            num_layers=16,
            d_ff=4096
        )


def benchmark_worker(rank: int, world_size: int, use_sharded: bool, model_size: str, 
                    num_iterations: int, results_queue):
    """Worker function for benchmarking in distributed setting"""
    try:
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + (1 if use_sharded else 0))
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        
        # Clear memory
        gc.collect()
        baseline_memory = get_memory_usage()
        
        # Create model
        model = create_model(model_size)
        total_params = sum(p.numel() for p in model.parameters())
        model_memory = get_memory_usage()
        
        # Create optimizer
        if use_sharded:
            optimizer = get_sharded_optimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        optimizer_memory = get_memory_usage()
        
        # Create loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Generate training data
        batch_size = 4
        if model_size == 'small':
            seq_len, vocab_size = 128, 5000
        elif model_size == 'medium':
            seq_len, vocab_size = 256, 10000
        elif model_size == 'large':
            seq_len, vocab_size = 512, 25000
        else:  # xl
            seq_len, vocab_size = 512, 32000
        
        # Create batch data for this rank
        batch_per_rank = batch_size // world_size
        input_ids = torch.randint(0, vocab_size, (batch_per_rank, seq_len))
        targets = torch.randint(0, vocab_size, (batch_per_rank, seq_len))
        
        data_memory = get_memory_usage()
        
        # Warmup iterations
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            
            # Handle sharded optimizer synchronization
            if hasattr(optimizer, 'finish_gradient_synchronization'):
                optimizer.finish_gradient_synchronization()
            
            optimizer.step()
        
        warmup_memory = get_memory_usage()
        
        # Benchmark iterations
        dist.barrier()  # Synchronize before timing
        start_time = time.time()
        iteration_times = []
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            
            if hasattr(optimizer, 'finish_gradient_synchronization'):
                optimizer.finish_gradient_synchronization()
            
            optimizer.step()
            
            iter_time = (time.time() - iter_start) * 1000  # Convert to ms
            iteration_times.append(iter_time)
            
            if i == 0:  # Measure peak memory after first iteration
                peak_memory = get_memory_usage()
        
        dist.barrier()  # Synchronize after timing
        total_time = time.time() - start_time
        
        # Calculate memory breakdown
        param_memory_mb = (total_params * 4) / (1024 * 1024)
        optimizer_states_mb = optimizer_memory - model_memory
        activations_mb = warmup_memory - data_memory
        
        # Expected optimizer memory for comparison
        if use_sharded:
            # Sharded optimizer should use approximately 1/world_size of the optimizer memory
            expected_optimizer_memory = (total_params * 4 * 3) / (1024 * 1024) / world_size  # 3x for Adam states
        else:
            expected_optimizer_memory = (total_params * 4 * 3) / (1024 * 1024)  # 3x for Adam states
        
        results = {
            'rank': rank,
            'world_size': world_size,
            'use_sharded': use_sharded,
            'model_size': model_size,
            'total_params': total_params,
            'batch_size_per_rank': batch_per_rank,
            'memory': {
                'baseline_mb': baseline_memory,
                'model_mb': model_memory,
                'optimizer_mb': optimizer_memory,
                'data_mb': data_memory,
                'warmup_mb': warmup_memory,
                'peak_mb': peak_memory,
                'breakdown': {
                    'param_memory_mb': param_memory_mb,
                    'optimizer_states_mb': optimizer_states_mb,
                    'expected_optimizer_mb': expected_optimizer_memory,
                    'activations_mb': activations_mb
                }
            },
            'timing': {
                'total_time_s': total_time,
                'avg_iteration_ms': sum(iteration_times) / len(iteration_times),
                'min_iteration_ms': min(iteration_times),
                'max_iteration_ms': max(iteration_times),
                'all_times_ms': iteration_times
            }
        }
        
        results_queue.put(results)
        
    except Exception as e:
        error_result = {
            'rank': rank,
            'error': str(e),
            'use_sharded': use_sharded,
            'model_size': model_size
        }
        results_queue.put(error_result)
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_benchmark(world_size: int, use_sharded: bool, model_size: str, num_iterations: int = 10):
    """Run benchmark with specified configuration"""
    config_name = f"{'sharded' if use_sharded else 'regular'}_{model_size}_{world_size}proc"
    print(f"Running {config_name} benchmark...")
    
    # Create results queue
    results_queue = mp.Queue()
    processes = []
    
    # Spawn worker processes
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_worker,
            args=(rank, world_size, use_sharded, model_size, num_iterations, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(world_size):
        result = results_queue.get()
        results.append(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    return results


def analyze_results(regular_results: List[Dict], sharded_results: List[Dict]):
    """Analyze and compare results between regular and sharded optimizers"""
    if not regular_results or not sharded_results:
        print("Cannot analyze - missing results")
        return
    
    # Check for errors
    regular_errors = [r for r in regular_results if 'error' in r]
    sharded_errors = [r for r in sharded_results if 'error' in r]
    
    if regular_errors or sharded_errors:
        print("Errors encountered:")
        for error in regular_errors + sharded_errors:
            print(f"  Rank {error['rank']} ({'sharded' if error['use_sharded'] else 'regular'}): {error['error']}")
        return
    
    # Get rank 0 results for comparison
    regular_rank0 = next(r for r in regular_results if r['rank'] == 0)
    sharded_rank0 = next(r for r in sharded_results if r['rank'] == 0)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    world_size = regular_rank0['world_size']
    total_params = regular_rank0['total_params']
    
    print(f"Configuration: {world_size} processes")
    print(f"Model: {regular_rank0['model_size']}")
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Memory analysis
    print("MEMORY ANALYSIS (Per-Rank):")
    reg_peak = regular_rank0['memory']['peak_mb']
    shard_peak = sharded_rank0['memory']['peak_mb']
    
    print(f"  Regular optimizer peak: {reg_peak:.1f} MB")
    print(f"  Sharded optimizer peak: {shard_peak:.1f} MB")
    
    memory_diff = reg_peak - shard_peak
    if memory_diff > 0:
        print(f"  Memory savings: {memory_diff:.1f} MB ({memory_diff/reg_peak*100:.1f}%)")
    else:
        print(f"  Memory increase: {-memory_diff:.1f} MB ({-memory_diff/reg_peak*100:.1f}%)")
    
    print()
    print("MEMORY BREAKDOWN (Rank 0):")
    reg_breakdown = regular_rank0['memory']['breakdown']
    shard_breakdown = sharded_rank0['memory']['breakdown']
    
    print("  Regular optimizer:")
    print(f"    Parameters: {reg_breakdown['param_memory_mb']:.1f} MB")
    print(f"    Optimizer states: {reg_breakdown['optimizer_states_mb']:.1f} MB")
    print(f"    Expected optimizer: {reg_breakdown['expected_optimizer_mb']:.1f} MB")
    print(f"    Activations: {reg_breakdown['activations_mb']:.1f} MB")
    
    print("  Sharded optimizer:")
    print(f"    Parameters: {shard_breakdown['param_memory_mb']:.1f} MB")
    print(f"    Optimizer states: {shard_breakdown['optimizer_states_mb']:.1f} MB")
    print(f"    Expected optimizer: {shard_breakdown['expected_optimizer_mb']:.1f} MB")
    print(f"    Activations: {shard_breakdown['activations_mb']:.1f} MB")
    
    # Performance analysis
    print()
    print("PERFORMANCE ANALYSIS (Per-Rank):")
    reg_timing = regular_rank0['timing']
    shard_timing = sharded_rank0['timing']
    
    print(f"  Regular optimizer:")
    print(f"    Avg iteration: {reg_timing['avg_iteration_ms']:.1f} ms")
    print(f"    Min iteration: {reg_timing['min_iteration_ms']:.1f} ms")
    print(f"    Max iteration: {reg_timing['max_iteration_ms']:.1f} ms")
    
    print(f"  Sharded optimizer:")
    print(f"    Avg iteration: {shard_timing['avg_iteration_ms']:.1f} ms")
    print(f"    Min iteration: {shard_timing['min_iteration_ms']:.1f} ms")
    print(f"    Max iteration: {shard_timing['max_iteration_ms']:.1f} ms")
    
    time_diff = shard_timing['avg_iteration_ms'] - reg_timing['avg_iteration_ms']
    if time_diff > 0:
        print(f"  Overhead: +{time_diff:.1f} ms ({time_diff/reg_timing['avg_iteration_ms']*100:.1f}%)")
    else:
        print(f"  Speedup: {-time_diff:.1f} ms ({-time_diff/reg_timing['avg_iteration_ms']*100:.1f}%)")


def main():
    """Main function"""
    print("=" * 80)
    print("MULTI-PROCESS SHARDED OPTIMIZER BENCHMARK")
    print("=" * 80)
    
    # Configuration
    world_size = 2  # Use 2 processes
    model_size = 'medium'  # Use medium model for reasonable memory usage
    num_iterations = 5  # Number of timing iterations
    
    print(f"World size: {world_size}")
    print(f"Model size: {model_size}")
    print(f"Iterations: {num_iterations}")
    print()
    
    # Run benchmarks
    print("Starting benchmarks...")
    
    # Benchmark regular optimizer
    regular_results = run_benchmark(world_size, False, model_size, num_iterations)
    
    # Wait between benchmarks
    time.sleep(2)
    
    # Benchmark sharded optimizer
    sharded_results = run_benchmark(world_size, True, model_size, num_iterations)
    
    # Analyze results
    analyze_results(regular_results, sharded_results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'timestamp': timestamp,
        'world_size': world_size,
        'model_size': model_size,
        'num_iterations': num_iterations,
        'regular_results': regular_results,
        'sharded_results': sharded_results
    }
    
    os.makedirs("benchmark_results", exist_ok=True)
    filename = f"multiprocess_sharded_optimizer_benchmark_{timestamp}.json"
    filepath = os.path.join("benchmark_results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {filepath}")
    
    print(f"\nNOTES:")
    print(f"- This benchmark uses {world_size} processes with Gloo backend")
    print(f"- Sharded optimizer distributes optimizer states across processes")
    print(f"- Memory savings should scale with number of processes")
    print(f"- Some communication overhead is expected with sharded optimizer")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()