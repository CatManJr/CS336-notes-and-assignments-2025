#!/usr/bin/env python3
"""
Multi-process Memory Profiling Script for Optimizer State Sharding Analysis.

This script automatically spawns multiple processes to measure peak memory usage 
with and without optimizer state sharding to analyze the memory savings and 
performance impact in realistic distributed training scenarios.
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
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def create_model(size='medium'):
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
            vocab_size=50000,
            context_length=1024,
            d_model=2048,
            num_heads=32,
            num_layers=24,
            d_ff=8192
        )


def generate_batch_data(batch_size: int, seq_len: int, vocab_size: int):
    """Generate random batch data"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, targets


def profile_worker(rank: int, world_size: int, use_sharded_optimizer: bool, 
                  model_size: str, results_queue):
    """
    Worker function for profiling memory usage in distributed setting
    
    Args:
        rank: Process rank
        world_size: Number of processes
        use_sharded_optimizer: Whether to use sharded optimizer
        model_size: Size of model to use
        results_queue: Queue to put results
    """
    try:
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + (1 if use_sharded_optimizer else 0))
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        
        print(f"Rank {rank}: Starting memory profiling {'WITH' if use_sharded_optimizer else 'WITHOUT'} sharded optimizer")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Measure baseline memory
        baseline_memory = get_memory_usage()
        print(f"Rank {rank}: Baseline memory: {baseline_memory:.1f} MB")
        
        # Create model
        model = create_model(model_size)
        model_memory = get_memory_usage()
        print(f"Rank {rank}: After model creation: {model_memory:.1f} MB (+{model_memory - baseline_memory:.1f} MB)")
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"Rank {rank}: Model parameters: {total_params:,} ({param_memory_mb:.1f} MB)")
        
        # Create optimizer
        if use_sharded_optimizer:
            optimizer = get_sharded_optimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        optimizer_memory = get_memory_usage()
        print(f"Rank {rank}: After optimizer creation: {optimizer_memory:.1f} MB (+{optimizer_memory - model_memory:.1f} MB)")
        
        # Create loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Generate batch data based on model size
        if model_size == 'small':
            batch_size, seq_len, vocab_size = 4, 128, 5000
        elif model_size == 'medium':
            batch_size, seq_len, vocab_size = 4, 256, 10000
        elif model_size == 'large':
            batch_size, seq_len, vocab_size = 2, 512, 25000
        else:  # xl
            batch_size, seq_len, vocab_size = 2, 1024, 50000
        
        input_ids, targets = generate_batch_data(batch_size, seq_len, vocab_size)
        
        # Split batch across ranks for distributed training
        if world_size > 1:
            batch_per_rank = batch_size // world_size
            start_idx = rank * batch_per_rank
            end_idx = start_idx + batch_per_rank
            input_ids = input_ids[start_idx:end_idx]
            targets = targets[start_idx:end_idx]
        
        batch_memory = get_memory_usage()
        print(f"Rank {rank}: After batch creation: {batch_memory:.1f} MB (+{batch_memory - optimizer_memory:.1f} MB)")
        
        # Training step - Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        forward_memory = get_memory_usage()
        print(f"Rank {rank}: After forward pass: {forward_memory:.1f} MB (+{forward_memory - batch_memory:.1f} MB)")
        
        # Backward pass
        loss.backward()
        
        backward_memory = get_memory_usage()
        print(f"Rank {rank}: After backward pass: {backward_memory:.1f} MB (+{backward_memory - forward_memory:.1f} MB)")
        
        # Optimizer step with synchronization
        if hasattr(optimizer, 'finish_gradient_synchronization'):
            optimizer.finish_gradient_synchronization()
        optimizer.step()
        
        step_memory = get_memory_usage()
        print(f"Rank {rank}: After optimizer step: {step_memory:.1f} MB (+{step_memory - backward_memory:.1f} MB)")
        
        # Calculate expected optimizer memory for comparison
        if use_sharded_optimizer and world_size > 1:
            # Sharded optimizer should use approximately 1/world_size of the optimizer memory
            expected_optimizer_mb = (total_params * 4 * 3) / (1024 * 1024) / world_size  # 3x for Adam states
        else:
            expected_optimizer_mb = (total_params * 4 * 3) / (1024 * 1024)  # 3x for Adam states
        
        # Calculate memory breakdown
        model_params_mb = param_memory_mb
        optimizer_states_mb = optimizer_memory - model_memory
        activations_mb = forward_memory - batch_memory
        gradients_mb = backward_memory - forward_memory
        
        results = {
            'use_sharded_optimizer': use_sharded_optimizer,
            'rank': rank,
            'world_size': world_size,
            'model_size': model_size,
            'total_params': total_params,
            'batch_size_per_rank': input_ids.size(0),
            'memory_progression': {
                'baseline_mb': baseline_memory,
                'model_mb': model_memory,
                'optimizer_mb': optimizer_memory,
                'batch_mb': batch_memory,
                'forward_mb': forward_memory,
                'backward_mb': backward_memory,
                'step_mb': step_memory,
                'peak_mb': step_memory
            },
            'memory_breakdown': {
                'model_params_mb': model_params_mb,
                'optimizer_states_mb': optimizer_states_mb,
                'expected_optimizer_mb': expected_optimizer_mb,
                'activations_mb': activations_mb,
                'gradients_mb': gradients_mb
            },
            'loss_value': loss.item()
        }
        
        results_queue.put(results)
        
    except Exception as e:
        error_result = {
            'rank': rank,
            'error': str(e),
            'use_sharded_optimizer': use_sharded_optimizer,
            'model_size': model_size
        }
        results_queue.put(error_result)
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_profile(world_size: int, use_sharded_optimizer: bool, model_size: str):
    """Run profiling with specified configuration"""
    config_name = f"{'sharded' if use_sharded_optimizer else 'regular'}_{model_size}_{world_size}proc"
    print(f"Running {config_name} memory profiling...")
    
    # Create results queue
    results_queue = mp.Queue()
    processes = []
    
    # Spawn worker processes
    for rank in range(world_size):
        p = mp.Process(
            target=profile_worker,
            args=(rank, world_size, use_sharded_optimizer, model_size, results_queue)
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


def analyze_memory_results(regular_results: List[Dict], sharded_results: List[Dict]):
    """Analyze and compare memory results between regular and sharded optimizers"""
    if not regular_results or not sharded_results:
        print("Cannot analyze - missing results")
        return
    
    # Check for errors
    regular_errors = [r for r in regular_results if 'error' in r]
    sharded_errors = [r for r in sharded_results if 'error' in r]
    
    if regular_errors or sharded_errors:
        print("Errors encountered:")
        for error in regular_errors + sharded_errors:
            print(f"  Rank {error['rank']} ({'sharded' if error['use_sharded_optimizer'] else 'regular'}): {error['error']}")
        return
    
    # Get rank 0 results for comparison
    regular_rank0 = next(r for r in regular_results if r['rank'] == 0)
    sharded_rank0 = next(r for r in sharded_results if r['rank'] == 0)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MEMORY ANALYSIS")
    print(f"{'='*80}")
    
    world_size = regular_rank0['world_size']
    total_params = regular_rank0['total_params']
    model_size = regular_rank0['model_size']
    
    print(f"Configuration: {world_size} processes")
    print(f"Model: {model_size}")
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Memory analysis
    print("PEAK MEMORY USAGE (Per-Rank):")
    reg_peak = regular_rank0['memory_progression']['peak_mb']
    shard_peak = sharded_rank0['memory_progression']['peak_mb']
    
    print(f"  Regular optimizer: {reg_peak:.1f} MB")
    print(f"  Sharded optimizer: {shard_peak:.1f} MB")
    
    memory_diff = reg_peak - shard_peak
    if memory_diff > 0:
        print(f"  Memory savings: {memory_diff:.1f} MB ({memory_diff/reg_peak*100:.1f}%)")
    else:
        print(f"  Memory increase: {-memory_diff:.1f} MB ({-memory_diff/reg_peak*100:.1f}%)")
    
    print()
    print("MEMORY PROGRESSION (Rank 0):")
    print("  Regular optimizer:")
    reg_prog = regular_rank0['memory_progression']
    print(f"    Baseline: {reg_prog['baseline_mb']:.1f} MB")
    print(f"    + Model: {reg_prog['model_mb']:.1f} MB (+{reg_prog['model_mb'] - reg_prog['baseline_mb']:.1f})")
    print(f"    + Optimizer: {reg_prog['optimizer_mb']:.1f} MB (+{reg_prog['optimizer_mb'] - reg_prog['model_mb']:.1f})")
    print(f"    + Batch: {reg_prog['batch_mb']:.1f} MB (+{reg_prog['batch_mb'] - reg_prog['optimizer_mb']:.1f})")
    print(f"    + Forward: {reg_prog['forward_mb']:.1f} MB (+{reg_prog['forward_mb'] - reg_prog['batch_mb']:.1f})")
    print(f"    + Backward: {reg_prog['backward_mb']:.1f} MB (+{reg_prog['backward_mb'] - reg_prog['forward_mb']:.1f})")
    print(f"    + Step: {reg_prog['step_mb']:.1f} MB (+{reg_prog['step_mb'] - reg_prog['backward_mb']:.1f})")
    
    print("  Sharded optimizer:")
    shard_prog = sharded_rank0['memory_progression']
    print(f"    Baseline: {shard_prog['baseline_mb']:.1f} MB")
    print(f"    + Model: {shard_prog['model_mb']:.1f} MB (+{shard_prog['model_mb'] - shard_prog['baseline_mb']:.1f})")
    print(f"    + Optimizer: {shard_prog['optimizer_mb']:.1f} MB (+{shard_prog['optimizer_mb'] - shard_prog['model_mb']:.1f})")
    print(f"    + Batch: {shard_prog['batch_mb']:.1f} MB (+{shard_prog['batch_mb'] - shard_prog['optimizer_mb']:.1f})")
    print(f"    + Forward: {shard_prog['forward_mb']:.1f} MB (+{shard_prog['forward_mb'] - shard_prog['batch_mb']:.1f})")
    print(f"    + Backward: {shard_prog['backward_mb']:.1f} MB (+{shard_prog['backward_mb'] - shard_prog['forward_mb']:.1f})")
    print(f"    + Step: {shard_prog['step_mb']:.1f} MB (+{shard_prog['step_mb'] - shard_prog['backward_mb']:.1f})")
    
    print()
    print("MEMORY BREAKDOWN COMPARISON (Rank 0):")
    reg_breakdown = regular_rank0['memory_breakdown']
    shard_breakdown = sharded_rank0['memory_breakdown']
    
    print(f"  Component              Regular      Sharded      Difference")
    print(f"  {'-'*60}")
    print(f"  Model parameters:      {reg_breakdown['model_params_mb']:.1f} MB     {shard_breakdown['model_params_mb']:.1f} MB     {shard_breakdown['model_params_mb'] - reg_breakdown['model_params_mb']:.1f} MB")
    print(f"  Optimizer states:      {reg_breakdown['optimizer_states_mb']:.1f} MB     {shard_breakdown['optimizer_states_mb']:.1f} MB     {shard_breakdown['optimizer_states_mb'] - reg_breakdown['optimizer_states_mb']:.1f} MB")
    print(f"  Expected optimizer:    {reg_breakdown['expected_optimizer_mb']:.1f} MB     {shard_breakdown['expected_optimizer_mb']:.1f} MB     {shard_breakdown['expected_optimizer_mb'] - reg_breakdown['expected_optimizer_mb']:.1f} MB")
    print(f"  Activations:           {reg_breakdown['activations_mb']:.1f} MB     {shard_breakdown['activations_mb']:.1f} MB     {shard_breakdown['activations_mb'] - reg_breakdown['activations_mb']:.1f} MB")
    print(f"  Gradients:             {reg_breakdown['gradients_mb']:.1f} MB     {shard_breakdown['gradients_mb']:.1f} MB     {shard_breakdown['gradients_mb'] - reg_breakdown['gradients_mb']:.1f} MB")
    
    # Calculate efficiency metrics
    print()
    print("EFFICIENCY ANALYSIS:")
    
    # Theoretical vs actual optimizer memory
    reg_efficiency = (reg_breakdown['optimizer_states_mb'] / reg_breakdown['expected_optimizer_mb']) * 100
    shard_efficiency = (shard_breakdown['optimizer_states_mb'] / shard_breakdown['expected_optimizer_mb']) * 100
    
    print(f"  Regular optimizer efficiency: {reg_efficiency:.1f}% of expected memory")
    print(f"  Sharded optimizer efficiency: {shard_efficiency:.1f}% of expected memory")
    
    # Memory savings scaling
    theoretical_savings = (1 - 1/world_size) * 100
    actual_savings = (memory_diff / reg_peak) * 100 if memory_diff > 0 else 0
    scaling_efficiency = (actual_savings / theoretical_savings) * 100 if theoretical_savings > 0 else 0
    
    print(f"  Theoretical memory savings with {world_size} processes: {theoretical_savings:.1f}%")
    print(f"  Actual memory savings: {actual_savings:.1f}%")
    print(f"  Scaling efficiency: {scaling_efficiency:.1f}%")


def main():
    """Main function for multi-process memory profiling"""
    print("=" * 80)
    print("MULTI-PROCESS OPTIMIZER STATE SHARDING MEMORY ANALYSIS")
    print("=" * 80)
    
    # Configuration
    world_size = 8  # Use 8 processes for realistic distributed testing
    model_size = 'medium'  # Use medium model for reasonable memory usage
    
    print(f"World size: {world_size}")
    print(f"Model size: {model_size}")
    print()
    
    results = {}
    
    # Profile without sharded optimizer
    try:
        print(f"\n{'='*60}")
        print("PROFILING WITHOUT SHARDED OPTIMIZER")
        print(f"{'='*60}")
        results['regular'] = run_profile(world_size, False, model_size)
    except Exception as e:
        print(f"Error profiling regular optimizer: {e}")
        results['regular'] = None
    
    # Clear memory between tests
    time.sleep(3)
    
    # Profile with sharded optimizer  
    try:
        print(f"\n{'='*60}")
        print("PROFILING WITH SHARDED OPTIMIZER")
        print(f"{'='*60}")
        results['sharded'] = run_profile(world_size, True, model_size)
    except Exception as e:
        print(f"Error profiling sharded optimizer: {e}")
        results['sharded'] = None
    
    # Analyze results
    if results['regular'] and results['sharded']:
        analyze_memory_results(results['regular'], results['sharded'])
    else:
        print("Some profiling failed - check error messages above")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'timestamp': timestamp,
        'world_size': world_size,
        'model_size': model_size,
        'regular_results': results['regular'],
        'sharded_results': results['sharded']
    }
    
    os.makedirs("benchmark_results", exist_ok=True)
    filename = f"multiprocess_sharded_optimizer_profile_{timestamp}.json"
    filepath = os.path.join("benchmark_results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {filepath}")
    
    print(f"\nNOTES:")
    print(f"- This profile uses {world_size} processes with Gloo backend")
    print(f"- Sharded optimizer distributes optimizer states across processes")
    print(f"- Memory savings should scale approximately with number of processes")
    print(f"- Run on GPU systems for more realistic distributed training scenarios")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()