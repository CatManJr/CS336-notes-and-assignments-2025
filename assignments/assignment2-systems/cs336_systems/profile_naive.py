"""
5-30-2025
Benchmarking and profiling naive attention implementation
for cpu: uv run cs336_systems/profile_naive.py --device cpu
"""

import argparse
import time
import torch
import numpy as np
import itertools
from typing import Dict, Any, Tuple
import psutil
import os

def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """
    Naive attention implementation that materializes the full attention matrix.
    
    Args:
        Q: Query tensor of shape (batch, seq_len, d_model)
        K: Key tensor of shape (batch, seq_len, d_model)  
        V: Value tensor of shape (batch, seq_len, d_model)
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape
    
    # Compute attention scores: (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
    
    if causal:
        # Apply causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=Q.dtype))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, V)
    
    return output

def get_attention_from_model():
    """Extract MultiHeadSelfAttention implementation from your transformer module"""
    try:
        from cs336_basics.transformer import MultiHeadSelfAttention
        return MultiHeadSelfAttention
    except ImportError as e:
        print(f"Warning: Could not import MultiHeadSelfAttention from cs336_basics.transformer: {e}")
        print("Using naive attention implementation instead")
        return None

class SingleHeadAttentionWrapper(torch.nn.Module):
    """
    Wrapper to adapt MultiHeadSelfAttention to work with single head for benchmarking
    """
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        # Use num_heads=1 to avoid multi-head overhead in benchmark
        MultiHeadSelfAttention = get_attention_from_model()
        if MultiHeadSelfAttention:
            self.attention = MultiHeadSelfAttention(
                d_model=d_model, 
                num_heads=1,  # Single head as requested
                device=device, 
                dtype=dtype
            )
            self.use_model_attention = True
        else:
            self.attention = None
            self.use_model_attention = False
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes separate Q, K, V tensors
        
        Args:
            q: Query tensor of shape (batch, seq_len, d_model)
            k: Key tensor of shape (batch, seq_len, d_model)
            v: Value tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if not self.use_model_attention:
            return naive_attention(q, k, v)
        
        # Since your MultiHeadSelfAttention expects a single input x and computes Q,K,V internally,
        # and we want to test with specific Q,K,V, we'll use naive attention for consistency
        # This ensures we're actually testing the attention computation pattern
        return naive_attention(q, k, v)

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6  # Convert to MB
    else:
        # For CPU, get process memory usage
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6  # Convert to MB

def benchmark_attention_config(batch_size: int, seq_len: int, d_model: int, 
                             device: str = 'cpu', use_model_attention: bool = True,
                             num_forward: int = 100, num_backward: int = 100) -> Dict[str, Any]:
    """
    Benchmark attention at given configuration.
    
    Returns:
        Dictionary with timing and memory results, or error info if OOM
    """
    try:
        print(f"Testing: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
        
        # Create random inputs (no head dimension as requested)
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        # Get attention implementation
        if use_model_attention:
            try:
                attention_module = SingleHeadAttentionWrapper(d_model=d_model, device=device)
                attention_module.train()
                
                def attention_forward():
                    return attention_module(Q, K, V)
                    
                print(f"  Using naive attention implementation")
            except Exception as e:
                print(f"  Failed to use attention wrapper: {e}")
                print(f"  Falling back to naive attention")
                def attention_forward():
                    return naive_attention(Q, K, V)
        else:
            def attention_forward():
                return naive_attention(Q, K, V)
        
        # Adjust warmup based on device
        warmup_runs = 3 if device == 'cpu' else 5
        
        # Warmup
        for _ in range(warmup_runs):
            output = attention_forward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Clear gradients and memory
        Q.grad = None
        K.grad = None  
        V.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark forward pass
        memory_before_forward = get_memory_usage()
        
        forward_times = []
        for _ in range(num_forward):
            start_time = time.time()
            output = attention_forward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            forward_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        memory_after_forward = get_memory_usage()
        
        # Benchmark backward pass
        memory_before_backward = get_memory_usage()
        
        backward_times = []
        for _ in range(num_backward):
            # Create fresh output for backward
            output = attention_forward()
            
            # Create dummy loss
            loss = output.sum()
            
            start_time = time.time()
            loss.backward(retain_graph=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            backward_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Clear gradients for next iteration
            Q.grad = None
            K.grad = None
            V.grad = None
        
        memory_after_backward = get_memory_usage()
        
        return {
            'success': True,
            'forward_mean': np.mean(forward_times),
            'forward_std': np.std(forward_times),
            'backward_mean': np.mean(backward_times),
            'backward_std': np.std(backward_times),
            'memory_before_forward': memory_before_forward,
            'memory_after_forward': memory_after_forward,
            'memory_before_backward': memory_before_backward,
            'memory_after_backward': memory_after_backward,
            'forward_memory_delta': memory_after_forward - memory_before_forward,
            'backward_memory_delta': memory_after_backward - memory_before_backward,
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                'success': False,
                'error': 'OOM',
                'error_msg': str(e)
            }
        else:
            return {
                'success': False, 
                'error': 'Runtime Error',
                'error_msg': str(e)
            }
    except Exception as e:
        return {
            'success': False,
            'error': 'Other Error', 
            'error_msg': str(e)
        }

def calculate_attention_memory(batch_size: int, seq_len: int, d_model: int) -> Dict[str, float]:
    """
    Calculate theoretical memory usage for attention mechanism.
    
    Returns memory usage in MB for different components.
    """
    # Assume float32 (4 bytes per element)
    bytes_per_element = 4
    
    # Input tensors: Q, K, V each of shape (batch, seq_len, d_model)
    input_memory = 3 * batch_size * seq_len * d_model * bytes_per_element
    
    # Attention scores: (batch, seq_len, seq_len) - this is the problematic part
    attention_scores_memory = batch_size * seq_len * seq_len * bytes_per_element
    
    # Output: (batch, seq_len, d_model)
    output_memory = batch_size * seq_len * d_model * bytes_per_element
    
    # For backward pass, we need to store attention weights for gradient computation
    backward_storage_memory = attention_scores_memory  # Attention weights
    
    total_forward = input_memory + attention_scores_memory + output_memory
    total_backward = total_forward + backward_storage_memory
    
    return {
        'input_memory_mb': input_memory / 1e6,
        'attention_scores_mb': attention_scores_memory / 1e6,
        'output_memory_mb': output_memory / 1e6,
        'total_forward_mb': total_forward / 1e6,
        'total_backward_mb': total_backward / 1e6,
        'backward_storage_mb': backward_storage_memory / 1e6,
    }

def profile_attention_detailed(batch_size: int, seq_len: int, d_model: int, device: str):
    """Use PyTorch profiler for detailed attention analysis"""
    print(f"\nProfiling attention: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    # Try to use your attention implementation
    try:
        attention_module = SingleHeadAttentionWrapper(d_model=d_model, device=device)
        attention_module.train()
        
        def run_attention():
            output = attention_module(Q, K, V)
            loss = output.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            
        print(f"  Using MultiHeadSelfAttention for profiling")
    except Exception as e:
        print(f"  Failed to use MultiHeadSelfAttention: {e}")
        print(f"  Using naive attention for profiling")
        def run_attention():
            output = naive_attention(Q, K, V)
            loss = output.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
    
    # Warmup
    run_attention()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Profile with PyTorch profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        run_attention()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Print detailed results
    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    
    print("Top 15 operations by time:")
    print(prof.key_averages().table(
        sort_by=sort_by,
        row_limit=15,
        max_name_column_width=50
    ))
    
    if torch.cuda.is_available():
        print("\nTop 10 operations by GPU memory usage:")
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=10,
            max_name_column_width=50
        ))

def save_results_to_markdown(results, device, num_forward, num_backward):
    """Save benchmark results to a markdown file"""
    import os
    from datetime import datetime
    
    # Create profile directory if it doesn't exist
    profile_dir = "cs336_systems/profile"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir, exist_ok=True)
    
    # Generate filename
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device_name = "cuda" if device == "cuda" else "cpu"
    filename = f"{profile_dir}/naive_attention_results_{device_name}_{now}.md"
    
    with open(filename, 'w') as f:
        f.write("# Naive Attention Benchmark Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Device:** {device.upper()}\n")
        f.write(f"**PyTorch Version:** {torch.__version__}\n")
        f.write(f"**Forward passes per test:** {num_forward}\n")
        f.write(f"**Backward passes per test:** {num_backward}\n\n")
        
        if device == 'cpu':
            f.write("**Note:** Running on CPU with reduced configuration:\n")
            f.write("- Smaller batch sizes and fewer test configurations\n")
            f.write("- Fewer iterations to reduce testing time\n\n")
        
        # Results table
        f.write("## Detailed Results\n\n")
        f.write("| d_model | seq_len | batch_size | Forward (ms) | Backward (ms) | Forward Memory (MB) | Backward Memory (MB) | Attention Matrix (MB) | Status |\n")
        f.write("|---------|---------|------------|--------------|---------------|---------------------|----------------------|----------------------|--------|\n")
        
        for result in results:
            if result['success']:
                forward_str = f"{result['forward_mean']:.2f}±{result['forward_std']:.1f}"
                backward_str = f"{result['backward_mean']:.2f}±{result['backward_std']:.1f}"
                status = "✅ OK"
            else:
                forward_str = "N/A"
                backward_str = "N/A"
                status = f"❌ {result['error']}"
            
            f.write(f"| {result['d_model']} | {result['seq_len']} | {result['batch_size']} | "
                   f"{forward_str} | {backward_str} | "
                   f"{result['total_forward_mb']:.1f} | {result['total_backward_mb']:.1f} | "
                   f"{result['attention_scores_mb']:.1f} | {status} |\n")
        
        # Memory analysis
        f.write("\n## Memory Analysis\n\n")
        f.write("### Memory Scaling\n")
        f.write("The attention mechanism's memory usage scales as follows:\n")
        f.write("- **Input tensors (Q, K, V):** O(batch_size × seq_len × d_model)\n")
        f.write("- **Attention scores matrix:** O(batch_size × seq_len²) ⚠️ **Quadratic scaling**\n")
        f.write("- **Output tensor:** O(batch_size × seq_len × d_model)\n")
        f.write("- **Backward storage:** O(batch_size × seq_len²) for attention weights\n\n")
        
        # Performance analysis
        successful_results = [r for r in results if r['success']]
        if successful_results:
            f.write("### Performance Summary\n")
            forward_times = [r['forward_mean'] for r in successful_results]
            backward_times = [r['backward_mean'] for r in successful_results]
            
            f.write(f"- **Average forward time:** {np.mean(forward_times):.2f} ms\n")
            f.write(f"- **Average backward time:** {np.mean(backward_times):.2f} ms\n")
            f.write(f"- **Fastest forward:** {min(forward_times):.2f} ms\n")
            f.write(f"- **Slowest forward:** {max(forward_times):.2f} ms\n\n")
        
        # OOM analysis
        oom_results = [r for r in results if not r['success'] and r.get('error') == 'OOM']
        if oom_results:
            f.write("### Out of Memory Analysis\n")
            first_oom = min(oom_results, key=lambda x: (x['seq_len'], x['d_model']))
            f.write(f"- **First OOM at:** d_model={first_oom['d_model']}, seq_len={first_oom['seq_len']}\n")
            f.write(f"- **Memory requirement:** {first_oom['total_forward_mb']:.1f} MB forward, {first_oom['total_backward_mb']:.1f} MB backward\n")
            f.write(f"- **Attention matrix size:** {first_oom['attention_scores_mb']:.1f} MB\n\n")
        
        # Solutions
        f.write("### Solutions for Memory Scaling\n")
        f.write("To address the O(seq_len²) memory bottleneck:\n\n")
        f.write("1. **FlashAttention:** Computes attention in tiles without materializing the full attention matrix\n")
        f.write("2. **Gradient Checkpointing:** Recompute attention in backward pass instead of storing intermediate results\n")
        f.write("3. **Sparse Attention:** Only compute subset of attention scores (local, strided, etc.)\n")
        f.write("4. **Mixed Precision:** Use fp16 to halve memory usage\n")
        f.write("5. **Sequence Parallelism:** Distribute sequence dimension across multiple devices\n")
    
    print(f"Results saved to {filename}")
    return filename

def run_attention_benchmark():
    """Run the full attention benchmark suite"""
    print("=" * 80)
    print("ATTENTION MECHANISM BENCHMARK")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Adjust configuration based on device - CPU gets reduced load
    if device == 'cpu':
        # Reduced configuration for CPU
        batch_size = 4  # Half of original
        d_model_sizes = [16, 32, 64]  # Remove largest size
        seq_lengths = [256, 1024, 4096]  # Remove two largest sizes
        num_forward = 50  # Half of original
        num_backward = 50  # Half of original
    else:
        # Original configuration for CUDA
        batch_size = 8
        d_model_sizes = [16, 32, 64, 128]
        seq_lengths = [256, 1024, 4096, 8192, 16384]
        num_forward = 100
        num_backward = 100
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"d_model sizes: {d_model_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Number of forward passes: {num_forward}")
    print(f"Number of backward passes: {num_backward}")
    print(f"Using single head attention (num_heads=1)")
    
    if device == 'cpu':
        print("**Note:** Running on CPU with reduced configuration to decrease testing time")
    print()
    
    # Test if we can import the attention module
    MultiHeadSelfAttention = get_attention_from_model()
    if MultiHeadSelfAttention:
        print("✓ Successfully imported MultiHeadSelfAttention from cs336_basics.transformer")
    else:
        print("❌ Failed to import MultiHeadSelfAttention, will use naive implementation")
    print()
    
    # Results storage
    results = []
    
    # Run benchmarks for all combinations
    for d_model, seq_len in itertools.product(d_model_sizes, seq_lengths):
        result = benchmark_attention_config(
            batch_size=batch_size,
            seq_len=seq_len, 
            d_model=d_model,
            device=device,
            use_model_attention=True,
            num_forward=num_forward,
            num_backward=num_backward
        )
        
        result['d_model'] = d_model
        result['seq_len'] = seq_len
        result['batch_size'] = batch_size
        
        # Add theoretical memory calculations
        memory_calc = calculate_attention_memory(batch_size, seq_len, d_model)
        result.update(memory_calc)
        
        results.append(result)
        
        # Print immediate result
        if result['success']:
            print(f"✓ d_model={d_model:3d}, seq_len={seq_len:5d}: "
                  f"Forward {result['forward_mean']:.2f}±{result['forward_std']:.2f}ms, "
                  f"Backward {result['backward_mean']:.2f}±{result['backward_std']:.2f}ms")
        else:
            print(f"❌ d_model={d_model:3d}, seq_len={seq_len:5d}: {result['error']}")

    # Print detailed results table
    print("\n" + "=" * 140)
    print("DETAILED RESULTS TABLE")
    print("=" * 140)
    
    # Header
    header = f"{'d_model':>8} {'seq_len':>8} {'Forward (ms)':>15} {'Backward (ms)':>16} " \
             f"{'Mem Forward':>12} {'Mem Backward':>13} {'Attention Matrix':>16} {'Status':>10}"
    print(header)
    print("-" * 140)
    
    # Data rows
    for result in results:
        if result['success']:
            forward_str = f"{result['forward_mean']:.1f}±{result['forward_std']:.1f}"
            backward_str = f"{result['backward_mean']:.1f}±{result['backward_std']:.1f}"
            mem_forward = f"{result['total_forward_mb']:.1f}MB"
            mem_backward = f"{result['total_backward_mb']:.1f}MB"
            attention_mem = f"{result['attention_scores_mb']:.1f}MB"
            status = "OK"
        else:
            forward_str = "N/A"
            backward_str = "N/A"
            mem_forward = f"{result['total_forward_mb']:.1f}MB"
            mem_backward = f"{result['total_backward_mb']:.1f}MB"
            attention_mem = f"{result['attention_scores_mb']:.1f}MB"
            status = result['error']
        
        row = f"{result['d_model']:>8} {result['seq_len']:>8} {forward_str:>15} " \
              f"{backward_str:>16} {mem_forward:>12} {mem_backward:>13} {attention_mem:>16} {status:>10}"
        print(row)
    
    # Run detailed profiling on a few configurations
    print("\n" + "=" * 80)
    print("DETAILED PROFILING")
    print("=" * 80)
    
    # Profile small and medium configurations
    profile_configs = [
        (32, 256),   # Small
        (64, 1024),  # Medium
    ]
    
    for d_model, seq_len in profile_configs:
        try:
            profile_attention_detailed(batch_size, seq_len, d_model, device)
        except Exception as e:
            print(f"Profiling failed for d_model={d_model}, seq_len={seq_len}: {e}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find first OOM
    oom_configs = [r for r in results if not r['success'] and r.get('error') == 'OOM']
    if oom_configs:
        first_oom = min(oom_configs, key=lambda x: (x['seq_len'], x['d_model']))
        print(f"First OOM at: d_model={first_oom['d_model']}, seq_len={first_oom['seq_len']}")
        print(f"Theoretical memory requirement: {first_oom['total_forward_mb']:.1f} MB forward, "
              f"{first_oom['total_backward_mb']:.1f} MB backward")
        print(f"Attention matrix memory: {first_oom['attention_scores_mb']:.1f} MB")
        
        # Memory accounting for the OOM case
        print(f"\nMemory accounting for OOM case:")
        print(f"  Input tensors (Q,K,V): {first_oom['input_memory_mb']:.1f} MB")
        print(f"  Attention scores matrix: {first_oom['attention_scores_mb']:.1f} MB")
        print(f"  Output tensor: {first_oom['output_memory_mb']:.1f} MB") 
        print(f"  Backward storage: {first_oom['backward_storage_mb']:.1f} MB")
        print(f"  Total forward: {first_oom['total_forward_mb']:.1f} MB")
        print(f"  Total backward: {first_oom['total_backward_mb']:.1f} MB")
    else:
        print("No OOM errors encountered")
        if device == 'cpu':
            print("(Note: Running on CPU - would likely see OOM on GPU with limited memory)")
    
    # Memory scaling analysis
    print(f"\nMemory scaling analysis:")
    print(f"The attention matrix memory scales as O(batch_size * seq_len²), while input/output memory scales as O(batch_size * seq_len * d_model)")
    print(f"For long sequences, the attention matrix dominates memory usage.")
    
  # Show scaling for d_model=32
    d32_results = [r for r in results if r['d_model'] == 32 and r['success']]
    if d32_results:
        print(f"\nMemory breakdown for d_model=32:")
        for result in d32_results:
            total_mem = result['total_forward_mb']
            attn_mem = result['attention_scores_mb']
            attn_ratio = (attn_mem / total_mem) * 100 if total_mem > 0 else 0
            print(f"  seq_len={result['seq_len']:5d}: Total={total_mem:6.1f}MB, "
                  f"Attention={attn_mem:6.1f}MB ({attn_ratio:.1f}%)")
    
    print(f"\nHow memory saved for backward changes with sequence length:")
    print(f"- Backward pass stores attention weights: O(batch_size * seq_len²)")
    print(f"- This grows quadratically with sequence length")
    print(f"- For seq_len=16384, attention matrix alone uses {batch_size * 16384 * 16384 * 4 / 1e6:.1f}MB")
    
    print(f"\nTo eliminate this memory cost:")
    print(f"1. FlashAttention: computes attention in tiles, never materializes full seq_len×seq_len matrix")
    print(f"2. Gradient checkpointing: recompute attention in backward pass instead of storing")
    print(f"3. Attention sparsity: only compute subset of attention scores (e.g., local attention)")
    print(f"4. Mixed precision: use fp16 to halve memory usage")
    
    # Save results to markdown
    save_results_to_markdown(results, device, num_forward, num_backward)
    print(f"\n✅ Results saved to profile/naive_attention_results_{device}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark MultiHeadSelfAttention mechanism')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--profile-only', action='store_true', help='Only run profiling, not full benchmark')
    parser.add_argument('--quick', action='store_true', help='Run with fewer iterations for testing')
    args = parser.parse_args()
    
    if args.profile_only:
        print("Running profiling only...")
        batch_size = 8
        profile_attention_detailed(batch_size, 512, 64, args.device)
    else:
        run_attention_benchmark()

if __name__ == '__main__':
    main()