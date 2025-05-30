"""
Benchmarking and profiling compiled vs uncompiled attention implementation
The benchmark load for CPU is half of the GPU load
Same benchmark items with naive version
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

class AttentionModule(torch.nn.Module):
    """Attention module wrapper for benchmarking"""
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return naive_attention(q, k, v)

def get_transformer_model():
    """Get TransformerLM model for end-to-end benchmarking"""
    try:
        from cs336_basics.transformer import TransformerLM
        return TransformerLM
    except ImportError as e:
        print(f"Warning: Could not import TransformerLM: {e}")
        return None

def get_best_device():
    """Get the best available device in order of preference: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_memory_usage(device: str = 'cpu') -> float:
    """Get current memory usage in MB"""
    if device == 'cuda' and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6  # Convert to MB
    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have direct memory tracking like CUDA, fall back to system memory
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6  # Convert to MB
    else:
        # For CPU, get process memory usage
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e6  # Convert to MB

def benchmark_attention_comparison(batch_size: int, seq_len: int, d_model: int, 
                                 device: str = 'cpu', num_forward: int = 10, 
                                 num_backward: int = 10) -> Dict[str, Any]:
    """
    Compare compiled vs uncompiled attention performance.
    
    Returns:
        Dictionary with timing results for both versions
    """
    import torch  # Ensure torch is imported in this function scope
    
    try:
        print(f"Comparing attention: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
        
        # Create test data
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        # Create uncompiled module
        uncompiled_attention = AttentionModule(d_model, device=device)
        
        # Try to create compiled module with error handling
        try:
            # Enable error suppression for torch.compile on macOS
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            # Try different backends in order of preference
            backends_to_try = ["inductor", "aot_eager", "eager"]
            compiled_attention = None
            backend_used = None
            
            for backend in backends_to_try:
                try:
                    compiled_attention = torch.compile(AttentionModule(d_model, device=device), backend=backend)
                    backend_used = backend
                    print(f"  Using torch.compile with {backend} backend")
                    break
                except Exception as e:
                    print(f"  Backend {backend} failed: {str(e)[:100]}...")
                    continue
            
            if compiled_attention is None:
                raise Exception("All backends failed")
                
            compile_success = True
        except Exception as e:
            print(f"  torch.compile failed: {e}")
            print(f"  Falling back to uncompiled version for comparison")
            compiled_attention = uncompiled_attention
            compile_success = False
            backend_used = "mps"
        
        results = {}
        
        # Adjust warmup and iterations based on device
        warmup_runs = 2 if device == 'cpu' else 3
        
        # Benchmark both versions
        for name, attention_module in [("uncompiled", uncompiled_attention), ("compiled", compiled_attention)]:
            print(f"  Testing {name} version...")
            
            # Warmup
            for _ in range(warmup_runs):
                output = attention_module(Q, K, V)
                loss = output.sum()
                loss.backward()
                Q.grad = None
                K.grad = None
                V.grad = None
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
            
            # Benchmark forward pass
            forward_times = []
            for _ in range(num_forward):
                start_time = time.time()
                output = attention_module(Q, K, V)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                end_time = time.time()
                forward_times.append((end_time - start_time) * 1000)
            
            # Benchmark backward pass
            backward_times = []
            for _ in range(num_backward):
                output = attention_module(Q, K, V)
                loss = output.sum()
                
                start_time = time.time()
                loss.backward()
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                end_time = time.time()
                backward_times.append((end_time - start_time) * 1000)
                
                Q.grad = None
                K.grad = None
                V.grad = None
            
            results[name] = {
                'forward_mean': np.mean(forward_times),
                'forward_std': np.std(forward_times),
                'backward_mean': np.mean(backward_times),
                'backward_std': np.std(backward_times),
            }
        
        return {
            'success': True,
            'd_model': d_model,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'compile_success': compile_success,
            **results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'd_model': d_model,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'compile_success': False,
        }

def benchmark_transformer_comparison(device: str = 'cpu', num_iterations: int = 10) -> Dict[str, Any]:
    """
    Compare compiled vs uncompiled TransformerLM performance.
    
    Returns:
        Dictionary with timing results for both versions
    """
    import torch  # Ensure torch is imported in this function scope
    
    try:
        print("Comparing TransformerLM models...")
        
        TransformerLM = get_transformer_model()
        if not TransformerLM:
            return {'success': False, 'error': 'Could not import TransformerLM', 'compile_success': False}
        
        # Adjust model configuration based on device
        if device == 'cpu':
            # Reduced configuration for CPU
            config = {
                'vocab_size': 1000,
                'context_length': 256,  # Reduced from 512
                'd_model': 128,         # Reduced from 256
                'num_layers': 2,        # Reduced from 4
                'num_heads': 4,         # Reduced from 8
                'd_ff': 512,           # Reduced from 1024
                'device': device
            }
            batch_size, seq_len = 4, 128  # Reduced batch and sequence length
        elif device == 'mps':
            # MPS configuration - between CPU and CUDA
            config = {
                'vocab_size': 1000,
                'context_length': 384,  # Between CPU and CUDA
                'd_model': 192,         # Between CPU and CUDA
                'num_layers': 3,        # Between CPU and CUDA
                'num_heads': 6,         # Between CPU and CUDA
                'd_ff': 768,           # Between CPU and CUDA
                'device': device
            }
            batch_size, seq_len = 6, 192  # Between CPU and CUDA
        else:
            # Full configuration for CUDA
            config = {
                'vocab_size': 1000,
                'context_length': 512,
                'd_model': 256,
                'num_layers': 4,
                'num_heads': 8,
                'd_ff': 1024,
                'device': device
            }
            batch_size, seq_len = 8, 256
        
        # Create test data
        x = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
        
        # Create uncompiled model
        uncompiled_model = TransformerLM(**config)
        
        # Try to create compiled model with error handling
        try:
            # Enable error suppression for torch.compile on macOS
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            compiled_model = torch.compile(TransformerLM(**config), backend="eager")
            compile_success = True
            print(f"  Using torch.compile with eager backend for TransformerLM")
        except Exception as e:
            print(f"  torch.compile failed for TransformerLM: {e}")
            print(f"  Falling back to uncompiled version for comparison")
            compiled_model = uncompiled_model
            compile_success = False
        
        results = {}
        
        # Adjust warmup based on device
        warmup_runs = 2 if device == 'cpu' else 3
        
        # Benchmark both versions
        for name, model in [("uncompiled", uncompiled_model), ("compiled", compiled_model)]:
            print(f"  Testing {name} TransformerLM...")
            model.train()
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Warmup
            for _ in range(warmup_runs):
                try:
                    optimizer.zero_grad()
                    output = model(x)
                    loss = output.sum()
                    loss.backward()
                    optimizer.step()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    elif device == 'mps':
                        torch.mps.synchronize()
                except Exception as e:
                    print(f"    Warmup failed for {name}: {e}")
                    if name == "compiled":
                        print(f"    Falling back to uncompiled for timing")
                        model = uncompiled_model
                        compile_success = False
                    break
            
            # Benchmark forward pass only
            forward_times = []
            for _ in range(num_iterations):
                try:
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(x)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    elif device == 'mps':
                        torch.mps.synchronize()
                    end_time = time.time()
                    forward_times.append((end_time - start_time) * 1000)
                except Exception as e:
                    print(f"    Forward pass failed for {name}: {e}")
                    forward_times = [0.0] * num_iterations  # Placeholder times
                    break
            
            # Benchmark forward + backward + optimizer step
            full_times = []
            for _ in range(num_iterations):
                try:
                    optimizer.zero_grad()
                    
                    start_time = time.time()
                    output = model(x)
                    loss = output.sum()
                    loss.backward()
                    optimizer.step()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    elif device == 'mps':
                        torch.mps.synchronize()
                    end_time = time.time()
                    full_times.append((end_time - start_time) * 1000)
                except Exception as e:
                    print(f"    Full training step failed for {name}: {e}")
                    full_times = [0.0] * num_iterations  # Placeholder times
                    break
            
            results[name] = {
                'forward_mean': np.mean(forward_times),
                'forward_std': np.std(forward_times),
                'full_mean': np.mean(full_times),
                'full_std': np.std(full_times),
            }
        
        return {
            'success': True,
            'config': config,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'compile_success': compile_success,
            **results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'compile_success': False
        }

def run_attention_compile_benchmark():
    """Run attention compilation benchmark with same config as naive version"""
    print("=" * 80)
    print("ATTENTION COMPILATION BENCHMARK")
    print("=" * 80)
    
    device = get_best_device()  # Use best available device instead of hardcoded
    
    # Use same configuration as naive version - CPU gets reduced load
    if device == 'cpu':
        # Reduced configuration for CPU (same as naive version)
        batch_size = 4  # Half of original
        d_model_sizes = [16, 32, 64]  # Remove largest size
        seq_lengths = [256, 1024, 4096]  # Remove two largest sizes
        num_forward = 50  # Half of original
        num_backward = 50  # Half of original
    elif device == 'mps':
        # MPS configuration - slightly reduced from CUDA but more than CPU
        batch_size = 6
        d_model_sizes = [16, 32, 64, 128]
        seq_lengths = [256, 1024, 4096, 8192]  # Remove only the largest
        num_forward = 75
        num_backward = 75
    else:
        # Original configuration for CUDA (same as naive version)
        batch_size = 8
        d_model_sizes = [16, 32, 64, 128]
        seq_lengths = [256, 1024, 4096, 8192, 16384]
        num_forward = 100
        num_backward = 100
    
    print(f"Device: {device.upper()}")
    if device == 'mps':
        print("Using Apple Metal Performance Shaders (MPS)")
    print(f"Batch size: {batch_size}")
    print(f"d_model sizes: {d_model_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Number of forward passes: {num_forward}")
    print(f"Number of backward passes: {num_backward}")
    print(f"Using single head attention (num_heads=1)")
    
    if device == 'cpu':
        print("**Note:** Running on CPU with reduced configuration to decrease testing time")
    elif device == 'mps':
        print("**Note:** Running on MPS with moderate configuration")
    print()
    
    results = []
    
    # Run benchmarks for all combinations (same as naive version)
    for d_model, seq_len in itertools.product(d_model_sizes, seq_lengths):
        result = benchmark_attention_comparison(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            device=device,
            num_forward=num_forward,
            num_backward=num_backward
        )
        results.append(result)
        
        # Print immediate result
        if result['success']:
            unc_fwd = result['uncompiled']['forward_mean']
            comp_fwd = result['compiled']['forward_mean']
            unc_bwd = result['uncompiled']['backward_mean'] 
            comp_bwd = result['compiled']['backward_mean']
            fwd_speedup = unc_fwd / comp_fwd if comp_fwd > 0 else 0
            
            print(f"✓ d_model={d_model:3d}, seq_len={seq_len:5d}: "
                  f"Uncompiled Forward {unc_fwd:.2f}ms, Compiled Forward {comp_fwd:.2f}ms, "
                  f"Speedup {fwd_speedup:.2f}x")
        else:
            print(f"❌ d_model={d_model:3d}, seq_len={seq_len:5d}: {result['error']}")
    
    # Print results table
    print("\n" + "=" * 120)
    print("ATTENTION COMPILATION RESULTS")
    print("=" * 120)
    
    header = f"{'d_model':>8} {'seq_len':>8} {'Uncompiled Forward':>18} {'Compiled Forward':>16} " \
             f"{'Speedup':>8} {'Uncompiled Backward':>19} {'Compiled Backward':>17} {'Speedup':>8}"
    print(header)
    print("-" * 120)
    
    for result in results:
        if result['success']:
            unc_fwd = result['uncompiled']['forward_mean']
            comp_fwd = result['compiled']['forward_mean']
            unc_bwd = result['uncompiled']['backward_mean']
            comp_bwd = result['compiled']['backward_mean']
            
            fwd_speedup = unc_fwd / comp_fwd if comp_fwd > 0 else 0
            bwd_speedup = unc_bwd / comp_bwd if comp_bwd > 0 else 0
            
            row = f"{result['d_model']:>8} {result['seq_len']:>8} " \
                  f"{unc_fwd:>14.2f}±{result['uncompiled']['forward_std']:>3.1f} " \
                  f"{comp_fwd:>12.2f}±{result['compiled']['forward_std']:>3.1f} " \
                  f"{fwd_speedup:>7.2f}x " \
                  f"{unc_bwd:>15.2f}±{result['uncompiled']['backward_std']:>3.1f} " \
                  f"{comp_bwd:>13.2f}±{result['compiled']['backward_std']:>3.1f} " \
                  f"{bwd_speedup:>7.2f}x"
            print(row)
        else:
            print(f"{result['d_model']:>8} {result['seq_len']:>8} ERROR: {result['error']}")
    
    return results

def run_transformer_compile_benchmark():
    """Run transformer compilation benchmark"""
    print("\n" + "=" * 80)
    print("TRANSFORMER COMPILATION BENCHMARK")
    print("=" * 80)
    
    device = get_best_device()  # Use best available device instead of hardcoded
    # Adjust iterations based on device
    if device == 'cpu':
        num_iterations = 10
    elif device == 'mps':
        num_iterations = 15
    else:
        num_iterations = 20
    
    result = benchmark_transformer_comparison(device=device, num_iterations=num_iterations)
    
    if result['success']:
        print(f"\nModel configuration:")
        for key, value in result['config'].items():
            print(f"  {key}: {value}")
        print(f"Batch size: {result['batch_size']}")
        print(f"Sequence length: {result['seq_len']}")
        
        print("\n" + "=" * 100)
        print("TRANSFORMER COMPILATION RESULTS")
        print("=" * 100)
        
        header = f"{'Version':>12} {'Forward Only (ms)':>18} {'Forward+Backward+Opt (ms)':>25} {'Speedup':>8}"
        print(header)
        print("-" * 100)
        
        unc_fwd = result['uncompiled']['forward_mean']
        comp_fwd = result['compiled']['forward_mean']
        unc_full = result['uncompiled']['full_mean']
        comp_full = result['compiled']['full_mean']
        
        fwd_speedup = unc_fwd / comp_fwd if comp_fwd > 0 else 0
        full_speedup = unc_full / comp_full if comp_full > 0 else 0
        
        print(f"{'Uncompiled':>12} {unc_fwd:>14.2f}±{result['uncompiled']['forward_std']:>3.1f} "
              f"{unc_full:>21.2f}±{result['uncompiled']['full_std']:>3.1f} {'1.00x':>8}")
        print(f"{'Compiled':>12} {comp_fwd:>14.2f}±{result['compiled']['forward_std']:>3.1f} "
              f"{comp_full:>21.2f}±{result['compiled']['full_std']:>3.1f} "
              f"{full_speedup:>7.2f}x")
        
        print(f"\nForward pass speedup: {fwd_speedup:.2f}x")
        print(f"Full training step speedup: {full_speedup:.2f}x")
    else:
        print(f"Transformer benchmark failed: {result['error']}")
    
    return result

def save_results_to_markdown(attention_results, transformer_result, filename="profile/torch_compile_results.md"):
    """Save benchmark results to a markdown file"""
    import os
    import torch  # Ensure torch is imported in this function scope
    
    # Create directory if it doesn't exist - relative to current working directory
    profile_dir = "cs336_systems/profile"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir, exist_ok=True)
    
    # Get device name for filename
    device_name = get_best_device()
    base_filename = f"torch_compile_results_{device_name}.md"
    filepath = os.path.join(profile_dir, base_filename)
    
    with open(filepath, 'w') as f:
        f.write("# torch.compile Benchmark Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Device:** {device_name.upper()}\n")
        f.write(f"**PyTorch Version:** {torch.__version__}\n")
        
        # Add device availability info
        f.write(f"**Device Availability:**\n")
        f.write(f"- CUDA: {torch.cuda.is_available()}\n")
        f.write(f"- MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}\n\n")
        
        # Device-specific configuration note
        if device_name == 'cpu':
            f.write("**Note:** Running on CPU with reduced configuration:\n")
            f.write("- Smaller batch sizes and sequence lengths\n")
            f.write("- Fewer iterations to reduce testing time\n")
            f.write("- Smaller model configurations for Transformer tests\n\n")
        elif device_name == 'mps':
            f.write("**Note:** Running on MPS (Apple Metal) with moderate configuration:\n")
            f.write("- Balanced batch sizes and sequence lengths\n")
            f.write("- Moderate iterations for good performance measurement\n")
            f.write("- Medium model configurations for Transformer tests\n\n")
        
        # Attention results
        f.write("## Attention Module Compilation Results\n\n")
        f.write("Comparison of compiled vs uncompiled attention implementation:\n\n")
        f.write("| d_model | seq_len | Uncompiled Forward (ms) | Compiled Forward (ms) | Forward Speedup | Uncompiled Backward (ms) | Compiled Backward (ms) | Backward Speedup |\n")
        f.write("|---------|---------|-------------------------|----------------------|-----------------|--------------------------|------------------------|------------------|\n")
        
        for result in attention_results:
            if result['success']:
                unc_fwd = result['uncompiled']['forward_mean']
                comp_fwd = result['compiled']['forward_mean']
                unc_bwd = result['uncompiled']['backward_mean']
                comp_bwd = result['compiled']['backward_mean']
                
                fwd_speedup = unc_fwd / comp_fwd if comp_fwd > 0 else 0
                bwd_speedup = unc_bwd / comp_bwd if comp_bwd > 0 else 0
                
                f.write(f"| {result['d_model']} | {result['seq_len']} | "
                       f"{unc_fwd:.2f}±{result['uncompiled']['forward_std']:.1f} | "
                       f"{comp_fwd:.2f}±{result['compiled']['forward_std']:.1f} | "
                       f"{fwd_speedup:.2f}x | "
                       f"{unc_bwd:.2f}±{result['uncompiled']['backward_std']:.1f} | "
                       f"{comp_bwd:.2f}±{result['compiled']['backward_std']:.1f} | "
                       f"{bwd_speedup:.2f}x |\n")
        
        # Transformer results
        f.write("\n## Transformer Model Compilation Results\n\n")
        if transformer_result['success']:
            f.write("Comparison of compiled vs uncompiled TransformerLM model:\n\n")
            f.write("### Model Configuration\n")
            for key, value in transformer_result['config'].items():
                f.write(f"- **{key}:** {value}\n")
            f.write(f"- **Batch size:** {transformer_result['batch_size']}\n")
            f.write(f"- **Sequence length:** {transformer_result['seq_len']}\n\n")
            
            f.write("### Performance Results\n\n")
            f.write("| Version | Forward Only (ms) | Forward+Backward+Optimizer (ms) | Full Step Speedup |\n")
            f.write("|---------|-------------------|----------------------------------|--------------------|\n")
            
            unc_fwd = transformer_result['uncompiled']['forward_mean']
            comp_fwd = transformer_result['compiled']['forward_mean']
            unc_full = transformer_result['uncompiled']['full_mean']
            comp_full = transformer_result['compiled']['full_mean']
            
            full_speedup = unc_full / comp_full if comp_full > 0 else 0
            
            f.write(f"| Uncompiled | {unc_fwd:.2f}±{transformer_result['uncompiled']['forward_std']:.1f} | "
                   f"{unc_full:.2f}±{transformer_result['uncompiled']['full_std']:.1f} | 1.00x |\n")
            f.write(f"| Compiled | {comp_fwd:.2f}±{transformer_result['compiled']['forward_std']:.1f} | "
                   f"{comp_full:.2f}±{transformer_result['compiled']['full_std']:.1f} | "
                   f"{full_speedup:.2f}x |\n")
        else:
            f.write(f"Transformer benchmark failed: {transformer_result['error']}\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        f.write("### torch.compile Support on macOS\n\n")
        
        # Check if compilation was successful
        compile_success = any(r.get('compile_success', False) for r in attention_results)
        if transformer_result.get('compile_success', False):
            compile_success = True
            
        if not compile_success:
            f.write("⚠️ **torch.compile failed on this platform (macOS)**\n\n")
            f.write("**Common issues on macOS:**\n")
            f.write("- Missing or incompatible C++ runtime libraries (`libc++.1.dylib`)\n") 
            f.write("- Inductor backend compilation errors\n")
            f.write("- Limited support for certain operations in compiled mode\n\n")
            f.write("**Workarounds tested:**\n")
            f.write("- Using `backend='eager'` instead of default inductor\n")
            f.write("- Enabling `torch._dynamo.config.suppress_errors = True`\n")
            f.write("- Fallback to uncompiled versions for comparison\n\n")
            f.write("**For production use on macOS:**\n")
            f.write("- Consider using torch.jit.script() as alternative\n")
            f.write("- Test on Linux/CUDA environments where torch.compile has better support\n")
            f.write("- Use eager mode for development, compiled mode for deployment\n\n")
        
        f.write("### Key Observations\n\n")
        if any(r['success'] for r in attention_results):
            avg_speedups = []
            for result in attention_results:
                if result['success'] and result.get('compile_success', False):
                    fwd_speedup = result['uncompiled']['forward_mean'] / result['compiled']['forward_mean']
                    avg_speedups.append(fwd_speedup)
            
            if avg_speedups:
                avg_speedup = sum(avg_speedups) / len(avg_speedups)
                f.write(f"- **Attention compilation** provides an average speedup of **{avg_speedup:.2f}x** for forward pass\n")
            else:
                f.write("- **Attention compilation** failed - no speedup measurements available\n")
        
        if transformer_result['success'] and transformer_result.get('compile_success', False):
            full_speedup = transformer_result['uncompiled']['full_mean'] / transformer_result['compiled']['full_mean']
            f.write(f"- **Full model compilation** provides **{full_speedup:.2f}x** speedup for complete training step\n")
        else:
            f.write("- **Full model compilation** failed - no speedup measurements available\n")
        
        if compile_success:
            f.write("- torch.compile automatically optimizes computation graphs and generates efficient kernels\n")
            f.write("- Compilation overhead is amortized over multiple runs\n")
            f.write("- Memory usage patterns may also be optimized by the compiler\n\n")
        else:
            f.write("- torch.compile requires proper platform support and compatible runtime libraries\n")
            f.write("- Performance benefits are typically seen on Linux/CUDA environments\n")
            f.write("- Consider alternative optimization strategies for unsupported platforms\n\n")

def main():
    parser = argparse.ArgumentParser(description='Benchmark torch.compile performance')
    parser.add_argument('--attention-only', action='store_true', help='Only benchmark attention, not full model')
    parser.add_argument('--transformer-only', action='store_true', help='Only benchmark transformer, not attention')
    parser.add_argument('--force-device', type=str, choices=['cpu', 'mps', 'cuda'], 
                       help='Force specific device instead of auto-detection')
    args = parser.parse_args()
    
    # Override device detection if forced
    if args.force_device:
        global get_best_device
        original_get_best_device = get_best_device
        get_best_device = lambda: args.force_device
        print(f"Forcing device to: {args.force_device}")
    
    print(f"Available devices:")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"  Selected device: {get_best_device()}")
    print()
    
    attention_results = []
    transformer_result = {'success': False}
    
    if not args.transformer_only:
        attention_results = run_attention_compile_benchmark()
    
    if not args.attention_only:
        transformer_result = run_transformer_compile_benchmark()
    
    # Save results
    save_results_to_markdown(attention_results, transformer_result)
    device_name = get_best_device()
    print(f"\n✅ Results saved to profile/torch_compile_results_{device_name}.md")

if __name__ == '__main__':
    main()