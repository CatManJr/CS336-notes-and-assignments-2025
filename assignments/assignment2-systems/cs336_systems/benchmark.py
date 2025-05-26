import argparse
import time
import torch
import numpy as np
from typing import Callable

def get_model(num_layers, d_model, vocab_size, context_length, num_heads, d_ff, device):
    """Initialize TransformerLM model with given hyperparameters"""
    from cs336_basics.transformer import TransformerLM
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    )
    return model

def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 10):
    """
    Benchmark `run` function by running it `num_trials` times, and return timing statistics.
    
    Args:
        description: Description of what is being benchmarked
        run: Function to benchmark
        num_warmups: Number of warmup runs (default: 5)
        num_trials: Number of measurement runs (default: 10)
    
    Returns:
        dict: Contains mean, std, and all times
    """
    # Warmup: first times might be slower due to compilation, things not cached.
    print(f"Running {num_warmups} warmup steps for {description}...")
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish

    # Time it for real now!
    print(f"Running {num_trials} measurement steps for {description}...")
    times = []
    for trial in range(num_trials):
        start_time = time.time()
        
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'mean': mean_time,
        'std': std_time,
        'times': times,
        'description': description
    }

def print_gpu_specs():
    """Print GPU specifications if CUDA is available"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    num_devices = torch.cuda.device_count()
    print(f"Found {num_devices} CUDA device(s)")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {properties.name}")
        print(f"  Memory: {properties.total_memory / 1e9:.1f} GB")
        print(f"  SMs: {properties.multi_processor_count}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark TransformerLM forward and backward passes')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=512, help='Maximum context length')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for benchmark')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--measure_steps', type=int, default=10, help='Number of measurement steps')
    parser.add_argument('--backward', action='store_true', help='Benchmark backward pass as well')
    parser.add_argument('--profile', action='store_true', help='Run profiling in addition to benchmarking')
    args = parser.parse_args()

    print("=" * 60)
    print("TRANSFORMER LANGUAGE MODEL BENCHMARK")
    print("=" * 60)
    
    # Print system information
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print_gpu_specs()
    else:
        print("Running on CPU")
    
    print(f"\nModel configuration:")
    print(f"  Layers: {args.num_layers}")
    print(f"  d_model: {args.d_model}")
    print(f"  Heads: {args.num_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Context length: {args.context_length}")
    
    print(f"\nBenchmark configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Measurement steps: {args.measure_steps}")
    print(f"  Include backward pass: {args.backward}")
    
    try:
        # Initialize model
        print(f"\nInitializing model...")
        model = get_model(
            num_layers=args.num_layers,
            d_model=args.d_model,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            device=args.device
        )
        model.train()

        # Generate random batch of data
        actual_seq_len = min(args.seq_len, args.context_length)
        print(f"Generating random data with shape ({args.batch_size}, {actual_seq_len})")
        x = torch.randint(0, args.vocab_size, (args.batch_size, actual_seq_len), device=args.device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, actual_seq_len), device=args.device)

        # Define the computation to benchmark
        def run_forward_backward():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, args.vocab_size), 
                y.view(-1)
            )
            if args.backward:
                loss.backward()
            model.zero_grad(set_to_none=True)
            return loss

        def run_forward_only():
            with torch.no_grad():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, args.vocab_size), 
                    y.view(-1)
                )
            return loss

        # Run benchmark
        print(f"\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        if args.backward:
            results = benchmark(
                "forward+backward", 
                run_forward_backward, 
                num_warmups=args.warmup_steps, 
                num_trials=args.measure_steps
            )
        else:
            results = benchmark(
                "forward only", 
                run_forward_only, 
                num_warmups=args.warmup_steps, 
                num_trials=args.measure_steps
            )

        # Print results
        print(f"\nOperation: {results['description']}")
        print(f"Average time: {results['mean']:.3f} ms")
        print(f"Standard deviation: {results['std']:.3f} ms")
        print(f"Min time: {min(results['times']):.3f} ms")
        print(f"Max time: {max(results['times']):.3f} ms")
        print(f"Coefficient of variation: {results['std']/results['mean']*100:.1f}%")
        
        if args.measure_steps <= 20:  # Only show individual times for small number of trials
            print(f"Individual times (ms): {[f'{t:.3f}' for t in results['times']]}")

        # Memory usage (if CUDA)
        if torch.cuda.is_available():
            print(f"\nGPU Memory usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        # Profiling (optional)
        if args.profile and torch.cuda.is_available():
            print(f"\n" + "=" * 60)
            print("PROFILING RESULTS")
            print("=" * 60)
            profile_model(run_forward_backward if args.backward else run_forward_only, 
                         "forward+backward" if args.backward else "forward")

    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        raise

def profile_model(run_func: Callable, description: str):
    """Profile the model using PyTorch profiler"""
    from torch.profiler import ProfilerActivity
    
    # Warmup
    run_func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False
    ) as prof:
        run_func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Print table
    table = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        max_name_column_width=60,
        row_limit=10
    )
    print(f"\nTop 10 operations for {description}:")
    print(table)

if __name__ == '__main__':
    main()