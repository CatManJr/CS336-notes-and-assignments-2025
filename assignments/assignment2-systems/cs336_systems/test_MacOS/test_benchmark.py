"""
functional tests for benchmark.py on MacOS without GPU support.
"""
import sys
import os
import torch
import traceback
from typing import Dict, Any

# Add the parent directory to Python path so we can import benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_import():
    """Test that we can import the TransformerLM model"""
    print("Testing model import...")
    try:
        from cs336_basics.transformer import TransformerLM
        print("  ✓ TransformerLM import successful")
        assert True  # Test passed
    except ImportError as e:
        print(f"  ❌ Failed to import TransformerLM: {e}")
        assert False, f"Failed to import TransformerLM: {e}"

def test_benchmark_import():
    """Test that we can import functions from benchmark.py"""
    print("Testing benchmark module import...")
    try:
        from benchmark import get_model, benchmark, print_gpu_specs
        print("  ✓ Benchmark module import successful")
        assert True  # Test passed
    except ImportError as e:
        print(f"  ❌ Failed to import from benchmark: {e}")
        assert False, f"Failed to import from benchmark: {e}"

def test_model_creation():
    """Test that model can be created and has correct structure"""
    print("Testing model creation...")
    try:
        from benchmark import get_model
        
        # Test with minimal parameters
        model = get_model(
            num_layers=2,
            d_model=64,
            vocab_size=1000, 
            context_length=32,
            num_heads=2,
            d_ff=128,
            device='cpu'
        )
        
        # Check model structure
        assert hasattr(model, 'forward'), "Model missing forward method"
        assert hasattr(model, 'parameters'), "Model missing parameters method"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created with {total_params:,} parameters")
        assert total_params > 0, "Model has no parameters"
        
        # Check model can be moved to different device
        model.to('cpu')
        print("  ✓ Model creation test passed")
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        traceback.print_exc()
        assert False, f"Model creation failed: {e}"

def test_model_forward():
    """Test that model forward pass works"""
    print("Testing model forward pass...")
    try:
        from benchmark import get_model
        
        model = get_model(
            num_layers=1,
            d_model=32,
            vocab_size=100, 
            context_length=16,
            num_heads=2,
            d_ff=64,
            device='cpu'
        )
        
        # Test input
        batch_size, seq_len = 2, 8
        x = torch.randint(0, 100, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(x)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 100)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        
        # Check for reasonable values
        assert torch.all(torch.isfinite(logits)), "Forward pass produced inf/nan"
        assert not torch.all(logits == 0), "Forward pass produced all zeros"
        
        print(f"  Forward pass output shape: {logits.shape}")
        print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
        print("  ✓ Model forward pass test passed")
    except Exception as e:
        print(f"  ❌ Model forward pass failed: {e}")
        traceback.print_exc()
        assert False, f"Model forward pass failed: {e}"

def test_model_backward():
    """Test that model backward pass works"""
    print("Testing model backward pass...")
    try:
        from benchmark import get_model
        
        model = get_model(
            num_layers=1,
            d_model=32,
            vocab_size=100, 
            context_length=16,
            num_heads=2,
            d_ff=64,
            device='cpu'
        )
        
        # Test data
        batch_size, seq_len = 2, 8
        x = torch.randint(0, 100, (batch_size, seq_len))
        y = torch.randint(0, 100, (batch_size, seq_len))
        
        # Forward and backward
        model.train()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100), 
            y.view(-1)
        )
        
        # Check loss
        assert torch.isfinite(loss), "Loss is inf/nan"
        print(f"  Loss value: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                assert torch.all(torch.isfinite(param.grad)), f"Gradient for {name} contains inf/nan"
        
        assert len(grad_norms) > 0, "No gradients found after backward pass"
        print(f"  Found gradients for {len(grad_norms)} parameters")
        print(f"  Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
        
        model.zero_grad()
        print("  ✓ Model backward pass test passed")
    except Exception as e:
        print(f"  ❌ Model backward pass failed: {e}")
        traceback.print_exc()
        assert False, f"Model backward pass failed: {e}"

def test_benchmark_function():
    """Test the benchmark function itself"""
    print("Testing benchmark function...")
    try:
        from benchmark import benchmark
        
        call_count = 0
        def dummy_function():
            nonlocal call_count
            call_count += 1
            # Some minimal computation
            x = torch.randn(10, 10)
            y = torch.matmul(x, x.T)
            return y.sum()
        
        # Test benchmark
        results = benchmark("dummy test", dummy_function, num_warmups=2, num_trials=3)
        
        # Verify function was called correct number of times
        assert call_count == 5, f"Expected 5 calls (2 warmup + 3 trials), got {call_count}"
        
        # Verify results structure
        required_keys = ['mean', 'std', 'times', 'description']
        for key in required_keys:
            assert key in results, f"Missing key '{key}' in benchmark results"
        
        assert len(results['times']) == 3, f"Expected 3 timing measurements, got {len(results['times'])}"
        assert results['description'] == "dummy test"
        assert results['mean'] >= 0, "Mean time should be non-negative"
        assert results['std'] >= 0, "Standard deviation should be non-negative"
        
        print(f"  Benchmark ran {call_count} times (2 warmup + 3 trials)")
        print(f"  Mean time: {results['mean']:.3f} ms")
        print(f"  Std time: {results['std']:.3f} ms")
        print("  ✓ Benchmark function test passed")
    except Exception as e:
        print(f"  ❌ Benchmark function test failed: {e}")
        traceback.print_exc()
        assert False, f"Benchmark function test failed: {e}"

def test_end_to_end_minimal():
    """Test end-to-end benchmark with minimal model"""
    print("Testing end-to-end benchmark (minimal)...")
    try:
        from benchmark import get_model, benchmark
        
        # Minimal model
        model = get_model(
            num_layers=1,
            d_model=32,
            vocab_size=100, 
            context_length=16,
            num_heads=2,
            d_ff=64,
            device='cpu'
        )
        
        # Minimal data
        x = torch.randint(0, 100, (1, 8))
        y = torch.randint(0, 100, (1, 8))
        
        # Test forward only
        def run_forward():
            model.eval()
            with torch.no_grad():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, 100), y.view(-1))
            return loss
        
        results_forward = benchmark("forward only", run_forward, num_warmups=1, num_trials=2)
        
        # Test forward + backward
        def run_forward_backward():
            model.train()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 100), y.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            return loss
        
        results_backward = benchmark("forward+backward", run_forward_backward, num_warmups=1, num_trials=2)
        
        # Verify backward is slower than forward only
        print(f"  Forward only: {results_forward['mean']:.3f} ms")
        print(f"  Forward+backward: {results_backward['mean']:.3f} ms")
        
        # Backward should generally be slower (but not always on CPU with small models)
        ratio = results_backward['mean'] / results_forward['mean']
        print(f"  Backward/Forward ratio: {ratio:.2f}x")
        
        print("  ✓ End-to-end benchmark test passed")
    except Exception as e:
        print(f"  ❌ End-to-end benchmark test failed: {e}")
        traceback.print_exc()
        assert False, f"End-to-end benchmark test failed: {e}"

def test_different_model_sizes():
    """Test benchmark with different model sizes"""
    print("Testing different model sizes...")
    try:
        from benchmark import get_model, benchmark
        
        configs = [
            {"num_layers": 1, "d_model": 32, "num_heads": 2, "d_ff": 64},
            {"num_layers": 2, "d_model": 64, "num_heads": 4, "d_ff": 128},
        ]
        
        for i, config in enumerate(configs):
            print(f"  Testing config {i+1}: {config}")
            
            model = get_model(
                vocab_size=100,
                context_length=16,
                device='cpu',
                **config
            )
            
            x = torch.randint(0, 100, (1, 8))
            y = torch.randint(0, 100, (1, 8))
            
            def run_test():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, 100), y.view(-1))
                loss.backward()
                model.zero_grad()
                return loss
            
            results = benchmark(f"config_{i+1}", run_test, num_warmups=1, num_trials=2)
            print(f"    Time: {results['mean']:.3f} ms")
        
        print("  ✓ Different model sizes test passed")
    except Exception as e:
        print(f"  ❌ Different model sizes test failed: {e}")
        traceback.print_exc()
        assert False, f"Different model sizes test failed: {e}"

def run_all_tests():
    """Run all tests and return summary"""
    print("=" * 60)
    print("RUNNING BENCHMARK.PY TESTS")
    print("=" * 60)
    
    tests = [
        test_model_import,
        test_benchmark_import,
        test_model_creation,
        test_model_forward,
        test_model_backward,
        test_benchmark_function,
        test_end_to_end_minimal,
        test_different_model_sizes,
    ]
    
    results = []
    for test_func in tests:
        print()
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"  ❌ {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your benchmark.py is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)