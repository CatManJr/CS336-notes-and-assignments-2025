# torch.compile Benchmark Results

**Date:** 2025-05-30 20:59:08
**Device:** MPS
**PyTorch Version:** 2.6.0
**Device Availability:**
- CUDA: False
- MPS: True

**Note:** Running on MPS (Apple Metal) with moderate configuration:
- Balanced batch sizes and sequence lengths
- Moderate iterations for good performance measurement
- Medium model configurations for Transformer tests

## Attention Module Compilation Results

Comparison of compiled vs uncompiled attention implementation:

| d_model | seq_len | Uncompiled Forward (ms) | Compiled Forward (ms) | Forward Speedup | Uncompiled Backward (ms) | Compiled Backward (ms) | Backward Speedup |
|---------|---------|-------------------------|----------------------|-----------------|--------------------------|------------------------|------------------|
| 16 | 256 | 0.64±0.3 | 0.66±0.4 | 0.98x | 0.53±0.1 | 0.55±0.0 | 0.96x |
| 16 | 1024 | 3.28±0.1 | 3.25±0.1 | 1.01x | 7.20±0.2 | 7.12±0.2 | 1.01x |
| 16 | 4096 | 49.20±1.1 | 48.26±1.1 | 1.02x | 111.94±1.8 | 112.13±1.8 | 1.00x |
| 16 | 8192 | 227.56±7.4 | 225.97±4.1 | 1.01x | 505.68±9.0 | 498.95±10.6 | 1.01x |
| 32 | 256 | 0.59±0.2 | 0.50±0.0 | 1.19x | 0.56±0.0 | 0.55±0.0 | 1.01x |
| 32 | 1024 | 3.29±0.0 | 3.27±0.0 | 1.01x | 7.18±0.1 | 7.16±0.0 | 1.00x |
| 32 | 4096 | 47.79±0.4 | 49.92±1.9 | 0.96x | 111.04±0.3 | 120.71±8.0 | 0.92x |
| 32 | 8192 | 228.57±4.1 | 221.01±6.7 | 1.03x | 516.83±5.7 | 493.88±1.2 | 1.05x |
| 64 | 256 | 0.55±0.1 | 0.51±0.0 | 1.07x | 0.57±0.0 | 0.58±0.0 | 0.99x |
| 64 | 1024 | 3.43±0.1 | 3.42±0.1 | 1.00x | 7.67±0.2 | 7.63±0.2 | 1.01x |
| 64 | 4096 | 51.49±0.5 | 51.54±0.3 | 1.00x | 120.96±1.2 | 121.25±0.8 | 1.00x |
| 64 | 8192 | 229.60±21.4 | 220.80±0.6 | 1.04x | 510.71±9.6 | 507.84±2.0 | 1.01x |
| 128 | 256 | 0.55±0.0 | 0.56±0.1 | 0.99x | 0.70±0.0 | 0.70±0.0 | 1.00x |
| 128 | 1024 | 3.82±0.0 | 3.82±0.0 | 1.00x | 8.79±0.0 | 8.76±0.0 | 1.00x |
| 128 | 4096 | 57.13±3.6 | 57.04±0.3 | 1.00x | 138.74±0.9 | 138.75±0.7 | 1.00x |
| 128 | 8192 | 241.24±0.8 | 241.78±1.7 | 1.00x | 585.92±1.6 | 584.29±17.9 | 1.00x |

## Transformer Model Compilation Results

Comparison of compiled vs uncompiled TransformerLM model:

### Model Configuration
- **vocab_size:** 1000
- **context_length:** 384
- **d_model:** 192
- **num_layers:** 3
- **num_heads:** 6
- **d_ff:** 768
- **device:** mps
- **Batch size:** 6
- **Sequence length:** 192

### Performance Results

| Version | Forward Only (ms) | Forward+Backward+Optimizer (ms) | Full Step Speedup |
|---------|-------------------|----------------------------------|--------------------|
| Uncompiled | 6.83±0.1 | 20.29±0.0 | 1.00x |
| Compiled | 16.00±34.7 | 20.22±0.1 | 1.00x |

## Analysis

### torch.compile Support on macOS

### Key Observations

- **Attention compilation** provides an average speedup of **1.02x** for forward pass
- **Full model compilation** provides **1.00x** speedup for complete training step
- torch.compile automatically optimizes computation graphs and generates efficient kernels
- Compilation overhead is amortized over multiple runs
- Memory usage patterns may also be optimized by the compiler

