# Experiment Log

## Overview
This document records experiments conducted for CS336 Assignment 1, focusing on training small Transformer language models on the TinyStories dataset.

## Experimental Infrastructure
Our experimental tracking infrastructure includes:
1. Experiment Tracker (`ExperimentTracker`): Records metrics during training, including training loss, validation loss, learning rate, etc.
2. Data Recording: Records metrics based on both gradient steps and actual time.
3. Visualization: Automatically generates loss curve graphs showing loss changes over gradient steps and real time.
4. Experiment Comparison: Provides tools to compare results from different experiments.
5. Weights & Biases Integration: Supports synchronizing experiment results to the Weights & Biases platform.

## Baseline Experiment

### Experiment 1: Base Model
**Configuration:**
```python
{
  "experiment_name": "baseline",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 512,
  "num_heads": 8,
  "num_layers": 6,
  "d_ff": 2048,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "max_iters": 10000
}
```

**Results:**
- Best validation loss: [fill in]
- Training time: [fill in]
- Final training loss: [fill in]

**Observations:**
[fill in observations about the baseline model]

## Ablation Studies

### Experiment 2: Reduced Attention Heads
**Configuration:**
```python
{
  "experiment_name": "fewer_heads",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 512,
  "num_heads": 4,  # reduced from 8 to 4
  "num_layers": 6,
  "d_ff": 2048,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "max_iters": 10000
}
```

**Results:**
- Best validation loss: [fill in]
- Training time: [fill in]
- Final training loss: [fill in]

**Observations:**
[fill in observations about the impact of reducing attention heads]

### Experiment 3: Reduced Number of Layers
**Configuration:**
```python
{
  "experiment_name": "fewer_layers",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 512,
  "num_heads": 8,
  "num_layers": 4,  # reduced from 6 to 4
  "d_ff": 2048,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "max_iters": 10000
}
```

**Results:**
- Best validation loss: [fill in]
- Training time: [fill in]
- Final training loss: [fill in]

**Observations:**
[fill in observations about the impact of reducing layers]

### Experiment 4: Modified Feedforward Network Size
**Configuration:**
```python
{
  "experiment_name": "larger_ff",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 512,
  "num_heads": 8,
  "num_layers": 6,
  "d_ff": 3072,  # increased from 2048 to 3072
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "max_iters": 10000
}
```

**Results:**
- Best validation loss: [fill in]
- Training time: [fill in]
- Final training loss: [fill in]

**Observations:**
[fill in observations about the impact of increasing feedforward network size]

### Experiment 5: Modified Learning Rate
**Configuration:**
```python
{
  "experiment_name": "higher_lr",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 512,
  "num_heads": 8,
  "num_layers": 6,
  "d_ff": 2048,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 3e-4,  # increased from 1e-4 to 3e-4
  "max_iters": 10000
}
```

**Results:**
- Best validation loss: [fill in]
- Training time: [fill in]
- Final training loss: [fill in]

**Observations:**
[fill in observations about the impact of increasing learning rate]

## Experiment Comparison and Analysis

### Loss Curve Comparison
[Insert comparison of loss curves from different experiments here]

### Parameter Efficiency Analysis
| Experiment | Parameter Count | Best Val Loss | Training Time | Samples/Second |
|------------|----------------|---------------|---------------|----------------|
| Base Model | [fill in] | [fill in] | [fill in] | [fill in] |
| Reduced Attention Heads | [fill in] | [fill in] | [fill in] | [fill in] |
| Reduced Layers | [fill in] | [fill in] | [fill in] | [fill in] |
| Modified FF Network Size | [fill in] | [fill in] | [fill in] | [fill in] |
| Modified Learning Rate | [fill in] | [fill in] | [fill in] | [fill in] |

## Conclusions
[Final conclusions based on the above experiments, discussing the impact of different parameter configurations on model performance and recommendations for optimal configurations]