#!/bin/bash
"""
Script to run GRPO training experiments with different configurations.
"""

set -e

echo "Starting GRPO Training Experiment"
echo "================================="

# Create output directory
mkdir -p ./grpo_results

# Check if model exists
if [ ! -d "./models/Qwen2.5-Math-1.5B" ]; then
    echo "Error: Model not found at ./models/Qwen2.5-Math-1.5B"
    echo "Please download the model first."
    exit 1
fi

# Check if we have 2 GPUs
if [ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -lt 2 ]; then
    echo "Warning: Less than 2 GPUs detected. Using single GPU for both training and evaluation."
    EVAL_DEVICE="cuda"
else
    echo "Using GPU 0 for training and GPU 1 for evaluation"
    EVAL_DEVICE="cuda:1"
fi

echo "Running GRPO training with baseline configuration..."

# Run GRPO training with baseline configuration
uv run -m cs336_alignment.train_grpo \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_path ./data/math_train.jsonl \
    --output_dir ./grpo_results/baseline \
    --n_grpo_steps 50 \
    --learning_rate 1e-5 \
    --advantage_eps 1e-6 \
    --rollout_batch_size 64 \
    --group_size 8 \
    --sampling_temperature 1.0 \
    --sampling_max_tokens 1024 \
    --epochs_per_rollout_batch 1 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 32 \
    --loss_type reinforce_with_baseline \
    --use_std_normalization \
    --device cuda \
    --eval_device $EVAL_DEVICE \
    --use_wandb \
    --wandb_project "grpo-math-experiment" \
    --wandb_run_name "grpo_baseline" \
    --seed 42 \
    --eval_steps 5 \
    --save_steps 25

echo ""
echo "GRPO Training Experiment Complete!"
echo "Results saved in ./grpo_results/baseline/"
echo "Check wandb for training curves and validation accuracy plots."