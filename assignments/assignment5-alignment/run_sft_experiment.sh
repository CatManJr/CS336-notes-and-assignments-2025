#!/bin/bash
"""
Script to run the SFT experiment on MATH dataset.
"""

set -e

echo "Starting SFT Training Experiment"
echo "================================="

# Create data directory
mkdir -p ./data/math_sft

# Step 1: Prepare dataset splits
echo "Step 1: Preparing dataset splits..."
python prepare_math_dataset.py \
    --output_dir ./data/math_sft \
    --test_size 0.2 \
    --max_examples 5000 \
    --filter_correct

echo ""
echo "Step 2: Running SFT training with different dataset sizes..."

# Check if model exists
if [ ! -d "./models/Qwen2.5-Math-1.5B" ]; then
    echo "Error: Model not found at ./models/Qwen2.5-Math-1.5B"
    echo "Please download the model first using:"
    echo "python download_model.py"
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

# Run training for different dataset sizes
python -m cs336_alignment.train_sft \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_path ./data/math_sft \
    --output_dir ./sft_results \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --device cuda \
    --eval_device $EVAL_DEVICE \
    --use_wandb \
    --wandb_project "sft-math-experiment" \
    --wandb_run_name "sft-experiment" \
    --gradient_clip_value 1.0 \
    --seed 42

echo ""
echo "Step 3: Evaluating final models..."

# Run evaluation on the final models
for size in 128 256 512 1024 full; do
    if [ -d "./sft_results/size_$size/final_model" ]; then
        echo "Evaluating model trained on size $size..."
        python -m cs336_alignment.benchmark.evaluate_math_baseline \
            --model_path "./sft_results/size_$size/final_model" \
            --output_dir "./sft_results/size_$size/evaluation" \
            --max_examples 500
    fi
done

echo ""
echo "SFT Training Experiment Complete!"
echo "Results saved in ./sft_results/"
echo "Check wandb for training curves and metrics."