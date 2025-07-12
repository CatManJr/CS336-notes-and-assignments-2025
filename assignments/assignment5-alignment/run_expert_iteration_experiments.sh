#!/bin/bash
"""
Script to run Expert Iteration experiments with different configurations.
"""

set -e

echo "Starting Expert Iteration Experiments"
echo "===================================="

# Create output directory
mkdir -p ./ei_results

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

# Experiment configurations
# Format: "rollouts,epochs,batch_size,name"
CONFIGS=(
    "4,1,512,rollouts4_epochs1_batch512"
    "8,1,512,rollouts8_epochs1_batch512"
    "4,2,512,rollouts4_epochs2_batch512"
    "6,1,1024,rollouts6_epochs1_batch1024"
    "4,1,1024,rollouts4_epochs1_batch1024"
    "8,2,1024,rollouts8_epochs2_batch1024"
)

echo "Running Expert Iteration experiments with the following configurations:"
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r rollouts epochs batch_size name <<< "$config"
    echo "  - $name: $rollouts rollouts, $epochs epochs, batch size $batch_size"
done

echo ""

# Run experiments
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r rollouts epochs batch_size name <<< "$config"
    
    echo "Running experiment: $name"
    echo "Configuration: $rollouts rollouts, $epochs epochs, batch size $batch_size"
    
    OUTPUT_DIR="./ei_results/$name"
    
    python -m cs336_alignment.train_expert_iteration \
        --model_path ./models/Qwen2.5-Math-1.5B \
        --data_path ./data/math_train.jsonl \
        --output_dir "$OUTPUT_DIR" \
        --n_ei_steps 5 \
        --batch_size_per_step "$batch_size" \
        --num_rollouts "$rollouts" \
        --sft_epochs "$epochs" \
        --sft_batch_size 2 \
        --sft_gradient_accumulation_steps 8 \
        --sft_learning_rate 5e-5 \
        --sampling_temperature 1.0 \
        --sampling_max_tokens 512 \
        --device cuda \
        --eval_device "$EVAL_DEVICE" \
        --use_wandb \
        --wandb_project "expert-iteration-math" \
        --wandb_run_name "$name" \
        --seed 42
    
    echo "Completed experiment: $name"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
done

echo "All Expert Iteration experiments completed!"
echo "Results summary:"
echo "================"

# Generate results summary
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r rollouts epochs batch_size name <<< "$config"
    OUTPUT_DIR="./ei_results/$name"
    
    if [ -f "$OUTPUT_DIR/final_model/config.json" ]; then
        echo "✓ $name: Training completed successfully"
        
        # Check if final results exist
        if [ -f "$OUTPUT_DIR/ei_step_5/step_results.json" ]; then
            # Extract final accuracy from results
            FINAL_ACC=$(python -c "
import json
try:
    with open('$OUTPUT_DIR/ei_step_5/step_results.json', 'r') as f:
        data = json.load(f)
        print(f\"{data['eval_results']['accuracy']:.3f}\")
except:
    print('N/A')
")
            echo "  Final validation accuracy: $FINAL_ACC"
        fi
    else
        echo "✗ $name: Training failed or incomplete"
    fi
done

echo ""
echo "Check wandb dashboard for detailed training curves and entropy plots."
echo "Individual experiment results are saved in ./ei_results/<experiment_name>/"