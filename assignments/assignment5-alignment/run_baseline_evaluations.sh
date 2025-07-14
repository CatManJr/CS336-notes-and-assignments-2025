#!/bin/bash
# Script to run all zero-shot baseline evaluations for the supplemental assignment.

set -e

echo "Starting Zero-Shot Baseline Evaluations"
echo "======================================="

# Create results directory
mkdir -p ./results

echo ""
echo "Step 1: Running MMLU baseline evaluation..."
echo "==========================================="

uv run python -m cs336_alignment.benchmark.evaluate_mmlu_baseline \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_dir ./data/mmlu \
    --output_dir ./results/mmlu_baseline \
    --device cuda \
    --seed 42

echo ""
echo "Step 2: Running GSM8K baseline evaluation..."
echo "============================================"

uv run python -m cs336_alignment.benchmark.evaluate_gsm8k_baseline \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_file ./data/gsm8k/test.jsonl \
    --output_dir ./results/gsm8k_baseline \
    --device cuda \
    --seed 42

echo ""
echo "Step 3: Running AlpacaEval baseline generation..."
echo "================================================"

uv run python -m cs336_alignment.benchmark.evaluate_alpaca_eval_baseline \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_file ./data/alpaca_eval/alpaca_eval.jsonl \
    --output_file ./results/alpaca_eval_baseline/alpaca_eval_outputs.json \
    --device cuda \
    --seed 42

echo ""
echo "Step 4: Running SimpleSafetyTests baseline generation..."
echo "======================================================="

uv run python -m cs336_alignment.benchmark.evaluate_simple_safety_tests_baseline \
    --model_path ./models/Qwen2.5-Math-1.5B \
    --data_file ./data/simple_safety_tests/simple_safety_tests.csv \
    --output_file ./results/simple_safety_tests_baseline/sst_outputs.jsonl \
    --device cuda \
    --seed 42

echo ""
echo "Zero-Shot Baseline Evaluations Complete!"
echo "========================================"
echo ""
echo "Results Summary:"
echo "- MMLU results: ./results/mmlu_baseline/"
echo "- GSM8K results: ./results/gsm8k_baseline/"
echo "- AlpacaEval outputs: ./results/alpaca_eval_baseline/alpaca_eval_outputs.json"
echo "- SimpleSafetyTests outputs: ./results/simple_safety_tests_baseline/sst_outputs.jsonl"
echo ""
echo "Next steps:"
echo "1. For AlpacaEval, run the evaluation command:"
echo "   uv run alpaca_eval --model_outputs ./results/alpaca_eval_baseline/alpaca_eval_outputs.json \\"
echo "                     --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\"
echo "                     --base-dir '.'"
echo ""
echo "2. For SimpleSafetyTests, run the safety evaluation:"
echo "   uv run python scripts/evaluate_safety.py \\"
echo "          --input-path ./results/simple_safety_tests_baseline/sst_outputs.jsonl \\"
echo "          --model-name-or-path ./models/Qwen2.5-Math-1.5B \\"
echo "          --num-gpus 1 \\"
echo "          --output-path ./results/simple_safety_tests_baseline/safety_annotations.jsonl"