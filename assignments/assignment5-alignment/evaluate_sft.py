#!/usr/bin/env python3
"""
Evaluate SFT model on MATH dataset.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os
from pathlib import Path

from cs336_alignment.benchmark.evaluate_math_baseline import (
    load_math_validation_data,
    format_prompts,
    evaluate_vllm,
    save_compact_results,
    analyze_examples,
    load_r1_zero_prompt
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_sft_model(
    model_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_examples: int = None
):
    """Evaluate SFT model on MATH dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating SFT model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Load validation data
    examples = load_math_validation_data(
        test_size=test_size,
        random_state=random_state,
        max_examples=None  # Use full dataset for evaluation
    )
    
    if max_examples:
        examples = examples[:max_examples]
        print(f"Using first {len(examples)} examples for evaluation")
    
    print(f"Evaluating on {len(examples)} examples")
    
    # Load prompt template and format prompts
    prompt_template = load_r1_zero_prompt()
    prompts = format_prompts(examples, prompt_template)
    ground_truths = [example.get('solution', '') for example in examples]
    
    # Load model with vLLM
    print("Loading model with vLLM...")
    llm = LLM(model=model_path)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # Run evaluation
    print("Running evaluation...")
    evaluation_results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params
    )
    
    # Print summary
    summary = evaluation_results['summary']
    print("\n" + "="*80)
    print("SFT MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Total examples: {summary['total_examples']}")
    print(f"Correct format and answer: {summary['correct_format_and_answer']} ({summary['correct_format_and_answer']/summary['total_examples']:.2%})")
    print(f"Correct format, wrong answer: {summary['correct_format_wrong_answer']} ({summary['correct_format_wrong_answer']/summary['total_examples']:.2%})")
    print(f"Wrong format: {summary['wrong_format']} ({summary['wrong_format']/summary['total_examples']:.2%})")
    print(f"Average format reward: {summary['avg_format_reward']:.3f}")
    print(f"Average answer reward: {summary['avg_answer_reward']:.3f}")
    print(f"Average total reward: {summary['avg_total_reward']:.3f}")
    print(f"Overall accuracy: {summary['accuracy']:.2%}")
    print(f"Format accuracy: {summary['format_accuracy']:.2%}")
    
    # Analyze examples
    analyze_examples(evaluation_results['results'])
    
    # Save results
    output_file = os.path.join(output_dir, 'sft_evaluation_results.json')
    save_compact_results(evaluation_results, output_file, max_examples_per_category=5)
    
    print(f"\nResults saved to {output_file}")
    print("SFT evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SFT model on MATH dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the SFT model to evaluate')
    parser.add_argument('--output_dir', type=str, default='./sft_evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for dataset splitting')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (for debugging)')
    
    args = parser.parse_args()
    
    evaluate_sft_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_examples=args.max_examples
    )


if __name__ == '__main__':
    main()