#!/usr/bin/env python3
"""
Script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH dataset.

This script:
1. Loads MATH validation examples
2. Formats them using the r1_zero prompt
3. Generates outputs using vLLM
4. Calculates evaluation metrics using r1_zero_reward_fn
5. Serializes results to disk for analysis
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
import argparse

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_math_validation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load MATH validation examples from JSONL file."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def load_r1_zero_prompt(prompt_path: str) -> str:
    """Load the r1_zero prompt template."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def format_prompts(examples: List[Dict[str, Any]], prompt_template: str) -> List[str]:
    """Format MATH examples using the r1_zero prompt template."""
    prompts = []
    for example in examples:
        # Assuming the question field is 'problem' or 'question'
        question = example.get('problem', example.get('question', ''))
        prompt = prompt_template.format(question=question)
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and return results.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    
    # Generate responses
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Extract generated texts
    generated_texts = [output.outputs[0].text for output in outputs]
    
    # Evaluate each response
    results = []
    format_rewards = []
    answer_rewards = []
    total_rewards = []
    
    for i, (prompt, generated_text, ground_truth) in enumerate(zip(prompts, generated_texts, ground_truths)):
        # Evaluate using reward function
        reward_dict = reward_fn(generated_text, ground_truth, fast=True)
        
        format_reward = reward_dict['format_reward']
        answer_reward = reward_dict['answer_reward']
        total_reward = reward_dict['reward']
        
        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
        total_rewards.append(total_reward)
        
        results.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'ground_truth': ground_truth,
            'format_reward': format_reward,
            'answer_reward': answer_reward,
            'total_reward': total_reward,
            'example_id': i
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(prompts)} examples")
    
    # Calculate summary statistics
    total_examples = len(results)
    correct_format_and_answer = sum(1 for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 1.0)
    correct_format_wrong_answer = sum(1 for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 0.0)
    wrong_format = sum(1 for r in results if r['format_reward'] == 0.0)
    
    avg_format_reward = sum(format_rewards) / len(format_rewards)
    avg_answer_reward = sum(answer_rewards) / len(answer_rewards)
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    
    summary = {
        'total_examples': total_examples,
        'correct_format_and_answer': correct_format_and_answer,
        'correct_format_wrong_answer': correct_format_wrong_answer,
        'wrong_format': wrong_format,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'avg_total_reward': avg_total_reward,
        'accuracy': correct_format_and_answer / total_examples,
        'format_accuracy': (correct_format_and_answer + correct_format_wrong_answer) / total_examples
    }
    
    return {
        'results': results,
        'summary': summary
    }


def analyze_examples(results: List[Dict[str, Any]], num_examples: int = 10):
    """Analyze examples from each category for debugging."""
    print("\n" + "="*80)
    print("ANALYSIS OF EXAMPLES")
    print("="*80)
    
    # Category 1: Correct format and answer (format_reward=1, answer_reward=1)
    correct_examples = [r for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 1.0]
    print(f"\n1. CORRECT FORMAT AND ANSWER ({len(correct_examples)} examples)")
    print("-" * 50)
    for i, example in enumerate(correct_examples[:num_examples]):
        print(f"Example {i+1}:")
        print(f"Generated: {example['generated_text'][:200]}...")
        print(f"Ground truth: {example['ground_truth']}")
        print()
    
    # Category 2: Correct format, wrong answer (format_reward=1, answer_reward=0)
    format_correct_examples = [r for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 0.0]
    print(f"\n2. CORRECT FORMAT, WRONG ANSWER ({len(format_correct_examples)} examples)")
    print("-" * 50)
    for i, example in enumerate(format_correct_examples[:num_examples]):
        print(f"Example {i+1}:")
        print(f"Generated: {example['generated_text'][:200]}...")
        print(f"Ground truth: {example['ground_truth']}")
        print()
    
    # Category 3: Wrong format (format_reward=0)
    format_wrong_examples = [r for r in results if r['format_reward'] == 0.0]
    print(f"\n3. WRONG FORMAT ({len(format_wrong_examples)} examples)")
    print("-" * 50)
    for i, example in enumerate(format_wrong_examples[:num_examples]):
        print(f"Example {i+1}:")
        print(f"Generated: {example['generated_text'][:200]}...")
        print(f"Ground truth: {example['ground_truth']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen 2.5 Math 1.5B on MATH dataset')
    parser.add_argument('--model_path', type=str, default='/data/a5-alignment/models/Qwen2.5-Math-1.5B',
                        help='Path to the Qwen 2.5 Math 1.5B model')
    parser.add_argument('--data_path', type=str, default='/data/a5-alignment/MATH/validation.jsonl',
                        help='Path to MATH validation data')
    parser.add_argument('--prompt_path', type=str, default='cs336_alignment/prompts/r1_zero.prompt',
                        help='Path to r1_zero prompt template')
    parser.add_argument('--output_dir', type=str, default='./math_baseline_results',
                        help='Directory to save results')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (for debugging)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading MATH validation data...")
    examples = load_math_validation_data(args.data_path)
    
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Using first {len(examples)} examples for debugging")
    
    print(f"Loaded {len(examples)} examples")
    
    print("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt(args.prompt_path)
    
    print("Formatting prompts...")
    prompts = format_prompts(examples, prompt_template)
    
    # Extract ground truth answers
    ground_truths = [example.get('solution', example.get('answer', '')) for example in examples]
    
    print("Loading vLLM model...")
    llm = LLM(model=args.model_path)
    
    # Set up sampling parameters as specified in the assignment
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    print("Running evaluation...")
    evaluation_results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params
    )
    
    # Print summary statistics
    summary = evaluation_results['summary']
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
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
    
    # Analyze examples from each category
    analyze_examples(evaluation_results['results'])
    
    # Save results to disk
    output_file = os.path.join(args.output_dir, 'math_baseline_evaluation.json')
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()