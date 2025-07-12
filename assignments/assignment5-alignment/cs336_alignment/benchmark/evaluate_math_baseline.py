#!/usr/bin/env python3
"""
Script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH dataset.

This script:
1. Loads MATH validation examples directly from allenai/tulu-3-sft-personas-math-filtered
2. Uses prompts from the dataset
3. Generates outputs using vLLM
4. Calculates evaluation metrics using r1_zero_reward_fn
5. Serializes results to disk for analysis
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_r1_zero_prompt(prompt_path: str) -> str:
    """Load the r1_zero prompt template."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_math_validation_data(test_size: float = 0.2, random_state: int = 42, max_examples: int = None) -> List[Dict[str, Any]]:
    """Load MATH validation examples from allenai/tulu-3-sft-personas-math-filtered dataset using pandas."""
    all_examples = []
    
    print("Loading MATH dataset from allenai/tulu-3-sft-personas-math-filtered...")
    
    try:
        # Load the dataset using pandas from parquet files
        print("Loading dataset using pandas...")
        df = pd.read_parquet("hf://datasets/allenai/tulu-3-sft-personas-math-filtered/data/train-00000-of-00001.parquet")
        
        print(f"Loaded {len(df)} examples from the dataset")
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Examine the first few examples to understand the structure
        print("Sample data structure:")
        if len(df) > 0:
            first_example = df.iloc[0]
            print(f"Keys: {first_example.index.tolist()}")
            for key in first_example.index:
                value = first_example[key]
                print(f"  {key}: {str(value)[:100]}...")
        
        print("Processing examples...")
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                # Handle conversational format from messages
                if "messages" in row and row["messages"] is not None:
                    messages = row["messages"]
                    
                    # Extract the math problem from user message
                    user_message = ""
                    assistant_message = ""
                    
                    for msg in messages:
                        if isinstance(msg, dict):
                            if msg.get("role") == "user":
                                user_message = msg.get("content", "")
                            elif msg.get("role") == "assistant":
                                assistant_message = msg.get("content", "")
                    
                    # Skip examples without both user and assistant messages
                    if not user_message or not assistant_message:
                        continue
                    
                    # Extract just the math problem from the user message
                    # The user message in the dataset might contain the full conversation format
                    # We need to extract just the question part
                    problem = user_message
                    
                    math_example = {
                        "problem": problem,  # Store the raw problem for formatting with r1_zero prompt
                        "solution": assistant_message,  # Use assistant message as solution
                        "level": row.get("level", "Unknown"),
                        "type": row.get("type", "Math"),
                        "messages": messages,  # Keep original messages for reference
                        "id": row.get("id", f"example_{i}")
                    }
                    all_examples.append(math_example)
                    
                elif "prompt" in row and row["prompt"] is not None:
                    # Handle cases where there's a direct prompt field
                    prompt = row["prompt"]
                    
                    # Try to extract solution from messages if available
                    solution = ""
                    if "messages" in row and row["messages"] is not None:
                        for msg in row["messages"]:
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                solution = msg.get("content", "")
                                break
                    
                    if solution:
                        math_example = {
                            "problem": prompt,
                            "solution": solution,
                            "level": row.get("level", "Unknown"),
                            "type": row.get("type", "Math"),
                            "messages": row.get("messages", []),
                            "id": row.get("id", f"example_{i}")
                        }
                        all_examples.append(math_example)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
            
            # Show progress for large datasets
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(df)} examples, found {len(all_examples)} valid examples")
        
        print(f"Successfully processed {len(all_examples)} examples out of {len(df)} total examples")
        
        # Check if we have any valid examples
        if len(all_examples) == 0:
            print("ERROR: No valid examples found in the dataset!")
            print("This could be due to:")
            print("- Incorrect data format")
            print("- Missing required fields (messages, prompt, etc.)")
            print("- Data corruption")
            raise ValueError("No valid examples found in the dataset")
        
        # Apply max_examples limit if specified (before splitting)
        if max_examples and len(all_examples) > max_examples:
            print(f"Limiting processed examples to {max_examples} (from {len(all_examples)})")
            # Use fixed seed to ensure reproducible sampling
            np.random.seed(random_state)
            indices = np.random.choice(len(all_examples), max_examples, replace=False)
            all_examples = [all_examples[i] for i in sorted(indices)]
        
        # Split the dataset into train and test sets based on dataset length
        if len(all_examples) > 1 and test_size > 0:
            print(f"Splitting dataset with test_size={test_size}, random_state={random_state}")
            
            # Calculate test size based on dataset length
            test_count = int(len(all_examples) * test_size)
            train_count = len(all_examples) - test_count
            
            print(f"Total examples: {len(all_examples)}")
            print(f"Test examples: {test_count} ({test_count/len(all_examples):.1%})")
            print(f"Train examples: {train_count} ({train_count/len(all_examples):.1%})")
            
            train_examples, test_examples = train_test_split(
                all_examples, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=True
            )
            
            print(f"Split completed: {len(train_examples)} training, {len(test_examples)} test examples")
            
            # For evaluation, we'll use the test set
            all_examples = test_examples
        else:
            print("No splitting performed (insufficient examples or test_size=0)")
                
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load dataset: {e}")
        print("="*80)
        print("DATASET LOADING FAILED")
        print("="*80)
        print("Cannot proceed with benchmark without real dataset.")
        print("Please ensure:")
        print("1. Internet connection is available")
        print("2. Hugging Face authentication is properly configured")
        print("3. The dataset 'allenai/tulu-3-sft-personas-math-filtered' exists and is accessible")
        print("4. Pandas and required dependencies are installed")
        print("="*80)
        raise SystemExit(f"Failed to load real dataset: {e}")
    
    print(f"Total examples loaded for evaluation: {len(all_examples)}")
    return all_examples


def check_model_availability(model_path: str) -> bool:
    """Check if the model is available at the given path."""
    if not os.path.exists(model_path):
        return False
    
    # Check for essential model files
    essential_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    for file in essential_files:
        if not os.path.exists(os.path.join(model_path, file)):
            # Try alternative file names
            alt_files = {
                'pytorch_model.bin': ['model.safetensors', 'pytorch_model-00001-of-00002.bin'],
                'tokenizer.json': ['tokenizer_config.json']
            }
            if file in alt_files:
                found = any(os.path.exists(os.path.join(model_path, alt)) for alt in alt_files[file])
                if not found:
                    return False
            else:
                return False
    
    return True


def print_model_download_instructions(model_path: str):
    """Print instructions for downloading the model manually."""
    print(f"\n{'='*80}")
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print(f"The model is not found at: {model_path}")
    print("\nTo download the Qwen 2.5 Math 1.5B model manually, you can use one of these methods:")
    print("\n1. Using Hugging Face Hub:")
    print("   pip install huggingface-hub")
    print("   huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir ./models/Qwen2.5-Math-1.5B")
    print("\n2. Using Git LFS:")
    print("   git lfs install")
    print("   git clone https://huggingface.co/Qwen/Qwen2.5-Math-1.5B ./models/Qwen2.5-Math-1.5B")
    print("\n3. Using Python script:")
    print("   from huggingface_hub import snapshot_download")
    print("   snapshot_download(repo_id='Qwen/Qwen2.5-Math-1.5B', local_dir='./models/Qwen2.5-Math-1.5B')")
    print(f"\nAfter downloading, update the --model_path argument to point to your local model directory.")
    print("="*80)


def format_prompts(examples: List[Dict[str, Any]], prompt_template: str) -> List[str]:
    """Format MATH examples using the r1_zero prompt template."""
    prompts = []
    for example in examples:
        # Use the 'problem' field and format it with the r1_zero template
        question = example.get('problem', '')
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
    
    # Handle empty dataset
    if len(prompts) == 0:
        print("No prompts to evaluate!")
        return {
            'results': [],
            'summary': {
                'total_examples': 0,
                'correct_format_and_answer': 0,
                'correct_format_wrong_answer': 0,
                'wrong_format': 0,
                'avg_format_reward': 0.0,
                'avg_answer_reward': 0.0,
                'avg_total_reward': 0.0,
                'accuracy': 0.0,
                'format_accuracy': 0.0
            }
        }
    
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
    
    # Calculate summary statistics (with safety checks)
    total_examples = len(results)
    correct_format_and_answer = sum(1 for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 1.0)
    correct_format_wrong_answer = sum(1 for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 0.0)
    wrong_format = sum(1 for r in results if r['format_reward'] == 0.0)
    
    # Safe division
    avg_format_reward = sum(format_rewards) / len(format_rewards) if format_rewards else 0.0
    avg_answer_reward = sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0.0
    avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    
    summary = {
        'total_examples': total_examples,
        'correct_format_and_answer': correct_format_and_answer,
        'correct_format_wrong_answer': correct_format_wrong_answer,
        'wrong_format': wrong_format,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'avg_total_reward': avg_total_reward,
        'accuracy': correct_format_and_answer / total_examples if total_examples > 0 else 0.0,
        'format_accuracy': (correct_format_and_answer + correct_format_wrong_answer) / total_examples if total_examples > 0 else 0.0
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


def save_compact_results(evaluation_results: Dict[str, Any], output_file: str, max_examples_per_category: int = 5):
    """Save compact evaluation results with only summary statistics and sample examples."""
    results = evaluation_results['results']
    summary = evaluation_results['summary']
    
    # Get sample examples from each category
    correct_examples = [r for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 1.0]
    format_correct_examples = [r for r in results if r['format_reward'] == 1.0 and r['answer_reward'] == 0.0]
    format_wrong_examples = [r for r in results if r['format_reward'] == 0.0]
    
    # Create compact result structure
    compact_results = {
        'summary': summary,
        'evaluation_info': {
            'total_examples_evaluated': len(results),
            'timestamp': pd.Timestamp.now().isoformat(),
            'categories': {
                'correct_format_and_answer': len(correct_examples),
                'correct_format_wrong_answer': len(format_correct_examples),
                'wrong_format': len(format_wrong_examples)
            }
        },
        'sample_examples': {
            'correct_format_and_answer': [
                {
                    'example_id': ex['example_id'],
                    'generated_text': ex['generated_text'][:500] + '...' if len(ex['generated_text']) > 500 else ex['generated_text'],
                    'ground_truth': ex['ground_truth'][:500] + '...' if len(ex['ground_truth']) > 500 else ex['ground_truth'],
                    'format_reward': ex['format_reward'],
                    'answer_reward': ex['answer_reward'],
                    'total_reward': ex['total_reward']
                }
                for ex in correct_examples[:max_examples_per_category]
            ],
            'correct_format_wrong_answer': [
                {
                    'example_id': ex['example_id'],
                    'generated_text': ex['generated_text'][:500] + '...' if len(ex['generated_text']) > 500 else ex['generated_text'],
                    'ground_truth': ex['ground_truth'][:500] + '...' if len(ex['ground_truth']) > 500 else ex['ground_truth'],
                    'format_reward': ex['format_reward'],
                    'answer_reward': ex['answer_reward'],
                    'total_reward': ex['total_reward']
                }
                for ex in format_correct_examples[:max_examples_per_category]
            ],
            'wrong_format': [
                {
                    'example_id': ex['example_id'],
                    'generated_text': ex['generated_text'][:500] + '...' if len(ex['generated_text']) > 500 else ex['generated_text'],
                    'ground_truth': ex['ground_truth'][:500] + '...' if len(ex['ground_truth']) > 500 else ex['ground_truth'],
                    'format_reward': ex['format_reward'],
                    'answer_reward': ex['answer_reward'],
                    'total_reward': ex['total_reward']
                }
                for ex in format_wrong_examples[:max_examples_per_category]
            ]
        }
    }
    
    # Save compact results
    with open(output_file, 'w') as f:
        json.dump(compact_results, f, indent=2)
    
    print(f"Compact results saved to {output_file}")
    print(f"File contains summary statistics and {max_examples_per_category} sample examples per category")
    
    return compact_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen 2.5 Math 1.5B on MATH dataset')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the Qwen 2.5 Math 1.5B model (default: ./models/Qwen2.5-Math-1.5B)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model name to use directly (e.g., Qwen/Qwen2.5-Math-1.5B)')
    parser.add_argument('--output_dir', type=str, default='./math_baseline_results',
                        help='Directory to save results')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (for debugging)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for dataset splitting (default: 42)')
    parser.add_argument('--max_dataset_size', type=int, default=None,
                        help='Maximum size of dataset to load before splitting (default: None - no limit)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always load real dataset - no sample data option
    examples = load_math_validation_data(
        test_size=args.test_size,
        random_state=args.random_state,
        max_examples=args.max_dataset_size
    )
    
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Using first {len(examples)} examples for evaluation")
    
    print(f"Final dataset size for evaluation: {len(examples)} examples")
    
    print("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt('cs336_alignment/prompts/r1_zero.prompt')
    
    print("Formatting prompts using r1_zero template...")
    prompts = format_prompts(examples, prompt_template)
    
    # Extract ground truth answers (use 'solution' field)
    ground_truths = [example.get('solution', '') for example in examples]
    
    # Determine model to use
    if args.model_name:
        print(f"Using HuggingFace model: {args.model_name}")
        model_identifier = args.model_name
    else:
        if check_model_availability(args.model_path):
            print(f"Using local model: {args.model_path}")
            model_identifier = args.model_path
        else:
            print_model_download_instructions(args.model_path)
            print("\nAlternatively, you can use a HuggingFace model directly with --model_name")
            print("Example: --model_name Qwen/Qwen2.5-Math-1.5B")
            print("\nFor testing purposes, you can use a smaller model like:")
            print("--model_name microsoft/DialoGPT-small")
            return
    
    print("Loading vLLM model...")
    try:
        llm = LLM(model=model_identifier)
    except Exception as e:
        print(f"Failed to load model: {e}")
        if not args.model_name:
            print("\nTrying to use HuggingFace model directly...")
            try:
                llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
                print("Successfully loaded model from HuggingFace")
            except Exception as e2:
                print(f"Failed to load from HuggingFace: {e2}")
                print("Please check your internet connection and try again.")
                return
        else:
            return
    
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
    
    # Save compact results to disk (not full results)
    output_file = os.path.join(args.output_dir, 'math_baseline_evaluation.json')
    print(f"\nSaving compact results to {output_file}")
    save_compact_results(evaluation_results, output_file, max_examples_per_category=5)
    
    # Optionally save detailed statistics
    stats_file = os.path.join(args.output_dir, 'math_baseline_stats.json')
    detailed_stats = {
        'summary': evaluation_results['summary'],
        'evaluation_info': {
            'total_examples_evaluated': len(evaluation_results['results']),
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': {
                'model_path': args.model_path if not args.model_name else None,
                'model_name': args.model_name,
                'test_size': args.test_size,
                'random_state': args.random_state,
                'max_dataset_size': args.max_dataset_size,
                'max_examples': args.max_examples
            }
        },
        'reward_distribution': {
            'format_rewards': [r['format_reward'] for r in evaluation_results['results']],
            'answer_rewards': [r['answer_reward'] for r in evaluation_results['results']],
            'total_rewards': [r['total_reward'] for r in evaluation_results['results']]
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    print(f"Detailed statistics saved to {stats_file}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()