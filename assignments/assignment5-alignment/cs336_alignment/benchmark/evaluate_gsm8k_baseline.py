#!/usr/bin/env python3
"""
GSM8K baseline evaluation script for Llama 3.1 8B.
"""

import json
import os
import re
import time
from typing import List, Optional
import argparse

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


def parse_gsm8k_response(response: str) -> Optional[str]:
    """
    Parse GSM8K response to extract the final numeric answer.
    
    Args:
        response: The model's response text
        
    Returns:
        The final numeric answer as a string or None if parsing fails
    """
    # Clean the response
    response = response.strip()
    
    # Find all numbers in the response
    # Look for numbers (including decimals and negative numbers)
    pattern = r'-?\d+(?:\.\d+)?'
    matches = re.findall(pattern, response)
    
    if not matches:
        return None
    
    # Take the last number as the final answer
    try:
        result = float(matches[-1])
        # Convert to int if it's a whole number, otherwise keep as float
        if result == int(result):
            return str(int(result))
        else:
            return str(result)
    except ValueError:
        return None


def load_gsm8k_data(data_file: str) -> List[dict]:
    """Load GSM8K test data."""
    examples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(data)
    
    return examples


def format_gsm8k_prompt(example: dict, system_prompt: str) -> str:
    """Format GSM8K example into a prompt."""
    gsm8k_prompt = f"{example['question']}\n\nAnswer:"
    full_prompt = system_prompt + "\n\n" + gsm8k_prompt
    return full_prompt


def init_vllm_model(model_path: str, device: str = "cuda", seed: int = 42) -> LLM:
    """Initialize vLLM model."""
    vllm_set_random_seed(seed)
    
    # Monkeypatch to avoid distributed issues
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_path,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.9,
        )


def evaluate_gsm8k(
    model_path: str,
    data_file: str,
    output_dir: str,
    device: str = "cuda",
    seed: int = 42,
    max_examples: Optional[int] = None
):
    """Evaluate model on GSM8K."""
    
    # System prompt from the assignment
    system_prompt = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:"""
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    llm = init_vllm_model(model_path, device, seed)
    
    # Load data
    print(f"Loading GSM8K data from {data_file}...")
    examples = load_gsm8k_data(data_file)
    
    if max_examples:
        examples = examples[:max_examples]
    
    print(f"Total examples: {len(examples)}")
    
    # Generate responses
    print("Generating responses...")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["# Query:"]
    )
    
    prompts = [format_gsm8k_prompt(ex, system_prompt) for ex in examples]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time
    
    # Process results
    results = []
    correct = 0
    failed_parse = 0
    
    for i, (example, output) in enumerate(zip(examples, outputs)):
        response = output.outputs[0].text.strip()
        predicted = parse_gsm8k_response(response)
        
        # Extract ground truth answer
        answer_text = example['answer']
        # The answer format in GSM8K is typically "#### 42"
        ground_truth_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
        if ground_truth_match:
            ground_truth_float = float(ground_truth_match.group(1))
            # Convert to string format same as predicted
            if ground_truth_float == int(ground_truth_float):
                ground_truth = str(int(ground_truth_float))
            else:
                ground_truth = str(ground_truth_float)
        else:
            # Fallback: try to parse the last number in the answer
            numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
            if numbers:
                ground_truth_float = float(numbers[-1])
                if ground_truth_float == int(ground_truth_float):
                    ground_truth = str(int(ground_truth_float))
                else:
                    ground_truth = str(ground_truth_float)
            else:
                ground_truth = None
        
        is_correct = False
        if predicted is not None and ground_truth is not None:
            is_correct = predicted == ground_truth
        
        if predicted is None:
            failed_parse += 1
        elif is_correct:
            correct += 1
        
        results.append({
            'question': example['question'],
            'answer': example['answer'],
            'response': response,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': is_correct
        })
    
    # Calculate metrics
    accuracy = correct / len(examples)
    throughput = len(examples) / generation_time
    
    metrics = {
        'total_examples': len(examples),
        'correct': correct,
        'failed_parse': failed_parse,
        'accuracy': accuracy,
        'generation_time': generation_time,
        'throughput': throughput
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gsm8k_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, 'gsm8k_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"\nGSM8K Evaluation Results:")
    print(f"Total examples: {len(examples)}")
    print(f"Correct: {correct}")
    print(f"Failed to parse: {failed_parse}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Throughput: {throughput:.2f} examples/second")
    
    # Sample failed parses
    if failed_parse > 0:
        print(f"\nSample failed parses:")
        failed_examples = [r for r in results if r['predicted'] is None]
        for i, ex in enumerate(failed_examples[:5]):
            print(f"{i+1}. Question: {ex['question'][:100]}...")
            print(f"   Response: {ex['response'][:150]}...")
    
    # Sample incorrect predictions
    incorrect_examples = [r for r in results if r['predicted'] is not None and not r['correct']]
    if incorrect_examples:
        print(f"\nSample incorrect predictions:")
        for i, ex in enumerate(incorrect_examples[:5]):
            print(f"{i+1}. Question: {ex['question'][:100]}...")
            print(f"   Ground truth: {ex['ground_truth']}, Predicted: {ex['predicted']}")
            print(f"   Response: {ex['response'][:150]}...")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5-Math-1.5B on GSM8K')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the model or HuggingFace model ID')
    parser.add_argument('--data_file', type=str, default='./data/gsm8k/test.jsonl',
                        help='Path to GSM8K test file')
    parser.add_argument('--output_dir', type=str, default='./results/gsm8k_baseline',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Max examples for testing')
    
    args = parser.parse_args()
    
    evaluate_gsm8k(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()