#!/usr/bin/env python3
"""
MMLU baseline evaluation script for Llama 3.1 8B.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


def parse_mmlu_response(response: str) -> Optional[str]:
    """
    Parse MMLU response to extract the predicted answer letter.
    
    Args:
        response: The model's response text
        
    Returns:
        The predicted answer letter (A, B, C, or D) or None if parsing fails
    """
    # Clean the response
    response = response.strip()
    
    # Look for "The correct answer is X" pattern
    pattern = r"The correct answer is ([ABCD])"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for standalone letter at the end of response
    pattern = r"\b([ABCD])\b"
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        return matches[-1].upper()  # Take the last match
    
    # Look for "Answer: X" pattern
    pattern = r"Answer:\s*([ABCD])"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for letter followed by period or parenthesis
    pattern = r"([ABCD])[.)]"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def load_mmlu_data(data_dir: str) -> Dict[str, List[Dict]]:
    """Load MMLU test data from all subjects."""
    test_data = {}
    test_dir = Path(data_dir) / "test"
    
    for file_path in test_dir.glob("*.csv"):
        subject = file_path.stem
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    question = parts[0]
                    options = parts[1:5]
                    answer = parts[5]
                    
                    examples.append({
                        'question': question,
                        'options': options,
                        'answer': answer,
                        'subject': subject
                    })
        
        test_data[subject] = examples
    
    return test_data


def format_mmlu_prompt(example: Dict, system_prompt: str) -> str:
    """Format MMLU example into a prompt."""
    subject = example['subject'].replace('_', ' ')
    
    mmlu_prompt = f"""Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {example['question']}
A. {example['options'][0]}
B. {example['options'][1]}
C. {example['options'][2]}
D. {example['options'][3]}

Answer:"""
    
    full_prompt = system_prompt + "\n\n" + mmlu_prompt
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


def evaluate_mmlu(
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str = "cuda",
    seed: int = 42,
    max_examples_per_subject: Optional[int] = None
):
    """Evaluate model on MMLU."""
    
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
    print(f"Loading MMLU data from {data_dir}...")
    test_data = load_mmlu_data(data_dir)
    
    # Prepare all examples
    all_examples = []
    for subject, examples in test_data.items():
        if max_examples_per_subject:
            examples = examples[:max_examples_per_subject]
        
        for example in examples:
            example['subject'] = subject
            all_examples.append(example)
    
    print(f"Total examples: {len(all_examples)}")
    
    # Generate responses
    print("Generating responses...")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=100,
        stop=["# Query:"]
    )
    
    prompts = [format_mmlu_prompt(ex, system_prompt) for ex in all_examples]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time
    
    # Process results
    results = []
    correct = 0
    failed_parse = 0
    
    for i, (example, output) in enumerate(zip(all_examples, outputs)):
        response = output.outputs[0].text.strip()
        predicted = parse_mmlu_response(response)
        
        is_correct = predicted == example['answer'] if predicted else False
        if predicted is None:
            failed_parse += 1
        elif is_correct:
            correct += 1
        
        results.append({
            'subject': example['subject'],
            'question': example['question'],
            'options': example['options'],
            'answer': example['answer'],
            'response': response,
            'predicted': predicted,
            'correct': is_correct
        })
    
    # Calculate metrics
    accuracy = correct / len(all_examples)
    throughput = len(all_examples) / generation_time
    
    metrics = {
        'total_examples': len(all_examples),
        'correct': correct,
        'failed_parse': failed_parse,
        'accuracy': accuracy,
        'generation_time': generation_time,
        'throughput': throughput
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'mmlu_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, 'mmlu_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"\nMMLU Evaluation Results:")
    print(f"Total examples: {len(all_examples)}")
    print(f"Correct: {correct}")
    print(f"Failed to parse: {failed_parse}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Throughput: {throughput:.2f} examples/second")
    
    # Sample failed parses
    if failed_parse > 0:
        print(f"\nSample failed parses:")
        failed_examples = [r for r in results if r['predicted'] is None]
        for i, ex in enumerate(failed_examples[:5]):
            print(f"{i+1}. Response: {ex['response'][:100]}...")
    
    # Sample incorrect predictions
    incorrect_examples = [r for r in results if r['predicted'] is not None and not r['correct']]
    if incorrect_examples:
        print(f"\nSample incorrect predictions:")
        for i, ex in enumerate(incorrect_examples[:5]):
            print(f"{i+1}. Question: {ex['question'][:100]}...")
            print(f"   Correct: {ex['answer']}, Predicted: {ex['predicted']}")
            print(f"   Response: {ex['response'][:100]}...")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5-Math-1.5B on MMLU')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the model or HuggingFace model ID')
    parser.add_argument('--data_dir', type=str, default='./data/mmlu',
                        help='Path to MMLU data directory')
    parser.add_argument('--output_dir', type=str, default='./results/mmlu_baseline',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_examples_per_subject', type=int, default=None,
                        help='Max examples per subject for testing')
    
    args = parser.parse_args()
    
    evaluate_mmlu(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_examples_per_subject=args.max_examples_per_subject
    )


if __name__ == "__main__":
    main()