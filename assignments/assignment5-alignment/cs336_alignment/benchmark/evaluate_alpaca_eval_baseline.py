#!/usr/bin/env python3
"""
AlpacaEval baseline evaluation script for Llama 3.1 8B.
"""

import json
import os
import time
from typing import List
import argparse

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


def load_alpaca_eval_data(data_file: str) -> List[dict]:
    """Load AlpacaEval data."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_alpaca_eval_prompt(instruction: str, system_prompt: str) -> str:
    """Format AlpacaEval instruction into a prompt."""
    full_prompt = system_prompt + "\n\n" + instruction
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


def evaluate_alpaca_eval(
    model_path: str,
    data_file: str,
    output_file: str,
    device: str = "cuda",
    seed: int = 42,
    max_examples: int = None
):
    """Generate outputs for AlpacaEval."""
    
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
    print(f"Loading AlpacaEval data from {data_file}...")
    eval_set = load_alpaca_eval_data(data_file)
    
    if max_examples:
        eval_set = eval_set[:max_examples]
    
    print(f"Total examples: {len(eval_set)}")
    
    # Generate responses
    print("Generating responses...")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["# Query:"]
    )
    
    prompts = [format_alpaca_eval_prompt(ex['instruction'], system_prompt) for ex in eval_set]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time
    
    # Process results
    for i, (example, output) in enumerate(zip(eval_set, outputs)):
        response = output.outputs[0].text.strip()
        example["output"] = response
        example["generator"] = "llama-3.1-8b-base"
    
    # Calculate throughput
    throughput = len(eval_set) / generation_time
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nAlpacaEval Generation Results:")
    print(f"Total examples: {len(eval_set)}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Results saved to: {output_file}")
    
    # Sample a few examples
    print(f"\nSample outputs:")
    for i, ex in enumerate(eval_set[:3]):
        print(f"{i+1}. Instruction: {ex['instruction'][:100]}...")
        print(f"   Output: {ex['output'][:150]}...")
        print()
    
    return {
        'total_examples': len(eval_set),
        'generation_time': generation_time,
        'throughput': throughput
    }


def main():
    parser = argparse.ArgumentParser(description='Generate AlpacaEval outputs for Qwen2.5-Math-1.5B')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the model or HuggingFace model ID')
    parser.add_argument('--data_file', type=str, default='./data/alpaca_eval/alpaca_eval.jsonl',
                        help='Path to AlpacaEval data file')
    parser.add_argument('--output_file', type=str, default='./results/alpaca_eval_baseline/alpaca_eval_outputs.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Max examples for testing')
    
    args = parser.parse_args()
    
    evaluate_alpaca_eval(
        model_path=args.model_path,
        data_file=args.data_file,
        output_file=args.output_file,
        device=args.device,
        seed=args.seed,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()