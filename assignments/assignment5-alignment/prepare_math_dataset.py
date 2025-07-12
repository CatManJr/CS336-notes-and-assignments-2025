#!/usr/bin/env python3
"""
Script to prepare MATH dataset splits for SFT training.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import argparse
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_validation_data(max_examples: int = 10000000) -> List[Dict[str, Any]]:
    """Load validation data from the dataset used in evaluate_math_baseline.py."""
    print("Loading validation data...")
    
    try:
        # Load the dataset using pandas from parquet files
        df = pd.read_parquet("hf://datasets/allenai/tulu-3-sft-personas-math-filtered/data/train-00000-of-00001.parquet")
        
        all_examples = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if len(all_examples) >= max_examples:
                break
                
            try:
                if "messages" in row and row["messages"] is not None:
                    messages = row["messages"]
                    
                    user_message = ""
                    assistant_message = ""
                    
                    for msg in messages:
                        if isinstance(msg, dict):
                            if msg.get("role") == "user":
                                user_message = msg.get("content", "")
                            elif msg.get("role") == "assistant":
                                assistant_message = msg.get("content", "")
                    
                    if user_message and assistant_message:
                        all_examples.append({
                            "problem": user_message,
                            "solution": assistant_message
                        })
            except Exception as e:
                continue
        
        print(f"Loaded {len(all_examples)} examples from dataset")
        return all_examples
        
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return []


def filter_correct_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter examples to only include those that produce correct answers."""
    print("Filtering examples for correct answers...")
    
    correct_examples = []
    for i, example in enumerate(examples):
        try:
            # Use the r1_zero_reward_fn to evaluate if the solution is correct
            reward_dict = r1_zero_reward_fn(example['solution'], example['solution'], fast=True)
            
            # Keep examples where both format and answer are correct
            if reward_dict['format_reward'] == 1.0 and reward_dict['answer_reward'] == 1.0:
                correct_examples.append(example)
                
        except Exception as e:
            # Skip examples that cause errors
            continue
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples, found {len(correct_examples)} correct")
    
    print(f"Found {len(correct_examples)} correct examples out of {len(examples)} total ({len(correct_examples)/len(examples):.2%})")
    return correct_examples


def create_dataset_splits(output_dir: str, test_size: float = 0.2, max_examples: int = 10000, 
                         filter_correct: bool = False):
    """Create train/test splits from the base dataset."""
    
    # Load data
    all_examples = load_validation_data(max_examples=max_examples)
    
    if not all_examples:
        print("No examples loaded, exiting...")
        return
    
    # Filter for correct examples if requested
    if filter_correct:
        all_examples = filter_correct_examples(all_examples)
        if not all_examples:
            print("No correct examples found, exiting...")
            return
    
    # Split into train and test
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    test_count = int(len(all_examples) * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    train_examples = [all_examples[i] for i in train_indices]
    test_examples = [all_examples[i] for i in test_indices]
    
    # Convert to SFT format
    train_sft = []
    test_sft = []
    
    for example in train_examples:
        train_sft.append({
            "prompt": example["problem"],
            "response": example["solution"]
        })
    
    for example in test_examples:
        test_sft.append({
            "prompt": example["problem"],
            "response": example["solution"]
        })
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.jsonl'), 'w') as f:
        for example in train_sft:
            f.write(json.dumps(example) + '\n')
    
    with open(os.path.join(output_dir, 'test.jsonl'), 'w') as f:
        for example in test_sft:
            f.write(json.dumps(example) + '\n')
    
    # Save metadata
    metadata = {
        "total_examples": len(all_examples),
        "train_examples": len(train_sft),
        "test_examples": len(test_sft),
        "test_size": test_size,
        "filtered_for_correct": filter_correct,
        "max_examples_loaded": max_examples
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created dataset splits:")
    print(f"  Train: {len(train_sft)} examples")
    print(f"  Test: {len(test_sft)} examples")
    print(f"  Total: {len(all_examples)} examples")
    print(f"  Filtered for correct: {filter_correct}")
    print(f"  Saved to: {output_dir}")
    
    return len(train_sft), len(test_sft)


def main():
    parser = argparse.ArgumentParser(description='Prepare MATH dataset splits for SFT training')
    parser.add_argument('--output_dir', type=str, default='./data/math_sft',
                        help='Directory to save dataset splits')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--max_examples', type=int, default=10000,
                        help='Maximum number of examples to load')
    parser.add_argument('--filter_correct', action='store_true',
                        help='Filter to only include examples with correct answers')
    
    args = parser.parse_args()
    
    print("Preparing MATH dataset splits...")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Max examples: {args.max_examples}")
    print(f"Filter correct: {args.filter_correct}")
    
    create_dataset_splits(
        output_dir=args.output_dir,
        test_size=args.test_size,
        max_examples=args.max_examples,
        filter_correct=args.filter_correct
    )
    
    print("Dataset preparation complete!")


if __name__ == '__main__':
    main()