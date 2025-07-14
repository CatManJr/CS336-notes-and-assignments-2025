#!/usr/bin/env python3
"""
SFT Training Script for MATH dataset.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import json
import logging
import os
from tqdm import tqdm
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from unittest.mock import patch
import wandb

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from .sft_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_entropy
)
from .drgrpo_grader import r1_zero_reward_fn


class MATHDataset(Dataset):
    """Dataset for MATH SFT examples."""
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """
        Initialize MATH dataset.
        
        Args:
            data_path: Path to the JSONL file or directory containing train/test splits
            max_examples: Maximum number of examples to use
        """
        self.examples = []
        
        if os.path.isdir(data_path):
            # Load from directory with train/test splits
            train_file = os.path.join(data_path, 'train.jsonl')
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    for line in f:
                        self.examples.append(json.loads(line.strip()))
            else:
                raise ValueError(f"No train.jsonl found in {data_path}")
        else:
            # Load from single JSONL file
            with open(data_path, 'r') as f:
                for line in f:
                    self.examples.append(json.loads(line.strip()))
        
        if max_examples and len(self.examples) > max_examples:
            self.examples = self.examples[:max_examples]
            
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'prompt': example['prompt'],
            'response': example['response']
        }


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path(__file__).parent / 'prompts/r1_zero.prompt'
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_validation_data(max_examples: int = 500) -> List[Dict[str, Any]]:
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
            raise ValueError("No valid examples found in the dataset")
        
        # Apply max_examples limit if specified (before splitting)
        if max_examples and len(all_examples) > max_examples:
            print(f"Limiting processed examples to {max_examples} (from {len(all_examples)})")
            # Use fixed seed to ensure reproducible sampling
            np.random.seed(42)
            indices = np.random.choice(len(all_examples), max_examples, replace=False)
            all_examples = [all_examples[i] for i in sorted(indices)]
        
        # Split the dataset into train and test sets based on dataset length
        if len(all_examples) > 1 and 0.2 > 0:
            print(f"Splitting dataset with test_size={0.2}, random_state={42}")
            
            from sklearn.model_selection import train_test_split
            
            # Calculate test size based on dataset length
            test_count = int(len(all_examples) * 0.2)
            train_count = len(all_examples) - test_count
            
            print(f"Total examples: {len(all_examples)}")
            print(f"Test examples: {test_count} ({test_count/len(all_examples):.1%})")
            print(f"Train examples: {train_count} ({train_count/len(all_examples):.1%})")
            
            train_examples, test_examples = train_test_split(
                all_examples, 
                test_size=0.2, 
                random_state=42,
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


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Load policy weights into vLLM instance.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_model(model, tokenizer, validation_examples: List[Dict[str, Any]], 
                  prompt_template: str, device: str, max_eval_examples: int = 100):
    """Evaluate model on validation examples."""
    if not validation_examples:
        return {"accuracy": 0.0, "format_accuracy": 0.0}
    
    eval_examples = validation_examples[:max_eval_examples]
    
    # Format prompts
    prompts = []
    ground_truths = []
    for example in eval_examples:
        prompt = prompt_template.format(question=example['problem'])
        prompts.append(prompt)
        ground_truths.append(example['solution'])
    
    # Generate responses
    model.eval()
    generated_responses = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Evaluating"):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated_responses.append(response)
    
    # Evaluate using reward function
    correct_answers = 0
    correct_format = 0
    
    for response, ground_truth in zip(generated_responses, ground_truths):
        try:
            reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
            if reward_dict['format_reward'] == 1.0:
                correct_format += 1
                if reward_dict['answer_reward'] == 1.0:
                    correct_answers += 1
        except Exception as e:
            continue
    
    accuracy = correct_answers / len(eval_examples) if eval_examples else 0.0
    format_accuracy = correct_format / len(eval_examples) if eval_examples else 0.0
    
    model.train()
    return {
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "num_examples": len(eval_examples)
    }


def collate_fn(batch, tokenizer):
    """Custom collate function for DataLoader."""
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    
    # Tokenize and create masks
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return tokenized


def create_dataset_splits(base_data_path: str, output_dir: str, test_size: float = 0.2):
    """Create train/test splits from the base dataset."""
    # Load validation data and create splits
    validation_examples = load_validation_data(max_examples=10000)
    
    # Split into train and test
    np.random.seed(42)
    indices = np.random.permutation(len(validation_examples))
    test_count = int(len(validation_examples) * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    train_examples = [validation_examples[i] for i in train_indices]
    test_examples = [validation_examples[i] for i in test_indices]
    
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
    
    print(f"Created dataset splits: {len(train_sft)} train, {len(test_sft)} test")
    return len(train_sft), len(test_sft)


def train_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    max_seq_length: int = 1024,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 1000,
    device: str = 'cuda',
    eval_device: str = 'cuda:1',
    max_examples: Optional[int] = None,
    use_wandb: bool = True,
    wandb_project: str = "sft-math",
    wandb_run_name: Optional[str] = None,
    gradient_clip_value: float = 1.0,
    seed: int = 42
):
    """Main SFT training function."""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_path": model_path,
                "data_path": data_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_examples": max_examples,
                "seed": seed
            }
        )
        
        # Setup wandb metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
    
    logger.info(f"Starting SFT training")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Max examples: {max_examples}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Training device: {device}")
    logger.info(f"Eval device: {eval_device}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.train()
    
    # Load prompt template
    prompt_template = load_r1_zero_prompt()
    logger.info(f"Loaded prompt template")
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = MATHDataset(data_path, max_examples=max_examples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0
    )
    
    # Load validation data
    validation_examples = load_validation_data(max_examples=500)
    
    # Initialize vLLM for evaluation (if different device)
    vllm_model = None
    if eval_device != device and torch.cuda.device_count() > 1:
        try:
            logger.info(f"Initializing vLLM on {eval_device}")
            vllm_model = init_vllm(model_path, eval_device, seed)
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM: {e}")
            vllm_model = None
    
    # Setup optimizer and scheduler
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Training loop
    global_step = 0
    train_step = 0
    eval_step = 0
    
    logger.info("Starting training loop...")
    
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            response_mask = batch['response_mask'].to(device)
            
            # Forward pass to get log probabilities
            with torch.cuda.amp.autocast():
                outputs = get_response_log_probs(model, input_ids, labels)
                policy_log_probs = outputs['log_probs']
                
                # Calculate loss using our microbatch function
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    gradient_accumulation_steps,
                    normalize_constant=1.0
                )
            
            # Take optimizer step every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                train_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    current_loss = loss.item() * gradient_accumulation_steps
                    logger.info(
                        f"Step {global_step}: loss={current_loss:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    
                    if use_wandb:
                        wandb.log({
                            "train/loss": current_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train_step": train_step
                        })
                
                # Evaluation
                if global_step > 0 and global_step % eval_steps == 0:
                    logger.info("Running evaluation...")
                    
                    # Load current weights into vLLM if available
                    if vllm_model is not None:
                        load_policy_into_vllm_instance(model, vllm_model)
                    
                    # Evaluate model
                    eval_results = evaluate_model(
                        model, tokenizer, validation_examples, 
                        prompt_template, device, max_eval_examples=100
                    )
                    
                    eval_step += 1
                    
                    logger.info(
                        f"Eval step {eval_step}: "
                        f"accuracy={eval_results['accuracy']:.3f}, "
                        f"format_accuracy={eval_results['format_accuracy']:.3f}"
                    )
                    
                    if use_wandb:
                        wandb.log({
                            "eval/accuracy": eval_results['accuracy'],
                            "eval/format_accuracy": eval_results['format_accuracy'],
                            "eval_step": eval_step
                        })
                
                # Save checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        logger.info(f"Epoch {epoch + 1} completed.")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval_results = evaluate_model(
        model, tokenizer, validation_examples, 
        prompt_template, device, max_eval_examples=500
    )
    
    logger.info(
        f"Final evaluation: "
        f"accuracy={final_eval_results['accuracy']:.3f}, "
        f"format_accuracy={final_eval_results['format_accuracy']:.3f}"
    )
    
    if use_wandb:
        wandb.log({
            "final/accuracy": final_eval_results['accuracy'],
            "final/format_accuracy": final_eval_results['format_accuracy'],
        })
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")
    
    if use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")
    return final_eval_results


def main():
    parser = argparse.ArgumentParser(description='SFT Training Script for MATH Dataset')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the base model')
    parser.add_argument('--data_path', type=str, default='./data/math_sft',
                        help='Path to the dataset directory or JSONL file')
    parser.add_argument('--output_dir', type=str, default='./sft_results',
                        help='Directory to save trained model and logs')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--eval_device', type=str, default='cuda:1',
                        help='Device to use for evaluation')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of training examples to use')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='sft-math',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--create_splits', action='store_true',
                        help='Create train/test splits from validation data')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
        args.eval_device = 'cpu'
    
    # Create dataset splits if requested
    if args.create_splits:
        print("Creating dataset splits...")
        create_dataset_splits(args.data_path, args.data_path)
    
    # Run different dataset sizes for the experiment
    dataset_sizes = [128, 256, 512, 1024, None]  # None means full dataset
    
    for size in dataset_sizes:
        print(f"\n{'='*50}")
        print(f"Training with dataset size: {size if size else 'full'}")
        print(f"{'='*50}")
        
        run_name = f"{args.wandb_run_name}_size_{size if size else 'full'}" if args.wandb_run_name else f"sft_size_{size if size else 'full'}"
        output_dir = os.path.join(args.output_dir, f"size_{size if size else 'full'}")
        
        train_sft(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            device=args.device,
            eval_device=args.eval_device,
            max_examples=size,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=run_name,
            gradient_clip_value=args.gradient_clip_value,
            seed=args.seed
        )


if __name__ == '__main__':
    main()