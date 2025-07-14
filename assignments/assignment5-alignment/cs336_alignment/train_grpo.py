#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training Script for MATH dataset.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from pathlib import Path
import json
import logging
import os
from tqdm import tqdm
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal
import pandas as pd
from unittest.mock import patch
import wandb
from collections import defaultdict
import typer
import matplotlib.pyplot as plt

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from .sft_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_entropy
)
from .grpo_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean
)
from .drgrpo_grader import r1_zero_reward_fn
from .train_sft import (
    setup_logging,
    load_r1_zero_prompt,
    load_validation_data,
    init_vllm,
    load_policy_into_vllm_instance,
    evaluate_model
)


class GRPODataset(Dataset):
    """Dataset for GRPO rollout data."""
    
    def __init__(self, rollout_data: List[Dict[str, Any]]):
        """
        Initialize GRPO dataset.
        
        Args:
            rollout_data: List of rollout examples with 'prompt' and 'response' keys
        """
        self.rollout_data = rollout_data
        print(f"Created GRPO dataset with {len(self.rollout_data)} rollouts")
    
    def __len__(self):
        return len(self.rollout_data)
    
    def __getitem__(self, idx):
        example = self.rollout_data[idx]
        return {
            'prompt': example['prompt'],
            'response': example['response']
        }


def load_math_questions(data_path: str, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MATH questions from the dataset."""
    questions = []
    
    try:
        # Load the dataset using pandas from parquet files (same as baseline evaluation)
        print("Loading MATH dataset from allenai/tulu-3-sft-personas-math-filtered...")
        df = pd.read_parquet("hf://datasets/allenai/tulu-3-sft-personas-math-filtered/data/train-00000-of-00001.parquet")
        
        print(f"Loaded {len(df)} examples from the dataset")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
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
                    
                    questions.append({
                        'question': user_message,
                        'answer': assistant_message
                    })
                    
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
                        questions.append({
                            'question': prompt,
                            'answer': solution
                        })
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
            
            # Show progress for large datasets
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(df)} examples, found {len(questions)} valid examples")
        
        print(f"Successfully processed {len(questions)} examples out of {len(df)} total examples")
        
        # Check if we have any valid examples
        if len(questions) == 0:
            print("ERROR: No valid examples found in the dataset!")
            raise ValueError("No valid examples found in the dataset")
        
    except Exception as e:
        print(f"Failed to load online dataset: {e}")
        # Fallback to validation data if online dataset fails
        print("Falling back to validation data...")
        try:
            # Import here to avoid circular imports
            from .train_sft import load_validation_data
            validation_data = load_validation_data(max_examples=max_questions or 5000)
            for item in validation_data:
                questions.append({
                    'question': item['problem'],
                    'answer': item['solution']
                })
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            raise e
    
    if max_questions and len(questions) > max_questions:
        # Use fixed seed to ensure reproducible sampling
        np.random.seed(42)
        indices = np.random.choice(len(questions), max_questions, replace=False)
        questions = [questions[i] for i in sorted(indices)]
    
    print(f"Loaded {len(questions)} questions")
    return questions


def generate_rollouts_with_vllm(
    llm: LLM,
    questions: List[str],
    prompt_template: str,
    group_size: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    min_tokens: int = 4,
    seed: int = 42
) -> List[List[str]]:
    """Generate rollouts using vLLM."""
    
    # Format prompts
    formatted_prompts = []
    for question in questions:
        prompt = prompt_template.format(question=question)
        formatted_prompts.append(prompt)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        n=group_size,
        seed=seed,
        stop=["</answer>"]  # Stop at second answer tag
    )
    
    print(f"Generating {group_size} rollouts for {len(questions)} questions...")
    
    # Generate responses
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs
    all_responses = []
    for output in outputs:
        responses = []
        for completion in output.outputs:
            response = completion.text.strip()
            # Ensure we have the answer tag
            if "</answer>" not in response:
                response += "</answer>"
            responses.append(response)
        all_responses.append(responses)
    
    return all_responses


def collate_fn_grpo(batch, tokenizer):
    """Custom collate function for GRPO DataLoader."""
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    
    # Tokenize and create masks
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return tokenized


def compute_token_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    device: str,
    max_examples: int = 100
) -> float:
    """Compute average token entropy for rollout responses."""
    
    model.eval()
    entropies = []
    
    eval_data = rollout_data[:max_examples]
    
    with torch.no_grad():
        for example in tqdm(eval_data, desc="Computing entropy"):
            prompt = example['prompt']
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
            
            # Generate with the model to get logits
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Compute entropy from the scores
            if outputs.scores:
                token_entropies = []
                for score in outputs.scores:
                    # score shape: (batch_size, vocab_size)
                    entropy = compute_entropy(score.unsqueeze(1)).squeeze()  # Remove seq_len dimension
                    token_entropies.append(entropy.item())
                
                if token_entropies:
                    entropies.append(np.mean(token_entropies))
    
    model.train()
    return np.mean(entropies) if entropies else 0.0


def grpo_training(
    model_path: str,
    data_path: str,
    output_dir: str,
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    device: str = 'cuda',
    eval_device: str = 'cuda:1',
    use_wandb: bool = True,
    wandb_project: str = "grpo-math",
    wandb_run_name: Optional[str] = None,
    seed: int = 42,
    eval_steps: int = 10,
    save_steps: int = 50
):
    """Main GRPO training function."""
    
    # Sanity checks
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    print(f"GRPO Training Configuration:")
    print(f"  Rollout batch size: {rollout_batch_size}")
    print(f"  Group size: {group_size}")
    print(f"  Prompts per rollout batch: {n_prompts_per_rollout_batch}")
    print(f"  Train batch size: {train_batch_size}")
    print(f"  Micro train batch size: {micro_train_batch_size}")
    print(f"  Microbatches per rollout batch: {n_microbatches_per_rollout_batch}")
    
    # Set seeds
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
                "n_grpo_steps": n_grpo_steps,
                "learning_rate": learning_rate,
                "rollout_batch_size": rollout_batch_size,
                "group_size": group_size,
                "loss_type": loss_type,
                "use_std_normalization": use_std_normalization,
                "seed": seed
            }
        )
        
        # Setup metrics
        wandb.define_metric("grpo_step")
        wandb.define_metric("grpo/*", step_metric="grpo_step")
    
    logger.info(f"Starting GRPO training")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"GRPO steps: {n_grpo_steps}")
    logger.info(f"Loss type: {loss_type}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    # Load prompt template
    prompt_template = load_r1_zero_prompt()
    
    # Load questions
    all_questions = load_math_questions(data_path, max_questions=5000)
    question_texts = [q['question'] for q in all_questions]
    ground_truths = [q['answer'] for q in all_questions]
    
    # Load validation data
    validation_examples = load_validation_data(max_examples=1024)
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM on {eval_device}")
    try:
        llm = init_vllm(model_path, eval_device, seed, gpu_memory_utilization)
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        return
    
    # Training metrics
    validation_rewards = []
    training_losses = []
    step_numbers = []
    
    # GRPO Training Loop
    for grpo_step in range(n_grpo_steps):
        logger.info(f"\n{'='*60}")
        logger.info(f"GRPO Step {grpo_step + 1}/{n_grpo_steps}")
        logger.info(f"{'='*60}")
        
        # Step 1: Sample questions for this rollout batch
        step_questions = random.sample(question_texts, n_prompts_per_rollout_batch)
        step_ground_truths = [ground_truths[question_texts.index(q)] for q in step_questions]
        
        # Step 2: Set old policy (copy current weights for GRPO-Clip)
        if loss_type == "grpo_clip":
            # Store old policy state
            old_policy_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Step 3: Load current policy weights into vLLM
        logger.info("Loading policy weights into vLLM...")
        load_policy_into_vllm_instance(model, llm)
        
        # Step 4: Generate rollouts
        logger.info(f"Generating {group_size} rollouts for {len(step_questions)} questions...")
        all_rollouts = generate_rollouts_with_vllm(
            llm=llm,
            questions=step_questions,
            prompt_template=prompt_template,
            group_size=group_size,
            temperature=sampling_temperature,
            max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens,
            seed=seed + grpo_step
        )
        
        # Step 5: Flatten rollouts and create repeated ground truths
        rollout_responses = []
        repeated_ground_truths = []
        rollout_prompts = []
        
        for question_idx, (question, rollouts) in enumerate(zip(step_questions, all_rollouts)):
            for rollout in rollouts:
                rollout_responses.append(rollout)
                repeated_ground_truths.append(step_ground_truths[question_idx])
                rollout_prompts.append(question)
        
        # Step 6: Compute rewards and advantages
        logger.info("Computing rewards and advantages...")
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        
        logger.info(f"Reward statistics:")
        logger.info(f"  Mean reward: {reward_metadata['mean_reward']:.3f}")
        logger.info(f"  Mean format reward: {reward_metadata['mean_format_reward']:.3f}")
        logger.info(f"  Mean answer reward: {reward_metadata['mean_answer_reward']:.3f}")
        
        # Step 7: Create rollout dataset and dataloader
        rollout_data = []
        for prompt, response in zip(rollout_prompts, rollout_responses):
            rollout_data.append({'prompt': prompt, 'response': response})
        
        dataset = GRPODataset(rollout_data)
        dataloader = DataLoader(
            dataset,
            batch_size=micro_train_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn_grpo(batch, tokenizer),
            num_workers=0
        )
        
        # Step 8: Get old log probabilities (for GRPO-Clip)
        old_log_probs_all = None
        if loss_type == "grpo_clip":
            logger.info("Computing old log probabilities...")
            model.eval()
            old_log_probs_list = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = get_response_log_probs(model, input_ids, labels)
                    old_log_probs_list.append(outputs['log_probs'])
            
            old_log_probs_all = torch.cat(old_log_probs_list, dim=0)
            model.train()
        
        # Step 9: Training epochs on rollout batch
        logger.info(f"Training for {epochs_per_rollout_batch} epochs...")
        
        epoch_losses = []
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            
            batch_losses = []
            gradient_norm = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                response_mask = batch['response_mask'].to(device)
                
                # Get batch indices for rewards/advantages
                start_idx = batch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                
                batch_raw_rewards = raw_rewards[start_idx:end_idx].unsqueeze(1)  # (batch_size, 1)
                batch_advantages = advantages[start_idx:end_idx].unsqueeze(1)  # (batch_size, 1)
                
                # Get old log probs for this batch (if needed)
                batch_old_log_probs = None
                if loss_type == "grpo_clip" and old_log_probs_all is not None:
                    batch_old_log_probs = old_log_probs_all[start_idx:end_idx]
                
                # Forward pass to get current log probabilities
                outputs = get_response_log_probs(model, input_ids, labels)
                policy_log_probs = outputs['log_probs']
                
                # GRPO microbatch training step
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=batch_raw_rewards if loss_type == "no_baseline" else None,
                    advantages=batch_advantages if loss_type != "no_baseline" else None,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange if loss_type == "grpo_clip" else None
                )
                
                batch_losses.append(loss.item())
                
                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_loss = np.mean(batch_losses) if batch_losses else 0.0
            epoch_losses.append(epoch_loss)
            
            logger.info(f"  Epoch {epoch + 1}/{epochs_per_rollout_batch}: loss = {epoch_loss:.4f}")
        
        # Step 10: Compute token entropy
        token_entropy = compute_token_entropy(model, tokenizer, rollout_data[:50], device)
        
        # Step 11: Validation evaluation
        if (grpo_step + 1) % eval_steps == 0:
            logger.info("Running validation evaluation...")
            eval_results = evaluate_model(
                model, tokenizer, validation_examples, 
                prompt_template, device, max_eval_examples=200
            )
            
            validation_rewards.append(eval_results['accuracy'])
            step_numbers.append(grpo_step + 1)
            
            logger.info(f"Validation accuracy: {eval_results['accuracy']:.3f}")
        
        # Step 12: Logging
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        training_losses.append(avg_loss)
        
        logger.info(f"Step {grpo_step + 1} Summary:")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Token entropy: {token_entropy:.3f}")
        logger.info(f"  Gradient norm: {gradient_norm:.3f}")
        
        if use_wandb:
            log_dict = {
                "grpo/step": grpo_step + 1,
                "grpo/loss": avg_loss,
                "grpo/token_entropy": token_entropy,
                "grpo/gradient_norm": gradient_norm,
                "grpo/mean_reward": reward_metadata['mean_reward'],
                "grpo/mean_format_reward": reward_metadata['mean_format_reward'],
                "grpo/mean_answer_reward": reward_metadata['mean_answer_reward'],
                "grpo_step": grpo_step + 1
            }
            
            if validation_rewards and step_numbers and step_numbers[-1] == grpo_step + 1:
                log_dict["grpo/validation_accuracy"] = validation_rewards[-1]
            
            wandb.log(log_dict)
        
        # Step 13: Save checkpoint
        if (grpo_step + 1) % save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"grpo_step_{grpo_step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval_results = evaluate_model(
        model, tokenizer, validation_examples, 
        prompt_template, device, max_eval_examples=1024
    )
    
    logger.info(f"Final validation accuracy: {final_eval_results['accuracy']:.3f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training results
    results = {
        "validation_rewards": validation_rewards,
        "training_losses": training_losses,
        "step_numbers": step_numbers,
        "final_accuracy": final_eval_results['accuracy']
    }
    
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot validation rewards
    if validation_rewards and step_numbers:
        plt.figure(figsize=(10, 6))
        plt.plot(step_numbers, validation_rewards, marker='o', linewidth=2)
        plt.xlabel('GRPO Step')
        plt.ylabel('Validation Accuracy')
        plt.title('GRPO Training: Validation Accuracy vs Steps')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "validation_rewards_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation rewards plot to {plot_path}")
    
    if use_wandb:
        wandb.log({
            "final/validation_accuracy": final_eval_results['accuracy'],
        })
        wandb.finish()
    
    logger.info("GRPO training completed!")
    return final_eval_results


def main():
    parser = argparse.ArgumentParser(description='GRPO Training Script')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the base model')
    parser.add_argument('--data_path', type=str, default='./data/math_train.jsonl',
                        help='Path to the MATH training data')
    parser.add_argument('--output_dir', type=str, default='./grpo_results',
                        help='Directory to save results')
    parser.add_argument('--n_grpo_steps', type=int, default=200,
                        help='Number of GRPO steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--advantage_eps', type=float, default=1e-6,
                        help='Small constant for advantage computation')
    parser.add_argument('--rollout_batch_size', type=int, default=256,
                        help='Rollout batch size')
    parser.add_argument('--group_size', type=int, default=8,
                        help='Group size for rollouts')
    parser.add_argument('--sampling_temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--sampling_max_tokens', type=int, default=1024,
                        help='Max tokens for sampling')
    parser.add_argument('--epochs_per_rollout_batch', type=int, default=1,
                        help='Training epochs per rollout batch')
    parser.add_argument('--train_batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=128,
                        help='Gradient accumulation steps')
    parser.add_argument('--loss_type', type=str, default='reinforce_with_baseline',
                        choices=['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
                        help='Type of loss to use')
    parser.add_argument('--use_std_normalization', action='store_true',
                        help='Use standard deviation normalization')
    parser.add_argument('--cliprange', type=float, default=0.2,
                        help='Clipping range for GRPO-Clip')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device')
    parser.add_argument('--eval_device', type=str, default='cuda:1',
                        help='Evaluation device')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='grpo-math',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_steps', type=int, default=10,
                        help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
        args.eval_device = 'cpu'
    
    # Run GRPO training
    grpo_training(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        cliprange=args.cliprange,
        device=args.device,
        eval_device=args.eval_device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )


if __name__ == '__main__':
    main()