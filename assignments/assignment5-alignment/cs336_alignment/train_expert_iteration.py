#!/usr/bin/env python3
"""
Expert Iteration Training Script for MATH dataset.
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
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from unittest.mock import patch
import wandb
from collections import defaultdict

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from .sft_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_entropy
)
from .drgrpo_grader import r1_zero_reward_fn
from .train_sft import (
    MATHDataset,
    setup_logging,
    load_r1_zero_prompt,
    load_validation_data,
    init_vllm,
    load_policy_into_vllm_instance,
    evaluate_model,
    collate_fn
)


class ExpertIterationDataset(Dataset):
    """Dataset for Expert Iteration generated examples."""
    
    def __init__(self, examples: List[Dict[str, Any]]):
        """
        Initialize Expert Iteration dataset.
        
        Args:
            examples: List of examples with 'prompt' and 'response' keys
        """
        self.examples = examples
        print(f"Created EI dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
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
        validation_data = load_validation_data(max_examples=max_questions or 5000)
        for item in validation_data:
            questions.append({
                'question': item['problem'],
                'answer': item['solution']
            })
    
    if max_questions and len(questions) > max_questions:
        # Use fixed seed to ensure reproducible sampling
        np.random.seed(42)
        indices = np.random.choice(len(questions), max_questions, replace=False)
        questions = [questions[i] for i in sorted(indices)]
    
    print(f"Loaded {len(questions)} questions")
    return questions


def generate_responses_with_vllm(
    llm: LLM, 
    questions: List[str], 
    prompt_template: str,
    num_samples: int = 4,
    temperature: float = 1.0,
    max_tokens: int = 512,
    min_tokens: int = 4,
    seed: int = 42
) -> List[List[str]]:
    """Generate responses using vLLM."""
    
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
        n=num_samples,
        seed=seed,
        stop=["</answer>"]  # Stop at second answer tag
    )
    
    print(f"Generating {num_samples} responses for {len(questions)} questions...")
    
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


def evaluate_responses(
    questions: List[str],
    responses: List[List[str]],
    ground_truths: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Evaluate generated responses and filter correct ones."""
    
    correct_examples = []
    stats = defaultdict(list)
    
    for question, response_list, ground_truth in zip(questions, responses, ground_truths):
        for response in response_list:
            try:
                # Evaluate using reward function
                reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
                
                stats['format_rewards'].append(reward_dict['format_reward'])
                stats['answer_rewards'].append(reward_dict['answer_reward'])
                stats['total_rewards'].append(reward_dict['total_reward'])
                
                # Keep only correct responses
                if reward_dict['total_reward'] > 0:
                    correct_examples.append({
                        'prompt': question,
                        'response': response,
                        'reward': reward_dict['total_reward']
                    })
            except Exception as e:
                print(f"Error evaluating response: {e}")
                continue
    
    # Compute statistics
    eval_stats = {
        'total_responses': sum(len(r) for r in responses),
        'correct_responses': len(correct_examples),
        'format_accuracy': np.mean(stats['format_rewards']) if stats['format_rewards'] else 0.0,
        'answer_accuracy': np.mean(stats['answer_rewards']) if stats['answer_rewards'] else 0.0,
        'total_accuracy': np.mean(stats['total_rewards']) if stats['total_rewards'] else 0.0
    }
    
    print(f"Evaluation results:")
    print(f"  Total responses: {eval_stats['total_responses']}")
    print(f"  Correct responses: {eval_stats['correct_responses']}")
    print(f"  Format accuracy: {eval_stats['format_accuracy']:.3f}")
    print(f"  Answer accuracy: {eval_stats['answer_accuracy']:.3f}")
    print(f"  Total accuracy: {eval_stats['total_accuracy']:.3f}")
    
    return correct_examples, eval_stats


def compute_response_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    prompt_template: str,
    device: str,
    max_examples: int = 100
) -> Dict[str, float]:
    """Compute entropy of model responses."""
    
    model.eval()
    entropies = []
    
    eval_questions = questions[:max_examples]
    
    with torch.no_grad():
        for question in tqdm(eval_questions, desc="Computing entropy"):
            prompt = prompt_template.format(question=question)
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
    
    return {
        'mean_entropy': np.mean(entropies) if entropies else 0.0,
        'std_entropy': np.std(entropies) if entropies else 0.0,
        'num_samples': len(entropies)
    }


def run_sft_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    correct_examples: List[Dict[str, Any]],
    device: str,
    num_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 50,
    gradient_clip_value: float = 1.0
) -> Dict[str, float]:
    """Run SFT step on correct examples."""
    
    if not correct_examples:
        print("No correct examples for SFT step, skipping...")
        return {"loss": 0.0, "num_examples": 0}
    
    print(f"Running SFT step on {len(correct_examples)} correct examples...")
    
    # Create dataset
    dataset = ExpertIterationDataset(correct_examples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    
    if total_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for step, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            response_mask = batch['response_mask'].to(device)
            
            # Forward pass
            outputs = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = outputs['log_probs']
            
            # Calculate loss
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                normalize_constant=1.0
            )
            
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
        
        if epoch_losses:
            losses.extend(epoch_losses)
            print(f"  Epoch {epoch + 1}/{num_epochs}: avg loss = {np.mean(epoch_losses):.4f}")
    
    return {
        "loss": np.mean(losses) if losses else 0.0,
        "num_examples": len(correct_examples)
    }


def expert_iteration_training(
    model_path: str,
    data_path: str,
    output_dir: str,
    n_ei_steps: int = 5,
    batch_size_per_step: int = 1024,
    num_rollouts: int = 4,
    sft_epochs: int = 1,
    sft_batch_size: int = 2,
    sft_gradient_accumulation_steps: int = 8,
    sft_learning_rate: float = 5e-5,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 512,
    device: str = 'cuda',
    eval_device: str = 'cuda:1',
    use_wandb: bool = True,
    wandb_project: str = "expert-iteration-math",
    wandb_run_name: Optional[str] = None,
    seed: int = 42
):
    """Main Expert Iteration training function."""
    
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
                "n_ei_steps": n_ei_steps,
                "batch_size_per_step": batch_size_per_step,
                "num_rollouts": num_rollouts,
                "sft_epochs": sft_epochs,
                "sampling_temperature": sampling_temperature,
                "seed": seed
            }
        )
        
        # Setup metrics
        wandb.define_metric("ei_step")
        wandb.define_metric("ei/*", step_metric="ei_step")
    
    logger.info(f"Starting Expert Iteration training")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"EI steps: {n_ei_steps}")
    logger.info(f"Batch size per step: {batch_size_per_step}")
    logger.info(f"Number of rollouts: {num_rollouts}")
    
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
    
    # Load prompt template
    prompt_template = load_r1_zero_prompt()
    
    # Load questions
    all_questions = load_math_questions(data_path, max_questions=5000)
    question_texts = [q['question'] for q in all_questions]
    ground_truths = [q['answer'] for q in all_questions]
    
    # Load validation data
    validation_examples = load_validation_data(max_examples=500)
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM on {eval_device}")
    try:
        llm = init_vllm(model_path, eval_device, seed)
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        return
    
    # Expert Iteration Loop
    for ei_step in range(n_ei_steps):
        logger.info(f"\n{'='*50}")
        logger.info(f"Expert Iteration Step {ei_step + 1}/{n_ei_steps}")
        logger.info(f"{'='*50}")
        
        # Sample questions for this step
        step_questions = random.sample(question_texts, min(batch_size_per_step, len(question_texts)))
        step_ground_truths = [ground_truths[question_texts.index(q)] for q in step_questions]
        
        # Load current policy weights into vLLM
        logger.info("Loading policy weights into vLLM...")
        load_policy_into_vllm_instance(model, llm)
        
        # Generate responses
        logger.info(f"Generating {num_rollouts} responses per question...")
        all_responses = generate_responses_with_vllm(
            llm=llm,
            questions=step_questions,
            prompt_template=prompt_template,
            num_samples=num_rollouts,
            temperature=sampling_temperature,
            max_tokens=sampling_max_tokens,
            min_tokens=4,
            seed=seed + ei_step
        )
        
        # Evaluate responses and filter correct ones
        logger.info("Evaluating generated responses...")
        correct_examples, eval_stats = evaluate_responses(
            step_questions, all_responses, step_ground_truths
        )
        
        # Compute entropy
        logger.info("Computing response entropy...")
        entropy_stats = compute_response_entropy(
            model, tokenizer, step_questions[:100], prompt_template, device
        )
        
        # Run SFT step on correct examples
        logger.info("Running SFT step...")
        sft_stats = run_sft_step(
            model=model,
            tokenizer=tokenizer,
            correct_examples=correct_examples,
            device=device,
            num_epochs=sft_epochs,
            batch_size=sft_batch_size,
            gradient_accumulation_steps=sft_gradient_accumulation_steps,
            learning_rate=sft_learning_rate
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_results = evaluate_model(
            model, tokenizer, validation_examples, 
            prompt_template, device, max_eval_examples=200
        )
        
        # Log results
        logger.info(f"Step {ei_step + 1} Results:")
        logger.info(f"  Generated responses: {eval_stats['total_responses']}")
        logger.info(f"  Correct responses: {eval_stats['correct_responses']}")
        logger.info(f"  Generation accuracy: {eval_stats['total_accuracy']:.3f}")
        logger.info(f"  SFT loss: {sft_stats['loss']:.4f}")
        logger.info(f"  Validation accuracy: {eval_results['accuracy']:.3f}")
        logger.info(f"  Mean entropy: {entropy_stats['mean_entropy']:.3f}")
        
        if use_wandb:
            wandb.log({
                "ei/step": ei_step + 1,
                "ei/generated_responses": eval_stats['total_responses'],
                "ei/correct_responses": eval_stats['correct_responses'],
                "ei/generation_accuracy": eval_stats['total_accuracy'],
                "ei/sft_loss": sft_stats['loss'],
                "ei/validation_accuracy": eval_results['accuracy'],
                "ei/validation_format_accuracy": eval_results['format_accuracy'],
                "ei/mean_entropy": entropy_stats['mean_entropy'],
                "ei/std_entropy": entropy_stats['std_entropy'],
                "ei_step": ei_step + 1
            })
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"ei_step_{ei_step + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save step results
        step_results = {
            "ei_step": ei_step + 1,
            "eval_stats": eval_stats,
            "sft_stats": sft_stats,
            "eval_results": eval_results,
            "entropy_stats": entropy_stats
        }
        
        with open(os.path.join(checkpoint_dir, "step_results.json"), "w") as f:
            json.dump(step_results, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval_results = evaluate_model(
        model, tokenizer, validation_examples, 
        prompt_template, device, max_eval_examples=500
    )
    
    logger.info(f"Final Results:")
    logger.info(f"  Validation accuracy: {final_eval_results['accuracy']:.3f}")
    logger.info(f"  Format accuracy: {final_eval_results['format_accuracy']:.3f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    if use_wandb:
        wandb.log({
            "final/validation_accuracy": final_eval_results['accuracy'],
            "final/format_accuracy": final_eval_results['format_accuracy'],
        })
        wandb.finish()
    
    logger.info("Expert Iteration training completed!")
    return final_eval_results


def main():
    parser = argparse.ArgumentParser(description='Expert Iteration Training Script')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the base model')
    parser.add_argument('--data_path', type=str, default='./data/math_train.jsonl',
                        help='Path to the MATH training data')
    parser.add_argument('--output_dir', type=str, default='./ei_results',
                        help='Directory to save results')
    parser.add_argument('--n_ei_steps', type=int, default=5,
                        help='Number of expert iteration steps')
    parser.add_argument('--batch_size_per_step', type=int, default=1024,
                        help='Number of questions per EI step')
    parser.add_argument('--num_rollouts', type=int, default=4,
                        help='Number of rollouts per question')
    parser.add_argument('--sft_epochs', type=int, default=1,
                        help='Number of SFT epochs per step')
    parser.add_argument('--sft_batch_size', type=int, default=2,
                        help='SFT batch size')
    parser.add_argument('--sft_gradient_accumulation_steps', type=int, default=8,
                        help='SFT gradient accumulation steps')
    parser.add_argument('--sft_learning_rate', type=float, default=5e-5,
                        help='SFT learning rate')
    parser.add_argument('--sampling_temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--sampling_max_tokens', type=int, default=512,
                        help='Max tokens for sampling')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device')
    parser.add_argument('--eval_device', type=str, default='cuda:1',
                        help='Evaluation device')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='expert-iteration-math',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
        args.eval_device = 'cpu'
    
    # Run expert iteration
    expert_iteration_training(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_ei_steps=args.n_ei_steps,
        batch_size_per_step=args.batch_size_per_step,
        num_rollouts=args.num_rollouts,
        sft_epochs=args.sft_epochs,
        sft_batch_size=args.sft_batch_size,
        sft_gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        sft_learning_rate=args.sft_learning_rate,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        device=args.device,
        eval_device=args.eval_device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed
    )


if __name__ == '__main__':
    main()