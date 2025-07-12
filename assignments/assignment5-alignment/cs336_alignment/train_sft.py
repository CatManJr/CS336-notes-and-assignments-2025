#!/usr/bin/env python3
"""
SFT Training Script.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import json
import logging
import os
from tqdm import tqdm
import time

from .sft_trainer import SFTDataset
from .sft_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
    compute_entropy
)


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
    # Path is relative to the project root
    prompt_path = Path(__file__).parent.parent / 'cs336_alignment/prompts/r1_zero.prompt'
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def collate_fn(batch, tokenizer):
    """Custom collate function for DataLoader."""
    prompts = [item['prompt'] for item in batch]
    outputs = [item['output'] for item in batch]
    
    # Tokenize and create masks
    tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    
    # Create response mask
    tokenized['response_mask'] = (tokenized['labels'] != -100)
    
    return tokenized


def train_sft(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    max_seq_length: int = 1024,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    device: str = 'cuda'
):
    """Main SFT training function."""
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Starting SFT training")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    
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
    logger.info(f"Loaded prompt template: {prompt_template[:100]}...")
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = SFTDataset(dataset_name, tokenizer, prompt_template)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0  # Use 0 for debugging
    )
    
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
    
    # Prepare some validation prompts for logging
    val_prompts = []
    val_ground_truths = []
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        val_prompts.append(item['prompt'])
        val_ground_truths.append(item['solution'])
    
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
            outputs = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = outputs['response_log_probs']
            
            # Calculate loss using our microbatch function
            loss, _ = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                normalize_constant=1.0
            )
            
            # Backward pass
            loss.backward()
            
            # Take optimizer step every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step}: loss={loss.item() * gradient_accumulation_steps:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                
                # Save checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
                
                # Log generations
                if global_step > 0 and global_step % (save_steps // 2) == 0:
                    logger.info("Generating sample outputs...")
                    # This function needs to be implemented or adapted
                    # log_generations(...)
        
        logger.info(f"Epoch {epoch + 1} completed.")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    log_generations(
        model, tokenizer, val_prompts, val_ground_truths, 
        device, num_examples=5
    )
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='SFT Training Script')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-Math-1.5B',
                        help='Path to the base model')
    parser.add_argument('--dataset_name', type=str, default='pan-nyu/countdown',
                        help='Name of the dataset on HuggingFace Hub')
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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    train_sft(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        device=args.device
    )


if __name__ == '__main__':
    main()