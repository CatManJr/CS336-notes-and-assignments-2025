#!/usr/bin/env python3
"""
CLI entrypoint for training on the TinyStories dataset.
This script provides a complete pipeline that:
1. Checks if tokenized data files exist
2. If not, trains a BPE tokenizer and tokenizes the data
3. Runs the training process on the tokenized data
"""
import argparse
import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Optional
import time
from cs336_basics.train_lm import train_tiny_stories
from cs336_basics.tokenizer import BPETokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize_file(tokenizer, input_file, output_file):
    """Tokenize a text file and save the tokens as a numpy file."""
    try:
        # Read the input file
        logger.info(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the text
        logger.info("Tokenizing text...")
        tokens = tokenizer.encode(text)
        
        # Save tokens to numpy file
        logger.info(f"Saving {len(tokens)} tokens to {output_file}")
        np.save(output_file, np.array(tokens, dtype=np.int32))
        logger.info(f"Tokenization complete. Output saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        return False

def ensure_tokenized_data(data_dir):
    """
    Ensure tokenized data exists. If not, create it.
    
    Args:
        data_dir: Directory containing raw data and where tokenized data will be saved
    """
    data_dir = Path(data_dir)
    
    # Define file paths
    train_raw_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    val_raw_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    train_tokens_path = data_dir / "tinystories_train_tokens.npy"
    val_tokens_path = data_dir / "tinystories_val_tokens.npy"
    vocab_path = data_dir / "tinystories_vocab.json"
    merges_path = data_dir / "tinystories_merges.txt"
    
    # Check if tokenized files already exist
    if train_tokens_path.exists() and val_tokens_path.exists():
        logger.info("Tokenized data files found. Skipping tokenization.")
        return True
    
    # Check if raw data files exist
    if not train_raw_path.exists() or not val_raw_path.exists():
        logger.error(f"Raw data files not found. Expected at {train_raw_path} and {val_raw_path}.")
        return False
    
    # Check if BPE tokenizer files exist
    if not vocab_path.exists() or not merges_path.exists():
        logger.error(f"BPE tokenizer files not found. Expected at {vocab_path} and {merges_path}. "
                    f"Please run train_bpe_tinystories.py first.")
        logger.info("Example command: uv run cs336_basics/train_bpe_tinystories.py --input ./data/TinyStoriesV2-GPT4-train.txt --output-dir ./data")
        return False
    
    # Load tokenizer using the proper class method
    logger.info(f"Loading tokenizer from {vocab_path} and {merges_path}")
    try:
        tokenizer = BPETokenizer.from_file(str(vocab_path), str(merges_path))
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False
    
    # Tokenize data
    logger.info("Starting tokenization process...")
    start_time = time.time()
    
    # Tokenize training data
    if not tokenize_file(tokenizer, train_raw_path, train_tokens_path):
        return False
        
    # Tokenize validation data
    if not tokenize_file(tokenizer, val_raw_path, val_tokens_path):
        return False
    
    elapsed = time.time() - start_time
    logger.info(f"Tokenization completed in {elapsed:.2f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories data")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing TinyStories tokenized data (.npy files)")
    parser.add_argument("--output_dir", type=str, default="./tiny_stories_model", help="Directory to save checkpoints and logs")
    parser.add_argument("--context_length", type=int, default=512, help="Model context length")
    parser.add_argument("--d_model", type=int, default=384, help="Transformer model dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--d_ff", type=int, default=1536, help="Feedforward network hidden dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_iters", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu, mps, cuda)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-tinystories", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip tokenization even if tokenized files don't exist")

    args = parser.parse_args()
    
    # Ensure paths exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure tokenized data exists
    if not args.skip_tokenization:
        success = ensure_tokenized_data(args.data_dir)
        if not success:
            logger.error("Failed to ensure tokenized data exists. Exiting.")
            return

    # Call the training function
    train_tiny_stories(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

if __name__ == "__main__":
    main()
