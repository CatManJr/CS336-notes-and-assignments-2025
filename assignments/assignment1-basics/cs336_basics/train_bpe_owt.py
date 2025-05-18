#!/usr/bin/env python
"""
Script to train a BPE tokenizer on the OpenWebText dataset.
"""
import os
import time
import argparse
from pathlib import Path
import psutil
import logging
import multiprocessing as mp

from cs336_basics.token_utils import find_chunk_boundaries, process_chunk,get_pair_stats, \
                                merge_byte_pairs
import cProfile

from train_bpe import TrainBPE, save_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_longest_token(vocab):
    """Find and return the longest token in the vocabulary"""
    longest_token = b''
    longest_token_id = None
    
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > len(longest_token):
            longest_token = token_bytes
            longest_token_id = token_id
    
    return longest_token, longest_token_id

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_input_path = project_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    default_output_dir = project_root / "data"
    
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on OpenWebText')
    parser.add_argument('--input', default=str(default_input_path), 
                        help='Path to OpenWebText training data')
    parser.add_argument('--vocab-size', type=int, default=32000, 
                        help='Maximum vocabulary size')
    parser.add_argument('--output-dir', default=str(default_output_dir), 
                        help='Directory to save the tokenizer files')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Number of processes to use for parallelization')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / 'owt_vocab.json'
    merges_path = output_dir / 'owt_merges.txt'
    
    # Track memory and time
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    # Train the tokenizer
    logging.info(f"Starting BPE training on OpenWebText with vocab size {args.vocab_size}...")
    train_bpe = TrainBPE(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"]
    )
    train_bpe.train(measurement=False)
    vocab, merges = train_bpe.vocab, train_bpe.merges
    
    # Calculate elapsed time and memory usage
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    elapsed_time = end_time - start_time
    
    # Save the tokenizer
    logging.info(f"Saving tokenizer to {vocab_path} and {merges_path}...")
    save_tokenizer(vocab, merges, vocab_path, merges_path)
    
    # Find longest token
    longest_token, longest_token_id = find_longest_token(vocab)
    
    # Display results
    hrs = elapsed_time // 3600
    mins = (elapsed_time % 3600) // 60
    secs = elapsed_time % 60
    
    logging.info(f"Training complete in {hrs:.0f}h {mins:.0f}m {secs:.1f}s")
    logging.info(f"Memory usage: {end_memory - start_memory:.2f} MB")
    logging.info(f"Vocabulary size: {len(vocab)}")
    logging.info(f"Number of merges: {len(merges)}")
    
    # Print info about the longest token
    try:
        longest_token_str = longest_token.decode('utf-8', errors='replace')
        logging.info(f"Longest token (ID {longest_token_id}): '{longest_token_str}' ({len(longest_token)} bytes)")
    except Exception as e:
        logging.info(f"Longest token (ID {longest_token_id}): {longest_token} ({len(longest_token)} bytes)")
    
if __name__ == "__main__":
    main()