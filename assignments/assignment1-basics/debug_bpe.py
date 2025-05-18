#!/usr/bin/env python
"""
Debug script to trace through the BPE implementation and find issues.
"""
import logging
import os
from pathlib import Path
from cs336_basics.train_bpe import train_bpe, save_tokenizer, bytes_to_unicode

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Small test corpus
TEST_CORPUS = "tests/fixtures/corpus.en"
OUTPUT_DIR = "debug_bpe_output"

def debug_train_bpe():
    """Run the BPE training with debug output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info(f"Starting debug BPE training on {TEST_CORPUS}")
    
    # Track merge counts to detect duplicates
    merge_counts = {}
    
    # Monkey patch the merged_byte_pair function to track duplicates
    original_merge_byte_pair = train_bpe.__globals__['merge_byte_pair']
    
    def merge_byte_pair_with_debug(token_counts, pair):
        if pair in merge_counts:
            merge_counts[pair] += 1
            logging.warning(f"Duplicate merge detected: {pair}, count: {merge_counts[pair]}")
        else:
            merge_counts[pair] = 1
            logging.debug(f"New merge: {pair}")
        return original_merge_byte_pair(token_counts, pair)
    
    train_bpe.__globals__['merge_byte_pair'] = merge_byte_pair_with_debug
    
    # Run the training
    vocab, merges = train_bpe(
        input_path=TEST_CORPUS,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        num_processes=1  # Use single process for debugging
    )
    
    # Save results for inspection
    vocab_path = os.path.join(OUTPUT_DIR, "debug_vocab.json")
    merges_path = os.path.join(OUTPUT_DIR, "debug_merges.txt")
    save_tokenizer(vocab, merges, vocab_path, merges_path)
    
    # Print statistics
    logging.info(f"Vocabulary size: {len(vocab)}")
    logging.info(f"Unique merges: {len(set(merges))}")
    logging.info(f"Total merges: {len(merges)}")
    logging.info(f"Duplicate merges: {len(merges) - len(set(merges))}")
    
    # Check for repeated merges
    from collections import Counter
    merge_counter = Counter(merges)
    duplicates = [(merge, count) for merge, count in merge_counter.items() if count > 1]
    if duplicates:
        logging.error(f"Found {len(duplicates)} duplicate merges:")
        for merge, count in duplicates:
            logging.error(f"  {merge}: {count} times")
    
    # Check if merges are ordered correctly
    is_ordered = all(m1 != m2 for m1, m2 in zip(merges[:-1], merges[1:]))
    logging.info(f"Merges are all unique in sequence: {is_ordered}")
    
    # Restore original function
    train_bpe.__globals__['merge_byte_pair'] = original_merge_byte_pair
    
    return vocab, merges

if __name__ == "__main__":
    debug_train_bpe()