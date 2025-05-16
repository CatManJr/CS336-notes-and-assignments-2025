import os
import re
import time
import json
from typing import Dict, List, Tuple, Set, Counter as CounterType
from collections import Counter, defaultdict
import multiprocessing
from multiprocessing import Pool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    This follows the GPT-2 byte-level encoding scheme.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs.copy()
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def unicode_to_bytes():
    """Inverse of bytes_to_unicode"""
    return {v: k for k, v in bytes_to_unicode().items()}

def process_chunk(args):
    """Process a chunk of text for BPE pre-tokenization"""
    chunk, regex_pattern, special_tokens = args
    
    # Strip out all special tokens before pre-tokenization
    # We need to be careful with special tokens since they might contain regex special characters
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    special_token_pattern = '|'.join(escaped_special_tokens)
    
    # Split on special tokens first - they will not be merged
    if special_token_pattern:
        parts = re.split(f'({special_token_pattern})', chunk)
        # Filter out empty strings that might result from the split
        parts = [p for p in parts if p]
    else:
        parts = [chunk]
    
    # Now process each part separately
    token_counts = Counter()
    
    for part in parts:
        # Skip special tokens in the pre-tokenization (keep them as is)
        if part in special_tokens:
            token_counts[part.encode('utf-8')] += 1
            continue
            
        # For normal text, apply the regex pattern to break into pre-tokens
        matches = regex_pattern.finditer(part)
        for match in matches:
            token = match.group()
            # Convert to bytes
            token_bytes = token.encode('utf-8')
            token_counts[token_bytes] += 1
            
    return token_counts

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_byte_pair_counts(token_counts: CounterType) -> CounterType:
    """
    Count frequency of all byte pairs in the given tokens.
    
    Args:
        token_counts: A counter mapping tokens (bytes) to their frequencies
        
    Returns:
        A counter mapping byte pairs (tuples of bytes) to their frequencies
    """
    pairs = Counter()
    
    for token, freq in token_counts.items():
        # If token has length 1 or less, there are no pairs
        if len(token) <= 1:
            continue
            
        # Count pairs in this token
        for i in range(len(token) - 1):
            pair = (token[i:i+1], token[i+1:i+2])
            pairs[pair] += freq
            
    return pairs

def merge_byte_pair(token_counts: CounterType, pair: Tuple[bytes, bytes]) -> CounterType:
    """
    Merge all occurrences of the given byte pair in the token counts.
    
    Args:
        token_counts: A counter mapping tokens (bytes) to their frequencies
        pair: The pair of bytes to merge
        
    Returns:
        A new counter with the pair merged
    """
    first, second = pair
    new_token_counts = Counter()
    
    # Pattern to match the pair, with optional boundary markers
    pattern = re.compile(b'(?<![\\S])' + re.escape(first) + re.escape(second) + b'(?![\\S])')
    
    # Replacement is the merged pair
    replacement = first + second
    
    for token, count in token_counts.items():
        # Check if the pair exists in this token
        if first + second in token:
            # Create a new merged token
            new_token = b''
            i = 0
            while i < len(token):
                # If we're at a position where the pair exists
                if (i < len(token) - len(first + second) + 1 and 
                    token[i:i+len(first)] == first and 
                    token[i+len(first):i+len(first+second)] == second):
                    # Add the merged pair
                    new_token += replacement
                    # Skip over the pair
                    i += len(first + second)
                else:
                    # Add the current byte
                    new_token += token[i:i+1]
                    i += 1
            
            # Add the new token to the counts
            new_token_counts[new_token] += count
        else:
            # If the pair doesn't exist in this token, keep it as is
            new_token_counts[token] += count
    
    return new_token_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    chunking_special_token_str: str = "<|endoftext|>",
    num_processes: int = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given input file.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Maximum vocabulary size (including initial bytes and special tokens)
        special_tokens: List of special tokens to add to the vocabulary
        chunking_special_token_str: Special token to use for chunking the input file
        num_processes: Number of processes to use for parallelization (defaults to CPU count)
        
    Returns:
        A tuple containing:
        - vocab: Dictionary mapping token indices to token bytes
        - merges: List of BPE merges as tuples of (first, second) bytes
    """
    start_time = time.time()
    
    # Use all available CPUs if not specified
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    logging.info(f"Starting BPE training with {num_processes} processes")
    
    # GPT-2 pre-tokenization pattern simplified for ASCII (corpus.en uses ASCII)
    # split on common contractions, words, numbers, punctuation, and whitespace
    pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
    
    # Initialize the vocabulary with single bytes
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    # Initial vocabulary is all possible bytes
    initial_vocab = {i: bytes([i]) for i in range(256)}
    
    # Convert special tokens to bytes
    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    
    # Open the input file and get chunking boundaries
    with open(input_path, 'rb') as f:
        chunking_special_token = chunking_special_token_str.encode('utf-8')
        boundaries = find_chunk_boundaries(f, num_processes, chunking_special_token)
        
        # Read chunks and process them in parallel
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            chunks.append(chunk)
    
    logging.info(f"File read and split into {len(chunks)} chunks. Starting pre-tokenization...")
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(
            process_chunk, 
            [(chunk, pattern, special_tokens) for chunk in chunks]
        )
    
    # Combine results from all chunks
    token_counts = Counter()
    for chunk_result in chunk_results:
        token_counts.update(chunk_result)
    
    logging.info(f"Pre-tokenization complete. Found {len(token_counts)} unique tokens.")
    logging.info(f"Starting merges to reach vocabulary size {vocab_size}...")
    
    # Add special tokens to vocabulary
    merges = []
    
    # Calculate how many merges we need to perform
    # We have 256 initial byte tokens + special tokens, so we need to perform (vocab_size - 256 - len(special_tokens)) merges
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # Perform merges
    for i in range(num_merges):
        if i % 1000 == 0:
            logging.info(f"Completed {i}/{num_merges} merges...")
        
        # Get counts of all byte pairs
        pair_counts = get_byte_pair_counts(token_counts)
        
        # If no more pairs to merge, we're done
        if not pair_counts:
            break
        
        # Find the most frequent pair
        best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
        
        # Record this merge
        merges.append(best_pair)
        
        # Apply the merge to all tokens
        token_counts = merge_byte_pair(token_counts, best_pair)
    
    # Build the final vocabulary
    vocab = {}
    token_id = 0
    
    # Add single bytes first (always part of the vocabulary)
    for byte_val in range(256):
        vocab[token_id] = bytes([byte_val])
        token_id += 1
    
    # Add merged tokens in order of creation
    unique_merged_tokens = set(token_counts.keys()) - set(vocab.values())
    for token in unique_merged_tokens:
        if token_id < vocab_size - len(special_tokens):
            vocab[token_id] = token
            token_id += 1
        else:
            break
    
    # Add special tokens at the end of the vocabulary
    for token in special_token_bytes:
        vocab[token_id] = token
        token_id += 1
    
    end_time = time.time()
    logging.info(f"BPE training complete in {end_time - start_time:.2f} seconds.")
    logging.info(f"Final vocabulary size: {len(vocab)}")
    
    return vocab, merges

# Function to serialize the tokenizer to disk
def save_tokenizer(vocab, merges, vocab_path, merges_path):
    """
    Save the tokenizer vocabulary and merges to disk.
    
    Args:
        vocab: Dictionary mapping token indices to token bytes
        merges: List of BPE merges
        vocab_path: Path to save the vocabulary JSON
        merges_path: Path to save the merges text file
    """
    # Convert bytes to strings for JSON serialization
    byte_encoder = bytes_to_unicode()
    str_vocab = {}
    
    for token_id, token_bytes in vocab.items():
        # Convert each byte to its Unicode representation
        token_str = ''.join(byte_encoder.get(b, '') for b in token_bytes)
        str_vocab[token_str] = token_id
    
    # Save vocabulary to JSON
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(str_vocab, f, ensure_ascii=False, indent=2)
    
    # Save merges to text file
    with open(merges_path, 'w', encoding='utf-8') as f:
        for first, second in merges:
            # Convert bytes to strings
            first_str = ''.join(byte_encoder.get(b, '') for b in first)
            second_str = ''.join(byte_encoder.get(b, '') for b in second)
            f.write(f"{first_str} {second_str}\n")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--input', required=True, help='Path to input text file')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Maximum vocabulary size')
    parser.add_argument('--special-tokens', nargs='+', default=["<|endoftext|>"], help='Special tokens to add to vocabulary')
    parser.add_argument('--out-vocab', default='vocab.json', help='Output path for vocabulary')
    parser.add_argument('--out-merges', default='merges.txt', help='Output path for merges')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use')
    
    args = parser.parse_args()
    
    # Train the tokenizer
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.processes
    )
    
    # Save the tokenizer
    save_tokenizer(vocab, merges, args.out_vocab, args.out_merges)