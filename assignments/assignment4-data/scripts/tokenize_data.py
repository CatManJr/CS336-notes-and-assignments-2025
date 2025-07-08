#!/usr/bin/env python3
"""
Tokenize filtered text data using GPT-2 tokenizer.
Falls back to simple tokenization if GPT-2 tokenizer is not available offline.
"""

import argparse
import multiprocessing
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# Try to import transformers, but don't fail if not available
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using simple tokenization")


class SimpleTokenizer:
    """Simple fallback tokenizer when GPT-2 tokenizer is not available"""
    
    def __init__(self):
        # Simple vocabulary based on common English words and characters
        # This is a very basic tokenizer for demonstration purposes
        self.vocab_size = 50257  # Match GPT-2 vocab size for compatibility
        self.eos_token_id = 50256  # EOS token ID
        
        # Create a simple character + word based vocabulary
        self.char_to_id = {}
        self.id_to_char = {}
        
        # ASCII characters (0-255)
        for i in range(256):
            char = chr(i)
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        
        # Add some common words to reduce sequence length
        common_words = [
            ' the', ' and', ' or', ' in', ' on', ' at', ' to', ' for', ' of', ' with',
            ' a', ' an', ' is', ' are', ' was', ' were', ' be', ' been', ' have', ' has',
            ' will', ' would', ' could', ' should', ' can', ' may', ' must', ' do', ' does',
            ' said', ' say', ' get', ' go', ' come', ' see', ' know', ' think', ' take',
            ' make', ' give', ' use', ' find', ' tell', ' ask', ' work', ' seem', ' feel',
            ' try', ' leave', ' call', ' put', ' set', ' become', ' turn', ' move', ' like',
            ' look', ' want', ' show', ' play', ' run', ' believe', ' begin', ' help',
            ' talk', ' turn', ' start', ' might', ' follow', ' around', ' between',
            ' under', ' never', ' after', ' back', ' other', ' many', ' than', ' only',
            ' those', ' come', ' its', ' over', ' think', ' also', ' your', ' work',
            ' life', ' only', ' can', ' still', ' should', ' after', ' being', ' now',
            ' made', ' before', ' here', ' through', ' when', ' where', ' how', ' all',
            ' any', ' may', ' say', ' each', ' which', ' she', ' do', ' has', ' her',
            ' will', ' one', ' our', ' out', ' day', ' had', ' his', ' her', ' you',
            ' but', ' not', ' what', ' all', ' were', ' they', ' we', ' been', ' have',
            ' their', ' said', ' each', ' which', ' them', ' these', ' way', ' could',
            ' first', ' also', ' after', ' back', ' other', ' many', ' than', ' then',
            ' them', ' these', ' two', ' more', ' very', ' what', ' know', ' just',
            ' first', ' get', ' has', ' him', ' his', ' how', ' man', ' new', ' now',
            ' old', ' see', ' two', ' who', ' boy', ' did', ' its', ' let', ' put',
            ' say', ' she', ' too', ' use'
        ]
        
        word_id = 256
        for word in common_words:
            if word_id < self.vocab_size - 1:  # Leave room for EOS token
                self.char_to_id[word] = word_id
                self.id_to_char[word_id] = word
                word_id += 1
    
    def encode(self, text):
        """Simple encoding: try to match words, fall back to characters"""
        if not text:
            return []
        
        tokens = []
        i = 0
        
        while i < len(text):
            # Try to match longest possible word first
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.char_to_id:
                    tokens.append(self.char_to_id[substr])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Fall back to character level
                char = text[i]
                if char in self.char_to_id:
                    tokens.append(self.char_to_id[char])
                else:
                    # Unknown character, map to space
                    tokens.append(self.char_to_id.get(' ', 32))
                i += 1
        
        return tokens
    
    def decode(self, token_ids):
        """Simple decoding"""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                chars.append(self.id_to_char[token_id])
            else:
                chars.append('?')  # Unknown token
        return ''.join(chars)


def get_tokenizer(tokenizer_name):
    """Get tokenizer, falling back to simple tokenizer if needed"""
    if TRANSFORMERS_AVAILABLE:
        try:
            # Try to load from local cache first
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            print(f"Loaded {tokenizer_name} tokenizer from local cache")
            return tokenizer, False
        except Exception as e:
            print(f"Could not load {tokenizer_name} from local cache: {e}")
            
            try:
                # Try to download (if internet available)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"Downloaded {tokenizer_name} tokenizer")
                return tokenizer, False
            except Exception as e:
                print(f"Could not download {tokenizer_name}: {e}")
    
    # Fall back to simple tokenizer
    print("Using simple fallback tokenizer")
    tokenizer = SimpleTokenizer()
    return tokenizer, True


def tokenize_line_and_add_eos(args):
    """Tokenize a single line and add end-of-sequence token"""
    line, tokenizer_name, use_simple = args
    
    # Skip empty lines
    if not line.strip():
        return []
    
    if use_simple:
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.encode(line.strip())
        return tokens + [tokenizer.eos_token_id]
    else:
        # Use transformers tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return tokenizer.encode(line.strip()) + [tokenizer.eos_token_id]


def main():
    parser = argparse.ArgumentParser(description="Tokenize filtered text data")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to filtered text file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output tokenized data file")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use (default: gpt2)")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--chunksize", type=int, default=100,
                       help="Chunk size for multiprocessing")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        print(f"Input file {input_path} does not exist")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Tokenizing {input_path} -> {output_path}")
    print(f"Using tokenizer: {args.tokenizer}")
    
    # Get tokenizer
    tokenizer, use_simple = get_tokenizer(args.tokenizer)
    
    if use_simple:
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")
    else:
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines")
    
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    print(f"Processing {len(non_empty_lines)} non-empty lines")
    
    # Set up multiprocessing
    max_workers = args.max_workers or multiprocessing.cpu_count()
    print(f"Using {max_workers} workers")
    
    # Prepare arguments for multiprocessing
    worker_args = [(line, args.tokenizer, use_simple) for line in non_empty_lines]
    
    # Tokenize in parallel
    if max_workers == 1 or use_simple:
        # Single process mode for simple tokenizer to avoid pickling issues
        print("Using single-process mode")
        results = []
        for worker_arg in tqdm(worker_args, desc="Tokenizing lines"):
            result = tokenize_line_and_add_eos(worker_arg)
            if result:
                results.append(result)
    else:
        # Multi-process mode for transformers tokenizer
        pool = multiprocessing.Pool(max_workers)
        results = []
        
        try:
            for result in tqdm(
                pool.imap(tokenize_line_and_add_eos, worker_args, chunksize=args.chunksize),
                total=len(worker_args),
                desc="Tokenizing lines"
            ):
                if result:  # Skip empty results
                    results.append(result)
        finally:
            pool.close()
            pool.join()
    
    # Flatten the list of token IDs and convert to numpy array
    all_ids = []
    for token_list in results:
        all_ids.extend(token_list)
    
    print(f"Total tokens: {len(all_ids):,}")
    
    if not all_ids:
        print("No tokens generated!")
        return
    
    # Convert to numpy array with appropriate dtype
    # Use uint16 if all token IDs fit, otherwise uint32
    max_token_id = max(all_ids)
    
    if max_token_id < 65536:
        dtype = np.uint16
        print("Using uint16 dtype (token IDs < 65536)")
    else:
        dtype = np.uint32
        print("Using uint32 dtype (token IDs >= 65536)")
    
    ids_array = np.array(all_ids, dtype=dtype)
    
    # Save to file
    ids_array.tofile(output_path)
    
    print(f"Tokenized data saved to: {output_path}")
    print(f"Array shape: {ids_array.shape}")
    print(f"Array dtype: {ids_array.dtype}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Verify the file can be loaded
    print("\nVerifying saved file...")
    loaded_array = np.fromfile(output_path, dtype=dtype)
    print(f"Loaded array shape: {loaded_array.shape}")
    
    # Show a sample of tokenized text
    print("\nSample of tokenized text:")
    sample_tokens = loaded_array[:200].tolist()
    if use_simple:
        sample_text = tokenizer.decode(sample_tokens)
    else:
        sample_text = tokenizer.decode(sample_tokens)
    
    print(f"First 200 tokens: {sample_text[:500]}...")


if __name__ == "__main__":
    main()