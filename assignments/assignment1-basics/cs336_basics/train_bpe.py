import os
import regex as re
from collections import Counter
from typing import List, Tuple, Iterable
from io import BytesIO
import io
import pstats
import logging
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

import multiprocessing as mp

from cs336_basics.token_utils import find_chunk_boundaries, process_chunk,get_pair_stats, \
                               merge_byte_pairs
import cProfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainBPE:
    """
    A class representing a Byte Pair Encoding (BPE) tokenizer.

    This class is designed to handle the training and application of BPE tokenization
    on a given text corpus. It allows for the creation of a vocabulary based on the
    frequency of byte pairs in the text, and provides methods for encoding and decoding
    text using the learned BPE merges.
    """
    
    def __init__(self,
                 input_path: str | os.PathLike,
                 vocab_size: int,
                 special_tokens: List[str],
                 verbose: bool = False):
        """
        Initializes the BPE tokenizer with a specified vocabulary size and special tokens.

        Parameters
        ----------
        input_path : str | os.PathLike
            Path to the input text file to be used for training the BPE tokenizer.
        vocab_size : int
            The desired size of the vocabulary to be created during training.
        special_tokens : List[str]
            A list of special tokens to be included in the vocabulary.
            These tokens will not be merged with other tokens during training.
            They are typically used for end-of-text markers or other special purposes.  
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges: List[tuple[bytes, bytes]] = []
        self.verbose = verbose
        
        # Read the input file in binary mode
        if self.input_path != "":
            with open(input_path, "rb") as f:
                self.file_object = f.read() 

        # Make sure text file is not empty
        assert self.file_object != b"", (
            "Text file is empty. Please provide a valid text file."
        )
            
        # --- Step 0: Initial Setup ---
        # Since we are training a BPE tokenizer, our initial vocabulary is 256
        # The special tokens are also added to the vocabulary
        # After each merge, the vocabulary size increases by 1 until it reaches the desired size
        # The vocabulary size is the number of special tokens + 256 base tokens + the number of merges
        self.num_special_tokens = len(special_tokens)
        # Calculate the number of merges needed
        self.num_merges = vocab_size - self.num_special_tokens - 256
        
        # Initialize the original vocabulary of size 256
        self.vocab: dict[int, bytes] = { 
            i: bytes([i]) for i in range(256)
        }
        
        # Initialize an empty list to store the special tokens converted to bytes.
        # We work with bytes because the training data is read as bytes.
        self.encoded_special_tokens: List[bytes] = []
        # Initialize an empty byte string that will hold a regex pattern for splitting based on special tokens.
        self.split_pattern_bytes: bytes = b""
        
        if special_tokens:
            # Encode special tokens to bytes
            self.encoded_special_tokens = [
                token.encode("utf-8") for token in special_tokens
            ]
            # # Create a regex pattern to split the text based on special tokens
            # # re.escape is used to treat special characters in the token as literal characters, even if they are regex metacharacters
            # # The splitting logic will be based on the byte representation of the special tokens
            # encoded_special_tokens = b"|".join(re.escape(token) for token in encoded_special_tokens)
            
        # GPT2-style regex pattern for splitting the text into potential initial tokens
        self.split_pattern_bytes = br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        self.split_pattern_bytes = re.compile(self.split_pattern_bytes)
        
    def train(self,
              verbose: bool = False,
              measurement: bool = False) -> Tuple[dict[int, bytes], List[tuple[bytes, bytes]]]:
        
        # --- Step 1: Find the chunk boundaries ---
        if verbose:
            print("Starting BPE training...")    
        # Ideal number of parallel chunks to read
        # It may appear slower for small files because of the overhead of multiprocessing
        # but for large files, it should be faster
        # because it can read multiple chunks in parallel
        # and the overhead of multiprocessing is negligible
        # compared to the time it takes to read the file
        # and process the chunks
        num_chunks = mp.cpu_count() - 1
        
        print("--- Step 1: Find the chunk boundaries ---")
        
        if measurement:
            profiler_step1 = cProfile.Profile()
            profiler_step1.enable()
        
            # Find chunk boundaries
            chunk_boundaries = find_chunk_boundaries(
                byte_text_file=self.file_object,  # type: ignore
                num_desired_chunks=num_chunks,
                special_split_tokens=self.encoded_special_tokens, 
            )
            profiler_step1.disable()
            print("--- Step 1: Completed ---")
            profiler_step1.print_stats(sort="cumtime")
            
            # Print the profile stats
            # if verbose:
            #     # Analyze and print results for Step 1
            #     s1 = io.StringIO() # Use StringIO to capture print output
            #     sortby1 = 'cumtime' # Sort by cumulative time to see total cost of the function call
            #     # Create a Stats object and print the stats
            #     ps1 = pstats.Stats(profiler_step1, stream=s1).sort_stats(sortby1)
            #     ps1.print_stats()
            #     print(s1.getvalue()) # Print the captured output
            #     print("-" * 30) # Separator
        else:
            # Find chunk boundaries
            chunk_boundaries = find_chunk_boundaries(
                byte_text_file=self.file_object,  # type: ignore
                num_desired_chunks=num_chunks,
                special_split_tokens=self.encoded_special_tokens, 
            )
        

        # print(f"Chunk boundaries: {chunk_boundaries}")
        # print(text_file[chunk_boundaries[0]:chunk_boundaries[1]])
        
        # --- Step 2: Pretokenize the text file in parallel ---
        
        print("--- Step 2: Pretokenize the text file in parallel ---")
        
        profiler_step2 = cProfile.Profile()
        
        if measurement:
            profiler_step2.enable()

        
        # This step prepares the raw input text for BPE merging.
        # It typically involves splitting the text into initial "words" or segments.
        # The splitting often respects whitespace and punctuation, and importantly, special tokens.
        # We will count the frequency of these initial segments (pretokens).
        
        # Dictionary to store the frequency of each pretoken
        # Key: The pretoken (bytes)
        # Value: 
        #   - A list of byte objects representing the pretoken. E.g., "hello" as [b'h', b'e', b'l', b'l', b'o'].
        #   - The frequency of the pretoken in the text file.
        # We use bytes as the key because the input text is in bytes
        pretoken_freq: dict[bytes, Tuple[List[bytes], int]] = {}

        
        # Number of chunks to process in parallel
        # it can be less than the number of chunks
        num_processes = len(chunk_boundaries) - 1
        
        # Create a pool of processes
        pool = mp.Pool(processes=num_processes)
        
        # --- Prepare Special Tokens ---
        # Special tokens (like <PAD>, <EOS>) need to be handled explicitly.
        # During pretokenization, we want to ensure these exact sequences are
        # identified and not broken down by the general pretokenization pattern (GPT2_PAT).
        # They are also added to the final vocabulary with dedicated IDs.
            
        # Create tasks for the multiprocessing pool
        # This utilizes the boundaries of the chunks to read the text file in parallel
        # Each task is a tuple containing the arguments for the process_chunk function
        tasks = [
            (self.file_object[chunk_boundaries[i]:chunk_boundaries[i + 1]], self.split_pattern_bytes, self.encoded_special_tokens)
            for i in range(num_processes)
        ]
        
        # Aggregate the results from all processes
        # Each process returns a Counter object with the frequency of each pretoken
        # The Counter objects are combined into a single dictionary
        # The frequency of each pretoken is summed across all processes
        aggregated_freq = Counter()
        
        # Use a multiprocessing pool to process the chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Map the process_chunk function to the tasks
            results = pool.starmap(process_chunk, tasks)
            
            # Combine the results from all processes
            for result in results:
                aggregated_freq.update(result)

        # Populate the pretoken_freq dictionary with the results
        for pretoken, freq in aggregated_freq.items():
            # Convert the byte objects into a list
            byte_list = [bytes([i]) for i in pretoken]
            # Store the frequency of the pretoken
            pretoken_freq[pretoken] = (byte_list, freq)
    
        if measurement:
            profiler_step2.disable()
            print("--- Step 2: Completed ---")
            print(profiler_step2.print_stats(sort="cumtime"))
            
            # if verbose:
                # # Analyze and print results for Step 2
                # s2 = io.StringIO()
                # sortby2 = 'cumtime' # Sort by cumulative time to see total cost of the function call 
                # # Create a Stats object and print the stats
                # ps2 = pstats.Stats(profiler_step2, stream=s2).sort_stats(sortby2)
                # ps2.print_stats()
                # print(s2.getvalue())
                # print("-" * 30) # Separator
        
        # --- Step 3: Merge the most frequent pairs of pretokens ---
        
        print("--- Step 3: Merge the most frequent pairs of pretokens ---")
        profiler_step3 = cProfile.Profile()
        
        if measurement:
            profiler_step3.enable()
        
        
        # The merging process is repeated until the desired vocabulary size is reached.
        # The merging process involves finding the most frequent pair of pretokens
        # and merging them into a new pretoken.
        
        # Calculate the initial frequencies of adjacent pairs of pretokens
        byte_pairs_freq: dict[tuple[bytes, bytes], int] = get_pair_stats(
            pretoken_freq=pretoken_freq
        )    
        
        # Loop until the desired vocabulary size is reached
        # The number of merges is equal to the vocabulary size minus the numbers of special tokens and initial tokens
        
        for iter in range(self.num_merges):
            # Find the best pair to merge: the most frequent pair.
            # max() with a key function finds the item with the maximum value returned by the key function.
            # The key function `lambda pair: (pair_freq[pair], pair)` sorts first by frequency (descending)
            # and then by the pair itself (lexicographically ascending) to break ties consistently.
            best_pair = max(byte_pairs_freq, key=lambda pair: (byte_pairs_freq[pair], pair))
            
            # print(best_pair)
            
            # Making sure the best pair is in an appropriate format
            assert (isinstance(best_pair, tuple) and 
                len(best_pair) == 2 and
                isinstance(best_pair[0], bytes) and
                    isinstance(best_pair[1], bytes)), (
                "Best pair should be a tuple of two bytes. Not {0} and {1}".format(
                    type(best_pair[0]), type(best_pair[1])
                )
            )
            
            # Add the best pair to the merges list
            self.merges.append(best_pair)
            
            # Add the new merged pretoken to the vocabulary
            # The new pretoken is the concatenation of the two bytes in the best pair
            new_pretoken = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = new_pretoken
            
            # Replace all the occurrences of the best pair in the pretoken_freq dictionary
            # with the new pretoken
            # also update the 'byte_pairs_stat' Counter based on changes.
            merge_byte_pairs(
                pretoken_freq=pretoken_freq,
                byte_pairs_freq=byte_pairs_freq,
                best_pair=best_pair
            )
            
            # --- Step 4: Add special tokens to the vocabulary ---
            # Add special tokens to the vocabulary
            # The special tokens are added to the vocabulary with dedicated IDs starting from 256 + the number of merges
        
        if measurement:
            profiler_step3.disable()
            print("--- Step 3: Completed ---")
            print(profiler_step3.print_stats(sort="cumtime"))
            # if verbose:
            #     # Analyze and print results for Step 3
            #     s3 = io.StringIO()
            #     sortby3 = 'cumtime'
            #     # Create a Stats object and print the stats
            #     ps3 = pstats.Stats(profiler_step3, stream=s3).sort_stats(sortby3)
            #     ps3.print_stats()
            #     print(s3.getvalue())
            #     print("-" * 30)
                
            
        for i in range(self.num_special_tokens):
            # Add the special token to the vocabulary
            self.vocab[len(self.vocab)] = self.encoded_special_tokens[i]
                
        return self.vocab, self.merges

def bytes_to_unicode():
    """
    Returns a dictionary mapping byte values (0-255) to unicode strings.
    This is used to ensure a reversible mapping between bytes and unicode for BPE.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_unicode = dict(zip(bs, (chr(c) for c in cs)))
    return byte_to_unicode

def unicode_to_bytes():
    """
    Returns a dictionary mapping unicode strings to byte values (0-255).
    This is used to ensure a reversible mapping between unicode and bytes for BPE.
    """
    byte_to_unicode = bytes_to_unicode()
    # Invert the mapping: unicode char -> byte value
    return {v: k for k, v in byte_to_unicode.items()}

def save_tokenizer(vocab, merges, vocab_path, merges_path):
    """
    Save the tokenizer vocabulary and merges to disk.

    Args:
        vocab: Dictionary mapping token indices to token bytes
        merges: List of BPE merges
        vocab_path: Path to save the vocabulary 
        merges_path: Path to save the merges text file
    """
    byte_encoder = bytes_to_unicode()
    
    # Create the vocab dict in the desired format: {id: unicode_str, ...}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write("{")
        for i, (token_id, token_bytes) in enumerate(vocab.items()):
            # Convert bytes to unicode string
            token_str = ''.join(byte_encoder[b] for b in token_bytes)
            # Format the entry with proper escaping for Unicode characters
            entry = f"{token_id}: \"{token_str}\""
            # Add comma after all entries except the last one
            if i < len(vocab) - 1:
                entry += ", "
            f.write(entry)
        f.write("}")

    # Save merges as a text file
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")
        for first, second in merges:
            first_str = ''.join(byte_encoder[b] for b in first)
            second_str = ''.join(byte_encoder[b] for b in second)
            f.write(f"{first_str} {second_str}\n")