import os
import regex as re
from collections import Counter
from typing import List, Union, Tuple, Dict
from io import BytesIO
import pickle

import multiprocessing as mp

def save_with_pickle(vocab_filepath: str, merges_filepath: str, vocab: dict[int, bytes], merges: List[tuple[bytes, bytes]]):
    """Saves vocab and merges using pickle."""
    try:
        # Save vocabulary
        with open(vocab_filepath, 'wb') as f: # 'wb' for binary write
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_filepath}")

        # Save merges
        with open(merges_filepath, 'wb') as f: # 'wb' for binary write
            pickle.dump(merges, f)
        print(f"Merges saved to {merges_filepath}")

    except Exception as e:
        print(f"Error saving with pickle: {e}")

# --- Deserialization (Loading) ---
def load_with_pickle(vocab_filepath: str, merges_filepath: str) -> tuple[dict[int, bytes], List[tuple[bytes, bytes]]]:
    """Loads vocab and merges using pickle."""
    vocab = None
    merges = None
    try:
        # Load vocabulary
        with open(vocab_filepath, 'rb') as f: # 'rb' for binary read
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {vocab_filepath}")

        # Load merges
        with open(merges_filepath, 'rb') as f: # 'rb' for binary read
            merges = pickle.load(f)
        print(f"Merges loaded from {merges_filepath}")

    except FileNotFoundError:
        print("Error: Vocab or merges file not found.")
        # Depending on your needs, you might return None, raise the error, or return empty structures
        raise # Re-raise the file not found error
    except Exception as e:
        print(f"Error loading with pickle: {e}")
        raise # Re-raise other exceptions

    return vocab, merges


def get_pair_stats(
    pretoken_freq: Dict[bytes, Tuple[List[bytes], int]]
) -> Counter[Tuple[bytes, bytes]]:
    """
    Computes the frequency of pairs of pretoken sequences.

    This function takes a dictionary where each key is a pretoken sequence (as bytes)
    and the value is a tuple containing a list of byte sequences and an integer.
    The function counts the frequency of each pair of pretoken sequences.

    Parameters
    ----------
    pretoken_freq : Dict[bytes, Tuple[List[bytes], int]]
        A dictionary where keys are pretoken sequences (as bytes) and values are tuples
        containing a list of the individual characters' bytes and an integer representing the frequency.

    Returns
    -------
    Counter[Tuple[bytes, bytes]]
        A Counter object represetning the frequency of pairs of pretoken sequences.
        For instance, if the input is {b"abc": ([b"a", b"b", b"c"], 3)},
        the output will be {(b"a", b"b"): 3, (b"b", b"c"): 3}.
        
    """
    frequency_counter = Counter()
    # Iterate through each pretoken and its frequency
    for _, (tokens, freq) in pretoken_freq.items():
        # Iterate through the list of byte sequences
        for i in range(len(tokens) - 1):
            # Create a pair of consecutive tokens
            token_pair = (tokens[i], tokens[i + 1])
            # Update the frequency counter for this pair
            frequency_counter[token_pair] += freq
    return frequency_counter


def find_chunk_boundaries(
    path: str = "",
    byte_text_file: bytes = b"",
    num_desired_chunks: int = 1,
    special_split_tokens: Union[bytes, List[bytes]] = b" "
):
    """
    Finds byte offsets in a binary file to serve as chunk boundaries.

    The goal is to split the file into approximately `desired_num_chunks` parts,
    but the actual boundaries are adjusted to coincide with occurrences of the
    `special_split_tokens`. This ensures that chunks can be processed independently
    without splitting the special token itself.

    May return fewer chunks if the boundaries end up overlapping after adjustment
    or if the file is too short to contain the desired number of token occurrences.

    Parameters
    ----------
    path : str
        Path to the file to be chunked.
    num_desired_chunks : int
        The target number of chunks to split the file into.
    special_split_tokens : bytes | List[bytes]
        The byte sequence(s) to use as delimiters for splitting the file.
        If a list is provided, the function will split on any of the tokens in the list.
        If a single bytes object is provided, it will be used as the delimiter.
        
    Returns
    -------
    List[int]
        A sorted list of byte offsets representing the boundaries of the chunks.
        The first element is always 0, and the last element is the end of the file.
    """
    
    # --- Initial Setup ---
    # --- Input Validation and Preparation ---
    # Ensure at least either path or text_file is provided
    if path == "" and (byte_text_file is None or byte_text_file == b""):
         raise ValueError("Either path or text_file (non-empty) must be provided")

    file_size = 0
    
    # If a path is provided, open the file in binary mode
    if path != "":
        try:
            file_object = open(path, "rb")
            file_object.seek(0, os.SEEK_END)
            file_size = file_object.tell()
            # Reset the file pointer to the beginning.
            file_object.seek(0)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return [0]
        except Exception as e:
            print(f"Error opening file: {path}, Error: {e}")
            return [0]
    elif isinstance(byte_text_file, bytes):
        # If a bytes object is provided, we can use it directly
        file_size = len(byte_text_file)
        # Create a file-like object from the bytes
        # This is a workaround to avoid opening a file
        file_object = BytesIO(byte_text_file)
    
    
    # If the file is empty, return a list with a single element: 0
    if file_size == 0:
        return [0]
    
    # Ensure the special_split_tokens is a correct format
    if not isinstance(special_split_tokens, (bytes, list)):
        raise ValueError("special_split_tokens must be a bytes or a list of bytes.")
    if isinstance(special_split_tokens, list):
         if not all(isinstance(token, bytes) for token in special_split_tokens):
              raise TypeError("If special_split_tokens is a list, all elements must be bytes")
         if not special_split_tokens: # Handle empty list case
             raise ValueError("special_split_tokens list is empty. Cannot find delimiters.")
         # If it's a list, keep it as the list for pattern compilation
         split_tokens_list = special_split_tokens
    else:
        # If it's a single bytes object, wrap it in a list for pattern compilation
        split_tokens_list = [special_split_tokens]
        
    # Compile a regex pattern to efficiently search for ANY of the specified tokens.
    # Escape each token's bytes and join them with the regex OR operator (|).
    # Example: [b'\n', b'<|end|>'] -> b'\\n|<\\|end\\|>'
    # This pattern will match the first occurrence of any token in the list.
    compiled_split_pattern = re.compile(b"|".join(re.escape(token) for token in split_tokens_list))
    

    
    # Calculate the approximated chunk size based on the file size and number of desired chunks
    # To generate the initial guess positions
    chunk_size = file_size // num_desired_chunks
    
    # Generate the original guess boundaries
    # These are uniformly spaced offsets.
    # The list will contain desired_num_chunks + 1 elements, representing the
    # start of desired_num_chunks segments and the end of the last segment.
    # [0, b1, b2...]
    # First chunk contains data from byte 0 to byte b1 - 1
    chunk_boundaries = [i * chunk_size for i in range(num_desired_chunks + 1)]
    # The last boundary is the end of the file
    chunk_boundaries[-1] = file_size
    
    # A small buffer size of reading chunks of data around the guessed boundaries
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time to find the token.
    
    last_boundary = 0
    
    # Iterate through the guessed boundaries
    # These are the boundaries that need adjustment.
    for boundery_iter in range(1, len(chunk_boundaries) - 1):
        # if chunk_boundaries[boundery_iter] <= chunk_boundaries[boundery_iter - 1]:
        #     # If the current boundary guess is less than or equal to the previous one,
        #     # we set it to be the same as the previous one.
        #     # This can happen if the file is too small or if the split token
        #     # is not found in the expected location.
        #     # This is a safety check to avoid overlapping boundaries.
        #     chunk_boundaries[boundery_iter] = chunk_boundaries[boundery_iter - 1]
        
        # Determine the initial position for searching the split token
        # The initial position is the current boundary guess
        initial_position = max(chunk_boundaries[boundery_iter], last_boundary + 1)
        
        # Set the pointer to the initial position
        file_object.seek(initial_position)
        
        current_position = initial_position
        
        # Search for the special split token starting from the initial guess position
        # We read in small chunks to avoid loading a huge chunk into memory.
        while True:
            # Read a mini chunk from the current position
            mini_chunk = file_object.read(mini_chunk_size)
            
            # If we read an empty bytestring, it means we have hit the end of the file.
            # The boundary should be set the the end of the file.
            if mini_chunk == b"":
                chunk_boundaries[boundery_iter] = file_size
                break
            
            # If not empty
            # Try to find the special split token within the chunk
            found_at = compiled_split_pattern.search(mini_chunk)
            
            # If the token is found, we adjust the boundary to the position of the token
            if found_at is not None:
                # Calculate the new boundary position
                # The position is the initial position + the offset of the found token
                new_boundary = current_position + found_at.start()
                
                chunk_boundaries[boundery_iter] = new_boundary
                
                # Break out of the while loop since we found a token
                break
            # If the token is not found, we continue reading the next mini chunk
            # This will continue until we either find the token or reach the end of the file.
            else:
                # Update the initial position to the end of the mini chunk
                initial_position += len(mini_chunk)
    
    # Convert the list to a set to remove duplicate positions
    # which can happen if multiple initial guess lead to the same token
    # This is a safety check to avoid overlapping boundaries.
    return sorted(set(chunk_boundaries))


def process_chunk(
    chunk: bytes,
    split_patterns: str,
    special_split_tokens: Union[bytes, List[bytes]] = b" "
) -> Counter[bytes]:
    """
    Worker function for multiprocessing during pretokenization.

    Reads a specific chunk of a file, splits it based on special tokens,
    then applies a pre-compiled regex pattern to further break down segments
    into initial pretokens, and counts the frequency of these pretokens within the chunk.

    Parameters
    ----------
    chunk : bytes
        The chunk of text to be processed.
    special_split_tokens : bytes | List[bytes]
        The byte sequence(s) to use as delimiters for splitting the chunk.
        If a list is provided, the function will split on any of the tokens in the list.
        If a single bytes object is provided, it will be used as the delimiter.

    Returns
    -------
    Counter[bytes]
        A Counter object containing the byte sequences of the pretokens and the pretoken's frequency.
        For instance, a "abc" token with a frequency of 3 will be represented as:
        {b"a", b"b", b"c": 3}
        The keys are the pretokens (as bytes), and the values are their respective counts.
    """
    
    # --- Input Validation ---
    # Ensure the chunk is not empty
    if not chunk:
        raise ValueError("Chunk is empty. Cannot process an empty chunk.")
    # Ensure the split_patterns is a valid regex pattern
    if not isinstance(split_patterns, (bytes, str, re.Pattern)):
        raise ValueError("split_patterns must be a bytes or a string.")
    if isinstance(split_patterns, bytes):
        # If it's a bytes object, decode it to a string for regex processing
        split_patterns = split_patterns.decode("utf-8")
    if not split_patterns:
        raise ValueError("split_patterns is empty. Cannot process an empty pattern.")

    # Ensure the special_split_token is a correct format
    if not isinstance(special_split_tokens, (bytes, list)):
        raise ValueError("special_split_token must be a bytes or a list of bytes.")
    if isinstance(special_split_tokens, list):
        if not all(isinstance(token, bytes) for token in special_split_tokens):
            raise TypeError("If special_split_token is a list, all elements must be bytes")
        if not special_split_tokens:
            # Handle empty list case
            raise ValueError("special_split_token list is empty. Cannot find delimiters.")
        # If it's a list, keep it as the list for pattern compilation
        special_split_tokens = special_split_tokens
    else:
        # If it's a single bytes object, wrap it in a list for pattern compilation
        special_split_tokens = [special_split_tokens]
    
    # Compile the regex pattern to efficiently search for ANY of the specified tokens.
    compiled_split_pattern = re.compile(b"|".join(re.escape(token) for token in special_split_tokens))
        
        
    
    # Initialize a Counter to store the frequency of pretokens within this chunk
    pretokens_counter = Counter()
    
    # Separate the chunk into segments based on the special tokens
    segments: List[bytes] = []
    # If the split token is provided, split the chunk into segments
    if special_split_tokens:
        segments = compiled_split_pattern.split(chunk)
    # If no split token is provided, treat the entire chunk as a single segment
    else:
        segments = [chunk]
        
    # Iterate through each segment
    for segment in segments:
        # If the segment is empty, skip it
        if not segment:
            continue
        
        # Use the pre-compiled regex pattern to find all matches in the segment
        matches = split_patterns.finditer(segment)

        # Iterate through the matches
        for match in matches:
            # Extract the matched pretokens
            pretokens = match.group(0)
            # Update the frequency count for this pretokens
            pretokens_counter[pretokens] += 1
    
    return pretokens_counter
    
def merge_byte_pairs(
    pretoken_freq: Dict[bytes, Tuple[List[bytes], int]],
    byte_pairs_freq: Counter[Tuple[bytes, bytes]],
    best_pair: Tuple[bytes, bytes]
):
    """
    Performs a single BPE merge operation on the pretoken frequency dictionary.

    This function takes the most frequent pair of bytes and merges them into a single token.
    It updates the frequency dictionary to reflect this merge, and also updates the byte pair statistics.
    The function returns the updated pretoken frequency dictionary.
    
    Parameters
    ----------
    pretoken_freq : Dict[bytes, Tuple[List[bytes], int]]
        A dictionary where keys are pretoken sequences (as bytes) and values are tuples
        containing a list of the individual characters' bytes and an integer representing the frequency.
    byte_pairs_freq : Counter[Tuple[bytes, bytes]]
        A Counter object representing the frequency of pairs of pretoken sequences.
        For instance, if the input is {b"abc": ([b"a", b"b", b"c"], 3)},
        the output will be {(b"a", b"b"): 3, (b"b", b"c"): 3}.
    best_pair : Tuple[bytes, bytes]
        The most frequent pair of bytes to be merged.
    Returns
    -------
    None
        The function modifies the pretoken_freq and byte_pairs_freq dictionaries in place.
        It does not return any value.

    """
    first, second = best_pair
    # Create a new token by merging the two bytes
    new_token = first + second

    # Use a list comprehension to get a list of items to iterate over
    # This avoids issues if the loop structure somehow interfered with the dictionary
    # during the update inside the loop, though updating the value for an existing
    # key is generally safe. Copying the items first is the most robust approach.
    items_to_process = list(pretoken_freq.items())
    
    deltas = Counter() # Counter to track changes in pair frequencies


    # Iterate through the pretoken frequency dictionary items
    for pretoken, (byte_list_tokens, freq) in items_to_process:
        num_bytes = len(byte_list_tokens)

        # If the pretoken contains less than 2 bytes, it cannot contain the pair to merge.
        if num_bytes < 2:
            continue

        # --- Perform Merge on this specific pretoken's byte list ---
        new_byte_list_tokens = [] # List to build the modified byte list for the current pretoken
        modified = False # Flag to track if the current pretoken was modified by the merge
        i = 0 # Initialize index for the while loop

        # Iterate through the byte list tokens using a while loop for manual index control
        while i < num_bytes:
            # Check if the current position starts the best_pair and we are not at the very last byte.
            if i < num_bytes - 1 and byte_list_tokens[i] == first and byte_list_tokens[i + 1] == second:
                # Found an occurrence of the pair. Add the new merged token.
                new_byte_list_tokens.append(new_token)

                # Skip the two bytes that were merged. Advance index by 2.
                i += 2

                # Mark that this pretoken's list was modified.
                modified = True
            else:
                # Pair not found at this position. Add the current token as is.
                new_byte_list_tokens.append(byte_list_tokens[i])

                # Move to the next token. Advance index by 1.
                i += 1

        # --- Update Frequencies and Dictionary if the pretoken's list was modified ---
        if modified:
            # Calculate changes in pair frequencies caused by the merge in this pretoken.

            # Decrement the frequency of all original pairs in this pretoken by its frequency.
            # These pairs are potentially destroyed by the merge.
            for pair in zip(byte_list_tokens[:-1], byte_list_tokens[1:]):
                deltas[pair] -= freq

            # Increment the frequency of all new pairs in the modified byte list by the pretoken's frequency.
            # These are pairs that are newly formed after the merge.
            # Note: This calculation needs the *final* state of new_byte_list_tokens for this pretoken.
            for pair in zip(new_byte_list_tokens[:-1], new_byte_list_tokens[1:]):
                deltas[pair] += freq

            # Update the pretoken_freq dictionary with the new byte list for this pretoken.
            # The key remains the original pretoken bytes, but the value's list is updated.
            pretoken_freq[pretoken] = (new_byte_list_tokens, freq)

    # --- Apply Accumulated Frequency Changes and Cleanup ---
    # Apply the total changes in pair frequencies to the main byte_pairs_freq Counter.
    byte_pairs_freq.update(deltas)

    # Clean up byte_pairs_freq: remove pairs whose frequency has dropped to 0 or less.
    # This is important for performance as it keeps the pair_freq dictionary from growing excessively
    # with pairs that no longer exist or are very rare.
    pairs_to_remove = [pair for pair, freq in byte_pairs_freq.items() if freq <= 0]
    for pair in pairs_to_remove:
        del byte_pairs_freq[pair]

    # Explicitly remove the merged pair (best_pair) from the byte_pairs_freq dictionary.
    # Its count should already be <= 0 from the deltas update, but this is a safeguard
    # and clearly indicates this pair is no longer available for merging.
    if best_pair in byte_pairs_freq:
        del byte_pairs_freq[best_pair]

    
    
if __name__ == "__main__":
    text_file = open("data/TinyStoriesV2-GPT4-valid.txt", "rb")
    special_tokens = ["<|endoftext|>".encode("utf-8"), "<|end|>".encode("utf-8")]
    chunks = find_chunk_boundaries("", text_file, 256, special_tokens)
    # print the second chunk
    # print(chunks)
    # text_file.seek(chunks[0])
    # print(text_file.read(chunks[1] - chunks[0]))
    
    # Regular expression pattern (as bytes) used for pretokenization, inspired by GPT-2.
    # Breaks text into potential initial tokens like words, numbers, punctuation, and whitespace.
    # Explanation of the pattern:
    # """'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # - '(?:[sdmt]|ll|ve|re): Matches common English contractions starting with an apostrophe ('s, 'd, 'm, 't, 'll, 've, 're).
    # - |: OR operator.
    # - ?\p{L}+: Matches one or more Unicode letters (\p{L}), optionally preceded by a space (?).
    # - | ?\p{N}+: Matches one or more Unicode numbers (\p{N}), optionally preceded by a space (?).
    # - | ?[^\s\p{L}\p{N}]+: Matches one or more characters that are NOT whitespace (\s), letters (\p{L}), or numbers (\p{N}), optionally preceded by a space (?). This captures punctuation and symbols.
    # - |\s+(?!\S): Matches one or more whitespace characters (\s+) that are NOT followed by a non-whitespace character ((?!\S) is a negative lookahead). This handles trailing whitespace.
    # - |\s+: Matches one or more whitespace characters (\s+). Catches any remaining whitespace.
    GPT2_PAT = br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    COMPILED_GPT2_PAT = re.compile(GPT2_PAT)
    
    chunk_counter = process_chunk(text_file.read(chunks[2] - chunks[1]), COMPILED_GPT2_PAT, special_tokens)
    # print(chunk_counter)