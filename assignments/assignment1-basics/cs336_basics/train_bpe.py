import os
import regex as re
from collections import defaultdict
from typing import BinaryIO, Iterable, Any
import json
import base64
from multiprocessing import Pool, cpu_count

# --- Constants and Helper from pretokenization_example.py ---
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_REGEX = re.compile(PAT_STR)

# Using the provided find_chunk_boundaries function
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    (Implementation from pretokenization_example.py)
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0, 0]
    if desired_num_chunks <= 0: # Ensure at least one chunk
        desired_num_chunks = 1
    if file_size < desired_num_chunks : # Cannot have more chunks than bytes
        desired_num_chunks = file_size if file_size > 0 else 1


    chunk_size = file_size // desired_num_chunks
    if chunk_size == 0 and file_size > 0 : # file_size < desired_num_chunks but file_size > 0
        chunk_size = 1 # Make smallest possible chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # Ensure last boundary is exactly file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1): # Exclude first (0) and last (file_size) fixed boundaries
        
        current_search_file_offset = chunk_boundaries[bi] # Start searching from the initial guess for this boundary
        
        # Boundary case: if the guess itself is already at or beyond file_size (e.g. very small file)
        if current_search_file_offset >= file_size:
            chunk_boundaries[bi] = file_size
            continue

        while True:
            file.seek(current_search_file_offset)
            
            # Adjust read amount if near EOF
            read_amount = mini_chunk_size
            if current_search_file_offset + read_amount > file_size:
                read_amount = file_size - current_search_file_offset
            
            if read_amount <= 0: # Nothing left to read at this position
                chunk_boundaries[bi] = file_size # Set boundary to EOF if token not found
                break

            mini_chunk = file.read(read_amount)

            if not mini_chunk: # Should be caught by read_amount <=0, but as a safeguard
                chunk_boundaries[bi] = file_size
                break

            found_at_in_minichunk = mini_chunk.find(split_special_token)
            if found_at_in_minichunk != -1:
                chunk_boundaries[bi] = current_search_file_offset + found_at_in_minichunk
                break
            
            current_search_file_offset += len(mini_chunk) # Advance by amount actually read

            if current_search_file_offset >= file_size: # If we've scanned past/to EOF
                chunk_boundaries[bi] = file_size # Set to EOF if not found
                break
    
    # Ensure 0 and file_size are present, and all are unique and sorted
    final_boundaries = sorted(list(set([0] + chunk_boundaries + [file_size])))
    # Filter out empty ranges if duplicates caused issues (e.g. [0, 50, 50, 100] -> [0,50,100])
    # The set automatically handles uniqueness.
    return final_boundaries


# --- BPE Training Logic ---

def get_word_counts_from_text_segment(
    text_segment: str, 
    pat_regex_compiled: re.Pattern
) -> dict[tuple[bytes,...], int]:
    """
    Pre-tokenizes a text segment (that doesn't contain special tokens) 
    and counts frequencies of pre-tokens (as tuple of single bytes).
    """
    word_counts = defaultdict(int)
    for match in pat_regex_compiled.finditer(text_segment):
        pre_token_str = match.group(0)
        # Represent pre-token as a tuple of its constituent UTF-8 bytes
        # Each element of the tuple is a single byte, represented as a `bytes` object of length 1
        pre_token_bytes_tuple = tuple(bytes([b]) for b in pre_token_str.encode("utf-8"))
        if pre_token_bytes_tuple: # Ensure not empty
            word_counts[pre_token_bytes_tuple] += 1
    return word_counts

def get_pair_stats(
    word_counts: dict[tuple[bytes,...], int] # word_counts maps (tuple of current tokens) -> count
) -> dict[tuple[bytes, bytes], int]:
    """Counts frequency of adjacent pairs of tokens in the current word_counts."""
    pair_freqs = defaultdict(int)
    for word_as_token_tuple, count in word_counts.items():
        for i in range(len(word_as_token_tuple) - 1):
            pair = (word_as_token_tuple[i], word_as_token_tuple[i+1]) # Pair of current tokens
            pair_freqs[pair] += count
    return pair_freqs

def merge_pair_in_word_counts(
    word_counts: dict[tuple[bytes,...], int], 
    pair_to_merge: tuple[bytes, bytes], # (token1_bytes, token2_bytes)
    new_token: bytes # token1_bytes + token2_bytes
) -> dict[tuple[bytes,...], int]:
    """Merges the given pair into a new token within all words in word_counts."""
    new_word_counts = defaultdict(int)
    for word_as_token_tuple, count in word_counts.items():
        new_word_list = []
        i = 0
        while i < len(word_as_token_tuple):
            if i < len(word_as_token_tuple) - 1 and \
               word_as_token_tuple[i] == pair_to_merge[0] and \
               word_as_token_tuple[i+1] == pair_to_merge[1]:
                new_word_list.append(new_token)
                i += 2
            else:
                new_word_list.append(word_as_token_tuple[i])
                i += 1
        new_word_counts[tuple(new_word_list)] += count
    return new_word_counts

def _process_file_chunk_for_pretok(args: tuple[str, int, int, list[str], re.Pattern, str]) -> dict[tuple[bytes,...], int]:
    """
    Worker function for parallel pre-tokenization.
    Reads a file chunk, decodes, splits by special tokens, then pre-tokenizes segments.
    """
    input_path, start_offset, end_offset, all_special_tokens_str, pat_regex_compiled, encoding = args
    
    chunk_word_counts = defaultdict(int)
    try:
        with open(input_path, "rb") as f:
            f.seek(start_offset)
            byte_data = f.read(end_offset - start_offset)
        
        text_chunk_str = byte_data.decode(encoding, errors="replace")

        if not text_chunk_str:
            return chunk_word_counts

        # Create a regex for splitting by any of the special tokens
        # Sort by length descending to match longer special tokens first
        sorted_special_tokens = sorted(all_special_tokens_str, key=len, reverse=True)
        
        current_segments_to_process = [text_chunk_str]

        if sorted_special_tokens:
            # Build a regex pattern to split by special tokens, e.g., (tok1|tok2)
            special_tokens_pattern_parts = [re.escape(st) for st in sorted_special_tokens]
            split_pattern = re.compile(f"({'|'.join(special_tokens_pattern_parts)})")
            
            temp_segments = []
            for segment in current_segments_to_process:
                parts = split_pattern.split(segment)
                for part in parts:
                    if part and part not in sorted_special_tokens: # Process non-empty, non-special parts
                        temp_segments.append(part)
            current_segments_to_process = temp_segments
        
        for segment in current_segments_to_process:
            if not segment.strip():
                continue
            segment_counts = get_word_counts_from_text_segment(segment, pat_regex_compiled)
            for k, v in segment_counts.items():
                chunk_word_counts[k] += v
        return chunk_word_counts
    except Exception as e:
        print(f"Error processing chunk {input_path} ({start_offset}-{end_offset}): {e}")
        return defaultdict(int)


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    chunking_special_token_str: str = "<|endoftext|>", # For physical file chunking
    encoding: str = "utf-8"
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer.
    """
    vocab: dict[int, bytes] = {}  # Maps token ID (int) to token (bytes)
    merges: list[tuple[bytes, bytes]] = [] # List of merged pairs (token_bytes, token_bytes)

    # 1. Vocabulary Initialization
    current_id = 0
    # Add all 256 individual bytes
    for i in range(256):
        vocab[current_id] = bytes([i])
        current_id += 1
    
    # Add special tokens to vocab
    # Keep track of their byte representations for pre-tokenization splitting
    special_tokens_bytes_set = set()
    for st_str in special_tokens:
        st_bytes = st_str.encode(encoding)
        special_tokens_bytes_set.add(st_bytes)
        # Add to vocab if not already there (e.g. if a special token is a single byte)
        # Check by value:
        already_in_vocab = False
        for _id, b_val in vocab.items():
            if b_val == st_bytes:
                already_in_vocab = True
                break
        if not already_in_vocab:
            vocab[current_id] = st_bytes
            current_id += 1
            
    initial_vocab_size = len(vocab)
    print(f"Initial vocab size (bytes + special tokens): {initial_vocab_size}")

    # 2. Pre-tokenization (Parallelized)
    # This step populates `word_counts`: dict[tuple_of_single_byte_tokens, int]
    # e.g., {( (b't',), (b'h',), (b'e',) ): 5}
    
    word_counts = defaultdict(int)
    
    chunking_special_token_bytes = chunking_special_token_str.encode(encoding)
    
    # Determine number of processes for parallelization
    # Use a reasonable number, e.g., cpu_count() or a fixed number like 4 or 8
    # For very large files, more processes might be beneficial if I/O is not the bottleneck.
    # The problem hint suggests multiprocessing.
    num_processes = min(max(1, cpu_count() // 2), 8) # Heuristic

    try:
        with open(input_path, "rb") as f_bin:
            # desired_num_chunks for find_chunk_boundaries can be heuristic, e.g., num_processes * factor
            boundaries = find_chunk_boundaries(f_bin, num_processes * 4, chunking_special_token_bytes)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return {}, []
    except Exception as e:
        print(f"Error during file chunking: {e}")
        return {}, []

    tasks = []
    if len(boundaries) > 1: # Ensure there's at least one chunk
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            if start < end: # Process non-empty chunks
                tasks.append((input_path, start, end, special_tokens, PAT_REGEX, encoding))
    
    print(f"Starting parallel pre-tokenization with {num_processes} processes for {len(tasks)} tasks...")
    if tasks:
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_file_chunk_for_pretok, tasks)
        
        for chunk_counts in results:
            for k, v in chunk_counts.items():
                word_counts[k] += v
    
    if not word_counts:
        print("Warning: Pre-tokenization resulted in empty word counts. No data to train on?")
        # Return initial vocab if no words found
        return vocab, []


    print(f"Pre-tokenization complete. Unique pre-token sequences: {len(word_counts)}")

    # 3. Compute BPE Merges
    # `current_processing_vocab` stores words as tuples of their current token representations (bytes)
    current_processing_vocab = word_counts.copy()
    
    num_merges_to_perform = vocab_size - len(vocab) # How many new tokens we want to create
    print(f"Targeting {num_merges_to_perform} merges.")

    for i in range(num_merges_to_perform):
        if not current_processing_vocab:
            print("No more data to merge.")
            break
            
        pair_stats = get_pair_stats(current_processing_vocab)
        
        if not pair_stats:
            print("No more pairs to merge.")
            break

        # Find the best pair (highest frequency, then lexicographically largest)
        max_freq = -1
        for freq in pair_stats.values():
            if freq > max_freq:
                max_freq = freq
        
        if max_freq == -1 : break # Should be caught by `if not pair_stats`

        best_pairs_candidates = [pair for pair, freq in pair_stats.items() if freq == max_freq]
        
        if not best_pairs_candidates: break # Should not happen

        best_pair = max(best_pairs_candidates) # Tuple comparison for lexicographical tie-breaking
        
        # Create new token from the best pair
        token1_bytes, token2_bytes = best_pair
        new_token_bytes = token1_bytes + token2_bytes
        
        # Add new token to vocab and record the merge
        # (The problem implies new merged token is added to vocab)
        vocab[current_id] = new_token_bytes
        merges.append(best_pair) # Record the pair of original tokens that were merged
        
        # Update word counts by replacing occurrences of `best_pair` with `new_token_bytes`
        current_processing_vocab = merge_pair_in_word_counts(current_processing_vocab, best_pair, new_token_bytes)
        
        current_id += 1
        if (i + 1) % 100 == 0 or i == num_merges_to_perform -1 :
            print(f"Merge {i+1}/{num_merges_to_perform}: Merged {best_pair[0].decode(encoding,'replace')}+{best_pair[1].decode(encoding,'replace')} -> {new_token_bytes.decode(encoding,'replace')}. Vocab size: {len(vocab)}")

        if len(vocab) >= vocab_size:
            break
            
    print(f"BPE training finished. Final vocab size: {len(vocab)}. Total merges: {len(merges)}.")
    return vocab, merges


# --- Tokenizer Class ---

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens_list: list[str] | None = None, encoding: str = "utf-8"):
        self.encoding = encoding
        self.vocab_int_to_bytes: dict[int, bytes] = vocab.copy()
        self.vocab_bytes_to_int: dict[bytes, int] = {b: i for i, b in vocab.items()}
        
        # Merges are (token1_bytes, token2_bytes)
        # These are actual token values from the vocab that were combined.
        self.merges: list[tuple[bytes, bytes]] = merges 

        self.pat_regex: re.Pattern = PAT_REGEX # Use the same regex as in training

        self.special_tokens_str_set: set[str] = set()
        self.special_tokens_bytes_map: dict[bytes, str] = {} # For quick check if a byte sequence is special

        if special_tokens_list:
            current_max_id = max(self.vocab_int_to_bytes.keys()) if self.vocab_int_to_bytes else -1
            for st_str in special_tokens_list:
                self.special_tokens_str_set.add(st_str)
                st_bytes = st_str.encode(self.encoding)
                self.special_tokens_bytes_map[st_bytes] = st_str

                # Add special tokens to vocab if not already present
                if st_bytes not in self.vocab_bytes_to_int:
                    current_max_id += 1
                    new_id = current_max_id
                    # Ensure ID is unique if vocab isn't contiguous (though it should be from training)
                    while new_id in self.vocab_int_to_bytes:
                        new_id += 1
                    
                    self.vocab_int_to_bytes[new_id] = st_bytes
                    self.vocab_bytes_to_int[st_bytes] = new_id
                    # print(f"Added special token '{st_str}' to vocab with ID {new_id}")
        
        # For splitting text by special tokens during encoding
        if self.special_tokens_str_set:
            sorted_special_tokens_for_splitting = sorted(list(self.special_tokens_str_set), key=len, reverse=True)
            special_tokens_pattern_parts = [re.escape(st) for st in sorted_special_tokens_for_splitting]
            self.special_tokens_split_pattern = re.compile(f"({'|'.join(special_tokens_pattern_parts)})")
        else:
            self.special_tokens_split_pattern = None
        
        # Precompute U+FFFD replacement character's ID if it's in vocab, or its bytes
        self.replacement_char_bytes = b'\xef\xbf\xbd' # UTF-8 for U+FFFD
        self.replacement_char_id = self.vocab_bytes_to_int.get(self.replacement_char_bytes)


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None, encoding: str = "utf-8"):
        try:
            with open(vocab_filepath, 'r') as f:
                raw_vocab_json = json.load(f)
            loaded_vocab = {int(k): base64.b64decode(v) for k, v in raw_vocab_json.items()}
        except Exception as e:
            raise ValueError(f"Error loading vocab from {vocab_filepath}: {e}")

        try:
            with open(merges_filepath, 'r') as f:
                raw_merges_json = json.load(f)
            loaded_merges = [(base64.b64decode(p1), base64.b64decode(p2)) for p1, p2 in raw_merges_json]
        except Exception as e:
            raise ValueError(f"Error loading merges from {merges_filepath}: {e}")
            
        return cls(loaded_vocab, loaded_merges, special_tokens, encoding)

    def _apply_merges_to_byte_sequence(self, byte_tokens: list[bytes]) -> list[bytes]:
        """Applies all learned BPE merges to a sequence of byte tokens."""
        if not byte_tokens or not self.merges:
            return byte_tokens

        current_sequence = list(byte_tokens) # Make a mutable copy

        for p1, p2 in self.merges: # Iterate through ordered merges
            merged_token = p1 + p2 # The token that results from this merge rule
            
            # Only apply if this merged_token is actually in our vocabulary
            # (It should be, if merges and vocab are consistent from training)
            if merged_token not in self.vocab_bytes_to_int:
                # This merge rule might be for an intermediate token not kept, or an issue.
                # For encoding, we only care about merges that result in final vocab tokens.
                # However, the BPE encoding process applies *all* learned merges.
                pass # Continue, apply the merge regardless of whether p1+p2 is a *final* large token

            new_sequence_after_this_merge_pass = []
            i = 0
            while i < len(current_sequence):
                if i < len(current_sequence) - 1 and \
                   current_sequence[i] == p1 and \
                   current_sequence[i+1] == p2:
                    new_sequence_after_this_merge_pass.append(merged_token)
                    i += 2
                else:
                    new_sequence_after_this_merge_pass.append(current_sequence[i])
                    i += 1
            current_sequence = new_sequence_after_this_merge_pass
            if len(current_sequence) == 1 and len(self.merges) > 100: # Optimization for very long merge lists
                # If fully merged to one token, and it's a known token, no more sub-merges can apply to it.
                # This is only safe if merges don't break down existing tokens.
                # BPE merges build up. So if current_sequence[0] is a token, it's final for this pre-token.
                # However, the example ('t','h')->'th', then ('th','e')->'the' implies iterative application.
                # The current loop structure (outer loop over merges, inner rebuilds sequence) is correct.
                pass
        return current_sequence

    def _encode_regular_text_segment(self, text_segment: str) -> list[int]:
        """Encodes a piece of text that is NOT a special token."""
        if not text_segment:
            return []

        # 1. Pre-tokenize using PAT regex
        initial_byte_sequences = [] # List of [list of single-byte tokens]
        for match in self.pat_regex.finditer(text_segment):
            word_str = match.group(0)
            # Each pre-token is initially a list of single-byte `bytes` objects
            single_byte_tokens = [bytes([b]) for b in word_str.encode(self.encoding)]
            if single_byte_tokens:
                initial_byte_sequences.append(single_byte_tokens)
        
        final_token_ids = []
        for byte_list_for_pretoken in initial_byte_sequences:
            # 2. Apply BPE merges to this pre-token's byte sequence
            merged_byte_tokens = self._apply_merges_to_byte_sequence(byte_list_for_pretoken)
            
            # 3. Convert final byte tokens to IDs
            for final_token_bytes in merged_byte_tokens:
                token_id = self.vocab_bytes_to_int.get(final_token_bytes)
                if token_id is not None:
                    final_token_ids.append(token_id)
                else:
                    # This is an issue: a token was formed that's not in the vocab.
                    # This implies the merges might have created something unexpected, or
                    # the original byte sequence couldn't be fully resolved to known tokens.
                    # Fallback: try to encode its constituent single bytes if it's multi-byte.
                    # print(f"Warning: Token {final_token_bytes} (from '{text_segment}') not in vocab after merges. Encoding constituent bytes.")
                    for single_byte_char in final_token_bytes: # Iterate over bytes in the unknown token
                        single_byte_token = bytes([single_byte_char])
                        fallback_id = self.vocab_bytes_to_int.get(single_byte_token)
                        if fallback_id is not None:
                            final_token_ids.append(fallback_id)
                        elif self.replacement_char_id is not None: # Should not happen for single bytes 0-255
                            final_token_ids.append(self.replacement_char_id)
                            # print(f"Critical Error: Single byte {single_byte_token} not in vocab.")
        return final_token_ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        final_ids: list[int] = []
        
        if not self.special_tokens_split_pattern or not self.special_tokens_str_set:
            # No special tokens defined, or pattern failed to compile
            return self._encode_regular_text_segment(text)

        # Split text by special tokens using the precompiled regex
        # `re.split` with a capturing group: (pattern) -> [text_before, special_token1, text_between, ...]
        parts = self.special_tokens_split_pattern.split(text)
        
        for part_str in parts:
            if not part_str: # Skip empty strings that can result from split
                continue

            if part_str in self.special_tokens_str_set:
                # This is a special token string
                st_bytes = part_str.encode(self.encoding)
                token_id = self.vocab_bytes_to_int.get(st_bytes)
                if token_id is not None:
                    final_ids.append(token_id)
                else: # Should not happen if __init__ ensures special tokens are in vocab
                    # print(f"Error: Special token '{part_str}' (bytes: {st_bytes}) not found in vocab during encode.")
                    if self.replacement_char_id is not None: final_ids.append(self.replacement_char_id)
            else:
                # This is a regular text segment
                final_ids.extend(self._encode_regular_text_segment(part_str))
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        Processes each string from the iterable independently.
        """
        for text_chunk in iterable:
            encoded_ids = self.encode(text_chunk)
            for token_id in encoded_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_pieces = []
        for token_id in ids:
            b = self.vocab_int_to_bytes.get(token_id)
            if b is not None:
                byte_pieces.append(b)
            else:
                # Unknown token ID, append replacement character bytes
                byte_pieces.append(self.replacement_char_bytes) 
        
        concatenated_bytes = b"".join(byte_pieces)
        return concatenated_bytes.decode(self.encoding, errors="replace")

    def save_model(self, vocab_filepath: str, merges_filepath: str):
        """Saves the tokenizer's vocabulary and merges to files."""
        # Vocab: dict[int, bytes] -> JSON: {"id_str": "base64_bytes_str"}
        serializable_vocab = {str(k): base64.b64encode(v).decode('ascii') for k, v in self.vocab_int_to_bytes.items()}
        with open(vocab_filepath, 'w') as f:
            json.dump(serializable_vocab, f)

        # Merges: list[tuple[bytes, bytes]] -> JSON: [["b64_str1", "b64_str2"], ...]
        serializable_merges = [[base64.b64encode(p1).decode('ascii'), base64.b64encode(p2).decode('ascii')] for p1, p2 in self.merges]
        with open(merges_filepath, 'w') as f:
            json.dump(serializable_merges, f)
        print(f"Tokenizer model saved: {vocab_filepath}, {merges_filepath}")

# Example Usage (Illustrative - requires a data file)
if __name__ == "__main__":
    # Create a dummy data file for testing train_bpe
    dummy_data_path = "dummy_train_data.txt"
    with open(dummy_data_path, "w", encoding="utf-8") as f:
        f.write("low low low low low\n")
        f.write("lower lower widest widest widest\n")
        f.write("newest newest newest newest newest newest\n")
        f.write("<|endoftext|>\n") # Chunking special token
        f.write("another document here with some more words\n")
        f.write("test test testing\n")
        f.write("<|endoftext|>\n")
        f.write("whitespace    and tabs\t\t\n")
        f.write("pattern's test for 's 'll 've 're 'd 'm 't tokens\n")


    print("--- Training BPE Tokenizer ---")
    # Define special tokens (ensure chunking_special_token_str is one of them or handled)
    # The problem example: vocab has <|endoftext|>
    # "Make sure to add the TinyStories <|endoftext|> special token to the vocabulary."
    custom_special_tokens = ["<|endoftext|>", "<|another_special|>"] 
    
    # The `chunking_special_token_str` for `find_chunk_boundaries` should be a known document delimiter.
    # If it's also in `custom_special_tokens`, it will be handled correctly.
    trained_vocab, trained_merges = train_bpe(
        input_path=dummy_data_path, 
        vocab_size=300, # Small vocab for quick test (256 bytes + a few merges + special)
        special_tokens=custom_special_tokens,
        chunking_special_token_str="<|endoftext|>"
    )

    if trained_vocab:
        print(f"\nTraining complete. Vocab size: {len(trained_vocab)}, Merges: {len(trained_merges)}")
        # print("Sample Merges:", trained_merges[:5])
        # print("Sample Vocab:", {k:v.decode('utf-8', 'replace') for i,(k,v) in enumerate(trained_vocab.items()) if i < 10 or i > len(trained_vocab)-5})


        # --- Using the Tokenizer ---
        print("\n--- Initializing Tokenizer with trained model ---")
        tokenizer = Tokenizer(trained_vocab, trained_merges, custom_special_tokens)
        
        # Save and load (optional test)
        tokenizer.save_model("dummy_vocab.json", "dummy_merges.json")
        tokenizer_loaded = Tokenizer.from_files("dummy_vocab.json", "dummy_merges.json", custom_special_tokens)


        test_text = "low lower newest<|endoftext|> test's pattern<|another_special|>"
        print(f"\nEncoding text: '{test_text}'")
        
        encoded_ids = tokenizer.encode(test_text)
        print(f"Encoded IDs: {encoded_ids}")

        decoded_text = tokenizer.decode(encoded_ids)
        print(f"Decoded text: '{decoded_text}'")
        assert decoded_text == test_text # Should ideally match

        print("\n--- Encoding with loaded tokenizer ---")
        encoded_ids_loaded = tokenizer_loaded.encode(test_text)
        print(f"Encoded IDs (loaded): {encoded_ids_loaded}")
        decoded_text_loaded = tokenizer_loaded.decode(encoded_ids_loaded)
        print(f"Decoded text (loaded): '{decoded_text_loaded}'")
        assert encoded_ids == encoded_ids_loaded
        assert decoded_text == decoded_text_loaded


        print("\n--- Testing encode_iterable ---")
        text_lines = ["first line of text.", "another line with<|endoftext|>token.", "last line."]
        all_ids_iterable = []
        for token_id in tokenizer.encode_iterable(text_lines):
            all_ids_iterable.append(token_id)
        
        all_ids_direct = tokenizer.encode("".join(text_lines)) # For comparison if lines are joined
        # Note: encode_iterable processes each string independently.
        # So, it's more like sum(tokenizer.encode(line) for line in text_lines, [])
        expected_iterable_ids = []
        for line in text_lines:
            expected_iterable_ids.extend(tokenizer.encode(line))
        
        print(f"Encoded IDs from iterable: {all_ids_iterable}")
        print(f"Expected IDs from iterable processing: {expected_iterable_ids}")
        assert all_ids_iterable == expected_iterable_ids

        # Clean up dummy files
        os.remove(dummy_data_path)
        os.remove("dummy_vocab.json")
        os.remove("dummy_merges.json")
        print("\nCleaned up dummy files.")

    else:
        print("BPE Training failed to produce a vocabulary.")
