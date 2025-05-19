import regex as re
import json
from typing import Dict, List, Tuple, Union, Optional, Iterable
from collections import defaultdict

class BPETokenizer:
    """
    A byte-level Byte Pair Encoding (BPE) tokenizer.
    
    This tokenizer implements the same algorithm as used in GPT-2.
    """
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize a BPE tokenizer with the given vocabulary and merges.
        
        Args:
            vocab: Dictionary mapping token IDs to token bytes
            merges: List of BPE merges as tuples of (first, second) bytes
            special_tokens: List of special tokens that should be treated atomically
            vocab_size: Size of vocabulary, used to enforce token ID bounds
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_size = vocab_size
        
        # Create a mapping from token bytes to token IDs
        self.token_bytes_to_id = {token: token_id for token_id, token in vocab.items()}
        
        # Create a dictionary of BPE merges for fast lookup
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # Pre-compile the regex pattern for tokenization
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Special tokens as bytes for fast lookup
        self.special_token_bytes = [token.encode('utf-8') for token in self.special_tokens]
        
        # Escaped special tokens for regex splitting (longer tokens first to handle overlaps)
        ordered_tokens = sorted(self.special_tokens or [], key=lambda s: len(s), reverse=True)
        escaped_special_tokens = [re.escape(token) for token in ordered_tokens]
        self.special_token_pattern = re.compile('|'.join(escaped_special_tokens)) if escaped_special_tokens else None
    
    @classmethod
    def from_file(cls, vocab_path: str, merges_path: str, special_tokens: Optional[List[str]] = None):
        """
        Load a BPE tokenizer from vocab and merges files.
        
        Args:
            vocab_path: Path to the vocabulary JSON file
            merges_path: Path to the merges text file
            special_tokens: List of special tokens that should be treated atomically
            
        Returns:
            A BPETokenizer instance
        """
        from .train_bpe import bytes_to_unicode, unicode_to_bytes
        
        # Load vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            str_vocab = json.load(f)
        
        # 计算并保存词汇表大小
        vocab_size = len(str_vocab)
        
        # Convert string vocab to bytes vocab
        byte_decoder = unicode_to_bytes()
        vocab = {}
        for token_str, token_id in str_vocab.items():
            token_bytes = bytes([byte_decoder.get(c, ord(c)) for c in token_str])
            vocab[token_id] = token_bytes
        
        # Load merges
        merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                first_str, second_str = line.rstrip().split(' ')
                first = bytes([byte_decoder.get(c, ord(c)) for c in first_str])
                second = bytes([byte_decoder.get(c, ord(c)) for c in second_str])
                merges.append((first, second))
        
        # 传递词汇表大小
        return cls(vocab, merges, special_tokens, vocab_size)
    
    def bpe(self, token: bytes) -> List[bytes]:
        """
        Apply the BPE algorithm to a single pre-token.
        
        Args:
            token: A bytes object representing a pre-token
            
        Returns:
            A list of bytes objects representing the BPE tokens
        """
        # Check if token is a special token
        if token in self.special_token_bytes:
            return [token]
        
        # If token is a single byte, return it
        if len(token) == 1:
            return [token]
        
        # Split the token into individual bytes
        parts = [bytes([b]) for b in token]
        
        # Apply merges iteratively
        while len(parts) > 1:
            # Find the highest-ranked pair
            min_rank = float('inf')
            min_pair_idx = -1
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair_idx = i
            
            # If no pair found, we're done
            if min_pair_idx == -1:
                break
            
            # Otherwise, merge the pair
            parts = (
                parts[:min_pair_idx] + 
                [parts[min_pair_idx] + parts[min_pair_idx + 1]] + 
                parts[min_pair_idx + 2:]
            )
        
        return parts
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            A list of token IDs
        """
        # Handle special tokens separately
        if self.special_token_pattern:
            parts = []
            last_end = 0
            for match in self.special_token_pattern.finditer(text):
                # Add text before the special token
                if match.start() > last_end:
                    parts.append(text[last_end:match.start()])
                
                # Add the special token
                parts.append(match.group(0))
                last_end = match.end()
            
            # Add any remaining text
            if last_end < len(text):
                parts.append(text[last_end:])
        else:
            parts = [text]
        
        token_ids = []
        
        for part in parts:
            # If part is a special token, add its ID directly
            if part in self.special_tokens:
                special_token_bytes = part.encode('utf-8')
                token_ids.append(self.token_bytes_to_id[special_token_bytes])
                continue
            
            # Otherwise, tokenize using regex and BPE
            matches = self.pattern.finditer(part)
            for match in matches:
                token = match.group(0)
                token_bytes = token.encode('utf-8')
                
                # Apply BPE to get the final tokens
                bpe_tokens = self.bpe(token_bytes)
                
                # Convert tokens to IDs
                for bpe_token in bpe_tokens:
                    if bpe_token in self.token_bytes_to_id:
                        token_ids.append(self.token_bytes_to_id[bpe_token])
                    else:
                        # Handle unknown tokens by splitting into individual bytes
                        for b in bpe_token:
                            token_ids.append(self.token_bytes_to_id[bytes([b])])
        
        # 在返回前检查是否有超出词汇表大小的token
        if self.vocab_size is not None:
            for idx, token_id in enumerate(token_ids):
                if token_id >= self.vocab_size:
                    # 选项1: 替换为<unk> token (假设ID为0)
                    token_ids[idx] = 0
                    # 选项2: 抛出警告
                    import warnings
                    warnings.warn(f"Token ID {token_id} exceeds vocabulary size {self.vocab_size}")
                    # 选项3: 抛出错误
                    # raise ValueError(f"Token ID {token_id} exceeds vocabulary size {self.vocab_size}")
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.
        
        Args:
            token_ids: The token IDs to decode
            
        Returns:
            The decoded text
        """
        # 检查是否有超出词汇表的token_id
        valid_tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                valid_tokens.append(self.vocab[token_id])
            else:
                # 选择1: 忽略无效的token_id
                continue
                # 选择2: 替换为一个特殊字符
                # valid_tokens.append(b'<unk>')
        
        # 连接所有token并解码为字符串
        text = b''.join(valid_tokens).decode('utf-8', errors='replace')
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string into a list of token strings.
        
        Args:
            text: The text to tokenize
            
        Returns:
            A list of token strings
        """
        token_ids = self.encode(text)
        tokens = [self.vocab[token_id].decode('utf-8', errors='replace') for token_id in token_ids]
        return tokens
    
    def save(self, vocab_path: str, merges_path: str):
        """
        Save the tokenizer vocabulary and merges to disk.
        
        Args:
            vocab_path: Path to save the vocabulary JSON
            merges_path: Path to save the merges text file
        """
        from .train_bpe import save_tokenizer
        save_tokenizer(self.vocab, self.merges, vocab_path, merges_path)