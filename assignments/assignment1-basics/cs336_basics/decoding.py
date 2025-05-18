"""
Decoding module, used for generating text from trained language models.

Provides different sampling strategies, including temperature parameter adjustment and top-p (nucleus sampling).
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Callable, Union


def temperature_scaled_logits(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Logits with shape [batch_size, vocab_size]
        temperature: Temperature parameter, controls the smoothness of the distribution
            - temperature < 1.0 makes the distribution sharper (more deterministic)
            - temperature > 1.0 makes the distribution smoother (more random)
            - temperature = 1.0 doesn't change the distribution
            
    Returns:
        Temperature-scaled logits
    """
    if temperature == 0.0:
        # Special case: when temperature is 0, perform greedy decoding (take maximum probability)
        return logits * float('inf')
    return logits / max(temperature, 1e-8)  # Prevent division by zero


def top_p_filtering(logits: Tensor, top_p: float = 1.0, filter_value: float = float('-inf')) -> Tensor:
    """
    Perform top-p (nucleus) filtering, keeping tokens whose cumulative probability reaches top_p.
    
    Args:
        logits: Logits with shape [batch_size, vocab_size]
        top_p: Cumulative probability threshold (0.0 to 1.0)
        filter_value: Logit value for filtered tokens
    
    Returns:
        Logits after top-p filtering
    """
    if top_p >= 1.0:
        return logits  # No filtering
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find threshold position (first position exceeding top_p)
    # We don't want to keep the first token that exceeds the threshold, so we need to remove the token that exactly exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Include the first token that exceeds the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Always keep the highest probability token
    
    # Create a mask with the same shape as the original logits
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # Apply the mask
    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    filter_value: float = float('-inf')
) -> Tensor:
    """
    Hybrid sampling: apply temperature scaling and top-p filtering to logits,
    then sample next token via multinomial.
    """
    # Temperature scaling
    scaled = temperature_scaled_logits(logits, temperature)
    # Nucleus filtering
    filtered = top_p_filtering(scaled, top_p, filter_value)
    # Convert to probabilities
    probs = F.softmax(filtered, dim=-1)
    # Sample next token
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    callback: Optional[Callable[[List[int]], None]] = None,
    device: str = "cpu",
) -> List[int]:
    """
    Generate text from a language model.
    
    Args:
        model: A language model with a forward method that takes a token sequence and returns logits for the next token
        prompt: Prompt (context) token sequence
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature used to adjust the logits distribution
            - temperature < 1.0: More conservative predictions (more deterministic)
            - temperature > 1.0: More diverse predictions (more random)
        top_p: Probability threshold for nucleus sampling (0.0 to 1.0)
        eos_token_id: Token ID indicating sequence end, stops generation when encountered
        callback: Callback function called after each token generation (can be used for real-time display)
        device: Device to use ("cpu", "mps", "cuda", etc.)
    
    Returns:
        Generated token sequence (including initial prompt)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move prompt to specified device
    if not isinstance(prompt, torch.Tensor):
        prompt = torch.tensor(prompt, dtype=torch.long)
    
    prompt = prompt.to(device)
    
    # Add batch dimension if needed
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    # Prepare list to store generated results
    generated_ids = prompt.clone()
    
    # Continue generating until max tokens reached or end token encountered
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass to get logits
            outputs = model(generated_ids)
            
            # Get logits for the last step
            next_token_logits = outputs[:, -1, :]
            
            # Sample next token using hybrid sampling
            next_token = sample(next_token_logits, temperature, top_p)
            
            # Append new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Call callback function if provided
            if callback:
                callback(generated_ids[0].tolist())
            
            # Stop generation if end token is generated
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    
    # Return generated sequence (removing batch dimension)
    return generated_ids[0].tolist()


def generate_text(
    model: torch.nn.Module,
    tokenizer,  # Any tokenizer object with encode and decode methods
    prompt_text: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: Optional[str] = "<|endoftext|>",
    verbose: bool = False,
    device: str = "cpu",
) -> str:
    """
    High-level interface for text generation, handling tokenization and decoding.
    
    Args:
        model: Language model
        tokenizer: Tokenizer object with encode and decode methods
        prompt_text: Text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token: End token (text form)
        verbose: Whether to print the generation process
        device: Device to use
    
    Returns:
        Generated text (including initial prompt)
    """
    # Set eos_token_id
    eos_token_id = None
    if eos_token:
        eos_token_id = tokenizer.encode(eos_token)[0]
    
    # Convert text prompt to tokens
    prompt_tokens = tokenizer.encode(prompt_text)
    
    # Callback function for generation process (for verbose mode)
    generated_so_far = []
    
    def callback(token_ids):
        if verbose and len(token_ids) > len(generated_so_far):
            new_tokens = token_ids[len(generated_so_far):]
            new_text = tokenizer.decode(new_tokens)
            print(new_text, end="", flush=True)
            generated_so_far.extend(new_tokens)
    
    # Generate token sequence
    generated_tokens = generate(
        model=model,
        prompt=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        callback=callback if verbose else None,
        device=device
    )
    
    # Convert tokens back to text
    generated_text = tokenizer.decode(generated_tokens)
    
    # Line break (if using verbose printing)
    if verbose:
        print()
    
    return generated_text