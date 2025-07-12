"""
SFT helper functions.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any, Optional, Tuple
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """
    Tokenize a batch of prompts and outputs.

    Args:
        prompt_strs: A list of strings, each an SFT prompt.
        output_strs: A list of strings, each an SFT output.
        tokenizer: The tokenizer to use.

    Returns:
        A dictionary with the following keys:
        - "input_ids": A tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
        - "labels": A tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
        - "response_mask": A tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
    """
    # Tokenize prompts and outputs separately, then manually concatenate
    prompt_tokens = []
    output_tokens = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        p_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        o_tokens = tokenizer.encode(output, add_special_tokens=False)
        prompt_tokens.append(p_tokens)
        output_tokens.append(o_tokens)
    
    # Manually concatenate tokens
    combined_tokens = []
    prompt_lengths = []
    
    for p_tokens, o_tokens in zip(prompt_tokens, output_tokens):
        combined = p_tokens + o_tokens
        combined_tokens.append(combined)
        prompt_lengths.append(len(p_tokens))
    
    # Find max length as per spec: max(prompt_and_output_lens)
    max_combined_len = max(len(tokens) for tokens in combined_tokens)
    target_len = max_combined_len - 1  # As per spec: max(prompt_and_output_lens) - 1
    
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    
    # Create input_ids: tokenized prompt and output strings, with final token sliced off for longer sequences
    input_ids = torch.full((len(combined_tokens), target_len), pad_token_id, dtype=torch.long)
    for i, tokens in enumerate(combined_tokens):
        if len(tokens) > target_len:
            # For sequences longer than target_len, slice off the final token
            input_tokens = tokens[:-1]
        else:
            # For sequences equal to or shorter than target_len, keep all tokens
            input_tokens = tokens
        
        actual_len = min(len(input_tokens), target_len)
        input_ids[i, :actual_len] = torch.tensor(input_tokens[:actual_len], dtype=torch.long)
    
    # Create labels: shifted input ids (input ids without the first token)
    labels = torch.full((len(combined_tokens), target_len), pad_token_id, dtype=torch.long)
    for i, tokens in enumerate(combined_tokens):
        if len(tokens) > target_len:
            # For sequences longer than target_len, labels = tokens[1:target_len+1]
            label_tokens = tokens[1:target_len+1]
        else:
            # For sequences equal to or shorter than target_len, labels = tokens[1:] + padding
            label_tokens = tokens[1:]
        
        actual_len = min(len(label_tokens), target_len)
        labels[i, :actual_len] = torch.tensor(label_tokens[:actual_len], dtype=torch.long)
    
    # Create response mask: mask on the response tokens in the labels
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    for i in range(len(labels)):
        prompt_len = prompt_lengths[i]
        original_len = len(combined_tokens[i])
        
        # Response tokens in labels start at (prompt_len - 1) due to shifting
        # and go until the end of actual content
        response_start = max(0, prompt_len - 1)  # -1 because labels shifted left by 1
        response_end = min(original_len - 1, target_len)  # -1 because we removed first token
        
        if response_end > response_start:
            response_mask[i, response_start:response_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of a batch of logits.

    Args:
        logits: A tensor of shape (batch_size, seq_len, vocab_size)

    Returns:
        A tensor of shape (batch_size, seq_len)
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, Any]:
    """
    Computes the log probabilities of the response tokens.

    Args:
        model: The model to use.
        input_ids: A tensor of shape (batch_size, seq_len)
        labels: A tensor of shape (batch_size, seq_len)
        return_token_entropy: Whether to return the token-level entropy.

    Returns:
        A dictionary with the following keys:
        - "log_probs": A tensor of shape (batch_size, seq_len)
        - "token_entropy" (optional): A tensor of shape (batch_size, seq_len)
    """
    # Forward pass to get the logits.
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the target tokens in labels
    # We need to handle the case where labels has -100 values
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out positions where labels == -100
    valid_mask = (labels != -100)
    log_probs_out = gathered_log_probs * valid_mask.float()

    result = {
        "log_probs": log_probs_out,
    }

    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy * valid_mask.float()

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Normalizes a tensor using a mask.

    Args:
        tensor: The tensor to normalize.
        mask: The mask to use.
        normalize_constant: The constant to normalize by.
        dim: The dimension to sum over.

    Returns:
        The normalized tensor.
    """
    # Apply the mask
    masked_tensor = tensor * mask.float()

    # Sum over specified dimension
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
    
    # Normalize by the normalize_constant
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Computes the loss for a single microbatch of SFT.

    The loss should be the mean of the negative log probabilities of the response tokens.
    The loss should be scaled by `1 / gradient_accumulation_steps`.
    """
    
    # Calculate the sum of negative log probabilities for response tokens only
    masked_log_probs = policy_log_probs * response_mask.float()
    sum_log_probs = masked_log_probs.sum()

    # The loss is the negative sum of log probabilities divided by normalize_constant
    # and scaled by 1/(gradient_accumulation_steps * 2)
    loss = -sum_log_probs / (normalize_constant * gradient_accumulation_steps * 2)

    # Call backward() as required by the specification
    loss.backward()

    # Return the loss for logging
    metadata = {"loss": loss.item()}

    return loss, metadata


def log_generations(
    prompts: List[str],
    generations: List[str],
    rewards: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    log_to_stdout: bool = True,
) -> None:
    pass
