# -*- coding: utf-8 -*-
"""
Created on 5-18-2025
Arthur: Catman Jr.
"""

import numpy as np
import torch
from typing import Tuple


def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and corresponding labels from the dataset.
    Optimized version for large batch sizes.
    
    Args:
        dataset: A one-dimensional numpy array containing integer token IDs
        batch_size: The batch size to sample
        context_length: The context length for each sampled example
        device: PyTorch device string (such as 'cpu', 'cuda:0', or 'mps'),
               indicating which device the sampled input sequences and labels should be placed on
    
    Returns:
        A tuple of torch.LongTensor with shape (batch_size, context_length).
        The first item in the tuple is the sampled input sequences, and the second is the corresponding language modeling labels.
    """
    # Validate device
    try:
        dev = torch.device(device)
    except Exception as e:
        raise RuntimeError(f"Invalid device: {device}") from e
        
    # Dataset length and possible starting indices
    data_len = dataset.shape[0]
    max_start = data_len - context_length - 1  # -1 because we need one extra token for the target
    if max_start < 0:
        raise RuntimeError("Dataset too small for the given context length")
    
    # Uniformly sample random starting indices
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    
    # Pre-allocate tensors on the target device
    x = torch.empty((batch_size, context_length), dtype=torch.long, device=dev)
    y = torch.empty((batch_size, context_length), dtype=torch.long, device=dev)
    
    # Use vectorized operations to fill the tensors
    for i, start in enumerate(starts):
        x[i] = torch.from_numpy(dataset[start:start + context_length])
        y[i] = torch.from_numpy(dataset[start + 1:start + 1 + context_length])
    
    return x, y
