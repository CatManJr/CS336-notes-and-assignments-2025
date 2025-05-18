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
    # Using numpy randint, includes 0, excludes max_start+1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    
    # Build x and y batches
    # x: each sample is context_length tokens starting from the random starting position
    # y: each sample is context_length tokens starting from the random starting position + 1
    x = torch.stack([torch.from_numpy(dataset[s:s + context_length]).long() for s in starts])
    y = torch.stack([torch.from_numpy(dataset[s + 1:s + 1 + context_length]).long() for s in starts])
    
    # Move to device
    try:
        x = x.to(dev)
        y = y.to(dev)
    except Exception as e:
        raise RuntimeError(f"CUDA error or invalid device: {device}") from e
        
    return x, y
