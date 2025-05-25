# -*- coding: utf-8 -*-
"""
Created on 5-18-2025
Arthur: Catman Jr.
"""
import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # truncated normal init: mean=0, std=sqrt(2/(in+out)), truncate at +/-3 std
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> output: (..., out_features)
        return x @ self.weight.t()

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # weight shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # truncated normal init: mean=0, std=1, truncate at +/-3
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...,) long tensor
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype
        orig_dtype = x.dtype
        # Upcast to float32 for numerical stability
        x_fp32 = x.to(torch.float32)
        # Compute RMS over last dimension
        rms = torch.sqrt(torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps)
        # Normalize and apply gain
        y = x_fp32 / rms * self.gain
        # Cast back to original dtype
        return y.to(orig_dtype)

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network: out = W2 * (SiLU(W1 x) * (W3 x))
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # define weight parameters: W1: (d_ff, d_model), W3: (d_ff, d_model), W2: (d_model, d_ff)
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        # initialize weights with truncated normal std = sqrt(2/(in+out))
        std1 = (2.0 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std1, a=-3*std1, b=3*std1)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std1, a=-3*std1, b=3*std1)
        std2 = (2.0 / (d_ff + d_model)) ** 0.5
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std2, a=-3*std2, b=3*std2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # linear projections
        x1 = torch.nn.functional.linear(x, self.w1)
        x3 = torch.nn.functional.linear(x, self.w3)
        # SwiGLU gating
        gate = torch.nn.functional.silu(x1) * x3
        # final projection
        return torch.nn.functional.linear(gate, self.w2)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize RoPE module.
        
        Args:
            theta: Θ value for the RoPE
            d_k: dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        # Pre-compute the cos and sin values for all positions
        # shape: (max_seq_len, d_k/2)
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_k, 2, device=device).float() * -(math.log(theta) / d_k)
        )
        
        # Compute frequency * position for all positions and freq pairs
        # This gives us the angles for the rotations
        freqs = position * div_term
        
        # Pre-compute cos and sin values, shape: (max_seq_len, d_k/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Register the values as buffers so they'll be saved with the model
        # but won't be updated during training
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
            token_positions: Tensor of shape (batch_size, seq_len) with token positions
            
        Returns:
            Tensor of same shape as x with rotary position embeddings applied
        """
        # Save original shape to reshape at the end
        orig_shape = x.shape
        seq_len = orig_shape[-2]
        
        # Reshape to handle batch dimensions
        # We need to view x as (batch_size, num_heads, seq_len, d_k/2, 2) to work on pairs
        x = x.view(*x.shape[:-1], -1, 2)
        
        # Get the cos and sin values for the positions in this sequence
        # First, we need to ensure token_positions are valid indices
        valid_positions = torch.clamp(token_positions, 0, self.max_seq_len - 1)
        
        # Select the right cos/sin values for these positions
        # Expand cos/sin to match the batch dimensions of x
        cos = self.cos[valid_positions]  # shape: (batch_size, seq_len, d_k/2)
        sin = self.sin[valid_positions]  # shape: (batch_size, seq_len, d_k/2)
        
        # Add head dimension for broadcasting
        cos = cos.unsqueeze(1)  # shape: (batch_size, 1, seq_len, d_k/2)
        sin = sin.unsqueeze(1)  # shape: (batch_size, 1, seq_len, d_k/2)
        
        # Extract even and odd dimensions
        x_even = x[..., 0]  # shape: (batch_size, num_heads, seq_len, d_k/2)
        x_odd = x[..., 1]   # shape: (batch_size, num_heads, seq_len, d_k/2)
        
        # Apply the rotation: 
        # [cos -sin] [x_even]
        # [sin  cos] [x_odd ]
        x_rotated = torch.stack([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ], dim=-1)
        
        # Reshape back to the original shape
        return x_rotated.reshape(orig_shape)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute softmax along a specific dimension with numerical stability.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
    
    Returns:
        Tensor of same shape as input with softmax applied along specified dimension
    """
    # Clone input tensor to avoid modifying it
    x_safe = x.clone()
    
    # For numerical stability, subtract the maximum value along the given dimension
    # keepdim=True ensures the output has the same shape for broadcasting
    max_vals = torch.max(x_safe, dim=dim, keepdim=True)[0]
    
    # Subtract max values and compute exponentials
    x_safe = x_safe - max_vals
    exp_x = torch.exp(x_safe)
    
    # Normalize by the sum of exponentials
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention as described in 'Attention Is All You Need'.
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., keys, d_v)
        mask: Optional mask tensor of shape (..., queries, keys)
    
    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    # Extract the scale factor from key dimension
    d_k = K.size(-1)
    scale = d_k ** 0.5
    
    # Compute scaled Q·K^T attention scores
    # (..., queries, d_k) @ (..., d_k, keys) -> (..., queries, keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    
    # Apply mask if provided (we want masked positions to have -inf scores)
    if mask is not None:
        # Convert to float and apply mask (True = keep, False = mask out)
        # We use -float('inf') to zero out attention weights with softmax
        scores = scores.masked_fill(mask == False, -float('inf'))
    
    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)
    
    # Apply attention weights to values
    # (..., queries, keys) @ (..., keys, d_v) -> (..., queries, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output
