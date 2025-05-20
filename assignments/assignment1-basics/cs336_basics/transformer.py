import torch
import torch.nn as nn
import math
from .nn_utils import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding
from .nn_utils import softmax, scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Initialize a Multi-Head Self-Attention module.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            device: Device to place the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Ensure d_model is divisible by num_heads for even splitting
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Store dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for queries, keys, values
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, use_rope: bool = False, rope: RotaryPositionalEmbedding = None, token_positions=None) -> torch.Tensor:
        """
        Compute causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            use_rope: Whether to use RoPE (rotary positional embeddings)
            rope: RoPE module to use if use_rope is True
            token_positions: Token positions for RoPE if use_rope is True
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_size = x.size(0)
        seq_len = x.size(-2)
        
        # Project queries, keys, and values
        q = self.q_proj(x)  # (..., seq_len, d_model)
        k = self.k_proj(x)  # (..., seq_len, d_model)
        v = self.v_proj(x)  # (..., seq_len, d_model)
        
        # Reshape to split heads
        # (..., seq_len, num_heads, head_dim)
        q = q.view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*x.shape[:-1], self.num_heads, self.head_dim)
        v = v.view(*x.shape[:-1], self.num_heads, self.head_dim)
        
        # Reshape to move head dim after batch dim (for easy parallel computation)
        # (..., num_heads, seq_len, head_dim)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        
        # Apply RoPE to queries and keys if specified
        if use_rope and rope is not None:
            # Apply RoPE to each head separately
            q = rope(q, token_positions)
            k = rope(k, token_positions)
            
        # Create causal mask (lower triangular)
        # shape: (seq_len, seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = ~causal_mask  # True means attend, False means mask out
        
        # Expand mask for broadcasting with attention scores
        # shape: (1, 1, seq_len, seq_len) - will broadcast to batch and head dimensions
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply scaled dot-product attention with mask
        # attn_output shape: (..., num_heads, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(q, k, v, causal_mask)
        
        # Reshape back to original shape
        # First, transpose back: (..., seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(-3, -2)
        
        # Combine heads: (..., seq_len, d_model)
        attn_output = attn_output.reshape(*x.shape[:-1], self.d_model)
        
        # Final projection
        output = self.output_proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head self-attention and feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 max_seq_len: int = 1024, theta: float = 10000.0,
                 device=None, dtype=None):
        """
        Initialize a Transformer block.
        
        Args:
            d_model: Dimensionality of the model input/output
            num_heads: Number of attention heads
            d_ff: Dimensionality of feed-forward network inner layer
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE parameter θ
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        # Normalization layers
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        # Position-wise feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
        # RoPE for positional encoding
        head_dim = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=head_dim,
            max_seq_len=max_seq_len,
            device=device
        )
        
    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional tensor of token positions for RoPE
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Generate default positions if not provided
        if token_positions is None:
            seq_len = x.size(1)
            # Create positions tensor with shape (batch_size, seq_len)
            token_positions = torch.arange(seq_len, device=x.device).expand(x.size(0), -1)
            
        # First sub-layer: Multi-head self-attention with pre-norm
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, use_rope=True, rope=self.rope, token_positions=token_positions)
        x = x + attn_out  # Residual connection
        
        # Second sub-layer: Feed-forward network with pre-norm
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual connection
        
        return x

class TransformerLM(nn.Module):
    """
    Full Transformer Language Model with token embeddings, 
    multiple transformer blocks, and a language model head.
    
    Decoder only architecture.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        """
        Initialize a Transformer LM.
        
        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length
            d_model: Hidden dimension of the model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            d_ff: Dimensionality of feed-forward network inner layer
            theta: RoPE parameter θ
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        # Token embedding layer
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Language model head (prediction for next token)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
        # Store model dimensions
        self.d_model = d_model
        self.context_length = context_length
        self.vocab_size = vocab_size
        
    def forward(self, token_ids):
        """
        Forward pass through the Transformer language model.
        
        Args:
            token_ids: Integer tensor of token ids, shape (batch_size, seq_len) or (seq_len,)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # 确保输入与模型在同一设备上
        device = self.token_embeddings.weight.device
        token_ids = token_ids.to(device)
        
        # Ensure token_ids is 2D
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # Ensure batch dimension
        
        # Get batch size and sequence length
        batch_size, seq_len = token_ids.size()
        
        # Token embeddings & RoPE
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        
        # Generate positions tensor for RoPE
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Pass through each transformer block
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Predict next token with lm_head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits