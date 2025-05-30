"""
FlashAttention-2 implementation in PyTorch and Triton.

This module implements the FlashAttention-2 algorithm for efficient attention computation
with reduced memory usage through tiling and online softmax computation.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _flash_attention_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    """
    FlashAttention-2 backward pass using PyTorch operations with recomputation.
    Follows Equations 13-19 from the FlashAttention-2 paper.
    
    Args:
        Q: (batch, seq_len, head_dim) Query tensor
        K: (batch, seq_len, head_dim) Key tensor
        V: (batch, seq_len, head_dim) Value tensor
        O: (batch, seq_len, head_dim) Output from forward pass
        dO: (batch, seq_len, head_dim) Gradient w.r.t. output
        L: (batch, seq_len) Logsumexp from forward pass
        is_causal: whether causal masking was used
        
    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
    batch_size, seq_len, head_dim = Q.shape
    device = Q.device
    dtype = Q.dtype
    
    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Initialize gradients
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    
    # Compute D vector (Equation 15)
    # D_i = rowsum(dO_i ⊙ O_i)
    D = torch.sum(dO * O, dim=-1)  # (batch, seq_len)
    
    # Tile sizes (use same as forward pass)
    Q_TILE_SIZE = min(64, seq_len)
    K_TILE_SIZE = min(64, seq_len)
    Q_TILE_SIZE = max(16, Q_TILE_SIZE)
    K_TILE_SIZE = max(16, K_TILE_SIZE)
    
    # Number of tiles
    Tq = math.ceil(seq_len / Q_TILE_SIZE)
    Tk = math.ceil(seq_len / K_TILE_SIZE)
    
    # Recompute attention and gradients tile by tile
    for i in range(Tq):
        # Query tile boundaries
        q_start = i * Q_TILE_SIZE
        q_end = min((i + 1) * Q_TILE_SIZE, seq_len)
        
        # Load query tile and related tensors
        Qi = Q[:, q_start:q_end, :]  # (batch, q_size, head_dim)
        dOi = dO[:, q_start:q_end, :]  # (batch, q_size, head_dim)
        Oi = O[:, q_start:q_end, :]  # (batch, q_size, head_dim)
        Li = L[:, q_start:q_end]  # (batch, q_size)
        Di = D[:, q_start:q_end]  # (batch, q_size)
        
        # Initialize gradients for this query tile
        dQi = torch.zeros_like(Qi)
        
        for j in range(Tk):
            # Key tile boundaries
            k_start = j * K_TILE_SIZE
            k_end = min((j + 1) * K_TILE_SIZE, seq_len)
            
            # Load key and value tiles
            Kj = K[:, k_start:k_end, :]  # (batch, k_size, head_dim)
            Vj = V[:, k_start:k_end, :]  # (batch, k_size, head_dim)
            
            # Recompute attention scores for this tile
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # (batch, q_size, k_size)
            
            # Apply causal mask if needed (same as forward pass)
            if is_causal:
                q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                causal_mask = q_indices >= k_indices
                Sij = torch.where(causal_mask, Sij, torch.tensor(float('-inf'), device=device))
            
            # Recompute softmax probabilities (Equation 16)
            # P_ij = exp(S_ij - L_i)
            Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (batch, q_size, k_size)
            
            # Compute dV for this tile (Equation 17)
            # dV_j += P_ij^T @ dO_i
            dVj = torch.matmul(Pij.transpose(-2, -1), dOi)  # (batch, k_size, head_dim)
            dV[:, k_start:k_end, :] += dVj
            
            # Compute dP (Equation 18)
            # dP_ij = dO_i @ V_j^T
            dPij = torch.matmul(dOi, Vj.transpose(-2, -1))  # (batch, q_size, k_size)
            
            # Compute dS (Equation 19)
            # dS_ij = P_ij ⊙ (dP_ij - D_i)
            dSij = Pij * (dPij - Di.unsqueeze(-1))  # (batch, q_size, k_size)
            
            # Apply causal mask to dS if needed
            if is_causal:
                dSij = torch.where(causal_mask, dSij, torch.zeros_like(dSij))
            
            # Compute dQ for this tile (Equation 13)
            # dQ_i += dS_ij @ K_j * scale
            dQi += torch.matmul(dSij, Kj) * scale
            
            # Compute dK for this tile (Equation 14)
            # dK_j += dS_ij^T @ Q_i * scale
            dKj = torch.matmul(dSij.transpose(-2, -1), Qi) * scale
            dK[:, k_start:k_end, :] += dKj
        
        # Store dQ for this query tile
        dQ[:, q_start:q_end, :] = dQi
    
    return dQ, dK, dV


# Compile the backward function for better performance
# Check if we're on macOS and disable compile if there are issues
import platform
if platform.system() == 'Darwin':  # macOS
    # On macOS, torch.compile may have issues, so use regular function
    _flash_attention_backward_compiled = _flash_attention_backward_pytorch
else:
    # On other platforms, use compiled version
    try:
        _flash_attention_backward_compiled = torch.compile(_flash_attention_backward_pytorch)
    except Exception:
        # Fallback to uncompiled if compilation fails
        _flash_attention_backward_compiled = _flash_attention_backward_pytorch


class FlashAttentionPyTorchFunction(torch.autograd.Function):
    """
    Pure PyTorch implementation of FlashAttention-2 forward pass
    following Algorithm 1 from the paper.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass using pure PyTorch operations.
        
        Args:
            Q: (batch, seq_len, head_dim) Query tensor
            K: (batch, seq_len, head_dim) Key tensor  
            V: (batch, seq_len, head_dim) Value tensor
            is_causal: whether to apply causal masking
            
        Returns:
            O: output tensor with same shape as Q
        """
        batch_size, seq_len, head_dim = Q.shape
        device = Q.device
        dtype = Q.dtype
        
        # Scale factor: 1/sqrt(d)
        scale = 1.0 / math.sqrt(head_dim)
        
        # Tile sizes (at least 16x16 as required)
        Q_TILE_SIZE = min(64, seq_len)  # Bq
        K_TILE_SIZE = min(64, seq_len)  # Bk
        
        # Ensure tile sizes are at least 16
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)
        
        # Number of tiles
        Tq = math.ceil(seq_len / Q_TILE_SIZE)
        Tk = math.ceil(seq_len / K_TILE_SIZE)
        
        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Process each query tile
        for i in range(Tq):
            # Query tile boundaries
            q_start = i * Q_TILE_SIZE
            q_end = min((i + 1) * Q_TILE_SIZE, seq_len)
            q_size = q_end - q_start
            
            # Load query tile
            Qi = Q[:, q_start:q_end, :]  # (batch, q_size, head_dim)
            
            # Initialize tile outputs (Algorithm 1 notation)
            Oi = torch.zeros(batch_size, q_size, head_dim, 
                           device=device, dtype=torch.float32)
            li = torch.zeros(batch_size, q_size, 
                           device=device, dtype=torch.float32)
            mi = torch.full((batch_size, q_size), 
                          float('-inf'), device=device, dtype=torch.float32)
            
            # Process each key tile  
            for j in range(Tk):
                # Key tile boundaries
                k_start = j * K_TILE_SIZE
                k_end = min((j + 1) * K_TILE_SIZE, seq_len)
                k_size = k_end - k_start
                
                # Load key and value tiles
                Kj = K[:, k_start:k_end, :]  # (batch, k_size, head_dim)
                Vj = V[:, k_start:k_end, :]  # (batch, k_size, head_dim)
                
                # Compute attention scores for this tile
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # (batch, q_size, k_size)
                
                # Apply causal mask if needed
                if is_causal:
                    # Create causal mask for this tile
                    q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                    causal_mask = q_indices >= k_indices
                    Sij = torch.where(causal_mask, Sij, torch.tensor(float('-inf'), device=device))
                
                # Online softmax update (Algorithm 1)
                # Compute new maximum
                mij = torch.max(Sij, dim=-1, keepdim=False)[0]  # (batch, q_size)
                mi_new = torch.maximum(mi, mij)
                
                # Compute exponentials with numerical stability
                Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))  # (batch, q_size, k_size)
                
                # Update normalization term
                li_new = torch.exp(mi - mi_new) * li + torch.sum(Pij_tilde, dim=-1)
                
                # Update output (reweight previous output and add new contribution)
                Oi = (torch.exp(mi - mi_new).unsqueeze(-1) * Oi + 
                     torch.matmul(Pij_tilde, Vj))
                
                # Update state
                mi = mi_new
                li = li_new
            
            # Final normalization for this query tile
            Oi = Oi / li.unsqueeze(-1)
            Li = mi + torch.log(li)
            
            # Store results
            O[:, q_start:q_end, :] = Oi.to(dtype)
            L[:, q_start:q_end] = Li
        
        # Save for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using PyTorch + torch.compile"""
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        # Use compiled backward function for better performance
        dQ, dK, dV = _flash_attention_backward_compiled(Q, K, V, O, grad_output, L, is_causal)
        
        return dQ, dK, dV, None  # None for is_causal gradient


if TRITON_AVAILABLE:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Setup K and V block pointers (will be advanced in loop)
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # Output block pointers
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load query tile (only once)
        q = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)

        # Initialize accumulator buffers (use float32 for numerical stability)
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

        # Compute number of key tiles
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        # Loop over key tiles (single loop as required)
        for j in range(num_k_tiles):
            # Current key tile start position
            k_start = j * K_TILE_SIZE
            
            # Load current K and V tiles with boundary check
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Compute attention scores: S_ij = Q_i @ K_j^T * scale
            s_ij = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

            # Apply causal mask if needed
            if is_causal:
                # Create index vectors for causal masking
                q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                # Broadcast to create mask matrix: q_idx >= k_idx for causal
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                s_ij = tl.where(causal_mask, s_ij, -1e6)

            # Online softmax computation (Algorithm 1)
            # Compute row-wise maximum: m_ij = rowmax(S_ij)
            m_ij = tl.max(s_ij, axis=1)  # (Q_TILE_SIZE,)
            
            # Update global maximum: m_new = max(m_i, m_ij)
            m_new = tl.maximum(m_i, m_ij)

            # Compute scaled exponentials: P~_ij = exp(S_ij - m_new)
            p_ij_tilde = tl.exp(s_ij - m_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

            # Update normalization term: l_new = exp(m_i - m_new) * l_i + rowsum(P~_ij)
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p_ij_tilde, axis=1)

            # Update accumulator with reweighting (Algorithm 1)
            # O_new = diag(exp(m_i - m_new)) * O_i + P~_ij @ V_j
            alpha = tl.exp(m_i - m_new)  # Reweighting factor
            
            # Cast P~_ij to V's dtype before multiplication (as suggested)
            p_ij_for_matmul = p_ij_tilde.to(v.dtype)
            
            # Accumulate with proper reweighting
            acc = acc * alpha[:, None] + tl.dot(p_ij_for_matmul, v, acc=tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32))

            # Update state variables
            m_i = m_new
            l_i = l_new

            # Advance K and V block pointers to next tile
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Final normalization: O_i = diag(l_i)^{-1} * O_i
        acc = acc / l_i[:, None]

        # Compute logsumexp: L_i = m_i + log(l_i)
        l_final = m_i + tl.log(l_i)

        # Store results (cast to appropriate dtype)
        tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, l_final, boundary_check=(0,))


    @triton.jit
    def flash_bwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_dqb, stride_dqq, stride_dqd,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """
        FlashAttention-2 Triton backward kernel following Algorithm 2.
        
        Key insight: We iterate over K/V tiles in the outer loop and Q tiles 
        in the inner loop to enable efficient accumulation of dK and dV without
        requiring synchronization across thread blocks.
        """
        # Program indices - each program handles one key tile
        key_tile_index = tl.program_id(0)  # j in Algorithm 2
        batch_index = tl.program_id(1)
        
        # Calculate key tile boundaries
        k_start = key_tile_index * K_TILE_SIZE
        k_end = tl.minimum(k_start + K_TILE_SIZE, N_KEYS)
        k_size = k_end - k_start
        
        # Setup K and V block pointers for current tile
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # Load K and V tiles (once per key tile)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Initialize accumulators for dK and dV (Algorithm 2)
        dk_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        
        # Compute number of query tiles
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
        
        # Inner loop: iterate over all query tiles (i in Algorithm 2)
        for q_tile_idx in range(num_q_tiles):
            q_start = q_tile_idx * Q_TILE_SIZE
            q_end = tl.minimum(q_start + Q_TILE_SIZE, N_QUERIES)
            
            # Setup Q, O, dO block pointers for current query tile
            Q_block_ptr = tl.make_block_ptr(
                Q_ptr + batch_index * stride_qb,
                shape=(N_QUERIES, D),
                strides=(stride_qq, stride_qd),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            O_block_ptr = tl.make_block_ptr(
                O_ptr + batch_index * stride_ob,
                shape=(N_QUERIES, D),
                strides=(stride_oq, stride_od),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            dO_block_ptr = tl.make_block_ptr(
                dO_ptr + batch_index * stride_dob,
                shape=(N_QUERIES, D),
                strides=(stride_doq, stride_dod),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            L_block_ptr = tl.make_block_ptr(
                L_ptr + batch_index * stride_lb,
                shape=(N_QUERIES,),
                strides=(stride_lq,),
                offsets=(q_start,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )
            
            D_block_ptr = tl.make_block_ptr(
                D_ptr + batch_index * stride_db,
                shape=(N_QUERIES,),
                strides=(stride_dq,),
                offsets=(q_start,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )
            
            dQ_block_ptr = tl.make_block_ptr(
                dQ_ptr + batch_index * stride_dqb,
                shape=(N_QUERIES, D),
                strides=(stride_dqq, stride_dqd),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            
            # Load current query tile data
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
            do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
            
            # Compute attention scores S_ij = Q_i @ K_j^T / sqrt(d)
            s_ij = tl.dot(q, tl.trans(k)) * scale
            
            # Apply causal mask if needed
            if is_causal:
                q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                s_ij = tl.where(causal_mask, s_ij, -1e6)
            
            # Compute attention probabilities P_ij = exp(S_ij - L_i)
            p_ij = tl.exp(s_ij - l[:, None])
            
            # Compute dV: dV_j += P_ij^T @ dO_i
            dv_contribution = tl.dot(tl.trans(p_ij), do, acc=tl.zeros((K_TILE_SIZE, D), dtype=tl.float32))
            dv_acc += dv_contribution
            
            # Compute dP: dP_ij = dO_i @ V_j^T
            dp_ij = tl.dot(do, tl.trans(v))
            
            # Compute dS: dS_ij = P_ij ⊙ (dP_ij - D_i) / sqrt(d)
            ds_ij = p_ij * (dp_ij - d[:, None]) * scale
            
            # Apply causal mask to dS if needed
            if is_causal:
                ds_ij = tl.where(causal_mask, ds_ij, 0.0)
            
            # Compute dQ: dQ_i += dS_ij @ K_j (accumulate atomically)
            dq_contribution = tl.dot(ds_ij, k, acc=tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32))
            
            # Load existing dQ, add contribution, and store back
            dq_existing = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero")
            dq_new = dq_existing + dq_contribution.to(dq_existing.dtype)
            tl.store(dQ_block_ptr, dq_new, boundary_check=(0, 1))
            
            # Compute dK: dK_j += dS_ij^T @ Q_i
            dk_contribution = tl.dot(tl.trans(ds_ij), q, acc=tl.zeros((K_TILE_SIZE, D), dtype=tl.float32))
            dk_acc += dk_contribution
        
        # Write accumulated dK and dV to global memory
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # Cast and store final results
        tl.store(dK_block_ptr, dk_acc.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_block_ptr, dv_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

    class FlashAttentionTritonFunction(torch.autograd.Function):
        """
        Triton implementation of FlashAttention-2 forward pass.
        """
        
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            """
            FlashAttention-2 forward pass using Triton kernel.
            
            Args:
                Q: (batch, seq_len, head_dim) Query tensor
                K: (batch, seq_len, head_dim) Key tensor  
                V: (batch, seq_len, head_dim) Value tensor
                is_causal: whether to apply causal masking
                
            Returns:
                O: output tensor with same shape as Q
            """
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Ensure tensors are contiguous and on CUDA
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            
            # Scale factor: 1/sqrt(d)
            scale = 1.0 / math.sqrt(head_dim)
            
            # Tile sizes - ensure they're powers of 2 and at least 16
            Q_TILE_SIZE = min(128, seq_len)
            K_TILE_SIZE = min(128, seq_len)
            
            # Ensure minimum tile size of 16
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            
            # Round down to nearest power of 2 for better performance
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Initialize output tensors
            O = torch.empty_like(Q)
            L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Launch grid: (T_q, batch_size) as specified
            num_q_tiles = triton.cdiv(seq_len, Q_TILE_SIZE)
            grid = (num_q_tiles, batch_size)
            
            # Launch kernel with proper configuration
            flash_fwd_kernel[grid](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                seq_len, seq_len,
                scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            # Save for backward pass
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            ctx.scale = scale
            
            return O
        
        @staticmethod
        def backward(ctx, grad_output):
            """Triton backward pass using Algorithm 2"""
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Ensure gradients are contiguous
            grad_output = grad_output.contiguous()
            
            # Compute D vector: D = rowsum(dO ⊙ O)
            D = torch.sum(grad_output * O, dim=-1)  # (batch, seq_len)
            
            # Initialize gradient tensors
            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K)
            dV = torch.zeros_like(V)
            
            # Tile sizes (same as forward pass)
            Q_TILE_SIZE = min(128, seq_len)
            K_TILE_SIZE = min(128, seq_len)
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Launch grid: (T_k, batch_size) - iterate over key tiles
            num_k_tiles = triton.cdiv(seq_len, K_TILE_SIZE)
            grid = (num_k_tiles, batch_size)
            
            # Launch backward kernel
            flash_bwd_kernel[grid](
                Q, K, V, O, grad_output, L, D,
                dQ, dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
                L.stride(0), L.stride(1),
                D.stride(0), D.stride(1),
                dQ.stride(0), dQ.stride(1), dQ.stride(2),
                dK.stride(0), dK.stride(1), dK.stride(2),
                dV.stride(0), dV.stride(1), dV.stride(2),
                seq_len, seq_len,
                scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            return dQ, dK, dV, None  # None for is_causal gradient

else:
    # Dummy Triton function for when Triton is not available
    class FlashAttentionTritonFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError(
                "Triton is not available on this platform. "
                "Triton requires CUDA support. Please run on a system with CUDA GPU, "
                "or use the PyTorch implementation instead."
            )
        
        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError("Backward pass not implemented yet")


def flash_attention_pytorch(Q, K, V, is_causal=False):
    """Convenience function for PyTorch FlashAttention-2."""
    return FlashAttentionPyTorchFunction.apply(Q, K, V, is_causal)


def flash_attention_triton(Q, K, V, is_causal=False):
    """Convenience function for Triton FlashAttention-2."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return FlashAttentionTritonFunction.apply(Q, K, V, is_causal)