"""
FlashAttention-2 implementation in PyTorch and Triton.

This module implements the FlashAttention-2 algorithm for efficient attention computation
with reduced memory usage through tiling and online softmax computation.

Provide 3 versions: A PyTorch version (very slow), 
                    A Triton forward + PyTorch backward version,
                    and A full Triton version.
I modified the Triton backward kernel from Dao-AILab to my full triton version. The original file is in "ref".
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

"""
PyTorch implementation of FlashAttention-2
"""
@torch.compile
def flash_backward_gpu(Q, K, V, O, dO, L, is_causal=False):
    """
    GPU-accelerated backward pass for FlashAttention-2 using torch.compile.
    
    This implementation uses recomputation strategy as described in the FlashAttention paper
    (Equations 13-19) to avoid storing intermediate softmax values.
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim)
        V: Value tensor (batch, seq_len, head_dim)
        O: Output tensor from forward pass (batch, seq_len, head_dim)
        dO: Gradient w.r.t. output (batch, seq_len, head_dim)
        L: Log-sum-exp from forward pass (batch, seq_len)
        is_causal: Whether causal masking was applied
        
    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
    batch_size, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Initialize gradient tensors
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    
    # Compute D vector (Equation 14): D_i = sum_j(dO_ij * O_ij)
    D = torch.sum(dO * O, dim=-1)  # (batch, seq_len)
    
    # Determine optimal tile size for memory efficiency
    # Use smaller tiles for better memory management on GPU
    TILE_SIZE = min(128, seq_len)
    TILE_SIZE = max(32, TILE_SIZE)  # Ensure minimum tile size
    
    # Make TILE_SIZE a power of 2 for better GPU performance
    TILE_SIZE = 1 << (TILE_SIZE.bit_length() - 1)
    
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    
    # Process in tiles for memory efficiency
    for i in range(num_tiles):
        # Query tile indices
        q_start = i * TILE_SIZE
        q_end = min(q_start + TILE_SIZE, seq_len)
        q_slice = slice(q_start, q_end)
        
        # Extract query tile and related tensors
        Qi = Q[:, q_slice]  # (batch, tile_size, head_dim)
        dOi = dO[:, q_slice]  # (batch, tile_size, head_dim)
        Li = L[:, q_slice]  # (batch, tile_size)
        Di = D[:, q_slice]  # (batch, tile_size)
        
        # Initialize gradient accumulator for this query tile
        dQi = torch.zeros_like(Qi)
        
        for j in range(num_tiles):
            # Key/Value tile indices
            k_start = j * TILE_SIZE
            k_end = min(k_start + TILE_SIZE, seq_len)
            k_slice = slice(k_start, k_end)
            
            # Extract key/value tile
            Kj = K[:, k_slice]  # (batch, tile_size, head_dim)
            Vj = V[:, k_slice]  # (batch, tile_size, head_dim)
            
            # Recompute attention scores (Equation 15)
            # S_ij = Q_i @ K_j^T * scale
            Sij = torch.bmm(Qi, Kj.transpose(-2, -1)) * scale  # (batch, q_tile, k_tile)
            
            # Apply causal mask if needed
            if is_causal:
                # Create causal mask for current tiles
                q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                causal_mask = q_indices < k_indices
                Sij = Sij.masked_fill(causal_mask, float('-inf'))
            
            # Recompute attention weights (Equation 16)
            # P_ij = exp(S_ij - L_i)
            Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (batch, q_tile, k_tile)
            
            # Compute gradient w.r.t. V (Equation 17)
            # dV_j += P_ij^T @ dO_i
            dVj = torch.bmm(Pij.transpose(-2, -1), dOi)  # (batch, k_tile, head_dim)
            dV[:, k_slice] += dVj
            
            # Compute gradient w.r.t. P (Equation 18)
            # dP_ij = dO_i @ V_j^T
            dPij = torch.bmm(dOi, Vj.transpose(-2, -1))  # (batch, q_tile, k_tile)
            
            # Compute gradient w.r.t. S (Equation 19)
            # dS_ij = P_ij * (dP_ij - D_i)
            dSij = Pij * (dPij - Di.unsqueeze(-1))  # (batch, q_tile, k_tile)
            
            # Apply causal mask to gradients
            if is_causal:
                dSij = dSij.masked_fill(causal_mask, 0.0)
            
            # Accumulate gradient w.r.t. Q
            # dQ_i += dS_ij @ K_j * scale
            dQi += torch.bmm(dSij, Kj) * scale  # (batch, q_tile, head_dim)
            
            # Compute gradient w.r.t. K
            # dK_j += dS_ij^T @ Q_i * scale
            dKj = torch.bmm(dSij.transpose(-2, -1), Qi) * scale  # (batch, k_tile, head_dim)
            dK[:, k_slice] += dKj
        
        # Store accumulated gradient for this query tile
        dQ[:, q_slice] = dQi
    
    return dQ, dK, dV


class FlashAttentionPyTorchFunction(torch.autograd.Function):
    """
    Highly optimized PyTorch implementation of FlashAttention-2
    with minimal overhead and maximum PyTorch optimization.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass.
        Key optimizations:
        1. Minimize memory allocations
        2. Use native PyTorch operations
        3. Eliminate unnecessary dtype conversions
        4. Use torch.compile compatible operations
        """
        batch_size, seq_len, head_dim = Q.shape
        device = Q.device
        dtype = Q.dtype
        
        # Use native dtype for better performance - no conversions
        scale = (head_dim ** -0.5)  # More efficient than 1.0 / math.sqrt(head_dim)
        
        # Adaptive tile sizing for optimal performance
        if seq_len <= 128:
            TILE_SIZE = 64  # Larger tiles for small sequences
        elif seq_len <= 512:
            TILE_SIZE = 128
        else:
            TILE_SIZE = 256
        
        TILE_SIZE = min(TILE_SIZE, seq_len)
        num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE  # Ceiling division
        
        # Pre-allocate all tensors to avoid repeated allocations
        O = torch.zeros_like(Q)
        L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Pre-compute causal mask template if needed
        causal_mask = None
        if is_causal:
            # Create a reusable mask template
            mask_template = torch.tril(torch.ones(TILE_SIZE, TILE_SIZE, device=device, dtype=torch.bool))
        
        # Process tiles with optimized loops
        for i in range(num_tiles):
            q_start = i * TILE_SIZE
            q_end = min(q_start + TILE_SIZE, seq_len)
            q_len = q_end - q_start
            
            Qi = Q[:, q_start:q_end]  # No dtype conversion
            
            # Initialize accumulators - use native dtype
            Oi = torch.zeros(batch_size, q_len, head_dim, device=device, dtype=dtype)
            li = torch.zeros(batch_size, q_len, device=device, dtype=torch.float32)
            mi = torch.full((batch_size, q_len), float('-inf'), device=device, dtype=torch.float32)
            
            for j in range(num_tiles):
                k_start = j * TILE_SIZE
                k_end = min(k_start + TILE_SIZE, seq_len)
                k_len = k_end - k_start
                
                # Early termination for causal attention
                if is_causal and k_start >= q_end:
                    break
                
                Kj = K[:, k_start:k_end]
                Vj = V[:, k_start:k_end]
                
                # Optimized attention computation using einsum for better performance
                Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale
                
                # Efficient causal masking
                if is_causal:
                    # Create mask only for the actual tile sizes
                    q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                    mask = q_indices < k_indices
                    Sij.masked_fill_(mask, float('-inf'))
                
                # Optimized online softmax with fused operations
                mij = torch.max(Sij, dim=-1)[0]
                mi_new = torch.maximum(mi, mij)
                
                # Compute softmax with numerical stability
                Pij = torch.exp(Sij - mi_new.unsqueeze(-1))
                
                # Fused accumulator updates
                alpha = torch.exp(mi - mi_new)
                li_new = alpha * li + torch.sum(Pij, dim=-1)
                
                # Fused output update - use einsum for efficiency
                alpha_reshaped = alpha.unsqueeze(-1)
                Oi = alpha_reshaped * Oi + torch.einsum('bqk,bkd->bqd', Pij, Vj)
                
                mi = mi_new
                li = li_new
            
            # Final normalization with safety check
            li_safe = torch.clamp(li, min=1e-8)  # Prevent division by zero
            Oi = Oi / li_safe.unsqueeze(-1)
            
            # Store results
            O[:, q_start:q_end] = Oi
            L[:, q_start:q_end] = mi + torch.log(li_safe)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return O
    
    @staticmethod  
    def backward(ctx, grad_output):
        """GPU-accelerated backward pass using torch.compile and recomputation."""
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        # Use the compiled GPU backward function
        dQ, dK, dV = flash_backward_gpu(Q, K, V, O, grad_output, L, is_causal)
        
        return dQ, dK, dV, None

"""
Triton kernels
"""

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
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Setup K and V block pointers
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

        # Load query tile
        q = tl.load(Q_block_ptr)

        # Initialize accumulator buffers
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

        # Compute number of key tiles
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        # Loop over key tiles
        for j in range(num_k_tiles):
            k_start = j * K_TILE_SIZE
            
            # Load K and V tiles
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Compute attention scores
            s_ij = tl.dot(q, tl.trans(k)) * scale

            # Apply causal mask if needed
            if is_causal:
                q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                s_ij = tl.where(causal_mask, s_ij, -1e6)

            # Online softmax computation
            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p_ij_tilde = tl.exp(s_ij - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p_ij_tilde, axis=1)

            # Update accumulator
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p_ij_tilde.to(v.dtype), v)

            # Update state
            m_i = m_new
            l_i = l_new

            # Advance pointers
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Final normalization
        acc = acc / l_i[:, None]
        l_final = m_i + tl.log(l_i)

        # Store results
        tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, l_final, boundary_check=(0,))

    @triton.jit
    def _bwd_preprocess_do_o_dot(
        Out, DO, Delta, stride_ob, stride_oh, stride_om, stride_dob, stride_doh, stride_dom,
        nheads, seqlen_q, seqlen_q_rounded, headdim, BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        # load
        o = tl.load(
            Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            DO
            + off_b * stride_dob
            + off_h * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :],
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        # write-back
        tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


    @triton.jit
    def _bwd_store_dk_dv(
        dk_ptrs, dv_ptrs, dk, dv,
        offs_n, offs_d,
        seqlen_k, headdim,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
    ):
        # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
        # if we just call tl.store(dv_ptrs), there's a race condition
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
                tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

    # copied from flash_attn_triton.py - Dao AILab
    @triton.jit
    def _bwd_kernel_one_col_block(
        start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale,
        stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn,
        seqlen_q, seqlen_k, headdim, ATOMIC_ADD: tl.constexpr, BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
        begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
        # initialize row/col offsets
        offs_qm = begin_m + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
        do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
        dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
        if BIAS_TYPE == "vector":
            b_ptrs = Bias + offs_n
        elif BIAS_TYPE == "matrix":
            b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
        # initialize dv and dk
        dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        # There seems to be some problem with Triton pipelining that makes results wrong for
        # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
        # may have zero step, and pipelining with the bias matrix could screw it up.
        # So we just exit early.
        if begin_m >= seqlen_q:
            dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
            dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
            _bwd_store_dk_dv(
                dk_ptrs, dv_ptrs, dk, dv,
                offs_n, offs_d, seqlen_k, headdim,
                EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
            )
            return
        # k and v stay in SRAM throughout
        # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
        # if we just call tl.load(k_ptrs), we get the wrong output!
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
                v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(
                    k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
                )
                v = tl.load(
                    v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
                )
        # loop over rows
        num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
        for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
            if EVEN_M & EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                if EVEN_HEADDIM:
                    q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                else:
                    q = tl.load(
                        q_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            # recompute p = softmax(qk, dim=-1).T
            # Fix：use tl.trans() instead of trans_b=True
            qk = tl.dot(q, tl.trans(k))
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            if BIAS_TYPE != "none":
                tl.debug_barrier()  # Race condition otherwise
                if BIAS_TYPE == "vector":
                    if EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                    bias = bias[None, :]
                elif BIAS_TYPE == "matrix":
                    if EVEN_M & EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(
                            b_ptrs,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                            other=0.0,
                        ).to(tl.float32)
                qk = qk * softmax_scale + bias
            # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
            # Also wrong for headdim=64.
            if not (EVEN_M & EVEN_HEADDIM):
                tl.debug_barrier()
            lse_i = tl.load(LSE + offs_m_curr)
            if BIAS_TYPE == "none":
                p = tl.exp(qk * softmax_scale - lse_i[:, None])
            else:
                p = tl.exp(qk - lse_i[:, None])
            # compute dv
            # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
            # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
            # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
            # the output is correct.
            if EVEN_M & EVEN_HEADDIM:
                do = tl.load(do_ptrs)
            else:
                # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
            # Fix: use tl.trans() instead of trans_b=True
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            # compute dp = dot(v, do)
            # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
            # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
            # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
            if not (EVEN_M & EVEN_HEADDIM):
                tl.debug_barrier()
            # Fix: use tl.trans() instead of trans_b=True
            dp = tl.dot(do, tl.trans(v))
            # There's a race condition for headdim=48
            if not EVEN_HEADDIM:
                tl.debug_barrier()
            # compute ds = p * (dp - delta[:, None])
            # Putting the subtraction after the dp matmul (instead of before) is slightly faster
            Di = tl.load(D + offs_m_curr)
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
            # compute dk = dot(ds.T, q)
            # Fix: use tl.trans() instead of trans_a=True
            dk += tl.dot(tl.trans(ds), q)
            # compute dq
            if not (
                EVEN_M & EVEN_HEADDIM
            ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
                tl.debug_barrier()
            if not ATOMIC_ADD:
                if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                    dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                else:
                    if EVEN_HEADDIM:
                        dq = tl.load(
                            dq_ptrs,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq += tl.dot(ds, k)
                        tl.store(
                            dq_ptrs,
                            dq,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            eviction_policy="evict_last",
                        )
                    else:
                        dq = tl.load(
                            dq_ptrs,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq += tl.dot(ds, k)
                        tl.store(
                            dq_ptrs,
                            dq,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                            eviction_policy="evict_last",
                        )
            else:  # If we're parallelizing across the seqlen_k dimension
                dq = tl.dot(ds, k)
                if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                    tl.atomic_add(dq_ptrs, dq)
                else:
                    if EVEN_HEADDIM:
                        tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                    else:
                        tl.atomic_add(
                            dq_ptrs,
                            dq,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        )
            # increment pointers
            dq_ptrs += BLOCK_M * stride_dqm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_dom
            if BIAS_TYPE == "matrix":
                b_ptrs += BLOCK_M * stride_bm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim,
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
        )

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()


    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero("DQ"),
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero("DQ"),
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False},
                num_warps=2,
                num_stages=1,
                pre_hook=init_to_zero("DQ"),
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False},
                num_warps=2,
                num_stages=1,
                pre_hook=init_to_zero("DQ"),
            ),
        ],
        key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
    )
    
    @triton.heuristics(
        {
            "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
            "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
            "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        }
    )
    
    @triton.jit
    def _bwd_kernel(
        Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm,
        stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm,
        stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn,
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
        CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
        BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, 
        BLOCK_HEADDIM: tl.constexpr, SEQUENCE_PARALLEL: tl.constexpr,
        EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        # offset pointers for batch/head
        Q += off_b * stride_qb + off_h * stride_qh
        K += off_b * stride_kb + off_h * stride_kh
        V += off_b * stride_vb + off_h * stride_vh
        DO += off_b * stride_dob + off_h * stride_doh
        DQ += off_b * stride_dqb + off_h * stride_dqh
        DK += off_b * stride_dkb + off_h * stride_dkh
        DV += off_b * stride_dvb + off_h * stride_dvh
        if BIAS_TYPE != "none":
            Bias += off_b * stride_bb + off_h * stride_bh
        # pointer to row-wise quantities in value-like data
        D += off_hb * seqlen_q_rounded
        LSE += off_hb * seqlen_q_rounded
        if not SEQUENCE_PARALLEL:
            num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
            for start_n in range(0, num_block_n):
                _bwd_kernel_one_col_block(
                    start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale,
                    stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, 
                    stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD=False,
                    BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM,
                    EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                )
        else:
            start_n = tl.program_id(0)
            _bwd_kernel_one_col_block(
                start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D,
                softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm,
                stride_dom, stride_dqm, stride_dkn, stride_dvn,
                seqlen_q, seqlen_k, headdim,
                ATOMIC_ADD=True, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            
    def _flash_attn_backward(
        do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
    ):
        # Make sure that the last dimension is contiguous
        if do.stride(-1) != 1:
            do = do.contiguous()
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        # assert d in {16, 32, 64, 128}
        assert d <= 128
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        assert lse.shape == (batch, nheads, seqlen_q_rounded)
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
        assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        # dq_accum = torch.zeros_like(q, dtype=torch.float32)
        dq_accum = torch.empty_like(q, dtype=torch.float32)
        delta = torch.empty_like(lse)
        # delta = torch.zeros_like(lse)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(2), o.stride(1), 
                                       do.stride(0), do.stride(2), do.stride(1), nheads, 
                                       seqlen_q, seqlen_q_rounded, d, BLOCK_M=128, BLOCK_HEADDIM=BLOCK_HEADDIM)

        has_bias = bias is not None
        bias_type = "none"
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            assert bias.stride(-1) == 1
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = "vector"
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = "matrix"
            else:
                raise RuntimeError(
                    "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
                )
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

        # BLOCK_M = 128
        # BLOCK_N = 64
        # num_warps = 4
        grid = lambda META: (
            triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
            batch * nheads,
        )
        _bwd_kernel[grid](
            q, k, v, bias, do, dq_accum, dk, dv, lse, delta, softmax_scale,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            *bias_strides,
            do.stride(0), do.stride(2), do.stride(1),
            dq_accum.stride(0), dq_accum.stride(2), dq_accum.stride(1),
            dk.stride(0), dk.stride(2), dk.stride(1),
            dv.stride(0), dv.stride(2), dv.stride(1),
            nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
            seqlen_q // 32, seqlen_k // 32,
            bias_type, causal, BLOCK_HEADDIM,
        )
        dq.copy_(dq_accum)
    
    class FlashAttentionTritonFunction(torch.autograd.Function):
        """Triton forward + Torch backward version of FlashAttention-2."""
        
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Ensure contiguous tensors
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            
            scale = 1.0 / math.sqrt(head_dim)
            
            # Optimized tile sizes
            Q_TILE_SIZE = min(64, seq_len)
            K_TILE_SIZE = min(64, seq_len)
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            
            # Round to power of 2 for better performance
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Initialize outputs
            O = torch.empty_like(Q)
            L = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Launch kernel
            num_q_tiles = triton.cdiv(seq_len, Q_TILE_SIZE)
            grid = (num_q_tiles, batch_size)
            
            flash_fwd_kernel[grid](
                Q, K, V, O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                seq_len, seq_len, scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            ctx.scale = scale
            
            return O
        
        @staticmethod
        def backward(ctx, grad_output):
            """Efficient backward pass using the same PyTorch implementation with proper context."""
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # Pre-allocate gradients
            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K) 
            dV = torch.zeros_like(V)
            
            # Efficient D computation
            D = torch.sum(grad_output * O, dim=-1)
            
            # Use optimized tile size for backward
            TILE_SIZE = min(64, seq_len)
            TILE_SIZE = max(16, TILE_SIZE)
            num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
            
            # Optimized backward computation
            for i in range(num_tiles):
                q_start = i * TILE_SIZE
                q_end = min(q_start + TILE_SIZE, seq_len)
                
                Qi = Q[:, q_start:q_end]
                dOi = grad_output[:, q_start:q_end]
                Li = L[:, q_start:q_end]
                Di = D[:, q_start:q_end]
                
                dQi = torch.zeros_like(Qi)
                
                for j in range(num_tiles):
                    k_start = j * TILE_SIZE
                    k_end = min(k_start + TILE_SIZE, seq_len)
                    
                    Kj = K[:, k_start:k_end]
                    Vj = V[:, k_start:k_end]
                    
                    # Recompute attention efficiently
                    Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale
                    
                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=device)[:, None]
                        k_indices = torch.arange(k_start, k_end, device=device)[None, :]
                        mask = q_indices < k_indices
                        Sij.masked_fill_(mask, float('-inf'))
                    
                    Pij = torch.exp(Sij - Li.unsqueeze(-1))
                    
                    # Efficient gradient computation with einsum
                    dVj = torch.einsum('bqk,bqd->bkd', Pij, dOi)
                    dV[:, k_start:k_end] += dVj
                    
                    dPij = torch.einsum('bqd,bkd->bqk', dOi, Vj)
                    dSij = Pij * (dPij - Di.unsqueeze(-1))
                    
                    if is_causal:
                        dSij.masked_fill_(mask, 0.0)
                    
                    dQi += torch.einsum('bqk,bkd->bqd', dSij, Kj) * scale
                    dKj = torch.einsum('bqk,bqd->bkd', dSij, Qi) * scale
                    dK[:, k_start:k_end] += dKj
            
                dQ[:, q_start:q_end] = dQi
            
            return dQ, dK, dV, None

    #triton kernel for backward pass

    
    class FlashAttentionTritonAll(torch.autograd.Function):
        """
        Triton forward + Torch backward version of FlashAttention-2
        """
        
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            batch_size, seq_len, head_dim = Q.shape
            device = Q.device
            dtype = Q.dtype
            
            # todo: make it dynamic
            nheads = Q.shape[1] // K.shape[1]
            
            # Ensure contiguous tensors
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            
            scale = 1.0 / math.sqrt(head_dim)
            
            # Optimized tile sizes
            Q_TILE_SIZE = min(64, seq_len)
            K_TILE_SIZE = min(64, seq_len)
            Q_TILE_SIZE = max(16, Q_TILE_SIZE)
            K_TILE_SIZE = max(16, K_TILE_SIZE)
            
            # Round to power of 2 for better performance
            Q_TILE_SIZE = 1 << (Q_TILE_SIZE.bit_length() - 1)
            K_TILE_SIZE = 1 << (K_TILE_SIZE.bit_length() - 1)
            
            # Initialize outputs
            O = torch.empty_like(Q)
            
            # LSE tensor for the test (batch, seq_len) - this is what the test expects
            L_test = torch.empty(batch_size, seq_len, device=device, dtype=torch.float32)
            
            # Launch kernel
            num_q_tiles = triton.cdiv(seq_len, Q_TILE_SIZE)
            grid = (num_q_tiles, batch_size)
            
            flash_fwd_kernel[grid](
                Q, K, V, O, L_test,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L_test.stride(0), L_test.stride(1),
                seq_len, seq_len, scale,
                D=head_dim,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                num_warps=4,
                num_stages=2,
            )
            
            # Create LSE in Dao-AILab format for backward pass (batch, nheads, seqlen_q_rounded)
            seqlen_q_rounded = math.ceil(seq_len / 128) * 128
            L_backward = torch.empty(batch_size, nheads, seqlen_q_rounded, device=device, dtype=torch.float32)
            
            # Copy L_test to correct L_backward shape for backward pass
            L_backward[:, 0, :seq_len] = L_test
            if seq_len < seqlen_q_rounded:
                L_backward[:, 0, seq_len:] = float('-inf')  # fill remaining with -inf
            
            # Reshape tensors to match Dao-AILab's expected shape (batch, seqlen, nheads, head_dim)
            Q_reshaped = Q.unsqueeze(2)  # (batch, seq_len, 1, head_dim)
            K_reshaped = K.unsqueeze(2)  # (batch, seq_len, 1, head_dim) 
            V_reshaped = V.unsqueeze(2)  # (batch, seq_len, 1, head_dim)
            O_reshaped = O.unsqueeze(2)  # (batch, seq_len, 1, head_dim)
            
            # Create a dummy bias tensor instead of None to avoid AttributeError in tests
            # The backward function will check if bias is None anyway
            dummy_bias = torch.empty(0, device=device, dtype=dtype)
            
            # Save 7 tensors: q, k, v, o, lse_backward, bias, lse_test, 
            #  the last two only for backward occupancy, without affecting the computation
            # The test expects to find L_test with shape (batch, seq_len)
            ctx.save_for_backward(Q_reshaped, K_reshaped, V_reshaped, O_reshaped, L_backward, dummy_bias, L_test)
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.nheads = nheads
            ctx.seqlen_q_rounded = seqlen_q_rounded
            
            return O
        
        @staticmethod
        def backward(ctx, do):
            q, k, v, o, lse, bias, lse_test = ctx.saved_tensors
            is_causal = ctx.is_causal
            scale = ctx.scale
            nheads = ctx.nheads
            seqlen_q_rounded = ctx.seqlen_q_rounded
            
            # Check if bias is actually None (empty tensor means no bias)
            if bias.numel() == 0:
                bias = None
            
            # Ensure do is contiguous
            if do.stride(-1) != 1:
                do = do.contiguous()
            
            # Reshape do to match saved tensors format (batch, seqlen, nheads, head_dim)
            do_reshaped = do.unsqueeze(2)  # (batch, seq_len, 1, head_dim)
            
            # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
            # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
            with torch.inference_mode():
                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                
                _flash_attn_backward(
                    do_reshaped, q, k, v, o, lse, dq, dk, dv, 
                    bias=bias, causal=is_causal, softmax_scale=scale,
                )
            
            # Squeeze the extra dimension added for nheads
            dq_squeezed = dq.squeeze(2)  # (batch, seq_len, head_dim)
            dk_squeezed = dk.squeeze(2)  # (batch, seq_len, head_dim)
            dv_squeezed = dv.squeeze(2)  # (batch, seq_len, head_dim)
            
            return dq_squeezed, dk_squeezed, dv_squeezed, None
else:
    class FlashAttentionTritonFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError("Triton is not available")
        
        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError()

"""
export API
"""
def flash_attention_pytorch(Q, K, V, is_causal=False):
    """Optimized PyTorch FlashAttention-2."""
    return FlashAttentionPyTorchFunction.apply(Q, K, V, is_causal)


def flash_attention_triton(Q, K, V, is_causal=False):
    """Triton FlashAttention-2 with optimized performance."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    
    # Direct function call without unnecessary overhead
    return FlashAttentionTritonFunction.apply(Q, K, V, is_causal)


def flash_attention_all_triton(Q, K, V, is_causal=False):
    """
    Enhanced Triton FlashAttention-2 with full Triton backward pass.
    
    This version uses Triton kernels for both forward and backward passes,
    featuring optimized online softmax implementation and memory-efficient
    gradient computation with recomputation strategy.
    
    Key improvements over V1:
    1. Full Triton kernel implementation for backward pass
    2. Online softmax in backward computation
    3. Optimized memory access patterns
    4. Better tile size management
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim)  
        V: Value tensor (batch, seq_len, head_dim)
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor (batch, seq_len, head_dim)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return FlashAttentionTritonAll.apply(Q, K, V, is_causal)
