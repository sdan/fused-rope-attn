"""
Fused RoPE + Attention for VLM's vision encoders that use RoPE.

The idea: instead of writing Q_rot and K_rot to HBM then reading them back,
we apply RoPE in registers during the attention kernel.
"""
import torch
import triton
import triton.language as tl
from collections import OrderedDict


def next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


# 2D RoPE table computation (with caching)

_rope_cache: OrderedDict = OrderedDict()
_ROPE_CACHE_MAX = 32


def _device_key(device: torch.device) -> tuple:
    """Canonical device key for caching."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = device.index
    return (device.type, idx)


def compute_rope_2d(
    grid_thw,
    head_dim: int,
    spatial_merge_size: int,
    theta: float,
    device,
    dtype,
    window_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D RoPE tables for vision encoder.

    Args:
        grid_thw: list of (t, h, w) tuples, or tensor of shape [N, 3]
        head_dim: dimension per head (must be divisible by 4 for 2D RoPE)
        spatial_merge_size: Qwen3-VL uses 2 (merges 2x2 patches)
        theta: RoPE base frequency (typically 10000)
        device: torch device
        dtype: output dtype
        window_idx: optional permutation for window ordering

    Returns:
        (cos, sin) each of shape [total_seq, head_dim // 2]
    """
    assert head_dim > 0 and head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"

    rotary_dim = head_dim // 2
    merge = spatial_merge_size
    device = torch.device(device)

    # Normalize grid_thw to list of tuples
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.is_cuda:
            grid_thw = grid_thw.cpu()
        thw_list = [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
    else:
        thw_list = [(int(t), int(h), int(w)) for t, h, w in grid_thw]

    # Check cache
    cache_key = (tuple(thw_list), head_dim, merge, theta, *_device_key(device), dtype)
    if cache_key in _rope_cache:
        _rope_cache.move_to_end(cache_key)
        cos, sin = _rope_cache[cache_key]
    else:
        # Inverse frequencies: [rotary_dim // 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))

        # Build position arrays for each image in the batch
        h_positions = []
        w_positions = []

        for t, h, w in thw_list:
            # Create 2D position grids
            h_pos = torch.arange(h, device=device, dtype=torch.int32)[:, None].expand(h, w)
            w_pos = torch.arange(w, device=device, dtype=torch.int32)[None, :].expand(h, w)

            # Reshape for spatial merge pattern (groups of merge x merge patches)
            h_pos = h_pos.reshape(h // merge, merge, w // merge, merge)
            h_pos = h_pos.permute(0, 2, 1, 3).reshape(-1)

            w_pos = w_pos.reshape(h // merge, merge, w // merge, merge)
            w_pos = w_pos.permute(0, 2, 1, 3).reshape(-1)

            # Repeat for temporal dimension if needed
            if t > 1:
                h_pos = h_pos.repeat(t)
                w_pos = w_pos.repeat(t)

            h_positions.append(h_pos)
            w_positions.append(w_pos)

        # Concatenate all positions
        h_positions = torch.cat(h_positions).float()
        w_positions = torch.cat(w_positions).float()

        # Compute frequencies: position * inv_freq
        freqs_h = h_positions[:, None] * inv_freq[None, :]  # [seq, rotary_dim // 2]
        freqs_w = w_positions[:, None] * inv_freq[None, :]  # [seq, rotary_dim // 2]

        # Concatenate h and w frequencies
        rotary_emb = torch.cat([freqs_h, freqs_w], dim=-1)  # [seq, rotary_dim]

        cos = rotary_emb.cos().to(dtype).contiguous()
        sin = rotary_emb.sin().to(dtype).contiguous()

        # Update cache
        _rope_cache[cache_key] = (cos, sin)
        _rope_cache.move_to_end(cache_key)
        while len(_rope_cache) > _ROPE_CACHE_MAX:
            _rope_cache.popitem(last=False)

    # Apply window reordering if provided
    if window_idx is not None:
        merge_unit = merge * merge
        num_positions = cos.shape[0] // merge_unit
        window_idx = window_idx.to(device=device, dtype=torch.long)

        cos = cos.reshape(num_positions, merge_unit, rotary_dim)
        cos = cos[window_idx].reshape(-1, rotary_dim)

        sin = sin.reshape(num_positions, merge_unit, rotary_dim)
        sin = sin[window_idx].reshape(-1, rotary_dim)

    return cos.contiguous(), sin.contiguous()


def fused_rope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RoPE + windowed attention for packed varlen windows.

    Expects Q/K/V already in window order and COS/SIN aligned to those positions.

    Shapes:
      - q, k, v: [total_seq, num_heads, head_dim]
      - cos, sin: [total_seq, head_dim // 2]  (NeoX-style split-half rotary)
      - cu_seqlens: [num_windows + 1] int32 cumulative lengths in token space
      - out: [total_seq, num_heads, head_dim] (optional)
    """
    if q.device.type != "cuda":
        raise ValueError(f"q must be on CUDA, got device={q.device}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have same shape, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.ndim != 3:
        raise ValueError(f"q/k/v must be [total_seq, num_heads, head_dim], got q.ndim={q.ndim}")

    total_seq, num_heads, head_dim = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]))
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")

    if cos.shape != (total_seq, head_dim // 2) or sin.shape != (total_seq, head_dim // 2):
        raise ValueError(
            f"cos/sin must be [total_seq, head_dim//2]={total_seq, head_dim//2}, got cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be 1D with length >= 2, got {tuple(cu_seqlens.shape)}")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q dtype must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q/k/v must share dtype, got q={q.dtype} k={k.dtype} v={v.dtype}")
    # cos/sin can be float32 (for precision) - kernel will handle conversion
    if cos.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"cos dtype must be float16, bfloat16 or float32, got {cos.dtype}")
    if sin.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"sin dtype must be float16, bfloat16 or float32, got {sin.dtype}")

    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    if not (cos.is_contiguous() and sin.is_contiguous()):
        cos = cos.contiguous()
        sin = sin.contiguous()

    if out is None:
        out = torch.empty_like(q)
    else:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}")
        if out.dtype != q.dtype or out.device != q.device:
            raise ValueError(f"out must match q dtype/device, got out={out.dtype}/{out.device} q={q.dtype}/{q.device}")

    cu_seqlens_i32 = cu_seqlens
    if cu_seqlens_i32.dtype != torch.int32:
        cu_seqlens_i32 = cu_seqlens_i32.to(dtype=torch.int32)
    if not cu_seqlens_i32.is_contiguous():
        cu_seqlens_i32 = cu_seqlens_i32.contiguous()

    num_windows = int(cu_seqlens_i32.numel() - 1)
    window_lens = cu_seqlens_i32[1:] - cu_seqlens_i32[:-1]
    max_seqlen = int(window_lens.max().item())
    if max_seqlen <= 0:
        return out.zero_()

    head_dim_padded = next_pow2(head_dim)
    half_dim_padded = next_pow2(head_dim // 2)

    grid = lambda meta: (triton.cdiv(max_seqlen, meta["BLOCK_M"]), num_windows * num_heads)
    fused_attn_kernel[grid](
        q, k, v, cos, sin, out,
        cu_seqlens_i32,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        cos.stride(0), cos.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_windows, num_heads, max_seqlen,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        HALF_DIM_PADDED=half_dim_padded,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ],
    key=["MAX_SEQLEN", "HEAD_DIM_PADDED"],
)
@triton.jit
def fused_attn_kernel(
    # Tensors
    Q, K, V, COS, SIN, Out,
    cu_seqlens,
    # Strides for Q [seq, heads, dim]
    stride_q_seq, stride_q_head, stride_q_dim,
    # Strides for K
    stride_k_seq, stride_k_head, stride_k_dim,
    # Strides for V
    stride_v_seq, stride_v_head, stride_v_dim,
    # Strides for cos/sin [seq, dim//2]
    stride_rope_seq, stride_rope_dim,
    # Strides for Out
    stride_o_seq, stride_o_head, stride_o_dim,
    # Dimensions
    NUM_WINDOWS, NUM_HEADS, MAX_SEQLEN,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,  # next power of 2
    HALF_DIM_PADDED: tl.constexpr,  # next power of 2 of HEAD_DIM // 2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused window attention with RoPE applied in registers.

    Each program handles one (window, head, row_block) combination.
    RoPE is applied to Q and K during load, never written to HBM.

    Layout:
        Q, K, V: [total_seq, num_heads, head_dim]
        COS, SIN: [total_seq, head_dim // 2]
        cu_seqlens: [num_windows + 1] (cumulative sequence lengths)
    """
    # Which window and head are we processing?
    row_block_id = tl.program_id(0)
    window_head_id = tl.program_id(1)

    window_id = window_head_id // NUM_HEADS
    head_id = window_head_id % NUM_HEADS

    if window_id >= NUM_WINDOWS:
        return

    # Get this window's boundaries from cu_seqlens
    window_start = tl.load(cu_seqlens + window_id)
    window_end = tl.load(cu_seqlens + window_id + 1)
    window_len = window_end - window_start

    if window_len <= 0:
        return

    # Which rows of Q are we processing?
    row_offset = row_block_id * BLOCK_M
    if row_offset >= window_len:
        return

    half_dim = HEAD_DIM // 2

    # Index arrays
    row_indices = row_offset + tl.arange(0, BLOCK_M)
    col_indices = tl.arange(0, BLOCK_N)
    dim_indices = tl.arange(0, HEAD_DIM_PADDED)
    half_dim_indices = tl.arange(0, HALF_DIM_PADDED)

    # Masks
    row_mask = row_indices < window_len
    dim_mask = dim_indices < HEAD_DIM
    half_mask = half_dim_indices < half_dim

    # Load Q and apply RoPE

    # Q is split into two halves for RoPE: q0 = Q[..., :half], q1 = Q[..., half:]
    q0_ptrs = (Q
               + (window_start + row_indices[:, None]) * stride_q_seq
               + head_id * stride_q_head
               + half_dim_indices[None, :] * stride_q_dim)
    q1_ptrs = (Q
               + (window_start + row_indices[:, None]) * stride_q_seq
               + head_id * stride_q_head
               + (half_dim_indices[None, :] + half_dim) * stride_q_dim)

    q0 = tl.load(q0_ptrs, mask=row_mask[:, None] & half_mask[None, :], other=0.0)
    q1 = tl.load(q1_ptrs, mask=row_mask[:, None] & half_mask[None, :], other=0.0)

    # Load RoPE tables for Q positions
    cos_q_ptrs = COS + (window_start + row_indices[:, None]) * stride_rope_seq + half_dim_indices[None, :] * stride_rope_dim
    sin_q_ptrs = SIN + (window_start + row_indices[:, None]) * stride_rope_seq + half_dim_indices[None, :] * stride_rope_dim

    cos_q = tl.load(cos_q_ptrs, mask=row_mask[:, None] & half_mask[None, :], other=0.0)
    sin_q = tl.load(sin_q_ptrs, mask=row_mask[:, None] & half_mask[None, :], other=0.0)

    # Apply rotation in fp32 for precision (matches HF behavior), then cast back
    # This happens in registers - no HBM cost
    q0_f = q0.to(tl.float32)
    q1_f = q1.to(tl.float32)
    cos_q_f = cos_q.to(tl.float32)
    sin_q_f = sin_q.to(tl.float32)
    q0_rot = (q0_f * cos_q_f - q1_f * sin_q_f).to(q0.dtype)
    q1_rot = (q0_f * sin_q_f + q1_f * cos_q_f).to(q0.dtype)

    # Initialize running max and sum

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Scale factor: log2(e) / sqrt(d) for numerical stability with exp2
    LOG2E = 1.4426950408889634
    scale = LOG2E / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    # Loop over K/V blocks

    for col_block_start in range(0, window_len, BLOCK_N):
        col_block_start = tl.multiple_of(col_block_start, BLOCK_N)
        curr_cols = col_block_start + col_indices
        col_mask = curr_cols < window_len

        # Load K halves and apply RoPE
        k0_ptrs = (K
                   + (window_start + curr_cols[:, None]) * stride_k_seq
                   + head_id * stride_k_head
                   + half_dim_indices[None, :] * stride_k_dim)
        k1_ptrs = (K
                   + (window_start + curr_cols[:, None]) * stride_k_seq
                   + head_id * stride_k_head
                   + (half_dim_indices[None, :] + half_dim) * stride_k_dim)

        k0 = tl.load(k0_ptrs, mask=col_mask[:, None] & half_mask[None, :], other=0.0)
        k1 = tl.load(k1_ptrs, mask=col_mask[:, None] & half_mask[None, :], other=0.0)

        # Load RoPE for K
        cos_k_ptrs = COS + (window_start + curr_cols[:, None]) * stride_rope_seq + half_dim_indices[None, :] * stride_rope_dim
        sin_k_ptrs = SIN + (window_start + curr_cols[:, None]) * stride_rope_seq + half_dim_indices[None, :] * stride_rope_dim

        cos_k = tl.load(cos_k_ptrs, mask=col_mask[:, None] & half_mask[None, :], other=0.0)
        sin_k = tl.load(sin_k_ptrs, mask=col_mask[:, None] & half_mask[None, :], other=0.0)

        # Apply RoPE to K in fp32 for precision, then cast back
        k0_f = k0.to(tl.float32)
        k1_f = k1.to(tl.float32)
        cos_k_f = cos_k.to(tl.float32)
        sin_k_f = sin_k.to(tl.float32)
        k0_rot = (k0_f * cos_k_f - k1_f * sin_k_f).to(k0.dtype)
        k1_rot = (k0_f * sin_k_f + k1_f * cos_k_f).to(k0.dtype)

        # Compute QK^T for rotated vectors
        # Since rotation is applied per-half: QK^T = Q0_rot @ K0_rot^T + Q1_rot @ K1_rot^T
        qk = (tl.dot(q0_rot, tl.trans(k0_rot)) + tl.dot(q1_rot, tl.trans(k1_rot))) * scale
        qk = tl.where(col_mask[None, :], qk, float("-inf"))

        # Softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_new[:, None])
        l_new = tl.math.exp2(m_i - m_new) * l_i + tl.sum(p, axis=1)

        # Rescale accumulator
        acc = acc * tl.math.exp2(m_i - m_new)[:, None]
        m_i = m_new
        l_i = l_new

        # Load V and accumulate
        v_ptrs = (V
                  + (window_start + curr_cols[:, None]) * stride_v_seq
                  + head_id * stride_v_head
                  + dim_indices[None, :] * stride_v_dim)
        v = tl.load(v_ptrs, mask=col_mask[:, None] & dim_mask[None, :], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    # Finalize and store

    acc = acc / l_i[:, None]

    out_ptrs = (Out
                + (window_start + row_indices[:, None]) * stride_o_seq
                + head_id * stride_o_head
                + dim_indices[None, :] * stride_o_dim)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=row_mask[:, None] & dim_mask[None, :])
