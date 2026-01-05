"""
Benchmark: Flash+RoPE vs Fused kernel.

Run: modal run bench.py
"""
import modal
from pathlib import Path

app = modal.App("fused-rope-attn")

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install("triton>=3.0.0")
    .add_local_file(Path(__file__).parent / "fused_attn.py", "/root/fused_attn.py")
    .add_local_file(Path(__file__).parent / "utils.py", "/root/utils.py")
)


@app.function(image=base_image, gpu="H100", timeout=600)
def run():
    import sys
    sys.path.insert(0, "/root")

    import torch
    import triton
    from flash_attn import flash_attn_varlen_func
    from fused_attn import compute_rope_2d, fused_rope_attention
    from utils import get_window_index

    device = torch.device("cuda")
    dtype = torch.float16

    # Qwen3-VL vision encoder config
    num_heads, head_dim = 16, 72
    spatial_merge_size, window_size, patch_size = 2, 112, 14

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: heads={num_heads}, head_dim={head_dim}\n")
    print("| Image | Seq Len | Flash+RoPE | Fused | Speedup |")
    print("|-------|---------|------------|-------|---------|")

    for img_size in [448, 672, 896, 1344]:
        patches = img_size // patch_size
        grid_thw = [(1, patches, patches)]
        seq_len = patches * patches

        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        num_windows = len(cu_seqlens) - 1
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)

        cos, sin = compute_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
            theta=10000.0,
            device=device,
            dtype=dtype,
            window_idx=window_index,
        )

        # Baseline: Flash + separate RoPE (what vLLM/HF do)
        def flash_rope():
            half = head_dim // 2
            c, s = cos[:, None, :], sin[:, None, :]
            q_rot = torch.cat([q[..., :half] * c - q[..., half:] * s,
                               q[..., :half] * s + q[..., half:] * c], dim=-1)
            k_rot = torch.cat([k[..., :half] * c - k[..., half:] * s,
                               k[..., :half] * s + k[..., half:] * c], dim=-1)
            return flash_attn_varlen_func(
                q_rot, k_rot, v, cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen, 0.0, head_dim ** -0.5, False,
            )

        # Ours: Fused RoPE + Attention
        def fused():
            return fused_rope_attention(q, k, v, cos, sin, cu_seqlens)

        # Warmup
        for _ in range(10):
            flash_rope()
            fused()
        torch.cuda.synchronize()

        # Benchmark
        flash_ms = triton.testing.do_bench(flash_rope, warmup=20, rep=100)
        fused_ms = triton.testing.do_bench(fused, warmup=20, rep=100)
        speedup = flash_ms / fused_ms

        print(f"| {img_size}x{img_size} | {seq_len:,} | {flash_ms:.2f} ms | {fused_ms:.2f} ms | **{speedup:.1f}x** |")


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            run.remote()
