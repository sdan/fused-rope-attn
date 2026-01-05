# Making flash_attn_varlen 2× faster with fused RoPE

 <img width="1482" height="879" alt="benchmark_chart" src="https://github.com/user-attachments/assets/49eb7d85-dc33-434e-9204-81ee636b1e1d" />


I've been experimenting with Qwen3-VL's vision encoder to understand how it works. While learning Triton, I stumbled onto a seemingly innocuous optimization: while loading Q and K tiles, we can apply RoPE in registers instead of reading pre-rotated tensors from HBM.

The problem: `flash_attn_varlen_func` has no RoPE parameter. `flash_attn_with_kvcache` does—but it's designed for LLM decoding with a KV cache. In VLMs like Qwen3-VL, we process all vision tokens so the vllm and HF implementation apply RoPE seperately -- writing Q_rot/K_rot to memory, then reading them back for attention.

Solution: I wrote a fused Triton kernel that applies RoPE in registers during the tile loads. Despite getting a 2x speedup, there are numerical precision issues when error accumulates through 32 transformer layers with images. Currently, I found big speedups with Qwen2.5-VL since it uses windowed attention, even for Qwen3-VL when batching or using multi-image inputs like with video, the speedup is still there but not as significant. This happens because on smaller windows, RoPE overhead is significant (memory-bound) but with long sequences, attention dominates (compute-bound) and Flash Attention's optimized CUDA kernel wins.


| Image Size | Baseline (ms) | Fused (ms) | Speedup |
|------------|---------------|------------|---------|
| 448×448    | 33.88         | 23.64      | **1.43x** |
| 672×672    | 54.12         | 27.69      | **1.95x** |
| 896×896    | 86.24         | 40.97      | **2.10x** |
| 1344×1344  | 180.38        | 79.85      | **2.26x** |