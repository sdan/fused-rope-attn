# Making flash_attn_varlen 2× faster with fused RoPE

 <img width="1482" height="879" alt="benchmark_chart" src="https://github.com/user-attachments/assets/49eb7d85-dc33-434e-9204-81ee636b1e1d" />


I've been experimenting with Qwen3-VL's vision encoder to understand how it works. While learning Triton, I stumbled onto a seemingly innocuous optimization: while loading Q and K tiles, we can apply RoPE in registers instead of reading pre-rotated tensors from HBM.

The problem: `flash_attn_varlen_func` has no RoPE parameter. `flash_attn_with_kvcache` does—but it's designed for LLM decoding with a KV cache. In VLMs like Qwen3-VL, we process all vision tokens so the vllm and HF implementation apply RoPE seperately -- writing Q_rot/K_rot to memory, then reading them back for attention.

I wrote a fused Triton kernel that applies RoPE in registers during the tile loads. Result: **2× faster vision encoder** on H200 for Qwen2.5-VL.


| Image Size | Baseline (ms) | Fused (ms) | Speedup |
|------------|---------------|------------|---------|
| 448×448    | 33.88         | 23.64      | **1.43x** |
| 672×672    | 54.12         | 27.69      | **1.95x** |
| 896×896    | 86.24         | 40.97      | **2.10x** |
| 1344×1344  | 180.38        | 79.85      | **2.26x** |

**Average speedup: 1.94x**

### Limitations

This kernel is designed for **windowed attention patterns** where many small sequences are processed:

- **Qwen2.5-VL** uses `window_size=112` with `fullatt_block_indexes=[7,15,23,31]` → 64-token windows → **good fit**
- **Qwen3-VL** removed windowed attention → global attention per frame → not a good fit

**Why**: With small windows, RoPE overhead is significant (memory-bound). With long sequences, attention dominates (compute-bound) and Flash Attention's optimized CUDA kernel wins.

See [this discussion](https://github.com/QwenLM/Qwen3-VL/issues/1717) on why Qwen3-VL dropped windowed attention.

## How to benchmark

```bash
# Micro-benchmark (kernel-only)
modal run bench.py

# Full Qwen2.5-VL model benchmark
modal run tests/test_vllm_vision_bench.py
```

## Usage