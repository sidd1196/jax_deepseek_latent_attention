# DeepSeek-V3 Multi-Head Latent Attention (MLA): JAX and PyTorch Benchmarks

Implementations and GPU benchmarks comparing DeepSeek-V3's Multi-Head Latent Attention (MLA) against standard MHA and Grouped Query Attention (GQA).

## What is MLA?

In standard attention, the model caches full key and value matrices for every token. This memory scales linearly with context length and number of heads. MLA compresses the KV state into a single low-rank latent vector before caching it, then re-expands it on-the-fly at decode time. At DeepSeek-V3's 128-head config, the cache per token shrinks from 32,768 elements (MHA) to 8,704 (MLA), a 3.76x theoretical reduction.

---

## Files

| File | Description |
|---|---|
| `deepseek_mla_benchmark.ipynb` | Original JAX/Flax implementation with formula-based memory projections |
| `deepseek_mla_pytorch__1_.ipynb` | PyTorch benchmark with real GPU measurements on A100 (128-head DeepSeek-V3 config) |
| `decode_benchmark.py` | Standalone decode latency script with NVTX annotations for Nsight Systems profiling |

---

## PyTorch Benchmark (`deepseek_mla_pytorch__1_.ipynb`)

Real GPU measurements at DeepSeek-V3's actual configuration: 128 heads, d_model=16384, on an A100-SXM4-40GB in bfloat16.

### KV Cache Memory (batch=1)

| Seq Len | MHA (MB) | GQA (MB) | MLA (MB) | MHA/MLA |
|---|---|---|---|---|
| 1,024 | 100.7 | 37.7 | 51.4 | 1.96x |
| 2,048 | 201.3 | 75.5 | 102.8 | 1.96x |
| 4,096 | 402.7 | 151.0 | 205.5 | 1.96x |
| 8,192 | 805.3 | 302.0 | 411.0 | 1.96x |
| 16,384 | 1,610.6 | 604.0 | 822.1 | 1.96x |

MLA/MHA memory ratio is a consistent 1.96x (not the theoretical 3.76x) because the A100 measurement includes model weights in the baseline, not just the cache tensors.

### Decode Latency (ms/token, batch=1, constant context per step)

| Ctx Len | MHA (ms) | GQA (ms) | MLA (ms) | MLA/MHA |
|---|---|---|---|---|
| 1,024 | 2.17 | 1.16 | 1.69 | 0.78x |
| 2,048 | 2.42 | 1.25 | 1.91 | 0.79x |
| 4,096 | 3.26 | 1.59 | 2.60 | 0.80x |
| 8,192 | 4.95 | 2.30 | 3.95 | 0.80x |
| 16,384 | 8.26 | 3.72 | 6.61 | 0.80x |

**MLA is ~0.8x of MHA at every context length** (faster, not slower). MHA's weight matrices are 2.15 GB vs MLA's 1.36 GB, so MLA's weight bandwidth savings outweigh the kv_up_proj overhead throughout the 1k-16k token range.

### TTFT (batch=1)

| Seq Len | MHA (ms) | GQA (ms) | MLA (ms) |
|---|---|---|---|
| 1,024 | 8.50 | 4.84 | 6.46 |
| 4,096 | 34.75 | 20.26 | 26.23 |
| 16,384 | 170.75 | 112.12 | 134.83 |

MLA TTFT is ~21% faster than MHA at 16k tokens (135ms vs 171ms). GQA is the fastest throughout due to its smaller weight matrices.

### Key numbers

- **3.76x** less KV cache per token vs MHA (theoretical, from correctness test)
- **~0.8x** MHA decode latency at all measured context lengths on A100
- **~21% faster** TTFT than MHA at 16k tokens
- GQA is the throughput winner: 1,326 tok/s vs MLA's 708 tok/s at batch=16, ctx=4096 (GQA's 8 KV heads give it a large weight bandwidth advantage)

---

## Nsight Systems Profiling (`decode_benchmark.py`)

Standalone script for profiling with NVTX annotations. Prefills to ctx=4096 then times 20 decode steps for each model.

```bash
# Clean run
python decode_benchmark.py

# Under Nsight Systems (Colab A100)
<nsys_path>/nsys profile --output=mla_decode --trace=cuda,nvtx \
    --gpu-metrics-devices=0 --force-overwrite=true \
    python decode_benchmark.py
```

NVTX regions annotated: `{MODEL}_prefill`, `{MODEL}_warmup_{i}`, `{MODEL}_decode_step_{i}`, and `kv_up_proj` inside MLA's forward pass.

---

## JAX/Flax Notebook (`deepseek_mla_benchmark.ipynb`)

Original proof-of-concept using formula-based memory projections (not real GPU measurements). Config: 32 heads, head_dim=128, kv_lora_rank=64, rope_dim=64.

- Average memory reduction: 3.88x
- Max sequence length tested: 32,768 tokens

---

## How to Run

**PyTorch notebook:** Upload `deepseek_mla_pytorch.ipynb` to Google Colab, switch runtime to A100 GPU, run all cells top to bottom.

**JAX notebook:** Upload `deepseek_mla_benchmark.ipynb` to Google Colab, any GPU runtime, run all cells.
