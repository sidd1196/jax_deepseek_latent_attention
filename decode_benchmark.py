"""
decode_benchmark.py
Standalone decode latency benchmark for MHA vs GQA vs DeepSeek MLA.
NVTX-annotated for Nsight Systems profiling on A100.

Run clean:
    python decode_benchmark.py

Run under Nsight Systems:
    nsys profile --output=mla_decode --trace=cuda,nvtx \
        --gpu-metrics-device=0 --force-overwrite=true \
        python decode_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

# ── Fixed hyperparameters (no adaptive tier logic) ────────────────────────────
NUM_HEADS    = 128
HEAD_DIM     = 128
NUM_KV_HEADS = 8
KV_LORA_RANK = 512
ROPE_DIM     = 64
D_MODEL      = NUM_HEADS * HEAD_DIM   # 16384
DTYPE        = torch.bfloat16
DEVICE       = "cuda"
SEQ_LEN      = 4096
BATCH_SIZE   = 1
DECODE_STEPS = 20
WARMUP_STEPS = 5

assert torch.cuda.is_available(), "CUDA GPU required"
print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"d_model    : {D_MODEL}  ({NUM_HEADS} heads × {HEAD_DIM} head_dim)")
print(f"SEQ_LEN    : {SEQ_LEN}")
print(f"DTYPE      : {DTYPE}")
print()


# ── Model definitions ─────────────────────────────────────────────────────────

class StandardMHA(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.q_proj   = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj   = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj   = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=dtype)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        H, D    = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
        new_cache  = {'k': k, 'v': v}
        is_prefill = (T > 1 and kv_cache is None)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill)
        out  = attn.transpose(1, 2).reshape(B, T, H * D)
        return self.out_proj(out), new_cache


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.groups       = num_q_heads // num_kv_heads
        self.q_proj   = nn.Linear(d_model, num_q_heads  * head_dim, bias=False, dtype=dtype)
        self.k_proj   = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj   = nn.Linear(d_model, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_q_heads * head_dim, d_model,  bias=False, dtype=dtype)

    def forward(self, x, kv_cache=None):
        B, T, _     = x.shape
        Hq, Hkv, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, Hq,  D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
        new_cache  = {'k': k, 'v': v}
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        is_prefill = (T > 1 and kv_cache is None)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill)
        out  = attn.transpose(1, 2).reshape(B, T, Hq * D)
        return self.out_proj(out), new_cache


class DeepSeekMLA(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, kv_lora_rank, rope_dim, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim     = rope_dim
        self.content_dim  = head_dim - rope_dim
        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank,             bias=False, dtype=dtype)
        self.kv_up_proj   = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_rope_proj  = nn.Linear(d_model, num_heads * rope_dim,     bias=False, dtype=dtype)
        self.q_proj       = nn.Linear(d_model, num_heads * self.content_dim, bias=False, dtype=dtype)
        self.q_rope_proj  = nn.Linear(d_model, num_heads * rope_dim,     bias=False, dtype=dtype)
        self.out_proj     = nn.Linear(num_heads * head_dim, d_model,     bias=False, dtype=dtype)

    def _rope(self, x, positions):
        half     = x.shape[-1] // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
        freqs    = torch.outer(positions.float(), inv_freq)
        cos      = freqs.cos()[None, None].to(x.dtype)
        sin      = freqs.sin()[None, None].to(x.dtype)
        x1, x2  = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        H, D    = self.num_heads, self.head_dim
        C, R    = self.content_dim, self.rope_dim

        cache_len = kv_cache['c_kv'].shape[1] if kv_cache is not None else 0
        positions = torch.arange(cache_len, cache_len + T, device=x.device)

        q_content = self.q_proj(x).view(B, T, H, C).transpose(1, 2)
        q_rope    = self.q_rope_proj(x).view(B, T, H, R).transpose(1, 2)
        q_rope    = self._rope(q_rope, positions)
        q         = torch.cat([q_content, q_rope], dim=-1)

        c_kv_new   = self.kv_down_proj(x)
        k_rope_new = self.k_rope_proj(x).view(B, T, H, R).transpose(1, 2)
        k_rope_new = self._rope(k_rope_new, positions)
        k_rope_new = k_rope_new.transpose(1, 2)

        if kv_cache is not None:
            c_kv   = torch.cat([kv_cache['c_kv'],   c_kv_new],  dim=1)
            k_rope = torch.cat([kv_cache['k_rope'], k_rope_new], dim=1)
        else:
            c_kv   = c_kv_new
            k_rope = k_rope_new

        new_cache = {'c_kv': c_kv, 'k_rope': k_rope}

        S = c_kv.shape[1]
        # ── kv_up_proj: the key MLA overhead — annotated for Nsight ──────────
        nvtx.range_push("kv_up_proj")
        kv = self.kv_up_proj(c_kv).view(B, S, H, D)
        nvtx.range_pop()

        k_content = kv[..., :C].transpose(1, 2)
        v         = kv.transpose(1, 2)
        k         = torch.cat([k_content, k_rope.transpose(1, 2)], dim=-1)

        is_prefill = (T > 1 and kv_cache is None)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill)
        out  = attn.transpose(1, 2).reshape(B, T, H * D)
        return self.out_proj(out), new_cache


# ── Instantiate models ────────────────────────────────────────────────────────
print("Instantiating models...")
mha_model = StandardMHA(D_MODEL, NUM_HEADS, HEAD_DIM, dtype=DTYPE).to(DEVICE).eval()
gqa_model = GroupedQueryAttention(D_MODEL, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE).to(DEVICE).eval()
mla_model = DeepSeekMLA(D_MODEL, NUM_HEADS, HEAD_DIM, KV_LORA_RANK, ROPE_DIM, dtype=DTYPE).to(DEVICE).eval()
torch.cuda.synchronize()
print("Models ready.\n")

# ── Benchmark loop ────────────────────────────────────────────────────────────
results = {}

for model_name, model in [('MHA', mha_model), ('GQA', gqa_model), ('MLA', mla_model)]:
    print(f"--- {model_name} ---")
    torch.cuda.empty_cache()

    # Prefill
    nvtx.range_push(f"{model_name}_prefill")
    x_pre = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        _, cache = model(x_pre)
    del x_pre
    torch.cuda.synchronize()
    nvtx.range_pop()

    cache_snapshot = {k: v.clone() for k, v in cache.items()}
    del cache
    x_dec = torch.randn(BATCH_SIZE, 1, D_MODEL, dtype=DTYPE, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for i in range(WARMUP_STEPS):
            nvtx.range_push(f"{model_name}_warmup_{i}")
            model(x_dec, kv_cache={k: v.clone() for k, v in cache_snapshot.items()})
            nvtx.range_pop()
    torch.cuda.synchronize()

    # Measure
    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        t_start.record()
        for i in range(DECODE_STEPS):
            nvtx.range_push(f"{model_name}_decode_step_{i}")
            model(x_dec, kv_cache={k: v.clone() for k, v in cache_snapshot.items()})
            nvtx.range_pop()
        t_end.record()
    torch.cuda.synchronize()

    ms_per_tok = t_start.elapsed_time(t_end) / DECODE_STEPS
    results[model_name] = ms_per_tok
    print(f"  mean decode latency: {ms_per_tok:.2f} ms/token")

    del x_dec, cache_snapshot
    torch.cuda.empty_cache()
    torch.cuda.synchronize()   # clean separation in Nsight timeline

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  Decode Latency Summary  |  ctx={SEQ_LEN}  |  batch={BATCH_SIZE}")
print(f"  {torch.cuda.get_device_name(0)}  |  {DTYPE}")
print("=" * 55)
print(f"  {'Model':>6}  {'ms/token':>10}  {'vs MHA':>10}  {'vs GQA':>10}")
print("-" * 55)
for name in ['MHA', 'GQA', 'MLA']:
    vs_mha = results[name] / results['MHA']
    vs_gqa = results[name] / results['GQA']
    print(f"  {name:>6}  {results[name]:>10.2f}  {vs_mha:>9.2f}x  {vs_gqa:>9.2f}x")
print("=" * 55)

torch.cuda.synchronize()
