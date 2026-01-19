# DeepSeek-V3 Multi-Head Latent Attention (MLA) - JAX/Flax Implementation

A proof-of-concept implementation of DeepSeek-V3's MLA architecture in pure JAX/Flax.

## What This Does

This notebook compares two attention mechanisms: standard Multi-Head Attention (MHA) and DeepSeek's Multi-Head Latent Attention (MLA).

In standard attention, the model stores full key and value matrices in memory for every token. This uses a lot of memory, especially for long sequences.

MLA solves this by compressing the key and value information into smaller vectors before storing them. When the model needs to compute attention, it expands these compressed vectors back to full size on-the-fly. This way, the cache takes up much less memory while still producing the same results.

The notebook implements both approaches and measures how much memory each one uses. It then creates a graph showing the memory difference between them.

## Files

- `deepseek_mla_benchmark.ipynb`: Complete Jupyter notebook with implementation and benchmark

## How to Run

1. Upload `deepseek_mla_benchmark.ipynb` to Google Colab
2. Run all cells sequentially from top to bottom
3. The notebook will show a comparison table and generate a graph

## Results

(Results will be added here after running the notebook)
