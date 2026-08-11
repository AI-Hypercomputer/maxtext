# Isolated Attention Kernel Parity: Splash Attention vs. RPA

**Date:** 2026-08-11 06:50:11 UTC  
**Hardware:** Google Cloud TPU v5p (`auto-v5p-8-bodaborg`)  
**Configuration:** `batch_size=4`, `seq_len=512`, `num_query_heads=16`, `num_kv_heads=2`, `head_dim=256`, `dtype=bfloat16`  

---

## 1. Direct Comparative Parity

| Comparison Pair | Max Abs Error ($L_\infty$) | MAE | MSE | Cosine Similarity | Relative Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Splash Attn (Train) vs. RPA (Infer)** | `1.562500e-02` | `3.249594e-04` | `3.740219e-07` | **`0.999913`** | `3.771769e-03` |

---

## 2. Key Diagnostic Takeaway

1. **Kernel Disparity Root Cause:** By isolating $(Q, K, V)$ to identical synthetic inputs, all outer network operations (projections, layernorms, RoPE, gating, and MoE) are completely eliminated.
2. **Current Metric:** Splash Attention and RPA produce a baseline cosine similarity of **99.99%** on identical inputs.
