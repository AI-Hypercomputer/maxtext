# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** 2026-08-11 06:56:47 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Precision:** `bfloat16`  

---

## 1. Key Component Parity Summary

| Component | Training Kernel | Inference Kernel | Cosine Similarity | Max Abs Error ($L_\infty$) | MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Attention (T01–T03)** | RMSNorm / Linear | RMSNorm / Linear | **`1.000000`** | **`0.000000e+00`** | **`0.000000e+00`** |
| **Attention Core (T12)** | Splash / Flash Attention | vLLM RPA (Pallas) | `0.738960` | `2.515625e+00` | `7.529779e-02` |
| **Attention Out Proj (T14)** | Linear Projection | Linear Projection | `0.739607` | `9.316406e-01` | `4.176092e-02` |
| **MoE Routing (T20)** | Top-K Router | Top-K Router | **`0.998316`** | `4.470215e-01` | `4.181680e-02` |
| **Routed MoE Compute (T23)** | Sparse Matmul | Pallas Fused MoE | **`0.995510`** | `3.637695e-02` | **`1.614570e-03`** |
| **Full Layer Output (T25)** | Full Decoder Layer | Full Decoder Layer | **`0.998024`** | `1.230469e+00` | `4.637457e-02` |

---

## 2. Complete 25-Intermediate Tensor Breakdown (BFloat16)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `6.890625e+00` | `7.028510e-02` | `0.937515` | `2.015305e-04` |
| `T04_q_proj_heads` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `8.265625e+00` | `2.819685e-01` | `0.749272` | `3.928261e-04` |
| `T07_v_proj_heads` | `4x512x2x256` | `6.218750e+00` | `2.815671e-01` | `0.749709` | `4.087020e-04` |
| `T08_q_norm_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T09_k_norm_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T10_q_rope_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T11_k_rope_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T12_attn_core_out` | `4x512x16x256` | `1.562500e-02` | `3.285446e-04` | `0.999993` | `2.142080e-05` |
| `T13_attn_gated_out` | `4x512x4096` | `1.562500e-02` | `1.646130e-04` | `0.999992` | `2.247695e-05` |
| `T14_attn_out_proj` | `4x512x2048` | `7.812500e-03` | `2.506587e-04` | `0.999990` | `2.505363e-05` |
| `T15_post_attn_residual` | `4x512x2048` | `1.562500e-02` | `2.511006e-04` | `1.000000` | `1.668314e-06` |
| `T16_post_attn_layernorm_out` | `4x512x2048` | `3.125000e-02` | `2.693845e-04` | `1.000000` | `1.430514e-06` |
| `T17_shared_expert_gate_logits` | `4x512x1` | `1.562500e-02` | `8.818870e-04` | `0.999998` | `5.861582e-05` |
| `T18_shared_expert_gate_prob` | `4x512x1` | `3.906250e-03` | `2.186298e-04` | `0.999999` | `3.339496e-05` |
| `T19_shared_expert_mlp_out` | `4x512x2048` | `1.953125e-02` | `1.549103e-03` | `0.999994` | `2.495250e-05` |
| `T20_router_gate_logits` | `4x512x8` | `1.562500e-02` | `9.060609e-04` | `0.999998` | `3.087031e-05` |
| `T23_routed_moe_out` | `4x512x2048` | `1.464844e-03` | `1.059606e-04` | `0.999982` | `1.538296e-04` |
| `T24_moe_combined_out` | `4x512x2048` | `2.343750e-02` | `8.825868e-04` | `0.999989` | `1.135776e-05` |
| `T25_layer_output` | `4x512x2048` | `3.125000e-02` | `1.065483e-03` | `0.999997` | `1.019896e-06` |
