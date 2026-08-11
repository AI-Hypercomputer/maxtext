# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** 2026-08-11 07:37:49 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Precision:** `bfloat16`  

---

## 1. Key Component Parity Summary

| Component | Training Kernel | Inference Kernel | Cosine Similarity | Max Abs Error ($L_\infty$) | MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Attention (T01)** | Layer Input | Layer Input | **`1.000000`** | **`0.000000e+00`** | **`0.000000e+00`** |
| **Attention Core (T12)** | Splash / Flash Attention | vLLM RPA (Pallas) | **`0.999912`** | `1.562500e-02` | `3.285446e-04` |
| **Attention Out Proj (T14)** | Linear Projection | Linear Projection | **`0.999947`** | `7.812500e-03` | `2.506588e-04` |
| **MoE Routing (T20)** | Top-K Router | Top-K Router | **`0.999998`** | `1.562500e-02` | `9.060609e-04` |
| **Routed MoE Compute (T23)** | Sparse Matmul | Pallas Fused MoE | **`0.999921`** | `1.464844e-03` | **`1.059607e-04`** |
| **Full Layer Output (T25)** | Full Decoder Layer | Full Decoder Layer | **`0.999976`** | `3.125000e-02` | `1.065484e-03` |

---

## 2. Complete 25-Intermediate Tensor Breakdown (BFloat16)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `7.531250e+00` | `7.031320e-02` | `0.937515` | `3.535181e-01` |
| `T04_q_proj_heads` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T07_v_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T08_q_norm_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T09_k_norm_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T10_q_rope_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T11_k_rope_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T12_attn_core_out` | `4x512x16x256` | `1.562500e-02` | `3.285446e-04` | `0.999912` | `3.800648e-03` |
| `T13_attn_gated_out` | `4x512x4096` | `1.562500e-02` | `1.646131e-04` | `0.999939` | `4.066877e-03` |
| `T14_attn_out_proj` | `4x512x2048` | `7.812500e-03` | `2.506588e-04` | `0.999947` | `4.624669e-03` |
| `T15_post_attn_residual` | `4x512x2048` | `1.562500e-02` | `2.511005e-04` | `0.999993` | `1.073334e-03` |
| `T16_post_attn_layernorm_out` | `4x512x2048` | `3.125000e-02` | `2.693846e-04` | `0.999994` | `1.142150e-03` |
| `T17_shared_expert_gate_logits` | `4x512x1` | `1.562500e-02` | `8.818870e-04` | `0.999998` | `1.982377e-03` |
| `T18_shared_expert_gate_prob` | `4x512x1` | `3.906250e-03` | `2.186298e-04` | `0.999999` | `1.479646e-03` |
| `T19_shared_expert_mlp_out` | `4x512x2048` | `1.953125e-02` | `1.549102e-03` | `0.999949` | `4.005917e-03` |
| `T20_router_gate_logits` | `4x512x8` | `1.562500e-02` | `9.060609e-04` | `0.999998` | `2.065531e-03` |
| `T23_routed_moe_out` | `4x512x2048` | `1.464844e-03` | `1.059607e-04` | `0.999921` | `6.133668e-03` |
| `T24_moe_combined_out` | `4x512x2048` | `4.960938e+00` | `9.413179e-02` | `0.638946` | `7.732792e-01` |
| `T25_layer_output` | `4x512x2048` | `3.125000e-02` | `1.065484e-03` | `0.999976` | `2.374252e-03` |
