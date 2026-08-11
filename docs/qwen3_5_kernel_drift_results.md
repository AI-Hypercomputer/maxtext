# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** 2026-08-11 04:28:34 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Precision:** `bfloat16`  

---

## 1. Attention Precision & Tiling Comparative Analysis

| Configuration | `T12_attn_core_out` ($L_\infty$) | `T14_attn_out_proj` ($L_\infty$) | `T25_layer_output` (CosSim) |
| :--- | :--- | :--- | :--- |
| **Baseline (Splash Block 512)** | `3.920898e+00` | `1.959229e+00` | `0.994996` |
| **Option 2 (Tile Alignment 128x128)** | `3.920898e+00` | `1.959229e+00` | `0.994996` |
| **Option 3 (Tile 128 + Exact Math)** | `3.920898e+00` | `1.959229e+00` | `0.994996` |

---

## 2. Baseline Full 25-Tensor Breakdown (BFloat16)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `0.000000e+00` | `0.000000e+00` | `1.000001` | `0.000000e+00` |
| `T04_q_proj_heads` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T07_v_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T08_q_norm_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T09_k_norm_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T10_q_rope_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T11_k_rope_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T12_attn_core_out` | `4x512x16x256` | `3.920898e+00` | `1.193858e-01` | `0.000726` | `6.149672e-01` |
| `T13_attn_gated_out` | `4x512x4096` | `3.328125e+00` | `5.988797e-02` | `0.000416` | `6.148620e-01` |
| `T14_attn_out_proj` | `4x512x2048` | `1.959229e+00` | `6.504712e-02` | `0.001387` | `6.152644e-01` |
| `T15_post_attn_residual` | `4x512x2048` | `1.957031e+00` | `6.504941e-02` | `0.995548` | `3.302607e-03` |
| `T16_post_attn_layernorm_out` | `4x512x2048` | `1.789062e+00` | `6.489170e-02` | `0.995662` | `1.335147e-05` |
| `T17_shared_expert_gate_logits` | `4x512x1` | `7.558594e-01` | `7.585297e-02` | `0.994397` | `4.168467e-03` |
| `T18_shared_expert_gate_prob` | `4x512x1` | `1.601562e-01` | `1.554990e-02` | `0.999147` | `1.714664e-03` |
| `T19_shared_expert_mlp_out` | `4x512x2048` | `1.484375e+00` | `5.549413e-02` | `0.991197` | `1.802246e-03` |
| `T20_router_gate_logits` | `4x512x8` | `1.099609e+00` | `6.515802e-02` | `0.995836` | `9.518938e-04` |
| `T23_routed_moe_out` | `4x512x2048` | `6.329346e-02` | `2.510488e-03` | `0.989014` | `7.230454e-05` |
| `T24_moe_combined_out` | `4x512x2048` | `1.044922e+00` | `2.963309e-02` | `0.989757` | `3.717284e-03` |
| `T25_layer_output` | `4x512x2048` | `2.238281e+00` | `7.227437e-02` | `0.994996` | `3.352284e-03` |
