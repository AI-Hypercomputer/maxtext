# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** 2026-08-11 05:52:49 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Precision:** `bfloat16`  

---

## 1. Attention Precision & Tiling Comparative Analysis

| Configuration | `T12_attn_core_out` ($L_\infty$) | `T14_attn_out_proj` ($L_\infty$) | `T25_layer_output` (CosSim) |
| :--- | :--- | :--- | :--- |
| **Baseline (Splash Block 512)** | `2.515625e+00` | `9.316406e-01` | `0.998024` |
| **Option 2 (Tile Alignment 128x128)** | `2.515625e+00` | `9.316406e-01` | `0.998024` |
| **Option 3 (Tile 128 + Exact Math)** | `2.515625e+00` | `9.316406e-01` | `0.998024` |

---

## 2. Baseline Full 25-Tensor Breakdown (BFloat16)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `0.000000e+00` | `0.000000e+00` | `1.000001` | `0.000000e+00` |
| `T04_q_proj_heads` | `4x512x16x256` | `7.140625e+00` | `7.029243e-02` | `0.937553` | `2.136424e-04` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T07_v_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T08_q_norm_out` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T09_k_norm_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T10_q_rope_out` | `4x512x16x256` | `6.703125e+00` | `7.045352e-02` | `0.937635` | `1.871315e-05` |
| `T11_k_rope_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T12_attn_core_out` | `4x512x16x256` | `2.515625e+00` | `7.529779e-02` | `0.738960` | `2.911364e-01` |
| `T13_attn_gated_out` | `4x512x4096` | `2.140625e+00` | `3.777171e-02` | `0.738955` | `2.905402e-01` |
| `T14_attn_out_proj` | `4x512x2048` | `9.316406e-01` | `4.176092e-02` | `0.739607` | `2.908629e-01` |
| `T15_post_attn_residual` | `4x512x2048` | `9.296875e-01` | `4.176160e-02` | `0.998239` | `1.886804e-03` |
| `T16_post_attn_layernorm_out` | `4x512x2048` | `8.750000e-01` | `4.169420e-02` | `0.998265` | `1.144411e-05` |
| `T17_shared_expert_gate_logits` | `4x512x1` | `4.257812e-01` | `4.306390e-02` | `0.998165` | `1.188136e-04` |
| `T18_shared_expert_gate_prob` | `4x512x1` | `9.375000e-02` | `8.890271e-03` | `0.999708` | `1.041549e-04` |
| `T19_shared_expert_mlp_out` | `4x512x2048` | `7.553711e-01` | `3.583633e-02` | `0.996427` | `3.966553e-05` |
| `T20_router_gate_logits` | `4x512x8` | `4.470215e-01` | `4.181680e-02` | `0.998316` | `7.226728e-04` |
| `T23_routed_moe_out` | `4x512x2048` | `3.637695e-02` | `1.614570e-03` | `0.995510` | `9.811441e-04` |
| `T24_moe_combined_out` | `4x512x2048` | `5.332031e-01` | `1.888696e-02` | `0.996015` | `1.455442e-04` |
| `T25_layer_output` | `4x512x2048` | `1.230469e+00` | `4.637457e-02` | `0.998024` | `1.696426e-03` |
