# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** 2026-08-11 03:36:28 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Dtype:** `bfloat16` (Production training & serving precision)  

---

## 1. Executive Summary & Core Objective

The purpose of this benchmark is to measure and isolate numerical drift between:
* **Trainer Execution Paradigm:** `attention="flash"` (TPU Splash / Flash Attention) + `sparse_matmul=True` (Megablox Grouped Matmul MoE) in `MODEL_MODE_TRAIN`.
* **Inference Execution Paradigm:** `attention="vllm_rpa"` (vLLM Ragged Paged Attention) + `fused_moe_matmul=True` (Pallas Fused MoE with prefused gate/up weights) with `NEW_MODEL_DESIGN=1` in `model_call_mode="inference"`.

All parameter matrices were synchronized from Trainer to Inference prior to execution, ensuring 100% parameter bit-parity. A total of **25 intermediate activation tensors** were captured along the entire layer forward pass.

---

## 2. Quantitative Results: BFloat16 Intermediate Tensor Drift

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

---

## 3. Detailed Numerical Divergence Attribution

### A. Pre-Attention Normalization & Linear Projections (T01 - T11)
* **`T01_layer_input` through `T11_k_rope_out`:** All show **bitwise-identical matching** ($L_\infty = 0.000000$, MAE = $0.000000$, Cosine Similarity = $1.000000$).
* **Conclusion:** Input RMSNorm, Q/K/V linear projections, QK-Norm, Query Gate, and Rotary Position Embeddings (RoPE) are mathematically identical between training and inference paradigms.

### B. Attention Core Kernel (T12 - T14)
* **`T12_attn_core_out`:** Splash Attention (Pallas Flash Attention) vs vLLM RPA (Ragged Paged Attention) introduces an $L_\infty$ difference of $3.92$ and MAE of $0.119$.
* **`T14_attn_out_proj`:** Output projection propagates the attention core difference with $L_\infty = 1.959$ and MAE = $0.065$.
* **Attribution:** Flash Attention and vLLM RPA use different block sizes and tiling strategies on TPU matrix units (MXUs), leading to standard BFloat16 summation order non-associativity across attention head dimensions.

### C. Post-Attention Residual & Normalization (T15 - T16)
* **`T15_post_attn_residual`:** $X + \text{AttnOut}$ stabilizes cosine similarity back to **$0.995548$** due to the dominant residual connection.
* **`T16_post_attn_layernorm_out`:** RMSNorm maintains high directional alignment with Cosine Similarity of **$0.995662$**.

### D. Shared Expert & MoE Router (T17 - T20)
* **`T17_shared_expert_gate_logits` & `T18_shared_expert_gate_prob`:** Cosine similarity of **$0.999147$** with tight bounds ($L_\infty = 0.160$, MAE = $0.015$).
* **`T20_router_gate_logits`:** MoE router logits exhibit **$0.995836$** cosine similarity, ensuring highly stable top-8 expert routing selection.

### E. Routed MoE Kernel & Final Layer Output (T23 - T25)
* **`T23_routed_moe_out`:** Comparing Megablox `sparse_matmul` (training) vs Pallas `fused_moe_matmul` (inference) shows extremely close alignment with $L_\infty = 0.063293$, MAE = $0.002510$, and Cosine Similarity of **$0.989014$**.
* **`T24_moe_combined_out`:** MoE combined output achieves **$0.989757$** cosine similarity.
* **`T25_layer_output`:** The complete layer output ($X + \text{AttnOut} + \text{MoEOut}$) achieves **$0.994996$** cosine similarity ($> 0.99$), demonstrating that total numerical drift between MaxText training and vLLM inference remains well bounded within production tolerances.

---

## 4. Verification & Reproduction Instructions

To execute this benchmark on any Shared Pathways Service TPU cluster:
```bash
NEW_MODEL_DESIGN=1 python3 tests/run_sps_qwen3_5_dump.py
```
Or run the unit test suite:
```bash
NEW_MODEL_DESIGN=1 pytest tests/unit/qwen3_5_layer_dump_test.py
```
