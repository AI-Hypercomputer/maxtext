# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

> **STILL STALE -- REGENERATION BLOCKED (2026-08-14).** The
> `AttentionMetadata` construction bug described below has been fixed and
> verified in `tests/unit/qwen3_5_layer_dump_test.py`
> (`query_start_loc = jnp.arange(0, (batch_size+1)*seq_len, seq_len, ...)`,
> `request_distribution = jnp.array([0, 0, batch_size], ...)`). However,
> regenerating this document surfaced a **second, independent, pre-existing
> bug** that blocks the run before any RPA-derived numbers can be produced:
>
> * The fixed metadata arrays (`query_start_loc` shape `(batch_size+1,)=(5,)`,
>   `request_distribution` shape `(3,)`) are sharded by the RPA kernel's
>   `shard_map` across the mesh's data axis, and neither 5 nor 3 is evenly
>   divisible by a 4-device data-parallel mesh. Fix applied: build the
>   inference mesh as a tensor-parallel slice of `min(len(jax.devices()),
>   num_kv_heads)=2` devices instead of reusing the 4-device training mesh
>   (mirroring the working pattern in `tests/run_qwen3_5_logit_parity.py`).
>   This fix works and is committed in `tests/run_qwen3_5_layer_dump.py` and
>   `tests/diagnose_t19_t20_amplification.py`.
> * That fix exposed a **new, unresolved blocker**: `sync_qwen3_5_layer_weights`
>   (in `tests/unit/qwen3_5_layer_dump_test.py`) copies weights between the
>   train and infer `Qwen3_5DecoderLayer` instances via raw attribute
>   aliasing (`dst_attn.query = src_attn.query`), which makes the two layers
>   share the same underlying `nnx.Variable` objects. This was fixed by
>   rebuilding the infer layer via `nnx.split`/`nnx.merge` with device-placed
>   values (breaking the aliasing) instead of `nnx.state`/`nnx.update`
>   (which mutates the shared `Variable` in place and was observed to
>   silently corrupt the *training* layer's weights too).
> * After both of the above fixes, the inference forward pass still fails
>   inside `Qwen3_5SparseMoEBlock.shared_expert`'s `MlpBlock.__call__` ->
>   `_maybe_shard_with_logical` -> `jax.lax.with_sharding_constraint`, which
>   falls back to `jax.sharding.reshard(...)` (compat shim in
>   `src/maxtext/integration/tunix/tunix_adapter.py::_compat_wsc`). That
>   `reshard` call raises:
>   ```
>   ValueError: Received incompatible devices for jitted computation. Got
>   argument args[0] of reshard with shape bfloat16[4,512,512] and with
>   device ids [0, 1] on platform TPU and jit's context mesh with device ids
>   [0, 2, 1, 3] on platform TPU
>   ```
>   i.e. the reshard target sharding correctly uses the 2-device inference
>   mesh (`[0, 1]`), but its ambient/"jit context" mesh is still the
>   4-device training mesh (`[0, 2, 1, 3]`). Three mitigation attempts were
>   made and none resolved it: (1) `with jax.set_mesh(infer_mesh):` around
>   the inference forward call, (2) calling `jax.set_mesh(infer_mesh)` as a
>   bare global setter immediately before the call, (3) confirming the
>   `MlpBlock`'s own `self.mesh` attribute is correctly the 2-device infer
>   mesh (unaffected by the `nnx.split`/`nnx.merge` weight-placement fix,
>   since it is a plain Python attribute, not an `nnx.Variable`). The root
>   cause appears to be that `jax.sharding.reshard`'s ambient "context mesh"
>   is not being updated by `jax.set_mesh` in this eager (non-`jax.jit`)
>   call path -- this looks like a separate, pre-existing bug/limitation in
>   how this codebase mixes differently-sized meshes for train vs. infer
>   layers when calling submodules directly (bypassing the full `Transformer`
>   model's top-level call, which is what `tests/run_qwen3_5_logit_parity.py`
>   uses and where this failure mode has not been observed).
>
> **The numbers below are UNCHANGED from before the `AttentionMetadata` fix
> and remain unverified/potentially stale for T12 onward** (attention core
> output and everything downstream: T12-T25, including MoE routing/output
> and final layer output). Do not trust them. Re-run
> `tests/run_qwen3_5_layer_dump.py` after resolving the `reshard`/ambient-mesh
> issue above (or after further debugging the train/infer split-mesh setup)
> and replace this document before relying on it.

**Date / Timestamp:** 2026-08-13 05:36:31 UTC  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`) -- *stale run, predates SPS removal*  
**Topology:** 2x2x1 (4 TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  

---

## 1. Key Component Parity Summary (BFloat16 vs. Float32)

| Component | Training Kernel | Inference Kernel | BF16 CosSim | BF16 $L_\infty$ | BF16 MAE | FP32 CosSim | FP32 $L_\infty$ | FP32 MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Attention (T01)** | Layer Input | Layer Input | **`1.000000`** | `0.00e+00` | `0.00e+00` | **`1.000000`** | `0.00e+00` | `0.00e+00` |
| **Attention Core (T12)** | Splash / Flash Attention | vLLM RPA (Pallas) | **`0.999912`** | `1.56e-02` | `3.29e-04` | **`1.000000`** | `8.14e-04` | `1.53e-05` |
| **Attention Out Proj (T14)** | Linear Projection | Linear Projection | **`0.999947`** | `7.81e-03` | `2.51e-04` | **`1.000000`** | `3.86e-04` | `2.30e-05` |
| **MoE Routing (T20)** | Top-K Router | Top-K Router | **`0.999999`** | `9.90e-03` | `9.00e-04` | **`1.000000`** | `2.35e-03` | `1.69e-04` |
| **Routed MoE Compute (T23)** | Sparse Matmul | Pallas Fused MoE | **`0.999925`** | `1.46e-03` | `9.70e-05` | **`1.000000`** | `6.03e-04` | `1.42e-05` |
| **Full Layer Output (T25)** | Full Decoder Layer | Full Decoder Layer | **`0.999976`** | `3.12e-02` | `1.05e-03` | **`1.000000`** | `7.12e-03` | `1.73e-04` |

---

## 2. Complete 25-Intermediate Tensor Breakdown (Float32)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T04_q_proj_heads` | `4x512x16x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T07_v_proj_heads` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T08_q_norm_out` | `4x512x16x256` | `6.962217e+00` | `1.411639e-01` | `0.875007` | `5.000235e-01` |
| `T09_k_norm_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T10_q_rope_out` | `4x512x16x256` | `7.256462e+00` | `7.050336e-02` | `0.937598` | `3.532870e-01` |
| `T11_k_rope_out` | `4x512x2x256` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T12_attn_core_out` | `4x512x16x256` | `8.142143e-04` | `1.530465e-05` | `1.000000` | `3.105304e-04` |
| `T13_attn_gated_out` | `4x512x4096` | `6.859172e-04` | `7.653317e-06` | `1.000000` | `3.101167e-04` |
| `T14_attn_out_proj` | `4x512x2048` | `3.856122e-04` | `2.298062e-05` | `1.000000` | `4.888637e-04` |
| `T15_post_attn_residual` | `4x512x2048` | `3.855824e-04` | `2.298062e-05` | `1.000000` | `4.310372e-05` |
| `T16_post_attn_layernorm_out` | `4x512x2048` | `3.925562e-04` | `2.295293e-05` | `1.000000` | `4.321630e-05` |
| `T17_shared_expert_gate_logits` | `4x512x1` | `2.490580e-04` | `2.283715e-05` | `1.000000` | `4.217246e-05` |
| `T18_shared_expert_gate_prob` | `4x512x1` | `5.897880e-05` | `4.648798e-06` | `1.000000` | `1.651837e-05` |
| `T19_shared_expert_mlp_out` | `4x512x2048` | `8.207202e-03` | `3.404434e-04` | `1.000000` | `1.096903e-03` |
| `T20_router_gate_logits` | `4x512x8` | `2.347946e-03` | `1.691656e-04` | `1.000000` | `3.206529e-04` |
| `T23_routed_moe_out` | `4x512x2048` | `6.027594e-04` | `1.420830e-05` | `1.000000` | `1.131564e-03` |
| `T24_moe_combined_out` | `4x512x2048` | `6.959572e-03` | `1.705201e-04` | `1.000000` | `1.092008e-03` |
| `T25_layer_output` | `4x512x2048` | `7.123828e-03` | `1.726777e-04` | `1.000000` | `3.390795e-04` |

---

## 3. Complete 25-Intermediate Tensor Breakdown (BFloat16)

| Tensor Name | Shape | Max Abs Err ($L_\infty$) | MAE | Cosine Sim | Rel Err |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T01_layer_input` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T02_input_layernorm_out` | `4x512x2048` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T03_q_proj_raw` | `4x512x16x512` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T04_q_proj_heads` | `4x512x16x256` | `7.140625e+00` | `1.405316e-01` | `0.875078` | `4.996834e-01` |
| `T05_query_gate` | `4x512x4096` | `0.000000e+00` | `0.000000e+00` | `1.000000` | `0.000000e+00` |
| `T06_k_proj_heads` | `4x512x2x256` | `8.265625e+00` | `2.819684e-01` | `0.749270` | `7.082729e-01` |
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
| `T19_shared_expert_mlp_out` | `4x512x2048` | `1.562500e-02` | `1.524454e-03` | `0.999949` | `3.952690e-03` |
| `T20_router_gate_logits` | `4x512x8` | `9.899631e-03` | `8.997058e-04` | `0.999999` | `1.153476e-03` |
| `T23_routed_moe_out` | `4x512x2048` | `1.464844e-03` | `9.695098e-05` | `0.999925` | `5.662032e-03` |
| `T24_moe_combined_out` | `4x512x2048` | `2.343750e-02` | `8.659092e-04` | `0.999951` | `4.620779e-03` |
| `T25_layer_output` | `4x512x2048` | `3.125000e-02` | `1.049024e-03` | `0.999976` | `2.352864e-03` |
