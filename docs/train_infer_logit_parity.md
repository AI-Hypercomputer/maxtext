# Train vs. Inference Final-Logit Parity (Qwen3.5)

**Date / Timestamp:** 2026-08-14 17:07:39 UTC
**Hardware Platform:** Google Cloud TPU v5p, local TPU VM (locally-attached chips, no SPS proxy)
**Script:** `tests/run_qwen3_5_logit_parity.py`

## Methodology

Runs the **full** Qwen3.5 model (`maxtext.models.models.Transformer`:
token embedding -> N decoder layers -> final RMSNorm -> lm_head) end-to-end
on identical random token-id input, through two paths, and compares the
final **logits** tensor `[batch, seq_len, vocab_size]` -- not intermediate
activations.

* **Training path:** `attention="flash"` (Tokamax Splash Attention),
  `megablox=True, use_tokamax_gmm=True, use_gmm_v2=True, sparse_matmul=True`
  (Tokamax GMM v2 MoE), `model_mode=MODEL_MODE_TRAIN`.
* **Inference path:** `attention="vllm_rpa"` (default Pallas RPA v3; the real
  `tpu_inference` Ragged Paged Attention Pallas kernel, as served by
  vLLM), `prefuse_moe_weights=True` (routes MoE through
  `RoutedMoE.fused_moe_matmul` -> `tpu_inference.layers.common.fused_moe_gmm.fused_moe_func`,
  vLLM's real Pallas MoE kernel), `model_mode=MODEL_MODE_PREFILL`.

Both models are constructed with the same `nnx.Rngs(params=42)` seed, and
weights are additionally force-synchronized via `nnx.state(train_model,
nnx.Param)` / `nnx.update(infer_model, ...)` so residual differences are
attributable to kernel numerics, not initialization. No CPU mocks and no
reimplemented attention/MoE math are used on either side.

Metrics computed on the final logits:
* **L_inf / MAE / Cosine similarity** -- raw tensor-distance metrics.
* **Top-1 argmax agreement rate** -- fraction of positions where
  `argmax(logits_train) == argmax(logits_infer)`. This is what greedy
  decoding parity actually depends on.
* **Top-5 agreement rate** -- average overlap between the top-5 token sets.
* **KL divergence (train‖infer)**, mean and max across positions -- how much
  the sampling distributions actually diverge.

## Results

**Status: EXECUTED on local TPU VM (locally-attached chips).**

| Layers | DType | Shape [B,S,V] | $L_\infty$ | MAE | CosSim | Top-1 Agreement | Top-5 Agreement | Mean KL(train‖infer) | Max KL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | bfloat16 | [4, 128, 32000] | 2.188e-01 | 1.997e-02 | 0.999560 | 94.5312% | 94.9219% | 3.273e-04 | 1.116e-03 |
| 1 | float32 | [4, 128, 32000] | 2.161e-01 | 1.969e-02 | 0.999690 | 96.0938% | 95.2734% | 3.133e-04 | 1.148e-03 |
| 2 | bfloat16 | [4, 128, 32000] | 2.383e-01 | 2.863e-02 | 0.999221 | 91.0156% | 92.3828% | 6.585e-04 | 1.439e-03 |
| 2 | float32 | [4, 128, 32000] | 2.346e-01 | 2.807e-02 | 0.999379 | 90.0391% | 92.7734% | 6.300e-04 | 1.643e-03 |
| 40 | bfloat16 | [4, 128, 32000] | 8.203e-01 | 8.074e-02 | 0.995496 | 76.7578% | 83.7500% | 5.393e-03 | 1.658e-02 |
| 40 | float32 | [4, 128, 32000] | 9.090e-01 | 1.142e-01 | 0.989684 | 72.0703% | 76.2500% | 1.034e-02 | 1.708e-02 |

## Learnings

This section consolidates everything learned across the whole kernel-parity
investigation on this branch (previously spread across `docs/learnings.md`,
`docs/parity_improvement_story.md`, and `docs/next_plan.md`, which have been
folded into this document and removed to avoid stale duplicates). All numbers
below were produced by the standalone kernel repro scripts
(`tests/run_attention_kernel_repro.py`,
`tests/run_attention_batched_rpa_repro.py`,
`tests/run_moe_kernel_repro.py`) and the 1-layer intermediate-tensor dump
(`tests/run_qwen3_5_layer_dump.py`, results in
`docs/qwen3_5_kernel_drift_results.md`), all executed directly on the local
TPU VM's locally-attached Cloud TPU v5p chips. They describe
**intermediate-tensor** parity; the
final-logit numbers in the Results section above (or the "NOT EXECUTED"
status) are the authoritative full-model parity numbers for this document.

### 1. Attention: Splash (training) vs. RPA (inference)

* In FP32, isolated Splash Attention vs. RPA has $L_\infty \approx 1.53
  \times 10^{-5}$, CosSim $\approx 0.999999$.
* In BF16, both kernels show $L_\infty \approx 1.56 \times 10^{-2}$
  against an exact FP32 math reference -- this is the **1-ULP quantization
  floor** of BF16's 7-bit mantissa for values in $[2.0, 4.0)$, not a kernel
  bug. Over 60% of output elements are bit-identical.
* Setting `sa_use_base2_exp=False` (native base-$e$ exponential, matching
  RPA and the exact reference, instead of Tokamax Splash's default base-2
  fast-exp) reduced attention-core MAE by ~10.8% and MSE by ~16.1% vs. the
  `sa_use_base2_exp=True` default. This fix (`sa_use_base2_exp=False,
  sa_fuse_reciprocal=True`) is applied on the training side throughout this
  investigation.

### 2. MoE: Tokamax GMM v2 (training) vs. vLLM Fused MoE (inference)

* In isolation (identical input activations, FP32), both kernels agree with
  the exact FP32 math reference to $L_\infty \approx 10^{-8}$-$10^{-5}$
  and CosSim = 1.000000 -- i.e. **the MoE kernels themselves have no
  meaningful numerical divergence.**
* Aligning the training-side GMM contraction tile size to match inference's
  auto-tiling (`wi_tile_fwd_batch_seq=256, wi_tile_fwd_embed_dim=128,
  wi_tile_fwd_mlp_dim=128`) reduced training-vs-inference MoE $L_\infty$
  from $3.32\times10^{-5}$ to $2.98\times10^{-8}$ (FP32).
* `float32_gate_logits=True` and `float32_weight_sum=True` keep router
  logits and the top-$K$ weighted combination in FP32, preventing
  boundary-token misrouting and rounding loss in the expert combination.

### 3. Error amplification through the MoE MLP (why 1-layer FP32 error was ~7e-3, not ~1e-5)

* A 1-layer full decoder (attention + MoE) end-to-end FP32 comparison showed
  $L_\infty \approx 7.12\times10^{-3}$ at the layer output, even though
  both kernels are individually near machine precision in isolation.
* Diagnosed via `tests/diagnose_t19_t20_amplification.py`, which compares
  the **cascaded** (real, error-compounding) execution against an
  **isolated** execution where both training and inference MoE sub-blocks
  are fed the *identical* clean post-attention-norm activation. The isolated
  run reproduces the machine-precision agreement from Learning 2, confirming
  the $7\times10^{-3}$ error is not intrinsic to the MoE kernel -- it is
  the small attention-core residual ($\Delta x \approx 1.53\times10^{-5}$)
  passed through 3 successive linear projections
  ($W_{gate}, W_{up}, W_{down}$) in the MoE MLP, whose combined spectral
  norm ($\sim 10^2$-$10^3$) amplifies it: $\Delta y \approx \|W_{gate}\|
  \cdot \|W_{up}\| \cdot \|W_{down}\| \cdot \Delta x$.
* Practical takeaway: don't chase intermediate-tensor $L_\infty$ deltas at
  the MoE block in isolation from what feeds it -- verify the *source*
  (attention) delta and treat downstream amplification as expected linear
  algebra, not a new bug. The metrics that matter for actual generation
  quality are the final-logit top-1/top-5 argmax agreement and KL divergence
  reported in the Results section above, since RMSNorm variance-reset and
  the $O(1/\sqrt{L})$ residual-stream relative-error decay in a full
  multi-layer Pre-LN stack are expected to keep this bounded rather than
  exploding across layers -- **this has not yet been verified empirically
  beyond a 1-2 layer stack on this branch; a depth-scaling sweep (1, 2, 4, 8+
  layers) remains open future work.**

### 4. What is verified vs. still open

Verified on real Cloud TPU v5p hardware (local TPU VM) with real kernels
(no mocks):
* Standalone attention kernel parity (Splash vs. default RPA vs. batched
  RPA vs. exact reference), FP32 and BF16.
* Standalone MoE kernel parity (Tokamax GMM v2 vs. vLLM fused MoE vs. exact
  reference), FP32.
* 1-decoder-layer (attention + MoE) intermediate-tensor parity, FP32 and
  BF16, 25-tensor breakdown (`docs/qwen3_5_kernel_drift_results.md`).
* Error-amplification root-cause diagnosis (isolated vs. cascaded MoE
  sub-block execution).

Still open / not yet verified on this branch:
* Full-model, multi-layer (>2 layer) logit parity at production depth
  (32-64+ layers) -- only the 1-2 layer results in this document exist so
  far; run this script with a larger `num_decoder_layers` to extend.
* Autoregressive multi-step generation / top-1 greedy-token-match parity
  across a full decode rollout (KV cache reuse across steps), as opposed to
  a single prefill forward pass.
* Real (non-random) token inputs / real checkpoint weights, as opposed to
  freshly-initialized random weights synchronized between the two paths.

## Standalone Diagnostic Scripts

| Script | Purpose |
| :--- | :--- |
| `tests/run_qwen3_5_logit_parity.py` | **This document's source.** Full-model training-path vs. inference-path final logit parity. |
| `tests/run_attention_kernel_repro.py` | Multi-config sweep: Splash (legacy JAX / Tokamax variants) vs. default RPA vs. batched RPA vs. exact reference. |
| `tests/run_attention_batched_rpa_repro.py` | Focused Splash vs. default-RPA-v3 vs. batched-RPA 3-way comparison. |
| `tests/run_moe_kernel_repro.py` | Multi-config sweep: Tokamax GMM v2 (various tile sizes) vs. legacy Megablox vs. dense-einsum reference vs. vLLM fused MoE. |
| `tests/run_qwen3_5_layer_dump.py` | 1-decoder-layer, 25-intermediate-tensor dump and drift comparison; writes `docs/qwen3_5_kernel_drift_results.md`. |
| `tests/diagnose_t19_t20_amplification.py` | Isolated-vs-cascaded MoE sub-block diagnostic explaining the 1-layer error amplification mechanism (Learning 3 above). |

All of the above run directly on the local TPU VM's locally-attached Cloud
TPU v5p chips (no remote proxy needed), and require the `vllm-tpu` /
`tpu_inference` packages for the real inference-side kernels. Run with:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python NEW_MODEL_DESIGN=1 VLLM_TARGET_DEVICE=tpu \
python3 tests/run_qwen3_5_logit_parity.py
```
