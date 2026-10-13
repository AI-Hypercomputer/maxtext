---
name: model-bringup
description: >-
  Performs new LLM/VLM model bring-up in MaxText on TPU VMs. Use when onboarding a new HuggingFace model (e.g., Qwen, Llama, Gemma variants) to MaxText. This includes analyzing the HuggingFace source code, implementing decoder layers in JAX/Flax, writing layer-wise unit tests with weight-copy/allclose checks, writing checkpoint conversion utilities for safetensors-to-orbax, and running comprehensive mini-model verification (forward, backward, multi-step optimizer, logits, and decode checks). Don't use for general MaxText training execution, standard inference, or debugging unrelated TPU compilation issues.
---

# MaxText New Model Bring-up Workflow

New model bring-up in MaxText is a structured 4-phase workflow designed to systematically port a PyTorch/HuggingFace model, implement it in JAX, and mathematically verify correctness using exact weight copies, comprehensive mini-model verification (forward, backward, multi-step optimization), logits validation, and autoregressive decoding.

Before beginning, review [docs/guides/model_bringup.md](../../../docs/guides/model_bringup.md) for the project's standard bring-up guide and [docs/reference/models/supported_models_and_architectures.md](../../../docs/reference/models/supported_models_and_architectures.md) to inspect existing supported architectures.

For a step-by-step checklist of the bring-up process and a registry explaining the origin of all hardcoded parameter configurations, see the [Implementation Roadmap & Hardcoded Parameters Reference](references/roadmap.md).

---

## Handover Document (`{model_name}_handover.md`)

To track progress and enable seamless collaboration across debugging sessions, maintain a live handover document in the MaxText root directory (`{model_name}_handover.md`).

Update this document at each phase of the bring-up:
1. **Original HF Model Learnings**: Key architectural findings (norm type/eps, RoPE theta, head dims, decoder patterns).
2. **Phase Status Tracking**: Mark each phase as `NOT STARTED`, `IN PROGRESS`, `BLOCKED`, or `DONE`.
3. **Bugs & Resolutions**: Log all bugs encountered during each phase and the exact fixes applied.
4. **Session Wrap-up & Pitfalls**: Summarize all model-specific pitfalls, gotchas, and caveats upon completion.
5. **Full Reproduction Commands**: Document exact CLI commands for mini safetensors slicing, checkpoint conversion, mini verification, decode, and forward pass logits checking.

---

## The 4-Phase Bring-up Workflow

### [Phase 1: HF Model Discovery & Architecture Analysis](references/hf_analysis.md)

- Identify the HuggingFace Model ID.
- **IMPORTANT**: Set `export HF_HOME=/dev/shm/$USER` before loading or downloading models to avoid filling up the TPU VM root disk.
- **Decoder Pattern Inspection**: Determine if layers are homogeneous (identical repeated blocks) or inhomogeneous (e.g., layers 0–2 dense followed by MoE, or alternating linear/full attention).
- **Identify Smallest Subset**: Determine the minimal representative layer count (e.g. 1 layer for homogeneous, 4 layers for a 4-layer cycle) plus any multimodal vision encoders or auxiliary heads (e.g. Multi-Token Prediction).
- **Create Mini Safetensors Subset**: Download/slice a mini safetensors directory in `/dev/shm/hf_mini/{model_name}_xlayers` containing only config, tokenizer, embeddings, output head, auxiliary/multimodal heads, and the smallest layer subset weights.
- Initialize `{model_name}_handover.md` in the MaxText root directory with HF findings and set Phase 1 status to `DONE`.
- For details, see [hf_analysis.md](references/hf_analysis.md).

### [Phase 2: JAX Layer Implementation & Verification](references/jax_layers.md)

- **Maximize Reuse**: Check what layers already exist in MaxText that can be configured/reused (e.g., in `src/maxtext/layers/attentions.py`).
- Implement new layers in JAX under `src/maxtext/layers/` or `src/maxtext/models/`.
- **Autoregressive Generation Support**: Ensure custom decoder layers properly support autoregressive decode (KV caching, causal mask updates during decode steps).
- Write layer-wise correctness unit tests under `tests/unit/{model_name}_layers_test.py` using **mini-layer sizes** (e.g. batch=2, seq=4, hidden=128) and assert output matches PyTorch via `assert_allclose`.
- **Reference**: Refer to existing layer tests such as `tests/unit/gemma3_layers_test.py` or `tests/unit/qwen3_5_layers_test.py` as implementation references.
- **Multimodal Preprocessing**: If multimodal (VLM/ALM), implement preprocessing in `src/maxtext/multimodal/processor_{model_name}.py` and test against HF processors.
- Add model config `src/maxtext/configs/models/{model_name}.yml`. Run a quick check via `maxtext.inference.decode` with random weights and `base_num_decoder_layers=1` to verify shapes.
- Update `{model_name}_handover.md` with Phase 2 status and any bugs/fixes.
- For details and templates, see [jax_layers.md](references/jax_layers.md).

### [Phase 3: Checkpoint Conversion Development](references/checkpoint_conversion.md)

- Turn layer weight-mapping rules from unit tests into central mappings in `src/maxtext/checkpoint_conversion/utils/param_mapping.py` (`PARAM_MAPPING` and `HOOK_FNS`) and `src/maxtext/checkpoint_conversion/utils/hf_shape.py`.
- **Always use central frameworks**: Extend `to_maxtext.py`, `to_huggingface.py`, and `slicing/slice_model.py`.
- **Fast Iteration on Mini Subset**: Run conversion first on the mini safetensors subset in `/dev/shm/hf_mini/{model_name}_xlayers` using real weights to save time and memory.
- Map and transform weights: load safetensors on CPU, **transpose** linear weights, reshape projections, and handle stack cycle indexing for scanned layers.
- Update `{model_name}_handover.md` with Phase 3 status and full conversion commands.
- For details, see [checkpoint_conversion.md](references/checkpoint_conversion.md).

### [Phase 4: Comprehensive Mini-Model Verification, E2E Logits & Decode Check](references/verification.md)

- **Mini-Model Verification Strategy (6 Core Dimensions + 4 Supplemental Checks)**:
  Use a representative trimmed version of the target model (sliced via `slice_model.py` and converted so both PyTorch/HF and MaxText load identical weights with `max_abs_diff == 0.0`) and verify parity across:
  1. **Design challenging inputs deliberately**: Test sequence/chunk/block boundaries, heterogeneous padded batches (right/left padding with attribution probes), packed-document boundaries (`segment_ids`), chat templates/special tokens, and multimodal/3D MRoPE grids.
  2. **Check forward outputs and masks at meaningful boundaries**: Compare both component-level intermediate outputs (ViT, spatial merger, post-scatter embeddings, sub-layer stages, norms) and complete small-model logits.
  3. **Give MoE / Dense its own coverage requirements**: For MoE, cover routing patterns deliberately (using forced routing if needed) and verify load-balancing loss; for Dense, verify SwiGLU/MLP sub-activations and confirm zero spurious MoE overhead.
  4. **Verify every enabled head and the actual training objective**: Inspect all enabled outputs (main LM head, MTP head, vision projector), expose loss components before aggregation (`main_ce_loss`, `z_loss`, `mtp_loss`, `moe_lb_loss`), and test objectives separately and together.
  5. **Make backward checks sensitive and independent**: Compare gradients per parameter group across all subsystems (`jax.value_and_grad` vs. PyTorch `.backward()`) and add isolated block-level upstream-gradient (VJP) probes.
  6. **Test actual multi-step updates and state propagation**: Run several real stateful optimizer updates (e.g., 3 steps of AdamW) with changing batches, compare weights and optimizer moments ($m, v$) at the end of the complete step, and verify trainable/frozen parameter selection.
  7. **Supplemental Production-Readiness Checks**: Verify `scan_layers=True` vs. `scan_layers=False` parity, asymmetric 3D MRoPE grids + negative controls, multi-chip 2D mesh sharding (`fsdp` + `tensor`), and `bfloat16` mixed-precision stability.
- **Standard Logits & Autoregressive Decode Verification**:
  - Run `forward_pass_logit_checker.py` using the mini-subset Orbax checkpoint against golden HF `.jsonl` logits (KL Divergence $\le 10^{-3}$).
  - Run 16-token autoregressive decode on both HF PyTorch and MaxText using real weights from the mini subset (must generate exact same token IDs and gibberish text).
- **Full Model Run & Handover Wrap-up**: Once the mini subset passes all checks, run full checkpoint conversion and finalize `{model_name}_handover.md`.
- For details, see [verification.md](references/verification.md).

---

## Critical Gotchas & Gotcha Resolution

> [!IMPORTANT]
> **Gotcha 1: TPU VM local storage space limits & Mini-Subset Speedup**.
> HuggingFace caching quickly fills VM disk. Always set `export HF_HOME=/dev/shm/$USER` and dump checkpoints to shared memory (e.g. `/dev/shm/$USER`).
> Create a mini safetensors subset in `/dev/shm/hf_mini/{model_name}_xlayers` to iterate rapidly without loading massive full-model weights.
> If verification fails, delete intermediate checkpoints immediately using `rm -rf`.

> [!IMPORTANT]
> **Gotcha 2: Transposing weights**.
> PyTorch linear weights are `(out_features, in_features)` while Flax/JAX expects `(in_features, out_features)`. Always transpose linear weights in hooks!

> [!IMPORTANT]
> **Gotcha 3: Checked/Scanned layers stack indexing**.
> Flax stacks scanned layers along a shared index. Ensure HF layer `l` maps to cycle index `l % cycle` and stack index `l // cycle`.

> [!WARNING]
> **Gotcha 4: RoPE permutations**.
> Rotary position embedding (RoPE) implementations in PyTorch models are often permuted differently than MaxText's default implementation. Use JAX `permute_to_match_maxtext_rope` helper functions to resolve discrepancies.

> [!IMPORTANT]
> **Gotcha 5: Autoregressive Decoder Support**.
> Ensure custom decoder layers implement KV cache state handling for autoregressive generation (`decode=True`), not just training/bidirectional modes.

> [!IMPORTANT]
> **Gotcha 6: Live Handover Documentation**.
> Always keep `{model_name}_handover.md` updated in the MaxText root directory at every phase so collaborators and future agents can inspect progress, reproduction commands, and resolved pitfalls.

> [!IMPORTANT]
> **Gotcha 7: TPU VM execution & command formatting**.
> Always execute debugging, decoding, checkpoint conversion, and verification python commands directly from the MaxText root directory on your TPU VM. Run commands using standard python module flags (e.g., `python3 -m maxtext.inference.decode`).
