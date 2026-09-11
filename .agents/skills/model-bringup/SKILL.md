---
name: model-bringup
description: >-
  Performs new LLM/VLM model bring-up in MaxText on TPU VMs. Use when onboarding a new HuggingFace model (e.g., Qwen, Llama, Gemma variants) to MaxText. This includes analyzing the HuggingFace source code, implementing decoder layers in JAX/Flax, writing layer-wise unit tests with weight-copy/allclose checks, writing checkpoint conversion utilities for safetensors-to-orbax, and running the end-to-end forward pass logits checker. Don't use for general MaxText training execution, standard inference, or debugging unrelated TPU compilation issues.
---

# MaxText New Model Bring-up Workflow

New model bring-up in MaxText is a structured 4-phase workflow designed to systematically port a PyTorch/HuggingFace model, implement it in JAX, and mathematically verify correctness using exact weight copies, logits validation, and autoregressive decoding.

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
5. **Full Reproduction Commands**: Document exact CLI commands for mini safetensors slicing, checkpoint conversion, decode, and forward pass logits checking.

---

## The 4-Phase Bring-up Workflow

### [Phase 1: HF Model Discovery & Architecture Analysis](references/hf_analysis.md)

- Identify the HuggingFace Model ID.
- **IMPORTANT**: Set `export HF_HOME=/dev/shm/$USER` before loading or downloading models to avoid filling up the TPU VM root disk.
- **Decoder Pattern Inspection**: Determine if layers are homogeneous (identical repeated blocks) or inhomogeneous (e.g., layers 0–2 dense followed by MoE, or alternating local/global sliding window attention).
- **Identify Smallest Subset**: Determine the minimal representative layer count (e.g. 1 layer for homogeneous, 4 layers for a 4-layer cycle).
- **Create Mini Safetensors Subset**: Download/slice a mini safetensors directory in `/dev/shm/hf_mini/{model_name}_xlayers` containing only config, tokenizer, embeddings, output head, and the smallest layer subset weights.
- Initialize `{model_name}_handover.md` in the MaxText root directory with HF findings and set Phase 1 status to `DONE`.
- For details, see [hf_analysis.md](references/hf_analysis.md).

### [Phase 2: JAX Layer Implementation & Verification](references/jax_layers.md)

- **Maximize Reuse**: Check what layers already exist in MaxText that can be configured/reused (e.g., in `src/maxtext/layers/attentions.py`).
- Implement new layers in JAX under `src/maxtext/layers/`.
- **Autoregressive Generation Support**: Ensure custom decoder layers properly support autoregressive decode (KV caching, causal mask updates during decode steps).
- Write layer-wise correctness unit tests under `tests/unit/{model_name}_layers_test.py` using **mini-layer sizes** (e.g. batch=2, seq=4, hidden=128) and assert output matches PyTorch via `assert_allclose`.
- **Reference**: Refer to existing layer tests such as `tests/unit/gemma3_layers_test.py` or `tests/unit/gemma4_layers_test.py` as implementation references.
- **Multimodal Preprocessing**: If multimodal (VLM/ALM), implement preprocessing in `src/maxtext/multimodal/processor_{model_name}.py` and test against HF processors.
- Add model config `src/maxtext/configs/models/{model_name}.yml`. Run a quick check via `maxtext.inference.decode` with random weights and `base_num_decoder_layers=1` to verify shapes.
- Update `{model_name}_handover.md` with Phase 2 status and any bugs/fixes.
- For details and templates, see [jax_layers.md](references/jax_layers.md).

### [Phase 3: Checkpoint Conversion Development](references/checkpoint_conversion.md)

- Turn layer weight-mapping rules from unit tests into central mappings in `src/maxtext/checkpoint_conversion/utils/param_mapping.py` (`PARAM_MAPPING` and `HOOK_FNS`).
- **Always use central frameworks**: Extend `to_maxtext.py` and `to_huggingface.py`.
- **Fast Iteration on Mini Subset**: Run conversion first on the mini safetensors subset in `/dev/shm/hf_mini/{model_name}_xlayers` using real weights to save time and memory.
- Map and transform weights: load safetensors on CPU, **transpose** linear weights, reshape projections, and handle stack cycle indexing for scanned layers.
- Update `{model_name}_handover.md` with Phase 3 status and full conversion commands.
- For details, see [checkpoint_conversion.md](references/checkpoint_conversion.md).

### [Phase 4: E2E Logits Validation & Autoregressive Decode Check](references/logits_checker.md)

- **Logits Verification**: Run `forward_pass_logit_checker.py` using the mini-subset Orbax checkpoint against golden HF `.jsonl` logits (KL Divergence $\\le 10^{-3}$).
- **Autoregressive Decode Verification**: Run full autoregressive decode to generate ~16 tokens on both HF PyTorch and MaxText using real weights from the mini subset.
  - **Success Criteria**: Because weights are from a truncated model subset, output text will be gibberish, but **MaxText and HF must generate the exact same token IDs and gibberish text**.
- **Deep Profiling via Activations Mapping**: If outputs diverge, run HF and MaxText step-by-step with `layers=1` and compare intermediate layer activations.
- **Full Model Run**: Once mini subset passes, run full checkpoint conversion and full logits verification.
- **Wrap Up Handover**: Finalize `{model_name}_handover.md` with all pitfalls found and full reproduction commands.
- For details, see [logits_checker.md](references/logits_checker.md).

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
> Always execute debugging, decoding, checkpoint conversion, and logits checking python commands directly from the MaxText root directory on your TPU VM. Run commands using standard python module flags (e.g., `python3 -m maxtext.inference.decode`).
