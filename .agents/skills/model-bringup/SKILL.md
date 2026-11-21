---
name: model-bringup
description: >-
  Performs new LLM/VLM model bring-up in MaxText on TPU VMs. Use when onboarding a new HuggingFace model (e.g., Qwen, Llama, Gemma variants) to MaxText. This includes analyzing the HuggingFace source code, implementing decoder layers in JAX/Flax, writing layer-wise unit tests with weight-copy/allclose checks, running mini-model-verification, verifying sequence-packing training parity, writing checkpoint conversion utilities for safetensors-to-orbax, and running multi-token masked prefill verification. Don't use for general MaxText training execution, standard inference, or debugging unrelated TPU compilation issues.
---

# MaxText New Model Bring-up Workflow

New model bring-up in MaxText is a structured 4-phase workflow designed to systematically port a PyTorch/HuggingFace model, implement it in JAX, and mathematically verify correctness using exact weight copies, mini-model-verification, sequence-packing training isolation, and multi-token masked prefill.

Before beginning, review [docs/guides/model_bringup.md](../../../docs/guides/model_bringup.md) for the project's standard bring-up guide and [docs/reference/models/supported_models_and_architectures.md](../../../docs/reference/models/supported_models_and_architectures.md) to inspect existing supported architectures.

For a step-by-step checklist of the bring-up process and a registry explaining the origin of all hardcoded parameter configurations, see the [Implementation Roadmap & Hardcoded Parameters Reference](references/roadmap.md).

---

## Handover Document (`{model_name}_handover.md`)

Maintain `{model_name}_handover.md` in the MaxText root directory as a single auditable record so reviewers can immediately verify all evidence is present (see template in [roadmap.md](references/roadmap.md)):
1. **HF Model Architectural Findings & Config Table**: Key specs (layer types/scanned structure, attention/RoPE/sliding window, MLP/MoE, norms, head/soft-cap) and reference links.
2. **Verification & Evidence Summary Table**: Status, key metrics, and log/report links for every task at a glance.
3. **Per-Stage Results, Commands & Logs**: For each stage (unit tests, `scanned`/`unscanned` conversion, `mini-model-verification`, sequence-packing training parity, multi-token masked prefill, training/SFT checks, and benchmark/roofline), record the result metrics table, exact CLI reproduction command, and log/checkpoint/golden paths together.
4. **Bugs & Pitfalls**: Key issues resolved and model-specific gotchas.

---

## Executable Verification Scripts (`scripts/`)

Run every script from the MaxText repository root (`--help` prints full CLI usage):

| Phase | Script | Purpose |
| :--- | :--- | :--- |
| **Phase 1 (HF Discovery)** | [scripts/inspect_hf_artifacts.py](scripts/inspect_hf_artifacts.py) | Parses `config.json`, `*.safetensors` headers (without loading weights), and reference `.py` ASTs for precision/quantization contracts |
| **Phase 1 (Reuse Diff)** | [scripts/family_diff.py](scripts/family_diff.py) | Compares closest supported family reference classes/configs against the target model and verifies reuse claims |
| **Phase 1 (Mini Slice)** | [scripts/slice_hf_mini.py](scripts/slice_hf_mini.py) | Slices a real HuggingFace checkpoint into a layer-renumbered mini-twin with `mini_manifest.json` SHA-256 digests |
| **Phase 2 (Static Audit)** | [scripts/check_jax_numerics_and_scale.py](scripts/check_jax_numerics_and_scale.py) | Audits JAX layer `.py` and `.yml` files for `bfloat16` router/RoPE traps, post-`exp` masking NaNs, unmasked segment reductions, and full-scale tensor OOMs |
| **Phase 2 (Goldens & Parity)** | [scripts/capture_golden.py](scripts/capture_golden.py), [scripts/layer_parity.py](scripts/layer_parity.py), [scripts/negctl.py](scripts/negctl.py) | Captures PyTorch hook activations with SHA-256 manifests; checks `fp32`/`fp64` forward + all-leaf gradient parity paired with `REL_NORM_BAND` single-leaf negative controls |
| **Phase 3 (Checkpoints)** | [scripts/ckpt_roundtrip_check.py](scripts/ckpt_roundtrip_check.py), [scripts/verify_checkpoint.py](scripts/verify_checkpoint.py), [scripts/audit_ckpt_vs_hf.py](scripts/audit_ckpt_vs_hf.py) | Verifies lossless `HF -> MaxText -> HF` key/shape/dtype/value round-trips (with negative controls), OCDbt manifest/chunk completeness, and routed-expert tensor slices against HF safetensors |
| **Phase 4 (E2E & Packing)** | [scripts/e2e_check.py](scripts/e2e_check.py), [scripts/packing_train_check.py](scripts/packing_train_check.py) | Checks E2E logit KL / token match / MoE `RouterAgreementTol` with negative controls, and verifies **packed-sequence forward isolation, per-leaf gradient superposition, multi-step Optax training parity, and `segment_ids=1` mutant-kill negative controls** |
| **Perf & TPU Preflight** | [scripts/aot_preflight.py](scripts/aot_preflight.py), [scripts/check_hlo_perf.py](scripts/check_hlo_perf.py), [scripts/extract_run_metrics.py](scripts/extract_run_metrics.py) | Compiles target TPU topologies ahead of time against `TOPOLOGY_HBM_GIB` limits, gates static HLO collective overlap & FLOP inflation, and extracts deterministic trainer step metrics |

---

## The 4-Phase Bring-up Workflow

### [Phase 1: HF Model Discovery & Architecture Analysis](references/hf_analysis.md)

- Identify the HuggingFace Model ID.
- **IMPORTANT**: Set `export HF_HOME=/dev/shm/$USER` before loading or downloading models to avoid filling up the TPU VM root disk.
- **Deterministic Artifact & Family Diff Inspection**: Run `python3 .agents/skills/model-bringup/scripts/inspect_hf_artifacts.py --hf-dir <hf_dir> --out /tmp/hf_contract.json` and `scripts/family_diff.py` to audit `config.json`, safetensors dtypes, and reusable components.
- **Decoder Pattern Inspection**: Determine if layers are homogeneous (identical repeated blocks) or inhomogeneous (e.g., layers 0–2 dense followed by MoE, or alternating local/global sliding window attention).
- **Identify Smallest Subset**: Determine the minimal representative layer count (e.g. 1 layer for homogeneous, 4 layers for a 4-layer cycle).
- **Create Mini Safetensors Subset**: Slice a mini safetensors directory in `/dev/shm/hf_mini/{model_name}_xlayers` via `scripts/slice_hf_mini.py` containing only config, tokenizer, embeddings, output head, and the smallest layer subset weights.
- Initialize `{model_name}_handover.md` in the MaxText root directory with HF findings and set Phase 1 status to `DONE`.
- For details, see [hf_analysis.md](references/hf_analysis.md).

### [Phase 2: JAX Layer Implementation & Verification](references/jax_layers.md)

- **Maximize Reuse**: Check what layers already exist in MaxText that can be configured/reused (e.g., in `src/maxtext/layers/attentions.py`).
- Implement new layers in JAX under `src/maxtext/layers/` and audit precision/masking invariants with `scripts/check_jax_numerics_and_scale.py`.
- **Autoregressive Generation Support**: Ensure custom decoder layers properly support autoregressive decode (KV caching, causal mask updates during decode steps).
- Write layer-wise correctness unit tests under `tests/unit/{model_name}_vs_reference_test.py` using **mini-layer sizes** (e.g. batch=2, seq=4, hidden=128), verifying forward and per-leaf gradient parity plus single-leaf perturbation negative controls via `scripts/layer_parity.py` and `scripts/negctl.py` (guarding `torch` with method-level `@unittest.skipUnless(HAS_TORCH, ...)` so torch-free CPU CI collects and runs pure-JAX tests).
- **Reference**: Refer to existing reference-parity tests such as `tests/unit/qwen3_next_vs_reference_test.py` or `tests/unit/hy4_vs_reference_test.py` as implementation references.
- **Multimodal Preprocessing**: If multimodal (VLM/ALM), implement preprocessing in `src/maxtext/multimodal/processor_{model_name}.py` and test against HF processors.
- Add model config `src/maxtext/configs/models/{model_name}.yml`. Run a quick check via `maxtext.inference.decode` with random weights and `base_num_decoder_layers=1` to verify shapes.
- Update `{model_name}_handover.md` with Phase 2 status and any bugs/fixes.
- For details and templates, see [jax_layers.md](references/jax_layers.md).

### [Phase 3: Checkpoint Conversion Development](references/checkpoint_conversion.md)

- Turn layer weight-mapping rules from unit tests into central mappings in `src/maxtext/checkpoint_conversion/utils/param_mapping.py` (`PARAM_MAPPING` and `HOOK_FNS`).
- **Always use central frameworks**: Extend `to_maxtext.py` and `to_huggingface.py`.
- **Fast Iteration on Mini Subset**: Run conversion first on the mini safetensors subset in `/dev/shm/hf_mini/{model_name}_xlayers` using real weights to save time and memory, and verify lossless `HF -> MaxText -> HF` bijectivity and negative controls with `scripts/ckpt_roundtrip_check.py`.
- Map and transform weights: load safetensors on CPU, **transpose** linear weights, reshape projections, and handle stack cycle indexing for scanned layers. Validate full converted OCDbt checkpoints with `scripts/verify_checkpoint.py` and `scripts/audit_ckpt_vs_hf.py`.
- Update `{model_name}_handover.md` with Phase 3 status and full conversion commands.
- For details, see [checkpoint_conversion.md](references/checkpoint_conversion.md).

### [Phase 4: Mini-Model-Verification & Multi-Token Masked Prefill](references/logits_checker.md)

- **Mini-Model-Verification**: Validate a representative trimmed version of the target model against PyTorch (see [mini_verification.md](references/mini_verification.md) and `scripts/e2e_check.py`).
- **Sequence-Packing Training Verification (Mandatory)**: Run `PYTHONPATH=$PWD/src python3 .agents/skills/model-bringup/scripts/packing_train_check.py --model-name={model_name}` (or call `verify_packed_training_parity(...)` in unit tests; for `m3` models also run `tests/unit/m3/conformance` Check 9b `PACKING_AND_PADDING_ISOLATION`) to verify packed vs. standalone forward isolation, per-leaf gradient superposition, zero cross-doc/padding input-embedding gradients, multi-step Optax training trajectory parity, and the `decoder_segment_ids=1` mutant-kill negative control.
- **Multi-Token Masked Prefill**: Convert full `scanned` (`scan_layers=true`) and `unscanned` (`scan_layers=false`) checkpoints and verify multi-token masked prefill parity via `zero_gen_logits_checker.py` / `forward_pass_logit_checker.py` (using router replay and unforced checks for MoE).
- **Training & SFT Check** *(if applicable)*: Run short pre-training and SFT smoke checks on both `scanned` and `unscanned` checkpoints.
- **Deep Profiling via Activations Mapping**: If outputs diverge, capture intermediate activations via `scripts/capture_golden.py` and compare HF and MaxText step-by-step with `layers=1`.
- **Wrap Up Handover**: Finalize `{model_name}_handover.md` with all pitfalls found and full reproduction commands.
- For details, see [logits_checker.md](references/logits_checker.md) and [mini_verification.md](references/mini_verification.md).

---

## [FLOP Calculation, Training Benchmarking & Roofline Analysis](references/performance_and_roofline.md)

- **FLOP Calculation**: Update `calculate_tflops_training_per_device` in `src/maxtext/utils/maxtext_utils.py` (with unit tests) for custom attention, latent/routed MoE, or head projections.
- **Ahead-of-Time TPU HBM Preflight & Static HLO Gate**: Before launching TPU cluster runs, run `scripts/aot_preflight.py` against the target `compile_topology` and audit static HLO overlap/FLOP bounds with `scripts/check_hlo_perf.py`.
- **Training Benchmarking**: Run a steady-state synthetic pre-training benchmark on TPU (`profiler=xplane`), extract deterministic steady-state metrics with `scripts/extract_run_metrics.py`, and record step time, `TFLOP/s/device` (MFU), `Tokens/s/device`, compile-time HBM estimates, and HLO memory reports.
- **Roofline Analysis**: Generate an analytical roofline report (`roofline_cli.py`, `jasmine` backend) and compare aggregate/per-block bounds and HBM budgets against measured MaxText performance. See [performance_and_roofline.md](references/performance_and_roofline.md).

---

## [Bring-up Troubleshooting Wiki](references/wiki/index.md)

When verification or conversion hits resource bottlenecks (such as TPU HBM OOM on single TPU VMs or multi-host TPU clusters), consult [references/wiki/index.md](references/wiki/index.md) (e.g., [hbm_verification.md](references/wiki/hbm_verification.md)) and merge newly discovered reusable workarounds back into the wiki.

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

---

## Optional: Large PR Cleanup & Atomic Split

`pr_cleanup=false` by default. Only an explicit user request for cleanup/splitting or `pr_cleanup=true` enables this phase; a large diff or passing validation does not. Otherwise, skip it without loading the reference.

After Phase 4 passes, follow [Large PR Cleanup & Atomic Split](references/pr_cleanup.md) for artifact removal, full-diff formatting, pylint fixes, and atomic PR stacking. This is separate from the four correctness phases.
