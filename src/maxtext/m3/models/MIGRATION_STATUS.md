# `m3` Model Garden Migration Status

This tracker records the 4-stage cutover governance status of every model family under `src/maxtext/m3/models/`.

## Stage Definitions

* **Stage 0 — Scaffolding:** Typed `<Family>Config` preset and pure-NNX model registered in `MODEL_GARDEN_REGISTRY` and `MODEL_REGISTRY`; passes `< 200ms` `verify_m3_model_invariants` and single-chip unit tests.
* **Stage 1 — Parity Certified:** Passes all 4 automated pre-training parity gates (`cutover_parity_suite_test.py`):
  1. Forward logit and backward parameter-gradient equivalence (`rel_frob <= 1e-5` fp32, `<= 1e-3` bf16) + `from_hf_config` preset schema verification.
  2. Packed-vs-unpacked document forward and gradient equivalence (`max |delta| <= 1e-5` fp32).
  3. Mid-run Orbax + stateful data iterator checkpoint save/resume bit-equivalence (`|delta loss| <= 1e-6`) + cross-layout checkpoint restore.
  4. Offline CPU `M3Trainer.compile_aot` static HBM verification (`< 90s`).
* **Stage 2 — Default Cutover:** Added to `DEFAULT_M3_MODELS` (`use_m3_model="auto"` resolves to `True`) with 30-day nightly CI soak on `use_m3_model=false`.
* **Stage 3 — Sunset Complete:** Superseded files under `src/maxtext/models/`, `src/maxtext/layers/`, and `src/maxtext/configs/models/*.yml` retired.

## Family Tracker

| Cohort | Model Family | Presets in `MODEL_GARDEN_REGISTRY` | Current Stage | `DEFAULT_M3_MODELS` | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cohort 1 | `qwen3` (Dense GQA) | `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`, `qwen3-mini` | Stage 1 (CPU gates passing) | `False` | Pure NNX `nnx.Sequential` + `DefaultSharding` + `CausalLMOutput` + `ScannedLayers` + `remat` |
| Cohort 1 | `qwen3-vl` (Multimodal VLM) | `qwen3-vl-2b` (`MODEL_REGISTRY`) | Stage 0 | `False` | `nnx.Sequential` decoder layer iteration aligned; `Qwen3VLConfig` preset pending |
| Cohort 2 | `glm5.3-flash` (Hybrid KDA + MLA + MoE) | `glm5.3-flash` (`MODEL_REGISTRY`) | Stage 0 | `False` | Shared `m3.core.layers` segment-isolation primitives ready |
| Cohort 2 | `qwen3.8-flash-next` (Hybrid GDN + QSA + PLE) | `qwen3.8-flash-next`, `qwen3.8-flash-next-mini` (`MODEL_REGISTRY`) | Stage 0 | `False` | Shared `m3.core.layers` segment-isolation primitives ready |
| Cohort 3 | `deepseek-v3` (MLA + Aux-Free MoE) | `deepseek3-671b`, `deepseek3-mini` (`DeepSeekV3Config`) | Stage 0 | `False` | Config preset + `RouterBias` step ready; model constructor port in Cohort 3 |
