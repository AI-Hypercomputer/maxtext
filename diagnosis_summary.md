# Diagnosis Summary: Gemma 3 Conversion and Decode Issues

This document summarizes the diagnosis and fixes for the Gemma 3 checkpoint conversion (`to-mt-ckpt`) and inference (`decode_gemma3_4b_mm`) issues on MaxText main.

## 1. Checkpoint Conversion (`to-mt-ckpt`) Failure

### Symptom
When running `to_maxtext.py` for Gemma 3, the process failed with:
```
ValueError: HuggingFace key model.language_model.embed_tokens.weight not found in state_dict.
```

### Root Cause
1.  **Lazy Loading Limitation**: The configuration in `launch.json` had `--lazy_load_tensors=True`. Lazy loading of Hugging Face tensors is not yet supported for multimodal models in MaxText and would fail.
2.  **Safetensors Key Mapping**: When switched to eager mode (`--lazy_load_tensors=False`), using `--eager_load_method=safetensors` failed because Gemma 3 requires Hugging Face `transformers` style key mapping, which was not compatible with the safetensors loader's assumptions.

### Solution
Updated the `to-mt-ckpt` configuration in `.vscode/launch.json` (at the projects root) to use:
*   `--eager_load_method=transformers`
*   `--lazy_load_tensors=False`

This allows the conversion to successfully complete using the Hugging Face transformers library to load and map the weights.

---

## 2. Decode (`decode_gemma3_4b_mm`) Failure

### Symptom
Running decode with the converted checkpoint failed during the prefill stage with:
```
RuntimeError: Array has been deleted with shape=float32[2560,10240].
```
The deleted array corresponded to MLP weights (e.g., `mlp.wi_0.kernel`).

### Root Cause
A structural mismatch existed between some Linen-style checkpoints and the NNX-style model. Specifically, there are two types of Linen checkpoints (checkpoints wrapped in `params.params`):

1.  **Legacy Linen Checkpoints** (converted before NNX migration): Lists of layers are flattened using underscores, e.g., `decoder.layers_0.mlp...`.
2.  **New Linen Checkpoints** (converted after NNX migration, e.g. Qwen3-VL): Lists of layers preserve the NNX dot-separated structure even inside the Linen wrapper, e.g., `decoder.layers.0.mlp...`. This is because `to_maxtext.py` wraps the NNX model in `ToLinen` for saving, which preserves the inner NNX structure.

When `pure_nnx=True` (default), the decode engine builds an NNX model. During `from_pretrained`, the template for restore is constructed from the NNX model state (using `layers.0` structure).

If loading a **Legacy Linen Checkpoint**:
*   The restore template (dots) did not match the checkpoint structure (underscores).
*   Orbax checkpointer silently skipped restoring the decoder layers.
*   The initial parameters (which were deleted from device memory to save RAM before restore) were never replaced with restored weights, leading to the `Array has been deleted` crash during execution.
*   The structure mismatch check in `from_pretrained` (which raises an error if >80% of parameters are missing) was bypassed because the **vision encoder** parameters still use underscores in both Linen and NNX structures (as they are not yet migrated to pure NNX). The matching vision encoder parameters kept the overall missing parameter ratio at ~50%, below the 80% threshold.

If loading a **New Linen Checkpoint** (like Qwen3-VL):
*   The restore template (dots) matched the checkpoint structure (dots).
*   Weights were restored successfully. This is why Qwen3-VL was **not impacted** initially.
*   A naive key translation fix that unconditionally converts dots to underscores would break these new checkpoints.

### Solution
Implemented a conditional structural translation mechanism in `maxtext/utils/model_creation_utils.py`:
1.  **Detection Helper**:
    *   `_is_legacy_linen_checkpoint(stored_tree)`: Recursively inspects the checkpoint metadata tree to detect if it contains any legacy underscore-style list keys (e.g., `layers_0`).
2.  **Translation Helpers**:
    *   `_nnx_to_linen_struct(tree)`: Converts NNX list structure (dot) to legacy Linen style (underscore).
    *   `_linen_to_nnx_struct(tree)`: Converts legacy Linen style (underscore) to NNX list structure (dot with integer keys).
3.  **Integration**:
    *   In `from_pretrained`, if loading a Linen checkpoint (`is_nnx_checkpoint=False`):
        *   Detect if it is legacy using `_is_legacy_linen_checkpoint`.
        *   If **legacy**: translate `target_for_restore` (NNX) to Linen structure before restore, and translate the `restored` checkpoint back to NNX structure after restore.
        *   If **new**: skip translation and load directly (matching by dots).

This ensures compatibility with both legacy checkpoints (e.g., older Gemma 3 runs) and newly converted checkpoints (e.g., Qwen 3, new Gemma 3 runs).

---

## Verification Results
All fixes were verified on the remote TPU VM:
1.  **Gemma 3 Conversion**: `to_maxtext.py` successfully converts `google/gemma-3-4b-it` using transformers eager load.
2.  **Gemma 3 Decode (Legacy Checkpoint)**: Successfully loads the legacy checkpoint (translating underscores to dots) and runs inference.
3.  **Qwen3-VL Decode (New Checkpoint)**: Successfully loads the new checkpoint (without translation) and runs inference.

