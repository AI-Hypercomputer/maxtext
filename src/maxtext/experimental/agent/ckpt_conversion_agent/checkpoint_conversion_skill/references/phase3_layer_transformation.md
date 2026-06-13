# Phase 3: Layer-by-Layer Transformation

## GOAL
Specify the necessary transformation logic for every individual layer. This involves transposing weight matrices (e.g., Hugging Face `[out, in]` to MaxText `[in, out]`), dealing with concatenated query/key/value weights, handling grouped-query attention formats, and any custom scaling or permutation requirements, using (`<model_family>_tracing.json` file from phase 1) and `{MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING` function from phase 2.

## Inputs from Previous Phases
- **`<model_family>_tracing.json`** (from Phase 1): Used to determine the exact tensor shapes for transposing weight matrices.
- **`{MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING`** (from Phase 2): Used as the base parameter mapping.

## Expected Outcome / Outputs Generated
- **`{MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN` function**: Developed and integrated into [`param_mapping.py`](../../../../../checkpoint_conversion/utils/param_mapping.py). Pass `layer_shape_verify.py` for shape matching and then `layerwise_verify.py` for layer-by-layer check. This hook function, along with the mapping function, is used in Phase 4.
  - **Note on concatenated outputs:** For models where HuggingFace uses fused QKV (`qkv_proj`) or MLP gate/up (`gate_up_proj`) weights, PyTorch's `forward_hook` will extract the full concatenated tensor. MaxText treats `query`, `key`, `value`, `wi_0` (gate), and `wi_1` (up) as distinct layers. When running `layerwise_verify.py`, you **MUST** dynamically slice PyTorch's `pt_out` within the `verify_layer` function to isolate the respective chunks before performing mathematical equivalence checks against MaxText outputs.
- **MaxText checkpoint**: If successful, a MaxText checkpoint may be generated during `to_maxtext.py` which can be used in Phase 4.

## Commands to run scripts:

**1. Verify the {MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN correctly handles the shape:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/layer_shape_verify.py \
--hf_path google/gemma-3-4b-it --model_name gemma3-4b --disable_trust_remote_code`

**2. Convert from Hugging Face to MaxText:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
-m maxtext.checkpoint_conversion.to_maxtext \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
base_output_directory=<maxtext_checkpoint_output_dir> \
hf_access_token=${HF_TOKEN} \
hardware=cpu skip_jax_distributed_system=True scan_layers=False`

**3. Ensure numeric correctness:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/layerwise_verify.py \
--hf_path google/gemma-3-4b-it --mt_path <path-to-maxtext-checkpoint> --model_name gemma3-4b --scan_layers --disable_trust_remote_code`

Note that you should set `--scan_layers` and `--use_multimodal` when possible. **Important:** If your model does not implement a dedicated `ScannableBlockToLinen` in `decoders.py:get_decoder_layers()`, you **must** append `scan_layers=False` to the `to_maxtext.py` command (as shown above). Also, use `--disable_trust_remote_code` to bypass Hugging Face's `trust_remote_code=True` bug when working with modern transformers versions. 

Note on `layerwise_verify.py`: Passing this script ONLY guarantees that linear projection matrices and parameter shapes are correctly mapped and scaled. It DOES NOT guarantee that the MaxText network graph is built correctly. Architectural bugs (like RoPE mismatch or activation mismatch) will only be caught in Phase 4.