# Phase 3: Layer-by-Layer Transformation

## Objective
Specify the necessary transformation logic for every individual layer. Pass `layer_shape_verify.py` for shape matching and then `layerwise_verify.py` for layer-by-layer check.

## Result
Development of a new `{MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN` function integrated into `param_mapping.py`. This involves transposing weight matrices (e.g., Hugging Face `[out, in]` to MaxText `[in, out]`), dealing with concatenated query/key/value weights, handling grouped-query attention formats, and any custom scaling or permutation requirements.

## Command to run script:
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu /home/yixuannwang_google_com/maxtext_venv/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/layer_shape_verify.py \
--hf_path google/gemma-3-4b-it --model_name gemma3-4b --scan_layers`


`TMPDIR=/dev/shm JAX_PLATFORMS=cpu /home/yixuannwang_google_com/maxtext_venv/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/layerwise_verify.py \
--hf_path google/gemma-3-4b-it --model_name gemma3-4b --scan_layers`