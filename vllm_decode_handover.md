# Handover: Multimodal Support in vLLM Decode for Qwen3-VL

## Implementation Details

We integrated multimodal support for `qwen3-vl-2b` in the MaxText vLLM adapter (`MaxTextForCausalLM` in `adapter.py`).

1.  **Multimodal Embeddings (`embed_multimodal`)**:
    *   Implemented `embed_multimodal` to extract `pixel_values` and `image_grid_thw` from inputs.
    *   Reconstructed the 5D tensor `(1, channel, T, H, W)` from the flattened 2D `pixel_values` `(total_patches, feature_dim)` using direct reshaping. This matches the behavior of MaxText's internal preprocessor (`preprocess_mm_data_qwen3_omni`).
    *   Invoked `self.model.vision_encoder` to obtain the visual embeddings.

2.  **JIT-Friendly Inputs Merge (`embed_input_ids`)**:
    *   We cannot use `tpu_runner`'s external merging directly because the MaxText model does not support `inputs_embeds` in its core interface (it always embeds tokens internally).
    *   To resolve this, we implemented a dynamic override in `Embed.__call__` (the embedding layer of the model).
    *   We dynamically added `active_inputs_embeds` and `use_inputs_embeds` as `nnx.Variable` state fields to `token_embedder` during `load_weights`.
    *   Class-patched `Embed.__call__` to check `use_inputs_embeds` and return `active_inputs_embeds` using `jax.lax.cond`.
    *   During the forward pass (`__call__`), if `inputs_embeds` is passed from the runner, we recreate the `active_inputs_embeds` variable with the new shape, cast it to the model's dtype (`bfloat16`), and set `use_inputs_embeds` to `True`.
    *   If `inputs_embeds` is not present (text-only steps or decode steps), we reset `use_inputs_embeds` to `False` and reset `active_inputs_embeds` shape to `(1, 1, dim)` to avoid JIT compilation errors in the unused branch of `lax.cond`.
    *   The patch is bypassed for 1D token inputs (used in `embed_input_ids`) by checking `x.ndim == 2`, which ensures compatibility with text-only paths.

## Pitfalls & Solutions

*   **JIT Capturing Global Variables**: Initial attempt to use global variables for override failed because JAX JIT captures global values during tracing and optimizes away branches. Solution was to use `nnx.Variable` which is tracked as part of the model state and seen as dynamic by the compiler.
*   **Compilation Shape Mismatch**: JAX `lax.cond` requires both branches to return the same shape. When `use_inputs_embeds` was False (e.g. during text-only runs), the unused `true_fn` branch still traced `active_inputs_embeds` (shape `(prefill_tokens, 1, dim)` from previous run), which failed to broadcast to the new smaller `x` shape `(1, 1, dim)` in the text run. Solution was to recreate `active_inputs_embeds` with shape `(1, 1, dim)` when disabling the override, and use `jnp.broadcast_to` in `true_fn` to allow dynamic broadcasting from `1` to `num_tokens`.
*   **Dtype Mismatch**: The merged embeddings from the runner were in `float32`, while the model embedding weights were `bfloat16`, causing `lax.cond` type mismatch. Solution was to cast the embeddings to `self.maxtext_config.dtype` before storing.
*   **Direct Reshape**: Attempting to permute the 2D tensor back to 5D using transposition math resulted in scrambled images because the model (or preprocessor) had already aligned the layout. Solution was to use a direct `reshape` to 5D matching `preprocess_mm_data_qwen3_omni`.

## Running Instructions

Run the following command on the TPU VM to execute the multimodal decode:

```bash
cd /home/hengtaoguo_google_com/projects/maxtext && \
PYTHONPATH=src:src/maxtext/integration/vllm /home/hengtaoguo_google_com/projects/venv3/bin/python3 src/maxtext/inference/vllm_decode.py \
  src/maxtext/configs/base.yml \
  run_name=vllm_decode_qwen3_vl \
  model_name=qwen3-vl-2b \
  load_parameters_path=/home/hengtaoguo_google_com/projects/checkpoints/qwen3-vl-2b/unscanned/2026-07-01-11-55/0/items \
  tokenizer_path=Qwen/Qwen3-VL-2B-Instruct \
  prompt="Describe this image: " \
  image_path=/home/hengtaoguo_google_com/projects/maxtext/tests/assets/test_image.jpg \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  per_device_batch_size=1 \
  scan_layers=false \
  weight_dtype=bfloat16 \
  max_num_batched_tokens=16384 \
  use_chat_template=true \
  vllm_hf_overrides='{"architectures":["MaxTextForCausalLM"]}'
```

You can also use the newly added VSCode debug configuration `"vllm_decode_qwen3_vl"`.

> [!NOTE]
> The VSCode debug configuration includes `PYTHONPATH` in its `env` settings to force the loader to use the local modified adapter (`src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py`) instead of the pre-installed version in the virtual environment (`venv3`).

