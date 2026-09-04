# Qwen3.8-27B Bring-up Handover Document

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: `Qwen/Qwen3.8-27B`
- **Decoder Pattern**: Inhomogeneous 4-layer cycle (`inhomogeneous_layer_cycle_interval: 4`: 3 Gated Delta Network `linear_attention` layers + 1 gated `full_attention` layer), repeated 16 times (64 layers total) with dense SwiGLU MLP (`num_experts: 1`).
- **Minimal Subset Layer Count ($N$)**: 4 layers (one complete 3 GDN + 1 Full Attention cycle).
- **Key Parameters**:
  - `vocab_size`: 248,320
  - `base_emb_dim` (`hidden_size`): 5,120
  - `base_mlp_dim` (`intermediate_size`): 17,408
  - `base_num_decoder_layers` (`num_hidden_layers`): 64
  - `base_num_query_heads` (`num_attention_heads`): 24
  - `base_num_kv_heads` (`num_key_value_heads`): 4
  - `head_dim`: 256
  - `partial_rotary_factor`: 0.25 (rotary dim = 64)
  - `rope_max_timescale` (`rope_theta`): 10,000,000.0
  - `normalization_layer_epsilon` (`rms_norm_eps`): 1e-06
  - `gdn_conv_kernel_dim` (`linear_conv_kernel_dim`): 4
  - `gdn_key_head_dim` (`linear_key_head_dim`): 128
  - `gdn_value_head_dim` (`linear_value_head_dim`): 128
  - `gdn_num_key_heads` (`linear_num_key_heads`): 16
  - `gdn_num_value_heads` (`linear_num_value_heads`): 48 (3:1 V-to-K head ratio)
  - `use_qk_norm_in_gdn`: True

## 2. Phase-by-Phase Status
- [x] Phase 1: HF Discovery & Mini Safetensors Creation (DONE)
- [x] Phase 2: JAX Layer Implementation & Layer Tests (DONE)
- [x] Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion (DONE)
- [x] Phase 4: E2E Logits & 16-Token Autoregressive Decode Verification (DONE)

## 3. Bugs Encountered & Solutions
- **Phase 2 (JAX Layers & KV Cache Dtype Conflict)**:
  - *Issue*: The original `fd3e241fd` commit forced `cached_prefill_key` in `src/maxtext/inference/kvcache.py` to `dtype=jnp.float32`, which conflicted with `origin/main` (`src/maxtext/models/qwen3.py` casting `next_recurrent_state` to `cfg.dtype`), causing `TypeError: lax.dynamic_update_slice requires arguments to have the same dtypes, got float32, bfloat16` during `maxtext.inference.decode`.
  - *Fix*: Reverted `src/maxtext/inference/kvcache.py` to use `dtype=dtype`, matching `qwen3.py`.
- **Phase 2 (Missing Unit Tests & Config Fields)**:
  - *Issue*: `qwen3.8-27b` was missing layer-wise PyTorch-vs-JAX equivalence tests (`tests/unit/qwen3_5_layers_test.py`) and config/mapping/FLOP unit test coverage (`configs_test.py`, `param_mapping_test.py`, `flop_calculation_test.py`), as well as `logits_via_embedding`, `normalize_embedding_logits`, and `tokenizer_type` in `qwen3.8-27b.yml`.
  - *Fix*: Added `TestQwen3_8_27BDecoderLayers` in `tests/unit/qwen3_5_layers_test.py` (verifying both GDN linear-attention + dense MLP at `layer_idx=0` with max diff `9.24e-7` and full-attention + dense MLP at `layer_idx=3` with max diff `3.58e-7`), and registered `qwen3.8-27b` in `configs_test.py`, `param_mapping_test.py`, and `flop_calculation_test.py`.
- **Phase 3 (Reverse Checkpoint Conversion `to_huggingface.py`)**:
  - *Issue*: `qwen3.8-27b` was missing from `HF_SHAPE` in `src/maxtext/checkpoint_conversion/utils/hf_shape.py`, which is required by `to_huggingface.py` on current `main`.
  - *Fix*: Registered `"qwen3.8-27b": QWEN3_5_HF_WEIGHTS_TO_SHAPE` in `hf_shape.py`; verified lossless round-trip `HF -> MaxText (scan_layers=True) -> HF` (`max_abs_diff = 0.0` across all 56 tensors).

## 4. Pitfalls & Model-Specific Gotchas (Session Wrap-up)
- **Interleaved GDN Projections (`V_per_K = 3`)**: `Qwen/Qwen3.8-27B` has `linear_num_key_heads = 16` and `linear_num_value_heads = 48` (`V_per_K = 3`). `in_proj_qkvz` and `in_proj_ba` must be interleaved per key-head via `concat_qkvz_and_transpose` and `concat_ba_and_transpose`.
- **Single-Host Execution on Multi-Host TPU Slices**: When running verification on a single host of a multi-host TPU slice (e.g., `v6e-16`), set `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 skip_jax_distributed_system=True`.

## 5. Full Executable Reproduction Commands
- **Mini Checkpoint Conversion (`to_maxtext`)**:
  ```bash
  export HF_HOME=/dev/shm/$USER
  python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=qwen3.8-27b base_num_decoder_layers=4 override_model_config=true \
    scan_layers=false base_output_directory=/dev/shm/$USER/checkpoints/qwen3.8-27b_mini_orbax \
    hardware=cpu skip_jax_distributed_system=True \
    --hf_model_path=/dev/shm/hf_mini/qwen3.8-27b_4layers
  ```
- **Round-Trip MaxText-to-HF Conversion (`to_huggingface`)**:
  ```bash
  export HF_HOME=/dev/shm/$USER
  python3 -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
    model_name=qwen3.8-27b base_num_decoder_layers=4 override_model_config=true \
    scan_layers=true load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_mini_scanned/0/items \
    base_output_directory=/dev/shm/$USER/checkpoints/qwen3.8-27b_mini_hf_rt \
    hardware=cpu skip_jax_distributed_system=True --override_model_architecture
  ```
- **Forward Pass Logits Check**:
  ```bash
  export HF_HOME=/dev/shm/$USER TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3
  python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers \
    load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_mini_orbax/0/items \
    model_name=qwen3.8-27b base_num_decoder_layers=4 override_model_config=true \
    per_device_batch_size=1 scan_layers=false dtype=bfloat16 weight_dtype=bfloat16 \
    attention=dot_product max_prefill_predict_length=16 max_target_length=16 \
    skip_jax_distributed_system=True --run_hf_model=true \
    --hf_model_path=/dev/shm/hf_mini/qwen3.8-27b_4layers --max_kl_div=0.01
  ```
- **Autoregressive Decode Check (16 Tokens)**:
  ```bash
  export HF_HOME=/dev/shm/$USER TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3
  python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    model_name=qwen3.8-27b tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers \
    tokenizer_type=huggingface \
    load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_mini_orbax/0/items \
    run_name=qwen3-8-27b-mini-decode base_num_decoder_layers=4 override_model_config=true \
    per_device_batch_size=1 max_prefill_predict_length=16 max_target_length=32 \
    steps=1 scan_layers=false dtype=bfloat16 weight_dtype=bfloat16 \
    attention=dot_product skip_jax_distributed_system=True prompt='I love to'
  ```
