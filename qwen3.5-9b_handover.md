# Qwen/Qwen3.5-9B Bring-up Handover Document

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: `Qwen/Qwen3.5-9B`
- **Decoder Pattern**: Inhomogeneous cyclic pattern with cycle interval = 4 (Layers 0, 1, 2 are Linear Attention via GatedDeltaNet; Layer 3 is Full Attention with output sigmoid gate). All layers use standard dense SwiGLU MLP (no MoE).
- **Minimal Subset Layer Count**: 4 layers (1 full cycle: layers 0..3).
- **Mini Safetensors Directory**: `/dev/shm/hf_mini/qwen3.5_9b_4layers` (Contains sliced config, tokenizer, embeddings, layers 0..3, norm, lm_head; verified working in PyTorch forward pass & generation).
- **Key Parameters (Text Trunk)**:
  - `vocab_size`: 248320
  - `hidden_size` (`base_emb_dim`): 4096
  - `num_hidden_layers` (`base_num_decoder_layers`): 32 (sliced to 4 for mini-checkpoint)
  - `num_attention_heads` (`base_num_query_heads`): 16
  - `num_key_value_heads` (`base_num_kv_heads`): 4
  - `head_dim`: 256
  - `intermediate_size` (`base_mlp_dim`): 12288
  - `mlp_activations`: `["silu", "linear"]`
  - `num_experts`: 1 (Dense MLP, no MoE)
  - `rms_norm_eps` (`normalization_layer_epsilon`): 1e-6
  - `rope_theta` (`rope_max_timescale`): 10000000
  - `partial_rotary_factor`: 0.25 (first 64 dimensions rotated)
  - `attn_output_gate`: True (Qwen3-Next / Qwen3.5 full attention sigmoid gate)
  - Linear Attention (GatedDeltaNet) parameters:
    - `gdn_conv_kernel_dim`: 4
    - `gdn_key_head_dim`: 128
    - `gdn_value_head_dim`: 128
    - `gdn_num_key_heads`: 16
    - `gdn_num_value_heads`: 32
    - `gdn_chunk_size`: 64
- **Checkpoint Parameter Layout**:
  - safetensors keys for text trunk are under `model.language_model.`
  - `model.language_model.embed_tokens.weight`
  - `model.language_model.norm.weight`
  - `lm_head.weight`
  - `model.language_model.layers.{i}.*`

## 2. Phase-by-Phase Status
- [x] Phase 1: HF Discovery & Mini Safetensors Creation (DONE)
- [x] Phase 2: JAX Layer Implementation & Model Config (DONE)
- [x] Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion (DONE)
- [x] Phase 4: E2E Logits & 16-Token Autoregressive Decode Verification (DONE)

## 3. Bugs Encountered & Solutions
- **Phase 1: Config dataclass validation**: Sliced `config.json` initially kept full 32 `layer_types` while setting `num_hidden_layers=4`, triggering Hugging Face dataclass validator `ValueError: num_hidden_layers (4) must be equal to the number of layer_types (32)`. Fixed by truncating `layer_types` to first 4 elements: `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']`.
- **Phase 2: Pydantic ModelName Literal**: MaxText enforces model names via `ModelName = Literal[...]` in `src/maxtext/configs/types.py`. Added `"qwen3.5-9b"`.
- **Phase 2: Dense MLP in Qwen3.5**: `Qwen3_5` previously only instantiated `MoeBlock`. Added a check `if self.config.num_experts <= 1:` to instantiate dense `MlpBlock` with `intermediate_dim=self.config.mlp_dim` and `activations=tuple(self.config.mlp_activations)`.
- **Phase 2: RMSNorm `scale_offset=1.0` vs `use_qk_norm`**:
  - In Qwen3.5 HF, `Qwen3_5RMSNorm` initializes weights to zeros and computes `x * (1.0 + weight)`.
  - In MaxText `attentions.py`, explicitly setting `use_qk_norm: True` forced the use of standard `RMSNorm` (which computes `x * scale` without `scale_offset=1.0`).
  - Resolution: Omit `use_qk_norm: True` from `qwen3.5-9b.yml`. When omitted, `is_qwen3_hybrid` automatically instantiates `Qwen3NextRMSNorm` (which uses `scale_offset=1.0`).
- **Phase 3: Dense MLP Checkpoint Param Mapping**:
  - `Qwen3_5` checkpoint conversion had only MoE mappings (`gate_proj`, `up_proj`, `down_proj` for experts). Added dynamic branch in `get_qwen3_5_param_names` for dense MLP: `layers.{i}.mlp.gate_up_proj.kernel` (interleaved concat of gate & up) and `layers.{i}.mlp.down_proj.kernel`.
  - Added SwiGLU hook in `param_mapping.py` to handle `gate_up_proj` concatenation for dense Qwen3.5.
  - Registered `qwen3_5_dense_mlp_mapping` in `hf_shape.py` for correct shape translation.
- **Phase 4: Decode Length Configuration**:
  - For generating 16 tokens with prompt length 3 and `max_prefill_predict_length=16`, set `max_target_length=31` (1 first token from prefill + 15 generate steps = 16 tokens).

## 4. Pitfalls & Model-Specific Gotchas
- **RMSNorm scale_offset**: Always ensure Qwen3.5/Qwen3Next RMSNorm layers use `scale_offset=1.0` (`Qwen3NextRMSNorm`). Do not enable `use_qk_norm: True` in YAML as that overrides the hybrid attention's native `Qwen3NextRMSNorm`.
- **Checkpoint Suffix & Unscanned Structure**: `to_maxtext` conversion writes to `${checkpoint_dir}/0/items`. For unscanned checkpoints (`scan_layers: False`), MaxText loads variables with unstacked layer names (`layers_0`, `layers_1`, etc.).

## 5. Full Executable Reproduction Commands

### Environment Setup
```bash
source /home/hengtaoguo_google_com/projects/venv1/bin/activate
cd /home/hengtaoguo_google_com/projects/maxtext
export HF_HOME=/dev/shm/hengtaoguo
export JAX_PLATFORMS=tpu
```

### 1. Mini Safetensors Slicing (HF PyTorch)
```bash
python3 /home/hengtaoguo_google_com/projects/slice_qwen3_5_9b.py
```
Output directory: `/dev/shm/hf_mini/qwen3.5_9b_4layers`

### 2. Checkpoint Conversion (HF -> MaxText Unscanned)
```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  --base-model-path /dev/shm/hf_mini/qwen3.5_9b_4layers \
  --maxtext-model-path /dev/shm/hengtaoguo/checkpoints/qwen3.5-9b-mini-unscanned \
  --model-size qwen3.5-9b \
  --scan-layers false
```

### 3. Forward Pass Logits Test
```bash
python3 -m tests.utils.forward_pass_logit_checker \
  src/maxtext/configs/pyconfig.py \
  src/maxtext/configs/models/qwen3.5-9b.yml \
  hf_model_path=/dev/shm/hf_mini/qwen3.5_9b_4layers \
  base_output_directory=/dev/shm/hengtaoguo \
  load_parameters_path=/dev/shm/hengtaoguo/checkpoints/qwen3.5-9b-mini-unscanned/0/items \
  scan_layers=false \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=64 \
  attention=dot_product \
  use_iota_embed=false \
  tokenizer_path=/dev/shm/hf_mini/qwen3.5_9b_4layers
```

### 4. 16-Token Autoregressive Decode Test
```bash
python3 -m maxtext.inference.decode \
  src/maxtext/configs/pyconfig.py \
  src/maxtext/configs/models/qwen3.5-9b.yml \
  load_parameters_path=/dev/shm/hengtaoguo/checkpoints/qwen3.5-9b-mini-unscanned/0/items \
  scan_layers=false \
  run_name=decode_test \
  per_device_batch_size=1 \
  max_prefill_predict_length=16 \
  max_target_length=31 \
  prompt="I love to" \
  tokenizer_path=/dev/shm/hf_mini/qwen3.5_9b_4layers \
  use_iota_embed=false
```

## 6. Verification Results

### 16-Token Autoregressive Generation
- **Prompt**: `"I love to"`
- **PyTorch Golden Token IDs**: `[96004, 275, 391, 391, 12, 220, 220, 220, 220, 220, 220, 7, 95810, 93, 45, 97098]`
- **PyTorch Golden Decoded Text**: `'提itlyly-      (生~N斗'`
- **MaxText Token IDs**: `[96004, 275, 391, 391, 12, 220, 220, 220, 220, 220, 220, 7, 95810, 93, 45, 97098]`
- **MaxText Decoded Text**: `'提itlyly-      (生~N斗'`
- **Result**: **100% exact match (16/16 tokens identical)**.

### Forward Pass Logit Check (4 Prompts)
| Prompt | Top-10 Overlap | Jaccard Index | Top-10 Rank Agreement | Avg KL Divergence | Max KL Divergence |
|---|---|---|---|---|---|
| "I love to" | 10 / 10 | 1.0 | 100% | $1.72 \times 10^{-5}$ | $2.74 \times 10^{-5}$ |
| "Today is a" | 10 / 10 | 1.0 | 100% | $1.03 \times 10^{-5}$ | $1.73 \times 10^{-5}$ |
| "What is the" | 10 / 10 | 1.0 | 100% | $4.47 \times 10^{-6}$ | $6.18 \times 10^{-6}$ |
| Position-dependent RoPE prompt | 10 / 10 | 1.0 | 100% | $8.97 \times 10^{-6}$ | $1.86 \times 10^{-5}$ |
- **Result**: **All 4 prompts achieved 10/10 overlap, 1.0 Jaccard similarity, and $\text{KL} \le 2.8 \times 10^{-5}$ (threshold $\le 10^{-3}$)**.
