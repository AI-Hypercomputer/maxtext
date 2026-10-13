# Qwen3.8-27B Bring-up Handover Document (`qwen3.8-27b`)

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`)
- **Decoder Pattern**: Inhomogeneous 4-layer cycle (`inhomogeneous_layer_cycle_interval: 4`: 3 Gated DeltaNet `linear_attention` layers + 1 gated `full_attention` layer with 3D MRoPE `[11, 11, 10]`), repeated 16 times (64 layers total) with **dense** SwiGLU MLP (`num_experts: 0`).
- **Multimodal & Auxiliary Heads**:
  - **Vision Encoder**: 27-layer `Qwen3_5VisionModel` (`hidden_size=1152`, `intermediate_size=4304`, `num_heads=16`, `patch_size=16`, `temporal_patch_size=2`) + `2x2` Spatial Merger (`4608 -> 4608 -> 5120`).
  - **Multi-Token Prediction (MTP)**: 1 full-attention `Qwen3_5DecoderLayer` (`mtp_num_layers: 1`) with `Qwen3NextRMSNorm` (`1.0 + weight`) on `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, and `mtp.norm`, plus linear projection `mtp.fc` (`10240 -> 5120`) and shared `lm_head`.
- **Minimal Representative Subset ($N$)**: 4 decoder layers (`layers_0..2` GDN + `layers_3` Full Attention) + 27-layer Vision Encoder & Spatial Merger + 1 MTP layer (`404` parameter tensors total).
- **Key Parameters**:
  - `vocab_size`: `248,320` (`<|image_pad|>`: `248056`, `<|video_pad|>`: `248057`, `<|vision_start|>`: `248053`, `<|vision_end|>`: `248054`)
  - `base_emb_dim` (`hidden_size`): `5,120` | `base_mlp_dim` (`intermediate_size`): `17,408`
  - `base_num_query_heads`: `24` | `base_num_kv_heads`: `4` | `head_dim`: `256`
  - `partial_rotary_factor`: `0.25` (rotary dim = `64`, `mrope_section = [11, 11, 10]`)
  - `rope_max_timescale` (`rope_theta`): `10,000,000.0` | `normalization_layer_epsilon`: `1e-06`
  - `gdn_conv_kernel_dim`: `4` | `gdn_chunk_size`: `64`
  - `gdn_key_head_dim`: `128` | `gdn_value_head_dim`: `128` | `gdn_num_key_heads`: `16` | `gdn_num_value_heads`: `48` (`V_per_K = 3`)

---

## 2. Phase-by-Phase Status & Phase 4 Mini-Verification Checklist

- [x] **Phase 1: HF Discovery & Mini Safetensors Creation** (`DONE`)
- [x] **Phase 2: JAX Layer Implementation & Layer Tests** (`DONE`)
- [x] **Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion** (`DONE`)
- [x] **Phase 4: Comprehensive Mini-Model Verification, E2E Logits & 16-Token Decode** (`DONE`)

### Phase 4 Mini-Model Verification Checklist (`qwen3.8-27b`)

#### Gate 0: Slicing & Round-Trip Checkpoint Parity
- [x] **Representative Trimmed Subset Selected**: `4` decoder layers (`3` GDN + `1` Full Attention) + `27`-layer Vision Encoder & Spatial Merger + `1` MTP layer
- [x] **MaxText Orbax Slicing (`slice_model.py`)**: Sliced from `8` layers to `4` layers (`404` total parameter tensors: `56` text + `333` vision + `15` MTP)
- [x] **Reverse Conversion (`to_huggingface.py`)**: Exported sliced Orbax checkpoint to HuggingFace safetensors with **`max_abs_diff = 0.0`** across all `404/404` tensors

#### Core Dimension 1: Design Challenging Inputs Deliberately
- [x] **1.1 Sequence & Block/Chunk Boundaries**: Tested $S \in \{3, 4, 5, 63, 64, 65\}$ straddling GDN `conv1d` kernel $K=4$ and chunk size $C=64$ (**`cosine_sim = 1.00000000`**, **`top1 = 100.0%`** across all 6 lengths)
- [x] **1.2 Heterogeneous Padded Batches**:
  - [x] Unpadded (`S=16`) & Right-Padded (`valid=11/16` masked): **`cosine_sim = 1.00000000`**, `top1 = 100.0%`
  - [x] Left-Padded (`pad=5, valid=11`) Attribution Probe: Isolated shared `GatedDeltaNet` missing `decoder_segment_ids` mask (`0.8920` raw $\rightarrow$ **`cosine_sim = 1.00000000`**, `max_abs_diff = 4.65e-06` with 1-line pre-`conv1d` mask hook)
- [x] **1.3 Packed-Document Boundaries (`segment_ids`)**: Tested multi-segment `[1..1 (len 9), 2..2 (len 7)]` with resetting positions `[0..8, 0..6]`:
  - [x] Isolated Full-Attention Layer 3 (`Qwen3_5Attention`) with packed `segment_ids`: **`cosine_sim = 1.00000000`**, `max_abs_diff = 3.93e-06`
  - [x] Document 1 prefix (`0..8`): **`cosine_sim = 1.00000000`**, `100.0%` top-1; zero cross-document leakage (`max_abs_diff = 0.0`) when Doc 2 tokens are perturbed
- [x] **1.4 Chat Templates & Special/Boundary Tokens**: Tested ChatML (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`), vision delimiters (`248053..248057`), and boundary token `vocab_size - 1 = 248319` (**`cosine_sim = 1.00000000`**, **`top1 = 100.0%`**)
- [x] **1.5 Multimodal + 3D MRoPE Inputs**: Tested synthetic `[1, 4, 4]` visual patch grid (`16` patches $\rightarrow$ `4` merged `<|image_pad|>` tokens) with explicit 3D `[T, H, W]` coordinates (**`cosine_sim = 0.99999999`**, **`top1 = 100.0%`**)

#### Core Dimension 2: Check Forward Outputs and Masks at Meaningful Boundaries (30 Tensors Probed)
- [x] **2.1 Vision Encoder & Scatter Boundaries**:
  - [x] Boundary 1a (`patch_embed` + `pos_embed` `[16, 1152]`): **`cosine_sim = 1.00000000`**, `max_abs_diff = 1.43e-06`
  - [x] Boundary 1b (27-Layer ViT + Spatial Merger `[4, 5120]`): **`cosine_sim = 0.99999944`**, `rel_l2 = 1.08e-03`
  - [x] Boundary 1c (Post-Scatter Multimodal Input Embeddings `[1, 16, 5120]`): **`cosine_sim = 0.99999992`**, `rel_l2 = 4.65e-04`
- [x] **2.2 Recurrent / Linear-Attention Sub-Stages (GDN Layer 0)**: Probed `input_layernorm`, `in_proj_qkvz`, `in_proj_ba`, `conv1d_silu`, `chunk_gated_delta_rule`, `rms_norm_gated`, and `out_proj` (**`cosine_sim = 1.00000000`** across all 7 sub-stages; `max_abs_diff <= 1.24e-05`)
- [x] **2.3 Full-Attention Sub-Stages (Layer 3 + 3D MRoPE)**: Probed `q_norm`, `k_norm`, `3d_mrope_q`, `3d_mrope_k`, `gated_attn_out`, and `o_proj` (**`cosine_sim = 1.00000000`** across all 6 sub-stages; `max_abs_diff <= 7.87e-06`)
- [x] **2.4 Dense SwiGLU MLP Block Boundary (Layer 3)**: Probed `post_attn_norm`, `swiglu_gate_up`, and `down_proj` (**`cosine_sim = 1.00000000`**, `max_abs_diff <= 1.43e-05`)
- [x] **2.5 All 4 Decoder Layer Outputs, Final Norm & Main LM Head Logits**: `layers_0..3` hidden states, `decoder_norm`, and `logits_dense` (**`cosine_sim = 1.00000000`**, `top1 = 100.0%`)
- [x] **2.6 Auxiliary / MTP Layer 1 Sub-Stages**: Probed `rolled_target_ids` (`diff = 0.0`), `concat_normed_2x5120`, `fc_projected`, `mtp_decoder_hidden`, `mtp_final_norm`, and `mtp_logits` (**`cosine_sim >= 0.99999994`** across all 6 sub-stages)

#### Core Dimension 3: MoE / Dense Coverage Requirements
- [x] **3.1 Dense Architecture Coverage (`num_experts = 0`)**: Verified dense SwiGLU `MlpBlock` intermediate and output projections (`cosine_sim = 1.00000000`), and confirmed `total_moe_lb_loss == 0.0` across all objective configurations

#### Core Dimension 4: Verify Every Enabled Head and the Actual Training Objective
- [x] **4.1 Pre-Aggregation Loss Components Exposed**: Inspected `total_loss`, `main_ce_loss`, `total_z_loss`, `mtp_loss`, and `total_moe_lb_loss` before aggregation
- [x] **4.2 4-Way Objective Ablation Matrix**:
  - [x] `Config_A` (Main CE Only): PT `11.883260` vs. MT `11.883260` (**`abs_diff = 3.28e-07`**)
  - [x] `Config_B` (Main CE + Z-Loss `1e-4`): PT `11.900180` vs. MT `11.900181` (**`abs_diff = 9.43e-07`**, `z_loss_diff = 6.15e-07`)
  - [x] `Config_C` (MTP Loss Isolated `1.0`): PT `mtp_loss = 1.653147` vs. MT `1.653147` (**`mtp_loss_diff = 7.15e-08`**)
  - [x] `Config_D` (Combined Multimodal + Main CE + MTP `0.1` + Z-Loss `1e-4`): PT `13.750153` vs. MT `13.750205` (**`abs_diff = 5.19e-05`**, **`rel_diff = 3.78e-06`**)

#### Core Dimension 5: Make Backward Checks Sensitive and Independent
- [x] **5.1 End-to-End Parameter Group Gradients (13 Groups on `Config_D`)**: Verified all 13 parameter groups from `1_vision_patch_embed_bias` (`cosine_sim = 0.999942`) and `3_vision_projector_merger_fc2_weight` (`0.999995`) through `token_embedder` (`0.99999994`), GDN Layer 0 (`0.99999999`), Full-Attention Layer 3 (`0.99999999`), Dense MLP (`0.99999999`), `decoder_norm` (`1.00000000`), `logits_dense` (`1.00000000`), and `mtp_layer1_fc` (`0.99999999`)
- [x] **5.2 Isolated Block-Level Upstream-Gradient (VJP) Probes (7 Probes)**: Injected deterministic synthetic cotangent $G$ (`seed=12345`) into isolated GDN Layer 0, Full-Attention Layer 3, and MTP Layer 1; verified both $\partial L / \partial x_{\text{in}}$ and $\partial L / \partial W_{\text{block}}$ (**`cosine_sim = 1.00000000`**, `rel_l2 <= 1.05e-06` across all 7 probes)

#### Core Dimension 6: Test Actual Multi-Step Updates and State Propagation
- [x] **6.1 Trainable / Frozen Parameter Selection**: Verified `freeze_vision_encoder_params=True` (`max|grad(ViT_encoder)| == 0.0`, `max|grad(ViT_projector)| = 8.29e-02`) and bitwise frozen weight invariance across 3 AdamW steps (`max_abs_diff == 0.0`)
- [x] **6.2 3-Step Stateful AdamW Loss Trajectory (`lr=1e-4, b1=0.9, b2=0.95, eps=1e-8, wd=0.01`)**:
  - [x] Step 1 (Batch 0): PT `15.301152` vs. MT `15.301202` (`abs_diff = 4.96e-05`)
  - [x] Step 2 (Batch 1): PT `11.971354` vs. MT `11.972184` (`abs_diff = 8.30e-04`)
  - [x] Step 3 (Batch 2): PT `10.510243` vs. MT `10.510308` (**`abs_diff = 6.49e-05`**, `rel_diff = 6.17e-06`)
- [x] **6.3 End-of-Step-3 Weights & Optimizer Moments**: Verified updated weights $W^{(3)}$ (**`cosine_sim >= 0.99999996`**) and AdamW moments $m^{(3)}, v^{(3)}$ (**`cosine_sim >= 0.99999722`**) across Vision Merger, GDN Layer 0, Dense MLP Layer 3, and MTP Layer 1

#### Supplemental Production-Readiness Checks (`tests/integration/qwen3_8_27b_mini_supplemental_test.py`)
- [x] **S.1 `scan_layers=True` (`Qwen3_5ScannableBlock`) Parity**: `QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(..., scan_layers=True)` mapped **`506/506` keys (`missing=0`)**; forward logits **`cosine_sim = 1.00000000`** (`max_abs_diff = 6.20e-06`), combined loss **`diff = 0.000e+00`**, and decoder layer gradients `max_abs_diff = 3.43e-05`
- [x] **S.2 Asymmetric 3D MRoPE Spatial Grid (`[1, 4, 6]`) & Negative Control**: Non-square `24`-patch $\rightarrow$ `6`-token grid matched with Merger **`cosine_sim = 0.99999951`** and E2E Logits **`cosine_sim = 0.99999999`** (`100%` top-1); swapping $T \leftrightarrow W$ degraded error by **`> 4.3x`** (`1.997e-02` vs. `4.60e-03`)
- [x] **S.3 Multi-Chip 2D Mesh Sharding Parity (`ici_fsdp_parallelism=2, ici_tensor_parallelism=2`)**: Verified physical `PartitionSpec`s across 4 TPU chips; 2D-sharded vs. unsharded forward logits **`cosine_sim = 1.00000000`** (`max_abs_diff = 8.82e-06`), loss **`diff = 0.000e+00`**, and all 13 gradient groups **`cosine_sim >= 0.999942`**
- [x] **S.4 Production Mixed-Precision `bfloat16` Sanity Check**: Confirmed **`0` NaNs / `0` Infs** across all 404 gradient tensors and outputs; logits **`cosine_sim = 0.99999240`** (`100%` top-1), loss **`rel_diff = 3.48e-05`**, and all 13 gradient groups **`cosine_sim = 0.9982`–`0.999999`**

#### Standard CLI Logits & 16-Token Autoregressive Decode Check
- [x] **CLI `forward_pass_logit_checker`**: Passed (`max_kl_div <= 1.54e-04`, well within `< 1e-3` threshold)
- [x] **CLI 16-Token Autoregressive Decode (`maxtext.inference.decode`)**: Exact 16-token ID and string match against HuggingFace greedy decode

---

## 3. Bugs Encountered & Solutions
1. **Phase 4 — `qwen3.8-27b` Missing from `QWEN_SPECIAL_TOKEN_CONFIGS` (`src/maxtext/multimodal/processor_qwen3_omni.py`)**:
   - *How Found*: Dimension 2 Boundary 1c probe showed `place_multimodal_embeddings` was not scattering ViT merger outputs into `<|image_pad|>` (`248056`) slots because `QwenTokens(config).image_pad` fell back to `151655` (`qwen3-omni`).
   - *Fix*: Added `"qwen3.8"` to `QWEN_SPECIAL_TOKEN_CONFIGS` (`image_pad: 248056`, `video_pad: 248057`, `vision_start: 248053`, `vision_end: 248054`).
2. **Phase 4 — Missing `matmul_precision` in `MultiTokenPredictionLayer.projection_layer` (`src/maxtext/layers/multi_token_prediction.py`)**:
   - *How Found*: Dimension 2 Boundary 7c and Dimension 4 `Config_C` probes localized a `7.29e-03` forward jump (`2.61e-04` MTP loss diff) to `mtp_1_projection` (`fc`), which omitted `matmul_precision=cfg.matmul_precision` in `DenseGeneral`.
   - *Fix*: Passed `matmul_precision=cfg.matmul_precision` to `DenseGeneral`, reducing `Config_C` MTP loss discrepancy by **3,650x** (to `7.15e-08`).
3. **Phase 4 — Reverse Checkpoint Conversion (`to_huggingface.py`) for Sliced Multimodal + MTP Checkpoints (`hf_shape.py`, `param_mapping.py`, `to_huggingface.py`)**:
   - *How Found*: Part 0 `to_huggingface.py` export of the sliced 4-layer + Vision + MTP checkpoint failed on missing `vision_config` shapes, missing `saving_to_hf=True` ViT QKV fusion hooks, and untruncated `layer_types`.
   - *Fix*: Preserved `vision_config` and added `model.visual.*` / `mtp.*` shapes in `QWEN3_5_HF_WEIGHTS_TO_SHAPE`, added `fuse_qkv_vision` / `fuse_qkv_bias_vision` hooks when `saving_to_hf=True`, and sliced `target_cfg.layer_types[:mt_value]` in `to_huggingface.py`.
4. **Phase 2–3 — Qwen3.5/3.8 MTP Block Architecture (`multi_token_prediction.py`, `param_mapping.py`)**:
   - *How Found*: Checkpoint inspection revealed `mtp.*` weights use zero-centered `Qwen3NextRMSNorm` (`1.0 + weight`) and a full-attention `Qwen3_5DecoderLayer`.
   - *Fix*: Added `_is_qwen3_5(cfg)` branch in `MultiTokenPredictionLayer` and complete MTP parameter/hook mappings in `param_mapping.py`.
5. **Phase 2–3 — Multimodal & 3D MRoPE Registration (`qwen3.8-27b.yml`, `nnx_decoders.py`, `processor.py`, `types.py`, `hf_model_configs.py`)**:
   - *Fix*: Configured 27-layer ViT + `mrope_section: [11, 11, 10]` in `qwen3.8-27b.yml` and registered `"qwen3.8-27b"` in multimodal allowlists.

---

## 4. Pitfalls & Shared MaxText Behaviors (Session Wrap-up)
- **Interleaved GDN Projections (`V_per_K = 3`)**: `Qwen/Qwen3.8-27B` has `linear_num_key_heads = 16` and `linear_num_value_heads = 48` (`V_per_K = 3`). `in_proj_qkvz` and `in_proj_ba` are interleaved per key-head via `concat_qkvz_and_transpose` and `concat_ba_and_transpose`.
- **Shared `GatedDeltaNet` Padding & Segment Masking (`src/maxtext/models/qwen3.py`)**: `GatedDeltaNet.__call__` does not currently consume `decoder_segment_ids` to zero out padding states before causal `conv1d` or reset recurrent state across packed segments (unlike HF's `apply_mask_to_padding_states`). Our attribution probe confirmed that a 1-line pre-`conv1d` mask restores exact `1.00000000` cosine similarity on left-padded batches.
- **Shared `Qwen3OmniMoeVisionEncoder` Multi-Frame (`T > 1`) Video Mask (`src/maxtext/models/qwen3.py`)**: For $T > 1$ video grids, HF restricts ViT self-attention to per-frame $H \times W$ windows via `cu_seqlens`, whereas MaxText's `Qwen3OmniMoeVisionEncoder` attends across all $T \times H \times W$ patches when `video_mask=None`. For single-frame images ($T = 1$, including asymmetric $H \neq W$), `video_mask=None` is exact (`cosine_sim = 0.99999951`).
- **Single-Host Execution on Multi-Host TPU Slices**: When running verification on a single worker of a multi-host TPU pod (e.g., `v6e-16`), set `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 skip_jax_distributed_system=True`.

---

## 5. Full Executable Reproduction Commands

### A. Slicing & Reverse Checkpoint Conversion (`slice_model.py` + `to_huggingface.py`)
```bash
export HF_HOME=/mnt/disk/.cache/huggingface
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3

# 1. Slice 8-layer MaxText Orbax checkpoint (with Vision + MTP) to 4 layers
python3 -m maxtext.checkpoint_conversion.slicing.slice_model \
  --in_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_8layers_orbax/0/items \
  --out_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_4layers_sliced_orbax/checkpoint/0/items \
  --from_layers=8 --to_layers=4

# 2. Convert sliced 4-layer MaxText Orbax checkpoint back to HuggingFace format
python3 -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
  model_name=qwen3.8-27b base_num_decoder_layers=4 override_model_config=true \
  use_multimodal=true mtp_num_layers=1 scan_layers=false \
  load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_4layers_sliced_orbax/checkpoint/0/items \
  base_output_directory=/dev/shm/hf_mini/qwen3.8-27b_4layers_sliced_hf \
  hardware=cpu skip_jax_distributed_system=true --override_model_architecture
```

### B. Phase 4 Mini-Model Verification Suites (6 Core Dimensions + 4 Supplemental Checks)
```bash
export HF_HOME=/mnt/disk/.cache/huggingface
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3

# 1. Run all 6 Core Verification Dimensions (records JSON evidence to /tmp/qwen3_8_27b_mini_verification_evidence.json)
python3 -u tests/integration/qwen3_8_27b_mini_verification_test.py

# 2. Run all 4 Supplemental Production-Readiness Checks (records JSON evidence to /tmp/qwen3_8_27b_mini_supplemental_evidence.json)
python3 -u tests/integration/qwen3_8_27b_mini_supplemental_test.py
```

### C. CLI Forward Pass Logits Checker & 16-Token Autoregressive Decode
```bash
export HF_HOME=/dev/shm/$USER TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3

# 1. Forward Pass Logits Checker
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
  tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers_sliced_hf \
  load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_4layers_sliced_orbax/checkpoint/0/items \
  model_name=qwen3.8-27b base_num_decoder_layers=4 override_model_config=true \
  per_device_batch_size=1 scan_layers=false dtype=bfloat16 weight_dtype=bfloat16 \
  attention=dot_product max_prefill_predict_length=16 max_target_length=16 \
  skip_jax_distributed_system=True --run_hf_model=true \
  --hf_model_path=/dev/shm/hf_mini/qwen3.8-27b_4layers_sliced_hf --max_kl_div=0.01

# 2. 16-Token Autoregressive Decode
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
  model_name=qwen3.8-27b tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers_sliced_hf \
  tokenizer_type=huggingface \
  load_parameters_path=/dev/shm/$USER/checkpoints/qwen3.8-27b_4layers_sliced_orbax/checkpoint/0/items \
  run_name=qwen3-8-27b-mini-decode base_num_decoder_layers=4 override_model_config=true \
  per_device_batch_size=1 max_prefill_predict_length=16 max_target_length=32 \
  steps=1 scan_layers=false dtype=bfloat16 weight_dtype=bfloat16 \
  attention=dot_product skip_jax_distributed_system=True prompt='I love to'
```
