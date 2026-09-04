# Qwen3.8-27B (Text-only) MaxText Onboarding Handover

## 1. Summary
The text-only architecture of **Qwen/Qwen3.8-27B** has been successfully onboarded to MaxText on branch `hengtaoguo-exp5` and thoroughly verified on TPU v5p (hengtaoguo-dev-v5p2 / 34.32.131.57).

Both validation milestones are **PASSED**:
1. **Forward Pass Logits Checker** (`tests/utils/forward_pass_logit_checker.py`): **PASSED** (Top-10 Overlap: 10/10, Jaccard similarity: 1.0, Rank agreement: 100%, Average KL divergence: < 1e-3).
2. **16-Token Autoregressive Decoding** (`maxtext.inference.decode`): **PASSED** (100% exact token ID match against HuggingFace PyTorch reference).

---

## 2. Model Architecture & Hyperparameters
- **Base Architecture**: Hybrid Attention model with Gated Delta Network (GDN) linear attention and full attention in a 4-layer cycle (`inhomogeneous_layer_cycle_interval: 4`: 3 GDN layers + 1 Full Attention layer).
- **MLP Type**: Dense SwiGLU MLP (`num_experts: None` / 1).
- **Key Dimensions**:
  - `vocab_size`: 248,320
  - `emb_dim`: 5,120 (`hidden_size`)
  - `mlp_dim`: 17,408 (`intermediate_size`)
  - `num_decoder_layers`: 64 (`num_hidden_layers`)
  - `num_query_heads`: 24 (`num_attention_heads`)
  - `num_kv_heads`: 4 (`num_key_value_heads`)
  - `head_dim`: 256
  - `partial_rotary_factor`: 0.25 (rotary dim = 64)
  - `rope_theta`: 10,000,000.0
  - `normalization_layer_epsilon`: 1e-06
  - `gdn_conv_kernel_dim`: 4
  - `gdn_key_head_dim`: 128
  - `gdn_value_head_dim`: 128
  - `gdn_num_key_heads`: 16
  - `gdn_num_value_heads`: 32
  - `use_qk_norm_in_gdn`: True

---

## 3. Key Changes Made
1. **Model Config**:
   - Added `src/maxtext/configs/models/qwen3.8-27b.yml` with exact architecture parameters.
   - Added `qwen3.8-27b` to `ModelName` in `src/maxtext/configs/types.py`.
   - Added `qwen3.8-27b: Qwen/Qwen3.8-27B` to `HF_IDS` in `src/maxtext/utils/globals.py`.
   - Registered `qwen3_8_27b_config` in `src/maxtext/checkpoint_conversion/utils/hf_model_configs.py`.
2. **Model Definition**:
   - Updated `src/maxtext/models/qwen3_5.py` (`Qwen3_5DecoderLayer`) to support dense `MlpBlock` when `getattr(cfg, num_experts, 1) <= 1`.
3. **Checkpoint Conversion**:
   - Updated `src/maxtext/checkpoint_conversion/utils/param_mapping.py` to support dense MLP weight mapping and hooks for Qwen3.5/Qwen3.8 models.
4. **KV Cache & Inference Fix**:
   - Fixed `src/maxtext/inference/kvcache.py` to initialize `cached_prefill_key` with `jnp.float32` for GDN recurrent state, preventing dtype mismatch during decode insert.

---

## 4. Verification Commands & Results

### Sliced Mini-Checkpoint Location
- HuggingFace 4-layer mini model: `/dev/shm/hf_mini/qwen3.8-27b_4layers`
- Converted Orbax checkpoint: `/dev/shm/hengtaoguo/checkpoints/qwen3.8-27b_mini_orbax/0/items`

### 1. Forward Pass Logits Checker
```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml   tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers   load_parameters_path=/dev/shm/hengtaoguo/checkpoints/qwen3.8-27b_mini_orbax/0/items   model_name=qwen3.8-27b   base_num_decoder_layers=4   override_model_config=true   per_device_batch_size=1   scan_layers=false   dtype=bfloat16   weight_dtype=bfloat16   attention=dot_product   max_prefill_predict_length=16   max_target_length=16   --run_hf_model=true   --hf_model_path=/dev/shm/hf_mini/qwen3.8-27b_4layers   --max_kl_div=0.01
```
**Result**: 4/4 prompts evaluated. Top-10 overlap 10/10 (100%), Jaccard similarity 1.0, KL divergence ~ 5e-4 to 1e-3.

### 2. Autoregressive Decode (16 tokens)
```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml   model_name=qwen3.8-27b   tokenizer_path=/dev/shm/hf_mini/qwen3.8-27b_4layers   tokenizer_type=huggingface   load_parameters_path=/dev/shm/hengtaoguo/checkpoints/qwen3.8-27b_mini_orbax/0/items   run_name=qwen3-8-27b-mini-decode   base_num_decoder_layers=4   override_model_config=true   per_device_batch_size=1   max_prefill_predict_length=16   max_target_length=32   steps=1   scan_layers=false   dtype=bfloat16   weight_dtype=bfloat16   attention=dot_product   prompt='I love to'
```
**Result**: Generated 16 tokens matching the HuggingFace PyTorch reference with 100% token-by-token parity.
