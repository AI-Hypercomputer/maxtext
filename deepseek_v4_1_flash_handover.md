# DeepSeek-V4.1-Flash Bring-up Handover Document

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: `deepseek-ai/DeepSeek-V4.1-Flash`
- **Model Type in HF**: `deepseek_v41` (multimodal vision + text, focusing on text-only first)
- **Decoder Pattern**:
  - Homogeneous sliding-window + MoE for prefix layers (`compress_ratio = 0` for layers 0-1)
  - Interleaved compressed attention for subsequent layers
  - Minimal representative subset for bring-up: **1 layer** (N=1, layer 0 with sliding window attention + 384-expert MoE + mHC)
- **Key Parameters**:
  - `vocab_size`: 129280
  - `hidden_size` (`base_emb_dim`): 5120
  - `num_hidden_layers`: 40 (text backbone)
  - `num_attention_heads`: 64
  - `num_key_value_heads`: 1
  - `head_dim`: 512
  - `qk_rope_head_dim`: 64
  - `q_lora_rank`: 1280
  - `kv_lora_rank`: 512
  - `o_groups`: 8
  - `o_lora_rank`: 1024
  - `moe_intermediate_size`: 2304
  - `n_routed_experts`: 384
  - `n_shared_experts`: 1
  - `num_experts_per_tok`: 6
  - `score_func`: `sqrtsoftplus`
  - `routed_scaling_factor`: 1.5
  - `swiglu_limit`: 10.0
  - `rms_norm_eps`: 1e-20
  - `rope_theta`: 10000
  - `sliding_window`: 128
  - `hc_mult`: 4
  - `hc_sinkhorn_iters`: 20
  - `hc_eps`: 1e-6
  - `first_num_hash_layers`: 0 (uses Top-K router with correction bias across all layers)

## 2. Phase-by-Phase Status
- [x] Phase 1: HF Discovery & Architecture Analysis (DONE)
- [x] Phase 2: JAX Layer Implementation & Verification (DONE)
- [x] Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion (DONE)
- [x] Phase 4: E2E Logits & 16-Token Autoregressive Decode Verification (DONE)

## 3. Bugs Encountered & Solutions
- **Phase 1**:
  - *HF model type deepseek_v41 unrecognized by Transformers library*: Mapped to `deepseek_v4` in mini config and aligned with `DeepseekV4Config`.
  - *Checkpoint Quantization*: Model weights on HuggingFace Hub are stored as block FP8 (`e4m3fn` with `e8m0fnu` 32x32 block scales) and FP4 (`e2m1fn_x2` packed into int8 with `e8m0fnu` 1x32 scales). Wrote clean dequantization logic into the mini safetensors slicer to produce unquantized bfloat16 tensors for exact FP32/BF16 mathematical verification.
  - *HyperHead Parameter Structure*: In DeepSeek-V4.1, the final stream collapse before RMSNorm uses the last layer FFN HC pre-mix parameters (`layer.hc_ffn_fn[:4, :]`, `base[:4]`, `scale[:1]`). Mapped these directly to `model.hc_head` to maintain exact parity with PyTorch and MaxText.
- **Phase 2 & 3**:
  - *Router Bias Discarded during NNX State Dict Extraction*: In `src/maxtext/layers/moe.py`, `MoEBiasVar` was originally defined as an `nnx.Variable` without being recognized as an active model parameter in `to_maxtext.py`. Changed `MoEBiasVar` to `nnx.Param` and updated `to_maxtext.py` so the router bias is tracked and saved properly into Orbax checkpoints.
  - *Gate Bias Floating Point Precision Downcast*: Router bias values in DeepSeek-V4 are small correction offsets (~ +/- 0.05). Downcasting them to BF16 during checkpoint conversion introduced roundoff errors that flipped expert rankings in top-6 expert selection. Preserved `gate-bias` in `float32` during conversion and enabled `float32_gate_logits: true` in `deepseek4.1-flash.yml`.
  - *Router Gate Bias Addition Float32 Promotion*: In `GateLogit.__call__`, ensured the router logits and bias addition is promoted to `float32` before top-k selection, matching PyTorch reference routing exactly (`diff = 0.0`).
- **Phase 4**:
  - *MoE Sparse Matmul Buffer Sizing on Small Batches*: In `RoutedMoE`, `sparse_matmul` applies ragged sort and ring-of-experts buffer sizing intended for massive pod-scale batches; for single sequences (B=1, S=1..7), `local_batch // num_ep` dropped expert activations across TPU shards. Setting `sparse_matmul: false` enables `dense_matmul` (`jnp.einsum`), achieving exact 0.0007 diff against PyTorch and zero token drop.
  - *Autoregressive Decoding Parity*: Greedy decode across 16 steps produced 100% identical token IDs between HuggingFace PyTorch and MaxText.

## 4. Verification Results

### A) Forward Pass Logits & KL Divergence (7 Tokens)
- Prompt: `"Paris is the capital of France and"`
- Token IDs: `[51119, 344, 270, 6102, 294, 8760, 305]`
- Per-token KL Divergence:
  - Token 0 (`Paris`): 6.67e-04
  - Token 1 (` is`): 1.87e-03
  - Token 2 (` the`): 9.30e-04
  - Token 3 (` capital`): 7.01e-04
  - Token 4 (` of`): 7.49e-04
  - Token 5 (` France`): 3.56e-04
  - Token 6 (` and`): 3.71e-04
- **Mean KL Divergence**: 8.066339e-04 (<= 1e-3, **PASSED**)
- **Max KL Divergence**: 1.872358e-03 (<= 1e-2, **PASSED**)
- Top-5 Next Token Predictions:
  - HF PyTorch: `[('/or', '17.25'), ('rew', '14.69'), (' Tant', '14.31'), ('tant', '13.75'), ('oho', '13.38')]`
  - MaxText:    `[('/or', '17.25'), ('rew', '14.62'), (' Tant', '14.31'), ('tant', '13.75'), ('ijani', '13.38')]`

### B) Autoregressive 16-Token Generation (Greedy)
- **HuggingFace PyTorch**:
  - Tokens: `[7959, 12096, 119672, 42572, 42490, 129276, 9665, 102598, 129275, 18952, 21189, 58837, 10669, 90032, 7328, 1209]`
  - Text: `"/originalterre kinainitan giiniton<｜box｜> Accessibilité<｜/box｜> Represent407 innocence162 Rossiipalities"`
- **MaxText on TPU VM**:
  - Tokens: `[7959, 12096, 119672, 42572, 42490, 129276, 9665, 102598, 129275, 18952, 21189, 58837, 10669, 90032, 7328, 1209]`
  - Text: `"/originalterre kinainitan giiniton<｜box｜> Accessibilité<｜/box｜> Represent407 innocence162 Rossiipalities"`
- **Parity Status**: **100% EXACT MATCH (16 / 16 tokens)**.

## 5. Pitfalls & Model-Specific Gotchas
- **Storage Limits**: Checkpoints and HuggingFace cache must reside in `/dev/shm/hengtaoguo` and `/dev/shm/hf_mini`.
- **Attention Configuration**: DeepSeek-V4 decoder uses `attention=dot_product`.
- **Sparse Matmul vs Dense Matmul**: When validating single-sequence or small-batch runs on multi-chip TPU meshes, set `sparse_matmul: false` to prevent token dropping from ragged buffer partitioning.
- **MoE Router Bias Precision**: Bias values must stay in FP32 during conversion and forward gating to maintain exact routing set parity.

## 6. Full Reproduction Commands

1. **Slice Mini 1-Layer Checkpoint**:
   ```bash
   /home/hengtaoguo_google_com/projects/venv1/bin/python /home/hengtaoguo_google_com/projects/slice_mini_1layer.py
   ```

2. **Convert Safetensors to MaxText Orbax Checkpoint**:
   ```bash
   cd /home/hengtaoguo_google_com/projects/maxtext
   /home/hengtaoguo_google_com/projects/venv1/bin/python -m maxtext.checkpoint_conversion.to_maxtext \
     src/maxtext/configs/base.yml \
     model_name=deepseek4.1-flash \
     base_num_decoder_layers=1 \
     override_model_config=true \
     scan_layers=false \
     attention=dot_product \
     sparse_matmul=false \
     base_output_directory=/dev/shm/hengtaoguo/checkpoints \
     run_name=deepseek_v4_1_flash_mini_orbax \
     hf_model_path=/dev/shm/hf_mini/deepseek_v4_1_flash_1layer \
     save_quantized_params=false \
     dtype=bfloat16 \
     weight_dtype=bfloat16
   ```

3. **Run Full Parity Validation (Logits KL & 16-Token Autoregressive Decode)**:
   ```bash
   /home/hengtaoguo_google_com/projects/venv1/bin/python /home/hengtaoguo_google_com/projects/validate_mini_parity.py
   ```

4. **Run On-the-Fly Hugging Face Comparison in Forward Pass Logit Checker**:
   ```bash
   cd /home/hengtaoguo_google_com/projects/maxtext
   PYTHONPATH=/home/hengtaoguo_google_com/projects/transformers/src:/home/hengtaoguo_google_com/projects/maxtext/src:/home/hengtaoguo_google_com/projects/maxtext    /home/hengtaoguo_google_com/projects/venv1/bin/python -m tests.utils.forward_pass_logit_checker      src/maxtext/configs/base.yml      model_name=deepseek4.1-flash      load_parameters_path=/dev/shm/hengtaoguo/checkpoints/deepseek_v4_1_flash_mini_orbax/0/items      tokenizer_path=/dev/shm/hf_mini/deepseek_v4_1_flash_1layer      base_num_decoder_layers=1      scan_layers=false      override_model_config=true      attention=dot_product      sparse_matmul=false      skip_jax_distributed_system=true      per_device_batch_size=1      max_target_length=32      ici_expert_parallelism=4      checkpoint_storage_use_ocdbt=false      checkpoint_storage_use_zarr3=false      --run_hf_model=true      --hf_model_path=/dev/shm/hf_mini/deepseek_v4_1_flash_1layer      --max_kl_div=0.1
   ```

### 7. Hugging Face Mini Model Loading Notes
- **State Dict Keys**: DeepseekV4ForCausalLM uses 3D stacked expert tensors (gate_up_proj [384, 4608, 5120] and down_proj [384, 5120, 2304]). The mini checkpoint safetensors in /dev/shm/hf_mini/deepseek_v4_1_flash_1layer contains exactly these 29 parameters.
- **RMSNorm Dtype Consistency**: In transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py, DeepseekV4RMSNorm.forward ensures the output matches input_dtype ((self.weight.to(input_dtype) * hidden_states.to(input_dtype))) even when loaded with FP32 norm parameters.
