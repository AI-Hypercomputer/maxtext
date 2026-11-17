<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# Running Muse-Spark-Open-816B-A42B in MaxText

[Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) (`model_name=muse-spark-816b-a42b`) is the text backbone of `MuseSparkForConditionalGeneration`:

* 62 decoder layers, hidden size 8192, gated GQA attention (128 query / 16 KV heads, head_dim 64) with scaleless QK-norm.
* Sliding-window attention (window 2048) with RoPE (θ = 500,000) on 46 layers, and global NoPE attention on every 4th layer counted from the last (layers 1, 5, …, 61).
* Every layer is a bottlenecked sparse MoE: 8192 → 4096 → 256 routed experts (top-8, sigmoid routing with a correction bias, SwiGLU 4096 → 4096) → 8192. There are no shared experts.
* Temperature-scaled residual gates, a normalized token embedding, an untied LM head, and `20·tanh(logits · 0.1768 / 20)` output soft-capping.

With `scan_layers=true` the first 2 layers are unrolled and the remaining 60 layers run as 15 scanned 4-layer blocks.

The end-to-end test scripts are:
* `tests/end_to_end/tpu/muse_spark/muse-spark-816b-a42b/1_test_muse_spark_816b_a42b.sh`: HF → MaxText checkpoint conversion, run once on a high-memory CPU VM.
* `tests/end_to_end/tpu/muse_spark/muse-spark-816b-a42b/2_test_muse_spark_816b_a42b.sh`: logit check, pre-training, fine-tuning and decoding on TPU.

## Checkpoint conversion (HuggingFace → MaxText)

The HF repo is gated: export `HF_TOKEN`, or pass a local copy of the safetensors with `--hf_model_path`.

```bash
# Scanned checkpoint (recommended for training)
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=muse-spark-816b-a42b \
  base_output_directory=gs://your-bucket/muse-spark-816b-a42b/scanned \
  scan_layers=true hardware=cpu skip_jax_distributed_system=true \
  --hf_model_path=/path/to/hf/muse-spark-816b-a42b --save_dtype=bfloat16

# Unscanned checkpoint
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=muse-spark-816b-a42b \
  base_output_directory=gs://your-bucket/muse-spark-816b-a42b/unscanned \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  --hf_model_path=/path/to/hf/muse-spark-816b-a42b --save_dtype=bfloat16
```

The router kernels, the routing correction biases (`MoEBiasVar`), and the residual gates (`post_attention_residual_gate` and `post_feedforward_residual_gate`) are saved in float32, matching `MuseSparkPreTrainedModel._keep_in_fp32_modules_strict`. All other weights are saved in bfloat16.

## Pre-training and fine-tuning

* Training does not fit on a 16-chip TPU v7x slice (`tpu7x-32`): the train step runs out of HBM at compile time. With `per_device_batch_size=1` and `max_target_length=256` it compiles within HBM on a 32-chip slice (`tpu7x-64`, about 88.5 GiB of the 94.7 GiB per TensorCore), and it has been run end to end on a 64-chip slice (`tpu7x-128`).
* On small slices use `remat_policy=full`. With `scan_layers=true` each scanned 4-layer block is rematerialized as a unit.
* The routed experts never drop tokens. `sparse_matmul`, `megablox` and `capacity_factor` do not apply to this model. The experts can be sharded with FSDP, tensor parallelism and expert parallelism (`ici_expert_parallelism > 1`, which needs `moe_expert_input_dim` to be a multiple of 128). With expert parallelism each device sorts the tokens routed to its experts into a buffer of `ragged_buffer_factor` times the average load; tokens beyond the buffer are processed in extra passes over the same buffer, so a small factor costs time on skewed routing but never changes the result. `ragged_buffer_factor=-1` sizes the buffer for the worst case.

See `2_test_muse_spark_816b_a42b.sh` for the daily workflow:
1. Forward-pass logit check (`tests.utils.forward_pass_logit_checker`)
2. Pre-training (`maxtext.trainers.pre_train.train`)
3. Fine-tuning from the converted checkpoint
4. Greedy decoding (`maxtext.inference.decode`)

## Training on 256 TPU v7x chips

Recommended flags for training the full model on one 256-chip TPU v7x slice (`tpu7x-512`, 4x8x8 chips, 512
TensorCores) with `scan_layers=true`, `max_target_length=4096` and `per_device_batch_size=2`.
`tests/unit/muse_spark_layers_test.py` checks that they parse and build the intended mesh (`TPU7X_256_TRAINING_FLAGS`).

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name=muse-spark-816b-a42b scan_layers=true max_target_length=4096 per_device_batch_size=2 \
  custom_mesh=ep8_z_fsdp64_xycore ici_fsdp_parallelism=64 ici_expert_parallelism=8 \
  logical_axis_rules="[['embed',['fsdp']],['embed_attn',['fsdp']],['embed_vocab',['fsdp']]]" sharding_tolerance=0.05 attention=flash \
  use_tokamax_splash=true sa_q_layout=SEQ_MINOR sa_k_layout=SEQ_MINOR \
  sa_v_layout=SEQ_MINOR local_sa_q_layout=SEQ_MINOR local_sa_k_layout=SEQ_MINOR \
  local_sa_v_layout=SEQ_MINOR sa_use_fused_bwd_kernel=true local_sa_use_fused_bwd_kernel=true \
  use_tokamax_gmm=true use_gmm_v2=true \
  ragged_buffer_factor=2.0 muse_spark_moe_num_token_chunks=2 remat_policy=custom \
  decoder_layer_input=remat attention_out=remat context=device \
  moe_combined=device moe_routing_maps=device out_proj=device \
  value_proj=device query_proj=device key_proj=device \
  mlpwi=device pre_expert_norm=device mlpwo=device \
  num_vocab_tiling=1 tie_lm_head_cotangents=true
```

* `custom_mesh=ep8_z_fsdp64_xycore` places expert parallelism (8) on the z ring of the torus and FSDP (64) on the
  (x, y) plane and the two TensorCores of each chip, so expert-parallel collectives stay on one ring.
* `muse_spark_moe_num_token_chunks=2` splits each device's tokens into two chunks so the expert-parallel collectives of
  one chunk overlap the expert matmuls of the other.
* `remat_policy=custom` with the listed tensors on `device` keeps those tensors in HBM for the backward pass instead of
  recomputing them.

### FP8 experts

`quantization=fp8_full` runs the routed-expert matmuls and the rest of the decoder layers in FP8 through Qwix; the
router, norms, residual gates, embedding and LM head stay in full precision. No other quantization mode is supported
for this model. Add these flags to the command above (`TPU7X_256_FP8_TRAINING_FLAGS` in the same test):

```bash
  quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false \
  te_moe_dispatch_remat_location=device query_proj=remat key_proj=remat \
  mlpwi=remat mlpwo=remat wi_tile_fwd_batch_seq=512 \
  wi_tile_fwd_embed_dim=2048 wi_tile_fwd_mlp_dim=2048
```

## Correctness tests

`tests/unit/muse_spark_vs_reference_test.py` compares each layer, the full model (scanned and unscanned, loaded through the conversion mapping), and the gradients of every parameter against the PyTorch reference in `transformers.models.muse_spark`. It needs a transformers version that ships `muse_spark` (≥ 5.18), so it is excluded from the default CI run. Run it with:

```bash
python3 -m pytest -o addopts="" tests/unit/muse_spark_vs_reference_test.py
```
