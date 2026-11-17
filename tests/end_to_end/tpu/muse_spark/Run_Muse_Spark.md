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
* Use `remat_policy=full`. With `scan_layers=true` each scanned 4-layer block is rematerialized as a unit, and the routed experts are additionally rematerialized per layer, so only one layer's routed-expert intermediates are live at a time.
* The routed experts are evaluated with a sharded `ragged_dot` (dropless). `sparse_matmul`, `megablox` and `capacity_factor` do not apply to this model, and expert parallelism (`ici_expert_parallelism > 1`) is not supported; shard the experts with FSDP and tensor parallelism instead.

See `2_test_muse_spark_816b_a42b.sh` for the daily workflow:
1. Forward-pass logit check (`tests.utils.forward_pass_logit_checker`)
2. Pre-training (`maxtext.trainers.pre_train.train`)
3. Fine-tuning from the converted checkpoint
4. Greedy decoding (`maxtext.inference.decode`)

## Correctness tests

`tests/unit/muse_spark_vs_reference_test.py` compares each layer, the full model (scanned and unscanned, loaded through the conversion mapping), and the gradients of every parameter against the PyTorch reference in `transformers.models.muse_spark`. It needs a transformers version that ships `muse_spark` (≥ 5.18), so it is excluded from the default CI run. Run it with:

```bash
python3 -m pytest -o addopts="" tests/unit/muse_spark_vs_reference_test.py
```
