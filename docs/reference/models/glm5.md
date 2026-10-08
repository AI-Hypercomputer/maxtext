<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

(glm5)=

# GLM-5.1 & GLM-5.2

MaxText supports Z.ai's `744B` sparse Mixture-of-Experts (MoE) models **GLM-5.1** (`glm5.1-744b`, [`zai-org/GLM-5.1`](https://huggingface.co/zai-org/GLM-5.1)) and **GLM-5.2** (`glm5.2-744b`, [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)), corresponding to Hugging Face's `GlmMoeDsaForCausalLM` (`model_type: "glm_moe_dsa"`) and MaxText's `DecoderBlockType.GLM5` (`decoder_block: "glm5"`).

Both models share a `744B`-parameter (`~40B`–`44B` active per token), `78`-layer (`3` dense + `75` MoE) backbone with **Multi-Head Latent Attention (MLA)** and a **Dynamic Sparse Attention (DSA) Lightning Indexer**. **GLM-5.2** extends the context window to `1M` tokens and introduces native **Cross-Layer IndexShare**, where `57` of the `78` layers (`73.1%`) reuse sparse top-$k$ token indices from `21` donor Full layers across both `jax.lax.scan` training and unscanned autoregressive decoding.

## Architecture highlights

| Parameter / Feature           | GLM-5.1 (`glm5.1-744b`)                                                                                                                                                      | GLM-5.2 (`glm5.2-744b`)                                                  |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **Hugging Face ID**           | `zai-org/GLM-5.1`                                                                                                                                                            | `zai-org/GLM-5.2`                                                        |
| **Total / Active Parameters** | `744B` / `~40B`–`44B`                                                                                                                                                        | `744B` / `~40B`–`44B`                                                    |
| **Decoder Block**             | `glm5` (`src/maxtext/models/glm5.py`)                                                                                                                                        | `glm5` (`src/maxtext/models/glm5.py`)                                    |
| **Decoder Layers**            | `78` (`3` dense + `75` MoE)                                                                                                                                                  | `78` (`3` dense + `75` MoE)                                              |
| **Hidden / Dense MLP Dim**    | `6144` / `12288` (SwiGLU)                                                                                                                                                    | `6144` / `12288` (SwiGLU)                                                |
| **MoE Routing & Experts**     | `256` routed (`moe_mlp_dim=2048`), top-`8` + `1` shared expert; sigmoid routing (`routed_scaling_factor=2.5`, `norm_topk_prob=true`) with loss-free `MoEBiasVar` (`float32`) | Same as GLM-5.1                                                          |
| **MLA Configuration**         | `64` heads, `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`                                                          | Same as GLM-5.1                                                          |
| **DSA Lightning Indexer**     | `use_indexer=true`, `32` heads × `128` dim, `indexer_topk=2048`, `indexer_rope_interleave=true`                                                                              | Same as GLM-5.1                                                          |
| **Cross-Layer IndexShare**    | Disabled (`use_index_share=false`; all `78` layers run Full indexers)                                                                                                        | Enabled (`use_index_share=true`, `21` Full `F` + `57` Shared `S` layers) |
| **Max Context / RoPE Theta**  | `202,752` / `1,000,000` (interleaved RoPE)                                                                                                                                   | `1,048,576` / `8,000,000` (interleaved RoPE)                             |
| **Vocabulary Size**           | `154,880` (untied embeddings)                                                                                                                                                | `154,880` (untied embeddings)                                            |

Key implementation details in MaxText:

- **Thin `DeepSeek` layer subclasses (`src/maxtext/models/glm5.py`)**: `GLMDenseLayer` and `GLMMoELayer` subclass `DeepSeekGenericLayer`, resolving per-layer IndexShare roles and threading `IndexerShareState(topk_indices, indexer_score)` across `dense_layers` and `moe_layers`.
- **Interleaved direct-frequency RoPE**: Both MLA and the DSA Lightning Indexer use interleaved rotary embeddings (`rope_interleave=true`, `indexer_rope_interleave=true`). MaxText represents plain unscaled interleaved RoPE via `rope_type="yarn"` with `rope_factor=1.0` and `rope_direct_position_freqs=true` (avoiding materializing a `[1M, 32]` `complex64` lookup table), and omits `rope_scaling` when exporting back to Hugging Face so `rope_parameters.rope_type` remains `"default"`.
- **Sparse top-$k$ activation threshold**: The DSA Lightning Indexer activates sparse top-$k$ token selection when `kv_len > indexer_topk` (default `2048`). When `kv_len <= indexer_topk`, the indexer takes an early dense causal mask shortcut across all layers.

## Cross-layer IndexShare (`GLM-5.2`)

GLM-5.2 exploits cross-layer token-selection stability via three configuration flags defined in `src/maxtext/configs/base.yml` and validated in `src/maxtext/configs/types.py`:

- **`use_index_share`** (`bool`, default `false`; `true` in `glm5.2-744b.yml`): Enables cross-layer propagation of `IndexerShareState(topk_indices, indexer_score)` across the `dense_layers` and `moe_layers` decoder groups. Requires `decoder_block="glm5"`, `attention_type="mla"`, `use_indexer=true`, and `indexer_topk > 0`.
- **`index_share_pattern`** (`str`, default `"FSSS"`): Tiled pattern of **Full (`F`)** donor layers and **Shared (`S`)** consumer layers. Layer `0` must be `'F'`, and the pattern's normalized length must evenly divide `num_decoder_layers`. Each Shared layer $\ell$ reuses the `topk_indices` computed by its nearest preceding Full donor layer $f(\ell) = \max\{j \le \ell : \text{pattern}[j] = \text{'F'}\}$. GLM-5.2's exact `78`-layer topology is written out in full as `"FFF" + 18 * "SSSF" + "SSS"` (`21` Full layers at indices `{0, 1, 2, 6, 10, ..., 74}` and `57` Shared layers). MLA indexer FLOP accounting automatically scales by `count("F") / num_decoder_layers`.
- **`prune_shared_indexers`** (`None | bool`, default `null`): When `null`, resolves automatically to `(not scan_layers)` when `use_index_share=true` (otherwise `false`):
  - **Scanned training (`scan_layers=true`, `prune_shared_indexers=false`)**: Shared layers retain parameter slots for `jax.lax.scan` PyTree homogeneity (`~0.5B` parameters on `744B`) and dispatch Full vs. Shared execution via `nnx.cond`. Retained Shared-layer indexer slots receive zero gradients.
  - **Unscanned inference (`scan_layers=false`, `prune_shared_indexers=true`)**: Shared layers omit `self_attention.indexer` parameters and `IndexerKVCache_0` entirely, reducing HBM footprint and KV-cache memory during decoding.

### Multi-layer indexer distillation

When training the Lightning Indexer (`indexer_loss_scaling_factor > 0` with `max_target_length > indexer_topk`), each donor Full layer $f$ minimizes the group-normalized KL divergence across all layers $\text{Served}(f)$ in its sharing group:

$$\mathcal{L}_{f} = \frac{\lambda}{|\text{Served}(f)|} \sum_{j \in \text{Served}(f)} D_{\text{KL}}\!\left(p^{(j)} \,\|\, q^{(f)}\right)$$

where $\lambda$ is `indexer_loss_scaling_factor`, $p^{(j)}$ is layer $j$'s main MLA attention distribution, and $q^{(f)}$ is donor layer $f$'s indexer score distribution. Under `scan_layers=true`, `indexer_score` (`float32` `[B, S, kv_len]`) is carried across scanned layers so Shared-layer KL gradients flow back to the donor Full layer's indexer; when `indexer_loss_scaling_factor == 0` or during inference, the scan carry holds `int32` `topk_indices` only.

### Configuration guards

`_validate_index_share` fails fast with a descriptive `ValueError` on unsupported combinations:

- `prune_shared_indexers=true` with `scan_layers=true` (scanned layers require homogeneous PyTrees) or with `stack_prefill_result_cache=true` (pruned Shared layers set `IndexerKVCache_0 = None`, which cannot be stacked across layers in MaxEngine; set `prune_shared_indexers=false` if stacking prefill caches).
- `use_index_share=true` with pipeline parallelism (`ici_pipeline_parallelism > 1` or `dcn_pipeline_parallelism > 1`), `engram_layers`, `use_batch_split_schedule=true`, `attention` in `("vllm_rpa", "vllm_batched_rpa")`, `mhc_expansion_rate > 1`, or `shard_mode="explicit"`.

## Checkpoint conversion

GLM-5.1 and GLM-5.2 support bidirectional conversion between Hugging Face (`GlmMoeDsaForCausalLM` safetensors) and MaxText Orbax checkpoints in both scanned and unscanned layouts. See [`1_test_glm5.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/glm5/glm5.2-744b/1_test_glm5.sh) for a complete CPU conversion script.

### 1. Download (and dequantize if needed)

Download the official weights from Hugging Face ([`zai-org/GLM-5.1`](https://huggingface.co/zai-org/GLM-5.1) or [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)). If starting from FP8 safetensors, first dequantize to BF16 on CPU:

```bash
python3 -m maxtext.checkpoint_conversion.standalone_scripts.deepseek_dequantize \
    --input-path=${LOCAL_FP8_PATH} \
    --output-path=${BF16_LOCAL_PATH}
```

### 2. Hugging Face to MaxText (`to_maxtext.py`)

Set `MODEL_NAME=glm5.2-744b` (or `glm5.1-744b`). Convert to **scanned** (`scan_layers=true`) format for pre-training, fine-tuning, and forward-pass logit checks, or **unscanned** (`scan_layers=false`) format for autoregressive decoding:

```bash
# Scanned Orbax checkpoint (for training, fine-tuning, and logit verification)
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    scan_layers=true \
    attention=dot_product \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/scanned \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_concurrent_gb=1024 \
    --hf_model_path=${BF16_LOCAL_PATH} \
    --lazy_load_tensors=false \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16

# Unscanned Orbax checkpoint (for decoding; prunes Shared-layer indexers on glm5.2-744b)
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    scan_layers=false \
    attention=dot_product \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/unscanned \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_concurrent_gb=1024 \
    --hf_model_path=${BF16_LOCAL_PATH} \
    --lazy_load_tensors=false \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16
```

Converted checkpoints are written to `${BASE_OUTPUT_DIRECTORY}/scanned/0/items` and `${BASE_OUTPUT_DIRECTORY}/unscanned/0/items`.

> **Note (`GLM-5.2` Shared-layer weights & `float32` preservation):** Hugging Face `zai-org/GLM-5.2` stores `self_attn.indexer.*` tensors only on the `21` Full (`F`) layers. When `scan_layers=true`, `to_maxtext` automatically populates each Shared (`S`) layer's scanned slot from its exact donor Full layer (failing fast if any Full or donor key is missing); when `scan_layers=false`, `prune_shared_indexers` defaults to `true` and Shared-layer indexer leaves are omitted. Even with `--save_dtype=bfloat16`, router `MoEBiasVar` (`e_score_correction_bias`) and indexer `k_norm` / `weights_proj` leaves are preserved in `float32`.

### 3. MaxText to Hugging Face (`to_huggingface.py`)

Use `to_huggingface.py` to export either a scanned (`scan_layers=true`) or unscanned (`scan_layers=false`) MaxText Orbax checkpoint back to Hugging Face `GlmMoeDsaForCausalLM` safetensors and `config.json`:

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    load_parameters_path=${MAXTEXT_CKPT_PATH} \
    base_output_directory=${HF_OUTPUT_DIRECTORY} \
    scan_layers=true \
    hardware=cpu \
    skip_jax_distributed_system=true \
    weight_dtype=bfloat16
```

When exporting a scanned `glm5.2-744b` checkpoint (`use_index_share=true`, `scan_layers=true`), `to_huggingface` automatically slices out the `57` Shared-layer indexer slots so the exported safetensors contain `self_attn.indexer.*` weights only on the `21` Full layers, while preserving `float32` `e_score_correction_bias`, canonical `rope_parameters.rope_type="default"`, `eos_token_id=[154820, 154827, 154829]`, and `pad_token_id=154820`.

### 4. Forward-pass logit verification

Compare MaxText forward-pass logits against Hugging Face golden logits using `tests.utils.forward_pass_logit_checker` (see [`2_test_glm5.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/glm5/glm5.2-744b/2_test_glm5.sh)). Because `indexer_topk` defaults to `2048`, short prompts with `max_target_length <= 2048` hit the dense causal shortcut unless you either test at `max_target_length > 2048` or override `indexer_topk` below the unpadded prompt length (for example `indexer_topk=2` at `max_target_length=512`, paired with golden logits generated using matching `index_topk=2` via `GENERATE_GOLDEN_LOGITS=true bash 1_test_glm5.sh`):

```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=glm52_logits_check \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    model_name=glm5.2-744b \
    tokenizer_path=zai-org/GLM-5.2 \
    override_model_config=true \
    scan_layers=true \
    attention=dot_product \
    per_device_batch_size=1 \
    max_prefill_predict_length=512 \
    max_target_length=512 \
    indexer_topk=2 \
    async_checkpointing=false \
    sparse_matmul=false \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=4 \
    checkpoint_storage_concurrent_gb=1024 \
    weight_dtype=bfloat16 \
    dtype=bfloat16 \
    matmul_precision=highest \
    float32_logits=true \
    float32_qk_product=true \
    --golden_logits_path=${GOLDEN_LOGITS_PATH} \
    --atol=1.5 \
    --rtol=1.5 \
    --max_kl_div=0.1
```

## Pre-training and fine-tuning

### Distributed pre-training with scanned layers & IndexShare

Run distributed pre-training for `glm5.2-744b` (or `glm5.1-744b` with `tokenizer_path=zai-org/GLM-5.1`) using `scan_layers=true` and hybrid FSDP + Expert Parallelism. Setting `max_target_length=4096` (`> indexer_topk=2048`) activates sparse top-$k$ attention and Cross-Layer IndexShare across all `78` layers; setting `indexer_loss_scaling_factor > 0` enables multi-layer KL distillation into the `21` donor Full indexers:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=glm52_pretrain \
    model_name=glm5.2-744b \
    override_model_config=true \
    dataset_type=synthetic \
    tokenizer_type=huggingface \
    tokenizer_path=zai-org/GLM-5.2 \
    per_device_batch_size=1 \
    max_target_length=4096 \
    indexer_topk=2048 \
    indexer_loss_scaling_factor=0.01 \
    indexer_sparse_training=true \
    scan_layers=true \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=8 \
    allow_split_physical_axes=true \
    use_iota_embed=true \
    remat_policy=custom \
    decoder_layer_input=offload \
    enable_checkpointing=false \
    steps=10
```

### Full fine-tuning from a converted checkpoint

To fine-tune from a converted scanned Orbax checkpoint (`SCANNED_CKPT_PATH`), pass `load_parameters_path` and point to your dataset (for example, Grain TFRecords):

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=glm52_finetune \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    model_name=glm5.2-744b \
    dataset_type=grain \
    grain_file_type=tfrecord \
    dataset_path=${DATASET_PATH} \
    tokenizer_type=huggingface \
    tokenizer_path=zai-org/GLM-5.2 \
    per_device_batch_size=1 \
    max_target_length=4096 \
    scan_layers=true \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=8 \
    allow_split_physical_axes=true \
    remat_policy=custom \
    decoder_layer_input=offload \
    async_checkpointing=false \
    enable_checkpointing=true \
    steps=100
```

## Checkpoint unrolling and decoding

### Unrolling a scanned checkpoint (`generate_param_only_checkpoint.py`)

Training saves full-state scanned checkpoints (`scan_layers=true`) containing stacked `dense_layers` (`3` layers), stacked `moe_layers` (`75` layers), and optimizer state. Before running unscanned autoregressive decoding, convert the full training checkpoint (`load_full_state_path`, or a scanned parameter-only checkpoint via `load_parameters_path`) into an unscanned parameter-only checkpoint with `force_unroll=true`:

```bash
python3 -m maxtext.utils.generate_param_only_checkpoint src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=glm52_unrolled_param_only \
    load_full_state_path=${FULL_STATE_CKPT_PATH} \
    model_name=glm5.2-744b \
    scan_layers=true \
    force_unroll=true \
    async_checkpointing=false \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=8
```

For `DecoderBlockType.GLM5` (and `DEEPSEEK`), `generate_param_only_checkpoint` with `force_unroll=true`:

1. Unrolls `dense_layers` into `dense_layers_0..2` and `moe_layers` into `moe_layers_0..74`.
2. Slices per-layer `MoEBiasVar` along axis `0` while keeping router bias and indexer `k_norm` / `weights_proj` leaves in `float32` (casting standard `nnx.Param` weights to `bfloat16`).
3. Prunes `self_attention.indexer` leaves from all `57` Shared (`S`) layers when `use_index_share=true`, producing an unscanned checkpoint at `${BASE_OUTPUT_DIRECTORY}/glm52_unrolled_param_only/checkpoints/0/items` ready for `scan_layers=false` (`prune_shared_indexers=true`) restore.

### Autoregressive decoding (`inference.decode`)

Run autoregressive decoding from an unscanned checkpoint (produced either by `to_maxtext` with `scan_layers=false` or by `generate_param_only_checkpoint` with `force_unroll=true`) using compressed MLA KV caching (`mla_naive_kvcache=false`). Note that to exercise sparse top-$k$ selection and IndexShare KV-cache reuse on short decode smoke tests (`max_target_length <= 2048`), pass `override_model_config=true indexer_topk=4` (as in `2_test_glm5.sh`); for production long-context decoding (`> 2048` tokens), keep the default `indexer_topk=2048`:

```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=glm52_decode \
    model_name=glm5.2-744b \
    tokenizer_type=huggingface \
    tokenizer_path=zai-org/GLM-5.2 \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=false \
    attention=dot_product \
    sparse_matmul=false \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=1 \
    max_prefill_predict_length=64 \
    max_target_length=128 \
    ici_fsdp_parallelism=-1 \
    ici_tensor_parallelism=1 \
    ici_expert_parallelism=4 \
    checkpoint_storage_concurrent_gb=1024 \
    mla_naive_kvcache=false \
    prompt="The capital of France is"
```

## References

- **Model configurations**: [`glm5.1-744b.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/models/glm5.1-744b.yml) | [`glm5.2-744b.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/models/glm5.2-744b.yml)
- **Model & IndexShare implementation**: [`src/maxtext/models/glm5.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/models/glm5.py) | [`src/maxtext/utils/index_share_utils.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/index_share_utils.py)
- **End-to-end scripts**: [`1_test_glm5.sh` (CPU conversion)](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/glm5/glm5.2-744b/1_test_glm5.sh) | [`2_test_glm5.sh` (TPU logit check, pre-training, and decode)](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/glm5/glm5.2-744b/2_test_glm5.sh)
