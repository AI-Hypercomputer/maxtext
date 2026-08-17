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

# Gemma4

Gemma is a family of open models built by Google DeepMind. [Gemma 4](https://ai.google.dev/gemma) models are multimodal, handling text and image input and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants, featuring a context window of up to 256K tokens and multilingual support in over 140 languages.

Gemma 4 in MaxText is available in five sizes — two small variants (E2B, E4B), the unified 12B, and the larger Dense (31B) / Mixture-of-Experts (MoE) (26B A4B) configurations — and is well-suited for tasks like text generation, coding, and reasoning. The models are designed for enhanced performance and efficiency, capable of running on environments ranging from laptops and servers.

We provide examples for checkpoint conversion scripts at [tests/end_to_end/tpu/gemma4](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu/gemma4).

## 12B unified variant

`gemma4-12b` ships upstream as the `gemma4_unified` Hugging Face architecture ([google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B)). Its **text tower is the dense Gemma 4 tower** — the same period-6 attention pattern (5 sliding + 1 global), per-layer scalar, wider global head dim and shared global K/V projections — so it reuses `decoder_block: "gemma4"` unchanged.

The **vision path is what differs**. There is no SigLIP-style ViT: the config sets `vision_encoder_block: "gemma4_unified"`, which selects `Gemma4UnifiedVisionEmbedder`. That module projects raw pixel patches straight into the language-model space:

```
patches -> LayerNorm -> Dense -> LayerNorm -> +factorized position embedding -> LayerNorm
```

followed by the same `Gemma4VisionProjector` (`RMSNorm` without scale, then a dense projection) the ViT variants use. Upstream cuts 16px "teacher" patches and merges them in 3x3 groups before the model sees them; MaxText cuts `patch_size_for_vit: 48` patches directly from the resized image, which is the identical operation with the identical element ordering. At the standard 672x960 input that yields 14x20 = 280 soft tokens, matching the upstream `num_soft_tokens`.

Three practical notes:

- Conversion and inference need a `transformers` build that ships `transformers.models.gemma4_unified`. Without it, `hf_model_configs.py` still falls back to a raw config so **checkpoint conversion works**, but loading the reference HF model for logit comparison does not.
- MaxText converts the text and vision modalities. The checkpoint's audio projection (`model.embed_audio.embedding_projection.weight`) is intentionally not converted.
- The multimodal logit check skips the image placeholder positions. An image placeholder is always followed by another image token, so the next-token distribution there is never trained and is badly conditioned: perturbing the vision embeddings by their float32 noise floor (~1e-4 relative) moves those logits by tens of nats, while text positions move by less than 1e-2. `forward_pass_logit_checker` masks them, the same way it already does for Qwen3.

**Status of the logit checks.** The text-only path is verified: `max_kl_div` is 2.4e-05 on the short prompts and 2.9e-03 on the long RoPE-probing prompt, against a 0.03 threshold — but only with `matmul_precision=highest`. At the default `matmul_precision` the long prompt reaches 8.6e-01, so the `12b/` scripts pass `matmul_precision=highest float32_logits=True float32_qk_product=True` explicitly. The multimodal path is also verified: `12b/test_gemma4_to_hf.sh` runs with `USE_MULTIMODAL=true` and
reports `max KL divergence = 3.7e-03` against the 0.1 threshold, with the top-10 tokens matching
exactly (`jaccard_similarity 1.0`).

**Multimodal masking: 12b keeps the image block on global layers.** Where the bidirectional
image-block overlay applies differs between the two Gemma 4 architectures, and MaxText follows the
Hugging Face reference for each:

| architecture | bidirectional image block applies in |
|---|---|
| `gemma4` (26b / 31b / e2b / e4b) | **sliding layers only** — global layers stay causal |
| `gemma4_unified` (12b) | **both** sliding *and* global layers |

`Gemma4Model.forward` builds masks with `create_masks_for_vision_model` (sliding only);
`Gemma4UnifiedModel.forward` uses the generic `create_masks_for_generate`, which overlays both.
`Gemma4DecoderLayer` mirrors that: it drops `bidirectional_mask` on non-`LOCAL_SLIDING` layers
except when the config is `gemma4_unified`. Golden logits are a plain `model(**inputs)` forward pass
for every variant.

Getting this wrong for 12b is not subtle: applying the sliding-only rule there costs KL ≈ 0.75 on
the text after the image and only ~50% next-token argmax agreement.

`tests/unit/gemma4_unified_vision_test.py::Gemma4BidirectionalMaskTest` pins all three behaviours
(sliding keeps the mask; 12b global keeps it; 31b global drops it), so a silent flip in either
direction — or the 12b carve-out leaking into the other variants — fails CI.

## E2B / E4B small variants

E2B and E4B are the **edge-device** ("E") variants — both **dense** (no MoE) — built for on-device deployment. The configs at `src/maxtext/configs/models/gemma4-e2b.yml` and `src/maxtext/configs/models/gemma4-e4b.yml` introduce two architecture features beyond the dense Gemma 4 path:

- **Per-Layer Embedding (PLE).** Each decoder layer consumes a per-layer slice of an extra embedding tensor injected by `Gemma4SmallPLE`. Controlled by `hidden_size_per_layer_input` and `vocab_size_per_layer_input`.
- **KV sharing.** The last `num_kv_shared_layers` layers reuse the K / V projections from the most recent non-shared layer of the same attention type (sliding↔sliding, full↔full). E2B additionally widens the MLP on shared layers (`use_double_wide_mlp: true`).

Both features are tied to per-layer state that is not expressible inside `nn.scan`, so E2B / E4B require `scan_layers=false`. Multimodal is currently gated off for these variants; the model validator raises a clear error if you try to enable `use_multimodal=true`.

## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining Gemma4-26B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_path=${DATASET_PATH?} tokenizer_path=google/gemma-4-26b-a4b-it per_device_batch_size=1 run_name=runner_pretrain_gemma4_26b steps=10 enable_checkpointing=false sharding_tolerance=0.03
```

### 12B tuned throughput config (Ironwood, tpu7x-4x4x4)

`tests/end_to_end/tpu/gemma4/12b/train_v7x_gemma4_12b.sh` reproduces the tuned gemma4-12b
configuration on one 64-chip Ironwood slice: **4.703 s/step, 521.3 TFLOP/s/device,
6968 tokens/s/device — 45.3% MFU** (peak 1150 bf16 TFLOP/s/device).

```sh
BASE_OUTPUT_DIRECTORY=gs://your-bucket/gemma4 tests/end_to_end/tpu/gemma4/12b/train_v7x_gemma4_12b.sh
```

The mesh is the whole story — `ici_data_parallelism=4 ici_fsdp_parallelism=32` beats pure
FSDP-128 by +4.4 pp (5.211 → 4.703 s/step) by cutting per-device weight all-gather volume, and
it is a narrow peak (dp2 +0.5 pp, dp8 −0.3 pp). `num_vocab_tiling=8` is a requirement rather
than a knob: untiled 262k-vocab logits are ~34 GB at pbs 8 and abort. Relaxing `remat_policy`,
setting `context=remat`, or moving to seq 8192 all measured *worse* — see the script's comments
for the numbers before changing any of them.

### Load balance loss (MoE only)
Gemma4-26B is a Mixture-of-Experts model and uses an auxiliary load balance loss during training to encourage uniform expert utilization. The weight is controlled by `load_balance_loss_weight` and defaults to `0.001` in `src/maxtext/configs/models/gemma4-26b.yml`. To tune or disable it, override from the command line, for example:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b <other flags> load_balance_loss_weight=0.01
```

Set `load_balance_loss_weight=0.0` to turn the auxiliary loss off. This flag has no effect on the dense Gemma4-31B model.

## Checkpoint Conversion
To obtain the Gemma4 model weights, you can access them on Hugging Face (e.g., [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)). You will need to accept the Gemma4 license through your Hugging Face account and provide your Hugging Face access token (as `HF_TOKEN`) for authentication. You can then convert them directly into a MaxText compatible format. Here's an example of converting the model weights using the conversion script (`tests/end_to_end/tpu/gemma4/26b/convert_gemma4.sh`):

```sh
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=gemma4-26b \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET}/26b/converted/${idx} \
    use_multimodal=true \
    scan_layers=false \
    --lazy_load_tensors=False
```

This will convert the checkpoints and save them to a Google Cloud Storage (GCS) bucket.

### E2B / E4B conversion (text-only)

For the small variants, drop `use_multimodal=true` — multimodal is not supported. Conversion scripts live at `tests/end_to_end/tpu/gemma4/e2b/convert_gemma4.sh` (instruction-tuned) and `tests/end_to_end/tpu/gemma4/e2b/convert_gemma4_base.sh` (pre-trained base), and the same pair under `tests/end_to_end/tpu/gemma4/e4b/`. They follow the same shape as the larger Gemma 4 scripts. Example:

```sh
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=gemma4-e2b \
    hf_access_token=${HF_TOKEN} \
    --hf_model_path=${HF_MODEL} \
    base_output_directory=${MODEL_BUCKET}/e2b/converted/${idx} \
    use_multimodal=false \
    scan_layers=false
```

### 12B conversion

Conversion scripts live at `tests/end_to_end/tpu/gemma4/12b/convert_gemma4.sh` (HF to MaxText) and
`tests/end_to_end/tpu/gemma4/12b/test_gemma4_to_hf.sh` (the reverse round-trip). Converting the
vision weights requires `--lazy_load_tensors=False`, which the script sets for you whenever
`USE_MULTIMODAL=true`:

```sh
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=gemma4-12b \
    hf_access_token=${HF_TOKEN} \
    --hf_model_path=${HF_MODEL} \
    base_output_directory=${MODEL_BUCKET}/12b/converted/${idx} \
    use_multimodal=true \
    scan_layers=false \
    --lazy_load_tensors=False
```

Each `convert_gemma4.sh` script ends with a `forward_pass_logit_checker` run that loads the just-saved MaxText checkpoint and the original HF model on the fly and asserts that the two produce equivalent logits (`--max_kl_div=0.03`). The round-trip is the recommended smoke test after touching the model code, the param map, or either YAML.

## Fine-tuning
After the conversion, you will have a MaxText compatible checkpoint which allows you to fine-tune it with different datasets. For more comprehensive guides, please refer to our tutorials on [Multimodal Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/multimodal.md#supervised-fine-tuning) and [Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/sft.md). One example command to fine-tune a Gemma4-26B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_type=synthetic tokenizer_type=huggingface load_parameters_path=${CONVERTED_CHECKPOINT?} tokenizer_path=google/gemma-4-26b-a4b-it per_device_batch_size=1 run_name=runner_finetune_gemma4_26b steps=10 enable_checkpointing=true sharding_tolerance=0.03
```

## Inference

Run Gemma 4 inference on vLLM using the MaxText model implementation, via the
[out-of-tree](https://github.com/vllm-project/tpu-inference/blob/main/docs/getting_started/out-of-tree.md)
vLLM model plugin. Weights are loaded directly from a MaxText (Orbax) checkpoint. See the general
[Inference Tutorial](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/inference.md)
for installation and online/RL workflows; this guide is the Gemma 4 quickstart.

### Installation

Install MaxText with the `tpu-post-train` extra (it provides the vLLM adapter plugin and the pinned `tpu-inference` / `vllm` versions), then verify the plugin is present:

```sh
pip show maxtext_vllm_adapter
```

If it is missing, run:

```sh
install_tpu_post_train_extra_deps
```

### Offline inference

`maxtext.inference.vllm_decode` runs offline decode through vLLM (it sets `NEW_MODEL_DESIGN=1` for you; you only set it yourself for direct `vllm serve`). Set `HF_HUB_OFFLINE=1` if the tokenizer is already cached locally. Pass `src/maxtext/configs/base.yml` as the config — the vLLM adapter applies `src/maxtext/configs/inference/vllm.yml` internally for the model. The vLLM path requires an **unscanned** checkpoint, so pass `scan_layers=False` (as the examples below do).

Dense models (e.g. `gemma4-31b`, `gemma4-12b`):

```sh
python3 -m maxtext.inference.vllm_decode src/maxtext/configs/base.yml \
    model_name=gemma4-31b \
    tokenizer_path=google/gemma-4-31b-it \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    ici_tensor_parallelism=4 scan_layers=False \
    prompt="Who was Albert Einstein?" use_chat_template=True
```

MoE models (e.g. `gemma4-26b`) additionally require `prefuse_moe_weights=True`, which pre-fuses
the expert gate/up projections into the per-shard layout the fused-MoE kernel expects — required
for correct output with `ici_tensor_parallelism` > 1:

```sh
python3 -m maxtext.inference.vllm_decode src/maxtext/configs/base.yml \
    model_name=gemma4-26b \
    tokenizer_path=google/gemma-4-26b-a4b-it \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    ici_tensor_parallelism=4 scan_layers=False prefuse_moe_weights=True \
    prompt="Who was Albert Einstein?" use_chat_template=True
```

Set `model_name`/`tokenizer_path` to your variant (`gemma4-12b`, `gemma4-26b`, `gemma4-31b`) and
`ici_tensor_parallelism` to the number of chips — pass an explicit count (e.g. `4` on a v5p-8), not
`-1`, since `vllm_decode` forwards this value directly to vLLM's `tensor_parallel_size`.

#### 12B

`gemma4-12b` runs through the same dense path, and needs nothing MoE-specific:

```sh
python3 -m maxtext.inference.vllm_decode src/maxtext/configs/base.yml \
    model_name=gemma4-12b \
    tokenizer_path=google/gemma-4-12B \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    ici_tensor_parallelism=4 scan_layers=False \
    prompt="Albert Einstein was a"
```

Unlike the `-it` variants above, `google/gemma-4-12B` is a **base** repo with no chat template, so
`use_chat_template=True` fails with `tokenizer.chat_template is not set`; give it a plain
completion-style prompt instead.

The 12B has a single global-attention KV head (`global_num_kv_heads: 1`). The adapter replicates it
up to the tensor-parallel degree automatically — you will see
`Padding num_global_kv_heads from 1 to 4 ...` in the logs — so `ici_tensor_parallelism` may exceed
the KV head count without any manual override. Its 48 decoder layers also divide evenly by the
period-6 block pattern, which the RL rollout converter requires.

#### E2B / E4B

`gemma4-e2b` and `gemma4-e4b` run through the same `vllm_decode` entry point as the larger variants, but the `-it` fine-tunes need **three things** the larger models tolerate without:

1. **A system prompt** ([per the HF model card](https://huggingface.co/google/gemma-4-E2B-it)) — without it the `-it` checkpoints drift off-topic at any temperature.
2. **Stochastic sampling** `temperature=1.0, top_p=0.95, top_k=64` (the model card's recommended settings). Greedy decoding tends to loop on these small checkpoints, independent of the MaxText path.
3. **The full stop-token set.** The upstream `google/gemma-4-*-it` repos declare `eos_token_id: [1, 106, 50]` (`<eos>`, `<turn|>`, `<|tool_response>`). If a converted checkpoint only carries `eos_token_id: 1`, end-of-turn `<turn|>` is no longer registered as a stop and generation runs to `max_tokens`. Using the upstream repo id for `tokenizer_path` keeps the full stop list automatically. A local checkpoint dir works equally well — just verify its `generation_config.json` carries the full list.

The CLI form, using the `system_prompt=` flag and the model card's sampling params:

```sh
python3 -m maxtext.inference.vllm_decode src/maxtext/configs/base.yml \
    model_name=gemma4-e2b \
    tokenizer_path=google/gemma-4-e2b-it \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    ici_tensor_parallelism=1 scan_layers=False \
    system_prompt="You are a helpful assistant." \
    prompt="Who was Albert Einstein?" use_chat_template=True \
    decode_sampling_temperature=1.0 \
    decode_sampling_nucleus_p=0.95 \
    decode_sampling_top_k=64
```

Or via the Python API, useful for fixing a seed or stitching multiple requests:

```python
import maxtext.integration.vllm.maxtext_vllm_adapter as adapter
adapter.register()
from vllm import LLM
from vllm.sampling_params import SamplingParams
import transformers

llm = LLM(
    model="google/gemma-4-e2b-it",                     # tokenizer + HF config dir
    hf_overrides={"architectures": ["MaxTextForCausalLM"]},
    additional_config={
        "maxtext_config": {
            "model_name": "gemma4-e2b",                # or gemma4-e4b
            "scan_layers": False,
            "load_parameters_path": "${CONVERTED_CHECKPOINT}",
        }
    },
    tensor_parallel_size=1,                            # set to chip count (e.g. 4 on v5p-8)
    max_model_len=1024,
)

tok = transformers.AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
prompt = tok.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Who was Albert Einstein?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

out = llm.generate(
    [prompt],
    SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                   seed=42, max_tokens=300),
)
print(out[0].outputs[0].text)
```
