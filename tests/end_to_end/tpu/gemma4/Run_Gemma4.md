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

Gemma 4 in MaxText is available in four sizes — two small variants (E2B, E4B) and the larger Dense (31B) / Mixture-of-Experts (MoE) (26B A4B) configurations — and is well-suited for tasks like text generation, coding, and reasoning. The models are designed for enhanced performance and efficiency, capable of running on environments ranging from laptops and servers.

We provide examples for checkpoint conversion scripts at [tests/end_to_end/tpu/gemma4](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu/gemma4).

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
    scan_layers=false
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

Each `convert_gemma4.sh` script ends with a `forward_pass_logit_checker` run that loads the just-saved MaxText checkpoint and the original HF model on the fly and asserts that the two produce equivalent logits (`--max_kl_div=0.03`). The round-trip is the recommended smoke test after touching the model code, the param map, or either YAML.

## Fine-tuning
After the conversion, you will have a MaxText compatible checkpoint which allows you to fine-tune it with different datasets. For more comprehensive guides, please refer to our tutorials on [Multimodal Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/multimodal.md#supervised-fine-tuning) and [Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/sft.md). One example command to fine-tune a Gemma4-26B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_type=synthetic tokenizer_type=huggingface load_parameters_path=${CONVERTED_CHECKPOINT?} tokenizer_path=google/gemma-4-26b-a4b-it per_device_batch_size=1 run_name=runner_finetune_gemma4_26b steps=10 enable_checkpointing=true sharding_tolerance=0.03
```

## Inference
For detailed instructions on running inference and decoding with MaxText, please refer to our [Inference Tutorial](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/inference.md).
