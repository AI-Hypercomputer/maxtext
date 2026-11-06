<!--
 Copyright 2023–2026 Google LLC

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

(decision-models)=

# Decision Models

With the emergence of JEV-style decision models, MaxText now supports a next-token scoring interface through `maxtext.inference.vllm_score`. This interface scores a fixed set of allowed answers and returns the selected answer and each option's probability. It can be used for classification, routing, and other bounded decisions with models supported by the MaxText on vLLM adapter.

This tutorial demonstrates the interface with LoRA weights from [Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B), a decision model built on Qwen3.5-9B. [Bespoke Nimble](https://github.com/bespokelabsai/nimble) is inspired by JEV's approach to making decisions directly, without generating reasoning first.

## How decision scoring works

1. **Define the decision.** Supply the context, question, and allowed answers in the prompt. Represent each answer with a single-token code, such as `A` for Yes and `B` for No. For schema-based decisions, map each code back to the corresponding enum or boolean value in your application.
2. **Prepare the answer boundary.** With `use_chat_template=true`, the scorer applies the tokenizer's chat template. `--enable_thinking=false` closes the reasoning block in Qwen-style templates so the next-token scores correspond to the answer.
3. **Score the candidates.** MaxText runs the model through vLLM on TPU. The scorer reads next-token log-probabilities; raw logits are not returned by this backend. Each text candidate must tokenize to exactly one additional token at the prompt boundary, without changing the prompt's existing tokens. Multi-token candidates are rejected.
4. **Normalize and select.** For candidate log-probabilities `l_i`, the scorer computes `p_i = exp(l_i - max(l)) / sum_j exp(l_j - max(l))` and selects the candidate with the highest score. This is equivalent to applying softmax to the candidates' raw logits because the full-vocabulary normalization cancels out.

`Option-prob` is normalized over **only the supplied candidates**, so these probabilities sum to one. `Log-prob` is the token's log-probability over the full vocabulary. The selected candidate can differ from the greedy next token when the model's highest-scoring token is outside the allowed set.

Use `--method=auto` to score the allowed answers. It retrieves scores from the top-k next tokens and automatically scores any missing candidates separately.

Each generation request produces at most one token. `max_target_length` limits the total sequence length. Prefix caching is disabled in the example because it has been observed to change next-token scores on hybrid models.

## Example: Qwen3.5-9B with Bespoke-Nimble LoRA weights

Follow the [inference installation instructions](inference.md#installation) to install `maxtext[tpu-post-train]` and the MaxText on vLLM adapter. Run the commands below from the MaxText repository root. Conversion and LoRA merging run on CPU; scoring runs on TPU.

### 1. Convert the Qwen3.5-9B base checkpoint

Choose separate output directories for the converted base and merged model. These can be local directories or GCS paths; the merged checkpoint must be accessible from the TPU VM.

```bash
export BASE_OUTPUT_PATH=/path/to/checkpoints/qwen3.5-9b
export OUTPUT_PATH=/path/to/checkpoints/bespoke-nimble-9b

JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=qwen3.5-9b \
  base_output_directory=$BASE_OUTPUT_PATH \
  use_multimodal=false scan_layers=false weight_dtype=bfloat16 \
  hardware=cpu skip_jax_distributed_system=true \
  checkpoint_storage_use_ocdbt=false \
  checkpoint_storage_use_zarr3=false \
  --hf_model_path=Qwen/Qwen3.5-9B \
  --save_dtype=bfloat16 --lazy_load_tensors=true

export CHECKPOINT_PATH=$BASE_OUTPUT_PATH/0/items
```

This downloads the Hugging Face base weights and saves a text-only, unscanned MaxText checkpoint. See [checkpoint conversion](../guides/checkpointing_solutions/convert_checkpoint.md) for additional conversion options.

### 2. Merge the LoRA weights

Pass both `load_parameters_path` and `hf_lora_adapter_path` to merge the adapter into the converted base weights. The resulting checkpoint contains the merged model weights and is saved under `$OUTPUT_PATH/0/items`.

```bash
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=qwen3.5-9b \
  load_parameters_path=$CHECKPOINT_PATH \
  hf_lora_adapter_path=bespokelabs/Bespoke-Nimble-9B \
  base_output_directory=$OUTPUT_PATH \
  use_multimodal=false scan_layers=false weight_dtype=bfloat16 \
  hardware=cpu skip_jax_distributed_system=true \
  checkpoint_storage_use_ocdbt=false \
  checkpoint_storage_use_zarr3=false \
  --lazy_load_tensors=true
```

### 3. Score the allowed answers on TPU

The prompt is simplified for this demo; use [Nimble's original context-and-schema template](https://github.com/bespokelabsai/nimble) to follow the model's intended decision-making format.

Use the same unscanned, bfloat16 settings as the converted checkpoint:

```bash
JAX_PLATFORMS=tpu python3 -m maxtext.inference.vllm_score \
  src/maxtext/configs/base.yml \
  model_name=qwen3.5-9b tokenizer_path=Qwen/Qwen3.5-9B \
  load_parameters_path=$OUTPUT_PATH/0/items \
  vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
  hbm_utilization_vllm=0.6 \
  scan_layers=false weight_dtype=bfloat16 max_target_length=2048 \
  use_chat_template=true \
  system_prompt='Answer with a single word.' \
  "prompt='The store offers refunds within 30 days of purchase. This item was bought 12 days ago. Is this item eligible for a refund? A: Yes, B: No'" \
  --candidates=A,B --top_k=10 --method=auto \
  --enable_thinking=false --enable_prefix_caching=false \
  --output_file=option_scores.jsonl
```

Keep literal quotes around prompt values containing `: ` so MaxText's YAML argument parser treats them as strings. The scorer prints the ranked options and writes the full result, including the top-k tokens, to `option_scores.jsonl`.

### Recorded run result

> **Question:** The store offers refunds within 30 days of purchase. This item was bought 12 days ago. Is this item eligible for a refund? A: Yes, B: No

The saved result for this prompt, with `system_prompt='Answer with a single word.'` and thinking disabled, selected **A (Yes)**:

```text
Option   Option-prob    Log-prob
'A'           99.86%     -0.0279
'B'            0.14%     -6.6315
Greedy next token: 'A' (id=32, log-prob=-0.0279)
Top-k next tokens:
  'A' (id=32, log-prob=-0.0279)
  'Yes' (id=9175, log-prob=-3.7042)
  'B' (id=33, log-prob=-6.6315)
  'Y' (id=56, log-prob=-8.1343)
  'No' (id=2665, log-prob=-8.4507)
  'yes' (id=9405, log-prob=-8.6459)
  '<think>' (id=248068, log-prob=-9.4559)
  'C' (id=34, log-prob=-9.7153)
  '<|im_end|>' (id=248046, log-prob=-10.0355)
  'a' (id=64, log-prob=-10.0631)
```

The model selected **A (Yes)**, indicating that the item is eligible for a refund under the stated policy. Both candidates appeared in the top ten next tokens, and the greedy next token was also `A`.

These are recorded values for this example; scores can vary with checkpoint, prompt, and runtime settings.
