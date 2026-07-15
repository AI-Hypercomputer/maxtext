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

# Native NNX LoRA on single-host TPUs

**Native Low-Rank Adaptation (LoRA)** under pure Flax NNX in MaxText provides a highly optimized, state-of-the-art parameter-efficient fine-tuning (PEFT) framework.

Unlike traditional Linen/Tunix adapter wrappers, Native NNX LoRA operates by directly wrapping NNX modules. This allows for:

- **Zero adapter-wrapping overhead**: Cleaner model codebases and simplified parameter matching.
- **Native Checkpoint Save and Restore**: Full out-of-the-box compatibility with Orbax checkpointers, allowing frozen base weights and active adapter parameters to be saved/loaded seamlessly.
- **Int8 Weight Quantization (QLoRA)**: Full support for memory-efficient 8-bit quantization during fine-tuning.

This tutorial provides step-by-step instructions for performing native LoRA/QLoRA fine-tuning and pre-training on single-host TPUs using pure Flax NNX.

______________________________________________________________________

## 🚀 Quick Experimentation with Notebooks

For interactive playground setups on Google Colab or local JupyterLab, we provide fully detailed demo notebooks:

- **Qwen3 Native LoRA Demo**: [qwen3_native_lora_demo.ipynb](../../../src/maxtext/examples/qwen3_native_lora_demo.ipynb)
- **Gemma4 Native LoRA Demo**: [gemma4_native_lora_demo.ipynb](../../../src/maxtext/examples/gemma4_native_lora_demo.ipynb)

______________________________________________________________________

## Setup environment variables

Log in to Hugging Face. Provide your access token when prompted:

```bash
hf auth login
```

Set the following environment variables before running LoRA Fine-tuning.

```sh
# -- Model configuration --
export MODEL_NAME=<MODEL_NAME> # e.g., 'qwen3-0.6b' or 'gemma4-e2b'
export TOKENIZER_PATH=<TOKENIZER_PATH> # e.g., 'Qwen/Qwen3-0.6B' or 'google/gemma-4-E2B-it'

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/my-output-directory or /path/to/my-output-directory
export RUN_NAME=<RUN_NAME> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
export STEPS=<STEPS> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<BATCH_SIZE_PER_DEVICE> # e.g., 1
export LORA_RANK=<LORA_RANK> # e.g., 16
export LORA_ALPHA=<LORA_ALPHA> # e.g., 32.0
export LEARNING_RATE=<LEARNING_RATE> # e.g., 3e-6
export MAX_TARGET_LENGTH=<MAX_TARGET_LENGTH> # e.g., 1024

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., openai/gsm8k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train
export HF_DATA_DIR=<DATASET_PATH> # e.g., main
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., "['question','answer']"
```

______________________________________________________________________

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items or /path/to/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer to the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a Hugging Face checkpoint to MaxText. Similar to Option 1, you can set the following environment variable and move on.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items or /path/to/my-model-checkpoint/0/items
```

______________________________________________________________________

## Run Native LoRA Fine-Tuning

Execute the following command to begin LoRA fine-tuning on a Hugging Face dataset (e.g. GSM8K) using the native SFT entrypoint `train_sft_native.py`:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft_native \
    src/maxtext/configs/post_train/sft.yml \
    run_name="${RUN_NAME?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    model_name="${MODEL_NAME?}" \
    load_parameters_path="${MAXTEXT_CKPT_PATH?}" \
    tokenizer_path="${TOKENIZER_PATH?}" \
    hf_path="${DATASET_NAME?}" \
    train_split="${TRAIN_SPLIT?}" \
    hf_data_dir="${HF_DATA_DIR?}" \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps="${STEPS?}" \
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE?}" \
    max_target_length="${MAX_TARGET_LENGTH?}" \
    learning_rate="${LEARNING_RATE?}" \
    weight_dtype=bfloat16 \
    dtype=bfloat16 \
    formatting_func_path="maxtext.input_pipeline.instruction_data_processing.math_qa_formatting" \
    formatting_func_kwargs="{'template_path': 'src/maxtext/examples/chat_templates/math_qa.json'}" \
    lora.enable_lora=True \
    lora.lora_rank="${LORA_RANK?}" \
    lora.lora_alpha="${LORA_ALPHA?}"
```

______________________________________________________________________

## Run Native Pre-training with QLoRA (8-bit Quantization)

To run a standard native pre-training loop with memory-efficient 8-bit quantized weights, execute:

```sh
python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    run_name="native_qlora_pretrain_demo" \
    model_name="gemma4-e2b" \
    scan_layers=False \
    steps=10 \
    dataset_type="synthetic" \
    per_device_batch_size=1 \
    max_target_length=32 \
    enable_checkpointing=True \
    checkpoint_period=5 \
    base_output_directory="/tmp/native_qlora_pretrain_checkpoint" \
    attention="dot_product" \
    weight_dtype="bfloat16" \
    dtype="bfloat16" \
    lora.enable_lora=True \
    lora.lora_weight_qtype="int8" \
    lora.lora_tile_size=32 \
    lora.lora_rank=4 \
    lora.lora_alpha=8.0
```

______________________________________________________________________

## ⚙️ LoRA/QLoRA Configuration Reference

All low-rank adaptation properties are prefixed under the `lora.` namespace inside the configuration. The key arguments are:

| Parameter                | Type    | Default | Description                                                 |
| ------------------------ | ------- | ------- | ----------------------------------------------------------- |
| `lora.enable_lora`       | `bool`  | `False` | Enables/Disables native LoRA wrapping.                      |
| `lora.lora_rank`         | `int`   | `4`     | The low-rank dimension ($r$) of the adapters.               |
| `lora.lora_alpha`        | `float` | `8.0`   | Scaling hyperparameter ($\alpha$) for the low-rank updates. |
| `lora.lora_weight_qtype` | `str`   | `""`    | Set to `"int8"` to enable 8-bit quantized weights (QLoRA).  |
| `lora.lora_tile_size`    | `int`   | `32`    | Tiling dimension for quantized linear layers.               |
