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

# LoRA Fine-tuning on single-host TPUs

**Low-Rank Adaptation (LoRA)** is a Parameter-Efficient Fine-Tuning (PEFT) technique designed to optimize large language models while minimizing resource consumption.

Unlike traditional full-parameter fine-tuning, LoRA:

- **Freezes the pre-trained model weights**, preserving the original knowledge.
- **Injects trainable rank decomposition matrices** into the Transformer layers.

This approach **greatly reduces the number of trainable parameters** required for downstream tasks, making the process faster and more memory-efficient.

This tutorial provides step-by-step instructions for setting up the environment and performing LoRA fine-tuning on a Hugging Face dataset using MaxText.

We use [Tunix](https://github.com/google/tunix), a JAX-based library, to power these post-training tasks.

In this tutorial we use a single host TPU VM such as `v6e-8/v5p-8`. Let's get started!

## Setup environment variables

Login to Hugging Face. Provide your access token when prompted:

```bash
hf auth login
```

Set the following environment variables before running LoRA Fine-tuning.

```sh
# -- Model configuration --
export MODEL_NAME=<MODEL_NAME> # e.g., 'gemma3-4b'

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
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['question','answer']
export CHAT_TEMPLATE_PATH=<TEMPLATE_PATH> # e.g., maxtext/examples/chat_templates/math_qa.json

# -- LoRA Conversion configuration (Optional) --
export HF_LORA_ADAPTER_PATH=<HF_LORA_ADAPTER_PATH> # e.g., 'username/adapter-name'
```

## Customizing Trainable Layers (Optional)

By default, MaxText determines which layers to apply LoRA to based on the model's architecture by reading `src/maxtext/configs/post_train/lora_module_path.yml`.

If you need to fine-tune specific components (e.g., targeting only Attention layers to optimize memory usage), you can override these defaults through the following hierarchy:

### Configuration Hierarchy

1. **Command Line Argument**: Pass the `lora_module_path` argument directly in your training command. This is the most flexible way for experimental iterations.
2. **Task-Specific Config (`sft.yml`)**: Define the `lora_module_path` parameter in `src/maxtext/configs/post_train/sft.yml` to set a persistent configuration for your SFT runs.
3. **Global Defaults**: Automatic detection via the model-to-regex mapping defined in `lora_module_path.yml`.

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items or /path/to/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer to the steps in [Hugging Face to MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/checkpointing_solutions/convert_checkpoint.html#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have the correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items or /path/to/my-model-checkpoint/0/items
```

## Run a Fresh LoRA Fine-Tuning on Hugging Face Dataset

Once your environment variables and checkpoints are ready, you can start the LoRA fine-tuning process.

Execute the following command to begin training:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name="${RUN_NAME?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    model_name="${MODEL_NAME?}" \
    load_parameters_path="${MAXTEXT_CKPT_PATH?}" \
    hf_path="${DATASET_NAME?}" \
    train_split="${TRAIN_SPLIT?}" \
    hf_data_dir="${HF_DATA_DIR?}" \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps="${STEPS?}" \
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE?}" \
    max_target_length="${MAX_TARGET_LENGTH?}" \
    learning_rate="${LEARNING_RATE?}" \
    chat_template_path="${CHAT_TEMPLATE_PATH?}" \
    enable_nnx=True \
    pure_nnx_decoder=True \
    lora.enable_lora=True \
    lora.lora_rank="${LORA_RANK?}" \
    lora.lora_alpha="${LORA_ALPHA?}"
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## (Optional) Resume from a previous LoRA checkpoint

If you want to resume training from a previous run or further fine-tune an existing LoRA adapter, you can specify the LoRA checkpoint path.

### Step 1: Convert HF LoRA adapter to MaxText format

If your LoRA adapter is currently in Hugging Face format, you must convert it to MaxText format before it can be loaded. Use the integrated conversion utility:

```sh
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name="${MODEL_NAME?}" \
    hf_lora_adapter_path="${HF_LORA_ADAPTER_PATH?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}/converted_adapter" \
    hardware=cpu skip_jax_distributed_system=True
```

### Step 2: Set the restore path

Point `LORA_RESTORE_PATH` to the converted MaxText adapter directory (the directory containing the `0/items` or Orbax files).

- **load_parameters_path**: Points to the frozen base model weights (the original model).
- **lora_restore_path**: Points to the previous LoRA adapter weights you wish to load.

```sh
export LORA_RESTORE_PATH=<LORA_RESTORE_PATH> # e.g., gs://my-bucket/run-1/checkpoints/0/items or /path/to/run-1/checkpoints/0/items
```

### Step 3: Run LoRA Fine-Tuning with the Restore Path

Once your environment variables and checkpoints are ready, you can start the LoRA fine-tuning process.

Execute the following command to begin training:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name="${RUN_NAME?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    model_name="${MODEL_NAME?}" \
    load_parameters_path="${MAXTEXT_CKPT_PATH?}" \
    lora.lora_restore_path="${LORA_RESTORE_PATH?}" \
    hf_path="${DATASET_NAME?}" \
    train_split="${TRAIN_SPLIT?}" \
    hf_data_dir="${HF_DATA_DIR?}" \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps="${STEPS?}" \
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE?}" \
    max_target_length="${MAX_TARGET_LENGTH?}" \
    learning_rate="${LEARNING_RATE?}" \
    chat_template_path="${CHAT_TEMPLATE_PATH?}" \
    enable_nnx=True \
    pure_nnx_decoder=True \
    lora.enable_lora=True \
    lora.lora_rank="${LORA_RANK?}" \
    lora.lora_alpha="${LORA_ALPHA?}"
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## (Optional) Convert Fine-tuned LoRA to Hugging Face Format

After completing the fine-tuning process, your LoRA weights are stored in MaxText/Orbax format. To use these weights with the Hugging Face ecosystem (e.g., for inference or sharing), convert them back using the `to_huggingface.py` script.

```sh
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name="${MODEL_NAME?}" \
    lora.lora_restore_path="${BASE_OUTPUT_DIRECTORY?}/${RUN_NAME?}/checkpoints/<STEPS>/model_params" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}/hf_lora_adapter"
```

- `lora.lora_restore_path`: Point this to the specific checkpoint directory (e.g., `.../checkpoints/1000/items`) that you want to export.
- `base_output_directory`: The local or GCS directory where the Hugging Face `adapter_model.safetensors` and `adapter_config.json` will be saved.
- `lora.lora_rank` / `lora.lora_alpha`: Must match the values used during the training phase to ensure the `adapter_config.json` is generated correctly.
