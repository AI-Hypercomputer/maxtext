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
* **Freezes the pre-trained model weights**, preserving the original knowledge.
* **Injects trainable rank decomposition matrices** into the Transformer layers.

This approach **greatly reduces the number of trainable parameters** required for downstream tasks, making the process faster and more memory-efficient.

This tutorial provides step-by-step instructions for setting up the environment and performing LoRA fine-tuning on a Hugging Face dataset using MaxText. 

We use [Tunix](https://github.com/google/tunix), a JAX-based library, to power these post-training tasks.

In this tutorial we use a single host TPU VM such as `v6e-8/v5p-8`. Let's get started!

**Note:** Since **qwix** support has been recently integrated into the **main branch**, you must **clone** the latest source code and install it in **editable mode** to ensure all dependencies are correctly linked.

```sh
# Install Qwix from source
git clone https://github.com/google/qwix.git
cd qwix
uv pip install -e .
```

## Setup environment variables

Set the following environment variables before running LoRA Fine-tuning.

```sh
# -- Model configuration --
export PRE_TRAINED_MODEL=<MaxText Model> # e.g., 'gemma3-4b'

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export RUN_NAME=<name for this run> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
export STEPS=<number of fine-tuning steps to run> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<batch size per device> # e.g., 1
export HF_TOKEN=<Hugging Face Access Token>
export LORA_RANK=<dimension of the low-rank update matrices> # e.g., 16
export LORA_ALPHA=<scaling factor for LoRA weights> # e.g., 32.0
export LEARNING_RATE=<step size for the optimizer> # e.g., 3e-6
export MAX_TARGET_LENGTH=<maximum sequence length for input and output tokens> # e.g., 1024
export WEIGHT_DTYPE=<data type for storing model weights> # e.g., bfloat16
export DTYPE=<data type for numerical computations> # e.g., bfloat16

# -- Dataset configuration --
export DATASET_NAME=<Hugging Face dataset name> # e.g., openai/gsm8k
export TRAIN_SPLIT=<data split for train> # e.g., train
export HF_DATA_DIR=<The directory or sub-config within the Hugging Face dataset> # e.g., main
export TRAIN_DATA_COLUMNS=<data columns to train on> # e.g., ['question','answer']

# -- LoRA Conversion configuration (Optional) --
export HF_LORA_ADAPTER_PATH=<hf_repo_id_or_local_path> # e.g., 'username/adapter-name'
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer to the steps in [Hugging Face to MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/checkpointing_solutions/convert_checkpoint.html#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Run a Fresh LoRA Fine-Tuning on Hugging Face Dataset

Once your environment variables and checkpoints are ready, you can start the LoRA fine-tuning process.

Execute the following command to begin training:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name="${RUN_NAME?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    model_name="${PRE_TRAINED_MODEL?}" \
    load_parameters_path="${PRE_TRAINED_MODEL_CKPT_PATH?}" \
    hf_access_token="${HF_TOKEN?}" \
    hf_path="${DATASET_NAME?}" \
    train_split="${TRAIN_SPLIT?}" \
    hf_data_dir="${HF_DATA_DIR?}" \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps="${STEPS?}" \
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE?}" \
    max_target_length="${MAX_TARGET_LENGTH?}" \
    learning_rate="${LEARNING_RATE?}" \
    weight_dtype="${WEIGHT_DTYPE?}" \
    dtype="${DTYPE?}" \
    enable_nnx=True \
    pure_nnx_decoder=True \
    enable_lora=True \
    lora_rank="${LORA_RANK?}" \
    lora_alpha="${LORA_ALPHA?}" \
    scan_layers=True
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## (Optional) Resume from a previous LoRA checkpoint

If you want to resume training from a previous run or further fine-tune an existing LoRA adapter, you can specify the LoRA checkpoint path.

### Step 1: Convert HF LoRA adapter to MaxText format

If your LoRA adapter is currently in Hugging Face format, you must convert it to MaxText format before it can be loaded. Use the provided conversion script:

```sh
python3 -m maxtext.checkpoint_conversion.hf_lora_to_maxtext \
    model_name="${PRE_TRAINED_MODEL?}" \
    hf_lora_adapter_path="${HF_LORA_ADAPTER_PATH?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    scan_layers=True
```
### Step 2: Set the restore path

Point `LORA_RESTORE_PATH` to the converted MaxText adapter directory (the directory containing the `0/items` or Orbax files).
- **load_parameters_path**: Points to the frozen base model weights (the original model).
- **lora_restore_path**: Points to the previous LoRA adapter weights you wish to load.

```sh
export LORA_RESTORE_PATH=<gcs_path_to_converted_adapter_items> # e.g., gs://my-bucket/run-1/checkpoints/0/items
```
### Step 3: Run LoRA Fine-Tuning with the Restore Path

Once your environment variables and checkpoints are ready, you can start the LoRA fine-tuning process.

Execute the following command to begin training:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name="${RUN_NAME?}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}" \
    model_name="${PRE_TRAINED_MODEL?}" \
    load_parameters_path="${PRE_TRAINED_MODEL_CKPT_PATH?}" \
    lora_restore_path="${LORA_RESTORE_PATH}" \
    hf_access_token="${HF_TOKEN?}" \
    hf_path="${DATASET_NAME?}" \
    train_split="${TRAIN_SPLIT?}" \
    hf_data_dir="${HF_DATA_DIR?}" \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps="${STEPS?}" \
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE?}" \
    max_target_length="${MAX_TARGET_LENGTH?}" \
    learning_rate="${LEARNING_RATE?}" \
    weight_dtype="${WEIGHT_DTYPE?}" \
    dtype="${DTYPE?}" \
    enable_nnx=True \
    pure_nnx_decoder=True \
    enable_lora=True \
    lora_rank="${LORA_RANK?}" \
    lora_alpha="${LORA_ALPHA?}" \
    scan_layers=True
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## (Optional) Convert Fine-tuned LoRA to Hugging Face Format

After completing the fine-tuning process, your LoRA weights are stored in MaxText/Orbax format. To use these weights with the Hugging Face ecosystem (e.g., for inference or sharing), convert them back using the `maxtext_lora_to_hf.py` script.

```sh
python3 -m maxtext.checkpoint_conversion.maxtext_to_hf_lora \
    model_name="${PRE_TRAINED_MODEL?}" \
    load_parameters_path="${BASE_OUTPUT_DIRECTORY?}/${RUN_NAME?}/checkpoints/<step_number>/model_params" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY?}/hf_lora_adapter" \
    lora_rank="${LORA_RANK?}" \
    lora_alpha="${LORA_ALPHA?}"
```

- ```load_parameters_path```: Point this to the specific checkpoint directory (e.g., ```.../checkpoints/1000/items```) that you want to export.
- ```base_output_directory```: The local or GCS directory where the Hugging Face ```adapter_model.safetensors``` and ```adapter_config.json``` will be saved.
- ```lora_rank``` / ```lora_alpha```: Must match the values used during the training phase to ensure the ```adapter_config.json``` is generated correctly.