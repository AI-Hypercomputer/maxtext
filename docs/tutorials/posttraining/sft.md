<!--
 Copyright 2023–2025 Google LLC

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

# SFT on single-host TPUs

Supervised fine-tuning (SFT) is a process where a pre-trained large language model is fine-tuned on a labeled dataset to adapt the model to perform better on specific tasks.

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the model on a Hugging Face dataset using SFT.

We use [Tunix](https://github.com/google/tunix), a JAX-based library designed for post-training tasks, to perform SFT.

In this tutorial we use a single host TPU VM such as `v6e-8/v5p-8`. Let's get started!

## Install MaxText and Post-Training dependencies

For instructions on installing MaxText with post-training dependencies on your VM, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) and use the `maxtext[tpu-post-train]` installation path to include all necessary post-training dependencies.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

## Setup environment variables

Login to Hugging Face. Provide your access token when prompted:

```bash
hf auth login
```

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g., 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

export STEPS=<STEPS> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<BATCH_SIZE_PER_DEVICE> # e.g., 1

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['messages']
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/checkpointing_solutions/convert_checkpoint.html#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Run SFT on Hugging Face Dataset

Now you are ready to run SFT using the following command:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name=${RUN_NAME?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} \
    steps=${STEPS?} \
    hf_path=${DATASET_NAME?} \
    train_split=${TRAIN_SPLIT?} \
    train_data_columns=${TRAIN_DATA_COLUMNS?} \
    profiler=xplane
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.
