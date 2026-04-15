<!--
 Copyright 2024 Google LLC

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

# Full fine-tuning on single-host TPUs

Full Fine-Tuning (FFT) is a common technique used in post-training to adapt a pre-trained Large Language Model (LLM) to a specific downstream task or dataset. In this process, all the parameters (weights) of the original model are "unfrozen" and updated during training on the new task-specific data. This allows the entire model to adjust and specialize, potentially leading to the best performance on the new task.

This tutorial demonstrates step-by-step instructions for setting up the environment, convert checkpoint and then training the model on a Hugging Face dataset using FFT.

In this tutorial we use a single host TPU VM such as `v6e-8/v5p-8`. Let's get started!

## Install dependencies

For instructions on installing MaxText on your VM, please refer to the [official documentation](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/install_maxtext.html) and use the `maxtext[tpu]` installation path to include all necessary dependencies.

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
export MODEL=<MaxText Model> # e.g., 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<gcs bucket path> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<Name for this run>
```

## Hugging Face checkpoint to Maxtext checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/checkpointing_solutions/convert_checkpoint.html#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # gs://my-bucket/my-checkpoint-directory/0/items
```

## Dataset

MaxText provides examples to work with [Common Crawl](https://commoncrawl.org/). The dataset is available in TFRecords format in a cloud bucket. MaxText provides scripts to copy the dataset to a Google Cloud Storage Bucket.

### Common Crawl (c4) dataset setup

Run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for downloading and retrieving the dataset and the other for storing the logs.
2. Download the dataset in your gcs bucket.

MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them.

```sh
export PROJECT_ID=<Google Cloud Project ID>
export DATASET_GCS_BUCKET=<GCS for dataset> # e.g., gs://my-bucket/my-dataset

bash tools/data_generation/download_dataset.sh ${PROJECT_ID?} ${DATASET_GCS_BUCKET?}
```

The above will download the c4 dataset to the GCS BUCKET.

## Sample Full Fine tuning script

Below is a sample training script.

```sh
python3 -m maxtext.trainers.pre_train.train \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  model_name=${MODEL?} \
  dataset_path=${DATASET_GCS_BUCKET?} \
  async_checkpointing=False  \
  steps=10 per_device_batch_size=1
```

You can find some [end to end scripts here](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu).
These scripts can provide a reference point for various scripts.

## Parameters to achieve high MFU

This content is in progress.
