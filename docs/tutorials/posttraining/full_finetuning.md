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

```sh
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Create virtual environment
export VENV_NAME=<your virtual env name> # e.g., maxtext_venv
pip install uv
uv venv --python 3.12 --seed $VENV_NAME
source $VENV_NAME/bin/activate

# 3. Install dependencies in editable mode
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps
```

## Setup environment variables

```sh
# -- Model configuration --
export MODEL_NAME=<model name> # e.g., 'llama3.1-8b'
export MODEL_TOKENIZER=<tokenizer path> # e.g., 'meta-llama/Llama-3.1-8B-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export RUN_NAME=<name for this run> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
```

## Hugging Face checkpoint to Maxtext checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # gs://my-bucket/my-checkpoint-directory/0/items
```

## Dataset

MaxText provides examples to work with [Common Crawl](https://commoncrawl.org/). The dataset is available in TFRecords format in a cloud bucket. MaxText provides scripts to copy the dataset to a Google Cloud Storage Bucket.

### Common Crawl (c4) dataset setup

Run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for downloading and retrieving the dataset and the other for storing the logs.
1. Download the dataset in your gcs bucket.

MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them.

```sh
export PROJECT=<Google Cloud Project ID>
export DATASET_GCS_BUCKET=<GCS for dataset> # e.g., gs://my-bucket/my-dataset

bash tools/data_generation/download_dataset.sh ${PROJECT} ${DATASET_GCS_BUCKET}
```

The above will download the c4 dataset to the GCS BUCKET.

## Sample Full Fine tuning script

Below is a sample training script.

```sh
python3 -m MaxText.train \
  src/MaxText/configs/base.yml \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${MODEL_CKPT_PATH} \
  model_name=${MODEL_NAME} \
  dataset_path=${DATASET_GCS_BUCKET} \
  async_checkpointing=False  \
  tokenizer_path=${MODEL_TOKENIZER} \
  hf_access_token=${HF_TOKEN} \
  steps=10 per_device_batch_size=1
```

You can find some [end to end scripts here](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu).
These scripts can provide a reference point for various scripts.

## Parameters to achieve high MFU

This content is in progress.
