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
bash tools/setup/setup_post_training_requirements.sh
```

## Setup environment variables

Set the following environment variables before running SFT.

```sh
# -- Model configuration --
export PRE_TRAINED_MODEL=<model name> # e.g., 'llama3.1-8b'
export PRE_TRAINED_MODEL_TOKENIZER=<tokenizer path> # e.g., 'meta-llama/Llama-3.1-8B-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export RUN_NAME=<name for this run> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
export STEPS=<number of fine-tuning steps to run> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<batch size per device> # e.g., 1

# -- Dataset configuration --
export DATASET_NAME=<Hugging Face dataset name> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<data split for train> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<data columns to train on> # e.g., ['messages']
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Run SFT on Hugging Face Dataset

Now you are ready to run SFT using the following command:

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft src/MaxText/configs/sft.yml \
    run_name=${RUN_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} \
    load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH} \
    hf_access_token=${HF_TOKEN} \
    tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    steps=${STEPS} \
    hf_path=${DATASET_NAME} \
    train_split=${TRAIN_SPLIT} \
    train_data_columns=${TRAIN_DATA_COLUMNS} \
    profiler=xplane
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.
