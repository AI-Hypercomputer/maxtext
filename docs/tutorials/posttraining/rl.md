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

# Reinforcement Learning on single-host TPUs

This tutorial demonstrates step-by-step instructions for setting up the
environment and then training the Llama3.1 8B-IT model on the GSM8K math
reasoning dataset using a single host TPU-VM such as `v6e-8/v5p-8`.

We utilize two RL algorithms, implemented via the Tunix library, to enhance the
model's reasoning capabilities:

- **Group Relative Policy Optimization (GRPO)**: GRPO is an RL algorithm
  designed to enhance the reasoning abilities of LLMs. It is a variant of
  Proximal Policy Optimization (PPO) that reduces memory usage by eliminating
  the need for a separate value function model. GRPO works by generating
  multiple responses for a given prompt, evaluating these responses using a
  reward model, and then calculating a relative advantage based on the group's
  performance to update the policy.

- **Group Sequence Policy Optimization (GSPO)**: GSPO is an RL algorithm that
  improves training efficiency and performance of LLMs by using sequence-level
  importance ratios and operations. GSPO defines the importance ratio based on
  sequence likelihood and performs sequence-level clipping, rewarding, and
  optimization.

For efficient model inference and response generation during this process, we
rely on the vLLM library.

Let's get started!

## Install MaxText and post-training dependencies

For instructions on installing MaxText with post-training dependencies on your VM, please refer to the [official documentation](../../install_maxtext) and use the `maxtext[tpu-post-train]` installation path to include all necessary post-training dependencies.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

## Setup environment variables

Login to Hugging Face. Provide your access token when prompted:
You can generate one at https://huggingface.co/settings/tokens.

```bash
hf auth login
```

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g. 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser) or via
# `gcloud storage buckets list --format="table(name, location)"`.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

# Number of accelerator chips per VM.
# - TPU v5e (single host): 8
# - TPU v5p (single host): 4
# - TPU v6e (single host): 8
export CHIPS_PER_VM=<CHIPS_PER_VM>
```

## Get your model checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](hf-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Run GRPO

Run the following command for GRPO:

```
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  chips_per_vm=${CHIPS_PER_VM?}
```

The overview of what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of the model
   checkpoint you specified (e.g., `Llama3.1-8b-Instruct`).
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark
   after the post-training with GRPO.

## Run GSPO

Run the following command for GSPO:

```
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  loss_algo=gspo-token \
  chips_per_vm=${CHIPS_PER_VM?}
```

The overview of what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of the model
   checkpoint you specified (e.g., `Llama3.1-8b-Instruct`).
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GSPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark
   after the post-training with GSPO.
